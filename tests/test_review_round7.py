"""Regression tests for the seventh review round (h5ad loader, provenance, batch mode,
navigation, size guards). One test per finding; the name says which behaviour is pinned."""
from __future__ import annotations

import json
import os
import pathlib
import re

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import xe_testlib as tl
from test_controls import _fab_ds
from test_review_round6 import _fake_run


def _quiet(_line):
    pass


@pytest.fixture(autouse=True)
def _clean(xe):
    before = list(xe._prov_recent)
    yield
    xe._prov_recent[:] = before
    xe._stats_reset()
    xe._controls_reset()
    for f in os.listdir(xe.CACHE_BASE_DIR):
        if f.startswith("design_") and f.endswith(".json"):
            os.remove(os.path.join(xe.CACHE_BASE_DIR, f))


@pytest.fixture()
def cfg(xe, bundle_dir, tmp_path):
    return {"data": {"path": str(bundle_dir)}, "output_dir": str(tmp_path / "out"),
            "steps": [{"kind": "moran", "params": {"k": 6}}]}


@pytest.fixture()
def two_runs(xe, D, monkeypatch):
    """Two cached Proseg runs of the dataset, 'proseg:aaaa' (40 cells) and 'proseg:bbbb'
    (30 cells, clusters 77 / 78), loaded by a stand-in of the cached-run loader."""
    from xenium_app import caching, cb_reseg
    runs = {"proseg:aaaa": dict(n=40, out_dir="/fake/run_aaaa", seed=1),
            "proseg:bbbb": dict(n=30, out_dir="/fake/run_bbbb", seed=2)}
    loads = []

    def load(seg_value, _b=None, _p=None):
        loads.append(seg_value)
        res = _fake_run(xe, D, **runs[seg_value])
        if seg_value.endswith("bbbb"):
            res["cells_df"]["cluster_kmeans_10"] = ["77" if i % 2 else "78" for i in range(len(res["cells_df"]))]
        return None

    monkeypatch.setattr(caching, "_list_cached_seg_runs",
                        lambda: [{"label": k, "value": k, "out_dir": v["out_dir"]} for k, v in runs.items()])
    monkeypatch.setattr(cb_reseg, "load_seg_run_from_cache", load)
    return loads


# ── R7-20 (high): a step runs on the run it names ────────────────────────────────────
def test_batch_step_runs_on_the_segmentation_run_it_names(xe, D, cfg, two_runs):
    cfg["seg_source"] = "proseg:aaaa"
    cfg["figures"] = False
    cfg["steps"] = [{"kind": "composition", "name": "top", "params": {"labels": "cluster"}},
                    {"kind": "composition", "name": "other", "params": {"labels": "cluster", "seg_source": "proseg:bbbb"}},
                    {"kind": "composition", "name": "again", "params": {"labels": "cluster"}}]
    out = xe._batch_run(cfg, log=_quiet)
    assert out["ok"], out
    root = pathlib.Path(cfg["output_dir"])
    n = [int(pd.read_parquet(root / d / "counts.parquet").select_dtypes("number").to_numpy().sum())
         for d in ("01_top", "02_other", "03_again")]
    assert n == [40, 30, 40], n                       # each step: the cells of the run it names
    assert out["steps"][1]["resolved"]["seg-source"] == "proseg:bbbb"
    assert two_runs == ["proseg:aaaa", "proseg:bbbb", "proseg:aaaa"] or two_runs.count("proseg:bbbb") >= 1
    # a run that does not exist is an error of the config, and the step does not fall back
    # to the Xenium cells
    bad = {**cfg, "seg_source": "xenium", "output_dir": cfg["output_dir"] + "2",
           "steps": [{"kind": "composition", "params": {"labels": "cluster", "seg_source": "proseg:zzzz"}}]}
    dry = xe._batch_run(bad, dry_run=True, log=_quiet)
    assert not dry["ok"] and "no such cached run" in "\n".join(dry["errors"])
    real = xe._batch_run(bad, log=_quiet)
    assert not real["ok"] and real["steps"][0]["status"] == "failed"
    assert "no such cached run" in real["steps"][0]["message"]


def test_annotation_and_niches_refuse_a_run_that_is_not_loaded(xe):
    out = xe.run_niche(1, 5, 10, None, "baysor:zz")
    assert "not loaded" in out[2] and xe._niche_state["status"] != "starting"
    out = xe.start_annotation(1, "celltypist", "Model.pkl", None, None, None, None, None, None, "proseg:zz")
    assert "not loaded" in out[1] and xe._annot_state["status"] != "running"


# ── R7-23: a dry run checks against the cells of the named run ───────────────────────
def test_batch_dry_run_validates_against_the_named_run(xe, cfg, two_runs):
    cfg["seg_source"] = "proseg:bbbb"
    cfg["steps"] = [{"kind": "aggregates", "params": {"labels": "cluster", "classes": ["77"]}}]
    out = xe._batch_run(cfg, dry_run=True, log=_quiet)
    assert out["ok"], out["errors"]                     # cluster 77 exists in the run only
    assert two_runs == ["proseg:bbbb"]
    cfg["steps"] = [{"kind": "aggregates", "params": {"labels": "cluster", "classes": ["no such"]}}]
    out = xe._batch_run(cfg, dry_run=True, log=_quiet)
    assert not out["ok"] and "labels: 77, 78" in "\n".join(out["errors"])


# ── R7-21 / R7-25: per-cell outputs of the annotation and gene-set steps ─────────────
def test_batch_annotation_step_writes_the_labels_the_job_stored(xe, D, labels, tmp_path):
    from xenium_app import batch
    step_dir = tmp_path / "ann"
    step_dir.mkdir()
    ctx = {"seg-source": "xenium", "counts-mode-store": "original", "cluster-method": None, "color-by": "cluster"}
    files = batch._batch_write_outputs({"kind": "annotation"}, {}, {"annot-source": "celltypist"}, ctx, None,
                                       str(step_dir), {"tables": True, "figures": False}, _quiet)
    assert any(f.endswith("cells.parquet") for f in files), files
    cells = pd.read_parquet(step_dir / "cells.parquet")
    assert list(cells.columns) == ["cell_id", "cell_type_celltypist"] and len(cells) == len(labels)


def test_batch_geneset_step_writes_cells_under_the_stored_name_and_counts(xe, D, cfg):
    view = xe._an_samples("xenium", False)
    genes = xe._an_common_genes(view)
    means = xe._an_gene_means(view, genes)
    expressed = [genes[i] for i in np.argsort(means)[::-1] if means[i] > 0]
    cfg["figures"] = False
    cfg["steps"] = [{"kind": "geneset", "params": {"name": "IFN  response", "text": " ".join(expressed[:12])}}]
    out = xe._batch_run(cfg, log=_quiet)
    assert out["ok"], out
    cells = pd.read_parquet(pathlib.Path(cfg["output_dir"]) / "01_geneset" / "cells.parquet")
    assert "geneset_IFN response" in cells.columns and len(cells) == len(D["df"])


# ── R7-22 / R7-27: the design of the config ──────────────────────────────────────────
def test_batch_design_replaces_the_saved_design(xe, D, cfg, tmp_path):
    from xenium_app import batch
    xe.EXTRA_DATASETS.extend([_fab_ds(D, 30, i, str(tmp_path / f"b{i}")) for i in range(3)])
    cfg["data"]["design"] = {"names": {"A": "post", "B": "pre"}, "samples": [
        {"sample": "S1", "condition": "post", "donor": "d1", "batch": "b1"},
        {"sample": "S2", "condition": "pre", "donor": "d1"},
        {"sample": "S3", "condition": "pre", "donor": "d2"}]}
    batch._batch_setup_samples(batch._batch_norm_config(cfg), _quiet)
    assert [(r["cond"], r["donor"]) for r in xe._design_samples()[1:]] == [("A", "d1"), ("B", "d1"), ("B", "d2")]
    # the next run: no donors, S3 not listed
    cfg["data"]["design"] = {"samples": [{"sample": "S1", "condition": "A"}, {"sample": "S2", "condition": "B"}]}
    batch._batch_setup_samples(batch._batch_norm_config(cfg), _quiet)
    got = xe._design_samples()[1:]
    assert [(r["cond"], r["donor"], r["batch"]) for r in got] == [("A", "", ""), ("B", "", ""), ("", "", "")]
    assert xe._design_names() == {"A": "A", "B": "B"}


def test_batch_design_accepts_conditions_written_as_zero_or_false(xe, D, cfg, tmp_path):
    from xenium_app import batch
    xe.EXTRA_DATASETS.extend([_fab_ds(D, 30, i, str(tmp_path / f"z{i}")) for i in range(2)])
    cfg["data"]["design"] = {"names": {"A": 1, "B": 0}, "samples": [
        {"sample": "S1", "condition": 1}, {"sample": "S2", "condition": 0}]}
    batch._batch_setup_samples(batch._batch_norm_config(cfg), _quiet)
    assert [r["cond"] for r in xe._design_samples()[1:]] == ["A", "B"]
    assert xe._design_names() == {"A": "1", "B": "0"}
    cfg["data"]["design"] = {"names": {"A": True, "B": False}, "samples": [
        {"sample": "S1", "condition": False}, {"sample": "S2", "condition": True}]}
    batch._batch_setup_samples(batch._batch_norm_config(cfg), _quiet)
    assert [r["cond"] for r in xe._design_samples()[1:]] == ["B", "A"]
    cfg["data"]["design"]["samples"][0]["condition"] = 7
    with pytest.raises(xe._BatchError, match="neither A / B"):
        batch._batch_setup_samples(batch._batch_norm_config(cfg), _quiet)


# ── R7-24: a synchronous step that computed nothing is not reported with an earlier result
def test_batch_sync_step_without_a_result_of_its_own_fails(xe, cfg):
    cfg["figures"] = False
    cfg["continue_on_error"] = True
    cfg["steps"] = [{"kind": "composition", "params": {"labels": "cluster"}},
                    {"kind": "composition", "name": "by_type", "params": {"transform": "asin"}}]   # labels forgotten
    lines = []
    out = xe._batch_run(cfg, log=lines.append)
    assert [s["status"] for s in out["steps"]] == ["done", "failed"] and out["ok"] is False
    assert "Pick what the proportions are computed of" in out["steps"][1]["message"]
    assert not (pathlib.Path(cfg["output_dir"]) / "02_by_type" / "counts.parquet").exists()
    assert any(l.strip().startswith("error: step 2 (by_type): parameter ") and "is required" in l for l in lines)


# ── R7-9 / R7-26: an inline h5ad mapping gets a file of its own ──────────────────────
def test_inline_h5ad_mapping_file_is_named_by_its_content(xe, tmp_path):
    import xenium_app
    a = xenium_app._batch_inline_mapping_file({"counts": "counts", "section": "sec"}, str(tmp_path))
    b = xenium_app._batch_inline_mapping_file({"counts": "X", "section": "sec"}, str(tmp_path))
    a2 = xenium_app._batch_inline_mapping_file({"section": "sec", "counts": "counts"}, str(tmp_path))
    assert a != b and a == a2 and os.path.basename(a) != "batch_h5ad_mapping.json"
    assert json.load(open(a)) == {"counts": "counts", "section": "sec"} and json.load(open(b))["counts"] == "X"
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


# ══ h5ad loader ══════════════════════════════════════════════════════════════════════
def _man(tmp_path, conds, donors=None, **mapping):
    secs = [{"name": f"sec{i}", "store": f"sec{i}.zarr", "n_cells": 30, "condition": c,
             "donor": (donors[i] if donors else f"d{i}")} for i, c in enumerate(conds)]
    return {"source": str(tmp_path / "study.h5ad"), "sections": secs, "n_cells": 30 * len(secs), "n_genes": 10,
            "counts": {"matrix": "X", "integer": True, "nnz": 10, "n_noninteger": 0, "n_negative": 0, "max": 3.0},
            "mapping": {"cond_a": None, "cond_b": None, "noninteger_counts": "refuse", **mapping}}


def _h5ad_samples(xe, D, tmp_path, man):
    """Fabricated samples that are known by the stores of *man* (as sections of an .h5ad are)."""
    out = []
    for i, s in enumerate(man["sections"]):
        ds = _fab_ds(D, 30, i, "unused")
        ds.pop("data_dir")
        ds["sdata_path"] = str(tmp_path / s["store"])
        out.append(ds)
    xe.EXTRA_DATASETS.extend(out)
    return out


# ── R7-1 / R7-5 / R7-6: the design prefilled from the file ───────────────────────────
def test_h5ad_design_follows_a_newly_given_pair_and_keeps_donor_edits(xe, D, tmp_path, monkeypatch):
    from xenium_app import h5ad
    man = _man(tmp_path, ["ctrl", "treated", "other"])
    _h5ad_samples(xe, D, tmp_path, man)
    msg = h5ad._h5ad_design(force=False, man=man, cache_dir=str(tmp_path))
    assert "no sample was assigned" in msg
    assert [r["cond"] for r in xe._design_samples()[1:]] == ["", "", ""]
    # the user edits a donor, then the pair is given (environment, dialog, or the mapping)
    rows = xe._design_samples()[1:]
    xe._design_set([{"key": rows[0]["key"], "cond": "", "batch": "b9", "donor": "edited"}])
    man2 = _man(tmp_path, ["ctrl", "treated", "other"], cond_a="ctrl", cond_b="treated")
    msg = h5ad._h5ad_design(force=False, man=man2, cache_dir=str(tmp_path))
    got = xe._design_samples()[1:]
    assert [r["cond"] for r in got] == ["A", "B", ""], msg
    assert got[0]["donor"] == "edited" and got[0]["batch"] == "b9" and got[1]["donor"] == "d1"
    assert xe._design_names() == {"A": "ctrl", "B": "treated"}
    # the same pair again: the saved design is kept (edits of the conditions included)
    xe._design_set([{"key": got[2]["key"], "cond": "A", "batch": "", "donor": "d2"}])
    assert "was kept" in h5ad._h5ad_design(force=False, man=man2, cache_dir=str(tmp_path))
    assert xe._design_samples()[3]["cond"] == "A"
    # a pair that is not two levels: the conditions stay unassigned, the donors are filled
    xe._stats_reset()
    for f in os.listdir(xe.CACHE_BASE_DIR):
        if f.startswith("design_"):
            os.remove(os.path.join(xe.CACHE_BASE_DIR, f))
    man3 = _man(tmp_path, ["0", "1", "1"], donors=["p", "q", "r"], cond_a="0.0", cond_b="1.0")
    msg = h5ad._h5ad_design(force=False, man=man3, cache_dir=str(tmp_path))
    assert "Conditions not assigned" in msg and [r["donor"] for r in xe._design_samples()[1:]] == ["p", "q", "r"]


def test_h5ad_design_keys_match_the_samples_with_a_relative_cache_dir(xe, D, tmp_path, monkeypatch):
    from xenium_app import h5ad
    man = _man(tmp_path, ["ctrl", "treated"])
    _h5ad_samples(xe, D, tmp_path, man)
    monkeypatch.chdir(tmp_path.parent)
    h5ad._h5ad_design(force=True, man=man, cache_dir=tmp_path.name)        # relative, as XENIUM_CACHE_DIR=./cache
    assert [r["cond"] for r in xe._design_samples()[1:]] == ["A", "B"]


def test_h5ad_inspect_lists_levels_as_the_conversion_writes_them(xe, tmp_path):
    ad = pytest.importorskip("anndata")
    from xenium_app import h5ad
    n = 12
    obs = pd.DataFrame({"x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float),
                        "treated": np.array([0.0, 1.0] * (n // 2)),
                        "grp": pd.Categorical(np.array([0.0, 1.0] * (n // 2)))},
                       index=[f"c{i}" for i in range(n)])
    a = ad.AnnData(X=sp.csr_matrix(np.ones((n, 3), dtype=np.float32)), obs=obs,
                   var=pd.DataFrame(index=["g0", "g1", "g2"]))
    a.obsm["spatial_um"] = np.column_stack([np.arange(n) * 10.0, np.arange(n) * 5.0])
    path = tmp_path / "lv.h5ad"
    a.write_h5ad(path)
    cols = {c["name"]: c for c in h5ad._h5ad_inspect(str(path))["obs_columns"]}
    assert cols["treated"]["levels"] == ["0", "1"] and cols["grp"]["levels"] == ["0", "1"]
    # R7-2: a sidecar that names only the obsm key is not overridden by guessed x / y columns
    (tmp_path / "lv.h5ad.json").write_text(json.dumps({"spatial": "spatial_um"}))
    h5ad._h5ad_inspect_cache.clear() if hasattr(h5ad, "_h5ad_inspect_cache") else None
    mp = h5ad._h5ad_mapping_for(str(path))
    assert mp["spatial"] == "spatial_um" and mp["x"] is None and mp["y"] is None
    (tmp_path / "lv.h5ad.json").write_text(json.dumps({"x": "x", "y": "y"}))
    mp = h5ad._h5ad_mapping_for(str(path))
    assert mp["x"] == "x" and mp["spatial"] is None


# ── R7-8: repeated obs names stay unique after numbering ─────────────────────────────
def test_h5ad_repeated_obs_names_are_numbered_without_collisions(xe):
    from xenium_app import h5ad
    got = h5ad._h5ad_unique_names(["a", "a-1", "a", "b", "a", "b"])
    assert len(set(got)) == len(got) and got[0] == "a" and got[1] == "a-1" and got[3] == "b"
    assert h5ad._h5ad_unique_names(["x", "y"]) == ["x", "y"]
    import inspect
    assert "_h5ad_unique_names(cid)" in inspect.getsource(h5ad._h5ad_convert)


# ── R7-7: raw counts are whole and not negative; the conversion's verdict wins ───────
def test_counts_with_negative_values_or_a_noninteger_manifest_are_not_raw(xe):
    whole = sp.csc_matrix(np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float32))
    assert xe._counts_are_integer({"csc": whole, "ds": {}})
    neg = sp.csc_matrix(np.array([[1.0, -2.0], [0.0, 3.0]], dtype=np.float32))
    assert not xe._counts_are_integer({"csc": neg, "ds": {}})
    assert not xe._counts_are_integer({"csc": sp.csc_matrix(np.array([[1, -2], [0, 3]], dtype=np.int32)), "ds": {}})
    # float32 lost the fraction the conversion saw at full precision
    assert not xe._counts_are_integer({"csc": whole, "ds": {"counts_noninteger": "refuse"}})
    s = {"csc": whole, "ds": {"counts_noninteger": "refuse", "h5ad": True}, "rows": np.arange(2),
         "label": "S1", "df": pd.DataFrame(index=["a", "b"])}
    with pytest.raises(ValueError, match="needs raw counts"):
        xe._pb_sum_counts(s, np.ones(2, bool))


# ── R7-3 / R7-10 / R7-4: the conversion record, and Cancel ───────────────────────────
def test_h5ad_record_is_written_after_the_install_not_by_the_job(xe, D, tmp_path, monkeypatch):
    from xenium_app import h5ad, cb_data, provenance
    man = dict(_man(tmp_path, ["ctrl", "treated"], noninteger_counts="round_sums"), from_cache=True)
    man["counts"].update({"integer": False, "n_noninteger": 7})
    monkeypatch.setattr(h5ad, "_h5ad_load", lambda path, mapping, progress=None: ([{}], man, str(tmp_path)))
    n0 = len(provenance._prov_recent)
    try:
        cb_data._h5ad_job(str(tmp_path / "study.h5ad"), {"counts": "X"})
        assert h5ad._h5ad_state["status"] == "done"
        assert len(provenance._prov_recent) == n0, "the job wrote the record while the old dataset was loaded"
        h5ad._h5ad_record(man)
        rec = provenance._prov_recent[-1]
        assert len(provenance._prov_recent) == n0 + 1 and rec["kind"] == "h5ad" and rec["cache"] == "cache"
        assert rec["data"]["dataset_key"] == xe._dataset_key()
        assert rec["headline"]["noninteger_counts"] == "round_sums" and rec["headline"]["counts_integer"] is False
        md = provenance._prov_methods_md([rec])
        assert "noninteger_counts = round_sums" in md and "NOT integer-valued" in md and "not from raw counts" in md
        h5ad._h5ad_record(man)                               # the same file and mapping again this session
        assert len(provenance._prov_recent) == n0 + 1
        import inspect
        assert "h5ad._h5ad_record(man)" in inspect.getsource(cb_data._h5ad_install)
    finally:
        provenance._prov_once.clear()
        with h5ad._h5ad_lock:
            h5ad._h5ad_state.update({"status": "idle", "message": "", "result": None})


def test_h5ad_load_cancelled_during_the_last_section_installs_nothing(xe, tmp_path, monkeypatch):
    from xenium_app import h5ad, cb_data, jobs

    def load(path, mapping, progress=None):
        jobs._job_current().cancel.set()                      # Cancel while the last store is read
        return ([{}], {}, str(tmp_path))

    monkeypatch.setattr(h5ad, "_h5ad_load", load)
    with h5ad._h5ad_lock:
        h5ad._h5ad_state.update({"status": "running", "message": "Starting…", "result": None})
    job = jobs._job_start("h5ad", "Load .h5ad", cb_data._h5ad_job, args=("x.h5ad", {}),
                          state=h5ad._h5ad_state, lock=h5ad._h5ad_lock)
    job.thread.join(30)
    try:
        assert job.state == "cancelled", job.state
        assert h5ad._h5ad_state["status"] != "done" and h5ad._h5ad_state.get("result") is None
    finally:
        with h5ad._h5ad_lock:
            h5ad._h5ad_state.update({"status": "idle", "message": "", "result": None})


# ══ provenance ═══════════════════════════════════════════════════════════════════════
def _rec(kind, **kw):
    base = {"id": f"prov-{kind}-{kw.pop('n', 1)}", "schema": 1, "kind": kind, "title": kw.pop("title", kind),
            "time": kw.pop("time", "2026-01-01T00:00:00+00:00"), "time_local": "2026-01-01T00:00:00",
            "data": {"seg_source": kw.pop("seg", "xenium"), "seg_tool": "xenium", "samples": [], "n_cells": 0,
                     "n_genes": 0, "counts": kw.pop("counts", {}), "labels": None, "subset": None},
            "params": kw.pop("params", {}), "unit": {}, "design": None, "headline": kw.pop("headline", {}),
            "warnings": [], "seeds": {}, "cache": kw.pop("cache", "computed"), "runtime_s": 1.0,
            "software": {"python": "3", "tools": kw.pop("tools", {})}, "status_line": kw.pop("status", ""),
            "outputs": []}
    base.update(kw)
    return base


# ── R7-11 / R7-12: the record states the counts that were used ───────────────────────
def test_record_states_the_counts_used_not_the_counts_requested(xe, D, tmp_path):
    from xenium_app import provenance, txfactor
    import inspect
    # one sample with corrected counts: used = corrected
    D["split_corrected_expr"] = sp.csr_matrix(D["expr"]).astype(np.float64)
    assert provenance._prov_counts_used("xenium", True, None) is True
    assert provenance._prov_data("xenium", True, None)["counts"]["used"] == "SPLIT-corrected"
    # a second sample without them: every analysis falls back to the original counts
    xe.EXTRA_DATASETS.append(_fab_ds(D, 40, 0, str(tmp_path / "gs0")))
    assert provenance._prov_counts_used("xenium", True, None) is False
    c = provenance._prov_data("xenium", True, None)["counts"]
    assert c == {"requested": "corrected", "used": "original", "n_cells_left_out": None}
    # what the job reports wins; a job that read no count matrix says nothing about counts
    assert provenance._prov_data("xenium", True, {"corrected": True})["counts"]["used"] == "SPLIT-corrected"
    assert provenance._prov_data("xenium", True, {"corrected": None})["counts"]["used"] is None
    assert provenance._prov_data("xenium", False, None)["counts"]["used"] == "original"
    # the gene-set job: its record and its status line
    view = xe._an_samples("xenium", False)
    genes = xe._an_common_genes(view)
    means = xe._an_gene_means(view, genes)
    expressed = [genes[i] for i in np.argsort(means)[::-1] if means[i] > 0][:12]
    with xe._geneset_lock:
        xe._geneset_state["sets"]["r7 set"] = expressed
    try:
        n0 = len(provenance._prov_recent)
        xe._run_geneset_score("r7 set", "xenium", True)
        assert xe._geneset_state["status"] == "done", xe._geneset_state["message"]
        rec = provenance._prov_recent[-1]
        assert len(provenance._prov_recent) == n0 + 1 and rec["kind"] == "geneset"
        assert rec["data"]["counts"]["requested"] == "corrected" and rec["data"]["counts"]["used"] == "original"
        assert "Original counts were used" in rec["status_line"]
        assert "SPLIT-corrected counts were requested but original counts were used" in provenance._prov_methods_md([rec])
    finally:
        with xe._geneset_lock:
            xe._geneset_state["sets"].pop("r7 set", None)
    src = inspect.getsource(txfactor._run_txfactor)
    assert '"corrected": _corr_used' in src and '"corrected": bool(use_corrected)' not in src


# ── R7-13: the job's own cache flag wins over words in its status line ───────────────
def test_cache_flag_of_the_result_wins_over_the_status_line(xe):
    from xenium_app import provenance
    line = "12 factors … (41.0 s: binning 0 s (cached), fit 35 s, projection 6 s)."
    assert provenance._prov_from_cache(None, {"from_cache": False}, line) is False
    assert provenance._prov_from_cache(None, {"from_cache": True}, line) is True
    assert provenance._prov_from_cache(None, {"cached": False}, "Done (cached) — 3 genes") is False
    assert provenance._prov_from_cache(None, {}, "Done (cached) — 3 genes") is True
    assert provenance._prov_from_cache(False, {"from_cache": True}, line) is False


# ── R7-14: one UMAP record per run, with the counts and the cells embedded ───────────
def test_rerun_umap_writes_one_record_with_the_counts_used(xe, D, monkeypatch):
    import sys
    from xenium_app import core, provenance, clustering
    res = _fake_run(xe, D, n=60)
    res["split_corrected_cell_ids"] = list(res["cells_df"].index[:50])          # SPLIT kept 50 of 60
    res["split_corrected_expr"] = res["expr"][:50]
    monkeypatch.setitem(sys.modules, "umap.umap_", None)                        # t-SNE: fast on 60 cells
    with core._umap_reseg_lock:
        core._umap_reseg_state.update({"status": "running", "seg_source": "proseg:t", "counts_mode": "corrected"})
    n0 = len(provenance._prov_recent)
    clustering._run_reseg_umap()
    assert core._umap_reseg_state["status"] == "done", core._umap_reseg_state["message"]
    new = [r for r in provenance._prov_recent[n0:] if r["kind"] == "umap"]
    assert len(new) == 1, f"{len(new)} records for one run"
    r = new[0]
    if r["data"]["counts"]["used"] == "SPLIT-corrected":
        assert r["headline"]["n_cells"] == 50 and r["data"]["counts"]["n_cells_left_out"] == 10
        assert "SPLIT-corrected" in r["status_line"]
    assert r["data"]["counts"]["requested"] == "corrected" and r["data"]["counts"]["used"] is not None
    assert r["params"]["method"] == "t-SNE" and r["params"]["n_components"] >= 2
    md = provenance._prov_methods_md([r])
    assert "t-SNE (umap-learn was not available)" in md and "counts were used" in md


# ── R7-15: the recorded subset is the active one ─────────────────────────────────────
def test_subset_record_is_the_last_call_and_only_applied_filters(xe, D):
    from xenium_app import core, provenance
    tc = D["df"]["transcript_counts"]
    lo, hi = float(tc.quantile(0.25)), float(tc.quantile(0.75))
    xe.subset(min_transcripts=lo)
    xe.subset(max_transcripts=hi)                       # starts from all cells again
    assert len(D["df"]) == int((tc <= hi).sum())
    assert D["_subset_desc"] == [{"max_transcripts": hi}]
    assert provenance._prov_data()["subset"]["filters"] == [{"max_transcripts": hi}]
    with core._annot_lock:
        saved = {k: core._annot_state[k] for k in list(core._annot_state) if str(k).startswith("labels_")}
        for k in saved:
            core._annot_state[k] = None
    try:
        xe.subset(min_transcripts=lo, cell_type="Nothing")          # no annotation: that filter is skipped
        assert D["_subset_desc"] == [{"min_transcripts": lo, "skipped": ["cell_type"]}]
    finally:
        with core._annot_lock:
            core._annot_state.update(saved)


# ── R7-16: niches without labels are recorded and described as what they are ─────────
def test_niches_without_labels_are_described_as_spatial_blocks(xe, D):
    from xenium_app import core, provenance, spatial
    n0 = len(provenance._prov_recent)
    spatial._run_spatial_niches(3, 5, None, None)
    assert core._niche_state["status"] == "done", core._niche_state["message"]
    assert "spatial blocks, not composition niches" in core._niche_state["message"]
    rec = provenance._prov_recent[-1]
    assert len(provenance._prov_recent) == n0 + 1 and rec["kind"] == "niches"
    assert rec["params"]["features"] == spatial._NICHE_FEATURES_COORDINATES
    md = provenance._prov_methods_md([rec])
    assert "normalised cell centroid coordinates were clustered" in md
    assert "cell-type composition of its" not in md
    labelled = _rec("niches", params={"n_clusters": 5, "n_neighbors": 20,
                                      "features": spatial._NICHE_FEATURES_COMPOSITION})
    assert "cell-type composition of its 20 nearest neighbours" in provenance._prov_methods_md([labelled])


# ── R7-17 / R7-19: proximity and contact templates say what the run did ──────────────
def test_proximity_methods_text_follows_the_recorded_estimator_fdr_and_covariate(xe):
    from xenium_app import provenance
    P = {"target": "T", "anchor": "X", "source": "none", "near": 25, "far": 100}
    slope = _rec("proximity", params=P, headline={"estimator": "slope", "fdr": "bh", "mode": "cohort"})
    md = provenance._prov_methods_md([slope])
    assert "The statistic the p-values and the FDR are of is the ordinary-least-squares slope" in md
    assert "Benjamini-Hochberg over the sign-flip p-values" in md and "Benjamini Y, Hochberg Y" in md
    assert "no density-covariate cell type was used" in md and "decile bins" not in md
    assert "the empirical FDR of a gene is" not in md
    nf = _rec("proximity", params={**P, "source": "Epi"},
              headline={"estimator": "nearfar", "fdr": "empirical", "mode": "single"})
    md = provenance._prov_methods_md([nf])
    assert "The effect of a gene is the mean residual" in md and "local fraction of 'Epi' cells" in md
    assert "the empirical FDR of a gene is" in md and "Benjamini Y" not in md
    assert "share of random anchor sets at least as extreme" in md
    # the job puts these facts where the template reads them
    import inspect
    from xenium_app import proximity
    assert '"estimator": info.get("estimator"), "fdr": info.get("fdr")' in inspect.getsource(proximity._run_proximity_de)


def test_contact_methods_text_names_the_donor_as_the_unit_when_pooled(xe):
    from xenium_app import provenance
    P = {"target": "T", "anchor": "X"}
    md = provenance._prov_methods_md([_rec("contact", params=P, headline={"by_donor": True})])
    assert "summed per donor and group" in md and "~ donor + group" in md
    assert "the unit of the p-values is the donor" in md
    md = provenance._prov_methods_md([_rec("contact", params=P, headline={"by_donor": False})])
    assert "summed per sample and group" in md and "the unit of the p-values is the sample" in md


# ── R7-18 + follow-up: a reload from the cache does not replace the computed record ──
def test_cache_reload_does_not_displace_the_record_of_the_run(xe):
    from xenium_app import provenance
    run = _rec("baysor", n=1, seg="baysor:abcd", title="Baysor segmentation",
               params={"scale": 6.5, "min_mol": 30}, tools={"baysor": "0.7.1"})
    load = _rec("baysor", n=2, seg="baysor:abcd", title="Baysor segmentation (cached run loaded)",
                params={"run": "baysor_x_abcd"}, cache="cache", time="2026-02-01T00:00:00+00:00",
                time_local="2026-02-01T00:00:00")
    md = provenance._prov_methods_md([run, load])
    assert "## Baysor segmentation\n" in md and "## Baysor segmentation (cached run loaded)" not in md
    assert "baysor: 0.7.1" in md and "read back from the cache later (newest: record prov-baysor-2, 2026-02-01)" in md
    # an annotation served from the cache with the same parameters: the computed record stays
    a1 = _rec("annotation", n=1, params={"method": "rctd", "mode": "full"})
    a2 = _rec("annotation", n=2, params={"method": "rctd", "mode": "full"}, cache="cache")
    md = provenance._prov_methods_md([a1, a2])
    assert "Record prov-annotation-1 (computed" in md and "Record prov-annotation-2" not in md.split("## All records")[0]
    # other parameters from the cache: that is another result, and it is the newest
    a3 = _rec("annotation", n=3, params={"method": "rctd", "mode": "doublet"}, cache="cache")
    md = provenance._prov_methods_md([a1, a3])
    assert "Record prov-annotation-3 (loaded from the cache" in md.split("## All records")[0]
    # only cache records: the newest one
    assert "Record prov-baysor-2" in provenance._prov_methods_md([load]).split("## All records")[0]


def test_loading_a_cached_run_is_recorded_once_per_session(xe):
    from xenium_app import provenance
    import inspect
    from xenium_app import reseg, imgseg
    n0 = len(provenance._prov_recent)
    try:
        for _ in range(3):                     # the run is picked in the dropdown three times
            provenance._prov_record("baysor", "Baysor segmentation (cached run loaded)", seg_source="baysor",
                                    params={"run": "r"}, cache=True, message="Loaded — 5 cells",
                                    once=("load", "/cache/baysor_r", "rid1"))
        assert len(provenance._prov_recent) == n0 + 1
        provenance._prov_record("baysor", "Baysor segmentation (cached run loaded)", seg_source="baysor",
                                params={"run": "r"}, cache=True, message="Loaded — 5 cells",
                                once=("load", "/cache/baysor_r", "rid2"))      # a re-run: another run
        assert len(provenance._prov_recent) == n0 + 2
    finally:
        provenance._prov_once.clear()
    assert inspect.getsource(reseg).count('once=("load", str(out_dir), core._seg_run_id(result))') == 2
    assert 'once=("load", str(out_dir)' in inspect.getsource(imgseg)


# ══ navigation ═══════════════════════════════════════════════════════════════════════
def _inputs_of(xe, output_id):
    cbs = [v for k, v in xe.app.callback_map.items() if output_id in k]
    assert cbs, output_id
    return {i["id"] for cb in cbs for i in cb["inputs"]}


def test_nav_notes_follow_what_the_tabs_do(xe, D, tmp_path, monkeypatch):
    from xenium_app import cb_nav, core
    # R7-28: the CellNEST note is re-evaluated when results are loaded (and R7-29: on a
    # change of segmentation)
    assert {"cellnest-version", "seg-source"} <= _inputs_of(xe, "nav-notice.children")
    # R7-29: the niche note is keyed on what the figure draws
    with core._niche_lock:
        core._niche_state["status"] = "idle"
    assert "No niches have been computed" in cb_nav._nav_notice("tab-niche-umap", "xenium")
    D["df"]["niche"] = "0"                                    # restored from a store: no job ran
    assert cb_nav._nav_notice("tab-niche-umap", "xenium") == ""
    with core._niche_lock:
        core._niche_state["status"] = "done"                  # … computed on another segmentation
    D["df"].drop(columns=["niche"], inplace=True)
    assert "No niches have been computed" in cb_nav._nav_notice("tab-niche-umap", "xenium")
    # R7-30: transcripts of any sample count
    monkeypatch.setitem(xe.DATA, "data_dir", None)
    assert not cb_nav._nav_has_transcripts() and "no transcripts to read" in cb_nav._nav_notice("tab-density")
    extra = _fab_ds(D, 20, 0, str(tmp_path / "with_tx"))
    os.makedirs(extra["data_dir"])
    open(os.path.join(extra["data_dir"], "transcripts.parquet"), "w").close()
    xe.EXTRA_DATASETS.append(extra)
    assert cb_nav._nav_has_transcripts()
    note = cb_nav._nav_notice("tab-segfree")
    assert "no transcripts to read" not in note and "S0 was loaded without one" in note and "runs on S1" in note


def test_type_pickers_follow_the_segmentation_cluster_set_and_samples(xe, D):
    import dash
    for out in ("sfree-type.options", "agg-classes.options", "ctl-dominant.options", "coloc-placebo.options",
                "prox-target.options"):
        got = _inputs_of(xe, out)
        assert {"seg-source", "cluster-method", "extra-datasets-version", "counts-mode-store"} <= got, (out, got)
    method = (D.get("cluster_methods") or [None])[0]
    opts, value = xe.update_coloc_placebo_options("cluster", 0, 0, "xenium", method, 0, 0, "original", 0, 0, "gone")
    assert opts and value is None                              # a pick that no longer exists is cleared
    keep = opts[0]["value"]
    assert xe.update_coloc_placebo_options("cluster", 0, 0, "xenium", method, 0, 0, "original", 0, 0, keep)[1] \
        is dash.no_update
    o, v = xe.update_agg_class_options("cluster", 0, 0, "xenium", method, 0, 0, "original", 0, 0, [keep, "gone"])
    assert v == [keep]
    dom, src, dv, sv = xe.update_control_type_options("cluster", 0, 0, "xenium", method, 0, 0, "original", 0, 0,
                                                      "gone", "auto")
    assert dv == "auto" and sv is dash.no_update
    assert xe.update_sfree_type_options("cluster", 0, 0, 0, "xenium", method, 0, "original", 0, 0, "gone")[1] is None


# ══ size guards ══════════════════════════════════════════════════════════════════════
def test_size_guard_counts_the_gram_matrix_the_container_limit_and_the_fallback(xe, tmp_path, monkeypatch):
    from xenium_app import banksy, clustersets
    import inspect
    # R7-32: the Gram matrix of the augmented features does not depend on the cell count
    assert banksy._bk_gram_gb(20000, True) > 60 and banksy._bk_gram_gb(500, True) < 0.1
    assert banksy._bk_gram_gb(20000, False) < banksy._bk_gram_gb(20000, True)
    monkeypatch.setattr(clustersets, "_cs_ram_gb", lambda: 64.0)
    monkeypatch.delenv("XENIUM_ALLOW_LARGE_JOBS", raising=False)
    stop, _est = clustersets._cs_size_guard("Banksy-style clustering", 10_000, 50,
                                            matrix_gb=banksy._bk_gram_gb(20000, True), graph=True)
    assert stop and "was not started" in stop and "lower the gene cap" in stop
    assert "_bk_gram_gb(min(int(p[\"max_genes\"]), len(genes_all)), p[\"agf\"])" in inspect.getsource(banksy._run_banksy)
    # R7-34: without Leiden no graph is built - its edges cost nothing
    assert clustersets._cs_cost(2_000_000, 50, matrix_gb=12.0, graph=False)["gb"] == 12.0
    assert clustersets._cs_cost(2_000_000, 50, matrix_gb=12.0)["gb"] > 40
    stop, est = clustersets._cs_size_guard("Banksy-style clustering", 2_000_000, 50, matrix_gb=12.0, graph=False)
    assert stop is None and "KMeans fallback" in est and "million edges" not in est
    monkeypatch.setattr(clustersets, "_cs_leiden_available", lambda: False)
    assert clustersets._cs_size_guard("Joint clustering", 2_000_000, 50, matrix_gb=12.0)[0] is None
    monkeypatch.setattr(clustersets, "_cs_leiden_available", lambda: True)
    assert clustersets._cs_size_guard("Joint clustering", 2_000_000, 50, matrix_gb=12.0)[0] is not None
    monkeypatch.undo()
    # R7-33: the memory limit of the container, when it is smaller than the machine's
    f2, f1 = tmp_path / "memory.max", tmp_path / "limit_in_bytes"
    f2.write_text("17179869184\n")
    assert clustersets._cs_cgroup_limit_gb([str(f2)]) == pytest.approx(17.18, abs=0.01)
    f2.write_text("max\n")
    f1.write_text("9223372036854771712\n")                    # the "unlimited" value of cgroup v1
    assert clustersets._cs_cgroup_limit_gb([str(f2), str(f1)]) == 0.0
    assert clustersets._cs_cgroup_limit_gb([str(tmp_path / "missing")]) == 0.0
    phys = clustersets._cs_ram_gb()
    monkeypatch.setattr(clustersets, "_cs_cgroup_limit_gb", lambda files=None: 0.5)
    assert clustersets._cs_ram_gb() == 0.5 < max(phys, 0.6)


# ══ follow-up: noninteger_counts = round_sums is off by default and said plainly ═════
def test_round_sums_is_off_by_default_and_stated_in_record_methods_and_dry_run(xe, D, cfg, tmp_path):
    from xenium_app import h5ad, provenance
    assert h5ad._h5ad_clean_mapping({"counts": "X"})["noninteger_counts"] == "refuse"
    with pytest.raises(ValueError, match="noninteger_counts"):
        h5ad._h5ad_clean_mapping({"counts": "X", "noninteger_counts": "yes"})
    # the methods text of a pseudobulk run that used it
    used = _rec("pseudobulk", params={"population": "all"}, headline={"counts_rounded": True, "n_genes": 10,
                                                                     "n_filtered": 0, "design": "~ condition"})
    md = provenance._prov_methods_md([used])
    assert "does NOT hold raw counts" in md and "noninteger_counts = round_sums" in md
    assert "not from raw counts" in md and "Original (uncorrected) counts were summed" not in md
    plain = _rec("pseudobulk", params={"population": "all"}, headline={"counts_rounded": False, "n_genes": 10,
                                                                       "n_filtered": 0, "design": "~ condition"})
    md = provenance._prov_methods_md([plain])
    assert "Original (uncorrected) counts were summed" in md and "round_sums" not in md
    # the dry run of a batch config says so before anything runs
    extras = [_fab_ds(D, 30, i, str(tmp_path / f"r{i}")) for i in range(4)]
    xe.EXTRA_DATASETS.extend(extras)
    xe._design_set([{"key": xe._ds_key(e), "cond": "AB"[i % 2], "batch": "", "donor": ""}
                    for i, e in enumerate(extras)])
    cfg["steps"] = [{"kind": "pseudobulk", "params": {"population": "all"}}]
    out = xe._batch_run(cfg, dry_run=True, log=_quiet)
    assert not any("round_sums" in w for w in out["warnings"])            # raw counts: nothing to say
    for e in extras:
        e["counts_noninteger"] = "round_sums"
    out = xe._batch_run(cfg, dry_run=True, log=_quiet)
    w = "\n".join(out["warnings"])
    assert "noninteger_counts = round_sums" in w and "not from raw counts" in w
    assert "step 1 (pseudobulk): runs on non-integer counts" in w
    for e in extras:
        e["counts_noninteger"] = "refuse"
    out = xe._batch_run(cfg, dry_run=True, log=_quiet)
    assert not out["ok"] and "pseudobulk refuses it" in "\n".join(out["errors"])
