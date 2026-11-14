"""Regression tests for the eighth review round (dataset switching, loading, external tool
arguments, SpaGE caches, sample-level statistics, methods text, batch mode). One test per
finding (R8-<n> in the comment above it); the name says which behaviour is pinned."""
from __future__ import annotations

import gzip
import inspect
import json
import os
import pathlib
import shutil
import sys
import types

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import xe_testlib as tl
from test_controls import _fab_ds
from test_review_round4 import _pb_view
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


def _refuse_job(*a, **k):
    raise AssertionError("a job was started")


# ── R8-1: a converted .h5ad is installed only when no other job runs ─────────────────
def test_h5ad_install_waits_for_running_jobs(xe, monkeypatch):
    from xenium_app import cb_data, h5ad
    installed = []
    monkeypatch.setattr(cb_data, "_h5ad_install", lambda *res: installed.append(res) or "installed")
    result = ([{"df": pd.DataFrame(index=["a"])}], {"source": "x.h5ad"}, "/nowhere")
    with h5ad._h5ad_lock:
        h5ad._h5ad_state.update({"status": "done", "message": "Converted", "result": result})
    try:
        monkeypatch.setattr(cb_data, "_running_jobs", lambda: ["Load .h5ad", "Baysor"])
        out = xe.h5ad_poll(1, 0, 0, 0, 0)
        assert not installed, "installed although Baysor was running"
        assert "Waiting for Baysor" in out[0] and out[1] is False            # the poll stays on
        assert h5ad._h5ad_state["status"] == "done" and h5ad._h5ad_state["result"] is result
        # the conversion's own job entry does not hold the install back
        monkeypatch.setattr(cb_data, "_running_jobs", lambda: ["Load .h5ad"])
        out = xe.h5ad_poll(2, 0, 0, 0, 0)
        assert len(installed) == 1 and out[0].startswith("installed") and out[1] is True
        assert h5ad._h5ad_state["status"] == "idle"
    finally:
        with h5ad._h5ad_lock:
            h5ad._h5ad_state.update({"status": "idle", "message": "", "result": None})


# ── R8-2: integer cell ids are matched to the (text) barcodes; never a silent zero matrix
def test_matrix_rows_are_matched_to_integer_cell_ids(xe, capsys):
    from xenium_app import loading
    expr = sp.csr_matrix(np.array([[1, 0], [0, 2], [3, 3]], dtype=np.float32))
    barcodes = ["1", "2", "3"]
    got = loading._align_matrix_to_cells(expr, barcodes, pd.Index(np.array([3, 1, 2], dtype=np.int32)))
    assert got.toarray().tolist() == [[3, 3], [1, 0], [0, 2]]
    got = loading._align_matrix_to_cells(expr, [b"1", b"2", b"3"], pd.Index([2.0, 3.0, 1.0]))
    assert got.toarray().tolist() == [[0, 2], [3, 3], [1, 0]]
    # ids that do not correspond at all: refused, with both id forms in the message
    with pytest.raises(ValueError, match="None of the 3 cell ids"):
        loading._align_matrix_to_cells(expr, barcodes, pd.Index(["aaaa-1", "bbbb-1", "cccc-1"]))
    # some cells without a row: zero rows, and a warning that says how many
    capsys.readouterr()
    got = loading._align_matrix_to_cells(expr, barcodes, pd.Index(["2", "zz", "1"]))
    assert got.toarray().tolist() == [[0, 2], [0, 0], [1, 0]]
    assert "WARNING: 1 of 3 cells" in capsys.readouterr().out


def test_bundle_with_integer_cell_ids_builds_a_nonzero_matrix(xe, bundle_dir, tmp_path):
    """The whole store builder on a copy of the fixture whose cells are numbered (the
    format of old Xenium Onboard Analysis versions): the matrix has its counts."""
    import h5py
    from xenium_app import loading
    src = pathlib.Path(bundle_dir)
    dst = tmp_path / "int_ids"
    dst.mkdir()
    for name in ("experiment.xenium", "cells.parquet", "cell_feature_matrix.h5", "cell_boundaries.parquet",
                 "nucleus_boundaries.parquet"):
        shutil.copy(src / name, dst / name)
    cells = pd.read_parquet(dst / "cells.parquet")
    num = {cid: i + 1 for i, cid in enumerate(cells["cell_id"])}
    cells["cell_id"] = cells["cell_id"].map(num).astype("int32")
    cells.to_parquet(dst / "cells.parquet", index=False)
    for name in ("cell_boundaries.parquet", "nucleus_boundaries.parquet"):
        b = pd.read_parquet(dst / name)
        b["cell_id"] = b["cell_id"].map(num).astype("int32")
        b.to_parquet(dst / name, index=False)
    with h5py.File(dst / "cell_feature_matrix.h5", "r+") as f:
        old = [x.decode() for x in f["matrix/barcodes"][:]]
        total = float(np.asarray(f["matrix/data"][:], dtype=np.float64).sum())
        del f["matrix/barcodes"]
        f["matrix"].create_dataset("barcodes", data=np.array([str(num[b]).encode() for b in old]))
    loading._build_xenium_sdata(str(dst))
    import spatialdata
    table = spatialdata.read_zarr(str(dst / "spatialdata_xenium.zarr")).tables["table"]
    assert table.n_obs == len(cells) and list(table.obs_names[:3]) == ["1", "2", "3"]
    assert float(table.X.sum()) > 0, "all-zero expression matrix"
    # the fixture's matrix also holds control features, which are dropped: compare per cell
    # through the app's own loader of the original bundle
    assert float(table.X.sum()) <= total
    assert float(table.X.sum()) == pytest.approx(float(xe.DATA["expr"][:, :table.n_vars].sum()), rel=1e-6)


# ── R8-3: assigning to a sibling-module name does not rebind it in the other modules ──
def test_repl_assignment_to_a_module_name_stays_local(xe):
    from xenium_app import compat, cb_data, stats as stats_mod
    ns = compat.ReplNamespace()
    exec("stats = {'a': 1}\nseen = stats['a']", ns)
    assert cb_data.stats is stats_mod, "cb_data.stats was replaced by the REPL variable"
    assert dict.__contains__(ns, "stats") and ns["seen"] == 1
    assert compat.assign("jobs", 5) is False and compat.delete("jobs") is False
    from xenium_app import jobs as jobs_mod
    assert cb_data.jobs is jobs_mod
    # the compat module: xe.<module name> = value is kept on xe only
    xe.regions = "mine"
    try:
        from xenium_app import regions as regions_mod
        assert cb_data.regions is regions_mod if hasattr(cb_data, "regions") else True
        from xenium_app import figures
        assert figures.regions is regions_mod
    finally:
        del xe.regions
    # a global of the app is still rebound where it is defined
    old = stats_mod._COMP_MIN_SAMPLES
    try:
        xe._COMP_MIN_SAMPLES = 99
        assert stats_mod._COMP_MIN_SAMPLES == 99
    finally:
        xe._COMP_MIN_SAMPLES = old
    assert stats_mod._COMP_MIN_SAMPLES == old


# ── R8-4: the cluster-set dropdown is refreshed on page load ─────────────────────────
def test_cluster_options_callback_runs_on_page_load(xe, D):
    key, cb = next((k, v) for k, v in xe.app.callback_map.items()
                   if "cluster-method.options" in k and "cluster-method.value" in k)
    spec = next(c for c in xe.app._callback_list if c["output"] == key)
    assert spec["prevent_initial_call"] is False
    # what the page-load call returns for the loaded dataset: its own sets, selection kept
    opts, value = xe.update_cluster_options(0, 0, "xenium", 0, 0, 0, D["cluster_methods"][0])
    assert D["cluster_methods"][0] in [o["value"] for o in opts]
    opts, value = xe.update_cluster_options(0, 0, "xenium", 0, 0, 0, "set_of_another_dataset")
    assert value == [o["value"] for o in opts][0]


# ── R8-5: the Cells row is the loaded table, not experiment.xenium ───────────────────
def _text_of(component) -> str:
    out = []

    def walk(c):
        if c is None:
            return
        if isinstance(c, (str, int, float)):
            out.append(str(c))
        elif isinstance(c, (list, tuple)):
            for x in c:
                walk(x)
        else:
            walk(getattr(c, "children", None))
    walk(component)
    return " | ".join(out)


def test_tissue_info_counts_the_loaded_cells(xe, D, monkeypatch):
    from xenium_app import core
    n = len(D["df"])
    monkeypatch.setitem(core.DATA, "metadata", {**D.get("metadata", {}), "num_cells": 987654})
    txt = _text_of(xe.update_tissue_info(0, "xenium", 0))
    assert f"{n:,}" in txt.split(" | ") and "987,654" not in txt
    monkeypatch.setitem(core.DATA, "metadata", {})                 # a store written from a run: no metadata
    txt = _text_of(xe.update_tissue_info(0, "xenium", 0))
    assert f"{n:,}" in txt.split(" | ")


# ── R8-6: an added sample is placed beside the whole primary sample ──────────────────
def test_added_sample_is_offset_from_the_full_primary_sample(xe, D, monkeypatch, tmp_path):
    from xenium_app import core, loading
    new_dir = tmp_path / "extra_bundle"
    new_dir.mkdir()
    (new_dir / "experiment.xenium").write_text("{}")
    new = _fab_ds(D, 30, 0, str(new_dir))
    monkeypatch.setattr(loading, "load_xenium_data", lambda p: new)
    full = D["df"]
    left = full[full["x_centroid"] <= full["x_centroid"].median()]
    assert left["x_centroid"].max() < full["x_centroid"].max()
    monkeypatch.setitem(core.DATA, "_df_original", full)           # subset() active: df is a slice
    monkeypatch.setitem(core.DATA, "df", left)
    try:
        out = xe.add_sample(1, None, str(new_dir), 0)
        assert str(out[0]).startswith("Added"), out
        want = float(full["x_centroid"].max()) - float(new["df"]["x_centroid"].min()) + 500.0
        assert xe.EXTRA_DATASETS[-1]["x_offset"] == pytest.approx(want)
    finally:
        xe.EXTRA_DATASETS.clear()
        core.DATA.pop("_df_original", None)


# ── R8-7: only integers reach the Proseg command line ────────────────────────────────
def test_proseg_schedule_accepts_integers_only(xe, monkeypatch):
    from xenium_app import core, jobs, reseg
    assert reseg._proseg_schedule(" 150  150,300 ") == "150 150 300"
    assert reseg._proseg_schedule("") is None and reseg._proseg_schedule(None) is None
    for bad in ("150 150 --output-expected-counts /somewhere/notes.csv.gz", "150 -5", "1e3", "150;rm", "0",
                "١٥٠", "9" * 12, " ".join(["10"] * 40)):
        with pytest.raises(ValueError, match="Sampler schedule"):
            reseg._proseg_schedule(bad)
    # the callback refuses before anything is claimed or started
    monkeypatch.setattr(jobs, "_job_start", _refuse_job)
    args = [1, "proseg", "skip"] + [None] * 12 + [None, None, None, None, None, None, None, None]
    out = xe.run_reseg_modal(*args, "150 150 --output-expected-counts /somewhere/notes.csv.gz", None, None, "nucleus")
    assert "not started" in out[0] and "Sampler schedule" in out[0]
    assert core._proseg_state["status"] != "running"
    out = xe.run_reseg_modal(*(args[:19] + ["--nthreads=99"] + args[20:]), "150 150", None, None, "nucleus")
    assert "not started" in out[0] and core._proseg_state["status"] != "running"


def test_proseg_command_line_holds_numbers_only(xe, monkeypatch, tmp_path):
    from xenium_app import jobs, reseg
    seen = {}

    def run(cmd, **kw):
        seen["cmd"] = list(cmd)
        return 1                                                   # "failed": nothing is read back

    monkeypatch.setattr(jobs, "_job_run_process", run)
    tx = tmp_path / "transcripts.parquet"
    pd.DataFrame({"x": [0.0]}).to_parquet(tx)
    with pytest.raises(RuntimeError, match="exited with code 1"):
        reseg._proseg_run_single(str(tx), str(tmp_path / "p"), "proseg", 2, 4, n_samples=10,
                                 recorded_samples=50, schedule="150,150 300")
    cmd = seen["cmd"]
    i = cmd.index("--schedule")
    assert cmd[i + 1:i + 4] == ["150", "150", "300"]
    assert cmd[cmd.index("--initial-voxel-size") + 1] == "2.0" and cmd[cmd.index("--nthreads") + 1] == "4"
    seen.clear()
    with pytest.raises(ValueError, match="Sampler schedule"):
        reseg._proseg_run_single(str(tx), str(tmp_path / "p"), "proseg", 2, 4,
                                 schedule="150 --output-expected-counts /somewhere/x")
    assert not seen, "Proseg was started with an injected option"
    with pytest.raises(ValueError):
        reseg._proseg_run_single(str(tx), str(tmp_path / "p"), "proseg", "2 --flag", 4)
    assert not seen


# ── R8-8: one CellNEST load at a time ────────────────────────────────────────────────
def test_cellnest_load_refuses_while_one_is_loading(xe, monkeypatch):
    from xenium_app import core, jobs
    monkeypatch.setattr(jobs, "_job_start", _refuse_job)
    with core._cellnest_lock:
        before = dict(core._cellnest_state)
        core._cellnest_state.update({"status": "loading", "message": "Loading…", "data": "A"})
    try:
        out = xe.load_cellnest(1, "/some/folder", "dataset_b", "")
        assert "Already loading" in out[0]
        assert core._cellnest_state["data"] == "A"                 # the running load was not reset
    finally:
        with core._cellnest_lock:
            core._cellnest_state.clear()
            core._cellnest_state.update(before)


# ── R8-9: global ids of a cached Proseg patch run follow the run's record ────────────
def _proseg_patch(root, k, cells):
    d = root / f"patch_{k:04d}"
    d.mkdir()
    pd.DataFrame({"cell": cells, "volume": [10.0 * (k + 1)] * len(cells)}).to_csv(d / "cell-metadata.csv.gz", index=False)
    feats = [{"type": "Feature", "properties": {"cell": c},
              "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}} for c in cells]
    with gzip.open(d / "cell-polygons.geojson.gz", "wt") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh)
    return d.name


def test_proseg_patch_ids_skip_a_failed_patch_that_left_its_files(xe, tmp_path):
    from xenium_app import reseg
    stride = 10_000_000
    names = [_proseg_patch(tmp_path, k, [0, 1]) for k in range(3)]
    # patch 1 failed after Proseg wrote its metadata: the live run numbered patch 2 second
    (tmp_path / "params.json").write_text(json.dumps({"patch_ok": [0, 2]}))
    bounds = reseg._load_proseg_patch_boundaries(str(tmp_path), names)
    assert set(bounds) == {"1", "2", str(1 + stride), str(2 + stride)}
    assert reseg._proseg_patch_offsets(str(tmp_path)) == [("patch_0000", 1), ("patch_0002", 1 + stride)]
    # a run recorded before patch_ok existed: the position among the directories, as before
    (tmp_path / "params.json").write_text(json.dumps({}))
    assert [o for _n, o in reseg._proseg_patch_offsets(str(tmp_path))] == [1, 1 + stride, 1 + 2 * stride]


# ── R8-10: a Cellpose run is recorded and described as one ───────────────────────────
def test_cellpose_run_is_described_as_cellpose(xe):
    from xenium_app import imgseg, provenance
    rec = {"title": "Image-based segmentation", "cache": "computed", "data": {"segmentation": "imgseg"},
           "params": {"backend": "cellpose:nuclei", "qv_min": 20},
           "headline": {"n_cells": 10, "n_transcripts": 100, "n_transcripts_assigned": 90}}
    sentences, refs = provenance._pm_imgseg(rec)
    text = " ".join(sentences)
    assert "Cellpose (model 'nuclei'" in text and "classical backend" not in text and "cellpose" in refs
    rec["params"]["backend"] = "classical"
    sentences, refs = provenance._pm_imgseg(rec)
    assert "classical backend" in " ".join(sentences) and "cellpose" not in refs
    # the package versions of the record follow the resolved backend, not the raw value
    src = inspect.getsource(imgseg._run_imgseg)
    assert '("cellpose", "torch") if info["backend"] == "cellpose"' in src


# ── R8-11: SpaGE results under a subset are matched by cell id ────────────────────────
def test_spage_result_of_another_subset_is_read_by_cell_id(xe, D):
    from xenium_app import core, dataview, spage
    df = D["df"]
    other = pd.DataFrame({"FAKEG": np.arange(len(df), dtype=np.float64)}, index=df.index[::-1])
    with core._spage_lock:
        saved = dict(core._spage_state)
        core._spage_state.update({"result": other, "result_path": None, "result_genes": None})
    dataview._gene_expr_cache.clear()
    try:
        got = dataview.get_gene_expression("FAKEG [imp]")
        want = np.log1p(other["FAKEG"].reindex(df.index).to_numpy())
        assert np.allclose(got, want), "values of another cell order were attached by position"
    finally:
        with core._spage_lock:
            core._spage_state.clear()
            core._spage_state.update(saved)
        dataview._gene_expr_cache.clear()
    # the cache key of a subset names its cells: same size, other cells -> another key
    a, b = spage._spage_ids_tag(df.index[:10]), spage._spage_ids_tag(df.index[5:15])
    assert a != b and a == spage._spage_ids_tag(list(df.index[:10]))
    src = inspect.getsource(spage._run_spage_imputation)
    assert src.count("_subset_ids_tag}") == 2                      # Xenium and run subsets


def test_streamed_spage_result_of_another_subset_is_refused(xe, D, monkeypatch):
    from xenium_app import core, figures, spage
    df = D["df"]
    figures._spage_streamed_cells_check(None)                      # a result of the whole dataset
    figures._spage_streamed_cells_check(spage._spage_ids_tag(df.index))
    with pytest.raises(ValueError, match="another subset"):
        figures._spage_streamed_cells_check(spage._spage_ids_tag(df.index[::-1]))


# ── R8-12: the SpaGE cache of a run is the cache of THAT run ─────────────────────────
def test_spage_cache_key_of_a_run_holds_its_run_id(xe):
    from xenium_app import spage
    src = inspect.getsource(spage._run_spage_imputation)
    key = src[src.index("_cache_seg_suffix = "):src.index("cache_file = _spage_cache_path")]
    assert "core._seg_run_id(_alt_res)" in key and '_cache_seg_suffix += f"_r{_run_id}"' in key
    # two runs in one directory under one tag differ in their run id only
    a = spage._spage_cache_path("/ref.rds", "G1__proseg_abcd_r" + "1" * 8)
    b = spage._spage_cache_path("/ref.rds", "G1__proseg_abcd_r" + "2" * 8)
    assert a != b


# ── R8-13: a run's SpaGE result kept in the parquet cache is readable ────────────────
def test_run_spage_result_in_the_parquet_cache_is_read_by_cell_id(xe, D, tmp_path):
    from xenium_app import figures
    res = _fake_run(xe, D, n=40)
    ids = list(res["cells_df"].index)
    part = pd.DataFrame({"FAKEG": np.arange(1, 21, dtype=np.float32)}, index=ids[:20][::-1])   # 20 of 40 cells
    path = tmp_path / "spage_abc.parquet"
    part.to_parquet(path)
    res["spage_result_path"], res["spage_result_genes"] = str(path), ["FAKEG"]
    got = figures._get_expr_values("FAKEG [imp]", alt_res=res)
    assert got is not None, "the gene is listed but has no values"
    want = np.log1p(part["FAKEG"].reindex(ids).fillna(0).to_numpy())
    assert np.allclose(got, want)
    # ... and the run loaders restore such a reference
    from xenium_app import imgseg, reseg
    assert "os.path.isdir(_sref" not in inspect.getsource(reseg) and "os.path.isdir(sref" not in inspect.getsource(imgseg)


# ── R8-14: Rerun UMAP on corrected counts embeds every cell that has them ────────────
def test_corrected_umap_embeds_all_cells_with_corrected_rows(xe, D, monkeypatch):
    from xenium_app import core, provenance, clustering, dataview
    df = D["df"]
    n_panel = len(dataview._panel_gene_names())
    corr = sp.csr_matrix(D["expr"])[:, :n_panel].copy()
    d2e = np.asarray(D["df_to_expr"])
    n_rows = int((np.asarray(corr.sum(axis=1)).ravel()[d2e] > 0).sum())
    assert n_rows > 8
    monkeypatch.setitem(core.DATA, "split_corrected_expr", corr)
    # the id list of a later SPLIT run under subset(): 8 cells, although the matrix (merged
    # with the earlier full run) has corrected rows for all
    monkeypatch.setitem(core.DATA, "split_corrected_cell_ids", [str(c) for c in df.index[:8]])
    monkeypatch.setitem(sys.modules, "umap.umap_", None)                        # t-SNE: fast on the fixture
    cols = [c for c in ("split_umap_1", "split_umap_2", "umap_1", "umap_2") if c in df.columns]
    keep = df[cols].copy()
    with core._umap_reseg_lock:
        core._umap_reseg_state.update({"status": "running", "seg_source": "xenium", "counts_mode": "corrected"})
    n0 = len(provenance._prov_recent)
    try:
        clustering._run_reseg_umap()
        assert core._umap_reseg_state["status"] == "done", core._umap_reseg_state["message"]
        r = [x for x in provenance._prov_recent[n0:] if x["kind"] == "umap"][-1]
        assert r["headline"]["n_cells"] == n_rows, r["status_line"]
        assert int(np.isfinite(df["split_umap_1"]).sum()) == n_rows
    finally:
        for c in ("split_umap_1", "split_umap_2"):
            if c in df.columns and c not in keep.columns:
                del df[c]
        for c in keep.columns:
            df[c] = keep[c]
        with core._umap_reseg_lock:
            core._umap_reseg_state.update({"status": "idle", "message": "", "result": None})
        dataview._umap_df_cache.clear()


# ── R8-15: the record of an unseeded UMAP does not state a seed ──────────────────────
def test_umap_record_states_no_seed_when_umap_ran_unseeded(xe, D, monkeypatch):
    from xenium_app import core, provenance, clustering
    assert clustering._umap_seeds(1000) == {"random_state": 0}
    big = clustering._umap_seeds(clustering._UMAP_PARALLEL_MIN_CELLS + 1)
    assert "random_state" not in big and "without a seed" in big["umap_random_state"] and big["svd_random_state"] == 0
    assert clustering._umap_seeds(clustering._UMAP_PARALLEL_MIN_CELLS + 1, "t-SNE") == {"random_state": 0}
    # through the job: a stand-in umap-learn, the threshold lowered below the 60 cells of the run
    kwargs = {}

    class UMAP:
        def __init__(self, **kw):
            kwargs.update(kw)

        def fit_transform(self, x):
            return np.random.default_rng(0).normal(size=(len(x), 2))

    fake = types.ModuleType("umap.umap_")
    fake.UMAP = UMAP
    monkeypatch.setitem(sys.modules, "umap.umap_", fake)
    monkeypatch.setattr(clustering, "_UMAP_PARALLEL_MIN_CELLS", 10)
    _fake_run(xe, D, n=60)
    with core._umap_reseg_lock:
        core._umap_reseg_state.update({"status": "running", "seg_source": "proseg:t", "counts_mode": "original"})
    n0 = len(provenance._prov_recent)
    clustering._run_reseg_umap()
    assert core._umap_reseg_state["status"] == "done", core._umap_reseg_state["message"]
    assert "random_state" not in kwargs and kwargs.get("n_jobs") == -1          # it did run unseeded
    r = [x for x in provenance._prov_recent[n0:] if x["kind"] == "umap"][-1]
    assert "random_state" not in r["seeds"] and "without a seed" in r["seeds"]["umap_random_state"]
    assert "Random seeds: random_state = 0" not in provenance._prov_methods_md([r])


# ── R8-16: a gene entered twice is imputed once ──────────────────────────────────────
def test_spage_gene_list_is_deduplicated(xe):
    from xenium_app import spage
    assert spage._spage_requested_genes("ACTA2, MYH11\nACTA2\n\n MYH11 ,CD3E") == ["ACTA2", "MYH11", "CD3E"]
    assert spage._spage_requested_genes("") == []
    assert "_spage_requested_genes(genes_input)" in inspect.getsource(spage._run_spage_imputation)


# ── R8-17: donor labels on some samples: the labelled donors' sections are one unit ──
def test_moderated_composition_combines_sections_with_partial_donor_labels(xe, D, tmp_path):
    types_ = ("Alpha", "Beta", "Gamma", "Delta")
    donors = ["a1", "a1", "a2", "", "b1", "b1", "b2", ""]
    conds = ["A"] * 4 + ["B"] * 4
    rng = np.random.default_rng(5)
    extras = [_fab_ds(D, 300, i, str(tmp_path / f"u{i}"), types_, rng.dirichlet([4, 3, 2, 1])) for i in range(8)]
    xe.EXTRA_DATASETS.extend(extras)
    xe._design_set([{"key": xe._ds_key(e), "cond": c, "batch": "", "donor": d}
                    for e, c, d in zip(extras, conds, donors)])
    res = xe._comp_compute("celltype:celltypist", "xenium", None, "logit", "moderated")
    assert (res["n_a"], res["n_b"]) == (3, 3), res["test_note"]      # not 4 vs 4 sections
    note = res["test_note"]
    assert "(3 independent units)" in note and "4 samples" not in note
    assert "Unit of the p-values: n = 8 samples (4 donors, 2 samples without a donor), 6 independent units" in note
    assert "their transformed proportions were averaged before the test" in note
    # the same units as the exact test of the same design
    ex = xe._comp_compute("celltype:celltypist", "xenium", None, "logit", "exact")
    assert "6 independent units" in ex["test_note"]


# ── R8-18: pooled cell-level DE normalises every cell over the shared genes ──────────
def _panel_sample(label, genes, counts):
    n = counts.shape[0]
    return {"label": label, "df": pd.DataFrame(index=[f"{label}_{i}" for i in range(n)]),
            "rows": np.arange(n), "csc": sp.csc_matrix(counts.astype(np.float32)), "genes": list(genes),
            "gni": {g: i for i, g in enumerate(genes)}, "corrected": False}


def test_pooled_de_uses_the_library_over_the_shared_genes(xe):
    from xenium_app import analysis
    rng = np.random.default_rng(3)
    shared = [f"g{i}" for i in range(5)]
    wide = shared + [f"x{i}" for i in range(45)]
    n = 200
    s0 = _panel_sample("S0", shared, rng.poisson(5.0, size=(n, 5)))
    # the same biology in the shared genes, measured with a ten times larger panel
    s1 = _panel_sample("S1", wide, np.hstack([rng.poisson(5.0, size=(n, 5)), rng.poisson(5.0, size=(n, 45))]))
    ones, zeros = np.ones(n, bool), np.zeros(n, bool)
    res = analysis._de_compute([s0, s1], [ones, zeros], [zeros, ones])
    t = res["table"].set_index("gene")
    assert list(t.index) == shared and res["lib_shared_genes"] == 5
    # before: every gene "higher in A" by log2(10) because B was divided by 250 counts, not 25
    assert float(t["log2fc"].abs().max()) < 0.5, t["log2fc"].round(2).to_dict()
    assert res["median_counts_a"] == pytest.approx(res["median_counts_b"], rel=0.2)
    assert float((t["rate_a"] / t["rate_b"]).max()) < 1.5
    # one panel: nothing changes, and nothing is announced
    s2 = _panel_sample("S1", shared, rng.poisson(5.0, size=(n, 5)))
    res = analysis._de_compute([s0, s2], [ones, zeros], [zeros, ones])
    assert res["lib_shared_genes"] is None
    sf_all, _ = analysis._an_size_factors(s0)
    sf_shared, _ = analysis._an_size_factors(s0, shared)
    assert np.array_equal(sf_all, sf_shared)
    # the statements that go into a manuscript say so
    from xenium_app import provenance
    rec = {"data": {"segmentation": "xenium"}, "params": {},
           "headline": {"label_a": "a", "label_b": "b", "n_a": n, "n_b": n, "n_genes": 5, "lib_shared_genes": 5}}
    assert "total over the 5 shared genes" in " ".join(provenance._pm_de(rec)[0])
    rec["headline"]["lib_shared_genes"] = None
    assert "shared genes" not in " ".join(provenance._pm_de(rec)[0])


def test_pooled_ligand_receptor_uses_the_shared_gene_library(xe):
    from xenium_app import stats
    src = inspect.getsource(stats._lr_compute)
    assert "analysis._an_size_factors(s, genes_all)" in src and "lib_shared_genes" in src


# ── R8-19: an excluded sample does not stop the pseudobulk test ──────────────────────
def test_pseudobulk_ignores_noninteger_counts_of_an_excluded_sample(xe, D, tmp_path):
    samples, masks, cd, bt, _e = _pb_view(xe, D, tmp_path, list("AAABBB"), [""] * 6)
    assert cd[0] == ""                                             # the primary sample is excluded
    samples[0] = {**samples[0], "csc": samples[0]["csc"].astype(np.float32) * 0.5, "ds": {}}
    assert not xe._counts_are_integer(samples[0])
    prep = xe._pb_prepare(samples, masks, cd, bt, 1, 0)
    assert prep["counts"].shape[1] == 6
    # a sample of the design with such counts is still refused
    samples[1] = {**samples[1], "csc": samples[1]["csc"].astype(np.float32) * 0.5, "ds": {}}
    with pytest.raises(ValueError, match="non-integer"):
        xe._pb_prepare(samples, masks, cd, bt, 1, 0)


# ── R8-20: the units note counts the donors that were given ──────────────────────────
def test_units_note_counts_named_donors_only(xe):
    u = xe._sl_units(list("AAABBB"), ["D1", "D1", "D2", "D3", "D4", ""])
    assert u["n_donors"] == 4 and u["n_units"] == 5
    assert u["note"].startswith("n = 6 samples (4 donors, 1 sample without a donor), 5 independent units")
    assert xe._sl_units(list("AABB"), ["d1", "d2", "d3", "d4"])["note"] == "n = 4 samples (4 donors)"
    assert xe._sl_units(list("AABB"), ["", "", "", ""])["note"] == "n = 4 samples"


# ── R8-21: the proximity methods text names the covariate the run used ───────────────
def _prox_rec(req, **headline):
    return {"data": {"segmentation": "xenium"},
            "params": {"target": "T cell", "anchor": "Tumour", "source": req, "near": 20, "far": 100},
            "headline": {"estimator": "nearfar", "fdr": "bh", "mode": "single", **headline}}


def test_proximity_methods_text_states_the_resolved_density_covariate(xe):
    from xenium_app import provenance
    txt = " ".join(provenance._pm_proximity(_prox_rec("auto", source="Fibroblast"))[0])
    assert "local fraction of 'Fibroblast' cells" in txt and "'auto'" not in txt
    txt = " ".join(provenance._pm_proximity(_prox_rec("auto", source=None))[0])
    assert "no density-covariate cell type was used" in txt and "'auto'" not in txt
    txt = " ".join(provenance._pm_proximity(_prox_rec("auto"))[0])     # an old record without the result field
    assert "'auto'" not in txt and "not recorded" in txt
    txt = " ".join(provenance._pm_proximity(_prox_rec("B cell"))[0])
    assert "local fraction of 'B cell' cells" in txt


# ── R8-22: marker rule and reference of the spillover / segmentation-free text ───────
def test_spillover_and_segfree_methods_follow_the_recorded_marker_rule(xe):
    from xenium_app import provenance
    base = {"data": {"segmentation": "xenium"}, "params": {"cell_type": "T cell", "n_null": 200, "seed": 0}}
    fb = {**base, "headline": {"reference_kind": "mean", "marker_info": {"rule": "fallback", "n_strict": 1}}}
    txt = " ".join(provenance._pm_spillover(fb)[0])
    assert "divided by the mean of that share" in txt and "median was 0" in txt
    assert "fewer than 3 genes" in txt and "not exclusive" in txt
    assert "Markers were the source type's most specific genes" not in txt
    sp_ = {**base, "headline": {"reference_kind": "median", "marker_info": {"rule": "specific"}}}
    txt = " ".join(provenance._pm_spillover(sp_)[0])
    assert "divided by the median of that share inside source cells." in txt and "10x the level of every other type" in txt
    assert "fewer than 3" not in txt and "listed by the user" not in txt
    us = {**base, "headline": {"reference_kind": "median", "marker_info": {"rule": "user"}}}
    assert "Markers were the genes listed by the user." in " ".join(provenance._pm_spillover(us)[0])
    txt = " ".join(provenance._pm_segfree(fb)[0])
    assert "fewer than 3 genes" in txt and "not exclusive" in txt
    txt = " ".join(provenance._pm_segfree(us)[0])
    assert "Markers were the genes listed by the user." in txt and "10x" not in txt
    # the headline of a real result keeps the rule (a small dict survives _prov_headline)
    h = provenance._prov_headline({"reference_kind": "mean", "marker_info": {"rule": "fallback", "n_strict": 0}})
    assert provenance._pm_marker_rule(h) == "fallback" and h["reference_kind"] == "mean"


# ── R8-23: no core is described when none was defined ────────────────────────────────
def test_compartments_methods_text_without_labels_claims_no_core(xe):
    from xenium_app import provenance
    q = {"bin_um": 20, "close_bins": 1, "hole_max_bins": 4, "radius": 30, "min_nb": 10, "min_frac": 0.3, "min_edge": 200}
    rec = {"data": {"segmentation": "xenium"}, "params": {"dominant": "auto", "resolved": q},
           "headline": {"dominant": None}}
    txt = " ".join(provenance._pm_compartments(rec)[0])
    assert "no core was defined" in txt and "Dense core" not in txt
    rec["headline"]["dominant"] = "Epithelial"
    txt = " ".join(provenance._pm_compartments(rec)[0])
    assert "Dense core of the cell type 'Epithelial' (the most frequent type of the view)" in txt
    assert "at least 10 other cells" in txt


# ── R8-24: Harmony gets its covariate as a DataFrame (every harmonypy release) ───────
def test_harmony_is_called_with_a_dataframe(xe, monkeypatch):
    from xenium_app import joint
    seen = {}

    def run_harmony(data_mat, meta_data, vars_use, **kw):
        # what the harmonypy 0.0.x releases do with the covariate table
        n = meta_data.shape[0]
        levels = pd.get_dummies(meta_data[vars_use])
        seen.update(n=n, levels=levels.shape[1], kw=kw)
        return types.SimpleNamespace(Z_corr=np.asarray(data_mat).T.copy())

    monkeypatch.setattr(joint, "_jc_harmony_module", lambda: types.SimpleNamespace(run_harmony=run_harmony))
    pcs = np.random.default_rng(0).normal(size=(12, 4)).astype(np.float32)
    out = joint._jc_harmony(pcs, ["a"] * 6 + ["b"] * 6, seed=3)
    assert out.shape == pcs.shape and seen["n"] == 12 and seen["levels"] == 2 and seen["kw"]["random_state"] == 3


# ── R8-25: a click in the joint UMAP selects the cell of its sample ──────────────────
def test_click_in_the_joint_umap_resolves_sample_and_cell(xe, D, monkeypatch, tmp_path):
    from xenium_app import cb_plots
    extras = [_fab_ds(D, 20, i, str(tmp_path / f"j{i}")) for i in range(2)]
    xe.EXTRA_DATASETS.extend(extras)
    monkeypatch.setattr(cb_plots, "callback_context", types.SimpleNamespace(triggered_id="umap-plot"))
    click = lambda cid: {"points": [{"customdata": cid, "x": 0.0, "y": 0.0}]}
    first = str(D["df"].index[0])
    assert xe.capture_click(None, click(f"S0:{first}"), None) == first
    assert xe.capture_click(None, click("S2:k1_7"), None) == {"id": "k1_7", "sample": 2}
    assert xe.capture_click(None, click("S1:k0_3"), None) == {"id": "k0_3", "sample": 1}
    # the ordinary UMAP (plain ids), an unknown sample or cell: unchanged
    assert xe.capture_click(None, click(first), None) == first
    assert xe.capture_click(None, click("S9:k0_3"), None) == "S9:k0_3"
    assert xe.capture_click(None, click("S1:nope"), None) == "S1:nope"
    # what the info panel does with the result: the cell is found in its sample
    info = xe.show_cell_info({"id": "k1_7", "sample": 2}, None)
    assert "not found" not in _text_of(info)


# ── R8-26: sequential values keep their colour scale in the joint UMAP ───────────────
def test_joint_umap_draws_sequential_values_like_the_map(xe, D, monkeypatch, tmp_path):
    from xenium_app import analysis, joint, stats
    xe.EXTRA_DATASETS.append(_fab_ds(D, 20, 0, str(tmp_path / "j0")))
    tables = stats._view_tables("xenium")
    rng = np.random.default_rng(1)
    view = {"variant": next(iter(joint._JC_VARIANTS)), "tables": tables,
            "x": [rng.normal(size=len(t["df"])) for t in tables], "y": [rng.normal(size=len(t["df"])) for t in tables]}
    monkeypatch.setattr(joint, "_jc_umap_view", lambda method, seg="xenium": view)
    n = sum(len(t["df"]) for t in tables)
    vals = rng.uniform(0, 3, n)
    monkeypatch.setattr(analysis, "_geneset_color", lambda cb, seg, uc=False: (vals, 3.0, "Control: demo"))
    monkeypatch.setitem(analysis._view_color_seq, "Control: demo", (0.0, 3.0))
    fig = joint._jc_umap_fig("geneset:demo", "joint", None, 3, 0.8)
    mk = fig.data[0].marker
    assert (mk.cmin, mk.cmax) == (0.0, 3.0), "a 0..hi value drawn on a scale centred on zero"
    assert [c[1] for c in mk.colorscale][0].lower() == "#440154"   # Viridis
    # a signed score: diverging around zero, as before
    analysis._view_color_seq.pop("Control: demo")
    mk = joint._jc_umap_fig("geneset:demo", "joint", None, 3, 0.8).data[0].marker
    assert (mk.cmin, mk.cmax) == (-3.0, 3.0)


# ── R8-27: the recorded design is paired by the rule of the tests ────────────────────
def test_recorded_design_is_paired_by_the_rule_of_the_tests(xe, D, tmp_path):
    from xenium_app import provenance
    extras = [_fab_ds(D, 30, i, str(tmp_path / f"d{i}")) for i in range(4)]
    xe.EXTRA_DATASETS.extend(extras)

    def design(conds, donors):
        xe._design_set([{"key": xe._ds_key(e), "cond": c, "batch": "", "donor": d}
                        for e, c, d in zip(extras, conds, donors)])
        de = provenance._prov_design()
        return de, provenance._pm_design({"design": de, "design_used": True})

    # one donor in both conditions: the tests treat this as unpaired
    de, txt = design(list("ABAB"), ["d1", "d1", "d2", "d3"])
    assert de["paired_donors"] is False and de["donors_in_both"] == ["d1"]
    assert "d1 occurs in both conditions, but the design is not paired" in txt and "(paired)" not in txt
    assert de["paired_donors"] == xe._sl_units(list("ABAB"), ["d1", "d1", "d2", "d3"])["paired"]
    # two donors in both conditions, a donor on every sample: paired
    de, txt = design(list("ABAB"), ["d1", "d1", "d2", "d2"])
    assert de["paired_donors"] is True and "the design is paired by donor" in txt
    # donors on some samples only: never paired
    de, txt = design(list("ABAB"), ["d1", "d1", "d2", ""])
    assert de["paired_donors"] is False and de["donors_on_all"] is False
    assert "Some included samples have no donor label" in txt
    de, txt = design(list("AABB"), ["d1", "d2", "d3", "d4"])
    assert "no donor occurs in both conditions" in txt


# ── R8-28: a design entry names one sample, the label first ──────────────────────────
def test_batch_design_resolves_labels_before_names(xe, D, cfg, tmp_path):
    from xenium_app import batch
    # the sample labelled S1 is NAMED S2 and the one labelled S2 is named S1
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    xe.EXTRA_DATASETS.extend([_fab_ds(D, 30, 0, str(tmp_path / "S2")), _fab_ds(D, 30, 1, str(tmp_path / "S1")),
                              _fab_ds(D, 30, 2, str(tmp_path / "x" / "Region_1")),
                              _fab_ds(D, 30, 3, str(tmp_path / "y" / "Region_1"))])
    cfg["data"]["design"] = {"samples": [{"sample": "S1", "condition": "A", "donor": "d1"},
                                         {"sample": "S2", "condition": "B", "donor": "d2"}]}
    batch._batch_setup_samples(batch._batch_norm_config(cfg), _quiet)
    rows = {r["label"]: r for r in xe._design_samples()}
    assert (rows["S1"]["cond"], rows["S1"]["donor"]) == ("A", "d1")
    assert (rows["S2"]["cond"], rows["S2"]["donor"]) == ("B", "d2")
    # a folder name two samples share is refused; the path names one of them
    cfg["data"]["design"] = {"samples": [{"sample": "Region_1", "condition": "A"}]}
    with pytest.raises(batch._BatchError, match="fits several loaded samples"):
        batch._batch_setup_samples(batch._batch_norm_config(cfg), _quiet)
    cfg["data"]["design"] = {"samples": [{"sample": str(tmp_path / "y" / "Region_1"), "condition": "A"}]}
    batch._batch_setup_samples(batch._batch_norm_config(cfg), _quiet)
    assert {r["label"]: r["cond"] for r in xe._design_samples()}["S4"] == "A"
    # two entries for one sample are refused
    cfg["data"]["design"] = {"samples": [{"sample": "S3", "condition": "A"},
                                         {"sample": str(tmp_path / "x" / "Region_1"), "condition": "B"}]}
    with pytest.raises(batch._BatchError, match="both name the sample"):
        batch._batch_setup_samples(batch._batch_norm_config(cfg), _quiet)


@pytest.fixture()
def cfg(xe, bundle_dir, tmp_path):
    return {"data": {"path": str(bundle_dir)}, "output_dir": str(tmp_path / "out"),
            "steps": [{"kind": "moran", "params": {"k": 6}}]}


# ── R8-29: a clustering step exports the embedding it computed ───────────────────────
def test_batch_clustering_table_holds_the_embedding_of_the_counts_mode(xe, D, monkeypatch):
    from xenium_app import batch, core, provenance
    df = D["df"].copy()
    df["umap_1"], df["umap_2"] = 1.0, 2.0                          # the earlier original-counts embedding
    df["split_umap_1"], df["split_umap_2"] = 7.0, 8.0              # what the corrected step wrote
    df["cluster_split_10"] = "3"
    monkeypatch.setitem(core.DATA, "df", df)
    ctx = {"seg-source": "xenium", "counts-mode-store": "corrected", "cluster-method": None, "color-by": "cluster"}
    monkeypatch.setattr(provenance, "_prov_recent", [{"kind": "umap", "headline": {"embedding_columns": "split_umap_1/2"}}])
    out = batch._batch_cell_table("clustering", {}, ctx)
    assert {"split_umap_1", "split_umap_2", "cluster_split_10"} <= set(out.columns) and "umap_1" not in out.columns
    assert float(out["split_umap_1"].iloc[0]) == 7.0
    # the job fell back to the original counts (it says so in its record): umap_1/2
    monkeypatch.setattr(provenance, "_prov_recent", [{"kind": "umap", "headline": {"embedding_columns": "umap_1/2"}}])
    out = batch._batch_cell_table("clustering", {}, ctx)
    assert "umap_1" in out.columns and "split_umap_1" not in out.columns
    monkeypatch.setattr(provenance, "_prov_recent", [])
    out = batch._batch_cell_table("clustering", {}, {**ctx, "counts-mode-store": "original"})
    assert "umap_1" in out.columns and "split_umap_1" not in out.columns


# ── R8-30: a niches step with extra samples still writes its cells ───────────────────
def test_batch_niche_table_is_written_when_another_sample_has_no_niches(xe, D, monkeypatch, tmp_path):
    from xenium_app import batch, core
    df = D["df"].copy()
    df["niche"] = ["n1" if i % 2 else "n2" for i in range(len(df))]
    monkeypatch.setitem(core.DATA, "df", df)
    xe.EXTRA_DATASETS.append(_fab_ds(D, 20, 0, str(tmp_path / "n0")))          # no niche column
    ctx = {"seg-source": "xenium", "counts-mode-store": "original", "cluster-method": None, "color-by": "cluster"}
    out = batch._batch_cell_table("niches", {}, ctx)
    assert out is not None and len(out) == len(df) and set(out["sample"]) == {"S0"}
    assert list(out["niche"]) == list(df["niche"])
    # no sample has the column: still no table
    monkeypatch.setitem(core.DATA, "df", D["df"].drop(columns=["niche"], errors="ignore"))
    assert batch._batch_cell_table("niches", {}, ctx) is None


# ── R8-31: the contact statements name the design that was fitted ────────────────────
def test_contact_csv_header_states_the_fitted_design(xe):
    from xenium_app import proximity
    res = {"shuffle": {"n": 0}, "label_a": "a", "label_b": "b", "corrected": False, "anchor": "X", "target": "T",
           "radius": 15.0, "fdr": 0.05, "lfc_min": 0.5, "samples": [], "design": "~ donor + group",
           "units_note": "n = 3 donors, each with a contact and a matched comparator pseudobulk"}
    head = proximity._contact_csv_header(res)
    assert "design: ~ donor + group;" in head and "~ sample + group" not in head
    res["design"] = "~ sample + group"
    assert "design: ~ sample + group;" in proximity._contact_csv_header(res)
    # status line and title are written from the same field
    src = inspect.getsource(proximity)
    assert 'f"design {res[\'design\']}. ' in src and "design ~ sample + group." not in src
    assert "{res['design']}; n = {k} {unit}" in src


# ── R8-32: a dry run leaves the saved design alone ───────────────────────────────────
def test_batch_dry_run_does_not_replace_the_saved_design(xe, D, cfg, tmp_path):
    extras = [_fab_ds(D, 30, i, str(tmp_path / f"b{i}")) for i in range(3)]
    xe.EXTRA_DATASETS.extend(extras)
    xe._design_set([{"key": xe._ds_key(e), "cond": "AB"[i % 2], "batch": "run1", "donor": f"d{i}"}
                    for i, e in enumerate(extras)], {"A": "post", "B": "pre"})
    path = xe._design_path()
    saved = pathlib.Path(path).read_text()
    cfg["data"]["design"] = {"samples": [{"sample": "S1", "condition": "B"}]}
    out = xe._batch_run(cfg, dry_run=True, log=_quiet)
    assert out["dry_run"] and pathlib.Path(path).read_text() == saved, "the dry run rewrote the design file"
    # a real run does replace it (the config is the design of the run)
    cfg["figures"] = False
    xe._batch_run(cfg, log=_quiet)
    now = json.loads(pathlib.Path(path).read_text())
    assert now != json.loads(saved) and now["assign"][xe._ds_key(extras[0])]["cond"] == "B"


# ── follow-up to R8-18: the depth note of a pseudobulk result is over the shared genes ─
def test_pseudobulk_depth_note_uses_the_genes_the_used_samples_share(xe):
    """Source-level (the job needs R): the per-cell libraries behind "median counts per
    cell" are summed over the genes shared by the used samples, as the pseudobulk sums."""
    from xenium_app import stats
    src = inspect.getsource(stats._run_pseudobulk_de)
    assert "analysis._an_size_factors(smp, _lib_genes)" in src
    assert '_lib_genes = analysis._an_common_genes([s for s, r in zip(samples, prep["sample_rows"]) if r["used"]])' in src


# ── R8-11 through the job (R replaced by the stand-in of the round-6 tests) ──────────
from test_review_round6 import spage_world, _spage_cache          # noqa: E402,F401  (fixture)


def test_spage_cache_of_one_subset_is_not_served_to_another_of_the_same_size(xe, D, spage_world):
    from xenium_app import spage as spage_mod
    ids = list(D["df"].index)
    half = max(3, len(ids) // 2)
    try:
        xe.subset(cell_ids=ids[:half])
        tag = spage_mod._spage_ids_tag(D["df"].index)
        _spage_cache(xe, D, spage_world, 3.0, suffix=f"_n{half}_{tag}")
        xe._run_spage_imputation(spage_world, 50, "GNEW")
        assert xe._spage_state["status"] == "done" and "(cached)" in xe._spage_state["message"]
        # other cells, the same number: the cached rows of the first subset are not used
        xe.subset(cell_ids=ids[-half:])
        assert len(D["df"]) == half and spage_mod._spage_ids_tag(D["df"].index) != tag
        xe._run_spage_imputation(spage_world, 50, "GNEW")
        assert "(cached)" not in str(xe._spage_state["message"]), xe._spage_state["message"]
    finally:
        xe.unsubset()
        with xe._spage_lock:
            xe._spage_state.update({"status": "idle", "message": "", "result": None, "result_path": None,
                                    "result_genes": None})
