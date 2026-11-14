"""Round-10 regression and stress tests for session, batch, and provenance:
- R10-20: Batch validation tracks step-produced label sources (niches, annotation,
  cluster-creating steps banksy/joint/clustering, txfactor layers) and downgrades missing
  step-produced labels/clusters from hard errors to warnings while preserving true errors
  and step-order checks.
- R10-21 (provenance side): _prov_labels_source resolves loaded reseg run tags ("tool:tag")
  when the out_dir md5 suffix matches (or no hash suffix is present) without misattributing
  mismatched hashes, and _pm_variant includes labels_key for untagged annotation/spatial
  records so distinct runs never collide in _prov_methods_md(latest_only=True).
- R10-22: _session_apply_subset filters a newly loaded cached segmentation run even when
  core.DATA["_subset_desc"] already matches the session's subset filters.
- R10-23: UI layout includes one-shot session-init-tick wired into the session-request
  collector, and _session_autosave compares first-seen page state against server-side
  defaults so edits made before the first tick save immediately while untouched pages do
  not write a file.
- R10-24: provenance._prov_suppress() suppresses provenance writes (and _prov_once token
  consumption) across threads during _session_restore(..., persist=False) and
  _batch_run(..., dry_run=True).
- R10-25: _session_restore parses restored geneset-genes via analysis._gs_parse so comma,
  semicolon, and whitespace-separated gene lists matching a stored set keep geneset-select.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _isolated_round10_state(xe, tmp_path, monkeypatch):
    """Isolate CACHE_BASE_DIR and reset mutable session/batch/provenance/reseg state."""
    monkeypatch.setattr(xe, "CACHE_BASE_DIR", str(tmp_path))
    prov_before = list(xe._prov_recent)
    once_before = set(xe._prov_once)
    xe._an_autoload()
    xe._stats_reset()
    xe._controls_reset()
    xe._session_auto.update({"baseline": None, "written": None})
    xe._session_auto.pop("pages", None)
    yield
    if xe.DATA.get("_subset_desc") is not None:
        xe.unsubset()
    for tool in ("baysor", "proseg", "imgseg"):
        st, lk = xe._seg_state(tool)
        with lk:
            st.update({"status": "idle", "message": "", "result": None, "loading_out_dir": None})
    xe._prov_recent[:] = prov_before
    with xe._prov_lock:
        xe._prov_once.clear()
        xe._prov_once.update(once_before)
    monkeypatch.undo()
    xe._an_autoload()
    xe._stats_reset()
    xe._controls_reset()
    xe._session_auto.update({"baseline": None, "written": None})
    xe._session_auto.pop("pages", None)


def _defaults(xe) -> dict:
    return {cid: xe._session_default(cid) for cid in xe._session_controls()}


def _session_doc(xe, controls=None, **state) -> dict:
    doc = xe._session_snapshot(_defaults(xe), "test")
    if controls:
        doc["state"]["controls"].update(controls)
    for key, val in state.items():
        doc["state"][key] = val
    return doc


def _write_cached_baysor_run(xe, D, tag: str = "abcd1234", n_cells: int = 24) -> str:
    """Create a real loadable cached Baysor run directory under xe.CACHE_BASE_DIR."""
    out_dir = xe._seg_run_dir("baysor", tag)
    os.makedirs(out_dir, exist_ok=True)
    dkey = xe._dataset_key()
    dhash = dkey.rsplit("_", 1)[-1] if "_" in dkey else dkey
    with open(os.path.join(out_dir, "params.json"), "w") as fh:
        json.dump({"dataset_hash": dhash, "min_molecules_per_cell": 5}, fh)
    with open(os.path.join(out_dir, "run_id"), "w") as fh:
        fh.write(f"run_{tag}\n")

    rng = np.random.default_rng(42)
    genes = list(D["gene_names"][:12])
    rows = []
    for ci in range(n_cells):
        cx = float(50.0 + (ci % 6) * 30.0)
        cy = float(50.0 + (ci // 6) * 30.0)
        # Half the cells have 4 transcripts, half have 16 transcripts (for subset tests)
        n_tx = 4 if ci < (n_cells // 2) else 16
        for ti in range(n_tx):
            rows.append({
                "x": cx + float(rng.normal(0, 1.5)),
                "y": cy + float(rng.normal(0, 1.5)),
                "gene": genes[(ci + ti) % len(genes)],
                "cell": f"bcell_{ci}",
            })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "segmentation.csv"), index=False)
    return out_dir


# ── R10-20: Batch validation of step-produced niches & cluster sets ──────────
def test_r10_20_batch_validate_allows_step_produced_niche_and_clusters(xe, bundle_dir, tmp_path):
    base = {"data": {"path": str(bundle_dir)}, "output_dir": str(tmp_path / "out")}
    assert "niche" not in xe.DATA["df"].columns

    # 1. [niches, composition {labels: niche}] without existing niche column -> warning, 0 errors
    cfg_niche_ok = {
        **base,
        "steps": [
            {"kind": "niches"},
            {"kind": "composition", "params": {"labels": "niche"}},
        ],
    }
    res = xe._batch_run(cfg_niche_ok, dry_run=True, log=lambda _s: None)
    assert res["ok"] is True
    assert res["errors"] == []
    assert any("label source 'niche' is not available" in w and "earlier niches step" in w
               for w in res["warnings"])

    # 2. [composition {labels: niche}] WITHOUT an earlier niches step -> still a hard error
    cfg_niche_missing = {
        **base,
        "steps": [{"kind": "composition", "params": {"labels": "niche"}}],
    }
    res_err = xe._batch_run(cfg_niche_missing, dry_run=True, log=lambda _s: None)
    assert res_err["ok"] is False
    assert any("label source 'niche' is not available" in e for e in res_err["errors"])

    # 3. Wrong order [composition {labels: niche}, niches] -> step 1 still errors!
    cfg_niche_wrong_order = {
        **base,
        "steps": [
            {"kind": "composition", "params": {"labels": "niche"}},
            {"kind": "niches"},
        ],
    }
    res_order = xe._batch_run(cfg_niche_wrong_order, dry_run=True, log=lambda _s: None)
    assert res_order["ok"] is False
    assert any("step 1 (composition): label source 'niche' is not available" in e
               for e in res_order["errors"])


def test_r10_20_batch_validate_cluster_creating_steps_and_true_errors(xe, bundle_dir, tmp_path):
    base = {"data": {"path": str(bundle_dir)}, "output_dir": str(tmp_path / "out")}
    genes = list(xe.DATA["gene_names"][:2])

    # 1. Top-level cluster_method created by an earlier banksy / clustering step -> warning, 0 errors
    for creator in ("banksy", "clustering"):
        cfg_ok = {
            **base,
            "cluster_method": f"{creator}_future_domains",
            "steps": [
                {"kind": creator},
                {"kind": "composition", "params": {"labels": "cluster"}},
                {"kind": "de", "params": {"group_a": "cluster:0", "group_b": "rest"}},
                {"kind": "segfree", "params": {"labels": "cluster", "cell_type": "0",
                                               "markers": f"{genes[0]},{genes[1]}"}},
            ],
        }
        out = xe._batch_run(cfg_ok, dry_run=True, log=lambda _s: None)
        assert out["ok"] is True, f"{creator} failed with errors: {out['errors']}"
        assert out["errors"] == []
        assert any(f"earlier {creator} step may provide it" in w for w in out["warnings"])

    # 2. Top-level unknown cluster_method WITHOUT any cluster-creating step -> hard error
    cfg_no_creator = {
        **base,
        "cluster_method": "nonexistent_clusters",
        "steps": [{"kind": "moran", "params": {"k": 6}}],
    }
    out_bad = xe._batch_run(cfg_no_creator, dry_run=True, log=lambda _s: None)
    assert out_bad["ok"] is False
    assert any("cluster_method 'nonexistent_clusters' is not a cluster set" in e
               for e in out_bad["errors"])

    # 3. Wrong order: step 1 uses cluster BEFORE step 2 (banksy) creates cluster_method -> error on step 1
    cfg_wrong_order = {
        **base,
        "cluster_method": "banksy_future_domains",
        "steps": [
            {"kind": "composition", "params": {"labels": "cluster"}},
            {"kind": "banksy"},
        ],
    }
    out_order = xe._batch_run(cfg_wrong_order, dry_run=True, log=lambda _s: None)
    assert out_order["ok"] is False
    assert any("step 1 (composition): cluster_method 'banksy_future_domains'" in e
               for e in out_order["errors"])

    # 4. Bogus annotation method celltype:nothing stays an error even after an annotation step
    cfg_bogus_annot = {
        **base,
        "steps": [
            {"kind": "annotation"},
            {"kind": "composition", "params": {"labels": "celltype:nothing"}},
        ],
    }
    out_bogus = xe._batch_run(cfg_bogus_annot, dry_run=True, log=lambda _s: None)
    assert out_bogus["ok"] is False
    assert any("label source 'celltype:nothing' is not available" in e for e in out_bogus["errors"])
    assert not any("earlier annotation step" in e for e in out_bogus["errors"])


# ── R10-21: Provenance _prov_labels_source and _pm_variant ───────────────────
def test_r10_21_prov_labels_source_resolves_loaded_reseg_tag(xe):
    assert xe._prov_labels_source("labels") == "xenium"
    assert xe._prov_labels_source("labels_celltypist") == "xenium"
    assert xe._prov_labels_source(None) == "xenium"

    for tool, tag in (("baysor", "1111aaaa"), ("proseg", "2222bbbb"), ("imgseg", "3333cccc")):
        out_dir = xe._seg_run_dir(tool, tag)
        md5_8 = hashlib.md5(out_dir.encode()).hexdigest()[:8]
        other_md5 = "ffffffff" if md5_8 != "ffffffff" else "00000000"

        # Before loading: no loaded result -> bare tool name
        assert xe._prov_labels_source(f"labels_celltypist_{tool}_{md5_8}") == tool

        st, lk = xe._seg_state(tool)
        with lk:
            st.update({"status": "done", "result": {"source": tool, "out_dir": out_dir}})
        try:
            # Matching out_dir md5 suffix -> "<tool>:<tag>"
            assert xe._prov_labels_source(f"labels_celltypist_{tool}_{md5_8}") == f"{tool}:{tag}"
            # RCTD mode suffix after the md5 hash -> "<tool>:<tag>"
            assert xe._prov_labels_source(f"labels_rctd_{tool}_{md5_8}_doublet") == f"{tool}:{tag}"
            # Legacy key without an 8-hex hash suffix while run is loaded -> "<tool>:<tag>"
            assert xe._prov_labels_source(f"labels_{tool}") == f"{tool}:{tag}"
            assert xe._prov_labels_source(f"labels_rctd_{tool}") == f"{tool}:{tag}"
            assert xe._prov_labels_source(f"labels_rctd_{tool}_doublet") == f"{tool}:{tag}"
            # Mismatched 8-hex hash suffix (belongs to a DIFFERENT run) -> must NOT misattribute!
            assert xe._prov_labels_source(f"labels_celltypist_{tool}_{other_md5}") == tool
        finally:
            with lk:
                st.update({"status": "idle", "result": None})


def test_r10_21_pm_variant_preserves_distinct_untagged_annotation_and_spatial_runs(xe):
    # Two untagged Baysor annotation runs with distinct labels_key must NOT collide
    rec_a1 = {
        "id": "prov-1", "kind": "annotation", "cache": "computed",
        "time": "2026-11-14T01:00:00+00:00",
        "data": {"seg_source": "baysor", "seg_tool": "baysor", "seg_tag": "",
                 "counts": {"used": "original"}, "n_cells": 100, "n_samples": 1},
        "params": {"method": "celltypist", "model": "ModelA.pkl",
                   "labels_key": "labels_celltypist_baysor_11111111"},
        "headline": {"n_cells_labelled": 100, "n_cell_types": 4},
        "warnings": [], "seeds": {}, "software": {},
    }
    rec_a2 = {
        **rec_a1,
        "id": "prov-2",
        "time": "2026-11-14T01:05:00+00:00",
        "params": {"method": "celltypist", "model": "ModelB.pkl",
                   "labels_key": "labels_celltypist_baysor_22222222"},
    }
    assert xe._pm_variant(rec_a1) != xe._pm_variant(rec_a2)
    md = xe._prov_methods_md([rec_a1, rec_a2], latest_only=True)
    assert "ModelA.pkl" in md and "ModelB.pkl" in md

    # Same labels_key on Xenium still deduplicates to the latest run under latest_only=True
    rec_x1 = {
        **rec_a1,
        "id": "prov-3",
        "data": {"seg_source": "xenium", "seg_tool": "xenium", "seg_tag": "",
                 "counts": {"used": "original"}, "n_cells": 100, "n_samples": 1},
        "params": {"method": "celltypist", "model": "OldModel.pkl", "labels_key": "labels_celltypist"},
    }
    rec_x2 = {
        **rec_x1,
        "id": "prov-4",
        "time": "2026-11-14T01:10:00+00:00",
        "params": {"method": "celltypist", "model": "NewModel.pkl", "labels_key": "labels_celltypist"},
    }
    assert xe._pm_variant(rec_x1) == xe._pm_variant(rec_x2)
    md_x = xe._prov_methods_md([rec_x1, rec_x2], latest_only=True)
    assert "NewModel.pkl" in md_x and "OldModel.pkl" not in md_x

    # Spatial records without seg_tag but with distinct labels_key also stay distinct
    sp1 = {
        "id": "prov-5", "kind": "spatial", "cache": "computed",
        "time": "2026-11-14T02:00:00+00:00",
        "data": {"seg_source": "baysor", "seg_tool": "baysor", "seg_tag": None,
                 "counts": {"used": "original"}, "n_cells": 80, "n_samples": 1},
        "params": {"labels_key": "labels_celltypist_baysor_11111111", "n_perms": 111},
        "headline": {}, "warnings": [], "seeds": {}, "software": {},
    }
    sp2 = {
        **sp1,
        "id": "prov-6",
        "time": "2026-11-14T02:05:00+00:00",
        "params": {"labels_key": "labels_celltypist_baysor_22222222", "n_perms": 222},
    }
    assert xe._pm_variant(sp1) != xe._pm_variant(sp2)
    md_sp = xe._prov_methods_md([sp1, sp2], latest_only=True)
    assert "111" in md_sp and "222" in md_sp


# ── R10-22: _session_apply_subset filters newly loaded segmentation runs ─────
def test_r10_22_session_restore_filters_newly_loaded_reseg_when_xenium_subset_matches(xe, D):
    _write_cached_baysor_run(xe, D, tag="cafe1234", n_cells=24)

    # Activate subset on Xenium BEFORE Baysor is loaded
    xe.subset(min_transcripts=10)
    sub_now = xe._session_subset_now()
    assert sub_now is not None and sub_now["filters"] == {"min_transcripts": 10}
    assert xe._baysor_state["result"] is None

    # Restore a session naming baysor:cafe1234 with the SAME subset filter
    doc = _session_doc(xe, controls={"seg-source": "baysor:cafe1234"})
    assert doc["state"]["subset"]["filters"] == {"min_transcripts": 10}

    report = xe._session_restore(doc, _defaults(xe), "test", record=False, echo=False)
    assert report["ok"] is True
    bres = xe._baysor_state["result"]
    assert bres is not None
    assert "_cells_df_original" in bres, "newly loaded Baysor run was left unfiltered"
    assert len(bres["_cells_df_original"]) == 24
    assert len(bres["cells_df"]) == 12
    assert (bres["cells_df"]["transcript_counts"] >= 10).all()
    assert bres["expr"].shape[0] == 12
    assert any(a.startswith("subset:") for a in report["applied"])
    assert "subset" not in report["unchanged"]

    # Restoring a second time when Baysor IS already filtered reports "unchanged"
    report2 = xe._session_restore(doc, report["controls"], "test", record=False, echo=False)
    assert "subset" in report2["unchanged"]


# ── R10-23: Autosave page-load baseline & pre-tick edit capture ──────────────
def test_r10_23_init_tick_wired_and_pre_tick_edits_autosaved_immediately(xe):
    from xenium_app import app_layout, cb_session

    # (a) Verify one-shot session-init-tick interval exists in layout and feeds collector
    comps = {}

    def _walk(node):
        cid = getattr(node, "id", None)
        if isinstance(cid, str):
            comps[cid] = node
        kids = getattr(node, "children", None)
        if isinstance(kids, (list, tuple)):
            for k in kids:
                _walk(k)
        elif kids is not None and hasattr(kids, "to_plotly_json"):
            _walk(kids)

    _walk(app_layout.app.layout)
    assert "session-init-tick" in comps
    tick = comps["session-init-tick"]
    assert getattr(tick, "max_intervals", None) == 1
    assert 0 < getattr(tick, "interval", 0) <= 2000
    assert cb_session._SESSION_TRIGGERS.get("session-init-tick") == "auto"

    collector_entries = [
        cb for key, cb in app_layout.app.callback_map.items()
        if "session-request.data" in key
    ]
    assert len(collector_entries) == 1
    collector_inputs = {(i["id"], i["property"]) for i in collector_entries[0]["inputs"]}
    assert ("session-init-tick", "n_intervals") in collector_inputs

    # (b1) Existing contract: untouched default controls on first call return "" & write nothing
    defs = _defaults(xe)
    assert xe._session_autosave(defs) == ""
    assert not os.path.exists(xe._session_path("last"))

    # (b2) Reset autosave state: simulate user editing controls BEFORE the first tick arrives
    xe._session_auto.update({"baseline": None, "written": None})
    xe._session_auto.pop("pages", None)
    edited_early = {**defs, "moran-k": 11}
    assert xe._session_autosave(edited_early, page="fast-page") == "saved"
    assert os.path.exists(xe._session_path("last"))
    with open(xe._session_path("last")) as fh:
        saved_doc = json.load(fh)
    assert saved_doc["state"]["controls"]["moran-k"] == 11

    # Repeating same state writes nothing; reverting back to defaults DOES save (baseline was defaults)
    assert xe._session_autosave(edited_early, page="fast-page") == ""
    assert xe._session_autosave(defs, page="fast-page") == "saved"
    with open(xe._session_path("last")) as fh:
        reverted_doc = json.load(fh)
    assert reverted_doc["state"]["controls"]["moran-k"] == defs["moran-k"]

    # (b3) Stress test: > _SESSION_AUTO_PAGES unopened pages do not clobber last slot,
    # while a new page with immediate pre-tick edits still saves!
    assert xe._session_autosave({**defs, "moran-k": 13}, page="fast-page") == "saved"
    for i in range(xe._SESSION_AUTO_PAGES + 8):
        assert xe._session_autosave(defs, page=f"idle-{i}") == ""
    with open(xe._session_path("last")) as fh:
        assert json.load(fh)["state"]["controls"]["moran-k"] == 13
    assert xe._session_autosave({**defs, "moran-k": 15}, page="brand-new-edited") == "saved"
    with open(xe._session_path("last")) as fh:
        assert json.load(fh)["state"]["controls"]["moran-k"] == 15


# ── R10-24: Dry-run / persist=False suppresses provenance across threads ─────
def test_r10_24_prov_suppress_reentrant_and_thread_safe(xe):
    prov_dir = os.path.join(xe.CACHE_BASE_DIR, "provenance")
    n_before = len(xe._prov_recent)

    errors = []

    def _worker():
        try:
            r = xe._prov_record("moran", "Suppressed worker", params={"k": 6},
                                once=("r10_24_token",))
            if r is not None:
                errors.append("worker wrote a record while suppressed")
        except Exception as exc:
            errors.append(str(exc))

    with xe._prov_suppress():
        with xe._prov_suppress():
            assert xe._prov_record("moran", "Nested suppressed", params={"k": 6},
                                   once=("r10_24_token",)) is None
            t = threading.Thread(target=_worker)
            t.start()
            t.join()
        # Outer context still active!
        assert xe._prov_record("moran", "Still suppressed", params={"k": 6},
                               once=("r10_24_token",)) is None

    assert not errors
    assert len(xe._prov_recent) == n_before
    assert not os.path.exists(prov_dir) or os.listdir(prov_dir) == []

    # After exiting _prov_suppress, the once-token was NOT burned: real call now records!
    rec = xe._prov_record("moran", "Unsuppressed", params={"k": 6}, once=("r10_24_token",))
    assert rec is not None
    assert len(xe._prov_recent) == n_before + 1


def test_r10_24_persist_false_and_batch_dry_run_write_zero_provenance(xe, D, bundle_dir, tmp_path):
    _write_cached_baysor_run(xe, D, tag="beef5678", n_cells=20)
    prov_dir = os.path.join(xe.CACHE_BASE_DIR, "provenance")
    n_before = len(xe._prov_recent)

    # 1. _session_restore(..., persist=False) loading cached baysor run writes zero provenance
    doc = _session_doc(xe, controls={"seg-source": "baysor:beef5678"})
    rep = xe._session_restore(doc, _defaults(xe), "dry", record=True, echo=False, persist=False)
    assert rep["ok"] is True
    assert xe._baysor_state["status"] == "done"
    assert len(xe._prov_recent) == n_before
    assert not os.path.exists(prov_dir) or os.listdir(prov_dir) == []

    # Reset loaded state so batch --dry-run also loads baysor:beef5678 from disk
    with xe._baysor_lock:
        xe._baysor_state.update({"status": "idle", "message": "", "result": None})

    # 2. _batch_run(..., dry_run=True) loading cached baysor run writes zero provenance
    session_file = tmp_path / "sess.json"
    session_file.write_text(json.dumps(doc))
    cfg = {
        "data": {"path": str(bundle_dir)},
        "output_dir": str(tmp_path / "batch_out"),
        "session": str(session_file),
        "steps": [{"kind": "moran", "params": {"k": 6}}],
    }
    out = xe._batch_run(cfg, dry_run=True, log=lambda _s: None)
    assert out["ok"] is True
    assert len(xe._prov_recent) == n_before
    assert not os.path.exists(prov_dir) or os.listdir(prov_dir) == []

    # 3. Subsequent persist=True load in the same process STILL records provenance (once token intact)
    with xe._baysor_lock:
        xe._baysor_state.update({"status": "idle", "message": "", "result": None})
    rep_real = xe._session_restore(doc, _defaults(xe), "real", record=True, echo=False, persist=True)
    assert rep_real["ok"] is True
    kinds = [r["kind"] for r in xe._prov_recent[n_before:]]
    assert "baysor" in kinds and "session" in kinds


# ── R10-25: Comma / semicolon / space-separated gene sets on restore ─────────
@pytest.mark.parametrize("sep_fmt", [
    lambda gs: ", ".join(gs),
    lambda gs: "; ".join(gs),
    lambda gs: "   ".join(gs),
    lambda gs: f" {gs[0]}, {gs[1]};\n{gs[2]}  {gs[0]} ",  # mixed delimiters + duplicate
])
def test_r10_25_restored_delimited_geneset_not_flagged_as_edited(xe, D, sep_fmt):
    stored = [str(g) for g in D["gene_names"][:3]]
    text = sep_fmt(stored)
    doc = _session_doc(
        xe,
        controls={"geneset-select": "cardiac", "geneset-genes": text},
        gene_sets={"cardiac": stored},
    )
    rep = xe._session_restore(doc, _defaults(xe), "test", record=False, echo=False)
    assert rep["ok"] is True
    assert rep["controls"].get("geneset-select") == "cardiac"
    assert not any("gene-set picker was left empty" in s for s in rep["skipped"])


def test_r10_25_truly_edited_delimited_geneset_still_clears_picker(xe, D):
    stored = [str(g) for g in D["gene_names"][:3]]
    edited_text = f"{stored[0]}, {stored[1]}, {D['gene_names'][7]}"
    doc = _session_doc(
        xe,
        controls={"geneset-select": "cardiac", "geneset-genes": edited_text},
        gene_sets={"cardiac": stored},
    )
    rep = xe._session_restore(doc, _defaults(xe), "test", record=False, echo=False)
    assert rep["ok"] is True
    assert rep["controls"].get("geneset-genes") == edited_text
    assert rep["controls"].get("geneset-select") is None
    assert any("gene-set picker was left empty" in s for s in rep["skipped"])
