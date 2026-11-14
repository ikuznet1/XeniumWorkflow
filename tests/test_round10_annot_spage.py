"""Round-10 regression and adversarial stress tests for annotation, spatial, CellNEST,
SpaGE, and WebGL viewer glue (R10-9, R10-10, R10-11, R10-12, R10-13, R10-19, R10-21).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import types

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import scipy.io as sio
import scipy.sparse as sp

import xe_testlib as tl


@pytest.fixture(autouse=True)
def _preserve_prov_and_cellnest(xe):
    """Keep session-wide _prov_recent and _cellnest_state clean across every test."""
    prov_before = list(xe._prov_recent)
    with xe._cellnest_lock:
        cn_before = dict(xe._cellnest_state)
    yield
    xe._prov_recent[:] = prov_before
    with xe._cellnest_lock:
        xe._cellnest_state.clear()
        xe._cellnest_state.update(cn_before)


def _make_reseg_run(xe, tmp_path, tool="proseg", tag="a1b20001", n_cells=120):
    """Build a synthetic cached resegmentation run directory and in-memory result dict."""
    assert re.fullmatch(r"[0-9a-f]{8}", tag), f"tag must be 8 lowercase hex chars, got {tag!r}"
    ds_name = xe._dataset_name()
    ds_hash = hashlib.md5(xe._dataset_key().encode()).hexdigest()[:8]
    run_dir = tmp_path / f"{tool}_{ds_name}_{ds_hash}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if tool == "proseg":
        (run_dir / "cell-metadata.csv.gz").write_text("ok\n")
    elif tool == "baysor":
        (run_dir / "segmentation.csv").write_text("ok\n")
    elif tool == "imgseg":
        (run_dir / "spatialdata_imgseg.zarr").mkdir(exist_ok=True)
    else:
        raise ValueError(f"unknown tool: {tool}")
    (run_dir / "params.json").write_text(
        json.dumps({"data_dir": os.path.realpath(xe.DATA["data_dir"]), "dataset_hash": ds_hash})
    )

    rng = np.random.default_rng(42)
    cell_ids = [f"{tool}_c{i}" for i in range(n_cells)]
    cells_df = pd.DataFrame(
        {
            "centroid_x": rng.uniform(0.0, 500.0, size=n_cells),
            "centroid_y": rng.uniform(0.0, 500.0, size=n_cells),
            "transcript_counts": rng.integers(20, 200, size=n_cells),
            f"cluster_{tool}": [f"cl_{i % 3}" for i in range(n_cells)],
        },
        index=pd.Index(cell_ids, dtype=object),
    )
    n_genes = min(12, len(xe.DATA["gene_names"]))
    expr = sp.csr_matrix(rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32))
    res = {
        "out_dir": str(run_dir),
        "source": tool,
        "cells_df": cells_df,
        "expr": expr,
    }
    return run_dir, f"{tool}:{tag}", res


# ── R10-9: Spatial-analysis job registers uses_run & commit hook clears stale result ──

def test_r10_9_spatial_analysis_registers_uses_run_and_blocks_delete_or_replace(xe, tmp_path, monkeypatch):
    """While run_spatial_analysis is running on a resegmentation run, deleting or
    replacing that run via confirm_seg_delete or _seg_stage_commit is refused; once
    finished, committing a replacement run clears stale _spatial_state."""
    from xenium_app import caching, core, jobs, spatial

    monkeypatch.setattr(xe, "CACHE_BASE_DIR", str(tmp_path))
    run_dir, seg_val, res = _make_reseg_run(xe, tmp_path, tool="proseg", tag="a1b20001", n_cells=30)
    with core._proseg_lock:
        core._proseg_state.update({"status": "done", "message": "ok", "result": res, "loading_out_dir": None})

    lkey = xe._labels_key_for_method("celltypist", res)
    lab = pd.Series(["T", "B"] * 15, index=res["cells_df"].index.astype(str))
    with core._annot_lock:
        xe._annot_state[lkey] = lab

    entered = threading.Event()
    release = threading.Event()
    real_run = spatial._run_spatial_analysis

    def _blocking_spatial(*args, **kwargs):
        entered.set()
        release.wait(timeout=10.0)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(spatial, "_run_spatial_analysis", _blocking_spatial)

    out = xe.run_spatial_analysis(1, "labels_celltypist", 5, seg_val)
    assert out == (True, False, "Running…")
    assert entered.wait(timeout=5.0), "spatial job never started"

    try:
        # 1) _run_in_use reports the spatial job
        busy_msg = jobs._run_in_use(str(run_dir), "deleted")
        assert busy_msg and "Spatial analysis" in busy_msg

        # 2) confirm_seg_delete refuses to delete the active run directory
        del_out = xe.confirm_seg_delete(1, seg_val, 0, 0)
        assert run_dir.is_dir(), "run_dir was deleted while spatial analysis was using it!"
        assert del_out[0] is True and "Spatial analysis" in str(del_out[-1])

        # 3) _seg_stage_commit refuses to overwrite the directory mid-job
        stage = caching._seg_stage_dir(str(run_dir))
        try:
            with pytest.raises(RuntimeError, match="Spatial analysis"):
                caching._seg_stage_commit(stage, str(run_dir))
        finally:
            caching._seg_stage_discard(stage)
    finally:
        release.set()
        tl.wait_state(xe._spatial_state, xe._spatial_lock)

    assert xe._spatial_state["status"] == "done"
    assert xe._spatial_state["result"]["out_dir"] == str(run_dir)

    # Once finished, committing a replacement run clears stale _spatial_state for THAT run
    # (and leaves another run's _spatial_state untouched).
    other_dir = tmp_path / "proseg_output_other"
    other_dir.mkdir()
    other_stage = caching._seg_stage_dir(str(other_dir))
    caching._seg_stage_commit(other_stage, str(other_dir))
    assert xe._spatial_state["status"] == "done" and xe._spatial_state["result"] is not None

    same_stage = caching._seg_stage_dir(str(run_dir))
    caching._seg_stage_commit(same_stage, str(run_dir))
    assert xe._spatial_state["status"] == "idle" and xe._spatial_state["result"] is None


def test_r10_9_spatial_analysis_cancellation_propagates_and_repl_spage_uses_run(xe, tmp_path, monkeypatch):
    """Cancelling _run_spatial_analysis raises _JobCancelled (not swallowed into 'error'),
    and spage.run_spage passes uses_run for an active resegmentation run."""
    from xenium_app import core, jobs, spage, spatial

    monkeypatch.setattr(xe, "CACHE_BASE_DIR", str(tmp_path))
    _run_dir, _seg_val, res = _make_reseg_run(xe, tmp_path, tool="proseg", tag="a1b20002", n_cells=20)
    lkey = xe._labels_key_for_method("celltypist", res)
    with core._annot_lock:
        xe._annot_state[lkey] = pd.Series(["A", "B"] * 10, index=res["cells_df"].index.astype(str))

    monkeypatch.setattr(jobs, "_job_step", lambda *_a, **_k: (_ for _ in ()).throw(jobs._JobCancelled("stop")))
    with pytest.raises(jobs._JobCancelled):
        spatial._run_spatial_analysis(lkey, 4, res)

    # Check REPL run_spage passes uses_run=_alt_res when a reseg run is active
    monkeypatch.undo()
    monkeypatch.setattr(xe, "CACHE_BASE_DIR", str(tmp_path))
    fake_rds = tmp_path / "ref.rds"
    fake_rds.write_bytes(b"RDS")
    with core._proseg_lock:
        core._proseg_state.update({"status": "done", "message": "ok", "result": res})
    captured = {}

    def _capture_job_start(kind, label, fn, **kw):
        captured["kind"] = kind
        captured["uses_run"] = kw.get("uses_run")
        with core._spage_lock:
            core._spage_state.update({"status": "idle", "message": "", "result": None})

    monkeypatch.setattr(jobs, "_job_start", _capture_job_start)
    spage.run_spage(str(fake_rds), n_pv=10)
    assert captured["kind"] == "spage"
    assert captured["uses_run"] is res


# ── R10-10: _reseg_cell_type_labels preserves valid <5% subset labels on unsubset ────

def test_r10_10_small_subset_reseg_labels_survive_unsubset_and_disjoint_views(xe, tmp_path):
    """Labels computed on a <5% subset of a resegmentation run must NOT be popped as stale
    when unsubset() restores the full table or when viewing a disjoint subset; only labels
    whose own IDs have <5% membership in the run's full cell IDs are popped as stale."""
    from xenium_app import annotation, core

    _run_dir, _seg_val, res = _make_reseg_run(xe, tmp_path, tool="baysor", tag="b1c20003", n_cells=200)
    full_df = res["cells_df"]
    lkey = annotation._labels_key_for_method("rctd", res)

    # Simulate subset() to 3 cells (1.5% < 5% of the 200-cell run) and annotating those 3 cells
    subset_ids = list(full_df.index[:3])
    disjoint_ids = list(full_df.index[50:65])
    res["_cells_df_original"] = full_df
    res["cells_df"] = full_df.loc[subset_ids].copy()

    small_labels = pd.Series(["Enterocyte", "Goblet", "Stem"], index=pd.Index(subset_ids, dtype=object))
    with core._annot_lock:
        core._annot_state[lkey] = small_labels

    # 1) On the 3-cell subset itself: exact 3 labels returned
    aligned_sub = annotation._reseg_cell_type_labels("rctd", res, subset_ids)
    assert aligned_sub is not None and list(aligned_sub) == ["Enterocyte", "Goblet", "Stem"]
    assert core._annot_state.get(lkey) is small_labels

    # 2) Simulate unsubset(): full 200 cells displayed -> 197/200 (98.5%) are "Unknown",
    #    yet 3/3 (100%) of small_labels.index belong to the run, so labels MUST stay!
    res.pop("_cells_df_original", None)
    res["cells_df"] = full_df
    aligned_full = annotation._reseg_cell_type_labels("rctd", res, list(full_df.index))
    assert aligned_full is not None
    assert len(aligned_full) == 200
    assert list(aligned_full.loc[subset_ids]) == ["Enterocyte", "Goblet", "Stem"]
    assert (aligned_full.iloc[3:] == "Unknown").all()
    assert core._annot_state.get(lkey) is small_labels, "valid <5% subset labels were deleted!"

    # 3) Viewing a disjoint 15-cell subset (0 displayed cells labelled, while _cells_df_original
    #    still holds all 200 cells): returns None for display fallback WITHOUT popping _annot_state!
    res["_cells_df_original"] = full_df
    res["cells_df"] = full_df.loc[disjoint_ids].copy()
    aligned_disjoint = annotation._reseg_cell_type_labels("rctd", res, disjoint_ids)
    assert aligned_disjoint is None
    assert core._annot_state.get(lkey) is small_labels, "disjoint subset view deleted valid labels!"

    # 4) Boundary stress test: 5% membership (5 of 100 label IDs in run) is KEPT;
    #    4% membership (4 of 100 label IDs in run) is POPPED as stale.
    res.pop("_cells_df_original", None)
    res["cells_df"] = full_df
    borderline_ids = list(full_df.index[:5]) + [f"foreign_{i}" for i in range(95)]
    borderline = pd.Series(["T"] * 100, index=pd.Index(borderline_ids, dtype=object))
    with core._annot_lock:
        core._annot_state[lkey] = borderline
    got_5pct = annotation._reseg_cell_type_labels("rctd", res, list(full_df.index))
    assert got_5pct is not None and (got_5pct.iloc[:5] == "T").all()
    assert core._annot_state.get(lkey) is borderline

    stale_ids = list(full_df.index[:4]) + [f"foreign_{i}" for i in range(96)]
    stale = pd.Series(["T"] * 100, index=pd.Index(stale_ids, dtype=object))
    with core._annot_lock:
        core._annot_state[lkey] = stale
    got_4pct = annotation._reseg_cell_type_labels("rctd", res, list(full_df.index))
    assert got_4pct is None
    assert lkey not in core._annot_state, "stale (<5% run-membership) labels were not cleared"


# ── R10-11: _load_cellnest_data candidate selection, subset originals & NaN skip ──────

def _write_cellnest_fixture(root, dataset_name, barcodes, edges):
    """Write minimal CellNEST CSV files under *root* for *dataset_name*."""
    out_dir = root / "output" / dataset_name
    meta_dir = root / "metadata" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(barcodes).to_csv(meta_dir / f"cell_barcode_{dataset_name}.csv",
                                  index=False, header=False)
    coords = np.column_stack([np.arange(len(barcodes), dtype=float) * 10.0,
                              np.arange(len(barcodes), dtype=float) * 5.0])
    pd.DataFrame(coords).to_csv(meta_dir / f"coordinates_{dataset_name}.csv",
                                index=False, header=False)

    ccc_rows = [
        {"from_id": u, "to_id": v, "ligand": lig, "receptor": rec, "attention_score": score}
        for (u, v, lig, rec, score) in edges
    ]
    pd.DataFrame(ccc_rows).to_csv(out_dir / f"CellNEST_{dataset_name}_top20percent.csv", index=False)
    pd.DataFrame({"Ligand-Receptor": ["L1-R1"], "Frequency": [len(edges)]}).to_csv(
        out_dir / f"CellNEST_{dataset_name}_histogram_byFrequency_table_top1500.csv", index=False
    )


def test_r10_11_cellnest_matches_imgseg_subset_original_ignores_nondone_and_skips_all_nan(
    xe, tmp_path, monkeypatch, capsys
):
    """_load_cellnest_data:
    - ignores non-'done' proseg/baysor states (e.g. 'error'/'running' with stale result),
    - includes imgseg (status == 'done') and uses _cells_df_original when subsetted,
    - handles strip_suffix properly and picks candidate with highest endpoint match (>0),
    - skips & logs columns from seg_df and _annot_state whose mapped edge values are all NaN."""
    from xenium_app import cellnest, core

    monkeypatch.setattr(xe, "CACHE_BASE_DIR", str(tmp_path))

    # Proseg has stale/errored result whose IDs happen to overlap if status wasn't checked
    proseg_df = pd.DataFrame(
        {"cluster_wrong_proseg": ["bad0", "bad1", "bad2"]},
        index=pd.Index(["iseg_0", "iseg_1", "iseg_2"], dtype=object),
    )
    with core._proseg_lock:
        core._proseg_state.update({
            "status": "error",
            "message": "failed",
            "result": {"out_dir": str(tmp_path / "p_err"), "source": "proseg", "cells_df": proseg_df},
        })

    # Baysor IS done, but only matches 1 of 4 cells (lower match fraction than imgseg's 100%)
    baysor_df = pd.DataFrame(
        {"cluster_baysor": ["b_only"]},
        index=pd.Index(["iseg_0"], dtype=object),
    )
    with core._baysor_lock:
        core._baysor_state.update({
            "status": "done",
            "message": "ok",
            "result": {"out_dir": str(tmp_path / "b_ok"), "source": "baysor", "cells_df": baysor_df},
        })

    # Imgseg IS done and currently subsetted to just ["iseg_3"], while _cells_df_original
    # holds all ["iseg_0", "iseg_1", "iseg_2", "iseg_3"], plus an all-NaN column!
    imgseg_full = pd.DataFrame(
        {
            "cluster_imgseg": ["C0", "C1", "C2", "C3"],
            "cell_type_allnan": [np.nan, np.nan, np.nan, np.nan],
        },
        index=pd.Index(["iseg_0", "iseg_1", "iseg_2", "iseg_3"], dtype=object),
    )
    imgseg_sub = imgseg_full.loc[["iseg_3"]].copy()
    with core._imgseg_lock:
        core._imgseg_state.update({
            "status": "done",
            "message": "ok",
            "result": {
                "out_dir": str(tmp_path / "i_ok"),
                "source": "imgseg",
                "cells_df": imgseg_sub,
                "_cells_df_original": imgseg_full,
            },
        })

    # _annot_state has both Xenium labels (0 matches -> all NaN on edges) and imgseg labels
    with core._annot_lock:
        core._annot_state["labels_xen_unmatched"] = pd.Series(
            ["XenA", "XenB"], index=pd.Index(["xen_998", "xen_999"], dtype=object)
        )
        core._annot_state["labels_celltypist_imgseg_12345678"] = pd.Series(
            ["Mac", "Fib", "Epi"], index=pd.Index(["iseg_0", "iseg_1", "iseg_2"], dtype=object)
        )

    cn_root = tmp_path / "cn"
    _write_cellnest_fixture(
        cn_root,
        "ds_img",
        barcodes=["iseg_0_4", "iseg_1_4", "iseg_2_4", "iseg_3_4"],
        edges=[
            (0, 1, "L1", "R1", 0.9),
            (1, 2, "L1", "R1", 0.8),
        ],
    )

    cellnest._load_cellnest_data(str(cn_root), "ds_img", strip_suffix="_4", seg_source="imgseg:tag1")
    out_log = capsys.readouterr().out
    assert core._cellnest_state["status"] == "done"
    data = core._cellnest_state["data"]
    ccc = data["ccc_df"]
    annot_cols = data["annot_cols"]

    # Must have chosen imgseg via _cells_df_original (even though cells_df only had iseg_3)
    assert "cluster_imgseg" in annot_cols
    assert list(ccc["from_cluster_imgseg"]) == ["C0", "C1"]
    assert list(ccc["to_cluster_imgseg"]) == ["C1", "C2"]

    # Must NOT have joined errored proseg or lower-matching baysor
    assert "cluster_wrong_proseg" not in annot_cols
    assert "cluster_baysor" not in annot_cols

    # All-NaN columns from seg_df and _annot_state must be skipped and logged
    assert "cell_type_allnan" not in annot_cols
    assert "from_cell_type_allnan" not in ccc.columns
    assert "celltype_xen_unmatched" not in annot_cols
    assert "from_celltype_xen_unmatched" not in ccc.columns
    assert "Skipping" in out_log

    # Matching _annot_state series IS joined
    expected_annot_col = "celltype_celltypist_imgseg_12345678"
    assert expected_annot_col in annot_cols
    assert list(ccc[f"from_{expected_annot_col}"]) == ["Mac", "Fib"]


def test_r10_11_cellnest_zero_match_joins_nothing_and_active_seg_breaks_ties(xe, tmp_path, monkeypatch):
    """When 0 endpoints match any table, no cluster columns are joined; when two tables tie
    at 100% endpoint match, the active seg_source wins; Xenium _df_original is used under subset()."""
    from xenium_app import cellnest, core

    monkeypatch.setattr(xe, "CACHE_BASE_DIR", str(tmp_path))

    # Case A: 0 matches against every candidate -> annot_cols must be empty
    cn_zero = tmp_path / "cn_zero"
    _write_cellnest_fixture(
        cn_zero,
        "ds_none",
        barcodes=["totally_unknown_a", "totally_unknown_b"],
        edges=[(0, 1, "L1", "R1", 0.5)],
    )
    cellnest._load_cellnest_data(str(cn_zero), "ds_none", strip_suffix="")
    assert core._cellnest_state["status"] == "done"
    assert core._cellnest_state["data"]["annot_cols"] == []

    # Case B: Tie between proseg (default earlier in fallback) and baysor (active seg_source)
    shared_ids = pd.Index(["tie_0", "tie_1"], dtype=object)
    tie_expr = sp.csr_matrix((2, len(core.DATA["gene_names"])), dtype=np.float32)
    with core._proseg_lock:
        core._proseg_state.update({
            "status": "done", "message": "ok",
            "result": {"out_dir": str(tmp_path / "p"), "source": "proseg",
                       "expr": tie_expr.copy(),
                       "cells_df": pd.DataFrame({"cluster_p": ["P0", "P1"]}, index=shared_ids)},
        })
    with core._baysor_lock:
        core._baysor_state.update({
            "status": "done", "message": "ok",
            "result": {"out_dir": str(tmp_path / "b"), "source": "baysor",
                       "expr": tie_expr.copy(),
                       "cells_df": pd.DataFrame({"cluster_b": ["B0", "B1"]}, index=shared_ids)},
        })
    cn_tie = tmp_path / "cn_tie"
    _write_cellnest_fixture(cn_tie, "ds_tie", barcodes=["tie_0", "tie_1"], edges=[(0, 1, "L1", "R1", 0.7)])
    cellnest._load_cellnest_data(str(cn_tie), "ds_tie", strip_suffix="", seg_source="baysor:active")
    assert core._cellnest_state["data"]["annot_cols"] == ["cluster_b"]

    # Case C: Xenium subset() active -> _df_original cells outside subset still match CellNEST edges
    all_xen_ids = [str(x) for x in core.DATA["df"].index]
    assert len(all_xen_ids) >= 4
    xe.subset(cell_ids=all_xen_ids[:2])
    cn_xen = tmp_path / "cn_xen"
    _write_cellnest_fixture(
        cn_xen, "ds_xen",
        barcodes=[all_xen_ids[2], all_xen_ids[3]],
        edges=[(0, 1, "L1", "R1", 0.6)],
    )
    cellnest._load_cellnest_data(str(cn_xen), "ds_xen", strip_suffix="", seg_source="xenium")
    assert core._cellnest_state["status"] == "done"
    if any(c.startswith("cluster") for c in core.DATA["_df_original"].columns):
        assert len(core._cellnest_state["data"]["annot_cols"]) >= 1


# ── R10-12: _spage_parquet_genes strips pandas index columns everywhere ──────────────

def test_r10_12_spage_parquet_genes_strips_unnamed_named_and_multiindex_columns(xe, tmp_path, monkeypatch):
    """_spage_parquet_genes excludes __index_level_0__, named single/multi-index columns,
    keeps RangeIndex parquets intact, and prevents off-by-one streaming threshold & fake
    '__index_level_0__' genes in both _spage_autoload and _run_spage_imputation."""
    from xenium_app import core, spage

    # 1) Unnamed string index -> pyarrow writes '__index_level_0__'
    p_unnamed = tmp_path / "unnamed.parquet"
    df_unnamed = pd.DataFrame(
        {"GATA6": [1.0, 2.0, 3.0], "EPCAM": [0.5, 1.5, 2.5]},
        index=pd.Index(["c0", "c1", "c2"], dtype=object),
    )
    df_unnamed.to_parquet(p_unnamed)
    pf_unnamed = pq.ParquetFile(p_unnamed)
    assert "__index_level_0__" in pf_unnamed.schema_arrow.names
    assert spage._spage_parquet_genes(pf_unnamed) == ["GATA6", "EPCAM"]

    # 2) Named index ('cell_id') -> listed in pandas_metadata['index_columns'] as string
    p_named = tmp_path / "named.parquet"
    df_named = df_unnamed.copy()
    df_named.index.name = "cell_id"
    df_named.to_parquet(p_named)
    pf_named = pq.ParquetFile(p_named)
    assert "cell_id" in pf_named.schema_arrow.names
    assert spage._spage_parquet_genes(pf_named) == ["GATA6", "EPCAM"]

    # 3) MultiIndex (one named, one unnamed -> 'sample' + '__index_level_1__')
    p_multi = tmp_path / "multi.parquet"
    df_multi = pd.DataFrame(
        {"SOX9": [1.0, 2.0]},
        index=pd.MultiIndex.from_tuples([("s1", "c0"), ("s1", "c1")], names=["sample", None]),
    )
    df_multi.to_parquet(p_multi)
    assert spage._spage_parquet_genes(pq.ParquetFile(p_multi)) == ["SOX9"]

    # 4) RangeIndex (index_columns contains a dict {"kind": "range", ...}, not a string)
    p_range = tmp_path / "range.parquet"
    pd.DataFrame({"MUC2": [1.0, 2.0], "LGR5": [3.0, 4.0]}).to_parquet(p_range)
    assert spage._spage_parquet_genes(pq.ParquetFile(p_range)) == ["MUC2", "LGR5"]

    # 5) Threshold boundary in _spage_autoload:
    #    3 rows * 2 real genes == 6 <= threshold 6 -> must load in-memory (NOT stream!).
    monkeypatch.setattr(core, "CACHE_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(core, "SPAGE_STREAM_THRESHOLD", 6)
    spage._spage_index_update(str(p_unnamed))
    with core._spage_lock:
        core._spage_state.update({"status": "idle", "message": "", "result": None,
                                  "result_path": None, "result_genes": []})
    spage._spage_autoload()
    assert core._spage_state["result"] is not None, "3x2 parquet wrongly streamed due to __index_level_0__!"
    assert list(core._spage_state["result"].columns) == ["GATA6", "EPCAM"]
    assert "__index_level_0__" not in core._spage_state.get("result_genes", [])

    # Lower threshold to 5 so 3 * 2 == 6 > 5 DOES stream, and verify result_genes excludes index!
    monkeypatch.setattr(core, "SPAGE_STREAM_THRESHOLD", 5)
    with core._spage_lock:
        core._spage_state.update({"status": "idle", "message": "", "result": None,
                                  "result_path": None, "result_genes": []})
    spage._spage_autoload()
    assert core._spage_state["result"] is None
    assert core._spage_state["result_path"] == str(p_unnamed)
    assert core._spage_state["result_genes"] == ["GATA6", "EPCAM"]


# ── R10-13: SpaGE R extraction never interpolates gene names or paths into R source ──

class _RecordingRpy2:
    """Stand-in rpy2 environment that forbids string-interpolated gene names/paths in ro.r
    and executes the MatrixMarket export using values bound in ro.globalenv."""

    def __init__(self, ref_genes, ref_cells, ref_counts, fail_on_extract=False):
        self.ref_genes = list(ref_genes)
        self.ref_cells = list(ref_cells)
        self.ref_counts = np.asarray(ref_counts, dtype=np.float32)  # genes x cells
        self.fail_on_extract = fail_on_extract
        self.globalenv = {}
        self.r_scripts = []
        self.rlock = threading.RLock()

    def StrVector(self, seq):
        return [str(x) for x in seq]

    def r(self, code: str):
        self.r_scripts.append(code)
        if 'rna_genes  <- rownames(rna_mat_data)' in code:
            self.globalenv["rna_ref"] = True
            self.globalenv["rna_mat_data"] = True
            self.globalenv["rna_genes"] = list(self.ref_genes)
            return None
        if code.strip() == "rna_genes":
            return list(self.ref_genes)
        if ".spage_cand" in code and "rowSums" in code:
            cand = [g for g in self.globalenv.get(".spage_cand", []) if g in self.ref_genes]
            return cand
        if "Matrix::writeMM" in code:
            if self.fail_on_extract:
                raise RuntimeError("simulated R writeMM failure")
            assert "drop = FALSE" in code, "single-gene extraction without drop = FALSE drops matrix dims"
            need = [g for g in self.globalenv[".spage_need"] if g in self.ref_genes]
            idx = [self.ref_genes.index(g) for g in need]
            sub = sp.coo_matrix(self.ref_counts[idx, :])
            sio.mmwrite(self.globalenv[".spage_mat_file"], sub)
            with open(self.globalenv[".spage_genes_file"], "w") as fh:
                fh.write("\n".join(need) + "\n")
            with open(self.globalenv[".spage_cells_file"], "w") as fh:
                fh.write("\n".join(self.ref_cells) + "\n")
            self.globalenv["mat_sub"] = True
            return None
        if "rm(" in code:
            for name in (".spage_cand", ".spage_need", ".spage_mat_file", ".spage_genes_file",
                         ".spage_cells_file", "mat_sub", "seurat_ref", "rna_ref",
                         "rna_mat_data", "rna_genes", "genes_needed"):
                if f'"{name}"' in code or name in code:
                    self.globalenv.pop(name, None)
            return None
        return None


def _install_fake_rpy2(monkeypatch, fake_r):
    rpy2_mod = types.ModuleType("rpy2")
    ro_mod = types.ModuleType("rpy2.robjects")
    ro_mod.r = fake_r.r
    ro_mod.globalenv = fake_r.globalenv
    ro_mod.StrVector = fake_r.StrVector
    ro_mod.default_converter = object()

    pkg_mod = types.ModuleType("rpy2.robjects.packages")
    base_ns = types.SimpleNamespace(readRDS=lambda p: {"rds": p})
    pkg_mod.importr = lambda name: base_ns

    conv_mod = types.ModuleType("rpy2.robjects.conversion")
    conv_mod.set_conversion = lambda _c: None
    ro_mod.packages = pkg_mod
    ro_mod.conversion = conv_mod

    rinterface_lib = types.ModuleType("rpy2.rinterface_lib")
    openrlib = types.SimpleNamespace(rlock=fake_r.rlock)
    rinterface_lib.openrlib = openrlib
    callbacks_mod = types.ModuleType("rpy2.rinterface_lib.callbacks")
    callbacks_mod.consolewrite_print = lambda _x: None
    callbacks_mod.consolewrite_warnerror = lambda _x: None

    rpy2_mod.robjects = ro_mod
    rpy2_mod.rinterface_lib = rinterface_lib

    for mod_name, mod_obj in (
        ("rpy2", rpy2_mod),
        ("rpy2.robjects", ro_mod),
        ("rpy2.robjects.packages", pkg_mod),
        ("rpy2.robjects.conversion", conv_mod),
        ("rpy2.rinterface_lib", rinterface_lib),
        ("rpy2.rinterface_lib.openrlib", openrlib),
        ("rpy2.rinterface_lib.callbacks", callbacks_mod),
    ):
        monkeypatch.setitem(sys.modules, mod_name, mod_obj)


def test_r10_13_spage_adversarial_gene_names_bound_via_globalenv_not_interpolated(xe, tmp_path, monkeypatch):
    """Adversarial gene names containing quotes, backslashes, newlines, and R injection
    payloads are passed via ro.globalenv['.spage_need'] (never pasted into ro.r source),
    cached parquet streaming inside _run_spage_imputation excludes __index_level_0__,
    and R globals are cleaned up on both success and failure."""
    from xenium_app import core, spage

    monkeypatch.setattr(core, "CACHE_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(core, "_redirect_rpy2_console", lambda: None)

    # Install lightweight PVComputation stub
    repo = tmp_path / "spage_stub"
    (repo / "SpaGE").mkdir(parents=True)
    (repo / "SpaGE" / "__init__.py").write_text("")
    (repo / "SpaGE" / "principal_vectors.py").write_text(textwrap.dedent("""
        import numpy as np
        class PVComputation:
            def __init__(self, n_factors, n_pv, **kw):
                self.n = n_pv
            def fit(self, source, target):
                g = source.shape[1]
                k = min(self.n, g)
                eye = np.eye(k, g, dtype=np.float32)
                self.source_components_ = eye
                self.target_components_ = eye
                self.cosine_similarity_matrix_ = np.eye(k, dtype=np.float32)
                return self
    """))
    monkeypatch.setattr(spage, "SPAGE_REPO", str(repo))
    for k in [k for k in sys.modules if k == "SpaGE" or k.startswith("SpaGE.")]:
        monkeypatch.delitem(sys.modules, k, raising=False)

    fake_rds = tmp_path / 'ref "tricky".rds'
    fake_rds.write_bytes(b"RDS")

    expr_arr = xe.DATA["expr_csc"].toarray()
    nz_cols = np.where(expr_arr.std(axis=0) > 0)[0][:12]
    panel_genes = [xe.DATA["gene_names"][i] for i in nz_cols]
    assert len(panel_genes) >= 3, "expected non-zero variance genes in bundle"
    evil_gene_1 = 'EVIL"); stop("R_INJECTION_EXECUTED"); ("X'
    evil_gene_2 = r"GENE\WITH\BACKSLASH_AND_'SINGLE'_QUOTE"
    predict_genes = [evil_gene_1, evil_gene_2]
    all_ref_genes = panel_genes + predict_genes

    rng = np.random.default_rng(11)
    n_ref_cells = 60
    ref_cells = [f"rc_{i}" for i in range(n_ref_cells)]
    ref_counts = rng.uniform(0.5, 5.0, size=(len(all_ref_genes), n_ref_cells)).astype(np.float32)

    fake_r = _RecordingRpy2(all_ref_genes, ref_cells, ref_counts)
    _install_fake_rpy2(monkeypatch, fake_r)

    genes_input = f"{evil_gene_1}\n{evil_gene_2}"
    spage._run_spage_imputation(str(fake_rds), n_pv=4, genes_input=genes_input, seg_source="xenium")

    assert core._spage_state["status"] == "done", core._spage_state["message"]
    imp_df = core._spage_state["result"]
    assert imp_df is not None
    assert list(imp_df.columns) == [evil_gene_1, evil_gene_2]

    # CRITICAL: No ro.r(...) script may contain any interpolated gene name or temp path!
    joined_scripts = "\n---\n".join(fake_r.r_scripts)
    assert "R_INJECTION_EXECUTED" not in joined_scripts
    assert evil_gene_1 not in joined_scripts
    assert evil_gene_2 not in joined_scripts
    assert "mat.mtx" not in joined_scripts
    assert "genes.txt" not in joined_scripts
    assert "cells.txt" not in joined_scripts
    # And all temporary globals were removed from globalenv
    for tmp_sym in (".spage_need", ".spage_mat_file", ".spage_genes_file", ".spage_cells_file",
                    ".spage_cand", "mat_sub", "seurat_ref", "rna_ref", "rna_mat_data"):
        assert tmp_sym not in fake_r.globalenv, f"{tmp_sym} leaked in R globalenv"

    # R10-12 inside _run_spage_imputation cached parquet streaming path:
    # Re-run against the just-written parquet cache with SPAGE_STREAM_THRESHOLD=1 so it streams!
    monkeypatch.setattr(core, "SPAGE_STREAM_THRESHOLD", 1)
    with core._spage_lock:
        core._spage_state.update({"status": "idle", "message": "", "result": None,
                                  "result_path": None, "result_genes": []})
    spage._run_spage_imputation(str(fake_rds), n_pv=4, genes_input=genes_input, seg_source="xenium")
    assert core._spage_state["status"] == "done"
    assert core._spage_state["result"] is None
    assert core._spage_state["result_genes"] == [evil_gene_1, evil_gene_2]
    assert "__index_level_0__" not in core._spage_state["result_genes"]
    assert f"{len(predict_genes):,} genes" in core._spage_state["message"]

    # Also verify cleanup when R extraction raises mid-flight (new rds path avoids cache)
    monkeypatch.setattr(core, "SPAGE_STREAM_THRESHOLD", 50_000_000)
    fake_rds_fail = tmp_path / "ref_fail.rds"
    fake_rds_fail.write_bytes(b"RDS2")
    fake_r_fail = _RecordingRpy2(all_ref_genes, ref_cells, ref_counts, fail_on_extract=True)
    _install_fake_rpy2(monkeypatch, fake_r_fail)
    spage._run_spage_imputation(str(fake_rds_fail), n_pv=4, genes_input=genes_input, seg_source="xenium")
    assert core._spage_state["status"] == "error"
    for tmp_sym in (".spage_need", ".spage_mat_file", ".spage_genes_file", ".spage_cells_file",
                    "mat_sub", "seurat_ref", "rna_ref", "rna_mat_data"):
        assert tmp_sym not in fake_r_fail.globalenv, f"{tmp_sym} leaked after R error"


# ── R10-19: WebGL overlayBase includes subsetVer so subset/unsubset refreshes overlays ──

def test_r10_19_webgl_overlay_base_refetches_on_subset_ver_change():
    """Executing assets/xenium_viewer_dash.js in Node VM (matching glue_harness.js):
    bumping subsetVer (subset / unsubset) must invalidate overlayKey and refetch
    /viewer/overlays, while keeping subsetVer unchanged does not refetch."""
    node = shutil.which("node") or "/opt/homebrew/bin/node"
    if not os.path.exists(node):
        pytest.skip("node binary not available")

    glue_js = str(tl.REPO / "assets" / "xenium_viewer_dash.js")
    driver = textwrap.dedent("""
        const fs = require("fs");
        const vm = require("vm");

        function stub() {
          return new Proxy(function () {}, {
            get(t, k) {
              if (k === Symbol.toPrimitive) return () => "";
              if (k === "then") return undefined;
              if (!(k in t)) t[k] = stub();
              return t[k];
            },
            set(t, k, v) { t[k] = v; return true; },
            apply() { return stub(); },
            construct() { return stub(); },
          });
        }

        const viewers = [];
        const overlayFetches = [];
        let seq = 0;

        function viewerRecord(el, opts) {
          const rec = { baseUrl: opts.baseUrl || "", opts: opts, states: [] };
          viewers.push(rec);
          const api = stub();
          api.setState = (s) => { rec.states.push(JSON.parse(JSON.stringify(s))); };
          api.resize = () => {};
          api.destroy = () => {};
          return api;
        }

        const document = stub();
        const els = {};
        document.getElementById = (id) => {
          if (id === "_dash-config") return { textContent: "{}" };
          if (!els[id]) { els[id] = stub(); els[id].id = id; }
          return els[id];
        };
        const window = stub();
        window.document = document;
        window.XeniumViewer = { mount: viewerRecord };
        window.dash_clientside = { no_update: "NO_UPDATE", set_props: () => {} };
        const sandbox = {
          window, document, console, JSON, Math, Date, Object, Array, Number, String, Boolean, RegExp,
          encodeURIComponent, isFinite, parseFloat, parseInt, Promise, Event: function () {},
          setTimeout: (fn) => { try { fn(); } catch (e) {} return 0; },
          clearTimeout: () => {}, setInterval: () => 0, clearInterval: () => {},
          requestAnimationFrame: () => 0, navigator: { userAgent: "node" }, URL: stub(),
          fetch: (url) => {
            const u = String(url);
            if (u.indexOf("/viewer/overlays") >= 0 && u.indexOf("/umap/") < 0) {
              overlayFetches.push(u);
              seq += 1;
              return Promise.resolve({
                json: () => Promise.resolve({
                  polygons: [],
                  lines: [{ name: "sep_" + seq, from: [10 * seq, 0], to: [10 * seq, -500] }],
                }),
              });
            }
            return Promise.resolve({ json: () => Promise.resolve({}) });
          },
        };
        vm.createContext(sandbox);
        vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
        const glue = window.dash_clientside.xenium_viewer;

        function callSync(subsetVer, segSource, baysorDone, prosegDone) {
          return glue.sync(
            "webgl", "cluster", "graphclust", "", 2, 0.8, [], segSource || "xenium", "original", null, null,
            [], [], [], 1, 1, /* datasetVer */ 1, /* subsetVer */ subsetVer,
            /* baysorDone */ baysorDone || 0, /* prosegDone */ prosegDone || 0,
            0, 0, 0, /* roiDone */ 0, 0,
            "pan", { display: "none" }, false, /* extraVer */ 1, {}
          );
        }

        (async () => {
          callSync(0, "xenium", 0, 0);
          await new Promise((r) => setImmediate(r));
          await new Promise((r) => setImmediate(r));
          const afterInit = overlayFetches.length;

          // Identical subsetVer -> cached, no refetch
          callSync(0, "xenium", 0, 0);
          await new Promise((r) => setImmediate(r));
          const afterSame = overlayFetches.length;

          // subset() bumps subsetVer -> 1: MUST refetch /viewer/overlays
          callSync(1, "xenium", 0, 0);
          await new Promise((r) => setImmediate(r));
          await new Promise((r) => setImmediate(r));
          const afterSubset = overlayFetches.length;

          // unsubset() bumps subsetVer -> 2: MUST refetch /viewer/overlays again
          callSync(2, "xenium", 0, 0);
          await new Promise((r) => setImmediate(r));
          await new Promise((r) => setImmediate(r));
          const afterUnsubset = overlayFetches.length;

          // Selecting a cached reseg run ("proseg:a1b20001") while it loads (prosegDone=0),
          // then finishing load (prosegDone=1, or baysorDone=1) MUST refetch /viewer/overlays!
          callSync(2, "proseg:a1b20001", 0, 0);
          await new Promise((r) => setImmediate(r));
          await new Promise((r) => setImmediate(r));
          const afterResegSelect = overlayFetches.length;

          callSync(2, "proseg:a1b20001", 0, 1);
          await new Promise((r) => setImmediate(r));
          await new Promise((r) => setImmediate(r));
          const afterProsegDone = overlayFetches.length;

          callSync(2, "proseg:a1b20001", 1, 1);
          await new Promise((r) => setImmediate(r));
          await new Promise((r) => setImmediate(r));
          const afterBaysorDone = overlayFetches.length;

          const mainViewer = viewers.find((v) => !v.baseUrl);
          const lastOverlayState = mainViewer.states.filter((s) => s.overlays && Array.isArray(s.overlays.lines)).pop();
          process.stdout.write(JSON.stringify({
            afterInit, afterSame, afterSubset, afterUnsubset,
            afterResegSelect, afterProsegDone, afterBaysorDone,
            names: (lastOverlayState ? lastOverlayState.overlays.lines : []).map((o) => o.name),
          }));
        })();
    """)
    proc = subprocess.run([node, "-e", driver, glue_js], capture_output=True, text=True, check=True)
    res = json.loads(proc.stdout)
    assert res["afterInit"] == 1
    assert res["afterSame"] == 1, "unchanged subsetVer refetched /viewer/overlays unnecessarily"
    assert res["afterSubset"] == 2, "subset() (subsetVer=1) failed to refetch /viewer/overlays!"
    assert res["afterUnsubset"] == 3, "unsubset() (subsetVer=2) failed to refetch /viewer/overlays!"
    assert res["afterResegSelect"] == 4
    assert res["afterProsegDone"] == 5, "cached proseg/imgseg load finish (prosegDone) failed to refetch overlays!"
    assert res["afterBaysorDone"] == 6, "cached baysor load finish (baysorDone) failed to refetch overlays!"
    assert res["names"] == ["sep_6"]


# ── R10-21: Call-site seg_source="tool:tag" propagation + fallback when None ─────────

def test_r10_21_callbacks_and_workers_preserve_reseg_tag_in_provenance(xe, tmp_path, monkeypatch):
    """start_annotation (celltypist, rctd, seurat) and run_spatial_analysis pass
    seg_source='proseg:<tag>' (or 'baysor:<tag>', 'imgseg:<tag>', 'xenium'), and the
    worker functions record that exact seg_source while falling back to
    provenance._prov_labels_source(labels_key) when seg_source is None."""
    from xenium_app import annotation, core, jobs, provenance, spatial

    monkeypatch.setattr(xe, "CACHE_BASE_DIR", str(tmp_path))

    # 1) Check callback wiring in start_annotation across all 3 methods + xenium
    for tool_name, tag in (("proseg", "99aa0011"), ("baysor", "88bb0022"), ("imgseg", "77cc0033")):
        _rdir, seg_val, res = _make_reseg_run(xe, tmp_path, tool=tool_name, tag=tag, n_cells=16)
        st, lk = core._seg_state(tool_name)
        with lk:
            st.update({"status": "done", "message": "ok", "result": res, "loading_out_dir": None})

        fake_rds = tmp_path / f"ref_{tool_name}.rds"
        fake_rds.write_bytes(b"RDS")
        _install_fake_rpy2(monkeypatch, _RecordingRpy2(["g0"], ["c0"], [[1.0]]))

        seen_kwargs = {}

        def _intercept_job(kind, label, fn, **kw):
            seen_kwargs[label] = dict(kw.get("kwargs") or {})
            with core._annot_lock:
                core._annot_state.update({"status": "idle", "message": ""})

        monkeypatch.setattr(jobs, "_job_start", _intercept_job)

        xe.start_annotation(1, "celltypist", "Immune.pkl", None, None, None, None, None, None, seg_val)
        xe.start_annotation(1, "rctd", None, str(fake_rds), "Names", "full", 2, 20, 100, seg_val)
        xe.start_annotation(1, "seurat", None, str(fake_rds), "Names", None, None, None, None, seg_val)

        assert len(seen_kwargs) == 3
        for lbl, kw in seen_kwargs.items():
            assert kw.get("seg_source") == f"{tool_name}:{tag}", (lbl, kw)

    # Xenium start_annotation passes seg_source="xenium"
    seen_xen = {}
    monkeypatch.setattr(
        jobs, "_job_start",
        lambda kind, label, fn, **kw: (
            seen_xen.update(kw.get("kwargs") or {}),
            core._annot_state.update({"status": "idle", "message": ""}),
        ),
    )
    xe.start_annotation(1, "celltypist", "Immune.pkl", None, None, None, None, None, None, "xenium")
    assert seen_xen.get("seg_source") == "xenium"
    monkeypatch.undo()

    # 2) End-to-end provenance record check for _run_spatial_analysis via run_spatial_analysis
    #    (preserves 'proseg:deadbeef') vs direct call with seg_source=None (falls back to _prov_labels_source)
    monkeypatch.setattr(xe, "CACHE_BASE_DIR", str(tmp_path))
    _rdir, seg_val, res = _make_reseg_run(xe, tmp_path, tool="proseg", tag="deadbeef", n_cells=24)
    with core._proseg_lock:
        core._proseg_state.update({"status": "done", "message": "ok", "result": res, "loading_out_dir": None})
    lkey = annotation._labels_key_for_method("celltypist", res)
    lab = pd.Series(["Alpha", "Beta"] * 12, index=res["cells_df"].index.astype(str))
    with core._annot_lock:
        xe._annot_state[lkey] = lab

    out = xe.run_spatial_analysis(1, "labels_celltypist", 4, seg_val)
    assert out == (True, False, "Running…")
    tl.wait_state(xe._spatial_state, xe._spatial_lock)
    recs = [r for r in provenance._prov_read() if r["kind"] == "spatial"]
    assert recs and recs[-1]["data"]["seg_source"] == "proseg:deadbeef"

    # Fallback check when seg_source is None
    fallback_calls = []
    orig_labels_source = provenance._prov_labels_source

    def _spy_labels_source(k):
        fallback_calls.append(k)
        return orig_labels_source(k)

    monkeypatch.setattr(provenance, "_prov_labels_source", _spy_labels_source)
    spatial._run_spatial_analysis(lkey, 4, res)  # seg_source=None -> falls back to _prov_labels_source
    assert fallback_calls == [lkey]

    # 3) Worker-level provenance seg_source check on _run_celltypist, _run_rctd_annotation,
    #    and _run_seurat_annotation (via cached label parquet hit so no external R/CellTypist needed!)
    cell_ids = [str(c) for c in res["cells_df"].index]
    cache_sig = f"sig_{len(cell_ids)}"
    cached_df = caching_frame = xe._annot_cache_frame(lab, cache_sig)

    # CellTypist cached hit with explicit seg_source="proseg:deadbeef"
    ct_cache = xe._cache_path(f"DemoModel.pkl_{lkey}")
    caching_frame.to_parquet(ct_cache)
    fake_ad = types.ModuleType("anndata")
    fake_ct = types.ModuleType("celltypist")
    fake_ct_models = types.ModuleType("celltypist.models")
    fake_ct.models = fake_ct_models
    monkeypatch.setitem(sys.modules, "anndata", fake_ad)
    monkeypatch.setitem(sys.modules, "celltypist", fake_ct)
    monkeypatch.setitem(sys.modules, "celltypist.models", fake_ct_models)

    annotation._run_celltypist(
        "DemoModel.pkl", labels_key=lkey, expr_override=res["expr"],
        cell_ids_override=cell_ids, cache_sig=cache_sig, seg_source="proseg:deadbeef",
    )
    annot_recs = [r for r in provenance._prov_read() if r["kind"] == "annotation"]
    assert annot_recs[-1]["data"]["seg_source"] == "proseg:deadbeef"

    # RCTD cached hit with explicit seg_source="baysor:88bb0022"
    rds_file = tmp_path / "rctd_ref.rds"
    rds_file.write_bytes(b"RDS")
    monkeypatch.setattr(xe, "_rw_thread_setup", lambda: None)
    rctd_lkey = "labels_rctd_baysor_11223344"
    rctd_ckey = (f"rctd_{rds_file.name}_{xe._ref_file_tag(str(rds_file))}_Names_full"
                 f"_umi20_sig100_{rctd_lkey}")
    cached_df.to_parquet(xe._cache_path(rctd_ckey))
    annotation._run_rctd_annotation(
        str(rds_file), "Names", "full", 2, labels_key=rctd_lkey,
        expr_override=res["expr"], cell_ids_override=cell_ids,
        cache_sig=cache_sig, seg_source="baysor:88bb0022",
    )
    annot_recs = [r for r in provenance._prov_read() if r["kind"] == "annotation"]
    assert annot_recs[-1]["data"]["seg_source"] == "baysor:88bb0022"

    # Seurat cached hit with explicit seg_source="imgseg:77cc0033" AND fallback when seg_source=None
    _install_fake_rpy2(monkeypatch, _RecordingRpy2(["g0"], ["c0"], [[1.0]]))
    seu_lkey = "labels_seurat_imgseg_55667788"
    seu_ckey = f"seurat_{rds_file.name}_{xe._ref_file_tag(str(rds_file))}_Names_{seu_lkey}"
    cached_df.to_parquet(xe._cache_path(seu_ckey))
    annotation._run_seurat_annotation(
        str(rds_file), "Names", labels_key=seu_lkey,
        expr_override=res["expr"], cell_ids_override=cell_ids,
        cache_sig=cache_sig, seg_source="imgseg:77cc0033",
    )
    annot_recs = [r for r in provenance._prov_read() if r["kind"] == "annotation"]
    assert annot_recs[-1]["data"]["seg_source"] == "imgseg:77cc0033"

    fallback_calls.clear()
    annotation._run_seurat_annotation(
        str(rds_file), "Names", labels_key=seu_lkey,
        expr_override=res["expr"], cell_ids_override=cell_ids,
        cache_sig=cache_sig, seg_source=None,
    )
    assert fallback_calls == [seu_lkey]


# ── Cross-slice follow-ups (#21/#22): confirm_seg_delete & standalone R converters ────

def test_r10_cross_confirm_seg_delete_forgets_spatial_annot_and_side_caches(xe, tmp_path, monkeypatch):
    """Deleting a cached resegmentation run via confirm_seg_delete runs caching._seg_forget_run,
    clearing _spatial_state, _annot_state, and cached annotation parquets for that run while
    preserving state belonging to other runs."""
    from xenium_app import annotation, core

    monkeypatch.setattr(xe, "CACHE_BASE_DIR", str(tmp_path))
    run_dir, seg_val, res = _make_reseg_run(xe, tmp_path, tool="proseg", tag="c0ffee01", n_cells=16)
    other_dir, _oseg, other_res = _make_reseg_run(xe, tmp_path, tool="baysor", tag="c0ffee02", n_cells=16)

    with core._proseg_lock:
        core._proseg_state.update({"status": "done", "message": "ok", "result": res, "loading_out_dir": None})

    lkey = annotation._labels_key_for_method("celltypist", res)
    other_lkey = annotation._labels_key_for_method("celltypist", other_res)
    lab = pd.Series(["T", "B"] * 8, index=res["cells_df"].index.astype(str))
    other_lab = pd.Series(["Mac", "Fib"] * 8, index=other_res["cells_df"].index.astype(str))
    with core._annot_lock:
        core._annot_state[lkey] = lab
        core._annot_state[other_lkey] = other_lab

    ct_pq = xe._cache_path(f"Demo_{lkey}")
    other_pq = xe._cache_path(f"Demo_{other_lkey}")
    xe._annot_cache_frame(lab, "sig1").to_parquet(ct_pq)
    xe._annot_cache_frame(other_lab, "sig2").to_parquet(other_pq)
    assert os.path.exists(ct_pq) and os.path.exists(other_pq)

    with core._spatial_lock:
        core._spatial_state.update({
            "status": "done",
            "message": "ok",
            "result": {"out_dir": str(run_dir), "cell_types": ["T", "B"]},
        })

    out = xe.confirm_seg_delete(1, seg_val, 0, 0)
    assert out[0] is False and out[1] == "xenium"
    assert not run_dir.exists()
    assert core._spatial_state["status"] == "idle" and core._spatial_state["result"] is None
    assert lkey not in core._annot_state
    assert not os.path.exists(ct_pq)
    # Other run's labels and cached parquet remain intact
    assert core._annot_state.get(other_lkey) is other_lab
    assert os.path.exists(other_pq) and other_dir.is_dir()


def test_r10_cross_standalone_converters_never_interpolate_into_r_source(tmp_path, monkeypatch):
    """AST guard + adversarial runtime stress test on functions/spatialdata2seurat.py and
    functions/seurat2cellnest.py: every ro.r(...) argument is a static string literal (no
    f-strings, % formatting, or + concatenation), and adversarial quotes/backslashes/R-code
    in assay_name, fov_name, output_rds, image names, and temp paths never enter R source
    and are cleaned up from ro.globalenv."""
    import ast
    import importlib.util

    # 1) Static AST check across spage.py, spatialdata2seurat.py, and seurat2cellnest.py
    checked_calls = 0
    for rel_path in (
        "xenium_app/spage.py",
        "functions/spatialdata2seurat.py",
        "functions/seurat2cellnest.py",
    ):
        src_file = tl.REPO / rel_path
        tree = ast.parse(src_file.read_text(), filename=str(src_file))
        store_counts: dict[str, int] = {}
        for sub in ast.walk(tree):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                store_counts[sub.id] = store_counts.get(sub.id, 0) + 1
        mod_str_consts = {
            n.targets[0].id
            for n in tree.body
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
            and store_counts.get(n.targets[0].id, 0) == 1
        }
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "r"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "ro"):
                checked_calls += 1
                assert len(node.args) == 1 and not node.keywords, f"{rel_path}:{node.lineno}"
                arg = node.args[0]
                is_static = (
                    (isinstance(arg, ast.Constant) and isinstance(arg.value, str))
                    or (isinstance(arg, ast.Name) and arg.id in mod_str_consts)
                )
                assert is_static, (
                    f"{rel_path}:{node.lineno} passes dynamic expression {ast.dump(arg)} to ro.r()"
                )
    assert checked_calls >= 12

    # 2) Runtime adversarial test on functions/seurat2cellnest.py
    try:
        import anndata as _real_ad  # noqa: F401
    except ImportError:
        monkeypatch.setitem(sys.modules, "anndata", types.ModuleType("anndata"))
    spec_s2c = importlib.util.spec_from_file_location(
        "seurat2cellnest_under_test", str(tl.REPO / "functions" / "seurat2cellnest.py")
    )
    s2c = importlib.util.module_from_spec(spec_s2c)
    spec_s2c.loader.exec_module(s2c)

    evil_img = 'fov1"); stop("INJECT_IMG"); ("x'
    evil_assay = 'Xenium"); stop("INJECT_ASSAY"); ("y'
    evil_layer = 'counts"); stop("INJECT_LAYER"); ("z'
    evil_cell = 'cell_0"); stop("INJECT_CELL"); ("w'

    class _FakeS2CRo:
        def __init__(self):
            self.globalenv = {}
            self.scripts = []
            self._null = object()

            class _RCallable:
                def __init__(self, outer):
                    self._o = outer

                def assign(self, name, val):
                    self._o.globalenv[name] = val

                def __call__(self, code):
                    self._o.scripts.append(code)
                    if code == "NULL":
                        return self._o._null
                    if "names(Images(" in code:
                        return [evil_img]
                    if "GetTissueCoordinates" in code:
                        assert self._o.globalenv.get("._s2c_img") == evil_img
                        self._o.globalenv["._s2c_coords"] = pd.DataFrame(
                            {"cell": [evil_cell, "cell_1"], "x": [12.5, 25.0], "y": [3.0, 6.0]}
                        )
                        return None
                    if code == "._s2c_coords":
                        return self._o.globalenv["._s2c_coords"]
                    if "Matrix::writeMM" in code:
                        assert self._o.globalenv["._s2c_assay"] == evil_assay
                        assert self._o.globalenv["._s2c_layer"] == evil_layer
                        cells = list(self._o.globalenv.get("._s2c_cells", [evil_cell, "cell_1"]))
                        mat = sp.coo_matrix(np.ones((2, len(cells)), dtype=np.float32))
                        sio.mmwrite(self._o.globalenv["._s2c_mat_file"], mat)
                        with open(self._o.globalenv["._s2c_genes_file"], "w") as fh:
                            fh.write("G1\nG2\n")
                        with open(self._o.globalenv["._s2c_cells_file"], "w") as fh:
                            fh.write("\n".join(cells) + "\n")
                        self._o.globalenv["._s2c_mat"] = True
                        return None
                    if "rm(" in code:
                        for k in list(self._o.globalenv):
                            if k != "._s2c_obj" and k in code:
                                self._o.globalenv.pop(k, None)
                        return None
                    return None

            self.r = _RCallable(self)

        def StrVector(self, seq):
            return [str(v) for v in seq]

    fake_p2ri = types.SimpleNamespace(rpy2py=lambda x: x, activate=lambda: None, converter=object())
    fake_ro_pkg = types.ModuleType("rpy2.robjects")
    fake_ro_pkg.pandas2ri = fake_p2ri
    monkeypatch.setitem(sys.modules, "rpy2", types.SimpleNamespace(robjects=fake_ro_pkg))
    monkeypatch.setitem(sys.modules, "rpy2.robjects", fake_ro_pkg)
    monkeypatch.setitem(sys.modules, "rpy2.robjects.pandas2ri", fake_p2ri)

    fake_s2c_ro = _FakeS2CRo()
    meta_df = pd.DataFrame({"a": [1, 2]}, index=pd.Index([evil_cell, "cell_1"], dtype=object))
    xs, ys = s2c._extract_spatial_coords(fake_s2c_ro, {"rds": True}, meta_df)
    assert list(xs) == [12.5, 25.0] and list(ys) == [3.0, 6.0]
    assert "._s2c_img" not in fake_s2c_ro.globalenv
    assert "._s2c_coords" not in fake_s2c_ro.globalenv

    mat_all, g_all, c_all = s2c._extract_counts_mtx(fake_s2c_ro, {"rds": True}, evil_assay, evil_layer)
    assert mat_all.shape == (2, 2) and g_all == ["G1", "G2"] and c_all == [evil_cell, "cell_1"]
    mat_sub, _g_sub, c_sub = s2c._extract_counts_mtx(
        fake_s2c_ro, {"rds": True}, evil_assay, evil_layer, cell_subset=[evil_cell]
    )
    assert mat_sub.shape == (1, 2) and c_sub == [evil_cell]
    joined_s2c = "\n".join(fake_s2c_ro.scripts)
    for forbidden in ("INJECT_IMG", "INJECT_ASSAY", "INJECT_LAYER", "INJECT_CELL",
                      "mat.mtx", "genes.txt", "cells.txt"):
        assert forbidden not in joined_s2c
    for sym in ("._s2c_assay", "._s2c_layer", "._s2c_cells", "._s2c_mat_file",
                "._s2c_genes_file", "._s2c_cells_file", "._s2c_mat"):
        assert sym not in fake_s2c_ro.globalenv

    # 3) Runtime adversarial test on functions/spatialdata2seurat.py
    spec_sd2s = importlib.util.spec_from_file_location(
        "spatialdata2seurat_under_test", str(tl.REPO / "functions" / "spatialdata2seurat.py")
    )
    sd2s = importlib.util.module_from_spec(spec_sd2s)
    spec_sd2s.loader.exec_module(sd2s)

    fake_adata = types.SimpleNamespace(
        obs_names=pd.Index(["c0", "c1"], dtype=object),
        var_names=pd.Index(["G1", "G2"], dtype=object),
        var=pd.DataFrame(index=pd.Index(["G1", "G2"], dtype=object)),
        obs=pd.DataFrame(
            {"x_centroid": [1.0, 2.0], "y_centroid": [3.0, 4.0], "umap_1": [0.1, 0.2], "umap_2": [0.3, 0.4]},
            index=pd.Index(["c0", "c1"], dtype=object),
        ),
        uns={},
        layers={},
        X=sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)),
    )
    fake_adata.__len__ = lambda self=fake_adata: 2
    # SimpleNamespace len() looks up type.__len__, so wrap in a tiny class:
    class _MiniAdata:
        def __init__(self):
            self.obs_names = fake_adata.obs_names
            self.var_names = fake_adata.var_names
            self.var = fake_adata.var
            self.obs = fake_adata.obs
            self.uns = {}
            self.layers = {}
            self.X = fake_adata.X

        def __len__(self):
            return 2

    fake_sd_mod = types.ModuleType("spatialdata")
    fake_sd_mod.read_zarr = lambda _p: types.SimpleNamespace(tables={"table": _MiniAdata()}, shapes={})
    monkeypatch.setitem(sys.modules, "spatialdata", fake_sd_mod)

    sd2s_scripts = []
    sd2s_env = {}
    saved_State = {}

    def _sd2s_r(code):
        sd2s_scripts.append(code)
        if "CreateSeuratObject" in code:
            saved_State["assay"] = sd2s_env.get(".sd2s_assay")
            saved_State["fov"] = sd2s_env.get(".sd2s_fov")
            saved_State["out"] = sd2s_env.get(".sd2s_output_rds")
            assert os.path.isfile(sd2s_env[".sd2s_panel_mtx"])
            assert os.path.isfile(sd2s_env[".sd2s_meta_file"])
            assert os.path.isfile(sd2s_env[".sd2s_umap_file"])
            with open(sd2s_env[".sd2s_output_rds"], "wb") as fh:
                fh.write(b"FAKE_RDS")
            sd2s_env[".seurat"] = True
            return None
        if "rm(" in code:
            for k in list(sd2s_env):
                if k in code:
                    sd2s_env.pop(k, None)
        return None

    ro_sd2s = types.ModuleType("rpy2.robjects")
    ro_sd2s.r = _sd2s_r
    ro_sd2s.globalenv = sd2s_env
    ro_sd2s.default_converter = 1
    p2ri_sd2s = types.SimpleNamespace(activate=lambda: None, converter=2)
    ro_sd2s.pandas2ri = p2ri_sd2s
    pkg_sd2s = types.ModuleType("rpy2.robjects.packages")
    pkg_sd2s.importr = lambda _n: types.SimpleNamespace()
    conv_sd2s = types.ModuleType("rpy2.robjects.conversion")
    conv_sd2s.set_conversion = lambda _c: None
    ro_sd2s.packages = pkg_sd2s
    ro_sd2s.conversion = conv_sd2s
    monkeypatch.setitem(sys.modules, "rpy2", types.SimpleNamespace(robjects=ro_sd2s))
    monkeypatch.setitem(sys.modules, "rpy2.robjects", ro_sd2s)
    monkeypatch.setitem(sys.modules, "rpy2.robjects.packages", pkg_sd2s)
    monkeypatch.setitem(sys.modules, "rpy2.robjects.conversion", conv_sd2s)
    monkeypatch.setitem(sys.modules, "rpy2.robjects.pandas2ri", p2ri_sd2s)

    evil_out_rds = tmp_path / 'out "); stop("INJECT_OUT"); (".rds'
    evil_sd2s_assay = 'Assay"); stop("INJECT_SD2S_ASSAY"); ("'
    evil_sd2s_fov = 'fov"); stop("INJECT_SD2S_FOV"); ("'
    ret_path = sd2s.spatialdata2seurat(
        str(tmp_path / "fake.zarr"),
        str(evil_out_rds),
        assay_name=evil_sd2s_assay,
        fov_name=evil_sd2s_fov,
        include_boundaries=False,
        include_nucleus=False,
        include_imputed=False,
        cache_dir=str(tmp_path / "empty_cache"),
    )
    assert os.path.isfile(ret_path)
    assert saved_State["assay"] == evil_sd2s_assay
    assert saved_State["fov"] == evil_sd2s_fov
    joined_sd2s = "\n".join(sd2s_scripts)
    for forbidden in ("INJECT_OUT", "INJECT_SD2S_ASSAY", "INJECT_SD2S_FOV",
                      "panel.mtx", "meta.csv", "centroids.csv", "umap.csv"):
        assert forbidden not in joined_sd2s
    assert sd2s_env == {}, f"leaked globals in spatialdata2seurat: {sd2s_env}"
