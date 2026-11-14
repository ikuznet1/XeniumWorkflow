"""Round-11 regression and adversarial stress tests for Slice 2 (r11/fix-stats-tools):
S1: SegQC _qc_metrics partial split_corrected_cols filtering (controls.py)
S2: Transcript-factor _txf_reference & _run_txfactor partial SPLIT coverage (txfactor.py)
S3: Reseg RCTD coordinates, subset-invariant annotation caching & measured panel genes
    (annotation.py, cb_tools.py, clustering.py)
S4: Reseg SpaGE write-back under active / mid-job subset() & unsubset() (spage.py)
S5: functions/impute.py --use_corrected SPLIT-covered cell & gene filtering + attrs preservation
S6: Standalone SPLIT / CellNEST / Seurat scripts (run_split.py, export_to_cellnest.py,
    spatialdata2seurat.py)
S7: functions/compare_cellnest.py zero-edge replicate preservation from sample_meta
S8: Small-N guards & zarr Group cluster_split_10 decoding (clustering.py, spatial.py)
S9: Joint clustering & Banksy cache signatures on genes and s["rows"] (joint.py, banksy.py)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import types

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_func_script(mod_name: str, filename: str):
    path = REPO_ROOT / "functions" / filename
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── S1: SegQC _qc_metrics under partial split_corrected_cols ─────────────────

def test_s1_qc_metrics_partial_split_corrected_cols_excludes_uncovered_markers(xe):
    """Under partial SPLIT gene coverage, _qc_metrics restricts identity sets,
    marker pairs, and library-size denominators to s['genes'] so own_per_100 <= 100,
    100%-retained covered markers yield retention == 1.0, and uncovered-only cell
    types are omitted rather than scored with exclusion_failure == 1.0."""
    from xenium_app import controls

    # 30 cells: 10 TypeA, 10 TypeB, 10 TypeUncov
    # Genes: G_A (covered), G_A_uncov (uncovered), G_B (covered), G_U_uncov (uncovered)
    all_genes = ["G_A", "G_A_uncov", "G_B", "G_U_uncov"]
    gni = {g: i for i, g in enumerate(all_genes)}
    raw_dense = np.zeros((30, 4), dtype=np.float32)
    raw_dense[:10, 0] = 8.0      # TypeA covered marker
    raw_dense[:10, 1] = 20.0     # TypeA uncovered marker
    raw_dense[10:20, 2] = 10.0   # TypeB covered marker
    raw_dense[20:30, 3] = 15.0   # TypeUncov uncovered-only marker

    # SPLIT covered only ["G_A", "G_B"] with 100% retention; uncovered cols are 0-padded
    corr_dense = np.zeros((30, 4), dtype=np.float32)
    corr_dense[:10, 0] = 8.0
    corr_dense[10:20, 2] = 10.0

    df = pd.DataFrame(
        {
            "x_centroid": np.linspace(0.0, 290.0, 30),
            "y_centroid": np.zeros(30),
            "cell_area": np.full(30, 50.0),
        },
        index=pd.Index([f"c{i}" for i in range(30)], dtype=object),
    )
    labels = ["TypeA"] * 10 + ["TypeB"] * 10 + ["TypeUncov"] * 10
    sets = {
        "TypeA": ["G_A_uncov", "G_A"],   # first marker uncovered, second covered!
        "TypeB": ["G_B"],
        "TypeUncov": ["G_U_uncov"],      # uncovered-only cell type
    }

    covered_genes = ["G_A", "G_B"]
    # Reference ident_sums on covered cells (first 20 cells) using covered_genes
    rows_cov = np.array(list(range(20)) + [-1] * 10, dtype=np.int64)
    s_ref = {
        "csc": sp.csc_matrix(raw_dense),
        "rows": rows_cov,
        "gni": gni,
        "genes": covered_genes,
        "df": df,
        "corrected": False,
        "ds": {},
    }
    s_corr = {
        "csc": sp.csc_matrix(corr_dense),
        "rows": rows_cov,
        "gni": gni,
        "genes": covered_genes,
        "df": df,
        "corrected": True,
        "ds": {},
    }

    ref_out = controls._qc_metrics(s_ref, labels, sets, "TypeA", 30.0)
    # In s_ref with genes=covered_genes, G_A_uncov (20/cell) MUST NOT be counted in ref_ident!
    assert ref_out["ident_sums"] == {"TypeA": 80.0, "TypeB": 100.0}
    assert float(ref_out["row"]["own_per_100"]) <= 100.0 + 1e-6

    corr_out = controls._qc_metrics(
        s_corr, labels, sets, "TypeA", 30.0,
        pairs=[("G_A", "G_B"), ("G_A_uncov", "G_U_uncov")],
        ref_ident=ref_out["ident_sums"],
    )
    pt = corr_out["per_type"].set_index("type")
    assert "TypeUncov" not in pt.index
    assert np.isclose(pt.loc["TypeA", "retention"], 1.0)
    assert np.isclose(pt.loc["TypeB", "retention"], 1.0)
    assert np.isclose(corr_out["row"]["retention"], 1.0)
    assert np.isclose(corr_out["row"]["exclusion_failure"], 0.0)
    # Uncovered user pair ("G_A_uncov", "G_U_uncov") is filtered out, leaving only ("G_A", "G_B")
    assert corr_out["row"]["n_marker_pairs"] == 1
    assert np.isclose(corr_out["row"]["coexpression_rate"], 0.0)


# ── S2: Transcript-factor _txf_reference & _run_txfactor partial SPLIT ──────

def test_s2_txf_reference_and_project_never_route_uncovered_genes_to_rare_type(xe, monkeypatch):
    """_txf_reference leaves uncovered gene columns at exact 0.0 (not 1e-12 / R_f),
    so _txf_project (both numpy and kernel paths) never assigns pure abundant-type bins
    containing uncovered transcripts to a rare cell type."""
    from xenium_app import analysis, txfactor

    # 3 genes: G_ab (covered, abundant type marker), G_rare (covered, rare type marker),
    # G_uncov (uncovered in SPLIT, expressed everywhere in raw transcript bins)
    all_genes = ["G_ab", "G_rare", "G_uncov"]
    gni = {g: i for i, g in enumerate(all_genes)}
    n_ab, n_rare = 100, 10
    n_tot = n_ab + n_rare
    mat = np.zeros((n_tot, 3), dtype=np.float32)
    mat[:n_ab, 0] = 50.0     # Abundant type: total R_ab = 5000
    mat[n_ab:, 1] = 2.0      # Rare type: total R_rare = 20 (250x smaller!)
    # Col 2 (G_uncov) is 0 in SPLIT-corrected matrix

    df = pd.DataFrame(
        {"x_centroid": np.arange(n_tot, dtype=float), "y_centroid": np.zeros(n_tot)},
        index=pd.Index([f"cell_{i}" for i in range(n_tot)], dtype=object),
    )
    sample = {
        "key": "smp1",
        "label": "S1",
        "name": "S1",
        "df": df,
        "csc": sp.csc_matrix(mat),
        "rows": np.arange(n_tot, dtype=np.int64),
        "gni": gni,
        "genes": ["G_ab", "G_rare"],   # G_uncov excluded by _corrected_cols!
        "corrected": True,
        "ds": {},
    }
    monkeypatch.setattr(analysis, "_an_samples", lambda seg, corr: [sample])

    tables = [{"key": "smp1", "df": df}]
    labels = [["Abundant"] * n_ab + ["Rare"] * n_rare]

    H, names, left = txfactor._txf_reference(
        "xenium", tables, labels, all_genes, use_corrected=True, min_cells=5
    )
    assert left == []
    ab_idx = names.index("Abundant")
    rare_idx = names.index("Rare")
    # Uncovered column MUST be exact 0.0 for every cell type, while rows still sum to 1.0
    assert np.all(H[:, 2] == 0.0)
    assert np.allclose(H.sum(axis=1), 1.0)

    # Bin with 5 covered G_ab transcripts + 40 uncovered G_uncov transcripts:
    # Previously, the 40 uncovered transcripts got routed 99.6% to Rare (because 1e-12/20 >> 1e-12/5000)!
    X_bins = sp.csr_matrix(np.array([[5.0, 0.0, 40.0], [0.0, 0.0, 25.0]], dtype=np.float32))
    monkeypatch.setattr(txfactor, "_TXF_USE_NUMBA", False)
    W_np = txfactor._txf_project(X_bins, H, n_iter=80)
    assert W_np[0, ab_idx] > 4.99 and W_np[0, rare_idx] < 1e-6
    # Pure uncovered bin has 0 covered transcripts -> 0 weight on all factors
    assert np.allclose(W_np[1], 0.0)


def test_s2_run_txfactor_reference_filters_genes_to_covered_before_binning(xe, tmp_path, monkeypatch):
    """_run_txfactor(method='reference', use_corrected=True) intersects the transcript
    gene list with SPLIT-covered genes before binning/projection and reports
    n_uncovered_genes."""
    from xenium_app import analysis, controls, core, provenance, stats, txfactor

    monkeypatch.setattr(xe, "CACHE_BASE_DIR", str(tmp_path))
    all_genes = ["G_cov1", "G_cov2", "G_uncov"]
    gni = {g: i for i, g in enumerate(all_genes)}
    n_cells = 40
    df = pd.DataFrame(
        {
            "x_centroid": np.linspace(5.0, 95.0, n_cells),
            "y_centroid": np.full(n_cells, 5.0),
            "cell_area": np.full(n_cells, 25.0),
        },
        index=pd.Index([f"c{i}" for i in range(n_cells)], dtype=object),
    )
    mat = np.zeros((n_cells, 3), dtype=np.float32)
    mat[:20, 0] = 12.0
    mat[20:, 1] = 12.0
    ds = {"data_dir": str(tmp_path), "sdata_path": str(tmp_path / "s.zarr")}
    tbl = {"key": "k1", "label": "S1", "name": "S1", "df": df, "ds": ds}
    smp = {
        "key": "k1", "label": "S1", "name": "S1", "df": df,
        "csc": sp.csc_matrix(mat), "rows": np.arange(n_cells, dtype=np.int64),
        "gni": gni, "genes": ["G_cov1", "G_cov2"], "corrected": True, "ds": ds,
    }
    labs = ["Type1"] * 20 + ["Type2"] * 20

    seen_count_genes = []

    def _fake_txf_counts(_ds, genes_arg, _bin, _shape, progress=None):
        seen_count_genes.append(list(genes_arg))
        # 2 bins: one over Type1 cells, one over Type2 cells
        X = sp.csr_matrix(np.array([[20.0, 0.0], [0.0, 20.0]], dtype=np.float32))
        return {"r": np.array([0, 0], dtype=np.int32), "c": np.array([0, 4], dtype=np.int32),
                "X": X, "cached": False}

    monkeypatch.setattr(analysis, "_an_require_seg", lambda _s: None)
    monkeypatch.setattr(stats, "_view_tables", lambda _s: [tbl])
    monkeypatch.setattr(controls, "_seg_tag", lambda _s: "xenium")
    monkeypatch.setattr(analysis, "_ds_panel_genes", lambda _d: list(all_genes))
    monkeypatch.setattr(stats, "_comp_labels", lambda *_a, **_k: ([labs], "CellTypist"))
    monkeypatch.setattr(analysis, "_an_samples", lambda _s, _c: [smp])
    monkeypatch.setattr(txfactor, "_txf_counts", _fake_txf_counts)
    monkeypatch.setattr(provenance, "_prov_counts_used", lambda *_a, **_k: True)
    monkeypatch.setattr(provenance, "_prov_record", lambda *_a, **_k: None)

    txfactor._run_txfactor("xenium", None, "annot:celltypist", method="reference",
                           params={"bin": 20.0, "min_tx": 5, "min_cells": 5}, use_corrected=True)
    assert txfactor._txf_state["status"] == "done", txfactor._txf_state["message"]
    res = txfactor._txf_state["result"]
    assert seen_count_genes == [["G_cov1", "G_cov2"]]
    assert res["genes"] == ["G_cov1", "G_cov2"]
    assert res["n_uncovered_genes"] == 1
    assert "not covered by SPLIT" in txfactor._txf_state["message"]


# ── S3: Reseg RCTD coords, subset-invariant annotation cache & panel genes ───

def test_s3_reseg_rctd_coords_subset_invariant_cache_and_panel_genes(xe, tmp_path, monkeypatch):
    """1) start_annotation on a subsetted reseg run passes full-run expr, cell IDs,
          measured panel genes (excluding Xenium SpaGE-imputed columns), and real
          reseg coordinates, preserving the full-run cache_sig.
       2) Direct _run_rctd_annotation on a reseg run passes real centroid_x/centroid_y
          to rworker._rw_rctd instead of (0.0, 0.0) and never deletes a valid full-run
          cache when subset() is active."""
    from xenium_app import annotation, cb_tools, clustering, core, jobs, provenance, rworker

    monkeypatch.setattr(xe, "CACHE_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(provenance, "_prov_record", lambda *_a, **_k: None)

    # Inject a Xenium SpaGE-imputed gene AT POSITION 0 of DATA["gene_names"] to prove
    # list(DATA["gene_names"])[:n_meas] is never used!
    orig_gnames = list(core.DATA["gene_names"])
    orig_gni = dict(core.DATA["gene_name_to_idx"])
    orig_gvar = core.DATA.get("gene_var")

    panel_genes = ["PG_0", "PG_1", "PG_2", "PG_3"]
    xen_all_genes = ["XEN_IMP_0"] + panel_genes + ["XEN_IMP_1"]
    gvar = pd.DataFrame(
        {"is_imputed": [True, False, False, False, False, True]},
        index=pd.Index(xen_all_genes, dtype=object),
    )
    monkeypatch.setitem(core.DATA, "gene_names", xen_all_genes)
    monkeypatch.setitem(core.DATA, "gene_name_to_idx", {g: i for i, g in enumerate(xen_all_genes)})
    monkeypatch.setitem(core.DATA, "gene_var", gvar)

    # Reseg run with 6 full cells (currently subsetted to 2 cells!) + 1 reseg-imputed col
    full_ids = [f"proseg_{i}" for i in range(6)]
    full_df = pd.DataFrame(
        {
            "centroid_x": [110.0, 220.0, 330.0, 440.0, 550.0, 660.0],
            "centroid_y": [15.0, 25.0, 35.0, 45.0, 55.0, 65.0],
            "transcript_counts": [50, 60, 70, 80, 90, 100],
        },
        index=pd.Index(full_ids, dtype=object),
    )
    # 4 panel cols + 1 reseg SpaGE-imputed col = 5 cols
    rng = np.random.default_rng(7)
    full_expr_5 = sp.csr_matrix(rng.integers(1, 10, size=(6, 5)).astype(np.float32))
    alt_res = {
        "out_dir": str(tmp_path / "proseg_run"),
        "source": "proseg",
        "_cells_df_original": full_df,
        "cells_df": full_df.iloc[:2].copy(),
        "_expr_original": full_expr_5,
        "expr": full_expr_5[:2, :],
        "gene_name_to_idx": {g: i for i, g in enumerate(panel_genes + ["RESEG_IMP"])},
        "imputed_genes": ["RESEG_IMP"],
    }
    os.makedirs(alt_res["out_dir"], exist_ok=True)

    # Check clustering._reseg_measured_expr excludes RESEG_IMP and picks the 4 panel cols
    meas_mat = clustering._reseg_measured_expr(full_expr_5, alt_res)
    assert meas_mat.shape == (6, 4)
    assert np.allclose(meas_mat.toarray(), full_expr_5[:, :4].toarray())

    with core._proseg_lock:
        prev_proseg = dict(core._proseg_state)
        core._proseg_state.update({"status": "done", "message": "ok", "result": alt_res})

    try:
        monkeypatch.setattr(core, "_seg_job_result", lambda seg, *_a: (alt_res, None))
        fake_rds = tmp_path / "ref.rds"
        fake_rds.write_bytes(b"RDS")
        monkeypatch.setattr(core, "_ref_rds_path", lambda p: str(p))
        monkeypatch.setattr(core, "_ref_rds_problem", lambda _p: None)

        # Stub rpy2 conversion check inside start_annotation
        fake_rconv = types.SimpleNamespace(set_conversion=lambda _c: None)
        fake_ro = types.ModuleType("rpy2.robjects")
        fake_ro.default_converter = object()
        fake_ro.conversion = fake_rconv
        monkeypatch.setitem(sys.modules, "rpy2", types.SimpleNamespace(robjects=fake_ro))
        monkeypatch.setitem(sys.modules, "rpy2.robjects", fake_ro)
        monkeypatch.setitem(sys.modules, "rpy2.robjects.conversion", fake_rconv)

        captured_start = {}

        def _fake_job_start(kind, label, fn, args=(), kwargs=None, **_kw):
            captured_start["fn"] = fn
            captured_start["args"] = args
            captured_start["kwargs"] = dict(kwargs or {})
            with core._annot_lock:
                core._annot_state["status"] = "idle"

        monkeypatch.setattr(jobs, "_job_start", _fake_job_start)

        cb_tools.start_annotation(
            1, "rctd", None, str(fake_rds), "Names", "full", 2, 20, 100, "proseg:deadbeef"
        )
        kw = captured_start["kwargs"]
        # Full 6 cells & 4 measured panel columns (NOT 2 subset cells, NOT XEN_IMP_0!)
        assert kw["cell_ids_override"] == full_ids
        assert kw["expr_override"].shape == (6, 4)
        assert kw["genes_override"] == panel_genes
        assert np.allclose(kw["coords_override"]["x_centroid"].to_numpy(), [110, 220, 330, 440, 550, 660])
        assert np.allclose(kw["coords_override"]["y_centroid"].to_numpy(), [15, 25, 35, 45, 55, 65])

        # Verify cache_sig is identical whether subset is active or unsubsetted
        alt_unsub = dict(alt_res)
        alt_unsub.pop("_cells_df_original")
        alt_unsub.pop("_expr_original")
        alt_unsub["cells_df"] = full_df
        alt_unsub["expr"] = full_expr_5
        monkeypatch.setattr(core, "_seg_job_result", lambda seg, *_a: (alt_unsub, None))
        cb_tools.start_annotation(
            1, "rctd", None, str(fake_rds), "Names", "full", 2, 20, 100, "proseg:deadbeef"
        )
        assert captured_start["kwargs"]["cache_sig"] == kw["cache_sig"]

        # Now run _run_rctd_annotation directly WITHOUT coords_override / genes_override to prove
        # fallback resolution via seg_source passes real centroids & panel genes to rworker!
        monkeypatch.setattr(core, "_seg_job_result", lambda seg, *_a: (alt_res, None))
        monkeypatch.setattr(rworker, "_rw_thread_setup", lambda: None)
        rw_payload = {}

        def _fake_rw_call(_fn, payload, **_rkw):
            rw_payload.update(payload)
            return {
                "cell_ids": list(payload["cell_ids"]),
                "types": ["CM"] * len(payload["cell_ids"]),
                "weights": np.ones((len(payload["cell_ids"]), 2), dtype=np.float32) * 0.5,
                "weight_cells": list(payload["cell_ids"]),
                "weight_types": ["CM", "FB"],
            }

        monkeypatch.setattr(rworker, "_rw_call", _fake_rw_call)
        lkey = annotation._labels_key_for_method("rctd", alt_res)
        annotation._run_rctd_annotation(
            str(fake_rds), "Names", "full", 2,
            labels_key=lkey,
            expr_override=full_expr_5,
            cell_ids_override=full_ids,
            cache_sig=kw["cache_sig"],
            seg_source="proseg:deadbeef",
        )
        assert rw_payload["genes"] == panel_genes
        assert rw_payload["expr"].shape == (6, 4)
        assert np.allclose(rw_payload["cx"], [110, 220, 330, 440, 550, 660])
        assert np.allclose(rw_payload["cy"], [15, 25, 35, 45, 55, 65])

        # Verify both .parquet and _weights.parquet exist on disk and stay intact on second call
        ckey = (f"rctd_{fake_rds.name}_{xe._ref_file_tag(str(fake_rds))}_Names_full"
                f"_umi20_sig100_{lkey}")
        cfile = xe._cache_path(ckey)
        wfile = cfile.replace(".parquet", "_weights.parquet")
        assert os.path.isfile(cfile) and os.path.isfile(wfile)
    finally:
        with core._proseg_lock:
            core._proseg_state.clear()
            core._proseg_state.update(prev_proseg)
        monkeypatch.setitem(core.DATA, "gene_names", orig_gnames)
        monkeypatch.setitem(core.DATA, "gene_name_to_idx", orig_gni)
        if orig_gvar is not None:
            monkeypatch.setitem(core.DATA, "gene_var", orig_gvar)


# ── S4: Reseg SpaGE write-back under active / mid-job subset() & unsubset() ──

def test_s4_reseg_spage_writeback_updates_expr_original_and_subset_expr(xe, tmp_path, monkeypatch):
    """When SpaGE writes imputed columns to a reseg zarr while subset() is active
    (both pre-existing subset keeping all reseg cells AND mid-job subset shrinking
    alt_res['expr'] to N_sub < N_full), _expr_original gets N_full x (P+M), expr gets
    N_sub x (P+M) aligned by cell ID, and unsubset() preserves all imputed columns."""
    from xenium_app import core, provenance, spage, subsetting

    monkeypatch.setattr(xe, "CACHE_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(provenance, "_prov_record", lambda *_a, **_k: None)

    panel = list(xe._panel_gene_names())[:4]
    full_ids = [f"rc_{i}" for i in range(6)]
    full_df = pd.DataFrame(
        {
            "centroid_x": np.linspace(10.0, 60.0, 6),
            "centroid_y": np.linspace(10.0, 60.0, 6),
            "transcript_counts": [20, 30, 40, 50, 60, 70],
        },
        index=pd.Index(full_ids, dtype=object),
    )
    full_expr = sp.csr_matrix(np.arange(1, 25, dtype=np.float32).reshape(6, 4))
    sdata_dir = tmp_path / "spatialdata_proseg.zarr"
    sdata_dir.mkdir()
    out_dir = tmp_path / "proseg_out"
    out_dir.mkdir()

    # Simulate mid-job subset() having shrunk alt_res["cells_df"] & alt_res["expr"] to [rc_4, rc_1]
    sub_ids = ["rc_4", "rc_1"]
    alt_res = {
        "out_dir": str(out_dir),
        "sdata_path": str(sdata_dir),
        "source": "proseg",
        "_cells_df_original": full_df,
        "cells_df": full_df.loc[sub_ids].copy(),
        "_expr_original": full_expr,
        "expr": full_expr[[4, 1], :],
        "_corrected_original": None,
        "gene_name_to_idx": {g: i for i, g in enumerate(panel)},
        "imputed_genes": [],
    }

    # Pre-populate SpaGE parquet cache with N_full=6 rows (in shuffled row order to test ID alignment!)
    shuffled_ids = ["rc_5", "rc_3", "rc_1", "rc_0", "rc_4", "rc_2"]
    imp_vals = np.array([[50, 500], [30, 300], [10, 100], [0, 0], [40, 400], [20, 200]], dtype=np.float32)
    imp_df_full = pd.DataFrame(imp_vals, index=pd.Index(shuffled_ids, dtype=object), columns=["IMP_A", "IMP_B"])

    fake_rds = tmp_path / "spage_ref.rds"
    fake_rds.write_bytes(b"RDS")

    # Stub spatialdata zarr write inside _spage_write_zarr so _publish runs its real write-back logic
    adata_written = ad.AnnData(
        X=sp.hstack([full_expr, sp.csr_matrix((6, 2), dtype=np.float32)]).tocsr(),
        obs=pd.DataFrame(index=pd.Index(full_ids, dtype=object)),
        var=pd.DataFrame(
            {"is_imputed": [False, False, False, False, True, True]},
            index=pd.Index(panel + ["IMP_A", "IMP_B"], dtype=object),
        ),
    )

    class _FakeSData:
        def __init__(self):
            self.tables = {
                "table": ad.AnnData(
                    X=full_expr.copy(),
                    obs=pd.DataFrame(index=pd.Index(full_ids, dtype=object)),
                    var=pd.DataFrame({"is_imputed": [False] * 4}, index=pd.Index(panel, dtype=object)),
                )
            }

        def delete_element_from_disk(self, _name):
            pass

        def write_element(self, _name):
            pass

    (sdata_dir / "tables" / "table").mkdir(parents=True, exist_ok=True)
    fake_sd_mod = types.ModuleType("spatialdata")
    fake_sd_mod.read_zarr = lambda _p: _FakeSData()
    monkeypatch.setitem(sys.modules, "spatialdata", fake_sd_mod)

    # Stub R reference discovery in _run_spage, and pre-place parquet cache so _publish(cached=True) runs
    fake_ro = types.ModuleType("rpy2.robjects")
    fake_ro.globalenv = {}
    fake_ro.default_converter = object()
    fake_ro.r = lambda code: panel + ["IMP_A", "IMP_B"] if code == "rna_genes" else None
    fake_rconv = types.ModuleType("rpy2.robjects.conversion")
    fake_rconv.set_conversion = lambda _c: None
    fake_rpkg = types.ModuleType("rpy2.robjects.packages")
    fake_rpkg.importr = lambda _n: types.SimpleNamespace(readRDS=lambda _p: True)
    fake_rlib = types.ModuleType("rpy2.rinterface_lib")
    import threading as _th
    fake_rlib.openrlib = types.SimpleNamespace(rlock=_th.RLock())
    fake_ro.conversion = fake_rconv
    fake_ro.packages = fake_rpkg
    monkeypatch.setitem(sys.modules, "rpy2", types.SimpleNamespace(robjects=fake_ro, rinterface_lib=fake_rlib))
    monkeypatch.setitem(sys.modules, "rpy2.robjects", fake_ro)
    monkeypatch.setitem(sys.modules, "rpy2.robjects.conversion", fake_rconv)
    monkeypatch.setitem(sys.modules, "rpy2.robjects.packages", fake_rpkg)
    monkeypatch.setitem(sys.modules, "rpy2.rinterface_lib", fake_rlib)

    # Start _run_spage as if subset was NOT active when cache_file was computed, OR pre-seed
    # the exact cache_file _run_spage will look up!
    with core._proseg_lock:
        prev_proseg = dict(core._proseg_state)
        core._proseg_state.update({"status": "done", "message": "ok", "result": alt_res})

    try:
        monkeypatch.setattr(core, "_seg_job_result", lambda seg, *_a: (alt_res, None))
        run_id = core._seg_run_id(alt_res)
        sub_tag = spage._spage_ids_tag(alt_res["cells_df"].index)
        genes_key = "IMP_A,IMP_B"
        ckey = f"{genes_key}__proseg_s4_r{run_id}_pv10_n2_{sub_tag}" if run_id else f"{genes_key}__proseg_s4_pv10_n2_{sub_tag}"
        cfile = spage._spage_cache_path(str(fake_rds), ckey)
        os.makedirs(os.path.dirname(cfile), exist_ok=True)
        imp_df_full.to_parquet(cfile)

        spage._run_spage_imputation(str(fake_rds), 10, "IMP_A, IMP_B", seg_source="proseg:s4", use_corrected=False)
        assert core._spage_state["status"] == "done", core._spage_state["message"]

        # 1) Subsetted alt_res["expr"] has shape (2, 6) aligned to ["rc_4", "rc_1"]!
        assert alt_res["expr"].shape == (2, 6)
        assert alt_res["_expr_original"].shape == (6, 6)
        j_a = alt_res["gene_name_to_idx"]["IMP_A"]
        j_b = alt_res["gene_name_to_idx"]["IMP_B"]
        assert np.allclose(alt_res["expr"][:, [j_a, j_b]].toarray(), [[40.0, 400.0], [10.0, 100.0]])
        assert np.allclose(
            alt_res["_expr_original"][:, [j_a, j_b]].toarray(),
            [[0.0, 0.0], [10.0, 100.0], [20.0, 200.0], [30.0, 300.0], [40.0, 400.0], [50.0, 500.0]],
        )

        # 2) Simulate unsubset() and verify all 6 columns survive without IndexError!
        monkeypatch.setitem(core.DATA, "_df_original", core.DATA["df"])
        monkeypatch.setitem(core.DATA, "_df_to_expr_orig", core.DATA["df_to_expr"])
        subsetting.unsubset()
        assert "_expr_original" not in alt_res
        assert alt_res["expr"].shape == (6, 6)
        assert alt_res["imputed_genes"] == ["IMP_A", "IMP_B"]
        assert np.allclose(alt_res["expr"][:, j_b].toarray().ravel(), [0, 100, 200, 300, 400, 500])
    finally:
        core.DATA.pop("_df_original", None)
        core.DATA.pop("_df_to_expr_orig", None)
        with core._proseg_lock:
            core._proseg_state.clear()
            core._proseg_state.update(prev_proseg)


# ── S5: functions/impute.py --use_corrected cell & gene coverage ─────────────

def test_s5_impute_use_corrected_filters_uncovered_cells_and_genes(tmp_path, monkeypatch):
    """functions/impute.py with use_corrected=True:
    1) Excludes uncovered zero-padded panel columns from SpaGE shared genes,
    2) Fits & predicts SpaGE ONLY on SPLIT-covered cells (leaving uncovered cells at 0.0),
    3) Records split_corrected_cell_ids & split_corrected_cols on output."""
    imp_mod = _load_func_script("impute_s5", "impute.py")

    # Fake SpaGE repo with deterministic PVComputation
    fake_repo = tmp_path / "SpaGE_repo"
    (fake_repo / "SpaGE").mkdir(parents=True)
    (fake_repo / "SpaGE" / "__init__.py").write_text("")
    (fake_repo / "SpaGE" / "principal_vectors.py").write_text(
        "import numpy as np\n"
        "class PVComputation:\n"
        "    def __init__(self, n_factors, n_pv, **kw):\n"
        "        self.n_pv = n_pv\n"
        "    def fit(self, Xs, Xt):\n"
        "        d = Xs.shape[1]\n"
        "        k = min(self.n_pv, d)\n"
        "        self.source_components_ = np.eye(k, d, dtype=np.float32)\n"
        "        self.cosine_similarity_matrix_ = np.eye(k, dtype=np.float32)\n"
        "        return self\n"
    )

    # 8 spatial cells: c0..c4 SPLIT-covered, c5..c7 SPLIT-dropped (all-zero X_corrected rows)
    # 4 panel genes: G0, G1, G2 SPLIT-covered; G_UNCOV uncovered (all-zero in X_corrected, non-zero in raw X & ref)
    obs_ids = [f"c{i}" for i in range(8)]
    panel_genes = ["G0", "G1", "G2", "G_UNCOV"]
    rng = np.random.default_rng(123)
    raw_X = sp.csr_matrix(rng.integers(2, 15, size=(8, 4)).astype(np.float32))
    corr_dense = np.zeros((8, 4), dtype=np.float32)
    corr_dense[:5, :3] = rng.integers(1, 20, size=(5, 3)).astype(np.float32)
    # G_UNCOV col (idx 3) and rows 5..7 are all zero in X_corrected!

    adata_in = ad.AnnData(
        X=raw_X,
        obs=pd.DataFrame(index=pd.Index(obs_ids, dtype=object)),
        var=pd.DataFrame({"is_imputed": [False] * 4}, index=pd.Index(panel_genes, dtype=object)),
        uns={
            "split_corrected_cols": ["G0", "G1", "G2"],
            "split_corrected_cell_ids": obs_ids[:5],
        },
        layers={"X_corrected": sp.csr_matrix(corr_dense)},
    )

    written_holder = {}

    class _FakeSD:
        def __init__(self, table):
            self.tables = {"table": table}

        def write(self, out_path):
            os.makedirs(out_path, exist_ok=True)
            written_holder["adata"] = self.tables["table"]

    fake_sd_mod = types.ModuleType("spatialdata")
    fake_sd_mod.read_zarr = lambda _p: _FakeSD(adata_in.copy())
    monkeypatch.setitem(sys.modules, "spatialdata", fake_sd_mod)

    # Stub reference loading: 60 ref cells with positive expression on G0..G2, G_UNCOV, IMP_TARGET
    ref_genes = ["G0", "G1", "G2", "G_UNCOV", "IMP_TARGET"]
    ref_mat = rng.uniform(1.0, 5.0, size=(60, 5)).astype(np.float32)
    ref_df = pd.DataFrame(ref_mat, index=[f"ref_{i}" for i in range(60)], columns=ref_genes)

    fake_ro = types.ModuleType("rpy2.robjects")
    fake_ro.globalenv = {}
    fake_ro.default_converter = object()
    fake_ro.r = lambda code: list(ref_genes) if code == ".tmp_genes" else None
    fake_rconv = types.ModuleType("rpy2.robjects.conversion")
    fake_rconv.set_conversion = lambda _c: None
    fake_rpkg = types.ModuleType("rpy2.robjects.packages")
    fake_rpkg.importr = lambda _n: types.SimpleNamespace(readRDS=lambda _p: True)
    fake_ro.conversion = fake_rconv
    fake_ro.packages = fake_rpkg
    monkeypatch.setitem(sys.modules, "rpy2", types.SimpleNamespace(robjects=fake_ro))
    monkeypatch.setitem(sys.modules, "rpy2.robjects", fake_ro)
    monkeypatch.setitem(sys.modules, "rpy2.robjects.conversion", fake_rconv)
    monkeypatch.setitem(sys.modules, "rpy2.robjects.packages", fake_rpkg)

    seen_spage_inputs = {}
    real_vec_spage = imp_mod._vectorized_spage

    def _spy_vec_spage(spatial_df, rna_df, n_pv, genes_to_predict, spage_repo):
        seen_spage_inputs["rows"] = list(spatial_df.index)
        seen_spage_inputs["cols"] = list(spatial_df.columns)
        return real_vec_spage(spatial_df, rna_df, n_pv, genes_to_predict, spage_repo)

    monkeypatch.setattr(imp_mod, "_vectorized_spage", _spy_vec_spage)
    monkeypatch.setattr(imp_mod, "_load_reference", lambda _rds, gn, **_k: (ref_df[gn], ref_genes))

    in_zarr = tmp_path / "in.zarr"
    in_zarr.mkdir()
    out_zarr = tmp_path / "out.zarr"
    rds_file = tmp_path / "ref.rds"
    rds_file.write_bytes(b"RDS")

    ret = imp_mod.impute(
        str(in_zarr), str(rds_file),
        genes=["IMP_TARGET"], n_pv=3,
        output=str(out_zarr), use_corrected=True,
        spage_repo=str(fake_repo),
    )
    assert ret == str(out_zarr)
    # Only SPLIT-covered cells c0..c4 and SPLIT-covered panel genes G0..G2 entered SpaGE!
    assert seen_spage_inputs["rows"] == ["c0", "c1", "c2", "c3", "c4"]
    assert seen_spage_inputs["cols"] == ["G0", "G1", "G2"]

    adata_out = written_holder["adata"]
    imp_col = adata_out.X[:, -1].toarray().ravel()
    assert (imp_col[:5] > 0).all(), f"covered cells should get non-zero imputed values, got {imp_col[:5]}"
    assert np.all(imp_col[5:] == 0.0), f"SPLIT-dropped cells must remain 0.0, got {imp_col[5:]}"
    assert adata_out.uns["split_corrected_imputed_genes"] == ["IMP_TARGET"]
    assert adata_out.uns["split_corrected_cols"] == ["G0", "G1", "G2"]
    assert adata_out.uns["split_corrected_cell_ids"] == ["c0", "c1", "c2", "c3", "c4"]


# ── S6: Standalone run_split.py, export_to_cellnest.py, spatialdata2seurat.py ─

class _MiniZarrNode:
    """Lightweight in-memory Zarr v2/v3 Group & Array stand-in for sandboxed tests
    when zarr is not installed, while also exercising real zarr when available."""

    def __init__(self, is_group=True, data=None, attrs=None):
        self._is_group = is_group
        self._children = {}
        self.attrs = dict(attrs or {})
        if not is_group:
            self._arr = np.asarray(data)
            self.shape = self._arr.shape
            self.dtype = self._arr.dtype

    def __contains__(self, k):
        return k in self._children

    def __getitem__(self, k):
        if not self._is_group:
            return self._arr[k]
        if isinstance(k, str) and "/" in k:
            cur = self
            for part in k.split("/"):
                cur = cur[part]
            return cur
        return self._children[k]

    def __delitem__(self, k):
        del self._children[k]

    def keys(self):
        return self._children.keys()

    def get(self, k, default=None):
        return self._children.get(k, default)

    def require_group(self, k):
        if k not in self._children:
            self._children[k] = _MiniZarrGroup()
        return self._children[k]

    def create_dataset(self, k, data, shape=None, dtype=None, overwrite=True):
        arr = np.asarray(data, dtype=dtype) if dtype is not None else np.asarray(data)
        self._children[k] = _MiniZarrArray(arr)
        return self._children[k]


class _MiniZarrGroup(_MiniZarrNode):
    def __init__(self, attrs=None):
        super().__init__(is_group=True, attrs=attrs)


class _MiniZarrArray(_MiniZarrNode):
    def __init__(self, data, attrs=None):
        super().__init__(is_group=False, data=data, attrs=attrs)


def _install_mini_zarr_and_elem(monkeypatch, store_map: dict):
    """Install zarr + write_elem/read_elem shims if real zarr is absent, or use real zarr."""
    try:
        import zarr as real_zarr  # noqa: F401
        return False
    except ImportError:
        pass

    zmod = types.ModuleType("zarr")
    zmod.Group = _MiniZarrGroup
    zmod.Array = _MiniZarrArray
    zmod.open_group = lambda path, mode="r", **_kw: store_map[os.path.abspath(str(path))]
    monkeypatch.setitem(sys.modules, "zarr", zmod)

    def _fake_write_elem(grp, key, val):
        if sp.issparse(val):
            csr = val.tocsr()
            g = grp.require_group(key)
            g.attrs["shape"] = list(csr.shape)
            g.attrs["encoding-type"] = "csr_matrix"
            g.create_dataset("data", csr.data)
            g.create_dataset("indices", csr.indices)
            g.create_dataset("indptr", csr.indptr)
        else:
            grp.create_dataset(key, np.asarray(val))

    def _fake_read_elem(node):
        if isinstance(node, _MiniZarrArray):
            return node[:]
        if "codes" in node and "categories" in node:
            return pd.Categorical.from_codes(node["codes"][:], node["categories"][:])
        if "values" in node and "mask" in node:
            vals = node["values"][:].astype(object)
            vals[node["mask"][:].astype(bool)] = pd.NA
            return pd.array(vals, dtype="string")
        raise ValueError("unknown group")

    try:
        import anndata.io as aio
        monkeypatch.setattr(aio, "write_elem", _fake_write_elem, raising=False)
        monkeypatch.setattr(aio, "read_elem", _fake_read_elem, raising=False)
    except ImportError:
        aio = types.ModuleType("anndata.io")
        aio.write_elem = _fake_write_elem
        aio.read_elem = _fake_read_elem
        monkeypatch.setitem(sys.modules, "anndata.io", aio)
        monkeypatch.setattr(ad, "io", aio, raising=False)
        try:
            import anndata.experimental as aexp
            monkeypatch.setattr(aexp, "write_elem", _fake_write_elem, raising=False)
            monkeypatch.setattr(aexp, "read_elem", _fake_read_elem, raising=False)
        except ImportError:
            pass
    return True


def _build_mini_sdata_tree(cell_ids, gene_names, X_csr, x_cent, y_cent):
    root = _MiniZarrGroup()
    tgrp = root.require_group("tables").require_group("table")
    obs = tgrp.require_group("obs")
    obs.attrs["_index"] = "_index"
    obs.create_dataset("_index", np.asarray(cell_ids, dtype=object))
    obs.create_dataset("x_centroid", np.asarray(x_cent, dtype=np.float64))
    obs.create_dataset("y_centroid", np.asarray(y_cent, dtype=np.float64))
    var = tgrp.require_group("var")
    var.attrs["_index"] = "_index"
    var.create_dataset("_index", np.asarray(gene_names, dtype=object))
    xg = tgrp.require_group("X")
    csr = X_csr.tocsr()
    xg.attrs["shape"] = list(csr.shape)
    xg.create_dataset("data", csr.data)
    xg.create_dataset("indices", csr.indices)
    xg.create_dataset("indptr", csr.indptr)
    return root


def test_s6_run_split_writes_coverage_and_export_cellnest_decodes_nullable_and_filters(
    tmp_path, monkeypatch
):
    """1) run_split._write_split_to_zarr writes corrected_cols, split_corrected_cols,
          and split_corrected_cell_ids when partial coverage occurs.
       2) export_to_cellnest decodes nullable-string-array & categorical obs Groups,
          and under use_corrected=True filters out uncovered cells and uncovered genes."""
    split_mod = _load_func_script("run_split_s6", "run_split.py")

    sdata_dir = tmp_path / "sample.zarr"
    sdata_dir.mkdir()
    cell_ids = ["c0", "c1", "c2", "c3"]
    panel_genes = ["G0", "G1", "G_UNCOV"]
    raw_X = sp.csr_matrix(np.array([
        [5, 2, 9],
        [4, 3, 8],
        [6, 1, 7],
        [1, 1, 5],
    ], dtype=np.float32))

    store_map = {}
    used_mini = _install_mini_zarr_and_elem(monkeypatch, store_map)
    if used_mini:
        root = _build_mini_sdata_tree(cell_ids, panel_genes, raw_X, [0, 10, 20, 30], [0, 5, 10, 15])
        # Add a nullable-string-array obs column ("cell_type_nullable") with mask on c2!
        ct_grp = root["tables"]["table"]["obs"].require_group("cell_type_nullable")
        ct_grp.attrs["encoding-type"] = "nullable-string-array"
        ct_grp.create_dataset("values", np.array(["T_cell", "B_cell", "", "Macro"], dtype=object))
        ct_grp.create_dataset("mask", np.array([False, False, True, False], dtype=bool))
        store_map[os.path.abspath(str(sdata_dir))] = root
    else:
        import anndata as _ad
        import zarr as rz
        from anndata.io import write_elem
        if hasattr(_ad, "settings"):
            monkeypatch.setattr(_ad.settings, "allow_write_nullable_strings", True, raising=False)
        root = rz.open_group(str(sdata_dir), mode="w")
        tgrp = root.require_group("tables").require_group("table")
        obs_df = pd.DataFrame(
            {
                "x_centroid": [0.0, 10.0, 20.0, 30.0],
                "y_centroid": [0.0, 5.0, 10.0, 15.0],
                "cell_type_nullable": pd.array(["T_cell", "B_cell", pd.NA, "Macro"], dtype="string"),
            },
            index=pd.Index(cell_ids, dtype=object),
        )
        write_elem(tgrp, "obs", obs_df)
        write_elem(tgrp, "var", pd.DataFrame(index=pd.Index(panel_genes, dtype=object)))
        write_elem(tgrp, "X", raw_X)

    # SPLIT kept only 3 of 4 cells (c0, c1, c2) and covered only 2 of 3 genes (G0, G1)
    corr_mat = sp.csr_matrix(np.array([
        [5.0, 2.0, 0.0],
        [4.0, 3.0, 0.0],
        [6.0, 1.0, 0.0],
    ], dtype=np.float32))
    cells_df = pd.DataFrame(
        {"x_centroid": [0.0, 10.0, 20.0, 30.0], "y_centroid": [0.0, 5.0, 10.0, 15.0],
         "cluster_split_10": ["1", "2", "1", ""]},
        index=pd.Index([0, 1, 2, 3]),   # int index tests string-aligned reindex too!
    )
    # Fix index to match c0..c3 while testing non-object index alignment
    cells_df.index = pd.Index(cell_ids, dtype=object)

    split_mod._write_split_to_zarr(
        str(sdata_dir), corr_mat, cells_df, panel_genes,
        corrected_cell_ids=["c0", "c1", "c2"],
        covered_genes=["G0", "G1"],
    )

    import zarr as zcheck
    zr = zcheck.open_group(str(sdata_dir), mode="r", use_consolidated=False)
    t_after = zr["tables"]["table"]
    assert list(t_after["layers"]["X_corrected"].attrs["corrected_cols"]) == ["G0", "G1"]

    # Now load export_to_cellnest AFTER zarr is available in sys.modules
    exp_mod = _load_func_script("export_cellnest_s6", "export_to_cellnest.py")
    assert exp_mod.read_uns_list(t_after, "split_corrected_cols") == ["G0", "G1"]
    assert exp_mod.read_uns_list(t_after, "split_corrected_cell_ids") == ["c0", "c1", "c2"]

    out_cn_dir = tmp_path / "cn_out"
    h5_path = exp_mod.export_sdata_to_cellnest(
        str(sdata_dir),
        output_dir=str(out_cn_dir),
        data_name="demo",
        use_corrected=True,
        include_imputed=False,
        cell_type_col="cell_type_nullable",
    )
    adata_cn = ad.read_h5ad(h5_path)
    # Uncovered cell c3 and uncovered gene G_UNCOV MUST both be excluded!
    assert list(adata_cn.obs_names) == ["c0", "c1", "c2"]
    assert list(adata_cn.var_names) == ["G0", "G1"]
    assert adata_cn.X.shape == (3, 2)
    assert adata_cn.layers["X_raw"].shape == (3, 2)
    assert adata_cn.obsm["spatial"].shape == (3, 2)

    ann_df = pd.read_csv(out_cn_dir / "demo_annotation.csv", index_col=0)
    assert list(ann_df.index) == ["c0", "c1", "c2"]
    # Masked nullable-string entry on c2 decoded cleanly to "Unknown" (not "<NA>" or crash!)
    assert list(ann_df["cell_type"]) == ["T_cell", "B_cell", "Unknown"]


def test_s6_spatialdata2seurat_roi_modes_error_values_and_corrected_cols(tmp_path, monkeypatch):
    """spatialdata2seurat:
    1) Accepts roi_path= alias,
    2) Supports roi_name-only and roi_cls-only filtering,
    3) Reports real available ROI values when 0 cells match (before slicing adata),
    4) Filters X_panel and panel_genes to SPLIT-covered columns under use_corrected=True."""
    sd2s = _load_func_script("sd2s_s6", "spatialdata2seurat.py")

    def _make_adata():
        obs = pd.DataFrame(
            {
                "x_centroid": [10.0, 20.0, 30.0, 40.0],
                "y_centroid": [10.0, 20.0, 30.0, 40.0],
                "roi_anat": ["LV", "RV", "LV", pd.NA],
            },
            index=pd.Index(["c0", "c1", "c2", "c3"], dtype=object),
        )
        X = sp.csr_matrix(np.ones((4, 3), dtype=np.float32))
        X_corr = sp.csr_matrix(np.array([
            [5.0, 2.0, 0.0],
            [4.0, 1.0, 0.0],
            [3.0, 6.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=np.float32))
        a = ad.AnnData(
            X=X,
            obs=obs,
            var=pd.DataFrame(index=pd.Index(["G0", "G1", "G_UNCOV"], dtype=object)),
            uns={"split_corrected_cols": ["G0", "G1"]},
            layers={"X_corrected": X_corr},
        )
        return a

    fake_sd_mod = types.ModuleType("spatialdata")
    fake_sd_mod.read_zarr = lambda _p: types.SimpleNamespace(tables={"table": _make_adata()}, shapes={})
    monkeypatch.setitem(sys.modules, "spatialdata", fake_sd_mod)

    captured_r = {}
    sd2s_env = {}

    def _fake_r(code):
        if "CreateSeuratObject" in code:
            with open(sd2s_env[".sd2s_panel_genes_file"]) as fh:
                captured_r["genes"] = [ln.strip() for ln in fh if ln.strip()]
            with open(sd2s_env[".sd2s_panel_cells_file"]) as fh:
                captured_r["cells"] = [ln.strip() for ln in fh if ln.strip()]
            with open(sd2s_env[".sd2s_output_rds"], "wb") as fh:
                fh.write(b"RDS")
        if "rm(" in code:
            sd2s_env.clear()

    ro_mod = types.ModuleType("rpy2.robjects")
    ro_mod.r = _fake_r
    ro_mod.globalenv = sd2s_env
    ro_mod.default_converter = 1
    p2ri = types.SimpleNamespace(activate=lambda: None, converter=2)
    ro_mod.pandas2ri = p2ri
    pkg_mod = types.ModuleType("rpy2.robjects.packages")
    pkg_mod.importr = lambda _n: types.SimpleNamespace()
    conv_mod = types.ModuleType("rpy2.robjects.conversion")
    conv_mod.set_conversion = lambda _c: None
    ro_mod.packages = pkg_mod
    ro_mod.conversion = conv_mod
    monkeypatch.setitem(sys.modules, "rpy2", types.SimpleNamespace(robjects=ro_mod))
    monkeypatch.setitem(sys.modules, "rpy2.robjects", ro_mod)
    monkeypatch.setitem(sys.modules, "rpy2.robjects.packages", pkg_mod)
    monkeypatch.setitem(sys.modules, "rpy2.robjects.conversion", conv_mod)
    monkeypatch.setitem(sys.modules, "rpy2.robjects.pandas2ri", p2ri)

    empty_cache = str(tmp_path / "empty_cache")
    # 1) Non-matching ROI value MUST list ['LV', 'RV'] (not []!)
    with pytest.raises(ValueError, match=r"LV.*RV|RV.*LV"):
        sd2s.spatialdata2seurat(
            str(tmp_path / "in.zarr"), str(tmp_path / "bad.rds"),
            roi_cls="anat", roi_name="NO_SUCH_ROI",
            include_boundaries=False, include_nucleus=False, include_imputed=False,
            cache_dir=empty_cache,
        )

    # 2) roi_name ONLY (no roi_cls) + use_corrected=True filters to LV cells AND covered genes G0, G1!
    sd2s.spatialdata2seurat(
        str(tmp_path / "in.zarr"), str(tmp_path / "lv.rds"),
        roi_name="LV", use_corrected=True,
        include_boundaries=False, include_nucleus=False, include_imputed=False,
        cache_dir=empty_cache,
    )
    assert captured_r["cells"] == ["c0", "c2"]
    assert captured_r["genes"] == ["G0", "G1"]

    # 3) roi_cls ONLY (no roi_name) keeps all cells assigned to any ROI in roi_anat (c0, c1, c2)
    sd2s.spatialdata2seurat(
        str(tmp_path / "in.zarr"), str(tmp_path / "anat.rds"),
        roi_cls="anat", use_corrected=False,
        include_boundaries=False, include_nucleus=False, include_imputed=False,
        cache_dir=empty_cache,
    )
    assert captured_r["cells"] == ["c0", "c1", "c2"]

    # 4) roi_path= alias works identically to roi_dir=
    empty_dir = tmp_path / "empty_rois"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        sd2s.spatialdata2seurat(
            str(tmp_path / "in.zarr"), str(tmp_path / "alias.rds"),
            roi_path=str(empty_dir), roi_cls="anat", roi_name="LV",
            include_boundaries=False, include_nucleus=False, include_imputed=False,
            cache_dir=empty_cache,
        )


# ── S7: functions/compare_cellnest.py zero-edge replicate preservation ───────

def test_s7_compare_cellnest_preserves_zero_edge_replicates_in_stats_and_barplot(tmp_path):
    """When one biological replicate in Condition B has 0 edges in metrics_df,
    passing sample_meta includes that replicate as 0.0 in run_pairwise_comparison
    and plot_barplot_top_hits instead of silently dropping it."""
    cc_mod = _load_func_script("compare_cellnest_s7", "compare_cellnest.py")

    # 3 Control replicates (all 0.30) vs 3 Disease replicates (two at 0.30, d3 has 0 edges!)
    sample_meta = {
        "c1": {"condition": "Ctrl", "n_cells": 100},
        "c2": {"condition": "Ctrl", "n_cells": 100},
        "c3": {"condition": "Ctrl", "n_cells": 100},
        "d1": {"condition": "Dis", "n_cells": 100},
        "d2": {"condition": "Dis", "n_cells": 100},
        "d3": {"condition": "Dis", "n_cells": 100},   # zero-edge replicate absent from metrics_df!
    }
    metrics_df = pd.DataFrame([
        {"sample": "c1", "condition": "Ctrl", "lr_pair": "L1-R1", "normalized_freq": 0.30, "mean_attention": 0.8},
        {"sample": "c2", "condition": "Ctrl", "lr_pair": "L1-R1", "normalized_freq": 0.30, "mean_attention": 0.8},
        {"sample": "c3", "condition": "Ctrl", "lr_pair": "L1-R1", "normalized_freq": 0.30, "mean_attention": 0.8},
        {"sample": "d1", "condition": "Dis",  "lr_pair": "L1-R1", "normalized_freq": 0.30, "mean_attention": 0.8},
        {"sample": "d2", "condition": "Dis",  "lr_pair": "L1-R1", "normalized_freq": 0.30, "mean_attention": 0.8},
    ])

    res = cc_mod.run_pairwise_comparison(
        metrics_df, "Ctrl", "Dis", n_perm=200, n_boot=100, min_presence=1, sample_meta=sample_meta
    )
    assert len(res) == 1
    # With d3=0.0 included, mean_Dis MUST be (0.30 + 0.30 + 0.00)/3 = 0.20 (NOT 0.30!)
    assert np.isclose(res.iloc[0]["mean_Ctrl"], 0.30)
    assert np.isclose(res.iloc[0]["mean_Dis"], 0.20)
    assert res.iloc[0]["log2FC"] < 0.0

    # Force significance to inspect plot_barplot_top_hits bar heights & replicate dots
    res_sig = res.copy()
    res_sig["perm_FDR"] = 0.01
    out_pdf = tmp_path / "bars.pdf"
    cc_mod.plot_barplot_top_hits(
        metrics_df, {("Ctrl", "Dis"): res_sig}, str(out_pdf), top_n=5, sample_meta=sample_meta
    )
    assert out_pdf.is_file() and out_pdf.stat().st_size > 0


# ── S8: Small-N guards in clustering._run_reseg_umap & spatial.py ────────────

def test_s8_small_n_guards_umap_spatial_analysis_niches_and_zarr_group_obs(xe, tmp_path, monkeypatch):
    """1) _run_reseg_umap cleanly sets status='error' when n_cells < 4 (both Xenium and reseg),
          succeeds with TSNE fallback on n_cells=4 (perplexity < 4), and decodes zarr Group
          cluster_split_10 without KeyError: slice(None).
       2) _run_spatial_analysis cleanly errors (no traceback) when <2 labelled cells exist.
       3) _run_spatial_niches cleanly errors when n_cells < n_clusters."""
    from xenium_app import clustering, core, provenance, spatial

    monkeypatch.setattr(provenance, "_prov_record", lambda *_a, **_k: None)

    # 1a) Reseg _run_reseg_umap with 2 cells -> clean error
    tiny_df = pd.DataFrame(
        {"centroid_x": [1.0, 2.0], "centroid_y": [3.0, 4.0]},
        index=pd.Index(["r0", "r1"], dtype=object),
    )
    tiny_res = {
        "out_dir": str(tmp_path / "tiny_reseg"),
        "source": "proseg",
        "cells_df": tiny_df,
        "expr": sp.csr_matrix(np.ones((2, 4), dtype=np.float32)),
    }
    monkeypatch.setattr(core, "_seg_job_result", lambda seg, *_a: (tiny_res, None))
    with core._umap_reseg_lock:
        core._umap_reseg_state.update({"status": "running", "seg_source": "proseg:tiny", "counts_mode": "original"})
    clustering._run_reseg_umap()
    assert core._umap_reseg_state["status"] == "error"
    assert "at least 4 cells" in core._umap_reseg_state["message"]

    # 1b) Reseg _run_reseg_umap with 4 cells + forced TSNE fallback -> succeeds (perplexity < 4)!
    four_df = pd.DataFrame(
        {"centroid_x": [1.0, 2.0, 3.0, 4.0], "centroid_y": [1.0, 2.0, 3.0, 4.0]},
        index=pd.Index(["r0", "r1", "r2", "r3"], dtype=object),
    )
    four_expr = sp.csr_matrix(np.array([
        [5.0, 1.0, 0.0, 2.0],
        [0.0, 4.0, 2.0, 1.0],
        [2.0, 0.0, 6.0, 1.0],
        [1.0, 3.0, 1.0, 5.0],
    ], dtype=np.float32))
    four_res = {
        "out_dir": str(tmp_path / "four_reseg"),
        "source": "proseg",
        "cells_df": four_df,
        "expr": four_expr,
    }
    monkeypatch.setattr(core, "_seg_job_result", lambda seg, *_a: (four_res, None))
    # Force umap import failure so TSNE fallback runs with n_cells == 4
    bad_umap = types.ModuleType("umap.umap_")
    monkeypatch.setitem(sys.modules, "umap.umap_", bad_umap)
    with core._umap_reseg_lock:
        core._umap_reseg_state.update({"status": "running", "seg_source": "proseg:four", "counts_mode": "original"})
    clustering._run_reseg_umap()
    assert core._umap_reseg_state["status"] == "done", core._umap_reseg_state["message"]
    assert core._umap_reseg_state["result"].shape == (4, 2)

    # 1c) Xenium _run_reseg_umap zarr Group cluster_split_10 fallback + <4 cell guard
    sdata_fb = tmp_path / "xen_fb.zarr"
    sdata_fb.mkdir()
    store_map = {}
    used_mini = _install_mini_zarr_and_elem(monkeypatch, store_map)
    xen_ids = [str(c) for c in core.DATA["df"].index[:4]]
    xen_genes = list(core.DATA["gene_names"][:4])
    xcorr_fb = sp.csr_matrix(np.array([
        [5.0, 2.0, 1.0, 0.0],
        [4.0, 1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32))
    if used_mini:
        root_fb = _build_mini_sdata_tree(xen_ids, xen_genes, xcorr_fb, [1, 2, 3, 4], [1, 2, 3, 4])
        lay_fb = root_fb["tables"]["table"].require_group("layers").require_group("X_corrected")
        lay_fb.attrs["shape"] = list(xcorr_fb.shape)
        lay_fb.create_dataset("data", xcorr_fb.data)
        lay_fb.create_dataset("indices", xcorr_fb.indices)
        lay_fb.create_dataset("indptr", xcorr_fb.indptr)
        cl_grp = root_fb["tables"]["table"]["obs"].require_group("cluster_split_10")
        cl_grp.attrs["encoding-type"] = "categorical"
        cl_grp.create_dataset("codes", np.array([0, 1, -1, -1], dtype=np.int32))
        cl_grp.create_dataset("categories", np.array(["1", "2"], dtype=object))
        store_map[os.path.abspath(str(sdata_fb))] = root_fb
    else:
        import zarr as rz
        from anndata.io import write_elem
        root_fb = rz.open_group(str(sdata_fb), mode="w")
        tgrp_fb = root_fb.require_group("tables").require_group("table")
        obs_fb = pd.DataFrame(
            {"cluster_split_10": pd.Categorical.from_codes([0, 1, -1, -1], ["1", "2"])},
            index=pd.Index(xen_ids, dtype=object),
        )
        write_elem(tgrp_fb, "obs", obs_fb)
        write_elem(tgrp_fb, "var", pd.DataFrame(index=pd.Index(xen_genes, dtype=object)))
        write_elem(tgrp_fb, "X", xcorr_fb)
        write_elem(tgrp_fb.require_group("layers"), "X_corrected", xcorr_fb)

    monkeypatch.setattr(core, "_seg_job_result", lambda seg, *_a: (None, None) if seg == "xenium" else (four_res, None))
    monkeypatch.setitem(core.DATA, "sdata_path", str(sdata_fb))
    monkeypatch.setitem(core.DATA, "split_corrected_expr", None)
    monkeypatch.setitem(core.DATA, "split_corrected_cell_ids", None)
    monkeypatch.setitem(core.DATA, "df", core.DATA["df"].iloc[:4].copy())
    monkeypatch.setitem(core.DATA, "expr", core.DATA["expr"][:4, :4])
    monkeypatch.setitem(core.DATA, "df_to_expr", np.arange(4, dtype=np.int64))
    with core._umap_reseg_lock:
        core._umap_reseg_state.update({"status": "running", "seg_source": "xenium", "counts_mode": "corrected"})
    clustering._run_reseg_umap()
    # Decoded zarr Group cluster_split_10 -> 2 SPLIT-kept cells -> hits clean <4 cell guard!
    assert core.DATA["split_corrected_cell_ids"] == xen_ids[:2]
    assert core._umap_reseg_state["status"] == "error"
    assert "at least 4 cells" in core._umap_reseg_state["message"]

    # 2) _run_spatial_analysis with only 1 labelled cell -> clean error without traceback
    one_label = pd.Series(["T"], index=pd.Index(["r0"], dtype=object))
    with core._annot_lock:
        core._annot_state["labels_one_test"] = one_label
    try:
        spatial._run_spatial_analysis("labels_one_test", 5, four_res, seg_source="proseg:four")
        assert core._spatial_state["status"] == "error"
        assert "at least 2 labelled cells" in core._spatial_state["message"]
        assert "Traceback" not in core._spatial_state["message"]
    finally:
        with core._annot_lock:
            core._annot_state.pop("labels_one_test", None)

    # 3) _run_spatial_niches with n_cells=4 < n_clusters=8 -> clean error
    spatial._run_spatial_niches(8, 3, None, four_res)
    assert core._niche_state["status"] == "error"
    assert "Need at least 8 cells" in core._niche_state["message"]


# ── S9: Joint clustering & Banksy cache signatures ───────────────────────────

def test_s9_joint_and_banksy_cache_signatures_distinguish_genes_and_rows(xe):
    """joint._jc_cache_tag changes when either common genes (e.g. split_corrected_cols)
    or s['rows'] change, and banksy._bk_signature changes when s['rows'] changes."""
    from xenium_app import banksy, joint

    csc = sp.csc_matrix(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32))
    df = pd.DataFrame({"x_centroid": [0.0, 1.0], "y_centroid": [0.0, 1.0]},
                      index=pd.Index(["a", "b"], dtype=object))
    gni = {"G0": 0, "G1": 1, "G2": 2}

    s_base = {
        "key": "k1", "corrected": True, "csc": csc, "df": df,
        "rows": np.array([0, 1], dtype=np.int64),
        "genes": ["G0", "G1", "G2"], "gni": gni,
    }
    s_diff_rows = {**s_base, "rows": np.array([0, 2], dtype=np.int64)}
    s_diff_genes = {**s_base, "genes": ["G0", "G1"]}

    params_jc = {"n_pcs": 10, "k": 5, "res": 0.5}
    tag_base = joint._jc_cache_tag([s_base], "xenium", params_jc, ["L0"])
    tag_rows = joint._jc_cache_tag([s_diff_rows], "xenium", params_jc, ["L0"])
    tag_genes = joint._jc_cache_tag([s_diff_genes], "xenium", params_jc, ["L0"])
    assert tag_base != tag_rows, "joint._jc_cache_tag ignored s['rows']!"
    assert tag_base != tag_genes, "joint._jc_cache_tag ignored common genes!"

    params_bk = banksy._bk_params({})
    bk_base = banksy._bk_signature([s_base], "xenium", params_bk, ["G0", "G1"])
    bk_rows = banksy._bk_signature([s_diff_rows], "xenium", params_bk, ["G0", "G1"])
    assert bk_base != bk_rows, "banksy._bk_signature ignored s['rows']!"
