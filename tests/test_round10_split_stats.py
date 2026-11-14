"""Round-10 regressions and adversarial stress tests for split / stats / analysis /
segfree / proximity / controls (R10-8, R10-14, R10-15, R10-16, R10-17, R10-18).
"""
from __future__ import annotations

import io
import os
import shutil

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import xe_testlib as tl


@pytest.fixture(autouse=True)
def _clean_round10_state(xe):
    """Ensure round-10 tests leave zero analysis/stats/segfree/controls state or cache files behind."""
    yield
    xe._test_baseline.settle()
    xe._controls_reset()
    xe._stats_reset()
    xe._sfree_reset()
    xe._lrd_reset()
    with xe._density_mem_lock:
        xe._density_mem.clear()
    for path in (xe._gs_path(), xe._de_groups_path()):
        if os.path.exists(path):
            os.remove(path)
    shutil.rmtree(os.path.dirname(xe._density_cache_file("x")), ignore_errors=True)
    for f in os.listdir(xe.CACHE_BASE_DIR):
        if f.startswith("design_") and f.endswith(".json"):
            os.remove(os.path.join(xe.CACHE_BASE_DIR, f))
    xe._an_autoload()


def _clone_zarr(src: str, dst) -> str:
    dst = str(dst)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def _synthetic_sample(label: str, n_cells: int, genes: list[str], seed: int = 0, x_offset: float = 0.0) -> dict:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.0, 500.0, size=(n_cells, 2))
    df = pd.DataFrame(
        {"x_centroid": xy[:, 0], "y_centroid": xy[:, 1], "transcript_counts": np.full(n_cells, 50.0)},
        index=[f"{label}_c{i}" for i in range(n_cells)],
    )
    mat = sp.csr_matrix(rng.poisson(2.0, size=(n_cells, len(genes))).astype(np.float32))
    gni = {g: i for i, g in enumerate(genes)}
    ds = {
        "df": df,
        "expr": mat,
        "df_to_expr": np.arange(n_cells, dtype=np.int64),
        "gene_names": list(genes),
        "gene_name_to_idx": gni,
        "sdata_path": f"/mem/{label}.zarr",
    }
    return {
        "label": label,
        "df": df,
        "csc": mat.tocsc(),
        "rows": np.arange(n_cells, dtype=np.int64),
        "gni": gni,
        "genes": list(genes),
        "x_offset": float(x_offset),
        "corrected": False,
        "n_uncorrected": 0,
        "n_genes_uncorrected": 0,
        "ds": ds,
        "key": f"/mem/{label}.zarr",
        "name": f"{label}.zarr",
    }


# ── R10-8: SPLIT checks subset state at start of job, never end ───────────────

def test_r10_8_xenium_split_unsubset_mid_run_preserves_zarr_and_full_counts(xe, D, tmp_path, monkeypatch):
    zcopy = _clone_zarr(D["sdata_path"], tmp_path / "xenium_r10_8.zarr")
    monkeypatch.setitem(D, "sdata_path", zcopy)
    monkeypatch.setattr(xe.rworker, "_rw_thread_setup", lambda: None)

    panel = list(xe.dataview._panel_gene_names())
    prior_full = (sp.csr_matrix(D["expr"])[:, :len(panel)] * 3.0).tocsr()
    D["split_corrected_expr"] = prior_full.copy()
    D["split_corrected_cols"] = list(panel)

    all_ids = [str(c) for c in D["df"].index]
    sub_ids = all_ids[: max(12, len(all_ids) // 2)]
    assert 10 <= len(sub_ids) < len(all_ids)
    xe.subset(cell_ids=sub_ids)
    assert "_df_original" in D

    z_before = tl.bundle_snapshot(zcopy)

    def _fake_rw_call(_fn, payload, **_kw):
        # Simulate user running unsubset() in REPL while R executes
        xe.unsubset()
        assert "_df_original" not in D
        cids = list(payload["cell_ids"])
        gids = list(payload["panel_genes"])
        # Return distinct value 7.0 on every cell/gene of the subset job
        purified = sp.csc_matrix(np.full((len(gids), len(cids)), 7.0, dtype=np.float32))
        return {"cell_ids": cids, "gene_ids": gids, "purified": purified}

    monkeypatch.setattr(xe.rworker, "_rw_call", _fake_rw_call)
    xe.split._run_split_correction("/tmp/fake_ref.rds", seg_source="xenium")

    assert xe._split_state["status"] == "done"
    assert "subset active: result kept in memory only" in xe._split_state["message"]
    assert tl.bundle_snapshot(zcopy) == z_before, "subset-started SPLIT job must never overwrite zarr"

    corr = D["split_corrected_expr"]
    assert corr.shape == (D["expr"].shape[0], len(panel))
    d2e = np.asarray(D["df_to_expr"])
    in_sub = D["df"].index.astype(str).isin(set(sub_ids))
    sub_rows = d2e[in_sub]
    out_rows = d2e[~in_sub]
    np.testing.assert_allclose(corr[sub_rows].toarray(), 7.0)
    np.testing.assert_allclose(corr[out_rows].toarray(), prior_full[out_rows].toarray())
    assert len(D["df"]["cluster_split_10"]) == len(D["df"])
    assert D["df"].loc[in_sub, "cluster_split_10"].notna().all()
    assert D["df"].loc[~in_sub, "cluster_split_10"].isna().all()


def test_r10_8_xenium_split_unsubset_mid_run_without_prior_full_leaves_uncovered_empty(xe, D, tmp_path, monkeypatch):
    zcopy = _clone_zarr(D["sdata_path"], tmp_path / "xenium_r10_8_noprior.zarr")
    monkeypatch.setitem(D, "sdata_path", zcopy)
    monkeypatch.setattr(xe.rworker, "_rw_thread_setup", lambda: None)
    D["split_corrected_expr"] = None

    all_ids = [str(c) for c in D["df"].index]
    sub_ids = all_ids[: max(4, len(all_ids) // 4)]
    xe.subset(cell_ids=sub_ids)

    def _fake_rw_call(_fn, payload, **_kw):
        xe.unsubset()
        cids = list(payload["cell_ids"])
        gids = list(payload["panel_genes"])
        purified = sp.csc_matrix(np.full((len(gids), len(cids)), 5.0, dtype=np.float32))
        return {"cell_ids": cids, "gene_ids": gids, "purified": purified}

    monkeypatch.setattr(xe.rworker, "_rw_call", _fake_rw_call)
    xe.split._run_split_correction("/tmp/fake_ref.rds", seg_source="xenium")

    assert xe._split_state["status"] == "done"
    assert "subset active" in xe._split_state["message"]
    corr = D["split_corrected_expr"]
    assert corr.shape[0] == D["expr"].shape[0]
    d2e = np.asarray(D["df_to_expr"])
    in_sub = np.asarray(D["df"].index.astype(str).isin(set(sub_ids)))
    assert (corr[d2e[in_sub]].toarray() == 5.0).all()
    assert corr[d2e[~in_sub]].nnz == 0


def test_r10_8_reseg_split_unsubset_mid_run_preserves_full_corrected_expr_and_zarr(xe, D, tmp_path, monkeypatch):
    zcopy = _clone_zarr(D["sdata_path"], tmp_path / "proseg_r10_8.zarr")
    monkeypatch.setattr(xe.rworker, "_rw_thread_setup", lambda: None)

    panel = list(xe.dataview._panel_gene_names())
    n = min(40, len(D["df"]))
    rdf = D["df"].iloc[:n].copy()
    rdf.index = pd.Index([f"r_{i}" for i in range(n)])
    rexpr = sp.csr_matrix(np.ones((n, len(panel)), dtype=np.float32))
    prior_corr = (rexpr * 4.0).tocsr()
    alt_res = {
        "source": "proseg",
        "out_dir": str(tmp_path / "proseg_out"),
        "sdata_path": zcopy,
        "cells_df": rdf,
        "expr": rexpr,
        "gene_name_to_idx": {g: i for i, g in enumerate(panel)},
        "split_corrected_expr": prior_corr.copy(),
        "split_corrected_cols": list(panel[:-1]),
    }
    with xe._proseg_lock:
        xe._proseg_state.update({"status": "done", "result": alt_res})

    sub_r_ids = [f"r_{i}" for i in range(10)]
    # Subset both Xenium (first 4 cells) and proseg (explicit cell_ids)
    xe.subset(cell_ids=sub_r_ids + [str(D["df"].index[0])])
    assert "_cells_df_original" in alt_res and len(alt_res["cells_df"]) == 10

    z_before = tl.bundle_snapshot(zcopy)
    kept_sub_genes = list(panel[1:])  # drops panel[0], while prior full dropped panel[-1]

    def _fake_rw_call(_fn, payload, **_kw):
        xe.unsubset()
        assert "_cells_df_original" not in alt_res and len(alt_res["cells_df"]) == n
        cids = list(payload["cell_ids"])
        purified = sp.csc_matrix(np.full((len(kept_sub_genes), len(cids)), 9.0, dtype=np.float32))
        return {"cell_ids": cids, "gene_ids": kept_sub_genes, "purified": purified}

    monkeypatch.setattr(xe.rworker, "_rw_call", _fake_rw_call)
    xe.split._run_split_correction("/tmp/fake_ref.rds", seg_source="proseg")

    assert xe._split_state["status"] == "done"
    assert "subset active: result kept in memory only" in xe._split_state["message"]
    assert tl.bundle_snapshot(zcopy) == z_before

    corr = alt_res["split_corrected_expr"]
    assert corr.shape == (n, len(panel))
    # First 10 rows updated (9.0 on kept_sub_genes, 0 on panel[0]); rows 10..n preserved at 4.0!
    np.testing.assert_allclose(corr[:10, 1:].toarray(), 9.0)
    np.testing.assert_allclose(corr[:10, :1].toarray(), 0.0)
    np.testing.assert_allclose(corr[10:].toarray(), 4.0)
    # Merged corrected_cols is intersection of prior full (panel[:-1]) and subset run (panel[1:])
    assert alt_res["split_corrected_cols"] == list(panel[1:-1])


def test_r10_8_xenium_split_started_full_then_subset_mid_run_never_mismatches_df(xe, D, tmp_path, monkeypatch):
    zcopy = _clone_zarr(D["sdata_path"], tmp_path / "xenium_r10_8_rev.zarr")
    monkeypatch.setitem(D, "sdata_path", zcopy)
    monkeypatch.setattr(xe.rworker, "_rw_thread_setup", lambda: None)

    all_ids = [str(c) for c in D["df"].index]
    sub_ids = all_ids[: max(4, len(all_ids) // 3)]

    def _fake_rw_call(_fn, payload, **_kw):
        xe.subset(cell_ids=sub_ids)
        cids = list(payload["cell_ids"])
        gids = list(payload["panel_genes"])
        purified = sp.csc_matrix(np.ones((len(gids), len(cids)), dtype=np.float32))
        return {"cell_ids": cids, "gene_ids": gids, "purified": purified}

    monkeypatch.setattr(xe.rworker, "_rw_call", _fake_rw_call)
    xe.split._run_split_correction("/tmp/fake_ref.rds", seg_source="xenium")

    assert xe._split_state["status"] == "done"
    assert "subset active" not in xe._split_state["message"]
    assert len(D["df"]["cluster_split_10"]) == len(sub_ids)
    assert len(D["_df_original"]["cluster_split_10"]) == len(all_ids)


# ── R10-14: Panel genes absent from SPLIT reference excluded as unmeasured ────

def test_r10_14_split_persists_corrected_cols_and_dataview_lazy_reads_zarr(xe, D, tmp_path, monkeypatch):
    import zarr

    zcopy = _clone_zarr(D["sdata_path"], tmp_path / "xenium_r10_14.zarr")
    monkeypatch.setitem(D, "sdata_path", zcopy)
    monkeypatch.setattr(xe.rworker, "_rw_thread_setup", lambda: None)

    panel = list(xe.dataview._panel_gene_names())
    assert len(panel) >= 6
    kept = panel[:-3]

    def _fake_rw_call(_fn, payload, **_kw):
        cids = list(payload["cell_ids"])
        purified = sp.csc_matrix(np.full((len(kept), len(cids)), 2.0, dtype=np.float32))
        return {"cell_ids": cids, "gene_ids": list(kept), "purified": purified}

    monkeypatch.setattr(xe.rworker, "_rw_call", _fake_rw_call)
    xe.split._run_split_correction("/tmp/fake_ref.rds", seg_source="xenium")
    assert xe._split_state["status"] == "done"
    assert D["split_corrected_cols"] == kept

    zroot = zarr.open_group(zcopy, mode="r", use_consolidated=False)
    assert list(zroot["tables"]["table"]["layers"]["X_corrected"].attrs["corrected_cols"]) == kept
    assert "split_corrected_cols" in zroot["tables"]["table"]["uns"]

    # Lazy reload from zarr when split_corrected_cols is not yet in holder
    fresh_holder = {"split_corrected_expr": D["split_corrected_expr"], "sdata_path": zcopy}
    got = xe.dataview._corrected_cols(fresh_holder)
    assert got == set(kept)
    assert fresh_holder["split_corrected_cols"] == kept

    # Full AnnData table rewrite (e.g. SpaGE / Save as SpatialData) drops low-level
    # layers/X_corrected.attrs["corrected_cols"] while preserving adata.uns["split_corrected_cols"]
    import anndata as _ad
    try:
        from anndata.io import read_elem as _read_elem, write_elem as _write_elem
    except ImportError:
        from anndata.experimental import read_elem as _read_elem, write_elem as _write_elem
    zrewrite = _clone_zarr(zcopy, tmp_path / "xenium_r10_14_adata_rewrite.zarr")
    zrw_grp = zarr.open_group(zrewrite, mode="r+", use_consolidated=False)
    adata_old = _read_elem(zrw_grp["tables"]["table"])
    adata_new = _ad.AnnData(
        X=adata_old.X,
        obs=adata_old.obs.copy(),
        var=adata_old.var.copy(),
        uns=dict(adata_old.uns),
        obsm=dict(adata_old.obsm),
        layers={k: v.copy() for k, v in adata_old.layers.items()},
    )
    del zrw_grp["tables"]["table"]
    _write_elem(zrw_grp["tables"], "table", adata_new)
    zrw_ro = zarr.open_group(zrewrite, mode="r", use_consolidated=False)
    assert zrw_ro["tables"]["table"]["layers"]["X_corrected"].attrs.get("corrected_cols") is None
    rewritten_holder = {"split_corrected_expr": D["split_corrected_expr"], "sdata_path": zrewrite}
    assert xe.dataview._corrected_cols(rewritten_holder) == set(kept)
    assert rewritten_holder["split_corrected_cols"] == kept

    # retry_split_zarr_write preserves corrected_cols on a fresh store
    zretry = _clone_zarr(D["sdata_path"], tmp_path / "xenium_r10_14_retry.zarr")
    # Remove X_corrected first to prove retry writes it with attrs + uns
    zr_rw = zarr.open_group(zretry, mode="r+", use_consolidated=False)
    if "X_corrected" in zr_rw["tables"]["table"].get("layers", {}):
        del zr_rw["tables"]["table"]["layers"]["X_corrected"]
    if "uns" in zr_rw["tables"]["table"] and "split_corrected_cols" in zr_rw["tables"]["table"]["uns"]:
        del zr_rw["tables"]["table"]["uns"]["split_corrected_cols"]
    monkeypatch.setitem(D, "sdata_path", zretry)
    xe.retry_split_zarr_write()
    zr_ro = zarr.open_group(zretry, mode="r", use_consolidated=False)
    assert list(zr_ro["tables"]["table"]["layers"]["X_corrected"].attrs["corrected_cols"]) == kept
    assert "split_corrected_cols" in zr_ro["tables"]["table"]["uns"]

    # Legacy store without corrected_cols attr or uns entry -> returns None and caches None
    zlegacy = _clone_zarr(zretry, tmp_path / "xenium_r10_14_legacy.zarr")
    zl_rw = zarr.open_group(zlegacy, mode="r+", use_consolidated=False)
    zl_rw["tables"]["table"]["layers"]["X_corrected"].attrs.pop("corrected_cols", None)
    if "uns" in zl_rw["tables"]["table"] and "split_corrected_cols" in zl_rw["tables"]["table"]["uns"]:
        del zl_rw["tables"]["table"]["uns"]["split_corrected_cols"]
    legacy_holder = {"split_corrected_expr": D["split_corrected_expr"], "sdata_path": zlegacy}
    assert xe.dataview._corrected_cols(legacy_holder) is None
    assert "split_corrected_cols" in legacy_holder and legacy_holder["split_corrected_cols"] is None


def test_r10_14_disk_loaded_unresolved_corrected_cols_preserved_across_subset_merge_and_retry(xe, D, tmp_path, monkeypatch):
    """When DATA or a cached reseg run was loaded from zarr on disk, 'split_corrected_cols'
    is not yet in the dict until dataview._corrected_cols lazily reads X_corrected.attrs.
    Verify subset(), subset SPLIT merges, and retry_split_zarr_write() resolve the disk
    attribute first instead of treating missing 'split_corrected_cols' as None ('all genes')."""
    import zarr

    monkeypatch.setattr(xe.rworker, "_rw_thread_setup", lambda: None)
    panel = list(xe.dataview._panel_gene_names())
    assert len(panel) >= 5
    disk_cols = panel[:2]
    sub_cols = panel[:3]  # includes panel[2], which is all-zero padding outside the subset!

    # Prepare Xenium zarr + Proseg zarr on disk with corrected_cols = disk_cols
    zx = _clone_zarr(D["sdata_path"], tmp_path / "xen_disk_lazy.zarr")
    zp = _clone_zarr(D["sdata_path"], tmp_path / "proseg_disk_lazy.zarr")
    monkeypatch.setitem(D, "sdata_path", zx)

    n_x = D["expr"].shape[0]
    disk_mat_x = sp.csr_matrix(np.ones((n_x, len(panel)), dtype=np.float32))
    xe.split._split_write_zarr(zx, disk_mat_x, D["df"], panel, corrected_cols=disk_cols)

    p_df = D["df"].copy()
    p_df.index = pd.Index([f"pd_{i}" for i in range(len(p_df))])
    # Align zp obs cell_id to p_df so _split_write_zarr succeeds on zp too
    zp_rw = zarr.open_group(zp, mode="r+", use_consolidated=False)
    if "cell_id" in zp_rw["tables"]["table"]:
        del zp_rw["tables"]["table"]["cell_id"]
    zp_rw["tables"]["table"].create_array("cell_id", data=np.asarray(p_df.index.astype(str), dtype=str))
    disk_mat_p = sp.csr_matrix(np.ones((len(p_df), len(panel)), dtype=np.float32) * 2.0)
    xe.split._split_write_zarr(zp, disk_mat_p, p_df, panel, corrected_cols=disk_cols)

    # (a) retry_split_zarr_write() when "split_corrected_cols" is NOT in D / alt_res
    D["split_corrected_expr"] = disk_mat_x.copy()
    D["split_panel_genes"] = list(panel)
    D.pop("split_corrected_cols", None)

    alt_res = {
        "source": "proseg",
        "out_dir": str(tmp_path / "proseg_lazy_out"),
        "sdata_path": zp,
        "cells_df": p_df.copy(),
        "expr": disk_mat_p.copy(),
        "gene_name_to_idx": {g: i for i, g in enumerate(panel)},
        "split_corrected_expr": disk_mat_p.copy(),
        "split_panel_genes": list(panel),
    }
    assert "split_corrected_cols" not in alt_res
    with xe._proseg_lock:
        xe._proseg_state.update({"status": "done", "result": alt_res})

    xe.retry_split_zarr_write()
    xe.retry_split_zarr_write("proseg")
    zx_ro = zarr.open_group(zx, mode="r", use_consolidated=False)
    zp_ro = zarr.open_group(zp, mode="r", use_consolidated=False)
    assert list(zx_ro["tables"]["table"]["layers"]["X_corrected"].attrs["corrected_cols"]) == disk_cols
    assert list(zp_ro["tables"]["table"]["layers"]["X_corrected"].attrs["corrected_cols"]) == disk_cols

    # (b) Reset both holders to freshly disk-loaded state (NO "split_corrected_cols" key!),
    # subset(), and run subset SPLIT returning sub_cols (disk_cols + panel[2]).
    D["split_corrected_expr"] = disk_mat_x.copy()
    D.pop("split_corrected_cols", None)
    alt_res["split_corrected_expr"] = disk_mat_p.copy()
    alt_res.pop("split_corrected_cols", None)
    alt_res.pop("_corrected_cols_original", None)

    x_sub_ids = [str(c) for c in D["df"].index[: max(12, len(D["df"]) // 2)]]
    p_sub_ids = [str(c) for c in p_df.index[: max(12, len(p_df) // 2)]]
    xe.subset(cell_ids=x_sub_ids + p_sub_ids)
    assert alt_res.get("_corrected_cols_original") == disk_cols

    def _fake_rw_sub(_fn, payload, **_kw):
        cids = list(payload["cell_ids"])
        purified = sp.csc_matrix(np.full((len(sub_cols), len(cids)), 9.0, dtype=np.float32))
        return {"cell_ids": cids, "gene_ids": list(sub_cols), "purified": purified}

    monkeypatch.setattr(xe.rworker, "_rw_call", _fake_rw_sub)
    xe.split._run_split_correction("/tmp/fake_ref.rds", seg_source="xenium")
    assert xe._split_state["status"] == "done"
    # Non-subset rows in D["split_corrected_expr"] came from disk_cols only -> intersected!
    assert D["split_corrected_cols"] == disk_cols

    xe.split._run_split_correction("/tmp/fake_ref.rds", seg_source="proseg")
    assert xe._split_state["status"] == "done"
    # Subset view covers all subset cells with sub_cols, while full copy intersects with disk_cols
    assert alt_res["split_corrected_cols"] == sub_cols
    assert alt_res["_corrected_cols_original"] == disk_cols

    # Retry zarr write while subsetted on proseg writes _corrected_original with disk_cols
    xe.retry_split_zarr_write("proseg")
    zp_ro2 = zarr.open_group(zp, mode="r", use_consolidated=False)
    assert list(zp_ro2["tables"]["table"]["layers"]["X_corrected"].attrs["corrected_cols"]) == disk_cols

    # Unsubsetting restores full proseg matrix with intersected disk_cols (panel[2] NOT exposed!)
    xe.unsubset()
    assert alt_res["split_corrected_cols"] == disk_cols
    assert xe._an_samples("xenium", True)[0]["genes"] == disk_cols
    assert xe._an_samples("proseg", True)[0]["genes"] == disk_cols


def test_r10_14_de_and_csv_exclude_uncorrected_genes_without_bh_inflation(xe, D):
    panel = list(xe.dataview._panel_gene_names())
    assert len(panel) >= 8
    kept = panel[:5]
    dropped = panel[5:]

    n_expr = D["expr"].shape[0]
    n_df = len(D["df"])
    d2e = np.asarray(D["df_to_expr"])
    half = n_df // 2
    assert half >= xe.analysis._DE_MIN_CELLS and (n_df - half) >= xe.analysis._DE_MIN_CELLS

    # Build a synthetic corrected matrix: gene kept[0] strongly separates A vs B,
    # kept[1..4] have background counts, dropped genes are all-zero padding columns.
    rng = np.random.default_rng(42)
    dense = np.zeros((n_expr, D["expr"].shape[1]), dtype=np.float32)
    for g in kept[1:]:
        dense[:, D["gene_name_to_idx"][g]] = rng.poisson(3.0, size=n_expr) + 1.0
    dense[d2e[:half], D["gene_name_to_idx"][kept[0]]] = 25.0
    dense[d2e[half:], D["gene_name_to_idx"][kept[0]]] = 0.0
    D["split_corrected_expr"] = sp.csr_matrix(dense)
    D["split_corrected_cols"] = list(kept)

    smps = xe._an_samples("xenium", True)
    assert len(smps) == 1
    assert smps[0]["genes"] == kept
    assert smps[0]["n_genes_uncorrected"] == len(dropped)

    mask_a = np.arange(n_df) < half
    res = xe._de_compute(smps, [mask_a], [~mask_a])
    assert list(res["table"]["gene"]) == kept
    assert set(res["table"]["gene"]).isdisjoint(set(dropped))
    assert res["n_genes"] == len(kept)
    assert res["n_genes_uncorrected"] == len(dropped)

    # Verify BH FDR uses m = len(kept), NOT len(panel)
    p0 = float(res["table"].loc[res["table"]["gene"] == kept[0], "pval"].iloc[0])
    padj0 = float(res["table"].loc[res["table"]["gene"] == kept[0], "padj"].iloc[0])
    assert padj0 == pytest.approx(min(1.0, p0 * len(kept)), rel=1e-5)
    assert padj0 < p0 * len(panel)

    # Full _run_de + download_de_table CSV header check
    x = D["df"]["x_centroid"].to_numpy(float)
    y = D["df"]["y_centroid"].to_numpy(float)
    xm = float(np.median(x))
    poly = [[x.min() - 1, y.min() - 1], [xm, y.min() - 1], [xm, y.max() + 1], [x.min() - 1, y.max() + 1]]
    xe.de_capture_webgl_selection({"polygon": poly, "count": int((x < xm).sum())}, 0)
    method = (D.get("cluster_methods") or [None])[0]
    xe.analysis._run_de("sel:current", "rest", "xenium", method, "cluster", True)
    assert xe._de_state["status"] == "done"
    assert f"{len(dropped):,} panel genes absent from the SPLIT reference" in xe._de_state["message"]
    dl = xe.download_de_table(1)
    assert f"# {len(dropped):,} panel genes absent from the SPLIT reference" in dl["content"]
    csv_df = pd.read_csv(io.StringIO(dl["content"]), comment="#")
    assert set(csv_df["gene"]) == set(kept)


def test_r10_14_lr_moran_geneset_proximity_spillover_and_segqc_consumers(xe, D):
    panel = list(xe.dataview._panel_gene_names())
    assert len(panel) >= 8
    kept = panel[:5]
    dropped = panel[5:]

    rng = np.random.default_rng(7)
    n_expr = D["expr"].shape[0]
    dense = np.zeros((n_expr, D["expr"].shape[1]), dtype=np.float32)
    for g in kept:
        dense[:, D["gene_name_to_idx"][g]] = rng.poisson(4.0, size=n_expr) + 1.0
    D["split_corrected_expr"] = sp.csr_matrix(dense)
    D["split_corrected_cols"] = list(kept)

    # Annotate 12 Alpha (target), 3 Beta (anchor), remainder Gamma (pool for matched random anchors)
    ids = [str(c) for c in D["df"].index]
    assert len(ids) >= 20
    raw_labs = ["Alpha"] * 12 + ["Beta"] * 3 + ["Gamma"] * (len(ids) - 15)
    labs = pd.Series(raw_labs, index=ids)
    with xe._annot_lock:
        xe._annot_state["labels_rctd"] = labs

    # 1. Ligand-receptor (_lr_compute & _run_lr): pair with dropped gene is NOT on-panel
    pairs_txt = f"{kept[0]},{kept[1]}\n{kept[0]},{dropped[0]}\n{dropped[0]},{dropped[1]}"
    xe.stats._run_lr(pairs_txt, "celltype:rctd", "xenium", None, True, 6, 10, seed=0)
    assert xe._lr_state["status"] == "done"
    lr_res = xe._lr_state["result"]
    assert lr_res["info"]["n_pairs_on_panel"] == 1
    assert lr_res["info"]["n_pairs_missing"] == 2
    assert lr_res["n_genes_uncorrected"] == len(dropped)
    assert "absent from the SPLIT reference" in xe._lr_state["message"]
    assert "absent from the SPLIT reference" in xe.download_lr_table(1)["content"]
    assert set(lr_res["table"]["ligand"]) <= set(kept)
    assert set(lr_res["table"]["receptor"]) <= set(kept)

    # 1b. LR exposure decomposition (_run_lrdecomp & _lrd_csv): dropped genes excluded + noted
    xe.lrdecomp._run_lrdecomp(
        pairs_txt, "celltype:rctd", "xenium", None, True,
        radius=500.0, tile=0, own=0.0, floor=0.0, binary=False, min_cells=2,
    )
    assert xe._lrd_state["status"] == "done", xe._lrd_state["message"]
    lrd_res = xe._lrd_state["result"]
    assert lrd_res["pairs_kept"] == [(kept[0], kept[1])]
    assert lrd_res["n_genes_uncorrected"] == len(dropped)
    assert "absent from the SPLIT reference" in xe._lrd_state["message"]
    assert "absent from the SPLIT reference" in xe.lrdecomp._lrd_csv(lrd_res)
    with pytest.raises(ValueError, match="absent from the SPLIT reference"):
        xe.lrdecomp._lrd_compute(xe._an_samples("xenium", True), [labs.to_numpy()], [(dropped[0], dropped[1])], min_cells=2)

    # 2. Moran's I (_run_moran & download_moran_table): tests only kept genes + 0-gene guard
    xe.stats._run_moran("xenium", True, k=5)
    assert xe._moran_state["status"] == "done"
    m_res = xe._moran_state["result"]
    assert set(m_res["table"]["gene"]) == set(kept)
    assert m_res["n_genes_uncorrected"] == len(dropped)
    assert "absent from the SPLIT reference" in xe._moran_state["message"]
    assert "absent from the SPLIT reference" in xe.download_moran_table(1)["content"]

    empty_s = dict(xe._an_samples("xenium", True)[0], genes=[])
    with pytest.raises(ValueError, match="no measured genes"):
        xe.stats._moran_sample(empty_s, 5)

    # 3. Gene-set score (_run_geneset_score): dropped genes treated as unmeasured
    with xe._geneset_lock:
        xe._geneset_state["sets"]["r10_gs"] = [kept[0], kept[1], dropped[0]]
    xe.analysis._run_geneset_score("r10_gs", "xenium", True)
    assert xe._geneset_state["status"] == "done"
    assert dropped[0] in xe._geneset_state["message"]
    assert "absent from the SPLIT reference" in xe._geneset_state["message"]

    # 4. Proximity DE (_prox_compute & _prox_csv_header)
    smps = xe._an_samples("xenium", True)
    xy = xe.stats._xy_of(D["df"])
    a_xy = xy[labs.to_numpy() == "Alpha"]
    b_xy = xy[labs.to_numpy() == "Beta"]
    dists = np.sqrt(((a_xy[:, None, :] - b_xy[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    near_d = float(np.percentile(dists, 35)) + 1e-3
    far_d = float(np.percentile(dists, 65)) - 1e-3
    if far_d <= near_d:
        near_d, far_d = float(dists.min()) + 1e-3, float(dists.max()) - 1e-3
    p_res = xe.proximity._prox_compute(
        smps,
        [labs.to_numpy()],
        "Alpha",
        "Beta",
        params={"min_target": 5, "min_anchors": 2, "min_near": 2, "min_far": 2, "min_slope_n": 4,
                "near": near_d, "far": far_d, "n_random": 10, "min_detect": 0.0, "seed": 0},
    )
    assert set(p_res["table"]["gene"]) <= set(kept)
    assert p_res["n_genes_uncorrected"] == len(dropped)
    assert "absent from the SPLIT reference" in xe.proximity._prox_csv_header(p_res)

    # 5. Spillover (_run_spillover & _spill_csv) and SegQC (_segqc_sources)
    xe.controls._run_spillover("xenium", None, "celltype:rctd", "Alpha", f"{kept[0]}, {kept[1]}", 5, True, own_frac=0.25)
    assert xe._spill_state["status"] == "done", xe._spill_state["message"]
    sp_res = xe._spill_state["result"]
    assert set(sp_res["ownership"]["gene"]) == set(kept)
    assert sp_res["n_genes_uncorrected"] == len(dropped)
    assert "absent from the SPLIT reference" in xe._spill_state["message"]
    assert "absent from the SPLIT reference" in xe.controls._spill_csv(sp_res)

    srcs = xe.controls._segqc_sources(run_values=[])
    xen_src = next(s for s in srcs if s["value"] == "xenium")
    assert xen_src["corrected"]["genes"] == kept


# ── R10-15: Exact float64 n_edges in LR above 2^24 (16,777,216) ───────────────

def test_r10_15_lr_compute_n_edges_exact_above_float32_precision_limit(xe, monkeypatch):
    """Above 2^24 = 16,777,216, float32 loses unit integer precision (e.g. 16,777,217 ->
    16,777,216 and 18,407,592 -> 18,407,592 / off-by-1 sums). Verify _lr_compute
    computes exact int64 n_edges on > 2^24 undirected edges."""
    # Construct 2 cell types: 4,097 Alpha + 4,096 Beta = 8,193 cells.
    # Connect first 4,096 Alpha to all 4,096 Beta (16,777,216 edges) + last Alpha to Beta[0]
    # = 16,777,217 = 2^24 + 1 exact undirected cross-type edges!
    # In float32: np.float32(16777217) == 16777216.0 (loses the +1!).
    assert int(np.float32(16_777_217)) == 16_777_216

    n_a, n_b = 4097, 4096
    n = n_a + n_b
    genes = ["Lig1", "Rec1"]
    rng = np.random.default_rng(15)
    df = pd.DataFrame(
        {"x_centroid": np.linspace(0.0, 1000.0, n), "y_centroid": np.linspace(0.0, 1000.0, n)},
        index=[f"c{i}" for i in range(n)],
    )
    csc = sp.csc_matrix(np.full((n, 2), 3.0, dtype=np.float32))
    smp = {
        "label": "S0",
        "df": df,
        "csc": csc,
        "rows": np.arange(n, dtype=np.int64),
        "gni": {"Lig1": 0, "Rec1": 1},
        "genes": genes,
        "x_offset": 0.0,
        "corrected": False,
        "n_uncorrected": 0,
        "n_genes_uncorrected": 0,
        "ds": {"df": df, "expr": csc.tocsr()},
        "key": "mem:r10_15",
        "name": "r10_15",
    }
    labels = np.array(["Alpha"] * n_a + ["Beta"] * n_b, dtype=object)

    def _mock_knn(_xy, _k):
        # Every cell has kk = 4096 directed neighbours:
        # - Alpha 0..4095 -> all Beta (n_a .. n_a+4095): 4096 * 4096 = 16,777,216 cross edges
        # - Alpha 4096    -> Beta[0] (n_a) + 4095 Alpha peers (0..4094): +1 cross edge = 16,777,217!
        # - Beta  0..4095 -> Alpha 0..4095 (symmetric, so undirected count stays 16,777,217!)
        idx = np.empty((n, n_b), dtype=np.int64)
        beta_all = np.arange(n_a, n, dtype=np.int64)
        idx[:n_b, :] = beta_all[None, :]
        idx[n_b, 0] = n_a
        idx[n_b, 1:] = np.arange(n_b - 1, dtype=np.int64)
        idx[n_a:, :] = np.arange(n_b, dtype=np.int64)[None, :]
        return idx

    monkeypatch.setattr(xe.stats, "_knn_indices", _mock_knn)
    out = xe.stats._lr_compute([smp], [labels], [("Lig1", "Rec1")], k=6, n_perm=2, seed=0)
    t = out["table"].set_index(["sender", "receiver"])
    assert int(t.loc[("Alpha", "Beta"), "n_edges"]) == 16_777_217
    assert int(t.loc[("Beta", "Alpha"), "n_edges"]) == 16_777_217
    assert int(t.loc[("Alpha", "Alpha"), "n_edges"]) == 4095


# ── R10-16: _run_segfree checks _ds_has_transcripts, not data_dir truthiness ──

def test_r10_16_segfree_skips_cell_only_directory_bundle_without_aborting(xe, D, tmp_path):
    if not xe.dataview._ds_has_transcripts(D):
        pytest.skip("bundle without transcript file")

    cell_only_dir = tmp_path / "cell_only_bundle"
    cell_only_dir.mkdir()
    (cell_only_dir / "experiment.xenium").write_text("{}")

    n = 60
    rng = np.random.default_rng(16)
    genes = list(D["gene_names"])
    eds = {
        "data_dir": str(cell_only_dir),
        "df": pd.DataFrame(
            {"x_centroid": rng.uniform(0, 300, n), "y_centroid": rng.uniform(0, 300, n)},
            index=[f"co_{i}" for i in range(n)],
        ),
        "expr": sp.csr_matrix(rng.poisson(1.0, size=(n, len(genes))).astype(np.float32)),
        "df_to_expr": np.arange(n),
        "gene_names": genes,
        "gene_name_to_idx": dict(D["gene_name_to_idx"]),
        "x_offset": 5000.0,
        "cluster_methods": [],
    }
    assert not xe.dataview._ds_has_transcripts(eds)
    xe.EXTRA_DATASETS.append(eds)

    mk = genes[0]
    xe.segfree._run_segfree("xenium", None, None, None, mk, 5, False, bins_text="100", area_mode="cells", hole_max_bins=0, n_null=0, seed=0)
    assert xe._sfree_state["status"] == "done", xe._sfree_state["message"]
    msg = xe._sfree_state["message"]
    res = xe._sfree_state["result"]
    assert res["skipped"] == ["S1"]
    assert list(res["table"]["sample"].unique()) == ["S0"]
    assert "Left out (no transcript file): S1" in msg
    assert "loaded from a .zarr" not in msg

    # Density status & _run_density sibling check on the same multi-sample view
    dst = xe.analysis._density_status(mk)
    assert "S1" in dst["unavailable"]
    xe.analysis._run_density([mk])
    assert xe._density_state["status"] == "done"
    assert "S1 (no transcript file)" in xe._density_state["message"]


def test_r10_16_segfree_only_cell_only_directory_bundle_reports_accurate_error(xe, D, tmp_path, monkeypatch):
    cell_only_dir = tmp_path / "only_cells_dir"
    cell_only_dir.mkdir()
    monkeypatch.setitem(D, "data_dir", str(cell_only_dir))

    with pytest.raises(ValueError, match="no transcripts.parquet and no transcripts.zarr.zip"):
        xe.segfree._sfree_ensure_grids(D, [D["gene_names"][0]])

    xe.segfree._run_segfree("xenium", None, None, None, D["gene_names"][0], 5, False, bins_text="100", area_mode="cells", hole_max_bins=0, n_null=0, seed=0)
    assert xe._sfree_state["status"] == "error"
    err = xe._sfree_state["message"]
    assert "no transcript file" in err
    assert "loaded from a .zarr" not in err


# ── R10-17: Contact-group DE disambiguates \x00-prefixed unlabelled units ─────

def _contact_sample(label: str, genes: list[str], seed: int) -> tuple[dict, np.ndarray]:
    """Synthetic sample with 30 Target cells (15 in contact with Anchor within 15 µm,
    15 far at > 100 µm) in a single stratum."""
    rng = np.random.default_rng(seed)
    # 15 contact Target near (10, 10), 5 Anchor at (10, 10), 15 comparator Target near (300, 300)
    xy = np.vstack([
        rng.normal(loc=10.0, scale=1.5, size=(15, 2)),
        rng.normal(loc=10.0, scale=0.5, size=(5, 2)),
        rng.normal(loc=300.0, scale=2.0, size=(15, 2)),
    ])
    n = len(xy)
    df = pd.DataFrame({"x_centroid": xy[:, 0], "y_centroid": xy[:, 1]}, index=[f"{label}_{i}" for i in range(n)])
    mat = sp.csr_matrix((rng.poisson(5.0, size=(n, len(genes))) + 1).astype(np.float32))
    gni = {g: i for i, g in enumerate(genes)}
    s = {
        "label": label,
        "name": label,
        "df": df,
        "csc": mat.tocsc(),
        "rows": np.arange(n, dtype=np.int64),
        "gni": gni,
        "genes": list(genes),
        "x_offset": 0.0,
        "corrected": False,
        "n_uncorrected": 0,
        "n_genes_uncorrected": 0,
        "ds": {"df": df, "expr": mat},
        "key": f"mem:{label}",
    }
    labs = np.array(["Target"] * 15 + ["Anchor"] * 5 + ["Target"] * 15, dtype=object)
    return s, labs


def _stub_pb_fit_recorder(seen: list):
    def _fit(prep: dict) -> dict:
        seen.append(prep)
        g = prep["genes"]
        m = len(g)
        t = pd.DataFrame({
            "gene": g,
            "log2fc": np.zeros(m),
            "base_mean": np.full(m, 10.0),
            "mean_a": np.full(m, 10.0),
            "mean_b": np.full(m, 10.0),
            "stat": np.zeros(m),
            "pval": np.ones(m),
            "padj": np.ones(m),
            "neg_log10_pval": np.zeros(m),
            "neg_log10_padj": np.zeros(m),
        })
        return {"table": t, "method_label": "stub", "stat_name": "stat", "shrink": None,
                "engine": "stub", "mean_unit": "norm", "notes": []}
    return _fit


def test_r10_17_contact_de_keeps_unlabelled_sample_distinct_from_matching_donor_name(xe):
    genes = ["G0", "G1", "G2", "G3"]

    # Case 1: 4 samples S0..S3 with donors ['S3', 'S3', 'd2', ''] -> 3 distinct blocks
    s4_pairs = [_contact_sample(f"S{i}", genes, 100 + i) for i in range(4)]
    smps4 = [p[0] for p in s4_pairs]
    labs4 = [p[1] for p in s4_pairs]
    strata4 = [np.zeros(len(s["df"]), dtype=np.int64) for s in smps4]
    seen4: list = []
    res4 = xe.proximity._contact_compute(
        smps4, labs4, "Target", "Anchor", strata=strata4, radius=15.0, min_cells=10,
        min_count=1, n_shuffle=0, donors=["S3", "S3", "d2", ""], fit=_stub_pb_fit_recorder(seen4),
        leak_gate=False,
    )
    assert len(res4["blocks"]) == 3
    assert len(set(res4["blocks"])) == 3
    b_per_sample = [r["block"] for r in res4["samples"]]
    assert b_per_sample[0] == b_per_sample[1] == "S3"
    assert b_per_sample[2] == "d2"
    assert b_per_sample[3] not in ("S3", "d2", "")
    assert seen4[0]["counts"].shape == (len(genes), 6)

    # Case 2: 3 samples S0..S2 with donors ['S2', 'S2', ''] -> 2 distinct blocks
    s3_pairs = [_contact_sample(f"S{i}", genes, 200 + i) for i in range(3)]
    smps3 = [p[0] for p in s3_pairs]
    labs3 = [p[1] for p in s3_pairs]
    strata3 = [np.zeros(len(s["df"]), dtype=np.int64) for s in smps3]
    seen3: list = []
    res3 = xe.proximity._contact_compute(
        smps3, labs3, "Target", "Anchor", strata=strata3, radius=15.0, min_cells=10,
        min_count=1, n_shuffle=0, donors=["S2", "S2", ""], fit=_stub_pb_fit_recorder(seen3),
        leak_gate=False,
    )
    assert len(res3["blocks"]) == 2
    assert len(set(res3["blocks"])) == 2
    assert res3["samples"][0]["block"] == res3["samples"][1]["block"] == "S2"
    assert res3["samples"][2]["block"] != "S2"
    assert seen3[0]["counts"].shape == (len(genes), 4)

    # Case 3 (adversarial): donor literally named 'sample_S3' alongside donor 'S3'
    seen_adv: list = []
    res_adv = xe.proximity._contact_compute(
        smps4, labs4, "Target", "Anchor", strata=strata4, radius=15.0, min_cells=10,
        min_count=1, n_shuffle=0, donors=["S3", "S3", "sample_S3", ""],
        fit=_stub_pb_fit_recorder(seen_adv), leak_gate=False,
    )
    assert len(res_adv["blocks"]) == 3 and len(set(res_adv["blocks"])) == 3


# ── R10-18: Multi-sample _run_coloc handles empty long without NaN->int crash ─

def test_r10_18_coloc_multisample_empty_long_succeeds_cleanly(xe, D, monkeypatch):
    # Use primary sample (23 cells in tiny bundle: 12 TypeA + 11 TypeB) and add a second sample
    # with 12 TypeA + 11 TypeB: pooled = 24 & 22 >= _COLOC_MIN_FOCAL (20), per-sample = 12 & 11 < 20 -> long is empty!
    n_sub = min(24, len(D["df"]))
    assert n_sub >= 22
    n_a = 12
    n_b = n_sub - n_a
    sub_ids = [str(c) for c in D["df"].index[:n_sub]]
    if n_sub < len(D["df"]):
        xe.subset(cell_ids=sub_ids)

    rng = np.random.default_rng(18)
    genes = list(D["gene_names"])
    edf = pd.DataFrame(
        {"x_centroid": rng.uniform(0, 200, n_sub), "y_centroid": rng.uniform(0, 200, n_sub)},
        index=[f"coloc_e_{i}" for i in range(n_sub)],
    )
    eds = {
        "df": edf,
        "expr": sp.csr_matrix(np.ones((n_sub, len(genes)), dtype=np.float32)),
        "df_to_expr": np.arange(n_sub),
        "gene_names": genes,
        "gene_name_to_idx": dict(D["gene_name_to_idx"]),
        "x_offset": 1000.0,
        "cluster_methods": [],
        "_cell_type_labels_by_method": {
            "rctd": pd.Series(["TypeA"] * n_a + ["TypeB"] * n_b, index=edf.index)
        },
    }
    xe.EXTRA_DATASETS.append(eds)
    with xe._annot_lock:
        xe._annot_state["labels_rctd"] = pd.Series(["TypeA"] * n_a + ["TypeB"] * n_b, index=sub_ids)

    xe.controls._run_coloc("xenium", None, "celltype:rctd", 30.0, 10, placebo_focal=None, seed=0)
    assert xe._coloc_state["status"] == "done", xe._coloc_state["message"]
    res = xe._coloc_state["result"]
    assert res["n_samples"] == 2 and res["mode"] == "per-sample"
    assert len(res["long"]) == 0 and len(res["summary"]) == 0
    assert "PER-SAMPLE RESULTS" in xe._coloc_state["message"]

    # Downstream display / export helpers must also handle empty long & summary cleanly
    fig = xe.make_coloc_fig()
    assert fig.layout.annotations[0].text == "No pair has enough cells."
    assert xe.controls._coloc_records(res) == []
    csv_txt = xe.controls._coloc_csv(res)
    assert csv_txt.startswith("# Stratified co-localisation")


def test_r10_18_coloc_cohort_mode_empty_long_succeeds_cleanly(xe, D):
    # S = 4 samples (cohort mode), 6 cells per type per sample -> 24 pooled, 0 testable per sample
    n_sub = 12
    sub_ids = [str(c) for c in D["df"].index[:n_sub]]
    xe.subset(cell_ids=sub_ids)
    with xe._annot_lock:
        xe._annot_state["labels_rctd"] = pd.Series(["TypeA"] * 6 + ["TypeB"] * 6, index=sub_ids)

    genes = list(D["gene_names"])
    rng = np.random.default_rng(180)
    for k in range(3):
        edf = pd.DataFrame(
            {"x_centroid": rng.uniform(0, 200, n_sub), "y_centroid": rng.uniform(0, 200, n_sub)},
            index=[f"coh_{k}_{i}" for i in range(n_sub)],
        )
        xe.EXTRA_DATASETS.append({
            "sdata_path": f"/mem/coh_{k}.zarr",
            "df": edf,
            "expr": sp.csr_matrix(np.ones((n_sub, len(genes)), dtype=np.float32)),
            "df_to_expr": np.arange(n_sub),
            "gene_names": genes,
            "gene_name_to_idx": dict(D["gene_name_to_idx"]),
            "x_offset": 1000.0 * (k + 1),
            "cluster_methods": [],
            "_cell_type_labels_by_method": {
                "rctd": pd.Series(["TypeA"] * 6 + ["TypeB"] * 6, index=edf.index)
            },
        })

    xe.controls._run_coloc("xenium", None, "celltype:rctd", 30.0, 10, placebo_focal=None, seed=0)
    assert xe._coloc_state["status"] == "done", xe._coloc_state["message"]
    res = xe._coloc_state["result"]
    assert res["n_samples"] == 4 and res["mode"] == "cohort"
    assert len(res["long"]) == 0 and len(res["summary"]) == 0
