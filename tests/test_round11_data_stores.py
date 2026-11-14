"""Regression and stress tests for Round 11 Slice 1 (r11/fix-data-stores: D1..D7).

Covers:
- D1: `sdata_store._save_sdata_to_disk` while `subset()` is active preserves non-subset rows
  (`umap_1/2`, `split_umap_1/2`, `roi_*`, `niche`, `clust__*`, `cluster_*`, `ctl_*`), keeps
  non-subset ROI cells under `roi_only=True`, and preserves full-dataset `split_corrected_cols`
  via `_split_corrected_cols_original` (both Xenium and reseg).
- D2: SpatialData `.zarr` roundtrip invokes `_annot_load_from_sdata()` unconditionally when
  `sdata_path` is a directory (`startup.py` & `cb_data.load_sample`), discovers dynamic
  `cell_type_<method>` (`obs0`) and fallback `"cell_type"`, masks `{"", "nan", "<NA>", "None",
  "Unknown"}` (including in `dataview._extra_cell_type_labels`), gates zarr `rctd_weights` on
  `labels_rctd` coming from zarr, exports/overwrites `clust__*` as categorical strings +
  `uns["cluster_methods"]`, and discovers `clust__*` / reseg `cluster_*` in
  `loading.load_from_sdata_zarr`.
- D3: `cb_data.add_sample` and `cb_data.remove_extra_samples` check `_running_jobs()`,
  `_running_jobs()` reports `"adding sample"` when `_add_sample_lock.locked()`, `add_sample`
  fences on `jobs._dataset_gen`, `remove_extra_samples` acquires `_add_sample_lock`
  non-blockingly and prunes `h5ad._h5ad_loaded` (`loaded` + `manifest["sections"]`).
- D4: `caching._clean_cache_for_dataset()` calls `core._seg_forget_run()` (and resets
  non-running loaded seg states for deleted runs) and prints `.zarr` basename instead of `""`.
- D5: Reseg cached loaders (`_load_cached_baysor` fast/slow, `_load_cached_proseg` fast/slow,
  `imgseg._load_cached_imgseg`) read `split_corrected_cell_ids` and `split_corrected_cols`,
  backfill missing `split_umap_1/2` on `imgseg`, and exclude RCTD-dropped zero rows when
  recomputing SPLIT clusters + UMAP.
- D6: `h5ad._h5ad_convert` rejects `NaN`/`Inf` (and float32-overflowing values) up front with
  `_H5adError` without leaving partial conversion directories, and `subsetting.subset()` +
  `dataview._extra_gene_values()` clamp negative finite values with `np.maximum(..., 0.0)`
  before `np.log1p` (preserving `NaN` on uncovered SPLIT rows).
- D7: `subsetting.get_genes()` returns `sorted(dataview._panel_gene_names())`, excluding
  SpaGE-imputed genes.
"""
from __future__ import annotations

import gzip
import inspect
import json
import os
import shutil
import threading
import warnings

from dash import no_update
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture(autouse=True)
def _isolate_round11_state(xe):
    """Snapshot and restore global app state around every test."""
    from xenium_app import analysis, core, dataview, h5ad, jobs, regions, sdata_store, stats, subsetting

    try:
        subsetting.unsubset()
    except Exception:
        pass

    before_prov = list(xe._prov_recent)
    orig_cache_base = core.CACHE_BASE_DIR
    orig_annot_methods = dict(core._ANNOT_METHODS)
    orig_extra_providers = list(sdata_store._extra_obs_providers)
    orig_data = dict(core.DATA)
    orig_df = core.DATA["df"].copy()
    orig_extras = list(dataview.EXTRA_DATASETS)
    orig_h5ad = {
        "manifest": h5ad._h5ad_loaded.get("manifest"),
        "cache_dir": h5ad._h5ad_loaded.get("cache_dir"),
        "loaded": list(h5ad._h5ad_loaded.get("loaded") or []),
    }
    with core._annot_lock:
        orig_annot_state = dict(core._annot_state)
    with core._sdata_lock:
        orig_sdata_state = dict(core._sdata_state)
    with core._baysor_lock:
        orig_baysor = dict(core._baysor_state)
    with core._proseg_lock:
        orig_proseg = dict(core._proseg_state)
    with core._imgseg_lock:
        orig_imgseg = dict(core._imgseg_state)
    with core._spatial_lock:
        orig_spatial = dict(core._spatial_state)
    with analysis._de_lock:
        orig_de = dict(analysis._de_state)

    yield

    try:
        subsetting.unsubset()
    except Exception:
        pass
    core.CACHE_BASE_DIR = orig_cache_base
    core._ANNOT_METHODS.clear()
    core._ANNOT_METHODS.update(orig_annot_methods)
    sdata_store._extra_obs_providers[:] = orig_extra_providers

    restored_data = dict(orig_data)
    restored_data["df"] = orig_df.copy()
    dataview._install_dataset(restored_data)
    dataview.EXTRA_DATASETS[:] = orig_extras
    dataview._umap_df_cache.clear()
    dataview._umap_fig_cache.clear()

    h5ad._h5ad_loaded.clear()
    h5ad._h5ad_loaded.update(orig_h5ad)
    with core._annot_lock:
        core._annot_state.clear()
        core._annot_state.update(orig_annot_state)
    with core._sdata_lock:
        core._sdata_state.clear()
        core._sdata_state.update(orig_sdata_state)
    with core._baysor_lock:
        core._baysor_state.clear()
        core._baysor_state.update(orig_baysor)
    with core._proseg_lock:
        core._proseg_state.clear()
        core._proseg_state.update(orig_proseg)
    with core._imgseg_lock:
        core._imgseg_state.clear()
        core._imgseg_state.update(orig_imgseg)
    with core._spatial_lock:
        core._spatial_state.clear()
        core._spatial_state.update(orig_spatial)
    with analysis._de_lock:
        analysis._de_state.clear()
        analysis._de_state.update(orig_de)
    with core._save_sdata_lock:
        core._save_sdata_state.update({"status": "idle", "message": ""})
    regions._roi_forget_known()
    stats._design_sync()
    xe._prov_recent[:] = before_prov
    with jobs._jobs_lock:
        jobs._jobs.clear()


def _box_plot_space(x0: float, x1: float, y_um_min: float, y_um_max: float) -> list[list[float]]:
    py0, py1 = -float(y_um_max), -float(y_um_min)
    return [[float(x0), py0], [float(x1), py0], [float(x1), py1], [float(x0), py1]]


def _write_table_sdata(zarr_path: str, expr, obs: pd.DataFrame, genes: list[str],
                       *, layers: dict | None = None, uns: dict | None = None,
                       obsm: dict | None = None, is_imputed: list[bool] | None = None) -> str:
    import anndata as ad
    import spatialdata as sd
    from spatialdata.models import TableModel

    imp = [False] * len(genes) if is_imputed is None else list(is_imputed)
    var = pd.DataFrame({"is_imputed": imp}, index=pd.Index([str(g) for g in genes], name="gene"))
    adata = ad.AnnData(X=expr, obs=obs.copy(), var=var, layers=dict(layers or {}))
    for k, v in (uns or {}).items():
        adata.uns[k] = v
    for k, v in (obsm or {}).items():
        adata.obsm[k] = v
    try:
        adata = TableModel.parse(adata)
    except Exception:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sd.SpatialData(tables={"table": adata}).write(str(zarr_path), overwrite=True)
    return str(zarr_path)


# ── Source-level wiring & WebGL-default comment preservation ──────────────────

def test_round11_slice1_wiring_invariants(xe):
    from xenium_app import (
        caching, cb_data, dataview, h5ad, imgseg, loading, reseg, sdata_store, startup, subsetting,
    )

    # Preserve L79 comment in dataview.py from 065cf3f
    dv_lines = inspect.getsource(dataview).splitlines()
    assert "the default WebGL view draws all cells" in dv_lines[78]

    assert "_sdata_full_cells_df(" in inspect.getsource(sdata_store._save_sdata_to_disk)
    assert "_split_corrected_cols_original" in inspect.getsource(sdata_store._save_sdata_to_disk)
    assert "_annot_load_from_sdata" in inspect.getsource(startup)
    assert "_annot_load_from_sdata" in inspect.getsource(cb_data._reset_dataset_state)
    assert "_normalize_loaded_cluster_cols" in inspect.getsource(loading._load_from_xenium_sdata)
    assert "_seg_forget_run(" in inspect.getsource(caching._clean_cache_for_dataset)
    assert "_reseg_ensure_split_umap(" in inspect.getsource(reseg._load_cached_baysor)
    assert "_reseg_ensure_split_umap(" in inspect.getsource(reseg._load_cached_proseg)
    assert "_reseg_ensure_split_umap(" in inspect.getsource(imgseg._load_cached_imgseg)
    assert issubclass(h5ad._H5adError, ValueError)
    assert "_panel_gene_names()" in inspect.getsource(subsetting.get_genes)


# ── D1: _save_sdata_to_disk under active subset() (Xenium & Reseg) ────────────

def test_d1_save_sdata_under_active_subset_preserves_nonsubset_obs_roi_and_corrected_cols(xe, tmp_path):
    import spatialdata as sd
    import zarr as _zarr_mod
    from xenium_app import core, dataview, regions, sdata_store, subsetting

    genes = ["G0", "G1", "G2", "G3"]
    ids = [f"cell_{i}" for i in range(8)]
    rng = np.random.default_rng(11)
    X = sp.csr_matrix(rng.integers(1, 9, size=(8, 4)).astype(np.float32))
    X_corr = sp.csr_matrix(rng.integers(1, 5, size=(8, 4)).astype(np.float32))

    obs = pd.DataFrame(
        {
            "cell_id": ids,
            "x_centroid": [float(i * 10 + 5) for i in range(8)],
            "y_centroid": [10.0] * 8,
            "transcript_counts": np.asarray(X.sum(axis=1)).ravel(),
            "cell_area": [50.0] * 8,
            "nucleus_area": [20.0] * 8,
            "umap_1": [float(i) for i in range(8)],
            "umap_2": [float(i * 2) for i in range(8)],
            "split_umap_1": [float(i + 0.5) for i in range(8)],
            "split_umap_2": [float(i + 1.5) for i in range(8)],
            "niche": [f"N{i % 2}" for i in range(8)],
            "clust__graphclust": pd.Categorical([str((i % 2) + 1) for i in range(8)]),
            "ctl_exclusion": np.linspace(0.1, 0.8, 8),
        },
        index=pd.Index(ids, name="cell_id"),
    )
    src_zarr = _write_table_sdata(
        str(tmp_path / "base_xenium.zarr"),
        X,
        obs,
        genes,
        layers={"X_corrected": X_corr},
        uns={"cluster_methods": ["graphclust"], "split_corrected_cols": ["G0", "G1", "G2"]},
    )
    zw = _zarr_mod.open_group(src_zarr, mode="r+", use_consolidated=False)
    zw["tables"]["table"]["layers"]["X_corrected"].attrs["corrected_cols"] = ["G0", "G1", "G2"]

    ds = xe.load_from_sdata_zarr(src_zarr)
    dataview._install_dataset(ds)

    # ROI spans cell_0, cell_1 (in subset) AND cell_6, cell_7 (outside subset!)
    roi = {
        "name": "both_halves",
        "cls": "zone",
        "polygon_xy": [[0.0, -20.0], [20.0, -20.0], [20.0, 0.0], [0.0, 0.0]],
    }
    roi2 = {
        "name": "right_half",
        "cls": "zone",
        "polygon_xy": [[60.0, -20.0], [80.0, -20.0], [80.0, 0.0], [60.0, 0.0]],
    }
    with core._roi_lock:
        core._roi_state["rois"] = [roi, roi2]
        regions._roi_apply_metadata_to_df(core.DATA["df"], [roi, roi2])

    # Register a subset-aware extra obs provider simulating control scores computed on active cells
    ctl_scores = pd.Series(np.linspace(0.91, 0.94, 4), index=ids[:4])
    sdata_store._extra_obs_providers.append(
        lambda cdf, seg: pd.DataFrame({"ctl_exclusion": ctl_scores.reindex(cdf.index)}, index=cdf.index)
    )

    # Activate subset on first 4 cells ONLY, then mutate niche & split_corrected_cols on subset
    subsetting.subset(cell_ids=ids[:4])
    assert len(core.DATA["df"]) == 4
    assert core.DATA.get("_split_corrected_cols_original") == ["G0", "G1", "G2"]
    core.DATA["split_corrected_cols"] = ["G0"]  # subset-only SPLIT narrowed gene list
    core.DATA["df"] = core.DATA["df"].copy()
    core.DATA["df"]["niche"] = ["SubN0", "SubN1", "SubN0", "SubN1"]

    # 1) Full export while subset() is active
    out_full = str(tmp_path / "exported_full.zarr")
    sdata_store._save_sdata_to_disk(out_full, seg_source="xenium", roi_only=False)
    assert core._save_sdata_state["status"] == "done", core._save_sdata_state["message"]

    tbl_full = sd.read_zarr(out_full).tables["table"]
    assert tbl_full.n_obs == 8
    fobs = tbl_full.obs.set_axis(tbl_full.obs["cell_id"].astype(str))

    # Non-subset cells (cell_4..cell_7) preserve umap, split_umap, roi_zone, clust, niche, and ctl_exclusion!
    np.testing.assert_allclose(fobs.loc[ids[4:], "umap_1"].to_numpy(float), [4.0, 5.0, 6.0, 7.0])
    np.testing.assert_allclose(fobs.loc[ids[4:], "split_umap_1"].to_numpy(float), [4.5, 5.5, 6.5, 7.5])
    assert fobs.loc["cell_6", "roi_zone"] == "right_half"
    assert fobs.loc["cell_7", "roi_zone"] == "right_half"
    assert fobs.loc[ids[4:], "clust__graphclust"].astype(int).tolist() == [1, 2, 1, 2]
    assert fobs.loc[ids[:4], "niche"].astype(str).tolist() == ["SubN0", "SubN1", "SubN0", "SubN1"]
    assert fobs.loc[ids[4:], "niche"].astype(str).tolist() == ["N0", "N1", "N0", "N1"]
    np.testing.assert_allclose(fobs.loc[ids[:4], "ctl_exclusion"].to_numpy(float), ctl_scores.to_numpy())
    np.testing.assert_allclose(fobs.loc[ids[4:], "ctl_exclusion"].to_numpy(float), obs.loc[ids[4:], "ctl_exclusion"].to_numpy())

    # Full-dataset split_corrected_cols preserved (NOT subset ["G0"]!)
    assert list(tbl_full.uns["split_corrected_cols"]) == ["G0", "G1", "G2"]
    zr_out = _zarr_mod.open_group(out_full, mode="r", use_consolidated=False)
    assert list(zr_out["tables"]["table"]["layers"]["X_corrected"].attrs["corrected_cols"]) == ["G0", "G1", "G2"]

    # 2) roi_only=True export while subset() is active keeps BOTH subset AND non-subset ROI cells!
    out_roi = str(tmp_path / "exported_roi.zarr")
    sdata_store._save_sdata_to_disk(out_roi, seg_source="xenium", roi_only=True)
    assert core._save_sdata_state["status"] == "done", core._save_sdata_state["message"]
    tbl_roi = sd.read_zarr(out_roi).tables["table"]
    exported_ids = tbl_roi.obs["cell_id"].astype(str).tolist()
    assert exported_ids == ["cell_0", "cell_1", "cell_6", "cell_7"]

    # 3) unsubset() restores full-dataset split_corrected_cols
    subsetting.unsubset()
    assert core.DATA["split_corrected_cols"] == ["G0", "G1", "G2"]
    assert "_split_corrected_cols_original" not in core.DATA


def test_d1_save_sdata_reseg_under_active_subset_preserves_nonsubset_cells(xe, tmp_path):
    import spatialdata as sd
    from xenium_app import core, regions, reseg, sdata_store, subsetting

    genes = list(core.DATA["gene_names"][:4])
    ids = [f"rc_{i}" for i in range(6)]
    expr = sp.csr_matrix(np.ones((6, len(genes)), dtype=np.float32) * 3.0)
    corr = sp.csr_matrix(np.ones((6, len(genes)), dtype=np.float32) * 2.0)
    cdf = pd.DataFrame(
        {
            "x_centroid": [5.0, 15.0, 25.0, 35.0, 65.0, 75.0],
            "y_centroid": [5.0] * 6,
            "transcript_counts": [12.0] * 6,
            "cluster_kmeans_10": ["1", "1", "2", "2", "3", "3"],
            "umap_1": np.arange(6, dtype=float),
            "umap_2": np.arange(6, dtype=float) * 2.0,
        },
        index=pd.Index(ids, name="cell_id"),
    )
    bounds = {cid: ([x - 1, x + 1, x + 1, x - 1, x - 1], [4, 4, 6, 6, 4]) for cid, x in zip(ids, cdf["x_centroid"])}
    run_dir = tmp_path / "baysor_d1"
    run_dir.mkdir()
    zpath = reseg._build_reseg_sdata(cdf, bounds, expr, genes, "baysor", str(run_dir))

    # Attach X_corrected layer to reseg zarr
    _write_table_sdata(
        zpath, expr, cdf, genes,
        layers={"X_corrected": corr},
        uns={"split_corrected_cols": genes[:3]},
    )

    roi_left = {"name": "r_left", "cls": "zone", "polygon_xy": _box_plot_space(0.0, 20.0, 0.0, 10.0)}
    roi_right = {"name": "r_right", "cls": "zone", "polygon_xy": _box_plot_space(60.0, 80.0, 0.0, 10.0)}
    with core._roi_lock:
        core._roi_state["rois"] = [roi_left, roi_right]
        regions._roi_apply_metadata_to_df(cdf, [roi_left, roi_right])

    res = {
        "cells_df": cdf,
        "cell_bounds": bounds,
        "out_dir": str(run_dir),
        "expr": expr,
        "source": "baysor",
        "sdata_path": zpath,
        "split_corrected_expr": corr,
        "split_corrected_cols": list(genes[:3]),
    }
    with core._baysor_lock:
        core._baysor_state.update({"status": "done", "message": "Loaded", "result": res})

    # Subset using transcript_counts / x coordinate via ROI or area? Use min_transcripts + custom subset on Xenium + reseg:
    cdf.loc["rc_0":"rc_2", "cell_area"] = 100.0
    cdf.loc["rc_3":"rc_5", "cell_area"] = 10.0
    subsetting.subset(min_cell_area=50.0)
    assert len(res["cells_df"]) == 3
    res["split_corrected_cols"] = [genes[0]]

    out_roi = str(tmp_path / "reseg_roi_sub.zarr")
    sdata_store._save_sdata_to_disk(out_roi, seg_source="baysor", roi_only=True)
    assert core._save_sdata_state["status"] == "done", core._save_sdata_state["message"]
    tbl = sd.read_zarr(out_roi).tables["table"]
    # Both rc_0, rc_1 (in subset) AND rc_4, rc_5 (outside subset, in roi_right) are exported!
    assert tbl.obs.index.astype(str).tolist() == ["rc_0", "rc_1", "rc_4", "rc_5"]
    assert list(tbl.uns["split_corrected_cols"]) == list(genes[:3])
    np.testing.assert_allclose(tbl.obs.loc[["rc_4", "rc_5"], "umap_1"].to_numpy(float), [4.0, 5.0])

    subsetting.unsubset()
    assert res["split_corrected_cols"] == list(genes[:3])


# ── D2: SpatialData .zarr roundtrip annotations, placeholders, rctd_weights & clust__* ──

def test_d2_zarr_roundtrip_annotations_dynamic_methods_masking_and_rctd_gating(xe, tmp_path):
    from xenium_app import cb_data, core, dataview, h5ad, sdata_store

    genes = ["G0", "G1", "G2"]
    ids = ["101", "102", "103", "104", "105"]
    X = sp.csr_matrix(np.ones((5, 3), dtype=np.float32))
    obs = pd.DataFrame(
        {
            "cell_id": ids,
            "x_centroid": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y_centroid": [1.0, 2.0, 3.0, 4.0, 5.0],
            "transcript_counts": [10.0] * 5,
            "cell_type_celltypist": ["T_cell", "Unknown", "", "nan", "B_cell"],
            "cell_type_rctd": ["Cardiomyocyte", "<NA>", "None", "Unknown", "Fibroblast"],
            "cell_type_obs0": ["Author_A", "Unknown", "", "Author_B", "nan"],
        },
        index=pd.Index([f"row_{i}" for i in range(5)]),
    )
    weights = np.array(
        [
            [0.9, 0.1],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )
    zpath = _write_table_sdata(
        str(tmp_path / "annot_roundtrip.zarr"),
        X,
        obs,
        genes,
        uns={"rctd_weight_columns": ["Cardiomyocyte", "Fibroblast"]},
        obsm={"rctd_weights": weights},
    )

    # Simulate switching away from an .h5ad dataset into this .zarr via cb_data.load_sample
    core.CACHE_BASE_DIR = str(tmp_path / "empty_cache")
    os.makedirs(core.CACHE_BASE_DIR, exist_ok=True)
    h5ad._h5ad_register({"celltype_methods": {"obs0": "old_col"}, "sections": []}, str(tmp_path))

    res = cb_data.load_sample(1, None, zpath, 0, 0, 0, 0)
    assert res[0] == "Loaded."

    # Dynamic obs0 survived load_sample (h5ad_register(None) ran BEFORE _reset_dataset_state)
    assert "obs0" in core._ANNOT_METHODS
    with core._annot_lock:
        ct = core._annot_state.get("labels_celltypist")
        rc = core._annot_state.get("labels_rctd")
        o0 = core._annot_state.get("labels_obs0")
        rw = core._annot_state.get("rctd_weights_labels_rctd")

    assert ct is not None and ct.to_dict() == {"101": "T_cell", "105": "B_cell"}
    assert rc is not None and rc.to_dict() == {"101": "Cardiomyocyte", "105": "Fibroblast"}
    assert o0 is not None and o0.to_dict() == {"101": "Author_A", "104": "Author_B"}
    assert rw is not None and list(rw.index) == ids and list(rw.columns) == ["Cardiomyocyte", "Fibroblast"]

    # Extra-sample stored cell_type_* column ALSO strips "Unknown", "", "nan", "<NA>", "None"
    eds = {"df": core.DATA["df"].copy()}
    eds["df"]["cell_type_celltypist"] = ["T_cell", "Unknown", "", "nan", "None"]
    elabels, esrc = dataview._extra_cell_type_labels(eds, "celltypist")
    assert "stored column" in esrc
    assert elabels.dropna().to_dict() == {"101": "T_cell"}

    # Gating check: when labels_rctd already came from parquet cache, stale zarr rctd_weights is NOT attached!
    with core._annot_lock:
        core._annot_state.clear()
        core._annot_state.update({
            "status": "done",
            "message": "Cached parquet",
            "labels_rctd": pd.Series({"101": "ParquetType"}),
        })
    sdata_store._annot_load_from_sdata()
    with core._annot_lock:
        assert core._annot_state.get("rctd_weights_labels_rctd") is None

    # Fallback "cell_type" column when no cell_type_<method> column exists (reseg zarr reopened directly)
    zpath_fb = _write_table_sdata(
        str(tmp_path / "reseg_reopen.zarr"),
        sp.csr_matrix(np.ones((3, 2), dtype=np.float32)),
        pd.DataFrame(
            {
                "x_centroid": [1.0, 2.0, 3.0],
                "y_centroid": [1.0, 2.0, 3.0],
                "transcript_counts": [5.0, 5.0, 5.0],
                "cell_type": ["Endothelial", "Unknown", "Pericyte"],
            },
            index=pd.Index(["r1", "r2", "r3"]),
        ),
        ["G0", "G1"],
    )
    res_fb = cb_data.load_sample(1, None, zpath_fb, 1, 1, 1, 1)
    assert res_fb[0] == "Loaded."
    with core._annot_lock:
        fb_lbl = core._annot_state.get("labels_cell_type")
    assert fb_lbl is not None and fb_lbl.to_dict() == {"r1": "Endothelial", "r3": "Pericyte"}


def test_d2_clust_categorical_overwrite_and_reseg_cluster_discovery_roundtrip(xe, tmp_path):
    import spatialdata as sd
    from xenium_app import core, dataview, loading, sdata_store

    genes = ["G0", "G1", "G2"]
    ids = ["c0", "c1", "c2", "c3"]
    X = sp.csr_matrix(np.ones((4, 3), dtype=np.float32))
    obs = pd.DataFrame(
        {
            "cell_id": ids,
            "x_centroid": [1.0, 2.0, 3.0, 4.0],
            "y_centroid": [1.0, 2.0, 3.0, 4.0],
            "transcript_counts": [5.0] * 4,
            "clust__graphclust": pd.Categorical(["1", "1", "2", "2"]),
        },
        index=pd.Index(ids, name="cell_id"),
    )
    base_zarr = _write_table_sdata(
        str(tmp_path / "clust_base.zarr"),
        X, obs, genes, uns={"cluster_methods": ["graphclust"]},
    )
    ds = loading.load_from_sdata_zarr(base_zarr)
    dataview._install_dataset(ds)

    # Add a new in-app cluster set clust__joint_leiden (with 0 = unassigned) and export
    core.DATA["df"]["clust__joint_leiden"] = np.array([1, 2, 0, 2], dtype=np.int64)
    core.DATA["cluster_methods"] = ["graphclust", "joint_leiden"]

    exp1 = str(tmp_path / "clust_exp1.zarr")
    sdata_store._save_sdata_to_disk(exp1, seg_source="xenium", roi_only=False)
    assert core._save_sdata_state["status"] == "done"

    t1 = sd.read_zarr(exp1).tables["table"]
    assert list(t1.uns["cluster_methods"]) == ["graphclust", "joint_leiden"]
    assert isinstance(t1.obs["clust__joint_leiden"].dtype, pd.CategoricalDtype)
    assert t1.obs["clust__joint_leiden"].astype(str).tolist() == ["1", "2", "0", "2"]
    assert t1.obs["clust__joint_leiden"].astype(int).tolist() == [1, 2, 0, 2]

    # Reopen exp1, update clust__joint_leiden, and re-export -> must overwrite existing obs column!
    ds2 = loading.load_from_sdata_zarr(exp1)
    assert ds2["cluster_methods"] == ["graphclust", "joint_leiden"]
    assert ds2["df"]["clust__joint_leiden"].tolist() == [1, 2, 0, 2]
    dataview._install_dataset(ds2)
    core.DATA["df"]["clust__joint_leiden"] = np.array([3, 3, 1, 0], dtype=np.int64)

    exp2 = str(tmp_path / "clust_exp2.zarr")
    sdata_store._save_sdata_to_disk(exp2, seg_source="xenium", roi_only=False)
    t2 = sd.read_zarr(exp2).tables["table"]
    assert t2.obs["clust__joint_leiden"].astype(str).tolist() == ["3", "3", "1", "0"]

    # Reseg .zarr lacking uns["cluster_methods"] discovers cluster_<m> (excluding cluster_split_10)
    reseg_zarr = _write_table_sdata(
        str(tmp_path / "reseg_no_uns_cm.zarr"),
        X,
        pd.DataFrame(
            {
                "x_centroid": [1.0, 2.0, 3.0, 4.0],
                "y_centroid": [1.0, 2.0, 3.0, 4.0],
                "transcript_counts": [5.0] * 4,
                "cluster_kmeans_10": ["1", "2", "1", "2"],
                "cluster_banksy": ["A", "B", "", "A"],
                "cluster_split_10": ["1", "", "2", ""],
            },
            index=pd.Index(ids),
        ),
        genes,
    )
    rds = loading.load_from_sdata_zarr(reseg_zarr)
    assert "kmeans_10" in rds["cluster_methods"]
    assert "banksy" in rds["cluster_methods"]
    assert "split_10" not in rds["cluster_methods"]
    assert rds["df"]["clust__kmeans_10"].tolist() == [1, 2, 1, 2]
    assert rds["df"]["clust__banksy"].tolist() == [1, 2, 0, 1]


# ── D3: add_sample / remove_extra_samples locking, job guards, gen fence & h5ad prune ──

def test_d3_add_and_remove_sample_locking_job_guards_gen_fence_and_h5ad_prune(xe, tmp_path, monkeypatch):
    from xenium_app import analysis, cb_data, core, dataview, h5ad, jobs, loading

    extra_zarr = _write_table_sdata(
        str(tmp_path / "sample_b.zarr"),
        sp.csr_matrix(np.ones((3, 2), dtype=np.float32)),
        pd.DataFrame(
            {"x_centroid": [10.0, 20.0, 30.0], "y_centroid": [1.0, 2.0, 3.0], "transcript_counts": [4.0] * 3},
            index=pd.Index(["b1", "b2", "b3"]),
        ),
        ["G0", "G1"],
    )

    # (a) Running job blocks both add_sample and remove_extra_samples
    dataview.EXTRA_DATASETS[:] = [{"df": core.DATA["df"].head(2).copy(), "sdata_path": extra_zarr, "x_offset": 500.0}]
    with analysis._de_lock:
        analysis._de_state["status"] = "running"
    try:
        r_add = cb_data.add_sample(1, None, extra_zarr, 0)
        assert "Still running: differential expression" in r_add[0]
        r_rem = cb_data.remove_extra_samples(1, 0)
        assert r_rem[0] is no_update and "Still running: differential expression" in r_rem[1]
        assert len(dataview.EXTRA_DATASETS) == 1
    finally:
        with analysis._de_lock:
            analysis._de_state["status"] = "idle"

    # (b) Holding _add_sample_lock surfaces "adding sample" in _running_jobs() and blocks load_sample & remove_extra_samples
    assert cb_data._add_sample_lock.acquire(blocking=False)
    try:
        assert "adding sample" in cb_data._running_jobs()
        r_load = cb_data.load_sample(1, None, extra_zarr, 0, 0, 0, 0)
        assert "adding sample" in r_load[0]
        r_rem_locked = cb_data.remove_extra_samples(1, 0)
        assert r_rem_locked[0] is no_update and "adding sample" in r_rem_locked[1]
    finally:
        cb_data._add_sample_lock.release()

    # (c) Dataset generation fence aborts add_sample if primary dataset changes mid-load
    dataview.EXTRA_DATASETS.clear()
    real_loader = loading.load_from_sdata_zarr

    def _bumping_loader(p):
        out = real_loader(p)
        jobs._dataset_gen += 1
        return out

    monkeypatch.setattr(loading, "load_from_sdata_zarr", _bumping_loader)
    r_fenced = cb_data.add_sample(1, None, extra_zarr, 0)
    assert "aborted" in r_fenced[0].lower()
    assert dataview.EXTRA_DATASETS == []
    monkeypatch.setattr(loading, "load_from_sdata_zarr", real_loader)

    # (d/e) remove_extra_samples prunes h5ad._h5ad_loaded["loaded"] and manifest["sections"]
    sec0_dir = _write_table_sdata(
        str(tmp_path / "sec0.zarr"),
        sp.csr_matrix(np.ones((2, 2), dtype=np.float32)),
        pd.DataFrame({"x_centroid": [1.0, 2.0], "y_centroid": [1.0, 2.0], "transcript_counts": [2.0, 2.0]},
                     index=pd.Index(["s0_1", "s0_2"])),
        ["G0", "G1"],
    )
    sec1_dir = _write_table_sdata(
        str(tmp_path / "sec1.zarr"),
        sp.csr_matrix(np.ones((3, 2), dtype=np.float32)),
        pd.DataFrame({"x_centroid": [1.0, 2.0, 3.0], "y_centroid": [1.0, 2.0, 3.0], "transcript_counts": [2.0] * 3},
                     index=pd.Index(["s1_1", "s1_2", "s1_3"])),
        ["G0", "G1"],
    )
    ds0 = real_loader(sec0_dir)
    ds1 = real_loader(sec1_dir)
    ds1["x_offset"] = 600.0
    man = {
        "source": str(tmp_path / "multi.h5ad"),
        "n_cells": 5,
        "celltype_methods": {},
        "sections": [
            {"name": "sec0", "store": "sec0.zarr", "n_cells": 2},
            {"name": "sec1", "store": "sec1.zarr", "n_cells": 3},
        ],
    }
    dataview._install_dataset(ds0)
    h5ad._h5ad_register(man, str(tmp_path))
    dataview.EXTRA_DATASETS[:] = [ds1]
    assert len(h5ad._h5ad_loaded["loaded"]) == 2

    ver, msg = cb_data.remove_extra_samples(1, 4)
    assert ver == 5 and msg == "Extra samples removed."
    assert dataview.EXTRA_DATASETS == []
    assert h5ad._h5ad_loaded["loaded"] == [os.path.abspath(sec0_dir)]
    assert [s["name"] for s in h5ad._h5ad_loaded["manifest"]["sections"]] == ["sec0"]
    assert h5ad._h5ad_loaded["manifest"]["n_cells"] == 2

    # Now adding sec1 back via add_sample succeeds (not falsely rejected as already loaded)
    r_readd = cb_data.add_sample(1, None, sec1_dir, 5)
    assert r_readd[0].startswith("Added")
    assert len(dataview.EXTRA_DATASETS) == 1


def test_d3_concurrent_add_remove_and_load_sample_stress(xe, tmp_path):
    from xenium_app import cb_data, dataview

    stores = []
    for k in range(4):
        p = _write_table_sdata(
            str(tmp_path / f"stress_{k}.zarr"),
            sp.csr_matrix(np.ones((3, 2), dtype=np.float32)),
            pd.DataFrame(
                {"x_centroid": [1.0, 2.0, 3.0], "y_centroid": [1.0, 2.0, 3.0], "transcript_counts": [3.0] * 3},
                index=pd.Index([f"k{k}_{i}" for i in range(3)]),
            ),
            ["G0", "G1"],
        )
        stores.append(p)

    errors = []
    barrier = threading.Barrier(6)

    def _worker_add(idx):
        try:
            barrier.wait(timeout=5.0)
            for _ in range(5):
                cb_data.add_sample(1, None, stores[idx % len(stores)], 0)
        except Exception as exc:
            errors.append(exc)

    def _worker_remove():
        try:
            barrier.wait(timeout=5.0)
            for _ in range(5):
                cb_data.remove_extra_samples(1, 0)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker_add, args=(i,)) for i in range(4)] + [
        threading.Thread(target=_worker_remove) for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)

    assert errors == []
    assert not cb_data._add_sample_lock.locked()
    # Every remaining entry in EXTRA_DATASETS must have a unique _ds_key
    keys = [dataview._ds_key(d) for d in dataview.EXTRA_DATASETS]
    assert len(keys) == len(set(keys))


# ── D4: _clean_cache_for_dataset forgets seg run hooks/state & prints .zarr basename ──

def test_d4_clean_cache_forgets_seg_runs_and_formats_zarr_dataset_name(xe, tmp_path, capsys):
    import hashlib
    import pathlib
    from xenium_app import annotation, caching, core, dataview

    cache_root = tmp_path / "cache_d4"
    cache_root.mkdir()
    core.CACHE_BASE_DIR = str(cache_root)

    # Part 1: Xenium dataset with cached Baysor/Proseg/Imgseg run dirs + registered side state
    xenium_dir = tmp_path / "xenium_sample_d4"
    xenium_dir.mkdir()
    core.DATA["data_dir"] = str(xenium_dir)
    assert caching._dataset_name() == "xenium_sample_d4"

    run_dirs = {}
    for tool, rid in (("baysor", "deadbeef"), ("proseg", "beadfeed"), ("imgseg", "cafebabe")):
        rdir = pathlib.Path(caching._seg_run_dir(tool, rid))
        rdir.mkdir()
        with open(rdir / "params.json", "w") as fh:
            json.dump({"dataset_hash": hashlib.md5(caching._dataset_key().encode()).hexdigest()[:8]}, fh)
        run_dirs[tool] = str(rdir)

    akey = annotation._labels_key_for_method("celltypist", {"out_dir": run_dirs["baysor"], "source": "baysor"})
    with core._annot_lock:
        core._annot_state[akey] = pd.Series(["CM"], index=["c0"])
    with core._spatial_lock:
        core._spatial_state.update({"status": "done", "message": "ok", "result": {"out_dir": run_dirs["baysor"]}})
    with core._baysor_lock:
        core._baysor_state.update({
            "status": "done",
            "message": "Loaded",
            "result": {"out_dir": run_dirs["baysor"], "cells_df": core.DATA["df"].copy()},
        })

    forgotten = []
    caching._seg_commit_hooks.append(forgotten.append)
    try:
        msg_xenium = caching._clean_cache_for_dataset()
    finally:
        caching._seg_commit_hooks.remove(forgotten.append)

    assert "Removed 3 cache entries for xenium_sample_d4." == msg_xenium
    assert {os.path.abspath(p) for p in forgotten} == {os.path.abspath(p) for p in run_dirs.values()}
    for tool in ("baysor", "proseg", "imgseg"):
        assert not os.path.exists(run_dirs[tool])
    with core._annot_lock:
        assert akey not in core._annot_state
    with core._spatial_lock:
        assert core._spatial_state.get("result") is None
    with core._baysor_lock:
        assert core._baysor_state["status"] == "idle"
        assert core._baysor_state.get("result") is None

    # Part 2: SpatialData .zarr dataset (data_dir=None) formats .zarr basename instead of ''
    zpath = _write_table_sdata(
        str(tmp_path / "cardiac_export.zarr"),
        sp.csr_matrix(np.ones((3, 2), dtype=np.float32)),
        pd.DataFrame({"x_centroid": [1.0, 2.0, 3.0], "y_centroid": [1.0, 2.0, 3.0], "transcript_counts": [3.0] * 3},
                     index=pd.Index(["a", "b", "c"])),
        ["G0", "G1"],
    )
    dataview._install_dataset(xe.load_from_sdata_zarr(zpath))
    assert caching._dataset_name() == ""  # .zarr has data_dir=None

    ztag = hashlib.md5(caching._dataset_key().encode()).hexdigest()[:8]
    (cache_root / f"celltypist_{ztag}.parquet").write_bytes(b"stub")

    capsys.readouterr()
    msg_zarr = caching._clean_cache_for_dataset()
    out_txt = capsys.readouterr().out

    assert msg_zarr == "Removed 1 cache entries for cardiac_export.zarr."
    assert "dataset cardiac_export.zarr" in out_txt
    assert "for ." not in msg_zarr


# ── D5: Reseg cached loaders backfill imgseg SPLIT UMAP & skip RCTD-dropped rows ──

def _make_split_reseg_fixture(run_dir: str, tool: str, genes: list[str], n_total: int = 18, n_cov: int = 14):
    import zarr as _zarr_mod
    from xenium_app import reseg

    rng = np.random.default_rng(42)
    ids = [f"rc_{i}" for i in range(n_total)]
    raw_dense = rng.integers(2, 10, size=(n_total, len(genes))).astype(np.float32)
    expr = sp.csr_matrix(raw_dense)

    corr_dense = np.zeros_like(raw_dense)
    corr_dense[:n_cov] = rng.integers(1, 8, size=(n_cov, len(genes))).astype(np.float32)
    # Last (n_total - n_cov) rows are all-zero (dropped by RCTD!) while raw expr > 0
    corr = sp.csr_matrix(corr_dense)

    cdf = pd.DataFrame(
        {
            "x_centroid": np.linspace(5.0, 90.0, n_total),
            "y_centroid": np.linspace(5.0, 90.0, n_total),
            "transcript_counts": np.asarray(expr.sum(axis=1)).ravel(),
            "cell_volume": np.full(n_total, 120.0),
            "cell_area": np.full(n_total, 40.0),
            "cluster_kmeans_10": [str((i % 3) + 1) for i in range(n_total)],
            "umap_1": np.linspace(-1.0, 1.0, n_total),
            "umap_2": np.linspace(1.0, -1.0, n_total),
            # Missing / all-NaN split_umap triggers backfill!
            "split_umap_1": np.full(n_total, np.nan),
            "split_umap_2": np.full(n_total, np.nan),
        },
        index=pd.Index(ids, name="cell_id"),
    )
    bounds = {
        cid: ([float(x - 1), float(x + 1), float(x + 1), float(x - 1), float(x - 1)],
              [float(y - 1), float(y - 1), float(y + 1), float(y + 1), float(y - 1)])
        for cid, x, y in zip(ids, cdf["x_centroid"], cdf["y_centroid"])
    }
    zpath = reseg._build_reseg_sdata(cdf, bounds, expr, genes, tool, run_dir)
    _write_table_sdata(
        zpath,
        expr,
        cdf,
        genes,
        layers={"X_corrected": corr},
        uns={
            "split_corrected_cols": list(genes[:3]),
            "split_corrected_cell_ids": ids[:n_cov],
        },
    )
    zw = _zarr_mod.open_group(zpath, mode="r+", use_consolidated=False)
    zw["tables"]["table"]["layers"]["X_corrected"].attrs["corrected_cols"] = list(genes[:3])
    return ids, zpath, raw_dense


def test_d5_cached_baysor_proseg_and_imgseg_backfill_split_umap_excluding_rctd_dropped_rows(
    xe, tmp_path, monkeypatch
):
    from xenium_app import core, imgseg, reseg

    genes = list(core.DATA["gene_names"][:6])
    assert len(genes) >= 4
    n_total, n_cov = 18, 14

    def _assert_split_backfill(state_dict, lock, expected_cols):
        with lock:
            assert state_dict["status"] == "done", state_dict.get("message")
            res = state_dict["result"]
        cov_ids = [f"rc_{i}" for i in range(n_cov)]
        drop_ids = [f"rc_{i}" for i in range(n_cov, n_total)]
        assert set(res["split_corrected_cell_ids"]) == set(cov_ids)
        assert len(res["split_corrected_cell_ids"]) == len(cov_ids)
        assert res["split_corrected_cols"] == expected_cols
        rdf = res["cells_df"]
        assert rdf.loc[cov_ids, "split_umap_1"].notna().all()
        assert rdf.loc[cov_ids, "split_umap_2"].notna().all()
        assert rdf.loc[cov_ids, "cluster_split_10"].notna().all()
        # RCTD-dropped all-zero rows MUST remain NaN / NA (never embedded or clustered!)
        assert rdf.loc[drop_ids, "split_umap_1"].isna().all()
        assert rdf.loc[drop_ids, "split_umap_2"].isna().all()
        assert rdf.loc[drop_ids, "cluster_split_10"].isna().all()

    # 1) Baysor fast path
    b_fast = tmp_path / "baysor_fast_d5"
    b_fast.mkdir()
    _make_split_reseg_fixture(str(b_fast), "baysor", genes, n_total, n_cov)
    reseg._load_cached_baysor(str(b_fast))
    _assert_split_backfill(core._baysor_state, core._baysor_lock, list(genes[:3]))

    # 2) Proseg fast path
    p_fast = tmp_path / "proseg_fast_d5"
    p_fast.mkdir()
    _make_split_reseg_fixture(str(p_fast), "proseg", genes, n_total, n_cov)
    reseg._load_cached_proseg(str(p_fast))
    _assert_split_backfill(core._proseg_state, core._proseg_lock, list(genes[:3]))

    # 3) Image segmentation cached loader (previously lacked _split_umap_missing backfill entirely!)
    i_dir = tmp_path / "imgseg_d5"
    i_dir.mkdir()
    _make_split_reseg_fixture(str(i_dir), "imgseg", genes, n_total, n_cov)
    imgseg._load_cached_imgseg(str(i_dir))
    _assert_split_backfill(core._imgseg_state, core._imgseg_lock, list(genes[:3]))

    # 4) Baysor slow CSV path & Proseg slow CSV path (force _fast_load_reseg_zarr to raise)
    b_slow = tmp_path / "baysor_slow_d5"
    b_slow.mkdir()
    ids, _, raw_dense = _make_split_reseg_fixture(str(b_slow), "baysor", genes, n_total, n_cov)
    seg_rows = []
    for i, cid in enumerate(ids):
        for g_idx, g in enumerate(genes):
            for _ in range(int(raw_dense[i, g_idx])):
                seg_rows.append({"cell": cid, "x": float(i * 5 + 5), "y": float(i * 5 + 5), "gene": g})
    pd.DataFrame(seg_rows).to_csv(b_slow / "segmentation.csv", index=False)

    p_slow = tmp_path / "proseg_slow_d5"
    p_slow.mkdir()
    _make_split_reseg_fixture(str(p_slow), "proseg", genes, n_total, n_cov)
    cm_df = pd.DataFrame({
        "cell": ids,
        "centroid_x": np.linspace(5.0, 90.0, n_total),
        "centroid_y": np.linspace(5.0, 90.0, n_total),
        "population": raw_dense.sum(axis=1),
        "volume": np.full(n_total, 120.0),
    })
    with gzip.open(p_slow / "cell-metadata.csv.gz", "wt") as gz:
        cm_df.to_csv(gz, index=False)

    monkeypatch.setattr(reseg, "_fast_load_reseg_zarr", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("force slow")))
    reseg._load_cached_baysor(str(b_slow))
    _assert_split_backfill(core._baysor_state, core._baysor_lock, list(genes[:3]))

    reseg._load_cached_proseg(str(p_slow))
    _assert_split_backfill(core._proseg_state, core._proseg_lock, list(genes[:3]))


# ── D6: .h5ad NaN/Inf upfront validation + negative finite clamp before log1p ──

def _write_test_h5ad(path: str, X, *, dense: bool = False) -> str:
    import anndata as ad

    mat = np.asarray(X, dtype=np.float64) if dense else sp.csr_matrix(X, dtype=np.float64)
    n, m = mat.shape
    obs = pd.DataFrame(
        {
            "x": np.arange(n, dtype=float) * 10.0,
            "y": np.arange(n, dtype=float) * 5.0,
            "section": ["secA" if i < n // 2 else "secB" for i in range(n)],
        },
        index=pd.Index([f"c_{i}" for i in range(n)]),
    )
    var = pd.DataFrame(index=pd.Index([f"G{j}" for j in range(m)]))
    ad.AnnData(X=mat, obs=obs, var=var).write_h5ad(path)
    return path


def test_d6_h5ad_nan_inf_rejected_without_partial_cache_dir(xe, tmp_path):
    from xenium_app import core, h5ad

    cache_root = tmp_path / "h5ad_cache_d6"
    cache_root.mkdir()
    core.CACHE_BASE_DIR = str(cache_root)
    mapping = {"x": "x", "y": "y", "section": "section", "counts": "X"}

    cases = [
        ("sparse_nan.h5ad", np.array([[1.0, 2.0], [3.0, 4.0], [0.0, np.nan], [2.0, 1.0]]), False),
        ("sparse_inf.h5ad", np.array([[1.0, 2.0], [3.0, np.inf], [1.0, 0.0], [2.0, 1.0]]), False),
        ("dense_neginf.h5ad", np.array([[1.0, 2.0], [3.0, 4.0], [-np.inf, 1.0], [2.0, 1.0]]), True),
        ("overflow_f32.h5ad", np.array([[1.0, 2.0], [3.0, 4.0], [1e308, 1.0], [2.0, 1.0]]), False),
    ]
    for fname, arr, is_dense in cases:
        fpath = _write_test_h5ad(str(tmp_path / fname), arr, dense=is_dense)
        expected_cdir = h5ad._h5ad_cache_dir(fpath, mapping)
        with pytest.raises(h5ad._H5adError, match="non-finite|overflow"):
            h5ad._h5ad_convert(fpath, mapping, block_nnz=2)
        assert not os.path.exists(expected_cdir), f"Partial cache dir left behind for {fname}: {expected_cdir}"
    assert os.listdir(cache_root) == []

    # sanity check: h5py raw injection on existing file also cleans up uncommitted dir
    good_path = _write_test_h5ad(str(tmp_path / "valid_neg.h5ad"), np.array([[-2.5, 1.0], [0.0, 3.0], [-1.0, 2.0], [4.0, 0.5]]), dense=False)
    man, cdir = h5ad._h5ad_convert(good_path, mapping)
    assert os.path.isfile(os.path.join(cdir, "manifest.json"))
    assert man["counts"]["n_negative"] == 2
    assert h5py.is_hdf5(good_path)


def test_d6_negative_finite_values_clamped_before_log1p_in_subset_and_extra_gene_values(xe):
    from xenium_app import core, dataview, subsetting

    genes = ["G0", "G1"]
    ids = ["n0", "n1", "n2", "n3"]
    # G0 has negative values: -5.0 (< -1!), -1.0 (== -1!), -0.25, and +4.0
    expr = sp.csr_matrix(np.array([[-5.0, 1.0], [-1.0, 1.0], [-0.25, 1.0], [4.0, 1.0]], dtype=np.float32))
    df = pd.DataFrame(
        {
            "x_centroid": [1.0, 2.0, 3.0, 4.0],
            "y_centroid": [1.0, 2.0, 3.0, 4.0],
            "transcript_counts": [1.0, 1.0, 1.0, 5.0],
            "umap_1": [0.0, 1.0, 2.0, 3.0],
            "umap_2": [0.0, 1.0, 2.0, 3.0],
        },
        index=pd.Index(ids),
    )
    ds = {
        "df": df,
        "expr": expr,
        "gene_names": genes,
        "gene_name_to_idx": {"G0": 0, "G1": 1},
        "gene_var": pd.DataFrame({"is_imputed": [False, False]}, index=pd.Index(genes)),
        "imputed_genes": [],
        "cluster_methods": [],
        "cell_bounds": {},
        "nuc_bounds": {},
        "metadata": {"run_name": "neg_h5ad", "num_cells": 4, "num_transcripts": 8},
        "data_dir": None,
        "sdata_path": "/tmp/fake_neg_primary.zarr",
        "df_to_expr": np.arange(4, dtype=np.int64),
    }
    dataview._install_dataset(ds)

    # Also attach a loaded Baysor run with the same negative values to hit subsetting.py:216!
    reseg_res = {
        "cells_df": df.copy(),
        "expr": expr.copy(),
        "cell_bounds": {},
        "out_dir": "/tmp/fake_baysor_neg",
        "source": "baysor",
    }
    with core._baysor_lock:
        core._baysor_state.update({"status": "done", "message": "Loaded", "result": reseg_res})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        subsetting.subset(gene="G0", max_expr=0.5)
        # Extra sample with negative corrected expression + 1 uncovered SPLIT row (row 2: corr=0 while raw>0)
        eds = {
            "df": df.copy(),
            "expr": sp.csr_matrix(np.array([[2.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]], dtype=np.float32)),
            "split_corrected_expr": sp.csr_matrix(
                np.array([[-5.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [3.0, 1.0]], dtype=np.float32)
            ),
            "gene_names": genes,
            "gene_name_to_idx": {"G0": 0, "G1": 1},
            "df_to_expr": np.arange(4, dtype=np.int64),
        }
        ev = dataview._extra_gene_values(eds, "G0", use_corrected=True)

    runtime_warns = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warns == [], f"Unexpected RuntimeWarnings from log1p: {runtime_warns}"

    # n0 (-5.0), n1 (-1.0), n2 (-0.25) clamp to log1p(0)=0.0 <= 0.5 -> ALL THREE KEPT!
    assert core.DATA["df"].index.tolist() == ["n0", "n1", "n2"]
    assert reseg_res["cells_df"].index.tolist() == ["n0", "n1", "n2"]

    # Extra sample: rows 0 & 1 (-5.0, -1.0) clamp to 0.0, row 2 (SPLIT-uncovered!) stays NaN, row 3 -> log1p(3.0)
    assert ev[0] == pytest.approx(0.0)
    assert ev[1] == pytest.approx(0.0)
    assert np.isnan(ev[2])
    assert ev[3] == pytest.approx(np.log1p(3.0))


# ── D7: subsetting.get_genes() excludes SpaGE-imputed genes ──────────────────

def test_d7_get_genes_excludes_spage_imputed_genes(xe, tmp_path):
    from xenium_app import dataview, subsetting

    panel = ["MYH7", "TNNT2", "ACTN2"]
    imputed = ["IMP_GENE1", "IMP_GENE2"]
    all_genes = panel + imputed
    ds = {
        "df": pd.DataFrame(
            {"x_centroid": [1.0, 2.0], "y_centroid": [1.0, 2.0], "transcript_counts": [5.0, 6.0]},
            index=pd.Index(["c1", "c2"]),
        ),
        "expr": sp.csr_matrix(np.ones((2, len(all_genes)), dtype=np.float32)),
        "gene_names": list(all_genes),
        "gene_name_to_idx": {g: i for i, g in enumerate(all_genes)},
        "gene_var": pd.DataFrame(
            {"is_imputed": [False, False, False, True, True]},
            index=pd.Index(all_genes),
        ),
        "imputed_genes": list(imputed),
        "cluster_methods": [],
        "cell_bounds": {},
        "nuc_bounds": {},
        "metadata": {"run_name": "d7", "num_cells": 2, "num_transcripts": 11},
        "data_dir": None,
        "sdata_path": "/tmp/fake_d7.zarr",
        "df_to_expr": np.arange(2, dtype=np.int64),
    }
    dataview._install_dataset(ds)

    out_file = tmp_path / "panel_genes.txt"
    got_panel = subsetting.get_genes(file=str(out_file))
    assert got_panel == sorted(panel)
    assert set(got_panel).isdisjoint(imputed)
    assert out_file.read_text().splitlines() == sorted(panel)
