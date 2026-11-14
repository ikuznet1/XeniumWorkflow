"""Regression and stress tests for Round 10 (data-reseg slice: R10-1 .. R10-7).

Covers:
- R10-1 & R10-6: Shared obs placeholder normalization (`regions._normalize_obs_placeholders`)
  across `loading.py` and `reseg.py` (`_fast_load_reseg_zarr`, `_load_zarr_obs_into_df`, and
  slow adata cache paths), plus clearing placeholder strings in `regions._roi_apply_metadata_to_df`
  without clobbering foreign ROI labels.
- R10-7: Live `_run_baysor`, `_run_proseg`, and `_run_imgseg` apply saved ROIs under
  `core._roi_lock` and persist `roi_*` columns to the newly written `spatialdata_<tool>.zarr`.
- R10-2: Non-blocking `_add_sample_lock` + post-load `_ds_key` / path duplicate check in
  `cb_data.add_sample` preventing concurrent duplicate sample appends.
- R10-3: `_running_jobs()` guard at top of `cb_data.start_sdata_clear` before claiming
  `_sdata_state` or deleting the SpatialData cache store.
- R10-4: `cb_data._h5ad_install` applies autoloaded ROIs to `EXTRA_DATASETS` (sections 2+)
  at their respective `x_offset`s and clears UMAP caches.
- R10-5: `sdata_store._save_sdata_to_disk(..., roi_only=True)` when session `rois` is empty:
  filters on stored non-NA `roi_*` columns when present, otherwise appends a clear notice and
  records `roi_filter_applied` and `n_cells_exported` in provenance; plus preserving
  `split_corrected_cols` across `sdata_store._save_sdata_to_disk` AnnData table rewrites.
"""
from __future__ import annotations

import inspect
import json
import os
import shutil
import threading
import time

from dash import no_update
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import xe_testlib as tl


@pytest.fixture(autouse=True)
def _preserve_session_provenance_and_sdata(xe):
    """Keep session provenance clean and restore DATA / _sdata_state / h5ad state after each test."""
    from xenium_app import core, h5ad

    before_prov = list(xe._prov_recent)
    orig_data = dict(core.DATA)
    orig_df = core.DATA["df"].copy()
    with core._sdata_lock:
        orig_sdata_state = dict(core._sdata_state)
    orig_h5ad = dict(h5ad._h5ad_loaded)
    yield
    xe._prov_recent[:] = before_prov
    if core.DATA.get("data_dir") != orig_data.get("data_dir") or core.DATA.get("sdata_path") != orig_data.get("sdata_path"):
        orig_copy = dict(orig_data)
        orig_copy["df"] = orig_df.copy()
        xe._install_dataset(orig_copy)
    h5ad._h5ad_loaded.clear()
    h5ad._h5ad_loaded.update(orig_h5ad)
    with core._sdata_lock:
        core._sdata_state.clear()
        core._sdata_state.update(orig_sdata_state)
    xe._roi_forget_known()


def _box_plot_space(x0: float, x1: float, y_um_min: float, y_um_max: float) -> list[list[float]]:
    """Polygon in spatial plot coordinates (x = x_centroid + x_offset, y = -y_centroid)."""
    py0, py1 = -float(y_um_max), -float(y_um_min)
    return [[float(x0), py0], [float(x1), py0], [float(x1), py1], [float(x0), py1]]


# ── Source-level wiring checks ───────────────────────────────────────────────

def test_round10_shared_helpers_wired_at_all_call_sites():
    from xenium_app import cb_data, imgseg, loading, regions, reseg, sdata_store

    assert hasattr(regions, "_normalize_obs_placeholders")
    assert "_normalize_obs_placeholders(" in inspect.getsource(loading._load_from_xenium_sdata)
    for fn in (
        reseg._load_zarr_obs_into_df,
        reseg._fast_load_reseg_zarr,
        reseg._load_cached_baysor,
        reseg._load_cached_proseg,
    ):
        assert "_normalize_obs_placeholders(" in inspect.getsource(fn), fn.__name__

    for live_fn in (reseg._run_baysor, reseg._run_proseg, imgseg._run_imgseg):
        src = inspect.getsource(live_fn)
        assert "_roi_apply_metadata_to_df(" in src, live_fn.__name__
        assert "_update_reseg_zarr_obs(" in src, live_fn.__name__

    assert "split_corrected_cols" in inspect.getsource(sdata_store._save_sdata_to_disk)
    assert "split_corrected_cols" in inspect.getsource(loading._load_from_xenium_sdata)

    clear_src = inspect.getsource(cb_data.start_sdata_clear)
    assert clear_src.index("_running_jobs()") < clear_src.index("clear_sdata_cache()")

    add_src = inspect.getsource(cb_data.add_sample)
    assert "_add_sample_lock.acquire(blocking=False)" in add_src
    assert "_add_sample_lock.release()" in add_src
    assert "_ds_key(" in add_src

    h5ad_src = inspect.getsource(cb_data._h5ad_install)
    assert "_roi_apply_to_sample(" in h5ad_src
    assert "_umap_df_cache.clear()" in h5ad_src
    assert "_umap_fig_cache.clear()" in h5ad_src

    save_src = inspect.getsource(sdata_store._save_sdata_to_disk)
    assert "roi_filter_applied" in save_src
    assert "n_cells_exported" in save_src


# ── R10-1 & R10-6: Obs placeholder normalization & ROI metadata application ──

def test_normalize_obs_placeholders_handles_all_dtypes_and_preserves_real_values():
    from xenium_app import regions

    df = pd.DataFrame(
        {
            "roi_zone": np.array(["tumor", "", "nan", "<NA>", "stroma"], dtype=str),
            "roi_bytes": np.array([b"core", b"", b"nan", b"<NA>", b"margin"], dtype=object),
            "roi_cat": pd.Categorical(["alpha", "", "<NA>", np.nan, "beta"]),
            "roi_obj_none": np.array(["left", None, "", "nan", "right"], dtype=object),
            "niche": np.array(["1", "", "nan", "<NA>", "2"], dtype=object),
            "cluster_split_10": np.array(["1", "", "3", "<NA>", "nan"], dtype=object),
            "other_str": np.array(["", "nan", "<NA>", "keep", "keep"], dtype=object),
            "x_centroid": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        index=["c0", "c1", "c2", "c3", "c4"],
    )
    out = regions._normalize_obs_placeholders(df)
    assert out is df

    assert list(df["roi_zone"].isna()) == [False, True, True, True, False]
    assert list(df["roi_zone"].dropna()) == ["tumor", "stroma"]

    assert list(df["roi_bytes"].isna()) == [False, True, True, True, False]
    assert list(df["roi_bytes"].dropna()) == ["core", "margin"]

    assert list(df["roi_cat"].isna()) == [False, True, True, True, False]
    assert list(df["roi_cat"].dropna()) == ["alpha", "beta"]

    # Python None stays NA (never turned into literal string "None")
    assert list(df["roi_obj_none"].isna()) == [False, True, True, True, False]
    assert list(df["roi_obj_none"].dropna()) == ["left", "right"]

    assert list(df["niche"].isna()) == [False, True, True, True, False]
    assert list(df["niche"].dropna()) == ["1", "2"]

    assert list(df["cluster_split_10"].isna()) == [False, True, False, True, True]
    assert list(df["cluster_split_10"].dropna()) == ["1", "3"]

    # Non-target columns are never touched
    assert list(df["other_str"]) == ["", "nan", "<NA>", "keep", "keep"]

    # Numeric niche columns are left numeric (not stringified to "0.0")
    df_num = pd.DataFrame({"niche": [0.0, 1.0, np.nan]}, index=["a", "b", "c"])
    regions._normalize_obs_placeholders(df_num)
    assert pd.api.types.is_float_dtype(df_num["niche"])
    assert df_num.loc["a", "niche"] == 0.0 and pd.isna(df_num.loc["c", "niche"])


def test_roi_apply_metadata_clears_placeholders_while_preserving_foreign_labels():
    from xenium_app import regions

    regions._roi_forget_known()
    df = pd.DataFrame(
        {
            "x_centroid": [1.0, 2.0, 10.0, 20.0, 30.0],
            "y_centroid": [1.0, 1.0, 1.0, 1.0, 1.0],
            # Foreign class (no polygon in session) with placeholders mixed in
            "roi_foreign": np.array(["ext_a", "", "nan", "<NA>", "ext_b"], dtype=object),
            # Same class as new polygon: has foreign label "prior" + placeholders
            "roi_zone": pd.Categorical(["prior", "", "nan", "<NA>", "z1"]),
        },
        index=["c0", "c1", "c2", "c3", "c4"],
    )
    roi = {"name": "z1", "cls": "zone", "polygon_xy": _box_plot_space(8.0, 12.0, 0.0, 2.0)}
    regions._roi_apply_metadata_to_df(df, [roi])

    # Foreign labels survive; placeholders are cleared to NA
    assert df.loc["c0", "roi_foreign"] == "ext_a"
    assert df.loc["c4", "roi_foreign"] == "ext_b"
    assert df.loc[["c1", "c2", "c3"], "roi_foreign"].isna().all()

    # In roi_zone: "prior" (foreign) survives, "z1" re-derived onto c2 only, placeholders cleared
    assert df.loc["c0", "roi_zone"] == "prior"
    assert df.loc["c2", "roi_zone"] == "z1"
    assert df.loc[["c1", "c3", "c4"], "roi_zone"].isna().all()


def test_reseg_zarr_roi_placeholders_roundtrip_and_roi_only_export(xe, D, tmp_path):
    """R10-1 + R10-6 end-to-end: _update_reseg_zarr_obs writes '' for non-ROI cells;
    both _fast_load_reseg_zarr and _load_zarr_obs_into_df restore '' -> pd.NA, and
    subsequent roi_only export keeps ONLY true ROI cells."""
    import spatialdata as sd
    from xenium_app import core, regions, reseg, sdata_store

    n = 6
    cells_df = pd.DataFrame(
        {
            "x_centroid": [5.0, 10.0, 50.0, 60.0, 70.0, 80.0],
            "y_centroid": [5.0, 5.0, 50.0, 50.0, 50.0, 50.0],
            "transcript_counts": [20, 25, 30, 15, 18, 22],
            "cluster_kmeans_10": [1, 1, 2, 2, 3, 3],
            "umap_1": np.linspace(0.0, 1.0, n),
            "umap_2": np.linspace(1.0, 0.0, n),
        },
        index=pd.Index([f"rc_{i}" for i in range(n)], name="cell_id"),
    )
    n_genes = min(8, len(D["gene_names"]))
    gene_names = list(D["gene_names"][:n_genes])
    expr = sp.csr_matrix(np.ones((n, n_genes), dtype=np.float32))
    bounds = {
        f"rc_{i}": ([x - 1, x + 1, x + 1, x - 1, x - 1], [y - 1, y - 1, y + 1, y + 1, y - 1])
        for i, (x, y) in enumerate(zip(cells_df["x_centroid"], cells_df["y_centroid"]))
    }

    run_dir = tmp_path / "baysor_run"
    run_dir.mkdir()
    zarr_path = reseg._build_reseg_sdata(cells_df, bounds, expr, gene_names, "baysor", str(run_dir))
    assert os.path.isdir(zarr_path)

    # Apply ROI covering ONLY rc_0 and rc_1, plus niche & cluster_split_10 with NAs
    roi = {"name": "focus", "cls": "region", "polygon_xy": _box_plot_space(0.0, 20.0, 0.0, 20.0)}
    with core._roi_lock:
        core._roi_state["rois"] = [roi]
        regions._roi_apply_metadata_to_df(cells_df, [roi])
    cells_df["niche"] = pd.array(["1", "1", pd.NA, pd.NA, "2", pd.NA], dtype="string")
    cells_df["cluster_split_10"] = pd.array(["1", pd.NA, "2", pd.NA, pd.NA, "3"], dtype="string")

    # _update_reseg_zarr_obs serialises pd.NA as "" on disk
    reseg._update_reseg_zarr_obs(zarr_path, cells_df)

    # 1) Fast zarr reload restores "" -> pd.NA
    fast_df, _, _, _, _ = reseg._fast_load_reseg_zarr(zarr_path)
    assert list(fast_df["roi_region"].isna()) == [False, False, True, True, True, True]
    assert list(fast_df["roi_region"].dropna()) == ["focus", "focus"]
    assert list(fast_df["niche"].isna()) == [False, False, True, True, False, True]
    assert list(fast_df["cluster_split_10"].isna()) == [False, True, False, True, True, False]

    # 2) _load_zarr_obs_into_df also restores "" -> pd.NA
    slow_df = cells_df[["x_centroid", "y_centroid", "transcript_counts"]].copy()
    reseg._load_zarr_obs_into_df(zarr_path, slow_df)
    assert list(slow_df["roi_region"].isna()) == [False, False, True, True, True, True]
    assert list(slow_df["niche"].isna()) == [False, False, True, True, False, True]
    assert list(slow_df["cluster_split_10"].isna()) == [False, True, False, True, True, False]

    # 3) Re-applying ROI metadata after deleting the ROI clears "focus" cleanly with zero ""
    regions._roi_apply_metadata_to_df(fast_df, [])
    assert fast_df["roi_region"].isna().all()
    regions._roi_apply_metadata_to_df(fast_df, [roi])
    assert list(fast_df.index[fast_df["roi_region"].notna()]) == ["rc_0", "rc_1"]

    # 4) ROI-only SpatialData export of this reseg run keeps ONLY rc_0 & rc_1 (2 cells, not 6!)
    with core._baysor_lock:
        core._baysor_state.update(
            {
                "status": "done",
                "message": "Loaded",
                "result": {
                    "cells_df": fast_df,
                    "cell_bounds": bounds,
                    "out_dir": str(run_dir),
                    "expr": expr,
                    "source": "baysor",
                    "sdata_path": zarr_path,
                },
            }
        )
    export_path = str(tmp_path / "reseg_roi_only.zarr")
    sdata_store._save_sdata_to_disk(export_path, seg_source="baysor", roi_only=True)
    with core._save_sdata_lock:
        assert core._save_sdata_state["status"] == "done", core._save_sdata_state["message"]
    exported = sd.read_zarr(export_path)
    assert list(exported.tables["table"].obs_names) == ["rc_0", "rc_1"]
    assert list(exported.shapes["cell_boundaries"].index.astype(str)) == ["rc_0", "rc_1"]


# ── R10-7: Live Baysor & Proseg runs apply saved ROIs and update zarr ────────

def test_live_baysor_and_proseg_apply_saved_rois_and_persist_to_zarr(xe, D, monkeypatch, tmp_path):
    import types
    from xenium_app import clustering, core, reseg

    roi = {"name": "isle", "cls": "tissue", "polygon_xy": _box_plot_space(0.0, 25.0, 0.0, 25.0)}
    with core._roi_lock:
        core._roi_state["rois"] = [roi]

    def _fake_clusters_umap(df, expr_mat, progress_fn=None):
        df["cluster_kmeans_10"] = 1
        df["umap_1"] = 0.1
        df["umap_2"] = 0.2

    monkeypatch.setattr(clustering, "_compute_reseg_clusters_umap", _fake_clusters_umap)
    monkeypatch.setattr(reseg, "_get_patch_bounds_um", lambda: None)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        reseg.subprocess,
        "run",
        lambda cmd, **kw: types.SimpleNamespace(returncode=0, stdout=b"0.7.1\n", stderr=b""),
    )

    gene0 = D["gene_names"][0]
    gene1 = D["gene_names"][min(1, len(D["gene_names"]) - 1)]

    # ── Live Baysor (single-run with integer CSV cell IDs to also test str index normalization)
    def _fake_baysor_single(tx_df, patch_dir, *args, **kwargs):
        os.makedirs(patch_dir, exist_ok=True)
        seg_df = pd.DataFrame(
            {
                "cell": [101, 101, 202, 202],
                "gene": [gene0, gene1, gene0, gene1],
                "x": [10.0, 12.0, 80.0, 82.0],
                "y": [10.0, 12.0, 80.0, 82.0],
            }
        )
        seg_df.to_csv(os.path.join(patch_dir, "segmentation.csv"), index=False)
        cdf = (
            seg_df.groupby("cell")
            .agg(x_centroid=("x", "mean"), y_centroid=("y", "mean"), transcript_counts=("x", "count"))
            .rename_axis("cell_id")
        )
        bnds = {
            101: ([9.0, 13.0, 13.0, 9.0, 9.0], [9.0, 9.0, 13.0, 13.0, 9.0]),
            202: ([79.0, 83.0, 83.0, 79.0, 79.0], [79.0, 79.0, 83.0, 83.0, 79.0]),
        }
        return cdf, bnds, seg_df

    monkeypatch.setattr(reseg, "_baysor_run_single", _fake_baysor_single)

    reseg._run_baysor(scale=5.0, min_mol=3, use_prior=False, prior_conf=0.8, use_patches=False)
    with core._baysor_lock:
        assert core._baysor_state["status"] == "done", core._baysor_state["message"]
        bres = core._baysor_state["result"]

    bdf = bres["cells_df"]
    assert list(bdf.index) == ["101", "202"]
    assert "roi_tissue" in bdf.columns
    assert bdf.loc["101", "roi_tissue"] == "isle"
    assert pd.isna(bdf.loc["202", "roi_tissue"])

    # Verify zarr on disk also has roi_tissue persisted and normalized on fast reload
    b_reloaded, _, _, _, _ = reseg._fast_load_reseg_zarr(bres["sdata_path"])
    assert b_reloaded.loc["101", "roi_tissue"] == "isle"
    assert pd.isna(b_reloaded.loc["202", "roi_tissue"])

    # ── Live Proseg
    def _fake_proseg_single(tx_input, patch_dir, *args, **kwargs):
        os.makedirs(patch_dir, exist_ok=True)
        cm = pd.DataFrame(
            {
                "cell": [1, 2],
                "centroid_x": [12.0, 90.0],
                "centroid_y": [12.0, 90.0],
                "population": [2, 2],
                "volume": [100.0, 120.0],
            }
        )
        cm.to_csv(os.path.join(patch_dir, "cell-metadata.csv.gz"), index=False, compression="gzip")
        tx_meta = pd.DataFrame(
            {
                "assignment": [1, 1, 2, 2],
                "gene": [gene0, gene1, gene0, gene1],
            }
        )
        tx_meta.to_csv(os.path.join(patch_dir, "transcript-metadata.csv.gz"), index=False, compression="gzip")
        cdf = pd.DataFrame(
            {
                "x_centroid": [12.0, 90.0],
                "y_centroid": [12.0, 90.0],
                "transcript_counts": [2, 2],
                "cell_volume": [100.0, 120.0],
            },
            index=pd.Index([1, 2], name="cell_id"),
        )
        bnds = {
            1: ([10.0, 14.0, 14.0, 10.0, 10.0], [10.0, 10.0, 14.0, 14.0, 10.0]),
            2: ([88.0, 92.0, 92.0, 88.0, 88.0], [88.0, 88.0, 92.0, 92.0, 88.0]),
        }
        return cdf, bnds, tx_meta

    monkeypatch.setattr(reseg, "_proseg_run_single", _fake_proseg_single)

    reseg._run_proseg(voxel_size=4, n_threads=1, n_samples=5, use_patches=False)
    with core._proseg_lock:
        assert core._proseg_state["status"] == "done", core._proseg_state["message"]
        pres = core._proseg_state["result"]

    pdf = pres["cells_df"]
    assert list(pdf.index) == ["1", "2"]
    assert "roi_tissue" in pdf.columns
    assert pdf.loc["1", "roi_tissue"] == "isle"
    assert pd.isna(pdf.loc["2", "roi_tissue"])

    p_reloaded, _, _, _, _ = reseg._fast_load_reseg_zarr(pres["sdata_path"])
    assert p_reloaded.loc["1", "roi_tissue"] == "isle"
    assert pd.isna(p_reloaded.loc["2", "roi_tissue"])

    # Also verify slow-path cached reloads (when _fast_load_reseg_zarr raises) normalize '' -> pd.NA
    # even after clearing session polygons (so stored 'isle' survives and non-ROI cell 202 / 2 stays NA!)
    with core._roi_lock:
        core._roi_state["rois"] = []
    xe._roi_forget_known()
    monkeypatch.setattr(
        reseg,
        "_fast_load_reseg_zarr",
        lambda zp: (_ for _ in ()).throw(RuntimeError("force slow CSV + zarr fallback")),
    )
    reseg._load_cached_baysor(bres["out_dir"])
    with core._baysor_lock:
        sb_df = core._baysor_state["result"]["cells_df"]
    assert sb_df.loc["101", "roi_tissue"] == "isle"
    assert pd.isna(sb_df.loc["202", "roi_tissue"])

    reseg._load_cached_proseg(pres["out_dir"])
    with core._proseg_lock:
        sp_df = core._proseg_state["result"]["cells_df"]
    assert sp_df.loc["1", "roi_tissue"] == "isle"
    assert pd.isna(sp_df.loc["2", "roi_tissue"])


# ── R10-2: Concurrent add_sample race & duplicate _ds_key re-check ───────────

def _make_standin_sample(D, path: str) -> dict:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "experiment.xenium"), "w") as fh:
        json.dump({"run_name": os.path.basename(path)}, fh)
    ds = dict(D)
    ds["df"] = D["df"].copy()
    ds["data_dir"] = os.path.abspath(path)
    ds["sdata_path"] = os.path.join(os.path.abspath(path), "spatialdata_xenium.zarr")
    return ds


def test_concurrent_add_sample_blocks_second_caller_and_never_duplicates(xe, D, monkeypatch, tmp_path):
    from xenium_app import cb_data, dataview, loading

    sample_dir = str(tmp_path / "extra_sample_a")
    standin = _make_standin_sample(D, sample_dir)

    entered = threading.Event()
    proceed = threading.Event()

    def _slow_load(path):
        entered.set()
        assert proceed.wait(timeout=10.0)
        out = dict(standin)
        out["df"] = standin["df"].copy()
        return out

    monkeypatch.setattr(loading, "load_xenium_data", _slow_load)

    results: list[tuple] = [None] * 8  # type: ignore[list-item]

    def _worker(idx: int):
        if idx > 0:
            assert entered.wait(timeout=10.0)
        results[idx] = cb_data.add_sample(1, None, sample_dir, 0)

    t0 = threading.Thread(target=_worker, args=(0,))
    t0.start()
    assert entered.wait(timeout=10.0)

    contenders = [threading.Thread(target=_worker, args=(i,)) for i in range(1, 8)]
    for t in contenders:
        t.start()
    for t in contenders:
        t.join(timeout=10.0)

    proceed.set()
    t0.join(timeout=10.0)

    assert results[0][0].startswith("Added (x_offset=")
    assert results[0][1:] == (False, 1, "2 samples loaded")
    for r in results[1:]:
        assert r == ("A sample is already being added.", no_update, no_update, no_update)
    assert len(dataview.EXTRA_DATASETS) == 1

    # Lock is released: calling again now hits the duplicate check cleanly
    dup = cb_data.add_sample(2, None, sample_dir, 1)
    assert dup == ("Sample already loaded.", no_update, no_update, no_update)
    assert len(dataview.EXTRA_DATASETS) == 1


def test_add_sample_rechecks_ds_key_if_mutated_during_load_and_releases_lock_on_error(xe, D, monkeypatch, tmp_path):
    from xenium_app import cb_data, dataview, loading

    sample_b = str(tmp_path / "extra_sample_b")
    standin_b = _make_standin_sample(D, sample_b)

    def _racing_load(path):
        # Simulate another code path installing a sample with the same _ds_key mid-load
        sneaky = dict(standin_b)
        sneaky["df"] = standin_b["df"].copy()
        sneaky["x_offset"] = 1200.0
        dataview.EXTRA_DATASETS.append(sneaky)
        out = dict(standin_b)
        out["df"] = standin_b["df"].copy()
        return out

    monkeypatch.setattr(loading, "load_xenium_data", _racing_load)
    out = cb_data.add_sample(1, None, sample_b, 0)
    assert out == ("Sample already loaded.", no_update, no_update, no_update)
    assert len(dataview.EXTRA_DATASETS) == 1

    # Exception during load still releases _add_sample_lock
    sample_c = str(tmp_path / "extra_sample_c")
    _make_standin_sample(D, sample_c)
    monkeypatch.setattr(loading, "load_xenium_data", lambda p: (_ for _ in ()).throw(RuntimeError("boom")))
    err = cb_data.add_sample(1, None, sample_c, 0)
    assert err == ("Error: boom", no_update, no_update, no_update)
    assert not cb_data._add_sample_lock.locked()


# ── R10-3: start_sdata_clear _running_jobs() guard ───────────────────────────

def test_start_sdata_clear_refuses_while_jobs_are_running(xe, monkeypatch, tmp_path):
    from xenium_app import cb_data, core, jobs, sdata_store

    sentinel_cache = tmp_path / "fake_sdata_cache.zarr"
    sentinel_cache.mkdir()
    (sentinel_cache / ".complete").write_text("ok")
    monkeypatch.setattr(sdata_store, "_sdata_cache_path", lambda: str(sentinel_cache))

    cleared = []
    monkeypatch.setattr(sdata_store, "clear_sdata_cache", lambda: cleared.append(True))
    monkeypatch.setattr(sdata_store, "to_spatialdata", lambda **kw: cleared.append(kw))

    with core._sdata_lock:
        core._sdata_state.update({"status": "patches_done", "message": "4 patches ready", "patches": [1, 2]})

    # 1) State-claimed job (e.g. Baysor running)
    with core._baysor_lock:
        core._baysor_state["status"] = "running"
    try:
        res = cb_data.start_sdata_clear(1, 20)
    finally:
        with core._baysor_lock:
            core._baysor_state["status"] = "idle"

    assert res == (True, "Not re-built — still running: Baysor", False, False, no_update, no_update)
    assert not cleared
    assert sentinel_cache.is_dir()
    with core._sdata_lock:
        assert core._sdata_state["status"] == "patches_done"
        assert core._sdata_state["patches"] == [1, 2]

    # 2) Active registry job
    gate = threading.Event()
    started = threading.Event()

    def _bg():
        started.set()
        gate.wait(timeout=10.0)

    jobs._job_start("de", "Differential expression", _bg)
    try:
        assert started.wait(timeout=5.0)
        res2 = cb_data.start_sdata_clear(2, 20)
        assert res2[0] is True
        assert res2[1].startswith("Not re-built — still running:")
        assert "differential expression" in res2[1].lower()
        assert res2[2:] == (False, False, no_update, no_update)
        assert not cleared
    finally:
        gate.set()
        xe._test_baseline.settle()


# ── R10-4: _h5ad_install applies ROIs to sections 2+ and clears UMAP caches ──

def test_h5ad_install_applies_rois_to_extra_sections_and_clears_umap_caches(xe, tmp_path):
    ad = pytest.importorskip("anndata")
    import hashlib
    from xenium_app import caching, cb_data, core, dataview, h5ad

    # Build a 3-section .h5ad with deterministic coordinates per section
    rows = []
    for sec in ("sec_1", "sec_2", "sec_3"):
        for i, (x, y) in enumerate([(10.0, 10.0), (20.0, 10.0), (150.0, 150.0)]):
            rows.append({"section": sec, "x": x, "y": y, "cell_type": "T", "id": f"{sec}_c{i}"})
    obs = pd.DataFrame(rows).set_index("id")
    obs["section"] = pd.Categorical(obs["section"])
    obs["cell_type"] = pd.Categorical(obs["cell_type"])
    X = sp.csr_matrix(np.ones((len(obs), 4), dtype=np.float32))
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=["G0", "G1", "G2", "G3"]))
    h5_path = tmp_path / "multi_sec.h5ad"
    adata.write_h5ad(h5_path)

    mapping = {"x": "x", "y": "y", "section": "section", "cell_types": ["cell_type"], "counts": "X"}
    datasets, man, h5_cache_dir = h5ad._h5ad_load(str(h5_path), mapping)
    assert len(datasets) == 3

    # Compute the ROI cache file path for datasets[0] (what _auto_load_rois reads during _reset_dataset_state)
    ds0_key = str(datasets[0].get("data_dir") or datasets[0].get("sdata_path"))
    tag = hashlib.md5(ds0_key.encode()).hexdigest()[:8]
    roi_cache_file = os.path.join(core.CACHE_BASE_DIR, f"rois_{tag}.json")

    off1 = float(datasets[1]["x_offset"])
    off2 = float(datasets[2]["x_offset"])
    assert off1 > 0 and off2 > off1

    saved_rois = [
        {"name": "s1_box", "cls": "zone", "polygon_xy": _box_plot_space(0.0, 30.0, 0.0, 30.0)},
        {"name": "s2_box", "cls": "zone", "polygon_xy": _box_plot_space(off1 + 0.0, off1 + 30.0, 0.0, 30.0)},
        {"name": "s3_box", "cls": "zone", "polygon_xy": _box_plot_space(off2 + 0.0, off2 + 30.0, 0.0, 30.0)},
    ]
    with open(roi_cache_file, "w") as fh:
        json.dump(saved_rois, fh)

    # Seed UMAP caches with stale sentinel entries to verify _h5ad_install clears them
    dataview._umap_df_cache["stale_key"] = "stale_df"
    dataview._umap_fig_cache["stale_key"] = "stale_fig"

    try:
        cb_data._h5ad_install(datasets, man, h5_cache_dir)

        assert len(dataview.EXTRA_DATASETS) == 2
        assert "stale_key" not in dataview._umap_df_cache
        assert "stale_key" not in dataview._umap_fig_cache

        # Primary section (sec_1)
        p_df = core.DATA["df"]
        assert list(p_df.loc[["sec_1_c0", "sec_1_c1"], "roi_zone"]) == ["s1_box", "s1_box"]
        assert pd.isna(p_df.loc["sec_1_c2", "roi_zone"])

        # Section 2 (EXTRA_DATASETS[0])
        e1_df = dataview.EXTRA_DATASETS[0]["df"]
        assert "roi_zone" in e1_df.columns
        assert list(e1_df.loc[["sec_2_c0", "sec_2_c1"], "roi_zone"]) == ["s2_box", "s2_box"]
        assert pd.isna(e1_df.loc["sec_2_c2", "roi_zone"])

        # Section 3 (EXTRA_DATASETS[1])
        e2_df = dataview.EXTRA_DATASETS[1]["df"]
        assert "roi_zone" in e2_df.columns
        assert list(e2_df.loc[["sec_3_c0", "sec_3_c1"], "roi_zone"]) == ["s3_box", "s3_box"]
        assert pd.isna(e2_df.loc["sec_3_c2", "roi_zone"])
    finally:
        if os.path.exists(roi_cache_file):
            os.remove(roi_cache_file)


# ── R10-5: roi_only export with empty session ROIs ───────────────────────────

def test_roi_only_export_without_rois_warns_and_records_provenance(xe, D, tmp_path):
    import spatialdata as sd
    from xenium_app import core, sdata_store

    with core._roi_lock:
        core._roi_state["rois"] = []
    assert not [c for c in D["df"].columns if c.startswith("roi_")]

    out_path = str(tmp_path / "no_rois_export.zarr")
    sdata_store._save_sdata_to_disk(out_path, seg_source="xenium", roi_only=True)

    with core._save_sdata_lock:
        status = core._save_sdata_state["status"]
        msg = core._save_sdata_state["message"]
    assert status == "done"
    assert "ROI only: no ROIs defined — all cells exported" in msg
    assert xe.poll_save_sdata(0, True)[0].startswith("✓")

    sdata = sd.read_zarr(out_path)
    assert sdata.tables["table"].n_obs == len(D["df"])

    rec = xe._prov_recent[-1]
    assert rec["kind"] == "save"
    assert rec["params"]["roi_only"] is True
    assert rec["params"]["roi_filter_applied"] is False
    assert rec["params"]["n_cells_exported"] == len(D["df"])
    assert rec["headline"]["roi_filter_applied"] is False
    assert rec["headline"]["n_cells_exported"] == len(D["df"])
    assert any("no ROIs defined" in w for w in rec["warnings"])


def test_roi_only_export_uses_stored_roi_columns_when_session_polygons_empty(xe, D, tmp_path):
    """When a dataset (e.g. a loaded exported .zarr) already carries non-NA roi_* labels
    in cells_df while session _roi_state['rois'] is empty, roi_only=True filters on those
    stored roi_* columns instead of exporting all cells."""
    import spatialdata as sd
    from xenium_app import core, sdata_store

    with core._roi_lock:
        core._roi_state["rois"] = []

    kept_ids = list(D["df"].index[:3].astype(str))
    col_vals = [pd.NA] * len(D["df"])
    for i in range(min(3, len(col_vals))):
        col_vals[i] = "exported_region"
    D["df"]["roi_stored"] = pd.array(col_vals, dtype="string")

    out_path = str(tmp_path / "stored_rois_export.zarr")
    sdata_store._save_sdata_to_disk(out_path, seg_source="xenium", roi_only=True)

    with core._save_sdata_lock:
        status = core._save_sdata_state["status"]
        msg = core._save_sdata_state["message"]
    assert status == "done"
    assert "no ROIs defined" not in msg

    sdata = sd.read_zarr(out_path)
    tbl = sdata.tables["table"]
    obs_ids = (
        list(tbl.obs["cell_id"].astype(str))
        if "cell_id" in tbl.obs.columns
        else list(tbl.obs_names.astype(str))
    )
    assert obs_ids == kept_ids

    rec = xe._prov_recent[-1]
    assert rec["kind"] == "save"
    assert rec["params"]["roi_only"] is True
    assert rec["params"]["roi_filter_applied"] is True
    assert rec["params"]["n_cells_exported"] == len(kept_ids)
    assert rec["headline"]["roi_filter_applied"] is True
    assert rec["headline"]["n_cells_exported"] == len(kept_ids)


# ── Cross-slice #18/#19: live _run_imgseg ROI persistence + X_corrected cols ─

def test_live_imgseg_run_applies_and_persists_saved_rois(xe, D, prepared_patches):
    """Live _run_imgseg applies saved ROIs to cells_df AND persists roi_* columns to
    spatialdata_imgseg.zarr via reseg._update_reseg_zarr_obs."""
    pytest.importorskip("shapely")
    from xenium_app import core, imgseg, reseg

    x_lo = float(D["df"]["x_centroid"].min()) - 50.0
    x_mid = float(D["df"]["x_centroid"].median())
    y_lo = float(D["df"]["y_centroid"].min()) - 50.0
    y_hi = float(D["df"]["y_centroid"].max()) + 50.0

    rois = [{"name": "iseg_half", "cls": "zone", "polygon_xy": _box_plot_space(x_lo, x_mid, y_lo, y_hi)}]
    with core._roi_lock:
        core._roi_state["rois"] = rois

    ch = imgseg._iseg_default_channel(imgseg._iseg_channels())
    assert ch
    out_dir = None
    try:
        imgseg._run_imgseg("classical", [ch], diameter_um=8, expand_um=5, min_tx=1, patch_um=200)
        with core._imgseg_lock:
            assert core._imgseg_state["status"] == "done", core._imgseg_state["message"]
            res = dict(core._imgseg_state["result"])
        out_dir = res["out_dir"]
        live_df = res["cells_df"]
        assert "roi_zone" in live_df.columns
        inside_mask = live_df["x_centroid"] <= x_mid
        assert inside_mask.any() and (~inside_mask).any()
        assert (live_df.loc[inside_mask, "roi_zone"] == "iseg_half").all()
        assert live_df.loc[~inside_mask, "roi_zone"].isna().all()
        assert not (live_df["roi_zone"] == "").any()

        # Persisted zarr store must match live_df without needing session ROIs
        disk_df, *_ = reseg._fast_load_reseg_zarr(res["sdata_path"])
        assert "roi_zone" in disk_df.columns
        assert (disk_df.loc[inside_mask, "roi_zone"] == "iseg_half").all()
        assert disk_df.loc[~inside_mask, "roi_zone"].isna().all()
        assert not (disk_df["roi_zone"] == "").any()

        # Reloading from cache after forgetting session ROIs preserves stored labels + pd.NA
        xe._roi_forget_known()
        with core._roi_lock:
            core._roi_state["rois"] = []
        imgseg._load_cached_imgseg(out_dir)
        with core._imgseg_lock:
            assert core._imgseg_state["status"] == "done"
            cached_df = core._imgseg_state["result"]["cells_df"]
        assert (cached_df.loc[inside_mask, "roi_zone"] == "iseg_half").all()
        assert cached_df.loc[~inside_mask, "roi_zone"].isna().all()
    finally:
        if out_dir and os.path.isdir(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)


def test_save_sdata_preserves_split_corrected_cols_across_sources_and_roi_filter(xe, D, tmp_path, monkeypatch):
    """Exporting X_corrected via _save_sdata_to_disk preserves split_corrected_cols in
    adata.uns['split_corrected_cols'] (and layers/X_corrected.attrs['corrected_cols'])
    from in-memory holder, existing adata.uns, or low-level zarr attrs, and clears stale
    uns entries when holder explicitly has split_corrected_cols=None."""
    ad = pytest.importorskip("anndata")
    sd = pytest.importorskip("spatialdata")
    import zarr as zarr_mod
    from xenium_app import core, loading, sdata_store

    def _make_store(path, *, uns_cols=None, attr_cols=None):
        ids = ["c1", "c2", "c3", "c4"]
        obs = pd.DataFrame(
            {
                "cell_id": ids,
                "x_centroid": [10.0, 20.0, 80.0, 90.0],
                "y_centroid": [10.0, 20.0, 80.0, 90.0],
            },
            index=pd.Index(ids, name="cell_id"),
        )
        var = pd.DataFrame(index=["G0", "G1", "G2", "G3"])
        X = sp.csr_matrix(np.arange(1, 17, dtype=np.float32).reshape(4, 4))
        X_corr = sp.csr_matrix(np.ones((4, 4), dtype=np.float32) * 2.0)
        uns = {}
        if uns_cols is not None:
            uns["split_corrected_cols"] = list(uns_cols)
        adata = ad.AnnData(X=X, obs=obs, var=var, layers={"X_corrected": X_corr}, uns=uns)
        s = sd.SpatialData(tables={"table": adata})
        s.write(str(path), overwrite=True)
        if attr_cols is not None:
            zr = zarr_mod.open_group(str(path), mode="r+", use_consolidated=False)
            zr["tables"]["table"]["layers"]["X_corrected"].attrs["corrected_cols"] = list(attr_cols)
        return obs.copy()

    # 1) Xenium holder with in-memory split_corrected_cols (both full and roi_only=True export)
    src_xe = tmp_path / "src_xe.zarr"
    obs_xe = _make_store(src_xe)
    obs_xe["roi_zone"] = pd.array(["keep", "keep", pd.NA, pd.NA], dtype="string")
    monkeypatch.setitem(core.DATA, "sdata_path", str(src_xe))
    monkeypatch.setitem(core.DATA, "df", obs_xe)
    core.DATA["split_corrected_cols"] = ["G0", "G1"]
    try:
        with core._roi_lock:
            core._roi_state["rois"] = []

        out_full = tmp_path / "out_xe_full.zarr"
        sdata_store._save_sdata_to_disk(str(out_full), seg_source="xenium", roi_only=False)
        tbl_full = sd.read_zarr(str(out_full)).tables["table"]
        assert "X_corrected" in tbl_full.layers
        assert list(tbl_full.uns["split_corrected_cols"]) == ["G0", "G1"]
        zr_full = zarr_mod.open_group(str(out_full), mode="r", use_consolidated=False)
        assert list(zr_full["tables"]["table"]["layers"]["X_corrected"].attrs["corrected_cols"]) == ["G0", "G1"]

        out_roi = tmp_path / "out_xe_roi.zarr"
        sdata_store._save_sdata_to_disk(str(out_roi), seg_source="xenium", roi_only=True)
        tbl_roi = sd.read_zarr(str(out_roi)).tables["table"]
        assert tbl_roi.n_obs == 2
        assert list(tbl_roi.uns["split_corrected_cols"]) == ["G0", "G1"]
        zr_roi = zarr_mod.open_group(str(out_roi), mode="r", use_consolidated=False)
        assert list(zr_roi["tables"]["table"]["layers"]["X_corrected"].attrs["corrected_cols"]) == ["G0", "G1"]

        # Round-trip reload via load_from_sdata_zarr populates split_corrected_cols
        reloaded = loading.load_from_sdata_zarr(str(out_roi))
        assert reloaded["split_corrected_cols"] == ["G0", "G1"]
        assert reloaded["split_corrected_expr"].shape == (2, 4)

        # 2) Explicit holder split_corrected_cols=None clears stale uns entry on export
        src_stale = tmp_path / "src_stale.zarr"
        obs_stale = _make_store(src_stale, uns_cols=["G0"])
        monkeypatch.setitem(core.DATA, "sdata_path", str(src_stale))
        monkeypatch.setitem(core.DATA, "df", obs_stale)
        core.DATA["split_corrected_cols"] = None
        out_cleared = tmp_path / "out_cleared.zarr"
        sdata_store._save_sdata_to_disk(str(out_cleared), seg_source="xenium", roi_only=False)
        tbl_cleared = sd.read_zarr(str(out_cleared)).tables["table"]
        assert "split_corrected_cols" not in tbl_cleared.uns
        zr_cleared = zarr_mod.open_group(str(out_cleared), mode="r", use_consolidated=False)
        assert "corrected_cols" not in zr_cleared["tables"]["table"]["layers"]["X_corrected"].attrs
    finally:
        core.DATA.pop("split_corrected_cols", None)

    # 3) Store where adata.uns["split_corrected_cols"] already exists and holder lacks the key
    src_uns = tmp_path / "src_uns.zarr"
    obs_uns = _make_store(src_uns, uns_cols=["G0", "G2"])
    monkeypatch.setitem(core.DATA, "sdata_path", str(src_uns))
    monkeypatch.setitem(core.DATA, "df", obs_uns)
    out_uns = tmp_path / "out_uns.zarr"
    sdata_store._save_sdata_to_disk(str(out_uns), seg_source="xenium", roi_only=False)
    tbl_uns = sd.read_zarr(str(out_uns)).tables["table"]
    assert list(tbl_uns.uns["split_corrected_cols"]) == ["G0", "G2"]
    zr_uns = zarr_mod.open_group(str(out_uns), mode="r", use_consolidated=False)
    assert list(zr_uns["tables"]["table"]["layers"]["X_corrected"].attrs["corrected_cols"]) == ["G0", "G2"]

    # 4) Reseg store where only low-level layers/X_corrected.attrs["corrected_cols"] exists on disk
    src_reseg = tmp_path / "src_reseg.zarr"
    obs_reseg = _make_store(src_reseg, attr_cols=["G1", "G3"])
    reseg_out_dir = tmp_path / "baysor_run"
    reseg_out_dir.mkdir()
    with core._baysor_lock:
        core._baysor_state.update({
            "status": "done",
            "message": "Loaded",
            "result": {
                "cells_df": obs_reseg,
                "cell_bounds": {},
                "out_dir": str(reseg_out_dir),
                "sdata_path": str(src_reseg),
                "source": "baysor",
            },
        })
    out_reseg = tmp_path / "out_reseg.zarr"
    sdata_store._save_sdata_to_disk(str(out_reseg), seg_source="baysor", roi_only=False)
    tbl_reseg = sd.read_zarr(str(out_reseg)).tables["table"]
    assert list(tbl_reseg.uns["split_corrected_cols"]) == ["G1", "G3"]
    zr_reseg = zarr_mod.open_group(str(out_reseg), mode="r", use_consolidated=False)
    assert list(zr_reseg["tables"]["table"]["layers"]["X_corrected"].attrs["corrected_cols"]) == ["G1", "G3"]
