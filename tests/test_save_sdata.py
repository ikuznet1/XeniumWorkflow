"""Save as SpatialData: what is written equals what the app shows, and the written
store opens again - with spatialdata, with the app's loader, and as the dataset of a
freshly started app."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import xe_testlib as tl


def _save(xe, out_dir, fname="export", options=(), seg="xenium"):
    msg, poll_disabled, btn_disabled = xe.start_save_sdata(1, str(out_dir), fname, list(options), seg)
    assert (poll_disabled, btn_disabled) == (False, True), msg
    tl.wait_state(xe._save_sdata_state, xe._save_sdata_lock, timeout=600)
    path = os.path.join(str(out_dir), fname if fname.endswith(".zarr") else fname + ".zarr")
    assert os.path.isdir(path)
    assert xe.poll_save_sdata(0, True)[0].startswith("✓")
    return path


def _obs_ids(table) -> pd.Index:
    ids = table.obs["cell_id"] if "cell_id" in table.obs.columns else pd.Series(table.obs_names)
    return pd.Index(ids.astype(str).to_numpy())


@pytest.fixture
def analysed(xe, D, raw, labels):
    """Labels, niches and two ROIs present in the session."""
    xe.run_niche(1, 3, 4, "labels_celltypist", "xenium")
    tl.wait_state(xe._niche_state, xe._niche_lock)
    x = raw.cells["x_centroid"]
    y = -raw.cells["y_centroid"]
    lo, hi = x.quantile(0.35) + 0.01, x.quantile(0.65) - 0.01
    left = [[x.min() - 1, y.min() - 1], [lo, y.min() - 1], [lo, y.max() + 1], [x.min() - 1, y.max() + 1]]
    right = [[hi, y.min() - 1], [x.max() + 1, y.min() - 1], [x.max() + 1, y.max() + 1], [hi, y.max() + 1]]
    with tl.new_threads():
        xe.save_roi(1, "left", "zone", left, 0, "xenium")
        xe.save_roi(1, "right", "zone", right, 1, "xenium")
    return D["df"].copy()


def test_export_round_trip(xe, D, raw, labels, analysed, tmp_path):
    import spatialdata
    path = _save(xe, tmp_path)
    sdata = spatialdata.read_zarr(path)
    table = sdata.tables["table"]
    ids = _obs_ids(table)
    df = analysed

    # cells and counts
    assert list(ids) == list(raw.cells.index), "every cell, in the app's order"
    counts, names, _ = raw.gene_counts
    assert list(table.var_names) == names
    assert tl.sparse_equal(sp.csr_matrix(table.X), counts)
    assert tl.sparse_equal(sp.csr_matrix(table.X), D["expr"])
    assert not table.var["is_imputed"].any()
    assert list(table.uns["cluster_methods"]) == D["cluster_methods"]

    # per-cell values, by cell id
    obs = table.obs.set_axis(ids)
    for col in ("x_centroid", "y_centroid", "transcript_counts", "cell_area", "nucleus_area"):
        if col in raw.cells.columns:
            np.testing.assert_allclose(obs[col].to_numpy(float), raw.cells[col].to_numpy(float), atol=1e-6)
    assert obs["cell_type_celltypist"].astype(str).tolist() == labels.loc[ids].tolist()
    assert obs["niche"].astype(str).tolist() == df["niche"].astype(str).loc[ids].tolist()
    roi = obs["roi_zone"].astype(str).replace({"": np.nan, "nan": np.nan, "<NA>": np.nan})
    want_roi = df["roi_zone"].astype(object).where(df["roi_zone"].notna(), np.nan)
    assert roi.dropna().to_dict() == want_roi.dropna().to_dict()
    assert set(roi.dropna()) == {"left", "right"} and roi.isna().any()
    for method in D["cluster_methods"][:3]:
        col = f"clust__{method}"
        assert obs[col].astype(int).tolist() == df[col].astype(int).tolist()
    for col, src in (("umap_1", "UMAP-1"), ("umap_2", "UMAP-2")):
        if raw.umap is not None:
            np.testing.assert_allclose(obs.loc[raw.umap.index, col].to_numpy(float),
                                       raw.umap[src].to_numpy(float), atol=1e-6)

    # shapes
    for kind in ("cell", "nucleus"):
        truth = raw.boundaries(kind)
        if not truth:
            continue
        gdf = sdata.shapes[f"{kind}_boundaries"]
        assert set(gdf.index.astype(str)) == set(truth)
        import shapely
        for cid in list(truth)[:: max(1, len(truth) // 100)]:
            geoms = gdf.loc[[cid]].geometry            # one row, or one per nucleus
            assert tl.point_set(shapely.get_coordinates(geoms.values)) == tl.point_set(truth[cid])

    # the app's own loader reads it back to the same dataset
    again = xe.load_from_sdata_zarr(path)
    assert again["data_dir"] is None and again["sdata_path"] == os.path.abspath(path)
    assert again["gene_names"] == D["gene_names"] and again["cluster_methods"] == D["cluster_methods"]
    assert list(again["df"].index.astype(str)) == list(raw.cells.index)
    assert tl.sparse_equal(again["expr"], D["expr"])
    assert set(map(str, again["cell_bounds"])) == set(map(str, D["cell_bounds"]))
    assert again["df"]["niche"].astype(str).tolist() == df["niche"].astype(str).tolist()

    # nothing was written anywhere but the chosen directory
    assert sorted(os.listdir(tmp_path)) == ["export.zarr"]


def test_export_roi_only_keeps_exactly_the_roi_cells(xe, D, raw, analysed, tmp_path):
    import spatialdata
    path = _save(xe, tmp_path, "rois", options=["roi_only"])
    sdata = spatialdata.read_zarr(path)
    table = sdata.tables["table"]
    in_roi = analysed.index[analysed["roi_zone"].notna()].astype(str)
    assert 0 < len(in_roi) < len(raw.cells)
    ids = _obs_ids(table)
    assert list(ids) == list(in_roi)
    counts, _, barcodes = raw.gene_counts
    rows = [barcodes.index(i) for i in ids]
    assert tl.sparse_equal(sp.csr_matrix(table.X), counts[rows])
    if raw.boundaries("cell"):
        assert set(sdata.shapes["cell_boundaries"].index.astype(str)) == set(in_roi)


def test_export_with_a_subset_active_marks_cells_outside_it(xe, D, raw, labels, tmp_path):
    """The export always holds the full table; values computed on a subset are present for
    the subset's cells only (and are not shifted onto other cells)."""
    import spatialdata
    keep = list(raw.cells.index[::2])
    xe.subset(cell_ids=keep)
    xe.run_niche(1, 2, 3, "labels_celltypist", "xenium")
    tl.wait_state(xe._niche_state, xe._niche_lock)
    niche = D["df"]["niche"].astype(str)
    path = _save(xe, tmp_path, "sub")
    table = spatialdata.read_zarr(path).tables["table"]
    ids = _obs_ids(table)
    assert list(ids) == list(raw.cells.index)
    got = pd.Series(table.obs["niche"].astype(str).to_numpy(), index=ids).replace({"": np.nan, "nan": np.nan})
    assert got.dropna().to_dict() == niche.to_dict()
    assert set(got.index[got.isna()]) == set(raw.cells.index) - set(keep)


def test_save_request_validation(xe, tmp_path):
    assert xe.start_save_sdata(1, "  ", "x", [], "xenium") == ("Enter an output directory.", True, False)
    msg, _, _ = xe.start_save_sdata(1, str(tmp_path), "", [], "xenium")
    assert msg.endswith("analysis.zarr…")                           # default file name
    tl.wait_state(xe._save_sdata_state, xe._save_sdata_lock, timeout=600)
    assert (tmp_path / "analysis.zarr").is_dir()
    # a second request while one is running is refused
    with xe._save_sdata_lock:
        xe._save_sdata_state["status"] = "running"
    assert xe.start_save_sdata(1, str(tmp_path), "b", [], "xenium")[0] == "Already saving…"
    with xe._save_sdata_lock:
        xe._save_sdata_state["status"] = "idle"
    # asking for a run that is not loaded is an error message, not a crash
    xe.start_save_sdata(1, str(tmp_path), "c", [], "baysor:deadbeef")
    with pytest.raises(AssertionError, match="No Baysor result loaded"):
        tl.wait_state(xe._save_sdata_state, xe._save_sdata_lock)
    assert not (tmp_path / "c.zarr").exists()


def test_exported_store_starts_the_app(xe, D, raw, labels, analysed, tmp_path):
    """DATA_DIR=<export>.zarr in a fresh process: the documented way to reopen an analysis."""
    path = _save(xe, tmp_path / "out", "reopen")
    gene = D["gene_names"][int(np.asarray(D["expr"].sum(axis=0)).ravel().argmax())]
    cache = tmp_path / "cache2"
    cache.mkdir()
    out = tl.run_in_fresh_app(f'''
        import numpy as np
        D = xe.DATA
        RESULT["n_cells"] = len(D["df"]); RESULT["n_genes"] = len(D["gene_names"])
        RESULT["data_dir"] = D["data_dir"]; RESULT["methods"] = list(D["cluster_methods"])
        RESULT["total"] = float(D["expr"].sum())
        RESULT["ids"] = list(D["df"].index.astype(str))
        RESULT["niche"] = D["df"]["niche"].astype(str).tolist()
        RESULT["gene"] = [float(v) for v in xe.get_gene_expression({gene!r})]
        fig = xe.make_spatial_fig("niche", (D["cluster_methods"] or [None])[0], {gene!r}, 3, 0.8, [], None)
        RESULT["traces"] = sorted(t.name for t in fig.data)
        c = xe.app.server.test_client()
        m = c.get("/viewer/meta").get_json()
        RESULT["meta"] = {{k: m[k] for k in ("has_transcripts", "images", "sources")}}
        RESULT["image_type"] = c.get("/viewer/image/0").mimetype
        RESULT["cache_base"] = xe.CACHE_BASE_DIR
    ''', path, cache)
    assert out["n_cells"] == len(raw.cells) and out["n_genes"] == len(D["gene_names"])
    assert out["data_dir"] is None, "a .zarr dataset has no raw bundle behind it"
    assert out["methods"] == D["cluster_methods"] and out["total"] == float(D["expr"].sum())
    assert out["ids"] == list(raw.cells.index)
    assert out["niche"] == analysed["niche"].astype(str).tolist()
    np.testing.assert_allclose(out["gene"], xe.get_gene_expression(gene), rtol=1e-6)
    assert out["traces"] == sorted("Niche " + analysed["niche"].astype(str).unique())
    assert out["meta"] == {"has_transcripts": False, "images": [], "sources": ["xenium"]}
    assert out["image_type"] != "image/tiff" and out["cache_base"] == str(cache)
