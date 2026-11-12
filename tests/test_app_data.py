"""What the app holds after import, checked against the bundle's raw files.

Every expectation is read from the bundle independently (tests/xe_testlib.RawBundle);
nothing is hard-coded for a particular dataset, so the same assertions run on the tiny
bundle in tests/data and on any bundle passed with --bundle.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import xe_testlib as tl


# ── where the app reads and writes ───────────────────────────────────────────
def test_app_runs_on_the_working_copy_and_tmp_cache(xe, D, bundle_dir, cache_dir, source_bundle):
    assert os.path.realpath(D["data_dir"]) == os.path.realpath(bundle_dir)
    assert os.path.realpath(D["data_dir"]) != os.path.realpath(source_bundle)
    assert xe.CACHE_BASE_DIR == str(cache_dir)
    home_cache = os.path.join(os.path.expanduser("~"), ".xenium_explorer_cache")
    assert not os.path.realpath(xe.CACHE_BASE_DIR).startswith(os.path.realpath(home_cache))
    # the generated SpatialData store sits in the working copy, not in the bundle
    assert os.path.realpath(D["sdata_path"]).startswith(os.path.realpath(bundle_dir))
    assert os.path.isdir(D["sdata_path"])


def test_cache_helpers_resolve_under_the_cache_root(xe, cache_dir):
    root = os.path.realpath(cache_dir)
    assert os.path.realpath(xe._cache_dir()) == root == os.path.realpath(xe._spage_cache_dir())
    for path in (xe._roi_cache_path(), xe._sdata_cache_path(), xe._cache_path("Some/Model.pkl"),
                 xe._spage_cache_path("/ref.rds", "g1,g2"), xe._seg_run_dir("baysor", "deadbeef"),
                 xe._seg_run_dir("proseg", "deadbeef"),
                 xe._overview_cache_path(xe.DATA["data_dir"], 0),
                 xe._overview_cache_path(xe.DATA["data_dir"], 0, medium=True)):
        assert os.path.realpath(path).startswith(root + os.sep), path
    # two datasets with the same folder name must not share cache entries
    name = xe._dataset_name()
    run = os.path.basename(xe._seg_run_dir("baysor", "deadbeef"))
    assert run.startswith(f"baysor_{name}_") and run.endswith("_deadbeef") and run != f"baysor_{name}_deadbeef"


def test_no_module_constant_points_into_the_home_cache(xe):
    home_cache = os.path.join(os.path.expanduser("~"), ".xenium_explorer_cache")
    assert {"CACHE_BASE_DIR", "_CELLNEST_DB_CACHE", "SPAGE_REPO"} <= set(vars(xe))   # all modules
    leaks = {k: v for k, v in vars(xe).items()
             if isinstance(v, str) and v.startswith(home_cache)}
    assert not leaks, leaks


# ── cells ────────────────────────────────────────────────────────────────────
def test_cells_are_the_bundles_cells_in_file_order(D, raw):
    df = D["df"]
    assert list(df.index.astype(str)) == list(raw.cells.index)
    assert df.index.is_unique
    for col in ("x_centroid", "y_centroid", "transcript_counts", "cell_area", "nucleus_area", "total_counts"):
        if col in raw.cells.columns:
            np.testing.assert_allclose(df[col].to_numpy(dtype=float),
                                       raw.cells[col].to_numpy(dtype=float), rtol=0, atol=1e-6,
                                       err_msg=col)
    assert len(df) == raw.experiment.get("num_cells", len(df))


def test_coordinates_are_micrometres_not_pixels(D, raw):
    """Centroids must lie inside the extent of the cell boundary polygons (µm)."""
    polys = raw.boundaries("cell")
    if not polys:
        pytest.skip("bundle has no cell boundaries")
    df = D["df"]
    sample = list(df.index[:: max(1, len(df) // 500)])[:500]
    inside = 0
    for cid in sample:
        v = polys[str(cid)]
        inside += (v[:, 0].min() <= df.at[cid, "x_centroid"] <= v[:, 0].max()
                   and v[:, 1].min() <= df.at[cid, "y_centroid"] <= v[:, 1].max())
    assert inside == len(sample)


# ── gene-expression-only features ────────────────────────────────────────────
def test_only_gene_expression_features_are_kept(xe, D, raw):
    _mat, names, ftype, _bc = raw.matrix
    if ftype is None:
        pytest.skip("matrix has no feature_type")
    genes = [n for n, t in zip(names, ftype) if t == "Gene Expression"]
    controls = {n for n, t in zip(names, ftype) if t != "Gene Expression"}
    assert D["gene_names"] == genes, "gene list must be the Gene Expression features, in file order"
    assert not controls & set(D["gene_names"])
    assert xe._panel_gene_names() == genes
    assert D["expr"].shape == (len(D["df"]), len(genes))
    assert D["gene_name_to_idx"] == {g: i for i, g in enumerate(genes)}
    assert sorted(genes) == xe._sorted_gene_names and xe._default_gene == sorted(genes)[0]
    if controls:        # the filter did something on this bundle
        assert len(names) > len(genes)
    declared = raw.experiment.get("panel_num_targets_predesigned")
    if declared:
        assert len(genes) == declared + raw.experiment.get("panel_num_targets_custom", 0)


def test_no_control_named_gene_in_the_panel(xe, D):
    assert not xe._tx_control_feature_mask(D["gene_names"]).any()


def test_expression_matrix_equals_the_h5_gene_rows(D, raw):
    counts, names, barcodes = raw.gene_counts
    assert D["barcodes"] == barcodes
    assert list(D["df"].index.astype(str)) == barcodes, "matrix columns and cells.parquet rows must align"
    np.testing.assert_array_equal(np.asarray(D["df_to_expr"]), np.arange(len(barcodes)))
    assert sp.issparse(D["expr"]) and D["expr"].format == "csr"
    assert tl.sparse_equal(D["expr"], counts)
    assert D["expr_csc"].format == "csc" and tl.sparse_equal(D["expr_csc"], counts)


def test_row_sums_equal_transcript_counts(D, raw):
    sums = np.asarray(D["expr"].sum(axis=1)).ravel()
    np.testing.assert_array_equal(sums, raw.cells["transcript_counts"].to_numpy(dtype=float))


def test_matrix_equals_a_recount_of_the_transcripts(D, raw):
    """cell x gene counts rebuilt from transcripts.parquet (Q >= 20, gene features,
    assigned to a cell) must reproduce the matrix: this ties the cell order, the gene
    order and the gene-only filter to a second file."""
    if not raw.has("transcripts.parquet"):
        pytest.skip("bundle has no transcripts.parquet")
    if not str(raw.experiment.get("analysis_sw_version", "")).startswith("xenium"):
        pytest.skip("not an instrument bundle: its matrix need not be a recount of its transcripts")
    df = D["df"]
    cols = ["cell_id", "feature_name", "qv", "is_gene", "codeword_category"]
    if raw.n_transcripts <= 5_000_000:
        rows = np.arange(len(df))
        tx = raw.transcripts(columns=cols)
    else:
        rows = np.unique(np.linspace(0, len(df) - 1, 300).astype(int))
        tx = raw.transcripts_of_cells(df.index[rows], columns=cols)
    tx = tl.gene_tx(tx)
    ids = df.index[rows].astype(str)
    recount = tl.count_matrix(tx["cell_id"], tx["feature_name"], ids, D["gene_names"])
    assert recount.sum() > 0
    assert tl.sparse_equal(recount, D["expr"][rows])


def test_get_gene_expression_is_log1p_of_the_counts(xe, D, raw):
    counts, names, _ = raw.gene_counts
    totals = np.asarray(counts.sum(axis=0)).ravel()
    picks = {names[int(totals.argmax())], names[0], names[-1], names[len(names) // 2]}
    for g in picks:
        col = np.asarray(counts[:, names.index(g)].todense()).ravel().astype(np.float32)
        got = xe.get_gene_expression(g)
        assert got.shape == (len(D["df"]),)
        np.testing.assert_allclose(got, np.log1p(col), rtol=1e-6)
        np.testing.assert_allclose(xe._get_expr_values(g), np.log1p(col), rtol=1e-6)
    with pytest.raises(KeyError):
        xe.get_gene_expression("__no_such_gene__")
    assert xe._get_expr_values("__no_such_gene__") is None


def test_gene_cache_is_bounded(xe, D):
    for g in D["gene_names"][: xe._GENE_EXPR_CACHE_MAX + 20]:
        xe.get_gene_expression(g)
    assert 0 < len(xe._gene_expr_cache) <= xe._GENE_EXPR_CACHE_MAX


# ── clustering / UMAP from the analysis folder ───────────────────────────────
def test_cluster_columns_match_the_analysis_csvs(xe, D, raw):
    if not D["cluster_methods"]:
        pytest.skip("bundle has no clustering")
    df = D["df"]
    checked = 0
    for method in D["cluster_methods"]:
        col = xe.cluster_col(method)
        assert col == f"clust__{method}" and col in df.columns
        truth = raw.clusters(method)
        if truth is None:
            continue
        got = df[col]
        common = got.index.astype(str).intersection(truth.index.astype(str))
        assert len(common) == len(truth), "every clustered barcode is a cell"
        np.testing.assert_array_equal(got.loc[common].to_numpy(dtype=int), truth.loc[common].to_numpy(dtype=int))
        rest = got.drop(index=common)
        assert (rest.to_numpy(dtype=int) == 0).all(), "cells the analysis left out are cluster 0"
        checked += 1
    assert checked, "no clusters.csv found for any listed method"


def test_umap_matches_the_projection_csv(D, raw):
    truth = raw.umap
    if truth is None:
        pytest.skip("bundle has no UMAP projection")
    df = D["df"]
    got = df.loc[truth.index, ["umap_1", "umap_2"]].to_numpy(dtype=float)
    np.testing.assert_allclose(got, truth[["UMAP-1", "UMAP-2"]].to_numpy(dtype=float), atol=1e-6)
    assert df.drop(index=truth.index)[["umap_1", "umap_2"]].isna().all().all()


# ── boundaries held in memory ────────────────────────────────────────────────
@pytest.mark.parametrize("kind,key", [("cell", "cell_bounds"), ("nucleus", "nucleus_bounds")])
def test_in_memory_boundaries_equal_the_parquet(D, raw, kind, key):
    truth = raw.boundaries(kind)
    if not truth:
        pytest.skip(f"bundle has no {kind} boundaries")
    bounds = D[key]
    assert {str(k) for k in bounds} <= set(D["df"].index.astype(str))
    assert len(bounds) == len(truth)
    for cid in list(truth)[:: max(1, len(truth) // 300)]:
        xs, ys = bounds[cid] if cid in bounds else bounds[type(next(iter(bounds)))(cid)]
        xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        ok = ~np.isnan(xs)
        got = np.column_stack([xs[ok], ys[ok]])
        want = truth[cid]
        # same ring: the stored polygon may start at another vertex / drop the closing one
        assert {tuple(np.round(p, 3)) for p in got} == {tuple(np.round(p, 3)) for p in want}


def test_viewport_cell_ids_is_a_centroid_box_query(xe, D):
    df = D["df"]
    x, y = df["x_centroid"].to_numpy(), df["y_centroid"].to_numpy()
    cx, cy = float(np.median(x)), float(np.median(y))
    w = max(40.0, float(np.ptp(x)) / 10)
    relayout = {"xaxis.range[0]": cx - w, "xaxis.range[1]": cx + w,
                "yaxis.range[0]": -(cy + w), "yaxis.range[1]": -(cy - w)}      # plot y = -y_centroid
    got = xe.viewport_cell_ids(relayout)
    want = df.index[(x >= cx - w) & (x <= cx + w) & (y >= cy - w) & (y <= cy + w)]
    if got is None:
        pytest.skip("viewport holds too many cells for a boundary query on this bundle")
    assert set(map(str, got)) == set(map(str, want)) and len(want) > 0
    assert not xe.viewport_cell_ids({})          # no range -> nothing to draw boundaries for


# ── dataset identity ─────────────────────────────────────────────────────────
def test_dataset_key_and_cache_tags(xe, D, bundle_dir):
    import hashlib
    assert xe._dataset_name() == os.path.basename(str(bundle_dir))
    assert os.path.realpath(xe._dataset_key()) == os.path.realpath(bundle_dir)
    tag = hashlib.md5(xe._dataset_key().encode()).hexdigest()[:8]
    assert os.path.basename(xe._roi_cache_path()) == f"rois_{tag}.json"
    assert xe._list_cached_seg_runs() == [], "a fresh cache has no segmentation runs"


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200 and b"ok" in r.data.lower()


def test_reload_of_generated_store_is_identical(xe, D):
    """Second start (store already generated) must give the same dataset as the first."""
    again = xe._load_from_xenium_sdata(D["sdata_path"], D["data_dir"])
    assert again["gene_names"] == D["gene_names"] and again["barcodes"] == D["barcodes"]
    assert again["cluster_methods"] == D["cluster_methods"]
    assert tl.sparse_equal(again["expr"], D["expr"])
    base_cols = [c for c in D["df"].columns if c in again["df"].columns]
    assert set(xe._test_baseline.columns) <= set(again["df"].columns)
    pd.testing.assert_frame_equal(again["df"][base_cols], D["df"][base_cols], check_dtype=False)
    assert set(map(str, again["cell_bounds"])) == set(map(str, D["cell_bounds"]))
