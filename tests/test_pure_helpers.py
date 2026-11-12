"""Unit tests of the data-independent helpers, on synthetic input.

The functions live in the xenium_app package, which cannot be imported without loading a
dataset, so they are reached through the session's ``xe`` import; nothing here depends
on which bundle that import loaded. The SPLIT gene re-alignment is not a function at
all (it is a block inside _run_split_correction / run_split_from_zarr), so that block
is cut out of the source with ``ast`` and executed on synthetic arrays.
"""
from __future__ import annotations

import ast
import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import xe_testlib as tl

# 2 x 2 grid of 110-wide patches overlapping by 20 µm: ownership lines at x = 100, y = 100
GRID = [(0.0, 0.0, 110.0, 110.0), (90.0, 0.0, 200.0, 110.0),
        (0.0, 90.0, 110.0, 200.0), (90.0, 90.0, 200.0, 200.0)]


def _depth(x, y, box):
    x0, y0, x1, y1 = box
    return np.minimum(np.minimum(x - x0, x1 - x), np.minimum(y - y0, y1 - y))


def _inside(x, y, box):
    x0, y0, x1, y1 = box
    return (x >= x0) & (x < x1) & (y >= y0) & (y < y1)


def _brute_owner(x, y, bounds, ok_idx):
    """Independent statement of the rule: the patch a point sits deepest inside, ties to
    the lower index; -1 when no patch in ok_idx contains the point."""
    owner = np.full(len(x), -1)
    best = np.full(len(x), -np.inf)
    for j in sorted(ok_idx):
        d = np.where(_inside(x, y, bounds[j]), _depth(x, y, bounds[j]), -np.inf)
        better = d > best
        owner[better] = j
        best[better] = d[better]
    return owner


# ── _patch_owned_mask ────────────────────────────────────────────────────────
@pytest.mark.parametrize("ok_idx", [[0, 1, 2, 3], [0, 3], [1, 2, 3], [2]])
def test_patch_ownership_partitions_the_points(xe, ok_idx):
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.uniform(0, 200, 4000), [100.0, 100.0, 95.0, 105.0, 100.0]])
    y = np.concatenate([rng.uniform(0, 200, 4000), [100.0, 50.0, 100.0, 100.0, 150.0]])
    expected = _brute_owner(x, y, GRID, ok_idx)
    n_owners = np.zeros(len(x), dtype=int)
    for pi in ok_idx:
        mine = _inside(x, y, GRID[pi])
        keep = xe._patch_owned_mask(x[mine], y[mine], pi, GRID, ok_idx)
        assert keep.dtype == bool and len(keep) == int(mine.sum())
        np.testing.assert_array_equal(keep, expected[mine] == pi)
        n_owners[np.flatnonzero(mine)[keep]] += 1
    covered = expected >= 0
    assert (n_owners[covered] == 1).all(), "a covered point must be kept by exactly one patch"
    assert (n_owners[~covered] == 0).all()


def test_patch_ownership_splits_the_overlap_at_its_midline(xe):
    y = np.full(5, 50.0)
    x = np.array([91.0, 99.9, 100.0, 100.1, 109.0])
    left = xe._patch_owned_mask(x, y, 0, GRID, [0, 1])
    right = xe._patch_owned_mask(x, y, 1, GRID, [0, 1])
    assert left.tolist() == [True, True, True, False, False]      # the tie at x=100 goes to patch 0
    assert right.tolist() == [False, False, False, True, True]


def test_patch_ownership_cedes_nothing_to_failed_patches(xe):
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 110, 500), rng.uniform(0, 110, 500)
    # patch 1 did not return cells: patch 0 keeps its whole box
    assert xe._patch_owned_mask(x, y, 0, GRID, [0]).all()
    assert xe._patch_owned_mask(x, y, 0, GRID, [0, 2]).sum() < 500


# ── _patch_seam_dedup_mask ───────────────────────────────────────────────────
PAIR = [(0.0, 0.0, 110.0, 100.0), (90.0, 0.0, 200.0, 100.0)]


def _square(cx, cy, half=3.0):
    return (np.array([cx - half, cx + half, cx + half, cx - half, cx - half]),
            np.array([cy - half, cy - half, cy + half, cy + half, cy - half]))


def _dedup(xe, cells, polys=None, cfg=None):
    """cells: list of (id, patch, x, y, n_tx) -> set of kept ids."""
    ids = [c[0] for c in cells]
    keep = xe._patch_seam_dedup_mask(ids, [c[1] for c in cells], [c[2] for c in cells],
                                     [c[3] for c in cells], [c[4] for c in cells],
                                     polys or {}, PAIR, [0, 1], cfg=cfg)
    assert keep.dtype == bool and len(keep) == len(cells)
    return {i for i, k in zip(ids, keep) if k}


def test_seam_dedup_drops_the_shallower_twin(xe):
    cells = [("a", 0, 99.5, 50.0, 10),      # 10.5 µm inside patch 0
             ("b", 1, 100.4, 50.0, 99),     # 10.4 µm inside patch 1: same cell, shallower
             ("far0", 0, 30.0, 50.0, 5), ("far1", 1, 170.0, 50.0, 5)]
    assert _dedup(xe, cells) == {"a", "far0", "far1"}


def test_seam_dedup_tie_goes_to_more_transcripts_then_lower_patch(xe):
    assert _dedup(xe, [("a", 0, 99.5, 50.0, 10), ("b", 1, 100.5, 50.0, 30)]) == {"b"}
    assert _dedup(xe, [("a", 0, 99.5, 50.0, 30), ("b", 1, 100.5, 50.0, 30)]) == {"a"}


def test_seam_dedup_keeps_neighbours_that_are_not_twins(xe):
    cells = [("a", 0, 99.0, 50.0, 10), ("b", 1, 101.0, 50.0, 10),      # 2 µm apart: > 1.5
             ("c", 0, 99.0, 20.0, 10), ("d", 0, 99.5, 20.0, 10)]       # same patch, 0.5 apart
    assert _dedup(xe, cells) == {"a", "b", "c", "d"}


def test_seam_dedup_uses_polygon_overlap(xe):
    # centroids 3 µm apart (beyond the centroid test); 6x6 squares overlap by 50% exactly
    # at 3 µm, by 2/3 at 2 µm (but then the centroid test is not what fires: 2 > 1.5)
    cells = [("a", 0, 98.9, 50.0, 10), ("b", 1, 100.9, 50.0, 10)]
    polys = {"a": _square(98.9, 50.0), "b": _square(100.9, 50.0)}
    assert _dedup(xe, cells) == {"a", "b"}, "without polygons these are two cells"
    assert _dedup(xe, cells, polys) == {"a"}, "polygons overlapping by 2/3 are one cell"
    cells = [("a", 0, 97.5, 50.0, 10), ("b", 1, 101.5, 50.0, 10)]      # squares overlap by 1/3
    polys = {"a": _square(97.5, 50.0), "b": _square(101.5, 50.0)}
    assert _dedup(xe, cells, polys) == {"a", "b"}


def test_seam_dedup_ignores_cells_far_from_the_seam(xe):
    # identical positions in different patches, but 60 µm from the ownership line: such
    # cells cannot both have survived ownership, the function does not look at them
    cfg = dict(xe._SEAM_DEDUP)
    cells = [("a", 0, 40.0, 50.0, 10), ("b", 1, 40.0, 50.0, 10)]
    assert _dedup(xe, cells, cfg=cfg) == {"a", "b"}


def test_seam_dedup_is_order_independent(xe):
    rng = np.random.default_rng(3)
    cells, polys = [], {}
    for k in range(60):
        cx, cy = rng.uniform(96, 104), rng.uniform(5, 95)
        for patch in (0, 1):
            jx, jy = rng.normal(0, 0.6, 2)
            cid = f"p{patch}_{k}"
            cells.append((cid, patch, cx + jx, cy + jy, int(rng.integers(5, 50))))
            polys[cid] = _square(cx + jx, cy + jy, half=2.0)
    ref = _dedup(xe, cells, polys)
    assert 0 < len(ref) < len(cells), "the synthetic layout should contain twins"
    for seed in range(4):
        order = np.random.default_rng(seed).permutation(len(cells))
        assert _dedup(xe, [cells[i] for i in order], polys) == ref
    # no two kept cells of different patches are still twins by the centroid rule
    kept = [c for c in cells if c[0] in ref]
    for i, a in enumerate(kept):
        for b in kept[i + 1:]:
            if a[1] != b[1]:
                assert np.hypot(a[2] - b[2], a[3] - b[3]) >= xe._SEAM_DEDUP["max_centroid_dist"]


def test_seam_dedup_needs_two_patches(xe):
    keep = xe._patch_seam_dedup_mask(["a", "b"], [0, 0], [99.5, 99.6], [50, 50], [1, 1], {}, PAIR, [0])
    assert keep.all()


# ── _reseg_patch_keep_mask ───────────────────────────────────────────────────
def _patch_cells(seed=5, n=300):
    """Cells as the patch_XXXX directories of a run hold them: each patch reports every
    cell whose centroid lies in its box, so the overlap strip is reported twice."""
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 200, n), rng.uniform(0, 200, n)
    rows = []
    for pi, box in enumerate(GRID):
        for k in np.flatnonzero(_inside(x, y, box)):
            rows.append((f"{pi}_{k}", pi, x[k], y[k]))
    df = pd.DataFrame(rows, columns=["id", "patch", "x", "y"])
    return df


def test_reseg_keep_mask_recomputes_ownership_from_params(xe, tmp_path):
    df = _patch_cells()
    (tmp_path / "params.json").write_text(json.dumps({"patch_bounds": GRID, "patch_ok": [0, 1, 3]}))
    keep = xe._reseg_patch_keep_mask(str(tmp_path), str(tmp_path / "missing.zarr"),
                                     df["id"], df["patch"].values, df["x"].values, df["y"].values)
    in_ok = df["patch"].isin([0, 1, 3]).values
    owner = _brute_owner(df["x"].values, df["y"].values, GRID, [0, 1, 3])
    np.testing.assert_array_equal(keep, in_ok & (owner == df["patch"].values))
    # every point covered by a successful patch is kept exactly once
    kept_points = df.loc[keep, "id"].str.split("_").str[1]
    assert kept_points.is_unique
    covered = {str(k) for k in df.loc[in_ok, "id"].str.split("_").str[1]}
    assert set(kept_points) == covered


def test_reseg_keep_mask_without_patch_ok_uses_patches_seen(xe, tmp_path):
    df = _patch_cells()
    df = df[df["patch"] != 2]
    (tmp_path / "params.json").write_text(json.dumps({"patch_bounds": GRID}))
    keep = xe._reseg_patch_keep_mask(str(tmp_path), str(tmp_path / "missing.zarr"),
                                     df["id"], df["patch"].values, df["x"].values, df["y"].values)
    owner = _brute_owner(df["x"].values, df["y"].values, GRID, [0, 1, 3])
    np.testing.assert_array_equal(keep, owner == df["patch"].values)


def test_reseg_keep_mask_applies_recorded_seam_dedup(xe, tmp_path):
    ids = ["0_a", "1_b", "0_c", "1_d"]
    patch = np.array([0, 1, 0, 1])
    x = np.array([99.5, 100.4, 30.0, 170.0])
    y = np.array([50.0, 50.0, 50.0, 50.0])
    base = {"patch_bounds": PAIR, "patch_ok": [0, 1]}
    (tmp_path / "params.json").write_text(json.dumps(base))
    keep = xe._reseg_patch_keep_mask(str(tmp_path), str(tmp_path / "none.zarr"), ids, patch, x, y,
                                     n_tx=[5, 5, 5, 5], polys={})
    assert keep.tolist() == [True, True, True, True], "runs recorded without seam_dedup are not de-duplicated"
    (tmp_path / "params.json").write_text(json.dumps({**base, "seam_dedup": dict(xe._SEAM_DEDUP)}))
    keep = xe._reseg_patch_keep_mask(str(tmp_path), str(tmp_path / "none.zarr"), ids, patch, x, y,
                                     n_tx=[5, 5, 5, 5], polys={})
    assert keep.tolist() == [True, False, True, True]


def test_reseg_keep_mask_prefers_the_runs_zarr_table(xe, tmp_path):
    import anndata as ad
    df = _patch_cells(n=40)
    live_kept = df["id"].iloc[::3].tolist()
    for prefix in ("", "p7_"):          # live Baysor patch runs prefix ids with p<patch>_
        zarr_path = tmp_path / f"run{len(prefix)}.zarr"
        table = ad.AnnData(X=sp.csr_matrix((len(live_kept), 2), dtype=np.float32),
                           obs=pd.DataFrame(index=[prefix + i for i in live_kept]))
        import zarr
        root = zarr.open_group(str(zarr_path), mode="w", zarr_format=ad.settings.zarr_write_format)
        ad.io.write_elem(root, "tables/table", table)
        # params.json would give a different answer: the table must win
        (tmp_path / "params.json").write_text(json.dumps({"patch_bounds": GRID}))
        keep = xe._reseg_patch_keep_mask(str(tmp_path), str(zarr_path), df["id"], df["patch"].values,
                                         df["x"].values, df["y"].values)
        assert df.loc[keep, "id"].tolist() == live_kept


def test_reseg_keep_mask_none_without_layout(xe, tmp_path):
    assert xe._reseg_patch_keep_mask(str(tmp_path), str(tmp_path / "x.zarr"), ["a"], [0], [1.0], [1.0]) is None
    (tmp_path / "params.json").write_text(json.dumps({"tool": "baysor"}))
    assert xe._reseg_patch_keep_mask(str(tmp_path), str(tmp_path / "x.zarr"), ["a"], [0], [1.0], [1.0]) is None


def test_patch_bounds_tag_is_stable_and_layout_sensitive(xe):
    t = xe._patch_bounds_tag(GRID)
    assert t == xe._patch_bounds_tag([tuple(np.float32(v) for v in b) for b in GRID])
    assert t == xe._patch_bounds_tag([tuple(v + 0.004 for v in b) for b in GRID]), "rounded to 0.1 µm"
    assert t != xe._patch_bounds_tag(GRID[:3])
    assert t != xe._patch_bounds_tag([GRID[1], GRID[0], GRID[2], GRID[3]])
    assert len(t) == 8 and int(t, 16) >= 0


# ── transcript filters ───────────────────────────────────────────────────────
def test_tx_gene_filters_by_schema(xe):
    f = xe._tx_gene_filters
    assert f(["qv", "is_gene", "codeword_category"]) == [("qv", ">=", 20.0), ("is_gene", "==", True)]
    assert f(["qv", "is_gene"], qv_min=30) == [("qv", ">=", 30.0), ("is_gene", "==", True)]
    assert f(["qv", "is_gene"], qv_min=None) == [("is_gene", "==", True)]
    got = f(["qv", "codeword_category"])
    assert got[0] == ("qv", ">=", 20.0) and got[1][:2] == ("codeword_category", "not in")
    assert set(got[1][2]) == {"unassigned_codeword", "negative_control_codeword",
                              "negative_control_probe", "genomic_control_probe", "deprecated_codeword"}
    assert f(["feature_name", "qv"]) == [("qv", ">=", 20.0)]      # name prefixes handled separately
    assert f(["feature_name"]) == []


def _synthetic_tx(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    names = np.array(["Actb", "Gapdh", "Cd3e", "NegControlProbe_00042", "NegControlCodeword_0500",
                      "BLANK_0006", "UnassignedCodeword_0003", "DeprecatedCodeword_0321",
                      "Intergenic_Region_12", "GenomicControl_7", "Blank1", "Negr1"])
    category = {"Actb": "predesigned_gene", "Gapdh": "predesigned_gene", "Cd3e": "custom_gene",
                "Blank1": "predesigned_gene", "Negr1": "predesigned_gene",
                "NegControlProbe_00042": "negative_control_probe",
                "NegControlCodeword_0500": "negative_control_codeword", "BLANK_0006": "unassigned_codeword",
                "UnassignedCodeword_0003": "unassigned_codeword",
                "DeprecatedCodeword_0321": "deprecated_codeword",
                "Intergenic_Region_12": "genomic_control_probe", "GenomicControl_7": "genomic_control_probe"}
    fn = rng.choice(names, n)
    cat = np.array([category[v] for v in fn])
    return pd.DataFrame({
        "transcript_id": np.arange(n, dtype=np.uint64),
        "feature_name": fn,
        "x_location": rng.uniform(0, 100, n).astype(np.float32),
        "y_location": rng.uniform(0, 100, n).astype(np.float32),
        "qv": rng.choice([5.0, 19.99, 20.0, 40.0], n).astype(np.float32),
        "codeword_category": cat,
        "is_gene": np.isin(cat, ["predesigned_gene", "custom_gene"]),
    })


@pytest.mark.parametrize("schema", ["is_gene", "codeword_category", "names_only"])
def test_tx_filters_select_the_matrix_transcripts(xe, tmp_path, schema):
    """The three generations of transcripts.parquet must all reduce to: Q >= 20 genes."""
    import pyarrow.parquet as pq
    tx = _synthetic_tx()
    expected = set(tx.loc[(tx["qv"] >= 20) & tx["is_gene"], "transcript_id"])
    assert 0 < len(expected) < len(tx)
    drop = {"is_gene": [], "codeword_category": ["is_gene"],
            "names_only": ["is_gene", "codeword_category"]}[schema]
    src = tmp_path / "transcripts.parquet"
    tx.drop(columns=drop).to_parquet(src, row_group_size=500)

    names = pq.read_schema(src).names
    got = xe._read_tx_region(str(src), extra_filters=xe._tx_gene_filters(names))
    got = xe._tx_drop_control_features(got) if "is_gene" not in names else got
    assert set(got["transcript_id"]) == expected

    dst = tmp_path / "gene_only.parquet"
    n = xe._tx_write_gene_only(str(src), str(dst))
    written = pd.read_parquet(dst)
    assert n == len(written) == len(expected)
    assert set(written["transcript_id"]) == expected
    assert list(written.columns) == list(tx.drop(columns=drop).columns), "all columns are kept"
    assert not (tmp_path / "gene_only.parquet.tmp").exists()


def test_read_tx_region_is_an_inclusive_box(xe, tmp_path):
    tx = _synthetic_tx(seed=2)
    src = tmp_path / "t.parquet"
    tx.to_parquet(src, row_group_size=300)
    x0, x1, y0, y1 = 10.0, 55.5, 20.25, 80.0
    got = xe._read_tx_region(str(src), columns=["transcript_id"], x0=x0, x1=x1, y0=y0, y1=y1)
    inside = ((tx.x_location >= x0) & (tx.x_location <= x1) & (tx.y_location >= y0) & (tx.y_location <= y1))
    assert sorted(got["transcript_id"]) == sorted(tx.loc[inside, "transcript_id"])
    assert len(xe._read_tx_region(str(src), columns=["transcript_id"])) == len(tx)


def test_control_feature_mask_handles_str_bytes_and_categoricals(xe):
    names = ["Actb", "NegControlProbe_1", "BLANK_0001", "Blank1", "UnassignedCodeword_9",
             "DeprecatedCodeword_2", "Intergenic_Region_3", "GenomicControl_4", "Negr1", "xBLANK_1"]
    want = [False, True, True, False, True, True, True, True, False, False]
    f = xe._tx_control_feature_mask
    assert f(names).tolist() == want
    assert f([n.encode() for n in names]).tolist() == want
    assert f(pd.Categorical(names)).tolist() == want
    with_null = pd.Categorical(names + [None])
    assert f(with_null).tolist() == want + [False]


def test_real_bundle_transcript_filter_matches_independent_filter(xe, raw):
    """On the bundle under test: the app's pyarrow filter == an independent pandas filter."""
    import pyarrow.parquet as pq
    if not raw.has("transcripts.parquet"):
        pytest.skip("bundle has no transcripts.parquet")
    if raw.n_transcripts > 5_000_000:
        pytest.skip("whole-file comparison is for small bundles; larger ones are covered per region")
    src = str(raw.path / "transcripts.parquet")
    names = pq.read_schema(src).names
    got = xe._read_tx_region(src, columns=["transcript_id"], extra_filters=xe._tx_gene_filters(names))
    truth = tl.gene_tx(raw.transcripts())
    assert sorted(got["transcript_id"]) == sorted(truth["transcript_id"])
    assert 0 < len(truth) < raw.n_transcripts


def test_proseg_prior_type_defaults_to_nucleus(xe):
    assert [xe._proseg_prior_type(v) for v in ("nucleus", "cell", "none", None, "", "NUCLEUS")] == \
        ["nucleus", "cell", "nucleus", "nucleus", "nucleus", "nucleus"]


def test_seg_tool_parses_source_values(xe):
    assert [xe._seg_tool(v) for v in ("xenium", None, "", "baysor:abc12345", "proseg:xyz", "baysor")] == \
        ["xenium", "xenium", "xenium", "baysor", "proseg", "baysor"]


# ── expression builders for resegmentation output ────────────────────────────
def test_build_baysor_expr_counts_cell_gene_pairs(xe):
    genes = xe._panel_gene_names()
    rng = np.random.default_rng(0)
    cells = pd.DataFrame(index=pd.Index([f"p0_{i}" for i in range(1, 9)], name="cell"))
    seg = pd.DataFrame({"cell": rng.choice(list(cells.index) + ["p0_999", None], 600),
                        "gene": rng.choice(genes[:40] + ["NotAGene"], 600)})
    seg = seg[seg["cell"].notna()]
    mat = xe._build_baysor_expr(seg, cells)
    assert tl.sparse_equal(mat, tl.count_matrix(seg["cell"], seg["gene"], cells.index, genes))
    assert mat.shape == (8, len(genes)) and mat.sum() < len(seg)
    # pre-aggregated input (column n) gives the same matrix
    agg = seg.groupby(["cell", "gene"]).size().rename("n").reset_index()
    assert tl.sparse_equal(xe._build_baysor_expr(agg, cells), mat)


def test_build_proseg_expr_skips_unassigned(xe):
    genes = xe._panel_gene_names()
    rng = np.random.default_rng(1)
    cells = pd.DataFrame(index=pd.Index([10_000_001, 10_000_002, 20_000_005], name="cell"))
    assignment = rng.choice([10_000_001, 10_000_002, 20_000_005, 4294967295, 77], 500)
    tx = pd.DataFrame({"gene": pd.Categorical(rng.choice(genes[:25], 500)), "assignment": assignment})
    mat = xe._build_proseg_expr(tx, cells)
    truth = tl.count_matrix(tx["assignment"], tx["gene"].astype(str), cells.index, genes)
    assert tl.sparse_equal(mat, truth)
    assert mat.sum() == int(np.isin(assignment, cells.index).sum())
    assert xe._build_proseg_expr(None, cells) is None


def test_align_zarr_rows_to_cells(xe):
    mat = sp.csr_matrix(np.arange(12, dtype=np.float32).reshape(4, 3))
    cells = pd.DataFrame(index=["c", "a", "d", "b"])
    same = xe._align_zarr_rows_to_cells(mat, ["c", "a", "d", "b"], cells, "t")
    assert same is mat
    out = xe._align_zarr_rows_to_cells(mat, ["a", "b", "c", "d"], cells, "t")
    np.testing.assert_array_equal(out.toarray(), mat.toarray()[[2, 0, 3, 1]])
    out = xe._align_zarr_rows_to_cells(mat, ["p0_a", "p0_b", "p1_c", "p1_d"], cells, "t")
    np.testing.assert_array_equal(out.toarray(), mat.toarray()[[2, 0, 3, 1]])
    assert xe._align_zarr_rows_to_cells(mat, ["w", "x", "y", "z"], cells, "t") is None
    assert xe._align_zarr_rows_to_cells(None, ["a"], cells, "t") is None


# ── neighbourhood statistics (formulae) ──────────────────────────────────────
def test_neighbor_matrix_and_scores_on_a_hand_example(xe):
    labels = np.array([0, 0, 1, 1, 2])
    neigh = np.array([[1, 2], [0, 2], [3, 0], [2, 4], [3, 4]])     # last row has a self-neighbour
    raw = xe._build_raw_neighbor_matrix(labels, neigh, 3)
    want = np.zeros((3, 3))
    for i, row in enumerate(neigh):
        for j in row:
            if j != i:
                want[labels[i], labels[j]] += 1
    np.testing.assert_array_equal(raw, want)
    assert raw.sum() == 9
    counts = np.bincount(labels, minlength=3).astype(float)
    freqs = counts / counts.sum()
    co = xe._compute_cooccurrence_from_raw(raw, freqs)
    np.testing.assert_allclose(co, np.log2(want / (np.outer(freqs, freqs) * 9 + 1e-9) + 1e-9))
    en = xe._compute_enrichment_from_raw(raw, counts, freqs)
    # P(neighbour is b | cell is a) / P(b): rows are normalised by their own pair count,
    # so the score does not grow with the number of neighbours k
    np.testing.assert_allclose(en, np.log2((want / want.sum(axis=1, keepdims=True)) / freqs[None, :] + 1e-9))
    mixed = np.outer(counts, freqs) * 4            # perfectly mixed tissue, k = 4
    np.testing.assert_allclose(xe._compute_enrichment_from_raw(mixed, counts, freqs), 0, atol=1e-6)
    np.testing.assert_allclose(xe._compute_cooccurrence_from_raw(mixed, freqs), 0, atol=1e-6)
    np.testing.assert_array_equal(raw, want), "inputs are not modified in place"
    np.testing.assert_array_equal(xe._compute_cooccurrence_from_raw(np.zeros((2, 2)), np.array([.5, .5])), 0)


# ── SPLIT: corrected counts back onto the panel's gene columns ───────────────
def _split_realign_block(path, function):
    tree = ast.parse(path.read_text())
    fn = tl.function_node(tree, function)
    hits = [n for n in ast.walk(fn) if isinstance(n, ast.If)
            and "corrected_gene_ids is not None" in ast.unparse(n.test)
            and "panel_genes" in ast.unparse(n.test)]
    assert len(hits) == 1, f"re-alignment block not found in {function} (moved or rewritten?)"
    return tl.compile_block(hits)


SPLIT_SOURCES = [(tl.app_source_of("_run_split_correction"), "_run_split_correction"),
                 (tl.REPO / "functions" / "run_split.py", "run_split_from_zarr")]


@pytest.mark.parametrize("path,function", SPLIT_SOURCES, ids=["app", "functions/run_split.py"])
def test_split_columns_are_realigned_by_gene_name(path, function):
    code = _split_realign_block(path, function)
    panel = [f"G{i}" for i in range(12)]
    rng = np.random.default_rng(0)
    # purify() returns the genes RCTD used: a shuffled subset of the panel, here with one
    # name the panel does not have (it must be dropped, not shift the columns)
    returned = ["G7", "G2", "NotInPanel", "G11", "G0", "G5"]
    dense = rng.integers(0, 6, size=(9, len(returned))).astype(np.float32)
    ns = {"np": np, "sp": sp, "sp_": sp, "corrected_mat": sp.csr_matrix(dense),
          "corrected_gene_ids": list(returned), "panel_genes": list(panel), "n_panel": len(panel)}
    exec(code, ns)
    out = ns["corrected_mat"]
    expected = (pd.DataFrame(dense, columns=returned).drop(columns="NotInPanel")
                .reindex(columns=panel, fill_value=0.0).to_numpy())
    assert out.shape == (9, len(panel))
    np.testing.assert_array_equal(out.toarray(), expected)
    assert sp.issparse(out) and out.format == "csr"


@pytest.mark.parametrize("path,function", SPLIT_SOURCES, ids=["app", "functions/run_split.py"])
def test_split_realign_is_a_noop_for_panel_order_and_refuses_unnamed_mismatch(path, function):
    code = _split_realign_block(path, function)
    panel = ["A", "B", "C"]
    mat = sp.csr_matrix(np.arange(6, dtype=np.float32).reshape(2, 3))
    ns = {"np": np, "sp": sp, "sp_": sp, "corrected_mat": mat, "corrected_gene_ids": list(panel),
          "panel_genes": panel, "n_panel": 3}
    exec(code, ns)
    assert ns["corrected_mat"] is mat
    ns.update(corrected_gene_ids=None, corrected_mat=mat[:, :2])
    with pytest.raises(RuntimeError, match="cannot align"):
        exec(code, ns)
