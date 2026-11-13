"""Segmentation-free abundance: marker transcripts per mm² of tissue counted from the
transcript file, the area-bin sweep, the segmented cell density beside it and the
concentration statistics against the redistribution null.

Fabricated transcript files with known contents; every number is checked against a
count made here by brute force.
"""
from __future__ import annotations

import os
import time

import dash
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import xe_testlib as tl


def _density_files(xe):
    out = set()
    for d in os.listdir(xe.CACHE_BASE_DIR):
        if d.startswith("density_"):
            out |= {os.path.join(xe.CACHE_BASE_DIR, d, f) for f in os.listdir(os.path.join(xe.CACHE_BASE_DIR, d))}
    return out


@pytest.fixture(autouse=True)
def _clean_segfree_state(xe):
    before = _density_files(xe)
    yield
    xe._test_baseline.settle()
    for f in _density_files(xe) - before:          # grids a test cached: other tests expect none
        os.remove(f)
    xe._sfree_reset()
    xe._stats_reset()
    with xe._density_mem_lock:
        xe._density_mem.clear()
    for f in os.listdir(xe.CACHE_BASE_DIR):
        if f.startswith("design_") and f.endswith(".json"):
            os.remove(os.path.join(xe.CACHE_BASE_DIR, f))


def _write_tx(dirpath, x, y, gene, qv=None, is_gene=None):
    """A transcripts.parquet with the columns the app reads."""
    os.makedirs(dirpath, exist_ok=True)
    n = len(x)
    pd.DataFrame({"x_location": np.asarray(x, dtype=np.float32), "y_location": np.asarray(y, dtype=np.float32),
                  "feature_name": np.asarray(gene, dtype=object),
                  "qv": np.full(n, 40.0, dtype=np.float32) if qv is None else np.asarray(qv, dtype=np.float32),
                  "is_gene": np.ones(n, dtype=bool) if is_gene is None else np.asarray(is_gene, dtype=bool)}
                 ).to_parquet(os.path.join(dirpath, "transcripts.parquet"), row_group_size=5000)
    return str(dirpath)


def _fab_tx(rng, n, size=1000.0, genes=("MK1", "MK2", "OTHER"), p=(0.2, 0.1, 0.7)):
    x = rng.uniform(0, size, n).astype(np.float32).astype(np.float64)      # as stored in the file
    y = rng.uniform(0, size, n).astype(np.float32).astype(np.float64)
    g = rng.choice(list(genes), size=n, p=list(p))
    qv = rng.choice([10.0, 30.0, 40.0], size=n, p=[0.1, 0.4, 0.5])
    return x, y, g, qv


def _hand_area_mm2(xy, b):
    """Occupied-tile area, the reference formula restated: distinct (x // b, y // b) × b²."""
    return len(set(zip((xy[:, 0] // b).astype(int), (xy[:, 1] // b).astype(int)))) * (b / 1000.0) ** 2


def _hand_tx_on_tiles(tx, cells_xy, b, ok):
    """Transcripts (rows of *tx* passing *ok*) lying in a tile that holds a cell."""
    tiles = set(zip((cells_xy[:, 0] // b).astype(int), (cells_xy[:, 1] // b).astype(int)))
    x, y = tx[0][ok], tx[1][ok]
    return sum((int(a // b), int(c // b)) in tiles for a, c in zip(x, y))


# ── 1. small statistics ──────────────────────────────────────────────────────
def test_gini_and_top_share_match_hand_values(xe):
    assert xe._sfree_gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert xe._sfree_gini([0, 0, 0, 8]) == pytest.approx(0.75)           # (n − 1) / n
    v = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    brute = np.abs(v[:, None] - v[None, :]).sum() / (2 * len(v) ** 2 * v.mean())
    assert xe._sfree_gini(v) == pytest.approx(brute)
    assert np.isnan(xe._sfree_gini([])) and np.isnan(xe._sfree_gini([0, 0]))
    assert xe._sfree_top_share(np.arange(1, 21), 0.10) == pytest.approx((19 + 20) / 210)
    assert xe._sfree_top_share([5, 0, 0], 0.10) == pytest.approx(1.0)


def test_concentration_null_is_zero_for_a_uniform_field_and_positive_when_concentrated(xe):
    rng = np.random.default_rng(3)
    uni = rng.multinomial(3000, np.full(400, 1 / 400))
    r = xe._sfree_concentration(uni, n_null=100, seed=1)
    # a sparse uniform count has a raw Gini well above 0; only the excess is ≈ 0
    assert r["gini"] > 0.15 and abs(r["gini_excess"]) < 0.02
    assert r["gini_null_lo"] <= r["gini"] <= r["gini_null_hi"] or r["gini_p"] > 0.02
    assert abs(r["top_share_excess"]) < 0.02 and r["n_tiles"] == 400
    p = np.full(400, 0.5 / 380)
    p[:20] = 0.5 / 20                               # half of the transcripts in 5 % of the tiles
    conc = rng.multinomial(3000, p)
    c = xe._sfree_concentration(conc, n_null=100, seed=1)
    assert c["gini_excess"] > 0.25 and c["gini_p"] == pytest.approx(1 / 101)
    assert c["top_share_excess"] > 0.25
    # same seed → same null; too few tiles or no transcripts → no statistic
    assert xe._sfree_concentration(conc, n_null=100, seed=1)["gini_null"] == c["gini_null"]
    assert np.isnan(xe._sfree_concentration(conc[:5], n_null=10)["gini"])
    assert np.isnan(xe._sfree_concentration(np.zeros(50, int), n_null=10)["gini"])
    assert np.isnan(xe._sfree_concentration(conc, n_null=0)["gini_excess"])


def test_bins_are_parsed_and_snapped_to_the_base_grid(xe):
    assert xe._sfree_parse_bins("") == [25.0, 50.0, 100.0, 200.0]
    assert xe._sfree_parse_bins("200; 52 50") == [50.0, 200.0]
    for bad in ("abc", "1", "25, 9000", ",".join(str(10 * i) for i in range(1, 12))):
        with pytest.raises(ValueError):
            xe._sfree_parse_bins(bad)


# ── 2. tissue mask ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("b", [25.0, 50.0, 100.0, 200.0])
def test_tissue_area_equals_occupied_tiles(xe, b):
    rng = np.random.default_rng(5)
    xy = np.column_stack([rng.uniform(-130, 900, 700), rng.uniform(40, 1300, 700)])
    m = xe._sfree_tissue_mask(xy, b, 0)
    assert m["mask"].sum() * (b / 1000) ** 2 == pytest.approx(_hand_area_mm2(xy, b))
    assert xe._sfree_cells_in_mask(xy, m).all()
    assert not xe._sfree_cells_in_mask(np.array([[5000.0, 5000.0], [-900.0, 0.0]]), m).any()


def test_holes_are_filled_by_the_compartment_rule(xe):
    # a 7 × 7 block of tiles with a 1-tile and a 4-tile interior hole and a notch open to the outside
    occ = np.ones((7, 7), dtype=bool)
    occ[1, 1] = False
    occ[3:5, 3:5] = False
    occ[6, 2] = False
    iy, ix = np.nonzero(occ)
    xy = np.column_stack([ix * 50.0 + 25, iy * 50.0 + 25])
    for hole, n in ((0, 43), (1, 44), (3, 44), (4, 48), (None, 48)):
        m = xe._sfree_tissue_mask(xy, 50.0, hole)
        assert m["mask"].sum() == n and m["n_occupied"] == 43
    # the same tissue through controls._edge_distance (the rule this mask copies)
    _d, info = xe._edge_distance(xy, 50.0, 0, 4)
    assert info["bins_after_fill"] == 48 and info["holes_filled"] == 2


# ── 3. the transcript scan ───────────────────────────────────────────────────
def test_multi_gene_scan_equals_the_single_gene_scan(xe, tmp_path):
    rng = np.random.default_rng(11)
    x, y, g, qv = _fab_tx(rng, 20000)
    d = _write_tx(tmp_path / "s", x, y, g, qv)
    multi = xe._sfree_scan_genes(os.path.join(d, "transcripts.parquet"), ["MK1", "MK2", "ABSENT"])
    for gene in ("MK1", "MK2"):
        one = xe._density_scan(d, gene)
        ok = (g == gene) & (qv >= 20)
        assert multi[gene]["total"] == one["total"] == int(ok.sum())
        a = pd.Series(multi[gene]["n"], index=pd.MultiIndex.from_arrays([multi[gene]["iy"], multi[gene]["ix"]]))
        b = pd.Series(one["n"], index=pd.MultiIndex.from_arrays([one["iy"], one["ix"]]))
        pd.testing.assert_series_equal(a.sort_index(), b.sort_index())
    assert multi["ABSENT"]["total"] == 0 and len(multi["ABSENT"]["n"]) == 0


def test_scan_of_the_bundle_matches_the_density_tab_scan(xe, D, expressed_gene):
    if not os.path.exists(os.path.join(D["data_dir"], "transcripts.parquet")):
        pytest.skip("bundle without transcripts.parquet")
    multi = xe._sfree_scan_genes(os.path.join(D["data_dir"], "transcripts.parquet"), [expressed_gene])
    one = xe._density_scan(D["data_dir"], expressed_gene)
    assert multi[expressed_gene]["total"] == one["total"] > 0
    assert int(multi[expressed_gene]["n"].sum()) == int(one["n"].sum())


# ── 4. one sample: density equals a hand count ───────────────────────────────
def test_density_equals_a_hand_count_for_every_tile_size(xe, tmp_path):
    rng = np.random.default_rng(21)
    tx = _fab_tx(rng, 30000)
    ds = {"data_dir": _write_tx(tmp_path / "hand", *tx)}
    cells = np.column_stack([rng.uniform(100, 800, 300), rng.uniform(0, 600, 300)])   # tissue: part of the field
    is_type = rng.random(300) < 0.2
    assert sorted(xe._sfree_ensure_grids(ds, ["MK1", "MK2"])) == ["MK1", "MK2"]
    assert xe._sfree_ensure_grids(ds, ["MK1", "MK2"]) == []                        # cached now
    bins = [25.0, 50.0, 100.0, 200.0]
    rows, per_gene = xe._sfree_sample(cells, is_type, ["MK1", "MK2"], ds, bins, n_null=20)
    ok = np.isin(tx[2], ["MK1", "MK2"]) & (tx[3] >= 20)
    assert per_gene == {"MK1": int(((tx[2] == "MK1") & (tx[3] >= 20)).sum()),
                        "MK2": int(((tx[2] == "MK2") & (tx[3] >= 20)).sum())}
    areas = []
    for r, b in zip(rows, bins):
        n_hand = _hand_tx_on_tiles(tx, cells, b, ok)
        area = _hand_area_mm2(cells, b)
        areas.append(area)
        assert r["n_tx"] == n_hand and r["area_mm2"] == pytest.approx(area)
        assert r["tx_per_mm2"] == pytest.approx(n_hand / area)
        assert r["n_tx_sample"] == int(ok.sum())
        assert r["n_cells_type"] == int(is_type.sum())
        assert r["cells_per_mm2"] == pytest.approx(is_type.sum() / area)
        assert r["tx_per_cell"] == pytest.approx(n_hand / is_type.sum())
        assert np.isnan(r["tx_per_10k"])                                              # no all-transcripts grid
        assert abs(r["gini_excess"]) < 0.05                                           # uniform transcripts
    assert areas == sorted(areas) and areas[-1] > areas[0], "the area grows with the tile size"
    assert rows[0]["tx_per_mm2"] > rows[-1]["tx_per_mm2"] * 0.8


def test_area_from_all_transcripts_and_share_of_all_transcripts(xe, tmp_path):
    rng = np.random.default_rng(22)
    tx = _fab_tx(rng, 20000, size=600.0)
    ds = {"data_dir": _write_tx(tmp_path / "alltx", *tx)}
    cells = np.column_stack([rng.uniform(0, 300, 50), rng.uniform(0, 300, 50)])
    with pytest.raises(ValueError, match="all-transcripts grid is missing"):
        xe._sfree_ensure_grids(ds, ["MK1"])
        xe._sfree_sample(cells, None, ["MK1"], ds, [50.0], area_mode="transcripts", n_null=0)
    xe._sfree_ensure_grids(ds, ["MK1", xe._DENSITY_ALL])
    good = tx[3] >= 20
    r = xe._sfree_sample(cells, None, ["MK1"], ds, [50.0], area_mode="transcripts", n_null=0)[0][0]
    tiles = set(zip((tx[0][good] // 50).astype(int), (tx[1][good] // 50).astype(int)))
    n_mk = int(((tx[2] == "MK1") & good).sum())
    assert r["n_tiles"] == len(tiles) and r["n_tx"] == n_mk and r["frac_tx_on_tissue"] == pytest.approx(1.0)
    assert r["tx_per_10k"] == pytest.approx(1e4 * n_mk / good.sum())
    assert np.isnan(r["cells_per_mm2"])
    # with the cells as the area, the share of all transcripts is taken on the cells' tiles
    rc = xe._sfree_sample(cells, None, ["MK1"], ds, [50.0], n_null=0)[0][0]
    ctiles = set(zip((cells[:, 0] // 50).astype(int), (cells[:, 1] // 50).astype(int)))
    on = np.array([(int(a // 50), int(b // 50)) in ctiles for a, b in zip(tx[0], tx[1])])
    assert rc["tx_per_10k"] == pytest.approx(1e4 * ((tx[2] == "MK1") & good & on).sum() / (good & on).sum())


def test_a_planted_cluster_of_transcripts_is_concentrated(xe, tmp_path):
    rng = np.random.default_rng(23)
    x, y, g, qv = _fab_tx(rng, 20000, genes=("BG", "OTHER"), p=(0.3, 0.7))
    cx, cy = rng.normal(500, 30, 3000), rng.normal(500, 30, 3000)                     # MK1: one tight focus
    ds = {"data_dir": _write_tx(tmp_path / "focus", np.r_[x, cx], np.r_[y, cy], np.r_[g, ["MK1"] * 3000])}
    cells = np.column_stack([rng.uniform(0, 1000, 3000), rng.uniform(0, 1000, 3000)])
    xe._sfree_ensure_grids(ds, ["MK1", "BG"])
    foc = xe._sfree_sample(cells, None, ["MK1"], ds, [50.0, 100.0], n_null=50)[0]
    bg = xe._sfree_sample(cells, None, ["BG"], ds, [50.0, 100.0], n_null=50)[0]
    for r in foc:
        assert r["gini_excess"] > 0.5 and r["gini_p"] == pytest.approx(1 / 51) and r["top_share_excess"] > 0.5
    for r in bg:
        assert abs(r["gini_excess"]) < 0.03 and r["gini_p"] > 0.02


# ── 5. the job on the bundle ─────────────────────────────────────────────────
def _bundle_hand(D, gene, b):
    tx = pd.read_parquet(os.path.join(D["data_dir"], "transcripts.parquet"))
    nm = tx["feature_name"].map(lambda v: v.decode() if isinstance(v, bytes) else v).astype(str)
    ok = (nm == gene).to_numpy() & (tx["qv"].to_numpy() >= 20)
    cells = np.column_stack([D["df"]["x_centroid"].to_numpy(float), D["df"]["y_centroid"].to_numpy(float)])
    return _hand_tx_on_tiles((tx["x_location"].to_numpy(), tx["y_location"].to_numpy()), cells, b, ok), \
        _hand_area_mm2(cells, b), int(ok.sum())


def test_job_on_the_bundle_gives_the_hand_count_and_the_cell_density(xe, D, labels, expressed_gene):
    if not os.path.exists(os.path.join(D["data_dir"], "transcripts.parquet")):
        pytest.skip("bundle without transcripts.parquet")
    assert xe.run_segfree(1, "celltype:celltypist", "Alpha", expressed_gene, 5, "25, 100", "cells", 0, 20,
                          "xenium", None, "original") == (True, False, "Running…")
    msg = tl.wait_state(xe._sfree_state, xe._sfree_lock)
    res = xe._sfree_state["result"]
    t = res["table"]
    assert list(t["bin_um"]) == [25.0, 100.0] and res["markers"] == [expressed_gene]
    n_alpha = int((labels.reindex(D["df"].index.astype(str)) == "Alpha").sum())
    for _i, r in t.iterrows():
        n_hand, area, n_all = _bundle_hand(D, expressed_gene, r["bin_um"])
        assert r["n_tx"] == n_hand and r["n_tx_sample"] == n_all
        assert r["area_mm2"] == pytest.approx(area) and r["tx_per_mm2"] == pytest.approx(n_hand / area)
        assert r["n_cells_type"] == n_alpha and r["cells_per_mm2"] == pytest.approx(n_alpha / area)
    # the share the count matrix assigns to cells: Σ counts of the gene ÷ transcripts of the gene
    col = np.asarray(D["expr_csc"][:, D["gene_name_to_idx"][expressed_gene]].todense()).ravel()
    rows = np.asarray(D.get("df_to_expr") if D.get("df_to_expr") is not None else np.arange(len(D["df"])))
    assert t["n_tx"].iloc[1] > 0
    assert t["assigned_to_cells"].iloc[1] == pytest.approx(col[rows].sum() / t["n_tx"].iloc[1])
    assert "Single sample" in msg and "listed" in msg and "unit: the tiles of one sample" in msg
    # figure, table and CSV of the result
    fig, recs = xe.update_sfree_results(1, 0, None, None)
    assert len(fig.data) == 3 and len(recs) == 2 and recs[0]["sample"] == "xenium"
    csv = xe.download_segfree(1)["content"]
    assert "# markers (user): " + expressed_gene in csv and "per marker gene" in csv
    # the poll publishes once
    out = xe.poll_segfree(1, 0)
    assert out[1] is False and out[2] == xe._sfree_state["version"] and out[3] is True
    assert xe.poll_segfree(2, out[2])[2] is dash.no_update


def test_markers_are_derived_from_the_annotation_when_none_are_listed(xe, D, labels):
    if not os.path.exists(os.path.join(D["data_dir"], "transcripts.parquet")):
        pytest.skip("bundle without transcripts.parquet")
    xe.run_segfree(1, "celltype:celltypist", "Alpha", "", 3, "50", "cells", 0, 0, "xenium", None, "original")
    tl.wait_state(xe._sfree_state, xe._sfree_lock)
    res = xe._sfree_state["result"]
    samples = xe._an_samples("xenium", False)
    lab = xe._an_celltype_labels(samples, "celltypist", None)
    want, info = xe._markers_of(xe._type_profiles(samples, lab), "Alpha", n=3)
    assert res["markers"] == want and res["marker_info"]["rule"] == info["rule"] and len(want) >= 1
    assert np.isnan(res["table"]["gini_excess"]).all()          # 0 redistributions asked


def test_subset_restricts_the_tissue_to_its_cells(xe, D, expressed_gene):
    if not os.path.exists(os.path.join(D["data_dir"], "transcripts.parquet")):
        pytest.skip("bundle without transcripts.parquet")
    ids = list(D["df"].sort_values("x_centroid").index[: max(2, len(D["df"]) // 2)])
    xe.subset(cell_ids=ids)
    cells = xe.DATA["df"][["x_centroid", "y_centroid"]].to_numpy(dtype=float)
    assert len(cells) == len(ids)
    xe.run_segfree(1, None, None, expressed_gene, 5, "50", "cells", 0, 0, "xenium", None, "original")
    msg = tl.wait_state(xe._sfree_state, xe._sfree_lock)
    r = xe._sfree_state["result"]["table"].iloc[0]
    assert r["area_mm2"] == pytest.approx(_hand_area_mm2(cells, 50.0)) and "and subset" in msg
    tx = pd.read_parquet(os.path.join(D["data_dir"], "transcripts.parquet"))
    nm = tx["feature_name"].map(lambda v: v.decode() if isinstance(v, bytes) else v).astype(str)
    ok = (nm == expressed_gene).to_numpy() & (tx["qv"].to_numpy() >= 20)
    assert r["n_tx"] == _hand_tx_on_tiles((tx["x_location"].to_numpy(), tx["y_location"].to_numpy()), cells, 50.0, ok)
    assert r["n_tx"] <= r["n_tx_sample"] == int(ok.sum())


def test_guards_and_empty_states(xe, D, monkeypatch):
    for fn, args in ((xe.run_segfree, (0, "cluster", "1", "", 5, "", "cells", 0, 10, "xenium", None, "original")),
                     (xe.run_segfree, (None, "cluster", "1", "", 5, "", "cells", 0, 10, "xenium", None, "original")),
                     (xe.download_segfree, (0,)), (xe.download_segfree, (1,))):       # (1: no result yet)
        with pytest.raises(dash.exceptions.PreventUpdate):
            fn(*args)
    assert "Pick a cell type" in xe.run_segfree(1, "cluster", None, "  ", 5, "", "cells", 0, 10, "xenium", None, "original")[2]
    assert "not a tile size" in xe.run_segfree(1, "cluster", None, "G", 5, "x", "cells", 0, 10, "xenium", None, "original")[2]
    assert xe._sfree_state["status"] == "idle" and not [j for j in xe._jobs.values() if j.kind == "segfree"]
    fig, recs = xe.update_sfree_results(0, 0, None, None)
    assert recs == [] and "No segmentation-free abundance" in fig.layout.annotations[0].text
    # listed genes that are not on the panel; a sample loaded from a .zarr (no transcript file)
    xe.run_segfree(1, None, None, "NOT_A_GENE_1, NOT_A_GENE_2", 5, "", "cells", 0, 10, "xenium", None, "original")
    with pytest.raises(AssertionError, match="None of the listed marker genes is on the panel"):
        tl.wait_state(xe._sfree_state, xe._sfree_lock)
    xe._test_baseline.settle()
    monkeypatch.setitem(xe.DATA, "data_dir", None)
    monkeypatch.setitem(xe.DATA, "sdata_path", "/somewhere/sample.zarr")
    xe.run_segfree(2, None, None, D["gene_names"][0], 5, "", "cells", 0, 10, "xenium", None, "original")
    with pytest.raises(AssertionError, match="loaded from a .zarr, which has cells but no transcripts"):
        tl.wait_state(xe._sfree_state, xe._sfree_lock)
    assert xe._sfree_state["result"] is None


def test_layout_has_the_block_in_its_own_tab(xe):
    # (it sat in the Cell Type Abundance tab before the tabs were grouped: QC > Segmentation-free)
    ids = set()

    def walk(c, inside):
        cid = getattr(c, "id", None)
        inside = inside or getattr(c, "tab_id", None) == "tab-segfree"
        if cid and inside:
            ids.add(cid)
        ch = getattr(c, "children", None)
        for k in (ch if isinstance(ch, (list, tuple)) else [ch]):
            if hasattr(k, "to_plotly_json"):
                walk(k, inside)

    walk(xe.app.layout, False)
    assert {"sfree-labels", "sfree-type", "sfree-markers", "sfree-bins", "sfree-area", "sfree-run-btn",
            "sfree-status", "sfree-plot", "sfree-table", "sfree-download-btn", "sfree-download"} <= ids
    assert xe._JOB_KIND_NAMES["segfree"] == "segmentation-free abundance"


# ── 6. several samples ───────────────────────────────────────────────────────
def _extra(D, tmp_path, i, n_mk, with_tx=True):
    """A fabricated extra sample: 150 cells on a 600 µm field, *n_mk* marker transcripts
    (uniform) in its own transcript file."""
    rng = np.random.default_rng(100 + i)
    n = 150
    genes = list(D["gene_names"])
    mk = genes[0]
    df = pd.DataFrame({"x_centroid": rng.uniform(0, 600, n), "y_centroid": rng.uniform(0, 600, n)},
                      index=[f"e{i}_{k}" for k in range(n)])
    m = sp.csr_matrix((rng.poisson(1.0, size=(n, len(genes)))).astype(np.float32))
    ds = {"expr": m, "df_to_expr": np.arange(n), "gene_name_to_idx": D["gene_name_to_idx"], "gene_names": genes,
          "df": df, "x_offset": 5000.0 * (i + 1), "cluster_methods": []}
    if with_tx:
        ds["data_dir"] = _write_tx(tmp_path / f"extra{i}", rng.uniform(0, 600, n_mk), rng.uniform(0, 600, n_mk),
                                   [mk] * n_mk)
    else:
        ds["sdata_path"] = str(tmp_path / f"extra{i}.zarr")
    return ds


def test_samples_are_scored_separately_and_tested_at_the_sample_level(xe, D, tmp_path):
    if not os.path.exists(os.path.join(D["data_dir"], "transcripts.parquet")):
        pytest.skip("bundle without transcripts.parquet")
    n_mk = [900, 1000, 1100, 300, 350, 400]                       # A: three times the density of B
    extras = [_extra(D, tmp_path, i, k) for i, k in enumerate(n_mk)] + [_extra(D, tmp_path, 6, 0, with_tx=False)]
    xe.EXTRA_DATASETS.extend(extras)
    xe._design_set([{"key": xe._ds_key(e), "cond": "A" if i < 3 else "B", "batch": ""}
                    for i, e in enumerate(extras[:6])])
    mk = D["gene_names"][0]
    xe.run_segfree(1, None, None, mk, 5, "50, 200", "cells", 0, 10, "xenium", None, "original")
    msg = tl.wait_state(xe._sfree_state, xe._sfree_lock)
    res = xe._sfree_state["result"]
    t = res["table"]
    assert list(t["sample"].unique()) == [f"S{i}" for i in range(7)] and res["skipped"] == ["S7"]
    assert "Left out (loaded from a .zarr, no transcript file): S7" in msg
    for i, k in enumerate(n_mk):                                  # each sample's own file and own cells
        sub = t[t["sample"] == f"S{i + 1}"].set_index("bin_um")
        cells = extras[i]["df"][["x_centroid", "y_centroid"]].to_numpy()
        assert sub.loc[200.0, "n_tx"] == k                         # 200 µm tiles: the whole 600 µm field
        assert sub.loc[200.0, "area_mm2"] == pytest.approx(0.36)
        assert sub.loc[50.0, "area_mm2"] == pytest.approx(_hand_area_mm2(cells, 50.0))
        assert np.isnan(sub.loc[50.0, "cells_per_mm2"])            # no cell type was picked
    a = res["across"]
    d = a[(a["metric"] == "tx_per_mm2") & (a["bin_um"] == 200.0)].iloc[0]
    want = (np.mean(n_mk[:3]) - np.mean(n_mk[3:])) / 0.36
    assert d["effect"] == pytest.approx(want) and d["p"] == pytest.approx(2 / 20) and d["min_p"] == pytest.approx(0.1)
    assert d["n_units"] == 6 and bool(d["exact"])
    assert "unpaired label permutation test, n = 6 samples" in msg and "not cells or tiles" in msg
    assert "between conditions" in xe.download_segfree(1)["content"]
    # too few units: no test, and the reason
    xe._test_baseline.settle()
    xe._design_set([{"key": xe._ds_key(e), "cond": c, "batch": ""}
                    for e, c in zip(extras[:4], ["A", "A", "B", "B"])])
    xe.run_segfree(2, None, None, mk, 5, "200", "cells", 0, 0, "xenium", None, "original")
    msg = tl.wait_state(xe._sfree_state, xe._sfree_lock)
    assert len(xe._sfree_state["result"]["across"]) == 0 and "no test between conditions: fewer than 3" in msg


# ── 7. cancel ────────────────────────────────────────────────────────────────
def test_job_can_be_cancelled_publishes_nothing_and_reruns(xe, D, tmp_path, monkeypatch):
    rng = np.random.default_rng(31)
    mk = D["gene_names"][0]
    d = _write_tx(tmp_path / "cancel", rng.uniform(0, 500, 4000), rng.uniform(0, 500, 4000), [mk] * 4000)
    monkeypatch.setitem(xe.DATA, "data_dir", d)
    assert not os.path.exists(xe._density_cache_file(mk))
    real = xe._sfree_scan_genes
    seen = []

    def slow_scan(path, genes, qv_min=20, progress=None):
        for i in range(400):
            seen.append(i)
            time.sleep(0.02)
            xe._job_check()
        return real(path, genes, qv_min, progress)

    monkeypatch.setattr(xe, "_sfree_scan_genes", slow_scan)
    assert xe.run_segfree(1, None, None, mk, 5, "50", "cells", 0, 10, "xenium", None, "original")[:2] == (True, False)
    job = [j for j in xe._jobs.values() if j.kind == "segfree"][-1]
    while not seen:
        time.sleep(0.01)
    assert xe.run_segfree(2, None, None, mk, 5, "50", "cells", 0, 10, "xenium", None, "original")[2].endswith(
        "already being computed.")
    assert job.snapshot()["cancellable"] and xe._job_cancel(job.id)
    job.thread.join(10)
    assert job.state == "cancelled" and len(seen) < 400
    assert xe._sfree_state["status"] == "error" and xe._sfree_state["message"] == "Cancelled"
    assert xe._sfree_state["result"] is None
    assert not os.path.exists(xe._density_cache_file(mk))
    assert "segmentation-free abundance" not in xe._running_jobs()
    assert xe.poll_segfree(1, 0)[:2] == ("Cancelled", False)
    # a rerun completes and gives the hand count
    monkeypatch.setattr(xe, "_sfree_scan_genes", real)
    assert xe.run_segfree(3, None, None, mk, 5, "50", "cells", 0, 10, "xenium", None, "original")[:2] == (True, False)
    tl.wait_state(xe._sfree_state, xe._sfree_lock)
    r = xe._sfree_state["result"]["table"].iloc[0]
    assert r["n_tx_sample"] == 4000 and os.path.exists(xe._density_cache_file(mk))
