"""Contamination and confound controls: sample-level exact tests and the donor design,
tissue compartments, spillover score, gene ownership, the leak gate, segmentation QC
metrics, stratified co-localisation with matched nulls, and the guards (graphs within
samples, detection on fractional counts, depth, label collapse, the volcano colouring).

Everything runs on fabricated data with a planted effect, and every formula is checked
against an independent computation written here (brute force, itertools enumeration, or
the reference formula the method was ported from, restated on arrays).
"""
from __future__ import annotations

import ast
import itertools
import os
import time

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy import ndimage
from scipy.spatial.distance import cdist

import xe_testlib as tl


@pytest.fixture(autouse=True)
def _clean_controls_state(xe):
    yield
    xe._controls_reset()
    xe._stats_reset()
    for f in os.listdir(xe.CACHE_BASE_DIR):
        if f.startswith("design_") and f.endswith(".json"):
            os.remove(os.path.join(xe.CACHE_BASE_DIR, f))


def _hand_sample(counts, xy, genes=None, label="hand"):
    n, g = counts.shape
    genes = genes or [f"g{i}" for i in range(g)]
    df = pd.DataFrame({"x_centroid": xy[:, 0], "y_centroid": xy[:, 1]}, index=[f"{label}_{i}" for i in range(n)])
    return {"label": label, "name": label, "key": label, "df": df, "csc": sp.csc_matrix(counts.astype(np.float32)),
            "rows": np.arange(n), "gni": {x: i for i, x in enumerate(genes)}, "genes": genes,
            "x_offset": 0.0, "corrected": False, "ds": {}}


# ── 1. exact sample-level tests ──────────────────────────────────────────────
@pytest.mark.parametrize("n", [1, 3, 6, 7, 10])
def test_signflip_equals_hand_enumeration(xe, n):
    rng = np.random.default_rng(n)
    d = rng.normal(0.4, 1.0, size=n)
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    means = (signs * d).mean(axis=1)
    r = xe._sl_signflip(d)
    assert r["exact"] and r["n_assignments"] == 2 ** n and r["n"] == n
    assert r["p"] == pytest.approx(np.mean(np.abs(means) >= abs(d.mean()) - 1e-12), abs=0)
    assert r["p_greater"] == pytest.approx(np.mean(means >= d.mean() - 1e-12), abs=0)
    assert r["p_less"] == pytest.approx(np.mean(means <= d.mean() + 1e-12), abs=0)
    assert r["min_p"] == pytest.approx(2 / 2 ** n if n > 1 else 1.0)
    # all in one direction: the floor itself
    assert xe._sl_signflip(np.abs(d) + 0.1)["p"] == pytest.approx(2 / 2 ** n if n > 1 else 1.0)


def test_signflip_never_samples_a_small_null_and_is_reproducible_above_the_limit(xe):
    d = np.array([0.9, 1.1, 0.7, 1.3, 0.8, 1.2, 1.0])
    assert xe._sl_signflip(d)["p"] == 2 / 128           # 7 units: 128 vectors, enumerated
    assert xe._SL_ENUM_UNITS >= 16
    big = np.random.default_rng(0).normal(0.2, 1, size=xe._SL_ENUM_UNITS + 4)
    a, b = xe._sl_signflip(big), xe._sl_signflip(big)
    assert not a["exact"] and a["p"] == b["p"] and a["n_assignments"] == xe._SL_MC >= 10_000
    assert a["min_p"] == pytest.approx(1 / (xe._SL_MC + 1))
    exact = xe._sl_signflip(big, enum_max=64)
    assert exact["exact"] and abs(exact["p"] - a["p"]) < 0.02
    # a zero difference raises the floor (it cannot change sign)
    assert xe._sl_signflip(np.array([1.0, 2.0, 0.0]))["min_p"] == pytest.approx(0.5)


@pytest.mark.parametrize("na,nb", [(3, 3), (4, 3), (2, 5)])
def test_label_permutation_equals_hand_enumeration(xe, na, nb):
    rng = np.random.default_rng(na * 10 + nb)
    a, b = rng.normal(1, 1, na), rng.normal(0, 1, nb)
    v = np.concatenate([a, b])
    stats = []
    for c in itertools.combinations(range(na + nb), na):
        m = np.zeros(na + nb, bool)
        m[list(c)] = True
        stats.append(v[m].mean() - v[~m].mean())
    stats = np.array(stats)
    obs = a.mean() - b.mean()
    r = xe._sl_permutation(a, b)
    assert r["exact"] and r["n_assignments"] == len(stats)
    assert r["p"] == pytest.approx(np.mean(np.abs(stats) >= abs(obs) - 1e-12), abs=0)
    assert r["p_greater"] == pytest.approx(np.mean(stats >= obs - 1e-12), abs=0)
    assert r["min_p"] == pytest.approx((2 if na == nb else 1) / len(stats))
    mc = xe._sl_permutation(a, b, enum_max=1)
    assert not mc["exact"] and mc["p"] == xe._sl_permutation(a, b, enum_max=1)["p"]
    assert abs(mc["p"] - r["p"]) < 0.03


def test_samples_of_a_donor_are_averaged_before_the_test(xe):
    #          d1A  d1A  d1B  d2A  d2B  d3A  d3B  d3B
    vals = np.array([3.0, 5.0, 1.0, 2.0, 1.5, 4.0, 1.0, 3.0])
    conds = ["A", "A", "B", "A", "B", "A", "B", "B"]
    donors = ["d1", "d1", "d1", "d2", "d2", "d3", "d3", "d3"]
    u = xe._sl_units(conds, donors)
    assert u["paired"] and u["n_units"] == 3 and u["n_samples"] == 8
    assert u["note"] == "n = 8 samples (3 donors, paired)"
    r = xe._sl_test(vals, conds, donors)
    d = np.array([4.0 - 1.0, 2.0 - 1.5, 4.0 - 2.0])
    assert np.allclose(r["per_unit"], d) and r["paired"] and r["n"] == 3
    assert r["p"] == xe._sl_signflip(d)["p"] == 0.25          # 3 pairs: the floor, never "p = 0"
    # treating the 8 samples as independent would claim far more
    assert xe._sl_permutation(vals[[0, 1, 3, 5]], vals[[2, 4, 6, 7]])["min_p"] < 0.03
    # a donor in one condition only is left out of the paired test and named
    u2 = xe._sl_units(conds + ["A"], donors + ["d9"])
    assert u2["paired"] and u2["dropped"] == ["d9"] and u2["n_units"] == 3
    # no donors: every sample is a unit; some donors only: not paired, and said
    u3 = xe._sl_units(conds, None)
    assert not u3["paired"] and u3["n_units"] == 8 and u3["note"] == "n = 8 samples"
    u4 = xe._sl_units(["A", "A", "B", "B"], ["d1", "d1", "", ""])
    assert not u4["paired"] and u4["n_units"] == 3 and "3 independent units" in u4["note"]


def test_bh_reach_warning(xe):
    assert xe._sl_reach(2 / 128, 3) is None                     # 0.047 < 0.05
    w = xe._sl_reach(2 / 128, 390)
    assert "at least 122 tests" in w                            # ceil(390 / 128 * 2 / 0.05)
    assert "no test can be significant" in xe._sl_reach(0.25, 1)
    assert xe._sl_reach(1 / 201, 25) is not None and xe._sl_reach(1 / 2001, 25) is None


# ── design: donor field, persistence, use in the tests ───────────────────────
def _fab_ds(D, n, seed, key, types=("Alpha", "Beta", "Gamma"), p=None, n_genes=60):
    rng = np.random.default_rng(seed)
    genes = list(D["gene_names"])
    dense = rng.poisson(2.0, size=(n, n_genes))
    m = sp.lil_matrix((n, len(genes)), dtype=np.float32)
    m[:, :n_genes] = dense
    df = pd.DataFrame({"x_centroid": rng.uniform(0, 300, n), "y_centroid": rng.uniform(0, 300, n),
                       "transcript_counts": dense.sum(axis=1)}, index=[f"k{seed}_{i}" for i in range(n)])
    return {"expr": m.tocsr(), "df_to_expr": np.arange(n), "gene_name_to_idx": D["gene_name_to_idx"],
            "gene_names": genes, "df": df, "x_offset": 2000.0 * (seed + 1), "cluster_methods": [],
            "data_dir": key, "_cell_type_labels": pd.Series(rng.choice(types, size=n, p=p), index=df.index)}


def test_design_has_a_donor_and_is_restored_for_the_same_samples(xe, D, tmp_path):
    extras = [_fab_ds(D, 60, i, str(tmp_path / f"smp{i}")) for i in range(4)]
    xe.EXTRA_DATASETS.extend(extras)
    rows = [{"key": xe._ds_key(e), "cond": "AB"[i % 2], "batch": "", "donor": f"d{i // 2}"}
            for i, e in enumerate(extras)]
    xe._design_set(rows, {"A": "post", "B": "pre"})
    path = xe._design_path()
    assert path and os.path.isfile(path) and os.path.dirname(path) == xe.CACHE_BASE_DIR
    got = xe._design_samples()
    assert [r["donor"] for r in got] == ["", "d0", "d0", "d1", "d1"]
    assert "donor: S1=d0" in xe._design_summary()
    # Load Sample clears the in-memory design; the same samples bring it back
    xe._stats_reset()
    assert xe._design_state["assign"] == {}
    got = xe._design_samples()
    assert [r["cond"] for r in got] == ["", "A", "B", "A", "B"] and got[1]["donor"] == "d0"
    assert xe._design_names() == {"A": "post", "B": "pre"}
    assert xe._design_donors([{"key": r["key"]} for r in got]) == ["", "d0", "d0", "d1", "d1"]
    # another set of samples: its own file, nothing restored into it
    xe.EXTRA_DATASETS.pop()
    assert xe._design_path() != path
    # a row without "donor" keeps the donor (older callers)
    xe._design_set([{"key": rows[0]["key"], "cond": "B", "batch": "x"}])
    assert xe._design_samples()[1]["donor"] == "d0" and xe._design_samples()[1]["cond"] == "B"
    # the editor has the donor column and Apply stores it
    ids = [c.id for row in xe._design_rows_children()[1:] for c in row.children if hasattr(c, "id")]
    assert {"type": "design-donor", "index": 1} in ids
    n = len(xe._design_samples())
    _v, msg = xe.apply_design(1, ["A", "B", "A", "B"][:n], [""] * n, [{"type": "design-cond", "index": i} for i in range(n)],
                              "A", "B", 0, ["p1", "p1", "p2", "p2"][:n])
    assert "saved in the cache directory" in msg and "donors" in msg


def test_composition_exact_test_is_the_sample_level_test(xe, D, tmp_path):
    types = ("Alpha", "Beta", "Gamma")
    extras = []
    for i in range(8):
        p = (0.6, 0.2, 0.2) if i % 2 == 0 else (0.2, 0.6, 0.2)       # A: more Alpha, B: more Beta
        extras.append(_fab_ds(D, 400, i, str(tmp_path / f"c{i}"), types, p))
    xe.EXTRA_DATASETS.extend(extras)
    rows = [{"key": xe._ds_key(e), "cond": "A" if i % 2 == 0 else "B", "batch": "", "donor": f"d{i // 2}"}
            for i, e in enumerate(extras)]
    xe._design_set(rows)
    res = xe._comp_compute("celltype:celltypist", "xenium", None, "logit", "exact")
    t = res["tests"].set_index("category")
    assert res["fdr_col"] == "fdr_exact" and "paired sign-flip" in res["test_note"]
    assert "n = 8 samples (4 donors, paired)" in res["test_note"]
    assert "Smallest achievable p: 0.125" in res["test_note"]
    assert t.loc["Alpha", "p_exact"] == t.loc["Beta", "p_exact"] == 0.125      # 4 pairs: 2 / 16
    assert t.loc["Alpha", "effect"] > 0 > t.loc["Beta", "effect"]
    assert "no test can be significant" in res["test_note"]
    # the value tested is the per-donor difference of the logit proportions
    y = xe._comp_transform(res["counts"][[f"S{i + 1}" for i in range(8)]].loc[list(t.index)].to_numpy())
    d = y[:, 0::2] - y[:, 1::2]
    assert t.loc["Alpha", "effect"] == pytest.approx(d[list(t.index).index("Alpha")].mean())
    cols, recs = xe._comp_table(res)
    assert "p_exact" in [c["id"] for c in cols] and "p_mod" not in [c["id"] for c in cols]
    assert "donor d0" in xe._comp_csv(res)
    # without donors: label permutation over the 8 samples, C(8, 4) = 70 relabellings
    xe._design_set([{**r, "donor": ""} for r in rows])
    res = xe._comp_compute("celltype:celltypist", "xenium", None, "logit", "exact")
    assert "unpaired label permutation" in res["test_note"] and "all 70 assignments" in res["test_note"]
    assert res["tests"].set_index("category").loc["Alpha", "p_exact"] == pytest.approx(2 / 70)
    # the moderated t-test says what its unit is, and that it ignores donors
    xe._design_set(rows)
    res = xe._comp_compute("celltype:celltypist", "xenium", None, "logit", "moderated")
    assert "Unit of the p-values: n = 8 samples" in res["test_note"] and "does not use them" in res["test_note"]


def test_pseudobulk_blocks_on_the_donor_and_sums_its_samples(xe, D, tmp_path):
    extras = [_fab_ds(D, 80, i, str(tmp_path / f"p{i}")) for i in range(7)]
    xe.EXTRA_DATASETS.extend(extras)
    conds = ["A", "B", "A", "B", "A", "B", "B"]
    donors = ["d1", "d1", "d2", "d2", "d3", "d3", "d3"]
    xe._design_set([{"key": xe._ds_key(e), "cond": c, "batch": "run1", "donor": d}
                    for e, c, d in zip(extras, conds, donors)])
    samples = xe._an_samples("xenium", False)
    masks = [np.ones(len(s["df"]), bool) for s in samples]
    cd, bt = xe._design_of(samples)
    prep = xe._pb_prepare(samples, masks, cd, bt, 1, 0, donors=xe._design_donors(samples))
    assert prep["design"] == "~ donor + condition" and prep["counts"].shape[1] == 6
    assert prep["labels"][-1] == "S6+S7" and prep["batch"] == ["d1", "d1", "d2", "d2", "d3", "d3"]
    assert prep["col_of_sample"]["S6"] == prep["col_of_sample"]["S7"] == 5
    plain = xe._pb_prepare(samples, masks, cd, ["" for _ in bt], 1, 0)
    assert plain["design"] == "~ condition" and plain["counts"].shape[1] == 7
    assert np.array_equal(prep["counts"][:, 5], plain["counts"][:, 5] + plain["counts"][:, 6])
    assert any("summed into one pseudobulk column" in w for w in prep["warnings"])
    assert any("batch labels are not used" in w for w in prep["warnings"])
    assert prep["units_note"] == "n = 7 samples (3 donors, paired)"
    with pytest.raises(ValueError, match="some samples only"):
        xe._pb_prepare(samples, masks, cd, bt, 1, 0, donors=["", "d1", "d1", "d2", "d2", "d3", "d3", ""])


# ── 2. compartments ──────────────────────────────────────────────────────────
def _reference_edge_distance(xy, BIN, HOLE_MAX):
    """The raster / hole-fill / distance-transform block of the analysis the method was
    ported from, restated."""
    x0, y0 = xy[:, 0].min(), xy[:, 1].min()
    bx = ((xy[:, 0] - x0) / BIN).astype(int); by = ((xy[:, 1] - y0) / BIN).astype(int)
    grid = np.zeros((bx.max() + 3, by.max() + 3), bool); grid[bx + 1, by + 1] = True
    d_std = ndimage.distance_transform_edt(grid)[bx + 1, by + 1] * BIN
    lab, n = ndimage.label(~grid); sizes = np.bincount(lab.ravel()); outside = lab[0, 0]
    small = [i for i in range(1, n + 1) if i != outside and sizes[i] <= HOLE_MAX]
    filled = grid | np.isin(lab, small)
    return ndimage.distance_transform_edt(filled)[bx + 1, by + 1] * BIN, d_std


def _section(seed=0, n=4000, size=1000.0):
    """A square section with one small hole (filled) and one large hole (left open)."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, size, size=(n, 2))
    small = (np.abs(xy[:, 0] - 300) < 52) & (np.abs(xy[:, 1] - 300) < 52)        # 4 bins of 50 µm
    large = (np.abs(xy[:, 0] - 700) < 160) & (np.abs(xy[:, 1] - 650) < 160)
    return xy[~small & ~large]


def test_edge_distance_equals_the_reference_computation(xe):
    xy = _section()
    for hole in (4, 0):
        d, info = xe._edge_distance(xy, 50.0, 0, hole)
        ref, unfilled = _reference_edge_distance(xy, 50.0, hole)
        assert np.array_equal(d, ref)
        if hole == 0:
            assert np.array_equal(d, unfilled) and info["holes_filled"] == 0
    d4, info = xe._edge_distance(xy, 50.0, 0, 4)
    d0, _ = xe._edge_distance(xy, 50.0, 0, 0)
    assert info["holes_filled"] >= 1 and info["holes_left"] >= 1
    near_small = (np.abs(xy[:, 0] - 300) < 120) & (np.abs(xy[:, 1] - 300) < 120)
    assert d4[near_small].min() > d0[near_small].min()          # the small hole no longer is an edge
    near_large = (np.abs(xy[:, 0] - 700) < 215) & (np.abs(xy[:, 1] - 650) < 215)
    assert d4[near_large].min() == 50.0                          # the large one still is
    centre = (np.abs(xy[:, 0] - 150) < 30) & (np.abs(xy[:, 1] - 800) < 30)
    assert 100 <= np.median(d4[centre]) <= 200                   # ~150 µm from the left border
    # None fills every interior hole; closing bridges a gap of one bin
    dn, infon = xe._edge_distance(xy, 50.0, 0, None)
    assert infon["holes_left"] == 0 and dn[near_large].min() > 50.0
    two = np.vstack([np.column_stack([np.full(50, 25.0), np.linspace(0, 400, 50)]),
                     np.column_stack([np.full(50, 125.0), np.linspace(0, 400, 50)])])
    assert xe._edge_distance(two, 50.0, 1, 0)[1]["bins_after_fill"] > xe._edge_distance(two, 50.0, 0, 0)[1]["bins_after_fill"]


def test_local_density_and_dense_core_equal_brute_force(xe):
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 600, size=(900, 2))
    dom = xy[:, 0] < 300
    dom ^= rng.random(900) < 0.05
    D_ = cdist(xy, xy)
    n_nb = (D_ <= 60).sum(1) - 1
    n_dom = ((D_ <= 60) & dom[None, :]).sum(1) - dom
    assert np.array_equal(xe._local_counts(xy, 60), n_nb)
    assert np.array_equal(xe._local_counts(xy, 60, dom), n_dom)
    cols, info = xe._compartments(xy, dom, bin_um=50, radius=60, min_nb=8, min_frac=0.5, min_edge=100)
    d, _ = xe._edge_distance(xy, 50)
    core = (n_nb >= 8) & (n_dom / np.maximum(n_nb, 1) >= 0.5) & (d > 100)
    assert np.array_equal(cols["core"].astype(bool), core) and 0 < core.mean() < 0.5
    assert set(cols["compartment"]) == {"core", "edge", "other"}
    assert np.array_equal(cols["compartment"] == "edge", ~core & (d <= 100))
    assert not core[xy[:, 0] > 400].any() and info["frac_core"] == pytest.approx(core.mean())
    cols2, _ = xe._compartments(xy, None, min_edge=100)
    assert "core" not in cols2 and set(cols2["compartment"]) == {"edge", "other"}


def test_compartment_job_stores_columns_for_color_by_both_renderers_and_export(xe, D, labels, client):
    xe._run_compartments("xenium", None, "celltype:celltypist", "auto",
                         {"bin_um": 20, "radius": 30, "min_nb": 2, "min_frac": 0.2, "min_edge": 20})
    st = xe._comp_ctl_state
    assert st["status"] == "done", st["message"]
    assert "median edge distance" in st["message"] and "Color By" in st["message"]
    t = xe._view_tables("xenium")[0]
    edge = xe._ctl_get(t, "xenium", "edge_distance_um")
    ref, _ = xe._edge_distance(xe._xy_of(D["df"]), 20, 0, 4)
    assert np.allclose(edge, ref) and len(edge) == len(D["df"])
    opts = [o["value"] for o in xe._refresh_colorby_options([{"label": "Cluster", "value": "cluster"}])]
    assert {"geneset:@edge_distance", "geneset:@density", "geneset:@core"} <= set(opts)
    # Plotly figure: one value per drawn cell, on a sequential scale
    vals, lim, label = xe._geneset_color("geneset:@edge_distance", "xenium", False)
    assert np.allclose(vals, ref) and label == "Edge distance (µm)" and xe._view_color_seq[label][0] == 0.0
    fig = xe.make_spatial_fig("geneset:@edge_distance", D["cluster_methods"][0] if D["cluster_methods"] else "",
                              None, 3, 0.8, [], None)
    tr = [t_ for t_ in fig.data if getattr(t_.marker, "colorscale", None) is not None][0]
    assert len(tr.marker.color) == len(D["df"]) and tr.marker.cmin == 0.0
    assert np.allclose(np.sort(np.asarray(tr.marker.color, float)), np.sort(ref), atol=1e-3)
    # WebGL endpoint: the same values, sequential colormap
    meta, arr = xe._viewer_provider.colors({"source": "xenium", "mode": "geneset:@edge_distance"})
    assert meta["kind"] == "continuous" and meta["colormap"] == "viridis" and meta["min"] == 0.0
    assert np.allclose(arr["values"], ref, atol=1e-3)
    # a column that was never computed falls back like an unknown gene set
    assert xe._geneset_color("geneset:@spillover", "xenium", False) is None
    # subset: the stored values follow the cells
    keep = D["df"].index[: len(D["df"]) // 2]
    sub = {"key": t["key"], "df": D["df"].loc[keep]}
    assert np.allclose(xe._ctl_get(sub, "xenium", "edge_distance_um"), ref[: len(keep)])
    # Save as SpatialData hook: numeric and text columns, named ctl_*
    df2, names = xe._with_extra_obs(D["df"], "xenium")
    assert {"ctl_edge_distance_um", "ctl_local_density", "ctl_core", "ctl_compartment"} <= set(names)
    assert df2 is not D["df"] and "ctl_core" not in D["df"].columns
    assert df2["ctl_compartment"].dtype == object and np.allclose(df2["ctl_edge_distance_um"], ref)
    # another dataset: nothing is kept
    xe._controls_reset()
    assert xe._ctl_available("xenium") == [] and xe._ctl_color_options() == []


# ── 3. spillover, ownership, leak gate ───────────────────────────────────────
G_T, G_L, G_M, G_X, G_BG = range(0, 10), range(10, 40), range(40, 70), range(70, 76), range(76, 90)


def _bleed_world(seed=0, n=900, planted=True, leak_a=0.20, leak_b=0.02):
    """Types T (focal), L and M. L cells fill the left third, T and M are everywhere. A T
    cell picks up L transcripts in proportion to exp(−distance to the nearest L cell / 25)
    (planted=False: independent of the distance). T cells are in group A (y high: strong
    bleed, and G_X truly 3× up) or B."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 600, size=(n, 2))
    typ = np.where(xy[:, 0] < 200, rng.choice(["L", "T", "M"], n, p=[0.6, 0.25, 0.15]),
                   rng.choice(["T", "M"], n, p=[0.7, 0.3]))
    prof = np.zeros((3, 90))
    prof[0, list(G_T)] = 8.0; prof[0, list(G_X)] = 4.0
    prof[1, list(G_L)] = 10.0
    prof[2, list(G_M)] = 9.0
    prof[:, list(G_BG)] = 3.0
    idx = {"T": 0, "L": 1, "M": 2}
    lam = prof[[idx[t] for t in typ]].copy()
    isT, isL = typ == "T", typ == "L"
    grpA = isT & (xy[:, 1] >= 300)
    grpB = isT & (xy[:, 1] < 300)
    lam[np.ix_(grpA, list(G_X))] *= 3.0
    d = cdist(xy, xy[isL]).min(axis=1)
    w = np.exp(-d / 25.0) if planted else np.full(n, 0.15)
    leak = np.where(grpA, leak_a, leak_b) / 0.15 * 0.15 if not planted else np.where(grpA, leak_a, leak_b) * 5 * w
    lam[isT] += (leak[isT, None] if not planted else leak[isT, None]) * prof[1][None, :] * (1.0 if planted else w[isT, None] / 0.15)
    lam[isT] += 0.05 * prof[2][None, :]
    counts = rng.poisson(lam)
    return _hand_sample(counts, xy, label=f"w{seed}"), typ.astype(object), grpA, grpB


def test_markers_and_gene_ownership_from_the_annotation(xe):
    s, typ, _a, _b = _bleed_world()
    prof = xe._type_profiles([s], [typ])
    assert prof["types"] == ["L", "M", "T"]
    dense = s["csc"].toarray()
    for i, t in enumerate(prof["types"]):
        assert np.allclose(prof["counts"][i], dense[typ == t].sum(0))
        assert prof["depth"][i] == pytest.approx(dense[typ == t].sum()) and prof["n"][i] == (typ == t).sum()
    mk, info = xe._markers_of(prof, "L", n=10)
    assert info["rule"] == "specific" and set(mk) <= {f"g{i}" for i in G_L} and 3 <= len(mk) <= 10
    # with the focal type excluded from the comparison, as the leak gate asks
    mk2, info2 = xe._markers_of(prof, "L", n=25, exclude=("T",), min_strict=10)
    assert info2["rule"] == "specific" and len(mk2) == 25
    # nothing ≥ 10× specific: the top genes by ratio, labelled as a fallback
    shared = {**prof, "counts": prof["counts"] + 1e5}         # every type has every gene
    mk3, info3 = xe._markers_of(shared, "T", n=5)
    assert info3["rule"] == "fallback" and len(mk3) == 5
    own = xe._gene_ownership(prof).set_index("gene")
    assert (own.loc[[f"g{i}" for i in G_L], "owner"] == "L").all()
    assert (own.loc[[f"g{i}" for i in G_M], "owner"] == "M").all()
    assert (own.loc[[f"g{i}" for i in G_T], "owner"] == "T").all()
    assert (own.loc[[f"g{i}" for i in G_BG], "n_owners"] == 3).all()            # shared genes
    # L genes are seen in T at well under 20 % of L's rate: T is not an owner
    assert not own.loc[[f"g{i}" for i in G_L], "owners"].str.contains("T").any()
    rate = prof["counts"] / prof["depth"][:, None]
    j = prof["genes"].index("g80")
    assert set(own.loc["g80", "owners"].split(";")) == {t for i, t in enumerate(prof["types"])
                                                       if rate[i, j] >= 0.2 * rate[:, j].max()}
    assert (xe._gene_ownership(prof, min_count=1e9)["owner"] == "").all()


def test_spillover_score_recovers_planted_bleed_and_its_validity_check_passes(xe):
    s, typ, _a, _b = _bleed_world()
    markers = [f"g{i}" for i in list(G_L)[:6]]
    res = xe._spillover([s], [typ], "L", markers)
    p = res["per_sample"][0]
    dense = s["csc"].toarray()
    share = dense[:, list(G_L)[:6]].sum(1) / np.maximum(dense.sum(1), 1)
    assert np.allclose(p["share"], share)
    assert res["reference"] == pytest.approx(np.median(share[typ == "L"]))
    assert np.allclose(p["score"], share / res["reference"])
    isL = typ == "L"
    xy = s["df"].to_numpy()
    Dm = cdist(xy, xy)
    d_ref = np.where(isL, np.sort(Dm[:, isL], axis=1)[:, 1], Dm[:, isL].min(axis=1))
    assert np.allclose(p["dist"], d_ref)
    assert np.allclose(p["local"], ((Dm <= 50) & isL[None, :]).sum(1) / (Dm <= 50).sum(1))
    v = res["validity"]
    assert v["valid"] is True and v["distance_ok"] and v["density_ok"]
    assert v["near_far_ratio"] > 5 and v["rho_local"] > 0.3
    non = ~isL
    assert v["near_mean"] == pytest.approx(share[non & (d_ref < 30)].mean())
    assert v["far_mean"] == pytest.approx(share[non & (d_ref >= 100)].mean())
    # T cells beside L cells are flagged, T cells far away are not
    isT = typ == "T"
    assert np.median(p["score"][isT & (d_ref < 15)]) > 0.10 > np.median(p["score"][isT & (d_ref > 150)])
    assert sum(res["flags"].values()) == res["n_non_source"] == non.sum()
    assert res["flags"]["none"] < non.sum() and res["by_type"].index[0] == "T"
    assert [b["band"] for b in res["profile"]][0] == "0–5" and len(res["by_depth"]) == 5


def test_spillover_validity_check_fails_when_the_signal_does_not_follow_distance(xe):
    s, typ, _a, _b = _bleed_world(seed=1, planted=False)
    res = xe._spillover([s], [typ], "L", [f"g{i}" for i in list(G_L)[:6]])
    v = res["validity"]
    assert v["distance_ok"] is False and v["valid"] is False and v["near_far_ratio"] < 2
    # markers the source type does not carry at all: refused
    X = s["csc"].toarray(); X[typ == "L", 41] = 0
    with pytest.raises(ValueError, match="not expressed by that type"):
        xe._spillover([{**s, "csc": sp.csc_matrix(X)}], [typ], "L", ["g41"])
    # sparse markers (most source cells have none): the mean is the reference, and said
    X = s["csc"].toarray().astype(float)
    keep = np.random.default_rng(0).random(len(X)) < 0.2
    X[~keep, 10] = 0
    r2 = xe._spillover([{**s, "csc": sp.csc_matrix(X)}], [typ], "L", ["g10"])
    share = X[:, 10] / np.maximum(X.sum(1), 1)
    assert r2["reference_kind"] == "mean" and r2["reference"] == pytest.approx(share[typ == "L"].mean())
    assert 0.1 < r2["source_detect"] < 0.3 and res["reference_kind"] == "median"


def test_spillover_is_computed_within_each_sample(xe):
    """Two samples at the same coordinates: the nearest source cell of a cell is always
    one of its own sample."""
    s1, t1, _a, _b = _bleed_world(seed=2, n=300)
    s2, t2, _a, _b = _bleed_world(seed=3, n=300)
    t2 = np.where(t2 == "L", "M", t2).astype(object)             # no source cell in sample 2
    res = xe._spillover([s1, s2], [t1, t2], "L", [f"g{i}" for i in list(G_L)[:6]])
    assert np.isfinite(res["per_sample"][0]["dist"]).all()
    assert np.isnan(res["per_sample"][1]["dist"]).all() and (res["per_sample"][1]["local"] == 0).all()


def _reference_gate(O, P, share_max=0.5, explain_min=0.5, floor=1e-9):
    """The gate of the analysis the method was ported from, for a paired cube [genes,
    pairs, (reference, treated)], restated."""
    lg = lambda x: np.log2(np.maximum(x, floor))
    o0, o1, p0, p1 = O[:, :, 0].mean(1), O[:, :, 1].mean(1), P[:, :, 0].mean(1), P[:, :, 1].mean(1)
    s0, s1 = p0 / np.maximum(o0, floor), p1 / np.maximum(o1, floor)
    e1 = (s0 > share_max) | (s1 > share_max)
    i0, i1 = np.maximum(o0 - p0, 0.0), np.maximum(o1 - p1, 0.0)
    d_obs = lg(o1) - lg(o0)
    d_leak = 0.5 * ((lg(i0 + p1) - lg(o0)) + (lg(o1) - lg(i1 + p0)))
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(d_obs != 0, d_leak / d_obs, np.nan)
    e2 = np.nan_to_num(f, nan=-np.inf) >= explain_min
    return s0, s1, f, e1 | e2


def test_leak_fraction_prediction_and_gate_equal_the_reference_formulas(xe):
    rng = np.random.default_rng(5)
    G, S = 300, 7
    xF = rng.gamma(2.0, 30.0, size=(G, 2 * S)); xF[rng.random((G, 2 * S)) < 0.1] = 0
    xL = {L: rng.gamma(2.0, 40.0, size=(G, 2 * S)) * (rng.random((G, 2 * S)) > 0.2) for L in "PQ"}
    M = {L: rng.choice(G, 20, replace=False) for L in "PQ"}
    leaks = {L: xe._leak_fraction(xF, xL[L], M[L]) for L in "PQ"}
    for L in "PQ":
        for s in range(2 * S):
            lv = xL[L][M[L], s]
            assert leaks[L][s] == pytest.approx(np.median(xF[M[L], s][lv > 0] / lv[lv > 0]))
    P, parts = xe._leak_predict(leaks, xL)
    assert np.allclose(P, sum(xL[L] * leaks[L][None, :] for L in "PQ"))
    P = P * rng.uniform(0, 3, size=(G, 1))                       # spread the shares around 0.5
    g = xe._leak_gate(xF[:, :S], xF[:, S:], P[:, :S], P[:, S:])
    s0, s1, f, flag = _reference_gate(np.stack([xF[:, :S], xF[:, S:]], 2), np.stack([P[:, :S], P[:, S:]], 2))
    assert np.allclose(g["share0"], s0) and np.allclose(g["share1"], s1)
    # the formulas are unchanged: "flag_all" is the reference rule on every gene
    assert np.allclose(g["f"], f, equal_nan=True) and np.array_equal(g["flag_all"], flag)
    assert 0.1 < flag.mean() < 0.95 and g["e1"].any() and (g["e2_all"] & ~g["e1"]).any()
    # the change rule (e2) is restricted to genes with a change: called, or |log2| >= the minimum
    big = np.abs(g["d_obs"]) >= xe._LEAK_CHANGE_MIN_LFC
    assert np.array_equal(g["change_ok"], big) and np.array_equal(g["e2"], g["e2_all"] & big)
    assert np.array_equal(g["flag"], g["e1"] | (g["e2_all"] & big)) and (g["flag_all"] & ~g["flag"]).any()
    called = rng.random(G) < 0.5
    gc = xe._leak_gate(xF[:, :S], xF[:, S:], P[:, :S], P[:, S:], called=called)
    assert np.array_equal(gc["flag"], g["e1"] | (g["e2_all"] & (big | called)))
    for k in ("share0", "share1", "f", "d_obs", "d_leak", "e1", "e2_all"):
        assert np.array_equal(gc[k], g[k], equal_nan=True)
    assert np.array_equal(xe._leak_gate(xF[:, :S], xF[:, S:], P[:, :S], P[:, S:], called=np.ones(G, bool))["flag"], flag)
    # a marker absent from a section is skipped; a section without any gives NaN → no leak
    z = np.zeros((G, 1))
    assert np.isnan(xe._leak_fraction(xF[:, :1], z, M["P"])[0])
    assert np.allclose(xe._leak_predict({"P": np.array([np.nan])}, {"P": xL["P"][:, :1]})[0], 0.0)


def test_leak_gate_flags_planted_bleed_and_spares_true_changes(xe):
    s, typ, A, B = _bleed_world(seed=4, n=2400)
    genes = s["genes"]
    g = xe._leak_gate_groups([s], [typ], [A], [B], genes)
    assert g["applicable"] and g["focal"] == "T" and g["mode"] == "both groups"
    t = g["table"].set_index("gene")
    L_genes, X_genes, T_genes = ([f"g{i}" for i in r] for r in (G_L, G_X, G_T))
    assert t.loc[L_genes, "leak_flag"].all()                    # bled-in genes: flagged
    assert (t.loc[L_genes, "top_leak_type"] == "L").all()
    assert not t.loc[X_genes, "leak_flag"].any()                # the real 3× change: kept
    assert not t.loc[T_genes, "leak_flag"].any()
    assert (t.loc[X_genes, "leak_fraction"] < 0.2).all() and (t.loc[L_genes, "leak_fraction"] > 0.5).all()
    # leak into group A is larger than into B, in the planted order
    assert g["leaks"]["L"]["a"] > 2 * g["leaks"]["L"]["b"] > 0 and g["leaks"]["L"]["rule"] == "specific"
    assert set(g["leaks"]["L"]["markers"]) <= set(L_genes) and len(g["leaks"]["L"]["markers"]) == 25
    # the genes the leak is measured on: the lowest focal ÷ neighbour ratios among L's specific genes
    prof = xe._type_profiles([s], [typ])
    mk, n_cand = xe._leak_markers(prof, "L", "T")
    cpm = prof["counts"] / prof["depth"][:, None]
    r = cpm[prof["types"].index("T")] / np.maximum(cpm[prof["types"].index("L")], 1e-300)
    cand = [f"g{i}" for i in G_L]
    assert n_cand == 30 and mk == sorted(cand, key=lambda g_: r[prof["genes"].index(g_)])[:25]
    # sub-types that share their expression: little leak is inferred, nothing is mass-flagged
    twin = np.where((typ == "M") & (np.arange(len(typ)) % 2 == 0), "T2", typ).astype(object)
    twin = np.where(typ == "T", np.where(np.arange(len(typ)) % 2 == 0, "T", "T2"), twin).astype(object)
    g2 = xe._leak_gate_groups([s], [twin], [A & (twin == "T")], [B & (twin == "T")], genes)
    assert g2["applicable"] and "T2" not in g2["leaks"] and set(g2["not_modelled"]) == {"M", "T2"} and "L" in g2["leaks"]
    assert not g2["table"].set_index("gene").loc[X_genes + T_genes, "leak_flag"].any()
    # a cell-level test calls the bled-in genes "up in A"; the gate is what tells them apart
    res = xe._de_compute([s], [A], [B])
    de = res["table"].set_index("gene")
    assert (de.loc[L_genes, "padj"] < 0.05).mean() > 0.9 and (de.loc[L_genes, "log2fc"] > 0).all()
    xe._annot_state["labels_celltypist"] = pd.Series(typ, index=s["df"].index.astype(str))
    ctx = {"samples": [s], "masks_a": [A], "masks_b": [B], "seg_source": "xenium", "color_by": "cell_type:celltypist",
           "options": {"leak_gate": True}}
    s["ds"] = xe.DATA                                            # labels are looked up for the primary sample
    xe._de_leak_annotate(res, ctx)
    tb = res["table"].set_index("gene")
    assert {"owner", "leak_fraction", "leak_flag", "leak_share_a", "leak_change_frac"} <= set(tb.columns)
    assert (tb.loc[L_genes, "owner"] == "L").all() and tb.loc[L_genes, "leak_flag"].all()
    assert (tb.loc[X_genes, "owner"] == "T").all() and not tb.loc[X_genes, "leak_flag"].any()
    assert "genes flagged" in res["leak_summary"] and "focal type T" in res["leak_summary"]
    # table, CSV columns and the volcano mark them
    cols = [c["id"] for c in xe._de_table_columns(res)]
    assert cols[-3:] == ["owner", "leak_fraction", "leak_flag"]
    recs = {r["gene"]: r for r in xe._de_table_records(res)}
    assert recs["g10"]["leak_flag"] == "FLAG" and recs["g70"]["leak_flag"] == "ok" and recs["g70"]["owner"] == "T"
    res.update({"label_a": "A", "label_b": "B", "seconds": 1})
    with xe._de_lock:
        xe._de_state["result"] = res
    try:
        fig = xe.make_de_volcano_fig()
        flagged = [tr for tr in fig.data if tr.name.startswith("leak-gate flagged")]
        assert len(flagged) == 1 and flagged[0].marker.symbol == "diamond-open"
        assert set(L_genes) <= set(flagged[0].text) and not set(X_genes) & set(flagged[0].text)
        up = [tr for tr in fig.data if tr.name.startswith("higher in A")][0]
        assert set(X_genes) <= set(up.text)
        # the option switches the annotation off
        res2 = xe._de_compute([s], [A], [B])
        xe._de_leak_annotate(res2, {**ctx, "options": {"leak_gate": False}})
        assert "owner" not in res2["table"].columns
    finally:
        with xe._de_lock:
            xe._de_state["result"] = None


def test_leak_gate_modes_and_refusals(xe):
    s, typ, A, B = _bleed_world(seed=6, n=1500)
    genes = s["genes"]
    rest = ~A
    g = xe._leak_gate_groups([s], [typ], [A], [rest], genes)       # T vs everything else
    assert g["applicable"] and g["mode"] == "group A only"
    t = g["table"].set_index("gene")
    assert (t["leak_share_b"] == 0).all() and t.loc[[f"g{i}" for i in G_L], "leak_flag"].all()
    mixed = np.ones(len(typ), bool)
    g = xe._leak_gate_groups([s], [typ], [mixed], [~mixed], genes)
    assert not g["applicable"] and "mixture of cell types" in g["reason"]
    g = xe._leak_gate_groups([s], [np.where(typ == "T", "T", None)], [A], [B], genes)
    assert not g["applicable"] and "no other cell type" in g["reason"]
    # pseudobulk: every sample is a section of its arm, with that sample's neighbours
    s2, typ2, A2, B2 = _bleed_world(seed=7, n=1500)
    none1, none2 = np.zeros(len(typ), bool), np.zeros(len(typ2), bool)
    g = xe._leak_gate_groups([s, s2], [typ, typ2], [A | B, none2], [none1, A2 | B2], genes, per_sample=True)
    assert g["applicable"] and g["n_sections_a"] == 1 and g["n_sections_b"] == 1 and g["mode"] == "both groups"


def test_leak_gate_change_rule_needs_a_change(xe):
    """A gene whose change is noise around 0 is not flagged by the change rule; a called
    gene whose change the bleed explains is; so is an uncalled gene with a large change."""
    #             null     called   large    real change
    O0 = np.array([100.0, 100.0, 100.0, 100.0])[:, None]
    O1 = np.array([101.0, 101.0, 160.0, 300.0])[:, None]
    P0 = np.array([1.0, 1.0, 0.0, 1.0])[:, None]
    P1 = np.array([2.0, 2.0, 60.0, 2.0])[:, None]
    called = np.array([False, True, False, True])
    g = xe._leak_gate(O0, O1, P0, P1, called=called)
    assert not g["e1"].any()                                   # no share above one half: only the change rule can flag
    assert abs(g["d_obs"][0]) < 0.02 and g["f"][0] > 0.9 and g["e2_all"][0]      # the old rule fired on the null gene
    assert g["flag"].tolist() == [False, True, True, False]
    assert g["flag_all"].tolist() == [True, True, True, False]
    assert g["change_ok"].tolist() == [False, True, True, True]
    # without calls only the minimum absolute effect opens the change rule
    assert xe._leak_gate(O0, O1, P0, P1)["flag"].tolist() == [False, False, True, False]
    assert xe._leak_gate(O0, O1, P0, P1, min_abs=0.0)["flag"].tolist() == [True, True, True, False]
    # the signal rule is untouched by the restriction
    g1 = xe._leak_gate(O0, O0, 0.6 * O0, 0.6 * O0)
    assert g1["e1"].all() and g1["flag"].all() and not g1["change_ok"].any()

    # through the groups: a world whose T cells carry a little of L everywhere (equal in A and B)
    s, typ, A, B = _bleed_world(seed=4, n=2400)
    genes = s["genes"]
    g0 = xe._leak_gate_groups([s], [typ], [A], [B], genes)
    none = xe._leak_gate_groups([s], [typ], [A], [B], genes, called=np.zeros(len(genes), bool))
    every = xe._leak_gate_groups([s], [typ], [A], [B], genes, called=np.ones(len(genes), bool))
    assert g0["n_flag"] == none["n_flag"] <= every["n_flag"] == every["n_flag_all"] == g0["n_flag_all"]
    assert g0["n_e2"] <= g0["n_e2_all"] and g0["n_change_ok"] <= len(genes) == every["n_change_ok"]
    t0, te = g0["table"].set_index("gene"), every["table"].set_index("gene")
    L_genes = [f"g{i}" for i in G_L]
    assert t0.loc[L_genes, "leak_flag"].all()                  # the planted bleed is flagged with or without calls
    # a gene flagged only by the unrestricted rule has a small change
    only = te.index[te["leak_flag"] & ~t0["leak_flag"]]
    prof = xe._type_profiles([s], [typ])
    for gname in only:
        assert gname not in L_genes


# ── 4. segmentation QC metrics ───────────────────────────────────────────────
def test_qc_metrics_equal_brute_force(xe):
    s, typ, _a, _b = _bleed_world(seed=8, n=700)
    typ = typ.copy(); typ[:25] = None
    prof = xe._type_profiles([s], [typ])
    sets = xe._identity_sets(prof, n=4)
    assert set(sets) == {"L", "M", "T"} and all(len(v) == 4 for v in sets.values())
    assert set(sets["L"]) <= {f"g{i}" for i in G_L}
    m = xe._qc_metrics(s, typ, sets, "L", radius=20.0)
    X = s["csc"].toarray().astype(float)
    tot = X.sum(1)
    gi = s["gni"]
    ident = {k: X[:, [gi[g] for g in v]].sum(1) for k, v in sets.items()}
    lab = pd.Series(typ)
    per = m["per_type"].set_index("type")
    fails, n = 0, 0
    for k in sets:
        c = (lab == k).to_numpy()
        own = ident[k][c] / np.maximum(tot[c], 1e-9)
        foreign = np.max([ident[o][c] / np.maximum(tot[c], 1e-9) for o in sets if o != k], axis=0)
        assert per.loc[k, "exclusion_failure"] == pytest.approx(np.mean(~(foreign < own)))
        assert per.loc[k, "own_per_100"] == pytest.approx(100 * ident[k][c].sum() / tot[c].sum())
        fails += (~(foreign < own)).sum(); n += c.sum()
    r = m["row"]
    assert r["exclusion_failure"] == pytest.approx(fails / n) and r["n_labelled"] == 675
    assert r["median_transcripts"] == np.median(tot) and r["median_genes"] == np.median((X >= 1).sum(1))
    xy = s["df"].to_numpy()
    isL = (lab == "L").to_numpy()
    labelled = lab.notna().to_numpy()
    Dm = cdist(xy, xy) <= 20.0
    n_src = (Dm & isL[None, :]).sum(1) - isL
    c = labelled & ~isL & (tot > 0)
    y = 100 * ident["L"][c] / tot[c]
    assert r["spill_slope"] == pytest.approx(np.polyfit(n_src[c], y, 1)[0]) and r["spill_slope"] > 0
    assert r["spill_mean"] == pytest.approx(y.mean())
    n_non = (Dm & (labelled & ~isL)[None, :]).sum(1) - (labelled & ~isL)
    c = isL & (tot > 0)
    y = 100 * (ident["M"][c] + ident["T"][c]) / tot[c]
    assert r["reverse_slope"] == pytest.approx(np.polyfit(n_non[c], y, 1)[0])
    tops = [sets[k][0] for k in sets]
    both = either = 0
    for a, b in itertools.combinations(tops, 2):
        da, db = X[:, gi[a]] >= 1, X[:, gi[b]] >= 1
        both += (da & db).sum(); either += (da | db).sum()
    assert r["coexpression_rate"] == pytest.approx(both / either) and r["n_marker_pairs"] == 3
    # a "correction" that removes the bleed: lower failure and slope, retention of own identity ~1;
    # one that also removes half of every cell's own identity: retention 0.5 (over-correction)
    Xc = X.copy()
    isT = (lab == "T").to_numpy()
    Xc[np.ix_(isT, [gi[f"g{i}"] for i in G_L])] = 0
    Xc *= 0.999                                                  # fractional values
    sc = {**s, "csc": sp.csc_matrix(Xc), "corrected": True}
    mc = xe._qc_metrics(sc, typ, sets, "L", 20.0, ref_ident=m["ident_sums"])
    assert mc["row"]["exclusion_failure"] <= r["exclusion_failure"] and abs(mc["row"]["spill_slope"]) < 1e-9
    assert mc["row"]["retention"] == pytest.approx(0.999)
    assert mc["row"]["median_genes"] < r["median_genes"]         # 0.999 of one transcript is not a detection
    Xo = Xc.copy()
    for k, v in sets.items():
        Xo[np.ix_((lab == k).to_numpy(), [gi[g] for g in v])] *= 0.5
    mo = xe._qc_metrics({**s, "csc": sp.csc_matrix(Xo)}, typ, sets, "L", 20.0, ref_ident=m["ident_sums"])
    assert mo["row"]["retention"] == pytest.approx(0.4995)
    # user sets and pairs; label transfer between segmentations of one section
    us, pairs = xe._parse_identity_sets("L: g10, G11 nope\n# c\nT: g0;g1\ng10, g40\ng1 g2 g3", s["genes"])
    assert us == {"L": ["g10", "g11"], "T": ["g0", "g1"]} and pairs == [("g10", "g40")]
    mp = xe._qc_metrics(s, typ, sets, "L", 20.0, pairs=pairs)
    da, db = X[:, 10] >= 1, X[:, 40] >= 1
    assert mp["row"]["coexpression_rate"] == pytest.approx((da & db).sum() / (da | db).sum())
    moved = xy + 3.0
    tr = xe._transfer_labels(xy, typ, np.vstack([moved, [[5000.0, 5000.0]]]), 10.0)
    assert tr[-1] is None and pd.Series(tr[:-1]).notna().mean() > 0.95
    assert (tr[30:200] == typ[30:200]).mean() > 0.7


def test_segmentation_qc_job_on_the_bundle(xe, D, labels):
    top = [D["gene_names"][i] for i in np.argsort(-np.asarray(D["expr"].sum(axis=0)).ravel())[:6]]
    sets = f"Alpha: {top[0]}, {top[1]}\nBeta: {top[2]}, {top[3]}\nGamma: {top[4]}, {top[5]}"
    xe._run_segqc("celltypist", sets, "auto", 15.0)
    st = xe._segqc_state
    assert st["status"] == "done", st["message"]
    res = st["result"]
    t = res["table"]
    assert list(t["source"]) [0] == "Xenium" and t.loc[0, "counts"] == "original" and t.loc[0, "labels"] == "own"
    assert t.loc[0, "n_cells"] == len(D["df"]) and res["sets_origin"] == "user"
    xe._run_segqc("celltypist", "", "auto", 15.0)        # 23 random-labelled cells: nothing to derive
    assert xe._segqc_state["status"] == "error" and "list the sets" in xe._segqc_state["message"]
    xe._run_segqc("celltypist", sets, "auto", 15.0)
    assert "no p-value is computed" in st["message"]
    recs = xe._segqc_records(res)
    assert recs[0]["source"] == "Xenium" and "exclusion_failure" in recs[0]
    csv = xe._segqc_csv(res)
    assert "# identity set" in csv and "# per cell type" in csv
    assert len(xe.make_segqc_fig().data) == 4
    xe._annot_state["labels_celltypist"] = None
    xe._run_segqc("celltypist", "", "auto", 15.0)
    assert xe._segqc_state["status"] == "error" and "no annotation" in xe._segqc_state["message"]


# ── 5. stratified co-localisation ────────────────────────────────────────────
def _niche_world(seed=0, n=5000, attract=False):
    """One section, 1000 × 1000 µm. Types A and B live only in the left 30 % ("niche"),
    at random positions among the C cells that fill everything; D is everywhere. With
    attract=True every B cell is put within 12 µm of an A cell."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 1000, size=(n, 2))
    left = xy[:, 0] < 300
    typ = np.where(left, rng.choice(["A", "B", "C", "D"], n, p=[0.15, 0.15, 0.5, 0.2]),
                   rng.choice(["C", "D"], n, p=[0.7, 0.3])).astype(object)
    if attract:
        a_xy = xy[typ == "A"]
        ib = np.flatnonzero(typ == "B")
        xy[ib] = a_xy[rng.integers(0, len(a_xy), len(ib))] + rng.normal(0, 6, size=(len(ib), 2))
    comp = np.where(left, 0, 1)
    return xy, typ, comp


def _prep(xe, xy, typ, comp, radius=30.0, sample=None):
    types = sorted(set(typ) - {None})
    code = np.array([types.index(t) if t is not None else -1 for t in typ])
    sample = np.zeros(len(xy), dtype=int) if sample is None else sample
    xys = [xy[sample == s] for s in np.unique(sample)]
    codes = [code[sample == s] for s in np.unique(sample)]
    cnt = xe._coloc_counts(xys, codes, len(types), radius)
    edge = np.concatenate([xe._edge_distance(x, 50.0)[0] for x in xys])
    dens = np.concatenate([xe._local_counts(x, 100.0) for x in xys])
    levels = xe._coloc_levels(sample, comp, edge, dens)
    return types, code, sample, cnt, levels


def test_neighbour_counts_and_expected_values_equal_brute_force(xe):
    xy, typ, comp = _niche_world(seed=1, n=1200)
    typ[:40] = None
    types, code, sample, cnt, levels = _prep(xe, xy, typ, comp, radius=40.0)
    Dm = cdist(xy, xy) <= 40.0
    np.fill_diagonal(Dm, False)
    for b, t in enumerate(types):
        assert np.array_equal(cnt[b], (Dm & (typ == t)[None, :]).sum(1))
    res = xe._coloc_enrich(cnt, code, sample, levels, len(types), 1, min_pool=3)
    a, b = types.index("A"), types.index("B")
    foc = np.flatnonzero(code == a)
    pool = (code >= 0) & (code != a) & (code != b)
    O = E = 0.0
    kept = lev0 = 0
    for i in foc:
        for li, lv in enumerate(levels):
            m = pool & (lv == lv[i])
            if m.sum() >= 3:
                O += cnt[b][i]; E += cnt[b][m].mean(); kept += 1; lev0 += li == 0
                break
    assert res["O"][0, a, b] == pytest.approx(O) and res["E"][0, a, b] == pytest.approx(E)
    assert res["e"][0, a, b] == pytest.approx(np.log2((O + 0.5) / (E + 0.5)))
    assert res["n_kept"][0, a, b] == kept and res["lev0"][0, a, b] == pytest.approx(lev0 / kept)
    assert res["n_type"][0, a] == len(foc)
    # diagonal: the pool is every other type
    pool = (code >= 0) & (code != a)
    O = sum(cnt[a][i] for i in foc if any((pool & (lv == lv[i])).sum() >= 3 for lv in levels))
    assert res["O"][0, a, a] == pytest.approx(O)
    # strata helpers
    assert np.array_equal(xe._rank_bin(np.arange(10), 5), np.repeat(np.arange(5), 2))
    assert len(np.unique(levels[3])) == 1 and len(np.unique(levels[2])) == 2
    assert all(len(np.unique(levels[i])) >= len(np.unique(levels[i + 1])) for i in range(3))


def test_shared_niche_is_called_by_the_naive_test_and_not_by_the_stratified_one(xe):
    xy, typ, comp = _niche_world(seed=2)
    types, code, sample, cnt, levels = _prep(xe, xy, typ, comp)
    T, a, b = len(types), types.index("A"), types.index("B")
    naive_levels = [levels[3]]
    naive = xe._coloc_enrich(cnt, code, sample, naive_levels, T, 1)["e"][0, a, b]
    e0n, epn = xe._coloc_perm(cnt, code, sample, levels[3], T, 1, n_perm=200, seed=0, focal_idx=[a])
    _bias, p_naive = xe._perm_p(e0n, epn)
    assert naive > 1.0 and p_naive[0, 0, b] == pytest.approx(1 / 201)          # "A attracts B", wrongly
    strat = xe._coloc_enrich(cnt, code, sample, levels, T, 1)["e"][0, a, b]
    e0, ep = xe._coloc_perm(cnt, code, sample, levels[0], T, 1, n_perm=200, seed=0, focal_idx=[a])
    bias, p = xe._perm_p(e0, ep)
    assert abs(strat - bias[0, 0, b]) < 0.2 and p[0, 0, b] > 0.05               # same place, no attraction
    assert np.isnan(p[0, 0, a])                                                 # focal = target: no permutation p
    # the same statistic finds a planted attraction inside the niche
    xy2, typ2, comp2 = _niche_world(seed=2, attract=True)
    types, code, sample, cnt, levels = _prep(xe, xy2, typ2, comp2)
    strat2 = xe._coloc_enrich(cnt, code, sample, levels, T, 1)["e"][0, a, b]
    e0, ep = xe._coloc_perm(cnt, code, sample, levels[0], T, 1, n_perm=200, seed=0, focal_idx=[a])
    bias2, p2 = xe._perm_p(e0, ep)
    assert strat2 - bias2[0, 0, b] > 0.4 and p2[0, 0, b] == pytest.approx(1 / 201)
    # the observed single-level statistic of the permutation function equals _coloc_enrich
    # restricted to that level, and permuting is reproducible
    one = xe._coloc_enrich(cnt, code, sample, [levels[0]], T, 1)["e"][0]
    e_all, _ = xe._coloc_perm(cnt, code, sample, levels[0], T, 1, n_perm=0)
    off = ~np.eye(T, dtype=bool)
    assert np.allclose(e_all[0][off], one[off])
    _e, ep_b = xe._coloc_perm(cnt, code, sample, levels[0], T, 1, n_perm=200, seed=0, focal_idx=[a])
    assert np.array_equal(ep, ep_b, equal_nan=True)


def test_permutation_keeps_the_number_of_focal_cells_per_stratum(xe):
    """One permutation by hand: with the same shuffle, the pseudo-focal cells are the
    first n_A candidates (not B) of each stratum."""
    xy, typ, comp = _niche_world(seed=3, n=1500)
    types, code, sample, cnt, levels = _prep(xe, xy, typ, comp)
    T, a, b = len(types), types.index("A"), types.index("B")
    st = np.unique(levels[0], return_inverse=True)[1]
    _e0, ep = xe._coloc_perm(cnt, code, sample, st, T, 1, n_perm=1, seed=11, focal_idx=[a])
    order = np.argsort(st + np.random.default_rng(11).random(len(st)), kind="stable")
    O = E = 0.0
    for s_ in np.unique(st):
        cand = [i for i in order if st[i] == s_ and code[i] != b]
        k = int(((st == s_) & (code == a)).sum())
        if k == 0 or len(cand) - k < 3:
            continue
        foc, pool = cand[:k], cand[k:]
        O += cnt[b][foc].sum(); E += k * cnt[b][pool].mean()
    assert ep[0, 0, 0, b] == pytest.approx(np.log2((O + 0.5) / (E + 0.5)))


def test_no_neighbour_count_joins_cells_of_two_samples(xe):
    """Two samples at IDENTICAL coordinates (the worst case: no offset at all): every
    count equals the count within the cell's own sample."""
    xy, typ, comp = _niche_world(seed=4, n=800)
    xy2 = np.vstack([xy, xy]); typ2 = np.concatenate([typ, typ]); comp2 = np.concatenate([comp, comp])
    sample = np.repeat([0, 1], len(xy))
    types, code, smp, cnt, levels = _prep(xe, xy2, typ2, comp2, sample=sample)
    _t, _c, _s, cnt1, _l = _prep(xe, xy, typ, comp)
    assert np.array_equal(cnt[:, :len(xy)], cnt1) and np.array_equal(cnt[:, len(xy):], cnt1)
    assert not np.intersect1d(levels[3][sample == 0], levels[3][sample == 1]).size      # strata are nested in samples
    res = xe._coloc_enrich(cnt, code, smp, levels, len(types), 2)
    one = xe._coloc_enrich(cnt1, code[:len(xy)], smp[:len(xy)], _l, len(types), 1)
    assert np.allclose(res["O"][0], one["O"][0]) and np.allclose(res["O"][1], one["O"][0])
    # compartments, density, spillover geometry: one section per call by construction
    assert np.array_equal(xe._local_counts(xy, 50.0), xe._local_counts(xy2[:len(xy)], 50.0))


def test_every_neighbour_search_of_the_app_is_in_a_function_that_takes_one_sample(xe):
    """Audit of the source: KD-tree / nearest-neighbour searches appear only in the
    functions listed here, each of which is given the cells of a single sample (the
    callers loop over the samples). A new call site fails this test until it is added
    here - after checking that it cannot join cells of different samples."""
    allowed = {
        "controls.py": {"_local_counts", "_source_geometry", "_coloc_counts", "_transfer_labels"},
        "stats.py": {"_knn_indices"},                    # Moran, LR: called per sample
        "proximity.py": {"_prox_sample", "_contact_near"},              # one sample's coordinates each
        "lrdecomp.py": {"_lrd_contacts"},                               # one KD-tree per sample
        "aggregates.py": {"_agg_adjacency", "_agg_edges", "_agg_tile_area", "_agg_hull", "_agg_sample"},
        "spatial.py": {"_run_spatial_analysis", "_run_spatial_niches"},   # primary sample's table only
        "reseg.py": {"_patch_seam_dedup_mask"},          # duplicate cells across patch seams of one run
        "annotation.py": {"_run_seurat_annotation"},     # expression space (PCA), not tissue space
        "spage.py": {"_spage_impute", "_run_spage_imputation", "_spage_core"},   # expression space
        "banksy.py": {"_bk_neighbours"},                 # ONE sample's coordinates (callers loop over samples)
        "clustersets.py": {"_cs_knn"},                   # embedding space (PCs), not tissue space
    }
    names = {"cKDTree", "NearestNeighbors", "KDTree", "BallTree", "Delaunay", "kneighbors_graph",
             "radius_neighbors_graph"}
    found = {}
    for path in tl.app_sources():
        tree = ast.parse(path.read_text())
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            for node in ast.walk(fn):
                if isinstance(node, ast.Call):
                    f = node.func
                    nm = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
                    if nm in names:
                        found.setdefault(path.name, set()).add(fn.name)
    spatial_files = {k: v for k, v in found.items() if k != "spage.py"}
    for fname, fns in spatial_files.items():
        inner = {f for f in fns}
        assert fname in allowed, f"{fname}: neighbour search in {fns}"
        # nested helper functions are reported with their own name and their parent's
        assert inner <= allowed[fname] | {"_smp"}, f"{fname}: unlisted neighbour search in {inner - allowed[fname]}"
    assert "controls.py" in found and "stats.py" in found


def test_moran_and_ligand_receptor_graphs_are_built_per_sample(xe, monkeypatch):
    """Two samples at the same coordinates: the kNN search is called once per sample with
    that sample's cells only, and the pooled LR adjacency has no edge between them."""
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 200, size=(120, 2))
    counts = rng.poisson(2.0, size=(120, 6))
    s1, s2 = _hand_sample(counts, xy, label="one"), _hand_sample(counts, xy, label="two")
    seen = []
    real = xe._knn_indices

    def spy(pts, k):
        seen.append(len(pts))
        idx = real(pts, k)
        assert idx.max() < len(pts)
        return idx
    monkeypatch.setattr(xe, "_knn_indices", spy)
    xe._moran_sample(s1, 6)
    lab = np.array(["u", "v"] * 60, dtype=object)
    r = xe._lr_compute([s1, s2], [lab, lab], [("g0", "g1")], k=6, n_perm=5, min_edges=1)
    assert seen == [120, 120, 120] and r["info"]["n_cells"] == 240
    # identical samples: twice the adjacent pairs of one sample, never a pair across them
    one = xe._lr_compute([s1], [lab], [("g0", "g1")], k=6, n_perm=5, min_edges=1)
    assert (r["table"]["n_edges"].to_numpy() == 2 * one["table"]["n_edges"].to_numpy()).all()


def test_view_offset_keeps_samples_apart_but_no_analysis_relies_on_it(xe, D, tmp_path):
    """Added samples are drawn at least 500 µm apart; the analyses above nevertheless use
    each sample's own coordinates (x_offset is applied only when drawing)."""
    e = _fab_ds(D, 50, 1, str(tmp_path / "off"))
    xe.EXTRA_DATASETS.append(e)
    tabs = xe._view_tables("xenium")
    assert tabs[1]["x_offset"] == e["x_offset"] > 0
    assert np.array_equal(xe._xy_of(tabs[1]["df"])[:, 0], e["df"]["x_centroid"].to_numpy())


def test_across_sample_inference_and_single_sample_statement(xe):
    T = 3
    rng = np.random.default_rng(0)
    e = rng.normal(0, 0.05, size=(6, T, T))
    e[:, 0, 1] += 1.0                                             # enriched in every sample
    e[:3, 1, 2] += 1.0                                            # only in condition A
    ok = np.ones((6, T, T), bool)
    conds, donors = ["A"] * 3 + ["B"] * 3, [""] * 6
    ac = xe._coloc_across(e, ok, conds, donors, list("abcdef"))
    assert ac["mode"] == "cohort" and ac["units_note"] == "n = 6 samples"
    assert ac["p_enrich"][0, 1] == ac["min_p"] == 2 / 64 and ac["p_enrich"][2, 0] > 0.05
    assert ac["p_diff"][1, 2] == ac["min_p_diff"] == 2 / 20 and abs(ac["effect_diff"][1, 2] - 1.0) < 0.15
    # paired donors: sign-flip over 3 donors; untestable samples are left out per pair
    ac = xe._coloc_across(e, ok, ["A", "B"] * 3, ["d1", "d1", "d2", "d2", "d3", "d3"], list("abcdef"))
    assert "3 donors" in ac["units_note"] and ac["p_enrich"][0, 1] == 0.25 and "paired" in ac["diff_note"]
    ok2 = ok.copy(); ok2[:4, 0, 1] = False
    assert np.isnan(xe._coloc_across(e, ok2, conds, donors, list("abcdef"))["p_enrich"][0, 1])
    # fewer than the minimum number of units: no test across samples
    ac = xe._coloc_across(e[:2], ok[:2], ["", ""], ["", ""], ["a", "b"])
    assert ac["mode"] == "per-sample" and np.isnan(ac["p_enrich"]).all() and ac["p_diff"] is None


def test_placebo_anchors_separate_a_specific_neighbour_from_a_compartment_effect(xe):
    # no attraction: A is near B only because both are niche cells → not specific
    for attract, want in ((False, False), (True, True)):
        xy, typ, comp = _niche_world(seed=7, attract=attract)
        rng = np.random.default_rng(9)
        left = np.flatnonzero((comp == 0) & (typ == "C"))
        typ[rng.choice(left, 250, replace=False)] = "E"           # a niche type of similar abundance
        types, code, sample, cnt, levels = _prep(xe, xy, typ, comp)
        prep = {"types": types, "code": code, "sample": sample, "cnt": cnt, "levels": levels,
                "comp": comp, "comp_names": ["niche", "rest"]}
        pl = xe._coloc_placebo(prep, "A", 1, draws=100, seed=1).set_index("target")
        row = pl.loc["B"]
        assert row["n_similar"] >= 1 and "E" in row["similar_types"]
        assert row["verdict"].startswith("specific") is want, row["verdict"]
        assert abs(row["random_mean"]) < 0.25 and row["random_lo"] < row["random_mean"] < row["random_hi"]
        if want:
            assert row["e_focal"] > row["random_hi"] and row["p_random"] == pytest.approx(1 / 101)


def _fake_view(xe, monkeypatch, n_samples, n=1500):
    """Make the active view *n_samples* fabricated sections (the bundle has 23 cells)."""
    tables, labs = [], []
    for i in range(n_samples):
        xy, typ, _comp = _niche_world(seed=20 + i, n=n, attract=True)
        df = pd.DataFrame({"x_centroid": xy[:, 0], "y_centroid": xy[:, 1]}, index=[f"v{i}_{k}" for k in range(n)])
        tables.append({"label": f"S{i}" if n_samples > 1 else "xenium", "df": df, "ds": {}, "key": f"/fake/s{i}",
                       "name": f"s{i}", "x_offset": 0.0})
        labs.append(typ)
    monkeypatch.setattr(xe, "_view_tables", lambda seg_source="xenium": tables)
    monkeypatch.setattr(xe, "_comp_labels", lambda kind, t, seg, method: (labs, "cell type (test)"))
    return tables, labs


def test_colocalisation_job_single_sample_and_multi_sample(xe, monkeypatch):
    tables, labs = _fake_view(xe, monkeypatch, 1)
    xe._run_coloc("xenium", None, "celltype:celltypist", 20.0, 20, "A")
    st = xe._coloc_state
    assert st["status"] == "done", st["message"]
    res = st["result"]
    assert res["mode"] == "per-sample" and "SINGLE-SAMPLE RESULT" in st["message"]
    assert "do not generalise" in st["message"] and "Compartments were not computed" in st["message"]
    assert "p_within" in res["summary"].columns and "p_enrich" not in res["summary"].columns
    assert len(res["summary"]) == 12 and res["placebo"] is not None and len(res["placebo"]) == 3
    sm = res["summary"].set_index(["focal", "target"])
    assert sm.loc[("A", "B"), "effect"] > 1 and sm.loc[("A", "B"), "p_within"] == pytest.approx(1 / 21)
    assert "p_within" in [c["id"] for c in xe._coloc_columns(res)]
    csv = xe._coloc_csv(res)
    assert "effect_naive" in csv and "placebo anchors for A" in csv and "unit = cells of one section" in csv
    assert xe.make_coloc_fig().data[0].type == "heatmap" and len(xe._placebo_records(res)) == 3
    # stored compartments are used as strata, and then nothing is said about defaults
    xe._run_compartments("xenium", None, "celltype:celltypist", "C", {"min_edge": 100})
    assert xe._comp_ctl_state["status"] == "done", xe._comp_ctl_state["message"]
    xe._run_coloc("xenium", None, "celltype:celltypist", 20.0, 5, None)
    assert "Compartments were not computed" not in xe._coloc_state["message"]
    assert "core" in xe._coloc_state["message"]
    # six sections → six units: the exact sign-flip test across samples, units stated
    xe._controls_reset()
    tables, labs = _fake_view(xe, monkeypatch, 6, n=1200)
    xe._run_coloc("xenium", None, "celltype:celltypist", 20.0, 10, None)
    st = xe._coloc_state
    assert st["status"] == "done", st["message"]
    res = st["result"]
    assert res["mode"] == "cohort" and res["n_samples"] == 6
    assert "unit of the p-values: n = 6 samples" in st["message"] and "smallest achievable p 0.0312" in st["message"]
    assert {"p_enrich", "fdr_enrich", "n_units"} <= set(res["summary"].columns)
    sm = res["summary"].set_index(["focal", "target"])
    assert sm.loc[("A", "B"), "p_enrich"] == 2 / 64 and sm.loc[("A", "B"), "n_positive"] == 6
    assert set(res["long"]["sample"]) == {f"S{i}" for i in range(6)}
    # with a design: the difference between conditions is tested as well
    xe._design_set([{"key": t["key"], "cond": "A" if i < 3 else "B", "batch": "", "donor": ""}
                    for i, t in enumerate(tables)])
    xe._run_coloc("xenium", None, "celltype:celltypist", 20.0, 0, None)
    res = xe._coloc_state["result"]
    assert "p_diff" in res["summary"].columns and "Between conditions" in xe._coloc_state["message"]
    # donors make it paired, and 3 pairs cannot reach 0.05: said
    xe._design_set([{"key": t["key"], "cond": "AB"[i % 2], "batch": "", "donor": f"d{i // 2}"}
                    for i, t in enumerate(tables)])
    xe._run_coloc("xenium", None, "celltype:celltypist", 20.0, 0, None)
    msg = xe._coloc_state["message"]
    assert "3 donors" in msg and "no test can be significant" in msg


# ── 6. guards ────────────────────────────────────────────────────────────────
def test_detection_on_fractional_counts_needs_one_transcript(xe):
    rng = np.random.default_rng(0)
    n = 400
    X = rng.poisson(1.5, size=(n, 8)).astype(float)
    Xf = X * rng.uniform(0.3, 1.0, size=X.shape)                 # "corrected": fractional residues
    xy = rng.uniform(0, 300, size=(n, 2))
    a = np.arange(n) < 200
    s = _hand_sample(Xf, xy)
    res = xe._de_compute([s], [a], [~a])
    t = res["table"]
    assert np.allclose(t["pct_a"], 100 * (Xf[a] >= 1).mean(0)) and np.allclose(t["pct_b"], 100 * (Xf[~a] >= 1).mean(0))
    assert (t["pct_a"].to_numpy() < 100 * (Xf[a] > 0).mean(0) - 5).all()      # "> 0" would count the residues
    assert np.allclose(t["rate_a"], 1e4 * Xf[a].sum(0) / Xf[a].sum())
    assert np.allclose(t["rate_b"], 1e4 * Xf[~a].sum(0) / Xf[~a].sum())
    # integer counts: the rule is the usual count > 0, and the test statistic is untouched
    si = _hand_sample(X, xy)
    ti = xe._de_compute([si], [a], [~a])["table"]
    assert np.allclose(ti["pct_a"], 100 * (X[a] > 0).mean(0))
    from scipy.stats import mannwhitneyu
    lib = X.sum(1, keepdims=True)
    ln = np.log1p(X / lib * 1e4)
    p = [mannwhitneyu(ln[a, j], ln[~a, j], alternative="two-sided", method="asymptotic", use_continuity=False).pvalue
         for j in range(8)]
    assert np.allclose(ti["pval"], p, rtol=2e-3)
    # Moran's % cells and the ligand-receptor detection fraction use the same rule
    m = xe._moran_sample(s, 6)["table"]
    assert np.allclose(m["pct_cells"], 100 * (Xf >= 1).mean(0))
    lab = np.array(["u", "v"] * (n // 2), dtype=object)
    lr = xe._lr_compute([s], [lab], [("g0", "g1")], k=6, n_perm=5, min_edges=1, min_frac=0.0)["table"]
    row = lr[(lr["sender"] == "u") & (lr["receiver"] == "v")].iloc[0]
    assert row["frac_ligand"] == pytest.approx((Xf[lab == "u", 0] >= 1).mean())
    assert row["frac_receptor"] == pytest.approx((Xf[lab == "v", 1] >= 1).mean())
    assert "count ≥ 1" in xe._DETECT_RULE and xe._DETECT_MIN == 1.0


def test_de_reports_the_depth_of_the_groups_and_warns(xe):
    rng = np.random.default_rng(1)
    n = 300
    a = np.arange(n) < 150
    X = rng.poisson(np.where(a[:, None], 6.0, 2.0), size=(n, 12)).astype(float)     # A is 3× deeper
    s = _hand_sample(X, rng.uniform(0, 100, size=(n, 2)))
    res = xe._de_compute([s], [a], [~a])
    assert res["median_counts_a"] == np.median(X[a].sum(1)) and res["median_counts_b"] == np.median(X[~a].sum(1))
    assert res["depth_ratio"] == pytest.approx(res["median_counts_a"] / res["median_counts_b"]) and res["depth_ratio"] > 2
    note = xe._depth_note(res["median_counts_a"], res["median_counts_b"])
    assert "differ in depth by more than 1.5×" in note and "rate columns" in note
    assert "⚠" not in xe._depth_note(100.0, 120.0) and "ratio 1.20" in xe._depth_note(100.0, 120.0)
    # every gene has the same share of the counts in both groups: more cells "express" it
    # in the deep group, the rate does not move
    t = res["table"]
    assert (t["pct_a"] > t["pct_b"]).all() and np.allclose(np.log2(t["rate_a"] / t["rate_b"]), 0, atol=0.25)
    assert t["low_count"].dtype == bool and not t["low_count"].any()


def test_de_status_states_the_unit_depth_and_detection_rule(xe, D, labels, tmp_path):
    xe.EXTRA_DATASETS.extend([_fab_ds(D, 200, i, str(tmp_path / f"de{i}")) for i in range(2)])
    xe._run_de("celltype:Alpha", "rest", "xenium", None, "cell_type:celltypist", False, {"leak_gate": True})
    st = xe._de_state
    assert st["status"] == "done", st["message"]
    n = st["result"]["n_a"] + st["result"]["n_b"]
    assert f"n = {n:,} cells, treated as independent" in st["message"]
    assert "Median counts per cell: A" in st["message"] and "count ≥ 1 transcript" in st["message"]
    assert "Gene ownership" in st["message"] and "owner" in st["result"]["table"].columns
    out = xe.download_de_table(1)["content"]
    assert "# unit of the p-values: n = " in out and "# median counts per cell" in out and "owner" in out
    assert "rate_a" in out.splitlines()[[i for i, l in enumerate(out.splitlines()) if not l.startswith("#")][0]]


def test_pseudobulk_volcano_colours_by_fdr_and_offers_both_estimates(xe):
    t = pd.DataFrame({"gene": ["sig_shrunk_to_0", "sig", "ns", "na"],
                      "log2fc": [2.0, -1.5, 0.3, 4.0], "log2fc_shrunk": [0.01, -1.2, 0.0, 0.0],
                      "padj": [0.001, 0.01, 0.6, np.nan], "pval": [1e-5, 1e-4, 0.5, 0.2]})
    t["neg_log10_padj"] = -np.log10(t["padj"])
    res = {"kind": "pseudobulk", "table": t, "label_a": "A", "label_b": "B", "n_a": 3, "n_b": 3,
           "shrink": "apeglm", "title": "t", "seconds": 1.0}
    with xe._de_lock:
        xe._de_state["result"] = res
    try:
        for axis, x_want, word in (("mle", 2.0, "unshrunken (MLE)"), ("shrunk", 0.01, "apeglm-shrunken"), (None, 2.0, "MLE")):
            fig = xe.make_de_volcano_fig(lfc_axis=axis)
            up = [tr for tr in fig.data if tr.name.startswith("higher in A")][0]
            assert list(up.text) == ["sig_shrunk_to_0"] and up.x[0] == pytest.approx(x_want)
            assert up.marker.color == "#f85149" and "FDR < 0.05" in up.name and "|log2FC|" not in up.name
            assert word in fig.layout.xaxis.title.text
            dn = [tr for tr in fig.data if tr.name.startswith("higher in B")][0]
            assert list(dn.text) == ["sig"]
        cols = [c["name"] for c in xe._de_table_columns(res)]
        assert "log2FC (MLE)" in cols and "log2FC (apeglm)" in cols
        # the cell-level test keeps its effect-size threshold, and says so
        res2 = {**res, "kind": None, "table": t.assign(pct_a=1, pct_b=1, mean_a=1, mean_b=1, z=1)}
        with xe._de_lock:
            xe._de_state["result"] = res2
        assert "|log2FC| ≥ 0.5" in [tr for tr in xe.make_de_volcano_fig().data if tr.name.startswith("higher in A")][0].name
    finally:
        with xe._de_lock:
            xe._de_state["result"] = None


def test_label_collapse_report(xe, D, labels):
    rng = np.random.default_rng(0)
    n = 1000
    lineage = np.repeat(["1", "2"], n // 2)
    sub = np.concatenate([np.repeat(["a", "b", "c"], [200, 200, 100]), np.repeat(["d"], 500)])
    lab = np.concatenate([np.where(rng.random(500) < 0.95, "T cell", "NK"),            # 3 sub-clusters, one label
                          np.where(rng.random(500) < 0.95, "Fibroblast", "Mural")])     # one cluster, one label: fine
    dist, flags = xe._label_collapse(lab, lineage, sub)
    assert dist.iloc[0]["n"] == max((lab == "T cell").sum(), (lab == "Fibroblast").sum()) and dist["pct"].sum() == pytest.approx(100)
    assert [f["lineage"] for f in flags] == ["1"] and flags[0]["label"] == "T cell" and flags[0]["n_sub"] == 3
    assert flags[0]["share"] == pytest.approx((lab[:500] == "T cell").mean())
    # without lineages only the share of the whole tissue is checked
    assert xe._label_collapse(lab)[1] == []
    _d, f1 = xe._label_collapse(np.where(rng.random(n) < 0.9, "X", "Y"))
    assert len(f1) == 1 and f1[0]["lineage"] == "all cells" and f1[0]["n_sub"] is None
    assert xe._label_collapse([None] * 10)[1] == [] and len(xe._label_collapse([None] * 10)[0]) == 0
    # the dialog shows the caution and the counts
    rep = xe.update_annot_label_report(True, 1, 0, "xenium")
    text = " ".join(str(getattr(c, "children", c)) for c in rep)
    assert "collapse" in xe._ANNOT_CAUTION or "same label" in xe._ANNOT_CAUTION
    assert "targeted panel" in text and "CellTypist" in text and "Alpha" in text
    with pytest.raises(Exception):
        xe.update_annot_label_report(False, 1, 0, "xenium")


def test_spillover_job_and_color_by(xe, D, labels, expressed_gene, tmp_path):
    xe.EXTRA_DATASETS.extend([_fab_ds(D, 200, i, str(tmp_path / f"sp{i}")) for i in range(2)])
    expressed_gene = D["gene_names"][0]                         # the fabricated samples carry counts in it
    xe._run_spillover("xenium", None, "celltype:celltypist", "auto", expressed_gene, 5, False)
    st = xe._spill_state
    assert st["status"] == "done", st["message"]
    assert "Validity check" in st["message"] and "Gene ownership" in st["message"] and "depth quintile" in st["message"]
    res = st["result"]
    assert res["marker_info"]["rule"] == "user" and res["markers"] == [expressed_gene]
    t = xe._view_tables("xenium")[0]
    sc = xe._ctl_get(t, "xenium", "spillover")
    assert len(sc) == len(D["df"]) and np.isfinite(sc).all() and res["reference"] > 0
    assert set(pd.unique(xe._ctl_get(t, "xenium", "spillover_flag"))) <= set(xe._SPILL_FLAG_NAMES)
    vals, _lim, label = xe._geneset_color("geneset:@spillover", "xenium", False)
    assert np.allclose(vals[:len(sc)], sc, equal_nan=True) and label.startswith("Spillover score")
    assert len(vals) == len(D["df"]) + 400
    assert "geneset:@spillover" in [o["value"] for o in xe._ctl_color_options()]
    assert len(xe.make_spill_fig().data) == 2 and xe._ownership_records(res)
    csv = xe._spill_csv(res)
    assert "# spillover source type" in csv and "second_ratio" in csv
    xe._run_spillover("xenium", None, "celltype:celltypist", "Nope", "", 5, False)
    assert xe._spill_state["status"] == "error" and "labelled 'Nope'" in xe._spill_state["message"]


def test_new_tabs_jobs_and_notes_are_in_the_app(xe):
    def walk(node, out):
        cid = getattr(node, "id", None)
        if isinstance(cid, str):
            out.add(cid)
        ch = getattr(node, "children", None)
        for c in (ch if isinstance(ch, (list, tuple)) else [ch] if ch is not None else []):
            if hasattr(c, "to_plotly_json") or isinstance(c, (list, tuple)):
                walk(c, out) if not isinstance(c, (list, tuple)) else [walk(x, out) for x in c if hasattr(x, "to_plotly_json")]
        return out
    ids = walk(xe.app.layout, set())
    for cid in ("ctl-comp-run-btn", "spill-run-btn", "segqc-run-btn", "coloc-run-btn", "ctl-poll", "comp-test",
                "de-leak-gate", "de-lfc-axis", "annot-label-report", "coloc-design-open-btn", "segqc-table",
                "coloc-placebo-table", "spill-own-table", "leak-note"):
        assert cid in ids, cid
    for kind in ("compartments", "spillover", "segqc", "coloc"):
        assert kind in xe._JOB_KIND_NAMES
    assert "do not generalise" not in xe._COLOC_NOTE and "exchangeable" in xe._COLOC_NOTE
    assert "leans the estimate low" in xe._LEAK_NOTE and "over-correction" in xe._SEGQC_NOTE
    assert "must not be used as a contamination mask" in xe._SPILL_NOTE
    assert "donor" in xe._PB_CAVEAT and "exact (sample-level)" in xe._COMP_NOTE
    # the poll releases the buttons and reports new results once
    out = xe.poll_controls(1, -1, -1, -1, -1)
    assert out[4:8] == (False, False, False, False) and out[-1] is True


# ── 7. cancel, export, fraction of transcripts assigned ──────────────────────
def _gate(xe, monkeypatch, name):
    """Replace xe.<name> by a version that reports it was entered and then waits until the
    test lets it go on: the job is held inside its computation while Cancel is pressed."""
    import threading
    real = getattr(xe, name)
    entered, release = threading.Event(), threading.Event()

    def held(*a, **k):
        entered.set()
        assert release.wait(20), "the test never released the job"
        return real(*a, **k)

    monkeypatch.setattr(xe, name, held)
    return real, entered, release


def _cancel_mid_run(xe, monkeypatch, kind, hold, start, state, published):
    """Start a controls job with *start*, cancel it while it is inside xe.<hold>, and check
    it ends cancelled with nothing published; then run it again to completion."""
    names = ("compartments", "spillover", "segqc", "coloc")
    real, entered, release = _gate(xe, monkeypatch, hold)
    v0 = state["version"]
    assert start()[:2] == (True, False)
    job = [j for j in xe._jobs.values() if j.kind == kind][-1]
    assert entered.wait(20), state["message"]
    assert job.snapshot()["cancellable"] and xe._job_cancel(job.id)
    assert job.state == "cancelling" and not xe._job_cancel(job.id)      # a second Cancel is refused
    release.set()
    job.thread.join(20)
    assert job.state == "cancelled" and job.cancelled and job.error is None
    assert state["status"] == "error" and state["message"] == "Cancelled"
    assert state["result"] is None and state["version"] == v0, "a cancelled job published a result"
    assert not published(), "a cancelled job left per-cell columns behind"
    assert xe._JOB_KIND_NAMES[kind] not in xe._running_jobs()
    # the tab shows "Cancelled" (not the last progress text) and the Run button is free again
    seen = [s["version"] for s in (xe._comp_ctl_state, xe._spill_state, xe._segqc_state, xe._coloc_state)]
    out = xe.poll_controls(1, *seen)
    i = names.index(kind)
    assert out[i] == "Cancelled" and out[4 + i] is False and out[12] is True
    # again, not held: completes and publishes
    monkeypatch.setattr(xe, hold, real)
    assert start()[:2] == (True, False)
    job2 = [j for j in xe._jobs.values() if j.kind == kind][-1]
    assert job2 is not job
    job2.thread.join(60)
    assert job2.state == "done" and state["status"] == "done", state["message"]
    assert state["result"] is not None and state["version"] == v0 + 1
    return state["result"]


def test_cancel_compartments_mid_run_publishes_nothing_and_a_rerun_completes(xe, D, labels, monkeypatch):
    table = xe._view_tables("xenium")[0]
    method = D["cluster_methods"][0] if D["cluster_methods"] else None
    start = lambda: xe.run_compartments(1, "celltype:celltypist", "auto", 20, 0, 4, 30, 2, 0.2, 20, "xenium", method)
    _cancel_mid_run(xe, monkeypatch, "compartments", "_compartments", start, xe._comp_ctl_state,
                    lambda: xe._ctl_available("xenium"))
    assert len(xe._ctl_get(table, "xenium", "edge_distance_um")) == len(D["df"])
    assert {"edge_distance_um", "local_density", "compartment"} <= set(xe._ctl_available("xenium"))


def test_cancel_spillover_mid_run_publishes_nothing_and_a_rerun_completes(xe, D, labels, monkeypatch, tmp_path):
    xe.EXTRA_DATASETS.extend([_fab_ds(D, 200, i, str(tmp_path / f"sp{i}")) for i in range(2)])
    gene = D["gene_names"][0]
    start = lambda: xe.run_spillover(1, "celltype:celltypist", "auto", gene, 5, 0.2, "xenium", None, "original")
    res = _cancel_mid_run(xe, monkeypatch, "spillover", "_source_geometry", start, xe._spill_state,
                          lambda: xe._ctl_available("xenium"))
    assert res["markers"] == [gene] and "spillover" in xe._ctl_available("xenium")


def test_cancel_segmentation_qc_mid_run_publishes_nothing_and_a_rerun_completes(xe, D, labels, monkeypatch):
    top = [D["gene_names"][i] for i in np.argsort(-np.asarray(D["expr"].sum(axis=0)).ravel())[:6]]
    sets = f"Alpha: {top[0]}, {top[1]}\nBeta: {top[2]}, {top[3]}\nGamma: {top[4]}, {top[5]}"
    start = lambda: xe.run_segqc(1, "celltypist", sets, "auto", 15.0)
    res = _cancel_mid_run(xe, monkeypatch, "segqc", "_qc_metrics", start, xe._segqc_state, lambda: [])
    assert len(res["table"]) >= 1 and res["table"].loc[0, "source"] == "Xenium"


def test_cancel_segmentation_qc_during_the_transcript_scan_writes_no_grid(xe, D, labels, monkeypatch):
    """The all-transcripts grid the QC job computes on demand is cancellable too, and a
    cancelled scan leaves no cache file."""
    path = xe._density_cache_file(xe._DENSITY_ALL)
    had = os.path.exists(path)
    keep = path + ".kept"
    if had:
        os.replace(path, keep)
    xe._density_mem.clear()
    seen = []

    def slow_batches(p, gene, qv_min=20):
        for i in range(400):
            seen.append(i)
            time.sleep(0.02)
            yield np.array([1.0], np.float32), np.array([1.0], np.float32), i / 400

    try:
        monkeypatch.setattr(xe, "_density_tx_batches_parquet", slow_batches)
        top = [D["gene_names"][i] for i in np.argsort(-np.asarray(D["expr"].sum(axis=0)).ravel())[:6]]
        sets = f"Alpha: {top[0]}, {top[1]}\nBeta: {top[2]}, {top[3]}\nGamma: {top[4]}, {top[5]}"
        assert xe.run_segqc(1, "celltypist", sets, "auto", 15.0)[:2] == (True, False)
        job = [j for j in xe._jobs.values() if j.kind == "segqc"][-1]
        while not seen and job.thread.is_alive():
            time.sleep(0.01)
        assert "counting all transcripts" in xe._segqc_state["message"]
        assert xe._job_cancel(job.id)
        job.thread.join(20)
        assert job.state == "cancelled" and len(seen) < 400
        assert xe._segqc_state["result"] is None and not os.path.exists(path)
    finally:
        xe._density_mem.clear()
        if had:
            os.replace(keep, path)


def test_cancel_colocalisation_mid_run_publishes_nothing_and_a_rerun_completes(xe, monkeypatch):
    _fake_view(xe, monkeypatch, 1, n=600)
    start = lambda: xe.run_coloc(1, "celltype:celltypist", 20.0, 20, "A", "xenium", None)
    res = _cancel_mid_run(xe, monkeypatch, "coloc", "_coloc_enrich", start, xe._coloc_state, lambda: [])
    assert res["n_perm"] == 20 and res["placebo"] is not None and len(res["summary"]) == 12


def test_fraction_of_transcripts_assigned_equals_a_hand_count(xe, raw, D, labels):
    """Segmentation QC computes the all-transcripts grid it needs itself; the column is
    Σ counts of the cells ÷ the Q ≥ 20 gene transcripts in the padded bounding box."""
    if raw.n_transcripts > 20_000_000:
        pytest.skip("all transcripts of a large bundle are not read into memory here")
    path = xe._density_cache_file(xe._DENSITY_ALL)
    if os.path.exists(path):
        os.remove(path)
    xe._density_mem.clear()
    top = [D["gene_names"][i] for i in np.argsort(-np.asarray(D["expr"].sum(axis=0)).ravel())[:6]]
    sets = f"Alpha: {top[0]}, {top[1]}\nBeta: {top[2]}, {top[3]}\nGamma: {top[4]}, {top[5]}"
    xe._run_segqc("celltypist", sets, "auto", 15.0)
    st = xe._segqc_state
    assert st["status"] == "done", st["message"]
    assert os.path.exists(path), "the QC job did not compute the all-transcripts grid"
    assert "Assigned = Σ counts of the cells ÷ gene transcripts" in st["message"] and "not available" not in st["message"]
    row = st["result"]["table"].iloc[0]
    assert row["source"] == "Xenium" and row["counts"] == "original"
    # by hand: the app's transcript filter, binned at 5 µm, summed over the bins the box touches
    names = list(D["gene_names"])
    tx = raw.transcripts(columns=["x_location", "y_location", "qv", "feature_name", "is_gene"])
    ok = tx["qv"].to_numpy(float) >= 20
    ok &= tx["is_gene"].to_numpy(bool) if "is_gene" in tx.columns else tx["feature_name"].isin(names).to_numpy()
    tx = tx[ok]
    df = D["df"]
    pad = float(np.sqrt(np.nanmedian(df["cell_area"].to_numpy(float)) / np.pi)) if "cell_area" in df.columns else 0.0
    bx = np.floor(tx["x_location"].to_numpy(np.float64) / 5.0)
    by = np.floor(tx["y_location"].to_numpy(np.float64) / 5.0)
    x, y = df["x_centroid"].to_numpy(float), df["y_centroid"].to_numpy(float)
    inside = ((bx >= np.floor((x.min() - pad) / 5.0)) & (bx <= np.floor((x.max() + pad) / 5.0))
              & (by >= np.floor((y.min() - pad) / 5.0)) & (by <= np.floor((y.max() + pad) / 5.0)))
    assert inside.sum() > 0
    want = float(D["expr"].sum()) / float(inside.sum())
    assert row["frac_assigned"] == pytest.approx(want, rel=1e-9) and 0 < want <= 1.0
    assert "frac_assigned" in xe._segqc_csv(st["result"]) and xe._segqc_records(st["result"])[0]["frac_assigned"] > 0
    # a second run reads the cached grid (no scan) and gives the same number
    def no_scan(*a, **k):
        raise AssertionError("the transcripts were scanned again")
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(xe, "_density_scan", no_scan)
        xe._run_segqc("celltypist", sets, "auto", 15.0)
    finally:
        mp.undo()
    assert xe._segqc_state["result"]["table"].iloc[0]["frac_assigned"] == pytest.approx(want, rel=1e-9)
    # a dataset without a transcript file: said, column empty
    mp = pytest.MonkeyPatch()
    try:
        mp.setitem(D, "data_dir", None)
        os.remove(path)
        xe._density_mem.clear()
        xe._run_segqc("celltypist", sets, "auto", 15.0)
    finally:
        mp.undo()
    assert "Fraction of transcripts assigned: not available (the dataset was loaded from a .zarr" in xe._segqc_state["message"]
    assert np.isnan(xe._segqc_state["result"]["table"].iloc[0]["frac_assigned"])


def test_save_as_spatialdata_writes_the_control_columns_and_they_read_back(xe, D, labels, expressed_gene, tmp_path):
    """An actual export with compartment and spillover columns present, read back with
    spatialdata: the ctl_* columns are in the table, per cell, with the values the app holds."""
    import spatialdata
    xe._run_compartments("xenium", None, "celltype:celltypist", "auto",
                         {"bin_um": 20, "radius": 30, "min_nb": 2, "min_frac": 0.2, "min_edge": 20})
    assert xe._comp_ctl_state["status"] == "done", xe._comp_ctl_state["message"]
    top = D["gene_names"][int(np.argmax(np.asarray(D["expr"].sum(axis=0)).ravel()))]
    xe._run_spillover("xenium", None, "celltype:celltypist", "auto", top, 5, False)
    assert xe._spill_state["status"] == "done", xe._spill_state["message"]
    table0 = xe._view_tables("xenium")[0]
    want = {c: xe._ctl_get(table0, "xenium", c) for c in xe._ctl_available("xenium")}
    assert {"edge_distance_um", "local_density", "dominant_fraction", "core", "compartment", "spillover",
            "spillover_flag"} <= set(want)
    n_cols_before = len(D["df"].columns)
    msg, poll_disabled, btn_disabled = xe.start_save_sdata(1, str(tmp_path), "with_controls", [], "xenium")
    assert (poll_disabled, btn_disabled) == (False, True), msg
    tl.wait_state(xe._save_sdata_state, xe._save_sdata_lock, timeout=600)
    assert xe._save_sdata_state["status"] == "done", xe._save_sdata_state["message"]
    path = os.path.join(str(tmp_path), "with_controls.zarr")
    obs = spatialdata.read_zarr(path).tables["table"].obs
    ids = obs["cell_id"].astype(str).to_numpy() if "cell_id" in obs.columns else obs.index.astype(str).to_numpy()
    assert list(ids) == list(D["df"].index.astype(str))
    for c, v in want.items():
        got = obs[f"ctl_{c}"]
        if c in ("compartment", "spillover_flag"):
            assert got.astype(str).tolist() == [str(x) for x in v], c
        else:
            np.testing.assert_allclose(got.to_numpy(float), np.asarray(v, float), rtol=1e-6, atol=1e-6, err_msg=c)
    assert set(obs["ctl_compartment"].astype(str)) <= {"core", "edge", "other"}
    assert len(D["df"].columns) == n_cols_before, "the export wrote the control columns into the app's cell table"
    # the app reads the exported store again, and the extra columns do not break its loader
    data = xe.load_from_sdata_zarr(path)
    assert list(data["df"].index.astype(str)) == list(D["df"].index.astype(str))
