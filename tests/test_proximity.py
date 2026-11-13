"""Proximity differential expression and contact-group pseudobulk differential expression.

Fabricated sections with planted effects: a gene of the target type that rises next to
the anchor type (must be found), a gene of the anchor type that bleeds into the adjacent
target cells (must be flagged), and genes with no relation to the anchor (must stay
quiet). The building blocks are checked against independent computations written here
(dense residuals by least squares, itertools enumeration, brute-force counting).
"""
from __future__ import annotations

import itertools
import threading

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.spatial import cKDTree

import xe_testlib as tl

T, X = "Target", "Anchor"


@pytest.fixture(autouse=True)
def _clean_state(xe):
    yield
    xe._controls_reset()
    xe._stats_reset()
    with xe._de_lock:
        xe._de_state.update({"status": "idle", "message": "", "result": None})


def _sample(counts, xy, genes, label):
    n = len(xy)
    df = pd.DataFrame({"x_centroid": xy[:, 0], "y_centroid": xy[:, 1]}, index=[f"{label}_{i}" for i in range(n)])
    return {"label": label, "name": label, "key": label, "df": df, "csc": sp.csc_matrix(counts.astype(np.float32)),
            "rows": np.arange(n), "gni": {g: i for i, g in enumerate(genes)}, "genes": list(genes),
            "x_offset": 0.0, "corrected": False, "ds": {}}


N_NULL, N_MARK = 30, 12
GENES = (["GRAD", "CONTACT"] + [f"xm{i}" for i in range(N_MARK)] + [f"om{i}" for i in range(N_MARK)]
         + [f"pm{i}" for i in range(N_MARK)] + [f"null{i}" for i in range(N_NULL)] + [f"hk{i}" for i in range(6)])


def _section(seed, n_t=1800, n_x=70, n_o=500, n_p=70, size=1500.0, grad=2.5, bleed=0.25, label="s"):
    """One section: Target cells everywhere; Anchor and Placebo cells scattered; Other
    cells; GRAD (a Target gene) rises within ~30 µm of an Anchor cell; CONTACT rises in
    Target cells with an Anchor within 50 µm; the Anchor markers xm* bleed into Target
    cells within 15 µm."""
    rng = np.random.default_rng(seed)
    n = n_t + n_x + n_o + n_p
    xy = rng.uniform(0, size, size=(n, 2))
    lab = np.array([T] * n_t + [X] * n_x + ["Other"] * n_o + ["Placebo"] * n_p, dtype=object)
    d = cKDTree(xy[lab == X]).query(xy)[0]
    gi = {g: i for i, g in enumerate(GENES)}
    lam = np.zeros((n, len(GENES)))
    for g in GENES:
        if g.startswith("null") or g.startswith("hk"):
            lam[:, gi[g]] = 2.0 if g.startswith("null") else 8.0
    is_t = lab == T
    lam[is_t, gi["GRAD"]] = 1.0 * (1.0 + grad * np.exp(-d[is_t] / 30.0))
    lam[is_t, gi["CONTACT"]] = 2.0 * np.where(d[is_t] <= 50.0, 2.5, 1.0)
    for i in range(N_MARK):
        lam[lab == X, gi[f"xm{i}"]] = 2.0
        lam[is_t, gi[f"xm{i}"]] = 0.04                       # ambient level
        lam[is_t & (d <= 15.0), gi[f"xm{i}"]] = 2.0 * bleed
        lam[lab == "Other", gi[f"om{i}"]] = 6.0
        lam[lab == "Placebo", gi[f"pm{i}"]] = 6.0
    counts = rng.poisson(lam)
    return _sample(counts, xy, GENES, label), lab, d


@pytest.fixture(scope="module")
def one():
    return _section(1, label="one")


@pytest.fixture(scope="module")
def four():
    return [_section(10 + i, n_t=1200, n_o=300, label=f"S{i}") for i in range(4)]


# ── building blocks against independent computations ─────────────────────────
def test_set_statistics_equal_dense_least_squares_residuals(xe):
    rng = np.random.default_rng(0)
    n, G = 400, 7
    Y = sp.random(n, G, density=0.4, random_state=1, data_rvs=lambda k: rng.gamma(2.0, 1.0, k)).tocsr()
    logd = rng.normal(5, 0.5, n)
    bins = xe._prox_quantile_bins(rng.random(n), 5)
    assert sorted(np.unique(bins)) == [0, 1, 2, 3, 4] and np.bincount(bins).min() >= 75
    D = xe._prox_design(logd, bins)
    assert D.shape == (n, 2 + 4)
    fit = xe._prox_fit(Y, D)
    Yd = Y.toarray()
    beta, *_ = np.linalg.lstsq(D, Yd, rcond=None)
    R = Yd - D @ beta
    assert np.allclose(fit["beta"], beta, atol=1e-8) and np.allclose(fit["rss"], (R ** 2).sum(0), rtol=1e-8)
    d1, d2 = rng.uniform(0, 300, n), rng.uniform(0, 300, n)
    keep2 = rng.random(n) > 0.1
    st = xe._prox_set_stats(Y, D, fit["beta"], [(d1, None), (d2, keep2)], near=25, far=100,
                            min_near=5, min_far=50, min_slope_n=50)
    for k, (d, keep) in enumerate([(d1, np.ones(n, bool)), (d2, keep2)]):
        Rk, dk = R[keep], d[keep]
        nf = Rk[dk <= 25].mean(0) - Rk[dk >= 100].mean(0)
        z = -np.log(dk + 1.0)
        slope = np.array([np.polyfit(z, Rk[:, g], 1)[0] for g in range(G)])
        assert np.allclose(st["nearfar"][k], nf, atol=1e-10)
        assert np.allclose(st["slope"][k], slope, atol=1e-10)
        assert st["n_near"][k] == (dk <= 25).sum() and st["n_far"][k] == (dk >= 100).sum() and st["n"][k] == keep.sum()
    # too few near cells: no near − far effect, the slope is still defined
    far_only = xe._prox_set_stats(Y, D, fit["beta"], [(d1 + 30.0, None)], near=25, far=100)
    assert np.isnan(far_only["nearfar"]).all() and np.isfinite(far_only["slope"]).all()
    # the free-length fit against a plain least-squares scan of the same grid
    grid = (5.0, 20.0, 60.0, 200.0)
    lf = xe._prox_length_fit(Y, D, fit, d1, grid)
    for g in range(G):
        best = None
        for L in grid:
            A = np.column_stack([np.ones(n), np.exp(-d1 / L)])
            coef, *_ = np.linalg.lstsq(A, R[:, g], rcond=None)
            sse = float(((R[:, g] - A @ coef) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, L, coef[1])
        assert lf["L"][g] == best[1] and lf["amplitude"][g] == pytest.approx(best[2], abs=1e-9)
        assert lf["r2"][g] == pytest.approx(1 - best[0] / float((R[:, g] ** 2).sum()), abs=1e-9)


def test_signflip_over_genes_equals_the_scalar_exact_test(xe):
    rng = np.random.default_rng(3)
    E = rng.normal(0.3, 1.0, size=(6, 40))
    E[2, :] = np.nan                                     # a unit without an effect is dropped
    r = xe._prox_signflip_genes(E)
    assert r["n"] == 5 and r["exact"] and r["min_p"] == 2 / 32
    for g in range(40):
        ref = xe._sl_signflip(E[[0, 1, 3, 4, 5], g])
        assert r["p"][g] == ref["p"] and r["mean"][g] == pytest.approx(ref["mean"])
    assert xe._prox_signflip_genes(E[:3]) is None        # 2 valid units: no test
    assert not hasattr(xe, "_PROX_MC") and not hasattr(xe, "_PROX_ENUM_UNITS")     # the shared limits are used
    mid = rng.normal(0.5, 1.0, size=(13, 3))                  # 13 units: enumerated (2^13), as stats does
    m = xe._prox_signflip_genes(mid)
    assert m["exact"] and m["n_vectors"] == 2 ** 13 and m["min_p"] == pytest.approx(2 / 2 ** 13)
    assert all(m["p"][g] == xe._sl_signflip(mid[:, g])["p"] for g in range(3))
    big = rng.normal(0.5, 1.0, size=(xe._SL_ENUM_UNITS + 2, 5))
    a, b = xe._prox_signflip_genes(big), xe._prox_signflip_genes(big)
    assert not a["exact"] and (a["p"] == b["p"]).all() and a["min_p"] == pytest.approx(1 / (xe._SL_MC + 1))
    assert a["n_vectors"] == xe._SL_MC == 20_000 and xe._SL_ENUM_UNITS == 16
    for g in range(5):
        assert abs(a["p"][g] - xe._sl_signflip(big[:, g])["p"]) < 0.03


def test_empirical_fdr_is_random_yield_over_observed_yield(xe):
    rng = np.random.default_rng(5)
    obs = np.concatenate([rng.normal(6, 1, 20), np.abs(rng.normal(0, 1, 200)), [np.nan]])
    rand = np.abs(rng.normal(0, 1, size=(25, len(obs))))
    q = xe._prox_empirical_fdr(obs, rand)
    assert np.isnan(q[-1]) and np.isfinite(q[:-1]).all()
    ok = np.isfinite(obs)
    raw = np.array([(1 + (rand[:, ok] >= t).sum()) / (25 * (obs[ok] >= t).sum()) for t in obs[ok]])
    order = np.argsort(obs[ok])
    mono = np.empty(ok.sum())
    mono[order] = np.minimum(np.minimum.accumulate(raw[order]), 1.0)
    assert np.allclose(q[ok], mono)
    assert (q[:20] < 0.05).all() and (q[20:-1] < 0.05).mean() < 0.05 and q[ok].min() > 0
    # a field with no signal: nothing is called
    null = np.abs(rng.normal(0, 1, 300))
    assert (xe._prox_empirical_fdr(null, np.abs(rng.normal(0, 1, size=(25, 300)))) < 0.05).sum() == 0


def test_random_anchors_are_count_matched_within_strata(xe):
    rng = np.random.default_rng(0)
    pool = np.arange(100, 1100)
    pbin = rng.integers(0, 4, size=1000)
    pbin[pbin == 3] = 2                                   # stratum 3 has no pool cell
    need = np.array([7, 0, 12, 4])
    draws, short = xe._prox_matched_random(pool, pbin, need, 6, np.random.default_rng(1))
    assert len(draws) == 6 and short == 6 * 4
    lookup = dict(zip(pool, pbin))
    for dr in draws:
        assert len(set(dr)) == len(dr) == 19
        assert np.bincount([lookup[i] for i in dr], minlength=4).tolist() == [7, 0, 12, 0]
    again, _ = xe._prox_matched_random(pool, pbin, need, 6, np.random.default_rng(1))
    assert all((a == b).all() for a, b in zip(draws, again))


# ── proximity DE: planted effects ────────────────────────────────────────────
def test_single_sample_gradient_is_found_and_the_bleed_gene_is_flagged(xe, one):
    s, lab, d = one
    res = xe._prox_compute([s], [lab], T, X, params={"n_random": 60, "fit_length": True}, source="Other")
    t = res["table"].set_index("gene")
    info = res["info"]
    assert res["kind"] == "proximity" and info["mode"] == "single" and info["n_units"] == 1
    assert info["min_p"] == pytest.approx(1 / 61)
    is_t = lab == T
    assert res["n_a"] == int((is_t & (d <= 25)).sum()) and res["n_b"] == int((is_t & (d >= 100)).sum())
    assert res["n_target"] == int(is_t.sum()) and res["n_anchor"] == int((lab == X).sum())
    # the planted Target gene: found, positive, not flagged, short length scale
    g = t.loc["GRAD"]
    assert g["log2fc"] > 0.3 and g["slope"] > 0 and g["pval"] == pytest.approx(1 / 61) and g["padj"] < 0.05
    assert not g["leak_flag"] and not g["anchor_owned"] and g["owner"] == T and g["z_random"] > 5
    assert g["L_um"] in (18.0, 25.0, 35.0, 50.0) and g["amplitude"] > 0 and g["r2"] > 0.01
    # "Gene ownership + leak gate" unticked: the same statistics, no owner, nothing flagged
    off = xe._prox_compute([s], [lab], T, X, params={"n_random": 60, "leak_gate": False}, source="Other")
    to = off["table"].set_index("gene")
    assert off["gate"]["off"] and not off["gate"]["applicable"] and "unticked" in off["gate"]["reason"]
    assert not to["leak_flag"].any() and not to["anchor_owned"].any() and not to["gate_flag"].any()
    assert (to["owner"] == "").all() and to["leak_fraction"].isna().all()
    assert np.array_equal(to["padj"].to_numpy(), t["padj"].to_numpy(), equal_nan=True)
    assert np.allclose(to["log2fc"], t["log2fc"]) and t["leak_flag"].any()
    # the Anchor's own genes bleed into adjacent Target cells: "significant", and flagged
    xm = t.loc[[f"xm{i}" for i in range(N_MARK)]]
    assert (xm["log2fc"] > 0).all() and (xm["padj"] < 0.05).sum() >= N_MARK - 2
    assert xm["anchor_owned"].all() and xm["leak_flag"].all() and (xm["owner"] == X).all()
    assert res["gate"]["applicable"] and xm["gate_flag"].sum() >= N_MARK - 2
    # genes unrelated to the anchor stay quiet
    null = t.loc[[x for x in t.index if x.startswith("null") or x.startswith("hk")]]
    assert (null["padj"] < 0.05).sum() <= 3 and null["padj"].min() > g["padj"] and not null["anchor_owned"].any()
    assert abs(null["log2fc"]).median() < 0.5 * g["log2fc"]
    # the other type of similar abundance is the placebo anchor; it does not reproduce GRAD
    assert res["placebos"] == ["Placebo"] and bool(g["specific"])
    # markers of types absent from the Target cells are not tested (detection floor)
    assert "om0" not in t.index and res["n_genes"] < len(GENES)


def test_near_far_effect_equals_an_independent_dense_computation(xe, one):
    s, lab, d = one
    res = xe._prox_compute([s], [lab], T, X, params={"n_random": 3}, source=None)
    is_t = lab == T
    M = s["csc"].toarray()[is_t]
    lib = M.sum(1)
    Y = np.log1p(M / lib[:, None] * 1e4)
    D = np.column_stack([np.ones(is_t.sum()), np.log1p(lib)])
    R = Y - D @ np.linalg.lstsq(D, Y, rcond=None)[0]
    dt = d[is_t]
    want = (R[dt <= 25].mean(0) - R[dt >= 100].mean(0)) / np.log(2)
    t = res["table"].set_index("gene")
    for gname in ("GRAD", "null3", "hk1", "xm0"):
        assert t.loc[gname, "log2fc"] == pytest.approx(want[GENES.index(gname)], abs=1e-6)


def test_random_anchors_remove_a_gradient_that_is_only_location(xe):
    """A gene that follows the tissue edge, with anchors that sit at the edge: near − far
    is large, but anchors drawn in the same edge stratum reproduce it."""
    rng = np.random.default_rng(4)
    n_t, n_x, n_o = 2500, 80, 400
    xy = rng.uniform(0, 1500, size=(n_t + n_x + n_o, 2))
    xy[n_t:n_t + n_x, 0] = rng.uniform(0, 150, n_x)                      # anchors in the left margin
    lab = np.array([T] * n_t + [X] * n_x + ["Other"] * n_o, dtype=object)
    genes = ["EDGE"] + [f"n{i}" for i in range(20)]
    lam = np.full((len(xy), len(genes)), 3.0)
    lam[:, 0] = 1.0 + 6.0 * np.exp(-xy[:, 0] / 150.0)
    s = _sample(rng.poisson(lam), xy, genes, "edge")
    strata = [(xy[:, 0] // 150).astype(int)]
    with_strata = xe._prox_compute([s], [lab], T, X, strata=strata, params={"n_random": 40}, source=None)
    no_strata = xe._prox_compute([s], [lab], T, X, strata=None, params={"n_random": 40}, source=None)
    a, b = with_strata["table"].set_index("gene").loc["EDGE"], no_strata["table"].set_index("gene").loc["EDGE"]
    assert a["log2fc"] == pytest.approx(b["log2fc"]) and a["log2fc"] > 0.5
    assert b["padj"] < 0.05 and b["pval"] == pytest.approx(1 / 41)       # unmatched anchors: "found"
    assert a["pval"] > 0.1 and a["padj"] > 0.2                           # matched anchors: location explains it
    assert abs(a["random_mean"] - a["log2fc"]) < 0.25 * a["log2fc"]


def test_cohort_uses_the_exact_sign_flip_across_samples(xe, four):
    samples, labels = [f[0] for f in four], [f[1] for f in four]
    res = xe._prox_compute(samples, labels, T, X, params={"n_random": 12}, source="Other")
    info, t = res["info"], res["table"].set_index("gene")
    assert info["mode"] == "cohort" and info["n_units"] == 4 and info["exact"] and info["min_p"] == 2 / 16
    assert not info["by_donor"] and res["n_samples"] == 4
    assert t.loc["GRAD", "n_pos"] == 4 and t.loc["GRAD", "pval"] == 2 / 16
    assert not t.loc["GRAD", "leak_flag"] and t.loc["xm0", "leak_flag"]
    assert t["padj"].between(0, 1).all() and t.loc["GRAD", "padj"] <= t["padj"].median()
    # two samples of one donor are one unit: three units, the floor moves to 2/8
    donors = ["d1", "d1", "d2", "d3"]
    by_donor = xe._prox_compute(samples, labels, T, X, params={"n_random": 4}, source="Other",
                                conds=["A", "A", "A", "A"], donors=donors)
    assert by_donor["info"]["n_units"] == 3 and by_donor["info"]["by_donor"]
    assert by_donor["table"].set_index("gene").loc["GRAD", "pval"] == 2 / 8
    assert "n = 3 donors" in xe._prox_units_statement(by_donor)
    # fewer than 3 units: the matched-anchor null, stated as such
    two = xe._prox_compute(samples[:2], labels[:2], T, X, params={"n_random": 20}, source="Other")
    assert two["info"]["mode"] == "single" and two["info"]["n_units"] == 2
    assert "matched random anchor sets" in xe._prox_units_statement(two)


def test_proximity_refuses_what_it_cannot_test(xe, one):
    s, lab, _d = one
    with pytest.raises(ValueError, match="two different types"):
        xe._prox_compute([s], [lab], T, T)
    with pytest.raises(ValueError, match="No cell of the view is labelled"):
        xe._prox_compute([s], [lab], T, "Nope")
    with pytest.raises(ValueError, match="near distance"):
        xe._prox_compute([s], [lab], T, X, params={"near": 100, "far": 50})
    few = lab.copy()
    few[np.flatnonzero(lab == X)[3:]] = "Other"
    with pytest.raises(ValueError, match="No sample can be tested"):
        xe._prox_compute([s], [few], T, X, params={"n_random": 2})


# ── contact groups ───────────────────────────────────────────────────────────
def test_contact_matching_reproduces_the_contact_groups_stratum_mix(xe):
    strata = np.array([0] * 10 + [1] * 10 + [2] * 6 + [3] * 4)
    a = np.zeros(30, bool)
    a[[0, 1, 2, 3, 10, 11, 20, 26, 27]] = True       # strata 0: 4, 1: 2, 2: 1, 3: 2
    c = ~a
    c[[28, 29]] = False                               # stratum 3 has no comparator
    keep_a, w, info = xe._contact_match(a, c, strata)
    assert info == {"n_a_all": 9, "n_c_all": 19, "n_a": 7, "n_c": 19, "n_a_dropped": 2, "n_strata": 3}
    assert keep_a.sum() == 7 and not keep_a[[26, 27]].any()
    assert w.sum() == pytest.approx(19) and (w[~c] == 0).all()
    for st, n_a in ((0, 4), (1, 2), (2, 1)):
        m = c & (strata == st)
        assert w[m].sum() / w.sum() == pytest.approx(n_a / 7)        # weighted comparator mix = contact mix
        assert np.allclose(w[m], (n_a / 7) / (m.sum() / 19))
    none, w0, i0 = xe._contact_match(a, np.zeros(30, bool), strata)
    assert not none.any() and not w0.any() and i0["n_a"] == 0


def _paired_fit(prep):
    """Stand-in for the R fit: paired t-test of the log2 CPM ratio contact / comparator
    across the samples (the default tier runs without R)."""
    from scipy import stats as sst
    c = prep["counts"].astype(float)
    cpm = 1e6 * (c + 0.5) / (c.sum(0, keepdims=True) + 1.0)
    a = np.array([x == "A" for x in prep["cond"]])
    assert prep["batch"][0::2] == prep["batch"][1::2] and prep["cond"][:2] == ["B", "A"]
    d = np.log2(cpm[:, a]) - np.log2(cpm[:, ~a])
    tt = sst.ttest_1samp(d, 0.0, axis=1)
    p = np.nan_to_num(tt.pvalue, nan=1.0)
    order = np.argsort(p)
    adj = np.empty(len(p))
    adj[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    table = pd.DataFrame({"gene": prep["genes"], "log2fc": d.mean(1), "base_mean": c.mean(1),
                          "mean_a": c[:, a].mean(1), "mean_b": c[:, ~a].mean(1), "stat": tt.statistic,
                          "pval": p, "padj": np.minimum(adj, 1.0)})
    table["neg_log10_padj"] = -np.log10(np.maximum(table["padj"], 1e-300))
    return {"table": table, "engine": "stub", "method_label": "paired t-test (stub)", "shrink": "none",
            "stat_name": "t", "mean_unit": "counts", "size_factors": np.ones(c.shape[1]), "notes": []}


def test_contact_effect_is_recovered_and_absent_under_shuffled_labels(xe, four):
    samples, labels = [f[0] for f in four], [f[1] for f in four]
    res = xe._contact_compute(samples, labels, T, X, radius=50, min_cells=20, n_shuffle=19, fit=_paired_fit)
    t = res["table"].set_index("gene")
    assert res["kind"] == "contact" and res["blocks"] == ["S0", "S1", "S2", "S3"] and not res["by_donor"]
    # the groups are what a brute-force distance count gives
    for (s, lab, d), row in zip(four, res["samples"]):
        is_t = lab == T
        assert row["n_a_all"] == int((is_t & (d <= 50)).sum()) and row["n_c_all"] == int((is_t & (d > 50)).sum())
        assert row["testable"] and row["n_a"] + row["n_a_dropped"] == row["n_a_all"]
    c = t.loc["CONTACT"]
    assert c["log2fc"] == pytest.approx(np.log2(2.5), abs=0.25) and c["padj"] < 0.05 and bool(c["pass"])
    assert not c["leak_flag"] and c["owner"] == T
    # unticked: nothing is flagged, so "pass" is FDR and effect size alone
    off = xe._contact_compute(samples, labels, T, X, radius=50, min_cells=20, n_shuffle=0, fit=_paired_fit,
                              leak_gate=False)
    to = off["table"].set_index("gene")
    assert off["gate"]["off"] and not to["leak_flag"].any() and (to["owner"] == "").all()
    sig = (to["padj"] < off["fdr"]) & (to["log2fc"].abs() > off["lfc_min"])
    assert (to["pass"] == sig).all() and to["pass"].sum() > t["pass"].sum()
    # the anchor's genes differ between the groups as well, and are flagged, never passed
    xm = t.loc[[f"xm{i}" for i in range(N_MARK)]]
    assert (xm["log2fc"] > 0.4).all() and xm["leak_flag"].all() and not xm["pass"].any()
    null = t.loc[[g for g in t.index if g.startswith("null") or g.startswith("hk")]]
    assert abs(null["log2fc"]).max() < 0.3
    sh = res["shuffle"]
    assert sh["n"] == 19 and sh["observed"] == res["n_pass"] >= 1
    assert sh["null_max"] < sh["observed"] and sh["p"] == pytest.approx(1 / 20) and sh["null_mean"] < 0.5
    # the same data with the contact labels shuffled across the Target cells: nothing passes
    rng = np.random.default_rng(0)
    scrambled = []
    for s, lab, _d in four:
        xy = s["df"][["x_centroid", "y_centroid"]].to_numpy().copy()
        ix = np.flatnonzero(lab == T)
        cnt = s["csc"].toarray()
        cnt[ix] = cnt[rng.permutation(ix)]             # expression no longer follows position
        scrambled.append(_sample(cnt, xy, GENES, s["label"]))
    flat = xe._contact_compute(scrambled, labels, T, X, radius=50, n_shuffle=0, fit=_paired_fit)
    ft = flat["table"].set_index("gene")
    assert not ft.loc["CONTACT", "pass"] and abs(ft.loc["CONTACT", "log2fc"]) < 0.2 and flat["n_pass"] == 0
    assert flat["shuffle"]["n"] == 0 and np.isnan(flat["shuffle"]["p"])


def test_contact_pseudobulk_is_the_weighted_sum_of_counts(xe, four):
    samples, labels = [f[0] for f in four[:2]], [f[1] for f in four[:2]]
    strata = [(s["df"]["x_centroid"].to_numpy() // 500).astype(int) for s in samples]
    seen = {}

    def fit(prep):
        seen["prep"] = prep
        return _paired_fit(prep)

    res = xe._contact_compute(samples, labels, T, X, strata=strata, radius=50, min_cells=5, min_count=0,
                              n_shuffle=0, fit=fit)
    prep = seen["prep"]
    assert prep["labels"] == ["S0_comparator", "S0_contact", "S1_comparator", "S1_contact"]
    assert prep["cond"] == ["B", "A", "B", "A"] and prep["batch"] == ["S0", "S0", "S1", "S1"]
    grp = xe._contact_groups(samples, labels, T, X, strata, 50, 5)
    for b, (s, g) in enumerate(zip(samples, grp)):
        M = s["csc"].toarray().astype(float)
        want_c = np.rint((g["w"][:, None] * M).sum(0))
        want_a = M[g["a"]].sum(0)
        cols = [GENES.index(x) for x in prep["genes"]]
        assert (prep["counts"][:, 2 * b] == want_c[cols]).all() and (prep["counts"][:, 2 * b + 1] == want_a[cols]).all()
        # the weighted comparator has the contact group's stratum mix
        for st in np.unique(strata[b][g["a"]]):
            assert g["w"][strata[b] == st].sum() / g["w"].sum() == pytest.approx((g["a"] & (strata[b] == st)).sum() / g["a"].sum())
    assert res["n_a"] == sum(int(g["a"].sum()) for g in grp)


def test_contact_refuses_fewer_than_two_usable_samples(xe, four):
    samples, labels = [f[0] for f in four], [f[1] for f in four]
    with pytest.raises(ValueError, match="at least 2 samples") as e:
        xe._contact_compute(samples[:1], labels[:1], T, X, fit=_paired_fit, n_shuffle=0)
    assert "Proximity to a cell type" in str(e.value) and "1 sample(s) qualify" in str(e.value)
    no_anchor = [np.where(l == X, "Other", l) for l in labels[1:]]
    with pytest.raises(ValueError, match="1 sample\\(s\\) qualify"):
        xe._contact_compute(samples, [labels[0]] + no_anchor, T, X, fit=_paired_fit, n_shuffle=0)
    with pytest.raises(ValueError, match="one donor"):
        xe._contact_compute(samples[:2], labels[:2], T, X, donors=["d", "d"], fit=_paired_fit, n_shuffle=0)
    # samples of one donor are summed into one block
    res = xe._contact_compute(samples, labels, T, X, donors=["d1", "d1", "d2", "d3"], fit=_paired_fit, n_shuffle=0)
    assert res["blocks"] == ["d1", "d2", "d3"] and res["by_donor"]


# ── the jobs: callbacks, guards, cancel ──────────────────────────────────────
def _patch_view(xe, monkeypatch, data):
    samples, labels = [f[0] for f in data], [f[1] for f in data]
    monkeypatch.setattr(xe, "_an_samples", lambda *a, **k: samples)
    monkeypatch.setattr(xe, "_comp_labels", lambda *a, **k: (labels, "cell type (test)"))
    monkeypatch.setattr(xe, "_prox_sample_strata",
                        lambda s, seg: (np.zeros(len(s["df"]), int), np.zeros(len(s["df"]), int), False))


def _run(xe, mode, **kw):
    args = dict(prox_labels="celltype:test", prox_target=T, prox_anchor=X, prox_near=25, prox_far=100,
                prox_nrand=8, prox_source="none", prox_length=[], contact_radius=50, contact_min_cells=20,
                contact_nshuffle=3)
    args.update(kw)
    return xe.run_de(1, None, None, "xenium", None, None, "original", mode, **args)


def test_guards_and_mode_switch(xe):
    import dash
    with pytest.raises(dash.exceptions.PreventUpdate):
        _run(xe, "proximity") if False else xe.run_de(0, None, None, "xenium", None, None, "original", "proximity")
    with pytest.raises(dash.exceptions.PreventUpdate):
        xe.open_design_modal(0, 0, 0, 0)
    assert xe.open_design_modal(None, None, None, 1)[0] is True
    assert "Pick where the cell types" in _run(xe, "proximity", prox_labels=None)[2]
    assert "Pick the cells" in _run(xe, "contact", prox_anchor=None)[2]
    assert "same cell type" in _run(xe, "proximity", prox_anchor=T)[2]
    assert "near distance" in _run(xe, "proximity", prox_near=200)[2]
    hidden = lambda st: st.get("display") == "none"
    cells, pb, note, prox, pp, cp = xe.toggle_de_mode("proximity")
    assert hidden(cells) and hidden(pb) and not hidden(prox) and not hidden(pp) and hidden(cp)
    assert note == xe._PROX_NOTE
    cells, pb, note, prox, pp, cp = xe.toggle_de_mode("contact")
    assert not hidden(prox) and hidden(pp) and not hidden(cp) and note == xe._CONTACT_NOTE
    cells, pb, note, prox, pp, cp = xe.toggle_de_mode("cells")
    assert not hidden(cells) and hidden(prox)
    with pytest.raises(dash.exceptions.PreventUpdate):
        xe.update_prox_label_options("tab-de", "cells", 0, 0, "xenium", 0)
    assert xe.update_prox_type_options(None, 0, 0, 0, "xenium", None)[0] == []


def test_proximity_job_publishes_into_the_de_tab(xe, monkeypatch, one):
    _patch_view(xe, monkeypatch, [one])
    assert _run(xe, "proximity", prox_nrand=20)[:2] == (True, False)
    tl.wait_state(xe._de_state, xe._de_lock)
    with xe._de_lock:
        res, msg = xe._de_state["result"], xe._de_state["message"]
    assert res["kind"] == "proximity" and "SINGLE-SAMPLE RESULT" in msg and "Unit of the p-values: n = " in msg
    assert "matched random anchor sets" in msg and "smallest achievable p 0.0476" in msg
    job = [j for j in xe._jobs_list() if j.kind == "proximity"][-1]
    job.thread.join(30)
    assert job.state == "done"
    cols = [c["id"] for c in xe._de_table_columns(res)]
    assert cols[:2] == ["gene", "log2fc"] and "padj" in cols and "leak_flag" in cols and "L_um" not in cols
    recs = xe._de_table_records(res)
    by = {r["gene"]: r for r in recs}
    assert by["GRAD"]["leak_flag"] == "ok" and by["xm0"]["leak_flag"] == "FLAG" and by["GRAD"]["specific"] == "yes"
    fig = xe.make_de_volcano_fig()
    names = [tr.name for tr in fig.data]
    assert any(n.startswith("leak-gate flagged") for n in names) and "near − far" in fig.layout.xaxis.title.text
    flagged = next(tr for tr in fig.data if tr.name.startswith("leak-gate flagged"))
    assert "xm0" in list(flagged.text) and "GRAD" not in list(flagged.text)
    csv = xe.download_de_table(1)
    assert csv["filename"] == "proximity_Target_near_Anchor.csv" and "# unit of the p-values: n = " in csv["content"]
    assert "anchor_owned" in csv["content"].split("\n", 12)[-1].split("\n")[0] or "anchor_owned" in csv["content"]
    assert len(xe._prox_samples_view(res)) == 1
    # the "Gene ownership + leak gate" checkbox of the tab reaches the job
    assert _run(xe, "proximity", prox_nrand=20, leak_gate=[])[:2] == (True, False)
    tl.wait_state(xe._de_state, xe._de_lock)
    with xe._de_lock:
        res2, msg2 = xe._de_state["result"], xe._de_state["message"]
    [j for j in xe._jobs_list() if j.kind == "proximity"][-1].thread.join(30)
    assert res2["gate"]["off"] and "switched off" in msg2 and not res2["table"]["leak_flag"].any()
    assert "genes flagged (" in msg and "by the change rule" in msg


def test_contact_job_publishes_and_refuses_a_single_sample(xe, monkeypatch, four):
    monkeypatch.setattr(xe, "_contact_pb_fit", lambda prep, engine="auto": _paired_fit(prep))
    _patch_view(xe, monkeypatch, four[:1])
    _run(xe, "contact")
    with pytest.raises(AssertionError):
        tl.wait_state(xe._de_state, xe._de_lock)
    with xe._de_lock:
        assert xe._de_state["status"] == "error" and "at least 2 samples" in xe._de_state["message"]
        assert "Proximity to a cell type" in xe._de_state["message"] and xe._de_state["result"] is None
    _patch_view(xe, monkeypatch, four)
    _run(xe, "contact", contact_nshuffle=5)
    tl.wait_state(xe._de_state, xe._de_lock)
    with xe._de_lock:
        res, msg = xe._de_state["result"], xe._de_state["message"]
    assert res["kind"] == "contact" and "Unit of the p-values: n = 4 samples" in msg
    assert "Label shuffle within sample × stratum (5 shuffles" in msg and "P = 0.167" in msg
    cols = [c["id"] for c in xe._de_table_columns(res)]
    assert {"gene", "log2fc", "stat", "padj", "owner", "leak_flag"} <= set(cols)
    fig = xe.make_de_volcano_fig()
    assert "contact / comparator" in fig.layout.xaxis.title.text
    csv = xe.download_de_table(1)
    assert csv["filename"] == "contact_Target_with_Anchor.csv" and "# label shuffle" in csv["content"]


@pytest.mark.parametrize("mode", ["proximity", "contact"])
def test_cancel_mid_run_publishes_nothing_and_a_rerun_completes(xe, monkeypatch, four, mode):
    _patch_view(xe, monkeypatch, four)
    started, release = threading.Event(), threading.Event()
    if mode == "proximity":
        real = xe._prox_sample

        def slow(*a, **k):
            started.set()
            release.wait(30)
            xe._job_check()
            return real(*a, **k)
        monkeypatch.setattr(xe, "_prox_sample", slow)
    else:
        def slow_fit(prep, engine="auto"):
            started.set()
            release.wait(30)
            return _paired_fit(prep)
        monkeypatch.setattr(xe, "_contact_pb_fit", slow_fit)
    assert _run(xe, mode)[:2] == (True, False)
    assert started.wait(30)
    assert "already in progress" in _run(xe, mode)[2]
    job = [j for j in xe._jobs_list() if j.kind == mode][-1]
    assert xe._job_cancel(job.id)
    release.set()
    job.thread.join(60)
    assert job.state == "cancelled"
    with xe._de_lock:
        assert xe._de_state["result"] is None and xe._de_state["status"] == "error"
        assert xe._de_state["message"] == "Cancelled"
    assert xe.poll_de(0, 0)[0] == "Cancelled"
    # rerun: completes
    started.clear()
    _run(xe, mode, contact_nshuffle=2)
    tl.wait_state(xe._de_state, xe._de_lock)
    with xe._de_lock:
        assert xe._de_state["result"]["kind"] == mode
    again = [j for j in xe._jobs_list() if j.kind == mode][-1]
    again.thread.join(30)
    assert again is not job and again.state == "done"


def test_no_neighbour_search_crosses_samples(xe, four):
    """Two samples at the same coordinates: the anchors of one must not be near the cells
    of the other."""
    (s0, l0, d0), (s1, l1, _d1) = four[0], four[1]
    lone = np.where(l1 == X, "Other", l1)                 # sample 1 has no anchor at all
    grp = xe._contact_groups([s0, s1], [l0, lone], T, X, None, 50, 20)
    assert grp[1]["n_a_all"] == 0 and grp[0]["n_a_all"] == int(((l0 == T) & (d0 <= 50)).sum())
    res = xe._prox_compute([s0, s1], [l0, lone], T, X, params={"n_random": 3}, source=None)
    assert [r["used"] for r in res["samples"]] == [True, False] and res["n_samples"] == 1
    assert res["n_a"] == int(((l0 == T) & (d0 <= 25)).sum())


@pytest.mark.tools
def test_contact_fit_in_r(xe, four):
    pytest.importorskip("rpy2")
    samples, labels = [f[0] for f in four], [f[1] for f in four]
    import rpy2.robjects as ro
    import rpy2.robjects.conversion as conversion
    conversion.set_conversion(ro.default_converter)
    try:
        res = xe._contact_compute(samples, labels, T, X, n_shuffle=2)
    except ValueError as exc:
        pytest.skip(str(exc)[:200])
    t = res["table"].set_index("gene")
    assert t.loc["CONTACT", "padj"] < 0.05 and t.loc["CONTACT", "log2fc"] == pytest.approx(np.log2(2.5), abs=0.3)
    assert res["fit"]["engine"] in ("deseq2", "edger") and res["shuffle"]["n"] == 2


def test_estimator_test_handles_unusable_random_sets(xe):
    rng = np.random.default_rng(2)
    U = rng.normal(0.5, 1.0, size=(5, 30))
    R = rng.normal(0.0, 1.0, size=(10, 5, 30))
    full = xe._prox_test(U, R, "cohort")
    assert full["fdr"] == "empirical" and full["n_valid"] == 10 and full["n_units"] == 5 and full["min_p"] == 2 / 32
    assert full["p"][3] == xe._sl_signflip(U[:, 3])["p"]
    R2 = R.copy()
    R2[:8, 2, :] = np.nan                                 # 8 of 10 sets lack an effect in one unit
    few = xe._prox_test(U, R2, "cohort")
    assert few["fdr"] == "bh" and few["n_valid"] == 2 and np.allclose(few["padj"], xe._bh(few["p"]))
    single = xe._prox_test(U[:1], R[:, :1], "single")
    assert single["fdr"] == "empirical" and single["min_p"] == pytest.approx(1 / 11) and single["n_units"] == 1
    mu = R[:, 0].mean(0)
    want = (1 + (np.abs(R[:, 0] - mu) >= np.abs(U[0] - mu) - 1e-12).sum(0)) / 11
    assert np.allclose(single["p"], want)
    none = xe._prox_test(U[:1], R2[:, 2:3], "single")
    assert none["fdr"] == "none" and np.isnan(none["p"]).all()
    # a unit without an observed effect is dropped before anything else
    U3 = U.copy()
    U3[4] = np.nan
    assert xe._prox_test(U3, R, "cohort")["n_units"] == 4


# ── review round 4 ───────────────────────────────────────────────────────────
def test_auto_density_covariate_is_never_the_anchor_or_the_tested_type(xe, one):
    prof = {"types": ["a", "b", "c"], "depth": np.array([5.0, 9.0, 1.0])}
    assert xe._auto_source(prof) == "b" and xe._auto_source(prof, exclude=("b",)) == "a"
    assert xe._auto_source(prof, exclude=("a", "b", "c")) is None and xe._auto_source({"types": [], "depth": []}) is None
    s, lab, _d = one
    # a section whose anchor type carries the most transcripts: "auto" used to pick it
    counts = s["csc"].toarray()
    counts[lab == X] *= 400
    heavy = _sample(counts, s["df"][["x_centroid", "y_centroid"]].to_numpy(), GENES, "heavy")
    labels = [lab]
    assert xe._auto_source(xe._type_profiles([heavy], labels, GENES)) == X
    res = xe._prox_compute([heavy], labels, T, X, params={"n_random": 8}, source="auto")
    assert res["source"] not in (T, X, None)
    with pytest.raises(ValueError, match="cannot be the anchor type"):
        xe._prox_compute([heavy], labels, T, X, params={"n_random": 8}, source=X)


def test_sample_without_a_condition_is_not_counted_in_the_result(xe, four):
    samples, labels = [f[0] for f in four], [f[1] for f in four]
    res = xe._prox_compute(samples, labels, T, X, params={"n_random": 6}, source="Other",
                           conds=["A", "B", "A", ""], donors=["", "", "", ""])
    rows = res["samples"]
    assert [r["used"] for r in rows] == [True, True, True, False]
    assert "no condition in the design" in rows[3]["note"]
    assert res["info"]["n_units"] == 3 and res["n_samples"] == 3
    assert res["n_target"] == sum(r["n_target"] for r in rows[:3])
    assert res["n_a"] == sum(r["n_near"] for r in rows[:3]) and res["n_b"] == sum(r["n_far"] for r in rows[:3])
    ref = xe._prox_compute(samples[:3], labels[:3], T, X, params={"n_random": 6}, source="Other",
                           conds=["A", "B", "A"], donors=["", "", ""])
    assert (res["n_a"], res["n_b"], res["n_target"], res["n_anchor"]) == (
        ref["n_a"], ref["n_b"], ref["n_target"], ref["n_anchor"])
    a, b = res["table"].set_index("gene"), ref["table"].set_index("gene")
    np.testing.assert_allclose(a.loc[b.index, "log2fc"], b["log2fc"])
    for col in ("leak_fraction", "leak_flag"):                       # the leak gate saw the same cells
        if col in b.columns:
            pd.testing.assert_series_equal(a.loc[b.index, col], b[col], check_names=False)


def test_type_pickers_follow_the_segmentation_cluster_set_and_samples(xe, D):
    from dash import no_update
    cb = next(v for k, v in xe.app.callback_map.items() if "prox-target.options" in k)
    ins = {(i["id"], i["property"]) for i in cb["inputs"]}
    assert {("seg-source", "value"), ("cluster-method", "value"), ("extra-datasets-version", "data"),
            ("prox-labels", "value")} <= ins
    method = (D.get("cluster_methods") or [None])[0]
    if not method:
        pytest.skip("the bundle has no cluster set")
    opts, _o2, src_opts, tv, av, sv = xe.update_prox_type_options("cluster", 0, 0, 0, "xenium", method, 0,
                                                                  None, None, "auto")
    have = [o["value"] for o in opts]
    assert have and (tv, av, sv) == (no_update, no_update, no_update)
    assert [o["value"] for o in src_opts][:2] == ["auto", "none"]
    # a pick that exists stays; one from another cluster set / segmentation is cleared
    out = xe.update_prox_type_options("cluster", 0, 0, 0, "xenium", method, 0, have[0], "no-such-type", "gone")
    assert out[3] is no_update and out[4] is None and out[5] == "auto"
