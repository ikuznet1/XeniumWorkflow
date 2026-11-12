"""Ligand-receptor exposure decomposition (xenium_app/lrdecomp.py): the core sums against
brute force (the unit test of the analysis the feature is ported from), the exact
identity exposure = K × R × L × γ × π, planted changes attributed to their factor, the
observed / expected variant, the tests across units, and the tab as a job.

Everything runs on fabricated cells; nothing here depends on the bundle beyond its gene
names (the job test puts fabricated counts on them).
"""

from __future__ import annotations

import io
import threading

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.spatial import cKDTree

import xe_testlib as tl

TERMS = ["contact", "receptor_expr", "ligand_expr", "regional", "preference"]


@pytest.fixture(autouse=True)
def _clean_lrd_state(xe):
    yield
    xe._lrd_reset()
    xe._stats_reset()


def _hand_sample(counts, xy, genes, label="hand"):
    n = len(counts)
    df = pd.DataFrame({"x_centroid": xy[:, 0], "y_centroid": xy[:, 1]}, index=[f"{label}_{i}" for i in range(n)])
    return {"label": label, "name": label, "key": label, "df": df, "csc": sp.csc_matrix(counts.astype(np.float64)),
            "rows": np.arange(n), "gni": {x: i for i, x in enumerate(genes)}, "genes": list(genes),
            "x_offset": 0.0, "corrected": False, "ds": {}}


GENES = ["L1", "R1", "L2", "R2"] + [f"bg{i}" for i in range(12)]
PAIRS = [("L1", "R1"), ("L2", "R2")]


def _world(seed, n=540, side=400.0, lig=4.0, rec=4.0, fractional=False):
    """Three types S / T / U mixed uniformly in a square. L1 is made by S, R1 by T; L2 / R2
    and the background genes by every cell."""
    rng = np.random.default_rng(seed)
    types = np.array(["S", "T", "U"])[np.arange(n) % 3]
    xy = rng.uniform(0, side, (n, 2))
    lam = np.full((n, len(GENES)), 2.0)
    lam[:, 0] = np.where(types == "S", lig, 0.1)
    lam[:, 1] = np.where(types == "T", rec, 0.1)
    c = rng.poisson(lam).astype(np.float64)
    if fractional:
        c = c * rng.uniform(0.3, 1.0, c.shape)
    return c, xy, types


# ── the core sums: the original's unit test, on the ported functions ─────────
def test_core_sums_equal_brute_force_and_the_identity_is_exact(xe):
    rng = np.random.default_rng(0)
    n = 400
    xy = rng.uniform(0, 600, (n, 2))
    punch = np.where(xy[:, 0] < 300, "p1", "p2")
    lin = rng.choice(["S", "T"], n)
    s = rng.uniform(20, 400, n)
    X = rng.poisson(0.3, (n, 3)) * rng.uniform(0.2, 1, (n, 3))            # fractional counts, 3 genes
    prs = [(i, j) for i, j in cKDTree(xy).query_pairs(40) if punch[i] == punch[j]]
    A = np.zeros((n, n))
    for i, j in prs:
        A[i, j] = A[j, i] = 1
    si, ti = np.flatnonzero(lin == "S"), np.flatnonzero(lin == "T")
    AS = sp.csr_matrix(A[np.ix_(ti, si)])
    for tile, binary in ((150.0, False), (None, False), (150.0, True)):
        Xb = (X >= 1).astype(float) if binary else X
        w = np.ones(n) if binary else s
        Lt = xe._lrd_tile_rates(xy[si], punch[si], Xb[si][:, [0, 1]], s[si], tile, binary)
        Rt = xe._lrd_tile_rates(xy[ti], punch[ti], Xb[ti][:, [2]], s[ti], tile, binary)
        nP, nE, sw = xe._lrd_pair_sums(AS, Xb[ti][:, [2]], Xb[si][:, [0, 1]], w[ti], w[si], Rt, Lt)

        def key(k):
            return (punch[k], None if tile is None else (int(xy[k, 0] // tile), int(xy[k, 1] // tile)))

        bP, bE, bw = np.zeros((1, 2)), np.zeros((1, 2)), 0.0
        for i in ti:
            for j in si:
                if A[i, j] == 0:
                    continue
                ri = [k for k in ti if key(k) == key(i)]
                rj = [k for k in si if key(k) == key(j)]
                rate_i = Xb[ri, 2].sum() / (len(ri) if binary else s[ri].sum())
                rate_j = Xb[rj][:, [0, 1]].sum(0) / (len(rj) if binary else s[rj].sum())
                bP[0] += Xb[i, 2] * Xb[j, [0, 1]]
                bE[0] += w[i] * w[j] * rate_i * rate_j
                bw += w[i] * w[j]
        assert np.allclose(nP, bP) and np.allclose(nE, bE) and np.isclose(sw, bw), (tile, binary)
        if not binary:
            K = AS.nnz / len(ti)
            R_T = X[ti, 2].sum() / s[ti].sum()
            L_S = X[si][:, [0, 1]].sum(0) / s[si].sum()
            P, Pexp = nP / sw, nE / sw
            gamma, pi = Pexp / (R_T * L_S), P / Pexp
            assert np.allclose(K * P, K * R_T * L_S * gamma * pi)


def test_contacts_are_symmetric_and_within_the_radius(xe):
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 100, (150, 2))
    A = xe._lrd_contacts(xy, 12.0).toarray()
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    assert np.array_equal(A, ((d <= 12.0) & ~np.eye(150, dtype=bool)).astype(float))
    assert xe._lrd_contacts(xy[:1], 12.0).shape == (1, 1)


# ── sums of a sample, the terms, the O/E variant ─────────────────────────────
@pytest.mark.parametrize("binary", [False, True], ids=["counts", "detection"])
def test_sample_terms_multiply_back_to_the_exposure_and_equal_a_hand_computation(xe, binary):
    c, xy, types = _world(1, fractional=True)
    lab = types.astype(object)
    lab[:6] = None                                          # unlabelled cells are left out
    smp = _hand_sample(c, xy, GENES)
    smp["rows"] = smp["rows"].copy()
    smp["rows"][6:9] = -1                                   # cells without a matrix row too
    tt = ["S", "T", "U"]
    sm = xe._lrd_sample_sums(smp, lab, PAIRS, tt, radius=30.0, tile=200.0, binary=binary)
    li, ri = np.array([0, 2]), np.array([1, 3])             # columns of the pairs' genes in sum_x
    t = xe._lrd_terms(sm, li, ri, binary)
    prod = t["contact"] * t["receptor_expr"] * t["ligand_expr"] * t["regional"] * t["preference"]
    m = t["exposure"] > 0                                   # (a zero exposure has an undefined factor)
    assert np.isfinite(t["exposure"]).all() and m.sum() >= 12 and m[0, 1, 0]
    assert np.allclose(prod[m], t["exposure"][m], rtol=1e-10, atol=0)
    # by hand for S → T, pair L1 → R1
    ok = np.ones(len(c), dtype=bool)
    ok[:9] = False
    S_, T_ = np.flatnonzero(ok & (types == "S")), np.flatnonzero(ok & (types == "T"))
    depth = c.sum(axis=1)
    x = (c >= 1).astype(float) if binary else c
    w = np.ones(len(c)) if binary else depth
    d = np.linalg.norm(xy[T_][:, None] - xy[S_][None], axis=2) <= 30.0
    K = d.sum() / len(T_)
    P = (x[T_, 1][:, None] * x[S_, 0][None, :] * d).sum() / (w[T_][:, None] * w[S_][None, :] * d).sum()
    R = x[T_, 1].sum() / w[T_].sum()
    L = x[S_, 0].sum() / w[S_].sum()
    assert sm["n_contacts"][0, 1] == d.sum() and sm["n_cells"].tolist() == [len(S_), len(T_), ok.sum() - len(S_) - len(T_)]
    assert np.isclose(t["contact"][0, 1, 0], K) and np.isclose(t["receptor_expr"][0, 1, 0], R)
    assert np.isclose(t["ligand_expr"][0, 1, 0], L) and np.isclose(t["exposure"][0, 1, 0], K * P)
    # observed / expected variant: expected = 1 − (1 − p_L)^k
    pos_l, pos_r = c[:, 0] >= 1, c[:, 1] >= 1
    k = d.sum(axis=1)
    obs = np.mean(pos_r[T_] & ((d & pos_l[S_][None, :]).sum(axis=1) >= 1))
    exp = np.mean(pos_r[T_] * (1.0 - (1.0 - pos_l[S_].mean()) ** k))
    assert np.isclose(t["oe_obs"][0, 1, 0], obs) and np.isclose(t["oe_exp"][0, 1, 0], exp)
    assert np.isclose(t["oe_enrich"][0, 1, 0], (obs + 1e-3) / (exp + 1e-3))
    # pooling two samples = summing their sums
    sm2 = xe._lrd_sample_sums(_hand_sample(*_world(2)[:2], GENES), _world(2)[2], PAIRS, tt, 30.0, 200.0, binary)
    pooled = xe._lrd_pool([sm, None, sm2])
    assert np.allclose(pooled["num_P"], sm["num_P"] + sm2["num_P"]) and np.allclose(pooled["n_cells"], sm["n_cells"] + sm2["n_cells"])


def test_uniform_field_has_no_co_location_or_preference_and_a_planted_niche_has(xe):
    c, xy, types = _world(5, n=1500, side=500.0)
    sm = xe._lrd_sample_sums(_hand_sample(c, xy, GENES), types, PAIRS, ["S", "T", "U"], 30.0, 250.0)
    t = xe._lrd_terms(sm, np.array([0, 2]), np.array([1, 3]))
    assert abs(np.log2(t["regional"][0, 1, 0])) < 0.15 and abs(np.log2(t["preference"][0, 1, 0])) < 0.2
    assert abs(np.log2(t["oe_enrich"][0, 1, 0])) < 0.15
    # ligand and receptor both high in the left half only: regional co-location, γ > 1
    left = xy[:, 0] < 250
    c2 = c.copy()
    rng = np.random.default_rng(6)
    c2[:, 0] = np.where(types == "S", rng.poisson(np.where(left, 8.0, 0.5)), c[:, 0])
    c2[:, 1] = np.where(types == "T", rng.poisson(np.where(left, 8.0, 0.5)), c[:, 1])
    sm = xe._lrd_sample_sums(_hand_sample(c2, xy, GENES), types, PAIRS, ["S", "T", "U"], 30.0, 250.0)
    t = xe._lrd_terms(sm, np.array([0, 2]), np.array([1, 3]))
    assert np.log2(t["regional"][0, 1, 0]) > 0.6 and abs(np.log2(t["preference"][0, 1, 0])) < 0.2
    # with the whole sample as one tile the same structure is contact preference instead
    sm = xe._lrd_sample_sums(_hand_sample(c2, xy, GENES), types, PAIRS, ["S", "T", "U"], 30.0, None)
    t = xe._lrd_terms(sm, np.array([0, 2]), np.array([1, 3]))
    assert abs(np.log2(t["regional"][0, 1, 0])) < 1e-9 and np.log2(t["preference"][0, 1, 0]) > 0.6


# ── planted change in one factor only ────────────────────────────────────────
def _sums(xe, worlds):
    return [xe._lrd_sample_sums(_hand_sample(c, xy, GENES, f"w{i}"), ty, PAIRS, ["S", "T", "U"], 30.0, 200.0)
            for i, (c, xy, ty) in enumerate(worlds)]


def _row(t, s="S", r="T", lig="L1"):
    return t[(t["sender"] == s) & (t["receiver"] == r) & (t["ligand"] == lig)].iloc[0]


@pytest.mark.parametrize("factor", ["ligand_expr", "receptor_expr", "contact"])
def test_planted_change_in_one_factor_is_attributed_to_that_factor(xe, factor):
    kw = {"ligand_expr": dict(lig=8.0), "receptor_expr": dict(rec=8.0), "contact": dict(side=400.0 / np.sqrt(2.0))}[factor]
    worlds = [_world(10 + i, **kw) for i in range(3)] + [_world(20 + i) for i in range(3)]
    res = xe._lrd_decompose(_sums(xe, worlds), ["S", "T", "U"], PAIRS, conds=["A"] * 3 + ["B"] * 3)
    assert res["mode"] == "unpaired" and res["test"] == "exact unpaired label permutation"
    assert res["units_note"] == "n = 6 samples" and np.isclose(res["min_p"], 0.1) and res["n_units"] == 6
    t = res["table"]
    r = _row(t)
    assert abs(r[f"d_{factor}"] - 1.0) < 0.25                      # the planted doubling, on log2
    assert all(abs(r[f"d_{k}"]) < 0.25 for k in TERMS if k != factor)
    assert np.isclose(r["d_exposure"], sum(r[f"d_{k}"] for k in TERMS), atol=1e-9)
    assert np.isclose(r[f"p_{factor}"], 0.1) and np.isclose(r["p_exposure"], 0.1)
    assert max(abs(r[f"d_{k}"]) for k in TERMS if k != factor) < 0.3 * abs(r[f"d_{factor}"])
    # the unchanged pair of the same types moves only with the contacts
    r2 = _row(t, lig="L2")
    assert abs(r2["d_ligand_expr"]) < 0.2 and abs(r2["d_receptor_expr"]) < 0.2
    assert (abs(r2["d_exposure"] - 1.0) < 0.3) == (factor == "contact")
    # type-pair table and omnibus
    pt = res["pairs"]
    one = pt[(pt["sender"] == "S") & (pt["receiver"] == "T") & (pt["component"] == factor)].iloc[0]
    assert one["n_interactions"] == 2 and 0 < one["p"] <= 1
    om = res["omnibus"][factor]
    assert om["exact"] and om["n_pairs"] == len(pt[(pt["component"] == factor) & (pt["sender"] != pt["receiver"])])
    assert 0 < om["p"] <= 1 and np.isfinite(om["T"])


def test_paired_design_uses_the_sign_flip_test_and_pools_a_donors_samples(xe):
    # 4 donors, each with one A sample (ligand doubled) and one B sample; donor d0 has two B samples
    worlds = [_world(30 + i, lig=8.0) for i in range(4)] + [_world(40 + i) for i in range(5)]
    conds = ["A"] * 4 + ["B"] * 5
    donors = [f"d{i}" for i in range(4)] + [f"d{i}" for i in range(4)] + ["d0"]
    sums = _sums(xe, worlds)
    res = xe._lrd_decompose(sums, ["S", "T", "U"], PAIRS, conds=conds, donors=donors)
    assert res["mode"] == "paired" and res["test"] == "exact paired sign-flip" and res["n_units"] == 4
    assert "4 donors, paired" in res["units_note"] and np.isclose(res["min_p"], 2 / 16)
    r = _row(res["table"])
    assert abs(r["d_ligand_expr"] - 1.0) < 0.2 and r["n_up_ligand_expr"] == 4 and np.isclose(r["p_ligand_expr"], 2 / 16)
    assert np.isclose(r["d_exposure"], sum(r[f"d_{k}"] for k in TERMS), atol=1e-9)
    # d0's two B samples were pooled by summing their sums: the same as one sample holding both
    li, ri = np.array([0, 2]), np.array([1, 3])
    tb = xe._lrd_terms(xe._lrd_pool([sums[4], sums[8]]), li, ri)
    ta = xe._lrd_terms(sums[0], li, ri)
    d0 = np.log2(ta["ligand_expr"][0, 1, 0] / tb["ligand_expr"][0, 1, 0])
    others = [np.log2(xe._lrd_terms(sums[i], li, ri)["ligand_expr"][0, 1, 0]
                      / xe._lrd_terms(sums[4 + i], li, ri)["ligand_expr"][0, 1, 0]) for i in (1, 2, 3)]
    assert np.isclose(r["d_ligand_expr"], np.mean([d0] + others))
    pt = res["pairs"]
    one = pt[(pt["sender"] == "S") & (pt["receiver"] == "T") & (pt["component"] == "ligand_expr")].iloc[0]
    assert one["n_up"] == 4 and np.isnan(one["lo"])        # 4 donors: no sign-flip p below 0.05, no 95 % interval
    om = res["omnibus"]["ligand_expr"]
    assert om["exact"] and 0 < om["p"] <= 1 and 0 < om["shared_p"] <= 1


def test_no_p_values_without_a_design_or_with_too_few_units_and_it_says_why(xe):
    worlds = [_world(50 + i, lig=8.0 if i < 2 else 4.0) for i in range(4)]
    sums = _sums(xe, worlds)
    one = xe._lrd_decompose(sums[:1], ["S", "T", "U"], PAIRS)
    assert one["mode"] == "descriptive" and "one sample is loaded" in one["reason"]
    assert "only one sample has labelled cells" in xe._lrd_decompose([sums[0], None], ["S", "T", "U"], PAIRS)["reason"]
    assert not [c for c in one["table"].columns if c.startswith(("p_", "d_", "fdr_"))] and not len(one["pairs"])
    t = one["table"]
    assert np.allclose(t["exposure"], t[TERMS].prod(axis=1), rtol=1e-10)
    same = xe._lrd_decompose(sums, ["S", "T", "U"], PAIRS, conds=["A"] * 4)
    assert same["mode"] == "descriptive" and "all in one condition" in same["reason"]
    none = xe._lrd_decompose(sums, ["S", "T", "U"], PAIRS)
    assert none["mode"] == "descriptive" and "no sample is assigned to conditions A and B" in none["reason"]
    assert none["table"]["n_receivers"].iloc[0] == 4 * 180                    # the samples are pooled
    few = xe._lrd_decompose(sums, ["S", "T", "U"], PAIRS, conds=["A", "A", "B", "B"])
    assert few["mode"] == "ratio" and "fewer than 3 independent units per condition" in few["reason"]
    r = _row(few["table"])
    assert abs(r["d_ligand_expr"] - 1.0) < 0.2 and not [c for c in few["table"].columns if c.startswith("p_")]
    assert np.isclose(r["d_exposure"], sum(r[f"d_{k}"] for k in TERMS), atol=1e-9)


def test_filters_floor_ownership_and_cells(xe):
    sums = _sums(xe, [_world(60)])
    base = xe._lrd_decompose(sums, ["S", "T", "U"], PAIRS)
    t = base["table"]
    # L1 is owned by S only and R1 by T only; L2 / R2 by everyone: 1 + 9 interactions
    assert len(t) == 10 and len(t[t["ligand"] == "L1"]) == 1 and base["n_below_floor"] == 8 and base["n_not_owned"] == 0
    assert _row(t)["ligand_share"] == 1.0 and _row(t)["receptor_share"] == 1.0
    assert xe._lrd_decompose(sums, ["S", "T", "U"], PAIRS, floor=0.0)["n_not_owned"] == 8
    loose = xe._lrd_decompose(sums, ["S", "T", "U"], PAIRS, own=0.0, floor=0.0)
    assert len(loose["table"]) == 18
    high = xe._lrd_decompose(sums, ["S", "T", "U"], PAIRS, floor=500.0)       # 180 cells × 4 = 720 of L1 / R1 only
    assert set(high["table"]["ligand"]) == {"L1"} and high["n_below_floor"] == 9 + 8
    assert not len(xe._lrd_decompose(sums, ["S", "T", "U"], PAIRS, min_cells=500)["table"])


# ── the matrix tests are the app's exact sample-level tests ──────────────────
def test_matrix_tests_equal_the_sample_level_tests(xe):
    rng = np.random.default_rng(7)
    D = rng.normal(0.3, 1.0, (6, 40))
    D[:, 5] = np.nan
    p = xe._lrd_signflip_matrix(D)
    assert np.isnan(p[5])
    assert np.allclose(np.delete(p, 5), [xe._sl_signflip(D[:, j])["p"] for j in range(40) if j != 5])
    A, B = rng.normal(0.5, 1.0, (4, 30)), rng.normal(0.0, 1.0, (5, 30))
    q = xe._lrd_permutation_matrix(A, B)
    assert np.allclose(q, [xe._sl_permutation(A[:, j], B[:, j])["p"] for j in range(30)])
    idx, exact = xe._lrd_assignments(4, 5)
    assert exact and len(idx) == 126 and idx[0].tolist() == [0, 1, 2, 3]
    idx, exact = xe._lrd_assignments(12, 12, limit=1000, n_mc=50)
    assert not exact and idx.shape == (50, 12)
    lo, hi = xe._lrd_signflip_ci(np.array([0.9, 1.1, 1.0, 1.2, 0.8, 1.05]))
    assert 0.75 < lo < 1.0 < hi < 1.25
    # omnibus: a shift shared by every pair is found, pure noise is not
    M = rng.normal(0, 0.2, (7, 12))
    assert xe._lrd_omnibus(M + 1.0, True)["p"] == 2 / 128
    assert xe._lrd_omnibus(M, True)["p"] > 0.05
    U = rng.normal(0, 0.2, (8, 12))
    U[:4] += 1.0
    o = xe._lrd_omnibus(U, False, n_a=4)
    assert o["exact"] and np.isclose(o["p"], 2 / 70) and abs(o["shared_shift"] - 1.0) < 0.2
    assert np.isnan(xe._lrd_omnibus(np.zeros((5, 0)), True)["p"])


# ── the tab ──────────────────────────────────────────────────────────────────
def _extra(D, world, seed):
    """An added-sample dataset dict holding a fabricated world on the bundle's gene names."""
    c, xy, types = world
    names = list(D["gene_names"])
    m = sp.lil_matrix((len(types), len(names)), dtype=np.float32)
    m[:, :c.shape[1]] = c
    df = pd.DataFrame({"x_centroid": xy[:, 0], "y_centroid": xy[:, 1], "transcript_counts": c.sum(axis=1)},
                      index=[f"e{seed}_{i}" for i in range(len(types))])
    return {"expr": m.tocsr(), "df_to_expr": np.arange(len(types)), "gene_name_to_idx": D["gene_name_to_idx"],
            "gene_names": names, "df": df, "x_offset": 1000.0 * (seed + 1), "cluster_methods": [],
            "_cell_type_labels": pd.Series(types, index=df.index)}


def test_decomposition_tab_runs_as_a_job_single_sample_then_with_a_design(xe, D, monkeypatch):
    names = list(D["gene_names"])
    L1, R1, L2, R2 = names[:4]
    text = f"ligand,receptor\n{L1},{R1}\n{L2},{R2}\n{L1},NOPE\n"
    args = (30, 200, 0.2, 100, 50, "counts", "xenium", None, "original")
    # guards: a dynamically created / never clicked button, no pairs, no labels
    with pytest.raises(Exception, match="PreventUpdate|^$"):
        xe.run_lrdecomp(0, text, "celltype:celltypist", *args)
    with pytest.raises(Exception, match="PreventUpdate|^$"):
        xe.open_design_modal_lrd(0)
    with pytest.raises(Exception, match="PreventUpdate|^$"):
        xe.download_lrd_table(1)                                              # nothing computed yet
    assert xe.run_lrdecomp(1, "", "celltype:celltypist", *args)[2].startswith("No ligand")
    assert xe.run_lrdecomp(1, text, None, *args)[2].startswith("Pick the cell types")
    fig, cols, rows = xe.update_lrd_results(0, 0, "interactions", 25, None, None)
    assert "No exposure decomposition computed yet" in fig.layout.annotations[0].text and rows == []
    # one labelled sample: descriptive
    xe.EXTRA_DATASETS.append(_extra(D, _world(70), 0))
    assert xe.run_lrdecomp(1, text, "celltype:celltypist", *args) == (True, False, "Running…")
    msg = tl.wait_state(xe._lrd_state, xe._lrd_lock)
    job = [j for j in xe._jobs.values() if j.kind == "lrdecomp"][-1]
    job.thread.join(10)
    assert job.state == "done" and xe.poll_lrdecomp(1, 0) == (msg, True, False, 1)
    assert "2 of 3 ligand–receptor pairs have both genes on the panel" in msg and "Σ counts / Σ depth" in msg
    assert "No p-values: only one sample has labelled cells" in msg and "not evidence of signalling" in msg
    assert "Exposure = K × R × L × γ × π exactly" in msg
    res = xe._lrd_result()
    assert res["mode"] == "descriptive" and len(res["table"]) == 10
    fig, cols, rows = xe.update_lrd_results(1, 0, "interactions", 25, None, None)
    assert [tr.name for tr in fig.data] == ["Tile co-location (γ)", "Contact preference (π)"]
    assert [c["id"] for c in cols][:4] == ["sender", "receiver", "ligand", "receptor"] and "p_exposure" not in [c["id"] for c in cols]
    assert len(rows) == 10 and rows[0]["exposure"] >= rows[-1]["exposure"]
    out = xe.download_lrd_table(1)
    head = [l for l in out["content"].splitlines() if l.startswith("#")]
    assert "mode: descriptive" in out["content"] and "not evidence of signalling" in "\n".join(head)
    assert len(pd.read_csv(io.StringIO(out["content"]), comment="#")) == 10
    # six samples in two conditions: tested across the samples
    del xe.EXTRA_DATASETS[:]
    extras = [_extra(D, _world(80 + i, lig=8.0 if i < 3 else 4.0), i) for i in range(6)]
    xe.EXTRA_DATASETS.extend(extras)
    xe._design_set([{"key": xe._ds_key(e), "cond": "A" if i < 3 else "B", "batch": ""} for i, e in enumerate(extras)],
                   {"A": "treated", "B": "control"})
    assert xe.run_lrdecomp(2, text, "celltype:celltypist", *args) == (True, False, "Running…")
    msg = tl.wait_state(xe._lrd_state, xe._lrd_lock)
    xe._test_baseline.settle()
    assert "log2 ratios A (treated) / B (control); exact unpaired label permutation per term" in msg
    assert "unit of the p-values: n = 6 samples (not cells); smallest achievable p 0.1" in msg
    assert "no test can be significant" in msg and "Omnibus over the between-type pairs" in msg
    res = xe._lrd_result()
    r = res["table"][(res["table"]["sender"] == "S") & (res["table"]["ligand"] == L1)].iloc[0]
    assert abs(r["d_ligand_expr"] - 1.0) < 0.25 and abs(r["d_contact"]) < 0.25 and np.isclose(r["p_ligand_expr"], 0.1)
    fig, cols, rows = xe.update_lrd_results(2, 0, "interactions", 5, None, None)
    assert [tr.name for tr in fig.data][-1] == "Exposure (sum)" and len(fig.data) == 6 and fig.layout.barmode == "relative"
    assert len(fig.data[0].y) == 5 and "p_exposure" in [c["id"] for c in cols]
    top = fig.data[-1]
    lab = f"{L1} → {R1}  |  S → T"
    k = list(top.y).index(lab)
    assert np.isclose(sum(tr.x[k] for tr in fig.data[:5]), top.x[k])          # the bars sum to the diamond
    fig, cols, rows = xe.update_lrd_results(2, 0, "pairs", 25, None, None)
    assert "S → T" in list(fig.data[0].y) and "n_interactions" in [c["id"] for c in cols] and "ligand" not in rows[0]
    out = xe.download_lrd_table(1)
    assert "# omnibus ligand_expr: T = " in out["content"] and "# sender -> receiver pairs" in out["content"]
    assert "condition A; donor -" in out["content"]
    # no labels of the chosen kind
    xe.run_lrdecomp(3, text, "niche", *args)
    with pytest.raises(AssertionError, match="no niche labels"):
        tl.wait_state(xe._lrd_state, xe._lrd_lock)
    xe._test_baseline.settle()
    assert xe._lrd_result() is res                                            # the last result stays
    # cancel mid-run: ends cancelled, nothing new is published, a rerun completes
    entered, release = threading.Event(), threading.Event()
    real = xe._lrd_contacts

    def slow(xy_, radius):
        entered.set()
        release.wait(30)
        return real(xy_, radius)

    monkeypatch.setattr(xe, "_lrd_contacts", slow)
    ver = xe._lrd_state["version"]
    xe.run_lrdecomp(4, text, "celltype:celltypist", *args)
    assert entered.wait(30)
    job = [j for j in xe._jobs.values() if j.kind == "lrdecomp"][-1]
    assert xe._job_cancel(job.id)
    release.set()
    job.thread.join(30)
    assert job.state == "cancelled" and xe._lrd_state["message"] == "Cancelled" and xe._lrd_state["status"] == "error"
    assert xe._lrd_result() is res and xe._lrd_state["version"] == ver
    monkeypatch.setattr(xe, "_lrd_contacts", real)
    assert xe.run_lrdecomp(5, text, "celltype:celltypist", 30, 200, 0.2, 100, 50, "binary", "xenium", None, "original")[2] == "Running…"
    msg = tl.wait_state(xe._lrd_state, xe._lrd_lock)
    xe._test_baseline.settle()
    assert "detection (count ≥ 1) / cells" in msg and xe._lrd_result() is not res and xe._lrd_result()["binary"]


def test_mode_switch_and_layout(xe):
    import json
    import plotly
    text = json.dumps(xe.app.layout, cls=plotly.utils.PlotlyJSONEncoder, ensure_ascii=False)
    for needle in ("lr-mode", "lrd-run-btn", "lrd-plot", "lrd-table", "lrd-design-open-btn", "lrd-download",
                   "Exposure decomposition", "not evidence of signalling", "1 − (1 − p_L)^k"):
        assert needle in text, needle
    assert xe._JOB_KIND_NAMES["lrdecomp"]
    a = xe.toggle_lr_mode("scores", {"display": "flex", "gap": "6px"}, {"display": "none", "flex": "1"})
    assert a[0]["display"] == "flex" and a[1] == {"display": "flex", "flex": "1"} and a[2]["display"] == "none" and a[3]["display"] == "none"
    b = xe.toggle_lr_mode("decomp", a[0], a[1])
    assert b[0]["display"] == "none" and b[1]["display"] == "none" and "display" not in b[2] and b[3]["display"] == "flex"
    # a result of another dataset generation is not shown
    with xe._lrd_lock:
        xe._lrd_state["result"] = {"gen": -1}
    assert xe._lrd_result() is None
