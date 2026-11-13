"""Ligand-receptor exposure decomposition (compute + layout).

Exposure of a receiver cell type T to a ligand L made by a sender type S, for the receptor
R: E = K × R_T × L_S × γ × π (exact), with K the sender contacts per receiver, R_T and L_S
the type-wide rates of the receptor and the ligand (Σ counts / Σ depth), γ the regional
co-location of the two rates among the cells that are in contact (tile-level rates) and π
the preference of the contacts within a tile. Between two conditions the log2 ratio of the
exposure is the sum of the log2 ratios of the five terms, so a change is attributed to
expression, contact or geometry. The simpler observed / expected variant is computed
beside it. Every contact search is made on the cells of ONE sample.
"""

import threading
import time as _time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from dash import dcc, html
from dash import dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from . import core, jobs, ui, figures, analysis, stats
from . import provenance


_lrd_state: dict = {"status": "idle", "message": "", "result": None, "version": 0}
_lrd_lock = threading.Lock()

_LRD_RADIUS_UM = 30.0       # two cells within this distance (centroids, same sample) are in contact
_LRD_TILE_UM = 250.0        # side of the square tiles of the regional rates (0 / None: the whole sample)
_LRD_OWN = 0.2              # the sender must reach this share of the top type's ligand rate (receiver: receptor)
_LRD_FLOOR = 100.0          # transcripts of the gene in the type, in every unit, below which a pair is left out
_LRD_MIN_CELLS = 50         # cells a type needs in every unit to be a sender / receiver
_LRD_MAX_TYPES = 40
_LRD_EPS = 1e-3             # added to observed and expected fractions of the O/E variant
_LRD_ENUM_ASSIGN = 20_000   # relabellings enumerated for the matrix permutation tests
_LRD_MC = 5_000             # Monte Carlo draws above the enumeration limits (fixed seed)
_LRD_TERMS = ("contact", "receptor_expr", "ligand_expr", "regional", "preference")
_LRD_TERM_NAMES = {"contact": "Contacts per receiver (K)", "receptor_expr": "Receptor rate (R)",
                   "ligand_expr": "Ligand rate (L)", "regional": "Tile co-location (γ)",
                   "preference": "Contact preference (π)", "exposure": "Exposure (E)"}
_LRD_TERM_COLORS = {"contact": "#58a6ff", "receptor_expr": "#3fb950", "ligand_expr": "#d29922",
                    "regional": "#bc8cff", "preference": "#f85149"}

_LRD_NOTE = (
    "Exposure decomposition. For sender type S, receiver type T, ligand L and receptor R, with the "
    "contact set C = (receiver i, sender j) pairs whose centroids are within the radius in the same "
    "sample and s = a cell's total count: K = |C| / N_T (sender contacts per receiver); R_T = Σ_T x_R / "
    "Σ_T s and L_S = Σ_S x_L / Σ_S s (type-wide rates; Σ counts / Σ depth, so fractional SPLIT-"
    "corrected counts are handled as they are); P = Σ_C x_iR x_jL / Σ_C s_i s_j; P_exp = the same sum "
    "with each cell's count replaced by its depth × the rate of its type in its tile; γ = P_exp / "
    "(R_T L_S) (tile co-location: do senders rich in the ligand sit in the tiles where receivers rich "
    "in the receptor sit, among the cells in contact); π = P / P_exp (contact preference: within a "
    "tile, are the cells in contact the ones that carry the two genes). Exposure E = K × P = K × R_T × "
    "L_S × γ × π exactly. With 'detection' the counts are replaced by 1[count ≥ 1] and the depth by 1. "
    "Between two conditions the log2 ratio A / B of E is the sum of the log2 ratios of the five terms: "
    "the bars show WHY an exposure differs - expression (R, L), contact (K) or geometry (γ, π). The "
    "sums of the samples of one donor within a condition are pooled before the ratios are taken. "
    "Tests are across the independent units of the sample design (donors, or samples): exact sign-"
    "flip test of the per-donor log2 ratios when paired, exact label permutation otherwise; per term "
    "an omnibus over the between-type pairs (Σ t² against all sign / label assignments). With one "
    "condition, one sample or fewer than 3 units per condition the decomposition is shown "
    "descriptively, without p-values. Interactions are kept when the sender owns the ligand and the "
    "receiver the receptor (rate ≥ the given share of the highest type's rate) and both reach the "
    "transcript floor in every unit. Observed / expected variant: observed = share of receivers that "
    "detect the receptor and have at least one ligand-positive sender in contact; expected = the same "
    "with ligand positivity assigned at random among the senders, 1 − (1 − p_L)^k for a receiver with "
    "k sender contacts. This explains why an exposure differs; like every expression-based score it "
    "is not evidence of signalling, and transcripts spilling between touching cells raise π.")


def _lrd_reset() -> None:
    with _lrd_lock:
        _lrd_state.update({"status": "idle", "message": "", "result": None})
        _lrd_state["version"] += 1


def _lrd_result():
    """The last result if it belongs to the loaded dataset, else None."""
    with _lrd_lock:
        res = _lrd_state.get("result")
    if res is not None and res.get("gen") != jobs._dataset_gen:
        return None
    return res


# ─── Core sums (ported from the original analysis; see the tests) ─────────────
def _lrd_tile_rates(xy, section, X, s, tile, binary):
    """Per cell, the rate of each gene among the cells of its tile (tile None: its
    section). The cells passed are those of ONE type. Rate = Σ x / Σ s (binary: positive
    cells / cells)."""
    if tile is None:
        key = pd.factorize(np.asarray(section))[0]
    else:
        key = pd.factorize(pd.Series(np.asarray(section)).astype(str) + ":" + (xy[:, 0] // tile).astype(int).astype(str)
                           + ":" + (xy[:, 1] // tile).astype(int).astype(str))[0]
    n_t = key.max() + 1
    O = sp.csr_matrix((np.ones(len(key)), (key, np.arange(len(key)))), shape=(n_t, len(key)))
    num = np.asarray((O @ sp.csr_matrix(X)).todense())
    den = O @ (np.ones(len(key)) if binary else np.asarray(s, float))
    return (num / den[:, None])[key]


def _lrd_pair_sums(AS, xr, xl, wT, wS, Rt, Lt):
    """AS: receivers × senders contact matrix. xr: receivers × receptors, xl: senders ×
    ligands (counts or 0/1). wT, wS: depth weights (s, or ones for binary). Rt, Lt:
    per-cell tile rates. Returns receptor × ligand matrices num_P = Σ_C xr_i xl_j, num_E =
    Σ_C wT_i wS_j Rt_i Lt_j, and the scalar Σ_C wT_i wS_j."""
    xr = sp.csr_matrix(xr); xl = sp.csr_matrix(xl)
    num_P = np.asarray((xr.T @ (AS @ xl)).todense())
    num_E = (wT[:, None] * Rt).T @ (AS @ (wS[:, None] * Lt))
    return num_P, np.asarray(num_E), float(wT @ (AS @ wS))


def _lrd_contacts(xy, radius: float) -> "sp.csr_matrix":
    """Symmetric 0/1 contact matrix of the cells of ONE sample: centroids within *radius*."""
    from scipy.spatial import cKDTree
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    if n < 2:
        return sp.csr_matrix((n, n))
    prs = cKDTree(xy).query_pairs(float(radius), output_type="ndarray")
    if not len(prs):
        return sp.csr_matrix((n, n))
    r = np.concatenate([prs[:, 0], prs[:, 1]])
    c = np.concatenate([prs[:, 1], prs[:, 0]])
    return sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))


# ─── Sums of one sample ───────────────────────────────────────────────────────
def _lrd_match_pairs(pairs, genes_all) -> list:
    """The pairs with both genes among *genes_all* (case-insensitive), as panel names."""
    names = list(dict.fromkeys([g for pr in pairs for g in pr]))
    found, _missing = analysis._gs_match(names, genes_all)
    lower = {g.lower(): g for g in found}
    have = set(found)
    _hit = lambda g: g if g in have else lower.get(str(g).lower())
    kept = []
    for lg, rc in pairs:
        a, b = _hit(lg), _hit(rc)
        if a is not None and b is not None and (a, b) not in kept:
            kept.append((a, b))
    return kept


def _lrd_sample_sums(s: dict, lab, kept, types, radius=_LRD_RADIUS_UM, tile=_LRD_TILE_UM,
                     binary: bool = False, progress=None) -> "dict | None":
    """The sums of one sample that every term is a ratio of.

    s: analysis sample; lab: type per cell (None: unlabelled); kept: [(ligand, receptor)]
    on the panel; types: the type names. Returns {n_cells [T], sum_s [T], sum_x [T, G],
    n_pos [T, G] (G: the genes of the pairs), n_contacts [S, T], num_P / num_E [S, T, P],
    sum_ww [S, T], oe_num / oe_exp [S, T, P], n_adj (contact pairs)}; sender first.
    None for a sample without labels. Cells without a matrix row or a label are left out."""
    if lab is None:
        return None
    T, P = len(types), len(kept)
    G = list(dict.fromkeys([g for pr in kept for g in pr]))
    gpos = {g: i for i, g in enumerate(G)}
    li = np.array([gpos[a] for a, _ in kept], dtype=np.int64)
    ri = np.array([gpos[b] for _, b in kept], dtype=np.int64)
    tpos = {t: i for i, t in enumerate(types)}
    rows = np.asarray(s["rows"])
    code = pd.Series(lab, dtype=object).map(lambda v: tpos.get(str(v), -1) if v is not None else -1).to_numpy(dtype=np.int64)
    ok = (rows >= 0) & (code >= 0)
    out = {"n_cells": np.zeros(T), "sum_s": np.zeros(T), "sum_x": np.zeros((T, len(G))),
           "n_pos": np.zeros((T, len(G))), "n_contacts": np.zeros((T, T)), "sum_ww": np.zeros((T, T)),
           "num_P": np.zeros((T, T, P)), "num_E": np.zeros((T, T, P)),
           "oe_num": np.zeros((T, T, P)), "oe_exp": np.zeros((T, T, P)), "n_adj": 0}
    if not ok.any():
        return out
    code = code[ok]
    csc = s["csc"]
    X = csc[:, np.array([s["gni"][g] for g in G], dtype=np.int64)].tocsr()[rows[ok]].astype(np.float64)
    wp = np.zeros(csc.shape[1])
    wp[np.array([s["gni"][g] for g in s["genes"]], dtype=np.int64)] = 1.0
    depth = np.asarray(csc @ wp, dtype=np.float64).ravel()[rows[ok]]
    B = X.copy()
    B.data = (B.data >= analysis._DETECT_MIN).astype(np.float64)
    B.eliminate_zeros()
    xy = stats._xy_of(s["df"])[ok]
    jobs._job_check()
    Adj = _lrd_contacts(xy, radius).tocsc()
    out["n_adj"] = int(Adj.nnz // 2)
    Xv = B if binary else X
    w = np.ones(len(code)) if binary else depth
    idx = [np.flatnonzero(code == t) for t in range(T)]
    rates = {}
    for t in range(T):
        i = idx[t]
        out["n_cells"][t] = len(i)
        if not len(i):
            continue
        out["sum_s"][t] = depth[i].sum()
        out["sum_x"][t] = np.asarray(X[i].sum(axis=0)).ravel()
        out["n_pos"][t] = np.asarray(B[i].sum(axis=0)).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            tr = _lrd_tile_rates(xy[i], np.zeros(len(i), dtype=np.int8), Xv[i], depth[i],
                                 float(tile) if tile else None, binary)
        rates[t] = np.nan_to_num(tr, nan=0.0, posinf=0.0, neginf=0.0)   # a tile without depth: weight 0 anyway
    lig = np.unique(li)
    rec = np.unique(ri)
    lcol = np.searchsorted(lig, li)
    rcol = np.searchsorted(rec, ri)
    for S in range(T):
        jobs._job_check()
        if progress is not None:
            progress(S / max(1, T))
        si = idx[S]
        if not len(si):
            continue
        AdjS = Adj[:, si].tocsr()                       # all cells × senders of type S
        if not AdjS.nnz:
            continue
        xl, wS, Lt = Xv[si][:, lig], w[si], rates[S][:, lig]
        k_all = np.asarray(AdjS.sum(axis=1)).ravel()    # sender contacts of every cell
        Bl = B[si][:, lig]
        pL = np.asarray(Bl.mean(axis=0)).ravel()        # ligand positivity among the senders
        has = (AdjS @ Bl).tocsr()
        has.data = (has.data >= 1).astype(np.float64)
        ku, kinv = np.unique(k_all, return_inverse=True)                 # expected: one row per distinct k
        pe_k = 1.0 - np.power(1.0 - pL[None, :], ku[:, None])            # distinct k × ligands
        for Tt in range(T):
            ti = idx[Tt]
            if not len(ti):
                continue
            AS = AdjS[ti]
            if not AS.nnz:
                continue
            nP, nE, sw = _lrd_pair_sums(AS, Xv[ti][:, rec], xl, w[ti], wS, rates[Tt][:, rec], Lt)
            out["n_contacts"][S, Tt] = AS.nnz
            out["sum_ww"][S, Tt] = sw
            out["num_P"][S, Tt] = nP[rcol, lcol]
            out["num_E"][S, Tt] = nE[rcol, lcol]
            Rr = B[ti][:, rec]
            out["oe_num"][S, Tt] = np.asarray((Rr.T @ has[ti]).todense())[rcol, lcol]
            Hk = sp.csr_matrix((np.ones(len(ti)), (np.arange(len(ti)), kinv[ti])), shape=(len(ti), len(ku)))
            out["oe_exp"][S, Tt] = (np.asarray((Rr.T @ Hk).todense()) @ pe_k)[rcol, lcol]   # Σ_i R_i (1 − (1 − p_L)^k_i)
    if progress is not None:
        progress(1.0)
    return out


_LRD_SUM_KEYS = ("n_cells", "sum_s", "sum_x", "n_pos", "n_contacts", "sum_ww", "num_P", "num_E",
                 "oe_num", "oe_exp")


def _lrd_pool(sums: list) -> "dict | None":
    """Sum the sums of several samples (one unit): ratios of pooled sums are exact."""
    sums = [x for x in sums if x is not None]
    if not sums:
        return None
    return {k: sum(x[k] for x in sums) for k in _LRD_SUM_KEYS}


def _lrd_terms(sm: dict, li, ri, binary: bool = False) -> dict:
    """The five terms, the exposure and the O/E variant from the sums of one unit; arrays
    [sender, receiver, pair]. li / ri: column of the ligand / receptor of every pair in
    sum_x. Undefined entries (no contacts, no depth, a zero rate) are NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        n = sm["n_cells"]
        if binary:
            rate = sm["n_pos"] / n[:, None]
        else:
            rate = sm["sum_x"] / sm["sum_s"][:, None]
        K = (sm["n_contacts"] / n[None, :])[:, :, None] + np.zeros_like(sm["num_P"])
        R = np.broadcast_to(rate[:, ri][None, :, :], sm["num_P"].shape)
        L = np.broadcast_to(rate[:, li][:, None, :], sm["num_P"].shape)
        Pm = sm["num_P"] / sm["sum_ww"][:, :, None]
        Pe = sm["num_E"] / sm["sum_ww"][:, :, None]
        obs = sm["oe_num"] / n[None, :, None]
        exp = sm["oe_exp"] / n[None, :, None]
        out = {"contact": K, "receptor_expr": R, "ligand_expr": L, "regional": Pe / (R * L),
               "preference": Pm / Pe, "exposure": K * Pm, "oe_obs": obs, "oe_exp": exp,
               "oe_enrich": (obs + _LRD_EPS) / (exp + _LRD_EPS)}
    for k, v in out.items():
        v = np.array(v, dtype=np.float64)
        v[~np.isfinite(v)] = np.nan
        out[k] = v
    return out


def _lrd_ownership(pooled: dict, li, ri) -> tuple:
    """(share of the top type's rate for the ligand in every sender [T, P], the same for
    the receptor in every receiver [T, P]) from the sums pooled over all samples."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = pooled["sum_x"] / pooled["sum_s"][:, None]
        rate = np.nan_to_num(rate)
        share = rate / np.maximum(rate.max(axis=0, keepdims=True), 1e-300)
    return share[:, li], share[:, ri]


# ─── Tests across units (matrix forms of stats._sl_signflip / _sl_permutation) ─
def _lrd_signflip_matrix(D) -> np.ndarray:
    """Two-sided exact sign-flip p of the mean of every column of D (units × tests); the
    value stats._sl_signflip gives per column. Columns with a NaN are NaN."""
    D = np.asarray(D, dtype=np.float64)
    n, m = D.shape
    p = np.full(m, np.nan)
    ok = np.isfinite(D).all(axis=0)
    if not n or not ok.any():
        return p
    if n > stats._SL_ENUM_UNITS:
        p[ok] = [stats._sl_signflip(D[:, j])["p"] for j in np.flatnonzero(ok)]
        return p
    S = stats._sl_signs(n)
    cols = np.flatnonzero(ok)
    for a in range(0, len(cols), 2000):
        c = cols[a:a + 2000]
        obs = np.abs(D[:, c].sum(axis=0))
        tol = 1e-12 * np.maximum(1.0, obs)
        p[c] = (np.abs(S @ D[:, c]) >= (obs - tol)[None, :]).mean(axis=0)
    return p


def _lrd_assignments(n_a: int, n_b: int, limit: int = _LRD_ENUM_ASSIGN, n_mc: int = _LRD_MC, seed: int = 0) -> tuple:
    """(index matrix [assignments, n_a] of the units called "A", exact?): all C(n, n_a)
    relabellings when they number at most *limit* (the observed one first), else *n_mc*
    random ones (fixed seed)."""
    from itertools import combinations
    from math import comb
    n = n_a + n_b
    if comb(n, n_a) <= limit:
        idx = np.fromiter((i for c in combinations(range(n), n_a) for i in c), dtype=np.int64,
                          count=comb(n, n_a) * n_a).reshape(-1, n_a)
        return idx, True
    rng = np.random.default_rng(seed)
    return np.stack([np.sort(rng.permutation(n)[:n_a]) for _ in range(int(n_mc))]), False


def _lrd_permutation_matrix(A, B, assign=None) -> np.ndarray:
    """Two-sided label-permutation p of mean(A) − mean(B) for every column (A: units of
    condition A × tests, B likewise); exact when the relabellings are enumerated - then the
    value stats._sl_permutation gives per column - else (1 + hits) / (1 + draws)."""
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    na, nb = A.shape[0], B.shape[0]
    m = A.shape[1]
    p = np.full(m, np.nan)
    V = np.vstack([A, B])
    ok = np.isfinite(V).all(axis=0)
    if not na or not nb or not ok.any():
        return p
    idx, exact = assign if assign is not None else _lrd_assignments(na, nb)
    cols = np.flatnonzero(ok)
    step = max(1, int(4_000_000 // max(1, len(idx))))
    for a in range(0, len(cols), step):
        c = cols[a:a + step]
        v = V[:, c]
        tot = v.sum(axis=0)
        sa = v[idx].sum(axis=1)                               # assignments × tests
        st = sa / na - (tot[None, :] - sa) / nb
        obs = A[:, c].mean(axis=0) - B[:, c].mean(axis=0)
        tol = 1e-12 * np.maximum(1.0, np.abs(obs))
        hits = (np.abs(st) >= (np.abs(obs) - tol)[None, :]).sum(axis=0)
        p[c] = hits / len(idx) if exact else (hits + 1.0) / (len(idx) + 1.0)
    return p


def _lrd_signflip_ci(x, lo: float = -3.0, hi: float = 3.0, n: int = 1201) -> tuple:
    """Exact sign-flip 95 % interval of the mean of *x* by test inversion on a grid."""
    x = np.asarray(x, dtype=np.float64)
    if not len(x) or not np.isfinite(x).all() or len(x) > stats._SL_ENUM_UNITS:
        return np.nan, np.nan
    S = stats._sl_signs(len(x))
    g = np.linspace(lo, hi, n)
    Y = x[None, :] - g[:, None]
    p = (np.abs(Y @ S.T) >= np.abs(Y.sum(axis=1))[:, None] - 1e-12).mean(axis=1)
    keep = g[p > 0.05]
    if not len(keep) or keep.min() <= lo or keep.max() >= hi:      # empty, or unbounded on the grid
        return np.nan, np.nan                                      # (fewer than 6 units: no p below 0.05)
    return float(keep.min()), float(keep.max())


def _lrd_omnibus(M, paired: bool, n_a: int = 0, seed: int = 0) -> dict:
    """Omnibus over the columns (type pairs) of M for one term.

    paired: M = donors × pairs of per-donor log2 ratios; T = Σ_pairs t² with t = mean /
    (sd / √n), against every sign assignment of the donors (all columns of a donor flip
    together). Unpaired: M = units × pairs of per-unit log2 values, the first *n_a* rows
    condition A; t = Welch statistic, against the relabellings of the units. Also the
    shared shift (mean over the pairs per unit) with its own test. Returns {T, p, n_pairs,
    shared_shift, shared_p, exact}."""
    M = np.asarray(M, dtype=np.float64)
    M = M[:, np.isfinite(M).all(axis=0)] if M.size else M
    n, k = M.shape if M.ndim == 2 else (0, 0)
    out = {"T": np.nan, "p": np.nan, "n_pairs": int(k), "shared_shift": np.nan, "shared_p": np.nan, "exact": True}
    if not k or n < 2:
        return out
    if paired:
        def _T(Mx):
            mu, sd = Mx.mean(axis=-2), Mx.std(axis=-2, ddof=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.nansum((mu / (sd / np.sqrt(n))) ** 2, axis=-1)
        if n <= stats._SL_ENUM_UNITS:
            S = stats._sl_signs(n)
        else:
            S = np.random.default_rng(seed).integers(0, 2, size=(_LRD_MC, n)) * 2.0 - 1.0
            S[0] = 1.0
            out["exact"] = False
        obs = float(_T(M))
        null = np.concatenate([_T(S[a:a + 4096, :, None] * M[None, :, :]) for a in range(0, len(S), 4096)])
        out.update({"T": obs, "p": float((null >= obs - 1e-9).mean())})
        shared = M.mean(axis=1)
        out["shared_shift"] = float(shared.mean())
        out["shared_p"] = float(_lrd_signflip_matrix(shared[:, None])[0])
        return out
    n_b = n - n_a
    if n_a < 2 or n_b < 2:
        return out
    idx, exact = _lrd_assignments(n_a, n_b, seed=seed)
    out["exact"] = exact

    def _T(mask):                                           # mask: assignments × units (True = "A")
        res = np.zeros(len(mask))
        for a in range(0, len(mask), 2048):
            mk = mask[a:a + 2048, :, None]
            A = np.where(mk, M[None], np.nan)
            Bm = np.where(~mk, M[None], np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((np.nanmean(A, axis=1) - np.nanmean(Bm, axis=1))
                     / np.sqrt(np.nanvar(A, axis=1, ddof=1) / n_a + np.nanvar(Bm, axis=1, ddof=1) / n_b))
            res[a:a + 2048] = np.nansum(np.where(np.isfinite(t), t, 0.0) ** 2, axis=1)
        return res

    obs_mask = np.zeros((1, n), dtype=bool)
    obs_mask[0, :n_a] = True
    masks = np.zeros((len(idx), n), dtype=bool)
    masks[np.arange(len(idx))[:, None], idx] = True
    obs = float(_T(obs_mask)[0])
    null = _T(masks)
    hits = float((null >= obs - 1e-9).sum())
    out.update({"T": obs, "p": hits / len(null) if exact else (hits + 1.0) / (len(null) + 1.0)})
    shared = M.mean(axis=1)
    out["shared_shift"] = float(shared[:n_a].mean() - shared[n_a:].mean())
    out["shared_p"] = float(_lrd_permutation_matrix(shared[:n_a, None], shared[n_a:, None], (idx, exact))[0])
    return out


# ─── From the sums of the samples to tables ───────────────────────────────────
def _lrd_decompose(sums, types, kept, conds=None, donors=None, binary: bool = False, own: float = _LRD_OWN,
                   floor: float = _LRD_FLOOR, min_cells: int = _LRD_MIN_CELLS, names=None) -> dict:
    """Tables of the decomposition from the per-sample sums (see _lrd_sample_sums).

    conds / donors: the sample design ("A" / "B" / "" per sample; donor labels). Returns
    {table (one row per interaction: sender, receiver, ligand, receptor and, per mode, the
    terms or their log2 ratios with p-values), pairs (one row per sender / receiver pair
    and term), omnibus ({term: …}), mode ("descriptive" / "ratio" / "paired" / "unpaired"),
    reason (why there are no p-values), units_note, min_p, n_dropped}."""
    T, P = len(types), len(kept)
    G = list(dict.fromkeys([g for pr in kept for g in pr]))
    gpos = {g: i for i, g in enumerate(G)}
    li = np.array([gpos[a] for a, _ in kept], dtype=np.int64)
    ri = np.array([gpos[b] for _, b in kept], dtype=np.int64)
    n_s = len(sums)
    conds = list(conds) if conds is not None else [""] * n_s
    donors = [str(d or "") for d in (donors if donors is not None else [""] * n_s)]
    have = [i for i in range(n_s) if sums[i] is not None]
    if not have:
        raise ValueError("No sample of the active view has labelled cells with counts.")
    names = names or {"A": "A", "B": "B"}
    both = any(conds[i] == "A" for i in have) and any(conds[i] == "B" for i in have)
    cu = [conds[i] if (i in have and conds[i] in ("A", "B")) else "" for i in range(n_s)]
    out = {"omnibus": {}, "pairs": pd.DataFrame(), "min_p": np.nan, "n_units": 0}
    if both:
        u = stats._sl_units(cu, donors)
        if u["paired"]:
            unit_ix = [(d, "A", ia) for d, ia, _ib in u["pairs"]] + [(d, "B", ib) for d, _ia, ib in u["pairs"]]
            k = len(u["pairs"])
        else:
            unit_ix = ([(nm, c, ix) for nm, c, ix in u["units"] if c == "A"]
                       + [(nm, c, ix) for nm, c, ix in u["units"] if c == "B"])
            k = min(sum(1 for x in unit_ix if x[1] == "A"), sum(1 for x in unit_ix if x[1] == "B"))
        out["units_note"] = u["note"]
        if u["dropped"]:
            out["units_note"] += ("; donors with samples in one condition only were left out: "
                                  + ", ".join(u["dropped"]))
        if k >= stats._SL_MIN_UNITS:
            mode, reason = ("paired" if u["paired"] else "unpaired"), ""
        else:
            mode = "ratio"
            reason = (f"no p-values: fewer than {stats._SL_MIN_UNITS} independent units per condition "
                      f"({u['note']}); the log2 ratios A / B of the pooled conditions are shown descriptively")
        used = sorted({i for _n, _c, ix in unit_ix for i in ix})
    else:
        mode = "descriptive"
        used = have
        n_cond = len({conds[i] for i in have if conds[i] in ("A", "B")})
        reason = ("no p-values: " + (("one sample is loaded" if n_s == 1 else "only one sample has labelled cells")
                                     if len(have) == 1 else
                                     ("the samples are all in one condition" if n_cond == 1 else
                                      "no sample is assigned to conditions A and B (Samples & conditions…)"))
                  + "; a difference needs two conditions with independent units. The terms of the "
                    f"{len(have)} sample(s) pooled are shown descriptively")
        unit_ix = [("", "", [i]) for i in have]
        out["units_note"] = f"n = {len(have)} sample(s), pooled"
    out.update({"mode": mode, "reason": reason, "samples_used": used})
    pooled = _lrd_pool([sums[i] for i in used])
    units = [_lrd_pool([sums[i] for i in ix]) for _n, _c, ix in unit_ix]
    # filters: enough cells of both types and the transcript floor in every unit; ownership pooled
    n_min = np.min([x["n_cells"] for x in units], axis=0)                       # [T]
    x_min = np.min([x["sum_x"] for x in units], axis=0)                         # [T, G]
    c_min = np.min([x["n_contacts"] for x in units], axis=0)                    # [S, T]
    sh_l, sh_r = _lrd_ownership(pooled, li, ri)
    cells_ok = (n_min[:, None] >= min_cells) & (n_min[None, :] >= min_cells) & (c_min > 0)
    floor_ok = (x_min[:, li][:, None, :] >= floor) & (x_min[:, ri][None, :, :] >= floor)
    own_ok = (sh_l[:, None, :] >= own) & (sh_r[None, :, :] >= own)
    cand = cells_ok[:, :, None] & np.ones((1, 1, P), dtype=bool)
    out["n_candidates"] = int(cand.sum())
    out["n_below_floor"] = int((cand & ~floor_ok).sum())
    out["n_not_owned"] = int((cand & floor_ok & ~own_ok).sum())
    keep = cand & floor_ok & own_ok
    tp = _lrd_terms(pooled, li, ri, binary)
    all_terms = list(_LRD_TERMS) + ["exposure"]
    keep &= np.isfinite(tp["exposure"]) & (tp["exposure"] > 0)
    for t in _LRD_TERMS:
        keep &= np.isfinite(tp[t]) & (tp[t] > 0)
    D = {}
    if mode != "descriptive":
        tu = [_lrd_terms(x, li, ri, binary) for x in units]
        ia = [j for j, x in enumerate(unit_ix) if x[1] == "A"]
        ib = [j for j, x in enumerate(unit_ix) if x[1] == "B"]
        if mode == "ratio":
            ta = _lrd_terms(_lrd_pool([sums[i] for j in ia for i in unit_ix[j][2]]), li, ri, binary)
            tb = _lrd_terms(_lrd_pool([sums[i] for j in ib for i in unit_ix[j][2]]), li, ri, binary)
            with np.errstate(divide="ignore", invalid="ignore"):
                for t in all_terms:
                    D[t] = np.log2(ta[t] / tb[t])[None]
                D["oe_obs"] = np.log2((ta["oe_obs"] + _LRD_EPS) / (tb["oe_obs"] + _LRD_EPS))[None]
                D["oe_enrich"] = np.log2(ta["oe_enrich"] / tb["oe_enrich"])[None]
            LA = LB = None
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                lg = {t: np.stack([np.log2(x[t]) for x in tu]) for t in all_terms + ["oe_enrich"]}
                lg["oe_obs"] = np.stack([np.log2(x["oe_obs"] + _LRD_EPS) for x in tu])
            LA = {t: v[ia] for t, v in lg.items()}
            LB = {t: v[ib] for t, v in lg.items()}
            with np.errstate(invalid="ignore"):
                if mode == "paired":                       # units are ordered donor by donor in both arms
                    D = {t: LA[t] - LB[t] for t in lg}
                else:
                    D = {t: (LA[t].mean(axis=0) - LB[t].mean(axis=0))[None] for t in lg}
        fin = np.ones(keep.shape, dtype=bool)
        for t in all_terms:
            fin &= np.isfinite(D[t]).all(axis=0)
            if LA is not None:
                fin &= np.isfinite(LA[t]).all(axis=0) & np.isfinite(LB[t]).all(axis=0)
        out["n_incomplete"] = int((keep & ~fin).sum())
        keep &= fin
    S_i, T_i, P_i = np.nonzero(keep)
    out["n_dropped"] = int(cand.sum() - keep.sum())
    tab = pd.DataFrame({"sender": [types[i] for i in S_i], "receiver": [types[i] for i in T_i],
                        "ligand": [kept[i][0] for i in P_i], "receptor": [kept[i][1] for i in P_i],
                        "n_receivers": pooled["n_cells"][T_i].astype(np.int64),
                        "n_senders": pooled["n_cells"][S_i].astype(np.int64),
                        "n_contacts": pooled["n_contacts"][S_i, T_i].astype(np.int64),
                        "ligand_share": sh_l[S_i, P_i], "receptor_share": sh_r[T_i, P_i]})
    sel = (S_i, T_i, P_i)
    for t in all_terms + ["oe_obs", "oe_exp", "oe_enrich"]:
        tab[t] = tp[t][sel]
    if mode != "descriptive":
        for c, nm in (("A", "a"), ("B", "b")):
            tc = _lrd_terms(_lrd_pool([sums[i] for _n, cc, ix in unit_ix if cc == c for i in ix]), li, ri, binary)
            tab[f"exposure_{nm}"] = tc["exposure"][sel]
            tab[f"oe_enrich_{nm}"] = tc["oe_enrich"][sel]
        for t in all_terms + ["oe_obs", "oe_enrich"]:
            d = D[t][(slice(None),) + sel]                                  # units × interactions
            tab[f"d_{t}"] = d.mean(axis=0)
            if mode == "paired":
                tab[f"n_up_{t}"] = (d > 0).sum(axis=0)
                tab[f"p_{t}"] = _lrd_signflip_matrix(d)
            elif mode == "unpaired":
                tab[f"p_{t}"] = _lrd_permutation_matrix(LA[t][(slice(None),) + sel], LB[t][(slice(None),) + sel])
            if f"p_{t}" in tab.columns:
                tab[f"fdr_{t}"] = stats._bh(tab[f"p_{t}"].to_numpy())
        if len(tab):
            resid = np.abs(tab["d_exposure"] - tab[[f"d_{t}" for t in _LRD_TERMS]].sum(axis=1))
            assert float(resid.max()) < 1e-6, "exposure identity broken"
        # sender / receiver pairs: mean over the pair's interactions per unit, then the same test
        if len(tab):
            pid, pair_index = pd.factorize(pd.MultiIndex.from_arrays([tab["sender"], tab["receiver"]]))
            n_int = np.bincount(pid)
            H = sp.csr_matrix((1.0 / n_int[pid], (pid, np.arange(len(pid)))), shape=(len(n_int), len(pid)))
            rows = []
            hetero = np.array([a != b for a, b in pair_index])
            for t in all_terms:
                d = (H @ D[t][(slice(None),) + sel].T).T                    # units × type pairs
                row = pd.DataFrame({"sender": [a for a, _ in pair_index], "receiver": [b for _, b in pair_index],
                                    "component": t, "mean": d.mean(axis=0), "n_interactions": n_int})
                if mode == "paired":
                    row["p"] = _lrd_signflip_matrix(d)
                    row["n_up"] = (d > 0).sum(axis=0)
                    ci = [_lrd_signflip_ci(d[:, j]) for j in range(d.shape[1])]
                    row["lo"], row["hi"] = [c[0] for c in ci], [c[1] for c in ci]
                    out["omnibus"][t] = _lrd_omnibus(d[:, hetero], True)
                elif mode == "unpaired":
                    a = (H @ LA[t][(slice(None),) + sel].T).T
                    b = (H @ LB[t][(slice(None),) + sel].T).T
                    row["p"] = _lrd_permutation_matrix(a, b)
                    out["omnibus"][t] = _lrd_omnibus(np.vstack([a, b])[:, hetero], False, n_a=len(a))
                rows.append(row)
            out["pairs"] = pd.concat(rows, ignore_index=True)
        if mode == "paired":
            r = stats._sl_signflip(np.ones(len(ia)))
            out["min_p"], out["n_units"], out["test"] = r["min_p"], len(ia), "exact paired sign-flip"
            out["exact"] = r["exact"]
        elif mode == "unpaired":
            r = stats._sl_permutation(np.ones(len(ia)), np.zeros(len(ib)), enum_max=_LRD_ENUM_ASSIGN, n_mc=_LRD_MC)
            out["min_p"], out["n_units"], out["exact"] = r["min_p"], len(ia) + len(ib), r["exact"]
            out["test"] = ("exact unpaired label permutation" if r["exact"]
                           else f"unpaired label permutation ({_LRD_MC} random relabellings)")
    out["table"] = tab
    out["names"] = dict(names)
    return out


def _lrd_compute(samples, labels, pairs, conds=None, donors=None, radius=_LRD_RADIUS_UM, tile=_LRD_TILE_UM,
                 binary: bool = False, own: float = _LRD_OWN, floor: float = _LRD_FLOOR,
                 min_cells: int = _LRD_MIN_CELLS, max_types: int = _LRD_MAX_TYPES, names=None,
                 progress=None) -> dict:
    """The decomposition for analysis samples: sums per sample, then _lrd_decompose."""
    genes_all = analysis._an_common_genes(samples)
    kept = _lrd_match_pairs(pairs, genes_all)
    if not kept:
        raise ValueError(f"None of the {len(pairs)} ligand–receptor pairs has both genes among the "
                         f"{len(genes_all):,} measured genes of this view.")
    parts = [pd.Series(l, dtype=object).dropna().astype(str) for l in labels if l is not None]
    vc = pd.concat(parts).value_counts() if parts else pd.Series(dtype=np.int64)
    vc = vc[vc >= min_cells]
    dropped = list(map(str, vc.index[max_types:]))
    types = sorted(map(str, vc.index[:max_types]), key=str)
    if not types:
        raise ValueError(f"No cell type has at least {min_cells} cells.")
    sums, n_adj = [], []
    for i, (s, lab) in enumerate(zip(samples, labels)):
        jobs._job_check()
        sm = _lrd_sample_sums(s, lab, kept, types, radius, tile, binary,
                              progress=(None if progress is None else
                                        (lambda f, _i=i: progress((_i + f) / len(samples)))))
        sums.append(sm)
        n_adj.append(0 if sm is None else sm["n_adj"])
    res = _lrd_decompose(sums, types, kept, conds, donors, binary, own, floor, min_cells, names)
    res.update({"types": types, "types_dropped": dropped, "pairs_kept": kept, "n_pairs_input": len(pairs),
                "n_adj": n_adj, "n_cells": [0 if x is None else int(x["n_cells"].sum()) for x in sums],
                "radius": float(radius), "tile": float(tile) if tile else None, "binary": bool(binary),
                "own": float(own), "floor": float(floor), "min_cells": int(min_cells)})
    return res


def _run_lrdecomp(pairs_text, label_kind, seg_source, method, use_corrected, radius=None, tile=None,
                  own=None, floor=None, binary=False, min_cells=None) -> None:
    """Background job: ligand-receptor exposure decomposition for the active view."""
    _gen = jobs._dataset_gen

    def _set(status, message, result=None):
        with _lrd_lock:
            if jobs._stale_job(_gen, status, message):
                return
            _lrd_state["status"], _lrd_state["message"] = status, message
            if result is not None:
                _lrd_state["result"] = result
            if status in ("done", "error"):
                _lrd_state["version"] += 1
        if status == "done":
            provenance._prov_record("lrdecomp", seg_source=seg_source, corrected=use_corrected, params={"pairs": pairs_text, "radius": radius, "tile": tile, "own": own, "floor": floor, "binary": binary, "min_cells": min_cells}, labels=(label_kind, method), uses_design=True, message=message, result=result)

    try:
        _t0 = _time.time()
        _set("running", "Collecting cells, labels and contacts…")
        pairs, n_skip = stats._lr_parse_pairs(pairs_text)
        if not pairs:
            _set("error", "No ligand–receptor pairs: paste or upload lines of 'ligand,receptor'.")
            return
        samples = analysis._an_samples(seg_source, use_corrected)
        if not samples:
            _set("error", analysis._an_no_matrix_msg(seg_source))
            return
        labels, cat = stats._comp_labels(label_kind, samples, seg_source, method, use_corrected)
        if all(l is None for l in labels):
            _set("error", f"The active view has no {cat} labels: run Cell Type Annotation first, or "
                          f"decompose between clusters.")
            return
        radius = float(max(1.0, _LRD_RADIUS_UM if radius is None else radius))
        tile = _LRD_TILE_UM if tile is None else float(tile)
        tile = tile if tile > 0 else None
        own = float(min(1.0, max(0.0, _LRD_OWN if own is None else own)))
        floor = float(max(0.0, _LRD_FLOOR if floor is None else floor))
        min_cells = int(max(2, _LRD_MIN_CELLS if min_cells is None else min_cells))
        conds, _b = stats._design_of(samples)
        donors = stats._design_donors(samples)
        names = stats._design_names()
        res = _lrd_compute(samples, labels, pairs, conds, donors, radius, tile, bool(binary), own, floor,
                           min_cells, names=names,
                           progress=lambda f: jobs._job_progress("Summing over contacts…", 0.05 + 0.85 * f))
        jobs._check_dataset_gen(_gen)
        corrected = bool(samples[0]["corrected"])
        res.update({"gen": _gen, "category": cat, "corrected": corrected, "seg_source": seg_source or "xenium",
                    "n_uncorrected": analysis._an_uncorrected(samples),
                    "n_samples": len(samples), "n_skipped_lines": n_skip, "seconds": _time.time() - _t0,
                    "samples": [{"label": s["label"], "name": s["name"], "cond": c, "donor": d,
                                 "n_cells": k, "n_adj": a, "used": i in res["samples_used"]}
                                for i, (s, c, d, k, a) in enumerate(zip(samples, conds, donors, res["n_cells"],
                                                                        res["n_adj"]))]})
        t = res["table"]
        n_tp = t[["sender", "receiver"]].drop_duplicates().shape[0] if len(t) else 0
        msg = (f"{len(res['pairs_kept']):,} of {res['n_pairs_input']:,} ligand–receptor pairs have both genes on "
               f"the panel. {len(res['types'])} groups ({cat}), {sum(res['n_cells']):,} labelled cells in "
               f"{len(samples)} sample(s); contacts within {radius:g} µm in the same sample "
               f"({sum(res['n_adj']):,} cell pairs); tiles of "
               f"{(f'{tile:g} µm' if tile else 'the whole sample')}; rates: "
               f"{'detection (count ≥ 1) / cells' if binary else 'Σ counts / Σ depth'}, "
               f"{'SPLIT-corrected' if corrected else 'original'} counts ({res['seconds']:.1f} s)."
               f"\n{len(t):,} interactions in {n_tp} sender → receiver pairs kept of "
               f"{res['n_candidates']:,} candidates (both types ≥ {min_cells} cells and in contact in every "
               f"unit): {res['n_below_floor']:,} below the floor of {floor:g} transcripts in a unit, "
               f"{res['n_not_owned']:,} with a gene its type does not own (< {own:g} of the top type's rate)"
               + (f", {res.get('n_incomplete', 0):,} with an undefined term in a unit" if res["mode"] != "descriptive" else "")
               + ". Exposure = K × R × L × γ × π exactly.")
        if not len(t):
            msg += (" No interaction passed: lower the floor (it applies to the shallowest unit), the "
                    "ownership share or the minimum number of cells.")
        nA, nB = names.get("A", "A"), names.get("B", "B")
        if res["mode"] in ("paired", "unpaired"):
            n_sig = int((t["fdr_exposure"] < 0.05).sum()) if len(t) else 0
            msg += (f"\nlog2 ratios A ({nA}) / B ({nB}); {res['test']} per term; unit of the p-values: "
                    f"{res['units_note']} (not cells); smallest achievable p {res['min_p']:.3g}; "
                    f"{n_sig:,} interactions with exposure FDR < 0.05.")
            reach = stats._sl_reach(res["min_p"], len(t))
            if reach:
                msg += f"\n⚠ {reach[0].upper() + reach[1:]}."
            om = res["omnibus"]
            if om:
                msg += ("\nOmnibus over the between-type pairs (Σ t², " + ("exact" if om["exposure"]["exact"] else "Monte Carlo")
                        + f"; {om['exposure']['n_pairs']} pairs): "
                        + "; ".join(f"{_LRD_TERM_NAMES[k].split(' (')[0].lower()} p {v['p']:.3g} (shared shift "
                                    f"{v['shared_shift']:+.2f}, p {v['shared_p']:.3g})" for k, v in om.items()) + ".")
        elif res["mode"] == "ratio":
            msg += f"\nlog2 ratios A ({nA}) / B ({nB}) of the pooled conditions. ⚠ {res['reason'][0].upper() + res['reason'][1:]}."
        else:
            msg += f"\n⚠ {res['reason'][0].upper() + res['reason'][1:]}."
        if use_corrected and not corrected:
            msg += " Original counts were used: not every sample has SPLIT-corrected counts."
        if res["n_uncorrected"]:
            msg += " " + analysis._an_uncorrected_note(res["n_uncorrected"]) + "."
        if res["types_dropped"]:
            msg += (f"\nOnly the {_LRD_MAX_TYPES} most abundant groups were used; left out: "
                    f"{', '.join(res['types_dropped'][:10])}{'…' if len(res['types_dropped']) > 10 else ''}.")
        if label_kind == "cluster" and len(samples) > 1:
            msg += " Cluster ids were taken as the same clusters in every sample."
        msg += ("\nThis explains why an exposure differs (expression vs contact vs geometry); it is not "
                "evidence of signalling.")
        _set("done", msg, result=res)
        print(f"  LR decomposition: {msg}", flush=True)
    except jobs._JobCancelled:
        raise
    except jobs._DatasetChanged:
        raise
    except ValueError as exc:
        _set("error", str(exc))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set("error", str(exc)[:300])


# ─── Figure, table, CSV ───────────────────────────────────────────────────────
def _lrd_level_frame(res, level: str = "interactions") -> pd.DataFrame:
    """Rows of the plot / table at the chosen level, strongest first. Type-pair level (only
    with ratios): one row per sender → receiver with the mean log2 ratio of every term."""
    t = res["table"]
    if level == "pairs" and res["mode"] != "descriptive" and len(res["pairs"]):
        w = res["pairs"].pivot_table(index=["sender", "receiver"], columns="component", values="mean")
        out = w.rename(columns={c: f"d_{c}" for c in w.columns}).reset_index()
        n = res["pairs"].drop_duplicates(["sender", "receiver"]).set_index(["sender", "receiver"])["n_interactions"]
        out["n_interactions"] = n.reindex(pd.MultiIndex.from_frame(out[["sender", "receiver"]])).to_numpy()
        if "p" in res["pairs"].columns:
            pw = res["pairs"].pivot_table(index=["sender", "receiver"], columns="component", values="p")
            for c in pw.columns:
                out[f"p_{c}"] = pw[c].to_numpy()
            out["fdr_exposure"] = stats._bh(out["p_exposure"].to_numpy())
        out["label"] = out["sender"] + " → " + out["receiver"]
        return out.reindex(out["d_exposure"].abs().sort_values(ascending=False, kind="stable").index)
    t = t.assign(label=t["ligand"] + " → " + t["receptor"] + "  |  " + t["sender"] + " → " + t["receiver"])
    if res["mode"] == "descriptive":
        return t.sort_values("exposure", ascending=False, kind="stable")
    key = "p_exposure" if "p_exposure" in t.columns else None
    t = t.assign(_abs=-t["d_exposure"].abs())
    return t.sort_values(([key] if key else []) + ["_abs"], kind="stable").drop(columns="_abs")


def make_lrd_fig(level="interactions", top_n=25, width=None, height=None):
    """Stacked diverging bars of the term contributions. With two conditions: log2 A / B of
    the five terms per interaction (they sum to the log2 ratio of the exposure, the
    diamond). One condition: log2 of the two dimensionless factors γ and π."""
    res = _lrd_result()
    if not res:
        return figures._make_placeholder_fig("No exposure decomposition computed yet.")
    if not len(res["table"]):
        return figures._make_placeholder_fig("No interaction passed the filters (floor, ownership, cells).")
    f = _lrd_level_frame(res, level).head(int(max(1, top_n or 25)))
    f = f.iloc[::-1]
    fig = go.Figure()
    nA, nB = res["names"].get("A", "A"), res["names"].get("B", "B")
    if res["mode"] == "descriptive":
        cd = np.column_stack([f[c] for c in ("contact", "receptor_expr", "ligand_expr", "regional", "preference",
                                             "exposure", "oe_enrich")])
        for t in ("regional", "preference"):
            fig.add_trace(go.Bar(y=f["label"], x=np.log2(f[t]), orientation="h", name=_LRD_TERM_NAMES[t],
                                 marker_color=_LRD_TERM_COLORS[t], customdata=cd,
                                 hovertemplate=("<b>%{y}</b><br>K %{customdata[0]:.3g} contacts / receiver<br>"
                                                "R %{customdata[1]:.3g} · L %{customdata[2]:.3g}<br>"
                                                "γ %{customdata[3]:.3g} · π %{customdata[4]:.3g}<br>"
                                                "exposure %{customdata[5]:.3g}<br>O/E %{customdata[6]:.3g}"
                                                "<extra></extra>")))
        title = f"log2 γ and π of the top {len(f)} interactions by exposure (descriptive, no p-values)"
        x_title = "log2 factor (0 = no co-location / preference beyond expression × contact)"
    else:
        for t in _LRD_TERMS:
            fig.add_trace(go.Bar(y=f["label"], x=f[f"d_{t}"], orientation="h", name=_LRD_TERM_NAMES[t],
                                 marker_color=_LRD_TERM_COLORS[t],
                                 hovertemplate="<b>%{y}</b><br>" + _LRD_TERM_NAMES[t] + ": log2 %{x:+.3f}<extra></extra>"))
        star = ""
        if "p_exposure" in f.columns:
            star = np.where(f["p_exposure"].to_numpy() < 0.05, " *", "")
        fig.add_trace(go.Scatter(y=f["label"], x=f["d_exposure"], mode="markers+text", name="Exposure (sum)",
                                 marker=dict(symbol="diamond", size=9, color=core.TEXT,
                                             line=dict(width=1, color="#0d1117")),
                                 text=star if len(star) else None, textposition="middle right",
                                 textfont=dict(color=core.TEXT, size=12),
                                 customdata=(f["p_exposure"] if "p_exposure" in f.columns else np.full(len(f), np.nan)),
                                 hovertemplate="<b>%{y}</b><br>exposure: log2 %{x:+.3f}<br>p %{customdata:.3g}<extra></extra>"))
        how = {"paired": "* p < 0.05, exact sign-flip across donors", "unpaired": "* p < 0.05, label permutation across units",
               "ratio": "descriptive, no p-values"}[res["mode"]]
        title = (f"log2 A ({nA}) / B ({nB}) by term, top {len(f)} "
                 f"{'type pairs' if level == 'pairs' else 'interactions'} ({how})")
        x_title = "log2 ratio A / B (bars sum to the exposure change, ◆)"
    fig.update_layout(barmode="relative", paper_bgcolor=core.CARD_BG, plot_bgcolor=core.CARD_BG,
                      font=dict(color=core.TEXT, size=10), margin=dict(l=220, r=10, t=88, b=44),
                      title=dict(text=title, font=dict(size=12), x=0.01, y=0.98, yanchor="top"),
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, font=dict(size=10),
                                  traceorder="normal"),
                      xaxis=dict(title=x_title, gridcolor=core.BORDER, zerolinecolor=core.MUTED),
                      yaxis=dict(gridcolor=core.BORDER, automargin=True, tickfont=dict(size=9)))
    if width:
        fig.update_layout(width=width)
    if height:
        fig.update_layout(height=height)
    return fig


def _lrd_columns(res=None, level="interactions") -> list:
    _text = ("sender", "receiver", "ligand", "receptor")
    _col = lambda cid, name: {"id": cid, "name": name, "type": "text" if cid in _text else "numeric"}
    head = [_col("sender", "Sender"), _col("receiver", "Receiver")]
    if res is None or res["mode"] == "descriptive":
        return head + [_col("ligand", "Ligand"), _col("receptor", "Receptor"), _col("contact", "K"),
                       _col("receptor_expr", "R /10k"), _col("ligand_expr", "L /10k"), _col("regional", "γ"),
                       _col("preference", "π"), _col("exposure", "Exposure"), _col("oe_obs", "Obs."),
                       _col("oe_exp", "Exp."), _col("oe_enrich", "O/E")]
    pairs = level == "pairs"
    cols = head + ([_col("n_interactions", "Interactions")] if pairs else
                   [_col("ligand", "Ligand"), _col("receptor", "Receptor")])
    cols += [_col("d_exposure", "Δ exposure")] + [_col(f"d_{t}", f"Δ {s}") for t, s in
                                                  zip(_LRD_TERMS, ("K", "R", "L", "γ", "π"))]
    if res["mode"] in ("paired", "unpaired"):
        cols += [_col("p_exposure", "p (exposure)"), _col("fdr_exposure", "FDR")]
        cols += [_col(f"p_{t}", f"p ({s})") for t, s in zip(_LRD_TERMS, ("K", "R", "L", "γ", "π"))]
    if not pairs:
        cols += [_col("oe_enrich_a", "O/E A"), _col("oe_enrich_b", "O/E B"), _col("d_oe_enrich", "Δ O/E")]
        if res["mode"] in ("paired", "unpaired"):
            cols.append(_col("p_oe_enrich", "p (O/E)"))
    return cols


def _lrd_records(res, level="interactions") -> list:
    if not res or not len(res["table"]):
        return []
    f = _lrd_level_frame(res, level).head(5000).copy()
    if not res.get("binary"):
        for c in ("receptor_expr", "ligand_expr"):
            if c in f.columns:
                f[c] = f[c] * analysis._AN_SCALE
    ids = [c["id"] for c in _lrd_columns(res, level)]
    out = pd.DataFrame({c: f[c] for c in ids if c in f.columns})
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda v: None if not np.isfinite(v) else float(f"{v:.3g}"))
    return out.astype(object).where(out.notna(), None).to_dict("records")


def _lrd_csv(res) -> str:
    nA, nB = res["names"].get("A", "A"), res["names"].get("B", "B")
    lines = [f"# Ligand-receptor exposure decomposition between groups ({res['category']}); segmentation: "
             f"{res['seg_source']}; {'SPLIT-corrected' if res['corrected'] else 'original'} counts; rates: "
             f"{'detection (count >= 1) / cells' if res['binary'] else 'sum of counts / sum of depth'}",
             f"# contacts: centroids within {res['radius']:g} um in the same sample; tiles: "
             f"{(str(res['tile']) + ' um') if res['tile'] else 'whole sample'}; kept: ligand owned by the sender and "
             f"receptor by the receiver (>= {res['own']:g} of the top type's rate), >= {res['floor']:g} transcripts "
             f"of each gene in its type and >= {res['min_cells']} cells of both types in every unit",
             "# exposure = contact (K, sender contacts per receiver) x receptor_expr (R) x ligand_expr (L) x "
             "regional (gamma, tile co-location) x preference (pi, contact preference), exactly",
             f"# oe_obs: share of receivers detecting the receptor with >= 1 ligand-positive sender in contact; "
             f"oe_exp: the same with ligand positivity random among the senders, 1 - (1 - p_L)^k; oe_enrich = "
             f"(obs + {_LRD_EPS:g}) / (exp + {_LRD_EPS:g})",
             f"# mode: {res['mode']}; unit: {res['units_note']}"
             + (f"; test: {res['test']}; smallest achievable p {res['min_p']:.4g}; d_* = mean log2 A ({nA}) / B ({nB}) "
                f"over the units; fdr_*: BH over the interactions" if res["mode"] in ("paired", "unpaired") else
                (f"; d_* = log2 A ({nA}) / B ({nB}) of the pooled conditions; {res['reason']}" if res["mode"] == "ratio"
                 else f"; {res['reason']}")),
             "# why an exposure differs (expression vs contact vs geometry); not evidence of signalling"]
    if res.get("n_uncorrected"):
        lines.append(f"# {analysis._an_uncorrected_note(res['n_uncorrected'])}")
    for smp in res["samples"]:
        lines.append(f"# sample {smp['label']} ({smp['name']}): {smp['n_cells']} labelled cells, {smp['n_adj']} "
                     f"contact pairs; condition {smp['cond'] or '-'}; donor {smp['donor'] or '-'}"
                     f"{'' if smp['used'] else '; not used'}")
    for k, v in (res.get("omnibus") or {}).items():
        lines.append(f"# omnibus {k}: T = {v['T']:.4g}, p = {v['p']:.4g} over {v['n_pairs']} between-type pairs; "
                     f"shared shift {v['shared_shift']:+.4g} (p = {v['shared_p']:.4g})")
    txt = "\n".join(lines) + "\n" + res["table"].to_csv(index=False)
    if len(res["pairs"]):
        txt += ("\n# sender -> receiver pairs: mean over the pair's interactions per unit, tested across the units\n"
                + res["pairs"].to_csv(index=False))
    return provenance._prov_csv_stamp(res, (txt))


# ─── Layout (rows added to the Ligand-Receptor tab by stats._lr_tab) ──────────
def _lrd_controls() -> list:
    """Rows of the Ligand–Receptor tab in the 'Exposure decomposition' mode."""
    S, L = analysis._AN_ROW_STYLE, analysis._AN_LABEL_STYLE
    _num = lambda value, id=None, **kw: dcc.Input(id=id, type="number", value=value, debounce=True,
                                                  style={**ui._PLOT_CTRL_INPUT, "width": "58px"}, **kw)
    return [
        html.Div([
            html.Span(f"Exposure decomposition (contact radius {_LRD_RADIUS_UM:g} µm; more under Options)",
                      style={**L, "flex": "1"}),
            dbc.Button("Run", id="lrd-run-btn", color="primary", size="sm", n_clicks=0,
                       style={"fontSize": "11px", "padding": "2px 12px"}),
        ], style=S),
        ui._options(html.Div([
            html.Span("Contact radius (µm):", style=L), _num(_LRD_RADIUS_UM, id="lrd-radius", min=1, max=500, step=1),
            html.Span("Tile (µm; 0 = whole sample):", style=L), _num(_LRD_TILE_UM, id="lrd-tile", min=0, max=10000, step=5),
            html.Span("Ownership ≥:", style=L), _num(_LRD_OWN, id="lrd-own", min=0, max=1, step=0.05),
            html.Span("Floor (transcripts):", style=L), _num(_LRD_FLOOR, id="lrd-floor", min=0, step=1),
            html.Span("Min. cells / type:", style=L), _num(_LRD_MIN_CELLS, id="lrd-min-cells", min=2, step=1),
            dcc.Dropdown(id="lrd-rates", value="counts", clearable=False,
                         options=[{"label": "Rates: Σ counts / Σ depth", "value": "counts"},
                                  {"label": "Rates: detection (≥ 1) / cells", "value": "binary"}],
                         style={"width": "210px", "fontSize": "11px"}),
        ], style=S), summary="Options (radius, tile, ownership, floor, minimum cells, rates)"),
        html.Div([
            dbc.Button("Samples & conditions…", id="lrd-design-open-btn", color="secondary", size="sm",
                       n_clicks=0, style={"fontSize": "11px", "padding": "2px 8px"}),
            html.Span(id="lrd-design-summary", style=L),
        ], style=S),
    ]


def _lrd_results() -> list:
    """Status, plot, table and note of the decomposition (the plot above the table: its row
    labels are long)."""
    L = analysis._AN_LABEL_STYLE
    return [
        html.Div(id="lrd-status", style={**analysis._AN_STATUS_STYLE, "whiteSpace": "pre-wrap"}),
        html.Div(dcc.Graph(id="lrd-plot", responsive=True,
                           figure=figures._make_placeholder_fig("No exposure decomposition computed yet."),
                           config={"displayModeBar": "hover",
                                   "toImageButtonOptions": {"format": "svg", "filename": "xenium_lr_decomposition"}},
                           style={"width": "100%", "height": "100%"}),
                 id="lrd-plot-container", style={"flexShrink": "0", "minWidth": "0", "height": "440px"}),
        html.Div([
            html.Div([
                dbc.Button("Download CSV", id="lrd-download-btn", color="secondary", size="sm",
                           n_clicks=0, style={"fontSize": "11px", "padding": "2px 8px"}),
                html.Span("Show:", style={**L, "marginLeft": "10px"}),
                dcc.RadioItems(id="lrd-level", value="interactions", inline=True,
                               options=[{"label": " interactions", "value": "interactions"},
                                        {"label": " sender → receiver pairs", "value": "pairs"}],
                               inputStyle={"marginRight": "3px"},
                               labelStyle={"marginRight": "10px", "cursor": "pointer", "fontSize": "11px",
                                           "color": core.TEXT}),
                html.Span("Top:", style=L),
                dcc.Input(id="lrd-top-n", type="number", min=1, max=200, step=1, value=25,
                          debounce=True, style=ui._PLOT_CTRL_INPUT),
                dcc.Download(id="lrd-download"),
            ], style={"padding": "4px 0", "display": "flex", "alignItems": "center", "gap": "4px",
                      "flexWrap": "wrap"}),
            dash_table.DataTable(id="lrd-table", data=[], columns=_lrd_columns(),
                                 **stats._dark_table_style(12)),
        ], style={"flexShrink": "0", "minWidth": "0", "padding": "0 8px"}),
        ui._plot_dim_controls("lrd"),
        html.Div(_LRD_NOTE, id="lrd-note", style=analysis._AN_NOTE_STYLE),
    ]
