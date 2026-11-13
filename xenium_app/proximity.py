"""Proximity differential expression (gene-wise, near vs far from an anchor cell type) and
contact-group pseudobulk differential expression (compute + the controls of the two
modes in the Differential Expression tab).

Both ask what changes in cells of one type T next to cells of another type X, and both
publish into the Differential Expression state (analysis._de_state), so the volcano, the
table and the CSV of that tab show them. Every distance and neighbour count is taken
within ONE sample: each KD-tree holds the cells of one sample.

Reimplemented in the explorer's structure from an earlier spatial study by the author;
every value that was fixed there is a parameter here (the defaults are the generic ones).
"""

import time as _time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from dash import dcc, html
import dash_bootstrap_components as dbc

from . import core, jobs, ui, analysis, stats, controls

# ─── Parameters ───────────────────────────────────────────────────────────────
_PROX_NEAR_UM = 25.0          # near: nearest anchor at most this far
_PROX_FAR_UM = 100.0          # far: nearest anchor at least this far
_PROX_MIN_DETECT = 0.02       # a gene is tested when at least this share of the T cells detects it
_PROX_N_BINS = 10             # quantile bins of the local source fraction (covariate)
_PROX_MIN_ANCHORS = 5         # anchor cells a sample needs
_PROX_MIN_NEAR = 5            # near cells below which a sample has no near − far effect
_PROX_MIN_FAR = 50
_PROX_MIN_SLOPE_N = 50        # cells below which a sample has no slope
_PROX_MIN_TARGET = 100        # T cells a sample needs
_PROX_N_RANDOM = 99           # matched random anchor sets
_PROX_SEED = 0
_PROX_REPRO_FRAC = 5.0 / 7.0  # a placebo anchor type "reproduces" a gene at this share of same-sign units
_PROX_MAX_PLACEBO = 3         # other anchor types of similar abundance that are run
_PROX_L_GRID = (2.0, 5.0, 8.0, 12.0, 18.0, 25.0, 35.0, 50.0, 70.0, 100.0, 140.0, 200.0, 300.0, 400.0, 600.0)
_PROX_SET_BYTES = 160e6       # dense weights handled at a time (anchor sets × cells)

_CONTACT_R_UM = 50.0          # contact: an X cell within this distance (centroid to centroid)
_CONTACT_MIN_CELLS = 20       # cells both groups need in a sample for it to be used
_CONTACT_N_SHUFFLE = 199
_CONTACT_LFC_MIN = 0.5        # a gene "passes" at FDR < _CONTACT_FDR, |log2FC| > this and no leak flag
_CONTACT_FDR = 0.05

_PROX_NOTE = (
    "Proximity to a cell type: which genes of the cells T change near cells of the anchor type X? "
    "Per sample, log1p(count / library size × 10,000) of every T cell is residualised (OLS) on "
    "log1p(library size) and on decile bins of the local fraction of the density-covariate type "
    "(cells within 50 µm; by default the type that contributes the most transcripts, the main "
    "source of bleed and of generic location effects). Effect = mean residual of the T cells with "
    "the nearest X cell within the near distance minus that of the T cells with none closer than "
    "the far distance (shown in log2 units), and the slope of the residual on −log(d + 1). "
    "Matched random anchors: sets of as many non-X cells as there are X cells, drawn within the "
    "same compartment × edge-distance stratum (Confound Controls tab), go through the same "
    "computation; the empirical FDR of a gene is the number of genes the random sets yield at its "
    "threshold divided by the number observed. With at least 3 independent units (donors, or "
    "samples) the p-value is the exact sign-flip test of the per-unit effects; with fewer it is "
    "the share of random anchor sets at least as far from their mean as the observed effect, "
    "which describes these sections only. Specific = not reproduced by other anchor types of "
    "similar abundance. Genes owned by X and not by T, and genes the leak gate flags for T near vs far, are "
    "drawn as flagged: next to X, T cells pick up X transcripts, which is not regulation. "
    "Optional: per gene, the length L of residual ~ exp(−d / L) that fits best. A gradient with "
    "distance is an association with position, not an effect of X.")
_CONTACT_NOTE = (
    "Contact groups: do T cells with an X cell within the radius differ from T cells without? "
    "Within each sample the comparator cells are weighted so that their mix of compartment × "
    "edge-distance bin × density quintile equals that of the contact cells (contact cells in a "
    "stratum without comparator are left out), so location alone cannot make the difference. "
    "Pseudobulk = Σ weight × counts per sample × group (rounded to whole counts), fitted with "
    "~ sample + group in R (DESeq2, or edgeR when it is not installed); the unit of the p-values "
    "is the sample. Needs at least 2 samples with both groups above the minimum cell count. Label "
    "shuffle: the contact / comparator labels are permuted within sample × stratum, the weights "
    "and the fit are redone, and the number of genes passing (FDR, |log2FC|, not flagged) is "
    "compared with the observed number (every shuffle is a model fit of a few seconds; Cancel "
    "stops it). Leak gate as in the other tests; genes owned by X and not by T are "
    "flagged as well (a T cell in contact with X carries X transcripts).")


# ─── Small numerical pieces ───────────────────────────────────────────────────
def _prox_quantile_bins(v, n_bins: int = _PROX_N_BINS) -> np.ndarray:
    """Quantile bin of every value: edges at the n_bins + 1 quantiles (duplicates merged),
    bin = the edge interval the value falls in, the last one closed."""
    v = np.asarray(v, dtype=np.float64)
    if not len(v):
        return np.zeros(0, dtype=np.int64)
    edges = np.unique(np.quantile(v, np.linspace(0, 1, int(n_bins) + 1)))
    if len(edges) < 2:
        return np.zeros(len(v), dtype=np.int64)
    return np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2).astype(np.int64)


def _prox_design(log_depth, bins=None) -> np.ndarray:
    """Design matrix of the residualisation: constant, log depth, bin dummies (first bin
    dropped)."""
    log_depth = np.asarray(log_depth, dtype=np.float64)
    cols = [np.ones(len(log_depth)), log_depth]
    if bins is not None:
        bins = np.asarray(bins)
        for b in np.unique(bins)[1:]:
            cols.append((bins == b).astype(np.float64))
    return np.column_stack(cols)


def _prox_fit(Y, D) -> dict:
    """OLS of every gene (columns of the sparse cells × genes matrix Y) on the design D:
    {beta (p × G), rss (residual sum of squares per gene)}. The residuals themselves are
    never formed: every statistic used here is linear in them (see _prox_set_stats)."""
    Y = sp.csr_matrix(Y)
    DtY = np.asarray(Y.T @ D).T                       # p × G
    beta = np.linalg.pinv(D.T @ D) @ DtY
    yy = np.asarray(Y.multiply(Y).sum(axis=0)).ravel()
    rss = np.maximum(yy - (beta * DtY).sum(axis=0), 0.0)
    return {"beta": beta, "rss": rss}


def _prox_set_stats(Y, D, beta, sets, near=_PROX_NEAR_UM, far=_PROX_FAR_UM, min_near=_PROX_MIN_NEAR,
                    min_far=_PROX_MIN_FAR, min_slope_n=_PROX_MIN_SLOPE_N, check=None) -> dict:
    """Near − far effect and slope of the residuals for a list of anchor sets.

    sets: [(d, keep)] - distance of every cell to the nearest anchor of the set and the
    cells that count (an anchor is not its own target). With R = Y − D·beta: near − far =
    mean R over kept cells with d ≤ near minus mean R over kept cells with d ≥ far (NaN
    for a set with fewer than min_near / min_far such cells); slope = OLS slope of R on z
    = −log(d + 1) over the kept cells (NaN below min_slope_n cells or without spread).
    Returns {nearfar, slope (sets × genes), n_near, n_far, n}."""
    Y = sp.csr_matrix(Y)
    n, G = Y.shape
    K = len(sets)
    out = {"nearfar": np.full((K, G), np.nan), "slope": np.full((K, G), np.nan),
           "n_near": np.zeros(K, dtype=np.int64), "n_far": np.zeros(K, dtype=np.int64),
           "n": np.zeros(K, dtype=np.int64)}
    Yt = Y.T.tocsr()
    step = max(1, int(_PROX_SET_BYTES / (32.0 * max(n, 1))))
    for k0 in range(0, K, step):
        if check is not None:
            check()
        chunk = sets[k0:k0 + step]
        A = np.zeros((4 * len(chunk), n))
        sz = np.zeros(len(chunk)); szz = np.zeros(len(chunk))
        for i, (d, keep) in enumerate(chunk):
            d = np.asarray(d, dtype=np.float64)
            keep = np.ones(n, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
            z = -np.log(d + 1.0)
            A[4 * i] = keep & (d <= near)
            A[4 * i + 1] = keep & (d >= far)
            A[4 * i + 2] = keep
            A[4 * i + 3] = np.where(keep, z, 0.0)
            sz[i], szz[i] = z[keep].sum(), (z[keep] ** 2).sum()
        SR = np.asarray(Yt @ A.T).T - (A @ D) @ beta          # Σ weight × residual, 4·sets × G
        for i in range(len(chunk)):
            k = k0 + i
            nn, nf, nk = int(A[4 * i].sum()), int(A[4 * i + 1].sum()), int(A[4 * i + 2].sum())
            out["n_near"][k], out["n_far"][k], out["n"][k] = nn, nf, nk
            if nn >= min_near and nf >= min_far:
                out["nearfar"][k] = SR[4 * i] / nn - SR[4 * i + 1] / nf
            if nk >= min_slope_n:
                vz = szz[i] - sz[i] ** 2 / nk
                if vz > 0:
                    out["slope"][k] = (SR[4 * i + 3] - sz[i] * SR[4 * i + 2] / nk) / vz
    return out


def _prox_length_fit(Y, D, fit: dict, d, grid=_PROX_L_GRID) -> dict:
    """Per gene, the length L of the grid for which residual ~ a + b·exp(−d / L) has the
    smallest sum of squares, over ALL cells of Y: {L, amplitude (b), r2}. Uses Σ x·R and
    the residual sum of squares of the residualisation, so no dense residual is formed."""
    Y = sp.csr_matrix(Y)
    d = np.asarray(d, dtype=np.float64)
    X = np.exp(-d[None, :] / np.asarray(grid, dtype=np.float64)[:, None])          # L × n
    Xc = X - X.mean(axis=1, keepdims=True)
    sxx = (Xc ** 2).sum(axis=1)
    SxR = np.asarray(Y.T.tocsr() @ Xc.T).T - (Xc @ D) @ fit["beta"]                 # L × G (ΣR = 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        amp = np.where(sxx[:, None] > 0, SxR / sxx[:, None], 0.0)
    gain = amp * SxR                                                                # SST − SSE
    best = gain.argmax(axis=0)
    g = np.arange(Y.shape[1])
    rss = fit["rss"]
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(rss > 0, gain[best, g] / rss, np.nan)
    return {"L": np.asarray(grid, dtype=np.float64)[best], "amplitude": amp[best, g], "r2": r2}


def _prox_matched_random(pool_idx, pool_bin, need, n_draws: int, rng) -> tuple:
    """Count-matched random anchor sets: per draw and per stratum b, need[b] cells of the
    pool cells of that stratum without replacement (all of them when there are fewer; the
    missing cells are counted). Returns ([index arrays], shortfall)."""
    pool_idx, pool_bin = np.asarray(pool_idx), np.asarray(pool_bin)
    need = np.asarray(need, dtype=np.int64)
    by_bin = [pool_idx[pool_bin == b] for b in range(len(need))]
    draws, short = [], 0
    for _ in range(int(n_draws)):
        take = []
        for b in range(len(need)):
            avail = by_bin[b]
            k = min(int(need[b]), avail.size)
            short += int(need[b]) - k
            if k > 0:
                take.append(rng.choice(avail, k, replace=False))
        draws.append(np.concatenate(take) if take else np.zeros(0, dtype=np.int64))
    return draws, short


def _prox_signflip_genes(E, enum_max: int = None, n_mc: int = None, seed: int = _PROX_SEED) -> "dict | None":
    """Sign-flip test of the mean over units (rows of E), for every gene (column) at once.
    Units with a NaN are dropped; None with fewer than stats._SL_MIN_UNITS. The limits are
    those of stats._sl_signflip: up to *enum_max* units (default stats._SL_ENUM_UNITS) all
    2^n sign vectors are used (exact, the same p); above, *n_mc* random ones (default
    stats._SL_MC) and p = (1 + hits) / (1 + n_mc).
    Returns {mean, p, npos, n, min_p, exact, n_vectors}."""
    enum_max = stats._SL_ENUM_UNITS if enum_max is None else int(enum_max)
    n_mc = stats._SL_MC if n_mc is None else int(n_mc)
    E = np.asarray(E, dtype=np.float64)
    ok = ~np.isnan(E).any(axis=1)
    n = int(ok.sum())
    if n < stats._SL_MIN_UNITS:
        return None
    Ev = E[ok]
    tot = Ev.sum(axis=0)
    tol = 1e-12 * np.maximum(1.0, np.abs(tot))
    exact = n <= enum_max
    if exact:
        signs = stats._sl_signs(n)
    else:
        signs = np.random.default_rng(seed).integers(0, 2, size=(int(n_mc), n)).astype(np.float64) * 2.0 - 1.0
    hits = np.zeros(Ev.shape[1])
    step = max(1, int(4e7 // max(1, len(signs))))
    for g0 in range(0, Ev.shape[1], step):
        sums = signs @ Ev[:, g0:g0 + step]
        hits[g0:g0 + step] = (np.abs(sums) >= np.abs(tot[g0:g0 + step])[None, :] - tol[None, g0:g0 + step]).sum(axis=0)
    p = hits / len(signs) if exact else (hits + 1.0) / (len(signs) + 1.0)
    return {"mean": tot / n, "p": p, "npos": (Ev > 0).sum(axis=0), "n": n, "exact": exact, "n_vectors": len(signs),
            "min_p": (2.0 / 2 ** n) if exact else 1.0 / (len(signs) + 1.0), "valid": ok}


def _prox_empirical_fdr(obs, rand) -> np.ndarray:
    """Empirical FDR of every gene from matched random anchor sets.

    obs: statistic per gene, larger = more extreme (NaN: not tested); rand: the same
    statistic for N random sets (N × genes). At a gene's own threshold t: FDR = (1 + Σ
    over the sets of #{random ≥ t}) / (N × #{observed ≥ t}), i.e. the mean number of
    genes a random set yields over the number observed, never 0; made monotone in t and
    capped at 1."""
    obs = np.asarray(obs, dtype=np.float64)
    rand = np.asarray(rand, dtype=np.float64)
    out = np.full(obs.shape, np.nan)
    ok = np.isfinite(obs)
    if rand.ndim != 2 or not ok.any():
        return out
    rand = rand[np.isfinite(rand).any(axis=1)]          # a set that gave no statistic is not a draw
    N = rand.shape[0]
    if N == 0:
        return out
    v = obs[ok]
    r = np.sort(rand[:, ok][np.isfinite(rand[:, ok])])
    sv = np.sort(v)
    tol = 1e-12 * np.maximum(1.0, np.abs(v))
    n_obs = len(sv) - np.searchsorted(sv, v - tol, side="left")
    n_rand = len(r) - np.searchsorted(r, v - tol, side="left")
    fdr = (1.0 + n_rand) / (N * np.maximum(n_obs, 1))
    order = np.argsort(v, kind="stable")                # least extreme first
    q = np.minimum.accumulate(fdr[order])               # a stricter threshold never has a larger FDR
    res = np.empty(len(v))
    res[order] = np.minimum(q, 1.0)
    out[ok] = res
    return out


# ─── Proximity DE ─────────────────────────────────────────────────────────────
def _prox_label_arrays(lab, n: int) -> tuple:
    """(labelled mask, labels as str) of one sample's label array (None: unlabelled)."""
    L = pd.Series(lab, dtype=object) if lab is not None else pd.Series([None] * n, dtype=object)
    return L.notna().to_numpy(), L.astype(str).to_numpy()


def _prox_sample_strata(s: dict, seg_source) -> tuple:
    """Stratum codes of the cells of one sample from the compartment columns of the
    Confound Controls tab: (compartment × edge bin × density quintile, compartment × edge
    bin, whether the defaults had to be computed here)."""
    xy = stats._xy_of(s["df"])
    n = len(xy)
    e, d, c = (controls._ctl_get(s, seg_source, k) for k in ("edge_distance_um", "local_density", "compartment"))
    default = False
    if e is None or d is None or np.isnan(pd.to_numeric(pd.Series(e), errors="coerce")).any():
        default = True
        if n >= 3:
            cols, _info = controls._compartments(xy)
            e, d, c = cols["edge_distance_um"], cols["local_density"], cols["compartment"]
        else:
            e, d, c = np.zeros(n), np.zeros(n), np.array(["other"] * n, dtype=object)
    comp = np.unique(pd.Series(c, dtype=object).fillna("other").astype(str).to_numpy(), return_inverse=True)[1]
    lev = controls._coloc_levels(np.zeros(n, dtype=np.int64), comp, np.asarray(e, dtype=np.float64),
                                 np.asarray(d, dtype=np.float64))
    return lev[0], lev[1], default


def _prox_sample(s: dict, lab, genes, target, anchor, placebos, strata, source, p: dict, rng) -> dict:
    """The per-sample part of the proximity test: {skip: reason} or the effects of the
    observed anchors, the placebo anchor types and the random anchor sets."""
    from scipy.spatial import cKDTree
    df = s["df"]
    n = len(df)
    rows = np.asarray(s["rows"])
    labelled, L = _prox_label_arrays(lab, n)
    sf, _iv = analysis._an_size_factors(s)
    sfc = np.where(rows >= 0, sf[np.clip(rows, 0, None)], 0.0)
    is_t = labelled & (L == target) & (sfc > 0)
    is_x = labelled & (L == anchor)
    out = {"n_target": int(is_t.sum()), "n_anchor": int(is_x.sum())}
    if out["n_target"] < p["min_target"]:
        return {**out, "skip": f"{out['n_target']} {target} cells (< {p['min_target']})"}
    if out["n_anchor"] < p["min_anchors"]:
        return {**out, "skip": f"{out['n_anchor']} {anchor} cells (< {p['min_anchors']})"}
    xy = stats._xy_of(df)
    t_idx = np.flatnonzero(is_t)
    cols = np.array([s["gni"][g] for g in genes], dtype=np.int64)
    Y = s["csc"][:, cols].tocsr()[rows[t_idx]].astype(np.float64)
    out["det"] = np.bincount(Y.indices[Y.data >= analysis._DETECT_MIN], minlength=len(genes))
    Y.data = np.log1p(Y.data * np.repeat(sfc[t_idx], np.diff(Y.indptr)))
    out["mean"] = np.asarray(Y.sum(axis=0)).ravel()
    log_depth = np.log1p(analysis._AN_SCALE / sfc[t_idx])
    bins = None
    if source is not None:
        is_src = labelled & (L == source)
        if is_src.any():
            frac = controls._source_geometry(xy, is_src, p["local_r"])[1]
            bins = _prox_quantile_bins(frac[t_idx], p["n_bins"])
    D = _prox_design(log_depth, bins)
    fit = _prox_fit(Y, D)
    jobs._job_check()
    t_xy = xy[t_idx]
    dist = lambda a_idx: cKDTree(xy[a_idx]).query(t_xy, k=1, workers=-1)[0]
    sets, names = [], []
    d_obs = dist(np.flatnonzero(is_x))
    sets.append((d_obs, None)); names.append(("obs", 0))
    for pl in placebos:
        a = np.flatnonzero(labelled & (L == pl))
        if len(a) >= p["min_anchors"]:
            sets.append((dist(a), None)); names.append(("placebo", pl))
    pool = np.flatnonzero(~is_x)
    strata = np.zeros(n, dtype=np.int64) if strata is None else np.asarray(strata, dtype=np.int64)
    nb = int(strata.max()) + 1 if n else 1
    draws, short = _prox_matched_random(pool, strata[pool], np.bincount(strata[is_x], minlength=nb),
                                   p["n_random"], rng)
    for j, dr in enumerate(draws):
        if len(dr) >= 1:
            sets.append((dist(dr), ~np.isin(t_idx, dr))); names.append(("rand", j))
    st = _prox_set_stats(Y, D, fit["beta"], sets, p["near"], p["far"], p["min_near"], p["min_far"],
                         p["min_slope_n"], check=jobs._job_check)
    near_m, far_m = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    near_m[t_idx[d_obs <= p["near"]]] = True
    far_m[t_idx[d_obs >= p["far"]]] = True
    out.update({"names": names, "nearfar": st["nearfar"], "slope": st["slope"],
                "n_near": int(st["n_near"][0]), "n_far": int(st["n_far"][0]), "short": int(short),
                "median_d": float(np.median(d_obs)), "near_mask": near_m, "far_mask": far_m,
                "n_dummies": D.shape[1] - 2})
    if p.get("fit_length"):
        out["length"] = _prox_length_fit(Y, D, fit, d_obs, p["l_grid"])
    return out


_PROX_CALL_FDR = 0.10         # a change is "called" below this FDR: the leak gate's change rule applies to it
_PROX_GATE_OFF = {"applicable": False, "off": True,
                  "reason": "switched off (Gene ownership + leak gate is unticked)"}
_PROX_GATE_OFF_MSG = ("Gene ownership and leak gate: switched off (Gene ownership + leak gate is unticked) - "
                      "no gene is flagged, and genes owned by the anchor type are not marked.")


def _prox_ownership(prof, genes, anchor, target, on: bool = True) -> tuple:
    """(owner, owners, anchor_owned) per gene of *genes*; empty / False when *on* is False."""
    if not on:
        return np.array([""] * len(genes), dtype=object), np.array([""] * len(genes), dtype=object), \
            np.zeros(len(genes), dtype=bool)
    own = controls._gene_ownership(prof).set_index("gene").reindex(genes)
    owners = own["owners"].fillna("").to_numpy()
    return own["owner"].fillna("").to_numpy(), owners, _prox_anchor_owned(owners, anchor, target)


def _prox_anchor_owned(owners, anchor, target) -> np.ndarray:
    """Genes the anchor type owns and the target type does not (controls._gene_ownership:
    an owner has at least 20 % of the top rate): in a target cell next to the anchor their
    transcripts are the anchor's. A gene both types own is left to the leak gate."""
    out = []
    for o in owners:
        own = str(o).split(";")
        out.append(anchor in own and target not in own)
    return np.array(out, dtype=bool)


def _prox_params(params=None) -> dict:
    p = {"near": _PROX_NEAR_UM, "far": _PROX_FAR_UM, "min_detect": _PROX_MIN_DETECT, "n_bins": _PROX_N_BINS,
         "local_r": controls._SPILL_LOCAL_R_UM, "min_anchors": _PROX_MIN_ANCHORS, "min_near": _PROX_MIN_NEAR,
         "min_far": _PROX_MIN_FAR, "min_slope_n": _PROX_MIN_SLOPE_N, "min_target": _PROX_MIN_TARGET,
         "n_random": _PROX_N_RANDOM, "seed": _PROX_SEED, "repro_frac": _PROX_REPRO_FRAC,
         "max_placebo": _PROX_MAX_PLACEBO, "fit_length": False, "l_grid": _PROX_L_GRID, "leak_gate": True}
    p.update({k: v for k, v in (params or {}).items() if v is not None and k in p})
    p["near"], p["far"] = float(p["near"]), float(p["far"])
    p["n_random"] = int(max(1, min(2000, p["n_random"])))
    if not (0 < p["near"] < p["far"]):
        raise ValueError("The near distance must be positive and smaller than the far distance.")
    return p


def _prox_unit_groups(n_samples: int, conds=None, donors=None) -> tuple:
    """Independent units of a one-sample-per-row effect matrix: ({unit name: [sample
    indices]}, by_donor, included sample indices). Samples with a condition are included
    (all samples when none has one); a unit is a donor when every included sample has a
    donor, else a sample."""
    conds = list(conds) if conds is not None else [""] * n_samples
    donors = [str(d or "").strip() for d in (donors if donors is not None else [""] * n_samples)]
    inc = ([i for i in range(n_samples) if conds[i] in ("A", "B")]
           if any(c in ("A", "B") for c in conds) else list(range(n_samples)))
    by_donor = bool(inc) and all(donors[i] for i in inc)
    groups: dict = {}
    for i in inc:
        groups.setdefault(donors[i] if by_donor else f"sample{i}", []).append(i)
    return groups, by_donor, inc


def _prox_unit_means(E, groups: dict) -> np.ndarray:
    """Mean over the samples of every unit (NaN for a unit without a defined sample)."""
    import warnings
    out = np.full((len(groups), E.shape[1]), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for u, ix in enumerate(groups.values()):
            out[u] = np.nanmean(E[ix], axis=0)
    return out


def _prox_test(U_obs, U_rand, mode: str) -> dict:
    """p-values and empirical FDR of one estimator (near − far, or the slope).

    U_obs: units × genes; U_rand: random sets × units × genes. Units without an observed
    effect are left out; a random set counts only when it has an effect in every remaining
    unit. "cohort": exact sign-flip p per gene across the units, FDR from the sign-flip
    p-values of the random sets. "single": p = share of the random sets at least as far
    from their mean as the observed effect, FDR from |z| against the random sets, each
    random set scored against the others. With fewer than max(3, half) usable random
    sets the FDR is Benjamini-Hochberg in the cohort mode ("bh") and absent otherwise
    ("none"). Returns {effect, p, padj, z, mu, r_eff, npos, n_valid, n_units, min_p,
    exact, fdr}."""
    import warnings
    G = U_obs.shape[1]
    N = U_rand.shape[0]
    nan = np.full(G, np.nan)
    ok = ~np.isnan(U_obs).any(axis=1)
    out = {"effect": nan, "p": nan, "padj": nan, "z": nan, "mu": nan, "r_eff": np.zeros((0, G)), "npos": None,
           "n_valid": 0, "n_units": int(ok.sum()), "min_p": np.nan, "exact": None, "fdr": "none"}
    if not ok.any():
        return out
    Uo = U_obs[ok]
    eff = Uo.mean(axis=0)
    draws = [j for j in range(N) if not np.isnan(U_rand[j][ok]).any()]
    r_eff = np.stack([U_rand[j][ok].mean(axis=0) for j in draws]) if draws else np.zeros((0, G))
    nv = len(draws)
    enough = nv >= max(3, (N + 1) // 2)
    out.update({"effect": eff, "n_valid": nv, "r_eff": r_eff})
    if nv > 1:
        mu, sd = r_eff.mean(axis=0), r_eff.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out.update({"mu": mu, "z": np.where(sd > 0, (eff - mu) / sd, np.nan)})
    if mode == "cohort":
        m = _prox_signflip_genes(Uo)
        if m is None:
            return out
        out.update({"p": m["p"], "npos": m["npos"], "min_p": m["min_p"], "exact": m["exact"], "n_units": m["n"]})
        if enough:
            p_rand = np.stack([_prox_signflip_genes(U_rand[j][ok])["p"] for j in draws])
            out.update({"padj": _prox_empirical_fdr(-m["p"], -p_rand), "fdr": "empirical"})
        else:
            out.update({"padj": stats._bh(m["p"]), "fdr": "bh"})
        return out
    if not enough:
        return out
    mu = out["mu"]
    hits = (np.abs(r_eff - mu[None, :]) >= np.abs(eff - mu)[None, :] - 1e-12).sum(axis=0)
    tot1, tot2 = r_eff.sum(axis=0), (r_eff ** 2).sum(axis=0)
    mu_j = (tot1[None, :] - r_eff) / (nv - 1)
    var_j = np.maximum((tot2[None, :] - r_eff ** 2) - (nv - 1) * mu_j ** 2, 0.0) / (nv - 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_j = np.where(var_j > 0, (r_eff - mu_j) / np.sqrt(var_j), np.nan)
    out.update({"p": (1.0 + hits) / (nv + 1.0), "padj": _prox_empirical_fdr(np.abs(out["z"]), np.abs(z_j)),
                "min_p": 1.0 / (nv + 1.0), "fdr": "empirical"})
    return out


def _prox_compute(samples, labels, target, anchor, strata=None, source="auto", conds=None, donors=None,
                  params=None, progress=None) -> dict:
    """Gene-wise proximity differential expression of the *target* cells relative to the
    *anchor* cells (see _PROX_NOTE).

    samples / labels: analysis samples and the label of every cell per sample; strata: per
    sample, the stratum code of every cell the random anchors are matched on (None: one
    stratum); source: the type whose local fraction is the covariate ("auto": the type
    with the most transcripts; None: no such covariate); conds / donors: the sample
    design. Returns the result dict of the Differential Expression tab (kind
    "proximity")."""
    p = _prox_params(params)
    target, anchor = str(target), str(anchor)
    if target == anchor:
        raise ValueError("The cells and the anchor type must be two different types.")
    if not samples:
        raise ValueError("The active view has no expression matrix.")
    S = len(samples)
    all_genes = analysis._an_common_genes(samples)
    if not all_genes:
        raise ValueError("The samples of this view share no measured genes.")
    prof = controls._type_profiles(samples, labels, all_genes)
    types = prof["types"]
    for nm in (target, anchor):
        if nm not in types:
            raise ValueError(f"No cell of the view is labelled '{nm}'.")
    if source == "auto":
        # never the anchor (its local fraction is nearly a function of the distance that is
        # tested: regressing it out removes the effect) nor the tested cells' own type
        source = controls._auto_source(prof, exclude=(anchor, target))
    source = str(source) if source and str(source) != "none" else None
    if source == anchor:
        raise ValueError(f"The density covariate cannot be the anchor type '{anchor}': the local {anchor} "
                         f"fraction is almost a function of the distance to the nearest {anchor} cell, so "
                         f"adjusting for it removes the proximity effect that is tested. Pick another type, "
                         f"'auto' or 'none'.")
    n_of = dict(zip(types, prof["n"]))
    sim = [t for t in types if t not in (target, anchor) and n_of[t] >= p["min_anchors"]
           and max(n_of[t], n_of[anchor]) / max(1, min(n_of[t], n_of[anchor])) <= controls._COLOC_SIMILAR_FOLD]
    placebos = sorted(sim, key=lambda t: abs(np.log(n_of[t] / n_of[anchor])))[:int(p["max_placebo"])]
    # tested genes: detected in at least min_detect of the target cells of the view
    ti = types.index(target)
    det_tot, n_t_tot = np.zeros(len(all_genes)), 0
    pre = []
    for s, lab in zip(samples, labels):
        rows = np.asarray(s["rows"])
        labelled, L = _prox_label_arrays(lab, len(s["df"]))
        sel = rows[labelled & (L == target) & (rows >= 0)]
        if len(sel):
            sub = s["csc"][:, np.array([s["gni"][g] for g in all_genes], dtype=np.int64)].tocsr()[sel]
            det_tot += np.bincount(sub.indices[sub.data >= analysis._DETECT_MIN], minlength=len(all_genes))
            n_t_tot += len(sel)
        pre.append(len(sel))
    if n_t_tot < p["min_target"]:
        raise ValueError(f"Only {n_t_tot:,} cells are labelled '{target}'; at least {p['min_target']} are needed "
                         f"in a sample.")
    keep_g = det_tot / n_t_tot >= p["min_detect"]
    genes = [g for g, k in zip(all_genes, keep_g) if k]
    G = len(genes)
    if G < 2:
        raise ValueError(f"Fewer than 2 genes are detected in at least {100 * p['min_detect']:g} % of the "
                         f"{target} cells.")
    rng = np.random.default_rng(p["seed"])
    per, rows_out = [], []
    for i, (s, lab) in enumerate(zip(samples, labels)):
        jobs._job_check()
        if progress is not None:
            progress(i / S, f"{s['label']}: residualising and scanning anchor sets…")
        r = _prox_sample(s, lab, genes, target, anchor, placebos, None if strata is None else strata[i],
                         source, p, rng)
        per.append(r)
        rows_out.append({"label": s["label"], "name": s["name"], "n_target": r["n_target"],
                         "n_anchor": r["n_anchor"], "n_near": r.get("n_near", 0), "n_far": r.get("n_far", 0),
                         "median_d": r.get("median_d", np.nan), "short": r.get("short", 0),
                         "used": "skip" not in r and r.get("n_near", 0) >= p["min_near"] and r.get("n_far", 0) >= p["min_far"],
                         "note": r.get("skip", "") or ("" if (r.get("n_near", 0) >= p["min_near"] and r.get("n_far", 0) >= p["min_far"])
                                                       else f"fewer than {p['min_near']} near or {p['min_far']} far cells"),
                         "cond": (conds[i] if conds is not None else ""), "donor": (donors[i] if donors is not None else "")})
    used = [i for i, r in enumerate(rows_out) if r["used"]]
    if not used:
        raise ValueError("No sample can be tested: " + "; ".join(f"{r['label']}: {r['note']}" for r in rows_out) + ".")

    def _stack(kind, key, what):
        E = np.full((S, G), np.nan)
        for i, r in enumerate(per):
            if "skip" in r:
                continue
            if (kind, key) in r["names"]:
                E[i] = r[what][r["names"].index((kind, key))]
        return E

    N = p["n_random"]
    E_obs = {w: _stack("obs", 0, w) for w in ("nearfar", "slope")}
    E_rand = {w: np.stack([_stack("rand", j, w) for j in range(N)]) for w in ("nearfar", "slope")}   # N × S × G
    E_pl = {w: {pl: _stack("placebo", pl, w) for pl in placebos} for w in ("nearfar", "slope")}
    groups, by_donor, inc = _prox_unit_groups(S, conds, donors)
    groups = {k: [i for i in ix if i in used] for k, ix in groups.items()}
    groups = {k: ix for k, ix in groups.items() if ix}
    if not groups:
        raise ValueError("No testable sample is included in the design (Samples & conditions).")
    # counts, the leak gate and the "used" flags follow the samples that are tested: a
    # sample without a condition in the design is not part of the result
    tested = sorted({i for ix in groups.values() for i in ix})
    for i in used:
        if i not in tested:
            rows_out[i]["used"] = False
            rows_out[i]["note"] = "no condition in the design (excluded)"
    used = tested
    n_units = len(groups)
    U_obs = {w: _prox_unit_means(E_obs[w], groups) for w in E_obs}
    U_rand = {w: np.stack([_prox_unit_means(E_rand[w][j], groups) for j in range(N)]) for w in E_rand}
    mode = "cohort" if n_units >= stats._SL_MIN_UNITS else "single"
    ln2 = np.log(2.0)
    if progress is not None:
        progress(0.9, "Testing…")
    tests = {w: _prox_test(U_obs[w], U_rand[w], mode) for w in ("nearfar", "slope")}
    if mode == "cohort" and all(t["exact"] is None for t in tests.values()):
        mode = "single"                   # fewer than 3 units have a defined effect
        tests = {w: _prox_test(U_obs[w], U_rand[w], mode) for w in ("nearfar", "slope")}
    # the statistic the p-values and the FDR are read from: near − far, or the slope when
    # too few random sets leave enough far (or near) cells for a near − far effect
    est = next((w for w in ("nearfar", "slope") if tests[w]["fdr"] == "empirical"), "nearfar")
    tt = tests[est]
    if mode == "single" and tt["fdr"] != "empirical":
        raise ValueError(
            f"Only {tests['nearfar']['n_valid']} of {N} matched random anchor sets give a near − far effect and "
            f"{tests['slope']['n_valid']} a slope in every sample (random anchors as numerous as the anchor cells "
            f"leave too few cells beyond the far distance): with fewer than {stats._SL_MIN_UNITS} independent "
            f"units there is then no null to compare with. Lower the far distance or pick a rarer anchor type.")
    eff, slope = tests["nearfar"]["effect"], tests["slope"]["effect"]
    table = pd.DataFrame({"gene": genes, "log2fc": eff / ln2, "slope": slope / ln2,
                          "detection": 100.0 * det_tot[keep_g] / n_t_tot,
                          "random_mean": tt["mu"] / ln2, "z_random": tt["z"]})
    if tt.get("npos") is not None:
        table["n_pos"] = tt["npos"]
    table["pval"], table["padj"] = tt["p"], tt["padj"]
    other = tests["slope" if est == "nearfar" else "nearfar"]
    table["p_slope" if est == "nearfar" else "p_nearfar"] = other["p"]
    info = {"mode": mode, "n_units": tt["n_units"], "by_donor": by_donor, "min_p": tt["min_p"], "exact": tt["exact"],
            "fdr": tt["fdr"], "n_random_valid": tt["n_valid"], "estimator": est,
            "n_random_valid_nearfar": tests["nearfar"]["n_valid"]}
    mu = tt["mu"]
    r_eff = tt["r_eff"]
    eff_t = tt["effect"]

    # specificity: other anchor types of similar abundance
    repro = np.zeros(G, dtype=np.int64)
    sgn = np.sign(eff_t)
    pl_used = []
    for pl in placebos:
        Up = _prox_unit_means(E_pl[est][pl], groups)
        okp = ~np.isnan(Up).any(axis=1)
        if not okp.any():
            continue
        pl_used.append(pl)
        table[f"placebo:{pl}"] = np.nanmean(Up[okp], axis=0) / ln2
        if mode == "cohort":
            same = (np.sign(Up[okp]) == sgn[None, :]).sum(axis=0)
            rep = same >= int(np.ceil(p["repro_frac"] * int(okp.sum()) - 1e-9))
        else:
            e_pl = np.nanmean(Up[okp], axis=0)
            p_pl = (1.0 + (np.abs(r_eff - mu[None, :]) >= np.abs(e_pl - mu)[None, :] - 1e-12).sum(axis=0)) / (len(r_eff) + 1.0)
            rep = (np.sign(e_pl - mu) == np.sign(eff_t - mu)) & (p_pl <= 0.05)
        repro += rep.astype(np.int64)
    table["placebo_repro"] = repro if pl_used else np.nan
    table["specific"] = (repro == 0) if pl_used else True
    # difference between the two conditions of the design
    diff_note = ""
    if conds is not None and any(c == "A" for c in conds) and any(c == "B" for c in conds):
        c_use = [conds[i] if (i in used and conds[i] in ("A", "B")) else "" for i in range(S)]
        u = stats._sl_units(c_use, donors)
        k = (len(u["pairs"]) if u["paired"] else min(sum(1 for _n, c, _ix in u["units"] if c == x) for x in ("A", "B")))
        if k >= stats._SL_MIN_UNITS:
            ed, pd_ = np.full(G, np.nan), np.full(G, np.nan)
            nd = np.zeros(G, dtype=np.int64)        # units a gene's difference is tested on
            first, floor = None, np.nan
            for g in range(G):
                if g % 500 == 0:
                    jobs._job_check()
                r = stats._sl_test(E_obs["nearfar"][:, g], c_use, donors, units=u)
                ed[g], pd_[g] = r["effect"] / ln2, r["p"]
                if np.isfinite(r["p"]):
                    nd[g] = r["n"]
                    # the floor of the family is that of its best-covered gene (a gene without an
                    # effect in some unit is tested on fewer units and has a larger floor of its own)
                    floor = r["min_p"] if not np.isfinite(floor) else min(floor, r["min_p"])
                first = first or r
            table["effect_diff"], table["n_units_diff"] = ed, nd
            table["p_diff"], table["fdr_diff"] = pd_, stats._bh(pd_)
            diff_note = (f"A − B of the effect: {first['test']} test, {u['note']}; smallest achievable p "
                         f"{floor:.3g}; BH FDR in column fdr_diff")
            tested = nd[nd > 0]
            if len(tested) and tested.min() != tested.max():
                diff_note += (f"; a gene is tested on the units in which its effect is defined: "
                              f"{int(tested.min())} to {int(tested.max())} per gene (column n_units_diff)")
            if u["dropped"]:
                diff_note += ("; donors with samples in one condition only were left out: "
                              + ", ".join(u["dropped"]))
        else:
            diff_note = (f"no test of the effect between conditions: fewer than {stats._SL_MIN_UNITS} "
                         f"independent units per condition ({u['note']})")
    if p.get("fit_length"):
        Ls = np.stack([per[i]["length"]["L"] for i in used])
        table["L_um"] = np.median(Ls, axis=0)
        table["amplitude"] = np.mean(np.stack([per[i]["length"]["amplitude"] for i in used]), axis=0) / ln2
        table["r2"] = np.nanmedian(np.stack([per[i]["length"]["r2"] for i in used]), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        table["neg_log10_padj"] = -np.log10(np.maximum(table["padj"].to_numpy(dtype=float), 1e-300))
    # ownership and leak gate: T near X (arm A) against T far from X (arm B); both are
    # skipped when "Gene ownership + leak gate" is unticked (no gene is then flagged)
    gate_on = bool(p.get("leak_gate", True))
    table["owner"], table["owners"], table["anchor_owned"] = _prox_ownership(prof, genes, anchor, target, gate_on)
    near_masks = [per[i]["near_mask"] if i in used else np.zeros(len(samples[i]["df"]), dtype=bool) for i in range(S)]
    far_masks = [per[i]["far_mask"] if i in used else np.zeros(len(samples[i]["df"]), dtype=bool) for i in range(S)]
    jobs._job_check()
    gate = (controls._leak_gate_groups(samples, labels, near_masks, far_masks, genes, prof, per_sample=len(used) > 1,
                                       called=np.nan_to_num(table["padj"].to_numpy(dtype=float), nan=1.0) < _PROX_CALL_FDR)
            if gate_on else dict(_PROX_GATE_OFF))
    flag = table["anchor_owned"].to_numpy().copy()
    if gate["applicable"]:
        lt = gate["table"].set_index("gene").reindex(genes)
        for c in ("leak_share_a", "leak_share_b", "leak_change_frac", "leak_fraction", "top_leak_type"):
            table[c] = lt[c].to_numpy()
        table["gate_flag"] = lt["leak_flag"].fillna(False).to_numpy(dtype=bool)
        flag |= table["gate_flag"].to_numpy()
    else:
        table["leak_fraction"] = np.nan
        table["gate_flag"] = False
    table["leak_flag"] = flag
    cpm = controls._cpm(prof["counts"], prof["depth"])
    gix = np.array([prof["genes"].index(g) for g in genes], dtype=np.int64) if G < len(prof["genes"]) else np.arange(G)
    table["anchor_ratio"] = (cpm[types.index(anchor)][gix] / np.maximum(cpm[ti][gix], 1e-9))
    return {"kind": "proximity", "table": table, "params": p, "info": info, "samples": rows_out,
            "target": target, "anchor": anchor, "source": source, "placebos": pl_used,
            "n_genes": G, "n_genes_all": len(all_genes), "n_samples": len(used), "diff_note": diff_note,
            "gate": {k: v for k, v in gate.items() if k != "table"},
            "n_a": int(sum(r["n_near"] for r in rows_out if r["used"])),
            "n_b": int(sum(r["n_far"] for r in rows_out if r["used"])),
            "n_target": int(sum(r["n_target"] for r in rows_out if r["used"])),
            "n_anchor": int(sum(r["n_anchor"] for r in rows_out if r["used"])),
            "short": int(sum(r["short"] for r in rows_out)),
            "corrected": bool(samples[0]["corrected"]),
            "n_uncorrected": analysis._an_uncorrected(samples)}


_PROX_COLUMNS = [("gene", "Gene", "text"), ("log2fc", "Near − far", "num"), ("slope", "Slope", "num"),
                 ("n_pos", "Units > 0", "num"), ("z_random", "z vs random", "num"),
                 ("random_mean", "Random anchors", "num"), ("pval", "p", "num"), ("padj", "Emp. FDR", "num"),
                 ("p_slope", "p (slope)", "num"), ("p_nearfar", "p (near − far)", "num"), ("specific", "Specific", "text"),
                 ("effect_diff", "A − B", "num"), ("n_units_diff", "Units (A vs B)", "num"), ("p_diff", "p (A vs B)", "num"), ("fdr_diff", "FDR (A vs B)", "num"),
                 ("L_um", "L (µm)", "num"), ("r2", "R² (L)", "num"), ("detection", "% cells", "num"),
                 ("owner", "Owner", "text"), ("anchor_ratio", "X ÷ T rate", "num"),
                 ("leak_fraction", "Leak fraction", "num"), ("leak_flag", "Leak gate", "text")]


def _prox_units_statement(res: dict) -> str:
    """Unit and n of the p-values of a proximity result."""
    info, N = res["info"], res["info"]["n_random_valid"]
    if info["mode"] == "cohort":
        return (f"n = {info['n_units']} {'donors' if info['by_donor'] else 'samples'} (exact sign-flip test of "
                f"the per-unit {'slopes' if info['estimator'] == 'slope' else 'effects'}" + ("" if info["exact"] else f", {stats._SL_MC:,} random sign vectors")
                + f"; smallest achievable p {info['min_p']:.3g})")
    return (f"n = {res['n_target']:,} cells of type {res['target']} in {res['n_samples']} sample(s); the p-value is the "
            f"share of {N} matched random anchor sets at least as far from their mean as the observed "
            f"{'slope' if info['estimator'] == 'slope' else 'effect'} "
            f"(smallest achievable p {info['min_p']:.3g})")


def _run_proximity_de(target, anchor, label_kind, seg_source, method, use_corrected, params=None,
                      source="auto") -> None:
    """Background job: proximity differential expression, published into the DE state."""
    _gen = jobs._dataset_gen

    def _set(status, message, result=None):
        with analysis._de_lock:
            if jobs._stale_job(_gen, status, message):
                return
            analysis._de_state["status"] = status
            analysis._de_state["message"] = message
            if result is not None:
                analysis._de_state["result"] = result

    try:
        _t0 = _time.time()
        _set("running", "Collecting cells, labels and strata…")
        samples = analysis._an_samples(seg_source, use_corrected)
        if not samples:
            _set("error", analysis._an_no_matrix_msg(seg_source))
            return
        labels, cat = stats._comp_labels(label_kind, samples, seg_source, method)
        if all(l is None for l in labels):
            _set("error", f"The active view has no {cat} labels: run Cell Type Annotation first, or use clusters.")
            return
        strata, defaults = [], []
        for s in samples:
            jobs._job_check()
            _fine, coarse, dflt = _prox_sample_strata(s, seg_source)
            strata.append(coarse)
            if dflt:
                defaults.append(s["label"])
        conds, _b = stats._design_of(samples)
        donors = stats._design_donors(samples)

        def _prog(f, text=None):
            jobs._job_progress(text, 0.05 + 0.9 * f)

        _set("running", f"Proximity of {target} to {anchor}…")
        res = _prox_compute(samples, labels, target, anchor, strata=strata, source=source, conds=conds,
                            donors=donors, params=params, progress=_prog)
        jobs._check_dataset_gen(_gen)
        p, info, t = res["params"], res["info"], res["table"]
        res.update({"category": cat, "seg_source": seg_source or "xenium", "seconds": _time.time() - _t0,
                    "label_a": f"{target} ≤ {p['near']:g} µm from {anchor}",
                    "label_b": f"{target} ≥ {p['far']:g} µm", "columns": _PROX_COLUMNS,
                    "lfc_title": f"near − far of {anchor} (residual log2 expression)",
                    "units_note": _prox_units_statement(res), "notes": [], "computed_defaults": defaults,
                    "fdr_name": "empirical FDR" if info["fdr"] == "empirical" else "FDR (BH)",
                    "up_name": f"higher near {anchor}", "down_name": f"lower near {anchor}"})
        res["title"] = (f"{target}: ≤ {p['near']:g} µm from {anchor} ({res['n_a']:,} cells) vs ≥ {p['far']:g} µm "
                        f"({res['n_b']:,}) — {'sign-flip across ' + str(info['n_units']) + ' units' if info['mode'] == 'cohort' else 'single-sample, matched-anchor null'}")
        fl = t["leak_flag"].to_numpy(dtype=bool)
        q = t["padj"].to_numpy(dtype=float)
        n05, n10 = int((q < 0.05).sum()), int((q < 0.10).sum())
        msg = (f"Proximity of {target} to {anchor} ({cat}): {res['n_genes']:,} of {res['n_genes_all']:,} genes "
               f"tested (detected in ≥ {100 * p['min_detect']:g} % of the cells of type {target}); {res['n_samples']} "
               f"sample(s), {res['n_target']:,} cells of type {target}, {res['n_anchor']:,} of type {anchor}; near ≤ "
               f"{p['near']:g} µm: {res['n_a']:,} cells, far ≥ {p['far']:g} µm: {res['n_b']:,} cells. "
               f"Residualised within sample on log depth"
               + (f" and deciles of the local {res['source']} fraction" if res["source"] else "")
               + f". {n05:,} genes with {res['fdr_name']} < 0.05 ({n10:,} < 0.10; {info['n_random_valid']} of "
                 f"{p['n_random']} matched random anchor sets usable, seed {p['seed']}); of the {n10:,}, "
                 f"{int(((q < 0.10) & fl).sum()):,} are flagged "
                 f"(owned by {anchor} or leak gate) and {int(((q < 0.10) & ~fl & t['specific'].to_numpy(dtype=bool)).sum()):,} "
                 f"are unflagged and specific ({'SPLIT-corrected' if res['corrected'] else 'original'} counts, "
                 f"{res['seconds']:.1f} s).")
        if info["estimator"] == "slope":
            msg += (f"\n⚠ p-values and FDR are those of the SLOPE on −log(d + 1): only "
                    f"{info['n_random_valid_nearfar']} of {p['n_random']} random anchor sets leave enough near and "
                    f"far cells in every unit for a near − far effect (random anchors as numerous as the anchor "
                    f"cells cover the tissue). The near − far column is still the observed effect.")
        top_x = (q < 0.10) & ~fl & (t["owner"].to_numpy() == anchor)
        if top_x.any():
            msg += (f" {int(top_x.sum()):,} of the unflagged genes have {anchor} as their top owner ({target} "
                    f"carries them at ≥ {100 * controls._OWN_FRAC:.0f} % of that rate, so they are not flagged): "
                    f"column 'X ÷ T rate'; read them as possible spillover.")
        if info["fdr"] == "bh":
            msg += (f"\n⚠ Only {info['n_random_valid']} of {p['n_random']} random anchor sets gave an effect in "
                    f"every unit (too few far or near cells): no empirical FDR; the FDR shown is "
                    f"Benjamini-Hochberg over the sign-flip p-values.")
        if res.get("n_uncorrected"):
            msg += " " + analysis._an_uncorrected_note(res["n_uncorrected"]) + "."
        msg += f"\nUnit of the p-values: {res['units_note']}."
        if info["mode"] != "cohort":
            msg += (f"\n⚠ SINGLE-SAMPLE RESULT: {info['n_units']} independent unit(s); at least "
                    f"{stats._SL_MIN_UNITS} are needed for a test across samples. The p-values and the "
                    f"empirical FDR describe the section(s) analysed (anchors exchangeable within "
                    f"compartment × edge-distance strata) and do not generalise to other sections or donors.")
        reach = stats._sl_reach(info["min_p"], res["n_genes"]) if info["mode"] == "cohort" else None
        if reach:
            msg += (f"\nNote: {reach}. The empirical FDR compares the number of genes at a threshold with "
                    f"what random anchors give, so it can be small where Benjamini-Hochberg cannot.")
        if res["placebos"]:
            msg += (f"\nSpecificity: other anchor types of similar abundance: {', '.join(res['placebos'])}; "
                    f"{int((~t['specific'].to_numpy(dtype=bool)).sum()):,} genes are reproduced by at least one.")
        else:
            msg += "\nSpecificity: no other type of similar abundance (within 3×) to compare with."
        g = res["gate"]
        if g.get("off"):
            msg += f"\n{_PROX_GATE_OFF_MSG}"
        else:
            msg += (f"\nOwnership: {int(t['anchor_owned'].sum()):,} tested genes are owned by {anchor} and not by {target}. Leak gate "
                    + (f"({target} near vs far; {g.get('mode')}): {int(t['gate_flag'].sum()):,} genes flagged "
                       f"({g.get('n_e1', 0):,} by the signal rule, {g.get('n_e2', 0):,} by the change rule, which is applied to "
                       f"the {g.get('n_change_ok', 0):,} genes with {res['fdr_name']} < {_PROX_CALL_FDR:g} or |log2 change| ≥ "
                       f"{controls._LEAK_CHANGE_MIN_LFC:g})."
                       if g.get("applicable") else f"not applied: {g.get('reason')}."))
        if res["diff_note"]:
            msg += f"\nBetween conditions: {res['diff_note']}."
        skipped = [r for r in res["samples"] if not r["used"]]
        if skipped:
            msg += "\nNot used: " + "; ".join(f"{r['label']} ({r['note']})" for r in skipped) + "."
        if res["short"]:
            msg += (f"\nNote: {res['short']:,} random anchors could not be drawn (a stratum had fewer other "
                    f"cells than anchors).")
        if defaults:
            msg += (f"\nCompartments were not computed for {', '.join(defaults)}: the strata of the random "
                    f"anchors use the default edge distance (Confound Controls tab).")
        if p.get("fit_length"):
            msg += ("\nL (µm): per gene and sample the length of exp(−d / L) with the smallest residual sum of "
                    "squares, median over the samples; R² is small for most genes - read L only where it is not.")
        _set("done", msg, result=res)
        print(f"  Proximity DE: {msg}", flush=True)
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


def _prox_csv_header(res: dict) -> str:
    p = res["params"]
    lines = [f"# Proximity differential expression: {res['target']} cells relative to {res['anchor']} cells; "
             f"labels: {res.get('category', '')}; segmentation: {res.get('seg_source', '')}; counts: "
             f"{'SPLIT-corrected' if res['corrected'] else 'original'}",
             f"# log2fc = mean residual at <= {p['near']:g} um minus mean residual at >= {p['far']:g} um of the "
             f"nearest {res['anchor']} cell, in log2 units; slope = OLS slope of the residual on -log(d + 1)",
             f"# residual: log1p(CP10k) regressed within sample on log1p(library size)"
             + (f" and {p['n_bins']} quantile bins of the fraction of {res['source']} cells within {p['local_r']:g} um"
                if res["source"] else ""),
             f"# test statistic: {res['info']['estimator']}; padj = {res.get('fdr_name', 'empirical FDR')} from {res['info']['n_random_valid']} usable of {p['n_random']} count-matched random anchor sets (same compartment x "
             f"edge-bin stratum, seed {p['seed']}); random_mean / z_random: the random sets' mean effect and "
             f"the observed effect in their SDs",
             f"# unit of the p-values: {res.get('units_note', '')}",
             f"# specific: not reproduced by the placebo anchor types ({', '.join(res['placebos']) or 'none'}); "
             f"anchor_owned: {res['anchor']} owns the gene and {res['target']} does not; gate_flag: leak gate, near vs far; "
             f"leak_flag = anchor_owned or gate_flag"]
    if res.get("diff_note"):
        lines.append(f"# {res['diff_note']}")
    if res.get("n_uncorrected"):
        lines.append(f"# {analysis._an_uncorrected_note(res['n_uncorrected'])}")
    for r in res["samples"]:
        lines.append(f"# sample {r['label']} ({r['name']}): {r['n_target']} target cells, {r['n_anchor']} anchors, "
                     f"{r['n_near']} near, {r['n_far']} far; condition {r['cond'] or '-'}; donor {r['donor'] or '-'}"
                     + (f"; not used: {r['note']}" if not r["used"] else ""))
    return "\n".join(lines) + "\n"


# ─── Contact-group pseudobulk DE ──────────────────────────────────────────────
def _contact_match(a_mask, c_mask, strata) -> tuple:
    """Match the comparator to the contact group within ONE sample: contact cells in a
    stratum without comparator are dropped, and every comparator cell of a shared stratum
    gets the weight (share of the kept contact cells in the stratum) / (share of the kept
    comparator cells in it), so the weighted comparator has the contact group's stratum
    mix and the weights sum to the number of kept comparator cells. Returns (kept contact
    mask, weights [0 outside the kept comparator], info)."""
    a_mask, c_mask = np.asarray(a_mask, dtype=bool), np.asarray(c_mask, dtype=bool)
    strata = np.asarray(strata)
    n = len(strata)
    w, keep_a = np.zeros(n), np.zeros(n, dtype=bool)
    ia, ic = np.flatnonzero(a_mask), np.flatnonzero(c_mask)
    na, nc = pd.Series(strata[ia]).value_counts(), pd.Series(strata[ic]).value_counts()
    both = na.index.intersection(nc.index)
    ka, kc = ia[np.isin(strata[ia], both)], ic[np.isin(strata[ic], both)]
    NA, NC = len(ka), len(kc)
    if NA and NC:
        fa, fc = na.loc[both] / NA, nc.loc[both] / NC
        w[kc] = (fa / fc).loc[strata[kc]].to_numpy()
        keep_a[ka] = True
    return keep_a, w, {"n_a_all": len(ia), "n_c_all": len(ic), "n_a": NA, "n_c": NC,
                       "n_a_dropped": len(ia) - NA, "n_strata": int(len(both))}


def _contact_near(xy, is_t, is_x, radius: float) -> np.ndarray:
    """For ONE sample: mask of the target cells with an anchor cell within *radius* µm."""
    from scipy.spatial import cKDTree
    near = np.zeros(len(xy), dtype=bool)
    if is_x.any() and is_t.any():
        k = cKDTree(xy[is_x]).query_ball_point(xy[is_t], float(radius), return_length=True, workers=-1)
        near[np.flatnonzero(is_t)] = np.asarray(k) > 0
    return near


def _contact_groups(samples, labels, target, anchor, strata, radius=_CONTACT_R_UM,
                    min_cells=_CONTACT_MIN_CELLS) -> list:
    """Per sample: the contact group (target cells with an anchor cell within *radius*),
    the matched comparator (target cells without) and their weights. One dict per sample:
    {a (kept contact mask), w (comparator weights), strata, testable, info…}."""
    out = []
    for i, (s, lab) in enumerate(zip(samples, labels)):
        n = len(s["df"])
        rows = np.asarray(s["rows"])
        labelled, L = _prox_label_arrays(lab, n)
        is_t = labelled & (L == target) & (rows >= 0)
        is_x = labelled & (L == anchor)
        near = _contact_near(stats._xy_of(s["df"]), is_t, is_x, radius)
        st = np.zeros(n, dtype=np.int64) if strata is None else np.asarray(strata[i], dtype=np.int64)
        keep_a, w, info = _contact_match(is_t & near, is_t & ~near, st)
        info.update({"label": s["label"], "name": s["name"], "n_target": int(is_t.sum()), "n_anchor": int(is_x.sum()),
                     "testable": bool(info["n_a"] >= min_cells and info["n_c"] >= min_cells)})
        out.append({"a": keep_a, "w": w, "strata": st, **info})
    return out


def _contact_pb_fit(prep: dict, engine: str = "auto") -> dict:
    """The model fit of the contact test: stats._pb_fit without shrinkage (replaced by a
    stub in the tests that run without R)."""
    return stats._pb_fit(prep, shrink=False, engine=engine)


def _contact_compute(samples, labels, target, anchor, strata=None, radius=_CONTACT_R_UM,
                     min_cells=_CONTACT_MIN_CELLS, min_count=None, n_shuffle=_CONTACT_N_SHUFFLE, seed=0,
                     donors=None, fit=None, lfc_min=_CONTACT_LFC_MIN, fdr=_CONTACT_FDR, progress=None,
                     leak_gate: bool = True) -> dict:
    """Contact-group pseudobulk differential expression (see _CONTACT_NOTE). *fit*: the
    function(prep) -> {table, …} that fits ~ sample + group (default _contact_pb_fit).
    Returns the result dict of the Differential Expression tab (kind "contact")."""
    target, anchor = str(target), str(anchor)
    fit = fit or _contact_pb_fit
    min_cells = int(max(1, _CONTACT_MIN_CELLS if min_cells is None else min_cells))
    min_count = int(stats._PB_MIN_COUNT if min_count is None else min_count)
    n_shuffle = int(max(0, min(5000, _CONTACT_N_SHUFFLE if n_shuffle is None else n_shuffle)))
    if target == anchor:
        raise ValueError("The cells and the contact type must be two different types.")
    if not samples:
        raise ValueError("The active view has no expression matrix.")
    genes_all = analysis._an_common_genes(samples)
    if not genes_all:
        raise ValueError("The samples of this view share no measured genes.")
    grp = _contact_groups(samples, labels, target, anchor, strata, radius, min_cells)
    test = [i for i, g in enumerate(grp) if g["testable"]]
    if len(test) < 2:
        detail = "; ".join(f"{g['label']}: {g['n_a']:,} in contact, {g['n_c']:,} matched comparators" for g in grp)
        raise ValueError(
            f"Contact-group differential expression needs at least 2 samples with at least {min_cells} "
            f"{target} cells in contact with {anchor} (within {radius:g} µm) and {min_cells} matched {target} "
            f"cells without; {len(test)} sample(s) qualify ({detail}). It compares pseudobulks within samples, "
            f"so one sample gives no test. For a single sample use the mode 'Proximity to a cell type' "
            f"(cell-level, with a matched-anchor null).")
    dn = [str((donors[i] if donors is not None else "") or "").strip() for i in range(len(samples))]
    by_donor = all(dn[i] for i in test) and len({dn[i] for i in test}) < len(test)
    if by_donor and len({dn[i] for i in test}) < 2:
        raise ValueError("The testable samples all come from one donor: they are one independent unit, and "
                         "the contact test needs at least 2.")
    block_of = {i: (dn[i] if by_donor else samples[i]["label"]) for i in test}
    blocks = list(dict.fromkeys(block_of[i] for i in test))
    # cells × genes matrices of the cells that are in a group
    mats, cells = {}, {}
    for i in test:
        s, g = samples[i], grp[i]
        ix = np.flatnonzero(g["a"] | (g["w"] > 0))
        cols = np.array([s["gni"][x] for x in genes_all], dtype=np.int64)
        mats[i] = s["csc"][:, cols].tocsr()[np.asarray(s["rows"])[ix]].astype(np.float64)
        cells[i] = ix

    def _pseudobulk(lab_of: dict, w_of: dict) -> np.ndarray:
        """genes × (blocks × [comparator, contact]) weighted sums, before rounding."""
        M = np.zeros((len(genes_all), 2 * len(blocks)))
        for i in test:
            b = blocks.index(block_of[i])
            is_a, w = lab_of[i], w_of[i]
            M[:, 2 * b] += np.asarray(mats[i].T @ np.where(is_a, 0.0, w)).ravel()
            M[:, 2 * b + 1] += np.asarray(mats[i].T @ is_a.astype(np.float64)).ravel()
        return M

    lab0 = {i: grp[i]["a"][cells[i]] for i in test}
    w0 = {i: grp[i]["w"][cells[i]] for i in test}
    M0 = _pseudobulk(lab0, w0)
    keep = M0.sum(axis=1) >= min_count
    if int(keep.sum()) < 2:
        raise ValueError(f"Fewer than 2 genes have a total count of at least {min_count} over the pseudobulks.")
    genes = [g for g, k in zip(genes_all, keep) if k]
    col_labels = [f"{b}_{x}" for b in blocks for x in ("comparator", "contact")]

    def _prep(M):
        return {"counts": np.rint(M[keep]).astype(np.int64), "genes": genes, "labels": col_labels,
                "cond": ["B", "A"] * len(blocks), "batch": [b for b in blocks for _ in (0, 1)],
                "design": "~ sample + group"}

    if progress is not None:
        progress(0.05, f"Fitting ~ sample + group on {len(blocks)} samples × {len(genes):,} genes in R…")
    jobs._job_check()
    f0 = fit(_prep(M0))
    table = f0["table"].copy()
    jobs._job_check()
    # ownership and leak gate (contact = arm A, comparator = arm B, per sample)
    prof = controls._type_profiles(samples, labels, genes_all)
    table["owner"], table["owners"], table["anchor_owned"] = _prox_ownership(prof, genes, anchor, target, leak_gate)
    ma = [grp[i]["a"] if i in test else np.zeros(len(samples[i]["df"]), dtype=bool) for i in range(len(samples))]
    mb = [(grp[i]["w"] > 0) if i in test else np.zeros(len(samples[i]["df"]), dtype=bool) for i in range(len(samples))]
    gate = (controls._leak_gate_groups(samples, labels, ma, mb, genes, prof, per_sample=True,
                                       called=np.nan_to_num(table["padj"].to_numpy(dtype=float), nan=1.0) < fdr)
            if leak_gate else dict(_PROX_GATE_OFF))
    flag = table["anchor_owned"].to_numpy().copy()
    if gate["applicable"]:
        lt = gate["table"].set_index("gene").reindex(genes)
        for c in ("leak_share_a", "leak_share_b", "leak_change_frac", "leak_fraction", "top_leak_type"):
            table[c] = lt[c].to_numpy()
        table["gate_flag"] = lt["leak_flag"].fillna(False).to_numpy(dtype=bool)
        flag |= table["gate_flag"].to_numpy()
    else:
        table["leak_fraction"] = np.nan
        table["gate_flag"] = False
    table["leak_flag"] = flag
    cpm = controls._cpm(prof["counts"], prof["depth"])
    gix = np.array([prof["genes"].index(g) for g in genes], dtype=np.int64)
    table["anchor_ratio"] = (cpm[prof["types"].index(anchor)][gix] / np.maximum(cpm[prof["types"].index(target)][gix], 1e-9)
                             if anchor in prof["types"] and target in prof["types"] else np.nan)

    def _calls(t):
        sig = (t["padj"].to_numpy(dtype=float) < fdr) & (np.abs(t["log2fc"].to_numpy(dtype=float)) > lfc_min)
        return sig, sig & ~flag

    sig0, pass0 = _calls(table)
    table["pass"] = pass0
    # label shuffle within sample × stratum
    rng = np.random.default_rng(seed)
    null_pass, null_sig, failed = [], [], 0
    for b in range(n_shuffle):
        jobs._job_check()
        if progress is not None:
            progress(0.15 + 0.85 * b / max(1, n_shuffle), f"Label shuffle {b + 1} / {n_shuffle}…")
        lab_s, w_s = {}, {}
        for i in test:
            st = grp[i]["strata"][cells[i]]
            lab = lab0[i].copy()
            order = np.argsort(st, kind="stable")
            bounds = np.flatnonzero(np.diff(st[order])) + 1
            for seg in np.split(order, bounds):
                lab[seg] = lab[seg][rng.permutation(len(seg))]
            _ka, w, _info = _contact_match(lab, ~lab, st)
            lab_s[i], w_s[i] = lab, w
        try:
            ts = fit(_prep(_pseudobulk(lab_s, w_s)))["table"]
        except jobs._DatasetChanged:
            raise
        except Exception:
            failed += 1
            continue
        sg, ps = _calls(ts)
        null_sig.append(int(sg.sum())); null_pass.append(int(ps.sum()))
    obs = int(pass0.sum())
    shuffle = {"n": len(null_pass), "requested": n_shuffle, "failed": failed, "observed": obs,
               "null_mean": float(np.mean(null_pass)) if null_pass else np.nan,
               "null_max": int(max(null_pass)) if null_pass else 0,
               "null_sig_mean": float(np.mean(null_sig)) if null_sig else np.nan,
               "n_ge": int(sum(v >= obs for v in null_pass)),
               "p": (1.0 + sum(v >= obs for v in null_pass)) / (len(null_pass) + 1.0) if null_pass else np.nan,
               "seed": seed}
    rows = [{k: g[k] for k in ("label", "name", "n_target", "n_anchor", "n_a_all", "n_c_all", "n_a", "n_c",
                               "n_a_dropped", "n_strata", "testable")} | {"block": block_of.get(i, "")}
            for i, g in enumerate(grp)]
    return {"kind": "contact", "table": table, "fit": {k: v for k, v in f0.items() if k != "table"},
            "samples": rows, "blocks": blocks, "by_donor": by_donor, "shuffle": shuffle, "gate": {k: v for k, v in gate.items() if k != "table"},
            "target": target, "anchor": anchor, "radius": float(radius), "min_cells": min_cells,
            "min_count": min_count, "lfc_min": lfc_min, "fdr": fdr, "n_genes": len(genes),
            "n_filtered": int((~keep).sum()), "n_sig": int(sig0.sum()), "n_pass": obs,
            "n_a": int(sum(grp[i]["n_a"] for i in test)), "n_b": int(sum(grp[i]["n_c"] for i in test)),
            "corrected": bool(samples[0]["corrected"]),
            "n_uncorrected": analysis._an_uncorrected(samples)}


def _run_contact_de(target, anchor, label_kind, seg_source, method, use_corrected, radius=None,
                    min_cells=None, n_shuffle=None, min_count=None, seed=0, leak_gate: bool = True) -> None:
    """Background job: contact-group pseudobulk DE, published into the DE state."""
    _gen = jobs._dataset_gen
    try:                                   # rpy2 thread rule: before anything else
        import rpy2.robjects as _ro_mod
        import rpy2.robjects.conversion as _rconv
        _rconv.set_conversion(_ro_mod.default_converter)
    except Exception:
        pass                               # reported by the fit with the install hint

    def _set(status, message, result=None):
        with analysis._de_lock:
            if jobs._stale_job(_gen, status, message):
                return
            analysis._de_state["status"] = status
            analysis._de_state["message"] = message
            if result is not None:
                analysis._de_state["result"] = result

    try:
        _t0 = _time.time()
        _set("running", "Collecting the contact and comparator cells…")
        samples = analysis._an_samples(seg_source, use_corrected)
        if not samples:
            _set("error", analysis._an_no_matrix_msg(seg_source))
            return
        labels, cat = stats._comp_labels(label_kind, samples, seg_source, method)
        if all(l is None for l in labels):
            _set("error", f"The active view has no {cat} labels: run Cell Type Annotation first, or use clusters.")
            return
        radius = float(_CONTACT_R_UM if not radius else radius)
        strata, defaults = [], []
        for s in samples:
            jobs._job_check()
            fine, _coarse, dflt = _prox_sample_strata(s, seg_source)
            strata.append(fine)
            if dflt:
                defaults.append(s["label"])

        def _prog(f, text=None):
            if text:
                _set("running", text)
            jobs._job_progress(text, f)

        res = _contact_compute(samples, labels, target, anchor, strata=strata, radius=radius,
                               min_cells=min_cells, min_count=min_count, n_shuffle=n_shuffle, seed=seed,
                               donors=stats._design_donors(samples), progress=_prog, leak_gate=leak_gate)
        jobs._check_dataset_gen(_gen)
        fitinfo, sh, t = res["fit"], res["shuffle"], res["table"]
        k = len(res["blocks"])
        unit = "donors" if res["by_donor"] else "samples"
        res.update({"category": cat, "seg_source": seg_source or "xenium", "seconds": _time.time() - _t0,
                    "label_a": f"{target} in contact with {anchor}", "label_b": f"{target} without contact (matched)",
                    "population": f"{target} near {anchor}", "method_label": fitinfo["method_label"],
                    "stat_name": fitinfo["stat_name"], "shrink": fitinfo["shrink"], "engine": fitinfo["engine"],
                    "mean_unit": fitinfo["mean_unit"], "design": "~ sample + group",
                    "units_note": f"n = {k} {unit}, each with a contact and a matched comparator pseudobulk",
                    "notes": list(fitinfo.get("notes") or []), "computed_defaults": defaults,
                    "lfc_title": "log2 fold change (contact / comparator), unshrunken (MLE)",
                    "up_name": "higher in contact", "down_name": "lower in contact"})
        res["title"] = (f"{target} within {radius:g} µm of {anchor} ({res['n_a']:,} cells) vs matched {target} "
                        f"without ({res['n_b']:,}) — {fitinfo['method_label']}, ~ sample + group; n = {k} {unit}")
        msg = (f"Contact groups ({cat}): {target} with a {anchor} cell within {radius:g} µm ({res['n_a']:,} cells) "
               f"vs {target} without ({res['n_b']:,} cells, reweighted to the contact cells' compartment × edge "
               f"bin × density quintile) in {k} {unit} [{', '.join(res['blocks'])}]; {fitinfo['method_label']}, "
               f"design ~ sample + group. {res['n_genes']:,} genes tested ({res['n_filtered']:,} with a total "
               f"count below {res['min_count']} not tested); {int((t['padj'] < 0.05).sum()):,} with FDR < 0.05; "
               f"{res['n_pass']:,} pass (FDR < {res['fdr']:g}, |log2FC| > {res['lfc_min']:g}, not flagged) "
               f"({'SPLIT-corrected' if res['corrected'] else 'original'} counts, weighted sums rounded; "
               f"{res['seconds']:.1f} s).")
        if res.get("n_uncorrected"):
            msg += " " + analysis._an_uncorrected_note(res["n_uncorrected"]) + "."
        msg += f"\nUnit of the p-values: {res['units_note']} (the samples, not the cells)."
        if k < 3:
            msg += f"\n⚠ Only {k} {unit}: variance estimates from fewer than 3 are unreliable."
        if sh["n"]:
            msg += (f"\nLabel shuffle within sample × stratum ({sh['n']} shuffles, seed {sh['seed']}): the "
                    f"shuffled data give {sh['null_mean']:.1f} passing genes on average (max {sh['null_max']}); "
                    f"{sh['n_ge']} shuffles give at least the observed {sh['observed']}: P = {sh['p']:.3g} "
                    f"(unit: one shuffle of the cell labels; smallest achievable {1.0 / (sh['n'] + 1):.3g}).")
            if sh["failed"]:
                msg += f" {sh['failed']} shuffles could not be fitted and were left out."
        else:
            msg += "\nNo label shuffle was run."
        g = res["gate"]
        if g.get("off"):
            msg += f"\n{_PROX_GATE_OFF_MSG} 'Pass' is then FDR and effect size alone."
        else:
            msg += (f"\nOwnership: {int(t['anchor_owned'].sum()):,} tested genes are owned by {anchor} and not by {target} (flagged). "
                    + (f"Leak gate (focal type {g.get('focal')}, {g.get('mode')}): {int(t['gate_flag'].sum()):,} genes flagged "
                       f"({g.get('n_e1', 0):,} by the signal rule, {g.get('n_e2', 0):,} by the change rule, applied to the "
                       f"{g.get('n_change_ok', 0):,} genes with FDR < {res['fdr']:g} or |log2 change| ≥ "
                       f"{controls._LEAK_CHANGE_MIN_LFC:g})."
                       if g.get("applicable") else f"Leak gate not applied: {g.get('reason')}."))
        top_x = t["pass"].to_numpy(dtype=bool) & (t["owner"].to_numpy() == anchor)
        if top_x.any():
            msg += (f" {int(top_x.sum()):,} of the {res['n_pass']:,} passing genes have {anchor} as their top owner "
                    f"({target} carries them at ≥ {100 * controls._OWN_FRAC:.0f} % of that rate, so they are not "
                    f"flagged): column anchor_ratio of the CSV; read them as possible spillover or a mixed label.")
        drop = [r for r in res["samples"] if not r["testable"]]
        if drop:
            msg += ("\nNot used (fewer than " + str(res["min_cells"]) + " cells in a group): "
                    + "; ".join(f"{r['label']} ({r['n_a']} contact, {r['n_c']} comparator)" for r in drop) + ".")
        nd = sum(r["n_a_dropped"] for r in res["samples"] if r["testable"])
        if nd:
            msg += f"\n{nd:,} contact cells sat in a stratum without comparator cells and were left out."
        if defaults:
            msg += (f"\nCompartments were not computed for {', '.join(defaults)}: the strata use the default "
                    f"edge distance and density (Confound Controls tab).")
        for nt in res["notes"]:
            msg += f"\nNote: {nt}."
        _set("done", msg, result=res)
        print(f"  Contact DE: {msg}", flush=True)
    except jobs._JobCancelled:
        raise
    except jobs._DatasetChanged:
        raise
    except ValueError as exc:
        _set("error", str(exc))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set("error", f"Contact-group differential expression failed: {str(exc)[:400]}")


def _contact_csv_header(res: dict) -> str:
    sh = res["shuffle"]
    lines = [f"# Contact-group pseudobulk differential expression: {res['label_a']} vs {res['label_b']}; "
             f"labels: {res.get('category', '')}; segmentation: {res.get('seg_source', '')}; counts: "
             f"{'SPLIT-corrected' if res['corrected'] else 'original'} (weighted sums, rounded)",
             f"# contact = a {res['anchor']} cell within {res['radius']:g} um in the same sample; comparator "
             f"weights reproduce the contact cells' compartment x edge bin x density quintile mix per sample",
             f"# method: {res.get('method_label', '')}; design: ~ sample + group; log2fc = contact / comparator",
             f"# unit of the p-values: {res.get('units_note', '')}",
             f"# pass = padj < {res['fdr']:g} and |log2fc| > {res['lfc_min']:g} and not leak_flag; leak_flag = "
             f"anchor_owned ({res['anchor']} owns the gene, {res['target']} does not) or gate_flag (leak gate)"]
    if sh["n"]:
        lines.append(f"# label shuffle within sample x stratum: {sh['n']} shuffles, seed {sh['seed']}; passing genes "
                     f"observed {sh['observed']}, null mean {sh['null_mean']:.2f}, max {sh['null_max']}; P = {sh['p']:.4g}")
    for r in res["samples"]:
        lines.append(f"# sample {r['label']} ({r['name']}): {r['n_a']} contact cells ({r['n_a_dropped']} dropped), "
                     f"{r['n_c']} comparator cells, {r['n_strata']} shared strata; "
                     + (f"block {r['block']}" if r["testable"] else "not used"))
    if res.get("n_uncorrected"):
        lines.append(f"# {analysis._an_uncorrected_note(res['n_uncorrected'])}")
    return "\n".join(lines) + "\n"


def _prox_samples_view(res):
    """Per-sample cell counts of a proximity or contact result (below the status)."""
    if not res or res.get("kind") not in ("proximity", "contact"):
        return []
    num = lambda v: f"{v:,}"
    if res["kind"] == "proximity":
        cols = [("label", "Sample", None), ("name", "Name", lambda v: str(v)[:38]),
                ("cond", "Condition", lambda v: v or "–"), ("donor", "Donor", lambda v: v or "–"),
                ("n_target", "T cells", num), ("n_anchor", "X cells", num), ("n_near", "Near", num),
                ("n_far", "Far", num), ("median_d", "Median d (µm)", lambda v: "–" if not np.isfinite(v) else f"{v:.0f}"),
                ("note", "", None)]
    else:
        cols = [("label", "Sample", None), ("name", "Name", lambda v: str(v)[:38]),
                ("block", "Block", lambda v: v or "not used"), ("n_target", "T cells", num),
                ("n_anchor", "X cells", num), ("n_a", "Contact", num), ("n_a_dropped", "Contact dropped", num),
                ("n_c", "Comparator", num), ("n_strata", "Shared strata", num)]
    return [stats._samples_table(res["samples"], cols)]


# ─── Controls of the two modes in the Differential Expression tab ─────────────
_PROX_DE_MODES = [{"label": " Proximity to a cell type", "value": "proximity"},
             {"label": " Contact groups (pseudobulk)", "value": "contact"}]


def _prox_controls() -> list:
    """Rows shown in the Differential Expression tab in the proximity and contact modes."""
    S, L = analysis._AN_ROW_STYLE, analysis._AN_LABEL_STYLE
    num = controls._num
    return [
        html.Div([
            html.Span("Cell types from:", style=L),
            dcc.Dropdown(id="prox-labels", options=[{"label": "Cluster (active cluster set)", "value": "cluster"}],
                         value=None, placeholder="Cell type annotation / cluster…",
                         style={"flex": "1", "minWidth": "160px", "fontSize": "11px"}),
            html.Span("Cells (T):", style=L),
            dcc.Dropdown(id="prox-target", options=[], value=None, placeholder="Cell type to test…",
                         style={"flex": "1", "minWidth": "160px", "fontSize": "11px"}),
            html.Span("near (X):", style=L),
            dcc.Dropdown(id="prox-anchor", options=[], value=None, placeholder="Anchor cell type…",
                         style={"flex": "1", "minWidth": "160px", "fontSize": "11px"}),
        ], style=S),
        html.Div([
            html.Span("Near ≤ (µm):", style=L), num(id="prox-near", value=_PROX_NEAR_UM, min=1, max=1000, step=1),
            html.Span("Far ≥ (µm):", style=L), num(id="prox-far", value=_PROX_FAR_UM, min=2, max=5000, step=1),
            html.Span("Random anchor sets:", style=L), num(id="prox-nrand", value=_PROX_N_RANDOM, min=1, max=2000, step=1),
            html.Span("Density covariate:", style=L),
            dcc.Dropdown(id="prox-source", options=[{"label": "auto (type with the most transcripts, not T or X)", "value": "auto"},
                                                    {"label": "none", "value": "none"}],
                         value="auto", clearable=False, style={"width": "220px", "fontSize": "11px"}),
            dcc.Checklist(id="prox-length",
                          options=[{"label": html.Span(" Fit exp(−d / L)", style={"fontSize": "11px", "color": core.TEXT}),
                                    "value": "on"}],
                          value=[], inputStyle={"marginRight": "3px"}, style={"display": "inline-flex"}),
        ], id="de-prox-params", style=S),
        html.Div([
            html.Span("Contact radius (µm):", style=L), num(id="contact-radius", value=_CONTACT_R_UM, min=1, max=1000, step=1),
            html.Span("Min. cells / sample / group:", style=L), num(id="contact-min-cells", value=_CONTACT_MIN_CELLS, min=1, step=1),
            html.Span("Label shuffles:", style=L), num(id="contact-nshuffle", value=_CONTACT_N_SHUFFLE, min=0, max=5000, step=1),
        ], id="de-contact-params", style={**S, "display": "none"}),
        html.Div([
            dbc.Button("Samples & conditions…", id="prox-design-open-btn", color="secondary", size="sm",
                       n_clicks=0, style={"fontSize": "11px", "padding": "2px 8px"}),
            html.Span(id="prox-design-summary", style=L),
        ], style=S),
    ]
