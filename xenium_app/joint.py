"""Joint clustering across samples: one normalisation, gene selection and PCA over the
cells of every loaded sample, Harmony on a chosen variable when harmonypy is installed,
then one neighbour graph, one clustering and one UMAP (compute + layout).

The result is a cluster set every sample shares, so "cluster 3" means the same thing in
every sample (pseudobulk differential expression by cluster, composition tests).
"""

import os
import json
import hashlib
import threading
import time as _time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from dash import dcc, html
from dash import dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from . import core, jobs, caching, dataview, ui, figures, analysis, stats, controls, clustering, clustersets
from . import provenance


# ─── State ────────────────────────────────────────────────────────────────────
_jc_state: dict = {"status": "idle", "message": "", "result": None, "version": 0}
_jc_lock = threading.Lock()

_JC_N_HVG = 2000             # genes kept for the PCA (all genes of a panel that has fewer)
_JC_N_PCS = 30
_JC_N_NEIGHBORS = 15
_JC_RESOLUTION = 1.0
_JC_THETA = 2.0              # Harmony diversity penalty (its default)
_JC_SEED = 0
_JC_SCALE_CLIP = 10.0        # scaled values are capped at this many standard deviations
_JC_HVG_BINS = 20            # mean-expression bins within which the dispersion is standardised
_JC_MIN_COUNTS = 1           # a cell needs at least this many counts over the shared genes
_JC_LISI_K = 30              # neighbours of the mixing index
_JC_LISI_QUERY = 5000        # cells the mixing index is computed for (random, fixed seed)
_JC_CACHE_VERSION = 2          # 2: matrix content and backend availability in the key; cell ids stored
# cost of the part before the graph, per million cells (measured, README "Performance at
# study scale"): working copies of the count matrix; normalisation, gene selection and
# PCA; Harmony; the UMAP
_JC_MATRIX_GB_PER_M_CELLS = 8.0
_JC_MIN_PER_M_CELLS = 2.0
_JC_HARMONY_MIN_PER_M_CELLS = 6.0
_JC_UMAP_MIN_PER_M_CELLS = 15.0

# variant → (cluster set name, prefix of the per-cell UMAP columns)
_JC_VARIANTS = {"plain": ("Joint clusters", "joint_umap"),
                "harmony": ("Joint clusters (Harmony)", "joint_harmony_umap"),
                "uncorrected": ("Joint clusters (uncorrected)", "joint_uncorrected_umap")}

_JC_HARMONY_HINT = ("harmonypy is not installed: the clusters come from the joint PCA without batch "
                    "correction and the set is named 'Joint clusters (uncorrected)'. For Harmony: "
                    "pip install harmonypy")
_JC_UMAP_HINT = "no joint UMAP: umap-learn could not be imported (pip install umap-learn)"
_JC_CONFOUND_WARNING = (
    "⚠ Integration removes differences between the levels of the integration variable. When the "
    "condition is confounded with it (every level belongs to one condition - always the case for "
    "'sample', and for donor or batch when a donor or batch is not split over the conditions), a real "
    "difference between the conditions is, to Harmony, a difference between levels: cells of a state "
    "that exists in one condition only can be pulled into a cluster of the other condition. Check the "
    "clusters against known markers, and compare with the run without integration.")

_JC_NOTE = (
    "Joint clustering needs at least two loaded samples. All cells of the view (active subset, active "
    "counts) are put in one matrix over the panel genes every sample has: counts per 10,000 of the "
    "cell's total over these genes, log1p; genes ranked by their dispersion (variance ÷ mean of the "
    "normalised values, standardised within 20 bins of mean expression) in each sample and by the mean "
    "of that score over the samples, the top N kept; each gene divided by its standard deviation over "
    "all cells, centred, capped at 10; PCA over all cells. With Harmony (Korsunsky et al., Nat Methods "
    "2019; the harmonypy package) the principal components are corrected for the chosen variable: the "
    "sample, or the donor / batch of the sample design (a sample without one counts as its own level). "
    "The neighbour graph, the clusters (Leiden when leidenalg and igraph are installed) and the UMAP "
    "are then computed once on all cells. Diagnostics: for every cluster the entropy of the "
    "integration variable among its cells ÷ log(number of levels) - 0: one level only, values near the "
    "entropy of the whole view: mixed as the view is - and a neighbourhood mixing index (inverse "
    "Simpson index of the variable among the 30 nearest neighbours of 5,000 random cells; a simplified "
    "LISI without its distance weighting; 1: neighbours of one level, upper reference: the index of "
    "the whole view), each before and after the correction. Select the set under Cluster Set; the UMAP "
    "panel then shows the joint embedding of all samples. " + _JC_CONFOUND_WARNING)


def _jc_result():
    """The published result when it belongs to the loaded dataset, else None."""
    with _jc_lock:
        res = _jc_state.get("result")
    if not res or res.get("gen") != jobs._dataset_gen:
        return None
    return res


# ─── Matrix, normalisation, gene selection, PCA ───────────────────────────────
def _jc_collect(samples) -> dict:
    """One counts matrix over the cells of all samples and the genes all of them have.

    samples: analysis._an_samples dicts. Returns X (cells × genes CSR, float32), genes,
    sample (sample index per row), take (per sample: positions in its df of the rows that
    are in X - cells with a matrix row and at least _JC_MIN_COUNTS counts over the shared
    genes), n_df (cells per sample table)."""
    genes = analysis._an_common_genes(samples)
    if not genes:
        raise ValueError("The samples share no panel gene.")
    blocks, take, code = [], [], []
    for si, s in enumerate(samples):
        jobs._job_check()
        rows = np.asarray(s["rows"], dtype=np.int64)[:len(s["df"])]
        cols = np.array([s["gni"][g] for g in genes], dtype=np.int64)
        pos = np.flatnonzero(rows >= 0)
        m = s["csc"][:, cols].tocsr()[rows[pos]].astype(np.float32)
        ok = np.asarray(m.sum(axis=1)).ravel() >= _JC_MIN_COUNTS
        blocks.append(m[ok])
        take.append(pos[ok])
        code.append(np.full(int(ok.sum()), si, dtype=np.int32))
    X = sp.vstack(blocks, format="csr") if blocks else sp.csr_matrix((0, len(genes)), dtype=np.float32)
    return {"X": X, "genes": list(genes), "sample": np.concatenate(code) if code else np.zeros(0, np.int32),
            "take": take, "n_df": [len(s["df"]) for s in samples]}


def _jc_normalise(X, scale: float = None) -> "sp.csr_matrix":
    """log1p(count ÷ the cell's total over the columns × *scale*) as CSR float32."""
    scale = float(analysis._AN_SCALE if scale is None else scale)
    X = sp.csr_matrix(X, dtype=np.float32, copy=True)
    tot = np.asarray(X.sum(axis=1)).ravel()
    tot[tot <= 0] = 1.0
    X = sp.diags((scale / tot).astype(np.float32)).dot(X).tocsr()
    X.data = np.log1p(X.data)
    return X


def _jc_moments(Xn) -> tuple:
    """(mean, variance with n − 1) of every column of a sparse matrix."""
    n = Xn.shape[0]
    mean = np.asarray(Xn.sum(axis=0)).ravel().astype(np.float64) / max(n, 1)
    sq = np.asarray(Xn.multiply(Xn).sum(axis=0)).ravel().astype(np.float64)
    var = np.maximum(sq - n * mean ** 2, 0.0) / max(n - 1, 1)
    return mean, var


def _jc_dispersion_z(mean, var, n_bins: int = _JC_HVG_BINS) -> np.ndarray:
    """Standardised dispersion of every gene of one sample: log(variance ÷ mean), minus
    the mean and divided by the standard deviation of the genes in the same quantile bin
    of mean expression (−inf for a gene the sample does not express; a bin with one
    gene, or without spread, gives 0)."""
    mean, var = np.asarray(mean, dtype=np.float64), np.asarray(var, dtype=np.float64)
    z = np.full(len(mean), -np.inf)
    ok = (mean > 0) & (var > 0)
    if not ok.any():
        return z
    disp = np.log(var[ok] / mean[ok])
    k = int(max(1, min(n_bins, ok.sum() // 5)))
    rank = np.argsort(np.argsort(mean[ok], kind="stable"), kind="stable")
    b = (rank * k // int(ok.sum())).astype(np.int64)
    cnt = np.bincount(b, minlength=k).astype(np.float64)
    mu = np.bincount(b, weights=disp, minlength=k) / np.maximum(cnt, 1)
    sd = np.sqrt(np.maximum(np.bincount(b, weights=disp ** 2, minlength=k) / np.maximum(cnt, 1) - mu ** 2, 0.0))
    zz = np.where(sd[b] > 0, (disp - mu[b]) / np.where(sd[b] > 0, sd[b], 1.0), 0.0)
    z[ok] = zz
    return z


def _jc_select_genes(Xn, sample, n_top: int = _JC_N_HVG) -> np.ndarray:
    """Columns of the *n_top* most variable genes, sorted: every gene is scored in each
    sample on its own (_jc_dispersion_z) and ranked by the mean score over the samples, so
    a gene that only differs BETWEEN samples is not preferred. All columns with a non-zero
    variance when there are at most *n_top*."""
    sample = np.asarray(sample)
    G = Xn.shape[1]
    _m, v_all = _jc_moments(Xn)
    alive = v_all > 0
    if int(alive.sum()) <= int(n_top):
        return np.flatnonzero(alive)
    zs = []
    for s in np.unique(sample):
        jobs._job_check()
        m, v = _jc_moments(Xn[np.flatnonzero(sample == s)])
        z = _jc_dispersion_z(m, v)
        zs.append(np.where(np.isfinite(z), z, np.nan))
    Z = np.vstack(zs)
    n_ok = np.isfinite(Z).sum(axis=0)
    score = np.where(n_ok > 0, np.nansum(Z, axis=0) / np.maximum(n_ok, 1), -np.inf)
    score[~alive] = -np.inf
    # a gene expressed in fewer samples loses ties
    order = np.lexsort((np.arange(G), -n_ok, -score))
    return np.sort(order[:int(n_top)])


def _jc_scale(Xn) -> tuple:
    """Columns divided by their standard deviation over all cells and capped so that the
    centred value is at most _JC_SCALE_CLIP: (CSC float32, sd). The matrix stays sparse;
    the PCA centres it."""
    Xc = sp.csc_matrix(Xn, dtype=np.float32, copy=True)
    mean, var = _jc_moments(Xc)
    sd = np.sqrt(var)
    sd[sd <= 0] = 1.0
    cap = (_JC_SCALE_CLIP + mean / sd).astype(np.float32)
    col = np.repeat(np.arange(Xc.shape[1]), np.diff(Xc.indptr))
    Xc.data = np.minimum(Xc.data / sd[col].astype(np.float32), cap[col])
    return Xc, sd


def _jc_pca(Xs, n_pcs: int = _JC_N_PCS, seed: int = _JC_SEED) -> tuple:
    """(scores cells × k float32, explained variance ratio) of the centred PCA of a
    sparse matrix, without densifying it (ARPACK on the implicitly centred matrix; the
    same start vector and signs for a fixed *seed*)."""
    from sklearn.decomposition import PCA
    n, p = Xs.shape
    k = int(max(1, min(int(n_pcs), n - 1, p - 1)))
    if min(n, p) <= 3 or k >= min(n, p) - 1:
        pca = PCA(n_components=min(k, min(n, p)), svd_solver="full")
        Z = pca.fit_transform(np.asarray(Xs.todense() if sp.issparse(Xs) else Xs, dtype=np.float64))
    else:
        pca = PCA(n_components=k, svd_solver="arpack", random_state=int(seed))
        try:
            Z = pca.fit_transform(Xs)
        except TypeError:        # a scikit-learn without sparse PCA
            Z = pca.fit_transform(np.asarray(Xs.todense(), dtype=np.float32))
    return np.ascontiguousarray(Z, dtype=np.float32), np.asarray(pca.explained_variance_ratio_, dtype=np.float64)


# ─── Harmony ──────────────────────────────────────────────────────────────────
def _jc_harmony_module():
    """The harmonypy module, or None when it is not installed (or does not import)."""
    try:
        import harmonypy
        return harmonypy
    except Exception:
        return None


def _jc_harmony(pcs, batch, seed: int = _JC_SEED, theta: float = _JC_THETA) -> np.ndarray:
    """Harmony-corrected embedding (cells × PCs, float32): harmonypy.run_harmony on the
    principal components with the level of every cell as the one covariate, its defaults
    otherwise (theta 2, automatic ridge penalty, at most 10 rounds) and random_state =
    *seed*. Raises ImportError without harmonypy."""
    hm = _jc_harmony_module()
    if hm is None:
        raise ImportError("harmonypy is not installed")
    pcs = np.asarray(pcs, dtype=np.float32)
    ho = hm.run_harmony(pcs, {"level": np.asarray(batch).astype(str)}, ["level"], theta=float(theta),
                        random_state=int(seed), verbose=False)
    Z = np.asarray(ho.Z_corr)
    if Z.shape != pcs.shape:
        Z = Z.T
    return np.ascontiguousarray(Z, dtype=np.float32)


def _jc_levels(var: str, tables, donors=None, batches=None) -> list:
    """Level of the integration variable for every sample of the view. "sample": the
    sample itself; "donor" / "batch": the value of the sample design, a sample without one
    being its own level (named after the sample)."""
    own = [f"{t['label']} {t['name']}" for t in tables]
    if var == "sample":
        return own
    vals = donors if var == "donor" else batches
    vals = list(vals) if vals is not None else [""] * len(tables)
    return [str(v) if str(v or "").strip() else f"(sample {o})" for v, o in zip(vals, own)]


def _jc_confounded(conds, levels) -> "str | None":
    """How the conditions of the design relate to the integration variable: None when no
    sample is assigned to both conditions; "full" when every level belongs to one
    condition (the variable separates the conditions: a condition difference is a level
    difference); "partial" when some level holds samples of both conditions."""
    pairs = {(lv, c) for lv, c in zip(levels, conds) if c in ("A", "B")}
    if {c for _lv, c in pairs} != {"A", "B"}:
        return None
    per = {}
    for lv, c in pairs:
        per.setdefault(lv, set()).add(c)
    return "full" if all(len(v) == 1 for v in per.values()) else "partial"


# ─── Diagnostics ──────────────────────────────────────────────────────────────
def _jc_entropy(labels, batch) -> dict:
    """Entropy of *batch* among the cells of every cluster ÷ log(number of levels).

    Returns clusters (sorted), levels, counts [clusters × levels], entropy (per cluster,
    0 … 1), mean (cell-weighted mean over the clusters) and reference (the same quantity
    for all cells together: what a cluster mixed as the view is would give)."""
    labels, batch = np.asarray(labels), np.asarray(batch).astype(str)
    cl, ci = np.unique(labels, return_inverse=True)
    lv, li = np.unique(batch, return_inverse=True)
    N = controls._bincount2(ci, li, len(cl), len(lv)).astype(np.float64)
    L = max(len(lv), 2)

    def _h(rows):
        tot = rows.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(tot > 0, rows / np.where(tot > 0, tot, 1.0), 0.0)
            return np.maximum(-(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)).sum(axis=1), 0.0) / np.log(L)

    h = _h(N)
    n = N.sum(axis=1)
    return {"clusters": cl, "levels": lv, "counts": N, "entropy": h,
            "mean": float((h * n).sum() / max(n.sum(), 1.0)), "reference": float(_h(N.sum(axis=0, keepdims=True))[0])}


def _jc_mixing(emb, batch, k: int = _JC_LISI_K, n_query: int = _JC_LISI_QUERY, seed: int = _JC_SEED) -> dict:
    """Neighbourhood mixing of *batch* in the embedding: for up to *n_query* random cells
    the inverse Simpson index 1 / Σ_b p_b² of the levels among their *k* nearest
    neighbours (Euclidean, the cell itself left out). A simplified LISI (Korsunsky et al.
    2019): unweighted counts over a fixed k instead of perplexity-based distance weights.
    Returns median, mean, reference (the index of all cells together: the value of perfect
    mixing), n (cells used), k."""
    emb = np.asarray(emb, dtype=np.float32)
    lv, li = np.unique(np.asarray(batch).astype(str), return_inverse=True)
    n = len(emb)
    k = int(min(k, n - 1))
    p_all = np.bincount(li, minlength=len(lv)) / max(n, 1)
    ref = float(1.0 / np.sum(p_all ** 2)) if n else np.nan
    if k < 1:
        return {"median": np.nan, "mean": np.nan, "reference": ref, "n": 0, "k": k}
    rng = np.random.default_rng(seed)
    q = np.sort(rng.choice(n, size=min(int(n_query), n), replace=False))
    sq = np.einsum("ij,ij->i", emb, emb)
    out = np.empty(len(q))
    step = max(1, int(2e8 // max(n * 4, 1)))
    for i0 in range(0, len(q), step):
        jobs._job_check()
        qi = q[i0:i0 + step]
        d = sq[qi][:, None] - 2.0 * (emb[qi] @ emb.T) + sq[None, :]
        d[np.arange(len(qi)), qi] = np.inf
        nb = np.argpartition(d, k - 1, axis=1)[:, :k]
        cnt = controls._bincount2(np.repeat(np.arange(len(qi)), k), li[nb].ravel(), len(qi), len(lv))
        p = cnt / float(k)
        out[i0:i0 + step] = 1.0 / np.sum(p ** 2, axis=1)
    return {"median": float(np.median(out)), "mean": float(out.mean()), "reference": ref, "n": int(len(q)), "k": k}


def _jc_umap(emb, seed: int = _JC_SEED) -> "np.ndarray | None":
    """UMAP of the embedding (cells × 2, float32), or None when umap-learn does not
    import. Same settings as the UMAP of a resegmentation run."""
    emb = np.asarray(emb, dtype=np.float32)
    n = len(emb)
    if n < 4:
        return None
    try:
        import sys, types, warnings
        # see clustering._compute_reseg_clusters_umap_impl: umap imports tensorflow for its
        # parametric variant, which is not used here
        if "tensorflow" not in sys.modules:
            sys.modules["tensorflow"] = types.ModuleType("tensorflow")
        from umap.umap_ import UMAP
    except Exception as exc:
        print(f"  Joint clustering: umap-learn is not available ({exc})", flush=True)
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value.*overridden")
        red = UMAP(n_neighbors=min(15, n - 1), min_dist=0.1, **clustering._umap_seed_kwargs(n, int(seed)))
        return np.asarray(red.fit_transform(emb), dtype=np.float32)


# ─── Cache ────────────────────────────────────────────────────────────────────
def _jc_cache_dir() -> str:
    tag = hashlib.md5(caching._dataset_key().encode()).hexdigest()[:8]
    return os.path.join(core.CACHE_BASE_DIR, f"joint_{tag}")


def _jc_backends() -> dict:
    """Optional packages a joint run uses when they are installed."""
    import importlib.util
    return {"umap": importlib.util.find_spec("umap") is not None,
            "leiden": bool(clustersets._cs_leiden_available())}


def _jc_cache_tag(samples, seg_source, params: dict, var_levels) -> str:
    """Identifies a run: the samples and their cells (ids and matrix), the counts used,
    the parameters and the level of every sample."""
    h = hashlib.sha1()
    # which optional backends can run is part of the key: a run made without umap-learn or
    # without Leiden (KMeans fallback) is not handed back once they are installed
    h.update(json.dumps({"v": _JC_CACHE_VERSION, "seg": str(seg_source or "xenium"),
                         "backends": _jc_backends(),
                         "params": {k: params[k] for k in sorted(params)},
                         "levels": list(var_levels)}, sort_keys=True, default=str).encode())
    for s in samples:
        h.update(str(s["key"]).encode())
        h.update(b"corrected" if s["corrected"] else b"original")
        h.update(analysis._an_matrix_digest(s["csc"]).encode())
        h.update(pd.util.hash_pandas_object(pd.Index(s["df"].index.astype(str)), index=False).values.tobytes())
    return h.hexdigest()[:16]


def _jc_cache_write(tag: str, arrays: dict, meta: dict) -> None:
    d = _jc_cache_dir()
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f"{tag}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, os.path.join(d, f"{tag}.npz"))
    with open(os.path.join(d, f"{tag}.json.tmp"), "w") as fh:
        json.dump(meta, fh, default=str)
    os.replace(os.path.join(d, f"{tag}.json.tmp"), os.path.join(d, f"{tag}.json"))


def _jc_cache_read(tag: str):
    """(arrays, meta) of a cached run, or None."""
    d = _jc_cache_dir()
    f_npz, f_js = os.path.join(d, f"{tag}.npz"), os.path.join(d, f"{tag}.json")
    if not (os.path.exists(f_npz) and os.path.exists(f_js)):
        return None
    try:
        with open(f_js) as fh:
            meta = json.load(fh)
        with np.load(f_npz, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        return arrays, meta
    except Exception as exc:
        print(f"  Joint clustering: cache entry {tag} is unreadable ({exc})", flush=True)
        return None


# ─── The job ──────────────────────────────────────────────────────────────────
def _jc_params(params=None) -> dict:
    p = {"integrate": "sample", "n_hvg": _JC_N_HVG, "n_pcs": _JC_N_PCS, "n_neighbors": _JC_N_NEIGHBORS,
         "resolution": _JC_RESOLUTION, "theta": _JC_THETA, "umap": True, "seed": _JC_SEED}
    p.update({k: v for k, v in (params or {}).items() if v is not None and k in p})
    p["integrate"] = p["integrate"] if p["integrate"] in ("none", "sample", "donor", "batch") else "sample"
    p["n_hvg"] = int(max(50, min(20000, p["n_hvg"])))
    p["n_pcs"] = int(max(2, min(100, p["n_pcs"])))
    p["n_neighbors"] = int(max(2, min(200, p["n_neighbors"])))
    p["resolution"] = float(max(0.01, min(10.0, p["resolution"])))
    p["theta"] = float(max(0.0, min(20.0, p["theta"])))
    p["umap"], p["seed"] = bool(p["umap"]), int(p["seed"])
    return p


def _jc_compute(samples, levels, p: dict, harmony: bool, step=None) -> dict:
    """The computation on analysis samples (no app state): returns take, sample, genes,
    hvg, pcs, emb (= pcs without Harmony), labels (1 … k by size), labels_before (clusters
    of the uncorrected PCs, when Harmony ran), umap (or None), backend info, timings."""
    step = step or (lambda msg, frac=None: None)
    tm, t0 = {}, _time.time()

    def _lap(name):
        nonlocal t0
        tm[name] = _time.time() - t0
        t0 = _time.time()

    step("Collecting the cells of all samples…", 0.02)
    col = _jc_collect(samples)
    X, code = col["X"], col["sample"]
    n = X.shape[0]
    if n < 20:
        raise ValueError(f"Only {n} cells with counts in the view: too few to cluster.")
    step(f"Normalising {n:,} cells × {X.shape[1]:,} genes…", 0.08)
    Xn = _jc_normalise(X)
    del X
    _lap("collect_normalise")
    step("Selecting variable genes…", 0.15)
    hvg = _jc_select_genes(Xn, code, p["n_hvg"])
    if len(hvg) < 3:
        raise ValueError("Fewer than 3 genes vary across the cells of the view.")
    Xs, _sd = _jc_scale(Xn[:, hvg])
    del Xn
    _lap("genes_scale")
    step(f"PCA of {n:,} cells × {len(hvg):,} genes…", 0.22)
    pcs, evr = _jc_pca(Xs, p["n_pcs"], p["seed"])
    del Xs
    _lap("pca")
    batch = np.asarray(levels, dtype=object)[code].astype(str)
    emb, labels_before = pcs, None
    if harmony:
        step(f"Harmony on {len(set(levels))} levels of the {p['integrate']}…", 0.35)
        emb = _jc_harmony(pcs, batch, p["seed"], p["theta"])
        _lap("harmony")
        step("Clustering the uncorrected components (for the diagnostics)…", 0.5)
        labels_before, _info0 = clustersets._cs_graph_cluster(pcs, n_neighbors=p["n_neighbors"],
                                                             resolution=p["resolution"], seed=p["seed"])
        _lap("cluster_before")
    step("Neighbour graph and clusters…", 0.62)
    labels, info = clustersets._cs_graph_cluster(emb, n_neighbors=p["n_neighbors"], resolution=p["resolution"],
                                                 seed=p["seed"])
    _lap("cluster")
    um = None
    if p["umap"]:
        step(f"Joint UMAP of {n:,} cells…", 0.75)
        um = _jc_umap(emb, p["seed"])
        _lap("umap")
    step("Diagnostics…", 0.95)
    diag = {"after": {"entropy": _jc_entropy(labels, batch), "mixing": _jc_mixing(emb, batch, seed=p["seed"])}}
    if harmony:
        diag["before"] = {"entropy": _jc_entropy(labels_before, batch),
                          "mixing": _jc_mixing(pcs, batch, seed=p["seed"])}
    _lap("diagnostics")
    return {"take": col["take"], "n_df": col["n_df"], "sample": code, "genes": col["genes"], "hvg": hvg,
            "pcs": pcs, "emb": emb, "evr": evr, "labels": np.asarray(labels, dtype=np.int64) + 1,
            "labels_before": None if labels_before is None else np.asarray(labels_before, dtype=np.int64) + 1,
            "umap": um, "info": info, "batch": batch, "diag": diag, "timings": tm}


def _jc_diag_table(tables, levels, code, labels, labels_before=None, batch=None) -> pd.DataFrame:
    """One row per joint cluster: cells, cells per sample, entropy of the integration
    variable (÷ log levels)."""
    S = len(tables)
    k = int(labels.max()) if len(labels) else 0
    N = controls._bincount2(labels - 1, code, k, S)
    ent = _jc_entropy(labels, batch)
    emap = dict(zip(ent["clusters"].tolist(), ent["entropy"].tolist()))
    rows = []
    for c in range(1, k + 1):
        r = {"cluster": c, "n_cells": int(N[c - 1].sum()), "entropy": float(emap.get(c, np.nan))}
        for si, t in enumerate(tables):
            r[f"n_{t['label']}"] = int(N[c - 1, si])
        tot = max(N[c - 1].sum(), 1)
        r["largest_sample"] = tables[int(N[c - 1].argmax())]["label"]
        r["largest_share"] = float(N[c - 1].max() / tot)
        rows.append(r)
    return pd.DataFrame(rows)


def _jc_diag_text(diag: dict, var: str) -> str:
    def _one(d):
        e, m = d["entropy"], d["mixing"]
        return (f"{len(e['clusters'])} clusters, mean entropy {e['mean']:.3f} (whole view {e['reference']:.3f}; "
                f"{int((e['entropy'] < 0.5 * e['reference']).sum())} cluster(s) below half of it), mixing index "
                f"{m['median']:.2f} (median over {m['n']:,} cells, k = {m['k']}; perfect mixing {m['reference']:.2f})")
    if "before" in diag:
        return (f"Mixing of the {var} - before Harmony: {_one(diag['before'])}; after: {_one(diag['after'])}.")
    return f"Mixing of the {var}, no correction: {_one(diag['after'])}."


def _run_joint(seg_source, use_corrected: bool = False, params=None) -> None:
    """Background job: joint clustering of the samples of the active view."""
    _gen = jobs._dataset_gen
    p = _jc_params(params)

    def _set(status, message, result=None):
        with _jc_lock:
            if jobs._stale_job(_gen, status, message):
                return
            _jc_state["status"], _jc_state["message"] = status, message
            if result is not None:
                _jc_state["result"] = result
            if status in ("done", "error"):
                _jc_state["version"] += 1
        if status == "done":
            provenance._prov_record("joint", seg_source=seg_source, corrected=use_corrected, params=dict(params or {}), packages=("harmonypy", "leidenalg", "igraph", "umap-learn"), uses_design=True, message=message, result=result)

    try:
        _t0 = _time.time()
        _set("running", "Collecting the samples…")
        if analysis._an_alt_res(seg_source) is not None:
            _set("error", "Joint clustering works on the loaded samples (Xenium segmentation): a Baysor / "
                          "Proseg / image segmentation run belongs to one sample. Switch the segmentation "
                          "source to Xenium.")
            return
        samples = analysis._an_samples(seg_source, use_corrected)
        tables = stats._view_tables(seg_source)
        seg_tag = controls._seg_tag(seg_source)
        if len(samples) < 2 or len(samples) != len(tables):
            _set("error", "Joint clustering needs at least two loaded samples with a count matrix "
                          "(Load Sample → add a sample).")
            return
        S = len(samples)
        conds, batches = stats._design_of(tables)
        donors = stats._design_donors(tables)
        var = p["integrate"]
        note_missing = ""
        if var == "none":
            levels = _jc_levels("sample", tables)
        else:
            levels = _jc_levels(var, tables, donors, batches)
            if len(set(levels)) < 2:
                _set("error", f"All samples have the same {var}: there is nothing to integrate over. "
                              f"Pick 'sample', or give the samples their {var} under Samples & conditions.")
                return
            vals = donors if var == "donor" else batches if var == "batch" else None
            if vals is not None and any(not str(v or "").strip() for v in vals):
                note_missing = (f" Samples without a {var} in the design count as their own level: "
                                + ", ".join(t["label"] for t, v in zip(tables, vals) if not str(v or "").strip()) + ".")
        have_harmony = _jc_harmony_module() is not None
        harmony = var != "none" and have_harmony
        variant = "harmony" if harmony else ("plain" if var == "none" else "uncorrected")
        name, col_prefix = _JC_VARIANTS[variant]
        p_run = dict(p, harmony=harmony)
        if not harmony:
            p_run["theta"] = None
        tag = _jc_cache_tag(samples, seg_source, p_run, levels if var != "none" else [])
        cached = _jc_cache_read(tag)
        from_cache = False
        code = None
        if cached is not None:
            arr, meta = cached
            try:
                take = [arr[f"take_{i}"] for i in range(S)]
                ok = all((len(tk) == 0 or tk.max() < len(s["df"])) for tk, s in zip(take, samples))
                if ok and (not p["umap"] or "umap" in arr or meta.get("umap_missing")):
                    r = {"take": take, "n_df": [len(s["df"]) for s in samples], "sample": arr["sample"],
                         "genes": meta["genes_n"], "hvg": arr["hvg"], "pcs": arr["pcs"], "emb": arr["emb"],
                         "evr": arr["evr"], "labels": arr["labels"],
                         "labels_before": arr["labels_before"] if "labels_before" in arr else None,
                         "umap": arr["umap"] if "umap" in arr else None, "info": meta["info"],
                         "timings": meta.get("timings", {})}
                    r["batch"] = np.asarray(levels, dtype=object)[r["sample"]].astype(str)
                    r["diag"] = {"after": {"entropy": _jc_entropy(r["labels"], r["batch"]), "mixing": meta["mixing_after"]}}
                    if r["labels_before"] is not None:
                        r["diag"]["before"] = {"entropy": _jc_entropy(r["labels_before"], r["batch"]),
                                               "mixing": meta["mixing_before"]}
                    from_cache = True
            except Exception as exc:
                print(f"  Joint clustering: cache entry {tag} not used ({exc})", flush=True)
        if not from_cache:
            def _step(msg, frac=None):
                _set("running", msg)
                if frac is not None:
                    jobs._job_progress(fraction=frac)
            # the cost up front (two clusterings when Harmony runs: before and after)
            _n = int(sum(int((np.asarray(smp["rows"])[:len(smp["df"])] >= 0).sum()) for smp in samples))
            _stop, _est = clustersets._cs_size_guard(
                "Joint clustering", _n, p["n_neighbors"], passes=2 if harmony else 1,
                matrix_gb=_JC_MATRIX_GB_PER_M_CELLS * _n / 1e6,
                extra_minutes=(_JC_MIN_PER_M_CELLS + (_JC_HARMONY_MIN_PER_M_CELLS if harmony else 0.0)
                               + (_JC_UMAP_MIN_PER_M_CELLS if p["umap"] else 0.0)) * _n / 1e6)
            if _stop:
                _set("error", _stop)
                return
            if _est:
                print(f"  Joint clustering: {_est}", flush=True)
                _set("running", _est)
            r = _jc_compute(samples, levels, p, harmony, step=_step)
        code = r["sample"]
        jobs._check_dataset_gen(_gen)
        jobs._job_shield("Installing the joint cluster set…")
        n_genes = r["genes"] if isinstance(r["genes"], int) else len(r["genes"])
        if not from_cache:
            arrays = {"sample": code, "hvg": r["hvg"], "pcs": r["pcs"], "emb": r["emb"], "evr": r["evr"],
                      "labels": r["labels"], **{f"take_{i}": tk for i, tk in enumerate(r["take"])},
                      # the embedded cells by id: what a later session re-attaches the UMAP by
                      **{f"ids_{i}": np.asarray(s["df"].index.astype(str))[np.asarray(tk, dtype=np.int64)].astype(str)
                         for i, (tk, s) in enumerate(zip(r["take"], samples))}}
            if r["labels_before"] is not None:
                arrays["labels_before"] = r["labels_before"]
            if r["umap"] is not None:
                arrays["umap"] = r["umap"]
            try:
                _jc_cache_write(tag, arrays, {
                    "genes_n": n_genes, "info": r["info"], "timings": r["timings"], "params": p_run,
                    "levels": levels, "name": name, "umap_missing": bool(p["umap"] and r["umap"] is None),
                    "sample_keys": [str(s["key"]) for s in samples],
                    "mixing_after": r["diag"]["after"]["mixing"],
                    "mixing_before": (r["diag"].get("before") or {}).get("mixing")})
            except Exception as exc:
                print(f"  Joint clustering: could not write the cache ({exc})", flush=True)
        # labels and UMAP coordinates per sample table
        per_labels, umap_cols = [], []
        for si, (t, s) in enumerate(zip(tables, samples)):
            lab = np.full(len(s["df"]), None, dtype=object)
            sel = code == si
            lab[r["take"][si]] = r["labels"][sel].astype(int).tolist()
            per_labels.append(lab)
            u1 = np.full(len(s["df"]), np.nan, dtype=np.float32)
            u2 = u1.copy()
            if r["umap"] is not None:
                u1[r["take"][si]], u2[r["take"][si]] = r["umap"][sel, 0], r["umap"][sel, 1]
            umap_cols.append({f"{col_prefix}_1": u1, f"{col_prefix}_2": u2})
        method_key = clustersets._cs_install(name, per_labels, seg_source, meta={
            "kind": "joint", "variant": variant, "params": p_run, "levels": levels, "cache_tag": tag,
            "backend": r["info"].get("backend")})
        # only now, with the set installed (the install validates the view against the
        # tables of this run): a failed install leaves the embedding of the last run in place
        for t, cols in zip(tables, umap_cols):
            controls._ctl_put(t, seg_source, cols, tag=seg_tag)
        dataview._umap_fig_cache.clear()
        table = _jc_diag_table(tables, levels, code, r["labels"], r["labels_before"], r["batch"])
        n = int(len(code))
        n_all = int(sum(len(s["df"]) for s in samples))
        conf = _jc_confounded(conds, levels) if var != "none" else None
        seconds = _time.time() - _t0
        res = {"gen": _gen, "name": name, "variant": variant, "method_key": method_key, "params": p_run,
               "levels": levels, "var": var, "diag": r["diag"], "table": table, "pcs": r["pcs"], "emb": r["emb"],
               "labels": r["labels"], "sample": code, "take": r["take"], "umap": r["umap"],
               "samples": [{"label": t["label"], "name": t["name"], "level": lv, "cond": c}
                           for t, lv, c in zip(tables, levels, conds)],
               "n_cells": n, "n_cells_view": n_all, "n_genes": n_genes, "n_hvg": int(len(r["hvg"])),
               "n_pcs": int(r["pcs"].shape[1]), "backend": r["info"], "harmony": harmony,
               "confounded": conf, "from_cache": from_cache, "timings": r["timings"], "seconds": seconds,
               "corrected": bool(samples[0]["corrected"]), "seg_source": seg_source or "xenium", "cache_tag": tag}
        k = int(r["labels"].max())
        msg = (f"'{name}': {k} clusters of {n:,} cells in {S} samples "
               f"({n_genes:,} shared genes, {len(r['hvg']):,} used, {r['pcs'].shape[1]} PCs, "
               f"{'SPLIT-corrected' if res['corrected'] else 'original'} counts; {r['info'].get('backend')}, "
               f"{p['n_neighbors']} neighbours, resolution {p['resolution']:g}, seed {p['seed']}; "
               + ("from the cache" if from_cache else f"{seconds:.0f} s") + ").")
        if n < n_all:
            msg += f" {n_all - n:,} cell(s) without counts have no cluster."
        if harmony:
            msg += f"\nHarmony (harmonypy, theta {p['theta']:g}) on the {var}: {len(set(levels))} levels." + note_missing
        elif var != "none":
            msg += "\n⚠ " + _JC_HARMONY_HINT + "."
        else:
            msg += "\nNo integration: clusters of the joint PCA."
        if r["info"].get("note"):
            msg += f"\n{r['info']['note']}"
        msg += "\n" + _jc_diag_text(r["diag"], var if var != "none" else "sample")
        if harmony and r["diag"]["after"]["entropy"]["mean"] < 0.5 * r["diag"]["after"]["entropy"]["reference"]:
            msg += (f"\n⚠ The clusters are still dominated by single levels of the {var} after Harmony: the "
                    f"difference between the levels is larger than Harmony's defaults remove (or the levels "
                    f"hold different cell populations).")
        if p["umap"] and r["umap"] is None:
            msg += f"\n{_JC_UMAP_HINT}."
        elif r["umap"] is not None:
            msg += ("\nSelect the set under Cluster Set (Color By → Cluster): the UMAP panel then shows the joint "
                    "embedding of all samples.")
        if harmony:
            msg += "\n" + _JC_CONFOUND_WARNING
            if conf == "full":
                msg += (f"\n⚠ In this design every level of the {var} belongs to one condition: the condition "
                        f"is fully confounded with the integration variable.")
            elif conf == "partial":
                msg += (f"\nIn this design at least one level of the {var} holds samples of both conditions "
                        f"(not fully confounded).")
        _set("done", msg, result=res)
        print(f"  Joint clustering: {msg}", flush=True)
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


# ─── The joint UMAP in the UMAP panel ─────────────────────────────────────────
_jc_restore_seen: dict = {}     # (dataset generation, method, cached run, sample keys) → True once a restore was tried


def _jc_variant_of(method) -> "str | None":
    """The joint variant behind a Cluster Set value (clustersets keeps the description the
    set was installed with, also for a set restored from the cache), else None."""
    if method is None:
        return None
    meta = clustersets._cs_meta(method)
    v = meta.get("variant") if meta.get("kind") == "joint" else None
    return v if v in _JC_VARIANTS else None


def _jc_restore_umap(method, tables, seg_source="xenium") -> None:
    """After a restart the cluster set is restored from the cache but its UMAP is not in
    memory: read the coordinates of the run the set came from. Tried once per dataset,
    set, cached run and list of loaded samples: a restore refused because a sample of the
    run is not loaded yet is tried again when the samples change."""
    meta = clustersets._cs_meta(method)
    key = (jobs._dataset_gen, str(method), str(meta.get("cache_tag") or ""),
           tuple(str(t["key"]) for t in tables))
    if _jc_restore_seen.get(key):
        return
    _jc_restore_seen.clear()
    _jc_restore_seen[key] = True
    variant = _jc_variant_of(method)
    if variant is None or not meta.get("cache_tag"):
        return
    got = _jc_cache_read(str(meta["cache_tag"]))
    if got is None or "umap" not in got[0]:
        return
    arr = got[0]
    prefix = _JC_VARIANTS[variant][1]
    try:
        # The coordinates go back to the samples and cells they were computed for, or not
        # at all: the run names its samples, and the cells are matched by id (row positions
        # differ once a subset is set or cleared, sample positions once the list changes).
        keys = [str(k) for k in (got[1].get("sample_keys") or [])]
        if keys != [str(t["key"]) for t in tables] or any(f"ids_{i}" not in arr for i in range(len(tables))):
            return
        cols = []
        for si, t in enumerate(tables):
            sel = arr["sample"] == si
            ids = pd.Index(arr[f"ids_{si}"].astype(str))
            if len(ids) != int(sel.sum()) or not ids.is_unique:
                return
            own = pd.Index(t["df"].index.astype(str))
            full = (t.get("ds") or {}).get("_df_original")       # the whole sample when a subset is active
            known = own if full is None else pd.Index(full.index.astype(str))
            if not len(ids) or not ids.isin(known).all():
                return                              # not the cells of this sample
            um = pd.DataFrame(arr["umap"][sel, :2], index=ids).reindex(own)
            cols.append({f"{prefix}_1": um[0].to_numpy(dtype=np.float32), f"{prefix}_2": um[1].to_numpy(dtype=np.float32)})
        for t, c in zip(tables, cols):
            controls._ctl_put(t, seg_source, c)
    except Exception as exc:
        print(f"  Joint clustering: the cached UMAP was not restored ({exc})", flush=True)


def _jc_umap_view(method, seg_source="xenium") -> "dict | None":
    """The joint embedding behind a Cluster Set value, for the samples now in view:
    {variant, tables, x, y (per table, NaN for a cell without coordinates)}; None when
    *method* is not a joint set, the view is a resegmentation run or has one sample, or no
    cell of the view has coordinates (no UMAP was computed)."""
    variant = _jc_variant_of(method)
    if variant is None or analysis._an_alt_res(seg_source) is not None:
        return None
    tables = stats._view_tables(seg_source)
    if len(tables) < 2:
        return None
    prefix = _JC_VARIANTS[variant][1]
    if all(controls._ctl_get(t, seg_source, f"{prefix}_1") is None for t in tables):
        _jc_restore_umap(method, tables, seg_source)
    xs, ys, any_xy = [], [], False
    for t in tables:
        u1, u2 = controls._ctl_get(t, seg_source, f"{prefix}_1"), controls._ctl_get(t, seg_source, f"{prefix}_2")
        if u1 is None or u2 is None:
            u1 = u2 = np.full(len(t["df"]), np.nan)
        x = pd.to_numeric(pd.Series(u1), errors="coerce").to_numpy(dtype=np.float64)
        y = pd.to_numeric(pd.Series(u2), errors="coerce").to_numpy(dtype=np.float64)
        any_xy = any_xy or bool((np.isfinite(x) & np.isfinite(y)).any())
        xs.append(x)
        ys.append(y)
    if not any_xy:
        return None
    return {"variant": variant, "tables": tables, "x": xs, "y": ys}


def _jc_umap_fig(color_by, method, gene, size, opacity, baysor_active=False, proseg_active=False,
                 use_corrected: bool = False, cbar_range=None):
    """UMAP figure provider (figures._umap_fig_providers): when the selected cluster set is
    a joint one that has a joint UMAP, the cells of ALL samples in that embedding, coloured
    by cluster, gene, cell type or a gene-set score (anything else: by sample). None
    otherwise - the UMAP panel then draws what it always did."""
    if baysor_active or proseg_active:
        return None
    seg = "xenium"
    view = _jc_umap_view(method, seg)
    if view is None:
        return None
    variant, tables = view["variant"], view["tables"]
    x, y = np.concatenate(view["x"]), np.concatenate(view["y"])
    sidx = np.concatenate([np.full(len(t["df"]), si) for si, t in enumerate(tables)])
    ids = np.concatenate([np.array([f"{t['label']}:{c}" for c in t["df"].index.astype(str)], dtype=object)
                          for t in tables])
    ok = np.isfinite(x) & np.isfinite(y)
    mk = max(2, size * 1.2)
    color_by = str(color_by or "cluster")

    def _categorical(lab, what, colors=None):
        lab = np.asarray(lab, dtype=object)
        keep = ok & np.array([v is not None and v == v for v in lab], dtype=bool)
        vals = np.array([str(v) for v in lab[keep]], dtype=object)
        def _nat(v):
            d = v[1:] if v[:1] == "C" else v
            return (0, int(d), "") if d.isdigit() else (1, 0, v)
        uniq = sorted(set(vals.tolist()), key=_nat)
        cmap = colors or {u: core.CLUSTER_COLORS[i % len(core.CLUSTER_COLORS)] for i, u in enumerate(uniq)}
        out = []
        for u in uniq:
            m = np.flatnonzero(keep)[vals == u]
            out.append(go.Scattergl(x=x[m], y=y[m], mode="markers",
                                    marker=dict(size=mk, color=cmap.get(u, "#888888"), opacity=opacity),
                                    name=u, showlegend=False, text=ids[m], customdata=ids[m],
                                    hovertemplate="<b>%{text}</b><extra>" + what + " " + u + "</extra>"))
        return out

    def _continuous(vals, label, cs="Viridis", rng=None):
        vals = np.asarray(vals, dtype=np.float64)
        ckw = {}
        lo, hi = (rng or (None, None))
        if cbar_range:
            lo = cbar_range[0] if cbar_range[0] is not None else lo
            hi = cbar_range[1] if cbar_range[1] is not None else hi
        if lo is not None:
            ckw["cmin"] = lo
        if hi is not None:
            ckw["cmax"] = hi
        m = np.flatnonzero(ok)
        return [go.Scattergl(x=x[m], y=y[m], mode="markers",
                             marker=dict(size=mk, color=vals[m], colorscale=cs, opacity=opacity,
                                         showscale=False, **ckw),
                             text=ids[m], customdata=ids[m], name=label,
                             hovertemplate="<b>%{text}</b><br>" + label + ": %{marker.color:.2f}<extra></extra>")]

    traces, what = None, ""
    try:
        if color_by == "cluster":
            labs = stats._raw_cluster_labels(tables, method, None)
            if all(l is not None for l in labs):
                traces, what = _categorical(np.concatenate(labs), "Cluster"), "cluster"
        elif color_by == "gene" and gene:
            # corrected counts only when every sample has them: one colour scale never
            # mixes corrected and original values (as the map and the WebGL colours)
            mc = dataview._multi_use_corrected(use_corrected)
            parts = [dataview._xenium_gene_values(gene, mc)]
            parts += [dataview._extra_gene_values(eds, gene, mc) for eds in dataview.EXTRA_DATASETS]
            if all(p is not None and len(p) == len(t["df"]) for p, t in zip(parts, tables)):
                traces, what = _continuous(np.concatenate(parts), str(gene)), str(gene)
        elif color_by.startswith("cell_type:"):
            samples = analysis._an_samples(seg, use_corrected)
            if len(samples) == len(tables):
                am = analysis._an_annot_method(color_by, samples, None)
                labs = analysis._an_celltype_labels(samples, am, None)
                if any(l is not None for l in labs):
                    lab = np.concatenate([np.full(len(t["df"]), None, dtype=object) if l is None
                                          else np.asarray(l, dtype=object) for l, t in zip(labs, tables)])
                    traces, what = _categorical(lab, "Type"), core._ANNOT_METHODS.get(am, str(am))
        elif color_by.startswith("geneset:"):
            gs = analysis._geneset_color(color_by, seg, use_corrected)
            if gs is not None and len(gs[0]) == len(x):
                lim = gs[1]
                rng = (lim[0], lim[1]) if isinstance(lim, (tuple, list)) else (-float(lim), float(lim))
                traces, what = _continuous(gs[0], str(gs[2]), "RdBu_r" if not isinstance(lim, (tuple, list)) else "Viridis",
                                           rng), str(gs[2])
    except Exception as exc:
        print(f"  Joint UMAP: colouring by {color_by} failed ({exc}); colouring by sample", flush=True)
        traces = None
    if traces is None:
        lab = np.array([tables[i]["label"] for i in sidx], dtype=object)
        traces, what = _categorical(lab, "Sample"), "sample"
    fig = go.Figure(data=traces)
    fig.update_layout(**figures._base_layout(f"Joint UMAP · {_JC_VARIANTS[variant][0]} · {what} "
                                             f"({int(ok.sum()):,} cells, {len(tables)} samples)",
                                             "UMAP 1", "UMAP 2"),
                      showlegend=False, uirevision="umap-joint")
    return fig


figures._umap_fig_providers.append(_jc_umap_fig)


# ─── Figure, table, CSV ───────────────────────────────────────────────────────
def make_jc_fig(width=None, height=None):
    """Left: cells of every joint cluster by sample (stacked shares). Right: entropy of the
    integration variable per cluster, before (clusters of the uncorrected PCs, grey) and
    after, each sorted, with the entropy of the whole view as the reference line."""
    from plotly.subplots import make_subplots
    res = _jc_result()
    if not res:
        return figures._make_placeholder_fig("No joint clustering computed yet.")
    tb = res["table"]
    labels = [s["label"] for s in res["samples"]]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], horizontal_spacing=0.1,
                        subplot_titles=("Share of each cluster's cells by sample",
                                        f"Entropy of the {res['var'] if res['var'] != 'none' else 'sample'} per cluster (sorted)"))
    tot = tb["n_cells"].clip(lower=1).to_numpy(dtype=float)
    for i, lb in enumerate(labels):
        fig.add_trace(go.Bar(x=tb["cluster"].astype(str), y=tb[f"n_{lb}"].to_numpy(dtype=float) / tot, name=lb,
                             marker_color=core.CLUSTER_COLORS[i % len(core.CLUSTER_COLORS)],
                             customdata=tb[f"n_{lb}"],
                             hovertemplate="cluster %{x}<br>" + lb + ": %{customdata:,} cells (%{y:.1%})<extra></extra>"),
                      row=1, col=1)
    d = res["diag"]
    if "before" in d:
        e = np.sort(d["before"]["entropy"]["entropy"])
        fig.add_trace(go.Scatter(x=np.arange(1, len(e) + 1), y=e, mode="lines+markers", name="before",
                                 line=dict(color="#8b949e"), marker=dict(size=5),
                                 hovertemplate="before: %{y:.3f}<extra></extra>"), row=1, col=2)
    e = np.sort(d["after"]["entropy"]["entropy"])
    fig.add_trace(go.Scatter(x=np.arange(1, len(e) + 1), y=e, mode="lines+markers",
                             name="after" if "before" in d else "clusters",
                             line=dict(color="#58a6ff"), marker=dict(size=5),
                             hovertemplate="%{y:.3f}<extra></extra>"), row=1, col=2)
    fig.add_hline(y=d["after"]["entropy"]["reference"], line=dict(color="#d29922", dash="dot", width=1), row=1, col=2)
    fig.update_layout(barmode="stack", showlegend=True,
                      legend=dict(orientation="h", y=-0.18, font=dict(size=9)))
    fig.update_annotations(font=dict(size=10, color=core.TEXT))
    fig.update_yaxes(range=[0, 1.02], row=1, col=2)
    return controls._fig_layout(fig, f"{res['name']}: {len(tb)} clusters, {res['n_cells']:,} cells, "
                                     f"{len(labels)} samples", width, height, margin=dict(l=50, r=10, t=60, b=70))


_JC_COLS = [("cluster", "Cluster"), ("n_cells", "Cells"), ("entropy", "Entropy"),
            ("largest_sample", "Largest sample"), ("largest_share", "Its share")]


def _jc_columns(res=None) -> list:
    cols = [{"id": c, "name": n, "type": "text" if c == "largest_sample" else "numeric"} for c, n in _JC_COLS]
    if res:
        cols += [{"id": f"n_{s['label']}", "name": s["label"], "type": "numeric"} for s in res["samples"]]
    return cols


def _jc_records(res) -> list:
    if not res:
        return []
    out = res["table"].copy()
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda v: None if not np.isfinite(v) else float(f"{v:.4g}"))
    return out.astype(object).where(out.notna(), None).to_dict("records")


def _jc_csv(res) -> str:
    p, d = res["params"], res["diag"]
    _left = int(res.get("n_cells_view", res["n_cells"])) - int(res["n_cells"])
    lines = [f"# {res['name']}; {res['n_cells']} cells of {len(res['samples'])} samples; "
             f"{'SPLIT-corrected' if res['corrected'] else 'original'} counts"
             + (f"; {_left} cell(s) without {'SPLIT-corrected ' if res['corrected'] else ''}counts were left out "
                f"and have no cluster" if _left > 0 else ""),
             f"# {res['n_genes']} shared genes, {res['n_hvg']} used, {res['n_pcs']} PCs; clustering: "
             f"{res['backend'].get('backend')}, {p['n_neighbors']} neighbours, resolution {p['resolution']:g}, "
             f"seed {p['seed']}",
             (f"# Harmony (harmonypy) on the {res['var']}, theta {p['theta']:g}" if res["harmony"] else
              "# no batch correction" + ("" if res["var"] == "none" else " (harmonypy is not installed)")),
             "# samples: " + "; ".join(f"{s['label']} {s['name']} = {s['level']}"
                                        + (f" [{s['cond']}]" if s["cond"] else "") for s in res["samples"]),
             "# entropy = entropy of the integration variable among the cells of the cluster / log(levels)",
             "# " + _jc_diag_text(d, res["var"] if res["var"] != "none" else "sample")]
    if res["harmony"]:
        lines.append("# " + _JC_CONFOUND_WARNING.replace("⚠ ", ""))
    return provenance._prov_csv_stamp(res, ("\n".join(lines) + "\n" + res["table"].to_csv(index=False)))


# ─── Layout ───────────────────────────────────────────────────────────────────
def _jc_integrate_options() -> list:
    have = _jc_harmony_module() is not None
    sfx = "" if have else " - needs harmonypy"
    return [{"label": "no integration", "value": "none"},
            {"label": "Harmony on sample" + sfx, "value": "sample"},
            {"label": "Harmony on donor (design)" + sfx, "value": "donor"},
            {"label": "Harmony on batch (design)" + sfx, "value": "batch"}]


def _jc_tab():
    S, L = analysis._AN_ROW_STYLE, analysis._AN_LABEL_STYLE
    num = controls._num
    return dbc.Tab(label="Joint clustering", tab_id="tab-joint",
                   label_style={"fontSize": "12px", "padding": "4px 12px"},
                   children=[html.Div([
        html.Div([
            html.Span("Integration:", style=L),
            dcc.Dropdown(id="jc-integrate", options=_jc_integrate_options(), value="sample", clearable=False,
                         style={"width": "250px", "fontSize": "11px"}),
            html.Span("Resolution:", style=L), num(id="jc-res", value=_JC_RESOLUTION, min=0.01, max=10, step=0.01),
            html.Span(style={"flex": "1"}),
            dbc.Button("Run", id="jc-run-btn", color="primary", size="sm", n_clicks=0,
                       style={"fontSize": "11px", "padding": "2px 12px"}),
            dbc.Button("Samples & conditions…", id="jc-design-open-btn", color="secondary", size="sm", n_clicks=0,
                       style={"fontSize": "11px", "padding": "2px 8px"}),
        ], style=S),
        ui._options(html.Div([
            html.Span("Genes:", style=L), num(id="jc-nhvg", value=_JC_N_HVG, min=50, max=20000, step=50, width="70px"),
            html.Span("PCs:", style=L), num(id="jc-npcs", value=_JC_N_PCS, min=2, max=100, step=1),
            html.Span("Neighbours:", style=L), num(id="jc-nn", value=_JC_N_NEIGHBORS, min=2, max=200, step=1),
            html.Span("Seed:", style=L), num(id="jc-seed", value=_JC_SEED, min=0, max=10**6, step=1),
            dcc.Checklist(id="jc-umap",
                          options=[{"label": html.Span(" Joint UMAP", style={"fontSize": "11px", "color": core.TEXT}),
                                    "value": "on"}],
                          value=["on"], inputStyle={"marginRight": "3px"}, style={"display": "inline-flex"}),
        ], style=S), summary=f"Options ({_JC_N_HVG:,} genes, {_JC_N_PCS} PCs, {_JC_N_NEIGHBORS} neighbours, "
                             f"seed {_JC_SEED}, joint UMAP on)"),
        html.Div(id="jc-design-summary", style={**L, "padding": "0 8px"}),
        html.Div(id="jc-status", style={**analysis._AN_STATUS_STYLE, "whiteSpace": "pre-wrap"}),
        html.Div(_JC_NOTE, id="jc-note", style=analysis._AN_NOTE_STYLE),
        html.Div([
            controls._graph(id="jc-plot", msg="No joint clustering computed yet.", filename="xenium_joint_clusters",
                            min_h="320px"),
            html.Div([
                dash_table.DataTable(id="jc-table", data=[], columns=_jc_columns(), **stats._dark_table_style(12)),
                html.Div([
                    dbc.Button("Download CSV", id="jc-download-btn", color="secondary", size="sm", n_clicks=0,
                               style={"fontSize": "11px", "padding": "2px 8px"}),
                    dcc.Download(id="jc-download"),
                ], style={"padding": "4px 0"}),
            ], style={"flex": "0.8", "minWidth": "0", "overflow": "auto", "padding": "0 6px"}),
        ], style={"display": "flex", "flex": "1", "minHeight": "400px", "gap": "6px"}),
        ui._plot_dim_controls("jc"),
    ], style={"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0",
              "overflowY": "auto"})])
