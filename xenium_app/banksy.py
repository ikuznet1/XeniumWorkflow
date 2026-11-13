"""Banksy-style spatial domains (compute + layout): clustering of the cells on their own
expression augmented with the mean expression of their spatial neighbours and its azimuthal
Gabor filter (Singhal et al., Nat Genet 2024), implemented with numpy / scipy / sklearn.

Every spatial neighbour of a cell is a cell of the SAME sample: with several samples side
by side the neighbour graph is block-diagonal, while scaling, PCA and clustering are done
on the concatenated cells, so one set of labels holds for all samples.
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

from . import core, jobs, dataview, ui, figures, analysis, stats, controls, clustersets
from . import provenance


# ─── State ────────────────────────────────────────────────────────────────────
_bk_state: dict = {"status": "idle", "message": "", "result": None, "version": 0}
_bk_lock = threading.Lock()

_BK_LAMBDA_DOMAINS = 0.8     # the published setting for tissue domains
_BK_LAMBDA_CELLTYPE = 0.2    # ... and for spatially informed cell typing
_BK_K_GEOM = 15              # spatial neighbours of the mean (the AGF uses twice as many)
_BK_MU = 1.5                 # normalisation of the two neighbourhood blocks (eq. 2 of the paper)
_BK_N_PCS = 20
_BK_K_EXPR = 50              # neighbours of the graph that is clustered
_BK_RESOLUTION = 0.5          # Leiden resolution of the domain preset …
_BK_RESOLUTION_CELLTYPE = 1.0 # … and of the cell-typing preset
_BK_MAX_GENES = 2000         # feature cap: the most variable genes of a larger panel
_BK_MIN_CELLS = 50
_BK_CHUNK = 8192             # cells per dense block
_BK_KEEP_BYTES = 1.2e9       # neighbourhood blocks up to this size are kept instead of recomputed
_BK_CACHE_VERSION = 3          # 2: the signature includes the content of the count matrix
                               # 3: ... and whether Leiden could run (else the KMeans fallback)
_BK_TOP_GENES = 5
# cost of the part before the graph, per million cells (measured, README "Performance at
# study scale"): the normalised matrix and its gene selection; the three passes over the
# neighbourhood features and the Gram matrix of the augmented features
_BK_MATRIX_GB_PER_M_CELLS = 6.0
_BK_MIN_PER_M_CELLS = 12.0

# the one sentence on what this implementation is, shown in the tab, the status line, the
# CSV header and the README
_BK_UNVALIDATED = ("This is a native implementation, not the Banksy package: it follows the published "
                   "method's construction (the three feature blocks and their λ scaling) but has not "
                   "been validated against the reference implementation.")

_BK_NOTE = (
    "Banksy-style spatial domains. Banksy (Singhal et al., Nat Genet 2024) clusters every cell on three blocks of features: its "
    "own expression, the weighted mean expression of its k nearest cells in space, and the azimuthal "
    "Gabor filter (AGF) of its 2k nearest cells - the magnitude of the expression gradient around it, "
    "which does not change when the section is rotated. Neighbour weights are Gaussian in the distance, "
    "with the distance to the median neighbour as the width, so the neighbourhood adapts to the local "
    "cell density. Each block is z-scaled per gene and the blocks are mixed by λ: √(1−λ) own, √(λ/1.5) "
    "mean, √(λ/3) AGF (√(1−λ) and √λ without the AGF). λ = 0.8 gives tissue domains, λ = 0.2 cell "
    "types informed by their surroundings, λ = 0 ordinary clustering. Then PCA, a nearest-neighbour "
    "graph of the cells in PC space and Leiden clustering. Counts are scaled to the median total per "
    "cell; with a panel larger than the gene cap the most variable genes are used. Spatial neighbours "
    "are always cells of the same sample; with several samples the features are scaled and clustered "
    "together, so a domain has the same number in every sample. " + _BK_UNVALIDATED + " Clusters depend on λ, k, the resolution and the seed, "
    "and are an exploratory partition, not a test. The result is a cluster set: Color By → Cluster → "
    "Clustering Method. Recompute after changing the subset, the segmentation or the counts.")


def _bk_result():
    with _bk_lock:
        res = _bk_state.get("result")
    if not res or res.get("gen") != jobs._dataset_gen:
        return None
    return res


# ─── Features ─────────────────────────────────────────────────────────────────
def _bk_neighbours(xy, k_geom: int = _BK_K_GEOM) -> dict:
    """Spatial neighbours and weights of the cells of ONE sample.

    For the mean: the k nearest other cells, weight exp(−r² / σ²) with σ the distance to
    the ⌊k / 2⌋-th nearest (the 7th for k = 15), normalised to sum 1 per cell. For the
    AGF: the 2k nearest, σ the distance to their ⌊2k / 2⌋-th, and the angle φ of each
    neighbour (atan2 of the offset). Returns idx [n, 2k'], gm [n, k'] (weights of the
    first k' columns), ga [n, 2k'], phi [n, 2k'] (k' ≤ k when the sample has few cells)."""
    from scipy.spatial import cKDTree
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    k2 = int(min(2 * int(k_geom), n - 1))
    k1 = int(min(int(k_geom), k2))
    if k2 < 1:
        z = np.zeros((n, 0))
        return {"idx": np.zeros((n, 0), dtype=np.int64), "gm": z, "ga": z, "phi": z, "k": 0, "k2": 0}
    d, idx = cKDTree(xy).query(xy, k=k2 + 1, workers=-1)
    # the cell itself is dropped (it is the first hit unless another cell sits on the same spot)
    me = idx == np.arange(n)[:, None]
    me[~me.any(axis=1), 0] = True
    order = np.argsort(me, axis=1, kind="stable")[:, :k2]
    d, idx = np.take_along_axis(d, order, axis=1), np.take_along_axis(idx, order, axis=1)

    def _gauss(dd):
        kk = dd.shape[1]
        sig = dd[:, max(kk // 2, 1) - 1].copy()
        sig[sig <= 0] = np.where(dd.max(axis=1) > 0, dd.max(axis=1), 1.0)[sig <= 0]
        w = np.exp(-(dd / sig[:, None]) ** 2)
        return w / w.sum(axis=1, keepdims=True)

    dxy = xy[idx] - xy[:, None, :]
    return {"idx": idx, "gm": _gauss(d[:, :k1]), "ga": _gauss(d), "phi": np.arctan2(dxy[..., 1], dxy[..., 0]),
            "k": k1, "k2": k2}


def _bk_weight_matrices(xys, k_geom: int = _BK_K_GEOM) -> dict:
    """Sparse neighbour operators over the concatenated cells of several samples (each
    sample's neighbours are found among its own cells only): m (weights of the mean), a
    (weights over the AGF neighbourhood), re / im (the same times cos φ / sin φ), and s
    = Σ weight × e^{iφ} per cell (complex)."""
    rows_m, cols_m, w_m, rows_a, cols_a, w_a, cs, sn, s_all = [], [], [], [], [], [], [], [], []
    off = 0
    for xy in xys:
        jobs._job_check()
        nb = _bk_neighbours(xy, k_geom)
        n = len(xy)
        k1, k2 = nb["k"], nb["k2"]
        r = np.arange(n) + off
        rows_m.append(np.repeat(r, k1))
        cols_m.append((nb["idx"][:, :k1] + off).ravel())
        w_m.append(nb["gm"].ravel())
        rows_a.append(np.repeat(r, k2))
        cols_a.append((nb["idx"] + off).ravel())
        w_a.append(nb["ga"].ravel())
        c, s = np.cos(nb["phi"]), np.sin(nb["phi"])
        cs.append((nb["ga"] * c).ravel())
        sn.append((nb["ga"] * s).ravel())
        s_all.append((nb["ga"] * (c + 1j * s)).sum(axis=1) if k2 else np.zeros(n, dtype=complex))
        off += n
    N = off
    cat = lambda parts, dt: np.concatenate(parts).astype(dt) if parts else np.zeros(0, dtype=dt)
    rm, cm, ra, ca = cat(rows_m, np.int64), cat(cols_m, np.int64), cat(rows_a, np.int64), cat(cols_a, np.int64)
    mk = lambda v, r, c: sp.csr_matrix((cat(v, np.float32), (r, c)), shape=(N, N))
    return {"m": mk(w_m, rm, cm), "a": mk(w_a, ra, ca), "re": mk(cs, ra, ca), "im": mk(sn, ra, ca),
            "s": cat(s_all, np.complex128), "n": N}


def _bk_blocks(Cn, W, r0: int, r1: int, agf: bool = True, center: bool = True) -> tuple:
    """Dense float32 blocks of the cells r0 … r1: (own expression C, neighbourhood mean M,
    AGF G or None).

    M = Σ_v Γ_uv g_v over the k neighbours. G = | Σ_v Γ'_uv e^{iφ_uv} (g_v − m'_u) | over
    the 2k neighbours with their own weights Γ'; m'_u is the Γ'-weighted mean of the same
    neighbours (center=True: the published implementations centre the neighbours'
    expression, so that an uneven arrangement of neighbours around a cell is not read as
    a gradient; center=False is equation 1 of the paper as printed)."""
    C = Cn[r0:r1].toarray().astype(np.float32, copy=False)
    M = (W["m"][r0:r1] @ Cn).toarray().astype(np.float32, copy=False)
    if not agf:
        return C, M, None
    re = (W["re"][r0:r1] @ Cn).toarray().astype(np.float32, copy=False)
    im = (W["im"][r0:r1] @ Cn).toarray().astype(np.float32, copy=False)
    if center:
        mu = (W["a"][r0:r1] @ Cn).toarray().astype(np.float32, copy=False)
        s = W["s"][r0:r1]
        re -= s.real.astype(np.float32)[:, None] * mu
        im -= s.imag.astype(np.float32)[:, None] * mu
    return C, M, np.hypot(re, im)


def _bk_block_weights(lam: float, agf: bool = True, mu: float = _BK_MU) -> tuple:
    """Multipliers of the z-scaled blocks (own, mean, AGF): √(1−λ), √(λ/μ), √(λ/(2μ)) with
    μ = 1.5 (equation 2 of the paper); without the AGF √(1−λ), √λ."""
    lam = float(min(1.0, max(0.0, lam)))
    if agf:
        return np.sqrt(1.0 - lam), np.sqrt(lam / mu), np.sqrt(lam / (2.0 * mu))
    return np.sqrt(1.0 - lam), np.sqrt(lam), 0.0


def _bk_embed(Cn, W, lam: float = _BK_LAMBDA_DOMAINS, agf: bool = True, n_pcs: int = _BK_N_PCS,
              center: bool = True, progress=None, chunk: int = _BK_CHUNK) -> dict:
    """Principal components of the neighbour-augmented matrix, without holding it in memory.

    Cn: cells × genes (sparse, normalised). The features [own | mean | AGF] are z-scaled
    per column over all cells and multiplied by the block weights; their Gram matrix is
    accumulated over blocks of cells, its leading eigenvectors are the PC loadings, and
    the cells are projected block by block. Returns pcs [cells, n_pcs] float32, variance
    (eigenvalues ÷ (cells − 1)), loadings, the column means / standard deviations, and the
    share of the total variance the PCs hold."""
    from scipy.sparse.linalg import eigsh
    Cn = sp.csr_matrix(Cn, dtype=np.float32)
    n, p = Cn.shape
    prog = progress or (lambda f, m=None: None)
    nb = 3 if agf else 2
    wts = _bk_block_weights(lam, agf)
    keep = (nb - 1) * n * p * 4 <= _BK_KEEP_BYTES
    kept = {}

    def blocks(r0, r1):
        if r0 in kept:
            M, G = kept[r0]
            return Cn[r0:r1].toarray().astype(np.float32, copy=False), M, G
        C, M, G = _bk_blocks(Cn, W, r0, r1, agf, center)
        if keep:
            kept[r0] = (M, G)
        return C, M, G

    starts = list(range(0, n, int(chunk)))
    # pass 1: moments of every feature
    ssum, ssq = np.zeros(nb * p), np.zeros(nb * p)
    for i, r0 in enumerate(starts):
        jobs._job_check()
        for b, X in enumerate(blocks(r0, min(n, r0 + chunk))[:nb]):
            X = X.astype(np.float64)
            ssum[b * p:(b + 1) * p] += X.sum(axis=0)
            ssq[b * p:(b + 1) * p] += (X * X).sum(axis=0)
        prog(0.3 * (i + 1) / len(starts), "Banksy: neighbourhood mean and AGF…")
    mean = ssum / max(n, 1)
    sd = np.sqrt(np.maximum(ssq / max(n, 1) - mean ** 2, 0.0) * (n / max(n - 1, 1)))
    ok = sd > 1e-8 * np.maximum(1.0, np.abs(mean))
    scale = np.where(ok, 1.0 / np.where(ok, sd, 1.0), 0.0) * np.repeat(wts[:nb], p)

    def zblock(r0, r1):
        Z = np.concatenate(blocks(r0, r1)[:nb], axis=1)
        Z -= mean.astype(np.float32)[None, :]
        Z *= scale.astype(np.float32)[None, :]
        return Z

    # pass 2: Gram matrix of the scaled features
    Gm = np.zeros((nb * p, nb * p))
    for i, r0 in enumerate(starts):
        jobs._job_check()
        Z = zblock(r0, min(n, r0 + chunk))
        Gm += (Z.T @ Z).astype(np.float64)
        prog(0.3 + 0.4 * (i + 1) / len(starts), "Banksy: covariance of the augmented features…")
    k = int(max(1, min(n_pcs, nb * p - 1, n - 1)))
    jobs._job_check()
    if nb * p <= max(64, k + 2):
        vals, vecs = np.linalg.eigh(Gm)
        vals, vecs = vals[::-1][:k], vecs[:, ::-1][:, :k]
    else:
        vals, vecs = eigsh(Gm, k=k, which="LA", v0=np.ones(nb * p) / np.sqrt(nb * p), tol=1e-7)
        order = np.argsort(-vals)
        vals, vecs = vals[order], vecs[:, order]
    # sign: the loading of largest magnitude is positive
    sgn = np.sign(vecs[np.abs(vecs).argmax(axis=0), np.arange(k)])
    vecs = vecs * np.where(sgn == 0, 1.0, sgn)[None, :]
    total = float(np.trace(Gm))
    # pass 3: project
    pcs = np.empty((n, k), dtype=np.float32)
    V32 = vecs.astype(np.float32)
    for i, r0 in enumerate(starts):
        jobs._job_check()
        r1 = min(n, r0 + chunk)
        pcs[r0:r1] = zblock(r0, r1) @ V32
        prog(0.7 + 0.3 * (i + 1) / len(starts), "Banksy: principal components…")
    return {"pcs": pcs, "variance": np.maximum(vals, 0.0) / max(n - 1, 1), "loadings": vecs,
            "mean": mean, "scale": scale, "explained": float(np.maximum(vals, 0).sum() / total) if total > 0 else 0.0,
            "n_features": int(ok.sum()), "kept": keep}


def _bk_augmented(Cn, W, lam: float, agf: bool = True, center: bool = True) -> np.ndarray:
    """The neighbour-augmented matrix itself (cells × 2p or 3p, z-scaled and weighted):
    for tests and small data."""
    Cn = sp.csr_matrix(Cn, dtype=np.float32)
    C, M, G = _bk_blocks(Cn, W, 0, Cn.shape[0], agf, center)
    out = []
    for X, w in zip((C, M, G)[:3 if agf else 2], _bk_block_weights(lam, agf)):
        X = X.astype(np.float64)
        sd = X.std(axis=0, ddof=1) if len(X) > 1 else np.zeros(X.shape[1])
        mu = X.mean(axis=0)
        ok = sd > 1e-8 * np.maximum(1.0, np.abs(mu))
        out.append(np.where(ok, (X - mu) / np.where(ok, sd, 1.0), 0.0) * w)
    return np.concatenate(out, axis=1)


# ─── Expression matrix of the view ────────────────────────────────────────────
def _bk_matrix(samples, genes, log1p: bool = False) -> tuple:
    """(cells × genes CSR of the cells with a matrix row, per sample the positions of those
    cells in its table, per sample the counts): counts ÷ the cell's total over *genes* ×
    the median total of all cells (the normalisation the paper uses), optionally log1p."""
    parts, pos, tots = [], [], []
    for s in samples:
        rows = np.asarray(s["rows"])[:len(s["df"])]
        ok = np.flatnonzero(rows >= 0)
        gi = np.array([s["gni"][g] for g in genes], dtype=np.int64)
        X = sp.csr_matrix(s["csc"][:, gi].tocsr()[rows[ok]], dtype=np.float32)
        parts.append(X)
        pos.append(ok)
        tots.append(np.asarray(X.sum(axis=1)).ravel())
    tot = np.concatenate(tots) if tots else np.zeros(0)
    med = float(np.median(tot[tot > 0])) if (tot > 0).any() else 1.0
    X = sp.vstack(parts, format="csr") if parts else sp.csr_matrix((0, len(genes)), dtype=np.float32)
    f = np.where(tot > 0, med / np.where(tot > 0, tot, 1.0), 0.0).astype(np.float32)
    X = sp.diags(f).dot(X).tocsr()
    if log1p:
        X.data = np.log1p(X.data)
    return X, pos, med


def _bk_select_genes(X, max_genes: int = _BK_MAX_GENES, min_cells: int = 10) -> np.ndarray:
    """Columns used as features: genes detected in at least *min_cells* cells (5 % of the
    cells when that is fewer); when there
    are more than *max_genes*, those with the largest variance of log1p(normalised
    expression) (indices in ascending order)."""
    X = sp.csc_matrix(X)
    n = X.shape[0]
    det = np.diff(X.indptr)
    cand = np.flatnonzero(det >= max(1, min(int(min_cells), n // 20)))
    if max_genes and len(cand) > int(max_genes):
        L = X[:, cand].copy()
        L.data = np.log1p(L.data)
        m = np.asarray(L.mean(axis=0)).ravel()
        v = np.asarray(L.multiply(L).mean(axis=0)).ravel() - m ** 2
        cand = np.sort(cand[np.argsort(-v, kind="stable")[:int(max_genes)]])
    return cand


def _bk_name(lam: float) -> str:
    return f"Banksy-style λ={float(lam):g}"


# ─── Summaries of a partition ─────────────────────────────────────────────────
def _bk_neighbour_purity(labels, Wm) -> np.ndarray:
    """Per cell: share of its spatial neighbours (the k of the mean, unweighted) that carry
    its own label."""
    labels = np.asarray(labels)
    A = Wm.tocoo()
    same = (labels[A.row] == labels[A.col]).astype(np.float64)
    num = np.bincount(A.row, weights=same, minlength=len(labels))
    den = np.bincount(A.row, minlength=len(labels))
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def _bk_markers(Cn, labels, genes, top: int = _BK_TOP_GENES) -> tuple:
    """(K × genes matrix of the cluster mean of every gene in standard deviations from the
    overall mean, the *top* genes of each cluster by that value)."""
    Cn = sp.csr_matrix(Cn)
    labels = np.asarray(labels, dtype=np.int64)
    K = int(labels.max()) + 1 if len(labels) else 0
    n = Cn.shape[0]
    H = sp.csr_matrix((np.ones(n, dtype=np.float32), (labels, np.arange(n))), shape=(K, n))
    cnt = np.maximum(np.bincount(labels, minlength=K), 1)[:, None]
    mean_k = np.asarray((H @ Cn).todense()) / cnt
    mu = np.asarray(Cn.mean(axis=0)).ravel()
    sd = np.sqrt(np.maximum(np.asarray(Cn.multiply(Cn).mean(axis=0)).ravel() - mu ** 2, 0.0))
    z = (mean_k - mu[None, :]) / np.where(sd > 0, sd, 1.0)[None, :]
    z[:, sd <= 0] = 0.0
    tops = [[genes[j] for j in np.argsort(-z[c], kind="stable")[:int(top)] if z[c, j] > 0] for c in range(K)]
    return z, tops


# ─── Cache ────────────────────────────────────────────────────────────────────
def _bk_cache_file(sig: str) -> str:
    return os.path.join(core.CACHE_BASE_DIR, f"banksy_{analysis._an_dataset_tag()}", f"{sig}.npz")


def _bk_gram_gb(n_genes: int, agf: bool = True) -> float:
    """Memory of the covariance step of _bk_embed, in GB: the Gram matrix of the augmented
    features is (blocks × genes)² float64 whatever the number of cells (blocks: own
    expression, neighbourhood mean and, with the azimuthal Gabor filter, its magnitude),
    beside the float32 product of a block and its float64 copy, and one dense block of
    cells. With the gene cap raised on a whole-transcriptome matrix this is tens of GB."""
    d = (3 if agf else 2) * max(0, int(n_genes))
    return 2.5 * d * d * 8 / 1e9 + _BK_CHUNK * d * 4 / 1e9


def _bk_signature(samples, seg_source, params, genes) -> str:
    h = hashlib.md5()
    # which clustering backend can run is part of the key (as joint._jc_backends): a
    # KMeans-fallback partition is not handed back once leidenalg / igraph are installed
    h.update(json.dumps({"v": _BK_CACHE_VERSION, "p": params, "seg": controls._seg_tag(seg_source),
                         "leiden": bool(clustersets._cs_leiden_available())},
                        sort_keys=True, default=str).encode())
    for s in samples:
        h.update(str(s["key"]).encode())
        h.update(b"corrected" if s["corrected"] else b"original")
        h.update(analysis._an_matrix_digest(s["csc"]).encode())     # the counts themselves, not only whose they are
        h.update(pd.util.hash_pandas_object(pd.Index(s["df"].index.astype(str)), index=False).values.tobytes())
    h.update("|".join(genes).encode())
    return h.hexdigest()[:16]


# ─── The job ──────────────────────────────────────────────────────────────────
def _bk_params(params=None) -> dict:
    p = {"lam": _BK_LAMBDA_DOMAINS, "k_geom": _BK_K_GEOM, "agf": True, "n_pcs": _BK_N_PCS, "k_expr": _BK_K_EXPR,
         "resolution": _BK_RESOLUTION, "max_genes": _BK_MAX_GENES, "log1p": False, "center": True, "seed": 0}
    p.update({k: v for k, v in (params or {}).items() if v is not None and k in p})
    p["lam"] = float(min(1.0, max(0.0, float(p["lam"]))))
    p["k_geom"] = int(max(2, min(200, int(p["k_geom"]))))
    p["n_pcs"] = int(max(2, min(100, int(p["n_pcs"]))))
    p["k_expr"] = int(max(2, min(200, int(p["k_expr"]))))
    p["resolution"] = float(max(0.01, min(10.0, float(p["resolution"]))))
    p["max_genes"] = int(max(10, min(20000, int(p["max_genes"]))))
    p["agf"], p["log1p"], p["center"], p["seed"] = bool(p["agf"]), bool(p["log1p"]), bool(p["center"]), int(p["seed"])
    return p


def _bk_compute(samples, params, progress=None) -> dict:
    """Banksy on the cells of *samples* (analysis._an_samples dicts). Returns labels (0 …
    K−1 over the cells with a matrix row, the samples one after the other), pos (per
    sample, the table positions of those cells), genes, pcs, W, Cn, info."""
    p = _bk_params(params)
    prog = progress or (lambda f, m=None: None)
    genes_all = analysis._an_common_genes(samples)
    if not genes_all:
        raise ValueError("The samples of the view have no gene in common.")
    prog(0.02, "Banksy: normalising the counts…")
    X, pos, med = _bk_matrix(samples, genes_all, p["log1p"])
    n = X.shape[0]
    if n < _BK_MIN_CELLS:
        raise ValueError(f"Banksy-style clustering needs at least {_BK_MIN_CELLS} cells with counts; the view has {n:,}.")
    jobs._job_check()
    sel = _bk_select_genes(X, p["max_genes"])
    if len(sel) < 2:
        raise ValueError("Fewer than 2 genes are detected in enough cells.")
    Cn = X[:, sel].tocsr()
    genes = [genes_all[j] for j in sel]
    prog(0.08, f"Banksy: spatial neighbours of {n:,} cells (k = {p['k_geom']}, within sample)…")
    xys = [stats._xy_of(s["df"])[ok] for s, ok in zip(samples, pos)]
    W = _bk_weight_matrices(xys, p["k_geom"])
    emb = _bk_embed(Cn, W, p["lam"], p["agf"], p["n_pcs"], p["center"],
                    progress=lambda f, m=None: prog(0.1 + 0.5 * f, m))
    labels, cinfo = clustersets._cs_graph_cluster(
        emb["pcs"], n_neighbors=p["k_expr"], resolution=p["resolution"], seed=p["seed"],
        progress=lambda f, m=None: prog(0.6 + 0.35 * f, None if m is None else f"Banksy: {m}"))
    return {"labels": labels, "pos": pos, "genes": genes, "n_genes_all": len(genes_all), "pcs": emb["pcs"],
            "W": W, "Cn": Cn, "cluster": cinfo, "explained": emb["explained"], "median_total": med,
            "params": p, "n": n}


def _bk_summarise(samples, out, tables_labels) -> dict:
    """Per-domain table, marker matrix and neighbour purity of a computed partition."""
    labels = out["labels"]
    K = int(labels.max()) + 1
    pur = _bk_neighbour_purity(labels, out["W"]["m"])
    z, tops = _bk_markers(out["Cn"], labels, out["genes"])
    n = len(labels)
    frac = np.bincount(labels, minlength=K) / max(n, 1)
    sidx = np.concatenate([np.full(len(ok), i) for i, ok in enumerate(out["pos"])])
    per_sample = np.zeros((K, len(samples)), dtype=np.int64)
    np.add.at(per_sample, (labels, sidx), 1)
    rows = []
    for c in range(K):
        m = labels == c
        rows.append({"domain": c + 1, "n_cells": int(m.sum()), "share": float(frac[c]),
                     "neighbour_purity": float(np.nanmean(pur[m])) if m.any() else np.nan,
                     "purity_expected": float(frac[c]),
                     "top_genes": ", ".join(tops[c]),
                     **{f"n_{s['label']}": int(per_sample[c, i]) for i, s in enumerate(samples)}})
    # the genes of the heat map: the top genes of every domain, in domain order
    hg = list(dict.fromkeys(g for t in tops for g in t[:3]))
    gi = [out["genes"].index(g) for g in hg]
    return {"table": pd.DataFrame(rows), "heat_genes": hg, "heat": z[:, gi] if gi else np.zeros((K, 0)),
            "purity": float(np.nanmean(pur)), "purity_expected": float((frac ** 2).sum()),
            "sample_labels": [s["label"] for s in samples]}


def _run_banksy(seg_source, use_corrected, params=None) -> None:
    """Background job: Banksy clusters of the cells of the active view, installed as the
    cluster set "Banksy-style λ=…"."""
    _gen = jobs._dataset_gen
    seg_source = seg_source or "xenium"

    def _set(status, message, result=None):
        with _bk_lock:
            if jobs._stale_job(_gen, status, message):
                return
            _bk_state["status"], _bk_state["message"] = status, message
            if result is not None:
                _bk_state["result"] = result
            if status in ("done", "error"):
                _bk_state["version"] += 1
        if status == "done":
            provenance._prov_record("banksy", seg_source=seg_source, corrected=use_corrected, params=dict(params or {}), packages=("leidenalg", "igraph"), message=message, result=result)

    try:
        _t0 = _time.time()
        p = _bk_params(params)
        _set("running", "Collecting the cells and counts of the view…")
        samples = analysis._an_samples(seg_source, bool(use_corrected))
        tables = stats._view_tables(seg_source)
        if not samples:
            _set("error", analysis._an_no_matrix_msg(seg_source) if analysis._an_seg_missing(seg_source)
                 else "The active view has no count matrix.")
            return
        by_key = {s["key"]: s for s in samples}
        missing = [t["label"] for t in tables if t["key"] not in by_key]
        genes_all = analysis._an_common_genes(samples)
        sig = _bk_signature(samples, seg_source, p, genes_all)
        cache = _bk_cache_file(sig)
        cached = None
        if os.path.isfile(cache):
            try:
                with np.load(cache, allow_pickle=False) as z:
                    cached = {"labels": z["labels"], "summary": json.loads(str(z["summary"]))}
                if len(cached["labels"]) != sum(len(t["df"]) for t in tables):
                    cached = None
            except Exception:
                cached = None

        def _progress(f, m=None):
            jobs._job_progress(m, f)
            if m:
                _set("running", m)

        if cached is None:
            # the cost up front: the graph of k_expr neighbours of every cell is what limits it
            _n = int(sum(int((np.asarray(smp["rows"])[:len(smp["df"])] >= 0).sum()) for smp in samples))
            _stop, _est = clustersets._cs_size_guard(
                "Banksy-style clustering", _n, p["k_expr"],
                matrix_gb=(_BK_MATRIX_GB_PER_M_CELLS * _n / 1e6
                           + _bk_gram_gb(min(int(p["max_genes"]), len(genes_all)), p["agf"])),
                extra_minutes=_BK_MIN_PER_M_CELLS * _n / 1e6)
            if _stop:
                _set("error", _stop)
                return
            if _est:
                print(f"  Banksy: {_est}", flush=True)
                _set("running", _est)
            out = _bk_compute(samples, p, _progress)
            jobs._check_dataset_gen(_gen)
            summ = _bk_summarise(samples, out, None)
            # labels per view table (−1: no label)
            per, flat = [], []
            si = {s["key"]: i for i, s in enumerate(samples)}
            offs = np.concatenate([[0], np.cumsum([len(o) for o in out["pos"]])])
            for t in tables:
                lab = np.full(len(t["df"]), -1, dtype=np.int64)
                i = si.get(t["key"])
                if i is not None:
                    lab[out["pos"][i]] = out["labels"][offs[i]:offs[i + 1]]
                per.append(lab)
            flat = np.concatenate(per)
            summary = {"table": summ["table"].to_dict("records"), "heat_genes": summ["heat_genes"],
                       "heat": np.round(summ["heat"], 4).tolist(), "purity": summ["purity"],
                       "purity_expected": summ["purity_expected"], "sample_labels": summ["sample_labels"],
                       "n": int(out["n"]), "n_genes": len(out["genes"]), "n_genes_all": int(out["n_genes_all"]),
                       "explained": float(out["explained"]), "cluster": out["cluster"],
                       "median_total": float(out["median_total"])}
        else:
            _set("running", "Banksy: reading the cached result…")
            flat = np.asarray(cached["labels"], dtype=np.int64)
            summary = cached["summary"]
            per, o = [], 0
            for t in tables:
                per.append(flat[o:o + len(t["df"])])
                o += len(t["df"])
        jobs._check_dataset_gen(_gen)
        jobs._job_shield("Installing the Banksy cluster set…")
        name = _bk_name(p["lam"])
        labs = [np.where(l >= 0, l, np.nan).astype(object) for l in per]
        for l in labs:
            l[pd.isna(l)] = None
        key = clustersets._cs_install(name, labs, seg_source,
                                      meta={"method": "banksy", "params": p, "shared": len(tables) > 1,
                                            "backend": summary["cluster"]["backend"]})
        if cached is None:
            try:
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                tmp = cache + ".tmp.npz"
                np.savez_compressed(tmp, labels=flat.astype(np.int32), summary=np.array(json.dumps(summary)))
                os.replace(tmp, cache)
            except Exception as exc:
                print(f"  Banksy: result not cached ({exc})", flush=True)
        table = pd.DataFrame(summary["table"])
        K = len(table)
        res = {"table": table, "heat_genes": summary["heat_genes"], "heat": np.asarray(summary["heat"], dtype=float),
               "params": p, "name": name, "key": key, "seg_source": seg_source, "gen": _gen,
               "seg_tag": controls._seg_tag(seg_source), "summary": summary, "cached": cached is not None,
               "corrected": bool(samples[0]["corrected"]), "seconds": _time.time() - _t0,
               "n_samples": len(samples), "has_unassigned": bool((flat < 0).any()),
               # cells of the samples that were clustered but have no cluster: no matrix row
               # (in corrected mode, no SPLIT-corrected counts) - left out, not zeros
               "n_no_counts": int(sum(int((np.asarray(l) < 0).sum()) for t, l in zip(tables, per)
                                      if t["key"] in by_key)),
               "n_uncorrected": analysis._an_uncorrected(samples)}
        ci = summary["cluster"]
        blocks = ("own expression, neighbourhood mean and AGF" if p["agf"] else "own expression and neighbourhood mean")
        how = "read from the cache" if cached is not None else f"{res['seconds']:.0f} s"
        msg = (f"{name}: {K} cluster(s) of {summary['n']:,} cells in {len(samples)} sample(s) ({how}). "
               f"Features: {blocks} of {summary['n_genes']:,} genes"
               + (f" (the most variable of {summary['n_genes_all']:,})" if summary["n_genes"] < summary["n_genes_all"] else "")
               + f"; k = {p['k_geom']} spatial neighbours within the sample ({2 * p['k_geom']} for the AGF); "
                 f"{p['n_pcs']} PCs hold {100 * summary['explained']:.0f} % of the variance; {ci['backend']} "
                 f"({ci.get('neighbours') or 'no graph'}, k = {p['k_expr']}, resolution {p['resolution']:g}, seed {p['seed']}); "
                 f"{'SPLIT-corrected' if res['corrected'] else 'original'} counts scaled to the median total"
               + (", log1p" if p["log1p"] else "") + "."
               + f"\nSpatial coherence: on average {100 * summary['purity']:.0f} % of a cell's {p['k_geom']} nearest cells "
                 f"are in its own cluster ({100 * summary['purity_expected']:.0f} % expected if the labels were "
                 f"scattered at random)."
               + f"\nShown as cluster set '{name}': Color By → Cluster, Clustering Method → {name}. "
                 f"0 marks cells without a label."
               + "\n" + _BK_UNVALIDATED)
        if ci["backend"] != "leiden":
            msg += "\n⚠ " + clustersets._CS_LEIDEN_HINT
        if len(samples) > 1:
            msg += ("\nSeveral samples: neighbours were taken within each sample; scaling, PCA and clustering "
                    "were done on all cells together, so the labels are the same in every sample. No batch "
                    "correction is applied: a cluster found in one sample only may be a batch effect.")
        if res["n_no_counts"]:
            msg += (f"\n{res['n_no_counts']:,} cell(s) without "
                    f"{'SPLIT-corrected ' if res['corrected'] else ''}counts were left out and have no cluster.")
        if missing:
            msg += f"\nLeft out (no count matrix): {', '.join(missing)}."
        if use_corrected and not res["corrected"]:
            msg += "\nOriginal counts were used: not every sample has SPLIT-corrected counts."
        _set("done", msg, result=res)
        print(f"  Banksy: {msg}", flush=True)
    except jobs._JobCancelled:
        raise
    except jobs._DatasetChanged:
        raise
    except ValueError as exc:
        _set("error", str(exc))
    except MemoryError:
        _set("error", "Not enough memory: lower the gene cap or subset the cells.")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set("error", str(exc)[:300])


# ─── Figure, table, CSV ───────────────────────────────────────────────────────
def make_bk_fig(width=None, height=None):
    """Left: cells per cluster (the bar colours of the map). Right: mean expression of the
    top genes of every cluster, in standard deviations from the mean over all cells."""
    from plotly.subplots import make_subplots
    res = _bk_result()
    if not res:
        return figures._make_placeholder_fig("No Banksy-style domains computed yet.")
    t = res["table"]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.3, 0.7], horizontal_spacing=0.08,
                        subplot_titles=("Cells per cluster", "Top genes (SD from the mean of all cells)"))
    # (colours as Color By → Cluster draws them: 0, when present, takes the first colour)
    names = [str(d) for d in t["domain"]]
    off = 1 if res.get("has_unassigned") else 0
    cols = [core.CLUSTER_COLORS[(i + off) % len(core.CLUSTER_COLORS)] for i in range(len(t))]
    fig.add_trace(go.Bar(x=names, y=t["n_cells"], marker_color=cols,
                         customdata=np.column_stack([100 * t["share"], 100 * t["neighbour_purity"]]),
                         hovertemplate="Cluster %{x}<br>%{y:,} cells (%{customdata[0]:.1f} %)<br>"
                                       "neighbour purity %{customdata[1]:.0f} %<extra></extra>"), row=1, col=1)
    if len(res["heat_genes"]):
        z = np.asarray(res["heat"], dtype=float)
        lim = float(np.nanmax(np.abs(z))) if z.size else 1.0
        fig.add_trace(go.Heatmap(z=z, x=res["heat_genes"], y=names, colorscale="RdBu_r", zmid=0,
                                 zmin=-lim, zmax=lim, colorbar=dict(thickness=10, len=0.8, title="SD"),
                                 hovertemplate="Cluster %{y}<br>%{x}: %{z:.2f} SD<extra></extra>"), row=1, col=2)
        fig.update_yaxes(type="category", autorange="reversed", row=1, col=2)
        fig.update_xaxes(tickangle=-60, tickfont=dict(size=8), row=1, col=2)
    fig.update_xaxes(type="category", row=1, col=1)
    fig.update_annotations(font=dict(size=10, color=core.TEXT))
    return controls._fig_layout(fig, f"{res['name']}: {len(t)} clusters", width, height,
                                margin=dict(l=50, r=10, t=60, b=90))


_BK_COLS = [("domain", "Cluster"), ("n_cells", "Cells"), ("share", "Share"),
            ("neighbour_purity", "Neighbour purity"), ("purity_expected", "… if random"),
            ("top_genes", "Top genes")]


def _bk_records(res) -> list:
    if not res:
        return []
    t = res["table"]
    out = pd.DataFrame({c: t[c] for c, _n in _BK_COLS if c in t.columns})
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda v: None if not np.isfinite(v) else float(f"{v:.3g}"))
    return out.astype(object).where(out.notna(), None).to_dict("records")


def _bk_csv(res) -> str:
    p, s = res["params"], res["summary"]
    lines = [f"# {res['name']}; segmentation: {res['seg_source']}; "
             f"{'SPLIT-corrected' if res['corrected'] else 'original'} counts scaled to the median total"
             + ("; log1p" if p["log1p"] else ""),
             f"# lambda {p['lam']:g}; k_geom {p['k_geom']} (AGF: {2 * p['k_geom'] if p['agf'] else 'off'}"
             f"{', centred' if p['agf'] and p['center'] else ''}); {s['n_genes']} of {s['n_genes_all']} genes; "
             f"{p['n_pcs']} PCs; clustering: {s['cluster']['backend']}, k_expr {p['k_expr']}, resolution "
             f"{p['resolution']:g}, seed {p['seed']}",
             "# Banksy-style spatial domains: native implementation following the construction of Banksy "
             "(Singhal et al., Nat Genet 2024), not validated against the reference implementation; "
             "neighbours within sample",
             f"# neighbour_purity: mean share of a cell's {p['k_geom']} nearest cells in its own cluster; "
             f"purity_expected: the share of the cluster among all cells",
             "# cluster ids are the values of the cluster set (0 = no label)"]
    if res.get("n_no_counts"):
        lines.append(f"# {res['n_no_counts']} cell(s) without "
                     f"{'SPLIT-corrected ' if res['corrected'] else ''}counts were left out and have no cluster")
    heat = pd.DataFrame(np.asarray(res["heat"], dtype=float), columns=res["heat_genes"])
    heat.insert(0, "domain", res["table"]["domain"].to_numpy())
    return provenance._prov_csv_stamp(res, ("\n".join(lines) + "\n# clusters\n" + res["table"].to_csv(index=False)
            + "\n# mean expression of the top genes, SD from the mean of all cells\n" + heat.to_csv(index=False)))


# ─── Layout ───────────────────────────────────────────────────────────────────
def _bk_tab():
    S, L = analysis._AN_ROW_STYLE, analysis._AN_LABEL_STYLE
    num = controls._num
    cols = [{"id": c, "name": n, "type": "text" if c == "top_genes" else "numeric"} for c, n in _BK_COLS]
    chk = lambda text, on, id=None: dcc.Checklist(
        id=id, options=[{"label": html.Span(text, style={"fontSize": "11px", "color": core.TEXT}), "value": "on"}],
        value=["on"] if on else [], inputStyle={"marginRight": "3px"}, style={"display": "inline-flex"})
    return dbc.Tab(label="Banksy-style domains", tab_id="tab-banksy",
                   label_style={"fontSize": "12px", "padding": "4px 12px"},
                   children=[html.Div([
        html.Div([
            html.Span("Preset:", style=L),
            dcc.Dropdown(id="bk-preset",
                         options=[{"label": f"Tissue domains (λ = {_BK_LAMBDA_DOMAINS:g})", "value": "domains"},
                                  {"label": f"Cell typing (λ = {_BK_LAMBDA_CELLTYPE:g})", "value": "celltype"},
                                  {"label": "Custom λ", "value": "custom"}],
                         value="domains", clearable=False, style={"width": "210px", "fontSize": "11px"}),
            html.Span("λ:", style=L), num(id="bk-lambda", value=_BK_LAMBDA_DOMAINS, min=0, max=1, step=0.05),
            html.Span("Resolution:", style=L), num(id="bk-resolution", value=_BK_RESOLUTION, min=0.01, max=10, step=0.05),
            html.Span("" if clustersets._cs_leiden_available() else "⚠ " + clustersets._CS_LEIDEN_HINT,
                      id="bk-backend-note", style={**L, "color": "#d29922", "flex": "1"}),
            dbc.Button("Run", id="bk-run-btn", color="primary", size="sm", n_clicks=0,
                       style={"fontSize": "11px", "padding": "2px 12px"}),
        ], style=S),
        ui._options([html.Div([
            html.Span("Spatial neighbours k:", style=L), num(id="bk-kgeom", value=_BK_K_GEOM, min=2, max=200, step=1),
            chk(" AGF (gradient) block", True, id="bk-agf"),
            html.Span("Genes ≤:", style=L), num(id="bk-maxgenes", value=_BK_MAX_GENES, min=10, max=20000, step=10, width="70px"),
            html.Span("PCs:", style=L), num(id="bk-npcs", value=_BK_N_PCS, min=2, max=100, step=1),
            html.Span("Graph neighbours:", style=L), num(id="bk-kexpr", value=_BK_K_EXPR, min=2, max=200, step=1),
            html.Span("Seed:", style=L), num(id="bk-seed", value=0, min=0, max=10 ** 6, step=1),
            chk(" log1p", False, id="bk-log1p"),
        ], style=S)], summary=f"Options (k = {_BK_K_GEOM} spatial neighbours, AGF on, ≤ {_BK_MAX_GENES:,} genes, "
                              f"{_BK_N_PCS} PCs, {_BK_K_EXPR} graph neighbours, seed 0)"),
        html.Div(id="bk-status", style={**analysis._AN_STATUS_STYLE, "whiteSpace": "pre-wrap"}),
        html.Div(_BK_NOTE, id="bk-note", style=analysis._AN_NOTE_STYLE),
        html.Div([
            controls._graph(id="bk-plot", msg="No Banksy-style domains computed yet.", filename="xenium_banksy",
                            min_h="320px"),
            html.Div([
                dash_table.DataTable(id="bk-table", data=[], columns=cols, **stats._dark_table_style(12)),
                html.Div([
                    dbc.Button("Download CSV", id="bk-download-btn", color="secondary", size="sm", n_clicks=0,
                               style={"fontSize": "11px", "padding": "2px 8px"}),
                    dcc.Download(id="bk-download"),
                ], style={"padding": "4px 0"}),
            ], style={"flex": "1", "minWidth": "0", "overflow": "auto", "padding": "0 6px"}),
        ], style={"display": "flex", "flex": "1", "minHeight": "400px", "gap": "6px"}),
        ui._plot_dim_controls("bk"),
    ], style={"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0",
              "overflowY": "auto"})])
