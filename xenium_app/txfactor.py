"""Transcript factors: a segmentation-free factorisation of binned transcripts (compute +
layout).

The gene transcripts of a sample (Q ≥ 20, gene features: the filter of the cell-feature
matrix) are counted on a square or hexagonal grid without looking at any cell, a
non-negative factorisation (NMF with the Kullback-Leibler loss, or online LDA) is fitted on
a random subsample of the bins, and every bin is then projected onto the fitted gene
loadings. A simplified transcript-binning factorisation in the spirit of FICTURE (Si et
al., Nat Methods 2024); it is not that tool: no anchors, no pixel-level decoding.

Every sample is binned on its own grid in its own µm; the loadings are fitted once on bins
pooled from all samples.
"""

import os
import re
import json
import hashlib
import threading
import time as _time
from collections import OrderedDict as _TxfOrderedDict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from dash import dcc, html
from dash import dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from . import core, jobs, dataview, ui, figures, analysis, reseg, stats, controls
from . import provenance


# ─── State ────────────────────────────────────────────────────────────────────
_txf_state: dict = {"status": "idle", "message": "", "result": None, "version": 0}
_txf_lock = threading.Lock()
_txf_mem: "_TxfOrderedDict" = _TxfOrderedDict()      # rasters of the map layer (small LRU)
_txf_mem_lock = threading.Lock()

_TXF_BIN_UM = 10.0            # bin size: side of a square, centre-to-centre distance of hexagons
_TXF_K = 12                   # factors
_TXF_MIN_TX = 10              # transcripts a bin needs to be fitted on / to get a dominant factor
_TXF_MAX_FIT = 20_000         # bins the loadings are fitted on (random subsample above that)
_TXF_SEED = 0
_TXF_FIT_ITER = 150           # NMF: multiplicative updates at most
_TXF_FIT_ROUND = 15           # ... run in rounds of this many (progress, cancellation, convergence)
_TXF_FIT_TOL = 0.005          # ... until no factor's gene distribution moves by more than this (L1) in a round
_TXF_LDA_EPOCHS = 5           # online LDA: passes over the fitted bins
_TXF_LDA_BATCH = 2048
_TXF_PROJ_ITER = 30           # projection of a bin onto fixed loadings: EM updates at most
_TXF_PROJ_TOL = 1e-3          # ... stop when the factor counts move by less than this share
_TXF_CHUNK_NNZ = 1_500_000    # non-zeros of the bin × gene matrix projected at a time
_TXF_THREADS = 4
_TXF_USE_NUMBA = True         # compiled EM when numba is installed (optional); numpy otherwise
_TXF_REF_MIN_CELLS = 20       # reference mode: a cell type needs this many cells to be a component
_TXF_CMP_MIN_CELLS = 5        # ... and this many to be compared with the fitted loadings (cosine)
_TXF_MAX_K = 60
_TXF_TOP_GENES = 10           # genes listed per factor in the table (the CSV has _TXF_CSV_GENES)
_TXF_CSV_GENES = 50
_TXF_RING = 0.6               # per cell: bins under the centroid and under 6 points at this share of its radius
_TXF_RASTER_MAX = 16_000_000  # hexagons are drawn on a square raster of at most this many pixels
_TXF_GENE = "@txf:"           # heat-layer "gene" values: @txf:dominant, @txf:<factor index>
_TXF_COLOR = "txf#"           # Color By values: geneset:txf#mixed, geneset:txf#top, geneset:txf#share:<i>
_TXF_METHODS = {"nmf": "NMF (KL loss)", "lda": "LDA (online)", "reference": "Reference: cell-type profiles"}

_TXF_NOTE = (
    "Transcript factors: a simplified transcript-binning factorisation in the spirit of FICTURE "
    "(Si et al., Nat Methods 2024) - not that tool (no anchors, no pixel-level decoding). No cell "
    "is used: the gene transcripts (Q ≥ 20, gene features - the filter of the cell-feature matrix) "
    "are counted per bin of a square or hexagonal grid (bin size = side of the square / "
    "centre-to-centre distance of the hexagons), each sample on its own grid. NMF with the "
    "Kullback-Leibler loss (Poisson likelihood; multiplicative updates) or online LDA is fitted on "
    "a random subsample of the bins with at least the given number of transcripts (fixed seed, "
    "bins pooled over the samples); every bin is then projected onto the fitted gene loadings (EM "
    "for the factor counts with the loadings fixed, i.e. the maximum-likelihood mixture without "
    "LDA's Dirichlet prior), so a factor's value in a bin is the number of the bin's transcripts "
    "attributed to it. Reference mode: the loadings are the mean profiles of the cell types of the "
    "chosen annotation (active counts mode) and are not fitted; the factors then carry the type "
    "names. Comparison with the segmentation: for every cell, the factor counts of the bins under "
    "it (its centroid and six points at 0.6 × the radius of a disc of its area) are summed into a "
    "mixture; 'mixed' = entropy of that mixture ÷ ln K (0: one factor, 1: all equal). A bin is "
    "larger than most cells, so the mixture describes the cell's position, not its own counts: a "
    "high value marks places where a segmented cell collects transcripts of several programs. "
    "Factors are fitted on one random initialisation / subsample; change the seed or K to see what "
    "is stable. Recompute after changing the subset, the segmentation or the annotation.")


def _txf_reset() -> None:
    """Forget the result (another dataset was loaded; tests)."""
    with _txf_lock:
        _txf_state.update({"status": "idle", "message": "", "result": None})
        _txf_state["version"] += 1
    with _txf_mem_lock:
        _txf_mem.clear()


def _txf_result():
    """The published result when it belongs to the loaded dataset, else None."""
    with _txf_lock:
        res = _txf_state.get("result")
    if not res or res.get("gen") != jobs._dataset_gen:
        return None
    return res


# ─── Bins ─────────────────────────────────────────────────────────────────────
_TXF_OFF, _TXF_W = 1 << 20, 1 << 22          # bin indices from -2**20 to 3·2**20 fit one key


def _txf_bin_index(x, y, bin_um: float, shape: str = "square") -> tuple:
    """(r, c) integer bin of points (µm). square: bin (r, c) covers [c·b, (c+1)·b) ×
    [r·b, (r+1)·b). hex: regular hexagons whose centres are *bin_um* apart, in doubled
    coordinates - the centre of bin (r, c) is (c · b / 2, r · b · √3 / 2) and r + c is
    even; a point belongs to the nearest centre."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    b = float(bin_um)
    if shape != "hex":
        return np.floor(y / b).astype(np.int64), np.floor(x / b).astype(np.int64)
    h = b * np.sqrt(3.0)
    ia, ja = np.rint(x / b), np.rint(y / h)                 # lattice A: (i·b, j·h)
    ib, jb = np.floor(x / b), np.floor(y / h)               # lattice B: ((i+½)·b, (j+½)·h)
    da = (x - ia * b) ** 2 + (y - ja * h) ** 2
    db = (x - (ib + 0.5) * b) ** 2 + (y - (jb + 0.5) * h) ** 2
    a = da <= db
    return (np.where(a, 2 * ja, 2 * jb + 1).astype(np.int64),
            np.where(a, 2 * ia, 2 * ib + 1).astype(np.int64))


def _txf_bin_centres(r, c, bin_um: float, shape: str = "square") -> tuple:
    """(x, y) centre in µm of bins (r, c)."""
    r, c, b = np.asarray(r, dtype=np.float64), np.asarray(c, dtype=np.float64), float(bin_um)
    if shape != "hex":
        return (c + 0.5) * b, (r + 0.5) * b
    return c * b / 2.0, r * b * np.sqrt(3.0) / 2.0


def _txf_bin_area(bin_um: float, shape: str = "square") -> float:
    """Area of a bin in µm²."""
    return float(bin_um) ** 2 * (np.sqrt(3.0) / 2.0 if shape == "hex" else 1.0)


def _txf_bin_key(r, c) -> np.ndarray:
    return (np.asarray(r, dtype=np.int64) + _TXF_OFF) * _TXF_W + (np.asarray(c, dtype=np.int64) + _TXF_OFF)


# ─── Transcript scan: bins × genes ────────────────────────────────────────────
def _txf_tx_batches(data_dir: str, genes, qv_min=20, source: str = "auto"):
    """Yield (x, y, gene index, fraction done) of the transcripts that pass the app's
    transcript filters (Q ≥ qv_min, gene features only) and belong to one of *genes*, one
    row group of transcripts.parquet (or one tile of transcripts.zarr.zip) at a time."""
    gpos = {g: i for i, g in enumerate(genes)}
    pq_path = os.path.join(data_dir, "transcripts.parquet")
    if source in ("auto", "parquet") and os.path.exists(pq_path):
        import pyarrow as _pa
        import pyarrow.compute as _pc
        import pyarrow.dataset as _ds
        import pyarrow.parquet as _pq
        dataset = _ds.dataset(pq_path, format="parquet")
        flt = reseg._tx_gene_filters(dataset.schema.names, qv_min)
        expr = _pq.filters_to_expression(flt) if flt else None
        groups = [rg for frag in dataset.get_fragments() for rg in frag.split_by_row_group()]
        for i, rg in enumerate(groups):
            for batch in rg.to_batches(columns=["x_location", "y_location", "feature_name"], filter=expr,
                                       batch_size=1 << 20):
                if not batch.num_rows:
                    continue
                col = batch.column("feature_name")
                if not _pa.types.is_dictionary(col.type):
                    col = _pc.dictionary_encode(col)
                names = col.dictionary.to_pylist()
                lut = np.array([gpos.get(v.decode() if isinstance(v, bytes) else v, -1) for v in names] + [-1],
                               dtype=np.int64)
                codes = col.indices.fill_null(len(names)).to_numpy(zero_copy_only=False).astype(np.int64)
                g = lut[codes]
                keep = g >= 0
                yield (batch.column("x_location").to_numpy(zero_copy_only=False)[keep],
                       batch.column("y_location").to_numpy(zero_copy_only=False)[keep], g[keep], i / len(groups))
            yield np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64), (i + 1) / len(groups)
        return
    if not os.path.exists(os.path.join(data_dir, "transcripts.zarr.zip")):
        raise ValueError("This bundle has no transcripts.parquet and no transcripts.zarr.zip.")
    import xenium_viewer_api as _va
    store = _va.tile_store(data_dir)
    m = store._tx_open()
    if not m:
        raise ValueError("This bundle has no transcripts.parquet and no transcripts.zarr.zip.")
    names = [n for n, _ in sorted(m["gene_index"].items(), key=lambda kv: kv[1])]
    lut = np.array([gpos.get(n, -1) for n in names], dtype=np.int64)
    keys = m["keys"][0]
    for i, key in enumerate(keys):
        grp = store._tx[f"grids/0/{key}"]
        gid = np.asarray(grp["gene_identity"][:]).reshape(-1).astype(np.int64)
        ok = (gid >= 0) & (gid < len(lut))
        g = np.full(len(gid), -1, dtype=np.int64)
        g[ok] = lut[gid[ok]]
        keep = g >= 0
        if "quality_score" in grp and qv_min is not None:
            keep &= np.asarray(grp["quality_score"][:], dtype=np.float32).reshape(-1) >= qv_min
        if keep.any():
            loc = np.asarray(grp["location"][:], dtype=np.float32)
            yield loc[keep, 0], loc[keep, 1], g[keep], (i + 1) / len(keys)
        else:
            yield np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64), (i + 1) / len(keys)


def _txf_scan(data_dir: str, genes, bin_um: float = _TXF_BIN_UM, shape: str = "square",
              progress=None, batches=None) -> dict:
    """Bin × gene counts of the filtered transcripts of one sample. One batch of
    transcripts is in memory at a time; the (bin, gene) pairs seen so far are kept as
    sorted keys with their counts and merged in chunks. Returns {r, c (int32, the
    non-empty bins in key order), X (bins × genes CSR, float32 counts), total}."""
    genes = list(genes)
    G = len(genes)
    keys, cnts, pending, total = [], [], 0, 0

    def _merge():
        if len(keys) > 1:
            uk, inv = np.unique(np.concatenate(keys), return_inverse=True)
            c = np.bincount(inv, weights=np.concatenate(cnts), minlength=len(uk)).astype(np.int32)
            keys[:], cnts[:] = [uk], [c]
        return len(keys[0]) if keys else 0

    merged = 0
    for x, y, g, frac in (batches if batches is not None else _txf_tx_batches(data_dir, genes)):
        jobs._job_check()
        if len(x):
            r, c = _txf_bin_index(x, y, bin_um, shape)
            uk, cn = np.unique(_txf_bin_key(r, c) * G + g, return_counts=True)
            keys.append(uk)
            cnts.append(cn.astype(np.int32))
            pending += len(uk)
            total += len(x)
            if pending > max(16_000_000, merged // 2):
                merged = _merge()
                pending = 0
        if progress is not None:
            progress(frac)
    if not keys:
        z = np.zeros(0, dtype=np.int32)
        return {"r": z, "c": z.copy(), "X": sp.csr_matrix((0, G), dtype=np.float32), "total": 0}
    _merge()
    uk, cn = keys[0], cnts[0]
    del keys, cnts
    bk = uk // G
    gi = (uk - bk * G).astype(np.int32)
    del uk
    start = np.flatnonzero(np.r_[True, bk[1:] != bk[:-1]])
    bins = bk[start]
    del bk
    indptr = np.r_[start, len(gi)].astype(np.int64)
    X = sp.csr_matrix((cn.astype(np.float32), gi, indptr), shape=(len(bins), G))
    return {"r": (bins // _TXF_W - _TXF_OFF).astype(np.int32), "c": (bins % _TXF_W - _TXF_OFF).astype(np.int32),
            "X": X, "total": int(total)}


def _txf_cache_dir(ds=None) -> str:
    """<cache>/txfactor_<dataset>/ (each sample has its own, as the density grids do)."""
    tag = (analysis._an_dataset_tag() if ds is None or ds is core.DATA
           else hashlib.md5(dataview._ds_key(ds).encode()).hexdigest()[:8])
    return os.path.join(core.CACHE_BASE_DIR, f"txfactor_{tag}")


def _txf_genes_hash(genes) -> str:
    return hashlib.md5("\n".join(genes).encode()).hexdigest()[:8]


def _txf_counts_file(ds, genes, bin_um: float, shape: str) -> str:
    return os.path.join(_txf_cache_dir(ds), f"counts__{shape}_b{float(bin_um):g}um__"
                                            f"{analysis._DENSITY_FILTER_VERSION}__g{_txf_genes_hash(genes)}.npz")


def _txf_counts(ds: dict, genes, bin_um: float, shape: str, progress=None) -> dict:
    """_txf_scan of dataset *ds*, from the cache when it was scanned before with the same
    bins and genes ("cached": True)."""
    data_dir = ds.get("data_dir")
    if not data_dir:
        raise ValueError(f"This sample was {dataview._origin_phrase(ds)}: it has no transcript file.")
    path = _txf_counts_file(ds, genes, bin_um, shape)
    if os.path.exists(path):
        try:
            with np.load(path) as z:
                X = sp.csr_matrix((z["data"].astype(np.float32), z["indices"].astype(np.int32),
                                   z["indptr"].astype(np.int64)), shape=(len(z["r"]), len(genes)))
                out = {"r": z["r"], "c": z["c"], "X": X, "total": int(z["total"]), "cached": True}
            if progress is not None:
                progress(1.0)
            return out
        except Exception as exc:
            print(f"  Transcript factors: unreadable count cache {os.path.basename(path)} ({exc}); rescanning",
                  flush=True)
    scan = _txf_scan(data_dir, genes, bin_um, shape, progress=progress)
    jobs._job_check()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    X = scan["X"]
    tmp = path + ".tmp.npz"
    np.savez(tmp, r=scan["r"], c=scan["c"], indptr=X.indptr.astype(np.int64),
             indices=X.indices.astype(np.uint16 if len(genes) < 65536 else np.int32),
             data=(X.data.astype(np.uint16) if (not len(X.data) or X.data.max() < 65536) else X.data.astype(np.float32)),
             total=np.int64(scan["total"]))
    os.replace(tmp, path)
    scan["cached"] = False
    return scan


# ─── Fit and projection ───────────────────────────────────────────────────────
def _txf_normalise(H) -> np.ndarray:
    """Rows of H as gene distributions (each sums to 1; a tiny floor keeps every gene
    explainable)."""
    H = np.maximum(np.asarray(H, dtype=np.float64), 0.0) + 1e-12
    return (H / H.sum(axis=1, keepdims=True)).astype(np.float32)


def _txf_fit(X, k: int, method: str = "nmf", seed: int = _TXF_SEED, max_iter: int = _TXF_FIT_ITER,
             progress=None) -> tuple:
    """Gene loadings of *k* factors fitted on the bin × gene counts X: (H, k × genes, rows
    sum to 1; iterations run). nmf: sklearn's NMF with the Kullback-Leibler loss
    (multiplicative updates, NNDSVDa start), run in rounds of _TXF_FIT_ROUND updates (it
    can be cancelled between them) until no factor's gene distribution moves by more than
    _TXF_FIT_TOL in a round, at most *max_iter* updates.
    lda: sklearn's online LatentDirichletAllocation, _TXF_LDA_EPOCHS passes in minibatches."""
    import warnings
    X = sp.csr_matrix(X, dtype=np.float32)
    n = X.shape[0]
    k = int(k)
    if method == "lda":
        from sklearn.decomposition import LatentDirichletAllocation
        lda = LatentDirichletAllocation(n_components=k, learning_method="online", random_state=int(seed),
                                        total_samples=n, batch_size=_TXF_LDA_BATCH)
        rng = np.random.default_rng(int(seed))
        steps = [(e, b0) for e in range(_TXF_LDA_EPOCHS) for b0 in range(0, n, _TXF_LDA_BATCH)]
        for si, (e, b0) in enumerate(steps):
            if b0 == 0:
                order = rng.permutation(n)
            jobs._job_check()
            lda.partial_fit(X[order[b0:b0 + _TXF_LDA_BATCH]])
            if progress is not None:
                progress((si + 1) / len(steps))
        return _txf_normalise(lda.components_), len(steps)
    from sklearn.decomposition import non_negative_factorization
    from sklearn.exceptions import ConvergenceWarning
    W = Hm = prev = None
    done = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        while done < int(max_iter):
            jobs._job_check()
            step = min(_TXF_FIT_ROUND, int(max_iter) - done)
            W, Hm, n_it = non_negative_factorization(
                X, W=W, H=Hm, n_components=k, init="nndsvda" if W is None else "custom", solver="mu",
                beta_loss="kullback-leibler", tol=0.0, max_iter=step, random_state=int(seed))
            done += int(n_it)
            if progress is not None:
                progress(done / float(max_iter))
            cur = _txf_normalise(Hm)
            if prev is not None and float(np.abs(cur - prev).sum(axis=1).max()) < _TXF_FIT_TOL:
                break
            prev = cur
    return _txf_normalise(Hm), done


def _txf_project_rows(indptr, indices, data, Ht, n_iter: int, tol: float, stop=None) -> np.ndarray:
    """EM for the factor counts of rows with at least one count each (CSR pieces), the
    loadings fixed. Ht: genes × k, columns sum to 1."""
    lens = np.diff(indptr)
    k = Ht.shape[1]
    off = indptr[:-1] - indptr[0]
    Hj = Ht[indices]                                         # nnz × k
    tot = np.add.reduceat(data, off).astype(np.float32)
    W = np.repeat((tot / k)[:, None], k, axis=1)
    denom = float(max(tot.sum(), 1e-30))
    for _ in range(int(n_iter)):
        if stop is not None and stop.is_set():
            break
        P = np.repeat(W, lens, axis=0)
        P *= Hj
        s = P.sum(axis=1)
        P *= (data / np.maximum(s, 1e-30))[:, None]
        Wn = np.add.reduceat(P, off, axis=0)
        moved = float(np.abs(Wn - W).sum()) / denom
        W = Wn
        if moved < tol:
            break
    return W


_txf_kernel_cache: dict = {}


def _txf_em_kernel():
    """The EM update compiled with numba (row by row, in parallel), or None when numba is
    not installed or _TXF_USE_NUMBA is off; the numpy implementation is used then."""
    if not _TXF_USE_NUMBA:
        return None
    if "fn" in _txf_kernel_cache:
        return _txf_kernel_cache["fn"]
    fn = None
    try:
        import numba

        @numba.njit(parallel=True, nogil=True)
        def _em(indptr, indices, data, Ht, n_iter, tol, W):
            n, k = len(indptr) - 1, Ht.shape[1]
            for i in numba.prange(n):
                a, b = indptr[i], indptr[i + 1]
                tot = 0.0
                for q in range(a, b):
                    tot += data[q]
                if tot <= 0.0:
                    continue
                w = np.full(k, tot / k)
                wn = np.empty(k)
                for _it in range(n_iter):
                    wn[:] = 0.0
                    for q in range(a, b):
                        j = indices[q]
                        s = 0.0
                        for f in range(k):
                            s += w[f] * Ht[j, f]
                        if s > 0.0:
                            x = data[q] / s
                            for f in range(k):
                                wn[f] += w[f] * Ht[j, f] * x
                    moved = 0.0
                    for f in range(k):
                        moved += abs(wn[f] - w[f])
                        w[f] = wn[f]
                    if moved < tol * tot:
                        break
                for f in range(k):
                    W[i, f] = w[f]

        fn = _em
    except Exception as exc:
        print(f"  Transcript factors: numba is not available ({exc}); using numpy", flush=True)
    _txf_kernel_cache["fn"] = fn
    return fn


def _txf_project(X, H, n_iter: int = _TXF_PROJ_ITER, tol: float = _TXF_PROJ_TOL, progress=None,
                 threads: int = _TXF_THREADS) -> np.ndarray:
    """Factor counts of every row of X for fixed loadings H (k × genes, rows sum to 1):
    W ≥ 0 minimising the Kullback-Leibler divergence of X from W·H, by the multiplicative
    (EM) update W ← W ∘ ((X ⊘ WH) Hᵀ). Every row of W sums to the row's count. With numba
    the rows are updated one by one in compiled code (each until it moves by less than
    *tol* of its count); without it, in numpy on chunks of about _TXF_CHUNK_NNZ non-zeros
    (each chunk until it moves by less than *tol*), a few chunks at a time."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    X = sp.csr_matrix(X, dtype=np.float32)
    n, k = X.shape[0], H.shape[0]
    W = np.zeros((n, k), dtype=np.float32)
    if not n or not X.nnz:
        return W
    Ht = np.ascontiguousarray(np.asarray(H, dtype=np.float32).T)
    kern = _txf_em_kernel()
    if kern is not None:
        import numba
        try:
            numba.set_num_threads(max(1, min(int(threads), numba.config.NUMBA_NUM_THREADS)))
        except Exception:
            pass
        ip = X.indptr.astype(np.int64)
        step = 100_000
        for a in range(0, n, step):
            jobs._job_check()
            b = min(n, a + step)
            kern(ip[a:b + 1] - ip[a], X.indices[ip[a]:ip[b]], X.data[ip[a]:ip[b]], Ht, int(n_iter), float(tol),
                 W[a:b])
            if progress is not None:
                progress(b / n)
        return W
    lens = np.diff(X.indptr)
    rows = np.flatnonzero(lens > 0)
    if len(rows) < n:                                         # empty rows keep 0
        X = X[rows]
    ip = X.indptr
    cuts = [0]
    while cuts[-1] < X.shape[0]:
        nxt = int(np.searchsorted(ip, ip[cuts[-1]] + _TXF_CHUNK_NNZ, side="right")) - 1
        cuts.append(min(X.shape[0], max(cuts[-1] + 1, nxt)))
    stop = threading.Event()
    out = np.zeros((X.shape[0], k), dtype=np.float32)

    def _one(a, b):
        return a, b, _txf_project_rows(ip[a:b + 1], X.indices[ip[a]:ip[b]], X.data[ip[a]:ip[b]], Ht, n_iter, tol, stop)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as ex:
        futs = [ex.submit(_one, a, b) for a, b in zip(cuts[:-1], cuts[1:])]
        try:
            for f in as_completed(futs):
                a, b, w = f.result()
                out[a:b] = w
                done += 1
                jobs._job_check()
                if progress is not None:
                    progress(done / len(futs))
        except BaseException:
            stop.set()
            for f in futs:
                f.cancel()
            raise
    if len(rows) < n:
        W[rows] = out
        return W
    return out


def _txf_match(H, R) -> tuple:
    """Best one-to-one matching of the rows of H to the rows of R by cosine similarity:
    (row of R matched to each row of H (-1: none), cosine of each match, full matrix)."""
    from scipy.optimize import linear_sum_assignment
    A = np.asarray(H, dtype=np.float64)
    B = np.asarray(R, dtype=np.float64)
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-30)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-30)
    C = A @ B.T
    ri, ci = linear_sum_assignment(-C)
    match = np.full(len(A), -1, dtype=np.int64)
    match[ri] = ci
    cos = np.where(match >= 0, C[np.arange(len(A)), np.clip(match, 0, None)], np.nan)
    return match, cos, C


def _txf_top_genes(H, genes, mass=None, n: int = _TXF_CSV_GENES) -> "pd.DataFrame":
    """The genes that characterise each factor. loading = the gene's share of the factor's
    transcripts; lift = loading ÷ the gene's share of all transcripts (factors weighted by
    *mass*, their share of the transcripts); the genes are ranked by loading × ln(lift),
    the gene's contribution to the divergence of the factor from the average profile.
    share_of_gene = the share of the gene's transcripts the model attributes to the factor."""
    H = np.asarray(H, dtype=np.float64)
    k = H.shape[0]
    w = np.full(k, 1.0 / k) if mass is None else np.asarray(mass, dtype=np.float64) / max(float(np.sum(mass)), 1e-30)
    avg = np.maximum(w @ H, 1e-30)
    rows = []
    for f in range(k):
        lift = H[f] / avg
        score = H[f] * np.log(np.maximum(lift, 1e-30))
        order = np.argsort(-score, kind="stable")[:int(n)]
        for rank, j in enumerate(order, start=1):
            rows.append({"factor": f, "rank": rank, "gene": genes[j], "loading": float(H[f, j]),
                         "lift": float(lift[j]), "share_of_gene": float(w[f] * H[f, j] / avg[j]),
                         "score": float(score[j])})
    return pd.DataFrame(rows, columns=["factor", "rank", "gene", "loading", "lift", "share_of_gene", "score"])


# ─── Cells against the bins ───────────────────────────────────────────────────
def _txf_lookup(keys_sorted, r, c) -> np.ndarray:
    """Row of each bin (r, c) in a sample's bin list (sorted keys), -1 when the bin has no
    transcript."""
    q = _txf_bin_key(r, c)
    if not len(keys_sorted):
        return np.full(len(q), -1, dtype=np.int64)
    pos = np.searchsorted(keys_sorted, q)
    pos[pos >= len(keys_sorted)] = 0
    return np.where(keys_sorted[pos] == q, pos, -1)


def _txf_cell_mixture(xy, area, smp: dict, bin_um: float, shape: str) -> np.ndarray:
    """cells × k: the factor counts of the bins under each cell, summed over the bin under
    its centroid and the bins under six points at _TXF_RING × the radius of a disc of the
    cell's area (*area* in µm², None: the centroid only). A cell over no transcript: 0."""
    xy = np.asarray(xy, dtype=np.float64)
    W = smp["W"]
    M = np.zeros((len(xy), W.shape[1]), dtype=np.float64)
    if not len(xy) or not len(W):
        return M
    keys = _txf_bin_key(smp["r"], smp["c"])
    pts = [(0.0, 0.0)]
    rad = None
    if area is not None:
        rad = np.sqrt(np.clip(np.nan_to_num(np.asarray(area, dtype=np.float64)), 0.0, None) / np.pi) * _TXF_RING
        pts += [(np.cos(a), np.sin(a)) for a in np.arange(6) * np.pi / 3.0]
    for dx, dy in pts:
        px = xy[:, 0] + (dx * rad if rad is not None else 0.0)
        py = xy[:, 1] + (dy * rad if rad is not None else 0.0)
        idx = _txf_lookup(keys, *_txf_bin_index(px, py, bin_um, shape))
        ok = idx >= 0
        M[ok] += W[idx[ok]]
    return M


def _txf_entropy(M) -> tuple:
    """(dominant column, its share, entropy ÷ ln k) of the rows of a non-negative matrix;
    -1 / NaN for rows that sum to 0."""
    M = np.asarray(M, dtype=np.float64)
    k = M.shape[1]
    tot = M.sum(axis=1)
    ok = tot > 0
    P = np.where(ok[:, None], M / np.maximum(tot, 1e-30)[:, None], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(np.where(P > 0, P * np.log(P), 0.0)).sum(axis=1) / (np.log(k) if k > 1 else 1.0)
    dom = np.where(ok, P.argmax(axis=1), -1)
    return dom, np.where(ok, P.max(axis=1), np.nan), np.where(ok, ent, np.nan)


def _txf_col(name: str) -> str:
    return "txf_share_" + re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_")


def _txf_cols(names) -> list:
    """One per-cell column name per factor, unique: names that differ only in punctuation
    ("CD4+ T", "CD4- T") sanitise to the same text, and the later ones get _2, _3, …"""
    out, seen = [], {}
    for nm in names:
        base = _txf_col(nm)
        col, k = base, 1
        while col in seen:
            k += 1
            col = f"{base}_{k}"
        seen[col] = True
        out.append(col)
    return out


def _txf_area_of(df):
    for col in ("cell_area", "area"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    return None


# ─── The job ──────────────────────────────────────────────────────────────────
def _txf_params(params=None) -> dict:
    p = {"k": _TXF_K, "bin": _TXF_BIN_UM, "shape": "square", "min_tx": _TXF_MIN_TX, "max_fit": _TXF_MAX_FIT,
         "seed": _TXF_SEED, "fit_iter": _TXF_FIT_ITER, "proj_iter": _TXF_PROJ_ITER}
    p.update({k: v for k, v in (params or {}).items() if v is not None and k in p})
    p["k"] = int(max(2, min(_TXF_MAX_K, int(p["k"]))))
    p["bin"] = float(min(500.0, max(1.0, float(p["bin"]))))
    p["shape"] = "hex" if p["shape"] == "hex" else "square"
    p["min_tx"] = int(max(1, int(p["min_tx"])))
    p["max_fit"] = int(max(200, int(p["max_fit"])))
    p["seed"] = int(p["seed"])
    return p


def _txf_fit_file(method, p, genes, keys, ref_hash="") -> str:
    blob = json.dumps({"method": method, "p": p, "genes": _txf_genes_hash(genes), "samples": list(keys),
                       "ref": ref_hash, "filter": analysis._DENSITY_FILTER_VERSION, "v": 1}, sort_keys=True)
    return os.path.join(_txf_cache_dir(), f"fit_{hashlib.md5(blob.encode()).hexdigest()[:12]}.npz")


def _txf_fit_save(path, H, names, smps, info) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrs = {"H": H, "names": np.array(names, dtype=object).astype(str), "info": np.array(json.dumps(info))}
    for i, s in enumerate(smps):
        arrs.update({f"r{i}": s["r"], f"c{i}": s["c"], f"W{i}": s["W"], f"n{i}": s["n_tx"]})
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def _txf_fit_load(path, n_samples: int):
    """(H, names, [{r, c, W, n_tx}], info) of a cached fit, or None."""
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            smps = [{"r": z[f"r{i}"], "c": z[f"c{i}"], "W": z[f"W{i}"], "n_tx": z[f"n{i}"]} for i in range(n_samples)]
            return z["H"], [str(x) for x in z["names"]], smps, json.loads(str(z["info"]))
    except Exception as exc:
        print(f"  Transcript factors: unreadable fit cache {os.path.basename(path)} ({exc}); refitting", flush=True)
        return None


def _txf_reference(seg_source, tables, labels, genes, use_corrected, min_cells=None) -> tuple:
    """Reference mode: (H, type names, types left out) - the mean profile of every cell
    type with at least *min_cells* (_TXF_REF_MIN_CELLS) cells, as a gene distribution over
    *genes*."""
    min_cells = _TXF_REF_MIN_CELLS if min_cells is None else int(min_cells)
    samples = analysis._an_samples(seg_source, use_corrected)
    by_key = {s["key"]: s for s in samples}
    use = [(by_key[t["key"]], lab) for t, lab in zip(tables, labels)
           if t["key"] in by_key and lab is not None and len(by_key[t["key"]]["df"]) == len(t["df"])]
    if not use:
        raise ValueError("The reference mode needs the count matrix and labels of the active view.")
    ss, ll = [u[0] for u in use], [u[1] for u in use]
    common = set(analysis._an_common_genes(ss))
    gsel = [g for g in genes if g in common]
    if len(gsel) < 2:
        raise ValueError("The count matrix shares fewer than 2 genes with the transcript file.")
    prof = controls._type_profiles(ss, ll, gsel)
    keep = [i for i, t in enumerate(prof["types"]) if prof["n"][i] >= min_cells and prof["counts"][i].sum() > 0]
    left = [prof["types"][i] for i in range(len(prof["types"])) if i not in keep]
    if len(keep) < 2:
        raise ValueError(f"Fewer than 2 cell types have at least {min_cells} cells: nothing to project onto.")
    H = np.zeros((len(keep), len(genes)), dtype=np.float64)
    pos = {g: j for j, g in enumerate(genes)}
    cols = np.array([pos[g] for g in gsel], dtype=np.int64)
    H[:, cols] = prof["counts"][keep]
    return _txf_normalise(H), [str(prof["types"][i]) for i in keep], left


def _txf_type_profiles(seg_source, tables, labels, genes, use_corrected):
    """(types, types × genes mean profiles as gene distributions) of the labelled cells of
    the view, or None when there is no count matrix / label."""
    try:
        H, names, _left = _txf_reference(seg_source, tables, labels, genes, use_corrected, min_cells=_TXF_CMP_MIN_CELLS)
        return names, H
    except Exception:
        return None


def _txf_crosstab(dom_names, types) -> "pd.DataFrame":
    """Cells by dominant factor (rows) and cell type (columns)."""
    d = pd.Series(dom_names, dtype=object)
    t = pd.Series(types, dtype=object)
    ok = d.notna() & t.notna()
    if not ok.any():
        return pd.DataFrame()
    return pd.crosstab(d[ok].astype(str), t[ok].astype(str))


def _txf_agreement(ct: "pd.DataFrame") -> dict:
    """How well the dominant factor tells the cell type: purity (share of the cells whose
    type is the commonest type of their dominant factor), adjusted Rand index, and the
    share of cells whose dominant factor carries the name of their type (reference mode)."""
    if ct is None or not ct.size:
        return {}
    N = ct.to_numpy(dtype=np.float64)
    n = N.sum()
    comb = lambda v: v * (v - 1) / 2.0
    a, b = comb(N.sum(axis=1)).sum(), comb(N.sum(axis=0)).sum()
    exp = a * b / comb(n) if n > 1 else 0.0
    ari = (comb(N).sum() - exp) / (0.5 * (a + b) - exp) if (0.5 * (a + b) - exp) != 0 else np.nan
    same = float(sum(ct.loc[x, x] for x in ct.index if x in ct.columns))
    return {"n": int(n), "purity": float(N.max(axis=1).sum() / n), "ari": float(ari), "same_name": same / n}


def _run_txfactor(seg_source, cluster_method, label_kind, method="nmf", params=None,
                  use_corrected: bool = False) -> None:
    """Background job: bin the transcripts of every sample, fit (or take) the gene loadings,
    project every bin, and compare with the cells of the active view."""
    _gen = jobs._dataset_gen
    p = _txf_params(params)
    method = method if method in _TXF_METHODS else "nmf"

    def _set(status, message, result=None):
        with _txf_lock:
            if jobs._stale_job(_gen, status, message):
                return
            _txf_state["status"], _txf_state["message"] = status, message
            if result is not None:
                _txf_state["result"] = result
            if status in ("done", "error"):
                _txf_state["version"] += 1
        if status == "done":
            provenance._prov_record("txfactor", seg_source=seg_source, corrected=use_corrected, params={"method": method, **(params or {})}, labels=(label_kind, cluster_method), message=message, result=result)

    try:
        _t0 = _time.time()
        _set("running", "Collecting samples…")
        analysis._an_require_seg(seg_source)
        tables = stats._view_tables(seg_source)
        seg_tag = controls._seg_tag(seg_source)       # the run these tables belong to (checked at every write)
        usable = [i for i, t in enumerate(tables) if t["ds"].get("data_dir")]
        skipped = [tables[i]["label"] for i in range(len(tables)) if i not in usable]
        if not usable:
            _set("error", "Transcript factors need the Xenium output directory (transcripts.parquet or "
                          "transcripts.zarr.zip); " + ("this dataset was" if len(tables) == 1 else "these samples were")
                          + f" {dataview._origin_phrase(tables[0]['ds'])}, which has cells but no transcripts.")
            return
        genes = analysis._ds_panel_genes(tables[usable[0]]["ds"])
        for i in usable[1:]:
            have = set(analysis._ds_panel_genes(tables[i]["ds"]))
            genes = [g for g in genes if g in have]
        if len(genes) < 2:
            _set("error", "The samples share fewer than 2 panel genes.")
            return
        labels, cat = (None, "")
        if label_kind:
            try:
                labels, cat = stats._comp_labels(label_kind, tables, seg_source, cluster_method)
            except ValueError:
                labels = None
            if labels is not None and all(l is None for l in labels):
                labels = None
        H_ref, names, left = None, None, []
        ref_hash = ""
        if method == "reference":
            if labels is None:
                _set("error", "The reference mode needs cell types: pick an annotation (or clusters) under "
                              "'Cell types from'.")
                return
            H_ref, names, left = _txf_reference(seg_source, tables, labels, genes, use_corrected)
            ref_hash = hashlib.md5(H_ref.tobytes() + "|".join(names).encode()).hexdigest()[:12]
            p = {**p, "k": len(names)}
        keys = [tables[i]["key"] for i in usable]
        fit_path = _txf_fit_file(method, p, genes, keys, ref_hash)
        cached = _txf_fit_load(fit_path, len(usable))
        n_un = len(usable)
        scan_s = fit_s = proj_s = 0.0
        if cached is not None and cached[0].shape[1] == len(genes):
            H, names, parts, info = cached
            from_cache = True
            jobs._job_progress("Loaded the factors from the cache.", 0.9)
        else:
            from_cache = False
            counts = []
            for k, i in enumerate(usable):
                t = tables[i]
                _set("running", f"Binning the transcripts of {t['label']} ({p['bin']:g} µm {p['shape']} bins)…")
                _ts = _time.time()
                c = _txf_counts(t["ds"], genes, p["bin"], p["shape"],
                                progress=lambda f, _k=k: jobs._job_progress(fraction=0.55 * (_k + f) / n_un))
                scan_s += _time.time() - _ts
                jobs._check_dataset_gen(_gen)
                c["n_tx"] = np.asarray(c["X"].sum(axis=1)).ravel().astype(np.float32)
                counts.append(c)
            n_bins = int(sum(len(c["r"]) for c in counts))
            if not n_bins:
                _set("error", "No transcript passes the filters (Q ≥ 20, gene features).")
                return
            info = {"n_fit": 0, "n_iter": 0, "n_eligible": 0, "counts_cached": [bool(c["cached"]) for c in counts]}
            if method == "reference":
                H = H_ref
            else:
                # bins with enough transcripts, pooled over the samples; a random subsample of them
                elig = [np.flatnonzero(c["n_tx"] >= p["min_tx"]) for c in counts]
                n_el = int(sum(len(e) for e in elig))
                if n_el < 5 * p["k"]:
                    _set("error", f"Only {n_el:,} bins hold at least {p['min_tx']} transcripts: too few to fit "
                                  f"{p['k']} factors. Lower the minimum, enlarge the bins or lower K.")
                    return
                rng = np.random.default_rng(p["seed"])
                pick = np.sort(rng.choice(n_el, size=min(n_el, p["max_fit"]), replace=False))
                offs = np.cumsum([0] + [len(e) for e in elig])
                Xf = sp.vstack([counts[j]["X"][elig[j][pick[(pick >= offs[j]) & (pick < offs[j + 1])] - offs[j]]]
                                for j in range(len(counts))], format="csr")
                _set("running", f"Fitting {p['k']} factors ({_TXF_METHODS[method]}) on {Xf.shape[0]:,} of "
                                f"{n_el:,} bins × {len(genes):,} genes…")
                _ts = _time.time()
                H, n_it = _txf_fit(Xf, p["k"], method, p["seed"], p["fit_iter"],
                                   progress=lambda f: jobs._job_progress(fraction=0.55 + 0.2 * f))
                fit_s = _time.time() - _ts
                info.update({"n_fit": int(Xf.shape[0]), "n_iter": int(n_it), "n_eligible": n_el})
                del Xf
            parts = []
            _ts = _time.time()
            for k, c in enumerate(counts):
                _set("running", f"Projecting the {len(c['r']):,} bins of {tables[usable[k]]['label']} onto the factors…")
                W = _txf_project(c["X"], H, p["proj_iter"],
                                 progress=lambda f, _k=k: jobs._job_progress(fraction=0.75 + 0.2 * (_k + f) / n_un))
                parts.append({"r": c["r"], "c": c["c"], "W": W, "n_tx": c["n_tx"]})
                c["X"] = None
            proj_s = _time.time() - _ts
            del counts
            if method != "reference":
                # factors in the order of their share of the transcripts
                order = np.argsort(-sum(s["W"].sum(axis=0, dtype=np.float64) for s in parts), kind="stable")
                H = H[order]
                for s in parts:
                    s["W"] = np.ascontiguousarray(s["W"][:, order])
                names = [f"F{j + 1}" for j in range(H.shape[0])]
            info.update({"scan_s": scan_s, "fit_s": fit_s, "proj_s": proj_s})
        jobs._check_dataset_gen(_gen)
        jobs._job_shield("Writing the transcript factors…")
        if not from_cache:
            try:
                _txf_fit_save(fit_path, H, names, parts, info)
            except Exception as exc:
                print(f"  Transcript factors: the result was not cached ({exc})", flush=True)
        K = H.shape[0]
        mass = sum(s["W"].sum(axis=0, dtype=np.float64) for s in parts)
        smps = []
        for k, i in enumerate(usable):
            t, s = tables[i], parts[k]
            smps.append({**s, "key": t["key"], "label": t["label"], "name": t["name"],
                         "n_bins": int(len(s["r"])), "total": float(np.sum(s["n_tx"], dtype=np.float64)),
                         "n_dom": int((s["n_tx"] >= p["min_tx"]).sum())})
        # the cells of the view against the bins
        dom_all, type_all = [], []
        share_cols = _txf_cols(names)
        n_cells = n_under = 0
        mixed_all = []
        for ti, t in enumerate(tables):
            s = next((x for x in smps if x["key"] == t["key"]), None)
            df = t["df"]
            n = len(df)
            n_cells += n
            with controls._ctl_columns_lock:
                store = controls._ctl_columns.get((t["key"], controls._ctl_check_tag(seg_source, seg_tag)))
                for old in [c for c in (store or {}) if str(c).startswith("txf_")]:
                    del store[old]
            if s is None or not n:
                continue
            M = _txf_cell_mixture(stats._xy_of(df), _txf_area_of(df), s, p["bin"], p["shape"])
            dom, top, ent = _txf_entropy(M)
            tot = M.sum(axis=1)
            dn = np.array([names[d] if d >= 0 else None for d in dom], dtype=object)
            cols = {"txf_dominant": dn, "txf_top_share": top.astype(np.float32), "txf_mixed": ent.astype(np.float32)}
            share = np.where(tot[:, None] > 0, M / np.maximum(tot, 1e-30)[:, None], np.nan)
            for j, nm in enumerate(names):
                cols[share_cols[j]] = share[:, j].astype(np.float32)
            controls._ctl_put(t, seg_source, cols, tag=seg_tag)
            n_under += int((dom >= 0).sum())
            mixed_all.append(ent[np.isfinite(ent)])
            dom_all.append(dn)
            type_all.append(np.full(n, None, dtype=object) if labels is None or labels[ti] is None
                            else pd.Series(labels[ti], dtype=object).to_numpy())
        ct = (_txf_crosstab(np.concatenate(dom_all), np.concatenate(type_all))
              if dom_all and labels is not None else pd.DataFrame())
        if ct.size:
            ct = ct.reindex(index=[nm for nm in names if nm in ct.index])
        agree = _txf_agreement(ct)
        # which annotated type each factor's loadings resemble
        best = [("", np.nan)] * K
        tp = _txf_type_profiles(seg_source, tables, labels, genes, use_corrected) if labels is not None else None
        if tp is not None:
            _m, _c, C = _txf_match(H, tp[1])
            best = [(tp[0][int(C[f].argmax())], float(C[f].max())) for f in range(K)]
        top = _txf_top_genes(H, genes, mass)
        top["factor_name"] = [names[f] for f in top["factor"]]
        frows = []
        for f in range(K):
            tg = top[top["factor"] == f].head(_TXF_TOP_GENES)["gene"].tolist()
            row = {"factor": names[f], "share_tx": float(mass[f] / max(mass.sum(), 1e-30)),
                   "n_bins_dominant": int(sum(int(((s["W"].argmax(axis=1) == f) & (s["n_tx"] >= p["min_tx"])).sum())
                                              for s in smps)),
                   # every cell the factor dominates, labelled or not (the cross-tab holds
                   # the labelled ones only: n_cells_labelled, the base of top_type_share)
                   "n_cells_dominant": int(sum(int((d == names[f]).sum()) for d in dom_all)),
                   "n_cells_labelled": int(ct.loc[names[f]].sum()) if ct.size and names[f] in ct.index else 0,
                   "best_type": best[f][0], "cosine": best[f][1],
                   "top_type_of_cells": "", "top_type_share": np.nan, "top_genes": ", ".join(tg)}
            if ct.size and names[f] in ct.index and ct.loc[names[f]].sum() > 0:
                r = ct.loc[names[f]]
                row["top_type_of_cells"], row["top_type_share"] = str(r.idxmax()), float(r.max() / r.sum())
            frows.append(row)
        res = {"gen": _gen, "params": p, "method": method, "genes": genes, "H": H, "names": names,
               "samples": smps, "skipped": skipped, "left_out_types": left, "info": info, "top": top,
               "factors": pd.DataFrame(frows), "crosstab": ct, "agreement": agree, "category": cat,
               "label_kind": label_kind, "seg_source": seg_source or "xenium",
               "seg_tag": seg_tag, "from_cache": from_cache, "share_cols": share_cols,
               "corrected": bool(use_corrected), "seconds": _time.time() - _t0, "uid": f"{_gen}-{_time.time():.3f}"}
        with _txf_mem_lock:
            _txf_mem.clear()
        n_bins = int(sum(s["n_bins"] for s in smps))
        n_tx = float(sum(s["total"] for s in smps))
        mixed = np.concatenate(mixed_all) if mixed_all else np.zeros(0)
        msg = (f"{K} factors ({_TXF_METHODS[method]}) on {n_bins:,} {p['bin']:g} µm {p['shape']} bins holding "
               f"{n_tx:,.0f} gene transcripts (Q ≥ 20) of {len(genes):,} genes in {len(smps)} sample(s)"
               + ("" if method == "reference" else
                  f"; fitted on {info.get('n_fit', 0):,} of the {info.get('n_eligible', 0):,} bins with ≥ "
                  f"{p['min_tx']} transcripts ({info.get('n_iter', 0)} "
                  f"{'minibatches' if method == 'lda' else 'updates'}), seed {p['seed']}")
               + (f"; loaded from the cache ({res['seconds']:.1f} s)." if from_cache else
                  f" ({res['seconds']:.1f} s: binning {info.get('scan_s', 0):.0f} s"
                  f"{' (cached)' if all(info.get('counts_cached') or [False]) else ''}, fit "
                  f"{info.get('fit_s', 0):.0f} s, projection {info.get('proj_s', 0):.0f} s).")
               + "\nA simplified transcript-binning factorisation in the spirit of FICTURE; not that tool.")
        if method == "reference":
            msg += (f"\nReference components: the mean profiles of {K} cell types ({cat}; "
                    f"{'SPLIT-corrected' if use_corrected else 'original'} counts) - not fitted."
                    + (f" Left out (fewer than {_TXF_REF_MIN_CELLS} cells): {', '.join(left[:8])}"
                       f"{'…' if len(left) > 8 else ''}." if left else ""))
        msg += (f"\nCells ({res['seg_source']}): {n_under:,} of {n_cells:,} lie over a bin with transcripts; "
                + (f"median mixed score {np.median(mixed):.2f}, {100 * float((mixed > 0.5).mean()):.1f} % above 0.5."
                   if len(mixed) else "no mixture."))
        if agree:
            msg += (f"\nDominant factor × {cat}: {agree['n']:,} labelled cells; purity {agree['purity']:.2f} "
                    f"(share of cells whose type is the commonest type of their dominant factor), adjusted Rand "
                    f"index {agree['ari']:.2f}"
                    + (f", {100 * agree['same_name']:.1f} % of cells have their own type as the dominant factor"
                       if method == "reference" else "") + ".")
        elif label_kind:
            msg += "\nNo cross-tab: the view has no labels of the chosen kind."
        if tp is not None and method != "reference":
            cs = np.array([b[1] for b in best], dtype=np.float64)
            msg += (f"\nLoadings against the mean profiles of the annotated types: best cosine per factor median "
                    f"{np.nanmedian(cs):.2f} (range {np.nanmin(cs):.2f} - {np.nanmax(cs):.2f}).")
        if skipped:
            msg += f"\nLeft out (loaded from a .zarr, no transcript file): {', '.join(skipped)}."
        msg += ("\nMap layer: choose it below (drawn under the cells in both renderers). Color By: "
                "'Factors: mixed score', 'Factors: share of …'.")
        _set("done", msg, result=res)
        print(f"  Transcript factors: {msg}", flush=True)
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


# ─── Color By ─────────────────────────────────────────────────────────────────
def _txf_color(name, seg_source="xenium", use_corrected: bool = False):
    """Colour provider (analysis._view_color_providers): the mixed score, the share of the
    dominant factor or the share of one factor under every cell of the view."""
    res = _txf_result()
    if res is None or res.get("seg_tag") != controls._seg_tag(seg_source):
        return None
    what = str(name)[len(_TXF_COLOR):]
    if what == "mixed":
        col, label = "txf_mixed", "Factors: mixed score (entropy of the factor mixture under the cell ÷ ln K)"
    elif what == "top":
        col, label = "txf_top_share", "Factors: share of the dominant factor under the cell"
    elif what.startswith("share:"):
        try:
            j = int(what.split(":", 1)[1])
            nm = res["names"][j]
        except (ValueError, IndexError):
            return None
        col = (res.get("share_cols") or _txf_cols(res["names"]))[j]     # the column of THIS factor
        label = f"Factors: share of {nm} under the cell"
    else:
        return None
    parts, have = [], False
    for t in stats._view_tables(seg_source):
        v = controls._ctl_get(t, seg_source, col)
        if v is None:
            parts.append(np.full(len(t["df"]), np.nan, dtype=np.float32))
        else:
            have = True
            parts.append(pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(dtype=np.float32))
    if not have:
        return None
    return (np.concatenate(parts) if len(parts) > 1 else parts[0]), (0.0, 1.0), label


def _txf_color_options() -> list:
    res = _txf_result()
    if res is None:
        return []
    return ([{"label": "Factors: mixed score (entropy under the cell)", "value": f"geneset:{_TXF_COLOR}mixed"},
             {"label": "Factors: share of the dominant factor", "value": f"geneset:{_TXF_COLOR}top"}]
            + [{"label": f"Factors: share of {nm}", "value": f"geneset:{_TXF_COLOR}share:{j}"}
               for j, nm in enumerate(res["names"])])


analysis._view_color_providers[_TXF_COLOR] = _txf_color
analysis._view_color_options.append(_txf_color_options)


# ─── Map layer (through the transcript-density heat layer) ────────────────────
def _txf_palette(k: int) -> list:
    return [core.CLUSTER_COLORS[i % len(core.CLUSTER_COLORS)] for i in range(int(k))]


def _txf_layer_values(res, smp, what):
    """(value per bin, categorical?) of a layer: 'dominant' -> 1 + index of the largest
    factor for bins with at least min_tx transcripts (0: none); a factor index -> the
    transcripts of the bin attributed to that factor."""
    if what == "dominant":
        v = (smp["W"].argmax(axis=1) + 1).astype(np.float32) if len(smp["W"]) else np.zeros(0, np.float32)
        v[smp["n_tx"] < res["params"]["min_tx"]] = 0.0
        return v, True
    return np.asarray(smp["W"][:, int(what)], dtype=np.float32), False


def _txf_layer_grid(gene, ds=None):
    """Grid dict of a factor layer for the density plumbing (analysis._density_grid): {grid
    (float32, rows = y), ix0, iy0, bin (µm per grid cell), total, vmax} and, for the
    dominant factor, palette / names. Square bins are the grid; hexagons are drawn on a
    finer square raster (every pixel takes the value of the hexagon its centre lies in).
    None without a result for this sample."""
    res = _txf_result()
    if res is None:
        return None
    what = str(gene)[len(_TXF_GENE):]
    K = len(res["names"])
    if what != "dominant":
        try:
            what = int(what)
        except ValueError:
            return None
        if not 0 <= what < K:
            return None
    key = dataview._ds_key(core.DATA if ds is None else ds)
    smp = next((s for s in res["samples"] if s["key"] == key), None)
    if smp is None:
        return None
    mkey = (res["uid"], key, what)
    with _txf_mem_lock:
        hit = _txf_mem.get(mkey)
        if hit is not None:
            _txf_mem.move_to_end(mkey)
            return hit
    p = res["params"]
    vals, cat = _txf_layer_values(res, smp, what)
    r, c = smp["r"].astype(np.int64), smp["c"].astype(np.int64)
    out = {"bin": p["bin"], "total": int(round(float(vals.sum()))) if not cat else int(smp["total"]),
           "ix0": 0, "iy0": 0, "grid": np.zeros((0, 0), dtype=np.float32), "vmax": 1.0}
    if len(r):
        if p["shape"] != "hex":
            y0, x0 = int(r.min()), int(c.min())
            h, w = int(r.max()) - y0 + 1, int(c.max()) - x0 + 1
            grid = np.zeros((h, w), dtype=np.float32)
            grid[r - y0, c - x0] = vals
            out.update({"grid": grid, "ix0": x0, "iy0": y0})
        else:
            cx, cy = _txf_bin_centres(r, c, p["bin"], "hex")
            lo_x, hi_x = cx.min() - p["bin"], cx.max() + p["bin"]
            lo_y, hi_y = cy.min() - p["bin"], cy.max() + p["bin"]
            px = p["bin"] / 4.0
            for div in (4.0, 3.0, 2.0, 1.0):
                px = p["bin"] / div
                if ((hi_x - lo_x) / px) * ((hi_y - lo_y) / px) <= _TXF_RASTER_MAX:
                    break
            x0, y0 = int(np.floor(lo_x / px)), int(np.floor(lo_y / px))
            w, h = int(np.ceil(hi_x / px)) - x0, int(np.ceil(hi_y / px)) - y0
            pk = (res["uid"], key, "pixmap")
            with _txf_mem_lock:
                idx = _txf_mem.get(pk)
            if idx is None:
                keys = _txf_bin_key(r, c)
                idx = np.empty((h, w), dtype=np.int32)
                xs = (np.arange(w) + x0 + 0.5) * px
                for a in range(0, h, 512):
                    ys = (np.arange(a, min(h, a + 512)) + y0 + 0.5) * px
                    rr, cc = _txf_bin_index(np.tile(xs, len(ys)), np.repeat(ys, w), p["bin"], "hex")
                    idx[a:a + len(ys)] = _txf_lookup(keys, rr, cc).reshape(len(ys), w)
                with _txf_mem_lock:
                    _txf_mem[pk] = idx
            grid = np.where(idx >= 0, vals[np.clip(idx, 0, None)], 0.0).astype(np.float32)
            out.update({"grid": grid, "ix0": x0, "iy0": y0, "bin": px})
        if cat:
            out["vmax"] = float(K)
        else:
            nz = vals[vals > 0]
            out["vmax"] = float(max(1e-6, np.percentile(nz, 99.5))) if len(nz) else 1.0
    if cat:
        out.update({"palette": _txf_palette(K), "names": list(res["names"])})
    with _txf_mem_lock:
        _txf_mem[mkey] = out
        keep = 3 * (1 + len(dataview.EXTRA_DATASETS)) + 2
        while len(_txf_mem) > keep:
            _txf_mem.popitem(last=False)
    return out


def _txf_layer_label(gene, bin_um=None) -> str:
    res = _txf_result()
    what = str(gene)[len(_TXF_GENE):]
    if res is None:
        return "Transcript factors"
    p = res["params"]
    unit = f"{p['bin']:g} µm {'hexagon' if p['shape'] == 'hex' else 'bin'}"
    if what == "dominant":
        return f"Dominant factor per {unit} (≥ {p['min_tx']} transcripts)"
    try:
        return f"Factor {res['names'][int(what)]}: transcripts per {unit}"
    except (ValueError, IndexError):
        return "Transcript factors"


def _txf_layer_options() -> list:
    res = _txf_result()
    if res is None:
        return []
    return ([{"label": "Factors: dominant factor per bin", "value": f"{_TXF_GENE}dominant"}]
            + [{"label": f"Factors: {nm} (transcripts per bin)", "value": f"{_TXF_GENE}{j}"}
               for j, nm in enumerate(res["names"])])


def _txf_layer_hint(gene) -> str:
    res = _txf_result()
    if res is None:
        return "No transcript factors computed yet (Transcript Factors tab)."
    g = _txf_layer_grid(gene)
    lab = _txf_layer_label(gene)
    if str(gene).endswith("dominant"):
        return (f"{lab}: {sum(s['n_dom'] for s in res['samples']):,} bins coloured by factor; the legend lists "
                f"the factors. Computed in the Transcript Factors tab.")
    return (f"{lab}; colour scale 0 – {(g or {}).get('vmax', 1.0):.3g} (99.5th percentile; set 'Colour max' to "
            f"change). Computed in the Transcript Factors tab.")


analysis._density_virtual_providers[_TXF_GENE] = {
    "grid": _txf_layer_grid, "label": _txf_layer_label, "options": _txf_layer_options, "hint": _txf_layer_hint}


# ─── Figure, tables, CSV ──────────────────────────────────────────────────────
def make_txf_fig(width=None, height=None):
    """Share of the cells of every cell type (rows) by dominant factor (columns); without
    labels, the share of the transcripts of every factor."""
    res = _txf_result()
    if not res:
        return figures._make_placeholder_fig("No transcript factors computed yet.")
    ct = res["crosstab"]
    if ct is None or not ct.size:
        f = res["factors"]
        fig = go.Figure(go.Bar(x=f["factor"], y=f["share_tx"], marker_color=_txf_palette(len(f)),
                               hovertemplate="%{x}<br>%{y:.1%} of the transcripts<extra></extra>"))
        fig.update_yaxes(title="share of transcripts", tickformat=".0%")
        return controls._fig_layout(fig, "Share of the gene transcripts per factor (no cell labels to compare with)",
                                    width, height)
    N = ct.to_numpy(dtype=np.float64).T                       # types × factors
    share = N / np.maximum(N.sum(axis=1, keepdims=True), 1.0)
    order = np.lexsort((-share.max(axis=1), share.argmax(axis=1)))
    types = [str(ct.columns[i]) for i in order]
    fig = go.Figure(go.Heatmap(z=share[order], x=list(ct.index), y=types, customdata=N[order], zmin=0, zmax=1,
                               colorscale="Viridis", colorbar=dict(title="share of<br>the type's cells", thickness=12),
                               hovertemplate="%{y} · %{x}<br>%{z:.1%} of the type's cells (%{customdata:.0f})"
                                             "<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    a = res["agreement"]
    return controls._fig_layout(fig, f"Cells of each {res['category']} by dominant factor under the cell "
                                     f"(purity {a.get('purity', np.nan):.2f}, ARI {a.get('ari', np.nan):.2f})",
                                width, height, margin=dict(l=140, r=10, t=40, b=70))


_TXF_COLS = [("factor", "Factor"), ("share_tx", "Share of tx"), ("n_bins_dominant", "Bins"),
             ("n_cells_dominant", "Cells"), ("n_cells_labelled", "Labelled cells"), ("best_type", "Most similar type"), ("cosine", "Cosine"),
             ("top_type_of_cells", "Commonest type of its cells"), ("top_type_share", "Share"),
             ("top_genes", "Top genes")]


def _txf_records(res) -> list:
    if not res:
        return []
    out = pd.DataFrame({c: res["factors"][c] for c, _n in _TXF_COLS})
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda v: None if not np.isfinite(v) else float(f"{v:.3g}"))
    return out.astype(object).where(out.notna(), None).to_dict("records")


def _txf_csv(res) -> str:
    p, info = res["params"], res["info"]
    lines = [f"# Transcript factors: a simplified transcript-binning factorisation in the spirit of FICTURE "
             f"(not that tool); method: {_TXF_METHODS[res['method']]}; K = {len(res['names'])}",
             f"# bins: {p['bin']:g} um {p['shape']}; transcripts: Q >= 20, gene features; {len(res['genes'])} genes; "
             f"fitted on {info.get('n_fit', 0)} of {info.get('n_eligible', 0)} bins with >= {p['min_tx']} "
             f"transcripts; seed {p['seed']}; samples: {', '.join(s['label'] for s in res['samples'])}",
             "# loading = gene's share of the factor's transcripts; lift = loading / gene's share of all "
             "transcripts; share_of_gene = share of the gene's transcripts attributed to the factor; genes ranked "
             "by loading x ln(lift)",
             f"# cells: segmentation {res['seg_source']}; labels: {res['category'] or 'none'}"]
    a = res.get("agreement") or {}
    if a:
        lines.append(f"# dominant factor x {res['category']}: n = {a['n']}; purity {a['purity']:.4g}; adjusted "
                     f"Rand index {a['ari']:.4g}")
    top = res["top"][["factor_name", "rank", "gene", "loading", "lift", "share_of_gene", "score"]]
    out = ("\n".join(lines) + "\n# factors\n" + res["factors"].to_csv(index=False)
           + "\n# top genes per factor\n" + top.to_csv(index=False))
    if res["crosstab"] is not None and res["crosstab"].size:
        out += "\n# cells by dominant factor (rows) and label (columns)\n" + res["crosstab"].to_csv()
    return provenance._prov_csv_stamp(res, (out))


# ─── Layout ───────────────────────────────────────────────────────────────────
def _txf_tab():
    S, L = analysis._AN_ROW_STYLE, analysis._AN_LABEL_STYLE
    num = controls._num
    _text = ("factor", "best_type", "top_type_of_cells", "top_genes")
    cols = [{"id": c, "name": n, "type": "text" if c in _text else "numeric"} for c, n in _TXF_COLS]
    return dbc.Tab(label="Transcript Factors", tab_id="tab-txf",
                   label_style={"fontSize": "12px", "padding": "4px 12px"},
                   children=[html.Div([
        html.Div([
            html.Span("Method:", style=L),
            dcc.Dropdown(id="txf-method", options=[{"label": v, "value": k} for k, v in _TXF_METHODS.items()],
                         value="nmf", clearable=False, style={"width": "210px", "fontSize": "11px"}),
            html.Span("K:", style=L), num(id="txf-k", value=_TXF_K, min=2, max=_TXF_MAX_K, step=1),
            html.Span(style={"flex": "1"}),
            dbc.Button("Run", id="txf-run-btn", color="primary", size="sm", n_clicks=0,
                       style={"fontSize": "11px", "padding": "2px 12px"}),
        ], style=S),
        ui._options(html.Div([
            html.Span("Bin (µm):", style=L), num(id="txf-bin", value=_TXF_BIN_UM, min=2, max=200, step=1),
            dcc.Dropdown(id="txf-shape", options=[{"label": "square", "value": "square"},
                                                  {"label": "hexagon", "value": "hex"}],
                         value="square", clearable=False, style={"width": "100px", "fontSize": "11px"}),
            html.Span("≥ transcripts per bin:", style=L),
            num(id="txf-mintx", value=_TXF_MIN_TX, min=1, max=10000, step=1),
            html.Span("Fit on ≤ bins:", style=L),
            num(id="txf-maxfit", value=_TXF_MAX_FIT, min=200, max=2_000_000, step=100, width="76px"),
            html.Span("Seed:", style=L), num(id="txf-seed", value=_TXF_SEED, min=0, max=10 ** 6, step=1),
        ], style=S), summary=f"Options ({_TXF_BIN_UM:g} µm square bins, ≥ {_TXF_MIN_TX} transcripts per bin, "
                             f"fit on ≤ {_TXF_MAX_FIT:,} bins, seed {_TXF_SEED})"),
        html.Div([
            html.Span("Cell types from:", style=L),
            dcc.Dropdown(id="txf-labels", options=[{"label": "Cluster (active cluster set)", "value": "cluster"}],
                         value=None, placeholder="Annotation / clusters to compare with (reference mode: required)…",
                         style={"width": "300px", "fontSize": "11px"}),
            html.Span("Map layer:", style={**L, "marginLeft": "10px"}),
            dcc.Dropdown(id="txf-layer", options=[], value=None, placeholder="Off - dominant factor / one factor…",
                         style={"width": "280px", "fontSize": "11px"}),
            html.Span("(opacity and colour maximum: Transcript Density tab)", style=L),
        ], style=S),
        html.Div(id="txf-status", style={**analysis._AN_STATUS_STYLE, "whiteSpace": "pre-wrap"}),
        html.Div(_TXF_NOTE, id="txf-note", style=analysis._AN_NOTE_STYLE),
        html.Div([
            controls._graph(id="txf-plot", msg="No transcript factors computed yet.",
                            filename="xenium_transcript_factors", min_h="320px"),
            html.Div([
                dash_table.DataTable(id="txf-table", data=[], columns=cols, **stats._dark_table_style(12)),
                html.Div([
                    dbc.Button("Download CSV", id="txf-download-btn", color="secondary", size="sm", n_clicks=0,
                               style={"fontSize": "11px", "padding": "2px 8px"}),
                    dcc.Download(id="txf-download"),
                ], style={"padding": "4px 0"}),
            ], style={"flex": "1.2", "minWidth": "0", "overflow": "auto", "padding": "0 6px"}),
        ], style={"display": "flex", "flex": "1", "minHeight": "400px", "gap": "6px"}),
        dcc.Store(id="txf-done", data=0),
        dcc.Interval(id="txf-poll", interval=1000, disabled=True),
        ui._plot_dim_controls("txf"),
    ], style={"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0",
              "overflowY": "auto"})])
