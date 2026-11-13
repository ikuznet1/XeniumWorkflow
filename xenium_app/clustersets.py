"""Cluster sets computed in the app (Banksy domains, joint clusters): graph clustering of
an embedding, and installing the labels as a cluster set of the view that Color By →
Cluster, the group pickers, the composition test and the export read like a Xenium one.

A set is an integer column: ``clust__<key>`` in the cell table of the primary and extra
samples (and in ``cluster_methods`` of the dataset dict), ``cluster_<key>`` in the cell
table of a Baysor / Proseg run. Clusters are numbered from 1; 0 marks a cell without a
label (outside the subset the set was computed on, or not in the matrix). The labels are
kept per dataset in ``<cache>/clustersets_<dataset>/`` and installed again on load.
"""

import os
import re
import uuid
import json
import hashlib
import threading
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import core, jobs, caching, dataview, ui, analysis, stats


_cs_registry: dict = {}        # key → {"name", "shared", "meta", "levels"}
_cs_lock = threading.RLock()
_CS_UNASSIGNED = 0             # value of a cell without a label
_CS_EXACT_KNN_MAX = 40_000     # rows up to which the neighbours are exact (brute force)
_CS_LEIDEN_HINT = ("Leiden needs the packages leidenalg and igraph (pip install leidenalg igraph); "
                   "without them the clusters are a KMeans partition of the same embedding.")


# ─── Graph clustering of an embedding ─────────────────────────────────────────
# ─── Size guard of the graph clustering ───────────────────────────────────────
# What clustering n cells on a k-neighbour graph costs, from runs on 0.1 to 0.35 million
# cells (README, "Performance at study scale"): the graph handed to Leiden holds about
# n × k edges, and igraph / leidenalg keep Python-level lists of them.
_CS_GB_PER_M_EDGES = 0.40          # memory of the edge lists, the graph and the partition
_CS_MIN_PER_M_EDGES = 0.25         # neighbours + shared-neighbour graph + Leiden
_CS_RAM_SHARE = 0.5                # a job estimated above this share of the machine's memory is refused


_CS_CGROUP_FILES = ("/sys/fs/cgroup/memory.max",                        # cgroup v2
                    "/sys/fs/cgroup/memory/memory.limit_in_bytes")       # cgroup v1


def _cs_cgroup_limit_gb(files=None) -> float:
    """Memory limit of the container this process runs in, in GB (0.0: none set or not
    readable). 'max' (v2) and the huge 'unlimited' value of v1 are no limit."""
    best = 0.0
    for f in (_CS_CGROUP_FILES if files is None else files):
        try:
            with open(f) as fh:
                txt = fh.read().strip()
            v = float(int(txt))
        except (OSError, ValueError):
            continue
        if 0 < v < 2 ** 60:
            best = v / 1e9 if not best else min(best, v / 1e9)
    return best


def _cs_ram_gb() -> float:
    """Memory this process can use, in GB (0.0 when it cannot be read): the physical
    memory of the machine, or the memory limit of the container when that is smaller (a
    Docker / Cloud Run container on a large host is killed at its limit, not the host's)."""
    try:
        phys = float(os.sysconf("SC_PAGE_SIZE")) * float(os.sysconf("SC_PHYS_PAGES")) / 1e9
    except (ValueError, OSError, AttributeError):
        phys = 0.0
    lim = _cs_cgroup_limit_gb()
    return min(phys, lim) if (phys > 0 and lim > 0) else (phys or lim)


def _cs_cost(n_cells: int, k: int, passes: int = 1, matrix_gb: float = 0.0, extra_minutes: float = 0.0,
             graph: bool = True) -> dict:
    """Rough cost of a clustering job: {"edges" (millions), "gb" (memory above the loaded
    data), "minutes"}. *passes*: clusterings run one after the other (time adds up, memory
    does not); *matrix_gb*: working copies of the count matrix the job makes;
    *extra_minutes*: the part before the graph (PCA, Harmony, neighbourhood features).
    graph=False: no neighbour graph is built (the KMeans fallback when Leiden is not
    installed), so the edges cost neither memory nor time."""
    edges = float(n_cells) * float(k) / 1e6 if graph else 0.0
    return {"edges": edges, "gb": edges * _CS_GB_PER_M_EDGES + float(matrix_gb),
            "minutes": edges * _CS_MIN_PER_M_EDGES * max(1, int(passes)) + float(extra_minutes)}


def _cs_size_guard(what: str, n_cells: int, k: int, passes: int = 1, matrix_gb: float = 0.0,
                   extra_minutes: float = 0.0, graph=None) -> tuple:
    """(refusal text or None, one-line estimate or "") for a clustering job.

    The estimate is given for anything above a minute or a gigabyte. The job is refused
    when its estimated memory exceeds _CS_RAM_SHARE of the machine's memory, unless
    XENIUM_ALLOW_LARGE_JOBS=1 is set: better a number up front than a machine that swaps
    for an hour."""
    # graph=None: whether a neighbour graph is built at all is what _cs_graph_cluster will
    # do - Leiden on a kNN / SNN graph, or (leidenalg / igraph missing) KMeans without one
    graph = _cs_leiden_available() if graph is None else bool(graph)
    c = _cs_cost(n_cells, k, passes, matrix_gb, extra_minutes, graph=graph)
    ram = _cs_ram_gb()
    est = (f"estimated for {int(n_cells):,} cells with {int(k)} graph neighbours: about {c['gb']:.0f} GB of "
           f"memory above the loaded data and about {max(1, round(c['minutes'])):,} min "
           f"(a graph of ~{c['edges']:.0f} million edges)" if graph else
           f"estimated for {int(n_cells):,} cells (KMeans fallback, no neighbour graph: Leiden is not "
           f"installed): about {c['gb']:.0f} GB of memory above the loaded data and about "
           f"{max(1, round(c['minutes'])):,} min")
    if c["gb"] < 1.0 and c["minutes"] < 1.0:
        return None, ""
    allow = os.environ.get("XENIUM_ALLOW_LARGE_JOBS", "").strip().lower() in ("1", "true", "yes", "on")
    if ram > 0 and c["gb"] > _CS_RAM_SHARE * ram and not allow:
        return (f"{what} was not started: {est}; this machine has {ram:.0f} GB"
                f"{' (the memory limit of its container)' if 0 < _cs_cgroup_limit_gb() <= ram else ''}. "
                f"Cluster fewer cells (subset() in the REPL, or fewer samples)"
                f"{', lower the number of graph neighbours' if graph else ''}"
                f"{', lower the gene cap' if 'gene' in str(what).lower() or 'banksy' in str(what).lower() else ''}"
                f", or set XENIUM_ALLOW_LARGE_JOBS=1 to run it anyway."), est
    return None, est[0].upper() + est[1:] + "."


def _cs_leiden_available() -> bool:
    try:
        import leidenalg  # noqa: F401
        import igraph     # noqa: F401
        return True
    except Exception:
        return False


def _cs_knn(X, k: int, seed: int = 0) -> tuple:
    """(indices [n, k] of the k nearest other rows of X (Euclidean), how they were found).
    Exact up to _CS_EXACT_KNN_MAX rows; above, pynndescent when it imports (approximate,
    fixed seed), else exact."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = len(X)
    k = int(max(1, min(k, n - 1)))
    if n > _CS_EXACT_KNN_MAX:
        try:
            from pynndescent import NNDescent
            idx = NNDescent(X, n_neighbors=k + 1, metric="euclidean", random_state=int(seed),
                            n_jobs=-1).neighbor_graph[0]
            out = np.empty((n, k), dtype=np.int64)
            for r0 in range(0, n, 200_000):
                blk = idx[r0:r0 + 200_000]
                me = np.arange(r0, r0 + len(blk))[:, None]
                # drop the row itself (first column as a rule; anywhere after a tie)
                order = np.argsort(blk == me, axis=1, kind="stable")
                out[r0:r0 + len(blk)] = np.take_along_axis(blk, order, axis=1)[:, :k]
            return out, "approximate neighbours (pynndescent)"
        except Exception:
            pass
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", n_jobs=-1).fit(X)
    out = np.empty((n, k), dtype=np.int64)
    for r0 in range(0, n, 20_000):
        jobs._job_check()
        blk = nn.kneighbors(X[r0:r0 + 20_000], return_distance=False)
        me = np.arange(r0, r0 + len(blk))[:, None]
        order = np.argsort(blk == me, axis=1, kind="stable")
        out[r0:r0 + len(blk)] = np.take_along_axis(blk, order, axis=1)[:, :k]
    return out, "exact neighbours"


def _cs_snn_graph(idx) -> "sp.csr_matrix":
    """Shared-nearest-neighbour graph of a kNN index [n, k]: for every kNN edge (i, j)
    the Jaccard index of the two neighbour sets (each row counts itself), made symmetric
    (upper-triangular CSR, weight = the larger of the two directions)."""
    idx = np.asarray(idx, dtype=np.int64)
    n, k = idx.shape
    rows = np.repeat(np.arange(n), k + 1)
    cols = np.concatenate([np.arange(n)[:, None], idx], axis=1).ravel()
    A = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0                                   # a repeated neighbour counts once
    size = np.asarray(A.sum(axis=1)).ravel()
    ei, ej = np.repeat(np.arange(n), k), idx.ravel()
    keep = ei != ej
    ei, ej = ei[keep], ej[keep]
    w = np.empty(len(ei), dtype=np.float32)
    for e0 in range(0, len(ei), 500_000):
        jobs._job_check()
        a, b = ei[e0:e0 + 500_000], ej[e0:e0 + 500_000]
        shared = np.asarray(A[a].multiply(A[b]).sum(axis=1)).ravel()
        w[e0:e0 + 500_000] = shared / (size[a] + size[b] - shared)
    lo, hi = np.minimum(ei, ej), np.maximum(ei, ej)
    G = sp.coo_matrix((w, (lo, hi)), shape=(n, n)).tocsr()       # duplicates are summed …
    C = sp.coo_matrix((np.ones(len(w), dtype=np.float32), (lo, hi)), shape=(n, n)).tocsr()
    G.data /= C.data                                             # … both directions have the same Jaccard
    return G


def _cs_order_by_size(labels) -> np.ndarray:
    """Relabel 0 … k−1 by cluster size, largest first (ties: the smaller old label)."""
    labels = np.asarray(labels, dtype=np.int64)
    cnt = np.bincount(labels)
    order = np.lexsort((np.arange(len(cnt)), -cnt))
    rank = np.empty(len(cnt), dtype=np.int64)
    rank[order] = np.arange(len(cnt))
    return rank[labels]


def _cs_graph_cluster(X, n_neighbors=15, resolution=1.0, seed=0, progress=None, n_clusters=None) -> tuple:
    """Cluster the rows of an embedding X (cells × components).

    k nearest neighbours → shared-nearest-neighbour graph (Jaccard weights) → Leiden
    (modularity with a resolution parameter, fixed seed) when leidenalg and igraph are
    installed; otherwise KMeans on X with *n_clusters* clusters (default: 10 ×
    resolution, between 2 and 50), named as the fallback it is.

    Returns (labels, info): labels int64 0 … k−1 ordered by cluster size, largest first;
    info = {"backend", "n_clusters", "note", "neighbours"}."""
    X = np.asarray(X, dtype=np.float32)
    n = len(X)
    prog = progress or (lambda f, m=None: None)
    if n < 3:
        return np.zeros(n, dtype=np.int64), {"backend": "none", "n_clusters": int(n > 0), "note": "fewer than 3 cells",
                                             "neighbours": ""}
    if _cs_leiden_available():
        import igraph as ig
        import leidenalg
        prog(0.0, "Nearest neighbours of the embedding…")
        jobs._job_check()
        idx, how = _cs_knn(X, int(n_neighbors), seed)
        prog(0.45, "Shared-nearest-neighbour graph…")
        jobs._job_check()
        G = _cs_snn_graph(idx).tocoo()
        jobs._job_check()
        prog(0.6, f"Leiden clustering (resolution {float(resolution):g})…")
        g = ig.Graph(n=n, edges=np.column_stack([G.row, G.col]).tolist(), directed=False)
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                        weights=G.data.astype(np.float64).tolist(),
                                        resolution_parameter=float(resolution), seed=int(seed), n_iterations=2)
        jobs._job_check()
        labels = _cs_order_by_size(np.asarray(part.membership, dtype=np.int64))
        prog(1.0, None)
        return labels, {"backend": "leiden", "n_clusters": int(labels.max()) + 1, "neighbours": how,
                        "note": f"Leiden on the shared-nearest-neighbour graph of {int(n_neighbors)} {how}"}
    from sklearn.cluster import MiniBatchKMeans, KMeans
    k = int(n_clusters) if n_clusters else int(max(2, min(50, round(10 * float(resolution)))))
    k = max(1, min(k, n))
    prog(0.2, f"KMeans (k = {k}; Leiden is not installed)…")
    jobs._job_check()
    km = (KMeans(n_clusters=k, n_init=5, random_state=int(seed)) if n <= 20_000
          else MiniBatchKMeans(n_clusters=k, n_init=5, random_state=int(seed), batch_size=4096))
    labels = _cs_order_by_size(km.fit_predict(X))
    prog(1.0, None)
    return labels, {"backend": f"kmeans (fallback, k = {k})", "n_clusters": int(labels.max()) + 1,
                    "neighbours": "", "note": _CS_LEIDEN_HINT}


# ─── Names, keys, files ───────────────────────────────────────────────────────
def _cs_key(name: str) -> str:
    """The method key of a set name: lower-case ASCII, usable as a column suffix."""
    # "Banksy-style λ=0.8" keeps the key of the sets installed as "Banksy λ=0.8" before the
    # feature was relabelled: banksy_lambda_0.8 (cached sets and their columns stay readable)
    s = re.sub(r"^Banksy-style\b", "Banksy", str(name)).replace("λ", "lambda").replace("=", " ")
    return re.sub(r"[^0-9a-z.]+", "_", s.lower()).strip("_") or "set"


def _cs_display_name(name) -> str:
    """The name a set is shown under: sets cached as "Banksy λ=…" (before the relabelling)
    read "Banksy-style λ=…"."""
    return re.sub(r"^Banksy(?= λ=)", "Banksy-style", str(name))


def _cs_label(method) -> str:
    """Display name of a cluster set: the name it was installed under, else the name the
    app derives from a Xenium analysis directory."""
    with _cs_lock:
        ent = _cs_registry.get(str(method))
    return _cs_display_name(ent["name"]) if ent else ui.method_label(str(method))


# (sample key, segmentation slug, set key) -> {"run": id of the install, "shared": bool}.
# Whether labels are comparable across samples is a property of ONE install over several
# tables, not of the set's name: a later install under the same name on another
# segmentation, or on other samples, must not change how the first is read.
_cs_table_run: dict = {}


def _cs_is_shared(method) -> bool:
    """True when the labels of this set mean the same in every sample of the Xenium view:
    every loaded sample carries the set from the same install, and that install was a
    joint one. The multi-sample view then does not prefix the labels per sample."""
    key = str(method)
    with _cs_lock:
        if key not in _cs_registry:
            return False
    try:
        keys = [t["key"] for t in stats._view_tables("xenium")]      # the loaded samples
    except Exception:
        return False
    with _cs_lock:
        recs = [_cs_table_run.get((k, "xenium", key)) for k in keys]
    if not recs or any(r is None for r in recs):
        return False
    return bool(all(r["shared"] for r in recs) and len({r["run"] for r in recs}) == 1)


def _cs_dir(ds) -> str:
    return os.path.join(core.CACHE_BASE_DIR,
                        f"clustersets_{hashlib.md5(dataview._ds_key(ds).encode()).hexdigest()[:8]}")


def _cs_seg_slug(alt, seg_source) -> str:
    if alt is None:
        return "xenium"
    run = os.path.basename(str(alt.get("out_dir") or "").rstrip("/")) or "run"
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", f"{core._seg_tool(seg_source)}_{run}")


def _cs_forget_run(final_dir) -> None:
    """A run was committed to *final_dir*: remove the cached sets of an earlier run stored
    there (the files are named by tool and directory) and their install records."""
    base = os.path.basename(str(final_dir).rstrip("/"))
    slugs = {re.sub(r"[^0-9A-Za-z_.-]+", "_", f"{t}_{base}") for t in core._SEG_TOOL_LABELS}
    with _cs_lock:
        for k in [k for k in _cs_table_run if k[1] in slugs]:
            _cs_table_run.pop(k, None)
        try:
            dirs = [os.path.join(core.CACHE_BASE_DIR, d) for d in os.listdir(core.CACHE_BASE_DIR)
                    if d.startswith("clustersets_")]
        except OSError:
            dirs = []
        for d in dirs:
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if any(fn.endswith(f"__{s}{ext}") for s in slugs for ext in (".json", ".parquet")):
                    try:
                        os.remove(os.path.join(d, fn))
                    except OSError:
                        pass


caching._seg_commit_hooks.append(_cs_forget_run)


def _cs_reseg_col(columns, method):
    """The column of an installed set in the cell table of a Baysor / Proseg run, when
    *method* names one that this run has; else None (the run's own clusters are shown)."""
    col = f"cluster_{method}"
    with _cs_lock:
        known = str(method) in _cs_registry
    return col if known and col in columns else None


# ─── Installing a set into the view ───────────────────────────────────────────
def _cs_codes(per_table_labels) -> tuple:
    """Integer codes 1 … k per table (0: no label) and the level names, from labels of any
    type. Integer-like labels keep their order (0 … k−1 becomes 1 … k); other labels are
    numbered in sorted order."""
    ser = [None if l is None else pd.Series(np.asarray(l, dtype=object)) for l in per_table_labels]
    pooled = pd.concat([s.dropna() for s in ser if s is not None]) if any(s is not None for s in ser) else pd.Series(dtype=object)
    uniq = list(pd.unique(pooled))
    try:
        nums = [float(u) for u in uniq]
        intlike = all(np.isfinite(v) and float(v).is_integer() for v in nums)
    except (TypeError, ValueError):
        intlike = False
    if intlike:
        levels = [u for _v, u in sorted(zip(nums, uniq), key=lambda p: p[0])]
        names = [str(int(float(u))) for u in levels]
    else:
        levels = sorted(uniq, key=str)
        names = [str(u) for u in levels]
    lut = {u: i + 1 for i, u in enumerate(levels)}
    codes = [None if s is None else s.map(lambda v: lut.get(v, _CS_UNASSIGNED) if v is not None and v == v
                                          else _CS_UNASSIGNED).to_numpy(dtype=np.int64) for s in ser]
    return codes, names


def _cs_by_id(df, codes) -> "pd.Series":
    s = pd.Series(np.asarray(codes, dtype=np.int64), index=df.index.astype(str))
    return s[~s.index.duplicated(keep="first")]


def _cs_onto(full_df, by_id) -> np.ndarray:
    return (by_id.reindex(full_df.index.astype(str)).fillna(_CS_UNASSIGNED).to_numpy(dtype=np.int64))


def _cs_write(table, alt, key: str, by_id) -> None:
    """Put the codes (a Series by cell id) into the cell table of one view table, and into
    the frame unsubset() restores when a subset is active."""
    if alt is not None:
        col = f"cluster_{key}"
        new = alt["cells_df"].copy()
        new[col] = _cs_onto(new, by_id)
        full = alt.get("_cells_df_original")
        if full is not None:
            full[col] = _cs_onto(full, by_id)
        alt["cells_df"] = new
        return
    ds = table.get("ds") or core.DATA
    col = dataview.cluster_col(key)
    full = ds.get("_df_original")
    if full is not None:
        full[col] = _cs_onto(full, by_id)
        new = ds["df"].copy()                   # a subset slice is not written to
        new[col] = _cs_onto(new, by_id)
        ds["df"] = new
    else:
        ds["df"][col] = _cs_onto(ds["df"], by_id)
    cm = ds.get("cluster_methods")
    if cm is None:
        cm = ds["cluster_methods"] = []
    if key not in cm:
        cm.append(key)


def _cs_touch_caches() -> None:
    dataview._umap_fig_cache.clear()
    dataview._umap_df_cache.clear()


def _cs_install(name, per_table_labels, seg_source, meta=None) -> str:
    """Install the cluster set *name* into the tables of the active view.

    per_table_labels[i]: labels of the cells of stats._view_tables(seg_source)[i], in row
    order (None / NaN: no label; None instead of an array: the sample is left out).
    Returns the method key (the value of the Clustering Method dropdown). The set is
    written to the cache of every sample; *meta* (JSON-able) is stored with it. A set
    installed over several samples is shared: its labels mean the same in each."""
    seg_source = seg_source or "xenium"
    key = _cs_key(name)
    tables = stats._view_tables(seg_source)
    if len(per_table_labels) != len(tables):
        raise ValueError(f"{len(per_table_labels)} label arrays for {len(tables)} samples of the view")
    for t, lab in zip(tables, per_table_labels):
        if lab is not None and len(lab) != len(t["df"]):
            raise ValueError(f"{t['label']}: {len(lab)} labels for {len(t['df'])} cells")
    alt = analysis._an_alt_res(seg_source)
    codes, levels = _cs_codes(per_table_labels)
    shared = bool((meta or {}).get("shared", len(tables) > 1))
    run = uuid.uuid4().hex[:12]
    slug = _cs_seg_slug(alt, seg_source)
    ent = {"name": str(name), "shared": shared, "meta": dict(meta or {}), "levels": levels, "run": run}
    with _cs_lock:
        if alt is None or key not in _cs_registry:
            # a set of the same name installed on a Baysor / Proseg run does not replace
            # the entry (meta, shared flag) of the one on the Xenium samples
            _cs_registry[key] = ent
        for t, c in zip(tables, codes):
            if c is None:
                _cs_table_run.pop((t["key"], slug, key), None)
                continue
            _cs_table_run[(t["key"], slug, key)] = {"run": run, "shared": shared}
            by_id = _cs_by_id(t["df"], c)
            _cs_write(t, alt, key, by_id)
            # the cache holds the labels of every cell of the sample (0 outside an active
            # subset), so that a set computed on a small subset comes back as well
            full = (alt.get("_cells_df_original") if alt is not None else (t.get("ds") or core.DATA).get("_df_original"))
            if full is not None:
                by_id = pd.Series(_cs_onto(full, by_id), index=full.index.astype(str))
                by_id = by_id[~by_id.index.duplicated(keep="first")]
            try:
                d = _cs_dir(t["ds"])
                os.makedirs(d, exist_ok=True)
                stem = os.path.join(d, f"{key}__{_cs_seg_slug(alt, seg_source)}")
                pd.DataFrame({"cell_id": by_id.index, "label": by_id.to_numpy()}).to_parquet(stem + ".parquet.tmp")
                os.replace(stem + ".parquet.tmp", stem + ".parquet")
                with open(stem + ".json", "w") as fh:
                    json.dump({"key": key, "seg": _cs_seg_slug(alt, seg_source),
                               "seg_run": core._seg_run_id(alt), "n_cells": int(len(by_id)), **ent},
                              fh, default=str)
            except Exception as exc:
                print(f"  cluster set '{name}': not written to the cache ({exc})", flush=True)
        _cs_touch_caches()
    return key


def _cs_restore(seg_source="xenium") -> list:
    """Install the cached sets of the samples of the view that are not installed yet, and
    list `clust__` columns a store brought along as cluster sets. Returns the keys added."""
    seg_source = seg_source or "xenium"
    try:
        tables = stats._view_tables(seg_source)
    except Exception:
        return []
    alt = analysis._an_alt_res(seg_source)
    slug = _cs_seg_slug(alt, seg_source)
    added = []
    with _cs_lock:
        for t in tables:
            ds = t.get("ds") or core.DATA
            df = t["df"]
            if alt is None:
                cm = ds.get("cluster_methods")
                if cm is None:
                    cm = ds["cluster_methods"] = []
                for c in df.columns:
                    if str(c).startswith("clust__") and str(c)[7:] not in cm:
                        cm.append(str(c)[7:])
                        added.append(str(c)[7:])
            d = _cs_dir(ds)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(f"__{slug}.json"):
                    continue
                try:
                    with open(os.path.join(d, fn)) as fh:
                        info = json.load(fh)
                    key = info["key"]
                    # a set of another run stored in this directory (a re-run with the
                    # same parameters) is not this run's: its cells are other cells
                    if alt is not None and str(info.get("seg_run") or "") != core._seg_run_id(alt):
                        continue
                    col = dataview.cluster_col(key) if alt is None else f"cluster_{key}"
                    _cs_registry.setdefault(key, {"name": info.get("name", key), "shared": bool(info.get("shared")),
                                                  "meta": info.get("meta") or {}, "levels": info.get("levels") or [],
                                                  "run": info.get("run")})
                    if col in df.columns:
                        continue
                    tab = pd.read_parquet(os.path.join(d, fn[:-5] + ".parquet"))
                    by_id = pd.Series(tab["label"].to_numpy(dtype=np.int64), index=tab["cell_id"].astype(str))
                    full = (alt.get("_cells_df_original") if alt is not None else ds.get("_df_original"))
                    ids = pd.Index((full if full is not None else df).index.astype(str))
                    # labels of other cells (another run under the same name) are refused:
                    # most of the file's cells must be cells of this table. (Not the reverse:
                    # a set computed on a small subset covers few cells of the table.)
                    if not len(ids) or not len(by_id) or float(by_id.index.isin(ids).mean()) < 0.5:
                        continue
                    _cs_write(t, alt, key, by_id)
                    # this table's own file says which install its labels come from
                    _cs_table_run[(t["key"], slug, key)] = {
                        "run": info.get("run") or f"legacy:{info.get('name', key)}",
                        "shared": bool(info.get("shared"))}
                    df = (alt["cells_df"] if alt is not None else ds["df"])
                    added.append(key)
                except Exception as exc:
                    print(f"  cluster sets: {fn} not restored ({exc})", flush=True)
        if added:
            _cs_touch_caches()
    return added


def _cs_methods(seg_source=None) -> list:
    """Keys of the installed sets; with *seg_source*, those the tables of that view carry."""
    with _cs_lock:
        keys = list(_cs_registry)
    if seg_source is None:
        return keys
    alt = analysis._an_alt_res(seg_source)
    tables = stats._view_tables(seg_source)
    col = (lambda k: f"cluster_{k}") if alt is not None else dataview.cluster_col
    return [k for k in keys if any(col(k) in t["df"].columns for t in tables)]


def _cs_meta(method) -> dict:
    with _cs_lock:
        ent = _cs_registry.get(str(method))
    return dict(ent["meta"]) if ent else {}


def _cs_remove(name) -> bool:
    """Remove a set (by name or key) from every loaded table and from the cache."""
    key = name if str(name) in _cs_registry else _cs_key(name)
    with _cs_lock:
        had = _cs_registry.pop(key, None) is not None
        for k3 in [k3 for k3 in _cs_table_run if k3[2] == key]:
            del _cs_table_run[k3]
        dsets = [core.DATA] + list(dataview.EXTRA_DATASETS)
        for ds in dsets:
            col = dataview.cluster_col(key)
            for fk in ("df", "_df_original"):
                f = ds.get(fk)
                if f is not None and col in f.columns:
                    ds[fk] = f.drop(columns=[col])
                    had = True
            cm = ds.get("cluster_methods") or []
            if key in cm and len(cm) > 0:
                cm.remove(key)
            d = _cs_dir(ds)
            if os.path.isdir(d):
                for fn in os.listdir(d):
                    if fn.startswith(f"{key}__"):
                        try:
                            os.remove(os.path.join(d, fn))
                        except OSError:
                            pass
        for state, lock in ((core._baysor_state, core._baysor_lock), (core._proseg_state, core._proseg_lock)):
            with lock:
                res = state.get("result")
            if res:
                for fk in ("cells_df", "_cells_df_original"):
                    f = res.get(fk)
                    if f is not None and f"cluster_{key}" in f.columns:
                        res[fk] = f.drop(columns=[f"cluster_{key}"])
                        had = True
        _cs_touch_caches()
    return had


def _cs_options(seg_source="xenium") -> list:
    """Options of the Clustering Method dropdown: the cluster sets of the primary sample,
    and for a Baysor / Proseg view the sets installed on that run."""
    methods = list(core.DATA.get("cluster_methods") or [])
    if analysis._an_alt_res(seg_source) is not None:
        methods += [k for k in _cs_methods(seg_source) if k not in methods]
    return [{"label": _cs_label(m), "value": m} for m in methods]
