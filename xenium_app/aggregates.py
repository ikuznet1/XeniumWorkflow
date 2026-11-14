"""Aggregates: connected groups of cells of a chosen class (e.g. the immune cell types)
whose local class count exceeds within-stratum label permutations (compute + layout).

Every neighbour query, Delaunay triangulation and permutation in this module is made on
the cells of ONE sample: no count, edge or expectation involves cells of two samples (the
view offsets samples in x only for drawing).
"""

import threading
import time as _time
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from dash import dcc, html
from dash import dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from . import core, jobs, dataview, ui, figures, analysis, stats, controls
from . import provenance


# ─── State ────────────────────────────────────────────────────────────────────
_agg_state: dict = {"status": "idle", "message": "", "result": None, "version": 0, "show": True}
_agg_lock = threading.Lock()

_AGG_RADIUS_UM = 50.0        # neighbourhood in which class cells are counted (the cell itself included)
_AGG_MIN_CLASS = 10          # an aggregate holds at least this many class cells
_AGG_LINK_UM = 30.0          # hotspot cells are linked by Delaunay edges of at most this length
_AGG_N_NULL = 200            # within-stratum label permutations
_AGG_ALPHA = 0.01            # hotspot: permutation p = (k + 1) / (permutations + 1) at or below this
_AGG_MIN_NB = 3              # hotspot: at least this many class cells within the radius
_AGG_N_CONTROL = 20          # permuted labellings run through the same detection (false-aggregate rate)
_AGG_RIM_UM = 25.0           # cells this close to an aggregate are not "outside" for the O/E
_AGG_TILE_GRID_UM = 5.0      # tissue area: grid step ...
_AGG_TILE_REACH_UM = 20.0    # ... and the furthest a grid point may be from its nearest cell
_AGG_HULL_ALPHA_UM = 30.0    # alpha hull: Delaunay triangles with every edge at most this long
_AGG_OE_MIN_POOL = 20        # outside cells a stratum needs to give the expected composition
_AGG_BATCH = 25              # permutations per sparse product (and per cancellation point)
_AGG_THREADS = max(1, min(4, (os.cpu_count() or 2) // 2))   # replicates of the permutation sorted at a time
_AGG_OUTLINE_COLOR = "#f0c040"
_AGG_COLUMNS = ("aggregate_id", "aggregate_hotspot", "aggregate_p")   # per-cell columns of a run
_AGG_COLOR_KEY = "agg#id"    # Color By value "geneset:agg#id"
_AGG_COLOR_LABEL = "Aggregate id (1 = most class cells)"

_AGG_NOTE = (
    "An aggregate is a connected group of cells sitting where the chosen class (a set of cell types, "
    "e.g. the immune types) is locally denser than label permutations allow. Per sample (section): "
    "for every cell the class cells within R µm are counted (the cell itself included); the class "
    "label is then permuted among the cells of the same stratum (compartment × edge-distance bin × "
    "local-density quintile, from the Confound Controls tab or computed with its defaults), so a "
    "class that merely prefers the edge or dense tissue does not create aggregates. A cell is a "
    "hotspot when at most α of the permuted counts reach its observed count, p = (k + 1) / "
    "(permutations + 1) ≤ α, and it has at least the given number of class cells around it. Hotspot "
    "cells (of any type) joined by Delaunay edges no longer than the link length form a component; "
    "components with at least N class cells are the aggregates. Area = Σ over the member cells of "
    "the tissue each one is nearest to (5 µm grid, at most 20 µm from a cell); tissue area of the "
    "sample likewise. Composition O/E per cell type: cells of the type inside ÷ the number expected "
    "from the cells of the same stratum that are outside every aggregate and its rim. Control: the "
    "same detection on permuted labellings gives the number of aggregates chance alone produces "
    "(false-aggregate rate = control mean ÷ observed). The per-cell p-values have the cells of one "
    "section as their unit (exchangeable within strata); they are a detection threshold, not a "
    "test across samples. Between conditions the unit is the sample / donor: aggregates per mm² "
    "and area fraction are compared with the exact sample-level test when the design has at least "
    "3 units per condition. Recompute after changing the subset, the segmentation or the annotation.")


def _agg_result(seg_source=None):
    """The published result when it belongs to the loaded dataset (and, with *seg_source*,
    to that segmentation), else None."""
    with _agg_lock:
        res = _agg_state.get("result")
    if not res or res.get("gen") != jobs._dataset_gen:
        return None
    if seg_source is not None and res.get("seg_tag") != controls._seg_tag(seg_source):
        return None
    return res


# ─── Detection on one sample ──────────────────────────────────────────────────
def _agg_adjacency(xy, radius: float) -> "sp.csr_matrix":
    """Cells × cells indicator of "within *radius* µm" for ONE section, the diagonal
    included (float32 CSR)."""
    from scipy.spatial import cKDTree
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    pr = cKDTree(xy).query_pairs(float(radius), output_type="ndarray")
    i = np.concatenate([pr[:, 0], pr[:, 1], np.arange(n)])
    j = np.concatenate([pr[:, 1], pr[:, 0], np.arange(n)])
    return sp.csr_matrix((np.ones(len(i), dtype=np.float32), (i, j)), shape=(n, n))


def _agg_perm_within(strata, y, rng, n_rep: int) -> np.ndarray:
    """*n_rep* permutations of *y* within the strata: (cells, n_rep) float32; every
    stratum keeps its number of class cells."""
    strata = np.asarray(strata)
    if len(strata) and 0 <= strata.min() and strata.max() < 65536:
        strata = strata.astype(np.uint16)          # a stable sort of 16-bit keys is a radix sort
    base = np.argsort(strata, kind="stable")
    n_rep = int(n_rep)
    out = np.empty((len(y), n_rep), dtype=np.float32)
    # the random numbers are drawn one replicate after the other, as before (the same
    # stream, the same permutations); only the sorting of each replicate, which is all the
    # time, runs in a few threads
    draws = [rng.random(len(y)) for _ in range(n_rep)]

    def _one(b):
        # cells ordered by (stratum, random number): the order np.lexsort((r, strata)) gives,
        # as a sort by r followed by a stable sort by stratum (several times faster)
        o1 = np.argsort(draws[b])
        order = o1[np.argsort(strata[o1], kind="stable")]
        yp = np.empty(len(y), dtype=np.float32)
        yp[base] = y[order]
        out[:, b] = yp

    n_thr = int(max(1, min(_AGG_THREADS, n_rep))) if len(y) >= 20_000 else 1
    if n_thr > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_thr) as pool:
            list(pool.map(_one, range(n_rep)))
    else:
        for b in range(n_rep):
            _one(b)
    return out


def _agg_detect(xy, ind, strata, radius: float = _AGG_RADIUS_UM, n_null: int = _AGG_N_NULL,
                n_control: int = _AGG_N_CONTROL, seed: int = 0, progress=None, adjacency=None) -> dict:
    """Local class count of every cell of ONE section against within-stratum permutations.

    ind: boolean class indicator per cell; strata: integer stratum per cell. Returns I
    (class cells within *radius*, the cell itself included), k (number of the *n_null*
    permuted labellings whose count at that cell is ≥ I), p = (k + 1) / (n_null + 1), and
    for *n_control* further permuted labellings judged against the same null draws:
    ind_nc, I_nc, k_nc (each [control, cell])."""
    ind = np.asarray(ind, dtype=bool)
    n = len(ind)
    strata = np.asarray(strata, dtype=np.int64)
    M = _agg_adjacency(xy, radius) if adjacency is None else adjacency
    y = ind.astype(np.float32)
    rng, rng_nc = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    I = np.rint(M @ y).astype(np.int32)
    K = int(max(0, n_control))
    Ync = _agg_perm_within(strata, y, rng_nc, K) if K else np.zeros((n, 0), dtype=np.float32)
    Inc = np.rint(M @ Ync).astype(np.int32) if K else np.zeros((n, 0), dtype=np.int32)
    cnt = np.zeros(n, dtype=np.int32)
    cnt_nc = np.zeros((n, K), dtype=np.int32)
    n_null = int(max(1, n_null))
    for b0 in range(0, n_null, _AGG_BATCH):
        jobs._job_check()
        nb = min(_AGG_BATCH, n_null - b0)
        Ip = np.rint(M @ _agg_perm_within(strata, y, rng, nb)).astype(np.int32)
        cnt += (Ip >= I[:, None]).sum(axis=1)
        for c in range(K):
            cnt_nc[:, c] += (Ip >= Inc[:, [c]]).sum(axis=1)
        if progress is not None:
            progress((b0 + nb) / n_null)
    return {"I": I, "k": cnt, "p": (cnt + 1.0) / (n_null + 1.0), "n_null": n_null,
            "ind_nc": Ync.T > 0.5, "I_nc": Inc.T, "k_nc": cnt_nc.T}


def _agg_hot(I, k, n_null: int, alpha: float = _AGG_ALPHA, min_nb: int = _AGG_MIN_NB) -> np.ndarray:
    """Hotspot cells: permutation p = (k + 1) / (n_null + 1) ≤ *alpha* and at least
    *min_nb* class cells within the radius."""
    return (((np.asarray(k) + 1.0) / (int(n_null) + 1.0)) <= float(alpha) + 1e-12) & (np.asarray(I) >= int(min_nb))


def _agg_edges(xy, max_len: float = _AGG_LINK_UM) -> tuple:
    """(u, v): the Delaunay edges of ONE section that are at most *max_len* µm long
    (u < v). Fewer than 3 cells, or all on a line: the pairs within *max_len*."""
    from scipy.spatial import Delaunay, cKDTree
    xy = np.asarray(xy, dtype=np.float64)
    try:
        if len(xy) < 3:
            raise ValueError
        tri = Delaunay(xy)
        ip, nb = tri.vertex_neighbor_vertices
        src = np.repeat(np.arange(len(xy)), np.diff(ip))
        keep = src < nb
        u, v = src[keep], nb[keep]
    except Exception:
        pr = cKDTree(xy).query_pairs(float(max_len), output_type="ndarray") if len(xy) > 1 else np.zeros((0, 2), int)
        u, v = pr[:, 0], pr[:, 1]
    ln = np.hypot(*(xy[u] - xy[v]).T) if len(u) else np.zeros(0)
    ok = ln <= float(max_len)
    return u[ok], v[ok]


def _agg_components(hot, ind, edges, n_min: int = _AGG_MIN_CLASS) -> np.ndarray:
    """Aggregate label per cell of ONE section (−1: none; 0 … m−1): connected components of
    the hotspot cells over *edges*, kept when they hold at least *n_min* class cells."""
    from scipy.sparse.csgraph import connected_components
    hot, ind = np.asarray(hot, dtype=bool), np.asarray(ind, dtype=bool)
    n = len(hot)
    lab = np.full(n, -1, dtype=np.int64)
    if not hot.any():
        return lab
    u, v = edges
    ok = hot[u] & hot[v]
    G = sp.coo_matrix((np.ones(int(ok.sum())), (u[ok], v[ok])), shape=(n, n))
    _, cc = connected_components(G, directed=False)
    cc = np.where(hot, cc, -1)
    valid = cc >= 0
    n_ind = np.bincount(cc[valid], weights=ind[valid].astype(np.float64), minlength=int(cc.max()) + 1)
    keep = np.flatnonzero(n_ind >= int(n_min))
    remap = np.full(int(cc.max()) + 1, -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    lab[valid] = remap[cc[valid]]
    return lab


def _agg_tile_area(xy, grid: float = _AGG_TILE_GRID_UM, reach: float = _AGG_TILE_REACH_UM) -> np.ndarray:
    """Tissue area (µm²) every cell of ONE section is the nearest cell of: the points of a
    *grid* µm lattice within *reach* µm of a cell are given to their nearest cell, each
    worth grid². The sum over a section is its tissue area; over an aggregate, its area."""
    from scipy.spatial import cKDTree
    xy = np.asarray(xy, dtype=np.float64)
    if not len(xy):
        return np.zeros(0)
    lo, hi = xy.min(axis=0) - reach, xy.max(axis=0) + reach
    gx = np.arange(lo[0] + grid / 2, hi[0], grid)
    gy = np.arange(lo[1] + grid / 2, hi[1], grid)
    tree = cKDTree(xy)
    cnt = np.zeros(len(xy))
    step = max(1, int(2_000_000 // max(1, len(gx))))
    for y0 in range(0, len(gy), step):
        jobs._job_check()
        G = np.stack(np.meshgrid(gx, gy[y0:y0 + step]), -1).reshape(-1, 2)
        d, j = tree.query(G, distance_upper_bound=float(reach), workers=-1)
        cnt += np.bincount(j[np.isfinite(d)], minlength=len(xy))
    return cnt * grid * grid


def _agg_hull(pts, kind: str = "convex", alpha: float = _AGG_HULL_ALPHA_UM) -> list:
    """Outline of a set of points as a list of rings ([[x, y], …], not closed).

    "convex": the convex hull (one ring). "alpha": the boundary of the union of the
    Delaunay triangles whose edges are all at most *alpha* µm long (may be several rings,
    holes included); when no triangle qualifies, the convex hull. Fewer than 3 points or
    points on a line: a thin box around them."""
    from scipy.spatial import ConvexHull, Delaunay
    pts = np.unique(np.asarray(pts, dtype=np.float64).reshape(-1, 2), axis=0)

    def _box():
        if not len(pts):
            return []
        lo, hi = pts.min(axis=0) - 2.0, pts.max(axis=0) + 2.0
        return [[[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]]

    def _convex():
        try:
            h = ConvexHull(pts)
            return [pts[h.vertices].tolist()]
        except Exception:
            return _box()

    if len(pts) < 3:
        return _box()
    if kind != "alpha":
        return _convex()
    try:
        tri = Delaunay(pts)
    except Exception:
        return _box()
    s = tri.simplices
    e = np.stack([s[:, [0, 1]], s[:, [1, 2]], s[:, [2, 0]]], axis=1)          # triangles × 3 edges × 2
    ln = np.hypot(*(pts[e[..., 0]] - pts[e[..., 1]]).transpose(2, 0, 1))
    good = (ln <= float(alpha)).all(axis=1)
    if not good.any():
        return _convex()
    ed = np.sort(e[good].reshape(-1, 2), axis=1)
    uq, cnt = np.unique(ed, axis=0, return_counts=True)
    border = uq[cnt == 1]
    nbrs: dict = {}
    for a, b in border.tolist():
        nbrs.setdefault(a, []).append(b)
        nbrs.setdefault(b, []).append(a)
    rings, used = [], set()
    for a0, b0 in border.tolist():
        if (a0, b0) in used:
            continue
        ring, prev, cur = [a0], a0, b0
        used.add((a0, b0))
        while cur != a0 and len(ring) <= len(border) + 1:
            ring.append(cur)
            nxt = [x for x in nbrs[cur] if x != prev and (min(cur, x), max(cur, x)) not in used]
            if not nxt:
                break
            used.add((min(cur, nxt[0]), max(cur, nxt[0])))
            prev, cur = cur, nxt[0]
        if len(ring) >= 3:
            rings.append(pts[ring].tolist())
    return rings or _convex()


def _agg_oe(lab, code, levels, T: int, outside, min_pool: int = _AGG_OE_MIN_POOL) -> tuple:
    """Composition of the aggregates of ONE section against matched outside cells.

    lab: aggregate per cell (−1 none), code: type code per cell (−1 unlabelled), levels:
    stratum codes, finest first; outside: boolean mask of the reference cells (outside
    every aggregate and its rim). For a labelled member cell the expected probability of
    each type is the type distribution of the labelled outside cells of its stratum, at the
    finest level with at least *min_pool* such cells (the whole section as the last
    resort). Returns (O, E): [aggregates, T] observed and expected numbers of cells."""
    lab, code = np.asarray(lab), np.asarray(code)
    m = int(lab.max()) + 1 if len(lab) else 0
    O, E = np.zeros((m, T)), np.zeros((m, T))
    mem = (lab >= 0) & (code >= 0)
    if not m or not mem.any():
        return O, E
    O = controls._bincount2(lab[mem], code[mem], m, T)
    ref = np.asarray(outside, dtype=bool) & (code >= 0)
    P = np.full((int(mem.sum()), T), np.nan)
    todo = np.ones(int(mem.sum()), dtype=bool)
    lv_all = list(levels) + [np.zeros(len(lab), dtype=np.int64)]
    for li, lv in enumerate(lv_all):
        lv = np.asarray(lv, dtype=np.int64)
        k = int(lv.max()) + 1
        N = controls._bincount2(lv[ref], code[ref], k, T) if ref.any() else np.zeros((k, T))
        tot = N.sum(axis=1)
        s = lv[mem]
        ok = todo & (tot[s] >= (min_pool if li < len(lv_all) - 1 else 1))
        P[ok] = N[s[ok]] / tot[s[ok]][:, None]
        todo &= ~ok
    P = np.nan_to_num(P)
    for t in range(T):
        E[:, t] = np.bincount(lab[mem], weights=P[:, t], minlength=m)
    return O, E


def _agg_sample(xy, ind, strata_levels, code=None, T: int = 0, radius=_AGG_RADIUS_UM,
                n_min=_AGG_MIN_CLASS, link=_AGG_LINK_UM, n_null=_AGG_N_NULL, alpha=_AGG_ALPHA,
                min_nb=_AGG_MIN_NB, n_control=_AGG_N_CONTROL, rim=_AGG_RIM_UM, seed=0,
                hull="convex", hull_alpha=_AGG_HULL_ALPHA_UM, progress=None) -> dict:
    """Aggregates of ONE section. xy: cells × 2 (µm); ind: class indicator; strata_levels:
    stratum codes per cell, finest first (the first is the permutation stratum); code / T:
    type code per cell (−1 unlabelled) and number of types, for the composition.

    Returns lab (aggregate per cell, −1 none, ordered by class cells descending), p, I, hot,
    tile (µm² per cell), table (one row per aggregate: n_cells, n_class, area_um2, cx, cy),
    O / E ([aggregates, T]), rings (per aggregate: list of rings), tissue_um2, control
    (aggregate counts and area of the permuted labellings), outside (reference mask)."""
    from scipy.spatial import cKDTree
    xy = np.asarray(xy, dtype=np.float64)
    ind = np.asarray(ind, dtype=bool)
    n = len(xy)
    prog = progress or (lambda f: None)
    det = _agg_detect(xy, ind, strata_levels[0], radius, n_null, n_control, seed,
                      progress=lambda f: prog(0.75 * f))
    hot = _agg_hot(det["I"], det["k"], det["n_null"], alpha, min_nb)
    edges = _agg_edges(xy, link)
    jobs._job_check()
    lab = _agg_components(hot, ind, edges, n_min)
    tile = _agg_tile_area(xy)
    prog(0.9)
    tissue = float(tile.sum())
    ctl_n, ctl_area = [], []
    for c in range(det["ind_nc"].shape[0]):
        h = _agg_hot(det["I_nc"][c], det["k_nc"][c], det["n_null"], alpha, min_nb)
        lc = _agg_components(h, det["ind_nc"][c], edges, n_min)
        ctl_n.append(int(lc.max()) + 1)
        ctl_area.append(float(tile[lc >= 0].sum() / tissue) if tissue > 0 else 0.0)
    m = int(lab.max()) + 1
    # order by class cells, largest first
    if m:
        n_class = np.bincount(lab[lab >= 0], weights=ind[lab >= 0].astype(float), minlength=m)
        n_all = np.bincount(lab[lab >= 0], minlength=m)
        order = np.lexsort((-n_all, -n_class))
        rank = np.empty(m, dtype=np.int64)
        rank[order] = np.arange(m)
        lab = np.where(lab >= 0, rank[np.clip(lab, 0, None)], -1)
    mem = lab >= 0
    outside = ~mem
    if mem.any() and rim > 0:
        d, _ = cKDTree(xy[mem]).query(xy, workers=-1)
        outside = ~mem & (d > float(rim))
    rows, rings = [], []
    for a in range(m):
        ix = np.flatnonzero(lab == a)
        rows.append({"n_cells": int(len(ix)), "n_class": int(ind[ix].sum()),
                     "area_um2": float(tile[ix].sum()), "cx": float(xy[ix, 0].mean()),
                     "cy": float(xy[ix, 1].mean())})
        rings.append(_agg_hull(xy[ix], hull, hull_alpha))
    table = pd.DataFrame(rows, columns=["n_cells", "n_class", "area_um2", "cx", "cy"])
    if code is not None and T:
        O, E = _agg_oe(lab, np.asarray(code), strata_levels, T, outside)
    else:
        O, E = np.zeros((m, 0)), np.zeros((m, 0))
    prog(1.0)
    return {"lab": lab, "p": det["p"], "I": det["I"], "hot": hot, "tile": tile, "table": table,
            "O": O, "E": E, "rings": rings, "tissue_um2": tissue, "outside": outside,
            "control": {"n": ctl_n, "area": ctl_area}, "n_hot": int(hot.sum())}


# ─── The job ──────────────────────────────────────────────────────────────────
def _agg_strata(table, seg_source, xy) -> tuple:
    """(stratum levels of one sample, finest first; True when the compartments had to be
    computed here with the defaults)."""
    n = len(xy)
    e, d, c = (controls._ctl_get(table, seg_source, k) for k in ("edge_distance_um", "local_density", "compartment"))
    computed = False
    if e is None or d is None or np.isnan(pd.to_numeric(pd.Series(e), errors="coerce")).any():
        if n >= 3:
            cols, _info = controls._compartments(xy)
        else:
            cols = {"edge_distance_um": np.zeros(n), "local_density": np.zeros(n),
                    "compartment": np.array(["other"] * n, dtype=object)}
        e, d, c = cols["edge_distance_um"], cols["local_density"], cols["compartment"]
        computed = True
    comp = pd.Series(c, dtype=object).fillna("other").astype(str).to_numpy()
    comp_code = np.unique(comp, return_inverse=True)[1]
    lv = controls._coloc_levels(np.zeros(n, dtype=np.int64), comp_code, np.asarray(e, dtype=np.float64),
                                np.asarray(d, dtype=np.float64))
    return lv[:3], computed


def _agg_across(values, conds, donors) -> dict:
    """Sample-level test of A − B for one value per sample, when the design has at least
    stats._SL_MIN_UNITS independent units per condition; else {"p": None, "why": …}."""
    c_use = [c if c in ("A", "B") else "" for c in conds]
    if "A" not in c_use or "B" not in c_use:
        return {"p": None, "why": "no samples are assigned to both conditions"}
    u = stats._sl_units(c_use, donors)
    k = (len(u["pairs"]) if u["paired"]
         else min(sum(1 for _n, c, _ix in u["units"] if c == x) for x in ("A", "B")))
    if k < stats._SL_MIN_UNITS:
        return {"p": None, "why": (f"fewer than {stats._SL_MIN_UNITS} independent units per condition "
                                   f"({u['note']})")}
    vals = np.asarray(values, dtype=np.float64)
    bad = [i for i, c in enumerate(c_use) if c and not np.isfinite(vals[i])]
    if bad:
        # never a test on a silently reduced set of samples
        return {"p": None, "why": (f"{len(bad)} sample(s) assigned to a condition have no value (no "
                                   f"labels, fewer than 3 cells, or no tissue area)")}
    r = stats._sl_test(vals, c_use, donors, units=u)
    note = r["units_note"]
    if r.get("dropped"):
        # a paired test uses the donors seen in both conditions: say who is not in it
        note += "; donors with samples in one condition only were left out: " + ", ".join(r["dropped"])
    return {"p": float(r["p"]), "effect": float(r["effect"]), "min_p": float(r["min_p"]),
            "test": r["test"], "exact": bool(r["exact"]), "units_note": note, "dropped": list(r.get("dropped") or [])}


def _run_aggregates(seg_source, method, label_kind, classes, params=None) -> None:
    """Background job: aggregates of the chosen class in every sample of the active view."""
    _gen = jobs._dataset_gen
    p = {"radius": _AGG_RADIUS_UM, "n_min": _AGG_MIN_CLASS, "link": _AGG_LINK_UM, "n_null": _AGG_N_NULL,
         "alpha": _AGG_ALPHA, "min_nb": _AGG_MIN_NB, "n_control": _AGG_N_CONTROL, "rim": _AGG_RIM_UM,
         "hull": "convex", "hull_alpha": _AGG_HULL_ALPHA_UM, "seed": 0}
    p.update({k: v for k, v in (params or {}).items() if v is not None and k in p})
    p["n_null"] = int(max(19, min(5000, p["n_null"])))
    p["n_control"] = int(max(0, min(200, p["n_control"])))

    def _set(status, message, result=None):
        with _agg_lock:
            if jobs._stale_job(_gen, status, message):
                return
            _agg_state["status"], _agg_state["message"] = status, message
            if result is not None:
                _agg_state["result"] = result
            if status in ("done", "error"):
                _agg_state["version"] += 1
        if status == "done":
            provenance._prov_record("aggregates", seg_source=seg_source, params={"classes": classes, **(params or {})}, labels=(label_kind, method), uses_design=True, message=message, result=result)

    try:
        _t0 = _time.time()
        _set("running", "Collecting cells and labels…")
        classes = [str(c) for c in (classes or [])]
        if not classes:
            _set("error", "Pick the cell types that make up the class (e.g. the immune types).")
            return
        analysis._an_require_seg(seg_source)
        tables = stats._view_tables(seg_source)
        seg_tag = controls._seg_tag(seg_source)
        labels, cat = stats._comp_labels(label_kind, tables, seg_source, method)
        if all(l is None for l in labels):
            _set("error", f"The active view has no {cat} labels: run Cell Type Annotation first, or use clusters.")
            return
        vc = controls._pooled_types(labels)
        missing = [c for c in classes if c not in vc.index]
        cls = [c for c in classes if c in vc.index]
        if not cls:
            _set("error", f"No cell of the view is labelled {', '.join(classes[:6])}.")
            return
        if p["alpha"] < 1.0 / (p["n_null"] + 1.0):
            _set("error", f"With {p['n_null']} permutations the smallest p is {1.0 / (p['n_null'] + 1):.3g}: "
                          f"no cell can reach α = {p['alpha']:g}. Raise the permutations or α.")
            return
        types = sorted(map(str, vc.index), key=str)
        tpos = {t: i for i, t in enumerate(types)}
        T, S = len(types), len(tables)
        per, defaults, tissue_of = [], [], {}
        unlabelled = []            # samples without any label: their density is unknown, not zero
        for si, (t, lab) in enumerate(zip(tables, labels)):
            jobs._job_check()
            xy = stats._xy_of(t["df"])
            n = len(xy)
            _set("running", f"Aggregates: {t['label']} ({n:,} cells)…")
            jobs._job_progress(f"{t['label']}: permuting labels within strata…", si / S)
            code = np.full(n, -1, dtype=np.int64)
            if lab is not None:
                code = pd.Series(lab, dtype=object).map(
                    lambda v: tpos.get(str(v), -1) if v is not None else -1).to_numpy(dtype=np.int64)
            ind = np.isin(code, [tpos[c] for c in cls])
            if n < 3 or not ind.any():
                # labelled, but no cell of the class: the sample has zero aggregates on its
                # tissue area (a value of the between-condition test, not a missing one).
                # No label at all (no annotation for this sample): nothing is known about
                # the class there - no value, and the test says so instead of using a 0.
                per.append(None)
                if not (code >= 0).any():
                    unlabelled.append(si)
                    tissue_of[si] = np.nan
                else:
                    tissue_of[si] = float(_agg_tile_area(xy).sum()) if n >= 3 else np.nan
                continue
            levels, computed = _agg_strata(t, seg_source, xy)
            if computed:
                defaults.append(t["label"])
            r = _agg_sample(xy, ind, levels, code, T, radius=p["radius"], n_min=p["n_min"], link=p["link"],
                            n_null=p["n_null"], alpha=p["alpha"], min_nb=p["min_nb"], n_control=p["n_control"],
                            rim=p["rim"], seed=int(p["seed"]) + 7919 * si, hull=p["hull"], hull_alpha=p["hull_alpha"],
                            progress=lambda f, _si=si: jobs._job_progress(fraction=(_si + f) / S))
            r.update({"ind": ind, "code": code, "xy": xy})
            per.append(r)
        jobs._check_dataset_gen(_gen)
        if all(r is None for r in per):
            _set("error", "No sample of the view has at least 3 cells and a cell of the class.")
            return
        jobs._job_shield("Writing the aggregates…")
        # ids over the view: by class cells, largest first
        rows = []
        for si, r in enumerate(per):
            if r is None:
                continue
            tb = r["table"]
            for a in range(len(tb)):
                rows.append({"si": si, "local": a, **tb.iloc[a].to_dict()})
        agg = pd.DataFrame(rows, columns=["si", "local", "n_cells", "n_class", "area_um2", "cx", "cy"])
        agg = agg.sort_values(["n_class", "n_cells", "si", "local"], ascending=[False, False, True, True],
                              kind="stable").reset_index(drop=True)
        agg.insert(0, "id", np.arange(1, len(agg) + 1))
        gid = {(int(r.si), int(r.local)): int(r.id) for r in agg.itertuples()}
        oe_rows, outlines, top_txt, enr_txt = [], [], {}, {}
        Otot, Etot = np.zeros(T), np.zeros(T)
        srows = []
        conds, _b = stats._design_of(tables)
        donors = stats._design_donors(tables)
        new_cols = {}
        for si, (t, r) in enumerate(zip(tables, per)):
            base = {"sample": t["label"], "name": t["name"], "cond": conds[si], "donor": donors[si],
                    "n_cells": int(len(t["df"]))}
            if r is None:
                tissue = tissue_of[si]
                has = bool(np.isfinite(tissue) and tissue > 0)
                srows.append({**base, "n_class": 0 if si not in unlabelled else np.nan,
                              "tissue_mm2": tissue / 1e6 if has else np.nan,
                              "n_aggregates": 0 if si not in unlabelled else np.nan,
                              "per_mm2": 0.0 if has else np.nan,
                              "area_fraction": 0.0 if has else np.nan, "share_class_inside": np.nan, "n_hotspot": 0,
                              "control_mean": np.nan, "control_max": np.nan, "false_rate": np.nan,
                              "control_area_max": np.nan})
                continue
            lab = r["lab"]
            m = int(lab.max()) + 1
            ids = np.full(len(lab), np.nan)
            if m:
                lut = np.array([gid[(si, a)] for a in range(m)], dtype=np.float64)
                ids[lab >= 0] = lut[lab[lab >= 0]]
            new_cols[t["key"]] = {"aggregate_id": ids, "aggregate_hotspot": r["hot"].astype(np.float32),
                                  "aggregate_p": r["p"].astype(np.float32)}
            mem = lab >= 0
            tissue = r["tissue_um2"]
            ctl = r["control"]
            n_ind = int(r["ind"].sum())
            srows.append({**base, "n_class": n_ind, "tissue_mm2": tissue / 1e6, "n_aggregates": m,
                          "per_mm2": m / (tissue / 1e6) if tissue > 0 else np.nan,
                          "area_fraction": float(r["tile"][mem].sum() / tissue) if tissue > 0 else np.nan,
                          "share_class_inside": float(r["ind"][mem].sum() / max(n_ind, 1)),
                          "n_hotspot": r["n_hot"],
                          "control_mean": float(np.mean(ctl["n"])) if ctl["n"] else np.nan,
                          "control_max": float(np.max(ctl["n"])) if ctl["n"] else np.nan,
                          "false_rate": (float(np.mean(ctl["n"]) / m) if ctl["n"] and m else np.nan),
                          "control_area_max": float(np.max(ctl["area"])) if ctl["area"] else np.nan})
            O, E = r["O"], r["E"]
            Otot += O.sum(axis=0)
            Etot += E.sum(axis=0)
            for a in range(m):
                g = gid[(si, a)]
                outlines.append({"id": g, "si": si, "key": t["key"], "x_offset": float(t.get("x_offset") or 0.0),
                                 "rings": r["rings"][a], "n_class": int(r["table"]["n_class"].iloc[a])})
                l2 = np.log2((O[a] + 0.5) / (E[a] + 0.5))
                order = np.argsort(-O[a], kind="stable")
                top_txt[g] = "; ".join(f"{types[j]} {int(O[a, j])}" for j in order[:4] if O[a, j] > 0)
                non = [j for j in np.argsort(-l2, kind="stable") if types[j] not in cls and O[a, j] >= 3][:3]
                enr_txt[g] = "; ".join(f"{types[j]} {l2[j]:+.1f}" for j in non if l2[j] > 0)
                for j in range(T):
                    if O[a, j] > 0 or E[a, j] >= 0.5:
                        oe_rows.append({"id": g, "sample": t["label"], "type": types[j], "in_class": types[j] in cls,
                                        "observed": float(O[a, j]), "expected": float(E[a, j]), "log2_oe": float(l2[j])})
        # every sample of the view: a sample without a result loses the columns of the last run
        controls._ctl_replace(tables, seg_source, _AGG_COLUMNS, new_cols, tag=seg_tag)
        agg["sample"] = [tables[int(i)]["label"] for i in agg["si"]]
        agg["class_fraction"] = agg["n_class"] / agg["n_cells"].clip(lower=1)
        agg["composition"] = [top_txt.get(int(i), "") for i in agg["id"]]
        agg["enriched_other"] = [enr_txt.get(int(i), "") for i in agg["id"]]
        samples = pd.DataFrame(srows)
        pooled = pd.DataFrame({"type": types, "in_class": [t in cls for t in types], "observed": Otot,
                               "expected": Etot, "log2_oe": np.log2((Otot + 0.5) / (Etot + 0.5))})
        pooled = pooled[(pooled["observed"] > 0) | (pooled["expected"] >= 0.5)].reset_index(drop=True)
        tests = {"per_mm2": _agg_across(samples["per_mm2"].to_numpy(), conds, donors),
                 "area_fraction": _agg_across(samples["area_fraction"].to_numpy(), conds, donors)} if S > 1 else {}
        n_agg = len(agg)
        res = {"aggregates": agg.drop(columns=["si", "local"]), "oe": pd.DataFrame(
                   oe_rows, columns=["id", "sample", "type", "in_class", "observed", "expected", "log2_oe"]),
               "samples": samples, "pooled": pooled, "outlines": outlines, "types": types, "classes": cls,
               "params": p, "category": cat, "seg_source": seg_source or "xenium",
               "seg_tag": seg_tag, "gen": _gen, "tests": tests, "n_samples": S,
               "seconds": _time.time() - _t0}
        n_lab = int(sum(int((r["code"] >= 0).sum()) for r in per if r is not None))
        _unl = {tables[si]["label"] for si in unlabelled}
        res["unlabelled"] = sorted(_unl)
        n_cells = int(sum(len(t["df"]) for t in tables))
        floor = 1.0 / (p["n_null"] + 1.0)
        msg = (f"{n_agg:,} aggregate(s) of ≥ {int(p['n_min'])} class cells in {S} sample(s); class = "
               f"{', '.join(cls[:8])}{'…' if len(cls) > 8 else ''} ({cat}); radius {p['radius']:g} µm, link ≤ "
               f"{p['link']:g} µm, hotspot: p ≤ {p['alpha']:g} of {p['n_null']} within-stratum permutations "
               f"(smallest possible p {floor:.3g}) and ≥ {int(p['min_nb'])} class cells within the radius; "
               f"seed {p['seed']} ({res['seconds']:.1f} s).\n"
               + "; ".join((f"{r['sample']}: no {cat} labels - not evaluated" if r["sample"] in _unl else
                            f"{r['sample']}: {r['n_aggregates']} aggregate(s)")
                           + (f", {r['per_mm2']:.2f} / mm², {100 * r['area_fraction']:.2f} % of "
                              f"{r['tissue_mm2']:.2f} mm² tissue, {100 * r['share_class_inside']:.0f} % of the "
                              f"{r['n_class']:,} class cells inside" if np.isfinite(r["tissue_mm2"]) else "")
                           + (f", control {r['control_mean']:.2f} (max {r['control_max']:.0f}) aggregates under "
                              f"permuted labels" if np.isfinite(r["control_mean"]) else "")
                           for r in srows)
               + f".\nUnit of the hotspot p-values: the cells of one section, exchangeable within strata (n = "
                 f"{n_lab:,} labelled of {n_cells:,} cells); they set the detection threshold and do not "
                 f"generalise to other sections or donors.")
        if S == 1:
            msg += ("\n⚠ SINGLE-SAMPLE RESULT: aggregate counts and areas describe this section; no test "
                    "across samples is possible with one sample.")
        else:
            for key, name in (("per_mm2", "aggregates per mm²"), ("area_fraction", "area fraction")):
                tt = tests[key]
                if tt["p"] is None:
                    msg += f"\nBetween conditions ({name}): no test - {tt['why']}."
                else:
                    msg += (f"\nBetween conditions ({name}): A − B = {tt['effect']:.3g}, p = {tt['p']:.3g} "
                            f"({tt['test']}{', exact' if tt['exact'] else ', Monte Carlo'}; unit of the p-value: "
                            f"{tt['units_note']}; smallest achievable p {tt['min_p']:.3g}).")
        if defaults:
            msg += (f"\nCompartments were not computed for {', '.join(defaults)}: edge distance and density "
                    f"were computed with the defaults and there is no core compartment (Confound Controls tab).")
        if missing:
            msg += f"\nNot in this view: {', '.join(missing[:8])}."
        if n_agg:
            msg += f"\nColor By: '{_AGG_COLOR_LABEL}'. Outlines are drawn on the map (both renderers)."
        _set("done", msg, result=res)
        print(f"  Aggregates: {msg}", flush=True)
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


# ─── Color By, outlines ───────────────────────────────────────────────────────
def _agg_color(name, seg_source="xenium", use_corrected: bool = False):
    """Colour provider (analysis._view_color_providers): the aggregate id of every cell of
    the view (NaN outside the aggregates), sequential scale 1 … number of aggregates."""
    res = _agg_result(seg_source)
    if res is None or not len(res["aggregates"]):
        return None
    parts, have = [], False
    for t in stats._view_tables(seg_source):
        v = controls._ctl_get(t, seg_source, "aggregate_id")
        if v is None:
            parts.append(np.full(len(t["df"]), np.nan, dtype=np.float32))
        else:
            have = True
            parts.append(pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(dtype=np.float32))
    if not have:
        return None
    vals = np.concatenate(parts) if len(parts) > 1 else parts[0]
    # the scale starts below 1 so that aggregate 1 is not drawn in the darkest colour
    k = float(max(2, len(res["aggregates"])))
    return vals, (-float(np.ceil(0.3 * k)), k), _AGG_COLOR_LABEL


def _agg_color_options() -> list:
    res = _agg_result()
    if res is None or not len(res["aggregates"]):
        return []
    return [{"label": f"Aggregates: {_AGG_COLOR_LABEL}", "value": f"geneset:{_AGG_COLOR_KEY}"}]


def _agg_visible_outlines(seg_source) -> list:
    """Outlines to draw for the active view: [(id, n_class, ring in µm with the sample's x
    offset, y as stored, True for the longest ring of the aggregate)]; nothing when hidden, when the result belongs to another
    segmentation or when a sample is no longer in the view."""
    with _agg_lock:
        show = _agg_state.get("show", True)
    res = _agg_result(seg_source) if show else None
    if res is None:
        return []
    off = {t["key"]: float(t.get("x_offset") or 0.0) for t in stats._view_tables(seg_source)}
    out = []
    for o in res["outlines"]:
        if o["key"] not in off:
            continue
        rings = sorted((r for r in o["rings"] if len(r) >= 3), key=len, reverse=True)
        for k, ring in enumerate(rings):
            out.append((o["id"], o["n_class"], [[x + off[o["key"]], y] for x, y in ring], k == 0))
    return out


def _agg_plotly_shapes(seg_source="xenium") -> list:
    """Layout shapes of the outlines for the Plotly map (y = −µm)."""
    return [{"type": "path",
             "path": "M " + " L ".join(f"{x:.1f},{-y:.1f}" for x, y in ring) + " Z",
             "fillcolor": "rgba(240,192,64,0.06)", "line": {"color": _AGG_OUTLINE_COLOR, "width": 1.5},
             "layer": "above", "xref": "x", "yref": "y"}
            for _id, _n, ring, _first in _agg_visible_outlines(seg_source)]


def _agg_viewer_polygons(seg_source="xenium") -> list:
    """The outlines for the WebGL view (µm, y down), labelled with the aggregate id."""
    return [{"points": [[float(x), float(y)] for x, y in ring], "color": _AGG_OUTLINE_COLOR, "width": 1.5,
             "fill": _AGG_OUTLINE_COLOR, "fillAlpha": 15, **({"label": f"A{_id}"} if first else {})}
            for _id, _n, ring, first in _agg_visible_outlines(seg_source)]


analysis._view_color_providers[_AGG_COLOR_KEY] = _agg_color
analysis._view_color_options.append(_agg_color_options)
figures._overlay_shape_providers.append(_agg_plotly_shapes)
figures._overlay_polygon_providers.append(_agg_viewer_polygons)


# ─── Figure, tables, CSV ──────────────────────────────────────────────────────
def make_agg_fig(width=None, height=None):
    """Left: aggregates per mm² per sample (bars) with the permuted-label control (markers).
    Right: log2 O/E of every cell type inside the aggregates against matched outside cells,
    pooled over the aggregates (class types in orange)."""
    from plotly.subplots import make_subplots
    res = _agg_result()
    if not res:
        return figures._make_placeholder_fig("No aggregates computed yet.")
    s, pooled = res["samples"], res["pooled"].sort_values("log2_oe", ascending=False)
    if not len(res["aggregates"]):
        return figures._make_placeholder_fig("No aggregate found with these settings.")
    fig = make_subplots(rows=1, cols=2, column_widths=[0.35, 0.65], horizontal_spacing=0.1,
                        subplot_titles=("Per mm² (◆ control)", "log2 O/E inside vs matched outside"))
    cond_col = {"A": "#f85149", "B": "#58a6ff"}
    fig.add_trace(go.Bar(x=s["sample"], y=s["per_mm2"],
                         marker_color=[cond_col.get(c, "#8b949e") for c in s["cond"]],
                         customdata=np.column_stack([s["n_aggregates"], 100 * s["area_fraction"], s["tissue_mm2"]]),
                         hovertemplate="%{x}<br>%{y:.2f} / mm²<br>%{customdata[0]} aggregates<br>"
                                       "%{customdata[1]:.2f} % of %{customdata[2]:.2f} mm²<extra></extra>"),
                  row=1, col=1)
    ctl = s["control_mean"] / s["tissue_mm2"]
    if np.isfinite(ctl).any():
        fig.add_trace(go.Scatter(x=s["sample"], y=ctl, mode="markers",
                                 marker=dict(symbol="diamond", size=9, color="#d29922"),
                                 hovertemplate="%{x}<br>control %{y:.2f} / mm²<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=pooled["type"], y=pooled["log2_oe"],
                         marker_color=["#d29922" if c else "#58a6ff" for c in pooled["in_class"]],
                         customdata=np.column_stack([pooled["observed"], pooled["expected"]]),
                         hovertemplate="%{x}<br>log2 O/E %{y:.2f}<br>observed %{customdata[0]:.0f}, expected "
                                       "%{customdata[1]:.1f}<extra></extra>"), row=1, col=2)
    fig.update_annotations(font=dict(size=10, color=core.TEXT))
    fig.update_xaxes(tickangle=-45, row=1, col=2)
    return controls._fig_layout(fig, f"{len(res['aggregates']):,} aggregates of {', '.join(res['classes'][:4])}"
                                     f"{'…' if len(res['classes']) > 4 else ''} (orange: class types)",
                                width, height, margin=dict(l=50, r=10, t=60, b=110))


_AGG_COLS = [("id", "ID"), ("sample", "Sample"), ("n_cells", "Cells"), ("n_class", "Class cells"),
             ("class_fraction", "Class fraction"), ("area_um2", "Area µm²"), ("cx", "x µm"), ("cy", "y µm"),
             ("composition", "Composition"), ("enriched_other", "Other types, log2 O/E")]
_AGG_SAMPLE_COLS = [("sample", "Sample"), ("cond", "Cond."), ("donor", "Donor"), ("n_cells", "Cells"),
                    ("n_class", "Class cells"), ("tissue_mm2", "Tissue mm²"), ("n_aggregates", "Aggregates"),
                    ("per_mm2", "per mm²"), ("area_fraction", "Area fraction"),
                    ("share_class_inside", "Class inside"), ("control_mean", "Control mean"),
                    ("control_max", "Control max"), ("false_rate", "False rate")]


def _agg_round(df, cols) -> list:
    out = pd.DataFrame({c: df[c] for c, _n in cols if c in df.columns})
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda v: None if not np.isfinite(v) else float(f"{v:.4g}"))
    return out.astype(object).where(out.notna(), None).to_dict("records")


def _agg_records(res) -> list:
    return _agg_round(res["aggregates"], _AGG_COLS)[:5000] if res else []


def _agg_sample_records(res) -> list:
    return _agg_round(res["samples"], _AGG_SAMPLE_COLS) if res else []


def _agg_csv(res) -> str:
    p = res["params"]
    lines = [f"# Aggregates; class: {' | '.join(res['classes'])}; labels: {res['category']}; segmentation: "
             f"{res['seg_source']}",
             f"# radius {p['radius']:g} um; link <= {p['link']:g} um; >= {int(p['n_min'])} class cells; hotspot: "
             f"p <= {p['alpha']:g} of {p['n_null']} within-stratum permutations and >= {int(p['min_nb'])} class "
             f"cells within the radius; {p['n_control']} control labellings; rim {p['rim']:g} um; hull "
             f"{p['hull']}; seed {p['seed']}",
             "# hotspot p: unit = cells of one section, exchangeable within strata (compartment x edge bin x "
             "density quintile); area = tissue nearest to the member cells (5 um grid, <= 20 um from a cell)",
             "# log2_oe = log2((observed + 0.5) / (expected + 0.5)); expected from the labelled cells of the "
             "same stratum outside every aggregate and its rim"]
    for key, name in (("per_mm2", "aggregates per mm2"), ("area_fraction", "area fraction")):
        tt = (res.get("tests") or {}).get(key)
        if tt:
            lines.append(f"# between conditions, {name}: "
                         + (f"no test ({tt['why']})" if tt["p"] is None else
                            f"A - B = {tt['effect']:.4g}; p = {tt['p']:.4g}; {tt['test']}; {tt['units_note']}; "
                            f"smallest achievable p {tt['min_p']:.4g}"))
    return provenance._prov_csv_stamp(res, (provenance._csv_comments(lines) + "# samples\n" + res["samples"].to_csv(index=False)
            + "\n# aggregates\n" + res["aggregates"].to_csv(index=False)
            + "\n# composition per aggregate\n" + res["oe"].to_csv(index=False)
            + "\n# composition pooled over the aggregates\n" + res["pooled"].to_csv(index=False)))


# ─── Layout ───────────────────────────────────────────────────────────────────
def _agg_tab():
    S, L = analysis._AN_ROW_STYLE, analysis._AN_LABEL_STYLE
    num = controls._num
    _text = ("sample", "composition", "enriched_other", "cond", "donor")
    cols = [{"id": c, "name": n, "type": "text" if c in _text else "numeric"} for c, n in _AGG_COLS]
    scols = [{"id": c, "name": n, "type": "text" if c in _text else "numeric"} for c, n in _AGG_SAMPLE_COLS]
    return dbc.Tab(label="Aggregates", tab_id="tab-agg",
                   label_style={"fontSize": "12px", "padding": "4px 12px"},
                   children=[html.Div([
        html.Div([
            html.Span("Cell types from:", style=L),
            dcc.Dropdown(id="agg-labels", options=[{"label": "Cluster (active cluster set)", "value": "cluster"}],
                         value=None, placeholder="Cell type annotation / cluster…",
                         style={"width": "220px", "fontSize": "11px"}),
            html.Span("Class:", style=L),
            dcc.Dropdown(id="agg-classes", options=[], value=[], multi=True,
                         placeholder="The cell types of the class, e.g. the immune types…",
                         style={"flex": "1", "minWidth": "240px", "fontSize": "11px"}),
            dbc.Button("Run", id="agg-run-btn", color="primary", size="sm", n_clicks=0,
                       style={"fontSize": "11px", "padding": "2px 12px"}),
        ], style=S),
        ui._options(html.Div([
            html.Span("Radius (µm):", style=L), num(id="agg-radius", value=_AGG_RADIUS_UM, min=5, max=1000, step=1),
            html.Span("≥ class cells:", style=L), num(id="agg-nmin", value=_AGG_MIN_CLASS, min=2, max=100000, step=1),
            html.Span("Link ≤ (µm):", style=L), num(id="agg-link", value=_AGG_LINK_UM, min=1, max=1000, step=1),
            html.Span("Permutations:", style=L), num(id="agg-nperm", value=_AGG_N_NULL, min=19, max=5000, step=1),
            html.Span("α:", style=L), num(id="agg-alpha", value=_AGG_ALPHA, min=0.0002, max=0.5, step=0.0001),
            html.Span("≥ class cells in radius:", style=L), num(id="agg-minnb", value=_AGG_MIN_NB, min=1, max=10000, step=1),
            html.Span("Outline:", style=L),
            dcc.Dropdown(id="agg-hull", options=[{"label": "convex hull", "value": "convex"},
                                                 {"label": "alpha hull (30 µm)", "value": "alpha"}],
                         value="convex", clearable=False, style={"width": "150px", "fontSize": "11px"}),
            dcc.Checklist(id="agg-outline-toggle",
                          options=[{"label": html.Span(" Show outlines on the map",
                                                       style={"fontSize": "11px", "color": core.TEXT}),
                                    "value": "on"}],
                          value=["on"], inputStyle={"marginRight": "3px"}, style={"display": "inline-flex"}),
        ], style=S), summary=f"Options (radius {_AGG_RADIUS_UM:g} µm, ≥ {_AGG_MIN_CLASS} class cells, link ≤ {_AGG_LINK_UM:g} µm, "
                             f"{_AGG_N_NULL} permutations, α {_AGG_ALPHA:g}, outlines on)"),
        html.Div([
            dbc.Button("Samples & conditions…", id="agg-design-open-btn", color="secondary", size="sm",
                       n_clicks=0, style={"fontSize": "11px", "padding": "2px 8px"}),
            html.Span(id="agg-design-summary", style=L),
        ], style=S),
        html.Div(id="agg-status", style={**analysis._AN_STATUS_STYLE, "whiteSpace": "pre-wrap"}),
        html.Div(_AGG_NOTE, id="agg-note", style=analysis._AN_NOTE_STYLE),
        html.Div([
            controls._graph(id="agg-plot", msg="No aggregates computed yet.", filename="xenium_aggregates",
                            min_h="320px"),
            html.Div([
                dash_table.DataTable(id="agg-table", data=[], columns=cols, **stats._dark_table_style(10)),
                html.Div([
                    dbc.Button("Download CSV", id="agg-download-btn", color="secondary", size="sm", n_clicks=0,
                               style={"fontSize": "11px", "padding": "2px 8px"}),
                    dcc.Download(id="agg-download"),
                ], style={"padding": "4px 0"}),
                html.Div("Per sample", style={"fontSize": "11px", "color": core.MUTED, "padding": "4px 0 2px 0"}),
                dash_table.DataTable(id="agg-sample-table", data=[], columns=scols, **stats._dark_table_style(8)),
            ], style={"flex": "1.1", "minWidth": "0", "overflow": "auto", "padding": "0 6px"}),
        ], style={"display": "flex", "flex": "1", "minHeight": "400px", "gap": "6px"}),
        ui._plot_dim_controls("agg"),
    ], style={"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0",
              "overflowY": "auto"})])
