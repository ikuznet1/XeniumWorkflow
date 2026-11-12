"""Segmentation-free abundance: marker transcripts per mm² of tissue straight from the
transcript file, with an area-bin sweep, the segmented cell density of the same type
beside it and concentration statistics against a redistribution null (compute + layout).

The transcripts of a sample are binned on that sample's own grid and compared with that
sample's own tissue mask: nothing here combines positions of two samples.
"""

import os
import re
import threading
import time as _time
import numpy as np
import pandas as pd
from dash import dcc, html
from dash import dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from . import core, jobs, ui, reseg, figures, analysis, stats, controls


# ─── State ────────────────────────────────────────────────────────────────────
_sfree_state: dict = {"status": "idle", "message": "", "result": None, "version": 0}
_sfree_lock = threading.Lock()

_SFREE_BINS = (25.0, 50.0, 100.0, 200.0)   # area-bin sweep (µm); multiples of the base grid
_SFREE_N_MARKERS = 5                       # markers derived from the annotation
_SFREE_HOLE_BINS = 0                       # interior holes of at most this many bins are tissue
_SFREE_N_NULL = 200                        # redistributions of the count over the tiles
_SFREE_MIN_TILES = 10                      # fewer tissue tiles: no concentration statistic
_SFREE_TOP_FRAC = 0.10                     # "top-tile share": share of the count in this fraction of the tiles
_SFREE_MIN_TX = 1                          # area from transcripts: a tile with at least this many is tissue
_SFREE_MAX_BINS = 8
_SFREE_PLOT_H = 340                        # px, unless a height is typed under the plot

_SFREE_NOTE = (
    "Marker transcripts per mm² of tissue, counted from the transcript file (quality Q ≥ 20, gene "
    "features only), not from the cells: no segmentation or cell-type call decides whether a "
    "transcript is counted. Markers: the most specific genes of the chosen cell type in the active "
    "annotation (≥ 50 counts per million in the type and ≥ 10× every other type; the top genes by "
    "ratio when fewer than 3 pass - the markers are then not exclusive), or the genes you list. "
    "Tissue area: the cells of the sample (active segmentation and subset) are put on a square "
    "grid and the occupied tiles are the tissue (optionally with small interior holes filled); or, "
    "with 'all transcripts', the tiles holding at least one gene transcript - then neither the "
    "numerator nor the denominator uses the segmentation, and a cell subset is not applied. The "
    "area of a tissue grows with the tile size (coarse tiles cover gaps and the margin), so the "
    "density is given for every tile size of the sweep: a difference between samples that changes "
    "sign or vanishes along the sweep is a property of the area estimate. Beside it: the density "
    "of the segmented cells of the type on the same area, the marker transcripts per segmented "
    "cell of the type, and the share of the marker transcripts on the tissue that the count matrix "
    "assigns to cells of the type (low: the transcripts sit in other cells or in none; read it at "
    "the coarse tiles - at fine ones a cell's transcripts can lie in a tile without a centroid). "
    "Concentration: Gini "
    "coefficient of the marker count over the tissue tiles and the share of the count in the top "
    "10 % of tiles, each minus its mean under a null that redistributes the same number of "
    "transcripts over the same tiles with equal probability (a sparse count gives a high Gini by "
    "itself; only the excess is clustering). p (one-sided) = (1 + redistributions with a Gini ≥ "
    "observed) / (1 + redistributions); unit: the tiles of one sample, so it describes that "
    "section only. Transcript density does not tell cell number from expression per cell. A sample "
    "loaded from a .zarr has no transcript file and is left out.")


def _sfree_reset() -> None:
    """Forget the result (another dataset was loaded; tests)."""
    with _sfree_lock:
        _sfree_state.update({"status": "idle", "message": "", "result": None})
        _sfree_state["version"] += 1


# ─── Small statistics ─────────────────────────────────────────────────────────
def _sfree_gini(v) -> float:
    """Gini coefficient of non-negative values (0: equal, → 1: all in one); NaN when the
    values are empty or sum to 0."""
    v = np.sort(np.asarray(v, dtype=np.float64).ravel())
    n = len(v)
    tot = float(v.sum())
    if n == 0 or tot <= 0:
        return float("nan")
    return float(2.0 * np.dot(np.arange(1, n + 1), v) / (n * tot) - (n + 1.0) / n)


def _sfree_top_share(v, frac: float = _SFREE_TOP_FRAC) -> float:
    """Share of the total held by the ceil(frac × n) largest values."""
    v = np.asarray(v, dtype=np.float64).ravel()
    tot = float(v.sum())
    if not len(v) or tot <= 0:
        return float("nan")
    k = max(1, int(np.ceil(frac * len(v))))
    return float(np.partition(v, len(v) - k)[len(v) - k:].sum() / tot)


def _sfree_concentration(counts, n_null: int = _SFREE_N_NULL, seed: int = 0,
                   min_tiles: int = _SFREE_MIN_TILES) -> dict:
    """Concentration of a count over tiles against a redistribution null.

    counts: transcripts per tissue tile. Null: the same total thrown onto the same tiles
    with equal probability (multinomial), *n_null* times. Returns gini, gini_null (mean),
    gini_excess (observed − null mean), gini_null_lo / _hi (2.5 / 97.5 %), gini_p
    (one-sided: (1 + draws ≥ observed) / (1 + draws)), top_share, top_share_null,
    top_share_excess, n_tiles. NaN when there are fewer than *min_tiles* tiles or no
    transcript."""
    c = np.asarray(counts, dtype=np.int64).ravel()
    out = {"n_tiles": int(len(c)), "gini": np.nan, "gini_null": np.nan, "gini_excess": np.nan,
           "gini_null_lo": np.nan, "gini_null_hi": np.nan, "gini_p": np.nan,
           "top_share": np.nan, "top_share_null": np.nan, "top_share_excess": np.nan}
    total = int(c.sum())
    if len(c) < int(min_tiles) or total <= 0:
        return out
    out["gini"], out["top_share"] = _sfree_gini(c), _sfree_top_share(c)
    n_null = int(max(0, n_null or 0))
    if not n_null:
        return out
    rng = np.random.default_rng(seed)
    p = np.full(len(c), 1.0 / len(c))
    g, t = np.empty(n_null), np.empty(n_null)
    for i in range(n_null):
        if i % 16 == 0:
            jobs._job_check()
        d = rng.multinomial(total, p)
        g[i], t[i] = _sfree_gini(d), _sfree_top_share(d)
    out.update({"gini_null": float(g.mean()), "gini_excess": float(out["gini"] - g.mean()),
                "gini_null_lo": float(np.quantile(g, 0.025)), "gini_null_hi": float(np.quantile(g, 0.975)),
                "gini_p": float((1.0 + (g >= out["gini"] - 1e-12).sum()) / (n_null + 1.0)),
                "top_share_null": float(t.mean()), "top_share_excess": float(out["top_share"] - t.mean())})
    return out


# ─── Tissue mask and grids ────────────────────────────────────────────────────
def _sfree_snap_bin(bin_um) -> float:
    """A tile size as a whole multiple of the base grid of the cached transcript grids."""
    b = analysis._DENSITY_BASE_UM
    return float(max(1, int(round(float(bin_um) / b))) * b)


def _sfree_parse_bins(text) -> list:
    """Tile sizes (µm) of the sweep from text ("25, 50, 100, 200"), snapped to multiples of
    the base grid, ascending, without repeats; the default sweep when the text is empty."""
    vals = []
    for tok in re.split(r"[,;\s]+", str(text or "").strip()):
        if not tok:
            continue
        try:
            v = float(tok)
        except ValueError:
            raise ValueError(f"'{tok}' is not a tile size in µm.") from None
        if not (analysis._DENSITY_BASE_UM <= v <= 5000):
            raise ValueError(f"Tile sizes must lie between {analysis._DENSITY_BASE_UM:g} and 5000 µm.")
        vals.append(_sfree_snap_bin(v))
    vals = sorted(set(vals)) or list(_SFREE_BINS)
    if len(vals) > _SFREE_MAX_BINS:
        raise ValueError(f"At most {_SFREE_MAX_BINS} tile sizes.")
    return vals


def _sfree_fill_holes(mask, hole_max_bins) -> tuple:
    """(mask with the interior holes of at most *hole_max_bins* tiles filled, holes filled).
    The rule of controls._edge_distance: a hole is an empty region not connected to the
    outside (4-connectivity); None fills every interior hole, 0 none."""
    if hole_max_bins is not None and int(hole_max_bins) <= 0:
        return mask, 0
    from scipy import ndimage
    g = np.pad(np.asarray(mask, dtype=bool), 1)
    lab, n = ndimage.label(~g)
    sizes = np.bincount(lab.ravel())
    outside = lab[0, 0]
    small = [i for i in range(1, n + 1)
             if i != outside and (hole_max_bins is None or sizes[i] <= int(hole_max_bins))]
    if small:
        g = g | np.isin(lab, small)
    return g[1:-1, 1:-1], len(small)


def _sfree_tissue_mask(xy, bin_um: float, hole_max_bins=_SFREE_HOLE_BINS) -> dict:
    """Tissue of ONE section from its cell positions: {mask (rows = y tiles, cols = x
    tiles), ix0, iy0 (tile index of mask[0, 0]), bin, n_occupied, holes_filled}.

    Tile (j, i) covers [i·bin, (i+1)·bin) × [j·bin, (j+1)·bin) µm - the tiles of
    analysis._density_grid at the same size, so a transcript grid can be laid on the
    mask. A tile with a cell centroid is tissue; see _sfree_fill_holes for the holes."""
    xy = np.asarray(xy, dtype=np.float64)
    b = float(bin_um)
    if not len(xy):
        return {"mask": np.zeros((0, 0), dtype=bool), "ix0": 0, "iy0": 0, "bin": b,
                "n_occupied": 0, "holes_filled": 0}
    ix = np.floor(xy[:, 0] / b).astype(np.int64)
    iy = np.floor(xy[:, 1] / b).astype(np.int64)
    ix0, iy0 = int(ix.min()), int(iy.min())
    mask = np.zeros((int(iy.max()) - iy0 + 1, int(ix.max()) - ix0 + 1), dtype=bool)
    mask[iy - iy0, ix - ix0] = True
    n_occ = int(mask.sum())
    mask, n_fill = _sfree_fill_holes(mask, hole_max_bins)
    return {"mask": mask, "ix0": ix0, "iy0": iy0, "bin": b, "n_occupied": n_occ, "holes_filled": n_fill}


def _sfree_mask_from_grid(g: dict, min_tx: int = _SFREE_MIN_TX, hole_max_bins=_SFREE_HOLE_BINS) -> dict:
    """Tissue of ONE section from its all-transcripts grid: tiles with at least *min_tx*
    transcripts (same dict as _sfree_tissue_mask)."""
    mask = np.asarray(g["grid"]) >= int(max(1, min_tx))
    n_occ = int(mask.sum())
    mask, n_fill = _sfree_fill_holes(mask, hole_max_bins)
    return {"mask": mask, "ix0": int(g["ix0"]), "iy0": int(g["iy0"]), "bin": float(g["bin"]),
            "n_occupied": n_occ, "holes_filled": n_fill}


def _sfree_grid_window(g, m: dict) -> np.ndarray:
    """The counts of transcript grid *g* (analysis._density_grid dict, or None) on the
    tiles of mask dict *m* (same tile size), 0 where the grid has no tile."""
    H, W = m["mask"].shape
    out = np.zeros((H, W), dtype=np.int64)
    if g is None or not H or not W or not g["grid"].size:
        return out
    if abs(float(g["bin"]) - float(m["bin"])) > 1e-9:
        raise ValueError("The transcript grid and the tissue mask have different tile sizes.")
    gh, gw = g["grid"].shape
    dy, dx = m["iy0"] - g["iy0"], m["ix0"] - g["ix0"]          # mask origin in grid tiles
    y0, y1 = max(0, dy), min(gh, dy + H)
    x0, x1 = max(0, dx), min(gw, dx + W)
    if y1 > y0 and x1 > x0:
        out[y0 - dy:y1 - dy, x0 - dx:x1 - dx] = g["grid"][y0:y1, x0:x1]
    return out


def _sfree_cells_in_mask(xy, m: dict) -> np.ndarray:
    """True for the cells whose centroid lies in a tissue tile of mask dict *m*."""
    xy = np.asarray(xy, dtype=np.float64)
    H, W = m["mask"].shape
    if not len(xy) or not H or not W:
        return np.zeros(len(xy), dtype=bool)
    ix = np.floor(xy[:, 0] / m["bin"]).astype(np.int64) - m["ix0"]
    iy = np.floor(xy[:, 1] / m["bin"]).astype(np.int64) - m["iy0"]
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    out = np.zeros(len(xy), dtype=bool)
    out[ok] = m["mask"][iy[ok], ix[ok]]
    return out


# ─── Transcript scan: several genes in one pass ───────────────────────────────
def _sfree_scan_genes(path: str, genes, qv_min=20, progress=None) -> dict:
    """{gene: scan} for several genes in ONE pass over transcripts.parquet; every scan is
    the dict analysis._density_scan returns for that gene (same filters: Q ≥ qv_min, gene
    features; same base bins), so it can be written with analysis._density_write."""
    import pyarrow as _pa
    import pyarrow.compute as _pc
    import pyarrow.dataset as _ds
    import pyarrow.parquet as _pq
    genes = list(dict.fromkeys(genes))
    dataset = _ds.dataset(path, format="parquet")
    names = dataset.schema.names
    flt = reseg._tx_gene_filters(names, qv_min)
    expr = _pq.filters_to_expression(flt) if flt else None
    ftype = dataset.schema.field("feature_name").type
    if _pa.types.is_dictionary(ftype):
        ftype = ftype.value_type
    as_bytes = _pa.types.is_binary(ftype) or _pa.types.is_large_binary(ftype)
    vals = _pa.array([g.encode() if as_bytes else g for g in genes], type=ftype)
    g_expr = _pc.field("feature_name").isin(vals)
    expr = g_expr if expr is None else (expr & g_expr)
    _OFF, _W = 1 << 20, 1 << 22
    keys = {g: [] for g in genes}
    cnts = {g: [] for g in genes}
    total = {g: 0 for g in genes}
    groups = [rg for frag in dataset.get_fragments() for rg in frag.split_by_row_group()]
    for i, rg in enumerate(groups):
        jobs._job_check()
        for batch in rg.to_batches(columns=["x_location", "y_location", "feature_name"], filter=expr,
                                   batch_size=1 << 20):
            if not batch.num_rows:
                continue
            nm = batch.column("feature_name")
            if _pa.types.is_dictionary(nm.type):
                nm = nm.dictionary_decode()
            nm = np.asarray(nm.to_numpy(zero_copy_only=False), dtype=object)
            if len(nm) and isinstance(nm[0], bytes):
                nm = np.array([v.decode() for v in nm], dtype=object)
            x = batch.column("x_location").to_numpy(zero_copy_only=False)
            y = batch.column("y_location").to_numpy(zero_copy_only=False)
            key = ((analysis._density_bin_index(y) + _OFF) * _W + (analysis._density_bin_index(x) + _OFF))
            for g in genes:
                sel = nm == g
                if sel.any():
                    k, c = np.unique(key[sel], return_counts=True)
                    keys[g].append(k.astype(np.int64)); cnts[g].append(c.astype(np.int64))
                    total[g] += int(sel.sum())
        if progress is not None:
            progress((i + 1) / max(1, len(groups)))
    out = {}
    z = np.zeros(0, dtype=np.int64)
    for g in genes:
        if not keys[g]:
            out[g] = {"iy": z, "ix": z.copy(), "n": z.copy(), "total": 0, "source": "parquet"}
            continue
        uk, inv = np.unique(np.concatenate(keys[g]), return_inverse=True)
        c = np.bincount(inv, weights=np.concatenate(cnts[g]), minlength=len(uk)).astype(np.int64)
        out[g] = {"iy": uk // _W - _OFF, "ix": uk % _W - _OFF, "n": c, "total": int(total[g]),
                  "source": "parquet"}
    return out


def _sfree_ensure_grids(ds: dict, genes, progress=None) -> list:
    """Make sure the base grid of every gene in *genes* (analysis._DENSITY_ALL allowed) is
    in the density cache of dataset *ds*; returns the genes that were scanned now. The
    panel genes missing from the cache are read in one pass over transcripts.parquet;
    bundles with transcripts.zarr.zip only, and the all-transcripts grid, go through
    analysis._density_scan."""
    data_dir = ds.get("data_dir")
    if not data_dir:
        raise ValueError("This sample was loaded from a .zarr: it has no transcript file.")
    missing = [g for g in dict.fromkeys(genes) if not os.path.exists(analysis._density_cache_file(g, ds))]
    if not missing:
        if progress is not None:
            progress(1.0)
        return []
    pq_path = os.path.join(data_dir, "transcripts.parquet")
    multi = [g for g in missing if g != analysis._DENSITY_ALL] if os.path.exists(pq_path) else []
    single = [g for g in missing if g not in multi]
    n_steps = (1 if multi else 0) + len(single)
    step = 0
    _p = (lambda f: progress((step + f) / n_steps)) if progress is not None else None
    if multi:
        scans = _sfree_scan_genes(pq_path, multi, progress=_p)
        jobs._job_check()
        for g in multi:
            analysis._density_write(analysis._density_cache_file(g, ds), scans[g], g)
        step += 1
    for g in single:
        scan = analysis._density_scan(data_dir, g, progress=_p)
        jobs._job_check()
        analysis._density_write(analysis._density_cache_file(g, ds), scan, g)
        step += 1
    return missing


def _sfree_marker_grid(genes, bin_um: float, ds: dict) -> tuple:
    """(grid dict of the summed marker transcripts at *bin_um*, {gene: transcripts in the
    whole sample}) from the cached per-gene grids of dataset *ds*."""
    parts, per_gene = [], {}
    for g in genes:
        d = analysis._density_grid(g, bin_um, ds)
        if d is None:
            raise ValueError(f"The transcript grid of {g} is missing from the cache.")
        per_gene[g] = int(d["total"])
        if d["grid"].size:
            parts.append(d)
    b = _sfree_snap_bin(bin_um)
    if not parts:
        return {"grid": np.zeros((0, 0), dtype=np.int64), "ix0": 0, "iy0": 0, "bin": b, "total": 0}, per_gene
    x0 = min(d["ix0"] for d in parts)
    y0 = min(d["iy0"] for d in parts)
    x1 = max(d["ix0"] + d["grid"].shape[1] for d in parts)
    y1 = max(d["iy0"] + d["grid"].shape[0] for d in parts)
    grid = np.zeros((y1 - y0, x1 - x0), dtype=np.int64)
    for d in parts:
        h, w = d["grid"].shape
        grid[d["iy0"] - y0:d["iy0"] - y0 + h, d["ix0"] - x0:d["ix0"] - x0 + w] += d["grid"]
    return {"grid": grid, "ix0": x0, "iy0": y0, "bin": b, "total": int(grid.sum())}, per_gene


# ─── One sample ───────────────────────────────────────────────────────────────
def _sfree_sample(xy, is_type, genes, ds: dict, bins, area_mode: str = "cells",
                  hole_max_bins=_SFREE_HOLE_BINS, n_null: int = _SFREE_N_NULL, seed: int = 0,
                  min_tx: int = _SFREE_MIN_TX, progress=None) -> tuple:
    """Segmentation-free abundance of ONE sample: (rows - one per tile size -, {gene:
    transcripts in the sample}).

    xy: cell centroids of the view (µm); is_type: the cells of the chosen type (None: no
    annotation, no cell density); genes: the markers, whose grids are in the density cache
    of *ds* (_sfree_ensure_grids); area_mode "cells": tissue = tiles with a cell (_sfree_tissue_mask),
    "transcripts": tiles with at least *min_tx* gene transcripts (the all-transcripts grid
    must be cached). Row: bin_um, n_tiles, area_mm2, n_tx (marker transcripts on tissue),
    n_tx_sample (in the whole sample), frac_tx_on_tissue, tx_per_mm2, n_cells_type,
    cells_per_mm2, tx_per_cell, tx_per_10k (of all gene transcripts on the tissue; NaN
    without the all-transcripts grid), the _sfree_concentration values, holes_filled."""
    rows, per_gene = [], {}
    all_ready = os.path.exists(analysis._density_cache_file(analysis._DENSITY_ALL, ds))
    for bi, b in enumerate(bins):
        jobs._job_check()
        g, per_gene = _sfree_marker_grid(genes, b, ds)
        g_all = analysis._density_grid(analysis._DENSITY_ALL, b, ds) if all_ready else None
        if area_mode == "transcripts":
            if g_all is None:
                raise ValueError("The all-transcripts grid is missing from the cache.")
            m = _sfree_mask_from_grid(g_all, min_tx, hole_max_bins)
        else:
            m = _sfree_tissue_mask(xy, b, hole_max_bins)
        counts = _sfree_grid_window(g, m)[m["mask"]]
        n_tiles = int(m["mask"].sum())
        area = n_tiles * (float(b) / 1000.0) ** 2
        n_tx = int(counts.sum())
        row = {"bin_um": float(b), "n_tiles": n_tiles, "area_mm2": area, "n_tx": n_tx,
               "n_tx_sample": int(g["total"]),
               "frac_tx_on_tissue": n_tx / g["total"] if g["total"] else np.nan,
               "tx_per_mm2": n_tx / area if area > 0 else np.nan,
               "n_cells_type": np.nan, "cells_per_mm2": np.nan, "tx_per_cell": np.nan,
               "tx_per_10k": np.nan, "holes_filled": int(m["holes_filled"])}
        if is_type is not None:
            n_c = int((np.asarray(is_type, dtype=bool) & _sfree_cells_in_mask(xy, m)).sum())
            row["n_cells_type"] = n_c
            row["cells_per_mm2"] = n_c / area if area > 0 else np.nan
            row["tx_per_cell"] = n_tx / n_c if n_c else np.nan
        if g_all is not None:
            tot_all = int(_sfree_grid_window(g_all, m)[m["mask"]].sum())
            row["tx_per_10k"] = 1e4 * n_tx / tot_all if tot_all else np.nan
        row.update(_sfree_concentration(counts, n_null, seed))
        rows.append(row)
        if progress is not None:
            progress((bi + 1) / len(bins))
    return rows, per_gene


def _sfree_across(table: pd.DataFrame, labels, conds, donors) -> tuple:
    """Sample-level tests A − B of the densities, per tile size: (rows, note). A test is
    run only with at least stats._SL_MIN_UNITS independent units per condition (pairs,
    when the design is paired); otherwise no rows and the reason."""
    if not ("A" in conds and "B" in conds):
        return [], ""
    u = stats._sl_units(conds, donors)
    k = (len(u["pairs"]) if u["paired"]
         else min(sum(1 for _n, c, _ix in u["units"] if c == x) for x in ("A", "B")))
    if k < stats._SL_MIN_UNITS:
        return [], (f"no test between conditions: fewer than {stats._SL_MIN_UNITS} independent units "
                    f"per condition ({u['note']})")
    pos = {lab: i for i, lab in enumerate(labels)}
    rows, test = [], ""
    for b, sub in table.groupby("bin_um"):
        for metric in ("tx_per_mm2", "cells_per_mm2", "tx_per_10k", "gini_excess"):
            v = np.full(len(labels), np.nan)
            for lab, val in zip(sub["sample"], sub[metric]):
                v[pos[lab]] = val
            used = [i for i, c in enumerate(conds) if c in ("A", "B")]
            if not np.isfinite(v[used]).all():
                continue                       # a sample without the value: no test of this metric
            r = stats._sl_test(v, conds, donors, units=u)
            test = r["test"]
            rows.append({"bin_um": float(b), "metric": metric, "effect": r["effect"], "p": r["p"],
                         "min_p": r["min_p"], "n_units": r["n_units"], "test": r["test"],
                         "exact": r["exact"]})
    return rows, f"A − B: {test or 'sample-level'} test, {u['note']}"


# ─── The job ──────────────────────────────────────────────────────────────────
def _sfree_parse_markers(text, available) -> tuple:
    """(genes on the panel, names not found) from a free-text marker list."""
    names = [t for t in re.split(r"[,;\s]+", str(text or "").strip()) if t]
    return analysis._gs_match(names, available)


def _run_segfree(seg_source, method, label_kind, cell_type, markers_text, n_markers, use_corrected,
                 bins_text=None, area_mode="cells", hole_max_bins=_SFREE_HOLE_BINS,
                 n_null=_SFREE_N_NULL, seed=0) -> None:
    """Background job: segmentation-free abundance of one cell type's markers in every
    sample of the active view."""
    _gen = jobs._dataset_gen

    def _set(status, message, result=None):
        with _sfree_lock:
            if jobs._stale_job(_gen, status, message):
                return
            _sfree_state["status"], _sfree_state["message"] = status, message
            if result is not None:
                _sfree_state["result"] = result
            if status in ("done", "error"):
                _sfree_state["version"] += 1

    try:
        _t0 = _time.time()
        _set("running", "Collecting cells and markers…")
        bins = _sfree_parse_bins(bins_text)
        area_mode = "transcripts" if area_mode == "transcripts" else "cells"
        n_null = int(max(0, min(5000, _SFREE_N_NULL if n_null is None else n_null)))
        hole = None if hole_max_bins is None else int(max(0, hole_max_bins))
        tables = stats._view_tables(seg_source)
        samples = analysis._an_samples(seg_source, use_corrected)
        by_key = {s["key"]: s for s in samples}
        labels, cat = [None] * len(tables), ""
        if label_kind:
            labels, cat = stats._comp_labels(label_kind, tables, seg_source, method)
        ctype = str(cell_type) if cell_type not in (None, "") else None
        if ctype is not None:
            vc = controls._pooled_types(labels)
            if ctype not in vc.index:
                _set("error", f"No cell of the view is labelled '{ctype}'.")
                return
        # markers: listed by the user, or derived from the type profiles of the view
        panel = analysis._an_common_genes(samples) if samples else list(core.DATA.get("gene_names") or [])
        user, missing = _sfree_parse_markers(markers_text, panel)
        info = {"rule": "user", "n_strict": 0}
        if user:
            markers = user
        elif str(markers_text or "").strip():
            _set("error", "None of the listed marker genes is on the panel of this view.")
            return
        else:
            if ctype is None:
                _set("error", "Pick a cell type (its markers are derived from the annotation), or list "
                              "marker genes.")
                return
            if len(samples) != len(tables):
                _set("error", "The view has no expression matrix to derive markers from: list the "
                              "marker genes.")
                return
            prof = controls._type_profiles(samples, labels)
            markers, info = controls._markers_of(prof, ctype, n=int(max(1, n_markers or _SFREE_N_MARKERS)))
            if not markers:
                _set("error", f"No gene reaches {controls._MARKER_MIN_CPM:g} counts per million in "
                              f"'{ctype}': list the marker genes.")
                return
        jobs._job_check()
        need = list(markers) + ([analysis._DENSITY_ALL] if area_mode == "transcripts" else [])
        usable = [i for i, t in enumerate(tables) if t["ds"].get("data_dir")]
        skipped = [tables[i]["label"] for i in range(len(tables)) if i not in usable]
        if not usable:
            _set("error", "Segmentation-free abundance needs the Xenium output directory "
                          "(transcripts.parquet or transcripts.zarr.zip); "
                          + ("this dataset was" if len(tables) == 1 else "these samples were")
                          + " loaded from a .zarr, which has cells but no transcripts.")
            return
        rows, gene_rows, scanned = [], [], []
        for k, i in enumerate(usable):
            t = tables[i]
            lo, span = k / len(usable), 1.0 / len(usable)
            _set("running", f"Reading marker transcripts of {t['label']}…")
            got = _sfree_ensure_grids(t["ds"], need, progress=lambda f, _lo=lo, _sp=span: jobs._job_progress(
                f"Reading transcripts ({t['label']})…", _lo + 0.6 * _sp * f))
            scanned += list(got)
            jobs._check_dataset_gen(_gen)
            _set("running", f"Tissue area and concentration of {t['label']}…")
            xy = stats._xy_of(t["df"])
            lab = labels[i]
            is_type = None
            if ctype is not None and lab is not None:
                is_type = pd.Series(lab, dtype=object).astype(str).to_numpy() == ctype
            elif ctype is not None:
                is_type = np.zeros(len(xy), dtype=bool)
            if len(xy) < 1 and area_mode == "cells":
                continue
            r_s, per_gene = _sfree_sample(
                xy, is_type, markers, t["ds"], bins, area_mode, hole, n_null, seed + i,
                progress=lambda f, _lo=lo, _sp=span: jobs._job_progress(
                    f"Tiles of {t['label']}…", _lo + _sp * (0.6 + 0.4 * f)))
            # what the count matrix assigns to cells (of the type / of any type)
            in_type = in_cells = np.nan
            s = by_key.get(t["key"]) if len(samples) == len(tables) else None
            if s is not None and len(s["df"]) == len(xy):
                m_cnt, _tot = controls._marker_counts(s, markers)
                in_cells = float(np.nansum(m_cnt))
                if is_type is not None:
                    in_type = float(np.nansum(m_cnt[is_type]))
            for r in r_s:
                r.update({"sample": t["label"], "name": t["name"],
                          "assigned_to_type": in_type / r["n_tx"] if r["n_tx"] else np.nan,
                          "assigned_to_cells": in_cells / r["n_tx"] if r["n_tx"] else np.nan})
            rows += r_s
            ref = r_s[min(len(r_s) - 1, max(0, len(r_s) // 2 - 1))]
            for g, n in per_gene.items():
                gene_rows.append({"sample": t["label"], "gene": g, "n_tx_sample": n,
                                  "share_of_markers": n / max(1, sum(per_gene.values())),
                                  "tx_per_mm2_sample_area": n / ref["area_mm2"] if ref["area_mm2"] else np.nan,
                                  "area_bin_um": ref["bin_um"]})
        jobs._check_dataset_gen(_gen)
        if not rows:
            _set("error", "No sample of the view has cells to define the tissue.")
            return
        first = ["sample", "name", "bin_um"]
        table = pd.DataFrame(rows)
        table = table[first + [c for c in table.columns if c not in first]]
        conds, _b = stats._design_of(tables)
        donors = stats._design_donors(tables)
        used_labels = [tables[i]["label"] for i in usable]
        across, across_note = _sfree_across(table, used_labels, [conds[i] for i in usable],
                                            [donors[i] for i in usable])
        corrected = bool(samples and samples[0]["corrected"])
        res = {"table": table, "genes": pd.DataFrame(gene_rows), "markers": list(markers),
               "marker_info": info, "markers_missing": missing, "cell_type": ctype, "category": cat,
               "bins": bins, "area_mode": area_mode, "hole_max_bins": hole, "n_null": n_null, "seed": seed,
               "seg_source": seg_source or "xenium", "corrected": corrected, "skipped": skipped,
               "across": pd.DataFrame(across), "across_note": across_note,
               "samples": [{"label": tables[i]["label"], "name": tables[i]["name"], "cond": conds[i],
                            "donor": donors[i], "n_cells": int(len(tables[i]["df"]))} for i in usable],
               "seconds": _time.time() - _t0}
        src = ("listed" if info["rule"] == "user" else
               f"derived from {cat}, rule '{info['rule']}'"
               + (" - fewer than 3 genes are ≥ 10× every other type, so the markers are NOT exclusive "
                  "to the type" if info["rule"] == "fallback" else ""))
        area_txt = ("tiles with a cell of the active segmentation"
                    + (" and subset" if "_df_original" in core.DATA else "")
                    if area_mode == "cells" else
                    f"tiles with ≥ {_SFREE_MIN_TX} gene transcript (segmentation-free; a cell subset is not applied)")
        msg = (f"Markers ({src}): {', '.join(markers)}"
               + (f" of '{ctype}'" if ctype else "") + f". Tissue area: {area_txt}"
               + (f", interior holes ≤ {hole} tiles filled" if hole else
                  ", every interior hole filled" if hole is None else "")
               + f"; tile sizes {', '.join(f'{b:g}' for b in bins)} µm; {len(usable)} sample(s) "
                 f"({res['seconds']:.1f} s).")
        for lab in used_labels:
            sub = table[table["sample"] == lab]
            lo, hi = sub.iloc[0], sub.iloc[-1]
            msg += (f"\n{lab}: {int(lo['n_tx_sample']):,} marker transcripts in the sample; "
                    f"{lo['tx_per_mm2']:,.0f} per mm² at {lo['bin_um']:g} µm tiles ({lo['area_mm2']:.2f} mm²) → "
                    f"{hi['tx_per_mm2']:,.0f} at {hi['bin_um']:g} µm ({hi['area_mm2']:.2f} mm²)")
            if ctype and np.isfinite(lo["cells_per_mm2"]):
                msg += (f"; segmented '{ctype}': {int(lo['n_cells_type']):,} cells, {lo['cells_per_mm2']:,.1f} → "
                        f"{hi['cells_per_mm2']:,.1f} per mm²")
            if np.isfinite(hi.get("assigned_to_type", np.nan)):
                msg += (f"; the count matrix assigns {100 * hi['assigned_to_type']:.0f} % of the marker "
                        f"transcripts on the tissue ({hi['bin_um']:g} µm tiles) to cells of the type "
                        f"({100 * hi['assigned_to_cells']:.0f} % to any cell)")
            ge = sub["gini_excess"].to_numpy(dtype=float)
            if np.isfinite(ge).any():
                msg += (f"; Gini − null: {np.nanmin(ge):+.3f} to {np.nanmax(ge):+.3f} over the sweep")
            msg += "."
        if len(gene_rows):
            gt = res["genes"].groupby("gene")["n_tx_sample"].sum().sort_values(ascending=False)
            if len(gt) > 1 and gt.sum() > 0 and gt.iloc[0] / gt.sum() > 0.6:
                msg += (f"\nNote: {gt.index[0]} alone gives {100 * gt.iloc[0] / gt.sum():.0f} % of the marker "
                        f"transcripts: the result is that gene's (per-gene counts are in the CSV).")
        if n_null:
            n_min = int(table["n_tiles"].min())
            msg += (f"\nConcentration: Gini over the tissue tiles minus the mean of {n_null} equal-probability "
                    f"redistributions of the same count over the same tiles (seed {seed}); p one-sided, "
                    f"smallest achievable {1.0 / (n_null + 1):.3g}; unit: the tiles of one sample "
                    f"(n = {n_min:,} to {int(table['n_tiles'].max()):,} tiles), so it describes that section only.")
        if across_note:
            msg += f"\nBetween conditions ({across_note})"
            if len(across):
                a = res["across"]
                d = a[a["metric"] == "tx_per_mm2"]
                msg += (f": transcripts per mm², A − B = "
                        + ", ".join(f"{e:+,.0f} (p {p:.3g}) at {b:g} µm" for b, e, p in zip(d["bin_um"], d["effect"], d["p"]))
                        + f"; smallest achievable p {float(a['min_p'].min()):.3g}; unit: the independent "
                          f"samples / donors named above, not cells or tiles.")
                reach = stats._sl_reach(float(a["min_p"].min()), max(1, len(d)))
                if reach:
                    msg += f"\n⚠ {reach[0].upper() + reach[1:]}."
                if len(d) and (np.sign(d["effect"]).nunique() > 1):
                    msg += " ⚠ The sign of the difference changes along the sweep: it depends on the area estimate."
            else:
                msg += "."
        elif len(usable) > 1:
            msg += ("\nNo test between conditions: assign the samples to conditions A and B ('Samples & "
                    "conditions…'); the rows are descriptive.")
        else:
            msg += ("\nSingle sample: the densities are descriptive; nothing here is a test across samples.")
        if missing:
            msg += f"\nNot on the panel: {', '.join(missing[:12])}."
        if skipped:
            msg += (f"\nLeft out (loaded from a .zarr, no transcript file): {', '.join(skipped)}.")
        if area_mode == "transcripts" and "_df_original" in core.DATA:
            msg += "\nNote: a cell subset is active; with the area from all transcripts it is not applied."
        if use_corrected and not corrected and samples:
            msg += "\nNote: original counts were used for the markers and the assigned share."
        _set("done", msg, result=res)
        print(f"  Segmentation-free abundance: {msg}", flush=True)
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
_SFREE_COLS = [("sample", "Sample"), ("bin_um", "Tile µm"), ("area_mm2", "Area mm²"),
               ("n_tx", "Marker tx"), ("tx_per_mm2", "Tx / mm²"), ("tx_per_10k", "Tx / 10k tx"),
               ("n_cells_type", "Cells (seg.)"), ("cells_per_mm2", "Cells / mm²"),
               ("tx_per_cell", "Tx / cell"), ("assigned_to_type", "In type's cells"),
               ("assigned_to_cells", "In any cell"), ("gini", "Gini"), ("gini_null", "Gini null"),
               ("gini_excess", "Gini − null"), ("gini_p", "p (tiles)"),
               ("top_share_excess", "Top 10 % − null"), ("n_tiles", "Tiles")]
_SFREE_PALETTE = ["#58a6ff", "#d29922", "#3fb950", "#f85149", "#bc8cff", "#39c5cf", "#ff7b72", "#e3b341"]


def _sfree_records(res) -> list:
    if not res:
        return []
    t = res["table"]
    out = pd.DataFrame({c: t[c] for c, _n in _SFREE_COLS if c in t.columns})
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda v: None if not np.isfinite(v) else float(f"{v:.4g}"))
    return out.astype(object).where(out.notna(), None).to_dict("records")


def make_sfree_fig(width=None, height=None):
    """Three panels against the tile size, one line per sample: marker transcripts per
    mm², segmented cells of the type per mm², Gini minus its redistribution null."""
    from plotly.subplots import make_subplots
    with _sfree_lock:
        res = _sfree_state.get("result")
    if not res:
        fig = figures._make_placeholder_fig("No segmentation-free abundance computed yet.")
        fig.update_layout(height=height or 160)
        if width:
            fig.update_layout(width=width)
        return fig
    t = res["table"]
    ct = f"'{res['cell_type']}'" if res["cell_type"] else "(no type picked)"
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
                        subplot_titles=("Marker transcripts per mm² (transcript file)",
                                        f"Segmented cells of {ct} per mm²",
                                        "Gini over tiles − redistribution null"))
    for k, (lab, sub) in enumerate(t.groupby("sample", sort=False)):
        col = _SFREE_PALETTE[k % len(_SFREE_PALETTE)]
        x = sub["bin_um"].to_numpy()
        for j, (c, fmt) in enumerate((("tx_per_mm2", ":,.0f"), ("cells_per_mm2", ":,.1f"), ("gini_excess", ":+.3f"))):
            fig.add_trace(go.Scatter(
                x=x, y=sub[c], mode="lines+markers", name=str(lab), legendgroup=str(lab), showlegend=(j == 0),
                line=dict(color=col, width=1.5), marker=dict(color=col, size=6),
                customdata=np.column_stack([sub["area_mm2"], sub["n_tx"]]),
                hovertemplate=(f"{lab}<br>tile %{{x:g}} µm<br>%{{y{fmt}}}<br>area %{{customdata[0]:.2f}} mm²"
                               "<br>%{customdata[1]:,} marker transcripts<extra></extra>")), row=1, col=j + 1)
    fig.add_hline(y=0, line=dict(color=core.MUTED, width=1, dash="dot"), row=1, col=3)
    fig.update_annotations(font=dict(size=10, color=core.TEXT))
    fig.update_xaxes(title_text="tile size (µm)", type="log", tickvals=res["bins"],
                     ticktext=[f"{b:g}" for b in res["bins"]])
    for j in (1, 2):
        fig.update_yaxes(rangemode="tozero", row=1, col=j)
    fig = controls._fig_layout(
        fig, f"Segmentation-free abundance: {', '.join(res['markers'][:6])}{'…' if len(res['markers']) > 6 else ''}"
             + (f" ({res['cell_type']})" if res["cell_type"] else ""), width, height or _SFREE_PLOT_H,
        margin=dict(l=50, r=10, t=60, b=50))
    fig.update_layout(showlegend=t["sample"].nunique() > 1,
                      legend=dict(orientation="h", y=-0.25, font=dict(size=10)))
    return fig


def _sfree_csv(res) -> str:
    info = res["marker_info"]
    lines = [f"# Segmentation-free abundance; cell type: {res['cell_type'] or '-'}; labels: {res['category'] or '-'}; "
             f"segmentation: {res['seg_source']}; counts (markers / assigned share): "
             f"{'SPLIT-corrected' if res['corrected'] else 'original'}",
             f"# markers ({info['rule']}): {' '.join(res['markers'])}",
             f"# tissue area: {'tiles with a cell centroid' if res['area_mode'] == 'cells' else 'tiles with a gene transcript'}; "
             f"interior holes filled up to {res['hole_max_bins']} tiles; area_mm2 = n_tiles x (bin_um / 1000)^2",
             "# n_tx: marker transcripts (Q >= 20) on tissue tiles, from the transcript file; tx_per_mm2 = n_tx / area_mm2; "
             "tx_per_10k: per 10,000 gene transcripts on the same tiles (needs the all-transcripts grid)",
             "# n_cells_type / cells_per_mm2: segmented cells of the type on the same tiles; tx_per_cell = n_tx / n_cells_type",
             "# assigned_to_type / assigned_to_cells: marker counts of the count matrix in the view's cells of the type / "
             "in all its cells, divided by n_tx (above 1 at small tiles: transcripts of a cell in a tile without a centroid)",
             f"# gini: Gini of n_tx over the tissue tiles; gini_null: mean over {res['n_null']} equal-probability "
             f"redistributions of the same count over the same tiles (seed {res['seed']}); gini_excess = gini - gini_null; "
             f"gini_p one-sided, unit = tiles of one sample (n_tiles); top_share: share of n_tx in the top "
             f"{100 * _SFREE_TOP_FRAC:g} % of tiles"]
    for s in res["samples"]:
        lines.append(f"# sample {s['label']} ({s['name']}): {s['n_cells']} cells; condition {s['cond'] or '-'}; "
                     f"donor {s['donor'] or '-'}")
    if res.get("skipped"):
        lines.append(f"# left out (no transcript file): {' '.join(res['skipped'])}")
    txt = "\n".join(lines) + "\n" + res["table"].to_csv(index=False)
    if len(res["genes"]):
        txt += "\n# per marker gene (whole sample; area of the tile size named in area_bin_um)\n" + res["genes"].to_csv(index=False)
    if len(res["across"]):
        txt += f"\n# between conditions: {res['across_note']}\n" + res["across"].to_csv(index=False)
    elif res.get("across_note"):
        txt += f"\n# between conditions: {res['across_note']}\n"
    return txt


# ─── Layout ───────────────────────────────────────────────────────────────────
def _sfree_section():
    """The "Segmentation-free abundance" block of the Cell Type Abundance tab."""
    S, L = analysis._AN_ROW_STYLE, analysis._AN_LABEL_STYLE
    cols = [{"id": c, "name": n, "type": "text" if c == "sample" else "numeric"} for c, n in _SFREE_COLS]
    return html.Div([
        html.Div("Segmentation-free abundance (marker transcripts per mm² of tissue)",
                 style={"fontSize": "12px", "color": core.TEXT, "fontWeight": "600", "padding": "6px 8px 0 8px",
                        "borderTop": f"1px solid {core.BORDER}"}),
        html.Div([
            html.Span("Cell types from:", style=L),
            dcc.Dropdown(id="sfree-labels", options=[{"label": "Cluster (active cluster set)", "value": "cluster"}],
                         value=None, placeholder="Cell type annotation / cluster…",
                         style={"flex": "1", "minWidth": "170px", "fontSize": "11px"}),
            html.Span("Type:", style=L),
            dcc.Dropdown(id="sfree-type", options=[], value=None, placeholder="Cell type…",
                         style={"width": "230px", "fontSize": "11px"}),
            html.Span("Markers to derive:", style=L),
            controls._num(id="sfree-n-markers", value=_SFREE_N_MARKERS, min=1, max=50, step=1),
            dbc.Button("Run", id="sfree-run-btn", color="primary", size="sm", n_clicks=0,
                       style={"fontSize": "11px", "padding": "2px 12px"}),
        ], style=S),
        html.Div([
            html.Span("Tile sizes (µm):", style=L),
            dcc.Input(id="sfree-bins", type="text", value=", ".join(f"{b:g}" for b in _SFREE_BINS), debounce=True,
                      style={**analysis._AN_INPUT_STYLE, "width": "130px"}),
            html.Span("Tissue area from:", style=L),
            dcc.Dropdown(id="sfree-area", clearable=False, value="cells",
                         options=[{"label": "cells (tiles with a cell)", "value": "cells"},
                                  {"label": "all transcripts (tiles with a transcript)", "value": "transcripts"}],
                         style={"width": "250px", "fontSize": "11px"}),
            html.Span("Fill holes ≤ (tiles):", style=L),
            controls._num(id="sfree-hole", value=_SFREE_HOLE_BINS, min=0, max=100000, step=1),
            html.Span("Redistributions:", style=L),
            controls._num(id="sfree-nnull", value=_SFREE_N_NULL, min=0, max=5000, step=10),
        ], style=S),
        dcc.Textarea(id="sfree-markers", value="",
                     placeholder="Optional: marker genes, separated by commas or new lines (leave empty to "
                                 "derive them from the annotation of the chosen type)…",
                     style={**analysis._AN_INPUT_STYLE, "margin": "2px 8px", "height": "34px", "flexShrink": "0",
                            "fontFamily": "monospace", "resize": "vertical"}),
        html.Div(id="sfree-status", style={**analysis._AN_STATUS_STYLE, "whiteSpace": "pre-wrap"}),
        html.Div(_SFREE_NOTE, id="sfree-note", style=analysis._AN_NOTE_STYLE),
        # the block has no height of its own (it sits in a scrolling column): the figure
        # carries its height, the graph follows it
        html.Div(dcc.Graph(id="sfree-plot", figure=make_sfree_fig(),
                           config={"displayModeBar": "hover",
                                   "toImageButtonOptions": {"format": "svg",
                                                            "filename": "xenium_segmentation_free_abundance"}},
                           style={"width": "100%"}),
                 id="sfree-plot-container", style={"flexShrink": "0", "minWidth": "0"}),
        html.Div([
            dash_table.DataTable(id="sfree-table", data=[], columns=cols, **stats._dark_table_style(12)),
            html.Div([
                dbc.Button("Download CSV", id="sfree-download-btn", color="secondary", size="sm", n_clicks=0,
                           style={"fontSize": "11px", "padding": "2px 8px"}),
                dcc.Download(id="sfree-download"),
            ], style={"padding": "4px 0"}),
        ], style={"padding": "0 8px", "flexShrink": "0"}),
        ui._plot_dim_controls("sfree"),
    ], id="sfree-section", style={"display": "flex", "flexDirection": "column", "flexShrink": "0"})
