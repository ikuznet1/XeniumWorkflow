"""Contamination and confound controls: tissue compartments (edge distance, local density,
dense core), spillover score, gene ownership and the leak gate for differential
expression, correction / segmentation QC metrics, and stratified co-localisation with
matched nulls and placebo anchors (compute + layouts).

Every neighbour query in this module is made on the cells of ONE sample: the functions
that build a KD-tree take one sample's coordinates, never the combined view, so no
neighbour count, distance or expected value can involve cells of two samples (the view
offsets samples in x only for drawing).
"""

import re
import threading
import time as _time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from dash import dcc, html
from dash import dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from . import core, jobs, caching, dataview, ui, annotation, reseg, figures, analysis, sdata_store, stats
from . import provenance


# ─── State ────────────────────────────────────────────────────────────────────
# Per-cell columns are kept by (sample key, segmentation) as Series indexed by cell id
# (str) and aligned to the cell table when used, so a subset or a re-ordered table reads
# the right values and the cell tables themselves are never written to.
_ctl_columns: dict = {}                 # (sample key, seg tag) -> {column name: Series}
_ctl_columns_lock = threading.Lock()

_comp_ctl_state: dict = {"status": "idle", "message": "", "result": None, "version": 0}
_comp_ctl_lock = threading.Lock()
_spill_state: dict = {"status": "idle", "message": "", "result": None, "version": 0}
_spill_lock = threading.Lock()
_segqc_state: dict = {"status": "idle", "message": "", "result": None, "version": 0}
_segqc_lock = threading.Lock()
_coloc_state: dict = {"status": "idle", "message": "", "result": None, "version": 0}
_coloc_lock = threading.Lock()

# compartments
_EDGE_BIN_UM = 50.0          # raster resolution of the tissue mask
_EDGE_CLOSE_BINS = 0         # iterations of morphological closing of the mask (0: none)
_EDGE_HOLE_BINS = 4          # interior holes of at most this many bins are filled
_DENSITY_R_UM = 100.0        # radius of the local cell density
_CORE_MIN_NB = 10            # dense core: at least this many other cells within the radius,
_CORE_MIN_FRAC = 0.30        # at least this fraction of them of the dominant type,
_CORE_MIN_EDGE_UM = 200.0    # and further than this from the tissue edge
# spillover
_SPILL_N_MARKERS = 10
_MARKER_MIN_CPM = 50.0       # a marker must reach this many counts per million in its own type
_MARKER_MIN_FOLD = 10.0      # and this many times the level of every other type
_MARKER_MIN_STRICT = 3       # fewer genes passing the fold rule: the top genes by ratio are used
_SPILL_LOCAL_R_UM = 50.0
_SPILL_NEAR_UM, _SPILL_FAR_UM = 30.0, 100.0
_SPILL_MIN_NEAR_FAR = 2.0    # validity: mean share near / far must reach this
_SPILL_FLAG_EDGES = (0.10, 0.25, 0.50)
_SPILL_FLAG_NAMES = ("none", "low", "moderate", "high")
_OWN_FRAC = 0.2              # a type owns a gene when its rate is at least this share of the top rate
_OWN_MIN_COUNT = 100         # genes with fewer transcripts in the view get no owner
# leak gate
_LEAK_N_MARKERS = 25
_LEAK_MIN_MARKERS = 10       # fewer genes passing the specificity rule: fallback to the top _LEAK_N_MARKERS
_LEAK_SHARE_MAX = 0.5
_LEAK_EXPLAIN_MIN = 0.5
_LEAK_CHANGE_MIN_LFC = 0.5   # the change rule applies to a gene whose change is called, or is at least this large (|log2|)
_LEAK_FLOOR = 1e-9
_LEAK_MIN_PURITY = 0.8       # share of a group's labelled cells its largest type must reach
_LEAK_MIN_CELLS = 50         # a neighbour type with fewer cells in a section is not a leak source there
_LEAK_MAX = 0.5              # a "leak" at or above this is shared expression: the neighbour is not modelled
# segmentation QC
_QC_RADIUS_UM = 15.0
_QC_SET_SIZE = 5
_QC_TRANSFER_UM = 10.0
# stratified co-localisation
_COLOC_RADIUS_UM = 50.0
_COLOC_EDGE_BINS = (50.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0)
_COLOC_DENSITY_Q = 5
_COLOC_MIN_POOL = 3
_COLOC_MIN_FOCAL, _COLOC_MIN_TARGET = 20, 10
_COLOC_MAX_TYPES = 40
_COLOC_N_PERM = 200
_COLOC_PLACEBO_DRAWS = 200
_COLOC_SIMILAR_FOLD = 3.0    # "similar abundance": within this factor of the focal type's count

_CTL_COLOR = {               # Color By entries: name -> (column, label)
    "@edge_distance": ("edge_distance_um", "Edge distance (µm)"),
    "@density": ("local_density", "Local density (cells within R)"),
    "@dominant_fraction": ("dominant_fraction", "Dominant-type fraction of neighbours"),
    "@core": ("core", "Dense core (1 = in core)"),
    "@spillover": ("spillover", "Spillover score (1 = as much as a source cell)"),
}


def _controls_reset() -> None:
    """Forget every per-cell column and result (another dataset was loaded; tests)."""
    with _ctl_columns_lock:
        _ctl_columns.clear()
    for st, lk in ((_comp_ctl_state, _comp_ctl_lock), (_spill_state, _spill_lock),
                   (_segqc_state, _segqc_lock), (_coloc_state, _coloc_lock)):
        with lk:
            st.update({"status": "idle", "message": "", "result": None})
            st["version"] += 1


# ─── Per-cell columns of the view ─────────────────────────────────────────────
def _seg_tag(seg_source) -> str:
    """Identity of the segmentation a column belongs to: "xenium", or the tool, the
    output directory and the run id of the loaded Baysor / Proseg run (a re-run with the
    same parameters is stored in the same directory and is another run)."""
    alt = analysis._an_alt_res(seg_source)
    if alt is None:
        return "xenium"
    rid = core._seg_run_id(alt)
    return f"{core._seg_tool(seg_source)}:{alt.get('out_dir', '')}" + (f"#{rid}" if rid else "")


def _ctl_forget_run(final_dir) -> None:
    """A run was committed to *final_dir*: drop the columns of an earlier run there."""
    with _ctl_columns_lock:
        for k in [k for k in _ctl_columns
                  if any(str(k[1]).split("#")[0] == f"{t}:{final_dir}" for t in core._SEG_TOOL_LABELS)]:
            _ctl_columns.pop(k, None)


caching._seg_commit_hooks.append(_ctl_forget_run)


_CTL_SEG_CHANGED = ("The segmentation was switched while this job was running: its per-cell values belong "
                    "to the cells of the other run and were discarded. Run it again.")


def _ctl_check_tag(seg_source, tag) -> str:
    """The segmentation tag to write under. *tag* is the one a job resolved when it read
    its tables; when the loaded run has changed since (another cached run was picked in
    the dropdown), the job's values belong to other cells: ValueError, nothing is written."""
    cur = _seg_tag(seg_source)
    if tag is not None and tag != cur:
        raise ValueError(_CTL_SEG_CHANGED)
    return cur


def _ctl_put(table: dict, seg_source, columns: dict, tag=None, drop=()) -> None:
    """Store per-cell columns of one view table: {name: values aligned to table["df"]}.
    tag: see _ctl_check_tag; drop: names removed first (the other columns of the same
    computation, so that a rerun never leaves a mix of two runs)."""
    key = (table["key"], _ctl_check_tag(seg_source, tag))
    idx = pd.Index(table["df"].index.astype(str))
    with _ctl_columns_lock:
        store = _ctl_columns.setdefault(key, {})
        for name in drop:
            store.pop(name, None)
        for name, vals in columns.items():
            store[name] = pd.Series(np.asarray(vals), index=idx)


def _ctl_replace(tables, seg_source, family, per_table: dict, tag=None) -> None:
    """Write phase of a job, for all samples at once: in every table of *tables* the
    columns named in *family* are replaced by per_table.get(table key) - a sample the run
    skipped loses the columns of the previous run instead of keeping them beside the new
    result. Call it after the job's last cancellation point (jobs._job_shield)."""
    _ctl_check_tag(seg_source, tag)
    for t in tables:
        _ctl_put(t, seg_source, per_table.get(t["key"]) or {}, tag=tag, drop=family)


_COMP_COLUMNS = ("edge_distance_um", "local_density", "dominant_fraction", "core", "compartment")
_SPILL_COLUMNS = ("spillover", "spillover_flag", "source_distance_um")


def _ctl_get(table: dict, seg_source, name: str):
    """A stored column aligned to the cells of *table* (NaN / None for cells it was not
    computed for), or None when it does not exist for this sample and segmentation."""
    with _ctl_columns_lock:
        ser = _ctl_columns.get((table["key"], _seg_tag(seg_source)), {}).get(name)
    if ser is None:
        return None
    return ser.reindex(pd.Index(table["df"].index.astype(str))).to_numpy()


def _ctl_available(seg_source="xenium") -> list:
    """Names of the columns at least one sample of the view has."""
    tag = _seg_tag(seg_source)
    keys = {t["key"] for t in stats._view_tables(seg_source)}
    out = []
    with _ctl_columns_lock:
        for (k, tg), cols in _ctl_columns.items():
            if tg == tag and k in keys:
                out += [c for c in cols if c not in out]
    return out


def _ctl_color(name, seg_source="xenium", use_corrected: bool = False):
    """Colour provider (see analysis._view_color_providers): (values over the cells of
    the view, the samples one after the other; (lo, hi); label), or None when the column
    was not computed for this view."""
    spec = _CTL_COLOR.get(str(name))
    if spec is None:
        return None
    col, label = spec
    tables = stats._view_tables(seg_source)
    parts, have = [], False
    for t in tables:
        v = _ctl_get(t, seg_source, col)
        if v is None:
            parts.append(np.full(len(t["df"]), np.nan, dtype=np.float32))
        else:
            have = True
            parts.append(pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(dtype=np.float32))
    if not have:
        return None
    vals = np.concatenate(parts) if len(parts) > 1 else parts[0]
    fin = vals[np.isfinite(vals)]
    hi = float(np.percentile(fin, 99.5)) if len(fin) else 1.0
    if col in ("core", "dominant_fraction"):
        hi = 1.0
    if col == "spillover":
        hi = float(min(max(hi, 0.5), 2.0))
    return vals, (0.0, hi if hi > 0 else 1.0), label


def _ctl_color_options() -> list:
    """Color By options of the columns that exist for some segmentation of the dataset."""
    with _ctl_columns_lock:
        have = {c for cols in _ctl_columns.values() for c in cols}
    return [{"label": f"Control: {label}", "value": f"geneset:{name}"}
            for name, (col, label) in _CTL_COLOR.items() if col in have]


def _ctl_obs_columns(cells_df, seg_source) -> "pd.DataFrame | None":
    """Columns for Save as SpatialData: the control columns of the primary sample for the
    exported segmentation, aligned to *cells_df*, named ctl_<column>."""
    table = {"key": dataview._ds_key(core.DATA), "df": cells_df}
    out = {}
    with _ctl_columns_lock:
        names = list(_ctl_columns.get((table["key"], _seg_tag(seg_source)), {}))
    for name in names:
        v = _ctl_get(table, seg_source, name)
        if v is not None:
            out[f"ctl_{name}"] = v
    return pd.DataFrame(out, index=cells_df.index) if out else None


analysis._view_color_providers["@"] = _ctl_color
analysis._view_color_options.append(_ctl_color_options)
sdata_store._extra_obs_providers.append(_ctl_obs_columns)


# ─── Tissue compartments ──────────────────────────────────────────────────────
_COMPARTMENT_NOTE = (
    "Per sample (section), on the cells of the active segmentation and subset. Edge distance: "
    "the cell positions are rasterised into square bins, occupied bins are the tissue (after an "
    "optional morphological closing; interior holes of at most the given number of bins are "
    "filled), and a Euclidean distance transform gives every bin its distance to the nearest "
    "empty bin, in µm; a cell inherits its bin's value, so the resolution is one bin. Local "
    "density: other cells within R. Dense core of the chosen dominant cell type: at least N "
    "other cells within R, at least the given fraction of them of that type, and further than "
    "D from the edge; compartment = core / edge (not core, within D of the edge) / other. The "
    "columns are offered under Color By, stratify the co-localisation test and are written by "
    "Save as SpatialData (ctl_*). They describe where a cell sits, not what it is: section "
    "margins are where sectioning damage and segmentation noise concentrate, and a cell type "
    "that prefers the margin will look depleted of whatever is rare there. Recompute after "
    "changing the subset or the segmentation.")


def _edge_distance(xy, bin_um: float = _EDGE_BIN_UM, close_bins: int = _EDGE_CLOSE_BINS,
                   hole_max_bins: "int | None" = _EDGE_HOLE_BINS) -> tuple:
    """(distance of every cell to the tissue edge in µm, info) for ONE section.

    The cells are rasterised into *bin_um* squares; occupied bins are tissue. *close_bins*
    > 0: morphological closing (4-neighbour structure, that many iterations). Interior
    holes (empty regions not connected to the outside, 4-connectivity) of at most
    *hole_max_bins* bins are filled (None: all interior holes; 0: none). The Euclidean
    distance transform of the mask × bin size is the distance to the nearest empty bin;
    a cell gets the value of its bin (so cells in an edge bin get one bin size)."""
    from scipy import ndimage
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) == 0:
        return np.zeros(0), {"tissue_bins": 0, "holes_filled": 0, "holes_left": 0}
    bin_um = float(bin_um)
    pad = 1 + max(0, int(close_bins or 0))
    bx = ((xy[:, 0] - xy[:, 0].min()) / bin_um).astype(np.int64)
    by = ((xy[:, 1] - xy[:, 1].min()) / bin_um).astype(np.int64)
    grid = np.zeros((int(bx.max()) + 1 + 2 * pad, int(by.max()) + 1 + 2 * pad), dtype=bool)
    grid[bx + pad, by + pad] = True
    n_tissue = int(grid.sum())
    if close_bins and int(close_bins) > 0:
        grid = ndimage.binary_closing(grid, iterations=int(close_bins)) | grid
    lab, n = ndimage.label(~grid)
    sizes = np.bincount(lab.ravel())
    outside = lab[0, 0]
    holes = [i for i in range(1, n + 1) if i != outside]
    small = [i for i in holes if hole_max_bins is None or sizes[i] <= int(hole_max_bins)]
    filled = grid | np.isin(lab, small) if small else grid
    d = ndimage.distance_transform_edt(filled)[bx + pad, by + pad] * bin_um
    return d, {"tissue_bins": n_tissue, "holes_filled": len(small), "holes_left": len(holes) - len(small),
               "bins_after_fill": int(filled.sum())}


def _local_counts(xy, radius: float, mask=None) -> np.ndarray:
    """Per cell of ONE section: number of OTHER cells within *radius* (µm); with *mask*,
    the number of other masked cells."""
    from scipy.spatial import cKDTree
    xy = np.asarray(xy, dtype=np.float64)
    if mask is None:
        return (cKDTree(xy).query_ball_point(xy, float(radius), return_length=True, workers=-1)
                .astype(np.int64) - 1)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros(len(xy), dtype=np.int64)
    n = cKDTree(xy[mask]).query_ball_point(xy, float(radius), return_length=True, workers=-1).astype(np.int64)
    return n - mask.astype(np.int64)


def _compartments(xy, is_dominant=None, bin_um=_EDGE_BIN_UM, close_bins=_EDGE_CLOSE_BINS,
                  hole_max_bins=_EDGE_HOLE_BINS, radius=_DENSITY_R_UM, min_nb=_CORE_MIN_NB,
                  min_frac=_CORE_MIN_FRAC, min_edge=_CORE_MIN_EDGE_UM) -> tuple:
    """Edge distance, local density and the dense core of ONE section: ({edge_distance_um,
    local_density, dominant_fraction, core, compartment}, info). Without *is_dominant*
    there is no core: the compartment is "edge" within *min_edge* of the edge, else "other"."""
    d_edge, info = _edge_distance(xy, bin_um, close_bins, hole_max_bins)
    n_nb = _local_counts(xy, radius)
    cols = {"edge_distance_um": d_edge.astype(np.float32), "local_density": n_nb.astype(np.float32)}
    core_mask = np.zeros(len(d_edge), dtype=bool)
    if is_dominant is not None:
        n_dom = _local_counts(xy, radius, is_dominant)
        f_dom = n_dom / np.maximum(n_nb, 1)
        core_mask = (n_nb >= int(min_nb)) & (f_dom >= float(min_frac)) & (d_edge > float(min_edge))
        cols["dominant_fraction"] = f_dom.astype(np.float32)
        cols["core"] = core_mask.astype(np.float32)
    comp = np.where(core_mask, "core", np.where(d_edge <= float(min_edge), "edge", "other")).astype(object)
    cols["compartment"] = comp
    info.update({"n_cells": int(len(d_edge)), "median_edge": float(np.median(d_edge)) if len(d_edge) else np.nan,
                 "median_density": float(np.median(n_nb)) if len(n_nb) else np.nan,
                 "frac_core": float(core_mask.mean()) if len(core_mask) else 0.0,
                 "frac_edge": float((comp == "edge").mean()) if len(comp) else 0.0})
    return cols, info


def _pooled_types(labels) -> "pd.Series":
    """Cell count of every label over the samples of a view, most abundant first."""
    parts = [pd.Series(l, dtype=object).dropna().astype(str) for l in labels if l is not None]
    if not parts:
        return pd.Series(dtype=np.int64)
    return pd.concat(parts).value_counts()


def _run_compartments(seg_source, method, label_kind, dominant, params=None) -> None:
    """Background job: compartment columns for every sample of the active view."""
    _gen = jobs._dataset_gen
    p = {"bin_um": _EDGE_BIN_UM, "close_bins": _EDGE_CLOSE_BINS, "hole_max_bins": _EDGE_HOLE_BINS,
         "radius": _DENSITY_R_UM, "min_nb": _CORE_MIN_NB, "min_frac": _CORE_MIN_FRAC,
         "min_edge": _CORE_MIN_EDGE_UM}
    p.update({k: v for k, v in (params or {}).items() if v is not None and k in p})

    def _set(status, message, result=None):
        with _comp_ctl_lock:
            if jobs._stale_job(_gen, status, message):
                return
            _comp_ctl_state["status"], _comp_ctl_state["message"] = status, message
            if result is not None:
                _comp_ctl_state["result"] = result
            if status in ("done", "error"):
                _comp_ctl_state["version"] += 1
        if status == "done":
            provenance._prov_record("compartments", seg_source=seg_source, params={"dominant": dominant, **(params or {})}, labels=(label_kind, method), message=message, result=result)

    try:
        _t0 = _time.time()
        _set("running", "Rasterising the tissue…")
        analysis._an_require_seg(seg_source)
        tables = stats._view_tables(seg_source)
        seg_tag = _seg_tag(seg_source)
        labels, cat = (None, "")
        if label_kind:
            labels, cat = stats._comp_labels(label_kind, tables, seg_source, method)
        dom = None
        if labels is not None and any(l is not None for l in labels):
            vc = _pooled_types(labels)
            dom = str(dominant) if dominant and str(dominant) != "auto" else (str(vc.index[0]) if len(vc) else None)
            if dom is not None and dom not in vc.index:
                _set("error", f"No cell of the view is labelled '{dom}'.")
                return
        infos, out = [], {}
        for i, t in enumerate(tables):
            jobs._job_check()
            jobs._job_progress(f"Compartments: {t['label']} ({len(t['df']):,} cells)…", i / len(tables))
            xy = stats._xy_of(t["df"])
            if len(xy) < 3:
                continue
            lab = labels[i] if labels is not None else None
            is_dom = None if (dom is None or lab is None) else (pd.Series(lab, dtype=object).astype(str).to_numpy() == dom)
            cols, info = _compartments(xy, is_dom, **p)
            jobs._check_dataset_gen(_gen)
            out[t["key"]] = cols
            info.update({"label": t["label"], "name": t["name"]})
            infos.append(info)
        if not infos:
            _set("error", "No sample of the view has at least 3 cells.")
            return
        # All samples are written together, after the last cancellation point: a cancelled
        # run changes nothing, and the columns of an earlier run (a core that this run
        # does not define, a sample this run skipped) do not stay beside the new ones.
        jobs._job_shield("Writing the compartment columns…")
        _ctl_replace(tables, seg_source, _COMP_COLUMNS, out, tag=seg_tag)
        res = {"samples": infos, "params": p, "dominant": dom, "category": cat,
               "seg_source": seg_source or "xenium", "seconds": _time.time() - _t0}
        msg = (f"Compartments ({p['bin_um']:g} µm bins"
               + (f", closing × {int(p['close_bins'])}" if p["close_bins"] else "")
               + f", holes ≤ {p['hole_max_bins']} bins filled; density radius {p['radius']:g} µm"
               + (f"; core of '{dom}': ≥ {int(p['min_nb'])} neighbours, ≥ {100 * p['min_frac']:.0f} % "
                  f"{dom}, > {p['min_edge']:g} µm from the edge" if dom else "; no annotation: no core")
               + f"; {res['seconds']:.1f} s). "
               + "; ".join(f"{i['label']}: {i['n_cells']:,} cells, median edge distance {i['median_edge']:.0f} µm, "
                           f"median density {i['median_density']:.0f}, {100 * i['frac_core']:.1f} % core, "
                           f"{100 * i['frac_edge']:.1f} % edge, {i['holes_filled']} holes filled"
                           + (f" ({i['holes_left']} larger left open)" if i["holes_left"] else "")
                           for i in infos)
               + ". Color By: 'Control: Edge distance', 'Local density', 'Dense core'.")
        _set("done", msg, result=res)
        print(f"  {msg}", flush=True)
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


# ─── Type profiles, markers, gene ownership ───────────────────────────────────
def _type_profiles(samples, labels, genes=None, masks=None) -> dict:
    """Counts of every gene summed over the cells of each label.

    samples / labels: analysis samples and, per sample, the label of each cell (None:
    unlabelled, left out); masks: optional per-sample boolean masks restricting the
    cells. Returns {types, genes, counts (types × genes, float64), depth (Σ of all panel
    counts of the type's cells), n (cells per type), per_sample: [{counts, depth, n}]}.
    The rate of a gene in a type is counts / depth (ratio of sums: defined for fractional
    counts, and not moved by the depth of single cells)."""
    genes = analysis._an_common_genes(samples) if genes is None else list(genes)
    vc = _pooled_types(labels)
    types = sorted(vc.index, key=str)
    tpos = {t: i for i, t in enumerate(types)}
    T, G = len(types), len(genes)
    tot = {"counts": np.zeros((T, G)), "depth": np.zeros(T), "n": np.zeros(T, dtype=np.int64)}
    per = []
    for si, (s, lab) in enumerate(zip(samples, labels)):
        c = {"counts": np.zeros((T, G)), "depth": np.zeros(T), "n": np.zeros(T, dtype=np.int64)}
        per.append(c)
        if lab is None or not T:
            continue
        rows = np.asarray(s["rows"])
        code = pd.Series(lab, dtype=object).map(lambda v: tpos.get(str(v), -1) if v is not None else -1).to_numpy(dtype=np.int64)
        ok = (rows >= 0) & (code >= 0)
        if masks is not None and masks[si] is not None:
            ok &= np.asarray(masks[si], dtype=bool)
        if not ok.any():
            continue
        csc = s["csc"]
        H = sp.csr_matrix((np.ones(int(ok.sum())), (code[ok], rows[ok])), shape=(T, csc.shape[0]))
        cols = np.array([s["gni"][g] for g in genes], dtype=np.int64)
        panel = np.array([s["gni"][g] for g in s["genes"]], dtype=np.int64)
        M = H @ csc                                   # types × all matrix columns
        M = np.asarray(M.todense()) if sp.issparse(M) else np.asarray(M)
        c["counts"] = M[:, cols].astype(np.float64)
        c["depth"] = M[:, panel].sum(axis=1).astype(np.float64)
        c["n"] = np.bincount(code[ok], minlength=T).astype(np.int64)
        for k in tot:
            tot[k] = tot[k] + c[k]
    return {"types": types, "genes": genes, "counts": tot["counts"], "depth": tot["depth"],
            "n": tot["n"], "per_sample": per}


def _cpm(counts, depth) -> np.ndarray:
    """Counts per million of the group's total counts (rows: groups)."""
    return 1e6 * np.asarray(counts, dtype=np.float64) / np.maximum(np.asarray(depth, dtype=np.float64), 1e-12)[:, None]


def _markers_of(prof: dict, source, n: int = _SPILL_N_MARKERS, min_cpm: float = _MARKER_MIN_CPM,
                min_fold: float = _MARKER_MIN_FOLD, exclude=(), min_strict: int = _MARKER_MIN_STRICT,
                min_cells: int = 10) -> tuple:
    """Marker genes of *source* derived from the type profiles: (genes, info).

    Candidates reach *min_cpm* in the source type. The specificity ratio of a gene is its
    source CPM over the highest CPM among the other types (those with at least *min_cells*
    cells, without the types in *exclude*). Rule "specific": the *n* candidates with the
    highest ratio among those at ≥ *min_fold*. When fewer than *min_strict* genes pass the
    fold rule, the *n* candidates with the highest ratio are used whatever their ratio
    (rule "fallback": the markers are then not exclusive, and anything built on them is an
    upper bound)."""
    types = prof["types"]
    if str(source) not in types:
        return [], {"rule": "none", "n_strict": 0}
    si = types.index(str(source))
    cpm = _cpm(prof["counts"], prof["depth"])
    others = [i for i, t in enumerate(types)
              if i != si and t not in set(map(str, exclude)) and prof["n"][i] >= min_cells]
    oth = cpm[others].max(axis=0) if others else np.zeros(cpm.shape[1])
    src = cpm[si]
    ok = src >= float(min_cpm)
    ratio = src / np.maximum(oth, 1e-9)
    strict = ok & (src >= float(min_fold) * oth)
    use, rule = (strict, "specific") if int(strict.sum()) >= int(min_strict) else (ok, "fallback")
    idx = np.flatnonzero(use)
    idx = idx[np.argsort(-ratio[idx], kind="stable")][:int(n)]
    return [prof["genes"][i] for i in idx], {"rule": rule, "n_strict": int(strict.sum()),
                                             "min_ratio": float(ratio[idx].min()) if len(idx) else np.nan,
                                             "n_candidates": int(ok.sum())}


def _gene_ownership(prof: dict, frac: float = _OWN_FRAC, min_count: float = _OWN_MIN_COUNT,
                    min_cells: int = 10) -> pd.DataFrame:
    """Which cell type(s) own each gene: the type with the highest rate (Σ counts / Σ
    depth) and every type whose rate is at least *frac* of it. A gene with fewer than
    *min_count* transcripts over all labelled cells gets no owner (its rates are noise);
    types with fewer than *min_cells* cells own nothing. Columns: gene, owner (top type),
    owners (all owners, ';'-separated), n_owners, owner_cpm, second_ratio (rate of the
    runner-up / top rate), total_count."""
    types, genes = prof["types"], prof["genes"]
    cpm = _cpm(prof["counts"], prof["depth"])
    big = prof["n"] >= min_cells
    cpm = np.where(big[:, None], cpm, 0.0)
    total = prof["counts"].sum(axis=0)
    if not len(types):
        return pd.DataFrame({"gene": genes, "owner": "", "owners": "", "n_owners": 0,
                             "owner_cpm": np.nan, "second_ratio": np.nan, "total_count": total})
    top = cpm.argmax(axis=0)
    top_v = cpm.max(axis=0)
    has = (total >= float(min_count)) & (top_v > 0)
    own = (cpm >= float(frac) * top_v[None, :]) & (top_v[None, :] > 0)
    srt = np.sort(cpm, axis=0)
    second = srt[-2] if len(types) > 1 else np.zeros(len(genes))
    tarr = np.array(types, dtype=object)
    owners = [";".join(tarr[own[:, j]][np.argsort(-cpm[own[:, j], j], kind="stable")]) if has[j] else ""
              for j in range(len(genes))]
    return pd.DataFrame({"gene": genes, "owner": np.where(has, tarr[top], ""), "owners": owners,
                         "n_owners": np.where(has, own.sum(axis=0), 0),
                         "owner_cpm": np.where(has, top_v, np.nan),
                         "second_ratio": np.where(has, second / np.maximum(top_v, 1e-12), np.nan),
                         "total_count": total})


# ─── Spillover score ──────────────────────────────────────────────────────────
_SPILL_NOTE = (
    "Spillover score of a cell for a source cell type = the share of the cell's counts in the "
    "source type's marker genes, divided by the median of that share inside source cells (the "
    "mean, and said so, when the markers are so sparse that the median is 0): 1 "
    "means the cell carries as much source-marker signal as a real source cell. Markers are "
    "the source type's most specific genes in the active annotation (≥ 50 counts per million "
    "in the source, ≥ 10× every other type; the top genes by ratio when fewer than 3 pass), or "
    "the genes you list. Flags: ≥ 0.10 low, ≥ 0.25 moderate, ≥ 0.50 high. Built-in validity "
    "check: if the score measures transcripts bleeding across segment boundaries, the marker "
    "share of non-source cells must fall with the distance to the nearest source cell (mean "
    "share within 30 µm at least 2× the mean beyond 100 µm) and rise with the fraction of "
    "source cells within 50 µm; distances and neighbourhoods are computed within each sample. "
    "When a check fails, the score measures something else (shared biology, an annotation that "
    "splits one population) and must not be used as a contamination mask. The score depends on "
    "the annotation, and a cell with few counts gets a noisy share: the flag rate per depth "
    "quintile is reported. Gene ownership: per gene, the type with the highest rate (Σ counts / "
    "Σ depth) and every type at ≥ 20 % of it; a 'programme' read in a type that owns few of its "
    "genes is largely a readout of its neighbours.")


def _marker_counts(s: dict, genes) -> tuple:
    """(Σ counts over *genes*, Σ counts over all panel genes) per cell of sample *s*
    (NaN for a cell without a matrix row)."""
    csc = s["csc"]
    cols = np.array([s["gni"][g] for g in genes if g in s["gni"]], dtype=np.int64)
    panel = np.array([s["gni"][g] for g in s["genes"]], dtype=np.int64)
    w = np.zeros(csc.shape[1])
    w[cols] = 1.0
    m_row = np.asarray(csc @ w).ravel()
    wp = np.zeros(csc.shape[1])
    wp[panel] = 1.0
    t_row = np.asarray(csc @ wp).ravel()
    rows = np.asarray(s["rows"])
    ok = rows >= 0
    r = np.clip(rows, 0, None)
    return np.where(ok, m_row[r], np.nan), np.where(ok, t_row[r], np.nan)


def _source_geometry(xy, is_source, local_r: float = _SPILL_LOCAL_R_UM) -> tuple:
    """For ONE section: (distance of every cell to the nearest source cell - for a source
    cell, to the nearest OTHER source cell; NaN when the section has none -, fraction of
    the cells within *local_r* that are source cells, the cell itself included)."""
    from scipy.spatial import cKDTree
    xy = np.asarray(xy, dtype=np.float64)
    src = np.asarray(is_source, dtype=bool)
    dist = np.full(len(xy), np.nan)
    n_all = cKDTree(xy).query_ball_point(xy, float(local_r), return_length=True, workers=-1).astype(np.float64)
    if not src.any():
        return dist, np.zeros(len(xy))
    tree = cKDTree(xy[src])
    n_src = tree.query_ball_point(xy, float(local_r), return_length=True, workers=-1).astype(np.float64)
    if (~src).any():
        dist[~src] = tree.query(xy[~src], k=1, workers=-1)[0]
    if int(src.sum()) > 1:
        dist[src] = tree.query(xy[src], k=2, workers=-1)[0][:, 1]
    return dist, n_src / np.maximum(n_all, 1.0)


def _spillover_validity(share, dist, local, non_source, near=_SPILL_NEAR_UM, far=_SPILL_FAR_UM,
                        min_ratio=_SPILL_MIN_NEAR_FAR) -> dict:
    """The two checks a bleed measure must pass, on the non-source cells: marker share
    falls with distance to the nearest source cell (mean within *near* µm ÷ mean at ≥
    *far* µm ≥ *min_ratio*) and rises with the local source fraction (Spearman ρ > 0 and
    the mean share in the top third of local source fraction above the bottom third)."""
    from scipy.stats import spearmanr
    share, dist, local = (np.asarray(v, dtype=np.float64) for v in (share, dist, local))
    m = np.asarray(non_source, dtype=bool) & np.isfinite(dist) & np.isfinite(share)
    out = {"n": int(m.sum()), "near_mean": np.nan, "far_mean": np.nan, "near_far_ratio": np.nan,
           "n_near": 0, "n_far": 0, "distance_ok": None, "rho_local": np.nan,
           "local_low": np.nan, "local_high": np.nan, "density_ok": None, "valid": None}
    if out["n"] < 20:
        out["reason"] = "fewer than 20 non-source cells with a source cell in their sample"
        return out
    nr, fr = m & (dist < near), m & (dist >= far)
    out["n_near"], out["n_far"] = int(nr.sum()), int(fr.sum())
    if out["n_near"] >= 10 and out["n_far"] >= 10:
        out["near_mean"], out["far_mean"] = float(share[nr].mean()), float(share[fr].mean())
        out["near_far_ratio"] = float(out["near_mean"] / max(out["far_mean"], 1e-9))
        out["distance_ok"] = bool(out["near_far_ratio"] >= min_ratio)
    lv = local[m]
    if np.ptp(lv) > 0:
        rho = spearmanr(lv, share[m]).statistic
        q1, q2 = np.quantile(lv, [1 / 3, 2 / 3])
        if q2 > q1:
            lo, hi = share[m][lv <= q1], share[m][lv >= q2]
        else:                         # most cells have no source cell around: none vs some
            lo, hi = share[m][lv <= lv.min()], share[m][lv > lv.min()]
        if len(lo) >= 10 and len(hi) >= 10:
            out["local_low"], out["local_high"] = float(lo.mean()), float(hi.mean())
            out["rho_local"] = float(rho)
            out["density_ok"] = bool(rho > 0 and out["local_high"] > out["local_low"])
    checks = [v for v in (out["distance_ok"], out["density_ok"]) if v is not None]
    out["valid"] = (all(checks) if len(checks) == 2 else (False if (checks and not all(checks)) else None))
    return out


def _spillover(samples, labels, source, markers, local_r=_SPILL_LOCAL_R_UM, near=_SPILL_NEAR_UM,
               far=_SPILL_FAR_UM, edges=_SPILL_FLAG_EDGES) -> dict:
    """Spillover score of every cell for *source* (see _SPILL_NOTE).

    Returns per_sample: [{share, score, dist, local, depth, is_source, labelled}], and
    reference (median marker share within source cells, pooled over the samples),
    validity (see _spillover_validity), profile (mean share of non-source cells by
    distance band), flags (non-source cells per flag), by_type (share of each type's
    cells at or above the first flag edge), by_depth (flag rate per depth quintile)."""
    per = []
    for s, lab in zip(samples, labels):
        m, tot = _marker_counts(s, markers)
        share = m / np.maximum(tot, 1.0)                 # a cell with no counts has share 0
        lab_s = (pd.Series(lab, dtype=object) if lab is not None
                 else pd.Series([None] * len(s["df"]), dtype=object))
        labelled = lab_s.notna().to_numpy()
        is_src = (lab_s.astype(str).to_numpy() == str(source)) & labelled
        dist, local = _source_geometry(stats._xy_of(s["df"]), is_src, local_r)
        per.append({"share": share, "depth": tot, "is_source": is_src, "labelled": labelled,
                    "dist": dist, "local": local, "label": lab_s.to_numpy()})
        jobs._job_check()
    cat = lambda k: np.concatenate([p[k] for p in per])
    share, is_src, labelled = cat("share"), cat("is_source"), cat("labelled")
    src_share = share[is_src & np.isfinite(share)]
    if not len(src_share):
        raise ValueError(f"No cell with counts is labelled '{source}'.")
    reference, ref_kind = float(np.median(src_share)), "median"
    src_detect = float((src_share > 0).mean())
    if reference <= 0:
        # sparse markers: most source cells have none of them. The mean share is then the
        # reference, and the per-cell score is coarse (0 for most cells).
        reference, ref_kind = float(src_share.mean()), "mean"
    if reference <= 0:
        raise ValueError(f"No '{source}' cell has a transcript of the markers "
                         f"({', '.join(markers[:8])}): they are not expressed by that type.")
    for p in per:
        p["score"] = p["share"] / reference
    score, dist, local, depth = cat("score"), cat("dist"), cat("local"), cat("depth")
    non = labelled & ~is_src & np.isfinite(score)
    val = _spillover_validity(share, dist, local, non, near, far)
    bands = [0, 5, 10, 15, 20, 30, 50, 100, np.inf]
    prof = []
    for lo, hi in zip(bands[:-1], bands[1:]):
        mm = non & (dist > lo) & (dist <= hi)
        prof.append({"band": f"{lo:g}–{hi:g}" if np.isfinite(hi) else f"> {lo:g}", "lo": lo, "hi": hi,
                     "n": int(mm.sum()), "mean_share": float(share[mm].mean()) if mm.any() else np.nan,
                     "mean_score": float(score[mm].mean()) if mm.any() else np.nan,
                     "median_depth": float(np.median(depth[mm])) if mm.any() else np.nan})
    flag = np.digitize(score, list(edges))                 # 0 = none … 3 = high
    flags = {nm: int(((flag == i) & non).sum()) for i, nm in enumerate(_SPILL_FLAG_NAMES)}
    labs = cat("label")
    by_type = (pd.DataFrame({"type": labs[non].astype(str), "f": score[non] >= edges[0]})
               .groupby("type")["f"].agg(["mean", "size"]).sort_values("mean", ascending=False))
    by_depth = []
    if non.sum() >= 50:
        q = pd.qcut(depth[non], 5, labels=False, duplicates="drop")
        by_depth = (pd.DataFrame({"q": q, "f": score[non] >= edges[0], "d": depth[non]})
                    .groupby("q").agg(rate=("f", "mean"), median_depth=("d", "median")).to_dict("records"))
    return {"per_sample": per, "reference": reference, "reference_kind": ref_kind,
            "source_detect": src_detect, "validity": val, "profile": prof, "flags": flags,
            "n_non_source": int(non.sum()), "n_source": int(len(src_share)),
            "n_source_labelled": int(is_src.sum()),
            "by_type": by_type, "by_depth": by_depth, "edges": tuple(edges)}


def _spill_flags(score, edges=_SPILL_FLAG_EDGES) -> np.ndarray:
    """The per-cell flag name of a spillover score; None for a cell that was not scored
    (NaN score: no matrix row, e.g. no SPLIT-corrected counts), never "none"."""
    score = np.asarray(score, dtype=np.float64)
    ok = np.isfinite(score)
    flag = np.array(_SPILL_FLAG_NAMES, dtype=object)[np.digitize(np.where(ok, score, 0.0), list(edges))]
    flag[~ok] = None
    return flag


def _auto_source(prof: dict, exclude=()) -> "str | None":
    """The type contributing the most transcripts to the view (abundance × expression),
    among the types not named in *exclude*; None when no type is left."""
    keep = [i for i, t in enumerate(prof["types"]) if t not in set(exclude)]
    if not keep:
        return None
    depth = np.asarray(prof["depth"], dtype=np.float64)[keep]
    return prof["types"][keep[int(np.argmax(depth))]]


def _run_spillover(seg_source, method, label_kind, source, markers_text, n_markers, use_corrected,
                   own_frac=_OWN_FRAC) -> None:
    """Background job: spillover score of every cell of the view for one source type, its
    validity check, and the gene ownership table."""
    _gen = jobs._dataset_gen

    def _set(status, message, result=None):
        with _spill_lock:
            if jobs._stale_job(_gen, status, message):
                return
            _spill_state["status"], _spill_state["message"] = status, message
            if result is not None:
                _spill_state["result"] = result
            if status in ("done", "error"):
                _spill_state["version"] += 1
        if status == "done":
            provenance._prov_record("spillover", seg_source=seg_source, corrected=use_corrected, params={"marker_source": source, "markers": markers_text, "n_markers": n_markers, "own_frac": own_frac}, labels=(label_kind, method), message=message, result=result)

    try:
        _t0 = _time.time()
        _set("running", "Summing counts per cell type…")
        samples = analysis._an_samples(seg_source, use_corrected)
        seg_tag = _seg_tag(seg_source)
        if not samples:
            _set("error", analysis._an_no_matrix_msg(seg_source))
            return
        labels, cat = stats._comp_labels(label_kind, samples, seg_source, method, use_corrected)
        if all(l is None for l in labels):
            _set("error", f"The active view has no {cat} labels: run Cell Type Annotation first.")
            return
        prof = _type_profiles(samples, labels)
        jobs._job_check()
        src = str(source) if source and str(source) != "auto" else _auto_source(prof)
        if src not in prof["types"]:
            _set("error", f"No cell of the view is labelled '{src}'.")
            return
        user = analysis._gs_parse(markers_text or "")
        if user:
            found, missing = analysis._gs_match(user, prof["genes"])
            if not found:
                _set("error", f"None of the {len(user)} listed marker genes is measured in this view.")
                return
            markers, minfo = found, {"rule": "user", "missing": missing}
        else:
            markers, minfo = _markers_of(prof, src, n=int(max(1, n_markers or _SPILL_N_MARKERS)))
            if not markers:
                _set("error", f"No gene reaches {_MARKER_MIN_CPM:g} counts per million in '{src}': "
                              f"no markers could be derived. List marker genes instead.")
                return
        jobs._job_progress("Scoring cells and measuring distances…", 0.3)
        _set("running", f"Spillover of '{src}': scoring cells ({len(markers)} markers)…")
        res = _spillover(samples, labels, src, markers)
        jobs._check_dataset_gen(_gen)
        own_frac = float(own_frac or _OWN_FRAC)
        own = _gene_ownership(prof, frac=own_frac)
        out = {}
        for s, p in zip(samples, res["per_sample"]):
            flag = _spill_flags(p["score"], res["edges"])
            out[s["key"]] = {"spillover": p["score"].astype(np.float32), "spillover_flag": flag,
                             "source_distance_um": p["dist"].astype(np.float32)}
        jobs._job_shield("Writing the spillover columns…")
        _by_key = {s["key"]: s for s in samples}
        _ctl_replace([_by_key.get(t["key"], t) for t in stats._view_tables(seg_source)], seg_source,
                     _SPILL_COLUMNS, out, tag=seg_tag)
        v = res["validity"]
        corrected = bool(samples[0]["corrected"])
        res.update({"source": src, "markers": markers, "marker_info": minfo, "ownership": own,
                    "category": cat, "corrected": corrected, "seg_source": seg_source or "xenium",
                    "n_uncorrected": analysis._an_uncorrected(samples),
                    "own_frac": own_frac,
                    "types": prof["types"], "n_cells_type": dict(zip(prof["types"], prof["n"].tolist())),
                    "samples": [{"label": s["label"], "name": s["name"], "n_cells": len(s["df"])} for s in samples],
                    "seconds": _time.time() - _t0})
        if v["valid"] is True:
            verdict = "Validity check PASSED"
        elif v["valid"] is False:
            verdict = ("⚠ Validity check FAILED: the score is not measuring bleed here and must not "
                       "be used as a contamination mask")
        else:
            verdict = "Validity check not fully testable"
        parts = []
        if v["distance_ok"] is not None:
            parts.append(f"marker share of non-source cells within {_SPILL_NEAR_UM:g} µm of a source cell "
                         f"{v['near_mean']:.5f} (n = {v['n_near']:,}) vs ≥ {_SPILL_FAR_UM:g} µm "
                         f"{v['far_mean']:.5f} (n = {v['n_far']:,}): {v['near_far_ratio']:.1f}× "
                         f"({'≥' if v['distance_ok'] else '<'} {_SPILL_MIN_NEAR_FAR:g}× required)")
        else:
            parts.append(f"distance gradient not testable ({v['n_near']:,} non-source cells within "
                         f"{_SPILL_NEAR_UM:g} µm, {v['n_far']:,} beyond {_SPILL_FAR_UM:g} µm; 10 of each needed)")
        if v["density_ok"] is not None:
            parts.append(f"share rises with the local source fraction: Spearman ρ = {v['rho_local']:.3f}, "
                         f"bottom third {v['local_low']:.5f} → top third {v['local_high']:.5f} "
                         f"({'yes' if v['density_ok'] else 'NO'})")
        else:
            parts.append("local-density trend not testable")
        nn = max(1, res["n_non_source"])
        fl = res["flags"]
        mk = (f"{len(markers)} markers "
              + {"user": "(listed by the user", "specific": "(derived: specific to the source",
                 "fallback": "(derived, FALLBACK: fewer than 3 genes are ≥ 10× specific, so the top "
                             "genes by ratio were used and the score is an upper bound"}[minfo["rule"]]
              + f"): {', '.join(markers[:12])}{'…' if len(markers) > 12 else ''}")
        msg = (f"Spillover of '{src}' ({cat}; {'SPLIT-corrected' if corrected else 'original'} counts; "
               f"{res['seconds']:.1f} s). {mk}. {res['reference_kind'].capitalize()} marker share inside "
               f"{src} cells: {res['reference']:.4f} (n = {res['n_source']:,} source cells; "
               f"{100 * res['source_detect']:.0f} % of them carry a marker transcript)."
               + (" ⚠ The median share is 0 - the markers are sparse - so the mean is the reference "
                  "and the per-cell score is coarse: list more, or more abundant, marker genes."
                  if res["reference_kind"] == "mean" else "") + "\n"
               f"{verdict}: " + "; ".join(parts) + ".\n"
               f"Non-source cells (n = {res['n_non_source']:,}): "
               + ", ".join(f"{k} {fl[k]:,} ({100 * fl[k] / nn:.1f} %)" for k in _SPILL_FLAG_NAMES)
               + f" (flag edges {', '.join(f'{e:g}' for e in res['edges'])} of a source cell's share).")
        if res["by_depth"]:
            msg += ("\nFlag rate (score ≥ " + f"{res['edges'][0]:g}" + ") by depth quintile, shallow → deep: "
                    + ", ".join(f"{100 * d['rate']:.1f} %" for d in res["by_depth"])
                    + " (a strong trend with depth, either way, means the score also follows cell size "
                      "or depth: a falling rate is the noise of small totals; compare cells of similar depth).")
        n_own = int((own["owner"] != "").sum())
        msg += (f"\nGene ownership: {n_own:,} of {len(own):,} genes have an owner (≥ {_OWN_MIN_COUNT} "
                f"transcripts; owner = highest rate, co-owners at ≥ {100 * float(own_frac or _OWN_FRAC):.0f} % "
                f"of it); {int((own['n_owners'] > 1).sum()):,} are shared by several types. "
                f"Color By: 'Control: Spillover score'.")
        if use_corrected and not corrected:
            msg += " Original counts were used: not every sample has SPLIT-corrected counts."
        if res["n_uncorrected"]:
            msg += " " + analysis._an_uncorrected_note(res["n_uncorrected"]) + "."
        _set("done", msg, result=res)
        print(f"  {msg}", flush=True)
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


# ─── Leak gate for differential expression ────────────────────────────────────
_LEAK_NOTE = (
    "Leak gate. Model: segmentation leakage adds leak(L) × (the profile of neighbouring type L) "
    "to the profile of the focal cells; the predicted leaked signal of a gene is Σ over the "
    "neighbour types of leak(L) × its counts per million in L. A gene is flagged when the "
    "predicted leak is more than half of its observed signal in either group, or when the leak "
    "alone predicts at least half of the observed log fold change; the second rule is applied only "
    "to genes whose change is called by the test (FDR below its threshold) or is at least 0.5 in "
    "log2 - explaining half of a change that is noise around zero means nothing. Measuring leak(L): the focal "
    "÷ L ratio of any gene is the leak plus whatever the focal type itself expresses, so the "
    "leak cannot exceed the lowest ratios; it is taken as the median of the 25 lowest focal ÷ L "
    "ratios among L's own genes (≥ 50 counts per million in L and ≥ 10× every third type). A "
    "neighbour type with fewer than 10 genes of its own, or whose own genes the focal type carries "
    "at half their level or more (shared expression: sub-types of one lineage), is not modelled, "
    "and is named: its contribution is then missing from the prediction. The original method "
    "chose these genes in "
    "a dissociated reference in which the focal type does not express them; choosing the lowest "
    "ratios in the same data leans the estimate low, so the gate errs towards not flagging, and "
    "for a focal type that shares most of its expression with a neighbour (sub-types of one "
    "lineage) it says little. It needs group A to be one cell type (≥ 80 % of its labelled "
    "cells); when group B is the same type both groups are modelled, otherwise only A. A flag "
    "says: look at the gene's owner and at the neighbours before reading it as the focal type's "
    "biology; it does not say the gene is wrong.")


def _lg(x):
    return np.log2(np.maximum(x, _LEAK_FLOOR))


def _leak_markers(prof: dict, neighbour, focal, n: int = _LEAK_N_MARKERS, min_cpm: float = _MARKER_MIN_CPM,
                  min_fold: float = _MARKER_MIN_FOLD, min_cells: int = 10) -> tuple:
    """Genes on which the leak of *neighbour* into *focal* is measured: (genes, number of
    candidate genes).

    Candidates identify the neighbour: genes at ≥ *min_cpm* in it that are ≥ *min_fold* ×
    every third type. The focal ÷ neighbour ratio of a gene is the leak plus whatever the
    focal type itself expresses, so the leak cannot exceed the lowest ratios: the *n*
    candidates with the lowest ratio are used (all of them when there are fewer). A
    neighbour with fewer than _LEAK_MIN_MARKERS candidates cannot be told apart from the
    other types and is not modelled as a leak source (the caller reports it)."""
    types = prof["types"]
    li, fi = types.index(str(neighbour)), types.index(str(focal))
    cpm = _cpm(prof["counts"], prof["depth"])
    third = [i for i in range(len(types)) if i not in (li, fi) and prof["n"][i] >= min_cells]
    oth = cpm[third].max(axis=0) if third else np.zeros(cpm.shape[1])
    idx = np.flatnonzero((cpm[li] >= float(min_cpm)) & (cpm[li] >= float(min_fold) * oth))
    ratio = cpm[fi][idx] / cpm[li][idx]
    keep = idx[np.argsort(ratio, kind="stable")][:int(n)]
    return [prof["genes"][i] for i in keep], int(len(idx))


def _leak_fraction(xF, xL, marker_idx) -> np.ndarray:
    """leak per section = median over the marker genes with CPM_L > 0 in that section of
    CPM_focal / CPM_L. xF, xL: genes × sections; NaN for a section without such a gene."""
    xF, xL = np.asarray(xF, dtype=np.float64), np.asarray(xL, dtype=np.float64)
    out = np.full(xL.shape[1], np.nan)
    for s in range(xL.shape[1]):
        lv = xL[marker_idx, s]
        ok = lv > 0
        if ok.any():
            out[s] = float(np.median(xF[marker_idx, s][ok] / lv[ok]))
    return out


def _leak_predict(leaks: dict, xLs: dict) -> tuple:
    """Predicted leaked CPM, genes × sections: Σ_L leak_L,s × CPM_L,s (a NaN leak counts
    as 0). Returns (P, per-type parts)."""
    parts = {L: np.asarray(xLs[L], dtype=np.float64) * np.nan_to_num(np.asarray(lk, dtype=np.float64))[None, :]
             for L, lk in leaks.items()}
    P = sum(parts.values()) if parts else 0.0
    return P, parts


def _leak_gate(O0, O1, P0, P1, share_max: float = _LEAK_SHARE_MAX,
               explain_min: float = _LEAK_EXPLAIN_MIN, d_obs=None, called=None,
               min_abs: float = _LEAK_CHANGE_MIN_LFC) -> dict:
    """The gate on observed (O) and predicted leaked (P) CPM of two arms (0 = reference /
    B, 1 = A); each array genes × sections of that arm, averaged over the sections.

    share = P / O per arm; e1: a share above *share_max*. Intrinsic signal i = max(O − P,
    0). d_obs = log2 O1 − log2 O0; d_leak = the change the leak alone predicts, the mean
    of (intrinsic of arm 0 with the leak of arm 1 vs arm 0) and (arm 1 vs intrinsic of
    arm 1 with the leak of arm 0); f = d_leak / d_obs; e2_all: f ≥ *explain_min*.

    The change rule needs a change: f is a ratio of two small numbers when d_obs is noise
    around 0. e2 = e2_all for the genes whose change is *called* (boolean per gene: the
    test of the calling mode passed its significance threshold) or whose |d_obs| is at
    least *min_abs* (log2); the other genes are judged by the signal rule alone. flag =
    e1 or e2 (flag_all = e1 or e2_all: the rule without that restriction, for reporting)."""
    o0, o1 = np.asarray(O0, dtype=np.float64).mean(axis=1), np.asarray(O1, dtype=np.float64).mean(axis=1)
    p0, p1 = np.asarray(P0, dtype=np.float64).mean(axis=1), np.asarray(P1, dtype=np.float64).mean(axis=1)
    s0, s1 = p0 / np.maximum(o0, _LEAK_FLOOR), p1 / np.maximum(o1, _LEAK_FLOOR)
    e1 = (s0 > share_max) | (s1 > share_max)
    i0, i1 = np.maximum(o0 - p0, 0.0), np.maximum(o1 - p1, 0.0)
    d_o = _lg(o1) - _lg(o0) if d_obs is None else np.asarray(d_obs, dtype=np.float64)
    d_leak = 0.5 * ((_lg(i0 + p1) - _lg(o0)) + (_lg(o1) - _lg(i1 + p0)))
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(d_o != 0, d_leak / d_o, np.nan)
    e2_all = np.nan_to_num(f, nan=-np.inf) >= explain_min
    change_ok = np.abs(np.nan_to_num(d_o, nan=0.0)) >= float(min_abs)
    if called is not None:
        change_ok = change_ok | np.asarray(called, dtype=bool)
    e2 = e2_all & change_ok
    return {"O0": o0, "O1": o1, "P0": p0, "P1": p1, "share0": s0, "share1": s1, "e1": e1,
            "d_obs": d_o, "d_leak": d_leak, "f": f, "e2": e2, "e2_all": e2_all, "change_ok": change_ok,
            "flag": e1 | e2, "flag_all": e1 | e2_all}


def _group_profile(samples, masks, genes) -> tuple:
    """(Σ counts per gene, Σ panel counts, cells) over the masked cells of the samples."""
    cnt, dep, n = np.zeros(len(genes)), 0.0, 0
    for s, m in zip(samples, masks):
        if m is None:
            continue
        rows = np.asarray(s["rows"])
        sel = rows[np.asarray(m, dtype=bool) & (rows >= 0)]
        if not len(sel):
            continue
        w = np.zeros(s["csc"].shape[0])
        w[sel] = 1.0
        tot = np.asarray(s["csc"].T @ w).ravel()
        cnt += tot[np.array([s["gni"][g] for g in genes], dtype=np.int64)]
        dep += float(tot[np.array([s["gni"][g] for g in s["genes"]], dtype=np.int64)].sum())
        n += int(len(sel))
    return cnt, dep, n


def _group_purity(labels, masks) -> tuple:
    """(largest type among the labelled cells of a group, its share, labelled cells)."""
    parts = [pd.Series(l, dtype=object)[np.asarray(m, dtype=bool)].dropna().astype(str)
             for l, m in zip(labels, masks) if l is not None and m is not None]
    vc = pd.concat(parts).value_counts() if parts else pd.Series(dtype=np.int64)
    if not len(vc) or not int(vc.sum()):
        return None, 0.0, 0
    return str(vc.index[0]), float(vc.iloc[0] / vc.sum()), int(vc.sum())


def _leak_gate_groups(samples, labels, masks_a, masks_b, genes, prof=None, per_sample: bool = False,
                      n_markers: int = _LEAK_N_MARKERS, min_purity: float = _LEAK_MIN_PURITY,
                      called=None) -> dict:
    """The leak gate for two groups of cells (A = arm 1, B = arm 0) of the view.

    per_sample False: every group is one pooled section, the neighbour profiles are those
    of the whole view. True (pseudobulk): every sample with cells of a group is a section
    of that group's arm, with the neighbour profiles of that sample (types with at least
    _LEAK_MIN_CELLS cells there). Returns {applicable, reason, focal, mode, table (gene,
    leak_share_a, leak_share_b, leak_change_frac, leak_fraction, leak_flag,
    top_leak_type), leaks: {type: {a, b, markers, rule}}, n_flag, n_e1, n_e2, n_e2_all,
    n_flag_all}.

    called: boolean per gene of *genes* - the change of the gene passed the significance
    threshold of the calling mode. The change rule (e2) is applied to those genes and to
    genes whose pooled |log2 change| is at least _LEAK_CHANGE_MIN_LFC; see _leak_gate.
    n_e2_all / n_flag_all count what the change rule would flag without that restriction."""
    prof = prof if prof is not None else _type_profiles(samples, labels, genes)
    out = {"applicable": False, "reason": "", "focal": None, "mode": None}
    ta, pa, na = _group_purity(labels, masks_a)
    tb, pb, nb = _group_purity(labels, masks_b)
    if ta is None:
        out["reason"] = "group A has no labelled cells"
        return out
    out["focal"], out["purity_a"], out["purity_b"], out["type_b"] = ta, pa, pb, tb
    if pa < min_purity:
        out["reason"] = (f"group A is a mixture of cell types (largest: {ta}, {100 * pa:.0f} % of its "
                         f"labelled cells; {100 * min_purity:.0f} % needed)")
        return out
    two_arm = tb == ta and pb >= min_purity
    out["mode"] = "both groups" if two_arm else "group A only"
    n_s = len(samples)
    if per_sample:
        sec_a = [i for i in range(n_s) if np.asarray(masks_a[i], dtype=bool).any()]
        sec_b = [i for i in range(n_s) if np.asarray(masks_b[i], dtype=bool).any()]
        one = lambda i, m: [m[j] if j == i else None for j in range(n_s)]
        arms = {1: [(one(i, masks_a), i) for i in sec_a], 0: [(one(i, masks_b), i) for i in sec_b]}
    else:
        arms = {1: [(masks_a, None)], 0: [(masks_b, None)]}
    if not arms[1] or not arms[0]:
        out["reason"] = "a group has no cells with counts"
        return out
    types = prof["types"]
    neigh = [t for t in types if t != ta and prof["n"][types.index(t)] >= _LEAK_MIN_CELLS]
    if not neigh:
        out["reason"] = f"no other cell type has at least {_LEAK_MIN_CELLS} cells"
        return out
    gpos = {g: i for i, g in enumerate(prof["genes"])}
    gsel = np.array([gpos[g] for g in genes], dtype=np.int64)
    marks, skipped = {}, []
    cpm_all = _cpm(prof["counts"], prof["depth"])
    for L in neigh:
        mg, n_cand = _leak_markers(prof, L, ta, n=n_markers)
        ix = np.array([gpos[g] for g in mg], dtype=np.int64)
        pooled = (float(np.median(cpm_all[types.index(ta)][ix] / cpm_all[types.index(L)][ix]))
                  if len(ix) else np.nan)
        # fewer than the minimum of own genes: L cannot be told apart from the other types;
        # the focal type at half of L's level in L's "own" genes: shared expression, not a leak
        if len(mg) >= _LEAK_MIN_MARKERS and pooled < _LEAK_MAX:
            marks[L] = (ix, mg, n_cand)
        else:
            skipped.append(L)
    out["not_modelled"] = skipped
    if not marks:
        out["reason"] = (f"no neighbour type has {_LEAK_MIN_MARKERS} genes of its own (≥ {_MARKER_MIN_CPM:g} "
                         f"counts per million and ≥ {_MARKER_MIN_FOLD:g}× every third type) that the focal "
                         f"type carries at less than {100 * _LEAK_MAX:.0f} % of its level: the annotation "
                         f"does not separate the types well enough to measure a leak")
        return out
    O, P, leaks, parts_all = {}, {}, {L: {} for L in marks}, {}
    for arm in (0, 1):
        cols_o, xl = [], {L: [] for L in marks}
        for masks, si in arms[arm]:
            cnt, dep, _n = _group_profile(samples, masks, prof["genes"])
            cols_o.append(1e6 * cnt / max(dep, 1e-12))
            src = prof if si is None else prof["per_sample"][si]
            for L in marks:
                li = types.index(L)
                if src["n"][li] >= (_LEAK_MIN_CELLS if si is not None else 1) and src["depth"][li] > 0:
                    xl[L].append(1e6 * src["counts"][li] / src["depth"][li])
                else:
                    xl[L].append(np.zeros(len(prof["genes"])))
        xF = np.column_stack(cols_o)
        xLs = {L: np.column_stack(v) for L, v in xl.items()}
        lk = {L: _leak_fraction(xF, xLs[L], marks[L][0]) for L in marks}
        if arm == 0 and not two_arm:
            lk = {L: np.zeros_like(v) for L, v in lk.items()}
        Pm, parts = _leak_predict(lk, xLs)
        O[arm], P[arm], parts_all[arm] = xF, np.asarray(Pm) + np.zeros_like(xF), parts
        for L in marks:
            leaks[L]["a" if arm == 1 else "b"] = float(np.nanmedian(lk[L])) if np.isfinite(lk[L]).any() else np.nan
    for L in marks:
        leaks[L].update({"markers": marks[L][1], "rule": "specific", "n_candidates": marks[L][2]})
    called_all = None
    if called is not None:
        called_all = np.zeros(len(prof["genes"]), dtype=bool)
        called_all[gsel] = np.nan_to_num(np.asarray(called, dtype=np.float64), nan=0.0) > 0
    g = _leak_gate(O[0], O[1], P[0], P[1], called=called_all)
    contrib = np.column_stack([parts_all[1][L].mean(axis=1) + parts_all[0][L].mean(axis=1) for L in marks])
    top = np.array(list(marks), dtype=object)[contrib.argmax(axis=1)]
    top = np.where(contrib.sum(axis=1) > 0, top, "")
    # the change fraction counts towards the displayed leak fraction only where the change rule applies
    frac = np.fmax(np.fmax(g["share1"], g["share0"]),
                   np.where(np.isfinite(g["f"]) & g["change_ok"], g["f"], -np.inf))
    frac = np.clip(frac, 0.0, None)
    tab = pd.DataFrame({"gene": prof["genes"], "leak_share_a": g["share1"], "leak_share_b": g["share0"],
                        "leak_change_frac": g["f"], "leak_fraction": frac, "leak_flag": g["flag"],
                        "top_leak_type": top}).iloc[gsel].reset_index(drop=True)
    out.update({"applicable": True, "table": tab, "leaks": leaks, "n_flag": int(tab["leak_flag"].sum()),
                "n_e1": int(g["e1"][gsel].sum()), "n_e2": int(g["e2"][gsel].sum()),
                "n_e2_all": int(g["e2_all"][gsel].sum()), "n_flag_all": int(g["flag_all"][gsel].sum()),
                "n_change_ok": int(g["change_ok"][gsel].sum()),
                "n_sections_a": len(arms[1]), "n_sections_b": len(arms[0])})
    return out


def _de_leak_annotate(res: dict, ctx: dict) -> None:
    """DE annotator (analysis._de_annotators): adds owner / owners and, when the gate
    applies, leak_share_a / leak_share_b / leak_change_frac / leak_fraction / leak_flag /
    top_leak_type to the result table, and res["leak_summary"]."""
    if not (ctx.get("options") or {}).get("leak_gate", True):
        return
    samples, seg = ctx["samples"], ctx.get("seg_source")
    alt = analysis._an_alt_res(seg)
    am = analysis._an_annot_method(ctx.get("color_by"), samples, alt)
    labels = analysis._an_celltype_labels(samples, am, alt)
    if all(l is None for l in labels):
        res["leak_summary"] = ("Gene ownership / leak gate: not computed (the active segmentation has "
                               "no cell-type annotation).")
        return
    t = res["table"]
    genes = list(t["gene"])
    prof = _type_profiles(samples, labels, analysis._an_common_genes(samples))
    jobs._job_check()
    own = _gene_ownership(prof).set_index("gene").reindex(genes)
    t["owner"] = own["owner"].fillna("").to_numpy()
    t["owners"] = own["owners"].fillna("").to_numpy()
    ma, mb = ctx.get("masks_a"), ctx.get("masks_b")
    A, B = [], []
    for i, s in enumerate(samples):
        a = None if ma is None else np.asarray(ma[i], dtype=bool)
        b = None if mb is None else np.asarray(mb[i], dtype=bool)
        if a is None:
            a = ~b
        if b is None:
            b = ~a
        A.append(a)
        B.append(b & ~a)
    name = core._ANNOT_METHODS.get(am, am)
    called = np.nan_to_num(t["padj"].to_numpy(dtype=float), nan=1.0) < 0.05
    g = _leak_gate_groups(samples, labels, A, B, genes, prof, per_sample=bool(ctx.get("per_sample")),
                          called=called)
    n_own = int((t["owner"] != "").sum())
    head = (f"Gene ownership ({name} labels): {n_own:,} of {len(t):,} genes have an owner (highest "
            f"Σ counts / Σ depth; ≥ {_OWN_MIN_COUNT} transcripts).")
    if not g["applicable"]:
        res["leak_summary"] = head + f" Leak gate not applied: {g['reason']}."
        res["leak_gate"] = g
        return
    lt = g["table"].set_index("gene").reindex(genes)
    for c in ("leak_share_a", "leak_share_b", "leak_change_frac", "leak_fraction", "top_leak_type"):
        t[c] = lt[c].to_numpy()
    t["leak_flag"] = lt["leak_flag"].fillna(False).to_numpy(dtype=bool)
    sig = t["padj"].to_numpy(dtype=float) < 0.05
    fl = t["leak_flag"].to_numpy(dtype=bool)
    not_owned = sig & (t["owner"].to_numpy() != "") & (t["owner"].to_numpy() != g["focal"]) \
        & ~np.array([g["focal"] in str(o).split(";") for o in t["owners"]])
    lk = sorted(g["leaks"].items(), key=lambda kv: -(kv[1].get("a") or 0))
    fb = list(g.get("not_modelled") or [])
    res["leak_gate"] = {k: v for k, v in g.items() if k != "table"}
    res["leak_summary"] = (
        head + f" Leak gate (focal type {g['focal']}, {g['mode']} modelled; "
        f"{g['n_sections_a']} + {g['n_sections_b']} section(s)): {g['n_flag']:,} of {len(t):,} genes "
        f"flagged ({g['n_e1']:,} with predicted leak > {100 * _LEAK_SHARE_MAX:.0f} % of the signal, "
        f"{g['n_e2']:,} with ≥ {100 * _LEAK_EXPLAIN_MIN:.0f} % of the change explained - a rule applied to the "
        f"{g['n_change_ok']:,} genes with FDR < 0.05 or |log2 change| ≥ {_LEAK_CHANGE_MIN_LFC:g}); of the "
        f"{int(sig.sum()):,} genes with FDR < 0.05, {int((sig & fl).sum()):,} are flagged and "
        f"{int(not_owned.sum()):,} are not owned by {g['focal']}. Leak into the focal cells (median "
        f"of the {_LEAK_N_MARKERS} lowest focal ÷ neighbour ratios among the neighbour's genes): "
        + ", ".join(f"{L} {100 * v['a']:.1f} %" for L, v in lk[:8] if np.isfinite(v.get("a", np.nan)))
        + ("…" if len(lk) > 8 else "") + "."
        + (f" Not modelled as leak sources (fewer than {_LEAK_MIN_MARKERS} genes of their own, or the focal "
           f"type carries those genes at ≥ {100 * _LEAK_MAX:.0f} % of their level - shared expression, not a "
           f"leak; their contribution is missing from the prediction): {', '.join(fb[:8])}"
           f"{'…' if len(fb) > 8 else ''}."
           if fb else ""))


analysis._de_annotators.append(_de_leak_annotate)


# ─── Correction / segmentation QC ─────────────────────────────────────────────
_SEGQC_NOTE = (
    "One row per segmentation source (Xenium and every cached Baysor / Proseg run of the primary "
    "sample) and counts mode (original, SPLIT-corrected). Identity marker sets: the most specific "
    "genes of every cell type, derived once from the Xenium segmentation's original counts and "
    "annotation (or the sets you list: one 'Type: gene, gene, …' per line) and applied to every "
    "source. Exclusion-rule failure: share of labelled cells whose own identity set is not the "
    "largest share of their transcripts (some foreign set is at least as large). Marker "
    "co-expression: over the pairs of top markers of two different types, cells with both ≥ 1 "
    "transcript ÷ cells with either. Spillover slope: in cells not of the source type, source-"
    "identity transcripts per 100 transcripts regressed on the number of source-type neighbours "
    "within the radius (reference-free; 0 = no dependence on neighbours); the reverse slope is "
    "foreign identity transcripts in source cells against non-source neighbours. Own-identity "
    "retention (corrected counts only): own identity transcripts after ÷ before correction (the row "
    "value pools all types: Σ after ÷ Σ before; a type without such transcripts before has none), which "
    "guards against over-correction; own identity per 100 is given for every row. A corrected row "
    "is computed on the cells the correction covered (a non-empty corrected row; their share is "
    "'Corrected share') and its retention compares the same cells before and after. A run without "
    "its own annotation gets the Xenium labels of the nearest Xenium centroid (within 10 µm), "
    "which favours segmentations close to Xenium's: compare such rows with care. Lower "
    "contamination with a retention well below 1 is over-correction, not improvement. "
    "Assigned (original counts only): Σ counts of the source's cells ÷ gene transcripts (Q ≥ 20) "
    "inside the bounding box of its cells, grown by one median cell radius; the transcripts of "
    "the section are counted once from the transcript file (the 'All transcripts' grid of the "
    "Transcript Density tab, kept in the cache), which takes about as long as that tab's scan "
    "the first time; a dataset loaded from a .zarr has no transcript file and no such column. "
    "The metrics that need a paired dissociated reference are not computed.")


def _identity_sets(prof: dict, n: int = _QC_SET_SIZE, min_cells: int = 10) -> dict:
    """{type: its *n* most specific genes} for every type with at least *min_cells* cells
    and at least 2 such genes (see _markers_of)."""
    out = {}
    for t, k in zip(prof["types"], prof["n"]):
        if k < min_cells:
            continue
        g, _info = _markers_of(prof, t, n=n)
        if len(g) >= 2:
            out[t] = g
    return out


def _parse_identity_sets(text, genes) -> tuple:
    """({type: [genes on the panel]}, [(gene, gene)] marker pairs) from text: a line 'Type:
    gene, gene, …' is an identity set; a line of exactly two genes 'geneA, geneB' is a pair
    of mutually exclusive markers."""
    sets, pairs = {}, []
    have = {g.lower(): g for g in genes}
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            name, rest = line.split(":", 1)
            gs = [have[x.lower()] for x in re.split(r"[,;\s]+", rest.strip()) if x.lower() in have]
            if name.strip() and gs:
                sets[name.strip()] = list(dict.fromkeys(gs))
        else:
            gs = [x for x in re.split(r"[,;\s]+", line) if x]
            if len(gs) == 2 and all(x.lower() in have for x in gs):
                pairs.append((have[gs[0].lower()], have[gs[1].lower()]))
    return sets, pairs


def _transfer_labels(xy_ref, labels_ref, xy_new, max_dist: float = _QC_TRANSFER_UM) -> np.ndarray:
    """Label of the nearest labelled reference cell within *max_dist* µm (None beyond),
    for cells of another segmentation of the SAME section."""
    from scipy.spatial import cKDTree
    lab = pd.Series(labels_ref, dtype=object)
    ok = lab.notna().to_numpy()
    out = np.full(len(xy_new), None, dtype=object)
    if not ok.any() or not len(xy_new):
        return out
    d, i = cKDTree(np.asarray(xy_ref)[ok]).query(np.asarray(xy_new), k=1, distance_upper_bound=float(max_dist))
    hit = np.isfinite(d)
    out[hit] = lab.to_numpy()[ok][i[hit]]
    return out


def _qc_metrics(s: dict, labels, sets: dict, source_type=None, radius: float = _QC_RADIUS_UM,
                pairs=None, ref_ident=None) -> dict:
    """QC metrics of ONE segmentation source / counts mode (see _SEGQC_NOTE).

    s: an analysis sample (csc, rows, gni, genes, df); labels: label per cell (None:
    unlabelled); sets: {type: identity genes}. Returns {row (the summary), per_type
    (DataFrame: type, n, exclusion_failure, own_per_100, retention), ident_sums ({type: Σ
    own identity counts}; pass it as *ref_ident* for the corrected counts of the same
    source)}."""
    csc = s["csc"]
    rows = np.asarray(s["rows"])
    okr = rows >= 0
    r = np.clip(rows, 0, None)
    setk = [k for k in sets if any(g in s["gni"] for g in sets[k])]
    member = np.zeros((csc.shape[1], len(setk)))
    n_in = []
    for j, k in enumerate(setk):
        cols = [s["gni"][g] for g in sets[k] if g in s["gni"] and s["gni"][g] < csc.shape[1]]
        member[cols, j] = 1.0
        n_in.append(len(cols))
    panel = np.zeros(csc.shape[1])
    panel[[s["gni"][g] for g in s["genes"]]] = 1.0
    ident = np.asarray(csc @ member)[r] * okr[:, None]
    tot = np.asarray(csc @ panel).ravel()[r] * okr
    # genes per cell: panel genes with a count ≥ 1 transcript (no copy of the matrix)
    hit = csc.data >= analysis._DETECT_MIN
    if panel.all():
        n_row = np.bincount(csc.indices[hit], minlength=csc.shape[0])
    else:
        col_of = np.repeat(np.arange(csc.shape[1]), np.diff(csc.indptr))
        n_row = np.bincount(csc.indices[hit & (panel[col_of] > 0)], minlength=csc.shape[0])
    n_genes = n_row[r] * okr
    lab = pd.Series(labels, dtype=object) if labels is not None else pd.Series([None] * len(rows), dtype=object)
    labelled = lab.notna().to_numpy() & okr
    L = lab.astype(str).to_numpy()
    share = ident / np.maximum(tot, 1e-9)[:, None]
    df = s["df"]
    row = {"n_cells": int(okr.sum()), "n_labelled": int(labelled.sum()),
           "median_transcripts": float(np.median(tot[okr])) if okr.any() else np.nan,
           "median_genes": float(np.median(n_genes[okr])) if okr.any() else np.nan,
           "median_area": (float(np.nanmedian(pd.to_numeric(df["cell_area"], errors="coerce")))
                           if "cell_area" in df.columns else np.nan),
           "total_transcripts": float(tot.sum())}
    per, ident_sums, fails, n_fail = [], {}, 0.0, 0
    for k in setk:
        c = labelled & (L == k)
        if not c.any():
            continue
        j = setk.index(k)
        foreign_cols = [setk.index(o) for o in setk if o != k and n_in[setk.index(o)] >= 2]
        own = share[c][:, j]
        foreign = share[c][:, foreign_cols].max(axis=1) if foreign_cols else np.zeros(int(c.sum()))
        fail = float(np.mean(~(foreign < own)))
        own_sum = float(ident[c, j].sum())
        ident_sums[k] = own_sum
        # undefined (not 0, not 1e9) when the type's cells had no own-identity transcript before
        ret = ((own_sum / ref_ident[k]) if (ref_ident is not None and ref_ident.get(k, 0) > 0) else np.nan)
        per.append({"type": k, "n": int(c.sum()), "exclusion_failure": fail,
                    "own_per_100": float(100.0 * own_sum / max(tot[c].sum(), 1e-9)), "retention": ret})
        fails += fail * int(c.sum())
        n_fail += int(c.sum())
    per = pd.DataFrame(per, columns=["type", "n", "exclusion_failure", "own_per_100", "retention"])
    row["exclusion_failure"] = fails / n_fail if n_fail else np.nan
    row["exclusion_failure_mean_type"] = float(per["exclusion_failure"].mean()) if len(per) else np.nan
    row["own_per_100"] = (float(100.0 * sum(ident_sums.values())
                                / max(sum(tot[labelled & (L == k)].sum() for k in ident_sums), 1e-9))
                          if ident_sums else np.nan)
    # the row value is the pooled ratio Σ after / Σ before over the types with a reference:
    # a type of two cells does not weigh as much as one of twenty thousand
    _rk = [k for k in ident_sums if ref_ident is not None and ref_ident.get(k, 0) > 0]
    row["retention"] = (float(sum(ident_sums[k] for k in _rk) / sum(ref_ident[k] for k in _rk))
                        if _rk else np.nan)
    # co-expression of mutually exclusive markers
    if not pairs:
        tops = [(k, sets[k][0]) for k in setk if sets[k] and sets[k][0] in s["gni"]]
        pairs = [(tops[a][1], tops[b][1]) for a in range(len(tops)) for b in range(a + 1, len(tops))
                 if tops[a][1] != tops[b][1]]
    both = either = 0.0
    if pairs:
        gs = list(dict.fromkeys(g for pr in pairs for g in pr if g in s["gni"]))
        D = csc[:, [s["gni"][g] for g in gs]].tocsc()
        D.data = (D.data >= analysis._DETECT_MIN).astype(np.float32)
        pos = {g: i for i, g in enumerate(gs)}
        in_view = np.zeros(csc.shape[0], dtype=bool)
        in_view[rows[okr]] = True
        dcols = {g: np.asarray(D[:, pos[g]].todense()).ravel().astype(bool) & in_view for g in gs}
        for a, b in pairs[:2000]:
            if a in dcols and b in dcols:
                both += float((dcols[a] & dcols[b]).sum())
                either += float((dcols[a] | dcols[b]).sum())
    row["coexpression_rate"] = both / either if either else np.nan
    row["n_marker_pairs"] = len(pairs)
    # spatial spillover slope
    row.update({"spill_slope": np.nan, "spill_mean": np.nan, "reverse_slope": np.nan, "reverse_mean": np.nan})
    if source_type in setk and labelled.any():
        xy = stats._xy_of(df)
        is_src = labelled & (L == source_type)
        j = setk.index(source_type)
        c = labelled & ~is_src & (tot > 0)
        if int(c.sum()) >= 10 and is_src.any():
            n_src = _local_counts(xy, radius, is_src)
            y = 100.0 * ident[c, j] / tot[c]
            if np.ptp(n_src[c]) > 0:
                row["spill_slope"] = float(np.polyfit(n_src[c], y, 1)[0])
            row["spill_mean"] = float(y.mean())
        c = is_src & (tot > 0)
        others = [setk.index(o) for o in setk if o != source_type]
        if int(c.sum()) >= 10 and others:
            n_non = _local_counts(xy, radius, labelled & ~is_src)
            y = 100.0 * ident[c][:, others].sum(axis=1) / tot[c]
            if np.ptp(n_non[c]) > 0:
                row["reverse_slope"] = float(np.polyfit(n_non[c], y, 1)[0])
            row["reverse_mean"] = float(y.mean())
    return {"row": row, "per_type": per, "ident_sums": ident_sums}


def _ensure_all_tx_grid(gen=None, progress=None) -> str:
    """Make sure the all-transcripts density grid of the primary sample exists (the
    denominator of the fraction of transcripts assigned): read once from the transcript
    file with the Transcript Density scan and kept in the same cache file, so the
    Transcript Density tab finds it too. Returns "" when the grid is available, else the
    reason it is not. Cancellable (the scan checks the job flag per batch); nothing is
    written when the job is cancelled or the dataset changed."""
    if analysis._density_grid(analysis._DENSITY_ALL, analysis._DENSITY_BASE_UM) is not None:
        return ""
    data_dir = core.DATA.get("data_dir")
    if not data_dir:
        return "the dataset was loaded from a .zarr, which has no transcript file"
    with analysis._density_lock:
        busy = analysis._density_state.get("status") == "running"
    if busy:
        return "a Transcript Density job is running; run Segmentation QC again when it has finished"
    try:
        scan = analysis._density_scan(data_dir, analysis._DENSITY_ALL, progress=progress)
    except ValueError as exc:
        return str(exc).rstrip(".")
    if gen is not None:
        jobs._check_dataset_gen(gen)
    jobs._job_check()
    analysis._density_write(analysis._density_cache_file(analysis._DENSITY_ALL), scan, analysis._DENSITY_ALL)
    with analysis._density_lock:
        analysis._density_state["version"] += 1
    return ""


def _frac_assigned(s: dict) -> float:
    """Fraction of transcripts assigned to cells: Σ counts of the source's cells ÷ gene
    transcripts (Q ≥ 20, the filter of the cell-feature matrix) inside the footprint of
    its cells, from the all-transcripts density grid of the primary sample (5 µm bins).
    The footprint is the bounding box of the centroids of the source's cells, grown by one
    median cell radius (√(median cell area / π)) when the table has cell areas, so that
    transcripts of the outermost cells are inside it. NaN when the grid does not exist
    (see _ensure_all_tx_grid) or the box holds no transcripts. For a segmentation of the
    whole section this is the share of the tissue's transcripts that ended up in a cell;
    for a run on a region it is that share within the region's bounding box (a box is
    larger than a non-rectangular region, which lowers the value)."""
    g = analysis._density_grid(analysis._DENSITY_ALL, analysis._DENSITY_BASE_UM)
    if g is None or not g["grid"].size:
        return np.nan
    rows = np.asarray(s["rows"])
    ok = rows >= 0
    if not ok.any():
        return np.nan
    df = s["df"]
    xy = stats._xy_of(df)[ok]
    pad = 0.0
    if "cell_area" in df.columns:
        area = float(np.nanmedian(pd.to_numeric(df["cell_area"], errors="coerce").to_numpy(dtype=np.float64)[ok]))
        if np.isfinite(area) and area > 0:
            pad = float(np.sqrt(area / np.pi))
    b = float(g["bin"])
    x0 = int(np.floor((xy[:, 0].min() - pad) / b)) - g["ix0"]
    x1 = int(np.floor((xy[:, 0].max() + pad) / b)) - g["ix0"]
    y0 = int(np.floor((xy[:, 1].min() - pad) / b)) - g["iy0"]
    y1 = int(np.floor((xy[:, 1].max() + pad) / b)) - g["iy0"]
    sub = g["grid"][max(y0, 0):max(y1 + 1, 0), max(x0, 0):max(x1 + 1, 0)]
    den = float(sub.sum())
    if den <= 0:
        return np.nan
    w = np.zeros(s["csc"].shape[0])
    w[rows[ok]] = 1.0
    panel = np.array([s["gni"][x] for x in s["genes"]], dtype=np.int64)
    return float(np.asarray(s["csc"].T @ w).ravel()[panel].sum() / den)


def _segqc_sources(run_values=None) -> list:
    """The segmentation sources of the primary sample to compare: [{label, value, tool,
    sample (original counts), corrected (sample with the SPLIT-corrected matrix or None),
    alt (stand-in result dict for label keys) or None}]. Cached runs that are not loaded
    are read from their zarr store."""
    import os
    out = []
    xs = analysis._an_samples("xenium", False)
    prim = next((s for s in xs if s["ds"] is core.DATA), None)
    if prim is not None:
        corr = None
        if core.DATA.get("split_corrected_expr") is not None:
            m = core.DATA["split_corrected_expr"]
            rows = prim["rows"]
            if m.shape[0] != core.DATA["expr"].shape[0]:
                rows = np.arange(len(prim["df"])) if m.shape[0] == len(prim["df"]) else None
            if rows is not None:
                corr = {**prim, "csc": figures._csc_view(core.DATA, "split_corrected_expr"),
                        "rows": np.asarray(rows), "corrected": True}
        out.append({"label": "Xenium", "value": "xenium", "tool": "xenium", "sample": prim,
                    "corrected": corr, "alt": None})
    panel = dataview._panel_gene_names()
    for o in caching._list_cached_seg_runs():
        if run_values is not None and o["value"] not in run_values:
            continue
        tool = o["value"].split(":")[0]
        state, lock = core._seg_state(tool)
        with lock:
            loaded = state.get("result")
        if loaded is not None and os.path.realpath(str(loaded.get("out_dir", ""))) == os.path.realpath(o["out_dir"]):
            df, expr, xcorr = loaded["cells_df"], loaded.get("expr"), loaded.get("split_corrected_expr")
            gni = loaded.get("gene_name_to_idx") or core.DATA.get("gene_name_to_idx", {})
        else:
            zarr_path = os.path.join(o["out_dir"], f"spatialdata_{tool}.zarr")
            if not os.path.isdir(zarr_path):
                out.append({"label": o["label"], "value": o["value"], "tool": tool, "sample": None,
                            "corrected": None, "alt": None, "error": "no SpatialData store in the cache"})
                continue
            try:
                df, expr, xcorr, _imp, _b = reseg._fast_load_reseg_zarr(zarr_path)
            except Exception as exc:
                out.append({"label": o["label"], "value": o["value"], "tool": tool, "sample": None,
                            "corrected": None, "alt": None, "error": f"could not read the store: {str(exc)[:120]}"})
                continue
            gni = (reseg._reseg_zarr_gene_index(zarr_path, expr.shape[1]).get("gene_name_to_idx")
                   or core.DATA.get("gene_name_to_idx", {}))
        if expr is None or not sp.issparse(expr):
            continue

        def _smp(m, corrected):
            csc = m.tocsc()
            return {"label": o["label"], "name": o["label"], "key": dataview._ds_key(core.DATA), "df": df,
                    "csc": csc, "rows": np.arange(len(df)), "gni": gni, "x_offset": 0.0,
                    "genes": [g for g in panel if g in gni and gni[g] < csc.shape[1]],
                    "corrected": corrected, "ds": core.DATA}

        ok_corr = xcorr is not None and sp.issparse(xcorr) and xcorr.shape[0] == len(df)
        out.append({"label": o["label"], "value": o["value"], "tool": tool, "sample": _smp(expr, False),
                    "corrected": _smp(xcorr, True) if ok_corr else None,
                    "alt": {"out_dir": o["out_dir"], "source": tool}})
    return out


def _run_segqc(method, sets_text, source_type, radius, run_values=None) -> None:
    """Background job: the QC comparison table over segmentation sources and counts modes."""
    _gen = jobs._dataset_gen

    def _set(status, message, result=None):
        with _segqc_lock:
            if jobs._stale_job(_gen, status, message):
                return
            _segqc_state["status"], _segqc_state["message"] = status, message
            if result is not None:
                _segqc_state["result"] = result
            if status in ("done", "error"):
                _segqc_state["version"] += 1
        if status == "done":
            provenance._prov_record("segqc", params={"annotation_method": method, "gene_sets": sets_text, "source_type": source_type, "radius": radius}, message=message, result=result)

    try:
        _t0 = _time.time()
        _set("running", "Collecting the segmentation sources…")
        radius = float(radius or _QC_RADIUS_UM)
        srcs = _segqc_sources(run_values)
        xen = next((s for s in srcs if s["value"] == "xenium" and s["sample"] is not None), None)
        if xen is None:
            _set("error", "The primary sample has no expression matrix.")
            return
        with core._annot_lock:
            ref_labels = core._annot_state.get(f"labels_{method}") if method else None
        if ref_labels is None:
            _set("error", "The Xenium segmentation has no annotation of the chosen method: run Cell "
                          "Type Annotation first (the identity marker sets and the labels come from it).")
            return
        xdf = xen["sample"]["df"]
        xlab = ref_labels.reindex(xdf.index.astype(str)).astype(object).to_numpy()
        xlab[pd.isna(xlab)] = None
        prof = _type_profiles([xen["sample"]], [xlab])
        user_sets, user_pairs = _parse_identity_sets(sets_text, prof["genes"])
        sets = user_sets or _identity_sets(prof)
        if len(sets) < 2:
            _set("error", "Fewer than 2 cell types have identity marker genes: list the sets "
                          "('Type: gene, gene, …' per line).")
            return
        missing_types = [k for k in sets if k not in prof["types"]]
        src_type = (str(source_type) if source_type and str(source_type) != "auto"
                    else max((t for t in prof["types"] if t in sets),
                             key=lambda t: prof["depth"][prof["types"].index(t)], default=None))
        rows, per_all, notes = [], [], []
        n_tool: dict = {}
        for src in srcs:                      # short unique names for the plot and the table
            if src["tool"] == "xenium":
                src["short"] = "Xenium"
            else:
                n_tool[src["tool"]] = n_tool.get(src["tool"], 0) + 1
                src["short"] = f"{'Image' if src['tool'] == 'imgseg' else src['tool'].capitalize()} {n_tool[src['tool']]}"
        n_steps = sum(1 + (s["corrected"] is not None) for s in srcs if s["sample"] is not None)
        step = 0
        # the denominator of "fraction of transcripts assigned": all gene transcripts of the
        # section, binned once (cached) - computed here when the grid does not exist yet
        scan_needed = (analysis._density_grid(analysis._DENSITY_ALL, analysis._DENSITY_BASE_UM) is None
                       and bool(core.DATA.get("data_dir")))
        if scan_needed:
            _set("running", "Segmentation QC: counting all transcripts of the section (once; kept in the cache)…")
        no_grid = _ensure_all_tx_grid(_gen, progress=lambda f: jobs._job_progress(
            "Counting all transcripts of the section (once)…", 0.5 * f))
        f0 = 0.5 if scan_needed and not no_grid else 0.0
        for src in srcs:
            if src["sample"] is None:
                notes.append(f"{src['label']}: {src.get('error', 'not available')}")
                continue
            jobs._job_check()
            s0 = src["sample"]
            if src["alt"] is None:
                lab, origin = xlab, "own"
            else:
                with core._annot_lock:
                    own = core._annot_state.get(annotation._labels_key_for_method(method, src["alt"]))
                lab = None
                if own is not None:
                    lab = own.reindex(pd.Index(s0["df"].index.astype(str))).astype(object).to_numpy()
                    lab[pd.isna(lab)] = None
                    origin = "own"
                    known = pd.Series(lab, dtype=object).dropna()
                    if not len(known) or float(known.astype(str).isin(set(sets)).mean()) < 0.5:
                        notes.append(f"{src['label']}: its own {core._ANNOT_METHODS.get(method, method)} "
                                     f"labels use other type names than the identity sets (another "
                                     f"model or reference); Xenium labels were transferred instead")
                        lab = None
                if lab is None:
                    lab = _transfer_labels(stats._xy_of(xdf), xlab, stats._xy_of(s0["df"]))
                    origin = f"Xenium labels, nearest centroid ≤ {_QC_TRANSFER_UM:g} µm"
            for mode, smp in (("original", s0), ("SPLIT-corrected", src["corrected"])):
                if smp is None:
                    continue
                jobs._job_check()
                step += 1
                _set("running", f"Segmentation QC: {src['label']} ({mode} counts)…")
                jobs._job_progress(f"{src['label']} ({mode})", f0 + (1.0 - f0) * (step - 1) / max(1, n_steps))
                ref_ident, coverage = None, np.nan
                if mode != "original":
                    # a correction may cover part of the cells (those the deconvolution
                    # classified); the others have an empty corrected row. The corrected
                    # row is scored on the covered cells, against the same cells' original.
                    rws = np.asarray(smp["rows"])
                    tot_c = np.asarray(smp["csc"].sum(axis=1)).ravel()
                    covered = (rws >= 0) & (tot_c[np.clip(rws, 0, None)] > 0)
                    coverage = float(covered.sum() / max(1, int((rws >= 0).sum())))
                    if not covered.any():
                        notes.append(f"{src['label']}: the corrected matrix is empty")
                        continue
                    smp = {**smp, "rows": np.where(covered, rws, -1)}
                    r0 = np.asarray(s0["rows"])
                    ref_ident = _qc_metrics({**s0, "rows": np.where(covered, r0, -1)}, lab, sets, None,
                                            radius, user_pairs or None)["ident_sums"]
                m = _qc_metrics(smp, lab, sets, src_type, radius, user_pairs or None, ref_ident)
                row = {"id": src["short"], "source": src["label"], "value": src["value"], "counts": mode,
                       "labels": origin, **m["row"], "coverage": coverage,
                       "frac_assigned": _frac_assigned(smp) if mode == "original" else np.nan}
                rows.append(row)
                pt = m["per_type"].copy()
                pt.insert(0, "counts", mode)
                pt.insert(0, "source", src["label"])
                per_all.append(pt)
        jobs._check_dataset_gen(_gen)
        if not rows:
            _set("error", "No segmentation source could be scored. " + " ".join(notes))
            return
        table = pd.DataFrame(rows)
        res = {"table": table, "per_type": pd.concat(per_all, ignore_index=True) if per_all else pd.DataFrame(),
               "sets": sets, "sets_origin": "user" if user_sets else "derived", "source_type": src_type,
               "radius": radius, "method": method, "notes": notes, "seconds": _time.time() - _t0}
        msg = (f"{len(table)} rows: {table['source'].nunique()} segmentation source(s) × counts mode "
               f"({res['seconds']:.1f} s). Identity sets ({res['sets_origin']}, {len(sets)} types, "
               f"{core._ANNOT_METHODS.get(method, method)} labels of the Xenium segmentation); spillover "
               f"source type: {src_type or 'none'}; radius {radius:g} µm. Units: cells of one sample; "
               f"these are descriptive metrics, no p-value is computed.")
        if missing_types:
            msg += f" Listed types without cells: {', '.join(missing_types[:8])}."
        tr = table[table["labels"] != "own"]["source"].unique()
        if len(tr):
            msg += (f" Labels transferred from Xenium by nearest centroid for: {', '.join(tr[:6])}"
                    f"{'…' if len(tr) > 6 else ''}.")
        if no_grid:
            msg += f" Fraction of transcripts assigned: not available ({no_grid})."
        elif table["frac_assigned"].notna().any():
            msg += (" Assigned = Σ counts of the cells ÷ gene transcripts (Q ≥ 20) in the bounding box of "
                    "the cells (grown by one median cell radius), original counts only.")
        for nt in notes:
            msg += f"\nNote: {nt}."
        _set("done", msg, result=res)
        print(f"  Segmentation QC: {msg}", flush=True)
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


# ─── Stratified co-localisation with matched nulls ────────────────────────────
_COLOC_NOTE = (
    "For every ordered pair (focal type A, target type B): O = the number of B cells within the "
    "radius of each A cell; E = the mean of that count around the POOL cells (labelled cells that "
    "are neither A nor B) of the same stratum: sample × compartment × edge-distance bin × local-"
    "density quintile (coarser strata when fewer than 3 pool cells share the stratum). Effect per "
    "sample = log2((ΣO + 0.5) / (ΣE + 0.5)): 0 means A has as many B neighbours as other cells at "
    "the same place in the tissue. Stratification is what removes 'both types live in the same "
    "region': the unstratified value (pool = the whole sample) is given beside it as 'naive'. "
    "Bias and within-sample p: the A label is permuted among A and pool cells within each finest "
    "stratum (B fixed); the mean permuted effect is subtracted, and p = (1 + permutations at "
    "least as far from that mean) / (1 + permutations). That p treats the cells of one section "
    "as exchangeable: it says the arrangement in THIS section is unlikely by chance, and nothing "
    "about other sections or donors. With at least 3 independent units (samples, or donors when "
    "given) the effect is tested across them with the exact sign-flip test (enrichment) and, "
    "between conditions, with the paired sign-flip or label-permutation test of the Samples & "
    "conditions design; the status states the unit, the smallest achievable p and whether any "
    "FDR below 0.05 can be reached. Placebo anchors (optional focal type): the same statistic "
    "for count-matched random anchor cells drawn within each sample × compartment, and for the "
    "other types of similar abundance in the same compartment. A proximity claim for the focal "
    "type needs it to stand out from both; otherwise the result says 'a cell of this compartment "
    "sits there', not 'this cell type does'. Neighbours are counted within each sample only. "
    "Compartments come from the Confound Controls tab (computed with defaults when absent).")


def _nanmean(a, axis=None):
    """np.nanmean without the warning for an all-NaN slice (the result is NaN, as wanted)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _rank_bin(v, k: int) -> np.ndarray:
    """Quantile bin 0 … k−1 of every value by average rank (ties share a bin boundary)."""
    v = np.asarray(v)
    if not len(v):
        return np.zeros(0, dtype=np.int64)
    r = pd.Series(v).rank(method="average").to_numpy()
    return np.minimum(((r - 0.5) / len(v) * k).astype(np.int64), k - 1)


def _compact(*cols) -> np.ndarray:
    """One compact integer code (0 … m−1) per distinct combination of integer columns."""
    key = np.zeros(len(cols[0]), dtype=np.int64)
    for c in cols:
        c = np.asarray(c, dtype=np.int64)
        key = key * (int(c.max()) + 1 if len(c) else 1) + c
    return np.unique(key, return_inverse=True)[1].astype(np.int64)


def _coloc_levels(sample, comp, edge_um, density, edge_bins=_COLOC_EDGE_BINS,
                  n_q: int = _COLOC_DENSITY_Q) -> list:
    """Stratum codes per cell, finest first: sample × compartment × edge bin × density
    quantile; sample × compartment × edge bin; sample × compartment; sample. The density
    quantile is taken within the sample."""
    sample = np.asarray(sample, dtype=np.int64)
    ebin = np.searchsorted(np.asarray(edge_bins, dtype=np.float64), np.asarray(edge_um, dtype=np.float64),
                           side="left")
    dq = np.zeros(len(sample), dtype=np.int64)
    for s in np.unique(sample):
        ix = np.flatnonzero(sample == s)
        dq[ix] = _rank_bin(np.asarray(density)[ix], n_q)
    comp = np.asarray(comp, dtype=np.int64)
    return [_compact(sample, comp, ebin, dq), _compact(sample, comp, ebin), _compact(sample, comp),
            _compact(sample)]


def _coloc_counts(xys, codes, T: int, radius: float, progress=None) -> np.ndarray:
    """cnt[b, i] = number of cells of type b within *radius* of cell i (the cell itself
    excluded), over the cells of all samples one after the other. *xys* / *codes*: one
    array per sample; every KD-tree holds the cells of ONE sample."""
    from scipy.spatial import cKDTree
    n = int(sum(len(x) for x in xys))
    cnt = np.zeros((T, n), dtype=np.int32)
    off, done, total = 0, 0, max(1, T * len(xys))
    for xy, code in zip(xys, codes):
        xy = np.asarray(xy, dtype=np.float64)
        for b in range(T):
            jobs._job_check()
            tm = code == b
            if tm.any():
                k = cKDTree(xy[tm]).query_ball_point(xy, float(radius), return_length=True,
                                                     workers=-1).astype(np.int32)
                k[tm] -= 1
                cnt[b, off:off + len(xy)] = k
            done += 1
            if progress is not None:
                progress(done / total)
        off += len(xy)
    return cnt


def _bincount2(a, b, na: int, nb: int, weights=None) -> np.ndarray:
    return np.bincount(a * nb + b, weights=weights, minlength=na * nb).reshape(na, nb).astype(np.float64)


def _coloc_enrich(cnt, code, sample, levels, T: int, n_samples: int,
                  min_pool: int = _COLOC_MIN_POOL) -> dict:
    """Observed and expected target counts of every (focal, target) pair per sample.

    For focal type a and target b the pool is every labelled cell that is neither a nor b
    (for a = b: every labelled cell not of that type); the expected count of a focal cell
    is the mean of cnt[b] over the pool cells of its stratum at the finest level of
    *levels* that has at least *min_pool* pool cells; focal cells with no such level are
    left out. Returns O, E, e = log2((O + 0.5) / (E + 0.5)) and n_kept (focal cells used),
    each samples × types × types [s, a, b], n_type (samples × types) and lev0 (share of the
    used focal cells whose expectation came from the finest level)."""
    lab = code >= 0
    cL, sL = code[lab], np.asarray(sample)[lab]
    levL = [np.unique(lv[lab], return_inverse=True)[1] for lv in levels]
    m = [int(lv.max()) + 1 if len(lv) else 0 for lv in levL]
    shape = (n_samples, T, T)
    O, E, NK, L0 = (np.zeros(shape) for _ in range(4))
    n_type = _bincount2(sL, cL, n_samples, T) if len(cL) else np.zeros((n_samples, T))
    if not len(cL):
        return {"O": O, "E": E, "e": np.zeros(shape), "n_kept": NK, "n_type": n_type, "lev0": L0}
    N = [_bincount2(levL[l], cL, m[l], T) for l in range(len(levL))]
    maps = []
    for l in range(len(levL)):
        mp = np.zeros(m[0], dtype=np.int64)
        mp[levL[0]] = levL[l]
        maps.append(mp)
    samp_of = np.zeros(m[0], dtype=np.int64)
    samp_of[levL[0]] = sL
    N0 = N[0]
    for b in range(T):
        v = cnt[b][lab].astype(np.float64)
        S = [_bincount2(levL[l], cL, m[l], T, weights=v) for l in range(len(levL))]
        Ef = np.full((m[0], T), np.nan)
        todo = N0 > 0
        lev0 = np.zeros((m[0], T), dtype=bool)
        for l in range(len(levL)):
            pn = N[l].sum(axis=1, keepdims=True) - N[l] - N[l][:, [b]]
            ps = S[l].sum(axis=1, keepdims=True) - S[l] - S[l][:, [b]]
            pn[:, b] += N[l][:, b]
            ps[:, b] += S[l][:, b]
            pnf, psf = pn[maps[l]], ps[maps[l]]
            ok = todo & (pnf >= min_pool)
            Ef[ok] = psf[ok] / pnf[ok]
            if l == 0:
                lev0 = ok.copy()
            todo &= ~ok
        kept = np.isfinite(Ef) & (N0 > 0)
        np.add.at(O[:, :, b], samp_of, np.where(kept, S[0], 0.0))
        np.add.at(E[:, :, b], samp_of, np.where(kept, N0 * np.nan_to_num(Ef), 0.0))
        np.add.at(NK[:, :, b], samp_of, np.where(kept, N0, 0.0))
        np.add.at(L0[:, :, b], samp_of, np.where(lev0, N0, 0.0))
    return {"O": O, "E": E, "e": np.log2((O + 0.5) / (E + 0.5)), "n_kept": NK, "n_type": n_type,
            "lev0": L0 / np.maximum(NK, 1.0)}


def _coloc_perm(cnt, code, sample, strata, T: int, n_samples: int, n_perm: int = _COLOC_N_PERM,
                seed: int = 0, min_pool: int = _COLOC_MIN_POOL, focal_idx=None, progress=None) -> tuple:
    """Within-stratum label permutation of the focal label (targets fixed).

    For focal a and target b the candidate cells are the labelled cells that are not b;
    in every stratum of *strata* the focal label is given to as many random candidates as
    there are real a cells, the rest are the pool, and the single-level statistic
    log2((ΣO + 0.5) / (ΣE + 0.5)) is recomputed (strata with fewer than *min_pool* pool
    cells are left out, as in the observed value returned beside it). One shuffle of the
    cells per permutation serves every pair. Returns (e0 [samples, focal, target] - the
    observed statistic at this single level -, e_perm [n_perm, samples, focal, target]);
    a = b is NaN. *focal_idx*: the focal types to do (default all)."""
    lab = code >= 0
    cL, sL = code[lab], np.asarray(sample)[lab]
    F = np.arange(T) if focal_idx is None else np.asarray(focal_idx, dtype=np.int64)
    e0 = np.full((n_samples, len(F), T), np.nan)
    ep = np.full((int(n_perm), n_samples, len(F), T), np.nan)
    if not len(cL):
        return e0, ep
    st = np.unique(np.asarray(strata)[lab], return_inverse=True)[1]
    m = int(st.max()) + 1
    samp_of = np.zeros(m, dtype=np.int64)
    samp_of[st] = sL
    N = _bincount2(st, cL, m, T)
    tot_n = N.sum(axis=1)
    K = N[:, F]                                         # focal cells per stratum
    cntL = np.ascontiguousarray(cnt[:, lab])
    same = F[None, :] == np.arange(T)[:, None]          # [b, f]: the focal type is the target

    def _stat(Osel, tot_c, nb_st, b):
        pn = nb_st[:, None] - K
        ok = (pn >= min_pool) & (K > 0)
        Est = np.where(ok, (tot_c[:, None] - Osel) / np.maximum(pn, 1.0), 0.0)
        Os, Es = np.zeros((n_samples, len(F))), np.zeros((n_samples, len(F)))
        np.add.at(Os, samp_of, np.where(ok, Osel, 0.0))
        np.add.at(Es, samp_of, np.where(ok, K * Est, 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):     # (focal = target: undefined, set below)
            e = np.log2((Os + 0.5) / (Es + 0.5))
        e[:, same[b]] = np.nan
        return e

    base = []
    for b in range(T):
        S0 = _bincount2(st, cL, m, T, weights=cntL[b].astype(np.float64))
        nb_st = tot_n - N[:, b]
        tot_c = S0.sum(axis=1) - S0[:, b]
        base.append((tot_c, nb_st))
        e0[:, :, b] = _stat(S0[:, F], tot_c, nb_st, b)
    rng = np.random.default_rng(seed)
    n = len(cL)
    for p in range(int(n_perm)):
        jobs._job_check()
        if progress is not None:
            progress(p / max(1, n_perm))
        order = np.argsort(st + rng.random(n), kind="stable")
        code_o = cL[order]
        for b in range(T):
            tot_c, nb_st = base[b]
            nonb = code_o != b
            pref = np.concatenate([[0.0], np.cumsum(cntL[b][order][nonb], dtype=np.float64)])
            start = (np.cumsum(nb_st) - nb_st).astype(np.int64)
            idx = start[:, None] + np.minimum(K, nb_st[:, None]).astype(np.int64)
            ep[p, :, :, b] = _stat(pref[idx] - pref[start][:, None], tot_c, nb_st, b)
    if progress is not None:
        progress(1.0)
    return e0, ep


def _perm_p(obs, perm) -> tuple:
    """(bias = mean permuted value, two-sided p = (1 + permutations at least as far from
    that mean as the observed value) / (1 + permutations)); NaN where undefined."""
    perm = np.asarray(perm, dtype=np.float64)
    if perm.shape[0] == 0:
        return np.full(np.shape(obs), np.nan), np.full(np.shape(obs), np.nan)
    with np.errstate(invalid="ignore"):
        mu = _nanmean(perm, axis=0)
        hits = (np.abs(perm - mu[None]) >= np.abs(obs - mu)[None] - 1e-12).sum(axis=0)
    p = (1.0 + hits) / (perm.shape[0] + 1.0)
    return mu, np.where(np.isfinite(obs) & np.isfinite(mu), p, np.nan)


def _coloc_prepare(tables, labels, seg_source, radius, max_types=_COLOC_MAX_TYPES,
                   min_cells=_COLOC_MIN_FOCAL, progress=None) -> dict:
    """Cells, type codes, strata and neighbour counts of a view for the co-localisation
    test: {types, dropped, code, sample, levels, cnt, n (cells per sample), comp_names,
    comp, computed_defaults (samples whose compartments were computed here)}."""
    vc = _pooled_types(labels)
    vc = vc[vc >= min_cells]
    dropped = []
    if len(vc) > max_types:
        dropped = list(map(str, vc.index[max_types:]))
        vc = vc.iloc[:max_types]
    types = sorted(map(str, vc.index), key=str)
    tpos = {t: i for i, t in enumerate(types)}
    xys, codes, samp, comp, edge, dens, defaults = [], [], [], [], [], [], []
    for si, (t, lab) in enumerate(zip(tables, labels)):
        xy = stats._xy_of(t["df"])
        n = len(xy)
        code = np.full(n, -1, dtype=np.int64)
        if lab is not None:
            code = pd.Series(lab, dtype=object).map(lambda v: tpos.get(str(v), -1) if v is not None else -1).to_numpy(dtype=np.int64)
        e, d, c = (_ctl_get(t, seg_source, k) for k in ("edge_distance_um", "local_density", "compartment"))
        if e is None or d is None or np.isnan(pd.to_numeric(pd.Series(e), errors="coerce")).any():
            cols, _info = _compartments(xy) if n >= 3 else ({"edge_distance_um": np.zeros(n), "local_density": np.zeros(n),
                                                             "compartment": np.array(["other"] * n, dtype=object)}, {})
            e, d, c = cols["edge_distance_um"], cols["local_density"], cols["compartment"]
            defaults.append(t["label"])
        xys.append(xy); codes.append(code); samp.append(np.full(n, si, dtype=np.int64))
        comp.append(pd.Series(c, dtype=object).fillna("other").astype(str).to_numpy())
        edge.append(np.asarray(e, dtype=np.float64)); dens.append(np.asarray(d, dtype=np.float64))
    code, sample = np.concatenate(codes), np.concatenate(samp)
    comp_names, comp_code = np.unique(np.concatenate(comp), return_inverse=True)
    levels = _coloc_levels(sample, comp_code, np.concatenate(edge), np.concatenate(dens))
    cnt = _coloc_counts(xys, codes, len(types), radius, progress=progress)
    return {"types": types, "dropped": dropped, "code": code, "sample": sample, "levels": levels,
            "cnt": cnt, "n": [len(x) for x in xys], "comp_names": list(comp_names), "comp": comp_code,
            "computed_defaults": defaults}


def _coloc_placebo(prep, focal, n_samples, draws=_COLOC_PLACEBO_DRAWS, seed=1, progress=None) -> pd.DataFrame:
    """Placebo anchors for one focal type (see _COLOC_NOTE): per target, the focal type's
    effect with compartment-level expectations, the same statistic for count-matched
    random anchors drawn within sample × compartment (mean, 95 % range, two-sided p of
    the focal value against them) and for the other types of similar abundance whose
    most common compartment is the focal type's. Effects are means over the samples in
    which the pair is testable (focal ≥ _COLOC_MIN_FOCAL and target ≥ _COLOC_MIN_TARGET cells)."""
    types, T = prep["types"], len(prep["types"])
    f = types.index(focal)
    strata = prep["levels"][2]                              # sample × compartment
    e_all, _ = _coloc_perm(prep["cnt"], prep["code"], prep["sample"], strata, T, n_samples, n_perm=0)
    e_f, e_rand = _coloc_perm(prep["cnt"], prep["code"], prep["sample"], strata, T, n_samples,
                              n_perm=draws, seed=seed, focal_idx=[f], progress=progress)
    code, comp = prep["code"], prep["comp"]
    # A sample where the focal or the target type is absent (or below the cell minimum of
    # the main table) has no effect for the pair: its statistic is log2(0.5 / 0.5) = 0, not
    # a measurement, and must not dilute the mean over the samples.
    lab = code >= 0
    n_st = np.zeros((n_samples, T))
    np.add.at(n_st, (np.asarray(prep["sample"])[lab], code[lab]), 1.0)
    ok = (n_st[:, :, None] >= _COLOC_MIN_FOCAL) & (n_st[:, None, :] >= _COLOC_MIN_TARGET)   # [s, a, b]
    e_f = np.where(ok[:, [f], :], e_f, np.nan)
    e_rand = np.where(ok[None, :, [f], :], e_rand, np.nan)
    e_all = np.where(ok, e_all, np.nan)
    with np.errstate(invalid="ignore"):
        ef = _nanmean(e_f[:, 0, :], axis=0)               # targets
        er = _nanmean(e_rand[:, :, 0, :], axis=1)         # draws × targets
        ea = _nanmean(e_all, axis=0)                      # types × targets
    n_t = np.bincount(code[code >= 0], minlength=T).astype(np.float64)
    major = np.array([np.bincount(comp[code == a], minlength=len(prep["comp_names"])).argmax()
                      if n_t[a] else -1 for a in range(T)])
    similar = [a for a in range(T) if a != f and n_t[a] > 0 and major[a] == major[f]
               and max(n_t[a], n_t[f]) / min(n_t[a], n_t[f]) <= _COLOC_SIMILAR_FOLD]
    rows = []
    for b in range(T):
        if b == f or not np.isfinite(ef[b]):
            continue
        r = er[:, b][np.isfinite(er[:, b])]
        mu = float(r.mean()) if len(r) else np.nan
        p = (1.0 + (np.abs(r - mu) >= abs(ef[b] - mu) - 1e-12).sum()) / (len(r) + 1.0) if len(r) else np.nan
        sim = [(types[a], float(ea[a, b])) for a in similar if a != b and np.isfinite(ea[a, b])]
        sv = np.array([v for _t, v in sim])
        beyond = bool(len(r) and (ef[b] > np.quantile(r, 0.975) or ef[b] < np.quantile(r, 0.025)))
        n_beat = int((np.abs(ef[b]) > np.abs(sv)).sum()) if len(sv) else 0
        if not beyond:
            verdict = "not distinguishable from random anchors"
        elif len(sv) and n_beat < len(sv):
            verdict = f"beyond random anchors, but matched by {len(sv) - n_beat} of {len(sv)} similar types"
        elif len(sv):
            verdict = "specific: beyond random anchors and all similar types"
        else:
            verdict = "beyond random anchors (no similar-abundance type to compare)"
        rows.append({"focal": focal, "target": types[b], "e_focal": float(ef[b]), "random_mean": mu,
                     "random_lo": float(np.quantile(r, 0.025)) if len(r) else np.nan,
                     "random_hi": float(np.quantile(r, 0.975)) if len(r) else np.nan, "p_random": p,
                     "n_similar": len(sv), "similar_median": float(np.median(sv)) if len(sv) else np.nan,
                     "similar_max_abs": float(sv[np.abs(sv).argmax()]) if len(sv) else np.nan,
                     "similar_types": "; ".join(f"{t} {v:+.2f}" for t, v in sim[:12]),
                     "verdict": verdict})
    return pd.DataFrame(rows)


def _coloc_across(e_corr, testable, conds, donors, labels) -> dict:
    """Sample-level inference for every pair. e_corr / testable: [samples, a, b]. Returns
    {mode, note, p_enrich, mean, n_units (each [a, b]), p_diff, effect_diff (or None),
    min_p, min_p_diff}. Enrichment: exact sign-flip of the per-unit effects (a unit is a
    donor when every included sample has one, else a sample); difference between
    conditions: stats._sl_test."""
    S, T, _ = e_corr.shape
    inc = [i for i in range(S) if conds[i] in ("A", "B")] if any(c in ("A", "B") for c in conds) else list(range(S))
    dn = [str(donors[i] or "") for i in range(S)]
    by_donor = bool(inc) and all(dn[i] for i in inc)
    groups: dict = {}
    for i in inc:
        groups.setdefault(dn[i] if by_donor else f"sample{i}", []).append(i)
    n_units = len(groups)
    out = {"inc": list(inc), "groups": [list(ix) for ix in groups.values()],
           "n_units_max": n_units, "p_enrich": np.full((T, T), np.nan), "mean": np.full((T, T), np.nan),
           "n_units": np.zeros((T, T), dtype=np.int64), "p_diff": None, "effect_diff": None,
           "min_p": np.nan, "min_p_diff": np.nan, "diff_note": "",
           "units_note": (f"n = {len(inc)} samples" + (f" ({n_units} donors; samples of a donor averaged)"
                                                       if by_donor else ""))}
    ev = np.where(testable, e_corr, np.nan)
    with np.errstate(invalid="ignore"):
        unit_vals = np.stack([_nanmean(ev[ix], axis=0) for ix in groups.values()]) if groups else np.zeros((0, T, T))
    out["n_units"] = np.isfinite(unit_vals).sum(axis=0)
    with np.errstate(invalid="ignore"):
        out["mean"] = _nanmean(unit_vals, axis=0) if len(unit_vals) else out["mean"]
    out["mode"] = "cohort" if n_units >= stats._SL_MIN_UNITS else "per-sample"
    if out["mode"] == "cohort":
        for a in range(T):
            for b in range(T):
                d = unit_vals[:, a, b]
                d = d[np.isfinite(d)]
                if len(d) >= stats._SL_MIN_UNITS:
                    r = stats._sl_signflip(d)
                    out["p_enrich"][a, b] = r["p"]
        out["min_p"] = stats._sl_signflip(np.ones(n_units))["min_p"]
        c_use = [conds[i] if i in inc and conds[i] in ("A", "B") else "" for i in range(S)]
        if "A" in c_use and "B" in c_use:
            u = stats._sl_units(c_use, dn)
            k = (len(u["pairs"]) if u["paired"]
                 else min(sum(1 for _n, c, _ix in u["units"] if c == x) for x in ("A", "B")))
            if k >= stats._SL_MIN_UNITS:
                out["p_diff"], out["effect_diff"] = np.full((T, T), np.nan), np.full((T, T), np.nan)
                # units a pair is tested on (donor pairs when paired, else units of A + B):
                # a pair that is not testable in every sample is tested on fewer units,
                # and its smallest achievable p is larger
                out["n_units_diff"] = np.zeros((T, T), dtype=np.int64)
                first, floors = None, []
                for a in range(T):
                    for b in range(T):
                        if a == b or not np.isfinite(ev[:, a, b]).any():
                            continue
                        r = stats._sl_test(ev[:, a, b], c_use, dn, units=u)
                        # paired: r["n"] counts the donors with the pair testable in both conditions
                        full = (True if u["paired"]
                                else r.get("n_a", 0) >= stats._SL_MIN_UNITS and r.get("n_b", 0) >= stats._SL_MIN_UNITS)
                        if r["n"] >= stats._SL_MIN_UNITS and full:
                            out["p_diff"][a, b], out["effect_diff"][a, b] = r["p"], r["effect"]
                            out["n_units_diff"][a, b] = r["n"]
                            floors.append(r["min_p"])
                            first = first or r
                # the floor of the family is that of its best-covered pair, not of whichever
                # pair happens to come first
                out["min_p_diff"] = float(np.nanmin(floors)) if floors else np.nan
                out["diff_note"] = f"A − B: {(first or {}).get('test', 'sample-level')} test, {u['note']}"
                if u["dropped"]:
                    out["diff_note"] += ("; donors with samples in one condition only were left out: "
                                         + ", ".join(u["dropped"]))
                tested = out["n_units_diff"][out["n_units_diff"] > 0]
                if len(tested) and tested.min() != tested.max():
                    what = "donors" if u["paired"] else "units"
                    out["diff_note"] += (f"; a pair is tested on the {what} in which it is testable in both "
                                         f"conditions: {int(tested.min())} to {int(tested.max())} per pair (column "
                                         f"'Units (A vs B)'), and its own smallest p is larger with fewer")
            else:
                out["diff_note"] = (f"no test between conditions: fewer than {stats._SL_MIN_UNITS} "
                                    f"independent units per condition ({u['note']})")
    return out


def _run_coloc(seg_source, method, label_kind, radius, n_perm, placebo_focal=None, seed=0) -> None:
    """Background job: stratified co-localisation of every ordered pair of cell types of
    the active view."""
    _gen = jobs._dataset_gen

    def _set(status, message, result=None):
        with _coloc_lock:
            if jobs._stale_job(_gen, status, message):
                return
            _coloc_state["status"], _coloc_state["message"] = status, message
            if result is not None:
                _coloc_state["result"] = result
            if status in ("done", "error"):
                _coloc_state["version"] += 1
        if status == "done":
            provenance._prov_record("coloc", seg_source=seg_source, params={"radius": radius, "n_perm": n_perm, "placebo_focal": placebo_focal, "seed": seed}, labels=(label_kind, method), seeds={"seed": seed}, uses_design=True, message=message, result=result)

    try:
        _t0 = _time.time()
        radius = float(max(1.0, radius or _COLOC_RADIUS_UM))
        n_perm = int(max(0, min(5000, _COLOC_N_PERM if n_perm is None else n_perm)))
        _set("running", "Counting neighbours within each sample…")
        analysis._an_require_seg(seg_source)
        tables = stats._view_tables(seg_source)
        labels, cat = stats._comp_labels(label_kind, tables, seg_source, method)
        if all(l is None for l in labels):
            _set("error", f"The active view has no {cat} labels: run Cell Type Annotation first, or "
                          f"use clusters.")
            return
        S = len(tables)
        prep = _coloc_prepare(tables, labels, seg_source, radius,
                              progress=lambda f: jobs._job_progress("Counting neighbours…", 0.25 * f))
        types, T = prep["types"], len(prep["types"])
        if T < 2:
            _set("error", f"Fewer than 2 groups have at least {_COLOC_MIN_FOCAL} cells.")
            return
        jobs._check_dataset_gen(_gen)
        t_cnt = _time.time() - _t0
        obs = _coloc_enrich(prep["cnt"], prep["code"], prep["sample"], prep["levels"], T, S)
        naive = _coloc_enrich(prep["cnt"], prep["code"], prep["sample"], [prep["levels"][-1]], T, S)
        _set("running", f"Permuting labels within strata ({n_perm} permutations)…")
        e0, ep = _coloc_perm(prep["cnt"], prep["code"], prep["sample"], prep["levels"][0], T, S,
                             n_perm=n_perm, seed=seed,
                             progress=lambda f: jobs._job_progress(
                                 f"Permuting labels within strata ({int(f * n_perm)} / {n_perm})…",
                                 0.25 + 0.6 * f))
        bias, p_within = _perm_p(e0, ep)
        jobs._check_dataset_gen(_gen)
        e = obs["e"]
        e_corr = e - np.nan_to_num(bias)
        n_type = obs["n_type"]
        testable = ((n_type[:, :, None] >= _COLOC_MIN_FOCAL) & (n_type[:, None, :] >= _COLOC_MIN_TARGET)
                    & (obs["n_kept"] > 0))
        conds, _b = stats._design_of(tables)
        donors = stats._design_donors(tables)
        ac = _coloc_across(e_corr, testable, conds, donors, [t["label"] for t in tables])
        # per-sample long table
        si, ai, bi = np.nonzero(testable)
        keep = ai != bi
        si, ai, bi = si[keep], ai[keep], bi[keep]
        long = pd.DataFrame({
            "sample": [tables[i]["label"] for i in si], "focal": [types[i] for i in ai],
            "target": [types[i] for i in bi],
            "n_focal": n_type[si, ai].astype(np.int64), "n_target": n_type[si, bi].astype(np.int64),
            "observed": obs["O"][si, ai, bi], "expected": obs["E"][si, ai, bi],
            "effect": e[si, ai, bi], "bias": bias[si, ai, bi], "effect_corrected": e_corr[si, ai, bi],
            "effect_naive": naive["e"][si, ai, bi], "p_within": p_within[si, ai, bi],
            "fallback_frac": 1.0 - obs["lev0"][si, ai, bi]})
        long["fdr_within"] = np.nan
        for lab_s, ix in long.groupby("sample").groups.items():
            long.loc[ix, "fdr_within"] = stats._bh(long.loc[ix, "p_within"].to_numpy())
        # pair-level summary
        # In cohort mode the effect shown beside a p-value is the statistic that p-value
        # tests: the mean over the tested units (samples with a condition; a donor's samples
        # averaged first). Samples left out of the design do not enter it, nor the counts.
        cohort = ac["mode"] == "cohort"
        inc = np.asarray(ac["inc"] if cohort else list(range(S)), dtype=np.int64)
        ev = np.where(testable, e_corr, np.nan)
        nv = np.where(testable, naive["e"], np.nan)
        any_t = testable[inc].any(axis=0)
        ai, bi = np.nonzero(any_t & ~np.eye(T, dtype=bool))
        with np.errstate(invalid="ignore"):
            if cohort:
                eff = ac["mean"][ai, bi]
                nai = _nanmean(np.stack([_nanmean(nv[ix], axis=0) for ix in ac["groups"]]), axis=0)[ai, bi]
            else:
                eff = _nanmean(ev[:, ai, bi], axis=0)
                nai = _nanmean(nv[:, ai, bi], axis=0)
            summ = pd.DataFrame({
                "focal": [types[i] for i in ai], "target": [types[i] for i in bi],
                "n_samples": testable[inc][:, ai, bi].sum(axis=0),
                "effect": eff,
                "fold": 2.0 ** eff,
                "effect_naive": nai,
                "n_positive": (ev[inc][:, ai, bi] > 0).sum(axis=0)})
        if ac["mode"] == "cohort":
            summ["n_units"] = ac["n_units"][ai, bi]
            summ["p_enrich"] = ac["p_enrich"][ai, bi]
            summ["fdr_enrich"] = stats._bh(summ["p_enrich"].to_numpy())
            if ac["p_diff"] is not None:
                summ["effect_diff"] = ac["effect_diff"][ai, bi]
                summ["n_units_diff"] = ac["n_units_diff"][ai, bi]
                summ["p_diff"] = ac["p_diff"][ai, bi]
                summ["fdr_diff"] = stats._bh(summ["p_diff"].to_numpy())
        if S == 1:
            summ["p_within"] = p_within[0, ai, bi]
            summ["fdr_within"] = stats._bh(summ["p_within"].to_numpy())
        placebo = None
        if placebo_focal and str(placebo_focal) in types:
            _set("running", f"Placebo anchors for '{placebo_focal}'…")
            placebo = _coloc_placebo(prep, str(placebo_focal), S,
                                     progress=lambda f: jobs._job_progress("Placebo anchors…", 0.85 + 0.15 * f))
        jobs._check_dataset_gen(_gen)
        n_cells = int(sum(prep["n"]))
        n_lab = int((prep["code"] >= 0).sum())
        res = {"summary": summ, "long": long, "types": types, "mode": ac["mode"], "placebo": placebo,
               "placebo_focal": str(placebo_focal) if placebo is not None else None,
               "radius": radius, "n_perm": n_perm, "seed": seed, "category": cat,
               "seg_source": seg_source or "xenium", "n_samples": S, "n_cells": n_cells,
               "samples": [{"label": t["label"], "name": t["name"], "n_cells": k, "cond": c, "donor": d}
                           for t, k, c, d in zip(tables, prep["n"], conds, donors)],
               "units_note": ac["units_note"], "across": {k: ac[k] for k in ("min_p", "min_p_diff", "diff_note")},
               "seconds": _time.time() - _t0, "seconds_counts": t_cnt}
        m_tests = len(summ)
        msg = (f"{m_tests:,} ordered pairs of {T} groups ({cat}), radius {radius:g} µm, {n_lab:,} labelled "
               f"of {n_cells:,} cells in {S} sample(s); strata: sample × compartment "
               f"({', '.join(prep['comp_names'])}) × edge bin × density quintile; {n_perm} within-stratum "
               f"permutations, seed {seed} ({res['seconds']:.1f} s, of which neighbour counting {t_cnt:.1f} s).")
        if ac["mode"] == "cohort":
            n_sig = int((summ["fdr_enrich"] < 0.05).sum())
            msg += (f"\nAcross samples: exact sign-flip test of the bias-corrected effect; unit of the "
                    f"p-values: {ac['units_note']}; smallest achievable p {ac['min_p']:.3g}; {n_sig:,} "
                    f"pairs with FDR < 0.05.")
            reach = stats._sl_reach(ac["min_p"], int(summ["p_enrich"].notna().sum()))
            if reach:
                msg += f"\n⚠ {reach[0].upper() + reach[1:]}."
            if ac["p_diff"] is not None:
                msg += (f"\nBetween conditions ({ac['diff_note']}): smallest achievable p "
                        f"{ac['min_p_diff']:.3g} (for a pair tested on every unit); "
                        f"{int((summ['fdr_diff'] < 0.05).sum()):,} pairs with FDR < 0.05.")
                reach = stats._sl_reach(ac["min_p_diff"], int(summ["p_diff"].notna().sum()))
                if reach:
                    msg += f"\n⚠ {reach[0].upper() + reach[1:]}."
            elif ac["diff_note"]:
                msg += f"\nBetween conditions: {ac['diff_note']}."
        else:
            floor = 1.0 / (n_perm + 1.0) if n_perm else np.nan
            msg += (f"\n⚠ {'SINGLE-SAMPLE RESULT' if S == 1 else 'PER-SAMPLE RESULTS'}: "
                    + (f"only {ac['n_units_max']} independent unit(s) ({ac['units_note']}); at least "
                       f"{stats._SL_MIN_UNITS} are needed for a test across samples. ")
                    + f"The p-values are within-sample permutation p-values (unit: the cells of one "
                      f"section, treated as exchangeable within strata; n = {n_lab:,} cells). They "
                      f"describe this section only and do not generalise to other sections or donors.")
            if n_perm:
                msg += f" Smallest achievable p {floor:.3g}."
                reach = stats._sl_reach(floor, m_tests if S == 1 else int(long.groupby('sample').size().max()))
                if reach:
                    msg += f"\n⚠ {reach[0].upper() + reach[1:]} (raise the number of permutations)."
        if prep["computed_defaults"]:
            msg += (f"\nCompartments were not computed for {', '.join(prep['computed_defaults'])}: edge "
                    f"distance and density were computed with the defaults and there is no core "
                    f"compartment (Confound Controls tab).")
        if prep["dropped"]:
            msg += (f"\nOnly the {_COLOC_MAX_TYPES} most abundant groups were used; left out: "
                    f"{', '.join(prep['dropped'][:10])}{'…' if len(prep['dropped']) > 10 else ''}.")
        fb = float(long["fallback_frac"].max()) if len(long) else 0.0
        if fb > 0.2:
            msg += (f"\nNote: for some pairs up to {100 * fb:.0f} % of the focal cells took their "
                    f"expectation from a coarser stratum (fewer than {_COLOC_MIN_POOL} pool cells in the "
                    f"finest one): column fallback_frac of the CSV.")
        if placebo is not None:
            spec = int(placebo["verdict"].str.startswith("specific").sum())
            msg += (f"\nPlacebo anchors for '{placebo_focal}': {len(placebo)} targets; {spec} specific "
                    f"(beyond the 95 % range of {_COLOC_PLACEBO_DRAWS} count-matched random anchor sets "
                    f"within sample × compartment and beyond every similar-abundance type).")
        _set("done", msg, result=res)
        print(f"  Stratified co-localisation: {msg}", flush=True)
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


# ─── Annotation: label distribution and lineage collapse ──────────────────────
_ANNOT_COLLAPSE_SHARE = 0.8      # share of a lineage one label must exceed to be flagged
_ANNOT_SUB_MIN = 0.10            # a sub-cluster counts when it holds at least this share of the lineage
_ANNOT_CAUTION = (
    "Caution: on a targeted panel a marker- or classifier-based annotation can give every cell "
    "of a lineage the same label with high confidence, because the genes that separate its "
    "subtypes are not on the panel or are too sparse; confidence then measures the lineage, not "
    "the subtype. Check the label counts below before reading subtype biology from the labels.")


def _label_collapse(labels, lineage=None, sub=None, share: float = _ANNOT_COLLAPSE_SHARE,
                    min_cells: int = 50) -> tuple:
    """(label counts: DataFrame label / n / pct, flags: [{lineage, n, label, share,
    n_sub}]).

    lineage: a coarse grouping of the cells (a coarse cluster set), sub: a finer one. A
    lineage is flagged when one label takes more than *share* of its labelled cells
    although the finer grouping splits it into at least 2 groups of ≥ 10 % of its cells
    each - the pattern of sub-states collapsed into one label. Without *lineage* the
    whole tissue is the one lineage and only the share is checked."""
    lab = pd.Series(labels, dtype=object)
    ok = lab.notna().to_numpy()
    vc = lab[ok].astype(str).value_counts()
    dist = pd.DataFrame({"label": vc.index, "n": vc.to_numpy(),
                         "pct": 100.0 * vc.to_numpy() / max(1, int(vc.sum()))})
    flags = []
    if not ok.any():
        return dist, flags
    lin = (pd.Series(lineage, dtype=object).astype(str).to_numpy() if lineage is not None
           else np.array(["all cells"] * len(lab), dtype=object))
    subv = pd.Series(sub, dtype=object).astype(str).to_numpy() if sub is not None else None
    L = lab.astype(str).to_numpy()
    for g in pd.unique(lin[ok]):
        m = ok & (lin == g)
        n = int(m.sum())
        if n < min_cells:
            continue
        v = pd.Series(L[m]).value_counts()
        top_share = float(v.iloc[0] / n)
        n_sub = None
        if subv is not None and lineage is not None:
            sv = pd.Series(subv[m]).value_counts(normalize=True)
            n_sub = int((sv >= _ANNOT_SUB_MIN).sum())
        if top_share > share and (n_sub is None or n_sub >= 2):
            flags.append({"lineage": str(g), "n": n, "label": str(v.index[0]), "share": top_share,
                          "n_sub": n_sub})
    return dist, flags


def _annot_label_report(seg_source="xenium", method=None) -> list:
    """Children of the annotation dialog's label report: per annotation of the active
    segmentation the label counts and the lineages one label dominates."""
    tables = stats._view_tables(seg_source)
    alt = analysis._an_alt_res(seg_source)
    t = tables[0]
    df = t["df"]
    lineage = sub = None
    lin_name = sub_name = ""
    if alt is None:
        cms = [m for m in (core.DATA.get("cluster_methods") or []) if dataview.cluster_col(m) in df.columns]
        if cms:
            nun = {m: int(df[dataview.cluster_col(m)].nunique()) for m in cms}
            coarse = min((m for m in cms if nun[m] >= 4), key=lambda m: nun[m], default=None)
            fine = max(cms, key=lambda m: nun[m])
            if coarse is not None and nun[fine] > nun[coarse]:
                lineage, sub = df[dataview.cluster_col(coarse)].to_numpy(), df[dataview.cluster_col(fine)].to_numpy()
                lin_name, sub_name = ui.method_label(coarse), ui.method_label(fine)
    out = [html.Div(_ANNOT_CAUTION, style={"fontSize": "10.5px", "color": "#d29922", "lineHeight": "1.35",
                                           "margin": "6px 0"})]
    S = {"fontSize": "10.5px", "color": core.TEXT, "lineHeight": "1.4"}
    any_lab = False
    for m in core._ANNOT_METHODS:
        lab = analysis._an_celltype_labels([t], m, alt)[0]
        if lab is None:
            continue
        any_lab = True
        dist, flags = _label_collapse(lab, lineage, sub)
        n = int(dist["n"].sum())
        top = ", ".join(f"{r.label} {r.n:,} ({r.pct:.1f} %)" for r in dist.head(12).itertuples())
        out.append(html.Div([html.B(f"{core._ANNOT_METHODS[m]}: "),
                             f"{len(dist)} labels over {n:,} labelled cells of {t['label']} — {top}"
                             + ("…" if len(dist) > 12 else "")], style=S))
        if lineage is None:
            out.append(html.Div("No coarse / fine cluster sets to define lineages for this segmentation: "
                                "only the share of the whole tissue is checked.",
                                style={**S, "color": core.MUTED}))
        for f in flags:
            where = (f"lineage '{f['lineage']}' ({lin_name})" if lineage is not None else "the labelled cells")
            out.append(html.Div(
                f"⚠ '{f['label']}' takes {100 * f['share']:.0f} % of {where} ({f['n']:,} cells)"
                + (f", which {sub_name} splits into {f['n_sub']} sub-clusters of ≥ {100 * _ANNOT_SUB_MIN:.0f} % "
                   f"each: sub-states may have been collapsed into one label" if f["n_sub"] else "")
                + f" (flag above {100 * _ANNOT_COLLAPSE_SHARE:.0f} %).", style={**S, "color": "#d29922"}))
        if not flags:
            out.append(html.Div(f"No lineage is dominated by one label above {100 * _ANNOT_COLLAPSE_SHARE:.0f} %"
                                + (f" (lineages: {lin_name}; sub-clusters: {sub_name})." if lineage is not None else "."),
                                style={**S, "color": core.MUTED}))
    if not any_lab:
        out.append(html.Div("No annotation yet for the active segmentation.", style={**S, "color": core.MUTED}))
    return out


# ─── Figures, tables, CSV ─────────────────────────────────────────────────────
def _fig_layout(fig, title, width=None, height=None, **kw):
    kw.setdefault("margin", dict(l=50, r=10, t=40, b=60))
    fig.update_layout(paper_bgcolor=core.CARD_BG, plot_bgcolor=core.CARD_BG,
                      font=dict(color=core.TEXT, size=10),
                      title=dict(text=title, font=dict(size=12), x=0.01), showlegend=False, **kw)
    fig.update_xaxes(gridcolor=core.BORDER, zerolinecolor=core.BORDER)
    fig.update_yaxes(gridcolor=core.BORDER, zerolinecolor=core.BORDER)
    if width:
        fig.update_layout(width=width)
    if height:
        fig.update_layout(height=height)
    return fig


def make_spill_fig(width=None, height=None):
    """Marker share of non-source cells by distance to the nearest source cell (the
    validity check, drawn) and the share of each cell type at or above the first flag."""
    from plotly.subplots import make_subplots
    with _spill_lock:
        res = _spill_state.get("result")
    if not res:
        return figures._make_placeholder_fig("No spillover score computed yet.")
    prof = [p for p in res["profile"] if p["n"]]
    bt = res["by_type"].head(25)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], horizontal_spacing=0.12,
                        subplot_titles=("Marker share of non-source cells by distance to the nearest "
                                        f"{res['source']} cell (µm)",
                                        f"Cells with score ≥ {res['edges'][0]:g}, by type (%)"))
    fig.add_trace(go.Bar(x=[p["band"] for p in prof], y=[p["mean_share"] for p in prof],
                         marker_color="#58a6ff", customdata=[[p["n"], p["median_depth"]] for p in prof],
                         hovertemplate="%{x} µm<br>mean share %{y:.5f}<br>%{customdata[0]:,} cells<br>"
                                       "median counts %{customdata[1]:.0f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=[str(i) for i in bt.index], y=100.0 * bt["mean"].to_numpy(), marker_color="#d29922",
                         customdata=bt["size"].to_numpy(),
                         hovertemplate="%{x}<br>%{y:.1f} % of %{customdata:,} cells<extra></extra>"), row=1, col=2)
    v = res["validity"]
    verdict = {True: "validity check passed", False: "VALIDITY CHECK FAILED", None: "validity not fully testable"}[v["valid"]]
    fig.update_annotations(font=dict(size=10, color=core.TEXT))
    fig.update_xaxes(tickangle=-40, row=1, col=2)
    return _fig_layout(fig, f"Spillover of {res['source']}: {verdict}", width, height)


def _ownership_records(res) -> list:
    if not res:
        return []
    o = res["ownership"]
    o = o[o["owner"] != ""].sort_values("total_count", ascending=False)
    out = pd.DataFrame({"gene": o["gene"], "owner": o["owner"], "owners": o["owners"],
                        "n_owners": o["n_owners"], "owner_cpm": o["owner_cpm"].round(1),
                        "second_ratio": o["second_ratio"].round(3),
                        "total_count": o["total_count"].round(0)})
    return out.head(6000).to_dict("records")


def _spill_csv(res) -> str:
    v = res["validity"]
    lines = [f"# Gene ownership and spillover; labels: {res['category']}; segmentation: {res['seg_source']}; "
             f"counts: {'SPLIT-corrected' if res['corrected'] else 'original'}",
             f"# spillover source type: {res['source']}; markers ({res['marker_info']['rule']}): "
             f"{' '.join(res['markers'])}",
             f"# {res.get('reference_kind', 'median')} marker share inside source cells: {res['reference']:.6g}; validity: "
             f"{'passed' if v['valid'] else 'FAILED' if v['valid'] is False else 'not fully testable'}; "
             f"near/far ratio {v['near_far_ratio']:.4g}; Spearman rho with local source fraction {v['rho_local']:.4g}",
             f"# owner = type with the highest rate (sum of counts / sum of depth); owners = all types at >= "
             f"{res.get('own_frac', _OWN_FRAC):g} of it; genes with < {_OWN_MIN_COUNT} transcripts have no owner; "
             f"second_ratio = runner-up rate / top rate"]
    for p in res["profile"]:
        lines.append(f"# distance {p['band']} um: {p['n']} non-source cells, mean marker share {p['mean_share']:.6g}")
    if res.get("n_uncorrected"):
        lines.append(f"# {analysis._an_uncorrected_note(res['n_uncorrected'])}")
    return provenance._prov_csv_stamp(res, ("\n".join(lines) + "\n" + res["ownership"].to_csv(index=False)))


_SEGQC_COLS = [("id", "ID"), ("source", "Source"), ("counts", "Counts"), ("labels", "Labels"), ("n_cells", "Cells"),
               ("n_labelled", "Labelled"), ("coverage", "Corrected share"),
               ("median_transcripts", "Tx / cell"), ("median_genes", "Genes / cell"),
               ("median_area", "Area µm²"), ("frac_assigned", "Assigned"),
               ("exclusion_failure", "Excl. failure"), ("coexpression_rate", "Co-expr."),
               ("spill_slope", "Spill slope"), ("spill_mean", "Spill /100"),
               ("reverse_slope", "Reverse slope"), ("own_per_100", "Own /100"),
               ("retention", "Retention")]


def _segqc_records(res) -> list:
    if not res:
        return []
    t = res["table"]
    out = pd.DataFrame({c: t[c] for c, _n in _SEGQC_COLS if c in t.columns})
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda v: None if not np.isfinite(v) else float(f"{v:.4g}"))
    return out.astype(object).where(out.notna(), None).to_dict("records")


def make_segqc_fig(width=None, height=None):
    """Small bar charts of the comparison: one bar per segmentation source × counts mode."""
    from plotly.subplots import make_subplots
    with _segqc_lock:
        res = _segqc_state.get("result")
    if not res:
        return figures._make_placeholder_fig("No segmentation QC computed yet.")
    t = res["table"]
    x = [f"{s} {'corr.' if c != 'original' else 'orig.'}" for s, c in zip(t["id"].astype(str), t["counts"])]
    panels = [("exclusion_failure", "Exclusion-rule failure (lower is better)"),
              ("spill_slope", f"Spillover slope, {res['source_type']} (→ 0)"),
              ("retention", "Own-identity retention (corrected; → 1)"),
              ("median_transcripts", "Median transcripts per cell")]
    fig = make_subplots(rows=1, cols=4, subplot_titles=[p[1] for p in panels], horizontal_spacing=0.06)
    colors = ["#58a6ff" if c == "original" else "#d29922" for c in t["counts"]]
    for i, (col, _ttl) in enumerate(panels):
        fig.add_trace(go.Bar(x=x, y=t[col], marker_color=colors, customdata=t["source"].astype(str),
                             hovertemplate="%{x}<br>%{customdata}<br>%{y:.4g}<extra></extra>"), row=1, col=i + 1)
    fig.update_annotations(font=dict(size=10, color=core.TEXT))
    fig.update_xaxes(tickangle=-50, tickfont=dict(size=8))
    return _fig_layout(fig, "Segmentation / correction QC (blue: original counts, orange: SPLIT-corrected)",
                       width, height, margin=dict(l=40, r=10, t=60, b=110))


def _segqc_csv(res) -> str:
    lines = [f"# Segmentation / correction QC; labels: {res['method']}; identity sets: {res['sets_origin']}; "
             f"spillover source type: {res['source_type']}; radius {res['radius']:g} um",
             "# exclusion_failure: share of labelled cells whose own identity share is not above the largest "
             "foreign identity share; coexpression_rate: cells with both markers / cells with either, over "
             "pairs of top markers of different types (count >= 1)",
             "# spill_slope: source identity transcripts per 100 in non-source cells regressed on the number of "
             "source neighbours within the radius; reverse_slope: foreign identity per 100 in source cells vs "
             "non-source neighbours; retention: own identity transcripts after / before correction",
             "# frac_assigned: sum of counts of the cells / gene transcripts (Q >= 20) in the bounding box of the "
             "cell centroids grown by one median cell radius (all-transcripts grid, 5 um bins; original counts only)"]
    for k, g in res["sets"].items():
        lines.append(f"# identity set {k}: {' '.join(g)}")
    for nt in res.get("notes") or []:
        lines.append(f"# note: {nt}")
    body = res["table"].drop(columns=["value"]).to_csv(index=False)
    per = res["per_type"]
    return provenance._prov_csv_stamp(res, ("\n".join(lines) + "\n" + body
            + ("\n# per cell type\n" + per.to_csv(index=False) if len(per) else "")))


def make_coloc_fig(width=None, height=None):
    """Heatmap focal (rows) × target (columns) of the bias-corrected effect, averaged over
    the samples; a star marks FDR < 0.05 of the test named in the title."""
    with _coloc_lock:
        res = _coloc_state.get("result")
    if not res:
        return figures._make_placeholder_fig("No co-localisation computed yet.")
    s, types = res["summary"], res["types"]
    if not len(s):
        return figures._make_placeholder_fig("No pair has enough cells.")
    z = s.pivot(index="focal", columns="target", values="effect").reindex(index=types, columns=types)
    qcol = ("fdr_enrich" if "fdr_enrich" in s.columns else "fdr_within" if "fdr_within" in s.columns else None)
    text = None
    if qcol:
        q = s.pivot(index="focal", columns="target", values=qcol).reindex(index=types, columns=types)
        text = np.where(q.to_numpy() < 0.05, "*", "")
    lim = float(np.nanmax(np.abs(z.to_numpy()))) if np.isfinite(z.to_numpy()).any() else 1.0
    fig = go.Figure(go.Heatmap(z=z.to_numpy(), x=types, y=types, colorscale="RdBu_r", zmid=0, zmin=-lim, zmax=lim,
                               text=text, texttemplate="%{text}" if text is not None else None,
                               colorbar=dict(title="log2 O/E", thickness=12, len=0.7),
                               hovertemplate="focal %{y}<br>target %{x}<br>log2 O/E %{z:.2f}<extra></extra>"))
    how = {"fdr_enrich": "* FDR < 0.05, exact sign-flip across samples",
           "fdr_within": "* FDR < 0.05, within-sample permutation (single sample)", None: "per-sample results: see the table"}[qcol]
    fig.update_yaxes(autorange="reversed", title="focal")
    fig.update_xaxes(tickangle=-45, title="target")
    return _fig_layout(fig, f"Stratified co-localisation, radius {res['radius']:g} µm ({how})", width, height,
                       margin=dict(l=110, r=10, t=40, b=110))


def _coloc_columns(res=None) -> list:
    _text = ("focal", "target", "sample")
    _col = lambda cid, name: {"id": cid, "name": name, "type": "text" if cid in _text else "numeric"}
    cols = [_col("focal", "Focal"), _col("target", "Target"), _col("n_samples", "Samples"),
            _col("effect", "log2 O/E (corr.)"), _col("fold", "Fold"), _col("effect_naive", "Naive log2 O/E")]
    s = None if not res else res["summary"]
    if s is not None:
        for cid, name in (("n_positive", "> 0"), ("n_units", "Units"), ("p_enrich", "p (sign-flip)"),
                          ("fdr_enrich", "FDR"), ("effect_diff", "A − B"), ("n_units_diff", "Units (A vs B)"),
                          ("p_diff", "p (A vs B)"), ("fdr_diff", "FDR (A vs B)"),
                          ("p_within", "p (within sample)"), ("fdr_within", "FDR (within)")):
            if cid in s.columns:
                cols.append(_col(cid, name))
    return cols


def _coloc_records(res) -> list:
    if not res:
        return []
    s = res["summary"].copy()
    key = next((c for c in ("p_enrich", "p_within") if c in s.columns), None)
    s = s.sort_values([key, "effect"] if key else ["effect"], ascending=[True, False] if key else [False],
                      kind="stable")
    for c in s.columns:
        if s[c].dtype.kind == "f":
            s[c] = s[c].map(lambda v: None if not np.isfinite(v) else float(f"{v:.3g}"))
    return s.head(5000).astype(object).where(s.notna(), None).to_dict("records")


_PLACEBO_COLS = [("target", "Target"), ("e_focal", "Focal log2 O/E"), ("random_mean", "Random anchors"),
                 ("random_lo", "2.5 %"), ("random_hi", "97.5 %"), ("p_random", "p vs random"),
                 ("n_similar", "Similar types"), ("similar_median", "Similar median"),
                 ("similar_max_abs", "Similar max |e|"), ("verdict", "Verdict")]


def _placebo_records(res) -> list:
    if not res or res.get("placebo") is None:
        return []
    p = res["placebo"].copy()
    for c in p.columns:
        if p[c].dtype.kind == "f":
            p[c] = p[c].map(lambda v: None if not np.isfinite(v) else float(f"{v:.3g}"))
    return p.astype(object).where(p.notna(), None).to_dict("records")


def _coloc_csv(res) -> str:
    lines = [f"# Stratified co-localisation; groups: {res['category']}; segmentation: {res['seg_source']}; "
             f"radius {res['radius']:g} um; {res['n_perm']} within-stratum permutations, seed {res['seed']}",
             "# effect = log2((sum O + 0.5) / (sum E + 0.5)); O = targets within the radius of focal cells; "
             "E = mean count around pool cells (neither focal nor target) of the same stratum "
             "(sample x compartment x edge bin x density quintile)",
             "# bias = mean effect under permutation of the focal label within strata; effect_corrected = "
             "effect - bias; effect_naive = pool is the whole sample (no strata)",
             f"# p_within: within-sample permutation, unit = cells of one section; mode: {res['mode']}; "
             f"sample-level unit: {res['units_note']}"]
    for smp in res["samples"]:
        lines.append(f"# sample {smp['label']} ({smp['name']}): {smp['n_cells']} cells; condition "
                     f"{smp['cond'] or '-'}; donor {smp['donor'] or '-'}")
    s = res["summary"].rename(columns={c: f"pair_{c}" for c in res["summary"].columns if c not in ("focal", "target")})
    out = res["long"].merge(s, on=["focal", "target"], how="left")
    txt = "\n".join(lines) + "\n" + out.to_csv(index=False)
    if res.get("placebo") is not None:
        txt += (f"\n# placebo anchors for {res['placebo_focal']}: compartment-level expectation; random = "
                f"count-matched random anchors within sample x compartment\n" + res["placebo"].to_csv(index=False))
    return provenance._prov_csv_stamp(res, (txt))


# ─── Layout ───────────────────────────────────────────────────────────────────
def _num(value, width="58px", id=None, **kw):
    cid = id
    return dcc.Input(id=cid, type="number", value=value, debounce=True,
                     style={**ui._PLOT_CTRL_INPUT, "width": width}, **kw)


def _section_title(text):
    return html.Div(text, style={"fontSize": "12px", "color": core.TEXT, "fontWeight": "600",
                                 "padding": "6px 8px 0 8px", "borderTop": f"1px solid {core.BORDER}",
                                 "flexShrink": "0"})


def _graph(msg, filename, min_h="260px", id=None):
    cid = id
    return html.Div(dcc.Graph(id=cid, responsive=True, figure=figures._make_placeholder_fig(msg),
                              config={"displayModeBar": "hover",
                                      "toImageButtonOptions": {"format": "svg", "filename": filename}},
                              style={"width": "100%", "height": "100%"}),
                    id=f"{cid}-container", style={"flex": "1", "minWidth": "0", "minHeight": min_h})


def _controls_tab():
    """The Confound Controls tab: tissue compartments, spillover score, gene ownership."""
    S, L = analysis._AN_ROW_STYLE, analysis._AN_LABEL_STYLE
    _col = lambda cid, name: {"id": cid, "name": name,
                              "type": "text" if cid in ("gene", "owner", "owners") else "numeric"}
    return dbc.Tab(label="Confound Controls", tab_id="tab-controls",
                   label_style={"fontSize": "12px", "padding": "4px 12px"},
                   children=[html.Div([
        html.Div([
            html.Span("Cell types from:", style=L),
            dcc.Dropdown(id="ctl-labels", options=[{"label": "Cluster (active cluster set)", "value": "cluster"}],
                         value=None, placeholder="Cell type annotation / cluster…",
                         style={"flex": "1", "minWidth": "180px", "fontSize": "11px"}),
            html.Span("(used by both sections; counts mode and segmentation as in the sidebar)", style=L),
        ], style=S),
        _section_title("Tissue compartments"),
        html.Div([
            html.Span("Dense core of:", style=L),
            dcc.Dropdown(id="ctl-dominant", options=[{"label": "auto (most abundant type)", "value": "auto"}],
                         value="auto", clearable=False, style={"width": "220px", "fontSize": "11px"}),
            html.Span(style={"flex": "1"}),
            dbc.Button("Compute compartments", id="ctl-comp-run-btn", color="primary", size="sm", n_clicks=0,
                       style={"fontSize": "11px", "padding": "2px 12px"}),
        ], style=S),
        ui._options([html.Div([
            html.Span("Bin (µm):", style=L), _num(id="ctl-bin", value=_EDGE_BIN_UM, min=5, max=1000, step=5),
            html.Span("Closing (bins):", style=L), _num(id="ctl-close", value=_EDGE_CLOSE_BINS, min=0, max=10, step=1),
            html.Span("Fill holes ≤ (bins):", style=L), _num(id="ctl-hole", value=_EDGE_HOLE_BINS, min=0, max=100000, step=1),
            html.Span("Density radius (µm):", style=L), _num(id="ctl-radius", value=_DENSITY_R_UM, min=5, max=2000, step=5),
        ], style=S),
        html.Div([
            html.Span("Core: ≥ neighbours:", style=L), _num(id="ctl-core-nb", value=_CORE_MIN_NB, min=1, max=10000, step=1),
            html.Span("≥ fraction of that type:", style=L), _num(id="ctl-core-frac", value=_CORE_MIN_FRAC, min=0, max=1, step=0.01),
            html.Span("> µm from edge:", style=L), _num(id="ctl-core-edge", value=_CORE_MIN_EDGE_UM, min=0, max=10000, step=10),
        ], style=S)], summary=f"Options ({_EDGE_BIN_UM:g} µm bins, density radius {_DENSITY_R_UM:g} µm, core thresholds)"),
        html.Div(id="ctl-comp-status", style={**analysis._AN_STATUS_STYLE, "whiteSpace": "pre-wrap"}),
        html.Div(_COMPARTMENT_NOTE, id="ctl-comp-note", style=analysis._AN_NOTE_STYLE),
        _section_title("Spillover score and gene ownership"),
        html.Div([
            html.Span("Source type:", style=L),
            dcc.Dropdown(id="spill-source", options=[{"label": "auto (type with the most transcripts)", "value": "auto"}],
                         value="auto", clearable=False, style={"width": "250px", "fontSize": "11px"}),
            html.Span(style={"flex": "1"}),
            dbc.Button("Run", id="spill-run-btn", color="primary", size="sm", n_clicks=0,
                       style={"fontSize": "11px", "padding": "2px 12px"}),
        ], style=S),
        ui._options([html.Div([
            html.Span("Markers to derive:", style=L), _num(id="spill-n-markers", value=_SPILL_N_MARKERS, min=1, max=100, step=1),
            html.Span("Co-owner ≥ fraction of top rate:", style=L), _num(id="spill-own-frac", value=_OWN_FRAC, min=0.01, max=1, step=0.01),
        ], style=S),
        dcc.Textarea(id="spill-markers", value="",
                     placeholder="Optional: marker genes of the source type, separated by commas or new lines "
                                 "(leave empty to derive them from the annotation)…",
                     style={**analysis._AN_INPUT_STYLE, "margin": "2px 8px", "height": "38px", "flexShrink": "0",
                            "fontFamily": "monospace", "resize": "vertical", "width": "calc(100% - 16px)"})],
            summary=f"Options ({_SPILL_N_MARKERS} markers derived from the annotation, co-owner fraction {_OWN_FRAC:g}, own marker list)"),
        html.Div(id="spill-status", style={**analysis._AN_STATUS_STYLE, "whiteSpace": "pre-wrap"}),
        html.Div(_SPILL_NOTE, id="spill-note", style=analysis._AN_NOTE_STYLE),
        html.Div(_LEAK_NOTE, id="leak-note", style=analysis._AN_NOTE_STYLE),
        html.Div([
            _graph(id="spill-plot", msg="No spillover score computed yet.", filename="xenium_spillover"),
            html.Div([
                dash_table.DataTable(id="spill-own-table", data=[],
                                     columns=[_col("gene", "Gene"), _col("owner", "Owner"), _col("owners", "All owners"),
                                              _col("n_owners", "Owners"), _col("owner_cpm", "Owner CPM"),
                                              _col("second_ratio", "2nd / top"), _col("total_count", "Transcripts")],
                                     **stats._dark_table_style(10)),
                html.Div([
                    dbc.Button("Download CSV", id="spill-download-btn", color="secondary", size="sm", n_clicks=0,
                               style={"fontSize": "11px", "padding": "2px 8px"}),
                    dcc.Download(id="spill-download"),
                ], style={"padding": "4px 0"}),
            ], style={"flex": "1", "minWidth": "0", "overflow": "auto", "padding": "0 6px"}),
        ], style={"display": "flex", "minHeight": "300px", "gap": "6px"}),
        ui._plot_dim_controls("spill"),
    ], style={"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0",
              "overflowY": "auto"})])


def _segqc_tab():
    S, L = analysis._AN_ROW_STYLE, analysis._AN_LABEL_STYLE
    _text = ("id", "source", "counts", "labels")
    cols = [{"id": c, "name": n, "type": "text" if c in _text else "numeric"} for c, n in _SEGQC_COLS]
    return dbc.Tab(label="Segmentation QC", tab_id="tab-segqc",
                   label_style={"fontSize": "12px", "padding": "4px 12px"},
                   children=[html.Div([
        html.Div([
            html.Span("Annotation:", style=L),
            dcc.Dropdown(id="segqc-method", options=[], value=None, placeholder="Cell type annotation…",
                         style={"width": "200px", "fontSize": "11px"}),
            html.Span(style={"flex": "1"}),
            dbc.Button("Run", id="segqc-run-btn", color="primary", size="sm", n_clicks=0,
                       style={"fontSize": "11px", "padding": "2px 12px"}),
        ], style=S),
        ui._options([html.Div([
            html.Span("Spillover source type:", style=L),
            dcc.Dropdown(id="segqc-source", options=[{"label": "auto (type with the most transcripts)", "value": "auto"}],
                         value="auto", clearable=False, style={"width": "250px", "fontSize": "11px"}),
            html.Span("Radius (µm):", style=L), _num(id="segqc-radius", value=_QC_RADIUS_UM, min=1, max=500, step=1),
        ], style=S),
        dcc.Textarea(id="segqc-sets", value="",
                     placeholder="Optional identity marker sets, one per line: 'Type: gene, gene, …' (types as in "
                                 "the annotation); a line 'geneA, geneB' adds a pair of mutually exclusive markers. "
                                 "Leave empty to derive the sets from the annotation…",
                     style={**analysis._AN_INPUT_STYLE, "margin": "2px 8px", "height": "48px", "flexShrink": "0",
                            "fontFamily": "monospace", "resize": "vertical", "width": "calc(100% - 16px)"})],
            summary=f"Options (source type: auto, radius {_QC_RADIUS_UM:g} µm, own marker sets)"),
        html.Div(id="segqc-status", style={**analysis._AN_STATUS_STYLE, "whiteSpace": "pre-wrap"}),
        html.Div(_SEGQC_NOTE, id="segqc-note", style=analysis._AN_NOTE_STYLE),
        _graph(id="segqc-plot", msg="No segmentation QC computed yet.", filename="xenium_segmentation_qc", min_h="280px"),
        html.Div([
            dash_table.DataTable(id="segqc-table", data=[], columns=cols, **stats._dark_table_style(12)),
            html.Div([
                dbc.Button("Download CSV", id="segqc-download-btn", color="secondary", size="sm", n_clicks=0,
                           style={"fontSize": "11px", "padding": "2px 8px"}),
                dcc.Download(id="segqc-download"),
            ], style={"padding": "4px 0"}),
        ], style={"padding": "0 8px", "flexShrink": "0"}),
        ui._plot_dim_controls("segqc"),
    ], style={"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0",
              "overflowY": "auto"})])


def _coloc_tab():
    S, L = analysis._AN_ROW_STYLE, analysis._AN_LABEL_STYLE
    pcols = [{"id": c, "name": n, "type": "text" if c in ("target", "verdict") else "numeric"}
             for c, n in _PLACEBO_COLS]
    return dbc.Tab(label="Stratified Co-localisation", tab_id="tab-coloc",
                   label_style={"fontSize": "12px", "padding": "4px 12px"},
                   children=[html.Div([
        html.Div([
            html.Span("Between:", style=L),
            dcc.Dropdown(id="coloc-labels", options=[{"label": "Cluster (active cluster set)", "value": "cluster"}],
                         value=None, placeholder="Cell type annotation / cluster…",
                         style={"flex": "1", "minWidth": "170px", "fontSize": "11px"}),
            dbc.Button("Run", id="coloc-run-btn", color="primary", size="sm", n_clicks=0,
                       style={"fontSize": "11px", "padding": "2px 12px"}),
        ], style=S),
        ui._options(html.Div([
            html.Span("Radius (µm):", style=L), _num(id="coloc-radius", value=_COLOC_RADIUS_UM, min=1, max=2000, step=1),
            html.Span("Permutations:", style=L), _num(id="coloc-nperm", value=_COLOC_N_PERM, min=0, max=5000, step=10),
            html.Span("Placebo anchors for:", style=L),
            dcc.Dropdown(id="coloc-placebo", options=[], value=None, placeholder="(none)",
                         style={"width": "190px", "fontSize": "11px"}),
        ], style=S), summary=f"Options (radius {_COLOC_RADIUS_UM:g} µm, {_COLOC_N_PERM} permutations, placebo anchors: none)"),
        html.Div([
            dbc.Button("Samples & conditions…", id="coloc-design-open-btn", color="secondary", size="sm",
                       n_clicks=0, style={"fontSize": "11px", "padding": "2px 8px"}),
            html.Span(id="coloc-design-summary", style=L),
        ], style=S),
        html.Div(id="coloc-status", style={**analysis._AN_STATUS_STYLE, "whiteSpace": "pre-wrap"}),
        html.Div(_COLOC_NOTE, id="coloc-note", style=analysis._AN_NOTE_STYLE),
        html.Div([
            _graph(id="coloc-plot", msg="No co-localisation computed yet.", filename="xenium_stratified_colocalisation", min_h="340px"),
            html.Div([
                dash_table.DataTable(id="coloc-table", data=[], columns=_coloc_columns(),
                                     **stats._dark_table_style(12)),
                html.Div([
                    dbc.Button("Download CSV", id="coloc-download-btn", color="secondary", size="sm", n_clicks=0,
                               style={"fontSize": "11px", "padding": "2px 8px"}),
                    dcc.Download(id="coloc-download"),
                ], style={"padding": "4px 0"}),
                html.Div("Placebo anchors", style={"fontSize": "11px", "color": core.MUTED, "padding": "4px 0 2px 0"}),
                dash_table.DataTable(id="coloc-placebo-table", data=[], columns=pcols,
                                     **stats._dark_table_style(8)),
            ], style={"flex": "0.9", "minWidth": "0", "overflow": "auto", "padding": "0 6px"}),
        ], style={"display": "flex", "flex": "1", "minHeight": "420px", "gap": "6px"}),
        ui._plot_dim_controls("coloc"),
    ], style={"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0",
              "overflowY": "auto"})])
