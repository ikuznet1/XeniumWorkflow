"""Derived data and accessors on DATA: gene expression, clusters, viewport cells, extra samples."""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import core, jobs, morph, figures


# Gene expression cache: keyed by (gene, data_dir, use_corrected) → np.ndarray
# Bounded LRU: one n_cells vector per gene adds up on a 5K panel.
from collections import OrderedDict as _OrderedDict
_gene_expr_cache: "_OrderedDict" = _OrderedDict()
_GENE_EXPR_CACHE_MAX = 32


def _gene_cache_put(key, arr):
    """Store arr in _gene_expr_cache, evicting the least recently used entries."""
    _gene_expr_cache[key] = arr
    try:
        _gene_expr_cache.move_to_end(key)
        while len(_gene_expr_cache) > _GENE_EXPR_CACHE_MAX:
            _gene_expr_cache.popitem(last=False)
    except KeyError:   # cleared by another callback thread meanwhile
        pass
    return arr


# UMAP figure cache: (key_tuple, figure)
_umap_fig_cache: dict = {}

# UMAP dataframe cache (dropna copies of DATA["df"]) and viewport centroid arrays
_umap_df_cache: dict = {}
_viewport_arrays: dict = {}


def _refresh_derived_data() -> None:
    """Recompute everything derived from DATA. Called at startup and after a dataset swap."""
    global _sorted_gene_names, _default_gene
    # CSC matrix for fast column (gene) access — O(nnz_col) vs O(nnz_total) for CSR
    if core.DATA.get("expr") is not None:
        core.DATA["expr_csc"] = core.DATA["expr"].tocsc()
    else:
        core.DATA.pop("expr_csc", None)
    # Sorted gene names and default gene (avoid re-sorting on every callback)
    _sorted_gene_names = sorted(core.DATA["gene_names"])
    _default_gene = _sorted_gene_names[0] if _sorted_gene_names else None
    _gene_expr_cache.clear()
    _umap_fig_cache.clear()
    _umap_df_cache.clear()
    _viewport_arrays.clear()


def _install_dataset(new_data: dict) -> None:
    """Replace the contents of DATA with new_data (Load Sample) and rebuild derived state."""
    # From here on, background jobs started for the previous dataset stop publishing
    jobs._dataset_gen += 1
    # Drop keys the new dataset does not carry (subset originals, SPLIT ids, sdata, …)
    # so they cannot leak from the previous dataset through DATA.update().
    for _k in [k for k in core.DATA if k not in new_data]:
        core.DATA.pop(_k, None)
    core.DATA.update(new_data)
    _refresh_derived_data()
    morph._reset_morph_caches()
    # WebGL viewer: release the previous bundle's tile / polygon caches
    _va = sys.modules.get("xenium_viewer_api")
    if _va is not None:
        _va.reset(core.DATA.get("data_dir"))
    _vp = getattr(sys.modules.get(__package__ + ".viewer_provider"), "_viewer_provider", None)
    if _vp is not None:
        _vp.forget()

# ─── Extra datasets for multi-sample display ──────────────────────────────────
# Each entry is like DATA but with an added "x_offset" key (µm shift on x-axis).
EXTRA_DATASETS: list = []


def _sample_separator_xs() -> list:
    """x (µm, plot space) of the dashed line drawn left of each extra sample: midway
    between the right edge of the sample before it and its own left edge. (x_offset moves
    a sample's *minimum* x 500 µm past the previous sample, so x_offset - 250 is in the
    gap only when that minimum is 0.) Used by both renderers."""
    xs = []
    try:
        prev_max = float(core.DATA["df"]["x_centroid"].max())
    except Exception:
        prev_max = float("nan")
    for eds in EXTRA_DATASETS:
        off = float(eds.get("x_offset", 0.0))
        ex = eds["df"]["x_centroid"]
        lo = float(ex.min()) + off if len(ex) else float("nan")
        hi = float(ex.max()) + off if len(ex) else float("nan")
        xs.append((prev_max + lo) / 2.0 if np.isfinite(prev_max) and np.isfinite(lo)
                  else off - 250.0)
        if np.isfinite(hi):
            prev_max = hi
    return xs


def _ds_key(ds: dict) -> str:
    """Identity of a dataset dict (DATA or an extra sample): its Xenium output directory,
    or the .zarr path for a sample loaded from a SpatialData store."""
    return str(ds.get("data_dir") or ds.get("sdata_path") or f"mem:{id(ds)}")


def _ds_name(ds: dict) -> str:
    """Short display name of a dataset dict: the last path component of its key."""
    key = _ds_key(ds)
    return os.path.basename(key.rstrip("/")) or key


def _ds_corrected(ds: dict):
    """The SPLIT-corrected count matrix of a dataset dict when it can be used beside its
    original matrix (one row per row of "expr"), else None."""
    m, e = ds.get("split_corrected_expr"), ds.get("expr")
    if m is None or e is None or not hasattr(m, "shape"):
        return None
    if m.shape[0] != e.shape[0] and not (ds is core.DATA and m.shape[0] == len(ds["df"])):
        return None
    return m


def _corrected_status() -> dict:
    """Which samples of the Xenium view carry SPLIT-corrected counts: {labels, have,
    missing (labels without them), all, any}. Corrected counts are shown and tested only
    when every loaded sample has them (never a mix of corrected and original values in
    one colour scale or one test)."""
    dsets = [core.DATA] + list(EXTRA_DATASETS)
    have = [_ds_corrected(ds) is not None for ds in dsets]
    labels = [f"S{i}" for i in range(len(dsets))]
    return {"labels": labels, "have": have,
            "missing": [l for l, h in zip(labels, have) if not h],
            "all": all(have), "any": any(have)}


def _multi_use_corrected(use_corrected: bool) -> bool:
    """Counts the multi-sample Xenium view uses: corrected only when asked for and every
    sample has them."""
    return bool(use_corrected) and _corrected_status()["all"]


def _extra_gene_values(eds: dict, gene, use_corrected: bool = False):
    """log1p counts of *gene* for the cells of an extra sample (original or, when asked
    for and present, its SPLIT-corrected matrix); None when the sample lacks the gene."""
    g_idx = (eds.get("gene_name_to_idx") or {}).get(gene)
    if g_idx is None or eds.get("expr") is None:
        return None
    key = "split_corrected_expr" if (use_corrected and _ds_corrected(eds) is not None) else "expr"
    m = figures._csc_view(eds, key)
    if g_idx >= m.shape[1]:
        return None
    raw = m[:, g_idx].toarray().ravel()
    idx_ = eds.get("df_to_expr")
    idx_ = np.arange(len(eds["df"])) if idx_ is None else np.asarray(idx_)
    return np.log1p(np.where(idx_ >= 0, raw[np.clip(idx_, 0, len(raw) - 1)], 0.0))


def _extra_cell_type_labels(eds: dict, method=None):
    """(labels, source) of an extra sample's cells for an annotation *method*: the labels
    attached to the sample (eds["_cell_type_labels_by_method"][method], filled from the
    annotation cache when the sample is added), else the obs column cell_type_<method>
    of a store written by Save as SpatialData, else the method-independent
    eds["_cell_type_labels"] / obs column cell_type. (None, None) when it has none."""
    by = eds.get("_cell_type_labels_by_method") or {}
    df = eds["df"]

    def _col(name):
        s = df[name].astype(object)
        return s.where(s.notna() & ~s.isin(["", "nan", "<NA>"]), np.nan)

    if method and by.get(method) is not None:
        return by[method], f"{method} (annotation cache)"
    if method and f"cell_type_{method}" in df.columns:
        return _col(f"cell_type_{method}"), f"cell_type_{method} (stored column)"
    if eds.get("_cell_type_labels") is not None:
        return eds["_cell_type_labels"], eds.get("_cell_type_labels_source") or "sample labels"
    if "cell_type" in df.columns:
        return _col("cell_type"), "cell_type (stored column)"
    return None, None


# ─── Data helpers ─────────────────────────────────────────────────────────────
def get_gene_expression(gene: str) -> np.ndarray:
    """log1p expression for *gene*, aligned to DATA['df'] row order.
    Genes with ' [imp]' suffix are served from DATA['expr'] if written back to zarr,
    with fallback to _spage_state['result'] for pre-zarr cached results.
    Results are cached to avoid recomputation on pan/zoom."""
    _cache_key = (gene, core.DATA.get("data_dir"))
    _hit = _gene_expr_cache.get(_cache_key)
    if _hit is not None:
        try:
            _gene_expr_cache.move_to_end(_cache_key)
        except KeyError:
            pass
        return _hit

    if gene.endswith(" [imp]"):
        base = gene[:-6]
        # Check if imputed gene was written back into DATA["expr"]
        if base in core.DATA.get("gene_name_to_idx", {}):
            gene_idx = core.DATA["gene_name_to_idx"][base]
            expr_csc = core.DATA.get("expr_csc")# or DATA["expr"]
            raw  = expr_csc[:, gene_idx].toarray().ravel()
            idx  = core.DATA["df_to_expr"]
            vals = np.where(idx >= 0, raw[np.clip(idx, 0, len(raw) - 1)], 0.0)
            result = np.log1p(vals)
            _gene_cache_put(_cache_key, result)
            return result
        # Fallback: old-style _spage_state result DataFrame, or streaming zarr/parquet
        with core._spage_lock:
            res       = core._spage_state.get("result")
            res_path  = core._spage_state.get("result_path")
            res_genes = core._spage_state.get("result_genes") or []
        if res is not None and base in res.columns:
            # Result rows follow DATA["df"] as it was when SpaGE ran; realign by
            # cell id when a subset has been applied or cleared since.
            if len(res) != len(core.DATA["df"]) and res.index.is_unique:
                _ridx = core.DATA["df"].index
                if res.index.dtype == object and _ridx.dtype != object:
                    _ridx = _ridx.astype(str)
                result = res[base].reindex(_ridx).fillna(0).values.astype(np.float64)
            else:
                result = res[base].values.astype(np.float64)
            # log1p like every other path (write-back column above, _get_expr_values):
            # the same imputed gene must not change scale with the path that serves it
            result = np.log1p(np.clip(result, 0, None))
            _gene_cache_put(_cache_key, result)
            return result
        if res_path and base in res_genes:
            try:
                if os.path.isdir(res_path):   # zarr
                    import zarr as _zarr_ge
                    _z_ge = _zarr_ge.open_array(res_path, mode='r')
                    _gi   = res_genes.index(base)
                    result = np.array(_z_ge[:, _gi], dtype=np.float64)
                else:                          # large parquet — read single column
                    result = pd.read_parquet(res_path, columns=[base])[base].values.astype(np.float64)
                result = figures._align_expr_rows_to_df(result)
                if len(result) != len(core.DATA["df"]):
                    # streamed under another subset(): rows cannot be matched to cells
                    raise ValueError(f"result has {len(result):,} rows for "
                                     f"{len(core.DATA['df']):,} cells — re-run SpaGE")
                result = np.log1p(np.clip(result, 0, None))   # same scale as the other paths
                _gene_cache_put(_cache_key, result)
                return result
            except Exception as _sge:
                print(f"  SpaGE: streaming read failed for {base}: {_sge}", flush=True)
        return np.zeros(len(core.DATA["df"]), dtype=np.float64)
    gene_idx = core.DATA["gene_name_to_idx"][gene]
    expr_csc = core.DATA.get("expr_csc")# or DATA["expr"]
    raw  = expr_csc[:, gene_idx].toarray().ravel()
    idx  = core.DATA["df_to_expr"]
    vals = np.where(idx >= 0, raw[np.clip(idx, 0, len(raw) - 1)], 0.0)
    result = np.log1p(vals)
    _gene_cache_put(_cache_key, result)
    return result


def _xenium_gene_values(gene, use_corrected: bool = False):
    """Values the Xenium source colours *gene* with (SPLIT-corrected when asked for and
    available, else get_gene_expression), or None when the dataset has no such gene - a
    selection left from another dataset, or no selection. Both renderers then draw
    transcript counts, as they do for a Baysor / Proseg run without the gene."""
    if not gene:
        return None
    vals = figures._get_expr_values(gene, use_corrected=True) if use_corrected else None
    if vals is None:
        try:
            vals = get_gene_expression(gene)
        except KeyError:
            return None
    return vals


def _panel_gene_names() -> list:
    """Measured panel genes: DATA['gene_names'] without SpaGE-imputed entries."""
    gv = core.DATA.get("gene_var")
    if gv is not None and "is_imputed" in gv.columns:
        return [g for g in core.DATA["gene_names"] if not bool(gv.loc[g, "is_imputed"])]
    return list(core.DATA["gene_names"])


def cluster_col(method: str) -> str:
    return f"clust__{method}"


def cluster_color_map(method: str) -> dict:
    col      = cluster_col(method)
    clusters = sorted(core.DATA["df"][col].unique())
    return {c: core.CLUSTER_COLORS[i % len(core.CLUSTER_COLORS)] for i, c in enumerate(clusters)}


def viewport_cell_ids(relayout: dict) -> list | None:
    """
    Return list of cell_ids whose centroid falls within the current spatial
    viewport, or None if no viewport info is available.
    Coordinates in relayout are in µm (plot space).
    """
    x0 = relayout.get("xaxis.range[0]")
    x1 = relayout.get("xaxis.range[1]")
    y0 = relayout.get("yaxis.range[0]")
    y1 = relayout.get("yaxis.range[1]")

    if None in (x0, x1, y0, y1):
        return None

    # Cache extracted numpy arrays to avoid pandas overhead on every call
    if "xc" not in _viewport_arrays:
        df = core.DATA["df"]
        _viewport_arrays["xc"] = df["x_centroid"].values
        _viewport_arrays["yc"] = df["y_centroid"].values
        _viewport_arrays["idx"] = df.index.values

    xc = _viewport_arrays["xc"]
    yc = _viewport_arrays["yc"]
    idx = _viewport_arrays["idx"]

    # Plot coords are already in µm (x = x_centroid, y = -y_centroid).
    xc_min, xc_max = float(x0), float(x1)
    yc_min, yc_max = float(-y1), float(-y0)   # un-negate

    mask = (xc >= xc_min) & (xc <= xc_max) & (yc >= yc_min) & (yc <= yc_max)
    return idx[mask].tolist()


def viewport_reseg_cell_ids(relayout: dict, cells_df) -> list | None:
    """Like viewport_cell_ids but uses a reseg cells_df (Baysor/Proseg) instead of DATA['df']."""
    x0 = relayout.get("xaxis.range[0]")
    x1 = relayout.get("xaxis.range[1]")
    y0 = relayout.get("yaxis.range[0]")
    y1 = relayout.get("yaxis.range[1]")
    if None in (x0, x1, y0, y1):
        return None
    xc = cells_df["x_centroid"].values
    yc = cells_df["y_centroid"].values
    idx = cells_df.index.values
    xc_min, xc_max = float(x0), float(x1)
    yc_min, yc_max = float(-y1), float(-y0)
    mask = (xc >= xc_min) & (xc <= xc_max) & (yc >= yc_min) & (yc <= yc_max)
    return [str(i) for i in idx[mask]]


def build_boundary_trace(cell_ids: list, bounds_dict: dict, color: str, name: str) -> go.Scatter | None:
    """Build a single NaN-separated Scatter trace for a set of polygon boundaries."""
    # Collect polygon arrays, compute total size for pre-allocation
    polys = []
    total = 0
    for cid in cell_ids:
        entry = bounds_dict.get(cid)
        if entry is not None:
            polys.append(entry)
            total += len(entry[0]) + 1  # +1 for NaN separator
    if not polys:
        return None
    all_x = np.empty(total, dtype=np.float64)
    all_y = np.empty(total, dtype=np.float64)
    pos = 0
    for vx, vy in polys:
        vx = np.asarray(vx, dtype=np.float64)
        vy = np.asarray(vy, dtype=np.float64)
        n = len(vx)
        all_x[pos:pos + n] = vx
        all_y[pos:pos + n] = -vy
        pos += n
        all_x[pos] = np.nan
        all_y[pos] = np.nan
        pos += 1
    return go.Scatter(
        x=all_x, y=all_y,
        mode="lines",
        line=dict(color=color, width=0.8),
        name=name,
        hoverinfo="skip",
        showlegend=True,
    )
