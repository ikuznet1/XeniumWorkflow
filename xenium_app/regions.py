"""ROI annotation helpers and RCTD pie shapes."""

import os
import json
import hashlib
import numpy as np
import pandas as pd
from dash import html
import dash_bootstrap_components as dbc

from . import core, caching, dataview, reseg


# ─── ROI helper functions ──────────────────────────────────────────────────────

_ROI_PALETTE = [
    "#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
    "#1abc9c","#e67e22","#34495e","#e91e63","#00bcd4","#cddc39","#ff5722",
]

def _roi_color(index: int) -> str:
    return _ROI_PALETTE[index % len(_ROI_PALETTE)]


def _roi_cache_path() -> str:
    cache_dir = core.CACHE_BASE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    tag = hashlib.md5(caching._dataset_key().encode()).hexdigest()[:8]
    return os.path.join(cache_dir, f"rois_{tag}.json")


def _roi_save_cache() -> None:
    """Serialise _roi_state["rois"] to JSON. Call inside _roi_lock."""
    try:
        with open(_roi_cache_path(), "w") as f:
            json.dump(core._roi_state["rois"], f)
    except Exception as e:
        print(f"  ROI cache write error: {e}", flush=True)


def _roi_load_cache() -> list:
    try:
        path = _roi_cache_path()
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


def _xy_to_svg_path(xy: list) -> str:
    if not xy:
        return ""
    parts = [f"M {xy[0][0]},{xy[0][1]}"]
    for x, y in xy[1:]:
        parts.append(f"L {x},{y}")
    parts.append("Z")
    return " ".join(parts)


def _svg_sector(cx, cy, r, theta1, theta2):
    """SVG path string for a filled wedge in data coordinates.

    The arc is written as line segments (at least two, one per 15° at most - the polygon
    the WebGL view draws): plotly.js converts only the M / L / Z style commands of a
    data-referenced shape path, an ``A`` command is dropped and the wedge collapses to a
    zero-area line."""
    import math
    span = theta2 - theta1
    n_seg = max(2, int(math.ceil(abs(span) / (math.pi / 12))))
    pts = [(cx, cy)] + [(cx + r * math.cos(theta1 + span * k / n_seg),
                         cy + r * math.sin(theta1 + span * k / n_seg)) for k in range(n_seg + 1)]
    return "M " + " L ".join(f"{px:.3f},{py:.3f}" for px, py in pts) + " Z"


def _build_rctd_pie_shapes(df, weights_df, ct_colors, radius_um=core.RCTD_PIE_RADIUS_UM):
    """Build Plotly layout shapes (SVG wedges) for RCTD weight pie charts.

    Parameters
    ----------
    df : DataFrame  — cells with x_centroid, y_centroid (y already negated on caller side)
    weights_df : DataFrame  — (n_cells, n_types), index = str cell IDs, cols = cell type names
    ct_colors : dict  — {cell_type: hex_color}
    radius_um : float  — wedge radius in µm (data coordinates)
    """
    import math
    shapes = []
    # Align weights to df; use str index
    idx_str = df.index.astype(str)
    w = weights_df.reindex(idx_str).fillna(0).values.astype(float)
    xs = df["x_centroid"].values
    ys = -df["y_centroid"].values      # negated, same as spatial plot y-axis
    col_names = list(weights_df.columns)

    for i in range(len(df)):
        row = w[i]
        total = row.sum()
        if total <= 0:
            continue
        row = row / total  # normalize to sum=1
        theta = -math.pi / 2   # start at top (12 o'clock)
        cx, cy = float(xs[i]), float(ys[i])
        for j, ct in enumerate(col_names):
            frac = row[j]
            if frac < 0.02:
                theta += frac * 2 * math.pi
                continue
            dtheta = frac * 2 * math.pi
            color = ct_colors.get(ct, "#888888")
            path = _svg_sector(cx, cy, radius_um, theta, theta + dtheta)
            shapes.append({
                "type": "path", "path": path,
                "fillcolor": color, "opacity": 0.85,
                "line": {"width": 0},
                "xref": "x", "yref": "y", "layer": "above",
            })
            theta += dtheta
    return shapes


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _roi_compute_hull(cell_ids, src_df=None):
    """Return convex hull exterior coords [[x,y],...] for the given cell_ids, or None."""
    try:
        from shapely.geometry import MultiPoint
        df = src_df if src_df is not None else core.DATA["df"]
        # match as strings: plot customdata is str, a reseg cells_df index may be int
        sub = df[df.index.astype(str).isin({str(c) for c in cell_ids})]
        if len(sub) < 3:
            return None
        pts = list(zip(sub["x_centroid"].values, -sub["y_centroid"].values))
        hull = MultiPoint(pts).convex_hull
        if hull.geom_type == "Polygon":
            return [list(c) for c in hull.exterior.coords]
        elif hull.geom_type == "LineString":
            return [list(c) for c in hull.coords]
        return None
    except Exception as e:
        print(f"  ROI hull error: {e}", flush=True)
        return None


def _roi_mask_in_polygon(polygon_xy: list, df: "pd.DataFrame") -> np.ndarray:
    """Boolean mask over df rows whose centroids lie strictly inside polygon_xy
    (vectorised; same result as shapely poly.contains(Point) per cell)."""
    mask = np.zeros(len(df), dtype=bool)
    try:
        import shapely
        from shapely.geometry import Polygon as _SPoly
        poly = _SPoly(polygon_xy)
        xs = df["x_centroid"].to_numpy(dtype=float)
        ys = -df["y_centroid"].to_numpy(dtype=float)
        minx, miny, maxx, maxy = poly.bounds
        cand = (xs >= minx) & (xs <= maxx) & (ys >= miny) & (ys <= maxy)
        if cand.any():
            mask[cand] = shapely.contains_xy(poly, xs[cand], ys[cand])
    except Exception as e:
        print(f"  ROI containment error: {e}", flush=True)
        mask[:] = False
    return mask


def _roi_cells_in_polygon(polygon_xy: list, src_df=None) -> list:
    """Return cell IDs (index values) whose centroids lie inside polygon_xy."""
    df = src_df if src_df is not None else core.DATA["df"]
    return df.index[_roi_mask_in_polygon(polygon_xy, df)].tolist()


def _roi_apply_metadata_to_df(df: "pd.DataFrame", rois: list) -> None:
    """Add/update roi_{cls} columns in df based on rois list."""
    # Reset ALL existing roi_* columns first (handles deletions of last ROI in a class)
    for col in [c for c in df.columns if c.startswith("roi_")]:
        df[col] = pd.NA
    # Ensure columns exist for all current classes
    for cls in {r["cls"] for r in rois}:
        col = f"roi_{cls}"
        if col not in df.columns:
            df[col] = pd.NA

    overlap_warned: set = set()
    for roi in rois:
        col = f"roi_{roi['cls']}"
        _in_roi = _roi_mask_in_polygon(roi["polygon_xy"], df)
        # Warn on overlaps with existing same-class assignments
        existing = df.loc[_in_roi, col]
        overlapping = existing.dropna()
        for existing_name in overlapping.unique():
            key = (existing_name, roi["name"])
            if key not in overlap_warned:
                print(f"  ROI WARNING: '{roi['name']}' overlaps with '{existing_name}' (class '{roi['cls']}')", flush=True)
                overlap_warned.add(key)
        df.loc[_in_roi, col] = roi["name"]


def _roi_apply_everywhere(rois: list) -> None:
    """Apply ROI metadata to every loaded cell table, including the full
    (pre-subset) frames, so roi_* columns survive unsubset()."""
    _roi_apply_metadata_to_df(core.DATA["df"], rois)
    _orig = core.DATA.get("_df_original")
    if _orig is not None and _orig is not core.DATA["df"]:
        _roi_apply_metadata_to_df(_orig, rois)
    for _lock, _state in ((core._baysor_lock, core._baysor_state), (core._proseg_lock, core._proseg_state)):
        with _lock:
            _res = _state.get("result")
        if _res:
            _roi_apply_metadata_to_df(_res["cells_df"], rois)
            _rorig = _res.get("_cells_df_original")
            if _rorig is not None and _rorig is not _res["cells_df"]:
                _roi_apply_metadata_to_df(_rorig, rois)
    # UMAP df cache holds dropna() copies of DATA["df"] without the new columns
    dataview._umap_df_cache.clear()
    dataview._umap_fig_cache.clear()


def _roi_manager_contents(rois: list):
    """Return (table, op_options) for the ROI manager modal."""
    if not rois:
        table = html.P("No ROIs defined yet. Use lasso on the spatial plot to draw one.",
                       style={"color": core.MUTED, "fontSize": "0.82rem"})
    else:
        rows = []
        for i, r in enumerate(rois):
            rows.append(html.Tr([
                html.Td(html.Div(style={"width": "18px", "height": "18px",
                                         "backgroundColor": r["color"], "borderRadius": "3px"})),
                html.Td(r["cls"], style={"padding": "4px 8px"}),
                html.Td(r["name"], style={"padding": "4px 8px"}),
                html.Td(dbc.Button("✕", id={"type": "roi-delete-btn", "index": i},
                                   size="sm", color="danger", style={"padding": "1px 7px"})),
            ]))
        table = dbc.Table([html.Tbody(rows)], bordered=False, hover=True,
                          style={"fontSize": "0.82rem", "color": core.TEXT})
    op_opts = [{"label": f"{r['cls']}: {r['name']}", "value": i} for i, r in enumerate(rois)]
    return table, op_opts


def _roi_write_all_zarrs_bg() -> None:
    """Daemon thread: write ROI metadata columns to all zarr stores."""
    with core._roi_lock:
        rois = list(core._roi_state["rois"])
    # Write to Xenium zarr
    try:
        sdata_path = core.DATA.get("sdata_path", "")
        if sdata_path and os.path.isdir(sdata_path):
            # Write the full frame, not the active subset
            _full_df = core.DATA.get("_df_original")
            if _full_df is None:
                _full_df = core.DATA["df"]
            # roi_* columns were already applied by the caller (_roi_apply_everywhere)
            reseg._update_reseg_zarr_obs(sdata_path, _full_df)
            print(f"  ROI: wrote {len(rois)} ROIs to xenium zarr", flush=True)
    except Exception as e:
        print(f"  ROI zarr write error (xenium): {e}", flush=True)
    # Write to Baysor/Proseg zarrs
    # (only the loaded result of each tool has its cell table in memory)
    for tool, _lock, _state in (("baysor", core._baysor_lock, core._baysor_state),
                                ("proseg", core._proseg_lock, core._proseg_state)):
        try:
            with _lock:
                result = _state.get("result")
            if result and result.get("sdata_path") and os.path.isdir(result["sdata_path"]):
                _full_cdf = result.get("_cells_df_original")
                if _full_cdf is None:
                    _full_cdf = result["cells_df"]
                reseg._update_reseg_zarr_obs(result["sdata_path"], _full_cdf)
                print(f"  ROI: wrote {len(rois)} ROIs to {tool} zarr", flush=True)
        except Exception as e:
            print(f"  ROI zarr write error ({tool}): {e}", flush=True)


def _roi_shapes_for_fig(rois: list, pending_hull=None) -> list:
    """Return list of Plotly shape dicts for ROI overlays."""
    shapes = []
    for r in rois:
        shapes.append({
            "type": "path",
            "path": _xy_to_svg_path(r["polygon_xy"]),
            "fillcolor": _hex_to_rgba(r["color"], 0.18),
            "line": {"color": r["color"], "width": 1.5},
            "layer": "above", "xref": "x", "yref": "y",
        })
    if pending_hull:
        shapes.append({
            "type": "path",
            "path": _xy_to_svg_path(pending_hull),
            "fillcolor": "rgba(200,200,200,0.2)",
            "line": {"color": "#aaaaaa", "width": 1.5, "dash": "dot"},
            "layer": "above", "xref": "x", "yref": "y",
        })
    return shapes


def _auto_load_rois() -> None:
    """Load ROIs from cache at startup and apply metadata to DATA['df']."""
    rois = _roi_load_cache()
    # Migration: if no ROIs found at the absolute-path hash (new scheme),
    # also try the relative-path hash that was used before os.path.abspath normalisation.
    if not rois and core.DATA.get("data_dir"):
        _rel_tag = hashlib.md5(os.path.basename(core.DATA["data_dir"]).encode()).hexdigest()[:8]
        _cache_dir_path = core.CACHE_BASE_DIR
        for _try_tag in (_rel_tag,):
            _old_path = os.path.join(_cache_dir_path, f"rois_{_try_tag}.json")
            if os.path.exists(_old_path):
                try:
                    with open(_old_path) as _f:
                        rois = json.load(_f)
                    if rois:
                        core._roi_state["rois"] = rois
                        _roi_save_cache()  # re-save under new absolute-path hash
                        print(f"  ROI: migrated {len(rois)} ROIs from rois_{_try_tag}.json", flush=True)
                except Exception:
                    pass
                break
    with core._roi_lock:
        core._roi_state["rois"] = rois
    if rois and "df" in core.DATA:
        _roi_apply_everywhere(rois)
        print(f"  ROI: auto-loaded {len(rois)} ROIs from cache", flush=True)
