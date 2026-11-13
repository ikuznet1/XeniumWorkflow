"""Plotly figure builders: spatial map, UMAP and the analysis plots."""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import plotly.graph_objects as go

from . import core, jobs, dataview, annotation, reseg, spatial, analysis, sdata_store, regions
from . import clustersets


# ─── Figure builders ──────────────────────────────────────────────────────────
def _base_layout(title, xlabel, ylabel, equal_aspect=False):
    layout = dict(
        template="plotly_dark",
        paper_bgcolor=core.PLOT_BG, plot_bgcolor=core.PLOT_BG,
        margin=dict(l=50, r=65, t=38, b=48),
        xaxis=dict(title=xlabel, showgrid=False, zeroline=False, color=core.TEXT),
        yaxis=dict(title=ylabel, showgrid=False, zeroline=False, color=core.TEXT),
        font=dict(color=core.TEXT, size=11),
        title=dict(text=title, font=dict(size=13, color=core.TEXT), x=0.5, xanchor="center"),
        hoverlabel=dict(bgcolor=core.CARD_BG, font_color=core.TEXT),
    )
    if equal_aspect:
        layout["yaxis"]["scaleanchor"]  = "x"
        layout["yaxis"]["scaleratio"]   = 1
        layout["yaxis"]["constrain"]    = "domain"   # shrink plot domain, not data range
        layout["xaxis"]["constrain"]    = "domain"   # keep both axes consistent
    return layout


def _categorical_traces(x, y, df, method, size, opacity, mode="spatial"):
    cmap = dataview.cluster_color_map(method)
    col  = dataview.cluster_col(method)
    traces = []
    for c in sorted(df[col].unique()):
        mask = (df[col] == c).values
        mk   = size if mode == "spatial" else max(2, size * 1.3)
        traces.append(go.Scattergl(
            x=x[mask], y=y[mask],
            mode="markers",
            marker=dict(size=mk, color=cmap[c], opacity=opacity),
            name=f"Cluster {c}",
            legendgroup=f"c{c}",
            showlegend=(mode == "spatial"),
            text=df.index[mask].tolist(),
            customdata=df.index[mask].tolist(),
            hovertemplate="<b>%{text}</b><extra>Cluster " + str(c) + "</extra>",
        ))
    return traces


def _cell_type_color_map(labels_key: str, df) -> dict:
    """Return {cell_type: color} map consistent with _cell_type_traces ordering."""
    with core._annot_lock:
        labels = core._annot_state.get(labels_key)
    if labels is None:
        return {}
    aligned = labels.reindex(df.index.astype(str)).fillna("Unknown")
    cell_types = sorted(aligned.unique())
    return {ct: core.CLUSTER_COLORS[i % len(core.CLUSTER_COLORS)] for i, ct in enumerate(cell_types)}


def _cell_type_traces(x, y, df, size, opacity, mode="spatial", labels_key="labels_celltypist",
                      invisible=False):
    """Categorical traces coloured by cell type annotation.

    When invisible=True, scatter markers are transparent (for hover/click only when pies
    are rendered as layout shapes). Legend entries are added as zero-length traces.
    """
    with core._annot_lock:
        labels = core._annot_state.get(labels_key)
    if labels is None:
        _ids = df.index.tolist()        # ids as on every other trace: hover and click work
        return [go.Scattergl(x=x, y=y, mode="markers",
                             marker=dict(size=(size if mode == "spatial" else max(2, size * 1.3)),
                                         color=core.MUTED, opacity=opacity),
                             text=_ids, customdata=_ids,
                             hovertemplate="<b>%{text}</b><extra>No annotation</extra>",
                             name="No annotation", showlegend=(mode == "spatial"))]

    # Align labels to df index — cast both to str to handle int vs str mismatch
    aligned = labels.reindex(df.index.astype(str)).fillna("Unknown")
    cell_types = sorted(aligned.unique())
    cmap = {ct: core.CLUSTER_COLORS[i % len(core.CLUSTER_COLORS)] for i, ct in enumerate(cell_types)}

    traces = []
    for ct in cell_types:
        mask = (aligned == ct).values
        mk   = size if mode == "spatial" else max(2, size * 1.3)
        if invisible:
            # Ghost scatter for hover/click; add a visible legend-only trace separately
            traces.append(go.Scattergl(
                x=x[mask], y=y[mask], mode="markers",
                marker=dict(size=mk, color=cmap[ct], opacity=0),
                name=ct, legendgroup=ct, showlegend=False,
                text=df.index[mask].tolist(), customdata=df.index[mask].tolist(),
                hovertemplate="<b>%{text}</b><extra>" + ct + "</extra>",
            ))
            # Zero-length legend entry so the legend still shows cell type → color
            traces.append(go.Scattergl(
                x=[None], y=[None], mode="markers",
                marker=dict(size=mk, color=cmap[ct]),
                name=ct, legendgroup=ct, showlegend=True,
            ))
        else:
            traces.append(go.Scattergl(
                x=x[mask], y=y[mask], mode="markers",
                marker=dict(size=mk, color=cmap[ct], opacity=opacity),
                name=ct, legendgroup=ct,
                showlegend=(mode == "spatial"),
                text=df.index[mask].tolist(), customdata=df.index[mask].tolist(),
                hovertemplate="<b>%{text}</b><extra>" + ct + "</extra>",
            ))
    return traces


def _align_expr_rows_to_df(full):
    """Map a per-expression-row array (one value per row of DATA['expr']) onto
    the current DATA['df'] rows via DATA['df_to_expr']. Arrays of any other
    length are returned unchanged. Identity mapping (no subset) is a no-op."""
    d2e = core.DATA.get("df_to_expr")
    if d2e is None or core.DATA.get("expr") is None or len(full) != core.DATA["expr"].shape[0]:
        return full
    d2e = np.asarray(d2e)
    if len(d2e) == len(full) and (d2e == np.arange(len(full))).all():
        return full
    return np.where(d2e >= 0, full[np.clip(d2e, 0, len(full) - 1)], 0).astype(full.dtype)


def _csc_view(holder: dict, key: str):
    """holder[key] as CSC for O(nnz_col) gene lookups. The CSC copy is built on
    first use and kept in holder[key + '_csc']; it is rebuilt automatically when
    holder[key] is replaced by another matrix object."""
    m = holder.get(key)
    if m is None or not sp.issparse(m) or m.format == "csc":
        return m
    ck = key + "_csc"
    c = holder.get(ck)
    if c is None or holder.get(ck + "_src") is not m:
        c = m.tocsc()
        holder[ck + "_src"] = m
        holder[ck] = c
    return c


def _get_expr_values(gene: str, alt_res=None, use_corrected: bool = False) -> "np.ndarray | None":
    """Unified expression lookup supporting original and SPLIT-corrected counts."""
    lookup = gene.replace(" [imp]", "").replace(" [corr+imp]", "")
    is_imputed_gene = ("[imp]" in gene) or ("[corr+imp]" in gene)
    if alt_res is not None:
        _corr = _csc_view(alt_res, "split_corrected_expr") if use_corrected else None
        expr = _corr if _corr is not None else _csc_view(alt_res, "expr")
        gni  = alt_res.get("gene_name_to_idx") or core.DATA.get("gene_name_to_idx", {})
    else:
        _corr = _csc_view(core.DATA, "split_corrected_expr") if use_corrected else None
        expr = _corr if _corr is not None else core.DATA.get("expr_csc")# or DATA.get("expr")
        gni  = core.DATA.get("gene_name_to_idx", {})
    if use_corrected and is_imputed_gene:
        # Imputed genes are stored in X only: X_corrected has no values for them (a
        # store reloaded from disk carries all-zero padding columns there, which must
        # not be shown as expression). A gene imputed from corrected counts
        # (functions/impute.py --use_corrected) is read from its X column; any other
        # imputed gene has no corrected values - the caller shows the original ones.
        _corr_imp = (alt_res.get("split_corrected_imputed_genes", []) if alt_res is not None
                     else core.DATA.get("split_corrected_imputed_genes", [])) or []
        if lookup not in _corr_imp:
            return None
        expr = _csc_view(alt_res, "expr") if alt_res is not None else core.DATA.get("expr_csc")
    if expr is not None and lookup in gni:
        idx = gni[lookup]
        if idx < expr.shape[1]:
            _col = expr.getcol(idx).toarray().ravel().astype(np.float32)
            if alt_res is None:
                _col = _align_expr_rows_to_df(_col)
            return np.log1p(_col)
    # If corrected mode is active and this is an imputed gene not in the corrected-imputed list,
    # do NOT fall through to the zarr fallback (which holds uncorrected imputed values).
    if use_corrected and is_imputed_gene:
        _corr_imp = (alt_res.get("split_corrected_imputed_genes", []) if alt_res is not None
                     else core.DATA.get("split_corrected_imputed_genes", []))
        if lookup not in _corr_imp:
            return None
    # Streaming SpaGE fallback — gene in zarr but not in expr index
    # For reseg: prefer per-alt_res zarr (correct row count) over global spage_state
    if alt_res is not None:
        _rp = alt_res.get("spage_result_path")
        _rg = alt_res.get("spage_result_genes") or []
    else:
        _rp, _rg = None, []
    if not (_rp and lookup in _rg):
        # Only fall back to global Xenium SpaGE state when NOT in reseg mode;
        # the global zarr has Xenium cell count rows and would misalign with reseg cells.
        if alt_res is None:
            with core._spage_lock:
                _rp = core._spage_state.get("result_path")
                _rg = core._spage_state.get("result_genes") or []
        else:
            return None
    if _rp and lookup in _rg:
        try:
            if os.path.isdir(_rp):
                import zarr as _zarr_ev
                _gi_ev = _rg.index(lookup)
                _zcol = np.array(
                    _zarr_ev.open_array(_rp, mode='r')[:, _gi_ev], dtype=np.float32)
                if alt_res is None:
                    _zcol = _align_expr_rows_to_df(_zcol)
                    if len(_zcol) != len(core.DATA["df"]):
                        # streamed under another subset(): rows cannot be matched to cells
                        raise ValueError(f"result has {len(_zcol):,} rows for "
                                         f"{len(core.DATA['df']):,} cells — re-run SpaGE")
                return np.log1p(_zcol)
        except Exception as _e:
            print(f"  SpaGE zarr read error for {lookup!r}: {_e}", flush=True)
    return None


def _get_reseg_expr_values(gene: str, alt_res: dict) -> "np.ndarray | None":
    """log1p expression for *gene* in resegmented cells, or None if not available.
    Uses alt_res['gene_name_to_idx'] if present (includes reseg-imputed genes),
    falling back to DATA['gene_name_to_idx'] for panel genes."""
    expr = _csc_view(alt_res, "expr")
    if expr is None:
        return None
    # Strip [imp] suffix if present
    lookup = gene[:-6] if gene.endswith(" [imp]") else gene
    gni = alt_res.get("gene_name_to_idx") or core.DATA["gene_name_to_idx"]
    if lookup not in gni:
        return None
    idx = gni[lookup]
    if idx >= expr.shape[1]:
        return None
    col = expr.getcol(idx)
    vals = col.toarray().ravel().astype(np.float32)
    return np.log1p(vals)


# Figure-specific layout shapes (multi-sample separators + RCTD pies) of the last
# spatial figure, so callbacks that patch layout.shapes can keep them.
_spatial_base_shapes: list = []


# Outlines other modules draw on the map (the aggregates module registers):
# functions(seg_source) -> Plotly layout shapes (plot space, y = -µm) for the Plotly map,
# functions(seg_source) -> polygon dicts of viewer_provider.overlays (µm, y down) for WebGL
_overlay_shape_providers: list = []
_overlay_polygon_providers: list = []
# Embeddings other modules draw in the UMAP panel (the joint module registers): functions
# with the arguments of make_umap_fig that return a figure, or None to leave the panel alone
_umap_fig_providers: list = []


_overlay_last_seg: dict = {"seg": "xenium"}   # segmentation of the last map that was built


def _provided_overlays(providers, seg_source) -> list:
    """Outlines of the registered providers; seg_source None: the segmentation of the
    last map built (callers that patch the shapes of the existing figure)."""
    if seg_source is None:
        seg_source = _overlay_last_seg["seg"]
    else:
        _overlay_last_seg["seg"] = seg_source
    out = []
    for fn in list(providers):
        try:
            out += list(fn(seg_source or "xenium"))
        except Exception as exc:
            print(f"  overlay provider error: {exc}", flush=True)
    return out


def _spatial_overlay_shapes(sdata_roi_on: bool = False, sdata_patches_on: bool = False,
                            seg_source=None) -> list:
    """Layout shapes that do not depend on the figure: saved ROI outlines, the
    pending lasso hull, the sopa tissue-ROI / patch-grid overlays and the outlines of
    the registered providers for the active segmentation."""
    shapes = _provided_overlays(_overlay_shape_providers, seg_source)
    with core._roi_lock:
        roi_visible = core._roi_state["show"]
        rois_snap   = list(core._roi_state["rois"])
        pending     = core._roi_state.get("pending_hull")
    if roi_visible and rois_snap:
        shapes += regions._roi_shapes_for_fig(rois_snap)
    if pending:
        shapes += regions._roi_shapes_for_fig([], pending_hull=pending)
    if sdata_roi_on or sdata_patches_on:
        shapes += sdata_store._sdata_overlays_to_shapes(sdata_roi_on, sdata_patches_on)
    return shapes


def _rctd_pie_shapes_for_view(color_by, relayout):
    """RCTD weight pies for the cells inside the current viewport (single-sample
    Xenium view). Returns None when pies do not apply to *color_by*, and an empty
    list when more than PIE_THRESHOLD cells are in view."""
    if dataview.EXTRA_DATASETS or not str(color_by or "").startswith("cell_type:"):
        return None
    _lkey = f"labels_{color_by.split(':')[1]}"
    with core._annot_lock:
        _weights_df = core._annot_state.get(f"rctd_weights_{_lkey}")
    if _weights_df is None:
        return None
    df = core.DATA["df"]
    df_vp = df
    if relayout:
        try:
            _x0 = float(relayout.get("xaxis.range[0]", -1e18))
            _x1 = float(relayout.get("xaxis.range[1]",  1e18))
            _y0 = float(relayout.get("yaxis.range[0]", -1e18))
            _y1 = float(relayout.get("yaxis.range[1]",  1e18))
            if _x1 - _x0 < 1e17:  # real range, not default placeholder
                x =  df["x_centroid"].values
                y = -df["y_centroid"].values
                df_vp = df[(x >= _x0) & (x <= _x1) & (y >= _y0) & (y <= _y1)]
        except Exception:
            pass
    if len(df_vp) > core.PIE_THRESHOLD:
        return []
    # colours from the full frame so pies match the scatter traces
    return regions._build_rctd_pie_shapes(df_vp, _weights_df, _cell_type_color_map(_lkey, df))


def make_spatial_fig(color_by, method, gene, size, opacity,
                     boundary_toggles, relayout,
                     morph_image=None, extra_title="", baysor_active=False, proseg_active=False,
                     use_corrected: bool = False, cbar_range=None,
                     sdata_roi_on: bool = False, sdata_patches_on: bool = False,
                     density=None, alt_tool=None):
    # Choose data source: Proseg > Baysor > original Xenium (priority order)
    with core._proseg_lock:
        pres = core._proseg_state["result"] if proseg_active and core._proseg_state["result"] else None
    with core._baysor_lock:
        bres = core._baysor_state["result"] if baysor_active and not pres and core._baysor_state["result"] else None
    # a further tool (alt_tool, e.g. "imgseg"): its loaded run, drawn like a Baysor / Proseg one
    _ores = (core._seg_result(alt_tool) if alt_tool not in (None, "xenium", "baysor", "proseg")
             and not (pres or bres) else None)
    alt_res = pres or bres or _ores
    _alt_tool = "proseg" if pres else "baysor" if bres else alt_tool if _ores else "xenium"
    # Gene-set score ("geneset:<name>"): one value per cell of the view, or None when
    # the set cannot be scored - transcript counts are drawn then, as for a missing gene
    _gs = None
    if str(color_by or "").startswith("geneset:"):
        _gs = analysis._geneset_color(color_by, _alt_tool, use_corrected)
        if _gs is None:
            color_by = "transcript_counts"

    def _gs_trace(gx, gy, ids):
        return go.Scattergl(
            x=gx, y=gy, mode="markers",
            marker=analysis._geneset_marker(_gs[0], _gs[1], _gs[2], cbar_range, size=size, opacity=opacity),
            text=ids, customdata=ids,
            hovertemplate="<b>%{text}</b><br>" + _gs[2] + ": %{marker.color:.2f}<extra></extra>",
            name=_gs[2])
    _separator_shapes = []   # populated later for multi-sample mode
    _pie_shapes = []         # RCTD pie chart wedge shapes (single-dataset, Xenium only)

    # Helper: apply log1p for transcript_counts; build colorbar kwargs with optional cmin/cmax
    def _color_vals(raw_vals, color_key):
        v = np.log1p(raw_vals) if color_key == "transcript_counts" else raw_vals.astype(float)
        return v
    def _cbar_kwargs():
        kw = {}
        if cbar_range:
            if cbar_range[0] is not None:
                kw["cmin"] = cbar_range[0]
            if cbar_range[1] is not None:
                kw["cmax"] = cbar_range[1]
        return kw
    if alt_res is not None:
        source_label = f" [{core._SEG_TOOL_LABELS.get(_alt_tool, _alt_tool)}]"
        bdf      = alt_res["cells_df"]
        x        =  bdf["x_centroid"].values
        y        = -bdf["y_centroid"].values
        cell_ids     = bdf.index.tolist()
        cell_ids_str = [str(c) for c in cell_ids]
        _ids_arr     = np.asarray(cell_ids_str)
        show_legend  = False
        traces       = None   # filled below

        if color_by == "niche":
            if "niche" in bdf.columns:
                niche_vals = spatial._niche_str(bdf["niche"])
                unique_niches = sorted(niche_vals.unique(), key=lambda v: (0, int(v)) if v.isdigit() else (1, v))
                cmap = {n: core.CLUSTER_COLORS[i % len(core.CLUSTER_COLORS)] for i, n in enumerate(unique_niches)}
                traces = []
                for n in unique_niches:
                    mask = (niche_vals == n).values
                    traces.append(go.Scattergl(
                        x=x[mask], y=y[mask], mode="markers",
                        marker=dict(size=size, color=cmap[n], opacity=opacity),
                        name=spatial._niche_name(n), legendgroup=n, showlegend=True,
                        text=_ids_arr[mask], customdata=_ids_arr[mask],
                        hovertemplate="<b>%{text}</b><extra>Niche " + n + "</extra>",
                    ))
                show_legend = True
            # else fall through to transcript_counts

        elif color_by == "cluster":
            # Look for any cluster column in cells_df (added after UMAP/clustering on reseg cells)
            # When corrected counts are active, prefer cluster_split_* columns
            _cluster_cols = reseg._reseg_cluster_cols(bdf.columns)
            if use_corrected:
                cluster_col = next((c for c in _cluster_cols if "split" in c), None) or next(iter(_cluster_cols), None)
            else:
                cluster_col = next((c for c in _cluster_cols if "split" not in c), None) or next(iter(_cluster_cols), None)
            # a cluster set installed on this run and picked in Clustering Method
            cluster_col = clustersets._cs_reseg_col(bdf.columns, method) or cluster_col
            if cluster_col is not None:
                cluster_vals = bdf[cluster_col].astype(str)
                unique_clusters = sorted(cluster_vals.unique(), key=lambda v: (0, int(v)) if v.isdigit() else (1, v))
                cmap = {cl: core.CLUSTER_COLORS[i % len(core.CLUSTER_COLORS)] for i, cl in enumerate(unique_clusters)}
                traces = []
                for cl in unique_clusters:
                    mask = (cluster_vals == cl).values
                    traces.append(go.Scattergl(
                        x=x[mask], y=y[mask], mode="markers",
                        marker=dict(size=size, color=cmap[cl], opacity=opacity),
                        name=f"Cluster {cl}", legendgroup=cl, showlegend=True,
                        text=_ids_arr[mask], customdata=_ids_arr[mask],
                        hovertemplate="<b>%{text}</b><extra>Cluster " + cl + "</extra>",
                    ))
                show_legend = True
            # If no cluster column: fall through to generic fallback (transcript_counts)

        elif color_by.startswith("cell_type:"):
            _method = color_by.split(":")[1]
            aligned = annotation._reseg_cell_type_labels(_method, alt_res, cell_ids_str)
            if aligned is None:     # no labels, or stale ones (cleared by the helper)
                color_by = "transcript_counts"
            else:
                cell_types = sorted(aligned.unique())
                cmap = {ct: core.CLUSTER_COLORS[i % len(core.CLUSTER_COLORS)] for i, ct in enumerate(cell_types)}
                traces = []
                for ct in cell_types:
                    mask = (aligned == ct).values
                    traces.append(go.Scattergl(
                        x=x[mask], y=y[mask], mode="markers",
                        marker=dict(size=size, color=cmap[ct], opacity=opacity),
                        name=ct, legendgroup=ct, showlegend=True,
                        text=_ids_arr[mask], customdata=_ids_arr[mask],
                        hovertemplate="<b>%{text}</b><extra>" + ct + "</extra>",
                    ))
                show_legend = True

        elif _gs is not None:
            if len(_gs[0]) == len(bdf):
                traces = [_gs_trace(x, y, cell_ids_str)]

        elif color_by == "gene":
            vals = _get_expr_values(gene, alt_res, use_corrected=use_corrected)
            if vals is None and use_corrected:
                # imputed gene: no corrected values exist, show the original ones
                # (as the Xenium path does)
                vals = _get_expr_values(gene, alt_res, use_corrected=False)
            if vals is not None:
                traces = [go.Scattergl(
                    x=x, y=y, mode="markers",
                    marker=dict(size=size, color=vals, colorscale="Plasma", opacity=opacity,
                                showscale=True,
                                colorbar=dict(title=f"{gene}<br>(log1p)", thickness=12, len=0.5, x=1.02),
                                **_cbar_kwargs()),
                    text=cell_ids_str, customdata=cell_ids_str,
                    hovertemplate="<b>%{text}</b><br>" + gene + ": %{marker.color:.2f}<extra></extra>",
                    name=gene,
                )]

        elif color_by and color_by.startswith("roi_") and color_by in bdf.columns:
            labels = bdf[color_by].fillna("(none)").astype(str)
            u_vals = [v for v in sorted(labels.unique()) if v != "(none)"]
            cmap   = {v: regions._roi_color(i) for i, v in enumerate(u_vals)}
            cmap["(none)"] = "#666666"
            colors = labels.map(cmap).values
            traces = [go.Scattergl(
                x=x, y=y, mode="markers",
                marker=dict(size=size, color=colors, opacity=opacity),
                text=cell_ids_str, customdata=cell_ids_str,
                hovertemplate="<b>%{text}</b><br>" + color_by + ": %{text}<extra></extra>",
                name=color_by,
            )]
            show_legend = False

        if traces is None:
            # Fallback: QC column if available, otherwise transcript counts
            if color_by in bdf.columns:
                vals  = _color_vals(bdf[color_by].values, color_by)
                label = core.QC_METRICS.get(color_by, color_by)
                if color_by == "transcript_counts":
                    label = "Transcript Counts (log1p)"
                cs    = core.COLORSCALES.get(color_by, "Viridis")
            else:
                vals  = np.log1p(bdf["transcript_counts"].values)
                label = "Transcript Counts (log1p)"
                cs    = "Viridis"
            traces = [go.Scattergl(
                x=x, y=y, mode="markers",
                marker=dict(size=size, color=vals, colorscale=cs, opacity=opacity, showscale=True,
                            colorbar=dict(title=label, thickness=12, len=0.5, x=1.02),
                            **_cbar_kwargs()),
                text=cell_ids_str, customdata=cell_ids_str,
                hovertemplate="<b>Cell %{text}</b><br>" + label + ": %{marker.color:.2f}<extra></extra>",
                name=label,
            )]
    else:
        df  = core.DATA["df"]
        x   =  df["x_centroid"].values   # x_centroid is in µm
        y   = -df["y_centroid"].values   # y_centroid is in µm, negate for plot
        source_label    = ""

        # ── Multi-sample: merge extra datasets ──────────────────────────
        if dataview.EXTRA_DATASETS:
            # Concatenate extra datasets into combined arrays
            all_x       = [x]
            all_y       = [y]
            all_ids     = [df.index.tolist()]
            all_dfs     = [df]
            all_offsets = [0.0]
            for eds in dataview.EXTRA_DATASETS:
                edf    = eds["df"]
                offset = eds["x_offset"]
                all_x.append(edf["x_centroid"].values + offset)
                all_y.append(-edf["y_centroid"].values)
                all_ids.append(edf.index.tolist())
                all_dfs.append(edf)
                all_offsets.append(offset)

            x_combined   = np.concatenate(all_x)
            y_combined   = np.concatenate(all_y)
            ids_combined = [cid for chunk in all_ids for cid in chunk]

            # Build per-sample separator annotations (vertical dashed lines)
            _separator_shapes = []
            for sep_x in dataview._sample_separator_xs():     # middle of each gap
                _separator_shapes.append(dict(
                    type="line", x0=sep_x, x1=sep_x,
                    y0=0, y1=1, yref="paper",
                    line=dict(color=core.MUTED, width=1, dash="dot"),
                ))

            # Build color values across all datasets
            if color_by == "cluster":
                # Combine cluster IDs per dataset — use dataset index as prefix to avoid collisions
                all_clust_vals  = []
                all_clust_names = []
                cmap_combined   = {}
                color_idx       = 0
                for di, (edf, offset) in enumerate(zip(all_dfs, all_offsets)):
                    _method = method or (edf.get("cluster_methods", [None])[0]
                                         if hasattr(edf, "get") else None)
                    # edf is a plain DataFrame; cluster_methods come from DATA or EXTRA_DATASETS[di-1]
                    ds = core.DATA if di == 0 else dataview.EXTRA_DATASETS[di - 1]
                    _method = method if method in ds.get("cluster_methods", []) else (ds.get("cluster_methods") or [method])[0]
                    col_name = f"clust__{_method}"
                    if col_name in edf.columns:
                        raw_vals = edf[col_name].values
                        # Prefix dataset index so cluster IDs don't collide across samples
                        # (not for a set computed jointly: its labels are the same everywhere)
                        _pre = "C" if (_method == method and clustersets._cs_is_shared(_method)) else f"S{di}:C"
                        prefixed = [f"{_pre}{v}" for v in raw_vals]
                        all_clust_vals.extend(prefixed)
                        for pv in prefixed:
                            if pv not in cmap_combined:
                                cmap_combined[pv] = core.CLUSTER_COLORS[color_idx % len(core.CLUSTER_COLORS)]
                                color_idx += 1
                        all_clust_names.extend(prefixed)
                    else:
                        # a sample without cluster sets (or without this one): one
                        # placeholder group, coloured like any other label
                        placeholders = [f"S{di}:C0"] * len(edf)
                        all_clust_vals.extend(placeholders)
                        all_clust_names.extend(placeholders)
                        if placeholders and placeholders[0] not in cmap_combined:
                            cmap_combined[placeholders[0]] = core.CLUSTER_COLORS[color_idx % len(core.CLUSTER_COLORS)]
                            color_idx += 1

                # Build one trace per unique cluster-label across all samples
                unique_labels = list(dict.fromkeys(all_clust_vals))
                arr_x   = x_combined
                arr_y   = y_combined
                arr_ids = np.array(ids_combined)
                arr_lbl = np.array(all_clust_vals)
                traces  = []
                for lbl in unique_labels:
                    mask = arr_lbl == lbl
                    traces.append(go.Scattergl(
                        x=arr_x[mask], y=arr_y[mask], mode="markers",
                        marker=dict(size=size, color=cmap_combined[lbl], opacity=opacity),
                        name=lbl, legendgroup=lbl, showlegend=True,
                        text=arr_ids[mask].tolist(),
                        customdata=arr_ids[mask].tolist(),
                        hovertemplate="<b>%{text}</b><extra>" + lbl + "</extra>",
                    ))
                show_legend = True

            elif color_by == "niche":
                # Combine niche labels across datasets
                all_niche_vals = []
                for edf in all_dfs:
                    if "niche" in edf.columns:
                        all_niche_vals.extend(spatial._niche_str(edf["niche"]).tolist())
                    else:
                        all_niche_vals.extend(["N/A"] * len(edf))
                arr_niche = np.array(all_niche_vals)
                arr_x   = x_combined
                arr_y   = y_combined
                arr_ids = np.array(ids_combined)
                unique_niches = sorted(set(all_niche_vals), key=lambda v: (0, int(v)) if v.isdigit() else (1, v))
                ncmap = {n: core.CLUSTER_COLORS[i % len(core.CLUSTER_COLORS)] for i, n in enumerate(unique_niches)}
                traces = []
                for _n in unique_niches:
                    mask = arr_niche == _n
                    traces.append(go.Scattergl(
                        x=arr_x[mask], y=arr_y[mask], mode="markers",
                        marker=dict(size=size, color=ncmap[_n], opacity=opacity),
                        name=spatial._niche_name(_n), legendgroup=_n, showlegend=True,
                        text=arr_ids[mask].tolist(),
                        customdata=arr_ids[mask].tolist(),
                        hovertemplate="<b>%{text}</b><extra>Niche " + _n + "</extra>",
                    ))
                show_legend = True

            elif color_by.startswith("cell_type:"):
                # Combine cell type labels from each dataset's annotation state
                _method = color_by.split(":")[1]
                all_labels_list = []
                with core._annot_lock:
                    primary_labels = core._annot_state.get(f"labels_{_method}")
                for di, edf in enumerate(all_dfs):
                    if di == 0:
                        lbl_series = primary_labels
                    else:
                        lbl_series = dataview._extra_cell_type_labels(
                            dataview.EXTRA_DATASETS[di - 1], _method)[0]
                    if lbl_series is not None:
                        aligned = lbl_series.reindex(edf.index).fillna("Unknown")
                        all_labels_list.extend(aligned.tolist())
                    else:
                        all_labels_list.extend(["Unknown"] * len(edf))

                arr_lbl  = np.array(all_labels_list)
                arr_x    = x_combined
                arr_y    = y_combined
                arr_ids  = np.array(ids_combined)
                cell_types = sorted(set(all_labels_list))
                cmap = {ct: core.CLUSTER_COLORS[i % len(core.CLUSTER_COLORS)] for i, ct in enumerate(cell_types)}
                traces = []
                for ct in cell_types:
                    mask = arr_lbl == ct
                    traces.append(go.Scattergl(
                        x=arr_x[mask], y=arr_y[mask], mode="markers",
                        marker=dict(size=size, color=cmap[ct], opacity=opacity),
                        name=ct, legendgroup=ct, showlegend=True,
                        text=arr_ids[mask].tolist(),
                        customdata=arr_ids[mask].tolist(),
                        hovertemplate="<b>%{text}</b><extra>" + ct + "</extra>",
                    ))
                show_legend = True

            elif _gs is not None and len(_gs[0]) == len(x_combined):
                traces = [_gs_trace(x_combined, y_combined, ids_combined)]
                show_legend = False

            elif color_by == "gene":
                # Collect expression per dataset
                # SPLIT-corrected counts only when every sample has them: one colour
                # scale never mixes corrected and original values
                _mc = dataview._multi_use_corrected(use_corrected)
                all_vals = []
                for di, edf in enumerate(all_dfs):
                    if di == 0:
                        _v0 = dataview._xenium_gene_values(gene, _mc)   # None: not in this sample
                    else:
                        _v0 = dataview._extra_gene_values(dataview.EXTRA_DATASETS[di - 1], gene, _mc)
                    all_vals.append(_v0 if _v0 is not None and len(_v0) == len(edf)
                                    else np.zeros(len(edf)))
                vals_combined = np.concatenate(all_vals)
                traces = [go.Scattergl(
                    x=x_combined, y=y_combined, mode="markers",
                    marker=dict(size=size, color=vals_combined, colorscale="Plasma", opacity=opacity,
                                showscale=True,
                                colorbar=dict(title=f"{gene}<br>(log1p{', SPLIT-corrected' if _mc else ''})",
                                              thickness=12, len=0.5, x=1.02),
                                **_cbar_kwargs()),
                    text=ids_combined, customdata=ids_combined,
                    hovertemplate="<b>%{text}</b><br>" + gene + ": %{marker.color:.2f}<extra></extra>",
                    name=gene,
                )]
                show_legend = False

            elif color_by.startswith("roi_"):
                # ROI class: categorical colours; samples without the column show "(none)"
                labels = pd.concat([
                    (edf[color_by] if color_by in edf.columns
                     else pd.Series(pd.NA, index=edf.index, dtype=object))
                    for edf in all_dfs
                ]).fillna("(none)").astype(str)
                u_vals = [v for v in sorted(labels.unique()) if v != "(none)"]
                cmap   = {v: regions._roi_color(i) for i, v in enumerate(u_vals)}
                cmap["(none)"] = "#666666"
                traces = [go.Scattergl(
                    x=x_combined, y=y_combined, mode="markers",
                    marker=dict(size=size, color=labels.map(cmap).values, opacity=opacity),
                    text=ids_combined, customdata=ids_combined,
                    hovertemplate="<b>%{text}</b><extra></extra>",
                    name=color_by,
                )]
                show_legend = False

            else:
                # QC / numeric column
                label = core.QC_METRICS.get(color_by, color_by)
                cs    = core.COLORSCALES.get(color_by, "Viridis")
                all_vals = []
                for edf in all_dfs:
                    if color_by in edf.columns:
                        all_vals.append(_color_vals(edf[color_by].values, color_by))
                    else:
                        all_vals.append(np.zeros(len(edf)))
                if color_by == "transcript_counts":
                    label = "Transcript Counts (log1p)"
                vals_combined = np.concatenate(all_vals)
                traces = [go.Scattergl(
                    x=x_combined, y=y_combined, mode="markers",
                    marker=dict(size=size, color=vals_combined, colorscale=cs, opacity=opacity,
                                showscale=True,
                                colorbar=dict(title=label, thickness=12, len=0.5, x=1.02),
                                **_cbar_kwargs()),
                    text=ids_combined, customdata=ids_combined,
                    hovertemplate="<b>%{text}</b><br>" + label + ": %{marker.color:.2f}<extra></extra>",
                    name=label,
                )]
                show_legend = False

            source_label = f" [{len(dataview.EXTRA_DATASETS) + 1} samples]"

        else:
            # ── Single sample (no extra datasets) ───────────────────────
            # _separator_shapes already = [] from function-level init

            # ── Cell scatter traces ──────────────────────────────────────────
            # A niche / ROI column can be missing (niches computed on a reseg run or a
            # subset, ROI deleted, dataset switched): fall back like the reseg branch.
            # The same goes for any plain column colour (a QC column absent from an
            # exported store) and for 'cluster' when its column does not exist (no cluster
            # sets, or a reseg source is selected but its result is not loaded).
            if _gs is not None and len(_gs[0]) != len(df):
                _gs, color_by = None, "transcript_counts"
            if (color_by not in ("cluster", "gene") and not color_by.startswith("cell_type:")
                    and _gs is None and color_by not in df.columns):
                color_by = "transcript_counts"
            # (`cluster_col` is a local name in this function: spell the column out)
            if color_by == "cluster" and f"clust__{method}" not in df.columns:
                color_by = "transcript_counts"
            # ... and for a gene this dataset does not have (as a reseg run without it)
            _gene_vals = None
            if color_by == "gene":
                _gene_vals = dataview._xenium_gene_values(gene, use_corrected)
                if _gene_vals is None or len(_gene_vals) != len(df):
                    color_by = "transcript_counts"
            if color_by == "niche":
                if "niche" in df.columns:
                    _niche_vals = spatial._niche_str(df["niche"])
                    _unique_niches = sorted(_niche_vals.unique(), key=lambda v: (0, int(v)) if v.isdigit() else (1, v))
                    _ncmap = {n: core.CLUSTER_COLORS[i % len(core.CLUSTER_COLORS)] for i, n in enumerate(_unique_niches)}
                    traces = []
                    for _n in _unique_niches:
                        _nmask = (_niche_vals == _n).values
                        _nids = df.index.tolist()
                        traces.append(go.Scattergl(
                            x=x[_nmask], y=y[_nmask], mode="markers",
                            marker=dict(size=size, color=_ncmap[_n], opacity=opacity),
                            name=spatial._niche_name(_n), legendgroup=_n, showlegend=True,
                            text=np.array(_nids)[_nmask].tolist(),
                            customdata=np.array(_nids)[_nmask].tolist(),
                            hovertemplate="<b>%{text}</b><extra>Niche " + _n + "</extra>",
                        ))
                    show_legend = True
                # else fall through — no niche column yet
            elif color_by == "cluster":
                traces, show_legend = _categorical_traces(x, y, df, method, size, opacity), True
            elif color_by.startswith("cell_type:"):
                _method = color_by.split(":")[1]
                _lkey = f"labels_{_method}"
                # Always draw every cell; RCTD weight pies (SVG wedge shapes) are added
                # on top for the cells in view and refreshed on pan/zoom by update_plots.
                traces = _cell_type_traces(x, y, df, size, opacity, "spatial",
                                           labels_key=_lkey)
                _pie_shapes = _rctd_pie_shapes_for_view(color_by, relayout) or []
                show_legend = True
            elif _gs is not None:
                traces, show_legend = [_gs_trace(x, y, df.index.tolist())], False
            elif color_by == "gene":
                vals = _gene_vals
                traces = [go.Scattergl(
                    x=x, y=y, mode="markers",
                    marker=dict(size=size, color=vals, colorscale="Plasma", opacity=opacity,
                                showscale=True,
                                colorbar=dict(title=f"{gene}<br>(log1p)", thickness=12, len=0.5, x=1.02),
                                **_cbar_kwargs()),
                    text=df.index.tolist(), customdata=df.index.tolist(),
                    hovertemplate="<b>%{text}</b><br>" + gene + ": %{marker.color:.2f}<extra></extra>",
                    name=gene,
                )]
                show_legend = False
            elif color_by and color_by.startswith("roi_") and color_by in df.columns:
                labels = df[color_by].fillna("(none)").astype(str)
                u_vals = [v for v in sorted(labels.unique()) if v != "(none)"]
                cmap   = {v: regions._roi_color(i) for i, v in enumerate(u_vals)}
                cmap["(none)"] = "#666666"
                colors = labels.map(cmap).values
                ids    = df.index.tolist()
                traces = [go.Scattergl(
                    x=x, y=y, mode="markers",
                    marker=dict(size=size, color=colors, opacity=opacity),
                    text=ids, customdata=ids,
                    hovertemplate="<b>%{text}</b><extra></extra>",
                    name=color_by,
                )]
                show_legend = False
            else:
                label = core.QC_METRICS.get(color_by, color_by)
                vals = (_color_vals(df[color_by].values, color_by) if color_by in df.columns
                        else np.zeros(len(df)))
                if color_by == "transcript_counts":
                    label = "Transcript Counts (log1p)"
                traces = [go.Scattergl(
                    x=x, y=y, mode="markers",
                    marker=dict(size=size, color=vals,
                                colorscale=core.COLORSCALES.get(color_by, "Viridis"),
                                opacity=opacity, showscale=True,
                                colorbar=dict(title=label, thickness=12, len=0.5, x=1.02),
                                **_cbar_kwargs()),
                    text=df.index.tolist(), customdata=df.index.tolist(),
                    hovertemplate="<b>%{text}</b><br>" + label + ": %{marker.color:.2f}<extra></extra>",
                    name=label,
                )]
                show_legend = False

    # ── Boundary overlay traces ──────────────────────────────────────────
    boundary_toggles = boundary_toggles or []
    boundary_status = ""

    # Each boundary source is rendered independently
    _needs_zoom_msg = False

    def _add_boundary(bounds_dict, color, name, reseg_cells_df=None):
        """Render a boundary set using viewport filtering. reseg_cells_df uses Baysor/Proseg cells."""
        nonlocal _needs_zoom_msg, show_legend
        if not bounds_dict:
            return
        if reseg_cells_df is not None:
            visible = dataview.viewport_reseg_cell_ids(relayout or {}, reseg_cells_df)
        else:
            visible = dataview.viewport_cell_ids(relayout or {})
        if visible is None:
            _needs_zoom_msg = True
            return
        if len(visible) > core.BOUNDARY_CELL_LIMIT:
            return
        t = dataview.build_boundary_trace(visible, bounds_dict, color, name)
        if t:
            traces.append(t)
            show_legend = True

    # "Xenium Cell/Nuclei Boundaries" always show original Xenium boundaries (zoom required).
    if "cell" in boundary_toggles:
        _add_boundary(core.DATA["cell_bounds"], "#00d4ff", "Xenium Cell Boundaries")
    if "nucleus" in boundary_toggles:
        _add_boundary(core.DATA.get("nucleus_bounds", {}), "#ff9f43", "Xenium Nuclei Boundaries")

    # Proseg boundaries (zoom required, uses Proseg cell centroids for viewport)
    if "proseg" in boundary_toggles:
        with core._proseg_lock:
            _pres_local = core._proseg_state.get("result")
        if _pres_local:
            _add_boundary(_pres_local["cell_bounds"], "#2ed573", "Proseg Boundaries",
                          reseg_cells_df=_pres_local["cells_df"])

    # Baysor boundaries (zoom required, uses Baysor cell centroids for viewport)
    if "baysor" in boundary_toggles:
        with core._baysor_lock:
            _bres_local = core._baysor_state.get("result")
        if _bres_local:
            _add_boundary(_bres_local["cell_bounds"], "#a29bfe", "Baysor Boundaries",
                          reseg_cells_df=_bres_local["cells_df"])

    # Image-segmentation boundaries (zoom required, uses that run's centroids for the viewport)
    if "imgseg" in boundary_toggles:
        _ires_local = core._seg_result("imgseg")
        if _ires_local:
            _add_boundary(_ires_local["cell_bounds"], "#ff6b9d", "Image segmentation Boundaries",
                          reseg_cells_df=_ires_local["cells_df"])

    if _needs_zoom_msg:
        boundary_status = " · zoom in to see Xenium boundaries"

    title_text = "Spatial View" + source_label + boundary_status + extra_title

    # Transcript density: an SVG heat layer, which plotly.js draws below the WebGL markers
    _dens = analysis._density_heatmap_traces(density, relayout) if density else []
    if _dens:
        if any(getattr(getattr(t, "marker", None), "showscale", False) for t in traces):
            for _dt in _dens:
                _dt.colorbar.x = 1.12        # beside the cells' own colour bar
        traces = list(_dens) + list(traces)
    fig = go.Figure(data=traces)
    if morph_image:
        fig.update_layout(images=[morph_image])
    layout_kwargs = dict(
        **_base_layout(title_text, "X (µm)", "Y (µm)", equal_aspect=True),
        showlegend=show_legend,
        legend=dict(
            bgcolor="rgba(0,0,0,0.45)", font=dict(size=10),
            itemsizing="constant", tracegroupgap=1,
            x=0.01, y=0.99, xanchor="left", yanchor="top",
        ),
        # per dataset: after Load Sample the client must drop the previous zoom
        uirevision=f"spatial-{jobs._dataset_gen}",
    )
    all_shapes = list(_separator_shapes) + list(_pie_shapes)
    _spatial_base_shapes[:] = all_shapes
    all_shapes += _spatial_overlay_shapes(sdata_roi_on, sdata_patches_on,
                                          _alt_tool)
    if all_shapes:
        layout_kwargs["shapes"] = all_shapes
    fig.update_layout(**layout_kwargs)
    # Explicitly preserve viewport ranges to prevent zoom resets on figure rebuild
    if relayout:
        rx0 = relayout.get("xaxis.range[0]")
        rx1 = relayout.get("xaxis.range[1]")
        ry0 = relayout.get("yaxis.range[0]")
        ry1 = relayout.get("yaxis.range[1]")
        if all(v is not None for v in (rx0, rx1, ry0, ry1)):
            fig.update_layout(
                xaxis_range=[float(rx0), float(rx1)],
                yaxis_range=[float(ry0), float(ry1)],
            )
    return fig


def make_cell_type_abundance_fig(width=None, height=None):
    """Build bar chart of cell type abundances for all computed annotation methods."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    with core._annot_lock:
        label_keys = [k for k in core._annot_state if k.startswith("labels_") and core._annot_state[k] is not None]
        # Skip generic keys (e.g. "labels_rctd") when a mode-specific key exists
        # (e.g. "labels_rctd_doublet") — they hold the same data and would show as duplicates.
        generic_with_specific = set()
        for key in label_keys:
            suffix = key[len("labels_"):]
            parts = suffix.split("_")
            if len(parts) >= 2 and parts[1] in ("full", "doublet", "multi"):
                generic_with_specific.add(f"labels_{parts[0]}")
        label_keys = [k for k in label_keys if k not in generic_with_specific]

        method_data = []
        for key in label_keys:
            val = core._annot_state[key]
            suffix = key[len("labels_"):]
            parts = suffix.split("_", 1)
            base = parts[0].lower()
            if base in core._SEG_TOOL_LABELS:
                display = suffix.replace("_", " ").title()
            elif base in ("celltypist", "seurat", "rctd"):
                method_map = {"celltypist": "CellTypist", "seurat": "Seurat", "rctd": "RCTD"}
                method_name = method_map.get(base, base.title())
                if len(parts) > 1:
                    display = f"{method_name} ({parts[1].replace('_', ':')})"
                else:
                    display = method_name
            else:
                display = suffix.replace("_", " ").title()
            method_data.append((display, val))

    if not method_data:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor=core.CARD_BG,
            plot_bgcolor=core.PLOT_BG,
            font=dict(color=core.TEXT, size=12),
            margin=dict(l=10, r=10, t=10, b=10),
            annotations=[dict(
                text="No annotation computed yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color=core.TEXT),
            )],
        )
        return fig

    n = len(method_data)
    colors = core.CLUSTER_COLORS if core.CLUSTER_COLORS else ["#5B9BD5"]
    fig = make_subplots(
        rows=n, cols=1,
        subplot_titles=[m[0] for m in method_data],
        shared_xaxes=False,
        vertical_spacing=max(0.08, 0.15 / max(n, 1)),
    )

    for i, (name, labels) in enumerate(method_data):
        if hasattr(labels, "value_counts"):
            counts = labels.value_counts(normalize=True).sort_values(ascending=False)
        else:
            import pandas as _pd
            counts = _pd.Series(labels).value_counts(normalize=True).sort_values(ascending=False)
        cell_types = [str(c) for c in counts.index]
        freqs = counts.values
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Bar(
                x=cell_types,
                y=freqs,
                name=name,
                marker_color=color,
                text=[f"{v:.1%}" for v in freqs],
                textposition="outside",
                showlegend=False,
            ),
            row=i + 1, col=1,
        )
        fig.update_yaxes(tickformat=".0%", row=i + 1, col=1,
                         gridcolor=core.BORDER, zeroline=False)
        fig.update_xaxes(tickangle=-45, row=i + 1, col=1)

    fig.update_layout(
        paper_bgcolor=core.CARD_BG,
        plot_bgcolor=core.PLOT_BG,
        font=dict(color=core.TEXT, size=11),
        margin=dict(l=50, r=20, t=40, b=60),
        showlegend=False,
    )
    for ann in fig.layout.annotations:
        ann.font.color = core.TEXT
        ann.font.size = 12

    return fig


def _make_placeholder_fig(msg="No analysis computed yet."):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(color=core.MUTED, size=13))
    fig.update_layout(paper_bgcolor=core.CARD_BG, plot_bgcolor=core.PLOT_BG,
                      margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def make_cooccurrence_fig(zmin=None, zmax=None, width=None, height=None):
    with core._spatial_lock:
        result = core._spatial_state.get("result")
    if result is None:
        return _make_placeholder_fig()
    cell_types = result["cell_types"]
    matrix = result["cooccurrence"]
    n = len(cell_types)
    hm_kw = {}
    if zmin is not None or zmax is not None:
        if zmin is not None: hm_kw["zmin"] = zmin
        if zmax is not None: hm_kw["zmax"] = zmax
    else:
        hm_kw["zmid"] = 0
    fig = go.Figure(go.Heatmap(
        z=matrix, x=cell_types, y=cell_types,
        colorscale="RdBu_r",
        hovertemplate="%{y} → %{x}<br>log2 score: %{z:.2f}<extra></extra>",
        colorbar=dict(title=dict(text="log2 score", font=dict(color=core.TEXT)), tickfont=dict(color=core.TEXT)),
        **hm_kw,
    ))
    fig.update_layout(
        paper_bgcolor=core.CARD_BG, plot_bgcolor=core.PLOT_BG,
        font=dict(color=core.TEXT, size=10),
        margin=dict(l=120, r=20, t=30, b=120),
        height=int(height) if height else max(400, 30 * n),
        xaxis=dict(tickangle=-45, gridcolor=core.BORDER, linecolor=core.BORDER),
        yaxis=dict(gridcolor=core.BORDER, linecolor=core.BORDER),
        title=dict(text="Co-occurrence (log2 observed/expected)", font=dict(color=core.TEXT, size=12), x=0.5),
    )
    if width:
        fig.update_layout(width=int(width))
    return fig


def make_enrichment_fig(zmin=None, zmax=None, width=None, height=None):
    with core._spatial_lock:
        result = core._spatial_state.get("result")
    if result is None:
        return _make_placeholder_fig()
    cell_types = result["cell_types"]
    matrix = result["enrichment"]
    n = len(cell_types)
    hm_kw = {}
    if zmin is not None or zmax is not None:
        if zmin is not None: hm_kw["zmin"] = zmin
        if zmax is not None: hm_kw["zmax"] = zmax
    else:
        hm_kw["zmid"] = 0
    fig = go.Figure(go.Heatmap(
        z=matrix, x=cell_types, y=cell_types,
        colorscale="RdBu_r",
        hovertemplate="%{y} → %{x}<br>log2 enrichment: %{z:.2f}<extra></extra>",
        colorbar=dict(title=dict(text="log2 enrichment", font=dict(color=core.TEXT)), tickfont=dict(color=core.TEXT)),
        **hm_kw,
    ))
    fig.update_layout(
        paper_bgcolor=core.CARD_BG, plot_bgcolor=core.PLOT_BG,
        font=dict(color=core.TEXT, size=10),
        margin=dict(l=120, r=20, t=30, b=120),
        height=int(height) if height else max(400, 30 * n),
        xaxis=dict(tickangle=-45, gridcolor=core.BORDER, linecolor=core.BORDER),
        yaxis=dict(gridcolor=core.BORDER, linecolor=core.BORDER),
        title=dict(text="Neighborhood Enrichment (log2 observed/expected)", font=dict(color=core.TEXT, size=12), x=0.5),
    )
    if width:
        fig.update_layout(width=int(width))
    return fig


def make_interaction_graph_fig(width=None, height=None, normalize=False, min_freq=0.0):
    with core._spatial_lock:
        result = core._spatial_state.get("result")
    if result is None:
        return _make_placeholder_fig()
    cell_types = list(result["cell_types"])
    adjacency = result["adjacency"].copy()
    freqs = result["freqs"].copy()
    if min_freq and min_freq > 0:
        keep = freqs >= (min_freq / 100.0)
        cell_types = [ct for i, ct in enumerate(cell_types) if keep[i]]
        adjacency = adjacency[np.ix_(keep, keep)]
        freqs = freqs[keep]
        if freqs.sum() > 0:
            freqs = freqs / freqs.sum()
    n = len(cell_types)
    if n == 0:
        return _make_placeholder_fig("No cell types above frequency threshold.")

    angles = [2 * np.pi * i / n for i in range(n)]
    node_x = [np.cos(a) for a in angles]
    node_y = [np.sin(a) for a in angles]

    # Top 30 edges by weight
    edges = []
    for a in range(n):
        # Same-type pairs (a == b) have no drawable edge in this layout; skip them
        # so they do not take top-30 slots or set the width scale.
        for b in range(a + 1, n):
            raw_w = adjacency[a, b] + adjacency[b, a]
            if raw_w > 0:
                if normalize:
                    # Observed / expected under random mixing
                    expected = freqs[a] * freqs[b] * adjacency.sum()
                    expected *= 2  # both orderings
                    w = (raw_w / expected) if expected > 0 else 0
                else:
                    w = raw_w
                if w > 0:
                    edges.append((w, a, b))
    edges.sort(reverse=True)
    edges = edges[:30]
    max_w = edges[0][0] if edges else 1

    traces = []
    for w, a, b in edges:
        lw = max(0.5, 5 * w / max_w)
        alpha = max(0.1, 0.7 * w / max_w)
        hover_txt = f"{cell_types[a]} ↔ {cell_types[b]}<br>{'O/E ratio' if normalize else 'count'}: {w:.2f}"
        traces.append(go.Scatter(
            x=[node_x[a], node_x[b], None],
            y=[node_y[a], node_y[b], None],
            mode="lines",
            line=dict(width=lw, color=f"rgba(150,150,200,{alpha:.2f})"),
            hoverinfo="text", text=[hover_txt, hover_txt, None],
            showlegend=False,
        ))

    colors = [core.CLUSTER_COLORS[i % len(core.CLUSTER_COLORS)] for i in range(n)]
    node_sizes = [max(8, 40 * f) for f in freqs]
    traces.append(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=node_sizes, color=colors, line=dict(width=1, color=core.BORDER)),
        text=cell_types, textposition="top center",
        textfont=dict(color=core.TEXT, size=9),
        hovertemplate="%{text}<br>freq: %{customdata:.1%}<extra></extra>",
        customdata=freqs,
        showlegend=False,
    ))

    title_text = "Interaction Graph — O/E normalized (top 30)" if normalize else "Interaction Graph — raw counts (top 30)"
    fig = go.Figure(traces)
    fig.update_layout(
        paper_bgcolor=core.CARD_BG, plot_bgcolor=core.PLOT_BG,
        font=dict(color=core.TEXT),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(visible=False, range=[-1.4, 1.4]),
        yaxis=dict(visible=False, range=[-1.4, 1.4], scaleanchor="x"),
        title=dict(text=title_text, font=dict(color=core.TEXT, size=12), x=0.5),
    )
    return fig


def make_chord_fig(width=None, height=None, normalize=False, min_freq=0.0):
    with core._spatial_lock:
        result = core._spatial_state.get("result")
    if result is None:
        return _make_placeholder_fig()
    cell_types = list(result["cell_types"])
    adjacency = result["adjacency"].copy()
    freqs = result["freqs"].copy()
    if min_freq and min_freq > 0:
        keep = freqs >= (min_freq / 100.0)
        cell_types = [ct for i, ct in enumerate(cell_types) if keep[i]]
        adjacency = adjacency[np.ix_(keep, keep)]
        freqs = freqs[keep]
        if freqs.sum() > 0:
            freqs = freqs / freqs.sum()
    n = len(cell_types)
    if n == 0:
        return _make_placeholder_fig("No cell types above frequency threshold.")

    gap = 0.02  # radians between arcs
    total_arc = 2 * np.pi - n * gap
    arc_sizes = freqs * total_arc  # arc in radians proportional to freq

    # Compute start/end angles for each type
    starts = np.zeros(n)
    ends = np.zeros(n)
    cur = 0.0
    for i in range(n):
        starts[i] = cur
        ends[i] = cur + arc_sizes[i]
        cur = ends[i] + gap

    midpoints = (starts + ends) / 2

    traces = []
    colors = [core.CLUSTER_COLORS[i % len(core.CLUSTER_COLORS)] for i in range(n)]

    # Build weight matrix — optionally normalize by expected frequency
    adj_sym = adjacency + adjacency.T
    # adjacency is directed: A + A.T would count same-type pairs twice
    adj_sym[np.diag_indices_from(adj_sym)] = np.diag(adjacency)
    if normalize:
        total = adjacency.sum()  # total directed neighbour pairs
        norm_mat = np.zeros_like(adj_sym, dtype=float)
        for a in range(n):
            for b in range(n):
                expected = freqs[a] * freqs[b] * total
                if a != b:
                    expected *= 2
                norm_mat[a, b] = (adj_sym[a, b] / expected) if expected > 0 else 0
        weight_mat = norm_mat
    else:
        weight_mat = adj_sym.astype(float)
    # Scale by between-type weights only; same-type pairs are not drawn as chords
    _offdiag = weight_mat[~np.eye(n, dtype=bool)]
    max_adj = _offdiag.max() if _offdiag.size else 0
    if max_adj == 0:
        max_adj = 1

    for a in range(n):
        for b in range(a + 1, n):
            w = weight_mat[a, b]
            if w < max_adj * 0.01:
                continue
            alpha = max(0.05, 0.4 * w / max_adj)
            t_vals = np.linspace(0, 1, 30)
            p0x, p0y = np.cos(midpoints[a]), np.sin(midpoints[a])
            p1x, p1y = np.cos(midpoints[b]), np.sin(midpoints[b])
            pcx, pcy = 0.0, 0.0
            bx = (1 - t_vals)**2 * p0x + 2 * t_vals * (1 - t_vals) * pcx + t_vals**2 * p1x
            by = (1 - t_vals)**2 * p0y + 2 * t_vals * (1 - t_vals) * pcy + t_vals**2 * p1y
            traces.append(go.Scatter(
                x=np.append(bx, None), y=np.append(by, None),
                mode="lines",
                line=dict(width=max(1, 6 * w / max_adj), color=f"rgba(150,150,200,{alpha:.2f})"),
                hoverinfo="skip", showlegend=False,
            ))

    # Draw arcs for each cell type
    arc_pts = 40
    for i in range(n):
        theta = np.linspace(starts[i], ends[i], arc_pts)
        ax = np.cos(theta)
        ay = np.sin(theta)
        traces.append(go.Scatter(
            x=ax, y=ay, mode="lines",
            line=dict(width=8, color=colors[i]),
            name=cell_types[i],
            hovertemplate=f"{cell_types[i]}<br>freq: {freqs[i]:.1%}<extra></extra>",
        ))
        mx, my = np.cos(midpoints[i]) * 1.12, np.sin(midpoints[i]) * 1.12
        traces.append(go.Scatter(
            x=[mx], y=[my], mode="text",
            text=[cell_types[i]], textfont=dict(color=core.TEXT, size=9),
            hoverinfo="skip", showlegend=False,
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        paper_bgcolor=core.CARD_BG, plot_bgcolor=core.PLOT_BG,
        font=dict(color=core.TEXT),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(visible=False, range=[-1.5, 1.5]),
        yaxis=dict(visible=False, range=[-1.5, 1.5], scaleanchor="x"),
        title=dict(text="Chord Diagram — " + ("O/E normalized" if normalize else "raw counts"),
                   font=dict(color=core.TEXT, size=12), x=0.5),
        showlegend=False,
    )
    return fig


def make_niche_umap_fig(width=None, height=None, baysor_active=False, proseg_active=False, alt_tool=None):
    if core.DATA is None or core.DATA.get("df") is None:
        return _make_placeholder_fig()
    # Niches computed on a Baysor/Proseg run live in that run's cells_df
    with core._proseg_lock:
        pres = core._proseg_state["result"] if proseg_active and core._proseg_state["result"] else None
    with core._baysor_lock:
        bres = core._baysor_state["result"] if baysor_active and not pres and core._baysor_state["result"] else None
    # a further tool (alt_tool, e.g. "imgseg"): its loaded run, drawn like a Baysor / Proseg one
    _ores = (core._seg_result(alt_tool) if alt_tool not in (None, "xenium", "baysor", "proseg")
             and not (pres or bres) else None)
    alt_res = pres or bres or _ores
    _alt_tool = "proseg" if pres else "baysor" if bres else alt_tool if _ores else "xenium"
    df = alt_res["cells_df"] if alt_res is not None else core.DATA["df"]
    if "niche" not in df.columns:
        return _make_placeholder_fig("Run Spatial Niches to populate this tab.")
    # Use re-run UMAP columns if present, otherwise fall back to bundled DATA["umap"]
    if "umap_1" in df.columns and "umap_2" in df.columns and df["umap_1"].notna().any():
        xu = df["umap_1"].values.astype(float)
        yu = df["umap_2"].values.astype(float)
        axis_labels = ("UMAP 1", "UMAP 2")
    elif alt_res is None and core.DATA.get("umap") is not None and len(core.DATA["umap"]) == len(df):
        xu = core.DATA["umap"][:, 0].astype(float)
        yu = core.DATA["umap"][:, 1].astype(float)
        axis_labels = ("UMAP 1", "UMAP 2")
    elif "x_centroid" in df.columns:
        xu = df["x_centroid"].values.astype(float)
        yu = -df["y_centroid"].values.astype(float)
        axis_labels = ("X (µm)", "Y (µm)")
    else:
        return _make_placeholder_fig("No UMAP coordinates available.")
    niche_arr = spatial._niche_str(df["niche"]).values
    valid_mask = ~(np.isnan(xu) | np.isnan(yu))
    xu = xu[valid_mask]
    yu = yu[valid_mask]
    niches = niche_arr[valid_mask]
    unique_niches = sorted(set(niches), key=lambda v: (0, int(v)) if v.isdigit() else (1, v))
    traces = []
    for j, niche in enumerate(unique_niches):
        mask = niches == niche
        color = core.CLUSTER_COLORS[j % len(core.CLUSTER_COLORS)]
        traces.append(go.Scattergl(
            x=xu[mask], y=yu[mask],
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.7),
            name=spatial._niche_name(niche),
            hovertemplate=f"{spatial._niche_name(niche)}<extra></extra>",
        ))
    fig = go.Figure(traces)
    layout_kw = dict(
        paper_bgcolor=core.CARD_BG, plot_bgcolor=core.PLOT_BG,
        font=dict(color=core.TEXT),
        margin=dict(l=30, r=10, t=30, b=30),
        xaxis=dict(title=axis_labels[0], gridcolor=core.BORDER, linecolor=core.BORDER, color=core.TEXT),
        yaxis=dict(title=axis_labels[1], gridcolor=core.BORDER, linecolor=core.BORDER, color=core.TEXT),
        legend=dict(bgcolor=core.CARD_BG, bordercolor=core.BORDER, font=dict(color=core.TEXT, size=9)),
        title=dict(text="Niche UMAP", font=dict(color=core.TEXT, size=12), x=0.5),
    )
    if width:
        layout_kw["width"] = int(width)
    if height:
        layout_kw["height"] = int(height)
    fig.update_layout(**layout_kw)
    return fig


def make_roi_area_fig(width=None, height=None):
    """Bar chart of ROI polygon areas in µm²."""
    with core._roi_lock:
        rois = list(core._roi_state.get("rois", []))
    if not rois:
        return _make_placeholder_fig("No ROIs defined. Use lasso selection to draw ROIs.")

    from shapely.geometry import Polygon as _Polygon

    names, areas, colors, classes = [], [], [], []
    for roi in rois:
        pts = roi.get("polygon_xy", [])
        if len(pts) < 3:
            continue
        try:
            area = _Polygon(pts).area  # µm²
        except Exception:
            area = 0.0
        names.append(roi.get("name", "?"))
        areas.append(area)
        colors.append(roi.get("color", "#aaaaaa"))
        classes.append(roi.get("cls", ""))

    if not names:
        return _make_placeholder_fig("ROI polygons have insufficient points.")

    # Sort by class then name
    order = sorted(range(len(names)), key=lambda i: (classes[i], names[i]))
    names  = [names[i]   for i in order]
    areas  = [areas[i]   for i in order]
    colors = [colors[i]  for i in order]
    classes = [classes[i] for i in order]

    # Build bar chart grouped by class
    fig = go.Figure()
    seen_cls = []
    for cls in classes:
        if cls not in seen_cls:
            seen_cls.append(cls)

    for cls in seen_cls:
        idx = [i for i, c in enumerate(classes) if c == cls]
        fig.add_trace(go.Bar(
            x=[names[i] for i in idx],
            y=[areas[i] for i in idx],
            name=cls or "ROI",
            marker_color=[colors[i] for i in idx],
            text=[f"{areas[i]:,.0f} µm²" for i in idx],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:,.0f} µm²<extra>" + (cls or "ROI") + "</extra>",
        ))

    fig.update_layout(
        paper_bgcolor=core.CARD_BG, plot_bgcolor=core.PLOT_BG,
        font=dict(color=core.TEXT),
        margin=dict(l=60, r=20, t=40, b=80),
        xaxis=dict(title="ROI", gridcolor=core.BORDER, linecolor=core.BORDER, color=core.TEXT,
                   tickangle=-30),
        yaxis=dict(title="Area (µm²)", gridcolor=core.BORDER, linecolor=core.BORDER, color=core.TEXT),
        legend=dict(bgcolor=core.CARD_BG, bordercolor=core.BORDER, font=dict(color=core.TEXT, size=10)),
        title=dict(text="ROI Areas", font=dict(color=core.TEXT, size=12), x=0.5),
        barmode="group",
    )
    if width:
        fig.update_layout(width=width)
    if height:
        fig.update_layout(height=height)
    return fig


def make_umap_fig(color_by, method, gene, size, opacity,
                  baysor_active=False, proseg_active=False,
                  use_corrected: bool = False, cbar_range=None, alt_tool=None):
    for _prov in _umap_fig_providers:
        try:
            # alt_tool is the tool of the segmentation source ("xenium" for the Xenium cells)
            _pfig = _prov(color_by, method, gene, size, opacity,
                          baysor_active or alt_tool not in (None, "", "xenium", "proseg"),
                          proseg_active, use_corrected, cbar_range)
        except Exception as _perr:
            print(f"  [make_umap_fig] embedding provider failed: {_perr}", flush=True)
            _pfig = None
        if _pfig is not None:
            return _pfig
    mk = max(2, size * 1.2)

    # Check if reseg UMAP is available and the matching source is active.
    # Only use the stored result if its counts_mode matches the current use_corrected,
    # otherwise the coordinates were computed on a different expression matrix.
    with core._umap_reseg_lock:
        reseg_umap = core._umap_reseg_state["result"] if core._umap_reseg_state["status"] == "done" else None
        _reseg_umap_mode = core._umap_reseg_state.get("counts_mode", "original")
    _reseg_mode_matches = (use_corrected and _reseg_umap_mode == "corrected") or \
                          (not use_corrected and _reseg_umap_mode != "corrected")
    if reseg_umap is not None and not _reseg_mode_matches:
        reseg_umap = None  # force fallback to cells_df which has both umap_1/2 and split_umap_1/2

    # Determine active reseg source
    with core._proseg_lock:
        pres = core._proseg_state["result"] if proseg_active and core._proseg_state["result"] else None
    with core._baysor_lock:
        bres = core._baysor_state["result"] if baysor_active and not pres and core._baysor_state["result"] else None
    # a further tool (alt_tool, e.g. "imgseg"): its loaded run, drawn like a Baysor / Proseg one
    _ores = (core._seg_result(alt_tool) if alt_tool not in (None, "xenium", "baysor", "proseg")
             and not (pres or bres) else None)
    alt_res = pres or bres or _ores
    _alt_tool = "proseg" if pres else "baysor" if bres else alt_tool if _ores else "xenium"

    # Fall back to UMAP columns embedded in cells_df (loaded from zarr) when no explicit reseg UMAP.
    # When SPLIT-corrected counts are active, prefer split_umap_1/2 if available.
    # A stored "Rerun UMAP" result belongs to the run it was computed on: ignore it
    # for any other run, and keep only cells still present (subset()).
    if alt_res is not None and reseg_umap is not None:
        with core._umap_reseg_lock:
            _reseg_owner = core._umap_reseg_state.get("owner")
        if _reseg_owner is None or _reseg_owner != alt_res.get("out_dir"):
            reseg_umap = None
        else:
            _keep = reseg_umap.index.astype(str).isin(alt_res["cells_df"].index.astype(str))
            if not _keep.all():
                reseg_umap = reseg_umap[_keep] if _keep.any() else None

    if alt_res is not None and reseg_umap is None:
        _bdf = alt_res["cells_df"]
        if use_corrected and "split_umap_1" in _bdf.columns and _bdf["split_umap_1"].notna().any():
            reseg_umap = _bdf.dropna(subset=["split_umap_1", "split_umap_2"])
        elif "umap_1" in _bdf.columns and _bdf["umap_1"].notna().any():
            reseg_umap = _bdf.dropna(subset=["umap_1", "umap_2"])

    if alt_res is not None and reseg_umap is not None:
        # Reseg UMAP: color by whatever is available
        udf = reseg_umap
        # Use SPLIT UMAP coordinates when corrected counts are active and available.
        # The stored reseg UMAP result may only have umap_1/2 if it was computed without
        # corrected counts — fall back to cells_df for split_umap coords in that case.
        _bdf = alt_res["cells_df"]
        print(f"  [make_umap_fig reseg] use_corrected={use_corrected}, "
              f"split_notna={_bdf['split_umap_1'].notna().sum() if 'split_umap_1' in _bdf.columns else 0}/{len(_bdf)}", flush=True)
        if use_corrected and "split_umap_1" in udf.columns and udf["split_umap_1"].notna().any():
            xu = udf["split_umap_1"].values
            yu = udf["split_umap_2"].values
            print(f"  [make_umap_fig reseg] → using udf split_umap (corrected), xu range={xu.min():.2f}..{xu.max():.2f}", flush=True)
        elif use_corrected and "split_umap_1" in _bdf.columns and _bdf["split_umap_1"].notna().any():
            udf = _bdf.dropna(subset=["split_umap_1", "split_umap_2"])
            xu = udf["split_umap_1"].values
            yu = udf["split_umap_2"].values
            print(f"  [make_umap_fig reseg] → using bdf split_umap (corrected fallback), xu range={xu.min():.2f}..{xu.max():.2f}", flush=True)
        else:
            xu = udf["umap_1"].values
            yu = udf["umap_2"].values
            print(f"  [make_umap_fig reseg] → using umap_1/2 (original), xu range={xu.min():.2f}..{xu.max():.2f}", flush=True)
        cell_ids     = udf.index.tolist()
        cell_ids_str = [str(c) for c in cell_ids]
        _ids_arr     = np.asarray(cell_ids_str)
        # udf can hold fewer cells than cells_df (cells without UMAP coords, SPLIT-
        # dropped cells): map per-cell arrays of the full cells_df onto udf rows.
        _upos = pd.Index(_bdf.index.astype(str)).get_indexer(pd.Index(cell_ids_str)) \
            if _bdf.index.is_unique else np.arange(len(cell_ids_str))

        def _al(a):
            a = np.asarray(a)
            if len(a) == len(_upos) and (_upos == np.arange(len(_upos))).all():
                return a
            return np.where(_upos >= 0,
                            a.astype(float)[np.clip(_upos, 0, max(len(a) - 1, 0))], np.nan)

        if color_by.startswith("cell_type:"):
            _method    = color_by.split(":")[1]
            aligned    = annotation._reseg_cell_type_labels(_method, alt_res, cell_ids_str)
            if aligned is not None:
                cell_types = sorted(aligned.unique())
                cmap = {ct: core.CLUSTER_COLORS[i % len(core.CLUSTER_COLORS)] for i, ct in enumerate(cell_types)}
                traces = []
                for ct in cell_types:
                    mask = (aligned == ct).values
                    traces.append(go.Scattergl(
                        x=xu[mask], y=yu[mask], mode="markers",
                        marker=dict(size=mk, color=cmap[ct], opacity=opacity),
                        name=ct, showlegend=False,
                        text=_ids_arr[mask], customdata=_ids_arr[mask],
                        hovertemplate="<b>%{text}</b><extra>" + ct + "</extra>",
                    ))
            else:
                color_by = "transcript_counts"  # fall through

        if color_by.startswith("geneset:"):
            _gs = analysis._geneset_color(color_by, _alt_tool, use_corrected)
            if _gs is not None and len(_gs[0]) == len(_bdf):
                _mkw = analysis._geneset_marker(_al(_gs[0]), _gs[1], _gs[2], cbar_range, size=mk, opacity=opacity)
                _mkw["showscale"] = False
                traces = [go.Scattergl(
                    x=xu, y=yu, mode="markers", marker=_mkw, text=cell_ids, customdata=cell_ids,
                    hovertemplate="<b>%{text}</b><br>" + _gs[2] + ": %{marker.color:.2f}<extra></extra>",
                    name=_gs[2])]
            else:
                color_by = "transcript_counts"

        if color_by == "gene":
            vals = _get_expr_values(gene, alt_res, use_corrected=use_corrected)
            if vals is None and use_corrected:
                vals = _get_expr_values(gene, alt_res, use_corrected=False)   # see make_spatial_fig
            if vals is not None and len(vals) == len(_bdf):
                vals = _al(vals)
            if vals is not None:
                traces = [go.Scattergl(
                    x=xu, y=yu, mode="markers",
                    marker=dict(size=mk, color=vals, colorscale="Plasma",
                                opacity=opacity, showscale=False),
                    text=cell_ids, customdata=cell_ids,
                    hovertemplate="<b>%{text}</b><extra></extra>", name=gene,
                )]
            else:
                color_by = "transcript_counts"

        if color_by == "niche":
            bdf = alt_res["cells_df"]
            if "niche" in bdf.columns:
                _nv = spatial._niche_str(bdf["niche"].reindex(udf.index))
                _un = sorted(_nv.unique(), key=lambda v: (0, int(v)) if v.isdigit() else (1, v))
                _nm = {n: core.CLUSTER_COLORS[i % len(core.CLUSTER_COLORS)] for i, n in enumerate(_un)}
                traces = []
                for _n in _un:
                    _mask = (_nv == _n).values
                    traces.append(go.Scattergl(
                        x=xu[_mask], y=yu[_mask], mode="markers",
                        marker=dict(size=mk, color=_nm[_n], opacity=opacity),
                        name=_n, showlegend=False,
                        text=np.array(cell_ids)[_mask].tolist(),
                        customdata=np.array(cell_ids)[_mask].tolist(),
                        hovertemplate="<b>%{text}</b><extra>Niche " + _n + "</extra>",
                    ))
            else:
                color_by = "transcript_counts"

        if color_by == "cluster" and not color_by.startswith("cell_type:") and color_by != "gene":
            bdf = alt_res["cells_df"]
            _all_cluster_cols = reseg._reseg_cluster_cols(bdf.columns)
            if use_corrected:
                _cluster_col = next((c for c in _all_cluster_cols if "split" in c), None) or next(iter(_all_cluster_cols), None)
            else:
                _cluster_col = next((c for c in _all_cluster_cols if "split" not in c), None) or next(iter(_all_cluster_cols), None)
            _cluster_col = clustersets._cs_reseg_col(bdf.columns, method) or _cluster_col
            if _cluster_col is not None:
                _cvals = bdf[_cluster_col].reindex(udf.index).astype(str)
                _unique = sorted(_cvals.unique(), key=lambda v: (0, int(v)) if v.isdigit() else (1, v))
                _cmap = {cl: core.CLUSTER_COLORS[i % len(core.CLUSTER_COLORS)] for i, cl in enumerate(_unique)}
                traces = []
                for cl in _unique:
                    mask = (_cvals == cl).values
                    traces.append(go.Scattergl(
                        x=xu[mask], y=yu[mask], mode="markers",
                        marker=dict(size=mk, color=_cmap[cl], opacity=opacity),
                        name=cl, showlegend=False,
                        text=_ids_arr[mask], customdata=_ids_arr[mask],
                        hovertemplate="<b>%{text}</b><extra>Cluster " + cl + "</extra>",
                    ))
            else:
                color_by = "transcript_counts"

        if color_by.startswith("roi_"):
            # ROI class column holds names / NA, not numbers: categorical colours
            bdf = alt_res["cells_df"]
            if color_by in bdf.columns:
                _lab = bdf[color_by].reindex(udf.index).fillna("(none)").astype(str)
                _uv  = [v for v in sorted(_lab.unique()) if v != "(none)"]
                _rcm = {v: regions._roi_color(i) for i, v in enumerate(_uv)}
                _rcm["(none)"] = "#666666"
                traces = [go.Scattergl(
                    x=xu, y=yu, mode="markers",
                    marker=dict(size=mk, color=_lab.map(_rcm).values, opacity=opacity),
                    text=cell_ids, customdata=cell_ids,
                    hovertemplate="<b>%{text}</b><extra></extra>", name=color_by,
                )]
            else:
                color_by = "transcript_counts"

        if (not color_by.startswith(("cell_type:", "roi_", "geneset:"))
                and color_by not in ("gene", "cluster", "niche")):
            bdf = alt_res["cells_df"]
            if color_by in bdf.columns:
                raw = bdf["transcript_counts"].values if color_by == "transcript_counts" \
                      else bdf[color_by].values
                raw   = _al(raw)
                vals  = np.log1p(raw) if color_by == "transcript_counts" else raw.astype(float)
                label = core.QC_METRICS.get(color_by, color_by)
                if color_by == "transcript_counts":
                    label = "Transcript Counts (log1p)"
                cs    = core.COLORSCALES.get(color_by, "Viridis")
            else:
                vals  = np.log1p(_al(bdf["transcript_counts"].values))
                label = "Transcript Counts (log1p)"
                cs    = "Viridis"
            _ckw = {}
            if cbar_range:
                if cbar_range[0] is not None: _ckw["cmin"] = cbar_range[0]
                if cbar_range[1] is not None: _ckw["cmax"] = cbar_range[1]
            traces = [go.Scattergl(
                x=xu, y=yu, mode="markers",
                marker=dict(size=mk, color=vals, colorscale=cs,
                            opacity=opacity, showscale=False, **_ckw),
                text=cell_ids, customdata=cell_ids,
                hovertemplate="<b>%{text}</b><br>" + label + ": %{marker.color:.2f}<extra></extra>",
                name=label,
            )]

        title = f"UMAP [{core._SEG_TOOL_LABELS.get(_alt_tool, _alt_tool)}]"
    else:
        # Original Xenium UMAP — switch to SPLIT UMAP coordinates when corrected counts active
        _has_split_umap = (use_corrected
                           and "split_umap_1" in core.DATA["df"].columns
                           and core.DATA["df"]["split_umap_1"].notna().any())
        print(f"  [UMAP] use_corrected={use_corrected}, "
              f"has_split={'split_umap_1' in core.DATA['df'].columns}, "
              f"using={'split' if _has_split_umap else 'original'} coords, "
              f"n_cells={len(core.DATA['df'])}", flush=True)
        if _has_split_umap:
            if "split_df" not in dataview._umap_df_cache:
                dataview._umap_df_cache["split_df"] = core.DATA["df"].dropna(subset=["split_umap_1", "split_umap_2"])
            df = dataview._umap_df_cache["split_df"]
            xu = df["split_umap_1"].values
            yu = df["split_umap_2"].values
        else:
            if "df" not in dataview._umap_df_cache:
                dataview._umap_df_cache["df"] = core.DATA["df"].dropna(subset=["umap_1", "umap_2"])
            df = dataview._umap_df_cache["df"]
            xu = df["umap_1"].values
            yu = df["umap_2"].values

        # Missing columns: niche → cluster → transcript counts, as in make_spatial_fig
        # (no cluster sets, a reseg source selected but not loaded, a QC column absent
        # from an exported store). ROI columns are handled in their own branch.
        if color_by == "niche" and "niche" not in df.columns:
            color_by = "cluster"
        if color_by == "cluster" and dataview.cluster_col(method) not in df.columns:
            color_by = "transcript_counts"
        _gs = None
        if color_by.startswith("geneset:"):
            # the UMAP shows the primary sample only: its part of the scores
            _gs = analysis._geneset_color(color_by, "xenium", use_corrected)
            if _gs is None or len(_gs[0]) < len(core.DATA["df"]):
                _gs, color_by = None, "transcript_counts"
        if (color_by not in ("cluster", "gene", "niche")
                and not color_by.startswith(("cell_type:", "roi_", "geneset:"))
                and color_by not in df.columns):
            color_by = "transcript_counts"
        _gene_vals = None
        if color_by == "gene":       # a gene this dataset does not have: see make_spatial_fig
            _gene_vals = dataview._xenium_gene_values(gene, use_corrected)
            if _gene_vals is None or len(_gene_vals) != len(core.DATA["df"]):
                color_by = "transcript_counts"

        if color_by == "niche":
            if "niche" in df.columns:
                _nv = spatial._niche_str(df["niche"])
                _un = sorted(_nv.unique(), key=lambda v: (0, int(v)) if v.isdigit() else (1, v))
                _nm = {n: core.CLUSTER_COLORS[i % len(core.CLUSTER_COLORS)] for i, n in enumerate(_un)}
                traces = []
                for _n in _un:
                    _mask = (_nv == _n).values
                    traces.append(go.Scattergl(
                        x=xu[_mask], y=yu[_mask], mode="markers",
                        marker=dict(size=mk, color=_nm[_n], opacity=opacity),
                        name=_n, showlegend=False,
                        text=df.index[_mask].tolist(),
                        customdata=df.index[_mask].tolist(),
                        hovertemplate="<b>%{text}</b><extra>Niche " + _n + "</extra>",
                    ))
            else:
                color_by = "cluster"
                traces = _categorical_traces(xu, yu, df, method, mk, opacity, mode="umap")
        elif color_by == "cluster":
            traces = _categorical_traces(xu, yu, df, method, mk, opacity, mode="umap")
        elif color_by.startswith("cell_type:"):
            _method = color_by.split(":")[1]
            traces = _cell_type_traces(xu, yu, df, mk, opacity, mode="umap",
                                       labels_key=f"labels_{_method}")
        elif _gs is not None:
            _mkw = analysis._geneset_marker(_gs[0][:len(core.DATA["df"])][core.DATA["df"].index.get_indexer(df.index)],
                                   _gs[1], _gs[2], cbar_range, size=mk, opacity=opacity)
            _mkw["showscale"] = False
            traces = [go.Scattergl(
                x=xu, y=yu, mode="markers", marker=_mkw,
                text=df.index.tolist(), customdata=df.index.tolist(),
                hovertemplate="<b>%{text}</b><br>" + _gs[2] + ": %{marker.color:.2f}<extra></extra>",
                name=_gs[2])]
        elif color_by == "gene":
            # df may be a subset of DATA["df"] (cells with valid UMAP coords).
            # Use positional index into the full DATA["df"] to align expression values.
            _df_idx = core.DATA["df"].index.get_indexer(df.index)
            vals = _gene_vals[_df_idx]
            _ckw = {}
            if cbar_range:
                if cbar_range[0] is not None: _ckw["cmin"] = cbar_range[0]
                if cbar_range[1] is not None: _ckw["cmax"] = cbar_range[1]
            traces = [go.Scattergl(
                x=xu, y=yu, mode="markers",
                marker=dict(size=mk, color=vals, colorscale="Plasma",
                            opacity=opacity, showscale=False, **_ckw),
                text=df.index.tolist(), customdata=df.index.tolist(),
                hovertemplate="<b>%{text}</b><extra></extra>", name=gene,
            )]
        elif color_by.startswith("roi_"):
            # ROI class column holds names / NA, not numbers: categorical colours.
            # Read the live frame; the cached dropna() copy may predate the column.
            _lab = (core.DATA["df"][color_by].reindex(df.index) if color_by in core.DATA["df"].columns
                    else pd.Series(pd.NA, index=df.index, dtype=object)
                    ).fillna("(none)").astype(str)
            _uv  = [v for v in sorted(_lab.unique()) if v != "(none)"]
            _rcm = {v: regions._roi_color(i) for i, v in enumerate(_uv)}
            _rcm["(none)"] = "#666666"
            traces = [go.Scattergl(
                x=xu, y=yu, mode="markers",
                marker=dict(size=mk, color=_lab.map(_rcm).values, opacity=opacity),
                text=df.index.tolist(), customdata=df.index.tolist(),
                hovertemplate="<b>%{text}</b><extra></extra>", name=color_by,
            )]
        else:
            label = core.QC_METRICS.get(color_by, color_by)
            if color_by not in df.columns:
                vals = np.zeros(len(df))
            else:
                vals = np.log1p(df[color_by].values) if color_by == "transcript_counts" else df[color_by].values
            if color_by == "transcript_counts":
                label = "Transcript Counts (log1p)"
            _ckw = {}
            if cbar_range:
                if cbar_range[0] is not None: _ckw["cmin"] = cbar_range[0]
                if cbar_range[1] is not None: _ckw["cmax"] = cbar_range[1]
            traces = [go.Scattergl(
                x=xu, y=yu, mode="markers",
                marker=dict(size=mk, color=vals,
                            colorscale=core.COLORSCALES.get(color_by, "Viridis"),
                            opacity=opacity, showscale=False, **_ckw),
                text=df.index.tolist(), customdata=df.index.tolist(),
                hovertemplate="<b>%{text}</b><br>" + label + ": %{marker.color:.2f}<extra></extra>",
                name=label,
            )]
        title = "UMAP"

    fig = go.Figure(data=traces)
    fig.update_layout(
        **_base_layout(title, "UMAP 1", "UMAP 2"),
        showlegend=False, uirevision="umap",
    )
    return fig
