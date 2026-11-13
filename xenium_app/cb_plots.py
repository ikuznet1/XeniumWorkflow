"""Callbacks: main spatial / UMAP figures, clicks, distance measurement, cell info."""

import os
from dash import no_update, Patch
import numpy as np
import pandas as pd
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go

from . import core, ui, dataview, morph, figures, cellnest, stats, app_layout


app_layout.app.clientside_callback(
    """function(colorBy) {
        var show = {"marginBottom": "14px"};
        var hide = {"marginBottom": "14px", "display": "none"};
        var cbShow = {};
        var cbHide = {"display": "none"};
        // Continuous color modes that have a colorbar
        var continuous = ["transcript_counts", "cell_area", "nucleus_area", "total_counts", "gene"];
        var isCont = continuous.indexOf(colorBy) >= 0;
        if (colorBy && colorBy.indexOf("geneset:") === 0) return [hide, hide, cbShow];
        if (colorBy === "cluster") return [show, hide, cbHide];
        if (colorBy === "gene") return [hide, show, cbShow];
        if (isCont) return [hide, hide, cbShow];
        return [hide, hide, cbHide];
    }""",
    Output("cluster-ctrl",  "style"),
    Output("gene-ctrl",     "style"),
    Output("colorbar-ctrl", "style"),
    Input("color-by", "value"),
)


@app_layout.app.callback(
    Output("morph-controls", "style"),
    Output("morph-hires-poll", "disabled", allow_duplicate=True),
    Input("morph-enable", "value"),
    Input("renderer-mode", "value"),
    prevent_initial_call=True,
)
def toggle_morph_controls(enabled, renderer_mode=None):
    active = "show" in (enabled or [])
    # (the hires poll feeds the Plotly figure only: the WebGL view reads the image itself)
    return ({} if active else {"display": "none"}), (not active or renderer_mode == "webgl")


app_layout.app.clientside_callback(
    """function(n) { return [null, null]; }""",
    Output("colorbar-min", "value"),
    Output("colorbar-max", "value"),
    Input("colorbar-reset-btn", "n_clicks"),
    prevent_initial_call=True,
)


@app_layout.app.callback(
    Output("spatial-relayout", "data"),
    Input("spatial-plot", "relayoutData"),
    State("spatial-relayout", "data"),
    prevent_initial_call=True,
)
def store_relayout(relayout_data, current):
    # Nothing to store -> no_update, not the unchanged value: writing the store re-runs
    # update_plots, and Dash drops the answer of a figure request that is still in flight
    # when a newer one for the same output starts. A graph that has just become visible
    # (renderer switched to Plotly, e.g. by the automatic fallback from WebGL) emits such
    # an autosize-only event while its figure is being built, and stayed on "Loading…".
    if not relayout_data:
        return no_update
    # If the event contains width/height, it's a container resize — ignore range
    # changes to prevent progressive zoom-out when UMAP panel opens/closes.
    if "width" in relayout_data or "height" in relayout_data:
        return no_update
    merged = dict(current or {})
    for ax in ("xaxis", "yaxis"):
        # Autoscale / double-click: the stored zoom is no longer what is on screen
        if relayout_data.get(ax + ".autorange") is True:
            for _k in (ax + ".range[0]", ax + ".range[1]", ax + ".range"):
                merged.pop(_k, None)
        # Reset-axes emits the list form; consumers read only the indexed keys
        rng = relayout_data.get(ax + ".range")
        if (isinstance(rng, (list, tuple)) and len(rng) == 2
                and rng[0] is not None and rng[1] is not None):
            merged[ax + ".range[0]"], merged[ax + ".range[1]"] = rng[0], rng[1]
    # Ignore autosize/autorange signals that would override explicit ranges
    ignore = {"autosize", "xaxis.autorange", "yaxis.autorange",
              "xaxis.showspikes", "yaxis.showspikes", "xaxis.range", "yaxis.range"}
    for k, v in relayout_data.items():
        if k not in ignore:
            merged[k] = v
    return no_update if merged == (current or {}) else merged


@app_layout.app.callback(
    Output("spatial-plot", "figure"),
    Output("umap-plot",    "figure"),
    Input("color-by",           "value"),
    Input("cluster-method",     "value"),
    Input("gene-selector",      "value"),
    Input("point-size",         "value"),
    Input("point-opacity",      "value"),
    Input("boundary-toggles",   "value"),
    Input("annot-done",         "data"),
    Input("morph-enable",       "value"),
    Input("morph-zlevel",       "value"),
    Input("morph-channels",     "value"),
    Input("morph-brightness",   "value"),
    Input("morph-opacity",      "value"),
    Input("spatial-relayout",   "data"),
    Input("baysor-done",        "data"),
    Input("proseg-done",        "data"),
    Input("seg-source",         "value"),
    Input("spage-done",         "data"),
    Input("subset-version",     "data"),
    Input("startup-trigger",        "n_intervals"),
    Input("dataset-version",        "data"),
    Input("extra-datasets-version", "data"),
    Input("split-done",             "data"),
    Input("roi-done",               "data"),
    Input("counts-mode-store",      "data"),
    Input("niche-done",             "data"),
    Input("colorbar-min",           "value"),
    Input("colorbar-max",           "value"),
    Input("renderer-mode",          "value"),
    Input("density-store",          "data"),
    Input("geneset-version",        "data"),
    State("cellnest-lr-pair",        "value"),
    State("cellnest-top-n",          "value"),
    State("cellnest-edge-width",     "value"),
    State("cellnest-overlay-enable", "value"),
    State("cellnest-options",        "value"),
    State("cellnest-color-by",       "value"),
    State("cellnest-filter-ligand",   "value"),
    State("cellnest-filter-receptor", "value"),
    State("cellnest-filter-sender",   "value"),
    State("cellnest-filter-receiver", "value"),
    State("cellnest-annot-col",       "value"),
    State("sdata-show-roi",           "value"),
    State("sdata-show-patches",       "value"),
    State("measure-store",            "data"),
)
def update_plots(color_by, method, gene, size, opacity, boundary_toggles,
                 _annot_done, morph_enable, morph_zlevel, morph_channels,
                 morph_brightness, morph_opacity, relayout,
                 _baysor_done, _proseg_done, seg_source,
                 _spage_done, _subset_ver, _startup, _ds_version, _extra_ds_version,
                 _split_done, _roi_done, counts_mode, _niche_done,
                 cbar_min, cbar_max, renderer_mode, density_spec, _geneset_ver,
                 cn_lr_pair, cn_top_n, cn_edge_width, cn_overlay_enable, cn_options,
                 cn_color_by, cn_f_ligand, cn_f_receptor, cn_f_sender, cn_f_receiver,
                 cn_annot_col, sdata_show_roi=None, sdata_show_patches=None,
                 measure_state=None):
    # If only the viewport changed, avoid full figure rebuild
    triggered = callback_context.triggered_id
    boundaries_active = bool(boundary_toggles)
    morph_active      = "show" in (morph_enable or [])
    sdata_roi_on      = "yes" in (sdata_show_roi     or [])
    sdata_patches_on  = "yes" in (sdata_show_patches or [])
    # transcript-density heat layer: follows the viewport like the boundaries do
    density_active    = bool(density_spec and density_spec.get("on"))
    # WebGL view: the browser draws the spatial and UMAP views itself from /viewer/* and
    # /umap/viewer/*, so neither Plotly figure is built. Switching back to Plotly triggers
    # this callback (renderer-mode is an Input) and rebuilds both.
    if renderer_mode == "webgl":
        return no_update, no_update
    # Too many cells for a Plotly figure: say so instead of building it (minutes of work
    # and a figure the browser cannot hold)
    _too_large = dataview._plotly_too_large(sum(len(t["df"]) for t in stats._view_tables(seg_source or "xenium")))
    if _too_large:
        if {t["prop_id"].split(".")[0] for t in callback_context.triggered} == {"spatial-relayout"}:
            return no_update, no_update
        _msg = _too_large.replace(": a figure", ":<br>a figure").replace(". Use", ".<br>Use").replace(
            ". XENIUM", ".<br>XENIUM")
        return figures._make_placeholder_fig(_msg), figures._make_placeholder_fig(
            "The UMAP of this many cells is drawn in the WebGL view.")

    # (sole trigger only: Load Sample resets the stored viewport together with
    # dataset-version, and that must rebuild the figure)
    if {t["prop_id"].split(".")[0] for t in callback_context.triggered} == {"spatial-relayout"}:
        # RCTD pies cover only the cells in view: refresh them (layout shapes only)
        _pies = None
        if not boundaries_active and core._seg_tool(seg_source or "xenium") == "xenium":
            _pies = figures._rctd_pie_shapes_for_view(color_by, relayout or {})
        if _pies is not None and not morph_active and not density_active:
            if not _pies and not figures._spatial_base_shapes:
                return no_update, no_update   # no pies before, none now
            figures._spatial_base_shapes[:] = _pies
            patched = Patch()
            patched["layout"]["shapes"] = _pies + figures._spatial_overlay_shapes(
                sdata_roi_on, sdata_patches_on, seg_source)
            return patched, no_update
        if not boundaries_active and not morph_active and not density_active:
            # Nothing overlay-related → skip entirely
            return no_update, no_update
        if morph_active and not boundaries_active and not density_active:
            # Only morphology image needs updating → Patch just layout.images
            morph_image, _ = morph.make_morphology_overlay(
                core.DATA["data_dir"], relayout or {},
                morph_zlevel or 0, morph_channels or [],
                morph_brightness or 2,
                morph_opacity if morph_opacity is not None else 0.85,
            )
            if morph_image is None:
                # Viewport too large or no valid range yet — preserve existing image,
                # but still refresh the RCTD pies for the new viewport
                if _pies is not None and (_pies or figures._spatial_base_shapes):
                    figures._spatial_base_shapes[:] = _pies
                    patched = Patch()
                    patched["layout"]["shapes"] = _pies + figures._spatial_overlay_shapes(
                        sdata_roi_on, sdata_patches_on, seg_source)
                    return patched, no_update
                return no_update, no_update
            patched = Patch()
            patched["layout"]["images"] = [morph_image]
            if _pies is not None and (_pies or figures._spatial_base_shapes):
                figures._spatial_base_shapes[:] = _pies
                patched["layout"]["shapes"] = _pies + figures._spatial_overlay_shapes(
                    sdata_roi_on, sdata_patches_on, seg_source)
            return patched, no_update

    if triggered in ("morph-brightness", "morph-opacity", "morph-channels",
                     "morph-zlevel") and morph_active:
        # Only morph settings changed → Patch image only, skip scatter rebuild
        morph_image, _ = morph.make_morphology_overlay(
            core.DATA["data_dir"], relayout or {},
            morph_zlevel or 0, morph_channels or [],
            morph_brightness or 2,
            morph_opacity if morph_opacity is not None else 0.85,
        )
        if morph_image is None:
            return no_update, no_update
        patched = Patch()
        patched["layout"]["images"] = [morph_image]
        return patched, no_update

    color_by         = color_by or "cluster"
    _cm              = core.DATA.get("cluster_methods") or []
    method           = method   or (_cm[0] if _cm else None)
    gene             = gene     or dataview._default_gene
    size             = size     or 2
    opacity          = opacity  or 0.85
    boundary_toggles = boundary_toggles or []
    seg_source  = seg_source or "xenium"
    baysor_on   = (core._seg_tool(seg_source) == "baysor")
    proseg_on   = (core._seg_tool(seg_source) == "proseg")
    alt_tool    = core._seg_tool(seg_source)
    if alt_tool == "xenium":
        # After Load Sample the dropdown State can still hold the previous dataset's
        # selections for one callback round; fall back to this dataset's defaults.
        if method is not None and method not in _cm:
            method = _cm[0] if _cm else None
        # (a gene this dataset does not have is not replaced here: the figure builders
        # draw transcript counts for it, as the WebGL view and the reseg sources do)
    if color_by == "cluster" and method is None and alt_tool == "xenium":
        color_by = "transcript_counts"   # dataset has no Xenium cluster sets

    # Morphology image overlay
    morph_image  = None
    morph_title  = ""
    if "show" in (morph_enable or []):
        morph_image, morph_title = morph.make_morphology_overlay(
            core.DATA["data_dir"],
            relayout or {},
            morph_zlevel or 0,
            morph_channels or [],
            morph_brightness or 2,
            morph_opacity if morph_opacity is not None else 0.85,
        )

    use_corr = (counts_mode == "corrected")
    cbar_range = (cbar_min, cbar_max) if cbar_min is not None or cbar_max is not None else None

    # When counts mode, SPLIT status, or subset changes, clear cached UMAP
    if triggered in ("counts-mode-store", "split-done", "subset-version",
                     "dataset-version", "niche-done"):
        dataview._umap_df_cache.clear()
        dataview._umap_fig_cache.clear()
        if triggered == "subset-version":
            print(f"  [update_plots] subset-version triggered, _subset_ver={_subset_ver}", flush=True)

    # UMAP cache: skip rebuild when only spatial-related inputs changed
    _umap_key = (color_by, method, gene, size, opacity, baysor_on, proseg_on, alt_tool, use_corr,
                 _annot_done, _spage_done, _ds_version, _extra_ds_version,
                 _baysor_done, _proseg_done, _split_done, _subset_ver, _niche_done,
                 _roi_done, cbar_min, cbar_max, _geneset_ver)
    _cache_hit = _umap_key in dataview._umap_fig_cache
    if not _cache_hit:
        print(f"  [UMAP cache] miss — use_corr={use_corr}, triggered={triggered}", flush=True)
    if _cache_hit:
        umap_fig = dataview._umap_fig_cache[_umap_key]
    else:
        umap_fig = figures.make_umap_fig(color_by, method, gene, size, opacity,
                                 baysor_active=baysor_on, proseg_active=proseg_on,
                                 use_corrected=use_corr, cbar_range=cbar_range, alt_tool=alt_tool)
        dataview._umap_fig_cache.clear()  # keep only latest entry
        dataview._umap_fig_cache[_umap_key] = umap_fig

    spatial_fig = figures.make_spatial_fig(color_by, method, gene, size, opacity,
                                   boundary_toggles, relayout,
                                   morph_image=morph_image, extra_title=morph_title,
                                   baysor_active=baysor_on, proseg_active=proseg_on,
                                   use_corrected=use_corr, cbar_range=cbar_range,
                                   sdata_roi_on=sdata_roi_on, sdata_patches_on=sdata_patches_on,
                                   density=density_spec if density_active else None, alt_tool=alt_tool)
    # Re-add CellNEST overlay so it persists through pan/zoom rebuilds
    with core._cellnest_lock:
        _cn_status = core._cellnest_state.get("status")
    if _cn_status == "done" and cn_lr_pair and "on" in (cn_overlay_enable or []):
        _cn_opts = cn_options or []
        for _t in cellnest._cellnest_overlay_traces(cn_lr_pair, cn_top_n or 100,
                                           edge_width=cn_edge_width or 1,
                                           color_by=cn_color_by or "lr_pair",
                                           options=_cn_opts,
                                           f_ligand=cn_f_ligand, f_receptor=cn_f_receptor,
                                           f_sender=cn_f_sender, f_receiver=cn_f_receiver,
                                           annot_col=cn_annot_col):
            spatial_fig.add_trace(_t)
    # Re-add the ruler drawing (draw_measurement only runs when measure-store changes)
    for _t in _measure_traces(measure_state):
        spatial_fig.add_trace(_t)
    return spatial_fig, umap_fig


@app_layout.app.callback(
    Output("selected-cell", "data"),
    Input("spatial-plot", "clickData"),
    Input("umap-plot",    "clickData"),
    State("measure-store", "data"),
    prevent_initial_call=True,
)
def capture_click(spatial_click, umap_click, measure_state):
    # Don't select cells when the ruler is active
    if measure_state and measure_state.get("active"):
        return no_update
    triggered = callback_context.triggered_id
    click = spatial_click if triggered == "spatial-plot" else umap_click
    if not click or not click.get("points"):
        return None
    pt = click["points"][0]
    cell_id = pt.get("customdata") or pt.get("text")
    if triggered == "spatial-plot" and dataview.EXTRA_DATASETS and cell_id is not None:
        # Side-by-side samples: ids are unique within a sample only, so the sample is
        # taken from where the click landed (the point's x carries the sample's offset).
        try:
            px = float(pt.get("x"))
        except (TypeError, ValueError):
            px = None
        if px is not None:
            _p = core.DATA["df"]
            _hit = np.flatnonzero(_p.index.astype(str) == str(cell_id))
            if not (len(_hit) and np.any(np.abs(_p["x_centroid"].values[_hit] - px) < 1e-3)):
                for si, eds in enumerate(list(dataview.EXTRA_DATASETS), start=1):
                    _e = eds["df"]
                    _hit = np.flatnonzero(_e.index.astype(str) == str(cell_id))
                    if len(_hit) and np.any(np.abs(_e["x_centroid"].values[_hit]
                                                   + float(eds["x_offset"]) - px) < 1e-3):
                        return {"id": cell_id, "sample": si}
    return cell_id


# ─── Distance measurement ────────────────────────────────────────────────────

def _measure_traces(store) -> list:
    """Ruler traces (end markers, plus the dotted line once two points are set) for the
    spatial figure; empty when the ruler is off or has no points."""
    if not store or not store.get("active"):
        return []
    points = store.get("points") or []
    if not points:
        return []
    traces = [{
        "type": "scattergl", "mode": "markers",
        "x": [p[0] for p in points],
        "y": [p[1] for p in points],
        "marker": {"size": 10, "color": "#f0c040",
                   "symbol": "cross", "line": {"width": 2, "color": "#f0c040"}},
        "name": "measure_points", "hoverinfo": "skip", "showlegend": False,
    }]
    if len(points) == 2:
        (x1, y1), (x2, y2) = points
        traces.append({
            "type": "scattergl", "mode": "lines",
            "x": [x1, x2], "y": [y1, y2],
            "line": {"color": "#f0c040", "width": 2, "dash": "dot"},
            "name": "measure_line", "hoverinfo": "skip", "showlegend": False,
        })
    return traces


@app_layout.app.callback(
    Output("measure-store", "data"),
    Output("measure-btn",   "style"),
    Input("measure-btn", "n_clicks"),
    State("measure-store", "data"),
    prevent_initial_call=True,
)
def toggle_measure(n_clicks, store):
    active = not store.get("active", False)
    btn_style = {
        "padding": "4px 7px", "fontSize": "14px",
        "border": "1px solid #58a6ff" if active else f"1px solid {core.BORDER}",
        "borderRadius": "4px", "cursor": "pointer", "lineHeight": "1",
        "backgroundColor": "#1f3b5a" if active else core.CARD_BG,
        "color": "#58a6ff" if active else core.TEXT,
    }
    return {"active": active, "points": []}, btn_style


@app_layout.app.callback(
    Output("measure-store", "data", allow_duplicate=True),
    Input("spatial-plot", "clickData"),
    State("measure-store", "data"),
    prevent_initial_call=True,
)
def measure_capture_click(click_data, store):
    if not store or not store.get("active"):
        return no_update
    if not click_data or not click_data.get("points"):
        return no_update
    pt = click_data["points"][0]
    x, y = pt.get("x"), pt.get("y")
    if x is None or y is None:
        return no_update
    points = store.get("points", [])
    if len(points) >= 2:
        # Start new measurement from this click
        points = [[x, y]]
    else:
        points = points + [[x, y]]
    return {**store, "points": points}


@app_layout.app.callback(
    Output("spatial-plot", "figure", allow_duplicate=True),
    Output("measure-dist-display", "children"),
    Output("measure-dist-display", "style"),
    Input("measure-store", "data"),
    State("spatial-plot", "figure"),
    State("renderer-mode", "value"),
    prevent_initial_call=True,
)
def draw_measurement(store, fig, renderer_mode=None):
    _dist_hidden = {"fontSize": "11px", "color": core.TEXT,
                    "backgroundColor": "rgba(13,17,23,0.85)",
                    "padding": "3px 6px", "borderRadius": "4px",
                    "marginLeft": "4px", "display": "none"}
    _dist_visible = {**_dist_hidden, "display": "inline"}

    if fig is None:
        return no_update, "", _dist_hidden

    if renderer_mode == "webgl":
        # The WebGL view draws the ruler itself; only the distance label comes from here
        # (update_plots re-adds the traces when the Plotly figure is rebuilt).
        _, _label, _style = draw_measurement(store, {"data": []})
        return no_update, _label, _style

    # Remove previous measure traces
    fig["data"] = [t for t in fig.get("data", [])
                   if not str(t.get("name", "")).startswith("measure_")]

    if not store or not store.get("active"):
        return fig, "", _dist_hidden

    points = store.get("points", [])
    if not points:
        return fig, "", _dist_hidden

    fig["data"].extend(_measure_traces(store))

    if len(points) == 2:
        x1, y1 = points[0]
        x2, y2 = points[1]
        # y coordinates in plot space are negated µm; distance is same either way
        dist_um = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        label = f"{dist_um:.1f} µm"
        if dist_um >= 1000:
            label += f"  ({dist_um / 1000:.2f} mm)"
        return fig, label, _dist_visible

    return fig, "click second point…", _dist_visible


@app_layout.app.callback(
    Output("cell-info-panel", "children"),
    Input("selected-cell", "data"),
    State("cluster-method", "value"),
    prevent_initial_call=True,
)
def show_cell_info(cell_id, method):
    if not cell_id:
        return html.Div("Click a cell to see details",
                        style={"color": core.MUTED, "fontSize": "13px", "fontStyle": "italic"})

    # A cell of a sample shown beside the primary one arrives as {"id", "sample"} (sample 1
    # = first extra sample): cell ids are unique within a sample only.
    ds, sample_tag, cell_pos = core.DATA, "", None
    if isinstance(cell_id, dict):
        _sample, cell_id = cell_id.get("sample"), cell_id.get("id")
        _extras = list(dataview.EXTRA_DATASETS)
        _eds = (_extras[_sample - 1] if isinstance(_sample, int) and 1 <= _sample <= len(_extras)
                else None)
        _pos = (np.flatnonzero(_eds["df"].index.astype(str) == str(cell_id))
                if _eds is not None else [])
        if not len(_pos):
            return html.Div(f"Cell '{cell_id}' not found (the samples shown have changed).",
                            style={"color": "#f85149"})
        ds, cell_pos = _eds, int(_pos[0])
        _name = os.path.basename((ds.get("data_dir") or "").rstrip("/"))
        sample_tag = f" [sample {_sample + 1}" + (f": {_name}]" if _name else "]")
    df = ds["df"]

    # ── Try Xenium cells first, then Proseg, then Baysor ─────────────────────
    if cell_pos is not None or cell_id in df.index:
        # Original Xenium cell (of the primary or of an extra sample)
        if cell_pos is None:
            cell_pos = df.index.get_loc(cell_id)
            if not isinstance(cell_pos, (int, np.integer)):      # duplicated id: first row
                cell_pos = int(np.flatnonzero(df.index == cell_id)[0])
        row    = df.iloc[cell_pos]
        _cm    = ds.get("cluster_methods") or []
        method = method if method in _cm else (_cm[0] if _cm else None)
        col    = dataview.cluster_col(method) if method else None
        clust  = "—"
        if col and col in row and not pd.isna(row[col]):
            try:
                clust = int(row[col])
            except (TypeError, ValueError):
                clust = row[col]

        expr_row_idx = ds["df_to_expr"][cell_pos]
        if expr_row_idx >= 0:
            raw       = ds["expr"][expr_row_idx, :].toarray().flatten()
            top_idx   = np.argsort(raw)[::-1][:12]
            top_genes = [(ds["gene_names"][i], int(raw[i])) for i in top_idx if raw[i] > 0]
        else:
            top_genes = []

        umap_1 = row.get("umap_1", float("nan"))
        umap_2 = row.get("umap_2", float("nan"))

        def _num(col, fmt):   # bundles built from GEO tables lack the area columns
            v = pd.to_numeric(row.get(col), errors="coerce")
            return fmt.format(v) if pd.notna(v) else "—"

        chips = [
            ui.info_chip("Cluster",     str(clust)),
            ui.info_chip("X",           f"{row['x_centroid']:.1f} µm"),
            ui.info_chip("Y",           f"{row['y_centroid']:.1f} µm"),
            ui.info_chip("Transcripts", _num("transcript_counts", "{:.0f}")),
            ui.info_chip("Cell Area",   _num("cell_area", "{:.0f} µm²")),
            ui.info_chip("Nuc Area",    _num("nucleus_area", "{:.0f} µm²")),
            ui.info_chip("UMAP 1",      f"{umap_1:.2f}" if not np.isnan(umap_1) else "—"),
            ui.info_chip("UMAP 2",      f"{umap_2:.2f}" if not np.isnan(umap_2) else "—"),
        ]
        source_tag = sample_tag

    else:
        # Try reseg sources (Proseg cell IDs are ints, Baysor are strings)
        row       = None
        alt_expr  = None
        source_tag = ""
        for src_name, state, lock in (
            ("Proseg", core._proseg_state, core._proseg_lock),
            ("Baysor", core._baysor_state, core._baysor_lock),
            ("Image segmentation", core._imgseg_state, core._imgseg_lock),
        ):
            with lock:
                res = state.get("result")
            if res is None:
                continue
            bdf = res["cells_df"]
            # Try direct match first, then type-coerced match
            key = cell_id
            if key not in bdf.index:
                try:
                    key = int(cell_id)
                except (ValueError, TypeError):
                    pass
            if key in bdf.index:
                row        = bdf.loc[key]
                alt_expr   = res.get("expr")
                source_tag = f" [{src_name}]"
                break

        if row is None:
            return html.Div(f"Cell '{cell_id}' not found.", style={"color": "#f85149"})

        # Top genes from reseg expression matrix
        top_genes = []
        if alt_expr is not None:
            # After SpaGE on a reseg result the matrix has extra imputed columns whose
            # names live only in res["gene_name_to_idx"], not in DATA["gene_names"].
            _gni = res.get("gene_name_to_idx") or core.DATA["gene_name_to_idx"]
            gene_names_list = sorted(_gni, key=_gni.__getitem__)
            _imp = set(res.get("imputed_genes") or [])
            # Find row index in reseg cells_df — always use integer position to avoid
            # boolean-array issues when cell IDs are non-unique (multi-patch Proseg runs)
            bdf = res["cells_df"]
            key_for_idx = key
            matches = np.where(bdf.index == key_for_idx)[0]
            if len(matches) == 0:
                alt_expr = None
            else:
                row_idx = int(matches[0])
                if row_idx >= alt_expr.shape[0]:
                    alt_expr = None
        if alt_expr is not None:
            raw     = alt_expr[row_idx, :].toarray().flatten()
            top_idx = np.argsort(raw)[::-1][:12]
            top_genes = [
                ((gene_names_list[i] + " [imp]", round(float(raw[i]), 2))
                 if gene_names_list[i] in _imp else (gene_names_list[i], int(raw[i])))
                for i in top_idx if raw[i] > 0 and i < len(gene_names_list)
            ]

        chips = [
            ui.info_chip("X",           f"{row['x_centroid']:.1f} µm"),
            ui.info_chip("Y",           f"{row['y_centroid']:.1f} µm"),
            ui.info_chip("Transcripts", str(int(row["transcript_counts"]))),
        ]
        if "cell_area" in row.index:
            chips.append(ui.info_chip("Cell Area", f"{row['cell_area']:.0f} µm²"))

    # ── Bar chart ─────────────────────────────────────────────────────────────
    bar_fig = go.Figure(go.Bar(
        x=[g for g, _ in top_genes],
        y=[v for _, v in top_genes],
        marker_color=core.ACCENT,
    ))
    bar_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=core.CARD_BG, plot_bgcolor=core.CARD_BG,
        margin=dict(l=40, r=10, t=5, b=55), height=150,
        xaxis=dict(tickfont=dict(size=10), color=core.TEXT),
        yaxis=dict(title="Counts", color=core.TEXT, tickfont=dict(size=10)),
        font=dict(color=core.TEXT),
    )

    return html.Div([
        html.Div([
            html.Span("Cell: ", style={"color": core.MUTED, "fontSize": "12px"}),
            html.Span(str(cell_id) + source_tag,
                      style={"color": core.ACCENT, "fontWeight": "700", "fontSize": "13px"}),
        ], style={"marginBottom": "8px"}),

        html.Div(chips, style={"display": "flex", "gap": "8px", "flexWrap": "wrap",
                               "marginBottom": "10px"}),

        html.Div("Top Expressed Genes",
                 style={"fontSize": "11px", "color": core.MUTED, "marginBottom": "4px"}),
        dcc.Graph(figure=bar_fig, config={"displayModeBar": False}),
    ])
