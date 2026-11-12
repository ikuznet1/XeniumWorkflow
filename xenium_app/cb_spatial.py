"""Callbacks: SPLIT, spatial niches, spatial analysis, ROI annotation."""

import os
import threading
from dash import no_update
import dash
from dash import Input, Output, State, callback_context, ALL

from . import core, jobs, dataview, annotation, split, spatial, figures, analysis, regions, app_layout


# ─── SPLIT correction callbacks ───────────────────────────────────────────────

@app_layout.app.callback(
    Output("split-modal", "is_open"),
    [Input("split-modal-open-btn", "n_clicks"),
     Input("split-modal-close-btn", "n_clicks")],
    [State("split-modal", "is_open")],
    prevent_initial_call=True,
)
def open_split_modal(open_clicks, close_clicks, is_open):
    if not open_clicks and not close_clicks:
        return is_open
    triggered = callback_context.triggered_id
    if triggered == "split-modal-open-btn" and open_clicks:
        return True
    return False


@app_layout.app.callback(
    [Output("split-modal-status", "children"),
     Output("split-poll", "disabled"),
     Output("split-run-btn", "disabled")],
    Input("split-run-btn", "n_clicks"),
    [State("split-rds-path", "value"),
     State("split-label-col", "value"),
     State("split-max-cores", "value"),
     State("split-min-umi", "value"),
     State("split-min-umi-sigma", "value"),
     State("split-purify-singlets", "value"),
     State("seg-source", "value")],
    prevent_initial_call=True,
)
def run_split(n_clicks, rds_path, label_col, max_cores, min_umi, min_umi_sigma, purify_singlets, seg_source):
    if not n_clicks:
        return no_update, True, False
    if not rds_path or not os.path.isfile(rds_path):
        return "⚠ RDS file not found.", True, False
    with core._split_lock:
        if core._split_state["status"] == "running":
            return "Already running…", False, True
        core._split_state["status"]  = "running"
        core._split_state["message"] = "Starting…"
        core._split_state["result"]  = None
    label_col  = (label_col or "Names").strip()
    max_cores  = int(max_cores or 4)
    min_umi       = int(min_umi or 10)
    min_umi_sigma = int(min_umi_sigma or 100)
    do_purify_singlets = bool(purify_singlets)
    jobs._job_start("split", "SPLIT correction", split._run_split_correction,
               args=(rds_path, label_col, max_cores, seg_source,
                     min_umi, min_umi_sigma, do_purify_singlets),
               state=core._split_state, lock=core._split_lock, copy_context=True)
    return "Starting SPLIT correction…", False, True


@app_layout.app.callback(
    [Output("split-status", "children"),
     Output("split-modal-status", "children", allow_duplicate=True),
     Output("split-modal", "is_open", allow_duplicate=True),
     Output("split-poll", "disabled", allow_duplicate=True),
     Output("split-run-btn", "disabled", allow_duplicate=True),
     Output("split-done", "data"),
     Output("counts-mode", "value", allow_duplicate=True),
     Output("dataset-version", "data", allow_duplicate=True),
     Output("color-by", "options", allow_duplicate=True)],
    Input("split-poll", "n_intervals"),
    [State("split-done", "data"),
     State("dataset-version", "data"),
     State("color-by", "options")],
    prevent_initial_call=True,
)
def poll_split(_, done_version, ds_version, current_color_options):
    with core._split_lock:
        status  = core._split_state["status"]
        message = core._split_state["message"]
    if status == "idle":
        return "", "", no_update, True, False, no_update, no_update, no_update, no_update
    if status == "running":
        return message, message, no_update, False, True, no_update, no_update, no_update, no_update
    if status == "done":
        # Enable cell_type:rctd in color-by dropdown since SPLIT stores RCTD labels
        with core._annot_lock:
            methods_with_labels = {
                m for m in core._ANNOT_METHODS
                if any(k.startswith(f"labels_{m}") and v is not None
                       for k, v in core._annot_state.items())
            }
        updated_options = []
        for opt in (current_color_options or []):
            val = opt.get("value", "")
            if val.startswith("cell_type:"):
                m = val.split(":")[1]
                updated_options.append({**opt, "disabled": m not in methods_with_labels})
            else:
                updated_options.append(opt)
        return (f"✓ {message}", f"✓ {message}", False, True, False,
                done_version + 1, "corrected", no_update, updated_options)
    if status == "error":
        return f"✗ {message}", f"✗ {message}", no_update, True, False, done_version, no_update, no_update, no_update
    return "", "", no_update, True, False, done_version, no_update, no_update, no_update


@app_layout.app.callback(
    Output("counts-mode-store", "data"),
    Input("counts-mode", "value"),
    prevent_initial_call=True,
)
def sync_counts_mode(value):
    mode = value or "original"
    core._active_counts_mode = mode
    print(f"  Now using {'SPLIT corrected' if mode == 'corrected' else 'original'} counts", flush=True)
    return mode


@app_layout.app.callback(
    [Output("counts-mode", "options"),
     Output("counts-mode", "value", allow_duplicate=True)],
    [Input("split-done",      "data"),
     Input("seg-source",      "value"),
     Input("baysor-done",     "data"),
     Input("proseg-done",     "data"),
     Input("startup-trigger", "n_intervals"),
     Input("dataset-version", "data"),
     Input("extra-datasets-version", "data")],
    State("counts-mode", "value"),
    prevent_initial_call=True,
)
def update_counts_options(_, seg_source, _bd, _pd, _startup, _ds_version, _extra_ver, current_value):
    _tool = core._seg_tool(seg_source)
    with core._baysor_lock:
        bres = core._baysor_state["result"] if core._baysor_state["status"] == "done" else None
    with core._proseg_lock:
        pres = core._proseg_state["result"] if core._proseg_state["status"] == "done" else None
    if _tool == "baysor":
        alt_res = bres
    elif _tool == "proseg":
        alt_res = pres
    else:
        alt_res = None
    if alt_res is not None:
        has_corrected = alt_res.get("split_corrected_expr") is not None
    else:
        has_corrected = core.DATA.get("split_corrected_expr") is not None
        # several samples: corrected counts only when every sample has them (never a mix
        # of corrected and original values in one colour scale or one test)
        if dataview.EXTRA_DATASETS:
            has_corrected = dataview._corrected_status()["all"]
    options = [
        {"label": " Original",        "value": "original"},
        {"label": " SPLIT Corrected", "value": "corrected", "disabled": not has_corrected},
    ]
    # Only force a value change when currently on "corrected" but it's no longer available.
    # Any other case: no_update avoids spuriously triggering sync_counts_mode's log print.
    new_value = "original" if (not has_corrected and current_value == "corrected") else no_update
    return options, new_value


# ─── Spatial Niches callbacks ─────────────────────────────────────────────────

@app_layout.app.callback(
    Output("niche-modal",        "is_open"),
    Output("niche-label-method", "options"),
    Input("niche-modal-open-btn",  "n_clicks"),
    Input("niche-modal-close-btn", "n_clicks"),
    State("niche-modal", "is_open"),
    prevent_initial_call=True,
)
def open_niche_modal(open_clicks, close_clicks, is_open):
    if not open_clicks and not close_clicks:
        raise dash.exceptions.PreventUpdate
    triggered = callback_context.triggered_id
    if triggered == "niche-modal-close-btn":
        return False, no_update
    if not open_clicks:
        raise dash.exceptions.PreventUpdate
    # Build cell-type method options from available _annot_state labels
    with core._annot_lock:
        method_options = []
        for key in core._annot_state:
            if key == "labels":
                if core._annot_state[key] is not None:
                    method_options.append({"label": "Xenium (current)", "value": key})
            elif key.startswith("labels_"):
                if core._annot_state[key] is not None:
                    suffix = key[len("labels_"):]
                    label = suffix.replace("_", " ").title()
                    method_options.append({"label": label, "value": key})
    return True, method_options


@app_layout.app.callback(
    Output("niche-run-btn",      "disabled"),
    Output("niche-poll",         "disabled", allow_duplicate=True),
    Output("niche-modal-status", "children"),
    Input("niche-run-btn",       "n_clicks"),
    State("niche-n-clusters",    "value"),
    State("niche-n-neighbors",   "value"),
    State("niche-label-method",  "value"),
    State("seg-source",          "value"),
    prevent_initial_call=True,
)
def run_niche(n_clicks, n_clusters, n_neighbors, label_key, seg_source):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    n_clusters  = int(n_clusters  or 10)
    n_neighbors = int(n_neighbors or 20)

    tool = core._seg_tool(seg_source)
    with core._baysor_lock:
        bres = core._baysor_state["result"] if tool == "baysor" and core._baysor_state["result"] else None
    with core._proseg_lock:
        pres = core._proseg_state["result"] if tool == "proseg" and core._proseg_state["result"] else None
    alt_res = pres or bres

    # Label keys are per segmentation (Xenium ids vs the ids of one Baysor/Proseg
    # run): use the chosen method's labels for the active one, or refuse.
    if label_key:
        label_key = annotation._label_key_for_active_seg(label_key, alt_res)
        if label_key is None:
            return (False, True,
                    f"No annotation for the active {tool} segmentation — run annotation first.")

    with core._niche_lock:
        core._niche_state["status"]  = "starting"
        core._niche_state["message"] = "Starting…"

    jobs._job_start("niches", f"Spatial niches (k={n_clusters})", spatial._run_spatial_niches,
               args=(n_clusters, n_neighbors, label_key, alt_res),
               state=core._niche_state, lock=core._niche_lock, copy_context=True)
    return True, False, "Starting niche computation…"


@app_layout.app.callback(
    Output("niche-status",       "children"),
    Output("niche-poll",         "disabled"),
    Output("niche-run-btn",      "disabled", allow_duplicate=True),
    Output("niche-done",         "data"),
    Output("color-by",           "options", allow_duplicate=True),
    Input("niche-poll",          "n_intervals"),
    State("niche-done",          "data"),
    State("color-by",            "options"),
    prevent_initial_call=True,
)
def poll_niche(_, done_ver, current_options):
    with core._niche_lock:
        status  = core._niche_state["status"]
        message = core._niche_state["message"]

    if status == "idle":
        raise dash.exceptions.PreventUpdate

    if status == "running" or status == "starting":
        return message, False, True, no_update, no_update

    if status == "done":
        # Enable "Spatial Niches" in color-by dropdown
        new_options = []
        for opt in (current_options or []):
            if opt.get("value") == "niche":
                new_options.append({**opt, "disabled": False})
            else:
                new_options.append(opt)
        return message, True, False, (done_ver or 0) + 1, new_options

    if status == "error":
        return f"Error: {message}", True, False, no_update, no_update

    raise dash.exceptions.PreventUpdate


# ─── Spatial analysis callbacks ───────────────────────────────────────────────

@app_layout.app.callback(
    Output("spatial-run-btn", "disabled"),
    Output("spatial-poll", "disabled"),
    Output("spatial-modal-status", "children"),
    Input("spatial-run-btn", "n_clicks"),
    State("niche-label-method", "value"),
    State("niche-n-neighbors", "value"),
    State("seg-source", "value"),
    prevent_initial_call=True,
)
def run_spatial_analysis(n_clicks, label_key, n_neighbors, seg_source):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not label_key:
        return False, True, "Select a cell type annotation from the dropdown first."
    n_neighbors = int(n_neighbors or 10)
    tool = core._seg_tool(seg_source)
    alt_res = None
    if tool == "baysor":
        with core._baysor_lock:
            alt_res = core._baysor_state.get("result")
    elif tool == "proseg":
        with core._proseg_lock:
            alt_res = core._proseg_state.get("result")
    # Label keys are per segmentation: use the chosen method's labels for the active one
    label_key = annotation._label_key_for_active_seg(label_key, alt_res)
    if label_key is None:
        return (False, True,
                f"No annotation for the active {tool} segmentation — run annotation first.")
    jobs._job_start("spatial", "Spatial analysis", spatial._run_spatial_analysis,
               args=(label_key, n_neighbors, alt_res),
               state=core._spatial_state, lock=core._spatial_lock)
    return True, False, "Running…"


@app_layout.app.callback(
    Output("spatial-analysis-status", "children"),
    Output("spatial-poll", "disabled", allow_duplicate=True),
    Output("spatial-run-btn", "disabled", allow_duplicate=True),
    Output("spatial-done", "data"),
    Input("spatial-poll", "n_intervals"),
    State("spatial-done", "data"),
    prevent_initial_call=True,
)
def poll_spatial(_, done_ver):
    with core._spatial_lock:
        status = core._spatial_state["status"]
        msg = core._spatial_state["message"]
    if status == "done":
        return msg, True, False, (done_ver or 0) + 1
    if status == "error":
        return msg, True, False, done_ver
    return msg, False, True, done_ver


@app_layout.app.callback(
    Output("cooccurrence-plot", "figure"),
    Output("enrichment-plot", "figure"),
    Output("interaction-plot", "figure"),
    Output("chord-plot", "figure"),
    Input("spatial-done", "data"),
    Input("dataset-version", "data"),
    Input("cooccurrence-cbar-min", "value"),
    Input("cooccurrence-cbar-max", "value"),
    Input("enrichment-cbar-min", "value"),
    Input("enrichment-cbar-max", "value"),
    Input("interaction-normalize", "value"),
    Input("interaction-min-freq", "value"),
    Input("chord-normalize", "value"),
    Input("chord-min-freq", "value"),
)
def update_spatial_plots(_spatial_done, _ds_version,
                         co_cmin, co_cmax,
                         en_cmin, en_cmax,
                         int_norm, int_min_freq,
                         ch_norm, ch_min_freq):
    return (figures.make_cooccurrence_fig(zmin=co_cmin, zmax=co_cmax),
            figures.make_enrichment_fig(zmin=en_cmin, zmax=en_cmax),
            figures.make_interaction_graph_fig(normalize="on" in (int_norm or []), min_freq=int_min_freq or 0.0),
            figures.make_chord_fig(normalize="on" in (ch_norm or []), min_freq=ch_min_freq or 0.0))


@app_layout.app.callback(
    Output("niche-umap-plot", "figure"),
    Input("niche-done", "data"),
    Input("spatial-done", "data"),
    Input("dataset-version", "data"),
    Input("niche-umap-plot-w", "value"),
    Input("niche-umap-plot-h", "value"),
    State("seg-source", "value"),
)
def update_niche_umap(_niche_done, _spatial_done, _ds_version, w, h, seg_source):
    _tool = core._seg_tool(seg_source)
    return figures.make_niche_umap_fig(width=w, height=h,
                               baysor_active=(_tool == "baysor"),
                               proseg_active=(_tool == "proseg"))


@app_layout.app.callback(
    Output("roi-area-plot", "figure"),
    Input("roi-done", "data"),
    Input("dataset-version", "data"),
)
def update_roi_area_plot(_roi_done, _ds_version):
    return figures.make_roi_area_fig()


# ─── ROI Annotation callbacks ─────────────────────────────────────────────────

@app_layout.app.callback(
    Output("roi-pending",      "data",     allow_duplicate=True),
    Output("roi-save-modal",   "is_open",  allow_duplicate=True),
    Output("roi-save-preview", "children"),
    Output("roi-save-error",   "children", allow_duplicate=True),
    Output("roi-done",         "data",     allow_duplicate=True),
    Input("spatial-plot",      "selectedData"),
    State("seg-source",        "value"),
    State("roi-done",          "data"),
    prevent_initial_call=True,
)
def capture_lasso_selection(selected_data, seg_source, roi_done):
    """CB-1: Lasso selection → compute hull → open save modal."""
    if not selected_data or not selected_data.get("points"):
        raise dash.exceptions.PreventUpdate
    pts = selected_data["points"]
    cell_ids = [p.get("customdata") for p in pts if p.get("customdata") is not None]
    if not cell_ids:
        raise dash.exceptions.PreventUpdate

    tool = core._seg_tool(seg_source)
    if tool == "baysor":
        with core._baysor_lock:
            result = core._baysor_state.get("result")
        src_df = result["cells_df"] if result else None
    elif tool == "proseg":
        with core._proseg_lock:
            result = core._proseg_state.get("result")
        src_df = result["cells_df"] if result else None
    else:
        src_df = None

    # Errors: the message Div lives inside the save modal, so open it (with no pending
    # polygon — Save then asks for a new selection) and leave roi-done alone so the
    # figure is not rebuilt for a rejected selection.
    if len(cell_ids) < 3:
        return None, True, "", "Need at least 3 cells to define an ROI.", dash.no_update

    hull = regions._roi_compute_hull(cell_ids, src_df=src_df)
    if hull is None:
        return None, True, "", "Could not compute hull (need ≥3 non-collinear cells).", dash.no_update

    with core._roi_lock:
        core._roi_state["pending_hull"] = hull
    new_done = (roi_done or 0) + 1
    preview = f"{len(cell_ids):,} cells selected — enter name and class below."
    return hull, True, preview, "", new_done


@app_layout.app.callback(
    Output("roi-save-modal",  "is_open",  allow_duplicate=True),
    Output("roi-save-error",  "children", allow_duplicate=True),
    Output("roi-done",        "data",     allow_duplicate=True),
    Input("roi-save-btn",     "n_clicks"),
    State("roi-name-input",   "value"),
    State("roi-class-input",  "value"),
    State("roi-pending",      "data"),
    State("roi-done",         "data"),
    State("seg-source",       "value"),
    prevent_initial_call=True,
)
def save_roi(n_clicks, name, cls, polygon_xy, roi_done, seg_source):
    """CB-2: Save ROI."""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    name = (name or "").strip()
    cls  = (cls  or "region").strip()
    if not name:
        return True, "ROI Name is required.", roi_done
    if not polygon_xy:
        return True, "No polygon data — please draw a lasso selection first.", roi_done

    with core._roi_lock:
        existing = core._roi_state["rois"]
        duplicate = any(r["name"] == name and r["cls"] == cls for r in existing)
        if duplicate:
            print(f"ERROR: ROI '{name}' class '{cls}' already exists", flush=True)
            return True, f"ROI '{name}' (class '{cls}') already exists.", roi_done

        color = regions._roi_color(len(existing))
        new_roi = {"name": name, "cls": cls, "polygon_xy": polygon_xy, "color": color}
        core._roi_state["rois"].append(new_roi)
        regions._roi_save_cache()
        core._roi_state["pending_hull"] = None

    # Apply metadata to all loaded dfs
    with core._roi_lock:
        rois = list(core._roi_state["rois"])
    regions._roi_apply_everywhere(rois)

    threading.Thread(target=regions._roi_write_all_zarrs_bg, daemon=True).start()
    return False, "", (roi_done or 0) + 1


@app_layout.app.callback(
    Output("roi-save-modal", "is_open", allow_duplicate=True),
    Output("roi-done", "data", allow_duplicate=True),
    Input("roi-save-cancel-btn", "n_clicks"),
    State("roi-done", "data"),
    prevent_initial_call=True,
)
def cancel_roi_save(n_clicks, roi_done):
    """CB-3: Cancel save modal — clears pending hull and bumps roi-done to re-render figure."""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    with core._roi_lock:
        core._roi_state["pending_hull"] = None
    return False, (roi_done or 0) + 1


@app_layout.app.callback(
    Output("roi-done", "data", allow_duplicate=True),
    Input("spatial-plot", "relayoutData"),
    State("roi-done", "data"),
    prevent_initial_call=True,
)
def clear_lasso_on_tool_switch(relayout_data, roi_done):
    """Clear pending lasso selection when user switches away from lasso/select tool."""
    if not relayout_data or "dragmode" not in relayout_data:
        raise dash.exceptions.PreventUpdate
    if relayout_data["dragmode"] in ("lasso", "select"):
        raise dash.exceptions.PreventUpdate
    with core._roi_lock:
        if core._roi_state.get("pending_hull") is None:
            raise dash.exceptions.PreventUpdate
        core._roi_state["pending_hull"] = None
    return (roi_done or 0) + 1


@app_layout.app.callback(
    Output("roi-done", "data", allow_duplicate=True),
    Input("roi-show-toggle", "value"),
    State("roi-done", "data"),
    prevent_initial_call=True,
)
def toggle_roi_show(value, roi_done):
    """CB-4: Toggle ROI visibility."""
    with core._roi_lock:
        core._roi_state["show"] = "show" in (value or [])
    return (roi_done or 0) + 1


@app_layout.app.callback(
    Output("roi-manage-modal", "is_open"),
    Output("roi-list-div",     "children"),
    Output("roi-op-a",         "options"),
    Output("roi-op-b",         "options"),
    Input("roi-manage-btn",       "n_clicks"),
    Input("roi-manage-close-btn", "n_clicks"),
    State("roi-manage-modal",     "is_open"),
    prevent_initial_call=True,
)
def open_roi_manager(open_clicks, close_clicks, is_open):
    """CB-5: Open/populate ROI manager."""
    triggered = callback_context.triggered_id
    if triggered == "roi-manage-close-btn":
        return False, dash.no_update, dash.no_update, dash.no_update
    if not open_clicks:
        raise dash.exceptions.PreventUpdate

    with core._roi_lock:
        rois = list(core._roi_state["rois"])

    table, op_opts = regions._roi_manager_contents(rois)
    return True, table, op_opts, op_opts


@app_layout.app.callback(
    Output("roi-done",     "data",     allow_duplicate=True),
    Output("roi-list-div", "children", allow_duplicate=True),
    Output("roi-op-a",     "options",  allow_duplicate=True),
    Output("roi-op-b",     "options",  allow_duplicate=True),
    Output("roi-op-a",     "value",    allow_duplicate=True),
    Output("roi-op-b",     "value",    allow_duplicate=True),
    Input({"type": "roi-delete-btn", "index": ALL}, "n_clicks"),
    State("roi-done", "data"),
    prevent_initial_call=True,
)
def delete_roi(n_clicks_list, roi_done):
    """CB-6: Delete ROI by index."""
    triggered = callback_context.triggered_id
    if triggered is None or not any(n_clicks_list):
        raise dash.exceptions.PreventUpdate
    idx = triggered["index"]
    with core._roi_lock:
        rois = core._roi_state["rois"]
        if idx < 0 or idx >= len(rois):
            raise dash.exceptions.PreventUpdate
        # Reassign colors to keep palette stable
        rois.pop(idx)
        for j, r in enumerate(rois):
            r["color"] = regions._roi_color(j)
        regions._roi_save_cache()

    with core._roi_lock:
        rois = list(core._roi_state["rois"])
    regions._roi_apply_everywhere(rois)
    threading.Thread(target=regions._roi_write_all_zarrs_bg, daemon=True).start()

    # Rebuild table and A/B options (option values are list positions, which shifted)
    table, op_opts = regions._roi_manager_contents(rois)
    return (roi_done or 0) + 1, table, op_opts, op_opts, None, None


@app_layout.app.callback(
    Output("roi-op-error", "children"),
    Output("roi-done",     "data",    allow_duplicate=True),
    Output("roi-list-div", "children", allow_duplicate=True),
    Output("roi-op-a",     "options",  allow_duplicate=True),
    Output("roi-op-b",     "options",  allow_duplicate=True),
    Input("roi-op-btn",    "n_clicks"),
    State("roi-op-a",      "value"),
    State("roi-op",        "value"),
    State("roi-op-b",      "value"),
    State("roi-op-name",   "value"),
    State("roi-op-class",  "value"),
    State("roi-done",      "data"),
    prevent_initial_call=True,
)
def apply_roi_operation(n_clicks, idx_a, op, idx_b, result_name, result_cls, roi_done):
    """CB-7: ROI set operation."""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    _nu3 = (no_update, no_update, no_update)   # roi-list-div, roi-op-a/b options
    result_name = (result_name or "").strip()
    result_cls  = (result_cls  or "region").strip()
    if idx_a is None or idx_b is None or not op:
        return ("Select ROI A, operation, and ROI B.", roi_done) + _nu3
    if not result_name:
        return ("Enter a result name.", roi_done) + _nu3

    try:
        from shapely.geometry import Polygon as _SPoly
        with core._roi_lock:
            rois = core._roi_state["rois"]
            if idx_a >= len(rois) or idx_b >= len(rois):
                return ("Invalid ROI index.", roi_done) + _nu3
            poly_a = _SPoly(rois[idx_a]["polygon_xy"])
            poly_b = _SPoly(rois[idx_b]["polygon_xy"])
            if op == "union":
                result_geom = poly_a.union(poly_b)
            elif op == "intersection":
                result_geom = poly_a.intersection(poly_b)
            else:
                result_geom = poly_a.difference(poly_b)
            # Store the exact outline: a convex hull would undo a difference/notch
            # and fill the gap between two separate ROIs.
            if result_geom.is_empty:
                return ("Operation produced empty result.", roi_done) + _nu3
            if result_geom.geom_type == "MultiPolygon":
                return ("Result has several separate parts, which one ROI cannot hold "
                        "— draw separate ROIs instead.", roi_done) + _nu3
            if result_geom.geom_type != "Polygon":
                return ("Operation produced empty result.", roi_done) + _nu3
            if len(result_geom.interiors) > 0:
                return ("Result has a hole (one ROI lies inside the other), which is "
                        "not supported.", roi_done) + _nu3
            polygon_xy = [list(c) for c in result_geom.exterior.coords]

            duplicate = any(r["name"] == result_name and r["cls"] == result_cls for r in rois)
            if duplicate:
                return (f"ROI '{result_name}' (class '{result_cls}') already exists.", roi_done) + _nu3

            color = regions._roi_color(len(rois))
            rois.append({"name": result_name, "cls": result_cls,
                          "polygon_xy": polygon_xy, "color": color})
            regions._roi_save_cache()

        with core._roi_lock:
            rois_snap = list(core._roi_state["rois"])
        regions._roi_apply_everywhere(rois_snap)
        threading.Thread(target=regions._roi_write_all_zarrs_bg, daemon=True).start()
        table, op_opts = regions._roi_manager_contents(rois_snap)
        return "", (roi_done or 0) + 1, table, op_opts, op_opts
    except Exception as e:
        return (f"Error: {e}", roi_done) + _nu3


def _refresh_colorby_options(current_options) -> list:
    """color-by options with the data-dependent entries recomputed for the loaded
    dataset: cell-type entries enabled per available labels, 'Spatial Niches'
    disabled when no niche column exists any more, roi_* entries rebuilt.
    Shared by the two callbacks below: Load Sample triggers both, and each starts
    from the same options snapshot, so they must return the same list."""
    with core._annot_lock:
        methods_with_labels = {
            m for m in core._ANNOT_METHODS
            if any(k.startswith(f"labels_{m}") and v is not None
                   for k, v in core._annot_state.items())
        }
    has_niche = "niche" in core.DATA["df"].columns
    for _lock, _state in ((core._baysor_lock, core._baysor_state), (core._proseg_lock, core._proseg_state)):
        with _lock:
            _cdf = (_state.get("result") or {}).get("cells_df")
        has_niche = has_niche or (_cdf is not None and "niche" in _cdf.columns)
    with core._roi_lock:
        rois = list(core._roi_state["rois"])
    classes = list(dict.fromkeys(r["cls"] for r in rois))  # preserve order, deduplicate
    options = []
    for opt in (current_options or []):
        val = opt["value"]
        if val.startswith("roi_") or val.startswith("geneset:"):
            continue
        if val.startswith("cell_type:"):
            m = val.split(":")[1]
            options.append({**opt, "disabled": m not in methods_with_labels})
        elif val == "niche" and not has_niche:
            options.append({**opt, "disabled": True})
        else:
            options.append(opt)
    for cls in classes:
        options.append({"label": f"ROI: {cls}", "value": f"roi_{cls}"})
    for name in analysis._geneset_names():
        options.append({"label": f"Gene set: {name}", "value": f"geneset:{name}"})
    return options


@app_layout.app.callback(
    Output("color-by", "options", allow_duplicate=True),
    Input("startup-trigger", "n_intervals"),
    Input("dataset-version", "data"),
    State("color-by",        "options"),
    prevent_initial_call=True,
)
def init_colorby_annot_options(_, _ds_version, current_options):
    """On startup and after Load Sample, enable cell-type color-by options for any
    pre-loaded annotations of the current dataset."""
    return _refresh_colorby_options(current_options)


@app_layout.app.callback(
    Output("color-by", "options", allow_duplicate=True),
    Input("roi-done",  "data"),
    State("color-by",  "options"),
    prevent_initial_call=True,
)
def update_color_by_for_rois(roi_done, current_options):
    """CB-8: Add/update roi_* options in color-by dropdown."""
    return _refresh_colorby_options(current_options)


@app_layout.app.callback(
    Output("roi-sidebar-summary", "children"),
    Input("roi-done", "data"),
    prevent_initial_call=True,
)
def update_roi_sidebar_summary(roi_done):
    """CB-9: Update sidebar ROI summary."""
    with core._roi_lock:
        rois = list(core._roi_state["rois"])
    if not rois:
        return "No ROIs defined."
    class_counts: dict = {}
    for r in rois:
        class_counts[r["cls"]] = class_counts.get(r["cls"], 0) + 1
    parts = [f"{v} {k}" for k, v in class_counts.items()]
    return f"{len(rois)} ROI{'s' if len(rois)!=1 else ''} ({', '.join(parts)})"
