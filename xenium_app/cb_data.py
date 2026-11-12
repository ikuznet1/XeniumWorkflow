"""Callbacks: SpatialData patches, tissue info, jobs panel, samples, cache clean-up, Save."""

import os
from dash import no_update, Patch
import dash
from dash import html, Input, Output, State, callback_context, ALL

from . import (
    core, jobs, caching, ui, loading, dataview, annotation, spage, morph, figures, analysis,
    sdata_store, regions, app_layout,
)


# ─── SpatialData / Sopa callbacks ─────────────────────────────────────────────

@app_layout.app.callback(
    Output("sdata-poll",              "disabled",  allow_duplicate=True),
    Output("sdata-status",            "children",  allow_duplicate=True),
    Output("sdata-clear-cache-btn",   "disabled",  allow_duplicate=True),
    Output("sdata-prepare-btn",       "disabled",  allow_duplicate=True),
    Output("reseg-patches-confirmed", "data",      allow_duplicate=True),
    Output("reseg-patch-confirm-div", "style",     allow_duplicate=True),
    Input("sdata-clear-cache-btn",    "n_clicks"),
    State("sdata-qv",                 "value"),
    prevent_initial_call=True,
)
def start_sdata_clear(n_clicks, qv):
    """Delete cache then trigger a forced re-read from raw files."""
    with core._sdata_lock:
        if core._sdata_state["status"] == "running":
            return False, "Already running…", True, True, no_update, no_update
        core._sdata_state.update({"status": "running", "message": "Clearing cache…",
                              "roi": None, "patches": None})
    try:
        if not core.DATA.get("data_dir"):
            raise RuntimeError("not available for a dataset loaded from a .zarr")
        sdata_store.clear_sdata_cache()
        sdata_store.to_spatialdata(qv_threshold=int(qv or 20), force=True)
    except Exception as exc:
        # leave "running", otherwise every later click reports "Already running…"
        with core._sdata_lock:
            core._sdata_state.update({"status": "error",
                                 "message": f"Clear cache failed: {exc}"[:300]})
        return True, f"✗ Clear cache failed: {exc}", False, False, no_update, no_update
    return False, "Re-reading from raw files…", True, True, 0, {"display": "none"}


@app_layout.app.callback(
    Output("sdata-status",            "children",  allow_duplicate=True),
    Output("sdata-poll",              "disabled",  allow_duplicate=True),
    Output("sdata-clear-cache-btn",   "disabled",  allow_duplicate=True),
    Output("sdata-prepare-btn",       "disabled",  allow_duplicate=True),
    Output("sdata-done",              "data"),
    Output("sdata-show-roi",          "value",     allow_duplicate=True),
    Output("sdata-show-patches",      "value",     allow_duplicate=True),
    Output("reseg-patch-confirm-div", "style",     allow_duplicate=True),
    Output("reseg-patch-confirm-msg", "children",  allow_duplicate=True),
    Input("sdata-poll",               "n_intervals"),
    State("sdata-done",               "data"),
    State("sdata-patch-width",        "value"),
    State("sdata-patch-min-tx",       "value"),
    prevent_initial_call=True,
)
def poll_sdata(_, done_version, patch_width, patch_min_tx):
    with core._sdata_lock:
        status      = core._sdata_state["status"]
        message     = core._sdata_state["message"]
        has_roi     = core._sdata_state.get("roi") is not None
        has_patches = core._sdata_state.get("patches") is not None
        has_sdata   = core._sdata_state.get("sdata") is not None

    HIDE = {"display": "none"}
    SHOW = {}
    no_chg = no_update

    if status == "running":
        # poll=enabled, clear-btn=disabled, prepare-btn=disabled
        return message, False, True, True, no_chg, no_chg, no_chg, HIDE, no_chg

    if status in ("done", "roi_done", "patches_done", "error"):
        # Chain: sdata done → auto-run ROI if not done
        if status == "done" and not has_roi and has_sdata:
            with core._sdata_lock:
                core._sdata_state.update({"status": "running", "message": "Detecting tissue ROI…"})
            sdata_store.segment_tissue_roi()
            return "Detecting tissue ROI…", False, True, True, no_chg, no_chg, no_chg, HIDE, no_chg

        # Chain: ROI done → auto-run patches if not done
        # (not on "error": a failed patch run must be shown, not retried every tick)
        if status in ("done", "roi_done") and has_roi and not has_patches:
            width  = float(patch_width  or 1000)
            min_tx = int(patch_min_tx   or 10)
            with core._sdata_lock:
                core._sdata_state.update({"status": "running", "message": "Creating patches…"})
            sdata_store.create_patches(width_um=width, min_transcripts=min_tx)
            return "Creating patches…", False, True, True, no_chg, no_chg, no_chg, HIDE, no_chg

        # Patches exist — show overlays + confirmation, re-enable buttons
        if has_patches:
            with core._sdata_lock:
                n_patches = len(core._sdata_state["patches"])
            confirm_msg = (f"✓ {n_patches} patches ready — review ROI and patches on the "
                           f"spatial plot, then confirm to proceed.")
            return (message, True, False, False, (done_version or 0) + 1,
                    ["yes"], ["yes"], SHOW, confirm_msg)

        # Error — re-enable buttons so user can retry
        if status == "error":
            return f"✗ {message}", True, False, False, no_chg, no_chg, no_chg, HIDE, no_chg

        return message, True, False, False, no_chg, no_chg, no_chg, no_chg, no_chg

    return (no_update, no_update, no_update, no_update, no_update,
            no_update, no_update, no_update, no_update)


@app_layout.app.callback(
    Output("sdata-poll",             "disabled",    allow_duplicate=True),
    Output("sdata-status",           "children",    allow_duplicate=True),
    Output("sdata-show-roi",         "value",       allow_duplicate=True),
    Output("sdata-show-patches",     "value",       allow_duplicate=True),
    Output("reseg-patch-confirm-div","style",       allow_duplicate=True),
    Output("reseg-patch-confirm-msg","children",    allow_duplicate=True),
    Output("sdata-prepare-btn",      "disabled",    allow_duplicate=True),
    Input("sdata-prepare-btn",       "n_clicks"),
    State("sdata-qv",                "value"),
    prevent_initial_call=True,
)
def start_prepare_patches(n_clicks, qv):
    """User clicked Prepare Patches — start pipeline or show cached result."""
    if not n_clicks:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update
    with core._sdata_lock:
        status      = core._sdata_state["status"]
        message     = core._sdata_state["message"]
        has_patches = core._sdata_state.get("patches") is not None
    # Patches already ready — just show overlays and confirmation
    if has_patches:
        with core._sdata_lock:
            n_patches = len(core._sdata_state["patches"])
        confirm_msg = (f"✓ {n_patches} patches ready — review ROI and patches on the "
                       f"spatial plot, then confirm to proceed.")
        return True, message, ["yes"], ["yes"], {}, confirm_msg, False
    # Already running — just enable poll to show progress
    if status == "running":
        return False, message, no_update, no_update, {"display": "none"}, "", True
    # Start the pipeline: sdata → ROI → patches (chained by poll_sdata)
    with core._sdata_lock:
        core._sdata_state.update({"status": "running", "message": "Building SpatialData…"})
    sdata_store.to_spatialdata(qv_threshold=int(qv or 20))
    return False, "Building SpatialData…", no_update, no_update, {"display": "none"}, "", True


@app_layout.app.callback(
    Output("reseg-patches-confirmed", "data"),
    Output("reseg-step2-content",     "style"),
    Output("reseg-step2-overlay",     "style"),
    Output("reseg-modal-run-btn",     "disabled"),
    Input("reseg-confirm-patches-btn", "n_clicks"),
    Input("reseg-skip-patches-btn",    "n_clicks"),
    prevent_initial_call=True,
)
def confirm_patches(n_clicks, skip_clicks):
    if not (n_clicks or skip_clicks):
        return no_update, no_update, no_update, no_update
    # "skip" = run on the full tissue: run_reseg_modal then ignores any patches
    # that exist in the session (for Baysor and Proseg alike)
    skipped = callback_context.triggered_id == "reseg-skip-patches-btn" and skip_clicks
    return ("skip" if skipped else 1), {}, {"display": "none"}, False


@app_layout.app.callback(
    Output("spatial-plot", "figure", allow_duplicate=True),
    Input("sdata-show-roi",     "value"),
    Input("sdata-show-patches", "value"),
    Input("sdata-done",         "data"),
    prevent_initial_call=True,
)
def update_sdata_overlays(show_roi, show_patches, _sdata_ver):
    roi_on     = "yes" in (show_roi     or [])
    patches_on = "yes" in (show_patches or [])
    # layout.shapes also holds ROI outlines, RCTD pies and sample separators:
    # rebuild the whole list instead of replacing it with the sdata overlays only
    shapes = list(figures._spatial_base_shapes) + figures._spatial_overlay_shapes(roi_on, patches_on)
    patched = Patch()
    patched["layout"]["shapes"] = shapes
    return patched


# ─── Dataset switcher callbacks ───────────────────────────────────────────────

@app_layout.app.callback(
    Output("tissue-info-content", "children"),
    Input("dataset-version", "data"),
    Input("seg-source", "value"),
    Input("split-done", "data"),
)
def update_tissue_info(_, seg_source, _split_done):
    meta = core.DATA.get("metadata", {})
    # experiment.xenium of XOA 3.x / Prime bundles has no num_transcripts:
    # show the transcripts assigned to cells instead
    _ntx_label, _ntx = "Transcripts", meta.get("num_transcripts")
    if _ntx is None:
        _tx_df = core.DATA.get("_df_original", core.DATA.get("df"))
        if _tx_df is not None and "transcript_counts" in _tx_df.columns:
            _ntx_label, _ntx = "Transcripts (assigned)", int(_tx_df["transcript_counts"].sum())
    try:
        _ntx_str = "\u2014" if _ntx is None else f"{int(_ntx):,}"
    except (TypeError, ValueError):
        _ntx_str = str(_ntx)
    cache_size = caching._cache_size_str()

    # Determine active cell count from reseg result if active
    tool = core._seg_tool(seg_source)
    if tool == "baysor":
        with core._baysor_lock:
            res = core._baysor_state.get("result")
        n_cells_display = f"{len(res['cells_df']):,}" if res else f"{meta.get('num_cells', 0):,}"
        seg_label = "Baysor"
    elif tool == "proseg":
        with core._proseg_lock:
            res = core._proseg_state.get("result")
        n_cells_display = f"{len(res['cells_df']):,}" if res else f"{meta.get('num_cells', 0):,}"
        seg_label = "Proseg"
    else:
        n_cells_display = f"{meta.get('num_cells', 0):,}"
        seg_label = "Xenium"

    return html.Div([
        html.Div([
            html.Div("XENIUM EXPLORER", style={
                "fontSize": "11px", "fontWeight": "700",
                "letterSpacing": "2px", "color": core.ACCENT, "marginBottom": "4px",
            }),
            html.Div(meta.get("run_name", "\u2014"),
                     style={"fontSize": "13px", "color": core.TEXT, "fontWeight": "600"}),
            html.Div(meta.get("region_name", ""),
                     style={"fontSize": "11px", "color": core.MUTED}),
        ], style={"marginBottom": "10px"}),

        html.Div([
            html.Hr(style={"borderColor": core.BORDER, "margin": "0 0 10px 0"}),
            html.Div([
                ui.stat_row("Cells",       n_cells_display),
                ui.stat_row("Seg",         seg_label),
                ui.stat_row(_ntx_label, _ntx_str),
                ui.stat_row("Panel",       meta.get("panel_name", "\u2014")),
                ui.stat_row("Genes",       str(len(core.DATA.get("gene_names", [])))),
                ui.stat_row("Tissue",      meta.get("panel_tissue_type", "\u2014")),
                ui.stat_row("Pixel",       f"{meta.get('pixel_size', core.PIXEL_SIZE_UM)} \u00b5m"),
            ], style={"marginBottom": "10px"}),
            html.Hr(style={"borderColor": core.BORDER, "margin": "0 0 10px 0"}),
            # Sample switcher
            html.Button("\U0001f4c2 Load Different Sample", id="sample-modal-open-btn", n_clicks=0,
                        style={"width": "100%", "padding": "5px 0",
                               "backgroundColor": core.CARD_BG, "color": core.TEXT,
                               "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                               "cursor": "pointer", "fontSize": "11px", "marginBottom": "4px"}),
            html.Button("\U0001f4be Save as SpatialData", id="save-sdata-open-btn", n_clicks=0,
                        style={"width": "100%", "padding": "5px 0",
                               "backgroundColor": core.CARD_BG, "color": core.TEXT,
                               "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                               "cursor": "pointer", "fontSize": "11px", "marginBottom": "4px"}),
            html.Button("\u2716 Remove Extra Samples", id="remove-extra-samples-btn", n_clicks=0,
                        style={"width": "100%", "padding": "5px 0",
                               "backgroundColor": core.CARD_BG, "color": core.MUTED,
                               "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                               "cursor": "pointer", "fontSize": "11px", "marginBottom": "8px"}),
            html.Div(id="extra-samples-status",
                     style={"fontSize": "10px", "color": core.MUTED, "marginBottom": "4px", "minHeight": "14px"}),
            # Cache info
            html.Div([
                html.Span(f"Cache: {cache_size}",
                          id="cache-size-display",
                          style={"fontSize": "11px", "color": core.MUTED}),
                html.Button("Clean", id="cache-clean-open-btn", n_clicks=0,
                            style={"marginLeft": "8px", "padding": "1px 8px",
                                   "backgroundColor": core.CARD_BG, "color": core.MUTED,
                                   "border": f"1px solid {core.BORDER}", "borderRadius": "3px",
                                   "cursor": "pointer", "fontSize": "10px"}),
            ], style={"display": "flex", "alignItems": "center"}),
        ], className="tissue-details"),
    ], id="tissue-info-section", style={"marginBottom": "14px"})


@app_layout.app.callback(
    Output("morph-enable", "options"),
    Output("morph-enable", "labelStyle"),
    Output("morph-enable", "value"),
    Input("dataset-version", "data"),
)
def update_morph_enable_state(_):
    """Disable morphology overlay toggle when the dataset has no morphology image
    (zarr input, or a bundle without image files such as one assembled from GEO)."""
    has_images = bool(core.DATA.get("data_dir")) and morph._morph_focus_path(core.DATA["data_dir"], 0) is not None
    options = [{"label": html.Span(
        " Enable Image Overlay" + ("" if has_images else " (no image data)"),
        style={"fontSize": "13px", "color": core.TEXT if has_images else core.MUTED},
    ), "value": "show", "disabled": not has_images}]
    label_style = {"display": "flex", "alignItems": "center",
                   "cursor": "pointer" if has_images else "not-allowed"}
    # a tick left from a dataset with images could not be removed once the option is
    # disabled, and would keep the overlay code paths of both renderers active
    return options, label_style, (no_update if has_images else [])


def _is_zarr_store(path: str) -> bool:
    """True when *path* holds a zarr group: v3 stores (zarr >= 3, what this app
    writes) have zarr.json at the root, v2 stores .zgroup / .zattrs."""
    return any(os.path.exists(os.path.join(path, m)) for m in ("zarr.json", ".zgroup", ".zattrs"))


def _running_jobs() -> list:
    """Names of the background jobs that are running right now: the active jobs of
    the registry, plus any job kind whose status dict is already claimed (a callback
    sets "running" just before it starts the job's thread)."""
    busy = []
    for job in jobs._jobs_list(active_only=True):
        if job.name not in busy:
            busy.append(job.name)
    for name, lock, state in (
            ("Baysor", core._baysor_lock, core._baysor_state), ("Proseg", core._proseg_lock, core._proseg_state),
            ("annotation", core._annot_lock, core._annot_state), ("SpaGE", core._spage_lock, core._spage_state),
            ("SpatialData / patches", core._sdata_lock, core._sdata_state),
            ("UMAP", core._umap_reseg_lock, core._umap_reseg_state), ("SPLIT", core._split_lock, core._split_state),
            ("niches", core._niche_lock, core._niche_state),
            ("spatial analysis", core._spatial_lock, core._spatial_state),
            ("Save as SpatialData", core._save_sdata_lock, core._save_sdata_state),
            ("CellNEST load", core._cellnest_lock, core._cellnest_state),
            ("differential expression", analysis._de_lock, analysis._de_state),
            ("gene-set scoring", analysis._geneset_lock, analysis._geneset_state),
            ("transcript density", analysis._density_lock, analysis._density_state)):
        with lock:
            if state.get("status") in ("running", "starting", "loading") and name not in busy:
                busy.append(name)
    return busy


# ─── Jobs panel callbacks ─────────────────────────────────────────────────────

@app_layout.app.callback(
    Output("jobs-panel-rows",    "children"),
    Output("jobs-panel-summary", "children"),
    Output("jobs-panel",         "style"),
    Output("jobs-poll",          "disabled"),
    Input("jobs-poll",           "n_intervals"),
)
def update_jobs_panel(_):
    """Redraw the jobs panel. The interval that drives it switches itself off when
    no job is running (and none ended in the last few seconds)."""
    rows = jobs._jobs_panel_rows()
    if not rows:
        return [], [], ui._JOBS_PANEL_HIDDEN, True
    return ui._jobs_panel_children(rows), ui._jobs_panel_summary(rows), ui._JOBS_PANEL_STYLE, False


@app_layout.app.callback(
    Output("jobs-poll", "disabled", allow_duplicate=True),
    Input("annot-poll",      "disabled"),
    Input("baysor-poll",     "disabled"),
    Input("proseg-poll",     "disabled"),
    Input("spage-poll",      "disabled"),
    Input("split-poll",      "disabled"),
    Input("cellnest-poll",   "disabled"),
    Input("niche-poll",      "disabled"),
    Input("umap-reseg-poll", "disabled"),
    Input("spatial-poll",    "disabled"),
    Input("sdata-poll",      "disabled"),
    Input("save-sdata-poll", "disabled"),
    Input("spage-repl-poll", "n_intervals"),
    State("jobs-poll",       "disabled"),
    prevent_initial_call=True,
)
def wake_jobs_poll(_a, _b, _p, _sg, _sp, _c, _n, _u, _s, _sd, _sv, _repl, poll_disabled):
    """Switch the jobs-panel interval on when a job starts: every Start button
    enables its own status poll, and the always-on REPL poll picks up jobs started
    from the REPL or from another browser tab."""
    if jobs._jobs_panel_rows() and poll_disabled:
        return False
    return no_update


@app_layout.app.callback(
    Output("jobs-panel-rows", "children", allow_duplicate=True),
    Input({"type": "job-cancel-btn", "job": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def cancel_job_clicked(n_clicks_list):
    """Cancel button of a job row."""
    triggered = callback_context.triggered_id
    # the panel is redrawn every second: new buttons arrive with n_clicks == 0
    if triggered is None or not any(n_clicks_list) \
            or not callback_context.triggered[0].get("value"):
        raise dash.exceptions.PreventUpdate
    jobs._job_cancel(triggered["job"])
    return ui._jobs_panel_children(jobs._jobs_panel_rows())


@app_layout.app.callback(
    Output("sample-modal", "is_open"),
    Output("sample-picker", "options"),
    Input("sample-modal-open-btn", "n_clicks"),
    Input("sample-modal-close-btn", "n_clicks"),
    State("sample-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_sample_modal(open_clicks, close_clicks, is_open):
    if callback_context.triggered_id == "sample-modal-open-btn" and open_clicks:
        return True, sdata_store._available_samples()
    return False, no_update


@app_layout.app.callback(
    Output("sample-load-status",  "children"),
    Output("sample-modal",        "is_open",       allow_duplicate=True),
    Output("dataset-version",     "data",          allow_duplicate=True),
    Output("seg-source",          "value",         allow_duplicate=True),
    Output("gene-selector",       "value",         allow_duplicate=True),
    Output("roi-done",            "data",          allow_duplicate=True),
    Output("spatial-relayout",    "data",          allow_duplicate=True),
    Output("extra-datasets-version", "data",       allow_duplicate=True),
    Output("cellnest-version",    "data",          allow_duplicate=True),
    Input("sample-load-btn",      "n_clicks"),
    State("sample-picker",        "value"),
    State("sample-custom-path",   "value"),
    State("dataset-version",      "data"),
    State("roi-done",             "data"),
    State("extra-datasets-version", "data"),
    State("cellnest-version",     "data"),
    prevent_initial_call=True,
)
def load_sample(n_clicks, picked_path, custom_path, ds_version, roi_done,
                extra_ver=None, cellnest_ver=None):
    # seg-source, gene-selector, roi-done, spatial-relayout, extra-datasets-version,
    # cellnest-version
    _nu3 = (no_update, no_update, no_update, no_update, no_update, no_update)
    new_dir = (custom_path or "").strip() or picked_path
    if not new_dir:
        return ("Select a sample or enter a path.", no_update, no_update) + _nu3
    new_dir = os.path.abspath(new_dir)
    is_zarr = new_dir.rstrip("/").endswith(".zarr")
    if is_zarr:
        if not _is_zarr_store(new_dir):
            return (f"Not a valid SpatialData zarr: {new_dir}", no_update, no_update) + _nu3
        if new_dir == core.DATA.get("sdata_path"):
            return ("Already loaded.", no_update, no_update) + _nu3
    else:
        if not os.path.exists(os.path.join(new_dir, "experiment.xenium")):
            return (f"Not a valid Xenium directory: {new_dir}", no_update, no_update) + _nu3
        if new_dir == core.DATA["data_dir"]:
            return ("Already loaded.", no_update, no_update) + _nu3
    # A job still running for the current dataset would lose its result (jobs are
    # fenced by the dataset generation, see _install_dataset) and leave its poll
    # waiting on a reset state.
    _busy = _running_jobs()
    if _busy:
        return (f"Still running: {', '.join(_busy)}. Wait for it to finish before "
                f"loading another sample.", no_update, no_update) + _nu3
    try:
        new_data = loading.load_from_sdata_zarr(new_dir) if is_zarr else loading.load_xenium_data(new_dir)
        # Swap DATA and rebuild everything derived from it (expr_csc, gene lists, caches)
        dataview._install_dataset(new_data)
        # Reset all per-dataset state
        with core._baysor_lock:
            core._baysor_state.update({"status": "idle", "message": "", "result": None})
        with core._proseg_lock:
            core._proseg_state.update({"status": "idle", "message": "", "result": None})
        with core._annot_lock:
            core._annot_state.update({"status": "idle", "message": "", "labels": None,
                                  "labels_proseg": None, "labels_baysor": None})
            # Per-method label / weight keys (labels_celltypist, rctd_weights_…) too
            for _k in list(core._annot_state):
                if _k not in ("status", "message"):
                    core._annot_state[_k] = None
        with core._spage_lock:
            core._spage_state.update({"status": "idle", "message": "", "result": None,
                                  "result_path": None, "result_genes": None})
        with core._sdata_lock:
            core._sdata_state.update({"status": "idle", "message": "", "sdata": None,
                                  "roi": None, "patches": None})
        with core._umap_reseg_lock:
            core._umap_reseg_state.update({"status": "idle", "message": "", "result": None})
        with core._split_lock:
            core._split_state.update({"status": "idle", "message": "", "result": None})
        with core._niche_lock:
            core._niche_state.update({"status": "idle", "message": ""})
        with core._spatial_lock:
            core._spatial_state.update({"status": "idle", "message": "", "result": None})
        with core._roi_lock:
            core._roi_state["rois"] = []
            core._roi_state["pending_hull"] = None
        # CellNEST results (edges drawn from their own coordinates) and the extra
        # side-by-side samples (offsets computed against the previous primary)
        # belong to the previous dataset.
        with core._cellnest_lock:
            core._cellnest_state.update({"status": "idle", "message": "", "data": None})
        if dataview.EXTRA_DATASETS:
            print(f"  Removed {len(dataview.EXTRA_DATASETS)} extra sample(s) of the previous dataset",
                  flush=True)
            dataview.EXTRA_DATASETS.clear()
        # Re-run autoloads for the new dataset
        regions._auto_load_rois()
        analysis._an_autoload()
        annotation._annot_autoload()
        sdata_store._sdata_autoload()
        spage._spage_autoload()
        print(f"  Loaded new dataset: {new_dir}", flush=True)
        # {}: the stored viewport belongs to the previous dataset
        return ("Loaded.", False, (ds_version or 0) + 1,
                "xenium", dataview._default_gene, (roi_done or 0) + 1, {},
                (extra_ver or 0) + 1, (cellnest_ver or 0) + 1)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return (f"Error: {exc}", no_update, no_update) + _nu3


@app_layout.app.callback(
    Output("sample-load-status",       "children",  allow_duplicate=True),
    Output("sample-modal",             "is_open",   allow_duplicate=True),
    Output("extra-datasets-version",   "data",      allow_duplicate=True),
    Output("extra-samples-status",     "children",  allow_duplicate=True),
    Input("sample-add-btn",            "n_clicks"),
    State("sample-picker",             "value"),
    State("sample-custom-path",        "value"),
    State("extra-datasets-version",    "data"),
    prevent_initial_call=True,
)
def add_sample(n_clicks, picked_path, custom_path, extra_ver):
    """Load a new Xenium dataset and append it to EXTRA_DATASETS side-by-side."""
    new_dir = (custom_path or "").strip() or picked_path
    if not new_dir:
        return "Select a sample or enter a path.", no_update, no_update, no_update
    new_dir = os.path.abspath(new_dir)
    is_zarr = new_dir.rstrip("/").endswith(".zarr")
    if is_zarr:
        if not _is_zarr_store(new_dir):
            return f"Not a valid SpatialData zarr: {new_dir}", no_update, no_update, no_update
    else:
        if not os.path.exists(os.path.join(new_dir, "experiment.xenium")):
            return f"Not a valid Xenium directory: {new_dir}", no_update, no_update, no_update
    # Check for duplicates (use sdata_path for zarr, data_dir for Xenium)
    all_dirs = [core.DATA.get("sdata_path") if is_zarr else core.DATA["data_dir"]] + [
        eds.get("sdata_path") if is_zarr else eds["data_dir"] for eds in dataview.EXTRA_DATASETS
    ]
    if new_dir in all_dirs:
        return "Sample already loaded.", no_update, no_update, no_update
    try:
        new_data = loading.load_from_sdata_zarr(new_dir) if is_zarr else loading.load_xenium_data(new_dir)
        # Compute x_offset: place new sample to the right of all current ones
        # x_offset = max(x_centroid) of the last dataset + 500 µm gap - min(x_centroid) of new dataset
        if dataview.EXTRA_DATASETS:
            last_ds = dataview.EXTRA_DATASETS[-1]
            last_max_x = last_ds["df"]["x_centroid"].max() + last_ds["x_offset"]
        else:
            last_max_x = core.DATA["df"]["x_centroid"].max()
        new_min_x  = new_data["df"]["x_centroid"].min()
        x_offset   = last_max_x - new_min_x + 500.0
        new_data["x_offset"] = x_offset
        dataview.EXTRA_DATASETS.append(new_data)
        n_total = 1 + len(dataview.EXTRA_DATASETS)
        status_msg = f"{n_total} samples loaded"
        print(f"  Added extra dataset: {new_dir} (x_offset={x_offset:.0f} µm)", flush=True)
        return f"Added (x_offset={x_offset:.0f} µm).", False, (extra_ver or 0) + 1, status_msg
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return f"Error: {exc}", no_update, no_update, no_update


@app_layout.app.callback(
    Output("extra-datasets-version", "data",     allow_duplicate=True),
    Output("extra-samples-status",   "children", allow_duplicate=True),
    Input("remove-extra-samples-btn", "n_clicks"),
    State("extra-datasets-version",   "data"),
    prevent_initial_call=True,
)
def remove_extra_samples(n_clicks, extra_ver):
    """Clear all extra datasets, reverting to single-sample view."""
    # The button is re-created (n_clicks=0) on every tissue-info re-render, which
    # requests this callback although nothing was clicked.
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not dataview.EXTRA_DATASETS:
        return no_update, "No extra samples loaded."
    n_removed = len(dataview.EXTRA_DATASETS)
    dataview.EXTRA_DATASETS.clear()
    print(f"  Removed {n_removed} extra dataset(s).", flush=True)
    return (extra_ver or 0) + 1, "Extra samples removed."


@app_layout.app.callback(
    Output("cluster-method", "options"),
    Output("cluster-method", "value"),
    Input("dataset-version", "data"),
    State("cluster-method", "value"),
    prevent_initial_call=True,
)
def update_cluster_options(_, current):
    methods = core.DATA["cluster_methods"]
    opts = [{"label": ui.method_label(m), "value": m} for m in methods]
    # dataset-version is also bumped by "Rerun UMAP": keep the user's selection
    # whenever it still exists in the (possibly new) dataset
    if current in methods:
        return opts, current
    return opts, (methods[0] if methods else None)


@app_layout.app.callback(
    Output("cache-clean-modal",        "is_open"),
    Output("cache-clean-confirm-text", "children"),
    Input("cache-clean-open-btn",      "n_clicks"),
    Input("cache-clean-cancel-btn",    "n_clicks"),
    Input("cache-clean-confirm-btn",   "n_clicks"),
    prevent_initial_call=True,
)
def toggle_cache_clean_modal(open_clicks, cancel_clicks, confirm_clicks):
    tid = callback_context.triggered_id
    if tid == "cache-clean-open-btn" and open_clicks:
        size = caching._cache_size_str(max_age=0)
        dataset_name = caching._dataset_name()
        _busy = _running_jobs()
        if _busy:
            return True, (f"Still running: {', '.join(_busy)}. The cache cannot be cleaned "
                          f"while a job is reading from or writing to it - wait for it to "
                          f"finish. (Confirm will do nothing.)")
        return True, (f"Delete all cache entries for dataset '{dataset_name}'? "
                      f"Total cache size: {size}. This cannot be undone.")
    return False, no_update


@app_layout.app.callback(
    Output("cache-clean-modal",  "is_open",  allow_duplicate=True),
    Output("cache-clean-done",   "data"),
    Output("cache-size-display", "children"),
    Output("seg-source",              "value", allow_duplicate=True),
    Output("baysor-done",             "data",  allow_duplicate=True),
    Output("proseg-done",             "data",  allow_duplicate=True),
    Output("reseg-patches-confirmed", "data",  allow_duplicate=True),
    Input("cache-clean-confirm-btn", "n_clicks"),
    State("cache-clean-done",        "data"),
    State("baysor-done",             "data"),
    State("proseg-done",             "data"),
    prevent_initial_call=True,
)
def do_clean_cache(n_clicks, version, baysor_ver, proseg_ver):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    # A running job reads from / writes to the directories this would delete
    # (run directories, the SpatialData store, annotation caches): refuse, as
    # load_sample does. Nothing is deleted and nothing is reset.
    _busy = _running_jobs()
    if _busy:
        print(f"  Cache clean refused - still running: {', '.join(_busy)}", flush=True)
        return (False, no_update, f"Cache not cleaned - still running: {', '.join(_busy)}",
                no_update, no_update, no_update, no_update)
    caching._clean_cache_for_dataset()
    # The cached runs and the patch store are gone from disk: drop the in-memory
    # handles to them and rebuild the seg-source dropdown (as confirm_seg_delete does).
    with core._baysor_lock:
        if core._baysor_state.get("status") != "running":
            core._baysor_state.update({"status": "idle", "message": "", "result": None})
    with core._proseg_lock:
        if core._proseg_state.get("status") != "running":
            core._proseg_state.update({"status": "idle", "message": "", "result": None})
    with core._sdata_lock:
        if core._sdata_state.get("status") != "running":
            core._sdata_state.update({"status": "idle", "message": "", "sdata": None,
                                  "roi": None, "patches": None})
            core.DATA.pop("sdata", None)
    new_size = caching._cache_size_str(max_age=0)
    return (False, (version or 0) + 1, f"Cache: {new_size}",
            "xenium", (baysor_ver or 0) + 1, (proseg_ver or 0) + 1, 0)


# ─── Save as SpatialData callbacks ────────────────────────────────────────────
@app_layout.app.callback(
    Output("save-sdata-modal", "is_open"),
    Input("save-sdata-open-btn",  "n_clicks"),
    Input("save-sdata-close-btn", "n_clicks"),
    State("save-sdata-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_save_sdata_modal(open_clicks, close_clicks, is_open):
    if callback_context.triggered_id == "save-sdata-open-btn" and open_clicks:
        return True
    if callback_context.triggered_id == "save-sdata-close-btn":
        return False
    return is_open


@app_layout.app.callback(
    Output("save-sdata-status",  "children"),
    Output("save-sdata-poll",    "disabled"),
    Output("save-sdata-run-btn", "disabled"),
    Input("save-sdata-run-btn",  "n_clicks"),
    State("save-sdata-dir",      "value"),
    State("save-sdata-name",     "value"),
    State("save-sdata-roi-only", "value"),
    State("seg-source",          "value"),
    prevent_initial_call=True,
)
def start_save_sdata(n_clicks, out_dir, fname, roi_only_val, seg_source):
    out_dir = (out_dir or "").strip()
    fname   = (fname or "").strip()
    if not out_dir:
        return "Enter an output directory.", True, False
    if not fname:
        fname = "analysis.zarr"
    if not fname.endswith(".zarr"):
        fname += ".zarr"
    save_path  = os.path.join(out_dir, fname)
    roi_only   = "roi_only" in (roi_only_val or [])
    seg_source = seg_source or "xenium"
    with core._save_sdata_lock:
        if core._save_sdata_state["status"] == "running":
            return "Already saving…", False, True
        core._save_sdata_state.update({"status": "running", "message": f"Writing to {save_path}…"})
    jobs._job_start("save", f"Save as SpatialData: {os.path.basename(save_path.rstrip('/'))}",
               sdata_store._save_sdata_to_disk, args=(save_path, seg_source, roi_only),
               state=core._save_sdata_state, lock=core._save_sdata_lock)
    return f"Writing to {save_path}…", False, True


@app_layout.app.callback(
    Output("save-sdata-status",  "children",  allow_duplicate=True),
    Output("save-sdata-poll",    "disabled",  allow_duplicate=True),
    Output("save-sdata-run-btn", "disabled",  allow_duplicate=True),
    Output("save-sdata-modal",   "is_open",   allow_duplicate=True),
    Input("save-sdata-poll",     "n_intervals"),
    State("save-sdata-modal",    "is_open"),
    prevent_initial_call=True,
)
def poll_save_sdata(_, modal_open):
    with core._save_sdata_lock:
        status  = core._save_sdata_state["status"]
        message = core._save_sdata_state["message"]
    if status == "running":
        return message, False, True, modal_open
    if status == "done":
        return f"✓ {message}", True, False, False
    if status == "error":
        return f"✗ {message}", True, False, modal_open
    return no_update, True, False, modal_open
