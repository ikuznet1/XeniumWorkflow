"""Callbacks: confound controls (compartments, spillover, gene ownership), segmentation QC,
stratified co-localisation, the annotation label report."""

import dash
from dash import no_update
from dash import Input, Output, State

from . import core, jobs, analysis, stats, controls, app_layout, cb_analysis, cb_spatial

_JOBS = (("comp", controls._comp_ctl_state, controls._comp_ctl_lock),
         ("spill", controls._spill_state, controls._spill_lock),
         ("segqc", controls._segqc_state, controls._segqc_lock),
         ("coloc", controls._coloc_state, controls._coloc_lock))


def _type_options(kind, seg_source, method, first) -> list:
    """Options of a cell-type picker for the label source *kind* of the active view."""
    opts = [dict(first)] if first else []
    if not kind:
        return opts
    try:
        tables = stats._view_tables(seg_source)
        labels, _cat = stats._comp_labels(kind, tables, seg_source, method)
    except ValueError:
        return opts
    vc = controls._pooled_types(labels)
    return opts + [{"label": f"{t} ({int(n):,} cells)", "value": str(t)} for t, n in vc.items()]


# ─── label / type pickers ─────────────────────────────────────────────────────
@app_layout.app.callback(
    Output("ctl-labels",   "options"),
    Output("ctl-labels",   "value"),
    Output("coloc-labels", "options"),
    Output("coloc-labels", "value"),
    Output("segqc-method", "options"),
    Output("segqc-method", "value"),
    Input("plots-tabs",      "active_tab"),
    Input("annot-done",      "data"),
    Input("niche-done",      "data"),
    Input("dataset-version", "data"),
    Input("seg-source",      "value"),
    Input("extra-datasets-version", "data"),
    State("color-by",        "value"),
    State("ctl-labels",      "value"),
    State("coloc-labels",    "value"),
    State("segqc-method",    "value"),
)
def update_control_label_options(active_tab, _annot, _niche, _ds, seg_source, _ex, color_by=None,
                                 cur_ctl=None, cur_coloc=None, cur_qc=None):
    if active_tab not in ("tab-controls", "tab-coloc", "tab-segqc"):
        raise dash.exceptions.PreventUpdate
    opts = [o for o in stats._comp_category_options(seg_source, color_by) if o["value"] != "niche"]
    have = [o["value"] for o in opts]
    want = (f"celltype:{str(color_by).split(':', 1)[1]}" if str(color_by or "").startswith("cell_type:") else None)
    first = want if want in have else next((v for v in have if v.startswith("celltype:")), have[0] if have else None)
    with core._annot_lock:
        qc = [{"label": core._ANNOT_METHODS[m], "value": m} for m in core._ANNOT_METHODS
              if core._annot_state.get(f"labels_{m}") is not None]
    qc_have = [o["value"] for o in qc]
    return (opts, no_update if cur_ctl in have else first,
            opts, no_update if cur_coloc in have else first,
            qc, no_update if cur_qc in qc_have else (qc_have[0] if qc_have else None))


@app_layout.app.callback(
    Output("ctl-dominant", "options"),
    Output("spill-source", "options"),
    Input("ctl-labels",      "value"),
    Input("dataset-version", "data"),
    Input("annot-done",      "data"),
    State("seg-source",      "value"),
    State("cluster-method",  "value"),
)
def update_control_type_options(kind, _ds, _annot, seg_source, method):
    return (_type_options(kind, seg_source, method, {"label": "auto (most abundant type)", "value": "auto"}),
            _type_options(kind, seg_source, method,
                          {"label": "auto (type with the most transcripts)", "value": "auto"}))


@app_layout.app.callback(
    Output("coloc-placebo", "options"),
    Input("coloc-labels",    "value"),
    Input("dataset-version", "data"),
    Input("annot-done",      "data"),
    State("seg-source",      "value"),
    State("cluster-method",  "value"),
)
def update_coloc_placebo_options(kind, _ds, _annot, seg_source, method):
    return _type_options(kind, seg_source, method, None)


@app_layout.app.callback(
    Output("segqc-source", "options"),
    Input("segqc-method",    "value"),
    Input("dataset-version", "data"),
    Input("annot-done",      "data"),
)
def update_segqc_source_options(method, _ds, _annot):
    return _type_options(f"celltype:{method}" if method else None, "xenium", None,
                         {"label": "auto (type with the most transcripts)", "value": "auto"})


@app_layout.app.callback(
    Output("coloc-design-summary", "children"),
    Input("design-version",         "data"),
    Input("extra-datasets-version", "data"),
    Input("dataset-version",        "data"),
)
def update_coloc_design_summary(_design, _extra, _ds):
    return stats._design_summary()


# ─── run buttons ──────────────────────────────────────────────────────────────
@app_layout.app.callback(
    Output("ctl-comp-run-btn", "disabled"),
    Output("ctl-poll",         "disabled"),
    Output("ctl-comp-status",  "children"),
    Input("ctl-comp-run-btn",  "n_clicks"),
    State("ctl-labels",    "value"),
    State("ctl-dominant",  "value"),
    State("ctl-bin",       "value"),
    State("ctl-close",     "value"),
    State("ctl-hole",      "value"),
    State("ctl-radius",    "value"),
    State("ctl-core-nb",   "value"),
    State("ctl-core-frac", "value"),
    State("ctl-core-edge", "value"),
    State("seg-source",    "value"),
    State("cluster-method", "value"),
    prevent_initial_call=True,
)
def run_compartments(n_clicks, kind, dominant, bin_um, close, hole, radius, nb, frac, edge,
                     seg_source, method):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not bin_um or bin_um <= 0 or not radius or radius <= 0:
        return False, no_update, "The bin size and the radius must be positive."
    if not cb_analysis._an_job_claim(controls._comp_ctl_state, controls._comp_ctl_lock):
        return True, False, "The compartments are already being computed."
    params = {"bin_um": bin_um, "close_bins": close, "hole_max_bins": hole, "radius": radius,
              "min_nb": nb, "min_frac": frac, "min_edge": edge}
    jobs._job_start("compartments", "Tissue compartments", controls._run_compartments,
                    args=(seg_source or "xenium", method, kind, dominant, params),
                    state=controls._comp_ctl_state, lock=controls._comp_ctl_lock)
    return True, False, "Running…"


@app_layout.app.callback(
    Output("spill-run-btn", "disabled"),
    Output("ctl-poll",      "disabled", allow_duplicate=True),
    Output("spill-status",  "children"),
    Input("spill-run-btn",  "n_clicks"),
    State("ctl-labels",      "value"),
    State("spill-source",    "value"),
    State("spill-markers",   "value"),
    State("spill-n-markers", "value"),
    State("spill-own-frac",  "value"),
    State("seg-source",      "value"),
    State("cluster-method",  "value"),
    State("counts-mode-store", "data"),
    prevent_initial_call=True,
)
def run_spillover(n_clicks, kind, source, markers, n_markers, own_frac, seg_source, method, counts_mode):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not kind:
        return False, no_update, "Pick where the cell types come from (an annotation, or clusters)."
    if not cb_analysis._an_job_claim(controls._spill_state, controls._spill_lock):
        return True, False, "The spillover score is already being computed."
    jobs._job_start("spillover", "Spillover score and gene ownership", controls._run_spillover,
                    args=(seg_source or "xenium", method, kind, source, markers, n_markers,
                          counts_mode == "corrected", own_frac),
                    state=controls._spill_state, lock=controls._spill_lock)
    return True, False, "Running…"


@app_layout.app.callback(
    Output("segqc-run-btn", "disabled"),
    Output("ctl-poll",      "disabled", allow_duplicate=True),
    Output("segqc-status",  "children"),
    Input("segqc-run-btn",  "n_clicks"),
    State("segqc-method",  "value"),
    State("segqc-sets",    "value"),
    State("segqc-source",  "value"),
    State("segqc-radius",  "value"),
    prevent_initial_call=True,
)
def run_segqc(n_clicks, method, sets_text, source, radius):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not method:
        return False, no_update, ("The Xenium segmentation has no cell-type annotation: run Cell Type "
                                  "Annotation first.")
    if not cb_analysis._an_job_claim(controls._segqc_state, controls._segqc_lock):
        return True, False, "Segmentation QC is already running."
    jobs._job_start("segqc", "Segmentation QC", controls._run_segqc,
                    args=(method, sets_text, source, radius),
                    state=controls._segqc_state, lock=controls._segqc_lock)
    return True, False, "Running…"


@app_layout.app.callback(
    Output("coloc-run-btn", "disabled"),
    Output("ctl-poll",      "disabled", allow_duplicate=True),
    Output("coloc-status",  "children"),
    Input("coloc-run-btn",  "n_clicks"),
    State("coloc-labels",  "value"),
    State("coloc-radius",  "value"),
    State("coloc-nperm",   "value"),
    State("coloc-placebo", "value"),
    State("seg-source",    "value"),
    State("cluster-method", "value"),
    prevent_initial_call=True,
)
def run_coloc(n_clicks, kind, radius, n_perm, placebo, seg_source, method):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not kind:
        return False, no_update, "Pick the cell types (or clusters) to test between."
    if not cb_analysis._an_job_claim(controls._coloc_state, controls._coloc_lock):
        return True, False, "Stratified co-localisation is already running."
    jobs._job_start("coloc", "Stratified co-localisation", controls._run_coloc,
                    args=(seg_source or "xenium", method, kind, radius, n_perm, placebo),
                    state=controls._coloc_state, lock=controls._coloc_lock)
    return True, False, "Running…"


# ─── one poll for the four jobs ───────────────────────────────────────────────
@app_layout.app.callback(
    Output("ctl-comp-status", "children", allow_duplicate=True),
    Output("spill-status",    "children", allow_duplicate=True),
    Output("segqc-status",    "children", allow_duplicate=True),
    Output("coloc-status",    "children", allow_duplicate=True),
    Output("ctl-comp-run-btn", "disabled", allow_duplicate=True),
    Output("spill-run-btn",    "disabled", allow_duplicate=True),
    Output("segqc-run-btn",    "disabled", allow_duplicate=True),
    Output("coloc-run-btn",    "disabled", allow_duplicate=True),
    Output("ctl-comp-done", "data"),
    Output("spill-done",    "data"),
    Output("segqc-done",    "data"),
    Output("coloc-done",    "data"),
    Output("ctl-poll",      "disabled", allow_duplicate=True),
    Input("ctl-poll",       "n_intervals"),
    State("ctl-comp-done", "data"),
    State("spill-done",    "data"),
    State("segqc-done",    "data"),
    State("coloc-done",    "data"),
    prevent_initial_call=True,
)
def poll_controls(_, comp_done=None, spill_done=None, segqc_done=None, coloc_done=None):
    done = (comp_done, spill_done, segqc_done, coloc_done)
    msgs, btns, vers, running = [], [], [], False
    for (name, state, lock), seen in zip(_JOBS, done):
        with lock:
            status, msg, ver = state["status"], state["message"], state["version"]
        if status == "running":
            running = True
            msgs.append(msg); btns.append(True); vers.append(no_update)
        else:
            btns.append(False)
            if ver != seen:
                msgs.append(msg); vers.append(ver)
            else:
                msgs.append(no_update); vers.append(no_update)
    return (*msgs, *btns, *vers, not running)


@app_layout.app.callback(
    Output("jobs-poll", "disabled", allow_duplicate=True),
    Input("ctl-poll",   "disabled"),
    State("jobs-poll",  "disabled"),
    prevent_initial_call=True,
)
def wake_jobs_poll_controls(_ctl, poll_disabled):
    if jobs._jobs_panel_rows() and poll_disabled:
        return False
    return no_update


# ─── results ──────────────────────────────────────────────────────────────────
@app_layout.app.callback(
    Output("color-by", "options", allow_duplicate=True),
    Input("ctl-comp-done", "data"),
    Input("spill-done",    "data"),
    State("color-by",      "options"),
    prevent_initial_call=True,
)
def update_color_by_for_controls(_comp, _spill, current_options):
    return cb_spatial._refresh_colorby_options(current_options)


@app_layout.app.callback(
    Output("spill-plot",      "figure"),
    Output("spill-own-table", "data"),
    Input("spill-done",      "data"),
    Input("dataset-version", "data"),
    Input("spill-plot-w",    "value"),
    Input("spill-plot-h",    "value"),
)
def update_spill_results(_done, _ds, w, h):
    with controls._spill_lock:
        res = controls._spill_state.get("result")
    return controls.make_spill_fig(width=w, height=h), controls._ownership_records(res)


@app_layout.app.callback(
    Output("segqc-plot",  "figure"),
    Output("segqc-table", "data"),
    Input("segqc-done",      "data"),
    Input("dataset-version", "data"),
    Input("segqc-plot-w",    "value"),
    Input("segqc-plot-h",    "value"),
)
def update_segqc_results(_done, _ds, w, h):
    with controls._segqc_lock:
        res = controls._segqc_state.get("result")
    return controls.make_segqc_fig(width=w, height=h), controls._segqc_records(res)


@app_layout.app.callback(
    Output("coloc-plot",          "figure"),
    Output("coloc-table",         "columns"),
    Output("coloc-table",         "data"),
    Output("coloc-placebo-table", "data"),
    Input("coloc-done",      "data"),
    Input("dataset-version", "data"),
    Input("coloc-plot-w",    "value"),
    Input("coloc-plot-h",    "value"),
)
def update_coloc_results(_done, _ds, w, h):
    with controls._coloc_lock:
        res = controls._coloc_state.get("result")
    return (controls.make_coloc_fig(width=w, height=h), controls._coloc_columns(res),
            controls._coloc_records(res), controls._placebo_records(res))


def _download(n_clicks, state, lock, builder, filename):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    with lock:
        res = state.get("result")
    if not res:
        raise dash.exceptions.PreventUpdate
    return dict(content=builder(res), filename=filename)


@app_layout.app.callback(
    Output("spill-download", "data"),
    Input("spill-download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_ownership(n_clicks):
    return _download(n_clicks, controls._spill_state, controls._spill_lock, controls._spill_csv,
                     "gene_ownership_spillover.csv")


@app_layout.app.callback(
    Output("segqc-download", "data"),
    Input("segqc-download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_segqc(n_clicks):
    return _download(n_clicks, controls._segqc_state, controls._segqc_lock, controls._segqc_csv,
                     "segmentation_qc.csv")


@app_layout.app.callback(
    Output("coloc-download", "data"),
    Input("coloc-download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_coloc(n_clicks):
    return _download(n_clicks, controls._coloc_state, controls._coloc_lock, controls._coloc_csv,
                     "stratified_colocalisation.csv")


# ─── annotation dialog: label counts and lineage collapse ─────────────────────
@app_layout.app.callback(
    Output("annot-label-report", "children"),
    Input("annot-modal",     "is_open"),
    Input("annot-done",      "data"),
    Input("dataset-version", "data"),
    State("seg-source",      "value"),
)
def update_annot_label_report(is_open, _done, _ds, seg_source):
    if not is_open:
        raise dash.exceptions.PreventUpdate
    try:
        return controls._annot_label_report(seg_source or "xenium")
    except Exception as exc:
        return f"Label report not available: {str(exc)[:200]}"
