"""Callbacks: ligand-receptor exposure decomposition (second mode of the Ligand–Receptor tab)."""

import dash
from dash import no_update
from dash import Input, Output, State

from . import jobs, stats, lrdecomp, app_layout, cb_analysis


@app_layout.app.callback(
    Output("lr-scores-controls", "style"),
    Output("lr-scores-results",  "style"),
    Output("lrd-controls",       "style"),
    Output("lrd-results",        "style"),
    Input("lr-mode", "value"),
    State("lr-scores-controls", "style"),
    State("lr-scores-results",  "style"),
)
def toggle_lr_mode(mode, row_style, wrap_style):
    """Show the controls and results of the chosen mode of the Ligand–Receptor tab."""
    row = {**(row_style or {}), "display": "flex"}
    wrap = {**(wrap_style or {}), "display": "flex"}
    hide = {"display": "none"}
    if mode == "decomp":
        return {**row, **hide}, {**wrap, **hide}, {"flexShrink": "0"}, wrap
    return row, wrap, {"display": "none", "flexShrink": "0"}, {**wrap, **hide}


@app_layout.app.callback(
    Output("design-modal",  "is_open", allow_duplicate=True),
    Output("design-rows",   "children", allow_duplicate=True),
    Output("design-name-a", "value", allow_duplicate=True),
    Output("design-name-b", "value", allow_duplicate=True),
    Output("design-status", "children", allow_duplicate=True),
    Input("lrd-design-open-btn", "n_clicks"),
    prevent_initial_call=True,
)
def open_design_modal_lrd(n_clicks):
    """Open the sample → condition editor from the decomposition controls."""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    names = stats._design_names()
    return True, stats._design_rows_children(), names["A"], names["B"], ""


@app_layout.app.callback(
    Output("lrd-design-summary", "children"),
    Input("design-version",         "data"),
    Input("extra-datasets-version", "data"),
    Input("dataset-version",        "data"),
)
def update_lrd_design_summary(_design, _extra, _ds):
    return stats._design_summary()


@app_layout.app.callback(
    Output("lrd-run-btn", "disabled"),
    Output("lrd-poll",    "disabled"),
    Output("lrd-status",  "children"),
    Input("lrd-run-btn",  "n_clicks"),
    State("lr-pairs",      "value"),
    State("lr-labels",     "value"),
    State("lrd-radius",    "value"),
    State("lrd-tile",      "value"),
    State("lrd-own",       "value"),
    State("lrd-floor",     "value"),
    State("lrd-min-cells", "value"),
    State("lrd-rates",     "value"),
    State("seg-source",    "value"),
    State("cluster-method", "value"),
    State("counts-mode-store", "data"),
    prevent_initial_call=True,
)
def run_lrdecomp(n_clicks, pairs_text, label_kind, radius, tile, own, floor, min_cells, rates,
                 seg_source, method, counts_mode):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    pairs, _skipped = stats._lr_parse_pairs(pairs_text)
    if not pairs:
        return False, True, ("No ligand–receptor pairs: paste or upload lines of 'ligand,receptor' "
                             "(no database is bundled or downloaded).")
    if not label_kind:
        return False, True, "Pick the cell types (or clusters) to decompose between."
    if radius is not None and radius <= 0:
        return False, True, "The contact radius must be positive."
    if not cb_analysis._an_job_claim(lrdecomp._lrd_state, lrdecomp._lrd_lock):
        return True, False, "The exposure decomposition is already running."
    jobs._job_start("lrdecomp", "Ligand–receptor exposure decomposition", lrdecomp._run_lrdecomp,
                    args=(pairs_text, label_kind, seg_source or "xenium", method, counts_mode == "corrected",
                          radius, tile, own, floor, rates == "binary", min_cells),
                    state=lrdecomp._lrd_state, lock=lrdecomp._lrd_lock)
    return True, False, "Running…"


@app_layout.app.callback(
    Output("lrd-status",  "children", allow_duplicate=True),
    Output("lrd-poll",    "disabled", allow_duplicate=True),
    Output("lrd-run-btn", "disabled", allow_duplicate=True),
    Output("lrd-done",    "data"),
    Input("lrd-poll",     "n_intervals"),
    State("lrd-done",     "data"),
    prevent_initial_call=True,
)
def poll_lrdecomp(_, done_ver):
    with lrdecomp._lrd_lock:
        status, msg = lrdecomp._lrd_state["status"], lrdecomp._lrd_state["message"]
    if status == "done":
        return msg, True, False, (done_ver or 0) + 1
    if status in ("error", "idle"):
        return msg, True, False, no_update
    return msg, False, True, no_update


@app_layout.app.callback(
    Output("jobs-poll", "disabled", allow_duplicate=True),
    Input("lrd-poll",   "disabled"),
    State("jobs-poll",  "disabled"),
    prevent_initial_call=True,
)
def wake_jobs_poll_lrdecomp(_lrd, poll_disabled):
    if jobs._jobs_panel_rows() and poll_disabled:
        return False
    return no_update


@app_layout.app.callback(
    Output("lrd-plot",  "figure"),
    Output("lrd-table", "columns"),
    Output("lrd-table", "data"),
    Input("lrd-done",        "data"),
    Input("dataset-version", "data"),
    Input("lrd-level",       "value"),
    Input("lrd-top-n",       "value"),
    Input("lrd-plot-w",      "value"),
    Input("lrd-plot-h",      "value"),
)
def update_lrd_results(_done, _ds_version, level, top_n, w, h):
    res = lrdecomp._lrd_result()
    return (lrdecomp.make_lrd_fig(level=level, top_n=top_n, width=w, height=h),
            lrdecomp._lrd_columns(res, level), lrdecomp._lrd_records(res, level))


@app_layout.app.callback(
    Output("lrd-download", "data"),
    Input("lrd-download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_lrd_table(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    res = lrdecomp._lrd_result()
    if not res:
        raise dash.exceptions.PreventUpdate
    return dict(content=lrdecomp._lrd_csv(res), filename="ligand_receptor_exposure_decomposition.csv")
