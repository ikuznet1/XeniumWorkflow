"""Callbacks: segmentation-free abundance (marker transcripts per mm² of tissue straight
from the transcript file) in the Cell Type Abundance tab."""

import dash
from dash import no_update
from dash import Input, Output, State

from . import core, jobs, stats, segfree, app_layout, cb_analysis, cb_controls


@app_layout.app.callback(
    Output("sfree-labels", "options"),
    Output("sfree-labels", "value"),
    Input("plots-tabs",      "active_tab"),
    Input("annot-done",      "data"),
    Input("dataset-version", "data"),
    Input("seg-source",      "value"),
    Input("extra-datasets-version", "data"),
    State("color-by",        "value"),
    State("sfree-labels",    "value"),
)
def update_sfree_label_options(active_tab, _annot, _ds, seg_source, _ex, color_by=None, current=None):
    if active_tab != "tab-abundance":
        raise dash.exceptions.PreventUpdate
    opts = [o for o in stats._comp_category_options(seg_source, color_by) if o["value"] != "niche"]
    have = [o["value"] for o in opts]
    want = (f"celltype:{str(color_by).split(':', 1)[1]}" if str(color_by or "").startswith("cell_type:") else None)
    first = want if want in have else next((v for v in have if v.startswith("celltype:")), have[0] if have else None)
    return opts, (no_update if current in have else first)


@app_layout.app.callback(
    Output("sfree-type", "options"),
    Output("sfree-type", "value"),
    Input("sfree-labels",    "value"),
    Input("dataset-version", "data"),
    Input("annot-done",      "data"),
    Input("subset-version",  "data"),
    State("seg-source",      "value"),
    State("cluster-method",  "value"),
    State("sfree-type",      "value"),
)
def update_sfree_type_options(kind, _ds, _annot, _sub, seg_source, method, current=None):
    opts = cb_controls._type_options(kind, seg_source, method, None)
    have = [o["value"] for o in opts]
    return opts, (no_update if current in have else None)


@app_layout.app.callback(
    Output("sfree-run-btn", "disabled"),
    Output("sfree-poll",    "disabled"),
    Output("sfree-status",  "children"),
    Input("sfree-run-btn",  "n_clicks"),
    State("sfree-labels",    "value"),
    State("sfree-type",      "value"),
    State("sfree-markers",   "value"),
    State("sfree-n-markers", "value"),
    State("sfree-bins",      "value"),
    State("sfree-area",      "value"),
    State("sfree-hole",      "value"),
    State("sfree-nnull",     "value"),
    State("seg-source",      "value"),
    State("cluster-method",  "value"),
    State("counts-mode-store", "data"),
    prevent_initial_call=True,
)
def run_segfree(n_clicks, kind, cell_type, markers, n_markers, bins, area, hole, n_null,
                seg_source, method, counts_mode):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not cell_type and not str(markers or "").strip():
        return False, no_update, ("Pick a cell type (its markers are derived from the annotation), or "
                                  "list marker genes.")
    try:
        segfree._sfree_parse_bins(bins)
    except ValueError as exc:
        return False, no_update, str(exc)
    if not cb_analysis._an_job_claim(segfree._sfree_state, segfree._sfree_lock):
        return True, False, "Segmentation-free abundance is already being computed."
    jobs._job_start("segfree", "Segmentation-free abundance", segfree._run_segfree,
                    args=(seg_source or "xenium", method, kind, cell_type, markers, n_markers,
                          counts_mode == "corrected", bins, area, hole, n_null),
                    state=segfree._sfree_state, lock=segfree._sfree_lock)
    return True, False, "Running…"


@app_layout.app.callback(
    Output("sfree-status",  "children", allow_duplicate=True),
    Output("sfree-run-btn", "disabled", allow_duplicate=True),
    Output("sfree-done",    "data"),
    Output("sfree-poll",    "disabled", allow_duplicate=True),
    Input("sfree-poll",     "n_intervals"),
    State("sfree-done",     "data"),
    prevent_initial_call=True,
)
def poll_segfree(_, seen=None):
    with segfree._sfree_lock:
        st = segfree._sfree_state
        status, msg, ver = st["status"], st["message"], st["version"]
    if status == "running":
        return msg, True, no_update, False
    if ver != seen:
        return msg, False, ver, True
    return msg, False, no_update, True


@app_layout.app.callback(
    Output("jobs-poll", "disabled", allow_duplicate=True),
    Input("sfree-poll", "disabled"),
    State("jobs-poll",  "disabled"),
    prevent_initial_call=True,
)
def wake_jobs_poll_segfree(_sf, poll_disabled):
    if jobs._jobs_panel_rows() and poll_disabled:
        return False
    return no_update


@app_layout.app.callback(
    Output("sfree-plot",  "figure"),
    Output("sfree-table", "data"),
    Input("sfree-done",      "data"),
    Input("dataset-version", "data"),
    Input("sfree-plot-w",    "value"),
    Input("sfree-plot-h",    "value"),
)
def update_sfree_results(_done, _ds, w, h):
    with segfree._sfree_lock:
        res = segfree._sfree_state.get("result")
    return segfree.make_sfree_fig(width=w, height=h), segfree._sfree_records(res)


@app_layout.app.callback(
    Output("sfree-download", "data"),
    Input("sfree-download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_segfree(n_clicks):
    return cb_controls._download(n_clicks, segfree._sfree_state, segfree._sfree_lock, segfree._sfree_csv,
                                 "segmentation_free_abundance.csv")
