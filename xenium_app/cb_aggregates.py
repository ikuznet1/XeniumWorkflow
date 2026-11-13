"""Callbacks: aggregates (connected groups of cells of a chosen class)."""

import dash
from dash import no_update
from dash import Input, Output, State

from . import jobs, stats, aggregates, app_layout, cb_analysis, cb_controls


@app_layout.app.callback(
    Output("agg-labels", "options"),
    Output("agg-labels", "value"),
    Input("plots-tabs",      "active_tab"),
    Input("annot-done",      "data"),
    Input("dataset-version", "data"),
    Input("seg-source",      "value"),
    Input("extra-datasets-version", "data"),
    State("color-by",        "value"),
    State("agg-labels",      "value"),
)
def update_agg_label_options(active_tab, _annot, _ds, seg_source, _ex, color_by=None, current=None):
    if active_tab != "tab-agg":
        raise dash.exceptions.PreventUpdate
    opts = [o for o in stats._comp_category_options(seg_source, color_by) if o["value"] != "niche"]
    have = [o["value"] for o in opts]
    want = (f"celltype:{str(color_by).split(':', 1)[1]}" if str(color_by or "").startswith("cell_type:") else None)
    first = want if want in have else next((v for v in have if v.startswith("celltype:")), have[0] if have else None)
    return opts, no_update if current in have else first


@app_layout.app.callback(
    Output("agg-classes", "options"),
    Output("agg-classes", "value"),
    Input("agg-labels",      "value"),
    Input("dataset-version", "data"),
    Input("annot-done",      "data"),
    Input("seg-source",      "value"),
    Input("cluster-method",  "value"),
    Input("subset-version",  "data"),
    Input("extra-datasets-version", "data"),
    Input("counts-mode-store",      "data"),
    Input("baysor-done",     "data"),
    Input("proseg-done",     "data"),
    State("agg-classes",     "value"),
)
def update_agg_class_options(kind, _ds, _annot, seg_source, method, _sub=None, _ex=None, _cm=None,
                             _bd=None, _pd=None, current=None):
    """The types of the label source in the view as it is now: another segmentation,
    cluster set, sample list, subset or counts mode has other types and counts, so the list
    is rebuilt for each, and a picked type that no longer exists is cleared."""
    opts = cb_controls._type_options(kind, seg_source, method, None)
    have = {o["value"] for o in opts}
    cur = current if isinstance(current, list) else ([current] if current else [])
    keep = [c for c in cur if c in have]
    return opts, (no_update if keep == cur else keep)


@app_layout.app.callback(
    Output("agg-design-summary", "children"),
    Input("design-version",         "data"),
    Input("extra-datasets-version", "data"),
    Input("dataset-version",        "data"),
)
def update_agg_design_summary(_design, _extra, _ds):
    return stats._design_summary()


@app_layout.app.callback(
    Output("agg-run-btn", "disabled"),
    Output("agg-poll",    "disabled"),
    Output("agg-status",  "children"),
    Input("agg-run-btn",  "n_clicks"),
    State("agg-labels",   "value"),
    State("agg-classes",  "value"),
    State("agg-radius",   "value"),
    State("agg-nmin",     "value"),
    State("agg-link",     "value"),
    State("agg-nperm",    "value"),
    State("agg-alpha",    "value"),
    State("agg-minnb",    "value"),
    State("agg-hull",     "value"),
    State("seg-source",   "value"),
    State("cluster-method", "value"),
    prevent_initial_call=True,
)
def run_aggregates(n_clicks, kind, classes, radius, n_min, link, n_perm, alpha, min_nb, hull,
                   seg_source, method):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not kind:
        return False, no_update, "Pick where the cell types come from (an annotation, or clusters)."
    if not classes:
        return False, no_update, "Pick the cell types that make up the class (e.g. the immune types)."
    for v, name in ((radius, "radius"), (link, "link length"), (n_min, "number of class cells"),
                    (alpha, "α")):
        if v is None or v <= 0:
            return False, no_update, f"The {name} must be positive."
    if not cb_analysis._an_job_claim(aggregates._agg_state, aggregates._agg_lock):
        return True, False, "Aggregates are already being computed."
    params = {"radius": radius, "n_min": n_min, "link": link, "n_null": n_perm, "alpha": alpha,
              "min_nb": min_nb, "hull": hull}
    jobs._job_start("aggregates", "Aggregates", aggregates._run_aggregates,
                    args=(seg_source or "xenium", method, kind, list(classes), params),
                    state=aggregates._agg_state, lock=aggregates._agg_lock)
    return True, False, "Running…"


@app_layout.app.callback(
    Output("agg-status",      "children", allow_duplicate=True),
    Output("agg-run-btn",     "disabled", allow_duplicate=True),
    Output("agg-done",        "data"),
    Output("agg-poll",        "disabled", allow_duplicate=True),
    Output("geneset-version", "data",     allow_duplicate=True),
    Input("agg-poll",         "n_intervals"),
    State("agg-done",         "data"),
    State("geneset-version",  "data"),
    prevent_initial_call=True,
)
def poll_agg(_, seen=None, gs_version=None):
    """The status while the job runs; when it ends, the result version - and a bump of the
    version both renderers redraw per-cell values for (Color By option, colours, outlines)."""
    with aggregates._agg_lock:
        st = aggregates._agg_state
        status, msg, ver = st["status"], st["message"], st["version"]
    if status == "running":
        return msg, True, no_update, False, no_update
    if ver != seen:
        return msg, False, ver, True, (gs_version or 0) + 1
    # ended without a new result version (cancelled, or the thread died): the final message,
    # not the last progress text
    return msg, False, no_update, True, no_update


@app_layout.app.callback(
    Output("jobs-poll", "disabled", allow_duplicate=True),
    Input("agg-poll",   "disabled"),
    State("jobs-poll",  "disabled"),
    prevent_initial_call=True,
)
def wake_jobs_poll_aggregates(_agg, poll_disabled):
    if jobs._jobs_panel_rows() and poll_disabled:
        return False
    return no_update


@app_layout.app.callback(
    Output("geneset-version", "data", allow_duplicate=True),
    Input("agg-outline-toggle", "value"),
    State("geneset-version",    "data"),
    prevent_initial_call=True,
)
def toggle_agg_outlines(value, gs_version=None):
    show = "on" in (value or [])
    with aggregates._agg_lock:
        changed = aggregates._agg_state.get("show", True) != show
        aggregates._agg_state["show"] = show
        has = aggregates._agg_state.get("result") is not None
    if not changed or not has:
        raise dash.exceptions.PreventUpdate
    return (gs_version or 0) + 1


@app_layout.app.callback(
    Output("agg-plot",         "figure"),
    Output("agg-table",        "data"),
    Output("agg-sample-table", "data"),
    Input("agg-done",        "data"),
    Input("dataset-version", "data"),
    Input("agg-plot-w",      "value"),
    Input("agg-plot-h",      "value"),
)
def update_agg_results(_done, _ds, w, h):
    res = aggregates._agg_result()
    return (aggregates.make_agg_fig(width=w, height=h), aggregates._agg_records(res),
            aggregates._agg_sample_records(res))


@app_layout.app.callback(
    Output("agg-download", "data"),
    Input("agg-download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_aggregates(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    res = aggregates._agg_result()
    if not res:
        raise dash.exceptions.PreventUpdate
    return dict(content=aggregates._agg_csv(res), filename="aggregates.csv")
