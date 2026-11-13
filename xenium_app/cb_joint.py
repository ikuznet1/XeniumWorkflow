"""Callbacks: joint clustering across samples (with Harmony when available)."""

import dash
from dash import no_update
from dash import Input, Output, State

from . import jobs, stats, joint, app_layout, cb_analysis


@app_layout.app.callback(
    Output("jc-design-summary", "children"),
    Output("jc-integrate",      "options"),
    Input("plots-tabs",             "active_tab"),
    Input("design-version",         "data"),
    Input("extra-datasets-version", "data"),
    Input("dataset-version",        "data"),
)
def update_jc_design_summary(active_tab, _design, _extra, _ds):
    if active_tab != "tab-joint":
        raise dash.exceptions.PreventUpdate
    n = len(stats._design_samples())
    head = (f"{n} samples loaded. " if n >= 2 else
            "One sample loaded: joint clustering needs at least two (Load Sample → add a sample). ")
    if joint._jc_harmony_module() is None:
        head += joint._JC_HARMONY_HINT + ". "
    return head + "Samples and conditions: " + stats._design_summary(), joint._jc_integrate_options()


@app_layout.app.callback(
    Output("design-modal",  "is_open",  allow_duplicate=True),
    Output("design-rows",   "children", allow_duplicate=True),
    Output("design-name-a", "value",    allow_duplicate=True),
    Output("design-name-b", "value",    allow_duplicate=True),
    Output("design-status", "children", allow_duplicate=True),
    Input("jc-design-open-btn", "n_clicks"),
    prevent_initial_call=True,
)
def open_design_modal_joint(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    names = stats._design_names()
    return True, stats._design_rows_children(), names["A"], names["B"], ""


@app_layout.app.callback(
    Output("jc-run-btn", "disabled"),
    Output("jc-poll",    "disabled"),
    Output("jc-status",  "children"),
    Input("jc-run-btn",  "n_clicks"),
    State("jc-integrate", "value"),
    State("jc-nhvg",     "value"),
    State("jc-npcs",     "value"),
    State("jc-nn",       "value"),
    State("jc-res",      "value"),
    State("jc-seed",     "value"),
    State("jc-umap",     "value"),
    State("seg-source",  "value"),
    State("counts-mode-store", "data"),
    prevent_initial_call=True,
)
def run_joint(n_clicks, integrate, n_hvg, n_pcs, n_nn, resolution, seed, umap, seg_source, counts_mode):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    for v, name in ((n_hvg, "number of genes"), (n_pcs, "number of PCs"), (n_nn, "number of neighbours"),
                    (resolution, "resolution")):
        if v is None or v <= 0:
            return False, no_update, f"The {name} must be positive."
    if len(stats._design_samples()) < 2:
        return False, no_update, ("Joint clustering needs at least two loaded samples "
                                  "(Load Sample → add a sample).")
    if not cb_analysis._an_job_claim(joint._jc_state, joint._jc_lock):
        return True, False, "Joint clustering is already running."
    params = {"integrate": integrate or "sample", "n_hvg": n_hvg, "n_pcs": n_pcs, "n_neighbors": n_nn,
              "resolution": resolution, "seed": seed, "umap": "on" in (umap or [])}
    jobs._job_start("joint", "Joint clustering", joint._run_joint,
                    args=(seg_source or "xenium", counts_mode == "corrected", params),
                    state=joint._jc_state, lock=joint._jc_lock)
    return True, False, "Running…"


@app_layout.app.callback(
    Output("jc-status",          "children", allow_duplicate=True),
    Output("jc-run-btn",         "disabled", allow_duplicate=True),
    Output("jc-done",            "data"),
    Output("jc-poll",            "disabled", allow_duplicate=True),
    Output("clusterset-version", "data",     allow_duplicate=True),
    Output("geneset-version",    "data",     allow_duplicate=True),
    Input("jc-poll",             "n_intervals"),
    State("jc-done",             "data"),
    State("clusterset-version",  "data"),
    State("geneset-version",     "data"),
    prevent_initial_call=True,
)
def poll_joint(_, seen=None, cs_version=None, gs_version=None):
    """The status while the job runs; when it ends, the result version, a bump of the
    cluster-set version (the Cluster Set options) and of the version both renderers redraw
    per-cell values for (a set of the same name may have been replaced)."""
    with joint._jc_lock:
        st = joint._jc_state
        status, msg, ver = st["status"], st["message"], st["version"]
    if status == "running":
        return msg, True, no_update, False, no_update, no_update
    if ver != seen:
        ok = status == "done"
        return (msg, False, ver, True, (cs_version or 0) + 1 if ok else no_update,
                (gs_version or 0) + 1 if ok else no_update)
    return no_update, False, no_update, True, no_update, no_update


@app_layout.app.callback(
    Output("jobs-poll", "disabled", allow_duplicate=True),
    Input("jc-poll",    "disabled"),
    State("jobs-poll",  "disabled"),
    prevent_initial_call=True,
)
def wake_jobs_poll_joint(_jc, poll_disabled):
    if jobs._jobs_panel_rows() and poll_disabled:
        return False
    return no_update


@app_layout.app.callback(
    Output("jc-plot",  "figure"),
    Output("jc-table", "data"),
    Output("jc-table", "columns"),
    Input("jc-done",         "data"),
    Input("dataset-version", "data"),
    Input("jc-plot-w",       "value"),
    Input("jc-plot-h",       "value"),
)
def update_jc_results(_done, _ds, w, h):
    res = joint._jc_result()
    return joint.make_jc_fig(width=w, height=h), joint._jc_records(res), joint._jc_columns(res)


@app_layout.app.callback(
    Output("jc-download", "data"),
    Input("jc-download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_joint(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    res = joint._jc_result()
    if not res:
        raise dash.exceptions.PreventUpdate
    return dict(content=joint._jc_csv(res), filename="joint_clusters.csv")
