"""Callbacks: Banksy spatial domains."""

import dash
from dash import no_update
from dash import Input, Output, State

from . import jobs, banksy, app_layout, cb_analysis


@app_layout.app.callback(
    Output("bk-lambda",     "value"),
    Output("bk-resolution", "value"),
    Input("bk-preset",  "value"),
    prevent_initial_call=True,
)
def set_banksy_preset(preset):
    """λ and the Leiden resolution of a preset (few large domains / finer cell types)."""
    if preset == "domains":
        return banksy._BK_LAMBDA_DOMAINS, banksy._BK_RESOLUTION
    if preset == "celltype":
        return banksy._BK_LAMBDA_CELLTYPE, banksy._BK_RESOLUTION_CELLTYPE
    return no_update, no_update


@app_layout.app.callback(
    Output("bk-run-btn", "disabled"),
    Output("bk-poll",    "disabled"),
    Output("bk-status",  "children"),
    Input("bk-run-btn",  "n_clicks"),
    State("bk-lambda",     "value"),
    State("bk-kgeom",      "value"),
    State("bk-agf",        "value"),
    State("bk-resolution", "value"),
    State("bk-maxgenes",   "value"),
    State("bk-npcs",       "value"),
    State("bk-kexpr",      "value"),
    State("bk-seed",       "value"),
    State("bk-log1p",      "value"),
    State("seg-source",    "value"),
    State("counts-mode-store", "data"),
    prevent_initial_call=True,
)
def run_banksy(n_clicks, lam, k_geom, agf, resolution, max_genes, n_pcs, k_expr, seed, log1p,
               seg_source, counts_mode):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if lam is None or lam < 0 or lam > 1:
        return False, no_update, "λ must be between 0 and 1."
    for v, name in ((k_geom, "number of spatial neighbours"), (resolution, "resolution"),
                    (max_genes, "gene cap"), (n_pcs, "number of PCs"), (k_expr, "number of graph neighbours")):
        if v is None or v <= 0:
            return False, no_update, f"The {name} must be positive."
    if not cb_analysis._an_job_claim(banksy._bk_state, banksy._bk_lock):
        return True, False, "Banksy is already running."
    params = {"lam": lam, "k_geom": k_geom, "agf": "on" in (agf or []), "resolution": resolution,
              "max_genes": max_genes, "n_pcs": n_pcs, "k_expr": k_expr, "seed": seed or 0,
              "log1p": "on" in (log1p or [])}
    jobs._job_start("banksy", f"Banksy λ={float(lam):g}", banksy._run_banksy,
                    args=(seg_source or "xenium", counts_mode == "corrected", params),
                    state=banksy._bk_state, lock=banksy._bk_lock)
    return True, False, "Running…"


@app_layout.app.callback(
    Output("bk-status",          "children", allow_duplicate=True),
    Output("bk-run-btn",         "disabled", allow_duplicate=True),
    Output("bk-done",            "data"),
    Output("bk-poll",            "disabled", allow_duplicate=True),
    Output("clusterset-version", "data",     allow_duplicate=True),
    Output("geneset-version",    "data",     allow_duplicate=True),
    Input("bk-poll",             "n_intervals"),
    State("bk-done",             "data"),
    State("clusterset-version",  "data"),
    State("geneset-version",     "data"),
    prevent_initial_call=True,
)
def poll_banksy(_, seen=None, cs_version=None, gs_version=None):
    """The status while the job runs; when it ends, the result version, a bump of the
    cluster-set version (the Clustering Method options) and of the version both renderers
    redraw per-cell values for (the set may be the one on the map)."""
    with banksy._bk_lock:
        st = banksy._bk_state
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
    Input("bk-poll",    "disabled"),
    State("jobs-poll",  "disabled"),
    prevent_initial_call=True,
)
def wake_jobs_poll_banksy(_bk, poll_disabled):
    if jobs._jobs_panel_rows() and poll_disabled:
        return False
    return no_update


@app_layout.app.callback(
    Output("bk-plot",  "figure"),
    Output("bk-table", "data"),
    Input("bk-done",         "data"),
    Input("dataset-version", "data"),
    Input("bk-plot-w",       "value"),
    Input("bk-plot-h",       "value"),
)
def update_banksy_results(_done, _ds, w, h):
    return banksy.make_bk_fig(width=w, height=h), banksy._bk_records(banksy._bk_result())


@app_layout.app.callback(
    Output("bk-download", "data"),
    Input("bk-download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_banksy(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    res = banksy._bk_result()
    if not res:
        raise dash.exceptions.PreventUpdate
    return dict(content=banksy._bk_csv(res), filename="banksy_clusters.csv")
