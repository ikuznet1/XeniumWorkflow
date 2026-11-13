"""Callbacks: transcript factors (segmentation-free factorisation of binned transcripts)."""

import dash
from dash import no_update
from dash import Input, Output, State

from . import jobs, stats, txfactor, app_layout, cb_analysis


@app_layout.app.callback(
    Output("txf-labels", "options"),
    Output("txf-labels", "value"),
    Input("plots-tabs",      "active_tab"),
    Input("annot-done",      "data"),
    Input("dataset-version", "data"),
    Input("seg-source",      "value"),
    Input("extra-datasets-version", "data"),
    State("color-by",        "value"),
    State("txf-labels",      "value"),
)
def update_txf_label_options(active_tab, _annot, _ds, seg_source, _ex, color_by=None, current=None):
    if active_tab != "tab-txf":
        raise dash.exceptions.PreventUpdate
    opts = [o for o in stats._comp_category_options(seg_source, color_by) if o["value"] != "niche"]
    have = [o["value"] for o in opts]
    want = (f"celltype:{str(color_by).split(':', 1)[1]}" if str(color_by or "").startswith("cell_type:") else None)
    first = want if want in have else next((v for v in have if v.startswith("celltype:")), None)
    return opts, no_update if current in have else first


@app_layout.app.callback(
    Output("txf-run-btn", "disabled"),
    Output("txf-poll",    "disabled"),
    Output("txf-status",  "children"),
    Input("txf-run-btn",  "n_clicks"),
    State("txf-method",   "value"),
    State("txf-k",        "value"),
    State("txf-bin",      "value"),
    State("txf-shape",    "value"),
    State("txf-mintx",    "value"),
    State("txf-maxfit",   "value"),
    State("txf-seed",     "value"),
    State("txf-labels",   "value"),
    State("seg-source",   "value"),
    State("cluster-method", "value"),
    State("counts-mode-store", "data"),
    prevent_initial_call=True,
)
def run_txfactor(n_clicks, method, k, bin_um, shape, min_tx, max_fit, seed, label_kind, seg_source,
                 cluster_method, counts_mode=None):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    method = method or "nmf"
    if method == "reference" and not label_kind:
        return False, no_update, ("The reference mode projects the bins onto the mean profiles of cell types: "
                                  "pick an annotation (or clusters) under 'Cell types from'.")
    if method != "reference" and (k is None or k < 2):
        return False, no_update, "K (the number of factors) must be at least 2."
    if bin_um is None or bin_um <= 0:
        return False, no_update, "The bin size must be positive."
    if not any(t["ds"].get("data_dir") for t in stats._view_tables(seg_source or "xenium")):
        return False, no_update, ("Transcript factors need the Xenium output directory (transcripts.parquet or "
                                  "transcripts.zarr.zip); this dataset was loaded from a .zarr.")
    if not cb_analysis._an_job_claim(txfactor._txf_state, txfactor._txf_lock):
        return True, False, "Transcript factors are already being computed."
    params = {"k": k, "bin": bin_um, "shape": shape, "min_tx": min_tx, "max_fit": max_fit, "seed": seed}
    jobs._job_start("txfactor", "Transcript factors", txfactor._run_txfactor,
                    args=(seg_source or "xenium", cluster_method, label_kind, method, params,
                          counts_mode == "corrected"),
                    state=txfactor._txf_state, lock=txfactor._txf_lock)
    return True, False, "Running…"


@app_layout.app.callback(
    Output("txf-status",      "children", allow_duplicate=True),
    Output("txf-run-btn",     "disabled", allow_duplicate=True),
    Output("txf-done",        "data"),
    Output("txf-poll",        "disabled", allow_duplicate=True),
    Output("geneset-version", "data",     allow_duplicate=True),
    Input("txf-poll",         "n_intervals"),
    State("txf-done",         "data"),
    State("geneset-version",  "data"),
    prevent_initial_call=True,
)
def poll_txf(_, seen=None, gs_version=None):
    """The status while the job runs; when it ends, the result version - and a bump of the
    version both renderers redraw per-cell values for (Color By options and colours)."""
    with txfactor._txf_lock:
        st = txfactor._txf_state
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
    Input("txf-poll",   "disabled"),
    State("jobs-poll",  "disabled"),
    prevent_initial_call=True,
)
def wake_jobs_poll_txfactor(_txf, poll_disabled):
    if jobs._jobs_panel_rows() and poll_disabled:
        return False
    return no_update


@app_layout.app.callback(
    Output("txf-plot",  "figure"),
    Output("txf-table", "data"),
    Output("txf-layer", "options"),
    Output("txf-layer", "value"),
    Input("txf-done",        "data"),
    Input("dataset-version", "data"),
    Input("txf-plot-w",      "value"),
    Input("txf-plot-h",      "value"),
    State("txf-layer",       "value"),
)
def update_txf_results(_done, _ds, w, h, layer=None):
    res = txfactor._txf_result()
    opts = txfactor._txf_layer_options()
    keep = layer if layer in {o["value"] for o in opts} else None
    return (txfactor.make_txf_fig(width=w, height=h), txfactor._txf_records(res), opts,
            no_update if keep == layer else keep)


@app_layout.app.callback(
    Output("density-gene",   "value", allow_duplicate=True),
    Output("density-enable", "value", allow_duplicate=True),
    Input("txf-layer",       "value"),
    State("density-gene",    "value"),
    prevent_initial_call=True,
)
def set_txf_layer(layer, current=None):
    """The factor layer is drawn by the transcript-density heat layer: picking one selects
    it there and switches the layer on; clearing it switches the layer off when it was
    showing a factor."""
    if layer:
        return layer, ["on"]
    if str(current or "").startswith(txfactor._TXF_GENE):
        return None, no_update
    raise dash.exceptions.PreventUpdate


@app_layout.app.callback(
    Output("txf-download", "data"),
    Input("txf-download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_txfactor(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    res = txfactor._txf_result()
    if not res:
        raise dash.exceptions.PreventUpdate
    return dict(content=txfactor._txf_csv(res), filename="transcript_factors.csv")
