"""Callbacks: CellNEST, cell type annotation, abundance plot, plot resizing."""

import hashlib
from dash import no_update
import numpy as np
import dash
from dash import Input, Output, State, callback_context
import plotly.graph_objects as go

from . import core, jobs, dataview, annotation, figures, cellnest, app_layout


@app_layout.app.callback(
    Output("spage-modal", "is_open"),
    Input("spage-modal-open-btn", "n_clicks"),
    Input("spage-modal-close-btn", "n_clicks"),
    State("spage-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_spage_modal(open_clicks, close_clicks, is_open):
    triggered = callback_context.triggered_id
    if triggered == "spage-modal-open-btn":
        return True
    if triggered == "spage-modal-close-btn":
        return False
    return is_open


@app_layout.app.callback(
    Output("cellnest-modal", "is_open"),
    Input("cellnest-modal-open-btn", "n_clicks"),
    Input("cellnest-modal-close-btn", "n_clicks"),
    State("cellnest-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_cellnest_modal(open_clicks, close_clicks, is_open):
    triggered = callback_context.triggered_id
    if triggered == "cellnest-modal-open-btn" and open_clicks:
        return True
    if triggered == "cellnest-modal-close-btn":
        return False
    return is_open


@app_layout.app.callback(
    Output("cellnest-dataset-dropdown", "options"),
    Output("cellnest-modal-status", "children", allow_duplicate=True),
    Input("cellnest-detect-btn", "n_clicks"),
    State("cellnest-folder-path", "value"),
    prevent_initial_call=True,
)
def detect_cellnest_datasets(n_clicks, folder_path):
    import os
    if not n_clicks or not folder_path:
        return [], ""
    out_dir = os.path.join(folder_path.strip(), "output")
    if not os.path.isdir(out_dir):
        return [], f"Not found: {out_dir}"
    datasets = [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]
    if not datasets:
        return [], "No dataset subdirectories found in output/"
    options = [{"label": d, "value": d} for d in sorted(datasets)]
    return options, f"Found {len(datasets)} dataset(s)"


@app_layout.app.callback(
    Output("cellnest-modal-status", "children", allow_duplicate=True),
    Output("cellnest-poll", "disabled"),
    Input("cellnest-load-btn", "n_clicks"),
    State("cellnest-folder-path", "value"),
    State("cellnest-dataset-dropdown", "value"),
    State("cellnest-strip-suffix", "value"),
    prevent_initial_call=True,
)
def load_cellnest(n_clicks, folder, dataset, strip_suffix):
    import threading as _thr
    if not n_clicks or not folder or not dataset:
        raise dash.exceptions.PreventUpdate
    with core._cellnest_lock:
        core._cellnest_state["status"] = "loading"
        core._cellnest_state["message"] = "Loading…"
        core._cellnest_state["data"]   = None
    jobs._job_start("cellnest", f"CellNEST load: {dataset}", cellnest._load_cellnest_data,
               args=(folder.strip(), dataset, strip_suffix or ""),
               state=core._cellnest_state, lock=core._cellnest_lock)
    return "Loading…", False   # enable poll interval


@app_layout.app.callback(
    Output("cellnest-version", "data"),
    Output("cellnest-sidebar-status", "children"),
    Output("cellnest-modal-status", "children", allow_duplicate=True),
    Output("cellnest-lr-pair", "options"),
    Output("cellnest-lr-pair", "value"),
    Output("cellnest-poll", "disabled", allow_duplicate=True),
    Input("cellnest-poll", "n_intervals"),
    State("cellnest-version", "data"),
    prevent_initial_call=True,
)
def poll_cellnest(n_intervals, cur_version):
    with core._cellnest_lock:
        status  = core._cellnest_state.get("status", "idle")
        message = core._cellnest_state.get("message", "")
        data    = core._cellnest_state.get("data")

    if status == "loading":
        raise dash.exceptions.PreventUpdate

    # done or error — disable poll and update UI
    if status == "done" and data is not None:
        lr_freq  = data["lr_freq"]
        col_pair = lr_freq.columns[0]
        col_cnt  = lr_freq.columns[1]
        lr_sorted = lr_freq.sort_values(col_cnt, ascending=False)
        lr_options = [{"label": p, "value": p} for p in lr_sorted[col_pair].tolist()]
        first_val  = [lr_options[0]["value"]] if lr_options else []
        return (cur_version or 0) + 1, message, message, lr_options, first_val, True
    else:
        return (cur_version or 0), message, message, [], None, True


@app_layout.app.callback(
    Output("cellnest-filter-ligand",   "options"),
    Output("cellnest-filter-receptor", "options"),
    Output("cellnest-filter-sender",   "options"),
    Output("cellnest-filter-receiver", "options"),
    Output("cellnest-annot-col",       "options"),
    Input("cellnest-version", "data"),
    Input("cellnest-annot-col", "value"),
    prevent_initial_call=True,
)
def populate_cellnest_filters(ccc_version, annot_col):
    with core._cellnest_lock:
        data = core._cellnest_state.get("data")
    if data is None:
        return [], [], [], [], []
    ccc_df = data["ccc_df"]
    ligands   = [{"label": v, "value": v} for v in sorted(ccc_df["ligand"].unique())]
    receptors = [{"label": v, "value": v} for v in sorted(ccc_df["receptor"].unique())]
    annot_opts = [{"label": c, "value": c} for c in data.get("annot_cols", [])]
    sender_opts = receiver_opts = []
    if annot_col:
        from_col = f"from_{annot_col}"
        to_col   = f"to_{annot_col}"
        if from_col in ccc_df.columns:
            sender_opts   = [{"label": str(v), "value": str(v)}
                             for v in sorted(ccc_df[from_col].dropna().unique(), key=str)]
        if to_col in ccc_df.columns:
            receiver_opts = [{"label": str(v), "value": str(v)}
                             for v in sorted(ccc_df[to_col].dropna().unique(), key=str)]
    return ligands, receptors, sender_opts, receiver_opts, annot_opts


@app_layout.app.callback(
    Output("cellnest-bar-plot", "figure"),
    Output("cellnest-spatial-plot", "figure"),
    Input("cellnest-version", "data"),
    Input("cellnest-lr-pair", "value"),
    Input("cellnest-top-n", "value"),
    Input("cellnest-color-by", "value"),
    Input("cellnest-annot-col", "value"),
    Input("cellnest-filter-ligand",   "value"),
    Input("cellnest-filter-receptor", "value"),
    Input("cellnest-filter-sender",   "value"),
    Input("cellnest-filter-receiver", "value"),
    Input("cellnest-options", "value"),
    Input("cellnest-edge-width", "value"),
    prevent_initial_call=True,
)
def update_cellnest_tab(ccc_version, lr_pair, top_n, color_by, annot_col,
                        f_ligand, f_receptor, f_sender, f_receiver, options, edge_width):
    with core._cellnest_lock:
        status = core._cellnest_state.get("status")
    if status != "done":
        return go.Figure(), go.Figure()
    opts = options or []
    ew = edge_width or 1
    return (
        cellnest._cellnest_make_bar(f_ligand=f_ligand, f_receptor=f_receptor,
                           f_sender=f_sender, f_receiver=f_receiver,
                           annot_col=annot_col),
        cellnest._cellnest_make_circos(lr_pair, annot_col, top_n or 500,
                              color_by=color_by or "lr_pair",
                              f_ligand=f_ligand, f_receptor=f_receptor,
                              f_sender=f_sender, f_receiver=f_receiver),
    )


@app_layout.app.callback(
    Output("cellnest-lr-heatmap",       "figure"),
    Output("cellnest-ligtarget-heatmap","figure"),
    Output("cellnest-dotplot",          "figure"),
    Input("cellnest-version",           "data"),
    Input("cellnest-nichenest-top-n",   "value"),
    Input("cellnest-annot-col",         "value"),
    Input("cellnest-filter-ligand",     "value"),
    Input("cellnest-filter-receptor",   "value"),
    Input("cellnest-filter-sender",     "value"),
    Input("cellnest-filter-receiver",   "value"),
    prevent_initial_call=True,
)
def update_cellnest_nichenest(ccc_version, top_n, annot_col,
                               f_ligand, f_receptor, f_sender, f_receiver):
    with core._cellnest_lock:
        status = core._cellnest_state.get("status")
    if status != "done":
        empty = go.Figure(layout=go.Layout(paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG))
        return empty, empty, empty
    n = top_n or 20
    kw = dict(f_ligand=f_ligand, f_receptor=f_receptor,
              f_sender=f_sender, f_receiver=f_receiver,
              annot_col=annot_col, top_n=n)
    return (
        cellnest._cellnest_make_lr_heatmap(**kw),
        cellnest._cellnest_make_ligtarget_heatmap(**kw),
        cellnest._cellnest_make_dotplot(**kw),
    )


@app_layout.app.callback(
    Output("spatial-plot", "figure", allow_duplicate=True),
    Input("cellnest-lr-pair", "value"),
    Input("cellnest-top-n", "value"),
    Input("cellnest-version", "data"),
    Input("cellnest-edge-width", "value"),
    Input("cellnest-overlay-enable", "value"),
    Input("cellnest-options", "value"),
    Input("cellnest-color-by", "value"),
    Input("cellnest-filter-ligand",   "value"),
    Input("cellnest-filter-receptor", "value"),
    Input("cellnest-filter-sender",   "value"),
    Input("cellnest-filter-receiver", "value"),
    Input("cellnest-annot-col",       "value"),
    State("spatial-plot", "figure"),
    prevent_initial_call=True,
)
def update_cellnest_main_overlay(lr_pair, top_n, ccc_version, edge_width,
                                 overlay_enable, options, color_by,
                                 f_ligand, f_receptor, f_sender, f_receiver, annot_col,
                                 current_fig):
    if current_fig is None:
        raise dash.exceptions.PreventUpdate
    # Remove existing CellNEST traces
    data_traces = [t for t in current_fig.get("data", [])
                   if not str(t.get("name", "")).startswith("cellnest_")]
    with core._cellnest_lock:
        status = core._cellnest_state.get("status")
    overlay_on = "on" in (overlay_enable or [])
    opts = options or []
    if status == "done" and lr_pair and overlay_on:
        new_traces = cellnest._cellnest_overlay_traces(lr_pair, top_n or 100,
                                              edge_width=edge_width or 1,
                                              color_by=color_by or "lr_pair",
                                              options=opts,
                                              f_ligand=f_ligand, f_receptor=f_receptor,
                                              f_sender=f_sender, f_receiver=f_receiver,
                                              annot_col=annot_col)
        data_traces = data_traces + [t.to_plotly_json() for t in new_traces]
    current_fig["data"] = data_traces
    return current_fig


@app_layout.app.callback(
    Output("cellnest-downstream-tf-select", "options"),
    Output("cellnest-downstream-status", "children"),
    Output("cellnest-downstream-results", "style"),
    Input("cellnest-downstream-btn", "n_clicks"),
    State("cellnest-downstream-receptor", "value"),
    State("cellnest-downstream-hops", "value"),
    prevent_initial_call=True,
)
def find_downstream_tfs(n_clicks, receptor, hops):
    if not n_clicks or not receptor:
        raise dash.exceptions.PreventUpdate
    receptor = receptor.strip()
    try:
        panel = set(core.DATA["gene_names"])
        tfs = cellnest._downstream_tfs(receptor, n_hops=int(hops or 2))
        # Filter to TFs that are in panel OR have targets in panel
        tf_targets = cellnest._downstream_tf_targets(tfs, panel)
        # TFs themselves in panel take priority
        tfs_in_panel = [tf for tf in tfs if tf in panel]
        all_tfs = sorted(set(tfs_in_panel) | set(tf_targets.keys()))
        if not all_tfs:
            return [], f"No downstream TFs found in panel for {receptor}", {"display": "none"}
        options = [{"label": f"{tf} {'✓' if tf in panel else '(targets only)'}", "value": tf}
                   for tf in all_tfs]
        msg = f"{len(all_tfs)} TF(s) found downstream of {receptor}"
        return options, msg, {"display": "block"}
    except Exception as exc:
        return [], f"Error: {exc}", {"display": "none"}


@app_layout.app.callback(
    Output("cellnest-downstream-plot", "figure"),
    Input("cellnest-downstream-tf-select", "value"),
    State("cellnest-downstream-receptor", "value"),
    prevent_initial_call=True,
)
def plot_downstream_tfs(selected_tfs, receptor):
    if not selected_tfs:
        raise dash.exceptions.PreventUpdate
    panel = set(core.DATA["gene_names"])
    df_app = core.DATA["df"]

    # Get expression for each selected TF
    fig = go.Figure()
    bg_coords = df_app[["x_centroid", "y_centroid"]]
    for tf in selected_tfs:
        _tf_label = tf
        if tf not in panel:
            # "(targets only)" TF: show the mean log1p expression of its in-panel targets
            try:
                _tg = cellnest._downstream_tf_targets([tf], panel).get(tf, [])
            except Exception:
                _tg = []
            _tg = [g for g in _tg if g in core.DATA["gene_name_to_idx"]]
            if not _tg:
                continue
            _tcols = [core.DATA["gene_name_to_idx"][g] for g in _tg]
            _tmean = np.asarray(core.DATA["expr_csc"][:, _tcols].log1p().mean(axis=1)).ravel()
            _tidx  = core.DATA["df_to_expr"]
            vals = np.where(_tidx >= 0, _tmean[np.clip(_tidx, 0, len(_tmean) - 1)], 0.0)
            _tf_label = f"{tf} targets (mean log1p, n={len(_tg)})"
        else:
            # log1p, aligned to DATA["df"] through df_to_expr (subset-safe)
            vals = dataview.get_gene_expression(tf)
        fig.add_trace(go.Scattergl(
            x=df_app["x_centroid"].tolist(),
            y=(-df_app["y_centroid"]).tolist(),
            mode="markers",
            marker={"size": 2, "color": vals.tolist(), "colorscale": "Viridis",
                    "showscale": True, "colorbar": {"title": _tf_label, "thickness": 10, "len": 0.6}},
            name=_tf_label,
            hovertemplate=f"{_tf_label}: %{{marker.color:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG,
        font={"color": core.TEXT, "size": 10},
        margin={"l": 10, "r": 10, "t": 25, "b": 10},
        title={"text": f"Downstream TF expression (← {receptor or ''})", "font": {"size": 11}},
        xaxis={"gridcolor": core.BORDER, "showticklabels": False},
        yaxis={"gridcolor": core.BORDER, "showticklabels": False, "scaleanchor": "x"},
    )
    return fig


@app_layout.app.callback(
    Output("annot-modal", "is_open"),
    Input("annot-modal-open-btn", "n_clicks"),
    Input("annot-modal-close-btn", "n_clicks"),
    State("annot-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_annot_modal(open_clicks, close_clicks, is_open):
    triggered = callback_context.triggered_id
    if triggered == "annot-modal-open-btn" and open_clicks:
        return True
    if triggered == "annot-modal-close-btn":
        return False
    return is_open


app_layout.app.clientside_callback(
    """function(source) {
        var show = {};
        var hide = {"display": "none"};
        if (source === "celltypist") return [show, hide, hide];
        if (source === "rctd") return [hide, show, show];
        return [hide, show, hide];
    }""",
    Output("annot-celltypist-div", "style"),
    Output("annot-seurat-div",     "style"),
    Output("annot-rctd-div",       "style"),
    Input("annot-source",          "value"),
    prevent_initial_call=True,
)


@app_layout.app.callback(
    Output("annot-poll",         "disabled"),
    Output("annot-status",       "children"),
    Output("annot-btn",          "disabled"),
    Output("annot-modal-status", "children"),
    Input("annot-btn",           "n_clicks"),
    State("annot-source",        "value"),
    State("annot-model",         "value"),
    State("annot-rds-path",      "value"),
    State("annot-label-col",     "value"),
    State("annot-rctd-mode",     "value"),
    State("annot-rctd-cores",    "value"),
    State("annot-rctd-umi-min",       "value"),
    State("annot-rctd-umi-min-sigma", "value"),
    State("seg-source",               "value"),
    prevent_initial_call=True,
)
def start_annotation(n_clicks, source, model_name, rds_path, label_col,
                     rctd_mode, rctd_cores, rctd_umi_min, rctd_umi_min_sigma, seg_source):
    """Kick off background annotation thread."""
    seg_source = seg_source or "xenium"
    with core._proseg_lock:
        pres = core._proseg_state["result"] if core._seg_tool(seg_source) == "proseg" and core._proseg_state["status"] == "done" else None
    with core._baysor_lock:
        bres = core._baysor_state["result"] if core._seg_tool(seg_source) == "baysor" and core._baysor_state["status"] == "done" else None
    alt_res = pres or bres or (core._seg_result(seg_source, done_only=True)
                               if core._seg_tool(seg_source) == "imgseg" else None)
    if alt_res is None and core._seg_tool(seg_source) != "xenium":
        # a run that is named but not loaded is never replaced by the Xenium cells
        msg = (f"✗ The selected segmentation ({seg_source}) is not loaded (it is still loading, or its "
               f"load failed): nothing was annotated.")
        return True, msg, False, msg   # no job started: keep the poll off

    if alt_res is not None:
        if alt_res.get("expr") is None:
            msg = "✗ No expression matrix for reseg cells — re-run Proseg/Baysor"
            return True, msg, False, msg   # no job started: keep the poll off
        # Measured panel columns only: SpaGE appends imputed columns at the end
        # (a run reloaded from its zarr has them without an 'imputed_genes' list)
        _n_meas = min(alt_res["expr"].shape[1], len(dataview._panel_gene_names()))
        expr_override     = alt_res["expr"][:, :_n_meas] if _n_meas < alt_res["expr"].shape[1] \
                            else alt_res["expr"]
        cell_ids_override = [str(c) for c in alt_res["cells_df"].index]
        # Identifies this segmentation's cells in the annotation disk cache
        cache_sig = (hashlib.md5("\n".join(cell_ids_override).encode()).hexdigest()[:12]
                     + f"_{expr_override.shape[0]}_{expr_override.nnz}")
    else:
        expr_override     = None
        cell_ids_override = None
        cache_sig         = None

    method     = source if source in core._ANNOT_METHODS else "celltypist"
    labels_key = annotation._labels_key_for_method(method, alt_res)

    if method == "rctd":
        mode_labels_key = f"{labels_key}_{rctd_mode or 'full'}"
    else:
        mode_labels_key = labels_key

    if source in ("seurat", "rctd"):
        rds_path = core._ref_rds_path(rds_path)
        _problem = core._ref_rds_problem(rds_path)
        if _problem:
            msg = f"✗ {_problem}"
            return True, msg, False, msg      # no job started: keep the poll off

    # Ensure rpy2 conversion rules are set in this callback thread BEFORE copy_context()
    # so the spawned thread's context copy includes them (done before the status is
    # set to "running" so a missing rpy2 cannot leave it stuck)
    if source in ("seurat", "rctd"):
        try:
            import rpy2.robjects.conversion as _rconv
            import rpy2.robjects as _ro_mod
            _rconv.set_conversion(_ro_mod.default_converter)
        except Exception as e:
            msg = f"✗ rpy2 not available: {e}"
            return True, msg, False, msg   # no job started: keep the poll off

    with core._annot_lock:
        if core._annot_state["status"] == "running":
            return False, "Already running…", True, "Already running…"
        core._annot_state.update({"status": "running", "message": "Starting…",
                              labels_key: None, mode_labels_key: None})

    _job_kw = dict(state=core._annot_state, lock=core._annot_lock)
    if source == "seurat":
        jobs._job_start("annotation", "Annotation: Seurat kNN", annotation._run_seurat_annotation,
                   args=(rds_path or "", label_col or "Names"),
                   kwargs=dict(labels_key=labels_key, expr_override=expr_override,
                               cell_ids_override=cell_ids_override, cache_sig=cache_sig),
                   copy_context=True, **_job_kw)
    elif source == "rctd":
        jobs._job_start("annotation", f"Annotation: RCTD ({rctd_mode or 'full'})", annotation._run_rctd_annotation,
                   args=(rds_path or "", label_col or "Names",
                         rctd_mode or "full", int(rctd_cores or 4)),
                   kwargs=dict(labels_key=labels_key, expr_override=expr_override,
                               cell_ids_override=cell_ids_override,
                               umi_min=int(rctd_umi_min or 20),
                               umi_min_sigma=int(rctd_umi_min_sigma or 100),
                               mode_labels_key=mode_labels_key, cache_sig=cache_sig),
                   copy_context=True, **_job_kw)
    else:
        jobs._job_start("annotation", f"Annotation: CellTypist ({model_name})", annotation._run_celltypist,
                   args=(model_name,),
                   kwargs={"labels_key": labels_key, "expr_override": expr_override,
                           "cell_ids_override": cell_ids_override, "cache_sig": cache_sig},
                   **_job_kw)
    return False, "Starting…", True, "Starting…"


@app_layout.app.callback(
    Output("annot-status",       "children",   allow_duplicate=True),
    Output("annot-poll",         "disabled",   allow_duplicate=True),
    Output("annot-btn",          "disabled",   allow_duplicate=True),
    Output("color-by",           "options"),
    Output("annot-done",         "data"),
    Output("annot-modal-status", "children",   allow_duplicate=True),
    Output("annot-modal",        "is_open",    allow_duplicate=True),
    Input("annot-poll",          "n_intervals"),
    State("color-by",            "options"),
    State("annot-done",          "data"),
    State("annot-modal",         "is_open"),
    prevent_initial_call=True,
)
def poll_annotation(_, current_options, done_version, modal_open):
    """Poll annotation state and update UI when done."""
    with core._annot_lock:
        status  = core._annot_state["status"]
        message = core._annot_state["message"]
        done    = (status == "done")
        error   = (status == "error")
        if done:
            # Consume the signal so repeated poll fires don't re-bump done_version
            core._annot_state["status"] = "loaded"

    # Enable each cell_type:method option if that method has any labels
    with core._annot_lock:
        methods_with_labels = {
            m for m in core._ANNOT_METHODS
            if any(k.startswith(f"labels_{m}") and v is not None
                   for k, v in core._annot_state.items())
        }
    options = []
    for opt in current_options:
        val = opt["value"]
        if val.startswith("cell_type:"):
            m = val.split(":")[1]
            options.append({**opt, "disabled": m not in methods_with_labels})
        else:
            options.append(opt)

    if done:
        # Close the modal and show status in the sidebar
        return f"✓ {message}", True, False, options, (done_version or 0) + 1, f"✓ {message}", False
    if error:
        # Keep the modal open so user can see the error
        return f"✗ {message}", True, False, options, done_version, f"✗ {message}", modal_open
    if status == "loaded":
        # Already reported done — disable interval but don't re-bump done_version
        return f"✓ {message}", True, False, options, done_version, f"✓ {message}", modal_open
    if status == "idle":
        # No job to wait for (nothing started, or the state was reset): stop polling
        return no_update, True, False, options, done_version, no_update, modal_open
    return message, False, True, options, done_version, message, modal_open   # keep polling


app_layout.app.clientside_callback(
    """function(n_clicks) {
        var sp = document.getElementById('spatial-plot');
        var spatialPanel = document.getElementById('spatial-panel');
        var xr0, xr1, yr0, yr1, hasRanges = false;
        if (sp && sp._fullLayout) {
            var xl = sp._fullLayout.xaxis, yl = sp._fullLayout.yaxis;
            if (xl && xl.range && yl && yl.range) {
                xr0 = xl.range[0]; xr1 = xl.range[1];
                yr0 = yl.range[0]; yr1 = yl.range[1];
                hasRanges = true;
            }
        }
        var hiding = (n_clicks % 2 === 1);
        if (hiding && spatialPanel) {
            // Reset spatial panel to flex layout so it fills the space
            spatialPanel.style.flex = '1';
            spatialPanel.style.width = '';
        }
        setTimeout(function() {
            window.dispatchEvent(new Event('resize'));
            if (hasRanges && sp) {
                setTimeout(function() {
                    Plotly.relayout(sp, {
                        'xaxis.range[0]': xr0, 'xaxis.range[1]': xr1,
                        'yaxis.range[0]': yr0, 'yaxis.range[1]': yr1
                    });
                }, 80);
            }
        }, 50);
        if (hiding) return [{"display": "none"}, "Show Plots"];
        return [{"flex": "1", "minWidth": "0", "display": "flex", "flexDirection": "column", "overflow": "hidden"}, "Hide Plots"];
    }""",
    Output("umap-panel",  "style"),
    Output("umap-toggle", "children"),
    Input("umap-toggle",  "n_clicks"),
)


@app_layout.app.callback(
    Output("cell-type-abundance-plot", "figure"),
    Input("annot-done", "data"),
    Input("seg-source", "value"),
    Input("dataset-version", "data"),
)
def update_abundance_plot(_annot_done, _seg_source, _ds_version):
    return figures.make_cell_type_abundance_fig()


_PLOT_RESIZE_JS = """function(w, h) {
    var s = {"flex": "1", "minHeight": "0"};
    if (w) s["width"] = w + "px";
    if (h) { s["height"] = h + "px"; s["flex"] = "none"; }
    return s;
}"""

app_layout.app.clientside_callback(
    _PLOT_RESIZE_JS,
    Output("umap-plot", "style"),
    Input("umap-plot-w", "value"),
    Input("umap-plot-h", "value"),
    prevent_initial_call=True,
)

for _container_id, _prefix in [
    ("abundance-plot-container",    "abundance"),
    ("cooccurrence-plot-container", "cooccurrence"),
    ("enrichment-plot-container",   "enrichment"),
    ("interaction-plot-container",  "interaction"),
    ("chord-plot-container",        "chord"),
    ("niche-umap-plot-container",   "niche-umap"),
    ("roi-area-plot-container",     "roi-area"),
]:
    app_layout.app.clientside_callback(
        _PLOT_RESIZE_JS,
        Output(_container_id, "style"),
        Input(f"{_prefix}-plot-w", "value"),
        Input(f"{_prefix}-plot-h", "value"),
        prevent_initial_call=True,
    )

# CellNEST plots container — custom resize that preserves display:flex
app_layout.app.clientside_callback(
    """function(w, h) {
        var s = {"display": "flex", "flex": "1", "minHeight": "0", "gap": "6px"};
        if (w) { s["width"] = w + "px"; s["flex"] = "none"; }
        if (h) { s["height"] = h + "px"; s["flex"] = "none"; }
        return s;
    }""",
    Output("cellnest-plots-container", "style"),
    Input("cellnest-plot-w", "value"),
    Input("cellnest-plot-h", "value"),
    prevent_initial_call=True,
)
