"""Callbacks: Baysor / Proseg, the resegmentation modal, cached runs."""

import os
import dash
from dash import no_update
from dash import Input, Output, State, callback_context

from . import core, jobs, caching, reseg, clustering, imgseg, app_layout


# ─── Baysor callbacks ─────────────────────────────────────────────────────────

app_layout.app.clientside_callback(
    """function(usePrior) {
        return (usePrior && usePrior.indexOf("yes") !== -1) ? {} : {"display": "none"};
    }""",
    Output("baysor-prior-conf-div", "style"),
    Input("baysor-use-prior", "value"),
)


@app_layout.app.callback(
    Output("baysor-xmin", "value"),
    Output("baysor-xmax", "value"),
    Output("baysor-ymin", "value"),
    Output("baysor-ymax", "value"),
    Input("baysor-use-viewport", "n_clicks"),
    State("spatial-relayout", "data"),
    prevent_initial_call=True,
)
def baysor_fill_viewport(_, relayout):
    r = relayout or {}
    x0 = r.get("xaxis.range[0]")
    x1 = r.get("xaxis.range[1]")
    y0 = r.get("yaxis.range[0]")  # negated in plot; y_plot = -y_µm
    y1 = r.get("yaxis.range[1]")
    if None in (x0, x1, y0, y1):
        return no_update, no_update, no_update, no_update
    # Convert plot Y back to µm (plot y = -y_µm)
    ym_min = round(-float(y1), 1)
    ym_max = round(-float(y0), 1)
    return round(float(x0), 1), round(float(x1), 1), ym_min, ym_max


@app_layout.app.callback(
    Output("baysor-status",  "children"),
    Output("baysor-poll",    "disabled",  allow_duplicate=True),
    Output("baysor-done",    "data"),
    Output("seg-source",     "value",     allow_duplicate=True),
    Input("baysor-poll",     "n_intervals"),
    State("baysor-done",     "data"),
    State("seg-source",      "value"),
    prevent_initial_call=True,
)
def poll_baysor(_, done_version, seg_value=None):
    with core._baysor_lock:
        status  = core._baysor_state["status"]
        message = core._baysor_state["message"]
        has_result = core._baysor_state.get("result") is not None

    if status == "done":
        out_dir = (core._baysor_state.get("result") or {}).get("out_dir", "")
        param_tag = os.path.basename(out_dir).split("_")[-1] if out_dir else ""
        seg_val = f"baysor:{param_tag}" if param_tag else "baysor"
        return f"✓ {message}", True, (done_version or 0) + 1, seg_val
    if status == "error":
        # Nothing is loaded for this tool (failed run or failed cached load): do not
        # leave the dropdown on a Baysor run that is not there.
        if not has_result and core._seg_tool(seg_value) == "baysor":
            return f"✗ {message}", True, (done_version or 0) + 1, "xenium"
        return f"✗ {message}", True, no_update, no_update
    return message, False, no_update, no_update


# ─── Proseg callbacks ─────────────────────────────────────────────────────────

@app_layout.app.callback(
    Output("proseg-xmin", "value"),
    Output("proseg-xmax", "value"),
    Output("proseg-ymin", "value"),
    Output("proseg-ymax", "value"),
    Input("proseg-use-viewport", "n_clicks"),
    State("spatial-relayout", "data"),
    prevent_initial_call=True,
)
def proseg_fill_viewport(_, relayout):
    r = relayout or {}
    x0 = r.get("xaxis.range[0]")
    x1 = r.get("xaxis.range[1]")
    y0 = r.get("yaxis.range[0]")
    y1 = r.get("yaxis.range[1]")
    if None in (x0, x1, y0, y1):
        return no_update, no_update, no_update, no_update
    ym_min = round(-float(y1), 1)
    ym_max = round(-float(y0), 1)
    return round(float(x0), 1), round(float(x1), 1), ym_min, ym_max


@app_layout.app.callback(
    Output("proseg-status",  "children"),
    Output("proseg-poll",    "disabled",  allow_duplicate=True),
    Output("proseg-done",    "data"),
    Output("seg-source",     "value",     allow_duplicate=True),
    Input("proseg-poll",     "n_intervals"),
    State("proseg-done",     "data"),
    State("seg-source",      "value"),
    prevent_initial_call=True,
)
def poll_proseg(_, done_version, seg_value=None):
    with core._proseg_lock:
        status  = core._proseg_state["status"]
        message = core._proseg_state["message"]
        has_result = core._proseg_state.get("result") is not None

    if status == "done":
        out_dir = (core._proseg_state.get("result") or {}).get("out_dir", "")
        param_tag = os.path.basename(out_dir).split("_")[-1] if out_dir else ""
        seg_val = f"proseg:{param_tag}" if param_tag else "proseg"
        return f"✓ {message}", True, (done_version or 0) + 1, seg_val
    if status == "error":
        # see poll_baysor
        if not has_result and core._seg_tool(seg_value) == "proseg":
            return f"✗ {message}", True, (done_version or 0) + 1, "xenium"
        return f"✗ {message}", True, no_update, no_update
    return message, False, no_update, no_update


# ─── Resegmentation modal callbacks ───────────────────────────────────────────

@app_layout.app.callback(
    Output("reseg-modal",              "is_open"),
    Output("reseg-patches-confirmed",  "data",     allow_duplicate=True),
    Output("reseg-modal-run-btn",      "disabled",  allow_duplicate=True),
    Output("reseg-patch-confirm-div",  "style",     allow_duplicate=True),
    Output("reseg-patch-confirm-msg",  "children",  allow_duplicate=True),
    Output("reseg-step2-content",      "style",     allow_duplicate=True),
    Output("reseg-step2-overlay",      "style",     allow_duplicate=True),
    Input("reseg-modal-open-btn",  "n_clicks"),
    Input("reseg-modal-close-btn", "n_clicks"),
    State("reseg-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_reseg_modal(open_clicks, close_clicks, is_open):
    triggered = callback_context.triggered_id
    if triggered == "reseg-modal-open-btn" and open_clicks:
        # Check if patches already exist from a previous run / cache
        with core._sdata_lock:
            has_patches = core._sdata_state.get("patches") is not None
        if has_patches:
            with core._sdata_lock:
                n_patches = len(core._sdata_state["patches"])
            confirm_msg = (f"✓ {n_patches} patches ready — review ROI and patches on the "
                           f"spatial plot, then confirm to proceed.")
            # Patches exist: show confirm div, keep Step 2 locked until user confirms
            return True, 0, True, {}, confirm_msg, {"opacity": "0.4", "pointerEvents": "none"}, {"display": "block"}
        # No patches yet: hide confirm div, lock Step 2
        return True, 0, True, {"display": "none"}, "", {"opacity": "0.4", "pointerEvents": "none"}, {"display": "block"}
    return False, no_update, no_update, no_update, no_update, no_update, no_update


@app_layout.app.callback(
    Output("modal-baysor-panel", "style"),
    Output("modal-proseg-panel", "style"),
    Output("reseg-modal-run-btn", "style"),
    Input("reseg-algo-tabs", "value"),
    prevent_initial_call=True,
)
def switch_reseg_tab(algo):
    show = {}
    hide = {"display": "none"}
    run_btn_baysor_style = {"padding": "7px 20px", "backgroundColor": "#1f6feb", "color": "#fff",
                            "border": "none", "borderRadius": "5px", "cursor": "pointer",
                            "fontSize": "13px", "fontWeight": "600"}
    run_btn_proseg_style = {**run_btn_baysor_style, "backgroundColor": "#2d6a4f"}
    if algo == "proseg":
        return hide, show, run_btn_proseg_style
    if algo == "imgseg":
        return hide, hide, {**run_btn_baysor_style, "backgroundColor": "#9c3d6b"}
    return show, hide, run_btn_baysor_style


def _loaded_run_busy(tool: str) -> str:
    """'' or why a new run of *tool* cannot start now: another job works on the run of
    this tool that is loaded. Starting a run unloads it, and a re-run with the same
    parameters replaces its directory - under a job that writes its result into that
    directory by path (SPLIT, SpaGE, Rerun UMAP). The message names the job."""
    state, lock = core._seg_state(tool)
    if state is None:
        return ""
    with lock:
        res = state.get("result")
    if not res:
        return ""
    msg = jobs._run_in_use(res, "replaced")
    return msg and (f"{core._SEG_TOOL_LABELS.get(tool, tool)} not started. " + msg)


def _run_delete_refusal(tool: str, out_dir: str) -> str:
    """'' or why the run directory *out_dir* cannot be deleted now: a job works on it, or
    it is being loaded from the cache (the load would be published with its files gone)."""
    msg = jobs._run_in_use(out_dir, "deleted")
    if msg:
        return msg
    state, lock = core._seg_state(tool)
    if state is not None:
        with lock:
            loading = (state.get("status") == "running" and state.get("loading_out_dir")
                       and os.path.realpath(state["loading_out_dir"]) == os.path.realpath(out_dir))
        if loading:
            return (f"The run {os.path.basename(out_dir)} is being loaded from the cache: it was not "
                    f"deleted. Wait for the load to finish and try again.")
    return ""


@app_layout.app.callback(
    Output("reseg-modal-run-status", "children"),
    Output("reseg-modal", "is_open", allow_duplicate=True),
    Output("baysor-poll", "disabled", allow_duplicate=True),
    Output("proseg-poll", "disabled", allow_duplicate=True),
    Input("reseg-modal-run-btn", "n_clicks"),
    State("reseg-algo-tabs", "value"),
    State("reseg-patches-confirmed", "data"),
    # Baysor states
    State("baysor-xmin", "value"),
    State("baysor-xmax", "value"),
    State("baysor-ymin", "value"),
    State("baysor-ymax", "value"),
    State("baysor-scale", "value"),
    State("baysor-scale-std", "value"),
    State("baysor-min-mol", "value"),
    State("baysor-n_clusters", "value"),
    State("baysor-use-patches", "value"),
    State("baysor-use-prior", "value"),
    State("baysor-prior-conf", "value"),
    State("baysor-prior-type", "value"),
    # Proseg states
    State("proseg-xmin", "value"),
    State("proseg-xmax", "value"),
    State("proseg-ymin", "value"),
    State("proseg-ymax", "value"),
    State("proseg-voxel-size", "value"),
    State("proseg-nthreads", "value"),
    State("proseg-samples", "value"),
    State("proseg-recorded-samples", "value"),
    State("proseg-schedule", "value"),
    State("proseg-nuclear-reassign-prob", "value"),
    State("proseg-prior-seg-prob", "value"),
    State("proseg-prior-type", "value"),
    prevent_initial_call=True,
)
def run_reseg_modal(n_clicks, algo, patches_confirmed,
                    bxmin, bxmax, bymin, bymax, bscale, bscale_std, bmin_mol, bn_clusters, buse_patches, buse_prior, bprior_conf, bprior_type,
                    pxmin, pxmax, pymin, pymax, pvoxel, pthreads, psamples,
                    precorded_samples, pschedule, pnuclear_prob, pprior_seg_prob, pprior_type):
    if algo == "imgseg":
        raise dash.exceptions.PreventUpdate          # run_imgseg_modal handles this tab
    if not patches_confirmed:
        return "✗ Please confirm patches in Step 1 before running segmentation.", no_update, no_update, no_update
    # "Skip - run on full tissue" overrides patches left over from earlier in the session
    skip_patches = patches_confirmed == "skip"
    _busy = _loaded_run_busy(algo)
    if _busy:
        return f"✗ {_busy}", no_update, no_update, no_update
    if algo == "baysor":
        with core._baysor_lock:
            if core._baysor_state["status"] == "running":
                return "Already running…", True, False, no_update
            core._baysor_state.update({"status": "running", "message": "Starting…", "result": None,
                                  "loading_out_dir": None})
        scale      = float(bscale) if bscale is not None else 20.0
        scale_std  = float(bscale_std) if bscale_std is not None else None
        min_mol     = int(bmin_mol    or 10)
        n_clusters  = int(bn_clusters or 10)
        use_patches = "yes" in (buse_patches or []) and not skip_patches
        prior_conf  = float(bprior_conf or 0.5)
        use_prior_bool = "yes" in (buse_prior or [])
        region = {k: v for k, v in
                  dict(x_min=bxmin, x_max=bxmax, y_min=bymin, y_max=bymax).items()
                  if v is not None}
        jobs._job_start("baysor", "Baysor segmentation" + (" (region)" if region else ""),
                   reseg._run_baysor,
                   args=(scale, min_mol, use_prior_bool, prior_conf),
                   kwargs=dict(**region, scale_std=scale_std, n_clusters=n_clusters,
                               use_patches=use_patches,
                               prior_type=bprior_type or "nucleus"),
                   state=core._baysor_state, lock=core._baysor_lock)
        return "Baysor starting…", False, False, no_update
    else:  # proseg
        # the request's values are converted before anything starts: numbers as numbers,
        # the free-text schedule as a short list of positive integers (nothing else may
        # reach the Proseg command line)
        try:
            pschedule = reseg._proseg_schedule(pschedule)
            pvoxel = float(pvoxel) if pvoxel not in (None, "") else None
            pthreads = int(pthreads) if pthreads not in (None, "") else None
        except (TypeError, ValueError) as exc:
            _why = str(exc) if str(exc).startswith("Sampler schedule") else "voxel size and threads must be numbers."
            return f"✗ Proseg not started: {_why}", no_update, no_update, no_update
        with core._proseg_lock:
            if core._proseg_state["status"] == "running":
                return "Already running…", True, no_update, False
            core._proseg_state.update({"status": "running", "message": "Starting…", "result": None,
                                  "loading_out_dir": None})
        region = {k: v for k, v in
                  dict(x_min=pxmin, x_max=pxmax, y_min=pymin, y_max=pymax).items()
                  if v is not None}
        jobs._job_start("proseg", "Proseg segmentation" + (" (region)" if region else ""),
                   reseg._run_proseg,
                   kwargs=dict(voxel_size=pvoxel, n_threads=pthreads,
                               n_samples=int(psamples) if psamples else 10,
                               recorded_samples=int(precorded_samples) if precorded_samples else None,
                               schedule=pschedule,
                               nuclear_reassign_prob=float(pnuclear_prob) if pnuclear_prob is not None else None,
                               prior_seg_prob=float(pprior_seg_prob) if pprior_seg_prob is not None else None,
                               prior_type=pprior_type or "nucleus",
                               use_patches=not skip_patches,
                               **region),
                   state=core._proseg_state, lock=core._proseg_lock)
        return "Proseg starting…", False, no_update, False


@app_layout.app.callback(
    Output("reseg-status", "children"),
    Input("baysor-done", "data"),
    Input("proseg-done", "data"),
    Input("seg-source",  "value"),
    prevent_initial_call=True,
)
def update_reseg_status(baysor_done, proseg_done, seg_source=None):
    """The finished run the sidebar reports: the one of the active segmentation source when
    that tool has one, else the first loaded one (Baysor, Proseg, image segmentation)."""
    done = {}
    for tool, lock, state in core._seg_states():
        with lock:
            if state["status"] == "done":
                done[tool] = state["message"]
    if not done:
        return ""
    tool = core._seg_tool(seg_source)
    if tool not in done:
        tool = next(iter(done))
    return f"\u2713 {core._SEG_TOOL_LABELS.get(tool, tool)}: {done[tool]}"


# ─── Reseg UMAP callbacks ─────────────────────────────────────────────────────

@app_layout.app.callback(
    Output("umap-reseg-poll",    "disabled"),
    Output("umap-reseg-status",  "children"),
    Output("umap-reseg-btn",     "disabled"),
    Input("umap-reseg-btn",      "n_clicks"),
    State("counts-mode-store",   "data"),
    State("seg-source",          "value"),
    prevent_initial_call=True,
)
def start_reseg_umap(_, counts_mode, seg_source):
    with core._umap_reseg_lock:
        if core._umap_reseg_state["status"] == "running":
            return False, "Already running…", True
        core._umap_reseg_state.update({"status": "running", "message": "Starting…", "result": None,
                                  "_xenium_bumped": False,
                                  "seg_source": seg_source or "xenium",
                                  "counts_mode": counts_mode or "original"})
    jobs._job_start("umap", "Rerun UMAP", clustering._run_reseg_umap,
               state=core._umap_reseg_state, lock=core._umap_reseg_lock)
    return False, "Starting UMAP…", True


@app_layout.app.callback(
    Output("umap-reseg-status",  "children",  allow_duplicate=True),
    Output("umap-reseg-poll",    "disabled",  allow_duplicate=True),
    Output("umap-reseg-btn",     "disabled",  allow_duplicate=True),
    Output("dataset-version",    "data",      allow_duplicate=True),
    Input("umap-reseg-poll",     "n_intervals"),
    State("dataset-version",     "data"),
    prevent_initial_call=True,
)
def poll_reseg_umap(_, version):
    with core._umap_reseg_lock:
        status   = core._umap_reseg_state["status"]
        message  = core._umap_reseg_state["message"]
        bumped   = core._umap_reseg_state.get("_xenium_bumped", True)
    if status == "done":
        if not bumped:
            with core._umap_reseg_lock:
                core._umap_reseg_state["_xenium_bumped"] = True
            return f"✓ {message}", True, False, (version or 0) + 1
        return f"✓ {message}", True, False, no_update
    if status == "error":
        return f"✗ {message}", True, False, no_update
    return message, False, True, no_update


# ─── Boundary overlay options: enable Proseg/Baysor when results are ready ────

# ─── Seg-source delete run callbacks ──────────────────────────────────────────

app_layout.app.clientside_callback(
    """function(seg_val) {
        if (seg_val && seg_val.includes(":")) {
            return {"display": "block", "marginBottom": "14px"};
        }
        return {"display": "none"};
    }""",
    Output("seg-delete-div",  "style"),
    Input("seg-source",       "value"),
)


@app_layout.app.callback(
    Output("seg-delete-modal",        "is_open"),
    Output("seg-delete-confirm-text", "children"),
    Input("seg-delete-btn",           "n_clicks"),
    Input("seg-delete-cancel-btn",    "n_clicks"),
    State("seg-source",               "value"),
    prevent_initial_call=True,
)
def toggle_seg_delete_modal(delete_clicks, cancel_clicks, seg_value):
    if callback_context.triggered_id == "seg-delete-btn" and delete_clicks:
        runs  = caching._list_cached_seg_runs()
        run   = next((r for r in runs if r["value"] == seg_value), None)
        label = run["label"] if run else seg_value
        return True, f"Permanently delete '{label}'? The cache directory will be removed and cannot be recovered."
    return False, no_update


@app_layout.app.callback(
    Output("seg-delete-modal", "is_open",  allow_duplicate=True),
    Output("seg-source",       "value",    allow_duplicate=True),
    Output("baysor-done",      "data",     allow_duplicate=True),
    Output("proseg-done",      "data",     allow_duplicate=True),
    Output("seg-delete-confirm-text", "children", allow_duplicate=True),
    Input("seg-delete-confirm-btn", "n_clicks"),
    State("seg-source",             "value"),
    State("baysor-done",            "data"),
    State("proseg-done",            "data"),
    prevent_initial_call=True,
)
def confirm_seg_delete(n_clicks, seg_value, baysor_ver, proseg_ver):
    if not n_clicks:
        return no_update, no_update, no_update, no_update, no_update
    import shutil
    runs = caching._list_cached_seg_runs()
    run  = next((r for r in runs if r["value"] == seg_value), None)
    if run:
        out_dir = run["out_dir"]
        _refusal = _run_delete_refusal(core._seg_tool(seg_value), out_dir)
        if _refusal:
            # the dialog stays open with the reason; nothing is removed, nothing unloaded
            print(f"  Delete refused: {_refusal}", flush=True)
            return True, no_update, no_update, no_update, f"✗ {_refusal}"
        try:
            shutil.rmtree(out_dir)
            print(f"  Deleted seg run: {os.path.basename(out_dir)}", flush=True)
        except Exception as e:
            print(f"  Delete failed: {e}", flush=True)
        # Clear in-memory state if this was the loaded run
        tool = core._seg_tool(seg_value)
        if tool == "baysor":
            with core._baysor_lock:
                if (core._baysor_state.get("result") or {}).get("out_dir", "") == out_dir:
                    core._baysor_state.update({"status": "idle", "message": "", "result": None})
        elif tool == "proseg":
            with core._proseg_lock:
                if (core._proseg_state.get("result") or {}).get("out_dir", "") == out_dir:
                    core._proseg_state.update({"status": "idle", "message": "", "result": None})
        elif tool == "imgseg":
            with core._imgseg_lock:
                if (core._imgseg_state.get("result") or {}).get("out_dir", "") == out_dir:
                    core._imgseg_state.update({"status": "idle", "message": "", "result": None})
    return False, "xenium", (baysor_ver or 0) + 1, (proseg_ver or 0) + 1, no_update


@app_layout.app.callback(
    Output("boundary-toggles", "options"),
    Output("seg-source",       "options"),
    Input("baysor-done",       "data"),
    Input("proseg-done",       "data"),
    Input("dataset-version",   "data"),
    State("boundary-toggles",  "options"),
)
def update_boundary_options(baysor_done, proseg_done, _ds_version, boundary_opts):
    baysor_ready = core._baysor_state["status"] == "done" and core._baysor_state["result"] is not None
    proseg_ready = core._proseg_state["status"] == "done" and core._proseg_state["result"] is not None

    # Rebuild boundary options (unchanged logic)
    updated_boundary = []
    for opt in boundary_opts:
        v = opt["value"]
        if v == "proseg":
            updated_boundary.append({**opt, "disabled": not proseg_ready})
        elif v == "baysor":
            updated_boundary.append({**opt, "disabled": not baysor_ready})
        elif v == "imgseg":
            updated_boundary.append({**opt, "disabled": not (core._imgseg_state["status"] == "done"
                                                             and core._imgseg_state["result"] is not None)})
        else:
            updated_boundary.append(opt)

    # Rebuild seg-source options dynamically from cache
    seg_opts = [{"label": "\u2605 Xenium (original)", "value": "xenium"}]
    cached = caching._list_cached_seg_runs()
    baysor_cached = [c for c in cached if c["value"].startswith("baysor:")]
    proseg_cached  = [c for c in cached if c["value"].startswith("proseg:")]
    current_baysor_dir = (core._baysor_state.get("result") or {}).get("out_dir", "")
    current_proseg_dir = (core._proseg_state.get("result") or {}).get("out_dir", "")
    for c in baysor_cached:
        is_current = (c["out_dir"] == current_baysor_dir)
        label = ("\u2605 " if is_current else "") + c["label"]
        seg_opts.append({"label": label, "value": c["value"]})
    for c in proseg_cached:
        is_current = (c["out_dir"] == current_proseg_dir)
        label = ("\u2605 " if is_current else "") + c["label"]
        seg_opts.append({"label": label, "value": c["value"]})
    current_imgseg_dir = (core._imgseg_state.get("result") or {}).get("out_dir", "")
    for c in cached:
        if c["value"].startswith("imgseg:"):
            seg_opts.append({"label": ("\u2605 " if c["out_dir"] == current_imgseg_dir else "") + c["label"],
                             "value": c["value"]})

    return updated_boundary, seg_opts



@app_layout.app.callback(
    Output("baysor-done",  "data",     allow_duplicate=True),
    Output("proseg-done",  "data",     allow_duplicate=True),
    Output("seg-source",   "value",    allow_duplicate=True),
    Output("baysor-poll",  "disabled", allow_duplicate=True),
    Output("proseg-poll",  "disabled", allow_duplicate=True),
    Input("seg-source",    "value"),
    State("baysor-done",   "data"),
    State("proseg-done",   "data"),
    prevent_initial_call=True,
)
def load_seg_run_from_cache(seg_value, baysor_ver, proseg_ver):
    """When user selects a cached run from the dropdown, load it from disk if needed."""
    if not seg_value or ":" not in seg_value:
        return no_update, no_update, no_update, no_update, no_update

    tool, param_tag = seg_value.split(":", 1)
    if tool == "baysor":
        state, lock, loader = core._baysor_state, core._baysor_lock, reseg._load_cached_baysor
        # Enable baysor-poll so it detects the load completion and bumps baysor-done
        poll_on = (no_update, False, no_update)   # seg-source, baysor-poll, proseg-poll
    elif tool == "proseg":
        state, lock, loader = core._proseg_state, core._proseg_lock, reseg._load_cached_proseg
        poll_on = (no_update, no_update, False)
    else:
        return no_update, no_update, no_update, no_update, no_update
    # Resolve the directory through the listing: it knows both directory name
    # forms and skips runs that belong to another dataset with the same folder name.
    run = next((r for r in caching._list_cached_seg_runs() if r["value"] == seg_value), None)
    if run is None:
        print(f"  {tool.capitalize()}: no cached run '{param_tag}' for this dataset", flush=True)
        return no_update, no_update, no_update, no_update, no_update
    out_dir = run["out_dir"]

    with lock:
        current_dir = (state.get("result") or {}).get("out_dir", "")
        if current_dir == out_dir and state["status"] == "done":
            return no_update, no_update, no_update, no_update, no_update
        busy = state["status"] == "running"
        if busy:
            busy_dir = state.get("loading_out_dir")
        else:
            # Claim the state before the thread starts so a second selection
            # (or Run) made meanwhile sees it as busy.
            # The previously loaded run is dropped here: the figures use `result`
            # whenever the dropdown names this tool, so keeping it would show (and
            # annotate / export) the old run under the newly selected name while the
            # new one loads, and for good if that load fails.
            state.update({"status": "running",
                          "message": f"Loading cached {tool.capitalize()} result…",
                          "loading_out_dir": out_dir, "result": None})
    if busy:
        # A live run or another cached load of this tool owns the state. A second
        # thread would let whichever finishes last replace the result behind the
        # dropdown, so put the dropdown back and keep polling the one in progress.
        back = (f"{tool}:{os.path.basename(busy_dir).split('_')[-1]}" if busy_dir else "xenium")
        if back == seg_value:
            return (no_update, no_update) + poll_on
        print(f"  {tool.capitalize()}: busy — select the cached run again when the "
              f"current {'load' if busy_dir else 'run'} has finished", flush=True)
        return (no_update, no_update, back) + poll_on[1:]
    jobs._job_start(tool, f"Load cached {tool.capitalize()} run [{param_tag}]", loader,
               args=(out_dir,), state=state, lock=lock, uses_run=out_dir)
    return (no_update, no_update) + poll_on


# ─── Image-based segmentation (third tab of the modal) ────────────────────────

@app_layout.app.callback(
    Output("modal-imgseg-panel", "style"),
    Output("iseg-backend",       "options"),
    Output("iseg-backend",       "value"),
    Output("iseg-backend-note",  "children"),
    Output("iseg-channel",       "options"),
    Output("iseg-channel",       "value"),
    Output("iseg-channel-cyto",  "options"),
    Input("reseg-algo-tabs",         "value"),
    Input("reseg-patches-confirmed", "data"),
    State("iseg-backend",            "value"),
    State("iseg-channel",            "value"),
    prevent_initial_call=True,
)
def switch_imgseg_tab(algo, _confirmed, backend, channel):
    """Show the panel on its tab and fill it with what this machine and this dataset have:
    the usable backends (Cellpose only with local weights) and the image's channels."""
    if algo != "imgseg":
        return {"display": "none"}, no_update, no_update, no_update, no_update, no_update, no_update
    opts = imgseg._iseg_backend_options()
    values = [o["value"] for o in opts if not o.get("disabled")]
    st = imgseg._iseg_cellpose_status()
    note = "Cellpose: " + st["message"]
    chans = imgseg._iseg_channels()
    if not chans:
        note += ("\nThe morphology image is read from the dataset's SpatialData store: run Step 1 "
                 "(Prepare Patches) to list its channels.")
    ch_opts = [{"label": c, "value": c} for c in chans]
    return ({}, opts, backend if backend in values else "classical", note, ch_opts,
            channel if channel in chans else imgseg._iseg_default_channel(chans), ch_opts)


@app_layout.app.callback(
    Output("iseg-xmin", "value"),
    Output("iseg-xmax", "value"),
    Output("iseg-ymin", "value"),
    Output("iseg-ymax", "value"),
    Input("iseg-use-viewport", "n_clicks"),
    State("spatial-relayout", "data"),
    prevent_initial_call=True,
)
def imgseg_fill_viewport(n_clicks, relayout):
    r = relayout or {}
    x0, x1 = r.get("xaxis.range[0]"), r.get("xaxis.range[1]")
    y0, y1 = r.get("yaxis.range[0]"), r.get("yaxis.range[1]")
    if not n_clicks or None in (x0, x1, y0, y1):
        return no_update, no_update, no_update, no_update
    # plot y = -y µm
    return round(float(x0), 1), round(float(x1), 1), round(-float(y1), 1), round(-float(y0), 1)


@app_layout.app.callback(
    Output("iseg-run-request", "data"),
    Input("reseg-modal-run-btn", "n_clicks"),
    State("reseg-algo-tabs",     "value"),
    prevent_initial_call=True,
)
def request_imgseg_run(n_clicks, algo):
    """Run on the Image tab: hand the click to run_imgseg_modal. (Two callbacks that write
    the same output with allow_duplicate must not have the same inputs, and the Run button
    is the input of run_reseg_modal.)"""
    if not n_clicks or algo != "imgseg":
        raise dash.exceptions.PreventUpdate
    return n_clicks


@app_layout.app.callback(
    Output("reseg-modal-run-status", "children", allow_duplicate=True),
    Output("reseg-modal",            "is_open",  allow_duplicate=True),
    Output("imgseg-poll",            "disabled", allow_duplicate=True),
    Input("iseg-run-request", "data"),
    State("reseg-algo-tabs",         "value"),
    State("reseg-patches-confirmed", "data"),
    State("iseg-backend",      "value"),
    State("iseg-channel",      "value"),
    State("iseg-channel-cyto", "value"),
    State("iseg-xmin", "value"),
    State("iseg-xmax", "value"),
    State("iseg-ymin", "value"),
    State("iseg-ymax", "value"),
    State("iseg-diameter", "value"),
    State("iseg-expand",   "value"),
    State("iseg-min-tx",   "value"),
    State("iseg-patch",    "value"),
    State("iseg-thr",      "value"),
    State("iseg-flow",     "value"),
    State("iseg-cellprob", "value"),
    State("iseg-gpu",      "value"),
    prevent_initial_call=True,
)
def run_imgseg_modal(n_clicks, algo, patches_confirmed, backend, channel, channel_cyto,
                     xmin, xmax, ymin, ymax, diameter, expand, min_tx, patch, thr, flow, cellprob, gpu):
    if not n_clicks or algo != "imgseg":
        raise dash.exceptions.PreventUpdate
    if not patches_confirmed:
        return "✗ Please confirm patches in Step 1 before running segmentation.", no_update, no_update
    backend = backend or "classical"
    if backend.startswith("cellpose"):
        st = imgseg._iseg_cellpose_status()
        if not st["available"] or (":" in backend and backend.split(":", 1)[1] not in st["models"]):
            return "✗ " + st["message"], no_update, no_update
    if not channel:
        return ("✗ Pick the nuclear channel (the channels are listed once Step 1 has built the "
                "SpatialData store)."), no_update, no_update
    if diameter is None or diameter <= 0:
        return "✗ The nucleus diameter must be positive.", no_update, no_update
    _busy = _loaded_run_busy("imgseg")
    if _busy:
        return f"✗ {_busy}", no_update, no_update
    with core._imgseg_lock:
        if core._imgseg_state["status"] == "running":
            return "Already running…", True, False
        core._imgseg_state.update({"status": "running", "message": "Starting…", "result": None,
                                   "loading_out_dir": None})
    # the membrane / cytoplasm channel goes to the models that use one (cyto-type); the
    # nuclei model and the classical backend get the nuclear channel alone
    channels = ([channel_cyto] if channel_cyto and channel_cyto != channel
                and imgseg._iseg_uses_second_channel(backend) else []) + [channel]
    region = {k: v for k, v in dict(x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax).items() if v is not None}
    jobs._job_start("imgseg", "Image segmentation" + (" (region)" if region else ""), imgseg._run_imgseg,
                    kwargs=dict(backend=backend, channels=channels, diameter_um=float(diameter),
                                expand_um=float(expand or 0.0),
                                min_tx=imgseg._ISEG_MIN_TX if min_tx is None else int(min_tx),
                                patch_um=float(patch or imgseg._ISEG_PATCH_UM), thr_scale=float(thr or 1.0),
                                flow_threshold=0.4 if flow is None else float(flow),
                                cellprob_threshold=0.0 if cellprob is None else float(cellprob),
                                gpu="yes" in (gpu or []), **region),
                    state=core._imgseg_state, lock=core._imgseg_lock)
    return "Image segmentation starting…", False, False


@app_layout.app.callback(
    Output("imgseg-status", "children"),
    Output("imgseg-poll",   "disabled", allow_duplicate=True),
    Output("proseg-done",   "data",     allow_duplicate=True),
    Output("seg-source",    "value",    allow_duplicate=True),
    Output("reseg-status",  "children", allow_duplicate=True),
    Input("imgseg-poll",    "n_intervals"),
    State("proseg-done",    "data"),
    State("seg-source",     "value"),
    prevent_initial_call=True,
)
def poll_imgseg(_, done_version, seg_value=None):
    """As poll_baysor / poll_proseg. "proseg-done" is the version every callback that
    depends on the loaded resegmentation listens to, so it is bumped here as well."""
    with core._imgseg_lock:
        status  = core._imgseg_state["status"]
        message = core._imgseg_state["message"]
        out_dir = (core._imgseg_state.get("result") or {}).get("out_dir", "")
    if status == "done":
        param_tag = os.path.basename(out_dir).split("_")[-1] if out_dir else ""
        return (f"✓ {message}", True, (done_version or 0) + 1,
                f"imgseg:{param_tag}" if param_tag else "imgseg", no_update)
    if status == "error":
        if not out_dir and core._seg_tool(seg_value) == "imgseg":
            return f"✗ {message}", True, (done_version or 0) + 1, "xenium", f"✗ Image segmentation: {message}"
        return f"✗ {message}", True, no_update, no_update, f"✗ Image segmentation: {message}"
    if status == "idle":
        return no_update, True, no_update, no_update, no_update
    return message, False, no_update, no_update, f"Image segmentation: {message}"


@app_layout.app.callback(
    Output("iseg-load-request", "data"),
    Input("seg-source",         "value"),
    State("iseg-load-request",  "data"),
    prevent_initial_call=True,
)
def request_imgseg_load(seg_value, previous=None):
    """An image-segmentation run was selected: hand it to load_imgseg_run_from_cache (the
    dropdown itself is the input of load_seg_run_from_cache, see request_imgseg_run)."""
    if not seg_value or core._seg_tool(seg_value) != "imgseg" or ":" not in seg_value:
        raise dash.exceptions.PreventUpdate
    return {"value": seg_value, "n": int((previous or {}).get("n", 0)) + 1}


@app_layout.app.callback(
    Output("seg-source",  "value",    allow_duplicate=True),
    Output("imgseg-poll", "disabled", allow_duplicate=True),
    Input("iseg-load-request", "data"),
    prevent_initial_call=True,
)
def load_imgseg_run_from_cache(request):
    """Selecting a cached image-segmentation run loads it (see load_seg_run_from_cache)."""
    seg_value = request.get("value") if isinstance(request, dict) else request
    if not seg_value or core._seg_tool(seg_value) != "imgseg" or ":" not in seg_value:
        raise dash.exceptions.PreventUpdate
    param_tag = seg_value.split(":", 1)[1]
    run = next((r for r in caching._list_cached_seg_runs() if r["value"] == seg_value), None)
    if run is None:
        print(f"  Image segmentation: no cached run '{param_tag}' for this dataset", flush=True)
        raise dash.exceptions.PreventUpdate
    out_dir = run["out_dir"]
    state, lock = core._imgseg_state, core._imgseg_lock
    with lock:
        if (state.get("result") or {}).get("out_dir", "") == out_dir and state["status"] == "done":
            raise dash.exceptions.PreventUpdate
        busy = state["status"] == "running"
        busy_dir = state.get("loading_out_dir") if busy else None
        if not busy:
            state.update({"status": "running", "message": "Loading cached image segmentation…",
                          "loading_out_dir": out_dir, "result": None})
    if busy:
        back = f"imgseg:{os.path.basename(busy_dir).split('_')[-1]}" if busy_dir else "xenium"
        return (no_update if back == seg_value else back), False
    jobs._job_start("imgseg", f"Load cached image segmentation [{param_tag}]", imgseg._load_cached_imgseg,
                    args=(out_dir,), state=state, lock=lock, uses_run=out_dir)
    return no_update, False


@app_layout.app.callback(
    Output("jobs-poll", "disabled", allow_duplicate=True),
    Input("imgseg-poll", "disabled"),
    State("jobs-poll",   "disabled"),
    prevent_initial_call=True,
)
def wake_jobs_poll_imgseg(_poll, poll_disabled):
    if jobs._jobs_panel_rows() and poll_disabled:
        return False
    return no_update
