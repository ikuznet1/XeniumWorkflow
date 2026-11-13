"""Callbacks: SpaGE, gene options, hi-res overlay, subset indicator, server log, REPL."""

import re
import threading
from dash import no_update, Patch
from dash import Input, Output, State

from . import compat, core, jobs, dataview, spage, subsetting, morph, app_layout


# ─── SpaGE callbacks ──────────────────────────────────────────────────────────

@app_layout.app.callback(
    Output("spage-poll",    "disabled"),
    Output("spage-status",       "children"),
    Output("spage-modal-status", "children"),
    Output("spage-run-btn",      "disabled"),
    Input("spage-run-btn",       "n_clicks"),
    State("spage-rds-path",      "value"),
    State("spage-npv",           "value"),
    State("spage-genes",         "value"),
    State("seg-source",          "value"),
    prevent_initial_call=True,
)
def start_spage(n_clicks, rds_path, n_pv, genes_input, seg_source):
    rds_path = core._ref_rds_path(rds_path or core.REF_RDS_DEFAULT)
    _problem = core._ref_rds_problem(rds_path)
    if _problem:                              # nothing is started: the poll stays off, Run stays enabled
        return True, f"✗ {_problem}", f"✗ {_problem}", False
    with core._spage_lock:
        if core._spage_state["status"] == "running":
            return False, "Already running…", "Already running…", True
        core._spage_state.update({"status": "running", "message": "Starting…"})
        if core._seg_tool(seg_source) == "xenium":
            # a Baysor/Proseg run stores its result on the run; keep the Xenium one
            core._spage_state["result"] = None
            dataview._gene_expr_cache.clear()
            dataview._umap_fig_cache.clear()
    n_pv = int(n_pv or 50)
    genes_str = genes_input or ""
    seg_src = seg_source or "xenium"
    # Copy the current contextvars context so rpy2 conversion rules are available in the thread
    jobs._job_start("spage", "SpaGE imputation", spage._run_spage_imputation,
               args=(rds_path, n_pv, genes_str, seg_src),
               state=core._spage_state, lock=core._spage_lock, copy_context=True)
    _start_msg = "Starting SpaGE…"
    if core._active_counts_mode == "corrected":
        # SpaGE always imputes from the original counts (see _run_spage_imputation)
        _start_msg = ("Starting SpaGE on the ORIGINAL counts (imputation does not use "
                      "SPLIT-corrected counts)…")
        print("  SpaGE: Active Counts is 'SPLIT Corrected', but imputation uses the "
              "original counts", flush=True)
    return False, _start_msg, _start_msg, True


@app_layout.app.callback(
    Output("spage-status",       "children",  allow_duplicate=True),
    Output("spage-modal-status", "children",  allow_duplicate=True),
    Output("spage-modal",        "is_open",   allow_duplicate=True),
    Output("spage-poll",         "disabled",  allow_duplicate=True),
    Output("spage-run-btn",      "disabled",  allow_duplicate=True),
    Output("spage-done",         "data"),
    Input("spage-poll",          "n_intervals"),
    State("spage-done",          "data"),
    State("spage-modal",         "is_open"),
    prevent_initial_call=True,
)
def poll_spage(_, done_version, modal_open):
    with core._spage_lock:
        status  = core._spage_state["status"]
        message = core._spage_state["message"]

    if status == "done":
        return f"✓ {message}", f"✓ {message}", False, True, False, (done_version or 0) + 1
    if status == "error":
        return f"✗ {message}", f"✗ {message}", modal_open, True, False, done_version
    return message, message, modal_open, False, True, done_version


@app_layout.app.callback(
    Output("spage-status",  "children",  allow_duplicate=True),
    Output("spage-done",    "data",      allow_duplicate=True),
    Input("spage-repl-poll", "n_intervals"),
    State("spage-done",      "data"),
    prevent_initial_call=True,  # required because spage-done has allow_duplicate
)
def poll_spage_repl(_, done_version):
    """Always-on poll: picks up SpaGE runs triggered from the REPL."""
    if not core._spage_repl_pending:
        return no_update, no_update
    with core._spage_lock:
        status  = core._spage_state["status"]
        message = core._spage_state["message"]
    if status == "running":
        if message != core._spage_last_logged:
            print(f"  SpaGE: {message}", flush=True)
            core._spage_last_logged = message
        return message, no_update
    if status == "done":
        core._spage_repl_pending = False
        core._spage_last_logged  = ""
        print(f"  SpaGE: done — {message}", flush=True)
        print("  Imputed genes are now available in the Color By → Gene Expression dropdown.", flush=True)
        dataview._gene_expr_cache.clear()  # clear gene cache after imputation
        return f"✓ {message}", (done_version or 0) + 1
    if status == "error":
        core._spage_repl_pending = False
        core._spage_last_logged  = ""
        print(f"  SpaGE error: {message}", flush=True)
        return f"✗ {message}", no_update
    return no_update, no_update


@app_layout.app.callback(
    Output("gene-selector", "options"),
    Input("spage-done",       "data"),
    Input("dataset-version",  "data"),
    Input("seg-source",       "value"),
)
def update_gene_options(_spage_version, _ds_version, seg_source):
    """Rebuild gene dropdown: native genes + imputed genes after SpaGE.
    Imputed genes are excluded when a reseg source (Baysor/Proseg) is active,
    since SpaGE has only been run against Xenium cells."""
    xenium_active = core._seg_tool(seg_source) == "xenium"

    # Separate measured vs imputed genes using gene_var metadata if available
    gene_var = core.DATA.get("gene_var")
    if gene_var is not None and "is_imputed" in gene_var.columns:
        measured = [g for g in core.DATA["gene_names"] if not gene_var.loc[g, "is_imputed"]]
    else:
        measured = list(core.DATA["gene_names"])

    base = [{"label": g, "value": g} for g in sorted(measured)]

    if not xenium_active:
        # Reseg active: show imputed genes specific to this reseg result
        tool = core._seg_tool(seg_source)
        if tool == "baysor":
            with core._baysor_lock:
                res = core._baysor_state.get("result") if core._baysor_state["status"] == "done" else None
        elif tool == "proseg":
            with core._proseg_lock:
                res = core._proseg_state.get("result") if core._proseg_state["status"] == "done" else None
        else:
            res = core._seg_result(seg_source, done_only=True)
        if res:
            imp_genes = res.get("imputed_genes", [])
            if imp_genes:
                imp = [{"label": f"{g} [imp]", "value": f"{g} [imp]"} for g in sorted(imp_genes)]
                return base + imp
            # Streaming SpaGE: genes not written to alt_res["expr"] — use reseg-specific gene list
            streaming_genes = res.get("spage_result_genes") or []
            if streaming_genes:
                imp = [{"label": f"{g} [imp]", "value": f"{g} [imp]"} for g in sorted(streaming_genes)]
                return base + imp
        return base

    # Xenium active: add imputed genes from zarr (already in DATA["expr"]) or _spage_state fallback
    imp_genes = []
    if gene_var is not None and "is_imputed" in gene_var.columns:
        imp_genes = [g for g in core.DATA["gene_names"] if gene_var.loc[g, "is_imputed"]]

    # Plus genes served from _spage_state (old-style result DataFrame or a streamed
    # result): they are not in DATA["expr"] and would otherwise not be listed once
    # the store holds written-back imputed genes.
    with core._spage_lock:
        result       = core._spage_state.get("result")
        result_genes = core._spage_state.get("result_genes") or []
    imp_gene_list = sorted(set(imp_genes) | set(result.columns if result is not None else result_genes))
    if imp_gene_list:
        # Genes a store lists as imputed from SPLIT-corrected counts (written by
        # functions/impute.py --use_corrected) are labelled as such; the value keeps
        # the " [imp]" form every lookup understands.
        _corr_imp = set(core.DATA.get("split_corrected_imputed_genes") or [])
        imp = [{"label": f"{g} [corr+imp]" if g in _corr_imp else f"{g} [imp]",
                "value": f"{g} [imp]"} for g in imp_gene_list]
        return base + imp

    return base


app_layout.app.clientside_callback(
    """function(n) {
        var infoBar = document.getElementById('info-bar');
        var vHandle = document.getElementById('vertical-resize-handle');
        if (n % 2 === 1) {
            // Hide: collapse the entire info bar and resize handle
            if (infoBar) {
                infoBar._savedHeight = infoBar.style.height;
                infoBar.style.height = '28px';
                infoBar.style.minHeight = '28px';
                infoBar.style.overflow = 'hidden';
            }
            if (vHandle) vHandle.style.display = 'none';
            setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 50);
            return [{"display": "none"}, "▲"];
        }
        // Show: restore the info bar
        if (infoBar) {
            infoBar.style.height = infoBar._savedHeight || '180px';
            infoBar.style.minHeight = '';
            infoBar.style.overflow = '';
        }
        if (vHandle) vHandle.style.display = '';
        setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 50);
        return [{"display": "flex", "gap": "10px", "height": "100%", "overflow": "hidden"}, "▼"];
    }""",
    Output("info-bar-body",    "style"),
    Output("info-bar-toggle",  "children"),
    Input("info-bar-toggle",   "n_clicks"),
    prevent_initial_call=True,
)


app_layout.app.clientside_callback(
    """function(triggerVal) {
        // Wire up drag-to-resize between spatial-panel and umap-panel
        function setupResize() {
            var handle = document.getElementById('panel-resize-handle');
            var leftPanel = document.getElementById('spatial-panel');
            var rightPanel = document.getElementById('umap-panel');
            if (!handle || !leftPanel || !rightPanel) {
                setTimeout(setupResize, 300);
                return;
            }
            // Highlight handle on hover
            handle.addEventListener('mouseenter', function() {
                handle.style.backgroundColor = '#444c56';
            });
            handle.addEventListener('mouseleave', function() {
                if (!handle._dragging) handle.style.backgroundColor = 'transparent';
            });
            handle.addEventListener('mousedown', function(e) {
                e.preventDefault();
                handle._dragging = true;
                handle.style.backgroundColor = '#58a6ff';
                var startX = e.clientX;
                var startLeft = leftPanel.getBoundingClientRect().width;
                var startRight = rightPanel.getBoundingClientRect().width;
                function onMove(ev) {
                    var dx = ev.clientX - startX;
                    var newLeft = startLeft + dx;
                    var newRight = startRight - dx;
                    if (newLeft > 150 && newRight > 150) {
                        leftPanel.style.flex = 'none';
                        leftPanel.style.width = newLeft + 'px';
                        rightPanel.style.flex = 'none';
                        rightPanel.style.width = newRight + 'px';
                    }
                }
                function onUp() {
                    handle._dragging = false;
                    handle.style.backgroundColor = 'transparent';
                    document.removeEventListener('mousemove', onMove);
                    document.removeEventListener('mouseup', onUp);
                    // Trigger Plotly resize so graphs reflow
                    window.dispatchEvent(new Event('resize'));
                }
                document.addEventListener('mousemove', onMove);
                document.addEventListener('mouseup', onUp);
            });
        }
        setupResize();

        // Wire up drag-to-resize between plots row and info bar (vertical)
        function setupVerticalResize() {
            var vHandle = document.getElementById('vertical-resize-handle');
            var plotsRow = document.getElementById('plots-row');
            var infoBar = document.getElementById('info-bar');
            if (!vHandle || !plotsRow || !infoBar) {
                setTimeout(setupVerticalResize, 300);
                return;
            }
            vHandle.addEventListener('mouseenter', function() {
                vHandle.style.backgroundColor = '#444c56';
            });
            vHandle.addEventListener('mouseleave', function() {
                if (!vHandle._dragging) vHandle.style.backgroundColor = 'transparent';
            });
            vHandle.addEventListener('mousedown', function(e) {
                e.preventDefault();
                vHandle._dragging = true;
                vHandle.style.backgroundColor = '#58a6ff';
                var startY = e.clientY;
                var startInfoH = infoBar.getBoundingClientRect().height;
                function onMove(ev) {
                    var dy = ev.clientY - startY;
                    var newInfoH = startInfoH - dy;
                    if (newInfoH >= 60 && newInfoH <= 600) {
                        infoBar.style.height = newInfoH + 'px';
                    }
                }
                function onUp() {
                    vHandle._dragging = false;
                    vHandle.style.backgroundColor = 'transparent';
                    document.removeEventListener('mousemove', onMove);
                    document.removeEventListener('mouseup', onUp);
                    window.dispatchEvent(new Event('resize'));
                }
                document.addEventListener('mousemove', onMove);
                document.addEventListener('mouseup', onUp);
            });
        }
        setupVerticalResize();

        return 'ready';
    }""",
    Output("panel-resize-handle", "title"),
    Input("startup-trigger", "n_intervals"),
)


@app_layout.app.callback(
    Output("spatial-plot", "figure", allow_duplicate=True),
    Input("morph-hires-poll", "n_intervals"),
    prevent_initial_call=True,
)
def push_hires_overlay(_):
    """Apply the latest hires tile render to the spatial plot (progressive rendering)."""
    if not morph._morph_hires_queue:
        return no_update
    img_dict = morph._morph_hires_queue.pop(0)
    if not img_dict:
        return no_update  # don't clear the preview if hires result is empty
    patched = Patch()
    patched["layout"]["images"] = [img_dict]
    return patched


@app_layout.app.callback(
    Output("subset-version", "data"),
    Input("subset-poll",     "n_intervals"),
    State("subset-version",  "data"),
)
def poll_subset(_, current):
    """Push _subset_version into the Dash store to trigger plot refresh."""
    if core._subset_version != (current or 0):
        return core._subset_version
    return no_update


@app_layout.app.callback(
    Output("subset-indicator", "style"),
    Output("subset-count",     "children"),
    Input("subset-version",    "data"),
    Input("dataset-version",   "data"),
)
def update_subset_indicator(_, _ds_version=None):
    active = "_df_original" in core.DATA
    if active:
        n_sub  = len(core.DATA["df"])
        n_orig = len(core.DATA["_df_original"])
        return {}, f"{n_sub:,} / {n_orig:,} cells"
    return {"display": "none"}, ""


@app_layout.app.callback(
    Output("subset-version", "data", allow_duplicate=True),
    Input("unsubset-btn",    "n_clicks"),
    prevent_initial_call=True,
)
def clear_subset_btn(_):
    subsetting.unsubset()
    return core._subset_version


@app_layout.app.callback(
    Output("server-log", "children"),
    Input("log-poll", "n_intervals"),
)
def update_server_log(_):
    with core._log_lock:
        lines = list(core._log_buffer)
    return "\n".join(lines[-60:])  # last 60 lines


_repl_globals = compat.ReplNamespace()  # the namespace of the whole app, shared with the REPL

_REPL_LOOPBACK_ADDRS = ("127.0.0.1", "::1", "::ffff:127.0.0.1")
_REPL_LOOPBACK_HOST_RE = re.compile(r"(localhost|127\.0\.0\.1|\[::1\])(:\d{1,5})?", re.IGNORECASE)


def _repl_local_request_refusal(request):
    """Why a REPL request is not accepted in local mode (XENIUM_ENABLE_REPL=1), or None.

    The peer address alone is not enough: a web page whose hostname is re-bound to
    127.0.0.1 (DNS rebinding) and any reverse proxy on this machine both connect from
    loopback. So the Host header must itself name loopback, and requests that carry
    a forwarding header are refused."""
    if request.remote_addr not in _REPL_LOOPBACK_ADDRS:
        return "not from localhost"
    if not _REPL_LOOPBACK_HOST_RE.fullmatch((request.headers.get("Host") or "").strip()):
        return ("Host header is not localhost / 127.0.0.1 / [::1] "
                "(open the app through one of these addresses)")
    for _h in ("X-Forwarded-For", "Forwarded", "X-Forwarded-Host", "X-Real-IP"):
        if request.headers.get(_h):
            return (f"request was forwarded by a proxy ({_h} header); local mode does not "
                    f"serve the REPL through a proxy")
    return None


@app_layout.app.callback(
    Output("repl-input", "value"),
    Input("repl-run",    "n_clicks"),
    Input("repl-input",  "n_submit"),
    State("repl-input",  "value"),
    prevent_initial_call=True,
)
def run_repl(_, __, code):
    if not code or not code.strip():
        return ""
    # The REPL executes arbitrary Python: refuse unless explicitly enabled, and
    # (unless XENIUM_ENABLE_REPL=remote) unless the request comes from this machine.
    if not app_layout._REPL_ENABLED:
        print("REPL is disabled. Start the app with XENIUM_ENABLE_REPL=1 to enable it.", flush=True)
        return ""
    if not app_layout._REPL_ALLOW_REMOTE:
        import flask as _flask
        _why = _repl_local_request_refusal(_flask.request)
        if _why:
            print(f"REPL request refused: {_why}.", flush=True)
            return ""
    with core._log_lock:
        core._log_buffer.append(f">>> {code}")

    def _exec():
        try:
            # Try eval first (expression), fall back to exec (statement)
            try:
                result = eval(code, _repl_globals)
                if result is not None:
                    print(repr(result), flush=True)
            except SyntaxError:
                exec(code, _repl_globals)
        except Exception:
            import traceback
            traceback.print_exc()

    threading.Thread(target=_exec, daemon=True).start()
    return ""
