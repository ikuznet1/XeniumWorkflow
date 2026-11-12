"""Callbacks: differential expression, gene-set scores, transcript density, WebGL wiring."""

import re
from dash import no_update
import dash
from dash import Input, Output, State, ClientsideFunction
import time as _time

from . import core, jobs, dataview, analysis, stats, regions, app_layout, cb_spatial


# ─── Analysis callbacks: differential expression ──────────────────────────────

@app_layout.app.callback(
    Output("jobs-poll", "disabled", allow_duplicate=True),
    Input("de-poll",      "disabled"),
    Input("geneset-poll", "disabled"),
    Input("density-poll", "disabled"),
    State("jobs-poll",    "disabled"),
    prevent_initial_call=True,
)
def wake_jobs_poll_analysis(_de, _gs, _dn, poll_disabled):
    """As wake_jobs_poll, for the three analysis jobs."""
    if jobs._jobs_panel_rows() and poll_disabled:
        return False
    return no_update


def _an_job_claim(state: dict, lock) -> bool:
    """Mark a job kind as running before its thread starts; False when it already runs."""
    with lock:
        if state.get("status") == "running":
            return False
        state["status"] = "running"
        state["message"] = "Starting…"
        return True


def _de_set_selection(poly, n, version):
    if not poly or len(poly) < 3:
        raise dash.exceptions.PreventUpdate
    if poly[0] != poly[-1]:
        poly = poly + [poly[0]]
    with analysis._de_lock:
        analysis._de_state["selection"] = {"polygon_xy": poly, "n": int(n), "t": _time.time()}
    return (version or 0) + 1


@app_layout.app.callback(
    Output("de-selection", "data"),
    Input("spatial-plot",  "selectedData"),
    State("de-selection",  "data"),
    State("seg-source",    "value"),
    prevent_initial_call=True,
)
def de_capture_plotly_selection(selected_data, version, seg_source):
    """Keep the outline of the last lasso / box drawn in the Plotly view as the "current
    selection" of the Differential Expression tab (plot space: y = -µm). The Save ROI
    dialog the same selection opens is independent of this."""
    if not selected_data or not selected_data.get("points"):
        raise dash.exceptions.PreventUpdate
    lp, rg = selected_data.get("lassoPoints"), selected_data.get("range")
    if lp and lp.get("x") and len(lp["x"]) >= 3:
        poly = [[float(x), float(y)] for x, y in zip(lp["x"], lp["y"])]
    elif rg and rg.get("x") and rg.get("y"):
        (xa, xb), (ya, yb) = rg["x"], rg["y"]
        poly = [[float(xa), float(ya)], [float(xb), float(ya)],
                [float(xb), float(yb)], [float(xa), float(yb)]]
    else:
        ids = [pt.get("customdata") for pt in selected_data["points"]
               if pt.get("customdata") is not None]
        _alt = analysis._an_alt_res(seg_source)
        poly = regions._roi_compute_hull(ids, src_df=_alt["cells_df"] if _alt else None)
    return _de_set_selection(poly, len(selected_data["points"]), version)


@app_layout.app.callback(
    Output("de-selection", "data", allow_duplicate=True),
    Input("webgl-select",  "data"),
    State("de-selection",  "data"),
    prevent_initial_call=True,
)
def de_capture_webgl_selection(webgl_sel, version):
    """As de_capture_plotly_selection, for a lasso / rectangle drawn in the WebGL view
    (its outline is in µm with y down)."""
    if not webgl_sel or not webgl_sel.get("polygon"):
        raise dash.exceptions.PreventUpdate
    poly = [[float(x), -float(y)] for x, y in webgl_sel["polygon"]]
    return _de_set_selection(poly, webgl_sel.get("count") or 0, version)


@app_layout.app.callback(
    Output("de-group-a", "options"),
    Output("de-group-b", "options"),
    Output("de-selection-info", "children"),
    Input("plots-tabs",        "active_tab"),
    Input("de-selection",      "data"),
    Input("de-groups-version", "data"),
    Input("roi-done",          "data"),
    Input("annot-done",        "data"),
    Input("dataset-version",   "data"),
    Input("subset-version",    "data"),
    Input("seg-source",        "value"),
    Input("cluster-method",    "value"),
    Input("color-by",          "value"),
    Input("counts-mode-store", "data"),
    Input("baysor-done",       "data"),
    Input("proseg-done",       "data"),
    Input("extra-datasets-version", "data"),
)
def de_update_group_options(active_tab, _sel, _grp, _roi, _annot, _ds, _sub, seg_source, method,
                            color_by, counts_mode, _bd, _pd, _ex):
    """Group pickers of the Differential Expression tab (built while the tab is open)."""
    if active_tab != "tab-de":
        raise dash.exceptions.PreventUpdate
    opts = analysis._de_group_options(seg_source, method, color_by, counts_mode == "corrected")
    with analysis._de_lock:
        sel = analysis._de_state.get("selection")
    info = (f"Current selection: {int(sel.get('n') or 0):,} cells." if sel
            else "No current selection (draw a lasso or box on the tissue).")
    return opts, opts, info


@app_layout.app.callback(
    Output("de-groups-version", "data"),
    Output("de-status",         "children", allow_duplicate=True),
    Input("de-save-group-btn",  "n_clicks"),
    State("de-group-name",      "value"),
    State("de-groups-version",  "data"),
    prevent_initial_call=True,
)
def de_save_group(n_clicks, name, version):
    """Save the current selection as a named group (kept in the cache dir per dataset)."""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    name = (name or "").strip()
    if not name:
        return no_update, "Enter a name for the group."
    with analysis._de_lock:
        sel = analysis._de_state.get("selection")
        if not sel:
            return no_update, "There is no current selection: draw a lasso or box on the tissue first."
        if any(g["name"] == name for g in analysis._de_state["groups"]):
            return no_update, f"A group named '{name}' already exists."
        analysis._de_state["groups"].append({"name": name, "polygon_xy": sel["polygon_xy"]})
        analysis._de_groups_save()
    return (version or 0) + 1, f"Saved the current selection as group '{name}'."


@app_layout.app.callback(
    Output("de-groups-version", "data",     allow_duplicate=True),
    Output("de-status",         "children", allow_duplicate=True),
    Output("de-group-a",        "value"),
    Input("de-delete-group-btn", "n_clicks"),
    State("de-group-a",          "value"),
    State("de-groups-version",   "data"),
    prevent_initial_call=True,
)
def de_delete_group(n_clicks, group_a, version):
    """Delete the saved group picked in A."""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not str(group_a or "").startswith("grp:"):
        return no_update, "Pick a saved group in A to delete it.", no_update
    with analysis._de_lock:
        analysis._de_state["groups"] = [g for g in analysis._de_state["groups"] if g["name"] != group_a[4:]]
        analysis._de_groups_save()
    return (version or 0) + 1, f"Deleted group '{group_a[4:]}'.", None


@app_layout.app.callback(
    Output("de-run-btn", "disabled"),
    Output("de-poll",    "disabled"),
    Output("de-status",  "children"),
    Input("de-run-btn",  "n_clicks"),
    State("de-group-a",  "value"),
    State("de-group-b",  "value"),
    State("seg-source",  "value"),
    State("cluster-method", "value"),
    State("color-by",    "value"),
    State("counts-mode-store", "data"),
    State("de-mode",           "value"),
    State("de-pb-population",  "value"),
    State("de-pb-min-cells",   "value"),
    State("de-pb-min-count",   "value"),
    State("de-pb-shrink",      "value"),
    prevent_initial_call=True,
)
def run_de(n_clicks, group_a, group_b, seg_source, method, color_by, counts_mode,
           mode="cells", population=None, min_cells=None, min_count=None, shrink=None):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if mode == "pseudobulk":
        if not population:
            return False, True, "Pick the cell population to sum the counts over."
        # rpy2 conversion rules for this thread, so the job's context copy carries them
        try:
            import rpy2.robjects.conversion as _rconv
            import rpy2.robjects as _ro_mod
            _rconv.set_conversion(_ro_mod.default_converter)
        except Exception as exc:
            return False, True, f"rpy2 / R is not available ({exc}). " + stats._PB_INSTALL_HINT
        if not _an_job_claim(analysis._de_state, analysis._de_lock):
            return True, False, "A differential expression run is already in progress."
        jobs._job_start("de", "Pseudobulk differential expression", stats._run_pseudobulk_de,
                   args=(population, seg_source or "xenium", method, color_by,
                         counts_mode == "corrected",
                         stats._PB_MIN_CELLS if min_cells is None else min_cells,
                         stats._PB_MIN_COUNT if min_count is None else min_count,
                         "on" in (shrink if shrink is not None else ["on"])),
                   state=analysis._de_state, lock=analysis._de_lock, copy_context=True)
        return True, False, "Running…"
    if not group_a or not group_b:
        return False, True, "Pick a group for A and for B."
    if group_a == group_b:
        return False, True, "A and B are the same group."
    if not _an_job_claim(analysis._de_state, analysis._de_lock):
        return True, False, "A differential expression run is already in progress."
    jobs._job_start("de", "Differential expression", analysis._run_de,
               args=(group_a, group_b, seg_source or "xenium", method, color_by,
                     counts_mode == "corrected"),
               state=analysis._de_state, lock=analysis._de_lock)
    return True, False, "Running…"


@app_layout.app.callback(
    Output("de-status",  "children", allow_duplicate=True),
    Output("de-poll",    "disabled", allow_duplicate=True),
    Output("de-run-btn", "disabled", allow_duplicate=True),
    Output("de-done",    "data"),
    Input("de-poll",     "n_intervals"),
    State("de-done",     "data"),
    prevent_initial_call=True,
)
def poll_de(_, done_ver):
    with analysis._de_lock:
        status, msg = analysis._de_state["status"], analysis._de_state["message"]
    if status == "done":
        return msg, True, False, (done_ver or 0) + 1
    if status in ("error", "idle"):
        return msg, True, False, no_update
    return msg, False, True, no_update


@app_layout.app.callback(
    Output("de-volcano-plot", "figure"),
    Output("de-table",        "data"),
    Input("de-done",          "data"),
    Input("dataset-version",  "data"),
    Input("de-plot-w",        "value"),
    Input("de-plot-h",        "value"),
)
def update_de_results(_done, _ds_version, w, h):
    with analysis._de_lock:
        res = analysis._de_state.get("result")
    return analysis.make_de_volcano_fig(width=w, height=h), analysis._de_table_records(res)


def _de_pick(gene, gene_options):
    have = {(o.get("value") if isinstance(o, dict) else o) for o in (gene_options or [])}
    if not gene or gene not in have:
        raise dash.exceptions.PreventUpdate
    return gene


@app_layout.app.callback(
    Output("gene-selector", "value", allow_duplicate=True),
    Input("de-volcano-plot", "clickData"),
    State("gene-selector",   "options"),
    prevent_initial_call=True,
)
def de_pick_gene_from_volcano(click, gene_options):
    """A click on a volcano point selects that gene in the sidebar."""
    pt = (click or {}).get("points") or [{}]
    return _de_pick(pt[0].get("customdata") or pt[0].get("text"), gene_options)


@app_layout.app.callback(
    Output("gene-selector", "value", allow_duplicate=True),
    Input("de-table",        "active_cell"),
    State("de-table",        "derived_viewport_data"),
    State("gene-selector",   "options"),
    prevent_initial_call=True,
)
def de_pick_gene_from_table(active_cell, shown_rows, gene_options):
    """A click on a table row selects that gene in the sidebar."""
    row = (active_cell or {}).get("row", -1)
    if not shown_rows or not (0 <= row < len(shown_rows)):
        raise dash.exceptions.PreventUpdate
    return _de_pick(shown_rows[row].get("gene"), gene_options)


@app_layout.app.callback(
    Output("de-download", "data"),
    Input("de-download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_de_table(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    with analysis._de_lock:
        res = analysis._de_state.get("result")
    if not res:
        raise dash.exceptions.PreventUpdate
    t = res["table"].sort_values(["padj", "pval"], kind="stable").drop(columns=["neg_log10_padj"])
    _safe = lambda v: re.sub(r"[^A-Za-z0-9_.-]+", "_", str(v))[:40]
    if res.get("kind") == "pseudobulk":
        return dict(content=stats._pb_csv_header(res) + t.to_csv(index=False),
                    filename=f"pseudobulk_{_safe(res['population'])}_{_safe(res['label_a'])}_vs_"
                             f"{_safe(res['label_b'])}.csv")
    header = (f"# A: {res['label_a']} (n={res['n_a']}); B: {res['label_b']} (n={res['n_b']}); "
              f"counts: {'SPLIT-corrected' if res['corrected'] else 'original'}; "
              f"segmentation: {res['seg_source']}; Wilcoxon rank-sum on log1p(CP10K), cells as "
              f"observations; BH FDR\n")
    return dict(content=header + t.to_csv(index=False),
                filename=f"de_{_safe(res['label_a'])}_vs_{_safe(res['label_b'])}.csv")


# ─── Analysis callbacks: gene-set scores ──────────────────────────────────────

@app_layout.app.callback(
    Output("geneset-select", "options"),
    Input("geneset-version", "data"),
    Input("dataset-version", "data"),
)
def update_geneset_options(_version, _ds_version):
    with analysis._geneset_lock:
        sets = dict(analysis._geneset_state["sets"])
    return [{"label": f"{n} ({len(g)} genes)", "value": n} for n, g in sets.items()]


@app_layout.app.callback(
    Output("geneset-name",  "value"),
    Output("geneset-genes", "value"),
    Input("geneset-select", "value"),
    prevent_initial_call=True,
)
def fill_geneset_form(name):
    with analysis._geneset_lock:
        genes = analysis._geneset_state["sets"].get(name)
    if not name or genes is None:
        raise dash.exceptions.PreventUpdate
    return name, "\n".join(genes)


@app_layout.app.callback(
    Output("color-by", "options", allow_duplicate=True),
    Input("geneset-version", "data"),
    State("color-by",  "options"),
    prevent_initial_call=True,
)
def update_color_by_for_genesets(_version, current_options):
    return cb_spatial._refresh_colorby_options(current_options)


@app_layout.app.callback(
    Output("geneset-run-btn", "disabled"),
    Output("geneset-poll",    "disabled"),
    Output("geneset-status",  "children"),
    Input("geneset-run-btn",  "n_clicks"),
    State("geneset-name",     "value"),
    State("geneset-genes",    "value"),
    State("seg-source",       "value"),
    State("counts-mode-store", "data"),
    prevent_initial_call=True,
)
def run_geneset_score(n_clicks, name, text, seg_source, counts_mode):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    name = re.sub(r"\s+", " ", (name or "").strip())
    if not name:
        return False, True, "Enter a name for the gene set."
    genes = analysis._gs_parse(text)
    if len(genes) < 2:
        return False, True, "Paste at least 2 gene names."
    samples = analysis._an_samples(seg_source, counts_mode == "corrected")
    if not samples:
        return False, True, "The active view has no expression matrix."
    found, missing = analysis._gs_match(genes, analysis._an_common_genes(samples))
    if len(found) < 2:
        return (False, True,
                f"Only {len(found)} of the {len(genes)} genes are measured in this dataset; at "
                f"least 2 are needed. Not found: {', '.join(missing[:40])}"
                f"{'…' if len(missing) > 40 else ''}")
    if not _an_job_claim(analysis._geneset_state, analysis._geneset_lock):
        return True, False, "A gene set is being scored."
    with analysis._geneset_lock:
        analysis._geneset_state["sets"][name] = genes
        analysis._geneset_state["last"] = name
        analysis._gs_save()
    jobs._job_start("geneset", f"Gene-set score: {name}", analysis._run_geneset_score,
               args=(name, seg_source or "xenium", counts_mode == "corrected"),
               state=analysis._geneset_state, lock=analysis._geneset_lock)
    return True, False, "Scoring…"


@app_layout.app.callback(
    Output("geneset-status",  "children", allow_duplicate=True),
    Output("geneset-poll",    "disabled", allow_duplicate=True),
    Output("geneset-run-btn", "disabled", allow_duplicate=True),
    Output("geneset-version", "data"),
    Output("color-by",        "value",    allow_duplicate=True),
    Input("geneset-poll",     "n_intervals"),
    State("geneset-version",  "data"),
    prevent_initial_call=True,
)
def poll_geneset(_, version):
    with analysis._geneset_lock:
        status, msg = analysis._geneset_state["status"], analysis._geneset_state["message"]
        last = analysis._geneset_state.get("last")
    if status == "done":
        # the new option is added by update_color_by_for_genesets; selecting it shows the score
        return msg, True, False, (version or 0) + 1, (f"geneset:{last}" if last else no_update)
    if status in ("error", "idle"):
        return msg, True, False, (version or 0) + 1, no_update
    return msg, False, True, no_update, no_update


@app_layout.app.callback(
    Output("geneset-version", "data",     allow_duplicate=True),
    Output("geneset-status",  "children", allow_duplicate=True),
    Output("geneset-select",  "value"),
    Output("color-by",        "value",    allow_duplicate=True),
    Input("geneset-delete-btn", "n_clicks"),
    State("geneset-select",   "value"),
    State("geneset-version",  "data"),
    State("color-by",         "value"),
    prevent_initial_call=True,
)
def delete_geneset(n_clicks, name, version, color_by):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not name:
        return no_update, "Pick a saved gene set to delete it.", no_update, no_update
    with analysis._geneset_lock:
        analysis._geneset_state["sets"].pop(name, None)
        analysis._geneset_state["version"] += 1
        analysis._gs_save()
    return ((version or 0) + 1, f"Deleted gene set '{name}'.", None,
            "transcript_counts" if color_by == f"geneset:{name}" else no_update)


# ─── Analysis callbacks: transcript density ───────────────────────────────────

@app_layout.app.callback(
    Output("density-gene", "options"),
    Output("density-gene", "value"),
    Input("dataset-version", "data"),
    State("density-gene", "value"),
)
def update_density_gene_options(_ds_version, current=None):
    panel = sorted(dataview._panel_gene_names())
    opts = ([{"label": "All genes (every gene transcript)", "value": analysis._DENSITY_ALL}]
            + [{"label": g, "value": g} for g in panel])
    keep = current if current == analysis._DENSITY_ALL or current in set(panel) else None
    return opts, (no_update if keep == current else keep)


@app_layout.app.callback(
    Output("density-run-btn", "disabled"),
    Output("density-poll",    "disabled"),
    Output("density-status",  "children"),
    Input("density-run-btn",  "n_clicks"),
    State("density-gene",     "value"),
    prevent_initial_call=True,
)
def run_density(n_clicks, gene):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not gene:
        return False, True, "Pick a gene (or 'All genes')."
    if not any(s["data_dir"] for s in analysis._density_samples()):
        return False, True, ("Transcript density needs the Xenium output directory "
                             "(transcripts.parquet); this dataset was loaded from a .zarr.")
    if gene != analysis._DENSITY_ALL and gene not in set(dataview._panel_gene_names()):
        return False, True, f"'{gene}' is not a measured gene of this dataset."
    if not _an_job_claim(analysis._density_state, analysis._density_lock):
        return True, False, "A transcript density grid is being computed."
    jobs._job_start("density", "Transcript density: " + ("all genes" if gene == analysis._DENSITY_ALL else gene),
               analysis._run_density, args=([gene],), state=analysis._density_state, lock=analysis._density_lock)
    return True, False, "Binning transcripts…"


@app_layout.app.callback(
    Output("density-status",  "children", allow_duplicate=True),
    Output("density-poll",    "disabled", allow_duplicate=True),
    Output("density-run-btn", "disabled", allow_duplicate=True),
    Output("density-done",    "data"),
    Input("density-poll",     "n_intervals"),
    State("density-done",     "data"),
    prevent_initial_call=True,
)
def poll_density(_, done_ver):
    with analysis._density_lock:
        status, msg = analysis._density_state["status"], analysis._density_state["message"]
    if status == "done":
        return msg, True, False, (done_ver or 0) + 1
    if status in ("error", "idle"):
        return msg, True, False, no_update
    return msg, False, True, no_update


@app_layout.app.callback(
    Output("density-store", "data"),
    Output("density-hint",  "children"),
    Input("density-enable",  "value"),
    Input("density-gene",    "value"),
    Input("density-bin",     "value"),
    Input("density-opacity", "value"),
    Input("density-max",     "value"),
    Input("density-done",    "data"),
    Input("dataset-version", "data"),
    Input("extra-datasets-version", "data"),
)
def update_density_store(enable, gene, bin_um, opacity, vmax, done_ver, ds_version, extra_ver=0):
    """What both renderers draw as the heat layer: {on, gene, bin, opacity, max, version}.
    With several samples the layer is on as soon as one of them has a grid; the hint
    names the samples that have none yet."""
    want = "on" in (enable or []) and bool(gene)
    st = analysis._density_status(gene) if gene else {"ready": [], "missing": [], "unavailable": []}
    ready = bool(st["ready"])
    multi = len(analysis._density_samples()) > 1
    hint = ""
    if gene and not ready:
        hint = "No grid for this gene yet: press Compute."
    elif gene and ready:
        name = "all genes" if gene == analysis._DENSITY_ALL else gene
        if not multi:
            g = analysis._density_grid(gene, bin_um or 10)
            total, n_bins, b, auto = g["total"], int((g["grid"] > 0).sum()), g["bin"], g["vmax"]
        else:
            wins = analysis._density_windows(gene, bin_um or 10, max_cells=10 ** 12)
            total, b, auto = wins["total"], wins["bin"], wins["vmax"]
            n_bins = int(sum(int((p["values"] > 0).sum()) for p in wins["parts"]))
        hint = (f"{name}: {total:,} transcripts (Q ≥ 20) in {n_bins:,} "
                f"non-empty {b:g} µm bins; colour scale 0 – "
                f"{(vmax if vmax else auto):g} transcripts per bin"
                f"{'' if vmax else ' (99.5th percentile)'}.")
        if multi:
            hint += f" Samples drawn: {', '.join(st['ready'])} (one colour scale)."
            if st["missing"]:
                hint += f" No grid yet for {', '.join(st['missing'])}: press Compute."
            if st["unavailable"]:
                hint += f" {', '.join(st['unavailable'])} loaded from a .zarr: no transcripts."
    return ({"on": bool(want and ready), "gene": gene, "bin": bin_um or 10,
             "opacity": 0.7 if opacity is None else opacity,
             "max": vmax if vmax else None,
             "version": f"{ds_version}|{done_ver}" + (f"|{extra_ver}" if extra_ver else "")}, hint)


# The heat layer and the gene-set colours in the WebGL view (the Plotly view gets both
# through update_plots).
app_layout.app.clientside_callback(
    ClientsideFunction(namespace="xenium_viewer", function_name="analysis"),
    Input("renderer-mode",   "value"),
    Input("density-store",   "data"),
    Input("geneset-version", "data"),
)


# ─── WebGL viewer wiring ──────────────────────────────────────────────────────
@app_layout.app.callback(
    Output("webgl-tx-genes", "options"),
    Output("webgl-tx-genes", "value"),
    Input("dataset-version", "data"),
    State("webgl-tx-genes", "value"),
)
def update_webgl_tx_gene_options(_ds_version, current=None):
    """Panel genes that can be drawn as transcripts. dataset-version is also bumped when
    the dataset stays (Rerun UMAP): the picked genes that the panel still has are kept,
    as update_cluster_options keeps its selection."""
    panel = sorted(dataview._panel_gene_names())
    have = set(panel)
    current = current if isinstance(current, list) else ([current] if current else [])
    keep = [g for g in current if g in have]
    return ([{"label": g, "value": g} for g in panel],
            no_update if keep == current else keep)


@app_layout.app.callback(
    Output("roi-pending",      "data",     allow_duplicate=True),
    Output("roi-save-modal",   "is_open",  allow_duplicate=True),
    Output("roi-save-preview", "children", allow_duplicate=True),
    Output("roi-save-error",   "children", allow_duplicate=True),
    Input("webgl-select", "data"),
    prevent_initial_call=True,
)
def capture_webgl_selection(sel):
    """A lasso / rectangle drawn in the WebGL view: the drawn outline itself becomes the
    pending ROI (the Plotly path has only the selected cells and uses their convex hull)."""
    if not sel or not sel.get("polygon"):
        raise dash.exceptions.PreventUpdate
    n = int(sel.get("count") or 0)
    pts = [[float(x), -float(y)] for x, y in sel["polygon"]]   # ROI space is plot space (y = -µm)
    if len(pts) < 3 or n < 3:
        return None, True, "", "Need at least 3 cells inside the outline to define an ROI."
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    try:
        from shapely.geometry import Polygon as _SPoly
        if not _SPoly(pts).is_valid:
            return None, True, "", "The outline crosses itself; draw a simple closed shape."
    except Exception:
        pass
    # The outline lives in the roi-pending store only (save_roi reads it from there) and in
    # the glue, which drops it when the modal closes - also by backdrop / Esc, which no
    # server callback sees. It is not kept in _roi_state["pending_hull"]: nothing would
    # clear it there, and a later Plotly figure would draw it as a phantom outline.
    with core._roi_lock:
        core._roi_state["pending_hull"] = None
    return pts, True, f"{n:,} cells selected — enter name and class below.", ""


# Drives window.XeniumViewer (assets/xenium_viewer.js) from the sidebar controls; the glue
# lives in assets/xenium_viewer_dash.js. Only state is sent: the viewer fetches its own data.
app_layout.app.clientside_callback(
    ClientsideFunction(namespace="xenium_viewer", function_name="sync"),
    Output("spatial-plot", "style"),
    Output("webgl-view",   "style"),
    Output("webgl-umap",   "style"),
    Output("morph-zlevel",      "disabled"),   # the WebGL image layer has no use for it
    Output("morph-zlevel-wrap", "title"),
    Input("renderer-mode",     "value"),
    Input("color-by",          "value"),
    Input("cluster-method",    "value"),
    Input("gene-selector",     "value"),
    Input("point-size",        "value"),
    Input("point-opacity",     "value"),
    Input("boundary-toggles",  "value"),
    Input("seg-source",        "value"),
    Input("counts-mode-store", "data"),
    Input("colorbar-min",      "value"),
    Input("colorbar-max",      "value"),
    Input("webgl-tx-genes",    "value"),
    Input("morph-enable",      "value"),
    Input("morph-channels",    "value"),
    Input("morph-opacity",     "value"),
    Input("morph-brightness",  "value"),
    Input("dataset-version",   "data"),
    Input("subset-version",    "data"),
    Input("baysor-done",       "data"),
    Input("proseg-done",       "data"),
    Input("annot-done",        "data"),
    Input("spage-done",        "data"),
    Input("split-done",        "data"),
    Input("roi-done",          "data"),
    Input("niche-done",        "data"),
    Input("webgl-tool",        "value"),
    Input("umap-panel",        "style"),     # the UMAP viewer is only fed while its panel is open
    Input("roi-save-modal",    "is_open"),   # closing it drops the lasso highlight / pending outline
    Input("extra-datasets-version", "data"), # samples added / removed: cells, colours, separators
    State("spatial-relayout",  "data"),      # Plotly's zoom, applied when switching to WebGL
)

# Overlays the Plotly view draws from other controls: the ruler, the SpatialData tissue ROI /
# patch grid and the CellNEST edges. The glue keeps these values and fetches what it needs.
app_layout.app.clientside_callback(
    ClientsideFunction(namespace="xenium_viewer", function_name="extras"),
    Input("renderer-mode",            "value"),
    Input("measure-store",            "data"),
    Input("sdata-show-roi",           "value"),
    Input("sdata-show-patches",       "value"),
    Input("sdata-done",               "data"),
    Input("cellnest-version",         "data"),
    Input("cellnest-overlay-enable",  "value"),
    Input("cellnest-lr-pair",         "value"),
    Input("cellnest-top-n",           "value"),
    Input("cellnest-edge-width",      "value"),
    Input("cellnest-color-by",        "value"),
    Input("cellnest-options",         "value"),
    Input("cellnest-filter-ligand",   "value"),
    Input("cellnest-filter-receptor", "value"),
    Input("cellnest-filter-sender",   "value"),
    Input("cellnest-filter-receiver", "value"),
    Input("cellnest-annot-col",       "value"),
)

app_layout.app.clientside_callback(
    ClientsideFunction(namespace="xenium_viewer", function_name="exportPng"),
    Input("webgl-png-btn", "n_clicks"),
    prevent_initial_call=True,
)
