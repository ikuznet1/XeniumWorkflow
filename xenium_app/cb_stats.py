"""Callbacks: sample design, pseudobulk DE controls, composition, Moran's I, ligand-receptor."""

import re
import base64
import dash
from dash import no_update
from dash import Input, Output, State, ALL

from . import core, jobs, dataview, analysis, stats, app_layout, cb_analysis


# ─── Samples and conditions (one editor for pseudobulk DE and composition) ────

@app_layout.app.callback(
    Output("design-modal",  "is_open"),
    Output("design-rows",   "children"),
    Output("design-name-a", "value"),
    Output("design-name-b", "value"),
    Output("design-status", "children"),
    Input("de-design-open-btn",   "n_clicks"),
    Input("comp-design-open-btn", "n_clicks"),
    Input("coloc-design-open-btn", "n_clicks"),
    prevent_initial_call=True,
)
def open_design_modal(de_clicks, comp_clicks, coloc_clicks=None):
    """Open the editor with one row per loaded sample."""
    if not de_clicks and not comp_clicks and not coloc_clicks:
        raise dash.exceptions.PreventUpdate
    names = stats._design_names()
    return True, stats._design_rows_children(), names["A"], names["B"], ""


@app_layout.app.callback(
    Output("design-modal", "is_open", allow_duplicate=True),
    Input("design-close-btn", "n_clicks"),
    prevent_initial_call=True,
)
def close_design_modal(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    return False


@app_layout.app.callback(
    Output("design-version", "data"),
    Output("design-status",  "children", allow_duplicate=True),
    Input("design-apply-btn", "n_clicks"),
    State({"type": "design-cond",  "index": ALL}, "value"),
    State({"type": "design-batch", "index": ALL}, "value"),
    State({"type": "design-cond",  "index": ALL}, "id"),
    State("design-name-a",  "value"),
    State("design-name-b",  "value"),
    State("design-version", "data"),
    State({"type": "design-donor", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def apply_design(n_clicks, conds, batches, ids, name_a, name_b, version, donors=None):
    """Store the assignment of the samples listed in the editor."""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    samples = {s["index"]: s for s in stats._design_samples()}
    rows = []
    donors = list(donors or []) + [""] * len(ids or [])
    for cid, cond, batch, donor in zip(ids or [], conds or [], batches or [], donors):
        s = samples.get((cid or {}).get("index"))
        if s is None:
            return no_update, ("The loaded samples changed while the editor was open: close it "
                               "and open it again.")
        rows.append({"key": s["key"], "cond": cond or "", "batch": batch or "", "donor": donor or ""})
    if (name_a or "A").strip() == (name_b or "B").strip():
        return no_update, "Give the two conditions different names."
    stats._design_set(rows, {"A": name_a, "B": name_b})
    n_a = sum(1 for r in rows if r["cond"] == "A")
    n_b = sum(1 for r in rows if r["cond"] == "B")
    u = stats._sl_units([r["cond"] for r in rows], [r["donor"] for r in rows])
    saved = "saved in the cache directory" if stats._design_path() else "kept for this session"
    return (version or 0) + 1, (f"Saved: {n_a} sample{'s' if n_a != 1 else ''} in A, {n_b} in B, "
                                f"{len(rows) - n_a - n_b} excluded ({saved}). Units of the "
                                f"sample-level tests: {u['note']}."
                                + (f" Donors in one condition only (left out of paired tests): "
                                   f"{', '.join(u['dropped'])}." if u["dropped"] else ""))


@app_layout.app.callback(
    Output("de-design-summary",   "children"),
    Output("comp-design-summary", "children"),
    Input("design-version",         "data"),
    Input("extra-datasets-version", "data"),
    Input("dataset-version",        "data"),
)
def update_design_summaries(_design, _extra, _ds):
    text = stats._design_summary()
    return text, text


# ─── Differential expression: the pseudobulk mode ─────────────────────────────

@app_layout.app.callback(
    Output("de-cells-controls", "style"),
    Output("de-pb-controls",    "style"),
    Output("de-caveat",         "children"),
    Input("de-mode", "value"),
)
def toggle_de_mode(mode):
    """Show the controls and the caveat of the chosen test."""
    show, hide = {"flexShrink": "0"}, {"display": "none", "flexShrink": "0"}
    if mode == "pseudobulk":
        return hide, show, stats._PB_CAVEAT
    return show, hide, analysis._DE_CAVEAT


@app_layout.app.callback(
    Output("de-pb-population", "options"),
    Output("de-pb-population", "value"),
    Input("plots-tabs",        "active_tab"),
    Input("de-mode",           "value"),
    Input("de-selection",      "data"),
    Input("de-groups-version", "data"),
    Input("roi-done",          "data"),
    Input("annot-done",        "data"),
    Input("dataset-version",   "data"),
    Input("subset-version",    "data"),
    Input("seg-source",        "value"),
    Input("cluster-method",    "value"),
    Input("color-by",          "value"),
    Input("extra-datasets-version", "data"),
    State("de-pb-population",  "value"),
)
def update_pb_population_options(active_tab, mode, _sel, _grp, _roi, _annot, _ds, _sub, seg_source,
                                 method, color_by, _ex, current=None):
    """Cell populations of the pseudobulk mode (built while it is shown)."""
    if active_tab != "tab-de" or mode != "pseudobulk":
        raise dash.exceptions.PreventUpdate
    opts = stats._pb_population_options(seg_source, method, color_by)
    have = {o["value"] for o in opts}
    return opts, (no_update if current in have else "all")


@app_layout.app.callback(
    Output("de-table",      "columns"),
    Output("de-pb-samples", "children"),
    Input("de-done",         "data"),
    Input("dataset-version", "data"),
)
def update_de_table_columns(_done, _ds_version):
    """Columns of the result table (they differ between the two tests) and, for a
    pseudobulk result, the per-sample cells / library sizes / size factors."""
    with analysis._de_lock:
        res = analysis._de_state.get("result")
    return analysis._de_table_columns(res), stats._pb_samples_view(res)


# ─── Composition between conditions ───────────────────────────────────────────

@app_layout.app.callback(
    Output("comp-category", "options"),
    Output("comp-category", "value"),
    Input("plots-tabs",      "active_tab"),
    Input("annot-done",      "data"),
    Input("niche-done",      "data"),
    Input("dataset-version", "data"),
    Input("seg-source",      "value"),
    Input("extra-datasets-version", "data"),
    State("color-by",        "value"),
    State("comp-category",   "value"),
)
def update_comp_category_options(active_tab, _annot, _niche, _ds, seg_source, _ex, color_by=None,
                                 current=None):
    if active_tab != "tab-abundance":
        raise dash.exceptions.PreventUpdate
    opts = stats._comp_category_options(seg_source, color_by)
    have = [o["value"] for o in opts]
    return opts, (no_update if current in have else have[0])


@app_layout.app.callback(
    Output("comp-version", "data"),
    Output("comp-status",  "children"),
    Input("comp-run-btn",  "n_clicks"),
    State("comp-category",  "value"),
    State("comp-transform", "value"),
    State("seg-source",     "value"),
    State("cluster-method", "value"),
    State("comp-version",   "data"),
    State("comp-test",      "value"),
    prevent_initial_call=True,
)
def run_composition(n_clicks, kind, transform, seg_source, method, version, test="moderated"):
    """Per-sample proportions and, with at least 3 samples per condition, the test of
    every category (a few thousand numbers: computed in the callback, no job)."""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not kind:
        return no_update, "Pick what the proportions are computed of."
    try:
        res = stats._comp_compute(kind, seg_source or "xenium", method,
                                  "asin" if transform == "asin" else "logit",
                                  "exact" if test == "exact" else "moderated")
    except ValueError as exc:
        with stats._comp_lock:
            stats._comp_state.update({"result": None, "message": str(exc)})
        return (version or 0) + 1, str(exc)
    msg = res.get("test_note", "")
    for w in res["warnings"]:
        msg += f"\n⚠ {w[0].upper() + w[1:]}."
    with stats._comp_lock:
        stats._comp_state.update({"result": res, "message": msg})
        stats._comp_state["version"] += 1
    return (version or 0) + 1, msg


@app_layout.app.callback(
    Output("comp-plot",  "figure"),
    Output("comp-table", "columns"),
    Output("comp-table", "data"),
    Input("comp-version",    "data"),
    Input("dataset-version", "data"),
    Input("comp-plot-w",     "value"),
    Input("comp-plot-h",     "value"),
)
def update_comp_results(_version, _ds_version, w, h):
    with stats._comp_lock:
        res = stats._comp_state.get("result")
    cols, rows = stats._comp_table(res)
    return stats.make_composition_fig(width=w, height=h), cols, rows


@app_layout.app.callback(
    Output("comp-download", "data"),
    Input("comp-download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_composition(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    with stats._comp_lock:
        res = stats._comp_state.get("result")
    if not res:
        raise dash.exceptions.PreventUpdate
    _safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(res["category"]))[:40]
    return dict(content=stats._comp_csv(res), filename=f"composition_{_safe}.csv")


# ─── Spatially variable genes (Moran's I) ─────────────────────────────────────

@app_layout.app.callback(
    Output("jobs-poll", "disabled", allow_duplicate=True),
    Input("moran-poll", "disabled"),
    Input("lr-poll",    "disabled"),
    State("jobs-poll",  "disabled"),
    prevent_initial_call=True,
)
def wake_jobs_poll_stats(_mo, _lr, poll_disabled):
    """As wake_jobs_poll, for the two spatial statistics jobs."""
    if jobs._jobs_panel_rows() and poll_disabled:
        return False
    return no_update


@app_layout.app.callback(
    Output("moran-run-btn", "disabled"),
    Output("moran-poll",    "disabled"),
    Output("moran-status",  "children"),
    Input("moran-run-btn",  "n_clicks"),
    State("moran-k",        "value"),
    State("seg-source",     "value"),
    State("counts-mode-store", "data"),
    prevent_initial_call=True,
)
def run_moran(n_clicks, k, seg_source, counts_mode):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not analysis._an_samples(seg_source, counts_mode == "corrected"):
        return False, True, "The active view has no expression matrix."
    if not cb_analysis._an_job_claim(stats._moran_state, stats._moran_lock):
        return True, False, "Moran's I is already being computed."
    jobs._job_start("moran", "Spatially variable genes (Moran's I)", stats._run_moran,
               args=(seg_source or "xenium", counts_mode == "corrected", k or stats._MORAN_K),
               state=stats._moran_state, lock=stats._moran_lock)
    return True, False, "Running…"


@app_layout.app.callback(
    Output("moran-status",  "children", allow_duplicate=True),
    Output("moran-poll",    "disabled", allow_duplicate=True),
    Output("moran-run-btn", "disabled", allow_duplicate=True),
    Output("moran-done",    "data"),
    Input("moran-poll",     "n_intervals"),
    State("moran-done",     "data"),
    prevent_initial_call=True,
)
def poll_moran(_, done_ver):
    with stats._moran_lock:
        status, msg = stats._moran_state["status"], stats._moran_state["message"]
    if status == "done":
        return msg, True, False, (done_ver or 0) + 1
    if status in ("error", "idle"):
        return msg, True, False, no_update
    return msg, False, True, no_update


@app_layout.app.callback(
    Output("moran-table", "data"),
    Input("moran-done",      "data"),
    Input("dataset-version", "data"),
)
def update_moran_results(_done, _ds_version):
    with stats._moran_lock:
        res = stats._moran_state.get("result")
    return stats._moran_table_records(res)


@app_layout.app.callback(
    Output("gene-selector", "value", allow_duplicate=True),
    Input("moran-table",     "active_cell"),
    State("moran-table",     "derived_viewport_data"),
    State("gene-selector",   "options"),
    prevent_initial_call=True,
)
def moran_pick_gene_from_table(active_cell, shown_rows, gene_options):
    """A click on a table row selects that gene in the sidebar."""
    row = (active_cell or {}).get("row", -1)
    if not shown_rows or not (0 <= row < len(shown_rows)):
        raise dash.exceptions.PreventUpdate
    return cb_analysis._de_pick(shown_rows[row].get("gene"), gene_options)


@app_layout.app.callback(
    Output("moran-download", "data"),
    Input("moran-download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_moran_table(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    with stats._moran_lock:
        res = stats._moran_state.get("result")
    if not res:
        raise dash.exceptions.PreventUpdate
    t = (res["table"].drop(columns=["neg_log10_padj"])
         .sort_values(["sample", "moran_i"], ascending=[True, False], kind="stable"))
    header = (f"# Moran's I per gene; kNN graph of the cell centroids, k = {res['k']}, "
              f"row-standardised weights; log1p(CP10K) of the "
              f"{'SPLIT-corrected' if res['corrected'] else 'original'} counts; segmentation: "
              f"{res['seg_source']}; z and two-sided p under the normality assumption; BH FDR per sample\n"
              f"# unit of the p-values: the cells of a sample, treated as independent; pct_cells: "
              f"count >= 1 transcript\n")
    for i in res["samples"]:
        header += (f"# sample {i['label']} ({i['name']}): {i['n_cells']} cells; E[I] = "
                   f"{i['expected']:.6g}; Var[I] = {i['variance']:.6g}\n")
    return dict(content=header + t.to_csv(index=False), filename="spatially_variable_genes_moran.csv")


# ─── Ligand-receptor scores ───────────────────────────────────────────────────

@app_layout.app.callback(
    Output("lr-labels", "options"),
    Output("lr-labels", "value"),
    Input("plots-tabs",      "active_tab"),
    Input("annot-done",      "data"),
    Input("dataset-version", "data"),
    Input("seg-source",      "value"),
    Input("extra-datasets-version", "data"),
    State("color-by",        "value"),
    State("lr-labels",       "value"),
)
def update_lr_label_options(active_tab, _annot, _ds, seg_source, _ex, color_by=None, current=None):
    if active_tab != "tab-lr":
        raise dash.exceptions.PreventUpdate
    opts = stats._lr_label_options(seg_source, color_by)
    have = [o["value"] for o in opts]
    want = f"celltype:{str(color_by).split(':', 1)[1]}" if str(color_by or "").startswith("cell_type:") else None
    return opts, (no_update if current in have else (want if want in have else have[0]))


@app_layout.app.callback(
    Output("lr-pairs",      "value"),
    Output("lr-pairs-info", "children"),
    Input("lr-cellnest-btn", "n_clicks"),
    prevent_initial_call=True,
)
def fill_lr_pairs_from_cellnest(n_clicks):
    """Put the ligand-receptor pairs of the loaded CellNEST results into the text box."""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    pairs = stats._lr_pairs_from_cellnest()
    if not pairs:
        return no_update, "No CellNEST results are loaded (CellNEST CCC tab)."
    return ("ligand,receptor\n" + "\n".join(f"{a},{b}" for a, b in pairs),
            f"{len(pairs):,} pairs from the loaded CellNEST results.")


@app_layout.app.callback(
    Output("lr-pairs",      "value",    allow_duplicate=True),
    Output("lr-pairs-info", "children", allow_duplicate=True),
    Input("lr-upload",       "contents"),
    State("lr-upload",       "filename"),
    prevent_initial_call=True,
)
def fill_lr_pairs_from_upload(contents, filename):
    """Put an uploaded pair list into the text box (which is what Run reads)."""
    if not contents:
        raise dash.exceptions.PreventUpdate
    try:
        text = base64.b64decode(contents.split(",", 1)[1]).decode("utf-8-sig", errors="replace")
    except Exception as exc:
        return no_update, f"Could not read {filename or 'the file'}: {exc}"
    if len(text) > 5_000_000:
        return no_update, f"{filename or 'The file'} is larger than 5 MB."
    pairs, skipped = stats._lr_parse_pairs(text)
    if not pairs:
        return no_update, f"{filename or 'The file'}: no 'ligand,receptor' lines found."
    return ("ligand,receptor\n" + "\n".join(f"{a},{b}" for a, b in pairs),
            f"{len(pairs):,} pairs from {filename or 'the file'}"
            + (f" ({skipped} lines skipped)." if skipped else "."))


@app_layout.app.callback(
    Output("lr-run-btn", "disabled"),
    Output("lr-poll",    "disabled"),
    Output("lr-status",  "children"),
    Input("lr-run-btn",  "n_clicks"),
    State("lr-pairs",    "value"),
    State("lr-labels",   "value"),
    State("lr-k",        "value"),
    State("lr-nperm",    "value"),
    State("seg-source",  "value"),
    State("cluster-method", "value"),
    State("counts-mode-store", "data"),
    prevent_initial_call=True,
)
def run_lr(n_clicks, pairs_text, label_kind, k, n_perm, seg_source, method, counts_mode):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    pairs, _skipped = stats._lr_parse_pairs(pairs_text)
    if not pairs:
        return False, True, ("No ligand–receptor pairs: paste or upload lines of 'ligand,receptor' "
                             "(no database is bundled or downloaded).")
    if not label_kind:
        return False, True, "Pick the cell types (or clusters) to score between."
    if not cb_analysis._an_job_claim(stats._lr_state, stats._lr_lock):
        return True, False, "Ligand–receptor scoring is already running."
    jobs._job_start("lr", "Ligand–receptor scores", stats._run_lr,
               args=(pairs_text, label_kind, seg_source or "xenium", method,
                     counts_mode == "corrected", k or stats._MORAN_K, n_perm or 200),
               state=stats._lr_state, lock=stats._lr_lock)
    return True, False, "Running…"


@app_layout.app.callback(
    Output("lr-status",  "children", allow_duplicate=True),
    Output("lr-poll",    "disabled", allow_duplicate=True),
    Output("lr-run-btn", "disabled", allow_duplicate=True),
    Output("lr-done",    "data"),
    Input("lr-poll",     "n_intervals"),
    State("lr-done",     "data"),
    prevent_initial_call=True,
)
def poll_lr(_, done_ver):
    with stats._lr_lock:
        status, msg = stats._lr_state["status"], stats._lr_state["message"]
    if status == "done":
        return msg, True, False, (done_ver or 0) + 1
    if status in ("error", "idle"):
        return msg, True, False, no_update
    return msg, False, True, no_update


@app_layout.app.callback(
    Output("lr-plot",  "figure"),
    Output("lr-table", "data"),
    Input("lr-done",         "data"),
    Input("dataset-version", "data"),
    Input("lr-top-n",        "value"),
    Input("lr-plot-w",       "value"),
    Input("lr-plot-h",       "value"),
)
def update_lr_results(_done, _ds_version, top_n, w, h):
    with stats._lr_lock:
        res = stats._lr_state.get("result")
    return stats.make_lr_dotplot(top_n=top_n, width=w, height=h), stats._lr_table_records(res)


@app_layout.app.callback(
    Output("lr-download", "data"),
    Input("lr-download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_lr_table(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    with stats._lr_lock:
        res = stats._lr_state.get("result")
    if not res:
        raise dash.exceptions.PreventUpdate
    info = res["info"]
    t = res["table"].sort_values(["padj", "pval", "score"], ascending=[True, True, False], kind="stable")
    header = (f"# Ligand-receptor scores between groups ({res['category']}); score = (mean ligand in sender + "
              f"mean receptor in receiver) / 2 on log1p(CP10K) of the "
              f"{'SPLIT-corrected' if res['corrected'] else 'original'} counts; segmentation: "
              f"{res['seg_source']}\n"
              f"# {info['n_pairs_on_panel']} of {info['n_pairs_input']} pairs have both genes on the "
              f"panel; {info['n_type_pairs_neighbouring']} of {info['n_type_pairs']} directed type pairs "
              f"are spatial neighbours (kNN k = {info['k']}, >= {stats._LR_MIN_EDGES} adjacent cell pairs)\n"
              f"# p: label permutation within sample, {info['n_perm']} permutations, seed {info['seed']}; "
              f"padj: BH over the {len(t)} tested combinations; expression-based inference, not "
              f"evidence of signalling\n"
              f"# unit of the p-values: n = {info['n_cells']} cells, exchangeable within their sample; "
              f"frac_ligand / frac_receptor: cells with count >= 1 transcript\n")
    return dict(content=header + t.to_csv(index=False), filename="ligand_receptor_scores.csv")


# ─── Counts mode with several samples ─────────────────────────────────────────

@app_layout.app.callback(
    Output("counts-mode-note", "children"),
    Input("counts-mode-store",      "data"),
    Input("extra-datasets-version", "data"),
    Input("split-done",             "data"),
    Input("dataset-version",        "data"),
)
def update_counts_mode_note(_mode, _extra, _split, _ds):
    """With several samples: which of them carry SPLIT-corrected counts, and what that
    means for the map and the tests."""
    if not dataview.EXTRA_DATASETS:
        return ""
    st = dataview._corrected_status()
    if st["all"]:
        return (f"All {len(st['labels'])} samples have SPLIT-corrected counts: 'SPLIT Corrected' "
                f"applies to every sample.")
    if not st["any"]:
        return ""
    return (f"SPLIT-corrected counts exist for {', '.join(l for l, h in zip(st['labels'], st['have']) if h)} "
            f"but not for {', '.join(st['missing'])}. With several samples corrected and original "
            f"counts are never mixed: the original counts are used for all samples, and "
            f"'SPLIT Corrected' is unavailable until every sample has them.")
