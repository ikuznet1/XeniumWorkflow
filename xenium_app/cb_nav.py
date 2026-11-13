"""Navigation of the plots panel: the two-level bar over plots-tabs (ui._nav_bar), the
shared "Samples & conditions…" entry and the note shown in place when the open tab cannot
work with the loaded data. Nothing here computes anything: a tab switch only reads state."""

import os
import json

import dash
from dash import Input, Output, State

from . import core, ui, dataview, stats, app_layout


_NAV_TABS = [tid for _g, _l, tabs in ui._NAV_GROUPS for tid, _ in tabs]
_NAV_JS_GROUPS = json.dumps({gid: [t for t, _ in tabs] for gid, _l, tabs in ui._NAV_GROUPS})

# A click on a group opens the tab last used in it (or its first); a click on a sub-tab
# opens that tab. Both only set plots-tabs.active_tab.
app_layout.app.clientside_callback(
    """function() {
        var nu = window.dash_clientside.no_update;
        var trig = (window.dash_clientside.callback_context.triggered || [])[0];
        if (!trig || !trig.value) return nu;
        var id = trig.prop_id.split('.')[0];
        var groups = %s;
        var last = arguments[arguments.length - 1] || {};
        if (id.indexOf('nav-tab-') === 0) return id.slice(8);
        var g = id.slice('nav-group-'.length);
        if (!groups[g]) return nu;
        return (last[g] && groups[g].indexOf(last[g]) >= 0) ? last[g] : groups[g][0];
    }""" % _NAV_JS_GROUPS,
    Output("plots-tabs", "active_tab"),
    [Input(f"nav-group-{gid}", "n_clicks") for gid, _l, _t in ui._NAV_GROUPS]
    + [Input(f"nav-tab-{tid}", "n_clicks") for tid in _NAV_TABS],
    State("nav-last", "data"),
    prevent_initial_call=True,
)

# plots-tabs.active_tab (set by the bar, by another callback or by a test) -> the bar:
# highlight the group and the tab, show the tabs of that group, remember the tab for its
# group, show the design entry on the tabs that use the design. A window resize event
# follows, so a figure that was laid out while its tab was hidden takes its real size.
app_layout.app.clientside_callback(
    """function(tab, last) {
        var groups = %s, designTabs = %s;
        var order = Object.keys(groups), cur = null;
        order.forEach(function(g) { if (groups[g].indexOf(tab) >= 0) cur = g; });
        var out = [];
        order.forEach(function(g) { out.push('nav-group' + (g === cur ? ' active' : '')); });
        order.forEach(function(g) {
            groups[g].forEach(function(t) {
                out.push('nav-sub' + (g === cur ? '' : ' nav-hidden') + (t === tab ? ' active' : ''));
            });
        });
        var mem = Object.assign({}, last || {});
        if (cur) mem[cur] = tab;
        out.push(mem);
        out.push('nav-design-bar' + (designTabs.indexOf(tab) >= 0 ? '' : ' nav-hidden'));
        setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 60);
        setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 400);
        return out;
    }""" % (_NAV_JS_GROUPS, json.dumps(list(ui._NAV_DESIGN_TABS))),
    [Output(f"nav-group-{gid}", "className") for gid, _l, _t in ui._NAV_GROUPS]
    + [Output(f"nav-tab-{tid}", "className") for tid in _NAV_TABS]
    + [Output("nav-last", "data"), Output("nav-design-bar", "className")],
    Input("plots-tabs", "active_tab"),
    State("nav-last", "data"),
)


# ─── the shared design entry ──────────────────────────────────────────────────
@app_layout.app.callback(
    Output("nav-design-summary", "children"),
    Input("design-version",         "data"),
    Input("extra-datasets-version", "data"),
    Input("dataset-version",        "data"),
)
def update_nav_design_summary(_design, _extra, _ds):
    return stats._design_summary()


@app_layout.app.callback(
    Output("design-modal",  "is_open",  allow_duplicate=True),
    Output("design-rows",   "children", allow_duplicate=True),
    Output("design-name-a", "value",    allow_duplicate=True),
    Output("design-name-b", "value",    allow_duplicate=True),
    Output("design-status", "children", allow_duplicate=True),
    Input("nav-design-open-btn", "n_clicks"),
    prevent_initial_call=True,
)
def open_design_modal_nav(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    names = stats._design_names()
    return True, stats._design_rows_children(), names["A"], names["B"], ""


# ─── "this tab cannot work with the loaded data", said in place ───────────────
def _nav_transcript_samples() -> tuple:
    """(labels of the loaded samples that have a transcript file, labels of those that do
    not). The transcript tabs run on every sample that has one and skip the others."""
    have, lack = [], []
    dsets = [core.DATA] + list(dataview.EXTRA_DATASETS)
    for i, ds in enumerate(dsets):
        d = ds.get("data_dir")
        ok = bool(d) and any(os.path.exists(os.path.join(str(d), f))
                             for f in ("transcripts.parquet", "transcripts.zarr.zip"))
        (have if ok else lack).append(f"S{i}" if len(dsets) > 1 else "the dataset")
    return have, lack


def _nav_has_transcripts() -> bool:
    """A sample of the view has a transcript file (not only the primary one)."""
    return bool(_nav_transcript_samples()[0])


def _nav_has_niches(seg_source="xenium") -> bool:
    """The condition the Niche UMAP figure draws on: a 'niche' column in a cell table of
    the active view (computed in this session or restored from a store) - not the state of
    the last niche job."""
    try:
        return any("niche" in t["df"].columns for t in stats._view_tables(seg_source or "xenium"))
    except Exception:
        return "niche" in core.DATA["df"].columns


def _nav_has_annotation() -> bool:
    with core._annot_lock:
        return any(str(k).startswith("labels") and v is not None for k, v in core._annot_state.items())


def _nav_notice(tab: str, seg_source="xenium") -> str:
    """Why the open tab has nothing to show for the loaded data ('' when it can work).
    Reads flags only; never loads or computes."""
    msgs = []
    n_samples = 1 + len(dataview.EXTRA_DATASETS)
    if tab in ("tab-density", "tab-segfree", "tab-txf") and not _nav_has_transcripts():
        msgs.append("This tab reads the transcript file (transcripts.parquet or transcripts.zarr.zip) of a "
                    "Xenium output directory; this dataset was " + dataview._origin_phrase() +
                    ", so there are no transcripts to read. Open the Xenium output directory to use it.")
    elif tab in ("tab-density", "tab-segfree", "tab-txf") and _nav_transcript_samples()[1]:
        _have, _lack = _nav_transcript_samples()
        msgs.append(f"This tab reads the transcript file of each sample: {', '.join(_lack)} "
                    f"{'was' if len(_lack) == 1 else 'were'} loaded without one and will be left out; "
                    f"it runs on {', '.join(_have)}.")
    if tab == "tab-composition" and n_samples < 2:       # (the Joint clusters tab says so itself)
        msgs.append("One sample is loaded. A composition test between conditions needs several samples: add "
                    f"them with '+ Add sample' in the sidebar (at least {stats._COMP_MIN_SAMPLES} per condition "
                    "for a test; the proportions of one sample are shown without a test).")
    if tab in ("tab-abundance", "tab-segqc") and not _nav_has_annotation():
        msgs.append("No cell-type annotation is loaded: run 'Annotate Cells' in the sidebar "
                    "(CellTypist needs no reference) and this tab fills in.")
    if tab in ("tab-cooccurrence", "tab-enrichment", "tab-interaction", "tab-chord"):
        with core._spatial_lock:
            done = core._spatial_state.get("result") is not None
        if not done:
            msgs.append("No spatial analysis has been run yet: "
                        + ("first annotate the cell types, then " if not _nav_has_annotation() else "")
                        + "open 'Spatial Analysis Tools' in the sidebar and run the spatial analysis; the plot appears here.")
    if tab == "tab-niche-umap":
        if not _nav_has_niches(seg_source):
            msgs.append("No niches have been computed yet: open 'Spatial Analysis Tools' in the sidebar and run the niches "
                        "(it needs a cell-type annotation or clusters).")
    if tab == "tab-cellnest":
        with core._cellnest_lock:
            done = core._cellnest_state.get("data") is not None
        if not done:
            msgs.append("No CellNEST results are loaded: CellNEST is run outside the app; load its output "
                        "folder with 'Load CellNEST Results' in the sidebar.")
    return " ".join(msgs)


@app_layout.app.callback(
    Output("nav-notice", "children"),
    Output("nav-notice", "style"),
    Input("plots-tabs",             "active_tab"),
    Input("dataset-version",        "data"),
    Input("extra-datasets-version", "data"),
    Input("annot-done",             "data"),
    Input("spatial-done",           "data"),
    Input("niche-done",             "data"),
    Input("cellnest-version",       "data"),
    Input("seg-source",             "value"),
    Input("baysor-done",            "data"),
    Input("proseg-done",            "data"),
)
def update_nav_notice(tab, _ds, _extra, _annot, _spatial, _niche, _cellnest=None, seg_source=None,
                      _baysor=None, _proseg=None):
    try:
        msg = _nav_notice(tab or "", seg_source or "xenium")
    except Exception as exc:                 # a note must never break the panel
        print(f"  nav notice: {exc}", flush=True)
        msg = ""
    return msg, ({} if msg else {"display": "none"})


# ─── room for the cell-type abundance figure ──────────────────────────────────
_NAV_ABUNDANCE_ROW_PX = 150


def _nav_abundance_rows() -> int:
    """Rows of bars of figures.make_cell_type_abundance_fig: one per annotation in memory
    (a generic key such as labels_rctd is not drawn beside its mode-specific copy)."""
    with core._annot_lock:
        keys = [k for k, v in core._annot_state.items() if str(k).startswith("labels_") and v is not None]
    generic = set()
    for k in keys:
        parts = k[len("labels_"):].split("_")
        if len(parts) >= 2 and parts[1] in ("full", "doublet", "multi"):
            generic.add(f"labels_{parts[0]}")
    return len([k for k in keys if k not in generic])


@app_layout.app.callback(
    Output("abundance-plot-sizer", "style"),
    Input("annot-done",      "data"),
    Input("seg-source",      "value"),
    Input("dataset-version", "data"),
)
def update_abundance_sizer(_annot, _seg, _ds):
    return {"display": "flex", "flexDirection": "column", "flex": "1 0 auto",
            "minHeight": f"{max(240, _NAV_ABUNDANCE_ROW_PX * _nav_abundance_rows())}px"}
