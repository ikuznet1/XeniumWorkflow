"""Callbacks: the pickers of the proximity and contact-group modes of the Differential
Expression tab (the Run button, the poll and the results are those of the tab: cb_analysis)."""

import dash
from dash import no_update
from dash import Input, Output, State

from . import stats, controls, app_layout


def _prox_type_options(kind, seg_source, method) -> list:
    """Cell types (with their cell counts) of the label source *kind* in the active view."""
    if not kind:
        return []
    try:
        tables = stats._view_tables(seg_source)
        labels, _cat = stats._comp_labels(kind, tables, seg_source, method)
    except ValueError:
        return []
    vc = controls._pooled_types(labels)
    return [{"label": f"{t} ({int(n):,} cells)", "value": str(t)} for t, n in vc.items()]


@app_layout.app.callback(
    Output("prox-labels", "options"),
    Output("prox-labels", "value"),
    Input("plots-tabs",      "active_tab"),
    Input("de-mode",         "value"),
    Input("annot-done",      "data"),
    Input("dataset-version", "data"),
    Input("seg-source",      "value"),
    Input("extra-datasets-version", "data"),
    State("color-by",        "value"),
    State("prox-labels",     "value"),
)
def update_prox_label_options(active_tab, mode, _annot, _ds, seg_source, _ex, color_by=None, current=None):
    """Label sources of the proximity / contact modes (built while one of them is shown)."""
    if active_tab != "tab-de" or mode not in ("proximity", "contact"):
        raise dash.exceptions.PreventUpdate
    opts = [o for o in stats._comp_category_options(seg_source, color_by) if o["value"] != "niche"]
    have = [o["value"] for o in opts]
    want = (f"celltype:{str(color_by).split(':', 1)[1]}" if str(color_by or "").startswith("cell_type:") else None)
    first = want if want in have else next((v for v in have if v.startswith("celltype:")), have[0] if have else None)
    return opts, (no_update if current in have else first)


@app_layout.app.callback(
    Output("prox-target", "options"),
    Output("prox-anchor", "options"),
    Output("prox-source", "options"),
    Output("prox-target", "value"),
    Output("prox-anchor", "value"),
    Output("prox-source", "value"),
    Input("prox-labels",     "value"),
    Input("dataset-version", "data"),
    Input("annot-done",      "data"),
    Input("subset-version",  "data"),
    Input("seg-source",      "value"),
    Input("cluster-method",  "value"),
    Input("extra-datasets-version", "data"),
    State("prox-target",     "value"),
    State("prox-anchor",     "value"),
    State("prox-source",     "value"),
)
def update_prox_type_options(kind, _ds, _annot, _sub, seg_source, method, _ex=None,
                             target=None, anchor=None, source=None):
    """The types of the label source in the view as it is now: another segmentation,
    cluster set or sample list has other types and counts, so the lists are rebuilt for
    each, and a picked type that no longer exists is cleared (the run would otherwise test
    a type of the same name in another clustering, or fail on a name that is not there)."""
    opts = _prox_type_options(kind, seg_source, method)
    have = {o["value"] for o in opts}
    return (opts, opts, ([{"label": "auto (type with the most transcripts, not T or X)", "value": "auto"},
                          {"label": "none", "value": "none"}] + opts),
            no_update if (target is None or target in have) else None,
            no_update if (anchor is None or anchor in have) else None,
            no_update if (source in (None, "auto", "none") or source in have) else "auto")


@app_layout.app.callback(
    Output("prox-design-summary", "children"),
    Input("design-version",         "data"),
    Input("extra-datasets-version", "data"),
    Input("dataset-version",        "data"),
)
def update_prox_design_summary(_design, _extra, _ds):
    return stats._design_summary()
