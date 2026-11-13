"""Callbacks of the Provenance tab: the list of records, the detail view, the exports."""

import dash
from dash import Input, Output, State, no_update

from . import core, dataview, provenance, app_layout


@app_layout.app.callback(
    Output("prov-table",  "data"),
    Output("prov-kind",   "options"),
    Output("prov-status", "children"),
    Input("plots-tabs",       "active_tab"),
    Input("prov-refresh-btn", "n_clicks"),
    Input("prov-kind",        "value"),
    Input("dataset-version",  "data"),
)
def update_prov_table(tab, _n, kind, _ds):
    """Reads the record file only while the tab is open (or on Refresh): cheap, and never
    a recomputation."""
    if tab != "tab-provenance":
        raise dash.exceptions.PreventUpdate
    rows = provenance._prov_rows(kind)
    n_all = len(provenance._prov_read())
    if not n_all:
        msg = ("No analysis has been recorded for this dataset yet. Every analysis run from now on "
               "adds a record here.")
    else:
        msg = (f"{n_all:,} record(s) for {dataview._ds_name(core.DATA)}"
               + (f"; {len(rows):,} of kind '{provenance._PROV_KINDS.get(kind, kind)}'" if kind else "")
               + ".")
    return rows, provenance._prov_kind_options(), msg


@app_layout.app.callback(
    Output("prov-detail", "children"),
    Input("prov-table", "active_cell"),
    State("prov-table", "derived_viewport_data"),
    prevent_initial_call=True,
)
def show_prov_detail(cell, rows):
    if not cell or not rows or cell.get("row") is None or cell["row"] >= len(rows):
        raise dash.exceptions.PreventUpdate
    return provenance._prov_detail_text(provenance._prov_get(rows[cell["row"]].get("id")))


@app_layout.app.callback(
    Output("prov-download", "data"),
    Input("prov-export-md-btn",   "n_clicks"),
    Input("prov-export-json-btn", "n_clicks"),
    prevent_initial_call=True,
)
def export_provenance(n_md, n_json):
    trig = dash.ctx.triggered_id
    if trig == "prov-export-md-btn" and n_md:
        return dict(content=provenance._prov_methods_md(), filename="methods.md")
    if trig == "prov-export-json-btn" and n_json:
        return dict(content=provenance._prov_export_json(), filename="provenance.json")
    raise dash.exceptions.PreventUpdate
