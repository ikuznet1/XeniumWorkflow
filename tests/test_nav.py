"""The two-level navigation of the plots panel (ui._nav_bar, cb_nav.py).

plots-tabs stays one dbc.Tabs: callbacks and tests keep using its active_tab. The bar on
top only groups its tabs. These tests keep the two in step: every tab is in exactly one
group, the design entry exists once, the availability notes read state only.
"""
from __future__ import annotations

import pytest


def _walk(c):
    yield c
    ch = getattr(c, "children", None)
    for k in (ch if isinstance(ch, (list, tuple)) else [ch]):
        if hasattr(k, "to_plotly_json"):
            yield from _walk(k)


def _plots_tabs(xe):
    return next(c for c in _walk(xe.app.layout) if getattr(c, "id", None) == "plots-tabs")


def test_every_tab_is_in_exactly_one_group(xe):
    tab_ids = [t.tab_id for t in _plots_tabs(xe).children]
    assert len(tab_ids) == len(set(tab_ids))
    nav = [tid for _g, _l, tabs in xe._NAV_GROUPS for tid, _ in tabs]
    assert sorted(nav) == sorted(tab_ids), "a tab without a place in the navigation cannot be opened"
    assert [g for g, _l, _t in xe._NAV_GROUPS] == ["overview", "qc", "expression", "neighbourhoods",
                                                   "communication", "provenance"]
    for tid in tab_ids:
        assert xe._nav_group_of(tid) is not None
    assert xe._nav_group_of("tab-nope") is None


def test_the_bar_has_a_button_per_group_and_tab(xe):
    ids = {getattr(c, "id", None) for c in _walk(xe.app.layout)}
    for gid, _label, tabs in xe._NAV_GROUPS:
        assert f"nav-group-{gid}" in ids
        for tid, _ in tabs:
            assert f"nav-tab-{tid}" in ids
    assert {"plots-nav", "nav-last", "nav-notice", "nav-design-open-btn", "nav-design-summary"} <= ids
    # the first view is Overview / UMAP, as before the regrouping
    assert _plots_tabs(xe).active_tab == "tab-umap"


def test_component_ids_did_not_change_when_blocks_moved(xe):
    """Composition and segmentation-free abundance left the Cell Type Abundance tab for
    tabs of their own; their ids (and those behind the Options disclosures) are the same."""
    by_tab = {}
    for t in _plots_tabs(xe).children:
        by_tab[t.tab_id] = {getattr(c, "id", None) for c in _walk(t)} - {None}
    assert {"comp-section", "comp-category", "comp-transform", "comp-test", "comp-run-btn", "comp-table",
            "comp-download-btn"} <= by_tab["tab-composition"]
    assert {"sfree-section", "sfree-labels", "sfree-type", "sfree-n-markers", "sfree-bins", "sfree-run-btn",
            "sfree-download-btn"} <= by_tab["tab-segfree"]
    assert "cell-type-abundance-plot" in by_tab["tab-abundance"]
    for tab, needed in {"tab-de": {"de-run-btn", "de-leak-gate", "de-lfc-axis", "de-pb-min-cells", "de-pb-shrink",
                                   "prox-near", "contact-radius", "de-download-btn"},
                        "tab-controls": {"ctl-comp-run-btn", "ctl-bin", "ctl-core-frac", "spill-run-btn",
                                         "spill-markers", "spill-own-frac"},
                        "tab-segqc": {"segqc-run-btn", "segqc-source", "segqc-radius", "segqc-sets"},
                        "tab-coloc": {"coloc-run-btn", "coloc-radius", "coloc-nperm", "coloc-placebo"},
                        "tab-agg": {"agg-run-btn", "agg-radius", "agg-hull", "agg-outline-toggle"},
                        "tab-txf": {"txf-run-btn", "txf-bin", "txf-seed"},
                        "tab-banksy": {"bk-run-btn", "bk-kgeom", "bk-agf", "bk-seed", "bk-backend-note"},
                        "tab-joint": {"jc-run-btn", "jc-nhvg", "jc-seed", "jc-umap"},
                        "tab-lr": {"lr-run-btn", "lrd-run-btn", "lrd-radius", "lrd-floor", "lrd-rates"}}.items():
        assert needed <= by_tab[tab], (tab, needed - by_tab[tab])


def test_options_disclosures_are_closed_and_name_the_defaults(xe):
    found = [c for c in _walk(_plots_tabs(xe)) if getattr(c, "className", None) == "tab-options"]
    assert len(found) >= 12
    for d in found:
        assert not d.open
        summary = d.children[0].children
        assert summary.startswith("Options (") and len(summary) > 20, summary


def test_tab_switch_callbacks_only_fill_option_lists(xe):
    """Nothing that listens to plots-tabs.active_tab may write a figure or a result table:
    opening a tab must never recompute an analysis."""
    heavy = []
    for key, cb in xe.app.callback_map.items():
        if any(i["id"] == "plots-tabs" for i in cb["inputs"]):
            outs = [o for o in key.strip(".").split("...")]
            for o in outs:
                prop = o.split("@")[0].rsplit(".", 1)[-1]
                cid = o.rsplit(".", 1)[0]
                if prop == "figure" or (prop == "data" and cid not in ("prov-table", "nav-last")):
                    heavy.append(o)
    assert not heavy, heavy


def test_design_entry_opens_the_editor(xe):
    out = xe.open_design_modal_nav(1)
    assert out[0] is True and out[4] == ""
    import dash
    with pytest.raises(dash.exceptions.PreventUpdate):
        xe.open_design_modal_nav(0)
    assert xe.update_nav_design_summary(0, 0, 0) == xe._design_summary()


def test_notice_says_why_a_tab_has_nothing_to_show(xe, monkeypatch):
    # one sample, the tiny bundle: composition needs several samples
    monkeypatch.setattr(xe, "EXTRA_DATASETS", [])
    msg, style = xe.update_nav_notice("tab-composition", 0, 0, 0, 0, 0)
    assert "One sample is loaded" in msg and style == {}
    # no transcripts when the dataset did not come from a Xenium output directory
    monkeypatch.setitem(xe.DATA, "data_dir", None)
    monkeypatch.setitem(xe.DATA, "sdata_path", "/somewhere/sample.zarr")
    for tab in ("tab-density", "tab-segfree", "tab-txf"):
        msg, style = xe.update_nav_notice(tab, 0, 0, 0, 0, 0)
        assert "loaded from a .zarr" in msg and "no transcripts" in msg and style == {}
    # a tab that can work says nothing and takes no room
    assert xe.update_nav_notice("tab-umap", 0, 0, 0, 0, 0) == ("", {"display": "none"})
    assert xe.update_nav_notice("tab-provenance", 0, 0, 0, 0, 0) == ("", {"display": "none"})


def test_notice_without_annotation_and_spatial_results(xe, monkeypatch):
    monkeypatch.setattr(xe, "_annot_state", {"status": "idle", "message": ""})
    monkeypatch.setattr(xe, "_spatial_state", {"status": "idle", "message": "", "result": None})
    assert "No cell-type annotation is loaded" in xe._nav_notice("tab-abundance")
    assert "No cell-type annotation is loaded" in xe._nav_notice("tab-segqc")
    msg = xe._nav_notice("tab-cooccurrence")
    assert "No spatial analysis has been run yet" in msg and "first annotate" in msg
    monkeypatch.setattr(xe, "_annot_state", {"status": "done", "message": "", "labels": object()})
    assert xe._nav_notice("tab-abundance") == ""
    assert "first annotate" not in xe._nav_notice("tab-enrichment")


def test_notice_never_breaks_the_panel(xe, monkeypatch):
    def boom(_tab):
        raise RuntimeError("x")
    monkeypatch.setattr(xe, "_nav_notice", boom)
    assert xe.update_nav_notice("tab-de", 0, 0, 0, 0, 0) == ("", {"display": "none"})


def test_jobs_panel_sits_clear_of_the_plots_panel(xe):
    st = xe._JOBS_PANEL_STYLE
    assert st["position"] == "fixed" and "left" in st and "right" not in st
