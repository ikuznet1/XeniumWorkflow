"""Aggregates: detection against within-stratum permutations, components, tissue area,
outlines, composition O/E, the job (single / several samples, cancel), Color By, outlines
of both renderers and the export column. All on fabricated cells with a planted aggregate.
"""
import os
import threading

import dash
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist


@pytest.fixture(autouse=True)
def _clean_aggregate_state(xe):
    yield
    with xe._agg_lock:
        xe._agg_state.update({"status": "idle", "message": "", "result": None, "show": True})
    xe._controls_reset()
    xe._stats_reset()
    for f in os.listdir(xe.CACHE_BASE_DIR):
        if f.startswith("design_") and f.endswith(".json"):
            os.remove(os.path.join(xe.CACHE_BASE_DIR, f))


def _world(seed=0, n=1500, size=500.0, planted=True, n_plant=45, centre=(250.0, 250.0)):
    """One section: *n* cells uniform on a square, types Imm (8 %), Str, Epi at random. With
    *planted*: *n_plant* further Imm cells and 15 "Fol" cells within 28 µm of *centre*."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, size, size=(n, 2))
    typ = rng.choice(["Imm", "Str", "Epi"], n, p=[0.08, 0.52, 0.40]).astype(object)
    is_planted = np.zeros(n, dtype=bool)
    if planted:
        k = n_plant + 15
        r, a = 28.0 * np.sqrt(rng.uniform(0, 1, k)), rng.uniform(0, 2 * np.pi, k)
        xy = np.vstack([xy, np.column_stack([centre[0] + r * np.cos(a), centre[1] + r * np.sin(a)])])
        typ = np.concatenate([typ, np.array(["Imm"] * n_plant + ["Fol"] * 15, dtype=object)])
        is_planted = np.concatenate([is_planted, np.ones(k, dtype=bool)])
    return xy, typ, is_planted


def _levels(xe, xy):
    cols, _ = xe._compartments(xy)
    comp = np.unique(cols["compartment"].astype(str), return_inverse=True)[1]
    return xe._coloc_levels(np.zeros(len(xy), dtype=int), comp, cols["edge_distance_um"], cols["local_density"])[:3]


def _fake_view(xe, monkeypatch, n_samples, planted=None, n=1200, primary_key=None):
    tables, labs = [], []
    for i in range(n_samples):
        pl = True if planted is None else planted[i]
        xy, typ, _ = _world(seed=30 + i, n=n, planted=pl)
        df = pd.DataFrame({"x_centroid": xy[:, 0], "y_centroid": xy[:, 1]}, index=[f"a{i}_{k}" for k in range(len(xy))])
        tables.append({"label": f"S{i}" if n_samples > 1 else "xenium", "df": df, "ds": {},
                       "key": primary_key if (primary_key and i == 0) else f"/fake/agg{i}",
                       "name": f"agg{i}", "x_offset": 1000.0 * i})
        labs.append(typ)
    monkeypatch.setattr(xe, "_view_tables", lambda seg_source="xenium": tables)
    monkeypatch.setattr(xe, "_comp_labels", lambda kind, t, seg, method: (labs, "cell type (test)"))
    return tables, labs


FAST = {"n_null": 99, "n_control": 3}


# ── detection ────────────────────────────────────────────────────────────────
def test_local_class_count_equals_a_hand_count_and_permutations_keep_strata(xe):
    xy, typ, _ = _world(seed=1, n=400, planted=True)
    ind = typ == "Imm"
    strata = (xy[:, 0] > 250).astype(int)
    det = xe._agg_detect(xy, ind, strata, radius=40.0, n_null=40, n_control=2, seed=3)
    hand = ((cdist(xy, xy) <= 40.0) & ind[None, :]).sum(axis=1)          # the cell itself included
    assert (det["I"] == hand).all()
    assert det["k"].min() >= 0 and det["k"].max() <= 40 and det["n_null"] == 40
    assert np.allclose(det["p"], (det["k"] + 1) / 41)
    assert det["ind_nc"].shape == (2, len(xy)) and det["I_nc"].shape == (2, len(xy))
    yp = xe._agg_perm_within(strata, ind.astype(np.float32), np.random.default_rng(0), 5)
    for b in range(5):
        assert (np.bincount(strata, weights=yp[:, b]) == np.bincount(strata, weights=ind)).all()
        assert not (yp[:, b] == ind).all()
    # the same seed gives the same draws
    again = xe._agg_detect(xy, ind, strata, radius=40.0, n_null=40, n_control=2, seed=3)
    assert (again["k"] == det["k"]).all()
    # hotspot rule: p <= alpha and at least min_nb class cells
    hot = xe._agg_hot(np.array([5, 5, 2]), np.array([0, 3, 0]), 199, alpha=0.01, min_nb=3)
    assert hot.tolist() == [True, False, False]


def test_planted_aggregate_is_found_and_a_uniform_field_yields_none(xe):
    xy, typ, planted = _world(seed=2)
    types = sorted(set(typ))
    code = np.array([types.index(t) for t in typ])
    r = xe._agg_sample(xy, typ == "Imm", _levels(xe, xy), code, len(types), n_null=199, n_control=5)
    assert len(r["table"]) == 1 and r["lab"].max() == 0
    mem = r["lab"] == 0
    assert mem[planted].mean() > 0.75                                 # the planted cells are members
    assert planted[mem].mean() > 0.5                                  # and make up most of it
    row = r["table"].iloc[0]
    assert row["n_cells"] == mem.sum() and row["n_class"] == ((typ == "Imm") & mem).sum() >= 40
    assert abs(row["cx"] - 250) < 10 and abs(row["cy"] - 250) < 10
    assert 0.5 * np.pi * 28 ** 2 < row["area_um2"] < 4 * np.pi * 28 ** 2
    assert row["area_um2"] == pytest.approx(r["tile"][mem].sum())
    assert max(r["control"]["n"]) == 0                               # permuted labels: no aggregate
    # composition: the planted co-located type is enriched against matched outside cells
    l2 = np.log2((r["O"][0] + 0.5) / (r["E"][0] + 0.5))
    assert l2[types.index("Fol")] > 2 and l2[types.index("Imm")] > 1 and l2[types.index("Str")] < 0
    assert r["O"][0].sum() == mem.sum() and r["E"][0].sum() == pytest.approx(mem.sum())
    assert not r["outside"][mem].any()
    # uniform field: nothing
    xy0, typ0, _ = _world(seed=2, planted=False)
    r0 = xe._agg_sample(xy0, typ0 == "Imm", _levels(xe, xy0), n_null=199, n_control=5)
    assert len(r0["table"]) == 0 and (r0["lab"] == -1).all() and r0["rings"] == []
    # a class of fewer than N cells cannot form one
    r1 = xe._agg_sample(xy, typ == "Imm", _levels(xe, xy), n_null=99, n_control=0, n_min=500)
    assert len(r1["table"]) == 0


def test_components_need_short_links_and_enough_class_cells(xe):
    # two groups of hotspot cells 100 µm apart, chain spacing 10 µm; a third with 2 class cells
    xs = np.concatenate([np.arange(0, 60, 10), np.arange(160, 220, 10), np.arange(400, 430, 10)])
    xy = np.column_stack([xs, np.where(np.arange(len(xs)) % 2, 3.0, 0.0)])
    ind = np.ones(len(xs), dtype=bool)
    ind[12:] = [True, True, False]
    hot = np.ones(len(xs), dtype=bool)
    u, v = xe._agg_edges(xy, 30.0)
    assert len(u) and np.hypot(*(xy[u] - xy[v]).T).max() <= 30.0
    lab = xe._agg_components(hot, ind, (u, v), n_min=5)
    assert lab[:6].tolist() == [0] * 6 and lab[6:12].tolist() == [1] * 6 and (lab[12:] == -1).all()
    # a non-hotspot cell breaks the chain
    hot2 = hot.copy()
    hot2[[2, 3]] = False
    assert (xe._agg_components(hot2, ind, (u, v), n_min=5)[:6] == -1).all()
    assert (xe._agg_components(np.zeros(len(xs), dtype=bool), ind, (u, v)) == -1).all()


def test_tile_area_on_a_lattice_and_hulls(xe):
    gx, gy = np.meshgrid(np.arange(0, 100, 10.0), np.arange(0, 100, 10.0))
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    area = xe._agg_tile_area(xy)
    interior = (xy[:, 0] > 0) & (xy[:, 0] < 90) & (xy[:, 1] > 0) & (xy[:, 1] < 90)
    assert np.allclose(area[interior], 100.0)                       # 4 grid points of 25 µm² each
    assert (area[~interior] > 100.0).all()                          # border cells reach 20 µm outwards
    # convex hull holds every point; an L shape: the alpha hull is smaller than the convex one
    from matplotlib.path import Path
    rng = np.random.default_rng(0)
    pts = np.vstack([rng.uniform([0, 0], [100, 20], (150, 2)), rng.uniform([0, 0], [20, 100], (150, 2))])
    conv = xe._agg_hull(pts, "convex")
    assert len(conv) == 1 and Path(np.array(conv[0])).contains_points(pts, radius=1e-6).mean() > 0.9

    def _area(r):
        r = np.array(r)
        return 0.5 * abs(np.dot(r[:, 0], np.roll(r[:, 1], 1)) - np.dot(r[:, 1], np.roll(r[:, 0], 1)))
    alpha = xe._agg_hull(pts, "alpha", 25.0)
    assert len(alpha) >= 1 and _area(alpha[0]) < 0.75 * _area(conv[0])
    assert _area(alpha[0]) == pytest.approx(3600, rel=0.25)         # the L itself
    assert len(xe._agg_hull(pts[:2], "convex")[0]) == 4 and xe._agg_hull(np.zeros((0, 2))) == []
    line = np.column_stack([np.arange(5.0), np.arange(5.0)])
    assert len(xe._agg_hull(line, "alpha")[0]) == 4


def test_composition_expectation_is_the_matched_outside_distribution(xe):
    # stratum 0: outside cells 50 % type 0; stratum 1: 10 %. Aggregate 0 lies in stratum 0.
    code = np.array([0] * 10 + [1] * 10 + [0] * 2 + [1] * 18 + [0, 0, 0, 1])
    strat = np.array([0] * 20 + [1] * 20 + [0] * 4)
    lab = np.array([-1] * 40 + [0] * 4)
    O, E = xe._agg_oe(lab, code, [strat], 2, lab < 0, min_pool=5)
    assert O.tolist() == [[3.0, 1.0]] and E[0] == pytest.approx([2.0, 2.0])
    # too few outside cells in the stratum: the whole section is the reference
    O, E = xe._agg_oe(lab, code, [strat], 2, lab < 0, min_pool=30)
    assert E[0] == pytest.approx([4 * 12 / 40, 4 * 28 / 40])
    # unlabelled members are not counted
    code2 = code.copy()
    code2[-1] = -1
    O, _ = xe._agg_oe(lab, code2, [strat], 2, lab < 0, min_pool=5)
    assert O.sum() == 3


# ── the job ──────────────────────────────────────────────────────────────────
def test_job_single_sample_color_outlines_and_export(xe, monkeypatch):
    tables, labs = _fake_view(xe, monkeypatch, 1, primary_key=xe._ds_key(xe.DATA))
    xe._run_aggregates("xenium", None, "celltype:celltypist", ["Imm", "Nope"], FAST)
    st = xe._agg_state
    assert st["status"] == "done", st["message"]
    res = xe._agg_result("xenium")
    assert res is not None and len(res["aggregates"]) == 1 and res["classes"] == ["Imm"]
    msg = st["message"]
    assert "SINGLE-SAMPLE RESULT" in msg and "Unit of the hotspot p-values: the cells of one section" in msg
    assert "smallest possible p 0.01" in msg and "Not in this view: Nope" in msg
    assert "Compartments were not computed" in msg
    a = res["aggregates"].iloc[0]
    assert a["id"] == 1 and a["sample"] == "xenium" and a["n_class"] >= 40 and "Imm" in a["composition"]
    assert "Fol" in a["enriched_other"]
    s = res["samples"].iloc[0]
    assert s["n_aggregates"] == 1 and s["per_mm2"] == pytest.approx(1 / s["tissue_mm2"])
    assert 0 < s["area_fraction"] < 0.1 and s["control_mean"] == 0 and s["tissue_mm2"] == pytest.approx(0.25, rel=0.2)
    # Color By: the id per cell, NaN outside; offered as an option; both renderers get it
    vals, rng_, label = xe._agg_color("agg#id", "xenium")
    assert np.nansum(vals == 1) == a["n_cells"] and np.isnan(vals).sum() == len(vals) - a["n_cells"]
    assert xe._agg_color_options()[0]["value"] == "geneset:agg#id"
    assert any(o["value"] == "geneset:agg#id" for o in xe._view_color_extra_options())
    got = xe._geneset_color("geneset:agg#id", "xenium", False)
    assert got is not None and got[2] == label and xe._view_color_seq[label] == rng_
    # outlines: one Plotly path (y negated), one WebGL polygon (y down), around the planted centre
    shp = xe._spatial_overlay_shapes(False, False, "xenium")
    mine = [x for x in shp if x["line"]["color"] == xe._AGG_OUTLINE_COLOR]
    assert len(mine) == 1 and mine[0]["path"].startswith("M ") and mine[0]["path"].endswith("Z")
    first = [float(v) for v in mine[0]["path"][2:].split(" L ")[0].split(",")]
    assert abs(first[0] - 250) < 60 and abs(first[1] + 250) < 60
    polys = xe._agg_viewer_polygons("xenium")
    assert len(polys) == 1 and polys[0]["label"] == "A1" and abs(polys[0]["points"][0][1] - 250) < 60
    ov = xe._viewer_provider.overlays("xenium")
    assert any(p.get("label") == "A1" for p in ov["polygons"])
    # another segmentation: neither outlines nor colours of this result
    real_tag = xe._seg_tag
    monkeypatch.setattr(xe, "_seg_tag", lambda seg: "baysor:/run" if str(seg).startswith("baysor") else real_tag(seg))
    assert xe._agg_plotly_shapes("baysor:abc") == [] and xe._agg_color("agg#id", "baysor:abc") is None
    assert len(xe._agg_plotly_shapes("xenium")) == 1
    monkeypatch.setattr(xe, "_seg_tag", real_tag)
    # a subset of the cells: the stored ids follow the cells by id
    full = tables[0]["df"]
    tables[0]["df"] = full.iloc[::2]
    sub = xe._agg_color("agg#id", "xenium")[0]
    assert len(sub) == len(full.iloc[::2]) and np.array_equal(sub, vals[::2], equal_nan=True)
    tables[0]["df"] = full
    # hidden with the toggle
    with xe._agg_lock:
        xe._agg_state["show"] = False
    assert xe._agg_viewer_polygons("xenium") == [] and xe._agg_plotly_shapes("xenium") == []
    # export column of Save as SpatialData (through the controls columns of the primary sample)
    out, names = xe._with_extra_obs(tables[0]["df"], "xenium")
    assert {"ctl_aggregate_id", "ctl_aggregate_hotspot", "ctl_aggregate_p"} <= set(names)
    assert (out["ctl_aggregate_id"] == 1).sum() == a["n_cells"]
    # figure, tables, CSV
    assert len(xe.make_agg_fig().data) == 3
    assert xe._agg_records(res)[0]["id"] == 1 and xe._agg_sample_records(res)[0]["n_aggregates"] == 1
    csv = xe._agg_csv(res)
    assert "# aggregates" in csv and "# composition per aggregate" in csv and "unit = cells of one section" in csv
    # a result of another dataset generation is not shown
    monkeypatch.setattr(xe, "_dataset_gen", xe._dataset_gen + 1)
    assert xe._agg_result() is None and xe._agg_color_options() == []
    assert xe.make_agg_fig().layout.annotations[0].text.startswith("No aggregates")


def test_job_errors_are_clear(xe, monkeypatch):
    _fake_view(xe, monkeypatch, 1, planted=[False])
    xe._run_aggregates("xenium", None, "celltype:celltypist", [], FAST)
    assert xe._agg_state["status"] == "error" and "Pick the cell types" in xe._agg_state["message"]
    xe._run_aggregates("xenium", None, "celltype:celltypist", ["Nope"], FAST)
    assert "No cell of the view is labelled Nope" in xe._agg_state["message"]
    xe._run_aggregates("xenium", None, "celltype:celltypist", ["Imm"], {"n_null": 50, "alpha": 0.001})
    assert "no cell can reach" in xe._agg_state["message"]
    # uniform field: done, nothing found, empty states
    xe._run_aggregates("xenium", None, "celltype:celltypist", ["Imm"], FAST)
    assert xe._agg_state["status"] == "done" and "0 aggregate(s)" in xe._agg_state["message"]
    assert xe._agg_color("agg#id", "xenium") is None and xe._agg_plotly_shapes("xenium") == []
    assert "No aggregate found" in xe.make_agg_fig().layout.annotations[0].text
    monkeypatch.setattr(xe, "_comp_labels", lambda kind, t, seg, method: ([None], "cell type (test)"))
    xe._run_aggregates("xenium", None, "celltype:celltypist", ["Imm"], FAST)
    assert "has no cell type (test) labels" in xe._agg_state["message"]


def test_job_several_samples_tests_density_between_conditions(xe, monkeypatch):
    planted = [True, True, True, False, False, False]
    tables, labs = _fake_view(xe, monkeypatch, 6, planted=planted, n=900)
    xe._design_set([{"key": t["key"], "cond": "A" if i < 3 else "B", "batch": "", "donor": ""}
                    for i, t in enumerate(tables)])
    xe._run_aggregates("xenium", None, "celltype:celltypist", ["Imm"], FAST)
    st = xe._agg_state
    assert st["status"] == "done", st["message"]
    res = xe._agg_result()
    s = res["samples"]
    assert s["n_aggregates"].tolist() == [1, 1, 1, 0, 0, 0] and s["cond"].tolist() == list("AAABBB")
    assert res["aggregates"]["id"].tolist() == [1, 2, 3] and set(res["aggregates"]["sample"]) == {"S0", "S1", "S2"}
    assert res["aggregates"]["n_class"].is_monotonic_decreasing
    tt = res["tests"]["per_mm2"]
    assert tt["p"] == pytest.approx(2 / 20) and tt["exact"] and tt["min_p"] == pytest.approx(0.1)
    assert tt["effect"] > 0 and "n = 6 samples" in tt["units_note"]
    assert "Between conditions (aggregates per mm²)" in st["message"] and "unit of the p-value: n = 6 samples" in st["message"]
    assert "SINGLE-SAMPLE" not in st["message"]
    # outlines carry the x offset of their sample
    polys = xe._agg_viewer_polygons("xenium")
    xs = sorted(round(np.mean([p[0] for p in poly["points"]]), -3) for poly in polys)
    assert xs == [0.0, 1000.0, 2000.0]
    vals, _r, _l = xe._agg_color("agg#id", "xenium")
    assert len(vals) == sum(len(t["df"]) for t in tables) and set(np.unique(vals[np.isfinite(vals)])) == {1, 2, 3}
    # too few units per condition: no test, and the reason is given
    xe._design_set([{"key": t["key"], "cond": "A" if i < 2 else ("B" if i < 4 else ""), "batch": "", "donor": ""}
                    for i, t in enumerate(tables)])
    xe._run_aggregates("xenium", None, "celltype:celltypist", ["Imm"], FAST)
    assert xe._agg_result()["tests"]["per_mm2"]["p"] is None
    assert "no test - fewer than 3 independent units per condition" in xe._agg_state["message"]


def test_neighbour_searches_stay_within_one_sample(xe, monkeypatch):
    """Two sections with the same coordinates: the planted Imm cells of the second lie on
    top of the first, which has none. The first must stay empty, whatever the second holds."""
    tables, labs = _fake_view(xe, monkeypatch, 2, planted=[False, True])
    for t in tables:
        t["x_offset"] = 0.0
    xe._run_aggregates("xenium", None, "celltype:celltypist", ["Imm"], FAST)
    res = xe._agg_result()
    assert res["samples"]["n_aggregates"].tolist() == [0, 1]
    alone = xe._ctl_get(tables[0], "xenium", "aggregate_p").copy()
    _fake_view(xe, monkeypatch, 1, planted=[False])
    monkeypatch.setattr(xe, "_view_tables", lambda seg_source="xenium": tables[:1])
    monkeypatch.setattr(xe, "_comp_labels", lambda kind, t, seg, method: (labs[:1], "cell type (test)"))
    xe._run_aggregates("xenium", None, "celltype:celltypist", ["Imm"], FAST)
    assert np.array_equal(xe._ctl_get(tables[0], "xenium", "aggregate_p"), alone)


def test_job_can_be_cancelled_and_rerun(xe, monkeypatch):
    _fake_view(xe, monkeypatch, 1)
    entered, release = threading.Event(), threading.Event()
    real = xe._agg_perm_within
    calls = {"n": 0}

    def slow(strata, y, rng, n_rep):
        calls["n"] += 1
        if calls["n"] == 2:                    # mid-run: the first batch of null draws
            entered.set()
            release.wait(30)
        return real(strata, y, rng, n_rep)

    monkeypatch.setattr(xe, "_agg_perm_within", slow)
    out = xe.run_aggregates(1, "celltype:celltypist", ["Imm"], 50, 10, 30, 99, 0.01, 3, "convex", "xenium", None)
    assert out == (True, False, "Running…")
    assert entered.wait(30)
    assert xe.run_aggregates(1, "celltype:celltypist", ["Imm"], 50, 10, 30, 99, 0.01, 3, "convex",
                             "xenium", None)[2] == "Aggregates are already being computed."
    job = [j for j in xe._jobs.values() if j.kind == "aggregates"][-1]
    assert job.name == "aggregates" and xe._job_cancel(job.id)
    release.set()
    job.thread.join(30)
    assert job.state == "cancelled" and xe._agg_state["status"] == "error"
    assert xe._agg_state["message"] == "Cancelled" and xe._agg_state["result"] is None
    assert xe._agg_result() is None and xe._agg_plotly_shapes("xenium") == []
    assert xe._ctl_get(xe._view_tables()[0], "xenium", "aggregate_id") is None     # no partial columns
    # the poll reports the end once and releases the button
    msg, disabled, ver, poll_off, gs = xe.poll_agg(1, 0, 4)
    assert msg == "Cancelled" and disabled is False and poll_off is True and gs == 5
    assert xe.poll_agg(2, ver, 5)[4] is dash.no_update
    # rerun completes
    monkeypatch.setattr(xe, "_agg_perm_within", real)
    xe.run_aggregates(2, "celltype:celltypist", ["Imm"], 50, 10, 30, 99, 0.01, 3, "alpha", "xenium", None)
    job2 = [j for j in xe._jobs.values() if j.kind == "aggregates"][-1]
    job2.thread.join(60)
    assert job2.state == "done" and xe._agg_state["status"] == "done", xe._agg_state["message"]
    assert len(xe._agg_result()["aggregates"]) == 1 and xe._agg_result()["params"]["hull"] == "alpha"
    fig, rows, srows = xe.update_agg_results(1, 0, None, None)
    assert len(rows) == 1 and len(srows) == 1


def test_callbacks_ignore_zero_clicks_and_validate_inputs(xe):
    with pytest.raises(dash.exceptions.PreventUpdate):
        xe.run_aggregates(0, "celltype:celltypist", ["Imm"], 50, 10, 30, 99, 0.01, 3, "convex", "xenium", None)
    with pytest.raises(dash.exceptions.PreventUpdate):
        xe.run_aggregates(None, "celltype:celltypist", ["Imm"], 50, 10, 30, 99, 0.01, 3, "convex", "xenium", None)
    with pytest.raises(dash.exceptions.PreventUpdate):
        xe.download_aggregates(0)
    with pytest.raises(dash.exceptions.PreventUpdate):
        xe.download_aggregates(1)                     # nothing computed
    assert "Pick where" in xe.run_aggregates(1, None, ["Imm"], 50, 10, 30, 99, 0.01, 3, "convex", "xenium", None)[2]
    assert "Pick the cell types" in xe.run_aggregates(1, "cluster", [], 50, 10, 30, 99, 0.01, 3, "convex", "xenium", None)[2]
    assert "radius must be positive" in xe.run_aggregates(1, "cluster", ["1"], 0, 10, 30, 99, 0.01, 3, "convex", "xenium", None)[2]
    assert not [j for j in xe._jobs.values() if j.kind == "aggregates"]
    with pytest.raises(dash.exceptions.PreventUpdate):
        xe.update_agg_label_options("tab-de", 0, 0, "xenium", 0)
    with pytest.raises(dash.exceptions.PreventUpdate):
        xe.toggle_agg_outlines(["on"], 3)             # nothing changed
    fig, rows, srows = xe.update_agg_results(0, 0, None, None)
    assert rows == [] and srows == []
    assert xe.update_agg_design_summary(0, 0, 0) == xe._design_summary()
    # the tab has its own "Samples & conditions…" button, which opens the shared editor
    assert "agg-design-open-btn" in str(xe._agg_tab()) and "agg-design-open-btn" in str(xe.app.layout)
    with pytest.raises(dash.exceptions.PreventUpdate):
        xe.open_design_modal(0, 0, 0, 0, 0)           # a re-created button, not a click
    assert xe.open_design_modal(None, None, None, None, 1)[0] is True
