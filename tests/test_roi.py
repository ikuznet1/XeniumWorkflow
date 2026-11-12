"""ROI helpers and the save / combine flow, checked against shapely computed here."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

import xe_testlib as tl


def _plot_xy(raw):
    """Cell positions in plot space (y negated), where ROI polygons live."""
    return raw.cells["x_centroid"].to_numpy(float), -raw.cells["y_centroid"].to_numpy(float)


def _rect_around(raw, frac_lo, frac_hi):
    """An axis-aligned rectangle (plot space) spanning a quantile band of x and all y."""
    x, y = _plot_xy(raw)
    x0, x1 = np.quantile(x, frac_lo) - 0.01, np.quantile(x, frac_hi) + 0.01
    return [[x0, y.min() - 5], [x1, y.min() - 5], [x1, y.max() + 5], [x0, y.max() + 5]]


def _brute_inside(poly, raw):
    from shapely.geometry import Point, Polygon
    p = Polygon(poly)
    x, y = _plot_xy(raw)
    return [cid for cid, a, b in zip(raw.cells.index, x, y) if p.contains(Point(a, b))]


def test_cells_in_polygon_equals_per_point_shapely(xe, raw):
    x, y = _plot_xy(raw)
    cx, cy = np.median(x), np.median(y)
    w, h = np.ptp(x) / 3 + 1, np.ptp(y) / 3 + 1
    shapes = {
        "rectangle": _rect_around(raw, 0.2, 0.7),
        "triangle": [[cx - w, cy - h], [cx + w, cy - h], [cx, cy + h]],
        "concave": [[cx - w, cy - h], [cx + w, cy - h], [cx + w, cy + h], [cx, cy], [cx - w, cy + h]],
        "far away": [[x.max() + 100, 0], [x.max() + 200, 0], [x.max() + 200, 50]],
    }
    hits = 0
    for name, poly in shapes.items():
        got = [str(c) for c in xe._roi_cells_in_polygon(poly)]
        assert got == _brute_inside(poly, raw), name
        hits += len(got)
    assert hits > 0
    mask = xe._roi_mask_in_polygon(shapes["rectangle"], xe.DATA["df"])
    assert mask.dtype == bool and mask.sum() == len(_brute_inside(shapes["rectangle"], raw))
    assert not xe._roi_mask_in_polygon([[0, 0], [1, 1]], xe.DATA["df"]).any(), "degenerate polygon: no cells"


def test_compute_hull_is_the_convex_hull_in_plot_space(xe, raw):
    from shapely.geometry import MultiPoint, Polygon
    ids = list(raw.cells.index[: min(12, len(raw.cells))])
    hull = xe._roi_compute_hull(ids)
    sub = raw.cells.loc[ids]
    want = MultiPoint(list(zip(sub["x_centroid"], -sub["y_centroid"]))).convex_hull
    assert Polygon(hull).symmetric_difference(want).area < 1e-6
    assert hull[0] == hull[-1], "closed ring"
    # every selected cell is inside or on the hull
    assert set(ids) <= set(_brute_inside(Polygon(hull).buffer(1e-6).exterior.coords, raw))
    assert xe._roi_compute_hull(ids[:2]) is None
    assert xe._roi_compute_hull(["__nope__"] * 5) is None
    # ids arrive from the plot as strings even when a table's index is not
    alt = pd.DataFrame({"x_centroid": [0.0, 10.0, 0.0, 10.0], "y_centroid": [0.0, 0.0, 10.0, 10.0]},
                       index=[1, 2, 3, 4])
    assert Polygon(xe._roi_compute_hull(["1", "2", "3", "4"], src_df=alt)).area == pytest.approx(100.0)


def test_lasso_capture_validates_the_selection(xe, raw):
    import dash
    ids = list(raw.cells.index[:6])
    points = {"points": [{"customdata": i} for i in ids]}
    hull, is_open, preview, err, done = xe.capture_lasso_selection(points, "xenium", 4)
    assert is_open and err == "" and done == 5 and str(len(ids)) in preview
    assert hull == xe._roi_state["pending_hull"] == xe._roi_compute_hull(ids)
    out = xe.capture_lasso_selection({"points": [{"customdata": i} for i in ids[:2]]}, "xenium", 4)
    assert out[0] is None and "at least 3" in out[3] and out[4] is dash.no_update
    for empty in (None, {}, {"points": []}, {"points": [{"x": 1}]}):
        with pytest.raises(dash.exceptions.PreventUpdate):
            xe.capture_lasso_selection(empty, "xenium", 0)


def test_save_roi_labels_cells_persists_and_reaches_every_view(xe, D, client, raw, cache_dir):
    poly_a, poly_b = _rect_around(raw, 0.0, 0.4), _rect_around(raw, 0.6, 1.0)
    in_a, in_b = _brute_inside(poly_a, raw), _brute_inside(poly_b, raw)
    assert in_a and in_b and not set(in_a) & set(in_b)
    with tl.new_threads():
        assert xe.save_roi(1, " left ", "zone", poly_a, 0, "xenium") == (False, "", 1)
        assert xe.save_roi(1, "right", "zone", poly_b, 1, "xenium") == (False, "", 2)
    col = D["df"]["roi_zone"]
    assert list(col.index[col == "left"].astype(str)) == in_a
    assert list(col.index[col == "right"].astype(str)) == in_b
    assert col.isna().sum() == len(raw.cells) - len(in_a) - len(in_b)

    # validation: nothing is added for a bad request
    assert xe.save_roi(1, "left", "zone", poly_a, 2, "xenium")[1].endswith("already exists.")
    assert xe.save_roi(1, "", "zone", poly_a, 2, "xenium")[1] == "ROI Name is required."
    assert "No polygon" in xe.save_roi(1, "x", "zone", None, 2, "xenium")[1]
    assert [r["name"] for r in xe._roi_state["rois"]] == ["left", "right"]
    assert len({r["color"] for r in xe._roi_state["rois"]}) == 2

    # cache file under the tmp cache root, readable by the next start
    path = xe._roi_cache_path()
    assert os.path.realpath(path).startswith(os.path.realpath(cache_dir))
    assert json.load(open(path)) == xe._roi_state["rois"] == xe._roi_load_cache()

    # written to the SpatialData store (full table), so a restart / export keeps the labels
    again = xe._load_from_xenium_sdata(D["sdata_path"], D["data_dir"])["df"]
    assert again["roi_zone"].replace({"": np.nan, "nan": np.nan}).dropna().to_dict() == col.dropna().to_dict()

    # figure overlay shapes, area chart, viewer overlay
    method = D["cluster_methods"][0] if D["cluster_methods"] else None
    fig = xe.make_spatial_fig("roi_zone", method, D["gene_names"][0], 3, 0.8, [], None)
    paths = [s for s in fig.layout.shapes if s.type == "path"]
    assert len(paths) == 2 and paths[0].line.color == xe._roi_state["rois"][0]["color"]
    cells = tl.fig_cells(fig)
    assert cells.loc[in_a, "color"].nunique() == 1 and cells.loc[in_a[0], "color"] != cells.loc[in_b[0], "color"]
    from shapely.geometry import Polygon
    area = xe.make_roi_area_fig()
    assert len(area.data) == 1 and list(area.data[0].x) == ["left", "right"]
    assert list(area.data[0].y) == pytest.approx([Polygon(poly_a).area, Polygon(poly_b).area])
    ov = client.get("/viewer/overlays").get_json()["polygons"]
    assert len(ov) == 2 and ov[0]["label"] == "left (zone)"
    # viewer polygons are in µm with y down: the plot-space polygon mirrored
    assert ov[0]["points"] == [[pytest.approx(x), pytest.approx(-y)] for x, y in poly_a]
    h, a = tl.get_bin(client, "/viewer/colors.bin?mode=roi_zone")
    names = np.array(h["categories"])[a["codes"]]
    assert names.tolist() == col.fillna("(none)").tolist()

    # hide toggle
    xe._roi_state["show"] = False
    assert client.get("/viewer/overlays").get_json()["polygons"] == []


def test_roi_labels_survive_subset_and_unsubset(xe, D, raw):
    poly = _rect_around(raw, 0.0, 0.5)
    inside = _brute_inside(poly, raw)
    keep = list(raw.cells.index[::2])
    xe.subset(cell_ids=keep)
    with tl.new_threads():
        xe.save_roi(1, "half", "zone", poly, 0, "xenium")
    sub = D["df"]["roi_zone"]
    assert set(sub.index[sub == "half"].astype(str)) == set(inside) & set(keep)
    xe.unsubset()
    full = D["df"]["roi_zone"]
    assert list(full.index[full == "half"].astype(str)) == inside, "cells outside the subset are labelled too"


def test_roi_metadata_is_rebuilt_from_the_roi_list(xe, raw):
    df = raw.cells[["x_centroid", "y_centroid"]].copy()
    a, b = _rect_around(raw, 0.0, 0.6), _rect_around(raw, 0.4, 1.0)
    rois = [{"name": "a", "cls": "k", "polygon_xy": a, "color": "#111111"},
            {"name": "b", "cls": "k", "polygon_xy": b, "color": "#222222"},
            {"name": "c", "cls": "other", "polygon_xy": a, "color": "#333333"}]
    xe._roi_apply_metadata_to_df(df, rois)
    in_a, in_b = set(_brute_inside(a, raw)), set(_brute_inside(b, raw))
    assert set(df.index[df["roi_k"] == "b"]) == in_b, "within a class the later ROI wins the overlap"
    assert set(df.index[df["roi_k"] == "a"]) == in_a - in_b
    assert set(df.index[df["roi_other"] == "c"]) == in_a
    # deleting the last ROI of a class clears its column instead of leaving stale labels
    xe._roi_apply_metadata_to_df(df, rois[:1])
    assert set(df.index[df["roi_k"] == "a"]) == in_a and df["roi_other"].isna().all()
    xe._roi_apply_metadata_to_df(df, [])
    assert df["roi_k"].isna().all()


@pytest.mark.parametrize("op", ["union", "intersection", "difference"])
def test_roi_set_operations_match_shapely(xe, D, raw, op):
    from shapely.geometry import Polygon
    a, b = _rect_around(raw, 0.0, 0.6), _rect_around(raw, 0.4, 1.0)
    with tl.new_threads():
        xe.save_roi(1, "a", "zone", a, 0, "xenium")
        xe.save_roi(1, "b", "zone", b, 1, "xenium")
        out = xe.apply_roi_operation(1, 0, op, 1, "res", "combo", 2)
    assert out[0] in ("", None) or "res" in str(out[0]) or out[1] == 3, out[0]
    res = xe._roi_state["rois"][-1]
    assert (res["name"], res["cls"]) == ("res", "combo")
    pa, pb = Polygon(a), Polygon(b)
    want = {"union": pa.union(pb), "intersection": pa.intersection(pb), "difference": pa.difference(pb)}[op]
    assert Polygon(res["polygon_xy"]).symmetric_difference(want).area < 1e-6
    col = D["df"]["roi_combo"]
    assert list(col.index[col == "res"].astype(str)) == _brute_inside(list(want.exterior.coords), raw)


def test_roi_set_operation_refuses_what_one_polygon_cannot_hold(xe, raw):
    a, b = _rect_around(raw, 0.0, 0.2), _rect_around(raw, 0.8, 1.0)
    x, y = _plot_xy(raw)
    inner = [[np.median(x) - 0.5, np.median(y) - 0.5], [np.median(x) + 0.5, np.median(y) - 0.5],
             [np.median(x) + 0.5, np.median(y) + 0.5], [np.median(x) - 0.5, np.median(y) + 0.5]]
    outer = _rect_around(raw, 0.0, 1.0)
    with tl.new_threads():
        for i, (n, p) in enumerate([("a", a), ("b", b), ("outer", outer), ("inner", inner)]):
            xe.save_roi(1, n, "zone", p, i, "xenium")
    n0 = len(xe._roi_state["rois"])
    assert "separate parts" in xe.apply_roi_operation(1, 0, "union", 1, "u", "z", 0)[0]
    assert "empty" in xe.apply_roi_operation(1, 0, "intersection", 1, "i", "z", 0)[0]
    assert "hole" in xe.apply_roi_operation(1, 2, "difference", 3, "d", "z", 0)[0]
    assert "Invalid ROI index" in xe.apply_roi_operation(1, 0, "union", 99, "x", "z", 0)[0]
    assert "result name" in xe.apply_roi_operation(1, 0, "union", 1, " ", "z", 0)[0]
    assert len(xe._roi_state["rois"]) == n0, "a refused operation adds nothing"


def test_rois_are_reloaded_from_the_cache_at_startup(xe, D, raw):
    poly = _rect_around(raw, 0.0, 0.5)
    with tl.new_threads():
        xe.save_roi(1, "persist", "zone", poly, 0, "xenium")
    saved = list(xe._roi_state["rois"])
    with xe._roi_lock:
        xe._roi_state["rois"] = []
    D["df"].drop(columns=["roi_zone"], inplace=True)
    xe._auto_load_rois()                      # what the module does at import
    assert xe._roi_state["rois"] == saved
    col = D["df"]["roi_zone"]
    assert list(col.index[col == "persist"].astype(str)) == _brute_inside(poly, raw)


def test_svg_path_helper(xe):
    assert xe._xy_to_svg_path([[0, 0], [1, 0], [1, 2]]).replace(" ", "") in ("M0,0L1,0L1,2Z", "M0.0,0.0L1.0,0.0L1.0,2.0Z")
    assert xe._hex_to_rgba("#ff0080", 0.5).replace(" ", "") == "rgba(255,0,128,0.5)"
    assert xe._roi_color(0) != xe._roi_color(1) and xe._roi_color(0) == xe._roi_color(len(xe._ROI_PALETTE))
