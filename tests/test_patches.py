"""Prepare Patches (SpatialData + tissue ROI + transcript patches via sopa), driven
through the same callbacks as the button. Pure Python, a few seconds on the tiny bundle.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

shapely = pytest.importorskip("shapely")


@pytest.fixture
def state(xe, prepared_patches):
    width, min_tx, bounds = prepared_patches
    st = xe._sdata_state
    roi = xe._sdata_transform_to_um(st["roi"])
    patches = xe._sdata_transform_to_um(st["patches"])
    return {"width": width, "min_tx": min_tx, "bounds": bounds, "sdata": st["sdata"],
            "roi": list(roi.geometry), "patches": list(patches.geometry), "qv": st.get("qv")}


def test_cache_store_lives_under_the_cache_root(xe, state, cache_dir):
    path = str(state["sdata"].path)
    assert path == xe._sdata_cache_path()
    assert os.path.realpath(path).startswith(os.path.realpath(cache_dir) + os.sep)
    assert state["sdata"].is_backed()
    assert os.path.exists(os.path.join(path, ".complete")), "a finished store is marked as such"


def test_spatialdata_holds_every_transcript_with_a_quality_flag(state, raw):
    points = state["sdata"].points["transcripts"]
    assert len(points) == raw.n_transcripts
    if raw.n_transcripts <= 5_000_000:
        df = points[["transcript_id", "qv", "low_quality_transcript"]].compute()
        truth = raw.transcripts(columns=["transcript_id", "qv"]).set_index("transcript_id")
        got = df.set_index("transcript_id")
        assert set(got.index) == set(truth.index)
        flag = got["low_quality_transcript"].astype(bool)
        assert (flag == (truth.loc[got.index, "qv"] < state["qv"])).all(), "flag == below the Q threshold"
    assert state["qv"] == 20


def test_tissue_roi_and_patches_are_in_micrometres_over_the_tissue(state, raw):
    c = raw.cells
    union = shapely.unary_union(state["roi"])
    assert union.area > 0
    x0, y0, x1, y1 = union.bounds
    pad = 60.0
    assert x0 > c.x_centroid.min() - pad and x1 < c.x_centroid.max() + pad, "ROI is in µm, near the cells"
    assert y0 > c.y_centroid.min() - pad and y1 < c.y_centroid.max() + pad
    in_roi = shapely.contains_xy(union, c.x_centroid.to_numpy(), c.y_centroid.to_numpy())
    assert in_roi.mean() > 0.5, f"only {in_roi.mean():.0%} of the cells are inside the tissue ROI"
    # patches are pieces of the ROI
    for p in state["patches"]:
        assert p.difference(union.buffer(1e-6)).area <= 1e-6 * max(p.area, 1.0)
    assert state["bounds"] == [tuple(float(v) for v in p.bounds) for p in state["patches"]]
    assert all(b[2] - b[0] <= state["width"] + 1e-6 and b[3] - b[1] <= state["width"] + 1e-6
               for b in state["bounds"])


def test_each_patch_holds_enough_good_transcripts(state, raw):
    if raw.n_transcripts > 5_000_000:
        pytest.skip("whole-file transcript read is for small bundles")
    tx = raw.transcripts(columns=["x_location", "y_location", "qv"])
    good = tx[tx["qv"] >= state["qv"]]
    counted = np.zeros(len(good), dtype=int)
    for p in state["patches"]:
        inside = shapely.contains_xy(p, good["x_location"].to_numpy(), good["y_location"].to_numpy())
        assert inside.sum() >= state["min_tx"], "a patch below the minimum should have been dropped"
        counted += inside
    assert (counted > 0).mean() > 0.5, "the patches hold most of the usable transcripts"


def test_patch_layout_is_recorded_and_restored_only_for_the_same_request(xe, state):
    sdata, w, m, qv = state["sdata"], state["width"], state["min_tx"], state["qv"]
    record_path = os.path.join(str(sdata.path), xe._SDATA_PATCH_PARAMS_FILE)
    record = json.load(open(record_path))
    assert record["patch_width"] == w and record["min_transcripts"] == m and record["qv_threshold"] == qv
    assert record["n_patches"] == len(state["patches"]) and record["patches_hash"] and record["roi_hash"]

    restored = xe._sdata_restore_patches(sdata, w, m, qv)
    assert restored is not None
    assert [g.bounds for g in xe._sdata_transform_to_um(restored).geometry] == [p.bounds for p in state["patches"]]
    # any other request must be recomputed, never answered with these patches
    assert xe._sdata_restore_patches(sdata, w + 50, m, qv) is None
    assert xe._sdata_restore_patches(sdata, w, m + 1, qv) is None
    assert xe._sdata_restore_patches(sdata, w, m, 30) is None
    # a record that no longer matches the stored shapes is not trusted
    try:
        json.dump({**record, "patches_hash": "0" * 32}, open(record_path, "w"))
        assert xe._sdata_restore_patches(sdata, w, m, qv) is None
        os.remove(record_path)
        assert xe._sdata_restore_patches(sdata, w, m, qv) is None
    finally:
        json.dump(record, open(record_path, "w"))
    assert xe._sdata_restore_patches(sdata, w, m, qv) is not None


def test_prepare_button_reports_existing_patches(xe, state):
    out = xe.start_prepare_patches(1, 20)
    assert out[0] is True and out[2] == ["yes"] and out[3] == ["yes"]
    assert f"{len(state['patches'])} patches ready" in out[5]
    tick = xe.poll_sdata(0, 4, state["width"], state["min_tx"])
    assert tick[1] is True and tick[4] == 5 and f"{len(state['patches'])} patches ready" in tick[8]


def test_overlays_show_roi_and_patch_grid(xe, client, state, D):
    assert client.get("/viewer/overlays").get_json()["polygons"] == []
    only_roi = client.get("/viewer/overlays?sdata_roi=1").get_json()["polygons"]
    both = client.get("/viewer/overlays?sdata_roi=1&sdata_patches=1").get_json()["polygons"]
    assert len(only_roi) == len(state["roi"]) and len(both) == len(state["roi"]) + len(state["patches"])
    for poly, geom in zip(only_roi, state["roi"]):
        assert shapely.Polygon(poly["points"]).symmetric_difference(geom).area < 1e-6 * geom.area
    for poly, b in zip(both[len(state["roi"]):], state["bounds"]):
        xs, ys = [p[0] for p in poly["points"]], [p[1] for p in poly["points"]]
        assert (min(xs), min(ys), max(xs), max(ys)) == pytest.approx(b)
    method = D["cluster_methods"][0] if D["cluster_methods"] else None
    plain = xe.make_spatial_fig("transcript_counts", method, D["gene_names"][0], 3, 0.8, [], None)
    shown = xe.make_spatial_fig("transcript_counts", method, D["gene_names"][0], 3, 0.8, [], None,
                                sdata_roi_on=True, sdata_patches_on=True)
    assert len(plain.layout.shapes) == 0
    assert len(shown.layout.shapes) == len(state["roi"]) + len(state["patches"])
    rects = [s for s in shown.layout.shapes if s.type == "rect"]
    # plot space: y is negated
    got = np.array(sorted((r.x0, r.x1, min(r.y0, r.y1), max(r.y0, r.y1)) for r in rects))
    want = np.array(sorted((b[0], b[2], -b[3], -b[1]) for b in state["bounds"]))
    np.testing.assert_allclose(got, want, atol=1e-6)
