"""Image-based resegmentation (xenium_app/imgseg.py): the classical backend on fabricated
images, transcript assignment against an independent point-in-polygon count, the Cellpose
detection (never a download), and a whole run on the bundle's morphology image through the
callback of the Run button: staging, params.json, cached-run entry, zarr, reload,
boundaries in both renderers, Segmentation QC source, cancellation between patches.

The real Cellpose run is in the ``tools`` tier and is skipped unless cellpose is installed
with its weights already on disk.
"""
from __future__ import annotations

import json
import os
import sys
import threading

import dash
import numpy as np
import pandas as pd
import pytest

import xe_testlib as tl

shapely = pytest.importorskip("shapely")


# ── fabricated images ────────────────────────────────────────────────────────
def _disc_image(centres, radius, shape=(300, 300), level=3000.0, background=200.0, noise=30.0, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    img = np.full(shape, background, dtype=np.float64)
    for cx, cy in centres:
        d = np.hypot(xx - cx, yy - cy)
        img = np.maximum(img, background + (level - background) * np.clip(1.2 - d / radius, 0, 1).clip(0, 1))
    img += rng.normal(0, noise, shape)
    return np.clip(img, 0, 65535).astype(np.uint16)[None]


def test_classical_mask_finds_the_planted_nuclei_and_splits_touching_ones(xe):
    centres = [(40, 40), (120, 50), (200, 60), (60, 150), (150, 150), (240, 160), (50, 250), (130, 240),
               (220, 240), (236, 250)]                      # the last two touch (16 px apart, radius 12)
    img = _disc_image(centres, radius=12)
    mask = xe._iseg_classical_mask(img, diameter_px=24)
    assert mask.shape == img.shape[1:] and mask.dtype == np.int32
    at_centres = [int(mask[cy, cx]) for cx, cy in centres]
    assert all(v > 0 for v in at_centres), "every planted nucleus is inside a label"
    assert len(set(at_centres)) == len(centres), "one label per nucleus, the touching pair split"
    assert int(mask.max()) == len(centres), "no spurious objects in the noise"
    assert sorted(np.unique(mask)) == list(range(len(centres) + 1)), "labels are 1..n"
    area = np.bincount(mask.ravel())[1:]
    assert (area > 0.4 * np.pi * 12 ** 2).all() and (area < 2.0 * np.pi * 12 ** 2).all()


def test_classical_mask_is_empty_on_noise_and_on_a_flat_image(xe):
    rng = np.random.default_rng(1)
    noise = rng.normal(500, 20, (1, 200, 200)).clip(0).astype(np.uint16)
    assert xe._iseg_classical_mask(noise, 24).max() == 0
    assert xe._iseg_classical_mask(np.full((1, 64, 64), 7, np.uint16), 24).max() == 0
    assert xe._iseg_classical_mask(np.zeros((1, 0, 0), np.uint16), 24).size == 0


def test_expansion_grows_every_label_without_overlap(xe):
    centres = [(50, 50), (100, 50), (200, 200)]
    img = _disc_image(centres, radius=10)
    base = xe._iseg_classical_mask(img, 20)
    grown = xe._iseg_classical_mask(img, 20, expand_px=15)
    assert int(base.max()) == int(grown.max()) == 3
    assert ((base > 0) <= (grown > 0)).all() and (grown[base > 0] == base[base > 0]).all()
    a0, a1 = np.bincount(base.ravel())[1:], np.bincount(grown.ravel())[1:]
    assert (a1 > 1.5 * a0).all()
    # no pixel further than the distance from its own nucleus; the gap between the two
    # close nuclei is shared, each pixel going to the nearer one
    from scipy import ndimage as ndi
    d, (iy, ix) = ndi.distance_transform_edt(base == 0, return_indices=True)
    new = (grown > 0) & (base == 0)
    assert (d[new] <= 15 + 1e-9).all() and (grown[new] == base[iy[new], ix[new]]).all()
    direct = xe._iseg_expand_labels(base, 15)
    assert (direct == grown).all() and (xe._iseg_expand_labels(base, 0) == base).all()


def test_scipy_only_watershed_gives_the_same_objects(xe, monkeypatch):
    centres = [(40, 40), (120, 50), (200, 60), (60, 150), (150, 150)]
    img = _disc_image(centres, radius=12)
    ref = xe._iseg_classical_mask(img, 24)
    monkeypatch.setitem(sys.modules, "skimage.segmentation", None)      # import fails -> scipy path
    alt = xe._iseg_classical_mask(img, 24)
    assert int(alt.max()) == int(ref.max()) == len(centres)
    for cx, cy in centres:
        a, b = alt == alt[cy, cx], ref == ref[cy, cx]
        assert (a & b).sum() / (a | b).sum() > 0.8


# ── transcripts -> polygons ──────────────────────────────────────────────────
def test_transcript_assignment_equals_an_independent_point_in_polygon_count(xe, tmp_path):
    rng = np.random.default_rng(3)
    n = 6000
    tx = pd.DataFrame({
        "x_location": rng.uniform(0, 2500, n).astype(np.float32),
        "y_location": rng.uniform(0, 1200, n).astype(np.float32),
        "feature_name": rng.choice(["GeneA", "GeneB", "GeneC", "NegControlProbe_1"], n),
        "qv": rng.choice([10.0, 25.0, 40.0], n),
    })
    tx["is_gene"] = ~tx["feature_name"].str.startswith("NegControl")
    path = str(tmp_path / "transcripts.parquet")
    tx.to_parquet(path, row_group_size=500)
    polys = [shapely.box(100, 100, 400, 500), shapely.box(350, 450, 700, 800),      # these two overlap
             shapely.Point(1900, 600).buffer(250), shapely.box(2300, 1000, 2500, 1200),
             shapely.box(5000, 5000, 5100, 5100)]                                    # no transcripts
    counts, n_read, n_assigned = xe._iseg_assign_transcripts(polys, path, qv_min=20, tile_um=700,
                                                             bounds=(0, 0, 2500, 1200))
    ok = tx[(tx.qv >= 20) & tx.is_gene]
    assert n_read == len(ok), "every Q>=20 gene transcript of the bounds is read exactly once"
    px, py = ok.x_location.to_numpy(float), ok.y_location.to_numpy(float)
    owner = np.full(len(ok), -1)
    for i in reversed(range(len(polys))):                    # the first polygon wins an overlap
        owner[shapely.contains_xy(polys[i], px, py)] = i
    truth = (ok.assign(cell=owner)[owner >= 0].groupby(["cell", "feature_name"]).size())
    got = counts.set_index(["cell", "gene"])["n"]
    assert n_assigned == int((owner >= 0).sum()) == int(got.sum())
    assert got.sort_index().to_dict() == truth.sort_index().to_dict()
    assert "NegControlProbe_1" not in set(counts["gene"])
    # default bounds: the polygons' bounding box
    c2, r2, a2 = xe._iseg_assign_transcripts(polys[:2], path, qv_min=20)
    assert a2 == int(np.isin(owner, [0, 1]).sum())
    assert xe._iseg_assign_transcripts([], path)[1:] == (0, 0)


# ── Cellpose detection: never a download ─────────────────────────────────────
def test_cellpose_absent_is_reported_with_what_to_install(xe, monkeypatch):
    monkeypatch.setitem(sys.modules, "cellpose", None)                 # "import cellpose" raises ImportError
    st = xe._iseg_cellpose_status()
    assert st["available"] is False and st["models"] == {}
    assert "pip install" in st["message"] and "downloads" in st["message"] and "Cellpose project" in st["message"]
    opts = xe._iseg_backend_options()
    assert [o["value"] for o in opts if not o.get("disabled")] == ["classical"]
    assert any(o.get("disabled") and o["value"] == "cellpose" for o in opts)
    with pytest.raises(RuntimeError, match="pip install"):
        xe._iseg_make_method("cellpose:nuclei", 30)
    # the Run button refuses with the same note and starts nothing
    out = xe.run_imgseg_modal(1, "imgseg", 1, "cellpose:nuclei", "DAPI", None, None, None, None, None,
                              8, 5, 5, 500, 1.0, 0.4, 0.0, [])
    assert out[0].startswith("✗") and "pip install" in out[0]
    assert xe._imgseg_state["status"] == "idle" and not xe._jobs_list(active_only=True)
    # the tab shows the note
    panel = xe.switch_imgseg_tab("imgseg", 1, "cellpose:nuclei", None)
    assert panel[2] == "classical" and "pip install" in panel[3]


def test_cellpose_without_local_weights_is_not_used(xe, monkeypatch):
    def status():
        return {"available": False, "version": "3.1.0", "models": {}, "model_dir": "x",
                "message": "Cellpose 3.1.0 is installed but none of its model weights are there. The first "
                           "Cellpose run downloads them from the Cellpose project"}
    monkeypatch.setattr(xe, "_iseg_cellpose_status", status)
    with pytest.raises(RuntimeError, match="downloads them from the Cellpose project"):
        xe._iseg_make_method("cellpose", 30)
    assert [o["value"] for o in xe._iseg_backend_options() if not o.get("disabled")] == ["classical"]


def test_cellpose_status_lists_only_weights_that_exist(xe, monkeypatch, tmp_path):
    import types
    fake = types.ModuleType("cellpose")
    fake.version = "3.1.0"
    models = types.ModuleType("cellpose.models")
    models.MODEL_DIR = tmp_path
    fake.models = models
    monkeypatch.setitem(sys.modules, "cellpose", fake)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    st = xe._iseg_cellpose_status()
    assert st["available"] is False and "downloads" in st["message"] and "never" in st["message"]
    (tmp_path / "nucleitorch_0").write_bytes(b"weights")
    (tmp_path / "cyto3").write_bytes(b"")                       # an empty file is not a model
    st = xe._iseg_cellpose_status()
    assert st["available"] is True and list(st["models"]) == ["nuclei"]
    assert st["models"]["nuclei"] == str(tmp_path / "nucleitorch_0")
    with pytest.raises(RuntimeError, match="not on this machine"):
        xe._iseg_make_method("cellpose:cyto3", 30)             # asked-for model has no weights: refused


def test_unknown_backend_and_missing_image_are_errors(xe, monkeypatch):
    with pytest.raises(RuntimeError, match="Unknown"):
        xe._iseg_make_method("stardist", 30)
    monkeypatch.setitem(xe._sdata_state, "sdata", None)
    assert xe._iseg_channels() == []
    xe._run_imgseg("classical", ["DAPI"])
    assert xe._imgseg_state["status"] == "error" and "Step 1" in xe._imgseg_state["message"]


# ── a whole run on the bundle's image ────────────────────────────────────────
def _run(xe, *, backend="classical", channel=None, region=(None, None, None, None), diameter=8, expand=5,
         min_tx=1, patch=200, timeout=1800):
    channel = channel or xe._iseg_default_channel(xe._iseg_channels())
    out = xe.run_imgseg_modal(1, "imgseg", 1, backend, channel, None, *region, diameter, expand, min_tx, patch,
                              1.0, 0.4, 0.0, [])
    assert out == ("Image segmentation starting…", False, False), out
    tl.wait_state(xe._imgseg_state, xe._imgseg_lock, timeout=timeout)
    return dict(xe._imgseg_state["result"])


@pytest.fixture(scope="module")
def run(xe, prepared_patches):
    res = _run(xe)
    yield res
    import shutil
    shutil.rmtree(res["out_dir"], ignore_errors=True)


@pytest.fixture
def loaded(xe, run):
    with xe._imgseg_lock:
        xe._imgseg_state.update({"status": "done", "message": "Loaded", "result": dict(run)})
    return run


def test_the_modal_tab_lists_the_image_channels(xe, prepared_patches, monkeypatch):
    # the tab is tested without importing Cellpose (its probe imports torch: seconds); what
    # the tab shows for each Cellpose situation is tested above with stand-in statuses
    monkeypatch.setattr(xe, "_iseg_cellpose_status", lambda: {
        "available": False, "version": None, "models": {}, "model_dir": None, "message": "not probed in this test"})
    chans = xe._iseg_channels()
    assert chans and chans == [str(c) for c in xe._iseg_image()[1].coords["c"].values]
    style, opts, value, note, ch_opts, ch, cyto_opts = xe.switch_imgseg_tab("imgseg", 1, None, None)
    assert style == {} and value == "classical" and opts[0]["value"] == "classical"
    assert [o["value"] for o in ch_opts] == chans and ch == xe._iseg_default_channel(chans)
    assert xe.switch_imgseg_tab("baysor", 1, None, None)[0] == {"display": "none"}
    assert xe.switch_reseg_tab("imgseg")[:2] == ({"display": "none"}, {"display": "none"})


def test_run_is_stored_like_a_baysor_or_proseg_run(xe, run, cache_dir):
    out_dir = run["out_dir"]
    name = os.path.basename(out_dir)
    assert name.startswith(f"imgseg_{xe._dataset_name()}_") and os.path.dirname(out_dir) == str(cache_dir)
    assert not [d for d in os.listdir(cache_dir) if d.endswith(".partial")], "no staging directory is left"
    assert not os.path.exists(os.path.join(out_dir, "patches")), "per-patch files are merged and removed"
    with open(os.path.join(out_dir, "params.json")) as f:
        p = json.load(f)
    assert p["tool"] == "imgseg" and p["backend"] == "classical" and p["qv_min"] == 20
    assert p["param_tag"] == name.rsplit("_", 1)[-1] and p["n_cells"] == len(run["cells_df"])
    assert p["channels"] == [xe._iseg_default_channel(xe._iseg_channels())] and p["n_patches"] >= 1
    assert 0 < p["assigned_fraction"] <= 1 and p["n_assigned"] == int(run["cells_df"]["transcript_counts"].sum())
    assert run["source"] == "imgseg" and os.path.isdir(run["sdata_path"])
    assert os.path.basename(run["sdata_path"]) == "spatialdata_imgseg.zarr"
    listed = [r for r in xe._list_cached_seg_runs() if r["value"] == f"imgseg:{p['param_tag']}"]
    assert len(listed) == 1 and listed[0]["out_dir"] == out_dir
    assert listed[0]["label"].startswith("Image: classical watershed") and f"{p['n_cells']:,} cells" in listed[0]["label"]
    assert xe._seg_run_tag(name, "imgseg") == p["param_tag"]
    assert xe._seg_run_tag(name, "baysor") is None and xe._seg_run_tag(name, "proseg") is None


def test_counts_equal_an_independent_recount_of_the_transcript_file(xe, run, raw):
    cells, expr = run["cells_df"], run["expr"]
    assert len(cells) >= 3, "the classical backend finds cells on the bundle's nuclear image"
    assert expr.shape == (len(cells), len(xe._panel_gene_names()))
    genes = tl.gene_tx(raw.transcripts())                       # Q >= 20, gene features only
    px, py = genes["x_location"].to_numpy(float), genes["y_location"].to_numpy(float)
    gix = {g: i for i, g in enumerate(xe._panel_gene_names())}
    col = genes["feature_name"].astype(str).map(gix).to_numpy()
    owner = np.full(len(genes), -1)
    ids = list(cells.index)
    for i in reversed(range(len(ids))):
        vx, vy = run["cell_bounds"][ids[i]]
        owner[shapely.contains_xy(shapely.Polygon(np.column_stack([vx, vy])), px, py)] = i
    truth = np.zeros(expr.shape)
    np.add.at(truth, (owner[owner >= 0], col[owner >= 0].astype(int)), 1)
    got = expr.toarray()
    # float32 boundary coordinates: a transcript within a hair of an outline may fall on either side
    assert np.abs(got - truth).sum() <= max(2, 0.002 * truth.sum())
    assert (cells["transcript_counts"].to_numpy() == got.sum(axis=1)).all()
    assert (cells["transcript_counts"] >= 1).all() and (cells["cell_area"] > 0).all()
    # centroids are inside the image, in µm, near the Xenium cells
    c = raw.cells
    assert cells["x_centroid"].between(c.x_centroid.min() - 60, c.x_centroid.max() + 60).all()
    assert cells["y_centroid"].between(c.y_centroid.min() - 60, c.y_centroid.max() + 60).all()
    assert any(k.startswith("cluster") for k in cells.columns) and "umap_1" in cells.columns


def test_cached_run_reloads_identically_and_is_selectable(xe, run):
    tag = os.path.basename(run["out_dir"]).rsplit("_", 1)[-1]
    assert xe._imgseg_state["result"] is None                   # state was reset after the previous test
    assert xe.load_seg_run_from_cache(f"imgseg:{tag}", 0, 0) == (dash.no_update,) * 5
    out = xe.load_imgseg_run_from_cache({"value": f"imgseg:{tag}", "n": 1})
    assert out == (dash.no_update, False)
    tl.wait_state(xe._imgseg_state, xe._imgseg_lock, timeout=300)
    got = xe._imgseg_state["result"]
    assert list(got["cells_df"].index) == list(run["cells_df"].index)
    assert tl.sparse_equal(got["expr"], run["expr"])
    assert set(got["cell_bounds"]) == set(run["cell_bounds"]) and got["out_dir"] == run["out_dir"]
    np.testing.assert_allclose(got["cells_df"]["x_centroid"], run["cells_df"]["x_centroid"], rtol=1e-6)
    status, disabled, done, seg, _line = xe.poll_imgseg(0, 4, "xenium")
    assert status.startswith("✓") and disabled is True and done == 5 and seg == f"imgseg:{tag}"
    bopts, sopts = xe.update_boundary_options(0, 5, 0, [{"label": "x", "value": "imgseg", "disabled": True},
                                                          {"label": "y", "value": "cell"}])
    assert bopts[0]["disabled"] is False
    assert any(o["value"] == f"imgseg:{tag}" and o["label"].startswith("★ Image:") for o in sopts)
    assert xe._seg_result(f"imgseg:{tag}") is got and xe._an_alt_res(f"imgseg:{tag}") is got
    assert xe._seg_result("xenium") is None


def test_both_renderers_draw_the_run_and_its_boundaries(xe, loaded, client):
    cells = loaded["cells_df"]
    seg = "imgseg:" + os.path.basename(loaded["out_dir"]).rsplit("_", 1)[-1]
    x0, x1 = float(cells.x_centroid.min()) - 30, float(cells.x_centroid.max()) + 30
    y0, y1 = float(cells.y_centroid.min()) - 30, float(cells.y_centroid.max()) + 30
    relayout = {"xaxis.range[0]": x0, "xaxis.range[1]": x1, "yaxis.range[0]": -y1, "yaxis.range[1]": -y0}
    fig = xe.make_spatial_fig("transcript_counts", None, None, 3, 0.9, ["imgseg"], relayout, alt_tool="imgseg")
    assert "[Image segmentation]" in fig.layout.title.text
    assert len(tl.fig_cells(fig)) == len(cells)
    lines = [t for t in fig.data if t.name == "Image segmentation Boundaries"]
    assert len(lines) == 1 and len(tl.line_polygons(lines[0])) == len(loaded["cell_bounds"])
    for color_by in ("cluster", "gene"):
        f2 = xe.make_spatial_fig(color_by, None, xe._panel_gene_names()[0], 3, 0.9, [], relayout, alt_tool="imgseg")
        assert len(tl.fig_cells(f2)) == len(cells)
    umap = xe.make_umap_fig("cluster", None, None, 3, 0.9, alt_tool="imgseg")
    assert "Image segmentation" in umap.layout.title.text
    # WebGL view: the source, its cells and its polygons
    assert "imgseg" in client.get("/viewer/meta").get_json()["sources"]
    h, a = tl.get_bin(client, f"/viewer/cells.bin?source={seg}")
    assert h["n"] == len(cells)
    hb, ab = tl.get_bin(client, f"/viewer/boundaries.bin?source={seg}&kind=imgseg&x0={x0}&y0={y0}&x1={x1}&y1={y1}")
    assert hb["n"] == len(loaded["cell_bounds"]) and len(tl.rings(ab)) == hb["n"]
    # without a loaded run the kind is reported as not available
    with xe._imgseg_lock:
        xe._imgseg_state.update({"status": "idle", "result": None})
    hb, _ = tl.get_bin(client, f"/viewer/boundaries.bin?kind=imgseg&x0={x0}&y0={y0}&x1={x1}&y1={y1}")
    assert hb["n"] == 0 and hb.get("available") is False


def test_run_is_a_segmentation_qc_source_and_a_view_table(xe, loaded):
    seg = "imgseg:" + os.path.basename(loaded["out_dir"]).rsplit("_", 1)[-1]
    srcs = {s["value"]: s for s in xe._segqc_sources()}
    assert seg in srcs and srcs[seg]["tool"] == "imgseg" and srcs[seg]["sample"] is not None
    assert srcs[seg]["sample"]["csc"].shape[0] == len(loaded["cells_df"])
    tabs = xe._view_tables(seg)
    assert len(tabs) == 1 and tabs[0]["df"] is loaded["cells_df"] and tabs[0]["label"] == "imgseg"
    smp = xe._an_samples(seg)
    assert len(smp) == 1 and smp[0]["csc"].shape[0] == len(loaded["cells_df"])
    assert xe._seg_tag(seg) == f"imgseg:{loaded['out_dir']}"
    assert [t for t, r in xe._loaded_reseg_results() if r["cells_df"] is loaded["cells_df"]] == ["imgseg"]
    key = xe._labels_key_for_method("celltypist", loaded)
    assert "_imgseg_" in key and xe._label_key_for_active_seg("labels_celltypist", None) == "labels_celltypist"
    assert xe._label_key_for_active_seg(key, None) != key, "labels of an image-segmentation run are not Xenium labels"


def test_region_run_keeps_only_cells_inside_and_gets_its_own_tag(xe, run, prepared_patches):
    cells = run["cells_df"]
    xm, ym = float(cells.x_centroid.iloc[0]), float(cells.y_centroid.iloc[0])
    region = (round(xm - 40, 1), round(xm + 40, 1), round(ym - 30, 1), round(ym + 30, 1))
    res = _run(xe, region=region)
    try:
        assert res["out_dir"] != run["out_dir"]
        c = res["cells_df"]
        assert len(c) and c.x_centroid.between(region[0], region[1]).all() and c.y_centroid.between(region[2], region[3]).all()
        with open(os.path.join(res["out_dir"], "params.json")) as f:
            p = json.load(f)
        assert (p["x_min"], p["x_max"], p["y_min"], p["y_max"]) == region
        assert "x=" in next(r["label"] for r in xe._list_cached_seg_runs() if r["out_dir"] == res["out_dir"])
        # the dataset's own store was not given image patches
        assert "image_patches" not in xe._sdata_state["sdata"].shapes
    finally:
        import shutil
        shutil.rmtree(res["out_dir"], ignore_errors=True)


def test_cancel_between_patches_leaves_nothing_behind(xe, prepared_patches, monkeypatch, cache_dir):
    started, release = threading.Event(), threading.Event()
    real = xe._iseg_make_method

    def slow(*a, **k):
        method, info = real(*a, **k)

        def blocked(image):
            started.set()
            release.wait(60)
            return method(image)
        return blocked, info
    monkeypatch.setattr(xe, "_iseg_make_method", slow)
    before = set(os.listdir(cache_dir))
    out = xe.run_imgseg_modal(1, "imgseg", 1, "classical", xe._iseg_channels()[0], None, None, None, None, None,
                              8, 5, 1, 150, 1.0, 0.4, 0.0, [])      # patch width 150: its own cache tag
    assert out[0] == "Image segmentation starting…"
    assert started.wait(120), "the first patch never started"
    assert any(d.endswith(".partial") for d in set(os.listdir(cache_dir)) - before), "the run works in a staging dir"
    assert xe.run_imgseg_modal(1, "imgseg", 1, "classical", xe._iseg_channels()[0], None, None, None, None, None,
                               8, 5, 1, 150, 1.0, 0.4, 0.0, [])[0] == "Already running…"
    job = next(j for j in xe._jobs_list(active_only=True) if j.kind == "imgseg")
    assert xe._job_cancel(job.id) is True
    release.set()
    job.thread.join(60)
    assert job.state == "cancelled"
    assert xe._imgseg_state["status"] == "error" and xe._imgseg_state["message"] == "Cancelled"
    assert xe._imgseg_state["result"] is None
    assert set(os.listdir(cache_dir)) == before, "no run directory and no staging directory"
    status, disabled, _done, _seg, line = xe.poll_imgseg(0, 0, "xenium")
    assert status == "✗ Cancelled" and disabled is True and "Cancelled" in line


def test_no_output_is_written_by_two_callbacks_with_the_same_inputs(xe):
    """Dash names an allow_duplicate output "id.prop@<hash of the callback's inputs>": two
    callbacks with the same inputs and a common duplicate output collide in the browser
    ("Duplicate callback outputs") although the server registers both."""
    seen, twice = {}, []
    for key, cb in xe.app.callback_map.items():
        outs = key[2:-2].split("...") if key.startswith("..") else [key]
        name = getattr(cb.get("callback"), "__name__", "clientside callback")
        for o in outs:
            if o in seen:
                twice.append(f"{o}: {seen[o]} and {name}")
            seen[o] = name
    mine = [t for t in twice if "imgseg" in t or "iseg" in t]
    assert not mine, mine
    assert not twice, twice
    # the Run button and the dropdown reach the image callbacks through their request stores
    assert xe.request_imgseg_run(3, "imgseg") == 3
    assert xe.request_imgseg_load("imgseg:abcd1234", {"n": 4}) == {"value": "imgseg:abcd1234", "n": 5}
    for call in (lambda: xe.request_imgseg_run(3, "baysor"), lambda: xe.request_imgseg_run(0, "imgseg"),
                 lambda: xe.request_imgseg_load("baysor:abcd1234"), lambda: xe.request_imgseg_load("xenium")):
        with pytest.raises(dash.exceptions.PreventUpdate):
            call()


def test_other_tabs_of_the_modal_ignore_the_image_tab(xe):
    import dash
    with pytest.raises(dash.exceptions.PreventUpdate):
        xe.run_reseg_modal(1, "imgseg", 1, *([None] * 24))
    with pytest.raises(dash.exceptions.PreventUpdate):
        xe.run_imgseg_modal(1, "baysor", 1, "classical", "DAPI", None, None, None, None, None, 8, 5, 5, 500,
                            1.0, 0.4, 0.0, [])
    with pytest.raises(dash.exceptions.PreventUpdate):
        xe.run_imgseg_modal(0, "imgseg", 1, "classical", "DAPI", None, None, None, None, None, 8, 5, 5, 500,
                            1.0, 0.4, 0.0, [])
    assert xe.run_imgseg_modal(1, "imgseg", None, "classical", "DAPI", None, None, None, None, None, 8, 5, 5,
                               500, 1.0, 0.4, 0.0, [])[0].startswith("✗ Please confirm")


# ── real Cellpose (only with local weights; never downloads) ─────────────────
@pytest.mark.tools
def test_cellpose_backend_runs_with_local_weights(xe, prepared_patches, raw):
    st = xe._iseg_cellpose_status()
    if not st["available"]:
        pytest.skip("cellpose with local model weights is not available: " + st["message"][:80])
    pytest.importorskip("sopa")
    model = "nuclei" if "nuclei" in st["models"] else next(iter(st["models"]))
    before = {f: os.path.getmtime(os.path.join(st["model_dir"], f)) for f in os.listdir(st["model_dir"])}
    res = _run(xe, backend=f"cellpose:{model}", min_tx=1, patch=300, timeout=3600)
    try:
        after = {f: os.path.getmtime(os.path.join(st["model_dir"], f)) for f in os.listdir(st["model_dir"])}
        assert after == before, "nothing was downloaded into the Cellpose model directory"
        with open(os.path.join(res["out_dir"], "params.json")) as f:
            p = json.load(f)
        assert p["backend"] == f"cellpose:{model}" and p["model"] == model and p["weights_file"]
        assert len(res["cells_df"]) >= 3 and res["expr"].sum() == res["cells_df"]["transcript_counts"].sum()
    finally:
        import shutil
        shutil.rmtree(res["out_dir"], ignore_errors=True)


# ── review round 5 ───────────────────────────────────────────────────────────
def test_only_cyto_type_models_get_the_membrane_channel(xe, monkeypatch, prepared_patches):
    models = {"nuclei": "/w/nucleitorch_0", "cyto3": "/w/cyto3"}
    monkeypatch.setattr(xe, "_iseg_cellpose_status",
                        lambda: {"available": True, "version": "3.1.0", "models": models, "model_dir": "/w", "message": ""})
    assert not xe._iseg_uses_second_channel("classical") and not xe._iseg_uses_second_channel("cellpose:nuclei")
    assert xe._iseg_uses_second_channel("cellpose:cyto3") and xe._iseg_uses_second_channel("cellpose:cyto2")
    assert xe._iseg_cellpose_model("cellpose", 1) == "nuclei" and xe._iseg_cellpose_model("classical") is None
    started = []
    monkeypatch.setattr(xe, "_job_start", lambda kind, label, target, **kw: started.append(kw["kwargs"]))
    for backend, want in (("cellpose:nuclei", ["DAPI"]), ("cellpose:cyto3", ["MEM", "DAPI"]), ("classical", ["DAPI"])):
        with xe._imgseg_lock:
            xe._imgseg_state.update({"status": "idle", "message": "", "result": None})
        out = xe.run_imgseg_modal(1, "imgseg", 1, backend, "DAPI", "MEM", None, None, None, None,
                                  8, 5, 5, 500, 1.0, 0.4, 0.0, [])
        assert out[0] == "Image segmentation starting…", out
        assert started[-1]["channels"] == want, (backend, started[-1]["channels"])
    with xe._imgseg_lock:
        xe._imgseg_state.update({"status": "idle", "message": "", "result": None})
    # a programmatic call with two channels and the nuclei model: the nuclear channel alone
    chans = xe._iseg_channels()
    if len(chans) >= 2:
        seen = {}

        def stop(backend, diameter_px, n_channels=1, **kw):
            seen["n"] = n_channels
            raise RuntimeError("stop here")

        monkeypatch.setattr(xe, "_iseg_make_method", stop)
        nuc = xe._iseg_default_channel(chans)
        other = next(c for c in chans if c != nuc)
        for backend, want in (("cellpose:nuclei", 1), ("cellpose:cyto3", 2)):
            xe._run_imgseg(backend=backend, channels=[other, nuc])
            assert seen["n"] == want, backend
        with xe._imgseg_lock:
            xe._imgseg_state.update({"status": "idle", "message": "", "result": None})


def test_sidebar_status_reports_the_run_of_the_active_segmentation(xe):
    keep = {}
    for tool, lock, state in xe._seg_states():
        with lock:
            keep[tool] = dict(state)
    try:
        with xe._baysor_lock:
            xe._baysor_state.update({"status": "done", "message": "Loaded - 180,000 cells"})
        with xe._imgseg_lock:
            xe._imgseg_state.update({"status": "done", "message": "Done — 150 cells"})
        assert xe.update_reseg_status(1, 2, "imgseg:abc") == "✓ Image segmentation: Done — 150 cells"
        assert xe.update_reseg_status(1, 2, "baysor:def") == "✓ Baysor: Loaded - 180,000 cells"
        assert xe.update_reseg_status(1, 2, "xenium") == "✓ Baysor: Loaded - 180,000 cells"
        assert xe.update_reseg_status(1, 2, "proseg:zzz") == "✓ Baysor: Loaded - 180,000 cells"   # no Proseg run: the first loaded
        with xe._baysor_lock:
            xe._baysor_state.update({"status": "idle", "message": ""})
        assert xe.update_reseg_status(1, 2, None) == "✓ Image segmentation: Done — 150 cells"
        with xe._imgseg_lock:
            xe._imgseg_state.update({"status": "idle", "message": ""})
        assert xe.update_reseg_status(1, 2, "xenium") == ""
    finally:
        for tool, lock, state in xe._seg_states():
            with lock:
                state.clear()
                state.update(keep[tool])


def test_whole_tissue_run_counts_the_transcripts_of_the_segmented_extent(xe, prepared_patches, monkeypatch):
    """Without a region the transcript denominator is the extent that was segmented (the
    image patches), not the bounding box of the cells that happened to be found."""
    seen = {}

    def record(polys, tx_path, qv_min=20, tile_um=None, bounds=None, progress=None):
        import shapely
        seen["bounds"] = bounds
        seen["cells"] = shapely.total_bounds(np.asarray(list(polys), dtype=object))
        raise RuntimeError("stop after the bounds are known")

    monkeypatch.setattr(xe, "_iseg_assign_transcripts", record)
    with xe._imgseg_lock:
        xe._imgseg_state.update({"status": "running", "message": "", "result": None})
    try:
        xe._run_imgseg(backend="classical", diameter_um=8, expand_um=5, min_tx=1, patch_um=200)
    finally:
        with xe._imgseg_lock:
            xe._imgseg_state.update({"status": "idle", "message": "", "result": None})
    assert seen.get("bounds") is not None, "a whole-tissue run passed no extent: the cells' bounding box was used"
    px = float(xe.PIXEL_SIZE_UM)
    image = xe._iseg_image()[1]
    w, h = int(image.sizes["x"]) * px, int(image.sizes["y"]) * px
    x0, y0, x1, y1 = seen["bounds"]
    cx0, cy0, cx1, cy1 = seen["cells"]
    assert 0 <= x0 < x1 <= w + 1e-6 and 0 <= y0 < y1 <= h + 1e-6        # inside the image (the patches over the tissue)
    assert x0 <= cx0 + 1e-6 and y0 <= cy0 + 1e-6 and x1 >= cx1 - 1e-6 and y1 >= cy1 - 1e-6   # and around every cell
    assert (x1 - x0) * (y1 - y0) >= (cx1 - cx0) * (cy1 - cy0)


@pytest.mark.tools
def test_cellpose_nuclei_model_runs_on_the_nuclear_channel_when_a_membrane_channel_is_picked(xe, prepared_patches):
    st = xe._iseg_cellpose_status()
    if not st["available"] or "nuclei" not in st["models"]:
        pytest.skip("cellpose with local nuclei weights is not available: " + st["message"][:80])
    pytest.importorskip("sopa")
    chans = xe._iseg_channels()
    nuc = xe._iseg_default_channel(chans)
    others = [c for c in chans if c != nuc]
    if not others:
        pytest.skip("the image has one channel")
    import shutil

    def go(cyto):
        with xe._imgseg_lock:
            xe._imgseg_state.update({"status": "idle", "message": "", "result": None})
        out = xe.run_imgseg_modal(1, "imgseg", 1, "cellpose:nuclei", nuc, cyto, None, None, None, None,
                                  8, 5, 1, 300, 1.0, 0.4, 0.0, [])
        assert out == ("Image segmentation starting…", False, False), out
        tl.wait_state(xe._imgseg_state, xe._imgseg_lock, timeout=3600)
        res = dict(xe._imgseg_state["result"])
        with open(os.path.join(res["out_dir"], "params.json")) as f:
            p = json.load(f)
        snap = (p["channels"], p["model"], len(res["cells_df"]), int(res["cells_df"]["transcript_counts"].sum()),
                sorted(res["cells_df"]["x_centroid"].round(2)))
        shutil.rmtree(res["out_dir"], ignore_errors=True)
        return snap

    alone = go(None)
    with_membrane = go(others[0])
    assert alone[0] == [nuc] and with_membrane[0] == [nuc] and with_membrane[1] == "nuclei"
    assert with_membrane[2:] == alone[2:] and alone[2] >= 3      # the same cells: the membrane stain was not segmented
