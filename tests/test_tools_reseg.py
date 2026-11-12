"""Real Baysor and Proseg runs on a very small region (``-m tools``).

Each tool is run once per module through the same callback the "Run" button uses
(run_reseg_modal), in a region of about 45 x 45 µm around the densest group of cells.
The checks do not trust the app's own bookkeeping: what was handed to the tool is
compared with an independent read of transcripts.parquet, and the expression matrix
with a recount of the tool's own per-transcript output.

Skipped when the binary is missing. Expect about a minute per tool run on the tiny
bundle (Julia start-up dominates Baysor); a real 5K bundle takes several minutes.
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import re

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import xe_testlib as tl

pytestmark = pytest.mark.tools

RUN_TIMEOUT = 2 * 3600
UNASSIGNED = 4294967295
PATCH_STRIDE = 10_000_000


# ── region and truth ─────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def region(raw):
    """(x0, x1, y0, y1): a 45 µm box centred on the cell with most neighbours within 25 µm."""
    from scipy.spatial import cKDTree
    xy = raw.cells[["x_centroid", "y_centroid"]].to_numpy(float)
    tree = cKDTree(xy)
    counts = np.array([len(n) for n in tree.query_ball_point(xy, 25.0)])
    members = tree.query_ball_point(xy[int(counts.argmax())], 25.0)
    cx, cy = xy[members].mean(axis=0)
    return (round(float(cx) - 22.5, 1), round(float(cx) + 22.5, 1),
            round(float(cy) - 22.5, 1), round(float(cy) + 22.5, 1))


@pytest.fixture(scope="module")
def truth(raw, region):
    """Independent read of the region: all transcripts, and the Q>=20 gene ones."""
    x0, x1, y0, y1 = region
    tx = raw.transcripts(bbox=(x0, y0, x1, y1))
    genes = tl.gene_tx(tx)
    assert len(genes) >= 50, "region too empty for a segmentation test"
    return tx, genes


def _wait(xe, tool):
    state, lock = ((xe._baysor_state, xe._baysor_lock) if tool == "baysor"
                   else (xe._proseg_state, xe._proseg_lock))
    tl.wait_state(state, lock, timeout=RUN_TIMEOUT, poll=0.5)
    xe._test_baseline.settle(timeout=600)
    res = state["result"]
    assert res is not None
    return res


def _activate(xe, tool, res):
    state, lock = ((xe._baysor_state, xe._baysor_lock) if tool == "baysor"
                   else (xe._proseg_state, xe._proseg_lock))
    with lock:
        state.update({"status": "done", "message": "Done", "result": res, "loading_out_dir": None})


def _tag(res):
    return os.path.basename(res["out_dir"]).split("_")[-1]


def _snapshot(res):
    cdf = res["cells_df"]
    return {"ids": list(cdf.index.astype(str)),
            "x": cdf["x_centroid"].to_numpy(float).copy(), "y": cdf["y_centroid"].to_numpy(float).copy(),
            "tc": cdf["transcript_counts"].to_numpy(float).copy(),
            "expr": sp.csr_matrix(res["expr"]).copy(),
            "bounds": sorted(str(k) for k in res["cell_bounds"]),
            "cluster": cdf["cluster_kmeans_10"].astype(str).to_numpy().copy(),
            "umap": cdf["umap_1"].to_numpy(float).copy() if "umap_1" in cdf else None}


def _reload(xe, tool, tag, force_slow=False):
    """Select the cached run through the dropdown callback; optionally disable the fast loader."""
    state, lock = ((xe._baysor_state, xe._baysor_lock) if tool == "baysor"
                   else (xe._proseg_state, xe._proseg_lock))
    with lock:
        state.update({"status": "idle", "message": "", "result": None, "loading_out_dir": None})
    assert f"{tool}:{tag}" in [r["value"] for r in xe._list_cached_seg_runs()]
    orig = xe._fast_load_reseg_zarr
    if force_slow:
        def refuse(_path):
            raise RuntimeError("fast path disabled by the test")
        xe._fast_load_reseg_zarr = refuse
    try:
        xe.load_seg_run_from_cache(f"{tool}:{tag}", 0, 0)
        return _wait(xe, tool)
    finally:
        xe._fast_load_reseg_zarr = orig


def _assert_same_run(live, res, strip_patch_prefix=False):
    """A reloaded result equals the live one, cell by cell (matched by id)."""
    got = _snapshot(res)
    strip = (lambda i: re.sub(r"^p\d+_", "", i)) if strip_patch_prefix else (lambda i: i)
    live_ids, got_ids = pd.Index([strip(i) for i in live["ids"]]), pd.Index([strip(i) for i in got["ids"]])
    assert set(live_ids) == set(got_ids) and len(live_ids) == len(got_ids)
    pos = got_ids.get_indexer(live_ids)
    np.testing.assert_allclose(got["x"][pos], live["x"], atol=1e-3)
    np.testing.assert_allclose(got["y"][pos], live["y"], atol=1e-3)
    np.testing.assert_array_equal(got["tc"][pos], live["tc"])
    assert tl.sparse_equal(got["expr"][pos], live["expr"])
    assert {strip(i) for i in got["bounds"]} == {strip(i) for i in live["bounds"]}
    assert (got["cluster"][pos] == live["cluster"]).all(), "clusters are restored, not recomputed"
    if live["umap"] is not None and got["umap"] is not None:
        np.testing.assert_allclose(got["umap"][pos], live["umap"], atol=1e-4, equal_nan=True)


def _assert_result_consistent(xe, res, tool, region, cache_dir):
    """cells_df, boundaries, matrix, zarr store and params.json describe the same cells."""
    from shapely.geometry import Point, Polygon
    cdf, bounds, expr = res["cells_df"], res["cell_bounds"], res["expr"]
    ids = pd.Index(cdf.index.astype(str))
    assert ids.is_unique and len(ids) > 0
    assert expr.shape == (len(cdf), len(xe._panel_gene_names()))
    np.testing.assert_array_equal(np.asarray(expr.sum(axis=1)).ravel(), cdf["transcript_counts"].to_numpy(float))
    assert {str(k) for k in bounds} <= set(ids)
    x0, x1, y0, y1 = region
    if tool == "baysor":
        # Baysor cells are hulls of their molecules. (Proseg grows cells through empty
        # space around a small island of transcripts, so its centroids may lie outside.)
        assert cdf["x_centroid"].between(x0, x1).all() and cdf["y_centroid"].between(y0, y1).all()
    for key, (vx, vy) in bounds.items():
        vx, vy = np.asarray(vx, float), np.asarray(vy, float)
        ok = ~np.isnan(vx)
        if tool == "baysor":
            assert vx[ok].min() >= x0 - 15 and vx[ok].max() <= x1 + 15
            assert vy[ok].min() >= y0 - 15 and vy[ok].max() <= y1 + 15
        if ok.all() and ok.sum() >= 4:
            row = cdf.loc[cdf.index.astype(str) == str(key)].iloc[0]
            assert Polygon(np.column_stack([vx, vy])).buffer(0).buffer(3.0).contains(
                Point(row["x_centroid"], row["y_centroid"])), f"centroid of {key} is not at its polygon"
    # run directory: under the tmp cache, with a parameter record
    out_dir = res["out_dir"]
    assert os.path.realpath(out_dir).startswith(os.path.realpath(cache_dir) + os.sep)
    params = json.load(open(os.path.join(out_dir, "params.json")))
    assert params["tool"] == tool and params["n_cells"] == len(cdf) and params["param_tag"] == _tag(res)
    assert (params["x_min"], params["x_max"], params["y_min"], params["y_max"]) == region
    # zarr store written next to it, equal to what is in memory
    import spatialdata
    assert res["sdata_path"] == os.path.join(out_dir, f"spatialdata_{tool}.zarr")
    table = spatialdata.read_zarr(res["sdata_path"]).tables["table"]
    assert list(table.obs_names.astype(str)) == list(ids)
    assert list(table.var_names) == xe._panel_gene_names()
    assert tl.sparse_equal(sp.csr_matrix(table.X), expr)
    shapes = spatialdata.read_zarr(res["sdata_path"]).shapes
    assert "cell_boundaries" in shapes and set(shapes["cell_boundaries"].index.astype(str)) <= set(ids)


def _assert_shown(xe, client, tool, res, gene):
    """Figure and viewer draw the run's cells with the run's values."""
    cdf = res["cells_df"]
    kw = {"baysor_active": tool == "baysor", "proseg_active": tool == "proseg"}
    fig = tl.fig_cells(xe.make_spatial_fig("gene", None, gene, 3, 0.8, [], None, **kw))
    assert set(fig.index) == set(cdf.index.astype(str))
    col = np.asarray(res["expr"][:, xe.DATA["gene_name_to_idx"][gene]].todense()).ravel()
    want = pd.Series(np.log1p(col), index=cdf.index.astype(str))
    np.testing.assert_allclose(fig["color"].to_numpy(float), want.loc[fig.index].to_numpy(), rtol=1e-5)
    src = f"{tool}:{_tag(res)}"
    assert tool in client.get("/viewer/meta").get_json()["sources"]
    h, a = tl.get_bin(client, f"/viewer/cells.bin?source={src}")
    assert h["n"] == len(cdf)
    np.testing.assert_array_equal(a["xy"][0::2], cdf["x_centroid"].to_numpy(np.float32))
    _, c = tl.get_bin(client, f"/viewer/colors.bin?source={src}&mode=gene&gene={gene}")
    np.testing.assert_allclose(c["values"], np.log1p(col), rtol=1e-5)
    x0, y0, x1, y1 = cdf.x_centroid.min() - 30, cdf.y_centroid.min() - 30, cdf.x_centroid.max() + 30, cdf.y_centroid.max() + 30
    hb, ab = tl.get_bin(client, f"/viewer/boundaries.bin?source={src}&kind={tool}&x0={x0}&y0={y0}&x1={x1}&y1={y1}")
    assert hb["n"] >= len(res["cell_bounds"]) > 0 and int(ab["cell"].max()) < len(cdf)
    assert xe.poll_baysor(0, 0)[3] == src if tool == "baysor" else xe.poll_proseg(0, 0)[3] == src


# ── cancellation ─────────────────────────────────────────────────────────────
def _group_pids(pgid):
    """Live (non-zombie) processes of process group *pgid*."""
    import subprocess
    out = subprocess.run(["ps", "-A", "-o", "pid=,pgid=,stat="], capture_output=True, text=True).stdout
    rows = [ln.split() for ln in out.splitlines()]
    return [int(r[0]) for r in rows if len(r) >= 3 and int(r[1]) == pgid and not r[2].startswith("Z")]


def _cancel_midway(xe, tool, start, midway, cache_dir):
    """Start a real run through the Run callback, cancel it once *midway(job)* holds, and
    check what is left: no process, no run directory, a released state. Returns the job."""
    import time
    state, lock = ((xe._baysor_state, xe._baysor_lock) if tool == "baysor"
                   else (xe._proseg_state, xe._proseg_lock))
    listed_before = [r["value"] for r in xe._list_cached_seg_runs()]
    entries_before = sorted(os.listdir(cache_dir))
    gen, df = xe._dataset_gen, xe.DATA["df"]
    assert start()[0] == f"{tool.capitalize()} starting…"
    jobs = [j for j in xe._jobs_list(active_only=True) if j.kind == tool]
    assert len(jobs) == 1, "the run is in the job registry"
    job = jobs[0]
    assert tool.capitalize() in xe._running_jobs()
    deadline = time.time() + RUN_TIMEOUT
    while not (job.procs and midway(job)):
        assert job.active(), f"the run ended before it could be cancelled: {job.state} / {state['message']}"
        assert time.time() < deadline
        time.sleep(0.02)
    snap = job.snapshot()
    leader = job.procs[0].pid
    pids = _group_pids(leader)
    assert leader in pids, "the tool runs in its own process group"
    staged = [e for e in os.listdir(cache_dir) if e.endswith(xe._SEG_STAGE_SUFFIX)]
    assert len(staged) == 1 and staged[0].startswith(f"{tool}_")
    assert [r["value"] for r in xe._list_cached_seg_runs()] == listed_before, \
        "a run in progress is not offered as a cached run"

    t0 = time.time()
    assert xe._job_cancel(job.id)
    job.thread.join(120)
    took = time.time() - t0
    assert not job.thread.is_alive() and job.state == "cancelled", (job.state, state["message"])
    assert took < xe._JOB_KILL_GRACE_S + 15, f"cancel took {took:.1f}s"
    # the process tree is gone
    deadline = time.time() + 20
    while _group_pids(leader) and time.time() < deadline:
        time.sleep(0.1)
    assert _group_pids(leader) == [], f"processes of the cancelled {tool} run are still alive"
    # no run directory, nothing new in the cached-run list
    assert sorted(os.listdir(cache_dir)) == entries_before, "the cancelled run left files in the cache"
    assert [r["value"] for r in xe._list_cached_seg_runs()] == listed_before
    # state released (what the poll reports), dataset untouched
    with lock:
        assert (state["status"], state["message"], state["result"]) == ("error", "Cancelled", None)
    assert tool.capitalize() not in xe._running_jobs()
    assert xe._dataset_gen == gen and xe.DATA["df"] is df
    poll = xe.poll_baysor(0, 0, "xenium") if tool == "baysor" else xe.poll_proseg(0, 0, "xenium")
    assert poll[0] == "✗ Cancelled" and poll[1] is True
    return job, snap


# ── Baysor ───────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def baysor_run(xe, region):
    if not tl.baysor_bin():
        pytest.skip("Baysor not installed (~/.julia/bin/baysor or $BAYSOR_BIN)")
    x0, x1, y0, y1 = region
    out = xe.run_reseg_modal(1, "baysor", "skip", x0, x1, y0, y1, 6.0, None, 20, 4, [], ["yes"], 0.5,
                             "nucleus", None, None, None, None, None, None, None, None, None, None, None, None)
    assert out[0] == "Baysor starting…"
    res = _wait(xe, "baysor")
    return res, _snapshot(res)


@pytest.fixture
def baysor(xe, baysor_run):
    _activate(xe, "baysor", baysor_run[0])
    return baysor_run


def test_baysor_cancelled_midway_leaves_nothing_and_the_same_run_then_completes(xe, region, cache_dir, request):
    if not tl.baysor_bin():
        pytest.skip("Baysor not installed (~/.julia/bin/baysor or $BAYSOR_BIN)")
    x0, x1, y0, y1 = region
    start = lambda: xe.run_reseg_modal(
        1, "baysor", "skip", x0, x1, y0, y1, 6.0, None, 20, 4, [], ["yes"], 0.5,
        "nucleus", None, None, None, None, None, None, None, None, None, None, None, None)
    # mid-way: Julia is up and Baysor has logged its first stage
    job, snap = _cancel_midway(xe, "baysor", start, lambda j: j.step.startswith("Baysor: "), cache_dir)
    assert 0 < snap["fraction"] < 0.9 and snap["n_procs"] == 1, snap
    # the same request again, to completion (the module's Baysor run)
    res, _ = request.getfixturevalue("baysor_run")
    _activate(xe, "baysor", res)
    _assert_result_consistent(xe, res, "baysor", region, cache_dir)
    assert [r["value"] for r in xe._list_cached_seg_runs() if r["value"].startswith("baysor:")] == [f"baysor:{_tag(res)}"]
    assert not [e for e in os.listdir(cache_dir) if e.endswith(xe._SEG_STAGE_SUFFIX)]


def test_baysor_receives_exactly_the_gene_transcripts_of_the_region(baysor, truth, region, xe):
    res, _ = baysor
    _all, genes = truth
    sent = pd.read_csv(os.path.join(res["out_dir"], "transcripts.csv"))
    x0, x1, y0, y1 = region
    assert len(sent) == len(genes), "Q >= 20 gene transcripts of the region, nothing else"
    assert sent["x_location"].between(x0, x1).all() and sent["y_location"].between(y0, y1).all()
    assert set(sent["feature_name"]) <= set(xe._panel_gene_names())
    pd.testing.assert_series_equal(sent["feature_name"].value_counts().sort_index(),
                                   genes["feature_name"].value_counts().sort_index(),
                                   check_names=False)
    if "transcript_id" in sent.columns:
        assert set(sent["transcript_id"]) == set(genes["transcript_id"])
    # nucleus prior: transcripts over a nucleus keep their cell as prior, others get 0
    prior_col = next(c for c in sent.columns if "prior" in c)
    over_nucleus = genes[(genes["overlaps_nucleus"] == 1) & (genes["cell_id"] != "UNASSIGNED")]
    assert int((sent[prior_col] != 0).sum()) == len(over_nucleus)


def test_baysor_matrix_is_a_recount_of_its_segmentation(baysor, xe):
    res, _ = baysor
    seg = pd.read_csv(os.path.join(res["out_dir"], "segmentation.csv"))
    assigned = seg[seg["cell"].notna() & (seg["cell"].astype(str) != "")]
    cdf = res["cells_df"]
    assert set(cdf.index.astype(str)) == set(assigned["cell"].astype(str))
    recount = tl.count_matrix(assigned["cell"], assigned["gene"], cdf.index, xe._panel_gene_names())
    assert tl.sparse_equal(recount, res["expr"])
    assert res["expr"].sum() == len(assigned)


def test_baysor_result_is_consistent(xe, baysor, region, cache_dir):
    _assert_result_consistent(xe, baysor[0], "baysor", region, cache_dir)
    params = json.load(open(os.path.join(baysor[0]["out_dir"], "params.json")))
    assert (params["scale"], params["min_mol"], params["use_prior"], params["prior_type"]) == (6.0, 20, True, "nucleus")
    assert params["n_patches"] == 0 and params["qv_min"] == 20


def test_baysor_run_is_shown(xe, client, baysor, truth):
    gene = truth[1]["feature_name"].value_counts().index[0]
    _assert_shown(xe, client, "baysor", baysor[0], gene)


def test_baysor_reload_from_cache_reproduces_the_run(xe, baysor):
    res, live = baysor
    tag = _tag(res)
    _assert_same_run(live, _reload(xe, "baysor", tag))                       # fast path (zarr)
    _assert_same_run(live, _reload(xe, "baysor", tag, force_slow=True))      # slow path, zarr present
    zarr_path = res["sdata_path"]
    os.rename(zarr_path, zarr_path + ".hidden")                              # slow path, zarr rebuilt
    try:
        rebuilt = _reload(xe, "baysor", tag)
        np.testing.assert_array_equal(_snapshot(rebuilt)["tc"], live["tc"])
        assert tl.sparse_equal(_snapshot(rebuilt)["expr"], live["expr"])
        assert _snapshot(rebuilt)["ids"] == live["ids"]
    finally:
        import shutil
        shutil.rmtree(zarr_path, ignore_errors=True)
        os.rename(zarr_path + ".hidden", zarr_path)


def test_same_parameters_map_to_the_same_cache_entry(xe, baysor, region):
    """Run directories are keyed by the parameters: asking again must not create a second one."""
    res, _ = baysor
    before = sorted(os.listdir(xe.CACHE_BASE_DIR))
    runs = [r for r in xe._list_cached_seg_runs() if r["value"].startswith("baysor:")]
    assert [r["value"] for r in runs] == [f"baysor:{_tag(res)}"]
    assert "r=6.0µm" in runs[0]["label"] and f"{len(res['cells_df'])} cells" in runs[0]["label"]
    assert sorted(os.listdir(xe.CACHE_BASE_DIR)) == before


# ── Proseg ───────────────────────────────────────────────────────────────────
def _proseg_request(xe, x0, x1, y0, y1, patches):
    return xe.run_reseg_modal(1, "proseg", True if patches else "skip",
                              None, None, None, None, None, None, None, None, None, None, None, None,
                              x0, x1, y0, y1, None, 4, 10, None, None, None, None, "nucleus")


@pytest.fixture(scope="module")
def proseg_run(xe, region):
    if not tl.proseg_bin():
        pytest.skip("Proseg not installed (~/.cargo/bin/proseg or $PROSEG_BIN)")
    assert _proseg_request(xe, *region, patches=False)[0] == "Proseg starting…"
    res = _wait(xe, "proseg")
    return res, _snapshot(res)


@pytest.fixture
def proseg(xe, proseg_run):
    _activate(xe, "proseg", proseg_run[0])
    return proseg_run


def test_proseg_cancelled_midway_leaves_nothing_and_the_same_run_then_completes(xe, region, cache_dir, request):
    if not tl.proseg_bin():
        pytest.skip("Proseg not installed (~/.cargo/bin/proseg or $PROSEG_BIN)")
    # mid-way: the sampler has done between 10 % and 80 % of its iterations (progress
    # parsed from Proseg's own progress bar)
    job, snap = _cancel_midway(xe, "proseg", lambda: _proseg_request(xe, *region, patches=False),
                               lambda j: (j.fraction or 0) >= 0.1 * 0.9, cache_dir)
    assert 0.09 <= snap["fraction"] < 0.8 and "sampling" in snap["step"], snap
    # the same request again, to completion (the module's Proseg run)
    res, _ = request.getfixturevalue("proseg_run")
    _activate(xe, "proseg", res)
    # a complete run directory under its real name (the content of the result is the
    # subject of test_proseg_result_is_consistent)
    out_dir = res["out_dir"]
    assert os.path.dirname(os.path.realpath(out_dir)) == os.path.realpath(cache_dir)
    assert not out_dir.endswith(xe._SEG_STAGE_SUFFIX)
    params = json.load(open(os.path.join(out_dir, "params.json")))
    assert params["tool"] == "proseg" and params["n_cells"] == len(res["cells_df"]) > 0
    assert (params["x_min"], params["x_max"], params["y_min"], params["y_max"]) == region
    import spatialdata
    assert res["sdata_path"] == os.path.join(out_dir, "spatialdata_proseg.zarr")
    table = spatialdata.read_zarr(res["sdata_path"]).tables["table"]
    assert list(table.obs_names.astype(str)) == list(res["cells_df"].index.astype(str))
    assert tl.sparse_equal(sp.csr_matrix(table.X), res["expr"])
    assert f"proseg:{_tag(res)}" in [r["value"] for r in xe._list_cached_seg_runs()]
    assert not [e for e in os.listdir(cache_dir) if e.endswith(xe._SEG_STAGE_SUFFIX)]


def _read_proseg_dir(d):
    cells = pd.read_csv(os.path.join(d, "cell-metadata.csv.gz"))
    tx = pd.read_csv(os.path.join(d, "transcript-metadata.csv.gz"))
    with gzip.open(os.path.join(d, "cell-polygons.geojson.gz"), "rt") as f:
        polys = json.load(f)
    return cells, tx, polys


def test_proseg_receives_exactly_the_gene_transcripts_of_the_region(proseg, truth, region, xe):
    res, _ = proseg
    _all, genes = truth
    sent = pd.read_parquet(os.path.join(res["out_dir"], "transcripts.parquet"))
    assert set(sent["transcript_id"]) == set(genes["transcript_id"]) and len(sent) == len(genes)
    x0, x1, y0, y1 = region
    assert sent["x_location"].between(x0, x1).all() and sent["y_location"].between(y0, y1).all()
    _cells, tx, _ = _read_proseg_dir(res["out_dir"])
    assert len(tx) == len(genes), "Proseg read every transcript it was given"
    pd.testing.assert_series_equal(tx["gene"].astype(str).value_counts().sort_index(),
                                   genes["feature_name"].value_counts().sort_index(), check_names=False)


def test_proseg_matrix_is_a_recount_of_its_assignments(proseg, xe):
    res, _ = proseg
    cells, tx, polys = _read_proseg_dir(res["out_dir"])
    cdf = res["cells_df"]
    assert set(cdf.index.astype(str)) == set(cells["cell"].astype(str))
    assigned = tx[tx["assignment"] != UNASSIGNED]
    recount = tl.count_matrix(assigned["assignment"], assigned["gene"].astype(str), cdf.index,
                              xe._panel_gene_names())
    assert tl.sparse_equal(recount, res["expr"])
    by_id = cells.set_index(cells["cell"].astype(str))
    np.testing.assert_allclose(cdf["x_centroid"].to_numpy(float),
                               by_id.loc[cdf.index.astype(str), "centroid_x"].to_numpy(float), atol=1e-3)
    assert {str(f["properties"]["cell"]) for f in polys["features"]} >= {str(k) for k in res["cell_bounds"]}


def test_proseg_result_is_consistent(xe, proseg, region, cache_dir):
    _assert_result_consistent(xe, proseg[0], "proseg", region, cache_dir)
    params = json.load(open(os.path.join(proseg[0]["out_dir"], "params.json")))
    assert params["prior_type"] == "nucleus" and params["n_samples"] == 10 and params["n_patches"] == 0


def test_proseg_run_is_shown(xe, client, proseg, truth):
    gene = truth[1]["feature_name"].value_counts().index[0]
    _assert_shown(xe, client, "proseg", proseg[0], gene)


def test_proseg_reload_from_cache_reproduces_the_run(xe, proseg):
    res, live = proseg
    tag = _tag(res)
    _assert_same_run(live, _reload(xe, "proseg", tag))
    _assert_same_run(live, _reload(xe, "proseg", tag, force_slow=True))


def test_subset_and_export_of_a_reseg_run(xe, proseg, tmp_path):
    """The run goes through the same downstream paths as the Xenium cells."""
    import spatialdata
    res, live = proseg
    src = f"proseg:{_tag(res)}"
    keep_n = int((res["cells_df"]["transcript_counts"] >= np.median(live["tc"])).sum())
    xe.subset(min_transcripts=float(np.median(live["tc"])))
    assert len(res["cells_df"]) == keep_n == res["expr"].shape[0]
    xe.unsubset()
    assert _snapshot(res)["ids"] == live["ids"] and tl.sparse_equal(res["expr"], live["expr"])
    xe.start_save_sdata(1, str(tmp_path), "reseg", [], src)
    tl.wait_state(xe._save_sdata_state, xe._save_sdata_lock, timeout=600)
    table = spatialdata.read_zarr(str(tmp_path / "reseg.zarr")).tables["table"]
    assert list(table.obs_names.astype(str)) == live["ids"]
    assert tl.sparse_equal(sp.csr_matrix(table.X)[:, : live["expr"].shape[1]], live["expr"])


# ── Proseg on patches ────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def proseg_patch_run(xe, prepared_patches, raw):
    if not tl.proseg_bin():
        pytest.skip("Proseg not installed")
    _w, _m, bounds = prepared_patches
    if len(bounds) < 2 or len(bounds) > 4:
        pytest.skip(f"patch test wants 2-4 patches, this bundle gives {len(bounds)}")
    if raw.n_transcripts > 2_000_000:
        pytest.skip("patch run is for the tiny bundle")
    assert _proseg_request(xe, None, None, None, None, patches=True)[0] == "Proseg starting…"
    res = _wait(xe, "proseg")
    return res, _snapshot(res), bounds


def test_proseg_patch_run_merges_patches_without_duplicates(xe, proseg_patch_run, raw):
    res, live, bounds = proseg_patch_run
    out_dir = res["out_dir"]
    params = json.load(open(os.path.join(out_dir, "params.json")))
    dirs = sorted(glob.glob(os.path.join(out_dir, "patch_*")))
    assert params["n_patches"] == len(bounds) and len(dirs) >= 1
    assert [tuple(b) for b in params["patch_bounds"]] == [pytest.approx(b) for b in bounds]
    cdf = res["cells_df"]
    ids = pd.Index(cdf.index.astype("int64"))
    assert ids.is_unique
    parts, sent_ids = [], []
    for d in dirs:
        pi = int(os.path.basename(d).split("_")[1])
        cells, tx, _ = _read_proseg_dir(d)
        sent = pd.read_parquet(os.path.join(d, "transcripts.parquet"), columns=["transcript_id", "x_location", "y_location"])
        b = bounds[pi]
        assert sent["x_location"].between(b[0] - 1e-3, b[2] + 1e-3).all() and \
            sent["y_location"].between(b[1] - 1e-3, b[3] + 1e-3).all(), "a patch gets its own transcripts"
        sent_ids.append(sent["transcript_id"])
        a = tx[tx["assignment"] != UNASSIGNED].copy()
        a["global"] = a["assignment"].astype("int64") + 1 + pi * PATCH_STRIDE      # documented id scheme
        parts.append(a[["global", "gene"]])
        mine = ids[(ids // PATCH_STRIDE) == pi]
        assert set(mine) <= set(cells["cell"].astype("int64") + 1 + pi * PATCH_STRIDE)
        # every kept cell lies in the patch that produced it
        own = cdf.loc[cdf.index.astype("int64").isin(mine)]
        assert own["x_centroid"].between(b[0], b[2]).all() and own["y_centroid"].between(b[1], b[3]).all()
    pairs = pd.concat(parts)
    pairs = pairs[pairs["global"].isin(ids)]
    recount = tl.count_matrix(pairs["global"], pairs["gene"].astype(str), ids, xe._panel_gene_names())
    assert tl.sparse_equal(recount, res["expr"])
    # only Q>=20 gene transcripts were handed out
    truth = tl.gene_tx(raw.transcripts(columns=["transcript_id", "qv", "is_gene", "codeword_category", "feature_name"]))
    assert set(pd.concat(sent_ids)) <= set(truth["transcript_id"])
    # no cell twice: no two kept cells of different patches at (nearly) the same place
    from scipy.spatial import cKDTree
    xy = cdf[["x_centroid", "y_centroid"]].to_numpy(float)
    patch_of = (ids // PATCH_STRIDE).to_numpy()
    close = cKDTree(xy).query_pairs(1.5, output_type="ndarray")
    assert not len(close) or (patch_of[close[:, 0]] == patch_of[close[:, 1]]).all()


def test_proseg_patch_run_reloads_identically(xe, proseg_patch_run):
    res, live, _ = proseg_patch_run
    tag = _tag(res)
    _assert_same_run(live, _reload(xe, "proseg", tag))
    _assert_same_run(live, _reload(xe, "proseg", tag, force_slow=True))
