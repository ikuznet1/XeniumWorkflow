"""Fixtures for the Xenium Explorer test-suite. See tests/README.md for the design.

One pytest process == one bundle. xenium_explorer.py loads its dataset while it is
being imported (from DATA_DIR) and fixes its cache root at the same moment (from
XENIUM_CACHE_DIR), so the ``xe`` fixture sets both, imports the module once per session
and every in-process test shares that import. Another bundle is another invocation
(``pytest --bundle PATH``); tests that need other import-time settings (REPL enabled,
WebGL renderer, a second dataset) start their own Python process.
"""
from __future__ import annotations

import os
import pathlib
import sys
import threading
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import xe_testlib as tl  # noqa: E402

MARKERS = {
    "slow": "larger bundles (located by env var); minutes",
    "tools": "real Baysor / Proseg / R runs on a very small region; skipped when the tool is missing",
    "browser": "node + Playwright tests of the WebGL viewer; skipped when unavailable",
}


def pytest_addoption(parser):
    parser.addoption("--bundle", default=os.environ.get("XENIUM_TEST_BUNDLE") or None,
                     help="Xenium output bundle to test against (default: the tiny bundle in "
                          "tests/data, or $XENIUM_TEST_BUNDLE)")


def pytest_configure(config):
    for name, text in MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {text}")


def pytest_report_header(config):
    return f"xenium bundle: {_source_bundle(config)}"


def _source_bundle(config) -> pathlib.Path:
    opt = config.getoption("--bundle")
    return pathlib.Path(opt).expanduser().resolve() if opt else tl.TINY_BUNDLE


# ── the bundle under test ────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def source_bundle(request) -> pathlib.Path:
    """The bundle as it sits on disk. Read-only: the session fails if it changes."""
    path = _source_bundle(request.config)
    if not (path / "experiment.xenium").exists():
        pytest.skip(f"no Xenium bundle at {path}")
    before = tl.bundle_snapshot(path)
    yield path
    after = tl.bundle_snapshot(path)
    changed = sorted(set(before.items()) ^ set(after.items()))
    assert not changed, f"the test run modified the source bundle {path}: {changed[:10]}"


@pytest.fixture(scope="session")
def raw(source_bundle) -> tl.RawBundle:
    """Independent readers over the bundle's raw files."""
    return tl.RawBundle(source_bundle)


@pytest.fixture(scope="session")
def is_tiny(source_bundle) -> bool:
    return source_bundle == tl.TINY_BUNDLE


@pytest.fixture(scope="session")
def session_dir(tmp_path_factory) -> pathlib.Path:
    return tmp_path_factory.mktemp("xenium")


@pytest.fixture(scope="session")
def cache_dir(session_dir) -> pathlib.Path:
    """The app's cache root for this session (never ~/.xenium_explorer_cache)."""
    d = session_dir / "cache"
    d.mkdir()
    return d


@pytest.fixture(scope="session")
def bundle_dir(source_bundle, session_dir) -> pathlib.Path:
    """Writable stand-in for the bundle (copy, or symlinks for a large bundle)."""
    return tl.make_working_copy(source_bundle, session_dir / source_bundle.name)


# ── the app, imported once ───────────────────────────────────────────────────
@pytest.fixture(scope="session")
def xe(bundle_dir, cache_dir):
    """xenium_explorer imported against the working copy of the bundle."""
    assert "xenium_explorer" not in sys.modules, "the app was imported outside the xe fixture"
    saved_env = {k: os.environ.get(k) for k in
                 ("DATA_DIR", "XENIUM_CACHE_DIR", "XENIUM_ENABLE_REPL", "XENIUM_RENDERER")}
    os.environ["DATA_DIR"] = str(bundle_dir)
    os.environ["XENIUM_CACHE_DIR"] = str(cache_dir)
    os.environ.pop("XENIUM_ENABLE_REPL", None)
    os.environ.pop("XENIUM_RENDERER", None)
    saved_streams = sys.stdout, sys.stderr
    sys.path.insert(0, str(tl.REPO))
    import xenium_explorer as mod
    # The module replaces sys.stdout / sys.stderr with a tee into its log buffer while it
    # is imported; they are put back at the end of the session. (Under pytest's capture
    # the app's prints go to the captured output, not to _log_buffer: use capsys.)
    assert mod.CACHE_BASE_DIR == str(cache_dir)
    assert os.path.realpath(mod.DATA["data_dir"]) == os.path.realpath(bundle_dir)
    mod._test_baseline = _Baseline(mod)
    yield mod
    sys.stdout, sys.stderr = saved_streams
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="session")
def client(xe):
    return xe.app.server.test_client()


@pytest.fixture(scope="session")
def D(xe):
    return xe.DATA


class _Baseline:
    """What the freshly imported app looks like; reset() puts it back after each test."""

    _STATES = ("_annot_state", "_baysor_state", "_proseg_state", "_imgseg_state", "_spatial_state",
               "_niche_state", "_save_sdata_state", "_split_state", "_spage_state",
               "_umap_reseg_state")

    def __init__(self, mod):
        self.mod = mod
        self.states = {n: dict(getattr(mod, n)) for n in self._STATES if hasattr(mod, n)}
        self.columns = list(mod.DATA["df"].columns)
        self.data_keys = set(mod.DATA)
        self.corrected = mod.DATA.get("split_corrected_expr")
        self.threads = set(threading.enumerate())

    def settle(self, timeout=60.0):
        """Wait for the app's background jobs started by a test (zarr writers, analysis
        threads). The app starts them as threading.Thread(target=<module function>), which
        Python names "Thread-N (<function>)"; library pools (zarr_io, asyncio_*) are
        long-lived and not waited for."""
        import re
        deadline = time.time() + timeout
        # Jobs started through the app's job registry (_job_start) are found there,
        # with their threads; the name pattern below covers the remaining helpers.
        for job in self.mod._jobs_list():
            t = job.thread
            if t is None or t is threading.current_thread():
                continue
            t.join(max(0.0, deadline - time.time()))
            assert not t.is_alive(), f"job {job.id} ({job.label}) still {job.state} after {timeout:.0f}s"
        for t in threading.enumerate():
            if t.name.startswith("xe-job-kill-"):
                t.join(max(0.0, deadline - time.time()))
        for t in threading.enumerate():
            m = re.match(r"Thread-\d+ \((\w+)\)$", t.name)
            # "run" is contextvars.Context.run, used for jobs that call into rpy2
            if (t in self.threads or t is threading.current_thread() or not m
                    or not (m.group(1) == "run" or hasattr(self.mod, m.group(1)))):
                continue
            t.join(max(0.0, deadline - time.time()))
            assert not t.is_alive(), f"background job {t.name} still running after {timeout:.0f}s"

    def reset(self):
        mod = self.mod
        self.settle()
        with mod._jobs_lock:
            mod._jobs.clear()
        if "_df_original" in mod.DATA:
            mod.unsubset()
        for name, snap in self.states.items():
            state = getattr(mod, name)
            state.clear()
            state.update(snap)
        with mod._roi_lock:
            mod._roi_state.update({"rois": [], "pending_hull": None, "show": True})
        try:
            os.remove(mod._roi_cache_path())
        except OSError:
            pass
        df = mod.DATA["df"]
        extra = [c for c in df.columns if c not in self.columns]
        if extra:
            df.drop(columns=extra, inplace=True)
        for k in [k for k in mod.DATA if k not in self.data_keys]:
            mod.DATA.pop(k, None)
        mod.DATA["split_corrected_expr"] = self.corrected
        del mod.EXTRA_DATASETS[:]
        for cache in ("_gene_expr_cache", "_umap_fig_cache", "_umap_df_cache", "_viewport_arrays"):
            getattr(mod, cache).clear()
        provider = getattr(mod, "_viewer_provider", None)
        if provider is not None:
            provider.forget()


@pytest.fixture(autouse=True)
def _isolate_app_state():
    """Undo whatever a test did to the app's global state (subset, labels, ROIs, ...)."""
    yield
    mod = sys.modules.get("xenium_explorer")
    baseline = getattr(mod, "_test_baseline", None)
    if baseline is not None:
        baseline.reset()


# ── small conveniences used by several modules ───────────────────────────────
@pytest.fixture
def labels(xe, D):
    """Deterministic synthetic cell-type labels installed as the CellTypist result."""
    import numpy as np
    import pandas as pd
    ids = D["df"].index.astype(str)
    rng = np.random.default_rng(7)
    lab = pd.Series(rng.choice(["Alpha", "Beta", "Gamma"], size=len(ids)), index=ids)
    lab.iloc[:3] = ["Alpha", "Beta", "Gamma"]      # every type present even in 3 cells
    with xe._annot_lock:
        xe._annot_state["labels_celltypist"] = lab
    return lab


@pytest.fixture
def expressed_gene(D):
    """The panel gene with the highest total count (so its values are not all zero)."""
    import numpy as np
    return D["gene_names"][int(np.asarray(D["expr"].sum(axis=0)).ravel().argmax())]


@pytest.fixture(scope="session")
def prepared_patches(xe):
    """SpatialData + tissue ROI + transcript patches, built once through the same
    callbacks the "Prepare Patches" button uses. Returns (width, min_tx, bounds in µm)."""
    pytest.importorskip("sopa")
    import numpy as np
    df = xe.DATA["df"]
    extent = max(float(np.ptp(df["x_centroid"])), float(np.ptp(df["y_centroid"])))
    width = 200.0 if extent < 2000 else 1000.0
    min_tx = 10
    xe.start_prepare_patches(1, 20)
    deadline = time.time() + 3600
    while True:
        xe.poll_sdata(0, 0, width, min_tx)          # the dcc.Interval tick: chains ROI and patches
        with xe._sdata_lock:
            status, message = xe._sdata_state["status"], xe._sdata_state["message"]
            ready = xe._sdata_state.get("patches") is not None
        if status == "error":
            pytest.fail(f"Prepare Patches failed: {message}")
        if ready and status != "running":
            break
        assert time.time() < deadline, f"Prepare Patches still {status!r}: {message}"
        time.sleep(0.2)
    xe._test_baseline.settle()
    bounds = xe._get_patch_bounds_um()
    assert bounds, "no patches"
    return width, min_tx, [tuple(float(v) for v in b) for b in bounds]
