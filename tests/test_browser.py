"""Browser tier (``-m browser``): the existing node/Playwright tests of the WebGL viewer.

The JavaScript tests live in viewer_src/test and are run as they are; these wrappers
start what they need and turn their exit code and PASS/FAIL lines into a pytest result:

  * app_smoke.mjs / glue_checks.mjs drive the real Dash app, started here on a free
    port with XENIUM_RENDERER=webgl against a private working copy of a bundle and a
    temporary cache directory;
  * run.mjs tests the viewer bundle against viewer_src/dev_server.py, which it starts
    itself on ports 8072 / 8073 (it needs the large skin bundle as well as the tiny one).

Skipped when node, the Playwright package or its Chromium are missing. On macOS the
scripts launch headless Chromium with --use-angle=metal --ignore-gpu-blocklist
themselves (XV_GL=swiftshader selects software WebGL instead).
"""
from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

import xe_testlib as tl

pytestmark = pytest.mark.browser

VIEWER_SRC = tl.REPO / "viewer_src"
SKIN_ENV = "XENIUM_TEST_BUNDLE_SKIN"


@pytest.fixture(scope="session")
def node():
    exe = shutil.which("node")
    if not exe:
        pytest.skip("node is not installed")
    if not (VIEWER_SRC / "node_modules" / "playwright").is_dir():
        pytest.skip("Playwright is not installed (run `npm install` in viewer_src)")
    probe = ("import {chromium} from 'playwright'; const b = await chromium.launch(); "
             "await b.close(); console.log('ok');")
    try:
        r = subprocess.run([exe, "--input-type=module", "-e", probe], cwd=VIEWER_SRC,
                           capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        pytest.skip("headless Chromium did not start within 120 s")
    if r.returncode != 0 or "ok" not in r.stdout:
        pytest.skip(f"Playwright cannot launch Chromium: {r.stderr.strip()[-300:]}")
    return exe


def _start_app(bundle, cache, log_path, env_extra=None):
    port = tl.free_port()
    env = {k: v for k, v in os.environ.items() if k not in ("XENIUM_ENABLE_REPL", "HOST", "XENIUM_HOST")}
    env.update({"DATA_DIR": str(bundle), "XENIUM_CACHE_DIR": str(cache), "PORT": str(port),
                "XENIUM_RENDERER": "webgl", "PYTHONWARNINGS": "ignore"})
    env.update(env_extra or {})
    log = open(log_path, "w")
    proc = subprocess.Popen([sys.executable, str(tl.APP_SOURCE)], env=env, cwd=str(cache),
                            stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 1800
    while True:
        if proc.poll() is not None:
            log.close()
            raise AssertionError(f"app exited with {proc.returncode}:\n{open(log_path).read()[-3000:]}")
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as r:
                if r.status == 200:
                    break
        except OSError:
            pass
        if time.time() > deadline:
            proc.kill()
            raise AssertionError("app did not answer /health within 30 min")
        time.sleep(0.5)
    return proc, base, log


def _stop(proc, log):
    proc.terminate()
    try:
        proc.wait(20)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()


@pytest.fixture(scope="module")
def tiny_app(node, tmp_path_factory):
    """The app in WebGL mode on a copy of the tiny bundle; yields its base URL."""
    root = tmp_path_factory.mktemp("browser_tiny")
    (root / "cache").mkdir()
    before = tl.bundle_snapshot(tl.TINY_BUNDLE)
    bundle = tl.make_working_copy(tl.TINY_BUNDLE, root / tl.TINY_BUNDLE.name)
    proc, base, log = _start_app(bundle, root / "cache", root / "app.log")
    yield base
    _stop(proc, log)
    assert tl.bundle_snapshot(tl.TINY_BUNDLE) == before, "tests/data was modified"


@pytest.fixture(scope="module")
def skin_bundle():
    path = os.environ.get(SKIN_ENV)
    if not path or not os.path.exists(os.path.join(path, "experiment.xenium")):
        pytest.skip(f"set {SKIN_ENV} to the Xenium Prime human skin bundle to run this test")
    return path


@pytest.fixture(scope="module")
def skin_app(node, skin_bundle, tmp_path_factory):
    root = tmp_path_factory.mktemp("browser_skin")
    (root / "cache").mkdir()
    before = tl.bundle_snapshot(skin_bundle)
    bundle = tl.make_working_copy(skin_bundle, root / os.path.basename(skin_bundle.rstrip("/")))
    proc, base, log = _start_app(bundle, root / "cache", root / "app.log")
    yield base
    _stop(proc, log)
    assert tl.bundle_snapshot(skin_bundle) == before, "the skin bundle was modified"


def _run_node(node, script, args, timeout, env=None):
    proc = subprocess.run([node, str(VIEWER_SRC / "test" / script), *map(str, args)], cwd=VIEWER_SRC,
                          capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, **(env or {})})
    out = proc.stdout + proc.stderr
    failed = [ln for ln in out.splitlines() if re.match(r"\s*(FAIL|✗|not ok)\b", ln)]
    passed = [ln for ln in out.splitlines() if re.match(r"\s*(PASS|✓|ok)\b", ln)]
    assert proc.returncode == 0 and not failed, \
        f"{script}: exit {proc.returncode}, {len(failed)} failed checks\n" + "\n".join(failed[:40]) + \
        f"\n--- tail ---\n{out[-3000:]}"
    assert passed, f"{script} reported no checks:\n{out[-2000:]}"
    return len(passed)


def test_app_serves_the_webgl_view(tiny_app):
    """Before any browser: the page the scripts will open is the WebGL configuration."""
    with urllib.request.urlopen(tiny_app + "/viewer/meta", timeout=30) as r:
        assert r.status == 200
    with urllib.request.urlopen(tiny_app + "/_dash-layout", timeout=60) as r:
        layout = r.read().decode()
    assert "webgl-view" in layout
    with urllib.request.urlopen(tiny_app + "/assets/xenium_viewer.js", timeout=60) as r:
        assert r.status == 200 and len(r.read()) > 100_000, "viewer bundle is built and served"


def test_app_smoke_on_the_skin_bundle(node, skin_app, tmp_path):
    """app_smoke.mjs uses human gene names and zoom levels of the skin dataset."""
    n = _run_node(node, "app_smoke.mjs", [skin_app, tmp_path], timeout=1800)
    assert n >= 10


def test_glue_checks_on_the_skin_bundle(node, skin_app, tmp_path):
    n = _run_node(node, "glue_checks.mjs", [skin_app, tmp_path], timeout=1800)
    assert n >= 10


def _port_free(port):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def test_viewer_bundle_suite(node, skin_bundle):
    """viewer_src/test/run.mjs: the viewer bundle against dev_server.py on both bundles
    (read-only use of the bundles; the script starts and stops its own servers)."""
    for port in (8072, 8073):
        if not _port_free(port):
            pytest.skip(f"port {port} (hard-coded in run.mjs) is in use")
    before = tl.bundle_snapshot(tl.TINY_BUNDLE)
    n = _run_node(node, "run.mjs", [], timeout=3600,
                  env={"XV_SKIN": skin_bundle, "XV_TINY": str(tl.TINY_BUNDLE), "XV_PYTHON": sys.executable})
    assert n >= 40
    assert tl.bundle_snapshot(tl.TINY_BUNDLE) == before
