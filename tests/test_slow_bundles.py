"""Slow tier (``-m slow``): the default tier again, on larger real bundles.

One bundle needs one Python process (the app loads its dataset at import), so each
bundle named by an environment variable gets its own ``pytest --bundle PATH`` child
process running the bundle-dependent modules of the default tier. Nothing is hard-coded
for those datasets: the default-tier tests derive every expectation from the bundle's
own raw files.

    XENIUM_TEST_BUNDLE_SKIN   Xenium Prime 5K human skin FFPE (~7 GB, ~10^5 cells)
    XENIUM_TEST_BUNDLE_GEO    a bundle assembled by functions/geo_to_bundle.py
                              (no boundaries, clustering or images)
    XENIUM_TEST_BUNDLE_EXTRA  any further bundle

Unset or missing paths are skipped. The bundles are used through a working directory
of symlinks and are checked for modifications by the child session.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

import xe_testlib as tl

pytestmark = pytest.mark.slow

BUNDLE_ENV = ["XENIUM_TEST_BUNDLE_SKIN", "XENIUM_TEST_BUNDLE_GEO", "XENIUM_TEST_BUNDLE_EXTRA"]
# Bundle-dependent modules. test_patches.py is left out: sopa's tissue segmentation and
# patching take tens of minutes on a whole 5K slide.
MODULES = ["test_app_data.py", "test_viewer_store.py", "test_figures.py", "test_viewer_endpoints.py",
           "test_subset.py", "test_roi.py", "test_spatial_analysis.py", "test_save_sdata.py",
           "test_cache_and_jobs.py"]


@pytest.mark.parametrize("var", BUNDLE_ENV)
def test_default_tier_on_bundle(var):
    path = os.environ.get(var)
    if not path:
        pytest.skip(f"{var} is not set")
    if not os.path.exists(os.path.join(path, "experiment.xenium")):
        pytest.skip(f"{var}={path} is not a Xenium bundle")
    cmd = [sys.executable, "-m", "pytest", "--bundle", path, "-m", "not slow and not tools and not browser",
           "-q", "-x", "--durations=10", "-p", "no:cacheprovider",
           *[str(tl.TESTS / m) for m in MODULES]]
    proc = subprocess.run(cmd, cwd=str(tl.REPO), capture_output=True, text=True,
                          timeout=int(os.environ.get("XENIUM_TEST_SLOW_TIMEOUT", 4 * 3600)))
    tail = "\n".join(proc.stdout.splitlines()[-60:])
    assert proc.returncode == 0, f"default tier failed on {path}\n{tail}\n{proc.stderr[-2000:]}"
    print(tail)
