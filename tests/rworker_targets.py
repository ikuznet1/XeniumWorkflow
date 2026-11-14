"""Targets for the R worker tests (tests/test_rworker.py).

The worker process loads this file by path, on its own: it must not import the app or
the test library. A target is ``target(payload, ctx) -> dict``; see xenium_app/rworker.py.
The first group is plain Python (the default tier runs without R), the second calls R.
"""
import os
import subprocess
import sys
import time

import numpy as np


# ── plain Python ─────────────────────────────────────────────────────────────
def echo(p, ctx):
    """The payload back, with the worker's pid; one progress report and one event."""
    ctx.progress("echo: started", 0.25)
    ctx.emit("half", {"pid": os.getpid(), "x2": p["x"] * 2, "tag": p.get("tag")})
    out = dict(p)
    out["pid"] = os.getpid()
    out["sum"] = float(np.asarray(p["x"], dtype=np.float64).sum())
    return out


def sleep_once(p, ctx):
    """One long call that never looks at a flag, like an R call in progress."""
    ctx.emit("pid", {"pid": os.getpid()})
    ctx.progress("sleeping")
    time.sleep(float(p["seconds"]))
    return {"slept": float(p["seconds"]), "pid": os.getpid()}


def sleep_with_child(p, ctx):
    """As sleep_once, with a child process of the worker (R starts some for RCTD)."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    ctx.emit("pid", {"pid": os.getpid(), "child": child.pid})
    ctx.progress("sleeping")
    time.sleep(float(p["seconds"]))
    child.kill()
    child.wait()
    return {"slept": float(p["seconds"])}


def fail(p, ctx):
    """Raise the exception the payload asks for."""
    ctx.progress("about to fail")
    if p["kind"] == "value":
        raise ValueError(p["message"])
    if p["kind"] == "key":
        raise KeyError(p["message"])
    raise RuntimeError(p["message"])


def bad_result(p, ctx):
    """A result that cannot be passed back (an array of Python objects)."""
    return {"x": np.array([object()], dtype=object)}


# ── R ────────────────────────────────────────────────────────────────────────
def r_stop(p, ctx):
    """An error raised by R itself."""
    import rpy2.robjects as ro
    import rpy2.robjects.conversion as conversion
    conversion.set_conversion(ro.default_converter)
    ro.r("function(m) stop(m)")(p["message"])
    return {}


def r_busy(p, ctx):
    """R in a loop that only ends with the process."""
    import rpy2.robjects as ro
    import rpy2.robjects.conversion as conversion
    conversion.set_conversion(ro.default_converter)
    ro.r("suppressPackageStartupMessages(library(stats))")
    ctx.emit("pid", {"pid": os.getpid()})
    ctx.progress("R is busy")
    ro.r("x <- 0; repeat { x <- x + 1 }")
    return {}
