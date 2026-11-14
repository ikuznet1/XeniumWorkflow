"""The R worker process (xenium_app/rworker.py): R steps of a job run in a child process
that the job registry knows, so that Cancel ends them at once.

Default tier - no R: the worker runs plain Python targets (tests/rworker_targets.py), which
exercises everything but R itself: start, payload and result exchange, progress and
intermediate results, error propagation, Cancel (the worker and its children are gone
within about a second, the job is cancelled, nothing is published, no file and no zombie
is left, the next run works), a worker that dies, the fall-back to R inside the app and
the rules of the source (which modules may import rpy2).

``tools`` tier - with R (skipped when R / rpy2 / the packages are missing): the pseudobulk
fit in the worker against the same fit inside the app (identical values), R errors with
their text, the R package versions of the provenance record, and Cancel during an R call
that would never return.
"""
from __future__ import annotations

import ast
import json
import os
import signal
import sys
import threading
import time

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import rworker_targets as T
import xe_testlib as tl
from test_jobs import _pid_alive, _wait
from test_stats import N_GENES, six  # noqa: F401  (six: the six fabricated samples, a fixture)

CANCEL_LIMIT_S = 1.5        # "within about a second"; the measured value is printed


# ── helpers ──────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _worker_mode(monkeypatch):
    """Every test starts with the default: R steps run in the worker."""
    monkeypatch.delenv("XENIUM_R_INPROCESS", raising=False)


def _no_worker_left(xe):
    assert not xe._rw_live_dirs, f"exchange directories left: {xe._rw_live_dirs}"
    assert not xe._job_procs_all, "a worker process is still registered"


class _Job:
    """A job written like the app's (own _set through _stale_job, own error handler)
    whose one long step is an R worker call."""

    def __init__(self, xe, target, payload):
        self.xe, self.seen, self.steps = xe, {}, []
        self.state = {"status": "running", "message": "Starting…", "result": None}
        self.lock = threading.Lock()
        gen = xe._dataset_gen

        def _set(status, message, result=None):
            with self.lock:
                if xe._stale_job(gen, status, message):
                    return
                self.state["status"], self.state["message"] = status, message
                if result is not None:
                    self.state["result"] = result
            self.steps.append(message)

        def body():
            try:
                _set("running", "preparing")
                out = xe._rw_call(target, payload, on_progress=lambda m, f=None: _set("running", m),
                                  on_event=lambda name, data: self.seen.update(data))
                _set("done", "finished", result=out)
            except xe._JobCancelled:
                raise
            except Exception as exc:
                _set("error", str(exc))

        self.job = xe._job_start("de", "worker test", body, state=self.state, lock=self.lock)

    def wait_in_step(self, text="sleeping"):
        _wait(lambda: self.state["message"] == text and "pid" in self.seen, timeout=60, what=f"step {text!r}")
        return int(self.seen["pid"])

    def cancel(self):
        """Cancel and wait for the job's thread; seconds from Cancel to the job's end."""
        t0 = time.time()
        assert self.xe._job_cancel(self.job.id)
        self.job.thread.join(20)
        assert not self.job.thread.is_alive(), "the job did not end after Cancel"
        return time.time() - t0


# ── exchange format ──────────────────────────────────────────────────────────
def test_payload_round_trip_keeps_values_dtypes_and_sparse_formats(xe, tmp_path):
    rng = np.random.default_rng(0)
    m = sp.random(7, 5, density=0.4, random_state=1, format="csr", dtype=np.float32)
    obj = {"i64": np.arange(6, dtype=np.int64).reshape(2, 3), "f64": rng.normal(size=4),
           "nan": np.array([np.nan, np.inf, -0.0, 1e-310]), "u": np.array(["a", "bc"]),
           "csr": m, "csc": m.tocsc(), "names": ["g1", "g2"], "none": None, "n": 3, "x": 0.1 + 0.2,
           "nested": {"a": [1, 2.5, "z", None, True]}, "tuple": (1, 2)}
    xe._rw_write(str(tmp_path), "t", obj)
    assert sorted(os.listdir(tmp_path)) == ["t.json", "t.npz"]
    back = xe._rw_read(str(tmp_path), "t")
    assert os.listdir(tmp_path) == [], "an exchange object is removed once it was read"
    same = xe._rw_normalise(obj)                       # what the in-process path hands over
    for got in (back, same):
        assert set(got) == set(obj)
        for k in ("i64", "f64", "nan", "u"):
            assert got[k].dtype == obj[k].dtype and got[k].shape == obj[k].shape
            assert np.array_equal(got[k], obj[k], equal_nan=(obj[k].dtype.kind == "f"))
        assert np.signbit(got["nan"][2])
        for k in ("csr", "csc"):
            assert got[k].format == k and got[k].dtype == np.float32 and tl.sparse_equal(got[k], obj[k])
        assert got["names"] == ["g1", "g2"] and got["none"] is None and got["n"] == 3
        assert got["x"] == 0.1 + 0.2 and got["nested"] == {"a": [1, 2.5, "z", None, True]}
        assert got["tuple"] == [1, 2], "JSON values arrive as JSON gives them, on both paths"
    assert back["f64"].flags.owndata and back["i64"].base is None or back["i64"].flags.owndata


def test_nothing_is_unpickled(xe, tmp_path):
    with pytest.raises(TypeError, match="Python objects"):
        xe._rw_pack({"x": np.array([{"a": 1}], dtype=object)})
    with pytest.raises(TypeError):
        xe._rw_pack({"x": sp.coo_matrix(np.eye(2))})
    with pytest.raises(TypeError):
        xe._rw_pack([1, 2])
    # a file with a pickled array (not something the worker writes) is refused, not loaded
    np.savez(tmp_path / "evil.npz", x=np.array([{"a": 1}], dtype=object))
    (tmp_path / "evil.json").write_text(json.dumps({"values": {}, "arrays": ["x"], "sparse": {}}))
    with pytest.raises(ValueError, match="allow_pickle"):
        xe._rw_read(str(tmp_path), "evil")
    src = (tl.APP_PACKAGE / "rworker.py").read_text()
    assert "allow_pickle=False" in src and "import pickle" not in src and "allow_pickle=True" not in src


# ── a call ───────────────────────────────────────────────────────────────────
def test_call_runs_the_target_in_another_process_and_cleans_up(xe):
    x = np.arange(12, dtype=np.float64).reshape(3, 4)
    m = sp.csr_matrix(np.array([[0, 2, 0], [1, 0, 3]], dtype=np.float32))
    progress, events, notes = [], [], []
    out = xe._rw_call(T.echo, {"x": x, "m": m, "names": ["a", "b"], "tag": "t1"},
                      on_progress=lambda msg, frac=None: progress.append((msg, frac)),
                      on_event=lambda name, data: events.append((name, data)), on_note=notes.append)
    assert out["pid"] != os.getpid() and not _pid_alive(out["pid"]), "the worker is gone after the call"
    assert np.array_equal(out["x"], x) and out["x"].dtype == x.dtype and out["x"].flags.owndata
    assert tl.sparse_equal(out["m"], m) and out["m"].format == "csr" and out["m"].dtype == np.float32
    assert out["names"] == ["a", "b"] and out["sum"] == float(x.sum())
    assert progress == [("echo: started", 0.25)]
    assert [n for n, _ in events] == ["half"] and events[0][1]["pid"] == out["pid"]
    assert np.array_equal(events[0][1]["x2"], x * 2) and events[0][1]["tag"] == "t1"
    assert notes == []
    _no_worker_left(xe)


def test_session_keeps_one_worker_for_several_calls(xe):
    with xe._rw_session():
        a = xe._rw_call(T.echo, {"x": np.ones(3)})
        (d,) = tuple(xe._rw_live_dirs)
        assert os.path.isdir(d) and (os.stat(d).st_mode & 0o077) == 0, "the exchange directory is private"
        assert os.listdir(d) == [], "request, event and result files are removed as they are read"
        with pytest.raises(ValueError, match="^planted refusal$"):
            xe._rw_call(T.fail, {"kind": "value", "message": "planted refusal"})
        b = xe._rw_call(T.echo, {"x": np.zeros(2)})          # the worker survives a failing target
        assert a["pid"] == b["pid"] != os.getpid() and _pid_alive(a["pid"])
        with xe._rw_session():                               # nested: the same worker
            assert xe._rw_call(T.echo, {"x": np.zeros(1)})["pid"] == a["pid"]
        assert _pid_alive(a["pid"])
    assert not _pid_alive(a["pid"]) and not os.path.exists(d)
    _no_worker_left(xe)


def test_errors_of_the_target_arrive_with_their_message(xe):
    r_text = "Error in DESeqDataSet(se, design = design) : \n  the model matrix is not full rank\n"
    with pytest.raises(xe._RwError) as err:
        xe._rw_call(T.fail, {"kind": "runtime", "message": r_text})
    assert str(err.value) == r_text and err.value.kind == "RuntimeError"
    assert err.value.formatted.strip().splitlines()[-1] == "  the model matrix is not full rank"
    with pytest.raises(xe._RwError) as err:
        xe._rw_call(T.fail, {"kind": "key", "message": "gene"})
    assert str(err.value) == "'gene'" and err.value.kind == "KeyError"
    with pytest.raises(ValueError) as err:                   # a refusal stays a ValueError
        xe._rw_call(T.fail, {"kind": "value", "message": "needs DESeq2"})
    assert str(err.value) == "needs DESeq2" and not isinstance(err.value, xe._RwRefused)
    with pytest.raises(xe._RwError, match="Python objects"):
        xe._rw_call(T.bad_result, {})
    with pytest.raises(TypeError):                           # refused before a worker starts
        xe._rw_call(T.echo, {"x": np.array([object()], dtype=object)})
    with pytest.raises(TypeError, match="module-level"):
        xe._rw_call(lambda p, ctx: {}, {})
    _no_worker_left(xe)


def test_a_job_reports_the_workers_error_as_its_own(xe):
    j = _Job(xe, T.fail, {"kind": "runtime", "message": "Error in purify(): singular"})
    j.job.thread.join(60)
    assert j.job.state == "error" and j.state == {"status": "error", "message": "Error in purify(): singular",
                                                   "result": None}
    assert j.steps == ["preparing", "about to fail", "Error in purify(): singular"]
    _no_worker_left(xe)


# ── Cancel ───────────────────────────────────────────────────────────────────
def test_cancel_terminates_the_worker_within_a_second_and_leaves_nothing(xe, capsys):
    j = _Job(xe, T.sleep_once, {"seconds": 600})
    pid = j.wait_in_step()
    (d,) = tuple(xe._rw_live_dirs)
    snap = j.job.snapshot()
    assert snap["n_procs"] == 1 and snap["cancellable"] and os.path.isdir(d) and _pid_alive(pid)
    seconds = j.cancel()
    with capsys.disabled():
        print(f"\n  [measured] Cancel → job ended: {seconds:.3f} s (Python target)", end="")
    assert seconds < CANCEL_LIMIT_S, f"Cancel took {seconds:.2f} s"
    assert j.job.state == "cancelled" and j.job.cancelled
    assert j.state == {"status": "error", "message": "Cancelled", "result": None}, "nothing is published"
    assert not _pid_alive(pid) and not os.path.exists(d) and j.job.procs == []
    _no_worker_left(xe)
    # no zombie: the worker was reaped, so the kernel no longer knows the pid as our child
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)
    # the same job runs again straight away
    j2 = _Job(xe, T.sleep_once, {"seconds": 0.05})
    j2.job.thread.join(60)
    assert j2.job.state == "done" and j2.state["result"]["slept"] == 0.05
    assert j2.state["result"]["pid"] != pid
    _no_worker_left(xe)


def test_cancel_also_stops_what_the_worker_started(xe):
    j = _Job(xe, T.sleep_with_child, {"seconds": 600})
    pid = j.wait_in_step()
    child = int(j.seen["child"])
    assert _pid_alive(child)
    assert j.cancel() < CANCEL_LIMIT_S
    _wait(lambda: not _pid_alive(child), timeout=5, what="the worker's child to end")
    assert not _pid_alive(pid) and j.job.state == "cancelled"
    _no_worker_left(xe)


def test_cancel_between_two_calls_of_a_session_stops_the_idle_worker(xe):
    state, lock = {"status": "running", "message": "Starting…", "result": None}, threading.Lock()
    seen, idle, go = {}, threading.Event(), threading.Event()

    def body():
        with xe._rw_session():
            seen["pid"] = xe._rw_call(T.echo, {"x": np.ones(2)})["pid"]
            idle.set()
            go.wait(30)
            xe._job_check()
            xe._rw_call(T.echo, {"x": np.ones(2)})
            seen["second"] = True

    job = xe._job_start("de", "worker test", body, state=state, lock=lock)
    assert idle.wait(60)
    assert xe._job_cancel(job.id)
    _wait(lambda: not _pid_alive(seen["pid"]), timeout=5, what="the idle worker to end")
    go.set()
    job.thread.join(20)
    assert job.state == "cancelled" and "second" not in seen
    _no_worker_left(xe)


def test_no_worker_is_started_for_a_job_that_is_already_cancelled(xe):
    state, lock = {"status": "running", "message": "Starting…", "result": None}, threading.Lock()
    started, go, seen = threading.Event(), threading.Event(), {}

    def body():
        started.set()
        go.wait(30)
        try:
            xe._rw_call(T.echo, {"x": np.ones(2)}, on_note=lambda n: seen.setdefault("note", n))
        except xe._JobCancelled:
            seen["cancelled"] = True
            raise

    job = xe._job_start("de", "worker test", body, state=state, lock=lock)
    assert started.wait(10) and xe._job_cancel(job.id)
    go.set()
    job.thread.join(20)
    assert seen == {"cancelled": True}, "a Cancel is not mistaken for a worker that cannot start"
    assert job.state == "cancelled"
    _no_worker_left(xe)


def test_shutdown_hook_stops_the_worker_and_removes_its_directory(xe):
    j = _Job(xe, T.sleep_once, {"seconds": 600})
    pid = j.wait_in_step()
    (d,) = tuple(xe._rw_live_dirs)
    xe._jobs_kill_all_processes()            # what atexit and the SIGTERM / SIGHUP handler call
    assert not _pid_alive(pid) and not os.path.exists(d)
    j.job.thread.join(20)
    assert j.job.state == "error" and "ended unexpectedly" in j.state["message"]
    _no_worker_left(xe)


# ── a worker that dies, a worker that cannot start ───────────────────────────
def test_a_worker_that_dies_is_an_error_not_a_hang(xe):
    j = _Job(xe, T.sleep_once, {"seconds": 600})
    pid = j.wait_in_step()
    (d,) = tuple(xe._rw_live_dirs)
    os.kill(pid, signal.SIGKILL)             # the worker this test started, by its pid
    j.job.thread.join(20)
    assert j.job.state == "error" and j.state["status"] == "error"
    assert "The R worker process ended unexpectedly (exit code -9)" in j.state["message"]
    assert not os.path.exists(d)
    _no_worker_left(xe)


@pytest.mark.parametrize("how", ["no interpreter", "exits at once"])
def test_falls_back_to_r_inside_the_app_when_the_worker_cannot_start(xe, monkeypatch, tmp_path, capsys, how):
    if how == "no interpreter":
        monkeypatch.setattr(xe, "_RW_PYTHON", str(tmp_path / "no-such-python"))
        reason = "FileNotFoundError"
    else:
        script = tmp_path / "broken_worker.py"
        script.write_text("import sys\nprint('cannot load libR')\nsys.exit(3)\n")
        monkeypatch.setattr(xe, "_RW_SCRIPT", str(script))
        reason = "it ended with exit code 3 before it was ready: cannot load libR"
    notes, progress = [], []
    out = xe._rw_call(T.echo, {"x": np.arange(3.0)}, on_note=notes.append,
                      on_progress=lambda m, f=None: progress.append(m))
    assert out["pid"] == os.getpid(), "the target ran in this process"
    assert np.array_equal(out["x"], np.arange(3.0)) and progress == ["echo: started"]
    assert len(notes) == 1 and "the R worker process could not start" in notes[0] and reason in notes[0]
    assert "Cancel only takes effect once the R call returns" in notes[0]
    assert "R worker: the R worker process could not start" in capsys.readouterr().out
    _no_worker_left(xe)
    # inside a session the start is tried once, every call then runs in-process
    notes.clear()
    with xe._rw_session():
        pids = {xe._rw_call(T.echo, {"x": np.ones(1)}, on_note=notes.append)["pid"] for _ in range(3)}
    assert pids == {os.getpid()} and len(notes) == 1
    _no_worker_left(xe)


def test_environment_switch_runs_the_target_inside_the_app(xe, monkeypatch):
    monkeypatch.setenv("XENIUM_R_INPROCESS", "1")
    assert xe._rw_inprocess()
    events = []
    out = xe._rw_call(T.echo, {"x": np.arange(4.0), "t": (1, 2)}, on_event=lambda n, d: events.append(d))
    assert out["pid"] == os.getpid() and events[0]["pid"] == os.getpid()
    assert out["t"] == [1, 2], "the same types as through the worker"
    with pytest.raises(RuntimeError, match="^boom$") as err:     # the original exception, not a copy
        xe._rw_call(T.fail, {"kind": "runtime", "message": "boom"})
    assert not isinstance(err.value, xe._RwError)
    with xe._rw_session():
        assert xe._rw_call(T.echo, {"x": np.ones(1)})["pid"] == os.getpid()
    _no_worker_left(xe)
    for off in ("", "0", "no"):
        monkeypatch.setenv("XENIUM_R_INPROCESS", off)
        assert not xe._rw_inprocess()


def test_pb_fit_passes_the_fallback_note_to_the_status_line(xe, monkeypatch):
    """_pb_fit with the R part replaced: the note of a worker that cannot start ends up in
    the notes of the fit, which the job prints as 'Note: …' in its status line."""
    def fake_call(target, payload, on_progress=None, on_event=None, r_packages=(), on_note=None):
        assert target is xe._rw_pb_fit and r_packages == xe._PB_R_PACKAGES
        assert payload["counts"].dtype == np.int64 and payload["install_hint"] == xe._PB_INSTALL_HINT
        on_note("the R worker process could not start (test): R ran inside the app")
        n = payload["counts"].shape[0]
        z = np.zeros(n)
        return {"engine": "deseq2", "size_factors": np.ones(payload["counts"].shape[1]), "log2fc": z,
                "stat": z, "pval": z + 1, "padj": z + 1, "note": "", "version": "0.0", "shrink_type": "none",
                "lfc_se": z, "base_mean": z}
    monkeypatch.setattr(xe, "_rw_call", fake_call)
    fit = xe._pb_fit({"counts": np.ones((4, 4), dtype=np.int64), "genes": list("abcd"),
                      "labels": ["s1", "s2", "s3", "s4"], "cond": ["A", "A", "B", "B"], "batch": None})
    assert fit["notes"] == ["the R worker process could not start (test): R ran inside the app"]
    assert fit["size_factors"].flags.owndata


# ── rules of the source ──────────────────────────────────────────────────────
# Functions that still call R inside the app process, and why they may:
RPY2_IN_APP = {
    "rworker.py": None,                              # the worker's own targets and helpers
    "core.py": {"_redirect_rpy2_console"},           # console callbacks only, no R call
    "annotation.py": {"_run_seurat_annotation"},     # reads the reference (readRDS + export)
    "spage.py": {"_run_spage_imputation"},           # reads the reference (readRDS + export)
    "cb_tools.py": {"start_annotation"},             # thread rule before an annotation job
}


def _rpy2_importers(path):
    """Top-level functions of *path* that import rpy2 ('<module>' for an import outside one)."""
    tree = ast.parse(path.read_text())
    owner = {}
    for fn in tree.body:
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for node in ast.walk(fn):
                owner[id(node)] = fn.name
    found = set()
    for node in ast.walk(tree):
        names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                 [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
        if any(n.split(".")[0] == "rpy2" for n in names):
            found.add(owner.get(id(node), "<module>"))
    return found


def test_r_is_called_inside_the_app_only_where_listed():
    """A long R step belongs in a target of rworker.py (run through _rw_call), where Cancel
    can end it. A new `import rpy2` anywhere else fails here until it is moved or listed."""
    for path in tl.app_sources():
        got = _rpy2_importers(path)
        allowed = RPY2_IN_APP.get(path.name, set())
        if allowed is None:
            continue
        assert got <= allowed, f"{path.name}: R is used inside the app in {sorted(got - allowed)}"
    for name, allowed in RPY2_IN_APP.items():
        if allowed:
            assert _rpy2_importers(tl.APP_PACKAGE / name) == allowed, f"{name}: the list above is out of date"


def test_the_worker_is_a_registered_child_process_and_the_targets_go_through_rw_call():
    src = (tl.APP_PACKAGE / "rworker.py").read_text()
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    attr = lambda n: getattr(n.func, "attr", getattr(n.func, "id", None))
    assert not [n for n in calls if attr(n) in ("Popen", "run", "check_output", "system")], \
        "the worker is started with jobs._job_popen only"
    assert [n for n in calls if attr(n) == "_job_popen"] and [n for n in calls if attr(n) == "_job_proc_done"]
    # the targets are only ever handed to _rw_call
    targets = {"_rw_pb_fit", "_rw_rctd", "_rw_split"}
    seen = set()
    for node in ast.walk(tl.app_tree()):
        if not isinstance(node, ast.Call):
            continue
        used = {n.attr if isinstance(n, ast.Attribute) else n.id for a in list(node.args)
                for n in ast.walk(a) if isinstance(n, (ast.Name, ast.Attribute))} & targets
        direct = attr(node) in targets
        assert not direct, f"line {node.lineno}: an R target is called directly, not through _rw_call"
        if used:
            assert attr(node) == "_rw_call", f"line {node.lineno}: {used} passed to {attr(node)}"
            seen |= used
    assert seen == targets
    # the worker script never imports the app (it would load the dataset a second time)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level:
            raise AssertionError("rworker.py imports the package unconditionally")
    assert "if __package__:\n    from . import jobs" in src


def test_worker_script_does_not_import_the_app(xe):
    """Asked of the worker itself: the modules loaded in it after a call."""
    with xe._rw_session():
        xe._rw_call(T.echo, {"x": np.ones(1)})
        (d,) = tuple(xe._rw_live_dirs)
        probe = os.path.join(os.path.dirname(d), os.path.basename(d) + "_probe.py")
        try:
            with open(probe, "w") as fh:
                fh.write("import sys\n\ndef loaded(p, ctx):\n"
                         "    return {'mods': sorted(m for m in sys.modules if m.split('.')[0] in "
                         "('xenium_app', 'xenium_explorer', 'dash', 'rpy2', 'jobs', 'core'))}\n")
            import importlib.util
            spec = importlib.util.spec_from_file_location("_rw_probe", probe)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            assert xe._rw_call(mod.loaded, {})["mods"] == []
        finally:
            os.remove(probe)


# ── with R ───────────────────────────────────────────────────────────────────
def _need_r(*pkgs):
    pytest.importorskip("rpy2")
    if not tl.r_packages_available(*pkgs):
        pytest.skip(f"R with {', '.join(pkgs) or 'no extra package'} is not available")


def _r_in_this_thread():
    import rpy2.robjects as ro
    import rpy2.robjects.conversion as conversion
    conversion.set_conversion(ro.default_converter)


def _prep(batch=None, n_genes=400, seed=0):
    """A pseudobulk matrix as _pb_prepare gives it: 3 + 3 samples, 30 genes 3× higher in A."""
    rng = np.random.default_rng(seed)
    mu = rng.gamma(2.0, 60.0, n_genes)
    effect = np.ones(n_genes)
    effect[:30] = 3.0
    depth = [1.0, 1.3, 0.8, 1.1, 0.9, 1.2]
    cols = [rng.negative_binomial(8, 8 / (8 + mu * (effect if j < 3 else 1.0) * depth[j])) for j in range(6)]
    return {"counts": np.column_stack(cols).astype(np.int64), "genes": [f"G{i}" for i in range(n_genes)],
            "labels": [f"S{j + 1}" for j in range(6)], "cond": ["A"] * 3 + ["B"] * 3, "batch": batch}


@pytest.mark.tools
@pytest.mark.parametrize("engine,shrink,batch", [("deseq2", True, None),
                                                 ("deseq2", False, ["x", "y", "z", "x", "y", "z"]),
                                                 ("edger", False, None)],
                         ids=["deseq2+shrink", "deseq2+batch", "edger"])
def test_pseudobulk_fit_in_the_worker_is_identical_to_the_fit_inside_the_app(xe, monkeypatch, capsys,
                                                                             engine, shrink, batch):
    _need_r("DESeq2" if engine == "deseq2" else "edgeR")
    prep = _prep(batch)
    t0 = time.time()
    w = xe._pb_fit(prep, shrink=shrink, engine=engine)           # in the worker
    t_worker = time.time() - t0
    _no_worker_left(xe)
    monkeypatch.setenv("XENIUM_R_INPROCESS", "1")
    _r_in_this_thread()
    t0 = time.time()
    a = xe._pb_fit(prep, shrink=shrink, engine=engine)           # inside the app (first: loads the package)
    t_first = time.time() - t0
    t0 = time.time()
    b = xe._pb_fit(prep, shrink=shrink, engine=engine)
    t_again = time.time() - t0
    with capsys.disabled():
        print(f"\n  [measured] {engine}: worker {t_worker:.2f} s; inside the app {t_first:.2f} s "
              f"(first call), {t_again:.2f} s (again)", end="")
    assert set(w) == set(a)
    for other in (a, b):
        pd.testing.assert_frame_equal(w["table"], other["table"], check_exact=True)
        assert np.array_equal(w["size_factors"], other["size_factors"])
        for k in ("engine", "method_label", "shrink", "stat_name", "mean_unit", "notes"):
            assert w[k] == other[k], k
    assert w["size_factors"].flags.owndata and a["size_factors"].flags.owndata
    assert w["engine"] == engine and int((w["table"]["padj"] < 0.05).sum()) >= 15
    if engine == "deseq2" and shrink:
        assert "log2fc_shrunk" in w["table"].columns and w["shrink"] in ("apeglm", "normal")


@pytest.mark.tools
def test_r_error_text_is_the_same_from_the_worker(xe, monkeypatch):
    _need_r()
    text = "planted: the model matrix is not full rank"
    with pytest.raises(xe._RwError) as w:
        xe._rw_call(T.r_stop, {"message": text})
    monkeypatch.setenv("XENIUM_R_INPROCESS", "1")
    _r_in_this_thread()
    with pytest.raises(Exception) as a:
        xe._rw_call(T.r_stop, {"message": text})
    assert type(a.value).__name__ == "RRuntimeError" == w.value.kind
    assert str(w.value) == str(a.value) and text in str(w.value)
    import traceback
    assert w.value.formatted == "".join(traceback.format_exception_only(type(a.value), a.value))
    _no_worker_left(xe)


@pytest.mark.tools
def test_cancel_during_an_r_call_that_never_returns(xe, capsys):
    _need_r()
    j = _Job(xe, T.r_busy, {})
    pid = j.wait_in_step("R is busy")
    time.sleep(0.3)                          # R is in its loop
    seconds = j.cancel()
    with capsys.disabled():
        print(f"\n  [measured] Cancel → job ended: {seconds:.3f} s (R in a loop)", end="")
    assert seconds < CANCEL_LIMIT_S
    assert j.job.state == "cancelled" and j.state["message"] == "Cancelled" and j.state["result"] is None
    assert not _pid_alive(pid)
    _no_worker_left(xe)


@pytest.mark.tools
def test_pseudobulk_job_through_the_worker_records_r_versions_and_can_be_cancelled(xe, D, six, capsys):  # noqa: F811
    _need_r("DESeq2")
    method = (D.get("cluster_methods") or [None])[0]
    run = lambda: xe.run_de(1, None, None, "xenium", method, "cluster", "original", "pseudobulk", "all",
                            10, 10, ["on"])
    last = lambda: [j for j in xe._jobs.values() if j.kind == "de"][-1]
    r_loaded_before = "rpy2.robjects" in sys.modules
    # 1. Cancel while R fits: the job ends at once and publishes nothing
    assert run() == (True, False, "Running…")
    job = last()
    _wait(lambda: job.procs and job.step.startswith("Fitting"), timeout=120, what="the R step")
    pid = job.procs[0].pid
    time.sleep(0.5)
    t0 = time.time()
    assert xe._job_cancel(job.id)
    job.thread.join(20)
    seconds = time.time() - t0
    with capsys.disabled():
        print(f"\n  [measured] Cancel → pseudobulk job ended: {seconds:.3f} s", end="")
    assert seconds < CANCEL_LIMIT_S and job.state == "cancelled" and not _pid_alive(pid)
    assert xe._de_state["status"] == "error" and xe._de_state["message"] == "Cancelled"
    assert xe._de_state.get("result") is None
    _no_worker_left(xe)
    n_records = len(xe._prov_recent)
    # 2. the following run works, and its record names the R of the worker
    assert run() == (True, False, "Running…")
    msg = tl.wait_state(xe._de_state, xe._de_lock, timeout=600)
    last().thread.join(10)
    assert "DESeq2" in msg and last().state == "done"
    recs = [r for r in xe._prov_recent[n_records:] if r["kind"] == "pseudobulk"]
    assert len(recs) == 1
    flat = json.dumps(recs[0]["software"])
    assert "DESeq2" in flat and "edgeR" in flat, flat
    assert xe._de_state["result"]["method_label"].split(",")[0].split()[-1] in flat, "the DESeq2 version of the fit"
    if not r_loaded_before:
        assert "rpy2.robjects" not in sys.modules, "R was not started in the app process"
    # 3. the same run with R inside the app: the same table, the same versions
    table_w = xe._de_state["result"]["table"].copy()
    os.environ["XENIUM_R_INPROCESS"] = "1"
    try:
        assert run() == (True, False, "Running…")
        tl.wait_state(xe._de_state, xe._de_lock, timeout=600)
        last().thread.join(10)
    finally:
        os.environ.pop("XENIUM_R_INPROCESS", None)
    rec_in = [r for r in xe._prov_recent if r["kind"] == "pseudobulk"][-1]
    assert rec_in["software"] == recs[0]["software"]
    cols = [c for c in table_w.columns if table_w[c].dtype.kind == "f"]
    pd.testing.assert_frame_equal(table_w[["gene"] + cols], xe._de_state["result"]["table"][["gene"] + cols],
                                  check_exact=True)


@pytest.mark.tools
def test_contact_job_fits_the_observed_and_shuffled_models_in_one_worker(xe, monkeypatch, capsys):
    """The contact job with real R: 1 + n label shuffles, each a DESeq2 fit, in ONE worker
    (R and DESeq2 load once), which is gone with its directory when the job ends."""
    _need_r("DESeq2")
    import test_proximity as tp
    four = [tp._section(10 + i, n_t=1200, n_o=300, label=f"S{i}") for i in range(4)]
    tp._patch_view(xe, monkeypatch, four)
    starts, real_start = [], xe._RwSession.start

    def counting_start(self):
        real_start(self)
        starts.append(self.proc.pid)

    monkeypatch.setattr(xe._RwSession, "start", counting_start)
    n_before = len(xe._prov_recent)
    t0 = time.time()
    tp._run(xe, "contact", contact_nshuffle=4)
    msg = tl.wait_state(xe._de_state, xe._de_lock, timeout=600)
    job = [j for j in xe._jobs_list() if j.kind == "contact"][-1]
    job.thread.join(30)
    with capsys.disabled():
        print(f"\n  [measured] contact job, 1 + 4 DESeq2 fits in one worker: {time.time() - t0:.1f} s", end="")
    res = xe._de_state["result"]
    assert job.state == "done" and res["kind"] == "contact" and "(4 shuffles" in msg
    assert res["shuffle"]["n"] == 4 and res["shuffle"]["failed"] == 0
    assert len(starts) == 1 and not _pid_alive(starts[0]), f"workers started: {len(starts)}"
    t = res["table"].set_index("gene")
    assert t.loc["CONTACT", "padj"] < 0.05
    rec = [r for r in xe._prov_recent[n_before:] if r["kind"] == "contact"][-1]
    flat = json.dumps(rec["software"])
    assert "DESeq2" in flat and "edgeR" in flat and '"R"' in flat, flat
    _no_worker_left(xe)
