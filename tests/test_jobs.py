"""The job registry, progress reporting and cancellation, with stub jobs.

No external tool is needed: the "tool" is /bin/sh with sleeping children, started
through the same _job_run_process() that runs Baysor and Proseg. The checks look at
the operating system (are the processes gone?), at the cache directory (would the
cached-run listing offer what the cancelled job left?) and at the job kind's status
dict (can the job be started again?), not at the registry's own bookkeeping.
The real tools are cancelled mid-run in test_tools_reseg.py (``-m tools``).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

import numpy as np
import pytest

import xe_testlib as tl

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")


# ── helpers ──────────────────────────────────────────────────────────────────
def _wait(cond, timeout=20.0, what="condition"):
    deadline = time.time() + timeout
    while not cond():
        assert time.time() < deadline, f"timed out waiting for {what}"
        time.sleep(0.02)


def _pid_alive(pid: int) -> bool:
    """True while *pid* is a running process (a zombie awaiting its parent counts as gone)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    import subprocess
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(out) and not out.startswith("Z")


def _new_state():
    return {"status": "running", "message": "Starting…", "result": None}, threading.Lock()


def _stepping_job(xe, state, lock, n_steps, entered=None, step_sleep=0.02):
    """A job body written like the app's: _set() through _stale_job, own error handler."""
    gen = xe._dataset_gen

    def _set(status, message, result=None):
        with lock:
            if xe._stale_job(gen, status, message):
                return
            state["status"], state["message"] = status, message
            if result is not None:
                state["result"] = result

    def body():
        try:
            for i in range(n_steps):
                _set("running", f"step {i + 1}/{n_steps}")
                xe._job_progress(fraction=i / n_steps)
                if entered is not None:
                    entered.set()
                time.sleep(step_sleep)
            _set("done", "finished", result={"steps": n_steps})
        except Exception as exc:              # as in _run_baysor & co.
            _set("error", str(exc))

    return body


@pytest.fixture
def fake_cache(xe, tmp_path, monkeypatch):
    root = tmp_path / "fake_cache"
    root.mkdir()
    monkeypatch.setattr(xe, "CACHE_BASE_DIR", str(root))
    return root


# ── registry ─────────────────────────────────────────────────────────────────
def test_registry_lists_a_running_job_with_thread_step_fraction_and_elapsed(xe):
    state, lock = _new_state()
    release, entered = threading.Event(), threading.Event()

    def body():
        xe._job_progress("counting", 0.25)
        current.append(xe._job_current())
        entered.set()
        release.wait(30)
        with lock:
            state.update(status="done", message="ok")

    current = []
    before = set(threading.enumerate())
    job = xe._job_start("spatial", "stub analysis", body, state=state, lock=lock)
    try:
        assert entered.wait(10)
        assert current == [job], "_job_current() inside the job is the registered job"
        assert job in xe._jobs_list(active_only=True)
        # the registered thread is the one the job runs in (no guessing by thread name)
        assert job.thread.is_alive() and job.thread not in before and job.thread.daemon
        snap = job.snapshot()
        assert (snap["kind"], snap["label"], snap["state"]) == ("spatial", "stub analysis", "running")
        assert snap["step"] == "counting" and snap["fraction"] == 0.25 and snap["cancellable"]
        t0 = snap["elapsed"]
        time.sleep(0.15)
        assert job.snapshot()["elapsed"] >= t0 + 0.1
        assert "spatial analysis" in xe._running_jobs(), "_running_jobs reads the registry"
        assert xe._job_current() is None, "the caller's thread is not a job"
    finally:
        release.set()
    job.thread.join(10)
    assert job.state == "done" and job.ended is not None and not job.active()
    assert job not in xe._jobs_list(active_only=True) and job in xe._jobs_list()
    assert "spatial analysis" not in xe._running_jobs()
    frozen = job.snapshot()["elapsed"]
    time.sleep(0.05)
    assert job.snapshot()["elapsed"] == frozen, "elapsed time stops when the job ends"


def test_running_jobs_still_sees_a_claimed_state_without_a_registered_job(xe):
    with xe._split_lock:
        xe._split_state["status"] = "running"
    assert "SPLIT" in xe._running_jobs()


def test_load_sample_is_refused_while_a_registered_job_runs(xe, tmp_path):
    other = tmp_path / "other_bundle"
    other.mkdir()
    (other / "experiment.xenium").write_text("{}")
    state, lock = _new_state()
    release = threading.Event()
    job = xe._job_start("umap", "stub", lambda: release.wait(30), state=state, lock=lock)
    try:
        gen, data_dir = xe._dataset_gen, xe.DATA["data_dir"]
        out = xe.load_sample(1, str(other), None, 0, 0)
        assert out[0].startswith("Still running: UMAP"), out[0]
        assert xe._dataset_gen == gen and xe.DATA["data_dir"] == data_dir
    finally:
        release.set()
    job.thread.join(10)


def test_a_job_that_raises_is_reported_as_an_error_and_releases_its_state(xe, capsys):
    state, lock = _new_state()

    def body():
        raise RuntimeError("boom")

    job = xe._job_start("umap", "stub", body, state=state, lock=lock)
    job.thread.join(10)
    assert job.state == "error"
    assert (state["status"], state["message"]) == ("error", "boom")


# ── cancellation: Python steps ───────────────────────────────────────────────
def test_cancel_stops_a_python_job_at_the_next_step_and_the_job_can_run_again(xe, D):
    state, lock = _new_state()
    entered = threading.Event()
    df_before, expr_before, gen = D["df"], D["expr"], xe._dataset_gen
    cells_before = list(D["df"].index)
    job = xe._job_start("umap", "stub steps", _stepping_job(xe, state, lock, 10_000, entered),
                        state=state, lock=lock)
    assert entered.wait(10)
    assert xe._job_cancel(job.id) is True
    assert job.state in ("cancelling", "cancelled")
    assert xe._job_cancel(job.id) is False, "a second Cancel is a no-op"
    job.thread.join(10)
    assert not job.thread.is_alive() and job.state == "cancelled"
    # the job kind's state is released: what the polls show, and Run is possible again
    assert (state["status"], state["message"], state["result"]) == ("error", "Cancelled", None)
    assert "UMAP" not in xe._running_jobs()
    done_steps = int(job.snapshot()["fraction"] * 10_000)
    assert done_steps < 5_000, "the job stopped early"
    # the active dataset is untouched
    assert D["df"] is df_before and D["expr"] is expr_before and xe._dataset_gen == gen
    assert list(D["df"].index) == cells_before and "_df_original" not in D

    # same job again, to completion
    with lock:
        state.update(status="running", message="Starting…", result=None)
    job2 = xe._job_start("umap", "stub steps", _stepping_job(xe, state, lock, 5, step_sleep=0.0),
                         state=state, lock=lock)
    job2.thread.join(10)
    assert job2.state == "done" and job2.id != job.id
    assert state["status"] == "done" and state["result"] == {"steps": 5}


def test_cancel_is_also_seen_at_check_dataset_gen(xe):
    """Jobs that keep their status dict by hand (niches, spatial analysis, the patch
    steps) only call _check_dataset_gen before they publish."""
    state, lock = _new_state()
    entered, go = threading.Event(), threading.Event()
    published = []

    def body():
        gen = xe._dataset_gen
        entered.set()
        go.wait(30)                       # "a library call in progress"
        with lock:
            xe._check_dataset_gen(gen)
            published.append(True)
            state.update(status="done", message="published")

    job = xe._job_start("niches", "stub", body, state=state, lock=lock)
    assert entered.wait(10)
    assert xe._job_cancel(job.id)
    time.sleep(0.2)
    assert job.thread.is_alive() and job.state == "cancelling", "a step in progress is not interrupted"
    go.set()
    job.thread.join(10)
    assert job.state == "cancelled" and not published
    assert (state["status"], state["message"]) == ("error", "Cancelled")


def test_a_job_that_finishes_before_it_sees_the_flag_is_done_not_cancelled(xe):
    state, lock = _new_state()
    entered, go = threading.Event(), threading.Event()

    def body():
        entered.set()
        go.wait(30)
        with lock:                         # no cancellation point before the result
            state.update(status="done", message="finished", result=1)

    job = xe._job_start("umap", "stub", body, state=state, lock=lock)
    assert entered.wait(10)
    assert xe._job_cancel(job.id)
    go.set()
    job.thread.join(10)
    assert job.state == "done" and (state["status"], state["result"]) == ("done", 1)


def test_shielded_job_cannot_be_cancelled_and_completes(xe):
    state, lock = _new_state()
    shielded, go = threading.Event(), threading.Event()

    def body():
        gen = xe._dataset_gen
        xe._job_shield("writing")
        shielded.set()
        go.wait(30)
        xe._check_dataset_gen(gen)         # no longer a cancellation point
        with lock:
            assert not xe._stale_job(gen, "running", "still writing")
            state.update(status="done", message="written")

    job = xe._job_start("spage", "stub", body, state=state, lock=lock)
    assert shielded.wait(10)
    snap = job.snapshot()
    assert snap["shielded"] and not snap["cancellable"] and snap["step"] == "writing"
    assert xe._job_cancel(job.id) is False
    go.set()
    job.thread.join(10)
    assert job.state == "done" and state["status"] == "done"


def test_shield_raises_when_cancel_was_already_requested(xe):
    state, lock = _new_state()
    entered, go = threading.Event(), threading.Event()
    wrote = []

    def body():
        entered.set()
        go.wait(30)
        xe._job_shield()
        wrote.append(True)

    job = xe._job_start("split", "stub", body, state=state, lock=lock)
    assert entered.wait(10)
    assert xe._job_cancel(job.id)
    go.set()
    job.thread.join(10)
    assert job.state == "cancelled" and not wrote and state["message"] == "Cancelled"


# ── cancellation: external processes ─────────────────────────────────────────
# a "tool" with two child processes; prints their pids, then waits for them
_TREE = "sleep 300 & echo child=$!; sleep 300 & echo child=$!; echo ready; wait"


def _fake_reseg_job(xe, state, lock, tag, script, pids, ready, complete_marker=True):
    """Stand-in for _run_proseg: staging directory, tool process, commit."""
    gen = xe._dataset_gen

    def _set(status, message, result=None):
        with lock:
            if xe._stale_job(gen, status, message):
                return
            state["status"], state["message"] = status, message
            if result is not None:
                state["result"] = result

    def on_line(line):
        if line.startswith("child="):
            pids.append(int(line.split("=")[1]))
        if line == "ready":
            ready.set()
        return False

    def body():
        out_dir = None
        try:
            final_dir = xe._seg_run_dir("proseg", tag)
            out_dir = xe._seg_stage_dir(final_dir)
            if complete_marker:
                # what makes _list_cached_seg_runs call a run directory complete
                open(os.path.join(out_dir, "cell-metadata.csv.gz"), "w").close()
                open(os.path.join(out_dir, "transcript-metadata.csv.gz"), "w").close()
            _set("running", "Running tool…")
            rc = xe._job_run_process(["/bin/sh", "-c", script], cwd=out_dir, log_tag="stub",
                                     on_line=on_line)
            if rc != 0:
                raise RuntimeError(f"tool exited with code {rc}")
            _set("running", "Writing…")
            xe._job_shield()
            with open(os.path.join(out_dir, "params.json"), "w") as f:
                json.dump({"tool": "proseg", "param_tag": tag, "n_cells": 1}, f)
            xe._seg_stage_commit(out_dir, final_dir)
            out_dir = final_dir
            _set("done", "Done", result={"out_dir": final_dir})
        except xe._JobCancelled:
            raise
        except Exception as exc:
            _set("error", str(exc))
        finally:
            xe._seg_stage_discard(out_dir)

    return body


def test_cancel_kills_the_process_tree_and_leaves_no_run_directory(xe, fake_cache, D):
    tag = "abcd1234"
    state, lock = _new_state()
    pids, ready = [], threading.Event()
    df_before, gen = D["df"], xe._dataset_gen
    job = xe._job_start("proseg", "stub Proseg", _fake_reseg_job(xe, state, lock, tag, _TREE, pids, ready),
                        state=state, lock=lock)
    assert ready.wait(20), "the stub tool did not start"
    leader = job.procs[0].pid
    assert len(pids) == 2 and all(_pid_alive(p) for p in pids + [leader])
    assert job.snapshot()["n_procs"] == 1
    stage = [p for p in os.listdir(fake_cache) if p.endswith(xe._SEG_STAGE_SUFFIX)]
    assert len(stage) == 1, "the run works in a staging directory"
    assert xe._list_cached_seg_runs() == [], "a run in progress is not offered as a cached run"

    t0 = time.time()
    assert xe._job_cancel(job.id)
    job.thread.join(30)
    assert not job.thread.is_alive() and job.state == "cancelled"
    assert time.time() - t0 < xe._JOB_KILL_GRACE_S, "sleep dies on SIGTERM: no need to wait for SIGKILL"
    # the whole process tree is gone
    _wait(lambda: not any(_pid_alive(p) for p in pids + [leader]), what="the tool's processes to exit")
    assert job.procs == []
    # nothing a later run or the cached-run listing could take for a finished run
    assert os.listdir(fake_cache) == []
    assert xe._list_cached_seg_runs() == []
    assert not os.path.exists(xe._seg_run_dir("proseg", tag))
    # state released, dataset untouched
    assert (state["status"], state["message"], state["result"]) == ("error", "Cancelled", None)
    assert "Proseg" not in xe._running_jobs()
    assert D["df"] is df_before and xe._dataset_gen == gen

    # the same job again, to completion
    with lock:
        state.update(status="running", message="Starting…", result=None)
    job2 = xe._job_start("proseg", "stub Proseg",
                         _fake_reseg_job(xe, state, lock, tag, "echo ready", [], threading.Event()),
                         state=state, lock=lock)
    job2.thread.join(30)
    assert job2.state == "done" and state["status"] == "done"
    final_dir = state["result"]["out_dir"]
    assert os.path.isdir(final_dir) and os.listdir(fake_cache) == [os.path.basename(final_dir)]
    assert [r["value"] for r in xe._list_cached_seg_runs()] == [f"proseg:{tag}"]


def test_cancelled_rerun_keeps_the_earlier_run_with_the_same_parameters(xe, fake_cache):
    tag = "0123abcd"
    state, lock = _new_state()
    first = xe._job_start("proseg", "stub", _fake_reseg_job(xe, state, lock, tag, "echo ready", [], threading.Event()),
                          state=state, lock=lock)
    first.thread.join(30)
    final_dir = state["result"]["out_dir"]
    before = sorted(os.listdir(final_dir))
    with lock:
        state.update(status="running", message="Starting…", result=None)
    pids, ready = [], threading.Event()
    job = xe._job_start("proseg", "stub", _fake_reseg_job(xe, state, lock, tag, _TREE, pids, ready),
                        state=state, lock=lock)
    assert ready.wait(20)
    assert xe._job_cancel(job.id)
    job.thread.join(30)
    assert job.state == "cancelled"
    assert sorted(os.listdir(final_dir)) == before and os.listdir(fake_cache) == [os.path.basename(final_dir)]
    assert [r["value"] for r in xe._list_cached_seg_runs()] == [f"proseg:{tag}"]
    _wait(lambda: not any(_pid_alive(p) for p in pids), what="children to exit")


def test_a_tool_that_ignores_sigterm_is_killed_after_the_grace_period(xe, fake_cache, monkeypatch):
    monkeypatch.setattr(xe, "_JOB_KILL_GRACE_S", 0.6)
    state, lock = _new_state()
    pids, ready = [], threading.Event()
    script = "trap '' TERM; sleep 300 & echo child=$!; echo ready; wait"
    job = xe._job_start("baysor", "stub Baysor", _fake_reseg_job(xe, state, lock, "feedbeef", script, pids, ready),
                        state=state, lock=lock)
    assert ready.wait(20)
    leader = job.procs[0].pid
    t0 = time.time()
    assert xe._job_cancel(job.id)
    time.sleep(0.25)
    assert _pid_alive(leader), "SIGTERM is ignored by this tool"
    job.thread.join(30)
    assert job.state == "cancelled" and time.time() - t0 >= 0.5
    _wait(lambda: not any(_pid_alive(p) for p in pids + [leader]), what="SIGKILL to take effect")
    assert os.listdir(fake_cache) == []


def test_a_failing_tool_is_an_error_not_a_cancel_and_leaves_no_run_directory(xe, fake_cache):
    state, lock = _new_state()
    job = xe._job_start("proseg", "stub", _fake_reseg_job(xe, state, lock, "deadbeef", "echo ready; exit 3",
                                                          [], threading.Event()),
                        state=state, lock=lock)
    job.thread.join(30)
    assert job.state == "error" and state["status"] == "error" and "code 3" in state["message"]
    assert os.listdir(fake_cache) == [] and xe._list_cached_seg_runs() == []


def test_run_process_reports_lines_split_at_cr_and_strips_ansi(xe, capsys):
    seen = []
    script = r"printf 'one\r\033[32mtwo\033[0m\rthree\nfour'"
    rc = xe._job_run_process(["/bin/sh", "-c", script], log_tag="t",
                             on_line=lambda ln: seen.append(ln) or ln == "two")
    assert rc == 0 and seen == ["one", "two", "three", "four"]
    out = capsys.readouterr().out
    assert "[t] one" in out and "[t] four" in out and "[t] two" not in out, "on_line can keep a line out of the log"


def test_run_process_with_a_pty_gives_the_tool_a_terminal(xe):
    seen = []
    rc = xe._job_run_process(["/bin/sh", "-c", "if [ -t 1 ]; then echo tty; else echo pipe; fi"],
                             log_tag="t", on_line=lambda ln: seen.append(ln) and False, use_pty=True)
    assert rc == 0 and seen == ["tty"]
    seen.clear()
    xe._job_run_process(["/bin/sh", "-c", "if [ -t 1 ]; then echo tty; else echo pipe; fi"],
                        log_tag="t", on_line=lambda ln: seen.append(ln) and False)
    assert seen == ["pipe"]


# ── progress parsers (lines as printed by Baysor 0.7.1 and Proseg 1.1.9) ─────
def test_baysor_progress_parser(xe):
    got = []
    on_line = xe._baysor_progress_parser(lambda f, t: got.append((round(f, 3), t)))
    lines = ["[01:38:03] Info: (2026-11-12) Run Baysor v0.7.1",
             "[01:38:03] Info: Loading data...",
             "[01:38:07] Info: Estimating noise level",
             "Progress:  50%|████████▍    |  ETA: 0:00:00",         # not the iterations: ignored
             "[01:38:23] Info: Clustering molecules...",
             "[01:38:24] Info: Initializing algorithm. Scale: 5.0, scale std: 1.25, initial #components: 216, #molecules: 540.",
             "Progress:  20%|████████▍                                |  ETA: 0:00:00",
             "         Iteration: 102",
             "Progress: 100%|█████████████████████████████████████████| Time: 0:00:00",
             "[01:38:25] Info: Processing complete.",
             "[01:38:36] Info: Estimating boundary polygons",
             "[01:38:36] Info: Saving results to /x/segmentation.csv",
             "Progress:  20%|████████▎                                |  ETA: 0:00:01",   # saving: ignored
             "[01:38:40] Info: All done!"]
    assert not any(on_line(ln) for ln in lines), "Baysor lines all stay in the log"
    fracs = [f for f, _ in got]
    assert fracs == sorted(fracs), "progress never goes backwards"
    assert (0.25, "segmentation iterations 20%") in got and (0.85, "segmentation iterations 100%") in got
    assert got[0] == (0.02, "loading data") and got[-1] == (1.0, "done")
    assert len([1 for _, t in got if t.startswith("segmentation iterations ")]) == 2


def test_proseg_progress_parser(xe):
    got = []
    on_line = xe._proseg_progress_parser(lambda f, t: got.append((f, t)))
    bar = "00:00:09 " + "#" * 30 + "-" * 30 + " | log-likelihood: -3043.7363 | assigned: 495 / 525 (94.29%) | non-background: (83.81%)"
    assert on_line(bar) is True, "bar redraws are kept out of the log"
    assert on_line("00:00:00 " + "#" * 60 + " |") is True
    assert on_line("Using 16 threads") is False and on_line("Read 525 transcripts") is False
    assert got == [(0.5, "sampling 50%"), (1.0, "sampling 100%")]


# ── the panel ────────────────────────────────────────────────────────────────
def _find(component, pred, out=None):
    out = [] if out is None else out
    if pred(component):
        out.append(component)
    kids = getattr(component, "children", None)
    for kid in (kids if isinstance(kids, (list, tuple)) else [kids]):
        if hasattr(kid, "to_plotly_json"):
            _find(kid, pred, out)
    return out


def _texts(component):
    return [c.children for c in _find(component, lambda c: isinstance(getattr(c, "children", None), str))]


def test_panel_is_in_the_layout_with_its_interval_off(xe):
    ids = {str(getattr(c, "id", None)): c for c in _find(xe.app.layout, lambda c: True)}
    assert {"jobs-panel", "jobs-panel-summary", "jobs-panel-rows", "jobs-poll"} <= set(ids)
    assert ids["jobs-poll"].disabled is True, "no polling while no job runs"
    assert ids["jobs-panel"].style == {"display": "none"}


def test_panel_shows_job_with_progress_and_cancel_then_switches_its_interval_off(xe, monkeypatch):
    from dash import no_update
    children, summary, style, poll_off = xe.update_jobs_panel(0)
    assert children == [] and summary == [] and style == {"display": "none"} and poll_off is True
    assert xe.wake_jobs_poll(*[None] * 12, True) is no_update, "nothing to wake up for"

    state, lock = _new_state()
    entered, release = threading.Event(), threading.Event()

    def body():
        xe._job_progress("Proseg patch 2/4: sampling 50%", 0.3375)
        entered.set()
        release.wait(30)
        with lock:
            state.update(status="done", message="ok")

    job = xe._job_start("proseg", "Proseg segmentation", body, state=state, lock=lock)
    assert entered.wait(10)
    try:
        assert xe.wake_jobs_poll(*[None] * 12, True) is False, "a start switches the interval on"
        assert xe.wake_jobs_poll(*[None] * 12, False) is no_update
        children, summary, style, poll_off = xe.update_jobs_panel(1)
        assert poll_off is False and style["position"] == "fixed" and style["zIndex"] > 1055
        import dash.html as html
        assert _texts(html.Div(summary)) == ["Jobs", "1 running"]
        panel = html.Div(children)
        rows = _find(panel, lambda c: getattr(c, "className", None) == "job-row")
        assert len(rows) == 1 and getattr(rows[0], "data-job") == job.id
        texts = _texts(rows[0])
        assert "proseg" in texts and "Proseg segmentation" in texts
        assert "Proseg patch 2/4: sampling 50%" in texts and "34%" in texts
        bar = _find(rows[0], lambda c: getattr(c, "className", None) == "job-progress-fill")
        assert len(bar) == 1 and bar[0].style["width"] == "33.8%"
        elapsed = _find(rows[0], lambda c: getattr(c, "className", None) == "job-elapsed")[0].children
        assert elapsed in ("0:00", "0:01")
        buttons = _find(rows[0], lambda c: isinstance(getattr(c, "id", None), dict))
        assert [b.id for b in buttons] == [{"type": "job-cancel-btn", "job": job.id}]
        assert buttons[0].children == "Cancel"

        # cancelling: no button any more, and the text says why it is not instant
        assert xe._job_cancel(job.id)
        row = _find(html.Div(xe.update_jobs_panel(2)[0]), lambda c: getattr(c, "className", None) == "job-row")[0]
        assert not _find(row, lambda c: isinstance(getattr(c, "id", None), dict))
        assert "cancelling after the current step…" in _texts(row)
    finally:
        release.set()
    job.thread.join(10)
    # finished jobs linger for a few seconds, then the panel hides and stops its interval
    children, summary, style, poll_off = xe.update_jobs_panel(3)
    assert poll_off is False and "✓ done" in _texts(html.Div(children))
    assert _texts(html.Div(summary)) == ["Jobs", "finished"]
    monkeypatch.setattr(xe, "_JOB_LINGER_S", 0.0)
    assert xe.update_jobs_panel(4) == ([], [], {"display": "none"}, True)


def test_panel_row_without_fraction_has_no_bar_and_a_shielded_job_no_button(xe):
    import dash.html as html
    row = xe._job_row({"id": "x-1", "kind": "split", "label": "SPLIT correction", "name": "SPLIT",
                       "state": "running", "step": "Running RCTD…", "fraction": None, "elapsed": 3725.0,
                       "n_procs": 0, "shielded": False, "cancellable": True})
    assert not _find(row, lambda c: getattr(c, "className", None) == "job-progress")
    assert "1:02:05" in _texts(row)
    row = xe._job_row({"id": "x-2", "kind": "spage", "label": "SpaGE", "name": "SpaGE",
                       "state": "running", "step": "Saving…", "fraction": None, "elapsed": 1.0,
                       "n_procs": 0, "shielded": True, "cancellable": False})
    assert not _find(row, lambda c: isinstance(getattr(c, "id", None), dict))
    assert any("cannot be cancelled" in t for t in _texts(row))
    row = xe._job_row({"id": "x-3", "kind": "baysor", "label": "Baysor", "name": "Baysor",
                       "state": "cancelling", "step": "…", "fraction": 0.5, "elapsed": 1.0,
                       "n_procs": 1, "shielded": False, "cancellable": False})
    assert "stopping the tool…" in _texts(row)
    row = xe._job_row({"id": "x-4", "kind": "proseg", "label": "Proseg", "name": "Proseg",
                       "state": "running", "step": "Proseg patch 1/33: sampling 17%", "fraction": 0.0046,
                       "elapsed": 63.0, "n_procs": 1, "shielded": False, "cancellable": True})
    assert "0.5%" in _texts(row), "below 10 % the percentage keeps one decimal"
    assert _find(row, lambda c: getattr(c, "className", None) == "job-progress-fill")[0].style["width"] == "0.5%"


# ── every job kind goes through the registry ─────────────────────────────────
def test_no_job_is_started_outside_the_registry():
    """The job functions are only ever handed to _job_start (never to a bare Thread)."""
    import ast
    tree = tl.app_tree()
    job_funcs = {"_run_baysor", "_run_proseg", "_run_celltypist", "_run_seurat_annotation",
                 "_run_rctd_annotation", "_run_split_correction", "_run_spage_imputation",
                 "_run_reseg_umap", "_run_spatial_analysis", "_run_spatial_niches",
                 "_load_cellnest_data", "_save_sdata_to_disk", "_load_cached_baysor",
                 "_load_cached_proseg", "_run_de", "_run_geneset_score", "_run_density",
                 "_run_pseudobulk_de", "_run_moran", "_run_lr", "_run_compartments",
                 "_run_spillover", "_run_segqc", "_run_coloc"}
    started = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        # a job function is named bare in its own module and as module.function elsewhere
        names = {n.id if isinstance(n, ast.Name) else n.attr
                 for a in list(node.args) + [k.value for k in node.keywords]
                 for n in ast.walk(a) if isinstance(n, (ast.Name, ast.Attribute))}
        if name == "Thread":
            assert not (names & job_funcs), f"line {node.lineno}: job started with a bare Thread"
        if name == "_job_start":
            started |= names & job_funcs
            kws = {k.arg for k in node.keywords}
            # (None: a **kwargs expansion, used where three calls share state / lock)
            assert {"state", "lock"} <= kws or None in kws, \
                f"line {node.lineno}: _job_start without state/lock"
    # the two cached-run loaders are passed through a variable
    assert job_funcs - started <= {"_load_cached_baysor", "_load_cached_proseg"}, job_funcs - started
