"""Dataset generation counter and the job registry (progress, Cancel, tool processes)."""

import os
import re
import threading
import subprocess
import contextvars

from . import core, caching


# ─── Dataset generation ───────────────────────────────────────────────────────
# Bumped by _install_dataset() (Load Sample). A background job captures the value
# when it starts and stops publishing once it has changed, so results computed for
# the previous dataset never reach the new dataset's state, DATA or cache files.
_dataset_gen = 0


class _DatasetChanged(Exception):
    """Raised inside a background job whose dataset was replaced while it ran."""


def _check_dataset_gen(gen: int) -> None:
    """Raise _DatasetChanged when the dataset was replaced since *gen* was captured.
    Jobs call this right before they publish or write a result, so it is also a
    cancellation point: _JobCancelled (a _DatasetChanged) when Cancel was requested
    for the job running in this thread."""
    _job_check()
    if gen != _dataset_gen:
        raise _DatasetChanged("dataset was replaced while this job was running — result discarded")


def _stale_job(gen: int, status: str, message=None) -> bool:
    """Guard for a job's _set(): False while the job's dataset is still loaded.
    After a dataset swap an "error" report is dropped (returns True); any other
    status raises _DatasetChanged so the job unwinds through its own error handler
    instead of writing old-dataset results.

    Also the job registry's hook into every job: a "running" report is a step
    boundary, so its *message* becomes the job's step text and a requested Cancel is
    honoured there (_JobCancelled); the "error" a cancelled job reports while it
    unwinds is dropped (_job_finish writes "Cancelled" instead)."""
    job = _job_current()
    if job is not None:
        if job.cancel.is_set() and not job.shielded:
            if status == "error":
                job.cancel_raised = True
                return True
            if status == "running":
                _job_check()
        elif status == "running" and message:
            job.step = str(message)
    if gen == _dataset_gen:
        return False
    if status == "error":
        return True
    _check_dataset_gen(gen)
    return True


# ─── Job registry ─────────────────────────────────────────────────────────────
# Every long-running background job (Baysor, Proseg, patch preparation, annotation,
# SPLIT, SpaGE, UMAP, spatial analysis, niches, CellNEST load, SpatialData export) is
# started through _job_start() and is listed here with its thread, its child
# processes, a step text, an optional fraction done and a cancel flag. The jobs panel
# (update_jobs_panel) and _running_jobs() read from it.
#
# Cancellation is cooperative for Python / R code: the flag is checked at step
# boundaries (_set("running", …) via _stale_job, _check_dataset_gen, _job_check), so a
# library or R call that is in progress finishes first. External tools are started
# with _job_run_process() in their own process group, which Cancel terminates
# (SIGTERM, then SIGKILL after _JOB_KILL_GRACE_S). Once a job starts to write its
# result it calls _job_shield(): from there it runs to completion.
import itertools as _itertools
import time as _time
import signal as _signal
import atexit as _atexit


class _JobCancelled(_DatasetChanged):
    """Raised inside a background job at a cancellation point after Cancel was requested.
    A _DatasetChanged, so the handlers that let a stale job unwind also let this through."""


_JOB_KILL_GRACE_S = 5.0     # SIGTERM → SIGKILL
_JOB_LINGER_S     = 8.0     # a finished job stays in the panel this long
_JOB_KEEP         = 50      # finished jobs kept in the registry
_JOB_ACTIVE       = ("running", "cancelling")

# kind → name used in the "Still running: …" refusal of Load Sample / Clean Cache
_JOB_KIND_NAMES = {
    "baysor": "Baysor", "proseg": "Proseg", "imgseg": "image segmentation", "annotation": "annotation", "spage": "SpaGE",
    "patches": "SpatialData / patches", "umap": "UMAP", "split": "SPLIT",
    "niches": "niches", "spatial": "spatial analysis", "save": "Save as SpatialData",
    "cellnest": "CellNEST load", "de": "differential expression",
    "geneset": "gene-set scoring", "density": "transcript density",
    "moran": "spatially variable genes", "lr": "ligand-receptor scoring",
    "compartments": "tissue compartments", "spillover": "spillover score",
    "segqc": "segmentation QC", "coloc": "stratified co-localisation",
    "proximity": "proximity differential expression", "contact": "contact-group differential expression",
    "lrdecomp": "ligand-receptor exposure decomposition", "aggregates": "aggregates",
    "segfree": "segmentation-free abundance", "txfactor": "transcript factors", "banksy": "Banksy clustering", "joint": "joint clustering",
}

_jobs: dict = {}                       # job id → _Job, in start order
_jobs_lock = threading.RLock()
_job_seq = _itertools.count(1)
_job_local = threading.local()         # .job: the _Job running in this thread


class _Job:
    """One background job. state: running → (cancelling →) done | error | cancelled."""

    def __init__(self, kind: str, label: str, state_dict=None, state_lock=None, name=None):
        self.id       = f"{kind}-{next(_job_seq)}"
        self.kind     = kind
        self.label    = label
        self.name     = name or _JOB_KIND_NAMES.get(kind, kind)
        self.started  = _time.time()
        self.ended    = None
        self.state    = "running"
        self.step     = "Starting…"
        self.fraction = None           # 0..1 when known
        self.thread   = None
        self.procs    = []             # live child processes (subprocess.Popen)
        self.cancel   = threading.Event()
        self.cancel_raised = False     # the job saw the flag and is unwinding
        self.cancelled = False         # final outcome, set by _job_finish
        self.shielded = False          # writing its result: no longer cancellable
        self.error    = None
        self.gen      = _dataset_gen
        self.state_dict = state_dict   # the job kind's status dict (e.g. _baysor_state)
        self.state_lock = state_lock
        self.cleanup  = []             # callables(job) run when the job ends

    def elapsed(self) -> float:
        return (self.ended or _time.time()) - self.started

    def active(self) -> bool:
        return self.state in _JOB_ACTIVE

    def snapshot(self) -> dict:
        with _jobs_lock:
            return {"id": self.id, "kind": self.kind, "label": self.label, "name": self.name,
                    "state": self.state, "step": self.step, "fraction": self.fraction,
                    "elapsed": self.elapsed(), "started": self.started, "ended": self.ended,
                    "n_procs": len(self.procs), "shielded": self.shielded,
                    "cancellable": self.state == "running" and not self.shielded}


def _job_current():
    """The _Job running in the calling thread, or None."""
    return getattr(_job_local, "job", None)


def _job_check() -> None:
    """Cancellation point: raise _JobCancelled when Cancel was requested for the job
    running in this thread (no-op outside a job and once the job is shielded)."""
    job = _job_current()
    if job is not None and job.cancel.is_set() and not job.shielded:
        job.cancel_raised = True
        raise _JobCancelled("Cancelled")


_JOB_KEEP_FRACTION = object()


def _job_progress(step=None, fraction=_JOB_KEEP_FRACTION, job=None) -> None:
    """Report the current step text and / or the fraction done (None: unknown) of the
    job running in this thread. Never raises; use _job_check() for cancellation."""
    job = job or _job_current()
    if job is None:
        return
    if step is not None:
        job.step = str(step)
    if fraction is not _JOB_KEEP_FRACTION:
        job.fraction = None if fraction is None else float(min(1.0, max(0.0, fraction)))


def _job_step(state: dict, lock, message: str, fraction=_JOB_KEEP_FRACTION) -> None:
    """Step boundary of a job that keeps its status dict by hand (no _set helper):
    cancellation point, then the message goes to the dict and to the job."""
    _job_check()
    with lock:
        state["message"] = message
    _job_progress(message, fraction)


def _job_shield(step=None) -> None:
    """Last cancellation point of a job: called right before it starts to write its
    result (cache files, zarr write-back, DATA). Raises _JobCancelled when Cancel was
    already requested; afterwards the job cannot be cancelled and runs to completion,
    so no half-written result is left behind."""
    job = _job_current()
    if job is None:
        return
    with _jobs_lock:
        if job.cancel.is_set() and not job.shielded:
            job.cancel_raised = True
            raise _JobCancelled("Cancelled")
        job.shielded = True
    if step is not None:
        job.step = str(step)


def _job_on_end(fn) -> None:
    """Register fn(job) to run when the current job ends (any outcome)."""
    job = _job_current()
    if job is not None:
        job.cleanup.append(fn)


def _job_start(kind: str, label: str, target, args=(), kwargs=None, *,
               state=None, lock=None, name=None, copy_context: bool = False) -> "_Job":
    """Start *target* in a registered daemon thread and return its _Job.

    state / lock: the job kind's status dict and its lock. When the job ends after a
    Cancel, the dict is set to status "error" / message "Cancelled", which is what the
    existing polls treat as "finished, nothing loaded": the state is released and the
    same job can be started again.
    copy_context: run inside a copy of the caller's contextvars context (rpy2
    conversion rules set by the callback, see start_annotation)."""
    kwargs = kwargs or {}
    job = _Job(kind, label, state, lock, name)
    if copy_context:
        _ctx = contextvars.copy_context()
        call = lambda: _ctx.run(target, *args, **kwargs)
    else:
        call = lambda: target(*args, **kwargs)

    def _job_main():
        _job_local.job = job
        try:
            call()
        except _JobCancelled:
            job.cancel_raised = True
        except _DatasetChanged:
            pass
        except BaseException as exc:      # a job without its own handler
            import traceback
            traceback.print_exc()
            job.error = str(exc)[:300] or type(exc).__name__
        finally:
            try:
                _job_finish(job)
            finally:
                _job_local.job = None

    job.thread = threading.Thread(target=_job_main, daemon=True, name=f"xe-job-{job.id}")
    with _jobs_lock:
        _jobs[job.id] = job
        _done = [j for j in _jobs.values() if not j.active()]
        for old in _done[:max(0, len(_done) - _JOB_KEEP)]:
            _jobs.pop(old.id, None)
    job.thread.start()
    return job


def _job_finish(job: "_Job") -> None:
    """End of a job's thread: stop leftover child processes, run its clean-up hooks,
    release the job kind's state after a cancel and record the final state."""
    for proc in list(job.procs):
        _kill_proc_tree(proc)
    cancel_asked = job.cancel.is_set() and not job.shielded
    status = None
    if job.state_dict is not None and job.state_lock is not None:
        with job.state_lock:
            status = job.state_dict.get("status")
            # still "running" / "loading": the job stopped at a cancellation point (or
            # died); "error": it failed while unwinding. Anything else: it completed.
            unfinished = status in ("running", "starting", "loading", "error")
            cancelled = cancel_asked and unfinished
            if job.gen == _dataset_gen:
                if cancelled:
                    job.state_dict["status"]  = "error"
                    job.state_dict["message"] = "Cancelled"
                elif job.error and status in ("running", "starting", "loading"):
                    job.state_dict["status"]  = "error"
                    job.state_dict["message"] = job.error
            status = job.state_dict.get("status")
    else:
        cancelled = cancel_asked and job.cancel_raised
    job.cancelled = cancelled
    for fn in job.cleanup:
        try:
            fn(job)
        except Exception as exc:
            print(f"  job {job.id}: clean-up failed: {exc}", flush=True)
    with _jobs_lock:
        job.ended = _time.time()
        if cancelled:
            job.state = "cancelled"
            job.step  = "Cancelled"
        elif job.error or status == "error":
            job.state = "error"
            if job.state_dict is not None:
                job.step = str(job.state_dict.get("message") or job.error or "Failed")[:200]
        else:
            job.state = "done"
            job.fraction = 1.0 if job.fraction is not None else None
    if cancelled:
        print(f"  {job.label}: cancelled after {job.elapsed():.0f} s", flush=True)


def _job_cancel(job_id: str) -> bool:
    """Request cancellation of a running job. Sets its flag and terminates its child
    processes (in a helper thread: SIGTERM, SIGKILL after the grace period). Returns
    False when there is no such running job or it is already writing its result."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None or job.state != "running" or job.shielded:
            return False
        job.cancel.set()
        job.state = "cancelling"
        procs = list(job.procs)
    print(f"  {job.label}: cancel requested"
          + (f" — stopping {len(procs)} process(es)" if procs else
             " — stops after the current step"), flush=True)
    if procs:
        threading.Thread(target=lambda: [_kill_proc_tree(p) for p in procs],
                         daemon=True, name=f"xe-job-kill-{job.id}").start()
    return True


def _jobs_list(active_only: bool = False) -> list:
    """The registered jobs in start order (finished ones included unless active_only)."""
    with _jobs_lock:
        return [j for j in _jobs.values() if j.active() or not active_only]


def _jobs_panel_rows(now=None) -> list:
    """Snapshots of the jobs the panel shows: active ones and those that ended less
    than _JOB_LINGER_S ago."""
    now = now or _time.time()
    with _jobs_lock:
        return [j.snapshot() for j in _jobs.values()
                if j.active() or (j.ended and now - j.ended < _JOB_LINGER_S)]


def _proc_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True      # exists but not signalable (macOS: only zombies left)


def _kill_proc_tree(proc, grace: float = None) -> None:
    """Stop *proc* and its children. Processes started by _job_run_process() lead
    their own process group: the group gets SIGTERM and, for whatever is left after
    *grace* seconds, SIGKILL (Julia workers, shell wrappers and their children
    included). Without process groups (Windows) only the process itself is stopped."""
    grace = _JOB_KILL_GRACE_S if grace is None else grace
    use_group = hasattr(os, "killpg") and getattr(proc, "_xe_new_group", False)

    def _send(sig):
        try:
            if use_group:
                os.killpg(proc.pid, sig)
            else:
                proc.send_signal(sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def _gone():
        if proc.poll() is None:
            return False
        return not (use_group and _proc_group_alive(proc.pid))

    if _gone():
        return
    _send(_signal.SIGTERM)
    deadline = _time.time() + grace
    while _time.time() < deadline:
        if _gone():
            return
        _time.sleep(0.05)
    _send(getattr(_signal, "SIGKILL", _signal.SIGTERM))
    try:
        proc.wait(timeout=10)
    except Exception:
        pass


def _job_run_process(cmd, cwd=None, log_tag: str = "", on_line=None, use_pty: bool = False,
                     env=None) -> int:
    """Run an external tool for the current job and return its exit code.

    The process is started in its own session / process group and registered with the
    job, so Cancel can terminate it together with its children. Output (stdout +
    stderr) is split at newlines and carriage returns, stripped of ANSI codes and
    printed as "  [log_tag] line"; on_line(line) sees every line first and returns
    True to keep it out of the log (progress-bar redraws).
    use_pty: give the tool a pseudo-terminal, for tools that only draw their progress
    bar on a terminal (Proseg); falls back to a pipe where there is none.
    Raises _JobCancelled when the job was cancelled while the tool ran."""
    import codecs, select
    job = _job_current()
    master = None
    popen_kw = dict(cwd=cwd, stdin=subprocess.DEVNULL, env=env)
    new_group = hasattr(os, "setsid")
    if new_group:
        popen_kw["start_new_session"] = True
    if use_pty:
        try:
            import pty
            master, slave = pty.openpty()
        except Exception:
            master = None
    with _jobs_lock:
        # under the lock: a Cancel either sees the process or is seen here
        try:
            _job_check()
            if master is not None:
                proc = subprocess.Popen(cmd, stdout=slave, stderr=slave, **popen_kw)
            else:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        **popen_kw)
        except BaseException:
            if master is not None:
                os.close(master)
                os.close(slave)
            raise
        proc._xe_new_group = new_group
        _job_procs_all.add(proc)
        if job is not None:
            job.procs.append(proc)
    if master is not None:
        os.close(slave)
    fd = master if master is not None else proc.stdout.fileno()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""

    def _emit(line):
        line = core._strip_ansi(line).rstrip()
        if not line:
            return
        try:
            if on_line is not None and on_line(line):
                return
        except Exception:
            pass
        print(f"  [{log_tag}] {line}", flush=True)

    try:
        while True:
            try:
                ready, _, _ = select.select([fd], [], [], 0.5)
            except (OSError, ValueError):
                break
            if not ready:
                if proc.poll() is not None:
                    break       # exited; nothing more buffered
                continue
            try:
                data = os.read(fd, 65536)
            except OSError:     # pty: EIO once the child side is closed
                break
            if not data:
                break
            pending += decoder.decode(data)
            parts = re.split(r"[\r\n]", pending)
            pending = parts.pop()
            for part in parts:
                _emit(part)
        if pending:
            _emit(pending)
        proc.wait()
    finally:
        if proc.poll() is None or (new_group and _proc_group_alive(proc.pid)):
            _kill_proc_tree(proc)
        with _jobs_lock:
            _job_procs_all.discard(proc)
            if job is not None and proc in job.procs:
                job.procs.remove(proc)
        try:
            if master is not None:
                os.close(master)
            else:
                proc.stdout.close()
        except OSError:
            pass
    _job_check()
    return proc.returncode


_job_procs_all: set = set()     # every live tool process (also those started outside a job)


def _jobs_kill_all_processes() -> None:
    """At interpreter exit: tools run in their own session, so Ctrl-C in the terminal
    no longer reaches them - stop them here."""
    for proc in list(_job_procs_all):
        _kill_proc_tree(proc, grace=2.0)
    for stage in list(caching._seg_stage_active):     # the runs these tools were working on
        caching._seg_stage_discard(stage)


_atexit.register(_jobs_kill_all_processes)
