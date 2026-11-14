"""R in a child process: the long R steps of a job run in a worker that Cancel terminates.

R runs inside the Python process through rpy2 and an R call cannot be interrupted from
another thread, so a job whose R step runs in the app only sees Cancel when R returns.
The steps below therefore run in a worker: a second Python process that executes this
very file as a script (it never imports the app, which would load the dataset again),
loads rpy2 and R itself and runs one *target* function per request. The worker is
started with jobs._job_popen(), i.e. it is a registered child process of the job like
Baysor or Proseg: Cancel (and the shutdown hooks) send SIGTERM to its process group,
which ends R at once, with whatever R started.

    parent (job thread)                         worker (python rworker.py <dir> <token>)
    _rw_call(target, payload)  -- stdin: one JSON line per request (target, number) -->
                               -- <dir>/req-N.json + .npz (the payload) -------------->
                               <-- stdout: R console output, and marker lines for
                                   "progress", "emit", "done", "error" ---------------
                               <-- <dir>/res-N.json + .npz (the result) --------------

* The exchange directory is private (tempfile.mkdtemp, mode 0700) and is removed when
  the worker is closed, cancelled or fails, and at exit. Files are JSON and .npz read
  with allow_pickle=False: nothing is unpickled, and no code is read from a file (the
  target is named on the worker's stdin, the R code is part of this file).
* A target is a module-level function ``target(payload: dict, ctx) -> dict`` of this
  file (or of another file that can be loaded on its own: no relative imports). The
  same function is called directly when R runs inside the app (XENIUM_R_INPROCESS=1, or
  the worker could not start), so both paths execute the same Python and R statements.
  Payload and result are flat dicts of JSON values, numpy arrays and scipy CSR / CSC
  matrices.
* _rw_session() keeps one worker for several calls of a job (the 1 + n label shuffles of
  the contact test fit one model each: R and DESeq2 are loaded once).

Run as a script this file is the worker; imported as xenium_app.rworker it is the
parent side.
"""
import os
import sys

if __spec__ is None:
    # run as a script (python rworker.py …), i.e. as the worker: the package directory,
    # which Python put first on the path, must not shadow other imports
    sys.path[:] = [_p for _p in sys.path
                   if os.path.abspath(_p or os.getcwd()) != os.path.dirname(os.path.abspath(__file__))]

import collections
import contextlib
import importlib.util
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
import numpy as np

if __package__:
    from . import jobs


_RW_SCRIPT = os.path.abspath(__file__)   # the script the worker runs
_RW_PYTHON = None                        # interpreter of the worker (None: this one)
_RW_MARK = "@@XE_RW"                     # marker of the worker's protocol lines on stdout
_RW_START_TIMEOUT_S = 60.0               # the worker must report "ready" within this time
_RW_QUIT_WAIT_S = 3.0                    # a worker asked to quit is terminated after this
_RW_KILL_GRACE_S = 1.0                   # SIGTERM → SIGKILL for a worker that is closed
_RW_FALLBACK_NOTE = ("the R worker process could not start ({reason}): R ran inside the app, "
                     "where Cancel only takes effect once the R call returns")


class _RwError(RuntimeError):
    """A failure of the target in the worker (an R error above all). str() is the message
    of the original exception, so a job reports it as it reports the in-process one;
    .kind is the class name of the original exception and .formatted its last traceback
    line(s) ("package.Class: message"), for the jobs that report that line."""

    def __init__(self, message: str, kind: str = "RuntimeError", formatted: str = ""):
        super().__init__(message)
        self.kind = kind
        self.formatted = formatted


class _RwRefused(ValueError):
    """Raised by a target for a condition the user can act on (missing R package, unknown
    label column, …): the message goes to the status line as it is."""


class _RwStartError(RuntimeError):
    """The worker process could not be started (the caller then runs R in-process)."""


class _RwGone(Exception):
    """The worker's output ended: the process is gone."""


def _rw_inprocess() -> bool:
    """True when R is to run inside the app process (XENIUM_R_INPROCESS=1)."""
    return os.environ.get("XENIUM_R_INPROCESS", "").strip().lower() in ("1", "true", "yes", "on")


def _rw_thread_setup() -> None:
    """Called by a callback or job before it uses R. With R inside the app: the rpy2
    conversion rules of the calling thread (rpy2 keeps them in a context variable that a
    new thread does not inherit). With the worker: only that rpy2 is installed - R is
    never started in the app process. Raises what `import rpy2` raises when it fails."""
    if _rw_inprocess():
        import rpy2.robjects as _ro_mod
        import rpy2.robjects.conversion as _rconv
        _rconv.set_conversion(_ro_mod.default_converter)
    elif importlib.util.find_spec("rpy2") is None:
        raise ModuleNotFoundError("No module named 'rpy2'")


# ─── Exchange format ──────────────────────────────────────────────────────────
def _rw_pack(obj: dict) -> tuple:
    """(JSON text, {name: array}) of a payload / result: a flat dict whose values are
    JSON values (str, numbers, bool, None, lists and dicts of them), numpy arrays (no
    object dtype) or scipy CSR / CSC matrices."""
    if not isinstance(obj, dict) or not all(isinstance(k, str) for k in obj):
        raise TypeError("an R worker payload / result is a dict with string keys")
    values, arrays, sparse = {}, {}, {}
    for k, v in obj.items():
        if isinstance(v, np.ndarray):
            if v.dtype.hasobject:
                raise TypeError(f"{k!r}: arrays of Python objects cannot be passed to the R worker")
            arrays[k] = v
        elif hasattr(v, "indptr") and hasattr(v, "format"):
            if v.format not in ("csr", "csc"):
                raise TypeError(f"{k!r}: only CSR / CSC matrices can be passed to the R worker")
            sparse[k] = {"format": v.format, "shape": [int(v.shape[0]), int(v.shape[1])]}
            arrays[k + ".data"], arrays[k + ".indices"], arrays[k + ".indptr"] = v.data, v.indices, v.indptr
        else:
            values[k] = v
    return json.dumps({"values": values, "arrays": sorted(set(obj) & set(arrays)), "sparse": sparse}), arrays


def _rw_unpack(text: str, arrays) -> dict:
    """The dict that _rw_pack() was given (lists for tuples; arrays as stored)."""
    head = json.loads(text)
    out = dict(head["values"])
    for k in head["arrays"]:
        out[k] = arrays[k]
    if head["sparse"]:
        import scipy.sparse as sp_
        for k, info in head["sparse"].items():
            cls = sp_.csr_matrix if info["format"] == "csr" else sp_.csc_matrix
            out[k] = cls((arrays[k + ".data"], arrays[k + ".indices"], arrays[k + ".indptr"]),
                         shape=tuple(info["shape"]))
    return out


def _rw_normalise(obj: dict) -> dict:
    """*obj* as it would arrive through the files, without writing any (the in-process
    path hands targets and callers the same types as the worker path)."""
    return _rw_unpack(*_rw_pack(obj))


def _rw_write(dirpath: str, stem: str, obj: dict) -> None:
    text, arrays = _rw_pack(obj)
    if arrays:
        np.savez(os.path.join(dirpath, stem + ".npz"), **arrays)
    with open(os.path.join(dirpath, stem + ".json"), "w", encoding="utf-8") as fh:
        fh.write(text)


def _rw_read(dirpath: str, stem: str) -> dict:
    """Read and remove one exchange object. Arrays are loaded into memory (they own it)."""
    jpath, npath = os.path.join(dirpath, stem + ".json"), os.path.join(dirpath, stem + ".npz")
    with open(jpath, encoding="utf-8") as fh:
        text = fh.read()
    arrays = {}
    if os.path.exists(npath):
        with np.load(npath, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    for path in (jpath, npath):
        try:
            os.remove(path)
        except OSError:
            pass
    return _rw_unpack(text, arrays)


class _RwCtx:
    """What a target gets besides its payload: progress(message, fraction=None) for the
    job's step text, emit(name, data) for an intermediate result (a flat dict like the
    payload) that the caller's on_event(name, data) receives while the target goes on."""

    def __init__(self, progress, emit):
        self.progress = progress
        self.emit = emit


# ─── R versions (provenance) ──────────────────────────────────────────────────
_rw_tls = threading.local()      # .session: the open _RwSession of this thread; .versions


def _rw_r_versions(packages) -> dict:
    """R and R package versions of the R running in THIS process; {} when rpy2 was not
    imported here (R was not used)."""
    if "rpy2.robjects" not in sys.modules:
        return {}
    out = {}
    try:
        ro = sys.modules["rpy2.robjects"]
        import rpy2.robjects.conversion as _rconv
        _rconv.set_conversion(ro.default_converter)
        out["R"] = str(ro.r("R.version.string")[0]).replace("R version ", "")
        for p in packages or ():
            try:
                out[p] = str(ro.r(f'as.character(utils::packageVersion("{p}"))')[0])
            except Exception:
                out[p] = "not found"
    except Exception as exc:
        out.setdefault("R", f"version not read ({type(exc).__name__})")
    return out


def _rw_versions_store() -> dict:
    """Versions reported by the workers of the current job (of this thread outside a job)."""
    job = jobs._job_current()
    if job is not None:
        return job.__dict__.setdefault("_rw_versions", {})
    if not hasattr(_rw_tls, "versions"):
        _rw_tls.versions = {}
    return _rw_tls.versions


def _rw_versions_seen(packages) -> dict:
    """R and package versions that the R worker of the current job reported ({} when the
    job used no worker). A package the worker was not asked about reads "not found"."""
    seen = _rw_versions_store()
    if "R" not in seen:
        return {}
    return {"R": seen["R"], **{p: seen.get(p, "not found") for p in packages or ()}}


# ─── Parent side ──────────────────────────────────────────────────────────────
_rw_live_dirs: set = set()       # exchange directories of the open sessions


def _rw_remove_dirs() -> None:
    """At exit (after the workers were stopped): no exchange directory is left behind."""
    for d in list(_rw_live_dirs):
        shutil.rmtree(d, ignore_errors=True)
        _rw_live_dirs.discard(d)


if __package__:
    jobs._job_exit_hooks.append(_rw_remove_dirs)


def _rw_target_ref(target) -> tuple:
    """(file, name) naming *target* for the worker; file "-" for a function of this file."""
    code = getattr(target, "__code__", None)
    name = getattr(target, "__name__", "")
    if code is None or getattr(target, "__qualname__", name) != name:
        raise TypeError("an R worker target is a module-level function")
    path = os.path.abspath(code.co_filename)
    return ("-" if path == os.path.abspath(__file__) else path), name


class _RwSession:
    """One worker process and its exchange directory. Started at the first call."""

    def __init__(self):
        self.proc = None
        self.dir = None
        self.failed = None           # why the worker could not start (calls then run in-process)
        self.n = 0
        self._mark = ""
        self._pending = ""
        self._lines = collections.deque()
        self._tail = collections.deque(maxlen=12)
        self._eof = False
        self._decoder = None

    # -- life cycle --
    def start(self) -> None:
        import codecs
        import secrets
        self.dir = tempfile.mkdtemp(prefix="xe-rworker-")
        _rw_live_dirs.add(self.dir)
        token = secrets.token_hex(8)
        self._mark = f"{_RW_MARK}:{token}@@ "
        self._pending, self._eof = "", False
        self._lines.clear()
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            self.proc = jobs._job_popen([_RW_PYTHON or sys.executable, "-u", _RW_SCRIPT, self.dir, token],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT)
        except jobs._DatasetChanged:
            self.close(kill=True)
            raise
        except Exception as exc:
            self.close(kill=True)
            raise _RwStartError(f"{type(exc).__name__}: {exc}") from None
        try:
            ev = self._next_event(timeout=_RW_START_TIMEOUT_S)
            if ev.get("event") != "ready":
                raise _RwGone()
        except (_RwGone, TimeoutError) as exc:
            code = self.proc.poll() if self.proc is not None else None
            tail = " | ".join(self._tail)[-300:]
            self.close(kill=True)
            jobs._job_check()                       # terminated by Cancel: not a start failure
            why = ("no answer within %.0f s" % _RW_START_TIMEOUT_S if isinstance(exc, TimeoutError)
                   else f"it ended with exit code {code} before it was ready")
            raise _RwStartError(why + (f": {tail}" if tail else "")) from None
        except BaseException:
            self.close(kill=True)
            raise

    def close(self, kill: bool = False) -> None:
        """Stop the worker (asked to quit, terminated when it does not, or at once with
        kill=True), reap it, take it off the job registry and remove the directory."""
        proc, self.proc = self.proc, None
        if proc is not None:
            try:
                if not kill and proc.poll() is None:
                    try:
                        proc.stdin.write(b'{"op": "quit"}\n')
                        proc.stdin.flush()
                        proc.stdin.close()
                        proc.wait(timeout=_RW_QUIT_WAIT_S)
                    except (OSError, ValueError, subprocess.TimeoutExpired):
                        pass
            finally:
                jobs._job_proc_done(proc, grace=_RW_KILL_GRACE_S)
                for pipe in (proc.stdin, proc.stdout):
                    try:
                        pipe.close()
                    except (OSError, ValueError):
                        pass
        d, self.dir = self.dir, None
        if d:
            shutil.rmtree(d, ignore_errors=True)
            _rw_live_dirs.discard(d)

    # -- protocol --
    def _next_event(self, timeout: float = None) -> dict:
        """The worker's next protocol event; its other output lines go to the log.
        Raises _RwGone when the output ended, TimeoutError after *timeout* seconds and
        _JobCancelled when Cancel was requested for the job of this thread."""
        import select
        fd = self.proc.stdout.fileno()
        t_end = None if timeout is None else time.time() + timeout
        while True:
            while self._lines:
                line = self._lines.popleft()
                i = line.find(self._mark)
                if i >= 0:
                    try:
                        return json.loads(line[i + len(self._mark):])
                    except ValueError:
                        pass
                line = line.rstrip()
                if line:
                    self._tail.append(line)
                    print(f"  [R] {line}", flush=True)
            if self._eof:
                raise _RwGone()
            if t_end is not None and time.time() > t_end:
                raise TimeoutError()
            try:
                ready, _, _ = select.select([fd], [], [], 0.25)
            except (OSError, ValueError):
                self._eof = True
                continue
            if not ready:
                jobs._job_check()
                if self.proc.poll() is not None and not select.select([fd], [], [], 0)[0]:
                    self._eof = True
                continue
            try:
                data = os.read(fd, 65536)
            except OSError:
                data = b""
            if not data:
                self._eof = True
                if self._pending:
                    self._lines.append(self._pending)
                    self._pending = ""
                continue
            self._pending += self._decoder.decode(data)
            parts = re.split(r"[\r\n]", self._pending)
            self._pending = parts.pop()
            self._lines.extend(parts)

    def call(self, target, payload: dict, on_progress=None, on_event=None, r_packages=()) -> dict:
        """Run target(payload, ctx) in the worker and return its result. Raises the
        target's failure as ValueError / _RwRefused (its own refusals) or _RwError (any
        other exception, with the original message), _JobCancelled when the job was
        cancelled (the worker is terminated), and _RwError when the worker died."""
        file, name = _rw_target_ref(target)
        self.n += 1
        n = self.n
        failure = None
        try:
            _rw_write(self.dir, f"req-{n}", payload)
            try:
                self.proc.stdin.write((json.dumps({"op": "call", "n": n, "file": file, "name": name,
                                                   "r_packages": list(r_packages or ())}) + "\n").encode())
                self.proc.stdin.flush()
            except (OSError, ValueError):
                raise _RwGone() from None
            while True:
                ev = self._next_event()
                kind = ev.get("event")
                if kind == "progress":
                    if on_progress is not None:
                        on_progress(ev.get("message"), ev.get("fraction"))
                elif kind == "emit":
                    data = _rw_read(self.dir, f"evt-{ev['k']}")
                    if on_event is not None:
                        on_event(ev.get("name"), data)
                elif kind == "done" and ev.get("n") == n:
                    _rw_versions_store().update(ev.get("versions") or {})
                    return _rw_read(self.dir, f"res-{n}")
                elif kind == "error" and ev.get("n") == n:
                    failure = ev
                    break
        except _RwGone:
            code = self.proc.poll() if self.proc is not None else None
            tail = " | ".join(self._tail)[-300:]
            self.close(kill=True)
            jobs._job_check()                       # Cancel terminated it
            raise _RwError("The R worker process ended unexpectedly"
                           + (f" (exit code {code})" if code is not None else "")
                           + (f": {tail}" if tail else ""), kind="WorkerDied") from None
        except BaseException:
            # Cancel, a dataset swap or a failing callback while R is still running
            self.close(kill=True)
            raise
        message = str(failure.get("message") or failure.get("kind") or "R worker error")
        if failure.get("refused"):
            raise _RwRefused(message)
        if failure.get("value_error"):
            raise ValueError(message)
        raise _RwError(message, kind=str(failure.get("kind") or "RuntimeError"),
                       formatted=str(failure.get("formatted") or ""))


def _rw_call_inprocess(target, payload: dict, on_progress=None, on_event=None) -> dict:
    """target(payload, ctx) in this process, with the types of the worker path."""
    ctx = _RwCtx(lambda message, fraction=None: on_progress(message, fraction) if on_progress else None,
                 lambda name, data: on_event(name, _rw_normalise(data)) if on_event else None)
    return _rw_normalise(target(_rw_normalise(payload), ctx) or {})


def _rw_call(target, payload: dict, on_progress=None, on_event=None, r_packages=(), on_note=None) -> dict:
    """Run one R step: target(payload, ctx) in the R worker of the current job, or inside
    the app with XENIUM_R_INPROCESS=1 and when the worker cannot start (on_note(text)
    then gets a sentence for the status line).

    on_progress(message, fraction) and on_event(name, data) are called in the calling
    thread while the target runs; r_packages names the R packages whose versions the
    provenance record of the job will ask for. Inside `with _rw_session():` the calls of
    a thread share one worker; otherwise each call has its own."""
    if _rw_inprocess():
        return _rw_call_inprocess(target, payload, on_progress, on_event)
    sess = getattr(_rw_tls, "session", None)
    own = sess is None
    if own:
        sess = _RwSession()
    try:
        if sess.proc is None and sess.failed is None:
            try:
                sess.start()
            except _RwStartError as exc:
                sess.failed = str(exc)
                note = _RW_FALLBACK_NOTE.format(reason=sess.failed)
                print(f"  R worker: {note}", flush=True)
                if on_note is not None:
                    on_note(note)
        if sess.failed is not None:
            return _rw_call_inprocess(target, payload, on_progress, on_event)
        return sess.call(target, payload, on_progress, on_event, r_packages)
    finally:
        if own:
            sess.close()


@contextlib.contextmanager
def _rw_session():
    """Inside the block the _rw_call()s of this thread share one worker (R and its
    packages are loaded once); it is closed, and its directory removed, at the end."""
    if _rw_inprocess() or getattr(_rw_tls, "session", None) is not None:
        yield
        return
    sess = _rw_tls.session = _RwSession()
    try:
        yield
    finally:
        _rw_tls.session = None
        sess.close()


# ─── Sparse matrices to R and back ────────────────────────────────────────────
def _r_dgcmatrix_to_scipy(r_mat):
    """Convert an R dgCMatrix to a scipy CSC matrix (genes × cells).

    dgCMatrix is column-sparse (CSC): p = column pointers, i = row indices.
    Callers that need cells × genes should call .T.tocsr() on the result.
    """
    import scipy.sparse as sp_
    i = np.array(r_mat.slots["i"])
    p = np.array(r_mat.slots["p"])
    x = np.array(r_mat.slots["x"])
    dims = list(r_mat.slots["Dim"])
    return sp_.csc_matrix((x, i, p), shape=(dims[0], dims[1]))


def _scipy_csc_parts(mat_cells_x_genes):
    """genes × cells CSC parts (i, p, x, shape) for an R dgCMatrix.
    i and p are 0-based int32 (dgCMatrix slots are 0-based), x is float64."""
    import scipy.sparse as sp_
    m = sp_.csc_matrix(mat_cells_x_genes.T, dtype=np.float64, copy=True)
    m.sum_duplicates()
    m.eliminate_zeros()
    m.sort_indices()
    if m.nnz >= 2**31 - 1:
        raise RuntimeError(f"Matrix has {m.nnz:,} non-zeros — too many for an R dgCMatrix "
                           f"(limit 2^31-1). Subset the cells first.")
    return (np.ascontiguousarray(m.indices, dtype=np.int32),
            np.ascontiguousarray(m.indptr, dtype=np.int32),
            np.ascontiguousarray(m.data, dtype=np.float64),
            m.shape)


def _scipy_to_r_dgc(mat_cells_x_genes, rownames, colnames, r_name: str) -> None:
    """Assign a cells × genes scipy matrix to the R global `r_name` as a
    genes × cells dgCMatrix, passing only the sparse parts (never dense)."""
    import rpy2.robjects as ro
    i, p, x, shape = _scipy_csc_parts(mat_cells_x_genes)
    if len(rownames) != shape[0] or len(colnames) != shape[1]:
        raise ValueError(f"dimnames {len(rownames)}×{len(colnames)} do not match matrix {shape}")
    ro.r.assign("._dgc_i", ro.IntVector(i))
    ro.r.assign("._dgc_p", ro.IntVector(p))
    ro.r.assign("._dgc_x", ro.FloatVector(x))
    ro.r.assign("._dgc_rn", ro.StrVector(list(rownames)))
    ro.r.assign("._dgc_cn", ro.StrVector(list(colnames)))
    ro.r(f"""
{r_name} <- Matrix::sparseMatrix(i=as.integer(._dgc_i), p=as.integer(._dgc_p),
    x=as.numeric(._dgc_x), dims=c({int(shape[0])}L, {int(shape[1])}L), index1=FALSE,
    dimnames=list(._dgc_rn, ._dgc_cn))
rm(._dgc_i, ._dgc_p, ._dgc_x, ._dgc_rn, ._dgc_cn)
""")


def _rw_console() -> None:
    """In the worker: R's console output as plain lines on stdout (rpy2 would prefix
    warnings with "R[write to console]"), like core._redirect_rpy2_console in the app.
    Called by the R targets; does nothing when the target runs inside the app."""
    if __spec__ is not None:
        return
    try:
        import rpy2.rinterface_lib.callbacks as rpy2_cb

        def _r_write(text):
            sys.stdout.write(text)
            sys.stdout.flush()

        rpy2_cb.consolewrite_print = _r_write
        rpy2_cb.consolewrite_warnerror = _r_write
    except Exception:
        pass


# ─── Target: the pseudobulk model (stats._pb_fit) ─────────────────────────────
_PB_R_FUNCS = r"""
.xe_pb_deseq2 <- function(counts, genes, samples, cond, batch, shrink) {
  suppressPackageStartupMessages(library(DESeq2))
  storage.mode(counts) <- "integer"
  dimnames(counts) <- list(genes, samples)
  cd <- data.frame(condition = factor(cond, levels = c("B", "A")), row.names = samples)
  if (!is.null(batch)) { cd$batch <- factor(batch); dsg <- ~ batch + condition } else dsg <- ~ condition
  dds <- suppressMessages(DESeqDataSetFromMatrix(countData = counts, colData = cd, design = dsg))
  note <- ""
  dds <- tryCatch(suppressMessages(DESeq(dds, quiet = TRUE)), error = function(e) {
    note <<- paste0("DESeq() stopped (", conditionMessage(e), "): gene-wise dispersion estimates ",
                    "were used without the trend fit")
    d <- tryCatch(estimateSizeFactors(dds), error = function(e2) estimateSizeFactors(dds, type = "poscounts"))
    d <- suppressMessages(estimateDispersionsGeneEst(d))
    dispersions(d) <- mcols(d)$dispGeneEst
    suppressMessages(nbinomWaldTest(d))
  })
  res <- results(dds, name = "condition_A_vs_B", alpha = 0.05)
  shr <- rep(NA_real_, nrow(res)); shr_type <- "none"
  if (isTRUE(shrink)) {
    for (ty in c(if (requireNamespace("apeglm", quietly = TRUE)) "apeglm", "normal")) {
      s <- tryCatch(suppressMessages(lfcShrink(dds, coef = "condition_A_vs_B", type = ty, quiet = TRUE)),
                    error = function(e) NULL)
      if (!is.null(s)) { shr <- s$log2FoldChange; shr_type <- ty; break }
    }
  }
  list(baseMean = as.numeric(res$baseMean), log2FoldChange = as.numeric(res$log2FoldChange),
       lfcSE = as.numeric(res$lfcSE), stat = as.numeric(res$stat), pvalue = as.numeric(res$pvalue),
       padj = as.numeric(res$padj), shrunk = as.numeric(shr), shrink_type = shr_type,
       size_factors = as.numeric(sizeFactors(dds)), note = note,
       version = as.character(packageVersion("DESeq2")))
}
.xe_pb_edger <- function(counts, genes, samples, cond, batch) {
  suppressPackageStartupMessages(library(edgeR))
  dimnames(counts) <- list(genes, samples)
  condition <- factor(cond, levels = c("B", "A"))
  design <- if (!is.null(batch)) model.matrix(~ factor(batch) + condition) else model.matrix(~ condition)
  y <- DGEList(counts = counts)
  y <- calcNormFactors(y)
  y <- estimateDisp(y, design)
  fit <- glmQLFit(y, design)
  qlf <- glmQLFTest(fit, coef = ncol(design))
  tt <- qlf$table
  list(log2FoldChange = as.numeric(tt$logFC), logCPM = as.numeric(tt$logCPM), stat = as.numeric(tt$F),
       pvalue = as.numeric(tt$PValue), padj = as.numeric(p.adjust(tt$PValue, method = "BH")),
       size_factors = as.numeric(y$samples$lib.size * y$samples$norm.factors / 1e6), note = "",
       version = as.character(packageVersion("edgeR")))
}
"""


def _rw_pb_fit(p: dict, ctx) -> dict:
    """The R part of stats._pb_fit: fit the pseudobulk model with DESeq2 or edgeR.

    p: counts (genes × samples, int64), genes, labels, cond, batch (list or None), shrink,
    engine ("auto" | "deseq2" | "edger"), install_hint (the text of the refusal).
    Returns engine, size_factors, log2fc, stat, pval, padj, note, version, shrink_type and,
    for DESeq2, lfc_se, base_mean and log2fc_shrunk (when a shrinkage estimator ran)."""
    try:
        import rpy2.robjects as ro
        import rpy2.robjects.conversion as _rconv
        from rpy2.robjects.packages import isinstalled
    except Exception as exc:
        raise ValueError(f"rpy2 / R is not available ({exc}). " + p["install_hint"]) from None
    _rw_console()
    _rconv.set_conversion(ro.default_converter)          # rpy2 thread rule: unconditional
    counts = np.asarray(p["counts"], dtype=np.int64)
    n_g, n_s = counts.shape
    engine = p["engine"]
    with _rconv.localconverter(ro.default_converter):
        have = {pk: bool(isinstalled(pk)) for pk in ("DESeq2", "edgeR")}
        if engine == "auto":
            engine = "deseq2" if have["DESeq2"] else ("edger" if have["edgeR"] else None)
        if engine is None or not have["DESeq2" if engine == "deseq2" else "edgeR"]:
            raise ValueError(p["install_hint"])
        ro.r(_PB_R_FUNCS)
        r_counts = ro.r["matrix"](ro.IntVector(counts.ravel(order="F").tolist()), nrow=n_g, ncol=n_s)
        r_batch = ro.NULL if p.get("batch") is None else ro.StrVector([str(b) for b in p["batch"]])
        args = (r_counts, ro.StrVector(list(p["genes"])), ro.StrVector(list(p["labels"])),
                ro.StrVector(list(p["cond"])), r_batch)
        if engine == "deseq2":
            res = ro.globalenv[".xe_pb_deseq2"](*args, bool(p["shrink"]))
        else:
            res = ro.globalenv[".xe_pb_edger"](*args)
        # a copy: np.asarray() of an R numeric vector is a view of R's memory, which R
        # reuses once the vector is collected (size factors read after a later R call
        # came back as garbage)
        get = lambda k: np.array(res.rx2(k), dtype=np.float64, copy=True)
        txt = lambda k: str(res.rx2(k)[0])
        out = {"engine": engine, "size_factors": get("size_factors"),
               "log2fc": get("log2FoldChange"), "stat": get("stat"), "pval": get("pvalue"),
               "padj": get("padj"), "note": txt("note"), "version": txt("version"),
               "shrink_type": "none"}
        if engine == "deseq2":
            out["lfc_se"], out["base_mean"] = get("lfcSE"), get("baseMean")
            out["shrink_type"] = txt("shrink_type")
            if out["shrink_type"] != "none":
                out["log2fc_shrunk"] = get("shrunk")
    return out


# ─── Target: RCTD cell type annotation (annotation._run_rctd_annotation) ──────
def _rw_rctd(p: dict, ctx) -> dict:
    """The R part of the RCTD annotation: reference, SpatialRNA, run.RCTD, labels, weights.

    p: rds_path, label_col, mode ("full" | "doublet" | "multi"), n_cores, umi_min,
    umi_min_sigma, genes (the panel genes, the columns of expr), expr (cells × genes,
    sparse), cell_ids, cx, cy (float64).
    Returns cell_ids, types (one label per kept cell) and, in full mode, weights
    (cells × types), weight_cells, weight_types - or weights_error."""
    import rpy2.robjects as ro
    import rpy2.robjects.conversion as _rconv
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.rinterface_lib import openrlib

    _rw_console()
    # Ensure rpy2 conversion rules are active in this thread
    _rconv.set_conversion(ro.default_converter)
    pandas2ri.activate()
    rds_path, label_col, mode = p["rds_path"], p["label_col"], p["mode"]
    xenium_genes, cell_ids = list(p["genes"]), list(p["cell_ids"])
    out = {}
    try:
        with openrlib.rlock:
            # ── 1. Check spacexr ─────────────────────────────────────────────
            ctx.progress("Loading spacexr…")
            try:
                importr("spacexr")
            except Exception as _spacexr_err:
                print(f"  RCTD: importr('spacexr') failed: {_spacexr_err}", flush=True)
                raise _RwRefused(f"spacexr not found by rpy2: {str(_spacexr_err)[:200]}") from None

            importr("SeuratObject")
            importr("Matrix")
            base = importr("base")

            # ── 2. Load Seurat reference ─────────────────────────────────────
            ctx.progress("Loading Seurat reference…")
            rds = base.readRDS(rds_path)
            meta_r  = ro.r['slot'](rds, "meta.data")
            meta_df = pandas2ri.rpy2py(meta_r)
            if label_col not in meta_df.columns:
                raise _RwRefused(f"Column '{label_col}' not found. Available: {list(meta_df.columns)[:10]}")

            # ── 3. Determine shared genes ────────────────────────────────────
            ctx.progress("Finding shared genes…")
            ro.r.assign("._rctd_rds", rds)
            mat_r     = ro.r("SeuratObject::LayerData(._rctd_rds[['RNA']], layer='counts')")
            ref_genes = list(ro.r['rownames'](mat_r))
            shared_genes = sorted(set(xenium_genes) & set(ref_genes))
            if len(shared_genes) < 10:
                raise _RwRefused(f"Only {len(shared_genes)} shared genes between reference and panel.")
            print(f"  RCTD: {len(shared_genes)} shared genes", flush=True)

            # ── 4. Build Reference object ────────────────────────────────────
            ctx.progress(f"Building RCTD Reference ({len(meta_df):,} ref cells)…")
            ro.r.assign("._rctd_genes", ro.StrVector(shared_genes))
            ro.r("""
._rctd_ref_mat <- SeuratObject::LayerData(._rctd_rds[['RNA']], layer='counts')[._rctd_genes, , drop=FALSE]
._rctd_ref_mat <- as(._rctd_ref_mat, 'dgCMatrix')
._rctd_ref_mat@x <- as.numeric(round(._rctd_ref_mat@x))
""")
            ref_labels_r       = ro.StrVector(meta_df[label_col].astype(str).tolist())
            ref_labels_r.names = ro.StrVector(list(meta_df.index.astype(str)))
            ro.r.assign("._rctd_ref_labels", ref_labels_r)
            ro.r("""
._rctd_ref_factor <- as.factor(._rctd_ref_labels)
._rctd_numi_ref   <- colSums(._rctd_ref_mat)
._rctd_reference  <- spacexr::Reference(
    counts     = ._rctd_ref_mat,
    cell_types = ._rctd_ref_factor,
    nUMI       = ._rctd_numi_ref
)
""")

            # ── 5. Build SpatialRNA from Xenium ──────────────────────────────
            ctx.progress("Building SpatialRNA object…")
            gene_idx   = {g: i for i, g in enumerate(xenium_genes)}
            shared_idx = [gene_idx[g] for g in shared_genes]
            # shared_genes × cells dgCMatrix built from the sparse parts
            # (no MatrixMarket text round trip)
            _scipy_to_r_dgc(p["expr"][:, shared_idx], shared_genes, cell_ids, "._rctd_xen_mat")
            ro.r.assign("._rctd_cx", ro.FloatVector(np.ascontiguousarray(p["cx"], dtype=np.float64)))
            ro.r.assign("._rctd_cy", ro.FloatVector(np.ascontiguousarray(p["cy"], dtype=np.float64)))
            ro.r("""
._rctd_coords   <- data.frame(x=._rctd_cx, y=._rctd_cy, row.names=colnames(._rctd_xen_mat))
._rctd_numi_xen <- colSums(._rctd_xen_mat)
._rctd_puck     <- spacexr::SpatialRNA(
    coords = ._rctd_coords,
    counts = ._rctd_xen_mat,
    nUMI   = ._rctd_numi_xen
)
""")

            # ── 6. Create and run RCTD ───────────────────────────────────────
            ctx.progress(f"Running RCTD ({mode} mode, {p['n_cores']} cores)…")
            ro.r.assign("._rctd_mode",    mode)
            ro.r.assign("._rctd_cores",   p["n_cores"])
            ro.r.assign("._rctd_umi_min",       ro.IntVector([p["umi_min"]]))
            ro.r.assign("._rctd_umi_min_sigma", ro.IntVector([p["umi_min_sigma"]]))
            ro.r("""
._rctd_obj <- spacexr::create.RCTD(._rctd_puck, ._rctd_reference, max_cores=._rctd_cores,
    UMI_min=._rctd_umi_min[1], UMI_min_sigma=._rctd_umi_min_sigma[1])
._rctd_obj <- spacexr::run.RCTD(._rctd_obj, doublet_mode=._rctd_mode)
""")
            print("  RCTD: run complete — extracting results…", flush=True)

            # ── 7. Extract per-cell labels ────────────────────────────────────
            ctx.progress("Extracting RCTD results…")
            if mode == "full":
                ro.r("""
._rctd_weights <- ._rctd_obj@results$weights
._rctd_types   <- colnames(._rctd_weights)[apply(._rctd_weights, 1, which.max)]
names(._rctd_types) <- rownames(._rctd_weights)
""")
            elif mode == "multi":
                # multi mode: @results is a per-pixel list (no results_df), in the
                # column order of the filtered SpatialRNA counts
                ro.r("""
._rctd_res   <- ._rctd_obj@results
._rctd_types <- vapply(._rctd_res, function(r)
    if (length(r$cell_type_list)) as.character(r$cell_type_list[1]) else NA_character_,
    character(1))
names(._rctd_types) <- colnames(._rctd_obj@spatialRNA@counts)[seq_along(._rctd_types)]
._rctd_types <- ._rctd_types[!is.na(._rctd_types)]
""")
            else:  # doublet
                ro.r("""
._rctd_res   <- ._rctd_obj@results$results_df
._rctd_types <- as.character(._rctd_res$first_type)
names(._rctd_types) <- rownames(._rctd_res)
""")

            types_r   = ro.r("._rctd_types")
            names_r   = ro.r("names(._rctd_types)")
            out["cell_ids"] = [str(v) for v in names_r]
            out["types"]    = [str(v) for v in types_r]
            if len(out["cell_ids"]) == 0:
                raise RuntimeError(f"RCTD ({mode}) returned no cell labels")

            # ── 8. Extract the weight matrix (full mode only) ─────────────────
            if mode == "full":
                ctx.progress("Extracting RCTD weight matrix…")
                try:
                    ro.r("._rctd_weights_dense <- as.matrix(._rctd_weights)")
                    out["weight_types"] = [str(v) for v in ro.r("colnames(._rctd_weights)")]
                    out["weight_cells"] = [str(v) for v in ro.r("rownames(._rctd_weights)")]
                    out["weights"]      = np.array(ro.r("._rctd_weights_dense"))
                except Exception as _we:
                    for k in ("weight_types", "weight_cells", "weights"):
                        out.pop(k, None)
                    out["weights_error"] = str(_we)
    finally:
        # Cleanup R temporaries on every exit path (best-effort).
        # The names start with a dot, so ls() needs all.names=TRUE.
        try:
            with openrlib.rlock:
                ro.r("rm(list=ls(pattern='^[.]_rctd_', all.names=TRUE, envir=globalenv()), "
                     "envir=globalenv()); invisible(gc())")
        except Exception:
            pass
    return out


# ─── Target: SPLIT (split._run_split_correction) ──────────────────────────────
def _rw_split(p: dict, ctx) -> dict:
    """The R part of SPLIT: RCTD in doublet mode (or its cached object) and SPLIT::purify.

    p: rds_path, label_col, panel_genes (the columns of expr), expr (cells × panel genes,
    sparse), cell_ids, x, y (float64), max_cores, min_umi, min_umi_sigma, purify_singlets,
    rctd_obj_file (where the RCTD object is cached), rctd_cached (the caller holds the
    labels of that object and they belong to these cells).
    Emits "rctd_from_cache" ({}) when the cached object was loaded, else "rctd_labels"
    ({cell_ids, types}) once RCTD is done and its object is saved. Returns purified
    (genes × cells CSC), cell_ids and gene_ids of its columns / rows (None when R gave no
    dimnames)."""
    import rpy2.robjects as ro
    import rpy2.robjects.conversion as _rconv
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.rinterface_lib import openrlib

    _rw_console()
    # rpy2 thread fix
    _rconv.set_conversion(ro.default_converter)
    pandas2ri.activate()
    rds_path, label_col = p["rds_path"], p["label_col"]
    panel_genes, cell_ids = list(p["panel_genes"]), list(p["cell_ids"])
    expr_panel = p["expr"]
    n_cells = expr_panel.shape[0]
    max_cores, min_umi, min_umi_sigma = p["max_cores"], p["min_umi"], p["min_umi_sigma"]
    _rctd_obj_file = p["rctd_obj_file"]
    with openrlib.rlock:
        # ── 2. Check R packages ──────────────────────────────────────────────
        ctx.progress("Loading R packages (spacexr, SPLIT)…")
        try:
            importr("spacexr")
        except Exception:
            raise _RwRefused("spacexr not installed. In R: devtools::install_github('dmcable/spacexr')") from None
        try:
            importr("SPLIT")
        except Exception:
            raise _RwRefused("SPLIT not installed. In R: remotes::install_github('bdsc-tds/SPLIT')") from None
        importr("SeuratObject")
        importr("Matrix")
        base = importr("base")

        # ── 3. Load Seurat reference ─────────────────────────────────────────
        ctx.progress("Loading Seurat reference…")
        rds = base.readRDS(rds_path)
        meta_r  = ro.r['slot'](rds, "meta.data")
        meta_df = pandas2ri.rpy2py(meta_r)
        if label_col not in meta_df.columns:
            raise _RwRefused(f"Column '{label_col}' not found. Available: {list(meta_df.columns)[:10]}")

        # ── 4. Shared genes ──────────────────────────────────────────────────
        ctx.progress("Finding shared genes…")
        ro.r.assign("._split_rds", rds)
        mat_r     = ro.r("SeuratObject::LayerData(._split_rds[['RNA']], layer='counts')")
        ref_genes = list(ro.r['rownames'](mat_r))
        shared_genes = sorted(set(panel_genes) & set(ref_genes))
        if len(shared_genes) < 10:
            raise _RwRefused(f"Only {len(shared_genes)} shared genes between reference and panel.")
        print(f"  SPLIT: {len(shared_genes)} shared genes", flush=True)

        # ── 5. Build Reference ───────────────────────────────────────────────
        ctx.progress(f"Building RCTD Reference ({len(meta_df):,} ref cells)…")
        ro.r.assign("._split_genes", ro.StrVector(shared_genes))
        ro.r("""
._split_ref_mat <- SeuratObject::LayerData(._split_rds[['RNA']], layer='counts')[._split_genes, , drop=FALSE]
._split_ref_mat <- as(._split_ref_mat, 'dgCMatrix')
._split_ref_mat@x <- as.numeric(round(._split_ref_mat@x))
""")
        ref_labels_r       = ro.StrVector(meta_df[label_col].astype(str).tolist())
        ref_labels_r.names = ro.StrVector(list(meta_df.index.astype(str)))
        ro.r.assign("._split_ref_labels", ref_labels_r)
        ro.r("""
._split_ref_factor <- as.factor(._split_ref_labels)
._split_numi_ref   <- colSums(._split_ref_mat)
._split_reference  <- spacexr::Reference(
    counts      = ._split_ref_mat,
    cell_types  = ._split_ref_factor,
    nUMI        = ._split_numi_ref
)
""")

        # ── 6. Build SpatialRNA (panel genes only) ───────────────────────────
        ctx.progress(f"Building SpatialRNA object ({n_cells:,} cells)…")
        # Subset expr_panel to shared genes
        _panel_pos = {g: i for i, g in enumerate(panel_genes)}
        shared_idx = [_panel_pos[g] for g in shared_genes if g in _panel_pos]
        shared_genes_present = [g for g in shared_genes if g in _panel_pos]
        expr_shared = expr_panel[:, shared_idx]  # cells × shared_genes
        # gene × cells dgCMatrix for R, built from the sparse parts
        _scipy_to_r_dgc(expr_shared, shared_genes_present, cell_ids, "._split_counts_mat")

        ro.r.assign("._split_x", ro.FloatVector(np.asarray(p["x"], dtype=np.float64).tolist()))
        ro.r.assign("._split_y", ro.FloatVector(np.asarray(p["y"], dtype=np.float64).tolist()))
        ro.r("""
._split_coords <- data.frame(x=._split_x, y=._split_y)
rownames(._split_coords) <- colnames(._split_counts_mat)
._split_numi <- colSums(._split_counts_mat)
._split_spatialrna <- spacexr::SpatialRNA(
    coords  = ._split_coords,
    counts  = ._split_counts_mat,
    nUMI    = ._split_numi
)
""")

        # ── 7. Run RCTD (doublet mode) — with caching ────────────────────────
        _rctd_from_cache = False
        if p.get("rctd_cached"):
            try:
                # path passed as a value (it contains the reference's file name)
                ro.globalenv["._split_rctd_path"] = _rctd_obj_file
                ro.r('._split_rctd <- readRDS(._split_rctd_path); rm(._split_rctd_path)')
                _rctd_from_cache = True
            except Exception as _ce:
                print(f"  SPLIT: RCTD cache load failed: {_ce}", flush=True)
            if _rctd_from_cache:
                ctx.emit("rctd_from_cache", {})
                ctx.progress("RCTD loaded from cache, running SPLIT::purify()…")
                print("  SPLIT: RCTD loaded from cache", flush=True)

        if not _rctd_from_cache:
            ctx.progress(f"Running RCTD doublet mode (max_cores={max_cores})… [~20-40 min]")
            print("  SPLIT: starting RCTD doublet mode…", flush=True)
            ro.r.assign("._split_max_cores",    ro.IntVector([max_cores]))
            ro.r.assign("._split_min_umi",      ro.IntVector([min_umi]))
            ro.r.assign("._split_min_umi_sigma", ro.IntVector([min_umi_sigma]))
            ro.r("""
options(mc.cores = ._split_max_cores[1])
cat("  SPLIT R: detected cores =", parallel::detectCores(),
    "| requesting =", ._split_max_cores[1], "\n")
._split_rctd <- spacexr::create.RCTD(._split_spatialrna, ._split_reference,
    max_cores=._split_max_cores[1], CELL_MIN_INSTANCE=5,
    UMI_min=._split_min_umi[1], UMI_min_sigma=._split_min_umi_sigma[1])
._split_rctd <- spacexr::run.RCTD(._split_rctd, doublet_mode='doublet')
._split_rctd <- SPLIT::run_post_process_RCTD(._split_rctd)
""")
            # Save the RCTD object for future re-runs and hand its labels to the caller.
            # Use as.character() in R to avoid rpy2 converting factor levels to
            # integer codes (which would show numbers instead of cell type names).
            _rctd_labels = None
            try:
                ro.globalenv["._split_rctd_path"] = _rctd_obj_file
                ro.r('saveRDS(._split_rctd, ._split_rctd_path); rm(._split_rctd_path)')
                ro.r("""
._split_rctd_res   <- ._split_rctd@results$results_df
._split_rctd_types <- as.character(._split_rctd_res$first_type)
names(._split_rctd_types) <- rownames(._split_rctd_res)
""")
                _types_r = ro.r("._split_rctd_types")
                _names_r = ro.r("names(._split_rctd_types)")
                _rctd_labels = {"cell_ids": [str(v) for v in _names_r],
                                "types": [str(v) for v in _types_r]}
            except Exception as _se:
                print(f"  SPLIT: RCTD cache save failed: {_se}", flush=True)
            if _rctd_labels is not None:
                ctx.emit("rctd_labels", _rctd_labels)

        print("  SPLIT: RCTD done, running purify()…", flush=True)

        # ── 8. Run SPLIT::purify ─────────────────────────────────────────────
        ctx.progress("Running SPLIT::purify()…")
        # Full counts matrix (all panel genes) gene × cells
        _scipy_to_r_dgc(expr_panel, list(panel_genes), cell_ids, "._split_full_counts")
        ro.r.assign("._split_do_purify_singlets", ro.BoolVector([p["purify_singlets"]]))
        ro.r("""
._split_res <- SPLIT::purify(
    counts          = ._split_full_counts,
    rctd            = ._split_rctd,
    DO_purify_singlets = ._split_do_purify_singlets[1]
)
._split_purified <- ._split_res$purified_counts
""")

        # ── 9. Convert back to Python ────────────────────────────────────────
        ctx.progress("Converting corrected counts to Python…")
        purified_r = ro.r["._split_purified"]
        # Extract cell IDs from R colnames — RCTD may drop low-UMI cells
        try:
            corrected_cell_ids = [str(v) for v in ro.r['colnames'](purified_r)]
        except Exception:
            corrected_cell_ids = None
        # purify() returns only the genes RCTD used (panel ∩ reference), not the panel
        try:
            corrected_gene_ids = [str(v) for v in ro.r['rownames'](purified_r)]
        except Exception:
            corrected_gene_ids = None
        # dgCMatrix → scipy CSC (genes × cells)
        purified = _r_dgcmatrix_to_scipy(purified_r)
    return {"purified": purified, "cell_ids": corrected_cell_ids, "gene_ids": corrected_gene_ids}


# ─── Worker side ──────────────────────────────────────────────────────────────
_rw_target_modules: dict = {}


def _rw_resolve(file: str, name: str):
    """The target function *name* of this file ("-") or of the Python file *file*."""
    if file == "-":
        fn = globals().get(name)
    else:
        mod = _rw_target_modules.get(file)
        if mod is None:
            spec = importlib.util.spec_from_file_location(f"_rw_target_{len(_rw_target_modules)}", file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _rw_target_modules[file] = mod
        fn = getattr(mod, name, None)
    if not callable(fn):
        raise LookupError(f"R worker target {name!r} not found")
    return fn


def _rw_worker_main(argv) -> int:
    """The worker: answer the requests read from stdin until it is closed."""
    import traceback
    dirpath, token = argv[1], argv[2]
    mark = f"{_RW_MARK}:{token}@@ "

    def send(**event) -> None:
        # on a line of its own, whatever R left unfinished on the current one
        sys.stdout.write("\n" + mark + json.dumps(event) + "\n")
        sys.stdout.flush()

    send(event="ready", pid=os.getpid())
    emitted = [0]

    def progress(message, fraction=None) -> None:
        send(event="progress", message=str(message), fraction=fraction)

    def emit(name, data) -> None:
        emitted[0] += 1
        _rw_write(dirpath, f"evt-{emitted[0]}", data)
        send(event="emit", name=str(name), k=emitted[0])

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("op") != "call":
            break
        n = req["n"]
        try:
            fn = _rw_resolve(req["file"], req["name"])
            result = fn(_rw_read(dirpath, f"req-{n}"), _RwCtx(progress, emit))
            _rw_write(dirpath, f"res-{n}", result or {})
            send(event="done", n=n, versions=_rw_r_versions(req.get("r_packages") or ()))
        except Exception as exc:
            if not isinstance(exc, ValueError):
                traceback.print_exc(file=sys.stdout)
            send(event="error", n=n, kind=type(exc).__name__, message=str(exc),
                 formatted="".join(traceback.format_exception_only(type(exc), exc)),
                 refused=type(exc).__name__ == "_RwRefused", value_error=isinstance(exc, ValueError))
    return 0


if __spec__ is None:          # run as a script: this process is the worker
    sys.exit(_rw_worker_main(sys.argv))
