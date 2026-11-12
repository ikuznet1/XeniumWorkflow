"""Helpers shared by the test modules.

Everything here is deliberately independent of the application code: the wire-format
decoder, the raw-bundle readers and the recount helpers are written from the file
formats, not by calling into xenium_explorer / xenium_viewer_api, so a test that
compares "app value" with "helper value" is a real cross-check.
"""
from __future__ import annotations

import ast
import io
import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tarfile
import textwrap
import threading
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = pathlib.Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"
TINY_BUNDLE = TESTS / "data" / "prime_mouse_ileum_tiny"
APP_SOURCE = REPO / "xenium_explorer.py"          # the entry point (and `import xenium_explorer`)
APP_PACKAGE = REPO / "xenium_app"                 # where the code lives

# Bundles larger than this are linked into the working directory instead of copied.
COPY_LIMIT_BYTES = 200 * 1024 * 1024
_GENERATED = ("spatialdata_xenium.zarr", "spatialdata_xenium.tmp.zarr")


# ── working copies ───────────────────────────────────────────────────────────
def bundle_snapshot(path) -> dict:
    """{relative path: (size, mtime_ns)} of every file below *path* (symlinks not followed)."""
    path = pathlib.Path(path)
    out = {}
    for root, dirs, files in os.walk(path):
        for name in files:
            p = pathlib.Path(root) / name
            try:
                st = p.lstat()
            except OSError:
                continue
            out[str(p.relative_to(path))] = (st.st_size, st.st_mtime_ns)
    return out


def _tree_size(path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def _copy_tree(src, dst) -> None:
    """Copy a directory; on macOS try an APFS clone first (instant, copy-on-write)."""
    if sys.platform == "darwin":
        r = subprocess.run(["cp", "-cR", str(src), str(dst)], capture_output=True)
        if r.returncode == 0:
            return
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, symlinks=True)


def make_working_copy(src, dst) -> pathlib.Path:
    """A directory the app may write into that stands in for the bundle *src*.

    The app writes ``spatialdata_xenium.zarr`` next to the raw files on first load and
    later updates that store in place (ROI columns, SPLIT layers), so it must never be
    pointed at the bundle itself. Small bundles are copied; for large ones every
    top-level entry is symlinked, except the generated SpatialData store, which is
    copied (cloned on APFS) when the bundle already has one and is otherwise generated
    by the app inside the working directory.
    """
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    dst.mkdir(parents=True, exist_ok=False)
    raw = [e for e in sorted(src.iterdir()) if e.name not in _GENERATED]
    small = sum(_tree_size(e) if e.is_dir() else e.lstat().st_size for e in raw) <= COPY_LIMIT_BYTES
    for entry in raw:
        target = dst / entry.name
        if not small:
            os.symlink(entry, target)
        elif entry.is_symlink():                 # e.g. a transcripts.parquet kept elsewhere
            os.symlink(os.path.realpath(entry), target)
        elif entry.is_dir():
            shutil.copytree(entry, target)
        else:
            shutil.copy2(entry, target)
    zarr = src / "spatialdata_xenium.zarr"
    if zarr.is_dir() and not small:
        _copy_tree(zarr, dst / zarr.name)
    return dst


# ── viewer wire format (see the docstring of xenium_viewer_api.py) ───────────
_WIRE_DTYPES = {"float32": "<f4", "uint32": "<u4", "uint16": "<u2", "uint8": "u1", "int32": "<i4"}


def unpack(blob: bytes):
    """Decode a ``*.bin`` payload into (header dict, {name: array}).

    Checks the framing as it goes: 4-byte aligned header, each array padded to 4 bytes,
    and no bytes left over."""
    assert len(blob) >= 4, "payload shorter than its length prefix"
    (hlen,) = struct.unpack("<I", blob[:4])
    assert hlen % 4 == 0, "header is not padded to a multiple of 4 bytes"
    header = json.loads(blob[4:4 + hlen].decode("utf-8"))
    off = 4 + hlen
    arrays = {}
    for spec in header["arrays"]:
        dt = np.dtype(_WIRE_DTYPES[spec["dtype"]])
        nbytes = spec["length"] * dt.itemsize
        assert off + nbytes <= len(blob), f"array {spec['name']} runs past the payload"
        arrays[spec["name"]] = np.frombuffer(blob[off:off + nbytes], dtype=dt)
        off += nbytes + (-nbytes % 4)
    assert off == len(blob), f"{len(blob) - off} trailing bytes after the last array"
    return header, arrays


def get_bin(client, url, expect=200):
    resp = client.get(url)
    assert resp.status_code == expect, (url, resp.status_code)
    if expect != 200:
        return resp
    assert resp.mimetype == "application/octet-stream"
    assert resp.headers.get("Cache-Control") == "no-store"
    return unpack(resp.data)


def rings(arrays):
    """Split a boundaries payload into [(cell row, (n, 2) vertex array)]."""
    start = arrays["start"].astype(np.int64)
    xy = arrays["xy"].reshape(-1, 2)
    return [(int(c), xy[start[i]:start[i + 1]]) for i, c in enumerate(arrays["cell"])]


# ── raw bundle readers (the independent "truth") ─────────────────────────────
class RawBundle:
    """Reads a Xenium output bundle straight from its files."""

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self._cache = {}

    def has(self, name) -> bool:
        p = self.path / name
        return p.exists() and (p.is_dir() or p.stat().st_size > 0)

    def _memo(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @property
    def experiment(self) -> dict:
        return self._memo("exp", lambda: json.loads((self.path / "experiment.xenium").read_text()))

    @property
    def cells(self) -> pd.DataFrame:
        def load():
            df = pd.read_parquet(self.path / "cells.parquet")
            ids = df["cell_id"]
            if len(ids) and isinstance(ids.iloc[0], bytes):
                ids = ids.str.decode("utf-8")
            df["cell_id"] = ids.astype(str)
            return df.set_index("cell_id")
        return self._memo("cells", load)

    @property
    def matrix(self):
        """(features x cells CSC matrix, feature names, feature types | None, barcodes)."""
        def load():
            import h5py
            with h5py.File(self.path / "cell_feature_matrix.h5", "r") as h:
                m = h["matrix"]
                shape = tuple(int(v) for v in m["shape"][:])
                mat = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=shape)
                names = [v.decode() for v in m["features"]["name"][:]]
                ftype = ([v.decode() for v in m["features"]["feature_type"][:]]
                         if "feature_type" in m["features"] else None)
                barcodes = [v.decode() for v in m["barcodes"][:]]
            return mat, names, ftype, barcodes
        return self._memo("matrix", load)

    @property
    def gene_counts(self):
        """(cells x genes CSR of the Gene Expression features only, gene names, barcodes)."""
        def load():
            mat, names, ftype, barcodes = self.matrix
            if ftype is None:
                pytest_skip("cell_feature_matrix.h5 has no feature_type; nothing independent to compare with")
            keep = np.array([t == "Gene Expression" for t in ftype])
            return (mat.tocsr()[keep].T.tocsr(), [n for n, k in zip(names, keep) if k], barcodes)
        return self._memo("genes", load)

    def rings(self, kind: str):
        """{cell_id: [(n, 2) vertex array per polygon]} from <kind>_boundaries.parquet.
        A cell has one polygon, except multinucleate cells in the nucleus file (the rows
        of one nucleus share a label_id)."""
        def load():
            p = self.path / f"{kind}_boundaries.parquet"
            if not p.exists():
                return {}
            df = pd.read_parquet(p)
            ids = df["cell_id"]
            if len(ids) and isinstance(ids.iloc[0], bytes):
                ids = ids.str.decode("utf-8")
            ids = ids.astype(str).to_numpy()
            xy = df[["vertex_x", "vertex_y"]].to_numpy(dtype=float)
            label = df["label_id"].to_numpy() if "label_id" in df.columns else np.zeros(len(df), dtype=int)
            # rows of a polygon are contiguous in the file
            new = np.r_[True, (ids[1:] != ids[:-1]) | (label[1:] != label[:-1])]
            starts = np.flatnonzero(new)
            out = {}
            for a, b in zip(starts, np.r_[starts[1:], len(df)]):
                out.setdefault(ids[a], []).append(xy[a:b])
            return out
        return self._memo(("rings", kind), load)

    def boundaries(self, kind: str):
        """{cell_id: (n, 2) array of all vertices of the cell} (rings concatenated)."""
        return self._memo(("bounds", kind),
                          lambda: {cid: np.concatenate(r) for cid, r in self.rings(kind).items()})

    @property
    def n_transcripts(self) -> int:
        import pyarrow.parquet as pq
        return pq.read_metadata(self.path / "transcripts.parquet").num_rows

    def transcripts(self, bbox=None, columns=None) -> pd.DataFrame:
        """transcripts.parquet rows (optionally inside bbox = (x0, y0, x1, y1), inclusive),
        with feature_name / cell_id decoded to str."""
        import pyarrow.parquet as pq
        p = self.path / "transcripts.parquet"
        flt = None
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            flt = [("x_location", ">=", float(x0)), ("x_location", "<=", float(x1)),
                   ("y_location", ">=", float(y0)), ("y_location", "<=", float(y1))]
        names = pq.read_schema(p).names
        cols = [c for c in (columns or names) if c in names]
        df = pq.read_table(p, columns=cols, filters=flt).to_pandas()
        for c in ("feature_name", "cell_id"):
            if c in df.columns:
                v = df[c]
                if isinstance(v.dtype, pd.CategoricalDtype):
                    v = v.astype(object)
                if len(v) and isinstance(v.iloc[0], bytes):
                    v = v.str.decode("utf-8")
                df[c] = v.astype(str)
        return df

    def transcripts_of_cells(self, cell_ids, columns=None) -> pd.DataFrame:
        """All transcripts.parquet rows assigned to *cell_ids* (predicate pushdown)."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        p = self.path / "transcripts.parquet"
        schema = pq.read_schema(p)
        field = schema.field("cell_id").type
        ids = [int(c) for c in cell_ids] if pa.types.is_integer(field) else [str(c) for c in cell_ids]
        cols = [c for c in (columns or schema.names) if c in schema.names]
        df = pq.read_table(p, columns=cols, filters=[("cell_id", "in", ids)]).to_pandas()
        for c in ("feature_name", "cell_id"):
            if c in df.columns:
                v = df[c]
                if isinstance(v.dtype, pd.CategoricalDtype):
                    v = v.astype(object)
                if len(v) and isinstance(v.iloc[0], bytes):
                    v = v.str.decode("utf-8")
                df[c] = v.astype(str)
        return df

    def _analysis_csv(self, rel):
        p = self.path / rel
        if p.exists():
            return pd.read_csv(p)
        tar = self.path / "analysis.tar.gz"
        if tar.exists():
            with tarfile.open(tar) as t:
                try:
                    return pd.read_csv(io.BytesIO(t.extractfile(rel).read()))
                except KeyError:
                    return None
        return None

    def clusters(self, method: str):
        """Series barcode -> cluster from analysis/clustering/<method>/clusters.csv, or None."""
        df = self._analysis_csv(f"analysis/clustering/{method}/clusters.csv")
        return None if df is None else df.set_index("Barcode")["Cluster"]

    @property
    def umap(self):
        df = self._analysis_csv("analysis/umap/gene_expression_2_components/projection.csv")
        return None if df is None else df.set_index("Barcode")


def pytest_skip(msg):
    import pytest
    pytest.skip(msg)


def gene_tx(tx: pd.DataFrame, qv_min=20.0) -> pd.DataFrame:
    """The transcripts the cell-feature matrix counts: Q >= qv_min, gene features only."""
    keep = np.ones(len(tx), dtype=bool)
    if "qv" in tx.columns and qv_min is not None:
        keep &= tx["qv"].to_numpy() >= qv_min
    if "is_gene" in tx.columns:
        keep &= tx["is_gene"].to_numpy(dtype=bool)
    elif "codeword_category" in tx.columns:
        keep &= ~tx["codeword_category"].astype(str).isin(
            ["unassigned_codeword", "negative_control_codeword", "negative_control_probe",
             "genomic_control_probe", "deprecated_codeword"]).to_numpy()
    else:
        keep &= ~tx["feature_name"].str.match(
            r"^(NegControl|BLANK_|UnassignedCodeword|DeprecatedCodeword|Intergenic_Region|GenomicControl)"
        ).to_numpy()
    return tx[keep]


def count_matrix(cell_ids, genes, cell_index, gene_index) -> sp.csr_matrix:
    """cells x genes counts from one (cell, gene) pair per transcript."""
    ci = pd.Index([str(c) for c in cell_index])
    gi = pd.Index(list(gene_index))
    r = ci.get_indexer(pd.Index([str(c) for c in cell_ids]))
    c = gi.get_indexer(pd.Index(list(genes)))
    ok = (r >= 0) & (c >= 0)
    return sp.csr_matrix((np.ones(int(ok.sum()), dtype=np.float32), (r[ok], c[ok])),
                         shape=(len(ci), len(gi)))


def point_set(xy, decimals=2) -> frozenset:
    """The vertices of a polygon as a set (start vertex / closing vertex do not matter)."""
    return frozenset(map(tuple, np.round(np.asarray(xy, dtype=float), decimals)))


def sparse_equal(a, b) -> bool:
    a, b = sp.csr_matrix(a), sp.csr_matrix(b)
    if a.shape != b.shape:
        return False
    d = (a - b)
    d.eliminate_zeros()
    return d.nnz == 0


# ── figures ──────────────────────────────────────────────────────────────────
def marker_traces(fig):
    return [t for t in fig.data if getattr(t, "mode", None) == "markers" and t.customdata is not None]


def fig_cells(fig) -> pd.DataFrame:
    """One row per plotted cell: id, x, y, trace name, and colour (str) or value (float)."""
    rows = []
    for t in marker_traces(fig):
        ids = [str(c[0]) if isinstance(c, (list, tuple, np.ndarray)) else str(c) for c in t.customdata]
        col = t.marker.color
        if isinstance(col, str) or col is None:
            cols = [col] * len(ids)
        else:
            cols = list(col)
        assert len(t.x) == len(t.y) == len(ids) == len(cols), f"ragged trace {t.name!r}"
        rows.append(pd.DataFrame({"id": ids, "x": np.asarray(t.x, dtype=float),
                                  "y": np.asarray(t.y, dtype=float), "trace": t.name, "color": cols}))
    if not rows:
        return pd.DataFrame(columns=["id", "x", "y", "trace", "color"]).set_index("id")
    out = pd.concat(rows, ignore_index=True)
    assert out["id"].is_unique, "a cell is drawn more than once"
    return out.set_index("id")


def line_polygons(trace):
    """Split a NaN/None-separated line trace into a list of (n, 2) vertex arrays."""
    x = np.array([np.nan if v is None else v for v in trace.x], dtype=float)
    y = np.array([np.nan if v is None else v for v in trace.y], dtype=float)
    out, cur = [], []
    for a, b in zip(x, y):
        if np.isnan(a) or np.isnan(b):
            if cur:
                out.append(np.array(cur))
            cur = []
        else:
            cur.append((a, b))
    if cur:
        out.append(np.array(cur))
    return out


# ── background jobs ──────────────────────────────────────────────────────────
def wait_state(state, lock, timeout=120.0, done=("done",), bad=("error",), poll=0.05):
    """Block until a job-state dict reaches *done*; raise with its message on error."""
    t0 = time.time()
    while True:
        with lock:
            status, msg = state.get("status"), state.get("message", "")
        if status in done:
            return msg
        if status in bad:
            raise AssertionError(f"background job failed: {msg}")
        if time.time() - t0 > timeout:
            raise TimeoutError(f"job still {status!r} after {timeout:.0f}s: {msg}")
        time.sleep(poll)


class new_threads:
    """Context manager: on exit, wait for the threads started inside the block."""

    def __init__(self, timeout=60.0):
        self.timeout = timeout

    def __enter__(self):
        self._before = set(threading.enumerate())
        return self

    def __exit__(self, *exc):
        deadline = time.time() + self.timeout
        for t in [t for t in threading.enumerate() if t not in self._before]:
            if t is not threading.current_thread():
                t.join(max(0.0, deadline - time.time()))
        return False


# ── source-level access to code that is not importable on its own ────────────
def app_sources() -> list:
    """Every source file of the app: the modules of xenium_app in import order (as listed
    in its MODULES), any other file of the package, then the entry point."""
    init = ast.parse((APP_PACKAGE / "__init__.py").read_text())
    order = next(ast.literal_eval(n.value) for n in init.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "MODULES" for t in n.targets))
    files = [APP_PACKAGE / f"{name}.py" for name in order]
    files += sorted(p for p in APP_PACKAGE.glob("*.py") if p not in files)
    return files + [APP_SOURCE]


def app_source_of(name) -> pathlib.Path:
    """The source file that defines the top-level function or class *name*."""
    hits = [p for p in app_sources() for n in ast.parse(p.read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name]
    if len(hits) != 1:
        raise LookupError(f"{name}: defined in {len(hits)} files")
    return hits[0]


def app_tree():
    """The whole app as one module: the statements of every source file, in import order.
    (Line numbers are those of the file a statement comes from.)"""
    body = []
    for path in app_sources():
        body.extend(ast.parse(path.read_text(), filename=str(path)).body)
    return ast.Module(body=body, type_ignores=[])


def function_node(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise LookupError(name)


def compile_block(nodes, filename="<extracted from the app source>"):
    """Compile a list of statement nodes taken from the app source into a code object."""
    mod = ast.Module(body=list(nodes), type_ignores=[])
    ast.fix_missing_locations(mod)
    return compile(mod, filename, "exec")


# ── fresh-process helper ─────────────────────────────────────────────────────
_PRELUDE = """
import json, os, sys, warnings
warnings.simplefilter("ignore")
sys.argv = sys.argv[:1]
sys.path.insert(0, {repo!r})
import xenium_explorer as xe
RESULT = {{}}
"""
_EPILOGUE = """
sys.__stdout__.write("\\n@@RESULT@@" + json.dumps(RESULT, default=str) + "\\n")
sys.__stdout__.flush()
"""


def run_in_fresh_app(code: str, data_dir, cache_dir, env=None, timeout=600) -> dict:
    """Import the app in a new Python process against *data_dir* and run *code* there.

    *code* sees ``xe`` (the app module) and fills the dict ``RESULT``, which is returned.
    The app reads DATA_DIR / XENIUM_CACHE_DIR / XENIUM_ENABLE_REPL once, at import, so
    anything that needs different values for them needs its own process."""
    full_env = {k: v for k, v in os.environ.items()
                if k not in ("XENIUM_ENABLE_REPL", "XENIUM_RENDERER", "PORT", "HOST")}
    full_env.update({"DATA_DIR": str(data_dir), "XENIUM_CACHE_DIR": str(cache_dir),
                     "PYTHONWARNINGS": "ignore"})
    full_env.update(env or {})
    script = _PRELUDE.format(repo=str(REPO)) + textwrap.dedent(code) + _EPILOGUE
    proc = subprocess.run([sys.executable, "-c", script], env=full_env, cwd=str(cache_dir),
                          capture_output=True, text=True, timeout=timeout)
    marker = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@RESULT@@")]
    if proc.returncode != 0 or not marker:
        raise AssertionError(f"app subprocess failed (exit {proc.returncode})\n"
                             f"--- stdout ---\n{proc.stdout[-3000:]}\n--- stderr ---\n{proc.stderr[-6000:]}")
    return json.loads(marker[-1][len("@@RESULT@@"):])


# ── external tools ───────────────────────────────────────────────────────────
def baysor_bin():
    p = os.environ.get("BAYSOR_BIN") or os.path.expanduser("~/.julia/bin/baysor")
    return p if os.path.isfile(p) and os.access(p, os.X_OK) else shutil.which("baysor")


def proseg_bin():
    p = os.environ.get("PROSEG_BIN") or os.path.expanduser("~/.cargo/bin/proseg")
    return p if os.path.isfile(p) and os.access(p, os.X_OK) else shutil.which("proseg")


def r_packages_available(*pkgs) -> bool:
    """True when Rscript exists and every package in *pkgs* loads."""
    rscript = shutil.which("Rscript")
    if not rscript:
        return False
    expr = "; ".join(f'suppressPackageStartupMessages(library("{p}", character.only=TRUE))' for p in pkgs) or "1"
    try:
        return subprocess.run([rscript, "-e", expr], capture_output=True, timeout=180).returncode == 0
    except Exception:
        return False


def free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
