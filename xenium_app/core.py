"""Configuration, constants and the shared state of the app (state dicts and their locks,
server log capture). Imported first; imports no other module of the package.
The import block below is the single-file app's, kept whole so that third-party libraries
are still imported in the same order.

DATA, the loaded dataset, lives here as ``core.DATA``; it is assigned by startup.py."""


import os
import re
import sys
import warnings
warnings.filterwarnings("ignore", message=".*not recognized as a component of a Zarr hierarchy.*")
warnings.filterwarnings("ignore", message=".*R is not initialized by the main thread.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in divide.*")
import json
import io
import base64
import threading
import subprocess
import hashlib
import contextvars

# ─── Configurable paths (override via env vars for cloud deployment) ─────────
CACHE_BASE_DIR = os.environ.get("XENIUM_CACHE_DIR",
                                os.path.join(os.path.expanduser("~"), ".xenium_explorer_cache"))
# Defaults of the reference inputs (the .rds reference of Seurat kNN / RCTD / SPLIT / SpaGE,
# and a file with the genes to impute): empty unless the environment names them. Nothing
# machine-specific is built in.
REF_RDS_DEFAULT = os.environ.get("XENIUM_REF_RDS", "").strip()
IMPUTE_GENES_DEFAULT = os.environ.get("XENIUM_IMPUTE_GENES", "").strip()
REF_RDS_PLACEHOLDER = "/path/to/reference.rds (a Seurat object with cell-type labels; default: $XENIUM_REF_RDS)"


def _ref_rds_path(path) -> str:
    """The reference path as every job gets it: what was typed or pasted, without the
    surrounding white space (a trailing space or newline) and with ~ expanded. The
    callbacks validate and start their job with this one value."""
    path = str(path or "").strip()
    return os.path.expanduser(path) if path else ""


def _ref_rds_problem(path) -> "str | None":
    """Why *path* cannot be used as the reference .rds file (None: it can)."""
    path = str(path or "").strip()
    if not path:
        return ("Enter the path of the reference .rds file (a Seurat object with the cell-type labels). "
                "Set the environment variable XENIUM_REF_RDS to prefill it.")
    if not os.path.isfile(os.path.expanduser(path)):
        return f"Reference .rds file not found: {path}"
    return None


def _impute_genes_default() -> str:
    """Text for the 'Genes to Impute' box from the file XENIUM_IMPUTE_GENES names (one gene
    per line or comma-separated); "" when it is unset or unreadable."""
    if not IMPUTE_GENES_DEFAULT:
        return ""
    try:
        with open(os.path.expanduser(IMPUTE_GENES_DEFAULT)) as fh:
            return fh.read().strip()
    except OSError as exc:
        print(f"  XENIUM_IMPUTE_GENES: {IMPUTE_GENES_DEFAULT} could not be read ({exc})", flush=True)
        return ""


from concurrent.futures import ThreadPoolExecutor
from dash import no_update, Patch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
from PIL import Image

import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL, ClientsideFunction
from dash import dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.colors import qualitative

_ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\x1b\[[?][0-9;]*[A-Za-z]|\x1b[()][0-9A-Za-z]|[\x00-\x08\x0b\x0e-\x1f\x7f]')

def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub('', text)


# ─── Constants ────────────────────────────────────────────────────────────────
PIXEL_SIZE_UM = 0.2125

# Max cells for which to render boundaries (performance guard)
BOUNDARY_CELL_LIMIT = 3000

# RCTD pie chart settings
PIE_THRESHOLD      = 1500   # max cells in view to render RCTD weight pie charts
RCTD_PIE_RADIUS_UM = 8.0    # pie chart radius in µm

CLUSTER_COLORS = (
    qualitative.Plotly + qualitative.D3 + qualitative.G10
    + qualitative.T10 + qualitative.Pastel
)

DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
CARD_BG  = "#21262d"
BORDER   = "#30363d"
TEXT     = "#c9d1d9"
MUTED    = "#8b949e"
ACCENT   = "#58a6ff"
PLOT_BG  = "#0d1117"
GRID     = "#21262d"

QC_METRICS = {
    "transcript_counts": "Transcript Counts",
    "cell_area":         "Cell Area (µm²)",
    "nucleus_area":      "Nucleus Area (µm²)",
    "total_counts":      "Total Counts",
}
COLORSCALES = {
    "transcript_counts": "Viridis",
    "cell_area":         "Plasma",
    "nucleus_area":      "Cividis",
    "total_counts":      "Magma",
}


# ─── Annotation state (shared across callbacks via background thread) ─────────
_annot_state: dict = {"status": "idle", "message": ""}
_annot_lock   = threading.Lock()

_ANNOT_METHODS = {"celltypist": "CellTypist", "seurat": "Seurat", "rctd": "RCTD"}

# ─── Baysor segmentation state ────────────────────────────────────────────────
_baysor_state: dict = {"status": "idle", "message": "", "result": None}
_baysor_lock  = threading.Lock()

# ─── Proseg segmentation state ────────────────────────────────────────────────
_proseg_state: dict = {"status": "idle", "message": "", "result": None}
_proseg_lock  = threading.Lock()

# ─── Image-based segmentation state (imgseg: Cellpose / classical) ───────────
_imgseg_state: dict = {"status": "idle", "message": "", "result": None}
_imgseg_lock  = threading.Lock()

# Tools whose runs replace the Xenium segmentation: seg-source values "<tool>:<tag>",
# cache directories "<tool>_<dataset>_…", zarr "spatialdata_<tool>.zarr".
_SEG_TOOL_LABELS = {"baysor": "Baysor", "proseg": "Proseg", "imgseg": "Image segmentation"}


def _seg_state(tool: str) -> tuple:
    """(state dict, lock) of a resegmentation tool, (None, None) for anything else."""
    if tool == "baysor":
        return _baysor_state, _baysor_lock
    if tool == "proseg":
        return _proseg_state, _proseg_lock
    if tool == "imgseg":
        return _imgseg_state, _imgseg_lock
    return None, None


def _seg_states() -> list:
    """[(tool, lock, state)] of every resegmentation tool."""
    return [(t,) + _seg_state(t)[::-1] for t in _SEG_TOOL_LABELS]


# A run directory is reused by a re-run with the same parameters, so the directory alone
# does not say which cells a result holds: every committed run also has a run id (a file
# in the directory, written by caching._seg_stage_commit). Runs stored before this
# existed have none ("").
_SEG_RUN_ID_FILE = "run_id"


def _seg_run_id(result) -> str:
    """The run id of a loaded Baysor / Proseg / image-segmentation result ("" for a run
    stored without one). Read from the run directory once and kept in the result, so a
    result that stays in memory keeps its own id when a re-run replaces the directory."""
    if not isinstance(result, dict):
        return ""
    rid = result.get("run_id")
    if rid is None:
        rid = ""
        try:
            with open(os.path.join(str(result.get("out_dir") or ""), _SEG_RUN_ID_FILE)) as fh:
                rid = fh.read().strip()[:64]
        except OSError:
            pass
        result["run_id"] = rid
    return rid


def _seg_result(seg_source, done_only: bool = False):
    """The loaded result of the tool *seg_source* names ("<tool>" or "<tool>:<tag>"),
    None for Xenium or when nothing is loaded (done_only: also while the tool's state is
    not "done")."""
    state, lock = _seg_state(_seg_tool(seg_source))
    if state is None:
        return None
    with lock:
        if done_only and state.get("status") != "done":
            return None
        res = state.get("result")
    _seg_run_id(res)         # fixed at first use (see _seg_run_id)
    return res


# ─── SpaGE imputation state ───────────────────────────────────────────────────
_spage_state: dict = {"status": "idle", "message": "", "result": None,
                      "result_path": None, "result_genes": None}
_spage_lock   = threading.Lock()
SPAGE_STREAM_THRESHOLD = 500_000_000  # ~2 GB at float32; above this stream to zarr on disk
_spage_repl_pending = False   # set by run_spage(); cleared once poll_spage_repl fires
_spage_last_logged  = ""      # last message printed by poll_spage_repl (dedup)
_active_counts_mode = "original"  # mirrors counts-mode-store; updated by sync_counts_mode

# ─── CellNEST CCC state ───────────────────────────────────────────────────────
_cellnest_state: dict = {"status": "idle", "message": "", "data": None}
_cellnest_lock  = threading.Lock()

# ─── Reseg UMAP state ─────────────────────────────────────────────────────────
_umap_reseg_state: dict = {"status": "idle", "message": "", "result": None}
_umap_reseg_lock  = threading.Lock()

# ─── Subset state ─────────────────────────────────────────────────────────────
_subset_version = 0   # incremented by subset()/unsubset() to trigger re-render

# ─── SpatialData / Sopa state ─────────────────────────────────────────────────
_sdata_state: dict = {"status": "idle", "message": "", "sdata": None, "roi": None, "patches": None}
_sdata_lock   = threading.Lock()
_save_sdata_state: dict = {"status": "idle", "message": ""}
_save_sdata_lock  = threading.Lock()
_sdata_version = 0  # incremented when ROI/patches change, to trigger overlay refresh

# ─── SPLIT ambient RNA correction state ───────────────────────────────────────
_split_state: dict = {"status": "idle", "message": "", "result": None}
_split_lock   = threading.Lock()

# ─── Spatial niches state ─────────────────────────────────────────────────────
_niche_state: dict = {"status": "idle", "message": ""}
_niche_lock   = threading.Lock()

# ─── Spatial analysis state ───────────────────────────────────────────────────
_spatial_state: dict = {"status": "idle", "message": "", "result": None}
_spatial_lock   = threading.Lock()

# ─── ROI Annotation state ─────────────────────────────────────────────────────
_roi_state: dict = {
    "rois": [],           # [{name, cls, polygon_xy:[[x,y],...], color}]
    "show": True,         # overlay visibility
    "pending_hull": None, # [[x,y],...] awaiting save
}
_roi_lock = threading.Lock()

# ─── Server log capture ────────────────────────────────────────────────────────
import collections
_log_buffer: collections.deque = collections.deque(maxlen=200)
_log_lock = threading.Lock()

class _LogCapture:
    """Tee writes to the original stream and to _log_buffer."""
    def __init__(self, original):
        self._orig = original

    def write(self, s):
        self._orig.write(s)
        if s.strip():
            with _log_lock:
                _log_buffer.append(s.rstrip())
        return len(s)

    def flush(self):
        self._orig.flush()

    def fileno(self):
        return self._orig.fileno()

    def isatty(self):
        return self._orig.isatty()

    # Forward all other attribute lookups to the original stream
    def __getattr__(self, name):
        return getattr(self._orig, name)

sys.stdout = _LogCapture(sys.stdout)
sys.stderr = _LogCapture(sys.stderr)


def _redirect_rpy2_console():
    """Call once inside any thread that uses rpy2 to capture R console output."""
    import warnings
    warnings.filterwarnings("ignore", message="R is not initialized by the main thread")
    try:
        import rpy2.rinterface_lib.callbacks as rpy2_cb
        def _r_write(x):
            x = x.rstrip()
            if x:
                with _log_lock:
                    _log_buffer.append(f"[R] {x}")
        rpy2_cb.consolewrite_print     = _r_write
        rpy2_cb.consolewrite_warnerror = _r_write
    except Exception:
        pass

CELLTYPIST_MODELS = {
    "Healthy_Adult_Heart.pkl": "Healthy Adult Heart (Azimuth/CellTypist)",
    "Immune_All_Low.pkl":      "Immune All – Low (CellTypist)",
    "Immune_All_High.pkl":     "Immune All – High (CellTypist)",
}


# ─── Callbacks ────────────────────────────────────────────────────────────────

def _seg_tool(seg_source: str) -> str:
    """Extract tool name from seg-source value: 'xenium', 'baysor', 'proseg' or 'imgseg'."""
    return (seg_source or "xenium").split(":")[0]
