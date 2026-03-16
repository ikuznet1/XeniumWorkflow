#!/usr/bin/env python3
"""
Xenium Explorer Clone
Interactive visualization for 10x Genomics Xenium spatial transcriptomics data.

Usage:
    python xenium_explorer.py [path/to/output-XETG...]
    python xenium_explorer.py          # auto-detects output-* directory
"""

import os
import re
import sys
import warnings
warnings.filterwarnings("ignore", message=".*not recognized as a component of a Zarr hierarchy.*")
warnings.filterwarnings("ignore", message=".*R is not initialized by the main thread.*")
import json
import io
import base64
import threading
import subprocess
import hashlib
import contextvars
from concurrent.futures import ThreadPoolExecutor
from dash import no_update, Patch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
from PIL import Image

import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
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

def _labels_key_for_method(method: str, alt_res=None) -> str:
    """Return the _annot_state key for a given annotation method + seg source."""
    if alt_res is None:
        return f"labels_{method}"
    out_tag = hashlib.md5((alt_res.get("out_dir", "")).encode()).hexdigest()[:8]
    tool    = alt_res.get("source", "baysor")
    return f"labels_{method}_{tool}_{out_tag}"

# ─── Baysor segmentation state ────────────────────────────────────────────────
_baysor_state: dict = {"status": "idle", "message": "", "result": None}
_baysor_lock  = threading.Lock()

# ─── Proseg segmentation state ────────────────────────────────────────────────
_proseg_state: dict = {"status": "idle", "message": "", "result": None}
_proseg_lock  = threading.Lock()

# ─── SpaGE imputation state ───────────────────────────────────────────────────
_spage_state: dict = {"status": "idle", "message": "", "result": None,
                      "result_path": None, "result_genes": None}
_spage_lock   = threading.Lock()
SPAGE_STREAM_THRESHOLD = 500_000_000  # ~2 GB at float32; above this stream to zarr on disk
_spage_repl_pending = False   # set by run_spage(); cleared once poll_spage_repl fires
_spage_last_logged  = ""      # last message printed by poll_spage_repl (dedup)
_active_counts_mode = "original"  # mirrors counts-mode-store; updated by sync_counts_mode

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


def _cache_path(model_name: str) -> str:
    """Return path for the cached annotation parquet file."""
    safe = model_name.replace(".pkl", "").replace("/", "_")
    cache_dir = os.path.join(os.path.expanduser("~"), ".xenium_explorer_cache")
    os.makedirs(cache_dir, exist_ok=True)
    # Include a hash of the data dir so caches from different datasets don't clash
    tag = hashlib.md5(DATA["data_dir"].encode()).hexdigest()[:8]
    return os.path.join(cache_dir, f"{safe}_{tag}.parquet")


def _run_celltypist(model_name: str, labels_key: str = "labels",
                    expr_override=None, cell_ids_override=None) -> None:
    """Background thread: annotate cells with CellTypist and store results."""
    _redirect_rpy2_console()
    import anndata as ad
    import celltypist
    from celltypist import models as ct_models

    def _set(status, message, labels=None):
        with _annot_lock:
            _annot_state["status"]  = status
            _annot_state["message"] = message
            if labels is not None:
                _annot_state[labels_key] = labels

    try:
        # ── Check disk cache first ───────────────────────────────────────
        _cache_key = model_name if labels_key == "labels" else f"{model_name}_{labels_key}"
        cache_file = _cache_path(_cache_key)
        if os.path.exists(cache_file):
            _set("running", "Loading cached annotation…")
            cached = pd.read_parquet(cache_file)
            pred_labels = cached["label"].astype(str)
            pred_labels.index = pred_labels.index.astype(str)
            if cell_ids_override is not None:
                _sample = set([str(c) for c in cell_ids_override[:200]])
                _overlap = len(set(pred_labels.index[:200]) & _sample)
                if _overlap < max(1, len(_sample) * 0.05):
                    print(f"  Warning: cached annotation index mismatch "
                          f"(overlap {_overlap}/{len(_sample)}) — re-running", flush=True)
                    try:
                        os.remove(cache_file)
                    except Exception:
                        pass
                    # Fall through to full re-run
                else:
                    unique_types = pred_labels.unique().tolist()
                    print(f"  Loaded annotation from cache: {len(unique_types)} cell types", flush=True)
                    _set("done", f"Done (cached) — {len(unique_types)} cell types", labels=pred_labels)
                    return
            else:
                unique_types = pred_labels.unique().tolist()
                print(f"  Loaded annotation from cache: {len(unique_types)} cell types", flush=True)
                _set("done", f"Done (cached) — {len(unique_types)} cell types", labels=pred_labels)
                return

        _set("running", "Downloading model…")
        ct_models.download_models(model=model_name, force_update=False)

        _set("running", "Building expression matrix…")
        # Build AnnData aligned to barcode order (expr rows = barcodes)
        expr     = expr_override if expr_override is not None else DATA["expr"]
        genes    = DATA["gene_names"]
        barcodes = cell_ids_override if cell_ids_override is not None else DATA["barcodes"]

        # Normalize to 10,000 counts per cell then log1p (CP10K + log1p)
        # CellTypist requires log1p-normalized input
        mat = expr.astype("float32").tocsr()
        row_sums = np.asarray(mat.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0          # avoid divide-by-zero
        scale = 10_000.0 / row_sums
        mat = sp.diags(scale).dot(mat).tocsr()  # scale each row
        mat.data = np.log1p(mat.data)           # log1p in-place on non-zeros

        adata = ad.AnnData(
            X   = mat,
            obs = pd.DataFrame(index=barcodes),
            var = pd.DataFrame(index=genes),
        )

        _set("running", f"Running CellTypist ({model_name})…")
        predictions = celltypist.annotate(
            adata, model=model_name, majority_voting=True
        )

        # CellTypist returns a DataFrame; columns may have category dtype
        pred_df = predictions.predicted_labels
        col = "majority_voting" if "majority_voting" in pred_df.columns else "predicted_labels"
        # Cast to plain str so reindex/fillna work correctly on any pandas version
        pred_labels = pred_df[col].astype(str)
        pred_labels.index = pred_labels.index.astype(str)
        unique_types = pred_labels.unique().tolist()
        print(f"  Annotation done: {len(unique_types)} cell types — {unique_types[:8]}", flush=True)

        # ── Save to disk cache ───────────────────────────────────────────
        try:
            cache_file = _cache_path(_cache_key)
            pd.DataFrame({"label": pred_labels}).to_parquet(cache_file)
            print(f"  Annotation cached to {cache_file}", flush=True)
        except Exception as ce:
            print(f"  Warning: could not cache annotation: {ce}", flush=True)

        _set("done", f"Done — {len(unique_types)} cell types", labels=pred_labels)

    except Exception as exc:
        _set("error", str(exc))


def _run_rctd_annotation(rds_path: str, label_col: str = "Names", mode: str = "full",
                          n_cores: int = 4, labels_key: str = "labels",
                          expr_override=None, cell_ids_override=None,
                          umi_min: int = 20, umi_min_sigma: int = 100,
                          mode_labels_key: str = None) -> None:
    """
    Background thread: run RCTD (spacexr) for cell-type assignment.
    Requires the spacexr R package:
        devtools::install_github('dmcable/spacexr')
    In 'full' mode each cell is assigned the highest-weight type.
    In 'doublet'/'multi' mode the first_type column is used.
    """
    _redirect_rpy2_console()
    import rpy2.robjects.conversion as _rconv
    import rpy2.robjects as _ro_mod
    _rconv.set_conversion(_ro_mod.default_converter)

    _key_for_cache = mode_labels_key or labels_key

    def _set(status, message, labels=None):
        with _annot_lock:
            _annot_state["status"]  = status
            _annot_state["message"] = message
            if labels is not None:
                _annot_state[labels_key] = labels
                if mode_labels_key and mode_labels_key != labels_key:
                    _annot_state[mode_labels_key] = labels

    try:
        cache_key  = f"rctd_{os.path.basename(rds_path)}_{label_col}_{mode}_umi{umi_min}_sig{umi_min_sigma}_{_key_for_cache}"
        cache_file = _cache_path(cache_key)
        if os.path.exists(cache_file):
            _set("running", "Loading cached RCTD annotation…")
            cached = pd.read_parquet(cache_file)
            labels = cached["label"].astype(str)
            labels.index = labels.index.astype(str)
            if cell_ids_override is not None:
                _sample = set([str(c) for c in cell_ids_override[:200]])
                _overlap = len(set(labels.index[:200]) & _sample)
                if _overlap < max(1, len(_sample) * 0.05):
                    print(f"  Warning: cached RCTD annotation index mismatch "
                          f"(overlap {_overlap}/{len(_sample)}) — re-running", flush=True)
                    try:
                        os.remove(cache_file)
                        weights_cache_del = cache_file.replace(".parquet", "_weights.parquet")
                        if os.path.exists(weights_cache_del):
                            os.remove(weights_cache_del)
                    except Exception:
                        pass
                    # Fall through to full re-run
                else:
                    unique_types = labels.unique().tolist()
                    print(f"  RCTD: loaded from cache — {len(unique_types)} cell types", flush=True)
                    # Restore weights so pie charts render on zoom
                    weights_cache = cache_file.replace(".parquet", "_weights.parquet")
                    weights_df = None
                    if os.path.exists(weights_cache):
                        try:
                            weights_df = pd.read_parquet(weights_cache)
                            weights_df.index = weights_df.index.astype(str)
                            print(f"  RCTD: loaded weights from cache ({weights_df.shape[1]} types)", flush=True)
                        except Exception as _we:
                            print(f"  RCTD: weight cache load failed: {_we}", flush=True)
                    weights_key = f"rctd_weights_{labels_key}"
                    with _annot_lock:
                        _annot_state[weights_key] = weights_df
                    _set("done", f"Done (cached) — {len(unique_types)} cell types", labels=labels)
                    return
            else:
                unique_types = labels.unique().tolist()
                print(f"  RCTD: loaded from cache — {len(unique_types)} cell types", flush=True)
                # Restore weights so pie charts render on zoom
                weights_cache = cache_file.replace(".parquet", "_weights.parquet")
                weights_df = None
                if os.path.exists(weights_cache):
                    try:
                        weights_df = pd.read_parquet(weights_cache)
                        weights_df.index = weights_df.index.astype(str)
                        print(f"  RCTD: loaded weights from cache ({weights_df.shape[1]} types)", flush=True)
                    except Exception as _we:
                        print(f"  RCTD: weight cache load failed: {_we}", flush=True)
                weights_key = f"rctd_weights_{labels_key}"
                with _annot_lock:
                    _annot_state[weights_key] = weights_df
                _set("done", f"Done (cached) — {len(unique_types)} cell types", labels=labels)
                return

        import tempfile
        import scipy.io as sio
        import scipy.sparse as sp_
        import rpy2.robjects as ro
        import rpy2.robjects.conversion as _rconv
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr

        # Ensure rpy2 conversion rules are active in this thread
        _rconv.set_conversion(ro.default_converter)
        pandas2ri.activate()

        from rpy2.rinterface_lib import openrlib
        with openrlib.rlock:
            # ── 1. Check spacexr ─────────────────────────────────────────────
            _set("running", "Loading spacexr…")
            try:
                importr("spacexr")
            except Exception as _spacexr_err:
                print(f"  RCTD: importr('spacexr') failed: {_spacexr_err}", flush=True)
                _set("error", f"spacexr not found by rpy2: {str(_spacexr_err)[:200]}")
                return

            importr("SeuratObject")
            importr("Matrix")
            base = importr("base")

            # ── 2. Load Seurat reference ─────────────────────────────────────
            _set("running", "Loading Seurat reference…")
            rds = base.readRDS(rds_path)
            meta_r  = ro.r['slot'](rds, "meta.data")
            meta_df = pandas2ri.rpy2py(meta_r)
            if label_col not in meta_df.columns:
                _set("error", f"Column '{label_col}' not found. Available: {list(meta_df.columns)[:10]}")
                return

            # ── 3. Determine shared genes ────────────────────────────────────
            _set("running", "Finding shared genes…")
            ro.r.assign("._rctd_rds", rds)
            mat_r     = ro.r("SeuratObject::LayerData(._rctd_rds[['RNA']], layer='counts')")
            ref_genes = list(ro.r['rownames'](mat_r))
            xenium_genes = list(DATA["gene_names"])
            shared_genes = sorted(set(xenium_genes) & set(ref_genes))
            if len(shared_genes) < 10:
                _set("error", f"Only {len(shared_genes)} shared genes between reference and panel.")
                return
            print(f"  RCTD: {len(shared_genes)} shared genes", flush=True)

            # ── 4. Build Reference object ────────────────────────────────────
            _set("running", f"Building RCTD Reference ({len(meta_df):,} ref cells)…")
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
            _set("running", "Building SpatialRNA object…")
            df = DATA["df"]
            if expr_override is not None:
                xen_mat_full = expr_override.T        # genes × cells
                cell_ids     = [str(c) for c in (cell_ids_override or range(expr_override.shape[0]))]
                coords_df    = df[["x_centroid", "y_centroid"]].reindex(
                                   pd.Index(cell_ids)).fillna(0.0)
            else:
                xen_mat_full = DATA["expr"].T  # genes × cells
                cell_ids     = [str(c) for c in df.index.tolist()]
                coords_df    = df[["x_centroid", "y_centroid"]].copy()
                coords_df.index = cell_ids

            gene_idx   = {g: i for i, g in enumerate(xenium_genes)}
            shared_idx = [gene_idx[g] for g in shared_genes]
            xen_sub    = sp_.csc_matrix(xen_mat_full[shared_idx, :])   # shared_genes × cells

            tmpdir = tempfile.mkdtemp(prefix="xenium_rctd_")
            sio.mmwrite(os.path.join(tmpdir, "xenium.mtx"), xen_sub)
            with open(os.path.join(tmpdir, "genes.txt"),    "w") as f: f.write("\n".join(shared_genes) + "\n")
            with open(os.path.join(tmpdir, "barcodes.txt"), "w") as f: f.write("\n".join(cell_ids) + "\n")
            coords_df.to_csv(os.path.join(tmpdir, "coords.csv"))

            ro.r.assign("._rctd_tmpdir", tmpdir)
            ro.r("""
._rctd_xen_mat  <- Matrix::readMM(file.path(._rctd_tmpdir, "xenium.mtx"))
._rctd_xen_genes <- readLines(file.path(._rctd_tmpdir, "genes.txt"))
._rctd_xen_bcs   <- readLines(file.path(._rctd_tmpdir, "barcodes.txt"))
rownames(._rctd_xen_mat) <- ._rctd_xen_genes
colnames(._rctd_xen_mat) <- ._rctd_xen_bcs
._rctd_xen_mat  <- as(._rctd_xen_mat, 'dgCMatrix')
._rctd_coords   <- read.csv(file.path(._rctd_tmpdir, "coords.csv"), row.names=1)
colnames(._rctd_coords) <- c('x', 'y')
._rctd_numi_xen <- colSums(._rctd_xen_mat)
._rctd_puck     <- spacexr::SpatialRNA(
    coords = ._rctd_coords,
    counts = ._rctd_xen_mat,
    nUMI   = ._rctd_numi_xen
)
""")

            # ── 6. Create and run RCTD ───────────────────────────────────────
            _set("running", f"Running RCTD ({mode} mode, {n_cores} cores)…")
            ro.r.assign("._rctd_mode",    mode)
            ro.r.assign("._rctd_cores",   n_cores)
            ro.r.assign("._rctd_umi_min",       ro.IntVector([umi_min]))
            ro.r.assign("._rctd_umi_min_sigma", ro.IntVector([umi_min_sigma]))
            ro.r("""
._rctd_obj <- spacexr::create.RCTD(._rctd_puck, ._rctd_reference, max_cores=._rctd_cores,
    UMI_min=._rctd_umi_min[1], UMI_min_sigma=._rctd_umi_min_sigma[1])
._rctd_obj <- spacexr::run.RCTD(._rctd_obj, doublet_mode=._rctd_mode)
""")
            print("  RCTD: run complete — extracting results…", flush=True)

            # ── 7. Extract per-cell labels ────────────────────────────────────
            _set("running", "Extracting RCTD results…")
            if mode == "full":
                ro.r("""
._rctd_weights <- ._rctd_obj@results$weights
._rctd_types   <- colnames(._rctd_weights)[apply(._rctd_weights, 1, which.max)]
names(._rctd_types) <- rownames(._rctd_weights)
""")
            else:  # doublet / multi
                ro.r("""
._rctd_res   <- ._rctd_obj@results$results_df
._rctd_types <- as.character(._rctd_res$first_type)
names(._rctd_types) <- rownames(._rctd_res)
""")

            types_r   = ro.r("._rctd_types")
            names_r   = ro.r("names(._rctd_types)")
            cell_ids  = list(names_r)
            cell_vals = [str(v) for v in types_r]
            labels    = pd.Series(dict(zip(cell_ids, cell_vals)), name="label").astype(str)
            labels.index  = labels.index.astype(str)

            unique_types = labels.unique().tolist()
            print(f"  RCTD: {len(labels):,} cells → {len(unique_types)} types", flush=True)

            pd.DataFrame({"label": labels}).to_parquet(cache_file)

            # ── 8. Extract and cache weight matrix (full mode only) ───────────
            weights_df = None
            if mode == "full":
                _set("running", "Extracting RCTD weight matrix…")
                try:
                    ro.r("._rctd_weights_dense <- as.matrix(._rctd_weights)")
                    ct_names   = list(ro.r("colnames(._rctd_weights)"))
                    cell_ids_w = list(ro.r("rownames(._rctd_weights)"))
                    w_arr      = np.array(ro.r("._rctd_weights_dense"))
                    weights_df = pd.DataFrame(w_arr, index=cell_ids_w, columns=ct_names)
                    weights_df.index = weights_df.index.astype(str)
                    weights_cache = cache_file.replace(".parquet", "_weights.parquet")
                    weights_df.to_parquet(weights_cache)
                    print(f"  RCTD: weights saved ({weights_df.shape[1]} types)", flush=True)
                except Exception as _we:
                    print(f"  RCTD: weight extraction failed: {_we}", flush=True)
            weights_key = f"rctd_weights_{labels_key}"
            with _annot_lock:
                _annot_state[weights_key] = weights_df

            _set("done", f"Done — {len(unique_types)} cell types in {len(labels):,} cells",
                 labels=labels)

            # Cleanup R temporaries (best-effort)
            try:
                ro.r("rm(list=ls(pattern='^\\._rctd_'))")
            except Exception:
                pass
            import shutil; shutil.rmtree(tmpdir, ignore_errors=True)

    except Exception as exc:
        import traceback; traceback.print_exc()
        _set("error", str(exc)[:300])


def _run_seurat_annotation(rds_path: str, label_col: str = "Names", labels_key: str = "labels",
                            expr_override=None, cell_ids_override=None) -> None:
    """
    Background thread: transfer cell type labels from a Seurat RDS reference
    to Xenium cells via PCA + cosine kNN majority voting.
    Uses the same rpy2 + SpaGE PV alignment already set up for SpaGE.
    """
    _redirect_rpy2_console()
    import rpy2.robjects.conversion as _rconv
    import rpy2.robjects as _ro_mod
    _rconv.set_conversion(_ro_mod.default_converter)

    def _set(status, message, labels=None):
        with _annot_lock:
            _annot_state["status"]  = status
            _annot_state["message"] = message
            if labels is not None:
                _annot_state[labels_key] = labels

    try:
        cache_key = f"seurat_{os.path.basename(rds_path)}_{label_col}_{labels_key}"
        cache_file = _cache_path(cache_key)
        if os.path.exists(cache_file):
            _set("running", "Loading cached annotation…")
            cached = pd.read_parquet(cache_file)
            labels = cached["label"].astype(str)
            labels.index = labels.index.astype(str)
            if cell_ids_override is not None:
                _sample = set([str(c) for c in cell_ids_override[:200]])
                _overlap = len(set(labels.index[:200]) & _sample)
                if _overlap < max(1, len(_sample) * 0.05):
                    print(f"  Warning: cached Seurat annotation index mismatch "
                          f"(overlap {_overlap}/{len(_sample)}) — re-running", flush=True)
                    try:
                        os.remove(cache_file)
                    except Exception:
                        pass
                    # Fall through to full re-run
                else:
                    unique_types = labels.unique().tolist()
                    print(f"  Loaded annotation from cache: {len(unique_types)} cell types", flush=True)
                    _set("done", f"Done (cached) — {len(unique_types)} cell types", labels=labels)
                    return
            else:
                unique_types = labels.unique().tolist()
                print(f"  Loaded annotation from cache: {len(unique_types)} cell types", flush=True)
                _set("done", f"Done (cached) — {len(unique_types)} cell types", labels=labels)
                return

        # ── 1. Load Seurat object via rpy2 ───────────────────────────────
        _set("running", "Loading Seurat RDS (may take ~30s)…")
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        from rpy2.rinterface_lib import openrlib
        pandas2ri.activate()

        with openrlib.rlock:
            base = importr("base")
            rds  = base.readRDS(rds_path)

            # ── 2. Extract metadata → label vector ──────────────────────────
            _set("running", f"Extracting '{label_col}' labels…")
            meta_r = ro.r['slot'](rds, "meta.data")
            meta_df = pandas2ri.rpy2py(meta_r)
            if label_col not in meta_df.columns:
                cols = list(meta_df.columns)
                _set("error", f"Column '{label_col}' not found. Available: {cols[:10]}")
                return

            ref_labels = meta_df[label_col].astype(str)  # index = ref cell barcodes

            unique_types = ref_labels.unique().tolist()
            print(f"  Reference: {len(ref_labels)} cells, {len(unique_types)} cell types", flush=True)

            # ── 3. Find shared genes first (before loading any matrix) ──────
            import scipy.sparse as sp_
            import tempfile

            importr("SeuratObject")
            importr("Matrix")

            _set("running", "Reading reference gene list…")
            ro.r.assign("._annot_rds_tmp", rds)
            mat_r = ro.r("SeuratObject::LayerData(._annot_rds_tmp[['RNA']], layer='data')")
            ref_genes_all = list(ro.r['rownames'](mat_r))
            ref_bcs = list(ro.r['colnames'](mat_r))

            xenium_genes = list(DATA["gene_names"])
            shared = sorted(set(xenium_genes) & set(ref_genes_all))
            if len(shared) < 10:
                _set("error", f"Only {len(shared)} shared genes — not enough for label transfer.")
                return
            print(f"  Shared genes for label transfer: {len(shared)}", flush=True)

            # ── 4. Export ONLY shared-gene rows via rpy2 (no subprocess) ────
            _set("running", f"Extracting {len(shared)} shared-gene submatrix…")
            tmpdir = tempfile.mkdtemp(prefix="xenium_annot_")
            ro.r.assign("._annot_mat",   mat_r)
            ro.r.assign("._annot_genes", ro.StrVector(shared))
            ro.r.assign("._annot_dir",   tmpdir)
            ro.r("""
mat_sub <- ._annot_mat[._annot_genes, , drop = FALSE]
Matrix::writeMM(mat_sub, file.path(._annot_dir, "matrix.mtx"))
write.table(data.frame(gene = rownames(mat_sub)),
            file.path(._annot_dir, "genes.txt"),
            row.names = FALSE, col.names = FALSE, quote = FALSE)
write.table(data.frame(bc = colnames(mat_sub)),
            file.path(._annot_dir, "barcodes.txt"),
            row.names = FALSE, col.names = FALSE, quote = FALSE)
# Clean up temp R objects
rm(._annot_mat, ._annot_genes, ._annot_dir, ._annot_rds_tmp, mat_sub)
""")

        _set("running", "Loading reference sub-matrix…")
        import scipy.io as sio_
        ref_mat = sio_.mmread(os.path.join(tmpdir, "matrix.mtx")).T.tocsr().astype("float32")
        # ref_mat is now (n_ref_cells × n_shared), rows in order of ref_bcs, cols = shared
        ref_genes = pd.read_csv(os.path.join(tmpdir, "genes.txt"), header=None)[0].tolist()
        ref_bcs   = pd.read_csv(os.path.join(tmpdir, "barcodes.txt"), header=None)[0].tolist()

        xen_gene_idx = {g: i for i, g in enumerate(xenium_genes)}
        shared_xen_idx = [xen_gene_idx[g] for g in ref_genes]  # ref_genes == shared after subset

        # Reference matrix for shared genes (cells × shared) — already subset
        ref_shared = ref_mat

        # Xenium/reseg matrix for shared genes — build from sparse expr
        xen_expr = expr_override if expr_override is not None else DATA["expr"]
        xen_shared = xen_expr[:, shared_xen_idx].toarray().astype("float32")
        # CP10K + log1p normalize Xenium
        row_sums = xen_shared.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        xen_shared = np.log1p(xen_shared / row_sums * 10_000)

        # Also normalize reference (already log-normalized from Seurat data layer, but z-score both)
        import scipy.stats as st_
        from sklearn.decomposition import TruncatedSVD
        from sklearn.neighbors import NearestNeighbors

        _set("running", "Z-scoring and fitting PCA…")
        # Z-score across cells (per gene)
        ref_dense = ref_shared.toarray() if sp_.issparse(ref_shared) else ref_shared
        ref_mean = ref_dense.mean(axis=0, keepdims=True)
        ref_std  = ref_dense.std(axis=0, keepdims=True) + 1e-8
        ref_z = (ref_dense - ref_mean) / ref_std

        xen_mean = xen_shared.mean(axis=0, keepdims=True)
        xen_std  = xen_shared.std(axis=0, keepdims=True) + 1e-8
        xen_z = (xen_shared - xen_mean) / xen_std

        # PCA on reference, project Xenium
        n_comp = min(50, len(shared) - 1, ref_dense.shape[0] - 1)
        _set("running", f"PCA ({n_comp} components)…")
        svd = TruncatedSVD(n_components=n_comp, random_state=0)
        ref_pca = svd.fit_transform(ref_z)   # (n_ref × n_comp)
        xen_pca = xen_z @ svd.components_.T  # (n_xen × n_comp)

        # ── 5. kNN label transfer ────────────────────────────────────────
        _set("running", "kNN label transfer (50 neighbours)…")
        # Align ref_labels to ref_bcs order
        ref_label_series = pd.Series(ref_labels.values, index=ref_bcs)

        k = min(50, len(ref_bcs))
        nbrs = NearestNeighbors(n_neighbors=k, metric="cosine").fit(ref_pca)
        _, indices = nbrs.kneighbors(xen_pca)

        # Majority vote
        ref_label_arr = ref_label_series.values  # ordered by ref_bcs
        xen_labels = []
        for row in indices:
            votes = ref_label_arr[row]
            winner = pd.Series(votes).mode()[0]
            xen_labels.append(winner)

        xen_barcodes = cell_ids_override if cell_ids_override is not None else DATA["barcodes"]
        pred_labels = pd.Series(xen_labels, index=[str(b) for b in xen_barcodes], name="label")
        unique_final = pred_labels.unique().tolist()
        print(f"  Label transfer done: {len(unique_final)} cell types", flush=True)

        # Cache
        try:
            pd.DataFrame({"label": pred_labels}).to_parquet(cache_file)
        except Exception as ce:
            print(f"  Warning: cache failed: {ce}", flush=True)

        _set("done", f"Done — {len(unique_final)} cell types", labels=pred_labels)

    except Exception as exc:
        import traceback
        _set("error", f"{exc}\n{traceback.format_exc()[-300:]}")


def _get_patch_bounds_um():
    """Return list of (x0, y0, x1, y1) in µm for each sopa patch, or None if none available."""
    with _sdata_lock:
        patches = _sdata_state.get("patches")
    if patches is None:
        return None
    try:
        patches_um = _sdata_transform_to_um(patches)
        return [geom.bounds for geom in patches_um.geometry]  # (minx, miny, maxx, maxy)
    except Exception as exc:
        print(f"  _get_patch_bounds_um: {exc}", flush=True)
        return None


def _baysor_run_single(tx_df, patch_dir, baysor_bin, scale, min_mol, use_prior, prior_conf,
                       scale_std=None,  n_clusters=10):
    """Run Baysor on tx_df, write outputs to patch_dir.
    Returns (cells_df, cell_bounds_dict, seg_df) or raises on failure.
    cells_df is indexed by string cell_id.
    """
    os.makedirs(patch_dir, exist_ok=True)
    tx_csv = os.path.join(patch_dir, "transcripts.csv")
    tx_df.to_csv(tx_csv, index=False)

    n_assigned = int((tx_df["cell_id"] != 0).sum())
    _use_prior = use_prior and n_assigned > 0
    if use_prior and not _use_prior:
        print(f"  Baysor [{patch_dir}]: no assigned transcripts — disabling prior", flush=True)

    cmd = [baysor_bin, "run"]
    if scale != -1:
        cmd += ["-s", str(scale)]
    if scale_std is not None:
        cmd += ["--scale-std", str(scale_std)]
    cmd += [
        "--min-molecules-per-cell", str(min_mol),
        "--n-clusters", str(n_clusters),
        "--polygon-format", "FeatureCollection",
        "-x", "x_location",
        "-y", "y_location",
        "-g", "feature_name",
        "-o", patch_dir,
        tx_csv,
    ]
    _prior_col = "nucleus_prior_id" if "nucleus_prior_id" in tx_df.columns else "cell_id"
    if _use_prior:
        cmd += ["--prior-segmentation-confidence", str(prior_conf), f":{_prior_col}"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, cwd=patch_dir)
    for line in proc.stdout:
        line = _strip_ansi(line).rstrip()
        if line:
            print(f"  [baysor] {line}", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Baysor exited with code {proc.returncode}")

    seg_csv = os.path.join(patch_dir, "segmentation.csv")
    if not os.path.exists(seg_csv):
        raise RuntimeError(f"segmentation.csv not found in {patch_dir}")

    seg = pd.read_csv(seg_csv)
    seg = seg[seg["cell"].notna()].copy()

    x_col = "x" if "x" in seg.columns else "x_location"
    y_col = "y" if "y" in seg.columns else "y_location"
    cells_df = (
        seg.groupby("cell")
        .agg(x_centroid=(x_col, "mean"), y_centroid=(y_col, "mean"),
             transcript_counts=(x_col, "count"))
        .rename_axis("cell_id")
    )

    cell_bounds: dict = {}
    for poly_name in ("segmentation_polygons_2d.json", "segmentation_polygons.json",
                      "polygons.json", "cell_polygons.json"):
        poly_path = os.path.join(patch_dir, poly_name)
        if os.path.exists(poly_path):
            break
        poly_path = None
    if poly_path:
        with open(poly_path) as f:
            gj = json.load(f)
        for feat in gj.get("features", []):
            props = feat.get("properties") or {}
            cid   = feat.get("id") or props.get("cell") or props.get("id")
            geom  = feat.get("geometry", {})
            gtype = geom.get("type")
            if cid is None:
                continue
            if gtype == "Polygon":
                ring = geom["coordinates"][0]
            elif gtype == "MultiPolygon":
                ring = max((poly[0] for poly in geom["coordinates"]), key=len)
            else:
                continue
            vx = np.array([p[0] for p in ring], dtype=np.float32)
            vy = np.array([p[1] for p in ring], dtype=np.float32)
            cell_bounds[cid] = (vx, vy)

    return cells_df, cell_bounds, seg


def _proseg_run_single(tx_df, patch_dir, proseg_bin, voxel_size, n_threads, n_samples=None,
                       recorded_samples=None, schedule=None,
                       nuclear_reassign_prob=None, prior_seg_prob=None):
    """Run Proseg on tx_df (written as parquet), write outputs to patch_dir.
    Returns (cells_df, cell_bounds_dict, tx_meta_df) or raises on failure.
    """
    import gzip
    os.makedirs(patch_dir, exist_ok=True)
    tx_path = os.path.join(patch_dir, "transcripts.parquet")
    tx_df.to_parquet(tx_path, index=False)

    poly_out    = os.path.join(patch_dir, "cell-polygons.geojson.gz")
    meta_out    = os.path.join(patch_dir, "cell-metadata.csv.gz")
    tx_meta_out = os.path.join(patch_dir, "transcript-metadata.csv.gz")

    cmd = [proseg_bin, "--xenium", tx_path,
           "--output-cell-polygons",       poly_out,
           "--output-cell-metadata",       meta_out,
           "--output-transcript-metadata", tx_meta_out]
    if voxel_size:
        cmd += ["--initial-voxel-size", str(voxel_size)]
    if n_threads:
        cmd += ["--nthreads", str(n_threads)]
    if n_samples:
        cmd += ["--ncomponents", str(n_samples)]
    if recorded_samples:
        cmd += ["--recorded-samples", str(recorded_samples)]
    if schedule:
        cmd += ["--schedule"] + schedule.split()
    if nuclear_reassign_prob is not None:
        cmd += ["--nuclear-reassignment-prob", str(nuclear_reassign_prob)]
    if prior_seg_prob is not None:
        cmd += ["--prior-seg-reassignment-prob", str(prior_seg_prob)]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, cwd=patch_dir)
    for line in proc.stdout:
        line = _strip_ansi(line).rstrip()
        if line:
            print(f"  [proseg] {line}", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Proseg exited with code {proc.returncode}")

    cell_meta = pd.read_csv(meta_out)
    cells_df = cell_meta.rename(columns={
        "cell": "cell_id", "centroid_x": "x_centroid",
        "centroid_y": "y_centroid", "population": "transcript_counts",
    }).set_index("cell_id")[["x_centroid", "y_centroid", "transcript_counts"]]

    cell_bounds: dict = {}
    if os.path.exists(poly_out):
        with gzip.open(poly_out, "rt") as f:
            gj = json.load(f)
        for feat in gj.get("features", []):
            props = feat.get("properties", {})
            cid   = props.get("cell") if props.get("cell") is not None else props.get("id")
            geom  = feat.get("geometry", {})
            gtype = geom.get("type")
            if cid is None:
                continue
            if gtype == "Polygon":
                rings = [geom["coordinates"][0]]
            elif gtype == "MultiPolygon":
                rings = [poly[0] for poly in geom["coordinates"]]
            else:
                continue
            ring = max(rings, key=len)
            vx = np.array([p[0] for p in ring], dtype=np.float32)
            vy = np.array([p[1] for p in ring], dtype=np.float32)
            cell_bounds[cid] = (vx, vy)

    tx_meta_df = None
    if os.path.exists(tx_meta_out):
        try:
            tx_meta_df = pd.read_csv(tx_meta_out)
        except Exception:
            pass

    return cells_df, cell_bounds, tx_meta_df


def _build_baysor_expr(seg_df, cells_df):
    """Build cell×gene CSR expression matrix from merged Baysor segmentation DataFrame."""
    gene_names_list = list(DATA["gene_names"])
    gene_to_idx = {g: i for i, g in enumerate(gene_names_list)}
    if "gene" not in seg_df.columns:
        return None
    cell_to_row = {c: i for i, c in enumerate(cells_df.index)}
    panel = seg_df[seg_df["gene"].isin(gene_to_idx) & seg_df["cell"].isin(cell_to_row)].copy()
    rows = panel["cell"].map(cell_to_row).astype(int).values
    cols = panel["gene"].map(gene_to_idx).astype(int).values
    mat = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(cells_df), len(gene_names_list))
    )
    print(f"  Baysor: built expression matrix {mat.shape}", flush=True)
    return mat


def _build_proseg_expr(tx_meta_df, cells_df):
    """Build cell×gene CSR expression matrix from merged Proseg transcript-metadata."""
    if tx_meta_df is None:
        return None
    gene_names_list = list(DATA["gene_names"])
    gene_to_idx = {g: i for i, g in enumerate(gene_names_list)}
    gene_col = "gene" if "gene" in tx_meta_df.columns else None
    cell_col = ("cell" if "cell" in tx_meta_df.columns
                else "assignment" if "assignment" in tx_meta_df.columns else None)
    if not gene_col or not cell_col:
        return None
    cell_to_row = {c: i for i, c in enumerate(cells_df.index)}
    assigned = tx_meta_df[tx_meta_df[cell_col].notna()].copy()
    if cell_col == "assignment":
        # Filter out proseg's "unassigned" sentinel value (0xFFFFFFFF)
        assigned = assigned[assigned[cell_col] != 4294967295]
    assigned[cell_col] = assigned[cell_col].astype(int).astype(str)
    valid = assigned[assigned[gene_col].isin(gene_to_idx) & assigned[cell_col].isin(cell_to_row)]
    rows = valid[cell_col].map(cell_to_row).astype(int).values
    cols = valid[gene_col].map(gene_to_idx).astype(int).values
    mat = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(cells_df), len(gene_names_list))
    )
    print(f"  Proseg: built expression matrix {mat.shape}", flush=True)
    return mat


def _load_zarr_obs_into_df(zarr_path: str, cells_df: pd.DataFrame) -> None:
    """Read obs columns from a SpatialData zarr and merge any missing ones into cells_df in-place."""
    try:
        import zarr as _zarr_mod
        _zgrp = _zarr_mod.open_group(zarr_path, mode="r")
        _obs = _zgrp["tables"]["table"]["obs"]
        _idx_key = _obs.attrs.get("_index", "_index")
        _zarr_idx = [str(v) for v in _obs[_idx_key][:]] if _idx_key in _obs else None
        # Strip patch prefix (e.g. "p0_") from zarr IDs if cells_df lacks it.
        # This happens when the zarr was built during a live patch run but the CSV
        # stores raw Baysor IDs without the patch prefix.
        if _zarr_idx is not None and len(_zarr_idx) > 0:
            import re as _re
            _stripped = [_re.sub(r'^p\d+_', '', _id) for _id in _zarr_idx]
            _n = min(30, len(_zarr_idx))
            if (len(set(_zarr_idx[:_n]) & set(cells_df.index[:_n].tolist())) == 0
                    and len(set(_stripped[:_n]) & set(cells_df.index[:_n].tolist())) > 0):
                print(f"  Stripping patch prefix from zarr obs index ({_zarr_idx[0]!r} → {_stripped[0]!r})", flush=True)
                _zarr_idx = _stripped
            elif len(set(_zarr_idx[:_n]) & set(cells_df.index[:_n].tolist())) == 0:
                # Genuinely different run — skip, let caller recompute
                print(f"  Warning: zarr obs index mismatch "
                      f"({_zarr_idx[0]!r} vs {cells_df.index[0]!r}). "
                      f"Will recompute cluster/UMAP.", flush=True)
                return
        for _col in _obs.keys():
            if _col in {"__categories", "_index"} or _col == _idx_key:
                continue
            if _col not in cells_df.columns:
                try:
                    _vals = _obs[_col][:]
                    if _zarr_idx is not None:
                        cells_df[_col] = pd.Series(_vals, index=_zarr_idx).reindex(cells_df.index).values
                    else:
                        cells_df[_col] = _vals
                except Exception:
                    pass
    except Exception as _ze:
        print(f"  Warning: could not load obs from zarr: {_ze}", flush=True)


def _load_zarr_boundaries_into_dict(sdata, cell_bounds: dict, tool_label: str) -> None:
    """Fill cell_bounds from zarr shapes element if it's currently empty."""
    if cell_bounds or "cell_boundaries" not in sdata.shapes:
        return
    _gdf = sdata.shapes["cell_boundaries"]
    for _cid, _geom in zip(_gdf.index, _gdf.geometry):
        try:
            _coords = list(_geom.exterior.coords)
            cell_bounds[str(_cid)] = ([c[0] for c in _coords], [c[1] for c in _coords])
        except Exception:
            pass
    if cell_bounds:
        print(f"  Loaded {len(cell_bounds):,} {tool_label} boundaries from zarr shapes", flush=True)


def _update_reseg_zarr_obs(zarr_path: str, cells_df: pd.DataFrame) -> None:
    """Update the obs (cell metadata) in a reseg SpatialData zarr with new columns from cells_df.
    Writes directly via zarr low-level API to avoid spatialdata's 'target path in use' error."""
    if not os.path.isdir(zarr_path):
        return
    try:
        import zarr as _zarr_mod
        _zroot = _zarr_mod.open_group(zarr_path, mode="r+", use_consolidated=False)
        if "tables" not in _zroot or "table" not in _zroot["tables"] \
                or "obs" not in _zroot["tables"]["table"]:
            return
        _obs_grp = _zroot["tables"]["table"]["obs"]
        existing_cols = set(_obs_grp.keys()) - {"__categories", "_index"}
        # Always re-write roi_* columns (may have changed due to add/delete)
        new_cols = [c for c in cells_df.columns
                    if c not in existing_cols or c.startswith("roi_")]
        if not new_cols:
            return
        # Read the existing index to align values correctly
        _idx_key = _obs_grp.attrs.get("_index", "_index")
        if _idx_key in _obs_grp:
            _existing_idx = [str(v) for v in _obs_grp[_idx_key][:]]
        else:
            _existing_idx = None
        # Detect stale zarr index (different run): update zarr index to match cells_df.
        # First strip any patch prefix (p0_, p1_, ...) from existing index before comparing.
        if _existing_idx is not None and len(_existing_idx) > 0:
            import re as _re2
            _stripped_existing = [_re2.sub(r'^p\d+_', '', _id) for _id in _existing_idx]
            _n = min(30, len(_existing_idx))
            _df_sample = set(cells_df.index[:_n].astype(str).tolist())
            _raw_overlap      = len(set(_existing_idx[:_n]) & _df_sample)
            _stripped_overlap = len(set(_stripped_existing[:_n]) & _df_sample)
            _overlap = _raw_overlap or _stripped_overlap
            if _overlap == 0:
                print(f"  Updating stale zarr obs index to match current cells_df", flush=True)
                _new_idx = np.array(list(cells_df.index.astype(str)), dtype=str)
                _obs_grp.create_dataset(_idx_key, data=_new_idx, shape=_new_idx.shape,
                                        dtype=_new_idx.dtype, overwrite=True)
                _existing_idx = list(cells_df.index.astype(str))
            elif _raw_overlap == 0 and _stripped_overlap > 0:
                # Zarr uses p{N}_ patch prefix but cells_df uses bare IDs — strip for reindex
                _existing_idx = _stripped_existing
        try:
            from anndata.io import write_elem as _write_elem
        except ImportError:
            from anndata.experimental import write_elem as _write_elem
        for col in new_cols:
            try:
                _ser = cells_df[col]
                if _existing_idx is not None:
                    _ser = _ser.reindex(_existing_idx)
                # For string/object columns bypass _write_elem (zarr warns on object dtype)
                # and write directly as fixed-width unicode numpy array.
                if _ser.dtype == object or pd.api.types.is_string_dtype(_ser):
                    _arr = np.array(_ser.fillna("").values, dtype=str)
                    _obs_grp.create_dataset(col, data=_arr, shape=_arr.shape, dtype=_arr.dtype, overwrite=True)
                else:
                    try:
                        _write_elem(_obs_grp, col, _ser)
                    except Exception:
                        _arr = _ser.values
                        _obs_grp.create_dataset(col, data=_arr, shape=_arr.shape, dtype=_arr.dtype, overwrite=True)
            except Exception as _col_err:
                print(f"  Warning: could not write obs col '{col}': {_col_err}", flush=True)
        print(f"  Updated zarr obs: added {new_cols} to {os.path.basename(zarr_path)}", flush=True)
    except Exception as _e:
        print(f"  Warning: failed to update zarr obs ({_e})", flush=True)


def _build_reseg_sdata(cells_df, cell_bounds, expr_mat, gene_names, source, out_dir) -> str:
    """Build and write a SpatialData zarr for a resegmentation result. Returns zarr path."""
    try:
        import spatialdata
        from spatialdata.models import TableModel, ShapesModel
        import anndata as ad
        import geopandas as gpd
        from shapely.geometry import Polygon
    except ImportError:
        print("  Warning: spatialdata/geopandas/shapely not available — skipping reseg zarr", flush=True)
        return ""

    var_df = pd.DataFrame({"is_imputed": False}, index=list(gene_names))
    obs_df = cells_df.copy()
    obs_df.index = obs_df.index.astype(str)

    n_genes = len(gene_names)
    if expr_mat is not None:
        X = sp.csr_matrix(expr_mat, dtype=np.float32)
    else:
        X = sp.csr_matrix((len(obs_df), n_genes), dtype=np.float32)

    adata = ad.AnnData(X=X, obs=obs_df, var=var_df)
    adata.uns["source"] = source
    adata.uns["imputed_genes"] = []
    try:
        adata = TableModel.parse(adata)
    except Exception:
        pass

    geometries = {}
    for cid, (vx, vy) in cell_bounds.items():
        try:
            geometries[str(cid)] = Polygon(zip(vx, vy))
        except Exception:
            pass

    if geometries:
        gdf = gpd.GeoDataFrame(
            geometry=list(geometries.values()),
            index=pd.Index(list(geometries.keys()), name="cell_id"),
        )
        try:
            shapes = ShapesModel.parse(gdf)
            sdata = spatialdata.SpatialData(
                tables={"table": adata},
                shapes={"cell_boundaries": shapes},
            )
        except Exception:
            sdata = spatialdata.SpatialData(tables={"table": adata})
    else:
        sdata = spatialdata.SpatialData(tables={"table": adata})

    zarr_path = os.path.join(out_dir, f"spatialdata_{source}.zarr")
    try:
        sdata.write(zarr_path)
        print(f"  Reseg zarr written: {zarr_path}", flush=True)
    except Exception as e:
        print(f"  Warning: failed to write reseg zarr: {e}", flush=True)
        return ""
    return zarr_path


def _fast_load_reseg_zarr(zarr_path: str):
    """Load a reseg result entirely from the zarr store using the low-level zarr API.

    Returns (cells_df, expr_mat, X_corrected, corr_imp_genes, cell_bounds).
    Raises on any error so callers can fall back to the CSV-based path.
    """
    import zarr as _zarr_mod

    _zroot = _zarr_mod.open_group(zarr_path, mode="r", use_consolidated=False)
    _tbl = _zroot["tables"]["table"]
    _obs = _tbl["obs"]

    # ── cells_df from obs ────────────────────────────────────────────
    _idx_key = _obs.attrs.get("_index", "_index")
    _ids = [str(v) for v in _obs[_idx_key][:]]
    _df_data = {}
    for _col in _obs.keys():
        if _col in {"__categories", "_index"} or _col == _idx_key:
            continue
        try:
            _df_data[_col] = _obs[_col][:]
        except Exception:
            pass
    if "x_centroid" not in _df_data or "y_centroid" not in _df_data:
        raise ValueError("zarr obs missing centroid columns — zarr predates centroid storage")
    cells_df = pd.DataFrame(_df_data, index=pd.Index(_ids, name="cell_id"))
    cells_df.index = cells_df.index.astype(str)

    # ── expr_mat from X ──────────────────────────────────────────────
    _xgrp = _tbl["X"]
    _enc = _xgrp.attrs.get("encoding-type", "")
    if "sparse" in _enc or "csr" in _enc or "csc" in _enc or (
            "data" in _xgrp and "indices" in _xgrp and "indptr" in _xgrp):
        _shape_attr = _xgrp.attrs.get("shape", None)
        if _shape_attr is not None:
            _shape = (int(_shape_attr[0]), int(_shape_attr[1]))
        else:
            _var_key = _tbl["var"].attrs.get("_index", "_index")
            _n_vars = len(_tbl["var"][_var_key][:])
            _shape = (len(_ids), _n_vars)
        expr_mat = sp.csr_matrix(
            (_xgrp["data"][:].astype(np.float32),
             _xgrp["indices"][:], _xgrp["indptr"][:]),
            shape=_shape,
        )
    else:
        expr_mat = sp.csr_matrix(np.array(_xgrp, dtype=np.float32))

    # ── X_corrected layer + imputed gene list ────────────────────────
    X_corrected = None
    corr_imp_genes = []
    try:
        if "layers" in _tbl and "X_corrected" in _tbl["layers"]:
            _lg = _tbl["layers"]["X_corrected"]
            _lshape = tuple(_lg.attrs.get("shape", expr_mat.shape))
            X_corrected = sp.csr_matrix(
                (_lg["data"][:].astype(np.float32),
                 _lg["indices"][:], _lg["indptr"][:]),
                shape=_lshape,
            )
        if "uns" in _tbl:
            try:
                _ig = _tbl["uns"].get("split_corrected_imputed_genes")
                if _ig is not None:
                    corr_imp_genes = [str(g) for g in _ig[:]]
            except Exception:
                pass
    except Exception:
        pass

    # ── boundaries from shapes parquet ──────────────────────────────
    cell_bounds: dict = {}
    try:
        import geopandas as _gpd
        _pq = os.path.join(zarr_path, "shapes", "cell_boundaries", "shapes.parquet")
        if os.path.exists(_pq):
            _gdf = _gpd.read_parquet(_pq)
            for _cid, _geom in zip(_gdf.index, _gdf.geometry):
                try:
                    if _geom.geom_type == "Polygon":
                        _arr = np.array(_geom.exterior.coords)
                    elif _geom.geom_type == "MultiPolygon":
                        _arr = np.array(max(_geom.geoms, key=lambda g: g.area).exterior.coords)
                    else:
                        continue
                    cell_bounds[str(_cid)] = (_arr[:, 0].tolist(), _arr[:, 1].tolist())
                except Exception:
                    pass
            if cell_bounds:
                print(f"  Loaded {len(cell_bounds):,} boundaries from zarr parquet", flush=True)
    except Exception:
        pass

    return cells_df, expr_mat, X_corrected, corr_imp_genes, cell_bounds


def _load_cached_baysor(out_dir: str) -> None:
    """Load a cached Baysor result from disk into _baysor_state."""
    def _set(status, message, result=None):
        with _baysor_lock:
            _baysor_state["status"]  = status
            _baysor_state["message"] = message
            if result is not None:
                _baysor_state["result"] = result
    print(f"Loading cached Baysor: {os.path.basename(out_dir)}…", flush=True)
    try:
        _set("running", "Loading cached Baysor result…")
        # Restore SpaGE result reference written by _run_spage_imputation
        _spage_rpath, _spage_rgenes = None, None
        _spage_ref_file = os.path.join(out_dir, "spage_result.json")
        if os.path.exists(_spage_ref_file):
            try:
                import json as _jref
                with open(_spage_ref_file) as _jf:
                    _sref = _jref.load(_jf)
                if os.path.isdir(_sref.get("path", "")):
                    _spage_rpath  = _sref["path"]
                    _spage_rgenes = _sref.get("genes", [])
                    print(f"  Baysor: restored SpaGE result ({len(_spage_rgenes)} genes)", flush=True)
            except Exception:
                pass
        zarr_path_fast = os.path.join(out_dir, "spatialdata_baysor.zarr")
        if os.path.isdir(zarr_path_fast):
            try:
                _set("running", "Loading Baysor from zarr (fast path)…")
                cells_df, expr_mat, _baysor_corr, _baysor_corr_imp, cell_bounds = \
                    _fast_load_reseg_zarr(zarr_path_fast)
                n_cells = len(cells_df)
                _has_cluster = any(c.startswith("cluster") for c in cells_df.columns)
                if ("umap_1" not in cells_df.columns or not _has_cluster) and expr_mat is not None:
                    _set("running", f"Computing UMAP and clusters for {n_cells:,} cells (one-time)…")
                    _compute_reseg_clusters_umap(cells_df, expr_mat,
                                                 progress_fn=lambda msg: _set("running", msg))
                    _update_reseg_zarr_obs(zarr_path_fast, cells_df)
                with _roi_lock:
                    _roi_apply_metadata_to_df(cells_df, _roi_state["rois"])
                _set("done", f"Loaded — {n_cells:,} cells", result={
                    "cells_df": cells_df, "cell_bounds": cell_bounds,
                    "out_dir": out_dir, "expr": expr_mat, "source": "baysor",
                    "sdata_path": zarr_path_fast,
                    "split_corrected_expr": _baysor_corr,
                    "split_corrected_imputed_genes": _baysor_corr_imp,
                    "spage_result_path": _spage_rpath,
                    "spage_result_genes": _spage_rgenes,
                })
                print(f"Loaded cached Baysor (fast): {n_cells:,} cells, "
                      f"{len(cell_bounds):,} boundaries", flush=True)
                return
            except Exception as _fe:
                print(f"  Fast zarr load failed ({_fe}), falling back to CSV…", flush=True)

        # ── Slow path: rebuild from CSV files ────────────────────────
        # Collect all segmentation.csv files (may be one per patch subdir)
        seg_parts = []
        for root, dirs, files in os.walk(out_dir):
            if "segmentation.csv" in files:
                seg_parts.append(pd.read_csv(os.path.join(root, "segmentation.csv")))
        if not seg_parts:
            _set("error", f"segmentation.csv not found in {out_dir}")
            return
        seg = pd.concat(seg_parts, ignore_index=True)
        seg = seg[seg["cell"].notna()].copy()
        seg["cell"] = seg["cell"].astype(str)
        x_col = "x" if "x" in seg.columns else "x_location"
        y_col = "y" if "y" in seg.columns else "y_location"
        cells_df = (
            seg.groupby("cell")
            .agg(x_centroid=(x_col, "mean"), y_centroid=(y_col, "mean"),
                 transcript_counts=(x_col, "count"))
            .rename_axis("cell_id")
        )
        cells_df.index = cells_df.index.astype(str)
        # Build cell boundaries from polygon files
        cell_bounds: dict = {}
        for root, dirs, files in os.walk(out_dir):
            for poly_name in ("segmentation_polygons_2d.json", "segmentation_polygons.json",
                              "polygons.json", "cell_polygons.json"):
                if poly_name in files:
                    poly_path = os.path.join(root, poly_name)
                    with open(poly_path) as f:
                        geo = json.load(f)
                    for feat in geo.get("features", []):
                        props = feat.get("properties", {})
                        cid = str(props.get("cell", props.get("cell_id", feat.get("id", ""))))
                        coords = feat.get("geometry", {}).get("coordinates", [[]])[0]
                        if coords:
                            cell_bounds[cid] = ([c[0] for c in coords], [c[1] for c in coords])
                    break  # only one polygon file per dir needed
        expr_mat = _build_baysor_expr(seg, cells_df)
        n_cells = len(cells_df)

        zarr_path = os.path.join(out_dir, "spatialdata_baysor.zarr")

        # ── Load or build zarr ────────────────────────────────────────────
        _baysor_corr     = None
        _baysor_corr_imp = []
        if not os.path.isdir(zarr_path):
            # First time: compute clusters/UMAP then build zarr (includes them)
            if expr_mat is not None:
                _set("running", f"Computing UMAP and clusters for {n_cells:,} cells…")
                _compute_reseg_clusters_umap(cells_df, expr_mat,
                                             progress_fn=lambda msg: _set("running", msg))
            zarr_path = _build_reseg_sdata(
                cells_df, cell_bounds, expr_mat, DATA["gene_names"], "baysor", out_dir
            )
        else:
            # Zarr exists — load obs columns (cluster, UMAP) from it directly
            # to avoid recomputing and the 'target path in use' write-back error.
            _load_zarr_obs_into_df(zarr_path, cells_df)
            _zarr_obs_cols = [c for c in cells_df.columns
                              if c not in ("x_centroid", "y_centroid", "transcript_counts")]
            print(f"  Baysor cache: loaded obs cols from zarr: {_zarr_obs_cols}", flush=True)

            # If cluster/UMAP still missing (very old cache or ID mismatch), compute and write back
            _has_cluster = any(c.startswith("cluster") for c in cells_df.columns)
            if ("umap_1" not in cells_df.columns or not _has_cluster) and expr_mat is not None:
                _set("running", f"Computing UMAP and clusters for {n_cells:,} cells (one-time migration)…")
                _compute_reseg_clusters_umap(cells_df, expr_mat,
                                             progress_fn=lambda msg: _set("running", msg))
                _update_reseg_zarr_obs(zarr_path, cells_df)

            # Load corrected expr + cell boundaries from zarr
            try:
                import spatialdata as _sd_b
                _sdata_b = _sd_b.read_zarr(zarr_path)
                _adata_b = _sdata_b.tables["table"]
                _baysor_corr = (
                    sp.csr_matrix(_adata_b.layers["X_corrected"])
                    if "X_corrected" in _adata_b.layers else None
                )
                _baysor_corr_imp = list(_adata_b.uns.get("split_corrected_imputed_genes", []))
                _load_zarr_boundaries_into_dict(_sdata_b, cell_bounds, "Baysor")
            except Exception as _ze2:
                print(f"  Warning: could not read Baysor zarr: {_ze2}", flush=True)

        with _roi_lock:
            _roi_apply_metadata_to_df(cells_df, _roi_state["rois"])
        _set("done", f"Loaded — {n_cells:,} cells", result={
            "cells_df": cells_df, "cell_bounds": cell_bounds,
            "out_dir": out_dir, "expr": expr_mat, "source": "baysor",
            "sdata_path": zarr_path,
            "split_corrected_expr": _baysor_corr,
            "split_corrected_imputed_genes": _baysor_corr_imp,
            "spage_result_path": _spage_rpath,
            "spage_result_genes": _spage_rgenes,
        })
        print(f"Loaded cached Baysor: {n_cells:,} cells, {len(cell_bounds):,} boundaries "
              f"from {os.path.basename(out_dir)}", flush=True)
    except Exception as exc:
        import traceback; traceback.print_exc()
        _set("error", str(exc)[:300])


def _load_proseg_patch_boundaries(out_dir: str, patch_names: list) -> dict:
    """Load Proseg cell boundaries from per-patch polygon files with global ID remapping.

    Applies the same ID offset used for cell-metadata: global_id = local_id + 1 + i * STRIDE.
    Returns a cell_bounds dict {str(global_id): ([x…], [y…])}.
    """
    _STRIDE = 10_000_000
    cell_bounds: dict = {}
    for _i, _pname in enumerate(patch_names):
        _offset = 1 + _i * _STRIDE
        _pdir = os.path.join(out_dir, _pname)
        for _poly_name in ("cell-polygons.geojson.gz", "polygons.geojson",
                           "polygons.json", "cell_polygons.json"):
            _poly_path = os.path.join(_pdir, _poly_name)
            if not os.path.exists(_poly_path):
                continue
            try:
                import gzip as _gz
                _opener = _gz.open if _poly_name.endswith(".gz") else open
                with _opener(_poly_path, "rt") as _pf:
                    _geo = json.load(_pf)
                for _feat in _geo.get("features", []):
                    _props = _feat.get("properties", {})
                    _local = _props.get("cell_id", _props.get("cell", _feat.get("id", "")))
                    try:
                        _gid = str(int(_local) + _offset)
                    except (ValueError, TypeError):
                        _gid = str(_local)
                    _geom = _feat.get("geometry", {})
                    _gtype = _geom.get("type")
                    if _gtype == "Polygon":
                        _rings = [_geom["coordinates"][0]]
                    elif _gtype == "MultiPolygon":
                        _rings = [_p[0] for _p in _geom["coordinates"]]
                    else:
                        _rings = [_geom.get("coordinates", [[]])[0]]
                    _coords = max(_rings, key=len) if _rings else []
                    if _coords:
                        cell_bounds[_gid] = ([c[0] for c in _coords], [c[1] for c in _coords])
            except Exception as _pe:
                print(f"  Proseg: patch polygon load error in {_pname}: {_pe}", flush=True)
            break  # stop after first polygon file found in this patch
    return cell_bounds


def _load_cached_proseg(out_dir: str) -> None:
    """Load a cached Proseg result from disk into _proseg_state."""
    def _set(status, message, result=None):
        with _proseg_lock:
            _proseg_state["status"]  = status
            _proseg_state["message"] = message
            if result is not None:
                _proseg_state["result"] = result
    print(f"Loading cached Proseg: {os.path.basename(out_dir)}…", flush=True)
    try:
        _set("running", "Loading cached Proseg result…")
        # Restore SpaGE result reference written by _run_spage_imputation
        _spage_rpath, _spage_rgenes = None, None
        _spage_ref_file = os.path.join(out_dir, "spage_result.json")
        if os.path.exists(_spage_ref_file):
            try:
                import json as _jref
                with open(_spage_ref_file) as _jf:
                    _sref = _jref.load(_jf)
                if os.path.isdir(_sref.get("path", "")):
                    _spage_rpath  = _sref["path"]
                    _spage_rgenes = _sref.get("genes", [])
                    print(f"  Proseg: restored SpaGE result ({len(_spage_rgenes)} genes)", flush=True)
            except Exception:
                pass
        zarr_path_fast = os.path.join(out_dir, "spatialdata_proseg.zarr")
        if os.path.isdir(zarr_path_fast):
            try:
                _set("running", "Loading Proseg from zarr (fast path)…")
                cells_df, expr_mat, _proseg_corr, _proseg_corr_imp, cell_bounds = \
                    _fast_load_reseg_zarr(zarr_path_fast)
                n_cells = len(cells_df)
                _has_cluster = any(c.startswith("cluster") for c in cells_df.columns)
                if ("umap_1" not in cells_df.columns or not _has_cluster) and expr_mat is not None:
                    _set("running", f"Computing UMAP and clusters for {n_cells:,} cells (one-time)…")
                    _compute_reseg_clusters_umap(cells_df, expr_mat,
                                                 progress_fn=lambda msg: _set("running", msg))
                    _update_reseg_zarr_obs(zarr_path_fast, cells_df)
                # Patch runs build the zarr without cell_boundaries shapes — load from polygon files
                if not cell_bounds:
                    _fp_patch_names = sorted([
                        _d for _d in os.listdir(out_dir)
                        if os.path.isdir(os.path.join(out_dir, _d)) and _d.startswith("patch_")
                        and os.path.exists(os.path.join(out_dir, _d, "cell-metadata.csv.gz"))
                    ])
                    if _fp_patch_names:
                        cell_bounds = _load_proseg_patch_boundaries(out_dir, _fp_patch_names)
                        if cell_bounds:
                            print(f"  Proseg: loaded {len(cell_bounds):,} patch boundaries "
                                  f"(polygon files)", flush=True)
                with _roi_lock:
                    _roi_apply_metadata_to_df(cells_df, _roi_state["rois"])
                _set("done", f"Loaded — {n_cells:,} cells", result={
                    "cells_df": cells_df, "cell_bounds": cell_bounds,
                    "out_dir": out_dir, "expr": expr_mat, "source": "proseg",
                    "sdata_path": zarr_path_fast,
                    "split_corrected_expr": _proseg_corr,
                    "split_corrected_imputed_genes": _proseg_corr_imp,
                    "spage_result_path": _spage_rpath,
                    "spage_result_genes": _spage_rgenes,
                })
                print(f"Loaded cached Proseg (fast): {n_cells:,} cells, "
                      f"{len(cell_bounds):,} boundaries", flush=True)
                return
            except Exception as _fe:
                print(f"  Fast zarr load failed ({_fe}), falling back to CSV…", flush=True)

        # ── Slow path: rebuild from CSV files ────────────────────────
        # Detect patch-based vs single run.
        # Patch runs use per-patch local cell IDs that must be remapped to global IDs
        # matching the scheme used during the live run: global_id = local_id + 1 + i*STRIDE
        _CELL_ID_STRIDE   = 10_000_000
        _PROSEG_UNASSIGNED = 4294967295  # 0xFFFFFFFF sentinel for unassigned transcripts
        _patch_names = sorted([
            d for d in os.listdir(out_dir)
            if os.path.isdir(os.path.join(out_dir, d)) and d.startswith("patch_")
            and os.path.exists(os.path.join(out_dir, d, "cell-metadata.csv.gz"))
        ])
        meta_parts: list = []
        tx_parts:   list = []
        if _patch_names:
            # Patch run: apply global-ID stride (replicates live-run numbering)
            for _i, _pname in enumerate(_patch_names):
                _offset = 1 + _i * _CELL_ID_STRIDE
                _pdir   = os.path.join(out_dir, _pname)
                _df     = pd.read_csv(os.path.join(_pdir, "cell-metadata.csv.gz"))
                _remap  = {int(lid): int(lid) + _offset for lid in _df["cell"].values}
                _df["cell"] = _df["cell"].map(_remap)
                meta_parts.append(_df)
                _tx_path = os.path.join(_pdir, "transcript-metadata.csv.gz")
                if os.path.exists(_tx_path):
                    _tx_df = pd.read_csv(_tx_path)
                    if "assignment" in _tx_df.columns:
                        _mask = _tx_df["assignment"] != _PROSEG_UNASSIGNED
                        _tx_df = _tx_df.copy()
                        _tx_df.loc[_mask, "assignment"] = (
                            _tx_df.loc[_mask, "assignment"].map(_remap)
                        )
                    elif "cell" in _tx_df.columns:
                        _tx_df = _tx_df.copy()
                        _tx_df["cell"] = _tx_df["cell"].map(_remap)
                    tx_parts.append(_tx_df)
        else:
            # Single run: flat walk (no ID remapping needed)
            for _root, _dirs, _files in os.walk(out_dir):
                if "cell-metadata.csv.gz" in _files:
                    meta_parts.append(pd.read_csv(os.path.join(_root, "cell-metadata.csv.gz")))
                if "transcript-metadata.csv.gz" in _files:
                    tx_parts.append(pd.read_csv(os.path.join(_root, "transcript-metadata.csv.gz")))
        if not meta_parts:
            _set("error", f"cell-metadata.csv.gz not found in {out_dir}")
            return
        cell_meta = pd.concat(meta_parts, ignore_index=True)
        tx_meta   = pd.concat(tx_parts,   ignore_index=True) if tx_parts else pd.DataFrame()
        cells_df = cell_meta.rename(columns={
            "cell": "cell_id", "centroid_x": "x_centroid", "centroid_y": "y_centroid",
            "population": "transcript_counts", "volume": "cell_area",
        })
        if "cell_id" in cells_df.columns:
            cells_df = cells_df.set_index("cell_id")
        cells_df.index = cells_df.index.astype(str)
        # Build cell boundaries from polygon files.
        cell_bounds: dict = {}
        if _patch_names:
            cell_bounds = _load_proseg_patch_boundaries(out_dir, _patch_names)
            if cell_bounds:
                print(f"  Proseg: loaded {len(cell_bounds):,} patch boundaries", flush=True)
        else:
            for root, dirs, files in os.walk(out_dir):
                for poly_name in ("cell-polygons.geojson.gz", "polygons.geojson", "polygons.json",
                                  "cell_polygons.json", "segmentation_polygons.json"):
                    if poly_name in files:
                        poly_path = os.path.join(root, poly_name)
                        import gzip as _gzip
                        _opener = _gzip.open if poly_name.endswith(".gz") else open
                        with _opener(poly_path, "rt") as f:
                            geo = json.load(f)
                        for feat in geo.get("features", []):
                            props = feat.get("properties", {})
                            cid = str(props.get("cell_id", props.get("cell", feat.get("id", ""))))
                            geom = feat.get("geometry", {})
                            gtype = geom.get("type")
                            if gtype == "Polygon":
                                rings = [geom["coordinates"][0]]
                            elif gtype == "MultiPolygon":
                                rings = [poly[0] for poly in geom["coordinates"]]
                            else:
                                rings = [geom.get("coordinates", [[]])[0]]
                            coords = max(rings, key=len) if rings else []
                            if coords:
                                cell_bounds[cid] = ([c[0] for c in coords], [c[1] for c in coords])
                        break
        expr_mat = _build_proseg_expr(tx_meta, cells_df) if not tx_meta.empty else None
        n_cells = len(cells_df)

        zarr_path = os.path.join(out_dir, "spatialdata_proseg.zarr")

        # ── Load or build zarr ────────────────────────────────────────────
        _proseg_corr     = None
        _proseg_corr_imp = []
        if not os.path.isdir(zarr_path):
            # First time: compute clusters/UMAP then build zarr (includes them)
            if expr_mat is not None:
                _set("running", f"Computing UMAP and clusters for {n_cells:,} cells…")
                _compute_reseg_clusters_umap(cells_df, expr_mat,
                                             progress_fn=lambda msg: _set("running", msg))
            zarr_path = _build_reseg_sdata(
                cells_df, cell_bounds, expr_mat, DATA["gene_names"], "proseg", out_dir
            )
        else:
            # Zarr exists — load obs columns (cluster, UMAP) from it directly
            _load_zarr_obs_into_df(zarr_path, cells_df)
            _zarr_obs_cols_p = [c for c in cells_df.columns
                                if c not in ("x_centroid", "y_centroid", "transcript_counts")]
            print(f"  Proseg cache: loaded obs cols from zarr: {_zarr_obs_cols_p}", flush=True)

            # If cluster/UMAP still missing (very old cache or ID mismatch), compute and write back
            _has_cluster_p = any(c.startswith("cluster") for c in cells_df.columns)
            if ("umap_1" not in cells_df.columns or not _has_cluster_p) and expr_mat is not None:
                _set("running", f"Computing UMAP and clusters for {n_cells:,} cells (one-time migration)…")
                _compute_reseg_clusters_umap(cells_df, expr_mat,
                                             progress_fn=lambda msg: _set("running", msg))
                _update_reseg_zarr_obs(zarr_path, cells_df)

            # Load expr (if missing), corrected expr + cell boundaries from zarr
            try:
                import spatialdata as _sd_p
                _sdata_p = _sd_p.read_zarr(zarr_path)
                _adata_p = _sdata_p.tables["table"]
                if expr_mat is None:
                    expr_mat = sp.csr_matrix(_adata_p.X)
                    print(f"  Proseg: loaded expr from zarr {expr_mat.shape}", flush=True)
                    # Align cells_df to zarr obs order (zarr may have fewer cells than CSV)
                    zarr_ids = [str(i) for i in _adata_p.obs_names]
                    if cells_df.index.duplicated().any():
                        cells_df = cells_df[~cells_df.index.duplicated(keep="first")]
                    cells_df = cells_df.reindex(zarr_ids).dropna(how="all")
                    cells_df.index = cells_df.index.astype(str)
                _proseg_corr = (
                    sp.csr_matrix(_adata_p.layers["X_corrected"])
                    if "X_corrected" in _adata_p.layers else None
                )
                _proseg_corr_imp = list(_adata_p.uns.get("split_corrected_imputed_genes", []))
                _load_zarr_boundaries_into_dict(_sdata_p, cell_bounds, "Proseg")
            except Exception as _ze2:
                print(f"  Warning: could not read Proseg zarr: {_ze2}", flush=True)

        with _roi_lock:
            _roi_apply_metadata_to_df(cells_df, _roi_state["rois"])
        _set("done", f"Loaded — {n_cells:,} cells", result={
            "cells_df": cells_df, "cell_bounds": cell_bounds,
            "out_dir": out_dir, "expr": expr_mat, "source": "proseg",
            "sdata_path": zarr_path,
            "split_corrected_expr": _proseg_corr,
            "split_corrected_imputed_genes": _proseg_corr_imp,
            "spage_result_path": _spage_rpath,
            "spage_result_genes": _spage_rgenes,
        })
        print(f"Loaded cached Proseg: {n_cells:,} cells, {len(cell_bounds):,} boundaries "
              f"from {os.path.basename(out_dir)}", flush=True)
    except Exception as exc:
        import traceback; traceback.print_exc()
        _set("error", str(exc)[:300])


def _run_baysor(scale: float, min_mol: int, use_prior: bool, prior_conf: float,
                x_min=None, x_max=None, y_min=None, y_max=None,
                scale_std=None, n_clusters=10, use_patches=True) -> None:
    """Background thread: run Baysor resegmentation, using sopa patches if available."""

    def _set(status, message, result=None):
        with _baysor_lock:
            _baysor_state["status"]  = status
            _baysor_state["message"] = message
            if result is not None:
                _baysor_state["result"] = result

    BAYSOR_CANDIDATES = [
        "baysor",
        os.path.expanduser("~/.julia/bin/baysor"),
        "/usr/local/bin/baysor",
    ]

    def _find_baysor():
        for candidate in BAYSOR_CANDIDATES:
            try:
                r = subprocess.run([candidate, "--version"], capture_output=True, timeout=10)
                if r.returncode == 0:
                    return candidate
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        return None

    try:
        baysor_bin = _find_baysor()
        if baysor_bin is None:
            _set("error", "baysor not found. Install from github.com/kharchenkolab/Baysor")
            return
        print(f"  Using baysor: {baysor_bin}", flush=True)

        _param_str = (f"s{scale}_ss{scale_std}_m{min_mol}_i{n_clusters}_p{int(use_prior)}"
                      f"{'_c'+str(prior_conf) if use_prior else ''}"
                      f"_x{x_min or 'A'}-{x_max or 'A'}_y{y_min or 'A'}-{y_max or 'A'}")
        _param_tag = hashlib.md5(_param_str.encode()).hexdigest()[:8]
        out_dir = os.path.join(os.path.expanduser("~"), ".xenium_explorer_cache",
                               f"baysor_{os.path.basename(DATA['data_dir'])}_{_param_tag}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"  Baysor: cache dir = {os.path.basename(out_dir)} ({_param_str})", flush=True)

        # ── Load & prepare all transcripts (once) ────────────────────────
        _set("running", "Loading transcripts…")
        import pyarrow.parquet as _pq
        _tx_path = os.path.join(DATA["data_dir"], "transcripts.parquet")
        _tx_schema_cols = _pq.read_schema(_tx_path).names
        _tx_cols = ["x_location", "y_location", "feature_name", "cell_id"]
        if "overlaps_nucleus" in _tx_schema_cols:
            _tx_cols.append("overlaps_nucleus")
        tx_all = pd.read_parquet(
            os.path.join(DATA["data_dir"], "transcripts.parquet"),
            columns=_tx_cols,
        )
        # Map Xenium string cell IDs → integers (0 = unassigned)
        cid = tx_all["cell_id"].astype(str)
        _unassigned = {"UNASSIGNED", "nan", "None", "<NA>", ""}
        unique_assigned = [c for c in cid.unique() if c not in _unassigned]
        id_map = {c: i + 1 for i, c in enumerate(unique_assigned)}
        tx_all["cell_id"] = cid.map(id_map).fillna(0).astype("int64")
        # Build nucleus_prior_id: cell_id only for transcripts overlapping a nucleus.
        # This gives Baysor a tighter nuclear prior instead of the full cell boundary prior.
        if "overlaps_nucleus" in tx_all.columns:
            tx_all["nucleus_prior_id"] = (
                tx_all["cell_id"].where(tx_all["overlaps_nucleus"].astype(bool), 0)
            )
            print(f"  Baysor: nucleus prior — "
                  f"{int((tx_all['nucleus_prior_id'] != 0).sum()):,} / {len(tx_all):,} "
                  f"transcripts inside nuclei", flush=True)
        else:
            tx_all["nucleus_prior_id"] = tx_all["cell_id"]
            print("  Baysor: overlaps_nucleus column not found — using cell_id as prior", flush=True)

        # Apply user's region filter
        if x_min is not None: tx_all = tx_all[tx_all["x_location"] >= x_min]
        if x_max is not None: tx_all = tx_all[tx_all["x_location"] <= x_max]
        if y_min is not None: tx_all = tx_all[tx_all["y_location"] >= y_min]
        if y_max is not None: tx_all = tx_all[tx_all["y_location"] <= y_max]

        # ── Check for sopa patches ────────────────────────────────────────
        patch_bounds = _get_patch_bounds_um() if use_patches else None
        # Intersect patches with user region filter
        if patch_bounds and any(v is not None for v in [x_min, x_max, y_min, y_max]):
            filtered = []
            for px0, py0, px1, py1 in patch_bounds:
                if x_min is not None and px1 < x_min: continue
                if x_max is not None and px0 > x_max: continue
                if y_min is not None and py1 < y_min: continue
                if y_max is not None and py0 > y_max: continue
                filtered.append((
                    max(px0, x_min or px0), max(py0, y_min or py0),
                    min(px1, x_max or px1), min(py1, y_max or py1),
                ))
            patch_bounds = filtered or None

        if patch_bounds:
            n_patches = len(patch_bounds)
            print(f"  Baysor: using {n_patches} sopa patch(es)", flush=True)

            all_cells: list   = []
            all_bounds: dict  = {}
            all_seg: list     = []
            buf = float(scale)  # overlap buffer = cell radius

            for pi, (px0, py0, px1, py1) in enumerate(patch_bounds):
                _set("running", f"Baysor patch {pi + 1}/{n_patches}…")
                # Filter with overlap buffer so cells at edges are fully captured
                tx_patch = tx_all[
                    (tx_all["x_location"] >= px0 - buf) & (tx_all["x_location"] <= px1 + buf) &
                    (tx_all["y_location"] >= py0 - buf) & (tx_all["y_location"] <= py1 + buf)
                ]
                if len(tx_patch) < min_mol * 2:
                    print(f"  Baysor: patch {pi + 1} has too few transcripts — skipping", flush=True)
                    continue
                print(f"  Baysor: patch {pi + 1}/{n_patches}: {len(tx_patch):,} transcripts", flush=True)

                patch_dir = os.path.join(out_dir, f"patch_{pi:04d}")
                try:
                    cells_i, bounds_i, seg_i = _baysor_run_single(
                        tx_patch, patch_dir, baysor_bin, scale, min_mol, use_prior, prior_conf,
                        scale_std=scale_std, n_clusters=n_clusters
                    )
                except Exception as pe:
                    print(f"  Baysor: patch {pi + 1} failed: {pe}", flush=True)
                    continue

                # Boundary resolution: keep cells whose centroid is inside this patch (non-buffered)
                in_patch = (
                    (cells_i["x_centroid"] >= px0) & (cells_i["x_centroid"] < px1) &
                    (cells_i["y_centroid"] >= py0) & (cells_i["y_centroid"] < py1)
                )
                cells_i = cells_i[in_patch]
                valid_ids = set(cells_i.index)

                # Remap IDs to be globally unique: "p{pi}_{original_id}"
                remap = {old: f"p{pi}_{old}" for old in valid_ids}
                cells_i.index = pd.Index([remap[c] for c in cells_i.index], name="cell_id")
                all_cells.append(cells_i)
                for old_id, bnd in bounds_i.items():
                    if old_id in valid_ids:
                        all_bounds[remap[old_id]] = bnd
                seg_i_valid = seg_i[seg_i["cell"].isin(valid_ids)].copy()
                seg_i_valid["cell"] = seg_i_valid["cell"].map(remap)
                all_seg.append(seg_i_valid)

            if not all_cells:
                _set("error", "No cells found across all patches. Check Baysor output.")
                return

            cells_df  = pd.concat(all_cells)
            cell_bounds = all_bounds
            seg_merged  = pd.concat(all_seg)
            print(f"  Baysor: merged {len(cells_df):,} cells from {n_patches} patch(es)", flush=True)

        else:
            # ── Single run (no patches or no sopa data) ───────────────────
            n_assigned = int((tx_all["cell_id"] != 0).sum())
            print(f"  Baysor: {n_assigned:,} / {len(tx_all):,} transcripts have prior cell assignment", flush=True)
            _up = use_prior and n_assigned > 0
            if use_prior and not _up:
                print("  Baysor: no assigned transcripts in region — disabling prior segmentation", flush=True)
            _set("running", "Running Baysor (this may take 10–30 min)…")
            try:
                cells_df, cell_bounds, seg_merged = _baysor_run_single(
                    tx_all, out_dir, baysor_bin, scale, min_mol, _up, prior_conf,
                    scale_std=scale_std, n_clusters=n_clusters
                )
            except Exception as e:
                _set("error", str(e)[:300])
                return

        # ── Build expression matrix ───────────────────────────────────────
        _set("running", "Building expression matrix…")
        expr_mat = _build_baysor_expr(seg_merged, cells_df)

        n_cells = len(cells_df)
        import json as _json
        _params_out = {
            "tool": "baysor", "scale": scale, "scale_std": scale_std,
            "min_mol": min_mol, "n_clusters": n_clusters,
            "use_prior": use_prior, "prior_conf": prior_conf,
            "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
            "n_cells": n_cells, "param_tag": _param_tag,
        }
        with open(os.path.join(out_dir, "params.json"), "w") as _f:
            _json.dump(_params_out, _f)
        _set("running", "Computing UMAP and clusters…")
        _compute_reseg_clusters_umap(cells_df, expr_mat,
                                     progress_fn=lambda msg: _set("running", msg))
        _set("running", "Writing SpatialData zarr…")
        zarr_path = _build_reseg_sdata(
            cells_df, cell_bounds, expr_mat, DATA["gene_names"], "baysor", out_dir
        )
        _set("done", f"Done — {n_cells:,} cells", result={
            "cells_df":    cells_df,
            "cell_bounds": cell_bounds,
            "out_dir":     out_dir,
            "expr":        expr_mat,
            "source":      "baysor",
            "sdata_path":  zarr_path,
        })
        print(f"  Baysor: {n_cells:,} cells segmented", flush=True)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set("error", str(exc)[:300])


# ─── Proseg segmentation ──────────────────────────────────────────────────────

def _run_proseg(voxel_size=None, n_threads=None, n_samples=None,
                recorded_samples=None, schedule=None,
                nuclear_reassign_prob=None, prior_seg_prob=None,
                x_min=None, x_max=None, y_min=None, y_max=None) -> None:
    """Background thread: run Proseg resegmentation, using sopa patches if available."""

    def _set(status, message, result=None):
        with _proseg_lock:
            _proseg_state["status"]  = status
            _proseg_state["message"] = message
            if result is not None:
                _proseg_state["result"] = result

    PROSEG_CANDIDATES = [
        "proseg",
        os.path.expanduser("~/.cargo/bin/proseg"),
        "/usr/local/bin/proseg",
    ]

    def _find_proseg():
        import shutil
        found = shutil.which("proseg")
        if found:
            return found
        for candidate in PROSEG_CANDIDATES:
            try:
                r = subprocess.run([candidate, "--version"], capture_output=True, timeout=10)
                if r.returncode == 0:
                    return candidate
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        return None

    try:
        proseg_bin = _find_proseg()
        if proseg_bin is None:
            _set("error", "proseg not found. Install via: conda install bioconda::rust-proseg")
            return
        print(f"  Using proseg: {proseg_bin}", flush=True)

        _param_str = (f"v{voxel_size or 'A'}_t{n_threads or 'A'}"
                      f"_x{x_min or 'A'}-{x_max or 'A'}_y{y_min or 'A'}-{y_max or 'A'}")
        _param_tag = hashlib.md5(_param_str.encode()).hexdigest()[:8]
        out_dir = os.path.join(os.path.expanduser("~"), ".xenium_explorer_cache",
                               f"proseg_{os.path.basename(DATA['data_dir'])}_{_param_tag}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"  Proseg: cache dir = {os.path.basename(out_dir)} ({_param_str})", flush=True)

        tx_src = os.path.join(DATA["data_dir"], "transcripts.parquet")

        # ── Check for sopa patches ────────────────────────────────────────
        patch_bounds = _get_patch_bounds_um()
        if patch_bounds and any(v is not None for v in [x_min, x_max, y_min, y_max]):
            filtered = []
            for px0, py0, px1, py1 in patch_bounds:
                if x_min is not None and px1 < x_min: continue
                if x_max is not None and px0 > x_max: continue
                if y_min is not None and py1 < y_min: continue
                if y_max is not None and py0 > y_max: continue
                filtered.append((
                    max(px0, x_min or px0), max(py0, y_min or py0),
                    min(px1, x_max or px1), min(py1, y_max or py1),
                ))
            patch_bounds = filtered or None

        if patch_bounds:
            n_patches = len(patch_bounds)
            print(f"  Proseg: using {n_patches} sopa patch(es)", flush=True)
            _set("running", "Loading transcripts for patch-based Proseg…")
            tx_all = pd.read_parquet(tx_src)
            if x_min is not None: tx_all = tx_all[tx_all["x_location"] >= x_min]
            if x_max is not None: tx_all = tx_all[tx_all["x_location"] <= x_max]
            if y_min is not None: tx_all = tx_all[tx_all["y_location"] >= y_min]
            if y_max is not None: tx_all = tx_all[tx_all["y_location"] <= y_max]

            all_cells: list   = []
            all_bounds: dict  = {}
            all_tx_meta: list = []
            CELL_ID_STRIDE    = 10_000_000  # max cells per patch
            cell_offset       = 1

            for pi, (px0, py0, px1, py1) in enumerate(patch_bounds):
                _set("running", f"Proseg patch {pi + 1}/{n_patches}…")
                tx_patch = tx_all[
                    (tx_all["x_location"] >= px0) & (tx_all["x_location"] <= px1) &
                    (tx_all["y_location"] >= py0) & (tx_all["y_location"] <= py1)
                ]
                if len(tx_patch) < 10:
                    print(f"  Proseg: patch {pi + 1} has too few transcripts — skipping", flush=True)
                    continue
                print(f"  Proseg: patch {pi + 1}/{n_patches}: {len(tx_patch):,} transcripts", flush=True)

                patch_dir = os.path.join(out_dir, f"patch_{pi:04d}")
                try:
                    cells_i, bounds_i, tx_meta_i = _proseg_run_single(
                        tx_patch, patch_dir, proseg_bin, voxel_size, n_threads, n_samples,
                        recorded_samples=recorded_samples, schedule=schedule,
                        nuclear_reassign_prob=nuclear_reassign_prob, prior_seg_prob=prior_seg_prob
                    )
                except Exception as pe:
                    print(f"  Proseg: patch {pi + 1} failed: {pe}", flush=True)
                    continue

                # Boundary resolution: keep cells with centroid in this patch
                in_patch = (
                    (cells_i["x_centroid"] >= px0) & (cells_i["x_centroid"] < px1) &
                    (cells_i["y_centroid"] >= py0) & (cells_i["y_centroid"] < py1)
                )
                cells_i  = cells_i[in_patch]
                valid_ids = set(cells_i.index)

                # Remap integer cell IDs with per-patch stride
                remap = {old: old + cell_offset for old in valid_ids}
                cells_i.index = pd.Index([remap[c] for c in cells_i.index], name="cell_id")
                all_cells.append(cells_i)
                for old_id, bnd in bounds_i.items():
                    if old_id in valid_ids:
                        all_bounds[remap[old_id]] = bnd
                if tx_meta_i is not None:
                    cell_col = "cell" if "cell" in tx_meta_i.columns else None
                    if cell_col:
                        tx_meta_i = tx_meta_i[tx_meta_i[cell_col].isin(valid_ids)].copy()
                        tx_meta_i[cell_col] = tx_meta_i[cell_col].map(remap)
                    all_tx_meta.append(tx_meta_i)
                cell_offset += CELL_ID_STRIDE

            if not all_cells:
                _set("error", "No cells found across all patches. Check Proseg output.")
                return

            cells_df       = pd.concat(all_cells)
            cell_bounds    = all_bounds
            tx_meta_merged = pd.concat(all_tx_meta) if all_tx_meta else None
            print(f"  Proseg: merged {len(cells_df):,} cells from {n_patches} patch(es)", flush=True)

        else:
            # ── Single run (no patches or no sopa data) ───────────────────
            if any(v is not None for v in [x_min, x_max, y_min, y_max]):
                _set("running", "Filtering transcripts to region…")
                tx = pd.read_parquet(tx_src)
                if x_min is not None: tx = tx[tx["x_location"] >= x_min]
                if x_max is not None: tx = tx[tx["x_location"] <= x_max]
                if y_min is not None: tx = tx[tx["y_location"] >= y_min]
                if y_max is not None: tx = tx[tx["y_location"] <= y_max]
                print(f"  Proseg: filtered to {len(tx):,} transcripts", flush=True)
            else:
                tx = pd.read_parquet(tx_src)

            _set("running", "Running Proseg…")
            try:
                cells_df, cell_bounds, tx_meta_merged = _proseg_run_single(
                    tx, out_dir, proseg_bin, voxel_size, n_threads, n_samples,
                    recorded_samples=recorded_samples, schedule=schedule,
                    nuclear_reassign_prob=nuclear_reassign_prob, prior_seg_prob=prior_seg_prob
                )
            except Exception as e:
                _set("error", str(e)[:300])
                return

        # ── Build expression matrix ───────────────────────────────────────
        _set("running", "Building expression matrix…")
        expr_mat = _build_proseg_expr(tx_meta_merged, cells_df)

        n_cells = len(cells_df)
        import json as _json
        _params_out = {
            "tool": "proseg", "voxel_size": voxel_size, "n_threads": n_threads,
            "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
            "n_cells": n_cells, "param_tag": _param_tag,
        }
        with open(os.path.join(out_dir, "params.json"), "w") as _f:
            _json.dump(_params_out, _f)
        _set("running", "Computing UMAP and clusters…")
        _compute_reseg_clusters_umap(cells_df, expr_mat,
                                     progress_fn=lambda msg: _set("running", msg))
        _set("running", "Writing SpatialData zarr…")
        zarr_path = _build_reseg_sdata(
            cells_df, cell_bounds, expr_mat, DATA["gene_names"], "proseg", out_dir
        )
        _set("done", f"Done — {n_cells:,} cells", result={
            "cells_df":    cells_df,
            "cell_bounds": cell_bounds,
            "out_dir":     out_dir,
            "expr":        expr_mat,
            "source":      "proseg",
            "sdata_path":  zarr_path,
        })
        print(f"  Proseg: {n_cells:,} cells segmented", flush=True)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set("error", str(exc)[:300])


# ─── Reseg clustering + UMAP ─────────────────────────────────────────────────

def _compute_reseg_clusters_umap(cells_df: pd.DataFrame, expr_mat,
                                  progress_fn=None) -> None:
    """Compute KMeans clustering and UMAP for reseg cells in-place on cells_df.
    Adds columns: cluster_kmeans_10, umap_1, umap_2.
    Also updates _umap_reseg_state so the UMAP scatter plot works immediately."""
    if expr_mat is None:
        return

    from sklearn.decomposition import TruncatedSVD

    n_cells = len(cells_df)
    if progress_fn:
        progress_fn(f"Normalising {n_cells:,} cells for UMAP/clustering…")

    mat = expr_mat[:, :len(DATA["gene_names"])].astype("float32").tocsr()
    row_sums = np.asarray(mat.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    mat = sp.diags(1e4 / row_sums).dot(mat).tocsr()
    mat.data = np.log1p(mat.data)

    n_comp = min(50, n_cells - 1, mat.shape[1] - 1)
    if progress_fn:
        progress_fn(f"PCA ({n_comp} components)…")
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    pca_coords = svd.fit_transform(mat)

    # KMeans clustering
    from sklearn.cluster import MiniBatchKMeans
    n_clusters = min(10, n_cells)
    if progress_fn:
        progress_fn(f"KMeans ({n_clusters} clusters)…")
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
    cells_df["cluster_kmeans_10"] = km.fit_predict(pca_coords) + 1  # 1-indexed

    # UMAP or t-SNE fallback
    if progress_fn:
        progress_fn("Computing UMAP…")
    try:
        import sys, types
        # Stub tensorflow so umap/__init__.py can load parametric_umap without
        # crashing on the NumPy 1.x vs 2.x ABI mismatch in the installed TF.
        if "tensorflow" not in sys.modules:
            sys.modules["tensorflow"] = types.ModuleType("tensorflow")
        import warnings
        from umap.umap_ import UMAP
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="n_jobs value.*overridden")
            reducer = UMAP(n_neighbors=min(15, n_cells - 1), min_dist=0.1, random_state=42)
            umap_coords = reducer.fit_transform(pca_coords)
    except Exception:
        from sklearn.manifold import TSNE
        perp = min(30, max(5, n_cells // 10))
        umap_coords = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(pca_coords)

    cells_df["umap_1"] = umap_coords[:, 0]
    cells_df["umap_2"] = umap_coords[:, 1]

    # Store result in _umap_reseg_state so the existing UMAP scatter works
    umap_df = pd.DataFrame(umap_coords, index=cells_df.index, columns=["umap_1", "umap_2"])
    with _umap_reseg_lock:
        _umap_reseg_state["status"]  = "done"
        _umap_reseg_state["message"] = f"Done — {n_cells:,} cells"
        _umap_reseg_state["result"]  = umap_df

    if progress_fn:
        progress_fn(f"UMAP + clustering done ({n_cells:,} cells)")
    print(f"  Reseg: UMAP + {n_clusters} KMeans clusters computed for {n_cells:,} cells", flush=True)


def _compute_split_clusters_umap(cells_df: pd.DataFrame, corrected_mat,
                                  corrected_cell_ids=None, progress_fn=None) -> None:
    """
    Compute KMeans clusters and UMAP on SPLIT-corrected counts.
    Writes cluster_split_10, split_umap_1, split_umap_2 columns into cells_df in-place.
    """
    try:
        import scipy.sparse as sp_
        from sklearn.preprocessing import normalize
        from sklearn.decomposition import TruncatedSVD
        from sklearn.cluster import KMeans
        try:
            from umap.umap_ import UMAP as _UMAP
            _have_umap = True
        except Exception:
            _have_umap = False

        n_cells, n_genes = corrected_mat.shape
        if progress_fn:
            progress_fn(f"SPLIT: normalizing {n_cells:,} cells × {n_genes} genes…")

        mat = corrected_mat if sp_.issparse(corrected_mat) else sp_.csr_matrix(corrected_mat)
        # Log-normalize
        log_mat = mat.copy().astype(np.float32)
        log_mat.data = np.log1p(log_mat.data)
        normed = normalize(log_mat, norm="l2", axis=1)

        # PCA via SVD
        n_comp = min(50, n_cells - 1, n_genes - 1)
        if progress_fn:
            progress_fn(f"SPLIT: SVD ({n_comp} components)…")
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        pca = svd.fit_transform(normed)

        # KMeans
        if progress_fn:
            progress_fn("SPLIT: KMeans clustering (k=10)…")
        km = KMeans(n_clusters=10, random_state=42, n_init=10)
        labels = km.fit_predict(pca)

        # UMAP
        embedding = None
        if _have_umap:
            if progress_fn:
                progress_fn("SPLIT: computing UMAP…")
            n_neighbors = min(30, n_cells - 1)
            reducer = _UMAP(n_components=2, n_neighbors=n_neighbors,
                                min_dist=0.3, random_state=42)
            embedding = reducer.fit_transform(pca)

        # Write back to cells_df — handle subset case (RCTD may drop low-UMI cells)
        if corrected_cell_ids is not None and len(corrected_cell_ids) < len(cells_df):
            try:
                sample_id = cells_df.index[0]
                cast = type(sample_id)
                int_idx = pd.Index([cast(c) for c in corrected_cell_ids])
            except Exception:
                int_idx = pd.Index(corrected_cell_ids)
            cells_df["cluster_split_10"] = pd.array([pd.NA] * len(cells_df), dtype="string")
            cells_df.loc[int_idx, "cluster_split_10"] = (labels + 1).astype(str)
            if embedding is not None:
                cells_df["split_umap_1"] = np.nan
                cells_df["split_umap_2"] = np.nan
                cells_df.loc[int_idx, "split_umap_1"] = embedding[:, 0].astype(np.float32)
                cells_df.loc[int_idx, "split_umap_2"] = embedding[:, 1].astype(np.float32)
        else:
            cells_df["cluster_split_10"] = (labels + 1).astype(str)
            if embedding is not None:
                cells_df["split_umap_1"] = embedding[:, 0].astype(np.float32)
                cells_df["split_umap_2"] = embedding[:, 1].astype(np.float32)
        if not _have_umap and progress_fn:
            progress_fn("SPLIT: umap-learn not available, skipping UMAP.")
    except Exception:
        import traceback
        traceback.print_exc()


def _split_write_zarr(sdata_path: str, corrected_mat, cells_df: pd.DataFrame,
                       gene_names: list, corrected_cell_ids=None) -> None:
    """Write SPLIT corrected counts + cluster/UMAP obs columns to a SpatialData zarr.
    Uses zarr low-level API to avoid spatialdata 'target path in use' error."""
    try:
        import zarr as _zarr_mod
        import scipy.sparse as sp_
        try:
            from anndata.io import write_elem as _write_elem
        except ImportError:
            from anndata.experimental import write_elem as _write_elem

        with _sdata_lock:
            zroot = _zarr_mod.open_group(sdata_path, mode="r+", use_consolidated=False)
            if "tables" not in zroot or "table" not in zroot["tables"]:
                print("  SPLIT: no table in zarr — skipping zarr write", flush=True)
                return
            table_grp = zroot["tables"]["table"]
            obs_grp   = table_grp["obs"]

            # Read zarr obs index for alignment
            idx_key  = obs_grp.attrs.get("_index", "_index")
            zarr_ids = [str(v) for v in obs_grp[idx_key][:]]

            # Align corrected_mat rows to zarr obs order
            # Use corrected_cell_ids (from R colnames) when available — RCTD may drop cells
            if corrected_cell_ids is not None:
                id_to_row = {str(cid): i for i, cid in enumerate(corrected_cell_ids)}
            else:
                id_to_row = {str(cid): i for i, cid in enumerate(cells_df.index)}
            # Strip patch prefix (e.g. "p0_") from zarr IDs — live patch runs add this
            # prefix but corrected_cell_ids come from R colnames which use bare Baysor IDs.
            import re as _re
            _stripped_zarr_ids = [_re.sub(r'^p\d+_', '', cid) for cid in zarr_ids]
            _lookup_ids = _stripped_zarr_ids if (
                len(set(_stripped_zarr_ids[:30]) & set(id_to_row)) >
                len(set(zarr_ids[:30]) & set(id_to_row))
            ) else zarr_ids
            row_indices = [id_to_row.get(cid, -1) for cid in _lookup_ids]
            mat         = corrected_mat if sp_.issparse(corrected_mat) \
                          else sp_.csr_matrix(corrected_mat)
            # Build aligned matrix; -1 means cell was dropped by RCTD → zero row
            valid_out = [i for i, r in enumerate(row_indices) if r >= 0]
            valid_in  = [r for r in row_indices if r >= 0]
            n_zarr    = len(zarr_ids)
            if valid_in:
                mat_sub = mat[valid_in, :]
                coo     = mat_sub.tocoo()
                new_row = np.array(valid_out)[coo.row]
                mat_aligned = sp_.csr_matrix(
                    (coo.data, (new_row, coo.col)),
                    shape=(n_zarr, mat.shape[1])
                )
            else:
                mat_aligned = sp_.csr_matrix((n_zarr, mat.shape[1]), dtype=mat.dtype)
            print(f"  SPLIT: aligned {len(valid_in)}/{n_zarr} cells for zarr write", flush=True)

            # Pad X_corrected to match zarr var width (panel genes only; zarr var may include
            # SpaGE-imputed genes appended later). Imputed columns get zeros.
            var_grp = table_grp["var"]
            try:
                var_key = var_grp.attrs.get("_index", "_index")
                n_var   = len(var_grp[var_key])
            except (KeyError, Exception):
                # Fallback: read n_var from X matrix shape attribute
                try:
                    _x_grp = table_grp["X"]
                    if isinstance(_x_grp, _zarr_mod.Array):
                        n_var = _x_grp.shape[1]
                    else:
                        n_var = _x_grp.attrs.get("shape", [0, mat_aligned.shape[1]])[1]
                except Exception:
                    n_var = mat_aligned.shape[1]
            if mat_aligned.shape[1] < n_var:
                n_pad       = n_var - mat_aligned.shape[1]
                mat_aligned = sp_.hstack(
                    [mat_aligned, sp_.csr_matrix((n_zarr, n_pad), dtype=mat_aligned.dtype)],
                    format="csr",
                )
                print(f"  SPLIT: padded X_corrected to {mat_aligned.shape} "
                      f"(+{n_pad} imputed gene cols = 0)", flush=True)

            # Write X_corrected layer — use_consolidated=False bypasses consolidated metadata.
            # Delete first so reruns with new parameters fully replace old results.
            layers_grp = table_grp.require_group("layers")
            if "X_corrected" in layers_grp:
                del layers_grp["X_corrected"]
            _write_elem(layers_grp, "X_corrected", mat_aligned)

            # Write obs columns — delete then rewrite so reruns fully replace old values
            for col in ["cluster_split_10", "split_umap_1", "split_umap_2"]:
                if col in obs_grp:
                    del obs_grp[col]
            for col in ["cluster_split_10", "split_umap_1", "split_umap_2"]:
                if col not in cells_df.columns:
                    continue
                ser = cells_df[col].reindex(zarr_ids)
                if cells_df[col].dtype == object or pd.api.types.is_string_dtype(cells_df[col]):
                    arr = np.array(ser.fillna("").values, dtype=str)
                    obs_grp.create_dataset(col, data=arr, shape=arr.shape,
                                           dtype=arr.dtype, overwrite=True)
                else:
                    arr = ser.values
                    obs_grp.create_dataset(col, data=arr, shape=arr.shape,
                                           dtype=arr.dtype, overwrite=True)

        print(f"  SPLIT: wrote corrected layer to {os.path.basename(sdata_path)}", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()


def _r_dgcmatrix_to_scipy(r_mat):
    """Convert an R dgCMatrix to a scipy CSC matrix (genes × cells).

    dgCMatrix is column-sparse (CSC): p = column pointers, i = row indices.
    Callers that need cells × genes should call .T.tocsr() on the result.
    """
    import scipy.sparse as sp_
    import numpy as np
    i = np.array(r_mat.slots["i"])
    p = np.array(r_mat.slots["p"])
    x = np.array(r_mat.slots["x"])
    dims = list(r_mat.slots["Dim"])
    return sp_.csc_matrix((x, i, p), shape=(dims[0], dims[1]))


def _run_split_correction(rds_path: str, label_col: str = "Names",
                           max_cores: int = 4, seg_source: str = "xenium",
                           min_umi: int = 10, min_umi_sigma: int = 100,
                           purify_singlets: bool = True) -> None:
    """
    Background thread: run RCTD (doublet mode) + SPLIT::purify() for ambient RNA correction.
    Requires R packages: spacexr, SPLIT (bdsc-tds/SPLIT).
    """
    print(f"  SPLIT: starting — min_umi={min_umi}, min_umi_sigma={min_umi_sigma}, "
          f"max_cores={max_cores}, purify_singlets={purify_singlets}, "
          f"seg_source={seg_source!r}", flush=True)
    _redirect_rpy2_console()

    def _set(status, message, result=None):
        with _split_lock:
            _split_state["status"]  = status
            _split_state["message"] = message
            if result is not None:
                _split_state["result"] = result

    try:
        import rpy2.robjects as ro
        import rpy2.robjects.conversion as _rconv
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        import scipy.sparse as sp_
        import scipy.io as sio
        import tempfile

        # rpy2 thread fix
        _rconv.set_conversion(ro.default_converter)
        pandas2ri.activate()

        # Clear stale "Rerun UMAP" result so make_umap_fig uses the fresh
        # split_umap_1/2 computed below, not a cached result from a previous
        # run with different parameters.
        with _umap_reseg_lock:
            _umap_reseg_state["status"] = "idle"
            _umap_reseg_state["result"] = None

        # ── 1. Get active data ──────────────────────────────────────────────
        _tool = _seg_tool(seg_source)
        with _baysor_lock:
            bres = _baysor_state["result"] if _baysor_state["status"] == "done" else None
        with _proseg_lock:
            pres = _proseg_state["result"] if _proseg_state["status"] == "done" else None
        if _tool == "baysor":
            _alt_res = bres
        elif _tool == "proseg":
            _alt_res = pres
        else:
            _alt_res = None

        if _alt_res is not None:
            expr       = _alt_res["expr"]
            cells_df   = _alt_res["cells_df"].copy()
            sdata_path = _alt_res.get("sdata_path", "")
            gni        = _alt_res.get("gene_name_to_idx") or DATA.get("gene_name_to_idx", {})
        else:
            expr       = DATA["expr"]
            cells_df   = DATA["df"].copy()
            sdata_path = DATA.get("sdata_path", "")
            gni        = DATA.get("gene_name_to_idx", {})

        panel_genes = DATA["gene_names"]  # always use panel gene list
        n_panel     = len(panel_genes)
        # Slice to panel-only columns (drop imputed)
        if expr.shape[1] > n_panel:
            expr_panel = expr[:, :n_panel]
        else:
            expr_panel = expr
        n_cells, n_genes = expr_panel.shape
        print(f"  SPLIT: {n_cells:,} cells × {n_genes} panel genes", flush=True)

        from rpy2.rinterface_lib import openrlib
        with openrlib.rlock:
            # ── 2. Check R packages ──────────────────────────────────────────────
            _set("running", "Loading R packages (spacexr, SPLIT)…")
            try:
                importr("spacexr")
            except Exception:
                _set("error", "spacexr not installed. In R: devtools::install_github('dmcable/spacexr')")
                return
            try:
                importr("SPLIT")
            except Exception:
                _set("error", "SPLIT not installed. In R: remotes::install_github('bdsc-tds/SPLIT')")
                return
            importr("SeuratObject")
            importr("Matrix")
            base = importr("base")

            # ── 3. Load Seurat reference ─────────────────────────────────────────
            _set("running", "Loading Seurat reference…")
            rds = base.readRDS(rds_path)
            meta_r  = ro.r['slot'](rds, "meta.data")
            meta_df = pandas2ri.rpy2py(meta_r)
            if label_col not in meta_df.columns:
                _set("error", f"Column '{label_col}' not found. Available: {list(meta_df.columns)[:10]}")
                return

            # ── 4. Shared genes ──────────────────────────────────────────────────
            _set("running", "Finding shared genes…")
            ro.r.assign("._split_rds", rds)
            mat_r     = ro.r("SeuratObject::LayerData(._split_rds[['RNA']], layer='counts')")
            ref_genes = list(ro.r['rownames'](mat_r))
            shared_genes = sorted(set(panel_genes) & set(ref_genes))
            if len(shared_genes) < 10:
                _set("error", f"Only {len(shared_genes)} shared genes between reference and panel.")
                return
            print(f"  SPLIT: {len(shared_genes)} shared genes", flush=True)

            # ── 5. Build Reference ───────────────────────────────────────────────
            _set("running", f"Building RCTD Reference ({len(meta_df):,} ref cells)…")
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
            _set("running", f"Building SpatialRNA object ({n_cells:,} cells)…")
            # Subset expr_panel to shared genes
            shared_idx = [gni[g] for g in shared_genes if g in gni and gni[g] < n_panel]
            shared_genes_present = [g for g in shared_genes if g in gni and gni[g] < n_panel]
            expr_shared = expr_panel[:, shared_idx]  # cells × shared_genes
            # gene × cells for R
            expr_t = expr_shared.T.toarray() if hasattr(expr_shared, 'toarray') else expr_shared.T
            counts_r_mat = ro.r.matrix(
                ro.FloatVector(expr_t.flatten(order='F').tolist()),
                nrow=len(shared_genes_present),
                ncol=n_cells
            )
            cell_ids = [str(c) for c in cells_df.index]
            ro.r.assign("._split_counts_mat", counts_r_mat)
            ro.r.assign("._split_rownames", ro.StrVector(shared_genes_present))
            ro.r.assign("._split_colnames", ro.StrVector(cell_ids))
            ro.r("""
rownames(._split_counts_mat) <- ._split_rownames
colnames(._split_counts_mat) <- ._split_colnames
._split_counts_mat <- as(._split_counts_mat, 'dgCMatrix')
""")

            ro.r.assign("._split_x", ro.FloatVector(cells_df["x_centroid"].astype(float).tolist()))
            ro.r.assign("._split_y", ro.FloatVector(cells_df["y_centroid"].astype(float).tolist()))
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
            # Compute a stable key for this RCTD run so we can skip it on re-runs.
            _rctd_lkey = "labels_rctd_doublet"
            if _alt_res is not None:
                _rctd_tag  = hashlib.md5((_alt_res.get("out_dir", "")).encode()).hexdigest()[:8]
                _rctd_tool = _alt_res.get("source", "baysor")
                _rctd_lkey = f"labels_rctd_doublet_{_rctd_tool}_{_rctd_tag}"
            _rctd_cache_key  = (f"rctd_{os.path.basename(rds_path)}_{label_col}_doublet"
                                f"_umi{min_umi}_sig{min_umi_sigma}_{_rctd_lkey}")
            _rctd_cache_file = _cache_path(_rctd_cache_key)
            _rctd_obj_file   = _rctd_cache_file.replace(".parquet", "_rctd_obj.rds")

            _rctd_from_cache = False
            if os.path.exists(_rctd_cache_file) and os.path.exists(_rctd_obj_file):
                try:
                    _cached_rctd = pd.read_parquet(_rctd_cache_file)
                    _split_labels_cached = _cached_rctd["label"].astype(str)
                    _split_labels_cached.index = _split_labels_cached.index.astype(str)
                    _sample = set(cell_ids[:200])
                    _overlap = len(set(_split_labels_cached.index[:200]) & _sample)
                    if _overlap >= max(1, len(_sample) * 0.05):
                        ro.r(f'._split_rctd <- readRDS("{_rctd_obj_file}")')
                        _split_labels = _split_labels_cached
                        _rctd_from_cache = True
                        _set("running", "RCTD loaded from cache, running SPLIT::purify()…")
                        print("  SPLIT: RCTD loaded from cache", flush=True)
                    else:
                        print(f"  SPLIT: RCTD cache index mismatch ({_overlap}/{len(_sample)}) — re-running", flush=True)
                except Exception as _ce:
                    print(f"  SPLIT: RCTD cache load failed: {_ce}", flush=True)

            if not _rctd_from_cache:
                _set("running", f"Running RCTD doublet mode (max_cores={max_cores})… [~20-40 min]")
                print("  SPLIT: starting RCTD doublet mode…", flush=True)
                ro.r.assign("._split_max_cores",    ro.IntVector([max_cores]))
                ro.r.assign("._split_min_umi",      ro.IntVector([min_umi]))
                ro.r.assign("._split_min_umi_sigma", ro.IntVector([min_umi_sigma]))
                ro.r("""
._split_rctd <- spacexr::create.RCTD(._split_spatialrna, ._split_reference,
    max_cores=._split_max_cores[1], CELL_MIN_INSTANCE=5,
    UMI_min=._split_min_umi[1], UMI_min_sigma=._split_min_umi_sigma[1])
._split_rctd <- spacexr::run.RCTD(._split_rctd, doublet_mode='doublet')
._split_rctd <- SPLIT::run_post_process_RCTD(._split_rctd)
""")
                # Save RCTD object + labels to cache for future re-runs.
                # Use as.character() in R to avoid rpy2 converting factor levels to
                # integer codes (which would show numbers instead of cell type names).
                try:
                    ro.r(f'saveRDS(._split_rctd, "{_rctd_obj_file}")')
                    ro.r("""
._split_rctd_res   <- ._split_rctd@results$results_df
._split_rctd_types <- as.character(._split_rctd_res$first_type)
names(._split_rctd_types) <- rownames(._split_rctd_res)
""")
                    _types_r = ro.r("._split_rctd_types")
                    _names_r = ro.r("names(._split_rctd_types)")
                    _split_labels = pd.Series(
                        list(_types_r), index=list(_names_r), name="label"
                    ).astype(str)
                    pd.DataFrame({"label": _split_labels}).to_parquet(_rctd_cache_file)
                    print(f"  SPLIT: RCTD cached ({len(_split_labels.unique())} types)", flush=True)
                except Exception as _se:
                    print(f"  SPLIT: RCTD cache save failed: {_se}", flush=True)
                    _split_labels = None

            # Store RCTD labels in _annot_state so cell-type coloring works
            if _split_labels is not None:
                _rctd_generic_key = "_".join(_rctd_lkey.split("_")[:2] + _rctd_lkey.split("_")[3:])
                with _annot_lock:
                    _annot_state[_rctd_lkey]       = _split_labels
                    _annot_state[_rctd_generic_key] = _split_labels

            print("  SPLIT: RCTD done, running purify()…", flush=True)

            # ── 8. Run SPLIT::purify ─────────────────────────────────────────────
            _set("running", "Running SPLIT::purify()…")
            # Full counts matrix (all panel genes) gene × cells
            expr_full_t = expr_panel.T.toarray() if hasattr(expr_panel, 'toarray') else expr_panel.T
            full_r_mat = ro.r.matrix(
                ro.FloatVector(expr_full_t.flatten(order='F').tolist()),
                nrow=n_panel,
                ncol=n_cells
            )
            ro.r.assign("._split_full_counts", full_r_mat)
            ro.r.assign("._split_panel_genes", ro.StrVector(list(panel_genes)))
            ro.r("""
rownames(._split_full_counts) <- ._split_panel_genes
colnames(._split_full_counts) <- ._split_colnames
""")
            ro.r("""._split_full_counts <- as(._split_full_counts, 'dgCMatrix')""")
            ro.r.assign("._split_do_purify_singlets", ro.BoolVector([purify_singlets]))
            ro.r("""
._split_res <- SPLIT::purify(
    counts          = ._split_full_counts,
    rctd            = ._split_rctd,
    DO_purify_singlets = ._split_do_purify_singlets[1]
)
._split_purified <- ._split_res$purified_counts
""")

            # ── 9. Convert back to Python ────────────────────────────────────────
            _set("running", "Converting corrected counts to Python…")
            purified_r = ro.r["._split_purified"]
            # Extract cell IDs from R colnames — RCTD may drop low-UMI cells
            try:
                corrected_cell_ids = list(ro.r['colnames'](purified_r))
            except Exception:
                corrected_cell_ids = None
            # Convert dgCMatrix → scipy CSR (cells × genes)
            purified_csr_t = _r_dgcmatrix_to_scipy(purified_r)  # gene × cells
        corrected_mat  = purified_csr_t.T.tocsr()             # cells × genes
        corrected_mat.data = np.clip(corrected_mat.data, 0, None)
        n_corr = corrected_mat.shape[0]
        n_orig = len(cells_df)
        print(f"  SPLIT: corrected matrix shape {corrected_mat.shape} "
              f"({n_corr}/{n_orig} cells)", flush=True)
        if corrected_cell_ids is None:
            corrected_cell_ids = [str(c) for c in cells_df.index[:n_corr]]

        # ── 10. Compute clusters + UMAP ──────────────────────────────────────
        _set("running", "Computing clusters and UMAP on corrected counts…")
        _compute_split_clusters_umap(cells_df, corrected_mat,
                                      corrected_cell_ids=corrected_cell_ids,
                                      progress_fn=lambda msg: _set("running", msg))

        # ── 11. Write to zarr ────────────────────────────────────────────────
        if sdata_path and os.path.isdir(sdata_path):
            _set("running", "Writing corrected counts to SpatialData zarr…")
            _split_write_zarr(sdata_path, corrected_mat, cells_df, list(panel_genes),
                              corrected_cell_ids=corrected_cell_ids)

        # Expand corrected_mat to full cells_df size before storing in memory.
        # After RCTD, n_corr ≤ n_orig; _get_expr_values returns a column of length
        # n_corr which misaligns with the n_orig spatial plot → bottom cells show NA.
        # The zarr write above does its own alignment, so expand only here.
        if n_corr < len(cells_df):
            _id_to_row = {str(cid): i for i, cid in enumerate(corrected_cell_ids)}
            _valid_out, _valid_in = [], []
            for _oi, _cell_id in enumerate(cells_df.index):
                _r = _id_to_row.get(str(_cell_id), -1)
                if _r >= 0:
                    _valid_out.append(_oi)
                    _valid_in.append(_r)
            _n_full = len(cells_df)
            if _valid_in:
                _sub = corrected_mat[_valid_in, :]
                _coo = _sub.tocoo()
                corrected_mat = sp_.csr_matrix(
                    (_coo.data, (np.array(_valid_out)[_coo.row], _coo.col)),
                    shape=(_n_full, corrected_mat.shape[1])
                )
            else:
                corrected_mat = sp_.csr_matrix(
                    (_n_full, corrected_mat.shape[1]), dtype=corrected_mat.dtype)
            print(f"  SPLIT: expanded in-memory corrected matrix to {corrected_mat.shape} "
                  f"({len(_valid_in)}/{n_corr} cells mapped)", flush=True)

        # ── 12. Update in-memory ─────────────────────────────────────────────
        if _alt_res is not None:
            _lock = _baysor_lock if _tool == "baysor" else _proseg_lock
            with _lock:
                _alt_res["split_corrected_expr"] = corrected_mat
                _alt_res["split_corrected_cell_ids"] = corrected_cell_ids
                _alt_res["split_panel_genes"] = list(panel_genes)
                _alt_res["split_corrected_imputed_genes"] = []
                # Merge cluster/UMAP cols back into alt_res cells_df
                for col in ["cluster_split_10", "split_umap_1", "split_umap_2"]:
                    if col in cells_df.columns:
                        _alt_res["cells_df"][col] = cells_df[col].values
        else:
            DATA["split_corrected_expr"] = corrected_mat
            DATA["split_corrected_cell_ids"] = corrected_cell_ids
            DATA["split_panel_genes"] = list(panel_genes)
            DATA["split_corrected_imputed_genes"] = []
            for col in ["cluster_split_10", "split_umap_1", "split_umap_2"]:
                if col in cells_df.columns:
                    DATA["df"][col] = cells_df[col].values

        _set("done", f"SPLIT complete — {n_cells:,} cells corrected", result={"n_cells": n_cells})
        print(f"  SPLIT: done — {n_cells:,} cells corrected", flush=True)

    except Exception:
        import traceback
        msg = traceback.format_exc()
        print(f"  SPLIT ERROR:\n{msg}", flush=True)
        with _split_lock:
            _split_state["status"]  = "error"
            _split_state["message"] = msg.strip().splitlines()[-1]


# ─── Reseg UMAP computation (manual re-run) ───────────────────────────────────

def _run_reseg_umap() -> None:
    """Background thread: compute UMAP for the active resegmented cells."""
    def _set(status, message, result=None):
        with _umap_reseg_lock:
            _umap_reseg_state["status"]  = status
            _umap_reseg_state["message"] = message
            if result is not None:
                _umap_reseg_state["result"] = result

    try:
        # Get the active reseg result (Proseg > Baysor)
        with _proseg_lock:
            pres = _proseg_state["result"] if _proseg_state["status"] == "done" else None
        with _baysor_lock:
            bres = _baysor_state["result"] if not pres and _baysor_state["status"] == "done" else None
        alt_res = pres or bres

        if alt_res is None:
            # Xenium mode: use SPLIT-corrected expr if counts_mode == "corrected",
            # otherwise use raw expression matrix.
            with _umap_reseg_lock:
                _counts_mode = _umap_reseg_state.get("counts_mode", "original")
            _corr_expr = DATA.get("split_corrected_expr") if _counts_mode == "corrected" else None
            _corr_ids  = DATA.get("split_corrected_cell_ids")  # IDs of cells SPLIT kept
            _xen_df    = DATA["df"]

            # If corrected expr was loaded from zarr at startup it is full-size
            # (all cells, zeros for non-SPLIT cells) but corr_ids is not populated.
            # Derive corr_ids from cluster_split_10 and subset the matrix.
            # If corrected expr is full-size (loaded from zarr at startup), corr_ids
            # won't be set. Identify SPLIT-kept cells by non-zero rows (zeroed for
            # non-corrected cells during zarr write) and subset the matrix.
            if _corr_expr is not None and _corr_ids is None and \
                    _corr_expr.shape[0] == len(_xen_df):
                _row_sums = np.asarray(_corr_expr.sum(axis=1)).flatten()
                _kept_idx = np.where(_row_sums > 0)[0]
                if len(_kept_idx) > 0:
                    _corr_ids = [str(i) for i in _xen_df.index[_kept_idx]]
                    _corr_expr = _corr_expr[_kept_idx, :]
                    DATA["split_corrected_cell_ids"] = _corr_ids
                    print(f"  UMAP: derived {len(_corr_ids):,} SPLIT cell IDs from non-zero rows",
                          flush=True)

            # Zarr fallback: if corrected expr not in memory (e.g. after app restart),
            # try loading X_corrected from the SpatialData zarr using low-level zarr API
            # (obs columns written via create_dataset are not exposed by spatialdata .obs).
            if (_counts_mode == "corrected") and (_corr_expr is None) and DATA.get("sdata_path") and \
                    os.path.isdir(DATA["sdata_path"]):
                try:
                    import zarr as _zarr_fb
                    _zfb    = _zarr_fb.open_group(DATA["sdata_path"], mode="r",
                                                  use_consolidated=False)
                    _tbl_fb = _zfb["tables"]["table"]
                    _obs_fb = _tbl_fb["obs"]
                    _lay_fb = _tbl_fb.get("layers", {})
                    if "X_corrected" in _lay_fb:
                        # Read sparse X_corrected
                        _xg = _lay_fb["X_corrected"]
                        _xcorr_full = sp.csr_matrix(
                            (_xg["data"][:], _xg["indices"][:], _xg["indptr"][:]),
                            shape=tuple(_xg.attrs["shape"]),
                        )
                        # Read cell IDs
                        _idx_key = _obs_fb.attrs.get("_index", "_index")
                        _all_ids = [str(v) for v in _obs_fb[_idx_key][:]]
                        # Identify SPLIT-kept cells via non-empty cluster_split_10
                        if "cluster_split_10" in _obs_fb:
                            _clust   = np.array([str(v) for v in _obs_fb["cluster_split_10"][:]])
                            _kept_mask = np.array([bool(v.strip()) for v in _clust], dtype=bool)
                        else:
                            _kept_mask = np.ones(len(_all_ids), dtype=bool)
                        _kept_rows = np.where(_kept_mask)[0]
                        _corr_expr = _xcorr_full[_kept_rows, :]
                        _corr_ids  = [_all_ids[i] for i in _kept_rows]
                        # Cache in DATA so subsequent calls don't re-read zarr
                        DATA["split_corrected_expr"]     = _corr_expr
                        DATA["split_corrected_cell_ids"] = _corr_ids
                        print(f"  Xenium UMAP: loaded X_corrected from zarr "
                              f"({len(_corr_ids):,} SPLIT-corrected cells)", flush=True)
                except Exception as _fb_err:
                    print(f"  Xenium UMAP: zarr X_corrected fallback failed: {_fb_err}",
                          flush=True)

            if _corr_expr is not None and _corr_ids is not None:
                # Build index mapping: corrected_cell_ids → rows in _corr_expr
                _id_to_row = {str(cid): i for i, cid in enumerate(_corr_ids)}
                _df_ids    = [str(i) for i in _xen_df.index]
                _sub_rows  = [_id_to_row[i] for i in _df_ids if i in _id_to_row]
                _sub_mask  = [i in _id_to_row for i in _df_ids]  # bool mask over DATA["df"]
                expr = _corr_expr[_sub_rows, :]
                _n   = len(_sub_rows)
                _set("running", f"Normalising {_n:,} SPLIT-corrected cells…")
                print(f"  Xenium UMAP: using SPLIT-corrected counts for {_n:,}/{len(_xen_df):,} cells",
                      flush=True)
            else:
                expr = DATA.get("expr")
                if expr is None:
                    _set("error", "No expression matrix loaded")
                    return
                _sub_mask = None
                _n        = len(_xen_df)
                _set("running", f"Normalising {_n:,} Xenium cells…")
                print(f"  Xenium UMAP: using original counts for {_n:,} cells", flush=True)

            _mat = expr.astype("float32").tocsr()
            _rs  = np.asarray(_mat.sum(axis=1)).flatten()
            _rs[_rs == 0] = 1.0
            _mat = sp.diags(1e4 / _rs).dot(_mat).tocsr()
            _mat.data = np.log1p(_mat.data)
            import warnings as _w2
            from sklearn.decomposition import TruncatedSVD as _TSVD2
            _nc = min(50, _n - 1, _mat.shape[1] - 1)
            _set("running", f"PCA ({_nc} components)…")
            _svd2 = _TSVD2(n_components=_nc, random_state=0)
            with _w2.catch_warnings():
                _w2.filterwarnings("ignore", category=RuntimeWarning,
                                   message="invalid value encountered in divide")
                _pca2 = _svd2.fit_transform(_mat)
            _set("running", "Running UMAP…")
            try:
                import sys as _sys2, types as _types2
                if "tensorflow" not in _sys2.modules:
                    _sys2.modules["tensorflow"] = _types2.ModuleType("tensorflow")
                from umap.umap_ import UMAP as _UMAP2
                with _w2.catch_warnings():
                    _w2.filterwarnings("ignore", message="n_jobs value.*overridden")
                    _emb2 = _UMAP2(n_neighbors=min(15, _n - 1), min_dist=0.1,
                                   random_state=0).fit_transform(_pca2)
            except Exception:
                from sklearn.manifold import TSNE as _TSNE2
                _set("running", "Running t-SNE (umap-learn unavailable)…")
                _emb2 = _TSNE2(n_components=2, perplexity=min(30, max(5, _n // 10)),
                               random_state=0).fit_transform(_pca2)

            if _sub_mask is not None:
                # Write NaN for cells SPLIT dropped, UMAP coords for cells it kept
                _u1 = np.full(len(_xen_df), np.nan)
                _u2 = np.full(len(_xen_df), np.nan)
                _sub_idx = np.where(_sub_mask)[0]
                _u1[_sub_idx] = _emb2[:, 0]
                _u2[_sub_idx] = _emb2[:, 1]
                DATA["df"]["umap_1"] = _u1
                DATA["df"]["umap_2"] = _u2
            else:
                DATA["df"]["umap_1"] = _emb2[:, 0]
                DATA["df"]["umap_2"] = _emb2[:, 1]
            _umap_df_cache.clear()
            with _umap_reseg_lock:
                _umap_reseg_state["_xenium_bumped"] = False
            _label = "SPLIT-corrected" if _sub_mask is not None else "Xenium"
            _set("done", f"{_label} UMAP — {_n:,} cells")
            print(f"  {_label} UMAP recomputed for {_n:,} cells", flush=True)
            return

        if alt_res.get("expr") is None:
            # Try loading expr from the reseg SpatialData zarr
            zarr_path = alt_res.get("sdata_path", "")
            if zarr_path and os.path.isdir(zarr_path):
                try:
                    import spatialdata as _sd_umap
                    _sdata_umap = _sd_umap.read_zarr(zarr_path)
                    alt_res["expr"] = sp.csr_matrix(_sdata_umap.tables["table"].X)
                    print(f"  Reseg UMAP: loaded expr from {os.path.basename(zarr_path)}", flush=True)
                except Exception as _ze:
                    _set("error", f"No expression matrix — re-run segmentation ({_ze})")
                    return
            else:
                _set("error", "No expression matrix — re-run segmentation first")
                return

        with _umap_reseg_lock:
            _counts_mode = _umap_reseg_state.get("counts_mode", "original")
        _corr = alt_res.get("split_corrected_expr") if _counts_mode == "corrected" else None
        _src_label = "SPLIT-corrected" if _corr is not None else "reseg"
        _all_cell_ids = [str(c) for c in alt_res["cells_df"].index]
        # Always subset to the SPLIT-corrected cells only (zarr-aligned matrix pads uncorrected
        # cells with zeros which would distort the UMAP embedding).
        if _corr is not None:
            _corr_ids = [str(c) for c in alt_res.get("split_corrected_cell_ids", [])]
            if not _corr_ids and _corr.shape[0] == len(_all_cell_ids):
                # Loaded from zarr (no corrected_cell_ids in memory) — derive from non-zero rows
                _nonzero_mask = np.diff(_corr.indptr) > 0
                _corr_ids = [_all_cell_ids[i] for i in np.where(_nonzero_mask)[0]]
                print(f"  Reseg UMAP: derived {len(_corr_ids):,} corrected cell IDs "
                      f"from non-zero X_corrected rows", flush=True)
            if _corr_ids:
                _id_to_row = {c: i for i, c in enumerate(_corr_ids)}
                if _corr.shape[0] == len(_all_cell_ids):
                    # zarr-aligned (full-size) matrix — row order matches _all_cell_ids
                    _sub_idx = [i for i, c in enumerate(_all_cell_ids) if c in _id_to_row]
                    _sub_cell_ids = [_all_cell_ids[i] for i in _sub_idx]
                    expr = _corr[_sub_idx, :]
                else:
                    # Raw SPLIT output — rows are in corrected_cell_ids order
                    expr = _corr
                    _sub_cell_ids = _corr_ids
                cell_ids = _sub_cell_ids
                print(f"  Reseg UMAP: SPLIT-corrected subset — "
                      f"{len(cell_ids):,}/{len(_all_cell_ids):,} cells", flush=True)
            else:
                # No corrected cells found at all — fall back to uncorrected
                print(f"  Reseg UMAP: no corrected cells found — using uncorrected", flush=True)
                expr = alt_res["expr"]
                _src_label = "reseg"
                cell_ids = _all_cell_ids
        else:
            expr = alt_res["expr"]
            cell_ids = _all_cell_ids
        n_cells = expr.shape[0]
        _set("running", f"Normalising {n_cells:,} {_src_label} cells…")
        print(f"  Reseg UMAP: using {_src_label} counts for {n_cells:,} cells", flush=True)

        # CP10K + log1p normalise
        mat = expr.astype("float32").tocsr()
        row_sums = np.asarray(mat.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        mat = sp.diags(1e4 / row_sums).dot(mat).tocsr()
        mat.data = np.log1p(mat.data)

        # PCA (TruncatedSVD) — reduce to 50 dims before UMAP
        import warnings as _warnings
        from sklearn.decomposition import TruncatedSVD
        n_comp = min(50, n_cells - 1, mat.shape[1] - 1)
        _set("running", f"PCA ({n_comp} components)…")
        svd = TruncatedSVD(n_components=n_comp, random_state=0)
        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", category=RuntimeWarning,
                                     message="invalid value encountered in divide")
            pca_coords = svd.fit_transform(mat)   # (n_cells × n_comp)

        # UMAP — prefer umap-learn, fall back to t-SNE
        _set("running", "Running UMAP…")
        try:
            import sys, types
            # Stub tensorflow so umap/__init__.py can load parametric_umap without
            # crashing on the NumPy 1.x vs 2.x ABI mismatch in the installed TF.
            if "tensorflow" not in sys.modules:
                sys.modules["tensorflow"] = types.ModuleType("tensorflow")
            import warnings
            from umap.umap_ import UMAP
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="n_jobs value.*overridden")
                reducer = UMAP(n_neighbors=min(15, n_cells - 1),
                               min_dist=0.1, random_state=0)
                embedding = reducer.fit_transform(pca_coords)
        except Exception:
            from sklearn.manifold import TSNE
            _set("running", "Running t-SNE (umap-learn not installed or TF error)…")
            perp = min(30, max(5, n_cells // 10))
            embedding = TSNE(n_components=2, perplexity=perp,
                             random_state=0).fit_transform(pca_coords)

        umap_df = pd.DataFrame(
            {"umap_1": embedding[:, 0], "umap_2": embedding[:, 1]},
            index=cell_ids,
        )
        _set("done", f"Done — {n_cells:,} cells", result=umap_df)
        print(f"  Reseg UMAP: computed for {n_cells:,} cells", flush=True)

        # ── Write UMAP coords back to cells_df + zarr for caching ────
        _set("running", "Caching UMAP to zarr…")
        _lock = _proseg_lock if pres else _baysor_lock
        _sdata_path = alt_res.get("sdata_path", "")
        # Build full-length arrays (NaN for cells not in this UMAP's index)
        _u1 = np.full(len(_all_cell_ids), np.nan)
        _u2 = np.full(len(_all_cell_ids), np.nan)
        _id_to_pos = {c: i for i, c in enumerate(_all_cell_ids)}
        for _cid, _e1, _e2 in zip(cell_ids, embedding[:, 0], embedding[:, 1]):
            _pos = _id_to_pos.get(_cid)
            if _pos is not None:
                _u1[_pos] = _e1
                _u2[_pos] = _e2
        with _lock:
            alt_res["cells_df"]["umap_1"] = _u1
            alt_res["cells_df"]["umap_2"] = _u2
        if _sdata_path and os.path.isdir(_sdata_path):
            _update_reseg_zarr_obs(_sdata_path, alt_res["cells_df"])
            print(f"  Reseg UMAP: cached to zarr", flush=True)
        _set("done", f"Done — {n_cells:,} cells", result=umap_df)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set("error", str(exc)[:200])


# ─── SpaGE imputation ─────────────────────────────────────────────────────────
SPAGE_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SpaGE_repo")


def _spage_cache_dir() -> str:
    d = os.path.join(os.path.expanduser("~"), ".xenium_explorer_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _spage_index_path() -> str:
    return os.path.join(_spage_cache_dir(), "spage_index.json")


def _spage_cache_path(rds_path: str, genes_key: str) -> str:
    tag = hashlib.md5((rds_path + DATA["data_dir"] + genes_key).encode()).hexdigest()[:12]
    return os.path.join(_spage_cache_dir(), f"spage_{tag}.parquet")


def _spage_index_update(cache_file: str) -> None:
    """Record the latest spage cache file for this dataset."""
    import json
    idx_path = _spage_index_path()
    try:
        with open(idx_path) as f:
            idx = json.load(f)
    except Exception:
        idx = {}
    idx[DATA["data_dir"]] = cache_file
    with open(idx_path, "w") as f:
        json.dump(idx, f, indent=2)


def _annot_autoload() -> None:
    """On startup, load all cached Xenium annotations for this dataset (all methods)."""
    import glob
    tag       = hashlib.md5(DATA["data_dir"].encode()).hexdigest()[:8]
    cache_dir = os.path.join(os.path.expanduser("~"), ".xenium_explorer_cache")
    pattern   = os.path.join(cache_dir, f"*_{tag}.parquet")
    label_files  = [f for f in glob.glob(pattern)
                    if not os.path.basename(f).startswith(("labels_", "spage_", "spatialdata_"))
                    and not os.path.basename(f).endswith("_weights.parquet")]
    weight_files = [f for f in glob.glob(pattern)
                    if os.path.basename(f).endswith("_weights.parquet")]
    if not label_files and not weight_files:
        return
    loaded = 0
    for fpath in label_files:
        name = os.path.basename(fpath)
        # Extract labels_key embedded in filename: strip "_{tag}.parquet" then find "_labels_"
        name_stem = name[:-(len(tag) + 1 + len(".parquet"))]
        lk_pos = name_stem.rfind("_labels_")
        if lk_pos >= 0:
            labels_key = name_stem[lk_pos + 1:]  # e.g. "labels_seurat" or "labels_seurat_baysor_abc12345"
        elif name.startswith("rctd_"):
            labels_key = "labels_rctd"
        elif name.startswith("seurat_"):
            labels_key = "labels_seurat"
        else:
            labels_key = "labels_celltypist"
        # Derive method name from labels_key for display purposes
        method = labels_key.split("_")[1] if "_" in labels_key else labels_key
        try:
            cached       = pd.read_parquet(fpath)
            labels       = cached["label"].astype(str)
            labels.index = labels.index.astype(str)
            # For RCTD mode-specific keys (e.g. "labels_rctd_doublet"), also set the
            # generic key (e.g. "labels_rctd") so make_spatial_fig keeps working after restart.
            parts = labels_key.split("_")
            if len(parts) >= 3 and parts[2] in ("full", "doublet", "multi"):
                generic_key = "_".join(parts[:2] + parts[3:])
            else:
                generic_key = None
            with _annot_lock:
                _annot_state[labels_key] = labels
                if generic_key:
                    _annot_state[generic_key] = labels
                _annot_state["status"]   = "done"
                _annot_state["message"]  = f"Auto-loaded ({_ANNOT_METHODS[method]})"
            print(f"  Annotation: auto-loaded {method} ({len(labels.unique())} types) from {name}", flush=True)
            loaded += 1
        except Exception as exc:
            print(f"  Annotation: auto-load failed for {name}: {exc}", flush=True)
    for fpath in weight_files:
        name = os.path.basename(fpath)
        try:
            weights_df = pd.read_parquet(fpath)
            weights_df.index = weights_df.index.astype(str)
            # Parse labels_key from weights filename (strip "_weights.parquet" then "_{tag}")
            w_name_stem = name[:-(len("_weights.parquet"))]
            w_name_stem2 = w_name_stem[:-(len(tag) + 1)]
            w_lk_pos = w_name_stem2.rfind("_labels_")
            if w_lk_pos >= 0:
                wlabels_key = w_name_stem2[w_lk_pos + 1:]
            else:
                wlabels_key = "labels_rctd"
            weights_key = f"rctd_weights_{wlabels_key}"
            w_parts = wlabels_key.split("_")
            if len(w_parts) >= 3 and w_parts[2] in ("full", "doublet", "multi"):
                generic_wkey = f"rctd_weights_{'_'.join(w_parts[:2] + w_parts[3:])}"
            else:
                generic_wkey = None
            with _annot_lock:
                _annot_state[weights_key] = weights_df
                if generic_wkey:
                    _annot_state[generic_wkey] = weights_df
            print(f"  Annotation: auto-loaded RCTD weights ({weights_df.shape[1]} types) from {name}", flush=True)
        except Exception as exc:
            print(f"  Annotation: weight auto-load failed for {name}: {exc}", flush=True)


def _spage_autoload() -> None:
    """On startup, restore the most recent SpaGE result for this dataset (if cached)."""
    import json
    idx_path = _spage_index_path()
    if not os.path.exists(idx_path):
        return
    try:
        with open(idx_path) as f:
            idx = json.load(f)
        cache_file = idx.get(DATA["data_dir"])
        if cache_file and (os.path.exists(cache_file) or os.path.isdir(cache_file)):
            zarr_path = (cache_file if os.path.isdir(cache_file)
                         else cache_file.replace(".parquet", ".zarr")
                         if os.path.isdir(cache_file.replace(".parquet", ".zarr")) else None)
            if zarr_path:
                import zarr as _zarr_al
                _zal = _zarr_al.open_array(zarr_path, mode='r')
                _genes_al = list(_zal.attrs.get('genes', []))
                if _genes_al:
                    with _spage_lock:
                        _spage_state["status"]       = "done"
                        _spage_state["message"]      = f"Auto-loaded (streaming) — {len(_genes_al):,} genes"
                        _spage_state["result_path"]  = zarr_path
                        _spage_state["result_genes"] = _genes_al
                    print(f"  SpaGE: streaming {len(_genes_al):,} genes from {zarr_path}", flush=True)
                    return
            # Normal parquet — check size before loading entire file
            try:
                import pyarrow.parquet as _pq_al
                _pfal = _pq_al.ParquetFile(cache_file)
                _nr_al = _pfal.metadata.num_rows
                _genes_al = _pfal.schema_arrow.names
                _nc_al = len(_genes_al)
                if _nr_al * _nc_al > SPAGE_STREAM_THRESHOLD:
                    with _spage_lock:
                        _spage_state["status"]       = "done"
                        _spage_state["message"]      = f"Auto-loaded (streaming) — {_nc_al:,} genes"
                        _spage_state["result_path"]  = cache_file
                        _spage_state["result_genes"] = list(_genes_al)
                    print(f"  SpaGE: large cache {_nr_al:,}×{_nc_al:,} → streaming read", flush=True)
                    return
            except Exception:
                pass
            print(f"  SpaGE: auto-loading cached results from {cache_file}", flush=True)
            imp_df = pd.read_parquet(cache_file)
            with _spage_lock:
                _spage_state["status"]  = "done"
                _spage_state["message"] = f"Auto-loaded (cached) — {len(imp_df.columns):,} genes"
                _spage_state["result"]  = imp_df
    except Exception as exc:
        print(f"  SpaGE: auto-load failed: {exc}", flush=True)


def _vectorized_spage(spatial_df: pd.DataFrame, rna_df: pd.DataFrame,
                      n_pv: int, genes_to_predict: list,
                      out_zarr: str = None) -> "pd.DataFrame | None":
    """
    Vectorized SpaGE: same algorithm as SpaGE.main.SpaGE but the kNN
    weighted-average step is batched with numpy/einsum instead of a
    Python for-loop — ~200× faster for 200k spatial cells.
    """
    import sys
    if SPAGE_REPO not in sys.path:
        sys.path.insert(0, SPAGE_REPO)
    from SpaGE.principal_vectors import PVComputation
    import scipy.stats as st
    from sklearn.neighbors import NearestNeighbors

    shared = np.intersect1d(spatial_df.columns, rna_df.columns)
    if len(shared) == 0:
        raise ValueError("No shared genes between spatial and reference data.")

    n_pv = min(n_pv, len(shared))

    # Z-score (SpaGE normalizes internally on shared genes) — use float32 to halve memory
    rna_scaled   = pd.DataFrame(
        st.zscore(rna_df[shared],     axis=0).astype(np.float32),
        index=rna_df.index,     columns=shared)
    spat_scaled  = pd.DataFrame(
        st.zscore(spatial_df[shared], axis=0).astype(np.float32),
        index=spatial_df.index, columns=shared)

    # Principal Vectors
    pv = PVComputation(n_factors=n_pv, n_pv=n_pv, dim_reduction="pca", dim_reduction_target="pca")
    pv.fit(rna_scaled, spat_scaled)
    S = pv.source_components_.T
    eff_pv = int(np.sum(np.diag(pv.cosine_similarity_matrix_) > 0.3))
    eff_pv = max(eff_pv, 1)
    S = S[:, :eff_pv]
    print(f"  SpaGE: {eff_pv} effective principal vectors", flush=True)

    rna_proj  = (rna_scaled.values  @ S).astype(np.float32)   # (n_rna,  eff_pv)
    spat_proj = (spat_scaled.values @ S).astype(np.float32)   # (n_spat, eff_pv)
    del rna_scaled, spat_scaled   # free ~(n_rna + n_spat) × n_shared × 4B

    # kNN in PV space
    nbrs = NearestNeighbors(n_neighbors=50, algorithm="auto", metric="cosine")
    nbrs.fit(rna_proj)
    distances, indices = nbrs.kneighbors(spat_proj)   # (n_spat, 50)
    del rna_proj, spat_proj

    # Compute weights once (shared across all genes)
    valid  = distances < 1                                            # (n_spat, 50)
    d_v    = np.where(valid, distances, 0.0).astype(np.float32)
    del distances
    d_sum  = d_v.sum(axis=1, keepdims=True)
    d_sum[d_sum == 0] = 1.0
    w      = np.where(valid, 1.0 - d_v / d_sum, 0.0).astype(np.float32)  # (n_spat, 50)
    del d_v, d_sum
    n_v    = np.maximum(valid.sum(axis=1, keepdims=True) - 1, 1).astype(np.float32)
    del valid
    w      = w / n_v                                                  # (n_spat, 50)
    del n_v

    # Impute in gene chunks to limit peak memory.
    # Y_nbrs per chunk: 222k × 50 × CHUNK × 4B — GENE_CHUNK=10 → ~440 MB vs 2.2 GB at 50.
    GENE_CHUNK = 10
    n_genes = len(genes_to_predict)
    n_spat  = len(spatial_df)
    n_chunks = (n_genes + GENE_CHUNK - 1) // GENE_CHUNK
    if n_chunks > 1:
        print(f"  SpaGE: imputing {n_genes} genes in {n_chunks} chunks of {GENE_CHUNK}", flush=True)

    if out_zarr is not None:
        # ── Streaming path: write each chunk directly to zarr (never accumulate in RAM) ──
        import zarr as _zarr
        _z = _zarr.open_array(
            out_zarr, mode='w', shape=(n_spat, n_genes), dtype=np.float32,
            chunks=(min(65536, n_spat), min(GENE_CHUNK, n_genes)),
        )
        for i in range(0, n_genes, GENE_CHUNK):
            chunk_genes = genes_to_predict[i:i + GENE_CHUNK]
            Y      = rna_df[chunk_genes].values.astype(np.float32)
            Y_nbrs = Y[indices]
            _z[:, i:i + len(chunk_genes)] = np.einsum("ij,ijk->ik", w, Y_nbrs)
            del Y, Y_nbrs
        _z.attrs['genes'] = genes_to_predict   # written last — marks completion
        del w, indices
        return None   # caller owns the zarr; use result_path/result_genes for lookup

    # ── In-memory path (small result) ──────────────────────────────────────────
    chunks = []
    for i in range(0, n_genes, GENE_CHUNK):
        chunk_genes = genes_to_predict[i:i + GENE_CHUNK]
        Y      = rna_df[chunk_genes].values.astype(np.float32)  # (n_rna, chunk)
        Y_nbrs = Y[indices]                                      # (n_spat, 50, chunk)
        chunks.append(np.einsum("ij,ijk->ik", w, Y_nbrs))       # (n_spat, chunk)
        del Y, Y_nbrs
    imp = np.concatenate(chunks, axis=1)                         # (n_spat, n_genes)
    del chunks, w, indices

    return pd.DataFrame(imp, index=spatial_df.index, columns=genes_to_predict)


def _run_spage_imputation(rds_path: str, n_pv: int, genes_input: str,
                          seg_source: str = None, use_corrected: bool = False) -> None:
    """Background thread: run SpaGE gene imputation and store results.
    When seg_source is a Baysor/Proseg run, imputes against reseg cells and writes
    back to the reseg SpatialData zarr (alt_res['sdata_path']).
    When use_corrected=True, uses SPLIT-corrected counts as the spatial input."""
    _redirect_rpy2_console()

    def _set(status, message, result=None):
        with _spage_lock:
            _spage_state["status"]  = status
            _spage_state["message"] = message
            if result is not None:
                _spage_state["result"] = result

    try:
        import tempfile
        import scipy.io as sio
        import rpy2.robjects as ro
        import rpy2.robjects.conversion as _rconv
        from rpy2.robjects.packages import importr

        # Ensure rpy2 conversion rules are active in this thread's context.
        # Dash callbacks run in threads where the rpy2 ContextVar may be unset.
        _rconv.set_conversion(ro.default_converter)

        # ── Determine active reseg (if any) ──────────────────────────────
        _alt_res = None
        if seg_source and _seg_tool(seg_source) != "xenium":
            if _seg_tool(seg_source) == "baysor":
                with _baysor_lock:
                    if _baysor_state["status"] == "done":
                        _alt_res = _baysor_state["result"]
            elif _seg_tool(seg_source) == "proseg":
                with _proseg_lock:
                    if _proseg_state["status"] == "done":
                        _alt_res = _proseg_state["result"]

        # ── Load reference gene list ─────────────────────────────────────
        from rpy2.rinterface_lib import openrlib
        _set("running", "Opening Seurat reference (this may take a few minutes)…")
        with openrlib.rlock:
            importr("SeuratObject")
            importr("Matrix")
            ro.r(f'seurat_ref <- readRDS("{rds_path}")')
            ro.r('rna_ref    <- seurat_ref[["RNA"]]')
            ro.r('rna_mat_data <- SeuratObject::LayerData(rna_ref, layer="data")')
            ro.r('rna_genes  <- rownames(rna_mat_data)')
            rna_genes_all = list(ro.r("rna_genes"))

        # Use only panel (non-imputed) genes for shared-gene computation
        gene_var = DATA.get("gene_var")
        if gene_var is not None and "is_imputed" in gene_var.columns:
            panel_gene_names = [g for g in DATA["gene_names"] if not gene_var.loc[g, "is_imputed"]]
        else:
            panel_gene_names = list(DATA["gene_names"])
        xenium_genes = set(panel_gene_names)
        shared_genes = sorted(xenium_genes & set(rna_genes_all))
        if not shared_genes:
            _set("error", "No shared genes between Xenium panel and reference!")
            return
        print(f"  SpaGE: {len(shared_genes)} shared genes with reference", flush=True)

        # ── Determine genes_to_predict ───────────────────────────────────
        auto_hvg = not (genes_input and genes_input.strip())
        if auto_hvg:
            # Will pick top-200 HVG after loading reference matrix
            predict_candidates = [g for g in rna_genes_all if g not in xenium_genes]
            genes_needed = shared_genes + predict_candidates[:2000]
            genes_key = "auto200"
        else:
            requested = [g.strip() for line in genes_input.replace(",", "\n").split("\n")
                         for g in [line.strip()] if g]
            genes_to_predict = [g for g in requested if g in rna_genes_all and g not in xenium_genes]
            if not genes_to_predict:
                _set("error", "None of the requested genes found in reference (or all in Xenium panel).")
                return
            genes_needed = shared_genes + genes_to_predict
            genes_key = ",".join(sorted(genes_to_predict))

        # Reseg runs get a separate cache key so their imputed values don't clash with Xenium
        _cache_seg_suffix = f"__{seg_source.replace(':', '_')}" if (seg_source and seg_source != "xenium") else ""
        cache_file = _spage_cache_path(rds_path, genes_key + _cache_seg_suffix)
        zarr_cache = cache_file.replace(".parquet", ".zarr")
        if os.path.isdir(zarr_cache):
            try:
                import zarr as _zarr_c
                _zc = _zarr_c.open_array(zarr_cache, mode='r')
                _genes_c = list(_zc.attrs.get('genes', []))
                if _genes_c:
                    with _spage_lock:
                        _spage_state["result_path"]  = zarr_cache
                        _spage_state["result_genes"] = _genes_c
                    if _alt_res is not None:
                        try:
                            import json as _json_sc
                            _sj = {"path": zarr_cache, "genes": _genes_c}
                            with open(os.path.join(_alt_res["out_dir"], "spage_result.json"), "w") as _sjf:
                                _json_sc.dump(_sj, _sjf)
                        except Exception:
                            pass
                        _alock = _baysor_lock if _seg_tool(seg_source or "") == "baysor" else _proseg_lock
                        with _alock:
                            _alt_res["spage_result_path"]  = zarr_cache
                            _alt_res["spage_result_genes"] = _genes_c
                    else:
                        _spage_index_update(zarr_cache)
                    _set("done", f"Done (cached) — {len(_genes_c):,} genes (streaming)", result=None)
                    return
            except Exception as _zce:
                print(f"  SpaGE: zarr cache load failed: {_zce} — re-running", flush=True)
        if os.path.exists(cache_file):
            _set("running", "Loading cached SpaGE results…")
            try:
                import pyarrow.parquet as _pq_c
                _pf_c = _pq_c.ParquetFile(cache_file)
                _nc_c = len(_pf_c.schema_arrow.names)
                _nr_c = _pf_c.metadata.num_rows
                if _nr_c * _nc_c > SPAGE_STREAM_THRESHOLD:
                    _genes_c = _pf_c.schema_arrow.names
                    with _spage_lock:
                        _spage_state["result_path"]  = cache_file
                        _spage_state["result_genes"] = list(_genes_c)
                    _spage_index_update(cache_file)
                    _set("done", f"Done (cached, streaming) — {_nc_c:,} genes", result=None)
                    return
            except Exception:
                pass
            imp_df = pd.read_parquet(cache_file)
            _spage_index_update(cache_file)
            _set("done", f"Done (cached) — {len(imp_df.columns):,} genes imputed", result=imp_df)
            return

        # ── Extract submatrix from R and save to temp files ──────────────
        n_predict_preview = len(genes_needed) - len(shared_genes)
        _set("running", f"Extracting {len(shared_genes)} shared + {n_predict_preview} predict genes…")

        with tempfile.TemporaryDirectory() as tmpdir:
            mat_file   = os.path.join(tmpdir, "mat.mtx").replace("\\", "/")
            genes_file = os.path.join(tmpdir, "genes.txt").replace("\\", "/")
            cells_file = os.path.join(tmpdir, "cells.txt").replace("\\", "/")

            # Build R character vector of needed genes
            genes_r = "c(" + ",".join(f'"{g}"' for g in genes_needed) + ")"
            with openrlib.rlock:
                ro.r(f"""
genes_needed <- {genes_r}
genes_needed <- genes_needed[genes_needed %in% rna_genes]
mat_sub <- rna_mat_data[genes_needed, ]
Matrix::writeMM(mat_sub, "{mat_file}")
writeLines(rownames(mat_sub), "{genes_file}")
writeLines(colnames(mat_sub), "{cells_file}")
""")
            mat_sp = sio.mmread(mat_file).T.tocsr()   # cells × genes
            with open(genes_file) as f:
                rna_genes_sub = [l.strip() for l in f]
            with open(cells_file) as f:
                rna_cells = [l.strip() for l in f]

        # Free R memory
        with openrlib.rlock:
            ro.r("rm(seurat_ref, rna_ref, rna_mat_data, mat_sub); gc()")

        # ── Convert to dense DataFrame ───────────────────────────────────
        _set("running", f"Building expression matrices ({len(rna_cells):,} ref cells)…")
        rna_df = pd.DataFrame(
            mat_sp.toarray().astype(np.float32),
            index=rna_cells, columns=rna_genes_sub,
        )
        del mat_sp  # sparse copy no longer needed

        # HVG selection for auto mode
        if auto_hvg:
            predict_pool = [g for g in rna_genes_sub if g not in xenium_genes]
            var_scores = rna_df[predict_pool].var(axis=0)
            genes_to_predict = var_scores.nlargest(200).index.tolist()
            print(f"  SpaGE: auto-selected {len(genes_to_predict)} HVGs", flush=True)
        else:
            genes_to_predict = [g for g in genes_to_predict if g in rna_genes_sub]

        if not genes_to_predict:
            _set("error", "No valid genes to predict after filtering against reference.")
            return

        # ── Build normalized expression matrix for SpaGE input ───────────
        shared_in_rna = [g for g in shared_genes if g in rna_genes_sub]

        if _alt_res is not None:
            # ── Reseg path: use alt_res["expr"] (panel genes × reseg cells) ──
            if use_corrected:
                reseg_expr = _alt_res.get("split_corrected_expr")
                if reseg_expr is None:
                    _set("error", "No SPLIT-corrected counts for this segmentation — run SPLIT first.")
                    return
                print("  SpaGE: using SPLIT-corrected counts (reseg)", flush=True)
            else:
                reseg_expr = _alt_res.get("expr")
            if reseg_expr is None:
                _set("error", "Reseg has no expression matrix — reload segmentation.")
                return
            # Panel gene index (columns 0..n_panel-1 of reseg expr)
            panel_gni = {g: i for i, g in enumerate(panel_gene_names)}
            shared_in_rna_use = [g for g in shared_in_rna if g in panel_gni]
            xen_idx_use = [panel_gni[g] for g in shared_in_rna_use]
            if not shared_in_rna_use:
                _set("error", "No shared genes found in reseg expression matrix.")
                return
            xen_raw = reseg_expr[:, xen_idx_use].toarray().astype(np.float32)
            rs = xen_raw.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
            xen_norm = np.log1p(xen_raw / rs * 10_000)
            del xen_raw, rs
            spatial_df = pd.DataFrame(xen_norm, index=_alt_res["cells_df"].index,
                                      columns=shared_in_rna_use)
            del xen_norm
        else:
            # ── Xenium path ───────────────────────────────────────────────
            if use_corrected:
                _xenium_expr = DATA.get("split_corrected_expr")
                if _xenium_expr is None:
                    _set("error", "No SPLIT-corrected counts available — run SPLIT first.")
                    return
                print("  SpaGE: using SPLIT-corrected counts (Xenium)", flush=True)
            else:
                _xenium_expr = DATA.get("expr_csc")# or DATA["expr"]
            panel_gni = DATA["gene_name_to_idx"]
            shared_in_rna_use = [g for g in shared_in_rna if g in panel_gni]
            xen_idx = [panel_gni[g] for g in shared_in_rna_use]
            xen_raw = _xenium_expr[:, xen_idx].toarray().astype(np.float32)
            rs = xen_raw.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
            xen_norm = np.log1p(xen_raw / rs * 10_000)
            del xen_raw, rs
            df  = DATA["df"]
            idx = DATA["df_to_expr"]
            good = idx >= 0
            xen_aligned = np.zeros((len(df), len(shared_in_rna_use)), dtype=np.float32)
            xen_aligned[good] = xen_norm[idx[good]]
            del xen_norm
            spatial_df = pd.DataFrame(xen_aligned, index=df.index, columns=shared_in_rna_use)
            del xen_aligned

        # ── Run SpaGE — streaming or in-memory based on output size ─────
        _n_out = len(spatial_df) * len(genes_to_predict)
        _streaming = _n_out > SPAGE_STREAM_THRESHOLD
        if _streaming:
            print(f"  SpaGE: large output ({len(spatial_df):,}×{len(genes_to_predict):,})"
                  f" → streaming to {os.path.basename(zarr_cache)}", flush=True)
            _set("running", f"Running SpaGE (n_pv={n_pv}, {len(genes_to_predict):,} genes, streaming)…")
            _vectorized_spage(spatial_df, rna_df, n_pv, genes_to_predict, out_zarr=zarr_cache)
            del spatial_df, rna_df
            with _spage_lock:
                _spage_state["result_path"]  = zarr_cache
                _spage_state["result_genes"] = genes_to_predict
            if _alt_res is None:
                _spage_index_update(zarr_cache)
            else:
                try:
                    import json as _json_sc
                    _sj = {"path": zarr_cache, "genes": genes_to_predict}
                    with open(os.path.join(_alt_res["out_dir"], "spage_result.json"), "w") as _sjf:
                        _json_sc.dump(_sj, _sjf)
                except Exception as _sje:
                    print(f"  SpaGE: could not persist result ref: {_sje}", flush=True)
                _alock = _baysor_lock if _seg_tool(seg_source or "") == "baysor" else _proseg_lock
                with _alock:
                    _alt_res["spage_result_path"]  = zarr_cache
                    _alt_res["spage_result_genes"] = genes_to_predict
            _set("done", f"Done — {len(genes_to_predict):,} genes imputed (streaming)")
            return

        _set("running", f"Running SpaGE (n_pv={n_pv}, {len(genes_to_predict)} genes)…")
        imp_df = _vectorized_spage(spatial_df, rna_df, n_pv, genes_to_predict)
        del spatial_df, rna_df

        # ── Cache to parquet ──────────────────────────────────────────────
        _set("running", "Saving imputed results…")
        imp_df.to_parquet(cache_file)
        if _alt_res is None:
            _spage_index_update(cache_file)
        else:
            try:
                import json as _json_sc
                _sj = {"path": cache_file, "genes": genes_to_predict}
                with open(os.path.join(_alt_res["out_dir"], "spage_result.json"), "w") as _sjf:
                    _json_sc.dump(_sj, _sjf)
            except Exception as _sje:
                print(f"  SpaGE: could not persist result ref: {_sje}", flush=True)
            _alock = _baysor_lock if _seg_tool(seg_source or "") == "baysor" else _proseg_lock
            with _alock:
                _alt_res["spage_result_path"]  = cache_file
                _alt_res["spage_result_genes"] = genes_to_predict
        print(f"  SpaGE: {len(imp_df.columns)} genes cached to {cache_file}", flush=True)

        def _spage_write_zarr(sdata_path, imp_df_sub, lock_ctx=None):
            """Append imputed gene columns to a SpatialData zarr table. Returns list of new genes written."""
            try:
                import spatialdata as _sd_w, anndata as _ad_w
                _sdata_w = _sd_w.read_zarr(sdata_path)
                _adata_w = _sdata_w.tables["table"]
                _existing = set(_adata_w.var_names)
                _new_genes = [g for g in imp_df_sub.columns if g not in _existing]
                if not _new_genes:
                    return []
                _new_X = sp.csr_matrix(imp_df_sub[_new_genes].values.astype(np.float32))
                _new_var = pd.DataFrame(
                    {"is_imputed": True},
                    index=pd.Index(_new_genes, name=_adata_w.var.index.name or "gene"),
                )
                _adata_new = _ad_w.AnnData(
                    X=sp.hstack([_adata_w.X, _new_X]).tocsr(),
                    obs=_adata_w.obs.copy(),
                    var=pd.concat([_adata_w.var, _new_var]),
                    uns=dict(_adata_w.uns),
                )
                _adata_new.uns["imputed_genes"] = list(_adata_w.uns.get("imputed_genes", [])) + _new_genes
                _sdata_w.tables["table"] = _adata_new
                try:
                    _sdata_w.write_element("table")
                except Exception:
                    _sdata_w.write(sdata_path, overwrite=True)
                return _new_genes, _adata_new
            except Exception as _ze:
                print(f"  SpaGE: zarr write-back failed (non-fatal): {_ze}", flush=True)
                return [], None

        if _alt_res is not None:
            # ── Reseg write-back: update alt_res["expr"] and per-reseg gene index ──
            sdata_path = _alt_res.get("sdata_path", "")
            if sdata_path and os.path.isdir(sdata_path):
                _set("running", "Writing imputed genes to reseg SpatialData zarr…")
                result_pair = _spage_write_zarr(sdata_path, imp_df)
                new_genes = result_pair[0] if result_pair else []
                if new_genes:
                    print(f"  SpaGE reseg: wrote {len(new_genes)} genes to {os.path.basename(sdata_path)}", flush=True)
                    with _spage_lock:
                        # Extend alt_res["expr"] with new imputed columns
                        new_cols = sp.csr_matrix(imp_df[new_genes].values.astype(np.float32))
                        _alt_res["expr"] = sp.hstack([_alt_res["expr"], new_cols]).tocsr()
                        # Rebuild per-reseg gene index (preserves order)
                        prev_gni = _alt_res.get("gene_name_to_idx") or DATA["gene_name_to_idx"]
                        prev_names = sorted(prev_gni, key=prev_gni.__getitem__)
                        extended_names = prev_names + new_genes
                        _alt_res["gene_name_to_idx"] = {g: i for i, g in enumerate(extended_names)}
                        _alt_res["imputed_genes"] = list(_alt_res.get("imputed_genes", [])) + new_genes
        else:
            # ── Xenium write-back: update DATA["expr"] and gene_names ─────────────
            sdata_path = DATA.get("sdata_path", "")
            if sdata_path and os.path.isdir(sdata_path):
                _set("running", "Writing imputed genes to SpatialData zarr…")
                result_pair = _spage_write_zarr(sdata_path, imp_df)
                new_genes = result_pair[0] if result_pair else []
                adata_new  = result_pair[1] if len(result_pair) > 1 else None
                if new_genes and adata_new is not None:
                    print(f"  SpaGE: wrote {len(new_genes)} imputed genes to zarr", flush=True)
                    with _spage_lock:
                        DATA["gene_names"]       = list(adata_new.var_names)
                        DATA["gene_name_to_idx"] = {g: i for i, g in enumerate(DATA["gene_names"])}
                        DATA["expr"]             = sp.csr_matrix(adata_new.X)
                        DATA["expr_csc"]         = DATA["expr"].tocsc()
                        DATA["gene_var"]         = adata_new.var
                        _gene_expr_cache.clear()

        _set("done", f"Done — {len(imp_df.columns):,} genes imputed", result=imp_df)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set("error", str(exc)[:300])


# ─── Subset / unsubset ────────────────────────────────────────────────────────

def subset(cluster=None, cell_type=None,
           gene=None, min_expr=None, max_expr=None,
           min_transcripts=None, max_transcripts=None,
           min_cell_area=None,    max_cell_area=None,
           min_nucleus_area=None, max_nucleus_area=None,
           cell_ids=None, method=None):
    """
    Subset the displayed cells. All non-None filters are ANDed together.

    Parameters
    ----------
    cluster : int | str | list
        Cluster label(s) to keep (integer IDs from the selected clustering method).
    method : str
        Clustering method name (default: first available).
        Use list(DATA['cluster_methods']) to see options.
    cell_type : str | list
        Cell type label(s) to keep (requires annotation to have been run).
    gene : str
        Gene name for expression-based filtering (use with min_expr / max_expr).
    min_expr, max_expr : float
        log1p expression thresholds for the selected gene.
    min_transcripts, max_transcripts : float
        Transcript count bounds.
    min_cell_area, max_cell_area : float
        Cell area bounds (µm²).
    min_nucleus_area, max_nucleus_area : float
        Nucleus area bounds (µm²).
    cell_ids : list
        Explicit list of cell IDs (barcodes) to keep.

    Examples
    --------
    subset(cluster=3)
    subset(cluster=[1, 2, 3], method='gene_expression_10_clusters')
    subset(cell_type='Cardiomyocyte')
    subset(gene='MYH7', min_expr=1.0)
    subset(min_transcripts=50, max_cell_area=500)
    """
    global _subset_version

    # Save originals on first subset call
    if "_df_original" not in DATA:
        DATA["_df_original"]      = DATA["df"]
        DATA["_df_to_expr_orig"]  = DATA["df_to_expr"]

    df   = DATA["_df_original"]
    mask = pd.Series(True, index=df.index)

    # ── Cluster filter ────────────────────────────────────────────────────
    if cluster is not None:
        _method = method or DATA["cluster_methods"][0]
        col  = f"clust__{_method}"
        vals = [cluster] if not isinstance(cluster, (list, tuple)) else list(cluster)
        vals = [int(v) for v in vals]
        mask &= df[col].isin(vals)

    # ── Cell type filter ──────────────────────────────────────────────────
    if cell_type is not None:
        with _annot_lock:
            labels = next((v for m in _ANNOT_METHODS
                           for k, v in _annot_state.items()
                           if k == f"labels_{m}" and v is not None), None)
        if labels is None:
            print("  subset: no cell type annotation loaded — skipping cell_type filter", flush=True)
        else:
            vals = [cell_type] if isinstance(cell_type, str) else list(cell_type)
            mask &= labels.reindex(df.index.astype(str)).isin(vals)

    # ── Gene expression filter ────────────────────────────────────────────
    if gene is not None and (min_expr is not None or max_expr is not None):
        expr_vals = pd.Series(get_gene_expression(gene), index=df.index)
        if min_expr is not None: mask &= expr_vals >= min_expr
        if max_expr is not None: mask &= expr_vals <= max_expr

    # ── Numeric range filters ─────────────────────────────────────────────
    if min_transcripts  is not None: mask &= df["transcript_counts"] >= min_transcripts
    if max_transcripts  is not None: mask &= df["transcript_counts"] <= max_transcripts
    if min_cell_area    is not None: mask &= df["cell_area"]          >= min_cell_area
    if max_cell_area    is not None: mask &= df["cell_area"]          <= max_cell_area
    if min_nucleus_area is not None: mask &= df["nucleus_area"]       >= min_nucleus_area
    if max_nucleus_area is not None: mask &= df["nucleus_area"]       <= max_nucleus_area

    # ── Explicit cell IDs ─────────────────────────────────────────────────
    if cell_ids is not None:
        mask &= df.index.isin(cell_ids)

    df_sub = df[mask]

    # Update active data
    DATA["df"] = df_sub

    # Rebuild df_to_expr for the subset (map subset row → expr matrix row)
    orig_to_expr = DATA["_df_to_expr_orig"]
    orig_pos     = {cid: i for i, cid in enumerate(df.index)}
    DATA["df_to_expr"] = np.array(
        [orig_to_expr[orig_pos[cid]] for cid in df_sub.index], dtype=np.int64
    )

    # ── Reseg path (Baysor / Proseg) ──────────────────────────────────────
    with _proseg_lock:
        _pres = _proseg_state["result"] if _proseg_state["status"] == "done" else None
    with _baysor_lock:
        _bres = _baysor_state["result"] if not _pres and _baysor_state["status"] == "done" else None
    _alt_res = _pres or _bres

    if _alt_res is not None:
        _tool = "proseg" if _pres else "baysor"

        # Save originals on first subset
        if "_cells_df_original" not in _alt_res:
            _alt_res["_cells_df_original"] = _alt_res["cells_df"]
            _alt_res["_expr_original"]      = _alt_res["expr"]
            _alt_res["_corrected_original"] = _alt_res.get("split_corrected_expr")

        bdf   = _alt_res["_cells_df_original"]
        bmask = pd.Series(True, index=bdf.index)

        # Cluster filter — reseg cluster column names start with "cluster"
        if cluster is not None:
            _bcols = [c for c in bdf.columns if c.startswith("cluster")]
            if _bcols:
                _bvals_str = [str(cluster)] if not isinstance(cluster, (list, tuple)) else [str(v) for v in cluster]
                bmask &= bdf[_bcols[0]].astype(str).isin(_bvals_str)
            else:
                print(f"  subset reseg: no cluster column in cells_df — skipping cluster filter", flush=True)

        # Cell type filter
        if cell_type is not None:
            _lk = f"labels_{_tool}"
            with _annot_lock:
                _blabels = _annot_state.get(_lk)
            if _blabels is None:
                print(f"  subset reseg: no cell type annotation for {_tool} — skipping cell_type filter", flush=True)
            else:
                _bct_vals = [cell_type] if isinstance(cell_type, str) else list(cell_type)
                bmask &= _blabels.reindex(bdf.index.astype(str)).isin(_bct_vals)

        # Gene expression filter — reseg uses its own expr matrix
        if gene is not None and (min_expr is not None or max_expr is not None):
            if gene in DATA["gene_names"]:
                _gidx    = DATA["gene_names"].index(gene)
                _braw    = _alt_res["_expr_original"].getcol(_gidx).toarray().ravel()
                _bgexpr  = pd.Series(np.log1p(_braw), index=bdf.index)
                if min_expr is not None: bmask &= _bgexpr >= min_expr
                if max_expr is not None: bmask &= _bgexpr <= max_expr
            else:
                print(f"  subset reseg: gene '{gene}' not in reseg expr — skipping gene filter", flush=True)

        # QC filters
        if min_transcripts  is not None and "transcript_counts" in bdf.columns: bmask &= bdf["transcript_counts"] >= min_transcripts
        if max_transcripts  is not None and "transcript_counts" in bdf.columns: bmask &= bdf["transcript_counts"] <= max_transcripts
        if min_cell_area    is not None and "cell_area"         in bdf.columns: bmask &= bdf["cell_area"]         >= min_cell_area
        if max_cell_area    is not None and "cell_area"         in bdf.columns: bmask &= bdf["cell_area"]         <= max_cell_area
        if min_nucleus_area is not None and "nucleus_area"      in bdf.columns: bmask &= bdf["nucleus_area"]      >= min_nucleus_area
        if max_nucleus_area is not None and "nucleus_area"      in bdf.columns: bmask &= bdf["nucleus_area"]      <= max_nucleus_area

        # Explicit cell IDs
        if cell_ids is not None:
            bmask &= bdf.index.isin(cell_ids)

        _kept_pos = np.where(bmask.values)[0]
        _bdf_sub  = bdf.iloc[_kept_pos]
        _alt_res["cells_df"] = _bdf_sub
        _alt_res["expr"]     = _alt_res["_expr_original"][_kept_pos, :]
        if _alt_res["_corrected_original"] is not None:
            _alt_res["split_corrected_expr"] = _alt_res["_corrected_original"][_kept_pos, :]
        print(f"  subset reseg ({_tool}): {len(_bdf_sub):,} / {len(bdf):,} cells selected", flush=True)

    _umap_df_cache.clear()
    _gene_expr_cache.clear()
    _viewport_arrays.clear()
    _subset_version += 1
    print(f"  subset: {len(df_sub):,} / {len(df):,} cells selected", flush=True)
    return df_sub


def unsubset():
    """Restore the full dataset after subsetting."""
    global _subset_version

    if "_df_original" not in DATA:
        print("  unsubset: no subset is active", flush=True)
        return

    DATA["df"]         = DATA.pop("_df_original")
    DATA["df_to_expr"] = DATA.pop("_df_to_expr_orig")

    # Restore reseg originals if subset was applied there too
    with _proseg_lock:
        _pres = _proseg_state["result"] if _proseg_state["status"] == "done" else None
    with _baysor_lock:
        _bres = _baysor_state["result"] if not _pres and _baysor_state["status"] == "done" else None
    _alt_res = _pres or _bres
    if _alt_res is not None and "_cells_df_original" in _alt_res:
        _alt_res["cells_df"] = _alt_res.pop("_cells_df_original")
        _alt_res["expr"]     = _alt_res.pop("_expr_original")
        _orig_corr = _alt_res.pop("_corrected_original")
        if _orig_corr is not None:
            _alt_res["split_corrected_expr"] = _orig_corr

    _umap_df_cache.clear()
    _gene_expr_cache.clear()
    _viewport_arrays.clear()
    _subset_version += 1
    print(f"  unsubset: restored {len(DATA['df']):,} cells", flush=True)


def retry_split_zarr_write(seg_source: str = None) -> None:
    """Write in-memory SPLIT corrected counts to the SpatialData zarr.

    Use this if SPLIT completed successfully in the current session but the zarr
    write failed (e.g. due to consolidated metadata error). Call from the REPL:

        retry_split_zarr_write()               # Xenium
        retry_split_zarr_write("baysor:xxxx")  # Baysor run
        retry_split_zarr_write("proseg:xxxx")  # Proseg run
    """
    tool = _seg_tool(seg_source) if seg_source else "xenium"

    if tool == "baysor":
        with _baysor_lock:
            res = _baysor_state.get("result")
    elif tool == "proseg":
        with _proseg_lock:
            res = _proseg_state.get("result")
    else:
        res = None  # Xenium uses DATA directly

    if res is not None:
        corrected_mat     = res.get("split_corrected_expr")
        corrected_ids     = res.get("split_corrected_cell_ids")
        panel_genes       = res.get("split_panel_genes")
        sdata_path        = res.get("sdata_path", "")
        cells_df          = res.get("cells_df")
    else:
        corrected_mat     = DATA.get("split_corrected_expr")
        corrected_ids     = DATA.get("split_corrected_cell_ids")
        panel_genes       = DATA.get("split_panel_genes")
        sdata_path        = DATA.get("sdata_path", "")
        cells_df          = DATA.get("df")

    if corrected_mat is None:
        print("  retry_split_zarr_write: no split_corrected_expr in memory — "
              "has SPLIT been run this session?", flush=True)
        return
    if not sdata_path or not os.path.isdir(sdata_path):
        print(f"  retry_split_zarr_write: sdata_path not found: {sdata_path!r}", flush=True)
        return
    if not panel_genes:
        print("  retry_split_zarr_write: split_panel_genes not stored — "
              "please rerun SPLIT (old session data)", flush=True)
        return

    print(f"  retry_split_zarr_write: writing {corrected_mat.shape} matrix to {sdata_path}",
          flush=True)
    _split_write_zarr(sdata_path, corrected_mat, cells_df, panel_genes,
                      corrected_cell_ids=corrected_ids)


def get_genes(file: str = None) -> list:
    """
    Print and return all non-imputed Xenium panel genes.

    Parameters
    ----------
    file : str, optional
        If provided, save the gene list to this file path (one gene per line).

    Returns
    -------
    list of gene name strings.
    """
    genes = sorted(DATA["gene_names"])
    print(f"  {len(genes)} panel genes:", flush=True)
    for g in genes:
        print(f"    {g}", flush=True)
    if file:
        with open(file, "w") as fh:
            fh.write("\n".join(genes) + "\n")
        print(f"  Saved to {file}", flush=True)
    return genes


def run_spage(rds_path: str, genes_file: str = None, n_pv: int = 50,
              use_corrected: bool = None) -> None:
    """
    Run SpaGE gene imputation from the REPL.

    Parameters
    ----------
    rds_path : str
        Path to the Seurat RDS reference file.
    genes_file : str, optional
        Path to a .txt file with gene names to impute (one per line, or
        comma-separated). If None, the top-200 highly variable genes from
        the reference are auto-selected.
    n_pv : int
        Number of principal vectors for cross-dataset alignment (default 50).
    use_corrected : bool, optional
        If True, use SPLIT-corrected counts as input. Defaults to the current
        Active Counts mode (True when "SPLIT Corrected" is selected in the UI).
        Set explicitly to override.

    Example
    -------
    run_spage(
        rds_path="/Users/ikuz/Documents/XeniumWorkflow/snRV_ref.rds",
        genes_file="/Users/ikuz/Documents/XeniumWorkflow/to_impute.txt",
        n_pv=50,
    )
    """
    global _spage_repl_pending
    with _spage_lock:
        if _spage_state["status"] == "running":
            print("  SpaGE is already running.", flush=True)
            return

    # Default use_corrected to the current Active Counts mode
    if use_corrected is None:
        use_corrected = (_active_counts_mode == "corrected")
    if use_corrected:
        print("  SpaGE: will use SPLIT-corrected counts as input", flush=True)

    # Build genes_input string (same format _run_spage_imputation expects)
    genes_input = ""
    if genes_file:
        try:
            with open(genes_file) as fh:
                genes_input = fh.read()
            n_genes = len([g for line in genes_input.replace(",", "\n").splitlines()
                           for g in [line.strip()] if g])
            print(f"  SpaGE: loaded {n_genes} genes from {genes_file}", flush=True)
        except FileNotFoundError:
            print(f"  SpaGE: file not found: {genes_file}", flush=True)
            return

    print(f"  SpaGE: starting imputation (n_pv={n_pv})…", flush=True)
    with _spage_lock:
        _spage_state.update({"status": "running", "message": "Starting…", "result": None})
    _spage_repl_pending = True

    import contextvars
    ctx = contextvars.copy_context()
    threading.Thread(
        target=lambda: ctx.run(_run_spage_imputation, rds_path, n_pv, genes_input,
                               use_corrected=use_corrected),
        daemon=True,
    ).start()


# ─── Morphology image overlay ─────────────────────────────────────────────────
# Max viewport width (µm) for which image tiles will load automatically
MORPH_MAX_UM = 5000

# Pseudocolors matching standard fluorescence conventions
MORPH_CHANNELS = [
    {"label": "DAPI",                   "value": "dapi",     "color": (70,  130, 255), "ch": 0},
    {"label": "ATP1A1/CD45/E-Cadherin", "value": "boundary", "color": (50,  255, 80),  "ch": 1},
    {"label": "18S",                    "value": "18s",      "color": (255, 60,  0),   "ch": 2},
    {"label": "αSMA/Vimentin",          "value": "sma",      "color": (255, 0,   220), "ch": 3},
]
MORPH_MAX_UM = 5000

# ── Change 1: reduced prefetch — only 15% extra on each side ──────────────────
_PREFETCH_FRAC = 0.15   # was 0.50; smaller = faster first render, less pan buffer

# ── Change 2 & 3: per-channel handles with LRU store cache ────────────────────
# One TiffFile + LRU-backed zarr per channel per z-level → parallel reads,
# no shared-lock needed, LRU avoids re-reading the same tiles from disk.
_morph_handles: dict = {}    # {z_level: {"handles": [(tif, arr)×n_ch], "H", "W", "n_ch"} | None}
_morph_init_lock = threading.Lock()   # only held during one-time initialisation

# Low-resolution overviews cached in memory and on disk
# Coarse (stride ~26): for viewport > ~4,400 µm  — fast zoomed-out view
# Medium (stride  ~8): for viewport 1,300–4,400 µm — eliminates mid-zoom tile reads
_morph_overview: dict = {}        # {z_level: ov_dict, "med_z_level": ov_dict}
_morph_overview_lock  = threading.Lock()
_overview_generating  = set()     # keys currently being generated in background

# Render cache: stores raw float32 channel data (brightness-independent)
# so brightness/opacity changes are instant without re-reading tiles.
_morph_render_cache: dict = {}
_morph_render_lock  = threading.Lock()

# ── Change 4: progressive rendering queue ─────────────────────────────────────
# Background hires fetches append here; the Dash callback polls and pushes a Patch.
_morph_hires_queue: list = []   # list of image-dict results; GIL protects appends/pops


class _ZarrTiffArray:
    """Thin array adapter over tifffile's ZarrTiffStore.

    Zarr ≥ 3 no longer accepts ZarrTiffStore (a plain MutableMapping) in
    zarr.open().  This class wraps the store directly, assembling tiles on
    demand and applying any stride requested via slice notation.

    Supports indexing of the form  arr[channel_int, y_slice, x_slice]
    which is all that _read_one_channel and _gen_overviews require.
    """

    def __init__(self, store):
        import json
        meta = json.loads(store[".zarray"])
        self.shape   = tuple(meta["shape"])   # (n_ch, H, W)
        self._csize  = meta["chunks"]         # [1, tile_h, tile_w]
        self._dtype  = np.dtype(meta["dtype"])
        self._store  = store

    def __getitem__(self, idx):
        if not isinstance(idx, tuple) or len(idx) != 3:
            raise IndexError(f"_ZarrTiffArray: expected (ch, y_slice, x_slice), got {idx!r}")
        c_idx, ys, xs = idx
        _, H, W = self.shape
        cy, cx   = self._csize[1], self._csize[2]

        # Resolve slices to (start, stop, step)
        y0, y1, dy = ys.indices(H) if isinstance(ys, slice) else (int(ys), int(ys) + 1, 1)
        x0, x1, dx = xs.indices(W) if isinstance(xs, slice) else (int(xs), int(xs) + 1, 1)
        y0 = max(0, y0);  y1 = min(H, y1)
        x0 = max(0, x0);  x1 = min(W, x1)
        h, w = y1 - y0, x1 - x0
        if h <= 0 or w <= 0:
            out_h = len(range(y0, y1, dy)) if h > 0 else 0
            out_w = len(range(x0, x1, dx)) if w > 0 else 0
            return np.zeros((out_h, out_w), dtype=self._dtype)

        out = np.zeros((h, w), dtype=self._dtype)
        cy0_blk = y0 // cy;  cy1_blk = (y1 - 1) // cy + 1
        cx0_blk = x0 // cx;  cx1_blk = (x1 - 1) // cx + 1

        for bcy in range(cy0_blk, cy1_blk):
            for bcx in range(cx0_blk, cx1_blk):
                raw = self._store.get(f"{c_idx}.{bcy}.{bcx}")
                if raw is None:
                    continue
                tile = np.asarray(raw).reshape(cy, cx)
                ty0 = max(y0, bcy * cy) - bcy * cy
                ty1 = min(y1, (bcy + 1) * cy) - bcy * cy
                tx0 = max(x0, bcx * cx) - bcx * cx
                tx1 = min(x1, (bcx + 1) * cx) - bcx * cx
                oy0 = max(y0, bcy * cy) - y0
                oy1 = min(y1, (bcy + 1) * cy) - y0
                ox0 = max(x0, bcx * cx) - x0
                ox1 = min(x1, (bcx + 1) * cx) - x0
                out[oy0:oy1, ox0:ox1] = tile[ty0:ty1, tx0:tx1]

        return out[::dy, ::dx]


def _open_tiff_as_array(tif):
    """Open a TiffFile as an array-like, falling back to _ZarrTiffArray for zarr v3."""
    import warnings
    warnings.filterwarnings("ignore", message=".*OME series cannot read multi-file pyramids.*")
    store = tif.aszarr()
    try:
        import zarr as zarr_mod
        return zarr_mod.open(store, mode="r")
    except (TypeError, AttributeError):
        return _ZarrTiffArray(store)


def _get_morph_handles(data_dir: str, z_level: int):
    """Lazily open one TiffFile per channel with a direct-tile array adapter."""
    with _morph_init_lock:
        if z_level in _morph_handles:
            return _morph_handles[z_level]

        import warnings, logging, tifffile
        warnings.filterwarnings("ignore", message=".*OME series cannot read multi-file pyramids.*")
        logging.getLogger("tifffile").setLevel(logging.ERROR)

        fname = f"morphology_focus_{z_level:04d}.ome.tif"
        path  = os.path.join(data_dir, "morphology_focus", fname)

        if not os.path.exists(path):
            _morph_handles[z_level] = None
            return None

        try:
            # Probe shape
            tif_probe = tifffile.TiffFile(path)
            arr_probe = _open_tiff_as_array(tif_probe)
            n_ch, H, W = arr_probe.shape
            tif_probe.close()

            # Open one handle per channel
            handles = []
            for _ in range(n_ch):
                tif_ch = tifffile.TiffFile(path)
                arr    = _open_tiff_as_array(tif_ch)
                handles.append((tif_ch, arr))

            _morph_handles[z_level] = {"handles": handles, "H": H, "W": W, "n_ch": n_ch}
            adapter = type(handles[0][1]).__name__
            print(f"  Morphology z{z_level}: {n_ch}ch × {H}×{W} px [{adapter}]", flush=True)
        except Exception as exc:
            print(f"  Warning: could not open {fname}: {exc}", flush=True)
            _morph_handles[z_level] = None

    return _morph_handles.get(z_level)


def _overview_cache_path(data_dir: str, z_level: int, medium: bool = False) -> str:
    tag = hashlib.md5(data_dir.encode()).hexdigest()[:8]
    cache_dir = os.path.join(os.path.expanduser("~"), ".xenium_explorer_cache")
    os.makedirs(cache_dir, exist_ok=True)
    suffix = "med_" if medium else ""
    return os.path.join(cache_dir, f"morph_overview_{suffix}{tag}_z{z_level}.npz")


def _load_overview_from_disk(path: str):
    """Load an NPZ overview file. Returns dict or None."""
    try:
        data = np.load(path)
        n_ch = sum(1 for k in data.files if k.startswith("ch"))
        return {
            "channels": [data[f"ch{i}"] for i in range(n_ch)],
            "stride":   int(data["stride"]),
        }
    except Exception:
        return None


def _load_or_generate_overview(data_dir: str, z_level: int):
    """Return coarse (stride ~26) overview dict, or None if not yet ready."""
    with _morph_overview_lock:
        if z_level in _morph_overview:
            return _morph_overview[z_level]

    ov_path = _overview_cache_path(data_dir, z_level, medium=False)
    if os.path.exists(ov_path):
        ov = _load_overview_from_disk(ov_path)
        if ov:
            with _morph_overview_lock:
                _morph_overview[z_level] = ov
            print(f"  Overview z{z_level} (coarse): loaded from cache", flush=True)
            return ov

    # Generate coarse + medium together in one background pass
    with _morph_overview_lock:
        if f"coarse_{z_level}" in _overview_generating:
            return None
        _overview_generating.add(f"coarse_{z_level}")

    def _gen_overviews():
        try:
            import tifffile
            fname = f"morphology_focus_{z_level:04d}.ome.tif"
            path  = os.path.join(data_dir, "morphology_focus", fname)
            if not os.path.exists(path):
                return

            tif_ov  = tifffile.TiffFile(path)
            zarr_ov = _open_tiff_as_array(tif_ov)
            H, W    = zarr_ov.shape[1], zarr_ov.shape[2]
            n_ch    = zarr_ov.shape[0]

            # ── Change 5: read at medium stride, derive coarse from it ────
            # Both reads touch all tiles anyway; reading medium first lets us
            # subsample to coarse without a second zarr pass.
            med_stride   = 8
            coarse_factor = max(1, max(H, W) // 4096) // med_stride or 3
            # Compute actual coarse stride as a multiple of med_stride
            coarse_stride = med_stride * coarse_factor

            print(f"  Generating overviews z{z_level} (medium stride={med_stride}, "
                  f"coarse stride≈{coarse_stride})…", flush=True)

            med_channels    = []
            coarse_channels = []
            for ch in range(n_ch):
                arr_med = zarr_ov[ch, ::med_stride, ::med_stride].astype(np.uint16)
                med_channels.append(arr_med)
                coarse_channels.append(arr_med[::coarse_factor, ::coarse_factor])
                print(f"    ch{ch}: medium {arr_med.shape}", flush=True)
            tif_ov.close()

            # Save medium overview
            med_path = _overview_cache_path(data_dir, z_level, medium=True)
            np.savez_compressed(med_path,
                                **{f"ch{i}": med_channels[i] for i in range(n_ch)},
                                stride=np.array(med_stride))
            ov_med = {"channels": med_channels, "stride": med_stride}

            # Save coarse overview
            ov_path_c = _overview_cache_path(data_dir, z_level, medium=False)
            np.savez_compressed(ov_path_c,
                                **{f"ch{i}": coarse_channels[i] for i in range(n_ch)},
                                stride=np.array(coarse_stride))
            ov_coarse = {"channels": coarse_channels, "stride": coarse_stride}

            with _morph_overview_lock:
                _morph_overview[z_level]              = ov_coarse
                _morph_overview[f"med_{z_level}"]     = ov_med
            print(f"  Overviews z{z_level}: generated and cached", flush=True)
        except Exception as exc:
            print(f"  Overview generation error z{z_level}: {exc}", flush=True)
        finally:
            with _morph_overview_lock:
                _overview_generating.discard(f"coarse_{z_level}")

    threading.Thread(target=_gen_overviews, daemon=True).start()
    return None


def _load_or_get_medium_overview(data_dir: str, z_level: int):
    """Return medium (stride ~8) overview dict, or None if not yet ready."""
    key = f"med_{z_level}"
    with _morph_overview_lock:
        if key in _morph_overview:
            return _morph_overview[key]

    med_path = _overview_cache_path(data_dir, z_level, medium=True)
    if os.path.exists(med_path):
        ov = _load_overview_from_disk(med_path)
        if ov:
            with _morph_overview_lock:
                _morph_overview[key] = ov
            print(f"  Overview z{z_level} (medium): loaded from cache", flush=True)
            return ov
    return None


def _compose_rgb(layers, p1p99_list, brightness, out_h, out_w, channels, crop=None):
    """Composite raw float32 layers into an RGB array, applying brightness."""
    ch_map = {c["value"]: c for c in MORPH_CHANNELS}
    rgb = np.zeros((out_h, out_w, 3), dtype=np.float32)
    for i, ch_val in enumerate(channels):
        info = ch_map.get(ch_val)
        if info is None or layers[i] is None:
            continue
        raw = layers[i]
        if crop:
            raw = raw[crop[0]:crop[1], crop[2]:crop[3]]
        p1, p99 = p1p99_list[i]
        if p99 > p1:
            norm = np.clip((raw - p1) / (p99 - p1) * brightness, 0, 1)
        else:
            norm = np.zeros_like(raw)
        color_f = np.array(info["color"], dtype=np.float32) / 255.0
        rgb += norm[..., np.newaxis] * color_f
    return rgb


def _overview_to_image(ov, channels, vp_px_x0, vp_px_y0, vp_px_x1, vp_px_y1,
                        rw_vp, rh_vp, brightness, img_opacity):
    """Crop and composite an overview dict for the given pixel viewport."""
    ov_stride = ov["stride"]
    ch_map    = {c["value"]: c for c in MORPH_CHANNELS}
    ov_y0 = vp_px_y0 // ov_stride
    ov_y1 = min(ov["channels"][0].shape[0], vp_px_y1 // ov_stride + 1)
    ov_x0 = vp_px_x0 // ov_stride
    ov_x1 = min(ov["channels"][0].shape[1], vp_px_x1 // ov_stride + 1)
    oh, ow = ov_y1 - ov_y0, ov_x1 - ov_x0
    if oh <= 0 or ow <= 0:
        return None
    rgb = np.zeros((oh, ow, 3), dtype=np.float32)
    for ch_val in channels:
        info = ch_map.get(ch_val)
        if info is None or info["ch"] >= len(ov["channels"]):
            continue
        region  = ov["channels"][info["ch"]][ov_y0:ov_y1, ov_x0:ov_x1].astype(np.float32)
        nonzero = region[region > 0]
        if nonzero.size > 50:
            p1, p99 = np.percentile(nonzero, [1, 99])
            norm = np.clip((region - p1) / (p99 - p1) * brightness, 0, 1) if p99 > p1 \
                   else np.zeros_like(region)
        else:
            norm = np.zeros_like(region)
        rgb += norm[..., np.newaxis] * (np.array(info["color"], dtype=np.float32) / 255.0)
    return _encode_overlay_jpeg(rgb, vp_px_x0, vp_px_y0, rw_vp, rh_vp, img_opacity)


def _encode_overlay_jpeg(rgb, vp_px_x0, vp_px_y0, rw_vp, rh_vp, img_opacity):
    """Clip, encode to JPEG base64, return plotly layout.images dict."""
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    img_pil   = Image.fromarray(rgb_uint8, mode="RGB")
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return dict(
        source=f"data:image/jpeg;base64,{b64}",
        xref="x", yref="y",
        x=vp_px_x0 * PIXEL_SIZE_UM,
        y=-(vp_px_y0 * PIXEL_SIZE_UM),
        sizex=rw_vp * PIXEL_SIZE_UM,
        sizey=rh_vp * PIXEL_SIZE_UM,
        xanchor="left", yanchor="top",
        layer="below", opacity=float(img_opacity),
    )


def _read_one_channel(ch_val, handles_info, py0, py1, px0, px1, stride, out_h, out_w):
    """Read one channel from its dedicated zarr handle (no lock needed)."""
    ch_map = {c["value"]: c for c in MORPH_CHANNELS}
    info   = ch_map.get(ch_val)
    if info is None or handles_info is None:
        return None, (0.0, 0.0)
    ch_idx = info["ch"]
    if ch_idx >= handles_info["n_ch"]:
        return None, (0.0, 0.0)
    _, arr = handles_info["handles"][ch_idx]
    # Each arr covers all channels; read only the one we need
    region  = arr[ch_idx, py0:py1:stride, px0:px1:stride].astype(np.float32)
    region  = region[:out_h, :out_w]
    nonzero = region[region > 0]
    if nonzero.size > 100:
        p1, p99 = float(np.percentile(nonzero, 1)), float(np.percentile(nonzero, 99))
    else:
        p1, p99 = 0.0, 0.0
    return region, (p1, p99)


def _read_channels_parallel(channels, handles_info, py0, py1, px0, px1, stride, out_h, out_w):
    """Read all requested channels in parallel threads (GIL released during JPEG2000 decode)."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(channels), 4)) as ex:
        futs = {
            ex.submit(_read_one_channel, ch, handles_info, py0, py1, px0, px1,
                      stride, out_h, out_w): ch
            for ch in channels
        }
        ordered = [concurrent.futures.as_completed  # silence linter
                   and futs[f].result()
                   for f in [next(f for f, c in futs.items() if c == ch)
                              for ch in channels]]
    layers = [r[0] for r in ordered]
    p1p99  = [r[1] for r in ordered]
    return layers, p1p99


def make_morphology_overlay(data_dir, relayout, z_level, channels, brightness, img_opacity):
    """
    Composite morphology tiles for the current viewport.

    Performance optimisations (all five changes):
      1. Prefetch fraction reduced 0.50 → 0.15 (3× fewer cold tiles).
      2. Per-channel zarr handles: parallel reads without a shared lock.
      3. LRU store cache: 50 MB per channel keeps compressed tiles in RAM.
      4. Progressive rendering: overview returned instantly; hires queued in bg.
      5. Medium-stride (stride~8) overview covers the 1,300–4,400 µm range.
    """
    if not channels or not relayout:
        return None, ""

    handles_info = _get_morph_handles(data_dir, z_level)
    if handles_info is None:
        return None, " · morphology images not found"

    H, W = handles_info["H"], handles_info["W"]
    full_w_um = W * PIXEL_SIZE_UM
    full_h_um = H * PIXEL_SIZE_UM

    # ── Viewport bounds (µm, Y-flipped plot coords) ───────────────────────
    x0_um = float(relayout.get("xaxis.range[0]", 0))
    x1_um = float(relayout.get("xaxis.range[1]", full_w_um))
    y0_um = float(relayout.get("yaxis.range[0]", -full_h_um))
    y1_um = float(relayout.get("yaxis.range[1]", 0))

    vp_w = x1_um - x0_um
    vp_h = abs(y1_um - y0_um)
    if max(vp_w, vp_h) > MORPH_MAX_UM:
        return None, f" · zoom in to ≤{MORPH_MAX_UM:,} µm for image overlay"

    # ── Viewport in image pixels ──────────────────────────────────────────
    vp_px_x0 = max(0, int(x0_um / PIXEL_SIZE_UM))
    vp_px_x1 = min(W, int(x1_um / PIXEL_SIZE_UM) + 1)
    vp_px_y0 = max(0, int(-y1_um / PIXEL_SIZE_UM))
    vp_px_y1 = min(H, int(-y0_um / PIXEL_SIZE_UM) + 1)
    if vp_px_x1 <= vp_px_x0 or vp_px_y1 <= vp_px_y0:
        return None, ""

    rw_vp  = vp_px_x1 - vp_px_x0
    rh_vp  = vp_px_y1 - vp_px_y0
    stride = max(1, max(rw_vp, rh_vp) // 800)

    # ── Path 1: coarse overview (instant, stride ≥ coarse_stride) ────────
    ov_coarse = _load_or_generate_overview(data_dir, z_level)
    if ov_coarse and stride >= ov_coarse["stride"]:
        img = _overview_to_image(ov_coarse, channels, vp_px_x0, vp_px_y0,
                                  vp_px_x1, vp_px_y1, rw_vp, rh_vp,
                                  brightness, img_opacity)
        return img, " (overview)"

    # ── Path 1b: medium overview (instant, stride ≥ 8) ───────────────────
    ov_med = _load_or_get_medium_overview(data_dir, z_level)
    if ov_med and stride >= ov_med["stride"]:
        img = _overview_to_image(ov_med, channels, vp_px_x0, vp_px_y0,
                                  vp_px_x1, vp_px_y1, rw_vp, rh_vp,
                                  brightness, img_opacity)
        if img:
            return img, " (medium overview)"

    sorted_ch  = tuple(sorted(channels))
    cache_key  = (z_level, sorted_ch, stride)

    # ── Path 2: render cache hit (brightness-independent raw layers) ──────
    with _morph_render_lock:
        entry = _morph_render_cache.get(cache_key)

    if entry is not None:
        cpx_x0, cpx_y0, cpx_x1, cpx_y1 = entry["px"]
        if (vp_px_x0 >= cpx_x0 and vp_px_y0 >= cpx_y0 and
                vp_px_x1 <= cpx_x1 and vp_px_y1 <= cpx_y1):
            cy0 = (vp_px_y0 - cpx_y0) // stride
            cy1 = cy0 + (rh_vp - 1) // stride + 1
            cx0 = (vp_px_x0 - cpx_x0) // stride
            cx1 = cx0 + (rw_vp - 1) // stride + 1
            rgb = _compose_rgb(entry["layers"], entry["p1p99"],
                               brightness, cy1 - cy0, cx1 - cx0, channels,
                               crop=(cy0, cy1, cx0, cx1))
            return _encode_overlay_jpeg(rgb, vp_px_x0, vp_px_y0,
                                        rw_vp, rh_vp, img_opacity), " (cached)"

    # ── Path 3: tile read ─────────────────────────────────────────────────
    # Prefetch region (Change 1: 15% instead of 50%)
    pad_x = int(rw_vp * _PREFETCH_FRAC)
    pad_y = int(rh_vp * _PREFETCH_FRAC)
    px_x0 = max(0, vp_px_x0 - pad_x);  px_x1 = min(W, vp_px_x1 + pad_x)
    px_y0 = max(0, vp_px_y0 - pad_y);  px_y1 = min(H, vp_px_y1 + pad_y)
    rw, rh  = px_x1 - px_x0, px_y1 - px_y0
    out_h   = (rh - 1) // stride + 1
    out_w   = (rw - 1) // stride + 1

    def _do_tile_read():
        """Read tiles for the prefetch region, update render cache, queue hires result."""
        # Change 2: parallel channel reads (each uses its own zarr handle)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(channels), 4)) as ex:
            futs    = {ex.submit(_read_one_channel, ch, handles_info,
                                 px_y0, px_y1, px_x0, px_x1,
                                 stride, out_h, out_w): ch
                       for ch in channels}
            results = {futs[f]: f.result() for f in concurrent.futures.as_completed(futs)}

        layers = [results[ch][0] for ch in channels]
        p1p99  = [results[ch][1] for ch in channels]

        with _morph_render_lock:
            _morph_render_cache[cache_key] = {
                "px": (px_x0, px_y0, px_x1, px_y1),
                "layers": layers, "p1p99": p1p99,
            }
            if len(_morph_render_cache) > 8:
                del _morph_render_cache[next(iter(_morph_render_cache))]

        # Crop to exact viewport and encode
        cy0 = (vp_px_y0 - px_y0) // stride;  cy1 = cy0 + (rh_vp - 1) // stride + 1
        cx0 = (vp_px_x0 - px_x0) // stride;  cx1 = cx0 + (rw_vp - 1) // stride + 1
        rgb = _compose_rgb(layers, p1p99, brightness, cy1 - cy0, cx1 - cx0, channels,
                           crop=(cy0, cy1, cx0, cx1))
        img = _encode_overlay_jpeg(rgb, vp_px_x0, vp_px_y0, rw_vp, rh_vp, img_opacity)
        _morph_hires_queue.append(img)   # picked up by push_hires_overlay callback

    # ── Change 4: progressive rendering ───────────────────────────────────
    # If any overview is available, return it immediately (< 20 ms) and
    # start a background thread to fetch the full-resolution tiles.
    preview = None
    if ov_med:
        preview = _overview_to_image(ov_med, channels, vp_px_x0, vp_px_y0,
                                     vp_px_x1, vp_px_y1, rw_vp, rh_vp,
                                     brightness, img_opacity)
    elif ov_coarse:
        preview = _overview_to_image(ov_coarse, channels, vp_px_x0, vp_px_y0,
                                     vp_px_x1, vp_px_y1, rw_vp, rh_vp,
                                     brightness, img_opacity)

    if preview:
        threading.Thread(target=_do_tile_read, daemon=True).start()
        return preview, " (preview → loading…)"

    # No overview available yet — read tiles synchronously (first-ever render)
    _do_tile_read()
    # The result was queued by _do_tile_read; pop it and return directly
    if _morph_hires_queue:
        img = _morph_hires_queue.pop()
        return img, ""
    return None, ""


# ─── UI micro-components ──────────────────────────────────────────────────────
def ctrl_label(text):
    return html.Div(text, style={
        "fontSize": "11px", "fontWeight": "600", "letterSpacing": "0.5px",
        "color": MUTED, "textTransform": "uppercase", "marginBottom": "6px",
    })


def stat_row(label, value):
    return html.Div([
        html.Span(label + ": ", style={"color": MUTED, "fontSize": "11px"}),
        html.Span(value,        style={"color": TEXT,  "fontSize": "12px", "fontWeight": "600"}),
    ], style={"marginBottom": "4px"})


def info_chip(label, value):
    return html.Div([
        html.Div(label, style={"fontSize": "10px", "color": MUTED}),
        html.Div(value, style={"fontSize": "13px", "fontWeight": "600", "color": TEXT}),
    ], style={
        "backgroundColor": DARK_BG, "border": f"1px solid {BORDER}",
        "borderRadius": "4px", "padding": "4px 10px",
    })


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_xenium_data(data_dir: str) -> dict:
    print(f"Loading from: {data_dir}", flush=True)
    zarr_path = os.path.join(data_dir, "spatialdata_xenium.zarr")
    if not os.path.isdir(zarr_path):
        print("  Generating spatialdata_xenium.zarr (one-time, ~30–60 s)…", flush=True)
        _build_xenium_sdata(data_dir)
    return _load_from_xenium_sdata(zarr_path, data_dir)


def _build_boundary_dict(df: pd.DataFrame) -> dict:
    """Return {cell_id: (vertex_x_array, vertex_y_array)} for fast polygon lookup."""
    # Pre-sort by cell_id for contiguous slicing via numpy split
    cell_ids = df["cell_id"].values
    vx = df["vertex_x"].values
    vy = df["vertex_y"].values
    order = np.argsort(cell_ids, kind="mergesort")
    cell_ids_s = cell_ids[order]
    vx_s = vx[order]
    vy_s = vy[order]
    # Find split points where cell_id changes
    change = np.flatnonzero(cell_ids_s[1:] != cell_ids_s[:-1]) + 1
    unique_ids = cell_ids_s[np.r_[0, change]]
    vx_groups = np.split(vx_s, change)
    vy_groups = np.split(vy_s, change)
    return {cid: (vxg, vyg) for cid, vxg, vyg in zip(unique_ids, vx_groups, vy_groups)}


def _gdf_to_bounds_dict(gdf) -> dict:
    """Convert GeoDataFrame of cell polygons back to {cell_id: (x_verts, y_verts)}."""
    result = {}
    for idx, geom in gdf.geometry.items():
        if geom is not None and not geom.is_empty:
            xy = geom.exterior.coords.xy
            result[idx] = (np.array(xy[0]), np.array(xy[1]))
    return result


def _build_xenium_sdata(data_dir: str) -> None:
    """Read all raw Xenium files and write spatialdata_xenium.zarr next to them (one-time generation)."""
    try:
        import spatialdata
        from spatialdata.models import TableModel, ShapesModel
        import anndata as ad
        import geopandas as gpd
        from shapely.geometry import Polygon
    except ImportError as e:
        raise ImportError(
            f"spatialdata, anndata, geopandas, and shapely are required to generate the zarr store: {e}\n"
            "Install with: pip install spatialdata anndata geopandas shapely"
        )

    print("  Reading raw files for spatialdata_xenium.zarr generation…", flush=True)

    with open(os.path.join(data_dir, "experiment.xenium")) as f:
        metadata = json.load(f)

    cells = pd.read_parquet(os.path.join(data_dir, "cells.parquet"))
    cells = cells.set_index("cell_id")

    umap_df = pd.read_csv(
        os.path.join(data_dir, "analysis/umap/gene_expression_2_components/projection.csv"),
        index_col="Barcode",
    )

    cluster_dir = os.path.join(data_dir, "analysis/clustering")
    cluster_methods = {}
    for method in sorted(os.listdir(cluster_dir)):
        path = os.path.join(cluster_dir, method, "clusters.csv")
        if os.path.exists(path):
            df_c = pd.read_csv(path, index_col="Barcode")
            cluster_methods[method] = df_c["Cluster"]

    with h5py.File(os.path.join(data_dir, "cell_feature_matrix.h5"), "r") as f:
        barcodes   = [b.decode() for b in f["matrix/barcodes"][:]]
        gene_names = [g.decode() for g in f["matrix/features/name"][:]]
        shape      = tuple(f["matrix/shape"][:])
        expr = sp.csc_matrix(
            (f["matrix/data"][:], f["matrix/indices"][:], f["matrix/indptr"][:]),
            shape=shape,
        ).T.tocsr()   # → (cells × genes)

    print("  Building obs DataFrame…", flush=True)
    df = cells.copy()
    df["umap_1"] = umap_df["UMAP-1"].reindex(df.index)
    df["umap_2"] = umap_df["UMAP-2"].reindex(df.index)
    for method, series in cluster_methods.items():
        df[f"clust__{method}"] = series.reindex(df.index).fillna(0).astype(int)

    # Align expression matrix rows to match df.index order
    barcode_idx = {b: i for i, b in enumerate(barcodes)}
    row_order   = np.array([barcode_idx.get(cid, -1) for cid in df.index], dtype=np.int64)
    valid_mask  = row_order >= 0
    n_cells, n_genes = len(df), len(gene_names)

    if valid_mask.all():
        expr_aligned = expr[row_order].tocsr()
    else:
        rows_v = np.where(valid_mask)[0]
        expr_aligned = sp.lil_matrix((n_cells, n_genes), dtype=np.float32)
        expr_aligned[rows_v] = expr[row_order[valid_mask]]
        expr_aligned = expr_aligned.tocsr()

    # Build AnnData
    var_df  = pd.DataFrame({"is_imputed": False}, index=pd.Index(gene_names, name="gene"))
    obs_df  = df.copy()
    obs_df.index = obs_df.index.astype(str)
    adata   = ad.AnnData(X=sp.csr_matrix(expr_aligned, dtype=np.float32), obs=obs_df, var=var_df)
    adata.uns["metadata"]        = metadata
    adata.uns["cluster_methods"] = list(cluster_methods.keys())
    adata.uns["imputed_genes"]   = []
    try:
        adata = TableModel.parse(adata)
    except Exception:
        pass

    # Build boundary GeoDataFrames
    print("  Building cell boundary shapes…", flush=True)
    cb_df = pd.read_parquet(os.path.join(data_dir, "cell_boundaries.parquet"))
    nb_df = pd.read_parquet(os.path.join(data_dir, "nucleus_boundaries.parquet"))

    def _bounds_to_gdf(bounds_dict):
        geoms = {}
        for cid, (vx, vy) in bounds_dict.items():
            try:
                geoms[str(cid)] = Polygon(zip(vx, vy))
            except Exception:
                pass
        gdf = gpd.GeoDataFrame(
            geometry=list(geoms.values()),
            index=pd.Index(list(geoms.keys()), name="cell_id"),
        )
        try:
            return ShapesModel.parse(gdf)
        except Exception:
            return gdf

    cell_shapes = _bounds_to_gdf(_build_boundary_dict(cb_df))
    nuc_shapes  = _bounds_to_gdf(_build_boundary_dict(nb_df))

    sdata = spatialdata.SpatialData(
        tables={"table": adata},
        shapes={"cell_boundaries": cell_shapes, "nucleus_boundaries": nuc_shapes},
    )

    zarr_path = os.path.join(data_dir, "spatialdata_xenium.zarr")
    print(f"  Writing {os.path.basename(zarr_path)} ({n_cells:,} cells, {n_genes} genes)…", flush=True)
    sdata.write(zarr_path)
    print("  spatialdata_xenium.zarr generation complete.", flush=True)


def _load_from_xenium_sdata(zarr_path: str, data_dir: str) -> dict:
    """Load DATA dict from a spatialdata_xenium.zarr store."""
    try:
        import spatialdata
    except ImportError:
        raise ImportError("spatialdata is required: pip install spatialdata")

    print(f"  Loading from spatialdata_xenium.zarr…", flush=True)
    sdata = spatialdata.read_zarr(zarr_path)
    adata = sdata.tables["table"]

    gene_names       = list(adata.var_names)
    gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
    barcodes         = list(adata.obs_names)
    n                = len(adata)

    # obs order == expr row order (guaranteed by _build_xenium_sdata write order)
    df_to_expr = np.arange(n, dtype=np.int64)

    cell_bounds = {}
    nuc_bounds  = {}
    try:
        cell_bounds = _gdf_to_bounds_dict(sdata.shapes["cell_boundaries"])
    except Exception as e:
        print(f"  Warning: could not load cell boundaries from zarr: {e}", flush=True)
    try:
        nuc_bounds = _gdf_to_bounds_dict(sdata.shapes["nucleus_boundaries"])
    except Exception as e:
        print(f"  Warning: could not load nucleus boundaries from zarr: {e}", flush=True)

    obs_df = adata.obs.copy()
    cluster_methods = list(adata.uns.get("cluster_methods", []))
    print(f"  Done: {n:,} cells | {len(gene_names)} genes | {len(cluster_methods)} cluster sets", flush=True)

    return {
        "metadata":          dict(adata.uns.get("metadata", {})),
        "df":                obs_df,
        "gene_names":        gene_names,
        "gene_name_to_idx":  gene_name_to_idx,
        "barcodes":          barcodes,
        "expr":              sp.csr_matrix(adata.X),
        "df_to_expr":        df_to_expr,
        "cluster_methods":   cluster_methods,
        "cell_bounds":       cell_bounds,
        "nucleus_bounds":    nuc_bounds,
        "data_dir":          data_dir,
        "sdata_path":        zarr_path,
        "gene_var":          adata.var,
        "split_corrected_expr": (
            sp.csr_matrix(adata.layers["X_corrected"])
            if "X_corrected" in adata.layers else None
        ),
        "split_corrected_imputed_genes": list(adata.uns.get("split_corrected_imputed_genes", [])),
    }


# ─── Locate & load data ───────────────────────────────────────────────────────
if len(sys.argv) > 1:
    DATA_DIR = sys.argv[1]
else:
    dirs = [d for d in os.listdir(".") if d.startswith("output-") and os.path.isdir(d)]
    if not dirs:
        print("ERROR: No Xenium output directory found. Pass path as argument.")
        sys.exit(1)
    DATA_DIR = dirs[0]

DATA = load_xenium_data(DATA_DIR)

# Precompute CSC matrix for fast column (gene) access — O(nnz_col) vs O(nnz_total) for CSR
if DATA.get("expr") is not None:
    DATA["expr_csc"] = DATA["expr"].tocsc()

# Precompute sorted gene names and default gene (avoid re-sorting on every callback)
_sorted_gene_names = sorted(DATA["gene_names"])
_default_gene = _sorted_gene_names[0] if _sorted_gene_names else None

# Gene expression cache: keyed by (gene, data_dir, use_corrected) → np.ndarray
_gene_expr_cache: dict = {}

# UMAP figure cache: (key_tuple, figure)
_umap_fig_cache: dict = {}

# ─── Extra datasets for multi-sample display ──────────────────────────────────
# Each entry is like DATA but with an added "x_offset" key (µm shift on x-axis).
EXTRA_DATASETS: list = []


# ─── Data helpers ─────────────────────────────────────────────────────────────
def get_gene_expression(gene: str) -> np.ndarray:
    """log1p expression for *gene*, aligned to DATA['df'] row order.
    Genes with ' [imp]' suffix are served from DATA['expr'] if written back to zarr,
    with fallback to _spage_state['result'] for pre-zarr cached results.
    Results are cached to avoid recomputation on pan/zoom."""
    _cache_key = (gene, DATA.get("data_dir"))
    if _cache_key in _gene_expr_cache:
        return _gene_expr_cache[_cache_key]

    if gene.endswith(" [imp]"):
        base = gene[:-6]
        # Check if imputed gene was written back into DATA["expr"]
        if base in DATA.get("gene_name_to_idx", {}):
            gene_idx = DATA["gene_name_to_idx"][base]
            expr_csc = DATA.get("expr_csc")# or DATA["expr"]
            raw  = expr_csc[:, gene_idx].toarray().ravel()
            idx  = DATA["df_to_expr"]
            vals = np.where(idx >= 0, raw[np.clip(idx, 0, len(raw) - 1)], 0.0)
            result = np.log1p(vals)
            _gene_expr_cache[_cache_key] = result
            return result
        # Fallback: old-style _spage_state result DataFrame, or streaming zarr/parquet
        with _spage_lock:
            res       = _spage_state.get("result")
            res_path  = _spage_state.get("result_path")
            res_genes = _spage_state.get("result_genes") or []
        if res is not None and base in res.columns:
            result = res[base].values.astype(np.float64)
            _gene_expr_cache[_cache_key] = result
            return result
        if res_path and base in res_genes:
            try:
                if os.path.isdir(res_path):   # zarr
                    import zarr as _zarr_ge
                    _z_ge = _zarr_ge.open_array(res_path, mode='r')
                    _gi   = res_genes.index(base)
                    result = np.array(_z_ge[:, _gi], dtype=np.float64)
                else:                          # large parquet — read single column
                    result = pd.read_parquet(res_path, columns=[base])[base].values.astype(np.float64)
                _gene_expr_cache[_cache_key] = result
                return result
            except Exception as _sge:
                print(f"  SpaGE: streaming read failed for {base}: {_sge}", flush=True)
        return np.zeros(len(DATA["df"]), dtype=np.float64)
    gene_idx = DATA["gene_name_to_idx"][gene]
    expr_csc = DATA.get("expr_csc")# or DATA["expr"]
    raw  = expr_csc[:, gene_idx].toarray().ravel()
    idx  = DATA["df_to_expr"]
    vals = np.where(idx >= 0, raw[np.clip(idx, 0, len(raw) - 1)], 0.0)
    result = np.log1p(vals)
    _gene_expr_cache[_cache_key] = result
    return result


def cluster_col(method: str) -> str:
    return f"clust__{method}"


def cluster_color_map(method: str) -> dict:
    col      = cluster_col(method)
    clusters = sorted(DATA["df"][col].unique())
    return {c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, c in enumerate(clusters)}


_viewport_arrays: dict = {}

def viewport_cell_ids(relayout: dict) -> list | None:
    """
    Return list of cell_ids whose centroid falls within the current spatial
    viewport, or None if no viewport info is available.
    Coordinates in relayout are in µm (plot space).
    """
    x0 = relayout.get("xaxis.range[0]")
    x1 = relayout.get("xaxis.range[1]")
    y0 = relayout.get("yaxis.range[0]")
    y1 = relayout.get("yaxis.range[1]")

    if None in (x0, x1, y0, y1):
        return None

    # Cache extracted numpy arrays to avoid pandas overhead on every call
    if "xc" not in _viewport_arrays:
        df = DATA["df"]
        _viewport_arrays["xc"] = df["x_centroid"].values
        _viewport_arrays["yc"] = df["y_centroid"].values
        _viewport_arrays["idx"] = df.index.values

    xc = _viewport_arrays["xc"]
    yc = _viewport_arrays["yc"]
    idx = _viewport_arrays["idx"]

    # Plot coords are already in µm (x = x_centroid, y = -y_centroid).
    xc_min, xc_max = float(x0), float(x1)
    yc_min, yc_max = float(-y1), float(-y0)   # un-negate

    mask = (xc >= xc_min) & (xc <= xc_max) & (yc >= yc_min) & (yc <= yc_max)
    return idx[mask].tolist()


def viewport_reseg_cell_ids(relayout: dict, cells_df) -> list | None:
    """Like viewport_cell_ids but uses a reseg cells_df (Baysor/Proseg) instead of DATA['df']."""
    x0 = relayout.get("xaxis.range[0]")
    x1 = relayout.get("xaxis.range[1]")
    y0 = relayout.get("yaxis.range[0]")
    y1 = relayout.get("yaxis.range[1]")
    if None in (x0, x1, y0, y1):
        return None
    xc = cells_df["x_centroid"].values
    yc = cells_df["y_centroid"].values
    idx = cells_df.index.values
    xc_min, xc_max = float(x0), float(x1)
    yc_min, yc_max = float(-y1), float(-y0)
    mask = (xc >= xc_min) & (xc <= xc_max) & (yc >= yc_min) & (yc <= yc_max)
    return [str(i) for i in idx[mask]]


def build_boundary_trace(cell_ids: list, bounds_dict: dict, color: str, name: str) -> go.Scatter | None:
    """Build a single NaN-separated Scatter trace for a set of polygon boundaries."""
    # Collect polygon arrays, compute total size for pre-allocation
    polys = []
    total = 0
    for cid in cell_ids:
        entry = bounds_dict.get(cid)
        if entry is not None:
            polys.append(entry)
            total += len(entry[0]) + 1  # +1 for NaN separator
    if not polys:
        return None
    all_x = np.empty(total, dtype=np.float64)
    all_y = np.empty(total, dtype=np.float64)
    pos = 0
    for vx, vy in polys:
        vx = np.asarray(vx, dtype=np.float64)
        vy = np.asarray(vy, dtype=np.float64)
        n = len(vx)
        all_x[pos:pos + n] = vx
        all_y[pos:pos + n] = -vy
        pos += n
        all_x[pos] = np.nan
        all_y[pos] = np.nan
        pos += 1
    return go.Scatter(
        x=all_x, y=all_y,
        mode="lines",
        line=dict(color=color, width=0.8),
        name=name,
        hoverinfo="skip",
        showlegend=True,
    )


# ─── Figure builders ──────────────────────────────────────────────────────────
def _base_layout(title, xlabel, ylabel, equal_aspect=False):
    layout = dict(
        template="plotly_dark",
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        margin=dict(l=50, r=65, t=38, b=48),
        xaxis=dict(title=xlabel, showgrid=False, zeroline=False, color=TEXT),
        yaxis=dict(title=ylabel, showgrid=False, zeroline=False, color=TEXT),
        font=dict(color=TEXT, size=11),
        title=dict(text=title, font=dict(size=13, color=TEXT), x=0.5, xanchor="center"),
        hoverlabel=dict(bgcolor=CARD_BG, font_color=TEXT),
    )
    if equal_aspect:
        layout["yaxis"]["scaleanchor"] = "x"
        layout["yaxis"]["scaleratio"]  = 1
    return layout


def _categorical_traces(x, y, df, method, size, opacity, mode="spatial"):
    cmap = cluster_color_map(method)
    col  = cluster_col(method)
    traces = []
    for c in sorted(df[col].unique()):
        mask = (df[col] == c).values
        mk   = size if mode == "spatial" else max(2, size * 1.3)
        traces.append(go.Scattergl(
            x=x[mask], y=y[mask],
            mode="markers",
            marker=dict(size=mk, color=cmap[c], opacity=opacity),
            name=f"Cluster {c}",
            legendgroup=f"c{c}",
            showlegend=(mode == "spatial"),
            text=df.index[mask].tolist(),
            customdata=df.index[mask].tolist(),
            hovertemplate="<b>%{text}</b><extra>Cluster " + str(c) + "</extra>",
        ))
    return traces


def _cell_type_color_map(labels_key: str, df) -> dict:
    """Return {cell_type: color} map consistent with _cell_type_traces ordering."""
    with _annot_lock:
        labels = _annot_state.get(labels_key)
    if labels is None:
        return {}
    aligned = labels.reindex(df.index.astype(str)).fillna("Unknown")
    cell_types = sorted(aligned.unique())
    return {ct: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, ct in enumerate(cell_types)}


def _cell_type_traces(x, y, df, size, opacity, mode="spatial", labels_key="labels_celltypist",
                      invisible=False):
    """Categorical traces coloured by cell type annotation.

    When invisible=True, scatter markers are transparent (for hover/click only when pies
    are rendered as layout shapes). Legend entries are added as zero-length traces.
    """
    with _annot_lock:
        labels = _annot_state.get(labels_key)
    if labels is None:
        return [go.Scattergl(x=x, y=y, mode="markers",
                             marker=dict(size=size, color=MUTED, opacity=opacity),
                             name="No annotation")]

    # Align labels to df index — cast both to str to handle int vs str mismatch
    aligned = labels.reindex(df.index.astype(str)).fillna("Unknown")
    cell_types = sorted(aligned.unique())
    cmap = {ct: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, ct in enumerate(cell_types)}

    traces = []
    for ct in cell_types:
        mask = (aligned == ct).values
        mk   = size if mode == "spatial" else max(2, size * 1.3)
        if invisible:
            # Ghost scatter for hover/click; add a visible legend-only trace separately
            traces.append(go.Scattergl(
                x=x[mask], y=y[mask], mode="markers",
                marker=dict(size=mk, color=cmap[ct], opacity=0),
                name=ct, legendgroup=ct, showlegend=False,
                text=df.index[mask].tolist(), customdata=df.index[mask].tolist(),
                hovertemplate="<b>%{text}</b><extra>" + ct + "</extra>",
            ))
            # Zero-length legend entry so the legend still shows cell type → color
            traces.append(go.Scattergl(
                x=[None], y=[None], mode="markers",
                marker=dict(size=mk, color=cmap[ct]),
                name=ct, legendgroup=ct, showlegend=True,
            ))
        else:
            traces.append(go.Scattergl(
                x=x[mask], y=y[mask], mode="markers",
                marker=dict(size=mk, color=cmap[ct], opacity=opacity),
                name=ct, legendgroup=ct,
                showlegend=(mode == "spatial"),
                text=df.index[mask].tolist(), customdata=df.index[mask].tolist(),
                hovertemplate="<b>%{text}</b><extra>" + ct + "</extra>",
            ))
    return traces


def _get_expr_values(gene: str, alt_res=None, use_corrected: bool = False) -> "np.ndarray | None":
    """Unified expression lookup supporting original and SPLIT-corrected counts."""
    lookup = gene.replace(" [imp]", "").replace(" [corr+imp]", "")
    if alt_res is not None:
        _corr = alt_res.get("split_corrected_expr") if use_corrected else None
        expr = _corr if _corr is not None else alt_res.get("expr")
        gni  = alt_res.get("gene_name_to_idx") or DATA.get("gene_name_to_idx", {})
    else:
        _corr = DATA.get("split_corrected_expr") if use_corrected else None
        expr = _corr if _corr is not None else DATA.get("expr_csc")# or DATA.get("expr")
        gni  = DATA.get("gene_name_to_idx", {})
    if expr is not None and lookup in gni:
        idx = gni[lookup]
        if idx < expr.shape[1]:
            return np.log1p(expr.getcol(idx).toarray().ravel().astype(np.float32))
    # Streaming SpaGE fallback — gene in zarr but not in expr index
    # For reseg: prefer per-alt_res zarr (correct row count) over global spage_state
    if alt_res is not None:
        _rp = alt_res.get("spage_result_path")
        _rg = alt_res.get("spage_result_genes") or []
    else:
        _rp, _rg = None, []
    if not (_rp and lookup in _rg):
        # Only fall back to global Xenium SpaGE state when NOT in reseg mode;
        # the global zarr has Xenium cell count rows and would misalign with reseg cells.
        if alt_res is None:
            with _spage_lock:
                _rp = _spage_state.get("result_path")
                _rg = _spage_state.get("result_genes") or []
        else:
            return None
    if _rp and lookup in _rg:
        try:
            if os.path.isdir(_rp):
                import zarr as _zarr_ev
                _gi_ev = _rg.index(lookup)
                return np.log1p(np.array(
                    _zarr_ev.open_array(_rp, mode='r')[:, _gi_ev], dtype=np.float32))
        except Exception as _e:
            print(f"  SpaGE zarr read error for {lookup!r}: {_e}", flush=True)
    return None


def _get_reseg_expr_values(gene: str, alt_res: dict) -> "np.ndarray | None":
    """log1p expression for *gene* in resegmented cells, or None if not available.
    Uses alt_res['gene_name_to_idx'] if present (includes reseg-imputed genes),
    falling back to DATA['gene_name_to_idx'] for panel genes."""
    expr = alt_res.get("expr")
    if expr is None:
        return None
    # Strip [imp] suffix if present
    lookup = gene[:-6] if gene.endswith(" [imp]") else gene
    gni = alt_res.get("gene_name_to_idx") or DATA["gene_name_to_idx"]
    if lookup not in gni:
        return None
    idx = gni[lookup]
    if idx >= expr.shape[1]:
        return None
    col = expr.getcol(idx)
    vals = np.asarray(col.todense()).flatten().astype(np.float32)
    return np.log1p(vals)


def make_spatial_fig(color_by, method, gene, size, opacity,
                     boundary_toggles, relayout,
                     morph_image=None, extra_title="", baysor_active=False, proseg_active=False,
                     use_corrected: bool = False):
    # Choose data source: Proseg > Baysor > original Xenium (priority order)
    with _proseg_lock:
        pres = _proseg_state["result"] if proseg_active and _proseg_state["result"] else None
    with _baysor_lock:
        bres = _baysor_state["result"] if baysor_active and not pres and _baysor_state["result"] else None

    alt_res = pres or bres
    _separator_shapes = []   # populated later for multi-sample mode
    _pie_shapes = []         # RCTD pie chart wedge shapes (single-dataset, Xenium only)
    if alt_res is not None:
        source_label = " [Proseg]" if pres else " [Baysor]"
        bdf      = alt_res["cells_df"]
        x        =  bdf["x_centroid"].values
        y        = -bdf["y_centroid"].values
        cell_ids     = bdf.index.tolist()
        cell_ids_str = [str(c) for c in cell_ids]
        show_legend  = False
        traces       = None   # filled below

        if color_by == "cluster":
            # Look for any cluster column in cells_df (added after UMAP/clustering on reseg cells)
            cluster_col = next((c for c in bdf.columns if c.startswith("cluster")), None)
            if cluster_col is not None:
                cluster_vals = bdf[cluster_col].astype(str)
                unique_clusters = sorted(cluster_vals.unique(), key=lambda v: (0, int(v)) if v.isdigit() else (1, v))
                cmap = {cl: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, cl in enumerate(unique_clusters)}
                traces = []
                for cl in unique_clusters:
                    mask = (cluster_vals == cl).values
                    traces.append(go.Scattergl(
                        x=x[mask], y=y[mask], mode="markers",
                        marker=dict(size=size, color=cmap[cl], opacity=opacity),
                        name=f"Cluster {cl}", legendgroup=cl, showlegend=True,
                        text=np.array(cell_ids_str)[mask].tolist(),
                        customdata=np.array(cell_ids_str)[mask].tolist(),
                        hovertemplate="<b>%{text}</b><extra>Cluster " + cl + "</extra>",
                    ))
                show_legend = True
            # If no cluster column: fall through to generic fallback (transcript_counts)

        elif color_by.startswith("cell_type:"):
            _method = color_by.split(":")[1]
            _lkey   = _labels_key_for_method(_method, alt_res)
            with _annot_lock:
                labels = _annot_state.get(_lkey)
            if labels is not None:
                aligned = labels.reindex(pd.Index(cell_ids_str)).fillna("Unknown")
                n_unknown = (aligned == "Unknown").sum()
                if n_unknown > len(aligned) * 0.95:
                    print(f"  Warning: cell_type:{_method} produced {n_unknown}/{len(aligned)} Unknown "
                          f"— labels index sample: {list(labels.index[:3])!r}, "
                          f"cells sample: {cell_ids_str[:3]!r}", flush=True)
                cell_types = sorted(aligned.unique())
                cmap = {ct: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, ct in enumerate(cell_types)}
                traces = []
                for ct in cell_types:
                    mask = (aligned == ct).values
                    traces.append(go.Scattergl(
                        x=x[mask], y=y[mask], mode="markers",
                        marker=dict(size=size, color=cmap[ct], opacity=opacity),
                        name=ct, legendgroup=ct, showlegend=True,
                        text=np.array(cell_ids_str)[mask].tolist(),
                        customdata=np.array(cell_ids_str)[mask].tolist(),
                        hovertemplate="<b>%{text}</b><extra>" + ct + "</extra>",
                    ))
                show_legend = True

        elif color_by == "gene":
            vals = _get_expr_values(gene, alt_res, use_corrected=use_corrected)
            if vals is not None:
                traces = [go.Scattergl(
                    x=x, y=y, mode="markers",
                    marker=dict(size=size, color=vals, colorscale="Plasma", opacity=opacity,
                                showscale=True,
                                colorbar=dict(title=f"{gene}<br>(log1p)", thickness=12, len=0.5, x=1.02)),
                    text=cell_ids_str, customdata=cell_ids_str,
                    hovertemplate="<b>%{text}</b><br>" + gene + ": %{marker.color:.2f}<extra></extra>",
                    name=gene,
                )]

        elif color_by and color_by.startswith("roi_") and color_by in bdf.columns:
            labels = bdf[color_by].fillna("(none)").astype(str)
            u_vals = [v for v in sorted(labels.unique()) if v != "(none)"]
            cmap   = {v: _roi_color(i) for i, v in enumerate(u_vals)}
            cmap["(none)"] = "#666666"
            colors = labels.map(cmap).values
            traces = [go.Scattergl(
                x=x, y=y, mode="markers",
                marker=dict(size=size, color=colors, opacity=opacity),
                text=cell_ids_str, customdata=cell_ids_str,
                hovertemplate="<b>%{text}</b><br>" + color_by + ": %{text}<extra></extra>",
                name=color_by,
            )]
            show_legend = False

        if traces is None:
            # Fallback: QC column if available, otherwise transcript counts
            if color_by in bdf.columns:
                vals  = bdf[color_by].values
                label = QC_METRICS.get(color_by, color_by)
                cs    = COLORSCALES.get(color_by, "Viridis")
            else:
                vals  = bdf["transcript_counts"].values
                label = "Transcript Counts"
                cs    = "Viridis"
            traces = [go.Scattergl(
                x=x, y=y, mode="markers",
                marker=dict(size=size, color=vals, colorscale=cs, opacity=opacity, showscale=True,
                            colorbar=dict(title=label, thickness=12, len=0.5, x=1.02)),
                text=cell_ids_str, customdata=cell_ids_str,
                hovertemplate="<b>Cell %{text}</b><br>" + label + ": %{marker.color:.2f}<extra></extra>",
                name=label,
            )]
    else:
        df  = DATA["df"]
        x   =  df["x_centroid"].values   # x_centroid is in µm
        y   = -df["y_centroid"].values   # y_centroid is in µm, negate for plot
        source_label    = ""

        # ── Viewport-filtered copies for pie chart threshold + rendering ─
        df_vp, x_vp, y_vp = df, x, y   # default: full dataset
        if relayout:
            try:
                _x0 = float(relayout.get("xaxis.range[0]", -1e18))
                _x1 = float(relayout.get("xaxis.range[1]",  1e18))
                _y0 = float(relayout.get("yaxis.range[0]", -1e18))
                _y1 = float(relayout.get("yaxis.range[1]",  1e18))
                if _x1 - _x0 < 1e17:  # real range, not default placeholder
                    _vp_mask = (x >= _x0) & (x <= _x1) & (y >= _y0) & (y <= _y1)
                    df_vp = df[_vp_mask]
                    x_vp  = x[_vp_mask]
                    y_vp  = y[_vp_mask]
            except Exception:
                pass

        # ── Multi-sample: merge extra datasets ──────────────────────────
        if EXTRA_DATASETS:
            # Concatenate extra datasets into combined arrays
            all_x       = [x]
            all_y       = [y]
            all_ids     = [df.index.tolist()]
            all_dfs     = [df]
            all_offsets = [0.0]
            for eds in EXTRA_DATASETS:
                edf    = eds["df"]
                offset = eds["x_offset"]
                all_x.append(edf["x_centroid"].values + offset)
                all_y.append(-edf["y_centroid"].values)
                all_ids.append(edf.index.tolist())
                all_dfs.append(edf)
                all_offsets.append(offset)

            x_combined   = np.concatenate(all_x)
            y_combined   = np.concatenate(all_y)
            ids_combined = [cid for chunk in all_ids for cid in chunk]

            # Build per-sample separator annotations (vertical dashed lines)
            _separator_shapes = []
            for eds in EXTRA_DATASETS:
                sep_x = eds["x_offset"] - 250  # midpoint of gap
                _separator_shapes.append(dict(
                    type="line", x0=sep_x, x1=sep_x,
                    y0=0, y1=1, yref="paper",
                    line=dict(color=MUTED, width=1, dash="dot"),
                ))

            # Build color values across all datasets
            if color_by == "cluster":
                # Combine cluster IDs per dataset — use dataset index as prefix to avoid collisions
                all_clust_vals  = []
                all_clust_names = []
                cmap_combined   = {}
                color_idx       = 0
                for di, (edf, offset) in enumerate(zip(all_dfs, all_offsets)):
                    _method = method or (edf.get("cluster_methods", [None])[0]
                                         if hasattr(edf, "get") else None)
                    # edf is a plain DataFrame; cluster_methods come from DATA or EXTRA_DATASETS[di-1]
                    ds = DATA if di == 0 else EXTRA_DATASETS[di - 1]
                    _method = method if method in ds.get("cluster_methods", []) else (ds.get("cluster_methods") or [method])[0]
                    col_name = f"clust__{_method}"
                    if col_name in edf.columns:
                        raw_vals = edf[col_name].values
                        # Prefix dataset index so cluster IDs don't collide across samples
                        prefixed = [f"S{di}:C{v}" for v in raw_vals]
                        all_clust_vals.extend(prefixed)
                        for pv in prefixed:
                            if pv not in cmap_combined:
                                cmap_combined[pv] = CLUSTER_COLORS[color_idx % len(CLUSTER_COLORS)]
                                color_idx += 1
                        all_clust_names.extend(prefixed)
                    else:
                        placeholders = [f"S{di}:C0"] * len(edf)
                        all_clust_vals.extend(placeholders)
                        all_clust_names.extend(placeholders)

                # Build one trace per unique cluster-label across all samples
                unique_labels = list(dict.fromkeys(all_clust_vals))
                arr_x   = x_combined
                arr_y   = y_combined
                arr_ids = np.array(ids_combined)
                arr_lbl = np.array(all_clust_vals)
                traces  = []
                for lbl in unique_labels:
                    mask = arr_lbl == lbl
                    traces.append(go.Scattergl(
                        x=arr_x[mask], y=arr_y[mask], mode="markers",
                        marker=dict(size=size, color=cmap_combined[lbl], opacity=opacity),
                        name=lbl, legendgroup=lbl, showlegend=True,
                        text=arr_ids[mask].tolist(),
                        customdata=arr_ids[mask].tolist(),
                        hovertemplate="<b>%{text}</b><extra>" + lbl + "</extra>",
                    ))
                show_legend = True

            elif color_by.startswith("cell_type:"):
                # Combine cell type labels from each dataset's annotation state
                _method = color_by.split(":")[1]
                all_labels_list = []
                with _annot_lock:
                    primary_labels = _annot_state.get(f"labels_{_method}")
                for di, edf in enumerate(all_dfs):
                    if di == 0:
                        lbl_series = primary_labels
                    else:
                        lbl_series = EXTRA_DATASETS[di - 1].get("_cell_type_labels")
                    if lbl_series is not None:
                        aligned = lbl_series.reindex(edf.index).fillna("Unknown")
                        all_labels_list.extend(aligned.tolist())
                    else:
                        all_labels_list.extend(["Unknown"] * len(edf))

                arr_lbl  = np.array(all_labels_list)
                arr_x    = x_combined
                arr_y    = y_combined
                arr_ids  = np.array(ids_combined)
                cell_types = sorted(set(all_labels_list))
                cmap = {ct: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, ct in enumerate(cell_types)}
                traces = []
                for ct in cell_types:
                    mask = arr_lbl == ct
                    traces.append(go.Scattergl(
                        x=arr_x[mask], y=arr_y[mask], mode="markers",
                        marker=dict(size=size, color=cmap[ct], opacity=opacity),
                        name=ct, legendgroup=ct, showlegend=True,
                        text=arr_ids[mask].tolist(),
                        customdata=arr_ids[mask].tolist(),
                        hovertemplate="<b>%{text}</b><extra>" + ct + "</extra>",
                    ))
                show_legend = True

            elif color_by == "gene":
                # Collect expression per dataset
                all_vals = []
                for di, edf in enumerate(all_dfs):
                    if di == 0:
                        all_vals.append(get_gene_expression(gene))
                    else:
                        eds = EXTRA_DATASETS[di - 1]
                        g_idx = eds.get("gene_name_to_idx", {}).get(gene)
                        if g_idx is not None:
                            raw  = eds["expr"][:, g_idx].toarray().flatten()
                            idx_ = eds["df_to_expr"]
                            vals_i = np.where(idx_ >= 0, raw[np.clip(idx_, 0, len(raw) - 1)], 0.0)
                            all_vals.append(np.log1p(vals_i))
                        else:
                            all_vals.append(np.zeros(len(edf)))
                vals_combined = np.concatenate(all_vals)
                traces = [go.Scattergl(
                    x=x_combined, y=y_combined, mode="markers",
                    marker=dict(size=size, color=vals_combined, colorscale="Plasma", opacity=opacity,
                                showscale=True,
                                colorbar=dict(title=f"{gene}<br>(log1p)", thickness=12, len=0.5, x=1.02)),
                    text=ids_combined, customdata=ids_combined,
                    hovertemplate="<b>%{text}</b><br>" + gene + ": %{marker.color:.2f}<extra></extra>",
                    name=gene,
                )]
                show_legend = False

            else:
                # QC / numeric column
                label = QC_METRICS.get(color_by, color_by)
                cs    = COLORSCALES.get(color_by, "Viridis")
                all_vals = []
                for edf in all_dfs:
                    if color_by in edf.columns:
                        all_vals.append(edf[color_by].values)
                    else:
                        all_vals.append(np.zeros(len(edf)))
                vals_combined = np.concatenate(all_vals)
                traces = [go.Scattergl(
                    x=x_combined, y=y_combined, mode="markers",
                    marker=dict(size=size, color=vals_combined, colorscale=cs, opacity=opacity,
                                showscale=True,
                                colorbar=dict(title=label, thickness=12, len=0.5, x=1.02)),
                    text=ids_combined, customdata=ids_combined,
                    hovertemplate="<b>%{text}</b><br>" + label + ": %{marker.color:.2f}<extra></extra>",
                    name=label,
                )]
                show_legend = False

            source_label = f" [{len(EXTRA_DATASETS) + 1} samples]"

        else:
            # ── Single sample (no extra datasets) ───────────────────────
            # _separator_shapes already = [] from function-level init

            # ── Cell scatter traces ──────────────────────────────────────────
            if color_by == "cluster":
                traces, show_legend = _categorical_traces(x, y, df, method, size, opacity), True
            elif color_by.startswith("cell_type:"):
                _method = color_by.split(":")[1]
                _lkey = f"labels_{_method}"
                _wkey = f"rctd_weights_{_lkey}"
                with _annot_lock:
                    _weights_df = _annot_state.get(_wkey)
                if _weights_df is not None and len(df_vp) <= PIE_THRESHOLD:
                    # Pie chart mode: invisible ghost scatter for hover + SVG wedge shapes
                    _ct_colors = _cell_type_color_map(_lkey, df_vp)
                    traces = _cell_type_traces(x_vp, y_vp, df_vp, size, opacity, "spatial",
                                               labels_key=_lkey, invisible=True)
                    _pie_shapes = _build_rctd_pie_shapes(df_vp, _weights_df, _ct_colors)
                else:
                    traces = _cell_type_traces(x, y, df, size, opacity, "spatial",
                                               labels_key=_lkey)
                show_legend = True
            elif color_by == "gene":
                _cv = _get_expr_values(gene, use_corrected=True) if use_corrected else None
                vals = _cv if _cv is not None else get_gene_expression(gene)
                traces = [go.Scattergl(
                    x=x, y=y, mode="markers",
                    marker=dict(size=size, color=vals, colorscale="Plasma", opacity=opacity,
                                showscale=True,
                                colorbar=dict(title=f"{gene}<br>(log1p)", thickness=12, len=0.5, x=1.02)),
                    text=df.index.tolist(), customdata=df.index.tolist(),
                    hovertemplate="<b>%{text}</b><br>" + gene + ": %{marker.color:.2f}<extra></extra>",
                    name=gene,
                )]
                show_legend = False
            elif color_by and color_by.startswith("roi_") and color_by in df.columns:
                labels = df[color_by].fillna("(none)").astype(str)
                u_vals = [v for v in sorted(labels.unique()) if v != "(none)"]
                cmap   = {v: _roi_color(i) for i, v in enumerate(u_vals)}
                cmap["(none)"] = "#666666"
                colors = labels.map(cmap).values
                ids    = df.index.tolist()
                traces = [go.Scattergl(
                    x=x, y=y, mode="markers",
                    marker=dict(size=size, color=colors, opacity=opacity),
                    text=ids, customdata=ids,
                    hovertemplate="<b>%{text}</b><extra></extra>",
                    name=color_by,
                )]
                show_legend = False
            else:
                label = QC_METRICS.get(color_by, color_by)
                traces = [go.Scattergl(
                    x=x, y=y, mode="markers",
                    marker=dict(size=size, color=df[color_by].values,
                                colorscale=COLORSCALES.get(color_by, "Viridis"),
                                opacity=opacity, showscale=True,
                                colorbar=dict(title=label, thickness=12, len=0.5, x=1.02)),
                    text=df.index.tolist(), customdata=df.index.tolist(),
                    hovertemplate="<b>%{text}</b><br>" + label + ": %{marker.color:.2f}<extra></extra>",
                    name=label,
                )]
                show_legend = False

    # ── Boundary overlay traces ──────────────────────────────────────────
    boundary_toggles = boundary_toggles or []
    boundary_status = ""

    # Each boundary source is rendered independently
    _needs_zoom_msg = False

    def _add_boundary(bounds_dict, color, name, reseg_cells_df=None):
        """Render a boundary set using viewport filtering. reseg_cells_df uses Baysor/Proseg cells."""
        nonlocal _needs_zoom_msg, show_legend
        if not bounds_dict:
            return
        if reseg_cells_df is not None:
            visible = viewport_reseg_cell_ids(relayout or {}, reseg_cells_df)
        else:
            visible = viewport_cell_ids(relayout or {})
        if visible is None:
            _needs_zoom_msg = True
            return
        if len(visible) > BOUNDARY_CELL_LIMIT:
            return
        t = build_boundary_trace(visible, bounds_dict, color, name)
        if t:
            traces.append(t)
            show_legend = True

    # "Xenium Cell/Nuclei Boundaries" always show original Xenium boundaries (zoom required).
    if "cell" in boundary_toggles:
        _add_boundary(DATA["cell_bounds"], "#00d4ff", "Xenium Cell Boundaries")
    if "nucleus" in boundary_toggles:
        _add_boundary(DATA.get("nucleus_bounds", {}), "#ff9f43", "Xenium Nuclei Boundaries")

    # Proseg boundaries (zoom required, uses Proseg cell centroids for viewport)
    if "proseg" in boundary_toggles:
        with _proseg_lock:
            _pres_local = _proseg_state.get("result")
        if _pres_local:
            _add_boundary(_pres_local["cell_bounds"], "#2ed573", "Proseg Boundaries",
                          reseg_cells_df=_pres_local["cells_df"])

    # Baysor boundaries (zoom required, uses Baysor cell centroids for viewport)
    if "baysor" in boundary_toggles:
        with _baysor_lock:
            _bres_local = _baysor_state.get("result")
        if _bres_local:
            _add_boundary(_bres_local["cell_bounds"], "#a29bfe", "Baysor Boundaries",
                          reseg_cells_df=_bres_local["cells_df"])

    if _needs_zoom_msg:
        boundary_status = " · zoom in to see Xenium boundaries"

    title_text = "Spatial View" + source_label + boundary_status + extra_title

    fig = go.Figure(data=traces)
    if morph_image:
        fig.update_layout(images=[morph_image])
    layout_kwargs = dict(
        **_base_layout(title_text, "X (µm)", "Y (µm)", equal_aspect=True),
        showlegend=show_legend,
        legend=dict(
            bgcolor="rgba(0,0,0,0.45)", font=dict(size=10),
            itemsizing="constant", tracegroupgap=1,
            x=0.01, y=0.99, xanchor="left", yanchor="top",
        ),
        uirevision="spatial",
    )
    all_shapes = list(_separator_shapes) + list(_pie_shapes)
    with _roi_lock:
        roi_visible = _roi_state["show"]
        rois_snap   = list(_roi_state["rois"])
        pending     = _roi_state.get("pending_hull")
    if roi_visible and rois_snap:
        all_shapes += _roi_shapes_for_fig(rois_snap)
    if pending:
        all_shapes += _roi_shapes_for_fig([], pending_hull=pending)
    if all_shapes:
        layout_kwargs["shapes"] = all_shapes
    fig.update_layout(**layout_kwargs)
    # Explicitly preserve viewport ranges to prevent zoom resets on figure rebuild
    if relayout:
        rx0 = relayout.get("xaxis.range[0]")
        rx1 = relayout.get("xaxis.range[1]")
        ry0 = relayout.get("yaxis.range[0]")
        ry1 = relayout.get("yaxis.range[1]")
        if all(v is not None for v in (rx0, rx1, ry0, ry1)):
            fig.update_layout(
                xaxis_range=[float(rx0), float(rx1)],
                yaxis_range=[float(ry0), float(ry1)],
            )
    return fig


_umap_df_cache = {}

def make_umap_fig(color_by, method, gene, size, opacity,
                  baysor_active=False, proseg_active=False,
                  use_corrected: bool = False):
    mk = max(2, size * 1.2)

    # Check if reseg UMAP is available and the matching source is active
    with _umap_reseg_lock:
        reseg_umap = _umap_reseg_state["result"] if _umap_reseg_state["status"] == "done" else None

    # Determine active reseg source
    with _proseg_lock:
        pres = _proseg_state["result"] if proseg_active and _proseg_state["result"] else None
    with _baysor_lock:
        bres = _baysor_state["result"] if baysor_active and not pres and _baysor_state["result"] else None
    alt_res = pres or bres

    # Fall back to UMAP columns embedded in cells_df (loaded from zarr) when no explicit reseg UMAP.
    # When SPLIT-corrected counts are active, prefer split_umap_1/2 if available.
    if alt_res is not None and reseg_umap is None:
        _bdf = alt_res["cells_df"]
        if use_corrected and "split_umap_1" in _bdf.columns and _bdf["split_umap_1"].notna().any():
            reseg_umap = _bdf.dropna(subset=["split_umap_1", "split_umap_2"])
        elif "umap_1" in _bdf.columns and _bdf["umap_1"].notna().any():
            reseg_umap = _bdf.dropna(subset=["umap_1", "umap_2"])

    if alt_res is not None and reseg_umap is not None:
        # Reseg UMAP: color by whatever is available
        udf = reseg_umap
        # Use SPLIT UMAP coordinates when corrected counts are active and available
        if use_corrected and "split_umap_1" in udf.columns and udf["split_umap_1"].notna().any():
            xu = udf["split_umap_1"].values
            yu = udf["split_umap_2"].values
        else:
            xu = udf["umap_1"].values
            yu = udf["umap_2"].values
        cell_ids  = udf.index.tolist()
        if color_by.startswith("cell_type:"):
            _method    = color_by.split(":")[1]
            labels_key = _labels_key_for_method(_method, alt_res)
            with _annot_lock:
                labels = _annot_state.get(labels_key)
            if labels is not None:
                aligned    = labels.reindex(pd.Index(cell_ids)).fillna("Unknown")
                cell_types = sorted(aligned.unique())
                cmap = {ct: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, ct in enumerate(cell_types)}
                traces = []
                for ct in cell_types:
                    mask = (aligned == ct).values
                    traces.append(go.Scattergl(
                        x=xu[mask], y=yu[mask], mode="markers",
                        marker=dict(size=mk, color=cmap[ct], opacity=opacity),
                        name=ct, showlegend=False,
                        text=np.array(cell_ids)[mask].tolist(),
                        customdata=np.array(cell_ids)[mask].tolist(),
                        hovertemplate="<b>%{text}</b><extra>" + ct + "</extra>",
                    ))
            else:
                color_by = "transcript_counts"  # fall through

        if color_by == "gene":
            vals = _get_expr_values(gene, alt_res, use_corrected=use_corrected)
            if vals is not None:
                traces = [go.Scattergl(
                    x=xu, y=yu, mode="markers",
                    marker=dict(size=mk, color=vals, colorscale="Plasma",
                                opacity=opacity, showscale=False),
                    text=cell_ids, customdata=cell_ids,
                    hovertemplate="<b>%{text}</b><extra></extra>", name=gene,
                )]
            else:
                color_by = "transcript_counts"

        if color_by == "cluster" and not color_by.startswith("cell_type:") and color_by != "gene":
            bdf = alt_res["cells_df"]
            _cluster_col = next((c for c in bdf.columns if c.startswith("cluster")), None)
            if _cluster_col is not None:
                _cvals = bdf[_cluster_col].reindex(udf.index).astype(str)
                _unique = sorted(_cvals.unique(), key=lambda v: (0, int(v)) if v.isdigit() else (1, v))
                _cmap = {cl: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, cl in enumerate(_unique)}
                traces = []
                for cl in _unique:
                    mask = (_cvals == cl).values
                    traces.append(go.Scattergl(
                        x=xu[mask], y=yu[mask], mode="markers",
                        marker=dict(size=mk, color=_cmap[cl], opacity=opacity),
                        name=cl, showlegend=False,
                        text=np.array(cell_ids)[mask].tolist(),
                        customdata=np.array(cell_ids)[mask].tolist(),
                        hovertemplate="<b>%{text}</b><extra>Cluster " + cl + "</extra>",
                    ))
            else:
                color_by = "transcript_counts"

        if not color_by.startswith("cell_type:") and color_by != "gene" and color_by != "cluster":
            bdf = alt_res["cells_df"]
            if color_by in bdf.columns:
                vals  = bdf["transcript_counts"].values if color_by == "transcript_counts" \
                        else bdf[color_by].values
                label = QC_METRICS.get(color_by, color_by)
                cs    = COLORSCALES.get(color_by, "Viridis")
            else:
                vals  = bdf["transcript_counts"].values
                label = "Transcript Counts"
                cs    = "Viridis"
            traces = [go.Scattergl(
                x=xu, y=yu, mode="markers",
                marker=dict(size=mk, color=vals, colorscale=cs,
                            opacity=opacity, showscale=False),
                text=cell_ids, customdata=cell_ids,
                hovertemplate="<b>%{text}</b><br>" + label + ": %{marker.color:.2f}<extra></extra>",
                name=label,
            )]

        title = "UMAP [Proseg]" if pres else "UMAP [Baysor]"
    else:
        # Original Xenium UMAP
        if "df" not in _umap_df_cache:
            _umap_df_cache["df"] = DATA["df"].dropna(subset=["umap_1", "umap_2"])
        df = _umap_df_cache["df"]
        xu = df["umap_1"].values
        yu = df["umap_2"].values

        if color_by == "cluster":
            traces = _categorical_traces(xu, yu, df, method, mk, opacity, mode="umap")
        elif color_by.startswith("cell_type:"):
            _method = color_by.split(":")[1]
            traces = _cell_type_traces(xu, yu, df, mk, opacity, mode="umap",
                                       labels_key=f"labels_{_method}")
        elif color_by == "gene":
            has_umap = DATA["df"]["umap_1"].notna()
            _cv = _get_expr_values(gene, use_corrected=use_corrected) if use_corrected else None
            vals = (_cv[has_umap.values] if _cv is not None
                    else get_gene_expression(gene)[has_umap.values])
            traces = [go.Scattergl(
                x=xu, y=yu, mode="markers",
                marker=dict(size=mk, color=vals, colorscale="Plasma",
                            opacity=opacity, showscale=False),
                text=df.index.tolist(), customdata=df.index.tolist(),
                hovertemplate="<b>%{text}</b><extra></extra>", name=gene,
            )]
        else:
            label = QC_METRICS.get(color_by, color_by)
            traces = [go.Scattergl(
                x=xu, y=yu, mode="markers",
                marker=dict(size=mk, color=df[color_by].values,
                            colorscale=COLORSCALES.get(color_by, "Viridis"),
                            opacity=opacity, showscale=False),
                text=df.index.tolist(), customdata=df.index.tolist(),
                hovertemplate="<b>%{text}</b><br>" + label + ": %{marker.color:.2f}<extra></extra>",
                name=label,
            )]
        title = "UMAP"

    fig = go.Figure(data=traces)
    fig.update_layout(
        **_base_layout(title, "UMAP 1", "UMAP 2"),
        showlegend=False, uirevision="umap",
    )
    return fig


# ─── App layout ───────────────────────────────────────────────────────────────
meta            = DATA["metadata"]
cluster_methods = DATA["cluster_methods"]
gene_names      = DATA["gene_names"]


def method_label(m):
    return (m.replace("gene_expression_", "")
             .replace("_clusters", "")
             .replace("_", " ").title())


cluster_options = [{"label": method_label(m), "value": m} for m in cluster_methods]

# Auto-load any previously cached SpaGE results for this dataset
_annot_autoload()
_spage_autoload()


# ─── SpatialData / Sopa backend ───────────────────────────────────────────────

def _sdata_cache_path() -> str:
    """Return the directory path for the SpatialData Zarr cache for this dataset."""
    tag = hashlib.md5(DATA["data_dir"].encode()).hexdigest()[:12]
    cache_dir = os.path.join(os.path.expanduser("~"), ".xenium_explorer_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"spatialdata_{tag}.zarr")


def _sdata_fix_categories(sdata) -> None:
    """Eagerly compute feature_name categories to avoid dask performance issues."""
    try:
        for pts_key in sdata.points:
            pts = sdata.points[pts_key]
            if "feature_name" in pts.columns:
                col = pts["feature_name"]
                if hasattr(col, "cat") and not col.cat.known:
                    sdata.points[pts_key]["feature_name"] = col.cat.as_known()
    except Exception as exc:
        print(f"  to_spatialdata: category fix skipped: {exc}", flush=True)


def _sdata_summary(sdata) -> str:
    n_tx = 0
    try:
        for k in sdata.points:
            n_tx += len(sdata.points[k])
    except Exception:
        pass
    images = list(sdata.images.keys())
    return f"{n_tx:,} transcripts | images: {', '.join(images) or 'none'}"


def _save_sdata_to_disk(save_path: str, seg_source: str = "xenium", roi_only: bool = False) -> None:
    """Background thread: write current SpatialData + analysis results to disk."""
    def _set(status, msg):
        with _save_sdata_lock:
            _save_sdata_state["status"]  = status
            _save_sdata_state["message"] = msg

    try:
        import spatialdata as sd
        tool = _seg_tool(seg_source)

        # ── Load sdata + cells_df from the active segmentation ─────────────────
        if tool in ("baysor", "proseg"):
            _lock = _baysor_lock if tool == "baysor" else _proseg_lock
            _state = _baysor_state if tool == "baysor" else _proseg_state
            with _lock:
                _alt_res = _state.get("result")
            if not _alt_res:
                _set("error", f"No {tool.capitalize()} result loaded. Select and load the run first.")
                return
            sdata_path = _alt_res.get("sdata_path", "")
            if not sdata_path or not os.path.isdir(sdata_path):
                _set("error", f"No SpatialData found for {tool}. Run resegmentation first.")
                return
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sdata = sd.read_zarr(sdata_path)
            cells_df  = _alt_res["cells_df"]
            cluster_methods = [c for c in cells_df.columns if c.startswith("cluster")]
            annot_key = f"labels_{tool}"   # e.g. "labels_baysor"
        else:
            with _sdata_lock:
                sdata = _sdata_state.get("sdata")
            if sdata is None:
                _set("error", "No SpatialData loaded. Open 'Resegment Cells' first to build it.")
                return
            cells_df        = DATA.get("df")
            cluster_methods = list(DATA.get("cluster_methods", {}).keys())
            annot_key       = None   # loop over _ANNOT_METHODS keys directly

        _set("running", f"Writing to {save_path}…")
        print(f"  save_spatialdata: writing to {save_path} (seg={seg_source}, roi_only={roi_only})…", flush=True)

        # ── Determine ROI cell-ID filter ───────────────────────────────────────
        roi_ids = None
        if roi_only and cells_df is not None:
            with _roi_lock:
                rois = list(_roi_state["rois"])
            if rois:
                roi_cols = [c for c in cells_df.columns if c.startswith("roi_")]
                if roi_cols:
                    mask    = cells_df[roi_cols].notna().any(axis=1)
                    roi_ids = set(cells_df[mask].index.astype(str))
                    print(f"  save_spatialdata: ROI filter — {len(roi_ids):,} of {len(cells_df):,} cells", flush=True)
                else:
                    print("  save_spatialdata: roi_only requested but no ROI columns found — exporting all cells", flush=True)

        # ── Attach analysis results to sdata table ─────────────────────────────
        if "table" in sdata.tables:
            tbl = sdata.tables["table"]
            has_cell_id   = "cell_id" in tbl.obs.columns
            zarr_cell_ids = tbl.obs["cell_id"].astype(str) if has_cell_id else None

            def _map(series: "pd.Series") -> list:
                """Align a Series (index=cell_id str) to zarr obs rows."""
                series = series.astype(str)
                series.index = series.index.astype(str)
                if has_cell_id:
                    return zarr_cell_ids.map(series.to_dict()).fillna("Unknown").tolist()
                return series.reindex(tbl.obs.index.astype(str)).fillna("Unknown").tolist()

            if cells_df is not None:
                # Cluster columns
                for method in cluster_methods:
                    if method in cells_df.columns:
                        tbl.obs[method] = _map(cells_df[method])

            # Cell-type annotations
            with _annot_lock:
                if annot_key:
                    # Reseg: single combined label
                    lbl = _annot_state.get(annot_key)
                    if lbl is not None:
                        tbl.obs["cell_type"] = _map(lbl)
                else:
                    # Xenium: per-method labels
                    for m in _ANNOT_METHODS:
                        lbl = _annot_state.get(f"labels_{m}")
                        if lbl is not None:
                            tbl.obs[f"cell_type_{m}"] = _map(lbl)
                    # RCTD weights → obsm
                    rctd_w = _annot_state.get("rctd_weights_labels_rctd")
                    if rctd_w is not None:
                        if has_cell_id:
                            aligned = rctd_w.reindex(zarr_cell_ids.values).fillna(0.0)
                        else:
                            aligned = rctd_w.reindex(tbl.obs.index.astype(str)).fillna(0.0)
                        tbl.obsm["rctd_weights"] = aligned.values
                        tbl.uns["rctd_weight_columns"] = list(rctd_w.columns)

            # ROI filter: subset table rows
            if roi_ids is not None:
                if has_cell_id:
                    keep_mask = zarr_cell_ids.isin(roi_ids).values
                else:
                    keep_mask = tbl.obs.index.astype(str).isin(roi_ids)
                tbl = tbl[keep_mask].copy()
                print(f"  save_spatialdata: table filtered to {tbl.n_obs:,} cells", flush=True)
            sdata.tables["table"] = tbl

        # ── Filter shapes to ROI cells ─────────────────────────────────────────
        out_shapes = {}
        for name, gdf in sdata.shapes.items():
            if roi_ids is not None:
                idx_str = gdf.index.astype(str)
                keep    = idx_str.isin(roi_ids)
                if keep.any():
                    out_shapes[name] = gdf[keep]
            else:
                out_shapes[name] = gdf

        # ── Write output (table + shapes only; images stay in original Xenium dir) ──
        out_tbl = sdata.tables.get("table")
        sdata_out = sd.SpatialData(
            shapes=out_shapes,
            tables={"table": out_tbl} if out_tbl is not None else {},
        )
        sdata_out.write(save_path, overwrite=True)
        n_cells = out_tbl.n_obs if out_tbl is not None else 0
        print(f"  save_spatialdata: done → {save_path} ({n_cells:,} cells)", flush=True)
        _set("done", f"Saved to {save_path}")
    except Exception as exc:
        import traceback; traceback.print_exc()
        _set("error", str(exc)[:300])


def _sdata_autoload() -> None:
    """If a Zarr cache exists for this dataset, load it into _sdata_state at startup."""
    cache = _sdata_cache_path()
    if not os.path.isdir(cache):
        return
    try:
        import spatialdata
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sdata = spatialdata.read_zarr(cache)
        _sdata_fix_categories(sdata)
        summary = _sdata_summary(sdata)
        with _sdata_lock:
            _sdata_state["status"]  = "done"
            _sdata_state["message"] = f"Auto-loaded (cached) — {summary}"
            _sdata_state["sdata"]   = sdata
        DATA["sdata"] = sdata
        print(f"  SpatialData: auto-loaded from cache — {summary}", flush=True)
        # Load any cell-type annotations stored in zarr tbl.obs into _annot_state
        _annot_load_from_sdata(sdata)
    except Exception as exc:
        print(f"  SpatialData: auto-load failed: {exc}", flush=True)


def _annot_load_from_sdata(sdata) -> None:
    """Load cell-type annotation columns from a SpatialData table into _annot_state."""
    try:
        if "table" not in sdata.tables:
            return
        tbl = sdata.tables["table"]
        obs = tbl.obs
        loaded = []
        # If obs has cell_id column, re-index labels by cell_id (integer Xenium IDs)
        # so they match DATA["df"].index when looked up in _cell_type_traces
        id_index = obs["cell_id"].astype(str) if "cell_id" in obs.columns else None
        for m in _ANNOT_METHODS:
            col = f"cell_type_{m}"
            if col in obs.columns:
                labels = obs[col].astype(str).copy()
                if id_index is not None:
                    labels.index = id_index
                else:
                    labels.index = labels.index.astype(str)
                # Only load if parquet cache didn't already populate this key
                with _annot_lock:
                    if _annot_state.get(f"labels_{m}") is None:
                        _annot_state[f"labels_{m}"] = labels
                        _annot_state["status"]  = "done"
                        _annot_state["message"] = f"Auto-loaded ({_ANNOT_METHODS[m]}) from zarr"
                        loaded.append(m)
        if loaded:
            print(f"  Annotation: loaded from zarr — {', '.join(loaded)}", flush=True)
        # Also load RCTD weights if present
        if "rctd_weights" in tbl.obsm and "rctd_weight_columns" in tbl.uns:
            with _annot_lock:
                if _annot_state.get("rctd_weights_labels_rctd") is None:
                    cols     = list(tbl.uns["rctd_weight_columns"])
                    w_index  = id_index if id_index is not None else obs.index.astype(str)
                    w        = pd.DataFrame(tbl.obsm["rctd_weights"],
                                            index=w_index, columns=cols)
                    _annot_state["rctd_weights_labels_rctd"] = w
                    print(f"  Annotation: loaded RCTD weights from zarr ({len(cols)} types)", flush=True)
    except Exception as exc:
        print(f"  Annotation: zarr load failed: {exc}", flush=True)


# ─── Cache utilities ───────────────────────────────────────────────────────────

def _cache_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".xenium_explorer_cache")


def _cache_size_str() -> str:
    """Return human-readable total size of the cache directory."""
    cache = _cache_dir()
    if not os.path.isdir(cache):
        return "0 B"
    total = 0
    for root, dirs, files in os.walk(cache):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if total < 1024 or unit == "TB":
            return f"{total:.1f} {unit}" if unit != "B" else f"{total} B"
        total /= 1024
    return f"{total:.1f} TB"


def _format_seg_label(tool: str, params: dict) -> str:
    """Format a human-readable label for a cached segmentation run."""
    if tool == "baysor":
        parts = [f"r={params.get('scale', '?')}µm", f"min={params.get('min_mol', '?')}tx"]
        if params.get("use_prior"):
            parts.append(f"prior={params.get('prior_conf', '?')}")
        x_min, x_max = params.get("x_min"), params.get("x_max")
        y_min, y_max = params.get("y_min"), params.get("y_max")
        if any(v is not None for v in [x_min, x_max, y_min, y_max]):
            parts.append(f"x={x_min}-{x_max} y={y_min}-{y_max}µm")
        else:
            parts.append("full slide")
        n = params.get("n_cells")
        if n:
            parts.append(f"{n:,} cells")
        return "Baysor: " + ", ".join(parts)
    else:
        parts = []
        vs = params.get("voxel_size")
        parts.append(f"voxel={'auto' if not vs else str(vs)+'µm'}")
        x_min, x_max = params.get("x_min"), params.get("x_max")
        y_min, y_max = params.get("y_min"), params.get("y_max")
        if any(v is not None for v in [x_min, x_max, y_min, y_max]):
            parts.append(f"x={x_min}-{x_max} y={y_min}-{y_max}µm")
        else:
            parts.append("full slide")
        n = params.get("n_cells")
        if n:
            parts.append(f"{n:,} cells")
        return "Proseg: " + ", ".join(parts)


def _list_cached_seg_runs() -> list:
    """Return list of {label, value} dicts for all complete cached Baysor/Proseg runs for the current dataset."""
    cache_base = os.path.join(os.path.expanduser("~"), ".xenium_explorer_cache")
    dataset = os.path.basename(DATA.get("data_dir", ""))
    opts = []
    if not os.path.isdir(cache_base):
        return opts
    for entry in sorted(os.listdir(cache_base)):
        out_dir = os.path.join(cache_base, entry)
        if not os.path.isdir(out_dir):
            continue
        # Determine tool
        if entry.startswith(f"baysor_{dataset}_"):
            tool = "baysor"
            param_tag = entry[len(f"baysor_{dataset}_"):]
            # Check completeness: any segmentation.csv anywhere in the tree
            complete = any(
                "segmentation.csv" in files
                for _, _, files in os.walk(out_dir)
            )
        elif entry.startswith(f"proseg_{dataset}_"):
            tool = "proseg"
            param_tag = entry[len(f"proseg_{dataset}_"):]
            complete = any(
                "cell-metadata.csv.gz" in files
                for _, _, files in os.walk(out_dir)
            )
        else:
            continue
        if not complete:
            continue
        params_path = os.path.join(out_dir, "params.json")
        if os.path.exists(params_path):
            with open(params_path) as f:
                params = json.load(f)
            label = _format_seg_label(tool, params)
        else:
            label = f"{tool.capitalize()}: [{param_tag}]"
        opts.append({"label": label, "value": f"{tool}:{param_tag}", "out_dir": out_dir, "param_tag": param_tag})
    return opts


# ─── ROI helper functions ──────────────────────────────────────────────────────

_ROI_PALETTE = [
    "#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
    "#1abc9c","#e67e22","#34495e","#e91e63","#00bcd4","#cddc39","#ff5722",
]

def _roi_color(index: int) -> str:
    return _ROI_PALETTE[index % len(_ROI_PALETTE)]


def _roi_cache_path() -> str:
    cache_dir = os.path.join(os.path.expanduser("~"), ".xenium_explorer_cache")
    os.makedirs(cache_dir, exist_ok=True)
    tag = hashlib.md5(DATA["data_dir"].encode()).hexdigest()[:8]
    return os.path.join(cache_dir, f"rois_{tag}.json")


def _roi_save_cache() -> None:
    """Serialise _roi_state["rois"] to JSON. Call inside _roi_lock."""
    try:
        with open(_roi_cache_path(), "w") as f:
            json.dump(_roi_state["rois"], f)
    except Exception as e:
        print(f"  ROI cache write error: {e}", flush=True)


def _roi_load_cache() -> list:
    try:
        path = _roi_cache_path()
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


def _xy_to_svg_path(xy: list) -> str:
    if not xy:
        return ""
    parts = [f"M {xy[0][0]},{xy[0][1]}"]
    for x, y in xy[1:]:
        parts.append(f"L {x},{y}")
    parts.append("Z")
    return " ".join(parts)


def _svg_sector(cx, cy, r, theta1, theta2):
    """SVG path string for a filled wedge in data coordinates."""
    import math
    large = 1 if (theta2 - theta1) > math.pi else 0
    x1 = cx + r * math.cos(theta1)
    y1 = cy + r * math.sin(theta1)
    x2 = cx + r * math.cos(theta2)
    y2 = cy + r * math.sin(theta2)
    if abs(theta2 - theta1 - 2 * math.pi) < 1e-6:   # full circle edge case
        x2 = cx + r * math.cos(theta1 + 2 * math.pi - 1e-6)
        y2 = cy + r * math.sin(theta1 + 2 * math.pi - 1e-6)
    return f"M {cx},{cy} L {x1},{y1} A {r},{r},0,{large},1,{x2},{y2} Z"


def _build_rctd_pie_shapes(df, weights_df, ct_colors, radius_um=RCTD_PIE_RADIUS_UM):
    """Build Plotly layout shapes (SVG wedges) for RCTD weight pie charts.

    Parameters
    ----------
    df : DataFrame  — cells with x_centroid, y_centroid (y already negated on caller side)
    weights_df : DataFrame  — (n_cells, n_types), index = str cell IDs, cols = cell type names
    ct_colors : dict  — {cell_type: hex_color}
    radius_um : float  — wedge radius in µm (data coordinates)
    """
    import math
    shapes = []
    # Align weights to df; use str index
    idx_str = df.index.astype(str)
    w = weights_df.reindex(idx_str).fillna(0).values.astype(float)
    xs = df["x_centroid"].values
    ys = -df["y_centroid"].values      # negated, same as spatial plot y-axis
    col_names = list(weights_df.columns)

    for i in range(len(df)):
        row = w[i]
        total = row.sum()
        if total <= 0:
            continue
        row = row / total  # normalize to sum=1
        theta = -math.pi / 2   # start at top (12 o'clock)
        cx, cy = float(xs[i]), float(ys[i])
        for j, ct in enumerate(col_names):
            frac = row[j]
            if frac < 0.02:
                theta += frac * 2 * math.pi
                continue
            dtheta = frac * 2 * math.pi
            color = ct_colors.get(ct, "#888888")
            path = _svg_sector(cx, cy, radius_um, theta, theta + dtheta)
            shapes.append({
                "type": "path", "path": path,
                "fillcolor": color, "opacity": 0.85,
                "line": {"width": 0},
                "xref": "x", "yref": "y", "layer": "above",
            })
            theta += dtheta
    return shapes


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _roi_compute_hull(cell_ids, src_df=None):
    """Return convex hull exterior coords [[x,y],...] for the given cell_ids, or None."""
    try:
        from shapely.geometry import MultiPoint
        df = src_df if src_df is not None else DATA["df"]
        sub = df.loc[[c for c in cell_ids if c in df.index]]
        if len(sub) < 3:
            return None
        pts = list(zip(sub["x_centroid"].values, -sub["y_centroid"].values))
        hull = MultiPoint(pts).convex_hull
        if hull.geom_type == "Polygon":
            return [list(c) for c in hull.exterior.coords]
        elif hull.geom_type == "LineString":
            return [list(c) for c in hull.coords]
        return None
    except Exception as e:
        print(f"  ROI hull error: {e}", flush=True)
        return None


def _roi_cells_in_polygon(polygon_xy: list, src_df=None) -> list:
    """Return cell IDs (index values) whose centroids lie inside polygon_xy."""
    try:
        from shapely.geometry import Polygon as _SPoly, Point as _SPoint
        poly = _SPoly(polygon_xy)
        df = src_df if src_df is not None else DATA["df"]
        xs = df["x_centroid"].values
        ys = -df["y_centroid"].values
        ids = df.index.tolist()
        return [cid for cid, px, py in zip(ids, xs, ys) if poly.contains(_SPoint(px, py))]
    except Exception as e:
        print(f"  ROI containment error: {e}", flush=True)
        return []


def _roi_apply_metadata_to_df(df: "pd.DataFrame", rois: list) -> None:
    """Add/update roi_{cls} columns in df based on rois list."""
    # Reset ALL existing roi_* columns first (handles deletions of last ROI in a class)
    for col in [c for c in df.columns if c.startswith("roi_")]:
        df[col] = pd.NA
    # Ensure columns exist for all current classes
    for cls in {r["cls"] for r in rois}:
        col = f"roi_{cls}"
        if col not in df.columns:
            df[col] = pd.NA

    overlap_warned: set = set()
    for roi in rois:
        col = f"roi_{roi['cls']}"
        cell_ids = _roi_cells_in_polygon(roi["polygon_xy"], src_df=df)
        # Warn on overlaps with existing same-class assignments
        existing = df.loc[[c for c in cell_ids if c in df.index], col]
        overlapping = existing.dropna()
        for existing_name in overlapping.unique():
            key = (existing_name, roi["name"])
            if key not in overlap_warned:
                print(f"  ROI WARNING: '{roi['name']}' overlaps with '{existing_name}' (class '{roi['cls']}')", flush=True)
                overlap_warned.add(key)
        df.loc[[c for c in cell_ids if c in df.index], col] = roi["name"]


def _roi_write_all_zarrs_bg() -> None:
    """Daemon thread: write ROI metadata columns to all zarr stores."""
    with _roi_lock:
        rois = list(_roi_state["rois"])
    # Write to Xenium zarr
    try:
        sdata_path = DATA.get("sdata_path", "")
        if sdata_path and os.path.isdir(sdata_path):
            _roi_apply_metadata_to_df(DATA["df"], rois)
            _update_reseg_zarr_obs(sdata_path, DATA["df"])
            print(f"  ROI: wrote {len(rois)} ROIs to xenium zarr", flush=True)
    except Exception as e:
        print(f"  ROI zarr write error (xenium): {e}", flush=True)
    # Write to Baysor/Proseg zarrs
    for run in _list_cached_seg_runs():
        try:
            tool = run["value"].split(":")[0]
            if tool == "baysor":
                with _baysor_lock:
                    result = _baysor_state.get("result")
            else:
                with _proseg_lock:
                    result = _proseg_state.get("result")
            if result and result.get("sdata_path") and os.path.isdir(result["sdata_path"]):
                _roi_apply_metadata_to_df(result["cells_df"], rois)
                _update_reseg_zarr_obs(result["sdata_path"], result["cells_df"])
                print(f"  ROI: wrote {len(rois)} ROIs to {tool} zarr", flush=True)
        except Exception as e:
            print(f"  ROI zarr write error ({run['value']}): {e}", flush=True)


def _roi_shapes_for_fig(rois: list, pending_hull=None) -> list:
    """Return list of Plotly shape dicts for ROI overlays."""
    shapes = []
    for r in rois:
        shapes.append({
            "type": "path",
            "path": _xy_to_svg_path(r["polygon_xy"]),
            "fillcolor": _hex_to_rgba(r["color"], 0.18),
            "line": {"color": r["color"], "width": 1.5},
            "layer": "above", "xref": "x", "yref": "y",
        })
    if pending_hull:
        shapes.append({
            "type": "path",
            "path": _xy_to_svg_path(pending_hull),
            "fillcolor": "rgba(200,200,200,0.2)",
            "line": {"color": "#aaaaaa", "width": 1.5, "dash": "dot"},
            "layer": "above", "xref": "x", "yref": "y",
        })
    return shapes


def _auto_load_rois() -> None:
    """Load ROIs from cache at startup and apply metadata to DATA['df']."""
    rois = _roi_load_cache()
    with _roi_lock:
        _roi_state["rois"] = rois
    if rois and "df" in DATA:
        _roi_apply_metadata_to_df(DATA["df"], rois)
        print(f"  ROI: auto-loaded {len(rois)} ROIs from cache", flush=True)


_auto_load_rois()


def _clean_cache_for_dataset() -> str:
    """Delete all cache entries that belong to the current dataset. Returns summary string."""
    cache = _cache_dir()
    if not os.path.isdir(cache):
        return "Cache directory not found."
    tag = hashlib.md5(DATA["data_dir"].encode()).hexdigest()[:8]
    dataset_name = os.path.basename(DATA["data_dir"])
    removed = 0
    import shutil
    for entry in os.listdir(cache):
        full = os.path.join(cache, entry)
        # Match files/dirs that contain the dataset tag or dataset name
        if tag in entry or dataset_name in entry:
            try:
                if os.path.isdir(full):
                    shutil.rmtree(full)
                else:
                    os.remove(full)
                removed += 1
            except Exception as e:
                print(f"  clean_cache: could not remove {entry}: {e}", flush=True)
    print(f"  clean_cache: removed {removed} cache entries for dataset {dataset_name}", flush=True)
    return f"Removed {removed} cache entries for {dataset_name}."


def _available_samples() -> list:
    """Scan CWD and parent for output-XETG* directories containing experiment.xenium."""
    candidates = []
    search_dirs = [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
    seen = set()
    for base in search_dirs:
        try:
            entries = sorted(os.listdir(base))
        except Exception:
            continue
        for entry in entries:
            full = os.path.join(base, entry)
            if (entry.startswith("output-") and os.path.isdir(full)
                    and os.path.exists(os.path.join(full, "experiment.xenium"))
                    and full not in seen):
                seen.add(full)
                candidates.append({"label": entry, "value": full})
    return candidates


def to_spatialdata(qv_threshold: int = 20, force: bool = False) -> None:
    """Convert Xenium data to SpatialData format using sopa.io.xenium().

    The result is cached to ~/.xenium_explorer_cache/spatialdata_<hash>.zarr
    and auto-loaded on subsequent startups.

    Callable from REPL:
        to_spatialdata()                  # use cache if available
        to_spatialdata(force=True)        # re-read from raw files and overwrite cache
        to_spatialdata(qv_threshold=30)
    """
    global _sdata_version

    def _run():
        global _sdata_version

        def _set(status, message, sdata=None):
            with _sdata_lock:
                _sdata_state["status"]  = status
                _sdata_state["message"] = message
                if sdata is not None:
                    _sdata_state["sdata"] = sdata
                    DATA["sdata"] = sdata

        try:
            _set("running", "Importing sopa…")
            try:
                import sopa, sopa.io, spatialdata
            except ImportError as e:
                _set("error", f"Missing package: {e}. Run: pip install sopa")
                return

            cache = _sdata_cache_path()

            # ── Try cache first ───────────────────────────────────────────
            if not force and os.path.isdir(cache):
                _set("running", "Loading from cache…")
                print(f"  to_spatialdata: loading from cache {cache} …", flush=True)
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sdata = spatialdata.read_zarr(cache)
                _sdata_fix_categories(sdata)
                summary = _sdata_summary(sdata)
                msg = f"Done (cached) — {summary}"
                print(f"  to_spatialdata: {msg}", flush=True)
                _sdata_version += 1
                _set("done", msg, sdata=sdata)
                return

            # ── Read from raw Xenium files ────────────────────────────────
            _set("running", f"Reading Xenium data (qv≥{qv_threshold})…")
            print(f"  to_spatialdata: reading from {DATA['data_dir']} …", flush=True)
            import warnings, logging
            # Suppress spatialdata logging warnings (uses Python logging, not warnings module)
            _sd_loggers = [logging.getLogger(n) for n in ("spatialdata", "spatialdata._logging")]
            _sd_prev_levels = [lg.level for lg in _sd_loggers]
            for lg in _sd_loggers:
                lg.setLevel(logging.ERROR)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*feature_key.*categorical.*")
                    warnings.filterwarnings("ignore", message=".*OME series.*")
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    sdata = sopa.io.xenium(DATA["data_dir"], qv_threshold=qv_threshold)
            finally:
                for lg, lvl in zip(_sd_loggers, _sd_prev_levels):
                    lg.setLevel(lvl)

            _sdata_fix_categories(sdata)

            # ── Save to Zarr cache ────────────────────────────────────────
            _set("running", "Saving to cache…")
            print(f"  to_spatialdata: saving to {cache} …", flush=True)
            try:
                import shutil
                if os.path.isdir(cache):
                    shutil.rmtree(cache)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    sdata.write(cache)
                print(f"  to_spatialdata: cache saved ✓", flush=True)
            except Exception as save_exc:
                print(f"  to_spatialdata: cache save failed: {save_exc}", flush=True)

            summary = _sdata_summary(sdata)
            msg = f"Done — {summary}"
            print(f"  to_spatialdata: {msg}", flush=True)
            _sdata_version += 1
            _set("done", msg, sdata=sdata)

        except Exception as exc:
            import traceback
            traceback.print_exc()
            _set("error", str(exc)[:300])

    threading.Thread(target=_run, daemon=True).start()
    print("  to_spatialdata: started in background…", flush=True)


def clear_sdata_cache() -> None:
    """Delete the SpatialData Zarr cache for this dataset (callable from REPL)."""
    import shutil
    cache = _sdata_cache_path()
    if os.path.isdir(cache):
        shutil.rmtree(cache)
        print(f"  SpatialData cache deleted: {cache}", flush=True)
    else:
        print(f"  SpatialData cache not found: {cache}", flush=True)


# Auto-load SpatialData cache if it exists for this dataset
_sdata_autoload()


def segment_tissue_roi() -> None:
    """Run sopa.segmentation.tissue() to detect tissue ROI in the SpatialData object.
    Callable from REPL. Requires to_spatialdata() to have completed first.
    """
    global _sdata_version

    def _run():
        global _sdata_version
        with _sdata_lock:
            sdata = _sdata_state.get("sdata")
        if sdata is None:
            print("  segment_tissue_roi: no SpatialData. Run to_spatialdata() first.", flush=True)
            return
        try:
            import sopa.segmentation
            with _sdata_lock:
                _sdata_state["status"]  = "running"
                _sdata_state["message"] = "Segmenting tissue ROI…"
            print("  segment_tissue_roi: running sopa.segmentation.tissue()…", flush=True)
            sopa.segmentation.tissue(sdata, mode="staining")
            # Find ROI shapes key (default key_added='region_of_interest')
            roi_key = next(
                (k for k in sdata.shapes
                 if k == "region_of_interest" or "roi" in k.lower() or "tissue" in k.lower()),
                None
            )
            if roi_key:
                roi_shapes = sdata.shapes[roi_key]
                with _sdata_lock:
                    _sdata_state["roi"]     = roi_shapes
                    _sdata_state["status"]  = "roi_done"
                    _sdata_state["message"] = f"ROI done — {len(roi_shapes)} region(s)"
                _sdata_version += 1
                print(f"  segment_tissue_roi: done, {len(roi_shapes)} region(s)", flush=True)
            else:
                print("  segment_tissue_roi: no ROI shapes found in sdata", flush=True)
                with _sdata_lock:
                    _sdata_state["message"] = "No ROI shapes found after tissue segmentation"
        except Exception as exc:
            import traceback
            traceback.print_exc()
            with _sdata_lock:
                _sdata_state["status"]  = "error"
                _sdata_state["message"] = str(exc)[:300]

    threading.Thread(target=_run, daemon=True).start()


def create_patches(width_um: float = 1000.0, min_transcripts: int = 10) -> None:
    """Create transcript patches using sopa.make_transcript_patches().
    Callable from REPL. Requires to_spatialdata() to have completed first.
    """
    global _sdata_version

    def _run():
        global _sdata_version
        with _sdata_lock:
            sdata = _sdata_state.get("sdata")
        if sdata is None:
            print("  create_patches: no SpatialData. Run to_spatialdata() first.", flush=True)
            return
        try:
            import sopa
            with _sdata_lock:
                _sdata_state["status"]  = "running"
                _sdata_state["message"] = "Creating transcript patches…"
            print(f"  create_patches: width={width_um}µm, min_tx={min_transcripts}…", flush=True)
            sopa.make_transcript_patches(
                sdata,
                patch_width=width_um,
                min_points_per_patch=min_transcripts,
            )
            # Find patches key (default key_added='transcripts_patches')
            patch_key = next(
                (k for k in sdata.shapes
                 if k == "transcripts_patches" or "patch" in k.lower()),
                None
            )
            if patch_key:
                patches = sdata.shapes[patch_key]
                with _sdata_lock:
                    _sdata_state["patches"] = patches
                    _sdata_state["status"]  = "patches_done"
                    _sdata_state["message"] = f"Patches done — {len(patches)} patches"
                _sdata_version += 1
                print(f"  create_patches: {len(patches)} patches created", flush=True)
            else:
                print("  create_patches: no patch shapes found in sdata", flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            with _sdata_lock:
                _sdata_state["status"]  = "error"
                _sdata_state["message"] = str(exc)[:300]

    threading.Thread(target=_run, daemon=True).start()


def _sdata_transform_to_um(gdf):
    """Transform a SpatialData GeoDataFrame from its intrinsic coordinate system
    to µm by applying the registered transform to the global coordinate system
    (morphology image pixels) and then scaling by PIXEL_SIZE_UM.

    Returns a new GeoDataFrame with coordinates in µm.
    """
    try:
        import shapely.affinity
        import geopandas as gpd
        from spatialdata.transformations import get_transformation
        t = get_transformation(gdf, to_coordinate_system="global")
        # Get the 3×3 affine matrix [x, y, 1] (row=output, col=input)
        M = t.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        # M transforms intrinsic coords → global pixel coords
        # Compose with PIXEL_SIZE_UM to go directly to µm
        sx = M[0, 0] * PIXEL_SIZE_UM
        sy = M[1, 1] * PIXEL_SIZE_UM
        tx = M[0, 2] * PIXEL_SIZE_UM
        ty = M[1, 2] * PIXEL_SIZE_UM
        geoms_um = [
            shapely.affinity.affine_transform(g, [sx, M[0, 1] * PIXEL_SIZE_UM,
                                                   M[1, 0] * PIXEL_SIZE_UM, sy,
                                                   tx, ty])
            for g in gdf.geometry
        ]
        return gpd.GeoDataFrame(geometry=geoms_um)
    except Exception as exc:
        print(f"  _sdata_transform_to_um failed: {exc}, using raw coordinates", flush=True)
        return gdf


def _sdata_overlays_to_shapes(show_roi: bool, show_patches: bool) -> list:
    """Extract ROI polygon and/or patch grid as Plotly layout shapes (in µm)."""
    shapes = []
    with _sdata_lock:
        roi     = _sdata_state.get("roi")
        patches = _sdata_state.get("patches")

    if show_roi and roi is not None:
        try:
            roi_um = _sdata_transform_to_um(roi)
            for geom in roi_um.geometry:
                coords = list(geom.exterior.coords)
                xs = [c[0] for c in coords]
                ys = [-c[1] for c in coords]  # negate Y for plot space
                shapes.append(dict(
                    type="path",
                    path=" ".join(
                        f"{'M' if i == 0 else 'L'}{x},{y}"
                        for i, (x, y) in enumerate(zip(xs, ys))
                    ) + " Z",
                    line=dict(color="#52b788", width=2),
                    fillcolor="rgba(82,183,136,0.05)",
                    layer="above",
                ))
        except Exception as exc:
            print(f"  ROI overlay error: {exc}", flush=True)

    if show_patches and patches is not None:
        try:
            patches_um = _sdata_transform_to_um(patches)
            for geom in patches_um.geometry:
                b = geom.bounds  # (minx, miny, maxx, maxy) in µm
                shapes.append(dict(
                    type="rect",
                    x0=b[0], y0=-b[3], x1=b[2], y1=-b[1],  # negate Y
                    line=dict(color="rgba(255,170,0,0.6)", width=1),
                    fillcolor="rgba(255,170,0,0.03)",
                    layer="above",
                ))
        except Exception as exc:
            print(f"  Patch overlay error: {exc}", flush=True)

    return shapes


with _spage_lock:
    _spage_autoload_result = _spage_state.get("result")
    _spage_autoload_genes  = _spage_state.get("result_genes") or []

_base_gene_options = [{"label": g, "value": g} for g in _sorted_gene_names]
_imp_gene_list = (sorted(_spage_autoload_result.columns) if _spage_autoload_result is not None
                  else sorted(_spage_autoload_genes))
if _imp_gene_list:
    _imp_opts    = [{"label": f"{g} [imp]", "value": f"{g} [imp]"} for g in _imp_gene_list]
    gene_options = _base_gene_options + _imp_opts
else:
    gene_options = _base_gene_options

sidebar = html.Div([
    # Header + collapsible tissue info (rebuilt dynamically on dataset change)
    html.Div(id="tissue-info-content"),

    html.Hr(style={"borderColor": BORDER, "margin": "0 0 14px 0"}),

    # Subset indicator (hidden when no subset is active)
    html.Div(id="subset-indicator", style={"display": "none"},
             children=[
                 html.Div([
                     html.Span("⬡ SUBSET ACTIVE", style={
                         "fontSize": "10px", "fontWeight": "700", "letterSpacing": "1px",
                         "color": "#f0a500",
                     }),
                     html.Span(id="subset-count", style={
                         "fontSize": "10px", "color": MUTED, "marginLeft": "6px",
                     }),
                 ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"}),
                 html.Button("✕ Clear Subset", id="unsubset-btn", n_clicks=0, style={
                     "width": "100%", "padding": "3px 0", "marginTop": "4px",
                     "backgroundColor": CARD_BG, "color": "#f0a500",
                     "border": "1px solid #f0a500", "borderRadius": "4px",
                     "cursor": "pointer", "fontSize": "11px",
                 }),
                 html.Hr(style={"borderColor": BORDER, "margin": "10px 0 14px 0"}),
             ]),

    ctrl_label("Segmentation Source"),
    dcc.Dropdown(
        id="seg-source",
        options=[
            {"label": "Xenium (original)", "value": "xenium"},
            {"label": "Baysor",            "value": "baysor", "disabled": True},
            {"label": "Proseg",            "value": "proseg", "disabled": True},
        ],
        value="xenium",
        clearable=False,
        style={"fontSize": "12px", "marginBottom": "6px"},
    ),
    html.Div(
        html.Button(
            "🗑 Delete this run",
            id="seg-delete-btn", n_clicks=0,
            style={
                "width": "100%", "padding": "4px 0",
                "backgroundColor": "transparent", "color": "#da3633",
                "border": "1px solid #da3633", "borderRadius": "4px",
                "cursor": "pointer", "fontSize": "11px",
            },
        ),
        id="seg-delete-div",
        style={"display": "none", "marginBottom": "14px"},
    ),

    html.Hr(style={"borderColor": BORDER, "margin": "0 0 14px 0"}),

    # Color by
    ctrl_label("Color By"),
    dcc.Dropdown(
        id="color-by",
        options=[
            {"label": "Cluster",                    "value": "cluster"},
            {"label": "Gene Expression",            "value": "gene"},
            {"label": "Cell Type (CellTypist)",     "value": "cell_type:celltypist", "disabled": True},
            {"label": "Cell Type (Seurat)",         "value": "cell_type:seurat",     "disabled": True},
            {"label": "Cell Type (RCTD)",           "value": "cell_type:rctd",       "disabled": True},
            {"label": "Transcript Counts",          "value": "transcript_counts"},
            {"label": "Cell Area",         "value": "cell_area"},
            {"label": "Nucleus Area",      "value": "nucleus_area"},
        ],
        value="cluster",
        clearable=False,
        style={"fontSize": "12px", "marginBottom": "14px"},
    ),

    html.Div(id="cluster-ctrl", children=[
        ctrl_label("Clustering Method"),
        dcc.Dropdown(
            id="cluster-method",
            options=cluster_options,
            value=cluster_methods[0] if cluster_methods else None,
            clearable=False, style={"fontSize": "12px"},
        ),
    ], style={"marginBottom": "14px"}),

    html.Div(id="gene-ctrl", children=[
        ctrl_label("Gene"),
        dcc.Dropdown(
            id="gene-selector",
            options=gene_options,
            value=_default_gene,
            clearable=False, placeholder="Search gene…",
            style={"fontSize": "12px"},
        ),
    ], style={"marginBottom": "14px", "display": "none"}),

    html.Hr(style={"borderColor": BORDER, "margin": "0 0 14px 0"}),

    # Boundary overlays
    ctrl_label("Overlays"),
    html.Div([
        dcc.Checklist(
            id="boundary-toggles",
            options=[
                {"label": html.Span(" Xenium Cell Boundaries",    style={"color": "#00d4ff", "fontSize": "13px"}),
                 "value": "cell"},
                {"label": html.Span(" Xenium Nuclei Boundaries", style={"color": "#ff9f43", "fontSize": "13px"}),
                 "value": "nucleus"},
                {"label": html.Span(" Proseg Boundaries",  style={"color": "#2ed573", "fontSize": "13px"}),
                 "value": "proseg", "disabled": True},
                {"label": html.Span(" Baysor Boundaries",  style={"color": "#a29bfe", "fontSize": "13px"}),
                 "value": "baysor", "disabled": True},
            ],
            value=[],
            inputStyle={"marginRight": "6px"},
            labelStyle={
                "display": "flex", "alignItems": "center",
                "marginBottom": "7px", "cursor": "pointer",
            },
        ),
        html.Div(
            f"Zoom in to see boundaries (limit {BOUNDARY_CELL_LIMIT:,} cells in viewport).",
            style={"fontSize": "10px", "color": MUTED, "marginTop": "4px", "lineHeight": "1.4"},
        ),
    ], style={"marginBottom": "10px"}),


    html.Hr(style={"borderColor": BORDER, "margin": "0 0 14px 0"}),

    # Point style
    ctrl_label("Point Size"),
    dcc.Slider(
        id="point-size", min=1, max=8, step=0.5, value=2,
        marks={1: "1", 4: "4", 8: "8"},
        tooltip={"placement": "bottom", "always_visible": False},
    ),
    html.Div(style={"marginBottom": "12px"}),

    ctrl_label("Opacity"),
    dcc.Slider(
        id="point-opacity", min=0.1, max=1.0, step=0.05, value=0.85,
        marks={0.1: "0.1", 0.5: "0.5", 1.0: "1.0"},
        tooltip={"placement": "bottom", "always_visible": False},
    ),

    html.Hr(style={"borderColor": BORDER, "margin": "14px 0"}),

    # ── Morphology image overlay ────────────────────────────────────────
    ctrl_label("Morphology Image"),
    html.Div(
        f"Loads tiles for viewports ≤{MORPH_MAX_UM:,} µm. Zoom in to activate.",
        style={"fontSize": "10px", "color": MUTED, "marginBottom": "8px", "lineHeight": "1.4"},
    ),
    dcc.Checklist(
        id="morph-enable",
        options=[{"label": html.Span(" Enable Image Overlay", style={"fontSize": "13px", "color": TEXT}),
                  "value": "show"}],
        value=[],
        inputStyle={"marginRight": "6px"},
        labelStyle={"display": "flex", "alignItems": "center", "cursor": "pointer"},
        style={"marginBottom": "8px"},
    ),
    html.Div(id="morph-controls", children=[
        ctrl_label("Z-Level"),
        dcc.Dropdown(
            id="morph-zlevel",
            options=[{"label": f"Z-level {i}", "value": i} for i in range(4)],
            value=0, clearable=False,
            style={"fontSize": "12px", "marginBottom": "8px"},
        ),
        ctrl_label("Channels"),
        dcc.Checklist(
            id="morph-channels",
            options=[
                {"label": html.Span(f" {c['label']}",
                                    style={"color": f"rgb{c['color']}", "fontSize": "12px"}),
                 "value": c["value"]}
                for c in MORPH_CHANNELS
            ],
            value=["dapi", "boundary"],
            inputStyle={"marginRight": "6px"},
            labelStyle={"display": "flex", "alignItems": "center",
                        "marginBottom": "5px", "cursor": "pointer"},
            style={"marginBottom": "8px"},
        ),
        ctrl_label("Brightness"),
        dcc.Slider(
            id="morph-brightness", min=0.5, max=8, step=0.25, value=2,
            marks={0.5: "0.5×", 4: "4×", 8: "8×"},
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        html.Div(style={"marginBottom": "10px"}),
        ctrl_label("Image Opacity"),
        dcc.Slider(
            id="morph-opacity", min=0.1, max=1.0, step=0.05, value=0.85,
            marks={0.1: "0.1", 0.5: "0.5", 1.0: "1.0"},
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        html.Div(style={"marginBottom": "8px"}),
    ], style={"display": "none"}),

    html.Hr(style={"borderColor": BORDER, "margin": "14px 0"}),

    # ── Cell type annotation ────────────────────────────────────────────
    ctrl_label("Cell Type Annotation"),
    html.Button(
        "Annotate Cells",
        id="annot-modal-open-btn",
        n_clicks=0,
        style={
            "width": "100%", "padding": "7px 0",
            "backgroundColor": "#238636", "color": "#fff",
            "border": "none", "borderRadius": "5px",
            "cursor": "pointer", "fontSize": "13px", "fontWeight": "600",
            "marginBottom": "8px",
        },
    ),
    html.Div(id="annot-status", style={"fontSize": "11px", "color": MUTED, "minHeight": "16px"}),
    dcc.Interval(id="annot-poll", interval=1000, disabled=True),

    html.Hr(style={"borderColor": BORDER, "margin": "14px 0"}),

    # ── Resegmentation ──────────────────────────────────────────────────
    ctrl_label("Resegmentation"),
    html.Button(
        "Resegment Cells",
        id="reseg-modal-open-btn",
        n_clicks=0,
        style={
            "width": "100%", "padding": "7px 0",
            "backgroundColor": "#1f6feb", "color": "#fff",
            "border": "none", "borderRadius": "5px",
            "cursor": "pointer", "fontSize": "13px", "fontWeight": "600",
            "marginBottom": "8px",
        },
    ),
    # Status line (shows whichever tool is running/done)
    html.Div(id="reseg-status", style={"fontSize": "11px", "color": MUTED, "minHeight": "16px", "marginBottom": "6px"}),
    # Hidden status divs for existing poll callbacks
    html.Div(id="baysor-status", style={"display": "none"}),
    html.Div(id="proseg-status", style={"display": "none"}),
    dcc.Interval(id="baysor-poll", interval=3000, disabled=True),
    dcc.Interval(id="proseg-poll", interval=3000, disabled=True),

    html.Hr(style={"borderColor": BORDER, "margin": "14px 0"}),

    # ── SpaGE gene imputation ───────────────────────────────────────────
    ctrl_label("SpaGE Gene Imputation"),
    html.Button(
        "Impute Genes",
        id="spage-modal-open-btn",
        n_clicks=0,
        style={
            "width": "100%", "padding": "7px 0",
            "backgroundColor": "#6f42c1", "color": "#fff",
            "border": "none", "borderRadius": "5px",
            "cursor": "pointer", "fontSize": "13px", "fontWeight": "600",
            "marginBottom": "8px",
        },
    ),
    html.Div(id="spage-status", style={"fontSize": "11px", "color": MUTED, "minHeight": "16px", "marginBottom": "6px"}),
    dcc.Interval(id="spage-poll", interval=3000, disabled=True),

    html.Hr(style={"borderColor": BORDER, "margin": "14px 0"}),
    ctrl_label("SPLIT Correction"),
    dbc.Button("✦ Correct Ambient RNA", id="split-modal-open-btn",
               style={"width":"100%","backgroundColor":"#e85d04","color":"white"}),
    html.Div(id="split-status",
             style={"fontSize":"0.75rem","marginTop":"4px","color":"#8b949e"}),
    dcc.Interval(id="split-poll", interval=3000, disabled=True),
    html.Hr(style={"borderColor": BORDER, "margin": "14px 0"}),
    ctrl_label("Active Counts"),
    dcc.RadioItems(
        id="counts-mode",
        options=[
            {"label": " Original",        "value": "original"},
            {"label": " SPLIT Corrected", "value": "corrected", "disabled": True},
        ],
        value="original",
        labelStyle={"display":"block","cursor":"pointer","marginBottom":"2px"},
        style={"fontSize":"0.82rem"},
    ),

    html.Hr(style={"borderColor": BORDER, "margin": "14px 0"}),
    ctrl_label("ROI Annotations"),
    dcc.Checklist(
        id="roi-show-toggle",
        options=[{"label": " Show ROIs", "value": "show"}],
        value=["show"],
        style={"fontSize": "0.82rem", "marginBottom": "6px"},
    ),
    html.Button(
        "Manage / Draw ROIs",
        id="roi-manage-btn",
        n_clicks=0,
        style={
            "width": "100%", "padding": "7px 0",
            "backgroundColor": "#5b2d8e", "color": "#fff",
            "border": "none", "borderRadius": "5px",
            "cursor": "pointer", "fontSize": "13px", "fontWeight": "600",
            "marginBottom": "6px",
        },
    ),
    html.Div(id="roi-sidebar-summary",
             style={"fontSize": "11px", "color": MUTED, "minHeight": "16px"}),

    html.Hr(style={"borderColor": BORDER, "margin": "14px 0"}),

    # UMAP toggle
    html.Button(
        "Show UMAP",
        id="umap-toggle",
        n_clicks=1,
        style={
            "width": "100%", "padding": "7px 0",
            "backgroundColor": CARD_BG, "color": TEXT,
            "border": f"1px solid {BORDER}", "borderRadius": "5px",
            "cursor": "pointer", "fontSize": "13px", "fontWeight": "600",
            "marginBottom": "6px",
        },
    ),
    html.Button(
        "Run UMAP on Reseg Cells",
        id="umap-reseg-btn",
        n_clicks=0,
        style={
            "width": "100%", "padding": "7px 0",
            "backgroundColor": CARD_BG, "color": TEXT,
            "border": f"1px solid {BORDER}", "borderRadius": "5px",
            "cursor": "pointer", "fontSize": "12px", "fontWeight": "600",
        },
    ),
    html.Div(id="umap-reseg-status",
             style={"fontSize": "11px", "color": MUTED, "minHeight": "14px",
                    "marginTop": "4px"}),
    dcc.Interval(id="umap-reseg-poll", interval=1500, disabled=True),
], style={
    "width": "240px", "minWidth": "240px",
    "backgroundColor": PANEL_BG, "borderRight": f"1px solid {BORDER}",
    "padding": "20px 16px", "overflowY": "auto",
    "height": "100vh", "boxSizing": "border-box",
})

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Xenium Explorer",
    suppress_callback_exceptions=True,
)

app.layout = html.Div([
    dcc.Store(id="selected-cell",    data=None),
    dcc.Store(id="spatial-relayout", data={}),
    dcc.Store(id="annot-done",       data=0),
    dcc.Store(id="baysor-done",      data=0),
    dcc.Store(id="proseg-done",      data=0),
    dcc.Store(id="reseg-patches-confirmed", data=0),
    dcc.Store(id="spage-done",       data=0),
    dcc.Store(id="dataset-version",        data=0),
    dcc.Store(id="extra-datasets-version", data=0),
    dcc.Store(id="cache-clean-done", data=0),
    dcc.Interval(id="spage-repl-poll", interval=4000, n_intervals=0),
    dcc.Store(id="subset-version",   data=0),
    dcc.Store(id="sdata-done",       data=0),
    dcc.Store(id="split-done",        data=0),
    dcc.Store(id="counts-mode-store", data="original"),
    dcc.Store(id="roi-done",          data=0),
    dcc.Store(id="roi-pending",       data=None),
    dcc.Interval(id="subset-poll",   interval=500,  n_intervals=0),
    dcc.Interval(id="morph-hires-poll", interval=400, n_intervals=0, disabled=True),
    # Fires once after 500 ms to guarantee initial plot render in Dash 4
    dcc.Interval(id="startup-trigger", interval=500, max_intervals=1),

    # ── Resegmentation modal ─────────────────────────────────────────────────────
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Resegment Cells"), close_button=True),
        dbc.ModalBody([
            # ── Patch Setup ──────────────────────────────────────────────────
            html.Div([
                html.Div("Step 1 — Prepare Patches", style={"fontSize": "13px", "fontWeight": "600", "color": TEXT, "marginBottom": "8px"}),
                html.Div(
                    "Set parameters then click Prepare Patches. The app will build a SpatialData object, "
                    "detect the tissue ROI, and partition it into patches. Review the overlays on the "
                    "spatial plot, then confirm before running segmentation.",
                    style={"fontSize": "10px", "color": MUTED, "marginBottom": "10px", "lineHeight": "1.4"},
                ),
                html.Div([
                    html.Div([
                        ctrl_label("QV Threshold"),
                        dcc.Slider(
                            id="sdata-qv", min=0, max=40, step=5, value=20,
                            marks={0: "0", 20: "20", 40: "40"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        html.Div(style={"marginBottom": "10px"}),
                    ], style={"flex": "1"}),
                    html.Div([
                        ctrl_label("Patch Width (µm)"),
                        dcc.Input(
                            id="sdata-patch-width", type="number", value=1000, min=100, max=10000,
                            debounce=True,
                            style={
                                "width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                                "border": f"1px solid {BORDER}", "borderRadius": "4px",
                                "padding": "4px 8px", "fontSize": "11px", "marginBottom": "8px",
                                "boxSizing": "border-box",
                            },
                        ),
                    ], style={"flex": "1"}),
                    html.Div([
                        ctrl_label("Min TX / Patch"),
                        dcc.Input(
                            id="sdata-patch-min-tx", type="number", value=10, min=1, max=1000,
                            debounce=True,
                            style={
                                "width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                                "border": f"1px solid {BORDER}", "borderRadius": "4px",
                                "padding": "4px 8px", "fontSize": "11px", "marginBottom": "8px",
                                "boxSizing": "border-box",
                            },
                        ),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "12px", "marginBottom": "6px"}),
                # Prepare button + status row
                html.Div([
                    html.Button(
                        "▶ Prepare Patches",
                        id="sdata-prepare-btn", n_clicks=0,
                        style={
                            "padding": "6px 14px", "backgroundColor": "#1f6feb", "color": "#fff",
                            "border": "none", "borderRadius": "4px",
                            "cursor": "pointer", "fontSize": "12px", "fontWeight": "600",
                        },
                    ),
                    html.Button(
                        "↺ Re-build",
                        id="sdata-clear-cache-btn", n_clicks=0,
                        title="Clear disk cache and re-read from raw files",
                        style={
                            "padding": "6px 10px", "backgroundColor": CARD_BG, "color": MUTED,
                            "border": f"1px solid {BORDER}", "borderRadius": "4px",
                            "cursor": "pointer", "fontSize": "11px",
                        },
                    ),
                    html.Button(
                        "⏭ Skip — run on full tissue",
                        id="reseg-skip-patches-btn", n_clicks=0,
                        title="Skip patch building and run segmentation on the full tissue",
                        style={
                            "padding": "6px 10px", "backgroundColor": CARD_BG, "color": "#e3b341",
                            "border": f"1px solid #e3b341", "borderRadius": "4px",
                            "cursor": "pointer", "fontSize": "11px",
                        },
                    ),
                    html.Div(id="sdata-status",
                             style={"fontSize": "11px", "color": MUTED, "flex": "1", "minHeight": "16px", "alignSelf": "center"}),
                ], style={"display": "flex", "gap": "6px", "marginBottom": "8px", "alignItems": "center"}),
                # Overlay checkboxes
                html.Div([
                    dcc.Checklist(
                        id="sdata-show-roi",
                        options=[{"label": html.Span(" Show tissue ROI", style={"fontSize": "11px", "color": "#52b788"}), "value": "yes"}],
                        value=[],
                        inputStyle={"marginRight": "5px"},
                        labelStyle={"display": "flex", "alignItems": "center"},
                    ),
                    dcc.Checklist(
                        id="sdata-show-patches",
                        options=[{"label": html.Span(" Show patch grid", style={"fontSize": "11px", "color": "#ffa500"}), "value": "yes"}],
                        value=[],
                        inputStyle={"marginRight": "5px"},
                        labelStyle={"display": "flex", "alignItems": "center"},
                    ),
                ], style={"display": "flex", "gap": "20px", "marginBottom": "10px"}),
                # Confirmation banner (hidden until patches are ready)
                html.Div(id="reseg-patch-confirm-div", children=[
                    html.Div(id="reseg-patch-confirm-msg",
                             style={"fontSize": "11px", "color": "#52b788", "marginBottom": "6px"}),
                    html.Button(
                        "✓ Patches look good — proceed to segmentation",
                        id="reseg-confirm-patches-btn",
                        n_clicks=0,
                        style={
                            "width": "100%", "padding": "7px 0",
                            "backgroundColor": "#2d6a4f", "color": "#fff",
                            "border": "none", "borderRadius": "5px",
                            "cursor": "pointer", "fontSize": "12px", "fontWeight": "600",
                            "marginBottom": "4px",
                        },
                    ),
                ], style={"display": "none"}),
                dcc.Interval(id="sdata-poll", interval=2000, disabled=True),
                # Hidden elements needed by existing callbacks
                html.Div(id="sdata-convert-btn", style={"display": "none"}),
                html.Div(id="sdata-roi-btn", style={"display": "none"}),
                html.Div(id="sdata-patches-btn", style={"display": "none"}),
            ], style={"backgroundColor": "#161b22", "borderRadius": "6px", "padding": "12px", "marginBottom": "14px"}),

            html.Hr(style={"borderColor": BORDER, "margin": "0 0 14px 0"}),

            html.Div([
                html.Div("Step 2 — Segmentation Settings", style={"fontSize": "13px", "fontWeight": "600", "color": TEXT, "marginBottom": "10px"}),
                html.Div(id="reseg-step2-overlay", style={"display": "block"}, children=[
                    html.Div("Confirm patches above to enable segmentation settings.",
                             style={"fontSize": "11px", "color": MUTED, "textAlign": "center", "padding": "8px 0"}),
                ]),
                html.Div(id="reseg-step2-content", style={"opacity": "0.4", "pointerEvents": "none"}, children=[
                    # Algorithm selector tabs
                    dcc.Tabs(id="reseg-algo-tabs", value="baysor", children=[
                        dcc.Tab(label="Baysor", value="baysor", style={"color": TEXT, "backgroundColor": CARD_BG},
                                selected_style={"color": "#fff", "backgroundColor": "#1f6feb", "fontWeight": "600"}),
                        dcc.Tab(label="Proseg", value="proseg", style={"color": TEXT, "backgroundColor": CARD_BG},
                                selected_style={"color": "#fff", "backgroundColor": "#2d6a4f", "fontWeight": "600"}),
                    ], style={"marginBottom": "16px"}),

                    # ── Baysor settings panel ────────────────────────────────────
            html.Div(id="modal-baysor-panel", children=[
                html.Div(
                    "Resegment cells using Baysor (must be installed). "
                    "See github.com/kharchenkolab/Baysor.",
                    style={"fontSize": "10px", "color": MUTED, "marginBottom": "8px", "lineHeight": "1.4"},
                ),
                ctrl_label("Spatial Region (µm) — leave blank for full slide"),
                html.Div([
                    html.Div([
                        html.Div("X min", style={"fontSize": "10px", "color": MUTED, "marginBottom": "2px"}),
                        dcc.Input(id="baysor-xmin", type="number", placeholder="auto",
                                  style={"width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                                         "border": f"1px solid {BORDER}", "borderRadius": "4px",
                                         "padding": "3px 6px", "fontSize": "11px", "boxSizing": "border-box"}),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Div("X max", style={"fontSize": "10px", "color": MUTED, "marginBottom": "2px"}),
                        dcc.Input(id="baysor-xmax", type="number", placeholder="auto",
                                  style={"width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                                         "border": f"1px solid {BORDER}", "borderRadius": "4px",
                                         "padding": "3px 6px", "fontSize": "11px", "boxSizing": "border-box"}),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "6px", "marginBottom": "6px"}),
                html.Div([
                    html.Div([
                        html.Div("Y min", style={"fontSize": "10px", "color": MUTED, "marginBottom": "2px"}),
                        dcc.Input(id="baysor-ymin", type="number", placeholder="auto",
                                  style={"width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                                         "border": f"1px solid {BORDER}", "borderRadius": "4px",
                                         "padding": "3px 6px", "fontSize": "11px", "boxSizing": "border-box"}),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Div("Y max", style={"fontSize": "10px", "color": MUTED, "marginBottom": "2px"}),
                        dcc.Input(id="baysor-ymax", type="number", placeholder="auto",
                                  style={"width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                                         "border": f"1px solid {BORDER}", "borderRadius": "4px",
                                         "padding": "3px 6px", "fontSize": "11px", "boxSizing": "border-box"}),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "6px", "marginBottom": "6px"}),
                html.Button(
                    "Use Current Viewport", id="baysor-use-viewport", n_clicks=0,
                    style={"width": "100%", "padding": "4px 0", "backgroundColor": CARD_BG,
                           "color": MUTED, "border": f"1px solid {BORDER}", "borderRadius": "4px",
                           "cursor": "pointer", "fontSize": "11px", "marginBottom": "10px"},
                ),
                html.Div([
                    html.Div([
                        ctrl_label("Cell Radius (µm, −1 = auto)"),
                        dcc.Input(
                            id="baysor-scale", type="number", value=20, min=-1, step=1,
                            style={"width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                                   "border": f"1px solid {BORDER}", "borderRadius": "4px",
                                   "padding": "4px 8px", "fontSize": "12px", "marginBottom": "8px"},
                        ),
                    ], style={"flex": "1"}),
                    html.Div([
                        ctrl_label("Scale Std (optional)"),
                        dcc.Input(
                            id="baysor-scale-std", type="number", value=None, min=0, step=0.05,
                            placeholder="e.g. 0.25",
                            style={"width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                                   "border": f"1px solid {BORDER}", "borderRadius": "4px",
                                   "padding": "4px 8px", "fontSize": "12px", "marginBottom": "8px"},
                        ),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "6px"}),
                html.Div([
                    html.Div([
                        ctrl_label("Min Transcripts / Cell"),
                        dcc.Input(
                            id="baysor-min-mol", type="number", value=10, min=1, max=500, step=1,
                            style={"width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                                   "border": f"1px solid {BORDER}", "borderRadius": "4px",
                                   "padding": "4px 8px", "fontSize": "12px", "marginBottom": "8px"},
                        ),
                    ], style={"flex": "1"}),
                    html.Div([
                        ctrl_label("Number of cell types"),
                        dcc.Input(
                            id="baysor-n_clusters", type="number", value=10, min=1, step=1,
                            style={"width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                                   "border": f"1px solid {BORDER}", "borderRadius": "4px",
                                   "padding": "4px 8px", "fontSize": "12px", "marginBottom": "8px"},
                        ),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "6px"}),
                dcc.Checklist(
                    id="baysor-use-patches",
                    options=[{"label": html.Span(
                        " Use patch-based segmentation (from Step 1)",
                        style={"fontSize": "12px", "color": TEXT},
                    ), "value": "yes"}],
                    value=["yes"],
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"display": "flex", "alignItems": "center", "marginBottom": "4px"},
                ),
                dcc.Checklist(
                    id="baysor-use-prior",
                    options=[{"label": html.Span(
                        " Use Xenium nuclei as prior segmentation",
                        style={"fontSize": "12px", "color": TEXT},
                    ), "value": "yes"}],
                    value=["yes"],
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"display": "flex", "alignItems": "center", "marginBottom": "4px"},
                ),
                html.Div(id="baysor-prior-conf-div", children=[
                    ctrl_label("Prior Confidence"),
                    dcc.Slider(
                        id="baysor-prior-conf", min=0.0, max=1.0, step=0.05, value=0.5,
                        marks={0: "0", 0.5: "0.5", 1: "1"},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    html.Div(style={"marginBottom": "6px"}),
                ]),
            ]),

            # ── Proseg settings panel ────────────────────────────────────
            html.Div(id="modal-proseg-panel", children=[
                html.Div(
                    "Probabilistic cell segmentation (faster than Baysor). "
                    "Install: conda install bioconda::rust-proseg",
                    style={"fontSize": "10px", "color": MUTED, "marginBottom": "8px", "lineHeight": "1.4"},
                ),
                ctrl_label("Spatial Region (µm) — leave blank for full slide"),
                html.Div([
                    html.Div([
                        html.Div("X min", style={"fontSize": "10px", "color": MUTED, "marginBottom": "2px"}),
                        dcc.Input(id="proseg-xmin", type="number", placeholder="auto",
                                  style={"width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                                         "border": f"1px solid {BORDER}", "borderRadius": "4px",
                                         "padding": "3px 6px", "fontSize": "11px", "boxSizing": "border-box"}),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Div("X max", style={"fontSize": "10px", "color": MUTED, "marginBottom": "2px"}),
                        dcc.Input(id="proseg-xmax", type="number", placeholder="auto",
                                  style={"width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                                         "border": f"1px solid {BORDER}", "borderRadius": "4px",
                                         "padding": "3px 6px", "fontSize": "11px", "boxSizing": "border-box"}),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "6px", "marginBottom": "6px"}),
                html.Div([
                    html.Div([
                        html.Div("Y min", style={"fontSize": "10px", "color": MUTED, "marginBottom": "2px"}),
                        dcc.Input(id="proseg-ymin", type="number", placeholder="auto",
                                  style={"width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                                         "border": f"1px solid {BORDER}", "borderRadius": "4px",
                                         "padding": "3px 6px", "fontSize": "11px", "boxSizing": "border-box"}),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Div("Y max", style={"fontSize": "10px", "color": MUTED, "marginBottom": "2px"}),
                        dcc.Input(id="proseg-ymax", type="number", placeholder="auto",
                                  style={"width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                                         "border": f"1px solid {BORDER}", "borderRadius": "4px",
                                         "padding": "3px 6px", "fontSize": "11px", "boxSizing": "border-box"}),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "6px", "marginBottom": "6px"}),
                html.Button(
                    "Use Current Viewport", id="proseg-use-viewport", n_clicks=0,
                    style={"width": "100%", "padding": "4px 0", "backgroundColor": CARD_BG,
                           "color": MUTED, "border": f"1px solid {BORDER}", "borderRadius": "4px",
                           "cursor": "pointer", "fontSize": "11px", "marginBottom": "10px"},
                ),
                ctrl_label("Initial voxel size (µm) — leave blank for default (4)"),
                dcc.Input(
                    id="proseg-voxel-size", type="number", placeholder="auto", min=0.1, max=10, step=0.1,
                    style={"width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                           "border": f"1px solid {BORDER}", "borderRadius": "4px",
                           "padding": "4px 8px", "fontSize": "12px", "marginBottom": "8px",
                           "boxSizing": "border-box"},
                ),
                ctrl_label("Threads"),
                dcc.Input(
                    id="proseg-nthreads", type="number", placeholder="all", min=1, max=128, step=1,
                    style={"width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                           "border": f"1px solid {BORDER}", "borderRadius": "4px",
                           "padding": "4px 8px", "fontSize": "12px", "marginBottom": "8px",
                           "boxSizing": "border-box"},
                ),
                ctrl_label("Components (--ncomponents)"),
                dcc.Input(
                    id="proseg-samples", type="number", placeholder="10", min=1, max=100, step=1,
                    style={"width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                           "border": f"1px solid {BORDER}", "borderRadius": "4px",
                           "padding": "4px 8px", "fontSize": "12px", "marginBottom": "8px",
                           "boxSizing": "border-box"},
                ),
                ctrl_label("Recorded samples (--recorded-samples)"),
                dcc.Input(
                    id="proseg-recorded-samples", type="number", placeholder="100", min=10, max=1000, step=10,
                    style={"width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                           "border": f"1px solid {BORDER}", "borderRadius": "4px",
                           "padding": "4px 8px", "fontSize": "12px", "marginBottom": "8px",
                           "boxSizing": "border-box"},
                ),
                ctrl_label("Sampler schedule (--schedule) — space-separated integers"),
                dcc.Input(
                    id="proseg-schedule", type="text", placeholder="150 150 300",
                    style={"width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                           "border": f"1px solid {BORDER}", "borderRadius": "4px",
                           "padding": "4px 8px", "fontSize": "12px", "marginBottom": "8px",
                           "boxSizing": "border-box"},
                ),
                ctrl_label("Nuclear reassignment prob (--nuclear-reassignment-prob)"),
                dcc.Input(
                    id="proseg-nuclear-reassign-prob", type="number", placeholder="0.2",
                    min=0.0, max=1.0, step=0.05,
                    style={"width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                           "border": f"1px solid {BORDER}", "borderRadius": "4px",
                           "padding": "4px 8px", "fontSize": "12px", "marginBottom": "8px",
                           "boxSizing": "border-box"},
                ),
                ctrl_label("Prior seg reassignment prob (--prior-seg-reassignment-prob)"),
                dcc.Input(
                    id="proseg-prior-seg-prob", type="number", placeholder="0.5",
                    min=0.0, max=1.0, step=0.05,
                    style={"width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                           "border": f"1px solid {BORDER}", "borderRadius": "4px",
                           "padding": "4px 8px", "fontSize": "12px", "marginBottom": "8px",
                           "boxSizing": "border-box"},
                ),
            ], style={"display": "none"}),  # hidden by default; shown when proseg tab is active
                ]),  # end reseg-step2-content
            ]),  # end Step 2 div
        ]),
        dbc.ModalFooter([
            html.Div(id="reseg-modal-run-status", style={"fontSize": "11px", "color": MUTED, "flex": "1"}),
            html.Button(
                "Run",
                id="reseg-modal-run-btn",
                n_clicks=0,
                disabled=True,
                style={
                    "padding": "7px 20px",
                    "backgroundColor": "#1f6feb", "color": "#fff",
                    "border": "none", "borderRadius": "5px",
                    "cursor": "pointer", "fontSize": "13px", "fontWeight": "600",
                },
            ),
            dbc.Button("Close", id="reseg-modal-close-btn", color="secondary", size="sm"),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
    ], id="reseg-modal", is_open=False, size="lg",
       style={"color": TEXT},
       content_style={"backgroundColor": DARK_BG, "color": TEXT},
       backdrop=True),

# ── Cell Type Annotation modal ────────────────────────────────────────────────
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Cell Type Annotation"), close_button=True),
        dbc.ModalBody([
            html.Div(
                "Annotate cells using CellTypist, Seurat label transfer, or RCTD (spacexr).",
                style={"fontSize": "11px", "color": MUTED, "marginBottom": "12px", "lineHeight": "1.4"},
            ),
            ctrl_label("Method"),
            dcc.RadioItems(
                id="annot-source",
                options=[
                    {"label": html.Span(" CellTypist", style={"fontSize": "12px", "color": TEXT}),
                     "value": "celltypist"},
                    {"label": html.Span(" Seurat kNN label transfer", style={"fontSize": "12px", "color": TEXT}),
                     "value": "seurat"},
                    {"label": html.Span(" RCTD (spacexr)", style={"fontSize": "12px", "color": TEXT}),
                     "value": "rctd"},
                ],
                value="celltypist",
                inputStyle={"marginRight": "5px"},
                labelStyle={"display": "flex", "alignItems": "center", "marginBottom": "5px", "cursor": "pointer"},
                style={"marginBottom": "10px"},
            ),
            # CellTypist controls
            html.Div(id="annot-celltypist-div", children=[
                ctrl_label("Model"),
                dcc.Dropdown(
                    id="annot-model",
                    options=[{"label": v, "value": k} for k, v in CELLTYPIST_MODELS.items()],
                    value="Healthy_Adult_Heart.pkl",
                    clearable=False,
                    style={"fontSize": "12px", "marginBottom": "10px",
                           "backgroundColor": CARD_BG, "color": TEXT},
                ),
            ]),
            # Seurat / RCTD shared RDS controls (shown for both)
            html.Div(id="annot-seurat-div", style={"display": "none"}, children=[
                ctrl_label("RDS File Path"),
                dcc.Input(
                    id="annot-rds-path",
                    type="text",
                    value="/Users/ikuz/Documents/XeniumWorkflow/snRV_ref.rds",
                    style={
                        "width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                        "border": f"1px solid {BORDER}", "borderRadius": "4px",
                        "padding": "4px 8px", "fontSize": "11px", "marginBottom": "8px",
                        "boxSizing": "border-box",
                    },
                ),
                ctrl_label("Cell Type Label Column"),
                dcc.Input(
                    id="annot-label-col",
                    type="text",
                    value="Names",
                    style={
                        "width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                        "border": f"1px solid {BORDER}", "borderRadius": "4px",
                        "padding": "4px 8px", "fontSize": "11px", "marginBottom": "8px",
                        "boxSizing": "border-box",
                    },
                ),
            ]),
            # RCTD-specific controls (shown only for rctd)
            html.Div(id="annot-rctd-div", style={"display": "none"}, children=[
                html.Div(
                    "Requires the spacexr R package. In R: devtools::install_github('dmcable/spacexr')",
                    style={"fontSize": "10px", "color": MUTED, "marginBottom": "8px",
                           "lineHeight": "1.4", "fontStyle": "italic"},
                ),
                ctrl_label("RCTD Mode"),
                dcc.Dropdown(
                    id="annot-rctd-mode",
                    options=[
                        {"label": "full — weights across all types (best for single cells)", "value": "full"},
                        {"label": "doublet — primary + secondary type",                      "value": "doublet"},
                        {"label": "multi — multiple types per spot",                         "value": "multi"},
                    ],
                    value="full",
                    clearable=False,
                    style={"fontSize": "11px", "marginBottom": "8px",
                           "backgroundColor": CARD_BG, "color": TEXT},
                ),
                ctrl_label("Max Cores"),
                dcc.Input(
                    id="annot-rctd-cores",
                    type="number", value=4, min=1, max=64, step=1,
                    style={
                        "width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                        "border": f"1px solid {BORDER}", "borderRadius": "4px",
                        "padding": "4px 8px", "fontSize": "11px", "marginBottom": "4px",
                        "boxSizing": "border-box",
                    },
                ),
                ctrl_label("Min UMI (cells below excluded)"),
                dcc.Input(
                    id="annot-rctd-umi-min",
                    type="number", value=20, min=1, max=500, step=1,
                    style={
                        "width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                        "border": f"1px solid {BORDER}", "borderRadius": "4px",
                        "padding": "4px 8px", "fontSize": "11px", "marginBottom": "4px",
                        "boxSizing": "border-box",
                    },
                ),
                ctrl_label("Min UMI sigma (UMI_min_sigma)"),
                dcc.Input(
                    id="annot-rctd-umi-min-sigma",
                    type="number", value=100, min=1, max=1000, step=1,
                    style={
                        "width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                        "border": f"1px solid {BORDER}", "borderRadius": "4px",
                        "padding": "4px 8px", "fontSize": "11px", "marginBottom": "4px",
                        "boxSizing": "border-box",
                    },
                ),
            ]),
        ]),
        dbc.ModalFooter([
            html.Div(id="annot-modal-status", style={"fontSize": "11px", "color": MUTED, "flex": "1"}),
            html.Button(
                "Run Annotation",
                id="annot-btn",
                n_clicks=0,
                style={
                    "padding": "7px 20px", "backgroundColor": "#238636", "color": "#fff",
                    "border": "none", "borderRadius": "5px",
                    "cursor": "pointer", "fontSize": "13px", "fontWeight": "600",
                },
            ),
            dbc.Button("Close", id="annot-modal-close-btn", color="secondary", size="sm"),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
    ], id="annot-modal", is_open=False, size="md",
       style={"color": TEXT},
       content_style={"backgroundColor": DARK_BG, "color": TEXT},
       backdrop=True),

# ── SpaGE Gene Imputation modal ───────────────────────────────────────────────
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("SpaGE Gene Imputation"), close_button=True),
        dbc.ModalBody([
            html.Div(
                "Impute expression of genes not in the Xenium panel using a scRNA-seq reference (Seurat RDS). "
                "Results are cached and auto-loaded on next startup.",
                style={"fontSize": "11px", "color": MUTED, "marginBottom": "14px", "lineHeight": "1.5"},
            ),
            ctrl_label("Reference .rds File"),
            dcc.Input(
                id="spage-rds-path",
                type="text",
                value="/Users/ikuz/Documents/XeniumWorkflow/snRV_ref.rds",
                debounce=True,
                style={
                    "width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                    "border": f"1px solid {BORDER}", "borderRadius": "4px",
                    "padding": "4px 8px", "fontSize": "11px", "marginBottom": "12px",
                    "boxSizing": "border-box",
                },
            ),
            ctrl_label("Principal Vectors (n_pv)"),
            dcc.Slider(
                id="spage-npv", min=10, max=100, step=5, value=50,
                marks={10: "10", 50: "50", 100: "100"},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(style={"marginBottom": "12px"}),
            ctrl_label("Genes to Impute"),
            html.Div(
                "Leave empty for top-200 highly variable genes. Separate gene names with commas or newlines.",
                style={"fontSize": "10px", "color": MUTED, "marginBottom": "6px", "lineHeight": "1.4"},
            ),
            dcc.Textarea(
                id="spage-genes",
                placeholder="e.g. NPPA, MYH7, TNNT2",
                style={
                    "width": "100%", "height": "100px", "backgroundColor": CARD_BG,
                    "color": TEXT, "border": f"1px solid {BORDER}", "borderRadius": "4px",
                    "padding": "6px 8px", "fontSize": "11px", "marginBottom": "4px",
                    "boxSizing": "border-box", "resize": "vertical",
                },
            ),
        ]),
        dbc.ModalFooter([
            html.Div(id="spage-modal-status",
                     style={"fontSize": "11px", "color": MUTED, "flex": "1"}),
            html.Button(
                "Run SpaGE",
                id="spage-run-btn",
                n_clicks=0,
                style={
                    "padding": "7px 20px",
                    "backgroundColor": "#6f42c1", "color": "#fff",
                    "border": "none", "borderRadius": "5px",
                    "cursor": "pointer", "fontSize": "13px", "fontWeight": "600",
                },
            ),
            dbc.Button("Close", id="spage-modal-close-btn", color="secondary", size="sm"),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
    ], id="spage-modal", is_open=False, size="lg",
       style={"color": TEXT},
       content_style={"backgroundColor": DARK_BG, "color": TEXT},
       backdrop=True),

# ─── ROI Save modal ───────────────────────────────────────────────────────────
dbc.Modal([
    dbc.ModalHeader("Save ROI"),
    dbc.ModalBody([
        html.Div(id="roi-save-preview",
                 style={"fontSize": "0.85rem", "color": MUTED, "marginBottom": "10px"}),
        dbc.Label("ROI Name", style={"fontSize": "0.82rem"}),
        dbc.Input(id="roi-name-input", placeholder="e.g. Tumor_1",
                  style={"backgroundColor": CARD_BG, "color": TEXT, "border": f"1px solid {BORDER}", "marginBottom": "8px"}),
        dbc.Label("ROI Class", style={"fontSize": "0.82rem"}),
        dbc.Input(id="roi-class-input", placeholder="e.g. region", value="region",
                  style={"backgroundColor": CARD_BG, "color": TEXT, "border": f"1px solid {BORDER}"}),
        html.Div(id="roi-save-error",
                 style={"color": "#e74c3c", "fontSize": "0.82rem", "marginTop": "8px"}),
    ]),
    dbc.ModalFooter([
        dbc.Button("Save", id="roi-save-btn", color="primary", size="sm"),
        dbc.Button("Cancel", id="roi-save-cancel-btn", color="secondary", size="sm", className="ms-2"),
    ]),
], id="roi-save-modal", is_open=False,
   style={"color": TEXT},
   content_style={"backgroundColor": DARK_BG, "color": TEXT},
   backdrop=True),

# ─── ROI Manager modal ────────────────────────────────────────────────────────
dbc.Modal([
    dbc.ModalHeader("Manage ROIs"),
    dbc.ModalBody([
        html.Div(id="roi-list-div", style={"marginBottom": "18px"}),
        html.Hr(style={"borderColor": BORDER}),
        html.Div([
            html.P("ROI Set Operations", style={"fontWeight": "600", "fontSize": "0.9rem", "marginBottom": "8px"}),
            html.Div([
                dcc.Dropdown(id="roi-op-a", placeholder="ROI A",
                             style={"backgroundColor": CARD_BG, "color": TEXT, "fontSize": "0.82rem", "flex": "1"}),
                dcc.Dropdown(id="roi-op", options=[
                    {"label": "Union", "value": "union"},
                    {"label": "Intersection", "value": "intersection"},
                    {"label": "Subtract A−B", "value": "subtract"},
                ], placeholder="Operation",
                    style={"backgroundColor": CARD_BG, "color": TEXT, "fontSize": "0.82rem", "width": "130px"}),
                dcc.Dropdown(id="roi-op-b", placeholder="ROI B",
                             style={"backgroundColor": CARD_BG, "color": TEXT, "fontSize": "0.82rem", "flex": "1"}),
            ], style={"display": "flex", "gap": "8px", "marginBottom": "8px", "alignItems": "center"}),
            html.Div([
                dbc.Input(id="roi-op-name", placeholder="Result name",
                          style={"backgroundColor": CARD_BG, "color": TEXT, "border": f"1px solid {BORDER}", "flex": "1"}),
                dbc.Input(id="roi-op-class", placeholder="Result class", value="region",
                          style={"backgroundColor": CARD_BG, "color": TEXT, "border": f"1px solid {BORDER}", "flex": "1"}),
                dbc.Button("Apply", id="roi-op-btn", color="primary", size="sm"),
            ], style={"display": "flex", "gap": "8px", "alignItems": "center"}),
            html.Div(id="roi-op-error",
                     style={"color": "#e74c3c", "fontSize": "0.82rem", "marginTop": "6px"}),
        ]),
    ]),
    dbc.ModalFooter([
        dbc.Button("Close", id="roi-manage-close-btn", color="secondary", size="sm"),
    ]),
], id="roi-manage-modal", is_open=False, size="lg",
   style={"color": TEXT},
   content_style={"backgroundColor": DARK_BG, "color": TEXT},
   backdrop=True),

# ─── SPLIT correction modal ───────────────────────────────────────────────────
dbc.Modal([
    dbc.ModalHeader("SPLIT Ambient RNA Correction"),
    dbc.ModalBody([
        html.P("Requires R packages: spacexr (dmcable/spacexr) and SPLIT (bdsc-tds/SPLIT).",
               style={"fontSize":"0.82rem","color":"#8b949e"}),
        html.P("⚠ Peak RAM usage ~21 GB for a full Xenium slide.",
               style={"fontSize":"0.82rem","color":"#e85d04"}),
        dbc.Label("Seurat Reference RDS path"),
        dbc.Input(id="split-rds-path", placeholder="/path/to/reference.rds",
                  type="text", style={"fontSize":"0.82rem"}),
        html.Br(),
        dbc.Label("Cell-type label column"),
        dbc.Input(id="split-label-col", value="Names",
                  type="text", style={"fontSize":"0.82rem"}),
        html.Br(),
        dbc.Label("Max R cores"),
        dbc.Input(id="split-max-cores", value=4, type="number",
                  min=1, max=32, style={"fontSize":"0.82rem"}),
        html.Br(),
        dbc.Label("Min UMI (cells below this are excluded from RCTD)"),
        dbc.Input(id="split-min-umi", value=10, type="number",
                  min=1, max=500, style={"fontSize":"0.82rem"}),
        html.Br(),
        dbc.Label("Min UMI sigma (UMI_min_sigma)"),
        dbc.Input(id="split-min-umi-sigma", value=100, type="number",
                  min=1, max=1000, style={"fontSize":"0.82rem"}),
        html.Br(),
        dbc.Checklist(
            id="split-purify-singlets",
            options=[{"label": " Purify singlets (DO_purify_singlets)", "value": "yes"}],
            value=["yes"],
            style={"fontSize":"0.82rem"},
        ),
        html.P(
            "When enabled, ambient RNA is also removed from singlet cells. "
            "Disable if corrected expression looks too aggressively reduced.",
            style={"fontSize":"0.75rem","color":"#8b949e","marginTop":"2px"},
        ),
        html.Br(),
        dbc.Button("▶ Run SPLIT", id="split-run-btn", color="danger",
                   style={"width":"100%"}),
        html.Div(id="split-modal-status",
                 style={"marginTop":"8px","fontSize":"0.82rem","color":"#8b949e"}),
    ]),
    dbc.ModalFooter(
        dbc.Button("Close", id="split-modal-close-btn", color="secondary", size="sm")
    ),
], id="split-modal", is_open=False, size="lg"),

# ── Load Sample modal ─────────────────────────────────────────────────────────
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Load Xenium Sample"), close_button=True),
        dbc.ModalBody([
            html.Div("Select an available sample or enter a custom path:",
                     style={"fontSize": "12px", "color": MUTED, "marginBottom": "10px"}),
            dcc.Dropdown(
                id="sample-picker",
                options=_available_samples(),
                placeholder="Available samples\u2026",
                style={"marginBottom": "10px", "backgroundColor": CARD_BG, "color": TEXT},
            ),
            html.Div("Or enter a custom path:", style={"fontSize": "11px", "color": MUTED, "marginBottom": "4px"}),
            dcc.Input(
                id="sample-custom-path",
                type="text",
                placeholder="/path/to/output-XETG\u2026",
                style={"width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                       "border": f"1px solid {BORDER}", "borderRadius": "4px",
                       "padding": "6px 10px", "fontSize": "12px", "boxSizing": "border-box"},
            ),
            html.Div(id="sample-load-status",
                     style={"fontSize": "11px", "color": MUTED, "marginTop": "8px", "minHeight": "16px"}),
        ]),
        dbc.ModalFooter([
            html.Button("Load (Replace)", id="sample-load-btn", n_clicks=0,
                        style={"padding": "7px 20px", "backgroundColor": "#1f6feb", "color": "#fff",
                               "border": "none", "borderRadius": "5px", "cursor": "pointer",
                               "fontSize": "13px", "fontWeight": "600"}),
            html.Button("Add as Additional Sample", id="sample-add-btn", n_clicks=0,
                        style={"padding": "7px 20px", "backgroundColor": "#238636", "color": "#fff",
                               "border": "none", "borderRadius": "5px", "cursor": "pointer",
                               "fontSize": "13px", "fontWeight": "600", "marginLeft": "8px"}),
            dbc.Button("Cancel", id="sample-modal-close-btn", color="secondary", size="sm"),
        ]),
    ], id="sample-modal", is_open=False, size="lg",
       content_style={"backgroundColor": DARK_BG, "color": TEXT}),

# ── Delete segmentation run confirmation modal ────────────────────────────────
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Delete Segmentation Run"), close_button=True),
        dbc.ModalBody(
            html.Div(id="seg-delete-confirm-text",
                     style={"fontSize": "13px", "color": TEXT, "lineHeight": "1.5"}),
        ),
        dbc.ModalFooter([
            html.Button(
                "Delete", id="seg-delete-confirm-btn", n_clicks=0,
                style={
                    "padding": "7px 20px", "backgroundColor": "#da3633", "color": "#fff",
                    "border": "none", "borderRadius": "5px",
                    "cursor": "pointer", "fontSize": "13px", "fontWeight": "600",
                },
            ),
            dbc.Button("Cancel", id="seg-delete-cancel-btn", color="secondary", size="sm"),
        ]),
    ], id="seg-delete-modal", is_open=False,
       content_style={"backgroundColor": DARK_BG, "color": TEXT}),

# ── Clean Cache confirmation modal ────────────────────────────────────────────
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Clean Cache"), close_button=True),
        dbc.ModalBody([
            html.Div(id="cache-clean-confirm-text",
                     style={"fontSize": "13px", "color": TEXT}),
        ]),
        dbc.ModalFooter([
            html.Button("Delete", id="cache-clean-confirm-btn", n_clicks=0,
                        style={"padding": "7px 20px", "backgroundColor": "#da3633", "color": "#fff",
                               "border": "none", "borderRadius": "5px", "cursor": "pointer",
                               "fontSize": "13px", "fontWeight": "600"}),
            dbc.Button("Cancel", id="cache-clean-cancel-btn", color="secondary", size="sm"),
        ]),
    ], id="cache-clean-modal", is_open=False,
       content_style={"backgroundColor": DARK_BG, "color": TEXT}),

# ── Save as SpatialData modal ──────────────────────────────────────────────────
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Save as SpatialData"), close_button=True),
        dbc.ModalBody([
            html.Div(
                "Export the current dataset (transcripts, images, shapes) plus any analysis results "
                "(cluster labels, cell-type annotations) to a SpatialData Zarr store.",
                style={"fontSize": "11px", "color": MUTED, "marginBottom": "14px", "lineHeight": "1.5"},
            ),
            ctrl_label("Output Directory"),
            dcc.Input(
                id="save-sdata-dir",
                type="text",
                placeholder="/path/to/output/directory",
                debounce=True,
                style={
                    "width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                    "border": f"1px solid {BORDER}", "borderRadius": "4px",
                    "padding": "5px 8px", "fontSize": "11px", "marginBottom": "10px",
                    "boxSizing": "border-box",
                },
            ),
            ctrl_label("Filename (folder name for the .zarr store)"),
            dcc.Input(
                id="save-sdata-name",
                type="text",
                placeholder="my_analysis.zarr",
                debounce=True,
                style={
                    "width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                    "border": f"1px solid {BORDER}", "borderRadius": "4px",
                    "padding": "5px 8px", "fontSize": "11px", "marginBottom": "10px",
                    "boxSizing": "border-box",
                },
            ),
            dcc.Checklist(
                id="save-sdata-roi-only",
                options=[{"label": "  Export only cells within ROIs (if ROIs are defined)", "value": "roi_only"}],
                value=[],
                style={"fontSize": "11px", "color": TEXT, "marginBottom": "8px"},
            ),
            html.Div(
                "Note: Exports the active segmentation. For Xenium, SpatialData must be built first "
                "(open 'Resegment Cells' → patches auto-build). Existing stores at the same path will be overwritten.",
                style={"fontSize": "10px", "color": MUTED, "lineHeight": "1.4"},
            ),
            dcc.Interval(id="save-sdata-poll", interval=1500, disabled=True),
        ]),
        dbc.ModalFooter([
            html.Div(id="save-sdata-status",
                     style={"fontSize": "11px", "color": MUTED, "flex": "1"}),
            html.Button(
                "Save",
                id="save-sdata-run-btn",
                n_clicks=0,
                style={
                    "padding": "7px 20px",
                    "backgroundColor": "#1f6feb", "color": "#fff",
                    "border": "none", "borderRadius": "5px",
                    "cursor": "pointer", "fontSize": "13px", "fontWeight": "600",
                },
            ),
            dbc.Button("Close", id="save-sdata-close-btn", color="secondary", size="sm"),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
    ], id="save-sdata-modal", is_open=False, size="lg",
       style={"color": TEXT},
       content_style={"backgroundColor": DARK_BG, "color": TEXT},
       backdrop=True),

    html.Div([
        # ── Hover-reveal sidebar ──────────────────────────────────────────────
        # Outer zone covers the trigger strip + sidebar; hover on either reveals it
        html.Div([
            html.Div(
                sidebar,
                id="sidebar-wrapper",
            ),
        ], id="sidebar-hover-zone", style={
            "position": "fixed", "top": "0", "left": "0",
            "height": "100vh", "zIndex": "100",
            # Wide enough to keep hover active while using sidebar
            "width": "250px",
        }),

        html.Div([
            # Plots row
            html.Div([
                html.Div([
                    dcc.Graph(
                        id="spatial-plot",
                        responsive=True,
                        figure={
                            "data": [],
                            "layout": {
                                "paper_bgcolor": PLOT_BG,
                                "plot_bgcolor": PLOT_BG,
                                "xaxis": {"visible": False},
                                "yaxis": {"visible": False},
                                "annotations": [{
                                    "text": "Loading…",
                                    "x": 0.5, "y": 0.5,
                                    "xref": "paper", "yref": "paper",
                                    "showarrow": False,
                                    "font": {"size": 18, "color": "#8b949e"},
                                }],
                                "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
                            },
                        },
                        config={
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["select2d"],
                            "toImageButtonOptions": {"format": "svg", "filename": "xenium_spatial"},
                        },
                        style={"height": "100%"},
                    )
                ], id="spatial-panel", style={"flex": "2", "minWidth": "0"}),

                html.Div([
                    dcc.Graph(
                        id="umap-plot",
                        responsive=True,
                        config={
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                            "toImageButtonOptions": {"format": "svg", "filename": "xenium_umap"},
                        },
                        style={"height": "100%"},
                    )
                ], id="umap-panel", style={"display": "none"}),
            ], style={"display": "flex", "flex": "1", "gap": "10px", "minHeight": "0"}),

            # ── Bottom info bar (collapsible) ───────────────────────────────
            html.Div([
                # Toggle arrow button
                html.Button(
                    "▼", id="info-bar-toggle", n_clicks=0,
                    title="Hide/show info bar",
                    style={
                        "position": "absolute", "top": "-14px", "left": "50%",
                        "transform": "translateX(-50%)",
                        "background": CARD_BG, "border": f"1px solid {BORDER}",
                        "borderRadius": "50%", "width": "24px", "height": "24px",
                        "cursor": "pointer", "fontSize": "10px", "color": MUTED,
                        "display": "flex", "alignItems": "center", "justifyContent": "center",
                        "padding": "0", "lineHeight": "1",
                    },
                ),
                # Collapsible body
                html.Div(
                    id="info-bar-body",
                    style={"display": "flex", "gap": "10px", "height": "100%", "overflow": "hidden"},
                    children=[
                        # Left: cell info (scrollable)
                        html.Div(
                            id="cell-info-panel",
                            children=[html.Div("Click a cell to see details",
                                               style={"color": MUTED, "fontSize": "13px", "fontStyle": "italic"})],
                            style={"flex": "1", "minWidth": "0", "overflowY": "auto"},
                        ),
                        # Divider
                        html.Div(style={"width": "1px", "backgroundColor": BORDER, "flexShrink": "0"}),
                        # Right: server log + REPL input
                        html.Div([
                            html.Div("SERVER LOG", style={
                                "fontSize": "10px", "color": MUTED, "fontWeight": "600",
                                "letterSpacing": "1px", "marginBottom": "4px", "flexShrink": "0",
                            }),
                            html.Pre(
                                id="server-log",
                                style={
                                    "flex": "1", "margin": "0", "fontSize": "11px", "color": "#8b949e",
                                    "overflowY": "auto", "whiteSpace": "pre-wrap",
                                    "wordBreak": "break-all",
                                    "fontFamily": "'Fira Code','Consolas',monospace",
                                },
                            ),
                            # Python REPL input
                            html.Div([
                                html.Span(">>>", style={
                                    "color": ACCENT, "fontFamily": "'Fira Code','Consolas',monospace",
                                    "fontSize": "12px", "flexShrink": "0", "lineHeight": "28px",
                                }),
                                dcc.Input(
                                    id="repl-input",
                                    type="text",
                                    debounce=False,
                                    placeholder="Python expression or statement (e.g. _run_seurat_annotation(...))",
                                    style={
                                        "flex": "1", "backgroundColor": "#0d1117",
                                        "color": TEXT, "border": "none", "outline": "none",
                                        "fontSize": "12px", "padding": "0 6px",
                                        "fontFamily": "'Fira Code','Consolas',monospace",
                                    },
                                    n_submit=0,
                                ),
                                html.Button("Run", id="repl-run", n_clicks=0, style={
                                    "padding": "2px 10px", "fontSize": "11px", "flexShrink": "0",
                                    "backgroundColor": CARD_BG, "color": TEXT,
                                    "border": f"1px solid {BORDER}", "borderRadius": "4px",
                                    "cursor": "pointer",
                                }),
                            ], style={
                                "display": "flex", "alignItems": "center", "gap": "6px",
                                "borderTop": f"1px solid {BORDER}", "paddingTop": "4px",
                                "flexShrink": "0",
                            }),
                            dcc.Interval(id="log-poll", interval=1500, n_intervals=0),
                        ], style={
                            "flex": "1", "minWidth": "0",
                            "display": "flex", "flexDirection": "column", "overflow": "hidden",
                        }),
                    ],
                ),
            ], style={
                "position": "relative",
                "backgroundColor": CARD_BG, "border": f"1px solid {BORDER}",
                "borderRadius": "6px", "padding": "10px 16px",
                "height": "180px",  # fixed height — both columns scroll independently
                "display": "flex", "flexDirection": "column",
                "boxSizing": "border-box",
            }),
        ], style={
            "flex": "1", "display": "flex", "flexDirection": "column",
            "padding": "12px", "gap": "10px",
            "minWidth": "0", "height": "100vh", "boxSizing": "border-box",
        }),
    ], style={
        "display": "flex", "height": "100vh",
        "backgroundColor": DARK_BG, "color": TEXT,
        "fontFamily": "'Inter','Segoe UI',sans-serif",
    }),
], style={"margin": "0", "padding": "0"})


# ─── Callbacks ────────────────────────────────────────────────────────────────

def _seg_tool(seg_source: str) -> str:
    """Extract tool name from seg-source value: 'xenium', 'baysor', or 'proseg'."""
    return (seg_source or "xenium").split(":")[0]


app.clientside_callback(
    """function(colorBy) {
        var show = {"marginBottom": "14px"};
        var hide = {"marginBottom": "14px", "display": "none"};
        if (colorBy === "cluster") return [show, hide];
        if (colorBy === "gene") return [hide, show];
        return [hide, hide];
    }""",
    Output("cluster-ctrl", "style"),
    Output("gene-ctrl",    "style"),
    Input("color-by", "value"),
)


@app.callback(
    Output("morph-controls", "style"),
    Output("morph-hires-poll", "disabled", allow_duplicate=True),
    Input("morph-enable", "value"),
    prevent_initial_call=True,
)
def toggle_morph_controls(enabled):
    active = "show" in (enabled or [])
    return ({} if active else {"display": "none"}), (not active)


@app.callback(
    Output("spatial-relayout", "data"),
    Input("spatial-plot", "relayoutData"),
    State("spatial-relayout", "data"),
    prevent_initial_call=True,
)
def store_relayout(relayout_data, current):
    if not relayout_data:
        return current or {}
    merged = dict(current or {})
    # Ignore autosize/autorange signals that would override explicit ranges
    ignore = {"autosize", "xaxis.autorange", "yaxis.autorange",
              "xaxis.showspikes", "yaxis.showspikes"}
    for k, v in relayout_data.items():
        if k not in ignore:
            merged[k] = v
    return merged


@app.callback(
    Output("spatial-plot", "figure"),
    Output("umap-plot",    "figure"),
    Input("color-by",           "value"),
    Input("cluster-method",     "value"),
    Input("gene-selector",      "value"),
    Input("point-size",         "value"),
    Input("point-opacity",      "value"),
    Input("boundary-toggles",   "value"),
    Input("annot-done",         "data"),
    Input("morph-enable",       "value"),
    Input("morph-zlevel",       "value"),
    Input("morph-channels",     "value"),
    Input("morph-brightness",   "value"),
    Input("morph-opacity",      "value"),
    Input("spatial-relayout",   "data"),
    Input("baysor-done",        "data"),
    Input("proseg-done",        "data"),
    Input("seg-source",         "value"),
    Input("spage-done",         "data"),
    Input("subset-version",     "data"),
    Input("startup-trigger",        "n_intervals"),
    Input("dataset-version",        "data"),
    Input("extra-datasets-version", "data"),
    Input("split-done",             "data"),
    Input("roi-done",               "data"),
    Input("counts-mode-store",      "data"),
)
def update_plots(color_by, method, gene, size, opacity, boundary_toggles,
                 _annot_done, morph_enable, morph_zlevel, morph_channels,
                 morph_brightness, morph_opacity, relayout,
                 _baysor_done, _proseg_done, seg_source,
                 _spage_done, _subset_ver, _startup, _ds_version, _extra_ds_version,
                 _split_done, _roi_done, counts_mode):
    # If only the viewport changed, avoid full figure rebuild
    triggered = callback_context.triggered_id
    boundaries_active = bool(boundary_toggles)
    morph_active      = "show" in (morph_enable or [])

    if triggered == "spatial-relayout":
        if not boundaries_active and not morph_active:
            # Nothing overlay-related → skip entirely
            return no_update, no_update
        if morph_active and not boundaries_active:
            # Only morphology image needs updating → Patch just layout.images
            morph_image, _ = make_morphology_overlay(
                DATA["data_dir"], relayout or {},
                morph_zlevel or 0, morph_channels or [],
                morph_brightness or 2,
                morph_opacity if morph_opacity is not None else 0.85,
            )
            if morph_image is None:
                # Viewport too large or no valid range yet — preserve existing image
                return no_update, no_update
            patched = Patch()
            patched["layout"]["images"] = [morph_image]
            return patched, no_update

    if triggered in ("morph-brightness", "morph-opacity", "morph-channels",
                     "morph-zlevel") and morph_active:
        # Only morph settings changed → Patch image only, skip scatter rebuild
        morph_image, _ = make_morphology_overlay(
            DATA["data_dir"], relayout or {},
            morph_zlevel or 0, morph_channels or [],
            morph_brightness or 2,
            morph_opacity if morph_opacity is not None else 0.85,
        )
        if morph_image is None:
            return no_update, no_update
        patched = Patch()
        patched["layout"]["images"] = [morph_image]
        return patched, no_update

    color_by         = color_by or "cluster"
    method           = method   or cluster_methods[0]
    gene             = gene     or _default_gene
    size             = size     or 2
    opacity          = opacity  or 0.85
    boundary_toggles = boundary_toggles or []
    seg_source  = seg_source or "xenium"
    baysor_on   = (_seg_tool(seg_source) == "baysor")
    proseg_on   = (_seg_tool(seg_source) == "proseg")

    # Morphology image overlay
    morph_image  = None
    morph_title  = ""
    if "show" in (morph_enable or []):
        morph_image, morph_title = make_morphology_overlay(
            DATA["data_dir"],
            relayout or {},
            morph_zlevel or 0,
            morph_channels or [],
            morph_brightness or 2,
            morph_opacity if morph_opacity is not None else 0.85,
        )

    use_corr = (counts_mode == "corrected")

    # UMAP cache: skip rebuild when only spatial-related inputs changed
    _umap_key = (color_by, method, gene, size, opacity, baysor_on, proseg_on, use_corr,
                 _annot_done, _spage_done, _ds_version, _extra_ds_version,
                 _baysor_done, _proseg_done, _split_done, _subset_ver)
    if _umap_key in _umap_fig_cache:
        umap_fig = _umap_fig_cache[_umap_key]
    else:
        umap_fig = make_umap_fig(color_by, method, gene, size, opacity,
                                 baysor_active=baysor_on, proseg_active=proseg_on,
                                 use_corrected=use_corr)
        _umap_fig_cache.clear()  # keep only latest entry
        _umap_fig_cache[_umap_key] = umap_fig

    return (
        make_spatial_fig(color_by, method, gene, size, opacity,
                         boundary_toggles, relayout,
                         morph_image=morph_image, extra_title=morph_title,
                         baysor_active=baysor_on, proseg_active=proseg_on,
                         use_corrected=use_corr),
        umap_fig,
    )


@app.callback(
    Output("selected-cell", "data"),
    Input("spatial-plot", "clickData"),
    Input("umap-plot",    "clickData"),
    prevent_initial_call=True,
)
def capture_click(spatial_click, umap_click):
    triggered = callback_context.triggered_id
    click = spatial_click if triggered == "spatial-plot" else umap_click
    if not click or not click.get("points"):
        return None
    pt = click["points"][0]
    return pt.get("customdata") or pt.get("text")


@app.callback(
    Output("cell-info-panel", "children"),
    Input("selected-cell", "data"),
    State("cluster-method", "value"),
    prevent_initial_call=True,
)
def show_cell_info(cell_id, method):
    if not cell_id:
        return html.Div("Click a cell to see details",
                        style={"color": MUTED, "fontSize": "13px", "fontStyle": "italic"})

    df = DATA["df"]

    # ── Try Xenium cells first, then Proseg, then Baysor ─────────────────────
    if cell_id in df.index:
        # Original Xenium cell
        row    = df.loc[cell_id]
        method = method or cluster_methods[0]
        col    = cluster_col(method)
        clust  = int(row[col]) if col in row and not pd.isna(row[col]) else "—"

        cell_pos     = df.index.get_loc(cell_id)
        expr_row_idx = DATA["df_to_expr"][cell_pos]
        if expr_row_idx >= 0:
            raw       = DATA["expr"][expr_row_idx, :].toarray().flatten()
            top_idx   = np.argsort(raw)[::-1][:12]
            top_genes = [(DATA["gene_names"][i], int(raw[i])) for i in top_idx if raw[i] > 0]
        else:
            top_genes = []

        umap_1 = row.get("umap_1", float("nan"))
        umap_2 = row.get("umap_2", float("nan"))

        chips = [
            info_chip("Cluster",     str(clust)),
            info_chip("X",           f"{row['x_centroid']:.1f} µm"),
            info_chip("Y",           f"{row['y_centroid']:.1f} µm"),
            info_chip("Transcripts", str(int(row["transcript_counts"]))),
            info_chip("Cell Area",   f"{row['cell_area']:.0f} µm²"),
            info_chip("Nuc Area",    f"{row['nucleus_area']:.0f} µm²"),
            info_chip("UMAP 1",      f"{umap_1:.2f}" if not np.isnan(umap_1) else "—"),
            info_chip("UMAP 2",      f"{umap_2:.2f}" if not np.isnan(umap_2) else "—"),
        ]
        source_tag = ""

    else:
        # Try reseg sources (Proseg cell IDs are ints, Baysor are strings)
        row       = None
        alt_expr  = None
        source_tag = ""
        for src_name, state, lock in (
            ("Proseg", _proseg_state, _proseg_lock),
            ("Baysor", _baysor_state, _baysor_lock),
        ):
            with lock:
                res = state.get("result")
            if res is None:
                continue
            bdf = res["cells_df"]
            # Try direct match first, then type-coerced match
            key = cell_id
            if key not in bdf.index:
                try:
                    key = int(cell_id)
                except (ValueError, TypeError):
                    pass
            if key in bdf.index:
                row        = bdf.loc[key]
                alt_expr   = res.get("expr")
                source_tag = f" [{src_name}]"
                break

        if row is None:
            return html.Div(f"Cell '{cell_id}' not found.", style={"color": "#f85149"})

        # Top genes from reseg expression matrix
        top_genes = []
        if alt_expr is not None:
            gene_names_list = list(DATA["gene_names"])
            # Find row index in reseg cells_df — always use integer position to avoid
            # boolean-array issues when cell IDs are non-unique (multi-patch Proseg runs)
            bdf = res["cells_df"]
            key_for_idx = key
            matches = np.where(bdf.index == key_for_idx)[0]
            if len(matches) == 0:
                alt_expr = None
            else:
                row_idx = int(matches[0])
                if row_idx >= alt_expr.shape[0]:
                    alt_expr = None
        if alt_expr is not None:
            raw     = alt_expr[row_idx, :].toarray().flatten()
            top_idx = np.argsort(raw)[::-1][:12]
            top_genes = [(gene_names_list[i], int(raw[i])) for i in top_idx if raw[i] > 0]

        chips = [
            info_chip("X",           f"{row['x_centroid']:.1f} µm"),
            info_chip("Y",           f"{row['y_centroid']:.1f} µm"),
            info_chip("Transcripts", str(int(row["transcript_counts"]))),
        ]
        if "cell_area" in row.index:
            chips.append(info_chip("Cell Area", f"{row['cell_area']:.0f} µm²"))

    # ── Bar chart ─────────────────────────────────────────────────────────────
    bar_fig = go.Figure(go.Bar(
        x=[g for g, _ in top_genes],
        y=[v for _, v in top_genes],
        marker_color=ACCENT,
    ))
    bar_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=40, r=10, t=5, b=55), height=150,
        xaxis=dict(tickfont=dict(size=10), color=TEXT),
        yaxis=dict(title="Counts", color=TEXT, tickfont=dict(size=10)),
        font=dict(color=TEXT),
    )

    return html.Div([
        html.Div([
            html.Span("Cell: ", style={"color": MUTED, "fontSize": "12px"}),
            html.Span(str(cell_id) + source_tag,
                      style={"color": ACCENT, "fontWeight": "700", "fontSize": "13px"}),
        ], style={"marginBottom": "8px"}),

        html.Div(chips, style={"display": "flex", "gap": "8px", "flexWrap": "wrap",
                               "marginBottom": "10px"}),

        html.Div("Top Expressed Genes",
                 style={"fontSize": "11px", "color": MUTED, "marginBottom": "4px"}),
        dcc.Graph(figure=bar_fig, config={"displayModeBar": False}),
    ])


@app.callback(
    Output("spage-modal", "is_open"),
    Input("spage-modal-open-btn", "n_clicks"),
    Input("spage-modal-close-btn", "n_clicks"),
    State("spage-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_spage_modal(open_clicks, close_clicks, is_open):
    triggered = callback_context.triggered_id
    if triggered == "spage-modal-open-btn":
        return True
    if triggered == "spage-modal-close-btn":
        return False
    return is_open


@app.callback(
    Output("annot-modal", "is_open"),
    Input("annot-modal-open-btn", "n_clicks"),
    Input("annot-modal-close-btn", "n_clicks"),
    State("annot-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_annot_modal(open_clicks, close_clicks, is_open):
    triggered = callback_context.triggered_id
    if triggered == "annot-modal-open-btn":
        return True
    if triggered == "annot-modal-close-btn":
        return False
    return is_open


app.clientside_callback(
    """function(source) {
        var show = {};
        var hide = {"display": "none"};
        if (source === "celltypist") return [show, hide, hide];
        if (source === "rctd") return [hide, show, show];
        return [hide, show, hide];
    }""",
    Output("annot-celltypist-div", "style"),
    Output("annot-seurat-div",     "style"),
    Output("annot-rctd-div",       "style"),
    Input("annot-source",          "value"),
    prevent_initial_call=True,
)


@app.callback(
    Output("annot-poll",         "disabled"),
    Output("annot-status",       "children"),
    Output("annot-btn",          "disabled"),
    Output("annot-modal-status", "children"),
    Input("annot-btn",           "n_clicks"),
    State("annot-source",        "value"),
    State("annot-model",         "value"),
    State("annot-rds-path",      "value"),
    State("annot-label-col",     "value"),
    State("annot-rctd-mode",     "value"),
    State("annot-rctd-cores",    "value"),
    State("annot-rctd-umi-min",       "value"),
    State("annot-rctd-umi-min-sigma", "value"),
    State("seg-source",               "value"),
    prevent_initial_call=True,
)
def start_annotation(n_clicks, source, model_name, rds_path, label_col,
                     rctd_mode, rctd_cores, rctd_umi_min, rctd_umi_min_sigma, seg_source):
    """Kick off background annotation thread."""
    seg_source = seg_source or "xenium"
    with _proseg_lock:
        pres = _proseg_state["result"] if _seg_tool(seg_source) == "proseg" and _proseg_state["status"] == "done" else None
    with _baysor_lock:
        bres = _baysor_state["result"] if _seg_tool(seg_source) == "baysor" and _baysor_state["status"] == "done" else None
    alt_res = pres or bres

    if alt_res is not None:
        if alt_res.get("expr") is None:
            msg = "✗ No expression matrix for reseg cells — re-run Proseg/Baysor"
            return False, msg, False, msg
        expr_override     = alt_res["expr"]
        cell_ids_override = [str(c) for c in alt_res["cells_df"].index]
        pass
    else:
        expr_override     = None
        cell_ids_override = None

    method     = source if source in _ANNOT_METHODS else "celltypist"
    labels_key = _labels_key_for_method(method, alt_res)

    if method == "rctd":
        mode_labels_key = f"{labels_key}_{rctd_mode or 'full'}"
    else:
        mode_labels_key = labels_key

    with _annot_lock:
        if _annot_state["status"] == "running":
            return False, "Already running…", True, "Already running…"
        _annot_state.update({"status": "running", "message": "Starting…",
                              labels_key: None, mode_labels_key: None})

    # Ensure rpy2 conversion rules are set in this callback thread BEFORE copy_context()
    # so the spawned thread's context copy includes them
    if source in ("seurat", "rctd"):
        import rpy2.robjects.conversion as _rconv
        import rpy2.robjects as _ro_mod
        _rconv.set_conversion(_ro_mod.default_converter)

    if source == "seurat":
        ctx = contextvars.copy_context()
        threading.Thread(
            target=lambda: ctx.run(
                _run_seurat_annotation, rds_path or "", label_col or "Names",
                labels_key=labels_key, expr_override=expr_override,
                cell_ids_override=cell_ids_override,
            ),
            daemon=True,
        ).start()
    elif source == "rctd":
        ctx = contextvars.copy_context()
        threading.Thread(
            target=lambda: ctx.run(
                _run_rctd_annotation,
                rds_path or "", label_col or "Names",
                rctd_mode or "full", int(rctd_cores or 4),
                labels_key=labels_key, expr_override=expr_override,
                cell_ids_override=cell_ids_override,
                umi_min=int(rctd_umi_min or 20),
                umi_min_sigma=int(rctd_umi_min_sigma or 100),
                mode_labels_key=mode_labels_key,
            ),
            daemon=True,
        ).start()
    else:
        threading.Thread(
            target=_run_celltypist,
            args=(model_name,),
            kwargs={"labels_key": labels_key, "expr_override": expr_override,
                    "cell_ids_override": cell_ids_override},
            daemon=True,
        ).start()
    return False, "Starting…", True, "Starting…"


@app.callback(
    Output("annot-status",       "children",   allow_duplicate=True),
    Output("annot-poll",         "disabled",   allow_duplicate=True),
    Output("annot-btn",          "disabled",   allow_duplicate=True),
    Output("color-by",           "options"),
    Output("annot-done",         "data"),
    Output("annot-modal-status", "children",   allow_duplicate=True),
    Output("annot-modal",        "is_open",    allow_duplicate=True),
    Input("annot-poll",          "n_intervals"),
    State("color-by",            "options"),
    State("annot-done",          "data"),
    State("annot-modal",         "is_open"),
    prevent_initial_call=True,
)
def poll_annotation(_, current_options, done_version, modal_open):
    """Poll annotation state and update UI when done."""
    with _annot_lock:
        status  = _annot_state["status"]
        message = _annot_state["message"]
        done    = (status == "done")
        error   = (status == "error")

    # Enable each cell_type:method option if that method has any labels
    with _annot_lock:
        methods_with_labels = {
            m for m in _ANNOT_METHODS
            if any(k.startswith(f"labels_{m}") and v is not None
                   for k, v in _annot_state.items())
        }
    options = []
    for opt in current_options:
        val = opt["value"]
        if val.startswith("cell_type:"):
            m = val.split(":")[1]
            options.append({**opt, "disabled": m not in methods_with_labels})
        else:
            options.append(opt)

    if done:
        # Close the modal and show status in the sidebar
        return f"✓ {message}", True, False, options, (done_version or 0) + 1, f"✓ {message}", False
    if error:
        # Keep the modal open so user can see the error
        return f"✗ {message}", True, False, options, done_version, f"✗ {message}", modal_open
    return message, False, True, options, done_version, message, modal_open   # keep polling


app.clientside_callback(
    """function(n_clicks) {
        if (n_clicks % 2 === 1) return [{"display": "none"}, "Show UMAP"];
        return [{"flex": "1", "minWidth": "0"}, "Hide UMAP"];
    }""",
    Output("umap-panel",  "style"),
    Output("umap-toggle", "children"),
    Input("umap-toggle",  "n_clicks"),
)


# ─── Baysor callbacks ─────────────────────────────────────────────────────────

app.clientside_callback(
    """function(usePrior) {
        return (usePrior && usePrior.indexOf("yes") !== -1) ? {} : {"display": "none"};
    }""",
    Output("baysor-prior-conf-div", "style"),
    Input("baysor-use-prior", "value"),
)


@app.callback(
    Output("baysor-xmin", "value"),
    Output("baysor-xmax", "value"),
    Output("baysor-ymin", "value"),
    Output("baysor-ymax", "value"),
    Input("baysor-use-viewport", "n_clicks"),
    State("spatial-relayout", "data"),
    prevent_initial_call=True,
)
def baysor_fill_viewport(_, relayout):
    r = relayout or {}
    x0 = r.get("xaxis.range[0]")
    x1 = r.get("xaxis.range[1]")
    y0 = r.get("yaxis.range[0]")  # negated in plot; y_plot = -y_µm
    y1 = r.get("yaxis.range[1]")
    if None in (x0, x1, y0, y1):
        return no_update, no_update, no_update, no_update
    # Convert plot Y back to µm (plot y = -y_µm)
    ym_min = round(-float(y1), 1)
    ym_max = round(-float(y0), 1)
    return round(float(x0), 1), round(float(x1), 1), ym_min, ym_max


@app.callback(
    Output("baysor-status",  "children"),
    Output("baysor-poll",    "disabled",  allow_duplicate=True),
    Output("baysor-done",    "data"),
    Output("seg-source",     "value",     allow_duplicate=True),
    Input("baysor-poll",     "n_intervals"),
    State("baysor-done",     "data"),
    prevent_initial_call=True,
)
def poll_baysor(_, done_version):
    with _baysor_lock:
        status  = _baysor_state["status"]
        message = _baysor_state["message"]

    if status == "done":
        out_dir = (_baysor_state.get("result") or {}).get("out_dir", "")
        param_tag = os.path.basename(out_dir).split("_")[-1] if out_dir else ""
        seg_val = f"baysor:{param_tag}" if param_tag else "baysor"
        return f"✓ {message}", True, (done_version or 0) + 1, seg_val
    if status == "error":
        return f"✗ {message}", True, done_version, no_update
    return message, False, done_version, no_update


# ─── Proseg callbacks ─────────────────────────────────────────────────────────

@app.callback(
    Output("proseg-xmin", "value"),
    Output("proseg-xmax", "value"),
    Output("proseg-ymin", "value"),
    Output("proseg-ymax", "value"),
    Input("proseg-use-viewport", "n_clicks"),
    State("spatial-relayout", "data"),
    prevent_initial_call=True,
)
def proseg_fill_viewport(_, relayout):
    r = relayout or {}
    x0 = r.get("xaxis.range[0]")
    x1 = r.get("xaxis.range[1]")
    y0 = r.get("yaxis.range[0]")
    y1 = r.get("yaxis.range[1]")
    if None in (x0, x1, y0, y1):
        return no_update, no_update, no_update, no_update
    ym_min = round(-float(y1), 1)
    ym_max = round(-float(y0), 1)
    return round(float(x0), 1), round(float(x1), 1), ym_min, ym_max


@app.callback(
    Output("proseg-status",  "children"),
    Output("proseg-poll",    "disabled",  allow_duplicate=True),
    Output("proseg-done",    "data"),
    Output("seg-source",     "value",     allow_duplicate=True),
    Input("proseg-poll",     "n_intervals"),
    State("proseg-done",     "data"),
    prevent_initial_call=True,
)
def poll_proseg(_, done_version):
    with _proseg_lock:
        status  = _proseg_state["status"]
        message = _proseg_state["message"]

    if status == "done":
        out_dir = (_proseg_state.get("result") or {}).get("out_dir", "")
        param_tag = os.path.basename(out_dir).split("_")[-1] if out_dir else ""
        seg_val = f"proseg:{param_tag}" if param_tag else "proseg"
        return f"✓ {message}", True, (done_version or 0) + 1, seg_val
    if status == "error":
        return f"✗ {message}", True, done_version, no_update
    return message, False, done_version, no_update


# ─── Resegmentation modal callbacks ───────────────────────────────────────────

@app.callback(
    Output("reseg-modal",              "is_open"),
    Output("reseg-patches-confirmed",  "data",     allow_duplicate=True),
    Output("reseg-modal-run-btn",      "disabled",  allow_duplicate=True),
    Output("reseg-patch-confirm-div",  "style",     allow_duplicate=True),
    Output("reseg-patch-confirm-msg",  "children",  allow_duplicate=True),
    Output("reseg-step2-content",      "style",     allow_duplicate=True),
    Output("reseg-step2-overlay",      "style",     allow_duplicate=True),
    Input("reseg-modal-open-btn",  "n_clicks"),
    Input("reseg-modal-close-btn", "n_clicks"),
    State("reseg-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_reseg_modal(open_clicks, close_clicks, is_open):
    triggered = callback_context.triggered_id
    if triggered == "reseg-modal-open-btn" and open_clicks:
        # Check if patches already exist from a previous run / cache
        with _sdata_lock:
            has_patches = _sdata_state.get("patches") is not None
        if has_patches:
            with _sdata_lock:
                n_patches = len(_sdata_state["patches"])
            confirm_msg = (f"✓ {n_patches} patches ready — review ROI and patches on the "
                           f"spatial plot, then confirm to proceed.")
            # Patches exist: show confirm div, keep Step 2 locked until user confirms
            return True, 0, True, {}, confirm_msg, {"opacity": "0.4", "pointerEvents": "none"}, {"display": "block"}
        # No patches yet: hide confirm div, lock Step 2
        return True, 0, True, {"display": "none"}, "", {"opacity": "0.4", "pointerEvents": "none"}, {"display": "block"}
    return False, no_update, no_update, no_update, no_update, no_update, no_update


@app.callback(
    Output("modal-baysor-panel", "style"),
    Output("modal-proseg-panel", "style"),
    Output("reseg-modal-run-btn", "style"),
    Input("reseg-algo-tabs", "value"),
    prevent_initial_call=True,
)
def switch_reseg_tab(algo):
    show = {}
    hide = {"display": "none"}
    run_btn_baysor_style = {"padding": "7px 20px", "backgroundColor": "#1f6feb", "color": "#fff",
                            "border": "none", "borderRadius": "5px", "cursor": "pointer",
                            "fontSize": "13px", "fontWeight": "600"}
    run_btn_proseg_style = {**run_btn_baysor_style, "backgroundColor": "#2d6a4f"}
    if algo == "proseg":
        return hide, show, run_btn_proseg_style
    return show, hide, run_btn_baysor_style


@app.callback(
    Output("reseg-modal-run-status", "children"),
    Output("reseg-modal", "is_open", allow_duplicate=True),
    Output("baysor-poll", "disabled", allow_duplicate=True),
    Output("proseg-poll", "disabled", allow_duplicate=True),
    Input("reseg-modal-run-btn", "n_clicks"),
    State("reseg-algo-tabs", "value"),
    State("reseg-patches-confirmed", "data"),
    # Baysor states
    State("baysor-xmin", "value"),
    State("baysor-xmax", "value"),
    State("baysor-ymin", "value"),
    State("baysor-ymax", "value"),
    State("baysor-scale", "value"),
    State("baysor-scale-std", "value"),
    State("baysor-min-mol", "value"),
    State("baysor-n_clusters", "value"),
    State("baysor-use-patches", "value"),
    State("baysor-use-prior", "value"),
    State("baysor-prior-conf", "value"),
    # Proseg states
    State("proseg-xmin", "value"),
    State("proseg-xmax", "value"),
    State("proseg-ymin", "value"),
    State("proseg-ymax", "value"),
    State("proseg-voxel-size", "value"),
    State("proseg-nthreads", "value"),
    State("proseg-samples", "value"),
    State("proseg-recorded-samples", "value"),
    State("proseg-schedule", "value"),
    State("proseg-nuclear-reassign-prob", "value"),
    State("proseg-prior-seg-prob", "value"),
    prevent_initial_call=True,
)
def run_reseg_modal(n_clicks, algo, patches_confirmed,
                    bxmin, bxmax, bymin, bymax, bscale, bscale_std, bmin_mol, bn_clusters, buse_patches, buse_prior, bprior_conf,
                    pxmin, pxmax, pymin, pymax, pvoxel, pthreads, psamples,
                    precorded_samples, pschedule, pnuclear_prob, pprior_seg_prob):
    if not patches_confirmed:
        return "✗ Please confirm patches in Step 1 before running segmentation.", no_update, no_update, no_update
    if algo == "baysor":
        with _baysor_lock:
            if _baysor_state["status"] == "running":
                return "Already running…", True, False, no_update
            _baysor_state.update({"status": "running", "message": "Starting…", "result": None})
        scale      = float(bscale) if bscale is not None else 20.0
        scale_std  = float(bscale_std) if bscale_std is not None else None
        min_mol     = int(bmin_mol    or 10)
        n_clusters  = int(bn_clusters or 10)
        use_patches = "yes" in (buse_patches or [])
        prior_conf  = float(bprior_conf or 0.5)
        use_prior_bool = "yes" in (buse_prior or [])
        region = {k: v for k, v in
                  dict(x_min=bxmin, x_max=bxmax, y_min=bymin, y_max=bymax).items()
                  if v is not None}
        threading.Thread(
            target=_run_baysor,
            args=(scale, min_mol, use_prior_bool, prior_conf),
            kwargs=dict(**region, scale_std=scale_std, n_clusters=n_clusters,
                        use_patches=use_patches),
            daemon=True,
        ).start()
        return "Baysor starting…", False, False, True
    else:  # proseg
        with _proseg_lock:
            if _proseg_state["status"] == "running":
                return "Already running…", True, no_update, False
            _proseg_state.update({"status": "running", "message": "Starting…", "result": None})
        region = {k: v for k, v in
                  dict(x_min=pxmin, x_max=pxmax, y_min=pymin, y_max=pymax).items()
                  if v is not None}
        threading.Thread(
            target=_run_proseg,
            kwargs=dict(voxel_size=pvoxel, n_threads=pthreads,
                        n_samples=int(psamples) if psamples else 10,
                        recorded_samples=int(precorded_samples) if precorded_samples else None,
                        schedule=pschedule.strip() if pschedule else None,
                        nuclear_reassign_prob=float(pnuclear_prob) if pnuclear_prob is not None else None,
                        prior_seg_prob=float(pprior_seg_prob) if pprior_seg_prob is not None else None,
                        **region),
            daemon=True,
        ).start()
        return "Proseg starting…", False, no_update, False


@app.callback(
    Output("reseg-status", "children"),
    Input("baysor-done", "data"),
    Input("proseg-done", "data"),
    prevent_initial_call=True,
)
def update_reseg_status(baysor_done, proseg_done):
    with _baysor_lock:
        b_status = _baysor_state["status"]
        b_msg    = _baysor_state["message"]
    with _proseg_lock:
        p_status = _proseg_state["status"]
        p_msg    = _proseg_state["message"]
    if b_status == "done":
        return f"\u2713 Baysor: {b_msg}"
    if p_status == "done":
        return f"\u2713 Proseg: {p_msg}"
    return ""


# ─── Reseg UMAP callbacks ─────────────────────────────────────────────────────

@app.callback(
    Output("umap-reseg-poll",    "disabled"),
    Output("umap-reseg-status",  "children"),
    Output("umap-reseg-btn",     "disabled"),
    Input("umap-reseg-btn",      "n_clicks"),
    State("counts-mode-store",   "data"),
    prevent_initial_call=True,
)
def start_reseg_umap(_, counts_mode):
    with _umap_reseg_lock:
        if _umap_reseg_state["status"] == "running":
            return False, "Already running…", True
        _umap_reseg_state.update({"status": "running", "message": "Starting…", "result": None,
                                  "_xenium_bumped": False,
                                  "counts_mode": counts_mode or "original"})
    threading.Thread(target=_run_reseg_umap, daemon=True).start()
    return False, "Starting UMAP…", True


@app.callback(
    Output("umap-reseg-status",  "children",  allow_duplicate=True),
    Output("umap-reseg-poll",    "disabled",  allow_duplicate=True),
    Output("umap-reseg-btn",     "disabled",  allow_duplicate=True),
    Output("dataset-version",    "data",      allow_duplicate=True),
    Input("umap-reseg-poll",     "n_intervals"),
    State("dataset-version",     "data"),
    prevent_initial_call=True,
)
def poll_reseg_umap(_, version):
    with _umap_reseg_lock:
        status   = _umap_reseg_state["status"]
        message  = _umap_reseg_state["message"]
        bumped   = _umap_reseg_state.get("_xenium_bumped", True)
    if status == "done":
        if not bumped:
            with _umap_reseg_lock:
                _umap_reseg_state["_xenium_bumped"] = True
            return f"✓ {message}", True, False, (version or 0) + 1
        return f"✓ {message}", True, False, no_update
    if status == "error":
        return f"✗ {message}", True, False, no_update
    return message, False, True, no_update


# ─── Boundary overlay options: enable Proseg/Baysor when results are ready ────

# ─── Seg-source delete run callbacks ──────────────────────────────────────────

app.clientside_callback(
    """function(seg_val) {
        if (seg_val && seg_val.includes(":")) {
            return {"display": "block", "marginBottom": "14px"};
        }
        return {"display": "none"};
    }""",
    Output("seg-delete-div",  "style"),
    Input("seg-source",       "value"),
)


@app.callback(
    Output("seg-delete-modal",        "is_open"),
    Output("seg-delete-confirm-text", "children"),
    Input("seg-delete-btn",           "n_clicks"),
    Input("seg-delete-cancel-btn",    "n_clicks"),
    State("seg-source",               "value"),
    prevent_initial_call=True,
)
def toggle_seg_delete_modal(delete_clicks, cancel_clicks, seg_value):
    if callback_context.triggered_id == "seg-delete-btn" and delete_clicks:
        runs  = _list_cached_seg_runs()
        run   = next((r for r in runs if r["value"] == seg_value), None)
        label = run["label"] if run else seg_value
        return True, f"Permanently delete '{label}'? The cache directory will be removed and cannot be recovered."
    return False, no_update


@app.callback(
    Output("seg-delete-modal", "is_open",  allow_duplicate=True),
    Output("seg-source",       "value",    allow_duplicate=True),
    Output("baysor-done",      "data",     allow_duplicate=True),
    Output("proseg-done",      "data",     allow_duplicate=True),
    Input("seg-delete-confirm-btn", "n_clicks"),
    State("seg-source",             "value"),
    State("baysor-done",            "data"),
    State("proseg-done",            "data"),
    prevent_initial_call=True,
)
def confirm_seg_delete(n_clicks, seg_value, baysor_ver, proseg_ver):
    if not n_clicks:
        return no_update, no_update, no_update, no_update
    import shutil
    runs = _list_cached_seg_runs()
    run  = next((r for r in runs if r["value"] == seg_value), None)
    if run:
        out_dir = run["out_dir"]
        try:
            shutil.rmtree(out_dir)
            print(f"  Deleted seg run: {os.path.basename(out_dir)}", flush=True)
        except Exception as e:
            print(f"  Delete failed: {e}", flush=True)
        # Clear in-memory state if this was the loaded run
        tool = _seg_tool(seg_value)
        if tool == "baysor":
            with _baysor_lock:
                if (_baysor_state.get("result") or {}).get("out_dir", "") == out_dir:
                    _baysor_state.update({"status": "idle", "message": "", "result": None})
        elif tool == "proseg":
            with _proseg_lock:
                if (_proseg_state.get("result") or {}).get("out_dir", "") == out_dir:
                    _proseg_state.update({"status": "idle", "message": "", "result": None})
    return False, "xenium", (baysor_ver or 0) + 1, (proseg_ver or 0) + 1


@app.callback(
    Output("boundary-toggles", "options"),
    Output("seg-source",       "options"),
    Input("baysor-done",       "data"),
    Input("proseg-done",       "data"),
    State("boundary-toggles",  "options"),
)
def update_boundary_options(baysor_done, proseg_done, boundary_opts):
    baysor_ready = _baysor_state["status"] == "done" and _baysor_state["result"] is not None
    proseg_ready = _proseg_state["status"] == "done" and _proseg_state["result"] is not None

    # Rebuild boundary options (unchanged logic)
    updated_boundary = []
    for opt in boundary_opts:
        v = opt["value"]
        if v == "proseg":
            updated_boundary.append({**opt, "disabled": not proseg_ready})
        elif v == "baysor":
            updated_boundary.append({**opt, "disabled": not baysor_ready})
        else:
            updated_boundary.append(opt)

    # Rebuild seg-source options dynamically from cache
    seg_opts = [{"label": "\u2605 Xenium (original)", "value": "xenium"}]
    cached = _list_cached_seg_runs()
    baysor_cached = [c for c in cached if c["value"].startswith("baysor:")]
    proseg_cached  = [c for c in cached if c["value"].startswith("proseg:")]
    current_baysor_dir = (_baysor_state.get("result") or {}).get("out_dir", "")
    current_proseg_dir = (_proseg_state.get("result") or {}).get("out_dir", "")
    for c in baysor_cached:
        is_current = (c["out_dir"] == current_baysor_dir)
        label = ("\u2605 " if is_current else "") + c["label"]
        seg_opts.append({"label": label, "value": c["value"]})
    for c in proseg_cached:
        is_current = (c["out_dir"] == current_proseg_dir)
        label = ("\u2605 " if is_current else "") + c["label"]
        seg_opts.append({"label": label, "value": c["value"]})

    return updated_boundary, seg_opts



@app.callback(
    Output("baysor-done",  "data",     allow_duplicate=True),
    Output("proseg-done",  "data",     allow_duplicate=True),
    Output("seg-source",   "value",    allow_duplicate=True),
    Output("baysor-poll",  "disabled", allow_duplicate=True),
    Output("proseg-poll",  "disabled", allow_duplicate=True),
    Input("seg-source",    "value"),
    State("baysor-done",   "data"),
    State("proseg-done",   "data"),
    prevent_initial_call=True,
)
def load_seg_run_from_cache(seg_value, baysor_ver, proseg_ver):
    """When user selects a cached run from the dropdown, load it from disk if needed."""
    if not seg_value or ":" not in seg_value:
        return no_update, no_update, no_update, no_update, no_update

    tool, param_tag = seg_value.split(":", 1)
    cache_base = os.path.join(os.path.expanduser("~"), ".xenium_explorer_cache")
    dataset = os.path.basename(DATA.get("data_dir", ""))
    out_dir = os.path.join(cache_base, f"{tool}_{dataset}_{param_tag}")

    if tool == "baysor":
        current_dir = (_baysor_state.get("result") or {}).get("out_dir", "")
        if current_dir == out_dir and _baysor_state["status"] == "done":
            return no_update, no_update, no_update, no_update, no_update
        threading.Thread(target=_load_cached_baysor, args=(out_dir,), daemon=True).start()
        # Enable baysor-poll so it detects the load completion and bumps baysor-done
        return no_update, no_update, no_update, False, no_update
    elif tool == "proseg":
        current_dir = (_proseg_state.get("result") or {}).get("out_dir", "")
        if current_dir == out_dir and _proseg_state["status"] == "done":
            return no_update, no_update, no_update, no_update, no_update
        threading.Thread(target=_load_cached_proseg, args=(out_dir,), daemon=True).start()
        # Enable proseg-poll so it detects the load completion and bumps proseg-done
        return no_update, no_update, no_update, no_update, False
    return no_update, no_update, no_update, no_update, no_update


# ─── SpaGE callbacks ──────────────────────────────────────────────────────────

@app.callback(
    Output("spage-poll",    "disabled"),
    Output("spage-status",       "children"),
    Output("spage-modal-status", "children"),
    Output("spage-run-btn",      "disabled"),
    Input("spage-run-btn",       "n_clicks"),
    State("spage-rds-path",      "value"),
    State("spage-npv",           "value"),
    State("spage-genes",         "value"),
    State("seg-source",          "value"),
    prevent_initial_call=True,
)
def start_spage(n_clicks, rds_path, n_pv, genes_input, seg_source):
    with _spage_lock:
        if _spage_state["status"] == "running":
            return False, "Already running…", "Already running…", True
        _spage_state.update({"status": "running", "message": "Starting…", "result": None})
    rds_path = rds_path or "/Users/ikuz/Documents/XeniumWorkflow/snRV_ref.rds"
    n_pv = int(n_pv or 50)
    genes_str = genes_input or ""
    seg_src = seg_source or "xenium"
    # Copy the current contextvars context so rpy2 conversion rules are available in the thread
    ctx = contextvars.copy_context()
    threading.Thread(
        target=lambda: ctx.run(_run_spage_imputation, rds_path, n_pv, genes_str, seg_src),
        daemon=True,
    ).start()
    return False, "Starting SpaGE…", "Starting SpaGE…", True


@app.callback(
    Output("spage-status",       "children",  allow_duplicate=True),
    Output("spage-modal-status", "children",  allow_duplicate=True),
    Output("spage-modal",        "is_open",   allow_duplicate=True),
    Output("spage-poll",         "disabled",  allow_duplicate=True),
    Output("spage-run-btn",      "disabled",  allow_duplicate=True),
    Output("spage-done",         "data"),
    Input("spage-poll",          "n_intervals"),
    State("spage-done",          "data"),
    State("spage-modal",         "is_open"),
    prevent_initial_call=True,
)
def poll_spage(_, done_version, modal_open):
    with _spage_lock:
        status  = _spage_state["status"]
        message = _spage_state["message"]

    if status == "done":
        return f"✓ {message}", f"✓ {message}", False, True, False, (done_version or 0) + 1
    if status == "error":
        return f"✗ {message}", f"✗ {message}", modal_open, True, False, done_version
    return message, message, modal_open, False, True, done_version


@app.callback(
    Output("spage-status",  "children",  allow_duplicate=True),
    Output("spage-done",    "data",      allow_duplicate=True),
    Input("spage-repl-poll", "n_intervals"),
    State("spage-done",      "data"),
    prevent_initial_call=True,  # required because spage-done has allow_duplicate
)
def poll_spage_repl(_, done_version):
    """Always-on poll: picks up SpaGE runs triggered from the REPL."""
    global _spage_repl_pending, _spage_last_logged
    if not _spage_repl_pending:
        return no_update, no_update
    with _spage_lock:
        status  = _spage_state["status"]
        message = _spage_state["message"]
    if status == "running":
        if message != _spage_last_logged:
            print(f"  SpaGE: {message}", flush=True)
            _spage_last_logged = message
        return message, no_update
    if status == "done":
        _spage_repl_pending = False
        _spage_last_logged  = ""
        print(f"  SpaGE: done — {message}", flush=True)
        print("  Imputed genes are now available in the Color By → Gene Expression dropdown.", flush=True)
        _gene_expr_cache.clear()  # clear gene cache after imputation
        return f"✓ {message}", (done_version or 0) + 1
    if status == "error":
        _spage_repl_pending = False
        _spage_last_logged  = ""
        print(f"  SpaGE error: {message}", flush=True)
        return f"✗ {message}", no_update
    return no_update, no_update


@app.callback(
    Output("gene-selector", "options"),
    Input("spage-done",       "data"),
    Input("dataset-version",  "data"),
    Input("seg-source",       "value"),
)
def update_gene_options(_spage_version, _ds_version, seg_source):
    """Rebuild gene dropdown: native genes + imputed genes after SpaGE.
    Imputed genes are excluded when a reseg source (Baysor/Proseg) is active,
    since SpaGE has only been run against Xenium cells."""
    xenium_active = _seg_tool(seg_source) == "xenium"

    # Separate measured vs imputed genes using gene_var metadata if available
    gene_var = DATA.get("gene_var")
    if gene_var is not None and "is_imputed" in gene_var.columns:
        measured = [g for g in DATA["gene_names"] if not gene_var.loc[g, "is_imputed"]]
    else:
        measured = list(DATA["gene_names"])

    base = [{"label": g, "value": g} for g in sorted(measured)]

    if not xenium_active:
        # Reseg active: show imputed genes specific to this reseg result
        tool = _seg_tool(seg_source)
        if tool == "baysor":
            with _baysor_lock:
                res = _baysor_state.get("result") if _baysor_state["status"] == "done" else None
        elif tool == "proseg":
            with _proseg_lock:
                res = _proseg_state.get("result") if _proseg_state["status"] == "done" else None
        else:
            res = None
        if res:
            imp_genes = res.get("imputed_genes", [])
            if imp_genes:
                imp = [{"label": f"{g} [imp]", "value": f"{g} [imp]"} for g in sorted(imp_genes)]
                return base + imp
        # Streaming SpaGE: genes not written to alt_res["expr"] — fall back to result_genes
        with _spage_lock:
            streaming_genes = _spage_state.get("result_genes") or []
        if streaming_genes:
            imp = [{"label": f"{g} [imp]", "value": f"{g} [imp]"} for g in sorted(streaming_genes)]
            return base + imp
        return base

    # Xenium active: add imputed genes from zarr (already in DATA["expr"]) or _spage_state fallback
    if gene_var is not None and "is_imputed" in gene_var.columns:
        imp_genes = [g for g in DATA["gene_names"] if gene_var.loc[g, "is_imputed"]]
        if imp_genes:
            imp = [{"label": f"{g} [imp]", "value": f"{g} [imp]"} for g in sorted(imp_genes)]
            return base + imp

    # Fallback: old-style _spage_state result DataFrame, or streaming gene list
    with _spage_lock:
        result       = _spage_state.get("result")
        result_genes = _spage_state.get("result_genes") or []
    imp_gene_list = (sorted(result.columns) if result is not None else sorted(result_genes))
    if imp_gene_list:
        imp = [{"label": f"{g} [imp]", "value": f"{g} [imp]"} for g in imp_gene_list]
        return base + imp

    return base


app.clientside_callback(
    """function(n) {
        if (n % 2 === 1) return [{"display": "none"}, "▲"];
        return [{"display": "flex", "gap": "10px", "height": "100%", "overflow": "hidden"}, "▼"];
    }""",
    Output("info-bar-body",    "style"),
    Output("info-bar-toggle",  "children"),
    Input("info-bar-toggle",   "n_clicks"),
    prevent_initial_call=True,
)


@app.callback(
    Output("spatial-plot", "figure", allow_duplicate=True),
    Input("morph-hires-poll", "n_intervals"),
    prevent_initial_call=True,
)
def push_hires_overlay(_):
    """Apply the latest hires tile render to the spatial plot (progressive rendering)."""
    if not _morph_hires_queue:
        return no_update
    img_dict = _morph_hires_queue.pop(0)
    if not img_dict:
        return no_update  # don't clear the preview if hires result is empty
    patched = Patch()
    patched["layout"]["images"] = [img_dict]
    return patched


@app.callback(
    Output("subset-version", "data"),
    Input("subset-poll",     "n_intervals"),
    State("subset-version",  "data"),
)
def poll_subset(_, current):
    """Push _subset_version into the Dash store to trigger plot refresh."""
    if _subset_version != (current or 0):
        return _subset_version
    return no_update


@app.callback(
    Output("subset-indicator", "style"),
    Output("subset-count",     "children"),
    Input("subset-version",    "data"),
)
def update_subset_indicator(_):
    active = "_df_original" in DATA
    if active:
        n_sub  = len(DATA["df"])
        n_orig = len(DATA["_df_original"])
        return {}, f"{n_sub:,} / {n_orig:,} cells"
    return {"display": "none"}, ""


@app.callback(
    Output("subset-version", "data", allow_duplicate=True),
    Input("unsubset-btn",    "n_clicks"),
    prevent_initial_call=True,
)
def clear_subset_btn(_):
    unsubset()
    return _subset_version


@app.callback(
    Output("server-log", "children"),
    Input("log-poll", "n_intervals"),
)
def update_server_log(_):
    with _log_lock:
        lines = list(_log_buffer)
    return "\n".join(lines[-60:])  # last 60 lines


_repl_globals = globals()  # share module namespace with REPL


@app.callback(
    Output("repl-input", "value"),
    Input("repl-run",    "n_clicks"),
    Input("repl-input",  "n_submit"),
    State("repl-input",  "value"),
    prevent_initial_call=True,
)
def run_repl(_, __, code):
    if not code or not code.strip():
        return ""
    with _log_lock:
        _log_buffer.append(f">>> {code}")

    def _exec():
        try:
            # Try eval first (expression), fall back to exec (statement)
            try:
                result = eval(code, _repl_globals)
                if result is not None:
                    print(repr(result), flush=True)
            except SyntaxError:
                exec(code, _repl_globals)
        except Exception:
            import traceback
            traceback.print_exc()

    threading.Thread(target=_exec, daemon=True).start()
    return ""


# ─── SpatialData / Sopa callbacks ─────────────────────────────────────────────

@app.callback(
    Output("sdata-poll",              "disabled",  allow_duplicate=True),
    Output("sdata-status",            "children",  allow_duplicate=True),
    Output("sdata-clear-cache-btn",   "disabled",  allow_duplicate=True),
    Output("sdata-prepare-btn",       "disabled",  allow_duplicate=True),
    Output("reseg-patches-confirmed", "data",      allow_duplicate=True),
    Output("reseg-patch-confirm-div", "style",     allow_duplicate=True),
    Input("sdata-clear-cache-btn",    "n_clicks"),
    State("sdata-qv",                 "value"),
    prevent_initial_call=True,
)
def start_sdata_clear(n_clicks, qv):
    """Delete cache then trigger a forced re-read from raw files."""
    with _sdata_lock:
        if _sdata_state["status"] == "running":
            return False, "Already running…", True, True, no_update, no_update
        _sdata_state.update({"status": "running", "message": "Clearing cache…",
                              "roi": None, "patches": None})
    clear_sdata_cache()
    to_spatialdata(qv_threshold=int(qv or 20), force=True)
    return False, "Re-reading from raw files…", True, True, 0, {"display": "none"}


@app.callback(
    Output("sdata-status",            "children",  allow_duplicate=True),
    Output("sdata-poll",              "disabled",  allow_duplicate=True),
    Output("sdata-clear-cache-btn",   "disabled",  allow_duplicate=True),
    Output("sdata-prepare-btn",       "disabled",  allow_duplicate=True),
    Output("sdata-done",              "data"),
    Output("sdata-show-roi",          "value",     allow_duplicate=True),
    Output("sdata-show-patches",      "value",     allow_duplicate=True),
    Output("reseg-patch-confirm-div", "style",     allow_duplicate=True),
    Output("reseg-patch-confirm-msg", "children",  allow_duplicate=True),
    Input("sdata-poll",               "n_intervals"),
    State("sdata-done",               "data"),
    State("sdata-patch-width",        "value"),
    State("sdata-patch-min-tx",       "value"),
    prevent_initial_call=True,
)
def poll_sdata(_, done_version, patch_width, patch_min_tx):
    with _sdata_lock:
        status      = _sdata_state["status"]
        message     = _sdata_state["message"]
        has_roi     = _sdata_state.get("roi") is not None
        has_patches = _sdata_state.get("patches") is not None
        has_sdata   = _sdata_state.get("sdata") is not None

    HIDE = {"display": "none"}
    SHOW = {}
    no_chg = no_update

    if status == "running":
        # poll=enabled, clear-btn=disabled, prepare-btn=disabled
        return message, False, True, True, no_chg, no_chg, no_chg, HIDE, no_chg

    if status in ("done", "roi_done", "patches_done", "error"):
        # Chain: sdata done → auto-run ROI if not done
        if status == "done" and not has_roi and has_sdata:
            with _sdata_lock:
                _sdata_state.update({"status": "running", "message": "Detecting tissue ROI…"})
            segment_tissue_roi()
            return "Detecting tissue ROI…", False, True, True, no_chg, no_chg, no_chg, HIDE, no_chg

        # Chain: ROI done → auto-run patches if not done
        if has_roi and not has_patches:
            width  = float(patch_width  or 1000)
            min_tx = int(patch_min_tx   or 10)
            with _sdata_lock:
                _sdata_state.update({"status": "running", "message": "Creating patches…"})
            create_patches(width_um=width, min_transcripts=min_tx)
            return "Creating patches…", False, True, True, no_chg, no_chg, no_chg, HIDE, no_chg

        # Patches exist — show overlays + confirmation, re-enable buttons
        if has_patches:
            with _sdata_lock:
                n_patches = len(_sdata_state["patches"])
            confirm_msg = (f"✓ {n_patches} patches ready — review ROI and patches on the "
                           f"spatial plot, then confirm to proceed.")
            return (message, True, False, False, (done_version or 0) + 1,
                    ["yes"], ["yes"], SHOW, confirm_msg)

        # Error — re-enable buttons so user can retry
        if status == "error":
            return f"✗ {message}", True, False, False, no_chg, no_chg, no_chg, HIDE, no_chg

        return message, True, False, False, no_chg, no_chg, no_chg, no_chg, no_chg

    return (no_update, no_update, no_update, no_update, no_update,
            no_update, no_update, no_update, no_update)


@app.callback(
    Output("sdata-poll",             "disabled",    allow_duplicate=True),
    Output("sdata-status",           "children",    allow_duplicate=True),
    Output("sdata-show-roi",         "value",       allow_duplicate=True),
    Output("sdata-show-patches",     "value",       allow_duplicate=True),
    Output("reseg-patch-confirm-div","style",       allow_duplicate=True),
    Output("reseg-patch-confirm-msg","children",    allow_duplicate=True),
    Output("sdata-prepare-btn",      "disabled",    allow_duplicate=True),
    Input("sdata-prepare-btn",       "n_clicks"),
    State("sdata-qv",                "value"),
    prevent_initial_call=True,
)
def start_prepare_patches(n_clicks, qv):
    """User clicked Prepare Patches — start pipeline or show cached result."""
    if not n_clicks:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update
    with _sdata_lock:
        status      = _sdata_state["status"]
        message     = _sdata_state["message"]
        has_patches = _sdata_state.get("patches") is not None
    # Patches already ready — just show overlays and confirmation
    if has_patches:
        with _sdata_lock:
            n_patches = len(_sdata_state["patches"])
        confirm_msg = (f"✓ {n_patches} patches ready — review ROI and patches on the "
                       f"spatial plot, then confirm to proceed.")
        return True, message, ["yes"], ["yes"], {}, confirm_msg, False
    # Already running — just enable poll to show progress
    if status == "running":
        return False, message, no_update, no_update, {"display": "none"}, "", True
    # Start the pipeline: sdata → ROI → patches (chained by poll_sdata)
    with _sdata_lock:
        _sdata_state.update({"status": "running", "message": "Building SpatialData…"})
    to_spatialdata(qv_threshold=int(qv or 20))
    return False, "Building SpatialData…", no_update, no_update, {"display": "none"}, "", True


@app.callback(
    Output("reseg-patches-confirmed", "data"),
    Output("reseg-step2-content",     "style"),
    Output("reseg-step2-overlay",     "style"),
    Output("reseg-modal-run-btn",     "disabled"),
    Input("reseg-confirm-patches-btn", "n_clicks"),
    Input("reseg-skip-patches-btn",    "n_clicks"),
    prevent_initial_call=True,
)
def confirm_patches(n_clicks, skip_clicks):
    if not (n_clicks or skip_clicks):
        return no_update, no_update, no_update, no_update
    return 1, {}, {"display": "none"}, False


@app.callback(
    Output("spatial-plot", "figure", allow_duplicate=True),
    Input("sdata-show-roi",     "value"),
    Input("sdata-show-patches", "value"),
    Input("sdata-done",         "data"),
    prevent_initial_call=True,
)
def update_sdata_overlays(show_roi, show_patches, _sdata_ver):
    roi_on     = "yes" in (show_roi     or [])
    patches_on = "yes" in (show_patches or [])
    shapes = _sdata_overlays_to_shapes(roi_on, patches_on)
    patched = Patch()
    patched["layout"]["shapes"] = shapes
    return patched


# ─── Dataset switcher callbacks ───────────────────────────────────────────────

@app.callback(
    Output("tissue-info-content", "children"),
    Input("dataset-version", "data"),
    Input("seg-source", "value"),
    Input("split-done", "data"),
)
def update_tissue_info(_, seg_source, _split_done):
    meta = DATA.get("metadata", {})
    cache_size = _cache_size_str()

    # Determine active cell count from reseg result if active
    tool = _seg_tool(seg_source)
    if tool == "baysor":
        with _baysor_lock:
            res = _baysor_state.get("result")
        n_cells_display = f"{len(res['cells_df']):,}" if res else f"{meta.get('num_cells', 0):,}"
        seg_label = "Baysor"
    elif tool == "proseg":
        with _proseg_lock:
            res = _proseg_state.get("result")
        n_cells_display = f"{len(res['cells_df']):,}" if res else f"{meta.get('num_cells', 0):,}"
        seg_label = "Proseg"
    else:
        n_cells_display = f"{meta.get('num_cells', 0):,}"
        seg_label = "Xenium"

    return html.Div([
        html.Div([
            html.Div("XENIUM EXPLORER", style={
                "fontSize": "11px", "fontWeight": "700",
                "letterSpacing": "2px", "color": ACCENT, "marginBottom": "4px",
            }),
            html.Div(meta.get("run_name", "\u2014"),
                     style={"fontSize": "13px", "color": TEXT, "fontWeight": "600"}),
            html.Div(meta.get("region_name", ""),
                     style={"fontSize": "11px", "color": MUTED}),
        ], style={"marginBottom": "10px"}),

        html.Div([
            html.Hr(style={"borderColor": BORDER, "margin": "0 0 10px 0"}),
            html.Div([
                stat_row("Cells",       n_cells_display),
                stat_row("Seg",         seg_label),
                stat_row("Transcripts", f"{meta.get('num_transcripts', 0):,}"),
                stat_row("Panel",       meta.get("panel_name", "\u2014")),
                stat_row("Genes",       str(len(DATA.get("gene_names", [])))),
                stat_row("Tissue",      meta.get("panel_tissue_type", "\u2014")),
                stat_row("Pixel",       f"{meta.get('pixel_size', PIXEL_SIZE_UM)} \u00b5m"),
            ], style={"marginBottom": "10px"}),
            html.Hr(style={"borderColor": BORDER, "margin": "0 0 10px 0"}),
            # Sample switcher
            html.Button("\U0001f4c2 Load Different Sample", id="sample-modal-open-btn", n_clicks=0,
                        style={"width": "100%", "padding": "5px 0",
                               "backgroundColor": CARD_BG, "color": TEXT,
                               "border": f"1px solid {BORDER}", "borderRadius": "4px",
                               "cursor": "pointer", "fontSize": "11px", "marginBottom": "4px"}),
            html.Button("\U0001f4be Save as SpatialData", id="save-sdata-open-btn", n_clicks=0,
                        style={"width": "100%", "padding": "5px 0",
                               "backgroundColor": CARD_BG, "color": TEXT,
                               "border": f"1px solid {BORDER}", "borderRadius": "4px",
                               "cursor": "pointer", "fontSize": "11px", "marginBottom": "4px"}),
            html.Button("\u2716 Remove Extra Samples", id="remove-extra-samples-btn", n_clicks=0,
                        style={"width": "100%", "padding": "5px 0",
                               "backgroundColor": CARD_BG, "color": MUTED,
                               "border": f"1px solid {BORDER}", "borderRadius": "4px",
                               "cursor": "pointer", "fontSize": "11px", "marginBottom": "8px"}),
            html.Div(id="extra-samples-status",
                     style={"fontSize": "10px", "color": MUTED, "marginBottom": "4px", "minHeight": "14px"}),
            # Cache info
            html.Div([
                html.Span(f"Cache: {cache_size}",
                          id="cache-size-display",
                          style={"fontSize": "11px", "color": MUTED}),
                html.Button("Clean", id="cache-clean-open-btn", n_clicks=0,
                            style={"marginLeft": "8px", "padding": "1px 8px",
                                   "backgroundColor": CARD_BG, "color": MUTED,
                                   "border": f"1px solid {BORDER}", "borderRadius": "3px",
                                   "cursor": "pointer", "fontSize": "10px"}),
            ], style={"display": "flex", "alignItems": "center"}),
        ], className="tissue-details"),
    ], id="tissue-info-section", style={"marginBottom": "14px"})


@app.callback(
    Output("sample-modal", "is_open"),
    Output("sample-picker", "options"),
    Input("sample-modal-open-btn", "n_clicks"),
    Input("sample-modal-close-btn", "n_clicks"),
    State("sample-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_sample_modal(open_clicks, close_clicks, is_open):
    if callback_context.triggered_id == "sample-modal-open-btn" and open_clicks:
        return True, _available_samples()
    return False, no_update


@app.callback(
    Output("sample-load-status",  "children"),
    Output("sample-modal",        "is_open",       allow_duplicate=True),
    Output("dataset-version",     "data",          allow_duplicate=True),
    Input("sample-load-btn",      "n_clicks"),
    State("sample-picker",        "value"),
    State("sample-custom-path",   "value"),
    State("dataset-version",      "data"),
    prevent_initial_call=True,
)
def load_sample(n_clicks, picked_path, custom_path, ds_version):
    new_dir = (custom_path or "").strip() or picked_path
    if not new_dir:
        return "Select a sample or enter a path.", no_update, no_update
    new_dir = os.path.abspath(new_dir)
    if not os.path.exists(os.path.join(new_dir, "experiment.xenium")):
        return f"Not a valid Xenium directory: {new_dir}", no_update, no_update
    if new_dir == DATA["data_dir"]:
        return "Already loaded.", no_update, no_update
    try:
        new_data = load_xenium_data(new_dir)
        DATA.update(new_data)
        # Reset all per-dataset state
        with _baysor_lock:
            _baysor_state.update({"status": "idle", "message": "", "result": None})
        with _proseg_lock:
            _proseg_state.update({"status": "idle", "message": "", "result": None})
        with _annot_lock:
            _annot_state.update({"status": "idle", "message": "", "labels": None,
                                  "labels_proseg": None, "labels_baysor": None})
        with _spage_lock:
            _spage_state.update({"status": "idle", "message": "", "result": None})
        with _sdata_lock:
            _sdata_state.update({"status": "idle", "message": "", "sdata": None,
                                  "roi": None, "patches": None})
        with _umap_reseg_lock:
            _umap_reseg_state.update({"status": "idle", "message": "", "result": None})
        # Re-run autoloads for the new dataset
        _annot_autoload()
        _sdata_autoload()
        _spage_autoload()
        print(f"  Loaded new dataset: {new_dir}", flush=True)
        return "Loaded.", False, (ds_version or 0) + 1
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return f"Error: {exc}", no_update, no_update


@app.callback(
    Output("sample-load-status",       "children",  allow_duplicate=True),
    Output("sample-modal",             "is_open",   allow_duplicate=True),
    Output("extra-datasets-version",   "data",      allow_duplicate=True),
    Output("extra-samples-status",     "children",  allow_duplicate=True),
    Input("sample-add-btn",            "n_clicks"),
    State("sample-picker",             "value"),
    State("sample-custom-path",        "value"),
    State("extra-datasets-version",    "data"),
    prevent_initial_call=True,
)
def add_sample(n_clicks, picked_path, custom_path, extra_ver):
    """Load a new Xenium dataset and append it to EXTRA_DATASETS side-by-side."""
    global EXTRA_DATASETS
    new_dir = (custom_path or "").strip() or picked_path
    if not new_dir:
        return "Select a sample or enter a path.", no_update, no_update, no_update
    new_dir = os.path.abspath(new_dir)
    if not os.path.exists(os.path.join(new_dir, "experiment.xenium")):
        return f"Not a valid Xenium directory: {new_dir}", no_update, no_update, no_update
    # Check for duplicates
    all_dirs = [DATA["data_dir"]] + [eds["data_dir"] for eds in EXTRA_DATASETS]
    if new_dir in all_dirs:
        return "Sample already loaded.", no_update, no_update, no_update
    try:
        new_data = load_xenium_data(new_dir)
        # Compute x_offset: place new sample to the right of all current ones
        # x_offset = max(x_centroid) of the last dataset + 500 µm gap - min(x_centroid) of new dataset
        if EXTRA_DATASETS:
            last_ds = EXTRA_DATASETS[-1]
            last_max_x = last_ds["df"]["x_centroid"].max() + last_ds["x_offset"]
        else:
            last_max_x = DATA["df"]["x_centroid"].max()
        new_min_x  = new_data["df"]["x_centroid"].min()
        x_offset   = last_max_x - new_min_x + 500.0
        new_data["x_offset"] = x_offset
        EXTRA_DATASETS.append(new_data)
        n_total = 1 + len(EXTRA_DATASETS)
        status_msg = f"{n_total} samples loaded"
        print(f"  Added extra dataset: {new_dir} (x_offset={x_offset:.0f} µm)", flush=True)
        return f"Added (x_offset={x_offset:.0f} µm).", False, (extra_ver or 0) + 1, status_msg
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return f"Error: {exc}", no_update, no_update, no_update


@app.callback(
    Output("extra-datasets-version", "data",     allow_duplicate=True),
    Output("extra-samples-status",   "children", allow_duplicate=True),
    Input("remove-extra-samples-btn", "n_clicks"),
    State("extra-datasets-version",   "data"),
    prevent_initial_call=True,
)
def remove_extra_samples(n_clicks, extra_ver):
    """Clear all extra datasets, reverting to single-sample view."""
    global EXTRA_DATASETS
    if not EXTRA_DATASETS:
        return no_update, "No extra samples loaded."
    n_removed = len(EXTRA_DATASETS)
    EXTRA_DATASETS.clear()
    print(f"  Removed {n_removed} extra dataset(s).", flush=True)
    return (extra_ver or 0) + 1, "Extra samples removed."


@app.callback(
    Output("cluster-method", "options"),
    Output("cluster-method", "value"),
    Input("dataset-version", "data"),
    prevent_initial_call=True,
)
def update_cluster_options(_):
    methods = DATA["cluster_methods"]
    opts = [{"label": method_label(m), "value": m} for m in methods]
    return opts, (methods[0] if methods else None)


@app.callback(
    Output("cache-clean-modal",        "is_open"),
    Output("cache-clean-confirm-text", "children"),
    Input("cache-clean-open-btn",      "n_clicks"),
    Input("cache-clean-cancel-btn",    "n_clicks"),
    Input("cache-clean-confirm-btn",   "n_clicks"),
    prevent_initial_call=True,
)
def toggle_cache_clean_modal(open_clicks, cancel_clicks, confirm_clicks):
    tid = callback_context.triggered_id
    if tid == "cache-clean-open-btn" and open_clicks:
        size = _cache_size_str()
        dataset_name = os.path.basename(DATA["data_dir"])
        return True, (f"Delete all cache entries for dataset '{dataset_name}'? "
                      f"Total cache size: {size}. This cannot be undone.")
    return False, no_update


@app.callback(
    Output("cache-clean-modal",  "is_open",  allow_duplicate=True),
    Output("cache-clean-done",   "data"),
    Output("cache-size-display", "children"),
    Input("cache-clean-confirm-btn", "n_clicks"),
    State("cache-clean-done",        "data"),
    prevent_initial_call=True,
)
def do_clean_cache(n_clicks, version):
    _clean_cache_for_dataset()
    new_size = _cache_size_str()
    return False, (version or 0) + 1, f"Cache: {new_size}"


# ─── Save as SpatialData callbacks ────────────────────────────────────────────
@app.callback(
    Output("save-sdata-modal", "is_open"),
    Input("save-sdata-open-btn",  "n_clicks"),
    Input("save-sdata-close-btn", "n_clicks"),
    State("save-sdata-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_save_sdata_modal(open_clicks, close_clicks, is_open):
    if callback_context.triggered_id == "save-sdata-open-btn" and open_clicks:
        return True
    if callback_context.triggered_id == "save-sdata-close-btn":
        return False
    return is_open


@app.callback(
    Output("save-sdata-status",  "children"),
    Output("save-sdata-poll",    "disabled"),
    Output("save-sdata-run-btn", "disabled"),
    Input("save-sdata-run-btn",  "n_clicks"),
    State("save-sdata-dir",      "value"),
    State("save-sdata-name",     "value"),
    State("save-sdata-roi-only", "value"),
    State("seg-source",          "value"),
    prevent_initial_call=True,
)
def start_save_sdata(n_clicks, out_dir, fname, roi_only_val, seg_source):
    out_dir = (out_dir or "").strip()
    fname   = (fname or "").strip()
    if not out_dir:
        return "Enter an output directory.", True, False
    if not fname:
        fname = "analysis.zarr"
    if not fname.endswith(".zarr"):
        fname += ".zarr"
    save_path  = os.path.join(out_dir, fname)
    roi_only   = "roi_only" in (roi_only_val or [])
    seg_source = seg_source or "xenium"
    with _save_sdata_lock:
        if _save_sdata_state["status"] == "running":
            return "Already saving…", False, True
        _save_sdata_state.update({"status": "running", "message": f"Writing to {save_path}…"})
    threading.Thread(target=_save_sdata_to_disk, args=(save_path, seg_source, roi_only), daemon=True).start()
    return f"Writing to {save_path}…", False, True


@app.callback(
    Output("save-sdata-status",  "children",  allow_duplicate=True),
    Output("save-sdata-poll",    "disabled",  allow_duplicate=True),
    Output("save-sdata-run-btn", "disabled",  allow_duplicate=True),
    Output("save-sdata-modal",   "is_open",   allow_duplicate=True),
    Input("save-sdata-poll",     "n_intervals"),
    State("save-sdata-modal",    "is_open"),
    prevent_initial_call=True,
)
def poll_save_sdata(_, modal_open):
    with _save_sdata_lock:
        status  = _save_sdata_state["status"]
        message = _save_sdata_state["message"]
    if status == "running":
        return message, False, True, modal_open
    if status == "done":
        return f"✓ {message}", True, False, False
    if status == "error":
        return f"✗ {message}", True, False, modal_open
    return no_update, True, False, modal_open


# ─── SPLIT correction callbacks ───────────────────────────────────────────────

@app.callback(
    Output("split-modal", "is_open"),
    [Input("split-modal-open-btn", "n_clicks"),
     Input("split-modal-close-btn", "n_clicks")],
    [State("split-modal", "is_open")],
    prevent_initial_call=True,
)
def open_split_modal(open_clicks, close_clicks, is_open):
    if not open_clicks and not close_clicks:
        return is_open
    triggered = callback_context.triggered_id
    if triggered == "split-modal-open-btn" and open_clicks:
        return True
    return False


@app.callback(
    [Output("split-modal-status", "children"),
     Output("split-poll", "disabled"),
     Output("split-run-btn", "disabled")],
    Input("split-run-btn", "n_clicks"),
    [State("split-rds-path", "value"),
     State("split-label-col", "value"),
     State("split-max-cores", "value"),
     State("split-min-umi", "value"),
     State("split-min-umi-sigma", "value"),
     State("split-purify-singlets", "value"),
     State("seg-source", "value")],
    prevent_initial_call=True,
)
def run_split(n_clicks, rds_path, label_col, max_cores, min_umi, min_umi_sigma, purify_singlets, seg_source):
    if not n_clicks:
        return no_update, True, False
    if not rds_path or not os.path.isfile(rds_path):
        return "⚠ RDS file not found.", True, False
    with _split_lock:
        if _split_state["status"] == "running":
            return "Already running…", False, True
        _split_state["status"]  = "running"
        _split_state["message"] = "Starting…"
        _split_state["result"]  = None
    label_col  = (label_col or "Names").strip()
    max_cores  = int(max_cores or 4)
    min_umi       = int(min_umi or 10)
    min_umi_sigma = int(min_umi_sigma or 100)
    do_purify_singlets = bool(purify_singlets)
    import contextvars
    _ctx = contextvars.copy_context()
    threading.Thread(
        target=_ctx.run,
        args=(_run_split_correction, rds_path, label_col, max_cores, seg_source,
              min_umi, min_umi_sigma, do_purify_singlets),
        daemon=True,
    ).start()
    return "Starting SPLIT correction…", False, True


@app.callback(
    [Output("split-status", "children"),
     Output("split-modal-status", "children", allow_duplicate=True),
     Output("split-modal", "is_open", allow_duplicate=True),
     Output("split-poll", "disabled", allow_duplicate=True),
     Output("split-run-btn", "disabled", allow_duplicate=True),
     Output("split-done", "data"),
     Output("counts-mode", "value", allow_duplicate=True),
     Output("dataset-version", "data", allow_duplicate=True)],
    Input("split-poll", "n_intervals"),
    [State("split-done", "data"),
     State("dataset-version", "data")],
    prevent_initial_call=True,
)
def poll_split(_, done_version, ds_version):
    with _split_lock:
        status  = _split_state["status"]
        message = _split_state["message"]
    if status == "idle":
        return "", "", no_update, True, False, done_version, no_update, no_update
    if status == "running":
        return message, message, no_update, False, True, done_version, no_update, no_update
    if status == "done":
        return f"✓ {message}", f"✓ {message}", False, True, False, done_version + 1, "corrected", no_update
    if status == "error":
        return f"✗ {message}", f"✗ {message}", no_update, True, False, done_version, no_update, no_update
    return "", "", no_update, True, False, done_version, no_update, no_update


@app.callback(
    Output("counts-mode-store", "data"),
    Input("counts-mode", "value"),
    prevent_initial_call=True,
)
def sync_counts_mode(value):
    global _active_counts_mode
    mode = value or "original"
    _active_counts_mode = mode
    print(f"  Now using {'SPLIT corrected' if mode == 'corrected' else 'original'} counts", flush=True)
    return mode


@app.callback(
    [Output("counts-mode", "options"),
     Output("counts-mode", "value", allow_duplicate=True)],
    [Input("split-done",   "data"),
     Input("seg-source",   "value"),
     Input("baysor-done",  "data"),
     Input("proseg-done",  "data")],
    State("counts-mode", "value"),
    prevent_initial_call=True,
)
def update_counts_options(_, seg_source, _bd, _pd, current_value):
    _tool = _seg_tool(seg_source)
    with _baysor_lock:
        bres = _baysor_state["result"] if _baysor_state["status"] == "done" else None
    with _proseg_lock:
        pres = _proseg_state["result"] if _proseg_state["status"] == "done" else None
    if _tool == "baysor":
        alt_res = bres
    elif _tool == "proseg":
        alt_res = pres
    else:
        alt_res = None
    if alt_res is not None:
        has_corrected = alt_res.get("split_corrected_expr") is not None
    else:
        has_corrected = DATA.get("split_corrected_expr") is not None
    options = [
        {"label": " Original",        "value": "original"},
        {"label": " SPLIT Corrected", "value": "corrected", "disabled": not has_corrected},
    ]
    # Only force a value change when currently on "corrected" but it's no longer available.
    # Any other case: no_update avoids spuriously triggering sync_counts_mode's log print.
    new_value = "original" if (not has_corrected and current_value == "corrected") else no_update
    return options, new_value


# ─── ROI Annotation callbacks ─────────────────────────────────────────────────

@app.callback(
    Output("roi-pending",      "data",     allow_duplicate=True),
    Output("roi-save-modal",   "is_open",  allow_duplicate=True),
    Output("roi-save-preview", "children"),
    Output("roi-save-error",   "children", allow_duplicate=True),
    Output("roi-done",         "data",     allow_duplicate=True),
    Input("spatial-plot",      "selectedData"),
    State("seg-source",        "value"),
    State("roi-done",          "data"),
    prevent_initial_call=True,
)
def capture_lasso_selection(selected_data, seg_source, roi_done):
    """CB-1: Lasso selection → compute hull → open save modal."""
    if not selected_data or not selected_data.get("points"):
        raise dash.exceptions.PreventUpdate
    pts = selected_data["points"]
    cell_ids = [p.get("customdata") for p in pts if p.get("customdata") is not None]
    if not cell_ids:
        raise dash.exceptions.PreventUpdate

    tool = _seg_tool(seg_source)
    if tool == "baysor":
        with _baysor_lock:
            result = _baysor_state.get("result")
        src_df = result["cells_df"] if result else None
    elif tool == "proseg":
        with _proseg_lock:
            result = _proseg_state.get("result")
        src_df = result["cells_df"] if result else None
    else:
        src_df = None

    if len(cell_ids) < 3:
        return dash.no_update, False, "", "Need at least 3 cells to define an ROI.", roi_done

    hull = _roi_compute_hull(cell_ids, src_df=src_df)
    if hull is None:
        return dash.no_update, False, "", "Could not compute hull (need ≥3 non-collinear cells).", roi_done

    with _roi_lock:
        _roi_state["pending_hull"] = hull
    new_done = (roi_done or 0) + 1
    preview = f"{len(cell_ids):,} cells selected — enter name and class below."
    return hull, True, preview, "", new_done


@app.callback(
    Output("roi-save-modal",  "is_open",  allow_duplicate=True),
    Output("roi-save-error",  "children", allow_duplicate=True),
    Output("roi-done",        "data",     allow_duplicate=True),
    Input("roi-save-btn",     "n_clicks"),
    State("roi-name-input",   "value"),
    State("roi-class-input",  "value"),
    State("roi-pending",      "data"),
    State("roi-done",         "data"),
    State("seg-source",       "value"),
    prevent_initial_call=True,
)
def save_roi(n_clicks, name, cls, polygon_xy, roi_done, seg_source):
    """CB-2: Save ROI."""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    name = (name or "").strip()
    cls  = (cls  or "region").strip()
    if not name:
        return True, "ROI Name is required.", roi_done
    if not polygon_xy:
        return True, "No polygon data — please draw a lasso selection first.", roi_done

    with _roi_lock:
        existing = _roi_state["rois"]
        duplicate = any(r["name"] == name and r["cls"] == cls for r in existing)
        if duplicate:
            print(f"ERROR: ROI '{name}' class '{cls}' already exists", flush=True)
            return True, f"ROI '{name}' (class '{cls}') already exists.", roi_done

        color = _roi_color(len(existing))
        new_roi = {"name": name, "cls": cls, "polygon_xy": polygon_xy, "color": color}
        _roi_state["rois"].append(new_roi)
        _roi_save_cache()
        _roi_state["pending_hull"] = None

    # Apply metadata to all loaded dfs
    with _roi_lock:
        rois = list(_roi_state["rois"])
    _roi_apply_metadata_to_df(DATA["df"], rois)
    tool = _seg_tool(seg_source)
    if tool == "baysor":
        with _baysor_lock:
            result = _baysor_state.get("result")
        if result:
            _roi_apply_metadata_to_df(result["cells_df"], rois)
    elif tool == "proseg":
        with _proseg_lock:
            result = _proseg_state.get("result")
        if result:
            _roi_apply_metadata_to_df(result["cells_df"], rois)

    threading.Thread(target=_roi_write_all_zarrs_bg, daemon=True).start()
    return False, "", (roi_done or 0) + 1


@app.callback(
    Output("roi-save-modal", "is_open", allow_duplicate=True),
    Output("roi-done", "data", allow_duplicate=True),
    Input("roi-save-cancel-btn", "n_clicks"),
    State("roi-done", "data"),
    prevent_initial_call=True,
)
def cancel_roi_save(n_clicks, roi_done):
    """CB-3: Cancel save modal — clears pending hull and bumps roi-done to re-render figure."""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    with _roi_lock:
        _roi_state["pending_hull"] = None
    return False, (roi_done or 0) + 1


@app.callback(
    Output("roi-done", "data", allow_duplicate=True),
    Input("spatial-plot", "relayoutData"),
    State("roi-done", "data"),
    prevent_initial_call=True,
)
def clear_lasso_on_tool_switch(relayout_data, roi_done):
    """Clear pending lasso selection when user switches away from lasso/select tool."""
    if not relayout_data or "dragmode" not in relayout_data:
        raise dash.exceptions.PreventUpdate
    if relayout_data["dragmode"] in ("lasso", "select"):
        raise dash.exceptions.PreventUpdate
    with _roi_lock:
        if _roi_state.get("pending_hull") is None:
            raise dash.exceptions.PreventUpdate
        _roi_state["pending_hull"] = None
    return (roi_done or 0) + 1


@app.callback(
    Output("roi-done", "data", allow_duplicate=True),
    Input("roi-show-toggle", "value"),
    State("roi-done", "data"),
    prevent_initial_call=True,
)
def toggle_roi_show(value, roi_done):
    """CB-4: Toggle ROI visibility."""
    with _roi_lock:
        _roi_state["show"] = "show" in (value or [])
    return (roi_done or 0) + 1


@app.callback(
    Output("roi-manage-modal", "is_open"),
    Output("roi-list-div",     "children"),
    Output("roi-op-a",         "options"),
    Output("roi-op-b",         "options"),
    Input("roi-manage-btn",       "n_clicks"),
    Input("roi-manage-close-btn", "n_clicks"),
    State("roi-manage-modal",     "is_open"),
    prevent_initial_call=True,
)
def open_roi_manager(open_clicks, close_clicks, is_open):
    """CB-5: Open/populate ROI manager."""
    triggered = callback_context.triggered_id
    if triggered == "roi-manage-close-btn":
        return False, dash.no_update, dash.no_update, dash.no_update
    if not open_clicks:
        raise dash.exceptions.PreventUpdate

    with _roi_lock:
        rois = list(_roi_state["rois"])

    def _make_table(rois):
        if not rois:
            return html.P("No ROIs defined yet. Use lasso on the spatial plot to draw one.",
                          style={"color": MUTED, "fontSize": "0.82rem"})
        rows = []
        for i, r in enumerate(rois):
            rows.append(html.Tr([
                html.Td(html.Div(style={"width": "18px", "height": "18px",
                                         "backgroundColor": r["color"], "borderRadius": "3px"})),
                html.Td(r["cls"], style={"padding": "4px 8px"}),
                html.Td(r["name"], style={"padding": "4px 8px"}),
                html.Td(dbc.Button("✕", id={"type": "roi-delete-btn", "index": i},
                                   size="sm", color="danger", style={"padding": "1px 7px"})),
            ]))
        return dbc.Table([html.Tbody(rows)], bordered=False, hover=True,
                         style={"fontSize": "0.82rem", "color": TEXT})

    table = _make_table(rois)
    op_opts = [{"label": f"{r['cls']}: {r['name']}", "value": i} for i, r in enumerate(rois)]
    return True, table, op_opts, op_opts


@app.callback(
    Output("roi-done",     "data",     allow_duplicate=True),
    Output("roi-list-div", "children", allow_duplicate=True),
    Input({"type": "roi-delete-btn", "index": ALL}, "n_clicks"),
    State("roi-done", "data"),
    prevent_initial_call=True,
)
def delete_roi(n_clicks_list, roi_done):
    """CB-6: Delete ROI by index."""
    triggered = callback_context.triggered_id
    if triggered is None or not any(n_clicks_list):
        raise dash.exceptions.PreventUpdate
    idx = triggered["index"]
    with _roi_lock:
        rois = _roi_state["rois"]
        if idx < 0 or idx >= len(rois):
            raise dash.exceptions.PreventUpdate
        # Reassign colors to keep palette stable
        rois.pop(idx)
        for j, r in enumerate(rois):
            r["color"] = _roi_color(j)
        _roi_save_cache()

    with _roi_lock:
        rois = list(_roi_state["rois"])
    _roi_apply_metadata_to_df(DATA["df"], rois)
    with _baysor_lock:
        bres = _baysor_state.get("result")
    if bres:
        _roi_apply_metadata_to_df(bres["cells_df"], rois)
    with _proseg_lock:
        pres = _proseg_state.get("result")
    if pres:
        _roi_apply_metadata_to_df(pres["cells_df"], rois)
    threading.Thread(target=_roi_write_all_zarrs_bg, daemon=True).start()

    # Rebuild table
    if not rois:
        table = html.P("No ROIs defined yet.", style={"color": MUTED, "fontSize": "0.82rem"})
    else:
        rows = []
        for i, r in enumerate(rois):
            rows.append(html.Tr([
                html.Td(html.Div(style={"width": "18px", "height": "18px",
                                         "backgroundColor": r["color"], "borderRadius": "3px"})),
                html.Td(r["cls"], style={"padding": "4px 8px"}),
                html.Td(r["name"], style={"padding": "4px 8px"}),
                html.Td(dbc.Button("✕", id={"type": "roi-delete-btn", "index": i},
                                   size="sm", color="danger", style={"padding": "1px 7px"})),
            ]))
        table = dbc.Table([html.Tbody(rows)], bordered=False, hover=True,
                          style={"fontSize": "0.82rem", "color": TEXT})

    return (roi_done or 0) + 1, table


@app.callback(
    Output("roi-op-error", "children"),
    Output("roi-done",     "data",    allow_duplicate=True),
    Input("roi-op-btn",    "n_clicks"),
    State("roi-op-a",      "value"),
    State("roi-op",        "value"),
    State("roi-op-b",      "value"),
    State("roi-op-name",   "value"),
    State("roi-op-class",  "value"),
    State("roi-done",      "data"),
    prevent_initial_call=True,
)
def apply_roi_operation(n_clicks, idx_a, op, idx_b, result_name, result_cls, roi_done):
    """CB-7: ROI set operation."""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    result_name = (result_name or "").strip()
    result_cls  = (result_cls  or "region").strip()
    if idx_a is None or idx_b is None or not op:
        return "Select ROI A, operation, and ROI B.", roi_done
    if not result_name:
        return "Enter a result name.", roi_done

    try:
        from shapely.geometry import Polygon as _SPoly
        with _roi_lock:
            rois = _roi_state["rois"]
            if idx_a >= len(rois) or idx_b >= len(rois):
                return "Invalid ROI index.", roi_done
            poly_a = _SPoly(rois[idx_a]["polygon_xy"])
            poly_b = _SPoly(rois[idx_b]["polygon_xy"])
            if op == "union":
                result_geom = poly_a.union(poly_b)
            elif op == "intersection":
                result_geom = poly_a.intersection(poly_b)
            else:
                result_geom = poly_a.difference(poly_b)
            hull = result_geom.convex_hull
            if hull.is_empty or hull.geom_type not in ("Polygon", "LineString"):
                return "Operation produced empty result.", roi_done
            if hull.geom_type == "Polygon":
                polygon_xy = [list(c) for c in hull.exterior.coords]
            else:
                polygon_xy = [list(c) for c in hull.coords]

            duplicate = any(r["name"] == result_name and r["cls"] == result_cls for r in rois)
            if duplicate:
                return f"ROI '{result_name}' (class '{result_cls}') already exists.", roi_done

            color = _roi_color(len(rois))
            rois.append({"name": result_name, "cls": result_cls,
                          "polygon_xy": polygon_xy, "color": color})
            _roi_save_cache()

        with _roi_lock:
            rois_snap = list(_roi_state["rois"])
        _roi_apply_metadata_to_df(DATA["df"], rois_snap)
        threading.Thread(target=_roi_write_all_zarrs_bg, daemon=True).start()
        return "", (roi_done or 0) + 1
    except Exception as e:
        return f"Error: {e}", roi_done


@app.callback(
    Output("color-by", "options", allow_duplicate=True),
    Input("startup-trigger", "n_intervals"),
    State("color-by",        "options"),
    prevent_initial_call=True,
)
def init_colorby_annot_options(_, current_options):
    """On startup, enable cell-type color-by options for any pre-loaded annotations."""
    with _annot_lock:
        methods_with_labels = {
            m for m in _ANNOT_METHODS
            if any(k.startswith(f"labels_{m}") and v is not None
                   for k, v in _annot_state.items())
        }
    options = []
    for opt in (current_options or []):
        val = opt["value"]
        if val.startswith("cell_type:"):
            m = val.split(":")[1]
            options.append({**opt, "disabled": m not in methods_with_labels})
        else:
            options.append(opt)
    return options


@app.callback(
    Output("color-by", "options", allow_duplicate=True),
    Input("roi-done",  "data"),
    State("color-by",  "options"),
    prevent_initial_call=True,
)
def update_color_by_for_rois(roi_done, current_options):
    """CB-8: Add/update roi_* options in color-by dropdown."""
    with _roi_lock:
        rois = list(_roi_state["rois"])
    classes = list(dict.fromkeys(r["cls"] for r in rois))  # preserve order, deduplicate
    options = [o for o in (current_options or []) if not o["value"].startswith("roi_")]
    for cls in classes:
        options.append({"label": f"ROI: {cls}", "value": f"roi_{cls}"})
    return options


@app.callback(
    Output("roi-sidebar-summary", "children"),
    Input("roi-done", "data"),
    prevent_initial_call=True,
)
def update_roi_sidebar_summary(roi_done):
    """CB-9: Update sidebar ROI summary."""
    with _roi_lock:
        rois = list(_roi_state["rois"])
    if not rois:
        return "No ROIs defined."
    class_counts: dict = {}
    for r in rois:
        class_counts[r["cls"]] = class_counts.get(r["cls"], 0) + 1
    parts = [f"{v} {k}" for k, v in class_counts.items()]
    return f"{len(rois)} ROI{'s' if len(rois)!=1 else ''} ({', '.join(parts)})"


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    print("\nXenium Explorer running at http://localhost:8050\n", flush=True)
    app.run(debug=False, host="0.0.0.0", port=8050)
