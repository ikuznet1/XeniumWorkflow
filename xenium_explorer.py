#!/usr/bin/env python3
"""
Xenium Explorer Clone
Interactive visualization for 10x Genomics Xenium spatial transcriptomics data.

Usage:
    python xenium_explorer.py [path/to/output-XETG...]
    python xenium_explorer.py          # auto-detects output-* directory
"""

import os
import sys
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
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.colors import qualitative

# ─── Constants ────────────────────────────────────────────────────────────────
PIXEL_SIZE_UM = 0.2125

# Max cells for which to render boundaries (performance guard)
BOUNDARY_CELL_LIMIT = 3000

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
_annot_state: dict = {"status": "idle", "message": "", "labels": None}
_annot_lock   = threading.Lock()

# ─── Baysor segmentation state ────────────────────────────────────────────────
_baysor_state: dict = {"status": "idle", "message": "", "result": None}
_baysor_lock  = threading.Lock()

# ─── SpaGE imputation state ───────────────────────────────────────────────────
_spage_state: dict = {"status": "idle", "message": "", "result": None}
_spage_lock   = threading.Lock()

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


def _run_celltypist(model_name: str) -> None:
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
                _annot_state["labels"] = labels

    try:
        # ── Check disk cache first ───────────────────────────────────────
        cache_file = _cache_path(model_name)
        if os.path.exists(cache_file):
            _set("running", "Loading cached annotation…")
            cached = pd.read_parquet(cache_file)
            pred_labels = cached["label"].astype(str)
            pred_labels.index = pred_labels.index.astype(str)
            unique_types = pred_labels.unique().tolist()
            print(f"  Loaded annotation from cache: {len(unique_types)} cell types", flush=True)
            _set("done", f"Done (cached) — {len(unique_types)} cell types", labels=pred_labels)
            return

        _set("running", "Downloading model…")
        ct_models.download_models(model=model_name, force_update=False)

        _set("running", "Building expression matrix…")
        # Build AnnData aligned to barcode order (expr rows = barcodes)
        expr     = DATA["expr"]           # (n_barcodes × n_genes) CSR
        genes    = DATA["gene_names"]
        barcodes = DATA["barcodes"]

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
            cache_file = _cache_path(model_name)
            pd.DataFrame({"label": pred_labels}).to_parquet(cache_file)
            print(f"  Annotation cached to {cache_file}", flush=True)
        except Exception as ce:
            print(f"  Warning: could not cache annotation: {ce}", flush=True)

        _set("done", f"Done — {len(unique_types)} cell types", labels=pred_labels)

    except Exception as exc:
        _set("error", str(exc))


def _run_seurat_annotation(rds_path: str, label_col: str = "Names") -> None:
    """
    Background thread: transfer cell type labels from a Seurat RDS reference
    to Xenium cells via PCA + cosine kNN majority voting.
    Uses the same rpy2 + SpaGE PV alignment already set up for SpaGE.
    """
    _redirect_rpy2_console()
    def _set(status, message, labels=None):
        with _annot_lock:
            _annot_state["status"]  = status
            _annot_state["message"] = message
            if labels is not None:
                _annot_state["labels"] = labels

    try:
        cache_key = f"seurat_{os.path.basename(rds_path)}_{label_col}"
        cache_file = _cache_path(cache_key)
        if os.path.exists(cache_file):
            _set("running", "Loading cached annotation…")
            cached = pd.read_parquet(cache_file)
            labels = cached["label"].astype(str)
            labels.index = labels.index.astype(str)
            unique_types = labels.unique().tolist()
            print(f"  Loaded annotation from cache: {len(unique_types)} cell types", flush=True)
            _set("done", f"Done (cached) — {len(unique_types)} cell types", labels=labels)
            return

        # ── 1. Load Seurat object via rpy2 ───────────────────────────────
        _set("running", "Loading Seurat RDS (may take ~30s)…")
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        pandas2ri.activate()

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
        mat_r = ro.r("slot(slot(._annot_rds_tmp, 'assays')[['RNA']], 'data')")
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

        # Xenium matrix for shared genes — build from sparse expr
        xen_expr = DATA["expr"]  # (n_cells × n_genes) CSR
        xen_shared = xen_expr[:, shared_xen_idx].toarray().astype("float32")
        # CP10K + log1p normalize Xenium
        row_sums = xen_shared.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        xen_shared = np.log1p(xen_shared / row_sums * 10_000)

        # Also normalize reference (already log-normalized from Seurat @data, but z-score both)
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

        xen_barcodes = DATA["barcodes"]
        pred_labels = pd.Series(xen_labels, index=xen_barcodes, name="label")
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


def _run_baysor(scale: float, min_mol: int, use_prior: bool, prior_conf: float) -> None:
    """Background thread: run Baysor resegmentation and store results."""

    def _set(status, message, result=None):
        with _baysor_lock:
            _baysor_state["status"]  = status
            _baysor_state["message"] = message
            if result is not None:
                _baysor_state["result"] = result

    # Locate baysor binary (may not be in PATH if app was launched from IDE)
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
        # ── Check that baysor is available ───────────────────────────────
        baysor_bin = _find_baysor()
        if baysor_bin is None:
            _set("error", "baysor not found. Install from github.com/kharchenkolab/Baysor")
            return
        print(f"  Using baysor: {baysor_bin}", flush=True)

        out_dir = os.path.join(os.path.expanduser("~"), ".xenium_explorer_cache",
                               "baysor_" + os.path.basename(DATA["data_dir"]))
        os.makedirs(out_dir, exist_ok=True)
        tx_csv = os.path.join(out_dir, "transcripts.csv")

        # ── Export transcripts to CSV ────────────────────────────────────
        _set("running", "Exporting transcripts to CSV…")
        tx = pd.read_parquet(
            os.path.join(DATA["data_dir"], "transcripts.parquet"),
            columns=["x_location", "y_location", "feature_name", "cell_id"],
        )
        tx.to_csv(tx_csv, index=False)
        print(f"  Baysor: exported {len(tx):,} transcripts to {tx_csv}", flush=True)

        # ── Build baysor command ─────────────────────────────────────────
        _set("running", "Running Baysor (this may take 10–30 min)…")
        cmd = [
            baysor_bin, "run",
            "-s", str(scale),
            "--min-molecules-per-cell", str(min_mol),
            "--polygon-format", "FeatureCollection",
            "-x", "x_location",
            "-y", "y_location",
            "-g", "feature_name",
            "-o", out_dir,
            tx_csv,
        ]
        if use_prior:
            cmd += ["--prior-segmentation-confidence", str(prior_conf), ":cell_id"]

        print(f"  Baysor cmd: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=out_dir,
        )
        # Stream output to console and update status with last line
        last_line = ""
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(f"  [baysor] {line}", flush=True)
                last_line = line
            _set("running", f"Running Baysor… {last_line[:60]}")
        proc.wait()
        if proc.returncode != 0:
            _set("error", f"Baysor exited with code {proc.returncode}. Check console for details.")
            return

        # ── Parse segmentation CSV ───────────────────────────────────────
        _set("running", "Loading Baysor results…")
        seg_csv = os.path.join(out_dir, "segmentation.csv")
        if not os.path.exists(seg_csv):
            _set("error", f"segmentation.csv not found in {out_dir}")
            return

        seg = pd.read_csv(seg_csv)
        # Baysor uses 0 for unassigned transcripts
        seg = seg[seg["cell"] != 0].copy()

        # Compute cell centroids and transcript counts
        cells_df = (
            seg.groupby("cell")
            .agg(
                x_centroid=("x_location", "mean"),
                y_centroid=("y_location", "mean"),
                transcript_counts=("x_location", "count"),
            )
            .rename_axis("cell_id")
        )

        # ── Parse polygon GeoJSON ────────────────────────────────────────
        cell_bounds: dict = {}
        # Baysor 0.7.x uses segmentation_polygons.json; older versions may differ
        for poly_name in ("segmentation_polygons.json", "polygons.json", "cell_polygons.json"):
            poly_path = os.path.join(out_dir, poly_name)
            if os.path.exists(poly_path):
                break
            poly_path = None
        if poly_path and os.path.exists(poly_path):
            with open(poly_path) as f:
                gj = json.load(f)
            for feat in gj.get("features", []):
                props = feat.get("properties", {})
                cid   = props.get("cell") or props.get("id")
                geom  = feat.get("geometry", {})
                if cid is None or geom.get("type") != "Polygon":
                    continue
                ring = geom["coordinates"][0]   # outer ring
                vx = np.array([p[0] for p in ring], dtype=np.float32)
                vy = np.array([p[1] for p in ring], dtype=np.float32)
                cell_bounds[cid] = (vx, vy)
            print(f"  Baysor: loaded {len(cell_bounds):,} polygons", flush=True)

        n_cells = len(cells_df)
        _set("done", f"Done — {n_cells:,} cells", result={
            "cells_df":    cells_df,
            "cell_bounds": cell_bounds,
            "out_dir":     out_dir,
        })
        print(f"  Baysor: {n_cells:,} cells segmented", flush=True)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set("error", str(exc)[:300])


# ─── SpaGE imputation ─────────────────────────────────────────────────────────
SPAGE_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SpaGE_repo")


def _spage_cache_path(rds_path: str, genes_key: str) -> str:
    tag = hashlib.md5((rds_path + DATA["data_dir"] + genes_key).encode()).hexdigest()[:12]
    cache_dir = os.path.join(os.path.expanduser("~"), ".xenium_explorer_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"spage_{tag}.parquet")


def _vectorized_spage(spatial_df: pd.DataFrame, rna_df: pd.DataFrame,
                      n_pv: int, genes_to_predict: list) -> pd.DataFrame:
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

    # Z-score (SpaGE normalizes internally on shared genes)
    rna_scaled   = pd.DataFrame(st.zscore(rna_df[shared],   axis=0), index=rna_df.index,   columns=shared)
    spat_scaled  = pd.DataFrame(st.zscore(spatial_df[shared], axis=0), index=spatial_df.index, columns=shared)

    # Principal Vectors
    pv = PVComputation(n_factors=n_pv, n_pv=n_pv, dim_reduction="pca", dim_reduction_target="pca")
    pv.fit(rna_scaled, spat_scaled)
    S = pv.source_components_.T
    eff_pv = int(np.sum(np.diag(pv.cosine_similarity_matrix_) > 0.3))
    eff_pv = max(eff_pv, 1)
    S = S[:, :eff_pv]
    print(f"  SpaGE: {eff_pv} effective principal vectors", flush=True)

    rna_proj  = rna_scaled.values @ S    # (n_rna,  eff_pv)
    spat_proj = spat_scaled.values @ S   # (n_spat, eff_pv)

    # kNN in PV space
    nbrs = NearestNeighbors(n_neighbors=50, algorithm="auto", metric="cosine")
    nbrs.fit(rna_proj)
    distances, indices = nbrs.kneighbors(spat_proj)   # (n_spat, 50)

    # Vectorized weighted average
    Y = rna_df[genes_to_predict].values.astype(np.float32)  # (n_rna, n_genes)
    Y_nbrs = Y[indices]                                      # (n_spat, 50, n_genes)

    valid  = distances < 1                                   # (n_spat, 50)
    d_v    = np.where(valid, distances, 0.0)
    d_sum  = d_v.sum(axis=1, keepdims=True)
    d_sum[d_sum == 0] = 1.0
    w      = np.where(valid, 1.0 - d_v / d_sum, 0.0)        # (n_spat, 50)
    n_v    = np.maximum(valid.sum(axis=1, keepdims=True) - 1, 1)
    w      = w / n_v
    imp    = np.einsum("ij,ijk->ik", w, Y_nbrs)             # (n_spat, n_genes)

    return pd.DataFrame(imp, index=spatial_df.index, columns=genes_to_predict)


def _run_spage_imputation(rds_path: str, n_pv: int, genes_input: str) -> None:
    """Background thread: run SpaGE gene imputation and store results."""
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
        from rpy2.robjects.packages import importr

        # ── Load reference gene list ─────────────────────────────────────
        _set("running", "Opening Seurat reference (this may take a few minutes)…")
        importr("SeuratObject")
        importr("Matrix")
        ro.r(f'seurat_ref <- readRDS("{rds_path}")')
        ro.r('rna_ref    <- seurat_ref[["RNA"]]')
        ro.r('rna_genes  <- rownames(rna_ref@data)')
        rna_genes_all = list(ro.r("rna_genes"))

        xenium_genes = set(DATA["gene_names"])
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

        cache_file = _spage_cache_path(rds_path, genes_key)
        if os.path.exists(cache_file):
            _set("running", "Loading cached SpaGE results…")
            imp_df = pd.read_parquet(cache_file)
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
            ro.r(f"""
genes_needed <- {genes_r}
genes_needed <- genes_needed[genes_needed %in% rna_genes]
mat_sub <- rna_ref@data[genes_needed, ]
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
        ro.r("rm(seurat_ref, rna_ref, mat_sub); gc()")

        # ── Convert to dense DataFrame ───────────────────────────────────
        _set("running", f"Building expression matrices ({len(rna_cells):,} ref cells)…")
        rna_df = pd.DataFrame(
            mat_sp.toarray().astype(np.float32),
            index=rna_cells, columns=rna_genes_sub,
        )

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

        # ── Build Xenium normalized matrix ───────────────────────────────
        shared_in_rna = [g for g in shared_genes if g in rna_genes_sub]
        xen_idx = [DATA["gene_name_to_idx"][g] for g in shared_in_rna]
        xen_raw = DATA["expr"][:, xen_idx].toarray().astype(np.float32)
        rs = xen_raw.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        xen_norm = np.log1p(xen_raw / rs * 10_000)

        df  = DATA["df"]
        idx = DATA["df_to_expr"]
        good = idx >= 0
        xen_aligned = np.zeros((len(df), len(shared_in_rna)), dtype=np.float32)
        xen_aligned[good] = xen_norm[idx[good]]
        spatial_df = pd.DataFrame(xen_aligned, index=df.index, columns=shared_in_rna)

        # ── Run SpaGE ────────────────────────────────────────────────────
        _set("running", f"Running SpaGE (n_pv={n_pv}, {len(genes_to_predict)} genes)…")
        imp_df = _vectorized_spage(spatial_df, rna_df, n_pv, genes_to_predict)

        # ── Cache ─────────────────────────────────────────────────────────
        _set("running", "Saving imputed results…")
        imp_df.to_parquet(cache_file)
        _set("done", f"Done — {len(imp_df.columns):,} genes imputed", result=imp_df)
        print(f"  SpaGE: {len(imp_df.columns)} genes cached to {cache_file}", flush=True)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set("error", str(exc)[:300])


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

_morph_zarr: dict  = {}   # {z_level: (TiffFile, zarr_array) | None}
_morph_lock        = threading.Lock()

# Low-resolution overview: pre-decoded and cached on disk for fast zoomed-out views
# {z_level: {"channels": [ndarray per ch], "stride": int, "H": int, "W": int}}
_morph_overview: dict = {}
_morph_overview_lock  = threading.Lock()
_overview_generating  = set()  # z_levels currently being generated

# Viewport render cache: stores raw (pre-brightness) normalised channel data
# so brightness changes are instant and don't re-read tiles.
# {(z_level, channels_key, stride): {"px": (x0,y0,x1,y1), "layers": list[ndarray], "p1p99": list}}
_morph_render_cache: dict = {}
_morph_render_lock  = threading.Lock()

# Pre-fetch margin: read this fraction extra on each side to absorb panning
_PREFETCH_FRAC = 0.50


def _get_morph_zarr(data_dir: str, z_level: int):
    """Lazily open and cache a zarr store for one morphology_focus z-level."""
    with _morph_lock:
        if z_level not in _morph_zarr:
            import tifffile, zarr as zarr_mod
            fname = f"morphology_focus_{z_level:04d}.ome.tif"
            path  = os.path.join(data_dir, "morphology_focus", fname)
            if os.path.exists(path):
                try:
                    tif  = tifffile.TiffFile(path)
                    arr  = zarr_mod.open(tif.aszarr(), mode="r")
                    _morph_zarr[z_level] = (tif, arr)
                    print(f"  Opened morphology z{z_level}: {arr.shape}", flush=True)
                except Exception as exc:
                    print(f"  Warning: could not open {fname}: {exc}", flush=True)
                    _morph_zarr[z_level] = None
            else:
                _morph_zarr[z_level] = None
    return _morph_zarr.get(z_level)


def _overview_cache_path(data_dir: str, z_level: int) -> str:
    tag = hashlib.md5(data_dir.encode()).hexdigest()[:8]
    cache_dir = os.path.join(os.path.expanduser("~"), ".xenium_explorer_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"morph_overview_{tag}_z{z_level}.npz")


def _load_or_generate_overview(data_dir: str, z_level: int):
    """Load overview from disk cache, or generate in background. Returns overview or None."""
    with _morph_overview_lock:
        if z_level in _morph_overview:
            return _morph_overview[z_level]

    # Try disk cache
    ov_path = _overview_cache_path(data_dir, z_level)
    if os.path.exists(ov_path):
        try:
            data = np.load(ov_path)
            ov = {
                "channels": [data[f"ch{i}"] for i in range(4)],
                "stride": int(data["stride"]),
            }
            with _morph_overview_lock:
                _morph_overview[z_level] = ov
            print(f"  Overview z{z_level}: loaded from disk cache", flush=True)
            return ov
        except Exception as exc:
            print(f"  Warning: could not load overview cache: {exc}", flush=True)

    # Kick off background generation (if not already running)
    with _morph_overview_lock:
        if z_level in _overview_generating:
            return None
        _overview_generating.add(z_level)

    def _gen():
        try:
            import tifffile, zarr as zarr_mod
            fname = f"morphology_focus_{z_level:04d}.ome.tif"
            path = os.path.join(data_dir, "morphology_focus", fname)
            if not os.path.exists(path):
                return
            # Open a SEPARATE tifffile handle to avoid concurrent access issues
            tif_ov = tifffile.TiffFile(path)
            zarr_ov = zarr_mod.open(tif_ov.aszarr(), mode="r")
            H, W = zarr_ov.shape[1], zarr_ov.shape[2]
            ov_stride = max(1, max(H, W) // 4096)
            print(f"  Generating overview z{z_level} (stride={ov_stride})…", flush=True)
            channels = []
            for ch in range(zarr_ov.shape[0]):
                arr = zarr_ov[ch, ::ov_stride, ::ov_stride].astype(np.uint16)
                channels.append(arr)
                print(f"    ch{ch}: {arr.shape}", flush=True)
            tif_ov.close()
            np.savez_compressed(
                ov_path,
                ch0=channels[0], ch1=channels[1],
                ch2=channels[2], ch3=channels[3],
                stride=np.array(ov_stride),
            )
            ov = {"channels": channels, "stride": ov_stride}
            with _morph_overview_lock:
                _morph_overview[z_level] = ov
            print(f"  Overview z{z_level}: generated and cached ({ov_path})", flush=True)
        except Exception as exc:
            print(f"  Overview generation error: {exc}", flush=True)
        finally:
            with _morph_overview_lock:
                _overview_generating.discard(z_level)

    threading.Thread(target=_gen, daemon=True).start()
    return None


def _compose_rgb(layers, p1p99_list, brightness, out_h, out_w, channels, crop=None):
    """Compose cached raw layers into an RGB uint8 array, applying brightness.
    If crop=(y0,y1,x0,x1), slice each layer before compositing (saves memory)."""
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


def _encode_overlay_jpeg(rgb, vp_px_x0, vp_px_y0, rw_vp, rh_vp, img_opacity):
    """Clip, encode to JPEG base64, return plotly layout.images dict."""
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    img_pil = Image.fromarray(rgb_uint8, mode="RGB")
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


def make_morphology_overlay(data_dir, relayout, z_level, channels, brightness, img_opacity):
    """
    Read morphology_focus tiles for the current viewport, composite selected
    channels into an RGB image, and return a plotly layout.images dict.

    Speed optimisations:
      1. Parallel channel reads via ThreadPoolExecutor (3–4× speedup).
      2. Pre-fetch margin: read 50 % extra on each side so pans hit cache.
      3. Raw-layer cache: stores pre-brightness data so brightness/opacity
         changes are instant (no tile re-read).
      4. Brightness applied as cheap post-process on cached data.
    """
    if not channels or not relayout:
        return None, ""

    pair = _get_morph_zarr(data_dir, z_level)
    if pair is None:
        return None, " · morphology images not found"

    _, zarr_arr = pair
    H, W = zarr_arr.shape[1], zarr_arr.shape[2]
    full_w_um = W * PIXEL_SIZE_UM
    full_h_um = H * PIXEL_SIZE_UM

    # ── Viewport bounds (plot µm, Y flipped) ─────────────────────────────
    x0_um = float(relayout.get("xaxis.range[0]", 0))
    x1_um = float(relayout.get("xaxis.range[1]", full_w_um))
    y0_um = float(relayout.get("yaxis.range[0]", -full_h_um))
    y1_um = float(relayout.get("yaxis.range[1]", 0))

    vp_w = x1_um - x0_um
    vp_h = abs(y1_um - y0_um)
    if max(vp_w, vp_h) > MORPH_MAX_UM:
        return None, f" · zoom in to ≤{MORPH_MAX_UM:,} µm for image overlay"

    # ── Exact viewport in image pixels ───────────────────────────────────
    vp_px_x0 = max(0, int(x0_um / PIXEL_SIZE_UM))
    vp_px_x1 = min(W, int(x1_um / PIXEL_SIZE_UM) + 1)
    vp_px_y0 = max(0, int(-y1_um / PIXEL_SIZE_UM))
    vp_px_y1 = min(H, int(-y0_um / PIXEL_SIZE_UM) + 1)
    if vp_px_x1 <= vp_px_x0 or vp_px_y1 <= vp_px_y0:
        return None, ""

    # ── Stride (target ≤ 800 px output) ──────────────────────────────────
    rw_vp = vp_px_x1 - vp_px_x0
    rh_vp = vp_px_y1 - vp_px_y0
    stride = max(1, max(rw_vp, rh_vp) // 800)

    # ── Try low-res overview first (instant for large viewports) ─────────
    # Kick off generation in background if not available; use overview when
    # the viewport would require many tile decodes (stride >= overview stride)
    ov = _load_or_generate_overview(data_dir, z_level)
    if ov is not None and stride >= ov["stride"]:
        ov_stride = ov["stride"]
        ch_map = {c["value"]: c for c in MORPH_CHANNELS}
        ov_y0 = vp_px_y0 // ov_stride
        ov_y1 = min(ov["channels"][0].shape[0], vp_px_y1 // ov_stride + 1)
        ov_x0 = vp_px_x0 // ov_stride
        ov_x1 = min(ov["channels"][0].shape[1], vp_px_x1 // ov_stride + 1)
        oh = ov_y1 - ov_y0
        ow = ov_x1 - ov_x0
        if oh > 0 and ow > 0:
            rgb = np.zeros((oh, ow, 3), dtype=np.float32)
            for ch_val in channels:
                info = ch_map.get(ch_val)
                if info is None:
                    continue
                region = ov["channels"][info["ch"]][ov_y0:ov_y1, ov_x0:ov_x1].astype(np.float32)
                nonzero = region[region > 0]
                if nonzero.size > 50:
                    p1, p99 = np.percentile(nonzero, [1, 99])
                    if p99 > p1:
                        norm = np.clip((region - p1) / (p99 - p1) * brightness, 0, 1)
                    else:
                        norm = np.zeros_like(region)
                else:
                    norm = np.zeros_like(region)
                color_f = np.array(info["color"], dtype=np.float32) / 255.0
                rgb += norm[..., np.newaxis] * color_f
            return _encode_overlay_jpeg(rgb, vp_px_x0, vp_px_y0,
                                        rw_vp, rh_vp, img_opacity), " (overview)"

    sorted_ch = tuple(sorted(channels))
    cache_key = (z_level, sorted_ch, stride)

    # ── Check render cache (raw layers, brightness-independent) ──────────
    with _morph_render_lock:
        entry = _morph_render_cache.get(cache_key)

    if entry is not None:
        cpx_x0, cpx_y0, cpx_x1, cpx_y1 = entry["px"]
        # Is the viewport fully inside the cached region?
        if (vp_px_x0 >= cpx_x0 and vp_px_y0 >= cpx_y0 and
                vp_px_x1 <= cpx_x1 and vp_px_y1 <= cpx_y1):
            crop_y0 = (vp_px_y0 - cpx_y0) // stride
            crop_y1 = crop_y0 + (rh_vp - 1) // stride + 1
            crop_x0 = (vp_px_x0 - cpx_x0) // stride
            crop_x1 = crop_x0 + (rw_vp - 1) // stride + 1
            out_h = crop_y1 - crop_y0
            out_w = crop_x1 - crop_x0
            rgb = _compose_rgb(entry["layers"], entry["p1p99"],
                               brightness, out_h, out_w, channels,
                               crop=(crop_y0, crop_y1, crop_x0, crop_x1))
            return _encode_overlay_jpeg(rgb, vp_px_x0, vp_px_y0,
                                        rw_vp, rh_vp, img_opacity), " (cached)"

    # ── Expand read region by pre-fetch margin ────────────────────────────
    pad_x = int(rw_vp * _PREFETCH_FRAC)
    pad_y = int(rh_vp * _PREFETCH_FRAC)
    px_x0 = max(0, vp_px_x0 - pad_x)
    px_x1 = min(W, vp_px_x1 + pad_x)
    px_y0 = max(0, vp_px_y0 - pad_y)
    px_y1 = min(H, vp_px_y1 + pad_y)

    rw, rh = px_x1 - px_x0, px_y1 - px_y0
    out_h = (rh - 1) // stride + 1
    out_w = (rw - 1) // stride + 1

    # ── Read channels sequentially (tifffile zarr not thread-safe) ──────
    def read_channel(ch_val):
        ch_map = {c["value"]: c for c in MORPH_CHANNELS}
        info = ch_map.get(ch_val)
        if info is None:
            return None, (0, 0)
        with _morph_lock:
            region = zarr_arr[info["ch"], px_y0:px_y1:stride, px_x0:px_x1:stride].astype(np.float32)
        region = region[:out_h, :out_w]
        nonzero = region[region > 0]
        if nonzero.size > 100:
            p1, p99 = np.percentile(nonzero, [1, 99])
        else:
            p1, p99 = 0.0, 0.0
        return region, (float(p1), float(p99))

    results = [read_channel(ch) for ch in channels]

    layers  = [r[0] for r in results]
    p1p99   = [r[1] for r in results]

    # ── Store raw layers in cache ─────────────────────────────────────────
    with _morph_render_lock:
        _morph_render_cache[cache_key] = {
            "px":     (px_x0, px_y0, px_x1, px_y1),
            "layers": layers,
            "p1p99":  p1p99,
        }
        if len(_morph_render_cache) > 6:
            oldest = next(iter(_morph_render_cache))
            del _morph_render_cache[oldest]

    # ── Compose and encode the viewport sub-region ───────────────────────
    crop_y0 = (vp_px_y0 - px_y0) // stride
    crop_y1 = crop_y0 + (rh_vp - 1) // stride + 1
    crop_x0 = (vp_px_x0 - px_x0) // stride
    crop_x1 = crop_x0 + (rw_vp - 1) // stride + 1
    c_h = crop_y1 - crop_y0
    c_w = crop_x1 - crop_x0
    rgb = _compose_rgb(layers, p1p99, brightness, c_h, c_w, channels,
                       crop=(crop_y0, crop_y1, crop_x0, crop_x1))

    return _encode_overlay_jpeg(rgb, vp_px_x0, vp_px_y0,
                                rw_vp, rh_vp, img_opacity), ""


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

    with open(os.path.join(data_dir, "experiment.xenium")) as f:
        metadata = json.load(f)

    print("  cells...", flush=True)
    cells = pd.read_parquet(os.path.join(data_dir, "cells.parquet"))
    cells = cells.set_index("cell_id")

    print("  UMAP...", flush=True)
    umap_df = pd.read_csv(
        os.path.join(data_dir, "analysis/umap/gene_expression_2_components/projection.csv"),
        index_col="Barcode",
    )

    print("  clusters...", flush=True)
    cluster_dir = os.path.join(data_dir, "analysis/clustering")
    cluster_methods = {}
    for method in sorted(os.listdir(cluster_dir)):
        path = os.path.join(cluster_dir, method, "clusters.csv")
        if os.path.exists(path):
            df_c = pd.read_csv(path, index_col="Barcode")
            cluster_methods[method] = df_c["Cluster"]

    print("  gene expression matrix...", flush=True)
    with h5py.File(os.path.join(data_dir, "cell_feature_matrix.h5"), "r") as f:
        barcodes   = [b.decode() for b in f["matrix/barcodes"][:]]
        gene_names = [g.decode() for g in f["matrix/features/name"][:]]
        shape = tuple(f["matrix/shape"][:])
        expr = sp.csc_matrix(
            (f["matrix/data"][:], f["matrix/indices"][:], f["matrix/indptr"][:]),
            shape=shape,
        ).T.tocsr()   # → (cells × genes)

    print("  cell boundaries...", flush=True)
    cb_df = pd.read_parquet(os.path.join(data_dir, "cell_boundaries.parquet"))
    cell_bounds = _build_boundary_dict(cb_df)

    print("  nucleus boundaries...", flush=True)
    nb_df = pd.read_parquet(os.path.join(data_dir, "nucleus_boundaries.parquet"))
    nucleus_bounds = _build_boundary_dict(nb_df)

    # Build merged dataframe
    df = cells.copy()
    df["umap_1"] = umap_df["UMAP-1"].reindex(df.index)
    df["umap_2"] = umap_df["UMAP-2"].reindex(df.index)
    for method, series in cluster_methods.items():
        df[f"clust__{method}"] = series.reindex(df.index).fillna(0).astype(int)

    barcode_idx = {b: i for i, b in enumerate(barcodes)}
    df_to_expr  = np.array([barcode_idx.get(cid, -1) for cid in df.index], dtype=np.int64)

    print(f"  Done: {len(df):,} cells | {len(gene_names)} genes | "
          f"{len(cluster_methods)} cluster sets", flush=True)

    gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}

    return {
        "metadata":         metadata,
        "df":               df,
        "gene_names":       gene_names,
        "gene_name_to_idx": gene_name_to_idx,
        "barcodes":         barcodes,
        "expr":             expr,
        "df_to_expr":       df_to_expr,
        "cluster_methods":  list(cluster_methods.keys()),
        "cell_bounds":      cell_bounds,
        "nucleus_bounds":   nucleus_bounds,
        "data_dir":         data_dir,
    }


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


# ─── Data helpers ─────────────────────────────────────────────────────────────
def get_gene_expression(gene: str) -> np.ndarray:
    """log1p expression for *gene*, aligned to DATA['df'] row order.
    Genes with ' [imp]' suffix are served from SpaGE imputed results."""
    if gene.endswith(" [imp]"):
        base = gene[:-6]
        with _spage_lock:
            res = _spage_state.get("result")
        if res is not None and base in res.columns:
            return res[base].values.astype(np.float64)
        return np.zeros(len(DATA["df"]), dtype=np.float64)
    gene_idx = DATA["gene_name_to_idx"][gene]
    raw  = DATA["expr"][:, gene_idx].toarray().flatten()
    idx  = DATA["df_to_expr"]
    vals = np.where(idx >= 0, raw[np.clip(idx, 0, len(raw) - 1)], 0.0)
    return np.log1p(vals)


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


def _cell_type_traces(x, y, df, size, opacity, mode="spatial"):
    """Categorical traces coloured by cell type annotation."""
    with _annot_lock:
        labels = _annot_state.get("labels")
    if labels is None:
        return [go.Scattergl(x=x, y=y, mode="markers",
                             marker=dict(size=size, color=MUTED, opacity=opacity),
                             name="No annotation")]

    # Align labels to df index
    aligned = labels.reindex(df.index).fillna("Unknown")
    cell_types = sorted(aligned.unique())
    cmap = {ct: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, ct in enumerate(cell_types)}

    traces = []
    for ct in cell_types:
        mask = (aligned == ct).values
        mk   = size if mode == "spatial" else max(2, size * 1.3)
        traces.append(go.Scattergl(
            x=x[mask], y=y[mask], mode="markers",
            marker=dict(size=mk, color=cmap[ct], opacity=opacity),
            name=ct, legendgroup=ct,
            showlegend=(mode == "spatial"),
            text=df.index[mask].tolist(), customdata=df.index[mask].tolist(),
            hovertemplate="<b>%{text}</b><extra>" + ct + "</extra>",
        ))
    return traces


def make_spatial_fig(color_by, method, gene, size, opacity,
                     show_cell_bounds, show_nuc_bounds, relayout,
                     morph_image=None, extra_title="", baysor_active=False):
    # Choose data source: Baysor result or original Xenium segmentation
    with _baysor_lock:
        bres = _baysor_state["result"] if baysor_active and _baysor_state["result"] else None

    if bres is not None:
        bdf = bres["cells_df"]
        x   =  bdf["x_centroid"].values
        y   = -bdf["y_centroid"].values
        cell_ids = bdf.index.tolist()
        tx_counts = bdf["transcript_counts"].values
        source_label = " [Baysor]"
        # Baysor: always color by transcript counts (no cluster/gene data aligned)
        traces = [go.Scattergl(
            x=x, y=y, mode="markers",
            marker=dict(size=size, color=tx_counts,
                        colorscale="Viridis", opacity=opacity, showscale=True,
                        colorbar=dict(title="Transcript<br>Counts", thickness=12, len=0.5, x=1.02)),
            text=cell_ids, customdata=cell_ids,
            hovertemplate="<b>Cell %{text}</b><br>Transcripts: %{marker.color}<extra></extra>",
            name="Transcript Counts",
        )]
        show_legend = False
        cell_bounds_src = bres["cell_bounds"]
        nuc_bounds_src  = {}
    else:
        df  = DATA["df"]
        x   =  df["x_centroid"].values   # x_centroid is in µm
        y   = -df["y_centroid"].values   # y_centroid is in µm, negate for plot
        source_label    = ""
        cell_bounds_src = DATA["cell_bounds"]
        nuc_bounds_src  = DATA["nucleus_bounds"]

        # ── Cell scatter traces ──────────────────────────────────────────
        if color_by == "cluster":
            traces, show_legend = _categorical_traces(x, y, df, method, size, opacity), True
        elif color_by == "cell_type":
            traces, show_legend = _cell_type_traces(x, y, df, size, opacity, "spatial"), True
        elif color_by == "gene":
            vals = get_gene_expression(gene)
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
    boundary_status = ""
    if show_cell_bounds or show_nuc_bounds:
        visible_ids = viewport_cell_ids(relayout or {})
        if visible_ids is None and bres is None:
            boundary_status = " · zoom in to see boundaries"
        else:
            # For Baysor, use all cells (no viewport filter needed for small sets)
            if bres is not None:
                ids_to_render = list(cell_bounds_src.keys())[:BOUNDARY_CELL_LIMIT]
            else:
                ids_to_render = visible_ids or []
            if len(ids_to_render) > BOUNDARY_CELL_LIMIT:
                boundary_status = f" · zoom in further ({len(ids_to_render):,} cells visible, limit {BOUNDARY_CELL_LIMIT:,})"
            else:
                if show_cell_bounds:
                    t = build_boundary_trace(
                        ids_to_render, cell_bounds_src, "#00d4ff", "Cell Boundaries"
                    )
                    if t:
                        traces.append(t)
                        show_legend = True
                if show_nuc_bounds and nuc_bounds_src:
                    t = build_boundary_trace(
                        ids_to_render if bres is None else [],
                        nuc_bounds_src, "#ff9f43", "Nucleus Boundaries"
                    )
                    if t:
                        traces.append(t)
                        show_legend = True

    title_text = "Spatial View" + source_label + boundary_status + extra_title

    fig = go.Figure(data=traces)
    if morph_image:
        fig.update_layout(images=[morph_image])
    fig.update_layout(
        **_base_layout(title_text, "X (µm)", "Y (µm)", equal_aspect=True),
        showlegend=show_legend,
        legend=dict(
            bgcolor="rgba(0,0,0,0.45)", font=dict(size=10),
            itemsizing="constant", tracegroupgap=1,
            x=0.01, y=0.99, xanchor="left", yanchor="top",
        ),
        uirevision="spatial",
    )
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

def make_umap_fig(color_by, method, gene, size, opacity):
    if "df" not in _umap_df_cache:
        _umap_df_cache["df"] = DATA["df"].dropna(subset=["umap_1", "umap_2"])
    df = _umap_df_cache["df"]
    xu = df["umap_1"].values
    yu = df["umap_2"].values
    mk = max(2, size * 1.2)

    if color_by == "cluster":
        traces = _categorical_traces(xu, yu, df, method, mk, opacity, mode="umap")
    elif color_by == "cell_type":
        traces = _cell_type_traces(xu, yu, df, mk, opacity, mode="umap")
    elif color_by == "gene":
        has_umap = DATA["df"]["umap_1"].notna()
        vals = get_gene_expression(gene)[has_umap.values]
        traces = [go.Scattergl(
            x=xu, y=yu, mode="markers",
            marker=dict(size=mk, color=vals, colorscale="Plasma", opacity=opacity, showscale=False),
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

    fig = go.Figure(data=traces)
    fig.update_layout(
        **_base_layout("UMAP", "UMAP 1", "UMAP 2"),
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
gene_options    = [{"label": g, "value": g} for g in sorted(gene_names)]

sidebar = html.Div([
    # Header
    html.Div([
        html.Div("XENIUM EXPLORER", style={
            "fontSize": "11px", "fontWeight": "700",
            "letterSpacing": "2px", "color": ACCENT, "marginBottom": "4px",
        }),
        html.Div(meta.get("run_name", "—"), style={"fontSize": "13px", "color": TEXT, "fontWeight": "600"}),
        html.Div(meta.get("region_name", ""), style={"fontSize": "11px", "color": MUTED}),
    ], style={"marginBottom": "18px"}),

    html.Hr(style={"borderColor": BORDER, "margin": "0 0 14px 0"}),

    # Stats
    html.Div([
        stat_row("Cells",       f"{meta.get('num_cells', 0):,}"),
        stat_row("Transcripts", f"{meta.get('num_transcripts', 0):,}"),
        stat_row("Panel",       meta.get("panel_name", "—")),
        stat_row("Genes",       str(len(gene_names))),
        stat_row("Tissue",      meta.get("panel_tissue_type", "—")),
        stat_row("Pixel",       f"{meta.get('pixel_size', PIXEL_SIZE_UM)} µm"),
    ], style={"marginBottom": "14px"}),

    html.Hr(style={"borderColor": BORDER, "margin": "0 0 14px 0"}),

    # Color by
    ctrl_label("Color By"),
    dcc.RadioItems(
        id="color-by",
        options=[
            {"label": "Cluster",           "value": "cluster"},
            {"label": "Gene Expression",   "value": "gene"},
            {"label": "Cell Type",         "value": "cell_type", "disabled": True},
            {"label": "Transcript Counts", "value": "transcript_counts"},
            {"label": "Cell Area",         "value": "cell_area"},
            {"label": "Nucleus Area",      "value": "nucleus_area"},
        ],
        value="cluster",
        inputStyle={"marginRight": "6px"},
        labelStyle={
            "display": "flex", "alignItems": "center",
            "marginBottom": "7px", "fontSize": "13px", "color": TEXT, "cursor": "pointer",
        },
        style={"marginBottom": "14px"},
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
            value=sorted(gene_names)[0] if gene_names else None,
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
                {"label": html.Span(" Cell Boundaries",    style={"color": "#00d4ff", "fontSize": "13px"}),
                 "value": "cell"},
                {"label": html.Span(" Nucleus Boundaries", style={"color": "#ff9f43", "fontSize": "13px"}),
                 "value": "nucleus"},
            ],
            value=[],
            inputStyle={"marginRight": "6px"},
            labelStyle={
                "display": "flex", "alignItems": "center",
                "marginBottom": "7px", "cursor": "pointer",
            },
        ),
        html.Div(
            f"Renders up to {BOUNDARY_CELL_LIMIT:,} cells in view. Zoom in to activate.",
            style={"fontSize": "10px", "color": MUTED, "marginTop": "4px", "lineHeight": "1.4"},
        ),
    ], style={"marginBottom": "10px"}),

    html.Button(
        "↻  Refresh Overlay",
        id="overlay-refresh-btn",
        n_clicks=0,
        style={
            "width": "100%", "padding": "6px 0",
            "backgroundColor": CARD_BG, "color": TEXT,
            "border": f"1px solid {BORDER}", "borderRadius": "5px",
            "cursor": "pointer", "fontSize": "12px", "marginBottom": "6px",
        },
    ),
    html.Div(
        "Click after zooming to reload boundaries / image tiles.",
        style={"fontSize": "10px", "color": MUTED, "lineHeight": "1.4", "marginBottom": "14px"},
    ),

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
    dcc.RadioItems(
        id="annot-source",
        options=[
            {"label": html.Span(" CellTypist model", style={"fontSize": "12px", "color": TEXT}),
             "value": "celltypist"},
            {"label": html.Span(" Seurat RDS reference", style={"fontSize": "12px", "color": TEXT}),
             "value": "seurat"},
        ],
        value="celltypist",
        inputStyle={"marginRight": "5px"},
        labelStyle={"display": "flex", "alignItems": "center", "marginBottom": "5px", "cursor": "pointer"},
        style={"marginBottom": "8px"},
    ),
    # CellTypist controls
    html.Div(id="annot-celltypist-div", children=[
        dcc.Dropdown(
            id="annot-model",
            options=[{"label": v, "value": k} for k, v in CELLTYPIST_MODELS.items()],
            value="Healthy_Adult_Heart.pkl",
            clearable=False,
            style={"fontSize": "12px", "marginBottom": "8px"},
        ),
    ]),
    # Seurat RDS controls
    html.Div(id="annot-seurat-div", style={"display": "none"}, children=[
        ctrl_label("RDS File Path"),
        dcc.Input(
            id="annot-rds-path",
            type="text",
            value="/Users/ikuz/Documents/XeniumWorkflow/Post_R3_FINAL_with_counts.rds",
            style={
                "width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
                "border": f"1px solid {BORDER}", "borderRadius": "4px",
                "padding": "4px 8px", "fontSize": "11px", "marginBottom": "8px",
                "boxSizing": "border-box",
            },
        ),
        ctrl_label("Label Column"),
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
    html.Button(
        "Run Annotation",
        id="annot-btn",
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

    # ── Baysor resegmentation ───────────────────────────────────────────
    ctrl_label("Baysor Resegmentation"),
    html.Div(
        "Resegment cells using Baysor (must be installed). "
        "See github.com/kharchenkolab/Baysor.",
        style={"fontSize": "10px", "color": MUTED, "marginBottom": "8px", "lineHeight": "1.4"},
    ),
    ctrl_label("Cell Radius (µm)"),
    dcc.Slider(
        id="baysor-scale", min=5, max=60, step=1, value=20,
        marks={5: "5", 20: "20", 40: "40", 60: "60"},
        tooltip={"placement": "bottom", "always_visible": False},
    ),
    html.Div(style={"marginBottom": "8px"}),
    ctrl_label("Min Transcripts / Cell"),
    dcc.Input(
        id="baysor-min-mol", type="number", value=10, min=1, max=500, step=1,
        style={
            "width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
            "border": f"1px solid {BORDER}", "borderRadius": "4px",
            "padding": "4px 8px", "fontSize": "12px", "marginBottom": "8px",
        },
    ),
    dcc.Checklist(
        id="baysor-use-prior",
        options=[{"label": html.Span(
            " Use Xenium prior segmentation",
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
    html.Button(
        "Run Baysor",
        id="baysor-run-btn",
        n_clicks=0,
        style={
            "width": "100%", "padding": "7px 0",
            "backgroundColor": "#1f6feb", "color": "#fff",
            "border": "none", "borderRadius": "5px",
            "cursor": "pointer", "fontSize": "13px", "fontWeight": "600",
            "marginBottom": "8px",
        },
    ),
    html.Div(id="baysor-status", style={"fontSize": "11px", "color": MUTED, "minHeight": "16px", "marginBottom": "6px"}),
    dcc.Interval(id="baysor-poll", interval=3000, disabled=True),
    dcc.Checklist(
        id="baysor-active",
        options=[{"label": html.Span(
            " Show Baysor segmentation",
            style={"fontSize": "12px", "color": "#58a6ff"},
        ), "value": "yes"}],
        value=[],
        inputStyle={"marginRight": "6px"},
        labelStyle={"display": "flex", "alignItems": "center", "marginBottom": "4px"},
    ),

    html.Hr(style={"borderColor": BORDER, "margin": "14px 0"}),

    # ── SpaGE gene imputation ───────────────────────────────────────────
    ctrl_label("SpaGE Gene Imputation"),
    html.Div(
        "Impute additional genes from snRNA-seq reference using SpaGE.",
        style={"fontSize": "10px", "color": MUTED, "marginBottom": "8px", "lineHeight": "1.4"},
    ),
    ctrl_label("Reference .rds File"),
    dcc.Input(
        id="spage-rds-path",
        type="text",
        value="/Users/ikuz/Documents/XeniumWorkflow/Post_R3_FINAL_with_counts.rds",
        debounce=True,
        style={
            "width": "100%", "backgroundColor": CARD_BG, "color": TEXT,
            "border": f"1px solid {BORDER}", "borderRadius": "4px",
            "padding": "4px 8px", "fontSize": "11px", "marginBottom": "8px",
            "boxSizing": "border-box",
        },
    ),
    ctrl_label("Principal Vectors (n_pv)"),
    dcc.Slider(
        id="spage-npv", min=10, max=100, step=5, value=50,
        marks={10: "10", 50: "50", 100: "100"},
        tooltip={"placement": "bottom", "always_visible": False},
    ),
    html.Div(style={"marginBottom": "8px"}),
    ctrl_label("Genes to Impute"),
    html.Div(
        "Leave empty for top-200 HVG. Paste gene names separated by commas or newlines.",
        style={"fontSize": "10px", "color": MUTED, "marginBottom": "4px", "lineHeight": "1.4"},
    ),
    dcc.Textarea(
        id="spage-genes",
        placeholder="e.g. NPPA, MYH7, TNNT2",
        style={
            "width": "100%", "height": "70px", "backgroundColor": CARD_BG,
            "color": TEXT, "border": f"1px solid {BORDER}", "borderRadius": "4px",
            "padding": "4px 8px", "fontSize": "11px", "marginBottom": "8px",
            "boxSizing": "border-box", "resize": "vertical",
        },
    ),
    html.Button(
        "Run SpaGE",
        id="spage-run-btn",
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

    # UMAP toggle
    html.Button(
        "Hide UMAP",
        id="umap-toggle",
        n_clicks=0,
        style={
            "width": "100%", "padding": "7px 0",
            "backgroundColor": CARD_BG, "color": TEXT,
            "border": f"1px solid {BORDER}", "borderRadius": "5px",
            "cursor": "pointer", "fontSize": "13px", "fontWeight": "600",
        },
    ),
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
    dcc.Store(id="spage-done",       data=0),
    # Fires once after 500 ms to guarantee initial plot render in Dash 4
    dcc.Interval(id="startup-trigger", interval=500, max_intervals=1),

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
                        config={
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                            "toImageButtonOptions": {"format": "png", "scale": 2},
                        },
                        style={"height": "100%"},
                    )
                ], id="spatial-panel", style={"flex": "1.65", "minWidth": "0"}),

                html.Div([
                    dcc.Graph(
                        id="umap-plot",
                        config={
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                        },
                        style={"height": "100%"},
                    )
                ], id="umap-panel", style={"flex": "1", "minWidth": "0"}),
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
                        # Right: server log (fills remaining height)
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

@app.callback(
    Output("cluster-ctrl", "style"),
    Output("gene-ctrl",    "style"),
    Input("color-by", "value"),
)
def toggle_controls(color_by):
    show = {"marginBottom": "14px"}
    hide = {"marginBottom": "14px", "display": "none"}
    return (show, hide) if color_by == "cluster" else (hide, show) if color_by == "gene" else (hide, hide)


@app.callback(
    Output("morph-controls", "style"),
    Input("morph-enable", "value"),
)
def toggle_morph_controls(enabled):
    return {} if "show" in (enabled or []) else {"display": "none"}


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
    Input("overlay-refresh-btn","n_clicks"),
    Input("spatial-relayout",   "data"),
    Input("baysor-done",        "data"),
    Input("baysor-active",      "value"),
    Input("spage-done",         "data"),
    Input("startup-trigger",    "n_intervals"),
)
def update_plots(color_by, method, gene, size, opacity, boundary_toggles,
                 _annot_done, morph_enable, morph_zlevel, morph_channels,
                 morph_brightness, morph_opacity, _refresh, relayout,
                 _baysor_done, baysor_active, _spage_done, _startup):
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
            patched = Patch()
            patched["layout"]["images"] = [morph_image] if morph_image else []
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
        patched = Patch()
        patched["layout"]["images"] = [morph_image] if morph_image else []
        return patched, no_update

    color_by         = color_by or "cluster"
    method           = method   or cluster_methods[0]
    gene             = gene     or sorted(gene_names)[0]
    size             = size     or 2
    opacity          = opacity  or 0.85
    boundary_toggles = boundary_toggles or []
    baysor_on        = "yes" in (baysor_active or [])

    show_cell_bounds = "cell"    in boundary_toggles
    show_nuc_bounds  = "nucleus" in boundary_toggles

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

    return (
        make_spatial_fig(color_by, method, gene, size, opacity,
                         show_cell_bounds, show_nuc_bounds, relayout,
                         morph_image=morph_image, extra_title=morph_title,
                         baysor_active=baysor_on),
        make_umap_fig(color_by, method, gene, size, opacity),
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
    if cell_id not in df.index:
        return html.Div(f"Cell '{cell_id}' not found.", style={"color": "#f85149"})

    row    = df.loc[cell_id]
    method = method or cluster_methods[0]
    col    = cluster_col(method)
    clust  = int(row[col]) if col in row and not pd.isna(row[col]) else "—"

    # Top expressed genes
    cell_pos     = df.index.get_loc(cell_id)
    expr_row_idx = DATA["df_to_expr"][cell_pos]
    if expr_row_idx >= 0:
        raw      = DATA["expr"][expr_row_idx, :].toarray().flatten()
        top_idx  = np.argsort(raw)[::-1][:12]
        top_genes = [(DATA["gene_names"][i], int(raw[i])) for i in top_idx if raw[i] > 0]
    else:
        top_genes = []

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

    umap_1 = row.get("umap_1", float("nan"))
    umap_2 = row.get("umap_2", float("nan"))

    return html.Div([
        html.Div([
            html.Span("Cell: ", style={"color": MUTED, "fontSize": "12px"}),
            html.Span(cell_id, style={"color": ACCENT, "fontWeight": "700", "fontSize": "13px"}),
        ], style={"marginBottom": "8px"}),

        html.Div([
            info_chip("Cluster",     str(clust)),
            info_chip("X",           f"{row['x_centroid']:.1f} µm"),
            info_chip("Y",           f"{row['y_centroid']:.1f} µm"),
            info_chip("Transcripts", str(int(row["transcript_counts"]))),
            info_chip("Cell Area",   f"{row['cell_area']:.0f} µm²"),
            info_chip("Nuc Area",    f"{row['nucleus_area']:.0f} µm²"),
            info_chip("UMAP 1",      f"{umap_1:.2f}" if not np.isnan(umap_1) else "—"),
            info_chip("UMAP 2",      f"{umap_2:.2f}" if not np.isnan(umap_2) else "—"),
        ], style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "10px"}),

        html.Div("Top Expressed Genes",
                 style={"fontSize": "11px", "color": MUTED, "marginBottom": "4px"}),
        dcc.Graph(figure=bar_fig, config={"displayModeBar": False}),
    ])


@app.callback(
    Output("annot-celltypist-div", "style"),
    Output("annot-seurat-div",     "style"),
    Input("annot-source",          "value"),
    prevent_initial_call=True,
)
def toggle_annot_source(source):
    show = {}
    hide = {"display": "none"}
    return (show, hide) if source == "celltypist" else (hide, show)


@app.callback(
    Output("annot-poll",    "disabled"),
    Output("annot-status",  "children"),
    Output("annot-btn",     "disabled"),
    Input("annot-btn",      "n_clicks"),
    State("annot-source",   "value"),
    State("annot-model",    "value"),
    State("annot-rds-path", "value"),
    State("annot-label-col","value"),
    prevent_initial_call=True,
)
def start_annotation(n_clicks, source, model_name, rds_path, label_col):
    """Kick off background annotation thread."""
    with _annot_lock:
        if _annot_state["status"] == "running":
            return False, "Already running…", True
        _annot_state.update({"status": "running", "message": "Starting…", "labels": None})
    if source == "seurat":
        ctx = contextvars.copy_context()
        threading.Thread(
            target=lambda: ctx.run(_run_seurat_annotation, rds_path or "", label_col or "Names"),
            daemon=True,
        ).start()
    else:
        threading.Thread(target=_run_celltypist, args=(model_name,), daemon=True).start()
    return False, "Starting…", True


@app.callback(
    Output("annot-status",  "children",   allow_duplicate=True),
    Output("annot-poll",    "disabled",   allow_duplicate=True),
    Output("annot-btn",     "disabled",   allow_duplicate=True),
    Output("color-by",      "options"),
    Output("annot-done",    "data"),
    Input("annot-poll",     "n_intervals"),
    State("color-by",       "options"),
    State("annot-done",     "data"),
    prevent_initial_call=True,
)
def poll_annotation(_, current_options, done_version):
    """Poll annotation state and update UI when done."""
    with _annot_lock:
        status  = _annot_state["status"]
        message = _annot_state["message"]
        done    = (status == "done")
        error   = (status == "error")

    options = []
    for opt in current_options:
        if opt["value"] == "cell_type":
            options.append({**opt, "disabled": not done})
        else:
            options.append(opt)

    if done:
        return f"✓ {message}", True, False, options, (done_version or 0) + 1
    if error:
        return f"✗ {message}", True, False, options, done_version
    return message, False, True, options, done_version   # keep polling


@app.callback(
    Output("umap-panel",   "style"),
    Output("spatial-panel", "style"),
    Output("umap-toggle",  "children"),
    Input("umap-toggle",   "n_clicks"),
)
def toggle_umap(n_clicks):
    if n_clicks % 2 == 1:
        # UMAP hidden — spatial takes full width
        return (
            {"display": "none"},
            {"flex": "1", "minWidth": "0"},
            "Show UMAP",
        )
    else:
        # UMAP visible — default split
        return (
            {"flex": "1", "minWidth": "0"},
            {"flex": "1.65", "minWidth": "0"},
            "Hide UMAP",
        )


# ─── Baysor callbacks ─────────────────────────────────────────────────────────

@app.callback(
    Output("baysor-prior-conf-div", "style"),
    Input("baysor-use-prior", "value"),
)
def toggle_prior_conf(use_prior):
    return {} if "yes" in (use_prior or []) else {"display": "none"}


@app.callback(
    Output("baysor-poll",       "disabled"),
    Output("baysor-status",     "children"),
    Output("baysor-run-btn",    "disabled"),
    Input("baysor-run-btn",     "n_clicks"),
    State("baysor-scale",       "value"),
    State("baysor-min-mol",     "value"),
    State("baysor-use-prior",   "value"),
    State("baysor-prior-conf",  "value"),
    prevent_initial_call=True,
)
def start_baysor(n_clicks, scale, min_mol, use_prior, prior_conf):
    with _baysor_lock:
        if _baysor_state["status"] == "running":
            return False, "Already running…", True
        _baysor_state.update({"status": "running", "message": "Starting…", "result": None})
    scale     = scale     or 20
    min_mol   = min_mol   or 10
    prior_conf= prior_conf or 0.5
    use_prior_bool = "yes" in (use_prior or [])
    threading.Thread(
        target=_run_baysor,
        args=(float(scale), int(min_mol), use_prior_bool, float(prior_conf)),
        daemon=True,
    ).start()
    return False, "Starting Baysor…", True


@app.callback(
    Output("baysor-status",  "children",  allow_duplicate=True),
    Output("baysor-poll",    "disabled",  allow_duplicate=True),
    Output("baysor-run-btn", "disabled",  allow_duplicate=True),
    Output("baysor-done",    "data"),
    Input("baysor-poll",     "n_intervals"),
    State("baysor-done",     "data"),
    prevent_initial_call=True,
)
def poll_baysor(_, done_version):
    with _baysor_lock:
        status  = _baysor_state["status"]
        message = _baysor_state["message"]

    if status == "done":
        return f"✓ {message}", True, False, (done_version or 0) + 1
    if status == "error":
        return f"✗ {message}", True, False, done_version
    return message, False, True, done_version


# ─── SpaGE callbacks ──────────────────────────────────────────────────────────

@app.callback(
    Output("spage-poll",    "disabled"),
    Output("spage-status",  "children"),
    Output("spage-run-btn", "disabled"),
    Input("spage-run-btn",  "n_clicks"),
    State("spage-rds-path", "value"),
    State("spage-npv",      "value"),
    State("spage-genes",    "value"),
    prevent_initial_call=True,
)
def start_spage(n_clicks, rds_path, n_pv, genes_input):
    with _spage_lock:
        if _spage_state["status"] == "running":
            return False, "Already running…", True
        _spage_state.update({"status": "running", "message": "Starting…", "result": None})
    rds_path = rds_path or "/Users/ikuz/Documents/XeniumWorkflow/Post_R3_FINAL_with_counts.rds"
    n_pv = int(n_pv or 50)
    genes_str = genes_input or ""
    # Copy the current contextvars context so rpy2 conversion rules are available in the thread
    ctx = contextvars.copy_context()
    threading.Thread(
        target=lambda: ctx.run(_run_spage_imputation, rds_path, n_pv, genes_str),
        daemon=True,
    ).start()
    return False, "Starting SpaGE…", True


@app.callback(
    Output("spage-status",    "children",  allow_duplicate=True),
    Output("spage-poll",      "disabled",  allow_duplicate=True),
    Output("spage-run-btn",   "disabled",  allow_duplicate=True),
    Output("spage-done",      "data"),
    Input("spage-poll",       "n_intervals"),
    State("spage-done",       "data"),
    prevent_initial_call=True,
)
def poll_spage(_, done_version):
    with _spage_lock:
        status  = _spage_state["status"]
        message = _spage_state["message"]

    if status == "done":
        return f"✓ {message}", True, False, (done_version or 0) + 1
    if status == "error":
        return f"✗ {message}", True, False, done_version
    return message, False, True, done_version


@app.callback(
    Output("gene-selector", "options"),
    Input("spage-done", "data"),
    prevent_initial_call=True,
)
def update_gene_options(_spage_version):
    """Rebuild gene dropdown: native genes + imputed genes after SpaGE."""
    base = [{"label": g, "value": g} for g in sorted(DATA["gene_names"])]
    with _spage_lock:
        result = _spage_state.get("result")
    if result is not None:
        imp = [{"label": f"{g} [imp]", "value": f"{g} [imp]"}
               for g in sorted(result.columns)]
        return base + imp
    return base


@app.callback(
    Output("info-bar-body",    "style"),
    Output("info-bar-toggle",  "children"),
    Input("info-bar-toggle",   "n_clicks"),
    prevent_initial_call=True,
)
def toggle_info_bar(n):
    if n % 2 == 1:  # collapsed
        return {"display": "none"}, "▲"
    return {"display": "flex", "gap": "10px", "height": "100%", "overflow": "hidden"}, "▼"


@app.callback(
    Output("server-log", "children"),
    Input("log-poll", "n_intervals"),
)
def update_server_log(_):
    with _log_lock:
        lines = list(_log_buffer)
    return "\n".join(lines[-60:])  # last 60 lines


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    print("\nXenium Explorer running at http://localhost:8050\n", flush=True)
    app.run(debug=False, host="0.0.0.0", port=8050)
