#!/usr/bin/env python3
"""
run_split.py — Standalone RCTD + SPLIT::purify ambient RNA correction

Reads expression counts and cell metadata directly from a SpatialData zarr store,
runs RCTD (doublet mode) + SPLIT::purify(), then writes X_corrected + cluster/UMAP
obs columns back to the same zarr.

Usage
-----
  python run_split.py <sdata_path> <rds_path> [options]

  python run_split.py /path/to/sdata.zarr /path/to/reference.rds
  python run_split.py /path/to/sdata.zarr /path/to/reference.rds \\
      --label_col Names --min_umi 10 --min_umi_sigma 100 --max_cores 8

Importable
----------
  from run_split import run_split_from_zarr
  run_split_from_zarr(
      sdata_path="/path/to/sdata.zarr",
      rds_path="/path/to/reference.rds",
  )

Requirements
------------
  R packages: spacexr (dmcable/spacexr), SPLIT (bdsc-tds/SPLIT), SeuratObject, Matrix
  Python:     rpy2, zarr, scipy, numpy, pandas, scikit-learn
  Optional:   umap-learn (for UMAP computation)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# zarr helpers (inline copies of export_to_cellnest.py helpers)
# ---------------------------------------------------------------------------

def _read_zarr_sparse(zroot, path):
    """Read a CSR sparse matrix stored as a zarr group (anndata encoding)."""
    import zarr
    grp = zroot[path]
    if isinstance(grp, zarr.Array):
        return sp.csr_matrix(grp[:])
    try:
        data    = grp["data"][:]
        indices = grp["indices"][:]
        indptr  = grp["indptr"][:]
        shape   = tuple(grp.attrs.get("shape", grp.attrs.get("h5sparse_shape", None)))
        if shape is None:
            raise KeyError("No shape attribute")
        return sp.csr_matrix((data, indices, indptr), shape=shape)
    except KeyError:
        return sp.csr_matrix(np.array(grp))


def _read_uns_list(table_grp, key):
    """Read a list stored in uns zarr group. Returns [] if absent."""
    import zarr
    try:
        uns_grp = table_grp["uns"]
    except KeyError:
        return []
    if key not in uns_grp:
        return []
    item = uns_grp[key]
    if isinstance(item, zarr.Array):
        return [str(v) for v in item[:]]
    try:
        return [str(v) for v in item]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# R output redirect (no Dash log buffer — write straight to stdout)
# ---------------------------------------------------------------------------

def _redirect_rpy2_console_stdout():
    import warnings
    warnings.filterwarnings("ignore", message="R is not initialized by the main thread")
    try:
        import rpy2.rinterface_lib.callbacks as rpy2_cb
        def _r_write(x):
            x = x.rstrip()
            if x:
                print(f"[R] {x}", flush=True)
        rpy2_cb.consolewrite_print     = _r_write
        rpy2_cb.consolewrite_warnerror = _r_write
    except Exception:
        pass


# ---------------------------------------------------------------------------
# R dgCMatrix → scipy CSC
# ---------------------------------------------------------------------------

def _r_dgcmatrix_to_scipy(r_mat):
    """Convert an R dgCMatrix (genes × cells, column-sparse) to scipy CSC."""
    i    = np.array(r_mat.slots["i"])
    p    = np.array(r_mat.slots["p"])
    x    = np.array(r_mat.slots["x"])
    dims = list(r_mat.slots["Dim"])
    return sp.csc_matrix((x, i, p), shape=(dims[0], dims[1]))


# ---------------------------------------------------------------------------
# KMeans + UMAP on corrected counts
# ---------------------------------------------------------------------------

def _compute_split_clusters_umap(cells_df, corrected_mat, corrected_cell_ids=None):
    """Compute KMeans (k=10) + UMAP on corrected counts.
    Writes cluster_split_10, split_umap_1, split_umap_2 into cells_df in-place.
    """
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans

    try:
        import umap as _umap_mod
        _have_umap = True
    except Exception:
        _have_umap = False
        print("  SPLIT: umap-learn not installed — skipping UMAP", flush=True)

    n_cells, n_genes = corrected_mat.shape
    print(f"  SPLIT: normalising {n_cells:,} cells × {n_genes} genes…", flush=True)

    mat = corrected_mat if sp.issparse(corrected_mat) else sp.csr_matrix(corrected_mat)

    # Log-normalise
    log_mat = mat.copy().astype(np.float32)
    log_mat.data = np.log1p(log_mat.data)
    normed = normalize(log_mat, norm="l2", axis=1)

    # PCA via truncated SVD
    n_comp = min(50, n_cells - 1, n_genes - 1)
    print(f"  SPLIT: SVD ({n_comp} components)…", flush=True)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    pca = svd.fit_transform(normed)

    # KMeans
    print("  SPLIT: KMeans clustering (k=10)…", flush=True)
    km     = KMeans(n_clusters=10, random_state=42, n_init=10)
    labels = km.fit_predict(pca)

    # UMAP
    embedding = None
    if _have_umap:
        print("  SPLIT: computing UMAP…", flush=True)
        n_neighbors = min(30, n_cells - 1)
        reducer   = _umap_mod.UMAP(n_components=2, n_neighbors=n_neighbors,
                                    min_dist=0.3, random_state=42)
        embedding = reducer.fit_transform(pca)

    # Write back — RCTD may drop low-UMI cells, so handle subset case
    if corrected_cell_ids is not None and len(corrected_cell_ids) < len(cells_df):
        try:
            sample_id = cells_df.index[0]
            cast      = type(sample_id)
            int_idx   = pd.Index([cast(c) for c in corrected_cell_ids])
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


# ---------------------------------------------------------------------------
# Write corrected counts to zarr
# ---------------------------------------------------------------------------

def _write_split_to_zarr(sdata_path, corrected_mat, cells_df, panel_genes,
                          corrected_cell_ids=None):
    """Write X_corrected layer + cluster/UMAP obs columns to SpatialData zarr."""
    import zarr
    try:
        from anndata.io import write_elem as _write_elem
    except ImportError:
        from anndata.experimental import write_elem as _write_elem

    zroot     = zarr.open_group(sdata_path, mode="r+", use_consolidated=False)
    if "tables" not in zroot or "table" not in zroot["tables"]:
        print("  SPLIT: no table in zarr — skipping zarr write", flush=True)
        return
    table_grp = zroot["tables"]["table"]
    obs_grp   = table_grp["obs"]

    # Read zarr obs index for alignment
    idx_key  = obs_grp.attrs.get("_index", "_index")
    zarr_ids = [str(v) for v in obs_grp[idx_key][:]]

    # Map corrected cell IDs (RCTD may drop low-UMI cells) to row indices
    if corrected_cell_ids is not None:
        id_to_row = {str(cid): i for i, cid in enumerate(corrected_cell_ids)}
    else:
        id_to_row = {str(cid): i for i, cid in enumerate(cells_df.index)}
    row_indices = [id_to_row.get(cid, -1) for cid in zarr_ids]

    mat      = corrected_mat if sp.issparse(corrected_mat) else sp.csr_matrix(corrected_mat)
    n_zarr   = len(zarr_ids)
    valid_out = [i for i, r in enumerate(row_indices) if r >= 0]
    valid_in  = [r for r in row_indices if r >= 0]
    if valid_in:
        mat_sub = mat[valid_in, :]
        coo     = mat_sub.tocoo()
        new_row = np.array(valid_out)[coo.row]
        mat_aligned = sp.csr_matrix(
            (coo.data, (new_row, coo.col)),
            shape=(n_zarr, mat.shape[1])
        )
    else:
        mat_aligned = sp.csr_matrix((n_zarr, mat.shape[1]), dtype=mat.dtype)
    print(f"  SPLIT: aligned {len(valid_in)}/{n_zarr} cells for zarr write", flush=True)

    # Pad to match zarr var width (imputed gene cols → 0)
    var_grp = table_grp["var"]
    var_key = var_grp.attrs.get("_index", "_index")
    n_var   = len(var_grp[var_key])
    if mat_aligned.shape[1] < n_var:
        n_pad       = n_var - mat_aligned.shape[1]
        mat_aligned = sp.hstack(
            [mat_aligned, sp.csr_matrix((n_zarr, n_pad), dtype=mat_aligned.dtype)],
            format="csr",
        )
        print(f"  SPLIT: padded X_corrected to {mat_aligned.shape} "
              f"(+{n_pad} imputed gene cols = 0)", flush=True)

    # Write X_corrected layer
    layers_grp = table_grp.require_group("layers")
    if "X_corrected" in layers_grp:
        del layers_grp["X_corrected"]
    _write_elem(layers_grp, "X_corrected", mat_aligned)

    # Write obs cluster/UMAP columns
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


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def run_split_from_zarr(
    sdata_path: str,
    rds_path: str,
    label_col: str = "Names",
    max_cores: int = 4,
    min_umi: int = 10,
    min_umi_sigma: int = 100,
    compute_umap: bool = True,
) -> None:
    """Run RCTD (doublet mode) + SPLIT::purify() on a SpatialData zarr store.

    Parameters
    ----------
    sdata_path : str
        Path to SpatialData zarr directory (must contain tables/table).
    rds_path : str
        Path to Seurat RDS reference file.
    label_col : str
        Column in RDS meta.data for cell type labels (default "Names").
    max_cores : int
        CPU cores for RCTD (default 4).
    min_umi : int
        RCTD UMI_min — cells below this threshold excluded (default 10).
    min_umi_sigma : int
        RCTD UMI_min_sigma threshold (default 100).
    compute_umap : bool
        Whether to compute KMeans clusters + UMAP on corrected counts (default True).
    """
    import zarr
    import rpy2.robjects as ro
    import rpy2.robjects.conversion as _rconv
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr

    sdata_path = os.path.abspath(sdata_path)
    rds_path   = os.path.abspath(rds_path)

    print(f"  SPLIT: sdata_path={sdata_path}", flush=True)
    print(f"  SPLIT: rds_path={rds_path}", flush=True)
    print(f"  SPLIT: min_umi={min_umi}, min_umi_sigma={min_umi_sigma}, "
          f"max_cores={max_cores}", flush=True)

    # ── Setup ────────────────────────────────────────────────────────────────
    _redirect_rpy2_console_stdout()
    _rconv.set_conversion(ro.default_converter)
    pandas2ri.activate()

    # ── Step 1: Read expression matrix + metadata from zarr ──────────────────
    print("  SPLIT [1/8]: reading zarr…", flush=True)
    zroot     = zarr.open_group(sdata_path, mode="r", use_consolidated=False)
    table_grp = zroot["tables"]["table"]
    obs_grp   = table_grp["obs"]
    var_grp   = table_grp["var"]

    obs_idx_key = obs_grp.attrs.get("_index", "_index")
    cell_ids    = [str(v) for v in obs_grp[obs_idx_key][:]]

    var_idx_key = var_grp.attrs.get("_index", "_index")
    gene_names  = [str(v) for v in var_grp[var_idx_key][:]]

    # Identify imputed genes — use only panel genes for SPLIT
    raw_imp  = set(_read_uns_list(table_grp, "imputed_genes"))
    corr_imp = set(_read_uns_list(table_grp, "split_corrected_imputed_genes"))
    all_imp  = raw_imp | corr_imp
    panel_mask  = np.array([g not in all_imp for g in gene_names], dtype=bool)
    panel_genes = [g for g, m in zip(gene_names, panel_mask) if m]

    print(f"  SPLIT: {len(cell_ids):,} cells, {len(panel_genes)} panel genes "
          f"(of {len(gene_names)} total, {len(all_imp)} imputed excluded)", flush=True)

    # Read expression — panel genes only
    expr_full  = _read_zarr_sparse(table_grp, "X")
    expr_panel = expr_full[:, panel_mask]   # cells × panel_genes
    n_cells, n_panel = expr_panel.shape

    # Build gene name → column index map for shared-gene lookup
    gni = {g: i for i, g in enumerate(panel_genes)}

    # Read coordinates — try Xenium naming, then Proseg naming
    x = y = None
    for xcol, ycol in [("x_centroid", "y_centroid"), ("centroid_x", "centroid_y")]:
        if xcol in obs_grp and ycol in obs_grp:
            x = obs_grp[xcol][:].astype(np.float64)
            y = obs_grp[ycol][:].astype(np.float64)
            print(f"  SPLIT: coordinates from '{xcol}'/'{ycol}'", flush=True)
            break
    if x is None:
        raise KeyError(
            "Could not find centroid columns in obs. "
            "Expected 'x_centroid'/'y_centroid' or 'centroid_x'/'centroid_y'."
        )

    cells_df = pd.DataFrame(
        {"x_centroid": x, "y_centroid": y},
        index=pd.Index(cell_ids)
    )

    # ── Step 2: Load R packages ──────────────────────────────────────────────
    print("  SPLIT [2/8]: loading R packages (spacexr, SPLIT)…", flush=True)
    try:
        importr("spacexr")
    except Exception:
        raise RuntimeError(
            "spacexr not installed. In R: devtools::install_github('dmcable/spacexr')"
        )
    try:
        importr("SPLIT")
    except Exception:
        raise RuntimeError(
            "SPLIT not installed. In R: remotes::install_github('bdsc-tds/SPLIT')"
        )
    importr("SeuratObject")
    importr("Matrix")
    base = importr("base")

    # ── Step 3: Load Seurat reference ────────────────────────────────────────
    print("  SPLIT [3/8]: loading Seurat reference…", flush=True)
    rds     = base.readRDS(rds_path)
    meta_r  = ro.r['slot'](rds, "meta.data")
    meta_df = pandas2ri.rpy2py(meta_r)
    if label_col not in meta_df.columns:
        raise KeyError(
            f"Column '{label_col}' not found in RDS meta.data. "
            f"Available: {list(meta_df.columns)[:20]}"
        )
    print(f"  SPLIT: reference {len(meta_df):,} cells, "
          f"{meta_df[label_col].nunique()} cell types", flush=True)

    # ── Step 4: Find shared genes ────────────────────────────────────────────
    print("  SPLIT [4/8]: finding shared genes…", flush=True)
    ro.r.assign("._split_rds", rds)
    mat_r     = ro.r("slot(slot(._split_rds, 'assays')[['RNA']], 'counts')")
    ref_genes = list(ro.r['rownames'](mat_r))
    shared_genes = sorted(set(panel_genes) & set(ref_genes))
    if len(shared_genes) < 10:
        raise ValueError(
            f"Only {len(shared_genes)} shared genes between reference and panel. "
            "Check that rds_path matches the Xenium panel."
        )
    print(f"  SPLIT: {len(shared_genes)} shared genes", flush=True)

    # ── Step 5: Build RCTD Reference ────────────────────────────────────────
    print("  SPLIT [5/8]: building RCTD Reference…", flush=True)
    ro.r.assign("._split_genes", ro.StrVector(shared_genes))
    ro.r("""
._split_ref_mat <- slot(slot(._split_rds, 'assays')[['RNA']], 'counts')[._split_genes, , drop=FALSE]
._split_ref_mat <- as(._split_ref_mat, 'dgCMatrix')
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

    # ── Step 6: Build SpatialRNA object ──────────────────────────────────────
    print(f"  SPLIT [6/8]: building SpatialRNA ({n_cells:,} cells)…", flush=True)
    shared_idx           = [gni[g] for g in shared_genes if g in gni and gni[g] < n_panel]
    shared_genes_present = [g for g in shared_genes if g in gni and gni[g] < n_panel]
    expr_shared = expr_panel[:, shared_idx]   # cells × shared_genes
    expr_t      = expr_shared.T.toarray() if hasattr(expr_shared, "toarray") else expr_shared.T

    counts_r_mat = ro.r.matrix(
        ro.FloatVector(expr_t.flatten(order="F").tolist()),
        nrow=len(shared_genes_present),
        ncol=n_cells,
    )
    ro.r.assign("._split_counts_mat", counts_r_mat)
    ro.r.assign("._split_rownames",   ro.StrVector(shared_genes_present))
    ro.r.assign("._split_colnames",   ro.StrVector(cell_ids))
    ro.r("""
rownames(._split_counts_mat) <- ._split_rownames
colnames(._split_counts_mat) <- ._split_colnames
._split_counts_mat <- as(._split_counts_mat, 'dgCMatrix')
""")

    coords_df = cells_df[["x_centroid", "y_centroid"]].copy()
    coords_df.columns = ["x", "y"]
    coords_r = pandas2ri.py2rpy(coords_df)
    ro.r.assign("._split_coords", coords_r)
    ro.r("""
rownames(._split_coords) <- colnames(._split_counts_mat)
._split_numi <- colSums(._split_counts_mat)
._split_spatialrna <- spacexr::SpatialRNA(
    coords  = ._split_coords,
    counts  = ._split_counts_mat,
    nUMI    = ._split_numi
)
""")

    # ── Step 7: Run RCTD doublet mode ────────────────────────────────────────
    print(f"  SPLIT [7/8]: running RCTD doublet mode "
          f"(max_cores={max_cores}, min_umi={min_umi})… [~20-40 min]", flush=True)
    ro.r.assign("._split_max_cores",     ro.IntVector([max_cores]))
    ro.r.assign("._split_min_umi",       ro.IntVector([min_umi]))
    ro.r.assign("._split_min_umi_sigma", ro.IntVector([min_umi_sigma]))
    ro.r("""
._split_rctd <- spacexr::create.RCTD(._split_spatialrna, ._split_reference,
    max_cores=._split_max_cores[1], CELL_MIN_INSTANCE=5,
    UMI_min=._split_min_umi[1], UMI_min_sigma=._split_min_umi_sigma[1])
._split_rctd <- spacexr::run.RCTD(._split_rctd, doublet_mode='doublet')
._split_rctd <- SPLIT::run_post_process_RCTD(._split_rctd)
""")
    print("  SPLIT: RCTD done, running purify()…", flush=True)

    # Full panel matrix (genes × cells) for purify
    expr_full_t = expr_panel.T.toarray() if hasattr(expr_panel, "toarray") else expr_panel.T
    full_r_mat  = ro.r.matrix(
        ro.FloatVector(expr_full_t.flatten(order="F").tolist()),
        nrow=n_panel,
        ncol=n_cells,
    )
    ro.r.assign("._split_full_counts", full_r_mat)
    ro.r.assign("._split_panel_genes", ro.StrVector(list(panel_genes)))
    ro.r("""
rownames(._split_full_counts) <- ._split_panel_genes
colnames(._split_full_counts) <- ._split_colnames
._split_full_counts <- as(._split_full_counts, 'dgCMatrix')
._split_res      <- SPLIT::purify(
    counts              = ._split_full_counts,
    rctd               = ._split_rctd,
    DO_purify_singlets  = TRUE
)
._split_purified <- ._split_res$purified_counts
""")

    # ── Step 8: Convert back to Python ───────────────────────────────────────
    print("  SPLIT [8/8]: converting corrected counts to Python…", flush=True)
    purified_r = ro.r["._split_purified"]
    try:
        corrected_cell_ids = list(ro.r["colnames"](purified_r))
    except Exception:
        corrected_cell_ids = None

    purified_csc = _r_dgcmatrix_to_scipy(purified_r)   # genes × cells (CSC)
    corrected_mat = purified_csc.T.tocsr()               # cells × genes (CSR)
    corrected_mat.data = np.clip(corrected_mat.data, 0, None)
    n_corr = corrected_mat.shape[0]
    print(f"  SPLIT: corrected matrix shape {corrected_mat.shape} "
          f"({n_corr}/{n_cells} cells)", flush=True)

    if corrected_cell_ids is None:
        corrected_cell_ids = cell_ids[:n_corr]

    # ── Optional: clusters + UMAP ────────────────────────────────────────────
    if compute_umap:
        _compute_split_clusters_umap(cells_df, corrected_mat,
                                     corrected_cell_ids=corrected_cell_ids)

    # ── Write to zarr ────────────────────────────────────────────────────────
    print(f"  SPLIT: writing corrected counts to zarr…", flush=True)
    _write_split_to_zarr(sdata_path, corrected_mat, cells_df, panel_genes,
                         corrected_cell_ids=corrected_cell_ids)

    print("  SPLIT: done.", flush=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run RCTD + SPLIT::purify ambient RNA correction on a SpatialData zarr",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("sdata_path", help="Path to SpatialData zarr directory")
    parser.add_argument("rds_path",   help="Path to Seurat RDS reference file")
    parser.add_argument("--label_col",      default="Names",
                        help="Column in RDS meta.data for cell type labels (default: Names)")
    parser.add_argument("--max_cores",      type=int, default=4,
                        help="CPU cores for RCTD (default: 4)")
    parser.add_argument("--min_umi",        type=int, default=10,
                        help="RCTD UMI_min threshold (default: 10)")
    parser.add_argument("--min_umi_sigma",  type=int, default=100,
                        help="RCTD UMI_min_sigma threshold (default: 100)")
    parser.add_argument("--no_umap",        action="store_true",
                        help="Skip KMeans + UMAP computation")
    args = parser.parse_args()

    run_split_from_zarr(
        sdata_path    = args.sdata_path,
        rds_path      = args.rds_path,
        label_col     = args.label_col,
        max_cores     = args.max_cores,
        min_umi       = args.min_umi,
        min_umi_sigma = args.min_umi_sigma,
        compute_umap  = not args.no_umap,
    )


if __name__ == "__main__":
    main()
