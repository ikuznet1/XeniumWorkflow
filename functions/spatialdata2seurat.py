#!/usr/bin/env python3
"""
spatialdata2seurat.py — Convert a SpatialData zarr to a Seurat RDS

Produces a Seurat object equivalent to LoadXenium() output:
  - @assays$Xenium      : raw panel counts (genes × cells, dgCMatrix)
  - @assays$Imputed     : SpaGE-imputed genes, if present (optional, --no_imputed to skip)
  - @meta.data          : cell_id, nCount_Xenium, nFeature_Xenium, cell_area,
                          nucleus_area, x_centroid, y_centroid, plus any obs columns
                          (clusters, cell types, umap_1/2, etc.)
  - @images$fov         : CreateFOV with centroids + cell/nucleus segmentation polygons

Usage:
  python spatialdata2seurat.py <sdata_path> <output_rds> [options]

Positional:
  sdata_path            Path to SpatialData zarr directory
  output_rds            Output .rds file path

Options:
  --assay NAME          Seurat assay name (default: Xenium)
  --fov NAME            FOV name in @images (default: fov)
  --no_boundaries       Skip cell segmentation polygons
  --no_nucleus          Skip nucleus segmentation polygons
  --no_imputed          Exclude imputed genes (default: include as separate assay)
  --use_corrected       Use X_corrected layer (SPLIT) as counts instead of X

Example:
  python spatialdata2seurat.py /data/xenium.zarr /data/xenium.rds
  python spatialdata2seurat.py /data/xenium.zarr /data/xenium.rds \\
      --assay Xenium --fov fov --no_nucleus
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io as sio


# ── Polygon helpers ────────────────────────────────────────────────────────────

def _shapes_to_longdf(gdf, cell_ids_set=None) -> pd.DataFrame:
    """
    Convert a GeoDataFrame of Polygon shapes to a long-format DataFrame
    with columns: cell (str), x (float), y (float).
    Optionally filtered to cell_ids_set for alignment.
    """
    rows = []
    for cid, geom in zip(gdf.index, gdf.geometry):
        cid_str = str(cid)
        if cell_ids_set is not None and cid_str not in cell_ids_set:
            continue
        try:
            if geom.geom_type == "MultiPolygon":
                coords = list(geom.geoms[0].exterior.coords)
            else:
                coords = list(geom.exterior.coords)
            for x, y in coords:
                rows.append((cid_str, float(x), float(y)))
        except Exception:
            pass
    return pd.DataFrame(rows, columns=["cell", "x", "y"])


# ── Main conversion function ───────────────────────────────────────────────────

def spatialdata2seurat(
    sdata_path: str,
    output_rds: str,
    assay_name: str = "Xenium",
    fov_name: str = "fov",
    include_boundaries: bool = True,
    include_nucleus: bool = True,
    include_imputed: bool = True,
    use_corrected: bool = False,
) -> str:
    """
    Convert a SpatialData zarr to a Seurat RDS file.

    Parameters
    ----------
    sdata_path          : Path to input SpatialData zarr
    output_rds          : Output .rds file path
    assay_name          : Seurat assay name (default "Xenium")
    fov_name            : Name of the FOV in @images (default "fov")
    include_boundaries  : Include cell segmentation polygons in FOV
    include_nucleus     : Include nucleus segmentation polygons in FOV
    include_imputed     : Include imputed genes as a separate "Imputed" assay
    use_corrected       : Use X_corrected layer (SPLIT output) as counts

    Returns
    -------
    Path to the output RDS file.
    """
    import spatialdata as sd
    import rpy2.robjects as ro
    import rpy2.robjects.conversion as _rconv
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri

    pandas2ri.activate()
    _rconv.set_conversion(ro.default_converter + pandas2ri.converter)

    importr("Seurat")
    importr("SeuratObject")
    importr("Matrix")

    output_rds = os.path.abspath(output_rds)
    os.makedirs(os.path.dirname(output_rds) or ".", exist_ok=True)

    # ── Load SpatialData ───────────────────────────────────────────────────────
    print(f"Reading SpatialData from {sdata_path} …", flush=True)
    sdata = sd.read_zarr(sdata_path)
    adata = sdata.tables["table"]

    print(f"  {len(adata)} cells, {len(adata.var)} genes", flush=True)

    # ── Split panel vs imputed genes ───────────────────────────────────────────
    if "is_imputed" in adata.var.columns:
        panel_mask   = ~adata.var["is_imputed"].astype(bool)
        imputed_mask =  adata.var["is_imputed"].astype(bool)
    else:
        panel_mask   = pd.Series(True,  index=adata.var_names)
        imputed_mask = pd.Series(False, index=adata.var_names)

    panel_genes   = adata.var_names[panel_mask].tolist()
    imputed_genes = adata.var_names[imputed_mask].tolist()
    print(f"  Panel genes: {len(panel_genes)}, imputed: {len(imputed_genes)}", flush=True)

    # ── Choose expression matrix ───────────────────────────────────────────────
    if use_corrected and "X_corrected" in adata.layers:
        print("  Using X_corrected layer (SPLIT)", flush=True)
        X_full = adata.layers["X_corrected"]
    else:
        X_full = adata.X
    if not sp.issparse(X_full):
        X_full = sp.csr_matrix(X_full)

    panel_idx   = [i for i, m in enumerate(panel_mask)   if m]
    imputed_idx = [i for i, m in enumerate(imputed_mask) if m]

    X_panel   = X_full[:, panel_idx]    # cells × panel_genes
    X_imputed = X_full[:, imputed_idx]  # cells × imputed_genes

    cell_ids = list(adata.obs_names)

    # ── Build metadata ─────────────────────────────────────────────────────────
    meta = adata.obs.copy()
    meta.index = [str(c) for c in cell_ids]

    # Compute nCount / nFeature if not already present
    panel_counts = np.asarray(X_panel.sum(axis=1)).ravel().astype(float)
    panel_feats  = np.asarray((X_panel > 0).sum(axis=1)).ravel().astype(int)
    count_col    = f"nCount_{assay_name}"
    feat_col     = f"nFeature_{assay_name}"
    if count_col not in meta.columns:
        meta[count_col] = panel_counts
    if feat_col not in meta.columns:
        meta[feat_col]  = panel_feats

    # Rename centroid columns to match LoadXenium convention if needed
    for src, dst in [("x_centroid", "x_centroid"), ("y_centroid", "y_centroid")]:
        if src in meta.columns and dst not in meta.columns:
            meta[dst] = meta[src]

    # Drop columns that Seurat can't handle (list/dict dtype)
    for col in list(meta.columns):
        if meta[col].dtype == object:
            try:
                meta[col] = meta[col].astype(str)
            except Exception:
                meta = meta.drop(columns=[col])

    print(f"  Meta columns: {list(meta.columns)}", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:

        # ── Write panel matrix to MatrixMarket ────────────────────────────────
        panel_mtx  = os.path.join(tmpdir, "panel.mtx").replace("\\", "/")
        panel_genes_file = os.path.join(tmpdir, "panel_genes.txt").replace("\\", "/")
        panel_cells_file = os.path.join(tmpdir, "panel_cells.txt").replace("\\", "/")

        print("  Writing panel count matrix …", flush=True)
        sio.mmwrite(panel_mtx, X_panel.T.tocsc())  # genes × cells for Seurat
        with open(panel_genes_file, "w") as f:
            f.write("\n".join(panel_genes))
        with open(panel_cells_file, "w") as f:
            f.write("\n".join(str(c) for c in cell_ids))

        # ── Write imputed matrix if needed ────────────────────────────────────
        imp_mtx = imp_genes_file = imp_cells_file = None
        if include_imputed and len(imputed_genes) > 0:
            imp_mtx        = os.path.join(tmpdir, "imputed.mtx").replace("\\", "/")
            imp_genes_file = os.path.join(tmpdir, "imp_genes.txt").replace("\\", "/")
            imp_cells_file = os.path.join(tmpdir, "imp_cells.txt").replace("\\", "/")
            print(f"  Writing imputed matrix ({len(imputed_genes)} genes) …", flush=True)
            sio.mmwrite(imp_mtx, X_imputed.T.tocsc())
            with open(imp_genes_file, "w") as f:
                f.write("\n".join(imputed_genes))
            with open(imp_cells_file, "w") as f:
                f.write("\n".join(str(c) for c in cell_ids))

        # ── Write centroids ───────────────────────────────────────────────────
        cents_file = os.path.join(tmpdir, "centroids.csv").replace("\\", "/")
        cents_df = pd.DataFrame({
            "cell": [str(c) for c in cell_ids],
            "x":    meta["x_centroid"].values if "x_centroid" in meta.columns
                    else np.zeros(len(cell_ids)),
            "y":    meta["y_centroid"].values if "y_centroid" in meta.columns
                    else np.zeros(len(cell_ids)),
        })
        cents_df.to_csv(cents_file, index=False)
        print(f"  Centroids written ({len(cents_df)} cells)", flush=True)

        # ── Write segmentation polygons ───────────────────────────────────────
        cell_seg_file = nuc_seg_file = None
        cell_ids_set  = set(str(c) for c in cell_ids)

        if include_boundaries and "cell_boundaries" in sdata.shapes:
            print("  Extracting cell segmentation polygons …", flush=True)
            cell_seg_df   = _shapes_to_longdf(sdata.shapes["cell_boundaries"], cell_ids_set)
            cell_seg_file = os.path.join(tmpdir, "cell_seg.csv").replace("\\", "/")
            cell_seg_df.to_csv(cell_seg_file, index=False)
            print(f"  Cell seg: {cell_seg_df['cell'].nunique():,} cells, "
                  f"{len(cell_seg_df):,} vertices", flush=True)

        if include_nucleus and "nucleus_boundaries" in sdata.shapes:
            print("  Extracting nucleus segmentation polygons …", flush=True)
            nuc_seg_df   = _shapes_to_longdf(sdata.shapes["nucleus_boundaries"], cell_ids_set)
            nuc_seg_file = os.path.join(tmpdir, "nuc_seg.csv").replace("\\", "/")
            nuc_seg_df.to_csv(nuc_seg_file, index=False)
            print(f"  Nucleus seg: {nuc_seg_df['cell'].nunique():,} cells, "
                  f"{len(nuc_seg_df):,} vertices", flush=True)

        # ── Write metadata CSV ────────────────────────────────────────────────
        meta_file = os.path.join(tmpdir, "meta.csv").replace("\\", "/")
        meta.to_csv(meta_file)
        print("  Metadata written", flush=True)

        # ── Build Seurat object in R ───────────────────────────────────────────
        print("Building Seurat object in R …", flush=True)

        output_rds_r = output_rds.replace("\\", "/")
        assay_r      = assay_name
        fov_r        = fov_name

        ro.r(f"""
# ── Read panel matrix ──────────────────────────────────────────────────────────
.panel_mat   <- Matrix::readMM("{panel_mtx}")
.panel_genes <- readLines("{panel_genes_file}")
.panel_cells <- readLines("{panel_cells_file}")
rownames(.panel_mat) <- .panel_genes
colnames(.panel_mat) <- .panel_cells
.panel_mat   <- as(.panel_mat, "dgCMatrix")

# ── Read metadata ──────────────────────────────────────────────────────────────
.meta_df <- read.csv("{meta_file}", row.names=1, stringsAsFactors=FALSE)

# ── Create Seurat object ───────────────────────────────────────────────────────
.seurat <- CreateSeuratObject(
    counts    = .panel_mat,
    assay     = "{assay_r}",
    meta.data = .meta_df
)

# ── Add imputed assay (if present) ────────────────────────────────────────────
{"" if not (include_imputed and imp_mtx) else f'''
.imp_mat   <- Matrix::readMM("{imp_mtx}")
.imp_genes <- readLines("{imp_genes_file}")
.imp_cells <- readLines("{imp_cells_file}")
rownames(.imp_mat) <- .imp_genes
colnames(.imp_mat) <- .imp_cells
.imp_mat   <- as(.imp_mat, "dgCMatrix")
.seurat[["Imputed"]] <- CreateAssayObject(counts = .imp_mat)
'''}

# ── Build FOV with centroids ──────────────────────────────────────────────────
.cents_df      <- read.csv("{cents_file}", stringsAsFactors=FALSE)
rownames(.cents_df) <- .cents_df$cell
.cents_df$cell <- NULL
.centroids     <- CreateCentroids(.cents_df)

.fov_coords <- list(centroids = .centroids)

{"" if not cell_seg_file else f'''
.cell_seg_long <- read.csv("{cell_seg_file}", stringsAsFactors=FALSE)
.cell_seg_list <- lapply(
    split(.cell_seg_long[, c("x","y")], .cell_seg_long$cell),
    function(df) {{ df }}
)
.fov_coords[["segmentation"]] <- CreateSegmentation(.cell_seg_list)
'''}

{"" if not nuc_seg_file else f'''
.nuc_seg_long <- read.csv("{nuc_seg_file}", stringsAsFactors=FALSE)
.nuc_seg_list <- lapply(
    split(.nuc_seg_long[, c("x","y")], .nuc_seg_long$cell),
    function(df) {{ df }}
)
.fov_coords[["nucleus_segmentation"]] <- CreateSegmentation(.nuc_seg_list)
'''}

.fov <- CreateFOV(
    coords = .fov_coords,
    type   = names(.fov_coords),
    assay  = "{assay_r}"
)
.seurat[["{fov_r}"]] <- .fov

# ── Save ───────────────────────────────────────────────────────────────────────
saveRDS(.seurat, file="{output_rds_r}")
cat(sprintf("Saved: %s\\n", "{output_rds_r}"))

# Cleanup large temporaries
rm(.panel_mat, .meta_df, .cents_df, .centroids, .fov_coords, .fov)
if (exists(".imp_mat"))     rm(.imp_mat)
if (exists(".cell_seg_long")) rm(.cell_seg_long, .cell_seg_list)
if (exists(".nuc_seg_long"))  rm(.nuc_seg_long,  .nuc_seg_list)
gc()
""")

    print(f"Done → {output_rds}", flush=True)
    return output_rds


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Convert a SpatialData zarr to a Seurat RDS (LoadXenium-compatible).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("sdata_path",  help="Input SpatialData zarr path")
    p.add_argument("output_rds",  help="Output .rds file path")
    p.add_argument("--assay",          default="Xenium", metavar="NAME",
                   help="Seurat assay name (default: Xenium)")
    p.add_argument("--fov",            default="fov",     metavar="NAME",
                   help="FOV name in @images (default: fov)")
    p.add_argument("--no_boundaries",  action="store_true",
                   help="Skip cell segmentation polygons")
    p.add_argument("--no_nucleus",     action="store_true",
                   help="Skip nucleus segmentation polygons")
    p.add_argument("--no_imputed",     action="store_true",
                   help="Exclude imputed genes (omits Imputed assay)")
    p.add_argument("--use_corrected",  action="store_true",
                   help="Use X_corrected layer (SPLIT output) as panel counts")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    spatialdata2seurat(
        sdata_path=args.sdata_path,
        output_rds=args.output_rds,
        assay_name=args.assay,
        fov_name=args.fov,
        include_boundaries=not args.no_boundaries,
        include_nucleus=not args.no_nucleus,
        include_imputed=not args.no_imputed,
        use_corrected=args.use_corrected,
    )
