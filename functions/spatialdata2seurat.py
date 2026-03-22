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
  --roi DIR             Directory containing ROI assignment parquet (roi_<cls>.parquet).
                        If omitted, ROI columns are read directly from the zarr obs.
  --roi_cls CLS         ROI class to filter on (obs column will be roi_<CLS>)
  --roi_name NAME       ROI name to keep (cells where roi_<CLS> == NAME)

Example:
  python spatialdata2seurat.py /data/xenium.zarr /data/xenium.rds
  python spatialdata2seurat.py /data/xenium.zarr /data/tumor_roi.rds \\
      --roi_cls tissue --roi_name Tumor_1
  python spatialdata2seurat.py /data/xenium.zarr /data/tumor_roi.rds \\
      --roi ./functions/output --roi_cls tissue --roi_name Tumor_1
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
    roi_dir: str = None,
    roi_cls: str = None,
    roi_name: str = None,
    cache_dir: str = None,
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
    roi_dir             : Optional directory with roi_<cls>.parquet for ROI assignment.
                          If None, ROI columns are read directly from zarr obs.
    roi_cls             : ROI class to filter on (column = roi_<roi_cls>)
    roi_name            : ROI name to keep (rows where roi_<roi_cls> == roi_name)
    cache_dir           : Path to xenium_explorer cache dir (default: ~/.xenium_explorer_cache).
                          Scanned for annotation parquets to add cell_type_seurat and
                          cell_type_rctd_doublet columns to meta.data.

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

    # ── ROI filter ─────────────────────────────────────────────────────────────
    if roi_cls and roi_name:
        roi_col = f"roi_{roi_cls}"

        # If an external ROI directory is given, load the parquet and merge
        if roi_dir:
            roi_pq = os.path.join(roi_dir, f"roi_{roi_cls}.parquet")
            if not os.path.exists(roi_pq):
                # Fall back to any parquet in the directory
                candidates = [f for f in os.listdir(roi_dir) if f.endswith(".parquet")]
                if not candidates:
                    raise FileNotFoundError(f"No parquet found in {roi_dir}")
                roi_pq = os.path.join(roi_dir, candidates[0])
                print(f"  ROI: using {roi_pq}", flush=True)
            roi_df = pd.read_parquet(roi_pq)
            # Expect index = cell_id or a cell_id column
            if "cell_id" in roi_df.columns:
                roi_df = roi_df.set_index("cell_id")
            if roi_col not in roi_df.columns:
                raise KeyError(f"Column '{roi_col}' not found in {roi_pq}. "
                               f"Available: {list(roi_df.columns)}")
            adata.obs[roi_col] = roi_df[roi_col].reindex(adata.obs_names)
            print(f"  ROI: loaded assignments from {roi_pq}", flush=True)

        if roi_col not in adata.obs.columns:
            raise KeyError(f"ROI column '{roi_col}' not found in obs. "
                           f"Run the app to assign ROIs first, or provide --roi DIR.")

        mask = adata.obs[roi_col].astype(str) == str(roi_name)
        n_before = len(adata)
        adata = adata[mask].copy()
        print(f"  ROI filter '{roi_col}' == '{roi_name}': "
              f"{mask.sum():,} / {n_before:,} cells kept", flush=True)
        if len(adata) == 0:
            raise ValueError(f"No cells found with {roi_col} == '{roi_name}'. "
                             f"Available values: {adata.obs[roi_col].dropna().unique().tolist()}")

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
        if not sp.issparse(X_full):
            X_full = sp.csr_matrix(X_full)
        # Drop cells where all corrected counts are zero
        nonzero_mask = np.asarray(X_full.sum(axis=1)).ravel() > 0
        n_before = len(adata)
        adata   = adata[nonzero_mask].copy()
        X_full  = X_full[nonzero_mask]
        print(f"  Dropped {n_before - nonzero_mask.sum():,} all-zero corrected cells "
              f"({nonzero_mask.sum():,} kept)", flush=True)
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

    # ── Load annotations from cache ────────────────────────────────────────────
    _cache_dir = os.path.expanduser(cache_dir) if cache_dir else os.path.expanduser("~/.xenium_explorer_cache")
    # If the provided cache_dir is a run dir (contains params.json), look for
    # annotation parquets in its parent (the global cache root).
    run_dir   = os.path.dirname(os.path.abspath(sdata_path))
    if os.path.exists(os.path.join(_cache_dir, "params.json")):
        run_dir    = _cache_dir          # user pointed at the run dir explicitly
        _cache_dir = os.path.dirname(_cache_dir)  # annotation parquets are one level up
    if os.path.isdir(_cache_dir):
        # Derive param_tag from params.json in the run directory
        param_tag = None
        params_json = os.path.join(run_dir, "params.json")
        if os.path.exists(params_json):
            import json as _json
            with open(params_json) as _f:
                param_tag = _json.load(_f).get("param_tag")

        # Mapping: fragment to look for in filename → target column name
        _annot_targets = {
            "labels_seurat":        "cell_type_seurat",
            "labels_rctd_doublet":  "cell_type_rctd_doublet",
        }
        for _fragment, _col in _annot_targets.items():
            if _col in meta.columns:
                continue  # already present in obs
            # Prefer files that contain the param_tag (run-specific), fall back to any match
            candidates = [
                f for f in os.listdir(_cache_dir)
                if f.endswith(".parquet") and _fragment in f
            ]
            if param_tag:
                specific = [f for f in candidates if param_tag in f]
                if specific:
                    candidates = specific
            if not candidates:
                continue
            # Pick the most recently modified
            candidates.sort(key=lambda f: os.path.getmtime(os.path.join(_cache_dir, f)))
            chosen = os.path.join(_cache_dir, candidates[-1])
            try:
                _ann = pd.read_parquet(chosen)
                if "label" in _ann.columns:
                    _ann = _ann["label"]
                else:
                    _ann = _ann.iloc[:, 0]

                # Try matching as strings first
                _ann_str   = _ann.copy(); _ann_str.index = _ann_str.index.astype(str)
                _meta_idx  = meta.index.astype(str)
                _matched   = _ann_str.reindex(_meta_idx)
                _n_matched = _matched.notna().sum()

                # If fewer than 10% matched, try matching as integers
                if _n_matched < 0.1 * len(_meta_idx):
                    try:
                        _ann_int  = _ann.copy(); _ann_int.index = _ann_int.index.astype(int)
                        _meta_int = _meta_idx.astype(int)
                        _matched2 = _ann_int.reindex(_meta_int)
                        _matched2.index = _meta_idx
                        if _matched2.notna().sum() > _n_matched:
                            _matched = _matched2
                            _n_matched = _matched.notna().sum()
                    except (ValueError, TypeError):
                        pass

                if _n_matched == 0:
                    print(f"  Warning: {_col} — 0/{len(_meta_idx)} cells matched. "
                          f"Parquet index sample: {list(_ann.index[:5])}, "
                          f"meta index sample: {list(_meta_idx[:5])}", flush=True)
                elif _n_matched < len(_meta_idx):
                    print(f"  Warning: {_col} — only {_n_matched}/{len(_meta_idx)} cells matched; "
                          f"rest set to 'Unknown'", flush=True)

                meta[_col] = _matched.fillna("Unknown")
                print(f"  Loaded {_col} from {os.path.basename(chosen)} "
                      f"({_ann.nunique()} types, {_n_matched:,}/{len(_meta_idx):,} matched)",
                      flush=True)
            except Exception as _e:
                print(f"  Warning: could not load {_col} from {chosen}: {_e}", flush=True)

    print(f"  Meta columns: {list(meta.columns)}", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:

        # ── Write panel matrix to MatrixMarket ────────────────────────────────
        panel_mtx  = os.path.join(tmpdir, "panel.mtx").replace("\\", "/")
        panel_genes_file = os.path.join(tmpdir, "panel_genes.txt").replace("\\", "/")
        panel_cells_file = os.path.join(tmpdir, "panel_cells.txt").replace("\\", "/")

        print("  Writing panel count matrix …", flush=True)
        sio.mmwrite(panel_mtx, X_panel.T.tocsc())  # genes × cells for Seurat
        with open(panel_genes_file, "w") as f:
            f.write("\n".join(panel_genes) + "\n")
        with open(panel_cells_file, "w") as f:
            f.write("\n".join(str(c) for c in cell_ids) + "\n")

        # ── Write imputed matrix if needed ────────────────────────────────────
        imp_mtx = imp_genes_file = imp_cells_file = None
        if include_imputed and len(imputed_genes) > 0:
            imp_mtx        = os.path.join(tmpdir, "imputed.mtx").replace("\\", "/")
            imp_genes_file = os.path.join(tmpdir, "imp_genes.txt").replace("\\", "/")
            imp_cells_file = os.path.join(tmpdir, "imp_cells.txt").replace("\\", "/")
            print(f"  Writing imputed matrix ({len(imputed_genes)} genes) …", flush=True)
            sio.mmwrite(imp_mtx, X_imputed.T.tocsc())
            with open(imp_genes_file, "w") as f:
                f.write("\n".join(imputed_genes) + "\n")
            with open(imp_cells_file, "w") as f:
                f.write("\n".join(str(c) for c in cell_ids) + "\n")

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

        # ── Write UMAP coordinates ────────────────────────────────────────────
        umap_file = None
        if use_corrected:
            # When using corrected counts, only write UMAP from corrected coords
            if "split_umap_1" in meta.columns and "split_umap_2" in meta.columns:
                u1_col, u2_col = "split_umap_1", "split_umap_2"
            else:
                print("  Warning: --use_corrected set but split_umap_1/2 not found in obs; "
                      "skipping UMAP (run SPLIT correction in the app first)", flush=True)
                u1_col, u2_col = None, None
        else:
            u1_col = "umap_1" if "umap_1" in meta.columns else None
            u2_col = "umap_2" if "umap_2" in meta.columns else None
        if u1_col and u2_col and u1_col in meta.columns and u2_col in meta.columns:
            umap_df = pd.DataFrame({
                "cell":   [str(c) for c in cell_ids],
                "UMAP_1": meta[u1_col].reindex(meta.index).values.astype(float),
                "UMAP_2": meta[u2_col].reindex(meta.index).values.astype(float),
            })
            umap_file = os.path.join(tmpdir, "umap.csv").replace("\\", "/")
            umap_df.to_csv(umap_file, index=False)
            print(f"  UMAP written ({u1_col}/{u2_col}, {len(umap_df):,} cells)", flush=True)

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
.panel_mat   <- as(.panel_mat, "CsparseMatrix")

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
.imp_mat   <- as(.imp_mat, "CsparseMatrix")
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
.fov_coords[["segmentation"]] <- CreateSegmentation(.cell_seg_long)
'''}

{"" if not nuc_seg_file else f'''
.nuc_seg_long <- read.csv("{nuc_seg_file}", stringsAsFactors=FALSE)
.fov_coords[["nucleus_segmentation"]] <- CreateSegmentation(.nuc_seg_long)
'''}

.fov <- CreateFOV(
    coords = .fov_coords,
    type   = names(.fov_coords),
    assay  = "{assay_r}"
)
.seurat[["{fov_r}"]] <- .fov

# ── Add UMAP reduction ────────────────────────────────────────────────────────
{"" if not umap_file else f'''
.umap_df         <- read.csv("{umap_file}", stringsAsFactors=FALSE)
rownames(.umap_df) <- .umap_df$cell
.umap_df$cell    <- NULL
.umap_mat        <- as.matrix(.umap_df)
colnames(.umap_mat) <- c("UMAP_1", "UMAP_2")
.seurat[["umap"]] <- CreateDimReducObject(
    embeddings = .umap_mat,
    key        = "UMAP_",
    assay      = "{assay_r}"
)
rm(.umap_df, .umap_mat)
'''}

# ── Save ───────────────────────────────────────────────────────────────────────
saveRDS(.seurat, file="{output_rds_r}")
cat(sprintf("Saved: %s\\n", "{output_rds_r}"))

# Cleanup large temporaries
rm(.panel_mat, .meta_df, .cents_df, .centroids, .fov_coords, .fov)
if (exists(".imp_mat"))     rm(.imp_mat)
if (exists(".cell_seg_long")) rm(.cell_seg_long)
if (exists(".nuc_seg_long"))  rm(.nuc_seg_long)
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
    p.add_argument("--roi",            default=None, metavar="DIR",
                   help="Directory with roi_<cls>.parquet for ROI assignment "
                        "(if omitted, reads roi_<cls> column directly from zarr obs)")
    p.add_argument("--roi_cls",        default=None, metavar="CLS",
                   help="ROI class to filter on (obs column: roi_<CLS>)")
    p.add_argument("--roi_name",       default=None, metavar="NAME",
                   help="ROI name to keep (cells where roi_<CLS> == NAME)")
    p.add_argument("--cache_dir",      default=None, metavar="DIR",
                   help="xenium_explorer cache dir to scan for annotation parquets "
                        "(default: ~/.xenium_explorer_cache)")
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
        roi_dir=args.roi,
        roi_cls=args.roi_cls,
        roi_name=args.roi_name,
        cache_dir=args.cache_dir,
    )
