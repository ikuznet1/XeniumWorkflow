#!/usr/bin/env python3
"""
seurat2cellnest.py — Convert Seurat RDS to CellNEST-ready AnnData .h5ad

Usage:
  python seurat2cellnest.py <rds_path> [options]

Positional:
  rds_path              Path to Seurat RDS file

Options:
  --output_dir DIR      Where to write outputs (default: same dir as rds_path)
  --data_name NAME      CellNEST dataset name (default: stem of rds_path)
  --combine_fovs        Combine all FOVs into a single h5ad (default: per-FOV)
  --assay NAME          Seurat assay name (default: auto-detect from DefaultAssay)
  --layer NAME          Layer to extract — raw counts for CellNEST (default: counts)
  --cell_type_col COL   obs column for annotation CSV
  --fov_col COL         meta.data column for FOV assignment (auto-detected if omitted)
  --list_columns        Print available meta.data columns and exit

Examples:
  # Per-FOV output (default):
  python functions/seurat2cellnest.py functions/Xenium_resegmented_imputed_final.rds \\
      --output_dir cellnest_input/

  # Combined single file:
  python functions/seurat2cellnest.py functions/Xenium_resegmented_imputed_final.rds \\
      --output_dir cellnest_input/ --combine_fovs

  # With cell type annotation:
  python functions/seurat2cellnest.py functions/Xenium_resegmented_imputed_final.rds \\
      --output_dir cellnest_input/ --cell_type_col cell_type
"""

import argparse
import os
import sys
import tempfile

import anndata
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# rpy2 helpers
# ---------------------------------------------------------------------------

def _init_rpy2():
    """Initialize rpy2 with thread-safe conversion (codebase pattern)."""
    import rpy2.robjects as ro
    import rpy2.robjects.conversion as _rconv
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr

    _rconv.set_conversion(ro.default_converter)
    pandas2ri.activate()

    base = importr("base")
    importr("SeuratObject")
    importr("Matrix")

    return ro, base


# ---------------------------------------------------------------------------
# Core extraction helpers
# ---------------------------------------------------------------------------

def _extract_metadata(ro, base, rds):
    """Extract meta.data from Seurat object as a pandas DataFrame."""
    from rpy2.robjects import pandas2ri

    meta_r = ro.r['slot'](rds, "meta.data")
    meta_df = pandas2ri.rpy2py(meta_r)
    return meta_df


def _safe_obs_col(series):
    """Convert a pandas Series to a type that h5py/anndata can write.

    Handles mixed-type columns, object arrays containing None/NaN, R vectors
    that didn't convert cleanly, embedded arrays, etc.
    Returns either float64 (for numeric) or plain object dtype with Python str
    values (for strings) — both are reliably writable by h5py.
    """
    # Already a clean numeric type — keep as-is
    if pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series):
        return series.astype(np.float64)

    # Try numeric conversion first
    try:
        return pd.to_numeric(series)
    except (ValueError, TypeError):
        pass

    # Convert element-by-element to plain Python str.
    # Return as object-dtype Series so anndata/h5py sees a uniform str array
    # with no pd.NA / pd.StringDtype complications.
    def _to_str(v):
        if v is None or v is pd.NA:
            return ""
        try:
            if isinstance(v, float) and np.isnan(v):
                return ""
        except Exception:
            pass
        if isinstance(v, (list, np.ndarray)):
            # Embedded array — stringify but truncate
            return str(v)[:200]
        try:
            return str(v)
        except Exception:
            return ""

    return pd.Series([_to_str(v) for v in series], index=series.index, dtype=object)


def _detect_fov_col(meta_df):
    """Auto-detect FOV column in meta.data (must have 2–100 unique values).

    Priority order:
    1. Named candidates commonly used for FOV/sample identity
    2. Any remaining column with a reasonable number of unique values
    """
    # Priority candidates
    candidates = ["fov", "FOV", "orig.ident", "sample", "dataset", "batch",
                  "slide", "section", "replicate"]
    for c in candidates:
        if c in meta_df.columns and 1 < meta_df[c].nunique() <= 100:
            return c

    # Fallback: scan all columns for one with 2–100 unique non-null values
    # Skip obviously non-FOV columns (continuous numerics, cell barcodes, etc.)
    for c in meta_df.columns:
        if c.startswith("_"):
            continue
        n = meta_df[c].nunique()
        if 1 < n <= 100:
            # Prefer string/categorical columns over float columns
            if meta_df[c].dtype == object or str(meta_df[c].dtype) == "category":
                return c

    # Last resort: any column with 2–100 unique values
    for c in meta_df.columns:
        if c.startswith("_"):
            continue
        n = meta_df[c].nunique()
        if 1 < n <= 100:
            return c

    return None


def _detect_fov_from_images(ro, rds, meta_df):
    """Assign FOV labels by querying Seurat's @images slot.

    Each image in the Seurat object corresponds to one FOV.
    Cells(Images(obj)[[img]]) returns the cell IDs belonging to that image.
    Returns a Series with index = cell ID, values = image/FOV name.
    Returns None if only one image is found or @images is NULL.
    """
    try:
        ro.r.assign("._s2c_obj", rds)
        # Safely get image names — Images() may return NULL
        img_names_r = ro.r('tryCatch(names(Images(._s2c_obj)), error=function(e) character(0))')
        if img_names_r is ro.r("NULL") or len(img_names_r) == 0:
            print("  No @images slot found")
            return None
        image_names = list(img_names_r)
        if len(image_names) <= 1:
            print(f"  Only {len(image_names)} image(s) in @images — skipping image-based FOV detection")
            return None

        print(f"  Detecting FOVs from {len(image_names)} Seurat images…")
        cell_to_fov = {}
        for img in image_names:
            ro.r.assign("._s2c_img", img)
            try:
                cells_r = ro.r('tryCatch(Cells(Images(._s2c_obj)[[._s2c_img]]), error=function(e) character(0))')
                cells = list(cells_r)
                valid = [c for c in cells if c in meta_df.index]
                for c in valid:
                    cell_to_fov[c] = img
                print(f"    {img}: {len(valid)} cells")
            except Exception as e:
                print(f"    Warning: could not get cells for image '{img}': {e}")

        fov_series = meta_df.index.map(cell_to_fov)
        fov_series = pd.Series(fov_series, index=meta_df.index)
        assigned = fov_series.notna()
        print(f"  Assigned {assigned.sum()}/{len(meta_df)} cells to {len(image_names)} FOVs")
        return fov_series
    except Exception as e:
        print(f"  Image-based FOV detection failed: {e}")
        return None


def _extract_spatial_coords(ro, rds, meta_df):
    """Extract spatial coordinates from Seurat object.

    Tries multiple strategies:
    1. GetTissueCoordinates for each FOV image
    2. meta.data columns (x_centroid/y_centroid)
    3. Embeddings(obj, 'spatial') if available

    Returns (x, y) arrays aligned to meta_df index.
    """
    n_cells = len(meta_df)
    x = np.full(n_cells, np.nan)
    y = np.full(n_cells, np.nan)

    # Strategy 1: GetTissueCoordinates per FOV image
    try:
        ro.r.assign("._s2c_obj", rds)
        image_names = list(ro.r('names(Images(._s2c_obj))'))
        if image_names:
            print(f"  Found {len(image_names)} FOV images: {image_names[:5]}"
                  f"{'...' if len(image_names) > 5 else ''}")
            all_coords = []
            for img in image_names:
                try:
                    ro.r(f'._s2c_coords <- GetTissueCoordinates(._s2c_obj, image="{img}")')
                    coords_r = ro.r("._s2c_coords")
                    from rpy2.robjects import pandas2ri
                    coords_df = pandas2ri.rpy2py(coords_r)
                    all_coords.append(coords_df)
                except Exception as e:
                    print(f"  Warning: GetTissueCoordinates failed for '{img}': {e}")
            if all_coords:
                coords_all = pd.concat(all_coords)
                # Identify x/y columns (Seurat uses various names)
                xcol = ycol = None
                for xc in ["x", "imagerow", "x_centroid", "centroid_x"]:
                    if xc in coords_all.columns:
                        xcol = xc
                        break
                for yc in ["y", "imagecol", "y_centroid", "centroid_y"]:
                    if yc in coords_all.columns:
                        ycol = yc
                        break
                if xcol and ycol:
                    # Align to meta_df index
                    matched = coords_all.reindex(meta_df.index)
                    valid = matched[xcol].notna()
                    if valid.sum() > 0:
                        x[valid.values] = matched.loc[valid, xcol].values.astype(np.float64)
                        y[valid.values] = matched.loc[valid, ycol].values.astype(np.float64)
                        print(f"  Extracted coords for {valid.sum()}/{n_cells} cells "
                              f"via GetTissueCoordinates (cols: {xcol}, {ycol})")
                        if valid.sum() == n_cells:
                            return x, y
    except Exception as e:
        print(f"  GetTissueCoordinates strategy failed: {e}")

    # Strategy 2: meta.data columns
    for xcol, ycol in [("x_centroid", "y_centroid"),
                       ("centroid_x", "centroid_y"),
                       ("x", "y")]:
        if xcol in meta_df.columns and ycol in meta_df.columns:
            x = meta_df[xcol].values.astype(np.float64)
            y = meta_df[ycol].values.astype(np.float64)
            print(f"  Extracted coords from meta.data columns: {xcol}, {ycol}")
            return x, y

    # Strategy 3: Embeddings
    try:
        ro.r.assign("._s2c_obj", rds)
        ro.r('._s2c_emb <- tryCatch(Embeddings(._s2c_obj, "spatial"), error=function(e) NULL)')
        emb = ro.r("._s2c_emb")
        if emb != ro.r("NULL"):
            emb_arr = np.array(emb)
            if emb_arr.shape[1] >= 2:
                x = emb_arr[:, 0].astype(np.float64)
                y = emb_arr[:, 1].astype(np.float64)
                print(f"  Extracted coords from Embeddings(obj, 'spatial')")
                return x, y
    except Exception:
        pass

    # Check if we got partial coords from Strategy 1
    if not np.all(np.isnan(x)):
        print(f"  Warning: only {np.sum(~np.isnan(x))}/{n_cells} cells have coordinates")
        return x, y

    raise ValueError(
        "Could not extract spatial coordinates. Tried: GetTissueCoordinates, "
        "meta.data (x_centroid/y_centroid, centroid_x/centroid_y, x/y), "
        "Embeddings(obj, 'spatial'). Check your Seurat object."
    )


def _extract_counts_mtx(ro, rds, assay, layer, cell_subset=None):
    """Extract expression matrix via R writeMM → Python mmread.

    Parameters
    ----------
    ro : rpy2.robjects module
    rds : R Seurat object
    assay : str — assay name
    layer : str — layer name (e.g. 'counts')
    cell_subset : list of str or None — cell IDs to subset (for per-FOV extraction)

    Returns
    -------
    mat : scipy.sparse.csr_matrix — cells × genes
    gene_names : list of str
    cell_ids : list of str
    """
    ro.r.assign("._s2c_obj", rds)
    ro.r.assign("._s2c_assay", assay)
    ro.r.assign("._s2c_layer", layer)

    with tempfile.TemporaryDirectory() as tmpdir:
        mat_file = os.path.join(tmpdir, "mat.mtx").replace("\\", "/")
        genes_file = os.path.join(tmpdir, "genes.txt").replace("\\", "/")
        cells_file = os.path.join(tmpdir, "cells.txt").replace("\\", "/")

        if cell_subset is not None:
            ro.r.assign("._s2c_cells", ro.StrVector(cell_subset))
            ro.r(f"""
._s2c_mat <- LayerData(._s2c_obj[[._s2c_assay]], layer=._s2c_layer)
._s2c_mat <- ._s2c_mat[, ._s2c_cells, drop=FALSE]
._s2c_mat <- as(._s2c_mat, 'dgCMatrix')
Matrix::writeMM(._s2c_mat, "{mat_file}")
writeLines(rownames(._s2c_mat), "{genes_file}")
writeLines(colnames(._s2c_mat), "{cells_file}")
rm(._s2c_mat); gc()
""")
        else:
            ro.r(f"""
._s2c_mat <- LayerData(._s2c_obj[[._s2c_assay]], layer=._s2c_layer)
._s2c_mat <- as(._s2c_mat, 'dgCMatrix')
Matrix::writeMM(._s2c_mat, "{mat_file}")
writeLines(rownames(._s2c_mat), "{genes_file}")
writeLines(colnames(._s2c_mat), "{cells_file}")
rm(._s2c_mat); gc()
""")

        mat = sio.mmread(mat_file).T.tocsr()  # genes×cells → cells×genes
        with open(genes_file) as f:
            gene_names = [line.strip() for line in f]
        with open(cells_file) as f:
            cell_ids = [line.strip() for line in f]

    return mat, gene_names, cell_ids


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------

def seurat2cellnest(
    rds_path: str,
    output_dir: str | None = None,
    data_name: str | None = None,
    combine_fovs: bool = False,
    assay: str | None = None,
    layer: str = "counts",
    cell_type_col: str | None = None,
    fov_col: str | None = None,
) -> list[str]:
    """Convert a Seurat RDS to CellNEST-ready AnnData h5ad file(s).

    Parameters
    ----------
    rds_path : str
        Path to Seurat RDS file.
    output_dir : str or None
        Output directory. Defaults to same directory as rds_path.
    data_name : str or None
        Dataset name prefix. Defaults to stem of rds_path.
    combine_fovs : bool
        If True, write a single h5ad for all FOVs. If False (default),
        write one h5ad per FOV.
    assay : str or None
        Seurat assay name. Auto-detected if None (tries DefaultAssay, then
        first available assay).
    layer : str
        Layer to extract — should be raw counts for CellNEST (default: "counts").
    cell_type_col : str or None
        meta.data column to export as annotation CSV.
    fov_col : str or None
        meta.data column for FOV assignment. Auto-detected if None.

    Returns
    -------
    list of str — paths to output h5ad files.
    """
    rds_path = os.path.abspath(rds_path)
    if not os.path.isfile(rds_path):
        raise FileNotFoundError(f"RDS file not found: {rds_path}")

    if data_name is None:
        data_name = os.path.splitext(os.path.basename(rds_path))[0]

    if output_dir is None:
        output_dir = os.path.dirname(rds_path)
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Load RDS ──────────────────────────────────────────────────
    print("\n[1/6] Loading RDS via rpy2…")
    ro, base = _init_rpy2()
    rds = base.readRDS(rds_path)

    # Auto-detect assay if not specified
    if assay is None:
        ro.r.assign("._s2c_obj", rds)
        available_assays = list(ro.r('names(._s2c_obj@assays)'))
        default_assay = str(ro.r('DefaultAssay(._s2c_obj)')[0])
        print(f"  Available assays: {available_assays}")
        print(f"  Default assay: {default_assay}")
        assay = default_assay

    print(f"\n[seurat2cellnest] rds_path    : {rds_path}")
    print(f"[seurat2cellnest] output_dir  : {output_dir}")
    print(f"[seurat2cellnest] data_name   : {data_name}")
    print(f"[seurat2cellnest] combine_fovs: {combine_fovs}")
    print(f"[seurat2cellnest] assay/layer : {assay}/{layer}")

    # ── Step 2: Extract metadata ──────────────────────────────────────────
    print("[2/6] Extracting meta.data…")
    meta_df = _extract_metadata(ro, base, rds)
    print(f"  {len(meta_df)} cells, {len(meta_df.columns)} metadata columns")
    print(f"  Columns: {list(meta_df.columns)}")

    # ── Step 3: Extract spatial coordinates ───────────────────────────────
    print("[3/6] Extracting spatial coordinates…")
    x, y = _extract_spatial_coords(ro, rds, meta_df)
    print(f"  x range: [{np.nanmin(x):.1f}, {np.nanmax(x):.1f}]")
    print(f"  y range: [{np.nanmin(y):.1f}, {np.nanmax(y):.1f}]")

    # Attach coords to meta_df for easy subsetting
    meta_df["_x"] = x
    meta_df["_y"] = y

    # ── Step 4: Detect FOV assignment ─────────────────────────────────────
    # fov_series: index=cell_id, value=fov_name  (used for per-FOV splitting)
    fov_series = None
    if not combine_fovs:
        if fov_col is not None:
            # Explicit column requested by user
            if fov_col not in meta_df.columns:
                raise KeyError(
                    f"FOV column '{fov_col}' not found. "
                    f"Available: {list(meta_df.columns)}"
                )
            fov_series = meta_df[fov_col].astype(str)
            print(f"[4/6] FOV column (user-specified): '{fov_col}' — "
                  f"{fov_series.nunique()} FOVs")
        else:
            # Strategy 1: @images slot (primary — works for merged Xenium objects)
            fov_series = _detect_fov_from_images(ro, rds, meta_df)
            if fov_series is not None and fov_series.nunique() > 1:
                print(f"[4/6] FOV assignment: from @images slot "
                      f"({fov_series.nunique()} FOVs)")
            else:
                # Strategy 2: meta.data column with >1 unique value
                fov_col = _detect_fov_col(meta_df)
                if fov_col is not None:
                    fov_series = meta_df[fov_col].astype(str)
                    unique_vals = sorted(fov_series.unique(), key=str)
                    print(f"[4/6] FOV column (auto-detected): '{fov_col}' — "
                          f"{fov_series.nunique()} FOVs: {unique_vals}")
                else:
                    raise ValueError(
                        "Could not detect multiple FOVs. Neither @images slot nor "
                        "any meta.data column has >1 unique value. "
                        "Use --fov_col to specify a column, or --combine_fovs."
                    )

        # Attach FOV labels to meta_df
        meta_df["_fov"] = fov_series
        fov_values = sorted(meta_df["_fov"].dropna().unique(), key=str)
        print(f"  FOVs: {fov_values}")
    else:
        fov_values = None
        print("[4/6] Combining all FOVs into single file")

    # ── Step 5: Extract counts and build h5ad(s) ─────────────────────────
    print("[5/6] Extracting expression matrix and writing h5ad…")
    output_paths = []

    if combine_fovs:
        # Single file: extract full matrix
        mat, gene_names, cell_ids = _extract_counts_mtx(ro, rds, assay, layer)
        print(f"  Full matrix: {mat.shape[0]} cells × {mat.shape[1]} genes")

        # Align coords to cell_ids order
        meta_aligned = meta_df.reindex(cell_ids)
        coords = np.column_stack([
            meta_aligned["_x"].values,
            meta_aligned["_y"].values,
        ])

        adata = anndata.AnnData(X=mat.astype(np.float32))
        adata.obs_names = cell_ids
        adata.var_names = gene_names
        adata.obsm["spatial"] = coords.astype(np.float64)

        out_path = os.path.join(output_dir, f"{data_name}.h5ad")
        adata.write_h5ad(out_path)
        print(f"  Written: {out_path} ({mat.shape[0]} cells × {mat.shape[1]} genes)")
        output_paths.append(out_path)

        # Annotation CSV
        if cell_type_col:
            _write_annotation_csv(meta_aligned, cell_type_col, cell_ids,
                                  output_dir, data_name)
    else:
        # Per-FOV: extract subsets in R to avoid full matrix materialization
        total_cells = 0
        for fov_name in fov_values:
            fov_mask = meta_df["_fov"].astype(str) == str(fov_name)
            fov_cells = list(meta_df.index[fov_mask])
            if not fov_cells:
                continue

            # Clean FOV name for filename
            fov_clean = str(fov_name).replace("/", "_").replace(" ", "_")

            print(f"  FOV '{fov_name}': {len(fov_cells)} cells…", end=" ")
            mat, gene_names, cell_ids = _extract_counts_mtx(
                ro, rds, assay, layer, cell_subset=fov_cells
            )

            # Get coords for this FOV
            meta_fov = meta_df.loc[fov_cells]
            coords = np.column_stack([
                meta_fov["_x"].values,
                meta_fov["_y"].values,
            ])

            adata = anndata.AnnData(X=mat.astype(np.float32))
            adata.obs_names = cell_ids
            adata.var_names = gene_names
            adata.obsm["spatial"] = coords.astype(np.float64)

            meta_fov_aligned = meta_fov.reindex(cell_ids)

            out_path = os.path.join(output_dir, f"{data_name}_{fov_clean}.h5ad")
            adata.write_h5ad(out_path)
            print(f"→ {out_path}")
            output_paths.append(out_path)
            total_cells += len(cell_ids)

            # Annotation CSV per FOV
            if cell_type_col:
                _write_annotation_csv(meta_fov_aligned, cell_type_col, cell_ids,
                                      output_dir, f"{data_name}_{fov_clean}")

        print(f"  Total: {total_cells} cells across {len(output_paths)} files")

    # ── Step 6: Summary ───────────────────────────────────────────────────
    print(f"\n[6/6] Done. Output files:")
    for p in output_paths:
        print(f"  {p}")

    # Print suggested CellNEST commands
    print("\nSuggested CellNEST commands:")
    print("-" * 60)
    for p in output_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        ann_arg = ""
        ann_csv = os.path.join(output_dir, f"{stem}_annotation.csv")
        if cell_type_col and os.path.isfile(ann_csv):
            ann_arg = f" \\\n  --annotation='{ann_csv}'"
        print(
            f"cellnest preprocess \\\n"
            f"  --data_name='{stem}' \\\n"
            f"  --data_from='{p}' \\\n"
            f"  --data_type=anndata \\\n"
            f"  --distance_measure=knn \\\n"
            f"  --split=1"
            f"{ann_arg}\n"
        )
    print("-" * 60)

    # Clean up R objects
    try:
        ro.r("rm(._s2c_obj); gc()")
    except Exception:
        pass

    return output_paths


def _write_annotation_csv(meta_df, cell_type_col, cell_ids, output_dir, name_prefix):
    """Write annotation CSV for CellNEST."""
    if cell_type_col not in meta_df.columns:
        print(f"  Warning: column '{cell_type_col}' not found — skipping annotation CSV")
        return
    labels = meta_df[cell_type_col].values
    if np.issubdtype(np.array(labels).dtype, np.integer) or \
       np.issubdtype(np.array(labels).dtype, np.floating):
        labels = np.array(labels).astype(int).astype(str)
    else:
        labels = np.array(labels).astype(str)
    csv_path = os.path.join(output_dir, f"{name_prefix}_annotation.csv")
    pd.DataFrame({"cell_type": labels}, index=cell_ids).to_csv(csv_path)
    print(f"  Annotation CSV: {csv_path}")


# ---------------------------------------------------------------------------
# list_columns helper
# ---------------------------------------------------------------------------

def list_columns(rds_path):
    """Load RDS and print available meta.data columns."""
    rds_path = os.path.abspath(rds_path)
    print(f"Loading {rds_path}…")

    ro, base = _init_rpy2()
    rds = base.readRDS(rds_path)
    meta_df = _extract_metadata(ro, base, rds)

    print(f"\n{len(meta_df)} cells, {len(meta_df.columns)} meta.data columns:\n")
    for col in meta_df.columns:
        dtype = meta_df[col].dtype
        nuniq = meta_df[col].nunique()
        sample = meta_df[col].dropna().iloc[:3].tolist() if len(meta_df[col].dropna()) > 0 else []
        print(f"  {col:<40s} {str(dtype):<15s} {nuniq:>6d} unique  "
              f"sample: {sample}")

    # Also show available images/FOVs
    try:
        ro.r.assign("._s2c_obj", rds)
        image_names = list(ro.r('names(Images(._s2c_obj))'))
        if image_names:
            print(f"\nFOV images ({len(image_names)}): {image_names}")
    except Exception:
        pass

    try:
        ro.r("rm(._s2c_obj); gc()")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert Seurat RDS to CellNEST-ready AnnData .h5ad",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("rds_path", help="Path to Seurat RDS file")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: same dir as rds_path)")
    parser.add_argument("--data_name", default=None,
                        help="CellNEST dataset name (default: stem of rds_path)")
    parser.add_argument("--combine_fovs", action="store_true",
                        help="Combine all FOVs into a single h5ad")
    parser.add_argument("--assay", default=None,
                        help="Seurat assay name (default: auto-detect)")
    parser.add_argument("--layer", default="counts",
                        help="Layer to extract (default: counts)")
    parser.add_argument("--cell_type_col", default=None,
                        help="meta.data column for annotation CSV")
    parser.add_argument("--fov_col", default=None,
                        help="meta.data column for FOV assignment (auto-detected)")
    parser.add_argument("--list_columns", action="store_true",
                        help="Print available meta.data columns and exit")
    args = parser.parse_args()

    if args.list_columns:
        list_columns(args.rds_path)
        sys.exit(0)

    seurat2cellnest(
        rds_path=args.rds_path,
        output_dir=args.output_dir,
        data_name=args.data_name,
        combine_fovs=args.combine_fovs,
        assay=args.assay,
        layer=args.layer,
        cell_type_col=args.cell_type_col,
        fov_col=args.fov_col,
    )


if __name__ == "__main__":
    main()
