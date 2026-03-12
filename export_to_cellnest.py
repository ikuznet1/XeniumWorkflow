#!/usr/bin/env python3
"""
export_to_cellnest.py — Export SpatialData zarr to CellNEST-ready AnnData .h5ad

Usage:
  python export_to_cellnest.py <sdata_path> [options]

Positional:
  sdata_path            Path to SpatialData zarr directory

Options:
  --output_dir DIR      Where to write outputs (default: sdata_path/../cellnest_input/)
  --data_name NAME      CellNEST dataset name (default: stem of sdata_path)
  --use_corrected       Use X_corrected layer (SPLIT) instead of X
  --include_imputed     Include SpaGE-imputed genes in the AnnData (in a separate 'imputed' layer)
  --cell_type_col COL   obs column for annotation CSV
  --list_obs_cols       Print available obs columns and exit
"""

import argparse
import os
import sys

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr


# ---------------------------------------------------------------------------
# Low-level zarr helpers
# ---------------------------------------------------------------------------

def read_zarr_sparse(zroot, path):
    """Read a CSR sparse matrix stored as a zarr group (anndata encoding)."""
    grp = zroot[path]
    # Check if it's actually a sparse group or a dense array
    if isinstance(grp, zarr.Array):
        # Dense array — wrap directly
        arr = grp[:]
        return sp.csr_matrix(arr)

    # Sparse group: must have data/indices/indptr
    try:
        data    = grp["data"][:]
        indices = grp["indices"][:]
        indptr  = grp["indptr"][:]
        shape   = tuple(grp.attrs.get("shape", grp.attrs.get("h5sparse_shape", None)))
        if shape is None:
            raise KeyError("No shape attribute found in sparse group")
        return sp.csr_matrix((data, indices, indptr), shape=shape)
    except KeyError:
        # Last resort: try reading as dense
        arr = np.array(grp)
        return sp.csr_matrix(arr)


def read_obs_column(obs_grp, col):
    """Read a single obs column from zarr, returning a numpy array."""
    item = obs_grp[col]
    if isinstance(item, zarr.Array):
        return item[:]
    # Could be a categorical group (anndata encoding)
    if isinstance(item, zarr.Group):
        codes = item["codes"][:]
        categories = item["categories"][:]
        return categories[codes]
    raise ValueError(f"Unexpected zarr type for obs column '{col}': {type(item)}")


def read_uns_list(table_grp, key):
    """Read a list stored in uns zarr group. Returns [] if absent."""
    try:
        uns_grp = table_grp["uns"]
    except KeyError:
        return []
    if key not in uns_grp:
        return []
    item = uns_grp[key]
    if isinstance(item, zarr.Array):
        return [str(v) for v in item[:]]
    # scalar or other — try iterating
    try:
        return [str(v) for v in item]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_sdata_to_cellnest(
    sdata_path,
    output_dir=None,
    data_name=None,
    use_corrected=False,
    include_imputed=False,
    cell_type_col=None,
):
    """Export a SpatialData zarr store to CellNEST-ready files.

    Parameters
    ----------
    sdata_path : str
        Path to SpatialData zarr directory.
    output_dir : str or None
        Output directory. Defaults to ``<sdata_path>/../cellnest_input/``.
    data_name : str or None
        CellNEST dataset name. Defaults to the stem of sdata_path.
    use_corrected : bool
        Use ``X_corrected`` layer (SPLIT output) instead of ``X``.
    include_imputed : bool
        Include SpaGE-imputed genes in the AnnData as a separate 'imputed' layer.
        When False (default), imputed genes are stripped from X so only panel genes remain.
    cell_type_col : str or None
        obs column to write as an annotation CSV.

    Returns
    -------
    out_h5ad : str
        Path of the written .h5ad file.
    """
    sdata_path = os.path.abspath(sdata_path)

    if data_name is None:
        stem = os.path.basename(sdata_path.rstrip("/\\"))
        data_name = stem if stem else "xenium"

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(sdata_path), "cellnest_input")

    print(f"[export_to_cellnest] sdata_path     : {sdata_path}")
    print(f"[export_to_cellnest] output_dir     : {output_dir}")
    print(f"[export_to_cellnest] data_name      : {data_name}")
    print(f"[export_to_cellnest] use_corrected  : {use_corrected}")
    print(f"[export_to_cellnest] include_imputed: {include_imputed}")

    # ------------------------------------------------------------------
    # Step 1 — Open zarr
    # ------------------------------------------------------------------
    print("[1/8] Opening zarr store …")
    zroot     = zarr.open_group(sdata_path, mode="r")
    table_grp = zroot["tables"]["table"]
    obs_grp   = table_grp["obs"]
    var_grp   = table_grp["var"]

    # ------------------------------------------------------------------
    # Step 2 — Read index arrays
    # ------------------------------------------------------------------
    print("[2/8] Reading cell and gene indices …")
    obs_idx_key = obs_grp.attrs.get("_index", "_index")
    cell_ids    = [str(v) for v in obs_grp[obs_idx_key][:]]

    var_idx_key = var_grp.attrs.get("_index", "_index")
    gene_names  = [str(v) for v in var_grp[var_idx_key][:]]

    # Read imputed gene sets from uns (may be empty)
    raw_imputed_set  = set(read_uns_list(table_grp, "imputed_genes"))
    corr_imputed_set = set(read_uns_list(table_grp, "split_corrected_imputed_genes"))
    all_imputed_set  = raw_imputed_set | corr_imputed_set

    panel_mask  = np.array([g not in all_imputed_set  for g in gene_names], dtype=bool)
    panel_genes = [g for g, m in zip(gene_names, panel_mask) if m]

    # When use_corrected, the relevant imputed genes are those from corrected counts.
    # When not, use raw-imputed genes.
    if use_corrected:
        active_imp_set = corr_imputed_set if corr_imputed_set else raw_imputed_set
    else:
        active_imp_set = raw_imputed_set

    active_imp_mask  = np.array([g in active_imp_set for g in gene_names], dtype=bool)
    active_imp_genes = [g for g, m in zip(gene_names, active_imp_mask) if m]

    print(f"       {len(cell_ids)} cells, {len(gene_names)} genes "
          f"({len(panel_genes)} panel, {len(raw_imputed_set)} raw-imputed, "
          f"{len(corr_imputed_set)} corrected-imputed)")

    # ------------------------------------------------------------------
    # Step 3 — Read raw X (always needed; contains panel + imputed columns)
    # ------------------------------------------------------------------
    print("[3/8] Reading raw X …")
    raw_full_csr = read_zarr_sparse(table_grp, "X")
    print(f"       shape {raw_full_csr.shape}")

    panel_raw_csr   = raw_full_csr[:, panel_mask]
    active_imp_csr  = raw_full_csr[:, active_imp_mask] if active_imp_genes else None

    # ------------------------------------------------------------------
    # Step 3b — Read X_corrected if requested (panel genes only, no imputed cols)
    # ------------------------------------------------------------------
    panel_corrected_csr = None
    if use_corrected:
        print("       Reading X_corrected (SPLIT) …")
        try:
            panel_corrected_csr = read_zarr_sparse(table_grp, "layers/X_corrected")
            print(f"       X_corrected shape {panel_corrected_csr.shape}")
        except KeyError:
            print("       WARNING: X_corrected not found — falling back to raw counts")

    # ------------------------------------------------------------------
    # Step 3c — Assemble final X and X_raw based on flag combination
    #
    #  use_corrected=F, include_imputed=F → X = panel_raw
    #  use_corrected=F, include_imputed=T → X = panel_raw + raw_imputed
    #                                       X_raw = panel_raw (zeros for imp cols)
    #  use_corrected=T, include_imputed=F → X = panel_corrected
    #                                       X_raw = panel_raw
    #  use_corrected=T, include_imputed=T → X = panel_corrected + corr_imputed
    #                                       X_raw = panel_raw (zeros for imp cols)
    # ------------------------------------------------------------------
    n_cells = len(cell_ids)
    zeros_for_imp = (
        sp.csr_matrix((n_cells, len(active_imp_genes)), dtype=np.float32)
        if active_imp_genes else None
    )

    base_csr = panel_corrected_csr if (use_corrected and panel_corrected_csr is not None) \
               else panel_raw_csr

    if include_imputed and active_imp_genes:
        expr_csr   = sp.hstack([base_csr, active_imp_csr], format="csr").astype(np.float32)
        gene_names = panel_genes + active_imp_genes
        raw_layer  = sp.hstack([panel_raw_csr, zeros_for_imp], format="csr").astype(np.float32)
        print(f"       X shape (panel + imputed): {expr_csr.shape}")
        print(f"       X_raw shape (panel only):  {raw_layer.shape}")
    elif use_corrected and panel_corrected_csr is not None:
        expr_csr   = base_csr.astype(np.float32)
        gene_names = panel_genes
        raw_layer  = panel_raw_csr.astype(np.float32)
        print(f"       X shape (corrected panel): {expr_csr.shape}")
        print(f"       X_raw shape (raw panel):   {raw_layer.shape}")
    else:
        if include_imputed and not active_imp_genes:
            print("       --include_imputed set but no imputed genes found in uns")
        expr_csr   = panel_raw_csr.astype(np.float32)
        gene_names = panel_genes
        raw_layer  = None
        print(f"       X shape: {expr_csr.shape}")

    # ------------------------------------------------------------------
    # Step 4 — Read spatial coordinates
    # ------------------------------------------------------------------
    print("[4/8] Reading spatial coordinates …")
    x = y = None

    # Try Xenium / Baysor naming first
    for xcol, ycol in [("x_centroid", "y_centroid"), ("centroid_x", "centroid_y")]:
        if xcol in obs_grp and ycol in obs_grp:
            x = read_obs_column(obs_grp, xcol).astype(np.float64)
            y = read_obs_column(obs_grp, ycol).astype(np.float64)
            print(f"       Using columns '{xcol}' / '{ycol}'")
            break

    if x is None:
        raise KeyError(
            "Could not find centroid columns in obs. "
            "Expected 'x_centroid'/'y_centroid' or 'centroid_x'/'centroid_y'."
        )

    print(f"       x range: [{x.min():.1f}, {x.max():.1f}] µm")
    print(f"       y range: [{y.min():.1f}, {y.max():.1f}] µm")

    # ------------------------------------------------------------------
    # Step 5 — Build AnnData
    # ------------------------------------------------------------------
    print("[5/8] Building AnnData …")
    adata = anndata.AnnData(X=expr_csr.astype(np.float32))
    adata.obs_names  = cell_ids
    adata.var_names  = gene_names
    adata.obsm["spatial"] = np.column_stack([x, y])  # (N, 2) in µm

    if raw_layer is not None:
        adata.layers["X_raw"] = raw_layer
        print(f"       layers['X_raw'] written ({len(panel_genes)} panel genes"
              + (f", {len(active_imp_genes)} imputed cols = 0" if active_imp_genes and include_imputed else "")
              + ")")

    # ------------------------------------------------------------------
    # Step 6 — Write .h5ad
    # ------------------------------------------------------------------
    print("[6/8] Writing .h5ad …")
    os.makedirs(output_dir, exist_ok=True)
    out_h5ad = os.path.join(output_dir, f"{data_name}.h5ad")
    adata.write_h5ad(out_h5ad)
    print(f"       Written: {out_h5ad}")

    # ------------------------------------------------------------------
    # Step 7 — Write annotation CSV (optional)
    # ------------------------------------------------------------------
    annotation_csv = None
    if cell_type_col:
        print(f"[7/8] Writing annotation CSV (column: '{cell_type_col}') …")
        if cell_type_col not in obs_grp:
            print(f"       WARNING: column '{cell_type_col}' not found in obs — skipping annotation CSV")
        else:
            labels = read_obs_column(obs_grp, cell_type_col)
            # Convert numeric cluster IDs to strings
            if np.issubdtype(labels.dtype, np.integer) or np.issubdtype(labels.dtype, np.floating):
                labels = labels.astype(int).astype(str)
            else:
                labels = labels.astype(str)
            annotation_csv = os.path.join(output_dir, f"{data_name}_annotation.csv")
            pd.DataFrame({"cell_type": labels}, index=cell_ids).to_csv(annotation_csv)
            print(f"       Written: {annotation_csv}")
    else:
        print("[7/8] No --cell_type_col specified — skipping annotation CSV")

    # ------------------------------------------------------------------
    # Step 8 — Print suggested CellNEST commands
    # ------------------------------------------------------------------
    print("\n[8/8] Suggested CellNEST commands:")
    print("-" * 60)
    ann_arg = (
        f" \\\n  --annotation='{annotation_csv}'"
        if annotation_csv else ""
    )
    print(
        f"# CellNEST preprocessing:\n"
        f"cellnest preprocess \\\n"
        f"  --data_name='{data_name}' \\\n"
        f"  --data_from='{out_h5ad}' \\\n"
        f"  --data_type=anndata \\\n"
        f"  --distance_measure=knn \\\n"
        f"  --split=1"
        f"{ann_arg}\n"
    )
    print(
        f"# CellNEST training:\n"
        f"cellnest run \\\n"
        f"  --data_name='{data_name}' \\\n"
        f"  --total_subgraphs=8 \\\n"
        f"  --num_epoch=50000 \\\n"
        f"  --model_name='CellNEST_{data_name}' \\\n"
        f"  --run_id=1\n"
    )
    print("-" * 60)

    return out_h5ad


# ---------------------------------------------------------------------------
# list_obs_cols helper
# ---------------------------------------------------------------------------

def list_obs_cols(sdata_path):
    """Print all obs columns available in the zarr store and exit."""
    zroot    = zarr.open_group(sdata_path, mode="r")
    obs_grp  = zroot["tables"]["table"]["obs"]
    cols     = sorted(obs_grp.keys())
    print(f"obs columns in {sdata_path}:")
    for c in cols:
        item = obs_grp[c]
        if isinstance(item, zarr.Array):
            dtype_str = str(item.dtype)
            shape_str = str(item.shape)
        elif isinstance(item, zarr.Group):
            dtype_str = "categorical"
            try:
                n_cats = len(item["categories"])
                shape_str = f"({len(item['codes'])},) [{n_cats} categories]"
            except Exception:
                shape_str = "group"
        else:
            dtype_str = type(item).__name__
            shape_str = ""
        print(f"  {c:<40s} {dtype_str:<20s} {shape_str}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export SpatialData zarr to CellNEST-ready AnnData .h5ad",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("sdata_path", help="Path to SpatialData zarr directory")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--data_name", default=None, help="CellNEST dataset name")
    parser.add_argument(
        "--use_corrected",
        action="store_true",
        help="Use X_corrected layer (SPLIT) instead of X",
    )
    parser.add_argument(
        "--include_imputed",
        action="store_true",
        help=(
            "Include SpaGE-imputed genes in the output. "
            "X will contain all genes (panel + imputed); layers['X_raw'] will contain "
            "panel counts only (zeros for imputed columns). "
            "By default imputed genes are stripped from the output."
        ),
    )
    parser.add_argument(
        "--cell_type_col",
        default=None,
        help="obs column to export as annotation CSV (e.g. cell_type_rctd, cluster_leiden_10)",
    )
    parser.add_argument(
        "--list_obs_cols",
        action="store_true",
        help="Print available obs columns and exit",
    )
    args = parser.parse_args()

    if args.list_obs_cols:
        list_obs_cols(args.sdata_path)
        sys.exit(0)

    export_sdata_to_cellnest(
        sdata_path=args.sdata_path,
        output_dir=args.output_dir,
        data_name=args.data_name,
        use_corrected=args.use_corrected,
        include_imputed=args.include_imputed,
        cell_type_col=args.cell_type_col,
    )


if __name__ == "__main__":
    main()
