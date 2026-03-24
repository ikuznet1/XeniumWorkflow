#!/usr/bin/env python3
"""Transfer cell identity labels between segmentations based on spatial polygon overlap.

For each cell in the query segmentation, finds the reference cell whose polygon
has the largest intersection area and assigns that cell's identity label.

Usage:
    python functions/transfer_seg_idents.py ref.zarr query.zarr \
        --identity_column cell_type_rctd_doublet \
        --query_boundary nucleus \
        --output_csv results.csv

    # List available obs columns in reference:
    python functions/transfer_seg_idents.py ref.zarr query.zarr --list_columns
"""

import argparse
import os
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from shapely import make_valid
from shapely.strtree import STRtree


def _extract_boundaries(sdata, boundary_type="cell"):
    """Extract boundary GeoDataFrame from a SpatialData object.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
    boundary_type : str
        "cell" or "nucleus"

    Returns
    -------
    gpd.GeoDataFrame
    """
    key_candidates = (
        [f"{boundary_type}_boundaries", f"{boundary_type}_circles"]
        if boundary_type == "cell"
        else [f"{boundary_type}_boundaries"]
    )
    for key in key_candidates:
        if key in sdata.shapes:
            return sdata.shapes[key]

    available = list(sdata.shapes.keys())
    raise KeyError(
        f"No '{boundary_type}_boundaries' found in shapes. "
        f"Available shape keys: {available}"
    )


def _find_annotation_parquet(reference_zarr: str, identity_column: str) -> tuple[str | None, str]:
    """Try to auto-locate an annotation parquet from the xenium explorer cache.

    The app stores annotation parquets in CACHE_BASE_DIR with filenames like:
      rctd_{rds_basename}_{label_col}_doublet_umi{N}_sig{N}_labels_rctd_doublet_{tool}_{run_tag}_{dataset_hash}.parquet
      seurat_{rds_basename}_{label_col}_labels_seurat_{tool}_{run_tag}_{dataset_hash}.parquet

    where:
      run_tag      = md5(out_dir)[:8]  (out_dir = the run cache dir, parent of the zarr)
      dataset_hash = md5(data_dir)[:8]

    The parquet column is always "label" regardless of identity_column.

    identity_column → filename search key mapping:
      "cell_type_rctd_doublet" → "rctd_doublet"
      "cell_type_rctd_multi"   → "rctd_multi"
      "cell_type_seurat"       → "labels_seurat"
      "cell_type_celltypist"   → strip "cell_type_" and search directly

    Returns (parquet_path, column_in_parquet).
    """
    import glob as _glob
    import hashlib as _hashlib

    zarr_path = os.path.abspath(reference_zarr.rstrip("/"))
    run_dir = os.path.dirname(zarr_path)    # e.g. ~/.xenium_explorer_cache/proseg_..._2dec91fb
    cache_dir = os.path.dirname(run_dir)    # e.g. ~/.xenium_explorer_cache

    # The app uses md5(out_dir)[:8] as the run tag inside filenames
    run_tag = _hashlib.md5(run_dir.encode()).hexdigest()[:8]

    # Derive search key from identity_column
    col = identity_column
    if col.startswith("cell_type_"):
        key = col[len("cell_type_"):]   # e.g. "rctd_doublet", "seurat", "celltypist"
    else:
        key = col

    # The column inside annotation parquets is always "label"
    label_col = "label"

    pattern = os.path.join(cache_dir, f"*{key}*{run_tag}*.parquet")
    matches = [m for m in _glob.glob(pattern)
               if not m.endswith("_rctd_obj.rds")]  # exclude RDS side-cars

    if len(matches) == 1:
        return matches[0], label_col
    if len(matches) > 1:
        print(f"  Multiple annotation parquets found for run_tag={run_tag}, key={key}:")
        for m in matches:
            print(f"    {m}")
        print(f"  Using: {matches[0]}")
        return matches[0], label_col
    return None, label_col


def transfer_seg_idents(
    reference_zarr: str,
    query_zarr: str,
    identity_column: str,
    query_boundary: str = "cell",
    output_csv: str | None = None,
    unmatched_label: str = "Unassigned",
    labels_file: str | None = None,
) -> str:
    """Transfer identity labels from reference to query segmentation via spatial overlap.

    Parameters
    ----------
    reference_zarr : str
        Path to reference SpatialData zarr store.
    query_zarr : str
        Path to query SpatialData zarr store.
    identity_column : str
        Column name in reference obs (adata.obs) containing identity labels,
        OR the column name in labels_file if --labels_file is provided.
    query_boundary : str
        Which query boundaries to use: "cell" or "nucleus".
    output_csv : str or None
        Output CSV path. Defaults to <query_stem>_transferred_idents.csv.
    unmatched_label : str
        Label assigned to query cells with no spatial overlap.
    labels_file : str or None
        Path to an external CSV or parquet file containing cell labels.
        The file must have a column matching identity_column and its index
        (or first column) should be cell IDs matching the reference boundaries.
        Use this when annotations are stored outside the zarr (e.g. xenium
        explorer annotation cache parquets).

    Returns
    -------
    str
        Path to the output CSV file.
    """
    t0 = time.time()

    # --- Load reference ---
    print(f"Loading reference: {reference_zarr}")
    ref_sdata = sd.read_zarr(reference_zarr)
    ref_boundaries = _extract_boundaries(ref_sdata, "cell")

    # Get identity labels — from external file or zarr table
    ref_cell_ids = ref_boundaries.index.tolist()

    # Auto-detect labels_file from xenium explorer cache if not provided
    _auto_label_col = None
    if labels_file is None:
        auto_path, auto_col = _find_annotation_parquet(reference_zarr, identity_column)
        if auto_path:
            print(f"  Auto-detected annotation parquet: {auto_path}")
            labels_file = auto_path
            _auto_label_col = auto_col  # "label"

    if labels_file is not None:
        # Load from external CSV or parquet (e.g. xenium explorer annotation cache)
        print(f"  Loading labels from: {labels_file}")
        if labels_file.endswith(".parquet"):
            labels_df = pd.read_parquet(labels_file)
        else:
            labels_df = pd.read_csv(labels_file, index_col=0)

        # App annotation parquets store labels in "label" column; explicit files use identity_column
        col_to_use = _auto_label_col if (_auto_label_col and _auto_label_col in labels_df.columns) else identity_column
        if col_to_use not in labels_df.columns:
            available_cols = sorted(labels_df.columns.tolist())
            raise KeyError(
                f"Column '{col_to_use}' not found in labels file. "
                f"Available columns: {available_cols}"
            )
        ref_labels_series = labels_df[col_to_use]
        ref_obs_index = labels_df.index.tolist()
    else:
        ref_table = ref_sdata.tables.get("table")
        if ref_table is None:
            available_tables = list(ref_sdata.tables.keys())
            raise KeyError(
                f"No 'table' found in reference SpatialData. "
                f"Available tables: {available_tables}"
            )

        if identity_column not in ref_table.obs.columns:
            available_cols = sorted(ref_table.obs.columns.tolist())
            raise KeyError(
                f"Column '{identity_column}' not found in reference obs. "
                f"Available columns: {available_cols}\n"
                f"Tip: run RCTD/Seurat annotation in the app first, or provide "
                f"--labels_file pointing to the annotation parquet."
            )
        ref_labels_series = ref_table.obs[identity_column]
        ref_obs_index = ref_table.obs.index.tolist()

    # Build mapping: boundary index → identity label
    ref_id_to_label = {}
    for cid in ref_cell_ids:
        cid_str = str(cid)
        if cid_str in ref_obs_index:
            ref_id_to_label[cid] = ref_labels_series.loc[cid_str]
        elif cid in ref_obs_index:
            ref_id_to_label[cid] = ref_labels_series.loc[cid]

    print(f"  Reference: {len(ref_boundaries)} cells, {len(ref_id_to_label)} with labels")

    # --- Load query ---
    print(f"Loading query: {query_zarr}")
    query_sdata = sd.read_zarr(query_zarr)
    query_boundaries = _extract_boundaries(query_sdata, query_boundary)
    print(f"  Query: {len(query_boundaries)} {query_boundary} boundaries")

    # --- Validate geometries ---
    print("Validating geometries...")
    ref_geoms = np.array([make_valid(g) for g in ref_boundaries.geometry])
    query_geoms = np.array([make_valid(g) for g in query_boundaries.geometry])

    # --- Build R-tree on reference ---
    print("Building spatial index...")
    tree = STRtree(ref_geoms)

    # --- Spatial overlap transfer ---
    print("Transferring identities...")
    n_query = len(query_geoms)
    query_cell_ids = query_boundaries.index.tolist()
    transferred = []
    n_matched = 0
    n_unmatched = 0

    for i, qpoly in enumerate(query_geoms):
        if (i + 1) % 10000 == 0 or i == n_query - 1:
            print(f"  Processed {i + 1}/{n_query} cells "
                  f"({n_matched} matched, {n_unmatched} unmatched)")

        # Query R-tree for bbox candidates
        candidates = tree.query(qpoly)

        best_label = unmatched_label
        best_area = 0.0

        for ci in candidates:
            ref_poly = ref_geoms[ci]
            try:
                inter = qpoly.intersection(ref_poly)
                area = inter.area
            except Exception:
                continue

            if area > best_area:
                best_area = area
                ref_cid = ref_cell_ids[ci]
                label = ref_id_to_label.get(ref_cid, unmatched_label)
                best_label = label

        if best_label != unmatched_label:
            n_matched += 1
        else:
            n_unmatched += 1

        transferred.append(best_label)

    # --- Write output ---
    if output_csv is None:
        query_stem = os.path.splitext(os.path.basename(query_zarr.rstrip("/")))[0]
        output_csv = f"{query_stem}_transferred_idents.csv"

    df_out = pd.DataFrame({
        "query_cell_id": query_cell_ids,
        "transferred_identity": transferred,
    })
    df_out.to_csv(output_csv, index=False)

    elapsed = time.time() - t0

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"Transfer complete in {elapsed:.1f}s")
    print(f"  Total query cells:  {n_query}")
    print(f"  Matched:            {n_matched} ({100*n_matched/n_query:.1f}%)")
    print(f"  Unmatched:          {n_unmatched} ({100*n_unmatched/n_query:.1f}%)")
    print(f"  Output:             {output_csv}")
    print(f"\nIdentity distribution:")
    counts = df_out["transferred_identity"].value_counts()
    for label, count in counts.items():
        print(f"  {label}: {count} ({100*count/n_query:.1f}%)")
    print(f"{'='*50}")

    return output_csv


def list_columns(reference_zarr: str):
    """Print available obs columns in reference zarr."""
    ref_sdata = sd.read_zarr(reference_zarr)
    table = ref_sdata.tables.get("table")
    if table is None:
        available_tables = list(ref_sdata.tables.keys())
        print(f"No 'table' found. Available tables: {available_tables}")
        return
    cols = sorted(table.obs.columns.tolist())
    print(f"Available obs columns in {reference_zarr}:")
    for c in cols:
        n_unique = table.obs[c].nunique()
        print(f"  {c}  ({n_unique} unique values)")


def main():
    parser = argparse.ArgumentParser(
        description="Transfer cell identity labels between segmentations via spatial overlap."
    )
    parser.add_argument("reference_zarr", help="Path to reference SpatialData zarr store")
    parser.add_argument("query_zarr", help="Path to query SpatialData zarr store")
    parser.add_argument(
        "--identity_column", default=None,
        help="Column name in reference obs with identity labels"
    )
    parser.add_argument(
        "--query_boundary", default="cell", choices=["cell", "nucleus"],
        help="Which query boundaries to use (default: cell)"
    )
    parser.add_argument(
        "--output_csv", default=None,
        help="Output CSV path (default: <query_stem>_transferred_idents.csv)"
    )
    parser.add_argument(
        "--unmatched_label", default="Unassigned",
        help="Label for query cells with no overlap (default: Unassigned)"
    )
    parser.add_argument(
        "--labels_file", default=None,
        help="External CSV or parquet with cell labels (index=cell_id). "
             "Use when annotations live outside the zarr, e.g. xenium explorer "
             "cache parquets like annotation_rctd_doublet_proseg:XXXX.parquet"
    )
    parser.add_argument(
        "--list_columns", action="store_true",
        help="List available obs columns in reference and exit"
    )
    args = parser.parse_args()

    if args.list_columns:
        list_columns(args.reference_zarr)
        sys.exit(0)

    if args.identity_column is None:
        parser.error("--identity_column is required (unless using --list_columns)")

    transfer_seg_idents(
        reference_zarr=args.reference_zarr,
        query_zarr=args.query_zarr,
        identity_column=args.identity_column,
        query_boundary=args.query_boundary,
        output_csv=args.output_csv,
        unmatched_label=args.unmatched_label,
        labels_file=args.labels_file,
    )


if __name__ == "__main__":
    main()
