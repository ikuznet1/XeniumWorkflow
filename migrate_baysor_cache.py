#!/usr/bin/env python3
"""
One-time migration: convert existing Baysor cache dirs to SpatialData zarr format.

Usage:
    python migrate_baysor_cache.py
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA_DIR = "output-XETG00217__0038213__Region_1__20241206__182124"
CACHE_ROOT = os.path.expanduser("~/.xenium_explorer_cache")
ZARR_PATH = os.path.join(DATA_DIR, "spatialdata_xenium.zarr")


def load_gene_names(zarr_path):
    import spatialdata
    print(f"Loading gene names from {zarr_path}…", flush=True)
    sdata = spatialdata.read_zarr(zarr_path)
    return list(sdata.tables["table"].var_names)


def load_baysor_cache(out_dir):
    """Replicate _load_cached_baysor: return (cells_df, cell_bounds, seg_df)."""
    seg_parts = []
    for root, dirs, files in os.walk(out_dir):
        if "segmentation.csv" in files:
            seg_parts.append(pd.read_csv(os.path.join(root, "segmentation.csv")))
    if not seg_parts:
        raise FileNotFoundError(f"segmentation.csv not found in {out_dir}")

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

    cell_bounds = {}
    for root, dirs, files in os.walk(out_dir):
        for poly_name in ("segmentation_polygons_2d.json", "segmentation_polygons.json",
                          "polygons.json", "cell_polygons.json"):
            if poly_name in files:
                with open(os.path.join(root, poly_name)) as f:
                    geo = json.load(f)
                for feat in geo.get("features", []):
                    props = feat.get("properties", {})
                    cid = str(props.get("cell", props.get("cell_id", feat.get("id", ""))))
                    coords = feat.get("geometry", {}).get("coordinates", [[]])[0]
                    if coords:
                        cell_bounds[cid] = ([c[0] for c in coords], [c[1] for c in coords])
                break

    return cells_df, cell_bounds, seg


def build_expr(seg_df, cells_df, gene_names):
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    if "gene" not in seg_df.columns:
        return None
    cell_to_row = {c: i for i, c in enumerate(cells_df.index)}
    panel = seg_df[seg_df["gene"].isin(gene_to_idx) & seg_df["cell"].isin(cell_to_row)].copy()
    rows = panel["cell"].map(cell_to_row).astype(int).values
    cols = panel["gene"].map(gene_to_idx).astype(int).values
    mat = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(cells_df), len(gene_names))
    )
    return mat


def build_reseg_sdata(cells_df, cell_bounds, expr_mat, gene_names, source, out_dir):
    import spatialdata
    from spatialdata.models import TableModel, ShapesModel
    import anndata as ad
    import geopandas as gpd
    from shapely.geometry import Polygon

    var_df = pd.DataFrame({"is_imputed": False}, index=pd.Index(gene_names, name="gene"))
    obs_df = cells_df.copy()
    obs_df.index = obs_df.index.astype(str)

    X = sp.csr_matrix(expr_mat, dtype=np.float32) if expr_mat is not None else \
        sp.csr_matrix((len(obs_df), len(gene_names)), dtype=np.float32)

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
                tables={"table": adata}, shapes={"cell_boundaries": shapes}
            )
        except Exception:
            sdata = spatialdata.SpatialData(tables={"table": adata})
    else:
        sdata = spatialdata.SpatialData(tables={"table": adata})

    zarr_path = os.path.join(out_dir, f"spatialdata_{source}.zarr")
    sdata.write(zarr_path)
    print(f"  Written: {zarr_path}", flush=True)
    return zarr_path


def migrate_dir(out_dir, gene_names, dataset_basename):
    print(f"\nMigrating: {os.path.basename(out_dir)}", flush=True)

    # Check if zarr already exists
    zarr_path = os.path.join(out_dir, "spatialdata_baysor.zarr")
    if os.path.isdir(zarr_path):
        print(f"  Already migrated — skipping.", flush=True)
        return

    # Create params.json if missing (old-style cache without param_tag)
    params_path = os.path.join(out_dir, "params.json")
    if not os.path.exists(params_path):
        print("  No params.json found — creating one for old-style cache…", flush=True)
        # Derive a stable param_tag from dir name
        import hashlib
        tag = hashlib.md5(os.path.basename(out_dir).encode()).hexdigest()[:8]
        params = {
            "tool": "baysor", "scale": None, "min_mol": None,
            "use_prior": None, "prior_conf": None,
            "x_min": None, "x_max": None, "y_min": None, "y_max": None,
            "n_cells": None, "param_tag": tag,
        }
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)
        print(f"  Created params.json (param_tag={tag})", flush=True)

    print("  Loading segmentation data…", flush=True)
    cells_df, cell_bounds, seg_df = load_baysor_cache(out_dir)
    print(f"  Loaded {len(cells_df):,} cells, {len(cell_bounds):,} boundaries", flush=True)

    # Update n_cells in params.json
    with open(params_path) as f:
        params = json.load(f)
    if params.get("n_cells") is None:
        params["n_cells"] = len(cells_df)
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)

    print("  Building expression matrix…", flush=True)
    expr_mat = build_expr(seg_df, cells_df, gene_names)
    if expr_mat is not None:
        print(f"  Expression matrix: {expr_mat.shape}", flush=True)

    print("  Writing SpatialData zarr…", flush=True)
    build_reseg_sdata(cells_df, cell_bounds, expr_mat, gene_names, "baysor", out_dir)


def main():
    gene_names = load_gene_names(ZARR_PATH)
    print(f"Loaded {len(gene_names)} gene names from xenium zarr.", flush=True)

    dataset_basename = os.path.basename(DATA_DIR)
    baysor_dirs = [
        os.path.join(CACHE_ROOT, d)
        for d in os.listdir(CACHE_ROOT)
        if d.startswith(f"baysor_{dataset_basename}") and
           os.path.isdir(os.path.join(CACHE_ROOT, d))
    ]

    if not baysor_dirs:
        print("No Baysor cache dirs found.")
        return

    print(f"Found {len(baysor_dirs)} Baysor cache dir(s).")
    for out_dir in sorted(baysor_dirs):
        migrate_dir(out_dir, gene_names, dataset_basename)

    print("\nMigration complete.")


if __name__ == "__main__":
    main()
