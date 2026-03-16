#!/usr/bin/env python3
"""
merge_spatialdata.py — Merge multiple SpatialData zarr stores into one.

Offsets x/y coordinates so datasets don't spatially overlap. The merged
store can be opened in xenium_explorer.py as a segmentation result via
the seg-source dropdown (appears as a cached run once copied to the
explorer cache directory).

Usage
-----
  python merge_spatialdata.py s1.zarr s2.zarr -o merged.zarr
  python merge_spatialdata.py s1.zarr s2.zarr s3.zarr -o merged.zarr --gap 500
  python merge_spatialdata.py s1.zarr s2.zarr -o merged.zarr --layout column
  python merge_spatialdata.py s1.zarr s2.zarr -o merged.zarr --layout grid
  python merge_spatialdata.py s1.zarr s2.zarr -o merged.zarr --labels ctrl treat

Options
-------
  --gap UM        Gap between datasets in µm (default: 200)
  --layout        Arrangement: row (default) | column | grid
  --labels        Dataset labels used as cell-ID prefix (default: s0, s1, …)
  --genes         Gene handling: union (default) | intersect
  --overwrite     Overwrite output if it already exists
"""

import argparse
import math
import sys
import warnings
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp


# ── Coordinate helpers ────────────────────────────────────────────────────────

def _bbox(obs: pd.DataFrame) -> tuple:
    """(xmin, ymin, xmax, ymax) from x_centroid / y_centroid in obs."""
    x = obs.get("x_centroid", pd.Series(dtype=float))
    y = obs.get("y_centroid", pd.Series(dtype=float))
    if x.empty or y.empty or x.isna().all():
        return (0.0, 0.0, 0.0, 0.0)
    return float(x.min()), float(y.min()), float(x.max()), float(y.max())


def compute_offsets(bboxes: list, layout: str, gap: float) -> list:
    """
    Return list of (dx, dy) offsets that shift each dataset so none overlap.

    layout : "row"    — datasets side-by-side along x (default)
             "column" — datasets stacked along y
             "grid"   — square-ish grid, row-major
    """
    n = len(bboxes)
    offsets = [(0.0, 0.0)] * n

    if layout == "row":
        cursor = 0.0
        for i, (xmin, ymin, xmax, ymax) in enumerate(bboxes):
            offsets[i] = (cursor - xmin, -ymin)          # align tops to y=0
            cursor += (xmax - xmin) + gap

    elif layout == "column":
        cursor = 0.0
        for i, (xmin, ymin, xmax, ymax) in enumerate(bboxes):
            offsets[i] = (-xmin, cursor - ymin)           # align lefts to x=0
            cursor += (ymax - ymin) + gap

    elif layout == "grid":
        cols = math.ceil(math.sqrt(n))
        cell_w = max(xmax - xmin for xmin, ymin, xmax, ymax in bboxes) + gap
        cell_h = max(ymax - ymin for xmin, ymin, xmax, ymax in bboxes) + gap
        for i, (xmin, ymin, xmax, ymax) in enumerate(bboxes):
            col = i % cols
            row = i // cols
            offsets[i] = (col * cell_w - xmin, row * cell_h - ymin)

    else:
        raise ValueError(f"Unknown layout {layout!r}. Choose row, column, or grid.")

    return offsets


# ── Core merge ────────────────────────────────────────────────────────────────

def merge_spatialdata(
    input_paths,
    output_path,
    gap: float = 200.0,
    layout: str = "row",
    labels: list = None,
    genes: str = "union",
    overwrite: bool = False,
):
    """
    Merge SpatialData zarr stores into a single output zarr.

    Parameters
    ----------
    input_paths : list[str | Path]
    output_path : str | Path
    gap         : float  — µm gap between adjacent datasets (default 200)
    layout      : str    — "row" | "column" | "grid"
    labels      : list[str] | None — prefix for cell IDs (default s0, s1, …)
    genes       : str    — "union" (fill missing with 0) | "intersect"
    overwrite   : bool
    """
    import spatialdata as sd

    input_paths = [Path(p) for p in input_paths]
    output_path = Path(output_path)
    n = len(input_paths)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass --overwrite to replace it."
        )

    if labels is None:
        labels = [f"s{i}" for i in range(n)]
    if len(labels) != n:
        raise ValueError("--labels must have the same number of entries as input zarrs.")

    # ── 1. Load stores ─────────────────────────────────────────────────────
    print(f"Loading {n} SpatialData stores…")
    sdatas, tables = [], []
    for p in input_paths:
        print(f"  {p}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sdata = sd.read_zarr(str(p))
        if "table" not in sdata.tables:
            raise ValueError(f"{p} has no 'table' element.")
        sdatas.append(sdata)
        tables.append(sdata.tables["table"])

    # ── 2. Compute coordinate offsets ──────────────────────────────────────
    bboxes  = [_bbox(tbl.obs) for tbl in tables]
    offsets = compute_offsets(bboxes, layout=layout, gap=gap)
    for lbl, (xmin, ymin, xmax, ymax), (dx, dy) in zip(labels, bboxes, offsets):
        print(f"  {lbl}: bbox ({xmax - xmin:.0f} × {ymax - ymin:.0f} µm) "
              f"→ offset dx={dx:.1f} dy={dy:.1f}")

    # ── 3. Build gene list ─────────────────────────────────────────────────
    if genes == "union":
        gene_list = []
        seen: set = set()
        for tbl in tables:
            for g in tbl.var_names:
                if g not in seen:
                    gene_list.append(g)
                    seen.add(g)
    else:  # intersect
        gene_set = set(tables[0].var_names)
        for tbl in tables[1:]:
            gene_set &= set(tbl.var_names)
        gene_list = [g for g in tables[0].var_names if g in gene_set]

    print(f"Merging {len(gene_list)} genes ({genes})…")

    # ── 4. Merge obs + expression ──────────────────────────────────────────
    obs_parts, X_parts = [], []

    for i, (tbl, (dx, dy)) in enumerate(zip(tables, offsets)):
        prefix = labels[i]
        obs = tbl.obs.copy()

        # Unique cell IDs
        obs.index = pd.Index([f"{prefix}_{c}" for c in obs.index], name=obs.index.name)
        if "cell_id" in obs.columns:
            obs["cell_id"] = obs["cell_id"].astype(str).map(lambda c: f"{prefix}_{c}")

        # Offset spatial coordinates
        for col, delta in (("x_centroid", dx), ("y_centroid", dy)):
            if col in obs.columns:
                obs[col] = obs[col].astype(float) + delta

        obs["dataset"] = prefix
        obs_parts.append(obs)

        # Align expression matrix to target gene list
        X = tbl.X if sp.issparse(tbl.X) else sp.csr_matrix(tbl.X)
        X_csc = X.tocsc()
        gene_to_col = {g: j for j, g in enumerate(tbl.var_names)}

        if list(tbl.var_names) == gene_list:
            X_parts.append(X_csc.tocsr())
        else:
            cols = []
            for g in gene_list:
                if g in gene_to_col:
                    cols.append(X_csc.getcol(gene_to_col[g]))
                else:
                    cols.append(sp.csc_matrix((X.shape[0], 1), dtype=np.float32))
            X_parts.append(sp.hstack(cols, format="csr"))

    # Concatenate (fill missing obs columns with NaN across datasets)
    merged_obs = pd.concat(obs_parts, axis=0, sort=False)
    merged_X   = sp.vstack(X_parts, format="csr").astype(np.float32)

    # var DataFrame — propagate is_imputed flag if present
    is_imputed: dict = {}
    for tbl in tables:
        if "is_imputed" in tbl.var.columns:
            for g, v in zip(tbl.var_names, tbl.var["is_imputed"]):
                is_imputed[g] = v
    var_name = tables[0].var.index.name or "gene"
    merged_var = pd.DataFrame(
        {"is_imputed": [is_imputed.get(g, False) for g in gene_list]},
        index=pd.Index(gene_list, name=var_name),
    )

    merged_adata = anndata.AnnData(X=merged_X, obs=merged_obs, var=merged_var)
    merged_adata.uns["dataset_labels"] = labels
    print(f"Merged table: {merged_adata.n_obs:,} cells × {merged_adata.n_vars:,} genes")

    # ── 5. Merge shapes ────────────────────────────────────────────────────
    merged_shapes = {}
    try:
        from shapely.affinity import translate as _translate
        import geopandas as gpd

        shape_keys: set = set()
        for sdata in sdatas:
            shape_keys |= set(sdata.shapes.keys())

        for key in shape_keys:
            parts = []
            for i, (sdata, (dx, dy)) in enumerate(zip(sdatas, offsets)):
                if key not in sdata.shapes:
                    continue
                gdf = sdata.shapes[key].copy()
                prefix = labels[i]
                gdf.index = pd.Index(
                    [f"{prefix}_{idx}" for idx in gdf.index],
                    name=gdf.index.name,
                )
                gdf["geometry"] = gdf["geometry"].apply(
                    lambda geom: _translate(geom, xoff=dx, yoff=dy)
                    if geom is not None else geom
                )
                parts.append(gdf)

            if parts:
                merged_gdf = gpd.GeoDataFrame(pd.concat(parts, axis=0, sort=False))
                merged_gdf.attrs = parts[0].attrs   # preserve coordinate_system etc.
                merged_shapes[key] = merged_gdf

        print(f"Merged shapes: {list(merged_shapes.keys())}")

    except ImportError as exc:
        print(f"WARNING: shapes not merged ({exc}). Install geopandas + shapely.")

    # ── 6. Parse table for SpatialData (requires region/instance_id linkage) ──
    if merged_shapes:
        region_name = next(iter(merged_shapes))
        merged_adata.obs["region"]      = region_name
        merged_adata.obs["instance_id"] = np.arange(len(merged_adata), dtype=np.int64)
        merged_adata = sd.models.TableModel.parse(
            merged_adata,
            region=region_name,
            region_key="region",
            instance_key="instance_id",
        )

    # ── 7. Write ───────────────────────────────────────────────────────────
    print(f"Writing to {output_path}…")
    sdata_out = sd.SpatialData(
        shapes=merged_shapes,
        tables={"table": merged_adata},
    )
    sdata_out.write(str(output_path), overwrite=overwrite)
    print(f"Done → {output_path}")
    return sdata_out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=(
            "Merge multiple SpatialData zarr stores into one, "
            "offsetting coordinates so datasets don't overlap."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Options")[0].strip(),
    )
    p.add_argument("inputs", nargs="+", metavar="ZARR",
                   help="Input SpatialData zarr paths (2 or more)")
    p.add_argument("-o", "--output", required=True, metavar="ZARR",
                   help="Output zarr path")
    p.add_argument("--gap", type=float, default=200.0, metavar="UM",
                   help="Gap between datasets in µm (default: 200)")
    p.add_argument("--layout", choices=["row", "column", "grid"], default="row",
                   help="Arrangement: row (default), column, or grid")
    p.add_argument("--labels", nargs="+", metavar="LABEL",
                   help="Dataset labels used as cell-ID prefix (default: s0, s1, …)")
    p.add_argument("--genes", choices=["union", "intersect"], default="union",
                   help="Gene handling: union fills missing with 0 (default); "
                        "intersect keeps only shared genes")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite output if it already exists")

    args = p.parse_args()

    if len(args.inputs) < 2:
        p.error("Need at least 2 input zarr paths.")

    merge_spatialdata(
        input_paths=args.inputs,
        output_path=args.output,
        gap=args.gap,
        layout=args.layout,
        labels=args.labels,
        genes=args.genes,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
