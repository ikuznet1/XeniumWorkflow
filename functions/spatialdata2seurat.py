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
  --roi PATH            ROI source: either a directory with roi_<cls>.parquet files,
                        or the app's rois_<hash>.json cache file (polygon-based filter).
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
    merge_assays: bool = False,
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
    merge_assays        : Merge imputed genes into the main assay (rounds to int);
                          if False (default), imputed genes go in a separate "Imputed" assay
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

    # Increase R's vector memory limit (default can be too low for large matrices)
    os.environ.setdefault("R_MAX_VSIZE", "32Gb")
    importr("Seurat")
    importr("SeuratObject")
    importr("Matrix")

    output_rds = os.path.abspath(output_rds)
    os.makedirs(os.path.dirname(output_rds) or ".", exist_ok=True)

    # ── Resolve run directory (parent of the zarr, or the cache_dir if it's a run dir) ──
    _cache_dir_early = os.path.expanduser(cache_dir) if cache_dir else os.path.expanduser("~/.xenium_explorer_cache")
    run_dir = os.path.dirname(os.path.abspath(sdata_path))
    if os.path.exists(os.path.join(_cache_dir_early, "params.json")):
        run_dir = _cache_dir_early

    # ── Load SpatialData ───────────────────────────────────────────────────────
    print(f"Reading SpatialData from {sdata_path} …", flush=True)
    sdata = sd.read_zarr(sdata_path)
    adata = sdata.tables["table"]

    print(f"  {len(adata)} cells, {len(adata.var)} genes", flush=True)

    # sd.read_zarr may miss obs columns added after initial write (stale consolidated
    # metadata). Check for specific missing columns directly from the raw zarr obs group.
    _wanted_extra = ["split_umap_1", "split_umap_2", "cluster_split_10",
                     "cell_area", "nucleus_area"]
    _missing = [c for c in _wanted_extra if c not in adata.obs.columns]
    if _missing:
        try:
            import zarr as _zarr_mod
            _obs_grp = _zarr_mod.open_group(
                os.path.join(sdata_path, "tables", "table", "obs"),
                mode="r")
            _recovered = []
            for _ec in _missing:
                if _ec in _obs_grp:
                    try:
                        adata.obs[_ec] = _obs_grp[_ec][:]
                        _recovered.append(_ec)
                    except Exception:
                        pass
            if _recovered:
                print(f"  Recovered obs columns from raw zarr: {_recovered}", flush=True)
        except Exception:
            pass

    # ── ROI filter ─────────────────────────────────────────────────────────────
    if roi_cls and roi_name:
        roi_col = f"roi_{roi_cls}"

        # If an external ROI source is given, load and apply it
        if roi_dir:
            roi_dir_expanded = os.path.expanduser(roi_dir)

            if roi_dir_expanded.endswith(".json") and os.path.isfile(roi_dir_expanded):
                # App-format ROI JSON: list of {cls, name, polygon_xy} dicts
                import json as _json
                from shapely.geometry import Polygon as _SPoly, Point as _SPoint
                with open(roi_dir_expanded) as _f:
                    _rois = _json.load(_f)
                # Find the matching ROI polygon
                _matching = [r for r in _rois
                             if str(r.get("cls")) == str(roi_cls)
                             and str(r.get("name")) == str(roi_name)]
                if not _matching:
                    _avail = [(r.get("cls"), r.get("name")) for r in _rois]
                    raise ValueError(f"No ROI with cls='{roi_cls}' name='{roi_name}' in "
                                     f"{roi_dir_expanded}. Available: {_avail}")
                # Run point-in-polygon for each matching polygon (union if multiple)
                xs = adata.obs["x_centroid"].values.astype(float)
                ys = -adata.obs["y_centroid"].values.astype(float)  # negated as in app
                in_roi = np.zeros(len(adata), dtype=bool)
                for _r in _matching:
                    _poly = _SPoly(_r["polygon_xy"])
                    in_roi |= np.array([_poly.contains(_SPoint(x, y))
                                        for x, y in zip(xs, ys)])
                adata.obs[roi_col] = np.where(in_roi, roi_name, pd.NA)
                print(f"  ROI: applied polygon from {os.path.basename(roi_dir_expanded)} "
                      f"({in_roi.sum():,} cells inside)", flush=True)

            else:
                # Directory with roi_<cls>.parquet
                roi_pq = os.path.join(roi_dir_expanded, f"roi_{roi_cls}.parquet")
                if not os.path.exists(roi_pq):
                    candidates = [f for f in os.listdir(roi_dir_expanded)
                                  if f.endswith(".parquet")]
                    if not candidates:
                        raise FileNotFoundError(f"No parquet found in {roi_dir_expanded}")
                    roi_pq = os.path.join(roi_dir_expanded, candidates[0])
                    print(f"  ROI: using {roi_pq}", flush=True)
                roi_df = pd.read_parquet(roi_pq)
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

    # ── Load SpaGE imputed genes from separate zarr (if present) ──────────────
    # SpaGE results are NOT stored in the main spatialdata zarr — they live in a
    # separate zarr array referenced by {run_dir}/spage_result.json.
    # Rows correspond to the full adata (before ROI / corrected filtering).
    _spage_genes  = []
    _spage_zarr   = None
    _spage_obs_idx = list(adata.obs_names)  # capture before any filtering
    _spage_ref = os.path.join(run_dir, "spage_result.json")
    if include_imputed and os.path.exists(_spage_ref):
        try:
            import json as _sj
            import zarr as _szarr
            _sref = _sj.load(open(_spage_ref))
            _zpath = _sref.get("path", "")
            if os.path.isdir(_zpath):
                _spage_zarr  = _szarr.open_array(_zpath, mode="r")
                _spage_genes = list(_spage_zarr.attrs.get("genes", _sref.get("genes", [])))
                print(f"  SpaGE: found {len(_spage_genes)} imputed genes in {os.path.basename(_zpath)}",
                      flush=True)
        except Exception as _se:
            print(f"  SpaGE: could not load spage_result.json: {_se}", flush=True)

    # ── Split panel vs imputed genes ───────────────────────────────────────────
    # Imputed genes come from the spage zarr; all genes in adata are panel genes.
    panel_genes   = list(adata.var_names)
    imputed_genes = _spage_genes
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

    X_panel = X_full  # all columns are panel genes

    # ── Build imputed matrix aligned to final cell set ─────────────────────────
    # Reads the zarr in chunks to avoid loading the full dense array into memory.
    _IMPUTE_CHUNK = 10_000  # rows per chunk
    X_imputed = None
    if _spage_zarr is not None and _spage_genes:
        try:
            # Map from full obs index to row positions in the spage zarr
            _idx_map = {str(c): i for i, c in enumerate(_spage_obs_idx)}
            _final_ids = list(adata.obs_names)
            _row_sel = [_idx_map[str(c)] for c in _final_ids if str(c) in _idx_map]
            _missing  = sum(1 for c in _final_ids if str(c) not in _idx_map)
            if _missing:
                print(f"  SpaGE: {_missing} cells not found in zarr; will be zero", flush=True)
            # Chunked read: convert each chunk to sparse immediately
            _chunks = []
            for _ci in range(0, len(_row_sel), _IMPUTE_CHUNK):
                _batch = _row_sel[_ci:_ci + _IMPUTE_CHUNK]
                _dense = _spage_zarr.oindex[_batch, :]
                _chunks.append(sp.csr_matrix(_dense.astype(np.float32)))
                del _dense
            X_imputed = sp.vstack(_chunks, format="csr") if _chunks else None
            del _chunks
            if X_imputed is not None:
                print(f"  SpaGE: loaded imputed matrix {X_imputed.shape} "
                      f"({_IMPUTE_CHUNK}-row chunks)", flush=True)
        except Exception as _ie:
            print(f"  SpaGE: could not build imputed matrix: {_ie}", flush=True)
            X_imputed = None

    if X_imputed is None:
        imputed_genes = []

    # ── Merge assays: combine panel + imputed into a single matrix ────────────
    if merge_assays and X_imputed is not None and len(imputed_genes) > 0:
        # Round imputed values to non-negative integers; zero out values < 1
        # (SpaGE produces continuous values — without thresholding the matrix
        #  is effectively dense and can exhaust R's memory)
        X_imputed_int = X_imputed.copy()
        X_imputed_int.data = np.round(X_imputed_int.data)
        X_imputed_int.data[X_imputed_int.data < 1] = 0
        X_imputed_int.eliminate_zeros()
        X_imputed_int = X_imputed_int.astype(np.float64)
        _nnz_frac = X_imputed_int.nnz / (X_imputed_int.shape[0] * X_imputed_int.shape[1])
        # Horizontal stack: append imputed gene columns to the cells×genes matrix
        X_panel = sp.hstack([X_panel, X_imputed_int], format="csr")
        del X_imputed_int
        panel_genes = panel_genes + imputed_genes
        print(f"  Merged assays: {len(panel_genes)} total genes "
              f"({len(imputed_genes)} imputed, rounded to int, "
              f"{_nnz_frac:.1%} non-zero)", flush=True)
        # Clear imputed so the separate assay path is skipped
        X_imputed = None
        imputed_genes = []

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

    # Proseg stores cell area as 'volume' — rename to cell_area
    if "cell_area" not in meta.columns and "volume" in meta.columns:
        meta["cell_area"] = meta["volume"]

    # Drop columns that Seurat can't handle (list/dict dtype)
    for col in list(meta.columns):
        if meta[col].dtype == object:
            try:
                meta[col] = meta[col].astype(str)
            except Exception:
                meta = meta.drop(columns=[col])

    # ── Load annotations from cache ────────────────────────────────────────────
    _cache_dir = _cache_dir_early
    # If the provided cache_dir is a run dir (contains params.json), annotation
    # parquets are one level up.
    if os.path.exists(os.path.join(_cache_dir, "params.json")):
        _cache_dir = os.path.dirname(_cache_dir)
    if os.path.isdir(_cache_dir):
        # Derive param_tag and dataset_hash from params.json in the run directory
        param_tag    = None
        dataset_hash = None
        params_json  = os.path.join(run_dir, "params.json")
        if os.path.exists(params_json):
            import json as _json
            _pj = _json.load(open(params_json))
            param_tag    = _pj.get("param_tag")
            dataset_hash = _pj.get("dataset_hash")
            print(f"  Cache: param_tag={param_tag}, dataset_hash={dataset_hash}", flush=True)

        # Mapping: fragment to look for in filename → target column name
        _annot_targets = {
            "labels_seurat":        "cell_type_seurat",
            "labels_rctd_doublet":  "cell_type_rctd_doublet",
        }
        for _fragment, _col in _annot_targets.items():
            if _col in meta.columns:
                continue  # already present in obs
            # All parquets containing this annotation type
            candidates = [
                f for f in os.listdir(_cache_dir)
                if f.endswith(".parquet") and _fragment in f
            ]
            # Narrow to same dataset (by hash suffix before .parquet)
            if dataset_hash:
                ds_specific = [f for f in candidates if f.endswith(f"_{dataset_hash}.parquet")]
                if ds_specific:
                    candidates = ds_specific
                elif candidates:
                    print(f"  Warning: {_col} — no parquets found for dataset_hash={dataset_hash}; "
                          f"run annotation in the app for this run first", flush=True)
                    continue
            # Within same-dataset candidates, prefer the exact param_tag run
            if param_tag:
                specific = [f for f in candidates if param_tag in f]
                if specific:
                    candidates = specific
            if not candidates:
                continue

            _meta_idx = meta.index.astype(str)

            def _load_ann(fname):
                _a = pd.read_parquet(os.path.join(_cache_dir, fname))
                return _a["label"] if "label" in _a.columns else _a.iloc[:, 0]

            def _match(ann):
                """Align ann to _meta_idx, trying str then int. Returns (series, n_matched)."""
                s = ann.copy(); s.index = s.index.astype(str)
                m = s.reindex(_meta_idx); n = int(m.notna().sum())
                try:
                    si = ann.copy(); si.index = si.index.astype(int)
                    m2 = si.reindex(_meta_idx.astype(int)); m2.index = _meta_idx
                    if int(m2.notna().sum()) > n:
                        m, n = m2, int(m2.notna().sum())
                except (ValueError, TypeError):
                    pass
                return m, n

            candidates.sort(key=lambda f: os.path.getmtime(os.path.join(_cache_dir, f)))

            # When dataset_hash is unknown, try every candidate and pick the best cell-ID overlap
            if not dataset_hash and len(candidates) > 1:
                _best_f, _best_m, _best_n = None, None, -1
                for _cf in candidates:
                    try:
                        _m, _n = _match(_load_ann(_cf))
                        if _n > _best_n:
                            _best_f, _best_m, _best_n = _cf, _m, _n
                    except Exception:
                        pass
                if _best_f is None:
                    continue
                chosen, _matched, _n_matched = os.path.join(_cache_dir, _best_f), _best_m, _best_n
                _ann = _load_ann(_best_f)
            else:
                chosen = os.path.join(_cache_dir, candidates[-1])
                try:
                    _ann = _load_ann(candidates[-1])
                    _matched, _n_matched = _match(_ann)
                except Exception as _e:
                    print(f"  Warning: could not load {_col} from {chosen}: {_e}", flush=True)
                    continue

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
            if "split_umap_1" in meta.columns and "split_umap_2" in meta.columns:
                u1_col, u2_col = "split_umap_1", "split_umap_2"
            elif "umap_1" in meta.columns and "umap_2" in meta.columns:
                print("  Warning: --use_corrected set but split_umap_1/2 not found in obs; "
                      "falling back to umap_1/2", flush=True)
                u1_col, u2_col = "umap_1", "umap_2"
            else:
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
    p.add_argument("--merge_assays",   action="store_true",
                   help="Merge imputed genes into the main assay (rounds to int) "
                        "instead of a separate Imputed assay")
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
        merge_assays=args.merge_assays,
        use_corrected=args.use_corrected,
        roi_dir=args.roi,
        roi_cls=args.roi_cls,
        roi_name=args.roi_name,
        cache_dir=args.cache_dir,
    )
