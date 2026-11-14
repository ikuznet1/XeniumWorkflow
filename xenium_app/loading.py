"""Readers: Xenium bundle -> SpatialData store -> the DATA dict."""

import os
import re
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py

from . import reseg


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_xenium_data(data_dir: str) -> dict:
    data_dir = os.path.abspath(data_dir)  # normalise so cache hashes are stable
    print(f"Loading from: {data_dir}", flush=True)
    zarr_path = os.path.join(data_dir, "spatialdata_xenium.zarr")
    if not os.path.isdir(zarr_path):
        print("  Generating spatialdata_xenium.zarr (one-time, ~30–60 s)…", flush=True)
        _build_xenium_sdata(data_dir)
    return _load_from_xenium_sdata(zarr_path, data_dir)


def _build_boundary_dict(df: pd.DataFrame) -> dict:
    """Return {cell_id: (vertex_x_array, vertex_y_array)} for fast polygon lookup.

    A cell with several rings (XOA 2+ nucleus boundaries: one `label_id` per nucleus)
    gets one NaN-separated vertex array, so the rings are not joined into one path."""
    if df.empty:
        return {}
    # Pre-sort by cell_id for contiguous slicing via numpy split
    cell_ids = df["cell_id"].values
    vx = df["vertex_x"].values
    vy = df["vertex_y"].values
    order = np.argsort(cell_ids, kind="mergesort")
    if "label_id" in df.columns:
        # stable two-pass sort → (cell_id, label_id), vertex order within a ring kept
        _lab = df["label_id"].values
        order = np.argsort(_lab, kind="mergesort")
        order = order[np.argsort(cell_ids[order], kind="mergesort")]
    cell_ids_s = cell_ids[order]
    vx_s = vx[order]
    vy_s = vy[order]
    if "label_id" in df.columns:
        _lab_s = _lab[order]
        _brk = np.flatnonzero((cell_ids_s[1:] == cell_ids_s[:-1]) & (_lab_s[1:] != _lab_s[:-1])) + 1
        if len(_brk):
            # NaN vertex between two rings of the same cell
            vx_s = np.insert(vx_s.astype(float), _brk, np.nan)
            vy_s = np.insert(vy_s.astype(float), _brk, np.nan)
            cell_ids_s = np.insert(cell_ids_s, _brk, cell_ids_s[_brk])
    # Find split points where cell_id changes
    change = np.flatnonzero(cell_ids_s[1:] != cell_ids_s[:-1]) + 1
    unique_ids = cell_ids_s[np.r_[0, change]]
    vx_groups = np.split(vx_s, change)
    vy_groups = np.split(vy_s, change)
    return {cid: (vxg, vyg) for cid, vxg, vyg in zip(unique_ids, vx_groups, vy_groups)}


def _gdf_to_bounds_dict(gdf) -> dict:
    """Convert GeoDataFrame of cell polygons back to {cell_id: (x_verts, y_verts)}."""
    import shapely
    geoms = np.asarray(gdf.geometry.values)
    # Polygons (type id 3) and MultiPolygons (6: a cell with several rings, e.g. several
    # nuclei); missing/empty geometries are skipped as before
    _tid = shapely.get_type_id(geoms)
    ok = ((_tid == 3) | (_tid == 6)) & ~shapely.is_empty(geoms)
    if not ok.any():
        return {}
    parts, pidx = shapely.get_parts(geoms[ok], return_index=True)
    coords, ridx = shapely.get_coordinates(shapely.get_exterior_ring(parts),
                                           return_index=True)
    cidx = pidx[ridx]                       # geometry (cell) index of every vertex
    # NaN vertex between two parts of the same cell
    _brk = np.flatnonzero((ridx[1:] != ridx[:-1]) & (cidx[1:] == cidx[:-1])) + 1
    cx, cy = coords[:, 0], coords[:, 1]
    if len(_brk):
        cx = np.insert(cx, _brk, np.nan)
        cy = np.insert(cy, _brk, np.nan)
        cidx = np.insert(cidx, _brk, cidx[_brk])
    change = np.flatnonzero(cidx[1:] != cidx[:-1]) + 1
    _starts = np.r_[0, change]
    keys = gdf.index.values[ok][cidx[_starts]]
    xs = np.split(cx, change)
    ys = np.split(cy, change)
    # Stores written before multi-ring support hold a multi-nucleus cell as ONE ring made of
    # its closed rings back to back. Split such a ring where it returns to its start vertex,
    # but only when every resulting piece is itself a closed ring.
    _first = np.repeat(_starts, np.diff(np.r_[_starts, len(cx)]))
    _pos   = np.arange(len(cx))
    _last  = np.r_[change, len(cx)] - 1
    _mid   = (cx == cx[_first]) & (cy == cy[_first]) & (_pos != _first)
    _mid[_last] = False
    for _g in np.unique(np.searchsorted(_starts, np.flatnonzero(_mid), side="right") - 1):
        gx, gy = xs[_g], ys[_g]
        if np.isnan(gx).any():
            continue
        pieces, a, n = [], 0, len(gx)
        while a < n:
            if a == n - 1 and gx[a] == gx[0] and gy[a] == gy[0]:
                break                       # closing vertex shapely added to the merged ring
            hit = np.flatnonzero((gx[a + 1:] == gx[a]) & (gy[a + 1:] == gy[a]))
            if not len(hit) or hit[0] < 2:
                pieces = []
                break
            b = a + 1 + hit[0]
            pieces.append((a, b + 1))
            a = b + 1
        if len(pieces) > 1:
            xs[_g] = np.concatenate([np.r_[gx[a:b], np.nan] for a, b in pieces])[:-1]
            ys[_g] = np.concatenate([np.r_[gy[a:b], np.nan] for a, b in pieces])[:-1]
    return dict(zip(keys, zip(xs, ys)))


def _cell_ids_as_str(values) -> pd.Index:
    """Cell ids / barcodes as text: bytes decoded, numbers written without a decimal part
    (a float column 1.0, 2.0 names the cells 1, 2)."""
    out = []
    for v in values:
        if isinstance(v, (bytes, bytearray)):
            v = bytes(v).decode("utf-8", "replace")
        elif isinstance(v, (float, np.floating)) and np.isfinite(v) and float(v).is_integer():
            v = int(v)
        out.append(str(v))
    return pd.Index(out, dtype=object, name=getattr(values, "name", None))


def _align_matrix_to_cells(expr, barcodes, cell_ids):
    """Rows of the cell x gene matrix *expr* (one per entry of *barcodes*) in the order of
    *cell_ids*, as CSR. Both id lists are compared as text. A cell without a matrix row
    gets an empty row, with a printed warning that says how many there are; when NO cell
    is found among the barcodes the ids of the two files do not correspond and a
    ValueError is raised - an all-zero matrix is never returned silently."""
    barcodes = _cell_ids_as_str(barcodes)
    cell_ids = _cell_ids_as_str(cell_ids)
    barcode_idx = {b: i for i, b in enumerate(barcodes)}
    row_order   = np.array([barcode_idx.get(cid, -1) for cid in cell_ids], dtype=np.int64)
    valid_mask  = row_order >= 0
    n_cells, n_found = len(cell_ids), int(valid_mask.sum())
    if n_cells and expr.shape[0] and n_found == 0:
        raise ValueError(
            f"None of the {n_cells:,} cell ids of cells.parquet (e.g. {list(cell_ids[:3])}) is among the "
            f"{len(barcodes):,} barcodes of cell_feature_matrix.h5 (e.g. {list(barcodes[:3])}): the two "
            f"files do not describe the same cells, and no expression matrix was built.")
    if valid_mask.all():
        return expr[row_order].tocsr()
    print(f"  WARNING: {n_cells - n_found:,} of {n_cells:,} cells of cells.parquet have no row in "
          f"cell_feature_matrix.h5 (e.g. {list(cell_ids[~valid_mask][:3])}): their counts are zero for "
          f"every gene. Check that the two files come from the same run.", flush=True)
    rows_v = np.where(valid_mask)[0]
    expr_aligned = sp.lil_matrix((n_cells, expr.shape[1]), dtype=np.float32)
    expr_aligned[rows_v] = expr[row_order[valid_mask]]
    return expr_aligned.tocsr()


def _build_xenium_sdata(data_dir: str) -> None:
    """Read all raw Xenium files and write spatialdata_xenium.zarr next to them (one-time generation)."""
    try:
        import spatialdata
        from spatialdata.models import TableModel, ShapesModel
        import anndata as ad
        import geopandas as gpd
        from shapely.geometry import Polygon
    except ImportError as e:
        raise ImportError(
            f"spatialdata, anndata, geopandas, and shapely are required to generate the zarr store: {e}\n"
            "Install with: pip install spatialdata anndata geopandas shapely"
        )

    print("  Reading raw files for spatialdata_xenium.zarr generation…", flush=True)

    with open(os.path.join(data_dir, "experiment.xenium")) as f:
        metadata = json.load(f)

    cells = pd.read_parquet(os.path.join(data_dir, "cells.parquet"))
    cells = cells.set_index("cell_id")
    # Cell ids are text everywhere in the app. Bundles of older Xenium Onboard Analysis
    # versions number the cells (integer cell_id) while the matrix barcodes are text:
    # matched without this cast, no cell would find its matrix row.
    cells.index = _cell_ids_as_str(cells.index)

    # Secondary analysis is optional: use analysis/ when extracted, else read the
    # same CSVs straight from analysis.tar.gz, else go without UMAP / clustering.
    umap_df = None
    cluster_methods = {}
    _umap_rel   = "analysis/umap/gene_expression_2_components/projection.csv"
    cluster_dir = os.path.join(data_dir, "analysis/clustering")
    _analysis_tar = os.path.join(data_dir, "analysis.tar.gz")
    if os.path.isdir(os.path.join(data_dir, "analysis")):
        if os.path.exists(os.path.join(data_dir, _umap_rel)):
            umap_df = pd.read_csv(os.path.join(data_dir, _umap_rel), index_col="Barcode", dtype={"Barcode": str})
        for method in (sorted(os.listdir(cluster_dir)) if os.path.isdir(cluster_dir) else []):
            path = os.path.join(cluster_dir, method, "clusters.csv")
            if os.path.exists(path):
                df_c = pd.read_csv(path, index_col="Barcode", dtype={"Barcode": str})
                cluster_methods[method] = df_c["Cluster"]
    elif os.path.exists(_analysis_tar):
        try:
            import tarfile
            with tarfile.open(_analysis_tar, "r:gz") as _tf:
                for _m in sorted(_tf.getmembers(), key=lambda m: m.name):
                    _name = _m.name.lstrip("./")
                    _parts = _name.split("/")
                    if not _m.isfile():
                        continue
                    if _name == _umap_rel:
                        umap_df = pd.read_csv(_tf.extractfile(_m), index_col="Barcode", dtype={"Barcode": str})
                    elif (len(_parts) == 4 and _parts[:2] == ["analysis", "clustering"]
                          and _parts[3] == "clusters.csv"):
                        df_c = pd.read_csv(_tf.extractfile(_m), index_col="Barcode", dtype={"Barcode": str})
                        cluster_methods[_parts[2]] = df_c["Cluster"]
        except Exception as exc:
            print(f"  Warning: could not read analysis.tar.gz: {exc}", flush=True)
    if umap_df is None or not {"UMAP-1", "UMAP-2"} <= set(umap_df.columns):
        print("  Note: no UMAP projection found — UMAP coordinates will be empty.", flush=True)
        umap_df = None
    if not cluster_methods:
        print("  Note: no clustering results found — no cluster sets available.", flush=True)

    with h5py.File(os.path.join(data_dir, "cell_feature_matrix.h5"), "r") as f:
        barcodes   = [b.decode() for b in f["matrix/barcodes"][:]]
        gene_names = [g.decode() for g in f["matrix/features/name"][:]]
        ftypes     = ([t.decode() for t in f["matrix/features/feature_type"][:]]
                      if "feature_type" in f["matrix/features"] else None)
        shape      = tuple(f["matrix/shape"][:])
        expr = sp.csc_matrix(
            (f["matrix/data"][:], f["matrix/indices"][:], f["matrix/indptr"][:]),
            shape=shape,
        ).T.tocsr()   # → (cells × genes)

    # Keep only real genes: control probes/codewords, unassigned/deprecated codewords
    # and genomic controls are not genes (thousands of them on Xenium Prime 5K).
    if ftypes is not None:
        _keep = np.array([t == "Gene Expression" for t in ftypes], dtype=bool)
    else:
        _ctrl = ("NegControl", "BLANK_", "UnassignedCodeword", "DeprecatedCodeword",
                 "Intergenic_Region", "GenomicControl")
        _keep = np.array([not g.startswith(_ctrl) for g in gene_names], dtype=bool)
    if _keep.any() and not _keep.all():
        print(f"  Dropping {int((~_keep).sum())} non-gene features "
              f"(controls / unassigned codewords); keeping {int(_keep.sum())} genes", flush=True)
        expr = expr[:, np.flatnonzero(_keep)].tocsr()
        gene_names = [g for g, k in zip(gene_names, _keep) if k]

    print("  Building obs DataFrame…", flush=True)
    df = cells.copy()
    if umap_df is not None:
        df["umap_1"] = umap_df["UMAP-1"].reindex(df.index)
        df["umap_2"] = umap_df["UMAP-2"].reindex(df.index)
    else:
        df["umap_1"] = np.nan
        df["umap_2"] = np.nan
    for method, series in cluster_methods.items():
        df[f"clust__{method}"] = series.reindex(df.index).fillna(0).astype(int)

    # Align expression matrix rows to match df.index order
    n_cells, n_genes = len(df), len(gene_names)
    expr_aligned = _align_matrix_to_cells(expr, barcodes, df.index)

    # Build AnnData
    var_df  = pd.DataFrame({"is_imputed": False}, index=pd.Index(gene_names, name="gene"))
    obs_df  = df.copy()
    obs_df.index = obs_df.index.astype(str)
    adata   = ad.AnnData(X=sp.csr_matrix(expr_aligned, dtype=np.float32), obs=obs_df, var=var_df)
    adata.uns["metadata"]        = metadata
    adata.uns["cluster_methods"] = list(cluster_methods.keys())
    adata.uns["imputed_genes"]   = []
    try:
        adata = TableModel.parse(adata)
    except Exception:
        pass

    # Build boundary GeoDataFrames
    print("  Building cell boundary shapes…", flush=True)
    _empty_bounds = pd.DataFrame(columns=["cell_id", "vertex_x", "vertex_y"])
    cb_path = os.path.join(data_dir, "cell_boundaries.parquet")
    nb_path = os.path.join(data_dir, "nucleus_boundaries.parquet")
    cb_df = pd.read_parquet(cb_path) if os.path.exists(cb_path) else _empty_bounds
    nb_df = pd.read_parquet(nb_path) if os.path.exists(nb_path) else _empty_bounds

    def _bounds_to_gdf(bounds_dict):
        geoms = {}
        try:
            # Vectorised: one shapely call for all polygons (rings need >= 3 vertices)
            import shapely
            # NaN-separated entries (several rings per cell) → one item per ring
            _items = []
            for cid, (vx, vy) in bounds_dict.items():
                vx = np.asarray(vx, dtype=float)
                if np.isnan(vx).any():
                    _cut = np.flatnonzero(np.isnan(vx))
                    _items.extend((cid, rx, ry) for rx, ry in
                                  zip(np.split(vx, _cut), np.split(np.asarray(vy, dtype=float), _cut)))
                else:
                    _items.append((cid, vx, vy))
            # (pieces after the first start with the NaN separator: drop it)
            _items = [(cid, vx[np.isfinite(vx)], np.asarray(vy)[np.isfinite(vx)])
                      for cid, vx, vy in _items]
            _items = [(cid, vx, vy) for cid, vx, vy in _items if len(vx) >= 3]
            _n     = np.array([len(vx) for _, vx, _ in _items])
            _xy    = np.column_stack([np.concatenate([vx for _, vx, _ in _items]),
                                      np.concatenate([vy for _, _, vy in _items])]).astype(float)
            _polys = shapely.polygons(shapely.linearrings(_xy, indices=np.repeat(np.arange(len(_items)), _n)))
            _multi: dict = {}
            for (cid, _, _), g in zip(_items, _polys):
                if str(cid) in geoms:               # further ring of the same cell
                    _multi.setdefault(str(cid), [geoms[str(cid)]]).append(g)
                else:
                    geoms[str(cid)] = g
            for cid, _parts in _multi.items():
                geoms[cid] = shapely.multipolygons(_parts)
        except Exception:
            geoms = {}
            for cid, (vx, vy) in bounds_dict.items():
                try:
                    _ok = np.isfinite(np.asarray(vx, dtype=float))   # first ring only if several
                    _end = len(_ok) if _ok.all() else int(np.argmin(_ok))
                    geoms[str(cid)] = Polygon(zip(vx[:_end], vy[:_end]))
                except Exception:
                    pass
        gdf = gpd.GeoDataFrame(
            geometry=list(geoms.values()),
            index=pd.Index(list(geoms.keys()), name="cell_id"),
        )
        try:
            return ShapesModel.parse(gdf)
        except Exception:
            return gdf

    shapes = {}
    for key, bdf in (("cell_boundaries", cb_df), ("nucleus_boundaries", nb_df)):
        d = _build_boundary_dict(bdf)
        if d:
            shapes[key] = _bounds_to_gdf(d)

    sdata = spatialdata.SpatialData(
        tables={"table": adata},
        shapes=shapes,
    )

    zarr_path = os.path.join(data_dir, "spatialdata_xenium.zarr")
    print(f"  Writing {os.path.basename(zarr_path)} ({n_cells:,} cells, {n_genes} genes)…", flush=True)
    # Write to a temporary store and rename, so an interrupted write is never
    # mistaken for a complete cache on the next start.
    import shutil
    tmp_path = os.path.join(data_dir, "spatialdata_xenium.tmp.zarr")
    shutil.rmtree(tmp_path, ignore_errors=True)
    sdata.write(tmp_path)
    os.replace(tmp_path, zarr_path)
    print("  spatialdata_xenium.zarr generation complete.", flush=True)


_IMPUTED_SUFFIX_RE = re.compile(r" \[(corr\+imp|imp)\]$")


def _plain_imputed_var_names(adata) -> int:
    """Stores written by older versions of functions/impute.py name imputed genes
    'GENE [imp]' / 'GENE [corr+imp]'. The app's convention is the plain gene name plus
    var['is_imputed'] (the suffix is only a dropdown label), so such names are renamed
    in the in-memory table: plain name, is_imputed=True, and the '[corr+imp]' genes
    listed in uns['split_corrected_imputed_genes']. A suffixed name whose plain form
    is already a column is left as it is. Returns the number of renamed genes."""
    names = [str(g) for g in adata.var_names]
    if not any(n.endswith("imp]") for n in names):
        return 0
    taken = {n for n in names if not _IMPUTED_SUFFIX_RE.search(n)}
    new_names, renamed, corr = [], [], []
    for n in names:
        m = _IMPUTED_SUFFIX_RE.search(n)
        base = n[:m.start()] if m else None
        if base and base not in taken:
            taken.add(base)
            renamed.append(base)
            if m.group(1) == "corr+imp":
                corr.append(base)
            n = base
        new_names.append(n)
    if not renamed:
        return 0
    _iname = adata.var.index.name
    adata.var_names = new_names
    adata.var.index.name = _iname
    if "is_imputed" not in adata.var.columns:
        adata.var["is_imputed"] = False
    adata.var["is_imputed"] = adata.var["is_imputed"].fillna(False).astype(bool)
    adata.var.loc[renamed, "is_imputed"] = True
    _strip = lambda g: _IMPUTED_SUFFIX_RE.sub("", str(g))
    _corr_all = list(dict.fromkeys(
        [_strip(g) for g in adata.uns.get("split_corrected_imputed_genes", [])] + corr))
    _raw_all = [g for g in dict.fromkeys(
        [_strip(g) for g in adata.uns.get("imputed_genes", [])]
        + [g for g in renamed if g not in corr]) if g not in _corr_all]
    adata.uns["split_corrected_imputed_genes"] = _corr_all
    adata.uns["imputed_genes"] = _raw_all
    print(f"  Imputed genes: {len(renamed)} var name(s) with an ' [imp]' / ' [corr+imp]' suffix "
          f"(older functions/impute.py output) are read under their plain names", flush=True)
    return len(renamed)


def _load_from_xenium_sdata(zarr_path: str, data_dir: str) -> dict:
    """Load DATA dict from a spatialdata_xenium.zarr store."""
    try:
        import spatialdata
    except ImportError:
        raise ImportError("spatialdata is required: pip install spatialdata")

    print(f"  Loading from spatialdata_xenium.zarr…", flush=True)
    sdata = spatialdata.read_zarr(zarr_path)
    adata = sdata.tables["table"]
    _plain_imputed_var_names(adata)

    gene_names       = list(adata.var_names)
    gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
    barcodes         = list(adata.obs_names)
    n                = len(adata)

    # obs order == expr row order (guaranteed by _build_xenium_sdata write order)
    df_to_expr = np.arange(n, dtype=np.int64)

    cell_bounds = {}
    nuc_bounds  = {}
    try:
        cell_bounds = _gdf_to_bounds_dict(sdata.shapes["cell_boundaries"])
    except Exception as e:
        print(f"  Warning: could not load cell boundaries from zarr: {e}", flush=True)
    try:
        nuc_bounds = _gdf_to_bounds_dict(sdata.shapes["nucleus_boundaries"])
    except Exception as e:
        print(f"  Warning: could not load nucleus boundaries from zarr: {e}", flush=True)

    obs_df = adata.obs.copy()
    # obs columns added later through the low-level zarr API (SPLIT clusters /
    # UMAP) are not in the table's column-order, so read_zarr does not return them.
    reseg._load_zarr_obs_into_df(zarr_path, obs_df)
    for _nacol in [c for c in obs_df.columns
                   if c == "cluster_split_10" or c == "niche" or str(c).startswith("roi_")]:
        # stored as plain strings with "" for cells without a value (dropped by
        # SPLIT, outside every ROI, outside the subset the niches were computed on)
        _cs = obs_df[_nacol].astype(str)
        if _nacol == "cluster_split_10" or _cs.isin(["", "nan", "<NA>"]).any():
            obs_df[_nacol] = pd.array(
                [v if v not in ("", "nan", "<NA>") else pd.NA for v in _cs], dtype="string")
    cluster_methods = list(adata.uns.get("cluster_methods", []))
    print(f"  Done: {n:,} cells | {len(gene_names)} genes | {len(cluster_methods)} cluster sets", flush=True)

    return {
        "metadata":          dict(adata.uns.get("metadata", {})),
        "df":                obs_df,
        "gene_names":        gene_names,
        "gene_name_to_idx":  gene_name_to_idx,
        "barcodes":          barcodes,
        "expr":              sp.csr_matrix(adata.X),
        "df_to_expr":        df_to_expr,
        "cluster_methods":   cluster_methods,
        "cell_bounds":       cell_bounds,
        "nucleus_bounds":    nuc_bounds,
        "data_dir":          data_dir,
        "sdata_path":        zarr_path,
        "gene_var":          adata.var,
        "split_corrected_expr": (
            sp.csr_matrix(adata.layers["X_corrected"])
            if "X_corrected" in adata.layers else None
        ),
        "split_corrected_imputed_genes": list(adata.uns.get("split_corrected_imputed_genes", [])),
    }


def load_from_sdata_zarr(zarr_path: str) -> dict:
    """Load DATA dict from an exported SpatialData zarr (e.g. from _save_sdata_to_disk).
    Sets data_dir=None as a sentinel meaning no OME-TIFF image data is available."""
    zarr_path = os.path.abspath(zarr_path)
    data = _load_from_xenium_sdata(zarr_path, data_dir=None)
    # Ensure umap columns exist so make_umap_fig doesn't crash
    for col in ("umap_1", "umap_2"):
        if col not in data["df"].columns:
            data["df"][col] = np.nan
    data["sdata_path"] = zarr_path
    return data
