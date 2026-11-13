"""Resegmentation: transcript regions, patches, Baysor and Proseg runs, cached-run loading."""

import os
import re
import json
import threading
import subprocess
import hashlib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import core, jobs, caching, dataview, clustering, split, sdata_store, regions


# Control / non-gene feature-name prefixes (used when transcripts.parquet has no is_gene column)
_TX_CONTROL_RE = r"^(NegControl|BLANK_|UnassignedCodeword|DeprecatedCodeword|Intergenic_Region|GenomicControl)"


def _read_tx_region(path, columns=None, x0=None, x1=None, y0=None, y1=None,
                    extra_filters=None, read_dictionary=None):
    """Read transcripts.parquet restricted to a bounding box (inclusive) with predicate
    pushdown, so memory scales with the region instead of the whole run."""
    import pyarrow.parquet as _pq
    flt = list(extra_filters or [])
    if x0 is not None: flt.append(("x_location", ">=", float(x0)))
    if x1 is not None: flt.append(("x_location", "<=", float(x1)))
    if y0 is not None: flt.append(("y_location", ">=", float(y0)))
    if y1 is not None: flt.append(("y_location", "<=", float(y1)))
    return _pq.read_table(path, columns=columns, filters=flt or None,
                          read_dictionary=read_dictionary).to_pandas()


_TX_NON_GENE_CATEGORIES = ["unassigned_codeword", "negative_control_codeword",
                           "negative_control_probe", "genomic_control_probe",
                           "deprecated_codeword"]


def _tx_gene_filters(schema_names, qv_min=20) -> list:
    """Row filters (pyarrow tuples) selecting the transcripts the Xenium cell-feature
    matrix counts: Q-score ≥ qv_min and gene features only. Bundles without `is_gene`
    additionally need _tx_drop_control_features() on the rows read."""
    flt = []
    if "qv" in schema_names and qv_min is not None:
        flt.append(("qv", ">=", float(qv_min)))
    if "is_gene" in schema_names:
        flt.append(("is_gene", "==", True))
    elif "codeword_category" in schema_names:
        flt.append(("codeword_category", "not in", list(_TX_NON_GENE_CATEGORIES)))
    return flt


def _tx_control_feature_mask(names) -> np.ndarray:
    """True for feature names that are control / non-gene features (by name prefix)."""
    names = pd.Series(names)
    if isinstance(names.dtype, pd.CategoricalDtype):
        cats = pd.Series(names.cat.categories)
        bad = _tx_control_feature_mask(cats)
        codes = names.cat.codes.to_numpy()
        return np.append(bad, False)[codes]          # code -1 (null) → trailing False
    names = names.map(lambda v: v.decode() if isinstance(v, bytes) else v).astype(str)
    return names.str.match(_TX_CONTROL_RE).to_numpy(dtype=bool)


def _tx_drop_control_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Drop control / non-gene rows by feature-name prefix (bundles without `is_gene`)."""
    if "feature_name" not in tx.columns or not len(tx):
        return tx
    bad = _tx_control_feature_mask(tx["feature_name"])
    return tx[~bad].copy() if bad.any() else tx


def _tx_write_gene_only(src: str, dst: str, qv_min=20) -> int:
    """Stream transcripts.parquet into *dst* keeping Q ≥ qv_min gene transcripts only
    (all columns kept; memory stays at one record batch). Returns the rows written."""
    import pyarrow as _pa
    import pyarrow.dataset as _ds
    import pyarrow.parquet as _pq
    dataset = _ds.dataset(src, format="parquet")
    schema = dataset.schema
    flt = _tx_gene_filters(schema.names, qv_min)
    expr = _pq.filters_to_expression(flt) if flt else None
    by_name = "is_gene" not in schema.names and "feature_name" in schema.names
    n = 0
    tmp = dst + ".tmp"
    with _pq.ParquetWriter(tmp, schema) as writer:
        for batch in dataset.scanner(filter=expr).to_batches():
            if batch.num_rows and by_name:
                bad = _tx_control_feature_mask(batch.column("feature_name").to_pandas())
                if bad.any():
                    batch = batch.filter(_pa.array(~bad))
            if batch.num_rows:
                writer.write_batch(batch)
                n += batch.num_rows
    os.replace(tmp, dst)
    return n


def _proseg_prior_type(prior_type) -> str:
    """Proseg prior the app supports: "nucleus" (default) or "cell". Anything else -
    e.g. the removed "none" option still held in a browser's saved state - is the
    default: proseg 1.x cannot run without an initial segmentation."""
    return prior_type if prior_type in ("nucleus", "cell") else "nucleus"


def _patch_owned_mask(x, y, pi, bounds, ok_idx):
    """True for centroids that patch `pi` keeps: a centroid lying in several patches that
    actually returned cells (`ok_idx`) belongs to the one it sits deepest inside (ties →
    lower patch index). On a regular grid this splits the overlap strip at its midline;
    nothing is ceded to skipped or failed patches."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x0, y0, x1, y1 = bounds[pi]
    depth = np.minimum(np.minimum(x - x0, x1 - x), np.minimum(y - y0, y1 - y))
    keep = np.ones(len(x), dtype=bool)
    for j in ok_idx:
        if j == pi:
            continue
        a0, b0, a1, b1 = bounds[j]
        inside = (x >= a0) & (x < a1) & (y >= b0) & (y < b1)
        if not inside.any():
            continue
        dj = np.minimum(np.minimum(x - a0, a1 - x), np.minimum(y - b0, b1 - y))
        keep &= ~(inside & ((dj > depth) | ((dj == depth) & (j < pi))))
    return keep


# Seam de-duplication of patch runs (stored in params.json as "seam_dedup" so a cache
# load repeats exactly what the live run did; runs without the key were not de-duplicated).
_SEAM_DEDUP = {"version": 1, "min_overlap": 0.5, "max_centroid_dist": 1.5, "margin": 25.0}


def _patch_seam_dedup_mask(ids, patch_idx, x, y, n_tx, polys, bounds, ok_idx, cfg=None):
    """Second step of the patch merge, after centroid ownership (_patch_owned_mask).

    Two neighbouring patches segment the shared strip independently, so the same
    biological cell can be kept from both when its two centroids fall on either side
    of the ownership line. Among the kept cells near such a line, two cells of
    different patches are one cell when their polygons overlap by more than
    cfg["min_overlap"] of the smaller polygon's area, or their centroids are closer
    than cfg["max_centroid_dist"] µm (the only test for cells without a polygon).
    Of such a pair the cell lying deeper inside its own patch is kept (ties: more
    transcripts, then lower patch index).

    ids / patch_idx / x / y / n_tx describe the cells kept by ownership; polys maps
    id → (xs, ys). Returns a bool mask (True = keep). The result depends only on the
    cells' values, not on their order, so the live run and a cache load agree."""
    cfg = cfg or _SEAM_DEDUP
    patch_idx = np.asarray(patch_idx, dtype=np.int64)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    keep = np.ones(n, dtype=bool)
    ok_idx = [int(j) for j in ok_idx]
    if n < 2 or len(ok_idx) < 2:
        return keep
    n_tx = np.nan_to_num(np.asarray(n_tx, dtype=float)) if n_tx is not None else np.zeros(n)
    b = np.asarray([tuple(float(v) for v in bb) for bb in bounds], dtype=float)
    own = b[patch_idx]
    depth = np.minimum(np.minimum(x - own[:, 0], own[:, 2] - x),
                       np.minimum(y - own[:, 1], own[:, 3] - y))
    # Only cells near an ownership line can have a twin in another patch: within
    # `margin` of patch j's box and at most 2·margin deeper in their own patch than in j
    margin = float(cfg["margin"])
    near = np.zeros(n, dtype=bool)
    for j in ok_idx:
        a0, b0, a1, b1 = b[j]
        dj = np.minimum(np.minimum(x - a0, a1 - x), np.minimum(y - b0, b1 - y))
        near |= (patch_idx != j) & (dj > -margin) & (depth - dj <= 2.0 * margin)
    cand = np.where(near)[0]
    if len(cand) < 2 or len(np.unique(patch_idx[cand])) < 2:
        return keep

    pairs = set()
    from scipy.spatial import cKDTree
    _cp = cKDTree(np.column_stack([x[cand], y[cand]])).query_pairs(
        float(cfg["max_centroid_dist"]), output_type="ndarray")
    for i, j in _cp:
        if patch_idx[cand[i]] != patch_idx[cand[j]]:
            pairs.add((int(min(i, j)), int(max(i, j))))
    try:
        import shapely
        geoms, gpos = [], []
        for k, ci in enumerate(cand):
            ring = polys.get(ids[ci]) if polys else None
            if ring is None or len(ring[0]) < 3:
                continue
            # float32 like the live run's arrays, so files re-read as float64 agree
            g = shapely.Polygon(np.column_stack([np.asarray(ring[0], dtype=np.float32),
                                                 np.asarray(ring[1], dtype=np.float32)]
                                                ).astype(float))
            if not g.is_valid:
                g = g.buffer(0)
            if g.is_empty or not g.area > 0:
                continue
            geoms.append(g)
            gpos.append(k)
        if len(geoms) > 1:
            geoms = np.asarray(geoms, dtype=object)
            gpos = np.asarray(gpos)
            qi, ti = shapely.STRtree(geoms).query(geoms, predicate="intersects")
            sel = (qi < ti) & (patch_idx[cand[gpos[qi]]] != patch_idx[cand[gpos[ti]]])
            qi, ti = qi[sel], ti[sel]
            if len(qi):
                inter = shapely.area(shapely.intersection(geoms[qi], geoms[ti]))
                small = np.minimum(shapely.area(geoms[qi]), shapely.area(geoms[ti]))
                for i, j in zip(gpos[qi[inter > float(cfg["min_overlap"]) * small]],
                                gpos[ti[inter > float(cfg["min_overlap"]) * small]]):
                    pairs.add((int(min(i, j)), int(max(i, j))))
    except ImportError:
        print("  Seam de-duplication: shapely not available — centroid test only", flush=True)
    if not pairs:
        return keep

    # Strict, order-independent priority; best cell first, a cell is dropped when a
    # twin of it has already been kept
    involved = np.array(sorted({k for p in pairs for k in p}), dtype=np.int64)
    ci = cand[involved]
    order = np.lexsort((y[ci], x[ci], patch_idx[ci], -n_tx[ci], -depth[ci]))
    twins: dict = {}
    for i, j in pairs:
        twins.setdefault(i, []).append(j)
        twins.setdefault(j, []).append(i)
    kept = set()
    for k in involved[order]:
        if any(t in kept for t in twins[int(k)]):
            keep[cand[k]] = False
        else:
            kept.add(int(k))
    return keep


def _patch_bounds_tag(bounds) -> str:
    """Short hash of a patch layout, for the reseg cache-dir tag."""
    return hashlib.md5(repr([tuple(round(float(v), 1) for v in b)
                             for b in bounds]).encode()).hexdigest()[:8]


def _get_patch_bounds_um():
    """Return list of (x0, y0, x1, y1) in µm for each sopa patch, or None if none available."""
    with core._sdata_lock:
        patches = core._sdata_state.get("patches")
    if patches is None:
        return None
    try:
        patches_um = sdata_store._sdata_transform_to_um(patches)
        return [geom.bounds for geom in patches_um.geometry]  # (minx, miny, maxx, maxy)
    except Exception as exc:
        print(f"  _get_patch_bounds_um: {exc}", flush=True)
        return None


def _baysor_progress_parser(progress_fn=None):
    """on_line callback for _job_run_process: turns Baysor 0.7 output into
    progress_fn(fraction 0..1, text). Baysor logs "[hh:mm:ss] Info: <stage>" lines and
    redraws a "Progress:  NN%|…" bar (ProgressMeter) for the segmentation iterations;
    the stages before and after it print no percentage, so they get fixed marks."""
    stage = {"em": False}
    marks = (("Loading data", 0.02, "loading data"),
             ("Estimating noise level", 0.05, "estimating noise level"),
             ("Clustering molecules", 0.08, "clustering molecules"),
             ("Initializing algorithm", 0.10, "segmentation iterations"),
             ("Processing complete", 0.85, "iterations complete"),
             ("Estimating local colors", 0.88, "estimating local colours"),
             ("Estimating boundary polygons", 0.92, "estimating boundary polygons"),
             ("Saving results", 0.95, "saving results"),
             ("All done", 1.0, "done"))

    def _on_line(line):
        if progress_fn is None:
            return False
        m = re.search(r"Progress:\s+(\d+)%", line)
        if m:
            if stage["em"]:
                pct = int(m.group(1))
                progress_fn(0.10 + 0.75 * pct / 100.0, f"segmentation iterations {pct}%")
            return False
        for key, frac, text in marks:
            if key in line:
                stage["em"] = key == "Initializing algorithm"
                progress_fn(frac, text)
                break
        return False

    return _on_line


# Share of a Baysor/Proseg job's progress bar given to the tool itself; the rest is
# the expression matrix, clustering / UMAP and the SpatialData store.
_TOOL_SHARE = 0.9

_PROSEG_BAR_RE = re.compile(r"^\s*\d+:\d\d:\d\d\s+([#\-]{8,})\s*\|")


def _proseg_progress_parser(progress_fn=None):
    """on_line callback for _job_run_process: Proseg 1.x draws an indicatif bar
    "hh:mm:ss ####------ | log-likelihood: … | assigned: …" for its sampler iterations,
    on a terminal only (hence use_pty). The filled share of the bar is the fraction of
    iterations done; bar redraws are kept out of the server log."""
    def _on_line(line):
        m = _PROSEG_BAR_RE.match(line)
        if not m:
            return False
        if progress_fn is not None:
            bar = m.group(1)
            frac = bar.count("#") / len(bar)
            progress_fn(frac, f"sampling {round(100 * frac)}%")
        return True

    return _on_line


def _baysor_run_single(tx_df, patch_dir, baysor_bin, scale, min_mol, use_prior, prior_conf,
                       scale_std=None,  n_clusters=10, progress_fn=None):
    """Run Baysor on tx_df, write outputs to patch_dir.
    Returns (cells_df, cell_bounds_dict, seg_df) or raises on failure.
    cells_df is indexed by string cell_id.
    progress_fn(fraction, text): progress of this one Baysor process.
    """
    os.makedirs(patch_dir, exist_ok=True)
    tx_csv = os.path.join(patch_dir, "transcripts.csv")
    tx_df.to_csv(tx_csv, index=False)

    n_assigned = int((tx_df["cell_id"] != 0).sum())
    _use_prior = use_prior and n_assigned > 0
    if use_prior and not _use_prior:
        print(f"  Baysor [{patch_dir}]: no assigned transcripts — disabling prior", flush=True)

    cmd = [baysor_bin, "run"]
    if scale != -1:
        cmd += ["-s", str(scale)]
    if scale_std is not None:
        cmd += ["--scale-std", str(scale_std)]
    cmd += [
        "--min-molecules-per-cell", str(min_mol),
        "--n-clusters", str(n_clusters),
        "--polygon-format", "FeatureCollection",
        "-x", "x_location",
        "-y", "y_location",
        "-g", "feature_name",
        "-o", patch_dir,
        tx_csv,
    ]
    _prior_col = "nucleus_prior_id" if "nucleus_prior_id" in tx_df.columns else "cell_id"
    if _use_prior:
        cmd += ["--prior-segmentation-confidence", str(prior_conf), f":{_prior_col}"]

    returncode = jobs._job_run_process(cmd, cwd=patch_dir, log_tag="baysor",
                                  on_line=_baysor_progress_parser(progress_fn))
    if returncode != 0:
        raise RuntimeError(f"Baysor exited with code {returncode}")

    seg_csv = os.path.join(patch_dir, "segmentation.csv")
    if not os.path.exists(seg_csv):
        raise RuntimeError(f"segmentation.csv not found in {patch_dir}")

    seg = pd.read_csv(seg_csv, usecols=lambda c: c in ("cell", "gene", "x", "y",
                                                       "x_location", "y_location"))
    seg = seg[seg["cell"].notna()].copy()

    x_col = "x" if "x" in seg.columns else "x_location"
    y_col = "y" if "y" in seg.columns else "y_location"
    cells_df = (
        seg.groupby("cell")
        .agg(x_centroid=(x_col, "mean"), y_centroid=(y_col, "mean"),
             transcript_counts=(x_col, "count"))
        .rename_axis("cell_id")
    )

    cell_bounds: dict = {}
    for poly_name in ("segmentation_polygons_2d.json", "segmentation_polygons.json",
                      "polygons.json", "cell_polygons.json"):
        poly_path = os.path.join(patch_dir, poly_name)
        if os.path.exists(poly_path):
            break
        poly_path = None
    if poly_path:
        with open(poly_path) as f:
            gj = json.load(f)
        for feat in gj.get("features", []):
            props = feat.get("properties") or {}
            cid   = feat.get("id") or props.get("cell") or props.get("id")
            geom  = feat.get("geometry", {})
            gtype = geom.get("type")
            if cid is None:
                continue
            if gtype == "Polygon":
                ring = geom["coordinates"][0]
            elif gtype == "MultiPolygon":
                ring = _largest_ring([poly[0] for poly in geom["coordinates"]])
            else:
                continue
            vx = np.array([p[0] for p in ring], dtype=np.float32)
            vy = np.array([p[1] for p in ring], dtype=np.float32)
            cell_bounds[cid] = (vx, vy)

    return cells_df, cell_bounds, seg


def _proseg_run_single(tx_df, patch_dir, proseg_bin, voxel_size, n_threads, n_samples=None,
                       recorded_samples=None, schedule=None,
                       nuclear_reassign_prob=None, prior_seg_prob=None,
                       prior_type: str = "nucleus", progress_fn=None):
    """Run Proseg on tx_df (written as parquet), write outputs to patch_dir.
    tx_df may also be the path of an existing transcripts.parquet to use as is.
    Returns (cells_df, cell_bounds_dict, tx_meta_df) or raises on failure.
    progress_fn(fraction, text): progress of this one Proseg process.
    """
    import gzip
    os.makedirs(patch_dir, exist_ok=True)
    prior_type = _proseg_prior_type(prior_type)

    poly_out    = os.path.join(patch_dir, "cell-polygons.geojson.gz")
    meta_out    = os.path.join(patch_dir, "cell-metadata.csv.gz")
    tx_meta_out = os.path.join(patch_dir, "transcript-metadata.csv.gz")

    if isinstance(tx_df, str):
        tx_path = tx_df
    else:
        tx_path = os.path.join(patch_dir, "transcripts.parquet")
        # Adjust prior based on prior_type before writing parquet
        if prior_type == "cell" and "overlaps_nucleus" in tx_df.columns and "cell_id" in tx_df.columns:
            # Override overlaps_nucleus: treat all cell-assigned transcripts as "in nucleus"
            # so Proseg uses cell boundaries as the prior instead of nuclear boundaries.
            tx_df = tx_df.copy()
            _cid = tx_df["cell_id"]
            # Xenium cell_id is a string ("UNASSIGNED" when not in a cell); old runs use ints
            if pd.api.types.is_numeric_dtype(_cid):
                _in_cell = _cid > 0
            else:
                _in_cell = _cid.notna() & ~_cid.isin(["UNASSIGNED", b"UNASSIGNED", ""])
            tx_df["overlaps_nucleus"] = _in_cell.astype(tx_df["overlaps_nucleus"].dtype)
        tx_df.to_parquet(tx_path, index=False)

    # Always the --xenium preset: proseg 1.x has no mode without an initial segmentation
    # (without the preset it stops on the missing --gene-column / --cell-id-column).
    cmd = [proseg_bin, "--xenium", tx_path,
           "--output-cell-polygons",       poly_out,
           "--output-cell-metadata",       meta_out,
           "--output-transcript-metadata", tx_meta_out]
    # The --xenium preset of proseg 1.x does not drop low-quality transcripts
    # (--min-qv defaults to 0): keep Q-score ≥ 20 like the Xenium matrix / Baysor path
    cmd += ["--min-qv", "20"]
    if voxel_size:
        cmd += ["--initial-voxel-size", str(voxel_size)]
    if n_threads:
        cmd += ["--nthreads", str(n_threads)]
    if n_samples:
        cmd += ["--ncomponents", str(n_samples)]
    if recorded_samples:
        cmd += ["--recorded-samples", str(recorded_samples)]
    if schedule:
        cmd += ["--schedule"] + schedule.split()
    if nuclear_reassign_prob is not None:
        cmd += ["--nuclear-reassignment-prob", str(nuclear_reassign_prob)]
    if prior_seg_prob is not None:
        cmd += ["--prior-seg-reassignment-prob", str(prior_seg_prob)]

    returncode = jobs._job_run_process(cmd, cwd=patch_dir, log_tag="proseg", use_pty=True,
                                  on_line=_proseg_progress_parser(progress_fn))
    if returncode != 0:
        raise RuntimeError(f"Proseg exited with code {returncode}")

    cell_meta = pd.read_csv(meta_out)
    cells_df = cell_meta.rename(columns={
        "cell": "cell_id", "centroid_x": "x_centroid",
        "centroid_y": "y_centroid", "population": "transcript_counts",
        "volume": "cell_area",
    }).set_index("cell_id")
    _keep_cols = ["x_centroid", "y_centroid", "transcript_counts"]
    if "cell_area" in cells_df.columns:
        _keep_cols.append("cell_area")
    cells_df = cells_df[_keep_cols]

    cell_bounds: dict = {}
    if os.path.exists(poly_out):
        with gzip.open(poly_out, "rt") as f:
            gj = json.load(f)
        for feat in gj.get("features", []):
            props = feat.get("properties", {})
            cid   = props.get("cell") if props.get("cell") is not None else props.get("id")
            geom  = feat.get("geometry", {})
            gtype = geom.get("type")
            if cid is None:
                continue
            if gtype == "Polygon":
                rings = [geom["coordinates"][0]]
            elif gtype == "MultiPolygon":
                rings = [poly[0] for poly in geom["coordinates"]]
            else:
                continue
            ring = _largest_ring(rings)
            vx = np.array([p[0] for p in ring], dtype=np.float32)
            vy = np.array([p[1] for p in ring], dtype=np.float32)
            cell_bounds[cid] = (vx, vy)

    tx_meta_df = None
    if os.path.exists(tx_meta_out):
        try:
            tx_meta_df = _read_proseg_tx_meta(tx_meta_out)
        except Exception:
            pass

    return cells_df, cell_bounds, tx_meta_df


def _largest_ring(rings):
    """Exterior ring enclosing the largest area (shoelace formula) among the parts of
    a multi-part cell. The ring with the most vertices is not always the largest part."""
    rings = [r for r in rings if r is not None and len(r)]
    if not rings:
        return []
    if len(rings) == 1:
        return rings[0]

    def _area(r):
        a = np.asarray(r, dtype=np.float64)
        if a.ndim != 2 or a.shape[0] < 3:
            return 0.0
        x, y = a[:, 0], a[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    return max(rings, key=lambda r: (_area(r), len(r)))


def _read_proseg_tx_meta(path):
    """Proseg transcript-metadata, reduced to the gene and cell-assignment columns."""
    return pd.read_csv(path, usecols=lambda c: c in ("gene", "cell", "assignment"),
                       dtype={"gene": "category"})


def _expr_gene_cols(genes, gene_to_idx):
    """Column index per transcript gene (-1 when the gene is not in the app's gene list)."""
    gcat = genes.astype("category")
    lut = np.array([gene_to_idx.get(c, -1) for c in gcat.cat.categories] + [-1], dtype=np.int64)
    return lut[gcat.cat.codes.to_numpy()]      # code -1 (missing gene) → trailing -1


def _build_baysor_expr(seg_df, cells_df):
    """Build cell×gene CSR expression matrix from merged Baysor segmentation DataFrame."""
    # measured panel genes only (DATA["gene_names"] may also list SpaGE-imputed genes)
    gene_names_list = dataview._panel_gene_names()
    gene_to_idx = {g: i for i, g in enumerate(gene_names_list)}
    if "gene" not in seg_df.columns:
        return None
    # seg_df is per transcript, or already reduced to per-(cell, gene) counts in column "n"
    _cidx = pd.Index(cells_df.index)
    if _cidx.is_unique:
        rows = _cidx.get_indexer(seg_df["cell"])
    else:
        cell_to_row = {c: i for i, c in enumerate(cells_df.index)}
        rows = seg_df["cell"].map(cell_to_row).fillna(-1).astype("int64").to_numpy()
    cols = _expr_gene_cols(seg_df["gene"], gene_to_idx)
    ok = (rows >= 0) & (cols >= 0)
    vals = (seg_df["n"].to_numpy()[ok].astype(np.float32) if "n" in seg_df.columns
            else np.ones(int(ok.sum()), dtype=np.float32))
    mat = sp.csr_matrix(
        (vals, (rows[ok], cols[ok])),
        shape=(len(cells_df), len(gene_names_list))
    )
    print(f"  Baysor: built expression matrix {mat.shape}", flush=True)
    return mat


def _build_proseg_expr(tx_meta_df, cells_df):
    """Build cell×gene CSR expression matrix from merged Proseg transcript-metadata."""
    if tx_meta_df is None:
        return None
    # measured panel genes only (DATA["gene_names"] may also list SpaGE-imputed genes)
    gene_names_list = dataview._panel_gene_names()
    gene_to_idx = {g: i for i, g in enumerate(gene_names_list)}
    gene_col = "gene" if "gene" in tx_meta_df.columns else None
    cell_col = ("cell" if "cell" in tx_meta_df.columns
                else "assignment" if "assignment" in tx_meta_df.columns else None)
    if not gene_col or not cell_col:
        return None
    # tx_meta_df is per transcript, or already reduced to per-(cell, gene) counts in column "n".
    # Ids are matched as integers (no per-row string conversion).
    ids = pd.to_numeric(tx_meta_df[cell_col], errors="coerce")
    keep = ids.notna().to_numpy()
    if cell_col == "assignment":
        # Filter out proseg's "unassigned" sentinel value (0xFFFFFFFF)
        keep &= (ids != 4294967295).to_numpy()
    ids = ids[keep].astype("int64")
    _cidx = pd.Index(pd.to_numeric(cells_df.index, errors="coerce"))
    if _cidx.hasnans or not _cidx.is_unique:
        # non-integer or repeated cell ids: match as strings
        cell_to_row = {str(c): i for i, c in enumerate(cells_df.index)}
        rows = ids.astype(str).map(cell_to_row).fillna(-1).astype("int64").to_numpy()
    else:
        rows = _cidx.get_indexer(ids)
    cols = _expr_gene_cols(tx_meta_df[gene_col][keep], gene_to_idx)
    ok = (rows >= 0) & (cols >= 0)
    vals = (tx_meta_df["n"].to_numpy()[keep][ok].astype(np.float32) if "n" in tx_meta_df.columns
            else np.ones(int(ok.sum()), dtype=np.float32))
    mat = sp.csr_matrix(
        (vals, (rows[ok], cols[ok])),
        shape=(len(cells_df), len(gene_names_list))
    )
    print(f"  Proseg: built expression matrix {mat.shape}", flush=True)
    return mat


def _proseg_counts_from_expr(cells_df, expr_mat, imputed_cols=None) -> bool:
    """Set cells_df['transcript_counts'] of a Proseg result to the row sums of its
    expression matrix (in place). Proseg's own `population` column is the final sampler
    state, while the matrix is built from the consensus `assignment` column, so the two
    differ by a transcript or two per cell; the matrix is what every count shown in the
    app comes from. *imputed_cols*: column indices holding SpaGE-imputed values (not
    counts), left out of the sum. Returns True when a value changed."""
    if expr_mat is None or expr_mat.shape[0] != len(cells_df):
        return False
    _m = expr_mat
    if imputed_cols is not None and len(imputed_cols):
        _keep = np.ones(_m.shape[1], dtype=bool)
        _keep[np.asarray(imputed_cols, dtype=np.int64)] = False
        _m = sp.csr_matrix(_m)[:, np.where(_keep)[0]]
    _tc = np.rint(np.asarray(_m.sum(axis=1), dtype=np.float64).ravel()).astype(np.int64)
    _old = (pd.to_numeric(cells_df["transcript_counts"], errors="coerce").to_numpy(dtype=float)
            if "transcript_counts" in cells_df.columns else None)
    _changed = _old is None or not np.array_equal(_old, _tc.astype(float))
    cells_df["transcript_counts"] = _tc
    return bool(_changed)


# Proseg's own cell-metadata columns that earlier cache rebuilds copied into the run's
# zarr obs. `cluster` is Proseg's internal component id, not the app's KMeans cluster.
_RESEG_FOREIGN_OBS = frozenset({"cluster", "fov", "centroid_z"})


def _reseg_cluster_cols(columns) -> list:
    """Cluster columns the app itself computed for a Baysor/Proseg run
    (cluster_kmeans_*, cluster_split_*), in column order; a bare `cluster` is not one."""
    return [c for c in columns if str(c).startswith("cluster_")]


def _reseg_display_cluster_col(columns):
    """Column used for "Color by Cluster" / the cluster filter on a reseg run: the
    original-count clusters first, the SPLIT clusters only when nothing else exists."""
    _cc = _reseg_cluster_cols(columns)
    return next((c for c in _cc if "split" not in str(c)), _cc[0] if _cc else None)


def _load_zarr_obs_into_df(zarr_path: str, cells_df: pd.DataFrame) -> None:
    """Read obs columns from a SpatialData zarr and merge any missing ones into cells_df in-place."""
    try:
        import zarr as _zarr_mod
        _zgrp = _zarr_mod.open_group(zarr_path, mode="r", use_consolidated=False)
        _obs = _zgrp["tables"]["table"]["obs"]
        _idx_key = _obs.attrs.get("_index", "_index")
        _zarr_idx = [str(v) for v in _obs[_idx_key][:]] if _idx_key in _obs else None
        # Strip patch prefix (e.g. "p0_") from zarr IDs if cells_df lacks it.
        # This happens when the zarr was built during a live patch run but the CSV
        # stores raw Baysor IDs without the patch prefix.
        if _zarr_idx is not None and len(_zarr_idx) > 0:
            import re as _re
            _stripped = [_re.sub(r'^p\d+_', '', _id) for _id in _zarr_idx]
            # Compare the whole index: the two frames can hold the same cells in a
            # different order, so a leading sample is not representative.
            _df_ids  = pd.Index(cells_df.index.astype(str))
            _n_raw   = int(pd.Index(_zarr_idx).isin(_df_ids).sum())
            _n_strip = int(pd.Index(_stripped).isin(_df_ids).sum())
            if max(_n_raw, _n_strip) >= 0.5 * len(_zarr_idx) and _n_strip > _n_raw:
                print(f"  Stripping patch prefix from zarr obs index ({_zarr_idx[0]!r} → {_stripped[0]!r})", flush=True)
                _zarr_idx = _stripped
            elif max(_n_raw, _n_strip) < 0.5 * len(_zarr_idx):
                # Genuinely different run — skip, let caller recompute
                print(f"  Warning: zarr obs index mismatch "
                      f"({_zarr_idx[0]!r} vs {cells_df.index[0]!r}). "
                      f"Will recompute cluster/UMAP.", flush=True)
                return
        for _col in _obs.keys():
            if _col in {"__categories", "_index"} or _col == _idx_key:
                continue
            if _col in _RESEG_FOREIGN_OBS:
                continue
            if _col not in cells_df.columns:
                try:
                    _vals = _obs[_col][:]
                    if _zarr_idx is not None:
                        cells_df[_col] = pd.Series(_vals, index=_zarr_idx).reindex(
                            cells_df.index.astype(str)).values
                    else:
                        cells_df[_col] = _vals
                except Exception:
                    pass
    except Exception as _ze:
        print(f"  Warning: could not load obs from zarr: {_ze}", flush=True)


def _align_zarr_rows_to_cells(mat, zarr_ids, cells_df, tool_label: str):
    """Reorder the rows of a zarr table matrix (one row per zarr obs) into cells_df
    order by cell id; cells missing from the zarr get zero rows. Returns None when
    the two id sets do not match (the caller then has no corrected counts)."""
    if mat is None:
        return None
    _zids  = pd.Index([str(i) for i in zarr_ids])
    _dfids = pd.Index(cells_df.index.astype(str))
    if len(_zids) == len(_dfids) and (_zids == _dfids).all():
        return mat
    _zstrip = _zids.str.replace(r'^p\d+_', '', regex=True)
    _use = _zids if _zids.isin(_dfids).sum() >= _zstrip.isin(_dfids).sum() else _zstrip
    _pos = _use.get_indexer(_dfids) if not _use.has_duplicates else np.full(len(_dfids), -1)
    if len(_dfids) == 0 or mat.shape[0] != len(_zids) or (_pos >= 0).mean() < 0.99:
        print(f"  Warning: {tool_label} X_corrected rows do not match the loaded cells "
              f"— SPLIT-corrected counts not loaded (re-run SPLIT)", flush=True)
        return None
    _ok  = np.where(_pos >= 0)[0]
    _coo = sp.csr_matrix(mat)[_pos[_ok], :].tocoo()
    return sp.csr_matrix((_coo.data, (_ok[_coo.row], _coo.col)),
                         shape=(len(_dfids), mat.shape[1]))


def _load_zarr_boundaries_into_dict(sdata, cell_bounds: dict, tool_label: str) -> None:
    """Fill cell_bounds from zarr shapes element if it's currently empty."""
    if cell_bounds or "cell_boundaries" not in sdata.shapes:
        return
    _gdf = sdata.shapes["cell_boundaries"]
    for _cid, _geom in zip(_gdf.index, _gdf.geometry):
        try:
            _coords = list(_geom.exterior.coords)
            cell_bounds[str(_cid)] = ([c[0] for c in _coords], [c[1] for c in _coords])
        except Exception:
            pass
    if cell_bounds:
        print(f"  Loaded {len(cell_bounds):,} {tool_label} boundaries from zarr shapes", flush=True)


# One lock per SpatialData store for the low-level table writers below (obs
# columns, X_corrected, imputed genes), so two of them never rewrite the same
# table at once. _sdata_lock only guards _sdata_state and is not held for I/O.
_zarr_write_locks: dict = {}
_zarr_write_locks_guard = threading.Lock()


def _zarr_write_lock(path: str):
    with _zarr_write_locks_guard:
        return _zarr_write_locks.setdefault(os.path.abspath(path), threading.RLock())


def _update_reseg_zarr_obs(zarr_path: str, cells_df: pd.DataFrame, force_cols=()) -> None:
    """Update the obs (cell metadata) in a reseg SpatialData zarr with new columns from cells_df.
    Writes directly via zarr low-level API to avoid spatialdata's 'target path in use' error.
    Columns named in force_cols are rewritten even when they already exist."""
    if not os.path.isdir(zarr_path):
        return
    _zlock = _zarr_write_lock(zarr_path)
    _zlock.acquire()
    try:
        import zarr as _zarr_mod
        _zroot = _zarr_mod.open_group(zarr_path, mode="r+", use_consolidated=False)
        if "tables" not in _zroot or "table" not in _zroot["tables"] \
                or "obs" not in _zroot["tables"]["table"]:
            return
        _obs_grp = _zroot["tables"]["table"]["obs"]
        existing_cols = set(_obs_grp.keys()) - {"__categories", "_index"}
        # Always re-write roi_* and split_* columns (may have changed or been fixed)
        new_cols = [c for c in cells_df.columns
                    if c not in existing_cols or c.startswith("roi_")
                    or c.startswith("split_umap_") or c == "cluster_split_10"
                    or c == "cell_area" or c in force_cols]
        if not new_cols:
            return
        # Read the existing index to align values correctly
        _idx_key = _obs_grp.attrs.get("_index", "_index")
        if _idx_key in _obs_grp:
            _existing_idx = [str(v) for v in _obs_grp[_idx_key][:]]
        else:
            _existing_idx = None
        # Detect a zarr that belongs to a different cell set. Compare the whole index
        # (same cells can be stored in another order) and never rewrite the obs index:
        # X, layers and the other obs columns would keep the old rows.
        # First strip any patch prefix (p0_, p1_, ...) from existing index before comparing.
        if _existing_idx is not None and len(_existing_idx) > 0:
            import re as _re2
            _stripped_existing = [_re2.sub(r'^p\d+_', '', _id) for _id in _existing_idx]
            _df_ids = pd.Index(cells_df.index.astype(str))
            _raw_overlap      = int(pd.Index(_existing_idx).isin(_df_ids).sum())
            _stripped_overlap = int(pd.Index(_stripped_existing).isin(_df_ids).sum())
            if max(_raw_overlap, _stripped_overlap) < 0.5 * len(_existing_idx):
                print(f"  Warning: zarr obs index does not match current cells_df "
                      f"({max(_raw_overlap, _stripped_overlap):,}/{len(_existing_idx):,} ids) "
                      f"— not updating {os.path.basename(zarr_path)}", flush=True)
                return
            elif _stripped_overlap > _raw_overlap:
                # Zarr uses p{N}_ patch prefix but cells_df uses bare IDs — strip for reindex
                _existing_idx = _stripped_existing
        try:
            from anndata.io import write_elem as _write_elem
        except ImportError:
            from anndata.experimental import write_elem as _write_elem
        for col in new_cols:
            try:
                _ser = cells_df[col].copy()
                _ser.index = _ser.index.astype(str)
                if _existing_idx is not None:
                    _ser = _ser.reindex(_existing_idx)
                # For string/object columns bypass _write_elem (zarr warns on object dtype)
                # and write directly as fixed-width unicode numpy array.
                if _ser.dtype == object or pd.api.types.is_string_dtype(_ser):
                    _arr = np.array(_ser.fillna("").values, dtype=str)
                    _obs_grp.create_dataset(col, data=_arr, shape=_arr.shape, dtype=_arr.dtype, overwrite=True)
                else:
                    try:
                        _write_elem(_obs_grp, col, _ser)
                    except Exception:
                        _arr = _ser.values
                        _obs_grp.create_dataset(col, data=_arr, shape=_arr.shape, dtype=_arr.dtype, overwrite=True)
            except Exception as _col_err:
                print(f"  Warning: could not write obs col '{col}': {_col_err}", flush=True)
        print(f"  Updated zarr obs: added {new_cols} to {os.path.basename(zarr_path)}", flush=True)
    except Exception as _e:
        print(f"  Warning: failed to update zarr obs ({_e})", flush=True)
    finally:
        _zlock.release()


def _build_reseg_sdata(cells_df, cell_bounds, expr_mat, gene_names, source, out_dir) -> str:
    """Build and write a SpatialData zarr for a resegmentation result. Returns zarr path."""
    try:
        import spatialdata
        from spatialdata.models import TableModel, ShapesModel
        import anndata as ad
        import geopandas as gpd
        from shapely.geometry import Polygon
    except ImportError:
        print("  Warning: spatialdata/geopandas/shapely not available — skipping reseg zarr", flush=True)
        return ""

    if expr_mat is not None and expr_mat.shape[1] < len(gene_names):
        # callers pass DATA["gene_names"]: panel genes first, then any genes imputed
        # into the Xenium table, which the reseg matrix does not have
        gene_names = list(gene_names)[:expr_mat.shape[1]]
    var_df = pd.DataFrame({"is_imputed": False}, index=list(gene_names))
    obs_df = cells_df.copy()
    obs_df.index = obs_df.index.astype(str)

    n_genes = len(gene_names)
    if expr_mat is not None:
        X = sp.csr_matrix(expr_mat, dtype=np.float32)
    else:
        X = sp.csr_matrix((len(obs_df), n_genes), dtype=np.float32)

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
                tables={"table": adata},
                shapes={"cell_boundaries": shapes},
            )
        except Exception:
            sdata = spatialdata.SpatialData(tables={"table": adata})
    else:
        sdata = spatialdata.SpatialData(tables={"table": adata})

    zarr_path = os.path.join(out_dir, f"spatialdata_{source}.zarr")
    try:
        # A re-run into the same cache dir must replace the previous run's store
        # (sdata.write refuses to overwrite and the old cells would be loaded later).
        if os.path.isdir(zarr_path):
            import shutil
            shutil.rmtree(zarr_path)
        sdata.write(zarr_path)
        print(f"  Reseg zarr written: {zarr_path}", flush=True)
    except Exception as e:
        print(f"  Warning: failed to write reseg zarr: {e}", flush=True)
        return ""
    return zarr_path


def _fast_load_reseg_zarr(zarr_path: str):
    """Load a reseg result entirely from the zarr store using the low-level zarr API.

    Returns (cells_df, expr_mat, X_corrected, corr_imp_genes, cell_bounds).
    Raises on any error so callers can fall back to the CSV-based path.
    """
    import zarr as _zarr_mod

    _zroot = _zarr_mod.open_group(zarr_path, mode="r", use_consolidated=False)
    _tbl = _zroot["tables"]["table"]
    _obs = _tbl["obs"]

    # ── cells_df from obs ────────────────────────────────────────────
    _idx_key = _obs.attrs.get("_index", "_index")
    _ids = [str(v) for v in _obs[_idx_key][:]]
    _df_data = {}
    for _col in _obs.keys():
        if _col in {"__categories", "_index"} or _col == _idx_key:
            continue
        if _col in _RESEG_FOREIGN_OBS:
            continue
        try:
            _df_data[_col] = _obs[_col][:]
        except Exception:
            pass
    if "x_centroid" not in _df_data or "y_centroid" not in _df_data:
        raise ValueError("zarr obs missing centroid columns — zarr predates centroid storage")
    cells_df = pd.DataFrame(_df_data, index=pd.Index(_ids, name="cell_id"))
    cells_df.index = cells_df.index.astype(str)

    # ── expr_mat from X ──────────────────────────────────────────────
    _xgrp = _tbl["X"]
    _enc = _xgrp.attrs.get("encoding-type", "")
    if "sparse" in _enc or "csr" in _enc or "csc" in _enc or (
            "data" in _xgrp and "indices" in _xgrp and "indptr" in _xgrp):
        _shape_attr = _xgrp.attrs.get("shape", None)
        if _shape_attr is not None:
            _shape = (int(_shape_attr[0]), int(_shape_attr[1]))
        else:
            _var_key = _tbl["var"].attrs.get("_index", "_index")
            _n_vars = len(_tbl["var"][_var_key][:])
            _shape = (len(_ids), _n_vars)
        expr_mat = sp.csr_matrix(
            (_xgrp["data"][:].astype(np.float32),
             _xgrp["indices"][:], _xgrp["indptr"][:]),
            shape=_shape,
        )
    else:
        expr_mat = sp.csr_matrix(np.array(_xgrp, dtype=np.float32))

    if expr_mat.nnz == 0:
        raise ValueError("Zarr expression matrix is all zeros — rebuilding from transcripts")

    # ── X_corrected layer + imputed gene list ────────────────────────
    X_corrected = None
    corr_imp_genes = []
    try:
        if "layers" in _tbl and "X_corrected" in _tbl["layers"]:
            _lg = _tbl["layers"]["X_corrected"]
            _lshape = tuple(_lg.attrs.get("shape", expr_mat.shape))
            X_corrected = sp.csr_matrix(
                (_lg["data"][:].astype(np.float32),
                 _lg["indices"][:], _lg["indptr"][:]),
                shape=_lshape,
            )
        if "uns" in _tbl:
            try:
                _ig = _tbl["uns"].get("split_corrected_imputed_genes")
                if _ig is not None:
                    corr_imp_genes = [str(g) for g in _ig[:]]
            except Exception:
                pass
    except Exception:
        pass

    # ── boundaries from shapes parquet ──────────────────────────────
    cell_bounds: dict = {}
    try:
        import geopandas as _gpd
        _pq = os.path.join(zarr_path, "shapes", "cell_boundaries", "shapes.parquet")
        if os.path.exists(_pq):
            _gdf = _gpd.read_parquet(_pq)
            for _cid, _geom in zip(_gdf.index, _gdf.geometry):
                try:
                    if _geom.geom_type == "Polygon":
                        _arr = np.array(_geom.exterior.coords)
                    elif _geom.geom_type == "MultiPolygon":
                        _arr = np.array(max(_geom.geoms, key=lambda g: g.area).exterior.coords)
                    else:
                        continue
                    cell_bounds[str(_cid)] = (_arr[:, 0].tolist(), _arr[:, 1].tolist())
                except Exception:
                    pass
            if cell_bounds:
                print(f"  Loaded {len(cell_bounds):,} boundaries from zarr parquet", flush=True)
    except Exception:
        pass

    return cells_df, expr_mat, X_corrected, corr_imp_genes, cell_bounds


def _reseg_zarr_gene_index(zarr_path: str, n_cols: int) -> dict:
    """Result-dict entries ('gene_name_to_idx', 'imputed_genes') for a reseg zarr whose
    table holds SpaGE-imputed columns, so a run reloaded from its cache lists and
    indexes them like the session that imputed them. Empty for a panel-only table."""
    try:
        import zarr as _zarr_mod
        _var = _zarr_mod.open_group(zarr_path, mode="r",
                                    use_consolidated=False)["tables"]["table"]["var"]
        _names = [str(v) for v in _var[_var.attrs.get("_index", "_index")][:]]
        if "is_imputed" not in _var or len(_names) != n_cols:
            return {}
        _imp = np.asarray(_var["is_imputed"][:]).astype(bool)
        if not _imp.any():
            return {}
        return {"gene_name_to_idx": {g: i for i, g in enumerate(_names)},
                "imputed_genes":    [g for g, f in zip(_names, _imp) if f]}
    except Exception as _ge:
        print(f"  Warning: could not read gene index from reseg zarr: {_ge}", flush=True)
        return {}


def _reseg_patch_keep_mask(out_dir: str, zarr_path: str, ids, patch_idx, x, y,
                           n_tx=None, polys=None):
    """Slow-path cache load of a patch run: which of the cells read from the patch_XXXX
    directories did the live run keep? Taken from the run's zarr table when it exists,
    else recomputed from the patch layout stored in params.json ('patch_bounds'):
    centroid ownership, then - for runs whose params.json records 'seam_dedup' - the
    same seam de-duplication (needs *n_tx* and the boundaries *polys*, id → (xs, ys)).
    Returns a bool mask over *ids*, or None when neither is available."""
    ids = pd.Index([str(i) for i in ids])
    try:
        if os.path.isdir(zarr_path):
            import zarr as _zarr_mod
            _obs = _zarr_mod.open_group(zarr_path, mode="r",
                                        use_consolidated=False)["tables"]["table"]["obs"]
            _zids = pd.Index([str(v) for v in _obs[_obs.attrs.get("_index", "_index")][:]])
            # live Baysor patch runs prefix the ids with "p<patch>_"
            for _z in (_zids, _zids.str.replace(r'^p\d+_', '', regex=True)):
                if len(_z) and _z.isin(ids).all():
                    return np.asarray(ids.isin(_z))
    except Exception as _ke:
        print(f"  Warning: could not read cell ids from reseg zarr: {_ke}", flush=True)
    try:
        with open(os.path.join(out_dir, "params.json")) as _pf:
            _params = json.load(_pf)
        _pb = [tuple(float(v) for v in b) for b in (_params.get("patch_bounds") or [])]
    except Exception:
        _params, _pb = {}, []
    if not _pb:
        return None
    patch_idx = np.asarray(patch_idx)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if _params.get("patch_ok") is not None:
        _ok = sorted(int(p) for p in _params["patch_ok"] if 0 <= int(p) < len(_pb))
    else:
        _ok = sorted(int(p) for p in set(patch_idx.tolist()) if 0 <= p < len(_pb))
    keep = np.zeros(len(ids), dtype=bool)
    for _pi in _ok:
        x0, y0, x1, y1 = _pb[_pi]
        _idx = np.where((patch_idx == _pi) & (x >= x0) & (x < x1) & (y >= y0) & (y < y1))[0]
        keep[_idx] = _patch_owned_mask(x[_idx], y[_idx], _pi, _pb, _ok)
    _cfg = _params.get("seam_dedup")
    if _cfg:
        _k = np.where(keep)[0]
        _seam = _patch_seam_dedup_mask(
            ids[_k], patch_idx[_k], x[_k], y[_k],
            None if n_tx is None else np.asarray(n_tx, dtype=float)[_k],
            polys, _pb, _ok, cfg=_cfg)
        keep[_k[~_seam]] = False
    return keep


def _load_cached_baysor(out_dir: str) -> None:
    """Load a cached Baysor result from disk into _baysor_state."""
    _gen = jobs._dataset_gen

    def _set(status, message, result=None):
        with core._baysor_lock:
            if jobs._stale_job(_gen, status, message):
                return
            core._baysor_state["status"]  = status
            core._baysor_state["message"] = message
            if result is not None:
                core._baysor_state["result"] = result
            if status != "running":
                core._baysor_state["loading_out_dir"] = None   # see load_seg_run_from_cache
    print(f"Loading cached Baysor: {os.path.basename(out_dir)}…", flush=True)
    try:
        _set("running", "Loading cached Baysor result…")
        # Restore SpaGE result reference written by _run_spage_imputation
        _spage_rpath, _spage_rgenes = None, None
        _spage_ref_file = os.path.join(out_dir, "spage_result.json")
        if os.path.exists(_spage_ref_file):
            try:
                import json as _jref
                with open(_spage_ref_file) as _jf:
                    _sref = _jref.load(_jf)
                if os.path.isdir(_sref.get("path", "")):
                    _spage_rpath  = _sref["path"]
                    _spage_rgenes = _sref.get("genes", [])
                    print(f"  Baysor: restored SpaGE result ({len(_spage_rgenes)} genes)", flush=True)
            except Exception:
                pass
        zarr_path_fast = os.path.join(out_dir, "spatialdata_baysor.zarr")
        if os.path.isdir(zarr_path_fast):
            try:
                _set("running", "Loading Baysor from zarr (fast path)…")
                cells_df, expr_mat, _baysor_corr, _baysor_corr_imp, cell_bounds = \
                    _fast_load_reseg_zarr(zarr_path_fast)
                n_cells = len(cells_df)
                _has_cluster = bool(_reseg_cluster_cols(cells_df.columns))
                if ("umap_1" not in cells_df.columns or not _has_cluster) and expr_mat is not None:
                    _set("running", f"Computing UMAP and clusters for {n_cells:,} cells (one-time)…")
                    clustering._compute_reseg_clusters_umap(cells_df, expr_mat,
                                                 progress_fn=lambda msg: _set("running", msg))
                    _update_reseg_zarr_obs(zarr_path_fast, cells_df)
                # If X_corrected exists but split_umap is all NaN (zarr write bug),
                # recompute SPLIT clusters + UMAP from the corrected matrix.
                _split_umap_missing = (
                    _baysor_corr is not None
                    and ("split_umap_1" not in cells_df.columns
                         or not cells_df["split_umap_1"].notna().any())
                )
                if _split_umap_missing:
                    _set("running", f"Recomputing SPLIT UMAP for {n_cells:,} cells (one-time fix)…")
                    split._compute_split_clusters_umap(
                        cells_df, _baysor_corr,
                        progress_fn=lambda msg: _set("running", msg))
                    _update_reseg_zarr_obs(zarr_path_fast, cells_df)
                    print(f"  Baysor: recomputed SPLIT UMAP ({cells_df['split_umap_1'].notna().sum()} "
                          f"non-NaN values)", flush=True)
                with core._roi_lock:
                    regions._roi_apply_metadata_to_df(cells_df, core._roi_state["rois"])
                _set("done", f"Loaded — {n_cells:,} cells", result={
                    "cells_df": cells_df, "cell_bounds": cell_bounds,
                    "out_dir": out_dir, "expr": expr_mat, "source": "baysor",
                    "sdata_path": zarr_path_fast,
                    "split_corrected_expr": _baysor_corr,
                    "split_corrected_imputed_genes": _baysor_corr_imp,
                    "spage_result_path": _spage_rpath,
                    "spage_result_genes": _spage_rgenes,
                    **_reseg_zarr_gene_index(zarr_path_fast, expr_mat.shape[1]),
                })
                print(f"Loaded cached Baysor (fast): {n_cells:,} cells, "
                      f"{len(cell_bounds):,} boundaries", flush=True)
                return
            except jobs._DatasetChanged:
                raise
            except Exception as _fe:
                print(f"  Fast zarr load failed ({_fe}), falling back to CSV…", flush=True)

        # ── Slow path: rebuild from CSV files ────────────────────────
        # Collect all segmentation.csv files (may be one per patch subdir)
        seg_parts = []
        for root, dirs, files in os.walk(out_dir):
            if "segmentation.csv" in files:
                _part = pd.read_csv(os.path.join(root, "segmentation.csv"))
                _pm = re.fullmatch(r"patch_(\d+)", os.path.basename(root))
                _part["_patch"] = int(_pm.group(1)) if _pm else -1
                seg_parts.append(_part)
        if not seg_parts:
            _set("error", f"segmentation.csv not found in {out_dir}")
            return
        seg = pd.concat(seg_parts, ignore_index=True)
        seg = seg[seg["cell"].notna()].copy()
        seg["cell"] = seg["cell"].astype(str)
        x_col = "x" if "x" in seg.columns else "x_location"
        y_col = "y" if "y" in seg.columns else "y_location"
        cells_df = (
            seg.groupby("cell")
            .agg(x_centroid=(x_col, "mean"), y_centroid=(y_col, "mean"),
                 transcript_counts=(x_col, "count"), _patch=("_patch", "first"))
            .rename_axis("cell_id")
        )
        cells_df.index = cells_df.index.astype(str)
        # Build cell boundaries from polygon files (before the patch merge below, which
        # uses them for the seam de-duplication)
        cell_bounds: dict = {}
        for root, dirs, files in os.walk(out_dir):
            for poly_name in ("segmentation_polygons_2d.json", "segmentation_polygons.json",
                              "polygons.json", "cell_polygons.json"):
                if poly_name in files:
                    poly_path = os.path.join(root, poly_name)
                    with open(poly_path) as f:
                        geo = json.load(f)
                    for feat in geo.get("features", []):
                        props = feat.get("properties", {})
                        cid = str(props.get("cell", props.get("cell_id", feat.get("id", ""))))
                        coords = feat.get("geometry", {}).get("coordinates", [[]])[0]
                        if coords:
                            cell_bounds[cid] = ([c[0] for c in coords], [c[1] for c in coords])
                    break  # only one polygon file per dir needed
        # Patch run: the per-patch CSVs also hold the cells of the overlap buffers that
        # the live run gave to one patch only - keep the cells the live run kept.
        _pkeep = None
        if (cells_df["_patch"] >= 0).any():
            _pkeep = _reseg_patch_keep_mask(
                out_dir, os.path.join(out_dir, "spatialdata_baysor.zarr"), cells_df.index,
                cells_df["_patch"].values, cells_df["x_centroid"].values,
                cells_df["y_centroid"].values,
                n_tx=cells_df["transcript_counts"].values, polys=cell_bounds)
            if _pkeep is None:
                print("  Warning: Baysor patch run without zarr table or stored patch layout "
                      "— cells in the patch overlaps may be loaded twice", flush=True)
            else:
                cells_df = cells_df[_pkeep]
                seg = seg[seg["cell"].isin(cells_df.index)]
        cells_df = cells_df.drop(columns="_patch")
        if _pkeep is not None:
            _kept_ids = set(cells_df.index)
            cell_bounds = {k: v for k, v in cell_bounds.items() if k in _kept_ids}
        expr_mat = _build_baysor_expr(seg, cells_df)
        n_cells = len(cells_df)

        zarr_path = os.path.join(out_dir, "spatialdata_baysor.zarr")

        # ── Load or build zarr ────────────────────────────────────────────
        _baysor_corr     = None
        _baysor_corr_imp = []
        if not os.path.isdir(zarr_path):
            # First time: compute clusters/UMAP then build zarr (includes them)
            if expr_mat is not None:
                _set("running", f"Computing UMAP and clusters for {n_cells:,} cells…")
                clustering._compute_reseg_clusters_umap(cells_df, expr_mat,
                                             progress_fn=lambda msg: _set("running", msg))
            jobs._check_dataset_gen(_gen)   # DATA["gene_names"] must be this run's dataset
            zarr_path = _build_reseg_sdata(
                cells_df, cell_bounds, expr_mat, core.DATA["gene_names"], "baysor", out_dir
            )
        else:
            # Zarr exists — load obs columns (cluster, UMAP) from it directly
            # to avoid recomputing and the 'target path in use' write-back error.
            _load_zarr_obs_into_df(zarr_path, cells_df)
            _zarr_obs_cols = [c for c in cells_df.columns
                              if c not in ("x_centroid", "y_centroid", "transcript_counts")]
            print(f"  Baysor cache: loaded obs cols from zarr: {_zarr_obs_cols}", flush=True)

            # If cluster/UMAP still missing (very old cache or ID mismatch), compute and write back
            _has_cluster = bool(_reseg_cluster_cols(cells_df.columns))
            if ("umap_1" not in cells_df.columns or not _has_cluster) and expr_mat is not None:
                _set("running", f"Computing UMAP and clusters for {n_cells:,} cells (one-time migration)…")
                clustering._compute_reseg_clusters_umap(cells_df, expr_mat,
                                             progress_fn=lambda msg: _set("running", msg))
                _update_reseg_zarr_obs(zarr_path, cells_df)

            # Load corrected expr + cell boundaries from zarr
            try:
                import spatialdata as _sd_b
                _sdata_b = _sd_b.read_zarr(zarr_path)
                _adata_b = _sdata_b.tables["table"]
                _baysor_corr = (
                    sp.csr_matrix(_adata_b.layers["X_corrected"])
                    if "X_corrected" in _adata_b.layers else None
                )
                # cells_df here is rebuilt from the CSVs: align the zarr rows by id
                _baysor_corr = _align_zarr_rows_to_cells(
                    _baysor_corr, _adata_b.obs_names, cells_df, "Baysor")
                _baysor_corr_imp = list(_adata_b.uns.get("split_corrected_imputed_genes", []))
                _load_zarr_boundaries_into_dict(_sdata_b, cell_bounds, "Baysor")
            except Exception as _ze2:
                print(f"  Warning: could not read Baysor zarr: {_ze2}", flush=True)

        with core._roi_lock:
            regions._roi_apply_metadata_to_df(cells_df, core._roi_state["rois"])
        _set("done", f"Loaded — {n_cells:,} cells", result={
            "cells_df": cells_df, "cell_bounds": cell_bounds,
            "out_dir": out_dir, "expr": expr_mat, "source": "baysor",
            "sdata_path": zarr_path,
            "split_corrected_expr": _baysor_corr,
            "split_corrected_imputed_genes": _baysor_corr_imp,
            "spage_result_path": _spage_rpath,
            "spage_result_genes": _spage_rgenes,
        })
        print(f"Loaded cached Baysor: {n_cells:,} cells, {len(cell_bounds):,} boundaries "
              f"from {os.path.basename(out_dir)}", flush=True)
    except jobs._JobCancelled:
        raise
    except Exception as exc:
        import traceback; traceback.print_exc()
        _set("error", str(exc)[:300])


def _load_proseg_patch_boundaries(out_dir: str, patch_names: list) -> dict:
    """Load Proseg cell boundaries from per-patch polygon files with global ID remapping.

    Applies the same ID offset used for cell-metadata: global_id = local_id + 1 + i * STRIDE.
    Returns a cell_bounds dict {str(global_id): ([x…], [y…])}.
    """
    _STRIDE = 10_000_000
    cell_bounds: dict = {}
    for _i, _pname in enumerate(patch_names):
        _offset = 1 + _i * _STRIDE
        _pdir = os.path.join(out_dir, _pname)
        for _poly_name in ("cell-polygons.geojson.gz", "polygons.geojson",
                           "polygons.json", "cell_polygons.json"):
            _poly_path = os.path.join(_pdir, _poly_name)
            if not os.path.exists(_poly_path):
                continue
            try:
                import gzip as _gz
                _opener = _gz.open if _poly_name.endswith(".gz") else open
                with _opener(_poly_path, "rt") as _pf:
                    _geo = json.load(_pf)
                for _feat in _geo.get("features", []):
                    _props = _feat.get("properties", {})
                    _local = _props.get("cell_id", _props.get("cell", _feat.get("id", "")))
                    try:
                        _gid = str(int(_local) + _offset)
                    except (ValueError, TypeError):
                        _gid = str(_local)
                    _geom = _feat.get("geometry", {})
                    _gtype = _geom.get("type")
                    if _gtype == "Polygon":
                        _rings = [_geom["coordinates"][0]]
                    elif _gtype == "MultiPolygon":
                        _rings = [_p[0] for _p in _geom["coordinates"]]
                    else:
                        _rings = [_geom.get("coordinates", [[]])[0]]
                    _coords = _largest_ring(_rings)
                    if _coords:
                        cell_bounds[_gid] = ([c[0] for c in _coords], [c[1] for c in _coords])
            except Exception as _pe:
                print(f"  Proseg: patch polygon load error in {_pname}: {_pe}", flush=True)
            break  # stop after first polygon file found in this patch
    return cell_bounds


def _load_cached_proseg(out_dir: str) -> None:
    """Load a cached Proseg result from disk into _proseg_state."""
    _gen = jobs._dataset_gen

    def _set(status, message, result=None):
        with core._proseg_lock:
            if jobs._stale_job(_gen, status, message):
                return
            core._proseg_state["status"]  = status
            core._proseg_state["message"] = message
            if result is not None:
                core._proseg_state["result"] = result
            if status != "running":
                core._proseg_state["loading_out_dir"] = None   # see load_seg_run_from_cache
    print(f"Loading cached Proseg: {os.path.basename(out_dir)}…", flush=True)
    try:
        _set("running", "Loading cached Proseg result…")
        # Restore SpaGE result reference written by _run_spage_imputation
        _spage_rpath, _spage_rgenes = None, None
        _spage_ref_file = os.path.join(out_dir, "spage_result.json")
        if os.path.exists(_spage_ref_file):
            try:
                import json as _jref
                with open(_spage_ref_file) as _jf:
                    _sref = _jref.load(_jf)
                if os.path.isdir(_sref.get("path", "")):
                    _spage_rpath  = _sref["path"]
                    _spage_rgenes = _sref.get("genes", [])
                    print(f"  Proseg: restored SpaGE result ({len(_spage_rgenes)} genes)", flush=True)
            except Exception:
                pass
        zarr_path_fast = os.path.join(out_dir, "spatialdata_proseg.zarr")
        if os.path.isdir(zarr_path_fast):
            try:
                _set("running", "Loading Proseg from zarr (fast path)…")
                cells_df, expr_mat, _proseg_corr, _proseg_corr_imp, cell_bounds = \
                    _fast_load_reseg_zarr(zarr_path_fast)
                n_cells = len(cells_df)
                _has_cluster = bool(_reseg_cluster_cols(cells_df.columns))
                if ("umap_1" not in cells_df.columns or not _has_cluster) and expr_mat is not None:
                    _set("running", f"Computing UMAP and clusters for {n_cells:,} cells (one-time)…")
                    clustering._compute_reseg_clusters_umap(cells_df, expr_mat,
                                                 progress_fn=lambda msg: _set("running", msg))
                    _update_reseg_zarr_obs(zarr_path_fast, cells_df)
                # If X_corrected exists but split_umap is all NaN (zarr write bug),
                # recompute SPLIT clusters + UMAP from the corrected matrix.
                _split_umap_missing = (
                    _proseg_corr is not None
                    and ("split_umap_1" not in cells_df.columns
                         or not cells_df["split_umap_1"].notna().any())
                )
                if _split_umap_missing:
                    _set("running", f"Recomputing SPLIT UMAP for {n_cells:,} cells (one-time fix)…")
                    split._compute_split_clusters_umap(
                        cells_df, _proseg_corr,
                        progress_fn=lambda msg: _set("running", msg))
                    _update_reseg_zarr_obs(zarr_path_fast, cells_df)
                    print(f"  Proseg: recomputed SPLIT UMAP ({cells_df['split_umap_1'].notna().sum()} "
                          f"non-NaN values)", flush=True)
                # One-time migration: read cell_area from cell-metadata.csv.gz if missing from zarr
                _need_cell_area = (
                    "cell_area" not in cells_df.columns
                    or cells_df["cell_area"].notna().sum() < len(cells_df) * 0.9
                )
                if _need_cell_area:
                    try:
                        _STRIDE = 10_000_000
                        # Prefer top-level cell-metadata.csv.gz (single-run or combined output)
                        _top_cm_path = os.path.join(out_dir, "cell-metadata.csv.gz")
                        if os.path.exists(_top_cm_path):
                            _cm = pd.read_csv(_top_cm_path, usecols=["cell", "volume"])
                            _cm["cell"] = _cm["cell"].astype(str)
                            _cm = _cm.rename(columns={"volume": "cell_area"}).set_index("cell")
                            cells_df["cell_area"] = _cm["cell_area"].reindex(cells_df.index)
                        else:
                            # Patch-based: read volumes from each patch, remap IDs
                            _patch_dirs = sorted([
                                _pd for _pd in os.listdir(out_dir)
                                if os.path.isdir(os.path.join(out_dir, _pd)) and _pd.startswith("patch_")
                                and os.path.exists(os.path.join(out_dir, _pd, "cell-metadata.csv.gz"))
                            ])
                            if _patch_dirs:
                                _area_map = {}
                                for _pi, _pd in enumerate(_patch_dirs):
                                    _cm = pd.read_csv(os.path.join(out_dir, _pd, "cell-metadata.csv.gz"),
                                                      usecols=["cell", "volume"])
                                    for _lid, _vol in zip(_cm["cell"].values, _cm["volume"].values):
                                        _gid = str(int(_lid) + 1 + _pi * _STRIDE)
                                        _area_map[_gid] = _vol
                                cells_df["cell_area"] = cells_df.index.map(
                                    lambda x: _area_map.get(str(x), np.nan)
                                )
                        _matched = cells_df["cell_area"].notna().sum()
                        if _matched > 0:
                            _update_reseg_zarr_obs(zarr_path_fast, cells_df)
                            print(f"  Proseg: migrated cell_area ({_matched:,}/{len(cells_df):,} cells)", flush=True)
                        else:
                            print(f"  Proseg: cell_area migration found 0 matches "
                                  f"(index sample: {list(cells_df.index[:3])})", flush=True)
                    except Exception as _ce:
                        import traceback
                        traceback.print_exc()
                        print(f"  Proseg: could not migrate cell_area: {_ce}", flush=True)

                # Patch runs build the zarr without cell_boundaries shapes — load from polygon files
                if not cell_bounds:
                    _fp_patch_names = sorted([
                        _d for _d in os.listdir(out_dir)
                        if os.path.isdir(os.path.join(out_dir, _d)) and _d.startswith("patch_")
                        and os.path.exists(os.path.join(out_dir, _d, "cell-metadata.csv.gz"))
                    ])
                    if _fp_patch_names:
                        cell_bounds = _load_proseg_patch_boundaries(out_dir, _fp_patch_names)
                        # the files also hold the cells the patch merge did not keep
                        _fp_ids = set(cells_df.index.astype(str))
                        cell_bounds = {k: v for k, v in cell_bounds.items() if k in _fp_ids}
                        if cell_bounds:
                            print(f"  Proseg: loaded {len(cell_bounds):,} patch boundaries "
                                  f"(polygon files)", flush=True)
                _fp_gidx = _reseg_zarr_gene_index(zarr_path_fast, expr_mat.shape[1])
                # transcript_counts = row sums of the matrix (caches written before this
                # rule hold Proseg's `population`: corrected here and in the zarr, once)
                _fp_imp = [_fp_gidx["gene_name_to_idx"][g]
                           for g in _fp_gidx.get("imputed_genes", [])]
                if _proseg_counts_from_expr(cells_df, expr_mat, _fp_imp):
                    _update_reseg_zarr_obs(zarr_path_fast, cells_df,
                                           force_cols=("transcript_counts",))
                    print("  Proseg: transcript_counts set to the matrix row sums", flush=True)
                with core._roi_lock:
                    regions._roi_apply_metadata_to_df(cells_df, core._roi_state["rois"])
                _set("done", f"Loaded — {n_cells:,} cells", result={
                    "cells_df": cells_df, "cell_bounds": cell_bounds,
                    "out_dir": out_dir, "expr": expr_mat, "source": "proseg",
                    "sdata_path": zarr_path_fast,
                    "split_corrected_expr": _proseg_corr,
                    "split_corrected_imputed_genes": _proseg_corr_imp,
                    "spage_result_path": _spage_rpath,
                    "spage_result_genes": _spage_rgenes,
                    **_fp_gidx,
                })
                print(f"Loaded cached Proseg (fast): {n_cells:,} cells, "
                      f"{len(cell_bounds):,} boundaries", flush=True)
                return
            except jobs._DatasetChanged:
                raise
            except Exception as _fe:
                print(f"  Fast zarr load failed ({_fe}), falling back to CSV…", flush=True)

        # ── Slow path: rebuild from CSV files ────────────────────────
        # Detect patch-based vs single run.
        # Patch runs use per-patch local cell IDs that must be remapped to global IDs
        # matching the scheme used during the live run: global_id = local_id + 1 + i*STRIDE
        _CELL_ID_STRIDE   = 10_000_000
        _PROSEG_UNASSIGNED = 4294967295  # 0xFFFFFFFF sentinel for unassigned transcripts
        _patch_names = sorted([
            d for d in os.listdir(out_dir)
            if os.path.isdir(os.path.join(out_dir, d)) and d.startswith("patch_")
            and os.path.exists(os.path.join(out_dir, d, "cell-metadata.csv.gz"))
        ])
        meta_parts: list = []
        tx_parts:   list = []
        if _patch_names:
            # Patch run: apply global-ID stride (replicates live-run numbering)
            for _i, _pname in enumerate(_patch_names):
                _offset = 1 + _i * _CELL_ID_STRIDE
                _pdir   = os.path.join(out_dir, _pname)
                _df     = pd.read_csv(os.path.join(_pdir, "cell-metadata.csv.gz"))
                _remap  = {int(lid): int(lid) + _offset for lid in _df["cell"].values}
                _df["cell"] = _df["cell"].map(_remap)
                _pm = re.fullmatch(r"patch_(\d+)", _pname)
                _df["_patch"] = int(_pm.group(1)) if _pm else -1
                meta_parts.append(_df)
                _tx_path = os.path.join(_pdir, "transcript-metadata.csv.gz")
                if os.path.exists(_tx_path):
                    _tx_df = _read_proseg_tx_meta(_tx_path)
                    if "assignment" in _tx_df.columns:
                        _mask = _tx_df["assignment"] != _PROSEG_UNASSIGNED
                        _tx_df = _tx_df.copy()
                        _tx_df.loc[_mask, "assignment"] = (
                            _tx_df.loc[_mask, "assignment"].map(_remap)
                        )
                    elif "cell" in _tx_df.columns:
                        _tx_df = _tx_df.copy()
                        _tx_df["cell"] = _tx_df["cell"].map(_remap)
                    tx_parts.append(_tx_df)
        else:
            # Single run: flat walk (no ID remapping needed)
            for _root, _dirs, _files in os.walk(out_dir):
                if "cell-metadata.csv.gz" in _files:
                    meta_parts.append(pd.read_csv(os.path.join(_root, "cell-metadata.csv.gz")))
                if "transcript-metadata.csv.gz" in _files:
                    tx_parts.append(_read_proseg_tx_meta(os.path.join(_root, "transcript-metadata.csv.gz")))
        if not meta_parts:
            _set("error", f"cell-metadata.csv.gz not found in {out_dir}")
            return
        cell_meta = pd.concat(meta_parts, ignore_index=True)
        tx_meta   = pd.concat(tx_parts,   ignore_index=True) if tx_parts else pd.DataFrame()
        cells_df = cell_meta.rename(columns={
            "cell": "cell_id", "centroid_x": "x_centroid", "centroid_y": "y_centroid",
            "population": "transcript_counts", "volume": "cell_area",
        })
        if "cell_id" in cells_df.columns:
            cells_df = cells_df.set_index("cell_id")
        cells_df.index = cells_df.index.astype(str)
        # Same columns as the live run (_proseg_run_single): Proseg's own `cluster`,
        # `fov`, `centroid_z` … are not carried into the result or its zarr.
        cells_df = cells_df[[c for c in ("x_centroid", "y_centroid", "transcript_counts",
                                         "cell_area", "_patch") if c in cells_df.columns]]
        # Patch run: sopa patches overlap, and the live run gave each cell of an overlap
        # to one patch only - keep the cells the live run kept.
        # Cell boundaries from the polygon files come first: the patch merge uses them
        # for the seam de-duplication.
        cell_bounds: dict = {}
        if _patch_names:
            cell_bounds = _load_proseg_patch_boundaries(out_dir, _patch_names)
        _pkeep = None
        if "_patch" in cells_df.columns:
            _pkeep = _reseg_patch_keep_mask(
                out_dir, os.path.join(out_dir, "spatialdata_proseg.zarr"), cells_df.index,
                cells_df["_patch"].values, cells_df["x_centroid"].values,
                cells_df["y_centroid"].values,
                n_tx=cells_df["transcript_counts"].values, polys=cell_bounds)
            if _pkeep is None:
                print("  Warning: Proseg patch run without zarr table or stored patch layout "
                      "— cells in the patch overlaps may be loaded twice", flush=True)
            else:
                cells_df = cells_df[_pkeep]
            cells_df = cells_df.drop(columns="_patch")
        if _patch_names:
            if _pkeep is not None:
                _kept_ids = set(cells_df.index)
                cell_bounds = {k: v for k, v in cell_bounds.items() if k in _kept_ids}
            if cell_bounds:
                print(f"  Proseg: loaded {len(cell_bounds):,} patch boundaries", flush=True)
        else:
            for root, dirs, files in os.walk(out_dir):
                for poly_name in ("cell-polygons.geojson.gz", "polygons.geojson", "polygons.json",
                                  "cell_polygons.json", "segmentation_polygons.json"):
                    if poly_name in files:
                        poly_path = os.path.join(root, poly_name)
                        import gzip as _gzip
                        _opener = _gzip.open if poly_name.endswith(".gz") else open
                        with _opener(poly_path, "rt") as f:
                            geo = json.load(f)
                        for feat in geo.get("features", []):
                            props = feat.get("properties", {})
                            cid = str(props.get("cell_id", props.get("cell", feat.get("id", ""))))
                            geom = feat.get("geometry", {})
                            gtype = geom.get("type")
                            if gtype == "Polygon":
                                rings = [geom["coordinates"][0]]
                            elif gtype == "MultiPolygon":
                                rings = [poly[0] for poly in geom["coordinates"]]
                            else:
                                rings = [geom.get("coordinates", [[]])[0]]
                            coords = _largest_ring(rings)
                            if coords:
                                cell_bounds[cid] = ([c[0] for c in coords], [c[1] for c in coords])
                        break
        expr_mat = _build_proseg_expr(tx_meta, cells_df) if not tx_meta.empty else None
        # Same rule as the live run: transcript_counts = row sums of the matrix (the
        # patch merge above ranked seam duplicates by `population`, as the live run did)
        cells_df = cells_df.copy()
        _proseg_counts_from_expr(cells_df, expr_mat)
        n_cells = len(cells_df)

        zarr_path = os.path.join(out_dir, "spatialdata_proseg.zarr")

        # ── Load or build zarr ────────────────────────────────────────────
        _proseg_corr     = None
        _proseg_corr_imp = []
        if not os.path.isdir(zarr_path):
            # First time: compute clusters/UMAP then build zarr (includes them)
            if expr_mat is not None:
                _set("running", f"Computing UMAP and clusters for {n_cells:,} cells…")
                clustering._compute_reseg_clusters_umap(cells_df, expr_mat,
                                             progress_fn=lambda msg: _set("running", msg))
            jobs._check_dataset_gen(_gen)   # DATA["gene_names"] must be this run's dataset
            zarr_path = _build_reseg_sdata(
                cells_df, cell_bounds, expr_mat, core.DATA["gene_names"], "proseg", out_dir
            )
        else:
            # Zarr exists — load obs columns (cluster, UMAP) from it directly
            _load_zarr_obs_into_df(zarr_path, cells_df)
            _zarr_obs_cols_p = [c for c in cells_df.columns
                                if c not in ("x_centroid", "y_centroid", "transcript_counts")]
            print(f"  Proseg cache: loaded obs cols from zarr: {_zarr_obs_cols_p}", flush=True)

            # If cluster/UMAP still missing (very old cache or ID mismatch), compute and write back
            _has_cluster_p = bool(_reseg_cluster_cols(cells_df.columns))
            if ("umap_1" not in cells_df.columns or not _has_cluster_p) and expr_mat is not None:
                _set("running", f"Computing UMAP and clusters for {n_cells:,} cells (one-time migration)…")
                clustering._compute_reseg_clusters_umap(cells_df, expr_mat,
                                             progress_fn=lambda msg: _set("running", msg))
                _update_reseg_zarr_obs(zarr_path, cells_df)
            elif "cell_area" in cells_df.columns and os.path.isdir(zarr_path):
                # cell_area present in CSV but may be missing from old zarr — write back
                _update_reseg_zarr_obs(zarr_path, cells_df)

            # Load expr (if missing), corrected expr + cell boundaries from zarr
            try:
                import spatialdata as _sd_p
                _sdata_p = _sd_p.read_zarr(zarr_path)
                _adata_p = _sdata_p.tables["table"]
                if expr_mat is None:
                    expr_mat = sp.csr_matrix(_adata_p.X)
                    print(f"  Proseg: loaded expr from zarr {expr_mat.shape}", flush=True)
                    # Align cells_df to zarr obs order (zarr may have fewer cells than CSV)
                    zarr_ids = [str(i) for i in _adata_p.obs_names]
                    if cells_df.index.duplicated().any():
                        cells_df = cells_df[~cells_df.index.duplicated(keep="first")]
                    cells_df = cells_df.reindex(zarr_ids).dropna(how="all")
                    cells_df.index = cells_df.index.astype(str)
                    _zimp = (np.where(np.asarray(_adata_p.var["is_imputed"]).astype(bool))[0]
                             if "is_imputed" in _adata_p.var.columns else None)
                    _proseg_counts_from_expr(cells_df, expr_mat, _zimp)
                    n_cells = len(cells_df)
                # a zarr written before transcript_counts followed the matrix
                _update_reseg_zarr_obs(zarr_path, cells_df, force_cols=("transcript_counts",))
                _proseg_corr = (
                    sp.csr_matrix(_adata_p.layers["X_corrected"])
                    if "X_corrected" in _adata_p.layers else None
                )
                # cells_df here is rebuilt from the CSVs: align the zarr rows by id
                _proseg_corr = _align_zarr_rows_to_cells(
                    _proseg_corr, _adata_p.obs_names, cells_df, "Proseg")
                _proseg_corr_imp = list(_adata_p.uns.get("split_corrected_imputed_genes", []))
                _load_zarr_boundaries_into_dict(_sdata_p, cell_bounds, "Proseg")
            except Exception as _ze2:
                print(f"  Warning: could not read Proseg zarr: {_ze2}", flush=True)

        with core._roi_lock:
            regions._roi_apply_metadata_to_df(cells_df, core._roi_state["rois"])
        _set("done", f"Loaded — {n_cells:,} cells", result={
            "cells_df": cells_df, "cell_bounds": cell_bounds,
            "out_dir": out_dir, "expr": expr_mat, "source": "proseg",
            "sdata_path": zarr_path,
            "split_corrected_expr": _proseg_corr,
            "split_corrected_imputed_genes": _proseg_corr_imp,
            "spage_result_path": _spage_rpath,
            "spage_result_genes": _spage_rgenes,
        })
        print(f"Loaded cached Proseg: {n_cells:,} cells, {len(cell_bounds):,} boundaries "
              f"from {os.path.basename(out_dir)}", flush=True)
    except jobs._JobCancelled:
        raise
    except Exception as exc:
        import traceback; traceback.print_exc()
        _set("error", str(exc)[:300])


def _run_baysor(scale: float, min_mol: int, use_prior: bool, prior_conf: float,
                x_min=None, x_max=None, y_min=None, y_max=None,
                scale_std=None, n_clusters=10, use_patches=True,
                prior_type: str = "nucleus", qv_min: float = 20) -> None:
    """Background thread: run Baysor resegmentation, using sopa patches if available."""
    _gen = jobs._dataset_gen

    def _set(status, message, result=None):
        with core._baysor_lock:
            if jobs._stale_job(_gen, status, message):
                return
            core._baysor_state["status"]  = status
            core._baysor_state["message"] = message
            if result is not None:
                core._baysor_state["result"] = result

    BAYSOR_CANDIDATES = [
        "baysor",
        os.path.expanduser("~/.julia/bin/baysor"),
        "/usr/local/bin/baysor",
    ]

    def _find_baysor():
        # `baysor --version` starts Julia: seconds when idle, far longer on a loaded
        # machine or a cold disk cache, so the probe gets a generous limit. A candidate
        # that does not exist fails immediately (FileNotFoundError), so the limit is
        # only ever waited out by a binary that is really there.
        _timed_out = None
        for candidate in BAYSOR_CANDIDATES:
            try:
                r = subprocess.run([candidate, "--version"], capture_output=True, timeout=180)
                if r.returncode == 0:
                    return candidate
            except (FileNotFoundError, PermissionError, NotADirectoryError):
                pass
            except subprocess.TimeoutExpired:
                _timed_out = _timed_out or candidate
        if _timed_out:
            # present but slow to answer: use it rather than report "not found"
            print(f"  Baysor: '{_timed_out} --version' did not answer within 180 s; "
                  f"using it anyway", flush=True)
        return _timed_out

    out_dir = None      # the run's working directory (see _seg_stage_dir)
    try:
        if not core.DATA.get("data_dir"):
            _set("error", "Resegmentation needs the original Xenium output directory "
                          "(transcripts.parquet); this dataset was " + dataview._origin_phrase() + ".")
            return
        baysor_bin = _find_baysor()
        if baysor_bin is None:
            _set("error", "baysor not found. Install from github.com/kharchenkolab/Baysor")
            return
        print(f"  Using baysor: {baysor_bin}", flush=True)

        # ── Check for sopa patches (first: the patch layout is part of the cache tag) ──
        patch_bounds = _get_patch_bounds_um() if use_patches else None
        # Intersect patches with user region filter
        if patch_bounds and any(v is not None for v in [x_min, x_max, y_min, y_max]):
            filtered = []
            for px0, py0, px1, py1 in patch_bounds:
                if x_min is not None and px1 < x_min: continue
                if x_max is not None and px0 > x_max: continue
                if y_min is not None and py1 < y_min: continue
                if y_max is not None and py0 > y_max: continue
                filtered.append((
                    max(px0, x_min or px0), max(py0, y_min or py0),
                    min(px1, x_max or px1), min(py1, y_max or py1),
                ))
            patch_bounds = filtered or None

        _A = lambda v: "A" if v is None else v
        _param_str = (f"s{scale}_ss{scale_std}_m{min_mol}_i{n_clusters}_p{int(use_prior)}"
                      f"{'_c'+str(prior_conf) if use_prior else ''}"
                      f"{'_pt'+prior_type if use_prior else ''}"
                      f"_x{_A(x_min)}-{_A(x_max)}_y{_A(y_min)}-{_A(y_max)}")
        # Appended only when it applies, so non-patch runs keep their previous tag
        if qv_min != 20:
            _param_str += f"_qv{qv_min}"
        if patch_bounds:
            _param_str += f"_pb{_patch_bounds_tag(patch_bounds)}"
        _param_tag = hashlib.md5(_param_str.encode()).hexdigest()[:8]
        final_dir = caching._seg_run_dir("baysor", _param_tag)
        out_dir = caching._seg_stage_dir(final_dir)
        print(f"  Baysor: cache dir = {os.path.basename(final_dir)} ({_param_str})", flush=True)

        # ── Transcript reader: one region at a time, filtered while reading ──
        _set("running", "Loading transcripts…")
        import pyarrow as _pa
        import pyarrow.parquet as _pq
        _tx_path = os.path.join(core.DATA["data_dir"], "transcripts.parquet")
        _tx_schema = _pq.read_schema(_tx_path)
        _tx_schema_cols = _tx_schema.names
        _tx_cols = ["x_location", "y_location", "feature_name", "cell_id"]
        if "overlaps_nucleus" in _tx_schema_cols:
            _tx_cols.append("overlaps_nucleus")
        # Dictionary-encode the string columns (no per-row Python strings)
        _tx_dict_cols = [
            c for c in ("feature_name", "cell_id")
            if _pa.types.is_string(_tx_schema.field(c).type)
            or _pa.types.is_large_string(_tx_schema.field(c).type)
            or _pa.types.is_binary(_tx_schema.field(c).type)
        ]
        # Same transcript set as the Xenium cell-feature matrix: Q-score ≥ qv_min, genes only
        _tx_filters = _tx_gene_filters(_tx_schema_cols, qv_min)
        _unassigned = {"UNASSIGNED", "nan", "None", "<NA>", ""}

        def _as_str_cat(col):
            col = col if isinstance(col.dtype, pd.CategoricalDtype) else col.astype("category")
            _cats = col.cat.categories
            if len(_cats) and isinstance(_cats[0], bytes):
                col = col.cat.rename_categories([c.decode() for c in _cats])
            return col

        def _load_tx(bx0, bx1, by0, by1):
            tx = _read_tx_region(_tx_path, _tx_cols, bx0, bx1, by0, by1,
                                 extra_filters=_tx_filters, read_dictionary=_tx_dict_cols)
            tx["feature_name"] = _as_str_cat(tx["feature_name"])
            if "is_gene" not in _tx_schema_cols:
                _fcats = pd.Index(tx["feature_name"].cat.categories.astype(str))
                tx = tx[~tx["feature_name"].isin(_fcats[_fcats.str.match(_TX_CONTROL_RE)])].copy()
            # Map Xenium string cell IDs → integers (0 = unassigned); ids only need to be
            # consistent within one Baysor run, so they are numbered per loaded region.
            _cid = _as_str_cat(tx["cell_id"])
            _bad = pd.Index(_cid.cat.categories.astype(str)).isin(_unassigned)
            # Older bundles store integer cell ids with -1 (or 0) for "unassigned"
            if pd.api.types.is_numeric_dtype(_cid.cat.categories.dtype):
                _bad = _bad | (np.asarray(_cid.cat.categories, dtype="float64") <= 0)
            _lut = np.zeros(len(_bad) + 1, dtype="int64")      # last slot: code -1 (null)
            _lut[:-1][~_bad] = np.arange(1, int((~_bad).sum()) + 1)
            tx["cell_id"] = _lut[_cid.cat.codes.to_numpy()]
            # Build nucleus_prior_id: cell_id only for transcripts overlapping a nucleus.
            # This gives Baysor a tighter nuclear prior instead of the full cell boundary prior.
            if prior_type == "nucleus":
                if "overlaps_nucleus" in tx.columns:
                    tx["nucleus_prior_id"] = (
                        tx["cell_id"].where(tx["overlaps_nucleus"].astype(bool), 0)
                    )
                    print(f"  Baysor: nucleus prior — "
                          f"{int((tx['nucleus_prior_id'] != 0).sum()):,} / {len(tx):,} "
                          f"transcripts inside nuclei", flush=True)
                else:
                    tx["nucleus_prior_id"] = tx["cell_id"]
                    print("  Baysor: overlaps_nucleus column not found — using cell_id as prior", flush=True)
            else:
                # prior_type == "cell": use all cell-assigned transcripts as prior (no nucleus restriction).
                # _baysor_run_single falls back to "cell_id" column when "nucleus_prior_id" is absent.
                print(f"  Baysor: cell boundary prior — "
                      f"{int((tx['cell_id'] != 0).sum()):,} / {len(tx):,} "
                      f"transcripts in cells", flush=True)
            return tx

        if patch_bounds:
            n_patches = len(patch_bounds)
            print(f"  Baysor: using {n_patches} sopa patch(es)", flush=True)
            ok_patches: list  = []   # indices of patches that returned cells

            all_cells: list   = []
            all_bounds: dict  = {}
            all_seg: list     = []
            # overlap buffer = cell radius; scale -1 means "estimate from prior" → use the default 20 µm
            buf = float(scale) if (scale is not None and float(scale) > 0) else 20.0

            for pi, (px0, py0, px1, py1) in enumerate(patch_bounds):
                _set("running", f"Baysor patch {pi + 1}/{n_patches}…")
                jobs._job_progress(fraction=_TOOL_SHARE * pi / n_patches)
                # Read with overlap buffer so cells at edges are fully captured
                # (clipped to the user's region filter)
                tx_patch = _load_tx(
                    px0 - buf if x_min is None else max(px0 - buf, x_min),
                    px1 + buf if x_max is None else min(px1 + buf, x_max),
                    py0 - buf if y_min is None else max(py0 - buf, y_min),
                    py1 + buf if y_max is None else min(py1 + buf, y_max),
                )
                if len(tx_patch) < min_mol * 2:
                    print(f"  Baysor: patch {pi + 1} has too few transcripts — skipping", flush=True)
                    continue
                print(f"  Baysor: patch {pi + 1}/{n_patches}: {len(tx_patch):,} transcripts", flush=True)

                patch_dir = os.path.join(out_dir, f"patch_{pi:04d}")
                try:
                    cells_i, bounds_i, seg_i = _baysor_run_single(
                        tx_patch, patch_dir, baysor_bin, scale, min_mol, use_prior, prior_conf,
                        scale_std=scale_std, n_clusters=n_clusters,
                        progress_fn=lambda f, t, _pi=pi: jobs._job_progress(
                            f"Baysor patch {_pi + 1}/{n_patches}: {t}",
                            _TOOL_SHARE * (_pi + f) / n_patches),
                    )
                except jobs._DatasetChanged:
                    raise          # cancelled (or dataset replaced): not a failed patch
                except Exception as pe:
                    print(f"  Baysor: patch {pi + 1} failed: {pe}", flush=True)
                    continue

                # Boundary resolution, step 1: keep cells whose centroid is inside this
                # patch's (non-buffered) box. Cells in the overlap with another patch are
                # resolved after the loop, once it is known which patches returned cells.
                in_patch = (
                    (cells_i["x_centroid"] >= px0) & (cells_i["x_centroid"] < px1) &
                    (cells_i["y_centroid"] >= py0) & (cells_i["y_centroid"] < py1)
                )
                cells_i = cells_i[in_patch]
                valid_ids = set(cells_i.index)

                # Remap IDs to be globally unique: "p{pi}_{original_id}"
                remap = {old: f"p{pi}_{old}" for old in valid_ids}
                cells_i.index = pd.Index([remap[c] for c in cells_i.index], name="cell_id")
                all_cells.append(cells_i)
                ok_patches.append(pi)
                for old_id, bnd in bounds_i.items():
                    if old_id in valid_ids:
                        all_bounds[remap[old_id]] = bnd
                seg_i_valid = seg_i[seg_i["cell"].isin(valid_ids)].copy()
                if "gene" in seg_i_valid.columns:
                    # keep per-(cell, gene) counts only, not every transcript of every patch
                    seg_i_valid = (seg_i_valid.groupby(["cell", "gene"], observed=True)
                                   .size().rename("n").reset_index())
                seg_i_valid["cell"] = seg_i_valid["cell"].map(remap)
                all_seg.append(seg_i_valid)

            if not all_cells:
                _set("error", "No cells found across all patches. Check Baysor output.")
                return

            # Boundary resolution, step 2: a centroid inside several successful patches is
            # kept by the patch it lies deepest in (no cell kept twice, none ceded to a
            # skipped or failed patch)
            all_cells = [
                c[_patch_owned_mask(c["x_centroid"].values, c["y_centroid"].values,
                                    _pi, patch_bounds, ok_patches)]
                for _pi, c in zip(ok_patches, all_cells)
            ]
            cells_df  = pd.concat(all_cells)
            # Step 3: one biological cell segmented by two patches with its centroids
            # on either side of the ownership line is kept once
            _seam_keep = _patch_seam_dedup_mask(
                cells_df.index, np.repeat(ok_patches, [len(c) for c in all_cells]),
                cells_df["x_centroid"].values, cells_df["y_centroid"].values,
                cells_df["transcript_counts"].values, all_bounds, patch_bounds, ok_patches)
            if not _seam_keep.all():
                print(f"  Baysor: dropped {int((~_seam_keep).sum()):,} seam duplicate(s)", flush=True)
                cells_df = cells_df[_seam_keep]
            _owned    = set(cells_df.index)
            cell_bounds = {k: v for k, v in all_bounds.items() if k in _owned}
            seg_merged  = pd.concat(all_seg)
            seg_merged  = seg_merged[seg_merged["cell"].isin(_owned)]
            print(f"  Baysor: merged {len(cells_df):,} cells from {n_patches} patch(es)", flush=True)

        else:
            # ── Single run (no patches or no sopa data) ───────────────────
            tx_all = _load_tx(x_min, x_max, y_min, y_max)
            n_assigned = int((tx_all["cell_id"] != 0).sum())
            print(f"  Baysor: {n_assigned:,} / {len(tx_all):,} transcripts have prior cell assignment", flush=True)
            _up = use_prior and n_assigned > 0
            if use_prior and not _up:
                print("  Baysor: no assigned transcripts in region — disabling prior segmentation", flush=True)
            _set("running", "Running Baysor (this may take 10–30 min)…")
            try:
                cells_df, cell_bounds, seg_merged = _baysor_run_single(
                    tx_all, out_dir, baysor_bin, scale, min_mol, _up, prior_conf,
                    scale_std=scale_std, n_clusters=n_clusters,
                    progress_fn=lambda f, t: jobs._job_progress(f"Baysor: {t}", _TOOL_SHARE * f),
                )
            except jobs._DatasetChanged:
                raise
            except Exception as e:
                _set("error", str(e)[:300])
                return

        # ── Build expression matrix ───────────────────────────────────────
        _set("running", "Building expression matrix…")
        jobs._job_progress(fraction=_TOOL_SHARE)
        expr_mat = _build_baysor_expr(seg_merged, cells_df)

        n_cells = len(cells_df)
        jobs._check_dataset_gen(_gen)   # params.json / zarr below describe this run's dataset
        import json as _json
        _params_out = {
            "tool": "baysor", "scale": scale, "scale_std": scale_std,
            "min_mol": min_mol, "n_clusters": n_clusters,
            "use_prior": use_prior, "prior_conf": prior_conf,
            "prior_type": prior_type, "qv_min": qv_min,
            "n_patches": len(patch_bounds) if patch_bounds else 0,
            # layout by patch_XXXX index: lets a cache load redo the overlap resolution
            "patch_bounds": [[float(v) for v in b] for b in patch_bounds] if patch_bounds else None,
            # patches that returned cells, and the seam de-duplication applied after ownership
            "patch_ok": [int(p) for p in ok_patches] if patch_bounds else None,
            "seam_dedup": dict(_SEAM_DEDUP) if patch_bounds else None,
            "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
            "n_cells": n_cells, "param_tag": _param_tag,
            "dataset_hash": hashlib.md5(caching._dataset_key().encode()).hexdigest()[:8],
        }
        with open(os.path.join(out_dir, "params.json"), "w") as _f:
            _json.dump(_params_out, _f)
        # SpaGE result of a previous run in this dir belongs to other cells
        if os.path.exists(os.path.join(out_dir, "spage_result.json")):
            os.remove(os.path.join(out_dir, "spage_result.json"))
        _set("running", "Computing UMAP and clusters…")
        jobs._job_progress(fraction=0.92)
        clustering._compute_reseg_clusters_umap(cells_df, expr_mat,
                                     progress_fn=lambda msg: _set("running", msg))
        # Normalize cell_bounds keys to strings (GeoJSON may produce int keys)
        cell_bounds = {str(k): v for k, v in cell_bounds.items()}
        _set("running", "Writing SpatialData zarr…")
        _gene_names = core.DATA["gene_names"]
        jobs._check_dataset_gen(_gen)
        # last cancellation point: from here the run is stored and published
        jobs._job_shield()
        jobs._job_progress(fraction=0.97)
        zarr_path = _build_reseg_sdata(
            cells_df, cell_bounds, expr_mat, _gene_names, "baysor", out_dir
        )
        caching._seg_stage_commit(out_dir, final_dir)
        if zarr_path:
            zarr_path = os.path.join(final_dir, os.path.basename(zarr_path))
        out_dir = final_dir
        _set("done", f"Done — {n_cells:,} cells", result={
            "cells_df":    cells_df,
            "cell_bounds": cell_bounds,
            "out_dir":     out_dir,
            "expr":        expr_mat,
            "source":      "baysor",
            "sdata_path":  zarr_path,
        })
        print(f"  Baysor: {n_cells:,} cells segmented", flush=True)

    except jobs._JobCancelled:
        raise                      # reported by _job_finish
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set("error", str(exc)[:300])
    finally:
        caching._seg_stage_discard(out_dir)


# ─── Proseg segmentation ──────────────────────────────────────────────────────

def _run_proseg(voxel_size=None, n_threads=None, n_samples=None,
                recorded_samples=None, schedule=None,
                nuclear_reassign_prob=None, prior_seg_prob=None,
                x_min=None, x_max=None, y_min=None, y_max=None,
                prior_type: str = "nucleus", use_patches: bool = True) -> None:
    """Background thread: run Proseg resegmentation, using sopa patches if available
    (use_patches=False: one run on the whole region even when patches exist)."""
    _gen = jobs._dataset_gen

    def _set(status, message, result=None):
        with core._proseg_lock:
            if jobs._stale_job(_gen, status, message):
                return
            core._proseg_state["status"]  = status
            core._proseg_state["message"] = message
            if result is not None:
                core._proseg_state["result"] = result

    PROSEG_CANDIDATES = [
        "proseg",
        os.path.expanduser("~/.cargo/bin/proseg"),
        "/usr/local/bin/proseg",
    ]

    def _find_proseg():
        import shutil
        found = shutil.which("proseg")
        if found:
            return found
        for candidate in PROSEG_CANDIDATES:
            try:
                r = subprocess.run([candidate, "--version"], capture_output=True, timeout=10)
                if r.returncode == 0:
                    return candidate
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        return None

    out_dir = None      # the run's working directory (see _seg_stage_dir)
    try:
        # a stale "none" (option removed from the UI) runs as the default prior and
        # therefore shares the default prior's cache tag
        prior_type = _proseg_prior_type(prior_type)
        if not core.DATA.get("data_dir"):
            _set("error", "Resegmentation needs the original Xenium output directory "
                          "(transcripts.parquet); this dataset was " + dataview._origin_phrase() + ".")
            return
        proseg_bin = _find_proseg()
        if proseg_bin is None:
            _set("error", "proseg not found. Install via: conda install bioconda::rust-proseg")
            return
        print(f"  Using proseg: {proseg_bin}", flush=True)

        tx_src = os.path.join(core.DATA["data_dir"], "transcripts.parquet")
        # Same transcript set as the Xenium cell-feature matrix and the Baysor path:
        # Q-score ≥ 20, genes only. Proseg has no option to exclude features, and its
        # --xenium preset keeps control / unassigned codewords, so the input is filtered here.
        import pyarrow.parquet as _pq
        _tx_schema_cols = _pq.read_schema(tx_src).names
        _tx_filters = _tx_gene_filters(_tx_schema_cols, 20)

        def _load_tx(bx0, bx1, by0, by1):
            tx = _read_tx_region(tx_src, None, bx0, bx1, by0, by1, extra_filters=_tx_filters)
            if "is_gene" not in _tx_schema_cols:
                tx = _tx_drop_control_features(tx)
            return tx

        # ── Check for sopa patches (first: the patch layout is part of the cache tag) ──
        patch_bounds = _get_patch_bounds_um() if use_patches else None
        if patch_bounds and any(v is not None for v in [x_min, x_max, y_min, y_max]):
            filtered = []
            for px0, py0, px1, py1 in patch_bounds:
                if x_min is not None and px1 < x_min: continue
                if x_max is not None and px0 > x_max: continue
                if y_min is not None and py1 < y_min: continue
                if y_max is not None and py0 > y_max: continue
                filtered.append((
                    max(px0, x_min or px0), max(py0, y_min or py0),
                    min(px1, x_max or px1), min(py1, y_max or py1),
                ))
            patch_bounds = filtered or None

        _A = lambda v: "A" if v is None else v
        _param_str = (f"v{voxel_size or 'A'}_t{n_threads or 'A'}"
                      f"_x{_A(x_min)}-{_A(x_max)}_y{_A(y_min)}-{_A(y_max)}")
        # Every option that changes the result is part of the tag; appended only when
        # non-default so runs made with the defaults keep their previous tag.
        for _k, _v in (("nc", None if n_samples in (None, 10) else n_samples),
                       ("rs", recorded_samples or None), ("sch", schedule or None),
                       ("nr", nuclear_reassign_prob), ("ps", prior_seg_prob),
                       ("pt", None if prior_type == "nucleus" else prior_type)):
            if _v is not None:
                _param_str += f"_{_k}{_v}"
        if patch_bounds:
            _param_str += f"_pb{_patch_bounds_tag(patch_bounds)}"
        _param_tag = hashlib.md5(_param_str.encode()).hexdigest()[:8]
        final_dir = caching._seg_run_dir("proseg", _param_tag)
        out_dir = caching._seg_stage_dir(final_dir)
        print(f"  Proseg: cache dir = {os.path.basename(final_dir)} ({_param_str})", flush=True)

        if patch_bounds:
            n_patches = len(patch_bounds)
            print(f"  Proseg: using {n_patches} sopa patch(es)", flush=True)
            ok_patches: list  = []   # indices of patches that returned cells

            all_cells: list   = []
            all_bounds: dict  = {}
            all_tx_meta: list = []
            CELL_ID_STRIDE    = 10_000_000  # max cells per patch
            cell_offset       = 1

            for pi, (px0, py0, px1, py1) in enumerate(patch_bounds):
                _set("running", f"Proseg patch {pi + 1}/{n_patches}…")
                jobs._job_progress(fraction=_TOOL_SHARE * pi / n_patches)
                # One patch at a time (patch bounds are already clipped to the user's region)
                tx_patch = _load_tx(px0, px1, py0, py1)
                if len(tx_patch) < 10:
                    print(f"  Proseg: patch {pi + 1} has too few transcripts — skipping", flush=True)
                    continue
                print(f"  Proseg: patch {pi + 1}/{n_patches}: {len(tx_patch):,} transcripts", flush=True)

                patch_dir = os.path.join(out_dir, f"patch_{pi:04d}")
                try:
                    cells_i, bounds_i, tx_meta_i = _proseg_run_single(
                        tx_patch, patch_dir, proseg_bin, voxel_size, n_threads, n_samples,
                        recorded_samples=recorded_samples, schedule=schedule,
                        nuclear_reassign_prob=nuclear_reassign_prob, prior_seg_prob=prior_seg_prob,
                        prior_type=prior_type,
                        progress_fn=lambda f, t, _pi=pi: jobs._job_progress(
                            f"Proseg patch {_pi + 1}/{n_patches}: {t}",
                            _TOOL_SHARE * (_pi + f) / n_patches),
                    )
                except jobs._DatasetChanged:
                    raise          # cancelled (or dataset replaced): not a failed patch
                except Exception as pe:
                    print(f"  Proseg: patch {pi + 1} failed: {pe}", flush=True)
                    continue

                # Boundary resolution, step 1: keep cells with centroid in this patch's box.
                # Cells in the overlap with another patch are resolved after the loop, once
                # it is known which patches returned cells.
                in_patch = (
                    (cells_i["x_centroid"] >= px0) & (cells_i["x_centroid"] < px1) &
                    (cells_i["y_centroid"] >= py0) & (cells_i["y_centroid"] < py1)
                )
                cells_i  = cells_i[in_patch]
                valid_ids = set(cells_i.index)

                # Remap integer cell IDs with per-patch stride
                remap = {old: old + cell_offset for old in valid_ids}
                cells_i.index = pd.Index([remap[c] for c in cells_i.index], name="cell_id")
                all_cells.append(cells_i)
                ok_patches.append(pi)
                for old_id, bnd in bounds_i.items():
                    if old_id in valid_ids:
                        all_bounds[remap[old_id]] = bnd
                _ccol = (None if tx_meta_i is None or "gene" not in tx_meta_i.columns
                         else "cell" if "cell" in tx_meta_i.columns
                         else "assignment" if "assignment" in tx_meta_i.columns else None)
                if _ccol:
                    # Keep only transcripts of the cells this patch owns (a rejected cell's
                    # local id must not be matched to another cell's global id), remapped
                    # like the cells and reduced to per-(cell, gene) counts.
                    _asgn = pd.to_numeric(tx_meta_i[_ccol], errors="coerce")
                    _valid_mask = _asgn.isin(valid_ids)
                    all_tx_meta.append(
                        pd.DataFrame({_ccol: _asgn[_valid_mask].astype("int64") + cell_offset,
                                      "gene": tx_meta_i.loc[_valid_mask, "gene"]})
                        .groupby([_ccol, "gene"], observed=True).size().rename("n").reset_index()
                    )
                cell_offset += CELL_ID_STRIDE

            if not all_cells:
                _set("error", "No cells found across all patches. Check Proseg output.")
                return

            # Boundary resolution, step 2: a centroid inside several successful patches is
            # kept by the patch it lies deepest in (no cell kept twice, none ceded to a
            # skipped or failed patch)
            all_cells = [
                c[_patch_owned_mask(c["x_centroid"].values, c["y_centroid"].values,
                                    _pi, patch_bounds, ok_patches)]
                for _pi, c in zip(ok_patches, all_cells)
            ]
            cells_df       = pd.concat(all_cells)
            # Step 3: one biological cell segmented by two patches with its centroids
            # on either side of the ownership line is kept once
            _seam_keep = _patch_seam_dedup_mask(
                cells_df.index, np.repeat(ok_patches, [len(c) for c in all_cells]),
                cells_df["x_centroid"].values, cells_df["y_centroid"].values,
                cells_df["transcript_counts"].values, all_bounds, patch_bounds, ok_patches)
            if not _seam_keep.all():
                print(f"  Proseg: dropped {int((~_seam_keep).sum()):,} seam duplicate(s)", flush=True)
                cells_df = cells_df[_seam_keep]
            _owned         = set(cells_df.index)
            cell_bounds    = {k: v for k, v in all_bounds.items() if k in _owned}
            all_tx_meta    = [t[t.iloc[:, 0].isin(_owned)] for t in all_tx_meta]
            tx_meta_merged = pd.concat(all_tx_meta) if all_tx_meta else None
            print(f"  Proseg: merged {len(cells_df):,} cells from {n_patches} patch(es)", flush=True)

        else:
            # ── Single run (no patches or no sopa data) ───────────────────
            if any(v is not None for v in [x_min, x_max, y_min, y_max]):
                _set("running", "Filtering transcripts to region…")
                tx = _load_tx(x_min, x_max, y_min, y_max)
                print(f"  Proseg: filtered to {len(tx):,} transcripts", flush=True)
            elif prior_type == "nucleus":
                # whole slide, columns unmodified: stream the gene-only Q ≥ 20 rows to a
                # file for Proseg instead of loading the slide into memory
                _set("running", "Filtering transcripts (genes, Q ≥ 20)…")
                tx = os.path.join(out_dir, "transcripts.parquet")
                _n_tx = _tx_write_gene_only(tx_src, tx, 20)
                print(f"  Proseg: {_n_tx:,} gene transcripts with Q ≥ 20 written", flush=True)
            else:
                tx = _load_tx(None, None, None, None)

            _set("running", "Running Proseg…")
            try:
                cells_df, cell_bounds, tx_meta_merged = _proseg_run_single(
                    tx, out_dir, proseg_bin, voxel_size, n_threads, n_samples,
                    recorded_samples=recorded_samples, schedule=schedule,
                    nuclear_reassign_prob=nuclear_reassign_prob, prior_seg_prob=prior_seg_prob,
                    prior_type=prior_type,
                    progress_fn=lambda f, t: jobs._job_progress(f"Proseg: {t}", _TOOL_SHARE * f),
                )
            except jobs._DatasetChanged:
                raise
            except Exception as e:
                _set("error", str(e)[:300])
                return
            finally:
                # the whole-slide filtered copy is large and can be rebuilt from the bundle
                if isinstance(tx, str) and tx != tx_src and os.path.exists(tx):
                    os.remove(tx)

        # ── Build expression matrix ───────────────────────────────────────
        _set("running", "Building expression matrix…")
        jobs._job_progress(fraction=_TOOL_SHARE)
        expr_mat = _build_proseg_expr(tx_meta_merged, cells_df)
        # transcript_counts = what the matrix holds (the patch merge above still ranks
        # seam duplicates by Proseg's `population`, like the slow cache load does)
        cells_df = cells_df.copy()
        _proseg_counts_from_expr(cells_df, expr_mat)

        n_cells = len(cells_df)
        jobs._check_dataset_gen(_gen)   # params.json / zarr below describe this run's dataset
        import json as _json
        _params_out = {
            "tool": "proseg", "voxel_size": voxel_size, "n_threads": n_threads,
            "n_samples": n_samples, "recorded_samples": recorded_samples,
            "schedule": schedule, "nuclear_reassign_prob": nuclear_reassign_prob,
            "prior_seg_prob": prior_seg_prob, "prior_type": prior_type,
            "n_patches": len(patch_bounds) if patch_bounds else 0,
            # layout by patch_XXXX index: lets a cache load redo the overlap resolution
            "patch_bounds": [[float(v) for v in b] for b in patch_bounds] if patch_bounds else None,
            # patches that returned cells, and the seam de-duplication applied after ownership
            "patch_ok": [int(p) for p in ok_patches] if patch_bounds else None,
            "seam_dedup": dict(_SEAM_DEDUP) if patch_bounds else None,
            "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
            "n_cells": n_cells, "param_tag": _param_tag,
            "dataset_hash": hashlib.md5(caching._dataset_key().encode()).hexdigest()[:8],
        }
        with open(os.path.join(out_dir, "params.json"), "w") as _f:
            _json.dump(_params_out, _f)
        # SpaGE result of a previous run in this dir belongs to other cells
        if os.path.exists(os.path.join(out_dir, "spage_result.json")):
            os.remove(os.path.join(out_dir, "spage_result.json"))
        _set("running", "Computing UMAP and clusters…")
        jobs._job_progress(fraction=0.92)
        clustering._compute_reseg_clusters_umap(cells_df, expr_mat,
                                     progress_fn=lambda msg: _set("running", msg))
        # Normalize cell_bounds keys to strings (GeoJSON may produce int keys)
        cell_bounds = {str(k): v for k, v in cell_bounds.items()}
        _set("running", "Writing SpatialData zarr…")
        _gene_names = core.DATA["gene_names"]
        jobs._check_dataset_gen(_gen)
        # last cancellation point: from here the run is stored and published
        jobs._job_shield()
        jobs._job_progress(fraction=0.97)
        zarr_path = _build_reseg_sdata(
            cells_df, cell_bounds, expr_mat, _gene_names, "proseg", out_dir
        )
        caching._seg_stage_commit(out_dir, final_dir)
        if zarr_path:
            zarr_path = os.path.join(final_dir, os.path.basename(zarr_path))
        out_dir = final_dir
        # Same string index as the cache-load paths, so plot customdata (str ids)
        # matches for lasso ROIs / ROI metadata in the session that ran Proseg.
        cells_df.index = cells_df.index.astype(str)
        _set("done", f"Done — {n_cells:,} cells", result={
            "cells_df":    cells_df,
            "cell_bounds": cell_bounds,
            "out_dir":     out_dir,
            "expr":        expr_mat,
            "source":      "proseg",
            "sdata_path":  zarr_path,
        })
        print(f"  Proseg: {n_cells:,} cells segmented", flush=True)

    except jobs._JobCancelled:
        raise                      # reported by _job_finish
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set("error", str(exc)[:300])
    finally:
        caching._seg_stage_discard(out_dir)
