"""SPLIT ambient RNA correction (RCTD doublet mode + SPLIT::purify through rpy2)."""

import os
import hashlib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import core, jobs, caching, dataview, reseg, clustering
from . import provenance


def _compute_split_clusters_umap(cells_df: pd.DataFrame, corrected_mat,
                                  corrected_cell_ids=None, progress_fn=None) -> None:
    """
    Compute KMeans clusters and UMAP on SPLIT-corrected counts.
    Writes cluster_split_10, split_umap_1, split_umap_2 columns into cells_df in-place.
    """
    try:
        import scipy.sparse as sp_
        from sklearn.preprocessing import normalize
        from sklearn.decomposition import TruncatedSVD
        from sklearn.cluster import KMeans
        try:
            import sys as _sys_umap, types as _types_umap
            if "tensorflow" not in _sys_umap.modules:
                _sys_umap.modules["tensorflow"] = _types_umap.ModuleType("tensorflow")
            from umap.umap_ import UMAP as _UMAP
            _have_umap = True
        except Exception:
            _have_umap = False

        n_cells, n_genes = corrected_mat.shape
        if progress_fn:
            progress_fn(f"SPLIT: normalizing {n_cells:,} cells × {n_genes} genes…")

        mat = corrected_mat if sp_.issparse(corrected_mat) else sp_.csr_matrix(corrected_mat)
        # Log-normalize
        log_mat = mat.copy().astype(np.float32)
        log_mat.data = np.log1p(log_mat.data)
        normed = normalize(log_mat, norm="l2", axis=1)

        # PCA via SVD
        n_comp = min(50, n_cells - 1, n_genes - 1)
        if progress_fn:
            progress_fn(f"SPLIT: SVD ({n_comp} components)…")
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        pca = svd.fit_transform(normed)

        # KMeans
        if progress_fn:
            progress_fn("SPLIT: KMeans clustering (k=10)…")
        if n_cells > clustering._UMAP_PARALLEL_MIN_CELLS:
            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(n_clusters=10, random_state=42, n_init=3)
        else:
            km = KMeans(n_clusters=10, random_state=42, n_init=10)
        labels = km.fit_predict(pca)

        # UMAP
        embedding = None
        if _have_umap:
            if progress_fn:
                progress_fn("SPLIT: computing UMAP…")
            try:
                n_neighbors = min(30, n_cells - 1)
                reducer = _UMAP(n_components=2, n_neighbors=n_neighbors,
                                    min_dist=0.3, **clustering._umap_seed_kwargs(n_cells, 42))
                embedding = reducer.fit_transform(pca)
                print(f"  [SPLIT UMAP] computed embedding shape={embedding.shape}", flush=True)
            except Exception as _umap_err:
                print(f"  [SPLIT UMAP] FAILED: {_umap_err}", flush=True)
                import traceback; traceback.print_exc()
        else:
            print("  [SPLIT UMAP] umap-learn not available, skipping", flush=True)

        # Write back to cells_df — handle subset case (RCTD may drop low-UMI cells)
        if corrected_cell_ids is not None and len(corrected_cell_ids) < len(cells_df):
            try:
                sample_id = cells_df.index[0]
                cast = type(sample_id)
                int_idx = pd.Index([cast(c) for c in corrected_cell_ids])
            except Exception:
                int_idx = pd.Index(corrected_cell_ids)
            cells_df["cluster_split_10"] = pd.array([pd.NA] * len(cells_df), dtype="string")
            cells_df.loc[int_idx, "cluster_split_10"] = (labels + 1).astype(str)
            if embedding is not None:
                cells_df["split_umap_1"] = np.nan
                cells_df["split_umap_2"] = np.nan
                cells_df.loc[int_idx, "split_umap_1"] = embedding[:, 0].astype(np.float32)
                cells_df.loc[int_idx, "split_umap_2"] = embedding[:, 1].astype(np.float32)
        else:
            cells_df["cluster_split_10"] = (labels + 1).astype(str)
            if embedding is not None:
                cells_df["split_umap_1"] = embedding[:, 0].astype(np.float32)
                cells_df["split_umap_2"] = embedding[:, 1].astype(np.float32)
        if not _have_umap and progress_fn:
            progress_fn("SPLIT: umap-learn not available, skipping UMAP.")
    except Exception:
        import traceback
        traceback.print_exc()


def _split_write_zarr(sdata_path: str, corrected_mat, cells_df: pd.DataFrame,
                       gene_names: list, corrected_cell_ids=None) -> None:
    """Write SPLIT corrected counts + cluster/UMAP obs columns to a SpatialData zarr.
    Uses zarr low-level API to avoid spatialdata 'target path in use' error."""
    try:
        import zarr as _zarr_mod
        import scipy.sparse as sp_
        try:
            from anndata.io import write_elem as _write_elem
        except ImportError:
            from anndata.experimental import write_elem as _write_elem

        # (per-store write lock, not _sdata_lock: that one is taken by the figure and
        # patch callbacks, which would wait for the whole X_corrected write)
        with reseg._zarr_write_lock(sdata_path):
            zroot = _zarr_mod.open_group(sdata_path, mode="r+", use_consolidated=False)
            if "tables" not in zroot or "table" not in zroot["tables"]:
                print("  SPLIT: no table in zarr — skipping zarr write", flush=True)
                return
            table_grp = zroot["tables"]["table"]
            obs_grp   = table_grp["obs"]

            # Read zarr obs index for alignment
            idx_key  = obs_grp.attrs.get("_index", "_index")
            zarr_ids = [str(v) for v in obs_grp[idx_key][:]]

            # Align corrected_mat rows to zarr obs order
            # Use corrected_cell_ids (from R colnames) when available — RCTD may drop cells
            # (ignored when they do not describe the matrix rows one to one)
            if corrected_cell_ids is not None and len(corrected_cell_ids) == corrected_mat.shape[0]:
                id_to_row = {str(cid): i for i, cid in enumerate(corrected_cell_ids)}
            else:
                id_to_row = {str(cid): i for i, cid in enumerate(cells_df.index)}
            # Strip patch prefix (e.g. "p0_") from zarr IDs — live patch runs add this
            # prefix but corrected_cell_ids come from R colnames which use bare Baysor IDs.
            import re as _re
            _stripped_zarr_ids = [_re.sub(r'^p\d+_', '', cid) for cid in zarr_ids]
            _lookup_ids = _stripped_zarr_ids if (
                len(set(_stripped_zarr_ids[:30]) & set(id_to_row)) >
                len(set(zarr_ids[:30]) & set(id_to_row))
            ) else zarr_ids
            row_indices = [id_to_row.get(cid, -1) for cid in _lookup_ids]
            mat         = corrected_mat if sp_.issparse(corrected_mat) \
                          else sp_.csr_matrix(corrected_mat)
            # Build aligned matrix; -1 means cell was dropped by RCTD → zero row
            valid_out = [i for i, r in enumerate(row_indices) if r >= 0]
            valid_in  = [r for r in row_indices if r >= 0]
            n_zarr    = len(zarr_ids)
            if valid_in:
                mat_sub = mat[valid_in, :]
                coo     = mat_sub.tocoo()
                new_row = np.array(valid_out)[coo.row]
                mat_aligned = sp_.csr_matrix(
                    (coo.data, (new_row, coo.col)),
                    shape=(n_zarr, mat.shape[1])
                )
            else:
                mat_aligned = sp_.csr_matrix((n_zarr, mat.shape[1]), dtype=mat.dtype)
            print(f"  SPLIT: aligned {len(valid_in)}/{n_zarr} cells for zarr write", flush=True)

            # Pad X_corrected to match zarr var width (panel genes only; zarr var may include
            # SpaGE-imputed genes appended later). Imputed columns get zeros.
            var_grp = table_grp["var"]
            try:
                var_key = var_grp.attrs.get("_index", "_index")
                n_var   = len(var_grp[var_key])
            except (KeyError, Exception):
                # Fallback: read n_var from X matrix shape attribute
                try:
                    _x_grp = table_grp["X"]
                    if isinstance(_x_grp, _zarr_mod.Array):
                        n_var = _x_grp.shape[1]
                    else:
                        n_var = _x_grp.attrs.get("shape", [0, mat_aligned.shape[1]])[1]
                except Exception:
                    n_var = mat_aligned.shape[1]
            if mat_aligned.shape[1] < n_var:
                n_pad       = n_var - mat_aligned.shape[1]
                mat_aligned = sp_.hstack(
                    [mat_aligned, sp_.csr_matrix((n_zarr, n_pad), dtype=mat_aligned.dtype)],
                    format="csr",
                )
                print(f"  SPLIT: padded X_corrected to {mat_aligned.shape} "
                      f"(+{n_pad} imputed gene cols = 0)", flush=True)

            # Write X_corrected layer — use_consolidated=False bypasses consolidated metadata.
            # Delete first so reruns with new parameters fully replace old results.
            layers_grp = table_grp.require_group("layers")
            if "X_corrected" in layers_grp:
                del layers_grp["X_corrected"]
            _write_elem(layers_grp, "X_corrected", mat_aligned)

            # Write obs columns — delete then rewrite so reruns fully replace old values
            for col in ["cluster_split_10", "split_umap_1", "split_umap_2"]:
                if col in obs_grp:
                    del obs_grp[col]
            for col in ["cluster_split_10", "split_umap_1", "split_umap_2"]:
                if col not in cells_df.columns:
                    continue
                # Ensure index types match for reindex — cells_df may have int index,
                # zarr_ids are strings.  Align both to string.
                _cdf_col = cells_df[col].copy()
                _cdf_col.index = _cdf_col.index.astype(str)
                ser = _cdf_col.reindex(zarr_ids)
                if cells_df[col].dtype == object or pd.api.types.is_string_dtype(cells_df[col]):
                    arr = np.array(ser.fillna("").values, dtype=str)
                    obs_grp.create_dataset(col, data=arr, shape=arr.shape,
                                           dtype=arr.dtype, overwrite=True)
                else:
                    arr = ser.values
                    obs_grp.create_dataset(col, data=arr, shape=arr.shape,
                                           dtype=arr.dtype, overwrite=True)

        print(f"  SPLIT: wrote corrected layer to {os.path.basename(sdata_path)}", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()


def _r_dgcmatrix_to_scipy(r_mat):
    """Convert an R dgCMatrix to a scipy CSC matrix (genes × cells).

    dgCMatrix is column-sparse (CSC): p = column pointers, i = row indices.
    Callers that need cells × genes should call .T.tocsr() on the result.
    """
    import scipy.sparse as sp_
    import numpy as np
    i = np.array(r_mat.slots["i"])
    p = np.array(r_mat.slots["p"])
    x = np.array(r_mat.slots["x"])
    dims = list(r_mat.slots["Dim"])
    return sp_.csc_matrix((x, i, p), shape=(dims[0], dims[1]))


def _scipy_csc_parts(mat_cells_x_genes):
    """genes × cells CSC parts (i, p, x, shape) for an R dgCMatrix.
    i and p are 0-based int32 (dgCMatrix slots are 0-based), x is float64."""
    m = sp.csc_matrix(mat_cells_x_genes.T, dtype=np.float64, copy=True)
    m.sum_duplicates()
    m.eliminate_zeros()
    m.sort_indices()
    if m.nnz >= 2**31 - 1:
        raise RuntimeError(f"Matrix has {m.nnz:,} non-zeros — too many for an R dgCMatrix "
                           f"(limit 2^31-1). Subset the cells first.")
    return (np.ascontiguousarray(m.indices, dtype=np.int32),
            np.ascontiguousarray(m.indptr, dtype=np.int32),
            np.ascontiguousarray(m.data, dtype=np.float64),
            m.shape)


def _scipy_to_r_dgc(mat_cells_x_genes, rownames, colnames, r_name: str) -> None:
    """Assign a cells × genes scipy matrix to the R global `r_name` as a
    genes × cells dgCMatrix, passing only the sparse parts (never dense)."""
    import rpy2.robjects as ro
    i, p, x, shape = _scipy_csc_parts(mat_cells_x_genes)
    if len(rownames) != shape[0] or len(colnames) != shape[1]:
        raise ValueError(f"dimnames {len(rownames)}×{len(colnames)} do not match matrix {shape}")
    ro.r.assign("._dgc_i", ro.IntVector(i))
    ro.r.assign("._dgc_p", ro.IntVector(p))
    ro.r.assign("._dgc_x", ro.FloatVector(x))
    ro.r.assign("._dgc_rn", ro.StrVector(list(rownames)))
    ro.r.assign("._dgc_cn", ro.StrVector(list(colnames)))
    ro.r(f"""
{r_name} <- Matrix::sparseMatrix(i=as.integer(._dgc_i), p=as.integer(._dgc_p),
    x=as.numeric(._dgc_x), dims=c({int(shape[0])}L, {int(shape[1])}L), index1=FALSE,
    dimnames=list(._dgc_rn, ._dgc_cn))
rm(._dgc_i, ._dgc_p, ._dgc_x, ._dgc_rn, ._dgc_cn)
""")


def _split_onto_cells(mat, job_df, cur_df, cols=("cluster_split_10", "split_umap_1", "split_umap_2")) -> tuple:
    """(matrix, {column: values}) of a SPLIT result for the cells of *cur_df*.

    *mat* has one row per cell of *job_df* (the table the job read at its start). When
    *cur_df* is still that table the inputs are returned as they are; otherwise each cell
    of *cur_df* gets the row and the column values of the cell with its id (an empty row
    and no value when the job did not see it). Raises RuntimeError before anything is
    stored when the rows cannot be told apart by id."""
    import scipy.sparse as sp_
    have = [c for c in cols if c in job_df.columns]
    if cur_df is job_df or cur_df.index.equals(job_df.index):
        return mat, {c: job_df[c].values for c in have}
    job_ids = pd.Index(job_df.index.astype(str))
    if not job_ids.is_unique or mat.shape[0] != len(job_ids):
        raise RuntimeError("The cells of the run changed while SPLIT was running and its result "
                           "cannot be matched to them by cell id. Run SPLIT again.")
    cur_ids = pd.Index(cur_df.index.astype(str))
    pos = job_ids.get_indexer(cur_ids)
    ok = pos >= 0
    out = sp_.diags(ok.astype(np.float32)).dot(sp_.csr_matrix(mat)[np.where(ok, pos, 0)]).tocsr()
    out.eliminate_zeros()
    print(f"  SPLIT: the run's cell table changed during the job - result matched by cell id "
          f"({int(ok.sum()):,} of {len(cur_ids):,} cells)", flush=True)
    return out, {c: pd.Series(job_df[c].values, index=job_ids).reindex(cur_ids).values for c in have}


def _split_publish_labels(state: dict, labels, mode_key: str, generic_key: str, partial: bool) -> list:
    """Put the labels of SPLIT's own RCTD run into the annotation state (call inside
    core._annot_lock); returns the keys written.

    partial (the run saw the cells of an active subset() only): a key that already holds
    labels is left alone - labels of part of the cells never replace an annotation of all
    of them. Otherwise both keys are replaced, and the RCTD weights stored under them are
    removed: they belong to the annotation run whose labels were just replaced (the rule
    of annotation._annot_autoload: weights only beside the labels of their own run)."""
    written = []
    for k in dict.fromkeys((mode_key, generic_key)):
        if partial and state.get(k) is not None:
            continue
        state[k] = labels
        state[f"rctd_weights_{k}"] = None
        written.append(k)
    return written


def _run_split_correction(rds_path: str, label_col: str = "Names",
                           max_cores: int = 4, seg_source: str = "xenium",
                           min_umi: int = 10, min_umi_sigma: int = 100,
                           purify_singlets: bool = True) -> None:
    """
    Background thread: run RCTD (doublet mode) + SPLIT::purify() for ambient RNA correction.
    Requires R packages: spacexr, SPLIT (bdsc-tds/SPLIT).
    """
    print(f"  SPLIT: starting — min_umi={min_umi}, min_umi_sigma={min_umi_sigma}, "
          f"max_cores={max_cores}, purify_singlets={purify_singlets}, "
          f"seg_source={seg_source!r}", flush=True)
    core._redirect_rpy2_console()
    _gen = jobs._dataset_gen

    def _set(status, message, result=None):
        with core._split_lock:
            if jobs._stale_job(_gen, status, message):
                return
            core._split_state["status"]  = status
            core._split_state["message"] = message
            if result is not None:
                core._split_state["result"] = result
        if status == "done":
            provenance._prov_record("split", seg_source=seg_source, params={"reference": os.path.basename(str(rds_path or '')), "label_col": label_col, "max_cores": max_cores, "min_umi": min_umi, "min_umi_sigma": min_umi_sigma, "purify_singlets": purify_singlets}, r_packages=("spacexr", "SPLIT", "Seurat", "SeuratObject"), message=message, result=result)

    try:
        import rpy2.robjects as ro
        import rpy2.robjects.conversion as _rconv
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        import scipy.sparse as sp_
        import scipy.io as sio
        import tempfile

        # rpy2 thread fix
        _rconv.set_conversion(ro.default_converter)
        pandas2ri.activate()

        # Clear stale "Rerun UMAP" result so make_umap_fig uses the fresh
        # split_umap_1/2 computed below, not a cached result from a previous
        # run with different parameters.
        # (left alone while a "Rerun UMAP" job is running: it owns the state)
        with core._umap_reseg_lock:
            if core._umap_reseg_state["status"] != "running":
                core._umap_reseg_state["status"] = "idle"
                core._umap_reseg_state["result"] = None

        # ── 1. Get active data ──────────────────────────────────────────────
        _tool = core._seg_tool(seg_source)
        with core._baysor_lock:
            bres = core._baysor_state["result"] if core._baysor_state["status"] == "done" else None
        with core._proseg_lock:
            pres = core._proseg_state["result"] if core._proseg_state["status"] == "done" else None
        if _tool == "baysor":
            _alt_res = bres
        elif _tool == "proseg":
            _alt_res = pres
        else:
            _alt_res = core._seg_result(seg_source, done_only=True)

        if _alt_res is not None:
            expr       = _alt_res["expr"]
            cells_df   = _alt_res["cells_df"].copy()
            sdata_path = _alt_res.get("sdata_path", "")
            gni        = _alt_res.get("gene_name_to_idx") or core.DATA.get("gene_name_to_idx", {})
        else:
            expr       = core.DATA["expr"]
            cells_df   = core.DATA["df"].copy()
            sdata_path = core.DATA.get("sdata_path", "")
            gni        = core.DATA.get("gene_name_to_idx", {})
            # DATA["df"] rows map to expr rows through df_to_expr (differs from
            # the identity when subset() is active) - pair them explicitly.
            _x_d2e = np.asarray(core.DATA["df_to_expr"])
            if len(_x_d2e) != expr.shape[0] or (_x_d2e != np.arange(len(_x_d2e))).any():
                _x_good  = _x_d2e >= 0
                cells_df = cells_df[_x_good]
                _x_d2e   = _x_d2e[_x_good]
                expr     = expr[_x_d2e, :]
            else:
                _x_d2e = None

        # Panel genes only: DATA["gene_names"] can also hold SpaGE-imputed genes
        panel_genes = [g for g in dataview._panel_gene_names() if g in gni and gni[g] < expr.shape[1]]
        _panel_idx  = [gni[g] for g in panel_genes]
        n_panel     = len(panel_genes)
        # Slice to panel-only columns (drop imputed)
        if _panel_idx != list(range(expr.shape[1])):
            expr_panel = expr[:, _panel_idx]
        else:
            expr_panel = expr
        n_cells, n_genes = expr_panel.shape
        print(f"  SPLIT: {n_cells:,} cells × {n_genes} panel genes", flush=True)

        from rpy2.rinterface_lib import openrlib
        with openrlib.rlock:
            # ── 2. Check R packages ──────────────────────────────────────────────
            _set("running", "Loading R packages (spacexr, SPLIT)…")
            try:
                importr("spacexr")
            except Exception:
                _set("error", "spacexr not installed. In R: devtools::install_github('dmcable/spacexr')")
                return
            try:
                importr("SPLIT")
            except Exception:
                _set("error", "SPLIT not installed. In R: remotes::install_github('bdsc-tds/SPLIT')")
                return
            importr("SeuratObject")
            importr("Matrix")
            base = importr("base")

            # ── 3. Load Seurat reference ─────────────────────────────────────────
            _set("running", "Loading Seurat reference…")
            rds = base.readRDS(rds_path)
            meta_r  = ro.r['slot'](rds, "meta.data")
            meta_df = pandas2ri.rpy2py(meta_r)
            if label_col not in meta_df.columns:
                _set("error", f"Column '{label_col}' not found. Available: {list(meta_df.columns)[:10]}")
                return

            # ── 4. Shared genes ──────────────────────────────────────────────────
            _set("running", "Finding shared genes…")
            ro.r.assign("._split_rds", rds)
            mat_r     = ro.r("SeuratObject::LayerData(._split_rds[['RNA']], layer='counts')")
            ref_genes = list(ro.r['rownames'](mat_r))
            shared_genes = sorted(set(panel_genes) & set(ref_genes))
            if len(shared_genes) < 10:
                _set("error", f"Only {len(shared_genes)} shared genes between reference and panel.")
                return
            print(f"  SPLIT: {len(shared_genes)} shared genes", flush=True)

            # ── 5. Build Reference ───────────────────────────────────────────────
            _set("running", f"Building RCTD Reference ({len(meta_df):,} ref cells)…")
            ro.r.assign("._split_genes", ro.StrVector(shared_genes))
            ro.r("""
._split_ref_mat <- SeuratObject::LayerData(._split_rds[['RNA']], layer='counts')[._split_genes, , drop=FALSE]
._split_ref_mat <- as(._split_ref_mat, 'dgCMatrix')
._split_ref_mat@x <- as.numeric(round(._split_ref_mat@x))
""")
            ref_labels_r       = ro.StrVector(meta_df[label_col].astype(str).tolist())
            ref_labels_r.names = ro.StrVector(list(meta_df.index.astype(str)))
            ro.r.assign("._split_ref_labels", ref_labels_r)
            ro.r("""
._split_ref_factor <- as.factor(._split_ref_labels)
._split_numi_ref   <- colSums(._split_ref_mat)
._split_reference  <- spacexr::Reference(
    counts      = ._split_ref_mat,
    cell_types  = ._split_ref_factor,
    nUMI        = ._split_numi_ref
)
""")

            # ── 6. Build SpatialRNA (panel genes only) ───────────────────────────
            _set("running", f"Building SpatialRNA object ({n_cells:,} cells)…")
            # Subset expr_panel to shared genes
            _panel_pos = {g: i for i, g in enumerate(panel_genes)}
            shared_idx = [_panel_pos[g] for g in shared_genes if g in _panel_pos]
            shared_genes_present = [g for g in shared_genes if g in _panel_pos]
            expr_shared = expr_panel[:, shared_idx]  # cells × shared_genes
            # gene × cells dgCMatrix for R, built from the sparse parts
            cell_ids = [str(c) for c in cells_df.index]
            _scipy_to_r_dgc(expr_shared, shared_genes_present, cell_ids, "._split_counts_mat")

            ro.r.assign("._split_x", ro.FloatVector(cells_df["x_centroid"].astype(float).tolist()))
            ro.r.assign("._split_y", ro.FloatVector(cells_df["y_centroid"].astype(float).tolist()))
            ro.r("""
._split_coords <- data.frame(x=._split_x, y=._split_y)
rownames(._split_coords) <- colnames(._split_counts_mat)
._split_numi <- colSums(._split_counts_mat)
._split_spatialrna <- spacexr::SpatialRNA(
    coords  = ._split_coords,
    counts  = ._split_counts_mat,
    nUMI    = ._split_numi
)
""")

            # ── 7. Run RCTD (doublet mode) — with caching ────────────────────────
            # Compute a stable key for this RCTD run so we can skip it on re-runs.
            _rctd_lkey = "labels_rctd_doublet"
            if _alt_res is not None:
                _rctd_tag  = hashlib.md5((_alt_res.get("out_dir", "")).encode()).hexdigest()[:8]
                _rctd_tool = _alt_res.get("source", "baysor")
                _rctd_lkey = f"labels_rctd_doublet_{_rctd_tool}_{_rctd_tag}"
            # Under an active subset() the cell set is part of the key (full-dataset
            # runs keep the old name). It goes before the labels key: _annot_autoload
            # takes everything after the last "_labels_" in the file name as that key.
            _cells_extra = ""
            if (_alt_res is not None and "_cells_df_original" in _alt_res) or \
                    (_alt_res is None and "_df_original" in core.DATA):
                _cells_extra = (f"_n{len(cell_ids)}_"
                                + hashlib.md5("\n".join(cell_ids).encode()).hexdigest()[:8])
            # "_cmi5pp": SPLIT's RCTD configuration (CELL_MIN_INSTANCE=5 plus
            # run_post_process_RCTD) differs from the annotation modal's doublet run,
            # so the two do not share a cache file even with equal UMI thresholds.
            _rctd_cache_key  = (f"rctd_{os.path.basename(rds_path)}_{caching._ref_file_tag(rds_path)}"
                                f"_{label_col}_doublet"
                                f"_umi{min_umi}_sig{min_umi_sigma}_cmi5pp{_cells_extra}_{_rctd_lkey}")
            _rctd_cache_file = caching._cache_path(_rctd_cache_key)
            _rctd_obj_file   = _rctd_cache_file.replace(".parquet", "_rctd_obj.rds")

            _rctd_from_cache = False
            if os.path.exists(_rctd_cache_file) and os.path.exists(_rctd_obj_file):
                try:
                    _cached_rctd = pd.read_parquet(_rctd_cache_file)
                    _split_labels_cached = _cached_rctd["label"].astype(str)
                    _split_labels_cached.index = _split_labels_cached.index.astype(str)
                    # Every cached cell must belong to this run (RCTD drops low-UMI
                    # cells, so the cache may hold fewer cells, never other ones).
                    _sample = _split_labels_cached.index
                    _overlap = int(_sample.isin(set(cell_ids)).sum())
                    if len(_sample) > 0 and _overlap == len(_sample):
                        # path passed as a value (it contains the reference's file name)
                        ro.globalenv["._split_rctd_path"] = _rctd_obj_file
                        ro.r('._split_rctd <- readRDS(._split_rctd_path); rm(._split_rctd_path)')
                        _split_labels = _split_labels_cached
                        _rctd_from_cache = True
                        _set("running", "RCTD loaded from cache, running SPLIT::purify()…")
                        print("  SPLIT: RCTD loaded from cache", flush=True)
                    else:
                        print(f"  SPLIT: RCTD cache index mismatch ({_overlap}/{len(_sample)}) — re-running", flush=True)
                except Exception as _ce:
                    print(f"  SPLIT: RCTD cache load failed: {_ce}", flush=True)

            if not _rctd_from_cache:
                _set("running", f"Running RCTD doublet mode (max_cores={max_cores})… [~20-40 min]")
                print("  SPLIT: starting RCTD doublet mode…", flush=True)
                ro.r.assign("._split_max_cores",    ro.IntVector([max_cores]))
                ro.r.assign("._split_min_umi",      ro.IntVector([min_umi]))
                ro.r.assign("._split_min_umi_sigma", ro.IntVector([min_umi_sigma]))
                ro.r("""
options(mc.cores = ._split_max_cores[1])
cat("  SPLIT R: detected cores =", parallel::detectCores(),
    "| requesting =", ._split_max_cores[1], "\n")
._split_rctd <- spacexr::create.RCTD(._split_spatialrna, ._split_reference,
    max_cores=._split_max_cores[1], CELL_MIN_INSTANCE=5,
    UMI_min=._split_min_umi[1], UMI_min_sigma=._split_min_umi_sigma[1])
._split_rctd <- spacexr::run.RCTD(._split_rctd, doublet_mode='doublet')
._split_rctd <- SPLIT::run_post_process_RCTD(._split_rctd)
""")
                # Save RCTD object + labels to cache for future re-runs.
                # Use as.character() in R to avoid rpy2 converting factor levels to
                # integer codes (which would show numbers instead of cell type names).
                jobs._check_dataset_gen(_gen)
                try:
                    ro.globalenv["._split_rctd_path"] = _rctd_obj_file
                    ro.r('saveRDS(._split_rctd, ._split_rctd_path); rm(._split_rctd_path)')
                    ro.r("""
._split_rctd_res   <- ._split_rctd@results$results_df
._split_rctd_types <- as.character(._split_rctd_res$first_type)
names(._split_rctd_types) <- rownames(._split_rctd_res)
""")
                    _types_r = ro.r("._split_rctd_types")
                    _names_r = ro.r("names(._split_rctd_types)")
                    _split_labels = pd.Series(
                        list(_types_r), index=list(_names_r), name="label"
                    ).astype(str)
                    pd.DataFrame({"label": _split_labels}).to_parquet(_rctd_cache_file)
                    print(f"  SPLIT: RCTD cached ({len(_split_labels.unique())} types)", flush=True)
                except Exception as _se:
                    print(f"  SPLIT: RCTD cache save failed: {_se}", flush=True)
                    _split_labels = None

            # Store RCTD labels in _annot_state so cell-type coloring works
            if _split_labels is not None:
                _rctd_generic_key = "_".join(_rctd_lkey.split("_")[:2] + _rctd_lkey.split("_")[3:])
                with core._annot_lock:
                    jobs._check_dataset_gen(_gen)
                    _split_publish_labels(core._annot_state, _split_labels, _rctd_lkey,
                                          _rctd_generic_key, partial=bool(_cells_extra))

            print("  SPLIT: RCTD done, running purify()…", flush=True)

            # ── 8. Run SPLIT::purify ─────────────────────────────────────────────
            _set("running", "Running SPLIT::purify()…")
            # Full counts matrix (all panel genes) gene × cells
            _scipy_to_r_dgc(expr_panel, list(panel_genes), cell_ids, "._split_full_counts")
            ro.r.assign("._split_do_purify_singlets", ro.BoolVector([purify_singlets]))
            ro.r("""
._split_res <- SPLIT::purify(
    counts          = ._split_full_counts,
    rctd            = ._split_rctd,
    DO_purify_singlets = ._split_do_purify_singlets[1]
)
._split_purified <- ._split_res$purified_counts
""")

            # ── 9. Convert back to Python ────────────────────────────────────────
            _set("running", "Converting corrected counts to Python…")
            purified_r = ro.r["._split_purified"]
            # Extract cell IDs from R colnames — RCTD may drop low-UMI cells
            try:
                corrected_cell_ids = list(ro.r['colnames'](purified_r))
            except Exception:
                corrected_cell_ids = None
            # purify() returns only the genes RCTD used (panel ∩ reference), not the panel
            try:
                corrected_gene_ids = list(ro.r['rownames'](purified_r))
            except Exception:
                corrected_gene_ids = None
            # Convert dgCMatrix → scipy CSR (cells × genes)
            purified_csr_t = _r_dgcmatrix_to_scipy(purified_r)  # gene × cells
        corrected_mat  = purified_csr_t.T.tocsr()             # cells × genes
        corrected_mat.data = np.clip(corrected_mat.data, 0, None)
        if corrected_gene_ids is not None and corrected_gene_ids != list(panel_genes):
            # Put every column back at its panel-gene position (one column per panel
            # gene, as all readers assume); genes purify() dropped stay zero.
            _pg_pos = {g: i for i, g in enumerate(panel_genes)}
            _gcols  = np.array([_pg_pos.get(str(g), -1) for g in corrected_gene_ids], dtype=np.int64)
            _coo    = corrected_mat.tocoo()
            _gok    = _gcols[_coo.col] >= 0
            corrected_mat = sp_.csr_matrix(
                (_coo.data[_gok], (_coo.row[_gok], _gcols[_coo.col[_gok]])),
                shape=(corrected_mat.shape[0], n_panel))
            print(f"  SPLIT: purify() returned {len(corrected_gene_ids)} of {n_panel} panel genes "
                  f"— columns re-aligned to the panel ({n_panel - int((_gcols >= 0).sum())} genes "
                  f"not in the reference have no corrected counts)", flush=True)
        elif corrected_gene_ids is None and corrected_mat.shape[1] != n_panel:
            raise RuntimeError(f"SPLIT::purify returned {corrected_mat.shape[1]} genes without names "
                               f"for a {n_panel}-gene panel — cannot align the corrected counts")
        n_corr = corrected_mat.shape[0]
        n_orig = len(cells_df)
        print(f"  SPLIT: corrected matrix shape {corrected_mat.shape} "
              f"({n_corr}/{n_orig} cells)", flush=True)
        if corrected_cell_ids is None:
            corrected_cell_ids = [str(c) for c in cells_df.index[:n_corr]]

        # ── 10. Compute clusters + UMAP ──────────────────────────────────────
        _set("running", "Computing clusters and UMAP on corrected counts…")
        _compute_split_clusters_umap(cells_df, corrected_mat,
                                      corrected_cell_ids=corrected_cell_ids,
                                      progress_fn=lambda msg: _set("running", msg))
        print(f"  [SPLIT] after cluster+umap: cols={[c for c in cells_df.columns if 'split' in c]}, "
              f"has_split_umap={'split_umap_1' in cells_df.columns}", flush=True)

        # Cells whose counts SPLIT actually changed (for the status message): of the
        # cells submitted, RCTD keeps n_corr, and purify() alters only part of those.
        n_changed = None
        try:
            _cpos = pd.Index(cells_df.index.astype(str)).get_indexer(
                [str(c) for c in corrected_cell_ids])
            if (len(_cpos) == corrected_mat.shape[0] and (_cpos >= 0).all()
                    and expr_panel.shape[1] == corrected_mat.shape[1]):
                _cdiff = corrected_mat - sp_.csr_matrix(expr_panel)[_cpos, :].astype(corrected_mat.dtype)
                n_changed = int((np.asarray(abs(_cdiff).sum(axis=1)).ravel() > 1e-6).sum())
                del _cdiff
        except Exception as _nce:
            print(f"  SPLIT: could not count changed cells: {_nce}", flush=True)

        # ── 11. Write to zarr ────────────────────────────────────────────────
        # Not under an active subset(): the store holds one X_corrected row and one
        # cluster / UMAP value per cell of the whole dataset, and writing a subset
        # run would replace an earlier full-dataset correction with zero rows.
        _split_subset = ((_alt_res is not None and "_cells_df_original" in _alt_res)
                         or (_alt_res is None and "_df_original" in core.DATA))
        _split_note = ""
        # Last cancellation point: the store and the in-memory result are updated together
        jobs._job_shield()
        if _split_subset:
            _split_note = (" (subset active: result kept in memory only, "
                           "the SpatialData store was not changed)")
            print("  SPLIT: subset() is active — corrected counts are kept in memory and "
                  "NOT written to the SpatialData store (run SPLIT without a subset to "
                  "store them)", flush=True)
        elif sdata_path and os.path.isdir(sdata_path):
            _set("running", "Writing corrected counts to SpatialData zarr…")
            _split_write_zarr(sdata_path, corrected_mat, cells_df, list(panel_genes),
                              corrected_cell_ids=corrected_cell_ids)

        # Expand corrected_mat to full cells_df size before storing in memory.
        # After RCTD, n_corr ≤ n_orig; _get_expr_values returns a column of length
        # n_corr which misaligns with the n_orig spatial plot → bottom cells show NA.
        # The zarr write above does its own alignment, so expand only here.
        if n_corr < len(cells_df):
            _id_to_row = {str(cid): i for i, cid in enumerate(corrected_cell_ids)}
            _valid_out, _valid_in = [], []
            for _oi, _cell_id in enumerate(cells_df.index):
                _r = _id_to_row.get(str(_cell_id), -1)
                if _r >= 0:
                    _valid_out.append(_oi)
                    _valid_in.append(_r)
            _n_full = len(cells_df)
            if _valid_in:
                _sub = corrected_mat[_valid_in, :]
                _coo = _sub.tocoo()
                corrected_mat = sp_.csr_matrix(
                    (_coo.data, (np.array(_valid_out)[_coo.row], _coo.col)),
                    shape=(_n_full, corrected_mat.shape[1])
                )
            else:
                corrected_mat = sp_.csr_matrix(
                    (_n_full, corrected_mat.shape[1]), dtype=corrected_mat.dtype)
            print(f"  SPLIT: expanded in-memory corrected matrix to {corrected_mat.shape} "
                  f"({len(_valid_in)}/{n_corr} cells mapped)", flush=True)

        # ── 12. Update in-memory ─────────────────────────────────────────────
        jobs._check_dataset_gen(_gen)
        if _alt_res is not None:
            _lock = core._seg_state(_tool)[1] or core._proseg_lock
            with _lock:
                # The run's cell table may have changed while SPLIT ran (subset() /
                # unsubset() in the REPL): rows are paired with the cells now in the
                # table by cell id, never by position.
                _mat_cur, _cols_cur = _split_onto_cells(corrected_mat, cells_df, _alt_res["cells_df"])
                _alt_res["split_corrected_expr"] = _mat_cur
                _alt_res["split_corrected_cell_ids"] = corrected_cell_ids
                _alt_res["split_panel_genes"] = list(panel_genes)
                _alt_res["split_corrected_imputed_genes"] = []
                # Merge cluster/UMAP cols back into alt_res cells_df
                for col, _v in _cols_cur.items():
                    _alt_res["cells_df"][col] = _v
                # SPLIT ran on a subset(): also keep a full-size copy (zero rows
                # outside the subset) so unsubset() restores an aligned matrix.
                _orig_df = _alt_res.get("_cells_df_original")
                if _orig_df is not None and _orig_df.index.is_unique:
                    _opos = _orig_df.index.get_indexer(cells_df.index)
                    if len(_opos) == corrected_mat.shape[0] and (_opos >= 0).all():
                        _coo = corrected_mat.tocoo()
                        _full_new = sp_.csr_matrix(
                            (_coo.data, (_opos[_coo.row], _coo.col)),
                            shape=(len(_orig_df), corrected_mat.shape[1]))
                        # An earlier full-run correction keeps its rows outside the subset
                        _full_old = _alt_res.get("_corrected_original")
                        if _full_old is not None and _full_old.shape == _full_new.shape:
                            _keep = np.ones(len(_orig_df), dtype=np.float32)
                            _keep[_opos] = 0.0
                            _full_new = (sp_.diags(_keep).dot(sp_.csr_matrix(_full_old))
                                         + _full_new).tocsr()
                        _alt_res["_corrected_original"] = _full_new
                        for col in ["cluster_split_10", "split_umap_1", "split_umap_2"]:
                            if col in cells_df.columns:
                                _orig_df[col] = cells_df[col].reindex(_orig_df.index).values
        else:
            if _x_d2e is not None:
                # Keep the invariant "one row per DATA['expr'] row": scatter the
                # rows (in cells_df order) back to their expression-matrix rows.
                _coo = corrected_mat.tocoo()
                corrected_mat = sp_.csr_matrix(
                    (_coo.data, (_x_d2e[_coo.row], _coo.col)),
                    shape=(core.DATA["expr"].shape[0], corrected_mat.shape[1]))
                # An earlier correction of the whole dataset keeps its rows for the
                # cells this (subset) run did not cover, instead of turning to zero.
                _prev_corr = core.DATA.get("split_corrected_expr")
                if _prev_corr is not None and _prev_corr.shape[0] == corrected_mat.shape[0] \
                        and _prev_corr.shape[1] >= corrected_mat.shape[1]:
                    _keep = np.ones(corrected_mat.shape[0], dtype=np.float32)
                    _keep[_x_d2e] = 0.0
                    corrected_mat = (sp_.diags(_keep).dot(
                        sp_.csr_matrix(_prev_corr)[:, :corrected_mat.shape[1]])
                        + corrected_mat).tocsr()
            core.DATA["split_corrected_expr"] = corrected_mat
            core.DATA["split_corrected_cell_ids"] = corrected_cell_ids
            core.DATA["split_panel_genes"] = list(panel_genes)
            core.DATA["split_corrected_imputed_genes"] = []
            for col in ["cluster_split_10", "split_umap_1", "split_umap_2"]:
                if col in cells_df.columns:
                    if _x_d2e is not None:
                        # cells_df is a subset of DATA["df"]: write back by cell id
                        core.DATA["df"][col] = cells_df[col].reindex(core.DATA["df"].index).values
                        if "_df_original" in core.DATA:
                            core.DATA["_df_original"][col] = \
                                cells_df[col].reindex(core.DATA["_df_original"].index).values
                    else:
                        core.DATA["df"][col] = cells_df[col].values
                    print(f"  [SPLIT] wrote {col} to DATA['df']", flush=True)
                else:
                    print(f"  [SPLIT] WARNING: {col} NOT in cells_df — skipping", flush=True)
            dataview._umap_df_cache.clear()  # split_umap_1/2 added; invalidate cached UMAP df

        # n_cells = cells submitted; n_corr = cells RCTD kept (returned by purify());
        # n_changed = cells whose counts differ from the input
        if n_changed is not None:
            _split_msg = (f"{n_changed:,} cells corrected "
                          f"({n_corr:,} of {n_cells:,} submitted cells classified by RCTD)")
        else:
            _split_msg = f"{n_corr:,} of {n_cells:,} submitted cells returned by SPLIT"
        _set("done", f"SPLIT complete — {_split_msg}{_split_note}",
             result={"n_cells": n_cells, "n_returned": n_corr, "n_changed": n_changed})
        print(f"  SPLIT: done — {_split_msg}{_split_note}", flush=True)

    except jobs._JobCancelled:
        raise                      # reported by _job_finish, not an error
    except Exception:
        import traceback
        msg = traceback.format_exc()
        print(f"  SPLIT ERROR:\n{msg}", flush=True)
        with core._split_lock:
            if not jobs._stale_job(_gen, "error"):
                core._split_state["status"]  = "error"
                core._split_state["message"] = msg.strip().splitlines()[-1]
