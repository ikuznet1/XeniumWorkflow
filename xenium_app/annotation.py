"""Cell type annotation: CellTypist, Seurat kNN and RCTD (rpy2), label lookup and autoload."""

import os
import re
import hashlib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import core, jobs, caching, dataview, split


def _labels_key_for_method(method: str, alt_res=None) -> str:
    """Return the _annot_state key for a given annotation method + seg source."""
    if alt_res is None:
        return f"labels_{method}"
    out_tag = hashlib.md5((alt_res.get("out_dir", "")).encode()).hexdigest()[:8]
    tool    = alt_res.get("source", "baysor")
    return f"labels_{method}_{tool}_{out_tag}"


def _label_key_for_active_seg(label_key, alt_res=None):
    """Check a chosen _annot_state labels key against the active segmentation.
    Xenium keys ('labels_<method>[_<mode>]') hold Xenium cell ids, reseg keys carry
    a '_<tool>_<run tag>' part. Returns the key itself when it belongs to the active
    segmentation, else the same method's key for it when that has labels, else None."""
    if not label_key or not label_key.startswith("labels_"):
        return label_key
    if alt_res is None:
        if "_baysor_" not in label_key and "_proseg_" not in label_key:
            return label_key
    elif _labels_key_for_method("", alt_res)[len("labels_"):] in label_key:
        return label_key
    _alt_key = _labels_key_for_method(label_key.split("_")[1], alt_res)
    with core._annot_lock:
        return _alt_key if core._annot_state.get(_alt_key) is not None else None


def _reseg_cell_type_labels(method: str, alt_res, cell_ids_str):
    """Cell-type labels of *method* for a Baysor / Proseg run, aligned to *cell_ids_str*
    ("Unknown" where a cell has none), or None when there are none. Labels that match
    fewer than 5 % of the run's cells belong to another segmentation (stale): they are
    removed from _annot_state and None is returned. Every view of a reseg source - Plotly
    spatial and UMAP, the WebGL colour endpoint - goes through here, so all of them fall
    back to transcript counts in the same state and leave the same state behind."""
    lkey = _labels_key_for_method(method, alt_res)
    with core._annot_lock:
        labels = core._annot_state.get(lkey)
    if labels is None:
        return None
    aligned = labels.reindex(pd.Index(cell_ids_str)).fillna("Unknown")
    n_unknown = int((aligned == "Unknown").sum())
    if n_unknown > len(aligned) * 0.95:
        print(f"  Warning: cell_type:{method} produced {n_unknown}/{len(aligned)} Unknown "
              f"— labels index sample: {list(labels.index[:3])!r}, "
              f"cells sample: {list(cell_ids_str[:3])!r}"
              f" — stale labels cleared, re-run annotation", flush=True)
        with core._annot_lock:
            if core._annot_state.get(lkey) is labels:
                core._annot_state.pop(lkey, None)
        return None
    return aligned


def _run_celltypist(model_name: str, labels_key: str = "labels",
                    expr_override=None, cell_ids_override=None, cache_sig=None) -> None:
    """Background thread: annotate cells with CellTypist and store results."""
    core._redirect_rpy2_console()
    _gen = jobs._dataset_gen

    def _set(status, message, labels=None):
        with core._annot_lock:
            if jobs._stale_job(_gen, status, message):
                return
            core._annot_state["status"]  = status
            core._annot_state["message"] = message
            if labels is not None:
                core._annot_state[labels_key] = labels

    try:
        # imports inside the try so a missing package reports an error instead of
        # leaving the status at "running"
        import anndata as ad
        import celltypist
        from celltypist import models as ct_models
        # ── Check disk cache first ───────────────────────────────────────
        _cache_key = model_name if labels_key == "labels" else f"{model_name}_{labels_key}"
        cache_file = caching._cache_path(_cache_key)
        if os.path.exists(cache_file):
            _set("running", "Loading cached annotation…")
            cached = pd.read_parquet(cache_file)
            pred_labels = cached["label"].astype(str)
            pred_labels.index = pred_labels.index.astype(str)
            if cell_ids_override is not None:
                if not caching._annot_cache_valid(cached, cell_ids_override, cache_sig):
                    print(f"  Warning: cached annotation belongs to another segmentation "
                          f"— re-running", flush=True)
                    try:
                        os.remove(cache_file)
                    except Exception:
                        pass
                    # Fall through to full re-run
                else:
                    unique_types = pred_labels.unique().tolist()
                    print(f"  Loaded annotation from cache: {len(unique_types)} cell types", flush=True)
                    _set("done", f"Done (cached) — {len(unique_types)} cell types", labels=pred_labels)
                    return
            else:
                unique_types = pred_labels.unique().tolist()
                print(f"  Loaded annotation from cache: {len(unique_types)} cell types", flush=True)
                _set("done", f"Done (cached) — {len(unique_types)} cell types", labels=pred_labels)
                return

        _set("running", "Downloading model…")
        ct_models.download_models(model=model_name, force_update=False)

        _set("running", "Building expression matrix…")
        # Build AnnData aligned to barcode order (expr rows = barcodes)
        expr     = expr_override if expr_override is not None else core.DATA["expr"]
        genes    = core.DATA["gene_names"]
        if expr_override is not None:
            # reseg matrix holds the measured panel columns only
            genes = list(genes)[:expr.shape[1]]
        elif len(dataview._panel_gene_names()) != len(genes):
            # Xenium: SpaGE-imputed columns in DATA["expr"] are not measured counts
            genes = dataview._panel_gene_names()
            expr  = expr[:, [core.DATA["gene_name_to_idx"][g] for g in genes]]
        barcodes = cell_ids_override if cell_ids_override is not None else core.DATA["barcodes"]

        # Normalize to 10,000 counts per cell then log1p (CP10K + log1p)
        # CellTypist requires log1p-normalized input
        mat = expr.astype("float32").tocsr()
        row_sums = np.asarray(mat.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0          # avoid divide-by-zero
        scale = 10_000.0 / row_sums
        mat = sp.diags(scale).dot(mat).tocsr()  # scale each row
        mat.data = np.log1p(mat.data)           # log1p in-place on non-zeros

        adata = ad.AnnData(
            X   = mat,
            obs = pd.DataFrame(index=barcodes),
            var = pd.DataFrame(index=genes),
        )

        _set("running", f"Running CellTypist ({model_name})…")
        # majority_voting builds a neighbour graph through scanpy, which imports umap:
        # stub tensorflow as the UMAP code paths do (NumPy ABI mismatch in the installed TF)
        import sys as _sys_ct, types as _types_ct
        if "tensorflow" not in _sys_ct.modules:
            _sys_ct.modules["tensorflow"] = _types_ct.ModuleType("tensorflow")
        predictions = celltypist.annotate(
            adata, model=model_name, majority_voting=True
        )

        # CellTypist returns a DataFrame; columns may have category dtype
        pred_df = predictions.predicted_labels
        col = "majority_voting" if "majority_voting" in pred_df.columns else "predicted_labels"
        # Cast to plain str so reindex/fillna work correctly on any pandas version
        pred_labels = pred_df[col].astype(str)
        pred_labels.index = pred_labels.index.astype(str)
        unique_types = pred_labels.unique().tolist()
        print(f"  Annotation done: {len(unique_types)} cell types — {unique_types[:8]}", flush=True)

        # ── Save to disk cache ───────────────────────────────────────────
        # (cache_file was resolved at the start, for the dataset this job belongs to)
        jobs._check_dataset_gen(_gen)
        try:
            caching._annot_cache_frame(pred_labels, cache_sig).to_parquet(cache_file)
            print(f"  Annotation cached to {cache_file}", flush=True)
        except Exception as ce:
            print(f"  Warning: could not cache annotation: {ce}", flush=True)

        _set("done", f"Done — {len(unique_types)} cell types", labels=pred_labels)

    except Exception as exc:
        _set("error", str(exc))


def _run_rctd_annotation(rds_path: str, label_col: str = "Names", mode: str = "full",
                          n_cores: int = 4, labels_key: str = "labels",
                          expr_override=None, cell_ids_override=None,
                          umi_min: int = 20, umi_min_sigma: int = 100,
                          mode_labels_key: str = None, cache_sig=None) -> None:
    """
    Background thread: run RCTD (spacexr) for cell-type assignment.
    Requires the spacexr R package:
        devtools::install_github('dmcable/spacexr')
    In 'full' mode each cell is assigned the highest-weight type.
    In 'doublet'/'multi' mode the first_type column is used.
    """
    core._redirect_rpy2_console()

    _key_for_cache = mode_labels_key or labels_key
    _gen = jobs._dataset_gen

    def _set(status, message, labels=None):
        with core._annot_lock:
            if jobs._stale_job(_gen, status, message):
                return
            core._annot_state["status"]  = status
            core._annot_state["message"] = message
            if labels is not None:
                core._annot_state[labels_key] = labels
                if mode_labels_key and mode_labels_key != labels_key:
                    core._annot_state[mode_labels_key] = labels

    ro = None       # set once R is used; the finally below cleans up after it
    tmpdir = None
    try:
        # inside the try so a missing rpy2 reports an error instead of a stuck "running"
        import rpy2.robjects.conversion as _rconv
        import rpy2.robjects as _ro_mod
        _rconv.set_conversion(_ro_mod.default_converter)
        cache_key  = (f"rctd_{os.path.basename(rds_path)}_{caching._ref_file_tag(rds_path)}_{label_col}_{mode}"
                      f"_umi{umi_min}_sig{umi_min_sigma}_{_key_for_cache}")
        cache_file = caching._cache_path(cache_key)
        if os.path.exists(cache_file):
            # an empty label file (failed earlier run) is not a usable cache
            try:
                if len(pd.read_parquet(cache_file)) == 0:
                    os.remove(cache_file)
            except Exception:
                pass
        if os.path.exists(cache_file):
            _set("running", "Loading cached RCTD annotation…")
            cached = pd.read_parquet(cache_file)
            labels = cached["label"].astype(str)
            labels.index = labels.index.astype(str)
            if cell_ids_override is not None:
                # RCTD drops low-UMI cells, so the cached ids may be a subset
                if not caching._annot_cache_valid(cached, cell_ids_override, cache_sig, allow_subset=True):
                    print(f"  Warning: cached RCTD annotation belongs to another segmentation "
                          f"— re-running", flush=True)
                    try:
                        os.remove(cache_file)
                        weights_cache_del = cache_file.replace(".parquet", "_weights.parquet")
                        if os.path.exists(weights_cache_del):
                            os.remove(weights_cache_del)
                    except Exception:
                        pass
                    # Fall through to full re-run
                else:
                    unique_types = labels.unique().tolist()
                    print(f"  RCTD: loaded from cache — {len(unique_types)} cell types", flush=True)
                    # Restore weights so pie charts render on zoom
                    weights_cache = cache_file.replace(".parquet", "_weights.parquet")
                    weights_df = None
                    if os.path.exists(weights_cache):
                        try:
                            weights_df = pd.read_parquet(weights_cache)
                            weights_df.index = weights_df.index.astype(str)
                            print(f"  RCTD: loaded weights from cache ({weights_df.shape[1]} types)", flush=True)
                        except Exception as _we:
                            print(f"  RCTD: weight cache load failed: {_we}", flush=True)
                    weights_key = f"rctd_weights_{labels_key}"
                    with core._annot_lock:
                        jobs._check_dataset_gen(_gen)
                        core._annot_state[weights_key] = weights_df
                    _set("done", f"Done (cached) — {len(unique_types)} cell types", labels=labels)
                    return
            else:
                unique_types = labels.unique().tolist()
                print(f"  RCTD: loaded from cache — {len(unique_types)} cell types", flush=True)
                # Restore weights so pie charts render on zoom
                weights_cache = cache_file.replace(".parquet", "_weights.parquet")
                weights_df = None
                if os.path.exists(weights_cache):
                    try:
                        weights_df = pd.read_parquet(weights_cache)
                        weights_df.index = weights_df.index.astype(str)
                        print(f"  RCTD: loaded weights from cache ({weights_df.shape[1]} types)", flush=True)
                    except Exception as _we:
                        print(f"  RCTD: weight cache load failed: {_we}", flush=True)
                weights_key = f"rctd_weights_{labels_key}"
                with core._annot_lock:
                    jobs._check_dataset_gen(_gen)
                    core._annot_state[weights_key] = weights_df
                _set("done", f"Done (cached) — {len(unique_types)} cell types", labels=labels)
                return

        import rpy2.robjects as ro
        import rpy2.robjects.conversion as _rconv
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr

        # Ensure rpy2 conversion rules are active in this thread
        _rconv.set_conversion(ro.default_converter)
        pandas2ri.activate()

        from rpy2.rinterface_lib import openrlib
        with openrlib.rlock:
            # ── 1. Check spacexr ─────────────────────────────────────────────
            _set("running", "Loading spacexr…")
            try:
                importr("spacexr")
            except Exception as _spacexr_err:
                print(f"  RCTD: importr('spacexr') failed: {_spacexr_err}", flush=True)
                _set("error", f"spacexr not found by rpy2: {str(_spacexr_err)[:200]}")
                return

            importr("SeuratObject")
            importr("Matrix")
            base = importr("base")

            # ── 2. Load Seurat reference ─────────────────────────────────────
            _set("running", "Loading Seurat reference…")
            rds = base.readRDS(rds_path)
            meta_r  = ro.r['slot'](rds, "meta.data")
            meta_df = pandas2ri.rpy2py(meta_r)
            if label_col not in meta_df.columns:
                _set("error", f"Column '{label_col}' not found. Available: {list(meta_df.columns)[:10]}")
                return

            # ── 3. Determine shared genes ────────────────────────────────────
            _set("running", "Finding shared genes…")
            ro.r.assign("._rctd_rds", rds)
            mat_r     = ro.r("SeuratObject::LayerData(._rctd_rds[['RNA']], layer='counts')")
            ref_genes = list(ro.r['rownames'](mat_r))
            # Xenium: measured panel genes only (SpaGE-imputed columns are not counts)
            xenium_genes = dataview._panel_gene_names()
            if expr_override is not None:
                # reseg matrix holds the measured panel columns only
                xenium_genes = list(core.DATA["gene_names"])[:expr_override.shape[1]]
            shared_genes = sorted(set(xenium_genes) & set(ref_genes))
            if len(shared_genes) < 10:
                _set("error", f"Only {len(shared_genes)} shared genes between reference and panel.")
                return
            print(f"  RCTD: {len(shared_genes)} shared genes", flush=True)

            # ── 4. Build Reference object ────────────────────────────────────
            _set("running", f"Building RCTD Reference ({len(meta_df):,} ref cells)…")
            ro.r.assign("._rctd_genes", ro.StrVector(shared_genes))
            ro.r("""
._rctd_ref_mat <- SeuratObject::LayerData(._rctd_rds[['RNA']], layer='counts')[._rctd_genes, , drop=FALSE]
._rctd_ref_mat <- as(._rctd_ref_mat, 'dgCMatrix')
._rctd_ref_mat@x <- as.numeric(round(._rctd_ref_mat@x))
""")
            ref_labels_r       = ro.StrVector(meta_df[label_col].astype(str).tolist())
            ref_labels_r.names = ro.StrVector(list(meta_df.index.astype(str)))
            ro.r.assign("._rctd_ref_labels", ref_labels_r)
            ro.r("""
._rctd_ref_factor <- as.factor(._rctd_ref_labels)
._rctd_numi_ref   <- colSums(._rctd_ref_mat)
._rctd_reference  <- spacexr::Reference(
    counts     = ._rctd_ref_mat,
    cell_types = ._rctd_ref_factor,
    nUMI       = ._rctd_numi_ref
)
""")

            # ── 5. Build SpatialRNA from Xenium ──────────────────────────────
            _set("running", "Building SpatialRNA object…")
            df = core.DATA["df"]
            if expr_override is not None:
                xen_mat_full = expr_override.T        # genes × cells
                cell_ids     = [str(c) for c in (cell_ids_override if cell_ids_override is not None
                                                 else range(expr_override.shape[0]))]
                coords_df    = df[["x_centroid", "y_centroid"]].reindex(
                                   pd.Index(cell_ids)).fillna(0.0)
            else:
                # All cells, in expression-row order (independent of an active
                # subset(), like CellTypist/Seurat): matrix columns == barcodes.
                df = core.DATA.get("_df_original", df)
                xen_mat_full = core.DATA["expr"].T  # genes × cells
                cell_ids     = [str(b) for b in core.DATA["barcodes"]]
                coords_df    = df[["x_centroid", "y_centroid"]].copy()
                coords_df.index = coords_df.index.astype(str)
                if list(coords_df.index) != cell_ids:
                    coords_df = coords_df.reindex(pd.Index(cell_ids)).fillna(0.0)

            gene_idx   = ({g: i for i, g in enumerate(xenium_genes)} if expr_override is not None
                          else core.DATA["gene_name_to_idx"])
            shared_idx = [gene_idx[g] for g in shared_genes]
            # shared_genes × cells dgCMatrix built from the sparse parts
            # (no MatrixMarket text round trip)
            split._scipy_to_r_dgc(xen_mat_full.T[:, shared_idx], shared_genes, cell_ids, "._rctd_xen_mat")
            del xen_mat_full
            ro.r.assign("._rctd_cx", ro.FloatVector(
                np.ascontiguousarray(coords_df["x_centroid"].to_numpy(dtype=np.float64))))
            ro.r.assign("._rctd_cy", ro.FloatVector(
                np.ascontiguousarray(coords_df["y_centroid"].to_numpy(dtype=np.float64))))
            ro.r("""
._rctd_coords   <- data.frame(x=._rctd_cx, y=._rctd_cy, row.names=colnames(._rctd_xen_mat))
._rctd_numi_xen <- colSums(._rctd_xen_mat)
._rctd_puck     <- spacexr::SpatialRNA(
    coords = ._rctd_coords,
    counts = ._rctd_xen_mat,
    nUMI   = ._rctd_numi_xen
)
""")

            # ── 6. Create and run RCTD ───────────────────────────────────────
            _set("running", f"Running RCTD ({mode} mode, {n_cores} cores)…")
            ro.r.assign("._rctd_mode",    mode)
            ro.r.assign("._rctd_cores",   n_cores)
            ro.r.assign("._rctd_umi_min",       ro.IntVector([umi_min]))
            ro.r.assign("._rctd_umi_min_sigma", ro.IntVector([umi_min_sigma]))
            ro.r("""
._rctd_obj <- spacexr::create.RCTD(._rctd_puck, ._rctd_reference, max_cores=._rctd_cores,
    UMI_min=._rctd_umi_min[1], UMI_min_sigma=._rctd_umi_min_sigma[1])
._rctd_obj <- spacexr::run.RCTD(._rctd_obj, doublet_mode=._rctd_mode)
""")
            print("  RCTD: run complete — extracting results…", flush=True)

            # ── 7. Extract per-cell labels ────────────────────────────────────
            _set("running", "Extracting RCTD results…")
            if mode == "full":
                ro.r("""
._rctd_weights <- ._rctd_obj@results$weights
._rctd_types   <- colnames(._rctd_weights)[apply(._rctd_weights, 1, which.max)]
names(._rctd_types) <- rownames(._rctd_weights)
""")
            elif mode == "multi":
                # multi mode: @results is a per-pixel list (no results_df), in the
                # column order of the filtered SpatialRNA counts
                ro.r("""
._rctd_res   <- ._rctd_obj@results
._rctd_types <- vapply(._rctd_res, function(r)
    if (length(r$cell_type_list)) as.character(r$cell_type_list[1]) else NA_character_,
    character(1))
names(._rctd_types) <- colnames(._rctd_obj@spatialRNA@counts)[seq_along(._rctd_types)]
._rctd_types <- ._rctd_types[!is.na(._rctd_types)]
""")
            else:  # doublet
                ro.r("""
._rctd_res   <- ._rctd_obj@results$results_df
._rctd_types <- as.character(._rctd_res$first_type)
names(._rctd_types) <- rownames(._rctd_res)
""")

            types_r   = ro.r("._rctd_types")
            names_r   = ro.r("names(._rctd_types)")
            cell_ids  = list(names_r)
            cell_vals = [str(v) for v in types_r]
            labels    = pd.Series(dict(zip(cell_ids, cell_vals)), name="label").astype(str)
            labels.index  = labels.index.astype(str)
            if len(labels) == 0:
                raise RuntimeError(f"RCTD ({mode}) returned no cell labels")

            unique_types = labels.unique().tolist()
            print(f"  RCTD: {len(labels):,} cells → {len(unique_types)} types", flush=True)

            jobs._check_dataset_gen(_gen)
            # last cancellation point: label cache, weight cache and labels go together
            jobs._job_shield()
            caching._annot_cache_frame(labels, cache_sig).to_parquet(cache_file)

            # ── 8. Extract and cache weight matrix (full mode only) ───────────
            weights_df = None
            if mode == "full":
                _set("running", "Extracting RCTD weight matrix…")
                try:
                    ro.r("._rctd_weights_dense <- as.matrix(._rctd_weights)")
                    ct_names   = list(ro.r("colnames(._rctd_weights)"))
                    cell_ids_w = list(ro.r("rownames(._rctd_weights)"))
                    w_arr      = np.array(ro.r("._rctd_weights_dense"))
                    weights_df = pd.DataFrame(w_arr, index=cell_ids_w, columns=ct_names)
                    weights_df.index = weights_df.index.astype(str)
                    weights_cache = cache_file.replace(".parquet", "_weights.parquet")
                    weights_df.to_parquet(weights_cache)
                    print(f"  RCTD: weights saved ({weights_df.shape[1]} types)", flush=True)
                except Exception as _we:
                    print(f"  RCTD: weight extraction failed: {_we}", flush=True)
            weights_key = f"rctd_weights_{labels_key}"
            with core._annot_lock:
                jobs._check_dataset_gen(_gen)
                core._annot_state[weights_key] = weights_df

            _set("done", f"Done — {len(unique_types)} cell types in {len(labels):,} cells",
                 labels=labels)

    except jobs._JobCancelled:
        raise                      # reported by _job_finish, not an error
    except Exception as exc:
        import traceback; traceback.print_exc()
        _set("error", str(exc)[:300])
    finally:
        # Cleanup R temporaries and the temp dir on every exit path (best-effort).
        # The names start with a dot, so ls() needs all.names=TRUE.
        if ro is not None:
            try:
                from rpy2.rinterface_lib import openrlib as _orl
                with _orl.rlock:
                    ro.r("rm(list=ls(pattern='^[.]_rctd_', all.names=TRUE, envir=globalenv()), "
                         "envir=globalenv()); invisible(gc())")
            except Exception:
                pass
        if tmpdir:
            import shutil; shutil.rmtree(tmpdir, ignore_errors=True)


def _run_seurat_annotation(rds_path: str, label_col: str = "Names", labels_key: str = "labels",
                            expr_override=None, cell_ids_override=None, cache_sig=None) -> None:
    """
    Background thread: transfer cell type labels from a Seurat RDS reference
    to Xenium cells via PCA + cosine kNN majority voting.
    Uses the same rpy2 + SpaGE PV alignment already set up for SpaGE.
    """
    core._redirect_rpy2_console()
    _gen = jobs._dataset_gen

    def _set(status, message, labels=None):
        with core._annot_lock:
            if jobs._stale_job(_gen, status, message):
                return
            core._annot_state["status"]  = status
            core._annot_state["message"] = message
            if labels is not None:
                core._annot_state[labels_key] = labels

    tmpdir = None        # MatrixMarket export of the reference; removed in the finally
    _r_loaded = False    # the reference was put into R's global env
    try:
        # inside the try so a missing rpy2 reports an error instead of a stuck "running"
        import rpy2.robjects.conversion as _rconv
        import rpy2.robjects as _ro_mod
        _rconv.set_conversion(_ro_mod.default_converter)
        cache_key = (f"seurat_{os.path.basename(rds_path)}_{caching._ref_file_tag(rds_path)}"
                     f"_{label_col}_{labels_key}")
        cache_file = caching._cache_path(cache_key)
        if os.path.exists(cache_file):
            _set("running", "Loading cached annotation…")
            cached = pd.read_parquet(cache_file)
            labels = cached["label"].astype(str)
            labels.index = labels.index.astype(str)
            if cell_ids_override is not None:
                if not caching._annot_cache_valid(cached, cell_ids_override, cache_sig):
                    print(f"  Warning: cached Seurat annotation belongs to another segmentation "
                          f"— re-running", flush=True)
                    try:
                        os.remove(cache_file)
                    except Exception:
                        pass
                    # Fall through to full re-run
                else:
                    unique_types = labels.unique().tolist()
                    print(f"  Loaded annotation from cache: {len(unique_types)} cell types", flush=True)
                    _set("done", f"Done (cached) — {len(unique_types)} cell types", labels=labels)
                    return
            else:
                unique_types = labels.unique().tolist()
                print(f"  Loaded annotation from cache: {len(unique_types)} cell types", flush=True)
                _set("done", f"Done (cached) — {len(unique_types)} cell types", labels=labels)
                return

        # ── 1. Load Seurat object via rpy2 ───────────────────────────────
        _set("running", "Loading Seurat RDS (may take ~30s)…")
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        from rpy2.rinterface_lib import openrlib
        pandas2ri.activate()

        with openrlib.rlock:
            base = importr("base")
            rds  = base.readRDS(rds_path)

            # ── 2. Extract metadata → label vector ──────────────────────────
            _set("running", f"Extracting '{label_col}' labels…")
            meta_r = ro.r['slot'](rds, "meta.data")
            meta_df = pandas2ri.rpy2py(meta_r)
            if label_col not in meta_df.columns:
                cols = list(meta_df.columns)
                _set("error", f"Column '{label_col}' not found. Available: {cols[:10]}")
                return

            ref_labels = meta_df[label_col].astype(str)  # index = ref cell barcodes

            unique_types = ref_labels.unique().tolist()
            print(f"  Reference: {len(ref_labels)} cells, {len(unique_types)} cell types", flush=True)

            # ── 3. Find shared genes first (before loading any matrix) ──────
            import scipy.sparse as sp_
            import tempfile

            importr("SeuratObject")
            importr("Matrix")

            _set("running", "Reading reference gene list…")
            _r_loaded = True
            ro.r.assign("._annot_rds_tmp", rds)
            mat_r = ro.r("SeuratObject::LayerData(._annot_rds_tmp[['RNA']], layer='data')")
            ref_genes_all = list(ro.r['rownames'](mat_r))
            ref_bcs = list(ro.r['colnames'](mat_r))

            # Xenium: measured panel genes only (SpaGE-imputed columns are not counts)
            xenium_genes = dataview._panel_gene_names()
            if expr_override is not None:
                # reseg matrix holds the measured panel columns only
                xenium_genes = list(core.DATA["gene_names"])[:expr_override.shape[1]]
            shared = sorted(set(xenium_genes) & set(ref_genes_all))
            if len(shared) < 10:
                _set("error", f"Only {len(shared)} shared genes — not enough for label transfer.")
                return
            print(f"  Shared genes for label transfer: {len(shared)}", flush=True)

            # ── 4. Export ONLY shared-gene rows via rpy2 (no subprocess) ────
            _set("running", f"Extracting {len(shared)} shared-gene submatrix…")
            tmpdir = tempfile.mkdtemp(prefix="xenium_annot_")
            ro.r.assign("._annot_mat",   mat_r)
            ro.r.assign("._annot_genes", ro.StrVector(shared))
            ro.r.assign("._annot_dir",   tmpdir)
            ro.r("""
mat_sub <- ._annot_mat[._annot_genes, , drop = FALSE]
Matrix::writeMM(mat_sub, file.path(._annot_dir, "matrix.mtx"))
write.table(data.frame(gene = rownames(mat_sub)),
            file.path(._annot_dir, "genes.txt"),
            row.names = FALSE, col.names = FALSE, quote = FALSE)
write.table(data.frame(bc = colnames(mat_sub)),
            file.path(._annot_dir, "barcodes.txt"),
            row.names = FALSE, col.names = FALSE, quote = FALSE)
# Clean up temp R objects
rm(._annot_mat, ._annot_genes, ._annot_dir, ._annot_rds_tmp, mat_sub)
""")

        _set("running", "Loading reference sub-matrix…")
        import scipy.io as sio_
        ref_mat = sio_.mmread(os.path.join(tmpdir, "matrix.mtx")).T.tocsr().astype("float32")
        # ref_mat is now (n_ref_cells × n_shared), rows in order of ref_bcs, cols = shared
        ref_genes = pd.read_csv(os.path.join(tmpdir, "genes.txt"), header=None)[0].tolist()
        ref_bcs   = pd.read_csv(os.path.join(tmpdir, "barcodes.txt"), header=None)[0].tolist()

        xen_gene_idx = ({g: i for i, g in enumerate(xenium_genes)} if expr_override is not None
                        else core.DATA["gene_name_to_idx"])
        shared_xen_idx = [xen_gene_idx[g] for g in ref_genes]  # ref_genes == shared after subset

        # Reference matrix for shared genes (cells × shared) — already subset
        ref_shared = ref_mat

        # Xenium/reseg matrix for shared genes — build from sparse expr
        xen_expr = expr_override if expr_override is not None else core.DATA["expr"]
        # Kept sparse: a dense cells x shared-genes copy is tens of GB on a 5K panel
        xen_shared = xen_expr[:, shared_xen_idx].tocsr().astype("float32")
        # CP10K + log1p normalize Xenium
        row_sums = np.asarray(xen_shared.sum(axis=1)).ravel().astype("float32")
        row_sums[row_sums == 0] = 1.0
        xen_shared = sp_.diags((1e4 / row_sums).astype("float32")).dot(xen_shared).tocsr()
        xen_shared.data = np.log1p(xen_shared.data)
        xen_shared.sort_indices()

        # Also normalize reference (already log-normalized from Seurat data layer, but z-score both)
        import scipy.stats as st_
        from sklearn.decomposition import TruncatedSVD
        from sklearn.neighbors import NearestNeighbors

        _set("running", "Z-scoring and fitting PCA…")
        # Z-score across cells (per gene)
        ref_dense = ref_shared.toarray() if sp_.issparse(ref_shared) else ref_shared
        ref_mean = ref_dense.mean(axis=0, keepdims=True)
        ref_std  = ref_dense.std(axis=0, keepdims=True) + 1e-8
        ref_z = (ref_dense - ref_mean) / ref_std

        # per-gene mean / population std from the sparse matrix (float64 sums)
        _n_xen  = xen_shared.shape[0]
        _n_sh   = xen_shared.shape[1]
        _xd64   = xen_shared.data.astype(np.float64)
        _x_m    = np.bincount(xen_shared.indices, weights=_xd64, minlength=_n_sh) / max(_n_xen, 1)
        _x_sq   = np.bincount(xen_shared.indices, weights=_xd64 * _xd64, minlength=_n_sh) / max(_n_xen, 1)
        del _xd64
        xen_mean = _x_m.astype("float32")[None, :]
        xen_std  = (np.sqrt(np.maximum(_x_sq - _x_m ** 2, 0.0)) + 1e-8).astype("float32")[None, :]

        # PCA on reference, project Xenium
        n_comp = min(50, len(shared) - 1, ref_dense.shape[0] - 1)
        _set("running", f"PCA ({n_comp} components)…")
        svd = TruncatedSVD(n_components=n_comp, random_state=0)
        ref_pca = svd.fit_transform(ref_z)   # (n_ref × n_comp)

        # ── 5. kNN label transfer ────────────────────────────────────────
        _set("running", "kNN label transfer (50 neighbours)…")
        # Align ref_labels to ref_bcs order
        ref_label_series = pd.Series(ref_labels.values, index=ref_bcs)

        k = min(50, len(ref_bcs))
        nbrs = NearestNeighbors(n_neighbors=k, metric="cosine").fit(ref_pca)

        # Majority vote (ties -> smallest label, as pd.Series.mode()[0])
        ref_label_arr = ref_label_series.values  # ordered by ref_bcs
        _lab_u, _lab_codes = np.unique(ref_label_arr.astype(str), return_inverse=True)
        _n_lab = len(_lab_u)
        xen_labels = []
        # z-score, project and vote in row chunks so the dense block stays small
        _chunk = 20_000
        for _s in range(0, _n_xen, _chunk):
            _zc = (xen_shared[_s:_s + _chunk].toarray() - xen_mean) / xen_std
            _pc = _zc @ svd.components_.T   # (chunk × n_comp)
            _, indices = nbrs.kneighbors(_pc)
            _m = indices.shape[0]
            _cnt = np.bincount((np.arange(_m)[:, None] * _n_lab + _lab_codes[indices]).ravel(),
                               minlength=_m * _n_lab).reshape(_m, _n_lab)
            xen_labels.extend(_lab_u[_cnt.argmax(axis=1)].tolist())

        xen_barcodes = cell_ids_override if cell_ids_override is not None else core.DATA["barcodes"]
        pred_labels = pd.Series(xen_labels, index=[str(b) for b in xen_barcodes], name="label")
        unique_final = pred_labels.unique().tolist()
        print(f"  Label transfer done: {len(unique_final)} cell types", flush=True)

        # Cache
        jobs._check_dataset_gen(_gen)
        try:
            caching._annot_cache_frame(pred_labels, cache_sig).to_parquet(cache_file)
        except Exception as ce:
            print(f"  Warning: cache failed: {ce}", flush=True)

        _set("done", f"Done — {len(unique_final)} cell types", labels=pred_labels)

    except Exception as exc:
        import traceback
        _set("error", f"{exc}\n{traceback.format_exc()[-300:]}")
    finally:
        # The exported reference sub-matrix can be several GB; early returns and
        # errors would also leave the whole Seurat object in R's global env.
        if tmpdir:
            import shutil as _sh_annot
            _sh_annot.rmtree(tmpdir, ignore_errors=True)
        if _r_loaded:
            try:
                import rpy2.robjects as _ro_fin
                from rpy2.rinterface_lib import openrlib as _orl_fin
                with _orl_fin.rlock:
                    _ro_fin.r('suppressWarnings(rm(list = intersect(c("._annot_mat", '
                              '"._annot_genes", "._annot_dir", "._annot_rds_tmp", "mat_sub"), '
                              'ls(all.names = TRUE, envir = globalenv())), envir = globalenv())); '
                              'invisible(gc())')
            except Exception:
                pass


def _annot_autoload() -> None:
    """On startup, load all cached Xenium annotations for this dataset (all methods)."""
    import glob
    tag       = hashlib.md5(caching._dataset_key().encode()).hexdigest()[:8]
    cache_dir = core.CACHE_BASE_DIR
    pattern   = os.path.join(cache_dir, f"*_{tag}.parquet")
    label_files  = [f for f in glob.glob(pattern)
                    if not os.path.basename(f).startswith(("labels_", "spage_", "spatialdata_"))
                    and not os.path.basename(f).endswith("_weights.parquet")
                    # SPLIT's RCTD labels of a subset() run cover only part of the cells
                    and not re.search(r"_n\d+_[0-9a-f]{8}_labels_", os.path.basename(f))]
    # weight side-cars are named <labels file stem>_weights.parquet, i.e. the dataset
    # tag is not at the end of their name (the labels pattern never matched them)
    weight_files = glob.glob(os.path.join(cache_dir, f"*_{tag}_weights.parquet"))
    if not label_files and not weight_files:
        return
    # Several cache files can map to the same labels key (RCTD runs with another
    # reference, mode or UMI threshold; SPLIT's own RCTD run). The rule: per key, the
    # most recently written file wins (ties broken by file name, so the outcome does
    # not depend on directory order), and the RCTD weights are taken only from the
    # side-car of that same file - never from another run.
    _order = lambda f: (os.path.getmtime(f), os.path.basename(f))
    label_files.sort(key=_order)
    _RCTD_MODES = ("full", "doublet", "multi")

    def _generic_key(key: str):
        """labels_rctd_<mode>[_<tool>_<tag>] or labels_rctd[_<tool>_<tag>]_<mode>
        → the key the figures read (labels_rctd[_<tool>_<tag>]); None for other keys."""
        p = key.split("_")
        if len(p) >= 3 and p[1] == "rctd" and p[2] in _RCTD_MODES:
            return "_".join(p[:2] + p[3:])
        if len(p) >= 3 and p[1] == "rctd" and p[-1] in _RCTD_MODES:
            return "_".join(p[:-1])
        return None

    _winner: dict = {}      # labels key → cache file whose labels it now holds
    _n_cand: dict = {}
    loaded = 0
    for fpath in label_files:
        name = os.path.basename(fpath)
        # Extract labels_key embedded in filename: strip "_{tag}.parquet" then find "_labels_"
        name_stem = name[:-(len(tag) + 1 + len(".parquet"))]
        lk_pos = name_stem.rfind("_labels_")
        if lk_pos >= 0:
            labels_key = name_stem[lk_pos + 1:]  # e.g. "labels_seurat" or "labels_seurat_baysor_abc12345"
        elif name.startswith("rctd_"):
            labels_key = "labels_rctd"
        elif name.startswith("seurat_"):
            labels_key = "labels_seurat"
        else:
            labels_key = "labels_celltypist"
        # Derive method name from labels_key for display purposes
        method = labels_key.split("_")[1] if "_" in labels_key else labels_key
        try:
            cached       = pd.read_parquet(fpath)
            labels       = cached["label"].astype(str)
            labels.index = labels.index.astype(str)
            # For RCTD mode-specific keys (e.g. "labels_rctd_doublet", or
            # "labels_rctd_baysor_<tag>_doublet" from the annotation modal), also set the
            # generic key (e.g. "labels_rctd") so make_spatial_fig keeps working after restart.
            generic_key = _generic_key(labels_key)
            with core._annot_lock:
                core._annot_state[labels_key] = labels
                if generic_key:
                    core._annot_state[generic_key] = labels
                for _k in (labels_key, generic_key):
                    if _k:
                        _winner[_k] = fpath
                        _n_cand[_k] = _n_cand.get(_k, 0) + 1
                core._annot_state["status"]   = "done"
                core._annot_state["message"]  = f"Auto-loaded ({core._ANNOT_METHODS[method]})"
            print(f"  Annotation: auto-loaded {method} ({len(labels.unique())} types) from {name}", flush=True)
            loaded += 1
        except Exception as exc:
            print(f"  Annotation: auto-load failed for {name}: {exc}", flush=True)
    for _k, _n in sorted(_n_cand.items()):
        if _n > 1:
            print(f"  Annotation: {_n} cached runs provide '{_k}' - using the most recent, "
                  f"{os.path.basename(_winner[_k])}", flush=True)
    weight_files.sort(key=_order)
    for fpath in weight_files:
        name = os.path.basename(fpath)
        try:
            # Parse labels_key from weights filename (strip "_weights.parquet" then "_{tag}")
            w_name_stem = name[:-(len("_weights.parquet"))]
            w_name_stem2 = w_name_stem[:-(len(tag) + 1)]
            w_lk_pos = w_name_stem2.rfind("_labels_")
            if w_lk_pos >= 0:
                wlabels_key = w_name_stem2[w_lk_pos + 1:]
            else:
                wlabels_key = "labels_rctd"
            # Only for the keys whose labels came from this file's own run
            _own_labels = fpath[:-len("_weights.parquet")] + ".parquet"
            _targets = [k for k in (wlabels_key, _generic_key(wlabels_key))
                        if k and _winner.get(k) == _own_labels]
            if not _targets:
                continue
            weights_df = pd.read_parquet(fpath)
            weights_df.index = weights_df.index.astype(str)
            with core._annot_lock:
                for _k in _targets:
                    core._annot_state[f"rctd_weights_{_k}"] = weights_df
            print(f"  Annotation: auto-loaded RCTD weights ({weights_df.shape[1]} types) from {name}", flush=True)
        except Exception as exc:
            print(f"  Annotation: weight auto-load failed for {name}: {exc}", flush=True)
