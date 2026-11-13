"""SpaGE gene imputation: cache, autoload, the vectorised implementation and the job."""

import os
import hashlib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import compat, core, jobs, caching, loading, dataview, reseg
from . import provenance


# ─── SpaGE imputation ─────────────────────────────────────────────────────────
SPAGE_REPO = os.environ.get("SPAGE_REPO") or os.path.join(
    os.path.dirname(os.path.abspath(compat.APP_FILE)), "SpaGE_repo")


def _spage_cache_dir() -> str:
    d = core.CACHE_BASE_DIR
    os.makedirs(d, exist_ok=True)
    return d


def _spage_index_path() -> str:
    return os.path.join(_spage_cache_dir(), "spage_index.json")


def _spage_cache_path(rds_path: str, genes_key: str) -> str:
    # _ref_file_tag: an edited reference (same path, new size / mtime) is a new key
    tag = hashlib.md5((rds_path + caching._ref_file_tag(rds_path) + caching._dataset_key()
                       + genes_key).encode()).hexdigest()[:12]
    return os.path.join(_spage_cache_dir(), f"spage_{tag}.parquet")


def _spage_index_update(cache_file: str, dataset_key: str = None) -> None:
    """Record the latest spage cache file for this dataset (*dataset_key*: the
    key captured when the job started; default: the dataset loaded now)."""
    import json
    idx_path = _spage_index_path()
    try:
        with open(idx_path) as f:
            idx = json.load(f)
    except Exception:
        idx = {}
    idx[dataset_key or caching._dataset_key()] = cache_file
    with open(idx_path, "w") as f:
        json.dump(idx, f, indent=2)


def _spage_autoload() -> None:
    """On startup, restore the most recent SpaGE result for this dataset (if cached)."""
    import json
    idx_path = _spage_index_path()
    if not os.path.exists(idx_path):
        return
    try:
        with open(idx_path) as f:
            idx = json.load(f)
        cache_file = idx.get(caching._dataset_key())
        if cache_file and (os.path.exists(cache_file) or os.path.isdir(cache_file)):
            zarr_path = (cache_file if os.path.isdir(cache_file)
                         else cache_file.replace(".parquet", ".zarr")
                         if os.path.isdir(cache_file.replace(".parquet", ".zarr")) else None)
            if zarr_path:
                import zarr as _zarr_al
                _zal = _zarr_al.open_array(zarr_path, mode='r')
                _genes_al = list(_zal.attrs.get('genes', []))
                if _genes_al:
                    with core._spage_lock:
                        core._spage_state["status"]       = "done"
                        core._spage_state["message"]      = f"Auto-loaded (streaming) — {len(_genes_al):,} genes"
                        core._spage_state["result_path"]  = zarr_path
                        core._spage_state["result_genes"] = _genes_al
                    print(f"  SpaGE: streaming {len(_genes_al):,} genes from {zarr_path}", flush=True)
                    return
            # Normal parquet — check size before loading entire file
            try:
                import pyarrow.parquet as _pq_al
                _pfal = _pq_al.ParquetFile(cache_file)
                _nr_al = _pfal.metadata.num_rows
                _genes_al = _pfal.schema_arrow.names
                _nc_al = len(_genes_al)
                if _nr_al * _nc_al > core.SPAGE_STREAM_THRESHOLD:
                    with core._spage_lock:
                        core._spage_state["status"]       = "done"
                        core._spage_state["message"]      = f"Auto-loaded (streaming) — {_nc_al:,} genes"
                        core._spage_state["result_path"]  = cache_file
                        core._spage_state["result_genes"] = list(_genes_al)
                    print(f"  SpaGE: large cache {_nr_al:,}×{_nc_al:,} → streaming read", flush=True)
                    return
            except Exception:
                pass
            print(f"  SpaGE: auto-loading cached results from {cache_file}", flush=True)
            imp_df = pd.read_parquet(cache_file)
            with core._spage_lock:
                core._spage_state["status"]  = "done"
                core._spage_state["message"] = f"Auto-loaded (cached) — {len(imp_df.columns):,} genes"
                core._spage_state["result"]  = imp_df
    except Exception as exc:
        print(f"  SpaGE: auto-load failed: {exc}", flush=True)


def _vectorized_spage(spatial_df: pd.DataFrame, rna_df: pd.DataFrame,
                      n_pv: int, genes_to_predict: list,
                      out_zarr: str = None) -> "pd.DataFrame | None":
    """
    Vectorized SpaGE: same algorithm as SpaGE.main.SpaGE but the kNN
    weighted-average step is batched with numpy/einsum instead of a
    Python for-loop — ~200× faster for 200k spatial cells.
    """
    import sys
    if not os.path.isfile(os.path.join(SPAGE_REPO, "SpaGE", "principal_vectors.py")):
        raise RuntimeError(
            f"SpaGE is not installed at {SPAGE_REPO} (the directory is a git submodule and is "
            "empty until fetched). Run `git submodule update --init SpaGE_repo` in the "
            "repository, or set SPAGE_REPO to a clone of https://github.com/tabdelaal/SpaGE.")
    if SPAGE_REPO not in sys.path:
        sys.path.insert(0, SPAGE_REPO)
    from SpaGE.principal_vectors import PVComputation
    import scipy.stats as st
    from sklearn.neighbors import NearestNeighbors

    shared = np.intersect1d(spatial_df.columns, rna_df.columns)
    if len(shared) == 0:
        raise ValueError("No shared genes between spatial and reference data.")

    # Z-score (SpaGE normalizes internally on shared genes) — use float32 to halve memory
    rna_scaled   = pd.DataFrame(
        st.zscore(rna_df[shared],     axis=0).astype(np.float32, copy=False),
        index=rna_df.index,     columns=shared)
    spat_scaled  = pd.DataFrame(
        st.zscore(spatial_df[shared], axis=0).astype(np.float32, copy=False),
        index=spatial_df.index, columns=shared)

    # A gene that is constant in either dataset (e.g. no counts in this tissue /
    # ROI) z-scores to NaN and breaks the PCA — drop those columns.
    _ok = (np.isfinite(rna_scaled.values.sum(axis=0))
           & np.isfinite(spat_scaled.values.sum(axis=0)))
    if not _ok.all():
        print(f"  SpaGE: dropping {int((~_ok).sum())} zero-variance shared genes", flush=True)
        shared = shared[_ok]
        if len(shared) == 0:
            raise ValueError("No shared genes with non-zero variance in both datasets.")
        rna_scaled  = rna_scaled.loc[:, shared]
        spat_scaled = spat_scaled.loc[:, shared]

    n_pv = min(n_pv, len(shared))

    # Principal Vectors
    pv = PVComputation(n_factors=n_pv, n_pv=n_pv, dim_reduction="pca", dim_reduction_target="pca")
    pv.fit(rna_scaled, spat_scaled)
    S = pv.source_components_.T
    eff_pv = int(np.sum(np.diag(pv.cosine_similarity_matrix_) > 0.3))
    eff_pv = max(eff_pv, 1)
    S = S[:, :eff_pv]
    print(f"  SpaGE: {eff_pv} effective principal vectors", flush=True)

    rna_proj  = (rna_scaled.values  @ S).astype(np.float32)   # (n_rna,  eff_pv)
    spat_proj = (spat_scaled.values @ S).astype(np.float32)   # (n_spat, eff_pv)
    del rna_scaled, spat_scaled   # free ~(n_rna + n_spat) × n_shared × 4B

    # kNN in PV space
    nbrs = NearestNeighbors(n_neighbors=50, algorithm="auto", metric="cosine")
    nbrs.fit(rna_proj)
    distances, indices = nbrs.kneighbors(spat_proj)   # (n_spat, 50)
    del rna_proj, spat_proj

    # Compute weights once (shared across all genes)
    valid  = distances < 1                                            # (n_spat, 50)
    d_v    = np.where(valid, distances, 0.0).astype(np.float32)
    del distances
    d_sum  = d_v.sum(axis=1, keepdims=True)
    d_sum[d_sum == 0] = 1.0
    w      = np.where(valid, 1.0 - d_v / d_sum, 0.0).astype(np.float32)  # (n_spat, 50)
    del d_v, d_sum
    n_v    = np.maximum(valid.sum(axis=1, keepdims=True) - 1, 1).astype(np.float32)
    del valid
    w      = w / n_v                                                  # (n_spat, 50)
    del n_v

    # Impute in gene chunks to limit peak memory.
    # Y_nbrs per chunk: 222k × 50 × CHUNK × 4B — GENE_CHUNK=10 → ~440 MB vs 2.2 GB at 50.
    GENE_CHUNK = 10
    n_genes = len(genes_to_predict)
    n_spat  = len(spatial_df)
    n_chunks = (n_genes + GENE_CHUNK - 1) // GENE_CHUNK
    if n_chunks > 1:
        print(f"  SpaGE: imputing {n_genes} genes in {n_chunks} chunks of {GENE_CHUNK}", flush=True)

    if out_zarr is not None:
        # ── Streaming path: write each chunk directly to zarr (never accumulate in RAM) ──
        import zarr as _zarr
        _z = _zarr.open_array(
            out_zarr, mode='w', shape=(n_spat, n_genes), dtype=np.float32,
            chunks=(min(65536, n_spat), min(GENE_CHUNK, n_genes)),
        )
        try:
            for i in range(0, n_genes, GENE_CHUNK):
                jobs._job_check()
                jobs._job_progress(fraction=i / n_genes)
                chunk_genes = genes_to_predict[i:i + GENE_CHUNK]
                Y      = rna_df[chunk_genes].values.astype(np.float32)
                Y_nbrs = Y[indices]
                _z[:, i:i + len(chunk_genes)] = np.einsum("ij,ijk->ik", w, Y_nbrs)
                del Y, Y_nbrs
            jobs._job_shield()
        except jobs._JobCancelled:
            # an array without the 'genes' attribute is not loaded as a result, but
            # it can be many GB: do not leave it in the cache
            import shutil as _sh_z
            del _z
            _sh_z.rmtree(out_zarr, ignore_errors=True)
            raise
        _z.attrs['genes'] = genes_to_predict   # written last — marks completion
        del w, indices
        return None   # caller owns the zarr; use result_path/result_genes for lookup

    # ── In-memory path (small result) ──────────────────────────────────────────
    chunks = []
    for i in range(0, n_genes, GENE_CHUNK):
        jobs._job_check()
        jobs._job_progress(fraction=i / n_genes)
        chunk_genes = genes_to_predict[i:i + GENE_CHUNK]
        Y      = rna_df[chunk_genes].values.astype(np.float32)  # (n_rna, chunk)
        Y_nbrs = Y[indices]                                      # (n_spat, 50, chunk)
        chunks.append(np.einsum("ij,ijk->ik", w, Y_nbrs))       # (n_spat, chunk)
        del Y, Y_nbrs
    imp = np.concatenate(chunks, axis=1)                         # (n_spat, n_genes)
    del chunks, w, indices

    return pd.DataFrame(imp, index=spatial_df.index, columns=genes_to_predict)


def _run_spage_imputation(rds_path: str, n_pv: int, genes_input: str,
                          seg_source: str = None, use_corrected: bool = False) -> None:
    """Background thread: run SpaGE gene imputation and store results.
    When seg_source is a Baysor/Proseg run, imputes against reseg cells and writes
    back to the reseg SpatialData zarr (alt_res['sdata_path']).

    The spatial input is always the original counts. use_corrected=True (SPLIT-
    corrected counts as input) is refused: the app stores imputed genes as plain
    '[imp]' columns of X, so such a run would be indistinguishable from - and would
    overwrite - an imputation from the original counts. functions/impute.py
    --use_corrected does this into a separate store and records the provenance."""
    core._redirect_rpy2_console()
    _gen = jobs._dataset_gen
    _ds_key = caching._dataset_key()   # the dataset this job belongs to (for the cache index)
    _r_ref_loaded = False      # reference objects were put into R's global env

    def _set(status, message, result=None):
        with core._spage_lock:
            if jobs._stale_job(_gen, status, message):
                return
            core._spage_state["status"]  = status
            core._spage_state["message"] = message
            if result is not None:
                core._spage_state["result"] = result
            if status == "done":
                # '<gene> [imp]' vectors of a previous run must not be served again
                dataview._gene_expr_cache.clear()
                dataview._umap_fig_cache.clear()
        if status == "done":
            provenance._prov_record("spage", seg_source=seg_source, corrected=use_corrected, params={"reference": os.path.basename(str(rds_path or '')), "n_pv": n_pv, "genes": genes_input}, r_packages=("Seurat", "SeuratObject"), message=message, result=result)

    try:
        import tempfile
        import scipy.io as sio
        import rpy2.robjects as ro
        import rpy2.robjects.conversion as _rconv
        from rpy2.robjects.packages import importr

        # Ensure rpy2 conversion rules are active in this thread's context.
        # Dash callbacks run in threads where the rpy2 ContextVar may be unset.
        _rconv.set_conversion(ro.default_converter)

        if use_corrected:
            _set("error", "SpaGE on SPLIT-corrected counts is not available in the app "
                          "(imputed genes are stored as '[imp]' columns computed from the "
                          "original counts). Use functions/impute.py --use_corrected, which "
                          "writes a separate store and marks the genes as imputed from "
                          "corrected counts.")
            return

        # ── Determine active reseg (if any) ──────────────────────────────
        _alt_res = None
        if seg_source and core._seg_tool(seg_source) != "xenium":
            if core._seg_tool(seg_source) == "baysor":
                with core._baysor_lock:
                    if core._baysor_state["status"] == "done":
                        _alt_res = core._baysor_state["result"]
            elif core._seg_tool(seg_source) == "proseg":
                with core._proseg_lock:
                    if core._proseg_state["status"] == "done":
                        _alt_res = core._proseg_state["result"]
            else:
                _alt_res = core._seg_result(seg_source, done_only=True)

        # ── Load reference gene list ─────────────────────────────────────
        from rpy2.rinterface_lib import openrlib
        _set("running", "Opening Seurat reference (this may take a few minutes)…")
        with openrlib.rlock:
            importr("SeuratObject")
            importr("Matrix")
            # The path is passed as a value, never pasted into R source (a quote or
            # backslash in it would break the parse or run as code)
            _r_ref_loaded = True
            ro.globalenv["seurat_ref"] = importr("base").readRDS(rds_path)
            ro.r('rna_ref    <- seurat_ref[["RNA"]]')
            ro.r('rna_mat_data <- SeuratObject::LayerData(rna_ref, layer="data")')
            ro.r('rna_genes  <- rownames(rna_mat_data)')
            rna_genes_all = list(ro.r("rna_genes"))

        # Use only panel (non-imputed) genes for shared-gene computation
        gene_var = core.DATA.get("gene_var")
        if gene_var is not None and "is_imputed" in gene_var.columns:
            panel_gene_names = [g for g in core.DATA["gene_names"] if not gene_var.loc[g, "is_imputed"]]
        else:
            panel_gene_names = list(core.DATA["gene_names"])
        xenium_genes = set(panel_gene_names)
        shared_genes = sorted(xenium_genes & set(rna_genes_all))
        if not shared_genes:
            _set("error", "No shared genes between Xenium panel and reference!")
            return
        print(f"  SpaGE: {len(shared_genes)} shared genes with reference", flush=True)

        # ── Determine genes_to_predict ───────────────────────────────────
        auto_hvg = not (genes_input and genes_input.strip())
        _auto_top = None
        if auto_hvg:
            # Top-200 genes by variance over ALL non-panel genes of the reference
            # (computed in R on the sparse matrix, in blocks of cells). Before, only
            # the first 2000 non-panel genes in the reference's row order were
            # candidates, so the choice depended on that order. "auto200v2": results
            # cached under the old selection are not reused.
            predict_candidates = [g for g in rna_genes_all if g not in xenium_genes]
            genes_key = "auto200v2"
            _set("running", f"Ranking {len(predict_candidates):,} reference genes by variance…")
            try:
                with openrlib.rlock:
                    ro.globalenv[".spage_cand"] = ro.StrVector(predict_candidates)
                    _auto_top = [str(g) for g in ro.r(_SPAGE_HVG_R)]
                    ro.r('rm(.spage_cand)')
            except Exception as _hve:
                print(f"  SpaGE: variance ranking over the whole reference failed ({_hve}) — "
                      f"falling back to the first 2000 non-panel genes as candidates", flush=True)
                _auto_top = None
            if _auto_top:
                genes_needed = shared_genes + _auto_top
                print(f"  SpaGE: auto-selected {len(_auto_top)} highly variable genes out of "
                      f"{len(predict_candidates):,} non-panel reference genes", flush=True)
            else:
                genes_needed = shared_genes + predict_candidates[:2000]
                genes_key = "auto200"
        else:
            requested = [g.strip() for line in genes_input.replace(",", "\n").split("\n")
                         for g in [line.strip()] if g]
            genes_to_predict = [g for g in requested if g in rna_genes_all and g not in xenium_genes]
            if not genes_to_predict:
                _set("error", "None of the requested genes found in reference (or all in Xenium panel).")
                return
            genes_needed = shared_genes + genes_to_predict
            genes_key = ",".join(sorted(genes_to_predict))

        # Reseg runs get a separate cache key so their imputed values don't clash with Xenium.
        # use_corrected also gets its own key so corrected/uncorrected runs don't share caches.
        _cache_corr_suffix = "_corr" if use_corrected else ""
        _cache_seg_suffix = (f"__{seg_source.replace(':', '_')}{_cache_corr_suffix}"
                             if (seg_source and seg_source != "xenium") else _cache_corr_suffix)
        # n_pv and an active subset() change the result; they are added to the key only
        # when they differ from the default / full dataset so existing caches stay valid.
        _cache_extra = f"_pv{n_pv}" if int(n_pv) != 50 else ""
        if _alt_res is not None and "_cells_df_original" in _alt_res:
            _cache_extra += f"_n{len(_alt_res['cells_df'])}"
        elif _alt_res is None and "_df_original" in core.DATA:
            _cache_extra += f"_n{len(core.DATA['df'])}"
        cache_file = _spage_cache_path(rds_path, genes_key + _cache_seg_suffix + _cache_extra)
        zarr_cache = cache_file.replace(".parquet", ".zarr")
        # A streamed result has one row per current DATA["df"] row and no cell ids:
        # under a subset() it must not be registered for auto-load at startup.
        _xen_subset = _alt_res is None and "_df_original" in core.DATA

        def _record_reseg_result(_path, _genes):
            """Reseg results are indexed by reseg cells: keep them on the run, not
            in the global (Xenium) SpaGE state / startup index."""
            try:
                import json as _json_sc
                with open(os.path.join(_alt_res["out_dir"], "spage_result.json"), "w") as _sjf:
                    _json_sc.dump({"path": _path, "genes": list(_genes)}, _sjf)
            except Exception as _sje:
                print(f"  SpaGE: could not persist result ref: {_sje}", flush=True)
            _alock = core._seg_state(core._seg_tool(seg_source or ""))[1] or core._proseg_lock
            with _alock:
                _alt_res["spage_result_path"]  = _path
                _alt_res["spage_result_genes"] = list(_genes)

        jobs._check_dataset_gen(_gen)
        if os.path.isdir(zarr_cache):
            try:
                import zarr as _zarr_c
                _zc = _zarr_c.open_array(zarr_cache, mode='r')
                _genes_c = list(_zc.attrs.get('genes', []))
                if _genes_c:
                    if _alt_res is None:
                        with core._spage_lock:
                            core._spage_state["result_path"]  = zarr_cache
                            core._spage_state["result_genes"] = _genes_c
                    if _alt_res is not None:
                        try:
                            import json as _json_sc
                            _sj = {"path": zarr_cache, "genes": _genes_c}
                            with open(os.path.join(_alt_res["out_dir"], "spage_result.json"), "w") as _sjf:
                                _json_sc.dump(_sj, _sjf)
                        except Exception:
                            pass
                        _alock = core._seg_state(core._seg_tool(seg_source or ""))[1] or core._proseg_lock
                        with _alock:
                            _alt_res["spage_result_path"]  = zarr_cache
                            _alt_res["spage_result_genes"] = _genes_c
                    elif not _xen_subset:
                        _spage_index_update(zarr_cache, _ds_key)
                    _set("done", f"Done (cached) — {len(_genes_c):,} genes (streaming)", result=None)
                    return
            except Exception as _zce:
                print(f"  SpaGE: zarr cache load failed: {_zce} — re-running", flush=True)
        if os.path.exists(cache_file):
            _set("running", "Loading cached SpaGE results…")
            try:
                import pyarrow.parquet as _pq_c
                _pf_c = _pq_c.ParquetFile(cache_file)
                _nc_c = len(_pf_c.schema_arrow.names)
                _nr_c = _pf_c.metadata.num_rows
                if _nr_c * _nc_c > core.SPAGE_STREAM_THRESHOLD:
                    _genes_c = _pf_c.schema_arrow.names
                    if _alt_res is None:
                        with core._spage_lock:
                            core._spage_state["result_path"]  = cache_file
                            core._spage_state["result_genes"] = list(_genes_c)
                        if not _xen_subset:
                            _spage_index_update(cache_file, _ds_key)
                    else:
                        _record_reseg_result(cache_file, _genes_c)
                    _set("done", f"Done (cached, streaming) — {_nc_c:,} genes", result=None)
                    return
            except Exception:
                pass
            imp_df = pd.read_parquet(cache_file)
            if _alt_res is None:
                _spage_index_update(cache_file, _ds_key)
            else:
                _record_reseg_result(cache_file, list(imp_df.columns))
            _set("done", f"Done (cached) — {len(imp_df.columns):,} genes imputed",
                 result=imp_df if _alt_res is None else None)
            return

        # ── Extract submatrix from R and save to temp files ──────────────
        n_predict_preview = len(genes_needed) - len(shared_genes)
        _set("running", f"Extracting {len(shared_genes)} shared + {n_predict_preview} predict genes…")

        with tempfile.TemporaryDirectory() as tmpdir:
            mat_file   = os.path.join(tmpdir, "mat.mtx").replace("\\", "/")
            genes_file = os.path.join(tmpdir, "genes.txt").replace("\\", "/")
            cells_file = os.path.join(tmpdir, "cells.txt").replace("\\", "/")

            # Build R character vector of needed genes
            genes_r = "c(" + ",".join(f'"{g}"' for g in genes_needed) + ")"
            with openrlib.rlock:
                ro.r(f"""
genes_needed <- {genes_r}
genes_needed <- genes_needed[genes_needed %in% rna_genes]
mat_sub <- rna_mat_data[genes_needed, ]
Matrix::writeMM(mat_sub, "{mat_file}")
writeLines(rownames(mat_sub), "{genes_file}")
writeLines(colnames(mat_sub), "{cells_file}")
""")
            mat_sp = sio.mmread(mat_file).T.tocsr()   # cells × genes
            with open(genes_file) as f:
                rna_genes_sub = [l.strip() for l in f]
            with open(cells_file) as f:
                rna_cells = [l.strip() for l in f]

        # Free R memory
        with openrlib.rlock:
            ro.r("rm(seurat_ref, rna_ref, rna_mat_data, mat_sub); gc()")

        # ── Convert to dense DataFrame ───────────────────────────────────
        _set("running", f"Building expression matrices ({len(rna_cells):,} ref cells)…")
        rna_df = pd.DataFrame(
            mat_sp.toarray().astype(np.float32),
            index=rna_cells, columns=rna_genes_sub,
        )
        del mat_sp  # sparse copy no longer needed

        # HVG selection for auto mode
        if auto_hvg and _auto_top:
            genes_to_predict = [g for g in _auto_top if g in rna_genes_sub]
        elif auto_hvg:
            predict_pool = [g for g in rna_genes_sub if g not in xenium_genes]
            var_scores = rna_df[predict_pool].var(axis=0)
            genes_to_predict = var_scores.nlargest(200).index.tolist()
            print(f"  SpaGE: auto-selected {len(genes_to_predict)} HVGs", flush=True)
        else:
            genes_to_predict = [g for g in genes_to_predict if g in rna_genes_sub]

        if not genes_to_predict:
            _set("error", "No valid genes to predict after filtering against reference.")
            return

        # ── Build normalized expression matrix for SpaGE input ───────────
        shared_in_rna = [g for g in shared_genes if g in rna_genes_sub]

        if _alt_res is not None:
            # ── Reseg path: use alt_res["expr"] (panel genes × reseg cells) ──
            if use_corrected:
                reseg_expr = _alt_res.get("split_corrected_expr")
                if reseg_expr is None:
                    _set("error", "No SPLIT-corrected counts for this segmentation — run SPLIT first.")
                    return
                print("  SpaGE: using SPLIT-corrected counts (reseg)", flush=True)
            else:
                reseg_expr = _alt_res.get("expr")
            if reseg_expr is None:
                _set("error", "Reseg has no expression matrix — reload segmentation.")
                return
            # Panel gene index (columns 0..n_panel-1 of reseg expr)
            panel_gni = {g: i for i, g in enumerate(panel_gene_names)}
            shared_in_rna_use = [g for g in shared_in_rna if g in panel_gni]
            xen_idx_use = [panel_gni[g] for g in shared_in_rna_use]
            if not shared_in_rna_use:
                _set("error", "No shared genes found in reseg expression matrix.")
                return
            xen_raw = reseg_expr[:, xen_idx_use].toarray().astype(np.float32, copy=False)
            rs = xen_raw.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
            # in place: no extra cells × shared-genes temporaries
            xen_raw /= rs; xen_raw *= 10_000; np.log1p(xen_raw, out=xen_raw)
            xen_norm = xen_raw
            del xen_raw, rs
            spatial_df = pd.DataFrame(xen_norm, index=_alt_res["cells_df"].index,
                                      columns=shared_in_rna_use)
            del xen_norm
        else:
            # ── Xenium path ───────────────────────────────────────────────
            if use_corrected:
                _xenium_expr = core.DATA.get("split_corrected_expr")
                if _xenium_expr is None:
                    _set("error", "No SPLIT-corrected counts available — run SPLIT first.")
                    return
                print("  SpaGE: using SPLIT-corrected counts (Xenium)", flush=True)
            else:
                _xenium_expr = core.DATA.get("expr_csc")# or DATA["expr"]
            panel_gni = core.DATA["gene_name_to_idx"]
            shared_in_rna_use = [g for g in shared_in_rna if g in panel_gni]
            xen_idx = [panel_gni[g] for g in shared_in_rna_use]
            xen_raw = _xenium_expr[:, xen_idx].toarray().astype(np.float32, copy=False)
            rs = xen_raw.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
            # in place: no extra cells × shared-genes temporaries
            xen_raw /= rs; xen_raw *= 10_000; np.log1p(xen_raw, out=xen_raw)
            xen_norm = xen_raw
            del xen_raw, rs
            df  = core.DATA["df"]
            idx = np.asarray(core.DATA["df_to_expr"])
            good = idx >= 0
            if (len(idx) == xen_norm.shape[0] and good.all()
                    and np.array_equal(idx, np.arange(len(idx)))):
                xen_aligned = xen_norm   # identity mapping: no second copy
            else:
                xen_aligned = np.zeros((len(df), len(shared_in_rna_use)), dtype=np.float32)
                xen_aligned[good] = xen_norm[idx[good]]
            del xen_norm
            spatial_df = pd.DataFrame(xen_aligned, index=df.index, columns=shared_in_rna_use)
            del xen_aligned

        # ── Run SpaGE — streaming or in-memory based on output size ─────
        _n_out = len(spatial_df) * len(genes_to_predict)
        # Large cell counts also stream: the in-memory path appends dense imputed
        # columns to the whole expression matrix and rewrites the table.
        _streaming = _n_out > core.SPAGE_STREAM_THRESHOLD or len(spatial_df) > 400_000
        if _streaming:
            print(f"  SpaGE: large output ({len(spatial_df):,}×{len(genes_to_predict):,})"
                  f" → streaming to {os.path.basename(zarr_cache)}", flush=True)
            _set("running", f"Running SpaGE (n_pv={n_pv}, {len(genes_to_predict):,} genes, streaming)…")
            _vectorized_spage(spatial_df, rna_df, n_pv, genes_to_predict, out_zarr=zarr_cache)
            del spatial_df, rna_df
            jobs._check_dataset_gen(_gen)
            if _alt_res is None:
                with core._spage_lock:
                    core._spage_state["result_path"]  = zarr_cache
                    core._spage_state["result_genes"] = genes_to_predict
                if not _xen_subset:
                    _spage_index_update(zarr_cache, _ds_key)
            else:
                try:
                    import json as _json_sc
                    _sj = {"path": zarr_cache, "genes": genes_to_predict}
                    with open(os.path.join(_alt_res["out_dir"], "spage_result.json"), "w") as _sjf:
                        _json_sc.dump(_sj, _sjf)
                except Exception as _sje:
                    print(f"  SpaGE: could not persist result ref: {_sje}", flush=True)
                _alock = core._seg_state(core._seg_tool(seg_source or ""))[1] or core._proseg_lock
                with _alock:
                    _alt_res["spage_result_path"]  = zarr_cache
                    _alt_res["spage_result_genes"] = genes_to_predict
            _set("done", f"Done — {len(genes_to_predict):,} genes imputed (streaming)")
            return

        _set("running", f"Running SpaGE (n_pv={n_pv}, {len(genes_to_predict)} genes)…")
        imp_df = _vectorized_spage(spatial_df, rna_df, n_pv, genes_to_predict)
        del spatial_df, rna_df

        # ── Cache to parquet ──────────────────────────────────────────────
        _set("running", "Saving imputed results…")
        # Last cancellation point. The cache file, the run's result reference, the
        # zarr table (deleted, then rewritten) and DATA are written as one unit.
        jobs._job_shield()
        jobs._job_progress(fraction=None)
        imp_df.to_parquet(cache_file)
        if _alt_res is None:
            _spage_index_update(cache_file, _ds_key)
        else:
            try:
                import json as _json_sc
                _sj = {"path": cache_file, "genes": genes_to_predict}
                with open(os.path.join(_alt_res["out_dir"], "spage_result.json"), "w") as _sjf:
                    _json_sc.dump(_sj, _sjf)
            except Exception as _sje:
                print(f"  SpaGE: could not persist result ref: {_sje}", flush=True)
            _alock = core._seg_state(core._seg_tool(seg_source or ""))[1] or core._proseg_lock
            with _alock:
                _alt_res["spage_result_path"]  = cache_file
                _alt_res["spage_result_genes"] = genes_to_predict
        print(f"  SpaGE: {len(imp_df.columns)} genes cached to {cache_file}", flush=True)

        def _spage_replace_cols(X, col_idx, dense_vals):
            """X (sparse, cells × genes) with the columns col_idx replaced by dense_vals."""
            Xc = sp.csc_matrix(X, copy=True)
            for _j in col_idx:
                Xc.data[Xc.indptr[_j]:Xc.indptr[_j + 1]] = 0
            Xc.eliminate_zeros()
            _vals = np.asarray(dense_vals, dtype=np.float32)
            _rr = np.tile(np.arange(Xc.shape[0]), len(col_idx))
            _cc = np.repeat(np.asarray(col_idx), Xc.shape[0])
            _add = sp.csc_matrix((_vals.ravel(order="F"), (_rr, _cc)), shape=Xc.shape)
            return (Xc + _add).tocsr()

        def _spage_write_zarr(sdata_path, imp_df_sub, lock_ctx=None):
            """Append imputed gene columns to a SpatialData zarr table. Returns list of new genes written."""
            _zlock = reseg._zarr_write_lock(sdata_path)
            _zlock.acquire()
            try:
                import spatialdata as _sd_w, anndata as _ad_w
                import shutil as _sh_w
                _sdata_w = _sd_w.read_zarr(sdata_path)
                _adata_w = _sdata_w.tables["table"]
                # 'GENE [imp]' var names of an older functions/impute.py store → plain
                # names, so a re-imputed gene replaces its column instead of adding a
                # second one (the rewritten table then uses the plain names)
                loading._plain_imputed_var_names(_adata_w)
                # Imputed rows follow the cells the run saw; the table has its own
                # row order. Pair them by cell id and write only a complete match
                # (under a subset() the result stays in the SpaGE cache only).
                _obs_ids = pd.Index(_adata_w.obs_names.astype(str))
                _imp_ids = pd.Index(imp_df_sub.index.astype(str))
                _pos = _imp_ids.get_indexer(_obs_ids) if _imp_ids.is_unique \
                    else np.full(len(_obs_ids), -1)
                if (_pos < 0).any() and _imp_ids.is_unique:
                    _pos = _imp_ids.get_indexer(_obs_ids.str.replace(r'^p\d+_', '', regex=True))
                if len(_imp_ids) != len(_obs_ids) or (_pos < 0).any():
                    print("  SpaGE: imputed cells do not match the zarr table "
                          "(subset active?) — zarr write-back skipped", flush=True)
                    return [], None
                if np.array_equal(_pos, np.arange(len(_pos))):
                    _pos = slice(None)   # same order: no reordered copy of the values
                _existing = set(_adata_w.var_names)
                _new_genes = [g for g in imp_df_sub.columns if g not in _existing]
                # Genes imputed by an earlier run get their column replaced
                _is_imp = (_adata_w.var["is_imputed"].astype(bool)
                           if "is_imputed" in _adata_w.var.columns
                           else pd.Series(False, index=_adata_w.var_names))
                _re_genes = [g for g in imp_df_sub.columns
                             if g in _existing and bool(_is_imp.get(g, False))]
                if not _new_genes and not _re_genes:
                    return [], None
                _old_X = sp.csr_matrix(_adata_w.X)
                if _re_genes:
                    _old_X = _spage_replace_cols(
                        _old_X, [_adata_w.var_names.get_loc(g) for g in _re_genes],
                        imp_df_sub[_re_genes].values[_pos])
                _n_new = len(_new_genes)
                _new_X = sp.csr_matrix(imp_df_sub[_new_genes].values[_pos].astype(np.float32))
                _new_var = pd.DataFrame(
                    {"is_imputed": True},
                    index=pd.Index(_new_genes, name=_adata_w.var.index.name or "gene"),
                )
                # obs columns added through the low-level zarr API (UMAP, SPLIT
                # clusters, ROI tags) are not returned by read_zarr: carry them over.
                _obs_w = _adata_w.obs.copy()
                reseg._load_zarr_obs_into_df(sdata_path, _obs_w)
                _adata_new = _ad_w.AnnData(
                    X=sp.hstack([_old_X, _new_X]).tocsr(),
                    obs=_obs_w,
                    var=pd.concat([_adata_w.var, _new_var]) if _n_new else _adata_w.var.copy(),
                    uns=dict(_adata_w.uns),
                    obsm=dict(_adata_w.obsm),
                    # other layers (SPLIT's X_corrected) get zero columns for the new genes
                    layers={_k: sp.hstack([sp.csr_matrix(_v), sp.csr_matrix(
                                (_v.shape[0], _n_new), dtype=_v.dtype)]).tocsr()
                            for _k, _v in _adata_w.layers.items()},
                )
                _adata_new.uns["imputed_genes"] = list(dict.fromkeys(
                    list(_adata_w.uns.get("imputed_genes", [])) + _new_genes + _re_genes))
                # a column re-imputed here now holds values from the original counts
                if len(_adata_w.uns.get("split_corrected_imputed_genes", [])):
                    _adata_new.uns["split_corrected_imputed_genes"] = [
                        g for g in _adata_w.uns["split_corrected_imputed_genes"]
                        if g not in _re_genes]
                _sdata_w.tables["table"] = _adata_new
                # The table on disk has to be deleted before it can be rewritten:
                # keep a copy (and the root metadata) to put back if the write fails.
                _root_w  = str(sdata_path).rstrip("/\\")
                _tbl_dir = os.path.join(_root_w, "tables", "table")
                _bak_dir = _root_w + ".spage_table_bak"
                _meta_w  = [f for f in ("zarr.json", ".zmetadata", ".zattrs", ".zgroup")
                            if os.path.isfile(os.path.join(_root_w, f))]
                _sh_w.rmtree(_bak_dir, ignore_errors=True)
                _sh_w.copytree(_tbl_dir, os.path.join(_bak_dir, "table"))
                for _f in _meta_w:
                    _sh_w.copy2(os.path.join(_root_w, _f), os.path.join(_bak_dir, _f))
                try:
                    _sdata_w.delete_element_from_disk("table")
                    _sdata_w.write_element("table")
                except Exception:
                    _sh_w.rmtree(_tbl_dir, ignore_errors=True)
                    _sh_w.copytree(os.path.join(_bak_dir, "table"), _tbl_dir)
                    for _f in _meta_w:
                        _sh_w.copy2(os.path.join(_bak_dir, _f), os.path.join(_root_w, _f))
                    _sh_w.rmtree(_bak_dir, ignore_errors=True)
                    raise
                _sh_w.rmtree(_bak_dir, ignore_errors=True)
                return _new_genes, _adata_new
            except Exception as _ze:
                print(f"  SpaGE: zarr write-back failed (non-fatal): {_ze}", flush=True)
                return [], None
            finally:
                _zlock.release()

        if _alt_res is not None:
            # ── Reseg write-back: update alt_res["expr"] and per-reseg gene index ──
            sdata_path = _alt_res.get("sdata_path", "")
            if sdata_path and os.path.isdir(sdata_path):
                _set("running", "Writing imputed genes to reseg SpatialData zarr…")
                result_pair = _spage_write_zarr(sdata_path, imp_df)
                new_genes = result_pair[0] if result_pair else []
                adata_new = result_pair[1] if len(result_pair) > 1 else None
                if adata_new is not None:
                    print(f"  SpaGE reseg: wrote {len(new_genes)} genes to {os.path.basename(sdata_path)}", flush=True)
                    with core._spage_lock:
                        _prev_imp = list(_alt_res.get("imputed_genes") or [])
                        _prev_gni = _alt_res.get("gene_name_to_idx") or {}
                        # Genes imputed before: replace their columns in memory too
                        _re_genes = [g for g in imp_df.columns
                                     if g in _prev_imp and g in _prev_gni
                                     and _prev_gni[g] < _alt_res["expr"].shape[1]]
                        if _re_genes:
                            _alt_res["expr"] = _spage_replace_cols(
                                _alt_res["expr"], [_prev_gni[g] for g in _re_genes],
                                imp_df[_re_genes].values)
                        if new_genes:
                            # Extend alt_res["expr"] with new imputed columns
                            new_cols = sp.csr_matrix(imp_df[new_genes].values.astype(np.float32))
                            _alt_res["expr"] = sp.hstack([_alt_res["expr"], new_cols]).tocsr()
                        if _alt_res["expr"].shape[1] == adata_new.n_vars:
                            # Same columns as the zarr table (also for a run reloaded
                            # from a zarr that already held imputed genes)
                            _alt_res["gene_name_to_idx"] = {
                                g: i for i, g in enumerate(adata_new.var_names)}
                            _alt_res["imputed_genes"] = [
                                g for g in adata_new.var_names
                                if bool(adata_new.var.loc[g, "is_imputed"])]
                        elif new_genes:
                            # Rebuild per-reseg gene index (preserves order)
                            prev_gni = _alt_res.get("gene_name_to_idx") or core.DATA["gene_name_to_idx"]
                            prev_names = sorted(prev_gni, key=prev_gni.__getitem__)
                            extended_names = prev_names + new_genes
                            _alt_res["gene_name_to_idx"] = {g: i for i, g in enumerate(extended_names)}
                            _alt_res["imputed_genes"] = _prev_imp + new_genes
        else:
            # ── Xenium write-back: update DATA["expr"] and gene_names ─────────────
            sdata_path = core.DATA.get("sdata_path", "")
            if sdata_path and os.path.isdir(sdata_path):
                _set("running", "Writing imputed genes to SpatialData zarr…")
                result_pair = _spage_write_zarr(sdata_path, imp_df)
                new_genes = result_pair[0] if result_pair else []
                adata_new  = result_pair[1] if len(result_pair) > 1 else None
                if adata_new is not None:
                    print(f"  SpaGE: wrote {len(new_genes)} imputed genes to zarr", flush=True)
                    with core._spage_lock:
                        jobs._check_dataset_gen(_gen)
                        core.DATA["gene_names"]       = list(adata_new.var_names)
                        core.DATA["gene_name_to_idx"] = {g: i for i, g in enumerate(core.DATA["gene_names"])}
                        core.DATA["expr"]             = sp.csr_matrix(adata_new.X)
                        core.DATA["expr_csc"]         = core.DATA["expr"].tocsc()
                        core.DATA["gene_var"]         = adata_new.var
                        if core.DATA.get("split_corrected_imputed_genes"):
                            core.DATA["split_corrected_imputed_genes"] = [
                                str(g) for g in
                                adata_new.uns.get("split_corrected_imputed_genes", [])]
                        dataview._gene_expr_cache.clear()

        _set("done", f"Done — {len(imp_df.columns):,} genes imputed",
             result=imp_df if _alt_res is None else None)

    except jobs._JobCancelled:
        raise                      # reported by _job_finish; the finally below still runs
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set("error", str(exc)[:300])
    finally:
        # Cached-result returns and errors leave before the rm() above: without this
        # the multi-GB reference would stay in the R heap for the life of the process.
        if _r_ref_loaded:
            try:
                import rpy2.robjects as _ro_fin
                from rpy2.rinterface_lib import openrlib as _orl_fin
                with _orl_fin.rlock:
                    _ro_fin.r('suppressWarnings(rm(list = intersect(c("seurat_ref", "rna_ref", '
                              '"rna_mat_data", "mat_sub", "rna_genes", "genes_needed", ".spage_cand"), '
                              'ls(all.names = TRUE, envir = globalenv())), envir = globalenv())); '
                              'invisible(gc())')
            except Exception:
                pass


# R: the 200 highest-variance genes among .spage_cand (rows of rna_mat_data), with the
# row sums accumulated over blocks of cells so that no second copy of the matrix is made.
_SPAGE_HVG_R = """
local({
  m <- rna_mat_data; n <- ncol(m)
  s1 <- numeric(nrow(m)); s2 <- numeric(nrow(m))
  for (j0 in seq(1L, n, by = 20000L)) {
    b  <- m[, j0:min(n, j0 + 19999L), drop = FALSE]
    s1 <- s1 + Matrix::rowSums(b)
    s2 <- s2 + Matrix::rowSums(b * b)
  }
  v <- (s2 - s1 * s1 / n) / max(n - 1, 1)
  names(v) <- rownames(m)
  cand <- .spage_cand[.spage_cand %in% names(v)]
  v <- v[cand]
  cand[order(-v)[seq_len(min(200L, length(cand)))]]
})
"""


def run_spage(rds_path: str, genes_file: str = None, n_pv: int = 50,
              use_corrected: bool = None) -> None:
    """
    Run SpaGE gene imputation from the REPL.

    Parameters
    ----------
    rds_path : str
        Path to the Seurat RDS reference file.
    genes_file : str, optional
        Path to a .txt file with gene names to impute (one per line, or
        comma-separated). If None, the top-200 highly variable genes from
        the reference are auto-selected.
    n_pv : int
        Number of principal vectors for cross-dataset alignment (default 50).
    use_corrected : bool, optional
        Not supported in the app: imputation always uses the original counts,
        whatever the Active Counts mode. Passing True prints why and does nothing;
        use functions/impute.py --use_corrected to impute from SPLIT-corrected
        counts into a separate store.

    Example
    -------
    run_spage(
        rds_path="/path/to/reference.rds",
        genes_file="/path/to/genes_to_impute.txt",
        n_pv=50,
    )
    """
    rds_path = (rds_path or core.REF_RDS_DEFAULT or "").strip()
    _problem = core._ref_rds_problem(rds_path)
    if _problem:
        print(f"  SpaGE: {_problem} Nothing was run.", flush=True)
        return
    with core._spage_lock:
        if core._spage_state["status"] == "running":
            print("  SpaGE is already running.", flush=True)
            return

    if use_corrected:
        print("  SpaGE: use_corrected=True is not supported in the app - imputed genes are "
              "stored as '[imp]' columns computed from the original counts, and a run on "
              "corrected counts would overwrite them under the same name.\n"
              "  Use functions/impute.py --use_corrected (writes a separate store and marks "
              "the genes as imputed from corrected counts). Nothing was run.", flush=True)
        return
    if use_corrected is None and core._active_counts_mode == "corrected":
        print("  SpaGE: Active Counts is 'SPLIT Corrected', but imputation uses the "
              "original counts", flush=True)
    use_corrected = False

    # Detect active reseg so imputation runs against the right cells
    seg_source = None
    with core._proseg_lock:
        if core._proseg_state["status"] == "done" and core._proseg_state.get("result"):
            _res_od = core._proseg_state["result"].get("out_dir", "")
            _tag = os.path.basename(_res_od).rsplit("_", 1)[-1] if _res_od else ""
            seg_source = f"proseg:{_tag}" if _tag else "proseg"
    if seg_source is None:
        with core._baysor_lock:
            if core._baysor_state["status"] == "done" and core._baysor_state.get("result"):
                _res_od = core._baysor_state["result"].get("out_dir", "")
                _tag = os.path.basename(_res_od).rsplit("_", 1)[-1] if _res_od else ""
                seg_source = f"baysor:{_tag}" if _tag else "baysor"
    if seg_source is None:
        with core._imgseg_lock:
            if core._imgseg_state["status"] == "done" and core._imgseg_state.get("result"):
                _res_od = core._imgseg_state["result"].get("out_dir", "")
                _tag = os.path.basename(_res_od).rsplit("_", 1)[-1] if _res_od else ""
                seg_source = f"imgseg:{_tag}" if _tag else "imgseg"
    if seg_source:
        print(f"  SpaGE: reseg mode — imputing against {core._seg_tool(seg_source)} cells", flush=True)

    # Build genes_input string (same format _run_spage_imputation expects)
    genes_input = ""
    if genes_file:
        try:
            with open(genes_file) as fh:
                genes_input = fh.read()
            n_genes = len([g for line in genes_input.replace(",", "\n").splitlines()
                           for g in [line.strip()] if g])
            print(f"  SpaGE: loaded {n_genes} genes from {genes_file}", flush=True)
        except FileNotFoundError:
            print(f"  SpaGE: file not found: {genes_file}", flush=True)
            return

    print(f"  SpaGE: starting imputation (n_pv={n_pv})…", flush=True)
    with core._spage_lock:
        core._spage_state.update({"status": "running", "message": "Starting…", "result": None})
    core._spage_repl_pending = True

    jobs._job_start("spage", "SpaGE imputation", _run_spage_imputation,
               args=(rds_path, n_pv, genes_input, seg_source),
               kwargs=dict(use_corrected=use_corrected),
               state=core._spage_state, lock=core._spage_lock, copy_context=True)
