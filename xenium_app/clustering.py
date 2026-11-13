"""Clustering and UMAP of a resegmentation result."""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import core, jobs, dataview, reseg


# ─── Reseg clustering + UMAP ─────────────────────────────────────────────────

# Above this many cells UMAP runs unseeded on all cores (a fixed random_state
# forces umap-learn to one thread) and the t-SNE fallback is skipped.
_UMAP_PARALLEL_MIN_CELLS = 300_000


def _umap_seed_kwargs(n_cells: int, seed: int) -> dict:
    if n_cells <= _UMAP_PARALLEL_MIN_CELLS:
        return {"random_state": seed}
    print(f"  UMAP: {n_cells:,} cells — running multi-threaded without a fixed seed "
          f"(layout not reproducible between runs)", flush=True)
    return {"n_jobs": -1}


def _compute_reseg_clusters_umap(cells_df: pd.DataFrame, expr_mat,
                                  progress_fn=None) -> None:
    """Compute KMeans clustering and UMAP for reseg cells in-place on cells_df.
    Adds columns: cluster_kmeans_10, umap_1, umap_2.
    Also updates _umap_reseg_state so the UMAP scatter plot works immediately.

    Never discards a finished segmentation: with too few cells to embed, or when the
    clustering itself fails, every cell gets cluster 1 and an empty (NaN) embedding."""
    if expr_mat is None:
        return

    def _trivial(reason):
        print(f"  Reseg: no clustering/UMAP for {len(cells_df):,} cell(s) ({reason}) — "
              f"all cells in cluster 1, no embedding", flush=True)
        cells_df["cluster_kmeans_10"] = 1
        cells_df["umap_1"] = np.nan
        cells_df["umap_2"] = np.nan

    if len(cells_df) < 4:
        _trivial("fewer than 4 cells")
        return
    try:
        _compute_reseg_clusters_umap_impl(cells_df, expr_mat, progress_fn)
    except jobs._DatasetChanged:
        raise
    except Exception as _cl_err:
        import traceback as _tb_cl
        _tb_cl.print_exc()
        _trivial(f"failed: {_cl_err}")


def _reseg_measured_expr(expr_mat):
    """Columns of a Baysor/Proseg matrix that hold measured counts. A run's matrix is
    [panel genes | genes SpaGE imputed for this run]; the panel genes come first."""
    _n = min(expr_mat.shape[1], len(dataview._panel_gene_names()))
    return expr_mat if _n == expr_mat.shape[1] else expr_mat[:, :_n]


def _compute_reseg_clusters_umap_impl(cells_df: pd.DataFrame, expr_mat,
                                       progress_fn=None) -> None:
    from sklearn.decomposition import TruncatedSVD

    n_cells = len(cells_df)
    if progress_fn:
        progress_fn(f"Normalising {n_cells:,} cells for UMAP/clustering…")

    # len(_panel_gene_names()), not len(DATA["gene_names"]): the latter also counts
    # genes imputed for the Xenium cells
    mat = _reseg_measured_expr(expr_mat).astype("float32").tocsr()
    row_sums = np.asarray(mat.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    mat = sp.diags(1e4 / row_sums).dot(mat).tocsr()
    mat.data = np.log1p(mat.data)

    n_comp = min(50, n_cells - 1, mat.shape[1] - 1)
    if progress_fn:
        progress_fn(f"PCA ({n_comp} components)…")
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    pca_coords = svd.fit_transform(mat)

    # KMeans clustering
    from sklearn.cluster import MiniBatchKMeans
    n_clusters = min(10, n_cells)
    if progress_fn:
        progress_fn(f"KMeans ({n_clusters} clusters)…")
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
    cells_df["cluster_kmeans_10"] = km.fit_predict(pca_coords) + 1  # 1-indexed

    # UMAP or t-SNE fallback
    if progress_fn:
        progress_fn("Computing UMAP…")
    try:
        import sys, types
        # Stub tensorflow so umap/__init__.py can load parametric_umap without
        # crashing on the NumPy 1.x vs 2.x ABI mismatch in the installed TF.
        if "tensorflow" not in sys.modules:
            sys.modules["tensorflow"] = types.ModuleType("tensorflow")
        import warnings
        from umap.umap_ import UMAP
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="n_jobs value.*overridden")
            reducer = UMAP(n_neighbors=min(15, n_cells - 1), min_dist=0.1,
                           **_umap_seed_kwargs(n_cells, 42))
            umap_coords = reducer.fit_transform(pca_coords)
    except Exception as _umap_err:
        if n_cells > _UMAP_PARALLEL_MIN_CELLS:
            # t-SNE on this many cells would not finish; leave the embedding empty
            print(f"  Reseg: UMAP failed ({_umap_err}) — no embedding for {n_cells:,} cells", flush=True)
            umap_coords = np.full((n_cells, 2), np.nan)
        else:
            from sklearn.manifold import TSNE
            perp = max(1, min(30, max(5, n_cells // 10), n_cells - 1))   # must be < n_cells
            umap_coords = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(pca_coords)

    cells_df["umap_1"] = umap_coords[:, 0]
    cells_df["umap_2"] = umap_coords[:, 1]

    # Store result in _umap_reseg_state so the existing UMAP scatter works
    umap_df = pd.DataFrame(umap_coords, index=cells_df.index, columns=["umap_1", "umap_2"])
    # Reported first: a caller whose dataset was replaced meanwhile stops here
    if progress_fn:
        progress_fn(f"UMAP + clustering done ({n_cells:,} cells)")
    with core._umap_reseg_lock:
        # A "Rerun UMAP" job owns the state while it runs; publishing here would end
        # its poll early. The embedding is in cells_df["umap_1/2"] either way.
        if core._umap_reseg_state["status"] != "running":
            core._umap_reseg_state["status"]  = "done"
            core._umap_reseg_state["message"] = f"Done — {n_cells:,} cells"
            core._umap_reseg_state["result"]  = umap_df
            # Owning run unknown here; make_umap_fig then reads umap_1/2 from cells_df
            core._umap_reseg_state["owner"]   = None
    print(f"  Reseg: UMAP + {n_clusters} KMeans clusters computed for {n_cells:,} cells", flush=True)


# ─── Reseg UMAP computation (manual re-run) ───────────────────────────────────

def _run_reseg_umap() -> None:
    """Background thread: compute UMAP for the active resegmented cells."""
    _gen = jobs._dataset_gen

    def _set(status, message, result=None):
        with core._umap_reseg_lock:
            if jobs._stale_job(_gen, status, message):
                return
            core._umap_reseg_state["status"]  = status
            core._umap_reseg_state["message"] = message
            if result is not None:
                core._umap_reseg_state["result"] = result

    try:
        # Get the reseg result for the segmentation source selected in the UI
        with core._umap_reseg_lock:
            _tool = core._seg_tool(core._umap_reseg_state.get("seg_source", "xenium"))
        pres = bres = None
        if _tool == "proseg":
            with core._proseg_lock:
                pres = core._proseg_state["result"] if core._proseg_state["status"] == "done" else None
        elif _tool == "baysor":
            with core._baysor_lock:
                bres = core._baysor_state["result"] if core._baysor_state["status"] == "done" else None
        alt_res = pres or bres or (core._seg_result(_tool, done_only=True) if _tool == "imgseg" else None)
        if _tool != "xenium" and alt_res is None:
            _set("error", "Selected segmentation is not loaded")
            return

        if alt_res is None:
            # Xenium mode: use SPLIT-corrected expr if counts_mode == "corrected",
            # otherwise use raw expression matrix.
            with core._umap_reseg_lock:
                _counts_mode = core._umap_reseg_state.get("counts_mode", "original")
            _corr_expr = core.DATA.get("split_corrected_expr") if _counts_mode == "corrected" else None
            _corr_ids  = core.DATA.get("split_corrected_cell_ids")  # IDs of cells SPLIT kept
            _xen_df    = core.DATA["df"]

            # If corrected expr was loaded from zarr at startup it is full-size
            # (all cells, zeros for non-SPLIT cells) but corr_ids is not populated.
            # Derive corr_ids from cluster_split_10 and subset the matrix.
            # If corrected expr is full-size (loaded from zarr at startup), corr_ids
            # won't be set. Identify SPLIT-kept cells by non-zero rows (zeroed for
            # non-corrected cells during zarr write) and subset the matrix.
            # DATA["df"] row -> DATA["expr"] row (identity unless subset() is active)
            _d2e = np.asarray(core.DATA["df_to_expr"])
            if _corr_expr is not None and _corr_ids is None and \
                    _corr_expr.shape[0] == core.DATA["expr"].shape[0] and len(_d2e) == len(_xen_df):
                _row_sums = np.asarray(_corr_expr.sum(axis=1)).flatten()
                _kept_idx = np.where((_d2e >= 0) & (_row_sums[np.clip(_d2e, 0, None)] > 0))[0]
                if len(_kept_idx) > 0:
                    _corr_ids = [str(i) for i in _xen_df.index[_kept_idx]]
                    if "_df_original" not in core.DATA:
                        core.DATA["split_corrected_cell_ids"] = _corr_ids
                    print(f"  UMAP: derived {len(_corr_ids):,} SPLIT cell IDs from non-zero rows",
                          flush=True)

            # Zarr fallback: if corrected expr not in memory (e.g. after app restart),
            # try loading X_corrected from the SpatialData zarr using low-level zarr API
            # (obs columns written via create_dataset are not exposed by spatialdata .obs).
            if (_counts_mode == "corrected") and (_corr_expr is None) and core.DATA.get("sdata_path") and \
                    os.path.isdir(core.DATA["sdata_path"]):
                try:
                    import zarr as _zarr_fb
                    _zfb    = _zarr_fb.open_group(core.DATA["sdata_path"], mode="r",
                                                  use_consolidated=False)
                    _tbl_fb = _zfb["tables"]["table"]
                    _obs_fb = _tbl_fb["obs"]
                    _lay_fb = _tbl_fb.get("layers", {})
                    if "X_corrected" in _lay_fb:
                        # Read sparse X_corrected
                        _xg = _lay_fb["X_corrected"]
                        _xcorr_full = sp.csr_matrix(
                            (_xg["data"][:], _xg["indices"][:], _xg["indptr"][:]),
                            shape=tuple(_xg.attrs["shape"]),
                        )
                        # Read cell IDs
                        _idx_key = _obs_fb.attrs.get("_index", "_index")
                        _all_ids = [str(v) for v in _obs_fb[_idx_key][:]]
                        # Identify SPLIT-kept cells via non-empty cluster_split_10
                        if "cluster_split_10" in _obs_fb:
                            _clust   = np.array([str(v) for v in _obs_fb["cluster_split_10"][:]])
                            _kept_mask = np.array([bool(v.strip()) for v in _clust], dtype=bool)
                        else:
                            _kept_mask = np.ones(len(_all_ids), dtype=bool)
                        _kept_rows = np.where(_kept_mask)[0]
                        _corr_expr = _xcorr_full[_kept_rows, :]
                        _corr_ids  = [_all_ids[i] for i in _kept_rows]
                        # Cache in DATA so subsequent calls don't re-read zarr.
                        # Keep the full-size matrix there: _get_expr_values needs
                        # one row per DATA["expr"] row.
                        jobs._check_dataset_gen(_gen)
                        core.DATA["split_corrected_expr"]     = _xcorr_full
                        core.DATA["split_corrected_cell_ids"] = _corr_ids
                        print(f"  Xenium UMAP: loaded X_corrected from zarr "
                              f"({len(_corr_ids):,} SPLIT-corrected cells)", flush=True)
                except Exception as _fb_err:
                    print(f"  Xenium UMAP: zarr X_corrected fallback failed: {_fb_err}",
                          flush=True)

            if _corr_expr is not None and _corr_ids is not None:
                _df_ids    = [str(i) for i in _xen_df.index]
                if _corr_expr.shape[0] == core.DATA["expr"].shape[0] and len(_d2e) == len(_df_ids):
                    # Full-size matrix, one row per DATA["expr"] row (zero rows for
                    # cells SPLIT dropped): select the kept cells through df_to_expr.
                    _corr_set = {str(cid) for cid in _corr_ids}
                    _sub_mask = [(c in _corr_set) and (r >= 0)
                                 for c, r in zip(_df_ids, _d2e)]  # bool mask over DATA["df"]
                    expr = _corr_expr[_d2e[np.where(_sub_mask)[0]], :]
                else:
                    # Rows in corrected_cell_ids order
                    _id_to_row = {str(cid): i for i, cid in enumerate(_corr_ids)}
                    _sub_rows  = [_id_to_row[i] for i in _df_ids if i in _id_to_row]
                    _sub_mask  = [i in _id_to_row for i in _df_ids]  # bool mask over DATA["df"]
                    expr = _corr_expr[_sub_rows, :]
                _n   = expr.shape[0]
                _set("running", f"Normalising {_n:,} SPLIT-corrected cells…")
                print(f"  Xenium UMAP: using SPLIT-corrected counts for {_n:,}/{len(_xen_df):,} cells",
                      flush=True)
            else:
                expr = core.DATA.get("expr")
                if expr is None:
                    _set("error", "No expression matrix loaded")
                    return
                _sub_mask = None
                # Embed only the cells of DATA["df"] (panel genes, no SpaGE-imputed columns)
                _good = _d2e >= 0
                if len(_d2e) != expr.shape[0] or (_d2e != np.arange(len(_d2e))).any():
                    expr = expr[_d2e[_good], :]
                _pg_idx = [core.DATA["gene_name_to_idx"][g] for g in dataview._panel_gene_names()
                           if g in core.DATA["gene_name_to_idx"]]
                if _pg_idx and _pg_idx != list(range(expr.shape[1])):
                    expr = expr[:, _pg_idx]
                _n        = expr.shape[0]
                _set("running", f"Normalising {_n:,} Xenium cells…")
                print(f"  Xenium UMAP: using original counts for {_n:,} cells", flush=True)

            _mat = expr.astype("float32").tocsr()
            _rs  = np.asarray(_mat.sum(axis=1)).flatten()
            _rs[_rs == 0] = 1.0
            _mat = sp.diags(1e4 / _rs).dot(_mat).tocsr()
            _mat.data = np.log1p(_mat.data)
            import warnings as _w2
            from sklearn.decomposition import TruncatedSVD as _TSVD2
            _nc = min(50, _n - 1, _mat.shape[1] - 1)
            _set("running", f"PCA ({_nc} components)…")
            _svd2 = _TSVD2(n_components=_nc, random_state=0)
            with _w2.catch_warnings():
                _w2.filterwarnings("ignore", category=RuntimeWarning,
                                   message="invalid value encountered in divide")
                _pca2 = _svd2.fit_transform(_mat)
            _set("running", "Running UMAP…")
            try:
                import sys as _sys2, types as _types2
                if "tensorflow" not in _sys2.modules:
                    _sys2.modules["tensorflow"] = _types2.ModuleType("tensorflow")
                from umap.umap_ import UMAP as _UMAP2
                with _w2.catch_warnings():
                    _w2.filterwarnings("ignore", message="n_jobs value.*overridden")
                    _emb2 = _UMAP2(n_neighbors=min(15, _n - 1), min_dist=0.1,
                                   **_umap_seed_kwargs(_n, 0)).fit_transform(_pca2)
            except Exception:
                if _n > _UMAP_PARALLEL_MIN_CELLS:
                    raise   # t-SNE on this many cells would not finish; report the UMAP error
                from sklearn.manifold import TSNE as _TSNE2
                _set("running", "Running t-SNE (umap-learn unavailable)…")
                _emb2 = _TSNE2(n_components=2, perplexity=min(30, max(5, _n // 10)),
                               random_state=0).fit_transform(_pca2)

            jobs._check_dataset_gen(_gen)
            jobs._job_shield()   # last cancellation point: DATA["df"] is updated below
            if _sub_mask is not None:
                # Write NaN for cells SPLIT dropped, UMAP coords for cells it kept.
                # Use split_umap_1/2 so make_umap_fig can distinguish corrected vs original UMAP.
                _u1 = np.full(len(_xen_df), np.nan)
                _u2 = np.full(len(_xen_df), np.nan)
                _sub_idx = np.where(_sub_mask)[0]
                _u1[_sub_idx] = _emb2[:, 0]
                _u2[_sub_idx] = _emb2[:, 1]
                core.DATA["df"]["split_umap_1"] = _u1
                core.DATA["df"]["split_umap_2"] = _u2
            else:
                _u1 = np.full(len(_xen_df), np.nan)
                _u2 = np.full(len(_xen_df), np.nan)
                _u1[_good] = _emb2[:, 0]
                _u2[_good] = _emb2[:, 1]
                core.DATA["df"]["umap_1"] = _u1
                core.DATA["df"]["umap_2"] = _u2
            dataview._umap_df_cache.clear()
            with core._umap_reseg_lock:
                core._umap_reseg_state["_xenium_bumped"] = False
            _label = "SPLIT-corrected" if _sub_mask is not None else "Xenium"
            _set("done", f"{_label} UMAP — {_n:,} cells")
            print(f"  {_label} UMAP recomputed for {_n:,} cells", flush=True)
            return

        if alt_res.get("expr") is None:
            # Try loading expr from the reseg SpatialData zarr
            zarr_path = alt_res.get("sdata_path", "")
            if zarr_path and os.path.isdir(zarr_path):
                try:
                    import spatialdata as _sd_umap
                    _sdata_umap = _sd_umap.read_zarr(zarr_path)
                    alt_res["expr"] = sp.csr_matrix(_sdata_umap.tables["table"].X)
                    print(f"  Reseg UMAP: loaded expr from {os.path.basename(zarr_path)}", flush=True)
                except Exception as _ze:
                    _set("error", f"No expression matrix — re-run segmentation ({_ze})")
                    return
            else:
                _set("error", "No expression matrix — re-run segmentation first")
                return

        with core._umap_reseg_lock:
            _counts_mode = core._umap_reseg_state.get("counts_mode", "original")
        _corr = alt_res.get("split_corrected_expr") if _counts_mode == "corrected" else None
        _src_label = "SPLIT-corrected" if _corr is not None else "reseg"
        _all_cell_ids = [str(c) for c in alt_res["cells_df"].index]
        # Always subset to the SPLIT-corrected cells only (zarr-aligned matrix pads uncorrected
        # cells with zeros which would distort the UMAP embedding).
        if _corr is not None:
            _corr_ids = [str(c) for c in alt_res.get("split_corrected_cell_ids", [])]
            if not _corr_ids and _corr.shape[0] == len(_all_cell_ids):
                # Loaded from zarr (no corrected_cell_ids in memory) — derive from non-zero rows
                _nonzero_mask = np.diff(_corr.indptr) > 0
                _corr_ids = [_all_cell_ids[i] for i in np.where(_nonzero_mask)[0]]
                print(f"  Reseg UMAP: derived {len(_corr_ids):,} corrected cell IDs "
                      f"from non-zero X_corrected rows", flush=True)
            if _corr_ids:
                _id_to_row = {c: i for i, c in enumerate(_corr_ids)}
                if _corr.shape[0] == len(_all_cell_ids):
                    # zarr-aligned (full-size) matrix — row order matches _all_cell_ids
                    _sub_idx = [i for i, c in enumerate(_all_cell_ids) if c in _id_to_row]
                    _sub_cell_ids = [_all_cell_ids[i] for i in _sub_idx]
                    expr = _corr[_sub_idx, :]
                else:
                    # Raw SPLIT output — rows are in corrected_cell_ids order
                    expr = _corr
                    _sub_cell_ids = _corr_ids
                cell_ids = _sub_cell_ids
                print(f"  Reseg UMAP: SPLIT-corrected subset — "
                      f"{len(cell_ids):,}/{len(_all_cell_ids):,} cells", flush=True)
            else:
                # No corrected cells found at all — fall back to uncorrected
                print(f"  Reseg UMAP: no corrected cells found — using uncorrected", flush=True)
                expr = _reseg_measured_expr(alt_res["expr"])
                _src_label = "reseg"
                cell_ids = _all_cell_ids
        else:
            # measured panel genes only: after SpaGE the run's matrix also holds
            # dense imputed columns, which are not counts
            expr = _reseg_measured_expr(alt_res["expr"])
            cell_ids = _all_cell_ids
        n_cells = expr.shape[0]
        _set("running", f"Normalising {n_cells:,} {_src_label} cells…")
        print(f"  Reseg UMAP: using {_src_label} counts for {n_cells:,} cells", flush=True)

        # CP10K + log1p normalise
        mat = expr.astype("float32").tocsr()
        row_sums = np.asarray(mat.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        mat = sp.diags(1e4 / row_sums).dot(mat).tocsr()
        mat.data = np.log1p(mat.data)

        # PCA (TruncatedSVD) — reduce to 50 dims before UMAP
        import warnings as _warnings
        from sklearn.decomposition import TruncatedSVD
        n_comp = min(50, n_cells - 1, mat.shape[1] - 1)
        _set("running", f"PCA ({n_comp} components)…")
        svd = TruncatedSVD(n_components=n_comp, random_state=0)
        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", category=RuntimeWarning,
                                     message="invalid value encountered in divide")
            pca_coords = svd.fit_transform(mat)   # (n_cells × n_comp)

        # UMAP — prefer umap-learn, fall back to t-SNE
        _set("running", "Running UMAP…")
        try:
            import sys, types
            # Stub tensorflow so umap/__init__.py can load parametric_umap without
            # crashing on the NumPy 1.x vs 2.x ABI mismatch in the installed TF.
            if "tensorflow" not in sys.modules:
                sys.modules["tensorflow"] = types.ModuleType("tensorflow")
            import warnings
            from umap.umap_ import UMAP
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="n_jobs value.*overridden")
                reducer = UMAP(n_neighbors=min(15, n_cells - 1),
                               min_dist=0.1, **_umap_seed_kwargs(n_cells, 0))
                embedding = reducer.fit_transform(pca_coords)
        except Exception:
            if n_cells > _UMAP_PARALLEL_MIN_CELLS:
                raise   # t-SNE on this many cells would not finish; report the UMAP error
            from sklearn.manifold import TSNE
            _set("running", "Running t-SNE (umap-learn not installed or TF error)…")
            perp = min(30, max(5, n_cells // 10))
            embedding = TSNE(n_components=2, perplexity=perp,
                             random_state=0).fit_transform(pca_coords)

        umap_df = pd.DataFrame(
            {"umap_1": embedding[:, 0], "umap_2": embedding[:, 1]},
            index=cell_ids,
        )
        jobs._job_shield()   # last cancellation point: publish, then cache to the run's zarr
        with core._umap_reseg_lock:
            jobs._check_dataset_gen(_gen)
            core._umap_reseg_state["owner"] = alt_res.get("out_dir")
        _set("done", f"Done — {n_cells:,} cells", result=umap_df)
        print(f"  Reseg UMAP: computed for {n_cells:,} cells", flush=True)

        # ── Write UMAP coords back to cells_df + zarr for caching ────
        _set("running", "Caching UMAP to zarr…")
        _lock = core._proseg_lock if pres else core._baysor_lock if bres else core._seg_state(_tool)[1]
        _sdata_path = alt_res.get("sdata_path", "")
        # Build full-length arrays (NaN for cells not in this UMAP's index)
        _u1 = np.full(len(_all_cell_ids), np.nan)
        _u2 = np.full(len(_all_cell_ids), np.nan)
        _id_to_pos = {c: i for i, c in enumerate(_all_cell_ids)}
        for _cid, _e1, _e2 in zip(cell_ids, embedding[:, 0], embedding[:, 1]):
            _pos = _id_to_pos.get(_cid)
            if _pos is not None:
                _u1[_pos] = _e1
                _u2[_pos] = _e2
        # An embedding of SPLIT-corrected counts goes to split_umap_1/2 (as in Xenium
        # mode); umap_1/2 keep the original-counts embedding.
        _c1, _c2 = (("split_umap_1", "split_umap_2") if _src_label == "SPLIT-corrected"
                    else ("umap_1", "umap_2"))
        with _lock:
            alt_res["cells_df"][_c1] = _u1
            alt_res["cells_df"][_c2] = _u2
        if _sdata_path and os.path.isdir(_sdata_path):
            # Existing umap_1/2 columns are rewritten too, except under subset():
            # the zarr holds every cell and the others would be set to NaN.
            reseg._update_reseg_zarr_obs(
                _sdata_path, alt_res["cells_df"],
                force_cols=() if "_cells_df_original" in alt_res else (_c1, _c2))
            print(f"  Reseg UMAP: cached to zarr", flush=True)
        _set("done", f"Done — {n_cells:,} cells", result=umap_df)

    except jobs._JobCancelled:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set("error", str(exc)[:200])
