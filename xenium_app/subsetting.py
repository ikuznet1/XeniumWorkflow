"""REPL helpers: subset() / unsubset(), retry_split_zarr_write(), get_genes()."""

import os
import numpy as np
import pandas as pd

from . import core, dataview, annotation, reseg, split


# ─── Subset / unsubset ────────────────────────────────────────────────────────

def _loaded_reseg_results() -> list:
    """[(tool, result dict)] for every Baysor/Proseg run that is loaded right now."""
    out = []
    with core._proseg_lock:
        if core._proseg_state.get("result") is not None:
            out.append(("proseg", core._proseg_state["result"]))
    with core._baysor_lock:
        if core._baysor_state.get("result") is not None:
            out.append(("baysor", core._baysor_state["result"]))
    with core._imgseg_lock:
        if core._imgseg_state.get("result") is not None:
            out.append(("imgseg", core._imgseg_state["result"]))
    return out


def subset(cluster=None, cell_type=None,
           gene=None, min_expr=None, max_expr=None,
           min_transcripts=None, max_transcripts=None,
           min_cell_area=None,    max_cell_area=None,
           min_nucleus_area=None, max_nucleus_area=None,
           cell_ids=None, method=None):
    """
    Subset the displayed cells. All non-None filters are ANDed together.

    Parameters
    ----------
    cluster : int | str | list
        Cluster label(s) to keep (integer IDs from the selected clustering method).
    method : str
        Clustering method name (default: first available).
        Use list(DATA['cluster_methods']) to see options.
    cell_type : str | list
        Cell type label(s) to keep (requires annotation to have been run).
    gene : str
        Gene name for expression-based filtering (use with min_expr / max_expr).
    min_expr, max_expr : float
        log1p expression thresholds for the selected gene.
    min_transcripts, max_transcripts : float
        Transcript count bounds.
    min_cell_area, max_cell_area : float
        Cell area bounds (µm²).
    min_nucleus_area, max_nucleus_area : float
        Nucleus area bounds (µm²).
    cell_ids : list
        Explicit list of cell IDs (barcodes) to keep.

    Examples
    --------
    subset(cluster=3)
    subset(cluster=[1, 2, 3], method='gene_expression_10_clusters')
    subset(cell_type='Cardiomyocyte')
    subset(gene='MYH7', min_expr=1.0)
    subset(min_transcripts=50, max_cell_area=500)
    """

    # The originals are stored in DATA only once the mask has been built (below): a
    # filter that raises must not leave the app flagged as "SUBSET ACTIVE".
    df            = core.DATA.get("_df_original", core.DATA["df"])
    _df_to_expr_o = core.DATA.get("_df_to_expr_orig", core.DATA["df_to_expr"])
    mask = pd.Series(True, index=df.index)
    # Set once a filter has really been applied (to the Xenium cells or to a loaded
    # Baysor/Proseg run). Filters that are skipped - no annotation, unknown gene,
    # missing column - select nothing: a call made of those alone changes nothing
    # and must not flag an "N of N" subset as active.
    _applied = False

    # ── Cluster filter ────────────────────────────────────────────────────
    if cluster is not None:
        _cm = core.DATA.get("cluster_methods") or []
        _method = method or (_cm[0] if _cm else None)
        if _method is None:
            raise ValueError("No cluster sets are available for this dataset.")
        col  = f"clust__{_method}"
        if col not in df.columns:
            raise ValueError(f"Unknown clustering method {_method!r}; available: {list(_cm)}")
        vals = [cluster] if not isinstance(cluster, (list, tuple)) else list(cluster)
        vals = [int(v) for v in vals]
        mask &= df[col].isin(vals)
        _applied = True

    # ── Cell type filter ──────────────────────────────────────────────────
    if cell_type is not None:
        _reseg_active = (core._proseg_state["status"] == "done" or core._baysor_state["status"] == "done"
                         or core._imgseg_state["status"] == "done")
        with core._annot_lock:
            # Only use Xenium-specific labels (exact key "labels_{method}") — not reseg-suffixed ones.
            # Reseg annotation labels (labels_{method}_proseg_xxx) have reseg cell IDs, not barcodes.
            if _reseg_active:
                labels = next((v for m in core._ANNOT_METHODS
                               for k, v in core._annot_state.items()
                               if k == f"labels_{m}" and v is not None), None)
            else:
                labels = next((v for m in core._ANNOT_METHODS
                               for k, v in core._annot_state.items()
                               if (k == f"labels_{m}" or k.startswith(f"labels_{m}_"))
                               and v is not None), None)
        if labels is None:
            if _reseg_active:
                print("  subset: no Xenium cell type annotation — skipping Xenium cell_type filter", flush=True)
            else:
                print("  subset: no cell type annotation loaded — skipping cell_type filter", flush=True)
        else:
            vals = [cell_type] if isinstance(cell_type, str) else list(cell_type)
            mask &= labels.reindex(df.index.astype(str)).isin(vals)
            _applied = True

    # ── Gene expression filter ────────────────────────────────────────────
    if gene is not None and (min_expr is not None or max_expr is not None):
        # Compute against the ORIGINAL cell→expr mapping: get_gene_expression() is
        # aligned to the current (possibly already subset) DATA["df"].
        _base = gene[:-6] if gene.endswith(" [imp]") else gene
        if _base in core.DATA["gene_name_to_idx"]:
            _raw = core.DATA["expr_csc"][:, core.DATA["gene_name_to_idx"][_base]].toarray().ravel()
            _idx = _df_to_expr_o
            expr_vals = pd.Series(
                np.log1p(np.where(_idx >= 0, _raw[np.clip(_idx, 0, len(_raw) - 1)], 0.0)),
                index=df.index)
            if min_expr is not None: mask &= expr_vals >= min_expr
            if max_expr is not None: mask &= expr_vals <= max_expr
            _applied = True
        else:
            print(f"  subset: gene '{gene}' not found - skipping gene filter", flush=True)

    # ── Numeric range filters ─────────────────────────────────────────────
    for _ncol, _lo, _hi in (("transcript_counts", min_transcripts, max_transcripts),
                            ("cell_area", min_cell_area, max_cell_area),
                            ("nucleus_area", min_nucleus_area, max_nucleus_area)):
        if _lo is None and _hi is None:
            continue
        if _ncol not in df.columns:
            print(f"  subset: this dataset has no '{_ncol}' column - skipping that filter", flush=True)
            continue
        if _lo is not None: mask &= df[_ncol] >= _lo
        if _hi is not None: mask &= df[_ncol] <= _hi
        _applied = True

    # ── Explicit cell IDs ─────────────────────────────────────────────────
    if cell_ids is not None:
        mask &= df.index.isin(cell_ids)
        _applied = True

    df_sub = df[mask]

    # ── Reseg path (Baysor / Proseg): build the masks ─────────────────────
    # Every loaded run is filtered, not only one of them: both tools can hold a
    # result at once and the Segmentation dropdown decides which one is displayed.
    # Nothing is changed in this loop; the masks are applied further down.
    _reseg_plans = []
    for _tool, _alt_res in _loaded_reseg_results():

        _b_first = "_cells_df_original" not in _alt_res      # no subset active yet
        bdf        = _alt_res["cells_df"] if _b_first else _alt_res["_cells_df_original"]
        _bexpr_all = _alt_res["expr"]     if _b_first else _alt_res["_expr_original"]
        bmask = pd.Series(True, index=bdf.index)
        _b_applied = False

        # Cluster filter — reseg cluster column names start with "cluster"
        if cluster is not None:
            # the column "Color by Cluster" shows (cluster_kmeans_*), never a bare `cluster`
            _bcol = reseg._reseg_display_cluster_col(bdf.columns)
            if _bcol is not None:
                _bvals_str = [str(cluster)] if not isinstance(cluster, (list, tuple)) else [str(v) for v in cluster]
                bmask &= bdf[_bcol].astype(str).isin(_bvals_str)
                _b_applied = True
            else:
                print(f"  subset reseg: no cluster column in cells_df — skipping cluster filter", flush=True)

        # Cell type filter
        if cell_type is not None:
            with core._annot_lock:
                _blabels = None
                for _m in core._ANNOT_METHODS:
                    _blabels = core._annot_state.get(annotation._labels_key_for_method(_m, _alt_res))
                    if _blabels is not None:
                        break
            if _blabels is None:
                print(f"  subset reseg: no cell type annotation for {_tool} — skipping cell_type filter", flush=True)
            else:
                _bct_vals = [cell_type] if isinstance(cell_type, str) else list(cell_type)
                bmask &= _blabels.reindex(bdf.index.astype(str)).isin(_bct_vals)
                _b_applied = True

        # Gene expression filter — reseg uses its own expr matrix
        if gene is not None and (min_expr is not None or max_expr is not None):
            if gene in core.DATA["gene_names"]:
                _gidx    = core.DATA["gene_names"].index(gene)
                _braw    = _bexpr_all.getcol(_gidx).toarray().ravel()
                _bgexpr  = pd.Series(np.log1p(_braw), index=bdf.index)
                if min_expr is not None: bmask &= _bgexpr >= min_expr
                if max_expr is not None: bmask &= _bgexpr <= max_expr
                _b_applied = True
            else:
                print(f"  subset reseg: gene '{gene}' not in reseg expr — skipping gene filter", flush=True)

        # QC filters
        for _bcol_q, _blo, _bhi in (("transcript_counts", min_transcripts, max_transcripts),
                                    ("cell_area", min_cell_area, max_cell_area),
                                    ("nucleus_area", min_nucleus_area, max_nucleus_area)):
            if (_blo is None and _bhi is None) or _bcol_q not in bdf.columns:
                continue
            if _blo is not None: bmask &= bdf[_bcol_q] >= _blo
            if _bhi is not None: bmask &= bdf[_bcol_q] <= _bhi
            _b_applied = True

        # Explicit cell IDs
        if cell_ids is not None:
            bmask &= bdf.index.isin(cell_ids)
            _b_applied = True

        _reseg_plans.append((_tool, _alt_res, bdf, bmask))
        _applied = _applied or _b_applied

    if not _applied:
        print("  subset: no filter was applied (none given, or all were skipped) — "
              "the displayed cells are unchanged", flush=True)
        return core.DATA["df"]

    # Masks built without error: save the originals (first subset call) and switch
    if "_df_original" not in core.DATA:
        core.DATA["_df_original"]      = df
        core.DATA["_df_to_expr_orig"]  = _df_to_expr_o

    # Update active data
    core.DATA["df"] = df_sub
    # what was asked for, for the provenance record of the analyses run on the subset
    # (a second subset() narrows the first: one entry per call)
    _asked = dict(cluster=cluster, cell_type=cell_type, gene=gene, min_expr=min_expr, max_expr=max_expr,
                  min_transcripts=min_transcripts, max_transcripts=max_transcripts,
                  min_cell_area=min_cell_area, max_cell_area=max_cell_area,
                  min_nucleus_area=min_nucleus_area, max_nucleus_area=max_nucleus_area, method=method,
                  n_cell_ids=(len(cell_ids) if cell_ids is not None else None))
    core.DATA.setdefault("_subset_desc", []).append(
        {k: (v if isinstance(v, (int, float, str, bool)) else [str(x) for x in list(v)[:50]] if isinstance(v, (list, tuple, set)) else str(v))
         for k, v in _asked.items() if v is not None})

    # Rebuild df_to_expr for the subset (map subset row → expr matrix row)
    orig_to_expr = core.DATA["_df_to_expr_orig"]
    orig_pos     = {cid: i for i, cid in enumerate(df.index)}
    core.DATA["df_to_expr"] = np.array(
        [orig_to_expr[orig_pos[cid]] for cid in df_sub.index], dtype=np.int64
    )

    # ── Reseg path: apply the masks ───────────────────────────────────────
    for _tool, _alt_res, bdf, bmask in _reseg_plans:
        # Save originals on first subset
        if "_cells_df_original" not in _alt_res:
            _alt_res["_cells_df_original"] = _alt_res["cells_df"]
            _alt_res["_expr_original"]      = _alt_res["expr"]
            _alt_res["_corrected_original"] = _alt_res.get("split_corrected_expr")

        _kept_pos = np.where(bmask.values)[0]
        _bdf_sub  = bdf.iloc[_kept_pos]
        _alt_res["cells_df"] = _bdf_sub
        _alt_res["expr"]     = _alt_res["_expr_original"][_kept_pos, :]
        if _alt_res["_corrected_original"] is not None:
            _alt_res["split_corrected_expr"] = _alt_res["_corrected_original"][_kept_pos, :]
        print(f"  subset reseg ({_tool}): {len(_bdf_sub):,} / {len(bdf):,} cells selected", flush=True)

    dataview._umap_df_cache.clear()
    dataview._gene_expr_cache.clear()
    dataview._viewport_arrays.clear()
    core._subset_version += 1
    print(f"  subset: {len(df_sub):,} / {len(df):,} cells selected", flush=True)
    return df_sub


def unsubset():
    """Restore the full dataset after subsetting."""

    if "_df_original" not in core.DATA:
        print("  unsubset: no subset is active", flush=True)
        return

    core.DATA["df"]         = core.DATA.pop("_df_original")
    core.DATA.pop("_subset_desc", None)
    core.DATA["df_to_expr"] = core.DATA.pop("_df_to_expr_orig")

    # Restore reseg originals if subset was applied there too
    for _tool, _alt_res in _loaded_reseg_results():
        if "_cells_df_original" not in _alt_res:
            continue
        _alt_res["cells_df"] = _alt_res.pop("_cells_df_original")
        _alt_res["expr"]     = _alt_res.pop("_expr_original")
        _orig_corr = _alt_res.pop("_corrected_original")
        if _orig_corr is not None:
            _alt_res["split_corrected_expr"] = _orig_corr
        elif (_alt_res.get("split_corrected_expr") is not None
              and _alt_res["split_corrected_expr"].shape[0] != len(_alt_res["cells_df"])):
            # Corrected matrix sized to the subset cannot be paired with the full cells_df
            _alt_res["split_corrected_expr"] = None
            print(f"  unsubset: dropped {_tool} SPLIT-corrected counts computed on the subset "
                  "(re-run SPLIT on the full dataset)", flush=True)

    dataview._umap_df_cache.clear()
    dataview._gene_expr_cache.clear()
    dataview._viewport_arrays.clear()
    core._subset_version += 1
    print(f"  unsubset: restored {len(core.DATA['df']):,} cells", flush=True)


def retry_split_zarr_write(seg_source: str = None) -> None:
    """Write in-memory SPLIT corrected counts to the SpatialData zarr.

    Use this if SPLIT completed successfully in the current session but the zarr
    write failed (e.g. due to consolidated metadata error). Call from the REPL:

        retry_split_zarr_write()               # Xenium
        retry_split_zarr_write("baysor:xxxx")  # Baysor run
        retry_split_zarr_write("proseg:xxxx")  # Proseg run
    """
    tool = core._seg_tool(seg_source) if seg_source else "xenium"

    if tool == "baysor":
        with core._baysor_lock:
            res = core._baysor_state.get("result")
    elif tool == "proseg":
        with core._proseg_lock:
            res = core._proseg_state.get("result")
    else:
        res = core._seg_result(seg_source)  # None: Xenium uses DATA directly

    # The in-memory matrix is already expanded to one row per cell (reseg: per
    # cells_df row; Xenium: per DATA["expr"] row), while split_corrected_cell_ids
    # lists only the cells RCTD kept - so rows are paired with cell ids here.
    if res is not None:
        corrected_mat     = res.get("split_corrected_expr")
        panel_genes       = res.get("split_panel_genes")
        sdata_path        = res.get("sdata_path", "")
        cells_df          = res.get("cells_df")
        if res.get("_cells_df_original") is not None and res.get("_corrected_original") is not None:
            # subset() active: write the full-size copy
            corrected_mat = res["_corrected_original"]
            cells_df      = res["_cells_df_original"]
    else:
        corrected_mat     = core.DATA.get("split_corrected_expr")
        panel_genes       = core.DATA.get("split_panel_genes")
        sdata_path        = core.DATA.get("sdata_path", "")
        cells_df          = core.DATA.get("_df_original", core.DATA.get("df"))
        if corrected_mat is not None and cells_df is not None:
            _d2e  = np.asarray(core.DATA.get("_df_to_expr_orig", core.DATA["df_to_expr"]))
            _good = (_d2e >= 0) & (_d2e < corrected_mat.shape[0])
            cells_df      = cells_df[_good]
            corrected_mat = corrected_mat[_d2e[_good], :]

    if corrected_mat is not None and (cells_df is None
                                      or corrected_mat.shape[0] != len(cells_df)):
        print("  retry_split_zarr_write: corrected matrix does not match the loaded cells "
              "— rerun SPLIT", flush=True)
        return
    if corrected_mat is None:
        print("  retry_split_zarr_write: no split_corrected_expr in memory — "
              "has SPLIT been run this session?", flush=True)
        return
    if not sdata_path or not os.path.isdir(sdata_path):
        print(f"  retry_split_zarr_write: sdata_path not found: {sdata_path!r}", flush=True)
        return
    if not panel_genes:
        print("  retry_split_zarr_write: split_panel_genes not stored — "
              "please rerun SPLIT (old session data)", flush=True)
        return

    print(f"  retry_split_zarr_write: writing {corrected_mat.shape} matrix to {sdata_path}",
          flush=True)
    split._split_write_zarr(sdata_path, corrected_mat, cells_df, panel_genes,
                      corrected_cell_ids=None)


def get_genes(file: str = None) -> list:
    """
    Print and return all non-imputed Xenium panel genes.

    Parameters
    ----------
    file : str, optional
        If provided, save the gene list to this file path (one gene per line).

    Returns
    -------
    list of gene name strings.
    """
    genes = sorted(core.DATA["gene_names"])
    print(f"  {len(genes)} panel genes:", flush=True)
    for g in genes:
        print(f"    {g}", flush=True)
    if file:
        with open(file, "w") as fh:
            fh.write("\n".join(genes) + "\n")
        print(f"  Saved to {file}", flush=True)
    return genes
