#!/usr/bin/env python3
"""
impute.py — Programmatic SpaGE gene imputation on a SpatialData zarr

Usage:
  python impute.py <sdata_path> <rds_path> [options]

Positional:
  sdata_path            Path to input SpatialData zarr directory
  rds_path              Path to Seurat RDS reference file

Options:
  --genes GENE [GENE ...]   Genes to impute. If omitted, auto-selects top-200 HVGs
                            from the reference that are absent from the spatial panel.
  --genes_file FILE         Text file with one gene per line (alternative to --genes)
  --n_pv INT                Number of principal vectors for SpaGE (default: 30)
  --output PATH             Output SpatialData zarr path. Defaults to
                            <sdata_path>_imputed.zarr (written next to the input)
  --overwrite               Overwrite output if it already exists
  --spage_repo PATH         Path to SpaGE_repo directory. Defaults to
                            ../SpaGE_repo relative to this script.

Example:
  python impute.py /data/xenium.zarr /refs/seurat_ref.rds \\
      --genes TNNT2 MYH7 ACTN2 --n_pv 30 --output /data/xenium_imputed.zarr
"""

import argparse
import os
import shutil
import sys
import tempfile

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io as sio


# ── SpaGE core ────────────────────────────────────────────────────────────────

def _vectorized_spage(spatial_df: pd.DataFrame,
                      rna_df: pd.DataFrame,
                      n_pv: int,
                      genes_to_predict: list,
                      spage_repo: str) -> pd.DataFrame:
    """
    Vectorized SpaGE imputation.
    spatial_df : cells × shared_genes (log-normalized counts)
    rna_df     : ref_cells × (shared_genes + predict_genes)
    Returns    : DataFrame cells × genes_to_predict (raw imputed values)
    """
    if spage_repo not in sys.path:
        sys.path.insert(0, spage_repo)
    from SpaGE.principal_vectors import PVComputation
    import scipy.stats as st
    from sklearn.neighbors import NearestNeighbors

    shared = np.intersect1d(spatial_df.columns, rna_df.columns)
    if len(shared) == 0:
        raise ValueError("No shared genes between spatial and reference data.")

    # float32 z-scores (as in the app) to halve memory
    rna_scaled  = pd.DataFrame(st.zscore(rna_df[shared],    axis=0).astype(np.float32),
                                index=rna_df.index,    columns=shared)
    spat_scaled = pd.DataFrame(st.zscore(spatial_df[shared], axis=0).astype(np.float32),
                                index=spatial_df.index, columns=shared)
    # Genes with zero variance in either dataset z-score to NaN and break the PCA (as in the app)
    _ok = (np.isfinite(rna_scaled.values.sum(axis=0))
           & np.isfinite(spat_scaled.values.sum(axis=0)))
    if not _ok.all():
        print(f"  SpaGE: dropping {int((~_ok).sum())} shared genes with zero variance", flush=True)
        shared = shared[_ok]
        if len(shared) == 0:
            raise ValueError("No shared genes with non-zero variance in both datasets.")
        rna_scaled  = rna_scaled.loc[:, shared]
        spat_scaled = spat_scaled.loc[:, shared]

    n_pv = min(n_pv, len(shared))

    pv = PVComputation(n_factors=n_pv, n_pv=n_pv,
                       dim_reduction="pca", dim_reduction_target="pca")
    pv.fit(rna_scaled, spat_scaled)
    S     = pv.source_components_.T
    eff   = int(np.sum(np.diag(pv.cosine_similarity_matrix_) > 0.3))
    eff   = max(eff, 1)
    S     = S[:, :eff]
    print(f"  SpaGE: {eff} effective principal vectors", flush=True)

    rna_proj  = (rna_scaled.values  @ S).astype(np.float32)   # (n_ref,  eff)
    spat_proj = (spat_scaled.values @ S).astype(np.float32)   # (n_spat, eff)
    del rna_scaled, spat_scaled

    nbrs = NearestNeighbors(n_neighbors=50, algorithm="auto", metric="cosine")
    nbrs.fit(rna_proj)
    distances, indices = nbrs.kneighbors(spat_proj)   # (n_spat, 50)
    del rna_proj, spat_proj

    valid  = distances < 1
    d_v    = np.where(valid, distances, 0.0).astype(np.float32)
    del distances
    d_sum  = d_v.sum(axis=1, keepdims=True)
    d_sum[d_sum == 0] = 1.0
    w      = np.where(valid, 1.0 - d_v / d_sum, 0.0).astype(np.float32)
    n_v    = np.maximum(valid.sum(axis=1, keepdims=True) - 1, 1).astype(np.float32)
    w      = w / n_v
    del d_v, d_sum, valid, n_v

    # Impute in gene chunks: Y_nbrs is (n_spat, 50, chunk) instead of (n_spat, 50, n_genes)
    GENE_CHUNK = 10
    imp = np.empty((len(spatial_df), len(genes_to_predict)), dtype=np.float32)
    for i in range(0, len(genes_to_predict), GENE_CHUNK):
        chunk_genes = genes_to_predict[i:i + GENE_CHUNK]
        Y = rna_df[chunk_genes].values.astype(np.float32)        # (n_ref, chunk)
        imp[:, i:i + len(chunk_genes)] = np.einsum("ij,ijk->ik", w, Y[indices])
        del Y

    return pd.DataFrame(imp, index=spatial_df.index, columns=genes_to_predict)


# ── Reference loading ──────────────────────────────────────────────────────────

# R: the %d highest-variance genes among .spage_cand (rows of rna_mat_data); row sums
# are accumulated over blocks of cells so that no second copy of the matrix is made.
_HVG_R = """
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
  cand <- .spage_cand[.spage_cand %%in%% names(v)]
  v <- v[cand]
  cand[order(-v)[seq_len(min(%dL, length(cand)))]]
})
"""


def _load_reference(rds_path: str, genes_needed: list,
                    auto_hvg_exclude=None, n_hvg: int = 200) -> tuple[pd.DataFrame, list]:
    """
    Load a Seurat RDS file and extract a cells × genes_needed submatrix via rpy2.
    With auto_hvg_exclude (the panel genes), the n_hvg highest-variance reference
    genes outside that set are selected in R and loaded as well.
    Returns (rna_df, all gene names of the reference).
    """
    import rpy2.robjects as ro
    import rpy2.robjects.conversion as _rconv
    from rpy2.robjects.packages import importr

    _rconv.set_conversion(ro.default_converter)
    importr("SeuratObject")
    importr("Matrix")

    try:
        # The path is passed as a value, never pasted into R source (a quote or a
        # backslash in it would break the parse or run as code)
        ro.globalenv["seurat_ref"] = importr("base").readRDS(rds_path)
        ro.r('rna_ref   <- seurat_ref[["RNA"]]')
        # LayerData works for both Assay (v3/v4) and Assay5 (v5) objects
        ro.r('rna_mat_data <- SeuratObject::LayerData(rna_ref, layer="data")')
        ro.r('rna_genes <- rownames(rna_mat_data)')
        all_rna_genes = list(ro.r("rna_genes"))

        if auto_hvg_exclude is not None:
            # Auto mode: top-n_hvg genes by variance over ALL genes outside the panel
            # (not only the first 2000 in the reference's row order)
            _excl = set(auto_hvg_exclude)
            _cand = [g for g in all_rna_genes if g not in _excl]
            ro.globalenv[".spage_cand"] = ro.StrVector(_cand)
            _top = [str(g) for g in ro.r(_HVG_R % int(n_hvg))]
            print(f"  Auto-selected {len(_top)} highly variable genes out of "
                  f"{len(_cand):,} non-panel reference genes", flush=True)
            genes_needed = list(genes_needed) + _top

        # Filter to genes that exist in the reference
        genes_needed_r = [g for g in genes_needed if g in all_rna_genes]
        if not genes_needed_r:
            raise ValueError("None of the requested genes were found in the reference.")

        with tempfile.TemporaryDirectory() as tmpdir:
            ro.globalenv[".spage_genes"] = ro.StrVector(genes_needed_r)
            ro.globalenv[".spage_dir"]   = tmpdir
            ro.r("""
genes_sub  <- .spage_genes[.spage_genes %in% rna_genes]
mat_sub    <- rna_mat_data[genes_sub, , drop=FALSE]
Matrix::writeMM(mat_sub, file.path(.spage_dir, "mat.mtx"))
writeLines(rownames(mat_sub), file.path(.spage_dir, "genes.txt"))
writeLines(colnames(mat_sub), file.path(.spage_dir, "cells.txt"))
""")
            mat_sp = sio.mmread(os.path.join(tmpdir, "mat.mtx")).T.tocsr()   # cells × genes
            with open(os.path.join(tmpdir, "genes.txt")) as f:
                loaded_genes = [l.strip() for l in f]
            with open(os.path.join(tmpdir, "cells.txt")) as f:
                ref_cells = [l.strip() for l in f]
    finally:
        # also on errors: do not leave the multi-GB reference in the R heap
        ro.r('suppressWarnings(rm(list = intersect(c("seurat_ref", "rna_ref", "rna_mat_data", '
             '"mat_sub", "rna_genes", "genes_sub", ".spage_cand", ".spage_genes", ".spage_dir"), '
             'ls(all.names = TRUE, envir = globalenv())), envir = globalenv())); invisible(gc())')

    rna_df = pd.DataFrame(
        mat_sp.toarray().astype(np.float32),
        index=ref_cells,
        columns=loaded_genes,
    )
    return rna_df, all_rna_genes


# ── Main imputation function ───────────────────────────────────────────────────

def impute(sdata_path: str,
           rds_path: str,
           genes: list | None = None,
           genes_file: str | None = None,
           n_pv: int = 30,
           output: str | None = None,
           overwrite: bool = False,
           use_corrected: bool = False,
           spage_repo: str | None = None) -> str:
    """
    Impute genes into a SpatialData zarr using SpaGE.

    Parameters
    ----------
    sdata_path    : Path to input SpatialData zarr
    rds_path      : Path to Seurat RDS reference
    genes         : List of gene names to impute. None → auto top-200 HVGs
    genes_file    : Path to a text file with one gene per line (merged with genes list)
    n_pv          : Number of SpaGE principal vectors (default 30)
    output        : Output zarr path. Defaults to <sdata_path>_imputed.zarr
    overwrite     : Overwrite output if it exists
    use_corrected : Use the X_corrected layer (SPLIT output) as input counts instead of X.
                    Raises ValueError if X_corrected is not present in the zarr.
                    The imputed genes are then listed in
                    adata.uns["split_corrected_imputed_genes"] (instead of
                    uns["imputed_genes"]); the app labels them '[corr+imp]'.

    Imputed genes are appended to X under their plain gene names with
    var["is_imputed"] = True, exactly as the app writes them.
    spage_repo    : Path to SpaGE_repo. Defaults to ../SpaGE_repo relative to this script

    Returns
    -------
    Path to the output SpatialData zarr.
    """
    import spatialdata as sd

    # ── Merge gene sources ─────────────────────────────────────────────────────
    if genes_file is not None:
        with open(genes_file) as f:
            file_genes = [l.strip() for l in f if l.strip()]
        genes = list(genes or []) + file_genes
    if genes is not None:
        genes = list(dict.fromkeys(genes))  # deduplicate, preserve order

    # ── Resolve paths ──────────────────────────────────────────────────────────
    sdata_path = os.path.abspath(sdata_path)
    rds_path   = os.path.abspath(rds_path)

    if spage_repo is None:
        spage_repo = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "SpaGE_repo")
    spage_repo = os.path.abspath(spage_repo)
    if not os.path.isfile(os.path.join(spage_repo, "SpaGE", "principal_vectors.py")):
        raise FileNotFoundError(
            f"SpaGE not found at {spage_repo} (the directory is a git submodule and is empty "
            f"until fetched). Run `git submodule update --init SpaGE_repo` in the repository, "
            f"or pass --spage_repo / spage_repo= pointing at a clone of "
            f"https://github.com/tabdelaal/SpaGE.")

    if output is None:
        stem   = sdata_path.rstrip("/\\")
        output = stem + "_imputed.zarr"
    output = os.path.abspath(output)

    if os.path.exists(output):
        if overwrite:
            shutil.rmtree(output)
        else:
            raise FileExistsError(f"{output} already exists. Use overwrite=True.")

    # ── Load SpatialData ───────────────────────────────────────────────────────
    print(f"Loading SpatialData from {sdata_path} …", flush=True)
    sdata = sd.read_zarr(sdata_path)
    adata = sdata.tables["table"]

    # SPLIT writes these obs columns as bare arrays (not in column-order), so
    # sd.read_zarr does not return them — recover them so the output keeps them.
    try:
        import zarr as _zarr_mod
        _obs_grp = _zarr_mod.open_group(
            os.path.join(sdata_path, "tables", "table", "obs"), mode="r")
        for _ec in ["cluster_split_10", "split_umap_1", "split_umap_2"]:
            if _ec not in adata.obs.columns and _ec in _obs_grp:
                _ev = _obs_grp[_ec][:]
                if len(_ev) == len(adata):
                    adata.obs[_ec] = _ev
    except Exception:
        pass

    # Identify panel (non-imputed) genes
    if "is_imputed" in adata.var.columns:
        panel_genes = adata.var_names[~adata.var["is_imputed"]].tolist()
    else:
        panel_genes  = list(adata.var_names)
        # Mark all existing as not imputed if column missing
        adata.var["is_imputed"] = False

    panel_gene_set = set(panel_genes)
    panel_gni      = {g: i for i, g in enumerate(adata.var_names)}

    print(f"  {len(adata)} cells, {len(panel_genes)} panel genes", flush=True)

    # ── Validate / select expression matrix ───────────────────────────────────
    if use_corrected:
        if "X_corrected" not in adata.layers:
            raise ValueError(
                "use_corrected=True but 'X_corrected' layer not found in the SpatialData zarr. "
                "Run SPLIT ambient RNA correction first (via the app or functions/run_split.py)."
            )
        print("  Using X_corrected layer (SPLIT-corrected counts)", flush=True)

    # ── Load reference to get full gene list ───────────────────────────────────
    print("Loading reference to discover gene universe …", flush=True)
    import rpy2.robjects as ro
    import rpy2.robjects.conversion as _rconv
    from rpy2.robjects.packages import importr
    _rconv.set_conversion(ro.default_converter)
    importr("SeuratObject")
    # R identifiers cannot start with "_" — use dot-prefixed temporaries.
    # The path is passed as a value, not pasted into R source.
    try:
        ro.globalenv[".tmp_ref"] = importr("base").readRDS(rds_path)
        ro.r('.tmp_genes <- rownames(SeuratObject::LayerData(.tmp_ref[["RNA"]], layer="data"))')
        all_rna_genes = list(ro.r(".tmp_genes"))
    finally:
        ro.r('suppressWarnings(rm(list = intersect(c(".tmp_ref", ".tmp_genes"), '
             'ls(all.names = TRUE, envir = globalenv())), envir = globalenv())); invisible(gc())')

    # ── Determine shared genes and genes_to_predict ────────────────────────────
    shared_genes = sorted(panel_gene_set & set(all_rna_genes))
    if not shared_genes:
        raise ValueError("No shared genes between the spatial panel and the reference.")
    print(f"  {len(shared_genes)} shared genes with reference", flush=True)

    auto_hvg = genes is None
    if auto_hvg:
        # The 200 highest-variance genes over all non-panel reference genes are
        # chosen in R when the reference is loaded (see _load_reference)
        genes_needed = list(shared_genes)
        print("  Auto-HVG mode: ranking all non-panel reference genes by variance …", flush=True)
    else:
        genes_to_predict = [g for g in genes
                            if g in all_rna_genes and g not in panel_gene_set]
        missing = [g for g in genes if g not in all_rna_genes]
        already_panel = [g for g in genes if g in panel_gene_set]
        if missing:
            print(f"  Warning: {len(missing)} genes not in reference: {missing[:10]}", flush=True)
        if already_panel:
            print(f"  Warning: {len(already_panel)} genes already in panel (skipped): "
                  f"{already_panel[:10]}", flush=True)
        if not genes_to_predict:
            raise ValueError("No valid genes to impute (all missing from reference or already in panel).")
        genes_needed = shared_genes + genes_to_predict
        print(f"  Will impute: {genes_to_predict}", flush=True)

    # ── Load reference submatrix ───────────────────────────────────────────────
    print("Extracting reference expression submatrix …", flush=True)
    rna_df, _ = _load_reference(rds_path, genes_needed,
                                auto_hvg_exclude=panel_gene_set if auto_hvg else None)

    # Auto mode: the genes _load_reference selected (in order of decreasing variance)
    if auto_hvg:
        genes_to_predict = [g for g in rna_df.columns if g not in panel_gene_set]
    else:
        genes_to_predict = [g for g in genes_to_predict if g in rna_df.columns]

    # Skip genes already imputed into this table (a re-run on own output would
    # duplicate var names). Imputed genes are stored under their plain names, as
    # the app does; stores written by older versions used a ' [imp]' / ' [corr+imp]'
    # suffix, which is recognised here too.
    _existing = set(adata.var_names)
    _dup = [g for g in genes_to_predict
            if g in _existing or f"{g} [imp]" in _existing or f"{g} [corr+imp]" in _existing]
    if _dup:
        print(f"  Warning: {len(_dup)} genes already imputed in this table (skipped): "
              f"{_dup[:10]}", flush=True)
        genes_to_predict = [g for g in genes_to_predict if g not in set(_dup)]

    if not genes_to_predict:
        raise ValueError("No valid genes to predict after filtering against loaded reference.")

    # ── Build spatial log-normalized matrix ───────────────────────────────────
    print("Building spatial expression matrix …", flush=True)
    shared_in_rna = [g for g in shared_genes if g in rna_df.columns]
    shared_idx    = [panel_gni[g] for g in shared_in_rna]

    X = adata.layers["X_corrected"] if use_corrected else adata.X
    if sp.issparse(X):
        xen_raw = X[:, shared_idx].toarray().astype(np.float32)
    else:
        xen_raw = np.asarray(X[:, shared_idx]).astype(np.float32)

    rs = xen_raw.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    xen_norm   = np.log1p(xen_raw / rs * 10_000)
    del xen_raw, rs
    spatial_df = pd.DataFrame(xen_norm, index=adata.obs_names, columns=shared_in_rna)

    # ── Run SpaGE ─────────────────────────────────────────────────────────────
    print(f"Running SpaGE (n_pv={n_pv}, {len(genes_to_predict)} genes, "
          f"{len(adata)} cells) …", flush=True)
    imp_df = _vectorized_spage(spatial_df, rna_df, n_pv, genes_to_predict, spage_repo)
    print(f"  Imputation done. Shape: {imp_df.shape}", flush=True)

    # ── Build output AnnData (original + imputed columns) ─────────────────────
    print("Building output AnnData …", flush=True)

    # Same convention as the app: plain gene name as var name plus var["is_imputed"].
    # (The app shows them as "GENE [imp]", or "GENE [corr+imp]" for the genes listed
    # in uns["split_corrected_imputed_genes"]; the suffix is a label, not a name.)
    tagged_genes  = list(genes_to_predict)

    imp_X   = sp.csr_matrix(imp_df.values.astype(np.float32))
    imp_var = pd.DataFrame(
        {"is_imputed": True},
        index=pd.Index(tagged_genes, name=adata.var.index.name or "gene"),
    )

    adata_out = anndata.AnnData(
        X=sp.hstack([adata.X, imp_X]).tocsr(),
        obs=adata.obs.copy(),
        var=pd.concat([adata.var, imp_var]),
        uns=dict(adata.uns),
        obsm=dict(adata.obsm),
    )
    # Preserve X_corrected layer (extended with zeros for new imputed columns)
    if "X_corrected" in adata.layers:
        n_imp = imp_X.shape[1]
        corr_ext = sp.hstack([
            adata.layers["X_corrected"],
            sp.csr_matrix((len(adata), n_imp), dtype=np.float32),
        ]).tocsr()
        adata_out.layers["X_corrected"] = corr_ext

    uns_key = "split_corrected_imputed_genes" if use_corrected else "imputed_genes"
    adata_out.uns[uns_key] = list(adata.uns.get(uns_key, [])) + tagged_genes
    # Also keep the plain imputed_genes list up to date
    if use_corrected:
        adata_out.uns["imputed_genes"] = list(adata.uns.get("imputed_genes", []))

    # ── Write output SpatialData ───────────────────────────────────────────────
    print(f"Writing output to {output} …", flush=True)
    # Copy the full original sdata (shapes, images, etc.) then swap the table
    sdata_out = sd.read_zarr(sdata_path)
    sdata_out.tables["table"] = adata_out
    sdata_out.write(output)

    print(f"Done. {len(genes_to_predict)} genes imputed → {output}", flush=True)
    return output


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="SpaGE gene imputation on a SpatialData zarr.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("sdata_path",  help="Input SpatialData zarr path")
    p.add_argument("rds_path",    help="Seurat RDS reference path")
    p.add_argument("--genes",      nargs="+", metavar="GENE", default=None,
                   help="Genes to impute (space-separated). Omit for auto top-200 HVG.")
    p.add_argument("--genes_file", metavar="FILE", default=None,
                   help="Text file with one gene per line.")
    p.add_argument("--n_pv",       type=int, default=30,
                   help="Number of principal vectors (default: 30)")
    p.add_argument("--output",     metavar="PATH", default=None,
                   help="Output zarr path (default: <sdata_path>_imputed.zarr)")
    p.add_argument("--overwrite",  action="store_true",
                   help="Overwrite output if it already exists")
    p.add_argument("--spage_repo", metavar="PATH", default=None,
                   help="Path to SpaGE_repo directory")
    p.add_argument("--use_corrected", action="store_true",
                   help="Use X_corrected layer (SPLIT output) as input counts. "
                        "Errors if X_corrected is not present in the zarr.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    impute(
        sdata_path=args.sdata_path,
        rds_path=args.rds_path,
        genes=args.genes or None,
        genes_file=args.genes_file,
        n_pv=args.n_pv,
        output=args.output,
        overwrite=args.overwrite,
        use_corrected=args.use_corrected,
        spage_repo=args.spage_repo,
    )
