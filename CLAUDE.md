# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
python xenium_explorer.py [path/to/output-XETG...]   # explicit dataset
python xenium_explorer.py                             # auto-detects output-* dir
# Opens at http://localhost:8050
# Kill: lsof -ti:8050 | xargs kill -9
```

No build step, no test suite, no linter config. The entire app is a single script.

## Architecture

**`xenium_explorer.py`** (~5,800 lines) is a Plotly Dash single-page app. All logic lives here.

### Data model

`DATA` is a global dict holding the active dataset. `EXTRA_DATASETS[]` holds additional samples for multi-dataset views. Key fields:
- `DATA["df"]` — pandas DataFrame from `cells.parquet` (one row per cell)
- `DATA["expr"]` — sparse gene expression matrix from `cell_feature_matrix.h5`
- `DATA["gene_names"]` — ordered list of panel genes (index matches expr columns)
- `DATA["cluster_methods"]` — dict of cluster assignment arrays keyed by method name
- `DATA["umap"]` — UMAP projection coordinates
- `DATA["data_dir"]` — path to the Xenium output directory

Coordinate convention: spatial plots use `x_centroid` (µm) on x-axis and `-y_centroid` (µm, negated) on y-axis. `PIXEL_SIZE_UM = 0.2125` is only used for image↔µm conversion.

### Background processing

Long-running operations run in daemon threads. Each has a paired state dict + lock:

| State dict | Feature |
|---|---|
| `_baysor_state` / `_baysor_lock` | Baysor resegmentation |
| `_proseg_state` / `_proseg_lock` | Proseg resegmentation |
| `_annot_state` / `_annot_lock` | Cell type annotation (CellTypist / Seurat / RCTD) |
| `_spage_state` / `_spage_lock` | SpaGE gene imputation |
| `_umap_reseg_state` / `_umap_reseg_lock` | UMAP on reseg cells |
| `_sdata_state` / `_sdata_lock` | SpatialData / sopa integration |

All reseg result dicts contain: `cells_df`, `cell_bounds`, `out_dir`, `expr` (CSR sparse matrix), `source`.

Dash `dcc.Interval` components poll these state dicts to update progress in the UI.

### Segmentation source

The seg-source dropdown supports multiple cached runs per tool:
- Values: `"xenium"`, `"baysor:abc12345"`, `"proseg:xyz99999"` (param-tagged)
- Helper: `_seg_tool(seg_source)` returns `"xenium"`, `"baysor"`, or `"proseg"`
- Always use `_seg_tool(seg_source) == "baysor"` (never `seg_source == "baysor"`)
- Cache dirs: `~/.xenium_explorer_cache/{tool}_{dataset}_{param_tag}/`
- Each run writes `params.json` with parameters, n_cells, and param_tag

### Key functions

- `load_xenium_data(data_dir)` — parses a Xenium output bundle into `DATA`
- `make_spatial_fig()` — builds the main scatter plot; handles Xenium, Baysor, and Proseg cells
- `make_morphology_overlay()` — loads OME-TIFF tiles for the current viewport
- `_ZarrTiffArray` class — memory-mapped tile accessor for large OME-TIFFs
- `_vectorized_spage()` — NumPy-optimized SpaGE imputation (z-score → PV align → cosine kNN)
- `_get_reseg_expr_values(gene, alt_res)` — extracts log1p expression for a gene from reseg CSR matrix
- `_build_baysor_expr()` / `_build_proseg_expr()` — build cell×gene CSR matrices from transcript assignments
- `_list_cached_seg_runs()` — scans cache dirs for completed Baysor/Proseg runs, returns dropdown options
- `subset(**kwargs)` / `unsubset()` — filter cells in-place; exposed in the Python REPL
- `_run_baysor()` / `_run_proseg()` — spawn external CLI tools in background threads
- `_run_seurat_annotation()` / `_run_rctd_annotation()` / `_run_celltypist()` — annotation backends
- `_run_spage_imputation()` — SpaGE imputation using Seurat RDS reference via rpy2
- `_save_sdata_to_disk(path)` — exports analysis as SpatialData Zarr (includes clusters + cell types)

### UI structure

- Left sidebar (mouse-activated): Color By, Overlays, Seg source, Resegmentation button, Impute Genes button, Annotate Cells button, UMAP controls, tissue info (with Save as SpatialData + Clear Cache)
- Main area: spatial scatter + UMAP (toggle)
- Bottom panel: cell info on click (left) + live server log + Python REPL (right)
- Modals: Resegmentation (Step 1: SpatialData patches, Step 2: Baysor/Proseg), SpaGE imputation, Cell type annotation, Save as SpatialData, Load sample, Clear cache
- 30+ `@app.callback` decorators wire UI interactions

### make_spatial_fig — Baysor/Proseg branch

When `alt_res is not None` (Baysor/Proseg active), color modes work as follows:
- **cluster**: looks for any column starting with `"cluster"` in `cells_df`; if none, falls through to transcript_counts
- **cell_type**: uses `_annot_state[labels_key]` where labels_key is `"labels_baysor"` or `"labels_proseg"`
- **gene**: calls `_get_reseg_expr_values(gene, alt_res)` → `expr.getcol(idx).todense()` → log1p
- **QC metrics** (transcript_counts, cell_area, nucleus_area): direct column lookup in cells_df

### Caching

All caches written to `~/.xenium_explorer_cache/` keyed by dataset hash:
- Annotation labels (`.parquet`): seurat, celltypist, rctd — separate files per segmentation source
- SpaGE results (`.parquet`); auto-loads on startup via `spage_index.json`
- Morphology overviews (`.npz`)
- Baysor/Proseg outputs (subdirs with `params.json`)
- SpatialData Zarr stores

## Known Pitfalls

### rpy2 in Dash callback threads — CRITICAL
Dash callbacks run in worker threads where rpy2's ContextVar for conversion rules is not set. Any function using rpy2 in a background thread needs this at the start:
```python
import rpy2.robjects.conversion as _rconv
try:
    if _rconv._get_cv().get(None) is None:
        _rconv.activate(ro.default_converter)
except Exception:
    try:
        _rconv.activate(ro.default_converter)
    except Exception:
        pass
```
For thread-spawning callbacks (not the thread itself), use `contextvars.copy_context()` + `ctx.run(fn, ...)`.
Applied in: `_run_spage_imputation`, `_run_seurat_annotation`, `_run_rctd_annotation`.

### Startup modal callbacks firing on dynamic component creation
`prevent_initial_call=True` does not prevent callbacks from firing when `update_tissue_info` dynamically creates button components. Always guard modal-open callbacks with `and n_clicks` (or `and open_clicks`) to no-op when `n_clicks == 0`.

### Dash duplicate outputs
When multiple callbacks share the same output (e.g., `seg-source.value`), use `allow_duplicate=True` on all but the primary one, and `prevent_initial_call=True` on all. Forgetting either causes a Dash error at startup.

## Dependencies

Core: `dash`, `dash-bootstrap-components`, `plotly`, `pandas`, `numpy`, `scipy`, `h5py`, `Pillow`, `tifffile`, `zarr`, `scikit-learn`

Optional:
- `anndata`, `celltypist` — CellTypist annotation
- `rpy2` + R packages `SeuratObject`, `Matrix` — Seurat label transfer and SpaGE reference
- `rpy2` + R package `spacexr` — RCTD annotation (`devtools::install_github('dmcable/spacexr')`)
- `spatialdata`, `sopa` — SpatialData patch-based resegmentation and export
- `umap-learn` — UMAP recomputation on reseg cells

External tools (must be on PATH or at default locations):
- `baysor` — expects `~/.julia/bin/baysor` (v0.7.1); flag: `--polygon-format FeatureCollection`
- `proseg` — `conda install bioconda::rust-proseg` or `cargo install proseg`
