# Xenium Visualizer

An interactive web application for visualizing and analyzing 10x Genomics Xenium spatial transcriptomics data. Supports cell clustering, gene expression, morphology overlays, cell type annotation, gene imputation, resegmentation, and ambient RNA correction (SPLIT) — all in a browser-based interface.

## Installation

### Option A — Conda (recommended)

Handles Python, R, and rpy2 linkage in one step:

```bash
conda env create -f environment.yml
conda activate xenium-explorer
```

Then install R packages (one-time, inside the activated environment):

```bash
Rscript -e "install.packages('BiocManager')"
Rscript -e "BiocManager::install(c('SeuratObject', 'Matrix'))"
Rscript -e "devtools::install_github('dmcable/spacexr')"   # RCTD + SPLIT
Rscript -e "remotes::install_github('bdsc-tds/SPLIT')"     # ambient RNA correction
```

And Baysor (Julia, optional — only needed for Baysor resegmentation):

```bash
julia -e 'using Pkg; Pkg.add("Baysor")'
# binary installed to ~/.julia/bin/baysor
```

### Option B — pip only

Requires Python 3.11 and (for rpy2 features) R ≥ 4.3 already on PATH:

```bash
pip install -r requirements.txt
```

R packages are the same as Option A above.

---

## Usage

```bash
python xenium_explorer.py [path/to/output-XETG...]
python xenium_explorer.py          # auto-detects output-* directory
```

Opens at **http://localhost:8050**.

To stop the server:
```bash
lsof -ti:8050 | xargs kill -9
```

## Input Data

Expects a standard **10x Genomics Xenium output directory** containing:

| File | Description |
|------|-------------|
| `experiment.xenium` | Run metadata (tissue type, panel, pixel size, counts) |
| `cells.parquet` | Cell centroids, transcript counts, cell/nucleus area |
| `cell_boundaries.parquet` | Cell polygon vertices |
| `nucleus_boundaries.parquet` | Nucleus polygon vertices |
| `cell_feature_matrix.h5` | Sparse gene expression matrix (HDF5) |
| `transcripts.parquet` | Individual transcript coordinates (used by Baysor/Proseg) |
| `analysis/umap/gene_expression_2_components/projection.csv` | UMAP coordinates *(optional)* |
| `analysis/clustering/*/clusters.csv` | Cluster assignments, one file per method *(optional)* |
| `morphology_focus/morphology_focus_ZZZZ.ome.tif` | Fluorescence z-stack images *(optional)* |

### SpatialData zarr store

On first launch, the app consolidates all cell-level data into a **`spatialdata_xenium.zarr`** file written next to the raw data (one-time, ~30–60 s for a typical dataset). Subsequent launches load exclusively from this zarr — the raw files are no longer read at startup.

The zarr stores:
- `table` — AnnData with the full cell × gene expression matrix, all obs metadata (centroids, QC metrics, UMAP, cluster assignments), and a `is_imputed` flag per gene
- `cell_boundaries` / `nucleus_boundaries` — GeoDataFrame shapes

Images are still read directly from `morphology_focus/` on demand.

## Features

### Sidebar

The left sidebar is hidden by default and slides in when the mouse moves to the left edge of the screen. Sections:

- **Tissue info** — Run name, region, and dataset stats (cells, transcripts, panel). Stats collapse by default and expand on hover. Includes buttons to load a different sample, save as SpatialData, and clear cache.
- **Color By** — Dropdown to select the coloring mode.
- **Segmentation source** — Dropdown to switch between Xenium (original), Baysor, and Proseg segmentations. Shows all cached runs per tool (labeled by parameters); select any cached run to load it instantly. A 🗑 delete button appears next to the dropdown to permanently remove a cached run (requires confirmation).
- **Overlays** — Multi-select checklist: Xenium Cell Boundaries, Xenium Nuclei Boundaries, Baysor Boundaries, Proseg Boundaries, and Morphology Image. Multiple overlays can be active simultaneously.
- **Point style** — Size and opacity sliders.
- **Resegmentation** — Single "Resegment Cells" button opens a modal with SpatialData patch setup (Step 1) and Baysor/Proseg algorithm tabs (Step 2).
- **Impute Genes** — Button opens the SpaGE imputation modal.
- **Annotate Cells** — Button opens the cell type annotation modal.
- **SPLIT Correction** — Button opens the SPLIT ambient RNA correction modal.
- **Active Counts** — Radio toggle to switch between original and SPLIT-corrected expression counts. The "SPLIT Corrected" option is grayed out until SPLIT has been run for the active segmentation.
- **UMAP** — Show/hide toggle and "Rerun UMAP" button. Works on Xenium data (recomputes and updates `DATA["df"]`) as well as Baysor/Proseg reseg cells.

A gold **⬡ SUBSET ACTIVE** banner appears below the tissue info when a cell subset is active, showing the count and a "✕ Clear Subset" button.

### Visualization

**Color modes** (Color By dropdown in sidebar):
- **Cluster** — Categorical coloring by clustering method; multiple methods supported via a secondary dropdown. For Baysor/Proseg cells, KMeans clusters (`cluster_kmeans_10`) are computed automatically after resegmentation. After running SPLIT, a `cluster_split_10` column (KMeans on corrected counts) also appears in the dropdown.
- **Gene Expression** — Continuous coloring by log1p expression; supports panel genes, SpaGE-imputed genes (marked `[imp]`), and genes imputed on corrected counts (marked `[corr+imp]`). The "Active Counts" toggle transparently switches the expression source between original and SPLIT-corrected matrices.
- **Cell Type** — Categorical coloring by annotation result (enabled after running annotation)
- **Transcript Counts** — QC metric, Viridis colorscale
- **Cell Area / Nucleus Area** — Cell size QC metrics, Plasma/Cividis colorscales

All color modes work on Xenium, Baysor, and Proseg cells after switching the segmentation source.

**Plots:**
- **Spatial plot** — 2D scatter of all cells at their µm centroids
- **UMAP plot** — Dimensionality reduction view; hidden by default, toggle with "Show/Hide UMAP" button. UMAP is computed automatically after resegmentation. Use the "Run UMAP on Reseg Cells" button to recompute with different parameters. After running SPLIT, switching to "SPLIT Corrected" in the Active Counts toggle displays a separate UMAP computed on the corrected expression matrix (`split_umap_1`/`split_umap_2`).

**Point rendering:**
- Point size: 1–8 px (default 2)
- Opacity: 0.1–1.0 (default 0.85)

### Overlays

Select any combination of overlays from the checklist in the sidebar:

**Xenium Cell/Nuclei Boundaries** — Original Xenium polygon outlines rendered for cells in the current viewport (limit: 3,000 cells). Zoom in to activate. These always show Xenium boundaries regardless of the active segmentation source.
- Xenium cell boundaries: cyan
- Xenium nuclei boundaries: orange
- Baysor/Proseg boundaries: their own separate checkboxes; shown for all reseg cells (no viewport limit)

**Morphology Image** — Fluorescence overlay loaded from OME-TIFF tiles for viewports ≤ 5,000 µm wide.
- Z-levels 0–3
- Channels (multi-select):
  | Channel | Marker | Color |
  |---------|--------|-------|
  | 0 | DAPI | Blue |
  | 1 | ATP1A1 / CD45 / E-Cadherin | Green |
  | 2 | 18S rRNA | Red |
  | 3 | αSMA / Vimentin | Magenta |
- Brightness: 0.5–8× (default 2×)
- Image opacity: 0.1–1.0 (default 0.85)
- Pan/zoom updates the overlay automatically; a low-resolution overview is pre-generated for zoomed-out views

### Cell Info Panel

Click any cell in the spatial or UMAP plot to see:
- Cell ID, cluster, X/Y coordinates (µm), transcript count, cell area, nucleus area, UMAP coordinates
- Bar chart of top 12 expressed genes
- Works for Xenium, Baysor, and Proseg cells

### Resegmentation

Click **"Resegment Cells"** in the sidebar to open the resegmentation modal. The modal has two steps:

**Step 1 — Patch setup (SpatialData)**

Configure patch parameters, then click **▶ Prepare Patches** to build the SpatialData object and spatial ROI:

| Parameter | Default |
|-----------|---------|
| Quality Value (QV) filter | 20 |
| Patch width (µm) | 1200 |
| Min transcripts/patch | 10 |

The ROI and patch grid are displayed as overlays on the spatial plot. Click **✓ Patches look good — proceed** to unlock Step 2, or **⏭ Skip — run on full tissue** to bypass patch setup and run directly on the full slide. Use **↺ Re-build** to clear the cache and re-run with different parameters.

**Step 2 — Algorithm selection**

Select the algorithm tab (Baysor or Proseg), configure parameters, and click **Run**. Progress appears in the server log.

Once complete:
- The segmentation source dropdown automatically switches to the new segmentation
- Baysor/Proseg boundary overlays become available
- KMeans clustering (10 clusters) and UMAP are computed automatically; results are saved to `spatialdata_{source}.zarr` and available immediately
- All color modes (gene expression, cluster, cell type, QC metrics) work on reseg cells
- Cell annotation and gene imputation can be run on reseg cells; imputed genes are saved to the reseg zarr
- UMAP can be re-computed manually with the "Rerun UMAP" button (works for Xenium, Baysor, and Proseg)

**Baysor** — Julia-based algorithm ([Baysor](https://github.com/kharchenkolab/Baysor)):

| Parameter | Range | Default |
|-----------|-------|---------|
| Spatial region X/Y (µm) | any | full slide |
| Cell radius (µm) | 5–60 | 20 |
| Min transcripts/cell | 1–500 | 10 |
| Use Xenium prior segmentation | checkbox | enabled |
| Prior confidence | 0–1 | 0.5 |

**Use Current Viewport** — fills the spatial region from the current pan/zoom state (run Baysor on just the visible area to avoid OOM on large slides).

**Proseg** — Rust-based probabilistic algorithm ([Proseg](https://github.com/dcjones/proseg)); faster than Baysor, morphologically constrained:

| Parameter | Flag | Default |
|-----------|------|---------|
| Spatial region X/Y (µm) | — | full slide |
| Initial voxel size (µm) | `--initial-voxel-size` | 4 |
| Threads | `--nthreads` | all cores |
| Components | `--ncomponents` | 10 |
| Recorded samples | `--recorded-samples` | 100 |
| Sampler schedule | `--schedule` | `150 150 300` |
| Nuclear reassignment prob | `--nuclear-reassignment-prob` | 0.2 |
| Prior seg reassignment prob | `--prior-seg-reassignment-prob` | 0.5 |

Resegmentation output is cached per run (keyed by parameters) in `~/.xenium_explorer_cache/baysor_{dataset}_{tag}/` or `proseg_{dataset}_{tag}/`. Multiple cached runs per tool are accessible from the segmentation source dropdown.

### Cell Type Annotation

Click **"Annotate Cells"** in the sidebar to open the annotation modal. Three methods available. Annotation runs on the currently active segmentation (Xenium, Baysor, or Proseg). Each segmentation's annotations are cached separately.

**CellTypist** — Uses pretrained Azimuth-compatible models:
- Healthy Adult Heart (default)
- Immune All – Low
- Immune All – High

**Seurat RDS Reference** — Transfer labels from a custom Seurat object:
- Specify the RDS file path and the metadata column containing cell type labels (default: `Names`)
- Performs PCA on shared genes → cosine kNN (k=50) → majority vote label transfer to each Xenium cell
- Only the shared genes (~500) are extracted in-process via rpy2 — no subprocess re-read of the full RDS

**RCTD (spacexr)** — Probabilistic cell type deconvolution using the spacexr R package:
- Specify the RDS file path and label column
- Mode: `full` (argmax of weight per cell), `doublet` (best single + doublet assignment), `multi` (multi-type)
- Max cores: number of parallel R workers (default 4)
- Requires `spacexr` R package: `devtools::install_github('dmcable/spacexr')`

Results are cached to `~/.xenium_explorer_cache/` and load instantly on subsequent runs.

### SpaGE Gene Imputation

Click **"Impute Genes"** in the sidebar to open the SpaGE imputation modal. Imputes expression of genes not in the Xenium panel using a scRNA-seq reference (Seurat RDS).

**GUI (modal):**
- **Reference RDS** — Path to Seurat object
- **n_pv** — Number of principal vectors for cross-dataset alignment (10–100, default 50)
- **Genes to impute** — Comma- or newline-separated gene list; leave blank to auto-select top-200 highly variable genes from the reference

**REPL (programmatic):**
```python
run_spage(
    rds_path="/path/to/reference.rds",
    genes_file="/path/to/genes_to_impute.txt",  # one gene per line; omit for auto-HVG
    n_pv=50,
)
```
Progress is printed to the server log. When complete, imputed genes appear in the Color By dropdown automatically.

Imputed genes appear in the gene expression dropdown with an `[imp]` suffix. Results are **written back into the active SpatialData zarr** (`spatialdata_xenium.zarr` for Xenium, or `spatialdata_{source}.zarr` for Baysor/Proseg) so they persist across restarts — imputed genes are available immediately on the next launch without re-running SpaGE. When a Baysor/Proseg segmentation is active, only genes imputed for that segmentation are shown in the dropdown. The parquet cache (`~/.xenium_explorer_cache/spage_*.parquet`) is also written for backward compatibility.

**Algorithm:** Z-score normalization → Principal Vector alignment (domain adaptation) → cosine kNN (k=50) → inverse-distance weighted average. Vectorized with NumPy einsum (~200× faster than the original SpaGE loop).

Reference: [Abdelaal et al. 2020, *Nucleic Acids Research*](https://academic.oup.com/nar/article/48/18/e107/5909530)

### SPLIT Ambient RNA Correction

Click **"✦ Correct Ambient RNA"** in the sidebar to open the SPLIT correction modal. SPLIT removes ambient/contaminating RNA from spatial transcriptomics data via RCTD doublet-mode decomposition followed by `SPLIT::purify()`.

**Requirements:**
- R packages: `spacexr` and `SPLIT`
  ```r
  devtools::install_github('dmcable/spacexr')
  remotes::install_github('bdsc-tds/SPLIT')
  ```
- Peak RAM: ~21 GB for a full Xenium slide
- Runtime: ~20–40 min (RCTD is the bottleneck)

**Modal fields:**

| Field | Default | Description |
|-------|---------|-------------|
| Seurat Reference RDS path | — | Path to `.rds` Seurat object used as cell-type reference |
| Cell-type label column | `Names` | Metadata column in the Seurat object containing cell type labels |
| Max R cores | 4 | Parallel workers for RCTD |
| Min UMI | 10 | Cells with fewer UMIs are excluded from RCTD (`UMI_min`) |
| Min UMI sigma | 100 | Minimum UMI count for sigma parameter estimation in RCTD (`UMI_min_sigma`) |
| Purify singlets | ✓ | Also remove ambient RNA from singlet cells (`DO_purify_singlets`). Disable if corrected expression appears over-reduced. |

Click **▶ Run SPLIT** to start. Progress is streamed to the server log. When complete:

- The **"SPLIT Corrected"** radio option in the sidebar becomes enabled
- The active counts mode switches automatically to "SPLIT Corrected"
- A `cluster_split_10` column (KMeans on corrected counts) appears in the cluster dropdown
- A corrected-counts UMAP (`split_umap_1`/`split_umap_2`) is shown when the corrected mode is active
- Corrected counts are persisted in the `X_corrected` layer of the active SpatialData zarr and reload automatically on the next launch

**Active Counts toggle behavior:**
- **Original** — all expression lookups use the raw Xenium/Baysor/Proseg count matrix; `[imp]` genes from SpaGE are shown
- **SPLIT Corrected** — expression lookups use the corrected matrix; `[corr+imp]` genes (imputed on corrected counts via SpaGE) are shown instead
- The "SPLIT Corrected" option is greyed out (disabled) whenever no corrected matrix is available — either in memory or in the active zarr's `X_corrected` layer. It re-enables automatically once the corrected data is loaded (e.g., after selecting a Baysor/Proseg run that has a previously-run SPLIT result stored in zarr)
- Switching segmentation sources resets the toggle to "Original" if no corrected data exists for that source

**SpaGE on corrected counts:** Run SpaGE while "SPLIT Corrected" is active to impute genes using the corrected expression matrix. Resulting genes are marked `[corr+imp]` and stored in `adata.uns["split_corrected_imputed_genes"]` in the zarr.

### Save as SpatialData

Click **"💾 Save as SpatialData"** in the tissue info section to export the current analysis as a SpatialData Zarr store. Specify an output directory and filename. The saved Zarr includes:
- Cell/nucleus boundaries
- Cell metadata (coordinates, QC metrics)
- Gene expression table
- Cluster labels (all methods)
- Cell type annotations (if run)

The file can be loaded by any SpatialData-compatible tool (e.g., napari-spatialdata, squidpy).

### Cell Subsetting

Filter the displayed cells programmatically from the **Python REPL** in the bottom panel. All filters are ANDed together. The plots refresh within ~0.5 s.

```python
# By cluster (integer ID, uses first cluster method by default)
subset(cluster=3)
subset(cluster=[1, 2, 3])
subset(cluster=5, method='gene_expression_10_clusters')

# By cell type (requires annotation)
subset(cell_type='Cardiomyocyte')
subset(cell_type=['Cardiomyocyte', 'Fibroblast'])

# By gene expression (log1p)
subset(gene='MYH7', min_expr=1.0)
subset(gene='CD45', min_expr=0.5, max_expr=3.0)

# By QC metrics
subset(min_transcripts=100, max_cell_area=800)

# Combinations
subset(cluster=2, min_transcripts=50, gene='MYH7', min_expr=0.5)

# Restore full dataset
unsubset()
```

A gold **⬡ SUBSET ACTIVE** indicator appears in the sidebar with the cell count and a "✕ Clear Subset" button.

### Server Log & Python REPL

The bottom panel is split into two columns:
- **Left** — Cell info on click (cell ID, coordinates, QC metrics, gene expression bar chart)
- **Right** — Live server log streaming Python stdout/stderr and R console output (rpy2). Auto-scrolls to the bottom; pauses when you scroll up and resumes when you scroll back down.

The entire panel can be collapsed/expanded with the ▼/▲ button.

**Python REPL** — A `>>> ` prompt at the bottom of the log panel accepts any Python expression or statement and executes it in the app's module namespace, with output appearing in the log. Tab-completion and Up/Down arrow history are supported.

```python
# Examples
subset(cluster=3)
unsubset()
get_genes(file="xenium_genes.txt")       # list panel genes; optionally save to file
run_spage("ref.rds", "to_impute.txt")    # programmatic SpaGE imputation
len(DATA["df"])
list(DATA["cluster_methods"])
_annot_state["status"]
```

## Caching

All caches are stored in `~/.xenium_explorer_cache/` and are keyed per dataset.

**In the Xenium output directory:**

| File | Contents |
|------|----------|
| `spatialdata_xenium.zarr` | Primary data store — generated on first launch, loaded on all subsequent launches. Contains expression matrix, obs metadata, cluster assignments, boundaries, SpaGE-imputed genes, and (after SPLIT) the `X_corrected` layer with corrected counts, `cluster_split_10`, and `split_umap_1/2`. |

**In `~/.xenium_explorer_cache/`:**

| Cache | Contents |
|-------|----------|
| `seurat_{basename}_{col}_{hash}.parquet` | Seurat kNN annotation labels |
| `celltypist_{model}_{hash}.parquet` | CellTypist annotation labels |
| `rctd_{basename}_{col}_{mode}_{hash}.parquet` | RCTD annotation labels |
| `labels_baysor_{hash}.parquet` | Cell type annotation labels (Baysor cells) |
| `labels_proseg_{hash}.parquet` | Cell type annotation labels (Proseg cells) |
| `spage_{hash}.parquet` | SpaGE imputed gene expression (backward-compat; primary store is the zarr) |
| `spage_index.json` | Maps dataset paths to most recent SpaGE cache |
| `morph_overview_{hash}_z{level}.npz` | Pre-downsampled morphology overview (generated on first load) |
| `spatialdata_{hash}.zarr` | SpatialData Zarr store for sopa-based patch resegmentation |
| `baysor_{dataset}_{tag}/` | Baysor output per parameter set; includes `params.json` and `spatialdata_baysor.zarr` (contains expression, clusters, UMAP, imputed genes, and `X_corrected` layer after SPLIT) |
| `proseg_{dataset}_{tag}/` | Proseg output per parameter set; includes `params.json` and `spatialdata_proseg.zarr` (contains expression, clusters, UMAP, imputed genes, and `X_corrected` layer after SPLIT) |

## Standalone Functions (`functions/`)

A set of standalone Python scripts for programmatic use outside the app. All can be imported as modules or run as CLI tools.

### `functions/impute.py` — SpaGE gene imputation

Imputes expression of genes not in the Xenium panel using a Seurat scRNA-seq reference, and writes results into a SpatialData zarr.

```python
from functions.impute import impute

# Explicit gene list
impute(
    sdata_path="/data/xenium.zarr",
    rds_path="/refs/heart_ref.rds",
    genes=["TNNT2", "MYH7", "ACTN2"],
    n_pv=30,
    output="/data/xenium_imputed.zarr",
)

# From a text file (one gene per line)
impute(
    sdata_path="/data/xenium.zarr",
    rds_path="/refs/heart_ref.rds",
    genes_file="/data/genes_to_impute.txt",
    output="/data/xenium_imputed.zarr",
)

# Auto top-200 HVGs from reference (omit genes and genes_file)
impute(
    sdata_path="/data/xenium.zarr",
    rds_path="/refs/heart_ref.rds",
    output="/data/xenium_imputed.zarr",
)
```

CLI:
```bash
python functions/impute.py /data/xenium.zarr /refs/heart_ref.rds \
    --genes TNNT2 MYH7 ACTN2 --n_pv 30 --output /data/xenium_imputed.zarr

python functions/impute.py /data/xenium.zarr /refs/heart_ref.rds \
    --genes_file /data/genes.txt --output /data/xenium_imputed.zarr
```

Output is a full SpatialData zarr with imputed gene columns appended to the AnnData table. `adata.var["is_imputed"]` marks imputed genes; `adata.uns["imputed_genes"]` lists them.

---

### `functions/spatialdata2seurat.py` — SpatialData → Seurat RDS

Converts a SpatialData zarr to a Seurat RDS equivalent to `LoadXenium()` output.

| Seurat slot | Content |
|---|---|
| `@assays$Xenium` | Panel gene raw counts (dgCMatrix, genes × cells) |
| `@assays$Imputed` | SpaGE-imputed genes, if present |
| `@meta.data` | All obs columns + `nCount_Xenium`, `nFeature_Xenium` |
| `@images$fov` | Centroids + cell segmentation + nucleus segmentation |

```python
from functions.spatialdata2seurat import spatialdata2seurat

# Standard conversion
spatialdata2seurat(
    sdata_path="/data/xenium.zarr",
    output_rds="/data/xenium.rds",
)

# SPLIT-corrected counts, skip imputed assay
spatialdata2seurat(
    sdata_path="/data/xenium.zarr",
    output_rds="/data/xenium_corrected.rds",
    use_corrected=True,
    include_imputed=False,
)
```

CLI:
```bash
python functions/spatialdata2seurat.py /data/xenium.zarr /data/xenium.rds
python functions/spatialdata2seurat.py /data/xenium.zarr /data/xenium.rds \
    --use_corrected --no_imputed --no_nucleus
```

Requires: `rpy2`, R packages `Seurat`, `SeuratObject`, `Matrix`.

---

### `functions/merge_spatialdata.py` — Merge multiple SpatialData zarrs

Merges two or more SpatialData zarr stores into a single store, offsetting x/y coordinates so datasets don't spatially overlap. Useful for side-by-side comparison of multiple Xenium runs, or combining Baysor/Proseg outputs from different conditions.

```python
from functions.merge_spatialdata import merge_spatialdata

merge_spatialdata(
    input_paths=["ctrl.zarr", "treat.zarr"],
    output_path="merged.zarr",
    gap=500,           # µm gap between datasets (default: 200)
    layout="row",      # "row" | "column" | "grid"
    labels=["ctrl", "treat"],   # cell-ID prefix per dataset (default: s0, s1, …)
    genes="union",     # "union" (fill missing with 0) | "intersect"
)
```

CLI:
```bash
python functions/merge_spatialdata.py ctrl.zarr treat.zarr -o merged.zarr --labels ctrl treat
python functions/merge_spatialdata.py s1.zarr s2.zarr s3.zarr -o merged.zarr --layout grid --gap 500
```

Coordinate offsets are printed per dataset. Cell IDs are prefixed with the label (e.g. `ctrl_cell123`). A `dataset` column is added to `obs`. Requires `geopandas` and `shapely` for shape merging (optional; shapes are skipped with a warning if unavailable).

---

### Other functions

| Script | Purpose |
|--------|---------|
| `functions/run_split.py` | Standalone RCTD + SPLIT ambient RNA correction; reads/writes directly to zarr |
| `functions/merge_spatialdata.py` | Merge multiple SpatialData zarr stores into one, offsetting coordinates so datasets don't overlap |
| `functions/export_to_cellnest.py` | Export SpatialData zarr to CellNEST-ready AnnData `.h5ad` |
| `functions/migrate_baysor_cache.py` | Migrate old-format Baysor cache directories to current zarr format |

---

## Dependencies

See `environment.yml` (conda) or `requirements.txt` (pip) for the full pinned dependency list.

**Summary:**

| Package | Purpose |
|---------|---------|
| `dash`, `dash-bootstrap-components`, `plotly` | Web framework and plots |
| `numpy`, `pandas`, `scipy`, `scikit-learn` | Core data processing |
| `h5py` | Reading `cell_feature_matrix.h5` |
| `Pillow`, `tifffile` | OME-TIFF morphology image loading |
| `zarr>=3.0` | SpatialData zarr store (v3 API required) |
| `anndata`, `geopandas`, `shapely` | SpatialData zarr generation |
| `spatialdata`, `sopa` | Patch-based resegmentation workflow |
| `celltypist` | CellTypist cell-type annotation |
| `umap-learn` | UMAP recomputation on reseg cells |
| `rpy2` + R ≥ 4.3 | Seurat label transfer, SpaGE, RCTD, SPLIT |

**R packages** (via rpy2):

| Package | Source |
|---------|--------|
| `SeuratObject`, `Matrix` | CRAN / Bioconductor |
| `spacexr` | `devtools::install_github('dmcable/spacexr')` |
| `SPLIT` | `remotes::install_github('bdsc-tds/SPLIT')` |

**External CLI tools:**
- [Baysor](https://github.com/kharchenkolab/Baysor) — Julia-based resegmentation; binary at `~/.julia/bin/baysor`
- [Proseg](https://github.com/dcjones/proseg) — Rust-based probabilistic resegmentation; `conda install bioconda::rust-proseg` or `cargo install proseg`
