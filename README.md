# Xenium Visualizer

An interactive web application for visualizing and analyzing 10x Genomics Xenium spatial transcriptomics data. Supports cell clustering, gene expression, morphology overlays, cell type annotation, gene imputation, resegmentation, and ambient RNA correction (SPLIT) — all in a browser-based interface.

Works with standard Xenium panels (a few hundred genes) and with **Xenium Prime 5K** output. A GPU-accelerated **WebGL view** (beta) is available alongside the default Plotly view for large datasets; see [WebGL view](#webgl-view-beta).

## Cloud Run Deployment

The app can be deployed to **Google Cloud Run** for remote access. The container files are in the repository root: `Dockerfile`, `.dockerignore`, `requirements-cloud.txt` and `deployment_instructions.txt` (the step-by-step guide). **The image has not been built or tested yet** — treat the files as a starting point and check a first build before relying on it.

- The image is **Python-only**: no R / rpy2, Baysor, Proseg or sopa. Seurat and RCTD annotation, SpaGE, SPLIT, pseudobulk differential expression (DESeq2 / edgeR), resegmentation and **Prepare Patches** are therefore unavailable there; CellTypist annotation and the Python analyses work.
- The container runs as a **non-root** user.
- Deploy with `--no-allow-unauthenticated --max-instances 1 --no-cpu-throttling`: the app keeps its state in one process and runs background jobs between requests, and it has no login of its own.
- **Never set `XENIUM_ENABLE_REPL`** on a deployed service: the REPL runs arbitrary code.
- On first load the app writes `spatialdata_xenium.zarr` **into the bundle directory**: mount the bucket read-write, or upload a bundle in which that store was already generated locally.
- The cache (`XENIUM_CACHE_DIR`) is **ephemeral** unless a volume is mounted for it: annotations, density grids, saved groups and segmentation runs are lost when the instance stops.

**Quick summary:**

```bash
# 1. Upload data to GCS
gsutil -m cp -r /path/to/output-XETG... gs://YOUR_BUCKET/

# 2. Build and push container
gcloud builds submit --tag gcr.io/YOUR_PROJECT/xenium-explorer

# 3. Deploy with GCS FUSE mount
gcloud run deploy xenium-explorer \
  --image gcr.io/YOUR_PROJECT/xenium-explorer \
  --memory 8Gi --cpu 4 --timeout 3600 \
  --no-allow-unauthenticated --max-instances 1 --no-cpu-throttling \
  --execution-environment gen2 \
  --add-volume name=data-vol,type=cloud-storage,bucket=YOUR_BUCKET \
  --add-volume-mount volume=data-vol,mount-path=/data \
  --set-env-vars DATA_DIR=/data/output-XETG...,XENIUM_HOST=0.0.0.0
```

The server binds to `127.0.0.1` unless `XENIUM_HOST` is set, so a container must set `XENIUM_HOST=0.0.0.0`. (The generic `HOST` variable is not read: shells such as tcsh and conda compiler packages export it for other purposes.)

**Env vars** (all optional locally, required or useful on Cloud Run):

| Env var | Purpose |
|---------|---------|
| `DATA_DIR` | Path to Xenium output directory (overrides `sys.argv[1]`) |
| `XENIUM_CACHE_DIR` | Cache root directory (default: `~/.xenium_explorer_cache`) |
| `PORT` | Server port (default: 8050; Cloud Run sets this automatically) |
| `XENIUM_HOST` | Bind address (default: `127.0.0.1`; set `0.0.0.0` in a container or to serve other machines). The address in use is printed at start-up |
| `XENIUM_ENABLE_REPL` | Enables the in-app Python REPL, which runs arbitrary code. Unset = disabled; `1` = requests made from this machine to `localhost` / `127.0.0.1` / `[::1]` only (not through a reverse proxy); `remote` = any client (trusted networks only) |
| `XENIUM_RENDERER` | Spatial renderer shown at startup: unset = Plotly; `webgl` = the WebGL view (it can also be switched in the app) |

**Cloud Run limitations:** R/Julia/Rust features (Seurat, RCTD, SpaGE, Baysor, Proseg, SPLIT, pseudobulk DESeq2 / edgeR) and Prepare Patches (sopa) are unavailable in the Python-only container image. CellTypist annotation works.

## Continuous integration

`.github/workflows/tests.yml` runs the **default pytest tier** on every push and pull request to `main` (Python 3.11, Ubuntu). The `tools`, `browser` and `slow` tiers are not run in CI (they need Baysor / Proseg / R, a browser with WebGL, or large bundles). If the fixture bundle (`tests/data/prime_mouse_ileum_tiny/`) is not in the checkout, the tests that need it are skipped with a warning.

---

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

Opens at **http://localhost:8050**. The server listens on localhost only; set `XENIUM_HOST=0.0.0.0` to reach it from another machine (there is no authentication, so do this on trusted networks only).

Useful start-up variants:

```bash
XENIUM_RENDERER=webgl python xenium_explorer.py <bundle>     # start in the WebGL view
XENIUM_ENABLE_REPL=1  python xenium_explorer.py <bundle>     # enable the in-app Python REPL
python xenium_explorer.py path/to/exported.zarr              # open a store written by "Save as SpatialData"
```

To stop the server:
```bash
lsof -ti:8050 | xargs kill -9
```

## Input Data

Expects a standard **10x Genomics Xenium output directory**. Only the first three files are required; everything else is used when present.

| File | Description |
|------|-------------|
| `experiment.xenium` | Run metadata (tissue type, panel, pixel size, counts) |
| `cells.parquet` | Cell centroids, transcript counts, cell/nucleus area |
| `cell_feature_matrix.h5` | Sparse gene expression matrix (HDF5) |
| `cell_boundaries.parquet` | Cell polygon vertices *(optional)* |
| `nucleus_boundaries.parquet` | Nucleus polygon vertices *(optional)* |
| `transcripts.parquet` | Individual transcript coordinates (used by Baysor/Proseg) *(optional)* |
| `analysis/` or `analysis.tar.gz` | UMAP coordinates (`umap/gene_expression_2_components/projection.csv`) and cluster assignments (`clustering/*/clusters.csv`). Read from the folder if extracted, otherwise straight from the archive *(optional)* |
| `morphology_focus/` | Morphology images, either `morphology_focus_NNNN.ome.tif` or the newer `chNNNN_<stain>.ome.tif` naming; single-channel and multi-file layouts are both supported *(optional)* |
| `cells.zarr.zip`, `transcripts.zarr.zip` | Tiled boundaries and transcripts written by Xenium Onboard Analysis; used by the WebGL view *(optional)* |

Without clustering the app starts with no cluster sets (Color By falls back to transcript counts); without a UMAP projection the UMAP is empty until "Rerun UMAP" is used.

**Control features.** Only features of type `Gene Expression` are loaded as genes. Negative-control probes and codewords, unassigned and deprecated codewords, and genomic controls are dropped when the store is first built. On a Xenium Prime 5K run this is the difference between 5,006 genes and 13,780 features.

**Processed deposits (e.g. GEO).** Public deposits often contain only a matrix, a per-cell table and transcripts per section. `functions/geo_to_bundle.py` assembles a loadable bundle from those; see [Standalone Functions](#standalone-functions-functions).

### SpatialData zarr store

On first launch, the app consolidates all cell-level data into a **`spatialdata_xenium.zarr`** file written next to the raw data (one-time, ~30–60 s for a typical dataset). Subsequent launches load exclusively from this zarr — the raw files are no longer read at startup.

The zarr stores:
- `table` — AnnData with the full cell × gene expression matrix, all obs metadata (centroids, QC metrics, UMAP, cluster assignments), and a `is_imputed` flag per gene
- `cell_boundaries` / `nucleus_boundaries` — GeoDataFrame shapes

Images are still read directly from `morphology_focus/` on demand.

A store built before control-feature filtering was added keeps its original feature list and loads unchanged. Deleting and rebuilding it drops the control features, after which older resegmentation, SpaGE and SPLIT caches built against the old gene list no longer line up and should be regenerated.

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
- **Gene Expression** — Continuous coloring by log1p expression; supports panel genes and SpaGE-imputed genes (marked `[imp]`; `[corr+imp]` for genes that `functions/impute.py --use_corrected` imputed from SPLIT-corrected counts). The "Active Counts" toggle switches the expression source of the measured genes between the original and the SPLIT-corrected matrix; imputed genes have one set of values and are shown unchanged in both modes.
- **Cell Type** — Categorical coloring by annotation result (enabled after running annotation)
- **Transcript Counts** — log1p(transcript count), Viridis colorscale
- **Cell Area / Nucleus Area** — Cell size QC metrics, Plasma/Cividis colorscales
- **Spatial Niche** — Categorical coloring by spatial niche assignment (enabled after running "Run" in the Spatial Analysis Tools modal, accessible from the Plot tab)

All color modes work on Xenium, Baysor, and Proseg cells after switching the segmentation source.

**Colorbar controls** — When coloring by a continuous variable (gene expression, transcript counts, cell area, nucleus area), a **Colorbar Range** control appears in the sidebar with Min and Max inputs and a Reset button. Drag the inputs to clamp the colorscale to a specific range without changing the underlying data.

**Plots:**
- **Spatial plot** — 2D scatter of all cells at their µm centroids
- **UMAP plot** — Dimensionality reduction view; hidden by default, toggle with "Show/Hide UMAP" button. UMAP is computed automatically after resegmentation. Use the "Run UMAP on Reseg Cells" button to recompute with different parameters. After running SPLIT, switching to "SPLIT Corrected" in the Active Counts toggle displays a separate UMAP computed on the corrected expression matrix (`split_umap_1`/`split_umap_2`).

**Point rendering:**
- Point size: 1–8 px (default 2)
- Opacity: 0.1–1.0 (default 0.85)

**Distance measurement tool:**

Click the **📏** button in the bottom-right corner of the spatial map to activate measure mode. Click any two cells to display the Euclidean distance between them in µm. The result appears next to the button. Click the button again to exit measure mode. Cell-click-to-select is suppressed while the tool is active.

### WebGL view (beta)

The Plotly view rebuilds and resends every cell on each recolour, which becomes slow as datasets grow. The WebGL view (deck.gl) keeps cell positions on the GPU and asks the server only for what changes: one value per cell on a recolour, and boundaries, transcripts and image tiles for the current viewport.

Switch with the **Plotly / WebGL (beta)** selector at the bottom left of the spatial panel, or start with `XENIUM_RENDERER=webgl`. Plotly remains the default. Switching back rebuilds the Plotly figures.

What it draws:

- **Cells** for every colour mode, on Xenium, Baysor and Proseg sources and under an active subset, with a legend (click an entry to hide that category) or colour bar. The Colorbar Range inputs apply without a server round trip.
- **UMAP** in the right-hand panel, from the same cells and colours.
- **Boundaries** for the viewport (zoom in when asked; the limit is 20,000 polygons per view): Xenium cell (cyan) and nucleus (orange) outlines from the bundle's `cells.zarr.zip`, plus Proseg (green) and Baysor (purple) outlines when a run is loaded. Each checkbox works under any segmentation source, as in the Plotly view.
- **Transcripts** for genes chosen under *Transcripts (WebGL view)* in the sidebar, read from the bundle's tiled `transcripts.zarr.zip`. Full resolution with a Q ≥ 20 filter when zoomed in; 10x's pre-subsampled levels (no quality scores) when zoomed out, reported in the status line.
- **Morphology images** read in the browser by HTTP range requests from the pyramidal OME-TIFFs and decoded in web workers, with per-channel colour, the brightness slider as a contrast limit, and opacity. No 5,000 µm viewport limit applies.
- **ROIs**: choose **Lasso** or **Box** next to the renderer selector and draw; the drawn outline itself becomes the ROI (the Plotly view uses the convex hull of the selected cells). Saved ROIs are drawn as labelled outlines.
- **Click a cell** to fill the cell-info panel, from either the spatial view or the UMAP. Hovering shows the cell id and its value. Double-click a legend entry to isolate that category.
- **Ruler**: end points snap to a cell centre when one is hit and are free otherwise.
- **Overlays**: SpatialData ROI and patch outlines, CellNEST edges with their colour and gray-out options, and RCTD weight pies.
- **Multiple samples** side by side, with a dotted separator midway in each gap; boundaries, transcripts, image and UMAP remain primary-sample only, as in the Plotly view. Clicking a cell of any sample fills the cell-info panel; cells of an added sample are marked `[sample N: <folder>]`.
- **PNG export**: the **PNG** button next to Pan / Lasso / Box downloads the current view at 2× with legend and scale bar.

Differences from the Plotly view: the Z-Level selector is disabled (the WebGL view reads the focus image); CellNEST arrowheads scale with edge length and edges have no hover text; gray-out dims every non-participating cell. CellNEST edges have so far been tested in this view with synthetic inputs only.

Both views degrade the same way when the selected colour cannot be drawn: a gene the dataset (or the Baysor/Proseg run) does not have is drawn as transcript counts; a cell-type mode with no labels loaded is drawn as one grey **No annotation** group on the Xenium source and as transcript counts on a Baysor/Proseg run; cell-type labels that match fewer than 5 % of a run's cells (left from another segmentation) are discarded and transcript counts are drawn.

Measured on a Xenium Prime 5K run with 112,551 cells: a recolour transfers about 0.45 MB per view and takes a few milliseconds of server time, against 5–7 MB of JSON per Plotly figure and roughly a second of server time.

Requirements and limits: a browser with WebGL2. When WebGL2 is missing or its context cannot be created, the app switches back to the Plotly view by itself, greys out the WebGL option and says why in the status line next to the selector (reload the page to try again). The Plotly view draws its points with WebGL 1, so with WebGL switched off altogether neither view can draw cells; the status line says so. The feature is new, so switch to the Plotly view if something looks wrong. The browser bundle is prebuilt in `assets/`; Node.js is needed only to rebuild it (see [Developing the WebGL viewer](#developing-the-webgl-viewer)).

**Layout:**
- The plots row and the bottom info bar are separated by a **vertical drag handle** — drag it to resize the panels.
- The bottom info bar can be **fully collapsed** with the ▼ button (the spatial/UMAP plots expand to fill the space). Click ▲ to restore.

### Overlays

Select any combination of overlays from the checklist in the sidebar:

**Xenium Cell/Nuclei Boundaries** — Original Xenium polygon outlines rendered for cells in the current viewport (limit: 3,000 cells). Zoom in to activate. These always show Xenium boundaries regardless of the active segmentation source.
- Xenium cell boundaries: cyan
- Xenium nuclei boundaries: orange
- Baysor/Proseg boundaries: their own separate checkboxes; shown for all reseg cells (no viewport limit)

**Morphology Image** — Fluorescence overlay loaded from OME-TIFF tiles for viewports ≤ 5,000 µm wide (Plotly view; the WebGL view has no such limit). Pyramidal images are read at the resolution level that matches the zoom.
- Z-levels 0–3 (Plotly view only)
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

### Plot Tab (right panel)

The right-side panel contains additional analysis plots, each in its own tab:

| Tab | Content | Controls |
|-----|---------|---------|
| **Cell Type Abundance** | Bar chart of cell type proportions; below it, per-sample proportions and a test between conditions; below that, marker transcripts per mm² counted from the transcript file | W × H; see [Composition between conditions](#composition-between-conditions) and [Segmentation-free abundance](#segmentation-free-abundance) |
| **Co-occurrence** | Spatial co-occurrence score heatmap | Colorbar min/max, W × H; download |
| **Neighborhood Enrichment** | Neighborhood enrichment heatmap: log2 of the fraction of a cell type's neighbours that are of each type over that type's overall frequency (0 = random mixing) | Colorbar min/max, W × H; download |
| **Interaction Graph** | Node-edge graph of cell type contacts | O/E normalize toggle, W × H; download |
| **Chord Diagram** | Chord diagram of cell type contacts | O/E normalize toggle, W × H; download |
| **Niche UMAP** | UMAP (or spatial map) colored by spatial niche | W × H; download |
| **UMAP** | UMAP plot (alternative to sidebar toggle) | W × H; download |
| **CellNEST CCC** | Cell-cell communication from CellNEST output | See below |
| **Differential Expression** | Wilcoxon test between two groups of cells, or pseudobulk DESeq2 between conditions across samples: volcano plot and sortable table | See [Differential expression](#differential-expression) and [Pseudobulk differential expression across samples](#pseudobulk-differential-expression-across-samples) |
| **Gene Sets** | Paste a gene list, score every cell, colour the views by the score | See [Gene-set scores](#gene-set-scores) |
| **Transcript Density** | Binned 2-D density of the transcripts of a gene as a heat layer under the cells | See [Transcript density](#transcript-density) |
| **Spatial Genes** | Moran's I of every gene on a spatial neighbour graph: sortable table | See [Spatially variable genes (Moran's I)](#spatially-variable-genes-morans-i) |
| **Ligand–Receptor** | Ligand–receptor scores between neighbouring cell types with a permutation p-value: dot plot and table | See [Ligand–receptor scores](#ligandreceptor-scores) |

A **download button** (camera icon) appears on hover over each plot; exports as SVG at the current display size.

### Differential expression

**Plots panel → Differential Expression.** Compares the expression of every measured gene between two groups of cells of the active view.

1. Pick **A** and **B**. A group can be:
   - **Current selection** — the last lasso or box drawn on the tissue, in either renderer (all cells whose centroid lies inside the outline). The Save ROI dialog that a lasso opens can be cancelled; the selection stays available here.
   - **Group: …** — a selection saved in this tab (type a name, press **Save current selection as group**). Saved groups are outlines, kept per dataset in the cache directory (`de_groups_{hash}.json`), so they also apply to another segmentation of the same tissue. **Delete group in A** removes one.
   - **ROI: …** — a saved ROI.
   - **Cluster …** — a cluster of the cluster set selected in the sidebar (for a Baysor/Proseg run: its KMeans clusters; with several samples: per sample, `S0:C3`).
   - **Cell type (…): …** — a label of the annotation shown under Color By, else of the first annotation available for the active segmentation.
   - **All other cells** — every cell of the view that is not in the other group.
2. Press **Run**. The job appears in the Jobs panel and can be cancelled. The status line gives the two group sizes, the number of genes tested and the number with FDR < 0.05.
3. **Volcano plot**: log2 fold change (A / B) against −log10 FDR; genes with FDR < 0.05 and |log2FC| ≥ 0.5 are coloured (red: higher in A, blue: higher in B) and the strongest are labelled. **Click a point** (or a table row) to select that gene in the sidebar. **Table**: sortable and filterable per column; **Download CSV** saves all genes with a header line describing the comparison.

**Statistics**

- Expression is `log1p(count / library size × 10,000)`, where the library size is the cell's total count over the measured panel genes in the active counts mode.
- Test: two-sided Wilcoxon rank-sum (Mann–Whitney U) per gene, normal approximation with tie correction and continuity correction. The implementation ranks only the non-zero values of the sparse matrix (all zeros of a gene share one average rank) and is numerically identical to `scipy.stats.mannwhitneyu(method="asymptotic")`; `scanpy.tl.rank_genes_groups(method="wilcoxon", tie_correct=True)` gives the same statistic without the continuity correction.
- `log2FC = log2((mean_A + 1) / (mean_B + 1))` of the group means of the normalised (not log-transformed) values, i.e. counts per 10,000 with a pseudocount of 1 (the Seurat definition). `% A` / `% B`: percentage of cells with at least one count.
- FDR: Benjamini–Hochberg over all tested genes.
- The test respects the **active segmentation**, the **Active Counts** mode (original or SPLIT-corrected) and an active **subset**. **Imputed genes are not tested**: their values are not counts.
- Groups under **10 cells** are refused. Cells that are in both groups are **removed from B**; the status line says how many.

**Caveats — read before quoting a p-value**

- **Every cell is treated as an independent observation.** Cells of one tissue section are not independent, and with hundreds of thousands of cells almost every gene reaches a tiny p-value; rank genes by effect size (log2FC, % expressing) as well.
- **Comparisons across samples or replicates** (several samples loaded side by side, or groups that coincide with samples) are **anticonservative**: the variation between samples is ignored. The proper analysis there is a **pseudobulk** test with samples as replicates: switch **Test** to *Pseudobulk across samples* (next section). With several samples loaded the cell-level test pools their cells and says so in the status line.
- **Groups defined from the same expression data** (clusters) are selected to differ, so their p-values are inflated as well ("double dipping").
- Fold changes of spatial groups also reflect differences in cell-type composition between the regions.

### Samples and conditions

Tests that compare **samples** (pseudobulk differential expression, composition) share one assignment: every loaded sample (the primary one, `S0`, and those added with **+ Add sample**, `S1`, `S2`, …) is in **condition A**, in **condition B**, or **excluded**, optionally with a **pairing / batch** label (a patient, a slide, a run). **Samples & conditions…** in either tab opens the editor; the two conditions can be named. The assignment is kept per sample for the session (it is not written to disk) and is cleared when another dataset is loaded.

- **Donor (pairing)**: the individual a sample comes from. Samples of one donor are not independent: within a condition they are combined into one unit (averaged; summed for pseudobulk). When every included sample has a donor and at least 2 donors occur in both conditions, the tests are **paired** (`~ donor + condition`; exact sign-flip test).
- **Batch** (a slide, a run): an optional blocking term of the model-based tests when no donors are given (`~ batch + condition`). A label must be given to every included sample or to none and must not be confounded with the condition; such a design is refused with the reason.
- The design is **saved in the cache directory** (`design_<hash>.json`, keyed by the set of loaded samples) and restored when the same samples are loaded again, also after Load Sample.

**Exact sample-level tests** (used by the composition test's *exact* option and by stratified co-localisation). Paired: the sign-flip test of the per-donor differences — all 2^n sign vectors are enumerated up to 16 donors, 20,000 random vectors with a fixed seed above. Unpaired: the label-permutation test of the difference of means — all C(n, n_A) relabellings up to 200,000, Monte Carlo above. A null that can be enumerated is never sampled. Every result states the **unit and the number of independent units** ("n = 6 samples (3 donors, paired)"), the **smallest achievable p** (2 / 2^n for n pairs: 0.25 with 3, 0.031 with 6), and a warning when that floor × the number of tests cannot reach 0.05 after Benjamini–Hochberg. Every p-value in the Differential Expression, composition, Spatial Genes, Ligand–Receptor and co-localisation tabs and CSVs now names its unit, e.g. "n = 112,551 cells, treated as independent".

**Cell types of added samples.** Annotation runs on the primary sample only. An added sample gets cell types from (1) the annotation cache, if that sample was annotated earlier as the primary dataset with the same cache directory (per method, the most recent Xenium-segmentation run), or (2) the `cell_type_<method>` columns of a store written by **Save as SpatialData**. A sample without labels for the chosen method is left out of cell-type populations and composition tests, and named. *Behaviour change:* added samples that have such labels are now coloured by cell type in both renderers (they were always "Unknown").

### Pseudobulk differential expression across samples

**Plots panel → Differential Expression → Test: Pseudobulk across samples.** The samples, not the cells, are the observations.

1. Assign the samples to the two conditions (**Samples & conditions…**).
2. Pick the **cells** whose counts are summed: all cells, a cell type, a cluster, a ROI class (all its ROIs — draw one per sample), a single ROI, a saved group or the current selection.
3. **Run**. The result uses the same volcano plot, table and **Download CSV** as the cell-level test; title, status line and CSV header name the method, the design and the samples. A table under the status line lists, per sample, the cells summed, the library size and the size factor.

- **Counts**: the raw integer counts of the chosen cells are summed per sample over the measured panel genes. Imputed genes are never used. **The original counts are always used, also when Active Counts is "SPLIT Corrected"** (the status line says so): SPLIT-corrected counts are modified, non-integer values, which a count model must not be given.
- **Paired design**: with donors on every included sample the design is `~ donor + condition` and the samples of one donor within a condition are summed into one pseudobulk column first (named in the status line).
- **Volcano**: points are coloured by **FDR < 0.05 alone**, and the x axis is the unshrunken (MLE) log2FC by default; **Volcano x axis** switches to the shrunken estimate. (Before, the shrunken estimate was plotted and a significant gene shrunk to ~0 was drawn as not significant.)
- **Model**: **DESeq2** through rpy2 — design `~ condition`, or `~ batch + condition` with batch labels; Wald test; independent filtering at alpha 0.05; log2 fold change **A / B**. The table gives the unshrunken (MLE) log2FC and, with **Shrink log2FC** ticked, the **apeglm**-shrunken one (DESeq2's `normal` prior when apeglm is not installed; the column and the volcano axis name the estimator, or say "unshrunken"). If DESeq2's dispersion trend cannot be fitted the gene-wise estimates are used and the status line says so.
- **Without DESeq2**: **edgeR**'s quasi-likelihood F-test is used when edgeR is installed, and the result is labelled as such. With neither package the run is refused with the install hint (`BiocManager::install(c("DESeq2", "apeglm"))`). There is no fallback to a t-test.
- **Refusals and warnings**: a condition with **fewer than 2 samples is refused**; with fewer than 3 the status line warns that the variance estimates are unreliable. A sample contributing fewer than **Min. cells / sample** (default 10) is dropped and named. Genes with a total count below **Min. total count / gene** (default 10) are not tested and counted in the status line, as are genes DESeq2 leaves without an adjusted p-value (they are not drawn in the volcano).
- A Baysor / Proseg run is a single sample: pseudobulk needs the Xenium segmentation with several samples loaded.
- **Validation**: on fabricated six-sample data and on six sub-bundles of the Prime 5K skin section the log2FC, p-values, adjusted p-values and apeglm-shrunken log2FC equal those of DESeq2 run directly in R on the same matrix (largest relative difference 3 × 10⁻¹³; 4,226 genes); likewise for the batch design and for the edgeR path.

**Caveats**

- **Replicates must be biological replicates.** Regions of one tissue section, or serial sections of one block, are pseudo-replicates: their dispersion is far too small and the p-values mean nothing.
- With 2–3 samples per condition power is low and a single odd sample decides the result; look at the per-sample table.
- **Cluster ids** are per sample: "cluster 3 in every sample" is the same population only if the samples were clustered together. Prefer cell types.
- Fold changes of a population defined by a ROI also reflect its cell-type composition.

### Proximity differential expression

**Plots panel → Differential Expression → Test: Proximity to a cell type.** "Which genes of the cells **T** change near cells of the anchor type **X**?" Pick where the cell types come from (an annotation, or the clusters), T and X, then **Run**. It works on one sample; with several it tests across them.

- **Per sample**: log1p(count / library size × 10,000) of every T cell is residualised (ordinary least squares) on log1p(library size) and on decile bins of the local fraction of the **density-covariate** type (cells within 50 µm; default: the type that contributes the most transcripts; "none" drops the term). d is the distance to the nearest X cell of the same sample.
- **Effect** (table column *Near − far*, the volcano's x axis, in log2 units): mean residual of the T cells with d ≤ **Near** (default 25 µm) minus that of the T cells with d ≥ **Far** (default 100 µm). **Slope**: OLS slope of the residual on −log(d + 1). A sample needs ≥ 100 T cells, ≥ 5 X cells, ≥ 5 near and ≥ 50 far cells; genes detected (≥ 1 transcript) in fewer than 2 % of the T cells are not tested.
- **Matched random anchors** (default 99 sets): as many non-X cells as there are X cells, drawn within the same compartment × edge-distance stratum (the columns of the Confound Controls tab; the defaults are computed when they are missing, and the status line says so), put through the same computation. They answer "what does any set of cells sitting where X sits produce?".
- **p-values**: with ≥ 3 independent units (donors when every included sample has one, else samples; *Samples & conditions…*) the **exact sign-flip test** of the per-unit effects — the status line gives n and the smallest achievable p. With fewer units (a **single sample**) p is the share of random anchor sets at least as far from their mean as the observed effect; the status line marks this as a single-sample result that does not generalise.
- **Empirical FDR** (column *Emp. FDR*, the volcano's y axis): the number of genes the random sets yield at a gene's threshold divided by the number observed (never 0, made monotone). In the cohort mode the random sets' sign-flip p-values are used, in the single-sample mode |z| against the random sets with every random set scored against the others.
- When fewer than half of the random sets leave enough near and far cells in every unit (anchors as numerous as an abundant type cover the tissue), p and FDR are taken from the **slope** instead and the status line says so; if that fails too, the cohort mode falls back to Benjamini–Hochberg and the single-sample mode refuses.
- **Specific**: not reproduced by up to 3 other anchor types of similar abundance (within 3×): in the cohort mode same sign in ≥ 5/7 of the units, in the single-sample mode same direction and beyond the random sets at p ≤ 0.05.
- **Leak gate**: the gate of the other tests is run for T near X against T far from X, and genes that X owns and T does not (Confound Controls → gene ownership) are flagged as well: next to X, T cells pick up X transcripts. Flagged genes are open orange diamonds in the volcano; *X ÷ T rate* shows how much more X expresses a gene. Genes with X as top owner that T also carries are counted in the status line.
- **Between conditions**: with a design that gives ≥ 3 units per condition the per-sample effects are also tested A − B (columns *A − B*, *p (A vs B)*, BH FDR).
- **Fit exp(−d / L)** (optional): per gene and sample the length L (grid 2–600 µm) for which residual ~ a + b·exp(−d / L) fits best; the table gives the median L and R². R² is small for most genes.
- Honours the segmentation, the subset and Active Counts. **Timing**: 112,551-cell Prime 5K skin section, 99 random sets: 10,816 fibroblasts near 263 mast cells 4.8 s; 62,817 T cells near 3,427 macrophages 10.5 s headless, 7.8 s in the app.
- **What it is not**: a gradient with distance is an association with position. The residualisation and the random anchors remove depth, a dominant neighbour's density and generic location; they do not remove everything that co-varies with being near X.
- Reimplemented from an earlier spatial study by the author (checked against that code on fabricated data: residuals, near − far and slope to 10⁻⁸, sign-flip p-values and random draws identical). Changed here: sufficient statistics on sparse matrices instead of dense residuals; strata of the random anchors are compartment × edge bin instead of deciles of one cell type's local fraction; per-sample effects are averaged per unit; the FDR is evaluated at every gene's own threshold instead of on a fixed grid; a single-sample mode exists.

### Contact-group pseudobulk differential expression

**Plots panel → Differential Expression → Test: Contact groups (pseudobulk).** "Do T cells in contact with X differ from T cells that are not?" Needs several samples.

- **Groups**: within each sample, T cells with an X cell within the **contact radius** (default 50 µm, centroid to centroid) against T cells without.
- **Matching**: the comparator cells are weighted so that their mix of compartment × edge-distance bin × density quintile equals that of the contact cells; contact cells in a stratum without comparator are left out (counted in the status line).
- **Model**: pseudobulk = Σ weight × counts per sample × group, rounded to whole counts, fitted with `~ sample + group` through the same R bridge as the pseudobulk mode (DESeq2, else edgeR, else refused with the install hint). With donors on every used sample, samples of one donor are summed into one block. The unit of the p-values is the sample (block); the status line states n.
- **Refusal**: fewer than 2 samples with at least **Min. cells / sample / group** (default 20) in both groups → no test, with the counts per sample and a pointer to the proximity mode. A Baysor / Proseg run is one sample.
- **Label shuffle** (default 199, 0 to skip): contact / comparator labels are permuted within sample × stratum, weights and fit are redone, and the number of genes passing (FDR < 0.05, |log2FC| > 0.5, not flagged) is compared with the observed number: P = (1 + shuffles with at least as many) / (1 + shuffles). Every shuffle is a model fit (about 4–6 s for 4,200 genes × 12 pseudobulks), so 199 take a quarter of an hour; the job shows progress and can be cancelled.
- **Leak gate** per sample (contact = arm A); genes X owns and T does not are flagged too and never count as passing.
- Uses the active counts mode (SPLIT-corrected counts are summed and rounded as well).
- Reimplemented from an earlier spatial study by the author (the matching weights equal that code's on fabricated data). Changed here: one generic comparison instead of fixed cell states; weighted sums are rounded because DESeq2 needs integers (the original fitted fractional sums with edgeR); the shuffle null counts passing genes as there.

### Composition between conditions

**Plots panel → Cell Type Abundance → Composition between conditions.** Tests whether the **proportion** of each cell type (or cluster, or spatial niche) differs between conditions A and B, with the same sample assignment.

- **Compute** shows, for every sample that has the labels, the proportion of each category: a plot with **one point per sample**, grouped and coloured by condition (bars: condition means; `*`: FDR < 0.05), a table (test results and each sample's proportion) and **Download CSV** (adds the cell counts).
- **Method (propeller-style, Phipson et al. 2022)**: per-sample proportions are **logit**-transformed (0.5 pseudocount per category, as `speckle`) or **arcsine-square-root**-transformed; a linear model (`~ condition`, or `~ batch + condition`) is fitted per category and tested with a **moderated t-test** — limma-style empirical Bayes shrinkage of the category variances towards a common prior (non-robust, no trend), implemented in Python. **Welch's t-test** on the same transformed values is given beside it (it ignores batch labels). Benjamini–Hochberg FDR over the categories.
- **With fewer than 3 samples in a condition no test is run** — the tab says so and shows the proportions only. With one sample its proportions are shown.
- **Test: exact (sample-level)** tests the same transformed proportions without a variance model: paired sign-flip over donors when the design is paired, otherwise label permutation over the independent units (see *Samples and conditions*). The moderated t-test does not use donor labels and says so when they are set.
- **Validation**: Welch's t, its degrees of freedom and p-value equal `scipy.stats.ttest_ind(equal_var=False)`; the moderated t, its p-value, the prior variance and the prior and total degrees of freedom equal `limma::lmFit` + `eBayes(robust = FALSE)` on the same transformed proportions (relative difference < 10⁻⁵), with and without a batch term. `speckle::propeller` itself was not installed: its default uses `robust = TRUE`, which can differ when one category's variance is an outlier. A quasi-binomial / beta-binomial GLM is not implemented.

**Caveats**

- **Proportions are compositional**: they sum to 1 in each sample, so a real expansion of one type lowers the proportions of all the others. The tests of the categories are not independent, and a "decrease" may be the shadow of another type's increase.
- With 3–5 samples per condition only large shifts are detectable, and the empirical Bayes prior is estimated from as many variances as there are categories (a dozen): treat the moderated p-values as approximate and compare with Welch's.
- Proportions depend on the tissue area sampled and on the segmentation; cells without a label are left out of the denominator.
- Cluster and niche ids are comparable across samples only when the samples were clustered together (niches are computed for the primary sample only).

### Segmentation-free abundance

**Plots panel → Cell Type Abundance → Segmentation-free abundance.** Answers "is there more of
this cell type's signal per area of tissue?" without asking the segmentation: the **marker
transcripts of a cell type per mm² of tissue**, counted straight from the transcript file
(quality Q ≥ 20, gene features only - the filter of the cell-feature matrix), next to the
density of the **segmented** cells of the same type on the same area. A cell that the
segmentation merged into a neighbour, split, or left unlabelled still contributes its
transcripts here.

| Control | What it does |
|---------|--------------|
| **Cell types from / Type** | The annotation (or the active cluster set) and the type whose markers are counted. The segmented cell density is computed for this type. |
| **Markers to derive** | Number of marker genes taken from the annotation (default 5): the type's most specific genes, by the rule of the spillover score (≥ 50 counts per million in the type and ≥ 10× every other type; the top genes by ratio when fewer than 3 pass - the status then says the markers are **not exclusive**). |
| **Marker genes (text box)** | Optional: your own markers; then no annotation is needed (and without a type there is no cell density). Names not on the panel are reported. |
| **Tile sizes (µm)** | The area-bin sweep, default 25, 50, 100, 200 (multiples of 5 µm). |
| **Tissue area from** | *cells*: a tile with a cell centroid of the active segmentation and subset is tissue (default). *all transcripts*: a tile with at least one gene transcript is tissue - neither numerator nor denominator then uses the segmentation; needs one pass over all transcripts (cached; the same grid as Transcript Density → all genes), and a cell subset is not applied. |
| **Fill holes ≤ (tiles)** | Interior holes of the mask of at most this many tiles count as tissue (the rule of the tissue compartments; default 0: none). |
| **Redistributions** | Size of the null of the concentration statistics (default 200; 0: none). |

**What is computed**, per sample and tile size:

- **Area** = tissue tiles × tile size². The area of the same tissue **grows with the tile
  size** (coarse tiles cover gaps and the margin), so every density is given along the sweep.
  A difference between samples that shrinks, vanishes or changes sign along the sweep is a
  property of the area estimate, not of the tissue; the status line says so when the sign of a
  difference between conditions changes.
- **Tx / mm²** = marker transcripts on the tissue tiles ÷ area. **Tx / 10k tx** = per 10,000
  gene transcripts on the same tiles (shown when the all-transcripts grid is in the cache).
- **Cells / mm²** = segmented cells of the type on the same tiles ÷ the same area; **Tx / cell**
  = marker transcripts per segmented cell of the type; **In type's cells / In any cell** = the
  share of the marker transcripts on the tissue that the count matrix (active counts mode)
  assigns to cells of the type / to any cell. A low first share with a high second one means
  the type's markers sit in cells called something else.
- **Concentration**: Gini coefficient of the marker count over the tissue tiles, and the share
  of the count in the top 10 % of tiles, each reported as **observed minus the mean of a null**
  that redistributes the same number of transcripts over the same tiles with equal
  probability. A sparse count has a high Gini by itself; only the excess is clustering.
  p = (1 + redistributions with a Gini ≥ observed) / (1 + redistributions), one-sided.
  **Unit of this p-value: the tiles of one sample** (n = number of tissue tiles, stated in the
  status line); it describes that section and does not generalise.
- **Between conditions**: with a [sample design](#samples-and-conditions) of at least 3
  independent units per condition (3 pairs, when paired), the exact sample-level test (paired
  sign-flip or label permutation) of A − B for transcripts per mm², cells per mm², transcripts
  per 10k and the Gini excess, at every tile size. **Unit: samples / donors**, with n and the
  smallest achievable p stated. With fewer units, or one sample, the rows are descriptive and
  the status line says so.

The plot shows the three quantities against the tile size, one line per sample; the table and
the CSV hold every row, the CSV also the transcripts of each marker gene (one gene can carry
the whole set - the status line names it when one gene gives more than 60 %).

The per-gene transcript grids are read in **one pass** over `transcripts.parquet` for all
markers and cached with the Transcript Density grids (`density_<dataset>/`), per sample.
Every sample is binned on its own grid against its own tissue mask. A sample loaded from a
`.zarr` has no transcript file: it is left out and named.

**Limits.** Transcript density does not tell cell number from expression per cell: a rise can
be more cells or more transcripts per cell. Markers that are not exclusive to the type count
other cells too. The null of the concentration statistics gives every tissue tile the same
probability, so a tissue that is unevenly dense in everything shows some excess for every
gene; compare with a gene that should be uniform. With the area from cells, the denominator
still depends on where the segmentation found cells (but not on what it called them).

### Gene-set scores

**Plots panel → Gene Sets.** Paste gene names (one per line, or separated by commas / semicolons / spaces), give the set a name and press **Save & score**. The set is saved per dataset in the cache directory (`genesets_{hash}.json`) and appears under **Saved** in the tab and as **Gene set: <name>** under Color By, which is switched to it when the score is ready. **Delete** removes the selected set.

- Names are matched to the measured panel genes exactly, else ignoring case. Genes that are **not found are listed** in the status line; a set with **fewer than 2** genes found is refused. Imputed genes are not used.
- **Score** per cell = mean of the z-scored log-normalised expression of the set genes − mean of the z-scored expression of a **control set**. Expression is `log1p(count / library size × 10,000)`. As in Seurat `AddModuleScore` / `scanpy.tl.score_genes`, all measured genes are ranked by mean expression and cut into 25 bins, and for every bin that holds a set gene 50 control genes are drawn from that bin (fixed seed, set genes excluded), so the control matches the set's expression level. Unlike those two, each gene is first standardised over the cells of the view (mean 0, sd 1), so that the highest-expressed genes of a set do not dominate its mean; genes without any variance are left out and reported.
- A score of 0 means "as expected for genes of this expression level"; the views use a **diverging colour scale centred at 0** (blue below, red above) in the spatial view and the UMAP of both renderers. The sidebar **Min / Max** inputs override either end.
- The score is computed for the **active view** — segmentation, counts mode, subset and, for the Xenium source, all loaded samples together (genes are standardised over all of them) — and recomputed when one of these changes (a fraction of a second to a few seconds).
- Agreement with scanpy (Prime 5K skin, 112,551 cells): identical scores when both use the same control genes (r = 1.000 without z-scoring); with each implementation's own random draw of control genes r ≈ 0.91–0.99 depending on the set, and r ≈ 0.85–0.96 for the z-scored default.

### Transcript density

**Plots panel → Transcript Density.** Shows where the **transcripts** of a gene (or of all genes) lie, independent of any cell segmentation: a 2-D grid of transcript counts drawn as a heat layer **under the cells**, in both renderers.

1. Pick a gene, or **All genes**, and press **Compute**. The app reads `transcripts.parquet` once for that gene and for every loaded sample (a registered, cancellable job with progress) and stores the grids in the cache directory; afterwards the gene is available immediately.
2. **Bin (µm)**: 5, 10, 20 or 50. **Show heat layer**, **Opacity** and **Colour max** control the layer; lowering the sidebar's point opacity makes it easier to see under the cells. The colour bar gives transcripts per bin (0 – the 99.5th percentile of the non-empty bins unless a maximum is entered); empty bins are transparent.

- Transcripts are filtered like the cell-feature matrix: **Q ≥ 20, gene features only** (no control probes or codewords, blank / unassigned / deprecated codewords). The sum of a gene's grid equals the number of its transcripts that pass this filter — including transcripts outside every cell, so it can exceed the gene's total in the count matrix.
- Bin *i* covers `[i × bin, (i + 1) × bin)` µm from the slide origin in x and in y. Grids are stored at 5 µm; the coarser bin sizes are exact sums of those bins.
- The parquet file is read one row group at a time with the filter pushed down and two columns decoded (bundles without `transcripts.parquet`: the full-resolution tiles of `transcripts.zarr.zip`), so memory does not grow with the number of transcripts. On the Prime 5K skin bundle (74 M transcripts) a gene takes about 2 s and all genes about 3 s.
- Only the bins of the current viewport are sent to the browser. When the viewport holds more bins than can be drawn (zoomed out on a large slide at a small bin size) neighbouring bins are **averaged** (not summed), so the values stay "transcripts per bin" and the colours do not change with the zoom; zooming in shows the bins themselves.
- The layer does not depend on the segmentation, the counts mode or a cell subset. It needs the Xenium output directory (not available for a dataset opened from a bare `.zarr`).
- **Several samples**: every sample has its own grid, cached in its own directory (`density_{hash of that sample}/`, so a grid is found again whether the sample is loaded as primary or added). Each grid is drawn at its sample's position in the combined view, in both renderers, on **one colour scale** (the 99.5th percentile of the non-empty bins of all samples together) and with one colour bar. The grids are not resampled: bin edges stay multiples of the bin size in each sample's own coordinates. A sample added after Compute has no grid until Compute is pressed again, and a sample loaded from a `.zarr` has none; the hint line names both. *Behaviour change:* the layer used to show the primary sample only.

### Spatially variable genes (Moran's I)

**Plots panel → Spatial Genes → Run.** Ranks the measured genes by spatial autocorrelation.

- **Graph**: the **k nearest neighbours** of every cell centroid (k = 6 by default), built for this analysis; weights are **row-standardised** (each neighbour weighs 1/k; the graph is not symmetrised).
- **Statistic**: Moran's I of `log1p(count / library size × 10,000)` per gene. **z-score and two-sided p-value** from the analytic mean (−1/(n−1)) and variance of I under the **normality assumption** (Cliff & Ord), **Benjamini–Hochberg FDR** over the tested genes of a sample. Genes not detected in any cell are not tested.
- Computed for the **active view** (segmentation, counts mode, subset) and **per sample** when several are loaded (each on its own graph; the table has a Sample column). A registered, cancellable job with progress.
- The table is sorted by I and can be filtered; a click on a row selects the gene in the sidebar; **Download CSV** writes all genes.
- **Cost**: the count matrix stays sparse and genes are processed in dense blocks of 64 MB. Prime 5K skin (112,551 cells × 5,006 genes, 33 M non-zeros): **5.5 s**, about 0.3 GB above the loaded dataset.
- **Validation**: I, z and p equal the textbook dense formulas on fabricated data and on the fixture bundle, and `spdep::moran.test(randomisation = FALSE)` on a k-nearest-neighbour `listw` (relative difference < 2 × 10⁻⁶, the precision of the float32 size factors). squidpy / esda were not installed and were not compared.

**Caveats**

- With tens of thousands of cells **almost every expressed gene is "significant"** (4,051 of 5,006 on the skin section): rank by I, not by the p-value. The normality assumption is rough for sparse counts.
- I measures autocorrelation at the scale of the k neighbours. A marker of a cell type that sits in clusters scores high for that reason alone; it does not separate "spatial pattern within a cell type" from "cell-type pattern".
- Segmentation spill-over between touching cells raises I for highly expressed genes.

### Ligand–receptor scores

**Plots panel → Ligand–Receptor.** Scores ligand–receptor pairs between cell types that are neighbours in the tissue, without CellNEST.

- **Pair list**: **no database is bundled and none is downloaded.** Upload a CSV (or paste lines) of `ligand,receptor` gene symbols — a header row naming a *ligand…* and a *receptor…* column picks those columns of a wider table — or press **Use pairs of loaded CellNEST results** to take the pairs present in results you loaded in the CellNEST CCC tab. (The files CellNEST's own code caches are a signalling-PPI and a TF-target table, not a ligand–receptor list, and are not used.)
- **Only pairs with both genes measured on the panel are scored**; the status line gives the number scored and not scored. Multi-subunit complexes are not modelled: each line is one ligand gene and one receptor gene.
- **Groups**: a cell-type annotation or the active cluster set. Cells of all loaded samples are pooled; the neighbour graph and the permutations stay within each sample. Unlabelled cells are left out; a group needs 10 cells.
- **Score** (CellPhoneDB-style) for sender type S, receiver type R and pair L → R: the mean of (mean log-normalised L in the cells of S) and (mean log-normalised R in the cells of R).
- **Spatial restriction**: a directed type pair is scored only if the two types are **neighbours**: at least 20 adjacent cell pairs in the k-nearest-neighbour graph of the centroids (k = 6 by default; a pair of cells counts once). The table gives the number of adjacent pairs. A combination is tested only if L is detected in ≥ 10 % of the cells of S and R in ≥ 10 % of the cells of R.
- **p-value**: the cell-type labels are permuted (fixed seed 0; 200 permutations by default, up to 10,000) and the score recomputed; p = (1 + permutations with a score ≥ observed) / (1 + permutations). **Benjamini–Hochberg FDR** over all tested combinations. A registered job: progress per permutation, **Cancel** stops it at once.
- **Output**: a dot plot (sender → receiver × pair; colour = score, size = −log10 FDR) of the pairs with the smallest FDR, a sortable table and **Download CSV**.
- **Cost** (Prime 5K skin, 112,551 cells, 13 types): 60-pair list (48 on the panel) 0.9 s with 200 permutations and 3 s with 1,000; 1,500 pairs 17 s with 200 permutations.
- **Validation**: on fabricated tissue with a planted interaction (ligand in type A, receptor in the adjacent type B) the pair is recovered for A → B only, a uniformly expressed pair is not significant anywhere, and a type pair that does not touch is not scored; the permutation p-value equals a plain re-implementation; results are identical for a fixed seed.

**Caveats — this is not evidence of signalling**

- A significant score says that the ligand is expressed **specifically** in one type and the receptor specifically in a neighbouring type, relative to shuffled labels. It does not show that the protein is made, secreted, bound or signalling. Marker genes of two adjacent types are "significant" whatever their function: on the skin section 1,924 of 3,458 tested combinations of **random gene pairs** had FDR < 0.05.
- The smallest possible p-value is 1 / (1 + permutations) (0.005 at 200): with few permutations many combinations tie at that floor.
- The spatial information enters only through which type pairs are scored; the means are over all cells of a type, not only those at the interface.
- Segmentation spill-over between touching cells of two types inflates both means.
- A targeted panel holds a fraction of any ligand–receptor resource; absence of a pair from the result means "not measured" as often as "not expressed".

### Ligand–receptor exposure decomposition

The second mode of the **Ligand–Receptor** tab (**Mode → Exposure decomposition**; it uses the same pair list and cell grouping). It answers a different question from the scores: not "is this pair expressed in neighbouring types" but **why the exposure of a receiver type to a ligand differs** between two conditions (or how it is made up in one): expression, contact or geometry. It is a general version of an analysis from an earlier spatial study by the author; the core sums are ported unchanged together with their brute-force unit test.

For sender type S, receiver type T, ligand L and receptor R, with the contact set C = (receiver *i*, sender *j*) pairs whose centroids lie within the contact radius **in the same sample**, and *s* = a cell's total count over the panel:

| Term | Definition | Reads as |
|------|-----------|----------|
| K, contacts per receiver | \|C\| / N_T | contact: how many S cells a T cell touches |
| R, receptor rate | Σ_T x_R / Σ_T s | expression in the receiver type |
| L, ligand rate | Σ_S x_L / Σ_S s | expression in the sender type |
| γ, tile co-location | P_exp / (R · L), with P_exp = Σ_C s_i s_j R̃_i L̃_j / Σ_C s_i s_j and R̃, L̃ the rate of the cell's type in its tile | geometry at the scale of the tile: do ligand-rich senders sit where receptor-rich receivers sit |
| π, contact preference | P / P_exp, with P = Σ_C x_iR x_jL / Σ_C s_i s_j | geometry below the tile: are the cells in contact the ones carrying the two genes |

**Exposure E = K · P = K · R · L · γ · π, exactly** (the log2 form is asserted on every run with two conditions). Rates are Σ counts / Σ depth, the rule used elsewhere in the app, so fractional SPLIT-corrected counts need no rounding and a section-wide capture factor cancels in γ and π; the **Rates: detection** option replaces the counts by 1[count ≥ 1] and the depth by 1.

- **Controls**: contact radius (default 30 µm), tile side (250 µm; 0 = the whole sample, which moves all geometry into π), ownership share (0.2), transcript floor (100), minimum cells per type (50), rates. An interaction is kept when the sender owns the ligand and the receiver owns the receptor (pooled rate ≥ the share of the highest type's rate), both genes reach the floor in their type **in every unit**, and both types have the minimum number of cells and are in contact in every unit. The status line counts what each filter removed. At most the 40 most abundant groups are used.
- **Two conditions** (*Samples & conditions…*, the editor of the pseudobulk test): the sums of a donor's samples within a condition are pooled (ratios of pooled sums are exact), then per unit the log2 ratio A / B of each term; **log2 E ratio = Σ of the five log2 term ratios**. Paired design (every sample has a donor, ≥ 3 donors in both conditions): exact sign-flip test of the per-donor log2 ratios. Otherwise, with ≥ 3 units per condition: exact label permutation of mean(log2 term in A) − mean(log2 term in B) (all relabellings up to 20,000, else 5,000 random ones with a fixed seed, and said so). Benjamini–Hochberg FDR over the interactions; the status line states the unit and n, the smallest achievable p, and warns when no test can reach 0.05. Per sender → receiver pair the interactions are averaged per unit and tested the same way (with the exact sign-flip 95 % interval when there are enough donors for one), and per term an **omnibus** over the between-type pairs: Σ t² against every sign (or label) assignment of the units, plus the shift shared by all pairs.
- **One condition, one sample, or fewer than 3 units per condition**: no p-values, and the status says why. One condition shows the terms themselves (pooled over the samples); two conditions with too few units show the log2 ratios of the pooled conditions.
- **Observed / expected variant** (columns Obs., Exp., O/E): observed = share of receivers that detect the receptor and have at least one ligand-positive sender in contact; expected = the same with ligand positivity assigned at random among the senders, 1 − (1 − p_L)^k for a receiver with k sender contacts; O/E = (obs + 0.001) / (exp + 0.001). With a design, its log2 ratio A / B is tested like the terms.
- **Plot**: stacked diverging bars, one row per ligand → receptor | sender → receiver (or per type pair: **Show**). With two conditions the five bars are the term contributions and the diamond their sum, the exposure change (* p < 0.05); with one condition the bars are log2 γ and log2 π. **Table** and **Download CSV** (all terms, per-unit statistics, the type-pair table, the omnibus and the sample list in the header).
- A registered job with progress and Cancel; honours the segmentation source, the subset, the counts mode and several samples (a contact never joins cells of two samples).

Caveats:

- The decomposition says **why an exposure differs** (expression vs contact vs geometry). Like every expression-based score it is **not evidence of signalling**.
- π rises when transcripts spill between touching cells: a ligand transcript assigned to the receiver's neighbour is exactly a "preferential contact". The ownership filter removes the pairs where that is the whole signal; read π beside the leak gate and the spillover score.
- γ depends on the tile size; rerun with another tile (or 0) to see how much of the geometry is regional.
- The floor applies to the shallowest unit, so a small sample removes interactions for all.


### Confound controls

The features of this and the next three sections (and the exact sample-level tests of **Samples and conditions**) are general versions of controls developed in an earlier spatial study by the author, where each was written after a specific false result: depth producing correlations, neighbour transcripts read as a cell type's programme, a shared niche read as an interaction, cells counted as replicates. They are reimplemented here without any study-specific gene list, cell-type name or threshold: every such value is a parameter with a default. All of them work on the active segmentation (Xenium, Baysor, Proseg), the current subset and every loaded sample; all neighbour searches are made **within one sample**.

**Plots panel → Confound Controls.** Pick where the cell types come from (an annotation or the cluster set) at the top.

#### Tissue compartments

Per sample: **edge distance** (the cell positions are rasterised into bins, default 50 µm; occupied bins are tissue; optional morphological closing; interior holes of at most 4 bins are filled; a Euclidean distance transform gives each bin its distance to the nearest empty bin, and a cell inherits its bin's value), **local density** (other cells within R, default 100 µm) and a **dense core** of a dominant cell type (default: the most abundant; ≥ 10 neighbours within R, ≥ 30 % of them of that type, > 200 µm from the edge). Compartment = `core` / `edge` (not core, within the edge threshold) / `other`. Every threshold is an input.

- The columns appear under **Color By** as *Control: Edge distance / Local density / Dominant-type fraction / Dense core* (Plotly and WebGL views, sequential colour scale), are the strata of the co-localisation test, and are written by **Save as SpatialData** as `ctl_edge_distance_um`, `ctl_local_density`, `ctl_dominant_fraction`, `ctl_core`, `ctl_compartment`.
- They are stored per sample and segmentation by cell id and are not recomputed automatically: **recompute after changing the subset or the segmentation**. They are dropped on Load Sample.
- Resolution is one bin: cells in an edge bin all get one bin size. A gap in the tissue wider than the hole limit is an edge.
- Why: section margins are where sectioning damage and segmentation noise concentrate and where some cell types are simply rare. A cell type that prefers the margin looks depleted of everything that is rare there.

#### Spillover score and gene ownership

**Spillover score** of a cell for a *source* cell type (default: the type contributing the most transcripts) = the share of the cell's counts in the source's marker genes ÷ the median of that share inside source cells. 1 means "as much source-marker signal as a real source cell"; flags at ≥ 0.10 (low), ≥ 0.25 (moderate), ≥ 0.50 (high). Markers are derived from the annotation (genes at ≥ 50 counts per million in the source and ≥ 10× every other type; when fewer than 3 pass, the top genes by ratio, labelled **FALLBACK** — the score is then an upper bound) or typed in.

- **Built-in validity check**, shown in the status line and the plot title: if the score measures transcripts crossing segment boundaries, the marker share of non-source cells must **fall with distance** to the nearest source cell (mean within 30 µm ≥ 2× the mean beyond 100 µm) and **rise with the fraction of source cells within 50 µm** (Spearman ρ > 0 and top third above bottom third). When a check fails the status says the score must not be used as a contamination mask: it is then measuring shared biology or an annotation that splits one population.
- When the markers are so sparse that the median share inside source cells is 0, the mean is the reference, the status says so and gives the share of source cells carrying a marker transcript; the per-cell score is then coarse.
- The status also gives the flag rate per **depth quintile**. A strong trend with depth means the score follows cell size or depth as well.
- **Color By → Control: Spillover score**; the columns `ctl_spillover`, `ctl_spillover_flag`, `ctl_source_distance_um` are exported.
- **Gene ownership** (table + CSV): per gene the cell type with the highest rate (Σ counts ÷ Σ depth of the type's cells) and every type at ≥ 20 % of it; genes with fewer than 100 transcripts get no owner. A "programme" read in a cell type that owns few of its genes is largely a readout of its neighbours.

#### Leak gate in Differential Expression

With **Gene ownership + leak gate** ticked (default) and a cell-type annotation present, both DE modes add `owner`, `leak_fraction` and `leak_flag` to the table and CSV (the CSV also has `owners`, `leak_share_a`, `leak_share_b`, `leak_change_frac`, `top_leak_type`); flagged genes are open orange diamonds on the volcano and the status line says how many genes, and how many of the significant ones, were flagged.

- **Model**: leakage adds leak(L) × (profile of neighbouring type L) to the focal cells. Predicted leaked signal of a gene = Σ_L leak(L) × its counts per million in L. **Flag** when the predicted leak is more than half of the observed signal in either group, or when the leak alone predicts at least half of the observed log fold change. In the pseudobulk mode every sample is a section with its own leak and neighbour profiles and the groups are averaged over sections; in the cell-level mode each group is one pooled section.
- **Measuring leak(L)** differs from the original, which chose marker genes the focal type does not express in a paired single-nucleus reference. Without a reference: a gene's focal ÷ L ratio is the leak plus the focal type's own expression, so the leak cannot exceed the lowest ratios; it is the median of the 25 lowest ratios among L's own genes (≥ 50 CPM in L, ≥ 10× every third type). A neighbour with fewer than 10 such genes, or whose own genes the focal type carries at ≥ 50 % of their level, is **not modelled and named** — its contribution is missing from the prediction.
- It needs group A to be one cell type (≥ 80 % of its labelled cells). When B is the same type both groups are modelled; otherwise only A (e.g. "type vs all other cells").
- **Limits**: the estimate leans low (the lowest ratios are selected in the same data), so the gate errs towards not flagging; with an annotation that does not separate the types (sub-types of one lineage, a classifier that calls most cells one label) few neighbours are modelled and the gate says little. A flag means "check the owner and the neighbours", not "wrong".

### Segmentation QC

**Plots panel → Segmentation QC.** One row per segmentation source of the primary sample (Xenium and every cached Baysor / Proseg run, loaded or not) and counts mode (original, SPLIT-corrected), as a registered, cancellable job; table, four small bar plots, CSV.

- **Identity marker sets**: the 5 most specific genes of every type, derived once from the Xenium segmentation's original counts and annotation, or typed in (`Type: gene, gene, …` per line; a line `geneA, geneB` adds a pair of mutually exclusive markers).
- **Exclusion-rule failure**: share of labelled cells whose own identity set is not the largest share of their transcripts. **Co-expression**: over pairs of top markers of different types, cells with both ÷ cells with either (count ≥ 1). **Spillover slope**: in cells not of the source type, source-identity transcripts per 100 regressed on the number of source-type neighbours within 15 µm (reference-free; 0 = no dependence), and the reverse. **Own-identity retention** (corrected counts): own identity transcripts after ÷ before correction; below 1 with a lower failure rate is over-correction, not improvement. Plus cells, median transcripts / genes (count ≥ 1) / area per cell.
- **Assigned** (original counts only): Σ counts of the source's cells ÷ gene transcripts (Q ≥ 20) inside the bounding box of its cell centroids, grown by one median cell radius. The transcripts of the section are counted by the job itself the first time (the same scan and cache file as the *All transcripts* grid of the Transcript Density tab; cancellable, nothing is written when cancelled) and read from the cache afterwards. For the Xenium segmentation of a whole section this equals the share of Q ≥ 20 gene transcripts with a cell id in `transcripts.parquet`; for a run on a region the box is at least as large as the region, which lowers the value. A dataset loaded from a `.zarr` has no transcript file: the column is empty and the status says so.
- A corrected row covers the cells the correction touched (non-empty corrected row; "Corrected share") and its retention compares those same cells before and after.
- A run without its own annotation — or whose annotation uses other type names — gets the Xenium label of the nearest Xenium centroid within 10 µm (the Labels column says so). This favours segmentations that resemble Xenium's.
- No p-values: these are descriptive. Runs on a small ROI give noisy slopes. The original benchmark's criteria that need a paired dissociated reference (patient-specific signal, arm balance) are not implemented.

### Stratified co-localisation with matched nulls

**Plots panel → Stratified Co-localisation**, beside the kNN-based Neighborhood Enrichment. For every ordered pair (focal A, target B) and radius R (default 50 µm):

- **O** = B cells within R of each A cell. **E** = the mean of that count around the *pool* cells (labelled cells that are neither A nor B) of the same **stratum**: sample × compartment × edge-distance bin (50, 100, 150, 200, 300, 500, 1000 µm) × local-density quintile; a coarser stratum when fewer than 3 pool cells share it. **Effect** per sample = log2((ΣO + 0.5) / (ΣE + 0.5)). The unstratified value (`effect_naive`, pool = the whole sample) is given beside it: the difference is what "both types live in the same region" contributed.
- **Bias and within-sample p**: the A label is permuted among A and pool cells within each finest stratum (B fixed, default 200 permutations, fixed seed); the mean permuted effect is subtracted (`effect_corrected`), p = (1 + permutations at least as far from that mean) / (1 + permutations).
- **Across samples**: with ≥ 3 independent units (donors when given, else samples) the bias-corrected effect is tested with the exact sign-flip test, and between conditions A and B with the paired sign-flip or label-permutation test of the design. With fewer units the tab says **SINGLE-SAMPLE / PER-SAMPLE RESULT**: the within-sample p treats the cells of one section as exchangeable and says nothing about other sections or donors.
- **Placebo anchors** (optional focal type): the same statistic for count-matched random anchor cells drawn within each sample × compartment (mean, 95 % range, p), and for the other types within 3× the focal type's abundance that share its most common compartment. The verdict is "specific" only when the focal type lies beyond both. Without a placebo, "X sits near Y" cannot be told from "X is a cell of that compartment".
- Heatmap (focal × target, `*` FDR < 0.05 of the test named in the title), table, CSV (per sample and pair, with the pair-level tests). Only types with ≥ 20 cells; a pair needs ≥ 20 focal and ≥ 10 target cells in a sample; the 40 most abundant types at most. Compartments come from the Confound Controls tab; when absent, edge distance and density are computed with the defaults and there is no core.
- **Speed** (measured): 112,551 labelled cells, 11 types, R = 50 µm, 200 permutations, placebo: 5.4 s, about 60 MB above the loaded dataset.

### Aggregates

**Aggregates** tab: finds connected groups of cells of a chosen *class* — a set of cell types
you pick, e.g. the immune types — that are locally denser than chance, and reports each one.
Ported from an earlier spatial study by the author; the detection (local count, within-stratum
null, hotspot rule, Delaunay components, tile area, permuted-label control) reproduces the
original's numbers exactly on fabricated data, and every value of that study is a parameter here.

- **Detection, per sample.** For every cell the class cells within *R* (default 50 µm, the cell
  itself included) are counted. The class label is permuted among the cells of the same stratum —
  compartment × edge-distance bin × local-density quintile, taken from the Confound Controls tab
  (or computed with its defaults, and said so) — 200 times by default. A cell is a *hotspot* when
  p = (k + 1) / (permutations + 1) ≤ α (default 0.01; k = permuted counts ≥ the observed one) and
  at least 3 class cells lie within *R*. Hotspot cells of any type that are joined by Delaunay
  edges of at most 30 µm form a component; components with at least *N* class cells (default 10)
  are the aggregates. Neighbour searches, triangulations and permutations never cross samples.
- **Per aggregate** (table and CSV): id (1 = most class cells), sample, cells, class cells, class
  fraction, area (the tissue its cells are nearest to, 5 µm grid, at most 20 µm from a cell),
  centroid, composition, and for every cell type the inside-vs-matched-outside log2 O/E: cells
  of the type inside ÷ the number expected from the labelled cells of the same stratum that lie
  outside every aggregate and its 25 µm rim. The class types are enriched by construction; the
  other types are the informative ones.
- **Per sample**: aggregates, aggregates per mm² of tissue, area fraction, share of the class
  cells inside, and the **control**: the same detection on 20 permuted labellings gives the
  number of aggregates chance alone produces (false-aggregate rate = control mean ÷ observed).
- **On the map, both renderers**: outline polygons (convex hull, or an alpha hull that follows
  the shape: Delaunay triangles with all edges ≤ 30 µm) labelled A1, A2, … in the WebGL view,
  with a "Show outlines" toggle; **Color By → "Aggregates: Aggregate id"** colours the member
  cells by id (cells outside are grey; the scale starts below 1 so that aggregate 1 is visible).
  The outlines are shown for the segmentation they were computed on.
- **Save as SpatialData** writes `ctl_aggregate_id` (empty outside), `ctl_aggregate_hotspot` and
  `ctl_aggregate_p` for the primary sample.
- **What the p-values mean.** The hotspot p-values have the cells of one section as their unit
  (exchangeable within strata): they are a detection threshold and say nothing about other
  sections or donors; a single sample is labelled as a single-sample result. Between conditions
  the unit is the sample or donor: aggregates per mm² and area fraction are compared with the
  exact sample-level test (paired sign-flip or label permutation, as in *Samples and conditions*)
  when there are at least 3 independent units per condition, with n and the smallest achievable
  p stated; otherwise no test is run and the reason is given.
- **Limits.** The result depends on the annotation: a class that is most of the tissue gives
  large "aggregates" and a high control rate (read the control column). Recompute after changing
  the subset, the segmentation or the annotation. The cells × cells neighbour matrix is held in
  memory (about 100 neighbours per cell at 50 µm in dense tissue). Registered job with progress
  and Cancel.

### CellNEST Cell-Cell Communication

Visualize cell-cell communication interactions inferred by [CellNEST](https://github.com/schwartzlab-methods/CellNEST). Requires running CellNEST externally first; use `functions/export_to_cellnest.py` to export the SpatialData zarr to a CellNEST-ready `.h5ad`.

**Loading results:**

Click **"Load CellNEST Results"** in the sidebar to open the modal:
1. Enter the CellNEST output folder path (the root containing `output/`, `metadata/`, `embedding_data/` subdirs)
2. Click **Detect Datasets** to auto-discover available datasets
3. Select a dataset and set the **Strip suffix** field (default `_4` — strip the `_4` appended to Proseg cell IDs by CellNEST)
4. Click **Load** — progress appears in the sidebar status

**CellNEST CCC tab:**

| Panel | Content |
|-------|---------|
| Left | Filtered LR-pair frequency bar chart (top 30 by count) |
| Right | Circos/chord diagram — cell-type → cell-type interaction weights, colored by selected scheme |

**Controls:**

| Control | Description |
|---------|-------------|
| LR Pair | Multi-select dropdown; select one or more ligand-receptor pairs to visualize |
| Top N | Maximum number of interactions to display per pair (ranked by attention score) |
| Edge W | Base edge line width (applies to spatial map overlay) |
| Color By | Edge color scheme: **LR Pair**, **Ligand**, **Receptor**, **Attention Score** (Viridis), or **Sender Cell Type** |
| Annotation | Cell-type annotation column to use for sender/receiver coloring and filtering |
| Ligand / Receptor / Sender / Receiver filters | Multi-select dropdowns to subset interactions |
| Options | **Scale edge width by attention score** — scales each edge's line width proportionally to its attention score (0.5×–2×); **Highlight autocrine** — rings around self-loop cells; **Color background by degree** — background cells colored by total interaction count; **Gray out non-participating cells on map** — adds dark overlay on main map |

Edges on the spatial map overlay are drawn as **directed arrows** (arrowhead at the receiver end), scaled with Edge W. Chord ribbon thickness in the circos diagram is automatically proportional to the aggregated attention weight (√-scaled for perceptual uniformity).

**Main map overlay:**

Check **"Show edges on map"** in the sidebar to overlay CellNEST interaction edges on the central spatial scatter. The overlay respects the selected LR pairs, Top N, Edge W, Color By, and gray-out option. The overlay persists through pan/zoom.

**Downstream TF analysis:**

Expand the **▶ Downstream TF Analysis** panel at the bottom of the CellNEST tab. Enter a receptor gene (e.g. `CCR7`), set the number of hops through the PPI network (default 2), and click **Find TFs**. The panel shows transcription factors downstream of the receptor that are either in the Xenium panel or have panel-gene targets. Select one or more TFs to plot their expression spatially.

Databases (`human_signaling_ppi.csv`, `human_tf_target.csv`) are downloaded from the CellNEST GitHub on first use and cached at `~/.xenium_explorer_cache/cellnest_db/`.

**O/E normalization** (Interaction Graph + Chord Diagram) — When enabled, each edge weight is divided by the expected number of contacts under a random mixing model (`observed / (freq_A × freq_B × total_edges)`). This removes bias from abundant cell types and highlights interactions that exceed random expectation.

**Dimension controls** — Each tab has W × H inputs (in pixels) at the bottom. The graph fills the remaining tab space automatically.

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

The ROI and patch grid are displayed as overlays on the spatial plot. Click **✓ Patches look good — proceed** to unlock Step 2, or **⏭ Skip — run on full tissue** to bypass patch setup and run directly on the full slide: Baysor and Proseg then make one run over the whole region even when patches from earlier in the session exist (a run made with patches shows "N patches" in its Segmentation dropdown label). Use **↺ Re-build** to clear the cache and re-run with different parameters.

The ROI and the patches are saved in the SpatialData cache store together with the settings that produced them (QV filter, patch width, min transcripts/patch). After a restart, **▶ Prepare Patches** with the same settings restores them in a few seconds; with different settings, or for a store written by an earlier version (which has no record of its settings), the patches are computed again.

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
| Prior type | — | nucleus boundaries (alternative: cell boundaries) |
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

SpaGE is a git submodule (`SpaGE_repo/`). Fetch it once with `git submodule update --init SpaGE_repo`, or set `SPAGE_REPO` to an existing clone of https://github.com/tabdelaal/SpaGE.

Click **"Impute Genes"** in the sidebar to open the SpaGE imputation modal. Imputes expression of genes not in the Xenium panel using a scRNA-seq reference (Seurat RDS).

**GUI (modal):**
- **Reference RDS** — Path to Seurat object
- **n_pv** — Number of principal vectors for cross-dataset alignment (10–100, default 50)
- **Genes to impute** — Comma- or newline-separated gene list; leave blank to auto-select the 200 genes with the highest variance among all reference genes that are not in the panel

SpaGE always imputes from the **original** counts of the active segmentation, also when Active Counts is set to "SPLIT Corrected" (the status line says so). To impute from SPLIT-corrected counts use `functions/impute.py --use_corrected`, which writes a separate store.

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
- **Original** — all expression lookups use the raw Xenium/Baysor/Proseg count matrix
- **SPLIT Corrected** — measured genes are read from the corrected matrix. SpaGE-imputed genes have no corrected values: they show the same imputed values as under "Original"
- The "SPLIT Corrected" option is greyed out (disabled) whenever no corrected matrix is available — either in memory or in the active zarr's `X_corrected` layer. It re-enables automatically once the corrected data is loaded (e.g., after selecting a Baysor/Proseg run that has a previously-run SPLIT result stored in zarr)
- Switching segmentation sources resets the toggle to "Original" if no corrected data exists for that source

**SpaGE on corrected counts:** not available inside the app. The Impute Genes dialog and `run_spage()` always use the original counts (`run_spage(use_corrected=True)` prints why and runs nothing), because the app stores imputed genes as plain `[imp]` columns and a corrected-input run would overwrite them under the same name. Use the standalone script instead: `python functions/impute.py <store.zarr> <ref.rds> --use_corrected` reads the `X_corrected` layer, writes a new store, and lists the genes in `adata.uns["split_corrected_imputed_genes"]`; when that store is opened in the app they are labelled `[corr+imp]`.

**SPLIT under an active `subset()`:** the corrected counts, clusters and UMAP of the subset are kept in memory for the session and are **not** written to the store (the status line says so); an earlier correction of the whole dataset stays in the store and keeps its rows for the cells outside the subset. Run SPLIT without a subset to store its result.

**Several samples.** SPLIT runs on the primary sample; an added sample carries corrected counts only if its own store already holds them (the `X_corrected` layer written by an earlier SPLIT run on that sample, or a store exported by Save as SpatialData). Corrected and original counts are never mixed in one colour scale or one test: with several samples loaded, **"SPLIT Corrected" is available only when every sample has corrected counts**, and then applies to all of them (gene colours in both renderers, cell-level differential expression, gene-set scores, Moran's I, ligand–receptor scores). Otherwise the option is disabled, the original counts are used for all samples, and a note under Active Counts names the samples that lack corrected counts. Pseudobulk differential expression always uses the original counts. *Behaviour change:* corrected counts used to be ignored whenever several samples were loaded, while the option stayed selectable.

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

A filter that cannot be applied is skipped with a message in the log (`cell_type` without an annotation, an unknown gene, a QC column the dataset does not have). A call in which no filter was applied - only skipped ones, or none given - changes nothing and does not mark a subset as active. Niches computed under a subset exist for the subset's cells only; after `unsubset()` the other cells are shown as **No niche**.

### Jobs panel: progress and Cancel

Every long-running job (Baysor, Proseg, Prepare Patches, cell type annotation, SPLIT, SpaGE, Rerun UMAP, spatial analysis, niches, CellNEST load, Save as SpatialData, loading a cached segmentation run) is listed in a small **Jobs** panel at the bottom right of the window while it runs. The panel floats above the page and above the open dialog, so it is visible next to the sidebar status lines and next to the status line of each dialog (those still work as before). Each row shows the kind of job, its label, the current step, the elapsed time, a progress bar when the fraction done is known, and a **Cancel** button. A click on the panel's header line folds the list away (it sits over the corner of the server log). The panel disappears a few seconds after the last job has ended and does not poll the server while no job is running.

| Job | Progress shown | What Cancel does |
|-----|----------------|------------------|
| Baysor | patch *i* of *n*; inside a patch the stage Baysor logs and the percentage of its segmentation iterations (the stages before and after the iterations print no percentage and are shown at fixed marks) | the Baysor (Julia) process and its children are terminated at once |
| Proseg | patch *i* of *n*; inside a patch the fraction of sampler iterations, read from Proseg's own progress bar (Proseg draws it on a terminal only, so the app runs it on a pseudo-terminal; without one the bar is missing and only the patch count advances) | the Proseg process is terminated at once |
| Prepare Patches | step 1/3 SpatialData, 2/3 tissue ROI, 3/3 patches | stops after the current step (the sopa call in progress finishes first) |
| CellTypist / Seurat / RCTD annotation | step text | stops after the current step |
| SPLIT | step text | stops after the current step (an RCTD run in R cannot be interrupted) |
| SpaGE | step text; fraction of gene chunks during the imputation itself | stops after the current step / gene chunk |
| Rerun UMAP, spatial analysis, niches, CellNEST load | step text (steps *k* of *n* for spatial analysis and niches) | stops after the current step |
| Save as SpatialData | step text | only before the store is written |

How cancelling works, and its limits:

- **External tools** run in their own process group. Cancel sends `SIGTERM` to the group and `SIGKILL` five seconds later to whatever is left, so Julia and any child processes go with it. The tools are also stopped, and their working directories removed, when the app exits or is stopped with Ctrl-C, `SIGTERM` or `SIGHUP`; after `kill -9` of the app a running tool has to be stopped by hand.
- **Python and R steps cannot be interrupted.** The job checks for a cancel request between steps, so the panel shows *cancelling after the current step…* until the step in progress returns. For RCTD (annotation or SPLIT) on a whole slide that step can take tens of minutes; reading a large Seurat reference, `sopa` reading the Xenium bundle or building patches, and a UMAP fit are other long single steps. `baysor --version`, which the app runs to find Baysor before the segmentation starts (Julia start-up), is also not interrupted.
- **Nothing half-written is kept.** A Baysor/Proseg run works in `<run directory>.partial` and gets its real name only when it is complete; a cancelled or failed run removes that directory, so it never appears in the Segmentation dropdown and an earlier run with the same parameters stays as it was. Once a job starts writing its result (run directory and zarr store, SPLIT / SpaGE write-back to the SpatialData store, annotation caches) it can no longer be cancelled; the row then reads *writing results — cannot be cancelled now*. A streamed SpaGE result that was cancelled is deleted.
- A cancelled job reports `✗ Cancelled` in its status line, the same job can be started again straight away, and the loaded dataset is not changed. As before, starting Baysor/Proseg replaces the run that was loaded for that tool, and an annotation run clears that method's previous labels, at the moment the job is started, not when it is cancelled. Intermediate results that are complete and reusable are kept: the SpatialData store, tissue ROI and patches of Prepare Patches, and the RCTD result SPLIT caches before `purify()` (with its labels).
- Loading another sample is still refused while any job is running, including one that is being cancelled.
- The bar is the share of the whole job: 90 % for the tool (split evenly over the patches, whatever their size), the rest for the expression matrix, clustering / UMAP and the store. It is not a time estimate.
- A failed Baysor/Proseg run no longer leaves its partial output in the cache (the tool's messages are in the server log).

From the REPL: `_jobs_list()` returns the registered jobs (`.kind`, `.label`, `.state`, `.step`, `.fraction`, `.elapsed()`, `.thread`, `.procs`), `_job_cancel(job.id)` cancels one.

### Server Log & Python REPL

The bottom panel is split into two columns:
- **Left** — Cell info on click (cell ID, coordinates, QC metrics, gene expression bar chart)
- **Right** — Live server log streaming Python stdout/stderr and R console output (rpy2). Auto-scrolls to the bottom; pauses when you scroll up and resumes when you scroll back down.

The entire panel can be collapsed/expanded with the ▼/▲ button.

**Python REPL** — A `>>> ` prompt at the bottom of the log panel accepts any Python expression or statement and executes it in the app's module namespace, with output appearing in the log. Tab-completion and Up/Down arrow history are supported.

The REPL runs arbitrary code on the server, so it is **disabled unless the app is started with `XENIUM_ENABLE_REPL=1`**, and then accepts requests from the same machine only: the connection must come from a loopback address, the page must have been opened as `localhost`, `127.0.0.1` or `[::1]` (the request's `Host` header is checked, which stops a web page that re-binds its own hostname to 127.0.0.1), and requests carrying `X-Forwarded-For` / `Forwarded` headers are refused, so the REPL does not work through a reverse proxy in this mode. `XENIUM_ENABLE_REPL=remote` lifts these restrictions; use it only on a network you trust.

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
| `baysor_{dataset}_{tag}/` or `baysor_{dataset}_{datasethash}_{tag}/` | Baysor output per parameter set (new runs include a hash of the dataset path so bundles with the same folder name do not share runs; older directories still load); includes `params.json` and `spatialdata_baysor.zarr` (contains expression, clusters, UMAP, imputed genes, and `X_corrected` layer after SPLIT) |
| `proseg_{dataset}_{tag}/` or `proseg_{dataset}_{datasethash}_{tag}/` | Proseg output per parameter set (same naming rule); includes `params.json` and `spatialdata_proseg.zarr` (contains expression, clusters, UMAP, imputed genes, and `X_corrected` layer after SPLIT) |
| `baysor_…_{tag}.partial/`, `proseg_…_{tag}.partial/` | Working directory of a Baysor/Proseg run in progress; renamed to the run directory when the run is complete, removed when it fails or is cancelled (one left behind by a killed app is removed by the next run with the same parameters or by Clean Cache) |
| `cellnest_db/` | Ligand-receptor / signalling database files used by the CellNEST tab |
| `genesets_{hash}.json` | Gene sets saved in the Gene Sets tab (per dataset) |
| `de_groups_{hash}.json` | Cell groups (outlines) saved in the Differential Expression tab (per dataset) |
| `density_{hash}/{gene}__b5um__qv20_genes_v1.npz` | Transcript-density grid of one gene (non-empty 5 µm bins); the name carries the bin size and the version of the transcript filter |

The cache root is `~/.xenium_explorer_cache/` unless `XENIUM_CACHE_DIR` is set; every cache above, including `cellnest_db/`, follows it.

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

Output is a full SpatialData zarr with imputed gene columns appended to the AnnData table under their plain gene names (as the app writes them). `adata.var["is_imputed"]` marks imputed genes; `adata.uns["imputed_genes"]` lists them. With `--use_corrected` the `X_corrected` layer (SPLIT) is the input and the genes are listed in `adata.uns["split_corrected_imputed_genes"]` instead. Without a gene list, the 200 genes with the highest variance among all non-panel reference genes are imputed.

---

### `functions/spatialdata2seurat.py` — SpatialData → Seurat RDS

Converts a SpatialData zarr to a Seurat RDS equivalent to `LoadXenium()` output.

| Seurat slot | Content |
|---|---|
| `@assays$Xenium` | Panel gene raw counts (dgCMatrix, genes × cells) |
| `@assays$Imputed` | SpaGE-imputed genes, if present |
| `@meta.data` | All obs columns + `nCount_Xenium`, `nFeature_Xenium`, `cell_area`, `nucleus_area` + cell type annotation columns (Seurat kNN, RCTD, CellTypist — if present in cache) |
| `@reductions$umap` | UMAP coordinates (`split_umap_1/2` when `--use_corrected`, else `umap_1/2`) |
| `@images$fov` | Centroids + cell segmentation + nucleus segmentation |

```python
from functions.spatialdata2seurat import spatialdata2seurat

# Standard conversion
spatialdata2seurat(
    sdata_path="/data/xenium.zarr",
    output_rds="/data/xenium.rds",
)

# SPLIT-corrected counts (drops uncorrected cells), skip imputed assay
spatialdata2seurat(
    sdata_path="/data/xenium.zarr",
    output_rds="/data/xenium_corrected.rds",
    use_corrected=True,
    include_imputed=False,
)

# Only cells within a named ROI
spatialdata2seurat(
    sdata_path="/data/xenium.zarr",
    output_rds="/data/xenium_roi.rds",
    roi_path="/path/to/rois.json",
    roi_name="Infarct zone",
)
```

CLI:
```bash
python functions/spatialdata2seurat.py /data/xenium.zarr /data/xenium.rds

python functions/spatialdata2seurat.py /data/xenium.zarr /data/xenium.rds \
    --use_corrected --no_imputed --no_nucleus \
    --cache_dir ~/.xenium_explorer_cache --assay Xenium --fov 1467

python functions/spatialdata2seurat.py /data/xenium.zarr /data/xenium_roi.rds \
    --roi ~/.xenium_explorer_cache/rois_XXXXXXXX.json \
    --roi_cls Sample --roi_name Top --fov 1697
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--use_corrected` | Use `X_corrected` (SPLIT-corrected) counts; drops cells without correction |
| `--no_imputed` | Skip the `Imputed` assay |
| `--no_nucleus` | Skip nucleus segmentation from `@images$fov` |
| `--roi PATH` | Path to ROI JSON file or ROI directory. JSON files use the app's polygon format (`[{cls, name, polygon_xy}]`); cells are filtered by point-in-polygon (requires `shapely`) |
| `--roi_cls CLS` | Filter ROIs to those with `cls == CLS` (default: all classes) |
| `--roi_name NAME` | Filter ROIs to the named ROI only |
| `--cache_dir DIR` | Cache directory to search for annotation parquets (default: `~/.xenium_explorer_cache`) |
| `--assay NAME` | Name of the Seurat assay (default: `Xenium`) |
| `--fov FOV` | FOV identifier for `@images$fov` (default: `fov`; use the numeric FOV count, e.g., `1467`) |

**Notes:**
- Cell type annotations (Seurat kNN, RCTD, CellTypist) are automatically discovered from `~/.xenium_explorer_cache/` (or `--cache_dir`) and written to `@meta.data`. Annotations are matched to the current dataset via `dataset_hash` in `params.json`; for older runs without a hash, the script falls back to cell-ID overlap matching. Run annotation in the app first.
- `cell_area` and `nucleus_area` are included in `@meta.data` when available in the zarr obs. For Proseg runs, `cell_area` is derived from the `volume` column in `cell-metadata.csv.gz`.
- `@reductions$umap` is written from the pre-computed UMAP stored in zarr. If `--use_corrected`, the SPLIT-corrected UMAP (`split_umap_1/2`) is used. Load the run in the app at least once to ensure UMAP is cached; re-run this script if the warning `split_umap_1/2 not found` appears after the UMAP has been recomputed.
- The ROI JSON path is printed to the server log when ROIs are saved; it is also at `~/.xenium_explorer_cache/rois_<hash>.json`.

Requires: `rpy2`, R packages `Seurat`, `SeuratObject`, `Matrix`; `shapely` (for `--roi`).

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
| `functions/geo_to_bundle.py` | Assemble a loadable bundle from a GEO-style deposit (matrix `.h5`, per-cell table, transcripts), from files on disk or by GEO sample accession |

#### `functions/geo_to_bundle.py` — open a GEO deposit

Public Xenium deposits usually hold a few processed files per section rather than the instrument output. This script writes `experiment.xenium`, `cells.parquet` and `cell_feature_matrix.h5` (plus `transcripts.parquet` when deposited) into a directory the app can open. Every extra column of the per-cell table (cell types, patient, group) is kept and appears in the app.

```bash
# download one sample's files from GEO, then build the bundle
python functions/geo_to_bundle.py --gsm GSM10012173 --out_dir rv_section/
python xenium_explorer.py rv_section/

# or from files already on disk (one section per directory)
python functions/geo_to_bundle.py --input_dir geo_files/ --out_dir rv_section/
```

The morphology image is skipped unless `--with_image` is given (it is large, and the download is a single stream). The resulting bundle has no boundaries, clustering or UMAP unless the deposit provides them.

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
| `rpy2` + R ≥ 4.3 | Seurat label transfer, SpaGE, RCTD, SPLIT, pseudobulk differential expression |

**R packages** (via rpy2):

| Package | Source |
|---------|--------|
| `SeuratObject`, `Matrix` | CRAN / Bioconductor |
| `spacexr` | `devtools::install_github('dmcable/spacexr')` |
| `SPLIT` | `remotes::install_github('bdsc-tds/SPLIT')` |
| `DESeq2` (optional `apeglm`), or `edgeR` | Bioconductor: `BiocManager::install(c("DESeq2", "apeglm"))` — pseudobulk differential expression only |

**WebGL viewer:** deck.gl, Viv, geotiff and an OpenJPEG WebAssembly decoder are bundled into `assets/xenium_viewer.js`, `assets/xenium_image_worker.js` and `assets/openjpegwasm_decode.wasm`. Nothing needs installing to use them.

**External CLI tools:**
- [Baysor](https://github.com/kharchenkolab/Baysor) — Julia-based resegmentation; binary at `~/.julia/bin/baysor`
- [Proseg](https://github.com/dcjones/proseg) — Rust-based probabilistic resegmentation; `conda install bioconda::rust-proseg` or `cargo install proseg`


---

## Code layout

`xenium_explorer.py` is the entry point; the code is in the `xenium_app/` package.
Importing the package starts the app as before: it loads the dataset named by `DATA_DIR`
or the command line, builds the Dash app and registers the callbacks.

| Module | Contents |
|--------|----------|
| `core` | configuration (`CACHE_BASE_DIR`), constants, the shared state dicts and their locks, server log capture; holds `DATA` |
| `jobs` | dataset generation counter and the job registry (progress, Cancel, tool processes) |
| `caching` | cache locations: dataset keys, segmentation run directories, cache size and clean-up |
| `ui` | small Dash building blocks: labels, plot-tab controls, jobs panel rows |
| `loading` | readers: Xenium bundle → SpatialData store → the `DATA` dict |
| `dataview` | derived data and accessors on `DATA`: gene expression, clusters, viewport cells, extra samples |
| `annotation` | CellTypist, Seurat kNN and RCTD |
| `reseg`, `clustering` | Baysor / Proseg runs and cached-run loading; clustering and UMAP of a run |
| `split`, `spage` | SPLIT ambient RNA correction; SpaGE imputation |
| `subsetting` | the REPL helpers `subset()`, `unsubset()`, `get_genes()`, `retry_split_zarr_write()` |
| `spatial`, `regions` | spatial analysis and niches; ROI helpers |
| `morph` | morphology image overlay of the Plotly view |
| `figures` | Plotly figure builders (spatial map, UMAP, analysis plots) |
| `analysis` | differential expression, gene-set scores, transcript density |
| `stats` | sample → condition / donor design (persisted), exact sample-level tests, pseudobulk differential expression (DESeq2 / edgeR), composition test, Moran's I, ligand–receptor scores |
| `controls` | confound controls: tissue compartments, spillover score, gene ownership, leak gate, segmentation QC metrics, stratified co-localisation, annotation label report |
| `proximity` | proximity differential expression (near vs far from an anchor cell type, matched random anchors) and contact-group pseudobulk differential expression |
| `lrdecomp` | ligand–receptor exposure decomposition into five factors, per-term sample-level tests |
| `aggregates` | aggregates: connected groups of cells of a chosen class, outlines, composition, per-sample density |
| `segfree` | segmentation-free abundance: marker transcripts per mm² of tissue from the transcript file, area-bin sweep, concentration against a redistribution null |
| `cellnest` | CellNEST loading, filters and plots |
| `sdata_store` | SpatialData / sopa backend: cache store, Save as SpatialData, tissue ROI and patches |
| `startup` | the import-time work: locate and load the dataset, autoload cached results |
| `app_layout` | the Dash `app` and its layout |
| `cb_plots`, `cb_tools`, `cb_reseg`, `cb_spage`, `cb_data`, `cb_spatial`, `cb_analysis`, `cb_stats`, `cb_controls`, `cb_proximity`, `cb_lrdecomp`, `cb_aggregates`, `cb_segfree` | the callbacks, by feature |
| `viewer_provider` | data provider for `xenium_viewer_api.py` (WebGL view) and `/health` |
| `compat` | the single namespace over all modules (below) |

Two rules keep this equivalent to the former single file:

- **A global is defined in one module and looked up there.** Code in another module writes
  `core.CACHE_BASE_DIR`, `figures.make_spatial_fig(...)`, never `from .core import ...`.
  `import xenium_explorer as xe` therefore still gives one namespace: `xe.<name>` reads any
  global of any module, and `xe.<name> = value` (or `monkeypatch.setattr(xe, ...)`) rebinds
  it in the module that defines it, where all code picks it up. The REPL uses the same
  namespace. `tests/test_package_layout.py` checks the rule and the rebinding.
- **Import order is the order of the side effects.** `xenium_app/__init__.py` imports the
  library modules first (they only define things), then `startup`, `app_layout`, the
  `cb_*` modules and `viewer_provider`, which do the work in the order the single file did.
  A library module must not import one of those.

---

## Developing the WebGL viewer

The viewer source lives in `viewer_src/` and is bundled with esbuild. Node.js is needed only for this.

```bash
cd viewer_src
npm install            # once
npm run build          # writes ../assets/xenium_viewer.js

# standalone server exposing the viewer's data routes over a bundle, with a bare test page
python viewer_src/dev_server.py /path/to/xenium_bundle 8070     # http://127.0.0.1:8070/

# tests (headless Chromium via Playwright)
node test/run.mjs                                  # viewer core against dev_server.py (bundles: XV_SKIN, XV_TINY or XV_TEST_DATA, see test/lib.mjs)
node image_test/verify.mjs                         # image layer: decode and placement
node test/app_smoke.mjs http://127.0.0.1:8050 out/ # the real app, started with XENIUM_RENDERER=webgl
node test/glue_checks.mjs http://127.0.0.1:8050    # renderer switching, selection and image handling in the real app
node test/jobs_panel.mjs http://127.0.0.1:8050 out/ # jobs panel: starts Proseg on cached patches, checks progress, clicks Cancel
node test/analysis_checks.mjs http://127.0.0.1:8050 out/ # Differential Expression / Gene Sets / Transcript Density tabs in both renderers (lasso → groups → test → gene set → density)
XS_SAMPLES=/bundle2,/bundle3,/bundle4 node test/stats_checks.mjs http://127.0.0.1:8050 out/ # several samples: design editor, pseudobulk DE, composition, density of added samples in both renderers, Moran's I, ligand-receptor (see the top of the file)
node test/proximity_checks.mjs http://127.0.0.1:8050 out/ # Differential Expression: "Proximity to a cell type" and "Contact groups" (with XS_SAMPLES=… the contact test runs in R; without, its single-sample refusal is checked)
node test/webgl_fallback.mjs http://127.0.0.1:8050 # automatic fallback to Plotly without WebGL2 (browsers started with WebGL disabled)
XA_CLASS="B cells|Plasma cells" node test/aggregates_checks.mjs http://127.0.0.1:8050 out/ # Aggregates tab in both renderers: run, table, Color By, outlines, toggle, alpha hulls, CSV (needs a cell-type annotation in the cache)
node test/real_results.mjs                         # WebGL against Plotly on real Baysor/Proseg/RCTD/SPLIT/SpaGE results; starts its own app (paths: see the top of the file)
```

| Path | Contents |
|------|----------|
| `xenium_viewer_api.py` | Binary data routes under `/viewer/*` (cells, colours, boundaries, transcripts, image byte ranges) and the lazy reader over `cells.zarr.zip` / `transcripts.zarr.zip`. The wire format is documented at the top of the file. |
| `viewer_src/src/` | Viewer core (`index.js`), viewport loader, selection, colour maps |
| `viewer_src/src/image/` | Morphology image layer: OME-TIFF pyramids, JPEG 2000 decoding in web workers |
| `assets/xenium_viewer_dash.js` | Glue between the Dash controls and the viewer (not bundled; edit directly) |

The same routes are registered a second time under `/umap/viewer/*` for the UMAP panel.

---

## Changes that affect results (November 2026)

A review of the code base fixed a number of defects. Most are invisible, but these change numbers or outputs relative to earlier versions:

- **"% of cells expressing" on SPLIT-corrected counts** (DE `% A` / `% B`, the ligand–receptor detection threshold, Moran's `% cells`, genes per cell in Segmentation QC): a cell detects a gene when its count is **≥ 1 transcript**. For integer counts this is the former count > 0; for corrected (fractional) counts a residue below one transcript no longer counts, so these percentages are lower than before. The DE table also gives **Rate A / Rate B** (Σ counts ÷ Σ total counts of the group, per 10,000), which is defined for fractional counts and does not move with depth; `low_count` marks genes with fewer than 100 transcripts over both groups.
- **DE status line and CSV** give the median counts per cell of both groups and warn when they differ by more than 1.5×; they state the unit of the p-values.
- **Pseudobulk volcano**: coloured by FDR alone; x axis = unshrunken log2FC unless the shrunken one is chosen (see above). The cell-level volcano keeps its |log2FC| ≥ 0.5 rule.
- **Sample design**: has a donor field, is saved per set of samples and restored (it used to be memory-only and lost on Load Sample).
- **Annotation dialog**: shows the label counts of every annotation of the active segmentation, a caution that labels on a targeted panel can collapse a lineage into one label at high confidence, and a flag when one label takes more than 80 % of a coarse cluster that the finest cluster set splits into several sub-clusters.
- **Baysor input** is restricted to gene transcripts with Q ≥ 20. Control and unassigned codewords and low-quality transcripts were previously passed through, so new Baysor segmentations and their counts differ.
- **SPLIT-corrected counts** are aligned to genes by name. `SPLIT::purify()` returns only the genes shared by the panel and the reference; these columns were previously written as the first *n* panel genes, so whenever the reference lacked a panel gene every later gene was mislabelled and the last columns were zero. Any `X_corrected` layer, corrected export or RDS produced with such a reference (in the app or with `functions/run_split.py`) must be regenerated. A quick test: the alphabetically last panel genes are all zero in an affected matrix.
- **Proseg input** is restricted to gene transcripts with Q ≥ 20, like Baysor. Low-quality transcripts are dropped with `--min-qv 20` (the `--xenium` preset of Proseg 1.1.9 applies no quality filter), and control, unassigned and deprecated codewords are removed from the input before Proseg reads it (Proseg has no option to exclude features; they were previously used for the segmentation and only left out of the count matrix). A whole-slide run therefore writes a filtered temporary copy of `transcripts.parquet` into the run's cache directory and deletes it afterwards. Cached Proseg runs made earlier still contain these transcripts. Proseg's own default of ignoring transcripts more than 60 µm from any nucleus (`--max-transcript-nucleus-distance`) is unchanged.
- **Proseg prior "None (unsupervised)"** was removed. Proseg 1.1.9 cannot run without an initial segmentation and stopped with a missing-argument error. A browser session that still holds this value runs with the default nucleus prior and shares that run's cache entry.
- **Annotation and new resegmentation runs** use measured panel genes only; after a SpaGE write-back, imputed columns were being treated as counts.
- **Reloading a patch run from cache** without its fast-load files returns the same cells as the live run rather than including overlap duplicates.
- **Patch-based Baysor/Proseg** keeps a cell whose centroid lies in the overlap between two patches once instead of twice, so patch runs yield fewer cells.
- **Seam duplicates in patch runs** are removed. Two neighbouring patches segment their shared strip independently, so one cell could still be kept from both when its two centroids fell on either side of the dividing line. After the centroid rule, cells of different patches near such a line are now compared: when two polygons overlap by more than 50% of the smaller one's area, or the centroids are closer than 1.5 µm, only the cell lying deeper inside its own patch is kept (ties: more transcripts). In a 4-patch test region this removed 7 of 1,423 Proseg cells and 8 of 3,474 Baysor cells. The rule is recorded in the run's `params.json`, so reloading from cache gives the same cells. Cached patch runs made earlier are loaded unchanged and still contain these duplicates; re-run them to remove the duplicates.
- **Prepare Patches after a restart** restores the tissue ROI and the patches from the SpatialData cache store when QV filter, patch width and min transcripts/patch are the ones they were made with (seconds instead of about 10 minutes on a 74-million-transcript slide). A store written by an earlier version has no record of these settings; its patches are computed once more and recorded.
- **Co-occurrence** scores are 1 higher in log2: the expected count was being doubled, so random mixing now reads about 0 rather than about −1.
- **Interaction graph and chord diagram** no longer double-count same-type pairs; edge widths and same-type O/E values change.
- **Rerun UMAP** acts on the segmentation selected in the dropdown. For Baysor/Proseg, a UMAP computed on corrected counts is stored as `split_umap_1/2` and no longer overwrites `umap_1/2`.
- **SpaGE-imputed genes** are written back into the store and displayed as log1p, like measured genes.
- **Multi-nucleus cells** are drawn as one outline per nucleus.
- **RCTD multi mode** returns the first listed type per cell (it previously returned no labels).
- **ROI set operations** store the exact resulting outline instead of its convex hull.
- **Load Sample** is refused while a background job is running.
- **Neighbourhood enrichment** values are lower by log2(k), k being the Neighbours setting (3.32 at the default 10). Each row was divided by the number of cells of the source type instead of by its number of neighbour pairs, so every pair looked enriched and the values changed with k. The heatmap is now log2 of P(neighbour is B | cell is A) over the frequency of B, and reads about 0 under random mixing.
- **Color by Cluster on a Proseg run loaded from cache** shows the app's KMeans clusters (`cluster_kmeans_10`), as on a run just made. Cache rebuilds had copied Proseg's own `cluster`, `fov` and `centroid_z` columns into the run, and that `cluster` column (Proseg's internal component, a single value in the test runs) was then used for colouring and for `subset(cluster=…)`. Stores that already contain these columns are read without them.
- **Proseg "Skip — run on full tissue"** now gives one run over the whole region. Once patches existed in a session every Proseg run was patch-based whatever was chosen; the same choice also overrides Baysor's "Use patch-based segmentation" box. Runs made with patches say so in their dropdown label.
- **Multi-part Proseg cells** keep the part with the largest area as their outline; the part with the most vertices was kept before, which for a few cells per thousand was a small fragment. Outlines, exported shapes and the seam de-duplication of patch runs change for those cells.
- **Proseg `transcript_counts`** is the row sum of the run's expression matrix. It used to be Proseg's own `population` column (the last state of its sampler), while the matrix is built from the consensus transcript assignment, so the two differed by a transcript or two for some cells. A run loaded from an older cache is corrected on load (and in its zarr store). Filters on transcript counts (`subset(min_transcripts=…)`) can therefore keep or drop a few borderline cells differently than before.
- **Baysor prior on bundles with integer cell ids**: `-1` / `0` (unassigned) is no longer handed to Baysor as one large prior cell.
- **Runs with fewer than 4 cells**, or runs whose clustering fails, are kept with all cells in cluster 1 and no UMAP instead of being reported as failed and becoming impossible to load from cache.
- **A cached run that fails to load** no longer leaves the previously loaded run of the same tool displayed under the newly selected name: the old result is dropped when the new one is selected, and on failure the Segmentation dropdown returns to Xenium.
- **Rerun UMAP on a Baysor/Proseg run** uses measured panel genes only. After SpaGE the run's matrix also holds imputed columns, which were being normalised and embedded as counts.
- **SpaGE auto-selection** ("top-200 highly variable genes") ranks all reference genes outside the panel by variance. Only the first 2000 of them in the reference's row order were considered before, so the gene set generally differs; results cached under the old selection are not reused.
- **SpaGE always imputes from the original counts.** The documented "SpaGE on SPLIT-corrected counts" never existed in the Impute Genes dialog, and the REPL form stored its result as ordinary `[imp]` genes, replacing earlier ones. `run_spage(use_corrected=True)` is now refused; `functions/impute.py --use_corrected` remains the way to do it.
- **Imputed genes under "SPLIT Corrected"** show their imputed values in every session. After a restart they showed all zeros, because the stored corrected layer has empty padding columns for them.
- **Imputed-gene colour scale** is log1p on every path. Genes served from the SpaGE cache instead of the store (subset active, no store, streamed results of large datasets) were shown untransformed, under a colour bar labelled log1p.
- **`functions/impute.py`** writes imputed genes under their plain names with `var["is_imputed"]`, as the app does; it used to name them `GENE [imp]` / `GENE [corr+imp]`, which the app listed as `GENE [imp] [imp]` and could not look up. Stores with the old names are read under the plain names.
- **Annotation, RCTD, SPLIT and SpaGE caches** are tied to the reference file's path, size and modification time, not only its name. An edited reference, or another file with the same name, is computed afresh instead of returning the old labels; existing caches are not found once and are recomputed. SPLIT's own RCTD run no longer shares a cache file with an annotation-dialog RCTD run that has the same UMI thresholds.
- **RCTD annotations at start-up**: labels made on a Baysor/Proseg run from the annotation dialog are restored (they were loaded under a name nothing read). When several cached RCTD runs exist for one dataset, the most recently written one is used, and the RCTD weights (pie charts) are taken only from that same run; weights were not restored at all before.
- **SPLIT under an active `subset()`** no longer rewrites the store: it replaced a stored whole-dataset correction with zero rows for every cell outside the subset. The subset result is kept in memory, and an existing whole-dataset correction keeps its rows for the other cells.
- **SPLIT status line** reports the number of cells whose counts changed and how many of the submitted cells RCTD classified, instead of the number of cells submitted.
- **Save as SpatialData** exports `split_umap_1/2` (and a re-run `umap_1/2`), and leaves cells without a SPLIT cluster, ROI or niche without a value. They were written as members of a group called `Unknown`.
- **`subset()` / `unsubset()`** apply to every loaded Baysor/Proseg run, not only to Proseg when both tools hold a result. Area filters on a bundle without area columns are skipped with a message, and a filter that fails no longer leaves the app marked "SUBSET ACTIVE".
- **Spatial niches computed under a subset** are kept after `unsubset()` (cells outside the subset have no niche).
- **Load Sample** removes the extra side-by-side samples and the CellNEST results of the previous dataset.
- **RCTD weight pies in the Plotly view** are drawn. Their wedges were sent as SVG arcs, which plotly.js does not draw, so each wedge was an invisible zero-area line.
- **Side-by-side samples (Plotly view)**: Color by Cluster no longer fails when an added sample has no cluster sets (its cells form one group, e.g. `S1:C0`); the dotted separator sits midway in the gap between two samples instead of inside the first one when the added sample's coordinates do not start at 0; clicking a cell of an added sample fills the cell-info panel instead of reporting the cell as not found.
- **Unavailable colours** are handled the same way in both views (see [WebGL view](#webgl-view-beta)): a gene missing from the dataset no longer raises an error in the Plotly view, and the WebGL view shows grey **No annotation** cells where it used to show transcript counts.
- **SPLIT-corrected counts with several samples** are used when every loaded sample has them and for no sample otherwise; "SPLIT Corrected" is disabled in the second case, with a note naming the samples without them. They were ignored for all samples before, while the option stayed selectable.
- **Transcript density with several samples** draws a grid for every sample that has transcripts, on one colour scale; it showed the primary sample only.
- **Cell types of added samples** are read from the annotation cache or from the `cell_type_<method>` columns of an exported store; added samples were always drawn as "Unknown".
- **Morphology overlay with tifffile ≥ 2025.5.21** works again (the Plotly view reported "morphology images not found": tifffile's Zarr store changed its interface).
- **Cache: Clean** is refused while a background job is running.
- **Bind address** is read from `XENIUM_HOST`; `HOST` is ignored. A container that set `HOST=0.0.0.0` must set `XENIUM_HOST=0.0.0.0`.
- **REPL in local mode** (`XENIUM_ENABLE_REPL=1`) additionally requires a loopback `Host` header and refuses forwarded requests, so it no longer works through a reverse proxy; use `remote` there, on a trusted network.

Behaviour changes that came with the jobs panel (no effect on computed values):

- **Baysor/Proseg run directories** are written as `<name>.partial` and renamed when the run is complete. A failed or cancelled run leaves nothing in the cache (it used to leave the tool's partial output in the run directory, which the Segmentation dropdown could list as a finished run once `segmentation.csv` / `cell-metadata.csv.gz` existed).
- **The CellNEST database cache** (`cellnest_db/`) follows `XENIUM_CACHE_DIR`. It used to be created in `~/.xenium_explorer_cache/` regardless; with `XENIUM_CACHE_DIR` set, the database files are fetched once more into the new location (or move the directory there).
- **Baysor and Proseg run in their own process group**, Proseg on a pseudo-terminal (so that it draws the progress bar the panel reads). Their log lines in the server log are unchanged; Proseg's progress-bar redraws are not logged.
- `subset()` with only skipped filters is a no-op (it used to flag an "N / N cells" subset); **No niche** replaces `Niche nan` in legends; the co-occurrence and enrichment heat maps honour `width` / `height` when called with them.
