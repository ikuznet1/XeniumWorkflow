# Xenium Visualizer

An interactive web application for visualizing and analyzing 10x Genomics Xenium spatial transcriptomics data. Supports cell clustering, gene expression, morphology overlays, cell type annotation, gene imputation, resegmentation, and ambient RNA correction (SPLIT) — all in a browser-based interface.

Works with standard Xenium panels (a few hundred genes) and with **Xenium Prime 5K** output. A GPU-accelerated **WebGL view** (beta) is available alongside the default Plotly view for large datasets; see [WebGL view](#webgl-view-beta).

## Cloud Run Deployment

The app can be deployed to **Google Cloud Run** for remote access. The container files (`Dockerfile`, `requirements-cloud.txt`, `.dockerignore`) are **not currently in this repository**; the commands below assume you supply a `Dockerfile` that installs the Python dependencies, copies `xenium_explorer.py`, `xenium_viewer_api.py`, `assets/` and `functions/`, and starts `python xenium_explorer.py`.

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
  --max-instances 1 --concurrency 1 \
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

**Cloud Run limitations:** R/Julia/Rust features (Seurat, RCTD, SpaGE, Baysor, Proseg, SPLIT) are unavailable in a Python-only container image. CellTypist annotation works.

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
| **Cell Type Abundance** | Bar chart of cell type proportions | W × H dimension inputs; download button (SVG) |
| **Co-occurrence** | Spatial co-occurrence score heatmap | Colorbar min/max, W × H; download |
| **Neighborhood Enrichment** | Neighborhood enrichment heatmap: log2 of the fraction of a cell type's neighbours that are of each type over that type's overall frequency (0 = random mixing) | Colorbar min/max, W × H; download |
| **Interaction Graph** | Node-edge graph of cell type contacts | O/E normalize toggle, W × H; download |
| **Chord Diagram** | Chord diagram of cell type contacts | O/E normalize toggle, W × H; download |
| **Niche UMAP** | UMAP (or spatial map) colored by spatial niche | W × H; download |
| **UMAP** | UMAP plot (alternative to sidebar toggle) | W × H; download |
| **CellNEST CCC** | Cell-cell communication from CellNEST output | See below |

A **download button** (camera icon) appears on hover over each plot; exports as SVG at the current display size.

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
| `rpy2` + R ≥ 4.3 | Seurat label transfer, SpaGE, RCTD, SPLIT |

**R packages** (via rpy2):

| Package | Source |
|---------|--------|
| `SeuratObject`, `Matrix` | CRAN / Bioconductor |
| `spacexr` | `devtools::install_github('dmcable/spacexr')` |
| `SPLIT` | `remotes::install_github('bdsc-tds/SPLIT')` |

**WebGL viewer:** deck.gl, Viv, geotiff and an OpenJPEG WebAssembly decoder are bundled into `assets/xenium_viewer.js`, `assets/xenium_image_worker.js` and `assets/openjpegwasm_decode.wasm`. Nothing needs installing to use them.

**External CLI tools:**
- [Baysor](https://github.com/kharchenkolab/Baysor) — Julia-based resegmentation; binary at `~/.julia/bin/baysor`
- [Proseg](https://github.com/dcjones/proseg) — Rust-based probabilistic resegmentation; `conda install bioconda::rust-proseg` or `cargo install proseg`


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
node test/run.mjs                                  # viewer core against dev_server.py
node image_test/verify.mjs                         # image layer: decode and placement
node test/app_smoke.mjs http://127.0.0.1:8050 out/ # the real app, started with XENIUM_RENDERER=webgl
node test/glue_checks.mjs http://127.0.0.1:8050    # renderer switching, selection and image handling in the real app
node test/jobs_panel.mjs http://127.0.0.1:8050 out/ # jobs panel: starts Proseg on cached patches, checks progress, clicks Cancel
node test/webgl_fallback.mjs http://127.0.0.1:8050 # automatic fallback to Plotly without WebGL2 (browsers started with WebGL disabled)
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
- **Cache: Clean** is refused while a background job is running.
- **Bind address** is read from `XENIUM_HOST`; `HOST` is ignored. A container that set `HOST=0.0.0.0` must set `XENIUM_HOST=0.0.0.0`.
- **REPL in local mode** (`XENIUM_ENABLE_REPL=1`) additionally requires a loopback `Host` header and refuses forwarded requests, so it no longer works through a reverse proxy; use `remote` there, on a trusted network.

Behaviour changes that came with the jobs panel (no effect on computed values):

- **Baysor/Proseg run directories** are written as `<name>.partial` and renamed when the run is complete. A failed or cancelled run leaves nothing in the cache (it used to leave the tool's partial output in the run directory, which the Segmentation dropdown could list as a finished run once `segmentation.csv` / `cell-metadata.csv.gz` existed).
- **The CellNEST database cache** (`cellnest_db/`) follows `XENIUM_CACHE_DIR`. It used to be created in `~/.xenium_explorer_cache/` regardless; with `XENIUM_CACHE_DIR` set, the database files are fetched once more into the new location (or move the directory there).
- **Baysor and Proseg run in their own process group**, Proseg on a pseudo-terminal (so that it draws the progress bar the panel reads). Their log lines in the server log are unchanged; Proseg's progress-bar redraws are not logged.
- `subset()` with only skipped filters is a no-op (it used to flag an "N / N cells" subset); **No niche** replaces `Niche nan` in legends; the co-occurrence and enrichment heat maps honour `width` / `height` when called with them.
