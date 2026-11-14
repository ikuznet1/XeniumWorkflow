# Xenium Visualizer

An interactive web application for visualizing and analyzing 10x Genomics Xenium spatial transcriptomics data. Supports cell clustering, gene expression, morphology overlays, cell type annotation, gene imputation, resegmentation, and ambient RNA correction (SPLIT) — all in a browser-based interface.

Works with standard Xenium panels (a few hundred genes) and with **Xenium Prime 5K** output. A GPU-accelerated **WebGL view** (beta) is available alongside the default Plotly view for large datasets; see [WebGL view](#webgl-view-beta).

## Contents

- [Installation](#installation)
  - [Option A — Conda (recommended)](#option-a--conda-recommended)
  - [Option B — pip only](#option-b--pip-only)
- [Usage](#usage)
- [Worked example: one section from load to methods text](#worked-example-one-section-from-load-to-methods-text)
- [Which analysis answers which question](#which-analysis-answers-which-question)
- [Methods and provenance record](#methods-and-provenance-record)
- [Headless batch mode](#headless-batch-mode)
- [Input Data](#input-data)
  - [AnnData `.h5ad` (processed cells, one or several sections)](#anndata-h5ad-processed-cells-one-or-several-sections)
  - [SpatialData zarr store](#spatialdata-zarr-store)
- [Statistical conventions](#statistical-conventions)
- [Features](#features)
  - [Sidebar](#sidebar)
  - [Visualization](#visualization)
  - [WebGL view (beta)](#webgl-view-beta)
  - [Overlays](#overlays)
  - [Plot Tab (right panel)](#plot-tab-right-panel)
  - [Differential expression](#differential-expression)
  - [Samples and conditions](#samples-and-conditions)
  - [Pseudobulk differential expression across samples](#pseudobulk-differential-expression-across-samples)
  - [Proximity differential expression](#proximity-differential-expression)
  - [Contact-group pseudobulk differential expression](#contact-group-pseudobulk-differential-expression)
  - [Composition between conditions](#composition-between-conditions)
  - [Segmentation-free abundance](#segmentation-free-abundance)
  - [Transcript factors (segmentation-free factorisation)](#transcript-factors-segmentation-free-factorisation)
  - [Gene-set scores](#gene-set-scores)
  - [Transcript density](#transcript-density)
  - [Spatially variable genes (Moran's I)](#spatially-variable-genes-morans-i)
  - [Ligand–receptor scores](#ligandreceptor-scores)
  - [Ligand–receptor exposure decomposition](#ligandreceptor-exposure-decomposition)
  - [Confound controls](#confound-controls)
    - [Tissue compartments](#tissue-compartments)
    - [Spillover score and gene ownership](#spillover-score-and-gene-ownership)
    - [Leak gate in Differential Expression](#leak-gate-in-differential-expression)
  - [Segmentation QC](#segmentation-qc)
  - [Stratified co-localisation with matched nulls](#stratified-co-localisation-with-matched-nulls)
  - [Aggregates](#aggregates)
  - [Banksy-style spatial domains](#banksy-style-spatial-domains)
    - [Cluster sets computed in the app](#cluster-sets-computed-in-the-app)
  - [Joint clustering across samples](#joint-clustering-across-samples)
  - [Working at study scale (a million cells and more)](#working-at-study-scale-a-million-cells-and-more)
  - [CellNEST Cell-Cell Communication](#cellnest-cell-cell-communication)
  - [Cell Info Panel](#cell-info-panel)
  - [Resegmentation](#resegmentation)
    - [Image-based resegmentation (Cellpose / classical)](#image-based-resegmentation-cellpose--classical)
  - [Cell Type Annotation](#cell-type-annotation)
  - [SpaGE Gene Imputation](#spage-gene-imputation)
  - [SPLIT Ambient RNA Correction](#split-ambient-rna-correction)
  - [Save as SpatialData](#save-as-spatialdata)
  - [Cell Subsetting](#cell-subsetting)
  - [Jobs panel: progress and Cancel](#jobs-panel-progress-and-cancel)
  - [Sessions: save and restore](#sessions-save-and-restore)
  - [Server Log & Python REPL](#server-log--python-repl)
- [Caching](#caching)
- [Standalone Functions (`functions/`)](#standalone-functions-functions)
  - [`functions/impute.py` — SpaGE gene imputation](#functionsimputepy--spage-gene-imputation)
  - [`functions/spatialdata2seurat.py` — SpatialData → Seurat RDS](#functionsspatialdata2seuratpy--spatialdata--seurat-rds)
  - [`functions/merge_spatialdata.py` — Merge multiple SpatialData zarrs](#functionsmerge_spatialdatapy--merge-multiple-spatialdata-zarrs)
  - [Other functions](#other-functions)
    - [`functions/geo_to_bundle.py` — open a GEO deposit](#functionsgeo_to_bundlepy--open-a-geo-deposit)
- [Dependencies](#dependencies)
- [Cloud Run Deployment](#cloud-run-deployment)
  - [Environment variables](#environment-variables)
- [Continuous integration](#continuous-integration)
- [Code layout](#code-layout)
- [Developing the WebGL viewer](#developing-the-webgl-viewer)
- [Changes that affect results (November 2026)](#changes-that-affect-results-november-2026)

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
XENIUM_PLOTLY_MAX_CELLS=1500000 python xenium_explorer.py …   # cells the Plotly map may draw (default 900,000)
XENIUM_ENABLE_REPL=1  python xenium_explorer.py <bundle>     # enable the in-app Python REPL
XENIUM_R_INPROCESS=1  python xenium_explorer.py <bundle>     # run R inside the app instead of in a worker process (see Jobs panel)
python xenium_explorer.py path/to/exported.zarr              # open a store written by "Save as SpatialData"
python xenium_explorer.py path/to/study.h5ad                 # an AnnData file, one sample per section (see below)
```

To stop the server:
```bash
lsof -ti:8050 | xargs kill -9
```

All start-up variables are listed under [Environment variables](#environment-variables).

## Worked example: one section from load to methods text

One pass through the app on public data: the Xenium Prime 5K human skin (melanoma, FFPE) section that 10x Genomics publishes on the datasets page of its website (panel "Xenium Human 5K Pan Tissue & Pathways"; download the output bundle and unpack it). 112,551 cells, 5,006 genes. Every step below was run as written; the times are from one machine (Apple M3 Max, 16 cores, 48 GB) with other jobs running, and say how long the analysis took, not how fast yours will be. The bundle had been opened once before, so the store the app writes beside the data on first load was already there.

| # | Step | Where to click | What to expect | Time | The caveat that matters |
|---|------|----------------|----------------|------|-------------------------|
| 1 | Load | `python xenium_explorer.py <bundle>`, open http://localhost:8050 | The section coloured by cluster | server ready in 5 s, page drawn in 7 s | The app loads one bundle per process; "+ Add sample" adds more |
| 2 | Look around | Pan and zoom the map; **Plotly / WebGL (beta)** under the map switches the renderer (or start with `XENIUM_RENDERER=webgl`) | The same cells in both views | page drawn in 7 s in either | The WebGL view is a beta: see [WebGL view](#webgl-view-beta) for what it does not draw |
| 3 | Annotate | Sidebar → **Annotate Cells** → CellTypist, model `Immune_All_Low.pkl` → **Run Annotation** | "Done — 13 cell types"; the annotation becomes a label source (`cell type (CellTypist)`) in every tab | 81 s | CellTypist needs no reference, but the model must fit the tissue. This one is an immune model: on skin it called 82,413 of 112,551 cells "Tcm/Naive helper T cells". The steps below show the mechanics on these labels; they are not a skin annotation |
| 4 | Compare segmentations | QC → Segmentation QC → **Run** | One row per segmentation source and counts mode. With the bundle alone: 1 row (Xenium). On a copy of the same data with cached Baysor, Proseg and image-based runs of small regions: 13 rows for 11 sources | 2.7 s (1 row), 3.5 s (13 rows) | Baysor and Proseg are optional tools. A run without its own annotation gets the Xenium label of the nearest centroid, which favours segmentations close to Xenium's; a region run and the whole section are not the same cells |
| 5 | Compartments | QC → Compartments & spillover → Cell types from: `cell type (CellTypist)` → **Compute compartments** | Edge distance, local density and dense core per cell (median edge distance 335 µm; 63.6 % core, 28.9 % edge), offered under Color By and used as strata later | 0.2 s | The columns say where a cell sits, not what it is; the "core" is that of the dominant type, here the mislabelled majority |
| 6 | Spillover and ownership | same tab → **Run** | Spillover score of every cell for the source type, gene ownership table (4,374 of 5,006 genes have an owner) | 0.6 s | Read the validity check. Here it said: markers by FALLBACK (fewer than 3 genes are 10× specific), distance gradient not testable. The score is then not a contamination mask |
| 7 | DE with the leak gate | Expression → Differential expression → A: `Cell type (CellTypist): Tem/Trm cytotoxic T cells`, B: `All other cells` → **Run** | Volcano and table: 4,070 vs 108,481 cells, 3,707 of 5,006 genes with FDR < 0.05. Leak gate: 765 genes flagged; of the 3,707, 501 are flagged and 1,360 are not owned by the focal type | 1.6 s | The unit is the cell (n = 112,551, one section): the p-values rank genes, they do not establish anything. The status line also warned that the groups differ 2.3× in depth |
| 8 | Gene set | Expression → Gene sets → name `Tcell`, genes `CD3E CD2 CD8A GZMK` → **Save & score** | "4 genes scored against 200 control genes"; pick `Gene set: Tcell` under Color By | under 2 s | Genes not on the panel are named and left out; a score is not a test |
| 9 | Spatial genes | Expression → Spatial genes → **Run** | Moran's I per gene; top of the table: DSG1 (I = 0.88), ARG1, CDSN | 1.1 s | 4,051 of 5,006 genes have FDR < 0.05: with 10⁵ cells nearly everything is significant. Rank by I |
| 10 | Co-localisation | Neighbourhoods → Co-localisation → Cell types from: `cell type (CellTypist)` → **Run** | 156 ordered pairs of 13 types against a stratum-matched expectation | 3.0 s | The status says SINGLE-SAMPLE RESULT: within-section permutation p-values, smallest achievable p 0.005, and FDR 0.05 reachable only if 16 tests sit at that floor. It describes this section only |
| 11 | Export | Provenance → Records & methods → **Methods text (.md)** and **Full record (.json)** | 8 records (one per analysis above); a 19 kB methods text with parameters, units, caveats, versions and references | immediate | The text states what was run. Read it against your study, and check the references marked [verify] |

The same steps without a browser: [Headless batch mode](#headless-batch-mode) ran them on this section: the annotation config took 68 s including start-up (62 s of it the annotation), and steps 4 to 10 together took 17 s of compute.

### Tests between conditions need replicates

Pseudobulk differential expression, the composition test, contact-group DE and the tests across samples of co-localisation, aggregates and the ligand–receptor decomposition compare **samples**, not cells. They need real biological replicates: several donors per condition. Sections or regions of one donor are not replicates, and no setting of the app turns them into replicates.

Add the samples with **+ Add sample**, then open **Samples & conditions…** (above every tab that uses the design) and give each sample a condition and, when you have them, a donor and a batch:

| Sample | Condition | Donor | Batch |
|--------|-----------|-------|-------|
| S0 | A (treated) | d1 | |
| S1 | A (treated) | d2 | |
| S2 | A (treated) | d3 | |
| S3 | B (control) | d1 | |
| S4 | B (control) | d2 | |
| S5 | B (control) | d3 | |

With a donor on every sample the donors are the units; a donor in both conditions makes the design paired (`~ donor + condition`). Three units per condition is the minimum for a test, and the status line then tells you the smallest p-value those units can produce: with three paired donors the exact sign-flip test cannot go below 0.25, whatever the data. We ran this layout on six spatial tiles cut from the one skin section, which exercises the code and nothing else (they are pseudo-replicates): pseudobulk DESeq2 of one cell type took 8.4 s, the composition test reported "smallest achievable p: 0.25, 0 with FDR < 0.05", as it must.


## Which analysis answers which question

Locations are given as `Group → Tab` of the plots panel. Units of inference are explained under [Statistical conventions](#statistical-conventions); "none" means the result is descriptive and carries no p-value.

| Question | Analysis | Where in the app | Unit of inference |
|----------|----------|------------------|-------------------|
| Which genes differ between two groups of cells? | [Cell-level differential expression](#differential-expression) (Wilcoxon) | Expression → Differential expression (cells) | Cells, treated as independent: rank by effect size |
| Do the conditions differ, with replicates? | [Pseudobulk differential expression](#pseudobulk-differential-expression-across-samples) (DESeq2, else edgeR) | Expression → Differential expression (pseudobulk) | Samples / donors |
| Which genes of a cell type change near another cell type? | [Proximity differential expression](#proximity-differential-expression) | Expression → Differential expression (proximity) | Donors / samples with ≥ 3 units; with one sample, matched random anchor sets (single-sample result) |
| Do cells in contact with another type differ from cells of the same type that are not? | [Contact-group pseudobulk differential expression](#contact-group-pseudobulk-differential-expression) | Expression → Differential expression (contact) | Samples (donor blocks) |
| Does the proportion of a cell type differ between conditions? | [Composition between conditions](#composition-between-conditions) | Overview → Composition | Samples / donors |
| Is this cell type's signal contamination from neighbours? | [Spillover score, gene ownership](#spillover-score-and-gene-ownership) and the [leak gate](#leak-gate-in-differential-expression) | QC → Compartments & spillover; the leak gate flags genes in Expression → Differential expression | None (scores and flags) |
| Is a cell at the tissue margin or in a dense core? | [Tissue compartments](#tissue-compartments) | QC → Compartments & spillover | None (covariates and strata of other tests) |
| Which segmentation is cleaner? | [Segmentation QC](#segmentation-qc) | QC → Segmentation QC | None |
| Do I need segmentation at all for abundance? | [Segmentation-free abundance](#segmentation-free-abundance) | QC → Segmentation-free | Concentration: tiles of one sample. Between conditions: samples / donors |
| Where do expression programs lie without using any cell, and where does a cell mix them? | [Transcript factors](#transcript-factors-segmentation-free-factorisation) | QC → Transcript factors | None |
| Is a gene set high in some cells? | [Gene-set scores](#gene-set-scores) | Expression → Gene sets | None (a score per cell) |
| Which genes are spatially patterned? | [Moran's I](#spatially-variable-genes-morans-i) | Expression → Spatial genes | Cells, per sample: rank by I |
| Where do the transcripts of a gene lie, independent of the segmentation? | [Transcript density](#transcript-density) | Expression → Transcript density | None |
| Which cell types are neighbours? | Co-occurrence, Neighborhood Enrichment, Interaction Graph, Chord Diagram ([Plot Tab](#plot-tab-right-panel)) | Neighbourhoods → Co-occurrence / Enrichment / Graph / Chord | None |
| Are two cell types closer than their tissue location explains? | [Stratified co-localisation with matched nulls](#stratified-co-localisation-with-matched-nulls) | Neighbourhoods → Co-localisation | Within a sample: cells of one section. Across samples and between conditions: samples / donors |
| Where are the immune aggregates? | [Aggregates](#aggregates) | Neighbourhoods → Aggregates | Hotspot detection: cells of one section. Between conditions: samples / donors |
| What tissue domains are there? | [Banksy-style spatial domains](#banksy-style-spatial-domains) | Neighbourhoods → Banksy-style | None (exploratory partition) |
| Are the clusters comparable across samples? | [Joint clustering across samples](#joint-clustering-across-samples) | Neighbourhoods → Joint clusters | None |
| What spatial niches do the cells form? | Spatial niches ([Visualization](#visualization), [Plot Tab](#plot-tab-right-panel)) | Neighbourhoods → Niches | None |
| Which ligand–receptor pairs are expressed in neighbouring cell types? | [Ligand–receptor scores](#ligandreceptor-scores) | Communication → Ligand–receptor | Cells (cell-type labels permuted) |
| Why does the exposure of a receiver type to a ligand differ between conditions? | [Ligand–receptor exposure decomposition](#ligandreceptor-exposure-decomposition) | Communication → Ligand–receptor (Mode → Exposure decomposition) | Samples / donors |
| What does CellNEST infer for this tissue? | [CellNEST Cell-Cell Communication](#cellnest-cell-cell-communication) (results computed outside the app) | Communication → CellNEST | None computed in the app |

## Methods and provenance record

Every analysis the app runs leaves a structured record, and one click exports the lot.

**What is recorded.** When a job publishes its result it writes one record: the analysis kind and
title; the time; the data (dataset, samples with their cell and gene numbers, the segmentation
source with its run tag and the run's stored parameters, the counts mode asked for and used and
the number of cells left out in corrected mode, an active `subset()` with its filters, the label
source); every parameter (those you set and the defaults that were used); the statistical unit
and n **as the analysis' own status line states them** (the record copies the sentence, it does
not derive it again); the sample design with your own labels; headline numbers (table sizes,
tests run, FDR < 0.05); every caveat the analysis raised (minimum achievable p, non-integer
counts, pooled samples, cells left out, leak gate, failed validity checks …), verbatim; random
seeds; whether the result was computed or came from the cache; the run time; and the software
versions (the app's git commit, Python, numpy / scipy / pandas / scikit-learn, the packages of
the method, R and the R packages for jobs that used R, the version string of Baysor / Proseg).
A job that fails, is cancelled or belongs to a dataset that was replaced writes nothing.

All analysis kinds report: Baysor, Proseg and image-based segmentation runs (and loading a cached
run), patch preparation, annotation (CellTypist, Seurat kNN, RCTD), SPLIT, SpaGE, clustering /
UMAP, spatial analysis, niches, the four differential expression modes, gene sets, transcript
density, composition, Moran's I, ligand–receptor scores and decomposition, compartments,
spillover / ownership, Segmentation QC, co-localisation, aggregates, segmentation-free
abundance, Banksy-style domains, joint clustering, transcript factors, `.h5ad` conversion,
loading CellNEST results and Save as SpatialData. A test reads the source and fails when a job
is registered that does not call the one helper (`provenance._prov_record`).

**Where it is kept.** One append-only JSON-lines file per dataset under
`<cache>/provenance/`; it survives a restart and "Clean cache", and is trimmed to its newest
1,500 records once it passes 4 MB or 2,000 records.

**Provenance tab** (Provenance group). The records, newest first; sortable, with a filter by
kind; click a row for the full record. The tab reads the file when it is opened or refreshed;
nothing is recomputed.

**Export.** *Full record (.json)* saves every record. *Methods text (.md)* writes a methods
section: a data paragraph, then one section per analysis (the newest run of each) from a
template that states the method, the parameters, the unit and n as the app stated them, the
multiple-testing procedure, the software versions and each caveat the app raised, followed by
the analysis' status line verbatim, the references of the published methods used, and a table
of all records. The text says what was done and no more: a cell-level test is described with
"cells were treated as independent observations", the Banksy-style domains as a native
implementation "not validated against the reference implementation". References the authors of
this app could not confirm in full carry "[verify]": check them before citing. Read the text
against your study before you use it.

**CSV files.** Every CSV the app writes names its record in the header
(`# provenance record: prov-…`), so a table can be traced to the run that made it. A result
restored from the cache of an earlier session says so instead.

Limits: the record of an analysis describes the state when it ran (a later change of the design
or the subset is not written back); parameters of interactive steps that are not jobs (drawing
an ROI, the sample design editor) appear only through the analyses that used them; tool
versions are recorded for runs made in this session, a cached run loaded later shows the
parameters stored with it.


## Headless batch mode

Run analyses from a config file, without a browser:

```bash
python -m xenium_app.batch config.yaml              # run
python -m xenium_app.batch config.yaml --dry-run    # validate against the data, compute nothing
python -m xenium_app.batch config.yaml --list-steps # every step kind, its parameters and defaults
```

The config is YAML (PyYAML is used when it is installed) or JSON. No web server is started and
no port is bound.

```yaml
data:
  path: /data/section1              # a Xenium bundle directory, a .zarr store or an .h5ad file
  extra_samples: [/data/section2]   # optional: further samples (bundles or .zarr)
  h5ad_mapping: mapping.json        # optional, for an .h5ad: the column mapping (file or inline)
  design:                           # optional: needed by the tests between conditions
    names: {A: treated, B: control}
    samples:
      - {sample: S0, condition: treated, donor: d1}    # sample: S0 / S1 …, a name or a path
      - {sample: S1, condition: control, donor: d1}
cache_dir: /scratch/cache           # the app's cache; also holds the provenance records
session: my_session.json            # optional: a saved session applied before the steps (see Sessions)
output_dir: /scratch/out
seg_source: xenium                  # or a cached run: baysor:<tag>, proseg:<tag>, imgseg:<tag>
counts: original                    # or corrected (SPLIT)
continue_on_error: false
steps:
  - kind: annotation
    params: {source: celltypist, model_name: Immune_All_Low.pkl}
  - kind: de
    params: {group_a: "celltype:Fibroblasts", group_b: rest}
  - kind: moran
```

Step kinds: `annotation`, `clustering`, `compartments`, `spillover`, `segqc`, `de`, `pseudobulk`,
`proximity`, `contact`, `geneset`, `density`, `composition`, `moran`, `lr`, `lrdecomp`, `coloc`,
`aggregates`, `segfree`, `banksy`, `joint`, `txfactor`, `spatial`, `niches`,
`export_spatialdata`. Resegmentation runs, SPLIT and SpaGE are not batch steps; a cached
segmentation run can be the `seg_source`.

**One implementation.** A step calls the function its Run button calls in the app (the Dash
callback, as a plain function): the same validation, the same argument handling, the same job
through the job registry. A parameter is named as that function's argument (`--list-steps`
prints them with the control each belongs to); one you leave out takes the default of its
control in the app. `labels` names "where the cell types come from" in every step that has it
(`cluster`, `celltype:<method>`). A test runs Moran's I and the seeded transcript factors once
through the Run callbacks and once through the command and compares the tables exactly.

**Output.** One folder per step (`01_<name>/` …) with: the CSV of the tab's Download button
(with its comment header), every table of the result as Parquet, per-cell results as
`cells.parquet` (compartments, gene-set scores, annotation labels, niches, and for a `clustering`
step the cluster columns with the embedding the step computed: `umap_1/2`, or `split_umap_1/2` with
corrected counts), the tab's figures,
`status.txt` (the analysis' status line) and `provenance.json`; at the top `provenance.json` and
`methods.md` for the run (see [Methods and provenance record](#methods-and-provenance-record))
and `batch_summary.json`. Figures are written as PNG when `kaleido` is installed; otherwise as
`.html` and `.plotly.json`, and the summary says so. Progress goes to stdout.

**Segmentation and design.** A step runs on the cells its `seg_source` names (the top-level one,
or its own `seg_source` parameter): a cached run is loaded before the step when another run of
that tool, or none, is in memory, and a run that does not exist is an error — a step never falls
back to the Xenium cells or to another run. `data.design` is the design of the run: it replaces
the design saved for these samples (a sample it does not list is unassigned; a listed sample
without `donor` or `batch` has none). A `sample:` entry is matched to a loaded sample by its label
first (`S0`, `S1`, …), then by its path, then by its name or folder name; a name that fits several
samples, and two entries that name the same sample, are errors (name the sample by label or path).
With `--dry-run` the design is applied in memory for the checks only: the design saved for these
samples is not replaced.

**Failures.** A failed step stops the run with the step's own message and exit code 1; with
`continue_on_error: true` the remaining steps run and the exit code is still 1. Exit code 2: the
config could not be used. `--dry-run` loads the data — and the cached segmentation run a step
names, whose labels and clusters the checks are made on — and checks parameter names, label
sources, cell types, genes, ligand–receptor pairs, DE groups, the design and the transcript file,
and exits 1 when it finds an error; a label source that an earlier `annotation` step of the same
config will create is reported as a warning, and so is `noninteger_counts: round_sums` (the
pseudobulk steps then run on counts that are not raw). A real run prints the same errors and is
never reported as ok when the config has one; a step that returns without a result of its own
(a required parameter is missing) fails.

**Example.** `examples/batch_tiny.yaml` runs five steps on the tiny test bundle; its header
gives the three commands (copy the fixture to a temporary directory first: the app writes a
store beside the data it loads).

The package builds the Dash app object when it is imported (the callbacks are its functions);
this costs about 0.4 s of a 4 s start on the test bundle and was kept. `import xenium_explorer`
is unchanged.


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

### AnnData `.h5ad` (processed cells, one or several sections)

A processed AnnData file can be loaded in place of a Xenium output directory: a cell × gene
matrix with coordinates in obs, optionally a column that says which section (sample) each cell
belongs to. Each section becomes a sample of the multi-sample view (the first one primary, the
others as with "+ Add sample"), so everything that works on cells works: the map in both
renderers, Color By cell type / cluster / gene, differential expression, pseudobulk, composition,
co-localisation, compartments, spillover / ownership / leak gate, proximity and contact DE,
ligand–receptor scores and decomposition, aggregates, Moran's I, Banksy-style domains, joint
clustering, Save as SpatialData.

**The column mapping** says where things are in the file:

| Key | Meaning |
|-----|---------|
| `x`, `y` | obs columns with the cell coordinates **in µm** (or `spatial`: an obsm key; default `obsm["spatial"]` when no x / y columns are found) |
| `section` | obs column naming the section / sample of each cell; without it the file is one sample |
| `cell_types` | obs columns with cell-type labels (any number); each appears under Color By as "Cell Type (<column>)" and as a grouping wherever cell types are used |
| `clusters` | obs columns to show as cluster sets (Clustering Method); the same id means the same cluster in every section |
| `condition`, `donor` | obs columns that prefill **Samples and conditions**: the donor of every section, and A / B for the two levels compared (`cond_a`, `cond_b`; with exactly two levels they are taken as they are; with more, name the two, or assign in the dialog). Sections of other levels are excluded |
| `counts` | the matrix with the **raw counts**: `X` or the name of a layer (default: the layer `counts` when there is one) |
| `umap` | obsm key of a 2-D embedding shown in the UMAP panel (default `X_umap`) |
| `noninteger_counts` | `refuse` (default) or `round_sums`, see below |

Give it either way:

```bash
# 1. a JSON file beside the data (study.h5ad.json), or anywhere via XENIUM_H5AD_MAPPING
echo '{"x": "x", "y": "y", "section": "sample", "cell_types": ["cell_type"],
       "condition": "group", "donor": "donor", "counts": "counts",
       "cond_a": "treated", "cond_b": "control"}' > study.h5ad.json
python xenium_explorer.py study.h5ad

# 2. environment variables (override the JSON; lists are comma-separated)
XENIUM_H5AD_SECTION=sample XENIUM_H5AD_CELLTYPES=cell_type,subtype \
XENIUM_H5AD_CONDITION=group XENIUM_H5AD_DONOR=donor XENIUM_H5AD_COUNTS=counts \
python xenium_explorer.py study.h5ad
# also: XENIUM_H5AD_X, XENIUM_H5AD_Y, XENIUM_H5AD_CLUSTERS, XENIUM_H5AD_UMAP,
#       XENIUM_H5AD_COND_A, XENIUM_H5AD_COND_B, XENIUM_H5AD_NONINTEGER
```

Without either, common column names are recognised (`x` / `y`, `sample` / `section`,
`cell_type…`, `condition` / `group`, `donor` / `patient…`, layer `counts`). In the app:
**Load Different Sample → AnnData .h5ad**: enter the path, *Read columns*, choose the columns in
the pickers, *Load .h5ad*. A mapping that does not fit the file is refused with the list of
columns the file has. `_h5ad_design("a", "b")` in the REPL refills the design for another pair
of condition levels of the same file.

**How it is loaded, and what it costs.** The file is converted once into one SpatialData store
per section under the cache directory (`h5ad_<name>_<hash>/<section>.zarr` and a
`manifest.json`); later starts with the same file and mapping read those stores. The matrix is
never read whole: it is read in blocks of rows straight from the HDF5 file (about 24 million
stored values, ~0.3 GB, at a time), each block is split by section, and a section is written
and released as soon as its last row has passed. With the sections stored one after the other
the peak is one section plus one block; with interleaved sections the pieces are held until
each section is complete (up to the whole matrix as float32). A CSC-encoded matrix cannot be
read by rows and is read whole once (the log says how much memory that takes). In the app every
sample holds its own rows as CSR (float32 values, int32 columns: 8 bytes per stored value in
total, no shared or duplicated copy), plus a CSC copy of a sample the first time a gene of it
is coloured or tested. As an order of magnitude, a file with over a million cells and a few
hundred million stored values in a few dozen sections converts in about a minute into roughly
1 GB of cache on an SSD, starts in well under a minute afterwards and takes about 5 GB of
memory once loaded. Converting once (rather than slicing the .h5ad at every start) is what makes
every sample an ordinary store, with its own cached design, cluster sets and exports. To convert
again (or to free the space), delete the `h5ad_…` directory in the cache directory; *Clean* in
the app does not touch it.

**Counts are checked.** Every stored value of the chosen matrix is tested for being a
non-negative whole number during the conversion. If it is not (log-normalised `X`, or counts
corrected for ambient RNA), the load says so, with the number of offending values, in the
terminal, in the dialog and in the tissue info; the data is still loaded, and pseudobulk
differential expression refuses the matrix. A file that has *only* corrected counts can be
used with `"noninteger_counts": "round_sums"`: the pseudobulk tests then sum the values per
sample and round the sums, and every such result carries a warning that the count model is an
approximation. Nothing is rounded per cell.

**What is not available.** A file has cells only. Transcript density, segmentation-free
abundance, transcript factors, Baysor / Proseg and image-based resegmentation, the morphology
overlay, cell / nucleus outlines and the "fraction of transcripts assigned" metric need the
transcript file or the images of a Xenium output directory: they are disabled or answer with
that reason. Cells without finite coordinates are left out (and counted in the load message).
A sample is one section: when the condition or donor column is not constant within a section
its most frequent value is used, with a warning.

### SpatialData zarr store

On first launch, the app consolidates all cell-level data into a **`spatialdata_xenium.zarr`** file written next to the raw data (one-time, ~30–60 s for a typical dataset). Subsequent launches load exclusively from this zarr — the raw files are no longer read at startup.

The zarr stores:
- `table` — AnnData with the full cell × gene expression matrix, all obs metadata (centroids, QC metrics, UMAP, cluster assignments), and a `is_imputed` flag per gene
- `cell_boundaries` / `nucleus_boundaries` — GeoDataFrame shapes

Images are still read directly from `morphology_focus/` on demand.

A store built before control-feature filtering was added keeps its original feature list and loads unchanged. Deleting and rebuilding it drops the control features, after which older resegmentation, SpaGE and SPLIT caches built against the old gene list no longer line up and should be regenerated.

## Statistical conventions

These rules hold for every analysis of the feature reference below; the feature sections give only what is specific to them. Conditions, donors and batches are assigned under [Samples and conditions](#samples-and-conditions).

**Units of inference.** Every p-value in the Differential Expression, composition, Spatial Genes, Ligand–Receptor and co-localisation tabs and CSVs names its unit, e.g. "n = 112,551 cells, treated as independent" or "n = 6 samples (3 donors, paired)".

- **Cells (or tiles) of one section.** Cell-level differential expression, Moran's I, the ligand–receptor permutation p-value, the within-sample p-value of stratified co-localisation and the hotspot p-value of the aggregates treat the cells as the observations (exchangeable within strata, where strata are used); the concentration p-value of segmentation-free abundance has the tissue tiles of one sample as its unit, and proximity differential expression on a single sample is judged against random anchor sets of that sample. Such a p-value describes that section and says nothing about other sections or donors; single-sample results are labelled as such.
- **Samples or donors.** Pseudobulk differential expression, contact-group pseudobulk differential expression, the composition test, and the sample-level tests of stratified co-localisation, aggregates, the ligand–receptor exposure decomposition, proximity differential expression and segmentation-free abundance take the samples as the observations (the donors, when donor labels are given: next point). A sample that has no value for a test — no labels for the chosen annotation, say — is not counted as a zero: the test is not run on it, and the status line says why.
- **Donor.** The donor is the individual a sample comes from. Samples of one donor are not independent: within a condition they are combined into one unit (averaged; summed for pseudobulk). When every included sample has a donor and at least 2 donors occur in both conditions, the tests are **paired** (`~ donor + condition`; exact sign-flip test); donors seen in one condition only are then left out. Otherwise the donors (or, without donor labels, the samples) are the independent units of an unpaired test (`~ condition`; label permutation). Several sections of one donor are never counted as replicates — also when only some samples have a donor label: the labelled sections of a donor are one unit, and a sample without a label is its own unit (one rule for every sample-level test: `stats._sl_units` / `_sl_unit_groups`).

**Exact sample-level tests.** Paired: the sign-flip test of the per-donor differences — all 2^n sign vectors are enumerated up to 16 donors, 20,000 random vectors with a fixed seed above. Unpaired: the label-permutation test of the difference of means — all C(n, n_A) relabellings up to 200,000, Monte Carlo above. A null that can be enumerated is never sampled. Every result states the **unit and the number of independent units**, the **smallest achievable p** (a property of the design, not of one test's values: 2 / 2^n for n pairs — 0.25 with 3, 0.031 with 6 — and 1 / C(n, n_A) for an unpaired design, 2 / C(n, n_A) when both conditions have the same number of units), and a warning when a single test at that floor would not pass Benjamini–Hochberg at 0.05 (floor × number of tests ≥ 0.05): when the floor itself is ≥ 0.05 no test can be significant, otherwise the warning says how many tests would have to sit at the floor. These tests need at least 3 independent units (per condition, when conditions are compared); with fewer, no sample-level test is run, the reason is given, and the result is labelled descriptive or single-sample. Feature sections state where their limits differ.

**Pseudo-replication.** Cells of one tissue section are not independent, and with hundreds of thousands of cells almost every gene reaches a tiny p-value: rank by effect size (log2FC, % expressing, Moran's I), not by the cell-level p-value. A cell-level comparison across samples or replicates is anticonservative, because the variation between samples is ignored; the proper analysis there is a test with samples as replicates. Replicates must be biological replicates: regions of one tissue section, or serial sections of one block, are pseudo-replicates whose dispersion is far too small, and their p-values mean nothing.

**SPLIT-corrected counts.** Corrected and original counts are never mixed in one colour scale or one test: with several samples loaded, "SPLIT Corrected" is used only when every sample has corrected counts, and then applies to all of them; otherwise the original counts are used for all samples. In corrected mode, cells without corrected counts are **left out, not counted as zeros**: such a cell (empty corrected row, counts in the original) is not part of differential expression, gene-set scores (no score), Moran's I, ligand–receptor scores and decomposition, proximity DE, spillover (no score and no flag), Banksy and joint clustering (no cluster), and the status line of each — and its CSV header, where the analysis has a CSV — states how many cells were left out. Pseudobulk and contact-group pseudobulk differential expression always use the original counts, also when Active Counts is "SPLIT Corrected" (the status line says so): SPLIT-corrected counts are modified, non-integer values, which a count model must not be given. On a Baysor / Proseg run the counts mode also picks the cluster column (the SPLIT clusters in corrected mode) for every analysis that groups by cluster, as on the map. A cell detects a gene when its count is **≥ 1 transcript**, so a corrected residue below one transcript does not count. Rates are Σ counts ÷ Σ depth, which is defined for fractional counts. Imputed genes are not tested: their values are not counts. Details: [SPLIT Ambient RNA Correction](#split-ambient-rna-correction).

**Leak gate and gene ownership.** Gene ownership names, per gene, the cell type with the highest rate and every type at ≥ 20 % of it. The leak gate of the Differential Expression modes flags a gene when the signal predicted to leak in from neighbouring cell types is more than half of the observed signal in either group, or when the leak alone predicts at least half of the observed log fold change (this second rule only for genes whose change is called or is at least 0.5 in log2); the proximity and contact modes also flag genes that the anchor type owns and the tested type does not. Nothing is computed without a cell-type annotation or with **Gene ownership + leak gate** unticked, at least 80 % of the labelled cells of group A must be of one cell type, and a neighbour type whose leak cannot be measured is not modelled and is named. The estimate leans low, so the gate errs towards not flagging: a flag means "check the owner and the neighbours", not "wrong". Reference: [Spillover score and gene ownership](#spillover-score-and-gene-ownership), [Leak gate in Differential Expression](#leak-gate-in-differential-expression).

**Multiple testing.** FDR is Benjamini–Hochberg over the tests of one result (genes, categories, type-pair combinations, interactions; per sample for Moran's I) unless a section names another procedure. A permutation p-value is (1 + permutations at least as extreme) / (1 + permutations), so it cannot be smaller than 1 / (1 + permutations). p-values below the float range are written as `1e-300` instead of `0` in the DE and Moran tables and CSVs; the CSVs keep `neg_log10_pval` and `neg_log10_padj` with the exact values, and "most significant first" orders by them.

**Depth differences.** The DE status line and CSV give the median counts per cell of both groups and warn when they differ by more than 1.5×. The DE table also gives **Rate A / Rate B** (Σ counts ÷ Σ total counts of the group, per 10,000), which does not move with depth.

**Cluster ids across samples.** A statistic that puts the cluster ids of several samples side by side (composition by cluster, pseudobulk "Cluster N", ligand–receptor scores by cluster) reads every sample with the cluster set chosen under *Clustering Method*. A sample that does not hold that set has no cluster labels there and is left out, with a message; the ids of another of its sets are never used in their place. (Whether cluster 3 of one sample is cluster 3 of another still depends on how the set was computed: only a set computed jointly has comparable ids.)

**Library size when panels differ.** An analysis that pools the cells of several samples and tests the genes they share (cell-level differential expression, the pooled ligand–receptor score, the ligand–receptor exposure decomposition, gene ownership with the leak gate and the spillover score, gene-set scores) takes as the library size of a cell its total count over **those shared genes**, the same gene set for every cell. With one panel this is the whole panel and nothing changes. With different panels (a ~500-gene sample beside a 5K sample) a library over each sample's own panel would divide the cells of the larger panel by counts of genes the other sample never measured, lowering every shared gene in that sample for a reason that is not biology. Rate A / Rate B and the median counts per cell of the cell-level test, and the median counts per cell reported with a pseudobulk test, are over the same shared genes. Analyses that work within one sample (Moran's I, the per-sample part of proximity DE) use that sample's own panel. Pseudobulk tests are not affected: their size factors come from DESeq2 / edgeR on the sums over the shared genes.

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

**Layout:**
- The plots row and the bottom info bar are separated by a **vertical drag handle** — drag it to resize the panels.
- The bottom info bar can be **fully collapsed** with the ▼ button (the spatial/UMAP plots expand to fill the space). Click ▲ to restore.

### WebGL view (beta)

**Large data starts here.** The Plotly map ships every cell to the browser inside one figure and
rebuilds it on every change of colour. Measured in headless Chromium with a GPU on a 5,006-gene
section replicated side by side, the first draw took 3 s at 113,000 cells, 4 s at 225,000, 5 s
at 450,000 and 7–10 s at 900,000, and a recolour 2.5, 3.5, 6 and 10 s (a slower machine or a
remote server takes longer). When the loaded samples hold more than `XENIUM_PLOTLY_MAX_CELLS`
cells (default 900,000, where the first draw stayed under about 15 s; 0 = no limit) the app starts in the WebGL view, an `.h5ad` loaded from the dialog switches to
it, and choosing Plotly shows a message with the cell count instead of building the figure.
`XENIUM_RENDERER=plotly` still selects the Plotly switch position at start; the limit applies
all the same.

The WebGL view (deck.gl) keeps cell positions on the GPU and asks the server only for what changes: one value per cell on a recolour, and boundaries, transcripts and image tiles for the current viewport.

Switch with the **Plotly / WebGL (beta)** selector at the bottom left of the spatial panel, or start with `XENIUM_RENDERER=webgl`. Plotly remains the default. Switching back rebuilds the Plotly figures.

What it draws:

- **Cells** for every colour mode, on Xenium, Baysor and Proseg sources and under an active subset, with a legend (click an entry to hide that category) or colour bar. The Colorbar Range inputs apply without a server round trip.
- **UMAP** in the right-hand panel, from the same cells and colours.
- **Boundaries** for the viewport (zoom in when asked; the limit is 20,000 polygons per view): Xenium cell (cyan) and nucleus (orange) outlines from the bundle's `cells.zarr.zip`, plus Proseg (green) and Baysor (purple) outlines when a run is loaded. Each checkbox works under any segmentation source, as in the Plotly view.
- **Transcripts** for genes chosen under *Transcripts (WebGL view)* in the sidebar, read from the bundle's tiled `transcripts.zarr.zip`. Full resolution with a Q ≥ 20 filter when zoomed in; 10x's pre-subsampled levels (no quality scores) when zoomed out, reported in the status line.
- **Morphology images** read in the browser by HTTP range requests from the pyramidal OME-TIFFs and decoded in web workers, with per-channel colour, the brightness slider as a contrast limit, and opacity. No 5,000 µm viewport limit applies.
- **ROIs**: choose **Lasso** or **Box** next to the renderer selector and draw; the drawn outline itself becomes the ROI (the Plotly view uses the convex hull of the selected cells; with samples side by side, of the selected points as drawn, since cell ids repeat between samples). Saved ROIs are drawn as labelled outlines. Every cell inside an outline gets the label `roi_<class>`, in the primary sample, in the samples beside it (at their offset) and in the loaded segmentation runs. `roi_*` labels that came with a loaded store and have no outline in this cache directory are left as they are.
- **Click a cell** to fill the cell-info panel, from either the spatial view or the UMAP. Hovering shows the cell id and its value. Double-click a legend entry to isolate that category.
- **Ruler**: end points snap to a cell centre when one is hit and are free otherwise.
- **Overlays**: SpatialData ROI and patch outlines, CellNEST edges with their colour and gray-out options, and RCTD weight pies.
- **Multiple samples** side by side, with a dotted separator midway in each gap; boundaries, transcripts, image and UMAP remain primary-sample only, as in the Plotly view. Clicking a cell of any sample fills the cell-info panel; cells of an added sample are marked `[sample N: <folder>]`.
- **PNG export**: the **PNG** button next to Pan / Lasso / Box downloads the current view at 2× with legend and scale bar.
- **Figure export** (SVG, PDF, PNG): **Export figure** next to the PNG button opens a small panel with the format, a pixel scale (1× to 4×) and a vector limit, and saves `xenium_figure.svg`, `.pdf` or `.png`. The figure shows what is in view: cells as circles in their current colours and opacity (hidden categories left out), the boundary sets that are switched on as outlines, transcripts as dots, ROI outlines, ruler and other overlays, and the morphology image and the density layer as one embedded image beneath them. A scale bar of a round length (1, 2 or 5 × 10^k µm, at most a fifth of the width) is drawn at the lower left, and the legend is placed in a margin to the right of the map: the visible categories with their colours, or a colour bar with the name, minimum and maximum. Scale bar, legend and labels are always vector shapes and text.
  - **Vector limit** (default 50,000, shown in the panel): a layer (cells, one boundary set, transcripts) with more elements in view than the limit is embedded as an image at the chosen scale instead of being written as shapes, so the file stays bounded (2.5 to 3 MB of SVG per 50,000 cells). The file says so in its title and description (`<title>` / `<desc>` in SVG, Title / Subject in PDF), and the panel and the status line report it. Zoom in, or raise the limit, to get every cell as a shape.
  - **Units**: one figure unit is one CSS pixel of the view (one point in the PDF); the SVG root carries `data-um-per-px` and `data-viewport` (µm), and the scale bar carries `data-length-um`.
  - **PDF**: written by the viewer itself (PDF 1.4, compressed, transparency for opacities, the colour bar as a shading). Text is set in Helvetica, which is referenced and not embedded; characters outside Windows-1252 (for example Greek or CJK category names) print as `?` in the PDF, while the SVG keeps them. A journal that requires embedded fonts needs the PDF re-saved from a vector editor, or the SVG.
  - **PNG**: the same figure (map plus legend margin) rasterized at the chosen scale; the **PNG** button remains the quick snapshot with the on-screen legend.
  - The image layer is exported as drawn at that moment (tiles still loading are missing). Exports are not recorded in the provenance log.

Differences from the Plotly view: the Z-Level selector is disabled (the WebGL view reads the focus image); CellNEST arrowheads scale with edge length and edges have no hover text; gray-out dims every non-participating cell. CellNEST edges have so far been tested in this view with synthetic inputs only.

Both views degrade the same way when the selected colour cannot be drawn: a gene the dataset (or the Baysor/Proseg run) does not have is drawn as transcript counts; a cell-type mode with no labels loaded is drawn as one grey **No annotation** group on the Xenium source and as transcript counts on a Baysor/Proseg run; cell-type labels that match fewer than 5 % of a run's cells (left from another segmentation) are discarded and transcript counts are drawn.

Measured on a Xenium Prime 5K run with 112,551 cells: a recolour transfers about 0.45 MB per view and takes a few milliseconds of server time, against 5–7 MB of JSON per Plotly figure and roughly a second of server time.

Requirements and limits: a browser with WebGL2. When WebGL2 is missing or its context cannot be created, the app switches back to the Plotly view by itself, greys out the WebGL option and says why in the status line next to the selector (reload the page to try again). The Plotly view draws its points with WebGL 1, so with WebGL switched off altogether neither view can draw cells; the status line says so. The feature is new, so switch to the Plotly view if something looks wrong. The browser bundle is prebuilt in `assets/`; Node.js is needed only to rebuild it (see [Developing the WebGL viewer](#developing-the-webgl-viewer)).

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

The right-side panel holds the analyses. Its navigation has two levels: a row of **groups** and, below it, the tabs of the open group. The app remembers the last tab of each group for the session. This README names a place as `Group → Tab`.

| Group | Tab | Content | Reference |
|-------|-----|---------|-----------|
| **Overview** | UMAP | UMAP plot (alternative to the sidebar toggle) | W × H; download |
| | Cell types | Bar chart of cell type proportions, one row per annotation | W × H |
| | Composition | Per-sample proportions and a test between conditions | [Composition between conditions](#composition-between-conditions) |
| | ROI area | Area of the saved ROIs | W × H |
| **QC** | Segmentation QC | One row of metrics per segmentation source and counts mode | [Segmentation QC](#segmentation-qc) |
| | Compartments & spillover | Tissue compartments, spillover score, gene ownership | [Confound controls](#confound-controls) |
| | Segmentation-free | Marker transcripts per mm² counted from the transcript file | [Segmentation-free abundance](#segmentation-free-abundance) |
| | Transcript factors | Factorisation of binned transcripts, without cells | [Transcript factors](#transcript-factors-segmentation-free-factorisation) |
| **Expression** | Differential expression | Four tests: cells (Wilcoxon), pseudobulk across samples, proximity to a cell type, contact groups; volcano plot and sortable table | [Differential expression](#differential-expression) and the three sections after it |
| | Gene sets | Paste a gene list, score every cell, colour the views by the score | [Gene-set scores](#gene-set-scores) |
| | Spatial genes | Moran's I of every gene on a spatial neighbour graph | [Spatially variable genes (Moran's I)](#spatially-variable-genes-morans-i) |
| | Transcript density | Binned 2-D density of the transcripts of a gene under the cells | [Transcript density](#transcript-density) |
| **Neighbourhoods** | Co-occurrence | Spatial co-occurrence score heatmap | Colorbar min/max, W × H; download |
| | Enrichment | Neighbourhood enrichment heatmap: log2 of the fraction of a cell type's neighbours that are of each type over that type's overall frequency (0 = random mixing) | Colorbar min/max, W × H; download |
| | Co-localisation | Cell-type pairs against a stratum-matched expectation | [Stratified co-localisation with matched nulls](#stratified-co-localisation-with-matched-nulls) |
| | Graph, Chord | Node-edge graph and chord diagram of cell type contacts | O/E normalize toggle, W × H; download |
| | Niches | UMAP (or spatial map) coloured by spatial niche | W × H; download |
| | Aggregates | Connected groups of cells of a chosen class | [Aggregates](#aggregates) |
| | Banksy-style | Clusters on own and neighbourhood expression | [Banksy-style spatial domains](#banksy-style-spatial-domains) |
| | Joint clusters | One cluster set shared by all samples | [Joint clustering across samples](#joint-clustering-across-samples) |
| **Communication** | Ligand–receptor | Scores between neighbouring cell types with a permutation p-value; exposure decomposition | [Ligand–receptor scores](#ligandreceptor-scores), [exposure decomposition](#ligandreceptor-exposure-decomposition) |
| | CellNEST | Cell-cell communication from CellNEST output | [CellNEST](#cellnest-cell-cell-communication) |
| **Provenance** | Records & methods | Every analysis run on the dataset; methods text and JSON export | [Methods and provenance record](#methods-and-provenance-record) |

Every analysis tab has the same order: the main choices and **Run** on the first row, further parameters behind an **Options** disclosure (its summary line names the defaults), then the status line, a method note (two lines; click to expand), the results and **Download**. On the tabs that use the sample design, one **Samples & conditions…** button with the current design summary sits above the tab. A tab whose analysis cannot run on the loaded data says why in place: no transcript file (Transcript density, Segmentation-free, Transcript factors), one sample (Composition), no annotation (Cell types, Segmentation QC), spatial analysis or niches not run yet, CellNEST results not loaded. The jobs panel sits at the bottom left, clear of the Run buttons. Switching tabs fills option lists only; nothing is recomputed.

A **download button** (camera icon) appears on hover over each plot; exports as SVG at the current display size.

**O/E normalization** (Interaction Graph + Chord Diagram) — When enabled, each edge weight is divided by the expected number of contacts under a random mixing model (`observed / (freq_A × freq_B × total_edges)`). This removes bias from abundant cell types and highlights interactions that exceed random expectation.

**Dimension controls** — Each tab has W × H inputs (in pixels) at the bottom. The graph fills the remaining tab space automatically.

### Differential expression

**Expression → Differential expression.** Compares the expression of every measured gene between two groups of cells of the active view.

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

- Expression is `log1p(count / library size × 10,000)`, where the library size is the cell's total count over the measured panel genes in the active counts mode. When cells of samples with different panels are pooled, it is the total over the genes the samples share (see [Statistical conventions](#statistical-conventions)); the status line, the CSV header and the methods text then say so.
- Test: two-sided Wilcoxon rank-sum (Mann–Whitney U) per gene, normal approximation with tie correction and continuity correction. The implementation ranks only the non-zero values of the sparse matrix (all zeros of a gene share one average rank) and is numerically identical to `scipy.stats.mannwhitneyu(method="asymptotic")`; `scanpy.tl.rank_genes_groups(method="wilcoxon", tie_correct=True)` gives the same statistic without the continuity correction.
- `log2FC = log2((mean_A + 1) / (mean_B + 1))` of the group means of the normalised (not log-transformed) values, i.e. counts per 10,000 with a pseudocount of 1 (the Seurat definition). `% A` / `% B`: percentage of cells with at least one count.
- FDR: Benjamini–Hochberg over all tested genes.
- The test respects the **active segmentation**, the **Active Counts** mode (original or SPLIT-corrected) and an active **subset**. **Imputed genes are not tested**: their values are not counts.
- Groups under **10 cells** are refused. Cells that are in both groups are **removed from B**; the status line says how many.

**Caveats — read before quoting a p-value**

- **Every cell is treated as an independent observation**, so the p-values are too small and should not be used to rank genes: see [Statistical conventions](#statistical-conventions) (units of inference, pseudo-replication).
- **Comparisons across samples or replicates** (several samples loaded side by side, or groups that coincide with samples): switch **Test** to *Pseudobulk across samples* ([Pseudobulk differential expression across samples](#pseudobulk-differential-expression-across-samples)). With several samples loaded the cell-level test pools their cells and says so in the status line.
- **Groups defined from the same expression data** (clusters) are selected to differ, so their p-values are inflated as well ("double dipping").
- Fold changes of spatial groups also reflect differences in cell-type composition between the regions.

### Samples and conditions

Tests that compare **samples** (pseudobulk differential expression, composition) share one assignment: every loaded sample (the primary one, `S0`, and those added with **+ Add sample**, `S1`, `S2`, …) is in **condition A**, in **condition B**, or **excluded**, optionally with a **pairing / batch** label (a patient, a slide, a run). **Samples & conditions…** in either tab opens the editor; the two conditions can be named. The assignment is kept per sample for the session (it is not written to disk) and is cleared when another dataset is loaded.

- **Donor (pairing)**: the individual a sample comes from. How donors define the independent units and when a design is paired: see [Statistical conventions](#statistical-conventions).
- **Batch** (a slide, a run): an optional blocking term of the model-based tests when the design is not paired by donor (`~ batch + condition`). A label must be given to every included sample or to none and must not be confounded with the condition; such a design is refused with the reason.
- The design is **saved in the cache directory** (`design_<hash>.json`, keyed by the set of loaded samples) and restored when the same samples are loaded again, also after Load Sample.

**Exact sample-level tests** (paired sign-flip, label permutation; used by the composition test's *exact* option, stratified co-localisation and the other sample-level tests): enumeration limits, the smallest achievable p and the unit named with every p-value are described once under [Statistical conventions](#statistical-conventions).

**Cell types of added samples.** Annotation runs on the primary sample only. An added sample gets cell types from (1) the annotation cache, if that sample was annotated earlier as the primary dataset with the same cache directory (per method, the most recent Xenium-segmentation run), or (2) the `cell_type_<method>` columns of a store written by **Save as SpatialData**. A sample without labels for the chosen method is left out of cell-type populations and composition tests, and named. *Behaviour change:* added samples that have such labels are now coloured by cell type in both renderers (they were always "Unknown").

### Pseudobulk differential expression across samples

**Expression → Differential expression → Test: Pseudobulk across samples.** The samples, not the cells, are the observations.

1. Assign the samples to the two conditions (**Samples & conditions…**).
2. Pick the **cells** whose counts are summed: all cells, a cell type, a cluster, a ROI class (all its ROIs — draw one per sample), a single ROI, a saved group or the current selection.
3. **Run**. The result uses the same volcano plot, table and **Download CSV** as the cell-level test; title, status line and CSV header name the method, the design and the samples. A table under the status line lists, per sample, the cells summed, the library size and the size factor.

- **Counts**: the raw integer counts of the chosen cells are summed per sample over the measured panel genes. A matrix with non-integer values is refused; the one exception is an `.h5ad` loaded with `"noninteger_counts": "round_sums"` (a file that only has corrected counts): the values are then summed per pseudobulk column — per donor and condition when donors are given — and each sum is rounded once, and the result, its status line and its CSV say that the counts are not raw. Imputed genes are never used. **The original counts are always used, also when Active Counts is "SPLIT Corrected"** (the status line says so; see [Statistical conventions](#statistical-conventions)).
- **Donors**: with donors on every included sample, the samples of one donor within a condition are summed into one pseudobulk column first (named in the status line), so a donor is one unit. The donor is a blocking factor (`~ donor + condition`) only when the design is **paired** by the rule of [Statistical conventions](#statistical-conventions) (at least 2 donors with samples in both conditions); donors seen in one condition only are then left out and listed. Otherwise (each donor in one condition, or fewer than 2 donors in both) the model is `~ condition` on the per-donor sums — `~ batch + condition` when batch labels are given and every donor column has one batch; a donor that occurs in both conditions of an unpaired design is named in a warning and its two columns are treated as independent.
- **Genes**: the genes shared by the samples assigned to A or B are tested; a sample marked *excluded* does not change the gene list. When the used samples share fewer genes than the largest panel among them, the status line says so.
- **Volcano**: points are coloured by **FDR < 0.05 alone**, and the x axis is the unshrunken (MLE) log2FC by default; **Volcano x axis** switches to the shrunken estimate. (Before, the shrunken estimate was plotted and a significant gene shrunk to ~0 was drawn as not significant.)
- **Model**: **DESeq2** through rpy2 — design `~ condition`, or `~ batch + condition` with batch labels; Wald test; independent filtering at alpha 0.05; log2 fold change **A / B**. The table gives the unshrunken (MLE) log2FC and, with **Shrink log2FC** ticked, the **apeglm**-shrunken one (DESeq2's `normal` prior when apeglm is not installed; the column and the volcano axis name the estimator, or say "unshrunken"). If DESeq2's dispersion trend cannot be fitted the gene-wise estimates are used and the status line says so.
- **Where R runs**: the fit runs in an R worker process of the job, so **Cancel** stops it at once (see [Jobs panel](#jobs-panel-progress-and-cancel)); each run pays about 3 s for R and DESeq2 to load. `XENIUM_R_INPROCESS=1` fits inside the app instead; the results are identical.
- **Without DESeq2**: **edgeR**'s quasi-likelihood F-test is used when edgeR is installed, and the result is labelled as such. With neither package the run is refused with the install hint (`BiocManager::install(c("DESeq2", "apeglm"))`). There is no fallback to a t-test.
- **Refusals and warnings**: a condition with **fewer than 2 samples is refused**; with fewer than 3 the status line warns that the variance estimates are unreliable. A sample contributing fewer than **Min. cells / sample** (default 10) is dropped and named. Genes with a total count below **Min. total count / gene** (default 10) are not tested and counted in the status line, as are genes DESeq2 leaves without an adjusted p-value (they are not drawn in the volcano).
- A Baysor / Proseg run is a single sample: pseudobulk needs the Xenium segmentation with several samples loaded.
- **Validation**: on fabricated six-sample data and on six sub-bundles of the Prime 5K skin section the log2FC, p-values, adjusted p-values and apeglm-shrunken log2FC equal those of DESeq2 run directly in R on the same matrix (largest relative difference 3 × 10⁻¹³; 4,226 genes); likewise for the batch design and for the edgeR path.

**Caveats**

- **Replicates must be biological replicates**, not regions or serial sections of one block: see [Statistical conventions](#statistical-conventions) (pseudo-replication).
- With 2–3 samples per condition power is low and a single odd sample decides the result; look at the per-sample table.
- **Cluster ids** are per sample: "cluster 3 in every sample" is the same population only if the samples were clustered together. Prefer cell types.
- Fold changes of a population defined by a ROI also reflect its cell-type composition.

### Proximity differential expression

**Expression → Differential expression → Test: Proximity to a cell type.** "Which genes of the cells **T** change near cells of the anchor type **X**?" Pick where the cell types come from (an annotation, or the clusters), T and X, then **Run**. It works on one sample; with several it tests across them.

- **Per sample**: log1p(count / library size × 10,000) of every T cell is residualised (ordinary least squares) on log1p(library size) and on decile bins of the local fraction of the **density-covariate** type (cells within 50 µm; default: the type, other than T and X, that contributes the most transcripts; "none" drops the term; X itself is refused, because its local fraction is nearly the distance that is tested). d is the distance to the nearest X cell of the same sample.
- **Effect** (table column *Near − far*, the volcano's x axis, in log2 units): mean residual of the T cells with d ≤ **Near** (default 25 µm) minus that of the T cells with d ≥ **Far** (default 100 µm). **Slope**: OLS slope of the residual on −log(d + 1). A sample needs ≥ 100 T cells, ≥ 5 X cells, ≥ 5 near and ≥ 50 far cells; genes detected (≥ 1 transcript) in fewer than 2 % of the T cells are not tested.
- **Matched random anchors** (default 99 sets): as many non-X cells as there are X cells, drawn within the same compartment × edge-distance stratum (the columns of the Confound Controls tab; the defaults are computed when they are missing, and the status line says so), put through the same computation. They answer "what does any set of cells sitting where X sits produce?".
- **p-values**: with ≥ 3 independent units (donors when every included sample has one, else samples; *Samples & conditions…*) the **exact sign-flip test** of the per-unit effects, with the limits of the sample-level tests ([Statistical conventions](#statistical-conventions)) — the status line gives n and the smallest achievable p. With fewer units (a **single sample**) p is the share of random anchor sets at least as far from their mean as the observed effect; the status line marks this as a single-sample result that does not generalise.
- **Empirical FDR** (column *Emp. FDR*, the volcano's y axis): the number of genes the random sets yield at a gene's threshold divided by the number observed (never 0, made monotone). In the cohort mode the random sets' sign-flip p-values are used, in the single-sample mode |z| against the random sets with every random set scored against the others.
- When fewer than half of the random sets leave enough near and far cells in every unit (anchors as numerous as an abundant type cover the tissue), p and FDR are taken from the **slope** instead and the status line says so; if that fails too, the cohort mode falls back to Benjamini–Hochberg and the single-sample mode refuses.
- **Specific**: not reproduced by up to 3 other anchor types of similar abundance (within 3×): in the cohort mode same sign in ≥ 5/7 of the units, in the single-sample mode same direction and beyond the random sets at p ≤ 0.05.
- **Leak gate**: the gate of the other tests is run for T near X against T far from X, and genes that X owns and T does not (Confound Controls → gene ownership) are flagged as well: next to X, T cells pick up X transcripts. Flagged genes are open orange diamonds in the volcano; *X ÷ T rate* shows how much more X expresses a gene. Genes with X as top owner that T also carries are counted in the status line. The gate's change rule is applied to the genes with an empirical FDR below 0.10 or a pooled |log2 change| of at least 0.5 (see [Leak gate in Differential Expression](#leak-gate-in-differential-expression)). Unticking **Gene ownership + leak gate** switches ownership and gate off for this mode too: no gene is then flagged, and the status line says so.
- **Between conditions**: with a design that gives ≥ 3 units per condition the per-sample effects are also tested A − B (columns *A − B*, *p (A vs B)*, BH FDR).
- **Fit exp(−d / L)** (optional): per gene and sample the length L (grid 2–600 µm) for which residual ~ a + b·exp(−d / L) fits best; the table gives the median L and R². R² is small for most genes.
- Honours the segmentation, the subset and Active Counts. **Timing**: 112,551-cell Prime 5K skin section, 99 random sets: 10,816 fibroblasts near 263 mast cells 4.8 s; 62,817 T cells near 3,427 macrophages 10.5 s headless, 7.8 s in the app.
- **What it is not**: a gradient with distance is an association with position. The residualisation and the random anchors remove depth, a dominant neighbour's density and generic location; they do not remove everything that co-varies with being near X.
- Reimplemented from an earlier spatial study by the author (checked against that code on fabricated data: residuals, near − far and slope to 10⁻⁸, sign-flip p-values and random draws identical). Changed here: sufficient statistics on sparse matrices instead of dense residuals; strata of the random anchors are compartment × edge bin instead of deciles of one cell type's local fraction; per-sample effects are averaged per unit; the FDR is evaluated at every gene's own threshold instead of on a fixed grid; a single-sample mode exists.

### Contact-group pseudobulk differential expression

**Expression → Differential expression → Test: Contact groups (pseudobulk).** "Do T cells in contact with X differ from T cells that are not?" Needs several samples.

- **Groups**: within each sample, T cells with an X cell within the **contact radius** (default 50 µm, centroid to centroid) against T cells without.
- **Matching**: the comparator cells are weighted so that their mix of compartment × edge-distance bin × density quintile equals that of the contact cells; contact cells in a stratum without comparator are left out (counted in the status line).
- **Model**: pseudobulk = Σ weight × counts per sample × group, rounded to whole counts, fitted with `~ sample + group` through the same R bridge as the pseudobulk mode (DESeq2, else edgeR, else refused with the install hint). With donors on every used sample, samples of one donor are summed into one block (`~ donor + group`). The unit of the p-values is the sample, or the donor when samples were pooled by donor; the status line states n.
- **Refusal**: fewer than 2 samples with at least **Min. cells / sample / group** (default 20) in both groups → no test, with the counts per sample and a pointer to the proximity mode. A Baysor / Proseg run is one sample.
- **Label shuffle** (default 199, 0 to skip): contact / comparator labels are permuted within sample × stratum, weights and fit are redone, and the number of genes passing (FDR < 0.05, |log2FC| > 0.5, not flagged) is compared with the observed number: P = (1 + shuffles with at least as many) / (1 + shuffles). Every shuffle is a model fit (about 4–6 s for 4,200 genes × 12 pseudobulks), so 199 take a quarter of an hour; the job shows progress and can be cancelled.
- **Leak gate** per sample (contact = arm A); genes X owns and T does not are flagged too and never count as passing. With **Gene ownership + leak gate** unticked neither is computed, nothing is flagged and *pass* is FDR and effect size alone (stated in the status line).
- Always uses the **original counts**, as the pseudobulk mode does: a count model is not given SPLIT-corrected counts. With Active Counts = "SPLIT Corrected" the status line says that the original counts were used (the counts mode still picks the cluster column of a Baysor / Proseg run). A matrix with non-integer values (an `.h5ad` without raw counts) is refused unless it was loaded with `"noninteger_counts": "round_sums"`.
- Reimplemented from an earlier spatial study by the author (the matching weights equal that code's on fabricated data). Changed here: one generic comparison instead of fixed cell states; weighted sums are rounded because DESeq2 needs integers (the original fitted fractional sums with edgeR); the shuffle null counts passing genes as there.

### Composition between conditions

**Overview → Composition.** Tests whether the **proportion** of each cell type (or cluster, or spatial niche) differs between conditions A and B, with the same sample assignment.

- **Compute** shows, for every sample that has the labels, the proportion of each category: a plot with **one point per sample**, grouped and coloured by condition (bars: condition means; `*`: FDR < 0.05), a table (test results and each sample's proportion) and **Download CSV** (adds the cell counts).
- **Method (propeller-style, Phipson et al. 2022)**: per-sample proportions are **logit**-transformed (0.5 pseudocount per category, as `speckle`) or **arcsine-square-root**-transformed; a linear model (`~ condition`, or `~ batch + condition`) is fitted per category and tested with a **moderated t-test** — limma-style empirical Bayes shrinkage of the category variances towards a common prior (non-robust, no trend), implemented in Python. **Welch's t-test** on the same transformed values is given beside it (it ignores batch labels). Benjamini–Hochberg FDR over the categories.
- **With fewer than 3 samples in a condition no test is run** — the tab says so and shows the proportions only. With one sample its proportions are shown.
- **Test: exact (sample-level)** tests the same transformed proportions without a variance model: paired sign-flip over donors when the design is paired, otherwise label permutation over the independent units (see [Statistical conventions](#statistical-conventions)). The moderated t-test follows the same rule for what a unit is: with a donor on every included sample, the samples of one donor within a condition are one unit (their transformed proportions are averaged), the model is `~ condition` over the donors, or `~ donor + condition` when the design is paired, and at least 3 donors per condition (3 pairs) are needed. Without donor labels each sample is a unit.
- **Validation**: Welch's t, its degrees of freedom and p-value equal `scipy.stats.ttest_ind(equal_var=False)`; the moderated t, its p-value, the prior variance and the prior and total degrees of freedom equal `limma::lmFit` + `eBayes(robust = FALSE)` on the same transformed proportions (relative difference < 10⁻⁵), with and without a batch term. `speckle::propeller` itself was not installed: its default uses `robust = TRUE`, which can differ when one category's variance is an outlier. A quasi-binomial / beta-binomial GLM is not implemented.

**Caveats**

- **Proportions are compositional**: they sum to 1 in each sample, so a real expansion of one type lowers the proportions of all the others. The tests of the categories are not independent, and a "decrease" may be the shadow of another type's increase.
- With 3–5 samples per condition only large shifts are detectable, and the empirical Bayes prior is estimated from as many variances as there are categories (a dozen): treat the moderated p-values as approximate and compare with Welch's.
- Proportions depend on the tissue area sampled and on the segmentation; cells without a label are left out of the denominator.
- Cluster and niche ids are comparable across samples only when the samples were clustered together (niches are computed for the primary sample only).

### Segmentation-free abundance

**QC → Segmentation-free.** Answers "is there more of
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
  status line).
- **Between conditions**: with a [sample design](#samples-and-conditions), the exact
  sample-level test of A − B for transcripts per mm², cells per mm², transcripts per 10k and
  the Gini excess, at every tile size. With fewer than 3 units per condition, or one sample,
  the rows are descriptive and the status line says so. Units and tests: see
  [Statistical conventions](#statistical-conventions).

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

### Transcript factors (segmentation-free factorisation)

**QC → Transcript factors.** Finds expression programs and where they lie **without
using any cell**: the gene transcripts are counted on a grid, the bin × gene counts are
factorised, and the result is laid under the segmented cells to show where a cell collects
transcripts of more than one program.

> This is a **simplified transcript-binning factorisation in the spirit of FICTURE** (Si et
> al., *Nat Methods* 2024, doi:10.1038/s41592-024-02415-2). It is **not that tool**: there are
> no anchors, no sliding or overlapping windows and no pixel-level decoding; a factor is
> resolved to the bin, not to the transcript.

| Control | What it does |
|---------|--------------|
| **Method** | *NMF (KL loss)*: non-negative matrix factorisation with the Kullback-Leibler loss (the Poisson likelihood), scikit-learn's multiplicative updates from an NNDSVDa start, until no factor's gene distribution moves by more than 0.5 % (L1) in 15 updates, at most 150 updates. *LDA (online)*: scikit-learn's online variational latent Dirichlet allocation (Hoffman et al. 2010), 5 passes in minibatches of 2,048 bins. *Reference: cell-type profiles*: nothing is fitted - the components are the mean profiles of the cell types of the chosen annotation (Σ counts of the type ÷ Σ over the genes, active counts mode; types with fewer than 20 cells are left out and named), so the factors carry the type names. |
| **K** | Number of factors (2 - 60; default 12; ignored in the reference mode). |
| **Bin (µm)**, **square / hexagon** | Square bins of that side, or regular hexagons whose centres are that far apart (area √3⁄2 × bin²). Default 10 µm. |
| **≥ transcripts per bin** | Bins with fewer transcripts (default 10) are not used for the fit and get no dominant factor on the map; they are still projected. |
| **Fit on ≤ bins** | The loadings are fitted on a random subsample of the eligible bins (default 20,000; fixed seed), pooled over the samples. |
| **Seed** | Seed of the subsample and of the initialisation (default 0). |
| **Cell types from** | The annotation (or the active cluster set) the factors are compared with; required for the reference mode. |
| **Map layer** | *Dominant factor per bin* (one colour per factor, bins below the transcript minimum left empty) or the transcripts of one factor per bin, drawn **under the cells in both renderers** through the heat layer of the Transcript Density tab (its opacity and colour maximum apply; the layers are also listed in its gene picker). |

**What is computed.**

1. **Bins.** `transcripts.parquet` (or the tiles of `transcripts.zarr.zip`) is read one row
   group at a time with the filter of the cell-feature matrix pushed down (**Q ≥ 20, gene
   features only**, panel genes); a batch is reduced to its (bin, gene) counts at once and the
   counts are merged in chunks, so the transcripts are never all in memory. The bin × gene
   matrix is sparse and is cached per sample, bin size and shape
   (`txfactor_<dataset>/counts__…npz`): another K, method or seed does not read the transcripts
   again.
2. **Fit** on the subsample → gene loadings, one distribution over the genes per factor.
3. **Projection** of every bin onto the loadings: the factor counts W ≥ 0 that minimise the
   Kullback-Leibler divergence of the bin's counts from W · loadings, by the multiplicative
   (EM) update with the loadings fixed. A bin's factor counts sum to its transcripts, so a
   factor's value is "transcripts of this bin attributed to the factor". For LDA this is the
   maximum-likelihood mixture, **without** the Dirichlet prior of LDA's own inference.
   Factors are ordered by their share of all transcripts (F1 = largest).
4. **Table / CSV.** Per factor: share of the transcripts, bins and cells it dominates, the
   annotated type whose mean profile is most similar to its loadings (cosine) and the
   commonest type among its cells, and the top genes ranked by loading × ln(lift), where lift
   = the gene's share of the factor ÷ its share of all transcripts. The CSV lists 50 genes
   per factor with loading, lift and the share of the gene's transcripts the factor holds,
   and the cross-tab.
5. **Against the segmentation** (the cells of the active segmentation and subset): for every
   cell the factor counts of the bins under it - under its centroid and under six points at
   0.6 × the radius of a disc of the cell's area (the centroid alone when the table has no
   area) - are summed into a mixture. Stored per cell and exported by Save as SpatialData as
   `ctl_txf_dominant`, `ctl_txf_top_share`, `ctl_txf_mixed` and one `ctl_txf_share_<factor>`
   per factor. **Mixed score** = entropy of the mixture ÷ ln K (0: one factor; 1: all equal).
   **Color By** offers the mixed score, the share of the dominant factor and the share of
   every factor. The plot is the cross-tab **cell type × dominant factor** (share of each
   type's cells); the status line gives its purity, the adjusted Rand index and, in the
   reference mode, the share of cells whose dominant factor is their own type.

**Several samples.** Each sample is binned on its own grid in its own coordinates and has its
own count cache; the loadings are fitted once on bins pooled from all samples, and the layer
is drawn for every sample at its position. A sample loaded from a `.zarr` has no transcript
file: it is left out and named. A registered, cancellable job with progress (checked between
row groups, fitting rounds and projection chunks); a finished result is cached per settings
(`txfactor_<dataset>/fit_<hash>.npz`) and loaded from there when the same run is requested
again.

**Limits.** (a) A 10 µm bin is larger than most cells, so the mixture "under a cell" describes
the cell's position, not its own transcripts: a high mixed score marks places where the
segmentation is likely to mix signals (cell borders, dense infiltrates), it does not prove
that a given cell is contaminated. (b) NMF and LDA are fitted from one initialisation on one
subsample: factors that change with the seed or with K are not stable programs. (c) A factor
is not a cell type; two types with similar profiles share a factor and one type can split
into several (the table shows the cosine and the commonest type so that this is visible).
(d) In the reference mode a bin is forced into the given profiles - transcripts of anything
not in the annotation are attributed to the nearest profile - and the profiles themselves
carry whatever the segmentation leaked into them. (e) The transcript file is not subset or
corrected: the cell subset and the counts mode affect the cells compared and the reference
profiles, not the bins. (f) Hexagons are drawn on a square raster of a quarter of the bin size
(coarser on very large sections), so their outlines are approximate on the map; the values are
those of the hexagons.

**Optional packages.** None is required beyond scikit-learn. With `numba` installed (it comes
with `umap-learn`) the projection runs in compiled code on 4 threads; without it a numpy
implementation is used (same optimum, slower).

### Gene-set scores

**Expression → Gene sets.** Paste gene names (one per line, or separated by commas / semicolons / spaces), give the set a name and press **Save & score**. The set is saved per dataset in the cache directory (`genesets_{hash}.json`) and appears under **Saved** in the tab and as **Gene set: <name>** under Color By, which is switched to it when the score is ready. **Delete** removes the selected set.

- Names are matched to the measured panel genes exactly, else ignoring case. Genes that are **not found are listed** in the status line; a set with **fewer than 2** genes found is refused. Imputed genes are not used.
- **Score** per cell = mean of the z-scored log-normalised expression of the set genes − mean of the z-scored expression of a **control set**. Expression is `log1p(count / library size × 10,000)`. As in Seurat `AddModuleScore` / `scanpy.tl.score_genes`, all measured genes are ranked by mean expression and cut into 25 bins, and for every bin that holds a set gene 50 control genes are drawn from that bin (fixed seed, set genes excluded), so the control matches the set's expression level. Unlike those two, each gene is first standardised over the cells of the view (mean 0, sd 1), so that the highest-expressed genes of a set do not dominate its mean; genes without any variance are left out and reported.
- A score of 0 means "as expected for genes of this expression level"; the views use a **diverging colour scale centred at 0** (blue below, red above) in the spatial view and the UMAP of both renderers. The sidebar **Min / Max** inputs override either end.
- The score is computed for the **active view** — segmentation, counts mode, subset and, for the Xenium source, all loaded samples together (genes are standardised over all of them) — and recomputed when one of these changes (a fraction of a second to a few seconds).
- Agreement with scanpy (Prime 5K skin, 112,551 cells): identical scores when both use the same control genes (r = 1.000 without z-scoring); with each implementation's own random draw of control genes r ≈ 0.91–0.99 depending on the set, and r ≈ 0.85–0.96 for the z-scored default.

### Transcript density

**Expression → Transcript density.** Shows where the **transcripts** of a gene (or of all genes) lie, independent of any cell segmentation: a 2-D grid of transcript counts drawn as a heat layer **under the cells**, in both renderers.

1. Pick a gene, or **All genes**, and press **Compute**. The app reads `transcripts.parquet` once for that gene and for every loaded sample (a registered, cancellable job with progress) and stores the grids in the cache directory; afterwards the gene is available immediately.
2. **Bin (µm)**: 5, 10, 20 or 50. **Show heat layer**, **Opacity** and **Colour max** control the layer; lowering the sidebar's point opacity makes it easier to see under the cells. The colour bar gives transcripts per bin (0 – the 99.5th percentile of the non-empty bins unless a maximum is entered); empty bins are transparent.

- Transcripts are filtered like the cell-feature matrix: **Q ≥ 20, gene features only** (no control probes or codewords, blank / unassigned / deprecated codewords). The sum of a gene's grid equals the number of its transcripts that pass this filter — including transcripts outside every cell, so it can exceed the gene's total in the count matrix.
- Bin *i* covers `[i × bin, (i + 1) × bin)` µm from the slide origin in x and in y. Grids are stored at 5 µm; the coarser bin sizes are exact sums of those bins.
- The parquet file is read one row group at a time with the filter pushed down and two columns decoded (bundles without `transcripts.parquet`: the full-resolution tiles of `transcripts.zarr.zip`), so memory does not grow with the number of transcripts. On the Prime 5K skin bundle (74 M transcripts) a gene takes about 2 s and all genes about 3 s.
- Only the bins of the current viewport are sent to the browser. When the viewport holds more bins than can be drawn (zoomed out on a large slide at a small bin size) neighbouring bins are **averaged** (not summed), so the values stay "transcripts per bin" and the colours do not change with the zoom; zooming in shows the bins themselves.
- The layer does not depend on the segmentation, the counts mode or a cell subset. It needs the Xenium output directory (not available for a dataset opened from a bare `.zarr`).
- **Several samples**: every sample has its own grid, cached in its own directory (`density_{hash of that sample}/`, so a grid is found again whether the sample is loaded as primary or added). Each grid is drawn at its sample's position in the combined view, in both renderers, on **one colour scale** (the 99.5th percentile of the non-empty bins of all samples together) and with one colour bar. The grids are not resampled: bin edges stay multiples of the bin size in each sample's own coordinates. A sample added after Compute has no grid until Compute is pressed again, and a sample loaded from a `.zarr` has none; the hint line names both. *Behaviour change:* the layer used to show the primary sample only.

### Spatially variable genes (Moran's I)

**Expression → Spatial genes → Run.** Ranks the measured genes by spatial autocorrelation.

- **Graph**: the **k nearest neighbours** of every cell centroid (k = 6 by default), built for this analysis; weights are **row-standardised** (each neighbour weighs 1/k; the graph is not symmetrised).
- **Statistic**: Moran's I of `log1p(count / library size × 10,000)` per gene. **z-score and two-sided p-value** from the analytic mean (−1/(n−1)) and variance of I under the **normality assumption** (Cliff & Ord), **Benjamini–Hochberg FDR** over the tested genes of a sample. Genes not detected in any cell are not tested.
- Computed for the **active view** (segmentation, counts mode, subset) and **per sample** when several are loaded (each on its own graph; the table has a Sample column). A registered, cancellable job with progress.
- The table is sorted by I and can be filtered; a click on a row selects the gene in the sidebar; **Download CSV** writes all genes.
- **Cost**: the count matrix stays sparse and genes are processed in dense blocks of 64 MB. Prime 5K skin (112,551 cells × 5,006 genes, 33 M non-zeros): **5.5 s**, about 0.3 GB above the loaded dataset.
- **Validation**: I, z and p equal the textbook dense formulas on fabricated data and on the fixture bundle, and `spdep::moran.test(randomisation = FALSE)` on a k-nearest-neighbour `listw` (relative difference < 2 × 10⁻⁶, the precision of the float32 size factors). squidpy / esda were not installed and were not compared.

**Caveats**

- With tens of thousands of cells **almost every expressed gene is "significant"** (4,051 of 5,006 on the skin section): rank by I, not by the p-value (the cells are the unit; see [Statistical conventions](#statistical-conventions)). The normality assumption is rough for sparse counts.
- I measures autocorrelation at the scale of the k neighbours. A marker of a cell type that sits in clusters scores high for that reason alone; it does not separate "spatial pattern within a cell type" from "cell-type pattern".
- Segmentation spill-over between touching cells raises I for highly expressed genes.

### Ligand–receptor scores

**Communication → Ligand–receptor.** Scores ligand–receptor pairs between cell types that are neighbours in the tissue, without CellNEST.

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
- **Two conditions** (*Samples & conditions…*, the editor of the pseudobulk test): the sums of a donor's samples within a condition are pooled (ratios of pooled sums are exact), then per unit the log2 ratio A / B of each term; **log2 E ratio = Σ of the five log2 term ratios**. Paired design (every sample has a donor, ≥ 3 donors in both conditions): exact sign-flip test of the per-donor log2 ratios. Otherwise, with ≥ 3 units per condition: exact label permutation of mean(log2 term in A) − mean(log2 term in B) (all relabellings up to 20,000, else 5,000 random ones with a fixed seed, and said so). Benjamini–Hochberg FDR over the interactions; unit, n and the smallest achievable p are reported as for every sample-level test ([Statistical conventions](#statistical-conventions)). Per sender → receiver pair the interactions are averaged per unit and tested the same way (with the exact sign-flip 95 % interval when there are enough donors for one), and per term an **omnibus** over the between-type pairs: Σ t² against every sign (or label) assignment of the units, plus the shift shared by all pairs.
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

The features of this and the next three sections (and the exact sample-level tests of [Statistical conventions](#statistical-conventions)) are general versions of controls developed in an earlier spatial study by the author, where each was written after a specific false result: depth producing correlations, neighbour transcripts read as a cell type's programme, a shared niche read as an interaction, cells counted as replicates. They are reimplemented here without any study-specific gene list, cell-type name or threshold: every such value is a parameter with a default. All of them work on the active segmentation (Xenium, Baysor, Proseg), the current subset and every loaded sample; all neighbour searches are made **within one sample**.

**QC → Compartments & spillover.** Pick where the cell types come from (an annotation or the cluster set) at the top.

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

- **Model**: leakage adds leak(L) × (profile of neighbouring type L) to the focal cells. Predicted leaked signal of a gene = Σ_L leak(L) × its counts per million in L. **Flag** when the predicted leak is more than half of the observed signal in either group, or when the leak alone predicts at least half of the observed log fold change. The second (change) rule is applied only to genes **whose change is called** (FDR < 0.05 in the two DE modes, the mode's FDR threshold in the proximity and contact modes) **or whose pooled |log2 change| is at least 0.5**: the explained share is a ratio, and for a change that is noise around zero it is a ratio of two small numbers that exceeds one half by chance. Other genes are judged by the signal rule alone. The status line gives the number of genes flagged by each rule and the number the change rule was applied to; `leak_change_frac` is in the CSV for every gene. In the pseudobulk mode every sample is a section with its own leak and neighbour profiles and the groups are averaged over sections; in the cell-level mode each group is one pooled section.
- **Measuring leak(L)** differs from the original, which chose marker genes the focal type does not express in a paired single-nucleus reference. Without a reference: a gene's focal ÷ L ratio is the leak plus the focal type's own expression, so the leak cannot exceed the lowest ratios; it is the median of the 25 lowest ratios among L's own genes (≥ 50 CPM in L, ≥ 10× every third type). A neighbour with fewer than 10 such genes, or whose own genes the focal type carries at ≥ 50 % of their level, is **not modelled and named** — its contribution is missing from the prediction.
- It needs group A to be one cell type (≥ 80 % of its labelled cells). When B is the same type both groups are modelled; otherwise only A (e.g. "type vs all other cells").
- **Limits**: the estimate leans low (the lowest ratios are selected in the same data), so the gate errs towards not flagging; with an annotation that does not separate the types (sub-types of one lineage, a classifier that calls most cells one label) few neighbours are modelled and the gate says little. A flag means "check the owner and the neighbours", not "wrong".

### Segmentation QC

**QC → Segmentation QC.** One row per segmentation source of the primary sample (Xenium and every cached Baysor / Proseg run, loaded or not) and counts mode (original, SPLIT-corrected), as a registered, cancellable job; table, four small bar plots, CSV.

- **Identity marker sets**: the 5 most specific genes of every type, derived once from the Xenium segmentation's original counts and annotation, or typed in (`Type: gene, gene, …` per line; a line `geneA, geneB` adds a pair of mutually exclusive markers).
- **Exclusion-rule failure**: share of labelled cells whose own identity set is not the largest share of their transcripts. **Co-expression**: over pairs of top markers of different types, cells with both ÷ cells with either (count ≥ 1). **Spillover slope**: in cells not of the source type, source-identity transcripts per 100 regressed on the number of source-type neighbours within 15 µm (reference-free; 0 = no dependence), and the reverse. **Own-identity retention** (corrected counts): own identity transcripts after ÷ before correction; below 1 with a lower failure rate is over-correction, not improvement. Plus cells, median transcripts / genes (count ≥ 1) / area per cell.
- **Assigned** (original counts only): Σ counts of the source's cells ÷ gene transcripts (Q ≥ 20) inside the bounding box of its cell centroids, grown by one median cell radius. The transcripts of the section are counted by the job itself the first time (the same scan and cache file as the *All transcripts* grid of the Transcript Density tab; cancellable, nothing is written when cancelled) and read from the cache afterwards. For the Xenium segmentation of a whole section this equals the share of Q ≥ 20 gene transcripts with a cell id in `transcripts.parquet`; for a run on a region the box is at least as large as the region, which lowers the value. A dataset loaded from a `.zarr` has no transcript file: the column is empty and the status says so.
- A corrected row covers the cells the correction touched (non-empty corrected row; "Corrected share") and its retention compares those same cells before and after. The row's retention is the pooled ratio (Σ own-identity transcripts after ÷ Σ before, over all types); a type whose cells had none before the correction has no per-type retention.
- A run without its own annotation — or whose annotation uses other type names — gets the Xenium label of the nearest Xenium centroid within 10 µm (the Labels column says so). This favours segmentations that resemble Xenium's.
- No p-values: these are descriptive. Runs on a small ROI give noisy slopes. The original benchmark's criteria that need a paired dissociated reference (patient-specific signal, arm balance) are not implemented.

### Stratified co-localisation with matched nulls

**Neighbourhoods → Co-localisation**, beside the kNN-based Neighborhood Enrichment. For every ordered pair (focal A, target B) and radius R (default 50 µm):

- **O** = B cells within R of each A cell. **E** = the mean of that count around the *pool* cells (labelled cells that are neither A nor B) of the same **stratum**: sample × compartment × edge-distance bin (50, 100, 150, 200, 300, 500, 1000 µm) × local-density quintile; a coarser stratum when fewer than 3 pool cells share it. **Effect** per sample = log2((ΣO + 0.5) / (ΣE + 0.5)). The unstratified value (`effect_naive`, pool = the whole sample) is given beside it: the difference is what "both types live in the same region" contributed.
- **Bias and within-sample p**: the A label is permuted among A and pool cells within each finest stratum (B fixed, default 200 permutations, fixed seed); the mean permuted effect is subtracted (`effect_corrected`), p = (1 + permutations at least as far from that mean) / (1 + permutations).
- **Across samples**: with ≥ 3 independent units (donors when given, else samples) the bias-corrected effect is tested with the exact sign-flip test, and between conditions A and B with the paired sign-flip or label-permutation test of the design ([Statistical conventions](#statistical-conventions)). A pair that is not testable in every sample is tested on the units where it is (at least 3): the table gives the units behind each p (**Units**, **Units (A vs B)**), the status line gives their range and the smallest achievable p of a pair tested on every unit, and donors seen in one condition only — which a paired test cannot use — are listed. With fewer units the tab says **SINGLE-SAMPLE / PER-SAMPLE RESULT**: only the within-sample p (unit: the cells of one section) is given.
- **Placebo anchors** (optional focal type): the same statistic for count-matched random anchor cells drawn within each sample × compartment (mean, 95 % range, p), and for the other types within 3× the focal type's abundance that share its most common compartment. The verdict is "specific" only when the focal type lies beyond both. Without a placebo, "X sits near Y" cannot be told from "X is a cell of that compartment".
- Heatmap (focal × target, `*` FDR < 0.05 of the test named in the title), table, CSV (per sample and pair, with the pair-level tests). Only types with ≥ 20 cells; a pair needs ≥ 20 focal and ≥ 10 target cells in a sample; the 40 most abundant types at most. Compartments come from the Confound Controls tab; when absent, edge distance and density are computed with the defaults and there is no core.
- **Speed** (measured): 112,551 labelled cells, 11 types, R = 50 µm, 200 permutations, placebo: 5.4 s, about 60 MB above the loaded dataset.

### Aggregates

**Aggregates** tab: finds connected groups of cells of a chosen *class* — a set of cell types
you pick, e.g. the immune types — that are locally denser than chance, and reports each one.
The tab has its own **Samples & conditions…** button (the shared editor) for the comparison between conditions.
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
- **What the p-values mean.** The hotspot p-values (unit: the cells of one section) are a
  detection threshold. Between conditions, aggregates per mm² and area fraction are compared
  with the exact sample-level test (unit: the sample or donor). Units and tests: see
  [Statistical conventions](#statistical-conventions).
- **Limits.** The result depends on the annotation: a class that is most of the tissue gives
  large "aggregates" and a high control rate (read the control column). Recompute after changing
  the subset, the segmentation or the annotation. The cells × cells neighbour matrix is held in
  memory (about 100 neighbours per cell at 50 µm in dense tissue). Registered job with progress
  and Cancel.

### Banksy-style spatial domains

**Neighbourhoods → Banksy-style.** Clusters the cells on their own expression *and* on the
expression around them, after BANKSY (Singhal et al., *Nat Genet* 56, 431-441, 2024). With a
large weight on the surroundings the clusters are tissue domains; with a small one they are cell
types informed by their neighbourhood. The result is a **cluster set** ("Banksy-style λ=0.8"):
pick it under **Color By → Cluster → Clustering Method**.

**Status.** This is a native implementation, not the Banksy package: it follows the published
method's construction (the three feature blocks and their λ scaling) but has not been validated
against the reference implementation. Hence the name "Banksy-style". Sets cached under the
earlier name ("Banksy λ=…") keep their key and are shown under the new name.

| Control | What it does |
|---------|--------------|
| **Preset / λ** | *Tissue domains* λ = 0.8 (Leiden resolution 0.5), *Cell typing* λ = 0.2 (resolution 1), or your own λ in [0, 1]; λ = 0 is ordinary clustering of the same features. One cluster set per λ: a new run with the same λ replaces it. |
| **Spatial neighbours k** | k_geom, default 15: the mean is taken over the k nearest cells, the AGF over the 2k nearest. |
| **AGF (gradient) block** | Include the azimuthal Gabor filter block (default on). |
| **Resolution** | Leiden resolution; the number of clusters grows with it. |
| **Genes ≤** | Feature cap, default 2,000: with a larger panel (Xenium Prime 5K) the genes with the largest variance of log1p(normalised expression) are used. Genes seen in fewer than 10 cells are left out. |
| **PCs / Graph neighbours / Seed** | 20 principal components, 50 neighbours in PC space (the published defaults), seed of Leiden and of the approximate neighbour search. |
| **log1p** | Log-transform the normalised counts before scaling (default off, as in the paper). |

**Method as implemented** (numpy / scipy / scikit-learn; the `Banksy` / `Banksy_py` packages are
not used):

1. Counts of the active counts mode, divided by the cell's total and multiplied by the median
   total of all cells; gene selection as above.
2. Per cell *u*, within its sample: the **neighbourhood mean** M = Σ Γ_uv g_v over the k nearest
   cells, with Γ_uv ∝ exp(−r_uv² / σ_u²), σ_u the distance to the ⌊k/2⌋-th nearest cell (the 7th
   for k = 15), normalised to sum 1; and the **AGF** G = | Σ Γ′_uv e^{iφ_uv} (g_v − m′_u) | over
   the 2k nearest cells with their own weights Γ′ (σ: their k-th), φ_uv the direction of *v* seen
   from *u*, m′_u the Γ′-weighted mean of the same neighbours. G is the size of the expression
   gradient around the cell and does not change when the section is rotated or mirrored.
3. Each of the three blocks is z-scaled per gene over all cells, and the blocks are joined as
   [√(1−λ) · own, √(λ/μ) · mean, √(λ/(2μ)) · AGF] with μ = 1.5 (√(1−λ) and √λ without the AGF).
4. PCA of that matrix (never held in memory: its Gram matrix is accumulated over blocks of
   8,192 cells), then the nearest neighbours of every cell in PC space → shared-nearest-neighbour
   graph (Jaccard weights) → **Leiden** (`leidenalg` + `igraph`). Without those two packages the
   clusters are a KMeans partition of the same PCs (k = 10 × resolution) and the status and the
   CSV say so. Above 40,000 cells the neighbours in PC space are approximate (`pynndescent`, when
   installed; exact otherwise).

Which details follow the paper and which are choices of this implementation: the weights Γ, the
2k neighbours of the AGF, the z-scaling of the three blocks, the mixing with μ = 1.5, λ = 0.8 /
0.2, k = 15, 20 PCs and 50 graph neighbours are the paper's Methods. Centring the neighbours'
expression inside the AGF (the `− m′_u`) follows the change notes of the reference
implementations ("normalisation with the mean neighbour expression"), not equation 1 of the
paper, which has no centring; the reference code was not run or read here. The gene selection
rule, the width σ of the AGF neighbourhood (the same rule applied to 2k cells), the
shared-nearest-neighbour weighting of the graph and the default resolutions are this
implementation's. **No comparison against the reference packages was possible** (not installed);
the implementation is validated on fabricated cells with planted domains that differ by a weak
programme in every cell: clustering on expression alone (λ = 0) finds the cell types and not the
domains (ARI with the domains < 0.05), λ = 0.8 finds the domains (ARI > 0.9), λ = 0.2 still finds
the cell types (ARI > 0.9).

**Several samples.** Spatial neighbours are always cells of the same sample (the view offset
plays no role). Normalisation, gene selection, scaling, PCA and clustering are done on all cells
together, so cluster 3 is the same cluster in every sample and the multi-sample view shows the
labels without a sample prefix (`C3`, not `S0:C3`). No batch correction is applied: a cluster
that exists in one sample only can be a batch effect - see
[Joint clustering across samples](#joint-clustering-across-samples) for Harmony.

**Output.** The tab shows the cells per cluster, the mean expression of each cluster's top genes
(in standard deviations from the mean of all cells), and per cluster the **neighbour purity**:
the mean share of a cell's k nearest cells that are in its own cluster, next to the share
expected if the labels were scattered at random (the cluster's share of all cells). CSV download.
The job is listed in the jobs panel (progress, Cancel); the result is cached per dataset, cells,
counts mode and parameters (`banksy_<dataset>/`), so the same request again is read back.

**Limits.** The clusters depend on λ, k, the resolution and the seed; they are an exploratory
partition, not a test, and neighbour purity is high by construction for large λ. Domains are only
as good as the cell positions: across a tear or the section edge the "nearest" cells can be far
away (no distance cap is applied). The z-scaling gives every gene the same weight, so on a sparse
5K panel many features are noise; the gene cap matters. Cells without counts get no label.
Recompute after changing the subset, the segmentation or the counts mode.

**Time and memory** (112,551 cells, 2,000 of 5,006 genes, λ = 0.8, k = 15, on a 16-core laptop
that was busy with other jobs): 354 s - neighbourhood features 44 s, covariance 54 s, projection
32 s, neighbours in PC space 72 s, graph 14 s, Leiden 130 s; peak memory 2.5 GB above the loaded
app (4.3 GB in total).

Optional packages: `leidenalg` + `igraph` (Leiden), `pynndescent` (fast neighbours above 40,000
cells).

#### Cluster sets computed in the app

Banksy and the joint clustering install their labels as a cluster set of the view: an integer
column (`clust__<name>` for the Xenium segmentation and extra samples, `cluster_<name>` for a
Baysor / Proseg run) that Color By → Cluster shows in both renderers and that the group pickers
of Differential Expression, the composition test, co-localisation, aggregates and Save as
SpatialData read like a Xenium cluster set. Clusters are numbered from 1 by size; **0 marks a
cell without a label** (outside the subset the set was computed on, or without counts) and is
shown as "Cluster 0" on the map but is in no group of the analyses. On a Baysor / Proseg run the
set is picked in Clustering Method like any other; the run's own KMeans clusters stay the
default. The sets are kept per dataset in `clustersets_<dataset>/` of the cache and come back on
the next start, for extra samples when they are loaded, and for a segmentation run when it is
selected.

### Joint clustering across samples

**Neighbourhoods → Joint clusters.** With two or more samples loaded, the clusters each sample
brought along are not comparable: cluster 3 of one sample is not cluster 3 of another. This tab
clusters the cells of **all samples together** and installs the result as one cluster set that
every sample shares, so pseudobulk differential expression by cluster and the composition test
use the same labels in every sample. It needs the Xenium segmentation (a Baysor / Proseg run
belongs to one sample) and honours the cell subset and the active counts.

| Control | What it does |
|---------|--------------|
| **Integration** | *no integration*: clusters of the joint PCA (set **Joint clusters**). *Harmony on sample / donor / batch*: the principal components are corrected for that variable first (set **Joint clusters (Harmony)**). Donor and batch come from the [sample design](#samples-and-conditions); a sample without one counts as its own level. Without the `harmonypy` package the Harmony choices run uncorrected, the set is named **Joint clusters (uncorrected)** and the status line gives the install hint. |
| **Genes** | Number of variable genes kept for the PCA (default 2,000; a panel with fewer uses all of them). |
| **PCs / Neighbours / Resolution / Seed** | Principal components (30), neighbours of the graph (15), Leiden resolution (1.0), seed (0). |
| **Joint UMAP** | Also compute one UMAP of all cells (needs `umap-learn`). |

**Method.** (1) One matrix of all cells of the view over the panel genes every sample has; counts
per 10,000 of the cell's total over these genes, log1p. (2) Genes are scored **within each
sample** (log of variance ÷ mean of the normalised values, standardised within 20 quantile bins
of mean expression - the "Seurat" dispersion rule) and ranked by the mean score over the samples,
so a gene that differs only between samples is not preferred. (3) Each gene is divided by its
standard deviation over all cells, centred, capped at 10; PCA of all cells (ARPACK on the sparse
matrix, fixed seed). (4) Optional Harmony (Korsunsky et al., *Nat Methods* 2019) through
`harmonypy.run_harmony` on the principal components with the chosen variable as the one
covariate, its defaults otherwise (theta 2, automatic ridge penalty, at most 10 rounds) and the
seed. The corrected embedding is exactly what `harmonypy` returns for those components (tested).
(5) One neighbour graph and one clustering of the embedding with the backend of the Banksy
domains (shared-nearest-neighbour graph → Leiden when `leidenalg` + `igraph` are installed, else
KMeans, named as such); clusters are numbered by size. (6) One UMAP of the embedding.

**Output.** The cluster set is selectable under **Cluster Set** (Color By → Cluster) in both
renderers and wherever clusters are used (DE groups and populations, composition,
co-localisation labels, exports); it is cached and restored with the samples. While it is
selected, the **UMAP panel shows the joint embedding of all samples**, coloured by cluster, gene,
cell type or gene-set score (any other Color By value: by sample); the per-cell coordinates
(`joint_umap_1/2`, `joint_harmony_umap_1/2`) are written by Save as SpatialData for the primary
sample as `ctl_…` columns. Cells without counts have no cluster.

**Diagnostics** (status line, plot, table, CSV), each before and after the correction:

- **Entropy per cluster** of the integration variable among the cluster's cells ÷ log(number of
  levels): 0 when the cluster holds one level only; the entropy of the whole view (reference
  line) when it is mixed as the view is. "Before" uses the clusters of the uncorrected components.
- **Mixing index**: the inverse Simpson index of the variable among the 30 nearest neighbours of
  5,000 random cells (median). 1: all neighbours from one level; the reference is the index of
  the whole view. This is a simplified LISI: unweighted counts over a fixed number of
  neighbours, not the perplexity-weighted index of the Harmony paper.
- The table lists, per cluster, the cells of every sample, the largest sample and its share.

**Caveats.**

- **Integration can remove real differences between conditions.** Harmony removes what
  separates the levels of the integration variable. When the condition is confounded with that
  variable - always for *sample*, and for donor or batch when no donor or batch spans both
  conditions - a cell state that exists in one condition only is, to Harmony, a level effect and
  can be pulled into a cluster of the other condition. The status line states this for every
  Harmony run and says whether the current design is fully confounded. Check the clusters
  against markers and compare with the run without integration.
- With its defaults Harmony does not merge levels whose separation is far larger than the
  differences between cell populations; the diagnostics then stay low and the status line says
  the clusters are still dominated by single levels.
- Clustering describes the cells; it is not a test. Tests between conditions stay at the
  sample / donor level (pseudobulk DE, composition).
- UMAP with a fixed seed runs on one thread (minutes above 100,000 cells); above 300,000 cells
  it runs unseeded on all cores, as elsewhere in the app.
- Results are cached per set of samples, cells, counts and parameters (`joint_<dataset>/`); a
  second run with the same input is read from the cache.

Optional packages: `harmonypy` (Harmony), `leidenalg` + `igraph` (Leiden), `umap-learn` (UMAP).

### Working at study scale (a million cells and more)

Everything cell-level runs on a multi-sample view of more than a million cells; three things
are different at that size.

- **The map.** The app starts in the WebGL view (see *WebGL view*); the Plotly map is not
  built above `XENIUM_PLOTLY_MAX_CELLS` cells and says so.
- **Memory.** A loaded sample holds its counts once, row-wise (8 bytes per stored value). The
  first test or gene colour that reads expression adds a column-wise copy of every sample it
  touches, so count on about twice the matrix plus working space: for half a billion stored
  values, roughly 5 GB after loading and 10–13 GB while a test over all samples runs.
- **Time.** Tests whose cost is linear in the stored values (cell-level DE, pseudobulk,
  spillover, ligand–receptor scores and decomposition, proximity DE) and the spatial ones
  (compartments, stratified co-localisation, aggregates, Moran's I of every gene) each take
  from seconds to about a minute at 1–1.5 million cells on a 16-core machine. Moran's I, the
  rank sums of cell-level DE and the permutations of the aggregates are computed by up to four
  threads; the numbers are the same with one thread (tests assert it).

**Graph clustering is the exception** (joint clustering, Banksy-style domains): the graph
handed to Leiden has about *cells × graph neighbours* edges, and igraph / leidenalg hold them
in Python lists. Before such a job starts it states its estimated cost, e.g. "estimated for
1,400,000 cells with 15 graph neighbours: about 20 GB of memory above the loaded data and
about 22 min (a graph of ~21 million edges)", and it is **not started** when the estimated
memory exceeds half of the machine's memory; the refusal carries the same numbers. The estimate
is 0.4 GB and 0.25 min per million edges (twice the time when Harmony runs: the clusters are
computed before and after) plus the working copies of the count matrix (joint: ~8 GB and
~2 min per million cells, Harmony ~6 min, the UMAP ~15 min; Banksy-style: ~6 GB and ~12 min
per million cells for the neighbourhood features). These constants come from runs on 0.1 and
0.35 million cells and are deliberately on the high side. To cluster more cells than fit:
`subset()` in the REPL, fewer samples, fewer graph neighbours (Banksy-style: *Graph
neighbours* 15 instead of 50), or `XENIUM_ALLOW_LARGE_JOBS=1` to run regardless.

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

### Cell Info Panel

Click any cell in the spatial or UMAP plot to see:
- Cell ID, cluster, X/Y coordinates (µm), transcript count, cell area, nucleus area, UMAP coordinates (a Proseg cell: the area of its outline in µm², and Proseg's cell volume in µm³)
- Bar chart of top 12 expressed genes
- Works for Xenium, Baysor, and Proseg cells; a cell clicked in the view of a segmentation run is looked up in that run (cell ids are not unique across segmentations)

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

Select the algorithm tab (Baysor, Proseg, or Image - see [Image-based resegmentation](#image-based-resegmentation-cellpose--classical)), configure parameters, and click **Run**. Progress appears in the server log.

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
| Initial voxel size (µm) | `--initial-voxel-size` | 4 (blank or 4: `--xenium` preset unchanged; any other value: the preset's column options are passed one by one instead of `--xenium`, because the preset of Proseg 1.1.9 sets the size back to 4 after reading the command line) |
| Threads | `--nthreads` | all cores |
| Components | `--ncomponents` | 10 |
| Recorded samples | `--recorded-samples` | 100 |
| Sampler schedule | `--schedule` | `150 150 300` (whole numbers of iterations only, separated by spaces or commas, at most 12 stages; anything else is refused before Proseg starts) |
| Nuclear reassignment prob | `--nuclear-reassignment-prob` | 0.2 |
| Prior seg reassignment prob | `--prior-seg-reassignment-prob` | 0.5 |

Resegmentation output is cached per run (keyed by parameters) in `~/.xenium_explorer_cache/baysor_{dataset}_{tag}/`, `proseg_{dataset}_{tag}/` or `imgseg_{dataset}_{tag}/`. Multiple cached runs per tool are accessible from the segmentation source dropdown.

#### Image-based resegmentation (Cellpose / classical)

**Resegment Cells → Step 2 → Image (Cellpose / classical).** Baysor and Proseg segment the
transcripts; this tab segments the **morphology image** and then asks which transcripts fall
inside each outline. The result is a run like theirs: an entry `Image: …` in the
Segmentation Source dropdown, its own boundary checkbox (**Image-segmentation Boundaries**,
both renderers), clusters and UMAP, annotation, SpaGE, SPLIT, Save as SpatialData, and a row
in [Segmentation QC](#segmentation-qc) next to Xenium, Baysor and Proseg.

| Control | What it does |
|---------|--------------|
| **Segmenter** | *Classical watershed* (built in, no model) or a Cellpose model. A Cellpose model is listed only when the `cellpose` package imports **and** that model's weights are already in Cellpose's model directory; otherwise the note under the dropdown says what is missing. |
| **Nuclear channel** | One of the bundle's morphology focus channels (read from the SpatialData store of Step 1; default: the channel named DAPI). |
| **Membrane / cytoplasm channel** | Optional second channel, used by the Cellpose `cyto` models only (image handed over as [cytoplasm, nucleus]). |
| **Spatial region / Use Current Viewport** | As for Baysor / Proseg; blank = the tissue ROI of Step 1. |
| **Nucleus diameter (µm)** | Expected diameter (default 8); converted to pixels with the image's pixel size. |
| **Expansion (µm)** | Every label is grown by this distance after segmentation, nearest label wins, no overlap (default 5; 0 = nuclei only). |
| **Min transcripts per cell** | Outlines with fewer assigned gene transcripts are dropped (default 5). |
| **Image patch width (µm)** | The image is cut into overlapping patches with `sopa.make_image_patches` (default 500 µm, 30 µm overlap, clipped to the tissue ROI and the region); patches are segmented one after the other and cells cut by a patch edge are merged with `sopa.segmentation.solve_conflicts`. Cancel is honoured between patches. |
| **Classical: threshold × Otsu** | Scales the classical backend's threshold (below 1: dimmer nuclei are included). |
| **Cellpose: flow threshold / cell probability / GPU** | Passed to `model.eval`. |

**Method.**

1. *Patches*: `sopa.make_image_patches` on a private in-memory copy of the store's image
   element (the dataset's cache store is not modified).
2. *Segmentation* of each patch through `sopa.segmentation.StainingSegmentation` with one of
   two backends behind the same interface (`image (C, Y, X) → label mask (Y, X)`):
   - **Classical watershed** (numpy / scipy; scikit-image's `watershed` when installed):
     Gaussian smoothing (σ = diameter / 10) → Otsu threshold of the patch (a patch whose two
     Otsu classes are not clearly apart is taken as empty) → holes filled → Euclidean
     distance transform, smoothed → its local maxima seed a watershed of the inverted
     distance inside the foreground → objects below 20 % of a disc of the given diameter
     removed → optional expansion. **This is a simple baseline, not a state-of-the-art
     segmenter**: one global threshold per patch misses dim nuclei next to bright ones, and
     the distance transform cannot split nuclei that touch along a broad contact, so dense
     tissue is under-segmented (on a dense skin region it found about half as many objects
     as there are Xenium cells). The expansion is a geometric guess, not a membrane.
   - **Cellpose** (Stringer, Wang, Michaelos & Pachitariu, *Nat Methods* 18:100–106, 2021;
     Cellpose 3: Stringer & Pachitariu, *Nat Methods* 22:592–599, 2025) through
     `sopa.segmentation.methods.cellpose_patch`, with sopa's preprocessing (Gaussian σ = 1,
     CLAHE). The weights file is handed to Cellpose **by its path** after checking that it
     exists, which is the one code path in which Cellpose does not download anything. The
     app never starts a download.
3. *Transcripts → cells*: `transcripts.parquet` is read tile by tile with the filter of the
   cell-feature matrix (Q ≥ 20, gene features only) and every transcript is given to the
   polygon it lies in (2D; a transcript inside two overlapping outlines goes to the first).
   The count matrix is then built, clustered and stored through the same functions as a
   Baysor run (`.partial` working directory renamed when complete, `params.json`,
   `spatialdata_imgseg.zarr`).

`params.json` records the backend, model and weights file, channels, all parameters, the
number of patches (and failed patches), polygons, cells, gene transcripts in the region and
the share assigned to the kept cells.

**Optional packages.** The classical backend needs nothing beyond the app's dependencies
(`sopa`, `spatialdata`, `shapely`). Cellpose: `pip install 'sopa[cellpose]'` (or
`pip install cellpose`); its weights are not part of the package - the first Cellpose run
downloads them from the Cellpose project into `~/.cellpose/models`. Do that once yourself
(e.g. `python -c "from cellpose import models; models.CellposeModel(model_type='nuclei')"`),
then the model appears in the Segmenter dropdown. With Cellpose ≥ 4 the only model is `cpsam`.

**Limits.** 2D: the focus image is one plane and transcripts are assigned by x, y only. A
nuclear stain with a fixed expansion says nothing about the true cell outline; with a
membrane channel the Cellpose `cyto` models can do better. Cellpose on the CPU takes about a
minute per 500 µm patch at full resolution - use a region, or the GPU option. Step 1 (Prepare
Patches) must have built the SpatialData store, because the image is read from it; a dataset
loaded from a `.zarr` has neither image nor transcript file.

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

SpaGE always imputes from the **original** counts of the active segmentation, also when Active Counts is set to "SPLIT Corrected" (the status line says so). The Impute Genes dialog and `run_spage()` always use the original counts (`run_spage(use_corrected=True)` prints why and runs nothing), because the app stores imputed genes as plain `[imp]` columns and a corrected-input run would overwrite them under the same name. To impute from SPLIT-corrected counts use the standalone script: `python functions/impute.py <store.zarr> <ref.rds> --use_corrected` reads the `X_corrected` layer, writes a new store, and lists the genes in `adata.uns["split_corrected_imputed_genes"]`; when that store is opened in the app they are labelled `[corr+imp]`.

**REPL (programmatic):**
```python
run_spage(
    rds_path="/path/to/reference.rds",
    genes_file="/path/to/genes_to_impute.txt",  # one gene per line; omit for auto-HVG
    n_pv=50,
)
```
Progress is printed to the server log. When complete, imputed genes appear in the Color By dropdown automatically.

Imputed genes appear in the gene expression dropdown with an `[imp]` suffix. Results are **written back into the active SpatialData zarr** (`spatialdata_xenium.zarr` for Xenium, or `spatialdata_{source}.zarr` for Baysor/Proseg) so they persist across restarts — imputed genes are available immediately on the next launch without re-running SpaGE. When a Baysor/Proseg segmentation is active, only genes imputed for that segmentation are shown in the dropdown (those in its store and those of a streamed result). A large result (more than 400,000 cells, or more than 5 × 10^8 values) is streamed to a file in the cache and not written into the store; it is read by row position, checked against the cells it was computed for, and takes precedence over stored columns of an earlier run of the same genes, as does a result whose write into the store was skipped (subset active). The parquet cache (`~/.xenium_explorer_cache/spage_*.parquet`) is also written for backward compatibility.

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
- **Cells without corrected counts are left out, not counted as zeros** (which analyses, and how it is reported: [Statistical conventions](#statistical-conventions)). SPLIT returns no values for the cells RCTD dropped (too few counts) and for cells outside the subset or ROI it was run on; the corrected matrix has an empty row for each. The gene colour of such a cell is "no value" in both renderers. A cell with no counts at all is unchanged (zeros in both modes).

**SpaGE on corrected counts:** not available inside the app; see [SpaGE Gene Imputation](#spage-gene-imputation).

**SPLIT under an active `subset()`:** the corrected counts, clusters and UMAP of the subset are kept in memory for the session and are **not** written to the store (the status line says so); an earlier correction of the whole dataset stays in the store and keeps its rows for the cells outside the subset. Run SPLIT without a subset to store its result.

**Several samples.** SPLIT runs on the primary sample; an added sample carries corrected counts only if its own store already holds them (the `X_corrected` layer written by an earlier SPLIT run on that sample, or a store exported by Save as SpatialData). With several samples loaded, **"SPLIT Corrected" is available only when every sample has corrected counts** (the rule and what it applies to: [Statistical conventions](#statistical-conventions)). Otherwise the option is disabled and a note under Active Counts names the samples that lack corrected counts. *Behaviour change:* corrected counts used to be ignored whenever several samples were loaded, while the option stayed selectable.

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
| CellTypist / Seurat annotation | step text | stops after the current step |
| RCTD annotation | step text | the R worker process is terminated at once |
| SPLIT | step text | during the R step (RCTD, `purify()`) the R worker process is terminated at once; afterwards it stops after the current step |
| Pseudobulk and contact-group differential expression | step text; label shuffle *i* of *n* | during a model fit the R worker process is terminated at once; otherwise at the next step |
| SpaGE | step text; fraction of gene chunks during the imputation itself | stops after the current step / gene chunk |
| Rerun UMAP, spatial analysis, niches, CellNEST load | step text (steps *k* of *n* for spatial analysis and niches) | stops after the current step |
| Save as SpatialData | step text | only before the store is written |

How cancelling works, and its limits:

- **External tools** run in their own process group. Cancel sends `SIGTERM` to the group and `SIGKILL` five seconds later to whatever is left, so Julia and any child processes go with it. The tools are also stopped, and their working directories removed, when the app exits or is stopped with Ctrl-C, `SIGTERM` or `SIGHUP`; after `kill -9` of the app a running tool has to be stopped by hand.
- **Long R steps run in a worker process, which Cancel terminates at once.** The pseudobulk and contact-group model fits (DESeq2 / edgeR), RCTD annotation and the R part of SPLIT (RCTD in doublet mode and `SPLIT::purify()`) run in a second Python process that loads R itself (`xenium_app/rworker.py`). It is a registered child process of the job like Baysor and Proseg: Cancel sends `SIGTERM` to its process group, R stops within a fraction of a second together with anything it started, the memory R held is returned to the system, the job ends as cancelled and publishes nothing. Input and result travel as JSON and `.npz` files in a private temporary directory that is removed when the step ends, is cancelled or fails, and when the app exits. Both paths run the same Python and R statements on the same values; a pseudobulk fit is bit-identical either way. The price is the start of the worker for each R step: about 0.3 s for R, plus the load time of the packages (about 3 s for DESeq2), which R inside the app pays only once per session; the contact-group test uses one worker for all its fits. If the worker cannot start, the step runs inside the app and the status line (pseudobulk, contact) or the server log (RCTD, SPLIT) says so. `XENIUM_R_INPROCESS=1` runs every R step inside the app as before; Cancel then waits for the R call in progress. The R package versions in the provenance record are read from the R that did the work, in either case.
- **Python steps, and the R that still runs inside the app, cannot be interrupted.** The job checks for a cancel request between steps, so the panel shows *cancelling after the current step…* until the step in progress returns. Still inside the app: reading the Seurat reference for Seurat kNN annotation and for SpaGE (`readRDS` and the export of the shared-gene matrix, seconds to a few minutes for a large reference). `sopa` reading the Xenium bundle or building patches and a UMAP fit are other long single steps. `baysor --version`, which the app runs to find Baysor before the segmentation starts (Julia start-up), is also not interrupted.
- **Nothing half-written is kept.** A Baysor/Proseg run works in `<run directory>.partial` and gets its real name only when it is complete; a cancelled or failed run removes that directory, so it never appears in the Segmentation dropdown and an earlier run with the same parameters stays as it was. Once a job starts writing its result (run directory and zarr store, SPLIT / SpaGE write-back to the SpatialData store, annotation caches) it can no longer be cancelled; the row then reads *writing results — cannot be cancelled now*. A streamed SpaGE result that was cancelled is deleted.
- **A run that a job works on is not replaced or deleted.** Jobs that work on a Baysor / Proseg / image-segmentation run are registered with its directory. While one runs, a new run of that tool, a re-run into the same directory and **Delete run** are refused with a message that names the job; wait for it or cancel it first. A run that is being loaded from the cache cannot be deleted either.
- A cancelled job reports `✗ Cancelled` in its status line, the same job can be started again straight away, and the loaded dataset is not changed. As before, starting Baysor/Proseg replaces the run that was loaded for that tool, and an annotation run clears that method's previous labels, at the moment the job is started, not when it is cancelled. Intermediate results that are complete and reusable are kept: the SpatialData store, tissue ROI and patches of Prepare Patches, and the RCTD result SPLIT caches before `purify()` (with its labels).
- Loading another sample is still refused while any job is running, including one that is being cancelled.
- The bar is the share of the whole job: 90 % for the tool (split evenly over the patches, whatever their size), the rest for the expression matrix, clustering / UMAP and the store. It is not a time estimate.
- A failed Baysor/Proseg run no longer leaves its partial output in the cache (the tool's messages are in the server log).

From the REPL: `_jobs_list()` returns the registered jobs (`.kind`, `.label`, `.state`, `.step`, `.fraction`, `.elapsed()`, `.thread`, `.procs`), `_job_cancel(job.id)` cancels one.

### Sessions: save and restore

The settings of a session live in the browser page and in the server process, so they are lost when the server restarts. **Session: save / restore…** (sidebar, under the dataset block) opens a small dialog:

| Action | What it does |
|--------|--------------|
| **Save session** | writes the session of the loaded dataset to `<cache>/sessions/` (one saved slot per dataset) |
| **Download…** | the same session as a JSON file |
| **Restore saved session** | applies the saved slot |
| **Restore last session** | applies the automatic save of the *previous run* of the app on this dataset |
| **Restore from file…** | applies an uploaded session file |

The app also saves the "last session" of the dataset by itself about once a minute while a page is open and something changed. Nothing is ever restored without one of the three Restore actions: a freshly started app shows its defaults. The first automatic save of a server run keeps the file of the run before, so "Restore last session" still means the previous run after the new run has started saving. Session files are not removed by "Clean cache".

**What a session file holds** (settings and choices, as one JSON document with a format version, the app commit, a timestamp and the identity of the data: the dataset key of the provenance log, and the name and number of cells of every sample):

- the value of every control on the allow-list in `xenium_app/session.py`: segmentation source, counts mode, active cluster set, Color By, gene, view settings, the annotation choice, and the parameters of every analysis tab and dialog;
- the sample design (conditions, donors, batches, condition names);
- ROIs, gene sets and saved DE groups (their outlines and gene lists);
- the filters of an active `subset()`.

**What it does not hold**: results of any kind (matrices, labels, cluster sets, tables, figures). After a restore, run an analysis again and its result comes back from the cache the app already keeps. Also not saved: path fields (reference `.rds`, CellNEST folder, sample path, SpatialData output), CellNEST results and their filters, the `.h5ad` mapping dialog, scratch fields of dialogs, the open tab, selections and the viewport, and a subset that was made from an explicit list of cell ids or from 50 or more values (the app records only how many; saving says so). The table at the top of `session.py` lists every item with the reason.

The design, ROIs, gene sets and DE groups already have their own cache files, which the app loads at start-up. A restore goes through the same functions as their dialogs (so those files are rewritten) and only adds or replaces: an ROI, gene set or group that exists now and is not in the session is kept, and the report says so.

**Security rules of restore.** A session file is input from outside, also when the app wrote it.

- JSON only (at most 8 MB, no `NaN` / `Infinity`, no repeated key). Nothing in the file is unpickled, evaluated, executed or imported.
- Every value is checked against an allow-list before it is applied: a known key, the right type, the range of its input or the options of its control, a pattern for text. A value that fails is refused and named in the report; an unknown key is ignored and named.
- Values that name something of the data (a segmentation run, a cluster set, a label source, a cell type, a gene, a group) are applied only when that thing exists now.
- No path in the file is opened, read or written. A path field is never restored. A sample the session had beside the primary one is added only when the Load Sample picker itself lists exactly that path (the same directories the dialog offers); otherwise it is reported as "needs loading" and you add it with '+ Add sample'. The free-text path field of that dialog has no counterpart in a session file.
- The REPL gate (`XENIUM_ENABLE_REPL`), the host bind (`XENIUM_HOST`), the port and the cache and data directories are settings of the server process; no key of a session file reaches them, and the REPL input is not a session control.

**When the data is not the same.** A session saved for another dataset key, another number of cells or another set of samples is restored *in part*: only the plain parameters (numbers, switches, fixed choices, text lists) are applied. The segmentation source, cluster set, colouring, label sources, cell types, genes, region boxes, the image-overlay controls and the renderer choice (their options belong to the loaded images and data size), the design, ROIs, gene sets, DE groups and the subset are skipped, and the report lists them. Cluster, label and ROI state is never applied to other cells. A session that names a segmentation run or a cluster set that no longer exists restores the rest and names what is missing; a session written by a newer version of the format is refused.

**Jobs.** Restore is refused while a job is running (the same rule as Load Sample). Loading the cached segmentation run a session names goes through the Segmentation dropdown's own callback.

Explicit saves and every restore are written to the provenance log (kind `session`); the automatic save is not. In batch mode, `session: file.json` applies a session before the steps: its control values become the defaults of the steps, while a step's own `params`, the config's top-level keys (`seg_source`, `counts`, `cluster_method`, `color_by`) and a `data.design` still take precedence. With `--dry-run` the session is applied in memory for the checks only: no design, ROI, gene-set or group file, no store and no provenance record is written.

Limits: a restore is applied step by step, not as one transaction (a refusal is decided before anything is changed; a value that turns out to be missing is skipped and the rest continues). A second browser tab on the same server shares the automatic save; a page that was only opened or reloaded saves nothing until something in it is changed. When the gene list of a session differs from the stored gene set of the name it had selected, the list is restored and the gene-set picker is left empty (selecting the set would replace the list); the report says so. A dataset that was moved to another directory has another key and is restored in part.

### Server Log & Python REPL

The bottom panel is split into two columns:
- **Left** — Cell info on click (see [Cell Info Panel](#cell-info-panel))
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
| `sessions/{id}.saved.json`, `.last.json`, `.prev.json` | Session files of a dataset: the slot of **Save session**, the automatic save of the current server run and that of the run before (settings only; `{id}` is the stem of the dataset's provenance file; kept by "Clean cache") |
| `density_{hash}/{gene}__b5um__qv20_genes_v1.npz` | Transcript-density grid of one gene (non-empty 5 µm bins); the name carries the bin size and the version of the transcript filter |
| `txfactor_{hash}/counts__{shape}_b{bin}um__qv20_genes_v1__g{genes}.npz`, `txfactor_{hash}/fit_{settings hash}.npz` | Transcript factors: the bin × gene counts of one sample for a bin size and shape (reused for any K, method or seed), and a finished fit (loadings and the factor counts of every bin) for one set of settings |

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
- `cell_area` and `nucleus_area` are included in `@meta.data` when available in the zarr obs. For Proseg runs written by the app, `cell_area` is the area of the cell's outline (µm²) and `cell_volume` is Proseg's `volume` (µm³); a store that only has Proseg's `volume` column gets `cell_volume`. A Proseg store written by an earlier version of the app holds the volume under the name `cell_area` until the run has been loaded in the app once (the load corrects the store).
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
| `cellpose` | Cellpose backend of the image-based resegmentation (optional; needs its model weights on disk - the app never downloads them; without it the built-in classical watershed is the only backend) |
| `celltypist` | CellTypist cell-type annotation |
| `umap-learn` | UMAP recomputation on reseg cells; joint UMAP |
| `leidenalg`, `igraph` | Leiden clustering of Banksy and the joint clustering (optional; without them KMeans on the same embedding, named as such) |
| `pynndescent` | approximate nearest neighbours in PC space above 40,000 cells (optional; exact search otherwise) |
| `harmonypy` | Harmony integration of the joint clustering across samples (optional; without it the joint clusters are uncorrected) |
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

## Cloud Run Deployment

The app can be deployed to **Google Cloud Run** for remote access. The container files are in the repository root: `Dockerfile`, `.dockerignore`, `requirements-cloud.txt` and `deployment_instructions.txt` (the step-by-step guide). **The image has not been built or tested yet** — treat the files as a starting point and check a first build before relying on it.

- The image is **Python-only**: no R / rpy2, Baysor (Julia), Proseg (Rust) or sopa. Seurat and RCTD annotation, SpaGE, SPLIT, pseudobulk differential expression (DESeq2 / edgeR), resegmentation and **Prepare Patches** are therefore unavailable there; CellTypist annotation and the Python analyses work.
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

### Environment variables

All optional locally, required or useful on Cloud Run:

| Env var | Purpose |
|---------|---------|
| `DATA_DIR` | Path to Xenium output directory (overrides `sys.argv[1]`) |
| `XENIUM_CACHE_DIR` | Cache root directory (default: `~/.xenium_explorer_cache`) |
| `PORT` | Server port (default: 8050; Cloud Run sets this automatically) |
| `XENIUM_HOST` | Bind address (default: `127.0.0.1`; set `0.0.0.0` in a container or to serve other machines). The address in use is printed at start-up |
| `XENIUM_ENABLE_REPL` | Enables the in-app Python REPL, which runs arbitrary code. Unset = disabled; `1` = requests made from this machine to `localhost` / `127.0.0.1` / `[::1]` only (not through a reverse proxy); `remote` = any client (trusted networks only) |
| `XENIUM_RENDERER` | Spatial renderer shown at startup: unset = Plotly; `webgl` = the WebGL view (it can also be switched in the app) |
| `XENIUM_REF_RDS` | Path of the reference `.rds` file (a Seurat object with cell-type labels) that prefills the *RDS File Path* fields of Cell Type Annotation (Seurat kNN, RCTD), SPLIT and SpaGE. Unset = the fields start empty; Run then asks for the path instead of starting |
| `XENIUM_IMPUTE_GENES` | Path of a text file with gene names (one per line or comma-separated) that prefills *Genes to Impute* of the SpaGE dialog. Unset = empty (the 200 most variable reference genes outside the panel are imputed) |

Further variables are described where they are used: `XENIUM_PLOTLY_MAX_CELLS` ([WebGL view](#webgl-view-beta)), `XENIUM_H5AD_*` ([AnnData `.h5ad`](#anndata-h5ad-processed-cells-one-or-several-sections)), `XENIUM_ALLOW_LARGE_JOBS` ([Working at study scale](#working-at-study-scale-a-million-cells-and-more)) and `SPAGE_REPO` ([SpaGE Gene Imputation](#spage-gene-imputation)).

## Continuous integration

`.github/workflows/tests.yml` runs the **default pytest tier** on every push and pull request to `main` (Python 3.11, Ubuntu). The `tools`, `browser` and `slow` tiers are not run in CI (they need Baysor / Proseg / R, a browser with WebGL, or large bundles). If the fixture bundle (`tests/data/prime_mouse_ileum_tiny/`) is not in the checkout, the tests that need it are skipped with a warning.

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
| `imgseg` | image-based resegmentation: the segmenter interface (classical watershed, Cellpose via sopa with local weights only), sopa image patches, transcripts assigned to the polygons, run and cached-run loading |
| `split`, `spage` | SPLIT ambient RNA correction; SpaGE imputation |
| `subsetting` | the REPL helpers `subset()`, `unsubset()`, `get_genes()`, `retry_split_zarr_write()` |
| `spatial`, `regions` | spatial analysis and niches; ROI helpers |
| `morph` | morphology image overlay of the Plotly view |
| `figures` | Plotly figure builders (spatial map, UMAP, analysis plots) |
| `analysis` | differential expression, gene-set scores, transcript density |
| `h5ad` | AnnData `.h5ad` input: column mapping, one-time conversion into per-section stores, design prefill |
| `stats` | sample → condition / donor design (persisted), exact sample-level tests, pseudobulk differential expression (DESeq2 / edgeR), composition test, Moran's I, ligand–receptor scores |
| `controls` | confound controls: tissue compartments, spillover score, gene ownership, leak gate, segmentation QC metrics, stratified co-localisation, annotation label report |
| `proximity` | proximity differential expression (near vs far from an anchor cell type, matched random anchors) and contact-group pseudobulk differential expression |
| `lrdecomp` | ligand–receptor exposure decomposition into five factors, per-term sample-level tests |
| `aggregates` | aggregates: connected groups of cells of a chosen class, outlines, composition, per-sample density |
| `segfree` | segmentation-free abundance: marker transcripts per mm² of tissue from the transcript file, area-bin sweep, concentration against a redistribution null |
| `clustersets` | cluster sets computed in the app: nearest-neighbour graph → Leiden (KMeans fallback) on an embedding, installing the labels as a cluster set of the view, their cache |
| `banksy` | Banksy-style spatial domains: neighbourhood mean and azimuthal Gabor filter features, PCA of the neighbour-augmented matrix, the job and its tab |
| `txfactor` | transcript factors: bin × gene counts of the transcript file on a square or hexagonal grid, NMF (KL) / online LDA / reference profiles, projection of every bin, map layer, per-cell factor mixture |
| `joint` | joint clustering across samples: joint normalisation, gene selection and PCA, Harmony (harmonypy), mixing diagnostics, the joint UMAP of the UMAP panel |
| `cellnest` | CellNEST loading, filters and plots |
| `sdata_store` | SpatialData / sopa backend: cache store, Save as SpatialData, tissue ROI and patches |
| `startup` | the import-time work: locate and load the dataset, autoload cached results |
| `app_layout` | the Dash `app` and its layout |
| `provenance` | methods and provenance record: the one call every analysis makes (`_prov_record`), the per-dataset record file, the Provenance tab, the JSON export and the methods text with its per-analysis templates and references |
| `batch` | headless batch mode (`python -m xenium_app.batch`): config, validation, steps run through the Run callbacks and the job registry, output files |
| `session`, `cb_session` | session save / restore: the state inventory and the allow-list of controls, validation of a session file, snapshot, restore through the functions of the UI, the automatic "last session", the dialog and its callbacks |
| `cb_plots`, `cb_tools`, `cb_reseg`, `cb_spage`, `cb_data`, `cb_spatial`, `cb_analysis`, `cb_stats`, `cb_controls`, `cb_proximity`, `cb_lrdecomp`, `cb_aggregates`, `cb_segfree`, `cb_banksy`, `cb_txfactor`, `cb_joint`, `cb_nav`, `cb_provenance` | the callbacks, by feature |
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
node test/nav_checks.mjs http://127.0.0.1:8050 out/  # the two-level navigation: every tab opens, figures fill their box, no tab switch recomputes
node test/provenance_checks.mjs http://127.0.0.1:8050 out/  # the Provenance tab: the record list, a row click opens the full record, kind filter, both exports
# (the browser scripts open a tab with openTab(page, 'tab-de') from test/lib.mjs, whatever group it is in)
node test/analysis_checks.mjs http://127.0.0.1:8050 out/ # Differential Expression / Gene Sets / Transcript Density tabs in both renderers (lasso → groups → test → gene set → density)
XS_SAMPLES=/bundle2,/bundle3,/bundle4 node test/stats_checks.mjs http://127.0.0.1:8050 out/ # several samples: design editor, pseudobulk DE, composition, density of added samples in both renderers, Moran's I, ligand-receptor (see the top of the file)
node test/proximity_checks.mjs http://127.0.0.1:8050 out/ # Differential Expression: "Proximity to a cell type" and "Contact groups" (with XS_SAMPLES=… the contact test runs in R; without, its single-sample refusal is checked)
node test/webgl_fallback.mjs http://127.0.0.1:8050 # automatic fallback to Plotly without WebGL2 (browsers started with WebGL disabled)
XA_CLASS="B cells|Plasma cells" node test/aggregates_checks.mjs http://127.0.0.1:8050 out/ # Aggregates tab in both renderers: run, table, Color By, outlines, toggle, alpha hulls, CSV (needs a cell-type annotation in the cache)
XJ_SAMPLES=/path/s1,/path/s2 node test/joint_checks.mjs http://127.0.0.1:8050 out/     # Joint clustering tab in both renderers: one-sample refusal, added samples, run (Harmony when installed), diagnostics, the shared Cluster Set on the map, the joint UMAP of all samples in the UMAP panel (WebGL and Plotly), CSV
XB_LAMBDA=0.8 node test/banksy_checks.mjs http://127.0.0.1:8050 out/   # Banksy tab: run, table, the cluster set in Clustering Method and on the map in both renderers, second run from the cache
TXF_K=8 node test/txfactor_checks.mjs http://127.0.0.1:8050 out/ # Transcript Factors tab: fit, table, cross-tab, map layer and mixed score in both renderers, reference mode (needs transcripts.parquet; the first run bins all transcripts)
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

- **A sample without the chosen cluster set is left out of cluster statistics.** Composition by cluster, the pseudobulk populations "Cluster N", ligand–receptor scores by cluster and the cluster colours of the joint UMAP compare the cluster ids of several samples. A sample that does not hold the set chosen under *Clustering Method* (a set computed on one sample only, a sample added later, a `.zarr` / `.h5ad` sample with its own set names) used to contribute the ids of its own first cluster set under the same numbers, without a message. It now has no cluster labels: composition names it as left out, and a pseudobulk population reads "cluster 3 (S1: without this cluster set, no cells)". The map still colours such a sample by its own first set.
- **Pseudobulk and contact-group DE after SpaGE**: the check that the counts are whole numbers looks at the measured genes only. A dataset whose matrix also holds imputed genes (fractions by construction, never summed) used to be refused as "non-integer counts".
- **Smallest achievable p of the exact tests** is the floor of the design (2 / 2^n for n pairs, 1 / C(n, n_A) or 2 / C(n, n_A) for an unpaired design), in the composition test, the aggregates, segmentation-free abundance and the between-condition tests of co-localisation and proximity DE. It used to be taken from one test's values, where ties or zero differences raise it (a category present in one sample only gave "smallest achievable p: 1 - no test can be significant" beside significant categories).
- **Composition CSV**: every line of the header block starts with `#` (the warning of an exact test with few units was written as a line without it, which a reader that skips `#` lines took for the column header). All CSV builders now write their header through one function.
- **Library size when panels differ**, second part: the ligand–receptor exposure decomposition (R, L and the ownership filter), gene ownership, the derived markers, the leak gate and the spillover score take the depth of a cell over the genes the pooled samples share, as cell-level DE and the ligand–receptor score already did. Results of a view whose samples have different panels change; status line, CSV header and methods text state the rule ("library size (depth) of a cell: its total over the N genes shared by the pooled samples (their panels differ)"). With one panel nothing changes.
- **Donor labels on some samples only**: stratified co-localisation, proximity DE and contact-group DE combine the sections of a labelled donor into one unit also when another sample has no donor label (each section used to be a unit in that case, so two sections of one donor could make "n = 3" out of 2 independent units). The units note says, for example, "n = 3 samples (1 donor, 1 sample without a donor), 2 independent units after combining the samples of one donor"; contact-group DE then fits `~ unit + group`.
- **Proseg initial voxel size**: a value other than 4 µm is now applied. It used to be passed beside `--xenium`, whose preset sets the size back to 4 µm after the command line is read (Proseg 1.1.9), so every earlier run was made with 4 µm voxels whatever its label said. Such a run is started with the preset's column options written out instead of `--xenium`. Cached runs that asked for another size are listed as "voxel=4µm (N µm was requested but not applied)" and keep their directory; a new run with that size gets its own.
- **Cell area of a Proseg run** is the area of the cell's outline in µm² (`cell_area`). Proseg reports a volume (µm³), which the app used to show, filter (`subset(max_cell_area=…)`) and compare (Segmentation QC median area) as an area; it is now kept as `cell_volume`. Cached runs are corrected the first time they are loaded (table in memory and the run's store); `functions/spatialdata2seurat.py` exports Proseg's `volume` as `cell_volume`.
- **A segmentation run that another job works on** is not replaced or deleted: a new run of that tool, the commit of a re-run into the same directory and **Delete run** are refused with a message naming the job (SPLIT, SpaGE, Rerun UMAP, annotation, a cached load …). A re-run with the same parameters used to replace the directory under a running SPLIT, whose corrected counts were then written into the new run and matched to other cells by id.
- **SPLIT, SpaGE and spatial analysis on a run that is not loaded** (still loading, or its load failed) are refused, like annotation, niches and Rerun UMAP; they used to run on the Xenium cells and write the result into the Xenium store under the run's name.
- **SpaGE**: a streamed result (large outputs) or one that could not be written into the store replaces the stored columns of an earlier run of the same genes on the map (the earlier values used to stay on display while status and provenance named the new run); a streamed result of a Baysor / Proseg run is matched to the run's cells (it could be drawn on other cells after `subset()`), and its genes are offered in the gene dropdown beside the stored ones.
- **SPLIT status line**: "N cells corrected" counts the cells whose counts purify() changed, compared over the genes it returned; panel genes that are not in the reference (no corrected counts) used to make every cell with a count in such a gene "corrected". Their number is given separately.
- **SPLIT-corrected mode after a restart, with imputed genes in the store**: a cell without measured counts keeps its (empty) corrected value, as in the session that ran SPLIT; it used to be counted among the cells "without SPLIT-corrected counts" because the imputed columns were taken for original counts.
- **ROIs**: `roi_*` labels that came with a loaded store and have no outline in this cache directory are kept when a ROI is saved or deleted (all of them used to be blanked, and the blank columns written back into the store). With samples side by side, the cells of an added sample inside an outline get the label as well (the Save ROI dialog counted them, the columns did not), and a Plotly lasso over an added sample gives the outline of the selected points (it was built from cells of the primary sample with the same ids).
- **Multi-sample colour by an imputed gene**: added samples that store the gene are drawn with their values (they were drawn as 0).
- **CellNEST sender / receiver filters** work on numeric annotation columns (`cluster_kmeans_10`): they matched nothing.
- **Cell info** of a cell clicked in a Baysor / Proseg / image-segmentation view is looked up in that run first (with numeric cell ids the Xenium cell of the same id was shown).
- **Joint UMAP restored under a subset** covers the whole sample once the subset is cleared.
- **Batch mode**: a `contact` step no longer needs conditions A / B in the design (it never read them), and the transcript steps (`density`, `segfree`, `txfactor`) are accepted when any loaded sample has `transcripts.parquet` or `transcripts.zarr.zip`. Both used to end a dry run, and mark a run that succeeded, as failed.
- **Provenance**: the proximity DE record states that the sample design was used (its methods text had no design paragraph).
- **Sessions**: choosing the same file under **Restore from file…** a second time restores it again (nothing happened); a page that was only reloaded no longer writes its default values over the automatic save of the running server; a restored gene list that differs from the stored set of the same name is kept (the picker is left empty and the report says so); a session file with an absurdly large integer is refused value by value instead of ending in an error.
- **Figure export**: every character of Windows-1252 is written to the PDF (œ, Š, ž, ™, ‰ and 13 more were printed as `?`); characters XML does not allow no longer produce an SVG that cannot be opened.
- **Contact-group pseudobulk DE uses the original counts**, also in "SPLIT Corrected" mode (it used to sum and round the corrected counts, against the rule that a count model is not given corrected counts); a matrix with non-integer values is refused as in the pseudobulk mode. Results obtained in corrected mode change.
- **Clusters of a Baysor / Proseg run in "SPLIT Corrected" mode**: ligand–receptor scores and decomposition, proximity and contact DE, spillover, composition, compartments, aggregates and the pseudobulk populations group by the SPLIT clusters, like the map and the DE group picker (they used the clusters of the original counts).
- **A sample without labels has no value, not zero**: aggregates (density, area fraction) and segmentation-free abundance (cells of the type per mm², share assigned to the type) give no value for a sample that has no annotation, name it in the status line, and do not test that value between conditions (it used to enter the test as 0).
- **Pseudobulk gene list**: the tested genes are those shared by the samples that are used; a sample dropped for too few cells, for no counts or by the paired rule no longer narrows the list. *Dropped* names each sample with its own reason.
- **Composition test**: categories, the Benjamini–Hochberg family and the *n_cells* column come from the tested samples only (samples of a donor left out by the paired rule used to add categories and cells).
- **Spillover**: a cell that was not scored (no SPLIT-corrected counts in corrected mode) has no flag (it used to be flagged "none"); *n* beside the reference counts the source cells the reference is computed on.
- **Sample design**: a design file saved for a set of samples fills in only samples that have no assignment in the session (it used to overwrite newer assignments when the set of loaded samples changed back). In batch mode the design of the config replaces the saved one.
- **A re-run segmentation with the same parameters** is a new run (it has a run id): control columns, cached cluster sets and annotation labels of the replaced run are not attached to its cells.
- **SpaGE**: a result served from the cache is written into the matrix and the store like a fresh one (the map could show the values of another run of the same genes); a result computed under a `subset()` is not registered for auto-load.
- **SPLIT**: its RCTD labels no longer replace an annotation of all cells when it ran on a `subset()`, and the RCTD weights of the replaced annotation are dropped with its labels.
- **Provenance and methods text**: records state the counts that were used (not the mode asked for), niches without labels are described as spatial blocks, the proximity text follows the estimator / FDR / covariate of the run, the UMAP record is written once and says which counts were embedded, a cached segmentation run is recorded once per session and does not replace the record of the run that computed it, the `.h5ad` conversion is recorded under the converted dataset, and `subset()` records only the filters of the last call that were applied.
- **Batch mode**: a step runs on the segmentation run it names (the run is loaded when another is in memory; a missing run is an error), the dry run checks against that run's cells, the annotation and gene-set steps write `cells.parquet`, and a step that produced no result of its own fails.
- **"% of cells expressing" on SPLIT-corrected counts** (DE `% A` / `% B`, the ligand–receptor detection threshold, Moran's `% cells`, genes per cell in Segmentation QC): a cell detects a gene when its count is **≥ 1 transcript**. For integer counts this is the former count > 0; for corrected (fractional) counts a residue below one transcript no longer counts, so these percentages are lower than before. The DE table also gives **Rate A / Rate B** (Σ counts ÷ Σ total counts of the group, per 10,000), which is defined for fractional counts and does not move with depth; `low_count` marks genes with fewer than 100 transcripts over both groups.
- **Every CSV the app writes** has one more comment line at the end of its header, `# provenance record: prov-…` (see [Methods and provenance record](#methods-and-provenance-record)). Readers that skip `#` lines are not affected; a script that counts header lines is.
- **Pooled cell-level DE and ligand–receptor scores of samples with different panels**: the library size of a cell is its total over the genes the pooled samples share (it was the total over each sample's own panel, which lowered every shared gene in the sample with the larger panel). log2 fold changes, p-values, Rate A / Rate B and the median counts per cell of such a comparison change; with one panel nothing changes. The median counts per cell of a pseudobulk result follow the same rule.
- **Composition, moderated test, with donor labels on some samples only**: the sections of a labelled donor within a condition are one unit, as in the exact test (every section used to be a replicate). The units note counts the donors that were given ("4 donors, 1 sample without a donor").
- **Rerun UMAP on corrected counts** embeds every cell that has corrected counts, also after a SPLIT run under `subset()` (only that run's cells were embedded). Above 300,000 cells the record no longer states a UMAP seed: UMAP runs unseeded there.
- **SpaGE**: a result computed under `subset()` is cached per set of cells and read back by cell id (two subsets of the same size shared one cache entry and could be shown on each other's cells); the cache of a Baysor / Proseg run belongs to that run (a re-run with the same parameters recomputes; imputations of runs cached by earlier versions are recomputed once). A gene entered twice is imputed once.
- **Old bundles with numbered cells** (integer `cell_id`): the expression matrix is matched to the cells (it was all zero, without a message). A `spatialdata_xenium.zarr` written from such a bundle by an earlier version must be deleted so that it is rebuilt. Cells and matrix that do not correspond at all are an error.
- **Methods text**: the paired-design sentence follows the rule of the tests; proximity names the density-covariate type that was used (not "auto"); spillover and segmentation-free abundance state the marker rule and reference that were used; compartments without labels describe no core; a Cellpose run is described as one. Contact-group DE names `~ donor + group` in its status line, title and CSV header when donors were pooled.
- **Batch mode**: `data.design` entries resolve by label, then path, then name (a name could shadow another sample's label); `--dry-run` no longer replaces the saved design; a `clustering` step with corrected counts exports `split_umap_1/2`; a `niches` step with extra samples writes its `cells.parquet`.
- **Plots panel**: the tabs are grouped in two levels; Composition and Segmentation-free abundance are tabs of their own (they were sections of Cell Type Abundance). No number changes.
- **DE status line and CSV** give the median counts per cell of both groups and warn when they differ by more than 1.5×; they state the unit of the p-values.
- **Pseudobulk volcano**: coloured by FDR alone; x axis = unshrunken log2FC unless the shrunken one is chosen (see above). The cell-level volcano keeps its |log2FC| ≥ 0.5 rule.
- **Sample design**: has a donor field, is saved per set of samples and restored (it used to be memory-only and lost on Load Sample).
- **SPLIT-corrected mode leaves out cells without corrected counts** (see *Active Counts toggle behavior*). Before, they entered every corrected-mode test and score as all-zero cells: on a section where SPLIT had been run on a region of 1,067 of 112,551 cells, a cluster-vs-rest test compared 8,781 with 103,770 cells of which 99 % were padding zeros (891 genes with FDR < 0.05, 3 of them with |log2FC| > 1); it now compares the 169 and 898 cells that have corrected counts (761 genes, 533 with |log2FC| > 1).
- **p-values below the float range** (|z| above about 38) are written as `1e-300` instead of `0`, in the DE and Moran tables and CSVs; the CSVs keep `neg_log10_pval` and `neg_log10_padj` with the exact values, and "most significant first" orders by them.
- **Pseudobulk with donors**: unpaired designs with donor labels are no longer refused as confounded; see *Donors* under pseudobulk. An excluded sample no longer narrows the tested gene list.
- **Stratified co-localisation**: with a design, the effect, fold and sample counts of a pair come from the tested samples only (they used to include samples left out of the design); placebo effects are averaged over the samples in which the pair is testable.
- **Segmentation QC retention** is the pooled ratio over all types; a type without own-identity transcripts before correction has no per-type value (it used to count as 0).
- **Aggregates**: a labelled sample without any cell of the class has 0 aggregates per mm² and area fraction 0 (it used to have no value and was silently dropped from the test); a test is refused when a sample assigned to a condition has no value (no labels, fewer than 3 cells, no tissue area).
- **Proximity DE**: the automatic density covariate is never the anchor type or the tested type; the anchor as covariate is refused. Samples without a condition are not counted in the result.
- **Segmentation-free abundance**: a cleared *Fill holes* field means 0 (no hole filled), as documented; it used to fill every interior hole.
- **A segmentation run that is not loaded** (still loading, or the load failed) is refused by every analysis instead of being computed on the Xenium cells under that run's name.
- **Cancel** during the last step of a job no longer publishes its result.
- **Cluster sets, joint clustering, Banksy**: the caches of Banksy and joint clustering are keyed by the content of the count matrix (old entries are recomputed once); a cached joint UMAP is re-attached by cell id and only to the samples it was computed on; whether a set's labels are shared across samples is recorded per install.
- **Transcript factors**: *Cells* counts every cell a factor dominates (a new column gives the labelled ones); share columns of cell types whose names differ only in punctuation no longer overwrite each other.
- **Image segmentation**: the Cellpose *nuclei* model gets the nuclear channel only (the membrane channel goes to the cyto-type models); for a whole-tissue run, "% of transcripts assigned" refers to the segmented extent (the image patches), not to the bounding box of the cells found.
- **Reference files**: no path is built in. `XENIUM_REF_RDS` / `XENIUM_IMPUTE_GENES` prefill the reference `.rds` and the gene list; with an empty path Run asks for it.
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

- **Leak gate**: the change rule (leak explains ≥ 50 % of the log fold change) is applied only to genes whose change is called or is at least 0.5 in log2; it used to fire on genes with a change of about zero. Fewer genes are flagged; the signal rule and the three formulas are unchanged.
- **Proximity and contact DE** honour the **Gene ownership + leak gate** checkbox (it was ignored: both always ran).
- **Proximity DE, cohort mode**: the sign-flip test enumerates all sign vectors up to 16 units and draws 20,000 above (it used 12 and 8,191); p-values of cohorts with 13 or more units change slightly, and those with 13 to 16 units are now exact.
- **Baysor/Proseg run directories** are written as `<name>.partial` and renamed when the run is complete. A failed or cancelled run leaves nothing in the cache (it used to leave the tool's partial output in the run directory, which the Segmentation dropdown could list as a finished run once `segmentation.csv` / `cell-metadata.csv.gz` existed).
- **The CellNEST database cache** (`cellnest_db/`) follows `XENIUM_CACHE_DIR`. It used to be created in `~/.xenium_explorer_cache/` regardless; with `XENIUM_CACHE_DIR` set, the database files are fetched once more into the new location (or move the directory there).
- **Baysor and Proseg run in their own process group**, Proseg on a pseudo-terminal (so that it draws the progress bar the panel reads). Their log lines in the server log are unchanged; Proseg's progress-bar redraws are not logged.
- `subset()` with only skipped filters is a no-op (it used to flag an "N / N cells" subset); **No niche** replaces `Niche nan` in legends; the co-occurrence and enrichment heat maps honour `width` / `height` when called with them.
