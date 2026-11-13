# Tests

A pytest suite for the app (`xenium_explorer.py` and the `xenium_app/` package behind
it), `xenium_viewer_api.py`, `functions/` and the WebGL viewer. It replaces the one-off check scripts that used to live in a scratch
folder.

```
pytest                      # default tier: fast, no external tools
pytest -m tools             # real Baysor / Proseg / R on a very small region
pytest -m browser           # node + Playwright tests of the WebGL viewer
pytest -m slow              # default tier again, on larger bundles; heavy variants (real UMAP fit, full-size LDA)
pytest --bundle /path/outs  # default tier on any other Xenium bundle
```

`pytest.ini` at the repository root selects the default tier
(`-m "not slow and not tools and not browser"`); a `-m` on the command line replaces
it. To run everything in one go: `pytest -m "slow or tools or browser or not slow"`.

Needs `pytest >= 7` (`pip install -r tests/requirements-test.txt`) in the same
environment as the app. Nothing else is installed or downloaded by the tests.

## Why one process per bundle

The app loads its dataset **while it is imported** (`import xenium_explorer` imports the
`xenium_app` package, which reads `DATA_DIR`) and
fixes its cache root (`XENIUM_CACHE_DIR`), REPL mode (`XENIUM_ENABLE_REPL`) and
default renderer (`XENIUM_RENDERER`) at the same moment. A Python process can
therefore hold exactly one dataset in one configuration. The suite is built around
that:

* **One pytest invocation = one bundle.** The session fixture `xe` (in `conftest.py`)
  sets the environment, imports the app once and every in-process test shares that
  import. The bundle is the tiny one in `tests/data/` unless `--bundle PATH` or
  `XENIUM_TEST_BUNDLE` names another. No test hard-codes anything about a dataset:
  expectations are computed from the bundle's raw files by `xe_testlib.RawBundle`, so
  the same tests run on any bundle.
* **Other import-time settings get their own process.** `xe_testlib.run_in_fresh_app`
  starts a new Python, imports the app there and returns a JSON result. It is used
  for the REPL modes, for reopening an exported `.zarr`, and for loading a bundle
  assembled by `geo_to_bundle.py`. The browser tier starts the real server the same
  way.
* **The slow tier is a loop over invocations.** `test_slow_bundles.py` runs
  `pytest --bundle <path>` as a child process for every bundle named by an
  environment variable.

The alternative (reloading the module per test) was rejected: the import takes
4-9 s even for 23 cells, registers Flask routes and replaces `sys.stdout`, and
cannot be undone cleanly.

### State isolation

All in-process tests share one imported app, whose state lives in module globals.
`conftest.py` records a baseline after import and an autouse fixture restores it after
every test: it waits for background threads the test started, calls `unsubset()`,
resets the job-state dicts and the job registry, clears ROIs / labels / niche columns
and the caches. A test
may therefore mutate the app freely, and tests can run in any order or alone (`-k`).

## The bundles are never written to

The app writes into its dataset directory (`spatialdata_xenium.zarr` is generated
next to the raw files on first load and updated later with ROI and SPLIT columns).
So the app is never pointed at a bundle itself:

* `bundle_dir` is a **working copy** in pytest's temporary directory: a real copy for
  small bundles, a directory of symlinks for large ones (the generated SpatialData
  store is copied, as an APFS clone where possible, or regenerated).
* `cache_dir` is a temporary directory; `xe` asserts that the app's `CACHE_BASE_DIR`
  is that directory, so nothing goes to `~/.xenium_explorer_cache`.
* The `source_bundle` fixture snapshots size and mtime of every file of the bundle at
  session start and **fails the session if anything changed**.

## Fixture data

`tests/data/prime_mouse_ileum_tiny/` (5.5 MB, 35 files) is the minimum subset of the
10x "tiny" Xenium Prime 5K mouse ileum example bundle (XOA 3.0, 23 cells, 5006 genes,
1985 transcripts) that the app and the viewer need: `experiment.xenium`,
`cells.parquet`, `cell_feature_matrix.h5`, `cell_boundaries.parquet`,
`nucleus_boundaries.parquet`, `transcripts.parquet`, `cells.zarr.zip`,
`transcripts.zarr.zip`, `analysis/` and `morphology_focus/`. The full bundle is 29 MB;
the rest (HTML summary, tarballs, duplicate CSV/zarr forms, `morphology.ome.tif`,
`gene_panel.json`) is not read by the app. Check the dataset's licence before
publishing the repository with it.

Larger bundles are optional and located by environment variable:

| variable | used by | what |
|---|---|---|
| `XENIUM_TEST_BUNDLE` | every tier (same as `--bundle`) | bundle for the in-process tests |
| `XENIUM_TEST_BUNDLE_SKIN` | `slow`, `browser` | Xenium Prime 5K human skin FFPE (~7 GB) |
| `XENIUM_TEST_BUNDLE_GEO` | `slow` | a bundle made by `functions/geo_to_bundle.py` |
| `XENIUM_TEST_BUNDLE_EXTRA` | `slow` | any further bundle |
| `BAYSOR_BIN`, `PROSEG_BIN` | `tools` | override `~/.julia/bin/baysor`, `~/.cargo/bin/proseg` |

Unset or missing paths are skipped, never an error.

## Layout

| file | tier | what it checks |
|---|---|---|
| `conftest.py` | | options, working copy, the `xe` import, state reset, shared fixtures |
| `xe_testlib.py` | | independent helpers: wire-format decoder, raw-bundle readers, recount helpers, figure decomposition, fresh-process runner, tool detection |
| `test_callback_audit.py` | default | AST audit of all Dash callbacks (argument count, return arity, duplicate outputs, unknown ids, clientside functions) plus the same questions asked of the registered app; layout ids unique; every Python source parses |
| `test_app_data.py` | default | gene-expression-only feature filter, cell/matrix alignment against the h5, a recount of the matrix from `transcripts.parquet`, clusters and UMAP against the analysis CSVs, boundaries against the parquet, cache paths under the cache root |
| `test_figures.py` | default | spatial and UMAP figure in every colour mode (values by cell id against the raw files), boundary overlay, colour-bar range, corrected counts, morphology overlay against the TIFF, placeholders |
| `test_viewer_store.py` | default | `xenium_viewer_api` alone: wire format, tile store against the parquet files, huge-bbox request |
| `test_viewer_endpoints.py` | default | every `/viewer/*` route via the Flask test client: meta, cells, colours (against raw files **and** cell-for-cell against the Plotly figure), boundaries (zip and in-memory paths), transcripts, image range requests / cache headers / stale tokens, path traversal, UMAP provider, a synthetic resegmentation source |
| `test_repl.py` | default | REPL off by default, loopback-only when enabled, remote only on explicit opt-in (through the real Dash endpoint, with spoofed forwarding headers), env parsing fails closed, server binds loopback |
| `test_subset.py` | default | each `subset()` filter against an independent mask; expression, figure and viewer stay aligned; `unsubset()` restores everything |
| `test_roi.py` | default | point-in-polygon and hull against shapely, save / combine / reload flow, labels survive subset, overlays |
| `test_spatial_analysis.py` | default | neighbour-type counts, co-occurrence and enrichment against a k-d-tree recount; niches against k-means on an independently built composition |
| `test_save_sdata.py` | default | Save as SpatialData round trip (table, obs columns, shapes), ROI-only export, export under a subset, reopening the export as the app's dataset |
| `test_pure_helpers.py` | default | `_patch_owned_mask`, `_patch_seam_dedup_mask`, `_reseg_patch_keep_mask`, transcript filters, expression builders, SPLIT gene re-alignment by name (app and `functions/run_split.py`) - all on synthetic input |
| `test_cache_and_jobs.py` | default | cached-run listing and cache cleaning on fabricated directories, annotation-cache validity, stale-job guard, dgCMatrix parts |
| `test_jobs.py` | default | job registry (thread, step, fraction, elapsed time, `_running_jobs`, Load Sample refusal), cancellation of a stub Python job at step boundaries and of a stub tool (`/bin/sh` with sleeping children): process tree gone, SIGKILL after the grace period, no run directory left, an earlier run kept, state reusable; Baysor / Proseg progress parsers; jobs-panel rendering and its interval; no job started outside the registry (AST) |
| `test_spage.py` | default | `_vectorized_spage` against the per-cell loop of the published algorithm (stand-in principal-vector class; the SpaGE submodule is not required) |
| `test_patches.py` | default | Prepare Patches through the callbacks (sopa): cache location, quality flag, ROI / patch geometry, restore only for the same request |
| `test_geo_to_bundle.py` | default | `functions/geo_to_bundle.py` on a synthetic deposit, error cases, and the app opening the result |
| `test_h5ad.py` | default | AnnData `.h5ad` input on fabricated files: the conversion into per-section stores against the object the file was written from, cell by cell (ids, coordinates, labels, counts; sections interleaved or in runs, read in small blocks; CSR, CSC and dense matrices); natural section order, global cluster ids, reuse of a finished conversion; the mapping from a sidecar JSON and the environment, and its refusals with the reason; one sample without a section column, cells without coordinates; a matrix that is not integer-valued is flagged, refused by pseudobulk and accepted only with `round_sums` (sums rounded); the app started from a file in its own process: one sample per section, design prefilled (conditions, donors, paired rule), cell-type columns as Color By / grouping options in both renderers, transcript- and image-based features answering with the reason, and the WebGL view as the start view above the Plotly cell limit |
| `test_analysis.py` | default | differential expression (U and p against `scipy.stats.mannwhitneyu`, z against `scanpy.tl.rank_genes_groups`, effect sizes and BH against dense recomputation, the sparse kernel on synthetic tied counts in many chunks and two matrices, overlap / minimum size / "all other cells", counts mode, a Proseg source, pooled samples, the tab's callbacks from lasso to CSV); gene-set scores (dense recomputation, control genes from the set's expression bins, `scanpy.tl.score_genes` with the same and with its own control genes, refusals, persistence, the colour in Plotly spatial + UMAP and both viewer endpoints, subset, corrected counts, two samples); transcript density (grid = `numpy.histogram2d` of the independently filtered transcripts at every bin size, zarr tiles = parquet, windows around cells, averaging of large windows, job / cache file / second run from cache, viewer endpoint and Plotly heat layer recounted, cancellation leaves no file) |
| `test_stats.py` | default | the sample → condition design and its editor; the pseudobulk matrix against dense sums, its refusals (fewer than 2 samples per condition, partial or confounded batch labels, non-integer counts) and dropped samples / filtered genes; **DESeq2 (with and without a batch term, apeglm shrinkage) and the edgeR fallback against the same matrix run by `Rscript`**; the composition test (Welch against scipy, the moderated t against `limma` run by `Rscript`, "no test below 3 samples per condition"); the density layer of added samples (each grid at its sample's offset, both renderers, per-sample cache); corrected counts with several samples (all or nothing, both renderers, the option and its note); Moran's I against the dense textbook formulas and against `spdep` run by `Rscript`, per sample, as a job, cancelled; ligand-receptor scores on fabricated tissue with a planted interaction (recovered; unspecific pair not significant; non-adjacent types not scored; p-value against a plain re-implementation; deterministic), pair-list parsing, the job and its cancel. The tests that need R skip when `Rscript` lacks the package (DESeq2, edgeR, limma, spdep) or rpy2 is missing: they load R inside the pytest process, like the `tools` tier does |
| `test_lrdecomp.py` | default | ligand–receptor exposure decomposition: the ported core sums against brute force for the tile, whole-sample and detection variants (the original's unit test) and the exact identity E = K·R·L·γ·π; sample sums, O/E (expected = 1 − (1 − p_L)^k) and pooling against a hand computation; a uniform field has γ ≈ π ≈ 1 and a planted niche moves γ (or π with one tile); a planted doubling of ligand rate, receptor rate or contact density is attributed to that term alone; paired (sign-flip, donor pooling) and unpaired (label permutation) designs, no p-values with one sample / one condition / < 3 units and the reason; floor, ownership and cell filters; the matrix tests equal `_sl_signflip` / `_sl_permutation`; omnibus; the tab as a job (guards, CSV, plot sums, cancel mid-run publishes nothing, rerun completes) |
| `test_controls.py` | default | exact sign-flip and label-permutation tests against itertools enumeration (never a sampled null at small n, Monte Carlo reproducible above the limit, donor averaging, BH reach warning); the donor field and the saved design; the composition test's exact option and donor-blocked pseudobulk; edge distance against the restated reference raster / hole-fill / distance-transform block, density and dense core against brute force; the compartment job through Color By in both renderers, subset alignment and the Save-as-SpatialData columns; markers, gene ownership; spillover on fabricated tissue with planted distance-dependent bleed (recovered, validity check passes) and with distance-independent signal (check fails), computed within samples; leak fraction / prediction / gate against the restated reference formulas, planted bleed flagged while a true change is kept, one-group / mixture / pseudobulk modes, sub-types not modelled; segmentation QC metrics against brute force, over-correction seen in retention; stratified co-localisation: counts and expected values against brute force, a shared niche called by the naive test and not by the stratified one, planted attraction found, one permutation by hand, placebo anchors, across-sample inference and the single-sample statement; guards: no neighbour count across samples (two samples at identical coordinates), an AST audit of every neighbour search in the package, Moran / LR graphs per sample, detection on fractional counts, depth note, pseudobulk volcano colouring, label-collapse report; the new tabs and job kinds; Cancel in the middle of each of the four jobs (compartments, spillover, segmentation QC, co-localisation: held inside its computation, cancelled, nothing published, the tab says Cancelled, a rerun completes) and during the QC job's transcript scan (no grid file left); the fraction of transcripts assigned against a hand count on the bundle's transcript file; an actual Save as SpatialData with compartment and spillover columns, read back with spatialdata and with the app's loader; the leak gate's change rule is restricted to genes whose change is called or at least 0.5 in log2 (a null gene with a change of about zero is not flagged by it, a called gene explained by bleed still is; the three formulas equal the reference on every gene) |
| `test_proximity.py` | default (+ one `tools` test with R) | proximity DE: set statistics, slope and free-length fit against dense least-squares residuals; sign-flip over genes against the scalar exact test; the empirical FDR against brute-force counting; count-matched random anchors within strata; a planted gradient found while the anchor's bleeding genes are flagged, quiet genes stay quiet; a location-only gradient removed by stratum-matched anchors; cohort mode (exact sign-flip, donors as units), unusable random sets (slope / BH fallback), refusals. Contact groups: matching weights by hand, pseudobulk = weighted sum, planted contact effect recovered with a stand-in fit and absent when expression is scrambled, label-shuffle null, refusal below 2 usable samples, donors as blocks; the two jobs through `run_de` (volcano, table, CSV), `n_clicks = 0` guards, Cancel mid-run (nothing published, rerun completes), no neighbour search across samples. Browser script for a real bundle: `viewer_src/test/proximity_checks.mjs` (see its header); the sign-flip limits are the shared ones (16 units / 20,000 vectors); the leak-gate checkbox reaches the proximity and contact computations |
| `test_aggregates.py` | default | aggregates on fabricated cells: the local class count against a brute-force count, within-stratum permutations keep every stratum's class count, the hotspot rule; a planted aggregate is found (members, centroid, area, enriched co-located type) and a uniform field yields none; Delaunay components need short links and enough class cells; tile area on a lattice, convex and alpha hulls; the composition expectation by hand; the job for one sample (single-sample statement, Color By value, Plotly shapes and WebGL polygons, other segmentation / subset / toggle, Save-as-SpatialData columns, CSV) and for six (exact sample-level test of the density, refusal below 3 units); two samples at identical coordinates do not see each other; Cancel in the middle of the job (nothing published, no columns, a rerun completes); n_clicks = 0 and input guards; the tab's Samples & conditions button opens the shared editor |
| `test_banksy.py` | default | Banksy and the cluster sets computed in the app. Features against the formulas of the paper: Gaussian neighbour weights with the median-neighbour width, brute-force mean and AGF of one cell (centred and as printed in equation 1), a constant gene has no gradient and a ramp a constant one, invariance under rotation / reflection / shift, neighbours never cross samples, the block weights √(1−λ), √(λ/1.5), √(λ/3); the chunked PCA equals the SVD of the explicit neighbour-augmented matrix; normalisation to the median total and the gene cap; exact neighbours and the Jaccard shared-neighbour graph by brute force, Leiden deterministic and ordered by size, the KMeans fallback named with the install hint (monkeypatched). **Planted domains** on fabricated cells (a weak programme in every cell of a quadrant, strong cell-type markers everywhere): λ = 0 finds the types and not the domains, λ = 0.8 the domains (ARI > 0.9), λ = 0.2 the types; two samples clustered together give the same label to the same domain. Cluster sets on the fixture bundle: install → column, dropdown, group labels (0 = no label), Plotly traces and the WebGL colours endpoint, cache files, restore after a "restart", refusal of labels of other cells, a subset, shared labels across two samples without the sample prefix, Save as SpatialData. The job: result, poll (cluster-set and redraw versions), CSV, second run from the cache, another λ, too few cells, `n_clicks = 0` and input guards, Cancel (nothing installed; a rerun completes) |
| `test_joint.py` | default | joint clustering across samples on fabricated samples with four planted cell types and a per-gene batch shift per sample: normalisation, per-gene moments and the dispersion score against dense formulas; gene selection prefers genes that vary within every sample over a gene that differs only between samples; scaling (cap at 10) and the sparse PCA against a dense `sklearn` PCA; cluster entropy and the neighbourhood mixing index by hand / brute force; levels of sample / donor / batch and the confounding rule; the Harmony step and the job's corrected embedding are **bit-identical to `harmonypy.run_harmony` on the same components** (skipped without `harmonypy`); with the planted shift the uncorrected clusters follow the samples and the Harmony clusters the planted types (ARI > 0.95); the job: one shared cluster set in every sample, raw labels identical across samples, diagnostics table / figure / CSV, joint UMAP columns and the UMAP panel showing all samples when the set is selected (skipped without `umap-learn`), second run from the cache and cache key changes with parameters or cells, no `harmonypy` (monkeypatched) → 'Joint clusters (uncorrected)' with the install hint, no integration → 'Joint clusters', donor levels from the design and the fully-confounded statement, one sample refused, cells without counts unlabelled, Cancel during the PCA (nothing published, installed or cached; a rerun completes), `n_clicks = 0` and input guards, poll bumps the cluster-set version |
| `test_segfree.py` | default | segmentation-free abundance on fabricated transcript files: Gini and top-tile share against hand values; the redistribution null (excess ≈ 0 for a uniform field, > 0 with p at the floor for a planted focus); tissue area = occupied tiles for every tile size, hole filling by the compartment rule; the one-pass multi-gene scan equals the per-gene density scan; transcripts per mm², cells per mm² and the share of all transcripts equal brute-force counts for every tile size (cells or all transcripts as the area); the job on the bundle (hand count, derived markers, subset, CSV, poll); several samples scored separately with the exact sample-level test and its refusal below 3 units; a `.zarr` sample without transcripts; cancel mid-scan (nothing published or cached, rerun completes); `n_clicks = 0` guards. `viewer_src/test/segfree_checks.mjs` drives the block in a browser on a real bundle (not part of a pytest tier; usage in its header). |
| `test_txfactor.py` | default | transcript factors on fabricated transcript files with planted spatial programs: square bins are floor division and a point belongs to the nearest hexagon centre (brute force over the neighbouring centres); the streamed bin × gene counts equal a brute-force count with the Q ≥ 20 / gene-feature / panel filters applied (several row groups, both shapes), the bundle's scan equals the transcripts of the cell-feature matrix per gene and the `transcripts.zarr.zip` tiles give the same matrix; the count cache (per bin size, shape and gene list; a broken file is rescanned); the projection keeps every bin's count, recovers a planted mixture and reaches no higher KL divergence than scikit-learn's NMF transform, with numba and with the numpy fallback, in chunks and threads; four planted programs are recovered up to permutation (cosine > 0.9; NMF on square and hexagonal bins, LDA) with the dominant factor of a bin equal to its stripe and the planted genes on top; entropy, cross-tab, purity and adjusted Rand index against hand values / scikit-learn; the job on the bundle's cells over fabricated transcripts (per-cell mixture recomputed from the bins under the seven points, both shapes; cross-tab, table, CSV, poll, exported columns); Color By values; the map layer through the density plumbing (square grid = the bins, hexagon raster = the hexagon under each pixel, categorical layer in the Plotly heat layer and with its palette in the WebGL endpoint, not averaged when coarsened); reference mode (components = hand-computed type profiles); the fit cache; several samples on their own grids with shared loadings and a `.zarr` sample left out; cancel mid-scan (nothing published or cached, rerun completes); guards and `n_clicks = 0`. `viewer_src/test/txfactor_checks.mjs` drives the tab in a browser on a real bundle in both renderers (not part of a pytest tier; usage in its header). |
| `test_imgseg.py` | default (+ one `tools` test with Cellpose) | image-based resegmentation: the classical backend on fabricated images (planted nuclei found, a touching pair split, nothing in noise, expansion without overlap, the scipy-only split); transcript assignment against an independent point-in-polygon count (Q ≥ 20, genes only, each transcript read once across tiles, overlapping outlines); Cellpose detection without any download (absent package, package without weights, only models whose weights file exists are offered; the Run button and the tab show the install note); a whole run on the bundle's morphology image through `run_imgseg_modal`: staging directory, `params.json`, cached-run entry and label, zarr, counts against a recount of the transcript file, reload from the cache, both renderers and the boundary checkbox, Segmentation QC source, view tables and label keys, a region run, Cancel between patches (nothing left behind), the other tabs ignore the image tab. `tools`: a real Cellpose run when cellpose is installed **with its weights already on disk** (skipped otherwise; asserts that the model directory is unchanged) |
| `test_tools_reseg.py` | `tools` | real Baysor and Proseg runs (single region; Proseg also on patches): each tool is first cancelled mid-run (processes gone, nothing left in the cache, state released), then the same request runs to completion; input equals an independent read of the region, matrix equals a recount of the tool's output, zarr / params / cache reload consistency |
| `test_tools_r.py` | `tools` | sparse matrices to R and back through rpy2, verified on the R side, also from a worker thread |
| `test_browser.py` | `browser` | wrappers around `viewer_src/test/{app_smoke,glue_checks,run}.mjs` (`viewer_src/test/analysis_checks.mjs`, which drives the three analysis tabs in both renderers on a real bundle, and `viewer_src/test/stats_checks.mjs`, which adds samples and drives the design editor, pseudobulk DE, composition, multi-sample density, Moran's I and ligand-receptor scores, are run by hand like `jobs_panel.mjs`) |
| `viewer_src/test/lrdecomp_checks.mjs` | by hand | not wrapped by pytest: drives the exposure-decomposition mode of the Ligand–Receptor tab in the real app (mode switch, empty state, upload of a pair list via `XL_PAIRS`, the descriptive single-sample run with plot / table / CSV; with `XL_SAMPLES` naming 5 further bundles also the design opened from the tab, the unpaired and paired tests, the type-pair level and the 2 v 2 no-p-value case); usage in the file header |
| `viewer_src/test/aggregates_checks.mjs` | by hand | not wrapped by pytest: drives the Aggregates tab in the real app in both renderers on a bundle with a cell-type annotation (run and refusal without a class, jobs panel, tables and plot, aggregate id under Color By, outline polygons in WebGL and layout shapes in Plotly, the outline toggle, a rerun with alpha hulls, the CSV download); usage at the top of the file |
| `viewer_src/test/joint_checks.mjs` | by hand | not wrapped by pytest: drives the Joint clustering tab in the real app in both renderers; `XJ_SAMPLES` names at least one further bundle that the suite adds through '+ Add sample' (refusal with one sample, the run and its diagnostics, the shared set under Cluster Set, WebGL categories without a per-sample prefix, the joint UMAP of all samples in the WebGL and the Plotly UMAP panel, Plotly map, CSV, Samples & conditions editor; usage in its header) |
| `test_package_layout.py` | default | the split into `xenium_app/`: every global is defined in one module, no module copies another's globals, every `module.NAME` and every global a function reads exists, library modules do not import the start-up modules; rebinding a constant, a state dict, a counter and a function through `xe` (assignment, `monkeypatch`, `mock.patch.object`) and through the REPL namespace reaches the code that uses them |
| `test_slow_bundles.py` | `slow` | the bundle-dependent default modules on each larger bundle |
| `test_review_round4.py` | default | one regression test per finding of the fourth review round (corrected-counts coverage, p-value floor, cancel at publish, pseudobulk donors, control columns, …) |
| `test_review_round6.py` | default | one regression test per finding of the sixth review round (counts mode of cluster labels on a segmentation run, run identity of a re-run, SpaGE cache hits, pseudobulk gene list, composition family, samples without labels, …). The tests of the Dash glue run `assets/xenium_viewer_dash.js` in node with a stand-in page (`glue_harness.js`; skipped without node) |
| `test_review_round7.py` | default | one regression test per finding of the seventh review round (batch steps on the run they name, the h5ad loader and its design prefill, what the provenance records and methods text state, navigation notes and pickers, size guards) |
| `test_provenance.py` | default | methods and provenance record. In the source: every job the registry starts belongs to a known analysis kind and its function calls `provenance._prov_record`, every record call names a known kind, every kind has a methods template, every CSV writer stamps the record id. The record on fabricated calls for each kind (required fields, plain JSON, unit copied from the status line, caveats verbatim, cache hit detection, subset), the per-dataset append-only file (bounded, unreadable lines skipped, survives a restart and Clean Cache - in a fresh process), real job runs on the tiny bundle (Moran, compartments, composition, spatial analysis, niches; a failed or stale job writes nothing), the Provenance tab callbacks, and the methods text: snapshots for four kinds (`tests/snapshots/methods_*.md`; `XENIUM_UPDATE_SNAPSHOTS=1` rewrites them) and what it must and must not claim |
| `test_batch.py` | default | headless batch mode: configs that cannot be used (message per case), `--dry-run` against the data (unknown parameter, gene, label source, cell type, pairs, design, samples, required parameter), the step table against the app (callbacks, controls, defaults, figures), a two-step run in process (Download CSV, Parquet tables, figure files, per-step and per-run provenance, methods text, summary), failure handling and exit codes, the example config, and one run of `python -m xenium_app.batch` in a new process whose tables equal those of the Run callbacks exactly (Moran's I, seeded transcript factors) |
| `test_zz_provenance_coverage.py` | default | runs last: the provenance records written by the jobs of all other modules during this run of the tier (kept in memory by the app) - required fields, JSON, a methods paragraph that renders - and the kinds the tier is known to run for real; skipped when only part of the tier ran |

## Tiers

**default** - no external binaries, no R, no network. 30-65 s on the tiny bundle
depending on machine load (most of it: the app import, two REPL processes, the export
round trips, sopa).

**tools** - `pytest -m tools`. Baysor and Proseg are run through `run_reseg_modal`,
exactly as the Run button does, on a 45 x 45 µm box around the densest group of
cells; about 3 min in total on the tiny bundle. Each test is skipped when its binary
(or R with the Matrix package) is missing. With `--bundle` pointing at a real slide
the same tests run on real tissue (several minutes per tool).

**browser** - `pytest -m browser`. Skipped without `node`, `viewer_src/node_modules`
(`npm install` in `viewer_src`) or a Chromium that Playwright can launch. The app is
started on a free port with `XENIUM_RENDERER=webgl`. `app_smoke.mjs`, `glue_checks.mjs`
and `run.mjs` use gene names, zoom levels and screenshot thresholds of the human skin
dataset, so they need `XENIUM_TEST_BUNDLE_SKIN`; without it only the check that the
app serves the WebGL configuration runs. `run.mjs` uses the fixed ports 8072 / 8073 and
writes screenshots to `viewer_src/test/screenshots/`. On macOS the scripts start
headless Chromium with `--use-angle=metal --ignore-gpu-blocklist` themselves
(`XV_GL=swiftshader` for software WebGL).

**slow** - `pytest -m slow` with one or more of the `XENIUM_TEST_BUNDLE_*` variables.
The GEO-style bundle (31 k cells, no boundaries / clustering / images) takes about a
minute and exercises all the "this bundle has no ..." paths; the 5K skin bundle
(112 k cells, multinucleate cells, a 4-channel image) takes several minutes and a few
GB of memory.

## Writing tests

* `xe` is one namespace over all modules of `xenium_app`: read any global as `xe.<name>`
  and rebind it with `monkeypatch.setattr(xe, "<name>", value)` or `xe.<name> = value`;
  the module that defines the name is rebound, so the app's own code sees the change.
  Tests that read the source use `tl.app_tree()` (all source files as one module),
  `tl.app_sources()` or `tl.app_source_of("function_name")`, not a single file.

* Assert a property, not the absence of an exception: recompute the expected value
  from the raw files (`raw` fixture) or with a second implementation, and compare by
  cell id.
* Do not hard-code cell counts, gene names or coordinates; skip with a reason when a
  bundle lacks what the test needs (`pytest.skip("bundle has no UMAP projection")`).
* Background jobs: start them through the callback, then
  `tl.wait_state(state, lock)`; wrap callbacks that spawn writer threads in
  `with tl.new_threads():`. Jobs are registered: `xe._jobs_list(active_only=True)`
  gives the running ones with their `.thread`, `.procs`, `.step` and `.fraction`,
  `xe._job_cancel(job.id)` cancels one. The state reset waits for every registered
  job and then empties the registry, so a test must let its stub jobs finish.
* Anything that needs another `DATA_DIR` or environment: `tl.run_in_fresh_app`.
* A test that exposes a bug stays in the suite as
  `@pytest.mark.xfail(strict=True, reason=...)` until the bug is fixed (strict, so the
  fix is noticed).

## Not covered

* Cell-type annotation (CellTypist, Seurat, RCTD), SPLIT and SpaGE **end to end**:
  they need a real single-cell reference and a tissue-sized dataset (15-25 min per run
  on the skin bundle), CellTypist needs a model download and the SpaGE submodule is not
  fetched. Their Python-side pieces are covered (matrix exchange with R, SPLIT gene
  re-alignment, the SpaGE core, label handling downstream of annotation).
* A Baysor run on patches (two more Julia start-ups); the patch merge is unit-tested
  on synthetic layouts and exercised for real with Proseg.
* CellNEST tabs, multi-sample display (Load / Add Sample) and the callbacks that need
  `dash.callback_context` (`update_plots`, `delete_roi`, `confirm_patches`,
  `cancel_job_clicked` - the Cancel button is clicked for real by
  `viewer_src/test/jobs_panel.mjs`, which is run by hand against a bundle with cached
  patches and is not part of the `browser` tier).
* The standalone scripts in `functions/` other than `geo_to_bundle.py` and the R
  bridge of `run_split.py`.
