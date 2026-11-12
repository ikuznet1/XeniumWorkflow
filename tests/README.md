# Tests

A pytest suite for the app (`xenium_explorer.py` and the `xenium_app/` package behind
it), `xenium_viewer_api.py`, `functions/` and the WebGL viewer. It replaces the one-off check scripts that used to live in a scratch
folder.

```
pytest                      # default tier: fast, no external tools
pytest -m tools             # real Baysor / Proseg / R on a very small region
pytest -m browser           # node + Playwright tests of the WebGL viewer
pytest -m slow              # default tier again, on larger bundles
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
| `test_analysis.py` | default | differential expression (U and p against `scipy.stats.mannwhitneyu`, z against `scanpy.tl.rank_genes_groups`, effect sizes and BH against dense recomputation, the sparse kernel on synthetic tied counts in many chunks and two matrices, overlap / minimum size / "all other cells", counts mode, a Proseg source, pooled samples, the tab's callbacks from lasso to CSV); gene-set scores (dense recomputation, control genes from the set's expression bins, `scanpy.tl.score_genes` with the same and with its own control genes, refusals, persistence, the colour in Plotly spatial + UMAP and both viewer endpoints, subset, corrected counts, two samples); transcript density (grid = `numpy.histogram2d` of the independently filtered transcripts at every bin size, zarr tiles = parquet, windows around cells, averaging of large windows, job / cache file / second run from cache, viewer endpoint and Plotly heat layer recounted, cancellation leaves no file) |
| `test_tools_reseg.py` | `tools` | real Baysor and Proseg runs (single region; Proseg also on patches): each tool is first cancelled mid-run (processes gone, nothing left in the cache, state released), then the same request runs to completion; input equals an independent read of the region, matrix equals a recount of the tool's output, zarr / params / cache reload consistency |
| `test_tools_r.py` | `tools` | sparse matrices to R and back through rpy2, verified on the R side, also from a worker thread |
| `test_browser.py` | `browser` | wrappers around `viewer_src/test/{app_smoke,glue_checks,run}.mjs` (`viewer_src/test/analysis_checks.mjs`, which drives the three analysis tabs in both renderers on a real bundle, is run by hand like `jobs_panel.mjs`) |
| `test_package_layout.py` | default | the split into `xenium_app/`: every global is defined in one module, no module copies another's globals, every `module.NAME` and every global a function reads exists, library modules do not import the start-up modules; rebinding a constant, a state dict, a counter and a function through `xe` (assignment, `monkeypatch`, `mock.patch.object`) and through the REPL namespace reaches the code that uses them |
| `test_slow_bundles.py` | `slow` | the bundle-dependent default modules on each larger bundle |

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
