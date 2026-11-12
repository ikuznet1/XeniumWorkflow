"""SpatialData / sopa backend: the cache store, Save as SpatialData, tissue ROI and patches."""

import os
import json
import hashlib
import pandas as pd

from . import compat, core, jobs, caching, annotation


# ─── SpatialData / Sopa backend ───────────────────────────────────────────────

def _sdata_cache_path() -> str:
    """Return the directory path for the SpatialData Zarr cache for this dataset."""
    tag = hashlib.md5(caching._dataset_key().encode()).hexdigest()[:12]
    cache_dir = core.CACHE_BASE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"spatialdata_{tag}.zarr")


def _sdata_cache_valid(cache: str) -> bool:
    """True if the SpatialData cache was written completely (an interrupted write
    leaves a directory that would otherwise load with elements or rows missing)."""
    if not os.path.isdir(cache):
        return False
    if os.path.exists(os.path.join(cache, ".complete")):
        return True
    # Caches written before the marker existed: consolidated metadata is written last
    if os.path.exists(os.path.join(cache, ".zmetadata")):
        return True
    try:
        with open(os.path.join(cache, "zarr.json")) as _f:
            return json.load(_f).get("consolidated_metadata") is not None
    except Exception:
        return False


def _sdata_fix_categories(sdata) -> None:
    """Eagerly compute feature_name categories to avoid dask performance issues."""
    try:
        for pts_key in sdata.points:
            pts = sdata.points[pts_key]
            if "feature_name" in pts.columns:
                col = pts["feature_name"]
                if hasattr(col, "cat") and not col.cat.known:
                    sdata.points[pts_key]["feature_name"] = col.cat.as_known()
    except Exception as exc:
        print(f"  to_spatialdata: category fix skipped: {exc}", flush=True)


def _sdata_summary(sdata) -> str:
    n_tx = 0
    try:
        for k in sdata.points:
            n_tx += len(sdata.points[k])
    except Exception:
        pass
    images = list(sdata.images.keys())
    return f"{n_tx:,} transcripts | images: {', '.join(images) or 'none'}"


# functions(cells_df, seg_source) -> DataFrame (aligned to cells_df) or None: per-cell
# columns of other modules to write with Save as SpatialData (the controls module registers)
_extra_obs_providers: list = []


def _with_extra_obs(cells_df, seg_source):
    """(cells_df with the registered extra columns joined - a copy when there are any -,
    their names)."""
    names = []
    if cells_df is None:
        return cells_df, names
    for fn in list(_extra_obs_providers):
        try:
            extra = fn(cells_df, seg_source)
        except Exception as exc:
            print(f"  save_spatialdata: extra columns skipped ({exc})", flush=True)
            continue
        if extra is None or not len(extra.columns):
            continue
        if not names:
            cells_df = cells_df.copy()
        for c in extra.columns:
            cells_df[c] = extra[c].to_numpy()
            names.append(c)
    return cells_df, names


def _save_sdata_to_disk(save_path: str, seg_source: str = "xenium", roi_only: bool = False) -> None:
    """Background thread: write current SpatialData + analysis results to disk."""
    _gen = jobs._dataset_gen

    def _set(status, msg):
        with core._save_sdata_lock:
            core._save_sdata_state["status"]  = status
            core._save_sdata_state["message"] = msg
        if status == "running":
            jobs._job_progress(msg)

    try:
        import spatialdata as sd
        tool = core._seg_tool(seg_source)

        # ── Load sdata + cells_df from the active segmentation ─────────────────
        if tool in ("baysor", "proseg"):
            _lock = core._baysor_lock if tool == "baysor" else core._proseg_lock
            _state = core._baysor_state if tool == "baysor" else core._proseg_state
            with _lock:
                _alt_res = _state.get("result")
            if not _alt_res:
                _set("error", f"No {tool.capitalize()} result loaded. Select and load the run first.")
                return
            sdata_path = _alt_res.get("sdata_path", "")
            if not sdata_path or not os.path.isdir(sdata_path):
                _set("error", f"No SpatialData found for {tool}. Run resegmentation first.")
                return
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sdata = sd.read_zarr(sdata_path)
            cells_df  = _alt_res["cells_df"]
            cluster_methods = [c for c in cells_df.columns
                               if c.startswith("cluster") or c == "niche" or c.startswith("roi_")]
        else:
            # The app's own per-cell store (table + cell / nucleus boundaries), read
            # fresh; the sopa object of the Resegment modal has no table and no cells.
            _alt_res   = None
            sdata_path = core.DATA.get("sdata_path", "")
            if not sdata_path or not os.path.isdir(sdata_path):
                _set("error", "No SpatialData store found for this dataset.")
                return
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sdata = sd.read_zarr(sdata_path)
            cells_df        = core.DATA.get("df")
            # The store already holds the columns DATA["df"] was loaded from; only
            # columns added in this session are attached below.
            cluster_methods = [] if cells_df is None else [
                c for c in cells_df.columns
                if c.startswith("cluster") or c == "niche" or c.startswith("roi_")]

        cells_df, _extra_cols = _with_extra_obs(cells_df, seg_source)
        # text columns go through the label path below, numeric ones through the numeric path
        cluster_methods = list(cluster_methods) + [c for c in _extra_cols if c not in cluster_methods
                                                   and cells_df[c].dtype == object]

        _set("running", f"Writing to {save_path}…")
        print(f"  save_spatialdata: writing to {save_path} (seg={seg_source}, roi_only={roi_only})…", flush=True)

        # ── Determine ROI cell-ID filter ───────────────────────────────────────
        roi_ids = None
        if roi_only and cells_df is not None:
            with core._roi_lock:
                rois = list(core._roi_state["rois"])
            if rois:
                roi_cols = [c for c in cells_df.columns if c.startswith("roi_")]
                if roi_cols:
                    mask    = cells_df[roi_cols].notna().any(axis=1)
                    roi_ids = set(cells_df[mask].index.astype(str))
                    print(f"  save_spatialdata: ROI filter — {len(roi_ids):,} of {len(cells_df):,} cells", flush=True)
                else:
                    print("  save_spatialdata: roi_only requested but no ROI columns found — exporting all cells", flush=True)

        # ── Attach analysis results to sdata table ─────────────────────────────
        out_tbl = None
        if "table" in sdata.tables:
            tbl = sdata.tables["table"]
            has_cell_id   = "cell_id" in tbl.obs.columns
            zarr_cell_ids = tbl.obs["cell_id"].astype(str) if has_cell_id else None

            def _map(series: "pd.Series") -> list:
                """Align a Series (index=cell_id str) to zarr obs rows."""
                series = series.dropna().astype(str)
                series.index = series.index.astype(str)
                if has_cell_id:
                    return zarr_cell_ids.map(series.to_dict()).fillna("Unknown").tolist()
                return series.reindex(tbl.obs.index.astype(str)).fillna("Unknown").tolist()

            def _align(series: "pd.Series") -> "pd.Series":
                series = series.copy()
                series.index = series.index.astype(str)
                series = series[~series.index.duplicated(keep="first")]
                _keys = zarr_cell_ids.values if has_cell_id else tbl.obs.index.astype(str)
                return series.reindex(_keys)

            def _map_na(series: "pd.Series") -> list:
                """As _map, but a cell without a value stays without one ("" in the
                store, read back as missing) instead of joining a group "Unknown"."""
                _al = _align(series.dropna().astype(str))
                return _al.where(_al.notna(), "").tolist()

            if cells_df is not None:
                # Cluster / niche / ROI columns: cells SPLIT dropped, cells outside
                # every ROI and cells outside the subset have no value
                for method in cluster_methods:
                    if method in cells_df.columns:
                        tbl.obs[method] = _map_na(cells_df[method])
                # Embeddings computed in this session (Rerun UMAP, SPLIT UMAP) and any
                # other numeric per-cell column the re-read table does not carry
                for _ncol in cells_df.columns:
                    _always = _ncol in ("umap_1", "umap_2", "split_umap_1", "split_umap_2")
                    if not _always and (_ncol in tbl.obs.columns or _ncol in cluster_methods
                                        or str(_ncol).startswith("_")):
                        continue
                    if not pd.api.types.is_numeric_dtype(cells_df[_ncol]) \
                            or pd.api.types.is_bool_dtype(cells_df[_ncol]):
                        continue
                    if _always and not cells_df[_ncol].notna().any():
                        continue
                    tbl.obs[_ncol] = pd.to_numeric(_align(cells_df[_ncol]),
                                                   errors="coerce").to_numpy(dtype="float64")

            # Cell-type annotations: per-method labels of the exported segmentation
            # (Xenium: labels_<method>; reseg: labels_<method>_<tool>_<run tag>)
            with core._annot_lock:
                _first_lbl = None
                for m in core._ANNOT_METHODS:
                    lbl = core._annot_state.get(annotation._labels_key_for_method(m, _alt_res))
                    if lbl is not None:
                        tbl.obs[f"cell_type_{m}"] = _map(lbl)
                        if _first_lbl is None:
                            _first_lbl = lbl
                if _alt_res is not None and _first_lbl is not None:
                    # Reseg: single combined label (first available method)
                    tbl.obs["cell_type"] = _map(_first_lbl)
                # RCTD weights → obsm
                rctd_w = core._annot_state.get(
                    f"rctd_weights_{annotation._labels_key_for_method('rctd', _alt_res)}")
                if rctd_w is not None:
                    rctd_w = rctd_w.copy()
                    rctd_w.index = rctd_w.index.astype(str)
                    if has_cell_id:
                        aligned = rctd_w.reindex(zarr_cell_ids.values).fillna(0.0)
                    else:
                        aligned = rctd_w.reindex(tbl.obs.index.astype(str)).fillna(0.0)
                    tbl.obsm["rctd_weights"] = aligned.values
                    tbl.uns["rctd_weight_columns"] = list(rctd_w.columns)

            # ROI filter: subset table rows
            if roi_ids is not None:
                if has_cell_id:
                    keep_mask = zarr_cell_ids.isin(roi_ids).values
                else:
                    keep_mask = tbl.obs.index.astype(str).isin(roi_ids)
                tbl = tbl[keep_mask].copy()
                print(f"  save_spatialdata: table filtered to {tbl.n_obs:,} cells", flush=True)
            out_tbl = tbl

        # ── Filter shapes to ROI cells ─────────────────────────────────────────
        out_shapes = {}
        for name, gdf in sdata.shapes.items():
            if roi_ids is not None:
                idx_str = gdf.index.astype(str)
                keep    = idx_str.isin(roi_ids)
                if keep.any():
                    out_shapes[name] = gdf[keep]
            else:
                out_shapes[name] = gdf

        # ── Write output (table + shapes only; images stay in original Xenium dir) ──
        sdata_out = sd.SpatialData(
            shapes=out_shapes,
            tables={"table": out_tbl} if out_tbl is not None else {},
        )
        jobs._check_dataset_gen(_gen)   # reported as an error: the export would mix two datasets
        sdata_out.write(save_path, overwrite=True)
        n_cells = out_tbl.n_obs if out_tbl is not None else 0
        print(f"  save_spatialdata: done → {save_path} ({n_cells:,} cells)", flush=True)
        _set("done", f"Saved to {save_path}")
    except jobs._JobCancelled:
        raise
    except Exception as exc:
        import traceback; traceback.print_exc()
        _set("error", str(exc)[:300])


def _sdata_autoload() -> None:
    """If a Zarr cache exists for this dataset, load it into _sdata_state at startup."""
    cache = _sdata_cache_path()
    if not _sdata_cache_valid(cache):
        return
    try:
        import spatialdata
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sdata = spatialdata.read_zarr(cache)
        # No _sdata_fix_categories here: it scans the whole transcript table and
        # is applied in the background before patching instead.
        summary = _sdata_summary(sdata)
        with core._sdata_lock:
            core._sdata_state["status"]  = "done"
            core._sdata_state["message"] = f"Auto-loaded (cached) — {summary}"
            core._sdata_state["sdata"]   = sdata
            core._sdata_state["qv"]      = None    # flag as stored; set by to_spatialdata()
        core.DATA["sdata"] = sdata
        print(f"  SpatialData: auto-loaded from cache — {summary}", flush=True)
        # Load any cell-type annotations stored in zarr tbl.obs into _annot_state
        _annot_load_from_sdata(sdata)
    except Exception as exc:
        print(f"  SpatialData: auto-load failed: {exc}", flush=True)


def _annot_load_from_sdata(sdata) -> None:
    """Load cell-type annotation columns from a SpatialData table into _annot_state."""
    try:
        if "table" not in sdata.tables:
            return
        tbl = sdata.tables["table"]
        obs = tbl.obs
        loaded = []
        # If obs has cell_id column, re-index labels by cell_id (integer Xenium IDs)
        # so they match DATA["df"].index when looked up in _cell_type_traces
        id_index = obs["cell_id"].astype(str) if "cell_id" in obs.columns else None
        for m in core._ANNOT_METHODS:
            col = f"cell_type_{m}"
            if col in obs.columns:
                labels = obs[col].astype(str).copy()
                if id_index is not None:
                    labels.index = id_index
                else:
                    labels.index = labels.index.astype(str)
                # Only load if parquet cache didn't already populate this key
                with core._annot_lock:
                    if core._annot_state.get(f"labels_{m}") is None:
                        core._annot_state[f"labels_{m}"] = labels
                        core._annot_state["status"]  = "done"
                        core._annot_state["message"] = f"Auto-loaded ({core._ANNOT_METHODS[m]}) from zarr"
                        loaded.append(m)
        if loaded:
            print(f"  Annotation: loaded from zarr — {', '.join(loaded)}", flush=True)
        # Also load RCTD weights if present
        if "rctd_weights" in tbl.obsm and "rctd_weight_columns" in tbl.uns:
            with core._annot_lock:
                if core._annot_state.get("rctd_weights_labels_rctd") is None:
                    cols     = list(tbl.uns["rctd_weight_columns"])
                    w_index  = id_index if id_index is not None else obs.index.astype(str)
                    w        = pd.DataFrame(tbl.obsm["rctd_weights"],
                                            index=w_index, columns=cols)
                    core._annot_state["rctd_weights_labels_rctd"] = w
                    print(f"  Annotation: loaded RCTD weights from zarr ({len(cols)} types)", flush=True)
    except Exception as exc:
        print(f"  Annotation: zarr load failed: {exc}", flush=True)


def _available_samples() -> list:
    """Scan CWD and parent for output-XETG* directories containing experiment.xenium."""
    candidates = []
    search_dirs = [os.getcwd(), os.path.dirname(os.path.abspath(compat.APP_FILE))]
    seen = set()
    for base in search_dirs:
        try:
            entries = sorted(os.listdir(base))
        except Exception:
            continue
        for entry in entries:
            full = os.path.join(base, entry)
            if (entry.startswith("output-") and os.path.isdir(full)
                    and os.path.exists(os.path.join(full, "experiment.xenium"))
                    and full not in seen):
                seen.add(full)
                candidates.append({"label": entry, "value": full})
    return candidates


def to_spatialdata(qv_threshold: int = 20, force: bool = False) -> None:
    """Convert Xenium data to SpatialData format using sopa.io.xenium().

    The result is cached to ~/.xenium_explorer_cache/spatialdata_<hash>.zarr
    and auto-loaded on subsequent startups.

    Callable from REPL:
        to_spatialdata()                  # use cache if available
        to_spatialdata(force=True)        # re-read from raw files and overwrite cache
        to_spatialdata(qv_threshold=30)
    """
    _gen = jobs._dataset_gen

    def _run():

        def _set(status, message, sdata=None):
            with core._sdata_lock:
                if jobs._stale_job(_gen, status, message):
                    return
                core._sdata_state["status"]  = status
                core._sdata_state["message"] = message
                if sdata is not None:
                    core._sdata_state["sdata"] = sdata
                    # threshold behind this object's low_quality_transcript flag
                    # (decides which transcripts count towards a patch)
                    core._sdata_state["qv"] = qv_threshold
                    core.DATA["sdata"] = sdata

        try:
            if not core.DATA.get("data_dir"):
                _set("error", "No raw Xenium data for a .zarr dataset — "
                              "SpatialData patch setup is not available.")
                return
            _set("running", "Importing sopa…")
            jobs._job_progress(fraction=0.0)
            try:
                import sopa, sopa.io, spatialdata
            except ImportError as e:
                _set("error", f"Missing package: {e}. Run: pip install sopa")
                return

            cache = _sdata_cache_path()

            # ── Try cache first ───────────────────────────────────────────
            if not force and _sdata_cache_valid(cache):
                _set("running", "Loading from cache…")
                print(f"  to_spatialdata: loading from cache {cache} …", flush=True)
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        sdata = spatialdata.read_zarr(cache)
                    # The cached flag was computed with the threshold used when the cache
                    # was built; re-derive it (lazily) for the requested threshold.
                    for _pk in list(sdata.points.keys()):
                        _pts = sdata.points[_pk]
                        if "qv" in _pts.columns and "low_quality_transcript" in _pts.columns:
                            sdata.points[_pk]["low_quality_transcript"] = _pts["qv"] < qv_threshold
                    _sdata_fix_categories(sdata)
                    summary = _sdata_summary(sdata)
                    msg = f"Done (cached) — {summary}"
                    print(f"  to_spatialdata: {msg}", flush=True)
                    core._sdata_version += 1
                    _set("done", msg, sdata=sdata)
                    return
                except (KeyError, Exception) as cache_err:
                    print(f"  to_spatialdata: cache read failed ({cache_err}), "
                          f"rebuilding from raw files…", flush=True)

            # ── Read from raw Xenium files ────────────────────────────────
            _set("running", f"Reading Xenium data (qv≥{qv_threshold})…")
            print(f"  to_spatialdata: reading from {core.DATA['data_dir']} …", flush=True)
            import warnings, logging
            # Suppress spatialdata logging warnings (uses Python logging, not warnings module)
            _sd_loggers = [logging.getLogger(n) for n in ("spatialdata", "spatialdata._logging")]
            _sd_prev_levels = [lg.level for lg in _sd_loggers]
            for lg in _sd_loggers:
                lg.setLevel(logging.ERROR)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*feature_key.*categorical.*")
                    warnings.filterwarnings("ignore", message=".*OME series.*")
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    sdata = sopa.io.xenium(core.DATA["data_dir"], qv_threshold=qv_threshold)
            finally:
                for lg, lvl in zip(_sd_loggers, _sd_prev_levels):
                    lg.setLevel(lvl)

            _sdata_fix_categories(sdata)

            # ── Save to Zarr cache ────────────────────────────────────────
            _set("running", "Saving to cache…")
            print(f"  to_spatialdata: saving to {cache} …", flush=True)
            try:
                import shutil
                if os.path.isdir(cache):
                    shutil.rmtree(cache)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    sdata.write(cache)
                open(os.path.join(cache, ".complete"), "w").close()
                print(f"  to_spatialdata: cache saved ✓", flush=True)
            except Exception as save_exc:
                print(f"  to_spatialdata: cache save failed: {save_exc}", flush=True)
                # do not leave a partial store that a later start would load as complete
                shutil.rmtree(cache, ignore_errors=True)

            summary = _sdata_summary(sdata)
            msg = f"Done — {summary}"
            print(f"  to_spatialdata: {msg}", flush=True)
            core._sdata_version += 1
            _set("done", msg, sdata=sdata)

        except jobs._JobCancelled:
            raise
        except Exception as exc:
            import traceback
            traceback.print_exc()
            _set("error", str(exc)[:300])

    jobs._job_start("patches", "Prepare patches 1/3: SpatialData", _run,
               state=core._sdata_state, lock=core._sdata_lock)
    print("  to_spatialdata: started in background…", flush=True)


def clear_sdata_cache() -> None:
    """Delete the SpatialData Zarr cache for this dataset (callable from REPL)."""
    import shutil
    cache = _sdata_cache_path()
    if os.path.isdir(cache):
        shutil.rmtree(cache)
        print(f"  SpatialData cache deleted: {cache}", flush=True)
    else:
        print(f"  SpatialData cache not found: {cache}", flush=True)


# ── Re-use of the tissue ROI and the patches saved in the SpatialData cache store ──
# sopa writes both into the store (shapes/…), but nothing in the store says which
# parameters produced the patches. The app therefore records them next to the shapes
# (_SDATA_PATCH_PARAMS_FILE) and restores the patches only when that record matches
# the request and still describes the shapes on disk.
_SDATA_PATCH_PARAMS_FILE = ".xe_patch_params.json"
_SDATA_PATCH_OVERLAP_UM  = 50.0    # sopa.make_transcript_patches default (not set by the app)


def _sdata_roi_key(sdata):
    return next((k for k in sdata.shapes
                 if k == "region_of_interest" or "roi" in k.lower() or "tissue" in k.lower()),
                None)


def _sdata_patch_key(sdata):
    return next((k for k in sdata.shapes
                 if k == "transcripts_patches" or "patch" in k.lower()), None)


def _sdata_disk_shapes_hash(sdata, key):
    """md5 of the geometries of shapes element *key* as stored on disk; None when the
    object is not backed by a store or the element is not in it."""
    try:
        if key is None or not sdata.is_backed():
            return None
        _p = os.path.join(str(sdata.path), "shapes", key, "shapes.parquet")
        if not os.path.exists(_p):
            return None
        import geopandas as _gpd
        _h = hashlib.md5()
        for _w in _gpd.read_parquet(_p).geometry.to_wkb():
            _h.update(_w)
        return _h.hexdigest()
    except Exception as _he:
        print(f"  SpatialData: could not hash shapes '{key}': {_he}", flush=True)
        return None


def _sdata_patch_request(sdata, width_um, min_transcripts, qv) -> dict:
    """Everything the patch layout depends on: grid width and overlap, the minimum
    transcript count and the Q-score threshold deciding which transcripts count,
    the tissue ROI the grid is clipped to, and the sopa version."""
    try:
        import sopa as _sopa
        _sopa_ver = str(_sopa.__version__)
    except Exception:
        _sopa_ver = None
    return {"version": 1,
            "patch_width": float(width_um),
            "patch_overlap": _SDATA_PATCH_OVERLAP_UM,
            "min_transcripts": int(min_transcripts),
            "qv_threshold": None if qv is None else float(qv),
            "roi_hash": _sdata_disk_shapes_hash(sdata, _sdata_roi_key(sdata)),
            "sopa": _sopa_ver}


def _sdata_restore_patches(sdata, width_um, min_transcripts, qv):
    """The patches stored in the cache store, if they are known to be the result of this
    request; else None (the caller then recomputes them)."""
    try:
        _key = _sdata_patch_key(sdata)
        if _key is None or not sdata.is_backed():
            return None
        _pfile = os.path.join(str(sdata.path), _SDATA_PATCH_PARAMS_FILE)
        if not os.path.exists(_pfile):
            print("  create_patches: stored patches have no parameter record — recomputing",
                  flush=True)
            return None
        with open(_pfile) as _f:
            _saved = json.load(_f)
        _want = _sdata_patch_request(sdata, width_um, min_transcripts, qv)
        _diff = [k for k, v in _want.items() if _saved.get(k) != v]
        if _diff:
            print("  create_patches: stored patches were made with other settings ("
                  + ", ".join(f"{k}: {_saved.get(k)!r} → {_want[k]!r}" for k in _diff)
                  + ") — recomputing", flush=True)
            return None
        _patches = sdata.shapes[_key]
        if (_saved.get("patches_key") != _key or _saved.get("n_patches") != len(_patches)
                or not _saved.get("patches_hash")
                or _saved.get("patches_hash") != _sdata_disk_shapes_hash(sdata, _key)):
            print("  create_patches: stored patches differ from their parameter record "
                  "— recomputing", flush=True)
            return None
        return _patches
    except Exception as _re:
        print(f"  create_patches: could not restore stored patches ({_re}) — recomputing",
              flush=True)
        return None


def _sdata_record_patches(sdata, width_um, min_transcripts, qv) -> None:
    """Write the parameter record for the patches just computed (if they reached the store)."""
    try:
        _key = _sdata_patch_key(sdata)
        _hash = _sdata_disk_shapes_hash(sdata, _key)
        if _hash is None:
            return
        _rec = _sdata_patch_request(sdata, width_um, min_transcripts, qv)
        _rec.update({"patches_key": _key, "n_patches": int(len(sdata.shapes[_key])),
                     "patches_hash": _hash})
        _pfile = os.path.join(str(sdata.path), _SDATA_PATCH_PARAMS_FILE)
        with open(_pfile + ".tmp", "w") as _f:
            json.dump(_rec, _f)
        os.replace(_pfile + ".tmp", _pfile)
    except Exception as _we:
        print(f"  create_patches: could not record patch parameters: {_we}", flush=True)


def segment_tissue_roi() -> None:
    """Run sopa.segmentation.tissue() to detect tissue ROI in the SpatialData object.
    Callable from REPL. Requires to_spatialdata() to have completed first.
    A ROI already saved in the cache store is re-used (it depends on the image only;
    Re-build clears the store and with it the ROI).
    """

    _gen = jobs._dataset_gen

    def _run():
        with core._sdata_lock:
            sdata = core._sdata_state.get("sdata")
        if sdata is None:
            print("  segment_tissue_roi: no SpatialData. Run to_spatialdata() first.", flush=True)
            with core._sdata_lock:
                core._sdata_state["status"]  = "error"
                core._sdata_state["message"] = "No SpatialData loaded - run Prepare Patches first"
            return
        try:
            import sopa.segmentation
            with core._sdata_lock:
                jobs._check_dataset_gen(_gen)
                core._sdata_state["status"]  = "running"
                core._sdata_state["message"] = "Segmenting tissue ROI…"
            jobs._job_progress("Segmenting tissue ROI…", 1 / 3)
            _cached_roi = (_sdata_roi_key(sdata) is not None
                           and _sdata_disk_shapes_hash(sdata, _sdata_roi_key(sdata)) is not None)
            if _cached_roi:
                print("  segment_tissue_roi: using the ROI saved in the cache store", flush=True)
            else:
                print("  segment_tissue_roi: running sopa.segmentation.tissue()…", flush=True)
                sopa.segmentation.tissue(sdata, mode="staining")
            # Find ROI shapes key (default key_added='region_of_interest')
            roi_key = _sdata_roi_key(sdata)
            if roi_key:
                roi_shapes = sdata.shapes[roi_key]
                with core._sdata_lock:
                    jobs._check_dataset_gen(_gen)
                    core._sdata_state["roi"]     = roi_shapes
                    core._sdata_state["status"]  = "roi_done"
                    core._sdata_state["message"] = (f"ROI done — {len(roi_shapes)} region(s)"
                                               + (" (cached)" if _cached_roi else ""))
                core._sdata_version += 1
                print(f"  segment_tissue_roi: done, {len(roi_shapes)} region(s)", flush=True)
            else:
                print("  segment_tissue_roi: no ROI shapes found in sdata", flush=True)
                with core._sdata_lock:
                    jobs._check_dataset_gen(_gen)
                    # status must leave "running" or the poll never re-enables the buttons
                    core._sdata_state["status"]  = "error"
                    core._sdata_state["message"] = "No tissue ROI found - use Skip, or Re-build and retry"
        except jobs._JobCancelled:
            raise
        except Exception as exc:
            import traceback
            traceback.print_exc()
            with core._sdata_lock:
                if not jobs._stale_job(_gen, "error"):
                    core._sdata_state["status"]  = "error"
                    core._sdata_state["message"] = str(exc)[:300]

    jobs._job_start("patches", "Prepare patches 2/3: tissue ROI", _run,
               state=core._sdata_state, lock=core._sdata_lock)


def create_patches(width_um: float = 1000.0, min_transcripts: int = 10) -> None:
    """Create transcript patches using sopa.make_transcript_patches().
    Callable from REPL. Requires to_spatialdata() to have completed first.
    """

    _gen = jobs._dataset_gen

    def _run():
        with core._sdata_lock:
            sdata = core._sdata_state.get("sdata")
        if sdata is None:
            print("  create_patches: no SpatialData. Run to_spatialdata() first.", flush=True)
            with core._sdata_lock:
                core._sdata_state["status"]  = "error"
                core._sdata_state["message"] = "No SpatialData loaded - run Prepare Patches first"
            return
        try:
            import sopa
            with core._sdata_lock:
                jobs._check_dataset_gen(_gen)
                core._sdata_state["status"]  = "running"
                core._sdata_state["message"] = "Creating transcript patches…"
                _qv = core._sdata_state.get("qv")
            jobs._job_progress("Creating transcript patches…", 2 / 3)
            print(f"  create_patches: width={width_um}µm, min_tx={min_transcripts}…", flush=True)
            # Patches saved in the cache store by an earlier session with the same
            # settings: use them instead of repeating sopa's pass over every transcript
            _stored = _sdata_restore_patches(sdata, width_um, min_transcripts, _qv)
            if _stored is not None:
                with core._sdata_lock:
                    jobs._check_dataset_gen(_gen)
                    core._sdata_state["patches"] = _stored
                    core._sdata_state["status"]  = "patches_done"
                    core._sdata_state["message"] = f"Patches done — {len(_stored)} patches (cached)"
                core._sdata_version += 1
                print(f"  create_patches: {len(_stored)} patches restored from the cache store",
                      flush=True)
                return
            try:
                if sdata.is_backed():   # the record must never describe older patches
                    os.remove(os.path.join(str(sdata.path), _SDATA_PATCH_PARAMS_FILE))
            except OSError:
                pass
            _sdata_fix_categories(sdata)   # no-op when categories are already known
            sopa.make_transcript_patches(
                sdata,
                patch_width=width_um,
                min_points_per_patch=min_transcripts,
            )
            # sopa also dumps every in-patch transcript to CSV in its cache dir; only
            # the patch polygons are used here (Baysor/Proseg re-read transcripts.parquet)
            try:
                import shutil as _sh_p
                from sopa.utils import get_cache_dir as _sopa_cache_dir
                _sh_p.rmtree(os.path.join(str(_sopa_cache_dir(sdata)), "transcript_patches"),
                             ignore_errors=True)
            except Exception:
                pass
            # Find patches key (default key_added='transcripts_patches')
            patch_key = _sdata_patch_key(sdata)
            if patch_key:
                patches = sdata.shapes[patch_key]
                _sdata_record_patches(sdata, width_um, min_transcripts, _qv)
                with core._sdata_lock:
                    jobs._check_dataset_gen(_gen)
                    core._sdata_state["patches"] = patches
                    core._sdata_state["status"]  = "patches_done"
                    core._sdata_state["message"] = f"Patches done — {len(patches)} patches"
                core._sdata_version += 1
                print(f"  create_patches: {len(patches)} patches created", flush=True)
            else:
                print("  create_patches: no patch shapes found in sdata", flush=True)
                with core._sdata_lock:
                    jobs._check_dataset_gen(_gen)
                    core._sdata_state["status"]  = "error"
                    core._sdata_state["message"] = "No patch shapes produced"
        except jobs._JobCancelled:
            raise
        except Exception as exc:
            import traceback
            traceback.print_exc()
            with core._sdata_lock:
                if not jobs._stale_job(_gen, "error"):
                    core._sdata_state["status"]  = "error"
                    core._sdata_state["message"] = str(exc)[:300]

    jobs._job_start("patches", "Prepare patches 3/3: transcript patches", _run,
               state=core._sdata_state, lock=core._sdata_lock)


def _sdata_transform_to_um(gdf):
    """Transform a SpatialData GeoDataFrame from its intrinsic coordinate system
    to µm by applying the registered transform to the global coordinate system
    (morphology image pixels) and then scaling by PIXEL_SIZE_UM.

    Returns a new GeoDataFrame with coordinates in µm.
    """
    try:
        import shapely.affinity
        import geopandas as gpd
        from spatialdata.transformations import get_transformation
        t = get_transformation(gdf, to_coordinate_system="global")
        # Get the 3×3 affine matrix [x, y, 1] (row=output, col=input)
        M = t.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        # M transforms intrinsic coords → global pixel coords
        # Compose with PIXEL_SIZE_UM to go directly to µm
        sx = M[0, 0] * core.PIXEL_SIZE_UM
        sy = M[1, 1] * core.PIXEL_SIZE_UM
        tx = M[0, 2] * core.PIXEL_SIZE_UM
        ty = M[1, 2] * core.PIXEL_SIZE_UM
        geoms_um = [
            shapely.affinity.affine_transform(g, [sx, M[0, 1] * core.PIXEL_SIZE_UM,
                                                   M[1, 0] * core.PIXEL_SIZE_UM, sy,
                                                   tx, ty])
            for g in gdf.geometry
        ]
        return gpd.GeoDataFrame(geometry=geoms_um)
    except Exception as exc:
        print(f"  _sdata_transform_to_um failed: {exc}, using raw coordinates", flush=True)
        return gdf


def _sdata_overlays_to_shapes(show_roi: bool, show_patches: bool) -> list:
    """Extract ROI polygon and/or patch grid as Plotly layout shapes (in µm)."""
    shapes = []
    with core._sdata_lock:
        roi     = core._sdata_state.get("roi")
        patches = core._sdata_state.get("patches")

    if show_roi and roi is not None:
        try:
            roi_um = _sdata_transform_to_um(roi)
            for geom in roi_um.geometry:
                coords = list(geom.exterior.coords)
                xs = [c[0] for c in coords]
                ys = [-c[1] for c in coords]  # negate Y for plot space
                shapes.append(dict(
                    type="path",
                    path=" ".join(
                        f"{'M' if i == 0 else 'L'}{x},{y}"
                        for i, (x, y) in enumerate(zip(xs, ys))
                    ) + " Z",
                    line=dict(color="#52b788", width=2),
                    fillcolor="rgba(82,183,136,0.05)",
                    layer="above",
                ))
        except Exception as exc:
            print(f"  ROI overlay error: {exc}", flush=True)

    if show_patches and patches is not None:
        try:
            patches_um = _sdata_transform_to_um(patches)
            for geom in patches_um.geometry:
                b = geom.bounds  # (minx, miny, maxx, maxy) in µm
                shapes.append(dict(
                    type="rect",
                    x0=b[0], y0=-b[3], x1=b[2], y1=-b[1],  # negate Y
                    line=dict(color="rgba(255,170,0,0.6)", width=1),
                    fillcolor="rgba(255,170,0,0.03)",
                    layer="above",
                ))
        except Exception as exc:
            print(f"  Patch overlay error: {exc}", flush=True)

    return shapes
