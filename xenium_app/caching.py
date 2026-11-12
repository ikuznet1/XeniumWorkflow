"""Cache locations: dataset keys, segmentation run directories, cache size and clean-up."""

import os
import re
import json
import hashlib
import pandas as pd

from . import core


def _dataset_key() -> str:
    """Identity of the current dataset for cache tags: the Xenium output directory,
    or the .zarr path when the app was started from an exported SpatialData store
    (data_dir is None there)."""
    return core.DATA.get("data_dir") or core.DATA.get("sdata_path") or ""


def _dataset_name() -> str:
    """Basename used in Baysor/Proseg cache directory names ('' for .zarr input)."""
    return os.path.basename((core.DATA.get("data_dir") or "").rstrip("/"))


def _dataset_hash_forms() -> set:
    """The 8-char md5 tags the current dataset may be recorded under. params.json
    'dataset_hash' hashes data_dir as it was passed at launch, so the relative,
    absolute and trailing-slash spellings of the same directory all count."""
    key = _dataset_key()
    if not key:
        return set()
    forms = {key, key.rstrip("/"), os.path.abspath(key)}
    try:
        forms.add(os.path.relpath(os.path.abspath(key)))
    except ValueError:
        pass
    forms |= {f + "/" for f in list(forms)}
    return {hashlib.md5(f.encode()).hexdigest()[:8] for f in forms}


def _seg_run_tag(entry: str, tool: str):
    """Parameter tag of cache entry *entry* when it is a Baysor/Proseg run directory
    of the current dataset, else None. Names are '<tool>_<dataset>_<dataset hash>_<tag>'
    or, for older runs, '<tool>_<dataset>_<tag>'; the latter are told apart from a
    same-named dataset in another folder by the 'dataset_hash' in params.json
    (accepted when that file or field is missing)."""
    dataset = _dataset_name()
    prefix = f"{tool}_{dataset}_"
    if not dataset or not entry.startswith(prefix):
        return None
    m = re.fullmatch(r"(?:([0-9a-f]{8})_)?([0-9a-f]{8})", entry[len(prefix):])
    if not m:
        return None   # e.g. a run of dataset '<dataset>_2'
    hashes = _dataset_hash_forms()
    if m.group(1):
        return m.group(2) if m.group(1) in hashes else None
    try:
        with open(os.path.join(core.CACHE_BASE_DIR, entry, "params.json")) as f:
            stored = json.load(f).get("dataset_hash")
    except Exception:
        stored = None
    if stored and stored not in hashes:
        return None
    return m.group(2)


def _seg_run_dir(tool: str, param_tag: str) -> str:
    """Cache directory of the current dataset's Baysor/Proseg run with *param_tag*:
    the existing directory when there is one (so older '<tool>_<dataset>_<tag>'
    runs are reused), otherwise a new name that includes the dataset hash so two
    datasets with the same folder name never share a run directory."""
    if os.path.isdir(core.CACHE_BASE_DIR):
        for entry in sorted(os.listdir(core.CACHE_BASE_DIR)):
            if (entry.endswith(param_tag) and os.path.isdir(os.path.join(core.CACHE_BASE_DIR, entry))
                    and _seg_run_tag(entry, tool) == param_tag):
                return os.path.join(core.CACHE_BASE_DIR, entry)
    ds_hash = hashlib.md5(_dataset_key().encode()).hexdigest()[:8]
    return os.path.join(core.CACHE_BASE_DIR, f"{tool}_{_dataset_name()}_{ds_hash}_{param_tag}")


# A Baysor/Proseg run works in '<run dir>.partial' and the directory gets its real
# name only when the run is complete (_seg_stage_commit). A run that fails, is
# cancelled or dies with the app therefore never leaves a run directory that the
# cached-run listing (_seg_run_tag does not match the suffix) or a later run with the
# same parameters could take for a finished one, and a cancelled re-run leaves the
# earlier result of the same parameters in place.
_SEG_STAGE_SUFFIX = ".partial"


def _seg_stage_dir(final_dir: str) -> str:
    """Create and return the empty working directory of a run that will be stored
    in *final_dir* (a stale one left by a killed process is removed first)."""
    import shutil
    stage = final_dir.rstrip("/\\") + _SEG_STAGE_SUFFIX
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    _seg_stage_active.add(stage)
    return stage


def _seg_stage_commit(stage: str, final_dir: str) -> None:
    """Give a finished run its real directory name, replacing an earlier run with
    the same parameters."""
    import shutil
    if os.path.isdir(final_dir):
        shutil.rmtree(final_dir)
    os.rename(stage, final_dir)
    _seg_stage_active.discard(stage)


_seg_stage_active: set = set()    # working directories of the runs in progress


def _seg_stage_discard(stage) -> None:
    """Remove the working directory of a run that did not finish (no-op after commit)."""
    _seg_stage_active.discard(stage)
    if stage and stage.endswith(_SEG_STAGE_SUFFIX) and os.path.isdir(stage):
        import shutil
        shutil.rmtree(stage, ignore_errors=True)
        print(f"  Removed unfinished run directory {os.path.basename(stage)}", flush=True)


def _cache_path(model_name: str) -> str:
    """Return path for the cached annotation parquet file."""
    safe = model_name.replace(".pkl", "").replace("/", "_")
    cache_dir = core.CACHE_BASE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    # Include a hash of the data dir so caches from different datasets don't clash
    tag = hashlib.md5(_dataset_key().encode()).hexdigest()[:8]
    return os.path.join(cache_dir, f"{safe}_{tag}.parquet")


def _ref_file_tag(path: str) -> str:
    """Short identity of a reference file for cache keys: hash of its absolute path,
    size and modification time. A reference that was edited and saved under the same
    name, or another file with the same basename elsewhere, gets a different tag, so
    its cached labels are not served for it. (Caches written before this tag existed
    are simply not found once.)"""
    try:
        _p = os.path.abspath(os.path.expanduser(path))
        _st = os.stat(_p)
        _ident = f"{_p}|{_st.st_size}|{_st.st_mtime_ns}"
    except OSError:
        _ident = str(path)
    return "f" + hashlib.md5(_ident.encode()).hexdigest()[:8]


def _annot_cache_valid(cached, cell_ids_override, cache_sig, allow_subset=False) -> bool:
    """Is a cached annotation parquet usable for the given resegmented cells?
    Xenium caches (cell_ids_override None) are always accepted. Reseg caches carry
    a '_sig' column (hash of the cell ids + matrix shape/nnz) which must match;
    files written before that column existed must cover exactly the same ids."""
    if cell_ids_override is None:
        return True
    if "_sig" in cached.columns:
        return len(cached) > 0 and cache_sig is not None and str(cached["_sig"].iloc[0]) == str(cache_sig)
    _ids = pd.Index([str(c) for c in cell_ids_override])
    _idx = pd.Index(cached.index.astype(str))
    if not _idx.isin(_ids).all():
        return False
    return len(_idx) <= len(_ids) if allow_subset else len(_idx) == len(_ids)


def _annot_cache_frame(labels, cache_sig):
    """DataFrame written to the annotation cache (adds '_sig' for reseg runs)."""
    _df = pd.DataFrame({"label": labels})
    if cache_sig is not None:
        _df["_sig"] = str(cache_sig)
    return _df


# ─── Cache utilities ───────────────────────────────────────────────────────────

def _cache_dir() -> str:
    return core.CACHE_BASE_DIR


_cache_size_memo: dict = {"t": 0.0, "dir": None, "val": None}


def _cache_size_str(max_age: float = 300.0) -> str:
    """Return human-readable total size of the cache directory.
    Walking a large cache tree is slow, so the value is reused for max_age
    seconds; pass max_age=0 for an exact figure."""
    import time
    cache = _cache_dir()
    if (max_age > 0 and _cache_size_memo["val"] is not None
            and _cache_size_memo["dir"] == cache
            and time.time() - _cache_size_memo["t"] < max_age):
        return _cache_size_memo["val"]
    val = _cache_size_walk(cache)
    _cache_size_memo.update({"t": time.time(), "dir": cache, "val": val})
    return val


def _cache_size_walk(cache: str) -> str:
    if not os.path.isdir(cache):
        return "0 B"
    total = 0
    for root, dirs, files in os.walk(cache):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if total < 1024 or unit == "TB":
            return f"{total:.1f} {unit}" if unit != "B" else f"{total} B"
        total /= 1024
    return f"{total:.1f} TB"


def _format_seg_label(tool: str, params: dict) -> str:
    """Format a human-readable label for a cached segmentation run."""
    if tool == "baysor":
        parts = [f"r={params.get('scale', '?')}µm", f"min={params.get('min_mol', '?')}tx"]
        if params.get("use_prior"):
            parts.append(f"prior={params.get('prior_conf', '?')}")
        x_min, x_max = params.get("x_min"), params.get("x_max")
        y_min, y_max = params.get("y_min"), params.get("y_max")
        if any(v is not None for v in [x_min, x_max, y_min, y_max]):
            parts.append(f"x={x_min}-{x_max} y={y_min}-{y_max}µm")
        else:
            parts.append("full slide")
        if params.get("n_patches"):
            parts.append(f"{params['n_patches']} patches")
        n = params.get("n_cells")
        if n:
            parts.append(f"{n:,} cells")
        return "Baysor: " + ", ".join(parts)
    else:
        parts = []
        vs = params.get("voxel_size")
        parts.append(f"voxel={'auto' if not vs else str(vs)+'µm'}")
        x_min, x_max = params.get("x_min"), params.get("x_max")
        y_min, y_max = params.get("y_min"), params.get("y_max")
        if any(v is not None for v in [x_min, x_max, y_min, y_max]):
            parts.append(f"x={x_min}-{x_max} y={y_min}-{y_max}µm")
        else:
            parts.append("full slide")
        if params.get("n_patches"):
            parts.append(f"{params['n_patches']} patches")
        n = params.get("n_cells")
        if n:
            parts.append(f"{n:,} cells")
        return "Proseg: " + ", ".join(parts)


def _list_cached_seg_runs() -> list:
    """Return list of {label, value} dicts for all complete cached Baysor/Proseg runs for the current dataset."""
    cache_base = core.CACHE_BASE_DIR
    dataset = _dataset_name()
    opts = []
    if not os.path.isdir(cache_base) or not dataset:
        # no dataset name (.zarr input): the prefix "baysor__" could match unrelated dirs
        return opts
    for entry in sorted(os.listdir(cache_base)):
        out_dir = os.path.join(cache_base, entry)
        if not os.path.isdir(out_dir):
            continue
        # Determine tool (exact name match plus dataset identity — see _seg_run_tag)
        if entry.startswith(f"baysor_{dataset}_"):
            tool = "baysor"
            param_tag = _seg_run_tag(entry, tool)
            if param_tag is None:
                continue
            # Check completeness: any segmentation.csv anywhere in the tree
            complete = any(
                "segmentation.csv" in files
                for _, _, files in os.walk(out_dir)
            )
        elif entry.startswith(f"proseg_{dataset}_"):
            tool = "proseg"
            param_tag = _seg_run_tag(entry, tool)
            if param_tag is None:
                continue
            complete = any(
                "cell-metadata.csv.gz" in files
                for _, _, files in os.walk(out_dir)
            )
        else:
            continue
        if not complete:
            continue
        params_path = os.path.join(out_dir, "params.json")
        # Patch-based runs write params.json only after all patches are merged, so
        # patch_* dirs without it mean an in-progress or aborted run.
        if not os.path.exists(params_path) and any(
                d.startswith("patch_") and os.path.isdir(os.path.join(out_dir, d))
                for d in os.listdir(out_dir)):
            continue
        label = f"{tool.capitalize()}: [{param_tag}]"
        if os.path.exists(params_path):
            try:
                with open(params_path) as f:
                    label = _format_seg_label(tool, json.load(f))
            except Exception:
                pass   # truncated/corrupt params.json: keep the tag label
        if any(o["value"] == f"{tool}:{param_tag}" for o in opts):
            continue   # same run under both the older and the hashed directory name
        opts.append({"label": label, "value": f"{tool}:{param_tag}", "out_dir": out_dir, "param_tag": param_tag})
    return opts


def _clean_cache_for_dataset() -> str:
    """Delete all cache entries that belong to the current dataset. Returns summary string."""
    cache = _cache_dir()
    if not os.path.isdir(cache):
        return "Cache directory not found."
    tag = hashlib.md5(_dataset_key().encode()).hexdigest()[:8]
    dataset_name = _dataset_name()
    removed = 0
    import shutil
    for entry in os.listdir(cache):
        full = os.path.join(cache, entry)
        # Match files/dirs that contain the dataset tag, and the Baysor/Proseg run
        # directories of this dataset (exact name + dataset identity, not a substring:
        # 'run1' must not match 'baysor_run10_…')
        # (a '.partial' working directory left by a killed run counts as its run directory)
        _run_entry = (entry[:-len(_SEG_STAGE_SUFFIX)] if entry.endswith(_SEG_STAGE_SUFFIX)
                      else entry)
        if tag in entry or (os.path.isdir(full) and (
                _seg_run_tag(_run_entry, "baysor") or _seg_run_tag(_run_entry, "proseg"))):
            try:
                if os.path.isdir(full):
                    shutil.rmtree(full)
                else:
                    os.remove(full)
                removed += 1
            except Exception as e:
                print(f"  clean_cache: could not remove {entry}: {e}", flush=True)
    print(f"  clean_cache: removed {removed} cache entries for dataset {dataset_name}", flush=True)
    return f"Removed {removed} cache entries for {dataset_name}."
