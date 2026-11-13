"""AnnData .h5ad input: a processed cell × gene object holding one or several sections.

The file is converted ONCE into one SpatialData store per section under the cache
directory (``h5ad_<name>_<hash>/<section>.zarr`` plus ``manifest.json``); the app then
loads those stores like any store written by "Save as SpatialData", the first section as
the primary dataset and the others as additional samples. Nothing of the app's analysis
code knows about .h5ad: every sample is an ordinary dataset dict with its own cell table
and its own count matrix (its rows of the file only, never a copy of the whole matrix).

The conversion never reads the matrix as a whole: obs is read, then the chosen matrix
(``X`` or a layer) is read in row blocks straight from the HDF5 file and each block is
split by section; a section is written and released as soon as its last row has passed,
so with sections stored one after the other (the usual case) the peak is one section
plus one block. A file has no transcripts and no images: features that need them stay
unavailable (see _h5ad_unavailable)."""

import os
import re
import json
import time
import hashlib
import threading
import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py

from . import core, jobs, loading, dataview, stats


_H5AD_FORMAT = 1                    # of the converted cache (part of its directory hash)
_H5AD_BLOCK_NNZ = 24_000_000        # stored values read per block of rows (~0.3 GB at float64)
_H5AD_SAMPLE_GAP_UM = 500.0         # gap between two samples in the side-by-side view
_H5AD_MAX_LEVELS = 60               # levels listed per obs column by _h5ad_inspect
_H5AD_ENV = {"x": "XENIUM_H5AD_X", "y": "XENIUM_H5AD_Y", "section": "XENIUM_H5AD_SECTION",
             "cell_types": "XENIUM_H5AD_CELLTYPES", "condition": "XENIUM_H5AD_CONDITION",
             "donor": "XENIUM_H5AD_DONOR", "counts": "XENIUM_H5AD_COUNTS",
             "clusters": "XENIUM_H5AD_CLUSTERS", "umap": "XENIUM_H5AD_UMAP",
             "cond_a": "XENIUM_H5AD_COND_A", "cond_b": "XENIUM_H5AD_COND_B",
             "noninteger_counts": "XENIUM_H5AD_NONINTEGER"}
# mapping keys that do not change the converted stores (applied when they are loaded)
_H5AD_LOAD_KEYS = ("cond_a", "cond_b", "noninteger_counts")
_H5AD_LIST_KEYS = ("cell_types", "clusters")
_H5AD_METHOD_RE = re.compile(r"^obs\d+$")     # annotation "methods" that are obs columns of a file

# the file behind the loaded samples (None: the view does not come from an .h5ad)
_h5ad_loaded: dict = {"manifest": None, "cache_dir": None}
# conversion + loading started from the Load Sample dialog
_h5ad_state: dict = {"status": "idle", "message": "", "result": None}
_h5ad_lock = threading.Lock()

_H5AD_UNAVAILABLE = (
    ("Transcript density", "needs the transcript file of a Xenium output directory"),
    ("Segmentation-free abundance", "needs the transcript file"),
    ("Transcript factors", "needs the transcript file"),
    ("Baysor / Proseg resegmentation", "needs the transcript file"),
    ("Image-based resegmentation", "needs the morphology images and the transcript file"),
    ("Morphology image overlay", "the file holds no images"),
    ("Cell / nucleus boundaries", "the file holds no polygons; cells are drawn as points"),
    ("Fraction of transcripts assigned (segmentation QC)", "needs the transcript file"),
)


def _is_h5ad_path(path) -> bool:
    return str(path or "").rstrip("/").lower().endswith(".h5ad")


def _h5ad_unavailable() -> list:
    """[(feature, reason)] of what cannot be used on samples that come from an .h5ad."""
    return list(_H5AD_UNAVAILABLE)


def _h5ad_unavailable_note() -> str:
    return ("Not available for data loaded from an .h5ad (cells only, no transcripts, no images): "
            + "; ".join(f"{f} ({r})" for f, r in _H5AD_UNAVAILABLE) + ".")


# ─── reading the file ─────────────────────────────────────────────────────────
def _h5ad_read_elem(node):
    try:
        from anndata.io import read_elem
    except ImportError:                                  # anndata < 0.11
        from anndata.experimental import read_elem
    return read_elem(node)


def _h5ad_attr(node, name) -> str:
    v = node.attrs.get(name, "")
    return v.decode() if isinstance(v, bytes) else str(v)


def _h5ad_matrix_node(f, counts):
    """The HDF5 node of the matrix a mapping names: "X" or a layer ("counts", "layers/counts")."""
    counts = str(counts or "X")
    if counts == "X":
        if "X" not in f:
            raise ValueError("The file has no X matrix.")
        return f["X"]
    name = counts.split("/", 1)[1] if counts.startswith("layers/") else counts
    if "layers" not in f or name not in f["layers"]:
        have = sorted(f["layers"].keys()) if "layers" in f else []
        raise ValueError(f"The file has no layer {name!r} (layers: {', '.join(have) or 'none'}; or use X).")
    return f["layers"][name]


def _h5ad_matrix_info(node) -> dict:
    """{encoding ("csr" / "csc" / "dense"), shape, nnz, dtype} of a matrix node."""
    if isinstance(node, h5py.Dataset):
        return {"encoding": "dense", "shape": tuple(int(v) for v in node.shape),
                "nnz": int(np.prod(node.shape)), "dtype": str(node.dtype)}
    enc = _h5ad_attr(node, "encoding-type") or _h5ad_attr(node, "h5sparse_format")
    enc = "csc" if "csc" in enc else "csr"
    shape = tuple(int(v) for v in (node.attrs["shape"] if "shape" in node.attrs else node.attrs["h5sparse_shape"]))
    return {"encoding": enc, "shape": shape, "nnz": int(node["data"].shape[0]),
            "dtype": str(node["data"].dtype)}


def _h5ad_inspect(path) -> dict:
    """What a mapping can be chosen from, read without touching the matrices: shape, the
    obs columns ({name, kind: "numeric" / "categorical", n_levels, levels}), layers
    (with their on-disk encoding and size), obsm keys, and a guessed mapping."""
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(path):
        raise ValueError(f"File not found: {path}")
    with h5py.File(path, "r") as f:
        if "obs" not in f or "var" not in f:
            raise ValueError("Not an AnnData .h5ad file (no obs / var).")
        obs = _h5ad_read_elem(f["obs"])
        n_var = int(len(_h5ad_read_elem(f["var"]).index))
        cols = []
        for c in obs.columns:
            s = obs[c]
            if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
                lv = pd.unique(s.dropna())
                cols.append({"name": str(c), "kind": "categorical", "n_levels": int(len(lv)),
                             "levels": [_h5ad_level_str(v) for v in lv[:_H5AD_MAX_LEVELS]]})
            else:
                nun = int(s.nunique(dropna=True)) if len(s) else 0
                cols.append({"name": str(c), "kind": "numeric", "n_levels": nun,
                             # as the conversion writes them ("0", not "0.0"): the values
                             # the dialog offers are the levels of the manifest
                             "levels": ([_h5ad_level_str(v) for v in sorted(pd.unique(s.dropna()))]
                                        if nun <= _H5AD_MAX_LEVELS else [])})
        mats = {}
        if "X" in f:
            mats["X"] = _h5ad_matrix_info(f["X"])
        for name in (f["layers"].keys() if "layers" in f else []):
            mats[str(name)] = _h5ad_matrix_info(f["layers"][name])
        obsm = {str(k): [int(v) for v in f["obsm"][k].shape] for k in (f["obsm"].keys() if "obsm" in f else [])
                if isinstance(f["obsm"][k], h5py.Dataset)}
    info = {"path": path, "n_cells": int(len(obs)), "n_genes": n_var, "obs_columns": cols,
            "matrices": mats, "obsm": obsm, "size_bytes": int(os.path.getsize(path))}
    info["guess"] = _h5ad_guess_mapping(info)
    return info


def _h5ad_guess_mapping(info: dict) -> dict:
    """A starting mapping from common column names; anything not recognised stays empty."""
    names = [c["name"] for c in info["obs_columns"]]
    kind = {c["name"]: c for c in info["obs_columns"]}
    low = {n.lower(): n for n in names}

    def pick(*cands):
        return next((low[c] for c in cands if c in low), None)

    x = pick("x", "x_centroid", "x_um", "center_x", "x_location")
    y = pick("y", "y_centroid", "y_um", "center_y", "y_location")
    out = {"x": x if x and y else None, "y": y if x and y else None}
    if out["x"] is None and "spatial" in info["obsm"]:
        out["spatial"] = "spatial"
    out["section"] = pick("section", "sample", "sample_id", "library_id", "fov", "slide", "region")
    out["cell_types"] = [n for n in names if kind[n]["kind"] == "categorical"
                         and re.search(r"cell[_ ]?type|celltype|annotation", n, re.I)][:4]
    out["condition"] = pick("condition", "group", "disease", "treatment", "status")
    out["donor"] = pick("donor", "donor_id", "patient", "patient_id", "individual", "subject")
    out["counts"] = "counts" if "counts" in info["matrices"] else "X"
    out["umap"] = "X_umap" if info["obsm"].get("X_umap", [0, 0])[-1] >= 2 else None
    out["clusters"] = []
    return out


def _h5ad_mapping_for(path, info=None) -> dict:
    """The mapping used when a file is named on the command line: the guess, overridden by
    a JSON file (named by XENIUM_H5AD_MAPPING, else ``<file>.json`` beside the .h5ad), then
    by the XENIUM_H5AD_* environment variables. Keys: x, y (obs columns, µm; or "spatial":
    an obsm key), section, cell_types (list), condition, donor, counts ("X" or a layer),
    clusters (list), umap (obsm key), cond_a / cond_b (the two condition levels compared)."""
    info = info or _h5ad_inspect(path)
    mapping = dict(info["guess"])
    side = os.environ.get("XENIUM_H5AD_MAPPING", "").strip() or str(info["path"]) + ".json"
    if os.path.isfile(side):
        with open(side) as fh:
            given = json.load(fh)
        if not isinstance(given, dict):
            raise ValueError(f"{side}: the column mapping must be a JSON object.")
        mapping.update(given)
        if ("x" in given or "y" in given) and "spatial" not in given:
            mapping.pop("spatial", None)
        if given.get("spatial") and "x" not in given and "y" not in given:
            # the mapping names the obsm key: guessed x / y columns do not override it
            mapping["x"] = mapping["y"] = None
        print(f"  h5ad: column mapping read from {side}", flush=True)
    elif os.environ.get("XENIUM_H5AD_MAPPING", "").strip():
        raise ValueError(f"XENIUM_H5AD_MAPPING names a file that does not exist: {side}")
    for key, var in _H5AD_ENV.items():
        if var in os.environ:
            v = os.environ[var].strip()
            mapping[key] = ([p.strip() for p in v.split(",") if p.strip()]
                            if key in _H5AD_LIST_KEYS else (v or None))
            if key in ("x", "y"):
                mapping.pop("spatial", None)
    return _h5ad_clean_mapping(mapping)


def _h5ad_clean_mapping(mapping: dict) -> dict:
    """Canonical form: every key present, lists as lists, empty strings as None."""
    m = dict(mapping or {})
    out = {}
    for k in ("x", "y", "spatial", "section", "condition", "donor", "umap", "cond_a", "cond_b"):
        v = m.get(k)
        out[k] = (str(v) if v not in (None, "") else None)
    for k in _H5AD_LIST_KEYS:
        v = m.get(k) or []
        v = [v] if isinstance(v, str) else list(v)
        out[k] = list(dict.fromkeys(str(c) for c in v if str(c)))
    c = str(m.get("counts") or "X")
    out["counts"] = c.split("/", 1)[1] if c.startswith("layers/") else c
    # what the sample-level count models do with a matrix that is not integer-valued:
    # "refuse" (default) or "round_sums" (sum per sample, round the sums; an explicit choice
    # for files that only hold corrected counts)
    ni = str(m.get("noninteger_counts") or "refuse").strip().lower()
    if ni not in ("refuse", "round_sums"):
        raise ValueError(f"noninteger_counts must be \"refuse\" or \"round_sums\", not {ni!r}.")
    out["noninteger_counts"] = ni
    return out


def _h5ad_check_mapping(info: dict, mapping: dict) -> None:
    """ValueError naming what a mapping asks for that the file does not have."""
    names = {c["name"]: c for c in info["obs_columns"]}
    have = ", ".join(names) or "none"
    if mapping.get("x") or mapping.get("y"):
        for k in ("x", "y"):
            c = mapping.get(k)
            if not c or c not in names:
                raise ValueError(f"{k} coordinate: obs has no column {c!r} (columns: {have}).")
            if names[c]["kind"] != "numeric":
                raise ValueError(f"{k} coordinate: obs column {c!r} is not numeric.")
    elif mapping.get("spatial"):
        if info["obsm"].get(mapping["spatial"], [0, 0])[-1] < 2:
            raise ValueError(f"obsm has no 2-column array {mapping['spatial']!r} "
                             f"(obsm: {', '.join(info['obsm']) or 'none'}).")
    else:
        raise ValueError("No coordinates: name the obs columns holding x and y in µm "
                         f"(columns: {have}) or an obsm key.")
    for k in ("section", "condition", "donor"):
        if mapping.get(k) and mapping[k] not in names:
            raise ValueError(f"{k}: obs has no column {mapping[k]!r} (columns: {have}).")
    for k in _H5AD_LIST_KEYS:
        for c in mapping.get(k) or []:
            if c not in names:
                raise ValueError(f"{k}: obs has no column {c!r} (columns: {have}).")
    if mapping["counts"] not in info["matrices"]:
        raise ValueError(f"counts: the file has no matrix {mapping['counts']!r} "
                         f"(X or a layer: {', '.join(info['matrices']) or 'none'}).")
    if mapping.get("umap") and info["obsm"].get(mapping["umap"], [0, 0])[-1] < 2:
        raise ValueError(f"umap: obsm has no 2-column array {mapping['umap']!r}.")


def _h5ad_estimate(info: dict, mapping: dict) -> dict:
    """Cost of loading the file: {nnz, ram_gb (the count matrices held by the app: CSR,
    float32 values + int32 columns; the per-gene colour path adds a CSC copy of the same
    size per sample on first use), disk_gb (rough size of the converted stores)}."""
    mi = info["matrices"][mapping["counts"]]
    nnz = int(mi["nnz"])
    ram = (nnz * 8 + info["n_cells"] * 8) / 1e9
    return {"nnz": nnz, "encoding": mi["encoding"], "ram_gb": ram, "disk_gb": nnz * 2.6 / 1e9}


def _h5ad_cache_dir(path, mapping) -> str:
    path = os.path.abspath(os.path.expanduser(str(path)))
    st = os.stat(path)
    conv = {k: v for k, v in _h5ad_clean_mapping(mapping).items() if k not in _H5AD_LOAD_KEYS}
    tag = hashlib.md5(json.dumps([_H5AD_FORMAT, path, int(st.st_size), int(st.st_mtime_ns), conv],
                                 sort_keys=True).encode()).hexdigest()[:10]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(path)[:-len(".h5ad")])[:40] or "data"
    return os.path.join(core.CACHE_BASE_DIR, f"h5ad_{stem}_{tag}")


def _h5ad_row_blocks(node, block_nnz=None):
    """Yield (row0, row1, CSR block of the rows [row0, row1)) over a matrix node, reading
    about *block_nnz* stored values at a time. A CSC matrix cannot be read by rows: it is
    read whole (one block), which needs about 12 bytes per stored value twice over."""
    block_nnz = int(block_nnz or _H5AD_BLOCK_NNZ)
    mi = _h5ad_matrix_info(node)
    n, m = mi["shape"]
    if mi["encoding"] == "dense":
        step = max(1, block_nnz // max(1, m))
        for a in range(0, n, step):
            b = min(n, a + step)
            yield a, b, sp.csr_matrix(np.asarray(node[a:b]))
        return
    if mi["encoding"] == "csc":
        print(f"  h5ad: the matrix is stored column-wise (CSC) and is read whole: about "
              f"{mi['nnz'] * 24 / 1e9:.1f} GB of memory for this step", flush=True)
        yield 0, n, sp.csr_matrix(_h5ad_read_elem(node))
        return
    ip = np.asarray(node["indptr"][:], dtype=np.int64)
    a = 0
    while a < n:
        b = int(np.searchsorted(ip, ip[a] + block_nnz, side="right")) - 1
        b = min(n, max(b, a + 1))
        s, e = int(ip[a]), int(ip[b])
        yield a, b, sp.csr_matrix((node["data"][s:e], node["indices"][s:e], ip[a:b + 1] - s),
                                  shape=(b - a, m))
        a = b


def _h5ad_natural_key(s) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def _h5ad_unique_names(names) -> list:
    """*names* with repeats numbered ("a", "a-1", "a-2" …) so that every name is unique: a
    suffixed name that already exists ("a-1" beside two "a") is never produced twice."""
    names = [str(n) for n in names]
    used = set(names)                      # names that exist as they are keep them
    seen, nxt, out = set(), {}, []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
            continue
        k = nxt.get(n, 1)
        while f"{n}-{k}" in used:
            k += 1
        nxt[n] = k + 1
        used.add(f"{n}-{k}")
        out.append(f"{n}-{k}")
    return out


def _h5ad_level_str(v) -> str:
    """A level of an obs column as text ("3", not "3.0", for a whole number)."""
    if isinstance(v, (float, np.floating)) and np.isfinite(v) and float(v) == int(v):
        return str(int(v))
    return str(v)


def _h5ad_text_column(s) -> np.ndarray:
    """An obs column as text per cell, "" for a missing value."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        cats = np.array([_h5ad_level_str(c) for c in s.cat.categories] + [""], dtype=object)
        return cats[s.cat.codes.to_numpy()]
    out = np.array([_h5ad_level_str(v) for v in s.to_numpy()], dtype=object)
    out[s.isna().to_numpy()] = ""
    return out


def _h5ad_safe_names(names) -> list:
    """File-system names of the sections, unique."""
    out, seen = [], set()
    for nm in names:
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(nm)).strip("._") or "section"
        cand, i = base, 2
        while cand.lower() in seen:
            cand, i = f"{base}_{i}", i + 1
        seen.add(cand.lower())
        out.append(cand)
    return out


def _h5ad_write_store(store: str, X, obs: pd.DataFrame, genes, uns: dict) -> None:
    """One section as a SpatialData store (a table and nothing else), written under a
    temporary name and renamed so that an interrupted write is never taken for a store."""
    import shutil
    import warnings
    import anndata as ad
    import spatialdata
    from spatialdata.models import TableModel
    var = pd.DataFrame({"is_imputed": False}, index=pd.Index(list(genes), name="gene"))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    for k, v in uns.items():
        adata.uns[k] = v
    try:
        adata = TableModel.parse(adata)
    except Exception:
        pass
    tmp = store[:-len(".zarr")] + ".tmp.zarr"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(store, ignore_errors=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spatialdata.SpatialData(tables={"table": adata}).write(tmp)
    os.replace(tmp, store)


def _h5ad_manifest(cache_dir: str) -> "dict | None":
    """The manifest of a finished conversion whose stores are all present, else None."""
    try:
        with open(os.path.join(cache_dir, "manifest.json")) as fh:
            man = json.load(fh)
    except (OSError, ValueError):
        return None
    if man.get("format") != _H5AD_FORMAT:
        return None
    if not all(os.path.isdir(os.path.join(cache_dir, s["store"])) for s in man.get("sections", [])):
        return None
    return man


def _h5ad_convert(path, mapping, progress=None, force: bool = False, block_nnz=None) -> tuple:
    """Convert *path* into per-section stores (see the module text); (manifest, cache_dir).
    A finished conversion of the same file (path, size, modification time) with the same
    mapping is reused. *progress*(fraction, message) is called as the rows are read."""
    path = os.path.abspath(os.path.expanduser(str(path)))
    info = _h5ad_inspect(path)
    mapping = _h5ad_clean_mapping(mapping)
    _h5ad_check_mapping(info, mapping)
    cache_dir = _h5ad_cache_dir(path, mapping)
    if not force:
        man = _h5ad_manifest(cache_dir)
        if man is not None:
            return man, cache_dir
    t0 = time.time()
    say = (lambda f, m: None) if progress is None else progress
    os.makedirs(cache_dir, exist_ok=True)
    try:
        os.remove(os.path.join(cache_dir, "manifest.json"))
    except OSError:
        pass
    est = _h5ad_estimate(info, mapping)
    print(f"  h5ad: converting {os.path.basename(path)} ({info['n_cells']:,} cells × {info['n_genes']:,} "
          f"genes, {est['nnz']:,} stored values in '{mapping['counts']}', {est['encoding']}) into "
          f"per-section stores in {cache_dir} — about {est['disk_gb']:.1f} GB on disk; the loaded count "
          f"matrices will take about {est['ram_gb']:.1f} GB of memory", flush=True)

    with h5py.File(path, "r") as f:
        obs = _h5ad_read_elem(f["obs"])
        genes = [str(g) for g in _h5ad_read_elem(f["var"]).index]
        if len(set(genes)) != len(genes):
            raise ValueError("var names are not unique: make them unique (adata.var_names_make_unique()) first.")
        n = len(obs)
        ids = np.asarray(obs.index.astype(str), dtype=object)
        if mapping.get("x"):
            xy = np.column_stack([pd.to_numeric(obs[mapping["x"]], errors="coerce").to_numpy(dtype=float),
                                  pd.to_numeric(obs[mapping["y"]], errors="coerce").to_numpy(dtype=float)])
        else:
            xy = np.asarray(f["obsm"][mapping["spatial"]][:, :2], dtype=float)
        umap = (np.asarray(f["obsm"][mapping["umap"]][:, :2], dtype=float) if mapping.get("umap")
                else np.full((n, 2), np.nan))
        ok_xy = np.isfinite(xy).all(axis=1)

        if mapping.get("section"):
            sec_txt = _h5ad_text_column(obs[mapping["section"]])
            if (sec_txt == "").any():
                raise ValueError(f"{int((sec_txt == '').sum()):,} cells have no value in the section column "
                                 f"{mapping['section']!r}: every cell must belong to a section.")
            # natural order of the names (section_2 before section_10)
            sec_names = sorted(set(pd.unique(sec_txt)), key=_h5ad_natural_key)
            code_of = {nm: i for i, nm in enumerate(sec_names)}
            codes = np.fromiter((code_of[v] for v in sec_txt), dtype=np.int32, count=n)
        else:
            sec_names = [os.path.basename(path)[:-len(".h5ad")]]
            codes = np.zeros(n, dtype=np.int32)
        n_sec = len(sec_names)
        safe = _h5ad_safe_names(sec_names)
        last_row = np.full(n_sec, -1, dtype=np.int64)
        np.maximum.at(last_row, codes, np.arange(n, dtype=np.int64))

        def _level_per_section(key):
            """The level of a condition / donor column for each section: its most frequent
            value there; sections where it is not constant are reported."""
            if not mapping.get(key):
                return [""] * n_sec, []
            txt = _h5ad_text_column(obs[mapping[key]])
            lv, mixed = [], []
            for c in range(n_sec):
                vc = pd.Series(txt[codes == c]).value_counts()
                lv.append(str(vc.index[0]) if len(vc) else "")
                if len(vc) > 1:
                    mixed.append(sec_names[c])
            return lv, mixed

        cond_of, cond_mixed = _level_per_section("condition")
        donor_of, donor_mixed = _level_per_section("donor")
        ct_cols = {f"obs{i}": c for i, c in enumerate(mapping["cell_types"])}
        ct_txt = {m: _h5ad_text_column(obs[c]) for m, c in ct_cols.items()}
        # cluster sets: one integer id per level, the same in every section
        cl_cols, cl_codes, cl_levels = {}, {}, {}
        for c in mapping["clusters"]:
            nm = re.sub(r"[^A-Za-z0-9]+", "_", c).strip("_") or "clusters"
            while nm in cl_cols:
                nm += "_"
            txt = _h5ad_text_column(obs[c])
            lv = sorted(set(txt) - {""}, key=_h5ad_natural_key)
            if lv and all(re.fullmatch(r"\d+", v) for v in lv) and "0" not in lv:
                # whole-number labels keep their value (0 is the app's "no cluster")
                cl_codes[nm] = np.array([int(v) if v else 0 for v in txt], dtype=np.int64)
                cl_levels[nm] = {v: int(v) for v in lv}
            else:
                lut = {v: i + 1 for i, v in enumerate(lv)}
                cl_codes[nm] = np.array([lut.get(v, 0) for v in txt], dtype=np.int64)
                cl_levels[nm] = lut
            cl_cols[nm] = c
        del obs

        node = _h5ad_matrix_node(f, mapping["counts"])
        mi = _h5ad_matrix_info(node)
        if mi["shape"] != (n, len(genes)):
            raise ValueError(f"Matrix '{mapping['counts']}' has shape {mi['shape']}, expected {(n, len(genes))}.")
        pieces: dict = {}
        stats_c = {"nnz": 0, "n_noninteger": 0, "n_negative": 0, "max": 0.0}
        sections = [None] * n_sec
        empty = []                       # (section, cells dropped) of sections without a usable cell

        def _finish(c):
            rows = np.concatenate([r for r, _ in pieces[c]])
            X = sp.vstack([p for _, p in pieces.pop(c)], format="csr") if len(rows) else \
                sp.csr_matrix((0, len(genes)), dtype=np.float32)
            keep = ok_xy[rows]
            n_drop = int((~keep).sum())
            if n_drop:
                rows, X = rows[keep], X[np.flatnonzero(keep)]
            if not len(rows):
                # nothing left of this section (no cell with coordinates): no sample for it
                empty.append((sec_names[c], n_drop))
                return
            cid = pd.Index(ids[rows])
            if not cid.is_unique:
                # the same obs name twice in one section: keep both cells, numbered
                cid = pd.Index(_h5ad_unique_names(cid))
            tot = np.asarray(X.sum(axis=1)).ravel()
            df = pd.DataFrame({"x_centroid": xy[rows, 0], "y_centroid": xy[rows, 1],
                               "transcript_counts": tot, "total_counts": tot,
                               "umap_1": umap[rows, 0], "umap_2": umap[rows, 1],
                               "source_row": rows.astype(np.int64)}, index=cid)
            for m_, txt in ct_txt.items():
                df[f"cell_type_{m_}"] = txt[rows].astype(str)
            for nm in cl_cols:
                df[f"clust__{nm}"] = cl_codes[nm][rows]
            meta = {"run_name": sec_names[c], "panel_name": os.path.basename(path),
                    "region_name": " · ".join(v for v in (cond_of[c], donor_of[c] and f"donor {donor_of[c]}") if v),
                    "num_cells": int(len(df)), "num_transcripts": int(round(float(tot.sum()))),
                    "source": "h5ad", "h5ad_section": sec_names[c], "h5ad_condition": cond_of[c],
                    "h5ad_donor": donor_of[c]}
            _h5ad_write_store(os.path.join(cache_dir, safe[c] + ".zarr"), X, df, genes,
                              {"metadata": meta, "cluster_methods": list(cl_cols), "imputed_genes": []})
            sections[c] = {"name": sec_names[c], "store": safe[c] + ".zarr", "n_cells": int(len(df)),
                           "n_dropped_no_coordinates": n_drop, "condition": cond_of[c], "donor": donor_of[c],
                           "nnz": int(X.nnz)}
            print(f"  h5ad: section {c + 1}/{n_sec} written ({len(df):,} cells)", flush=True)

        for a, b, blk in _h5ad_row_blocks(node, block_nnz):
            jobs._job_check()
            d = blk.data
            stats_c["nnz"] += int(d.size)
            if d.size:
                if d.dtype.kind == "f":
                    stats_c["n_noninteger"] += int(np.count_nonzero(d != np.rint(d)))
                if d.dtype.kind in "fi":
                    stats_c["n_negative"] += int(np.count_nonzero(d < 0))
                stats_c["max"] = max(stats_c["max"], float(d.max()))
            blk = blk.astype(np.float32)
            cb = codes[a:b]
            edges = np.flatnonzero(np.diff(cb)) + 1
            if len(edges) < 64:
                # runs of one section: plain row slices
                for s, e in zip(np.r_[0, edges], np.r_[edges, b - a]):
                    pieces.setdefault(int(cb[s]), []).append((np.arange(a + s, a + e), blk[s:e]))
            else:
                for c in np.unique(cb):
                    r = np.flatnonzero(cb == c)
                    pieces.setdefault(int(c), []).append((a + r, blk[r]))
            for c in [c for c in pieces if last_row[c] < b]:
                _finish(c)
            say(b / max(1, n), f"Converting: {b:,} of {n:,} cells read…")
        for c in list(pieces):
            _finish(c)

    sections = [sec for sec in sections if sec is not None]
    if not sections:
        raise ValueError("No cell of the file has finite coordinates in the chosen columns.")
    integer = stats_c["n_noninteger"] == 0 and stats_c["n_negative"] == 0
    man = {"format": _H5AD_FORMAT, "source": path, "source_size": info["size_bytes"],
           "mapping": mapping, "n_cells": int(sum(s["n_cells"] for s in sections)), "n_genes": len(genes),
           "counts": {"matrix": mapping["counts"], "integer": bool(integer), **stats_c},
           "celltype_methods": ct_cols, "cluster_sets": cl_cols, "cluster_levels": cl_levels,
           "sections": sections,
           "n_dropped_no_coordinates": int(sum(s["n_dropped_no_coordinates"] for s in sections)
                                           + sum(n for _nm, n in empty)),
           "sections_without_cells": [nm for nm, _n in empty],
           "condition_not_constant_in": cond_mixed, "donor_not_constant_in": donor_mixed,
           "seconds": round(time.time() - t0, 1)}
    tmp = os.path.join(cache_dir, "manifest.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(man, fh, indent=1)
    os.replace(tmp, os.path.join(cache_dir, "manifest.json"))
    print(f"  h5ad: {len(sections)} section(s), {man['n_cells']:,} cells converted in {man['seconds']:.0f} s", flush=True)
    for w in _h5ad_warnings(man):
        print(f"  h5ad: WARNING — {w}", flush=True)
    return man, cache_dir


def _h5ad_warnings(man: dict) -> list:
    """What the user must know about a converted file before testing anything on it."""
    out = []
    c = man["counts"]
    if not c["integer"]:
        what = []
        if c["n_noninteger"]:
            what.append(f"{c['n_noninteger']:,} of {c['nnz']:,} stored values are not whole numbers")
        if c["n_negative"]:
            what.append(f"{c['n_negative']:,} are negative")
        rs = (man.get("mapping") or {}).get("noninteger_counts") == "round_sums"
        out.append(f"Matrix '{c['matrix']}' does not hold raw counts ({'; '.join(what)}; maximum "
                   f"{c['max']:.4g}). It was loaded as it is. Everything that models counts assumes raw "
                   f"integer counts: name the matrix with the raw counts if the file has one. "
                   + ("Pseudobulk tests sum these values per sample and round the sums "
                      "(noninteger_counts = round_sums), which the results say."
                      if rs else
                      "Pseudobulk differential expression (DESeq2 / edgeR) refuses this matrix; if the "
                      "file only has corrected counts, set \"noninteger_counts\": \"round_sums\" in the "
                      "mapping to sum them per sample and round the sums."))
    if man.get("n_dropped_no_coordinates"):
        out.append(f"{man['n_dropped_no_coordinates']:,} cells without finite x / y were left out.")
    if man.get("sections_without_cells"):
        out.append(f"{len(man['sections_without_cells'])} section(s) have no cell with coordinates and were "
                   f"not loaded: {', '.join(man['sections_without_cells'][:6])}"
                   f"{'…' if len(man['sections_without_cells']) > 6 else ''}.")
    for key, what in (("condition_not_constant_in", "condition"), ("donor_not_constant_in", "donor")):
        if man.get(key):
            out.append(f"The {what} column is not constant within {len(man[key])} section(s) "
                       f"({', '.join(man[key][:6])}{'…' if len(man[key]) > 6 else ''}): the most frequent "
                       f"value of each section was used. A sample of the app is one section.")
    return out


# ─── loading the converted stores ─────────────────────────────────────────────
def _h5ad_next_offset(prev: "dict | None", new: dict) -> float:
    """x offset (µm) that places *new* to the right of *prev* (None: first sample)."""
    if prev is None:
        return 0.0
    px, nx = prev["df"]["x_centroid"], new["df"]["x_centroid"]
    if not len(px) or not len(nx):
        return float(prev.get("x_offset", 0.0))
    return float(px.max()) + float(prev.get("x_offset", 0.0)) - float(nx.min()) + _H5AD_SAMPLE_GAP_UM


def _h5ad_load_stores(man: dict, cache_dir: str, progress=None) -> list:
    """The dataset dicts of the sections of a converted file, in order, each with its
    x offset in the side-by-side view and the section's levels under "h5ad"."""
    out = []
    n = len(man["sections"])
    for i, s in enumerate(man["sections"]):
        jobs._job_check()
        if progress is not None:
            progress(i / max(1, n), f"Loading section {i + 1} of {n}…")
        ds = loading.load_from_sdata_zarr(os.path.join(cache_dir, s["store"]))
        ds["h5ad"] = {"section": s["name"], "condition": s.get("condition", ""),
                      "donor": s.get("donor", ""), "source": man["source"]}
        if not man["counts"]["integer"]:
            ds["counts_noninteger"] = (man.get("mapping") or {}).get("noninteger_counts") or "refuse"
        if i:
            ds["x_offset"] = _h5ad_next_offset(out[-1], ds)
        out.append(ds)
    return out


def _h5ad_load(path, mapping=None, progress=None) -> tuple:
    """(dataset dicts, manifest, cache_dir) for an .h5ad: converts it when no finished
    conversion with this mapping is cached, then loads the stores."""
    path = os.path.abspath(os.path.expanduser(str(path)))
    mapping = _h5ad_mapping_for(path) if mapping is None else _h5ad_clean_mapping(mapping)
    conv = (lambda f, m: progress(0.8 * f, m)) if progress is not None else None
    reused = _h5ad_manifest(_h5ad_cache_dir(path, _h5ad_clean_mapping(mapping))) is not None
    man, cache_dir = _h5ad_convert(path, mapping, progress=conv)
    # from_cache: the stores of a finished conversion were reused (for the provenance record)
    man = dict(man, mapping=dict(man["mapping"], **{k: mapping.get(k) for k in _H5AD_LOAD_KEYS}),
               from_cache=bool(reused))
    load = (lambda f, m: progress(0.8 + 0.2 * f, m)) if progress is not None else None
    return _h5ad_load_stores(man, cache_dir, progress=load), man, cache_dir


def _h5ad_record(man=None) -> None:
    """The provenance record of the conversion, written once the converted sections are
    the loaded dataset (so it is filed under them and describes them): the file, the
    column mapping, whether the count matrix is integer-valued and what the mapping says
    about a matrix that is not (noninteger_counts), and whether the stores were reused."""
    man = man or _h5ad_loaded.get("manifest")
    if not man:
        return
    from . import provenance
    c = man.get("counts") or {}
    mp = dict(man.get("mapping") or {})
    provenance._prov_record(
        "h5ad", params={"file": os.path.basename(str(man.get("source") or "")), "mapping": mp},
        packages=("anndata", "h5py"), message=_h5ad_summary(man), cache=bool(man.get("from_cache")),
        result={"n_sections": len(man.get("sections") or []), "counts_matrix": c.get("matrix"),
                "counts_integer": bool(c.get("integer", True)),
                "n_noninteger": int(c.get("n_noninteger") or 0), "n_negative": int(c.get("n_negative") or 0),
                "noninteger_counts": str(mp.get("noninteger_counts") or "refuse")},
        once=("h5ad", str(man.get("source") or ""), json.dumps(mp, sort_keys=True, default=str)))


def _h5ad_register(man: "dict | None", cache_dir=None) -> None:
    """Make the cell-type columns of the file annotation "methods" (obs0, obs1 … shown
    under the column's name) and remember the file; None forgets both."""
    for m in [m for m in core._ANNOT_METHODS if _H5AD_METHOD_RE.match(m)]:
        core._ANNOT_METHODS.pop(m, None)
    _h5ad_loaded.update({"manifest": man, "cache_dir": cache_dir if man else None})
    if man:
        for m, col in (man.get("celltype_methods") or {}).items():
            core._ANNOT_METHODS[m] = str(col)


def _h5ad_primary_labels() -> list:
    """Put the cell-type columns of the primary sample into the annotation state (the
    additional samples are read from their cell_type_<method> columns); the methods set."""
    man = _h5ad_loaded.get("manifest")
    done = []
    if not man:
        return done
    df = core.DATA["df"]
    with core._annot_lock:
        for m in (man.get("celltype_methods") or {}):
            col = f"cell_type_{m}"
            if col not in df.columns:
                continue
            lab = df[col].astype(str)
            lab = lab[~lab.isin(["", "nan", "<NA>"])]
            lab.index = lab.index.astype(str)
            core._annot_state[f"labels_{m}"] = lab
            core._annot_state["status"] = "done"
            core._annot_state["message"] = f"Cell types read from the file ({core._ANNOT_METHODS.get(m, m)})"
            done.append(m)
    return done


def _h5ad_levels(man=None) -> list:
    """Condition levels of the sections, in order of first appearance."""
    man = man or _h5ad_loaded.get("manifest") or {}
    return list(dict.fromkeys(s.get("condition", "") for s in man.get("sections", []) if s.get("condition")))


def _h5ad_design(cond_a=None, cond_b=None, force: bool = True, man=None, cache_dir=None) -> str:
    """Fill the sample design from the file: donor of every section, and condition A / B
    for the sections of the two levels compared (other levels: excluded). The levels are
    *cond_a* / *cond_b*, else the mapping's, else the two levels when the column has
    exactly two. With force=False a design already saved for these samples is kept.
    Callable from the REPL to compare another pair of levels: _h5ad_design("ctrl", "treated").
    Returns a sentence for the status line."""
    man = man or _h5ad_loaded.get("manifest")
    cache_dir = cache_dir or _h5ad_loaded.get("cache_dir")
    if not man:
        return "No .h5ad is loaded."
    # the keys the loaded samples are known by (dataview._ds_key: an absolute path), also
    # when the cache directory was given as a relative one
    keys = [os.path.abspath(os.path.join(cache_dir, s["store"])) for s in man["sections"]]
    levels = _h5ad_levels(man)
    mp = man.get("mapping") or {}
    a, b = cond_a or mp.get("cond_a"), cond_b or mp.get("cond_b")
    saved = {}
    if not force:
        p = stats._design_path(stats._design_keys())
        if p is not None and os.path.isfile(p):
            # A saved design is kept - unless a pair of levels is asked for that it does not
            # hold (the first load saves a design even when nothing could be assigned, and
            # the cache directory does not depend on cond_a / cond_b). Then the conditions
            # are assigned again; donor and batch edits of the saved design stay.
            cur = stats._design_samples()
            names = stats._design_names()
            if not (a or b) or (any(r["cond"] for r in cur)
                                and (names.get("A"), names.get("B")) == (str(a), str(b))):
                return "The sample design saved for these samples was kept."
            saved = {r["key"]: r for r in cur}
    note = ""
    if (a or b) and (a not in levels or b not in levels or a == b):
        # the donors are still filled from the file; only the conditions stay unassigned
        note = (f" Conditions not assigned: {a!r} / {b!r} are not two levels of the condition column "
                f"(levels: {', '.join(levels) or 'none'}).")
        a = b = None
    elif not a and len(levels) == 2:
        a, b = levels
    elif not a and len(levels) > 2:
        note = (f" The condition column has {len(levels)} levels ({', '.join(levels)}): no sample was "
                f"assigned to A or B — pick the two to compare (Samples and conditions, or "
                f"XENIUM_H5AD_COND_A / XENIUM_H5AD_COND_B, or _h5ad_design(a, b) in the REPL).")
    rows = []
    for k, s in zip(keys, man["sections"]):
        c = s.get("condition", "")
        old = saved.get(k) or {}
        rows.append({"key": k, "cond": "A" if (a and c == a) else "B" if (b and c == b) else "",
                     "batch": str(old.get("batch") or ""),
                     "donor": str(old.get("donor") or s.get("donor", "") or "")})
    stats._design_set(rows, {"A": a or "A", "B": b or "B"})
    n_a, n_b = sum(r["cond"] == "A" for r in rows), sum(r["cond"] == "B" for r in rows)
    donors = {r["donor"] for r in rows if r["donor"]}
    msg = f"Sample design filled from the file: {len(donors)} donor(s) over {len(rows)} section(s)"
    if a:
        d_a = {r["donor"] for r in rows if r["cond"] == "A" and r["donor"]}
        d_b = {r["donor"] for r in rows if r["cond"] == "B" and r["donor"]}
        msg += (f"; A = {a}: {n_a} section(s)" + (f" of {len(d_a)} donor(s)" if donors else "")
                + f", B = {b}: {n_b} section(s)" + (f" of {len(d_b)} donor(s)" if donors else ""))
        if len(rows) - n_a - n_b:
            msg += f"; {len(rows) - n_a - n_b} section(s) of other levels excluded"
    return msg + "." + note


def _h5ad_summary(man=None) -> str:
    """One paragraph for the load status: what was loaded, warnings, what is unavailable."""
    man = man or _h5ad_loaded.get("manifest")
    if not man:
        return ""
    c = man["counts"]
    parts = [f"{len(man['sections'])} section(s), {man['n_cells']:,} cells × {man['n_genes']:,} genes from "
             f"{os.path.basename(man['source'])}; counts from '{c['matrix']}'"
             + (" (integer-valued: verified over every stored value)." if c["integer"] else " (NOT raw counts).")]
    parts += ["WARNING: " + w for w in _h5ad_warnings(man)]
    parts.append(_h5ad_unavailable_note())
    return " ".join(parts)
