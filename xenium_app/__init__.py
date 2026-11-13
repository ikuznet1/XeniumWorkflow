"""Xenium Explorer, as a package. Start it with ``python xenium_explorer.py <bundle>``.

Importing the package starts the app exactly as importing the single file did: it loads
the dataset named by DATA_DIR / sys.argv, builds the Dash app and registers the callbacks
and the WebGL viewer endpoints. The order below is the order of those side effects, so it
matters:

1. library modules - definitions only (plus the log capture in core and the atexit hook
   in jobs). They refer to each other as ``module.NAME`` and may import each other
   freely, but never import a module of group 2.
2. startup -> app_layout -> cb_* -> viewer_provider - the import-time work, in the order
   the single file ran it (callbacks are registered in their original order).

See compat.py for how ``xenium_explorer.<name>`` and the REPL reach the globals of these
modules.
"""
MODULES = (
    # 1. libraries
    "core", "jobs", "caching", "ui", "loading", "dataview", "annotation", "reseg",
    "clustering", "split", "spage", "subsetting", "spatial", "morph", "figures",
    "analysis", "cellnest", "sdata_store", "regions", "imgseg", "stats", "h5ad", "controls",
    "proximity", "lrdecomp", "aggregates", "segfree", "txfactor", "clustersets", "banksy",
    "joint", "provenance", "batch",
    # 2. import-time work
    "startup", "app_layout", "cb_plots", "cb_tools", "cb_reseg", "cb_spage", "cb_data",
    "cb_spatial", "cb_analysis", "cb_stats", "cb_controls", "cb_proximity", "cb_lrdecomp",
    "cb_aggregates", "cb_segfree", "cb_txfactor", "cb_banksy", "cb_joint", "cb_provenance", "cb_nav",
    "viewer_provider",
)

import json as _json
import os as _os
import sys as _sys


def _batch_config_file(path: str) -> dict:
    """A batch config file as a dict: JSON, or YAML when PyYAML is installed. Raises the
    batch module's error type with a message for the user (ValueError before that module
    exists, i.e. while the package is being imported)."""
    err = getattr(_sys.modules.get(__name__ + ".batch") or _sys.modules.get("__main__"), "_BatchError", ValueError)
    try:
        with open(_os.path.expanduser(str(path)), "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise err(f"cannot read the config file {path}: {exc}") from None
    if str(path).lower().endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError:
            raise err("this config is YAML, and PyYAML is not installed: install it, or write the "
                      "config as JSON") from None
        try:
            cfg = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise err(f"{path} is not valid YAML: {exc}") from None
    else:
        try:
            cfg = _json.loads(text)
        except ValueError as exc:
            raise err(f"{path} is not valid JSON: {exc}") from None
    if not isinstance(cfg, dict):
        raise err(f"{path}: the config must be a mapping")
    return cfg


def _batch_preconfigure() -> bool:
    """``python -m xenium_app.batch <config>``: the dataset and the cache root are read
    from the environment while the package is imported (below), so they are taken from
    the config file now. True when Python was started that way."""
    if _sys.argv[:1] != ["-m"] or f"{__name__}.batch" not in list(getattr(_sys, "orig_argv", [])):
        return False
    files = [a for a in _sys.argv[1:] if not a.startswith("-")]
    if not files:
        if not _os.environ.get("DATA_DIR"):
            print("usage: python -m xenium_app.batch CONFIG [--dry-run] [--list-steps]\n"
                  "The app loads a dataset when it starts: name a config file (its data.path is used), "
                  "or set DATA_DIR for --list-steps alone.", flush=True)
            raise SystemExit(2)
        return True
    try:
        cfg = _batch_config_file(files[0])
        data = cfg.get("data")
        data = {"path": data} if isinstance(data, str) else (data or {})
        if not isinstance(data, dict) or not data.get("path"):
            raise ValueError("data.path is missing (a Xenium bundle directory, a .zarr store or an .h5ad file)")
    except ValueError as exc:
        print(f"Batch error: {exc}", flush=True)
        raise SystemExit(2)
    path = _os.path.abspath(_os.path.expanduser(str(data["path"])))
    if not _os.path.exists(path):
        print(f"Batch error: data.path does not exist: {path}", flush=True)
        raise SystemExit(2)
    _os.environ["DATA_DIR"] = path
    if cfg.get("cache_dir"):
        _os.environ["XENIUM_CACHE_DIR"] = _os.path.abspath(_os.path.expanduser(str(cfg["cache_dir"])))
        _os.makedirs(_os.environ["XENIUM_CACHE_DIR"], exist_ok=True)
    mapping = data.get("h5ad_mapping")
    if isinstance(mapping, str):
        _os.environ["XENIUM_H5AD_MAPPING"] = _os.path.abspath(_os.path.expanduser(mapping))
    elif isinstance(mapping, dict):
        mdir = _os.environ.get("XENIUM_CACHE_DIR") or _os.path.join(_os.path.expanduser("~"), ".xenium_explorer_cache")
        _os.makedirs(mdir, exist_ok=True)
        mfile = _os.path.join(mdir, "batch_h5ad_mapping.json")
        with open(mfile, "w", encoding="utf-8") as fh:
            _json.dump(mapping, fh)
        _os.environ["XENIUM_H5AD_MAPPING"] = mfile
    for var in ("XENIUM_ENABLE_REPL", "XENIUM_RENDERER"):      # a batch run has no browser and no REPL
        _os.environ.pop(var, None)
    return True


_BATCH_MAIN = _batch_preconfigure()

from . import compat  # noqa: E402,F401
from . import (  # noqa: E402,F401
    core, jobs, caching, ui, loading, dataview, annotation, reseg, clustering, split,
    spage, subsetting, spatial, morph, figures, analysis, cellnest, sdata_store, regions, imgseg,
    stats, h5ad, controls, proximity, lrdecomp, aggregates, segfree, txfactor, clustersets, banksy,
)
from . import (  # noqa: E402,F401
    startup, app_layout, cb_plots, cb_tools, cb_reseg, cb_spage, cb_data, cb_spatial,
    cb_analysis, cb_stats, cb_controls, cb_proximity, cb_lrdecomp, cb_aggregates, cb_segfree, cb_txfactor,
    cb_banksy, viewer_provider,
)
from . import joint, cb_joint  # noqa: E402,F401
from . import provenance, cb_provenance  # noqa: E402,F401
if not _BATCH_MAIN:      # run as a script, batch.py is "__main__" and registers itself under its own name
    from . import batch  # noqa: E402,F401
from . import cb_nav  # noqa: E402,F401
