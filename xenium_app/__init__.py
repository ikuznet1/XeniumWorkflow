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
    "analysis", "cellnest", "sdata_store", "regions", "imgseg", "stats", "controls",
    "proximity", "lrdecomp", "aggregates", "segfree", "txfactor", "clustersets", "banksy",
    "joint",
    # 2. import-time work
    "startup", "app_layout", "cb_plots", "cb_tools", "cb_reseg", "cb_spage", "cb_data",
    "cb_spatial", "cb_analysis", "cb_stats", "cb_controls", "cb_proximity", "cb_lrdecomp",
    "cb_aggregates", "cb_segfree", "cb_txfactor", "cb_banksy", "cb_joint", "viewer_provider",
)

from . import compat  # noqa: E402,F401
from . import (  # noqa: E402,F401
    core, jobs, caching, ui, loading, dataview, annotation, reseg, clustering, split,
    spage, subsetting, spatial, morph, figures, analysis, cellnest, sdata_store, regions, imgseg,
    stats, controls, proximity, lrdecomp, aggregates, segfree, txfactor, clustersets, banksy,
)
from . import (  # noqa: E402,F401
    startup, app_layout, cb_plots, cb_tools, cb_reseg, cb_spage, cb_data, cb_spatial,
    cb_analysis, cb_stats, cb_controls, cb_proximity, cb_lrdecomp, cb_aggregates, cb_segfree, cb_txfactor,
    cb_banksy, viewer_provider,
)
from . import joint, cb_joint  # noqa: E402,F401
