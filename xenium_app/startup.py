"""Import-time start-up, in the order the single-file app ran it: locate and load the
dataset, derive caches, autoload cached results. Imported after every library module."""

import os
import sys

from . import core, ui, loading, dataview, annotation, spage, analysis, sdata_store, regions
from . import clustersets


# ─── Locate & load data ───────────────────────────────────────────────────────
if os.environ.get("DATA_DIR"):
    _startup_arg = os.environ["DATA_DIR"]
elif len(sys.argv) > 1:
    _startup_arg = sys.argv[1]
else:
    dirs = [d for d in os.listdir(".") if d.startswith("output-") and os.path.isdir(d)]
    if not dirs:
        print("ERROR: No Xenium output directory found. Pass path as argument or set DATA_DIR env var.")
        sys.exit(1)
    _startup_arg = dirs[0]

if _startup_arg.rstrip("/").endswith(".zarr"):
    core.DATA = loading.load_from_sdata_zarr(_startup_arg)
else:
    core.DATA = loading.load_xenium_data(_startup_arg)


dataview._refresh_derived_data()


# ─── App layout ───────────────────────────────────────────────────────────────
# Cluster sets computed in the app for this dataset (Banksy domains, joint clusters)
clustersets._cs_restore("xenium")

meta            = core.DATA["metadata"]
cluster_methods = core.DATA["cluster_methods"]
gene_names      = core.DATA["gene_names"]


cluster_options = clustersets._cs_options("xenium")

# Auto-load any previously cached SpaGE results for this dataset
annotation._annot_autoload()
spage._spage_autoload()


regions._auto_load_rois()
analysis._an_autoload()


# Auto-load SpatialData cache if it exists for this dataset
sdata_store._sdata_autoload()
