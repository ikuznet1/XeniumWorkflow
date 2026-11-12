"""Standalone test server for the WebGL viewer bundle.

Serves the same /viewer/* endpoints as the app, backed directly by a Xenium output
bundle, plus a bare test page, so the JavaScript can be developed and tested without
starting the full Dash app.

    python viewer_src/dev_server.py /path/to/xenium_outs [port]

Then open http://127.0.0.1:8070/ (the page loads assets/xenium_viewer.js).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from flask import Flask, send_from_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import xenium_viewer_api as va  # noqa: E402

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
           "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896"]


class BundleProvider:
    """Minimal provider: cells.parquet for coordinates and QC colours, the feature matrix
    for genes, and a quantile binning of transcript counts as a stand-in categorical."""

    def __init__(self, data_dir):
        self._dir = os.path.abspath(data_dir)
        self.df = pd.read_parquet(os.path.join(self._dir, "cells.parquet"))
        self._expr = None

    def data_dir(self):
        return self._dir

    def _bounds(self):
        d = self.df
        return [float(d.x_centroid.min()), float(d.y_centroid.min()),
                float(d.x_centroid.max()), float(d.y_centroid.max())]

    def image_files(self):
        return sorted(glob.glob(os.path.join(self._dir, "morphology_focus", "*.ome.tif")))

    def meta(self):
        store = va.tile_store(self._dir)
        return {
            "dataset": os.path.basename(self._dir),
            "sources": ["xenium"],
            "bounds": self._bounds(),
            "pixel_size": 0.2125,
            "has_boundaries": store.has_boundaries,
            "has_transcripts": store.has_transcripts,
            "images": [{"index": i, "name": os.path.basename(p), "v": va.image_token(p)}
                       for i, p in enumerate(self.image_files())],
            "color_modes": ["cluster", "transcript_counts", "cell_area", "nucleus_area", "gene"],
        }

    def cells(self, source):
        d = self.df
        xy = np.empty(2 * len(d), dtype=np.float32)
        xy[0::2] = d.x_centroid.values
        xy[1::2] = d.y_centroid.values
        return {"n": len(d), "token": f"{source}:{len(d)}", "bounds": self._bounds()}, {"xy": xy}

    def _gene(self, gene):
        if self._expr is None:
            import h5py
            import scipy.sparse as sp
            with h5py.File(os.path.join(self._dir, "cell_feature_matrix.h5"), "r") as f:
                names = [g.decode() for g in f["matrix/features/name"][:]]
                mat = sp.csc_matrix((f["matrix/data"][:], f["matrix/indices"][:],
                                     f["matrix/indptr"][:]), shape=tuple(f["matrix/shape"][:]))
            self._expr = ({n: i for i, n in enumerate(names)}, mat.tocsr())
        idx, mat = self._expr
        if gene not in idx:
            return None
        return np.log1p(np.asarray(mat[idx[gene]].todense()).ravel()).astype(np.float32)

    def colors(self, args):
        mode = args.get("mode", "transcript_counts")
        d = self.df
        if mode == "cluster":
            codes = pd.qcut(d.transcript_counts, 8, labels=False, duplicates="drop").astype(np.uint16)
            k = int(codes.max()) + 1
            return ({"kind": "categorical", "label": "Cluster (test bins)",
                     "categories": [f"Bin {i + 1}" for i in range(k)],
                     "colors": [PALETTE[i % len(PALETTE)] for i in range(k)]}, {"codes": codes.values})
        if mode == "gene":
            vals = self._gene(args.get("gene", ""))
            if vals is None:
                vals = np.full(len(d), np.nan, dtype=np.float32)
            label = f"{args.get('gene', '')} (log1p)"
        else:
            col = mode if mode in d.columns else "transcript_counts"
            vals = d[col].values.astype(np.float32)
            if col == "transcript_counts":
                vals = np.log1p(vals)
            label = col
        finite = vals[np.isfinite(vals)]
        return ({"kind": "continuous", "label": label,
                 "min": float(finite.min()) if len(finite) else 0.0,
                 "max": float(finite.max()) if len(finite) else 1.0}, {"values": vals})

    def boundaries(self, source, kind, x0, y0, x1, y1):
        return None   # use the bundle's cells.zarr.zip

    def overlays(self, source, args=None):
        return {"polygons": [], "lines": []}

    def cell_at(self, source, index):
        if index < 0 or index >= len(self.df):
            return {}
        row = self.df.iloc[index]
        return {"index": index, "id": str(row.cell_id),
                "transcript_counts": int(row.transcript_counts), "cell_area": float(row.cell_area)}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8070
    app = Flask(__name__)
    va.register(app, BundleProvider(sys.argv[1]))

    @app.route("/")
    def index():
        return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "dev.html")

    @app.route("/assets/<path:name>")
    def assets(name):
        return send_from_directory(os.path.join(ROOT, "assets"), name)

    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
