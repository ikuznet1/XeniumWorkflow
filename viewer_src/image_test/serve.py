"""Test server for the morphology-image layer.

Serves the real /viewer/* routes (xenium_viewer_api via dev_server.BundleProvider, so
image bytes go through the same Flask send_file(conditional=True) Range path as the
app) plus this directory's test page and build output.

    python viewer_src/image_test/serve.py /path/to/xenium_outs [port]

    /                     test page (index.html)
    /assets/<name>        image_test/dist/<name>   (mimics Dash's assets/ URL layout)
    /test/cells?n=20&x0=&y0=&x1=&y1=   n cell centroids (µm) inside a box, from cells.parquet,
                          each with its nucleus outline from nucleus_boundaries.parquet
"""
import os
import sys

from flask import Flask, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import dev_server  # noqa: E402  (not modified; reused for its BundleProvider)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8085
    provider = dev_server.BundleProvider(sys.argv[1])
    app = Flask(__name__)
    dev_server.va.register(app, provider)

    @app.route("/")
    def index():
        return send_from_directory(HERE, "index.html")

    @app.route("/assets/<path:name>")
    def assets(name):
        resp = send_from_directory(os.path.join(HERE, "dist"), name)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.route("/test/cells")
    def cells():
        d = provider.df
        a = request.args
        if "x0" in a:
            d = d[(d.x_centroid >= float(a["x0"])) & (d.x_centroid <= float(a["x1"]))
                  & (d.y_centroid >= float(a["y0"])) & (d.y_centroid <= float(a["y1"]))]
        n = min(int(a.get("n", 20)), len(d))
        d = d.sample(n=n, random_state=0) if n else d
        import pandas as pd
        nb = pd.read_parquet(os.path.join(provider.data_dir(), "nucleus_boundaries.parquet"),
                             filters=[("cell_id", "in", list(d.cell_id))])
        outlines = {k: [[float(x), float(y)] for x, y in zip(g.vertex_x, g.vertex_y)]
                    for k, g in nb.groupby("cell_id")}
        return jsonify([{"centroid": [float(x), float(y)], "nucleus": outlines.get(cid)}
                        for cid, x, y in zip(d.cell_id, d.x_centroid, d.y_centroid)])

    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
