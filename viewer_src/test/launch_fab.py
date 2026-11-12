"""The real app plus test-only routes that fabricate state the parity features consume.
   DATA_DIR=... XENIUM_RENDERER=webgl PORT=8098 python launch_fab.py"""
import os, sys
sys.path.insert(0, "/Users/ikuz/Documents/XeniumWorkflow")
import numpy as np, pandas as pd
import xenium_explorer as xe
from flask import jsonify, request
rng = np.random.default_rng(1)

def _near(n=400):
    df = xe.DATA["df"]
    x, y = float(request.args["x"]), float(request.args["y"])
    d = np.hypot(df.x_centroid.values - x, df.y_centroid.values - y)
    return df.iloc[np.argsort(d)[:n]]

@xe.app.server.route("/_fab/<what>")
def fab(what):
    df = xe.DATA["df"]
    if what == "proseg":
        sub = _near(300)
        cdf = pd.DataFrame({"x_centroid": sub.x_centroid.values + 3, "y_centroid": sub.y_centroid.values + 3,
                            "transcript_counts": 10.0}, index=pd.Index(np.arange(1, len(sub) + 1), name="cell"))
        t = np.linspace(0, 2 * np.pi, 9)
        b = {int(i): (x + 5 * np.cos(t), y + 5 * np.sin(t)) for i, (x, y) in zip(cdf.index, cdf[["x_centroid", "y_centroid"]].values)}
        with xe._proseg_lock:
            xe._proseg_state.update({"status": "done", "result": {"cells_df": cdf, "cell_bounds": b, "out_dir": "/tmp/fake", "expr": None, "source": "proseg"}})
        return jsonify(n=len(cdf))
    if what == "sdata":
        import geopandas as gpd
        from shapely.geometry import Polygon, box
        x, y = float(request.args["x"]), float(request.args["y"])
        roi = gpd.GeoDataFrame(geometry=[Polygon([(x - 120, y - 80), (x + 110, y - 90), (x + 130, y + 70), (x - 100, y + 85)])])
        patches = gpd.GeoDataFrame(geometry=[box(x - 120, y - 90, x + 10, y + 85), box(x - 10, y - 90, x + 130, y + 85)])
        with xe._sdata_lock:
            xe._sdata_state.update({"roi": roi, "patches": patches})
        return jsonify(ok=True)
    if what == "cellnest":
        sub = _near(200)
        ids = sub.index.astype(str).to_numpy(); n = len(sub)
        coords = pd.DataFrame({"x": sub.x_centroid.values, "y": sub.y_centroid.values})
        E = 400
        fi = rng.integers(0, n, E)
        # receivers: a near neighbour of the sender
        xy = coords.values
        ti = np.array([np.argsort(np.hypot(*(xy - xy[i]).T))[rng.integers(1, 6)] for i in fi])
        ccc = pd.DataFrame({"from_id": fi, "to_id": ti, "ligand": rng.choice(["LA", "LB"], E), "receptor": rng.choice(["RA", "RB"], E),
                            "attention_score": rng.uniform(0, 1, E)})
        ccc["from_x"] = xy[fi, 0]; ccc["from_y"] = xy[fi, 1]; ccc["to_x"] = xy[ti, 0]; ccc["to_y"] = xy[ti, 1]
        ccc["from_app_id"] = ids[fi]; ccc["to_app_id"] = ids[ti]
        with xe._cellnest_lock:
            xe._cellnest_state.update({"status": "done", "message": "fake", "data": {
                "ccc_df": ccc, "coords": coords, "annot_cols": [], "barcodes": list(ids), "dataset_name": "fake", "folder": "/tmp",
                "lr_freq": pd.DataFrame({"ligand-receptor": ["LA-RA"], "count": [1]})}})
        return jsonify(edges=E)
    if what == "rctd":
        types = ["T cell", "B cell", "Macrophage", "Fibroblast"]
        w = pd.DataFrame(rng.dirichlet([0.4] * 4, len(df)), index=df.index.astype(str), columns=types)
        with xe._annot_lock:
            xe._annot_state["labels_rctd"] = pd.Series(np.array(types)[w.values.argmax(1)], index=w.index)
            xe._annot_state["rctd_weights_labels_rctd"] = w
        return jsonify(ok=True)
    if what == "extra":
        eds = xe.load_xenium_data(request.args["dir"])
        eds["x_offset"] = float(df.x_centroid.max() - eds["df"].x_centroid.min() + 500.0)
        xe.EXTRA_DATASETS.append(eds)
        return jsonify(n=len(eds["df"]), offset=eds["x_offset"])
    if what == "extra_clear":
        xe.EXTRA_DATASETS.clear()
        return jsonify(ok=True)
    return jsonify(error="unknown"), 404

if __name__ == "__main__":
    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    print("Dash is running (fab launcher)", flush=True)
    xe.app.run(debug=False, host="127.0.0.1", port=int(os.environ.get("PORT", 8098)))
