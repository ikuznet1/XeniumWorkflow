"""functions/geo_to_bundle.py on a synthetic GEO-style deposit (no network).

The deposit is written here in the layout GEO uses (prefixed, gzipped files), turned
into a bundle, and the bundle is then opened by the app in a fresh process: the point
of the script is that the result loads.
"""
from __future__ import annotations

import gzip
import json
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import xe_testlib as tl

PREFIX = "GSM0000001_secA_"


@pytest.fixture(scope="module")
def g2b():
    sys.path.insert(0, str(tl.REPO / "functions"))
    import geo_to_bundle
    return geo_to_bundle


def _write_h5(path, mat_features_x_cells, names, types, barcodes):
    import h5py
    m = sp.csc_matrix(mat_features_x_cells)
    with h5py.File(path, "w") as h:
        g = h.create_group("matrix")
        g["data"] = m.data.astype(np.int32)
        g["indices"] = m.indices.astype(np.int64)
        g["indptr"] = m.indptr.astype(np.int64)
        g["shape"] = np.array(m.shape, dtype=np.int32)
        g["barcodes"] = np.array(barcodes, dtype="S")
        f = g.create_group("features")
        f["name"] = np.array(names, dtype="S")
        f["id"] = np.array([f"ENSG{i:05d}" for i in range(len(names))], dtype="S")
        f["feature_type"] = np.array(types, dtype="S")
        f["genome"] = np.array(["GRCh38"] * len(names), dtype="S")


@pytest.fixture
def deposit(tmp_path):
    rng = np.random.default_rng(0)
    genes = [f"GENE{i}" for i in range(30)]
    controls = ["NegControlProbe_00001", "NegControlCodeword_0001", "UnassignedCodeword_0002", "BLANK_0003"]
    types = ["Gene Expression"] * 30 + ["Negative Control Probe", "Negative Control Codeword",
                                         "Unassigned Codeword", "Blank Codeword"]
    barcodes = [f"cell{i:03d}-1" for i in range(40)]
    counts = rng.poisson(1.2, size=(34, 40))
    counts[30:] = rng.poisson(3.0, size=(4, 40))               # controls: must not be counted
    d = tmp_path / "geo"
    d.mkdir()
    _write_h5(d / f"{PREFIX}cell_feature_matrix.h5", counts, genes + controls, types, barcodes)
    _write_h5(d / f"{PREFIX}ambient_corrected_matrix.h5", counts, genes + controls, types, barcodes)
    # per-cell table: shuffled relative to the matrix, one cell the matrix does not have,
    # and two matrix barcodes missing from the table
    table_ids = list(rng.permutation(barcodes[2:])) + ["orphan-1"]
    cells = pd.DataFrame({"cell_id": table_ids,
                          "x_centroid": rng.uniform(0, 500, len(table_ids)),
                          "y_centroid": rng.uniform(0, 300, len(table_ids)),
                          "celltype": rng.choice(["CM", "FB", "EC"], len(table_ids)),
                          "patient": "P1"})
    cells.to_csv(d / f"{PREFIX}cells_stats.csv.gz", index=False, compression="gzip")
    tx = pd.DataFrame({"transcript_id": np.arange(200, dtype=np.uint64),
                       "cell_id": rng.choice(barcodes, 200),
                       "feature_name": rng.choice(genes + controls, 200),
                       "x_location": rng.uniform(0, 500, 200).astype(np.float32),
                       "y_location": rng.uniform(0, 300, 200).astype(np.float32),
                       "qv": rng.uniform(5, 40, 200).astype(np.float32)})
    tx.to_parquet(d / "plain_transcripts.parquet")
    with open(d / "plain_transcripts.parquet", "rb") as src, \
            gzip.open(d / f"{PREFIX}transcripts.parquet.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    (d / "plain_transcripts.parquet").unlink()
    (d / "README.txt").write_text("ignored")
    return {"dir": d, "genes": genes, "barcodes": barcodes, "counts": counts, "cells": cells, "tx": tx}


def test_role_patterns_classify_geo_file_names(g2b):
    def role(name):
        return sorted(r for r, p in g2b._ROLES.items() if p.search(name))
    assert role("GSM1_s_cell_feature_matrix.h5") == ["matrix"]
    assert role("GSM1_s_ambient_corrected_matrix.h5") == ["corrected"]
    assert role("GSM1_s_corrected_cell_feature_matrix.h5") == [], "a corrected matrix is never taken as the counts"
    assert role("GSM1_s_cells_stats.csv.gz") == ["cells"] and role("cells.parquet") == ["cells"]
    assert role("GSM1_s_cell.tsv") == ["cells"]
    assert role("GSM1_s_transcripts.parquet.gz") == ["transcripts"]
    assert role("GSM1_s_morphology.ome.tif.gz") == ["image"] and role("morphology_focus.ome.tiff") == ["image"]
    assert role("README.txt") == [] and role("GSM1_s_cell_boundaries.csv.gz") == []


def test_bundle_contents(g2b, deposit, tmp_path):
    out = tmp_path / "bundle"
    assert g2b.build_bundle(str(deposit["dir"]), str(out), pixel_size=0.25, name="secA") == str(out)
    assert sorted(p.name for p in out.iterdir()) == ["ambient_corrected_matrix.h5", "cell_feature_matrix.h5",
                                                     "cells.parquet", "experiment.xenium",
                                                     "transcripts.parquet"]
    meta = json.loads((out / "experiment.xenium").read_text())
    assert meta["num_cells"] == 39 and meta["pixel_size"] == 0.25 and meta["run_name"] == "secA"
    assert meta["images"] == {}

    cells = pd.read_parquet(out / "cells.parquet")
    src = deposit["cells"]
    assert cells["cell_id"].tolist() == src["cell_id"].tolist(), "table order is kept"
    assert cells["celltype"].tolist() == src["celltype"].tolist() and (cells["patient"] == "P1").all()
    np.testing.assert_allclose(cells["x_centroid"], src["x_centroid"])
    # transcript_counts = gene features only, matched by barcode; the orphan cell has none
    gene_totals = pd.Series(deposit["counts"][:30].sum(axis=0), index=deposit["barcodes"])
    want = gene_totals.reindex(cells["cell_id"]).fillna(0).to_numpy()
    np.testing.assert_array_equal(cells["transcript_counts"].to_numpy(), want)
    assert cells.loc[cells["cell_id"] == "orphan-1", "transcript_counts"].item() == 0
    all_totals = pd.Series(deposit["counts"].sum(axis=0), index=deposit["barcodes"])
    assert not np.array_equal(want[:-1], all_totals.reindex(cells["cell_id"][:-1]).to_numpy()), \
        "the synthetic controls would change the totals if they were counted"
    np.testing.assert_array_equal(cells["total_counts"], cells["transcript_counts"])

    pd.testing.assert_frame_equal(pd.read_parquet(out / "transcripts.parquet"), deposit["tx"])
    assert (out / "cell_feature_matrix.h5").read_bytes() == \
        (deposit["dir"] / f"{PREFIX}cell_feature_matrix.h5").read_bytes()


def test_existing_count_columns_are_kept(g2b, deposit, tmp_path):
    d = deposit["dir"]
    cells = deposit["cells"].assign(transcript_counts=7, total_counts=9)
    cells.to_csv(d / f"{PREFIX}cells_stats.csv.gz", index=False, compression="gzip")
    out = tmp_path / "b2"
    g2b.build_bundle(str(d), str(out))
    got = pd.read_parquet(out / "cells.parquet")
    assert (got["transcript_counts"] == 7).all() and (got["total_counts"] == 9).all()
    assert json.loads((out / "experiment.xenium").read_text())["pixel_size"] == 0.2125


@pytest.mark.parametrize("damage,message", [
    ("no_matrix", "No matrix file"), ("no_cells", "No cells file"), ("dup_ids", "not unique"),
    ("no_xy", "must have columns"), ("two_matrices", "Two files look like the matrix"),
    ("no_overlap", "No cell_id"),
])
def test_unusable_deposits_are_refused_with_a_reason(g2b, deposit, tmp_path, damage, message):
    d = deposit["dir"]
    cells_path = d / f"{PREFIX}cells_stats.csv.gz"
    if damage == "no_matrix":
        (d / f"{PREFIX}cell_feature_matrix.h5").unlink()
    elif damage == "no_cells":
        cells_path.unlink()
    elif damage == "dup_ids":
        c = deposit["cells"].copy()
        c.loc[1, "cell_id"] = c.loc[0, "cell_id"]
        c.to_csv(cells_path, index=False, compression="gzip")
    elif damage == "no_xy":
        deposit["cells"].drop(columns="x_centroid").to_csv(cells_path, index=False, compression="gzip")
    elif damage == "two_matrices":
        shutil.copy(d / f"{PREFIX}cell_feature_matrix.h5", d / "GSM0000002_secB_cell_feature_matrix.h5")
    elif damage == "no_overlap":
        deposit["cells"].assign(cell_id=lambda t: "zz" + t["cell_id"]).to_csv(cells_path, index=False,
                                                                             compression="gzip")
    with pytest.raises(SystemExit) as exc:
        g2b.build_bundle(str(d), str(tmp_path / "out"))
    assert message in str(exc.value)
    assert not (tmp_path / "out" / "experiment.xenium").exists(), "no half-written bundle that looks loadable"


def test_download_refuses_anything_but_a_gsm_accession(g2b, tmp_path):
    # sys.exit happens before any network access
    for bad in ("../../etc", "GSE12345", "GSM12/../x", "", "GSM"):
        with pytest.raises(SystemExit, match="Not a GEO sample accession"):
            g2b._download_gsm(bad, str(tmp_path / "dl"), False)
    assert not (tmp_path / "dl").exists()


def test_cli_and_app_load_the_assembled_bundle(deposit, tmp_path):
    """End to end: the command line builds the bundle and xenium_explorer opens it."""
    import subprocess
    out = tmp_path / "bundle"
    proc = subprocess.run([sys.executable, str(tl.REPO / "functions" / "geo_to_bundle.py"),
                           "--input_dir", str(deposit["dir"]), "--out_dir", str(out), "--name", "secA"],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "39 cells, 30 genes" in proc.stdout
    cache = tmp_path / "cache"
    cache.mkdir()
    res = tl.run_in_fresh_app('''
        import numpy as np
        D = xe.DATA
        RESULT["ids"] = list(D["df"].index.astype(str))
        RESULT["genes"] = list(D["gene_names"])
        RESULT["methods"] = list(D["cluster_methods"])
        RESULT["row_sums"] = np.asarray(D["expr"].sum(axis=1)).ravel().tolist()
        RESULT["tc"] = D["df"]["transcript_counts"].astype(float).tolist()
        RESULT["celltype"] = D["df"]["celltype"].astype(str).tolist() if "celltype" in D["df"].columns else None
        RESULT["bounds"] = [len(D["cell_bounds"]), len(D["nucleus_bounds"])]
        fig = xe.make_spatial_fig("transcript_counts", None, D["gene_names"][0], 3, 0.8, [], None)
        RESULT["drawn"] = sum(len(t.x) for t in fig.data)
        RESULT["umap_traces"] = len(xe.make_umap_fig("transcript_counts", None, D["gene_names"][0], 3, 0.8).data)
        c = xe.app.server.test_client()
        m = c.get("/viewer/meta").get_json()
        RESULT["meta"] = {k: m[k] for k in ("has_boundaries", "has_transcripts", "images")}
        RESULT["viewer_n"] = len(c.get("/viewer/cells.bin").data) > 0
        import json, plotly
        RESULT["layout_ok"] = len(json.dumps(xe.app.layout, cls=plotly.utils.PlotlyJSONEncoder)) > 1000
    ''', out, cache)
    cells = deposit["cells"]
    assert res["genes"] == deposit["genes"], "control features of the deposit are not genes in the app"
    assert res["methods"] == [] and res["bounds"] == [0, 0]
    assert res["meta"] == {"has_boundaries": False, "has_transcripts": False, "images": []}
    assert res["layout_ok"] and res["viewer_n"]
    # cells: the per-cell table decides which cells exist (a cell without a matrix column
    # is kept with zero counts); every cell is drawn
    assert res["ids"] == cells["cell_id"].tolist()
    assert res["drawn"] == len(res["ids"]) == 39
    by_id = dict(zip(res["ids"], res["row_sums"]))
    gene_totals = pd.Series(deposit["counts"][:30].sum(axis=0), index=deposit["barcodes"])
    for cid, total in by_id.items():
        assert total == gene_totals.get(cid, 0), f"counts of {cid} are not its own"
    assert by_id["orphan-1"] == 0
    assert res["tc"] == [by_id[c] for c in res["ids"]]
    # extra columns of the deposit reach the app's cell table
    assert res["celltype"] == cells["celltype"].tolist()
