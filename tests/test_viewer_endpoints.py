"""The WebGL viewer's HTTP endpoints, through the Flask test client.

Three kinds of assertion: the payload decodes under the documented wire format
(tl.unpack checks framing), its content equals values read independently from the
bundle, and it agrees cell-for-cell with the Plotly figure of the same colour mode
(the two renderers must show the same thing).
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import xe_testlib as tl


def colours(client, query, prefix=""):
    """Per-cell colour (categorical: hex string, continuous: float) from colors.bin."""
    h, a = tl.get_bin(client, f"{prefix}/viewer/colors.bin?{query}")
    if h["kind"] == "categorical":
        assert len(h["categories"]) == len(h["colors"])
        codes = a["codes"].astype(int)
        assert a["codes"].dtype == np.uint16
        assert ((codes < len(h["colors"])) | (codes == 65535)).all(), "codes index the legend or are 65535"
        lut = np.array(h["colors"] + ["<none>"], dtype=object)
        names = np.array(h["categories"] + ["<none>"], dtype=object)
        idx = np.where(codes == 65535, len(h["colors"]), codes)
        return h, lut[idx], names[idx]
    assert a["values"].dtype == np.float32
    finite = a["values"][np.isfinite(a["values"])]
    if len(finite):
        assert h["min"] == pytest.approx(float(finite.min())) and h["max"] == pytest.approx(float(finite.max()))
    return h, a["values"], None


@pytest.fixture
def method(D):
    return D["cluster_methods"][0] if D["cluster_methods"] else ""


# ── meta / cells ─────────────────────────────────────────────────────────────
def test_meta_describes_the_bundle(xe, client, raw, bundle_dir):
    m = client.get("/viewer/meta").get_json()
    c = raw.cells
    assert m["bounds"] == pytest.approx([c.x_centroid.min(), c.y_centroid.min(),
                                         c.x_centroid.max(), c.y_centroid.max()])
    assert m["pixel_size"] == pytest.approx(raw.experiment.get("pixel_size", 0.2125))
    assert m["sources"] == ["xenium"]
    assert m["dataset"] == bundle_dir.name
    assert m["has_transcripts"] == raw.has("transcripts.zarr.zip")
    assert m["has_boundaries"] == (raw.has("cells.zarr.zip") or raw.has("cell_boundaries.parquet"))
    files = xe._viewer_provider.image_files()
    assert [i["index"] for i in m["images"]] == list(range(len(files)))
    for entry, path in zip(m["images"], files):
        assert entry["name"] == os.path.basename(path) and entry["v"]
        assert os.path.realpath(path).startswith(os.path.realpath(raw.path) + os.sep) or \
            os.path.realpath(path).startswith(os.path.realpath(bundle_dir) + os.sep)


def test_cells_bin_is_the_centroids_in_table_order(client, raw):
    h, a = tl.get_bin(client, "/viewer/cells.bin")
    c = raw.cells
    assert h["n"] == len(c) and h["samples"] == 1
    xy = a["xy"].reshape(-1, 2)
    np.testing.assert_array_equal(xy[:, 0], c["x_centroid"].to_numpy(dtype=np.float32))
    np.testing.assert_array_equal(xy[:, 1], c["y_centroid"].to_numpy(dtype=np.float32))
    assert h["bounds"] == client.get("/viewer/meta").get_json()["bounds"]


def test_cell_lookup_by_row(client, raw):
    c = raw.cells
    for i in (0, len(c) // 2, len(c) - 1):
        got = client.get(f"/viewer/cell/{i}").get_json()
        assert got["id"] == c.index[i] and got["index"] == i
        assert got["x"] == pytest.approx(c["x_centroid"].iloc[i]) and got["y"] == pytest.approx(c["y_centroid"].iloc[i])
    assert client.get(f"/viewer/cell/{len(c)}").get_json() == {}
    for bad in ("-1", "abc", "1.5"):          # not the int route: Dash's catch-all page answers
        r = client.get(f"/viewer/cell/{bad}")
        assert r.status_code == 404 or (r.mimetype == "text/html" and not r.is_json)


# ── colours: against the raw files ───────────────────────────────────────────
def test_colors_gene_is_log1p_of_the_h5_counts(client, raw, expressed_gene):
    counts, names, _ = raw.gene_counts
    col = np.asarray(counts[:, names.index(expressed_gene)].todense()).ravel()
    h, vals, _ = colours(client, f"mode=gene&gene={expressed_gene}")
    assert h["kind"] == "continuous" and expressed_gene in h["label"]
    np.testing.assert_allclose(vals, np.log1p(col.astype(np.float32)), rtol=1e-6)


@pytest.mark.parametrize("mode", ["transcript_counts", "cell_area", "nucleus_area", "total_counts"])
def test_colors_qc_metrics_come_from_cells_parquet(client, raw, mode):
    if mode not in raw.cells.columns:
        pytest.skip(f"bundle has no {mode}")
    h, vals, _ = colours(client, f"mode={mode}")
    want = raw.cells[mode].to_numpy(dtype=float)
    want = np.log1p(want) if mode == "transcript_counts" else want
    np.testing.assert_allclose(vals, want.astype(np.float32), rtol=1e-6)
    assert ("log1p" in h["label"]) == (mode == "transcript_counts")


def test_colors_cluster_codes_decode_to_the_analysis_clusters(client, raw, D):
    checked = 0
    for method in D["cluster_methods"][:3]:
        truth = raw.clusters(method)
        if truth is None:
            continue
        h, _cols, names = colours(client, f"mode=cluster&method={method}")
        want = truth.reindex(raw.cells.index).fillna(0).astype(int)
        assert [str(n) for n in names] == [f"Cluster {k}" for k in want] or \
            [str(n) for n in names] == [str(k) for k in want]
        assert len(set(h["colors"])) == len(h["colors"]) or len(h["colors"]) > 50
        checked += 1
    if not checked:
        pytest.skip("bundle has no clustering")


def test_colors_fallbacks(client, raw):
    tc = np.log1p(raw.cells["transcript_counts"].to_numpy(dtype=float)).astype(np.float32)
    for query in ("mode=__bogus__", "", "mode=niche", "mode=roi_none"):
        h, vals, _ = colours(client, query)
        assert h["kind"] == "continuous"
        np.testing.assert_allclose(vals, tc, rtol=1e-6, err_msg=query)
    # a gene this dataset does not have (e.g. left selected after another sample was
    # loaded): either "no data" or the same fallback as the Plotly callback, but never
    # numbers presented as that gene's expression
    h, vals, _ = colours(client, "mode=gene&gene=__no_such_gene__")
    assert len(vals) == len(raw.cells)
    if "unavailable" in h["label"]:
        assert np.isnan(vals).all()
    else:
        assert "__no_such_gene__" not in h["label"] and "Transcript Counts" in h["label"]
        np.testing.assert_allclose(vals, tc, rtol=1e-6)


def test_colors_corrected_counts(xe, D, client, raw, expressed_gene):
    counts, names, _ = raw.gene_counts
    col = np.asarray(counts[:, names.index(expressed_gene)].todense()).ravel().astype(np.float32)
    # no corrected counts: corrected=1 falls back to the original ones
    _, vals, _ = colours(client, f"mode=gene&gene={expressed_gene}&corrected=1")
    np.testing.assert_allclose(vals, np.log1p(col), rtol=1e-6)
    D["split_corrected_expr"] = (D["expr"] * 0.25).tocsr()
    _, vals, _ = colours(client, f"mode=gene&gene={expressed_gene}&corrected=1")
    np.testing.assert_allclose(vals, np.log1p(0.25 * col), rtol=1e-6)
    _, vals, _ = colours(client, f"mode=gene&gene={expressed_gene}")
    np.testing.assert_allclose(vals, np.log1p(col), rtol=1e-6)


# ── colours: WebGL == Plotly ─────────────────────────────────────────────────
def _plotly(xe, mode, method, gene, **kw):
    return tl.fig_cells(xe.make_spatial_fig(mode, method or None, gene, 3, 0.8, [], None, **kw))


def test_cell_type_without_labels_shows_one_placeholder_group_in_both_renderers(xe, client, raw, method,
                                                                                 expressed_gene):
    fig = xe.make_spatial_fig("cell_type:celltypist", method or None, expressed_gene, 3, 0.8, [], None)
    (trace,) = [t for t in fig.data if t.mode == "markers"]
    h, cols, names = colours(client, "mode=cell_type:celltypist")
    assert len(cols) == len(raw.cells)
    if h["kind"] == "categorical":
        assert set(names) == {trace.name} == {"No annotation"} and set(cols) == {trace.marker.color}
    else:       # older behaviour: fall back to transcript counts
        np.testing.assert_allclose(cols, np.log1p(raw.cells["transcript_counts"].to_numpy(float)), rtol=1e-6)


@pytest.mark.parametrize("mode", ["cluster", "gene", "transcript_counts", "cell_area", "nucleus_area",
                                  "cell_type:celltypist", "niche", "roi_zone"])
def test_viewer_colours_equal_the_plotly_figure(xe, D, client, method, expressed_gene, labels, mode):
    df = D["df"]
    if mode == "cluster" and not method:
        pytest.skip("bundle has no clustering")
    if mode in ("cell_area", "nucleus_area") and mode not in df.columns:
        pytest.skip(f"bundle has no {mode}")
    df["niche"] = (np.arange(len(df)) * 5 % 4).astype(int)
    roi = pd.Series(pd.NA, index=df.index, dtype=object)
    roi.iloc[: max(2, len(df) // 3)] = "a"
    roi.iloc[-max(2, len(df) // 4):] = "b"
    df["roi_zone"] = roi
    fig = _plotly(xe, mode, method, expressed_gene)
    h, vals, names = colours(client, f"mode={mode}&method={method}&gene={expressed_gene}")
    ids = df.index.astype(str)
    assert len(vals) == len(ids)
    plot = fig.loc[ids]
    if h["kind"] == "categorical":
        assert (plot["color"].to_numpy() == vals).all(), "same colour for every cell"
        if mode != "roi_zone":       # the ROI figure is one trace with per-cell colours
            assert (plot["trace"].to_numpy() == names).all(), "same legend entry for every cell"
    else:
        np.testing.assert_allclose(plot["color"].to_numpy(dtype=float), vals, rtol=1e-4, atol=1e-5)


# ── boundaries ───────────────────────────────────────────────────────────────
def _box(raw, half):
    c = raw.cells
    mid = c.iloc[((c.x_centroid - c.x_centroid.median()) ** 2 + (c.y_centroid - c.y_centroid.median()) ** 2).argmin()]
    return (float(mid.x_centroid) - half, float(mid.y_centroid) - half,
            float(mid.x_centroid) + half, float(mid.y_centroid) + half)


def _q(box):
    return "x0={}&y0={}&x1={}&y1={}".format(*box)


@pytest.mark.parametrize("kind", ["cell", "nucleus"])
def test_boundaries_bin_returns_the_polygons_of_the_right_cells(client, raw, kind):
    truth = raw.boundaries(kind)
    if not truth:
        pytest.skip(f"bundle has no {kind} boundaries")
    box = _box(raw, 40.0)
    h, a = tl.get_bin(client, f"/viewer/boundaries.bin?kind={kind}&{_q(box)}")
    assert h["available"] and not h["truncated"] and h["n"] > 0
    got = tl.rings(a)
    assert len(got) == h["n"]
    polygons = raw.rings(kind)
    for row, verts in got:
        cid = raw.cells.index[row]
        assert tl.point_set(verts) in {tl.point_set(w) for w in polygons[cid]}, f"{cid}: not its own outline"
    # nothing that lies in the box is missing
    inside = {cid for cid, v in truth.items()
              if v[:, 0].min() >= box[0] and v[:, 0].max() <= box[2]
              and v[:, 1].min() >= box[1] and v[:, 1].max() <= box[3]}
    assert inside <= {raw.cells.index[row] for row, _ in got}


def test_boundaries_in_memory_path_equals_zip_path(xe, D, client, raw):
    """With a subset active the provider serves polygons from memory instead of from
    cells.zarr.zip; both must describe the same rings, and row indices must refer to
    the frame the browser has (cells.bin)."""
    if not (raw.has("cells.zarr.zip") and raw.boundaries("cell")):
        pytest.skip("needs cells.zarr.zip and boundary parquet")
    box = _box(raw, 60.0)
    url = f"/viewer/boundaries.bin?kind=cell&{_q(box)}"
    assert xe._viewer_provider.boundaries("xenium", "cell", *box) is None, "full table: zip path"
    def by_cell(arrays, ids):
        out = {}
        for r, v in tl.rings(arrays):
            out[ids[r]] = out.get(ids[r], frozenset()) | tl.point_set(v)
        return out

    _, a = tl.get_bin(client, url)
    full = by_cell(a, raw.cells.index)

    keep = list(D["df"].index[::2])
    xe.subset(cell_ids=keep)
    assert xe._viewer_provider.boundaries("xenium", "cell", *box) is not None, "subset: in-memory path"
    h, a = tl.get_bin(client, url)
    sub_ids = D["df"].index.astype(str)
    got = by_cell(a, sub_ids)
    # The two paths select differently at the rim (zip: polygon bbox meets the box;
    # memory: centroid within a margin of it), so compare what both must return: every
    # subset cell whose polygon lies inside the box, with the same outline.
    truth = raw.boundaries("cell")
    inside = {cid for cid in set(sub_ids) & set(full)
              if truth[cid][:, 0].min() >= box[0] and truth[cid][:, 0].max() <= box[2]
              and truth[cid][:, 1].min() >= box[1] and truth[cid][:, 1].max() <= box[3]}
    assert inside and inside <= set(got) <= set(sub_ids)
    assert not (set(got) - set(sub_ids)), "no outline for a cell outside the subset"
    common = set(got) & set(full)
    assert all(got[cid] == full[cid] for cid in common)
    assert all(got[cid] == tl.point_set(truth[cid]) for cid in got)
    xe.unsubset()
    assert xe._viewer_provider.boundaries("xenium", "cell", *box) is None, "zip path again"


@pytest.mark.parametrize("query", ["x0=a&y0=0&x1=1&y1=1", "x0=5&y0=0&x1=1&y1=1", "x0=0&y0=5&x1=1&y1=1",
                                   "x0=0&y0=0&x1=nan&y1=1", "x0=0&y0=0&x1=inf&y1=1", "x0=0&y0=0&x1=1", ""])
def test_bad_boxes_are_rejected(client, query):
    assert client.get(f"/viewer/boundaries.bin?{query}").status_code == 400
    assert client.get(f"/viewer/transcripts.bin?genes=A&{query}").status_code == 400


def test_unknown_boundary_kind_is_unavailable_not_an_error(client):
    h, a = tl.get_bin(client, "/viewer/boundaries.bin?kind=membrane&x0=0&y0=0&x1=10&y1=10")
    assert h["n"] == 0 and h["available"] is False and a == {}


# ── transcripts ──────────────────────────────────────────────────────────────
def test_transcripts_bin_equals_the_parquet_rows(client, raw):
    if not (raw.has("transcripts.zarr.zip") and raw.has("transcripts.parquet")):
        pytest.skip("bundle has no transcripts")
    box = _box(raw, 25.0)
    tx = raw.transcripts(bbox=box, columns=["feature_name", "x_location", "y_location", "qv", "is_gene",
                                            "codeword_category"])
    genes = list(tl.gene_tx(tx)["feature_name"].value_counts().index[:5])
    assert genes
    h, a = tl.get_bin(client, f"/viewer/transcripts.bin?genes={','.join(genes)}&{_q(box)}")
    assert h["available"] and h["level"] == 0 and h["filtered"] and not h["clustered"]
    want = tx[tx["feature_name"].isin(genes) & (tx["qv"] >= 20)]
    got = pd.DataFrame({"g": np.asarray(genes)[a["gene"]], "x": a["xy"][0::2], "y": a["xy"][1::2]})
    key = ["g", "x", "y"]
    want = pd.DataFrame({"g": want["feature_name"].values, "x": want["x_location"].values.astype(np.float32),
                         "y": want["y_location"].values.astype(np.float32)})
    pd.testing.assert_frame_equal(got.sort_values(key).reset_index(drop=True),
                                  want.sort_values(key).reset_index(drop=True))
    assert h["n"] == len(want) and "weight" not in a
    # qv is a request parameter; rubbish falls back to the default of 20
    h40, _ = tl.get_bin(client, f"/viewer/transcripts.bin?genes={','.join(genes)}&qv=40&{_q(box)}")
    assert h40["n"] == int((tx["feature_name"].isin(genes) & (tx["qv"] >= 40)).sum())
    hbad, _ = tl.get_bin(client, f"/viewer/transcripts.bin?genes={','.join(genes)}&qv=abc&{_q(box)}")
    assert hbad["n"] == h["n"]


def test_transcripts_bin_huge_box_answers_quickly(client, raw, expressed_gene):
    if not raw.has("transcripts.zarr.zip"):
        pytest.skip("bundle has no transcripts.zarr.zip")
    tl.get_bin(client, f"/viewer/transcripts.bin?genes={expressed_gene}&x0=0&y0=0&x1=1&y1=1")     # warm-up
    for box in ("x0=0&y0=0&x1=1e300&y1=1e300", "x0=-1e300&y0=-1e300&x1=1e300&y1=1e300"):
        t0 = time.time()
        h, a = tl.get_bin(client, f"/viewer/transcripts.bin?genes={expressed_gene}&{box}&weights=1")
        assert time.time() - t0 < 20
        assert h["available"] and h["clustered"] == (h["level"] > 0) and h["filtered"] == (h["level"] == 0)
        assert h["n"] <= 400_000 and (h["n"] == 0 or len(a["weight"]) == h["n"] == len(a["gene"]))
        if h["n"]:
            assert np.isfinite(a["xy"]).all() and (a["gene"] == 0).all()


def test_transcripts_bin_without_genes_and_gene_cap(client, raw, D):
    if not raw.has("transcripts.zarr.zip"):
        pytest.skip("bundle has no transcripts.zarr.zip")
    h, a = tl.get_bin(client, "/viewer/transcripts.bin?x0=0&y0=0&x1=1e5&y1=1e5")
    assert h == {"n": 0, "available": True, "arrays": []} and a == {}
    many = ",".join(D["gene_names"][:200])
    h, a = tl.get_bin(client, f"/viewer/transcripts.bin?genes={many}&x0=0&y0=0&x1=1e5&y1=1e5")
    if h["n"]:
        assert int(a["gene"].max()) < 64, "at most 64 genes per request"


# ── images ───────────────────────────────────────────────────────────────────
@pytest.fixture
def image(xe, client):
    files = xe._viewer_provider.image_files()
    if not files:
        pytest.skip("bundle has no morphology focus image")
    meta = client.get("/viewer/meta").get_json()["images"][0]
    return files[0], meta["v"]


def test_image_range_requests_return_the_file_bytes(client, image):
    path, _ = image
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(4096)
        f.seek(max(0, size - 100))
        tail = f.read()
    r = client.get("/viewer/image/0", headers={"Range": "bytes=0-4095"})
    assert r.status_code == 206 and r.data == head
    assert r.headers["Content-Range"] == f"bytes 0-{len(head) - 1}/{size}"
    assert r.headers["Accept-Ranges"] == "bytes" and r.mimetype == "image/tiff"
    r = client.get("/viewer/image/0", headers={"Range": "bytes=-100"})
    assert r.status_code == 206 and r.data == tail
    r = client.get("/viewer/image/0", headers={"Range": f"bytes={size + 10}-{size + 20}"})
    assert r.status_code == 416
    assert head[:2] in (b"II", b"MM"), "served file is a TIFF"


def test_image_cache_headers(client, image):
    _, token = image
    bare = client.get("/viewer/image/0", headers={"Range": "bytes=0-99"})
    assert bare.headers["Cache-Control"] == "no-cache" and bare.headers.get("ETag")
    assert client.get("/viewer/image/0", headers={"If-None-Match": bare.headers["ETag"]}).status_code == 304
    pinned = client.get(f"/viewer/image/0?v={token}", headers={"Range": "bytes=0-99"})
    assert pinned.status_code == 206 and pinned.data == bare.data
    cc = pinned.headers["Cache-Control"]
    assert "max-age=3600" in cc and "no-cache" not in cc and "no-store" not in cc
    assert client.get("/viewer/image/0?v=deadbeef").status_code == 404, "a stale token must not be served"
    assert client.get("/viewer/image/0?v=").status_code == 404


def test_image_index_out_of_range(xe, client):
    n = len(xe._viewer_provider.image_files())
    assert client.get(f"/viewer/image/{n}").status_code == 404
    assert client.get("/viewer/image/999").status_code == 404
    r = client.get("/viewer/image/-1")
    assert r.status_code == 404 or r.mimetype == "text/html", "never an image"


# ── path traversal ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("url", ["/viewer/image/../../etc/hosts", "/viewer/image/0/../../../etc/hosts",
                                 "/viewer/image/%2e%2e%2f%2e%2e%2fetc%2fhosts", "/viewer/image/0%2f..%2f..",
                                 "/viewer/extra/..%2f..%2fetc%2fhosts.bin", "/viewer/extra/%2e%2e.bin",
                                 "/viewer/cell/..%2f0", "/assets/../xenium_explorer.py",
                                 "/assets/..%2fxenium_explorer.py", "/assets/%2e%2e/requirements.txt"])
def test_routes_do_not_serve_paths_outside_their_scope(client, url):
    r = client.get(url)
    # an unmatched URL falls through to Dash's catch-all route, which answers 200 with
    # the app's index page: acceptable, as long as no file content is ever returned
    is_index_page = r.mimetype == "text/html" and b"_dash-config" in r.data
    assert r.status_code in (400, 404) or is_index_page, (url, r.status_code, r.mimetype)
    assert b"import dash" not in r.data and b"localhost" not in r.data and b"numpy" not in r.data
    assert r.mimetype not in ("image/tiff", "application/octet-stream", "text/x-python", "text/plain")


def test_image_path_from_metadata_must_stay_inside_the_bundle(xe, D, tmp_path):
    """experiment.xenium names the focus image; a hostile bundle must not be able to make
    the viewer serve an arbitrary file."""
    bundle = tmp_path / "bundle"
    (bundle / "sub").mkdir(parents=True)
    (bundle / "sub" / "m.ome.tif").write_bytes(b"II*\0")
    (bundle / "notes.txt").write_text("not an image")
    secret = tmp_path / "outside_secret.tif"
    secret.write_bytes(b"II*\0secret")
    images = D.setdefault("metadata", {}).setdefault("images", {})
    saved = images.get("morphology_focus_filepath")
    try:
        for evil in ("/etc/hosts", "../outside_secret.tif", str(secret), "../../../../../../etc/hosts",
                     "sub/../../outside_secret.tif", "notes.txt", "sub", "", None, 5):
            images["morphology_focus_filepath"] = evil
            assert xe._morph_focus_path(str(bundle), 0) is None, f"{evil!r} accepted"
        images["morphology_focus_filepath"] = "sub/m.ome.tif"
        assert xe._morph_focus_path(str(bundle), 0) == os.path.join(str(bundle), "sub/m.ome.tif")
        assert xe._morph_focus_path(str(bundle), 1) is None, "the metadata path is file 0 only"
    finally:
        if saved is None:
            images.pop("morphology_focus_filepath", None)
        else:
            images["morphology_focus_filepath"] = saved
    # and the provider only lists files that exist
    for f in xe._viewer_provider.image_files():
        assert os.path.isfile(f)


# ── overlays / extras ────────────────────────────────────────────────────────
def test_overlays_and_extras_default_to_empty(client):
    assert client.get("/viewer/overlays").get_json() == {"lines": [], "polygons": []}
    assert client.get("/viewer/extra/__nope__.bin").status_code == 404


# ── UMAP provider ────────────────────────────────────────────────────────────
def test_umap_viewer_serves_the_embedding(client, raw):
    truth = raw.umap
    if truth is None:
        pytest.skip("bundle has no UMAP projection")
    m = client.get("/umap/viewer/meta").get_json()
    assert m["embedding"] == "umap" and not m["has_transcripts"] and not m["has_boundaries"] and m["images"] == []
    h, a = tl.get_bin(client, "/umap/viewer/cells.bin")
    assert h["n"] == len(raw.cells) and h["n_embedded"] == len(truth)
    xy = pd.DataFrame(a["xy"].reshape(-1, 2), index=raw.cells.index, columns=["x", "y"])
    np.testing.assert_allclose(xy.loc[truth.index, "x"], truth["UMAP-1"].astype(np.float32), rtol=1e-6)
    np.testing.assert_allclose(xy.loc[truth.index, "y"], -truth["UMAP-2"].astype(np.float32), rtol=1e-6)
    rest = xy.drop(index=truth.index)
    assert not np.isfinite(rest.to_numpy()).any(), "cells without a projection must not be drawn"
    # one colour value per cell, same order as the spatial viewer
    hs, vs, _ = colours(client, "mode=transcript_counts")
    hu, vu, _ = colours(client, "mode=transcript_counts", prefix="/umap")
    np.testing.assert_array_equal(vs, vu)
    hb, ab = tl.get_bin(client, "/umap/viewer/boundaries.bin?x0=-100&y0=-100&x1=100&y1=100")
    assert hb["n"] == 0
    ht, _ = tl.get_bin(client, "/umap/viewer/transcripts.bin?genes=A&x0=-100&y0=-100&x1=100&y1=100")
    assert ht["n"] == 0


# ── a resegmentation result as source ────────────────────────────────────────
@pytest.fixture
def fake_proseg(xe, D):
    """A synthetic Proseg result installed as the active run (no tool involved)."""
    rng = np.random.default_rng(0)
    n = 300
    c = D["df"]
    ids = np.arange(1, n + 1)
    cdf = pd.DataFrame({
        "x_centroid": rng.uniform(c.x_centroid.min(), c.x_centroid.max(), n),
        "y_centroid": rng.uniform(c.y_centroid.min(), c.y_centroid.max(), n),
        "transcript_counts": rng.integers(1, 500, n).astype(float),
        "cell_area": rng.uniform(10, 100, n)}, index=pd.Index(ids, name="cell"))
    cdf["cluster_kmeans_10"] = rng.integers(1, 13, n)
    split = pd.array(rng.integers(1, 13, n).astype(str), dtype="string")
    split[:5] = pd.NA
    cdf["cluster_split_10"] = split
    bounds = {}
    for i, (x, y) in zip(ids, cdf[["x_centroid", "y_centroid"]].values):
        sq = (np.array([x - 2, x + 2, x + 2, x - 2, x - 2]), np.array([y - 2, y - 2, y + 2, y + 2, y - 2]))
        if i % 7 == 0:          # multi-ring cell
            sq = (np.r_[sq[0], np.nan, sq[0] + 5], np.r_[sq[1], np.nan, sq[1]])
        bounds[int(i)] = sq
    n_genes = len(xe._panel_gene_names())
    expr = (sp.random(n, n_genes, density=min(0.2, 2000 / n_genes), format="csr", random_state=0,
                      dtype=np.float32) * 10).tocsr()
    res = {"cells_df": cdf, "cell_bounds": bounds, "out_dir": "/nonexistent/fake_proseg", "expr": expr,
           "source": "proseg", "split_corrected_expr": (expr * 0.5).tocsr(),
           "split_corrected_imputed_genes": [], "sdata_path": None}
    with xe._proseg_lock:
        xe._proseg_state.update({"status": "done", "result": res})
    gene = D["gene_names"][int(np.asarray(expr.sum(axis=0)).ravel().argmax())]
    return res, "proseg:abc12345", gene


def test_reseg_source_cells_and_meta(client, fake_proseg, raw):
    res, src, _ = fake_proseg
    assert client.get("/viewer/meta").get_json()["sources"] == ["xenium", "proseg"]
    h, a = tl.get_bin(client, f"/viewer/cells.bin?source={src}")
    cdf = res["cells_df"]
    assert h["n"] == len(cdf)
    np.testing.assert_array_equal(a["xy"][0::2], cdf.x_centroid.values.astype(np.float32))
    np.testing.assert_array_equal(a["xy"][1::2], cdf.y_centroid.values.astype(np.float32))
    hx, _ = tl.get_bin(client, "/viewer/cells.bin?source=xenium")
    assert hx["n"] == len(raw.cells) and hx["token"] != h["token"]
    assert client.get(f"/viewer/cell/0?source={src}").get_json()["id"] == str(cdf.index[0])


@pytest.mark.parametrize("mode,corrected", [("cluster", False), ("cluster", True), ("gene", False),
                                            ("gene", True), ("transcript_counts", False),
                                            ("cell_area", False), ("nucleus_area", False),
                                            ("cell_type:celltypist", False), ("niche", False),
                                            ("roi_t", False)])
def test_reseg_source_colours_equal_the_plotly_figure(xe, client, fake_proseg, mode, corrected):
    res, src, gene = fake_proseg
    cdf = res["cells_df"]
    rng = np.random.default_rng(1)
    with xe._annot_lock:
        xe._annot_state[xe._labels_key_for_method("celltypist", res)] = pd.Series(
            rng.choice(["A", "B", "C"], len(cdf)), index=cdf.index.astype(str))
    cdf["niche"] = rng.integers(0, 5, len(cdf))
    cdf["roi_t"] = rng.choice(np.array(["a", None], dtype=object), len(cdf))
    fig = tl.fig_cells(xe.make_spatial_fig(mode, None, gene, 2, 0.8, [], None, proseg_active=True,
                                           use_corrected=corrected))
    h, vals, names = colours(client, f"source={src}&mode={mode}&gene={gene}&corrected={int(corrected)}")
    plot = fig.loc[cdf.index.astype(str)]
    if h["kind"] == "categorical":
        assert (plot["color"].to_numpy() == vals).all()
    else:
        np.testing.assert_allclose(plot["color"].to_numpy(dtype=float), vals, rtol=1e-4, atol=1e-5,
                                   equal_nan=True)
    if mode == "gene":
        col = np.asarray(res["expr"][:, xe.DATA["gene_name_to_idx"][gene]].todense()).ravel()
        np.testing.assert_allclose(vals, np.log1p(col * (0.5 if corrected else 1.0)), rtol=1e-5)
        assert vals.max() > 0
    if mode == "cluster":
        col = "cluster_split_10" if corrected else "cluster_kmeans_10"
        assert [n.replace("Cluster ", "") for n in names] == list(cdf[col].astype(str))


def test_reseg_source_boundaries(xe, client, fake_proseg, raw):
    res, src, _ = fake_proseg
    cdf = res["cells_df"]
    box = (cdf.x_centroid.min() - 10, cdf.y_centroid.min() - 10, cdf.x_centroid.max() + 10, cdf.y_centroid.max() + 10)
    for keys in ("int", "str"):
        if keys == "str":
            res["cell_bounds"] = {str(k): v for k, v in res["cell_bounds"].items()}
            xe._viewer_provider.forget()
        h, a = tl.get_bin(client, f"/viewer/boundaries.bin?source={src}&kind=proseg&{_q(box)}")
        got = tl.rings(a)
        n_rings = sum(2 if i % 7 == 0 else 1 for i in cdf.index)
        assert h["n"] == len(got) == n_rings, keys
        assert {r for r, _ in got} == set(range(len(cdf))), "every cell has an outline; rows index cells_df"
        for row, verts in got[:50]:
            x, y = cdf.x_centroid.iloc[row], cdf.y_centroid.iloc[row]
            assert abs(verts[:, 1].mean() - y) < 2.5 and (abs(verts[:, 0].mean() - x) < 2.5
                                                       or abs(verts[:, 0].mean() - x - 5) < 2.5)
    # the Xenium layers stay the Xenium polygons while a reseg run is loaded
    if raw.boundaries("cell"):
        hx, ax = tl.get_bin(client, f"/viewer/boundaries.bin?source=xenium&kind=cell&{_q(_box(raw, 40.0))}")
        assert hx["n"] > 0 and int(ax["cell"].max()) < len(raw.cells)


def test_reseg_source_follows_subset(xe, client, fake_proseg):
    res, src, gene = fake_proseg
    n_all = len(res["cells_df"])
    h0, _ = tl.get_bin(client, f"/viewer/cells.bin?source={src}")
    xe.subset(min_transcripts=100)
    sub = res["cells_df"]
    assert 0 < len(sub) < n_all and (sub["transcript_counts"] >= 100).all()
    h, a = tl.get_bin(client, f"/viewer/cells.bin?source={src}")
    assert h["n"] == len(sub) and h["token"] != h0["token"]
    _, vals, _ = colours(client, f"source={src}&mode=gene&gene={gene}")
    col = np.asarray(res["expr"][:, xe.DATA["gene_name_to_idx"][gene]].todense()).ravel()
    assert res["expr"].shape[0] == len(sub)
    np.testing.assert_allclose(vals, np.log1p(col), rtol=1e-5)
    assert client.get(f"/viewer/cell/0?source={src}").get_json()["id"] == str(sub.index[0])
    xe.unsubset()
    assert len(res["cells_df"]) == n_all and res["expr"].shape[0] == n_all
    h2, _ = tl.get_bin(client, f"/viewer/cells.bin?source={src}")
    assert h2["n"] == n_all
