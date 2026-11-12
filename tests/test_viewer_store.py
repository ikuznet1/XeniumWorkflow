"""xenium_viewer_api without the app: wire format and the lazy tile store.

This module imports on its own, so these tests read the bundle in place (read-only;
the session guard in conftest fails the run if the bundle changes) and compare what
the store returns from cells.zarr.zip / transcripts.zarr.zip with the same data read
independently from the parquet files of the bundle.
"""
from __future__ import annotations

import json
import struct
import time

import numpy as np
import pandas as pd
import pytest

import xe_testlib as tl


@pytest.fixture(scope="module")
def va():
    import sys
    sys.path.insert(0, str(tl.REPO))
    import xenium_viewer_api
    return xenium_viewer_api


@pytest.fixture(scope="module")
def store(va, source_bundle):
    return va.XeniumTileStore(str(source_bundle))


@pytest.fixture(scope="module")
def region(raw):
    """A small box around the median cell, and the gene transcripts inside it."""
    if not raw.has("transcripts.parquet"):
        pytest.skip("bundle has no transcripts.parquet")
    cx, cy = raw.cells["x_centroid"].median(), raw.cells["y_centroid"].median()
    near = raw.cells.iloc[((raw.cells["x_centroid"] - cx) ** 2 + (raw.cells["y_centroid"] - cy) ** 2).argmin()]
    box = (float(near["x_centroid"]) - 25, float(near["y_centroid"]) - 25,
           float(near["x_centroid"]) + 25, float(near["y_centroid"]) + 25)
    tx = raw.transcripts(bbox=box, columns=["feature_name", "x_location", "y_location", "qv",
                                             "is_gene", "codeword_category"])
    return box, tx


# ── wire format ──────────────────────────────────────────────────────────────
def test_pack_layout_is_as_documented(va):
    arrays = {"a": np.arange(5, dtype=np.uint8), "b": np.array([1.5, -2.25, 3e20], dtype=np.float32),
              "c": np.array([65535, 7, 9], dtype=np.uint16), "d": np.array([-1, 2], dtype=np.int32),
              "e": np.array([4294967295], dtype=np.uint32), "empty": np.zeros(0, dtype=np.float32)}
    blob = va.pack({"kind": "x", "n": 3, "label": "µm ✓"}, arrays)
    # hand-decoded, independent of tl.unpack
    (hlen,) = struct.unpack("<I", blob[:4])
    assert hlen % 4 == 0 and len(blob) % 4 == 0
    header = json.loads(blob[4:4 + hlen])
    assert header["label"] == "µm ✓" and header["kind"] == "x" and header["n"] == 3
    assert [(a["name"], a["dtype"], a["length"]) for a in header["arrays"]] == [
        ("a", "uint8", 5), ("b", "float32", 3), ("c", "uint16", 3), ("d", "int32", 2),
        ("e", "uint32", 1), ("empty", "float32", 0)]
    off = 4 + hlen
    assert blob[off:off + 5] == bytes(range(5)) and blob[off + 5:off + 8] == b"\0\0\0"
    assert struct.unpack("<3f", blob[off + 8:off + 20]) == (1.5, -2.25, np.float32(3e20))
    assert struct.unpack("<3H", blob[off + 20:off + 26]) == (65535, 7, 9)
    # and through the shared decoder
    h2, out = tl.unpack(blob)
    assert h2 == header
    for k, v in arrays.items():
        np.testing.assert_array_equal(out[k], v)
        assert out[k].dtype == v.dtype


def test_pack_accepts_non_contiguous_and_rejects_unknown_dtypes(va):
    base = np.arange(20, dtype=np.float32)
    _, out = tl.unpack(va.pack({}, {"v": base[::3]}))
    np.testing.assert_array_equal(out["v"], base[::3])
    with pytest.raises(TypeError):
        va.pack({}, {"v": np.arange(3, dtype=np.float64)})
    with pytest.raises(TypeError):
        va.pack({}, {"v": np.arange(3, dtype=np.int64)})


def test_image_token_follows_the_file(va, tmp_path):
    f = tmp_path / "a.tif"
    f.write_bytes(b"x" * 10)
    t1 = va.image_token(str(f))
    assert t1 and t1 == va.image_token(str(f))
    f.write_bytes(b"y" * 11)
    assert va.image_token(str(f)) != t1
    assert va.image_token(str(tmp_path / "missing.tif")) == ""


def test_store_registry_reset_keeps_only_the_current_bundle(va, tmp_path):
    saved = dict(va._stores)
    try:
        va._stores.clear()
        a, b = va.tile_store(str(tmp_path / "a")), va.tile_store(str(tmp_path / "b"))
        assert va.tile_store(str(tmp_path / "a")) is a and a is not b
        va.reset(str(tmp_path / "a"))
        assert list(va._stores) == [str(tmp_path / "a")]
        va.reset()
        assert not va._stores
    finally:
        va._stores.clear()
        va._stores.update(saved)


def test_store_on_a_directory_without_the_zarr_zips(va, tmp_path):
    s = va.XeniumTileStore(str(tmp_path))
    assert not s.has_boundaries and not s.has_transcripts
    assert s.boundaries_bbox("cell", 0, 0, 10, 10) == ({"n": 0, "truncated": False, "available": False}, {})
    assert s.transcripts_bbox(0, 0, 10, 10, ["A"]) == ({"n": 0, "available": False}, {})
    assert s.cell_ids() is None


# ── boundaries ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("kind", ["cell", "nucleus"])
def test_boundaries_equal_the_parquet_polygons(store, raw, kind):
    if not store.has_boundaries or not raw.has(f"{kind}_boundaries.parquet"):
        pytest.skip("bundle has no cells.zarr.zip / boundary parquet")
    cells = raw.cells
    cx, cy = cells["x_centroid"].median(), cells["y_centroid"].median()
    box = (cx - 150, cy - 150, cx + 150, cy + 150) if len(cells) > 5000 else (-1e9, -1e9, 1e9, 1e9)
    meta, arrays = store.boundaries_bbox(kind, *box)
    assert meta["available"] and not meta["truncated"]
    got = tl.rings(arrays)
    assert meta["n"] == len(got) == len(arrays["start"]) - 1 > 0
    assert int(arrays["start"][-1]) * 2 == len(arrays["xy"])
    truth = raw.rings(kind)
    seen = set()
    per_cell = {}
    for row, verts in got:
        cid = cells.index[row]            # payload indexes cells in file order
        seen.add(cid)
        per_cell[cid] = per_cell.get(cid, 0) + 1
        # one of the cell's polygons (several only for a multinucleate cell's nuclei);
        # the zip may store the ring without its closing vertex
        match = [w for w in truth[cid] if len(verts) in (len(w), len(w) - 1)
                 and np.allclose(verts, w[:len(verts)], atol=1e-3)]
        assert match, f"ring of {cid} is not one of its {kind} polygons"
        assert (verts[:, 0].max() >= box[0]) and (verts[:, 0].min() <= box[2])
        assert (verts[:, 1].max() >= box[1]) and (verts[:, 1].min() <= box[3])
    assert all(n <= len(truth[cid]) for cid, n in per_cell.items()), "no polygon twice"

    def meets(v):
        return (v[:, 0].max() >= box[0] and v[:, 0].min() <= box[2]
                and v[:, 1].max() >= box[1] and v[:, 1].min() <= box[3])
    # completeness: every polygon whose bounding box meets the query box is returned
    expected = {cid: sum(meets(v) for v in rings) for cid, rings in truth.items()}
    expected = {cid: n for cid, n in expected.items() if n}
    assert per_cell == expected


def test_boundaries_empty_box_and_limit(store):
    if not store.has_boundaries:
        pytest.skip("bundle has no cells.zarr.zip")
    meta, arrays = store.boundaries_bbox("cell", -1e6, -1e6, -9e5, -9e5)
    assert meta == {"n": 0, "truncated": False, "available": True}
    assert len(arrays["cell"]) == 0 and arrays["start"].tolist() == [0]
    meta, arrays = store.boundaries_bbox("cell", -1e9, -1e9, 1e9, 1e9, limit=3)
    assert meta["truncated"] and meta["n"] > 3 and arrays == {}
    assert store.boundaries_bbox("membrane", 0, 0, 1, 1)[0]["available"] is False


# ── transcripts ──────────────────────────────────────────────────────────────
def _as_table(arrays, genes):
    return (pd.DataFrame({"g": np.asarray(genes)[arrays["gene"]],
                          "x": arrays["xy"][0::2], "y": arrays["xy"][1::2]})
            .sort_values(["g", "x", "y"]).reset_index(drop=True))


def test_transcripts_in_a_box_equal_the_parquet_rows(store, region):
    if not store.has_transcripts:
        pytest.skip("bundle has no transcripts.zarr.zip")
    box, tx = region
    genes = list(tl.gene_tx(tx)["feature_name"].value_counts().index[:6])
    assert genes, "no gene transcripts around the median cell"
    genes = genes + ["__not_a_gene__"]                   # unknown names are ignored
    meta, arrays = store.transcripts_bbox(*box, genes)
    assert meta["level"] == 0 and meta["filtered"] and not meta["clustered"] and not meta["subsampled"]
    want = tx[tx["feature_name"].isin(genes) & (tx["qv"] >= 20)]
    want = (pd.DataFrame({"g": want["feature_name"].values,
                          "x": want["x_location"].values.astype(np.float32),
                          "y": want["y_location"].values.astype(np.float32)})
            .sort_values(["g", "x", "y"]).reset_index(drop=True))
    got = _as_table(arrays, genes)
    assert meta["n"] == len(got) == len(want) > 0
    pd.testing.assert_frame_equal(got, want)
    assert arrays["gene"].dtype == np.uint16 and arrays["xy"].dtype == np.float32
    assert "weight" not in arrays

    meta_w, arrays_w = store.transcripts_bbox(*box, genes, weights=True)
    assert meta_w["n"] == meta["n"] and (arrays_w["weight"] == 1).all(), "level 0 points are single transcripts"

    # the Q-score threshold is applied at level 0
    strict = store.transcripts_bbox(*box, genes, qv_min=1000.0)[0]
    loose = store.transcripts_bbox(*box, genes, qv_min=0.0)[0]
    assert strict["n"] == 0 and loose["n"] >= meta["n"]
    hi = float(tx["qv"].max())
    n_hi = int((tx["feature_name"].isin(genes) & (tx["qv"] >= hi)).sum())
    assert store.transcripts_bbox(*box, genes, qv_min=hi)[0]["n"] == n_hi


def test_transcripts_gene_order_defines_the_codes(store, region):
    if not store.has_transcripts:
        pytest.skip("bundle has no transcripts.zarr.zip")
    box, tx = region
    genes = list(tl.gene_tx(tx)["feature_name"].value_counts().index[:3])
    if len(genes) < 2:
        pytest.skip("fewer than two genes in the region")
    a = _as_table(store.transcripts_bbox(*box, genes)[1], genes)
    b = _as_table(store.transcripts_bbox(*box, genes[::-1])[1], genes[::-1])
    pd.testing.assert_frame_equal(a, b)
    one = store.transcripts_bbox(*box, genes[:1])[1]
    assert (one["gene"] == 0).all() and len(one["gene"]) == int((a["g"] == genes[0]).sum())


def test_transcripts_huge_box_is_bounded_and_fast(store, raw, region):
    """A box of the whole plane (the viewer sends one while zoomed far out) must come
    back quickly at a coarse level of detail instead of decoding every tile."""
    if not store.has_transcripts:
        pytest.skip("bundle has no transcripts.zarr.zip")
    _box, tx = region
    genes = list(tl.gene_tx(tx)["feature_name"].value_counts().index[:3])
    store.transcripts_bbox(0, 0, 1, 1, genes)                   # open the store outside the timing
    for box in [(-1e300, -1e300, 1e300, 1e300), (0, 0, 1e300, 1e300), (-3e38, -3e38, 3e38, 3e38)]:
        t0 = time.time()
        meta, arrays = store.transcripts_bbox(*box, genes, weights=True)
        dt = time.time() - t0
        assert dt < 20, f"huge box took {dt:.1f}s"
        assert meta["available"] and meta["n"] <= 400_000
        assert meta["clustered"] == (meta["level"] > 0) and meta["filtered"] == (meta["level"] == 0)
        if meta["n"]:
            assert np.isfinite(arrays["xy"]).all()
            assert len(arrays["weight"]) == meta["n"] and (arrays["weight"] >= 1).all()
            if meta["level"] > 0:
                assert meta["subsampled"]
    # at a coarse level a point stands for several transcripts: weights add up to no
    # more than the transcripts of those genes in the file
    if raw.n_transcripts <= 5_000_000 and meta["n"]:
        all_tx = raw.transcripts(columns=["feature_name", "qv"])
        total = int(all_tx["feature_name"].isin(genes).sum())
        assert int(arrays["weight"].sum()) <= total
        if meta["level"] == 0:
            assert meta["n"] == int((all_tx["feature_name"].isin(genes) & (all_tx["qv"] >= 20)).sum())


def test_transcripts_tile_cache_is_bounded(store, region):
    if not store.has_transcripts:
        pytest.skip("bundle has no transcripts.zarr.zip")
    box, tx = region
    genes = list(tl.gene_tx(tx)["feature_name"].value_counts().index[:2])
    first = store.transcripts_bbox(*box, genes)
    again = store.transcripts_bbox(*box, genes)
    assert first[0] == again[0]
    np.testing.assert_array_equal(first[1]["xy"], again[1]["xy"])
    cache = getattr(store, "_tiles", None)
    if cache is not None:
        assert len(cache) <= store.TILE_CACHE
