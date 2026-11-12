"""Cache bookkeeping and background-job guards, on fabricated cache directories.

xe.CACHE_BASE_DIR is monkeypatched to a per-test directory so that fabricated run
folders never mix with the session's cache.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture
def fake_cache(xe, tmp_path, monkeypatch):
    root = tmp_path / "fake_cache"
    root.mkdir()
    monkeypatch.setattr(xe, "CACHE_BASE_DIR", str(root))
    return root


def _run_dir(root, name, marker=None, params=None, patches=0):
    d = root / name
    d.mkdir()
    if marker:
        (d / marker).write_text("x")
    for i in range(patches):
        (d / f"patch_{i:04d}").mkdir()
        (d / f"patch_{i:04d}" / "segmentation.csv").write_text("x")
    if params is not None:
        (d / "params.json").write_text(params if isinstance(params, str) else json.dumps(params))
    return d


def test_cached_run_listing_is_scoped_to_the_dataset(xe, fake_cache):
    name = xe._dataset_name()
    own = hashlib.md5(xe._dataset_key().encode()).hexdigest()[:8]
    other = "0123abcd" if own != "0123abcd" else "abcd0123"
    _run_dir(fake_cache, f"baysor_{name}_{own}_aaaaaaaa", "segmentation.csv",
             {"scale": 6.0, "min_mol": 20, "n_cells": 1234, "x_min": 0, "x_max": 10, "y_min": 0, "y_max": 5})
    _run_dir(fake_cache, f"proseg_{name}_{own}_bbbbbbbb", "cell-metadata.csv.gz", {"voxel_size": None})
    _run_dir(fake_cache, f"baysor_{name}_cccccccc", "segmentation.csv", {"dataset_hash": own})       # old naming
    _run_dir(fake_cache, f"baysor_{name}_dddddddd", "segmentation.csv")                              # old, no params
    # not ours / not usable:
    _run_dir(fake_cache, f"baysor_{name}_{other}_eeeeeeee", "segmentation.csv", {})     # same name, other folder
    _run_dir(fake_cache, f"baysor_{name}_ffffffff", "segmentation.csv", {"dataset_hash": other})
    _run_dir(fake_cache, f"baysor_{name}_2_{own}_11111111", "segmentation.csv", {})     # dataset "<name>_2"
    _run_dir(fake_cache, f"baysor_{name}x_{own}_22222222", "segmentation.csv", {})
    _run_dir(fake_cache, f"baysor_{name}_{own}_33333333", None, {})                     # no output yet
    _run_dir(fake_cache, f"proseg_{name}_{own}_44444444", "segmentation.csv", {})       # wrong tool's marker
    _run_dir(fake_cache, f"baysor_{name}_{own}_55555555", None, None, patches=2)        # patch run in progress
    _run_dir(fake_cache, f"baysor_{name}_{own}_66666666", None, {"n_patches": 2}, patches=2)   # finished patch run
    _run_dir(fake_cache, f"baysor_{name}_{own}_77777777", "segmentation.csv", "{not json")     # corrupt params
    (fake_cache / f"baysor_{name}_{own}_88888888").write_text("a file, not a run")

    runs = {r["value"]: r["label"] for r in xe._list_cached_seg_runs()}
    assert set(runs) == {"baysor:aaaaaaaa", "proseg:bbbbbbbb", "baysor:cccccccc", "baysor:dddddddd",
                         "baysor:66666666", "baysor:77777777"}
    assert runs["baysor:aaaaaaaa"] == "Baysor: r=6.0µm, min=20tx, x=0-10 y=0-5µm, 1,234 cells"
    assert runs["proseg:bbbbbbbb"] == "Proseg: voxel=auto, full slide"
    assert "2 patches" in runs["baysor:66666666"]
    assert runs["baysor:77777777"] == "Baysor: [77777777]"

    # an existing run directory is reused (old naming included); a new one gets the hash
    assert xe._seg_run_dir("baysor", "cccccccc") == str(fake_cache / f"baysor_{name}_cccccccc")
    assert xe._seg_run_dir("baysor", "aaaaaaaa") == str(fake_cache / f"baysor_{name}_{own}_aaaaaaaa")
    assert xe._seg_run_dir("baysor", "99999999") == str(fake_cache / f"baysor_{name}_{own}_99999999")
    assert xe._seg_run_dir("proseg", "aaaaaaaa") == str(fake_cache / f"proseg_{name}_{own}_aaaaaaaa")


def test_clean_cache_removes_only_this_datasets_entries(xe, fake_cache):
    name = xe._dataset_name()
    own = hashlib.md5(xe._dataset_key().encode()).hexdigest()[:8]
    other = "0123abcd"
    mine = [f"baysor_{name}_{own}_aaaaaaaa", f"proseg_{name}_bbbbbbbb", f"spatialdata_{own}1234.zarr"]
    theirs = [f"baysor_{name}_{other}_cccccccc", f"baysor_{name}0_{other}_dddddddd",
              f"spatialdata_{other}5678.zarr", "cellnest_db"]
    for d in mine + theirs:
        _run_dir(fake_cache, d, "segmentation.csv")
    for f in (f"rois_{own}.json", f"Immune_All_Low_{own}.parquet", f"morph_overview_{own}_z0.npz"):
        (fake_cache / f).write_text("x")
        mine.append(f)
    (fake_cache / f"rois_{other}.json").write_text("x")
    theirs.append(f"rois_{other}.json")
    msg = xe._clean_cache_for_dataset()
    assert sorted(os.listdir(fake_cache)) == sorted(theirs)
    assert msg == f"Removed {len(mine)} cache entries for {name}."


def test_cache_size_is_memoised_and_exact_on_request(xe, fake_cache):
    (fake_cache / "a").write_bytes(b"x" * 2048)
    assert xe._cache_size_str(max_age=0) == "2.0 KB"
    (fake_cache / "sub").mkdir()
    (fake_cache / "sub" / "b").write_bytes(b"x" * 2048)
    assert xe._cache_size_str() == "2.0 KB", "reused within max_age"
    assert xe._cache_size_str(max_age=0) == "4.0 KB"
    assert xe._cache_size_str() == "4.0 KB"


def test_annotation_cache_validity(xe):
    f = xe._annot_cache_valid
    cached = pd.DataFrame({"label": ["a", "b", "c"]}, index=["1", "2", "3"])
    assert f(cached, None, None) is True, "Xenium caches are keyed by dataset only"
    assert f(cached, [1, 2, 3], None) is True
    assert f(cached, [1, 2, 3, 4], None) is False and f(cached, [1, 2, 3, 4], None, allow_subset=True) is True
    assert f(cached, [1, 2], None, allow_subset=True) is False and f(cached, [1, 2, 9], None) is False
    signed = cached.assign(_sig="abc")
    assert f(signed, [7, 8, 9], "abc") is True, "the signature decides, not the ids"
    assert f(signed, [1, 2, 3], "zzz") is False and f(signed, [1, 2, 3], None) is False
    assert f(signed.iloc[:0], [1], "abc") is False


def test_stale_job_guard(xe, monkeypatch):
    gen = xe._dataset_gen
    assert xe._stale_job(gen, "running") is False and xe._stale_job(gen, "error") is False
    xe._check_dataset_gen(gen)
    monkeypatch.setattr(xe, "_dataset_gen", gen + 1)          # another dataset was loaded meanwhile
    assert xe._stale_job(gen, "error") is True, "an old job's error is dropped silently"
    for status in ("running", "done"):
        with pytest.raises(xe._DatasetChanged):
            xe._stale_job(gen, status)
    with pytest.raises(xe._DatasetChanged):
        xe._check_dataset_gen(gen)


def test_job_of_a_replaced_dataset_does_not_publish(xe, D, labels, monkeypatch):
    """A spatial-analysis thread that finishes after a dataset swap must not write."""
    real = xe._build_raw_neighbor_matrix

    def swap_then_compute(*a, **k):
        xe._dataset_gen += 1                       # the swap happens mid-job
        return real(*a, **k)

    gen0 = xe._dataset_gen
    monkeypatch.setattr(xe, "_build_raw_neighbor_matrix", swap_then_compute)
    try:
        xe._run_spatial_analysis("labels_celltypist", 4, None)
        assert xe._spatial_state["result"] is None
        assert xe._spatial_state["status"] != "done"
    finally:
        xe._dataset_gen = gen0


def test_dgc_matrix_parts(xe):
    """_scipy_csc_parts feeds R's dgCMatrix: genes x cells, 0-based int32 i/p, float64 x,
    sorted row indices, no explicit zeros, input untouched."""
    m = sp.random(300, 40, density=0.1, format="csr", random_state=1, dtype=np.float32)
    m.data = np.ceil(m.data * 5).astype(np.float32)
    m.data[::7] = 0
    before = (m.data.copy(), m.indices.copy(), m.indptr.copy())
    i, p, x, shape = xe._scipy_csc_parts(m)
    assert shape == (40, 300) and i.dtype == np.int32 and p.dtype == np.int32 and x.dtype == np.float64
    assert len(p) == shape[1] + 1 and p[0] == 0 and p[-1] == len(i) == len(x)
    assert i.min() >= 0 and i.max() < shape[0] and (x != 0).all()
    np.testing.assert_array_equal(sp.csc_matrix((x, i, p), shape=shape).toarray(), m.T.toarray())
    for q in range(shape[1]):
        assert (np.diff(i[p[q]:p[q + 1]]) > 0).all()
    for a, b in zip(before, (m.data, m.indices, m.indptr)):
        np.testing.assert_array_equal(a, b)
    assert i.flags.c_contiguous and x.flags.c_contiguous


def test_csc_view_is_cached_per_matrix_object(xe):
    m = sp.random(50, 8, density=0.3, format="csr", random_state=0)
    holder = {"expr": m}
    c = xe._csc_view(holder, "expr")
    assert c.format == "csc" and xe._csc_view(holder, "expr") is c
    np.testing.assert_array_equal(c.toarray(), m.toarray())
    m2 = (m * 2).tocsr()
    holder["expr"] = m2                               # e.g. subset() replaced the matrix
    c2 = xe._csc_view(holder, "expr")
    assert c2 is not c
    np.testing.assert_array_equal(c2.toarray(), m2.toarray())
    assert xe._csc_view(holder, "missing") is None
    already = {"e": c}
    assert xe._csc_view(already, "e") is c and list(already) == ["e"]


def test_align_expr_rows_follows_df_to_expr(xe, D):
    n = D["expr"].shape[0]
    full = np.arange(n, dtype=np.float32) + 1
    assert xe._align_expr_rows_to_df(full) is full, "identity mapping: no copy"
    keep = list(D["df"].index[::-1][: max(2, n // 2)])
    xe.subset(cell_ids=keep)
    got = xe._align_expr_rows_to_df(full)
    np.testing.assert_array_equal(got, full[np.asarray(D["df_to_expr"])])
    assert len(got) == len(D["df"])
    other = np.arange(n + 3, dtype=np.float32)
    assert xe._align_expr_rows_to_df(other) is other, "arrays of another length are passed through"
