"""R bridge (``-m tools``): sparse matrices handed to R and back through rpy2.

RCTD, SPLIT and the Seurat reference reader all receive the counts as a dgCMatrix
built by _scipy_to_r_dgc and return results read by _r_dgcmatrix_to_scipy. These
tests run that exchange against a real R session with the bundle's own counts and
ask R (not Python) what it received. Skipped without R + the Matrix package + rpy2.

Full annotation / SPLIT / SpaGE runs are not part of the suite: they need a real
single-cell reference and a tissue-sized dataset (see tests/README.md).
"""
from __future__ import annotations

import sys
import threading

import numpy as np
import pytest
import scipy.sparse as sp

import xe_testlib as tl

pytestmark = pytest.mark.tools


@pytest.fixture(scope="module")
def ro():
    if not tl.r_packages_available("Matrix"):
        pytest.skip("R with the Matrix package is not available")
    pytest.importorskip("rpy2")
    import rpy2.robjects as robjects
    import rpy2.robjects.conversion as conversion
    conversion.set_conversion(robjects.default_converter)
    return robjects


@pytest.fixture
def counts(D):
    """A cells x genes slice of the bundle's matrix that has non-zeros, plus names."""
    totals = np.asarray(D["expr"].sum(axis=0)).ravel()
    cols = np.sort(np.argsort(-totals)[:150])
    m = D["expr"][:, cols].tocsr().astype(np.float32)
    assert m.nnz > 0
    return m, [D["gene_names"][i] for i in cols], [str(c) for c in D["df"].index]


def _r_facts(ro, name):
    """What R itself says about the object."""
    r = ro.r
    return {"class": str(r(f"class({name})")[0]),
            "dim": [int(v) for v in r(f"dim({name})")],
            "rownames": list(r(f"rownames({name})")), "colnames": list(r(f"colnames({name})")),
            "sum": float(r(f"sum({name})")[0]),
            "col_sums": np.array(r(f"Matrix::colSums({name})")),
            "row_sums": np.array(r(f"Matrix::rowSums({name})")),
            "nnz": int(r(f"Matrix::nnzero({name})")[0]),
            "valid": bool(r(f"isTRUE(methods::validObject({name}, test=TRUE))")[0])}


def _check(ro, name, m, genes, cells):
    f = _r_facts(ro, name)
    assert f["class"] == "dgCMatrix" and f["valid"]
    assert f["dim"] == [len(genes), len(cells)], "R receives genes x cells"
    assert f["rownames"] == genes and f["colnames"] == cells
    np.testing.assert_allclose(f["col_sums"], np.asarray(m.sum(axis=1)).ravel())     # per cell
    np.testing.assert_allclose(f["row_sums"], np.asarray(m.sum(axis=0)).ravel())     # per gene
    assert f["sum"] == pytest.approx(float(m.sum())) and f["nnz"] == m.nnz
    # a single entry, addressed by name on the R side
    r, c = (int(v[0]) for v in m.nonzero())
    assert float(ro.r(f'{name}["{genes[c]}", "{cells[r]}"]')[0]) == pytest.approx(float(m[r, c]))


def test_app_matrix_reaches_r_as_named_dgcmatrix_and_returns(xe, ro, counts):
    m, genes, cells = counts
    xe._scipy_to_r_dgc(m, genes, cells, ".xe_test_counts")
    _check(ro, ".xe_test_counts", m, genes, cells)
    back = xe._r_dgcmatrix_to_scipy(ro.r[".xe_test_counts"])
    assert back.shape == (len(genes), len(cells))
    assert tl.sparse_equal(back.T, m)
    # R-side arithmetic survives the return trip (what SPLIT::purify hands back)
    ro.r(".xe_test_half <- .xe_test_counts * 0.5")
    assert tl.sparse_equal(xe._r_dgcmatrix_to_scipy(ro.r[".xe_test_half"]).T, m * 0.5)
    assert not bool(ro.r('exists("._dgc_i")')[0]), "temporaries are removed from the R session"
    with pytest.raises(ValueError, match="do not match"):
        xe._scipy_to_r_dgc(m, genes[:-1], cells, ".xe_bad")
    ro.r("rm(.xe_test_counts, .xe_test_half)")


def test_matrix_exchange_works_from_a_callback_thread(xe, ro, counts):
    """Dash runs callbacks in worker threads, where rpy2's conversion context is unset;
    the app sets it at the start of every R job. Same sequence here."""
    m, genes, cells = counts
    result = {}

    def job():
        try:
            import rpy2.robjects as _ro
            import rpy2.robjects.conversion as _conv
            _conv.set_conversion(_ro.default_converter)       # as _run_split_correction does
            xe._scipy_to_r_dgc(m, genes, cells, ".xe_thread_counts")
            result["back"] = xe._r_dgcmatrix_to_scipy(_ro.r[".xe_thread_counts"])
        except Exception as exc:          # surfaced in the main thread below
            result["error"] = exc

    t = threading.Thread(target=job)
    t.start()
    t.join(300)
    assert "error" not in result, result.get("error")
    assert tl.sparse_equal(result["back"].T, m)
    _check(ro, ".xe_thread_counts", m, genes, cells)
    ro.r("rm(.xe_thread_counts)")


def test_standalone_run_split_uses_the_same_exchange(ro, counts):
    sys.path.insert(0, str(tl.REPO / "functions"))
    import run_split
    m, genes, cells = counts
    run_split._assign_r_dgcmatrix(".xe_rs_counts", m, genes, cells)
    _check(ro, ".xe_rs_counts", m, genes, cells)
    assert tl.sparse_equal(run_split._r_dgcmatrix_to_scipy(ro.r[".xe_rs_counts"]).T, m)
    ro.r("rm(.xe_rs_counts)")


def test_explicit_zeros_and_duplicates_do_not_reach_r(xe, ro):
    rows = np.array([0, 0, 1, 2, 2])
    cols = np.array([1, 1, 0, 2, 3])
    vals = np.array([2.0, 3.0, 0.0, 4.0, 1.0])
    m = sp.coo_matrix((vals, (rows, cols)), shape=(3, 4)).tocsr()      # duplicate summed, explicit 0 kept
    xe._scipy_to_r_dgc(m, list("abcd"), ["c1", "c2", "c3"], ".xe_z")
    f = _r_facts(ro, ".xe_z")
    assert f["valid"] and f["nnz"] == 3 and f["sum"] == 10.0
    assert int(ro.r("length(.xe_z@x)")[0]) == 3, "no explicit zeros stored"
    assert float(ro.r('.xe_z["b", "c1"]')[0]) == 5.0
    ro.r("rm(.xe_z)")
