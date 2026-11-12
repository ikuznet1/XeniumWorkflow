"""SpaGE imputation core (_vectorized_spage) against a straightforward loop.

The SpaGE package itself is a git submodule that may not be fetched (and must not be
downloaded by the tests), so a small stand-in for its PVComputation class is written
to a temporary directory and xe.SPAGE_REPO is pointed at it. What is under test is
the app's own part: scaling, the kNN in principal-vector space and the batched
weighted average, compared with the per-cell loop of the published algorithm.
"""
from __future__ import annotations

import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

STUB = textwrap.dedent('''
    import numpy as np
    from sklearn.decomposition import PCA

    class PVComputation:
        """Principal vectors between two datasets (PCA of each, SVD of the factor overlap)."""
        def __init__(self, n_factors, n_pv, dim_reduction="pca", dim_reduction_target="pca"):
            self.n = n_pv
        def fit(self, source, target):
            ps = PCA(self.n, random_state=0).fit(source).components_
            pt = PCA(self.n, random_state=0).fit(target).components_
            u, s, vt = np.linalg.svd(ps @ pt.T)
            self.source_components_ = u.T @ ps
            self.target_components_ = vt @ pt
            self.cosine_similarity_matrix_ = np.diag(s)
            return self
''')


@pytest.fixture
def spage(xe, tmp_path, monkeypatch):
    repo = tmp_path / "spage_stub"
    (repo / "SpaGE").mkdir(parents=True)
    (repo / "SpaGE" / "__init__.py").write_text("")
    (repo / "SpaGE" / "principal_vectors.py").write_text(STUB)
    monkeypatch.setattr(xe, "SPAGE_REPO", str(repo))
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "SpaGE" or k.startswith("SpaGE.")}
    yield xe._vectorized_spage
    for k in [k for k in sys.modules if k == "SpaGE" or k.startswith("SpaGE.")]:
        sys.modules.pop(k)
    sys.modules.update(saved)
    if str(repo) in sys.path:
        sys.path.remove(str(repo))


def _data(seed=0, n_rna=400, n_spat=250, n_shared=60, n_extra=12):
    """Two cell populations with distinct programmes, seen by both technologies."""
    rng = np.random.default_rng(seed)
    genes = [f"g{k}" for k in range(n_shared)]
    extra = [f"p{k}" for k in range(n_extra)]
    prof = rng.gamma(2.0, 1.0, size=(2, n_shared + n_extra))
    prof[0, : n_shared // 2] *= 4
    prof[1, n_shared // 2: n_shared] *= 4
    prof[0, n_shared:] *= 6                         # held-out genes follow population 0
    t_rna, t_spat = rng.integers(0, 2, n_rna), rng.integers(0, 2, n_spat)
    rna = pd.DataFrame(np.log1p(rng.poisson(prof[t_rna])).astype(np.float32), columns=genes + extra,
                       index=[f"r{i}" for i in range(n_rna)])
    spat = pd.DataFrame(np.log1p(rng.poisson(prof[t_spat][:, :n_shared] * 0.5)).astype(np.float32),
                        columns=genes, index=[f"s{i}" for i in range(n_spat)])
    return spat, rna, extra, t_spat


def _reference(spat, rna, n_pv, predict):
    """The published algorithm, one spatial cell at a time."""
    import scipy.stats as st
    from sklearn.neighbors import NearestNeighbors
    from SpaGE.principal_vectors import PVComputation
    shared = np.intersect1d(spat.columns, rna.columns)
    rs = st.zscore(rna[shared], axis=0).astype(np.float32)
    ss = st.zscore(spat[shared], axis=0).astype(np.float32)
    pv = PVComputation(n_factors=n_pv, n_pv=n_pv).fit(rs, ss)
    eff = max(int(np.sum(np.diag(pv.cosine_similarity_matrix_) > 0.3)), 1)
    S = pv.source_components_.T[:, :eff]
    rp, sp_ = (np.asarray(rs) @ S).astype(np.float32), (np.asarray(ss) @ S).astype(np.float32)
    dist, idx = NearestNeighbors(n_neighbors=50, metric="cosine").fit(rp).kneighbors(sp_)
    Y = rna[predict].to_numpy(dtype=np.float64)
    out = np.zeros((len(spat), len(predict)))
    for j in range(len(spat)):
        ok = dist[j] < 1
        d = dist[j][ok]
        w = 1 - d / d.sum()
        w = w / (len(w) - 1)
        out[j] = w @ Y[idx[j][ok]]
    return out


def test_vectorised_imputation_equals_the_per_cell_loop(spage):
    spat, rna, extra, _ = _data()
    got = spage(spat, rna, 20, extra)
    assert list(got.index) == list(spat.index) and list(got.columns) == extra
    np.testing.assert_allclose(got.to_numpy(), _reference(spat, rna, 20, extra), rtol=2e-3, atol=2e-3)
    assert np.isfinite(got.to_numpy()).all()


def test_imputed_genes_follow_the_population(spage):
    """Sanity of the result itself: genes held out of the spatial panel and high in one
    population come out higher in the spatial cells of that population."""
    spat, rna, extra, t_spat = _data(seed=3)
    got = spage(spat, rna, 20, extra).to_numpy()
    assert got[t_spat == 0].mean() > 1.3 * got[t_spat == 1].mean()


def test_streaming_to_zarr_writes_the_same_values(spage, tmp_path):
    import zarr
    spat, rna, extra, _ = _data(seed=1)
    in_memory = spage(spat, rna, 20, extra)
    path = str(tmp_path / "imp.zarr")
    assert spage(spat, rna, 20, extra, out_zarr=path) is None
    z = zarr.open_array(path, mode="r")
    assert z.shape == in_memory.shape and list(z.attrs["genes"]) == extra
    np.testing.assert_allclose(np.asarray(z[:]), in_memory.to_numpy(), rtol=1e-6)


def test_constant_genes_are_dropped_not_propagated_as_nan(spage):
    spat, rna, extra, _ = _data(seed=2)
    spat2, rna2 = spat.copy(), rna.copy()
    spat2["g5"] = np.float32(0)                 # no counts in this tissue
    rna2["g9"] = np.float32(1)                  # constant in the reference
    got = spage(spat2, rna2, 20, extra)
    without = spage(spat.drop(columns=["g5", "g9"]), rna.drop(columns=["g5", "g9"]), 20, extra)
    assert np.isfinite(got.to_numpy()).all()
    np.testing.assert_allclose(got.to_numpy(), without.to_numpy(), rtol=1e-5, atol=1e-6)


def test_no_shared_genes_is_a_clear_error(spage):
    spat, rna, extra, _ = _data()
    with pytest.raises(ValueError, match="No shared genes"):
        spage(spat.rename(columns=lambda c: "x" + c), rna, 20, extra)
    with pytest.raises(ValueError, match="non-zero variance"):
        spage(spat * 0, rna, 20, extra)


def test_missing_spage_checkout_is_reported(xe, tmp_path, monkeypatch):
    monkeypatch.setattr(xe, "SPAGE_REPO", str(tmp_path / "empty"))
    spat, rna, extra, _ = _data()
    with pytest.raises(RuntimeError, match="SpaGE is not installed"):
        xe._vectorized_spage(spat, rna, 20, extra)
