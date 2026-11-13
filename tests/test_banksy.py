"""Banksy spatial domains and the cluster sets computed in the app.

The feature construction is checked against the formulas of the paper on small fabricated
neighbourhoods; the method is validated on fabricated cells with planted domains that
expression-only clustering cannot see; the cluster-set plumbing (install, labels, both
renderers, cache, subset, export) and the job are run on the fixture bundle.
"""
import os
import threading

import dash
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import xe_testlib as tl


@pytest.fixture(autouse=True)
def _clean_banksy_state(xe):
    yield
    for key in list(xe._cs_registry):
        xe._cs_remove(key)
    xe._cs_registry.clear()
    with xe._bk_lock:
        xe._bk_state.update({"status": "idle", "message": "", "result": None})
    import shutil
    for f in os.listdir(xe.CACHE_BASE_DIR):
        if f.startswith(("clustersets_", "banksy_")):
            shutil.rmtree(os.path.join(xe.CACHE_BASE_DIR, f), ignore_errors=True)


def _ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


def _world(seed=0, n=4000, size=800.0, G=120, depth=40, shift=2.0, quad=True):
    """Cells uniform on a square. Three cell types with strong markers (genes 60-119, x4),
    identical in every domain; domains (quadrants, or halves) differ by a weak programme
    in every cell (15 genes x *shift*). ~40 counts per cell: a single cell does not show
    its domain, the mean of its neighbours does."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, size, (n, 2))
    dom = (xy[:, 0] > size / 2).astype(int)
    if quad:
        dom = dom * 2 + (xy[:, 1] > size / 2).astype(int)
    typ = rng.integers(0, 3, n)
    prof = np.ones((n, G))
    for t in range(3):
        prof[typ == t, 60 + t * 20:80 + t * 20] *= 4.0
    for d in range(dom.max() + 1):
        prof[dom == d, d * 15:(d + 1) * 15] *= shift
    prof /= prof.sum(axis=1, keepdims=True)
    return xy, dom, typ, sp.csr_matrix(rng.poisson(prof * depth).astype(np.float32))


def _sample(X, xy, key="/fake/bk0", label="xenium", prefix="c"):
    n, G = X.shape
    genes = [f"g{j}" for j in range(G)]
    df = pd.DataFrame({"x_centroid": xy[:, 0], "y_centroid": xy[:, 1]}, index=[f"{prefix}{i}" for i in range(n)])
    return {"label": label, "df": df, "csc": sp.csc_matrix(X), "rows": np.arange(n),
            "gni": {g: j for j, g in enumerate(genes)}, "genes": genes, "x_offset": 0.0,
            "corrected": False, "ds": {}, "key": key, "name": os.path.basename(key)}


# ── neighbour weights ────────────────────────────────────────────────────────
def test_neighbour_weights_follow_the_paper(xe):
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 100, (300, 2))
    nb = xe._bk_neighbours(xy, 15)
    assert nb["idx"].shape == (300, 30) and nb["gm"].shape == (300, 15) and nb["ga"].shape == (300, 30)
    assert not (nb["idx"] == np.arange(300)[:, None]).any()                  # never its own neighbour
    np.testing.assert_allclose(nb["gm"].sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(nb["ga"].sum(axis=1), 1.0, atol=1e-12)
    # brute force for one cell: Gaussian in r with the 7th (of 15) / 15th (of 30) neighbour as width
    u = 17
    d = np.hypot(*(xy - xy[u]).T)
    order = np.argsort(d)[1:31]
    np.testing.assert_array_equal(nb["idx"][u], order)
    for k, got in ((15, nb["gm"][u]), (30, nb["ga"][u])):
        r = d[order[:k]]
        w = np.exp(-r ** 2 / r[k // 2 - 1] ** 2)
        np.testing.assert_allclose(got, w / w.sum(), rtol=1e-10)
    off = xy[order] - xy[u]
    np.testing.assert_allclose(nb["phi"][u], np.arctan2(off[:, 1], off[:, 0]))


def test_few_cells_and_coincident_cells(xe):
    nb = xe._bk_neighbours(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]), 15)
    assert nb["k"] == 2 and nb["k2"] == 2 and np.isfinite(nb["gm"]).all()
    nb = xe._bk_neighbours(np.zeros((6, 2)), 3)                 # all on one spot
    assert np.isfinite(nb["gm"]).all() and np.isfinite(nb["ga"]).all()
    np.testing.assert_allclose(nb["gm"].sum(axis=1), 1.0)
    assert not (nb["idx"] == np.arange(6)[:, None]).any()


def test_neighbours_never_cross_samples(xe):
    rng = np.random.default_rng(1)
    a, b = rng.uniform(0, 50, (200, 2)), rng.uniform(0, 50, (150, 2))     # the same place in two samples
    W = xe._bk_weight_matrices([a, b], 10)
    for key in ("m", "a", "re", "im"):
        M = W[key].tocoo()
        assert ((M.row < 200) == (M.col < 200)).all(), key
    assert W["m"].shape == (350, 350) and W["s"].shape == (350,)


# ── the three blocks ─────────────────────────────────────────────────────────
def test_mean_and_agf_blocks(xe):
    rng = np.random.default_rng(2)
    g = np.arange(30)
    xy = np.array([[x, y] for x in g for y in g], dtype=float) * 10.0 + rng.normal(0, 0.5, (900, 2))
    const = np.full(900, 3.0)
    ramp = xy[:, 0] / 10.0                              # a linear gradient along x
    noise = rng.poisson(2.0, 900).astype(float)
    Cn = sp.csr_matrix(np.column_stack([const, ramp, noise]).astype(np.float32))
    W = xe._bk_weight_matrices([xy], 15)
    C, M, G = xe._bk_blocks(Cn, W, 0, 900)
    np.testing.assert_allclose(C, Cn.toarray())
    np.testing.assert_allclose(M[:, 0], 3.0, rtol=1e-5)                     # mean of a constant
    np.testing.assert_allclose(G[:, 0], 0.0, atol=1e-4)                     # no gradient
    inner = (xy[:, 0] > 60) & (xy[:, 0] < 230) & (xy[:, 1] > 60) & (xy[:, 1] < 230)
    np.testing.assert_allclose(M[inner, 1], ramp[inner], atol=0.15)         # mean of a ramp = its value
    assert G[inner, 1].min() > 0.5 and G[inner, 1].std() / G[inner, 1].mean() < 0.1     # a constant slope
    assert G[inner, 1].mean() > 3 * G[inner, 2].mean()                      # ... far above that of noise
    # brute force for one cell, with and without centring
    u = int(np.flatnonzero(inner)[40])
    nb = xe._bk_neighbours(xy, 15)
    w, phi, idx = nb["ga"][u], nb["phi"][u], nb["idx"][u]
    for col in (1, 2):
        v = Cn.toarray()[idx, col].astype(float)
        raw = abs((w * v * np.exp(1j * phi)).sum())
        cen = abs((w * (v - (w * v).sum()) * np.exp(1j * phi)).sum())
        np.testing.assert_allclose(G[u, col], cen, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(xe._bk_blocks(Cn, W, 0, 900, True, False)[2][u, col], raw, rtol=1e-4, atol=1e-5)
    # without the AGF there is no third block
    assert xe._bk_blocks(Cn, W, 0, 900, agf=False)[2] is None


def test_blocks_do_not_change_under_rotation_and_reflection(xe):
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 200, (500, 2))
    Cn = sp.csr_matrix(rng.poisson(1.5, (500, 6)).astype(np.float32))
    a = 0.7
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    ref = xe._bk_blocks(Cn, xe._bk_weight_matrices([xy], 12), 0, 500)
    for other in (xy @ R.T, xy * np.array([1.0, -1.0]), xy + 1000.0):
        got = xe._bk_blocks(Cn, xe._bk_weight_matrices([other], 12), 0, 500)
        np.testing.assert_allclose(got[1], ref[1], atol=1e-4)
        np.testing.assert_allclose(got[2], ref[2], atol=1e-4)


def test_block_weights(xe):
    for lam in (0.0, 0.2, 0.8, 1.0):
        own, mean, agf = xe._bk_block_weights(lam, True)
        np.testing.assert_allclose([own ** 2, mean ** 2, agf ** 2], [1 - lam, lam / 1.5, lam / 3.0])
        np.testing.assert_allclose(own ** 2 + mean ** 2 + agf ** 2, 1.0)    # convex: mu = 1.5
        own, mean, agf = xe._bk_block_weights(lam, False)
        np.testing.assert_allclose([own ** 2, mean ** 2, agf], [1 - lam, lam, 0.0])


def test_embedding_is_the_pca_of_the_augmented_matrix(xe):
    xy, _dom, _typ, X = _world(n=900, G=40, quad=False)
    Cn, _pos, _med = xe._bk_matrix([_sample(X, xy)], [f"g{j}" for j in range(40)])
    W = xe._bk_weight_matrices([xy], 15)
    for agf in (True, False):
        B = xe._bk_augmented(Cn, W, 0.8, agf)
        assert B.shape == (900, 40 * (3 if agf else 2))
        np.testing.assert_allclose(B.mean(axis=0), 0.0, atol=1e-5)
        sd = B.std(axis=0, ddof=1)
        np.testing.assert_allclose(sd[:40], np.sqrt(0.2), rtol=1e-4)        # z-scaled, then weighted
        np.testing.assert_allclose(sd[40:80], np.sqrt(0.8 / 1.5 if agf else 0.8), rtol=1e-4)
        emb = xe._bk_embed(Cn, W, 0.8, agf, 8, chunk=250)                   # several blocks of cells
        U, S, _Vt = np.linalg.svd(B, full_matrices=False)
        ref = U[:, :8] * S[:8]
        for i in range(8):
            assert abs(np.corrcoef(ref[:, i], emb["pcs"][:, i])[0, 1]) > 0.9999
        np.testing.assert_allclose(emb["variance"], S[:8] ** 2 / 899, rtol=1e-3)
        assert 0 < emb["explained"] < 1
    # recomputing the neighbourhood blocks instead of keeping them gives the same components
    old = xe._BK_KEEP_BYTES
    try:
        xe._BK_KEEP_BYTES = 0
        again = xe._bk_embed(Cn, W, 0.8, False, 8, chunk=250)
    finally:
        xe._BK_KEEP_BYTES = old
    assert not again["kept"] and emb["kept"]
    np.testing.assert_allclose(again["pcs"], emb["pcs"], atol=1e-3)


def test_normalisation_and_gene_cap(xe):
    rng = np.random.default_rng(4)
    X = sp.csr_matrix(rng.poisson(1.0, (400, 30)).astype(np.float32))
    X = sp.vstack([X, sp.csr_matrix((1, 30), dtype=np.float32)]).tocsr()         # a cell without counts
    xy = rng.uniform(0, 100, (401, 2))
    s = _sample(X, xy)
    s["rows"] = np.concatenate([np.arange(400), [-1]])                             # ... and not in the matrix
    Cn, pos, med = xe._bk_matrix([s], s["genes"])
    assert Cn.shape == (400, 30) and len(pos[0]) == 400 and 400 not in pos[0]
    tot = np.asarray(X[:400].sum(axis=1)).ravel()
    assert med == np.median(tot)
    np.testing.assert_allclose(np.asarray(Cn.sum(axis=1)).ravel(), med, rtol=1e-5)  # every cell at the median total
    logged, _p, _m = xe._bk_matrix([s], s["genes"], log1p=True)
    np.testing.assert_allclose(logged.toarray(), np.log1p(Cn.toarray()), rtol=1e-5)
    # the cap keeps the most variable genes; genes seen in too few cells are never used
    V = np.zeros((500, 8), dtype=np.float32)
    V[:, 0] = 1.0
    V[::2, 1] = 6.0
    V[::2, 2] = 2.0
    V[:3, 3] = 9.0                                                                  # 3 cells only
    V[:, 4:] = rng.poisson(0.3, (500, 4))
    sel = xe._bk_select_genes(sp.csr_matrix(V), max_genes=2)
    assert list(sel) == [1, 2]
    assert 3 not in xe._bk_select_genes(sp.csr_matrix(V), max_genes=100)
    assert list(xe._bk_select_genes(sp.csr_matrix(V), max_genes=100)) == [0, 1, 2, 4, 5, 6, 7]


# ── graph clustering ─────────────────────────────────────────────────────────
def test_graph_cluster_backends(xe, monkeypatch):
    rng = np.random.default_rng(5)
    centres = np.array([[0, 0], [8, 0], [0, 8]], dtype=float)
    truth = np.repeat([0, 1, 2], [300, 200, 100])
    X = centres[truth] + rng.normal(0, 0.5, (600, 2))
    perm = rng.permutation(600)
    X, truth = X[perm], truth[perm]
    lab, info = xe._cs_graph_cluster(X, n_neighbors=15, resolution=0.1, seed=0)
    if xe._cs_leiden_available():
        assert info["backend"] == "leiden" and "exact" in info["neighbours"]
    assert _ari(truth, lab) == 1.0 and info["n_clusters"] == 3
    assert list(np.bincount(lab)) == [300, 200, 100]                     # ordered by size
    again, _ = xe._cs_graph_cluster(X, n_neighbors=15, resolution=0.1, seed=0)
    np.testing.assert_array_equal(lab, again)                             # fixed seed
    steps = []
    xe._cs_graph_cluster(X, progress=lambda f, m=None: steps.append(f))
    assert steps and steps[-1] == 1.0 and steps == sorted(steps)
    # without leidenalg / igraph: KMeans, said so, with the install hint
    monkeypatch.setattr(xe, "_cs_leiden_available", lambda: False)
    lab, info = xe._cs_graph_cluster(X, n_neighbors=15, resolution=0.3, seed=0)      # k = 10 x resolution
    assert info["backend"].startswith("kmeans (fallback") and "leidenalg" in info["note"]
    assert info["n_clusters"] == 3 and _ari(truth, lab) == 1.0
    assert xe._cs_graph_cluster(X, n_clusters=5)[1]["n_clusters"] == 5
    assert xe._cs_graph_cluster(X[:2])[1]["backend"] == "none"


def test_snn_graph_is_the_jaccard_index_of_the_neighbour_sets(xe):
    rng = np.random.default_rng(6)
    X = rng.normal(size=(60, 3)).astype(np.float32)
    idx, how = xe._cs_knn(X, 5)
    assert how == "exact neighbours" and idx.shape == (60, 5) and not (idx == np.arange(60)[:, None]).any()
    d = np.linalg.norm(X[:, None] - X[None], axis=2)
    np.fill_diagonal(d, np.inf)
    np.testing.assert_array_equal(np.sort(idx, axis=1), np.sort(np.argsort(d, axis=1)[:, :5], axis=1))
    G = xe._cs_snn_graph(idx).toarray()
    sets = [set(idx[i]) | {i} for i in range(60)]
    for i in range(60):
        for j in range(i + 1, 60):
            want = (len(sets[i] & sets[j]) / len(sets[i] | sets[j])) if (j in idx[i] or i in idx[j]) else 0.0
            assert abs(G[i, j] - want) < 1e-6
    assert not np.tril(G).any()


# ── planted domains ──────────────────────────────────────────────────────────
def test_planted_domains_are_recovered_where_expression_alone_fails(xe):
    if not xe._cs_leiden_available():
        pytest.skip("leidenalg / igraph not installed")
    xy, dom, typ, X = _world(n=1800, size=540.0)          # the same cell density as 3000 on 700 µm
    s = [_sample(X, xy)]
    plain = xe._bk_compute(s, {"lam": 0.0, "resolution": 0.5})
    assert _ari(dom, plain["labels"]) < 0.05 and _ari(typ, plain["labels"]) > 0.9      # cell types only
    bk = xe._bk_compute(s, {"lam": 0.8, "resolution": 0.5})
    assert _ari(dom, bk["labels"]) > 0.9 and abs(_ari(typ, bk["labels"])) < 0.05         # the domains
    assert bk["cluster"]["n_clusters"] == dom.max() + 1
    cell = xe._bk_compute(s, {"lam": 0.2, "resolution": 0.5})
    assert _ari(typ, cell["labels"]) > 0.9                                                # still the cell types
    summ = xe._bk_summarise(s, bk, None)
    assert summ["purity"] > 0.9 and summ["purity_expected"] < 0.55
    assert xe._bk_summarise(s, plain, None)["purity"] < 0.45
    # the top genes of a domain are those of its programme
    t = summ["table"]
    for c in range(len(t)):
        d = int(np.bincount(dom[bk["labels"] == c]).argmax())
        top = [int(g[1:]) for g in t["top_genes"].iloc[c].split(", ")]
        assert all(d * 15 <= g < (d + 1) * 15 for g in top[:3]), (d, top)


def test_two_samples_are_clustered_together_with_neighbours_within_sample(xe):
    if not xe._cs_leiden_available():
        pytest.skip("leidenalg / igraph not installed")
    xa, da, _ta, Xa = _world(seed=1, n=1500, quad=False)
    xb, db, _tb, Xb = _world(seed=2, n=1200, quad=False)
    xb = xb[:, ::-1].copy()                      # the second section lies the other way round
    db = (xb[:, 1] > 400).astype(int)
    samples = [_sample(Xa, xa, "/fake/bkA", "S0", "a"), _sample(Xb, xb, "/fake/bkB", "S1", "b")]
    out = xe._bk_compute(samples, {"lam": 0.8, "resolution": 0.5})
    M = out["W"]["m"].tocoo()
    assert ((M.row < 1500) == (M.col < 1500)).all()
    la, lb = out["labels"][:1500], out["labels"][1500:]
    assert _ari(da, la) > 0.9 and _ari(db, lb) > 0.9
    # the same domain carries the same label in both samples
    for d in (0, 1):
        assert np.bincount(la[da == d]).argmax() == np.bincount(lb[db == d]).argmax()
    summ = xe._bk_summarise(samples, out, None)
    assert {"n_S0", "n_S1"} <= set(summ["table"].columns)
    assert int(summ["table"][["n_S0", "n_S1"]].to_numpy().sum()) == 2700


def test_too_few_cells_or_genes(xe):
    xy, _d, _t, X = _world(n=30, G=20)
    with pytest.raises(ValueError, match="at least"):
        xe._bk_compute([_sample(X, xy)], {})
    a = _sample(X, xy)
    b = _sample(X, xy, "/fake/other")
    b["genes"] = ["zz"]
    with pytest.raises(ValueError, match="no gene in common"):
        xe._bk_compute([a, b], {})


# ── cluster sets in the app ──────────────────────────────────────────────────
def _labels_for(D, seed=0, k=3):
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, k, len(D["df"])).astype(object)
    lab[:k] = np.arange(k)
    return lab


def test_install_makes_a_cluster_set_of_the_view(xe, D, client):
    lab = _labels_for(D)
    lab[5] = None
    n_methods = len(D["cluster_methods"])
    key = xe._cs_install("Banksy λ=0.8", [lab], "xenium", meta={"params": {"lam": 0.8}})
    assert key == "banksy_lambda_0.8" and xe._cs_label(key) == "Banksy λ=0.8"
    assert D["cluster_methods"][-1] == key and len(D["cluster_methods"]) == n_methods + 1
    col = D["df"][xe.cluster_col(key)]
    assert col.dtype.kind == "i" and col.iloc[5] == 0
    np.testing.assert_array_equal(col.to_numpy()[6:], np.asarray(lab[6:], dtype=int) + 1)     # 0 … k−1 → 1 … k
    assert xe._cs_methods() == [key] and xe._cs_methods("xenium") == [key]
    assert xe._cs_meta(key)["params"] == {"lam": 0.8} and not xe._cs_is_shared(key)
    # the dropdown, the label helpers, both renderers
    opts, value = xe.update_cluster_options(1, 1, "xenium", 0, 0, 0, D["cluster_methods"][0])
    assert {"label": "Banksy λ=0.8", "value": key} in opts and value is dash.no_update
    labs = xe._an_cluster_labels(xe._an_samples(), key, False, None)[0]
    assert labs[5] is None and set(labs[6:]) == {"1", "2", "3"}                 # a cell without a label is in no group
    raw = xe._raw_cluster_labels(xe._view_tables(), key, None)[0]
    assert raw[5] is None and raw[6] == str(int(lab[6]) + 1)
    fig = xe.make_spatial_fig("cluster", key, None, 2, 0.8, [], None)
    assert sorted(t.name for t in tl.marker_traces(fig)) == ["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"]
    h, a = tl.get_bin(client, f"/viewer/colors.bin?mode=cluster&method={key}")
    names = np.array(h["categories"], dtype=object)[a["codes"].astype(int)]
    np.testing.assert_array_equal(names, ["Cluster " + str(v) for v in col.to_numpy()])      # WebGL view
    # in the cache, per dataset
    d = xe._cs_dir(D)
    assert sorted(os.listdir(d)) == [f"{key}__xenium.json", f"{key}__xenium.parquet"]
    # removing it takes it out of the table, the list and the cache
    assert xe._cs_remove("Banksy λ=0.8")
    assert key not in D["cluster_methods"] and xe.cluster_col(key) not in D["df"].columns
    assert os.listdir(d) == [] and xe._cs_methods() == []


def test_cached_sets_come_back_on_load(xe, D):
    lab = _labels_for(D, seed=1, k=4)
    key = xe._cs_install("Joint clusters", [lab], "xenium", meta={"shared": True})
    want = D["df"][xe.cluster_col(key)].to_numpy().copy()
    # as after a restart: no column, no entry
    D["df"].drop(columns=[xe.cluster_col(key)], inplace=True)
    D["cluster_methods"].remove(key)
    xe._cs_registry.clear()
    assert xe._cs_restore("xenium") == [key]
    np.testing.assert_array_equal(D["df"][xe.cluster_col(key)].to_numpy(), want)
    assert key in D["cluster_methods"] and xe._cs_label(key) == "Joint clusters" and xe._cs_is_shared(key)
    assert xe._cs_restore("xenium") == []                                       # once
    assert [o["label"] for o in xe._cs_options("xenium")][-1] == "Joint clusters"
    # labels of other cells under the same name are not installed
    D["df"].drop(columns=[xe.cluster_col(key)], inplace=True)
    D["cluster_methods"].remove(key)
    p = os.path.join(xe._cs_dir(D), f"{key}__xenium.parquet")
    t = pd.read_parquet(p)
    t["cell_id"] = ["other_" + c for c in t["cell_id"]]
    t.to_parquet(p)
    assert xe._cs_restore("xenium") == [] and key not in D["cluster_methods"]
    # a `clust__` column a store brought along is listed as a cluster set
    D["df"]["clust__from_store"] = 1
    assert xe._cs_restore("xenium") == ["from_store"] and "from_store" in D["cluster_methods"]
    D["cluster_methods"].remove("from_store")


def test_install_with_a_subset_active(xe, D):
    n = len(D["df"])
    lo = float(np.sort(D["df"]["transcript_counts"].to_numpy())[n // 3])
    xe.subset(min_transcripts=lo)
    sub = D["df"]
    assert 0 < len(sub) < n
    key = xe._cs_install("Banksy λ=0.2", [np.zeros(len(sub), dtype=int)], "xenium")
    assert (D["df"][xe.cluster_col(key)] == 1).all() and len(D["df"]) == len(sub)
    xe.unsubset()
    col = D["df"][xe.cluster_col(key)]
    assert len(col) == n and (col[sub.index] == 1).all()
    assert (col.drop(sub.index) == 0).all()                                     # outside the subset: no label
    labs = xe._an_cluster_labels(xe._an_samples(), key, False, None)[0]
    assert sum(l is None for l in labs) == n - len(sub)


def test_install_checks_its_input_and_names_levels(xe, D):
    with pytest.raises(ValueError, match="label arrays"):
        xe._cs_install("x", [], "xenium")
    with pytest.raises(ValueError, match="labels for"):
        xe._cs_install("x", [np.zeros(3)], "xenium")
    lab = np.array(["b", "a", None] * (len(D["df"]) // 3 + 1), dtype=object)[:len(D["df"])]
    key = xe._cs_install("Named", [lab], "xenium")
    assert xe._cs_registry[key]["levels"] == ["a", "b"]
    assert list(D["df"][xe.cluster_col(key)].iloc[:3]) == [2, 1, 0]
    assert xe._cs_key("Joint clusters (Harmony)") == "joint_clusters_harmony"


def test_shared_set_keeps_its_labels_across_samples(xe, D, monkeypatch):
    extra = {"df": D["df"].copy(), "cluster_methods": list(D["cluster_methods"]), "data_dir": "/fake/second",
             "x_offset": 5000.0, "expr": D["expr"], "gene_names": D["gene_names"],
             "gene_name_to_idx": D["gene_name_to_idx"], "df_to_expr": D["df_to_expr"]}
    xe.EXTRA_DATASETS.append(extra)
    try:
        labs = [_labels_for(D, 2), _labels_for(D, 3)]
        key = xe._cs_install("Joint clusters", labs, "xenium")
        assert xe._cs_is_shared(key) and key in extra["cluster_methods"]
        assert xe.cluster_col(key) in extra["df"].columns
        got = xe._an_cluster_labels(xe._view_tables(), key, False, None)
        assert set(got[0]) == set(got[1]) == {"C1", "C2", "C3"}                 # not "S0:C1" / "S1:C1"
        raw = xe._raw_cluster_labels(xe._view_tables(), key, None)
        assert set(raw[0]) == {"1", "2", "3"}
        # a Xenium set stays per sample
        own = xe._an_cluster_labels(xe._view_tables(), D["cluster_methods"][0], False, None)
        assert all(str(l).startswith("S0:C") for l in own[0]) and all(str(l).startswith("S1:C") for l in own[1])
        fig = xe.make_spatial_fig("cluster", key, None, 2, 0.8, [], None)
        assert sorted({t.name for t in tl.marker_traces(fig)}) == ["C1", "C2", "C3"]
        assert os.path.isfile(os.path.join(xe._cs_dir(extra), f"{key}__xenium.parquet"))
    finally:
        del xe.EXTRA_DATASETS[:]


def test_cluster_set_on_a_resegmentation_run(xe, D, client, small_view):
    n = len(D["df"])
    rng = np.random.default_rng(8)
    cdf = pd.DataFrame({"x_centroid": D["df"]["x_centroid"].to_numpy(), "y_centroid": D["df"]["y_centroid"].to_numpy(),
                        "transcript_counts": 5, "cluster_kmeans_10": rng.integers(0, 2, n).astype(str)},
                       index=[str(i + 1) for i in range(n)])
    expr = sp.csr_matrix(D["expr"])[np.asarray(D["df_to_expr"])][:, :len(xe._panel_gene_names())].tocsr()
    with xe._proseg_lock:
        xe._proseg_state.update({"status": "done", "result": {
            "cells_df": cdf, "cell_bounds": {}, "out_dir": "/nonexistent/proseg_run_x", "expr": expr,
            "source": "proseg", "sdata_path": None}})
    src = "proseg:abc12345"
    _run(xe, seg=src)
    res = xe._bk_result()
    key = res["key"]
    bdf = xe._proseg_state["result"]["cells_df"]
    assert f"cluster_{key}" in bdf.columns and key not in D["cluster_methods"]       # on the run, not on Xenium
    assert xe._cs_methods(src) == [key] and xe._cs_methods("xenium") == []
    assert {"label": "Banksy λ=0.8", "value": key} in xe.update_cluster_options(1, 1, src, 0, 0, 0, None)[0]
    assert key not in [o["value"] for o in xe._cs_options("xenium")]
    want = bdf[f"cluster_{key}"].astype(str).to_numpy()
    # picked in Clustering Method: the set; any other method: the run's own clusters
    labs = xe._an_cluster_labels(xe._an_samples(src), key, False, xe._proseg_state["result"])[0]
    np.testing.assert_array_equal(labs, want)
    own = xe._an_cluster_labels(xe._an_samples(src), D["cluster_methods"][0], False, xe._proseg_state["result"])[0]
    np.testing.assert_array_equal(own, cdf["cluster_kmeans_10"].to_numpy())
    h, a = tl.get_bin(client, f"/viewer/colors.bin?mode=cluster&method={key}&source={src}")
    np.testing.assert_array_equal(np.array(h["categories"], dtype=object)[a["codes"].astype(int)],
                                  ["Cluster " + v for v in want])
    fig = xe.make_spatial_fig("cluster", key, None, 2, 0.8, [], None, proseg_active=True)
    got = {t.name: len(t.x) for t in tl.marker_traces(fig)}
    assert got == {f"Cluster {v}": int(c) for v, c in zip(*np.unique(want, return_counts=True))}
    assert os.path.isfile(os.path.join(xe._cs_dir(D), f"{key}__proseg_proseg_run_x.parquet"))
    # as when the run is loaded again from the cache
    xe._proseg_state["result"]["cells_df"] = cdf.copy()
    xe._cs_registry.clear()
    assert xe._cs_restore(src) == [key]
    np.testing.assert_array_equal(xe._proseg_state["result"]["cells_df"][f"cluster_{key}"].astype(str), want)


# ── the job ──────────────────────────────────────────────────────────────────
@pytest.fixture
def small_view(xe, monkeypatch):
    """The fixture bundle has 23 cells: let the job run on them."""
    monkeypatch.setattr(xe, "_BK_MIN_CELLS", 5)


def _run(xe, **kw):
    args = dict(lam=0.8, k_geom=5, agf=["on"], resolution=0.5, max_genes=300, n_pcs=5, k_expr=5, seed=0,
                log1p=[], seg="xenium", counts="original")
    args.update(kw)
    out = xe.run_banksy(1, args["lam"], args["k_geom"], args["agf"], args["resolution"], args["max_genes"],
                        args["n_pcs"], args["k_expr"], args["seed"], args["log1p"], args["seg"], args["counts"])
    assert out == (True, False, "Running…"), out
    tl.wait_state(xe._bk_state, xe._bk_lock, timeout=300)


def test_job_installs_the_cluster_set_and_caches_it(xe, D, small_view):
    assert xe._JOB_KIND_NAMES["banksy"] == "Banksy clustering"
    _run(xe)
    res = xe._bk_result()
    key = res["key"]
    assert res["name"] == "Banksy λ=0.8" and key in D["cluster_methods"] and not res["cached"]
    col = D["df"][xe.cluster_col(key)].to_numpy()
    assert col.min() >= 0 and col.max() == len(res["table"]) >= 1
    assert int(res["table"]["n_cells"].sum()) == int((col > 0).sum())
    msg = xe._bk_state["message"]
    assert "Banksy λ=0.8" in msg and "Spatial coherence" in msg and "Clustering Method" in msg
    job = [j for j in xe._jobs.values() if j.kind == "banksy"][-1]
    assert job.state == "done" and job.name == "Banksy clustering"
    # poll: reports the end once, bumps the cluster-set version and the redraw version
    out = xe.poll_banksy(1, 0, 3, 7)
    assert out[1] is False and out[3] is True and out[4] == 4 and out[5] == 8
    assert xe.poll_banksy(2, out[2], 4, 8)[4] is dash.no_update
    fig, rows = xe.update_banksy_results(out[2], 0, None, None)
    assert len(rows) == len(res["table"]) and len(fig.data) >= 1
    csv = xe.download_banksy(1)["content"]
    assert "# clusters" in csv and "lambda 0.8" in csv and "Singhal" in csv
    # the same request again is read from the cache and gives the same labels
    _run(xe)
    assert xe._bk_result()["cached"] and "read from the cache" in xe._bk_state["message"]
    np.testing.assert_array_equal(D["df"][xe.cluster_col(key)].to_numpy(), col)
    # other parameters: another run, and for another λ another set
    _run(xe, lam=0.2, agf=[])
    r2 = xe._bk_result()
    assert r2["name"] == "Banksy λ=0.2" and not r2["cached"] and r2["key"] != key
    assert {key, r2["key"]} <= set(D["cluster_methods"])
    assert "own expression and neighbourhood mean of" in xe._bk_state["message"]


def test_job_refuses_a_view_with_too_few_cells(xe):
    out = xe.run_banksy(1, 0.8, 5, ["on"], 0.5, 300, 5, 5, 0, [], "xenium", "original")
    assert out == (True, False, "Running…")
    with pytest.raises(AssertionError, match="at least 50 cells"):
        tl.wait_state(xe._bk_state, xe._bk_lock, timeout=120)
    assert xe._bk_result() is None and xe._cs_methods() == []


def test_job_guards_and_preset(xe):
    for n in (0, None):
        with pytest.raises(dash.exceptions.PreventUpdate):
            xe.run_banksy(n, 0.8, 15, ["on"], 0.5, 2000, 20, 50, 0, [], "xenium", "original")
        with pytest.raises(dash.exceptions.PreventUpdate):
            xe.download_banksy(n)
    with pytest.raises(dash.exceptions.PreventUpdate):
        xe.download_banksy(1)                                     # nothing computed
    assert xe.run_banksy(1, 1.5, 15, ["on"], 0.5, 2000, 20, 50, 0, [], "xenium", "original")[2].startswith("λ must")
    assert "positive" in xe.run_banksy(1, 0.8, 0, ["on"], 0.5, 2000, 20, 50, 0, [], "xenium", "original")[2]
    assert xe.set_banksy_preset("domains") == (0.8, 0.5) and xe.set_banksy_preset("celltype") == (0.2, 1.0)
    assert xe.set_banksy_preset("custom") == (dash.no_update, dash.no_update)
    fig, rows = xe.update_banksy_results(0, 0, None, None)
    assert rows == [] and "No Banksy" in fig.layout.annotations[0].text


def test_job_can_be_cancelled_and_rerun(xe, D, monkeypatch, small_view):
    entered, release = threading.Event(), threading.Event()
    real = xe._bk_weight_matrices

    def slow(xys, k):
        entered.set()
        release.wait(30)
        return real(xys, k)

    monkeypatch.setattr(xe, "_bk_weight_matrices", slow)
    args = (1, 0.8, 5, ["on"], 0.5, 300, 5, 5, 0, [], "xenium", "original")
    assert xe.run_banksy(*args) == (True, False, "Running…")
    assert entered.wait(60)
    assert xe.run_banksy(*args)[2] == "Banksy is already running."
    job = [j for j in xe._jobs.values() if j.kind == "banksy"][-1]
    assert xe._job_cancel(job.id)
    release.set()
    job.thread.join(60)
    assert job.state == "cancelled" and xe._bk_state["status"] == "error" and xe._bk_state["message"] == "Cancelled"
    assert xe._bk_result() is None and xe._cs_methods() == []          # nothing installed
    out = xe.poll_banksy(1, 0, 3, 7)
    assert out[0] == "Cancelled" and out[4] is dash.no_update and out[5] is dash.no_update
    monkeypatch.setattr(xe, "_bk_weight_matrices", real)
    _run(xe)
    assert xe._bk_result() is not None


def test_cluster_set_is_exported_by_save_as_spatialdata(xe, D, tmp_path):
    spatialdata = pytest.importorskip("spatialdata")
    lab = _labels_for(D, 5)
    key = xe._cs_install("Banksy λ=0.8", [lab], "xenium")
    msg, poll_disabled, btn_disabled = xe.start_save_sdata(1, str(tmp_path), "bk", [], "xenium")
    assert (poll_disabled, btn_disabled) == (False, True), msg
    tl.wait_state(xe._save_sdata_state, xe._save_sdata_lock, timeout=600)
    obs = spatialdata.read_zarr(os.path.join(str(tmp_path), "bk.zarr")).tables["table"].obs
    col = xe.cluster_col(key)
    assert col in obs.columns
    ids = obs["cell_id"].astype(str).to_numpy() if "cell_id" in obs.columns else obs.index.astype(str).to_numpy()
    want = D["df"][col].copy()
    want.index = want.index.astype(str)
    np.testing.assert_array_equal(pd.to_numeric(obs[col]).to_numpy(), want.reindex(ids).to_numpy())


# ── review round 5 ───────────────────────────────────────────────────────────
def test_cache_signature_covers_the_counts_not_only_whose_they_are(xe):
    xy, _dom, _typ, X = _world(seed=3, n=300)
    s = _sample(X, xy)
    genes = list(s["genes"])
    p = xe._bk_params({})
    sig = xe._bk_signature([s], "xenium", p, genes)
    assert sig == xe._bk_signature([dict(s)], "xenium", p, genes)
    # the same cells, genes, shape and number of stored values - other counts (SPLIT run again)
    m = s["csc"].copy()
    m.data = m.data[::-1].copy()
    assert m.shape == s["csc"].shape and m.nnz == s["csc"].nnz
    assert xe._bk_signature([{**s, "csc": m}], "xenium", p, genes) != sig


def test_shared_flag_belongs_to_the_install_not_to_the_name(xe, D):
    extra = {"df": D["df"].copy(), "cluster_methods": list(D["cluster_methods"]), "data_dir": "/fake/second",
             "x_offset": 5000.0, "expr": D["expr"], "gene_names": D["gene_names"],
             "gene_name_to_idx": D["gene_name_to_idx"], "df_to_expr": D["df_to_expr"]}
    xe.EXTRA_DATASETS.append(extra)
    try:
        key = xe._cs_install("Banksy λ=0.8", [_labels_for(D, 2), _labels_for(D, 3)], "xenium")
        assert xe._cs_is_shared(key)
        # the same name installed on a Proseg run (one table) does not change the Xenium set
        n = len(D["df"])
        cdf = pd.DataFrame({"x_centroid": D["df"]["x_centroid"].to_numpy(), "y_centroid": D["df"]["y_centroid"].to_numpy(),
                            "transcript_counts": 5}, index=[str(i + 1) for i in range(n)])
        with xe._proseg_lock:
            xe._proseg_state.update({"status": "done", "result": {
                "cells_df": cdf, "cell_bounds": {}, "out_dir": "/nonexistent/proseg_run_y", "expr": None,
                "source": "proseg", "sdata_path": None}})
        assert xe._cs_install("Banksy λ=0.8", [_labels_for(D, 4)], "proseg:abc") == key
        assert xe._cs_is_shared(key), "a single-table install on another segmentation cleared the flag"
        got = xe._an_cluster_labels(xe._view_tables(), key, False, None)
        assert set(got[0]) == set(got[1]) == {"C1", "C2", "C3"}
        # after a restart every sample says which install its labels are from
        for ds in (D, extra):
            ds["df"] = ds["df"].drop(columns=[xe.cluster_col(key)])
            ds["cluster_methods"].remove(key)
        xe._cs_registry.clear()
        xe._cs_table_run.clear()
        assert key in xe._cs_restore("xenium") and xe._cs_is_shared(key)
        # one sample gets the set again from another (single-sample) run: the two samples no
        # longer hold labels of one install, so they are not read as the same clusters
        xe._cs_install("Banksy λ=0.8", [_labels_for(D, 5), None], "xenium")
        assert not xe._cs_is_shared(key)
        got = xe._an_cluster_labels(xe._view_tables(), key, False, None)
        assert all(str(l).startswith("S0:C") for l in got[0] if l is not None)
    finally:
        del xe.EXTRA_DATASETS[:]
        with xe._proseg_lock:
            xe._proseg_state.update({"status": "idle", "result": None})


def test_set_computed_on_a_small_subset_comes_back_on_load(xe, D):
    n = len(D["df"])
    keep = list(D["df"].index.astype(str)[: max(3, n // 4)])            # well under half of the cells
    xe.subset(cell_ids=keep)
    assert len(D["df"]) == len(keep) < n / 2
    key = xe._cs_install("Banksy λ=0.2", [np.arange(len(keep)) % 2], "xenium")
    xe.unsubset()
    want = D["df"][xe.cluster_col(key)].to_numpy().copy()
    assert (want[: len(keep)] > 0).all() and (want[len(keep):] == 0).all()
    # as after a restart
    D["df"].drop(columns=[xe.cluster_col(key)], inplace=True)
    D["cluster_methods"].remove(key)
    xe._cs_registry.clear()
    assert xe._cs_restore("xenium") == [key]
    np.testing.assert_array_equal(D["df"][xe.cluster_col(key)].to_numpy(), want)
