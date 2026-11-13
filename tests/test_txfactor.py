"""Transcript factors: the segmentation-free factorisation of binned transcripts.

Fabricated transcript files with planted spatial programs: the bin × gene counts are
checked against a brute-force count, the fitted loadings against the planted ones (up to
permutation), the projection against its defining properties, and the per-cell comparison,
the map layer and the Color By values against the bins they are read from.
"""
from __future__ import annotations

import os
import shutil
import time

import dash
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import xe_testlib as tl


def _txf_dirs(xe):
    return {os.path.join(xe.CACHE_BASE_DIR, d) for d in os.listdir(xe.CACHE_BASE_DIR) if d.startswith("txfactor_")}


@pytest.fixture(autouse=True)
def _clean_txfactor_state(xe):
    before = _txf_dirs(xe)
    yield
    xe._test_baseline.settle()
    for d in _txf_dirs(xe) - before:               # caches a test wrote: other tests expect none
        shutil.rmtree(d, ignore_errors=True)
    xe._txf_reset()
    xe._controls_reset()


def _write_tx(dirpath, x, y, gene, qv=None, is_gene=None, row_group_size=5000):
    os.makedirs(dirpath, exist_ok=True)
    n = len(x)
    pd.DataFrame({"x_location": np.asarray(x, dtype=np.float32), "y_location": np.asarray(y, dtype=np.float32),
                  "feature_name": np.asarray(gene, dtype=object),
                  "qv": np.full(n, 40.0, dtype=np.float32) if qv is None else np.asarray(qv, dtype=np.float32),
                  "is_gene": np.ones(n, dtype=bool) if is_gene is None else np.asarray(is_gene, dtype=bool)}
                 ).to_parquet(os.path.join(dirpath, "transcripts.parquet"), row_group_size=row_group_size)
    return str(dirpath)


def _planted(rng, genes, n_tx=120_000, size=600.0, k=4, per=12, noise=0.05):
    """Transcripts of *k* spatial programs: the field is cut into k vertical stripes, a
    transcript in stripe j is drawn from program j (its own *per* genes carry 90 % of it,
    10 % is spread over all genes); a share *noise* of the transcripts is uniform.
    Returns (x, y, gene names, programs k × genes, stripe of each transcript)."""
    G = len(genes)
    P = np.full((k, G), 0.1 / G)
    for j in range(k):
        w = rng.uniform(0.5, 1.5, per)
        P[j, j * per:(j + 1) * per] += 0.9 * w / w.sum()
    x = rng.uniform(0, size, n_tx)
    y = rng.uniform(0, size, n_tx)
    stripe = np.minimum((x / (size / k)).astype(int), k - 1)
    g = np.empty(n_tx, dtype=np.int64)
    for j in range(k):
        m = stripe == j
        g[m] = rng.choice(G, size=int(m.sum()), p=P[j])
    uni = rng.random(n_tx) < noise
    g[uni] = rng.integers(0, G, int(uni.sum()))
    xs = x.astype(np.float32).astype(np.float64)                    # as stored in the file
    ys = y.astype(np.float32).astype(np.float64)
    return xs, ys, np.array(genes, dtype=object)[g], P, np.minimum((xs / (size / k)).astype(int), k - 1)


# ── 1. bins ──────────────────────────────────────────────────────────────────
def test_square_bins_are_floor_division_and_hexagons_are_the_nearest_centre(xe):
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-300, 900, 4000), rng.uniform(-200, 700, 4000)
    r, c = xe._txf_bin_index(x, y, 10.0, "square")
    assert np.array_equal(r, np.floor(y / 10).astype(int)) and np.array_equal(c, np.floor(x / 10).astype(int))
    cx, cy = xe._txf_bin_centres(r, c, 10.0, "square")
    assert np.all(np.abs(cx - x) <= 5 + 1e-9) and np.all(np.abs(cy - y) <= 5 + 1e-9)
    for d in (7.0, 10.0, 25.0):
        r, c = xe._txf_bin_index(x, y, d, "hex")
        assert np.all((r + c) % 2 == 0)
        cx, cy = xe._txf_bin_centres(r, c, d, "hex")
        own = np.hypot(cx - x, cy - y)
        assert own.max() <= d / np.sqrt(3) + 1e-9                    # circumradius of the hexagon
        # no other centre of the lattice is nearer: brute force over the neighbourhood
        for dr in range(-2, 3):
            for dc in range(-4, 5):
                if (dr + dc) % 2:
                    continue
                ox, oy = xe._txf_bin_centres(r + dr, c + dc, d, "hex")
                assert np.all(np.hypot(ox - x, oy - y) >= own - 1e-9)
        # the six neighbours of a hexagon are one bin size away
        nx, ny = xe._txf_bin_centres(np.array([0, 0, 1, 1, -1, -1]), np.array([2, -2, 1, -1, 1, -1]), d, "hex")
        assert np.allclose(np.hypot(nx, ny), d)
    assert xe._txf_bin_area(10, "square") == 100 and xe._txf_bin_area(10, "hex") == pytest.approx(86.6025, abs=1e-3)


# ── 2. the scan ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("shape", ["square", "hex"])
def test_scan_equals_a_brute_force_count_with_the_filters_applied(xe, tmp_path, shape):
    rng = np.random.default_rng(2)
    genes = [f"G{i}" for i in range(30)]
    n = 30_000
    x, y = rng.uniform(0, 400, n).astype(np.float32), rng.uniform(0, 300, n).astype(np.float32)
    g = rng.choice(genes + ["NegControlProbe_1", "NOT_ON_PANEL"], size=n)
    qv = rng.choice([10.0, 25.0, 40.0], size=n)
    is_gene = g != "NegControlProbe_1"
    d = _write_tx(tmp_path / "s", x, y, g, qv, is_gene, row_group_size=4000)       # 8 row groups
    scan = xe._txf_scan(d, genes, 12.0, shape)
    ok = (qv >= 20) & is_gene & np.isin(g, genes)
    assert scan["total"] == int(ok.sum()) == int(scan["X"].sum())
    r, c = xe._txf_bin_index(x[ok].astype(np.float64), y[ok].astype(np.float64), 12.0, shape)
    hand = pd.DataFrame({"r": r, "c": c, "g": pd.Categorical(g[ok], categories=genes).codes}).value_counts()
    coo = scan["X"].tocoo()
    got = pd.Series(coo.data, index=pd.MultiIndex.from_arrays(
        [scan["r"][coo.row], scan["c"][coo.row], coo.col], names=["r", "c", "g"]))
    assert len(got) == len(hand)
    assert (got.sort_index().to_numpy() == hand.sort_index().to_numpy()).all()
    assert got.sort_index().index.equals(hand.sort_index().index)
    # bins are unique and in key order
    key = xe._txf_bin_key(scan["r"], scan["c"])
    assert np.all(np.diff(key) > 0) and scan["X"].shape == (len(key), 30)


def test_scan_of_the_bundle_counts_the_transcripts_of_the_cell_feature_matrix(xe, D):
    pq = os.path.join(D["data_dir"], "transcripts.parquet")
    if not os.path.exists(pq):
        pytest.skip("bundle without transcripts.parquet")
    genes = xe._ds_panel_genes(D)
    scan = xe._txf_scan(D["data_dir"], genes, 10.0, "square")
    tx = tl.gene_tx(pd.read_parquet(pq))
    nm = tx["feature_name"].map(lambda v: v.decode() if isinstance(v, bytes) else v).astype(str)
    assert scan["total"] == int(nm.isin(genes).sum()) > 0
    per_gene = np.asarray(scan["X"].sum(axis=0)).ravel()
    want = nm.value_counts().reindex(genes).fillna(0).to_numpy()
    assert np.array_equal(per_gene, want)
    # the tiles of transcripts.zarr.zip give the same counts per gene
    if os.path.exists(os.path.join(D["data_dir"], "transcripts.zarr.zip")):
        z = xe._txf_scan(D["data_dir"], genes, 10.0, "square",
                         batches=xe._txf_tx_batches(D["data_dir"], genes, source="zarr"))
        assert z["total"] == scan["total"] and tl.sparse_equal(z["X"], scan["X"])
        assert np.array_equal(z["r"], scan["r"]) and np.array_equal(z["c"], scan["c"])


def test_counts_are_cached_per_bin_and_shape_and_a_broken_cache_is_rescanned(xe, D, tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    genes = list(D["gene_names"])[:20]
    x, y, g, _P, _s = _planted(rng, genes, n_tx=5000, size=200, k=2, per=5)
    d = _write_tx(tmp_path / "s", x, y, g)
    ds = {"data_dir": d, "gene_names": genes}
    a = xe._txf_counts(ds, genes, 10.0, "square")
    path = xe._txf_counts_file(ds, genes, 10.0, "square")
    assert a["cached"] is False and os.path.exists(path)
    assert path != xe._txf_counts_file(ds, genes, 10.0, "hex") != xe._txf_counts_file(ds, genes, 20.0, "hex")
    assert path != xe._txf_counts_file(ds, genes[:-1], 10.0, "square")
    monkeypatch.setattr(xe, "_txf_scan", lambda *a, **k: pytest.fail("rescanned although cached"))
    b = xe._txf_counts(ds, genes, 10.0, "square")
    assert b["cached"] is True and tl.sparse_equal(a["X"], b["X"]) and b["total"] == a["total"] == 5000
    assert np.array_equal(a["r"], b["r"]) and np.array_equal(a["c"], b["c"])
    monkeypatch.undo()
    with open(path, "wb") as fh:
        fh.write(b"not an npz")
    c = xe._txf_counts(ds, genes, 10.0, "square")
    assert c["cached"] is False and tl.sparse_equal(a["X"], c["X"])
    with pytest.raises(ValueError, match="loaded from a .zarr"):
        xe._txf_counts({"sdata_path": "/x/y.zarr"}, genes, 10.0, "square")


# ── 3. projection and fit ────────────────────────────────────────────────────
@pytest.mark.parametrize("use_numba", [True, False], ids=["numba", "numpy"])
def test_projection_keeps_the_counts_and_finds_the_planted_mixture(xe, monkeypatch, use_numba):
    if use_numba:
        pytest.importorskip("numba")
    monkeypatch.setattr(xe, "_TXF_USE_NUMBA", use_numba)
    monkeypatch.setattr(xe, "_txf_kernel_cache", {})
    assert (xe._txf_em_kernel() is not None) == use_numba
    rng = np.random.default_rng(4)
    k, G, n = 3, 60, 400
    H = rng.dirichlet(np.full(G, 0.2), size=k)
    theta = rng.dirichlet(np.full(k, 0.5), size=n)
    tot = rng.integers(200, 2000, n)
    X = sp.csr_matrix(np.vstack([rng.multinomial(t, th @ H) for t, th in zip(tot, theta)]).astype(np.float32))
    Hn = xe._txf_normalise(H)
    assert np.allclose(Hn.sum(axis=1), 1.0, atol=1e-5)
    W = xe._txf_project(X, Hn, n_iter=200, tol=1e-7)
    assert W.shape == (n, k) and (W >= 0).all()
    assert np.allclose(W.sum(axis=1), tot, rtol=1e-4)               # every transcript is attributed
    est = W / W.sum(axis=1, keepdims=True)
    assert np.abs(est - theta).mean() < 0.03 and np.corrcoef(est.ravel(), theta.ravel())[0, 1] > 0.99
    # it is the minimiser of the KL divergence: sklearn's transform reaches no lower value

    def kl(Wm):
        M = np.maximum(Wm @ Hn.astype(np.float64), 1e-30)
        Xd = X.toarray().astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.where(Xd > 0, Xd * np.log(Xd / M), 0.0).sum() - Xd.sum() + M.sum())

    from sklearn.decomposition import NMF
    ref = NMF(n_components=k, solver="mu", beta_loss="kullback-leibler", max_iter=2000, tol=1e-9)
    ref.components_, ref.n_components_ = Hn.astype(np.float64), k
    assert kl(W.astype(np.float64)) <= kl(ref.transform(X.astype(np.float64))) * (1 + 1e-4)
    # chunks and threads do not change the result; empty rows stay 0
    X2 = sp.vstack([X[:5], sp.csr_matrix((2, G), dtype=np.float32), X[5:]], format="csr")
    monkeypatch.setattr(xe, "_TXF_CHUNK_NNZ", 500)
    W2 = xe._txf_project(X2, Hn, n_iter=200, tol=1e-7, threads=3)
    assert (W2[5:7] == 0).all() and np.abs(np.delete(W2, [5, 6], axis=0) / tot[:, None] - est).max() < 0.01
    assert xe._txf_project(sp.csr_matrix((0, G), dtype=np.float32), Hn).shape == (0, k)


@pytest.mark.parametrize("method,shape", [("nmf", "square"), ("nmf", "hex"), ("lda", "square")])
def test_planted_spatial_programs_are_recovered_up_to_permutation(xe, tmp_path, method, shape):
    rng = np.random.default_rng(5)
    genes = [f"G{i}" for i in range(80)]
    x, y, g, P, _stripe = _planted(rng, genes, n_tx=150_000, size=600.0, k=4, per=12)
    d = _write_tx(tmp_path / "s", x, y, g, row_group_size=20_000)
    scan = xe._txf_scan(d, genes, 10.0, shape)
    n_tx = np.asarray(scan["X"].sum(axis=1)).ravel()
    H, n_it = xe._txf_fit(scan["X"][n_tx >= 10], 4, method, seed=0)
    assert H.shape == (4, 80) and np.allclose(H.sum(axis=1), 1.0, atol=1e-5) and n_it > 0
    match, cos, _C = xe._txf_match(H, P)
    assert sorted(match) == [0, 1, 2, 3]                             # one factor per planted program
    assert cos.min() > 0.9, cos
    # the dominant factor of a bin is the program of its stripe (bins astride a border left out)
    W = xe._txf_project(scan["X"], H)
    cx, _cy = xe._txf_bin_centres(scan["r"], scan["c"], 10.0, shape)
    inside = (np.abs((cx % 150.0) - 75.0) < 60.0) & (n_tx >= 10) & (cx > 0) & (cx < 600)
    want = np.minimum((cx / 150.0).astype(int), 3)
    assert (match[W.argmax(axis=1)][inside] == want[inside]).mean() > 0.97
    # a fixed seed gives the same loadings; the top genes of a factor are its planted genes
    H2, _ = xe._txf_fit(scan["X"][n_tx >= 10], 4, method, seed=0)
    assert np.allclose(H, H2, atol=1e-6)
    top = xe._txf_top_genes(H, genes, W.sum(axis=0), n=8)
    for f in range(4):
        planted = {genes[j] for j in range(match[f] * 12, match[f] * 12 + 12)}
        assert set(top[top["factor"] == f]["gene"]) <= planted
        assert (top[top["factor"] == f]["lift"] > 2).all()


def test_entropy_and_agreement_match_hand_values(xe):
    M = np.array([[4.0, 0, 0, 0], [1, 1, 1, 1], [3, 1, 0, 0], [0, 0, 0, 0]])
    dom, top, ent = xe._txf_entropy(M)
    assert list(dom) == [0, 0, 0, -1]
    assert top[0] == 1 and top[1] == 0.25 and top[2] == 0.75 and np.isnan(top[3])
    assert ent[0] == 0 and ent[1] == pytest.approx(1.0) and np.isnan(ent[3])
    assert ent[2] == pytest.approx(-(0.75 * np.log(0.75) + 0.25 * np.log(0.25)) / np.log(4))
    ct = xe._txf_crosstab(["F1", "F1", "F2", "F2", None, "F2"], ["a", "a", "b", "a", "a", None])
    assert ct.loc["F1", "a"] == 2 and ct.loc["F2", "b"] == 1 and ct.to_numpy().sum() == 4
    a = xe._txf_agreement(ct)
    from sklearn.metrics import adjusted_rand_score
    assert a["n"] == 4 and a["purity"] == pytest.approx(0.75)
    assert a["ari"] == pytest.approx(adjusted_rand_score(["a", "a", "b", "a"], ["F1", "F1", "F2", "F2"]))
    assert xe._txf_agreement(pd.DataFrame()) == {}
    same = xe._txf_agreement(xe._txf_crosstab(["a", "b", "b"], ["a", "b", "a"]))
    assert same["same_name"] == pytest.approx(2 / 3)


# ── 4. the job on the bundle's cells over fabricated transcripts ─────────────
def _two_sided(xe, D, tmp_path, monkeypatch, name="two", n_tx=60_000):
    """Transcripts of two programs, left and right of the middle of the bundle's cells
    (genes 0-19 left, 20-39 right), in place of the bundle's transcript file; the cells are
    labelled Left / Right by their side."""
    rng = np.random.default_rng(6)
    genes = list(D["gene_names"])
    df = D["df"]
    xc, yc = df["x_centroid"].to_numpy(float), df["y_centroid"].to_numpy(float)
    mid = float(np.median(xc))
    x0, x1, y0, y1 = xc.min() - 60, xc.max() + 60, yc.min() - 60, yc.max() + 60
    x = rng.uniform(x0, x1, n_tx).astype(np.float32).astype(np.float64)
    y = rng.uniform(y0, y1, n_tx).astype(np.float32).astype(np.float64)
    left = x < mid
    g = np.where(left, rng.integers(0, 20, n_tx), rng.integers(20, 40, n_tx))
    bg = rng.random(n_tx) < 0.05
    g[bg] = rng.integers(40, 60, int(bg.sum()))
    d = _write_tx(tmp_path / name, x, y, np.array(genes, dtype=object)[g])
    monkeypatch.setitem(xe.DATA, "data_dir", d)
    lab = pd.Series(np.where(xc < mid, "Left", "Right"), index=df.index.astype(str))
    with xe._annot_lock:
        xe._annot_state["labels_celltypist"] = lab
    return {"x": x, "y": y, "g": g, "mid": mid, "labels": lab, "genes": genes, "dir": d}


def _run(xe, n, method="nmf", k=2, bin_um=10, shape="square", min_tx=5, labels="celltype:celltypist", seed=0):
    out = xe.run_txfactor(n, method, k, bin_um, shape, min_tx, 20000, seed, labels, "xenium", None, "original")
    assert out == (True, False, "Running…"), out
    return tl.wait_state(xe._txf_state, xe._txf_lock)


@pytest.mark.parametrize("shape", ["square", "hex"])
def test_job_compares_the_cells_with_the_bins_under_them(xe, D, tmp_path, monkeypatch, shape):
    f = _two_sided(xe, D, tmp_path, monkeypatch)
    msg = _run(xe, 1, shape=shape)
    res = xe._txf_result()
    assert res["names"] == ["F1", "F2"] and res["method"] == "nmf" and res["from_cache"] is False
    assert "in the spirit of FICTURE; not that tool" in msg and "Q ≥ 20" in msg and "seed 0" in msg
    smp = res["samples"][0]
    assert smp["total"] == len(f["x"]) and np.allclose(smp["W"].sum(axis=1), smp["n_tx"], rtol=1e-3)
    # the two factors are the two sides
    side = np.array(["L" if (res["H"][j, :20].sum() > res["H"][j, 20:40].sum()) else "R" for j in range(2)])
    assert sorted(side) == ["L", "R"]
    assert min(res["H"][side == "L", :20].sum(), res["H"][side == "R", 20:40].sum()) > 0.9
    # per cell: the mixture is the sum of the factor counts of the bins under the 7 points
    df = D["df"]
    xy = df[["x_centroid", "y_centroid"]].to_numpy(float)
    keys = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(smp["r"], smp["c"]))}
    rad = np.sqrt(df["cell_area"].to_numpy(float) / np.pi) * xe._TXF_RING
    t = xe._view_tables("xenium")[0]
    mixed = xe._ctl_get(t, "xenium", "txf_mixed")
    dom = xe._ctl_get(t, "xenium", "txf_dominant")
    share1 = xe._ctl_get(t, "xenium", "txf_share_F1")
    for i in range(len(df)):
        m = np.zeros(2)
        for dx, dy in [(0.0, 0.0)] + [(np.cos(a), np.sin(a)) for a in np.arange(6) * np.pi / 3]:
            r, c = xe._txf_bin_index([xy[i, 0] + dx * rad[i]], [xy[i, 1] + dy * rad[i]], 10.0, shape)
            j = keys.get((int(r[0]), int(c[0])))
            if j is not None:
                m += smp["W"][j]
        p = m / m.sum()
        ent = -(p[p > 0] * np.log(p[p > 0])).sum() / np.log(2)
        assert mixed[i] == pytest.approx(ent, abs=1e-4) and share1[i] == pytest.approx(p[0], abs=1e-4)
        assert dom[i] == res["names"][int(p.argmax())]
    # cells away from the border sit over one program; the cross-tab says so
    far = np.abs(xy[:, 0] - f["mid"]) > 25
    want = np.where(xy[:, 0] < f["mid"], "L", "R")
    got = side[np.array([res["names"].index(d) for d in dom])]
    assert (got[far] == want[far]).all() and np.nanmax(mixed[far]) < 0.5
    ct = res["crosstab"]
    assert ct.to_numpy().sum() == len(df) and set(ct.columns) == {"Left", "Right"}
    assert res["agreement"]["purity"] >= far.mean() and f"Dominant factor × cell type" in msg
    fr = res["factors"].set_index("factor")
    assert set(fr["top_type_of_cells"]) == {"Left", "Right"} and (fr["cosine"] > 0).all()
    assert fr["share_tx"].sum() == pytest.approx(1.0) and fr["n_bins_dominant"].sum() == smp["n_dom"]
    # figure, table, CSV, poll
    fig, recs, opts, val = xe.update_txf_results(1, 0, None, None, None)
    assert fig.data[0].type == "heatmap" and len(recs) == 2 and recs[0]["factor"] == "F1"
    assert [o["value"] for o in opts] == ["@txf:dominant", "@txf:0", "@txf:1"] and val is dash.no_update
    csv = xe.download_txfactor(1)["content"]
    assert "in the spirit of FICTURE (not that tool)" in csv and "# top genes per factor" in csv
    assert "# cells by dominant factor" in csv and f"{shape}" in csv
    out = xe.poll_txf(1, 0, 3)
    assert out[1] is False and out[2] == xe._txf_state["version"] and out[3] is True and out[4] == 4
    assert xe.poll_txf(2, out[2], 4)[2] is dash.no_update
    # the columns are exported by Save as SpatialData (ctl_ prefix)
    obs = xe._ctl_obs_columns(df, "xenium")
    assert {"ctl_txf_dominant", "ctl_txf_mixed", "ctl_txf_top_share", "ctl_txf_share_F1", "ctl_txf_share_F2"} <= set(obs.columns)


def test_color_by_values_and_options(xe, D, tmp_path, monkeypatch):
    assert xe._txf_color_options() == [] and xe._txf_color("txf#mixed") is None
    _two_sided(xe, D, tmp_path, monkeypatch)
    _run(xe, 1)
    opts = [o["value"] for o in xe._txf_color_options()]
    assert opts == ["geneset:txf#mixed", "geneset:txf#top", "geneset:txf#share:0", "geneset:txf#share:1"]
    assert set(opts) <= {o["value"] for o in xe._view_color_extra_options()}
    t = xe._view_tables("xenium")[0]
    for name, col in (("txf#mixed", "txf_mixed"), ("txf#top", "txf_top_share"), ("txf#share:1", "txf_share_F2")):
        vals, rng_, label = xe._geneset_color("geneset:" + name, "xenium")[:3]
        assert np.allclose(vals, xe._ctl_get(t, "xenium", col), equal_nan=True) and label.startswith("Factors:")
    assert xe._txf_color("txf#share:7") is None and xe._txf_color("txf#nope") is None
    fig = xe.make_spatial_fig("geneset:txf#mixed", None, None, 4, 0.8, [], None)
    assert np.allclose(np.asarray(tl.marker_traces(fig)[0].marker.color, dtype=float),
                       xe._ctl_get(t, "xenium", "txf_mixed"), equal_nan=True)


def test_map_layer_goes_through_the_density_layer_of_both_renderers(xe, D, tmp_path, monkeypatch):
    assert xe._density_virtual("@txf:dominant") is not None and xe._density_virtual("ACTB") is None
    assert xe._density_grid("@txf:dominant", 10) is None
    assert xe.update_density_store(["on"], "@txf:dominant", 10, 0.7, None, 0, 0)[0]["on"] is False
    _two_sided(xe, D, tmp_path, monkeypatch)
    _run(xe, 1)
    res = xe._txf_result()
    smp = res["samples"][0]
    # square bins are the grid: every bin's value at its own place
    g = xe._density_grid("@txf:1", 10)
    assert g["bin"] == 10 and g["grid"].dtype == np.float32
    assert np.allclose(g["grid"][smp["r"] - g["iy0"], smp["c"] - g["ix0"]], smp["W"][:, 1])
    assert g["grid"].sum() == pytest.approx(smp["W"][:, 1].sum(), rel=1e-4)
    dom = xe._density_grid("@txf:dominant", 10)
    want = np.where(smp["n_tx"] >= 5, smp["W"].argmax(axis=1) + 1, 0)
    assert np.array_equal(dom["grid"][smp["r"] - dom["iy0"], smp["c"] - dom["ix0"]], want)
    assert dom["palette"] == [xe.CLUSTER_COLORS[0], xe.CLUSTER_COLORS[1]] and dom["names"] == ["F1", "F2"]
    assert xe._density_grid("@txf:9", 10) is None and xe._density_grid("@txf:x", 10) is None
    # the dropdown of the density tab offers the layers; picking one in the factors tab selects it there
    opts, _ = xe.update_density_gene_options(0, 1, None)
    assert [o["value"] for o in opts[:4]] == ["__all__", "@txf:dominant", "@txf:0", "@txf:1"]
    assert xe.set_txf_layer("@txf:dominant", None) == ("@txf:dominant", ["on"])
    assert xe.set_txf_layer(None, "@txf:dominant")[0] is None
    with pytest.raises(dash.exceptions.PreventUpdate):
        xe.set_txf_layer(None, "ACTB")
    store, hint = xe.update_density_store(["on"], "@txf:dominant", 10, 0.7, None, 0, 0, 0, 1)
    assert store["on"] is True and store["version"].endswith("|f1") and "Dominant factor per 10 µm bin" in hint
    assert "not a gene's transcript grid" in xe.run_density(1, "@txf:0")[2]
    # Plotly: a heat layer under the cells, categorical colours for the dominant factor
    fig = xe.make_spatial_fig("cluster", None, None, 4, 0.8, [], None, density=store)
    hm = fig.data[0]
    assert hm.type == "heatmap" and hm.zmin == 0.5 and hm.zmax == 2.5
    assert {c[1] for c in hm.colorscale} == set(dom["palette"]) and list(hm.colorbar.ticktext) == ["F1", "F2"]
    z = np.asarray(hm.z, dtype=float)
    assert set(np.unique(z[np.isfinite(z)])) <= {1.0, 2.0}
    cont = xe.make_spatial_fig("cluster", None, None, 4, 0.8, [], None, density={**store, "gene": "@txf:0"}).data[0]
    assert cont.zmin == 0 and cont.zmax == pytest.approx(xe._density_grid("@txf:0", 10)["vmax"])
    # WebGL: the density endpoint serves the same grid, with the palette
    prov = xe._viewer_provider
    head, arrs = prov.extra("density", {"gene": "@txf:dominant", "bin": "10", "x0": "-1e6", "y0": "-1e6",
                                        "x1": "1e6", "y1": "1e6"})
    assert head["ready"] and head["palette"] == dom["palette"] and head["names"] == ["F1", "F2"]
    assert head["label"].startswith("Dominant factor per 10 µm bin")
    assert np.array_equal(arrs["values"].reshape(head["h"], head["w"]), dom["grid"])
    head, arrs = prov.extra("density", {"gene": "@txf:0", "bin": "10", "x0": "-1e6", "y0": "-1e6",
                                        "x1": "1e6", "y1": "1e6"})
    assert "palette" not in head and head["label"] == "Factor F1: transcripts per 10 µm bin"
    # a categorical grid is not averaged when it is coarsened
    win = xe._density_window("@txf:dominant", 10, max_cells=40)
    assert win["coarsened"] and set(np.unique(win["values"])) <= {0.0, 1.0, 2.0}
    # a new run replaces the layer
    _run(xe, 2, k=3)
    assert xe._density_grid("@txf:2", 10) is not None and len(xe._density_grid("@txf:dominant", 10)["palette"]) == 3


def test_hexagons_are_drawn_on_a_raster_of_their_own_values(xe, D, tmp_path, monkeypatch):
    _two_sided(xe, D, tmp_path, monkeypatch)
    _run(xe, 1, shape="hex", bin_um=12)
    res = xe._txf_result()
    smp = res["samples"][0]
    g = xe._density_grid("@txf:0", 10)
    assert g["bin"] == pytest.approx(3.0)                            # a quarter of the hexagon size
    h, w = g["grid"].shape
    rng = np.random.default_rng(0)
    keys = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(smp["r"], smp["c"]))}
    for iy, ix in zip(rng.integers(0, h, 300), rng.integers(0, w, 300)):
        r, c = xe._txf_bin_index([(ix + g["ix0"] + 0.5) * g["bin"]], [(iy + g["iy0"] + 0.5) * g["bin"]], 12.0, "hex")
        j = keys.get((int(r[0]), int(c[0])))
        assert g["grid"][iy, ix] == pytest.approx(0.0 if j is None else smp["W"][j, 0])
    assert "hexagon" in xe._txf_layer_label("@txf:0")


# ── 5. reference mode, cache, guards ─────────────────────────────────────────
def test_reference_mode_projects_onto_the_type_profiles(xe, D, labels):
    if not os.path.exists(os.path.join(D["data_dir"], "transcripts.parquet")):
        pytest.skip("bundle without transcripts.parquet")
    lab = labels.copy()
    lab[:] = np.where(np.arange(len(lab)) % 2 == 0, "Alpha", "Beta")
    with xe._annot_lock:
        xe._annot_state["labels_celltypist"] = lab
    monkeypatch_min = xe._TXF_REF_MIN_CELLS
    xe._TXF_REF_MIN_CELLS = 5
    try:
        msg = _run(xe, 1, method="reference", k=99)
    finally:
        xe._TXF_REF_MIN_CELLS = monkeypatch_min
    res = xe._txf_result()
    assert res["names"] == ["Alpha", "Beta"] and res["params"]["k"] == 2 and res["info"]["n_fit"] == 0
    assert "Reference components: the mean profiles of 2 cell types" in msg and "not fitted" in msg
    genes = res["genes"]
    X = D["expr"].tocsr()
    rows = np.asarray(D.get("df_to_expr") if D.get("df_to_expr") is not None else np.arange(len(D["df"])))
    cols = [D["gene_name_to_idx"][g] for g in genes]
    for j, name in enumerate(res["names"]):
        m = (lab.reindex(D["df"].index.astype(str)) == name).to_numpy()
        prof = np.asarray(X[rows[m]][:, cols].sum(axis=0)).ravel()
        assert np.allclose(res["H"][j], prof / prof.sum(), atol=1e-6)
    smp = res["samples"][0]
    assert np.allclose(smp["W"].sum(axis=1), smp["n_tx"], rtol=1e-3)
    t = xe._view_tables("xenium")[0]
    assert xe._ctl_get(t, "xenium", "txf_share_Alpha") is not None
    assert "have their own type as the dominant factor" in msg
    # without labels the mode is refused before a job starts
    out = xe.run_txfactor(2, "reference", 5, 10, "square", 5, 20000, 0, None, "xenium", None, "original")
    assert out[0] is False and "pick an annotation" in out[2]


def test_a_second_run_with_the_same_settings_comes_from_the_cache(xe, D, tmp_path, monkeypatch):
    _two_sided(xe, D, tmp_path, monkeypatch)
    _run(xe, 1)
    a = xe._txf_result()
    files = sorted(os.listdir(xe._txf_cache_dir()))
    assert len([f for f in files if f.startswith("fit_")]) == 1 and len([f for f in files if f.startswith("counts__")]) == 1
    assert not [f for f in files if ".tmp" in f]
    monkeypatch.setattr(xe, "_txf_fit", lambda *a, **k: pytest.fail("refitted although cached"))
    monkeypatch.setattr(xe, "_txf_scan", lambda *a, **k: pytest.fail("rescanned although cached"))
    msg = _run(xe, 2)
    b = xe._txf_result()
    assert b["from_cache"] is True and "loaded from the cache" in msg
    assert np.array_equal(a["H"], b["H"]) and np.array_equal(a["samples"][0]["W"], b["samples"][0]["W"])
    assert a["crosstab"].equals(b["crosstab"])
    monkeypatch.undo()
    # another seed, K or bin is another fit (the counts of the same bins are reused)
    _two_sided(xe, D, tmp_path, monkeypatch, name="two")
    monkeypatch.setattr(xe, "_txf_scan", lambda *a, **k: pytest.fail("rescanned although cached"))
    assert "loaded from the cache" not in _run(xe, 3, seed=1)
    assert len([f for f in os.listdir(xe._txf_cache_dir()) if f.startswith("fit_")]) == 2


def test_guards_and_empty_states(xe, D, tmp_path, monkeypatch):
    for fn, args in ((xe.run_txfactor, (0, "nmf", 5, 10, "square", 5, 20000, 0, None, "xenium", None, "original")),
                     (xe.run_txfactor, (None, "nmf", 5, 10, "square", 5, 20000, 0, None, "xenium", None, "original")),
                     (xe.download_txfactor, (0,)), (xe.download_txfactor, (1,)),
                     (xe.update_txf_label_options, ("tab-umap", 0, 0, "xenium", 0))):
        with pytest.raises(dash.exceptions.PreventUpdate):
            fn(*args)
    assert "at least 2" in xe.run_txfactor(1, "nmf", 1, 10, "square", 5, 20000, 0, None, "xenium", None, "original")[2]
    assert "must be positive" in xe.run_txfactor(1, "nmf", 4, 0, "square", 5, 20000, 0, None, "xenium", None, "original")[2]
    assert xe._txf_state["status"] == "idle" and not [j for j in xe._jobs.values() if j.kind == "txfactor"]
    fig, recs, opts, _v = xe.update_txf_results(0, 0, None, None, "@txf:0")
    assert recs == [] and opts == [] and "No transcript factors" in fig.layout.annotations[0].text and _v is None
    opts, val = xe.update_txf_label_options("tab-txf", 0, 0, "xenium", 0, "cluster", None)
    assert {"cluster"} <= {o["value"] for o in opts} and "niche" not in {o["value"] for o in opts}
    # too few bins with enough transcripts
    rng = np.random.default_rng(8)
    genes = list(D["gene_names"])
    d = _write_tx(tmp_path / "few", rng.uniform(0, 100, 300), rng.uniform(0, 100, 300), rng.choice(genes[:30], 300))
    monkeypatch.setitem(xe.DATA, "data_dir", d)
    xe.run_txfactor(1, "nmf", 8, 10, "square", 10, 20000, 0, None, "xenium", None, "original")
    with pytest.raises(AssertionError, match="too few to fit 8 factors"):
        tl.wait_state(xe._txf_state, xe._txf_lock)
    xe._test_baseline.settle()
    # a dataset loaded from a .zarr has no transcript file
    monkeypatch.setitem(xe.DATA, "data_dir", None)
    monkeypatch.setitem(xe.DATA, "sdata_path", "/somewhere/sample.zarr")
    out = xe.run_txfactor(2, "nmf", 4, 10, "square", 5, 20000, 0, None, "xenium", None, "original")
    assert out[0] is False and "loaded from a .zarr" in out[2]
    assert xe._txf_result() is None and xe._txf_layer_options() == []


def test_layout_has_the_tab_and_the_note_names_what_this_is(xe):
    ids = set()

    def walk(c, inside):
        cid = getattr(c, "id", None)
        inside = inside or getattr(c, "tab_id", None) == "tab-txf"
        if cid and inside:
            ids.add(cid)
        ch = getattr(c, "children", None)
        for k in (ch if isinstance(ch, (list, tuple)) else [ch]):
            if hasattr(k, "to_plotly_json"):
                walk(k, inside)

    walk(xe.app.layout, False)
    assert {"txf-method", "txf-k", "txf-bin", "txf-shape", "txf-mintx", "txf-maxfit", "txf-seed", "txf-run-btn",
            "txf-labels", "txf-layer", "txf-status", "txf-note", "txf-plot", "txf-table", "txf-download-btn",
            "txf-download", "txf-done", "txf-poll", "txf-plot-w", "txf-plot-h"} <= ids
    assert xe._JOB_KIND_NAMES["txfactor"] == "transcript factors"
    assert "in the spirit of FICTURE" in xe._TXF_NOTE and "not that tool" in xe._TXF_NOTE


# ── 6. several samples ───────────────────────────────────────────────────────
def _extra(D, tmp_path, i, with_tx=True, flip=False):
    """A fabricated extra sample: 120 cells on a 300 µm field; genes 0-19 on its left half
    and 20-39 on its right half (the other way round with *flip*)."""
    rng = np.random.default_rng(200 + i)
    n, n_tx = 120, 30_000
    genes = list(D["gene_names"])
    df = pd.DataFrame({"x_centroid": rng.uniform(0, 300, n), "y_centroid": rng.uniform(0, 300, n),
                       "cell_area": np.full(n, 60.0)}, index=[f"e{i}_{k}" for k in range(n)])
    m = sp.csr_matrix((rng.poisson(1.0, size=(n, len(genes)))).astype(np.float32))
    ds = {"expr": m, "df_to_expr": np.arange(n), "gene_name_to_idx": D["gene_name_to_idx"], "gene_names": genes,
          "df": df, "x_offset": 5000.0 * (i + 1), "cluster_methods": []}
    if with_tx:
        x, y = rng.uniform(0, 300, n_tx), rng.uniform(0, 300, n_tx)
        left = (x < 150) != flip
        g = np.where(left, rng.integers(0, 20, n_tx), rng.integers(20, 40, n_tx))
        ds["data_dir"] = _write_tx(tmp_path / f"extra{i}", x, y, np.array(genes, dtype=object)[g])
    else:
        ds["sdata_path"] = str(tmp_path / f"extra{i}.zarr")
    return ds


def test_samples_are_binned_on_their_own_grids_and_share_the_loadings(xe, D, tmp_path, monkeypatch):
    _two_sided(xe, D, tmp_path, monkeypatch)
    extras = [_extra(D, tmp_path, 0), _extra(D, tmp_path, 1, flip=True), _extra(D, tmp_path, 2, with_tx=False)]
    xe.EXTRA_DATASETS.extend(extras)
    msg = _run(xe, 1, labels=None)
    res = xe._txf_result()
    assert [s["label"] for s in res["samples"]] == ["S0", "S1", "S2"] and res["skipped"] == ["S3"]
    assert "Left out (loaded from a .zarr, no transcript file): S3" in msg and "in 3 sample(s)" in msg
    assert res["crosstab"].size == 0 and "Dominant factor ×" not in msg
    assert xe.make_txf_fig().data[0].type == "bar"
    # every sample has its own bins (its own µm) and its own cache; the loadings are shared
    for s, e in zip(res["samples"][1:], extras[:2]):
        assert s["total"] == 30_000 and s["r"].min() >= 0 and s["r"].max() <= 29 and s["c"].max() <= 29
        assert os.path.exists(xe._txf_counts_file(e, res["genes"], 10.0, "square"))
    left_factor = int(np.argmax(res["H"][:, :20].sum(axis=1)))
    for s, flip in zip(res["samples"][1:], (False, True)):
        on_left = s["c"] < 14
        frac = (s["W"].argmax(axis=1)[on_left] == left_factor).mean()
        assert (frac < 0.05) if flip else (frac > 0.95)
    # the cells of every sample are scored against that sample's bins
    for t, e, flip in zip(xe._view_tables("xenium")[1:3], extras[:2], (False, True)):
        dom = xe._ctl_get(t, "xenium", "txf_dominant")
        x = e["df"]["x_centroid"].to_numpy()
        far = np.abs(x - 150) > 20
        is_left = (dom == res["names"][left_factor])
        assert (is_left[far] == ((x[far] < 150) != flip)).all()
    assert xe._ctl_get(xe._view_tables("xenium")[3], "xenium", "txf_dominant") is None
    vals = xe._txf_color("txf#mixed", "xenium")[0]
    assert len(vals) == sum(len(t["df"]) for t in xe._view_tables("xenium")) and np.isnan(vals[-120:]).all()
    # the layer: one grid per sample at its offset
    wins = xe._density_windows("@txf:dominant", 10)
    assert wins["labels"] == ["S0", "S1", "S2"] and wins["unavailable"] == ["S3"] and wins["palette"]
    assert [round(p["x0"] - (p["x0"] % 5000)) for p in wins["parts"]][1:] == [5000, 10000]


# ── 7. cancel ────────────────────────────────────────────────────────────────
def test_job_can_be_cancelled_publishes_nothing_and_reruns(xe, D, tmp_path, monkeypatch):
    _two_sided(xe, D, tmp_path, monkeypatch, n_tx=20_000)
    real = xe._txf_scan
    seen = []

    def slow_scan(*a, **k):
        for i in range(400):
            seen.append(i)
            time.sleep(0.02)
            xe._job_check()
        return real(*a, **k)

    monkeypatch.setattr(xe, "_txf_scan", slow_scan)
    args = ("nmf", 2, 10, "square", 5, 20000, 0, None, "xenium", None, "original")
    assert xe.run_txfactor(1, *args)[:2] == (True, False)
    job = [j for j in xe._jobs.values() if j.kind == "txfactor"][-1]
    while not seen:
        time.sleep(0.01)
    assert xe.run_txfactor(2, *args)[2].endswith("already being computed.")
    assert job.snapshot()["cancellable"] and xe._job_cancel(job.id)
    job.thread.join(10)
    assert job.state == "cancelled" and len(seen) < 400
    assert xe._txf_state["status"] == "error" and xe._txf_state["message"] == "Cancelled"
    assert xe._txf_result() is None and not os.path.exists(xe._txf_cache_dir())
    assert "transcript factors" not in xe._running_jobs()
    assert xe.poll_txf(1, 0, 0)[:2] == ("Cancelled", False)
    monkeypatch.setattr(xe, "_txf_scan", real)
    assert xe.run_txfactor(3, *args)[:2] == (True, False)
    tl.wait_state(xe._txf_state, xe._txf_lock)
    assert xe._txf_result()["samples"][0]["total"] == 20_000
