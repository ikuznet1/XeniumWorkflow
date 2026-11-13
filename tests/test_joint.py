"""Joint clustering across samples: normalisation, gene selection and PCA against dense
formulas, the Harmony step against harmonypy run directly, the diagnostics by hand, and
the job on fabricated samples with planted cell types and a planted batch shift (cluster
set in every sample, joint UMAP in the UMAP panel, cache, no-harmonypy path, cancel).
"""
import os
import shutil
import threading

import dash
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import xe_testlib as tl

N_GENES = 240
TYPES = ["T0", "T1", "T2", "T3"]


def _have(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


HAVE_HARMONY = _have("harmonypy")
needs_harmony = pytest.mark.skipif(not HAVE_HARMONY, reason="harmonypy is not installed")


@pytest.fixture(autouse=True)
def _clean_joint_state(xe):
    yield
    with xe._jc_lock:
        xe._jc_state.update({"status": "idle", "message": "", "result": None})
    for name, _prefix in xe._JC_VARIANTS.values():
        xe._cs_remove(name)
    xe._controls_reset()
    xe._stats_reset()
    shutil.rmtree(xe._jc_cache_dir(), ignore_errors=True)
    for f in os.listdir(xe.CACHE_BASE_DIR):
        if f.startswith("design_") and f.endswith(".json"):
            os.remove(os.path.join(xe.CACHE_BASE_DIR, f))
        if f.startswith("clustersets_"):
            shutil.rmtree(os.path.join(xe.CACHE_BASE_DIR, f), ignore_errors=True)


# ── fabricated samples: planted cell types, planted batch shift ──────────────
def _fab(seed, n, shift=0.5, props=None, n_genes=N_GENES, key=None):
    """One sample: *n* cells of four types (30 marker genes each, 6× over the base), every
    gene multiplied by a sample-specific factor exp(N(0, shift)) - the batch shift.
    Returns (dataset dict as '+ Add sample' builds it, type of every cell)."""
    rng = np.random.default_rng(seed)
    base = np.random.default_rng(99).gamma(2.0, 0.6, size=n_genes) + 0.2      # same in every sample
    prog = np.ones((len(TYPES), n_genes))
    for t in range(len(TYPES)):
        prog[t, 30 * t:30 * (t + 1)] = 6.0
    typ = rng.choice(len(TYPES), size=n, p=props)
    batch = np.exp(rng.normal(0.0, shift, size=n_genes))
    lib = rng.gamma(10.0, 1 / 10.0, size=n)
    dense = rng.poisson(lib[:, None] * base[None, :] * prog[typ] * batch[None, :] * 0.5)
    genes = [f"g{i}" for i in range(n_genes)]
    df = pd.DataFrame({"x_centroid": rng.uniform(0, 300, n), "y_centroid": rng.uniform(0, 300, n),
                       "transcript_counts": dense.sum(axis=1)}, index=[f"s{seed}_{i}" for i in range(n)])
    ds = {"expr": sp.csr_matrix(dense.astype(np.float32)), "df_to_expr": np.arange(n),
          "gene_name_to_idx": {g: i for i, g in enumerate(genes)}, "gene_names": genes, "df": df,
          "x_offset": 1000.0 * seed, "cluster_methods": [], "data_dir": key or f"/fake/joint{seed}"}
    return ds, np.array(TYPES, dtype=object)[typ]


def _samples_of(dsets):
    """analysis._an_samples-like dicts and stats._view_tables-like dicts of fabricated datasets."""
    samples, tables = [], []
    for i, ds in enumerate(dsets):
        key = ds["data_dir"]
        base = {"label": f"S{i}", "df": ds["df"], "ds": ds, "key": key, "name": os.path.basename(key),
                "x_offset": ds["x_offset"]}
        tables.append(dict(base))
        samples.append({**base, "csc": ds["expr"].tocsc(), "rows": np.arange(len(ds["df"])),
                        "gni": ds["gene_name_to_idx"], "genes": list(ds["gene_names"]), "corrected": False})
    return samples, tables


def _ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(a, b)


@pytest.fixture
def world(xe, monkeypatch):
    """Three fabricated samples as the whole view (the bundle is not part of it)."""
    made = [_fab(seed, n) for seed, n in ((1, 420), (2, 380), (3, 400))]
    dsets = [m[0] for m in made]
    truth = [m[1] for m in made]

    class View:
        pass

    v = View()
    v.dsets, v.truth = dsets, truth

    def _tables(seg_source="xenium"):
        return _samples_of(v.dsets)[1]

    def _samples(seg_source="xenium", use_corrected=False):
        # the tables always hold the current df of each dataset (a cluster set replaces it)
        return _samples_of(v.dsets)[0]

    monkeypatch.setattr(xe, "_view_tables", _tables)
    monkeypatch.setattr(xe, "_an_samples", _samples)
    monkeypatch.setattr(xe, "_design_samples", lambda: [
        {"index": i, "label": f"S{i}", "key": d["data_dir"], "name": os.path.basename(d["data_dir"]),
         "n_cells": len(d["df"]), "cond": "", "batch": "", "donor": ""} for i, d in enumerate(v.dsets)])
    return v


def _run(xe, params=None, seg="xenium", corrected=False):
    with xe._jc_lock:
        xe._jc_state.update({"status": "running", "message": ""})
    xe._run_joint(seg, corrected, params)
    with xe._jc_lock:
        return dict(xe._jc_state)


# ── the numerical steps ──────────────────────────────────────────────────────
def test_normalise_is_log1p_counts_per_10k(xe):
    rng = np.random.default_rng(0)
    d = rng.poisson(0.8, size=(50, 30)).astype(np.float32)
    d[3] = 0                                              # an empty cell stays zero
    got = xe._jc_normalise(sp.csr_matrix(d)).toarray()
    tot = d.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1
    assert np.allclose(got, np.log1p(d / tot * 1e4), rtol=1e-5)
    assert not got[3].any()


def test_moments_and_dispersion_score(xe):
    rng = np.random.default_rng(1)
    d = rng.gamma(1.0, 1.0, size=(200, 60)) * (rng.random((200, 60)) < 0.4)
    m, v = xe._jc_moments(sp.csr_matrix(d))
    assert np.allclose(m, d.mean(axis=0)) and np.allclose(v, d.var(axis=0, ddof=1))
    z = xe._jc_dispersion_z(m, v, n_bins=4)
    # by hand: log(v / m), standardised within 4 quantile bins of the mean
    disp = np.log(v / m)
    order = np.argsort(np.argsort(m, kind="stable"), kind="stable")
    b = order * 4 // 60
    want = np.array([(disp[i] - disp[b == b[i]].mean()) / disp[b == b[i]].std() for i in range(60)])
    assert np.allclose(z, want)
    # a gene nobody expresses cannot be selected
    assert xe._jc_dispersion_z(np.array([0.0, 1.0]), np.array([0.0, 1.0]))[0] == -np.inf


def test_gene_selection_prefers_within_sample_variation(xe):
    """Genes that vary between cell types inside every sample are selected; a gene that is
    flat within each sample and differs only BETWEEN samples is not."""
    rng = np.random.default_rng(2)
    n, G = 900, 400
    sample = np.repeat([0, 1, 2], n // 3)
    typ = rng.integers(0, 2, n)
    base = rng.gamma(2.0, 1.0, size=G) + 0.3                      # a spread of expression levels
    lam = np.tile(base, (n, 1))
    lam[:, :10] *= np.where(typ[:, None] == 1, 5.0, 0.2)          # planted: genes 0-9
    lam[:, 10] = np.array([0.3, 3.0, 12.0])[sample]               # batch-only gene
    Xn = xe._jc_normalise(sp.csr_matrix(rng.poisson(lam).astype(np.float32)))
    sel = xe._jc_select_genes(Xn, sample, n_top=10)
    assert sorted(sel.tolist()) == list(range(10))
    # pooled over the samples the batch-only gene is in the most dispersed quarter
    m, v = xe._jc_moments(Xn)
    pooled_rank = np.argsort(-np.log(v / m)).tolist().index(10)
    assert pooled_rank < G // 4
    # at most n_top genes exist: all genes with variance, none dropped
    assert len(xe._jc_select_genes(Xn, sample, n_top=500)) == G


def test_scale_and_pca_equal_the_dense_computation(xe):
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(3)
    lam = rng.gamma(1.0, 1.0, size=(1, 80)) * np.where(rng.random((400, 1)) < 0.5, 1.0, 3.0)
    d = rng.poisson(lam).astype(np.float32)
    d[0, 0] = 400.0                                               # an outlier that the cap clips
    Xn = xe._jc_normalise(sp.csr_matrix(d))
    Xs, sd = xe._jc_scale(Xn)
    dense = Xn.toarray().astype(np.float64)
    mu, s = dense.mean(axis=0), dense.std(axis=0, ddof=1)
    want = np.minimum((dense - mu) / s, xe._JC_SCALE_CLIP) + mu / s      # the app keeps the matrix uncentred
    assert np.allclose(sd, s, rtol=1e-4)
    assert np.allclose(Xs.toarray()[dense > 0], want[dense > 0], rtol=1e-3, atol=1e-4)
    assert ((dense - mu) / s).max() > xe._JC_SCALE_CLIP and (Xs.toarray() - mu / s).max() <= xe._JC_SCALE_CLIP + 1e-3
    Z, evr = xe._jc_pca(Xs, n_pcs=8, seed=0)
    ref = PCA(n_components=8, svd_solver="full").fit(Xs.toarray().astype(np.float64))
    Zr = ref.transform(Xs.toarray().astype(np.float64))
    assert Z.shape == (400, 8) and Z.dtype == np.float32
    assert np.allclose(evr, ref.explained_variance_ratio_, rtol=1e-3)
    for j in range(4):                                            # well separated components: same up to sign
        c = np.corrcoef(Z[:, j], Zr[:, j])[0, 1]
        assert abs(c) > 0.9999
    assert np.array_equal(Z, xe._jc_pca(Xs, n_pcs=8, seed=0)[0])   # deterministic


def test_entropy_by_hand(xe):
    labels = np.array([1] * 4 + [2] * 4 + [3] * 4)
    batch = np.array(list("aabb") + list("aaaa") + list("abbb"))
    e = xe._jc_entropy(labels, batch)
    h3 = -(0.25 * np.log(0.25) + 0.75 * np.log(0.75)) / np.log(2)
    assert np.allclose(e["entropy"], [1.0, 0.0, h3])
    assert np.isclose(e["mean"], (1.0 + 0.0 + h3) / 3)
    p = np.array([7, 5]) / 12
    assert np.isclose(e["reference"], -(p * np.log(p)).sum() / np.log(2))
    assert e["counts"].tolist() == [[2, 2], [4, 0], [1, 3]]


def test_mixing_index_separated_and_mixed(xe):
    rng = np.random.default_rng(4)
    n = 600
    batch = np.repeat(["a", "b", "c"], n // 3)
    mixed = rng.normal(size=(n, 5))
    apart = mixed + np.array([0, 50, 100])[np.repeat([0, 1, 2], n // 3)][:, None]
    m_mixed = xe._jc_mixing(mixed, batch, k=30, n_query=600)
    m_apart = xe._jc_mixing(apart, batch, k=30, n_query=600)
    assert m_apart["median"] == 1.0 and np.isclose(m_apart["reference"], 3.0)
    assert 2.5 < m_mixed["median"] <= 3.0
    # one cell by brute force
    q = 17
    d = ((mixed - mixed[q]) ** 2).sum(axis=1)
    d[q] = np.inf
    nb = np.argsort(d)[:30]
    p = np.array([(batch[nb] == b).mean() for b in "abc"])
    one = xe._jc_mixing(mixed, batch, k=30, n_query=n)       # all cells are queried, in index order
    emb = mixed.astype(np.float32)
    assert one["n"] == n
    full = []
    for i in range(n):
        dd = ((emb - emb[i]) ** 2).sum(axis=1)
        dd[i] = np.inf
        pb = np.array([(batch[np.argsort(dd)[:30]] == b).mean() for b in "abc"])
        full.append(1.0 / (pb ** 2).sum())
    assert np.isclose(one["mean"], np.mean(full), rtol=1e-3)
    assert np.isclose(full[q], 1.0 / (p ** 2).sum(), rtol=1e-3)


def test_levels_and_confounding(xe):
    tables = [{"label": f"S{i}", "name": f"n{i}"} for i in range(4)]
    assert xe._jc_levels("sample", tables) == ["S0 n0", "S1 n1", "S2 n2", "S3 n3"]
    lv = xe._jc_levels("donor", tables, donors=["d1", "d1", "", "d2"])
    assert lv == ["d1", "d1", "(sample S2 n2)", "d2"]
    assert xe._jc_levels("batch", tables, batches=["b", "b", "c", "c"]) == ["b", "b", "c", "c"]
    # every level in one condition: fully confounded; a level with both: partially; no design: None
    assert xe._jc_confounded(["A", "A", "B", "B"], ["x", "x", "y", "z"]) == "full"
    assert xe._jc_confounded(["A", "B", "A", "B"], ["x", "x", "y", "y"]) == "partial"
    assert xe._jc_confounded(["", "", "", ""], ["x", "x", "y", "y"]) is None
    assert xe._jc_confounded(["A", "A", "", ""], ["x", "x", "y", "y"]) is None


# ── Harmony ──────────────────────────────────────────────────────────────────
@needs_harmony
def test_harmony_step_equals_harmonypy_on_the_same_pcs(xe):
    import harmonypy
    rng = np.random.default_rng(5)
    n = 900
    batch = np.repeat(["a", "b", "c"], n // 3)
    pcs = rng.normal(size=(n, 12)).astype(np.float32)
    pcs[batch == "b", 0] += 3.0
    pcs[batch == "c", 1] -= 3.0
    got = xe._jc_harmony(pcs, batch, seed=7)
    ho = harmonypy.run_harmony(pcs, pd.DataFrame({"level": batch}), ["level"], random_state=7, verbose=False)
    ref = np.asarray(ho.Z_corr)
    ref = ref if ref.shape == pcs.shape else ref.T
    assert got.shape == pcs.shape
    assert np.array_equal(got, ref.astype(np.float32))
    # the shift is gone, and another seed is another run of the same thing
    assert abs(pcs[batch == "b", 0].mean() - pcs[batch == "a", 0].mean()) > 2.5
    assert abs(got[batch == "b", 0].mean() - got[batch == "a", 0].mean()) < 1.0
    assert np.array_equal(got, xe._jc_harmony(pcs, batch, seed=7))


@needs_harmony
def test_planted_batch_shift_is_removed_and_types_recovered(xe):
    """Three samples, four planted cell types, a per-gene shift per sample. Without
    correction the clusters follow the samples; with Harmony they follow the types."""
    made = [_fab(seed, n) for seed, n in ((1, 420), (2, 380), (3, 400))]
    samples, tables = _samples_of([m[0] for m in made])
    truth = np.concatenate([m[1] for m in made])
    levels = xe._jc_levels("sample", tables)
    p = xe._jc_params({"integrate": "sample", "umap": False, "n_pcs": 20})
    plain = xe._jc_compute(samples, levels, p, harmony=False)
    harm = xe._jc_compute(samples, levels, p, harmony=True)
    assert np.array_equal(plain["pcs"], harm["pcs"])              # the same joint PCA
    assert np.array_equal(plain["emb"], plain["pcs"]) and not np.array_equal(harm["emb"], harm["pcs"])
    sample = harm["sample"]
    ari_type_plain, ari_type_harm = _ari(truth, plain["labels"]), _ari(truth, harm["labels"])
    e0, e1 = plain["diag"]["after"]["entropy"], harm["diag"]["after"]["entropy"]
    m0, m1 = plain["diag"]["after"]["mixing"], harm["diag"]["after"]["mixing"]
    print(f"ARI vs planted types: uncorrected {ari_type_plain:.3f}, Harmony {ari_type_harm:.3f}; "
          f"entropy {e0['mean']:.3f} -> {e1['mean']:.3f} (reference {e1['reference']:.3f}); "
          f"mixing {m0['median']:.2f} -> {m1['median']:.2f} (reference {m1['reference']:.2f})")
    assert e0["mean"] < 0.05 and m0["median"] < 1.05              # clusters = samples
    assert _ari(sample, plain["labels"]) > _ari(truth, plain["labels"]) or ari_type_plain < 0.5
    assert ari_type_harm > 0.95
    assert e1["mean"] > 0.9 * e1["reference"] and m1["median"] > 0.8 * m1["reference"]
    # the "before" diagnostics of the Harmony run are those of the uncorrected run
    assert np.array_equal(harm["labels_before"], plain["labels"])
    assert np.isclose(harm["diag"]["before"]["entropy"]["mean"], e0["mean"])
    assert harm["diag"]["before"]["mixing"] == m0


# ── the job ──────────────────────────────────────────────────────────────────
@needs_harmony
def test_job_installs_one_shared_cluster_set_with_harmony(xe, world):
    import harmonypy
    # no UMAP here (nothing below reads it; the embedding has its own test): the fit and its
    # numba compilation were most of this test's time
    st = _run(xe, {"integrate": "sample", "n_pcs": 20, "umap": False})
    assert st["status"] == "done", st["message"]
    res = xe._jc_result()
    assert res["name"] == "Joint clusters (Harmony)" and res["harmony"] and not res["from_cache"]
    key = res["method_key"]
    # the app's corrected embedding = harmonypy run directly on the app's PCs
    batch = np.asarray(res["levels"], dtype=object)[res["sample"]].astype(str)
    ho = harmonypy.run_harmony(res["pcs"], pd.DataFrame({"level": batch}), ["level"],
                               random_state=res["params"]["seed"], verbose=False)
    ref = np.asarray(ho.Z_corr)
    assert np.array_equal(res["emb"], (ref if ref.shape == res["pcs"].shape else ref.T).astype(np.float32))
    # one set, present in every sample, with the same label vocabulary; labels follow the planted types
    col = xe.cluster_col(key)
    got = []
    for ds in world.dsets:
        assert key in ds["cluster_methods"] and col in ds["df"].columns
        got.append(ds["df"][col].to_numpy())
    assert all((g > 0).all() for g in got)
    assert _ari(np.concatenate(world.truth), np.concatenate(got)) > 0.95
    assert np.array_equal(np.concatenate(got), res["labels"])
    assert xe._cs_is_shared(key) and xe._cs_label(key) == "Joint clusters (Harmony)"
    # composition / pseudobulk read the same labels in every sample
    samples, tables = _samples_of(world.dsets)
    raw = xe._raw_cluster_labels(samples, key, None)
    assert {x for lab in raw for x in lab} == {str(c) for c in range(1, int(res["labels"].max()) + 1)}
    # diagnostics table and status
    tb = res["table"]
    assert tb["n_cells"].sum() == sum(len(d["df"]) for d in world.dsets)
    assert [c for c in tb.columns if c.startswith("n_S")] == ["n_S0", "n_S1", "n_S2"]
    assert (tb[["n_S0", "n_S1", "n_S2"]].sum(axis=1) == tb["n_cells"]).all()
    msg = st["message"]
    assert "before Harmony" in msg and "after" in msg and "Integration removes differences" in msg
    assert "harmonypy, theta 2" in msg
    assert res["diag"]["after"]["entropy"]["mean"] > res["diag"]["before"]["entropy"]["mean"] + 0.5
    # CSV and figure
    csv = xe._jc_csv(res)
    assert "Harmony (harmonypy) on the sample" in csv and "cluster,n_cells,entropy" in csv
    fig = xe.make_jc_fig()
    assert len(fig.data) == 3 + 2                                  # 3 samples stacked + before / after
    assert len(xe._jc_records(res)) == len(tb)


@needs_harmony
@pytest.mark.parametrize("real_umap", [False, pytest.param(True, marks=pytest.mark.slow)])
def test_joint_umap_columns_and_umap_panel(xe, world, client, monkeypatch, real_umap):
    """How the joint embedding reaches the per-cell columns, the UMAP panel, the WebGL viewer
    and the cache. Default tier: a stand-in embedding (the first two components, spread by
    cell), so that no UMAP is fitted; slow tier: the same checks on a real umap-learn fit."""
    if real_umap:
        if xe._jc_umap(np.random.default_rng(0).normal(size=(50, 5))) is None:
            pytest.skip("umap-learn does not import")
    else:
        def stand_in(emb, seed=0):
            emb = np.asarray(emb, dtype=np.float32)
            return emb[:, :2] + np.random.default_rng(int(seed)).normal(0, 1e-3, size=(len(emb), 2)).astype(np.float32)
        monkeypatch.setattr(xe, "_jc_umap", stand_in)
    st = _run(xe, {"integrate": "sample", "n_pcs": 20})
    assert st["status"] == "done", st["message"]
    res = xe._jc_result()
    key = res["method_key"]
    tables = xe._view_tables("xenium")
    for t in tables:
        u1 = xe._ctl_get(t, "xenium", "joint_harmony_umap_1")
        assert u1 is not None and np.isfinite(u1).all() and len(u1) == len(t["df"])
    n_all = sum(len(d["df"]) for d in world.dsets)
    # selected set → the UMAP panel shows all samples in the joint embedding
    fig = xe.make_umap_fig("cluster", key, None, 3, 0.8)
    # the plot callback passes the tool of the segmentation source: "xenium" is not a resegmentation
    assert "Joint UMAP" in xe.make_umap_fig("cluster", key, None, 3, 0.8, alt_tool="xenium").layout.title.text
    assert "Joint UMAP" not in str(xe.make_umap_fig("cluster", key, None, 3, 0.8, alt_tool="imgseg").layout.title.text)
    assert "Joint UMAP" in fig.layout.title.text and "3 samples" in fig.layout.title.text
    assert sum(len(tr.x) for tr in fig.data) == n_all
    assert len(fig.data) == int(res["labels"].max())
    assert {tr.name for tr in fig.data} == {str(c) for c in range(1, len(fig.data) + 1)}
    # a cluster's trace holds that cluster's cells at their coordinates
    tr = next(t for t in fig.data if t.name == "1")
    ids = np.concatenate([[f"S{i}:{c}" for c in d["df"].index] for i, d in enumerate(world.dsets)])
    want = set(ids[res["labels"] == 1].tolist())
    assert set(tr.text.tolist() if hasattr(tr.text, "tolist") else tr.text) == want
    xy = np.column_stack([np.concatenate([xe._ctl_get(t, "xenium", f"joint_harmony_umap_{a}") for t in tables])
                          for a in (1, 2)])
    assert np.allclose(np.sort(tr.x), np.sort(xy[res["labels"] == 1, 0]))
    # a gene colours the same embedding; anything else falls back to the sample
    by_sample = xe.make_umap_fig("transcript_counts", key, None, 3, 0.8)
    assert [t.name for t in by_sample.data] == ["S0", "S1", "S2"] and "sample" in by_sample.layout.title.text
    # another cluster set, or a resegmentation source: the panel is left alone
    assert xe._jc_umap_fig("cluster", "some_other_set", None, 3, 0.8) is None
    assert xe._jc_umap_fig("cluster", key, None, 3, 0.8, baysor_active=True) is None
    # WebGL: the UMAP viewer gets the same embedding of all samples (y down) while the set is selected
    h, a = tl.get_bin(client, f"/umap/viewer/cells.bin?source=xenium%7Cemb:{key}")
    assert h["n"] == n_all and h["samples"] == 3 and h["n_embedded"] == n_all
    assert np.allclose(a["xy"][0::2], xy[:, 0], atol=1e-4) and np.allclose(a["xy"][1::2], -xy[:, 1], atol=1e-4)
    h1, _ = tl.get_bin(client, "/umap/viewer/cells.bin?source=xenium")
    assert h1["n"] == len(xe.DATA["df"]) and h1["samples"] == 1          # the sample's own UMAP otherwise
    h2, _ = tl.get_bin(client, "/umap/viewer/cells.bin?source=xenium%7Cemb:not_a_joint_set")
    assert h2["n"] == len(xe.DATA["df"])
    prov = xe._UmapViewerProvider()
    assert prov._split(f"xenium|corr|emb:{key}") == ("xenium", True) and prov._split("xenium|corr") == ("xenium", True)
    n0 = len(world.dsets[0]["df"])
    assert prov.cell_at(f"xenium|emb:{key}", n0 + 2) == {"index": n0 + 2, "id": str(world.dsets[1]["df"].index[2]), "sample": "S1"}
    # per-cell columns are what Save as SpatialData exports for the primary sample
    assert xe._ctl_get(tables[0], "xenium", "joint_harmony_umap_2") is not None
    # after a restart the columns are gone but the set is restored: the embedding comes back from the cache
    xe._controls_reset()
    xe._jc_restore_seen.clear()
    assert xe._ctl_get(tables[0], "xenium", "joint_harmony_umap_1") is None
    view = xe._jc_umap_view(key)
    assert view is not None and view["variant"] == "harmony"
    assert np.allclose(np.concatenate(view["x"]), xy[:, 0], atol=1e-5) and np.allclose(np.concatenate(view["y"]), xy[:, 1], atol=1e-5)
    assert xe._jc_umap_view("graphclust") is None


@needs_harmony
def test_second_run_comes_from_the_cache_and_params_change_the_key(xe, world):
    st = _run(xe, {"integrate": "sample", "n_pcs": 20, "umap": False})
    assert st["status"] == "done", st["message"]
    first = xe._jc_result()
    files = sorted(os.listdir(xe._jc_cache_dir()))
    assert files == [f"{first['cache_tag']}.json", f"{first['cache_tag']}.npz"]
    xe._cs_remove(first["name"])
    st = _run(xe, {"integrate": "sample", "n_pcs": 20, "umap": False})
    second = xe._jc_result()
    assert second["from_cache"] and "from the cache" in st["message"]
    assert np.array_equal(first["labels"], second["labels"]) and np.array_equal(first["emb"], second["emb"])
    assert first["diag"]["after"]["mixing"] == second["diag"]["after"]["mixing"]
    assert np.allclose(first["table"]["entropy"], second["table"]["entropy"])
    col = xe.cluster_col(second["method_key"])
    assert np.array_equal(np.concatenate([d["df"][col].to_numpy() for d in world.dsets]), second["labels"])
    # other parameters, or other cells, are another run
    _run(xe, {"integrate": "sample", "n_pcs": 20, "umap": False, "resolution": 0.5})
    assert not xe._jc_result()["from_cache"] and len(os.listdir(xe._jc_cache_dir())) == 4
    world.dsets[0]["df"] = world.dsets[0]["df"].iloc[:-5]
    world.dsets[0]["expr"] = world.dsets[0]["expr"][:-5]
    world.dsets[0]["df_to_expr"] = world.dsets[0]["df_to_expr"][:-5]
    _run(xe, {"integrate": "sample", "n_pcs": 20, "umap": False})
    assert not xe._jc_result()["from_cache"]


def test_without_harmonypy_the_set_is_named_uncorrected_with_a_hint(xe, world, monkeypatch):
    monkeypatch.setattr(xe, "_jc_harmony_module", lambda: None)
    st = _run(xe, {"integrate": "sample", "n_pcs": 20, "umap": False})
    assert st["status"] == "done", st["message"]
    res = xe._jc_result()
    assert res["name"] == "Joint clusters (uncorrected)" and not res["harmony"] and res["variant"] == "uncorrected"
    assert "harmonypy is not installed" in st["message"] and "pip install harmonypy" in st["message"]
    assert "before Harmony" not in st["message"] and "before" not in res["diag"]
    assert np.array_equal(res["emb"], res["pcs"])
    assert "joint_clusters_uncorrected" in world.dsets[0]["cluster_methods"]
    # with the planted shift the uncorrected clusters are sample-specific, and the diagnostics say so
    assert res["diag"]["after"]["entropy"]["mean"] < 0.05
    assert (res["table"]["largest_share"] > 0.95).all()
    assert any("needs harmonypy" in o["label"] for o in xe._jc_integrate_options())
    with pytest.raises(ImportError):
        xe._jc_harmony(res["pcs"], res["sample"])
    assert "not installed" in xe._jc_csv(res)


def test_no_integration_is_named_joint_clusters(xe, world):
    st = _run(xe, {"integrate": "none", "n_pcs": 20, "umap": False})
    assert st["status"] == "done", st["message"]
    res = xe._jc_result()
    assert res["name"] == "Joint clusters" and res["variant"] == "plain" and not res["harmony"]
    assert "No integration" in st["message"] and "Integration removes differences" not in st["message"]
    assert "joint_clusters" in world.dsets[2]["cluster_methods"]


@needs_harmony
def test_donor_levels_from_the_design_and_confounding_statement(xe, world):
    keys = [d["data_dir"] for d in world.dsets]
    xe._design_set([{"key": keys[0], "cond": "A", "batch": "", "donor": "d1"},
                    {"key": keys[1], "cond": "A", "batch": "", "donor": "d1"},
                    {"key": keys[2], "cond": "B", "batch": "", "donor": ""}], {"A": "a", "B": "b"})
    st = _run(xe, {"integrate": "donor", "n_pcs": 20, "umap": False})
    assert st["status"] == "done", st["message"]
    res = xe._jc_result()
    assert res["levels"] == ["d1", "d1", f"(sample S2 {os.path.basename(keys[2])})"]
    assert "without a donor in the design count as their own level: S2" in st["message"]
    assert res["confounded"] == "full" and "fully confounded" in st["message"]
    # one batch for all samples: nothing to integrate over
    st = _run(xe, {"integrate": "batch", "n_pcs": 20, "umap": False})
    assert st["status"] == "done"                      # blank batches: every sample its own level
    xe._design_set([{"key": k, "cond": "", "batch": "b1", "donor": ""} for k in keys])
    st = _run(xe, {"integrate": "batch", "n_pcs": 20, "umap": False})
    assert st["status"] == "error" and "same batch" in st["message"]


def test_one_sample_and_a_resegmentation_are_refused(xe, D):
    st = _run(xe, {"integrate": "none"})                 # the bundle alone
    assert st["status"] == "error" and "at least two loaded samples" in st["message"]
    assert xe._jc_result() is None
    assert xe.make_jc_fig().layout.annotations[0].text.startswith("No joint clustering")
    assert xe._jc_records(None) == []


def test_cells_without_counts_get_no_cluster(xe, world):
    ds = world.dsets[1]
    m = ds["expr"].tolil()
    m[:7] = 0
    ds["expr"] = m.tocsr()
    st = _run(xe, {"integrate": "none", "n_pcs": 20, "umap": False})
    assert st["status"] == "done", st["message"]
    res = xe._jc_result()
    col = xe.cluster_col(res["method_key"])
    lab = world.dsets[1]["df"][col].to_numpy()
    assert (lab[:7] == 0).all() and (lab[7:] > 0).all()
    assert "7 cell(s) without counts have no cluster" in st["message"]
    assert res["n_cells"] == res["n_cells_view"] - 7


def test_cancel_publishes_nothing_and_a_rerun_completes(xe, world, monkeypatch):
    started, release = threading.Event(), threading.Event()
    real = xe._jc_pca

    def slow(*a, **k):
        started.set()
        release.wait(20)
        return real(*a, **k)

    monkeypatch.setattr(xe, "_jc_pca", slow)
    with xe._jc_lock:
        xe._jc_state.update({"status": "running", "message": ""})
    job = xe._job_start("joint", "Joint clustering", xe._run_joint,
                        args=("xenium", False, {"integrate": "none", "n_pcs": 20, "umap": False}),
                        state=xe._jc_state, lock=xe._jc_lock)
    assert started.wait(20)
    assert xe._job_cancel(job.id)
    release.set()
    job.thread.join(30)
    assert job.state == "cancelled"
    with xe._jc_lock:
        assert xe._jc_state["status"] == "error" and xe._jc_state["message"] == "Cancelled"
        assert xe._jc_state["result"] is None
    assert all("joint_clusters" not in d["cluster_methods"] for d in world.dsets)
    assert not os.path.isdir(xe._jc_cache_dir()) or not os.listdir(xe._jc_cache_dir())
    monkeypatch.setattr(xe, "_jc_pca", real)
    st = _run(xe, {"integrate": "none", "n_pcs": 20, "umap": False})
    assert st["status"] == "done", st["message"]


# ── callbacks ────────────────────────────────────────────────────────────────
def test_callbacks_guards_and_poll(xe, world):
    with pytest.raises(dash.exceptions.PreventUpdate):
        xe.run_joint(0, "sample", 2000, 30, 15, 1.0, 0, ["on"], "xenium", "original")
    out = xe.run_joint(1, "sample", 2000, 0, 15, 1.0, 0, ["on"], "xenium", "original")
    assert out[0] is False and "number of PCs must be positive" in out[2]
    with pytest.raises(dash.exceptions.PreventUpdate):
        xe.open_design_modal_joint(0)
    assert xe.open_design_modal_joint(1)[0] is True
    with pytest.raises(dash.exceptions.PreventUpdate):
        xe.download_joint(0)
    with pytest.raises(dash.exceptions.PreventUpdate):
        xe.download_joint(1)                                  # nothing computed
    with pytest.raises(dash.exceptions.PreventUpdate):
        xe.update_jc_design_summary("tab-agg", 0, 0, 0)
    with tl.new_threads(120):
        out = xe.run_joint(1, "none", 2000, 20, 15, 1.0, 0, [], "xenium", "original")
        assert out == (True, False, "Running…")
        again = xe.run_joint(1, "none", 2000, 20, 15, 1.0, 0, [], "xenium", "original")
        assert again[2] == "Joint clustering is already running." or xe._jc_state["status"] != "running"
    tl.wait_state(xe._jc_state, xe._jc_lock)
    msg, disabled, ver, poll_off, cs_ver, gs_ver = xe.poll_joint(1, 0, 4, 9)
    assert "Joint clusters" in msg and disabled is False and poll_off is True and cs_ver == 5 and gs_ver == 10
    # no new version: the final message stays (a cancelled run shows "Cancelled", not its last step)
    again = xe.poll_joint(2, ver, 5, 10)
    assert again[0] == msg and again[2] is dash.no_update and again[4] is dash.no_update
    assert xe.download_joint(1)["filename"] == "joint_clusters.csv"
    fig, rows, cols = xe.update_jc_results(ver, 0, None, None)
    assert len(rows) >= 1 and [c["id"] for c in cols][-3:] == ["n_S0", "n_S1", "n_S2"]
    summary, opts = xe.update_jc_design_summary("tab-joint", 0, 0, 0)
    assert "samples loaded" in summary and [o["value"] for o in opts] == ["none", "sample", "donor", "batch"]


def test_one_sample_callback_refusal(xe, D):
    out = xe.run_joint(1, "sample", 2000, 30, 15, 1.0, 0, ["on"], "xenium", "original")
    assert out[0] is False and "at least two loaded samples" in out[2]
    summary, _ = xe.update_jc_design_summary("tab-joint", 0, 0, 0)
    assert summary.startswith("One sample loaded")


# ── review round 5 ───────────────────────────────────────────────────────────
def _fake_joint_set(xe, world, embedded, sample_keys=None, with_ids=True):
    """A 'Joint clusters' set installed on the world's samples, with a cache entry holding a
    UMAP for the cells *embedded* (per sample: positions in the frame the run saw)."""
    tables = xe._view_tables("xenium")
    code = np.concatenate([np.full(len(tk), i) for i, tk in enumerate(embedded)])
    umap = np.column_stack([np.arange(len(code), dtype=np.float32), -np.arange(len(code), dtype=np.float32)])
    arrays = {"sample": code, "umap": umap, **{f"take_{i}": np.asarray(tk) for i, tk in enumerate(embedded)}}
    ids = [np.asarray(t["df"].index.astype(str))[np.asarray(tk)] for t, tk in zip(tables, embedded)]
    if with_ids:
        arrays.update({f"ids_{i}": v.astype(str) for i, v in enumerate(ids)})
    xe._jc_cache_write("tagtest", arrays, {"sample_keys": sample_keys or [t["key"] for t in tables]})
    key = xe._cs_install("Joint clusters", [np.zeros(len(t["df"]), dtype=int) for t in tables], "xenium",
                         meta={"kind": "joint", "variant": "plain", "cache_tag": "tagtest"})
    xe._jc_restore_seen.clear()
    return key, tables, ids, umap, code


def test_cached_joint_umap_is_reattached_by_cell_id_not_by_row(xe, world):
    # the cached run embedded every second cell (as a run made under a subset did)
    embedded = [np.arange(1, len(d["df"]), 2) for d in world.dsets]
    key, tables, ids, umap, code = _fake_joint_set(xe, world, embedded)
    # … and the frames are in another row order now (the subset was cleared, a store was re-read)
    for d in world.dsets:
        d["df"] = d["df"].iloc[::-1]
    tables = xe._view_tables("xenium")
    xe._jc_restore_umap(key, tables)
    for si, t in enumerate(tables):
        u1 = pd.Series(xe._ctl_get(t, "xenium", "joint_umap_1"), index=t["df"].index.astype(str))
        want = pd.Series(umap[code == si, 0], index=ids[si])
        np.testing.assert_array_equal(u1.loc[want.index].to_numpy(), want.to_numpy())    # each cell its own point
        assert u1.drop(want.index).isna().all()                                         # the others: none


@pytest.mark.parametrize("case", ["other_samples", "other_order", "other_cells", "no_ids"])
def test_cached_joint_umap_is_not_attached_to_other_samples_or_cells(xe, world, case):
    embedded = [np.arange(len(d["df"])) for d in world.dsets]
    keys = [d["data_dir"] for d in world.dsets]
    key, tables, _ids, _umap, _code = _fake_joint_set(
        xe, world, embedded, with_ids=case != "no_ids",
        sample_keys={"other_samples": ["/x/a", "/x/b", "/x/c"], "other_order": keys[::-1]}.get(case))
    if case == "other_cells":                 # same sample key, another set of cells (another subset, a new run)
        d = world.dsets[1]
        d["df"] = d["df"].rename(index=lambda c: "other_" + str(c))
        tables = xe._view_tables("xenium")
    xe._jc_restore_umap(key, tables)
    assert all(xe._ctl_get(t, "xenium", "joint_umap_1") is None for t in tables)
    assert xe._jc_umap_view(key) is None


def test_joint_umap_columns_are_written_only_after_the_set_is_installed(xe, world, monkeypatch):
    tables = xe._view_tables("xenium")
    old = [np.full(len(t["df"]), 7.0, dtype=np.float32) for t in tables]
    for t, o in zip(tables, old):
        xe._ctl_put(t, "xenium", {"joint_umap_1": o, "joint_umap_2": o})

    def refuse(*a, **k):
        raise ValueError("3 label arrays for 2 samples of the view")

    monkeypatch.setattr(xe, "_cs_install", refuse)
    st = _run(xe, {"integrate": "none", "n_pcs": 10, "umap": False})
    assert st["status"] == "error" and "label arrays" in st["message"]
    for t, o in zip(xe._view_tables("xenium"), old):
        np.testing.assert_array_equal(xe._ctl_get(t, "xenium", "joint_umap_1"), o)       # the last run's embedding


def test_cache_key_covers_matrix_content_and_available_backends(xe, world, monkeypatch):
    samples, _t = _samples_of(world.dsets)
    p = {"n_pcs": 10}
    tag = xe._jc_cache_tag(samples, "xenium", p, [])
    assert tag == xe._jc_cache_tag(_samples_of(world.dsets)[0], "xenium", p, [])
    # the same cells, shape and number of stored values, other counts (SPLIT run again)
    other = [dict(s) for s in samples]
    m = other[0]["csc"].copy()
    m.data = m.data[::-1].copy()
    assert m.shape == samples[0]["csc"].shape and m.nnz == samples[0]["csc"].nnz
    other[0]["csc"] = m
    assert xe._jc_cache_tag(other, "xenium", p, []) != tag
    assert xe._an_matrix_digest(m) != xe._an_matrix_digest(samples[0]["csc"])
    assert xe._an_matrix_digest(samples[0]["csc"].copy()) == xe._an_matrix_digest(samples[0]["csc"])
    # a run made without umap-learn / Leiden is not the run of a machine that has them
    have = xe._jc_backends()
    assert set(have) == {"umap", "leiden"}
    monkeypatch.setattr(xe, "_jc_backends", lambda: {"umap": not have["umap"], "leiden": have["leiden"]})
    assert xe._jc_cache_tag(samples, "xenium", p, []) != tag
    monkeypatch.setattr(xe, "_jc_backends", lambda: {"umap": have["umap"], "leiden": not have["leiden"]})
    assert xe._jc_cache_tag(samples, "xenium", p, []) != tag


def test_joint_umap_by_gene_uses_corrected_counts_only_when_every_sample_has_them(xe, world, monkeypatch):
    tables = xe._view_tables("xenium")
    key = xe._cs_install("Joint clusters", [np.zeros(len(t["df"]), dtype=int) for t in tables], "xenium",
                         meta={"kind": "joint", "variant": "plain"})
    for t in tables:
        v = np.arange(len(t["df"]), dtype=np.float32)
        xe._ctl_put(t, "xenium", {"joint_umap_1": v, "joint_umap_2": v})
    asked = []
    monkeypatch.setattr(xe, "EXTRA_DATASETS", world.dsets[1:])
    monkeypatch.setattr(xe, "_xenium_gene_values",
                        lambda gene, corr=False: asked.append(("S0", corr)) or np.zeros(len(tables[0]["df"])))
    monkeypatch.setattr(xe, "_extra_gene_values",
                        lambda eds, gene, corr=False: asked.append(("extra", corr)) or np.zeros(len(eds["df"])))
    for all_have, want in ((False, False), (True, True)):
        asked.clear()
        monkeypatch.setattr(xe, "_multi_use_corrected", lambda uc, _a=all_have: bool(uc) and _a)
        fig = xe._jc_umap_fig("gene", key, "g0", 3, 0.8, use_corrected=True)
        assert fig is not None and len(asked) == 3
        assert {c for _w, c in asked} == {want}, asked           # never corrected for one sample and original for another


def test_webgl_joint_umap_token_changes_with_a_joint_rerun(xe, world):
    tables = xe._view_tables("xenium")
    key = xe._cs_install("Joint clusters", [np.zeros(len(t["df"]), dtype=int) for t in tables], "xenium",
                         meta={"kind": "joint", "variant": "plain"})
    for t in tables:
        v = np.arange(len(t["df"]), dtype=np.float32)
        xe._ctl_put(t, "xenium", {"joint_umap_1": v, "joint_umap_2": v})
    from xenium_app import viewer_provider
    prov = viewer_provider._UmapViewerProvider()
    src = f"xenium|emb:{key}"
    tok = prov.cells(src)[0]["token"]
    with xe._jc_lock:
        xe._jc_state["version"] = xe._jc_state.get("version", 0) + 1      # what a finished run does
    assert prov.cells(src)[0]["token"] != tok
    assert prov.cells("xenium")[0]["token"] == prov.cells("xenium")[0]["token"]
    # the page's cells key of the joint embedding carries the version the poll bumps
    js = open(os.path.join(os.path.dirname(os.path.dirname(viewer_provider.__file__)), "assets",
                           "xenium_viewer_dash.js")).read()
    assert "jointEmb ? an.gsVersion : 0" in js
