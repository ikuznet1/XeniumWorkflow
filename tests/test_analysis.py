"""Differential expression, gene-set scores and transcript density.

Each feature is checked against an independent computation on the bundle under test:
the rank-sum test against scipy.stats.mannwhitneyu and scanpy.tl.rank_genes_groups on
dense arrays, the score against a dense numpy recomputation and scanpy.tl.score_genes,
the density grid against numpy.histogram2d over the transcripts read with pandas.
The module also runs against a large bundle (``pytest --bundle PATH``).
"""
from __future__ import annotations

import io
import json
import os
import time

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy import stats

import xe_testlib as tl


@pytest.fixture(autouse=True)
def _clean_analysis_state(xe):
    """Saved gene sets / DE groups / density grids of a test do not reach the next one."""
    yield
    import shutil
    for path in (xe._gs_path(), xe._de_groups_path()):
        if os.path.exists(path):
            os.remove(path)
    shutil.rmtree(os.path.dirname(xe._density_cache_file("x")), ignore_errors=True)
    xe._an_autoload()


# ── helpers: the independent side ────────────────────────────────────────────
def _lognorm_dense(s, genes, cells=None):
    """log1p(count / library × 10,000) as a dense float32 array (cells of the view × genes)."""
    X = sp.csr_matrix(s["csc"])
    lib = np.asarray(X[:, [s["gni"][g] for g in s["genes"]]].sum(axis=1), dtype=np.float64).ravel()
    rows = s["rows"] if cells is None else s["rows"][cells]
    sub = X[rows][:, [s["gni"][g] for g in genes]].toarray().astype(np.float32)
    sf = np.where(lib[rows] > 0, 1e4 / np.maximum(lib[rows], 1e-300), 0.0).astype(np.float32)
    return np.log1p(sub * sf[:, None])


def _halves(s):
    """Two disjoint groups of cells: left and right of the median x."""
    x = s["df"]["x_centroid"].to_numpy(float)
    a = x < np.median(x)
    return a, ~a


def _some_genes(table, n=80, seed=0):
    """All genes of a small panel; otherwise the strongest ones plus a random draw."""
    genes = list(table["gene"])
    if len(genes) <= 1200:
        return genes
    rng = np.random.default_rng(seed)
    top = list(table.sort_values("pval")["gene"][:20])
    return list(dict.fromkeys(top + list(rng.choice(genes, n, replace=False))))


@pytest.fixture(scope="module")
def view(xe):
    samples = xe._an_samples("xenium", False)
    assert len(samples) == 1 and samples[0]["genes"], "the bundle has an expression matrix"
    return samples


@pytest.fixture(scope="module")
def de_halves(xe, view):
    a, b = _halves(view[0])
    return a, b, xe._de_compute(view, [a], [b])


# ── differential expression: statistics ──────────────────────────────────────
def test_de_statistic_and_p_value_equal_scipy_mannwhitneyu(xe, view, de_halves):
    a, b, res = de_halves
    s = view[0]
    assert (res["n_a"], res["n_b"]) == (int(a.sum()), int(b.sum())) and res["n_overlap"] == 0
    tab = res["table"].set_index("gene")
    genes = _some_genes(res["table"])
    L = _lognorm_dense(s, genes)
    n_tied = 0
    for j, g in enumerate(genes):
        va, vb = L[a, j], L[b, j]
        if np.ptp(L[:, j]) == 0:                       # no variation: nothing to test
            assert tab.loc[g, "pval"] == 1.0 and tab.loc[g, "z"] == 0.0
            continue
        u, p = stats.mannwhitneyu(va, vb, alternative="two-sided", method="asymptotic")
        assert tab.loc[g, "U"] == pytest.approx(u, abs=1e-6), g
        assert np.log10(max(tab.loc[g, "pval"], 1e-300)) == pytest.approx(np.log10(max(p, 1e-300)), abs=1e-6), g
        nzv = np.concatenate([va, vb])
        n_tied += len(np.unique(nzv)) < len(nzv)
    assert n_tied > 0, "the comparison covered genes with tied values (zeros at least)"


def test_de_effect_sizes_and_fdr(xe, view, de_halves):
    a, b, res = de_halves
    s, tab = view[0], res["table"]
    genes = _some_genes(tab)
    sub = tab.set_index("gene").loc[genes]
    N = np.expm1(_lognorm_dense(s, genes).astype(np.float64))          # normalised, not log
    np.testing.assert_allclose(sub["mean_a"], N[a].mean(axis=0), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(sub["log2fc"], np.log2((N[a].mean(axis=0) + 1) / (N[b].mean(axis=0) + 1)),
                               rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(sub["pct_a"], 100 * (N[a] > 0).mean(axis=0), atol=1e-9)
    np.testing.assert_allclose(sub["pct_b"], 100 * (N[b] > 0).mean(axis=0), atol=1e-9)
    # Benjamini-Hochberg over every tested gene
    np.testing.assert_allclose(tab["padj"], stats.false_discovery_control(tab["pval"].to_numpy(), method="bh"),
                               rtol=1e-9, atol=1e-300)
    assert (tab["padj"] >= tab["pval"] - 1e-15).all() and tab["padj"].max() <= 1.0
    # -log10 FDR is kept in log space: exact where the FDR itself underflows to 0
    big = tab["padj"].to_numpy() > 1e-290
    np.testing.assert_allclose(tab["neg_log10_padj"][big], -np.log10(tab["padj"][big]), atol=1e-6)
    assert (tab["neg_log10_padj"][~big] >= 289.9).all() and np.isfinite(tab["neg_log10_padj"]).all()


def test_de_agrees_with_scanpy_wilcoxon(xe, view, de_halves):
    sc = pytest.importorskip("scanpy")
    import anndata as ad
    a, b, res = de_halves
    s, tab = view[0], res["table"].set_index("gene")
    genes = list(tab.index)
    if len(s["df"]) * len(genes) <= 5e7:
        X = sp.csr_matrix(_lognorm_dense(s, genes))
    else:                                              # large bundle: stay sparse
        X = sp.csr_matrix(s["csc"])[s["rows"]][:, [s["gni"][g] for g in genes]].astype(np.float32)
        sf, _ = xe._an_size_factors(s)
        X = sp.diags(sf[s["rows"]]).dot(X).tocsr().astype(np.float32)
        X.data = np.log1p(X.data)
    adata = ad.AnnData(X=X, var=pd.DataFrame(index=genes))
    adata.obs["g"] = pd.Categorical(np.where(a, "a", "b"))
    sc.tl.rank_genes_groups(adata, "g", groups=["a"], reference="b", method="wilcoxon", tie_correct=True)
    ref = sc.get.rank_genes_groups_df(adata, "a").set_index("names").loc[genes]
    # scanpy: z = (U - n1 n2 / 2) / sd without continuity correction; here 0.5 is taken off |U - mean|
    d = tab["U"].to_numpy() - res["n_a"] * res["n_b"] / 2.0
    z = tab["z"].to_numpy()
    ok = (np.abs(d) > 0.5) & np.isfinite(ref["scores"].to_numpy())
    assert ok.sum() >= 3          # (the tiny bundle's matrix has three expressed genes)
    sd = (np.abs(d[ok]) - 0.5) / np.abs(z[ok])
    np.testing.assert_allclose(ref["scores"].to_numpy()[ok], d[ok] / sd, rtol=2e-4, atol=2e-4)
    assert (np.abs(ref["scores"].to_numpy()[ok] - z[ok]) <= 0.5 / sd + 2e-4).all()
    # p-values: equal up to that continuity correction
    lp_ref = np.log10(np.maximum(2 * stats.norm.sf(np.abs(d[ok] / sd)), 1e-300))
    np.testing.assert_allclose(np.log10(np.maximum(ref["pvals"].to_numpy()[ok], 1e-300)), lp_ref,
                               rtol=1e-3, atol=1e-3)
    lp_mine = np.log10(np.maximum(tab["pval"].to_numpy()[ok], 1e-300))
    assert (lp_mine >= lp_ref - 1e-6).all(), "the continuity correction only makes p larger"


def test_de_rank_sums_on_two_matrices_with_ties_in_many_chunks(xe):
    """The sparse kernel on synthetic counts: heavy ties, two matrices whose columns are
    in different orders (side-by-side samples), a gene missing from one, tiny chunks."""
    rng = np.random.default_rng(3)
    n1, n2, ng = 700, 500, 60
    dense = [rng.poisson(rng.gamma(0.3, 2.0, size=ng), size=(n, ng)).astype(np.float32) for n in (n1, n2)]
    perm = rng.permutation(ng)
    cols = [np.arange(ng), np.argsort(perm)]            # gene j sits in column cols[k][j]
    mats = [sp.csc_matrix(dense[0]), sp.csc_matrix(dense[1][:, perm])]
    cols[1] = cols[1].copy()
    cols[1][7] = -1                                     # gene 7 is not measured in sample 2
    grp = [rng.integers(0, 3, size=n).astype(np.int8) for n in (n1, n2)]
    sf = [rng.uniform(0.5, 2.0, size=n).astype(np.float32) for n in (n1, n2)]
    r = xe._de_rank_sums([{"csc": m, "cols": c, "grp": g, "sf": f}
                          for m, c, g, f in zip(mats, cols, grp, sf)], ng, chunk_nnz=900)
    u, z, log10p = xe._de_test_stats(r["rank_sum_a"], r["tie_term"], r["n_a"], r["n_b"])
    for j in range(ng):
        vals = [np.log1p(dense[0][:, j] * sf[0])]
        vals.append(np.zeros(n2, dtype=np.float32) if j == 7 else np.log1p(dense[1][:, j] * sf[1]))
        va = np.concatenate([v[g == 1] for v, g in zip(vals, grp)])
        vb = np.concatenate([v[g == 2] for v, g in zip(vals, grp)])
        if np.ptp(np.concatenate([va, vb])) == 0:
            continue
        # (float64 for the reference: scipy >= 1.17 computes in the input's precision)
        u_ref, p_ref = stats.mannwhitneyu(va.astype(np.float64), vb.astype(np.float64),
                                          alternative="two-sided", method="asymptotic")
        assert u[j] == pytest.approx(u_ref, abs=1e-6)
        assert log10p[j] == pytest.approx(np.log10(p_ref), abs=1e-7)
        assert r["nnz_a"][j] == (va > 0).sum() and r["nnz_b"][j] == (vb > 0).sum()


def test_de_groups_overlap_complement_and_minimum_size(xe, view):
    s = view[0]
    n = len(s["df"])
    a, b = _halves(s)
    both = a.copy()
    extra = np.flatnonzero(b)[:2]
    both[extra] = True                                   # two cells of B are also in A
    res = xe._de_compute(view, [both], [b])
    assert res["n_overlap"] == 2 and res["n_a"] == int(a.sum()) + 2 and res["n_b"] == int(b.sum()) - 2
    rest = xe._de_compute(view, [a], None)               # "all other cells"
    assert rest["n_a"] + rest["n_b"] == n and rest["n_b"] == int((~a).sum())
    tiny = np.zeros(n, dtype=bool)
    tiny[:xe._DE_MIN_CELLS - 1] = True
    with pytest.raises(ValueError, match=f"Group A has {xe._DE_MIN_CELLS - 1} cells.*at least {xe._DE_MIN_CELLS}"):
        xe._de_compute(view, [tiny], None)
    with pytest.raises(ValueError, match="Group B has 0 cells after the .* it shares with A"):
        xe._de_compute(view, [a], [a])
    with pytest.raises(ValueError, match="Only one of the two groups"):
        xe._de_compute(view, None, None)


def test_de_uses_the_active_counts_segmentation_and_samples(xe, D, view):
    s = view[0]
    a, b = _halves(s)
    base = xe._de_compute(view, [a], [b])["table"]
    # SPLIT-corrected counts
    corr = sp.csr_matrix(D["expr"]).copy()
    corr.data = np.floor(corr.data * 0.5)
    corr.eliminate_zeros()
    D["split_corrected_expr"] = corr
    cs = xe._an_samples("xenium", True)
    assert cs[0]["corrected"] and not xe._an_samples("xenium", False)[0]["corrected"]
    got = xe._de_compute(cs, [a], [b])
    assert got["corrected"]
    # cells left without any corrected count are not in the corrected view (not zeros)
    a_cov = a & (cs[0]["rows"] >= 0)
    assert got["n_a"] == int(a_cov.sum())
    np.testing.assert_allclose(got["table"]["pct_a"], 100 * np.asarray(
        (corr[s["rows"][a_cov]][:, [s["gni"][g] for g in s["genes"]]] > 0).mean(axis=0)).ravel(), atol=1e-9)
    # a Proseg run: its own cells and matrix
    n = len(s["df"])
    keep = np.arange(n)[::-1]
    cdf = pd.DataFrame({"x_centroid": s["df"]["x_centroid"].to_numpy()[keep],
                        "y_centroid": s["df"]["y_centroid"].to_numpy()[keep],
                        "transcript_counts": 1}, index=[f"p{i}" for i in range(n)])
    expr = sp.csr_matrix(D["expr"])[s["rows"][keep]][:, :len(xe._panel_gene_names())].tocsr()
    with xe._proseg_lock:
        xe._proseg_state.update({"status": "done", "result": {
            "cells_df": cdf, "cell_bounds": {}, "out_dir": "/nonexistent/fake", "expr": expr,
            "source": "proseg", "sdata_path": None}})
    ps = xe._an_samples("proseg:abc", False)
    assert ps[0]["label"] == "proseg" and len(ps[0]["df"]) == n
    pa, pb = _halves(ps[0])
    alt = xe._de_compute(ps, [pa], [pb])["table"]
    if not (np.asarray(s["rows"]) < 0).any():            # same cells in reverse order: same test
        np.testing.assert_allclose(alt.set_index("gene").loc[base["gene"], "U"], base["U"], atol=1e-6)
    # two samples side by side: cells are pooled, labels are per sample
    xe.EXTRA_DATASETS.append({**{k: D[k] for k in ("expr", "df_to_expr", "gene_name_to_idx", "gene_names")},
                              "df": D["df"], "x_offset": 1e5, "cluster_methods": D.get("cluster_methods")})
    ms = xe._an_samples("xenium", False)
    assert [m["label"] for m in ms] == ["S0", "S1"] and ms[1]["x_offset"] == 1e5
    pooled = xe._de_compute(ms, [a, a], [b, b])
    assert pooled["n_a"] == 2 * int(a.sum()) and pooled["n_samples"] == 2
    # an outline drawn over the second sample selects that sample's cells only
    x, y = D["df"]["x_centroid"].to_numpy(float), -D["df"]["y_centroid"].to_numpy(float)
    box = [[x.min() - 1 + 1e5, y.min() - 1], [x.max() + 1 + 1e5, y.min() - 1],
           [x.max() + 1 + 1e5, y.max() + 1], [x.min() - 1 + 1e5, y.max() + 1]]
    assert not xe._an_polygon_mask(box, ms[0]).any() and xe._an_polygon_mask(box, ms[1]).all()
    if D.get("cluster_methods"):
        labs = xe._an_cluster_labels(ms, D["cluster_methods"][0], False, None)
        assert str(labs[0][0]).startswith("S0:C") and str(labs[1][0]).startswith("S1:C")


# ── differential expression: the tab ─────────────────────────────────────────
def _left_box(D, um=True):
    """Outline around the cells left of the median x (µm, y down as the WebGL view sends it)."""
    x, y = D["df"]["x_centroid"].to_numpy(float), D["df"]["y_centroid"].to_numpy(float)
    xm = float(np.median(x))
    return [[x.min() - 1, y.min() - 1], [xm, y.min() - 1], [xm, y.max() + 1], [x.min() - 1, y.max() + 1]], x < xm


def test_de_tab_selection_groups_run_volcano_table_and_csv(xe, D, cache_dir):
    from dash import no_update
    assert xe.make_de_volcano_fig().layout.annotations[0].text.startswith("No differential expression")
    assert xe.update_de_results(0, 0, None, None)[1] == []
    method = (D.get("cluster_methods") or [None])[0]
    opts = xe.de_update_group_options("tab-de", 0, 0, 0, 0, 0, 0, "xenium", method, "cluster", "original", 0, 0, 0)
    assert [o["value"] for o in opts[0]][0] == "rest" and "No current selection" in opts[2]
    assert xe.de_save_group(1, "left", 0)[1].startswith("There is no current selection")
    # a WebGL lasso (µm, y down) and a Plotly box (plot space) give the same group
    poly, inside = _left_box(D)
    assert xe.de_capture_webgl_selection({"polygon": poly, "count": int(inside.sum())}, 0) == 1
    sel_gl = [tuple(p) for p in xe._de_state["selection"]["polygon_xy"]]
    xs, ys = [p[0] for p in poly], [-p[1] for p in poly]
    assert xe.de_capture_plotly_selection(
        {"points": [{"customdata": "c"}] * int(inside.sum()),
         "range": {"x": [min(xs), max(xs)], "y": [min(ys), max(ys)]}}, 1, "xenium") == 2
    s = xe._an_samples("xenium")[0]
    np.testing.assert_array_equal(xe._an_polygon_mask(xe._de_state["selection"]["polygon_xy"], s), inside)
    np.testing.assert_array_equal(xe._an_polygon_mask([list(p) for p in sel_gl], s), inside)
    opts = xe.de_update_group_options("tab-de", 2, 0, 0, 0, 0, 0, "xenium", method, "cluster", "original", 0, 0, 0)
    assert opts[0][0]["value"] == "sel:current" and f"{int(inside.sum()):,} cells" in opts[2]
    if method:
        assert any(o["value"].startswith("cluster:") for o in opts[0])
    # save it; the group is written under the cache dir and survives a reload
    assert xe.de_save_group(1, "", 0)[1] == "Enter a name for the group."
    ver, msg = xe.de_save_group(1, "left", 0)
    assert ver == 1 and "left" in msg and xe.de_save_group(1, "left", 1)[1].endswith("already exists.")
    assert os.path.dirname(xe._de_groups_path()) == str(cache_dir)
    assert json.load(open(xe._de_groups_path()))[0]["name"] == "left"
    xe._an_autoload()
    assert [g["name"] for g in xe._de_state["groups"]] == ["left"] and xe._de_state["selection"] is None
    # run: refusals first
    assert xe.run_de(1, None, "rest", "xenium", method, "cluster", "original")[2] == "Pick a group for A and for B."
    assert xe.run_de(1, "grp:left", "grp:left", "xenium", method, "cluster", "original")[2] == "A and B are the same group."
    assert xe.run_de(1, "grp:left", "rest", "xenium", method, "cluster", "original") == (True, False, "Running…")
    msg = tl.wait_state(xe._de_state, xe._de_lock)
    job = [j for j in xe._jobs.values() if j.kind == "de"][-1]
    job.thread.join(10)
    assert job.state == "done" and job.fraction == 1.0
    assert f"(A, {int(inside.sum()):,} cells)" in msg and f"(B, {int((~inside).sum()):,} cells)" in msg
    assert xe.poll_de(1, 0) == (msg, True, False, 1)
    fig, rows = xe.update_de_results(1, 0, None, None)
    res = xe._de_state["result"]
    assert len(rows) == res["n_genes"] and sum(len(t.x) for t in fig.data) == res["n_genes"]
    assert [r["padj"] for r in rows] == sorted(r["padj"] for r in rows)
    ref = xe._de_compute(xe._an_samples("xenium"), [inside], None)["table"]
    pd.testing.assert_frame_equal(res["table"], ref)
    # click → gene selector; download
    gene = rows[0]["gene"]
    options = [{"label": g, "value": g} for g in D["gene_names"]]
    assert xe.de_pick_gene_from_volcano({"points": [{"customdata": gene}]}, options) == gene
    assert xe.de_pick_gene_from_table({"row": 0, "column": 0}, rows[:5], options) == gene
    out = xe.download_de_table(1)
    assert out["filename"] == "de_left_vs_all_other_cells.csv" and out["content"].startswith("# A: left (n=")
    csv = pd.read_csv(io.StringIO(out["content"]), comment="#")
    assert list(csv.columns[:3]) == ["gene", "log2fc", "pct_a"] and len(csv) == res["n_genes"]
    # too small a group: a message, not a result
    x, y = D["df"]["x_centroid"].to_numpy(float), D["df"]["y_centroid"].to_numpy(float)
    i = int(np.argmin(x))
    dot = [[x[i] - .01, y[i] - .01], [x[i] + .01, y[i] - .01], [x[i] + .01, y[i] + .01], [x[i] - .01, y[i] + .01]]
    xe.de_capture_webgl_selection({"polygon": dot, "count": 1}, 2)
    xe.run_de(2, "sel:current", "rest", "xenium", method, "cluster", "original")
    with pytest.raises(AssertionError, match="Group A has 1 cells; at least 10"):
        tl.wait_state(xe._de_state, xe._de_lock)
    assert xe.de_delete_group(1, "grp:left", 1) == (2, "Deleted group 'left'.", None)
    assert json.load(open(xe._de_groups_path())) == []
    assert xe.de_delete_group(1, "rest", 2)[0] is no_update


# ── gene-set scores ──────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def marker_set(xe, view):
    """Genes that are expressed in this bundle, spread over the expression range."""
    genes = xe._an_common_genes(view)
    means = xe._an_gene_means(view, genes)
    order = np.argsort(means)[::-1]
    expressed = [genes[i] for i in order if means[i] > 0]
    assert len(expressed) >= 3, "the bundle has at least three expressed genes"
    if len(expressed) >= 40:
        return expressed[:40:3]
    # a small matrix: leave one expressed gene to be the control of the others
    return expressed[:12] if len(expressed) > 12 else expressed[:len(expressed) - 1]


def test_gene_set_score_equals_a_dense_recomputation(xe, view, marker_set):
    s = view[0]
    arrays, info = xe._gs_score(view, marker_set + ["NOT_A_GENE", marker_set[0].lower()])
    assert info["missing"] == ["NOT_A_GENE"] and info["found"] == marker_set
    assert info["used"] and info["n_control"] == len(info["control"]) > 0
    assert not set(info["control"]) & set(marker_set)
    L = _lognorm_dense(s, info["used"] + info["control"]).astype(np.float64)
    Z = (L - L.mean(axis=0)) / L.std(axis=0)
    k = len(info["used"])
    ref = Z[:, :k].mean(axis=1) - Z[:, k:].mean(axis=1)
    ok = s["rows"] >= 0
    np.testing.assert_allclose(arrays[0][ok], ref[ok], rtol=2e-3, atol=2e-3)
    # control genes come from the expression bins of the set genes (scanpy's binning)
    genes = xe._an_common_genes(view)
    means = pd.Series(xe._an_gene_means(view, genes), index=genes)
    cut = means.rank(method="min") // max(1, int(np.round(len(genes) / 24)))
    assert set(cut[info["control"]]) <= set(cut[marker_set])
    # fixed seed: the same call gives the same control set and scores
    again, info2 = xe._gs_score(view, marker_set)
    assert info2["control"] == info["control"] and np.array_equal(again[0], arrays[0], equal_nan=True)
    assert xe._gs_score(view, marker_set, seed=1)[1]["control"] != info["control"] or info["n_control"] < 50


def test_gene_set_score_agrees_with_scanpy_score_genes(xe, view, marker_set):
    sc = pytest.importorskip("scanpy")
    import anndata as ad
    s = view[0]
    genes = xe._an_common_genes(view)
    X = sp.csr_matrix(s["csc"])[s["rows"]][:, [s["gni"][g] for g in genes]].astype(np.float32).tocsr()
    adata = ad.AnnData(X=X, var=pd.DataFrame(index=genes))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    ok = s["rows"] >= 0
    # 1. Without the random draw (control = every gene of the set's expression bins) the
    #    two implementations use the same control genes: the scores are the same numbers.
    sc.tl.score_genes(adata, marker_set, score_name="all", ctrl_size=10 ** 6, random_state=0)
    whole, winfo = xe._gs_score(view, marker_set, zscore=False, ctrl_size=10 ** 6)
    ref_all = adata.obs["all"].to_numpy()
    r_all = np.corrcoef(whole[0][ok], ref_all[ok])[0, 1]
    assert r_all >= 0.9995
    np.testing.assert_allclose(whole[0][ok], ref_all[ok], atol=0.02 * max(1e-9, float(np.ptp(ref_all))))
    # 2. Defaults (50 control genes per bin): each implementation draws its own control
    #    genes, so the scores differ by that draw - most for a set of very highly
    #    expressed genes, whose bin holds genes of very different expression.
    sc.tl.score_genes(adata, marker_set, score_name="ref", random_state=0)
    ref = adata.obs["ref"].to_numpy()
    plain, _ = xe._gs_score(view, marker_set, zscore=False)       # scanpy's definition
    zs, _ = xe._gs_score(view, marker_set)                        # the app's (z-scored genes)
    r_plain = np.corrcoef(plain[0][ok], ref[ok])[0, 1]
    r_z = np.corrcoef(zs[0][ok], ref[ok])[0, 1]
    print(f"gene-set score vs scanpy.tl.score_genes: r = {r_all:.5f} (same control genes), "
          f"{r_plain:.4f} (own draw of 50 per bin), {r_z:.4f} (own draw, z-scored genes: the app's default); "
          f"{len(marker_set)} genes, {winfo['n_control']} genes in their bins")
    assert r_plain >= 0.85
    # The app standardises every gene first, so that the set's highest-expressed genes do
    # not dominate the mean; this changes the weights of the genes, not what is scored.
    assert r_z >= 0.80


def test_gene_set_refusals_and_matching(xe, view, marker_set):
    assert xe._gs_parse("A, B;C\nD\tE  A\n\"F\"") == ["A", "B", "C", "D", "E", "F"]
    assert xe._gs_score(view, [marker_set[0], "NOPE1", "NOPE2"])[0] is None
    assert xe.run_geneset_score(1, "", "A\nB", "xenium", "original")[2] == "Enter a name for the gene set."
    assert xe.run_geneset_score(1, "s", marker_set[0], "xenium", "original")[2] == "Paste at least 2 gene names."
    out = xe.run_geneset_score(1, "s", f"{marker_set[0]}, NOPE1, NOPE2", "xenium", "original")
    assert out[:2] == (False, True) and "Only 1 of the 3 genes" in out[2] and "NOPE1, NOPE2" in out[2]
    assert xe._geneset_names() == [] and not os.path.exists(xe._gs_path())


def test_gene_set_becomes_a_colour_in_both_renderers(xe, D, client, cache_dir, marker_set):
    text = ",\n".join(marker_set) + ", NOT_A_GENE"
    assert xe.run_geneset_score(1, " my  set ", text, "xenium", "original") == (True, False, "Scoring…")
    msg = tl.wait_state(xe._geneset_state, xe._geneset_lock)
    assert "Not in this dataset (1): NOT_A_GENE" in msg and "Pick 'Gene set: my set'" in msg
    assert [j for j in xe._jobs.values() if j.kind == "geneset"][-1].label == "Gene-set score: my set"
    out = xe.poll_geneset(1, 0)
    assert out[1:] == (True, False, 1, "geneset:my set")
    # persisted per dataset under the cache dir, offered under Color By and in the tab
    assert os.path.dirname(xe._gs_path()) == str(cache_dir)
    assert json.load(open(xe._gs_path())) == {"my set": marker_set + ["NOT_A_GENE"]}
    assert {"label": "Gene set: my set", "value": "geneset:my set"} in xe.update_color_by_for_genesets(1, [])
    assert xe.update_geneset_options(1, 0) == [{"label": f"my set ({len(marker_set) + 1} genes)", "value": "my set"}]
    assert xe.fill_geneset_form("my set") == ("my set", "\n".join(marker_set + ["NOT_A_GENE"]))
    xe._an_autoload()
    assert xe._geneset_names() == ["my set"]
    arrays, info = xe._geneset_values("my set")
    vals = arrays[0]
    lim = float(np.nanmax(np.abs(vals)))
    # Plotly: spatial and UMAP, diverging scale centred at 0, range inputs honoured
    method = (D.get("cluster_methods") or [None])[0]
    fig = xe.make_spatial_fig("geneset:my set", method, None, 3, 0.8, [], None)
    m = fig.data[0].marker
    np.testing.assert_allclose(np.asarray(m.color, dtype=float), vals, rtol=1e-6, equal_nan=True)
    assert (m.cmin, m.cmax) == (pytest.approx(-lim), pytest.approx(lim)) and m.showscale
    assert m.colorscale[0][1] != m.colorscale[-1][1] and "Gene set" in m.colorbar.title.text
    fig = xe.make_spatial_fig("geneset:my set", method, None, 3, 0.8, [], None, cbar_range=(None, 0.5))
    assert (fig.data[0].marker.cmin, fig.data[0].marker.cmax) == (pytest.approx(-lim), 0.5)
    if "umap_1" in D["df"].columns and D["df"]["umap_1"].notna().any():
        ufig = xe.make_umap_fig("geneset:my set", method, None, 3, 0.8)
        udf = D["df"].dropna(subset=["umap_1", "umap_2"])
        np.testing.assert_allclose(np.asarray(ufig.data[0].marker.color, dtype=float),
                                   vals[D["df"].index.get_indexer(udf.index)], rtol=1e-6, equal_nan=True)
        assert ufig.data[0].marker.cmin == pytest.approx(-lim)
    # WebGL: the colour endpoint of the spatial and of the UMAP viewer
    for prefix in ("", "/umap"):
        h, a = tl.get_bin(client, f"{prefix}/viewer/colors.bin?mode=geneset:my%20set")
        assert h["kind"] == "continuous" and h["colormap"] == "rdbu" and h["label"] == "Gene set: my set"
        assert h["min"] == pytest.approx(-lim) and h["max"] == pytest.approx(lim)
        np.testing.assert_allclose(a["values"], vals, rtol=1e-6, equal_nan=True)
    # an unknown set is drawn as transcript counts, in both
    h, _ = tl.get_bin(client, "/viewer/colors.bin?mode=geneset:nope")
    assert h["label"].startswith("Transcript Counts")
    assert "Transcript" in xe.make_spatial_fig("geneset:nope", method, None, 3, 0.8, [], None).data[0].name
    # the score follows the view: subset, counts mode
    med = float(np.median(D["df"]["transcript_counts"]))
    xe.subset(min_transcripts=med)
    sub, _ = xe._geneset_values("my set")
    assert len(sub[0]) == len(D["df"]) < len(vals)
    h, a = tl.get_bin(client, "/viewer/colors.bin?mode=geneset:my%20set")
    np.testing.assert_allclose(a["values"], sub[0], rtol=1e-6, equal_nan=True)
    xe.unsubset()
    corr = sp.csr_matrix(D["expr"]).copy()
    j = D["gene_name_to_idx"][info["used"][0]]
    corr = sp.csr_matrix(corr.multiply(np.where(np.arange(corr.shape[1]) == j, 0.0, 1.0)))
    D["split_corrected_expr"] = corr
    cvals, cinfo = xe._geneset_values("my set", "xenium", True)
    assert info["used"][0] not in cinfo["used"] and not np.allclose(cvals[0], vals, equal_nan=True)
    h, a = tl.get_bin(client, "/viewer/colors.bin?mode=geneset:my%20set&corrected=1")
    np.testing.assert_allclose(a["values"], cvals[0], rtol=1e-6, equal_nan=True)
    # two samples: one value per cell of each
    xe.EXTRA_DATASETS.append({**{k: D[k] for k in ("expr", "df_to_expr", "gene_name_to_idx", "gene_names")},
                              "df": D["df"], "x_offset": 1e5})
    two, _ = xe._geneset_values("my set")
    assert [len(v) for v in two] == [len(D["df"])] * 2
    np.testing.assert_allclose(two[0], two[1], equal_nan=True)
    del xe.EXTRA_DATASETS[:]
    # delete: gone from the file and from Color By
    from dash import no_update
    assert xe.delete_geneset(1, "my set", 1, "geneset:my set") == (2, "Deleted gene set 'my set'.", None,
                                                                   "transcript_counts")
    assert json.load(open(xe._gs_path())) == {} and xe.update_color_by_for_genesets(2, []) == []
    assert xe.delete_geneset(1, None, 2, "cluster")[0] is no_update


# ── transcript density ───────────────────────────────────────────────────────
def _filtered_tx(raw, D, gene=None):
    """Q ≥ 20 gene transcripts of the bundle (of one gene), read independently of the app."""
    import pyarrow.parquet as pq
    path = os.path.join(D["data_dir"], "transcripts.parquet")
    cols = [c for c in ("feature_name", "x_location", "y_location", "qv", "is_gene", "codeword_category")
            if c in pq.read_schema(path).names]
    tx = pq.read_table(path, columns=cols,
                       filters=[("feature_name", "==", gene)] if gene else None).to_pandas()
    return tl.gene_tx(tx)


@pytest.fixture(scope="module")
def top_gene(raw, D):
    import pyarrow.parquet as pq
    path = os.path.join(D["data_dir"], "transcripts.parquet")
    if not os.path.exists(path):
        pytest.skip("bundle without transcripts.parquet")
    df = D["df"]
    expr = sp.csr_matrix(D["expr"])
    gene = D["gene_names"][int(np.asarray(expr.sum(axis=0)).ravel()[:len(D["gene_names"])].argmax())]
    return gene


def _hist(tx, b, iy0, ix0, h, w):
    """Independent recount: numpy's 2-D histogram on bin edges at multiples of b."""
    ey = (iy0 + np.arange(h + 1)) * float(b)
    ex = (ix0 + np.arange(w + 1)) * float(b)
    y, x = tx["y_location"].to_numpy(np.float64), tx["x_location"].to_numpy(np.float64)
    assert y.min() >= ey[0] and y.max() < ey[-1] and x.min() >= ex[0] and x.max() < ex[-1]
    return np.histogram2d(y, x, bins=[ey, ex])[0]


@pytest.mark.parametrize("which", ["gene", "all"])
def test_density_grid_equals_a_numpy_histogram_of_the_filtered_transcripts(xe, raw, D, top_gene, which):
    if which == "all" and raw.n_transcripts > 20_000_000:
        pytest.skip("all transcripts of a large bundle are not read into memory here")
    gene = top_gene if which == "gene" else xe._DENSITY_ALL
    tx = _filtered_tx(raw, D, top_gene if which == "gene" else None)
    scan = xe._density_scan(D["data_dir"], gene)
    assert scan["source"] == "parquet" and scan["total"] == len(tx) == int(scan["n"].sum())
    assert len(np.unique(np.column_stack([scan["iy"], scan["ix"]]), axis=0)) == len(scan["n"])
    xe._density_write(xe._density_cache_file(gene), scan, gene)
    for b in xe._DENSITY_BINS:
        g = xe._density_grid(gene, b)
        assert g["bin"] == b and g["total"] == len(tx) and g["grid"].sum() == len(tx)
        np.testing.assert_array_equal(g["grid"], _hist(tx, b, g["iy0"], g["ix0"], *g["grid"].shape))


def test_density_from_the_zarr_tiles_equals_the_parquet_grid(xe, raw, D, top_gene):
    if not os.path.exists(os.path.join(D["data_dir"], "transcripts.zarr.zip")):
        pytest.skip("bundle without transcripts.zarr.zip")
    if raw.n_transcripts > 20_000_000:
        pytest.skip("large bundle")
    for gene in (top_gene, xe._DENSITY_ALL):
        a = xe._density_scan(D["data_dir"], gene, source="parquet")
        b = xe._density_scan(D["data_dir"], gene, source="zarr")
        assert b["source"] == "zarr" and a["total"] == b["total"] > 0
        for k in ("iy", "ix", "n"):
            np.testing.assert_array_equal(a[k], b[k])


def test_density_bins_line_up_with_the_cells(xe, raw, D, top_gene):
    """The bins of a window around a cell hold the transcripts that lie around that cell."""
    tx = _filtered_tx(raw, D, top_gene)
    xe._density_write(xe._density_cache_file(top_gene), xe._density_scan(D["data_dir"], top_gene), top_gene)
    x, y = tx["x_location"].to_numpy(np.float64), tx["y_location"].to_numpy(np.float64)
    df = D["df"]
    j = D["gene_name_to_idx"][top_gene]
    col = np.asarray(sp.csc_matrix(D["expr"])[:, j].todense()).ravel()[np.asarray(D["df_to_expr"])]
    for ci in np.argsort(col)[::-1][:3]:                 # the cells with most of this gene
        cx, cy = float(df["x_centroid"].iloc[ci]), float(df["y_centroid"].iloc[ci])
        for b in (5, 20):
            win = xe._density_window(top_gene, b, cx - 30, cy - 30, cx + 30, cy + 30)
            v = win["values"]
            assert win["cell"] == b and not win["coarsened"]
            g = xe._density_grid(top_gene, b)          # (the window is clipped to the grid)
            assert win["x0"] == max(np.floor((cx - 30) / b), g["ix0"]) * b
            assert win["y0"] == max(np.floor((cy - 30) / b), g["iy0"]) * b
            inside = ((x >= win["x0"]) & (x < win["x0"] + b * v.shape[1])
                      & (y >= win["y0"]) & (y < win["y0"] + b * v.shape[0]))
            assert v.sum() == inside.sum() > 0
            # the bin under the centroid counts the transcripts of that bin
            r, c = int((cy - win["y0"]) // b), int((cx - win["x0"]) // b)
            assert v[r, c] == ((x // b == cx // b) & (y // b == cy // b)).sum()
    # a window too large to send is averaged over f × f bins: the total is kept
    full = xe._density_window(top_gene, 5, max_cells=10 ** 12)
    coarse = xe._density_window(top_gene, 5, max_cells=max(4, full["values"].size // 9))
    f = coarse["cell"] / 5
    assert coarse["coarsened"] and f >= 2 and coarse["bin"] == 5
    assert coarse["values"].sum() * f * f == pytest.approx(full["values"].sum(), rel=1e-5)
    assert (coarse["x0"], coarse["y0"], coarse["vmax"]) == (full["x0"], full["y0"], full["vmax"])


def test_density_tab_job_cache_heat_layer_in_both_renderers(xe, raw, D, client, cache_dir, top_gene):
    tx = _filtered_tx(raw, D, top_gene)
    opts, _ = xe.update_density_gene_options(0, None)
    assert opts[0]["value"] == xe._DENSITY_ALL and top_gene in [o["value"] for o in opts]
    assert xe.run_density(1, None)[2].startswith("Pick a gene")
    assert xe.run_density(1, "NOT_A_GENE")[2] == "'NOT_A_GENE' is not a measured gene of this dataset."
    # before it is computed: nothing drawn, and both views say so
    store, hint = xe.update_density_store(["on"], top_gene, 10, 0.6, None, 0, 0)
    assert store["on"] is False and "press Compute" in hint
    q = f"gene={top_gene}&bin=10&x0=-1e9&y0=-1e9&x1=1e9&y1=1e9"
    assert tl.get_bin(client, f"/viewer/extra/density.bin?{q}")[0]["ready"] is False
    method = (D.get("cluster_methods") or [None])[0]
    spec = {"on": True, "gene": top_gene, "bin": 10, "opacity": 0.6, "max": None}
    assert xe.make_spatial_fig("transcript_counts", method, None, 3, 0.8, [], None, density=spec).data[0].type == "scattergl"
    # compute through the tab: a registered job, the grid in the cache dir
    assert xe.run_density(1, top_gene) == (True, False, "Binning transcripts…")
    msg = tl.wait_state(xe._density_state, xe._density_lock)
    job = [j for j in xe._jobs.values() if j.kind == "density"][-1]
    job.thread.join(10)
    assert job.state == "done" and job.label == f"Transcript density: {top_gene}"
    assert f"{top_gene}: {len(tx):,} transcripts" in msg
    path = xe._density_cache_file(top_gene)
    assert os.path.exists(path) and path.startswith(str(cache_dir) + os.sep)
    name = os.path.basename(path)
    assert name.startswith(top_gene[:10]) and "__b5um__" in name and xe._DENSITY_FILTER_VERSION in name
    assert xe.poll_density(1, 0) == (msg, True, False, 1)
    # a second request reads the cache instead of the transcripts
    mtime = os.path.getmtime(path)
    xe.run_density(2, top_gene)
    assert tl.wait_state(xe._density_state, xe._density_lock) == "Grid loaded from the cache."
    assert os.path.getmtime(path) == mtime
    store, hint = xe.update_density_store(["on"], top_gene, 10, 0.6, None, 1, 0)
    assert store == {"on": True, "gene": top_gene, "bin": 10, "opacity": 0.6, "max": None, "version": "0|1"}
    assert f"{len(tx):,} transcripts" in hint
    assert xe.update_density_store([], top_gene, 10, 0.6, None, 1, 0)[0]["on"] is False
    g = xe._density_grid(top_gene, 10)
    # WebGL: the grid of the viewport
    h, a = tl.get_bin(client, f"/viewer/extra/density.bin?{q}")
    assert h["ready"] and h["bin"] == 10 and h["total"] == len(tx)
    assert (h["x0"], h["y0"]) == (g["ix0"] * 10, g["iy0"] * 10) and h["vmax"] == g["vmax"]
    assert h["label"] == f"{top_gene}: transcripts per 10 µm bin"
    f = int(round(h["cell"] / 10))                      # > 1: a large slide is sent averaged
    assert h["coarsened"] == h["subsampled"] == (f > 1) and h["h"] * h["w"] <= xe._DENSITY_MAX_CELLS
    gh, gw = g["grid"].shape
    padded = np.pad(g["grid"], ((0, -gh % f), (0, -gw % f)))
    blocks = padded.reshape(padded.shape[0] // f, f, padded.shape[1] // f, f).sum(axis=(1, 3)) / (f * f)
    np.testing.assert_allclose(a["values"].reshape(h["h"], h["w"]), blocks, rtol=1e-6)
    x, y = tx["x_location"].to_numpy(np.float64), tx["y_location"].to_numpy(np.float64)
    mx, my = float(np.median(x)), float(np.median(y))
    h2, a2 = tl.get_bin(client, f"/viewer/extra/density.bin?gene={top_gene}&bin=20"
                                f"&x0={mx - 40}&y0={my - 40}&x1={mx + 40}&y1={my + 40}")
    v2 = a2["values"].reshape(h2["h"], h2["w"])
    inside = ((x >= h2["x0"]) & (x < h2["x0"] + 20 * h2["w"]) & (y >= h2["y0"]) & (y < h2["y0"] + 20 * h2["h"]))
    assert h2["cell"] == 20 and v2.sum() == inside.sum() and h2["x0"] % 20 == 0 and h2["y0"] % 20 == 0
    assert tl.get_bin(client, "/umap/viewer/extra/density.bin?" + q, expect=404) is not None
    # Plotly: a heatmap trace under the cells, in plot space (y = -µm)
    fig = xe.make_spatial_fig("transcript_counts", method, None, 3, 0.8, [], None, density=spec)
    hm = fig.data[0]
    assert hm.type == "heatmap" and fig.data[1].type == "scattergl" and hm.opacity == 0.6
    assert (hm.zmin, hm.zmax) == (0, g["vmax"])
    assert np.nansum(np.asarray(hm.z, dtype=float)) * (hm.dx / 10) ** 2 == pytest.approx(len(tx), rel=1e-5)
    # ... for the current viewport only (never averaged there), with the colour maximum of the tab
    rl = {"xaxis.range[0]": mx - 40, "xaxis.range[1]": mx + 40,
          "yaxis.range[0]": -my - 40, "yaxis.range[1]": -my + 40}
    hm = xe.make_spatial_fig("gene", method, top_gene, 3, 0.8, [], rl, density={**spec, "max": 7}).data[0]
    assert hm.type == "heatmap" and hm.zmax == 7 and hm.dx == hm.dy == 10
    assert hm.colorbar.x > 1.05, "the density colour bar sits beside the cells' own"
    z = np.asarray(hm.z, dtype=float)
    near = ((x >= hm.x0 - 5) & (x < hm.x0 - 5 + 10 * z.shape[1])
            & (-y > hm.y0 - 5) & (-y <= hm.y0 - 5 + 10 * z.shape[0]))
    assert near.sum() > 0 and z.shape[1] * 10 <= 80 * 1.5 + 20
    x, y = x[near], y[near]
    # x0 / y0 are the centres of the first column / row; a bin [Y, Y + b) in µm is the
    # interval (-Y - b, -Y] of the plot's y axis
    cols = np.floor((x - (hm.x0 - hm.dx / 2)) / hm.dx).astype(int)
    rows = (np.ceil((-y - (hm.y0 - hm.dy / 2)) / hm.dy) - 1).astype(int)
    recount = np.zeros_like(z)
    np.add.at(recount, (rows, cols), 1)
    np.testing.assert_array_equal(np.nan_to_num(z), recount)


def test_density_job_can_be_cancelled_and_leaves_no_grid(xe, D, top_gene, monkeypatch):
    seen = []

    def slow_batches(path, gene, qv_min=20):
        for i in range(400):
            seen.append(i)
            time.sleep(0.02)
            yield np.array([1.0], np.float32), np.array([1.0], np.float32), i / 400

    monkeypatch.setattr(xe, "_density_tx_batches_parquet", slow_batches)
    assert xe.run_density(1, top_gene)[:2] == (True, False)
    job = [j for j in xe._jobs.values() if j.kind == "density"][-1]
    while not seen:
        time.sleep(0.01)
    assert job.snapshot()["cancellable"] and xe._job_cancel(job.id)
    job.thread.join(10)
    assert job.state == "cancelled" and len(seen) < 400
    assert xe._density_state["status"] == "error" and xe._density_state["message"] == "Cancelled"
    assert not os.path.exists(xe._density_cache_file(top_gene))
    assert "transcript density" not in xe._running_jobs()


def test_analysis_layout_and_state_are_wired(xe):
    ids = set()

    def walk(c):
        if getattr(c, "id", None):
            ids.add(c.id)
        ch = getattr(c, "children", None)
        for k in (ch if isinstance(ch, (list, tuple)) else [ch]):
            if hasattr(k, "to_plotly_json"):
                walk(k)

    walk(xe.app.layout)
    assert {"de-group-a", "de-group-b", "de-run-btn", "de-save-group-btn", "de-volcano-plot", "de-table",
            "de-download-btn", "de-plot-w", "de-caveat", "geneset-genes", "geneset-run-btn",
            "geneset-select", "density-gene", "density-bin", "density-opacity", "density-run-btn",
            "density-store"} <= ids
    assert "independent observation" in xe._DE_CAVEAT and "pseudobulk" in xe._DE_CAVEAT
    for kind in ("de", "geneset", "density"):
        assert kind in xe._JOB_KIND_NAMES
