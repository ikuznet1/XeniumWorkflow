"""Statistics across samples and in space: the sample design, pseudobulk differential
expression, composition tests, multi-sample density / corrected counts, Moran's I and
ligand-receptor scores.

Every number is checked against an independent computation: dense numpy / scipy
formulas written here, and - when R with the package is installed - DESeq2, edgeR, limma
and spdep run by ``Rscript`` on the same matrix (those tests skip otherwise). Several
samples are fabricated: random counts over the bundle's own gene list, placed beside the
bundle as "extra samples" the way '+ Add sample' does.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy import stats as sstats

import xe_testlib as tl

N_GENES = 300          # genes of the fabricated samples that carry counts


@pytest.fixture(autouse=True)
def _clean_stats_state(xe):
    yield
    shutil.rmtree(os.path.dirname(xe._density_cache_file("x")), ignore_errors=True)
    xe._an_autoload()
    xe._stats_reset()


# ── fabricated samples ───────────────────────────────────────────────────────
def _fab_sample(D, n, seed, effect=None, types=None, type_p=None, n_genes=N_GENES):
    """A dataset dict like an added sample: *n* cells with negative-binomial-like counts
    in the first *n_genes* genes of the bundle's panel (gene means × a per-sample
    factor; *effect*: per-gene multiplier), random centroids, optional cell types."""
    rng = np.random.default_rng(seed)
    genes = list(D["gene_names"])
    base = np.random.default_rng(1).gamma(2.0, 1.5, size=n_genes)        # same in every sample
    mu = base * np.exp(rng.normal(0, 0.25, size=n_genes))
    if effect is not None:
        mu = mu * effect
    lib = rng.gamma(8.0, 1 / 8.0, size=n)
    dense = rng.poisson(lib[:, None] * mu[None, :] * 0.3)
    m = sp.lil_matrix((n, len(genes)), dtype=np.float32)
    m[:, :n_genes] = dense
    df = pd.DataFrame({"x_centroid": rng.uniform(0, 400, n), "y_centroid": rng.uniform(0, 400, n),
                       "transcript_counts": dense.sum(axis=1)},
                      index=[f"f{seed}_{i}" for i in range(n)])
    ds = {"expr": m.tocsr(), "df_to_expr": np.arange(n), "gene_name_to_idx": D["gene_name_to_idx"],
          "gene_names": genes, "df": df, "x_offset": 1000.0 * (seed + 1), "cluster_methods": []}
    if types is not None:
        ds["_cell_type_labels"] = pd.Series(rng.choice(types, size=n, p=type_p), index=df.index)
    return ds


@pytest.fixture
def six(xe, D):
    """Six fabricated samples beside the bundle: S1-S3 in A, S4-S6 in B, the bundle (S0)
    excluded. Genes 0-19 are 3× higher in A. Returns the extra dataset dicts."""
    effect = np.ones(N_GENES)
    effect[:20] = 3.0
    extras = [_fab_sample(D, 220 + 15 * i, i, effect if i < 3 else None) for i in range(6)]
    xe.EXTRA_DATASETS.extend(extras)
    xe._design_set([{"key": xe._ds_key(e), "cond": "A" if i < 3 else "B", "batch": ""}
                    for i, e in enumerate(extras)], {"A": "treated", "B": "control"})
    return extras


def _rscript(code: str, tmp_path, name="ref.R") -> None:
    path = tmp_path / name
    path.write_text(code)
    out = subprocess.run(["Rscript", str(path)], capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]


def _need_r(*pkgs):
    if not tl.r_packages_available(*pkgs):
        pytest.skip(f"R with {', '.join(pkgs)} is not available")


def _need_rpy2():
    pytest.importorskip("rpy2")
    import rpy2.robjects as ro
    import rpy2.robjects.conversion as conversion
    conversion.set_conversion(ro.default_converter)


# ── the sample design ────────────────────────────────────────────────────────
def test_design_is_kept_per_sample_and_summarised(xe, D, six):
    rows = xe._design_samples()
    assert [r["label"] for r in rows] == [f"S{i}" for i in range(7)]
    assert [r["cond"] for r in rows] == ["", "A", "A", "A", "B", "B", "B"]
    assert rows[0]["key"] == xe._ds_key(D) and rows[1]["n_cells"] == 220
    text = xe._design_summary()
    assert "A (treated): S1, S2, S3" in text and "B (control): S4, S5, S6" in text and "excluded: S0" in text
    # the assignment follows the sample, not its position
    first = xe.EXTRA_DATASETS.pop(0)
    assert [r["cond"] for r in xe._design_samples()] == ["", "A", "A", "B", "B", "B"]
    xe.EXTRA_DATASETS.append(first)
    assert xe._design_samples()[-1]["cond"] == "A"
    # the editor: one row per sample; Apply stores what its pickers hold
    is_open, children, name_a, name_b, _ = xe.open_design_modal(1, 0)
    assert is_open and len(children) == 1 + 7 and (name_a, name_b) == ("treated", "control")
    with pytest.raises(Exception):
        xe.open_design_modal(0, 0)                       # a re-created button, not a click
    ids = [{"type": "design-cond", "index": i} for i in range(7)]
    ver, msg = xe.apply_design(1, ["B", "A", "", "A", "B", "B", "A"], ["p1"] * 7, ids, "x", "y", 3)
    assert ver == 4 and msg.startswith("Saved: 3 samples in A, 3 in B, 1 excluded")
    assert xe._design_names() == {"A": "x", "B": "y"} and xe._design_samples()[0]["batch"] == "p1"
    assert xe.apply_design(2, ["A"] * 7, [""] * 7, ids, "same", "same", 4)[1].startswith("Give the two")
    assert "changed while the editor was open" in xe.apply_design(
        3, ["A"] * 8, [""] * 8, ids + [{"type": "design-cond", "index": 9}], "a", "b", 4)[1]
    assert xe.update_design_summaries(0, 0, 0)[0] == xe.update_design_summaries(0, 0, 0)[1]
    xe._stats_reset()
    assert all(r["cond"] == "" for r in xe._design_samples())


def test_design_check_refuses_and_warns(xe):
    with pytest.raises(ValueError, match="Condition B has 1 usable sample; at least 2"):
        xe._design_check(["A", "A", "B"], ["", "", ""])
    batch, warn = xe._design_check(["A", "A", "B", "B", "B"], [""] * 5)
    assert batch is None and len(warn) == 1 and "condition A has only 2 samples" in warn[0]
    assert xe._design_check(["A"] * 3 + ["B"] * 3, [""] * 6) == (None, [])
    with pytest.raises(ValueError, match="some samples only"):
        xe._design_check(["A", "A", "B", "B"], ["p1", "", "p1", "p2"])
    with pytest.raises(ValueError, match="confounded"):
        xe._design_check(["A", "A", "B", "B"], ["p1", "p1", "p2", "p2"])
    with pytest.raises(ValueError, match="confounded|degrees of freedom"):
        xe._design_check(["A", "A", "B", "B"], ["p1", "p2", "p3", "p4"])
    batch, _ = xe._design_check(["A", "A", "B", "B"], ["p1", "p2", "p1", "p2"])
    assert batch == ["p1", "p2", "p1", "p2"]
    X = xe._pb_design_matrix(["A", "A", "B", "B"], batch)
    np.testing.assert_array_equal(X, [[1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 0]])


# ── pseudobulk: the matrix ───────────────────────────────────────────────────
def test_pseudobulk_matrix_is_the_sum_of_raw_counts_per_sample(xe, D, six):
    samples = xe._an_samples("xenium", False)
    assert [s["label"] for s in samples] == [f"S{i}" for i in range(7)]
    masks, label = xe._pb_population_masks("all", samples, "xenium", None, "cluster")
    conds, batches = xe._design_of(samples)
    assert label == "all cells" and conds == ["", "A", "A", "A", "B", "B", "B"]
    prep = xe._pb_prepare(samples, masks, conds, batches, min_cells=10, min_count=10)
    assert prep["labels"] == [f"S{i}" for i in range(1, 7)] and prep["design"] == "~ condition"
    assert prep["cond"] == ["A"] * 3 + ["B"] * 3 and prep["batch"] is None and prep["warnings"] == []
    genes = xe._an_common_genes(samples)
    full = np.column_stack([np.asarray(e["expr"].sum(axis=0)).ravel() for e in six]).astype(np.int64)
    full = full[[D["gene_name_to_idx"][g] for g in genes]]
    keep = full.sum(axis=1) >= 10
    assert prep["counts"].dtype == np.int64 and prep["genes"] == [g for g, k in zip(genes, keep) if k]
    np.testing.assert_array_equal(prep["counts"], full[keep])
    assert prep["n_filtered"] == int((~keep).sum()) and prep["n_filtered"] >= len(genes) - N_GENES
    rows = {r["label"]: r for r in prep["sample_rows"]}
    assert rows["S0"]["note"] == "excluded" and not rows["S0"]["used"]
    assert rows["S2"]["n_cells"] == len(six[1]["df"]) and rows["S2"]["lib_size"] == int(six[1]["expr"].sum())
    # a population: only its cells are summed; a sample with too few of them is dropped and named
    m2 = [np.zeros(len(s["df"]), dtype=bool) for s in samples]
    for i in range(1, 7):
        m2[i][: (5 if i == 6 else 40)] = True
    p2 = xe._pb_prepare(samples, m2, conds, batches, min_cells=10, min_count=0)
    assert p2["labels"] == ["S1", "S2", "S3", "S4", "S5"] and p2["n_filtered"] == 0
    r6 = [r for r in p2["sample_rows"] if r["label"] == "S6"][0]
    assert r6["note"] == "dropped: fewer than 10 cells" and r6["n_cells"] == 5
    assert any("condition B (control) has only 2 samples" in w for w in p2["warnings"])
    col = np.asarray(six[2]["expr"][:40].sum(axis=0)).ravel()[[D["gene_name_to_idx"][g] for g in p2["genes"]]]
    np.testing.assert_array_equal(p2["counts"][:, 2], col.astype(np.int64))
    # fewer than 2 samples left in a condition: refused, with what was dropped
    for i in (4, 5):
        m2[i][:] = False
    with pytest.raises(ValueError, match=r"Condition B \(control\) has 0 usable samples.*Dropped: S4"):
        xe._pb_prepare(samples, m2, conds, batches, min_cells=10, min_count=0)
    # non-integer values are not counts
    six[0]["expr"] = six[0]["expr"] * 0.5
    with pytest.raises(ValueError, match="non-integer"):
        xe._pb_prepare(xe._an_samples("xenium", False), masks, conds, batches)


def test_pseudobulk_populations(xe, D, six, labels):
    for i, e in enumerate(six):
        e["_cell_type_labels"] = pd.Series(np.where(np.arange(len(e["df"])) % 3 == 0, "Alpha", "Beta"),
                                           index=e["df"].index)
        e["df"]["clust__k"] = (np.arange(len(e["df"])) % 4).astype(str)
        e["cluster_methods"] = ["k"]
    samples = xe._an_samples("xenium", False)
    opts = {o["value"] for o in xe._pb_population_options("xenium", "k", "cell_type:celltypist")}
    assert {"all", "celltype:Alpha", "celltype:Beta", "cluster:0", "cluster:3"} <= opts and "rest" not in opts
    masks, label = xe._pb_population_masks("celltype:Alpha", samples, "xenium", "k", "cell_type:celltypist")
    assert label == "Alpha" and int(masks[1].sum()) == len(np.arange(220)[::3])
    assert int(masks[0].sum()) == int((labels == "Alpha").sum())
    masks, label = xe._pb_population_masks("cluster:3", samples, "xenium", "k", "cluster")
    assert label == "cluster 3" and [int(m.sum()) for m in masks[1:3]] == [55, 58]     # 220 / 4, 235 // 4
    with pytest.raises(ValueError, match="Pick the cell population"):
        xe._pb_population_masks("rest", samples, "xenium", "k", "cluster")
    # a ROI class is the union of its ROIs (one drawn over each of two samples)
    def box(ds):
        x = ds["df"]["x_centroid"].to_numpy() + ds["x_offset"]
        y = -ds["df"]["y_centroid"].to_numpy()
        return [[x.min() - 1, y.min() - 1], [x.max() + 1, y.min() - 1], [x.max() + 1, y.max() + 1],
                [x.min() - 1, y.max() + 1], [x.min() - 1, y.min() - 1]]
    with xe._roi_lock:
        xe._roi_state["rois"] = [{"name": "r1", "cls": "tumour", "polygon_xy": box(six[0])},
                                 {"name": "r2", "cls": "tumour", "polygon_xy": box(six[3])}]
    assert "roicls:tumour" in {o["value"] for o in xe._pb_population_options("xenium", "k", "cluster")}
    masks, label = xe._pb_population_masks("roicls:tumour", samples, "xenium", "k", "cluster")
    assert label == "ROI class tumour"
    assert [bool(m.all()) for m in masks] == [False, True, False, False, True, False, False]
    assert not masks[2].any()


def test_pseudobulk_is_refused_without_enough_samples(xe, D):
    """One sample, and a condition with one sample: an error for the user, no model."""
    method = (D.get("cluster_methods") or [None])[0]
    assert xe.run_de(1, None, None, "xenium", method, "cluster", "original", "pseudobulk", None)[2] \
        .startswith("Pick the cell population")
    try:
        import rpy2  # noqa: F401
    except Exception:
        pytest.skip("rpy2 is not installed: the callback reports that instead")
    assert xe.run_de(1, None, None, "xenium", method, "cluster", "original", "pseudobulk", "all",
                     10, 10, ["on"]) == (True, False, "Running…")
    with pytest.raises(AssertionError, match="Condition A has 0 usable samples; at least 2"):
        tl.wait_state(xe._de_state, xe._de_lock)
    xe._test_baseline.settle()
    xe.EXTRA_DATASETS.extend(_fab_sample(D, 60, i) for i in range(3))
    xe._design_set([{"key": xe._ds_key(e), "cond": c, "batch": ""}
                    for e, c in zip(xe.EXTRA_DATASETS, ["A", "B", "B"])])
    xe.run_de(2, None, None, "xenium", method, "cluster", "original", "pseudobulk", "all", 10, 10, ["on"])
    with pytest.raises(AssertionError, match="Condition A has 1 usable sample; at least 2"):
        tl.wait_state(xe._de_state, xe._de_lock)


# ── pseudobulk: the model, against R run directly ─────────────────────────────
_DESEQ_REF = r"""
suppressPackageStartupMessages(library(DESeq2))
counts <- as.matrix(read.csv("%(dir)s/counts.csv", row.names = 1, check.names = FALSE))
storage.mode(counts) <- "integer"
cd <- read.csv("%(dir)s/coldata.csv", row.names = 1, stringsAsFactors = FALSE)
cd$condition <- factor(cd$condition, levels = c("B", "A"))
if (%(batch)s) { cd$batch <- factor(cd$batch); dsg <- ~ batch + condition } else dsg <- ~ condition
dds <- DESeq(DESeqDataSetFromMatrix(counts, cd, dsg), quiet = TRUE)
res <- results(dds, name = "condition_A_vs_B", alpha = 0.05)
out <- data.frame(gene = rownames(res), log2fc = res$log2FoldChange, lfc_se = res$lfcSE, stat = res$stat,
                  pval = res$pvalue, padj = res$padj, base_mean = res$baseMean)
if (requireNamespace("apeglm", quietly = TRUE)) {
  out$shrunk <- lfcShrink(dds, coef = "condition_A_vs_B", type = "apeglm", quiet = TRUE)$log2FoldChange
} else out$shrunk <- lfcShrink(dds, coef = "condition_A_vs_B", type = "normal", quiet = TRUE)$log2FoldChange
write.csv(out, "%(dir)s/ref.csv", row.names = FALSE)
write.csv(data.frame(sf = sizeFactors(dds)), "%(dir)s/sf.csv")
"""

_EDGER_REF = r"""
suppressPackageStartupMessages(library(edgeR))
counts <- as.matrix(read.csv("%(dir)s/counts.csv", row.names = 1, check.names = FALSE))
cd <- read.csv("%(dir)s/coldata.csv", row.names = 1, stringsAsFactors = FALSE)
condition <- factor(cd$condition, levels = c("B", "A"))
design <- model.matrix(~ condition)
y <- estimateDisp(calcNormFactors(DGEList(counts = counts)), design)
tt <- glmQLFTest(glmQLFit(y, design), coef = 2)$table
write.csv(data.frame(gene = rownames(tt), log2fc = tt$logFC, stat = tt$F, pval = tt$PValue,
                     padj = p.adjust(tt$PValue, "BH")), "%(dir)s/ref.csv", row.names = FALSE)
"""


def _write_matrix(prep, tmp_path):
    pd.DataFrame(prep["counts"], index=prep["genes"], columns=prep["labels"]).to_csv(tmp_path / "counts.csv")
    pd.DataFrame({"condition": prep["cond"], "batch": prep["batch"] or [""] * len(prep["cond"])},
                 index=prep["labels"]).to_csv(tmp_path / "coldata.csv")


def _same(a, b, rtol=1e-6, atol=1e-9):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize("batch", [False, True], ids=["condition", "batch+condition"])
def test_pseudobulk_deseq2_equals_deseq2_run_directly_in_r(xe, D, six, tmp_path, batch):
    _need_r("DESeq2")
    _need_rpy2()
    if batch:
        xe._design_set([{"key": xe._ds_key(e), "cond": "A" if i < 3 else "B", "batch": f"p{i % 3}"}
                        for i, e in enumerate(six)])
    samples = xe._an_samples("xenium", False)
    masks, _ = xe._pb_population_masks("all", samples, "xenium", None, "cluster")
    prep = xe._pb_prepare(samples, masks, *xe._design_of(samples))
    assert prep["design"] == ("~ batch + condition" if batch else "~ condition")
    fit = xe._pb_fit(prep, shrink=True)
    _write_matrix(prep, tmp_path)
    _rscript(_DESEQ_REF % {"dir": tmp_path, "batch": "TRUE" if batch else "FALSE"}, tmp_path)
    ref = pd.read_csv(tmp_path / "ref.csv")
    t = fit["table"]
    assert fit["engine"] == "deseq2" and "DESeq2" in fit["method_label"] and "Wald" in fit["method_label"]
    assert list(t["gene"]) == list(ref["gene"]) and len(t) > 250
    for col in ("log2fc", "lfc_se", "stat", "pval", "padj", "base_mean"):
        _same(t[col], ref[col])
    assert fit["shrink"] in ("apeglm", "normal") and "log2fc_shrunk" in t.columns
    assert fit["shrink"] == ("apeglm" if tl.r_packages_available("apeglm") else "normal")
    _same(t["log2fc_shrunk"], ref["shrunk"], rtol=1e-5, atol=1e-7)
    _same(fit["size_factors"], pd.read_csv(tmp_path / "sf.csv")["sf"])
    # the planted genes (3× in A) are found, with the sign of A / B
    hit = t.iloc[:20] if list(t["gene"][:20]) == prep["genes"][:20] else t[t["gene"].isin(prep["genes"][:20])]
    assert (hit["padj"] < 0.05).mean() > 0.8 and (hit["log2fc"] > 0.8).mean() > 0.8
    assert (t["padj"].iloc[40:] < 0.05).mean() < 0.1
    # unshrunken on request: no shrunken column, same test
    plain = xe._pb_fit(prep, shrink=False)
    assert plain["shrink"] == "none" and "log2fc_shrunk" not in plain["table"].columns
    _same(plain["table"]["padj"], ref["padj"])


def test_pseudobulk_edger_fallback_equals_edger_run_directly_in_r(xe, D, six, tmp_path):
    _need_r("edgeR")
    _need_rpy2()
    samples = xe._an_samples("xenium", False)
    masks, _ = xe._pb_population_masks("all", samples, "xenium", None, "cluster")
    prep = xe._pb_prepare(samples, masks, *xe._design_of(samples))
    fit = xe._pb_fit(prep, engine="edger")
    _write_matrix(prep, tmp_path)
    _rscript(_EDGER_REF % {"dir": tmp_path}, tmp_path)
    ref = pd.read_csv(tmp_path / "ref.csv")
    assert fit["engine"] == "edger" and "edgeR" in fit["method_label"] and "quasi-likelihood" in fit["method_label"]
    assert "DESeq2 is not installed" in fit["method_label"] and fit["shrink"] == "none"
    for col in ("log2fc", "stat", "pval", "padj"):
        _same(fit["table"][col], ref[col])


def test_pseudobulk_job_fills_the_de_tab(xe, D, six, tmp_path):
    """Run button → job → volcano, table, per-sample table, CSV; corrected mode is stated."""
    _need_r("DESeq2")
    _need_rpy2()
    method = (D.get("cluster_methods") or [None])[0]
    assert xe.toggle_de_mode("pseudobulk")[2] == xe._PB_CAVEAT and xe.toggle_de_mode("cells")[2] == xe._DE_CAVEAT
    opts, value = xe.update_pb_population_options("tab-de", "pseudobulk", 0, 0, 0, 0, 0, 0, "xenium",
                                                  method, "cluster", 0, "gone")
    assert value == "all" and opts[0] == {"label": "All cells", "value": "all"}
    assert xe.run_de(1, None, None, "xenium", method, "cluster", "corrected", "pseudobulk", "all",
                     10, 10, ["on"]) == (True, False, "Running…")
    msg = tl.wait_state(xe._de_state, xe._de_lock, timeout=600)
    job = [j for j in xe._jobs.values() if j.kind == "de"][-1]
    job.thread.join(10)
    assert job.state == "done" and job.label == "Pseudobulk differential expression"
    assert "A treated [S1, S2, S3] vs B control [S4, S5, S6]" in msg and "design ~ condition" in msg
    assert "DESeq2" in msg and "Original counts." in msg
    assert "the counts mode is SPLIT-corrected, but pseudobulk uses the original integer counts" in msg
    res = xe._de_state["result"]
    assert res["kind"] == "pseudobulk" and res["n_a"] == res["n_b"] == 3 and not res["corrected"]
    assert res["n_filtered"] >= len(D["gene_names"]) - N_GENES and f"{res['n_filtered']:,} with a total count below 10" in msg
    samples = xe._an_samples("xenium", False)
    masks, _ = xe._pb_population_masks("all", samples, "xenium", None, "cluster")
    prep = xe._pb_prepare(samples, masks, *xe._design_of(samples))
    _write_matrix(prep, tmp_path)
    _rscript(_DESEQ_REF % {"dir": tmp_path, "batch": "FALSE"}, tmp_path)
    ref = pd.read_csv(tmp_path / "ref.csv")
    _same(res["table"]["log2fc"], ref["log2fc"])
    _same(res["table"]["padj"], ref["padj"])
    # volcano: the shrunken fold change, named; genes without an adjusted p are not drawn
    fig, rows = xe.update_de_results(1, 0, None, None)
    assert "Pseudobulk, all cells: A treated (3 samples) vs B control (3 samples)" in fig.layout.title.text
    assert "~ condition" in fig.layout.title.text and res["shrink"] in fig.layout.xaxis.title.text
    n_drawn = sum(int(np.isfinite(np.asarray(t.y, dtype=float)).sum()) for t in fig.data)
    assert n_drawn == int(res["table"]["padj"].notna().sum())
    cols, sample_view = xe.update_de_table_columns(1, 0)
    ids = [c["id"] for c in cols]
    assert ids[:3] == ["gene", "log2fc", "log2fc_shrunk"] and "base_mean" in ids and "pct_a" not in ids
    assert [c["name"] for c in cols if c["id"] == "stat"] == ["Wald"]
    assert set(rows[0]) == set(ids) and len(rows) == len(res["table"])
    assert rows[0]["gene"] in prep["genes"][:20]
    table = sample_view[0]
    assert len(table.children[1].children) == 7                       # one row per loaded sample
    # CSV: method, design and every sample in the header
    out = xe.download_de_table(1)
    head = [l for l in out["content"].splitlines() if l.startswith("#")]
    assert out["filename"] == "pseudobulk_all_cells_treated_vs_control.csv"
    assert any("design: ~ condition" in l and "DESeq2" in l for l in head)
    assert sum(l.startswith("# sample S") for l in head) == 7 and any("S0" in l and "excluded" in l for l in head)
    assert any(l.startswith("# sample S1") and "220 cells" in l and "size factor" in l for l in head)
    body = pd.read_csv(io.StringIO(out["content"]), comment="#")
    _same(body.set_index("gene").loc[ref["gene"], "padj"], ref["padj"])
    # the cell-level test afterwards gets its own columns back
    a = np.arange(len(D["df"])) < len(D["df"]) // 2
    with xe._de_lock:
        xe._de_state["result"] = None
    assert [c["id"] for c in xe.update_de_table_columns(2, 0)[0]][2] == "pct_a"
    assert xe.update_de_table_columns(2, 0)[1] == [] and a.any()


# ── composition ──────────────────────────────────────────────────────────────
def _comp_counts(seed=0, n_cat=7, n_a=4, n_b=5):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.full(n_cat, 3.0))
    out = []
    for i in range(n_a + n_b):
        q = p * np.exp(rng.normal(0, 0.2, n_cat))
        if i < n_a:
            q[0] *= 2.5
        out.append(rng.multinomial(1500 + 200 * i, q / q.sum()))
    return np.array(out).T, ["A"] * n_a + ["B"] * n_b


@pytest.mark.parametrize("transform", ["logit", "asin"])
def test_composition_welch_and_transforms_equal_scipy(xe, transform):
    counts, conds = _comp_counts()
    t = xe._comp_test(counts, conds, None, transform)
    a = np.array(conds) == "A"
    props = counts / counts.sum(axis=0)
    if transform == "logit":
        pp = (counts + 0.5) / (counts + 0.5).sum(axis=0)
        y = np.log(pp / (1 - pp))
    else:
        y = np.arcsin(np.sqrt(props))
    np.testing.assert_allclose(xe._comp_transform(counts, transform), y, rtol=1e-12)
    for i in range(len(counts)):
        w = sstats.ttest_ind(y[i, a], y[i, ~a], equal_var=False)
        np.testing.assert_allclose([t["t_welch"][i], t["p_welch"][i]], [w.statistic, w.pvalue], rtol=1e-10)
        np.testing.assert_allclose(t["df_welch"][i], w.df, rtol=1e-10)
        np.testing.assert_allclose(t["coef"][i], y[i, a].mean() - y[i, ~a].mean(), rtol=1e-10)
    np.testing.assert_allclose(t["mean_prop_a"], props[:, a].mean(axis=1))
    np.testing.assert_allclose(t["prop_ratio"], props[:, a].mean(axis=1) / props[:, ~a].mean(axis=1))
    # BH over the categories, written out
    p = t["p_mod"].to_numpy()
    order = np.argsort(p)
    adj = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    np.testing.assert_allclose(t["fdr_mod"].to_numpy()[order], np.minimum(adj, 1), rtol=1e-12)
    assert t["fdr_mod"][0] < 0.05 and t["t_mod"][0] > 0            # the planted category


def test_variance_shrinkage_pieces(xe):
    """The parts of the moderated t-test that need no R: the trigamma inverse, the
    posterior variance formula and the cases without a usable prior."""
    from scipy.special import polygamma
    for v in (0.05, 0.7, 4.0, 300.0, 1e-8, 1e9):
        np.testing.assert_allclose(float(polygamma(1, xe._trigamma_inverse(v))), v, rtol=1e-6)
    # fewer than 2 variances: nothing to estimate a prior from, nothing is shrunk
    post, s20, df0 = xe._squeeze_var(np.array([0.3]), 4)
    assert df0 == 0.0 and np.isnan(s20) and post[0] == 0.3
    # variances that vary more than chi-square sampling explains: a finite prior
    rng = np.random.default_rng(0)
    true = np.exp(rng.normal(0, 1.0, 400))
    s2 = true * rng.chisquare(5, 400) / 5
    post, s20, df0 = xe._squeeze_var(s2, 5)
    assert 1.0 < df0 < 20 and 0.5 < s20 < 2.0
    np.testing.assert_allclose(post, (5 * s2 + df0 * s20) / (5 + df0))
    assert np.abs(np.log(post) - np.log(true)).mean() < np.abs(np.log(s2) - np.log(true)).mean()
    # variances no more variable than chance: infinite prior df, one common variance
    s2 = rng.chisquare(50, 300) / 50 * (1 + 0.0 * true[:300])
    post, s20, df0 = xe._squeeze_var(np.full(300, 0.2) * (1 + 1e-9 * np.arange(300)), 6)
    assert np.isinf(df0) and np.allclose(post, s20) and np.isclose(s20, 0.2 * np.exp(
        -float(__import__("scipy.special", fromlist=["digamma"]).digamma(3.0)) + np.log(3.0)), rtol=1e-6)


_LIMMA_REF = r"""
suppressPackageStartupMessages(library(limma))
y <- as.matrix(read.csv("%(dir)s/y.csv", row.names = 1))
cd <- read.csv("%(dir)s/cd.csv", stringsAsFactors = FALSE)
condition <- factor(cd$condition, levels = c("B", "A"))
design <- if (%(batch)s) model.matrix(~ factor(cd$batch) + condition) else model.matrix(~ condition)
fit <- eBayes(lmFit(y, design), robust = FALSE, trend = FALSE)
k <- ncol(design)
write.csv(data.frame(t = fit$t[, k], p = fit$p.value[, k], coef = fit$coefficients[, k],
                     df_total = fit$df.total, df_prior = fit$df.prior, s2_prior = fit$s2.prior,
                     s2_post = fit$s2.post), "%(dir)s/limma.csv", row.names = FALSE)
"""


@pytest.mark.parametrize("batch", [False, True], ids=["condition", "batch+condition"])
def test_composition_moderated_t_equals_limma(xe, tmp_path, batch):
    """The propeller recipe (transformed proportions → lmFit → eBayes) run in R."""
    _need_r("limma")
    counts, conds = _comp_counts(seed=11, n_cat=9, n_a=4, n_b=4)
    batches = [f"p{i % 4}" for i in range(8)] if batch else None
    t = xe._comp_test(counts, conds, batches, "logit")
    y = xe._comp_transform(counts, "logit")
    pd.DataFrame(y, index=[f"c{i}" for i in range(len(y))]).to_csv(tmp_path / "y.csv")
    pd.DataFrame({"condition": conds, "batch": batches or [""] * 8}).to_csv(tmp_path / "cd.csv", index=False)
    _rscript(_LIMMA_REF % {"dir": tmp_path, "batch": "TRUE" if batch else "FALSE"}, tmp_path)
    ref = pd.read_csv(tmp_path / "limma.csv")
    np.testing.assert_allclose(t["coef"], ref["coef"], rtol=1e-8)
    np.testing.assert_allclose(t["prior_df"], ref["df_prior"], rtol=1e-5)
    np.testing.assert_allclose(t["prior_var"], ref["s2_prior"], rtol=1e-5)
    np.testing.assert_allclose(t["df_mod"], ref["df_total"], rtol=1e-5)
    np.testing.assert_allclose(t["t_mod"], ref["t"], rtol=1e-5)
    np.testing.assert_allclose(t["p_mod"], ref["p"], rtol=1e-4)


def test_composition_tab_proportions_test_plot_table_csv(xe, D, labels):
    types = ["Alpha", "Beta", "Gamma", "Delta"]
    extras = [_fab_sample(D, 300 + 20 * i, i, types=types,
                          type_p=[0.55, 0.2, 0.15, 0.1] if i < 3 else [0.25, 0.4, 0.2, 0.15]) for i in range(6)]
    # one sample only: its proportions, no test
    opts, value = xe.update_comp_category_options("tab-abundance", 0, 0, 0, "xenium", 0, "cluster", None)
    assert value == "celltype:celltypist" and {"cluster"} <= {o["value"] for o in opts}
    with pytest.raises(Exception):
        xe.update_comp_category_options("tab-umap", 0, 0, 0, "xenium", 0, "cluster", None)
    assert xe.run_composition(1, None, "logit", "xenium", None, 0)[1].startswith("Pick what")
    ver, msg = xe.run_composition(1, "celltype:celltypist", "logit", "xenium", None, 0)
    assert ver == 1 and msg.startswith("One sample: its proportions are shown")
    res = xe._comp_state["result"]
    exp = labels.value_counts()
    np.testing.assert_array_equal(res["counts"]["xenium"].reindex(exp.index), exp)
    assert res["tests"] is None and xe.make_composition_fig().data[0].name == "not assigned"
    # two samples per condition: still no test, said plainly
    xe.EXTRA_DATASETS.extend(extras[1:3] + extras[3:5])
    xe._design_set([{"key": xe._ds_key(e), "cond": "A" if i < 2 else "B", "batch": ""}
                    for i, e in enumerate(xe.EXTRA_DATASETS)])
    ver, msg = xe.run_composition(2, "celltype:celltypist", "logit", "xenium", None, 1)
    assert msg.startswith("No test was run: condition A has 2 and condition B has 2 samples")
    assert "at least 3 per condition" in msg and xe._comp_state["result"]["tests"] is None
    cols, rows = xe._comp_table(xe._comp_state["result"])
    assert "p_mod" not in [c["id"] for c in cols] and "prop_S1" in [c["id"] for c in cols]
    # three per condition: tested
    del xe.EXTRA_DATASETS[:]
    xe.EXTRA_DATASETS.extend(extras)
    xe._design_set([{"key": xe._ds_key(e), "cond": "A" if i < 3 else "B", "batch": ""}
                    for i, e in enumerate(extras)], {"A": "tumour", "B": "normal"})
    ver, msg = xe.run_composition(3, "celltype:celltypist", "logit", "xenium", None, 2)
    assert "4 categories tested between A tumour (3 samples) and B normal (3 samples)" in msg
    assert "logit-transformed proportions, ~ condition, moderated t-test" in msg
    res = xe._comp_state["result"]
    counts = pd.DataFrame({f"S{i + 1}": e["_cell_type_labels"].value_counts() for i, e in enumerate(extras)})
    counts = counts.fillna(0).astype(int).loc[res["tests"]["category"]]
    ref = xe._comp_test(counts.to_numpy(), ["A"] * 3 + ["B"] * 3, None, "logit")
    np.testing.assert_allclose(res["tests"]["p_mod"], ref["p_mod"])
    np.testing.assert_allclose(res["props"].loc["Alpha", "S2"],
                               (extras[1]["_cell_type_labels"] == "Alpha").mean())
    np.testing.assert_allclose(res["props"].sum(axis=0), 1.0)
    alpha = res["tests"].set_index("category").loc["Alpha"]
    assert alpha["fdr_mod"] < 0.05 and alpha["t_mod"] > 0 and alpha["prop_ratio"] > 1.5
    # the primary sample has labels too but no condition: shown, not tested
    assert "S0" in res["props"].columns and res["n_a"] == res["n_b"] == 3
    fig, cols, rows = xe.update_comp_results(3, 0, None, None)
    assert [t.name for t in fig.data if t.name] == ["A: tumour", "B: normal", "not assigned"]
    assert len(fig.data[0].x) == 3 * 4 and any(a.text == "*" for a in fig.layout.annotations)
    ids = [c["id"] for c in cols]
    assert ids[:4] == ["category", "mean_prop_a", "mean_prop_b", "prop_ratio"] and "prop_S6" in ids
    assert {r["category"] for r in rows} == set(types) and all(set(r) == set(ids) for r in rows)
    out = xe.download_composition(1)
    assert out["filename"] == "composition_cell_type_CellTypist_.csv"
    head = [l for l in out["content"].splitlines() if l.startswith("#")]
    assert sum(l.startswith("# sample S") for l in head) == 7 and any("compositional" in l for l in head)
    body = pd.read_csv(io.StringIO(out["content"]), comment="#").set_index("category")
    np.testing.assert_allclose(body.loc[res["tests"]["category"], "p_welch"], res["tests"]["p_welch"])
    np.testing.assert_array_equal(body.loc["Beta", "n_S4"], counts.loc["Beta", "S4"])
    # arcsine transform is another test of the same proportions
    xe.run_composition(4, "celltype:celltypist", "asin", "xenium", None, 3)
    assert "arcsine-square-root-transformed" in xe._comp_state["message"]
    # a batch label on some samples only: no test, with the reason
    xe._design_set([{"key": xe._ds_key(extras[0]), "cond": "A", "batch": "p1"}])
    assert "some samples only" in xe.run_composition(5, "celltype:celltypist", "logit", "xenium", None, 4)[1]
    # clusters: samples without the cluster set are left out and named
    msg = xe.run_composition(6, "cluster", "logit", "xenium", (D.get("cluster_methods") or ["x"])[0], 5)[1]
    assert "No test was run" in msg and ("S1 has no cluster labels" in msg or not D.get("cluster_methods"))


# ── transcript density with several samples ──────────────────────────────────
def test_density_layer_covers_added_samples_each_at_its_offset(xe, raw, D, client, tmp_path):
    gene = D["gene_names"][int(np.asarray(D["expr"].sum(axis=0)).ravel().argmax())]
    other = tmp_path / "second_sample"
    other.mkdir()
    for name in ("transcripts.parquet", "transcripts.zarr.zip"):
        src = os.path.join(D["data_dir"], name)
        if os.path.exists(src):
            os.symlink(src, other / name)
    off = 12_345.5                                   # not a multiple of any bin size
    extra = {**{k: D[k] for k in ("expr", "df_to_expr", "gene_name_to_idx", "gene_names")},
             "df": D["df"], "x_offset": off, "data_dir": str(other)}
    zarr_only = {**extra, "data_dir": None, "sdata_path": str(tmp_path / "x.zarr"), "x_offset": 2 * off}
    assert xe.run_density(1, gene) == (True, False, "Binning transcripts…")
    single_msg = tl.wait_state(xe._density_state, xe._density_lock)
    single = xe._density_window(gene, 10)
    h1, a1 = tl.get_bin(client, f"/viewer/extra/density.bin?gene={gene}&bin=10&x0=-1e9&y0=-1e9&x1=1e9&y1=1e9")
    assert len(h1["parts"]) == 1 and h1["w"] == single["values"].shape[1] and "(S" not in single_msg
    xe.EXTRA_DATASETS.extend([extra, zarr_only])
    st = xe._density_status(gene)
    assert st == {"ready": ["S0"], "missing": ["S1"], "unavailable": ["S2"]}
    store, hint = xe.update_density_store(["on"], gene, 10, 0.7, None, 1, 0, 1)
    assert store["on"] and "No grid yet for S1" in hint and "S2 loaded from a .zarr" in hint
    assert store["version"] == "0|1|1"
    # Compute bins the sample that has none, into that sample's own cache directory
    xe.run_density(2, gene)
    msg = tl.wait_state(xe._density_state, xe._density_lock)
    assert f"{gene} (S1): {single['total']:,} transcripts" in msg and "(S0)" not in msg
    assert "No transcripts for S2 (loaded from a .zarr)" in msg
    f0, f1 = xe._density_cache_file(gene), xe._density_cache_file(gene, extra)
    assert f0 != f1 and os.path.exists(f1) and os.path.dirname(f1).startswith(str(xe.CACHE_BASE_DIR))
    assert xe._density_status(gene)["ready"] == ["S0", "S1"]
    for b in (5, 20):
        wins = xe._density_windows(gene, b)
        one = xe._density_window(gene, b)
        assert wins["labels"] == ["S0", "S1"] and wins["unavailable"] == ["S2"] and wins["total"] == 2 * one["total"]
        p0, p1 = wins["parts"]
        np.testing.assert_array_equal(p0["values"], one["values"])
        np.testing.assert_array_equal(p1["values"], one["values"])          # the same transcripts
        assert p0["x0"] == one["x0"] and p1["x0"] == one["x0"] + off and p1["y0"] == one["y0"]
        nz = xe._density_grid(gene, b)["grid"]
        nz = nz[nz > 0]
        assert wins["vmax"] == max(1.0, np.percentile(np.concatenate([nz, nz]), 99.5))   # one scale, all samples
    # a box over the second sample only (view coordinates): the first part is empty
    x = D["df"]["x_centroid"].to_numpy(float)
    box = (x.min() + off - 20, -1e9, x.max() + off + 20, 1e9)
    wins = xe._density_windows(gene, 10, *box)
    assert wins["parts"][0]["values"].size == 0 and wins["parts"][1]["values"].sum() > 0
    np.testing.assert_allclose(wins["parts"][1]["values"].sum(),
                               xe._density_window(gene, 10, box[0] - off, -1e9, box[2] - off, 1e9)["values"].sum())
    # every part is coarsened alike
    full = xe._density_windows(gene, 5, max_cells=10 ** 12)
    coarse = xe._density_windows(gene, 5, max_cells=max(8, sum(p["values"].size for p in full["parts"]) // 6))
    assert coarse["coarsened"] and len({p["cell"] for p in coarse["parts"]}) == 1 and coarse["parts"][0]["cell"] > 5
    # WebGL: one response, one entry per sample
    h, a = tl.get_bin(client, f"/viewer/extra/density.bin?gene={gene}&bin=10&x0=-1e9&y0=-1e9&x1=1e9&y1=1e9")
    assert [p["label"] for p in h["parts"]] == ["S0", "S1"] and h["ready"] and h["total"] == 2 * single["total"]
    n0 = h["parts"][0]["w"] * h["parts"][0]["h"]
    assert h["parts"][1]["offset"] == n0 and len(a["values"]) == 2 * n0
    np.testing.assert_array_equal(a["values"][:n0], a1["values"])
    np.testing.assert_array_equal(a["values"][n0:], a1["values"])
    np.testing.assert_allclose(h["parts"][1]["x0"] - h["parts"][0]["x0"], off)
    # Plotly: one heat trace per sample, one colour bar
    spec = {"on": True, "gene": gene, "bin": 10, "opacity": 0.7, "max": None}
    method = (D.get("cluster_methods") or [None])[0]
    fig = xe.make_spatial_fig("transcript_counts", method, None, 3, 0.8, [], None, density=spec)
    heat = [t for t in fig.data if t.type == "heatmap"]
    assert len(heat) == 2 and [bool(t.showscale) for t in heat] == [True, False]
    assert heat[0].zmax == heat[1].zmax and np.isclose(heat[1].x0 - heat[0].x0, off)
    np.testing.assert_array_equal(np.nan_to_num(heat[0].z), np.nan_to_num(heat[1].z))
    assert [t.type for t in fig.data[:2]] == ["heatmap", "heatmap"]         # below the cells
    hint = xe.update_density_store(["on"], gene, 10, 0.7, None, 2, 0, 1)[1]
    assert f"{2 * single['total']:,} transcripts" in hint and "Samples drawn: S0, S1" in hint


# ── SPLIT-corrected counts with several samples ──────────────────────────────
def test_corrected_counts_with_several_samples_are_all_or_nothing(xe, D, client, expressed_gene):
    gene = expressed_gene
    j = D["gene_name_to_idx"][gene]
    corr = sp.csr_matrix(sp.csr_matrix(D["expr"]) * 2.0)            # "corrected": twice the counts
    D["split_corrected_expr"] = corr
    extra = {**{k: D[k] for k in ("expr", "df_to_expr", "gene_name_to_idx", "gene_names")},
             "df": D["df"], "x_offset": 1e5}
    xe.EXTRA_DATASETS.append(extra)
    n = len(D["df"])
    orig = np.log1p(np.asarray(sp.csr_matrix(D["expr"])[:, j].todense()).ravel()[np.asarray(D["df_to_expr"])])
    twice = np.log1p(2 * np.expm1(orig))
    method = (D.get("cluster_methods") or [None])[0]

    def both(use_corrected):
        fig = xe.make_spatial_fig("gene", method, gene, 3, 0.8, [], None, use_corrected=use_corrected)
        trace = [t for t in fig.data if t.type == "scattergl"][0]
        h, a = tl.get_bin(client, f"/viewer/colors.bin?mode=gene&gene={gene}&corrected={int(use_corrected)}")
        np.testing.assert_allclose(np.asarray(trace.marker.color, dtype=float), a["values"], rtol=1e-6)
        return np.asarray(trace.marker.color, dtype=float), trace.marker.colorbar.title.text, h

    # one sample lacks corrected counts: the original counts for all, and the UI says so
    st = xe._corrected_status()
    assert st["have"] == [True, False] and st["missing"] == ["S1"] and not st["all"] and st["any"]
    vals, title, _ = both(True)
    np.testing.assert_allclose(vals, np.concatenate([orig, orig]), rtol=1e-6)
    assert "SPLIT-corrected" not in title
    assert [s["corrected"] for s in xe._an_samples("xenium", True)] == [False, False]
    note = xe.update_counts_mode_note("corrected", 1, 0, 0)
    assert "exist for S0 but not for S1" in note and "never mixed" in note
    options, value = xe.update_counts_options(0, "xenium", 0, 0, 0, 0, 1, "corrected")
    assert options[1]["disabled"] is True and value == "original"
    # every sample has them: corrected for all, in both renderers and in the tests
    extra["split_corrected_expr"] = corr
    assert xe._corrected_status()["all"] and "All 2 samples" in xe.update_counts_mode_note("corrected", 2, 0, 0)
    vals, title, h = both(True)
    np.testing.assert_allclose(vals, np.concatenate([twice, twice]), rtol=1e-6)
    assert "SPLIT-corrected" in title
    np.testing.assert_allclose(both(False)[0], np.concatenate([orig, orig]), rtol=1e-6)
    ss = xe._an_samples("xenium", True)
    assert [s["corrected"] for s in ss] == [True, True] and ss[1]["csc"].sum() == corr.sum()
    assert [s["corrected"] for s in xe._an_samples("xenium", False)] == [False, False]
    options, value = xe.update_counts_options(0, "xenium", 0, 0, 0, 0, 2, "corrected")
    assert options[1]["disabled"] is False
    # a corrected matrix with another number of rows is not usable
    extra["split_corrected_expr"] = corr[: n - 1]
    assert xe._corrected_status()["missing"] == ["S1"]
    # a single sample is as before
    del xe.EXTRA_DATASETS[:]
    assert xe.update_counts_mode_note("corrected", 3, 0, 0) == ""
    assert xe._an_samples("xenium", True)[0]["corrected"] is True


def test_added_sample_gets_its_cached_annotation_and_stored_cell_types(xe, D, cache_dir):
    import hashlib
    key = "/data/some/other_sample"
    tag = hashlib.md5(key.encode()).hexdigest()[:8]
    ids = [f"c{i}" for i in range(30)]
    pd.DataFrame({"label": ["T"] * 10 + ["B"] * 20}, index=ids).to_parquet(
        os.path.join(xe.CACHE_BASE_DIR, f"Immune_All_Low_labels_celltypist_{tag}.parquet"))
    pd.DataFrame({"label": ["x"] * 30}, index=ids).to_parquet(
        os.path.join(xe.CACHE_BASE_DIR, f"Immune_All_Low_labels_celltypist_baysor_abc12345_{tag}.parquet"))
    try:
        by = xe._annot_cached_labels(key)
        assert list(by) == ["celltypist"] and by["celltypist"].value_counts().to_dict() == {"B": 20, "T": 10}
        assert xe._annot_cached_labels("/data/never/annotated") == {}
        df = pd.DataFrame({"x_centroid": np.arange(30.0), "y_centroid": 0.0,
                           "cell_type_seurat": ["a"] * 15 + [""] * 15}, index=ids)
        eds = {"df": df, "_cell_type_labels_by_method": by}
        lab, src = xe._extra_cell_type_labels(eds, "celltypist")
        assert src == "celltypist (annotation cache)" and lab is by["celltypist"]
        lab, src = xe._extra_cell_type_labels(eds, "seurat")
        assert src.startswith("cell_type_seurat") and int(lab.notna().sum()) == 15
        assert xe._extra_cell_type_labels(eds, "rctd") == (None, None)       # never another method's labels
    finally:
        for f in os.listdir(xe.CACHE_BASE_DIR):
            if f.endswith(f"_{tag}.parquet"):
                os.remove(os.path.join(xe.CACHE_BASE_DIR, f))


# ── Moran's I ────────────────────────────────────────────────────────────────
def _hand_sample(counts, xy, genes=None, label="hand"):
    """An analysis sample dict (as _an_samples returns) around a dense count matrix."""
    n, g = counts.shape
    genes = genes or [f"g{i}" for i in range(g)]
    df = pd.DataFrame({"x_centroid": xy[:, 0], "y_centroid": xy[:, 1]}, index=[f"c{i}" for i in range(n)])
    return {"label": label, "name": label, "key": label, "df": df, "csc": sp.csc_matrix(counts.astype(np.float32)),
            "rows": np.arange(n), "gni": {x: i for i, x in enumerate(genes)}, "genes": genes,
            "x_offset": 0.0, "corrected": False, "ds": {}}


def _moran_dense(x, W):
    """Moran's I, its z-score and two-sided p under normality, from the textbook formulas."""
    n = len(x)
    z = x - x.mean()
    s0 = W.sum()
    I = n / s0 * (z @ W @ z) / (z @ z)
    s1 = 0.5 * ((W + W.T) ** 2).sum()
    s2 = ((W.sum(axis=1) + W.sum(axis=0)) ** 2).sum()
    ei = -1 / (n - 1)
    var = (n * n * s1 - n * s2 + 3 * s0 * s0) / (s0 * s0 * (n * n - 1)) - ei * ei
    zz = (I - ei) / np.sqrt(var)
    return I, zz, 2 * sstats.norm.sf(abs(zz))


def _brute_knn_w(xy, k):
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    W = np.zeros_like(d)
    idx = np.argsort(d, axis=1, kind="stable")[:, :k]
    W[np.arange(len(xy))[:, None], idx] = 1.0 / k
    return W


def _spatial_counts(seed=0, n=260, g=24):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 300, size=(n, 2))
    lam = np.full((n, g), 1.5)
    lam[:, :6] = 0.3 + 6.0 * (xy[:, [0]] < 110)            # six genes with a spatial domain
    lam[:, 6] = 0.0                                        # never detected
    return rng.poisson(lam), xy


def test_morans_i_equals_the_dense_formulas(xe):
    counts, xy = _spatial_counts()
    s = _hand_sample(counts, xy)
    W = _brute_knn_w(xy, 6)
    np.testing.assert_allclose(xe._knn_weights(xy, 6).toarray(), W)
    r = xe._moran_sample(s, 6)
    t = r["table"]
    lib = counts.sum(axis=1)
    X = np.log1p(counts / lib[:, None] * 1e4)
    for j in list(range(6)) + [7, 15, 23]:
        I, z, p = _moran_dense(X[:, j], W)
        np.testing.assert_allclose([t["moran_i"][j], t["z"][j]], [I, z], rtol=2e-6)   # (float32 size factors)
        np.testing.assert_allclose(t["pval"][j], p, rtol=1e-4, atol=1e-300)
    assert np.isnan(t["moran_i"][6]) and np.isnan(t["padj"][6]) and t["pct_cells"][6] == 0
    assert r["n_cells"] == len(xy) and r["k"] == 6 and np.isclose(r["expected"], -1 / (len(xy) - 1))
    assert (t["moran_i"][:6] > 0.3).all() and (t["padj"][:6] < 1e-6).all()
    assert (t["moran_i"][7:].abs() < 0.15).all() and (t["padj"][7:] < 0.05).mean() < 0.2
    ok = t["moran_i"].notna()
    np.testing.assert_allclose(t["padj"][ok], xe._bh(t["pval"][ok].to_numpy()), rtol=1e-9)
    np.testing.assert_allclose(t["pct_cells"], 100 * (counts > 0).mean(axis=0))
    # the block size only bounds memory
    small = xe._moran_sample(s, 6, chunk_bytes=8 * len(xy) * 5)["table"]
    np.testing.assert_allclose(small["moran_i"], t["moran_i"], rtol=1e-12, equal_nan=True)
    # cells without a matrix row (a subset, a reseg run) are not in the graph
    s2 = dict(s, rows=np.where(np.arange(len(xy)) % 5 == 0, -1, np.arange(len(xy))))
    keep = s2["rows"] >= 0
    r2 = xe._moran_sample(s2, 4)["table"]
    lib2 = counts[keep].sum(axis=1)
    I, z, _ = _moran_dense(np.log1p(counts[keep, 2] / lib2 * 1e4), _brute_knn_w(xy[keep], 4))
    np.testing.assert_allclose([r2["moran_i"][2], r2["z"][2]], [I, z], rtol=2e-6)
    # coincident points are never their own neighbour
    dup = np.repeat(xy[:20], 2, axis=0)
    idx = xe._knn_indices(dup, 3)
    assert idx.shape == (40, 3) and not (idx == np.arange(40)[:, None]).any()
    with pytest.raises(ValueError, match="fewer than 3 cells"):
        xe._moran_sample(dict(s, rows=np.full(len(xy), -1)), 6)


def test_morans_i_equals_spdep(xe, tmp_path):
    _need_r("spdep")
    counts, xy = _spatial_counts(seed=5, n=180, g=12)
    t = xe._moran_sample(_hand_sample(counts, xy), 6)["table"]
    X = np.log1p(counts / counts.sum(axis=1)[:, None] * 1e4)
    pd.DataFrame(X).to_csv(tmp_path / "x.csv", index=False)
    pd.DataFrame(xy).to_csv(tmp_path / "xy.csv", index=False)
    _rscript(r"""
suppressPackageStartupMessages(library(spdep))
x <- as.matrix(read.csv("%(d)s/x.csv")); xy <- as.matrix(read.csv("%(d)s/xy.csv"))
lw <- nb2listw(knn2nb(knearneigh(xy, k = 6)), style = "W")
out <- t(sapply(seq_len(ncol(x)), function(j) {
  if (var(x[, j]) == 0) return(c(NA, NA, NA))
  m <- moran.test(x[, j], lw, randomisation = FALSE, alternative = "two.sided")
  c(m$estimate[1], m$statistic, m$p.value) }))
write.csv(out, "%(d)s/spdep.csv", row.names = FALSE)
""" % {"d": tmp_path}, tmp_path)
    ref = pd.read_csv(tmp_path / "spdep.csv").to_numpy()
    np.testing.assert_allclose(t["moran_i"], ref[:, 0], rtol=2e-6, equal_nan=True)
    np.testing.assert_allclose(t["z"], ref[:, 1], rtol=2e-6, equal_nan=True)
    np.testing.assert_allclose(t["pval"], ref[:, 2], rtol=1e-4, atol=1e-300, equal_nan=True)


def test_morans_i_tab_runs_per_sample_as_a_job(xe, D, expressed_gene):
    counts, xy = _spatial_counts(seed=2, n=150, g=N_GENES)
    extra = _fab_sample(D, 150, 0)
    extra["expr"] = sp.hstack([sp.csr_matrix(counts.astype(np.float32)),
                               sp.csr_matrix((150, len(D["gene_names"]) - N_GENES), dtype=np.float32)]).tocsr()
    extra["df"]["x_centroid"], extra["df"]["y_centroid"] = xy[:, 0], xy[:, 1]
    xe.EXTRA_DATASETS.append(extra)
    assert xe.run_moran(1, 6, "xenium", "original") == (True, False, "Running…")
    msg = tl.wait_state(xe._moran_state, xe._moran_lock)
    job = [j for j in xe._jobs.values() if j.kind == "moran"][-1]
    job.thread.join(10)
    assert job.state == "done" and xe.poll_moran(1, 4) == (msg, True, False, 5)
    res = xe._moran_state["result"]
    n0 = len(D["df"])
    assert [i["label"] for i in res["samples"]] == ["S0", "S1"] and res["samples"][1]["n_cells"] == 150
    assert f"S0: {n0:,} cells" in msg and "S1: 150 cells" in msg and "original counts" in msg and "k = 6" in msg
    t = res["table"]
    assert set(t["sample"]) == {"S0", "S1"} and len(t) == 2 * len(xe._an_common_genes(xe._an_samples()))
    # each sample on its own graph: S1 is the hand-made sample, S0 the bundle's cells
    s1 = t[t["sample"] == "S1"].set_index("gene")
    ref = xe._moran_sample(_hand_sample(counts, xy, genes=list(D["gene_names"][:N_GENES])), 6)["table"]
    np.testing.assert_allclose(s1.loc[ref["gene"], "moran_i"], ref["moran_i"], rtol=1e-9, equal_nan=True)
    s0 = xe._an_samples()[0]
    X = sp.csr_matrix(s0["csc"])[s0["rows"]].toarray().astype(float)
    lib0 = X[:, [s0["gni"][g] for g in s0["genes"]]].sum(axis=1, keepdims=True)
    Xn = np.log1p(np.where(lib0 > 0, X / np.maximum(lib0, 1), 0.0) * 1e4)       # no counts: no expression
    I, z, p = _moran_dense(Xn[:, s0["gni"][expressed_gene]], _brute_knn_w(xe._xy_of(s0["df"]), min(6, n0 - 1)))
    got = t[(t["sample"] == "S0") & (t["gene"] == expressed_gene)].iloc[0]
    np.testing.assert_allclose([got["moran_i"], got["z"], got["pval"]], [I, z, p], rtol=1e-5)
    rows = xe.update_moran_results(1, 0)
    assert rows[0]["moran_i"] == max(r["moran_i"] for r in rows) and set(rows[0]) == {
        "sample", "gene", "moran_i", "z", "pval", "padj", "pct_cells", "mean"}
    assert len(rows) == int(t["moran_i"].notna().sum())
    opts = [{"label": g, "value": g} for g in D["gene_names"]]
    assert xe.moran_pick_gene_from_table({"row": 0, "column": 1}, rows[:15], opts) == rows[0]["gene"]
    with pytest.raises(Exception):
        xe.moran_pick_gene_from_table({"row": 40, "column": 1}, rows[:15], opts)
    out = xe.download_moran_table(1)
    head = [l for l in out["content"].splitlines() if l.startswith("#")]
    assert "k = 6" in head[0] and "row-standardised" in head[0] and len(head) == 3
    body = pd.read_csv(io.StringIO(out["content"]), comment="#")
    assert len(body) == len(t) and list(body.columns[:3]) == ["sample", "gene", "moran_i"]
    # corrected counts are another matrix: asked for, absent in S1 → original, and said
    D["split_corrected_expr"] = sp.csr_matrix(D["expr"])
    xe.run_moran(2, 3, "xenium", "corrected")
    assert "Original counts were used: not every sample has SPLIT-corrected counts" in \
        tl.wait_state(xe._moran_state, xe._moran_lock)
    assert xe._moran_state["result"]["k"] == 3 and not xe._moran_state["result"]["corrected"]


def test_morans_i_job_can_be_cancelled(xe, D, monkeypatch):
    import threading
    entered, release = threading.Event(), threading.Event()
    real = xe._knn_weights

    def slow(xy, k=6):
        entered.set()
        release.wait(30)
        return real(xy, k)

    monkeypatch.setattr(xe, "_knn_weights", slow)
    xe.run_moran(1, 6, "xenium", "original")
    assert entered.wait(30)
    job = [j for j in xe._jobs.values() if j.kind == "moran"][-1]
    assert xe._job_cancel(job.id)
    release.set()
    job.thread.join(30)
    assert job.state == "cancelled" and xe._moran_state["status"] == "error"
    assert xe._moran_state["message"] == "Cancelled" and xe._moran_state["result"] is None


# ── ligand-receptor scores ───────────────────────────────────────────────────
def _lr_world(seed=0, per=150):
    """Three cell types in vertical stripes A | B | C (A touches B, B touches C, A and C
    are 100 µm apart). L1 is made by A and R1 by B: the planted interaction. L2 / R2 are
    expressed alike everywhere. R3 is made by C only."""
    rng = np.random.default_rng(seed)
    types = np.repeat(["A", "B", "C"], per)
    x = np.concatenate([rng.uniform(0, 100, per), rng.uniform(100, 200, per), rng.uniform(200, 300, per)])
    xy = np.column_stack([x, rng.uniform(0, 100, 3 * per)])
    genes = ["L1", "R1", "L2", "R2", "R3"] + [f"bg{i}" for i in range(25)]
    lam = np.full((3 * per, len(genes)), 2.0)
    lam[:, 0] = np.where(types == "A", 8.0, 0.4)
    lam[:, 1] = np.where(types == "B", 8.0, 0.4)
    lam[:, 4] = np.where(types == "C", 8.0, 0.4)
    return rng.poisson(lam), xy, types, genes


_PAIRS = [("L1", "R1"), ("L2", "R2"), ("L1", "R3"), ("L1", "NOT_ON_PANEL"), ("l2", "r2")]


def test_ligand_receptor_recovers_the_planted_interaction_only(xe):
    counts, xy, types, genes = _lr_world()
    s = _hand_sample(counts, xy, genes)
    r = xe._lr_compute([s], [types.astype(object)], _PAIRS, k=6, n_perm=300, seed=0)
    info, t = r["info"], r["table"]
    assert info["n_pairs_input"] == 5 and info["n_pairs_on_panel"] == 3 and info["n_pairs_missing"] == 2
    assert info["types"] == ["A", "B", "C"] and info["n_cells"] == 450
    # A and C do not touch: that type pair is not scored at all
    pairs_of_types = set(zip(t["sender"], t["receiver"]))
    assert ("A", "C") not in pairs_of_types and ("C", "A") not in pairs_of_types
    assert {("A", "B"), ("B", "A"), ("B", "C"), ("C", "B"), ("A", "A")} <= pairs_of_types
    assert info["n_type_pairs"] == 9 and info["n_type_pairs_neighbouring"] == 7
    row = lambda sd, rc, lg, rp: t[(t["sender"] == sd) & (t["receiver"] == rc) & (t["ligand"] == lg)
                                   & (t["receptor"] == rp)]
    hit = row("A", "B", "L1", "R1").iloc[0]
    assert hit["pval"] == 1 / 301 and hit["padj"] < 0.05
    # the score is the mean of the two type means of log1p(CP10K), written out
    X = np.log1p(counts / counts.sum(axis=1)[:, None] * 1e4)
    mL, mR = X[types == "A", 0].mean(), X[types == "B", 1].mean()
    np.testing.assert_allclose([hit["mean_ligand"], hit["mean_receptor"], hit["score"]],
                               [mL, mR, (mL + mR) / 2], rtol=1e-6)
    np.testing.assert_allclose(hit["frac_ligand"], (counts[types == "A", 0] > 0).mean())
    # adjacent cell pairs between A and B in the symmetrised kNN graph, counted by hand
    W = _brute_knn_w(xy, 6) > 0
    U = W | W.T
    assert hit["n_edges"] == int(U[np.ix_(types == "A", types == "B")].sum())
    assert row("A", "A", "L1", "R1").iloc[0]["n_edges"] == int(U[np.ix_(types == "A", types == "A")].sum()) // 2
    # the reverse direction and the unspecific pair are not significant anywhere
    assert row("B", "A", "L1", "R1").iloc[0]["pval"] > 0.5
    flat = t[(t["ligand"] == "L2")]
    assert len(flat) == 7 and (flat["padj"] > 0.05).all() and flat["pval"].min() > 0.01
    # L1 → R3 would be "significant" for A → C by expression alone; it is never tested there
    assert len(row("A", "C", "L1", "R3")) == 0
    np.testing.assert_allclose(t["padj"], xe._bh(t["pval"].to_numpy()))
    assert info["n_tested"] == len(t)
    # the permutation p-value against a plain re-implementation for one combination
    rng = np.random.default_rng(0)
    obs, ge = (mL + mR) / 2, 0
    for _ in range(300):
        perm = rng.permutation(types)
        ge += (X[perm == "A", 0].mean() + X[perm == "B", 1].mean()) / 2 >= obs - 1e-12
    assert hit["pval"] == (ge + 1) / 301
    # deterministic for a seed; another seed permutes differently
    again = xe._lr_compute([s], [types.astype(object)], _PAIRS, k=6, n_perm=300, seed=0)["table"]
    pd.testing.assert_frame_equal(again, t)
    other = xe._lr_compute([s], [types.astype(object)], _PAIRS, k=6, n_perm=300, seed=1)["table"]
    assert not np.array_equal(other["pval"].to_numpy(), t["pval"].to_numpy())
    assert other[(other["sender"] == "A") & (other["ligand"] == "L1") & (other["receptor"] == "R1")
                 & (other["receiver"] == "B")]["padj"].iloc[0] < 0.05


def test_ligand_receptor_pools_samples_and_leaves_unlabelled_cells_out(xe):
    c1, xy1, t1, genes = _lr_world(seed=1, per=120)
    c2, xy2, t2, _ = _lr_world(seed=2, per=90)
    lab1 = t1.astype(object)
    lab1[::7] = None                                         # unlabelled cells
    s1, s2 = _hand_sample(c1, xy1, genes, "S0"), _hand_sample(c2, xy2, genes, "S1")
    r = xe._lr_compute([s1, s2], [lab1, t2.astype(object)], [("L1", "R1"), ("L2", "R2")], n_perm=100, seed=3)
    n1 = int(sum(x is not None for x in lab1))
    assert r["info"]["n_cells"] == n1 + 270
    t = r["table"]
    hit = t[(t["sender"] == "A") & (t["receiver"] == "B") & (t["ligand"] == "L1")].iloc[0]
    ok = np.array([x is not None for x in lab1])
    X = np.vstack([np.log1p(c1[ok] / c1[ok].sum(axis=1)[:, None] * 1e4), np.log1p(c2 / c2.sum(axis=1)[:, None] * 1e4)])
    ty = np.concatenate([t1[ok], t2])
    np.testing.assert_allclose(hit["mean_ligand"], X[ty == "A", 0].mean(), rtol=1e-6)
    assert hit["padj"] < 0.05 and (t[t["ligand"] == "L2"]["padj"] > 0.05).all()
    # no edges across samples: counted within each
    e = sum(int(((_brute_knn_w(xy, 6) > 0) | (_brute_knn_w(xy, 6) > 0).T)[np.ix_(tt == "A", tt == "B")].sum())
            for xy, tt in ((xy1[ok], t1[ok]), (xy2, t2)))
    assert hit["n_edges"] == e
    # a sample without labels is skipped; nothing labelled at all is an error
    r1 = xe._lr_compute([s1, s2], [None, t2.astype(object)], [("L1", "R1")], n_perm=20)
    assert r1["info"]["n_cells"] == 270
    with pytest.raises(ValueError, match="no sample|No sample"):
        xe._lr_compute([s1], [None], [("L1", "R1")], n_perm=20)
    with pytest.raises(ValueError, match="None of the 1 ligand–receptor pairs has both genes"):
        xe._lr_compute([s1], [lab1], [("X", "Y")], n_perm=20)
    with pytest.raises(ValueError, match="No cell type has at least 10 cells"):
        xe._lr_compute([s1], [np.array([f"t{i}" for i in range(len(t1))], dtype=object)], [("L1", "R1")], n_perm=20)


def test_ligand_receptor_pair_lists(xe):
    pairs, skipped = xe._lr_parse_pairs("ligand,receptor\nTGFB1,TGFBR1\nTGFB1,TGFBR1\n\n# note\nCXCL12\tCXCR4\nIL6 IL6R\nbad\n")
    assert pairs == [("TGFB1", "TGFBR1"), ("CXCL12", "CXCR4"), ("IL6", "IL6R")] and skipped == 2
    wide = "id;pathway;Ligand_symbol;Receptor_symbol\n1;TGFb;TGFB1;TGFBR2\n2;CXCL;\"CXCL12\";'ACKR3'\n"
    assert xe._lr_parse_pairs(wide) == ([("TGFB1", "TGFBR2"), ("CXCL12", "ACKR3")], 0)
    assert xe._lr_parse_pairs("") == ([], 0) and xe._lr_parse_pairs(None) == ([], 0)
    # upload → text box; nothing is bundled, CellNEST pairs only when results are loaded
    import base64
    payload = "data:text/csv;base64," + base64.b64encode(b"\xef\xbb\xbfligand,receptor\nA,B\nC,D\n").decode()
    text, info = xe.fill_lr_pairs_from_upload(payload, "my.csv")
    assert text == "ligand,receptor\nA,B\nC,D" and info == "2 pairs from my.csv."
    assert "no 'ligand,receptor' lines" in xe.fill_lr_pairs_from_upload(
        "data:text/csv;base64," + base64.b64encode(b"\n\n").decode(), "e.csv")[1]
    assert xe._lr_pairs_from_cellnest() == []
    assert xe.fill_lr_pairs_from_cellnest(1)[1].startswith("No CellNEST results are loaded")
    with xe._cellnest_lock:
        xe._cellnest_state["data"] = {"ccc_df": pd.DataFrame({"ligand": ["A", "A", "C"], "receptor": ["B", "B", "D"]})}
    try:
        assert xe._lr_pairs_from_cellnest() == [("A", "B"), ("C", "D")]
        assert xe.fill_lr_pairs_from_cellnest(1)[0] == "ligand,receptor\nA,B\nC,D"
    finally:
        with xe._cellnest_lock:
            xe._cellnest_state["data"] = None
    src = (tl.APP_PACKAGE / "stats.py").read_text() + (tl.APP_PACKAGE / "cb_stats.py").read_text()
    assert "urlopen" not in src and "urlretrieve" not in src and "requests" not in src and "http" not in src


def test_ligand_receptor_tab_runs_as_a_job(xe, D, monkeypatch):
    c, xy, types, genes = _lr_world(seed=4, per=80)
    names = list(D["gene_names"])
    extra = _fab_sample(D, len(types), 0)
    m = sp.lil_matrix((len(types), len(names)), dtype=np.float32)
    m[:, :len(genes)] = c
    extra["expr"] = m.tocsr()
    extra["df"]["x_centroid"], extra["df"]["y_centroid"] = xy[:, 0], xy[:, 1]
    extra["_cell_type_labels"] = pd.Series(types, index=extra["df"].index)
    xe.EXTRA_DATASETS.append(extra)
    L1, R1, L2, R2 = names[0], names[1], names[2], names[3]
    text = f"ligand,receptor\n{L1},{R1}\n{L2},{R2}\n{L1},NOPE\n"
    opts, value = xe.update_lr_label_options("tab-lr", 0, 0, "xenium", 0, "cell_type:celltypist", None)
    assert value == "celltype:celltypist" and opts[-1]["value"] == "cluster"
    assert xe.run_lr(1, "", "celltype:celltypist", 6, 50, "xenium", None, "original")[2].startswith("No ligand")
    assert xe.run_lr(1, text, None, 6, 50, "xenium", None, "original")[2].startswith("Pick the cell types")
    assert xe.run_lr(1, text, "celltype:celltypist", 6, 100, "xenium", None, "original") == (True, False, "Running…")
    msg = tl.wait_state(xe._lr_state, xe._lr_lock)
    job = [j for j in xe._jobs.values() if j.kind == "lr"][-1]
    job.thread.join(10)
    assert job.state == "done" and xe.poll_lr(1, 0) == (msg, True, False, 1)
    assert "2 of 3 ligand–receptor pairs have both genes on the panel (1 not scored)" in msg
    assert "100 permutations, seed 0" in msg and "original counts" in msg and "were pooled" in msg
    res = xe._lr_state["result"]
    t = res["table"]
    hit = t[(t["sender"] == "A") & (t["receiver"] == "B") & (t["ligand"] == L1)].iloc[0]
    assert hit["padj"] < 0.05 and (t[t["ligand"] == L2]["padj"] > 0.05).all()
    fig, rows = xe.update_lr_results(1, 0, 25, None, None)
    assert fig.data[0].type == "scatter" and "A → B" in list(fig.data[0].x) and f"{L1} → {R1}" in list(fig.data[0].y)
    assert rows[0]["sender"] == "A" and rows[0]["ligand"] == L1 and len(rows) == len(t)
    out = xe.download_lr_table(1)
    head = [l for l in out["content"].splitlines() if l.startswith("#")]
    assert "2 of 3 pairs have both genes on the panel" in head[1] and "not evidence of signalling" in head[2]
    assert len(pd.read_csv(io.StringIO(out["content"]), comment="#")) == len(t)
    # no labels of the chosen kind: an error for the user
    del extra["_cell_type_labels"]
    xe.run_lr(2, text, "celltype:rctd", 6, 50, "xenium", None, "original")
    with pytest.raises(AssertionError, match="no cell type \\(RCTD\\) labels"):
        tl.wait_state(xe._lr_state, xe._lr_lock)
    xe._test_baseline.settle()
    # cancel during the permutations
    import threading
    entered, release = threading.Event(), threading.Event()
    real = xe._knn_indices

    def slow(xy_, k):
        entered.set()
        release.wait(30)
        return real(xy_, k)

    monkeypatch.setattr(xe, "_knn_indices", slow)
    extra["_cell_type_labels"] = pd.Series(types, index=extra["df"].index)
    xe.run_lr(3, text, "celltype:celltypist", 6, 100, "xenium", None, "original")
    assert entered.wait(30)
    job = [j for j in xe._jobs.values() if j.kind == "lr"][-1]
    assert xe._job_cancel(job.id)
    release.set()
    job.thread.join(30)
    assert job.state == "cancelled" and xe._lr_state["message"] == "Cancelled"


def test_new_tabs_are_in_the_layout_with_their_caveats(xe):
    import json
    import plotly
    text = json.dumps(xe.app.layout, cls=plotly.utils.PlotlyJSONEncoder)
    for needle in ("tab-moran", "tab-lr", "comp-section", "design-modal", "de-mode", "de-pb-population",
                   "counts-mode-note", "not evidence of signalling", "Proportions are compositional",
                   "rank by I, not by the p-value", "no test is run"):
        assert needle in text, needle
    assert xe._JOB_KIND_NAMES["moran"] and xe._JOB_KIND_NAMES["lr"]
