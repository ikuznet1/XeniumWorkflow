"""Regression tests for the fourth review round (one test per finding; the name says
which behaviour is pinned). They use the bundle under test or small fabricated inputs."""
from __future__ import annotations

import io
import threading
import time

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import xe_testlib as tl


def _halves(s):
    x = s["df"]["x_centroid"].to_numpy(float)
    a = x < np.median(x)
    return a, ~a


def _corrected_with_gaps(D, every=4):
    """A stand-in for SPLIT output: the original counts, with every *every*-th cell that
    has counts given an empty row (as split.py pads the cells RCTD dropped)."""
    corr = sp.csr_matrix(D["expr"]).astype(np.float64).tolil()
    tot = np.asarray(sp.csr_matrix(D["expr"]).sum(axis=1)).ravel()
    gap = np.zeros(corr.shape[0], dtype=bool)
    gap[np.flatnonzero(tot > 0)[::every]] = True
    for r in np.flatnonzero(gap):
        corr.rows[r], corr.data[r] = [], []
    return corr.tocsr(), gap


# ── SPLIT-corrected mode: cells without corrected counts are not zeros ───────────────
def test_corrected_mode_leaves_out_cells_without_corrected_counts(xe, D, monkeypatch):
    from xenium_app import analysis
    monkeypatch.setattr(analysis, "_DE_MIN_CELLS", 3)
    s0 = xe._an_samples("xenium", False)[0]
    a, b = _halves(s0)
    corr, gap = _corrected_with_gaps(D)
    D["split_corrected_expr"] = corr
    cs = xe._an_samples("xenium", True)
    s = cs[0]
    gap_df = gap[np.clip(s0["rows"], 0, None)] & (s0["rows"] >= 0)
    assert gap_df.sum() >= 2 and s["corrected"]
    # the view: these cells have no matrix row, and the number is recorded
    assert s["n_uncorrected"] == int(gap_df.sum()) == xe._an_uncorrected(cs)
    assert (s["rows"][gap_df] == -1).all() and (s["rows"][~gap_df] == s0["rows"][~gap_df]).all()
    assert s0.get("n_uncorrected") == 0 and "without SPLIT-corrected counts" in xe._an_uncorrected_note(3)
    assert xe._an_uncorrected_note(0) == ""
    # differential expression: the same test as on the covered cells alone
    got = xe._de_compute(cs, [a], [b])
    assert got["n_a"] == int((a & ~gap_df).sum()) and got["n_b"] == int((b & ~gap_df).sum())
    assert got["n_no_matrix"] == int(gap_df.sum())
    ref = xe._de_compute([{**s, "rows": s0["rows"]}], [a & ~gap_df], [b & ~gap_df])
    np.testing.assert_allclose(got["table"]["U"], ref["table"]["U"])
    np.testing.assert_allclose(got["table"]["pct_a"], ref["table"]["pct_a"])
    # ... and not the test with these cells as zeros (what the padding rows gave before)
    zero = xe._de_compute([{**s, "rows": s0["rows"]}], [a], [b])
    assert zero["n_a"] + zero["n_b"] == got["n_a"] + got["n_b"] + int(gap_df.sum())
    assert not np.allclose(zero["table"]["pct_a"], got["table"]["pct_a"])
    # gene-set score: no value for them
    genes = list(got["table"].sort_values("pct_a", ascending=False)["gene"][:4])
    arrays, _info = xe._gs_score(cs, genes)
    if arrays is not None:
        assert np.isnan(arrays[0][gap_df]).all() and np.isfinite(arrays[0][~gap_df]).all()
    # gene colour: NaN ("no value"), never 0, in the values both renderers draw
    vals = xe._get_expr_values(genes[0], use_corrected=True)
    assert len(vals) == len(s["df"]) and np.isnan(vals[gap_df]).all() and np.isfinite(vals[~gap_df]).all()
    assert np.isfinite(xe._get_expr_values(genes[0], use_corrected=False)).all()
    eds = {**{k: D[k] for k in ("expr", "df_to_expr", "gene_name_to_idx", "gene_names")},
           "df": D["df"], "split_corrected_expr": corr}
    ev = xe._extra_gene_values(eds, genes[0], True)
    assert np.isnan(ev[gap_df]).all() and np.isfinite(xe._extra_gene_values(eds, genes[0], False)).all()
    # Moran's I and the other consumers go through the same row map: the cell count says so
    if (s["rows"] >= 0).sum() >= 12:
        assert xe._moran_sample(s, k=3)["n_cells"] == int((s["rows"] >= 0).sum())


def test_corrected_mode_de_job_and_csv_state_the_number_left_out(xe, D, monkeypatch):
    from xenium_app import analysis
    monkeypatch.setattr(analysis, "_DE_MIN_CELLS", 3)
    s0 = xe._an_samples("xenium", False)[0]
    corr, gap = _corrected_with_gaps(D)
    D["split_corrected_expr"] = corr
    n_gap = int((gap[np.clip(s0["rows"], 0, None)] & (s0["rows"] >= 0)).sum())
    a, _b = _halves(s0)
    x = s0["df"]["x_centroid"].to_numpy(float)
    with xe._de_lock:
        xe._de_state["selection"] = None
        xe._de_state["groups"] = [{"name": "left", "polygon_xy": [
            [x.min() - 1, -1e9], [float(np.median(x)), -1e9], [float(np.median(x)), 1e9], [x.min() - 1, 1e9]]}]
    method = (D.get("cluster_methods") or [None])[0]
    assert xe.run_de(1, "grp:left", "rest", "xenium", method, "cluster", "corrected")[0] is True
    msg = tl.wait_state(xe._de_state, xe._de_lock)
    res = xe._de_state["result"]
    assert res["corrected"] and res["n_uncorrected"] == n_gap
    assert f"{n_gap:,} cells without SPLIT-corrected counts" in msg
    assert res["n_a"] + res["n_b"] == int((s0["rows"] >= 0).sum()) - n_gap
    csv = xe.download_de_table(1)["content"]
    assert f"# {n_gap:,} cells without SPLIT-corrected counts" in csv
    head = [ln for ln in csv.splitlines() if not ln.startswith("#")][0].split(",")
    assert "neg_log10_pval" in head and "neg_log10_padj" in head and "1e-300" in csv
    # original counts: nothing is left out, nothing is said
    assert xe.run_de(2, "grp:left", "rest", "xenium", method, "cluster", "original")[0] is True
    msg = tl.wait_state(xe._de_state, xe._de_lock)
    assert "without SPLIT-corrected" not in msg and not xe._de_state["result"]["n_uncorrected"]


# ── p-values below the float range ───────────────────────────────────────────────────
def _hand_sample(counts, xy, label="hand"):
    n, g = counts.shape
    genes = [f"g{i}" for i in range(g)]
    df = pd.DataFrame({"x_centroid": xy[:, 0], "y_centroid": xy[:, 1]}, index=[f"{label}_{i}" for i in range(n)])
    return {"label": label, "name": label, "key": label, "df": df, "csc": sp.csc_matrix(counts.astype(np.float32)),
            "rows": np.arange(n), "gni": {x: i for i, x in enumerate(genes)}, "genes": genes,
            "x_offset": 0.0, "corrected": False, "ds": {}}


def test_de_p_values_below_the_float_range_are_not_zero_and_still_order(xe):
    rng = np.random.default_rng(0)
    n = 20000
    a = np.arange(n) < n // 2
    counts = rng.poisson(3.0, size=(n, 5)).astype(float)
    counts[:, 0] = np.where(a, 9, 1) + rng.poisson(0.2, n)          # |z| far above 38
    counts[:, 1] = np.where(a, 6, 2) + rng.poisson(1.0, n)          # above 38 too, but weaker
    s = _hand_sample(counts, rng.uniform(0, 100, (n, 2)))
    res = xe._de_compute([s], [a], [~a])
    t = res["table"].set_index("gene")
    assert abs(t.loc["g0", "z"]) > abs(t.loc["g1", "z"]) > 40
    assert 10.0 ** -t.loc["g0", "neg_log10_pval"] == 0.0            # what the table used to hold
    assert t.loc["g0", "pval"] == t.loc["g1", "pval"] == xe._P_FLOOR > 0 and (t["padj"] > 0).all()
    assert np.isfinite(t["neg_log10_pval"]).all() and t.loc["g0", "neg_log10_pval"] > t.loc["g1", "neg_log10_pval"] > 320
    assert t.loc["g0", "neg_log10_padj"] > t.loc["g1", "neg_log10_padj"] > 300
    big = t["pval"] > 1e-290                                       # the representable ones are unchanged
    np.testing.assert_allclose(-np.log10(t["pval"][big]), t["neg_log10_pval"][big], rtol=1e-9)
    # "most significant first" holds among the floored genes whatever the panel order
    want = list(t.sort_values("neg_log10_pval", ascending=False).index)
    assert want[0] == "g0" and (t["pval"] == xe._P_FLOOR).sum() >= 2     # several genes share the floor
    for order in ([0, 1, 2, 3, 4], [4, 1, 0, 3, 2]):
        tab = res["table"].iloc[order]
        assert list(xe._de_sorted(tab)["gene"]) == want
        assert [r["gene"] for r in xe._de_table_records({**res, "table": tab})] == want
    # a table without log columns (pseudobulk from R): ties at 0 are broken by |statistic|
    pb = pd.DataFrame({"gene": ["x", "y", "z"], "padj": [0.0, 0.0, 0.01], "pval": [0.0, 0.0, 1e-4],
                       "stat": [-40.0, 55.0, 3.0]})
    assert list(xe._de_sorted(pb)["gene"]) == ["y", "x", "z"]


def test_moran_p_values_below_the_float_range_are_not_zero(xe):
    rng = np.random.default_rng(1)
    n = 5000
    xy = rng.uniform(0, 1000, (n, 2))
    counts = rng.poisson(2.0, size=(n, 3)).astype(float)
    counts[:, 0] = np.where(xy[:, 0] < 500, 12, 1)                 # one half of the section
    t = xe._moran_sample(_hand_sample(counts, xy), k=8)["table"].set_index("gene")
    assert t.loc["g0", "z"] > 40 and t.loc["g0", "pval"] == xe._P_FLOOR and t.loc["g0", "padj"] > 0
    assert t.loc["g0", "neg_log10_pval"] > 320 and np.isfinite(t["neg_log10_padj"]).all()
    np.testing.assert_allclose(-np.log10(t.loc["g2", "pval"]), t.loc["g2", "neg_log10_pval"], rtol=1e-9)


# ── nothing is published after Cancel ────────────────────────────────────────────────
def test_a_job_cancelled_during_its_last_step_does_not_publish(xe):
    state, lock = {"status": "running", "message": "", "result": None}, threading.Lock()
    entered, go = threading.Event(), threading.Event()
    gen = xe._dataset_gen

    def _set(status, message, result=None):
        with lock:
            if xe._stale_job(gen, status, message):
                return
            state["status"], state["message"] = status, message
            if result is not None:
                state["result"] = result

    def body():
        try:
            _set("running", "last step (no further step boundary)")
            entered.set()
            go.wait(30)
            _set("done", "finished", result={"x": 1})
        except xe._JobCancelled:
            raise
        except Exception as exc:
            _set("error", str(exc))

    job = xe._job_start("umap", "stub", body, state=state, lock=lock)
    assert entered.wait(10) and xe._job_cancel(job.id)
    go.set()
    job.thread.join(10)
    assert job.state == "cancelled" and (state["status"], state["message"]) == ("error", "Cancelled")
    assert state["result"] is None, "a cancelled job published its result"
    # not cancelled: the same body publishes
    state.update(status="running", message="", result=None)
    entered.clear()
    job = xe._job_start("umap", "stub", body, state=state, lock=lock)
    job.thread.join(10)
    assert job.state == "done" and state["result"] == {"x": 1}
    # shielded (already writing): "done" goes through although Cancel was asked for
    state.update(status="running", message="", result=None)
    entered.clear(), go.clear()

    def shielded():
        xe._job_shield()
        entered.set()
        go.wait(30)
        _set("done", "finished", result={"x": 2})

    job = xe._job_start("umap", "stub", shielded, state=state, lock=lock)
    assert entered.wait(10)
    job.cancel.set()
    go.set()
    job.thread.join(10)
    assert job.state == "done" and state["result"] == {"x": 2}


def test_prepare_patches_cancelled_before_publishing_installs_nothing(xe, D, monkeypatch):
    if not D.get("data_dir"):
        pytest.skip("needs a raw bundle")
    real = xe._sdata_summary
    entered, release = threading.Event(), threading.Event()

    def held(sd):
        entered.set()
        assert release.wait(60)
        return real(sd)

    monkeypatch.setattr(xe, "_sdata_summary", held)
    with xe._sdata_lock:
        before, v0 = xe._sdata_state.get("sdata"), xe._sdata_version
        xe._sdata_state["sdata"] = None
    D_before = D.get("sdata")
    D["sdata"] = None
    try:
        xe.to_spatialdata()
        job = [j for j in xe._jobs.values() if j.kind == "patches"][-1]
        assert entered.wait(120), xe._sdata_state["message"]
        assert xe._job_cancel(job.id)
        release.set()
        job.thread.join(60)
        assert job.state == "cancelled", (job.state, xe._sdata_state["message"])
        assert xe._sdata_state["status"] == "error" and xe._sdata_state["message"] == "Cancelled"
        assert xe._sdata_state.get("sdata") is None and D.get("sdata") is None, "installed after Cancel"
        assert xe._sdata_version == v0
    finally:
        with xe._sdata_lock:
            xe._sdata_state["sdata"] = before
        D["sdata"] = D_before


# ── a segmentation run that is not loaded is never replaced by the Xenium cells ──────
def test_unloaded_segmentation_run_is_not_computed_on_the_xenium_cells(xe, D):
    assert xe._an_samples("xenium", False)
    with xe._proseg_lock:
        xe._proseg_state.update({"status": "loading", "result": None})
    assert xe._an_seg_missing("proseg:abc") and not xe._an_seg_missing("xenium")
    assert xe._an_samples("proseg:abc", False) == []
    assert "proseg:abc" in xe._an_no_matrix_msg("proseg:abc") and "not loaded" in xe._an_no_matrix_msg("proseg:abc")
    with xe._de_lock:
        xe._de_state["result"] = None
    xe._run_de({"kind": "all"}, {"kind": "rest"}, "proseg:abc", None, "cluster", False)
    assert xe._de_state["status"] == "error" and "not loaded" in xe._de_state["message"]
    assert xe._de_state["result"] is None
    with pytest.raises(ValueError, match="not loaded"):
        xe._comp_compute("cluster", "proseg:abc", None)
    with pytest.raises(ValueError, match="not loaded"):
        xe._an_require_seg("baysor:zz")
    xe._an_require_seg("xenium")


# ── pseudobulk: donors and the gene list ─────────────────────────────────────────────
def _pb_view(xe, D, tmp_path, conds, donors, batches=None, n=60):
    from test_controls import _fab_ds
    extras = [_fab_ds(D, n, i, str(tmp_path / f"pb{i}")) for i in range(len(conds))]
    xe.EXTRA_DATASETS.extend(extras)
    xe._design_set([{"key": xe._ds_key(e), "cond": c, "batch": (batches[i] if batches else ""), "donor": d}
                    for i, (e, c, d) in enumerate(zip(extras, conds, donors))])
    samples = xe._an_samples("xenium", False)
    masks = [np.ones(len(s["df"]), bool) for s in samples]
    cd, bt = xe._design_of(samples)
    return samples, masks, cd, bt, extras


def test_pseudobulk_unpaired_donors_are_units_of_condition_only(xe, D, tmp_path):
    # three donors per condition, each labelled: was refused as "confounded"
    samples, masks, cd, bt, _e = _pb_view(xe, D, tmp_path, list("AAABBB"), ["d1", "d2", "d3", "d4", "d5", "d6"])
    prep = xe._pb_prepare(samples, masks, cd, bt, 1, 0, donors=xe._design_donors(samples))
    assert prep["design"] == "~ condition" and prep["batch"] is None and prep["counts"].shape[1] == 6
    assert prep["units_note"] == "n = 6 samples (6 donors)"
    plain = xe._pb_prepare(samples, masks, cd, bt, 1, 0)
    assert np.array_equal(prep["counts"], plain["counts"])


def test_pseudobulk_sections_of_a_donor_are_summed_in_an_unpaired_design(xe, D, tmp_path):
    donors = ["d1", "d1", "d2", "d3", "d4", "d4", "d5", "d6"]
    samples, masks, cd, bt, _e = _pb_view(xe, D, tmp_path, list("AAAABBBB"), donors)
    prep = xe._pb_prepare(samples, masks, cd, bt, 1, 0, donors=xe._design_donors(samples))
    assert prep["design"] == "~ condition" and prep["counts"].shape[1] == 6
    assert prep["labels"][0] == "S1+S2" and prep["col_of_sample"]["S1"] == prep["col_of_sample"]["S2"] == 0
    plain = xe._pb_prepare(samples, masks, cd, bt, 1, 0)["counts"]
    assert np.array_equal(prep["counts"][:, 0], plain[:, 0] + plain[:, 1])
    assert any("summed into one pseudobulk column" in w for w in prep["warnings"])
    assert "6 independent units" in prep["units_note"]


def test_pseudobulk_one_donor_in_both_conditions_is_not_a_paired_design(xe, D, tmp_path):
    samples, masks, cd, bt, _e = _pb_view(xe, D, tmp_path, list("ABAABB"), ["d1", "d1", "d2", "d3", "d4", "d5"])
    prep = xe._pb_prepare(samples, masks, cd, bt, 1, 0, donors=xe._design_donors(samples))
    assert prep["design"] == "~ condition" and prep["counts"].shape[1] == 6
    assert any("d1 occurs in both conditions" in w and "not" in w for w in prep["warnings"])


def test_pseudobulk_paired_design_blocks_on_donor_and_drops_unpaired_donors(xe, D, tmp_path):
    donors = ["d1", "d1", "d2", "d2", "d3", "d3", "d9"]
    samples, masks, cd, bt, _e = _pb_view(xe, D, tmp_path, list("ABABABA"), donors, batches=["r"] * 7)
    prep = xe._pb_prepare(samples, masks, cd, bt, 1, 0, donors=xe._design_donors(samples))
    assert prep["design"] == "~ donor + condition" and prep["counts"].shape[1] == 6
    assert prep["batch"] == ["d1", "d1", "d2", "d2", "d3", "d3"]
    row = [r for r in prep["sample_rows"] if r["label"] == "S7"][0]
    assert not row["used"] and "one condition only" in row["note"]
    assert any("d9" in w for w in prep["warnings"]) and any("batch labels are not used" in w for w in prep["warnings"])
    assert prep["units_note"] == "n = 6 samples (3 donors, paired)"
    # unpaired donors with one batch each: the batch is the blocking term
    xe._design_set([{"key": r["key"], "cond": c, "batch": b, "donor": d} for r, c, b, d in zip(
        xe._design_samples()[1:], list("ABABAB") + [""], ["r1", "r1", "r2", "r2", "r1", "r2", ""],
        ["e1", "e2", "e3", "e4", "e5", "e6", ""])])
    cd, bt = xe._design_of(samples)
    prep = xe._pb_prepare(samples, masks, cd, bt, 1, 0, donors=xe._design_donors(samples))
    assert prep["design"] == "~ batch + condition" and prep["batch"] == ["r1", "r1", "r2", "r2", "r1", "r2"]


def test_pseudobulk_excluded_sample_does_not_change_the_gene_list(xe, D, tmp_path):
    samples, masks, cd, bt, extras = _pb_view(xe, D, tmp_path, ["A", "A", "B", "B", ""], [""] * 5)
    full = xe._pb_prepare(samples, masks, cd, bt, 1, 0)
    extras[4]["gene_names"] = list(D["gene_names"])[:30]            # another, smaller panel - and excluded
    samples = xe._an_samples("xenium", False)
    assert len(samples[5]["genes"]) == 30 and len(xe._an_common_genes(samples)) == 30
    prep = xe._pb_prepare(samples, masks, cd, bt, 1, 0)
    assert prep["n_genes_all"] == full["n_genes_all"] > 30 and prep["genes"] == full["genes"]
    assert np.array_equal(prep["counts"], full["counts"])
    assert [r["note"] for r in prep["sample_rows"] if r["label"] == "S5"] == ["excluded"]
    # the same sample included: the shared genes are tested, and the status says so
    xe._design_set([{"key": r["key"], "cond": c, "batch": "", "donor": ""}
                    for r, c in zip(xe._design_samples()[1:], ["A", "A", "B", "B", "B"])])
    cd, bt = xe._design_of(samples)
    prep = xe._pb_prepare(samples, masks, cd, bt, 1, 0)
    assert prep["n_genes_all"] == 30 and any("share 30 genes" in w for w in prep["warnings"])


# ── confound controls ────────────────────────────────────────────────────────────────
@pytest.fixture
def _controls_clean(xe):
    yield
    xe._controls_reset()
    xe._stats_reset()
    import os
    for f in os.listdir(xe.CACHE_BASE_DIR):
        if f.startswith("design_") and f.endswith(".json"):
            os.remove(os.path.join(xe.CACHE_BASE_DIR, f))


def test_placebo_effect_is_averaged_over_the_samples_that_have_the_pair(xe):
    from test_controls import _niche_world, _prep
    parts = []
    for i in range(4):
        xy, typ, comp = _niche_world(seed=7 + i, n=1500, attract=True)
        if i:
            typ[typ == "A"] = "C"                                   # the focal type exists in sample 0 only
        parts.append((xy, typ, comp, np.full(len(xy), i)))
    xy, typ, comp, sample = (np.concatenate([p[k] for p in parts]) for k in range(4))
    types, code, smp, cnt, levels = _prep(xe, xy, typ, comp, sample=sample)
    prep = {"types": types, "code": code, "sample": smp, "cnt": cnt, "levels": levels,
            "comp": comp, "comp_names": ["niche", "rest"]}
    multi = xe._coloc_placebo(prep, "A", 4, draws=20, seed=1).set_index("target")
    x0, t0, c0, _s = parts[0]
    types1, code1, smp1, cnt1, levels1 = _prep(xe, x0, t0, c0)
    one = xe._coloc_placebo({"types": types1, "code": code1, "sample": smp1, "cnt": cnt1, "levels": levels1,
                             "comp": c0, "comp_names": ["niche", "rest"]}, "A", 1, draws=20, seed=1).set_index("target")
    assert one.loc["B", "e_focal"] > 1.0
    # the three samples without A add nothing: not a quarter of the effect
    assert multi.loc["B", "e_focal"] == pytest.approx(one.loc["B", "e_focal"], rel=0.05)
    assert multi.loc["B", "e_focal"] > multi.loc["B", "random_hi"]


def test_colocalisation_cohort_effect_is_the_tested_statistic(xe, monkeypatch, _controls_clean):
    from test_controls import _fake_view
    tables, labs = _fake_view(xe, monkeypatch, 5, n=1200)
    rng = np.random.default_rng(3)
    for i in (3, 4):                                               # no structure in the two samples left out
        labs[i][:] = rng.permutation(labs[i])
    xe._design_set([{"key": t["key"], "cond": c, "batch": "", "donor": ""}
                    for t, c in zip(tables, ["A", "B", "A", "", ""])])
    xe._run_coloc("xenium", None, "celltype:celltypist", 20.0, 5, None)
    st = xe._coloc_state
    assert st["status"] == "done", st["message"]
    res = st["result"]
    assert res["mode"] == "cohort" and "n = 3 samples" in res["units_note"]
    row = res["summary"].set_index(["focal", "target"]).loc[("A", "B")]
    lg = res["long"]
    lg = lg[(lg["focal"] == "A") & (lg["target"] == "B")].set_index("sample")["effect_corrected"]
    assert len(lg) == 5 and lg[["S3", "S4"]].abs().max() < 0.5 * lg[["S0", "S1", "S2"]].min()
    assert row["n_samples"] == 3 and row["n_positive"] == 3 and row["n_units"] == 3
    assert row["effect"] == pytest.approx(lg[["S0", "S1", "S2"]].mean()) and row["fold"] == pytest.approx(2 ** row["effect"])
    assert row["effect"] > lg.mean() + 0.2                         # not the mean over all five sections
    # without a design all five are the units, and the effect is their mean
    xe._design_set([{"key": t["key"], "cond": "", "batch": "", "donor": ""} for t in tables])
    xe._run_coloc("xenium", None, "celltype:celltypist", 20.0, 5, None)
    row = xe._coloc_state["result"]["summary"].set_index(["focal", "target"]).loc[("A", "B")]
    assert row["n_samples"] == 5 and row["effect"] == pytest.approx(lg.mean())


def test_retention_is_pooled_and_undefined_without_a_reference(xe):
    n_x, n_r = 60, 2
    counts = np.zeros((n_x + n_r, 5))
    counts[:n_x, 0] = 5
    counts[:n_x, 4] = 2
    counts[n_x:, 4] = 3                                             # the rare type: none of its identity genes
    rng = np.random.default_rng(0)
    s = _hand_sample(counts, rng.uniform(0, 200, (n_x + n_r, 2)))
    typ = np.array(["X"] * n_x + ["R"] * n_r, dtype=object)
    sets = {"X": ["g0", "g1"], "R": ["g2", "g3"]}
    m = xe._qc_metrics(s, typ, sets, None, 20.0)
    assert m["ident_sums"]["R"] == 0 and m["ident_sums"]["X"] == 5 * n_x
    half = {**s, "csc": sp.csc_matrix(counts * 0.5), "corrected": True}
    mc = xe._qc_metrics(half, typ, sets, None, 20.0, ref_ident=m["ident_sums"])
    per = mc["per_type"].set_index("type")["retention"]
    assert per["X"] == pytest.approx(0.5) and np.isnan(per["R"])   # was 0 / 1e-9 = 0
    assert mc["row"]["retention"] == pytest.approx(0.5)            # was the mean of 0.5 and 0
    # pooled over the types, weighted by their transcripts (not the mean of the ratios)
    counts2 = counts.copy()
    counts2[n_x:, 2] = 1
    s2 = _hand_sample(counts2, s["df"][["x_centroid", "y_centroid"]].to_numpy())
    m2 = xe._qc_metrics(s2, typ, sets, None, 20.0)
    c3 = counts2 * 0.5
    c3[n_x:, 2] = 0                                                 # the rare type loses everything
    m3 = xe._qc_metrics({**s2, "csc": sp.csc_matrix(c3)}, typ, sets, None, 20.0, ref_ident=m2["ident_sums"])
    assert m3["row"]["retention"] == pytest.approx(150.0 / 302.0)   # the mean of 0.5 and 0 would be 0.25


def test_compartment_rerun_replaces_the_column_set_and_a_cancelled_run_writes_nothing(xe, monkeypatch, _controls_clean):
    from test_controls import _fake_view
    tables, _labs = _fake_view(xe, monkeypatch, 2, n=1200)
    xe._run_compartments("xenium", None, "celltype:celltypist", "C", {"min_edge": 100})
    assert xe._comp_ctl_state["status"] == "done", xe._comp_ctl_state["message"]
    assert {"core", "dominant_fraction", "compartment"} <= set(xe._ctl_available("xenium"))
    old = [xe._ctl_get(t, "xenium", "edge_distance_um").copy() for t in tables]
    # a rerun cancelled while the second sample is computed: neither sample changes
    real, calls = xe._compartments, []

    def second_is_cancelled(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise xe._JobCancelled("Cancelled")
        return real(*a, **k)

    monkeypatch.setattr(xe, "_compartments", second_is_cancelled)
    with pytest.raises(xe._JobCancelled):
        xe._run_compartments("xenium", None, "celltype:celltypist", "C", {"min_edge": 100, "bin_um": 60})
    monkeypatch.setattr(xe, "_compartments", real)
    for t, o in zip(tables, old):
        np.testing.assert_array_equal(xe._ctl_get(t, "xenium", "edge_distance_um"), o)
    # a rerun without labels has no core: the core of the first run does not stay behind
    xe._run_compartments("xenium", None, None, None, {"min_edge": 100})
    assert "no core" in xe._comp_ctl_state["message"]
    have = set(xe._ctl_available("xenium"))
    assert {"edge_distance_um", "local_density", "compartment"} <= have and not have & {"core", "dominant_fraction"}
    assert set(np.unique(xe._ctl_get(tables[0], "xenium", "compartment"))) <= {"edge", "other"}


def test_columns_of_a_job_are_not_stored_under_another_segmentation_run(xe, monkeypatch, _controls_clean):
    df = pd.DataFrame({"x_centroid": [0.0, 1.0], "y_centroid": [0.0, 1.0]}, index=["c1", "c2"])
    table = {"key": "/fake/tag", "df": df}
    now = {"tag": "baysor:/runs/A"}
    monkeypatch.setattr(xe, "_seg_tag", lambda seg_source: now["tag"])
    tag = xe._seg_tag("baysor:x")                                  # resolved when the job reads its tables
    xe._ctl_put(table, "baysor:x", {"spillover": [1.0, 2.0]}, tag=tag)
    assert xe._ctl_get(table, "baysor:x", "spillover").tolist() == [1.0, 2.0]
    now["tag"] = "baysor:/runs/B"                                  # another cached run was picked meanwhile
    with pytest.raises(ValueError, match="segmentation was switched"):
        xe._ctl_put(table, "baysor:x", {"spillover": [9.0, 9.0]}, tag=tag)
    with pytest.raises(ValueError, match="segmentation was switched"):
        xe._ctl_replace([table], "baysor:x", ("spillover",), {"/fake/tag": {"spillover": [9.0, 9.0]}}, tag=tag)
    assert xe._ctl_get(table, "baysor:x", "spillover") is None     # nothing under run B
    now["tag"] = "baysor:/runs/A"
    assert xe._ctl_get(table, "baysor:x", "spillover").tolist() == [1.0, 2.0]
    # every writer of per-cell columns passes the tag it resolved at the start
    import re
    for mod in ("controls", "aggregates", "txfactor"):
        import xenium_app, pathlib
        src = pathlib.Path(xenium_app.__file__).parent.joinpath(f"{mod}.py").read_text()
        calls = re.findall(r"(?<!def )_ctl_(?:put|replace)\((?:[^()]|\([^()]*\))*\)", src)
        calls = [c for c in calls if "tag=tag, drop=family" not in c]
        assert calls and all("tag=" in c for c in calls), (mod, [c[:80] for c in calls if "tag=" not in c])


def test_ownership_csv_states_the_threshold_and_reference_that_were_used(xe, D, labels, tmp_path, _controls_clean):
    from test_controls import _fab_ds
    xe.EXTRA_DATASETS.extend([_fab_ds(D, 200, i, str(tmp_path / f"own{i}")) for i in range(2)])
    gene = D["gene_names"][0]
    assert xe.run_spillover(1, "celltype:celltypist", "auto", gene, 5, 0.5, "xenium", None, "original")[:2] == (True, False)
    job = [j for j in xe._jobs.values() if j.kind == "spillover"][-1]
    job.thread.join(60)
    assert xe._spill_state["status"] == "done", xe._spill_state["message"]
    res = xe._spill_state["result"]
    assert res["own_frac"] == 0.5
    csv = xe._spill_csv(res)
    assert "owners = all types at >= 0.5 of it" in csv and ">= 0.2 of it" not in csv
    assert f"# {res['reference_kind']} marker share inside source cells" in csv
    assert "# mean marker share" in xe._spill_csv({**res, "reference_kind": "mean"})


# ── aggregates ───────────────────────────────────────────────────────────────────────
FAST = {"n_null": 99, "n_control": 3}


def test_aggregates_sample_without_the_class_counts_as_zero(xe, monkeypatch, _controls_clean):
    from test_aggregates import _fake_view
    tables, labs = _fake_view(xe, monkeypatch, 6, planted=[True, True, True, False, False, False], n=900)
    labs[5][labs[5] == "Imm"] = "Str"                              # the strongest "fewer aggregates"
    xe._design_set([{"key": t["key"], "cond": "A" if i < 3 else "B", "batch": "", "donor": ""}
                    for i, t in enumerate(tables)])
    try:
        xe._run_aggregates("xenium", None, "celltype:celltypist", ["Imm"], FAST)
        assert xe._agg_state["status"] == "done", xe._agg_state["message"]
        res = xe._agg_result()
        row = res["samples"].set_index("sample").loc["S5"]
        assert row["n_class"] == 0 and row["per_mm2"] == 0.0 and row["area_fraction"] == 0.0
        assert row["tissue_mm2"] == pytest.approx(res["samples"]["tissue_mm2"].iloc[4], rel=0.2)
        tt = res["tests"]["per_mm2"]
        assert tt["p"] == pytest.approx(2 / 20) and "n = 6 samples" in tt["units_note"]   # all six, not five
        # a sample without a value never shrinks the test silently
        got = xe._agg_across([1.0, 2.0, 3.0, 0.0, 0.0, np.nan], list("AAABBB"), [""] * 6)
        assert got["p"] is None and "no value" in got["why"]
        assert xe._agg_across([1.0, 2.0, 3.0, 0.0, 0.0, np.nan], ["A", "A", "A", "B", "B", ""], [""] * 6)["p"] is None
        assert xe._agg_across([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, np.nan], list("AAABBB") + [""], [""] * 7)["p"] is not None
    finally:
        with xe._agg_lock:
            xe._agg_state.update({"status": "idle", "message": "", "result": None})


def test_aggregates_rerun_clears_the_columns_of_a_sample_it_skipped(xe, monkeypatch, _controls_clean):
    from test_aggregates import _fake_view
    tables, labs = _fake_view(xe, monkeypatch, 2, planted=[True, True], n=900)
    try:
        xe._run_aggregates("xenium", None, "celltype:celltypist", ["Imm"], FAST)
        assert xe._agg_state["status"] == "done", xe._agg_state["message"]
        for t in tables:
            assert np.isfinite(xe._ctl_get(t, "xenium", "aggregate_id")).any()
        labs[1][labs[1] == "Fol"] = "Str"                          # the second class has no cell in S1
        xe._run_aggregates("xenium", None, "celltype:celltypist", ["Fol"], {**FAST, "n_min": 5})
        assert xe._agg_state["status"] == "done", xe._agg_state["message"]
        assert xe._agg_result()["samples"]["n_class"].tolist()[1] == 0
        for name in ("aggregate_id", "aggregate_hotspot", "aggregate_p"):
            v = xe._ctl_get(tables[1], "xenium", name)
            assert v is None or not np.isfinite(pd.to_numeric(pd.Series(v), errors="coerce")).any(), name
        col = xe._agg_color("agg#id", "xenium")
        if col is not None:
            assert not np.isfinite(col[0][len(tables[0]["df"]):]).any()
    finally:
        with xe._agg_lock:
            xe._agg_state.update({"status": "idle", "message": "", "result": None})


@pytest.mark.parametrize("poll, state, lock, n_out", [
    ("poll_agg", "_agg_state", "_agg_lock", 5), ("poll_txf", "_txf_state", "_txf_lock", 5),
    ("poll_banksy", "_bk_state", "_bk_lock", 6), ("poll_joint", "_jc_state", "_jc_lock", 6)])
def test_poll_shows_the_final_message_of_a_cancelled_job(xe, poll, state, lock, n_out):
    st = getattr(xe, state)
    with getattr(xe, lock):
        keep = dict(st)
        st.update({"status": "error", "message": "Cancelled"})    # what _job_finish writes; no version bump
        seen = st["version"]
    try:
        out = getattr(xe, poll)(1, seen, 0, 0) if n_out == 6 else getattr(xe, poll)(1, seen, 0)
        assert len(out) == n_out and out[0] == "Cancelled" and out[1] is False and out[3] is True
        with getattr(xe, lock):
            st.update({"status": "running", "message": "step 2"})
        out = getattr(xe, poll)(1, seen, 0, 0) if n_out == 6 else getattr(xe, poll)(1, seen, 0)
        assert out[0] == "step 2" and out[1] is True and out[3] is False
    finally:
        with getattr(xe, lock):
            st.clear()
            st.update(keep)


# ── niches, segmentation-free abundance, export ──────────────────────────────────────
def test_subset_made_while_niches_are_computed_is_not_undone(xe, D, raw, monkeypatch):
    n_full = len(D["df"])
    keep = list(raw.cells.index[: max(8, 2 * len(raw.cells) // 3)])
    real = xe._job_step
    entered, release = threading.Event(), threading.Event()

    def held(*a, **k):                     # the job has taken its snapshot of DATA["df"]
        entered.set()
        assert release.wait(30)
        return real(*a, **k)

    monkeypatch.setattr(xe, "_job_step", held)
    th = threading.Thread(target=xe._run_spatial_niches, args=(2, 3, None, None))
    th.start()
    assert entered.wait(30)
    xe.subset(cell_ids=keep)
    assert len(D["df"]) == len(keep) < n_full
    release.set()
    th.join(60)
    assert xe._niche_state["status"] == "done", xe._niche_state["message"]
    # the subset is still the view, aligned with its expression rows, and carries the niches
    assert list(D["df"].index.astype(str)) == keep and len(D["df_to_expr"]) == len(D["df"])
    assert D["df"]["niche"].notna().all()
    xe.unsubset()
    assert len(D["df"]) == n_full and D["df"]["niche"].loc[keep].notna().all()


def test_cleared_hole_input_means_no_hole_is_filled(xe, D, labels, expressed_gene):
    import os
    if not os.path.exists(os.path.join(D["data_dir"], "transcripts.parquet")):
        pytest.skip("bundle without transcripts.parquet")
    out = {}
    for hole in (None, 0, "all"):
        xe._run_segfree("xenium", None, "celltype:celltypist", "Alpha", expressed_gene, 5, False,
                        "25", "cells", hole, 0)
        assert xe._sfree_state["status"] == "done", xe._sfree_state["message"]
        res = xe._sfree_state["result"]
        out[hole] = (res["hole_max_bins"], xe._sfree_state["message"], xe._sfree_csv(res) if hasattr(xe, "_sfree_csv") else "")
    assert out[None][0] == 0 and out[0][0] == 0 and out["all"][0] is None
    assert "every interior hole filled" not in out[None][1] and "every interior hole filled" in out["all"][1]
    assert "up to None" not in out["all"][2]


def test_resaving_rewrites_numeric_control_columns_the_store_already_has(xe, D, labels, tmp_path, monkeypatch, _controls_clean):
    import os
    import spatialdata
    order = np.argsort(-np.asarray(D["expr"].sum(axis=0)).ravel())
    g1, g2 = D["gene_names"][int(order[0])], D["gene_names"][int(order[1])]
    table0 = xe._view_tables("xenium")[0]

    def spill_and_save(gene, name):
        xe._run_spillover("xenium", None, "celltype:celltypist", "auto", gene, 5, False)
        assert xe._spill_state["status"] == "done", xe._spill_state["message"]
        want = xe._ctl_get(table0, "xenium", "spillover").astype(float)
        flag = [str(v) for v in xe._ctl_get(table0, "xenium", "spillover_flag")]
        msg, poll_disabled, _btn = xe.start_save_sdata(1, str(tmp_path), name, [], "xenium")
        assert poll_disabled is False, msg
        tl.wait_state(xe._save_sdata_state, xe._save_sdata_lock, timeout=600)
        path = os.path.join(str(tmp_path), f"{name}.zarr")
        obs = spatialdata.read_zarr(path).tables["table"].obs
        return path, want, flag, obs

    path1, want1, _f1, obs1 = spill_and_save(g1, "first")
    np.testing.assert_allclose(obs1["ctl_spillover"].to_numpy(float), want1, rtol=1e-6, atol=1e-6, equal_nan=True)
    # the app now works from that store (as when it is started from an exported .zarr)
    monkeypatch.setitem(D, "sdata_path", path1)
    _p2, want2, flag2, obs2 = spill_and_save(g2, "second")
    assert not np.allclose(np.nan_to_num(want1), np.nan_to_num(want2)), "the two runs must differ for this test"
    np.testing.assert_allclose(obs2["ctl_spillover"].to_numpy(float), want2, rtol=1e-6, atol=1e-6, equal_nan=True)
    assert obs2["ctl_spillover_flag"].astype(str).tolist() == flag2     # number and text of the same run


# ── no machine-specific defaults; an empty reference path is a message, not a crash ──
def test_reference_inputs_default_to_the_environment_and_empty_is_refused_clearly(xe, tmp_path):
    import os
    import pathlib
    import xenium_app
    assert xe.REF_RDS_DEFAULT == os.environ.get("XENIUM_REF_RDS", "").strip()
    root = pathlib.Path(xenium_app.__file__).parent
    for f in list(root.glob("*.py")) + list((root.parent / "viewer_src" / "image_test").glob("*.mjs")):
        assert "/Users/" not in f.read_text(), f"{f.name} holds a path of one machine"
    assert "XENIUM_REF_RDS" in xe._ref_rds_problem("") and "XENIUM_REF_RDS" in xe._ref_rds_problem(None)
    assert "not found" in xe._ref_rds_problem(str(tmp_path / "nope.rds"))
    ok = tmp_path / "ref.rds"
    ok.write_bytes(b"x")
    assert xe._ref_rds_problem(str(ok)) is None
    n_jobs = len(xe._jobs)
    # SpaGE, SPLIT, Seurat kNN and RCTD: nothing starts, the status says what to enter
    out = xe.start_spage(1, "", 50, "", "xenium")
    assert out[0] is True and out[1].startswith("✗") and "reference .rds" in out[1] and out[3] is False
    assert xe._spage_state["status"] != "running"
    out = xe.run_split(1, "", "Names", 2, 10, 100, [], "xenium")
    assert "reference .rds" in out[0] and out[1] is True and xe._split_state["status"] != "running"
    out = xe.run_split(1, str(tmp_path / "nope.rds"), "Names", 2, 10, 100, [], "xenium")
    assert "not found" in out[0]
    for source in ("seurat", "rctd"):
        out = xe.start_annotation(1, source, None, "", "Names", "full", 2, 20, 100, "xenium")
        assert out[1].startswith("✗") and "reference .rds" in out[1] and out[2] is False
        assert xe._annot_state["status"] != "running"
    assert len(xe._jobs) == n_jobs
    xe.run_spage("")                                   # the REPL helper prints the same and returns
    assert xe._spage_state["status"] != "running"
    # the gene list default comes from a file named by the environment
    genes = tmp_path / "genes.txt"
    genes.write_text("NPPA\nMYH7\n")
    from xenium_app import core
    keep = core.IMPUTE_GENES_DEFAULT
    try:
        core.IMPUTE_GENES_DEFAULT = str(genes)
        assert xe._impute_genes_default() == "NPPA\nMYH7"
        core.IMPUTE_GENES_DEFAULT = str(tmp_path / "missing.txt")
        assert xe._impute_genes_default() == ""
        core.IMPUTE_GENES_DEFAULT = ""
        assert xe._impute_genes_default() == ""
    finally:
        core.IMPUTE_GENES_DEFAULT = keep
