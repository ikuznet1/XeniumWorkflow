"""Regression tests for the ninth review round. One test per finding (R9-<n> in the comment
above it, n = the finding's number in the review list); the name says which behaviour is
pinned. Where a rule was moved into one shared helper, a source-level test checks that
every call site uses the helper (the earlier rounds fixed several of these rules at one
call site only)."""
from __future__ import annotations

import inspect
import io
import json
import os
import pathlib
import re
import sys
import threading
import types

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import xe_testlib as tl
from test_controls import _fab_ds

REPO = pathlib.Path(__file__).resolve().parents[1]


def _src(name: str) -> str:
    return (REPO / "xenium_app" / f"{name}.py").read_text(encoding="utf-8")


def _all_sources() -> dict:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted((REPO / "xenium_app").glob("*.py"))}


@pytest.fixture(autouse=True)
def _clean(xe):
    before = list(xe._prov_recent)
    yield
    xe._prov_recent[:] = before
    xe._stats_reset()
    xe._controls_reset()
    for f in os.listdir(xe.CACHE_BASE_DIR):
        if f.startswith("design_") and f.endswith(".json"):
            os.remove(os.path.join(xe.CACHE_BASE_DIR, f))


# ── R9-17: a sample without the chosen cluster set never lends another set's ids ─────
def test_statistics_never_read_another_cluster_set(xe, D, tmp_path):
    from xenium_app import stats
    e1 = _fab_ds(D, 40, 1, str(tmp_path / "a"))
    e2 = _fab_ds(D, 40, 2, str(tmp_path / "b"))
    for e, sets in ((e1, {"mine": [1, 2] * 20, "graph": [5] * 40}), (e2, {"graph": [3, 7] * 20})):
        e["cluster_methods"] = list(sets)
        for k, v in sets.items():
            e["df"][f"clust__{k}"] = v
    xe.EXTRA_DATASETS.extend([e1, e2])
    samples = xe._an_samples("xenium", False)
    assert [s["label"] for s in samples] == ["S0", "S1", "S2"]
    labs = stats._raw_cluster_labels(samples, "mine", None)
    assert labs[0] is None, "the primary sample has no set 'mine'"
    assert sorted(set(labs[1])) == ["1", "2"]
    assert labs[2] is None, "S2 has no set 'mine': its 'graph' ids were returned as clusters of 'mine'"
    # the population of a pseudobulk: no cell of S2, and the label says why
    masks, label = stats._pb_population_masks("cluster:3", samples, "xenium", "mine", "cluster")
    assert not masks[2].any() and not masks[0].any()
    assert "S2" in label and "without this cluster set" in label
    # composition by cluster: S2 is reported as left out, never counted
    xe._design_set([{"key": xe._ds_key(e), "cond": c, "batch": "", "donor": ""} for e, c in ((e1, "A"), (e2, "B"))])
    res = stats._comp_compute("cluster", "xenium", "mine", "logit", "exact")
    assert list(res["counts"].columns) == ["S1"]
    assert any("S2 has no cluster labels and is left out" in w for w in res["warnings"])
    # the view keeps its fallback: a panel without the chosen set is still coloured
    view = xe._an_cluster_labels(samples, "mine", False, None)
    assert view[2] is not None and view[2][0] == "S2:C3"


# ── R9-18: imputed gene columns do not make the measured counts "not raw" ───────────
def test_integer_check_looks_at_measured_genes_only():
    from xenium_app import stats
    rng = np.random.default_rng(0)
    counts = rng.poisson(1.0, size=(50, 5)).astype(np.float32)
    imputed = rng.uniform(0.01, 0.9, size=(50, 2)).astype(np.float32)
    genes = [f"g{i}" for i in range(5)]
    gni = {g: i for i, g in enumerate(genes + ["imp0", "imp1"])}

    def _s(mat, panel, ds=None):
        return {"csc": sp.csc_matrix(mat), "rows": np.arange(50), "gni": gni, "genes": panel,
                "df": pd.DataFrame(index=range(50)), "label": "S0", "ds": {} if ds is None else ds}

    s = _s(np.hstack([counts, imputed]), genes)
    assert stats._counts_are_integer(s), "imputed (fractional) columns were taken for counts"
    tot, n = stats._pb_sum_counts(s, np.ones(50, bool))
    assert n == 50 and np.array_equal(tot[:5], counts.sum(axis=0).astype(np.int64))
    # a measured column in the middle with a fraction: still refused
    bad = counts.copy()
    bad[3, 2] = 1.5
    assert not stats._counts_are_integer(_s(np.hstack([bad, imputed]), genes))
    # imputed columns first, measured genes after them (any column order)
    gni2 = {"imp0": 0, "imp1": 1, **{g: i + 2 for i, g in enumerate(genes)}}
    s2 = _s(np.hstack([imputed, counts]), genes)
    s2["gni"] = gni2
    assert stats._counts_are_integer(s2)
    # the verdict is kept per (matrix, measured columns): a wider panel is checked again
    ds = {}
    s3 = _s(np.hstack([counts, imputed]), genes, ds)
    assert stats._counts_are_integer(s3)
    s3["genes"] = genes + ["imp0"]
    assert not stats._counts_are_integer(s3)
    # negative values are never counts
    neg = counts.copy()
    neg[0, 0] = -1
    assert not stats._counts_are_integer(_s(np.hstack([neg, imputed]), genes))


# ── R9-19: every line of a CSV header block is a comment line ───────────────────────
def _header_lines(text: str) -> list:
    """Lines before the column header of a CSV download."""
    out = []
    for ln in text.split("\n"):
        if ln.startswith("category,") or ln.startswith("gene,"):
            break
        out.append(ln)
    return out


def test_composition_csv_header_has_no_uncommented_line(xe, D, tmp_path):
    types_ = ("Alpha", "Beta", "Gamma")
    extras = [_fab_ds(D, 300, i, str(tmp_path / f"c{i}"), types_, (0.6, 0.2, 0.2) if i < 3 else (0.2, 0.6, 0.2))
              for i in range(6)]
    xe.EXTRA_DATASETS.extend(extras)
    xe._design_set([{"key": xe._ds_key(e), "cond": "A" if i < 3 else "B", "batch": "", "donor": ""}
                    for i, e in enumerate(extras)])
    res = xe._comp_compute("celltype:celltypist", "xenium", None, "logit", "exact")
    assert "\n⚠" in res["test_note"] and "no test can be significant" in res["test_note"]   # 3 vs 3: floor 0.1
    csv = xe._comp_csv(res)
    head = _header_lines(csv)
    assert head and all(ln.startswith("#") for ln in head), [ln for ln in head if not ln.startswith("#")]
    assert any("no test can be significant" in ln for ln in head)
    got = pd.read_csv(io.StringIO(csv), comment="#")
    assert list(got.columns)[0] == "category" and sorted(got["category"]) == sorted(types_)
    assert head[-1].startswith("# provenance record")


def test_every_csv_builder_passes_its_header_through_one_helper():
    from xenium_app import provenance
    assert provenance._csv_comments(["# a", "b\n⚠ c", ""]) == "# a\n# b\n# ⚠ c\n"
    assert provenance._csv_comments("# one\r\ntwo") == "# one\n# two\n"
    assert provenance._csv_comments([]) == ""
    missing = []
    for name, text in _all_sources().items():
        for m in re.finditer(r"_prov_csv_stamp\(", text):
            if name == "provenance":
                continue
            # the enclosing function builds its comment block with the helper
            start = text.rfind("\ndef ", 0, m.start())
            if "_csv_comments(" not in text[start:m.end() + 400]:
                missing.append(f"{name}:{text.count(chr(10), 0, m.start()) + 1}")
    assert not missing, f"CSV builders that do not use provenance._csv_comments: {missing}"


# ── R9-20: the reported p floor is that of the design, not of one value vector ──────
def test_p_floor_of_a_family_is_the_design_floor(xe):
    from xenium_app import stats
    conds = list("AAAABBBB")
    r = stats._sl_test(np.array([0.7, 0, 0, 0, 0, 0, 0, 0.0]), conds)      # all relabellings tie
    assert r["p"] == 1.0 and r["min_p"] == pytest.approx(2 / 70)
    # the closed form is what the enumeration gives for values without ties
    for na, nb in ((3, 3), (3, 4), (4, 4), (2, 5)):
        e = stats._sl_permutation(np.arange(na) + 10.0, -np.arange(nb) * 1.5)
        assert stats._sl_design_floor(e) == pytest.approx(e["min_p"]) == pytest.approx(e["p"])
    for n in (3, 4, 6):
        e = stats._sl_signflip(np.arange(1, n + 1) * 1.1)
        assert stats._sl_design_floor(e) == pytest.approx(e["min_p"]) == pytest.approx(2 / 2 ** n)
    # paired: a donor with a zero difference does not double the floor
    r = stats._sl_test(np.array([1.0, 0, 1, 0, 0, 0, 2, 0]), list("ABABABAB"), ["a", "a", "b", "b", "c", "c", "d", "d"])
    assert r["paired"] and r["min_p"] == pytest.approx(2 / 2 ** 4)
    # composition: first category present in one sample only
    sub = pd.DataFrame([[50, 0, 0, 0, 0, 0, 0, 0], [100, 120, 110, 130, 300, 310, 290, 320],
                        [300, 310, 290, 320, 100, 120, 110, 130]], index=["a0", "b", "c"],
                       columns=[f"S{i}" for i in range(8)])
    tests, note = stats._comp_exact(sub, list(range(8)), 8, conds, [""] * 8, "asin", {"A": "A", "B": "B"}, "x")
    assert tests["p_exact"].iloc[0] == 1.0 and tests["p_exact"].iloc[1] == pytest.approx(2 / 70)
    assert "Smallest achievable p: 0.0286" in note and "no test can be significant" not in note


# ── R9-21 / R9-22: depth over the genes the pooled samples share ────────────────────
def _mixed_panel_view(xe, D, tmp_path, n=120):
    """Two extra samples with the SAME cells and counts; the first one's panel is the
    first 30 genes only (the counts of genes 30-59 exist in the matrix of the second)."""
    small = _fab_ds(D, n, 0, str(tmp_path / "small"))
    big = _fab_ds(D, n, 0, str(tmp_path / "big"))
    big["df"].index = [f"b{i}" for i in range(n)]
    big["_cell_type_labels"].index = big["df"].index
    small["gene_names"] = list(D["gene_names"])[:30]
    small["expr"] = sp.csr_matrix(sp.hstack([small["expr"][:, :30],
                                              sp.csr_matrix((n, small["expr"].shape[1] - 30), dtype=np.float32)]))
    xe.EXTRA_DATASETS.extend([small, big])
    samples = xe._an_samples("xenium", False)[1:]
    assert [len(s["genes"]) for s in samples][0] == 30 < len(samples[1]["genes"])
    labels = [small["_cell_type_labels"].to_numpy(dtype=object), big["_cell_type_labels"].to_numpy(dtype=object)]
    return samples, labels


def test_pooled_rates_use_the_library_over_shared_genes(xe, D, tmp_path, monkeypatch):
    from xenium_app import analysis, controls, lrdecomp
    samples, labels = _mixed_panel_view(xe, D, tmp_path)
    # R9-22: type profiles, the per-cell marker share and the group profile
    prof = controls._type_profiles(samples, labels)
    assert np.allclose(prof["per_sample"][0]["depth"], prof["per_sample"][1]["depth"]), \
        "the same cells got another depth in the sample with the larger panel"
    assert prof["lib_shared_genes"] == 30
    assert analysis._an_lib_genes(samples) == samples[0]["genes"] and analysis._an_lib_genes(samples[:1]) is None
    assert np.allclose(prof["per_sample"][0]["counts"], prof["per_sample"][1]["counts"])
    markers = samples[0]["genes"][:3]
    sp_res = controls._spillover(samples, labels, "Alpha", markers)
    assert np.allclose(sp_res["per_sample"][0]["share"], sp_res["per_sample"][1]["share"], equal_nan=True)
    ones = [np.ones(len(s["df"]), bool) for s in samples]
    _c, dep2, _n = controls._group_profile(samples, ones, prof["genes"])
    _c, dep1, _n = controls._group_profile(samples[:1], ones[:1], prof["genes"])
    assert dep2 == pytest.approx(2 * dep1)
    # one sample, or the same panel everywhere: the sample's own panel, and no note
    alone = controls._type_profiles(samples[1:], labels[1:])
    assert alone["lib_shared_genes"] is None and (alone["depth"] > prof["per_sample"][1]["depth"]).all()
    # R9-21: the exposure decomposition
    seen = []
    real = lrdecomp._lrd_sample_sums
    monkeypatch.setattr(lrdecomp, "_lrd_sample_sums", lambda *a, **k: seen.append(real(*a, **k)) or seen[-1])
    g = samples[0]["genes"]
    res = lrdecomp._lrd_compute(samples, labels, [(g[0], g[1]), (g[2], g[3])], min_cells=5, floor=0, own=0)
    assert len(seen) == 2 and np.allclose(seen[0]["sum_s"], seen[1]["sum_s"]) and seen[0]["sum_s"].sum() > 0
    assert res["lib_shared_genes"] == 30
    assert "30 genes shared" in analysis._an_lib_note(res["lib_shared_genes"])
    binary = lrdecomp._lrd_compute(samples, labels, [(g[0], g[1])], min_cells=5, floor=0, own=0, binary=True)
    assert binary["lib_shared_genes"] is None            # detection rates do not use the depth


def test_depth_columns_come_from_one_helper():
    """No module builds the library columns from s["genes"] itself (the rule - shared genes
    when samples are pooled - lives in analysis._an_lib_cols / _an_depth)."""
    pat = re.compile(r"""\[\s*s\["gni"\]\[\w+\]\s+for\s+\w+\s+in\s+s\["genes"\]""")
    hits = [name for name, text in _all_sources().items() if pat.search(text)]
    assert hits == [], hits
    for name, fn in (("lrdecomp", "_lrd_sample_sums"), ("controls", "_type_profiles"),
                     ("controls", "_marker_counts"), ("controls", "_group_profile")):
        mod = __import__(f"xenium_app.{name}", fromlist=[fn])
        body = inspect.getsource(getattr(mod, fn))
        assert "analysis._an_lib_cols(" in body or "analysis._an_depth(" in body, fn
    # status line, CSV header and methods text of the two results state the rule
    assert "_an_lib_note" in _src("lrdecomp") and "_an_lib_note" in _src("controls")
    assert "_pm_lib_shared(r)" in inspect.getsource(sys.modules["xenium_app.provenance"]._pm_lrdecomp)
    assert "_pm_lib_shared(r)" in inspect.getsource(sys.modules["xenium_app.provenance"]._pm_spillover)


# ── R9-23: sections of one donor are one unit also when another sample is unlabelled ─
def test_partial_donor_labels_do_not_make_sections_replicates(xe):
    from xenium_app import controls, proximity, stats
    e3 = np.random.default_rng(0).normal(1.0, 0.05, size=(3, 2, 2))
    ac = controls._coloc_across(e3, np.ones((3, 2, 2), bool), ["", "", ""], ["d1", "d1", ""], ["a", "b", "c"])
    assert ac["n_units_max"] == 2 and ac["mode"] == "per-sample", "two sections of d1 were two units"
    assert len(proximity._prox_unit_groups(3, None, ["d1", "d1", ""])[0]) == 2
    ug = stats._sl_unit_groups([0, 1, 2], ["d1", "d1", ""])
    assert sorted(map(len, ug["groups"].values())) == [1, 2] and not ug["by_donor"] and ug["combined"]
    assert ug["note"] == ("n = 3 samples (1 donor, 1 sample without a donor), 2 independent units after "
                          "combining the samples of one donor")
    assert stats._sl_unit_groups([0, 1], ["", ""])["word"] == "samples"
    assert stats._sl_unit_groups([0, 1], ["a", "b"])["word"] == "donors"
    # a sample that happens to be named like a donor stays its own unit
    assert len(stats._sl_unit_groups([0, 1], ["S1", ""], names={0: "S0", 1: "S1"})["groups"]) == 2
    # co-localisation across samples: 2 units, so no cohort test (3 are needed)
    T = 2
    e = np.random.default_rng(0).normal(1.0, 0.05, size=(3, T, T))
    ac = controls._coloc_across(e, np.ones((3, T, T), bool), ["", "", ""], ["d1", "d1", ""], ["a", "b", "c"])
    assert ac["n_units_max"] == 2 and ac["mode"] == "per-sample" and np.isnan(ac["p_enrich"]).all()
    assert "2 independent units" in ac["units_note"]
    # the labelled and the unlabelled designs keep their statements
    ac = controls._coloc_across(e, np.ones((3, T, T), bool), ["", "", ""], ["", "", ""], ["a", "b", "c"])
    assert ac["mode"] == "cohort" and ac["units_note"] == "n = 3 samples"
    # proximity DE: the same units
    groups, by_donor, inc = proximity._prox_unit_groups(3, None, ["d1", "d1", ""])
    assert len(groups) == 2 and not by_donor and inc == [0, 1, 2]
    groups, by_donor, _inc = proximity._prox_unit_groups(4, None, ["d1", "d1", "d2", "d2"])
    assert by_donor and sorted(groups) == ["d1", "d2"]
    # every site takes its units from the one helper
    for fn in (controls._coloc_across, proximity._prox_unit_groups, proximity._contact_compute):
        assert "stats._sl_unit_groups(" in inspect.getsource(fn), fn.__name__
    for name, text in _all_sources().items():
        assert not re.search(r"by_donor\s*=\s*(bool\(inc\) and )?all\(", text), name


# ── R9-1: roi_* labels that came with the store are not blanked by a new ROI ────────
def _square(x0, x1, y0=-1e6, y1=1e6):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _forget_rois():
    from xenium_app import regions
    getattr(regions, "_roi_forget_known", lambda: None)()


def test_stored_roi_labels_without_polygons_survive_a_new_roi():
    from xenium_app import regions
    _forget_rois()
    try:
        df = pd.DataFrame({"x_centroid": [1.0, 2.0, 3.0, 4.0], "y_centroid": [0.0, 0.0, 0.0, 0.0],
                           "roi_region": ["left", "left", pd.NA, "right"],
                           "roi_zone": pd.Categorical(["a", None, None, None])}, index=list("abcd"))
        new = {"name": "z1", "cls": "zone", "polygon_xy": _square(2.5, 4.5), "color": "#fff"}
        regions._roi_apply_metadata_to_df(df, [new])
        # class "region": no polygon was ever held for it - untouched
        assert df["roi_region"].tolist()[:2] == ["left", "left"] and df["roi_region"].tolist()[3] == "right"
        # class "zone": the stored label "a" has no polygon either and stays; the new ROI is added
        assert df["roi_zone"].tolist()[0] == "a" and df["roi_zone"].tolist()[2:] == ["z1", "z1"]
        # deleting the app's own ROI clears its cells only
        regions._roi_apply_metadata_to_df(df, [])
        assert df["roi_zone"].tolist()[0] == "a" and df["roi_zone"].isna().tolist() == [False, True, True, True]
        assert df["roi_region"].notna().sum() == 3
        # a ROI moved: the old cells lose the label, the new ones get it
        regions._roi_apply_metadata_to_df(df, [{**new, "polygon_xy": _square(0.5, 1.5)}])
        assert df["roi_zone"].tolist()[0] == "z1" and df["roi_zone"].isna().tolist()[1:] == [True, True, True]
    finally:
        _forget_rois()
    # callers hold core._roi_lock (a plain lock) while they apply the ROIs: the helper must
    # not take it
    body = inspect.getsource(regions._roi_known_names) + inspect.getsource(regions._roi_apply_metadata_to_df)
    assert "with core._roi_lock" not in body


# ── R9-27: the samples beside the primary one get the ROI label, at their offset ────
def test_roi_labels_reach_added_samples_with_their_offset(xe, D, tmp_path, monkeypatch):
    from xenium_app import core, regions
    e = _fab_ds(D, 50, 3, str(tmp_path / "x"))
    xe.EXTRA_DATASETS.append(e)
    off = float(e["x_offset"])
    half = float(np.median(e["df"]["x_centroid"]))
    roi = {"name": "r1", "cls": "r9test", "polygon_xy": _square(off - 1, off + half), "color": "#fff"}
    monkeypatch.setitem(core.DATA, "df", D["df"].copy())
    _forget_rois()
    try:
        regions._roi_apply_everywhere([roi])
        want = (e["df"]["x_centroid"] < half).to_numpy()
        got = e["df"]["roi_r9test"].notna().to_numpy()
        assert got.sum() > 0 and np.array_equal(got, want), "cells of the added sample inside the outline were not labelled"
        assert not core.DATA["df"]["roi_r9test"].notna().any()          # the outline is beside the primary sample
        # the same cells the Save ROI preview counts
        s = [t for t in xe._an_samples("xenium", False) if t["ds"] is e][0]
        assert int(xe._an_polygon_mask(roi["polygon_xy"], s).sum()) == int(got.sum())
        # a sample added while ROIs exist
        e2 = _fab_ds(D, 50, 4, str(tmp_path / "y"))
        e2["x_offset"] = off
        regions._roi_apply_to_sample(e2, [roi])
        assert e2["df"]["roi_r9test"].notna().sum() > 0
        assert "_roi_apply_to_sample(new_data" in _src("cb_data")
    finally:
        _forget_rois()
        e["df"].drop(columns=["roi_r9test"], inplace=True, errors="ignore")


# ── R9-25: a lasso over an added sample gives the outline of the selected points ────
def test_lasso_hull_in_multi_sample_view_uses_plot_coordinates(xe, D, tmp_path):
    from xenium_app import core, regions
    e = _fab_ds(D, 40, 5, str(tmp_path / "x"))
    xe.EXTRA_DATASETS.append(e)
    off = float(e["x_offset"])
    pts = [{"customdata": cid, "x": float(x) + off, "y": -float(y)}
           for cid, x, y in zip(e["df"].index[:20], e["df"]["x_centroid"][:20], e["df"]["y_centroid"][:20])]
    # ids that also exist in the primary sample must not pull the hull over there
    pts[0]["customdata"] = str(D["df"].index[0])
    with core._roi_lock:
        before = core._roi_state.get("pending_hull")
    try:
        out = xe.capture_lasso_selection({"points": pts}, "xenium", 0)
        hull = out[0]
        assert hull is not None and out[3] == "", out[3]
        xs = [p[0] for p in hull]
        assert min(xs) >= off - 1e-6, "the outline was built from the primary sample's cells"
        assert max(xs) <= off + float(e["df"]["x_centroid"].max()) + 1e-6
    finally:
        with core._roi_lock:
            core._roi_state["pending_hull"] = before
    # single-sample view: by cell id, as before
    del xe.EXTRA_DATASETS[:]
    ids = [str(c) for c in D["df"].index[:12]]
    assert regions._roi_selection_hull([{"customdata": c, "x": 1e9, "y": 1e9} for c in ids]) == regions._roi_compute_hull(ids)
    # both callbacks that turn selected points into an outline use the one rule
    assert "_roi_selection_hull(" in inspect.getsource(sys.modules["xenium_app.cb_spatial"].capture_lasso_selection)
    assert "_roi_selection_hull(" in inspect.getsource(sys.modules["xenium_app.cb_analysis"].de_capture_plotly_selection)


# ── R9-3: an imputed gene of an added sample is found under its dropdown value ──────
def test_extra_sample_gene_values_understand_the_imputed_label(xe, D, tmp_path):
    from xenium_app import core, dataview
    e = _fab_ds(D, 30, 6, str(tmp_path / "x"))
    n_g = e["expr"].shape[1]
    imp = np.linspace(0.1, 3.0, 30).astype(np.float32)
    e["expr"] = sp.hstack([e["expr"], sp.csr_matrix(imp[:, None])]).tocsr()
    e["gene_name_to_idx"] = {**D["gene_name_to_idx"], "R9IMP": n_g}
    got = dataview._extra_gene_values(e, "R9IMP [imp]")
    assert got is not None and np.allclose(got, np.log1p(imp)), "'G [imp]' was not found in the sample that stores 'G'"
    assert np.allclose(dataview._extra_gene_values(e, "R9IMP"), np.log1p(imp))
    # corrected mode: an imputed gene has values in X only (the corrected layer is padding)
    e["split_corrected_expr"] = sp.csr_matrix(e["expr"].shape, dtype=np.float32)
    assert np.allclose(dataview._extra_gene_values(e, "R9IMP [imp]", use_corrected=True), np.log1p(imp))
    # one helper strips the label everywhere
    assert core._gene_base("G [imp]") == "G" == core._gene_base("G [corr+imp]") and core._gene_base("G") == "G"
    assert core._gene_base("A [imp] B") == "A [imp] B"
    for name, text in _all_sources().items():
        if name != "core":
            assert "[:-6]" not in text, f"{name} strips the ' [imp]' label itself"


# ── R9-4: imputed columns are not "original counts" when coverage is decided ────────
def test_corrected_coverage_ignores_imputed_columns():
    from xenium_app import dataview
    counts = np.array([[2, 0], [0, 0], [1, 1]], dtype=np.float32)        # cell 1: no measured counts
    imputed = np.array([[0.4], [0.7], [0.2]], dtype=np.float32)
    expr = sp.csr_matrix(np.hstack([counts, imputed]))
    corr = np.array([[1, 0], [0, 0], [0, 0]], dtype=np.float32)          # cell 2: dropped by RCTD
    gv = pd.DataFrame({"is_imputed": [False, False, True]}, index=["a", "b", "imp"])
    gni = {"a": 0, "b": 1, "imp": 2}
    fresh = {"expr": expr, "split_corrected_expr": sp.csr_matrix(corr), "gene_name_to_idx": gni, "gene_var": gv}
    padded = {"expr": expr, "split_corrected_expr": sp.csr_matrix(np.hstack([corr, np.zeros((3, 1), np.float32)])),
              "gene_name_to_idx": gni, "gene_var": gv}
    assert dataview._corrected_covered(fresh).tolist() == [True, True, False]
    assert dataview._corrected_covered(padded).tolist() == [True, True, False], \
        "after a reload (padded layer) the cell without measured counts lost its corrected value"
    # a segmentation run names its imputed genes in a list
    run = {"expr": expr, "split_corrected_expr": padded["split_corrected_expr"], "gene_name_to_idx": gni,
           "imputed_genes": ["imp"]}
    assert dataview._corrected_covered(run).tolist() == [True, True, False]
    assert dataview._imputed_cols({"gene_name_to_idx": gni}) == set()


# ── R9-9 / R9-10 / R9-14: a run that is named but not loaded is never the Xenium cells
@pytest.fixture
def _proseg_loading(xe):
    """The Proseg state as a cached load leaves it while it runs: named, no result."""
    from xenium_app import core
    with core._proseg_lock:
        before = dict(core._proseg_state)
        core._proseg_state.update({"status": "running", "message": "Loading…", "result": None,
                                   "loading_out_dir": "/fake/run_a"})
    yield "proseg:abcd1234"
    with core._proseg_lock:
        core._proseg_state.clear()
        core._proseg_state.update(before)


def _no_job(*a, **k):
    raise AssertionError("a job was started on the Xenium cells under the name of a run that is not loaded")


def test_jobs_refuse_a_named_run_that_is_not_loaded(xe, D, monkeypatch, _proseg_loading):
    from xenium_app import core, jobs
    seg = _proseg_loading
    monkeypatch.setattr(jobs, "_job_start", _no_job)
    monkeypatch.setattr(core, "_ref_rds_problem", lambda p: None)
    split_before, spage_before = dict(core._split_state), dict(core._spage_state)
    try:
        # SPLIT (R9-9)
        out = xe.run_split(1, "/ref/ref.rds", "Names", 4, 10, 100, [], seg)
        assert "is not loaded" in str(out[0]) and "nothing was corrected" in str(out[0]) and out[2] is False
        assert core._split_state["status"] != "running"
        # SpaGE (R9-10)
        out = xe.start_spage(1, "/ref/ref.rds", 50, "", seg)
        assert out[0] is True and "is not loaded" in out[1] and "nothing was imputed" in out[1]
        assert core._spage_state["status"] != "running"
        # spatial analysis (R9-14)
        out = xe.run_spatial_analysis(1, "labels_celltypist", 10, seg)
        assert out[:2] == (False, True) and "is not loaded" in out[2]
    finally:
        with core._split_lock:
            core._split_state.clear(); core._split_state.update(split_before)
        with core._spage_lock:
            core._spage_state.clear(); core._spage_state.update(spage_before)
    # the one guard, and who uses it
    assert core._seg_job_result("xenium") == (None, "") and core._seg_job_result(None) == (None, "")
    res, err = core._seg_job_result(seg, "nothing was done")
    assert res is None and err.endswith("nothing was done.") and seg in err
    from xenium_app import split, spage, clustering, cb_spatial, cb_spage, cb_tools
    for fn in (split._run_split_correction, spage._run_spage_imputation, clustering._run_reseg_umap,
               cb_spatial.run_split, cb_spatial.run_niche, cb_spatial.run_spatial_analysis,
               cb_spage.start_spage, cb_tools.start_annotation):
        assert "core._seg_job_result(" in inspect.getsource(fn), fn.__name__
    # no job picks the run of a tool by its status on its own any more
    for name in ("split", "spage", "clustering", "cb_tools"):
        assert not re.search(r'_state\["result"\] if .*\["status"\] == "done"', _src(name)), name


def test_job_functions_stop_on_a_run_that_is_not_loaded(xe, monkeypatch, _proseg_loading):
    """The job functions themselves (a session restore or the batch mode reach them without
    the callback): SPLIT and SpaGE end in an error before any R work."""
    from xenium_app import core, rworker, split, spage
    monkeypatch.setattr(rworker, "_rw_call", _no_job)
    monkeypatch.setattr(core, "_ref_rds_problem", lambda p: None)
    for mod, state, lock, call in (
            (split, core._split_state, core._split_lock,
             lambda: split._run_split_correction("/ref/ref.rds", "Names", 1, _proseg_loading)),
            (spage, core._spage_state, core._spage_lock,
             lambda: spage._run_spage_imputation("/ref/ref.rds", 50, "", _proseg_loading))):
        with lock:
            before = dict(state)
        try:
            # (inspect first: on the unguarded code the call below would start R on the Xenium cells)
            fn = split._run_split_correction if mod is split else spage._run_spage_imputation
            assert "core._seg_job_result(" in inspect.getsource(fn)
            call()
            assert state["status"] == "error" and "is not loaded" in state["message"], state["message"]
        finally:
            with lock:
                state.clear(); state.update(before)


# ── R9-2 / R9-13: a streamed SpaGE result of a run is matched to the run's cells ────
def _streamed(path, values, genes):
    import zarr
    z = zarr.open_array(str(path), mode="w", shape=values.shape, dtype="float32")
    z[:] = values
    z.attrs["genes"] = list(genes)
    return str(path)


def test_streamed_spage_of_a_run_is_matched_to_its_cells(xe, D, tmp_path):
    from test_review_round6 import _fake_run
    from xenium_app import figures, spage
    res = _fake_run(xe, D, n=12)
    full = res["cells_df"]
    vals = np.arange(12, dtype=np.float32)[:, None] + 1.0
    res.update({"spage_result_path": _streamed(tmp_path / "r.zarr", vals, ["R9G"]), "spage_result_genes": ["R9G"],
                "spage_result_ids_tag": spage._spage_ids_tag(full.index)})
    got = figures._get_expr_values("R9G [imp]", res)
    assert np.allclose(got, np.log1p(vals[:, 0]))
    # subset() of the run: cells_df / expr are sliced, the positional result is not
    keep = [1, 4, 7, 10]
    res["_cells_df_original"] = full
    res["cells_df"] = full.iloc[keep]
    res["expr"] = res["expr"][keep]
    got = figures._get_expr_values("R9G [imp]", res)
    assert got is not None and len(got) == 4, "one value per cell of the FULL run was returned for a subset"
    assert np.allclose(got, np.log1p(vals[keep, 0])), "values of other cells"
    # computed under another subset of the same size: refused, not drawn on other cells
    res["spage_result_ids_tag"] = spage._spage_ids_tag(full.index[[0, 2, 3, 5]])
    res["spage_result_path"] = _streamed(tmp_path / "s.zarr", vals[:4], ["R9G"])
    assert figures._get_expr_values("R9G [imp]", res) is None
    # ... and accepted for exactly that subset
    res["spage_result_ids_tag"] = spage._spage_ids_tag(res["cells_df"].index)
    assert np.allclose(figures._get_expr_values("R9G [imp]", res), np.log1p(vals[:4, 0]))
    # a reference recorded before the tag existed covers the whole run, or nothing
    res["spage_result_ids_tag"] = None
    assert figures._get_expr_values("R9G [imp]", res) is None           # 4 rows, 12 cells in the run
    assert spage._spage_run_rows({"cells_df": full}, 12) == slice(None)
    assert spage._spage_run_rows({"cells_df": full}, 11) is None
    # every loader of a run restores the reference (with its tag) through one function
    for name, n in (("reseg", 2), ("imgseg", 1)):
        assert _src(name).count("_spage_ref_restore(out_dir)") >= n, name
    assert '"ids_tag": _tag' in _src("spage") and _src("spage").count("spage_result.json") <= 4


# ── R9-11: the current SpaGE result is served, not the stored column of an older run ─
def test_current_spage_result_supersedes_stored_columns(xe, D, tmp_path, expressed_gene):
    from test_review_round6 import _fake_run
    from xenium_app import core, dataview, figures, spage
    # a run: the gene has a (stale) matrix column and is listed by a streamed result
    res = _fake_run(xe, D, n=10)
    gene = list(D["gene_names"])[0]
    new = np.linspace(5.0, 6.0, 10).astype(np.float32)[:, None]
    stale = figures._get_expr_values(gene + " [imp]", res)
    res.update({"spage_result_path": _streamed(tmp_path / "n.zarr", new, [gene]), "spage_result_genes": [gene],
                "spage_result_ids_tag": spage._spage_ids_tag(res["cells_df"].index), "spage_result_stored": False})
    got = figures._get_expr_values(gene + " [imp]", res)
    assert np.allclose(got, np.log1p(new[:, 0])) and not np.allclose(got, stale), "the earlier run's column was served"
    res["spage_result_stored"] = True                    # written into the matrix: the column is the result
    assert np.allclose(figures._get_expr_values(gene + " [imp]", res), stale)
    # the Xenium cells: an in-memory result that was not written back (subset active, no store)
    g = expressed_gene
    df = core.DATA["df"]
    frame = pd.DataFrame({g: np.full(len(df), 7.0, dtype=np.float32)}, index=df.index)
    with core._spage_lock:
        before = dict(core._spage_state)
        core._spage_state.update({"result": frame, "result_path": None, "result_genes": None, "result_stored": False})
    dataview._gene_expr_cache.clear()
    try:
        assert spage._spage_supersedes(g) and not spage._spage_supersedes("NOT_A_GENE")
        got = dataview.get_gene_expression(g + " [imp]")
        assert np.allclose(got, np.log1p(7.0)), "the stored column was served instead of the current result"
        with core._spage_lock:
            core._spage_state["result_stored"] = True
        assert not spage._spage_supersedes(g)
    finally:
        with core._spage_lock:
            core._spage_state.clear(); core._spage_state.update(before)
        dataview._gene_expr_cache.clear()
    # the job says which of the two a result is, at every place it publishes one
    src = _src("spage")
    assert "_wrote_store = True" in src and "_record_xenium_streamed(" in src
    assert src.count('_alt_res["spage_result_path"]') == 1, "a publish path sets the reference on its own"


# ── R9-15: the gene dropdown of a run lists the genes of a streamed result as well ──
def test_run_gene_dropdown_lists_streamed_genes(xe, D):
    from test_review_round6 import _fake_run
    _fake_run(xe, D, n=10, imputed_genes=["OLD1"], spage_result_genes=["NEW1", "OLD1"], spage_result_path="/x.zarr")
    opts = [o["value"] for o in xe.update_gene_options(0, 0, "proseg:t")]
    assert "OLD1 [imp]" in opts and "NEW1 [imp]" in opts, "genes of the streamed result cannot be selected"
    assert opts.count("OLD1 [imp]") == 1


# ── R9-12: CellNEST sender / receiver filters on a numeric annotation column ────────
def test_cellnest_filters_compare_as_text():
    from xenium_app import cellnest
    ccc = pd.DataFrame({"ligand": ["A", "A", "B"], "receptor": ["X", "Y", "X"],
                        "from_cluster_kmeans_10": [1, 2, 3], "to_cluster_kmeans_10": [1.0, np.nan, 2.0]})
    opts = cellnest._cellnest_filter_values(ccc["from_cluster_kmeans_10"]) if hasattr(cellnest, "_cellnest_filter_values") \
        else sorted(str(v) for v in ccc["from_cluster_kmeans_10"].dropna().unique())
    assert opts == ["1", "2", "3"]
    sub = cellnest._apply_cellnest_filters(ccc, f_sender=["3"], annot_col="cluster_kmeans_10")
    assert sub["ligand"].tolist() == ["B"], "a numeric cluster column matched none of the dropdown's values"
    sub = cellnest._apply_cellnest_filters(ccc, f_receiver="1.0", annot_col="cluster_kmeans_10")
    assert len(sub) == 1 and sub["receptor"].tolist() == ["X"]
    sub = cellnest._apply_cellnest_filters(ccc, f_ligand="A", f_receptor=["Y"])
    assert len(sub) == 1
    assert cellnest._cellnest_filter_values(ccc["to_cluster_kmeans_10"]) == ["1.0", "2.0"]
    assert "_cellnest_filter_values(" in _src("cb_tools")


# ── R9-16: "N cells corrected" counts the cells purify() changed ────────────────────
def test_split_changed_cells_ignore_genes_without_corrected_counts():
    from xenium_app import split
    orig = sp.csr_matrix(np.array([[3, 0, 4], [1, 2, 0], [0, 5, 0], [2, 2, 2]], dtype=np.float32))
    # gene 2 is not in the reference: purify() returned genes 0 and 1 only
    corr = sp.csr_matrix(np.array([[3, 0, 0], [1, 1, 0], [0, 5, 0], [2, 2, 0]], dtype=np.float32))
    assert split._split_n_changed(corr, orig, np.array([0, 1])) == 1          # only cell 1 was changed
    assert split._split_n_changed(corr, orig) == 3                            # what was reported before
    body = inspect.getsource(split._run_split_correction)
    assert "_split_n_changed(corrected_mat" in body and "_ret_cols" in body and "n_genes_without_corrected" in body


# ── R9-24: a joint UMAP restored under a subset covers the whole sample ─────────────
from test_joint import world, _fake_joint_set     # noqa: E402,F401  (fixture)


def test_joint_umap_restored_under_a_subset_covers_the_whole_sample(xe, world):
    embedded = [np.arange(len(d["df"])) for d in world.dsets]
    key, tables, _ids, umap, code = _fake_joint_set(xe, world, embedded)
    d0 = world.dsets[0]
    full = d0["df"]
    d0["_df_original"] = full                      # subset() of the first sample: every third cell
    d0["df"] = full.iloc[::3]
    try:
        tables = xe._view_tables("xenium")
        xe._jc_restore_umap(key, tables)
        sub = xe._ctl_get(tables[0], "xenium", "joint_umap_1")
        assert np.isfinite(sub).all() and len(sub) == len(d0["df"])
        d0["df"] = full                            # the subset is cleared
        del d0["_df_original"]
        tables = xe._view_tables("xenium")
        u1 = xe._ctl_get(tables[0], "xenium", "joint_umap_1")
        assert np.isfinite(u1).all(), (f"{int(np.isnan(u1).sum())} of {len(u1)} cells lost their coordinates "
                                       f"when the subset was cleared")
        assert np.array_equal(u1, umap[code == 0, 0])
    finally:
        d0["df"] = full
        d0.pop("_df_original", None)


# ── R9-26: a clicked cell is looked up in the segmentation it was clicked in ────────
def test_cell_info_looks_in_the_clicked_segmentation_first(xe, D):
    from test_review_round6 import _fake_run
    from xenium_app import cb_plots
    res = _fake_run(xe, D, n=8)
    xid = str(D["df"].index[0])                     # the same id in the Xenium table and in the run
    res["cells_df"].index = [xid] + list(res["cells_df"].index[1:])
    res["cells_df"].loc[xid, "x_centroid"] = 123456.0
    panel = xe.show_cell_info({"id": xid, "source": "proseg:t"}, None)
    text = json.dumps(panel.to_plotly_json(), default=str)
    assert "[Proseg]" in text and "123456.0" in text, "the Xenium cell of the same id was shown"
    # a click in the Xenium view, and an id only a run has, behave as before
    assert "[Proseg]" not in json.dumps(xe.show_cell_info(xid, None).to_plotly_json(), default=str)
    other = str(res["cells_df"].index[3])
    if other not in D["df"].index:
        assert "[Proseg]" in json.dumps(xe.show_cell_info(other, None).to_plotly_json(), default=str)
    # what the two renderers put into the store
    assert cb_plots._cell_with_source("c1", "proseg:t") == {"id": "c1", "source": "proseg:t"}
    assert cb_plots._cell_with_source("c1", "xenium") == "c1" == cb_plots._cell_with_source("c1", None)
    assert cb_plots._cell_with_source({"id": "c1", "sample": 1}, "proseg:t") == {"id": "c1", "sample": 1}
    click = {"points": [{"customdata": "c9", "x": 1.0, "y": 2.0}]}
    from dash import callback_context  # noqa: F401
    glue = (REPO / "assets" / "xenium_viewer_dash.js").read_text(encoding="utf-8")
    assert glue.count("selectedCell(cell, info.source)") == 2 and "function selectedCell(cell, source)" in glue
    assert "_cell_with_source(cell_id, seg_source)" in inspect.getsource(cb_plots.capture_click) and click


# ── R9-5: a voxel size other than the preset's reaches Proseg ───────────────────────
def test_proseg_voxel_size_is_applied_without_the_preset(xe, tmp_path, monkeypatch):
    from xenium_app import caching, jobs, reseg
    seen = {}
    monkeypatch.setattr(jobs, "_job_run_process", lambda cmd, **kw: seen.update(cmd=list(cmd)) or 1)
    tx = tmp_path / "transcripts.parquet"
    pd.DataFrame({"x": [0.0]}).to_parquet(tx)

    def _cmd(voxel):
        with pytest.raises(RuntimeError, match="exited with code 1"):
            reseg._proseg_run_single(str(tx), str(tmp_path / "p"), "proseg", voxel, 4)
        return seen["cmd"]

    cmd = _cmd(2)
    # `proseg --xenium` sets the voxel size to 4 after the command line is parsed (1.1.9):
    # the preset's column options are given one by one instead
    assert "--xenium" not in cmd, "--initial-voxel-size beside --xenium has no effect"
    assert cmd[cmd.index("--initial-voxel-size") + 1] == "2.0"
    for flag, val in (("--gene-column", "feature_name"), ("--x-column", "x_location"), ("--y-column", "y_location"),
                      ("--z-column", "z_location"), ("--compartment-column", "overlaps_nucleus"),
                      ("--compartment-nuclear", "1"), ("--cell-id-column", "cell_id"),
                      ("--cell-id-unassigned", "UNASSIGNED"), ("--qv-column", "qv"), ("--fov-column", "fov_name"),
                      ("--transcript-id-column", "transcript_id")):
        assert cmd[cmd.index(flag) + 1] == val, flag
    assert cmd[cmd.index("--min-qv") + 1] == "20" and str(tx) in cmd
    for default in (None, 4, 4.0):
        cmd = _cmd(default)
        assert "--xenium" in cmd and "--initial-voxel-size" not in cmd and "--gene-column" not in cmd
    # the preset written out is what the installed Proseg's own preset sets (when its source is at hand)
    src = sorted(pathlib.Path.home().glob(".cargo/registry/src/*/proseg-1.1.9/src/main.rs"))
    if src:
        body = src[0].read_text().split("fn set_xenium_presets", 1)[1].split("\n}\n", 1)[0]
        named = dict(re.findall(r'args\s*\.(\w+)\s*\.get_or_insert\(String::from\("([^"]*)"\)\)', body.replace("\n", " ")))
        ours = {f.strip("-").replace("-", "_"): v for f, v in reseg._PROSEG_XENIUM_COLUMNS}
        assert named == ours and "args.initial_voxel_size = 4.0;" in body
    # a run of an earlier version that asked for another size is listed with the size it used
    old = caching._format_seg_label("proseg", {"tool": "proseg", "voxel_size": 2.0})
    assert "voxel=4µm" in old and "2.0µm was requested but not applied" in old
    assert "voxel=2.0µm" in caching._format_seg_label("proseg", {"tool": "proseg", "voxel_size": 2.0, "voxel_applied": True})
    assert "voxel=auto" in caching._format_seg_label("proseg", {"tool": "proseg", "voxel_size": None})
    body = inspect.getsource(reseg._run_proseg)
    assert '"voxel_applied": True' in body and "_vapplied" in body


# ── R9-6: cell_area of a Proseg run is an area ──────────────────────────────────────
def test_proseg_cell_area_is_the_outline_area(xe, tmp_path, monkeypatch):
    import gzip
    from xenium_app import jobs, reseg
    sq = lambda s: ([0.0, s, s, 0.0, 0.0], [0.0, 0.0, s, s, 0.0])
    # what one Proseg process leaves behind: a 10 x 10 µm cell of 3 µm height has volume 300 µm³
    pdir = tmp_path / "p"
    pdir.mkdir()
    pd.DataFrame({"cell": [1], "centroid_x": [5.0], "centroid_y": [5.0], "population": [7], "volume": [300.0]}
                 ).to_csv(pdir / "cell-metadata.csv.gz", index=False)
    with gzip.open(pdir / "cell-polygons.geojson.gz", "wt") as fh:
        json.dump({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"cell": 1}, "geometry": {
            "type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]}}]}, fh)
    tx = tmp_path / "transcripts.parquet"
    pd.DataFrame({"x": [0.0]}).to_parquet(tx)
    monkeypatch.setattr(jobs, "_job_run_process", lambda cmd, **kw: 0)
    cells, cb, _tx = reseg._proseg_run_single(str(tx), str(pdir), "proseg", None, 1)
    assert "cell_area" not in cells.columns or cells["cell_area"].iloc[0] != 300.0, \
        "Proseg's volume (µm³) is called cell_area"
    assert cells["cell_volume"].iloc[0] == 300.0
    assert reseg._proseg_cell_area(cells, cb) and cells["cell_area"].iloc[0] == pytest.approx(100.0)
    bounds = {"1": sq(10.0), "2": sq(4.0)}
    # a cached run of an earlier version: Proseg's volume (µm³) under the name cell_area
    df = pd.DataFrame({"x_centroid": [0.0] * 3, "cell_area": [900.0, 150.0, 77.0]}, index=["1", "2", "3"])
    assert reseg._proseg_cell_area(df, bounds) is True
    assert df["cell_volume"].tolist() == [900.0, 150.0, 77.0]
    assert df["cell_area"].tolist()[:2] == [100.0, 16.0] and np.isnan(df["cell_area"].iloc[2])
    assert reseg._proseg_cell_area(df, bounds) is False, "nothing left to migrate: the store is not rewritten"
    # a fresh run: integer cell ids, volume already under its own name
    df = pd.DataFrame({"cell_volume": [900.0, 150.0]}, index=[1, 2])
    assert reseg._proseg_cell_area(df, bounds) is True and df["cell_area"].tolist() == [100.0, 16.0]
    assert reseg._ring_area([0, 2, 0], [0, 0, 2]) == 2.0
    # the live run and both cache-load paths go through it; no path names the volume an area
    src = _src("reseg")
    assert '"volume": "cell_area"' not in src and src.count('"volume": "cell_volume"') >= 3
    assert "_proseg_cell_area(" in inspect.getsource(reseg._run_proseg)
    assert inspect.getsource(reseg._load_cached_proseg).count("_proseg_cell_area(") == 2


# ── R9-7 / R9-8: a run another job works on is not re-run, replaced or deleted ──────
def test_run_in_use_is_not_rerun_replaced_or_deleted(xe, D, tmp_path, monkeypatch):
    from test_review_round6 import _fake_run
    from xenium_app import caching, cb_reseg, core, jobs
    run_dir = tmp_path / "proseg_r9run_abcd1234"
    run_dir.mkdir()
    (run_dir / "marker").write_text("x")
    res = _fake_run(xe, D, out_dir=str(run_dir))
    seg = "proseg:abcd1234"
    monkeypatch.setattr(caching, "_list_cached_seg_runs",
                        lambda: [{"value": seg, "out_dir": str(run_dir), "label": "r9 run"}])
    release, started = threading.Event(), threading.Event()

    def _work():                                   # a job that works on the loaded run (as SPLIT does)
        assert core._seg_result(seg) is res
        started.set()
        release.wait(60)

    state, lock = {"status": "running", "message": "", "result": None}, threading.Lock()
    job = jobs._job_start("split", "SPLIT correction", _work, state=state, lock=lock)
    try:
        assert started.wait(20)
        # Delete run (R9-8): refused, with the job's name; nothing removed, nothing unloaded
        out = xe.confirm_seg_delete(1, seg, 0, 0)
        assert run_dir.is_dir() and (run_dir / "marker").exists(), "the run was deleted under a job that uses it"
        assert out[0] is True and "SPLIT correction" in str(out[-1]) and core._proseg_state["result"] is res
        # a new Proseg run (R9-7): refused before the loaded run is dropped
        monkeypatch.setattr(jobs, "_job_start", _no_job)
        out = xe.run_reseg_modal(1, "proseg", "skip", *([None] * 24))
        assert str(out[0]).startswith("✗") and "SPLIT correction" in out[0] and "in use" in out[0]
        assert core._proseg_state["status"] == "done" and core._proseg_state["result"] is res
        assert cb_reseg._loaded_run_busy("baysor") == ""
        # the commit of a re-run into the same directory (the last line of defence)
        stage = tmp_path / "stage"
        stage.mkdir()
        with pytest.raises(RuntimeError, match="in use by SPLIT correction"):
            caching._seg_stage_commit(str(stage), str(run_dir))
        assert (run_dir / "marker").exists()
        assert jobs._jobs_using_run(res) == ["SPLIT correction"] and jobs._jobs_using_run(str(tmp_path)) == []
    finally:
        release.set()
        job.thread.join(20)
    monkeypatch.undo()
    monkeypatch.setattr(caching, "_list_cached_seg_runs",
                        lambda: [{"value": seg, "out_dir": str(run_dir), "label": "r9 run"}])
    assert jobs._run_in_use(res, "deleted") == ""
    # a cached load of the run in progress: not deleted either (it would be published without its files)
    with core._proseg_lock:
        core._proseg_state.update({"status": "running", "result": None, "loading_out_dir": str(run_dir)})
    out = xe.confirm_seg_delete(1, seg, 0, 0)
    assert run_dir.is_dir() and "being loaded" in str(out[-1])
    # nothing uses it: deleted
    with core._proseg_lock:
        core._proseg_state.update({"status": "done", "result": res, "loading_out_dir": None})
    out = xe.confirm_seg_delete(1, seg, 0, 0)
    assert not run_dir.exists() and out[0] is False and out[1] == "xenium"
    # the loads and the jobs that get the run from their callback are registered as well
    assert _src("cb_reseg").count("uses_run=out_dir") == 2
    for name in ("cb_spatial", "cb_spage", "cb_tools"):
        assert "uses_run=" in _src(name), name


# ── R9-28 / R9-29: the batch dry run asks for what the jobs need, not more ──────────
def test_batch_validation_of_contact_and_transcript_steps(xe, bundle_dir, tmp_path, monkeypatch):
    from xenium_app import dataview
    genes = list(xe.DATA["gene_names"])
    cfg = {"data": {"path": str(bundle_dir)}, "output_dir": str(tmp_path / "out"),
           "steps": [{"kind": "contact"}, {"kind": "density", "params": {"gene": genes[0]}}]}
    # a bundle that has transcripts.zarr.zip and no transcripts.parquet
    real = os.path.exists
    monkeypatch.setattr(os.path, "exists", lambda p: (str(p).endswith("transcripts.zarr.zip")
                                                      or (not str(p).endswith("transcripts.parquet") and real(p))))
    out = xe._batch_run(cfg, dry_run=True, log=lambda _l: None)
    err = "\n".join(out["errors"])
    # contact-group DE compares cells within samples: it never reads the conditions A / B
    assert "step 1 (contact): needs the sample design" not in err, err
    # the transcript jobs read either file, of any loaded sample
    assert "step 2 (density)" not in err, err
    monkeypatch.undo()
    d = tmp_path / "zipped"
    d.mkdir()
    assert not dataview._ds_has_transcripts({"data_dir": str(d)}) and not dataview._ds_has_transcripts({})
    (d / "transcripts.zarr.zip").write_bytes(b"")
    assert dataview._ds_has_transcripts({"data_dir": str(d)})
    # one check for the tabs' notes and the batch validation
    assert "dataview._ds_has_transcripts(ds)" in _src("cb_nav") and "dataview._ds_has_transcripts(ds)" in _src("batch")
    spec = sys.modules["xenium_app.batch"]._BATCH_STEPS
    assert not spec["contact"].get("design") and spec["pseudobulk"].get("design")


# ── R9-30: a job that reads the sample design says so in its record ─────────────────
def test_design_reading_jobs_record_the_design():
    import ast
    missing, seen = [], []
    for name, text in _all_sources().items():
        tree = ast.parse(text)
        for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
            seg = ast.get_source_segment(text, fn)
            if "_prov_record(" not in seg or not re.search(r"_design_of\(|_design_donors\(", seg):
                continue
            for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call)
                         and getattr(n.func, "attr", "") == "_prov_record"):
                seen.append(fn.name)
                kw = {k.arg: k.value for k in call.keywords}
                if not (isinstance(kw.get("uses_design"), ast.Constant) and kw["uses_design"].value is True):
                    missing.append(f"{name}.{fn.name}")
    assert "_run_proximity_de" in seen and "_run_contact_de" in seen and len(seen) >= 8
    assert not missing, f"jobs that read the design but record design_used = False: {missing}"


# ── R9-31: the upload trigger - no dependency cycle, and the same file fires again ──
def test_session_upload_trigger_has_no_cycle_and_is_reset_in_the_page(xe):
    """(The browser check - the same file uploaded twice, no 'Dependency Cycle' on the
    console - is viewer_src/test/session_checks.mjs.)"""
    from xenium_app import app_layout
    cmap = app_layout.app.callback_map
    writers = [k for k in cmap if "session-upload." in k]
    collectors = [(k, cb) for k, cb in cmap.items() if k.split("@")[0].strip(".") == "session-request.data"]
    assert len(collectors) == 1
    ins = {(i["id"], i["property"]) for i in collectors[0][1]["inputs"]}
    assert ("session-upload", "contents") in ins, "the trigger must change whenever a file is chosen"
    assert ("session-upload", "last_modified") not in ins, "the same file chosen again has the same modification time"
    assert writers == [], f"a callback writes the upload component (a cycle with its Input): {writers}"
    src = _src("cb_session")
    assert "window.dash_clientside.set_props('session-upload'" in src and "contents: null" in src
    assert "req.upload = trig.value" in src and 'request.get("upload")' in src
    # the server reads the file from the request
    import base64
    doc = xe._session_snapshot({cid: xe._session_default(cid) for cid in xe._session_controls()}, "download")
    doc["state"]["controls"]["point-size"] = 4
    up = "data:application/json;base64," + base64.b64encode(json.dumps(doc).encode()).decode()
    out = xe.session_action({"action": "upload", "t": 1, "upload": up,
                             "controls": {cid: xe._session_default(cid) for cid in xe._session_controls()}})
    assert out[0].startswith("Session restored") and out[2]["controls"] == {"point-size": 4}
    assert xe.session_action({"action": "upload", "controls": {}})[0].startswith("Not done. the upload is empty")


# ── R9-32: a page that was only reloaded does not replace what this run saved ───────
def test_autosave_ignores_a_reopened_page(xe, D):
    xe._session_auto.update({"baseline": None, "written": None})
    xe._session_auto.pop("pages", None)
    page = {cid: xe._session_default(cid) for cid in xe._session_controls()}
    said = lambda controls, pid: [o for o in xe.session_action({"action": "auto", "controls": controls, "page": pid})
                                  if isinstance(o, str)]
    try:
        assert said(page, "p1") == []                                  # first tick of the run
        assert said({**page, "moran-k": 9}, "p1") == ["saved"]
        saved = json.load(open(xe._session_path("last")))["state"]["controls"]["moran-k"]
        assert saved == 9
        # the page is reloaded: every control is back at its default
        assert said(page, "p2") == [], "the defaults of a reloaded page were saved over the session"
        assert said(page, "p2") == []
        assert json.load(open(xe._session_path("last")))["state"]["controls"]["moran-k"] == 9
        # once that page shows another state its states are saved - also the way back to the defaults
        assert said({**page, "moran-k": 5}, "p2") == ["saved"]
        assert said(page, "p2") == ["saved"]
        assert json.load(open(xe._session_path("last")))["state"]["controls"]["moran-k"] == page["moran-k"]
        # the collector names the page load
        assert "window.__xeSessionPage" in _src("cb_session") and 'request.get("page")' in _src("cb_session")
    finally:
        xe._session_auto.update({"baseline": None, "written": None})
        xe._session_auto.pop("pages", None)
        for slot in ("last", "prev"):
            if os.path.exists(xe._session_path(slot)):
                os.remove(xe._session_path(slot))


# ── R9-33: an integer of any size in a session file is refused, not raised on ───────
def test_huge_integers_in_a_session_file_are_refused_not_raised(xe, D):
    from test_session import _doc, _restore
    from xenium_app import session
    big = 10 ** 400
    assert session._session_is_num(big) is False and session._session_is_num(-big) is False
    assert session._session_is_num(3) and session._session_is_num(2.5) and not session._session_is_num(True)
    assert not session._session_is_num(float("inf")) and not session._session_is_num("3")
    doc = _doc(xe, controls={"moran-k": 7})
    text = json.dumps(doc).replace('"moran-k": 7', '"moran-k": ' + "9" * 400)
    assert "9" * 400 in text
    report = xe._session_restore_text(text, {cid: xe._session_default(cid) for cid in xe._session_controls()}, "test")
    assert "moran-k" not in report["controls"]
    assert any("moran-k" in line for line in report["skipped"] + report["missing"]) or "moran-k" in report["text"]
    doc = _doc(xe, rois=[{"name": "r", "cls": "region", "polygon_xy": [[0, 0], [big, 0], [1, 1]]}])
    report = _restore(xe, doc)                      # refused or skipped with a message: no exception
    assert not xe._roi_state["rois"]


# ── R9-34: a restored gene list is not overwritten by the gene-set picker ───────────
def test_restored_geneset_text_is_not_overwritten_by_the_picker(xe, D):
    from test_session import _doc, _restore
    from xenium_app import app_layout
    stored = list(D["gene_names"][:4])
    with xe._geneset_lock:
        before = dict(xe._geneset_state["sets"])
        xe._geneset_state["sets"]["mine"] = stored
    try:
        edited = "\n".join(stored[:3] + [D["gene_names"][10]])
        report = _restore(xe, _doc(xe, controls={"geneset-select": "mine", "geneset-genes": edited},
                                   genesets={"mine": stored}))
        assert report["ok"] and report["controls"].get("geneset-genes") == edited
        assert report["controls"].get("geneset-select") is None, \
            "the picker would refill the text from the stored set"
        assert "gene-set picker was left empty" in report["text"]
        # the same list as the stored set: the picker is restored
        report = _restore(xe, _doc(xe, controls={"geneset-select": "mine", "geneset-genes": "\n".join(stored) + "\n"},
                                   genesets={"mine": stored}))
        assert report["controls"].get("geneset-select") == "mine"
    finally:
        with xe._geneset_lock:
            xe._geneset_state["sets"].clear()
            xe._geneset_state["sets"].update(before)
    # every callback through which one restored control rewrites the VALUE THE USER TYPED into
    # another restored control is handled in the restore (a dropdown whose options depend on
    # another control is re-validated by its own callback and keeps a value that is valid)
    spec = xe._session_controls()
    ids = set(spec)
    pairs = set()
    for key, cb in app_layout.app.callback_map.items():
        outs = {m.group(1) for m in re.finditer(r"\.{0,3}([A-Za-z0-9_-]+)\.value(?:@[0-9a-f]+)?", key)}
        for i in cb.get("inputs", []):
            if isinstance(i.get("id"), str) and i["property"] == "value" and i["id"] in ids:
                pairs |= {(i["id"], o) for o in outs if o in ids and o != i["id"] and spec[o][0] != "dynamic"}
    handled = {("bk-preset", "bk-lambda"), ("bk-preset", "bk-resolution"), ("geneset-select", "geneset-genes")}
    assert pairs <= handled, f"restored controls that overwrite another restored control: {sorted(pairs - handled)}"
    assert ("geneset-select", "geneset-genes") in pairs


# ── R9-35: no tool process starts once the shutdown sweep has begun ─────────────────
def test_no_tool_process_starts_after_the_shutdown_sweep(xe):
    from xenium_app import jobs
    reset = getattr(jobs, "_jobs_shutdown_reset", lambda: None)
    started = []
    try:
        jobs._jobs_kill_all_processes()                       # (no tool is running in this test)
        try:
            p = jobs._job_popen([sys.executable, "-c", "import time; time.sleep(30)"])
            started.append(p)
        except jobs._JobCancelled:
            pass
        assert not started, "a process was started after the sweep: it would outlive the app"
    finally:
        reset()
        for p in started:
            jobs._job_proc_done(p, grace=0.5)
    # ... and again afterwards (the flag is for a process that is exiting)
    p = jobs._job_popen([sys.executable, "-c", "pass"])
    p.wait(20)
    jobs._job_proc_done(p)
    # the sweep repeats until no process is left that it has not stopped
    body = inspect.getsource(jobs._jobs_kill_all_processes)
    assert "while True" in body and "_jobs_shutting_down = True" in body


# ── R9-36 / R9-37: PDF text encoding and XML-forbidden code points (viewer source) ──
def test_figure_text_encoding_in_pdf_and_svg():
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    fig = (REPO / "viewer_src" / "src" / "figure.js").as_uri()
    code = f"""
      const f = await import({json.dumps(fig)});
      const dec = new TextDecoder('windows-1252'), bad = [];
      for (let b = 0x80; b <= 0x9f; b++) {{
        const ch = dec.decode(Uint8Array.of(b));
        if (ch.codePointAt(0) > 0xff && f.pdfText(ch) !== '(\\\\' + b.toString(8) + ')') bad.push(ch);
      }}
      console.log(JSON.stringify({{bad, oe: f.pdfText('œ'), greek: f.pdfText('α'),
        xml: f.xmlText('T\\uffffcell\\ufffe\\u0001 \\ud83d<x> \\ud83e\\udda0'), keep: f.xmlText('a\\tb\\nc')}}));
    """
    r = subprocess.run([node, "--input-type=module", "-e", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-600:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["bad"] == [], f"Windows-1252 characters written as '?': {got['bad']}"
    assert got["oe"] == "(\\234)" and got["greek"] == "(?)"
    assert got["xml"] == "Tcell &lt;x&gt; \U0001f9a0" and got["keep"] == "a\tb\nc"
    # the bundle the app serves was rebuilt from this source
    bundle = (REPO / "assets" / "xenium_viewer.js").read_text(encoding="utf-8")
    assert "\\ufffe\\uffff" in bundle and re.search(r"(339|0x0?153): ?(156|0x9c)", bundle), \
        "assets/xenium_viewer.js was not rebuilt (node build.mjs --with-image)"
