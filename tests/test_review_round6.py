"""Regression tests for the sixth review round (one test per finding; the name says which
behaviour is pinned). They use the bundle under test or small fabricated inputs."""
from __future__ import annotations

import json
import os
import re
import threading
import time

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import xe_testlib as tl
from test_controls import _fab_ds
from test_review_round4 import _corrected_with_gaps


def _fake_run(xe, D, tool="proseg", n=40, out_dir="/fake/run_a", seed=0, **extra):
    """Install a fabricated Baysor / Proseg result as the loaded run of *tool*: n cells
    with counts in the first genes of the panel, a k-means and a SPLIT cluster column
    that disagree, and corrected counts for every cell."""
    rng = np.random.default_rng(seed)
    genes = list(D["gene_names"])
    ng = min(len(genes), 30)
    dense = np.zeros((n, len(genes)), dtype=np.float32)
    dense[:, :ng] = rng.poisson(3.0, size=(n, ng))
    ids = [str(i + 1) for i in range(n)]
    df = pd.DataFrame({"x": rng.uniform(0, 200, n), "y": rng.uniform(0, 200, n),
                       "x_centroid": rng.uniform(0, 200, n), "y_centroid": rng.uniform(0, 200, n),
                       "transcript_counts": dense.sum(axis=1),
                       "cluster_kmeans_10": [str(i % 2) for i in range(n)],
                       "cluster_split_10": [str((i // 2) % 3 + 5) for i in range(n)]}, index=ids)
    res = {"cells_df": df, "cell_bounds": {}, "out_dir": out_dir, "expr": sp.csr_matrix(dense),
           "split_corrected_expr": sp.csr_matrix(dense), "split_corrected_imputed_genes": [],
           "gene_name_to_idx": dict(D["gene_name_to_idx"]), "source": tool, "sdata_path": None}
    res.update(extra)
    state, lock = xe._seg_state(tool)
    with lock:
        state.update({"status": "done", "message": "", "result": res})
    return res


# ── 1: cluster labels of a segmentation run follow the counts mode ───────────────────
def test_cluster_labels_of_a_run_follow_the_counts_mode_like_the_map(xe, D, monkeypatch):
    from xenium_app import core, stats
    res = _fake_run(xe, D)
    df = res["cells_df"]
    tables = xe._view_tables("proseg:t")
    alt = xe._an_alt_res("proseg:t")
    assert alt is res
    km, spl = df["cluster_kmeans_10"].to_numpy(), df["cluster_split_10"].to_numpy()
    # the reference: what the DE group picker and the map use
    ref = xe._an_cluster_labels(tables, None, True, alt)[0]
    assert list(ref) == list(spl)
    assert list(xe._comp_labels("cluster", tables, "proseg:t", None, True)[0][0]) == list(spl)
    assert list(xe._comp_labels("cluster", tables, "proseg:t", None, False)[0][0]) == list(km)
    assert list(xe._raw_cluster_labels(tables, None, alt, True)[0]) == list(spl)
    # a caller without a counts argument (compartments, composition, the pickers) follows
    # the active mode
    assert list(xe._comp_labels("cluster", tables, "proseg:t", None)[0][0]) == list(km)
    monkeypatch.setattr(core, "_active_counts_mode", "corrected")
    assert list(xe._comp_labels("cluster", tables, "proseg:t", None)[0][0]) == list(spl)
    masks, _ = xe._pb_population_masks("cluster:5", tables, "proseg:t", None, "cluster")
    assert (masks[0] == (spl == "5")).all()
    monkeypatch.setattr(core, "_active_counts_mode", "original")

    # every job that carries a counts mode hands it to the label lookup
    seen = []

    def spy(kind, tables, seg_source, method, use_corrected=None):
        seen.append(use_corrected)
        raise ValueError("stop here")
    monkeypatch.setattr(stats, "_comp_labels", spy)
    xe._run_spillover("proseg:t", None, "cluster", "auto", "", 5, True)
    xe._run_lr("", "cluster", "proseg:t", None, True, 5, 10, 0)
    xe._run_lrdecomp("", "cluster", "proseg:t", None, True, 30.0, 100.0, 0.2, 0.0, False, 5)
    xe._run_proximity_de("5", "6", "cluster", "proseg:t", None, True, {}, None)
    xe._run_contact_de("5", "6", "cluster", "proseg:t", None, True, 20.0, 5, 10, 1, 0, False)
    assert seen and all(v is True for v in seen), seen
    assert len(seen) >= 3, seen


# ── 2: a cell that was not scored carries no spillover flag ──────────────────────────
def test_unscored_cells_get_no_spillover_flag(xe, D, labels, expressed_gene):
    flags = xe._spill_flags(np.array([np.nan, 0.0, 0.3, 1.0]))
    assert list(flags) == [None, "none", "moderate", "high"]
    corr, gap = _corrected_with_gaps(D, every=3)
    D["split_corrected_expr"] = corr
    xe._run_spillover("xenium", None, "celltype:celltypist", "auto", expressed_gene, 5, True)
    st = xe._spill_state
    if st["status"] != "done":
        pytest.skip(f"spillover not computable on this bundle: {st['message']}")
    t = xe._view_tables("xenium")[0]
    sc = np.asarray(xe._ctl_get(t, "xenium", "spillover"), dtype=float)
    fl = pd.Series(xe._ctl_get(t, "xenium", "spillover_flag"), dtype=object)
    assert np.isnan(sc).any(), "the stand-in left no cell without corrected counts"
    assert fl[np.isnan(sc)].isna().all(), "a cell without a score was flagged"
    assert fl[np.isfinite(sc)].isin(xe._SPILL_FLAG_NAMES).all()


# ── 3: n beside the spillover reference counts the cells the reference is over ───────
def test_spillover_source_n_is_the_cells_the_reference_is_computed_on(xe):
    from test_controls import _bleed_world, G_L
    s, typ, _a, _b = _bleed_world()
    markers = [f"g{i}" for i in list(G_L)[:6]]
    full = xe._spillover([s], [typ], "L", markers)
    assert full["n_source"] == int((typ == "L").sum()) == full["n_source_labelled"]
    rows = np.asarray(s["rows"]).copy()
    src = np.flatnonzero(typ == "L")
    rows[src[::3]] = -1                                    # no matrix row (no corrected counts)
    res = xe._spillover([{**s, "rows": rows}], [typ], "L", markers)
    n_cov = int((rows[src] >= 0).sum())
    assert res["n_source"] == n_cov < len(src) == res["n_source_labelled"]


# ── 4: the Plotly UMAP uses the multi-sample counts rule ─────────────────────────────
def test_plotly_umap_uses_original_counts_when_not_every_sample_is_corrected(xe, D, tmp_path):
    if "umap_1" not in D["df"].columns:
        pytest.skip("bundle without a UMAP")
    gene = D["gene_names"][0]
    D["split_corrected_expr"] = sp.csr_matrix(D["expr"]).astype(np.float64) * 0 + sp.csr_matrix(
        np.full(D["expr"].shape, 7.0))
    D["df"]["split_umap_1"] = -D["df"]["umap_1"] * 3 + 100
    D["df"]["split_umap_2"] = D["df"]["umap_2"] * 3 + 100
    method = (D.get("cluster_methods") or [None])[0]
    alone = xe.make_umap_fig("gene", method, gene, 3, 1.0, use_corrected=True)
    assert np.allclose(np.asarray(alone.data[0].marker.color, dtype=float), np.log1p(7.0))
    xe.EXTRA_DATASETS.append(_fab_ds(D, 30, 0, str(tmp_path / "u0")))   # no corrected counts
    xe._umap_fig_cache.clear()
    assert not xe._multi_use_corrected(True)
    fig = xe.make_umap_fig("gene", method, gene, 3, 1.0, use_corrected=True)
    ref = xe.make_umap_fig("gene", method, gene, 3, 1.0, use_corrected=False)
    assert np.allclose(np.asarray(fig.data[0].marker.color, dtype=float),
                       np.asarray(ref.data[0].marker.color, dtype=float))
    assert np.allclose(np.asarray(fig.data[0].x, dtype=float), np.asarray(ref.data[0].x, dtype=float))


# ── 5: the sample prefix of a cluster label is the sample's own label ────────────────
def test_cluster_label_prefix_is_the_samples_label_after_a_skipped_sample(xe):
    def smp(label, n):
        df = pd.DataFrame({"clust__k": [str(i % 3) for i in range(n)]}, index=[f"{label}_{i}" for i in range(n)])
        return {"label": label, "df": df, "ds": {"cluster_methods": ["k"]}}
    labs = xe._an_cluster_labels([smp("S0", 4), smp("S2", 4)], "k", False, None)   # S1 has no matrix
    assert set(labs[0]) == {"S0:C0", "S0:C1", "S0:C2"}
    assert set(labs[1]) == {"S2:C0", "S2:C1", "S2:C2"}
    raw = xe._raw_cluster_labels([smp("S0", 4), smp("S2", 4)], "k", None)
    assert set(raw[1]) == {"0", "1", "2"}


# ── 6: the Banksy cache key says which clustering backend could run ──────────────────
def test_banksy_cache_key_depends_on_the_clustering_backend(xe, monkeypatch):
    from xenium_app import clustersets
    samples = xe._an_samples("xenium", False)
    p = xe._bk_params({})
    genes = xe._an_common_genes(samples)
    monkeypatch.setattr(clustersets, "_cs_leiden_available", lambda: False)
    without = xe._bk_signature(samples, "xenium", p, genes)
    monkeypatch.setattr(clustersets, "_cs_leiden_available", lambda: True)
    assert xe._bk_signature(samples, "xenium", p, genes) != without


# ── 7: a re-run stored in the same directory is another run ──────────────────────────
def test_rerun_in_the_same_directory_does_not_inherit_columns_sets_or_labels(xe, D, tmp_path, monkeypatch):
    from xenium_app import core, caching, controls, clustersets, annotation
    import hashlib
    final = str(tmp_path / "proseg_fake_abc")

    def commit():
        stage = caching._seg_stage_dir(final)
        with open(os.path.join(stage, "params.json"), "w") as fh:
            json.dump({"tool": "proseg"}, fh)
        caching._seg_stage_commit(stage, final)

    commit()
    first = _fake_run(xe, D, out_dir=final)
    rid1 = core._seg_run_id(first)
    assert rid1 and xe._seg_tag("proseg:t").endswith("#" + rid1)
    # per-cell columns, a cluster set and annotation labels of the first run
    t = xe._view_tables("proseg:t")[0]
    controls._ctl_put(t, "proseg:t", {"edge_distance_um": np.arange(len(t["df"]), dtype=np.float32)})
    assert xe._ctl_get(t, "proseg:t", "edge_distance_um") is not None
    key = xe._cs_install("Round six set", [np.arange(len(t["df"])) % 3 + 1], "proseg:t")
    assert f"cluster_{key}" in first["cells_df"].columns
    cs_files = [f for f in os.listdir(clustersets._cs_dir(D)) if "proseg_fake_abc" in f]
    assert cs_files
    lkey = annotation._labels_key_for_method("celltypist", first)
    lab_file = os.path.join(core.CACHE_BASE_DIR,
                            f"Model_{lkey}_{hashlib.md5(caching._dataset_key().encode()).hexdigest()[:8]}.parquet")
    pd.DataFrame({"label": ["T"] * len(t["df"])}, index=t["df"].index).to_parquet(lab_file)
    with core._annot_lock:
        core._annot_state[lkey] = pd.Series("T", index=t["df"].index.astype(str))
    try:
        # the same parameters again: the directory is replaced, the cells keep their ids
        commit()
        second = _fake_run(xe, D, out_dir=final, seed=1)
        rid2 = core._seg_run_id(second)
        assert rid2 and rid2 != rid1 and core._seg_run_id(first) == rid1
        t2 = xe._view_tables("proseg:t")[0]
        assert list(t2["df"].index) == list(t["df"].index)                 # same ids, other cells
        assert xe._ctl_get(t2, "proseg:t", "edge_distance_um") is None
        assert xe._ctl_available("proseg:t") == []
        assert xe._cs_restore("proseg:t") == [] and f"cluster_{key}" not in second["cells_df"].columns
        assert not [f for f in os.listdir(clustersets._cs_dir(D)) if "proseg_fake_abc" in f]
        assert core._annot_state.get(lkey) is None and not os.path.exists(lab_file)
        # a set of this run is cached with the run's id, and restored only for it
        key2 = xe._cs_install("Round six set", [np.arange(len(t2["df"])) % 2 + 1], "proseg:t")
        second["cells_df"].drop(columns=[f"cluster_{key2}"], inplace=True)
        assert xe._cs_restore("proseg:t") == [key2]
        second["cells_df"].drop(columns=[f"cluster_{key2}"], inplace=True)
        second["run_id"] = "another"
        assert xe._cs_restore("proseg:t") == []
    finally:
        with core._annot_lock:
            core._annot_state.pop(lkey, None)
        if os.path.exists(lab_file):
            os.remove(lab_file)
        import shutil
        shutil.rmtree(clustersets._cs_dir(D), ignore_errors=True)
        with clustersets._cs_lock:
            clustersets._cs_registry.pop(key, None)
            for k in [k for k in clustersets._cs_table_run if k[2] == key]:
                clustersets._cs_table_run.pop(k, None)
        controls._controls_reset()


# ── 8: the joint UMAP restore is tried again when the loaded samples change ──────────
from test_joint import world, _fake_joint_set     # noqa: E402,F401  (fixture)


def test_joint_umap_restore_is_retried_after_the_missing_sample_is_loaded(xe, world):
    embedded = [np.arange(len(d["df"])) for d in world.dsets]
    key, tables, _ids, _umap, _code = _fake_joint_set(xe, world, embedded)
    assert len(tables) >= 3
    xe._jc_restore_umap(key, tables[:-1])            # one sample of the run is not loaded yet
    assert all(xe._ctl_get(t, "xenium", "joint_umap_1") is None for t in tables)
    xe._jc_restore_umap(key, tables)                 # now it is
    assert all(xe._ctl_get(t, "xenium", "joint_umap_1") is not None for t in tables)


# ── SpaGE through the job, with R replaced by a stand-in ─────────────────────────────
class _FakeR:
    """rpy2 as far as _run_spage_imputation needs it up to its cache lookup."""

    def __init__(self, genes):
        import types
        self.genes = genes
        self.mods = {}
        ro = types.ModuleType("rpy2.robjects")
        ro.default_converter = object()
        ro.globalenv = {}
        ro.r = lambda code: list(self.genes) if str(code).strip() == "rna_genes" else None
        ro.StrVector = list
        conv = types.ModuleType("rpy2.robjects.conversion")
        conv.set_conversion = lambda c: None
        pk = types.ModuleType("rpy2.robjects.packages")
        pk.importr = lambda name: types.SimpleNamespace(readRDS=lambda path: object())
        rl = types.ModuleType("rpy2.rinterface_lib")
        orl = types.ModuleType("rpy2.rinterface_lib.openrlib")
        orl.rlock = threading.RLock()
        rl.openrlib = orl
        top = types.ModuleType("rpy2")
        top.robjects, top.rinterface_lib = ro, rl
        ro.conversion, ro.packages = conv, pk
        self.mods = {"rpy2": top, "rpy2.robjects": ro, "rpy2.robjects.conversion": conv,
                     "rpy2.robjects.packages": pk, "rpy2.rinterface_lib": rl,
                     "rpy2.rinterface_lib.openrlib": orl}

    def install(self, monkeypatch):
        import sys
        from xenium_app import core
        for k, v in self.mods.items():
            monkeypatch.setitem(sys.modules, k, v)
        monkeypatch.setattr(core, "_redirect_rpy2_console", lambda: None)


@pytest.fixture
def spage_world(xe, D, tmp_path, monkeypatch):
    """A reference file, R replaced by a stand-in, the store copied so that write-back
    does not touch the session's, and DATA put back afterwards."""
    import shutil
    from xenium_app import spage as spage_mod
    ref = tmp_path / "ref.rds"
    ref.write_bytes(b"x")
    panel = [g for g in D["gene_names"]][:5]
    _FakeR(panel + ["GNEW"]).install(monkeypatch)
    monkeypatch.setattr(spage_mod, "_spage_cache_dir", lambda: str(tmp_path))
    keep = {k: D.get(k) for k in ("gene_names", "gene_name_to_idx", "expr", "expr_csc", "gene_var", "sdata_path")}
    if D.get("sdata_path") and os.path.isdir(D["sdata_path"]):
        D["sdata_path"] = shutil.copytree(D["sdata_path"], str(tmp_path / "store.zarr"))
    yield str(ref)
    for k, v in keep.items():
        D[k] = v
    xe._gene_expr_cache.clear()


def _spage_cache(xe, D, ref, value, n_pv=50, suffix="", index=None):
    from xenium_app import spage as spage_mod
    extra = f"_pv{n_pv}" if int(n_pv) != 50 else ""
    path = spage_mod._spage_cache_path(ref, "GNEW" + suffix + extra)
    idx = D["df"].index if index is None else index
    pd.DataFrame({"GNEW": np.full(len(idx), float(value), dtype=np.float32)}, index=idx).to_parquet(path)
    return path


# ── 15: a SpaGE cache hit installs its values ────────────────────────────────────────
def test_spage_cache_hit_writes_the_cached_values_back(xe, D, spage_world):
    if not (D.get("sdata_path") and os.path.isdir(D["sdata_path"])):
        pytest.skip("needs the SpatialData store")
    _spage_cache(xe, D, spage_world, 1.5, n_pv=50)
    _spage_cache(xe, D, spage_world, 4.0, n_pv=30)

    def col():
        j = D["gene_name_to_idx"].get("GNEW")
        return None if j is None else np.asarray(D["expr"][:, j].todense()).ravel()

    xe._run_spage_imputation(spage_world, 30, "GNEW")
    assert xe._spage_state["status"] == "done" and "cached" in xe._spage_state["message"], xe._spage_state["message"]
    assert col() is not None and np.allclose(col(), 4.0)
    # the other run of the same gene, again from the cache: its values replace the column
    xe._run_spage_imputation(spage_world, 50, "GNEW")
    assert xe._spage_state["status"] == "done" and "cached" in xe._spage_state["message"]
    assert np.allclose(col(), 1.5), "the status names one run, the matrix holds another"
    vals = np.asarray(xe.get_gene_expression("GNEW [imp]"), dtype=float)
    assert len(set(np.round(vals, 5))) == 1 and vals[0] > 0


# ── 9: Cancel while a cached SpaGE result is read publishes nothing ──────────────────
def test_cancel_during_a_cached_spage_load_leaves_no_result_on_the_run(xe, D, spage_world, tmp_path, monkeypatch):
    from xenium_app import core, jobs
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    res = _fake_run(xe, D, out_dir=str(run_dir))
    _spage_cache(xe, D, spage_world, 2.0, suffix="__proseg_t", index=res["cells_df"].index)
    real = pd.read_parquet

    def slow_read(path, *a, **k):
        job = jobs._job_current()
        if job is not None and os.path.basename(str(path)).startswith("spage_"):
            job.cancel.set()                          # the user clicks Cancel during the read
        return real(path, *a, **k)

    monkeypatch.setattr(pd, "read_parquet", slow_read)
    job = xe._job_start("spage", "SpaGE", xe._run_spage_imputation, args=(spage_world, 50, "GNEW", "proseg:t"),
                        state=core._spage_state, lock=core._spage_lock)
    job.thread.join(60)
    assert job.state == "cancelled", (job.state, core._spage_state["message"])
    assert not os.path.exists(run_dir / "spage_result.json"), "a cancelled job recorded its result"
    assert "spage_result_path" not in res and "spage_result_genes" not in res


# ── 22: a result computed under a subset is not registered for auto-load ─────────────
def test_in_memory_spage_result_under_a_subset_is_not_registered(xe, D, spage_world):
    from xenium_app import spage as spage_mod
    keep = list(D["df"].index[: max(3, len(D["df"]) // 2)])
    xe.subset(cell_ids=keep)
    # the key of a subset names its cells: their number and a hash of their ids (round 8)
    _spage_cache(xe, D, spage_world, 3.0, suffix=f"_n{len(D['df'])}_{spage_mod._spage_ids_tag(D['df'].index)}")
    xe._run_spage_imputation(spage_world, 50, "GNEW")
    assert xe._spage_state["status"] == "done", xe._spage_state["message"]
    idx = {}
    if os.path.exists(spage_mod._spage_index_path()):
        idx = json.load(open(spage_mod._spage_index_path()))
    assert idx == {}, "a subset-only result was registered as the dataset's SpaGE result"


# ── 10 / 12: pseudobulk gene list and "Dropped" wording ──────────────────────────────
def _narrow_ds(D, n, seed, key, n_genes=20):
    """A sample measured on the first *n_genes* genes of the panel only."""
    ds = _fab_ds(D, n, seed, key, n_genes=n_genes)
    genes = list(D["gene_names"])[:n_genes]
    ds["expr"] = ds["expr"][:, :n_genes].tocsr()
    ds["gene_names"], ds["gene_name_to_idx"] = genes, {g: i for i, g in enumerate(genes)}
    return ds


def test_pseudobulk_genes_are_those_of_the_used_samples(xe, D, tmp_path):
    extras = [_fab_ds(D, 60, i, str(tmp_path / f"g{i}")) for i in range(6)]
    extras.append(_narrow_ds(D, 60, 6, str(tmp_path / "g6")))          # A only, donor d9, small panel
    xe.EXTRA_DATASETS.extend(extras)
    conds = list("ABABAB") + ["A"]
    donors = ["d1", "d1", "d2", "d2", "d3", "d3", "d9"]
    xe._design_set([{"key": xe._ds_key(e), "cond": c, "batch": "", "donor": d}
                    for e, c, d in zip(extras, conds, donors)])
    samples = xe._an_samples("xenium", False)
    masks = [np.ones(len(s["df"]), bool) for s in samples]
    cd, bt = xe._design_of(samples)
    prep = xe._pb_prepare(samples, masks, cd, bt, 1, 0, donors=xe._design_donors(samples))
    row = {r["label"]: r for r in prep["sample_rows"]}["S7"]
    assert not row["used"] and "one condition only" in row["note"]
    # the dropped sample's 20-gene panel does not narrow the list of the six used ones
    assert prep["n_genes_all"] == len(D["gene_names"]) > 20
    assert not any("share" in w for w in prep["warnings"])
    # ... and it is reported with its own reason, not as "fewer than N cells"
    txt = xe._pb_dropped_text([row])
    assert "S7 (60 cells; its donor has samples in one condition only" in txt and "fewer than" not in txt
    # dropped for too few cells: the same
    masks[7] = np.zeros(len(samples[7]["df"]), bool)
    xe._design_set([{"key": xe._ds_key(e), "cond": c, "batch": "", "donor": ""}
                    for e, c in zip(extras, list("ABABABA"))])
    cd, bt = xe._design_of(samples)
    prep = xe._pb_prepare(samples, masks, cd, bt, 5, 0)
    assert prep["n_genes_all"] == len(D["gene_names"])
    # a used sample with a smaller panel still narrows it, and the note says so
    masks[7] = np.ones(len(samples[7]["df"]), bool)
    prep = xe._pb_prepare(samples, masks, cd, bt, 5, 0)
    assert prep["n_genes_all"] == 20 and any("share 20 genes" in w for w in prep["warnings"])


# ── 11: a saved design does not overwrite newer assignments of this session ──────────
def test_saved_design_of_a_sample_set_does_not_revert_newer_assignments(xe, D, tmp_path):
    extras = [_fab_ds(D, 30, i, str(tmp_path / f"d{i}")) for i in range(3)]
    xe.EXTRA_DATASETS.extend(extras[:2])
    k1 = xe._ds_key(extras[0])
    xe._design_set([{"key": k1, "cond": "A", "batch": "", "donor": "old"}], {"A": "treated", "B": "control"})
    xe.EXTRA_DATASETS.append(extras[2])                       # another sample set
    assert xe._design_samples()[1]["cond"] == "A"
    xe._design_set([{"key": k1, "cond": "B", "batch": "b2", "donor": "new"}], {"A": "post", "B": "pre"})
    del xe.EXTRA_DATASETS[2]                                  # back to the set that has a saved design
    got = xe._design_samples()[1]
    assert (got["cond"], got["batch"], got["donor"]) == ("B", "b2", "new")
    assert xe._design_names() == {"A": "post", "B": "pre"}
    # a fresh session (nothing in memory) still gets the saved design back
    xe._stats_reset()
    assert xe._design_samples()[1]["cond"] in ("A", "B") and xe._design_names()["A"] in ("treated", "post")


# ── 14: composition family and n_cells are built from the tested samples ─────────────
@pytest.mark.parametrize("test", ["exact", "moderated"])
def test_composition_ignores_samples_the_paired_rule_leaves_out(xe, D, tmp_path, test):
    types = ("Alpha", "Beta", "Gamma")
    extras = [_fab_ds(D, 200, i, str(tmp_path / f"c{i}"), types, (0.5, 0.3, 0.2)) for i in range(6)]
    extras.append(_fab_ds(D, 200, 6, str(tmp_path / "c6"), ("Alpha", "Zeta"), (0.5, 0.5)))   # donor d9, A only
    xe.EXTRA_DATASETS.extend(extras)
    donors = ["d1", "d1", "d2", "d2", "d3", "d3", "d9"]
    xe._design_set([{"key": xe._ds_key(e), "cond": c, "batch": "", "donor": d}
                    for e, c, d in zip(extras, list("ABABABA"), donors)])
    res = xe._comp_compute("celltype:celltypist", "xenium", None, "logit", test)
    t = res["tests"]
    assert t is not None, res["test_note"]
    assert "Zeta" not in set(t["category"]) and len(t) == 3
    n_alpha = int(res["counts"].loc["Alpha", [f"S{i}" for i in range(1, 7)]].sum())
    assert int(t.set_index("category").loc["Alpha", "n_cells"]) == n_alpha
    assert "left out: d9" in res["test_note"]


# ── 16: a partial obs update merges, it does not blank the other cells ───────────────
def test_obs_update_from_a_subset_keeps_the_values_of_the_other_cells(xe, tmp_path):
    import zarr
    from xenium_app import reseg
    path = str(tmp_path / "run.zarr")
    root = zarr.open_group(path, mode="w")
    obs = root.create_group("tables").create_group("table").create_group("obs")
    ids = np.array([str(i) for i in range(1, 11)])
    stored = {"_index": ids, "cell_area": np.arange(10, dtype=np.float64) + 1.0,
              "cluster_split_10": np.array([str(i % 3) for i in range(10)]),
              "split_umap_1": np.linspace(0, 1, 10), "umap_1": np.linspace(5, 6, 10)}
    for k, v in stored.items():
        obs.create_dataset(k, data=v, shape=v.shape, dtype=v.dtype)
    obs.attrs["_index"] = "_index"
    keep = ids[:8]                                           # an active subset(): 8 of 10 cells
    df = pd.DataFrame({"cell_area": np.full(8, 99.0), "cluster_split_10": ["7"] * 8,
                       "split_umap_1": np.full(8, -1.0), "umap_1": np.full(8, -2.0),
                       "new_col": np.arange(8, dtype=np.float64)}, index=keep)
    reseg._update_reseg_zarr_obs(path, df)
    got = zarr.open_group(path, mode="r")["tables"]["table"]["obs"]
    area = np.asarray(got["cell_area"][:], dtype=float)
    assert np.allclose(area[:8], 99.0) and np.allclose(area[8:], [9.0, 10.0]), area
    cl = [str(v) for v in got["cluster_split_10"][:]]
    assert cl[:8] == ["7"] * 8 and cl[8:] == ["2", "0"], cl
    u = np.asarray(got["split_umap_1"][:], dtype=float)
    assert np.allclose(u[:8], -1.0) and np.allclose(u[8:], stored["split_umap_1"][8:])
    assert np.allclose(np.asarray(got["umap_1"][:], dtype=float), stored["umap_1"])   # not forced: untouched
    new = np.asarray(got["new_col"][:], dtype=float)
    assert np.allclose(new[:8], np.arange(8)) and np.isnan(new[8:]).all()
    # every cell present: plain replacement, as before
    reseg._update_reseg_zarr_obs(path, pd.DataFrame({"cell_area": np.zeros(10)}, index=ids))
    assert np.allclose(np.asarray(zarr.open_group(path, mode="r")["tables/table/obs/cell_area"][:], dtype=float), 0.0)


# ── 17: SPLIT's labels do not replace an annotation by part of the cells / keep foreign weights
def test_split_labels_do_not_overwrite_under_a_subset_and_drop_stale_weights(xe):
    from xenium_app import split
    full = pd.Series("T", index=[str(i) for i in range(100)])
    w = pd.DataFrame({"T": 1.0}, index=full.index)
    part = pd.Series("B", index=[str(i) for i in range(10)])
    st = {"labels_rctd": full, "rctd_weights_labels_rctd": w}
    assert split._split_publish_labels(st, part, "labels_rctd_doublet", "labels_rctd", partial=True) \
        == ["labels_rctd_doublet"]
    assert st["labels_rctd"] is full and st["rctd_weights_labels_rctd"] is w
    # all cells: the labels are replaced, and the weights of the replaced run with them
    st = {"labels_rctd": full, "rctd_weights_labels_rctd": w}
    lab2 = pd.Series("B", index=full.index)
    split._split_publish_labels(st, lab2, "labels_rctd_doublet", "labels_rctd", partial=False)
    assert st["labels_rctd"] is lab2 and st["labels_rctd_doublet"] is lab2
    assert st["rctd_weights_labels_rctd"] is None
    import inspect
    assert "_split_publish_labels(core._annot_state" in inspect.getsource(split._run_split_correction)


# ── 18: subset(cell_type=…) on the Xenium cells never reads a run's labels ───────────
def test_subset_cell_type_ignores_labels_of_a_segmentation_run(xe, D):
    from xenium_app import core
    ids = D["df"].index.astype(str)
    mine = pd.Series(["Keep" if i % 2 == 0 else "Other" for i in range(len(ids))], index=ids)
    with core._annot_lock:
        for k in [k for k in core._annot_state if str(k).startswith("labels_")]:
            core._annot_state[k] = None
        core._annot_state["labels_celltypist_proseg_abcd1234"] = pd.Series("Keep", index=["1", "2", "3"])
        core._annot_state["labels_seurat"] = mine
    try:
        xe.subset(cell_type="Keep")
        assert len(D["df"]) == int((mine == "Keep").sum()) > 0
    finally:
        with core._annot_lock:
            core._annot_state.pop("labels_celltypist_proseg_abcd1234", None)


# ── 19: SPLIT's result is paired with the run's current cells by id ──────────────────
def test_split_result_is_matched_to_a_changed_cell_table_by_id(xe):
    from xenium_app import split
    job = pd.DataFrame({"cluster_split_10": ["a", "b", "c", "d"], "split_umap_1": [1.0, 2.0, 3.0, 4.0]},
                       index=["c1", "c2", "c3", "c4"])
    mat = sp.csr_matrix(np.array([[1, 0], [0, 2], [3, 0], [0, 4]], dtype=np.float32))
    same, cols = split._split_onto_cells(mat, job, job.copy())
    assert same is mat and list(cols["cluster_split_10"]) == ["a", "b", "c", "d"]
    cur = pd.DataFrame(index=["c4", "c2", "c9"])              # same length is not the same cells
    out, cols = split._split_onto_cells(mat, job, cur)
    assert out.shape == (3, 2) and np.array_equal(out.toarray(), [[0, 4], [0, 2], [0, 0]])
    assert list(cols["cluster_split_10"][:2]) == ["d", "b"] and pd.isna(cols["cluster_split_10"][2])
    assert np.allclose(cols["split_umap_1"][:2], [4.0, 2.0]) and np.isnan(cols["split_umap_1"][2])
    dup = pd.DataFrame({"cluster_split_10": ["a", "b"]}, index=["c1", "c1"])
    with pytest.raises(RuntimeError, match="cannot be matched"):
        split._split_onto_cells(mat[:2], dup, cur)
    import inspect
    assert "_split_onto_cells(corrected_mat, cells_df, _alt_res[\"cells_df\"])" in inspect.getsource(
        split._run_split_correction)


# ── 20: the gene filter of subset() on a run uses the run's own gene index ───────────
def test_subset_gene_filter_on_a_run_uses_the_runs_gene_index(xe, D):
    from xenium_app import subsetting
    genes = list(D["gene_names"])
    res = _fake_run(xe, D)
    # the run's matrix has its own column order: the first two genes are swapped
    gni = dict(res["gene_name_to_idx"])
    gni[genes[0]], gni[genes[1]] = 1, 0
    res["gene_name_to_idx"] = gni
    dense = res["expr"].toarray()
    dense[:, 0] = 0.0                                         # column 0 (gene 1 of the panel): empty
    dense[:, 1] = np.arange(len(dense)) % 2 * 5.0             # column 1 = gene 0: every second cell
    res["expr"] = sp.csr_matrix(dense)
    assert subsetting._subset_reseg_gene_col(genes[0], res, 10 ** 6) == 1
    assert subsetting._subset_reseg_gene_col(genes[0] + " [imp]", res, 10 ** 6) == 1
    assert subsetting._subset_reseg_gene_col(genes[0], res, 1) is None       # beyond the run's columns
    assert subsetting._subset_reseg_gene_col("NOPE", res, 10 ** 6) is None
    want = list(res["cells_df"].index[np.arange(len(dense)) % 2 == 1])
    xe.subset(gene=genes[0], min_expr=0.5)
    assert list(xe._proseg_state["result"]["cells_df"].index) == want


# ── 21: every polygon loader takes the largest part of a multi-part cell ─────────────
def test_geojson_outline_of_a_multipolygon_is_one_ring(xe):
    from xenium_app import reseg
    small = [[0, 0], [1, 0], [1, 1], [0, 0]]
    big = [[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]]
    assert reseg._geojson_ring({"type": "Polygon", "coordinates": [big, small]}) == big
    ring = reseg._geojson_ring({"type": "MultiPolygon", "coordinates": [[small], [big]]})
    assert ring == big and all(len(p) == 2 and np.isscalar(p[0]) for p in ring)
    assert reseg._geojson_ring({"type": "MultiPolygon", "coordinates": []}) == []
    assert reseg._geojson_ring({}) == [] and reseg._geojson_ring(None) == []
    import inspect
    src = inspect.getsource(reseg)
    assert '.get("coordinates", [[]])[0]' not in src, "a loader still takes coordinates[0] as the ring"


# ── 23: the reference path is normalised once and that value is given to the job ─────
def test_reference_path_is_stripped_before_the_job_gets_it(xe, tmp_path, monkeypatch):
    from xenium_app import core, jobs
    ref = tmp_path / "ref.rds"
    ref.write_bytes(b"x")
    assert core._ref_rds_path(f"  {ref}\n") == str(ref) and core._ref_rds_path(None) == ""
    started = []
    monkeypatch.setattr(jobs, "_job_start", lambda kind, label, target, args=(), **kw: started.append((kind, args)))
    with core._split_lock:
        core._split_state["status"] = "idle"
    out = xe.run_split(1, f"{ref} \n", "Names", 2, 10, 100, True, "xenium")
    assert started and started[-1][0] == "split" and started[-1][1][0] == str(ref), (out, started)
    import inspect
    from xenium_app import cb_tools
    assert "rds_path = core._ref_rds_path(rds_path)" in inspect.getsource(cb_tools.start_annotation)


# ── the glue (assets/xenium_viewer_dash.js), run in node with a stand-in page ────────
def _glue(scenario):
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    here = os.path.dirname(os.path.abspath(__file__))
    out = subprocess.run([node, os.path.join(here, "glue_harness.js"),
                          os.path.join(os.path.dirname(here), "assets", "xenium_viewer_dash.js"), scenario],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


# ── 24: a joint re-run makes the WebGL UMAP fetch the coordinates again ──────────────
def test_webgl_joint_umap_refetches_coordinates_after_a_rerun():
    got = _glue("joint_rerun")
    assert got["after"] and got["after"] != got["before"], got     # new cells key → cells.bin refetched
    assert got["color"] and got["color"].endswith("|2")
    assert got["plain"] is False            # another cluster set: colours only, as before


# ── 25: a click in the joint UMAP names the sample of the cell ───────────────────────
def test_webgl_joint_umap_click_keeps_the_sample(xe, D, world):
    assert _glue("umap_click")["selected"] == [{"id": "cellX", "sample": 1}, "cellY"]
    # ... and the server answers with the sample's position, which show_cell_info resolves
    from xenium_app import viewer_provider
    embedded = [np.arange(len(d["df"])) for d in world.dsets]
    key, tables, _ids, umap, _code = _fake_joint_set(xe, world, embedded)
    for t, c in zip(tables, np.split(umap, np.cumsum([len(t["df"]) for t in tables])[:-1])):
        xe._ctl_put(t, "xenium", {"joint_umap_1": c[:, 0], "joint_umap_2": c[:, 1]})
    prov = viewer_provider._UmapViewerProvider()
    n0 = len(tables[0]["df"])
    got = prov.cell_at(f"xenium|emb:{key}", n0 + 1)
    assert got["sample"] == 1 and got["id"] == str(tables[1]["df"].index[1])
    assert prov.cell_at(f"xenium|emb:{key}", 0)["sample"] == 0


# ── 26: a finished compartments / spillover run makes the map recolour ───────────────
def test_controls_rerun_bumps_the_colour_version(xe):
    assert xe.recolour_after_controls(3, None, 7) == 8 and xe.recolour_after_controls(None, 1, None) == 1
    cb = [v for k, v in xe.app.callback_map.items() if "geneset-version" in k
          and {i["id"] for i in v["inputs"]} == {"ctl-comp-done", "spill-done"}]
    assert len(cb) == 1, "no callback bumps geneset-version when a control job finishes"


# ── 27: the lasso count is the number of cells the outline holds ─────────────────────
def test_webgl_lasso_count_is_the_outlines_not_the_visible_cells(xe, D):
    df = D["df"]
    x, y = df["x_centroid"].to_numpy(float), df["y_centroid"].to_numpy(float)
    pad = 5.0
    poly = [[x.min() - pad, y.min() - pad], [x.max() + pad, y.min() - pad],
            [x.max() + pad, y.max() + pad], [x.min() - pad, y.max() + pad]]      # viewer space: y down
    # the viewer counted 2 cells (the others belong to legend-hidden categories)
    pts, is_open, preview, err = xe.capture_webgl_selection({"polygon": poly, "count": 2}, "xenium")
    assert err == "" and is_open and pts, err
    assert preview.startswith(f"{len(df):,} cells selected")
    xe.de_capture_webgl_selection({"polygon": poly, "count": 2}, 0, "xenium")
    assert xe._de_state["selection"]["n"] == len(df)
    # an outline around nothing is still refused, whatever the viewer counted
    far = [[x.max() + 1e4, y.max() + 1e4], [x.max() + 1e4 + 5, y.max() + 1e4], [x.max() + 1e4 + 5, y.max() + 1e4 + 5]]
    assert "at least 3 cells" in xe.capture_webgl_selection({"polygon": far, "count": 500}, "xenium")[3]


# ── 28: a sample without labels has no aggregate density (not 0) ─────────────────────
def test_aggregates_unlabelled_sample_has_no_value_and_blocks_the_test(xe, monkeypatch):
    from test_aggregates import _fake_view, FAST
    tables, labs = _fake_view(xe, monkeypatch, 6, n=500)
    labs = list(labs)
    no_class = np.array(["Other"] * len(labs[4]), dtype=object)      # labelled, no cell of the class
    labs[4], labs[5] = no_class, None                                # … and no labels at all
    monkeypatch.setattr(xe, "_comp_labels", lambda *a, **k: (labs, "cell type (test)"))
    xe._design_set([{"key": t["key"], "cond": "A" if i < 3 else "B", "batch": "", "donor": ""}
                    for i, t in enumerate(tables)])
    try:
        xe._run_aggregates("xenium", None, "celltype:celltypist", ["Imm"], FAST)
        st = xe._agg_state
        assert st["status"] == "done", st["message"]
        s = st["result"]["samples"].set_index("sample")
        assert s.loc["S4", "per_mm2"] == 0.0 and s.loc["S4", "n_aggregates"] == 0     # a measured zero
        assert np.isnan(s.loc["S5", "per_mm2"]) and np.isnan(s.loc["S5", "area_fraction"])
        assert np.isnan(s.loc["S5", "n_aggregates"])
        tt = st["result"]["tests"]["per_mm2"]
        assert tt["p"] is None and "no labels" in tt["why"]
        assert "S5: no cell type (test) labels - not evaluated" in st["message"]
        assert st["result"]["unlabelled"] == ["S5"]
    finally:
        xe._stats_reset()


# ── 30: the padj column is headed by the adjustment that produced it ─────────────────
def test_proximity_padj_column_is_named_after_the_adjustment_used(xe):
    from xenium_app import proximity
    import inspect
    head = lambda fdr: dict((c, n) for c, n, _k in proximity._prox_columns(fdr))["padj"]
    assert head("empirical") == "Emp. FDR" and head("bh") == "FDR (BH)" and head("none") == "FDR"
    assert [c for c, _n, _k in proximity._prox_columns("bh")] == [c for c, _n, _k in proximity._PROX_COLUMNS]
    assert '"columns": _prox_columns(info["fdr"])' in inspect.getsource(proximity._run_proximity_de)
