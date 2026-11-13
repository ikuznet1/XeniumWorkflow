"""AnnData .h5ad input (xenium_app/h5ad.py): one file, one or several sections.

Every file here is fabricated in a temporary directory. The conversion is checked against
the AnnData object the file was written from, cell by cell (ids, coordinates, labels and
counts), for the three on-disk encodings of the matrix and for sections stored in runs or
interleaved; the app is then started from a file in its own process (the dataset is fixed
at import) to check the samples, the prefilled design, the cell types and what is
unavailable without transcripts and images."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import xe_testlib as tl

ad = pytest.importorskip("anndata")

SECTIONS = ["sec_10", "sec_2", "sec_1", "sec_3", "sec_4", "sec_5"]
COND = {"sec_1": "ctrl", "sec_2": "ctrl", "sec_3": "treated", "sec_4": "treated", "sec_5": "other",
        "sec_10": "treated"}
DONOR = {"sec_1": "d1", "sec_2": "d2", "sec_3": "d1", "sec_4": "d2", "sec_5": "d3", "sec_10": "d3"}


def make_adata(n=900, g=30, seed=0, interleaved=True):
    """Counts in layers["counts"], log-normalised values in X, sections interleaved or in runs."""
    rng = np.random.default_rng(seed)
    sec = rng.choice(SECTIONS, size=n)
    if not interleaved:
        sec = np.sort(sec)
    counts = sp.random(n, g, density=0.3, random_state=seed, format="csr")
    counts.data = np.ceil(counts.data * 6)
    obs = pd.DataFrame({
        "section": pd.Categorical(sec),
        "x": rng.uniform(0, 900, n), "y": rng.uniform(0, 700, n),
        "cell_type": pd.Categorical(rng.choice(["T", "B", "Mono", "Stroma"], size=n)),
        "fine": rng.choice(["a", "b", "c"], size=n),
        "condition": [COND[s] for s in sec], "donor": [DONOR[s] for s in sec],
        "leiden": pd.Categorical(rng.integers(0, 5, n).astype(str)),
    }, index=[f"c{i}" for i in range(n)])
    a = ad.AnnData(X=sp.csr_matrix(np.log1p(counts.toarray())), obs=obs,
                   var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    a.layers["counts"] = counts
    a.obsm["X_umap"] = rng.normal(size=(n, 2)).astype("float32")
    return a


MAPPING = {"x": "x", "y": "y", "section": "section", "cell_types": ["cell_type", "fine"],
           "condition": "condition", "donor": "donor", "counts": "counts", "clusters": ["leiden"],
           "umap": "X_umap"}


def _read_store(path):
    import spatialdata
    return spatialdata.read_zarr(str(path)).tables["table"]


def _check_against(a, man, cache_dir, counts):
    """Every cell of *a* is in exactly one store, with its own coordinates, labels and counts."""
    seen = []
    for s in man["sections"]:
        t = _read_store(f"{cache_dir}/{s['store']}")
        ids = list(t.obs_names)
        ref = a[ids]
        assert set(ref.obs["section"].astype(str)) == {s["name"]}
        assert len(ids) == s["n_cells"] == int((a.obs["section"].astype(str) == s["name"]).sum())
        assert np.array_equal(t.obs["x_centroid"].to_numpy(), ref.obs["x"].to_numpy())
        assert np.array_equal(t.obs["y_centroid"].to_numpy(), ref.obs["y"].to_numpy())
        assert list(t.obs["cell_type_obs0"].astype(str)) == list(ref.obs["cell_type"].astype(str))
        assert list(t.obs["cell_type_obs1"].astype(str)) == list(ref.obs["fine"].astype(str))
        assert np.array_equal(t.obs["source_row"].to_numpy(), a.obs_names.get_indexer(ids))
        want = counts[a.obs_names.get_indexer(ids)]
        assert (sp.csr_matrix(t.X) != sp.csr_matrix(want).astype(np.float32)).nnz == 0
        assert np.allclose(t.obs["transcript_counts"].to_numpy(), np.asarray(want.sum(axis=1)).ravel())
        assert np.allclose(t.obs[["umap_1", "umap_2"]].to_numpy(), a.obsm["X_umap"][a.obs_names.get_indexer(ids)])
        assert list(t.var_names) == list(a.var_names)
        seen += ids
    assert sorted(seen) == sorted(a.obs_names)


@pytest.mark.parametrize("interleaved", [True, False])
def test_sections_are_split_with_ids_coordinates_labels_and_counts_aligned(xe, tmp_path, interleaved):
    a = make_adata(interleaved=interleaved)
    path = tmp_path / "data.h5ad"
    a.write_h5ad(path)
    # small blocks: sections span several blocks, blocks span several sections
    man, cache_dir = xe._h5ad_convert(str(path), MAPPING, block_nnz=700)
    assert [s["name"] for s in man["sections"]] == ["sec_1", "sec_2", "sec_3", "sec_4", "sec_5", "sec_10"]
    assert man["n_cells"] == a.n_obs and man["n_genes"] == a.n_vars
    assert man["counts"] == {"matrix": "counts", "integer": True, "nnz": int(a.layers["counts"].nnz),
                             "n_noninteger": 0, "n_negative": 0, "max": float(a.layers["counts"].max())}
    assert [(s["condition"], s["donor"]) for s in man["sections"]] == \
        [(COND[s["name"]], DONOR[s["name"]]) for s in man["sections"]]
    assert man["celltype_methods"] == {"obs0": "cell_type", "obs1": "fine"}
    _check_against(a, man, cache_dir, a.layers["counts"].tocsr())
    # cluster ids are the same in every section, 0 kept for "no cluster"
    lut = man["cluster_levels"]["leiden"]
    for s in man["sections"]:
        t = _read_store(f"{cache_dir}/{s['store']}")
        assert list(t.uns["cluster_methods"]) == ["leiden"]
        assert [lut[v] for v in a[list(t.obs_names)].obs["leiden"].astype(str)] == list(t.obs["clust__leiden"])
    assert 0 not in lut.values()
    # a second call reuses the stores (same file, same mapping) …
    again, again_dir = xe._h5ad_convert(str(path), dict(MAPPING, cond_a="ctrl", cond_b="treated"))
    assert again_dir == cache_dir and again["seconds"] == man["seconds"]
    # … another mapping is another conversion
    assert xe._h5ad_cache_dir(str(path), dict(MAPPING, counts="X")) != cache_dir


@pytest.mark.parametrize("encoding", ["csc", "dense"])
def test_csc_and_dense_matrices_give_the_same_stores(xe, tmp_path, encoding):
    a = make_adata(n=400, g=12, seed=3)
    counts = a.layers["counts"].tocsr()
    a.layers["counts"] = counts.tocsc() if encoding == "csc" else counts.toarray()
    path = tmp_path / f"{encoding}.h5ad"
    a.write_h5ad(path)
    info = xe._h5ad_inspect(str(path))
    assert info["matrices"]["counts"]["encoding"] == encoding
    man, cache_dir = xe._h5ad_convert(str(path), MAPPING, block_nnz=500)
    assert man["counts"]["integer"]
    _check_against(a, man, cache_dir, counts)


def test_inspect_lists_columns_matrices_and_guesses_a_mapping(xe, tmp_path):
    a = make_adata(n=200, g=8)
    path = tmp_path / "data.h5ad"
    a.write_h5ad(path)
    info = xe._h5ad_inspect(str(path))
    assert (info["n_cells"], info["n_genes"]) == (200, 8)
    cols = {c["name"]: c for c in info["obs_columns"]}
    assert cols["x"]["kind"] == "numeric" and cols["cell_type"]["kind"] == "categorical"
    assert sorted(cols["condition"]["levels"]) == ["ctrl", "other", "treated"]
    assert set(info["matrices"]) == {"X", "counts"} and info["obsm"] == {"X_umap": [200, 2]}
    g = info["guess"]
    assert (g["x"], g["y"], g["section"], g["condition"], g["donor"], g["counts"], g["umap"]) == \
        ("x", "y", "section", "condition", "donor", "counts", "X_umap")
    assert g["cell_types"] == ["cell_type"]
    est = xe._h5ad_estimate(info, xe._h5ad_clean_mapping(g))
    assert est["nnz"] == a.layers["counts"].nnz and est["ram_gb"] > 0


def test_mapping_from_a_sidecar_json_and_the_environment(xe, tmp_path, monkeypatch):
    a = make_adata(n=200, g=8)
    path = tmp_path / "data.h5ad"
    a.write_h5ad(path)
    for var in xe._H5AD_ENV.values():
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("XENIUM_H5AD_MAPPING", raising=False)
    (tmp_path / "data.h5ad.json").write_text(json.dumps({"cell_types": ["fine"], "cond_a": "ctrl", "cond_b": "treated"}))
    m = xe._h5ad_mapping_for(str(path))
    assert m["cell_types"] == ["fine"] and (m["cond_a"], m["cond_b"]) == ("ctrl", "treated") and m["x"] == "x"
    monkeypatch.setenv("XENIUM_H5AD_CELLTYPES", "cell_type, fine")
    monkeypatch.setenv("XENIUM_H5AD_COUNTS", "X")
    monkeypatch.setenv("XENIUM_H5AD_SECTION", "")
    m = xe._h5ad_mapping_for(str(path))
    assert m["cell_types"] == ["cell_type", "fine"] and m["counts"] == "X" and m["section"] is None
    other = tmp_path / "elsewhere.json"
    other.write_text(json.dumps({"donor": None}))
    monkeypatch.setenv("XENIUM_H5AD_MAPPING", str(other))
    assert xe._h5ad_mapping_for(str(path))["donor"] is None
    monkeypatch.setenv("XENIUM_H5AD_MAPPING", str(tmp_path / "missing.json"))
    with pytest.raises(ValueError, match="does not exist"):
        xe._h5ad_mapping_for(str(path))


def test_a_mapping_that_does_not_fit_the_file_is_refused_with_the_reason(xe, tmp_path):
    a = make_adata(n=120, g=6)
    path = tmp_path / "data.h5ad"
    a.write_h5ad(path)
    for bad, msg in (({"x": "nope"}, "x coordinate: obs has no column 'nope'"),
                     ({"x": "cell_type"}, "is not numeric"),
                     ({"x": None, "y": None}, "No coordinates"),
                     ({"section": "nope"}, "section: obs has no column"),
                     ({"cell_types": ["nope"]}, "cell_types: obs has no column"),
                     ({"counts": "raw"}, "no matrix 'raw'"),
                     ({"umap": "X_tsne"}, "umap: obsm has no")):
        with pytest.raises(ValueError, match=msg):
            xe._h5ad_convert(str(path), dict(MAPPING, **bad))
    with pytest.raises(ValueError, match="noninteger_counts"):
        xe._h5ad_clean_mapping(dict(MAPPING, noninteger_counts="whatever"))


def test_one_sample_without_a_section_column_and_cells_without_coordinates(xe, tmp_path):
    a = make_adata(n=150, g=6)
    a.obs.loc[a.obs_names[:4], "x"] = np.nan
    path = tmp_path / "single.h5ad"
    a.write_h5ad(path)
    man, cache_dir = xe._h5ad_convert(str(path), dict(MAPPING, section=None, condition=None, donor=None))
    assert [s["name"] for s in man["sections"]] == ["single"] and man["n_cells"] == 146
    assert man["n_dropped_no_coordinates"] == 4
    assert any("4 cells without finite x / y" in w for w in xe._h5ad_warnings(man))
    t = _read_store(f"{cache_dir}/{man['sections'][0]['store']}")
    assert not set(a.obs_names[:4]) & set(t.obs_names)
    # a section whose cells all lack coordinates gives no sample, and the load says so
    b = make_adata(n=150, g=6, seed=2)
    b.obs.loc[b.obs["section"] == "sec_3", "y"] = np.nan
    other = tmp_path / "hole.h5ad"
    b.write_h5ad(other)
    man2, _ = xe._h5ad_convert(str(other), MAPPING)
    assert "sec_3" not in [s["name"] for s in man2["sections"]] and len(man2["sections"]) == 5
    assert man2["sections_without_cells"] == ["sec_3"]
    assert man2["n_cells"] == int((b.obs["section"] != "sec_3").sum())
    assert any("1 section(s) have no cell with coordinates" in w for w in xe._h5ad_warnings(man2))


def test_a_matrix_that_is_not_integer_valued_is_flagged(xe, tmp_path):
    a = make_adata(n=150, g=6)
    path = tmp_path / "lognorm.h5ad"
    a.write_h5ad(path)
    man, _ = xe._h5ad_convert(str(path), dict(MAPPING, counts="X"))       # X holds log1p values
    c = man["counts"]
    assert c["integer"] is False and c["n_noninteger"] > 0 and c["nnz"] == a.X.nnz
    (w,) = [w for w in xe._h5ad_warnings(man) if "does not hold raw counts" in w]
    assert f"{c['n_noninteger']:,} of {c['nnz']:,} stored values are not whole numbers" in w
    assert "refuses this matrix" in w
    assert "NOT raw counts" in xe._h5ad_summary(man)
    ok, _ = xe._h5ad_convert(str(path), MAPPING)
    assert ok["counts"]["integer"] and "integer-valued" in xe._h5ad_summary(ok)
    assert not [w for w in xe._h5ad_warnings(ok) if "raw counts" in w]


def test_pseudobulk_sums_refuse_non_integer_counts_unless_the_mapping_allows_rounding(xe):
    m = sp.csc_matrix(np.array([[1.0, 0.4], [2.0, 0.4], [0.0, 0.4]]))
    s = {"label": "S0", "csc": m, "rows": np.arange(3), "ds": {"h5ad": {"section": "a"}}}
    assert not xe._counts_are_integer(s)
    with pytest.raises(ValueError, match="non-integer values.*noninteger_counts"):
        xe._pb_sum_counts(s, np.ones(3, dtype=bool))
    s["ds"]["counts_noninteger"] = "round_sums"
    tot, n = xe._pb_sum_counts(s, np.ones(3, dtype=bool))
    assert tot.tolist() == [3, 1] and n == 3                 # 0.4 × 3 = 1.2 → 1: the SUM is rounded
    whole = {"label": "S1", "csc": sp.csc_matrix(np.array([[1.0, 0.0], [2.0, 5.0]])), "rows": np.arange(2), "ds": {}}
    assert xe._counts_are_integer(whole) and xe._pb_sum_counts(whole, np.ones(2, dtype=bool))[0].tolist() == [3, 5]
    # a Xenium sample with non-integer values (never the case for raw counts) is refused as before
    plain = {"label": "S2", "csc": m, "rows": np.arange(3), "ds": {}}
    with pytest.raises(ValueError, match="needs raw counts\\.$"):
        xe._pb_sum_counts(plain, np.ones(3, dtype=bool))


def test_what_is_unavailable_is_listed_with_reasons(xe):
    feats = dict(xe._h5ad_unavailable())
    for f in ("Transcript density", "Segmentation-free abundance", "Transcript factors",
              "Baysor / Proseg resegmentation", "Image-based resegmentation", "Morphology image overlay"):
        assert f in feats and feats[f]
    note = xe._h5ad_unavailable_note()
    assert "no transcripts, no images" in note and "Transcript factors (needs the transcript file)" in note


APP_PROBE = r'''
import numpy as np, pandas as pd
RESULT["names"] = [xe._ds_name(d) for d in [xe.DATA] + list(xe.EXTRA_DATASETS)]
RESULT["n_cells"] = [int(len(d["df"])) for d in [xe.DATA] + list(xe.EXTRA_DATASETS)]
RESULT["offsets_increase"] = bool(np.all(np.diff([d.get("x_offset", 0.0) for d in [xe.DATA] + list(xe.EXTRA_DATASETS)]) > 0))
RESULT["design"] = [(r["name"], r["cond"], r["donor"]) for r in xe._design_samples()]
RESULT["design_names"] = xe._design_names()
RESULT["methods"] = {k: v for k, v in xe._ANNOT_METHODS.items() if k.startswith("obs")}
samples = xe._an_samples("xenium")
RESULT["n_samples"] = len(samples)
labs = xe._an_celltype_labels(samples, "obs0", None)
RESULT["labels_ok"] = all(
    list(l) == list(s["df"]["cell_type_obs0"]) for l, s in zip(labs, samples))
RESULT["label_levels"] = sorted({x for l in labs for x in l})
opts = xe._refresh_colorby_options([{"label": "Cluster", "value": "cluster"},
                                    {"label": "Gene Expression", "value": "gene"},
                                    {"label": "Transcript Counts", "value": "transcript_counts"}])
RESULT["colorby"] = [(o["value"], bool(o.get("disabled"))) for o in opts if o["value"].startswith("cell_type:obs")]
RESULT["colorby_at"] = [o["value"] for o in opts].index("cell_type:obs0")
_cb = [c for c in xe.app.layout._traverse() if getattr(c, "id", None) == "color-by"][0]
RESULT["layout_colorby"] = [o["value"] for o in _cb.options if o["value"].startswith("cell_type:obs")]
# the map in both renderers, by cell type and by gene
for cb in ("cell_type:obs0", "gene", "cluster"):
    fig = xe.make_spatial_fig(cb, "leiden", "G1", 3, 0.8, [], {})
    RESULT["plotly_" + cb] = int(sum(len(t.x) for t in fig.data if getattr(t, "x", None) is not None))
client = xe.app.server.test_client()
RESULT["viewer_samples"] = int(client.get("/viewer/meta").get_json().get("samples", 0))
RESULT["viewer_meta_keys"] = sorted(client.get("/viewer/meta").get_json())
# sample-level tests on the prefilled design
conds, batches = xe._design_of(samples)
donors = xe._design_donors(samples)
masks = [np.ones(len(s["df"]), dtype=bool) for s in samples]
prep = xe._pb_prepare(samples, masks, conds, batches, donors=donors)
RESULT["pb_design"] = prep["design"]; RESULT["pb_units"] = prep["units_note"]; RESULT["pb_cols"] = len(prep["labels"])
comp = xe._comp_compute("celltype:obs0", "xenium", None, "logit", "exact")
RESULT["comp_shape"] = list(comp["counts"].shape)
RESULT["comp_note"] = comp.get("test_note")
# what needs transcripts or images says so instead of failing
RESULT["density"] = xe.run_density(1, "G1")[2]
RESULT["morph"] = xe.update_morph_enable_state(0)[0][0]["disabled"]
RESULT["info_rows"] = [str(getattr(c, "children", "")) for c in xe._h5ad_info_rows()]
RESULT["summary"] = xe._h5ad_summary()
RESULT["renderer"] = xe._DEFAULT_RENDERER
RESULT["too_large"] = xe._plotly_too_large()
RESULT["limit"] = xe.PLOTLY_MAX_CELLS
for name, call in (("txf_scan", lambda: xe._txf_counts(xe.EXTRA_DATASETS[0], ["G1"], 20.0, "square")),
                   ("sfree_scan", lambda: xe._sfree_ensure_grids(xe.DATA, ["G1"]))):
    try:
        call()
        RESULT[name] = "no error"
    except Exception as exc:
        RESULT[name] = f"{type(exc).__name__}: {exc}"
# another pair of levels from the same file
RESULT["redesign"] = xe._h5ad_design("treated", "other")
RESULT["design2"] = [(r["name"], r["cond"]) for r in xe._design_samples()]
RESULT["bad_design"] = xe._h5ad_design("ctrl", "nope")
'''


@pytest.fixture(scope="module")
def app_from_h5ad(tmp_path_factory):
    root = tmp_path_factory.mktemp("h5ad_app")
    a = make_adata(n=1500, g=30, seed=5)
    path = root / "study.h5ad"
    a.write_h5ad(path)
    (root / "study.h5ad.json").write_text(json.dumps(dict(MAPPING, cond_a="ctrl", cond_b="treated")))
    (root / "cache").mkdir()
    # 1,500 cells over a limit of 1,000: the app must start in the WebGL view
    res = tl.run_in_fresh_app(APP_PROBE, path, root / "cache", env={"XENIUM_PLOTLY_MAX_CELLS": "1000"})
    return a, res


def test_app_started_from_an_h5ad_has_one_sample_per_section(app_from_h5ad):
    a, r = app_from_h5ad
    order = ["sec_1", "sec_2", "sec_3", "sec_4", "sec_5", "sec_10"]
    assert r["names"] == order and r["n_samples"] == 6 and r["offsets_increase"]
    assert r["n_cells"] == [int((a.obs["section"] == s).sum()) for s in order]
    assert a.n_obs == sum(r["n_cells"])
    assert r["plotly_cell_type:obs0"] >= a.n_obs and r["plotly_gene"] >= a.n_obs and r["plotly_cluster"] >= a.n_obs


def test_design_is_prefilled_from_the_condition_and_donor_columns(app_from_h5ad):
    _a, r = app_from_h5ad
    want = {"ctrl": "A", "treated": "B", "other": ""}
    assert r["design"] == [[s, want[COND[s]], DONOR[s]] for s in r["names"]]
    assert r["design_names"] == {"A": "ctrl", "B": "treated"}
    # d1 and d2 have a section in each condition: paired; d3 (treated only) is left out
    assert r["pb_design"] == "~ donor + condition" and r["pb_cols"] == 4
    assert "2 donors, paired" in r["pb_units"]
    assert r["comp_shape"] == [4, 6] and r["comp_note"]
    assert "A = treated: 3 section(s) of 3 donor(s), B = other: 1 section(s) of 1 donor(s)" in r["redesign"]
    assert dict(map(tuple, r["design2"])) == {s: {"treated": "A", "other": "B", "ctrl": ""}[COND[s]] for s in r["names"]}
    assert "are not two levels of the condition column" in r["bad_design"]


def test_cell_type_columns_become_colour_and_grouping_options(app_from_h5ad):
    _a, r = app_from_h5ad
    assert r["methods"] == {"obs0": "cell_type", "obs1": "fine"}
    assert r["labels_ok"] and r["label_levels"] == ["B", "Mono", "Stroma", "T"]
    assert r["colorby"] == [["cell_type:obs0", False], ["cell_type:obs1", False]] and r["colorby_at"] == 2
    assert r["layout_colorby"] == ["cell_type:obs0", "cell_type:obs1"]


def test_transcript_and_image_features_are_unavailable_with_a_reason(app_from_h5ad):
    _a, r = app_from_h5ad
    assert "needs the Xenium output directory" in r["density"] and "loaded from an .h5ad (cells only)" in r["density"]
    for k in ("txf_scan", "sfree_scan"):
        assert r[k] == "ValueError: This sample was loaded from an .h5ad (cells only): it has no transcript file."
    assert r["morph"] is True
    assert any("cells only" in row and "no transcripts" in row for row in r["info_rows"])
    assert "Not available for data loaded from an .h5ad" in r["summary"]
    assert "integer-valued" in r["summary"]


def test_more_cells_than_plotly_draws_start_in_the_webgl_view(app_from_h5ad, xe):
    a, r = app_from_h5ad
    assert r["limit"] == 1000 and r["renderer"] == "webgl"
    assert f"{a.n_obs:,} cells are too many for the Plotly renderer (limit 1,000)" in r["too_large"]
    assert "WebGL view" in r["too_large"]
    # the rule itself (the default limit is far above the test bundle)
    assert xe._plotly_too_large(xe.PLOTLY_MAX_CELLS) is None
    assert "too many for the Plotly renderer" in xe._plotly_too_large(xe.PLOTLY_MAX_CELLS + 1)
    assert xe._plotly_too_large() is None and xe._DEFAULT_RENDERER in ("plotly", "webgl")
