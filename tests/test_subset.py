"""subset() / unsubset(): which cells stay, and that every per-cell consumer (expression
lookup, figures, the viewer endpoints) stays aligned to the filtered table.

The classic failure here is silent misalignment: a value array computed for the full
table shown against the filtered one. So each check recomputes the expected values for
the expected cells from the raw files and compares by cell id.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import xe_testlib as tl


def _gene_values(raw, gene):
    counts, names, barcodes = raw.gene_counts
    col = np.asarray(counts[:, names.index(gene)].todense()).ravel().astype(np.float64)
    return pd.Series(np.log1p(col), index=barcodes)


def _assert_consistent(xe, D, client, raw, expected_ids, gene):
    """DATA, expression, figure and viewer all describe exactly *expected_ids*, in order."""
    expected_ids = list(expected_ids)
    df = D["df"]
    assert list(df.index.astype(str)) == expected_ids, "subset keeps the original row order"
    np.testing.assert_array_equal(np.asarray(D["df_to_expr"]), raw.cells.index.get_indexer(expected_ids))
    want = _gene_values(raw, gene).loc[expected_ids].to_numpy()
    np.testing.assert_allclose(xe.get_gene_expression(gene), want, rtol=1e-6)
    got = xe._get_expr_values(gene)
    np.testing.assert_allclose(got, want, rtol=1e-6)
    method = D["cluster_methods"][0] if D["cluster_methods"] else None
    fig = tl.fig_cells(xe.make_spatial_fig("gene", method, gene, 3, 0.8, [], None))
    assert list(fig.index) == expected_ids
    np.testing.assert_allclose(fig["color"].to_numpy(dtype=float), want, rtol=1e-5)
    np.testing.assert_allclose(fig["x"], raw.cells.loc[expected_ids, "x_centroid"], atol=1e-4)
    h, a = tl.get_bin(client, "/viewer/cells.bin")
    assert h["n"] == len(expected_ids)
    np.testing.assert_array_equal(a["xy"][0::2], raw.cells.loc[expected_ids, "x_centroid"].to_numpy(np.float32))
    np.testing.assert_array_equal(a["xy"][1::2], raw.cells.loc[expected_ids, "y_centroid"].to_numpy(np.float32))
    _, c = tl.get_bin(client, f"/viewer/colors.bin?mode=gene&gene={gene}")
    np.testing.assert_allclose(c["values"], want.astype(np.float32), rtol=1e-6)
    if expected_ids:
        assert client.get("/viewer/cell/0").get_json()["id"] == expected_ids[0]
    return h["token"]


def _median(series):
    return float(series.median())


def test_subset_by_transcript_counts(xe, D, client, raw, expressed_gene):
    c = raw.cells
    lo = _median(c["transcript_counts"])
    out = xe.subset(min_transcripts=lo)
    want = c.index[c["transcript_counts"] >= lo]
    assert 0 < len(want) < len(c)
    assert list(out.index.astype(str)) == list(want)
    _assert_consistent(xe, D, client, raw, want, expressed_gene)


def test_subset_filters_are_anded(xe, D, client, raw, expressed_gene):
    c = raw.cells
    if "cell_area" not in c.columns or "nucleus_area" not in c.columns:
        pytest.skip("bundle has no area columns")
    lo_t, hi_a, lo_n = c["transcript_counts"].quantile(0.25), c["cell_area"].quantile(0.8), c["nucleus_area"].quantile(0.1)
    xe.subset(min_transcripts=lo_t, max_cell_area=hi_a, min_nucleus_area=lo_n)
    want = c.index[(c["transcript_counts"] >= lo_t) & (c["cell_area"] <= hi_a) & (c["nucleus_area"] >= lo_n)]
    assert 0 < len(want) < len(c)
    _assert_consistent(xe, D, client, raw, want, expressed_gene)


def test_subset_by_cluster_uses_the_chosen_method(xe, D, client, raw, expressed_gene):
    if len(D["cluster_methods"]) < 1:
        pytest.skip("bundle has no clustering")
    method = D["cluster_methods"][-1]
    truth = raw.clusters(method)
    if truth is None:
        pytest.skip("no clusters.csv for the method")
    keep = [int(truth.value_counts().index[0])]
    xe.subset(cluster=keep[0], method=method)
    want = [i for i in raw.cells.index if i in truth.index and truth[i] in keep]
    _assert_consistent(xe, D, client, raw, want, expressed_gene)
    both = [int(v) for v in truth.value_counts().index[:2]]
    xe.subset(cluster=both, method=method)
    want = [i for i in raw.cells.index if i in truth.index and truth[i] in both]
    _assert_consistent(xe, D, client, raw, want, expressed_gene)


def test_subset_by_gene_expression_thresholds(xe, D, client, raw, expressed_gene):
    vals = _gene_values(raw, expressed_gene)
    lo = float(np.median(vals[vals > 0]))
    xe.subset(gene=expressed_gene, min_expr=lo)
    want = vals.index[vals >= lo]
    assert 0 < len(want) < len(vals)
    _assert_consistent(xe, D, client, raw, want, expressed_gene)
    xe.subset(gene=expressed_gene, max_expr=0.0)
    _assert_consistent(xe, D, client, raw, vals.index[vals <= 0], expressed_gene)


def test_subset_by_cell_type_and_ids(xe, D, client, raw, expressed_gene, labels):
    xe.subset(cell_type="Beta")
    _assert_consistent(xe, D, client, raw, labels.index[labels == "Beta"], expressed_gene)
    xe.subset(cell_type=["Alpha", "Gamma"])
    _assert_consistent(xe, D, client, raw, labels.index[labels != "Beta"], expressed_gene)
    ids = list(raw.cells.index[::3])
    xe.subset(cell_ids=list(reversed(ids)) + ["__not_a_cell__"])
    _assert_consistent(xe, D, client, raw, ids, expressed_gene)


def test_second_subset_starts_from_the_full_table(xe, D, client, raw, expressed_gene):
    c = raw.cells
    hi = _median(c["transcript_counts"])
    xe.subset(min_transcripts=hi)
    xe.subset(max_transcripts=hi)          # not "min and max": the previous filter is replaced
    want = c.index[c["transcript_counts"] <= hi]
    _assert_consistent(xe, D, client, raw, want, expressed_gene)


def test_unsubset_restores_everything(xe, D, client, raw, expressed_gene):
    before_df = D["df"]
    before_map = np.asarray(D["df_to_expr"]).copy()
    v0 = xe._subset_version
    token_full = _assert_consistent(xe, D, client, raw, raw.cells.index, expressed_gene)
    xe.subset(min_transcripts=_median(raw.cells["transcript_counts"]))
    assert xe._subset_version == v0 + 1 and "_df_original" in D
    token_sub = tl.get_bin(client, "/viewer/cells.bin")[0]["token"]
    assert token_sub != token_full, "the viewer must be told to reload the cells"
    xe.unsubset()
    assert D["df"] is before_df and "_df_original" not in D and "_df_to_expr_orig" not in D
    np.testing.assert_array_equal(np.asarray(D["df_to_expr"]), before_map)
    assert xe._subset_version == v0 + 2
    token_back = _assert_consistent(xe, D, client, raw, raw.cells.index, expressed_gene)
    assert token_back != token_sub
    xe.unsubset()                          # no subset active: a no-op, not an error
    assert xe._subset_version == v0 + 2


def test_empty_subset_does_not_break_the_views(xe, D, client, raw, expressed_gene):
    xe.subset(min_transcripts=float(raw.cells["transcript_counts"].max()) + 1)
    assert len(D["df"]) == 0
    method = D["cluster_methods"][0] if D["cluster_methods"] else None
    for mode in ("gene", "transcript_counts", "cluster"):
        fig = xe.make_spatial_fig(mode, method, expressed_gene, 3, 0.8, [], None)
        assert sum(len(t.x) for t in fig.data if t.x is not None) == 0
    assert tl.get_bin(client, "/viewer/cells.bin")[0]["n"] == 0
    assert len(tl.get_bin(client, f"/viewer/colors.bin?mode=gene&gene={expressed_gene}")[1]["values"]) == 0
    assert client.get("/viewer/meta").status_code == 200
    xe.unsubset()
    assert len(D["df"]) == len(raw.cells)


def test_invalid_subset_leaves_no_half_applied_state(xe, D, raw):
    if not D["cluster_methods"]:
        pytest.skip("bundle has no clustering")
    v0 = xe._subset_version
    with pytest.raises(ValueError):
        xe.subset(cluster=1, method="__no_such_method__")
    assert "_df_original" not in D and len(D["df"]) == len(raw.cells) and xe._subset_version == v0
    xe.subset(gene="__no_such_gene__", min_expr=1.0)       # unknown gene: filter skipped, not an error
    assert len(D["df"]) == len(raw.cells)


def test_umap_follows_subset(xe, D, raw, expressed_gene):
    if raw.umap is None:
        pytest.skip("bundle has no UMAP projection")
    keep = list(raw.cells.index[::2])
    xe.subset(cell_ids=keep)
    method = D["cluster_methods"][0] if D["cluster_methods"] else None
    cells = tl.fig_cells(xe.make_umap_fig("gene", method, expressed_gene, 3, 0.8))
    assert set(cells.index) == set(keep) & set(raw.umap.index)
    np.testing.assert_allclose(cells["color"].to_numpy(dtype=float),
                               _gene_values(raw, expressed_gene).loc[cells.index].to_numpy(), rtol=1e-5)


def test_subset_of_only_skipped_filters_is_not_a_subset(xe, D, raw, capsys):
    """A filter that cannot be applied (no annotation, unknown gene, missing column)
    selects nothing; a call made of those alone must not flag 'N of N cells' as a subset."""
    v0 = xe._subset_version
    with xe._annot_lock:
        assert not any(k.startswith("labels_") and v is not None for k, v in xe._annot_state.items())
    out = xe.subset(cell_type="Alpha")                       # no annotation loaded
    assert "_df_original" not in D and "_df_to_expr_orig" not in D
    assert out is D["df"] and len(out) == len(raw.cells) and xe._subset_version == v0
    xe.subset(gene="__no_such_gene__", min_expr=1.0)
    xe.subset()
    assert "_df_original" not in D and xe._subset_version == v0
    assert "no filter was applied" in capsys.readouterr().out
    assert xe.update_subset_indicator(0)[0] == {"display": "none"}, "no SUBSET ACTIVE badge"

    # a skipped filter next to an applied one still subsets by the applied one
    keep = list(raw.cells.index[::2])
    xe.subset(cell_type="Alpha", cell_ids=keep)
    assert list(D["df"].index.astype(str)) == keep and "_df_original" in D
    # ... and an all-skipped call does not disturb an active subset
    xe.subset(gene="__no_such_gene__", min_expr=1.0)
    assert list(D["df"].index.astype(str)) == keep
    xe.unsubset()
    assert len(D["df"]) == len(raw.cells)


def test_subset_matching_every_cell_is_still_a_subset(xe, D, raw):
    xe.subset(min_transcripts=0)          # applied; happens to keep everything
    assert "_df_original" in D and len(D["df"]) == len(raw.cells)
