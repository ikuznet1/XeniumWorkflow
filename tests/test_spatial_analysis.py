"""Spatial statistics and niches, against an independent recount.

The app builds its neighbour graph with scikit-learn's ball tree; the recount here
uses scipy's k-d tree and plain loops over (cell, neighbour) pairs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

import xe_testlib as tl

K = 5


def _neighbours(xy, k):
    """Indices of the k nearest other cells (self excluded) for every cell."""
    _, idx = cKDTree(xy).query(xy, k=min(k + 1, len(xy)))
    return idx[:, 1:]


def _pair_counts(labels: pd.Series, xy: np.ndarray, k: int):
    """types, counts[type of cell, type of neighbour], cells per type - by explicit loop."""
    types = sorted(set(labels))
    pos = {t: i for i, t in enumerate(types)}
    nb = _neighbours(xy, k)
    lab = labels.to_numpy()
    m = np.zeros((len(types), len(types)))
    for i in range(len(lab)):
        for j in nb[i]:
            m[pos[lab[i]], pos[lab[j]]] += 1
    n = np.array([(lab == t).sum() for t in types], dtype=float)
    return types, m, n


def _run_analysis(xe, key="labels_celltypist", k=K, seg="xenium"):
    out = xe.run_spatial_analysis(1, key, k, seg)
    assert out == (True, False, "Running…")
    tl.wait_state(xe._spatial_state, xe._spatial_lock)
    return xe._spatial_state["result"]


def _unambiguous(xy, k):
    """False when the k-th and (k+1)-th neighbour of some cell are equidistant, i.e. the
    neighbour set depends on tie-breaking and two correct implementations may differ."""
    if len(xy) <= k + 1:
        return True
    d, _ = cKDTree(xy).query(xy, k=k + 2)
    return bool((d[:, k + 1] - d[:, k] > 1e-9).all())


def test_neighbour_counts_equal_an_independent_recount(xe, D, raw, labels):
    xy = raw.cells[["x_centroid", "y_centroid"]].to_numpy(float)
    if not _unambiguous(xy, K):
        pytest.skip("equidistant neighbours: recount not unique")
    res = _run_analysis(xe)
    types, counts, n = _pair_counts(labels, xy, K)
    assert res["cell_types"] == types
    np.testing.assert_array_equal(np.asarray(res["adjacency"]), counts)
    assert counts.sum() == len(labels) * min(K, len(labels) - 1)
    np.testing.assert_allclose(res["freqs"], n / n.sum())
    # scores, from their definitions
    total = counts.sum()
    f = n / n.sum()
    np.testing.assert_allclose(res["cooccurrence"], np.log2(counts / (np.outer(f, f) * total + 1e-9) + 1e-9))
    # enrichment: P(neighbour is b | cell is a) / P(b); 0 under random mixing, whatever k is
    cond = counts / counts.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(res["enrichment"], np.log2(cond / f[None, :] + 1e-9))
    assert "Done" in xe._spatial_state["message"] and str(len(types)) in xe._spatial_state["message"]
    assert xe.poll_spatial(0, 3)[1:] == (True, False, 4)


def test_unlabelled_cells_are_left_out_of_the_graph(xe, D, raw, labels):
    """Cells without a label are dropped before neighbours are searched: the graph is
    built among labelled cells only."""
    partial = labels.iloc[3:]
    with xe._annot_lock:
        xe._annot_state["labels_celltypist"] = partial
    xy = raw.cells.loc[partial.index, ["x_centroid", "y_centroid"]].to_numpy(float)
    if not _unambiguous(xy, K):
        pytest.skip("equidistant neighbours: recount not unique")
    res = _run_analysis(xe)
    types, counts, n = _pair_counts(partial, xy, K)
    np.testing.assert_array_equal(np.asarray(res["adjacency"]), counts)
    assert f"{len(partial):,} cells" in xe._spatial_state["message"]


def test_analysis_runs_on_the_active_subset(xe, D, raw, labels):
    keep = list(raw.cells.index[::2])
    xe.subset(cell_ids=keep)
    xy = raw.cells.loc[keep, ["x_centroid", "y_centroid"]].to_numpy(float)
    if not _unambiguous(xy, K):
        pytest.skip("equidistant neighbours: recount not unique")
    res = _run_analysis(xe)
    types, counts, _ = _pair_counts(labels.loc[keep], xy, K)
    assert res["cell_types"] == types
    np.testing.assert_array_equal(np.asarray(res["adjacency"]), counts)


def test_analysis_without_labels_reports_instead_of_running(xe):
    assert xe.run_spatial_analysis(1, None, K, "xenium") == \
        (False, True, "Select a cell type annotation from the dropdown first.")
    xe.run_spatial_analysis(1, "labels_seurat", K, "xenium")
    with pytest.raises(AssertionError, match="No annotation found"):
        tl.wait_state(xe._spatial_state, xe._spatial_lock)
    assert xe._spatial_state["result"] is None


def test_analysis_figures_show_the_computed_matrices(xe, raw, labels):
    res = _run_analysis(xe)
    types = res["cell_types"]
    for builder, key in (("make_cooccurrence_fig", "cooccurrence"), ("make_enrichment_fig", "enrichment")):
        fig = getattr(xe, builder)()
        (heat,) = fig.data
        assert heat.type == "heatmap" and list(heat.x) == types and list(heat.y) == types
        np.testing.assert_allclose(np.asarray(heat.z, dtype=float), np.asarray(res[key], dtype=float))
        ranged = getattr(xe, builder)(zmin=-1.5, zmax=2.5)
        assert (ranged.data[0].zmin, ranged.data[0].zmax) == (-1.5, 2.5)

    (bar,) = xe.make_cell_type_abundance_fig().data
    share = labels.value_counts(normalize=True)
    assert dict(zip(bar.x, np.round(bar.y, 9))) == {k: round(v, 9) for k, v in share.items()} or \
        dict(zip(bar.x, np.round(np.asarray(bar.y) / 100, 9))) == {k: round(v, 9) for k, v in share.items()}

    graph = xe.make_interaction_graph_fig()
    counts = np.asarray(res["adjacency"])
    n_edges = sum(1 for a in range(len(types)) for b in range(a + 1, len(types))
                  if counts[a, b] + counts[b, a] > 0)
    edge_traces = [t for t in graph.data if t.mode == "lines"]
    assert len(edge_traces) == min(n_edges, 30)
    node_text = [str(t) for tr in graph.data if tr.mode and "markers" in tr.mode for t in (tr.text or [])]
    assert all(any(ty in txt for txt in node_text) for ty in types)
    # frequency threshold removes rare types from graph and chord alike
    rare = float(np.sort(res["freqs"])[0] * 100)
    kept = int((np.asarray(res["freqs"]) * 100 >= rare + 1e-6).sum())
    assert kept < len(types)
    chord = xe.make_chord_fig(min_freq=rare + 1e-6)
    assert {t.name for t in chord.data if t.name} == {ty for ty, f in zip(types, res["freqs"]) if f * 100 >= rare + 1e-6}
    assert len(xe.make_interaction_graph_fig(min_freq=101).data) == 0
    for normalize in (False, True):
        assert len(xe.make_chord_fig(normalize=normalize).data) > 0
        assert len(xe.make_interaction_graph_fig(normalize=normalize).data) > 0


# ── niches ───────────────────────────────────────────────────────────────────
def _composition(labels: pd.Series, xy, k):
    types = sorted(set(labels))
    pos = {t: i for i, t in enumerate(types)}
    nb = _neighbours(xy, k)
    lab = labels.to_numpy()
    comp = np.zeros((len(lab), len(types)), dtype=np.float32)
    for i in range(len(lab)):
        for j in nb[i]:
            comp[i, pos[lab[j]]] += 1
    return comp / comp.sum(axis=1, keepdims=True)


def _same_partition(a, b) -> bool:
    return len(pd.crosstab(pd.Series(list(a)), pd.Series(list(b))).stack().loc[lambda s: s > 0]) == \
        len(set(a)) == len(set(b))


def test_niches_cluster_the_neighbourhood_composition(xe, D, raw, labels):
    from sklearn.cluster import KMeans
    n_niches = 3
    xy = raw.cells[["x_centroid", "y_centroid"]].to_numpy(float)
    if not _unambiguous(xy, K):
        pytest.skip("equidistant neighbours: recount not unique")
    assert xe.run_niche(1, n_niches, K, "labels_celltypist", "xenium")[:2] == (True, False)
    tl.wait_state(xe._niche_state, xe._niche_lock)
    niche = D["df"]["niche"]
    assert niche.notna().all() and niche.nunique() == n_niches
    comp = _composition(labels, xy, K)
    # (1) cells with the same neighbourhood composition are in the same niche
    by_comp = pd.Series(niche.to_numpy(), index=[tuple(np.round(r, 6)) for r in comp])
    assert by_comp.groupby(level=0).nunique().max() == 1
    # (2) the partition is the k-means partition of the independently built composition
    want = KMeans(n_clusters=n_niches, random_state=42, n_init=10).fit_predict(comp)
    assert _same_partition(niche.to_numpy(), want)
    # (3) and it is shown: one trace per niche with exactly its cells
    method = D["cluster_methods"][0] if D["cluster_methods"] else None
    cells = tl.fig_cells(xe.make_spatial_fig("niche", method, D["gene_names"][0], 3, 0.8, [], None))
    assert (cells["trace"] == ("Niche " + niche.astype(str)).rename(index=str).loc[cells.index]).all()
    assert len(xe.make_niche_umap_fig().data) > 0
    h, a = tl.get_bin(xe.app.server.test_client(), "/viewer/colors.bin?mode=niche")
    assert (np.array(h["categories"])[a["codes"]] == ("Niche " + niche.astype(str)).to_numpy()).all()


def test_niches_on_a_subset_label_only_the_subset_and_survive_unsubset(xe, D, raw, labels):
    keep = list(raw.cells.index[: max(8, 2 * len(raw.cells) // 3)])
    xe.subset(cell_ids=keep)
    xe.run_niche(1, 2, 3, "labels_celltypist", "xenium")
    tl.wait_state(xe._niche_state, xe._niche_lock)
    assert list(D["df"].index.astype(str)) == keep and D["df"]["niche"].notna().all()
    xe.unsubset()
    full = D["df"]["niche"]
    assert full.loc[keep].notna().all()
    assert full.drop(index=keep).isna().all(), "cells outside the subset have no niche"


def test_niches_without_labels_cluster_positions(xe, D, raw):
    xe.run_niche(1, 2, K, None, "xenium")
    tl.wait_state(xe._niche_state, xe._niche_lock)
    niche = D["df"]["niche"]
    assert niche.nunique() == 2
    # two position clusters: each cell is nearer to its own cluster's centre
    xy = raw.cells[["x_centroid", "y_centroid"]].to_numpy(float)
    z = (xy - xy.mean(0)) / (xy.std(0) + 1e-8)
    centres = {k: z[(niche == k).to_numpy()].mean(0) for k in niche.unique()}
    nearer = 0
    for k, c in centres.items():
        other = next(v for kk, v in centres.items() if kk != k)
        mine = z[(niche == k).to_numpy()]
        nearer += int((np.linalg.norm(mine - c, axis=1) <= np.linalg.norm(mine - other, axis=1) + 1e-9).sum())
    # k-means stops at a tolerance, so a few cells on the dividing line may sit on either side
    assert nearer >= 0.995 * len(z)


def test_label_key_is_checked_against_the_active_segmentation(xe, labels):
    f = xe._label_key_for_active_seg
    assert f("labels_celltypist") == "labels_celltypist"
    assert f("labels_celltypist_proseg_abcd1234") is None or f("labels_celltypist_proseg_abcd1234") == "labels_celltypist"
    assert f(None) is None and f("something_else") == "something_else"
    alt = {"out_dir": "/x/run", "source": "proseg"}
    key = xe._labels_key_for_method("celltypist", alt)
    assert key.startswith("labels_celltypist_proseg_") and key != xe._labels_key_for_method("celltypist", {"out_dir": "/y", "source": "proseg"})
    assert f("labels_celltypist", alt) is None, "Xenium labels must not be used for a Proseg run"
    with xe._annot_lock:
        xe._annot_state[key] = labels
    assert f("labels_celltypist", alt) == key


def test_cells_without_a_niche_are_not_called_nan(xe, D, raw, labels, client):
    """After unsubset() the cells outside the subset have no niche (NaN): the legend
    and the viewer categories call them 'No niche', never 'Niche nan'."""
    keep = list(raw.cells.index[: max(8, 2 * len(raw.cells) // 3)])
    rest = [c for c in raw.cells.index if c not in set(keep)]
    assert rest
    xe.subset(cell_ids=keep)
    xe.run_niche(1, 2, 3, "labels_celltypist", "xenium")
    tl.wait_state(xe._niche_state, xe._niche_lock)
    xe.unsubset()
    niche = D["df"]["niche"]
    want = pd.Series(np.where(niche.isna(), "No niche", "Niche " + niche.astype(str)),
                     index=D["df"].index.astype(str))
    method = D["cluster_methods"][0] if D["cluster_methods"] else None
    cells = tl.fig_cells(xe.make_spatial_fig("niche", method, D["gene_names"][0], 3, 0.8, [], None))
    assert (cells["trace"] == want.loc[cells.index]).all()
    assert set(cells.loc[rest, "trace"]) == {"No niche"}
    assert not any("nan" in str(n).lower() for n in set(cells["trace"]))
    h, a = tl.get_bin(client, "/viewer/colors.bin?mode=niche")
    assert (np.array(h["categories"])[a["codes"]] == want.to_numpy()).all()
    names = [t.name for t in xe.make_niche_umap_fig().data]
    assert not any("nan" in str(n).lower() for n in names)
    if raw.umap is not None:
        assert "No niche" in names


def test_heatmap_figures_use_the_requested_size(xe):
    types = ["Alpha", "Beta", "Gamma"]
    m = np.arange(9, dtype=float).reshape(3, 3)
    with xe._spatial_lock:
        xe._spatial_state.update({"status": "done", "result": {
            "cell_types": types, "cooccurrence": m, "enrichment": -m, "adjacency": m,
            "freqs": np.full(3, 1 / 3), "niche_labels": None, "umap_coords": None}})
    for make in (xe.make_cooccurrence_fig, xe.make_enrichment_fig):
        default = make()
        assert default.layout.height == 400 and default.layout.width is None
        sized = make(width=640, height=520)
        assert (sized.layout.width, sized.layout.height) == (640, 520)
        only_w = make(width=700)
        assert (only_w.layout.width, only_w.layout.height) == (700, 400)
        np.testing.assert_array_equal(np.asarray(sized.data[0].z), np.asarray(default.data[0].z))
