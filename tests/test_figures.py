"""Figure builders in every colour mode.

For each mode the figure is taken apart again (tl.fig_cells gives one row per drawn
cell) and compared with values read independently from the bundle: which cells are
drawn, where, in which trace, and with which colour value.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pandas as pd
import pytest

import xe_testlib as tl

CONTINUOUS = ["gene", "transcript_counts", "cell_area", "nucleus_area"]


@pytest.fixture
def method(D):
    return D["cluster_methods"][0] if D["cluster_methods"] else None


@pytest.fixture
def spatial(xe, method, expressed_gene):
    def build(color_by, **kw):
        return xe.make_spatial_fig(color_by, kw.pop("method", method), kw.pop("gene", expressed_gene),
                                   kw.pop("size", 3), kw.pop("opacity", 0.8),
                                   kw.pop("boundaries", []), kw.pop("relayout", None), **kw)
    return build


@pytest.fixture
def umap(xe, method, expressed_gene):
    def build(color_by, **kw):
        return xe.make_umap_fig(color_by, kw.pop("method", method), kw.pop("gene", expressed_gene),
                                kw.pop("size", 3), kw.pop("opacity", 0.8), **kw)
    return build


@pytest.fixture
def niche_column(D):
    df = D["df"]
    df["niche"] = (np.arange(len(df)) * 7 % 3).astype(int)
    return df["niche"].copy()


@pytest.fixture
def roi_column(D):
    df = D["df"]
    col = pd.Series(pd.NA, index=df.index, dtype=object)
    col.iloc[: max(2, len(df) // 4)] = "left"
    col.iloc[-max(2, len(df) // 5):] = "right"
    df["roi_zone"] = col
    return col


def _raw_values(raw, mode, gene):
    """Independent colour value per cell for a continuous mode."""
    if mode == "gene":
        counts, names, barcodes = raw.gene_counts
        col = np.asarray(counts[:, names.index(gene)].todense()).ravel()
        return pd.Series(np.log1p(col.astype(np.float64)), index=barcodes)
    if mode == "transcript_counts":
        return np.log1p(raw.cells["transcript_counts"].astype(float))
    return raw.cells[mode].astype(float)


# ── spatial: geometry ────────────────────────────────────────────────────────
@pytest.mark.parametrize("mode", ["cluster", "gene", "transcript_counts", "cell_area", "nucleus_area",
                                  "cell_type:celltypist", "niche", "roi_zone"])
def test_spatial_draws_every_cell_once_at_its_centroid(spatial, raw, labels, niche_column, roi_column, mode):
    if mode == "cluster" and raw.clusters("gene_expression_graphclust") is None:
        pytest.skip("bundle has no clustering")
    fig = spatial(mode)
    cells = tl.fig_cells(fig)                          # asserts no cell is drawn twice
    truth = raw.cells
    assert set(cells.index) == set(truth.index)
    truth = truth.loc[cells.index]
    np.testing.assert_allclose(cells["x"], truth["x_centroid"], atol=1e-4)
    np.testing.assert_allclose(cells["y"], -truth["y_centroid"], atol=1e-4)     # plot y is negated
    assert all(t.type == "scattergl" for t in tl.marker_traces(fig))


def test_spatial_marker_size_and_opacity_are_applied(spatial):
    fig = spatial("transcript_counts", size=7, opacity=0.35)
    (trace,) = tl.marker_traces(fig)
    assert trace.marker.size == 7 and trace.marker.opacity == 0.35


# ── spatial: continuous modes ────────────────────────────────────────────────
@pytest.mark.parametrize("mode", CONTINUOUS)
def test_spatial_continuous_values_come_from_the_raw_files(spatial, raw, expressed_gene, mode):
    if mode != "gene" and mode not in raw.cells.columns:
        pytest.skip(f"bundle has no {mode}")
    fig = spatial(mode)
    cells = tl.fig_cells(fig)
    assert len(tl.marker_traces(fig)) == 1, "a continuous mode is one trace with a colour array"
    want = _raw_values(raw, mode, expressed_gene).loc[cells.index]
    np.testing.assert_allclose(cells["color"].to_numpy(dtype=float), want.to_numpy(dtype=float), rtol=1e-5)
    assert tl.marker_traces(fig)[0].marker.showscale in (True, None)
    assert tl.marker_traces(fig)[0].marker.colorscale is not None
    if mode == "gene":
        assert float(np.nanmax(cells["color"].to_numpy(dtype=float))) > 0, "fixture gene is expressed"


def test_spatial_colourbar_range_is_passed_through(spatial):
    marker = tl.marker_traces(spatial("gene", cbar_range=[0.5, 2.0]))[0].marker
    assert (marker.cmin, marker.cmax) == (0.5, 2.0)
    marker = tl.marker_traces(spatial("gene", cbar_range=[None, 1.25]))[0].marker
    assert marker.cmin is None and marker.cmax == 1.25
    marker = tl.marker_traces(spatial("gene"))[0].marker
    assert marker.cmin is None and marker.cmax is None


def test_spatial_gene_uses_corrected_counts_only_when_asked(xe, D, spatial, raw, expressed_gene):
    counts, names, barcodes = raw.gene_counts
    col = np.asarray(counts[:, names.index(expressed_gene)].todense()).ravel().astype(float)
    D["split_corrected_expr"] = (D["expr"] * 0.5).tocsr()        # stands in for SPLIT output
    for cache in (xe._gene_expr_cache,):
        cache.clear()
    corrected = tl.fig_cells(spatial("gene", use_corrected=True))["color"].to_numpy(dtype=float)
    original = tl.fig_cells(spatial("gene", use_corrected=False))["color"].to_numpy(dtype=float)
    np.testing.assert_allclose(corrected, np.log1p(0.5 * col), rtol=1e-5)
    np.testing.assert_allclose(original, np.log1p(col), rtol=1e-5)
    assert not np.allclose(corrected, original)


# ── spatial: categorical modes ───────────────────────────────────────────────
def _assert_partition(cells, expected_groups: pd.Series, name_of):
    """Every trace holds exactly the cells of one category, in one colour, and distinct
    categories get distinct traces."""
    expected_groups = expected_groups.loc[cells.index]
    by_trace = cells.groupby("trace")
    assert by_trace["color"].nunique().eq(1).all(), "a category trace has one colour"
    got = {name: set(g.index) for name, g in by_trace}
    want = {name_of(k): set(g.index) for k, g in expected_groups.groupby(expected_groups)}
    assert got == want
    colours = by_trace["color"].first()
    if len(colours) <= 20:
        assert colours.is_unique, "categories must be distinguishable"


def test_spatial_cluster_traces_are_the_analysis_clusters(spatial, raw, D, xe):
    if not D["cluster_methods"]:
        pytest.skip("bundle has no clustering")
    for method in D["cluster_methods"][:3]:
        truth = raw.clusters(method)
        if truth is None:
            continue
        cells = tl.fig_cells(spatial("cluster", method=method))
        full = truth.reindex(cells.index).fillna(0).astype(int)        # cells left out by the analysis
        _assert_partition(cells, full, lambda k: f"Cluster {k}")
        # colours are a function of the cluster id (stable when a cluster is empty in view)
        cmap = xe.cluster_color_map(method)
        for name, colour in cells.groupby("trace")["color"].first().items():
            assert colour == cmap[int(name.split()[-1])]


def test_spatial_cell_type_without_labels_is_a_placeholder(spatial, raw):
    fig = spatial("cell_type:celltypist")
    assert [t.name for t in fig.data if t.mode == "markers"] == ["No annotation"]
    assert len(fig.data[0].x) == len(raw.cells)


def test_spatial_cell_type_traces_follow_the_labels(spatial, labels):
    cells = tl.fig_cells(spatial("cell_type:celltypist"))
    _assert_partition(cells, labels, str)
    assert set(cells["trace"]) == {"Alpha", "Beta", "Gamma"}


def test_spatial_cell_type_unlabelled_cells_are_unknown(xe, spatial, labels, D):
    with xe._annot_lock:
        xe._annot_state["labels_celltypist"] = labels.iloc[2:]          # two cells have no label
    cells = tl.fig_cells(spatial("cell_type:celltypist"))
    missing = list(D["df"].index.astype(str)[:2])
    assert len(cells) == len(D["df"])
    assert set(cells.loc[missing, "trace"]) == {"Unknown"}
    assert "Unknown" not in set(cells.drop(index=missing)["trace"])


def test_spatial_niche_traces(spatial, niche_column, raw):
    cells = tl.fig_cells(spatial("niche"))
    _assert_partition(cells, niche_column.rename(index=str), lambda k: f"Niche {k}")


def test_spatial_niche_without_niches_falls_back_to_counts(spatial, raw):
    cells = tl.fig_cells(spatial("niche"))
    want = np.log1p(raw.cells["transcript_counts"].astype(float)).loc[cells.index]
    np.testing.assert_allclose(cells["color"].to_numpy(dtype=float), want.to_numpy(), rtol=1e-5)


def test_spatial_roi_mode_colours_members_and_greys_the_rest(xe, spatial, roi_column):
    cells = tl.fig_cells(spatial("roi_zone"))
    col = roi_column.rename(index=str).loc[cells.index]
    colours = cells["color"]
    for name in ("left", "right"):
        assert colours[(col == name).to_numpy()].nunique() == 1
    assert colours[(col == "left").to_numpy()].iloc[0] != colours[(col == "right").to_numpy()].iloc[0]
    outside = colours[col.isna().to_numpy()]
    assert outside.nunique() == 1 and outside.iloc[0] not in set(colours[col.notna().to_numpy()])


# ── spatial: boundary overlay ────────────────────────────────────────────────
def _viewport(raw, half=30.0):
    c = raw.cells
    mid = c.iloc[((c["x_centroid"] - c["x_centroid"].median()) ** 2
                  + (c["y_centroid"] - c["y_centroid"].median()) ** 2).argmin()]
    x0, x1 = float(mid["x_centroid"]) - half, float(mid["x_centroid"]) + half
    y0, y1 = float(mid["y_centroid"]) - half, float(mid["y_centroid"]) + half
    inside = c.index[c["x_centroid"].between(x0, x1) & c["y_centroid"].between(y0, y1)]
    relayout = {"xaxis.range[0]": x0, "xaxis.range[1]": x1, "yaxis.range[0]": -y1, "yaxis.range[1]": -y0}
    return relayout, inside


@pytest.mark.parametrize("kind,trace_name", [("cell", "Xenium Cell Boundaries"),
                                             ("nucleus", "Xenium Nuclei Boundaries")])
def test_boundary_overlay_draws_the_polygons_in_view(spatial, raw, kind, trace_name):
    truth = raw.boundaries(kind)
    if not truth:
        pytest.skip(f"bundle has no {kind} boundaries")
    relayout, inside = _viewport(raw)
    fig = spatial("transcript_counts", boundaries=[kind], relayout=relayout)
    lines = [t for t in fig.data if t.mode == "lines"]
    assert [t.name for t in lines] == [trace_name]
    polys = tl.line_polygons(lines[0])
    rings = raw.rings(kind)
    wanted = [ring * [1, -1] for cid in inside for ring in rings.get(cid, [])]     # plot y is negated
    assert len(polys) == len(wanted) > 0, "the outlines of the cells whose centroid is in view"
    assert {tl.point_set(p) for p in polys} == {tl.point_set(w) for w in wanted}
    # the viewport is kept
    assert tuple(fig.layout.xaxis.range) == (relayout["xaxis.range[0]"], relayout["xaxis.range[1]"])
    assert tuple(fig.layout.yaxis.range) == (relayout["yaxis.range[0]"], relayout["yaxis.range[1]"])


def test_no_boundary_traces_without_a_viewport_or_toggle(spatial, raw):
    relayout, _ = _viewport(raw)
    assert not [t for t in spatial("cluster", boundaries=[], relayout=relayout).data if t.mode == "lines"]
    both = spatial("cluster", boundaries=["cell", "nucleus"], relayout=relayout)
    assert len([t for t in both.data if t.mode == "lines"]) == \
        bool(raw.boundaries("cell")) + bool(raw.boundaries("nucleus"))


def test_build_boundary_trace_closes_and_separates_rings(xe, D):
    ids = list(D["cell_bounds"])[:3]
    if not ids:
        pytest.skip("no boundaries in memory")
    trace = xe.build_boundary_trace(ids, D["cell_bounds"], "#123456", "T")
    polys = tl.line_polygons(trace)
    assert len(polys) == len(ids) and trace.line.color == "#123456" and trace.name == "T"
    for cid, p in zip(ids, polys):
        xs, ys = (np.asarray(v, dtype=float) for v in D["cell_bounds"][cid])
        np.testing.assert_allclose(p[:, 0], xs[~np.isnan(xs)], atol=1e-5)
        np.testing.assert_allclose(p[:, 1], -ys[~np.isnan(ys)], atol=1e-5)
    assert xe.build_boundary_trace([], D["cell_bounds"], "#000", "T") is None
    assert xe.build_boundary_trace(["__nope__"], D["cell_bounds"], "#000", "T") is None


# ── UMAP ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mode", ["cluster", "gene", "transcript_counts", "cell_area", "nucleus_area",
                                  "cell_type:celltypist", "niche", "roi_zone"])
def test_umap_draws_embedded_cells_at_their_projection(umap, raw, labels, niche_column, roi_column, mode):
    truth = raw.umap
    if truth is None:
        pytest.skip("bundle has no UMAP projection")
    cells = tl.fig_cells(umap(mode))
    assert set(cells.index) == set(truth.index), "exactly the cells that have a projection"
    np.testing.assert_allclose(cells["x"], truth.loc[cells.index, "UMAP-1"], atol=1e-5)
    np.testing.assert_allclose(cells["y"], truth.loc[cells.index, "UMAP-2"], atol=1e-5)


@pytest.mark.parametrize("mode", CONTINUOUS)
def test_umap_continuous_values_come_from_the_raw_files(umap, raw, expressed_gene, mode):
    if raw.umap is None or (mode != "gene" and mode not in raw.cells.columns):
        pytest.skip("bundle has no UMAP / column")
    cells = tl.fig_cells(umap(mode))
    want = _raw_values(raw, mode, expressed_gene).loc[cells.index]
    np.testing.assert_allclose(cells["color"].to_numpy(dtype=float), want.to_numpy(dtype=float), rtol=1e-5)


def test_umap_categorical_modes_partition_like_the_spatial_view(umap, spatial, raw, labels, niche_column):
    if raw.umap is None:
        pytest.skip("bundle has no UMAP projection")
    for mode, strip in (("cluster", "Cluster "), ("cell_type:celltypist", ""), ("niche", "Niche ")):
        u = tl.fig_cells(umap(mode))
        s = tl.fig_cells(spatial(mode)).loc[u.index]
        norm = lambda v: v.str.replace(strip, "", regex=False) if strip else v      # noqa: E731
        assert (norm(u["trace"]) == norm(s["trace"])).all(), mode
        # same category -> same colour in both views
        pairs = pd.DataFrame({"u": u["color"], "s": s["color"]}).drop_duplicates()
        assert pairs["u"].is_unique and pairs["s"].is_unique, mode


def test_umap_figure_cache_does_not_serve_stale_colours(xe, umap, labels, raw):
    if raw.umap is None:
        pytest.skip("bundle has no UMAP projection")
    first = tl.fig_cells(umap("cell_type:celltypist"))
    flipped = labels.map({"Alpha": "Beta", "Beta": "Gamma", "Gamma": "Alpha"})
    with xe._annot_lock:
        xe._annot_state["labels_celltypist"] = flipped
    xe._umap_fig_cache.clear()        # what poll_annotation's annot-done bump leads to
    second = tl.fig_cells(umap("cell_type:celltypist"))
    assert (second["trace"] == flipped.loc[second.index]).all()
    assert (first["trace"] != second["trace"].loc[first.index]).all()


# ── morphology overlay ───────────────────────────────────────────────────────
def test_morphology_overlay_is_the_image_region_under_the_viewport(xe, D):
    import tifffile
    from PIL import Image
    path = xe._morph_focus_path(D["data_dir"], 0)
    if path is None:
        pytest.skip("bundle has no morphology focus image")
    assert xe.make_morphology_overlay(D["data_dir"], None, 0, ["dapi"], 1.0, 0.8) == (None, "")
    assert xe.make_morphology_overlay(D["data_dir"], {"xaxis.range[0]": 0}, 0, [], 1.0, 0.8) == (None, "")
    px = xe.PIXEL_SIZE_UM
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        shape = series.shape[-2:]
        small = shape[0] * shape[1] <= 100_000_000
        coarse = series.levels[-1].asarray()
    coarse = coarse[0] if coarse.ndim == 3 else coarse
    # look where there is signal: a 50 µm window centred on the brightest spot
    r, c = np.unravel_index(int(np.argmax(coarse)), coarse.shape)
    cx = (c + 0.5) * shape[1] / coarse.shape[1] * px
    cy = (r + 0.5) * shape[0] / coarse.shape[0] * px
    relayout = {"xaxis.range[0]": cx - 25, "xaxis.range[1]": cx + 25,
                "yaxis.range[0]": -(cy + 25), "yaxis.range[1]": -(cy - 25)}
    xe._morph_hires_queue.clear()
    with tl.new_threads():       # the first answer may be an overview; full resolution follows
        img, note = xe.make_morphology_overlay(D["data_dir"], relayout, 0, ["dapi"], 1.0, 0.8)
    assert img is not None, note
    if "preview" in note:
        preview = img
        assert len(xe._morph_hires_queue) == 1, "the full-resolution render was not queued"
        img = xe._morph_hires_queue.pop()
        assert (img["x"], img["y"], img["sizex"], img["sizey"]) == \
               (preview["x"], preview["y"], preview["sizex"], preview["sizey"])
    # placed on the pixel grid, covering the viewport, top-left anchored in plot coords
    assert img["xref"] == "x" and img["yref"] == "y" and img["layer"] == "below" and img["opacity"] == 0.8
    assert img["xanchor"] == "left" and img["yanchor"] == "top"
    assert img["x"] <= relayout["xaxis.range[0]"] + 1e-6 and img["y"] >= relayout["yaxis.range[1]"] - 1e-6
    assert img["x"] + img["sizex"] >= relayout["xaxis.range[1]"] - px
    assert img["y"] - img["sizey"] <= relayout["yaxis.range[0]"] + px
    assert img["sizex"] < 50 + 4 * px + 0.3 * 50, "no more than the viewport plus the prefetch margin"
    assert abs(img["x"] / px - round(img["x"] / px)) < 1e-6
    assert img["source"].startswith("data:image/jpeg;base64,")
    jpeg = np.asarray(Image.open(io.BytesIO(base64.b64decode(img["source"].split(",", 1)[1]))).convert("L"),
                      dtype=float)
    assert jpeg.std() > 0, "the window was chosen around the brightest spot of the image"
    if not small:
        return          # reading a region of a multi-GB JPEG-2000 image is the app's own job
    # content: the JPEG correlates with the same region read straight from the TIFF
    full = tifffile.imread(path, level=0)
    full = full[0] if full.ndim == 3 else full
    c0, r0 = int(round(img["x"] / px)), int(round(-img["y"] / px))
    c1, r1 = c0 + int(round(img["sizex"] / px)), r0 + int(round(img["sizey"] / px))
    region = np.asarray(Image.fromarray(full[r0:r1, c0:c1].astype(np.float32)).resize(
        (jpeg.shape[1], jpeg.shape[0]), Image.BILINEAR), dtype=float)
    assert np.corrcoef(region.ravel(), jpeg.ravel())[0, 1] > 0.9


# ── analysis tabs before any analysis ────────────────────────────────────────
@pytest.mark.parametrize("builder", ["make_cell_type_abundance_fig", "make_cooccurrence_fig",
                                     "make_enrichment_fig", "make_interaction_graph_fig",
                                     "make_chord_fig", "make_niche_umap_fig", "make_roi_area_fig"])
def test_analysis_figures_are_placeholders_until_computed(xe, builder):
    fig = getattr(xe, builder)()
    assert len(fig.data) == 0
    assert len(fig.layout.annotations) == 1 and fig.layout.annotations[0].text.strip()


def test_cell_info_panel_shows_the_cells_own_values(xe, D, raw, method):
    import json
    import plotly
    cid = D["df"].index[len(D["df"]) // 2]
    text = json.dumps(xe.show_cell_info(cid, method), cls=plotly.utils.PlotlyJSONEncoder, ensure_ascii=False)
    row = raw.cells.loc[str(cid)]
    assert str(cid) in text
    assert f"{row['x_centroid']:.1f}" in text and f"{row['y_centroid']:.1f}" in text
    if "cell_area" in row.index:
        assert "Cell Area" in text
