"""Data provider of the WebGL viewer (xenium_viewer_api endpoints) and /health."""

import os
import re
import json
import numpy as np
import pandas as pd
from plotly.colors import qualitative

from . import (
    core, jobs, caching, dataview, annotation, reseg, spatial, morph, figures, analysis,
    cellnest, sdata_store, regions, app_layout,
)
from . import clustersets
from . import joint


# ─── WebGL viewer data provider (binary endpoints under /viewer/*) ────────────
class _ViewerProvider:
    """Feeds xenium_viewer_api's routes from the app's current state. The browser keeps
    cell coordinates on the GPU and asks only for one value per cell on a recolour and
    for boundaries / transcripts in the current viewport."""

    _BOUNDARY_LIMIT = 20000   # polygons per viewport request before asking to zoom in
    _BOUNDARY_PAD   = 30.0    # µm: a cell is "in view" if its centroid is within this margin

    def __init__(self):
        self._packed = {}     # (tool, kind) -> packed in-memory boundary dict
        self._zip_ok = {}     # kind -> whether cells.zarr.zip matches DATA["df"]

    @staticmethod
    def _alt_res(source):
        return core._seg_result(source)

    def _frame(self, source):
        alt = self._alt_res(source)
        return (alt["cells_df"] if alt is not None else core.DATA["df"]), alt

    _multi_sample = True      # extra samples are drawn beside the primary one (not in the UMAP)

    def _extras(self, alt=None):
        """The extra samples shown with the Xenium source, as in make_spatial_fig (a
        Baysor / Proseg source shows that run's cells only)."""
        return list(dataview.EXTRA_DATASETS) if (self._multi_sample and alt is None) else []

    def _xy(self, df, extras=()):
        """Centroids of *df*'s cells followed by those of the extra samples, each shifted
        by its x offset."""
        x, y = df["x_centroid"].values, df["y_centroid"].values
        if extras:
            x = np.concatenate([x] + [e["df"]["x_centroid"].values + e["x_offset"] for e in extras])
            y = np.concatenate([y] + [e["df"]["y_centroid"].values for e in extras])
        return x, y

    def data_dir(self):
        return core.DATA.get("data_dir")

    def image_files(self):
        d = core.DATA.get("data_dir")
        files = []
        if d:
            for i in range(16):
                f = morph._morph_focus_path(d, i)
                if f is None:
                    break
                files.append(f)
        return files

    def _bounds(self, df, extras=()):
        x, y = self._xy(df, extras)
        if len(x) == 0:
            return [0.0, 0.0, 1.0, 1.0]
        return [float(np.nanmin(x)), float(np.nanmin(y)), float(np.nanmax(x)), float(np.nanmax(y))]

    def meta(self):
        d = core.DATA.get("data_dir")
        sources = ["xenium"]
        for tool, lock, state in core._seg_states():
            with lock:
                if state.get("result"):
                    sources.append(tool)
        import xenium_viewer_api as _va
        store = _va.tile_store(d) if d else None
        return {
            "dataset": caching._dataset_name(),
            "dataset_gen": jobs._dataset_gen,
            "sources": sources,
            "bounds": self._bounds(core.DATA["df"], self._extras()),
            "pixel_size": core.PIXEL_SIZE_UM,
            "has_boundaries": bool(core.DATA.get("cell_bounds")) or bool(store and store.has_boundaries),
            "has_transcripts": bool(store and store.has_transcripts),
            # "v" identifies the file: /viewer/image/<index>?v=<v> may be cached by the browser
            "images": [{"index": i, "name": os.path.basename(f), "v": _va.image_token(f)}
                       for i, f in enumerate(self.image_files())],
        }

    def cells(self, source):
        df, alt = self._frame(source)
        extras = self._extras(alt)
        x, y = self._xy(df, extras)
        xy = np.empty(2 * len(x), dtype=np.float32)
        xy[0::2] = x
        xy[1::2] = y
        return ({"n": int(len(x)), "bounds": self._bounds(df, extras), "samples": 1 + len(extras),
                 "token": f"{jobs._dataset_gen}:{core._subset_version}:{core._seg_tool(source)}:{len(x)}:{len(extras)}"},
                {"xy": xy})

    @staticmethod
    def _categorical(labels, label, colors=None, order=None, names=None):
        """Pack a Series of category labels as codes + legend."""
        labels = pd.Series(np.asarray(labels))
        cats = list(order) if order is not None else sorted(labels.dropna().unique(), key=str)
        lut = {c: i for i, c in enumerate(cats)}
        codes = labels.map(lut).fillna(65535).astype(np.uint16).values
        cols = colors or [core.CLUSTER_COLORS[i % len(core.CLUSTER_COLORS)] for i in range(len(cats))]
        return ({"kind": "categorical", "label": label,
                 "categories": [str(names[c]) if names else str(c) for c in cats],
                 "colors": list(cols)}, {"codes": codes})

    @staticmethod
    def _continuous(vals, label, colormap=None):
        """Pack one value per cell. *colormap* (viridis / plasma / magma / cividis) names the
        scale the values are drawn with when it is not the one of the requested mode."""
        vals = np.asarray(vals, dtype=np.float32)
        finite = vals[np.isfinite(vals)]
        lo = float(finite.min()) if len(finite) else 0.0
        hi = float(finite.max()) if len(finite) else 1.0
        if len(finite) and hi == lo:
            # one value everywhere (e.g. a gene with no counts in this run): Plotly's
            # automatic colour range is then value ± 0.5, i.e. the middle of the scale
            lo, hi = lo - 0.5, hi + 0.5
        meta = {"kind": "continuous", "label": label, "min": lo, "max": hi}
        if colormap:
            meta["colormap"] = colormap
        return meta, {"values": vals}

    def colors(self, args):
        source = args.get("source", "xenium")
        mode = args.get("mode") or "transcript_counts"
        method = args.get("method") or ""
        gene = args.get("gene") or dataview._default_gene or ""    # no selection: as update_plots
        corrected = str(args.get("corrected", "0")).lower() in ("1", "true", "corrected")
        df, alt = self._frame(source)
        extras = self._extras(alt)
        if mode.startswith("geneset:"):
            # score of a saved gene set: diverging scale centred at 0 (values of the
            # primary sample first, then the extra ones; the UMAP shows the primary only)
            gs = analysis._geneset_color(mode, source, corrected)
            n_all = len(df) + sum(len(e["df"]) for e in extras)
            if gs is not None and len(gs[0]) >= n_all:
                seq = analysis._view_color_seq.get(gs[2])
                if seq is not None:        # a sequential per-cell value (distance, density, score)
                    meta, arr = self._continuous(gs[0][:n_all], gs[2], "viridis")
                    meta["min"], meta["max"] = seq
                    return meta, arr
                meta, arr = self._continuous(gs[0][:n_all], gs[2], "rdbu")
                meta["min"], meta["max"] = -gs[1], gs[1]
                return meta, arr
            mode = "transcript_counts"
        if extras:
            return self._colors_multi(mode, method, gene, df, extras, corrected)

        if mode == "cluster":
            if alt is None:
                col = dataview.cluster_col(method)
                if col not in df.columns and core.DATA.get("cluster_methods"):
                    col = dataview.cluster_col(core.DATA["cluster_methods"][0])
            else:
                cands = reseg._reseg_cluster_cols(df.columns)      # as make_spatial_fig picks them
                pref = [c for c in cands if ("split" in c) == corrected]
                col = clustersets._cs_reseg_col(df.columns, method) or (pref or cands or [None])[0]
            if col is not None and col in df.columns:
                if alt is not None:
                    # same order, names and NA handling as make_spatial_fig's reseg branch
                    vals = df[col].astype(str)
                    order = sorted(vals.unique(), key=lambda v: (0, int(v)) if v.isdigit() else (1, v))
                    return self._categorical(vals.values, "Cluster", order=order,
                                             names={c: f"Cluster {c}" for c in order})
                vals = df[col]
                order = sorted(vals.dropna().unique())
                # legend entries named as the Plotly traces are (_categorical_traces)
                return self._categorical(vals.values, "Cluster", order=order,
                                         names={c: f"Cluster {c}" for c in order})
            mode = "transcript_counts"

        if mode.startswith("cell_type:"):
            _ct_method = mode.split(":", 1)[1]
            if alt is None:
                with core._annot_lock:
                    labels = core._annot_state.get(f"labels_{_ct_method}")
                if labels is None:
                    # as _cell_type_traces: every cell in one grey "No annotation" group
                    return self._categorical(np.full(len(df), "No annotation", dtype=object),
                                             "Cell type", colors=[core.MUTED], order=["No annotation"])
                aligned = labels.reindex(df.index.astype(str)).fillna("Unknown")
                return self._categorical(aligned.values, "Cell type")
            # a reseg run: no labels, or labels of another segmentation (stale, cleared by
            # the helper as in make_spatial_fig) -> transcript counts
            aligned = annotation._reseg_cell_type_labels(_ct_method, alt, df.index.astype(str))
            if aligned is not None:
                return self._categorical(aligned.values, "Cell type")
            mode = "transcript_counts"

        if mode == "niche":
            if "niche" in df.columns:
                vals = spatial._niche_str(df["niche"])
                order =sorted(vals.unique(), key=lambda v: (0, int(v)) if v.isdigit() else (1, v))
                return self._categorical(vals.values, "Niche", order=order,
                                         names={n: spatial._niche_name(n) for n in order})
            mode = "transcript_counts"

        if mode.startswith("roi_"):
            if mode in df.columns:
                vals = df[mode].fillna("(none)").astype(str)
                named = [v for v in sorted(vals.unique()) if v != "(none)"]
                return self._categorical(vals.values, mode[4:] or "ROI", order=named + ["(none)"],
                                         colors=[regions._roi_color(i) for i in range(len(named))] + ["#666666"])
            mode = "transcript_counts"

        if mode == "gene" and gene:
            # Xenium source: as in make_spatial_fig, _get_expr_values only for corrected
            # counts (its fallback for a streamed SpaGE gene applies log1p a second time)
            if alt is None:
                vals = dataview._xenium_gene_values(gene, corrected)
            else:
                vals = figures._get_expr_values(gene, alt, use_corrected=corrected)
                if vals is None and corrected:
                    # imputed gene of a reseg run: no corrected values exist, the original
                    # ones are shown (as make_spatial_fig does)
                    vals = figures._get_expr_values(gene, alt, use_corrected=False)
            if vals is not None and len(vals) == len(df):
                return self._continuous(vals, f"{gene} (log1p)", "plasma")
            # a dataset / reseg run without this gene: transcript counts, as make_spatial_fig

        # The colour scale goes with what is drawn (COLORSCALES; Viridis for the transcript
        # count fallback), not with the mode that was asked for.
        col = mode if mode in df.columns else "transcript_counts"
        if col not in df.columns:
            return self._continuous(np.full(len(df), np.nan), "No data", "viridis")
        raw = pd.to_numeric(df[col], errors="coerce").values.astype(np.float64)
        if col == "transcript_counts":
            return self._continuous(np.log1p(raw), "Transcript Counts (log1p)", "viridis")
        return self._continuous(raw, core.QC_METRICS.get(col, col), core.COLORSCALES.get(col, "Viridis").lower())

    def _colors_multi(self, mode, method, gene, df, extras, corrected=False):
        """Colours for the primary sample followed by the extra ones, with the per-mode
        rules of make_spatial_fig's multi-sample branch (cluster names prefixed per
        sample; SPLIT-corrected counts only when every sample has them)."""
        dsets = [core.DATA] + list(extras)
        dfs = [df] + [e["df"] for e in extras]
        if mode == "cluster" and not core.DATA.get("cluster_methods"):
            mode = "transcript_counts"      # as update_plots does before building the figure
        if mode == "gene" and not gene:
            mode = "transcript_counts"

        if mode == "cluster":
            parts = []
            for di, (ds, edf) in enumerate(zip(dsets, dfs)):
                cm = ds.get("cluster_methods") or []
                col = f"clust__{method if method in cm else (cm or [method])[0]}"
                # a set computed jointly keeps its labels without the per-sample prefix
                pre = "C" if (method in cm and clustersets._cs_is_shared(method)) else f"S{di}:C"
                parts.append((pre + edf[col].astype(str)).values if col in edf.columns
                             else np.full(len(edf), f"S{di}:C0", dtype=object))
            labels = np.concatenate(parts)
            return self._categorical(labels, "Cluster", order=list(pd.unique(labels)))

        if mode == "niche":
            labels = np.concatenate([
                spatial._niche_str(edf["niche"]).values if "niche" in edf.columns
                else np.full(len(edf), spatial._NICHE_NONE, dtype=object) for edf in dfs])
            order = sorted(set(labels), key=lambda v: (0, int(v)) if v.isdigit() else (1, v))
            return self._categorical(labels, "Niche", order=order,
                                     names={n: spatial._niche_name(n) for n in order})

        if mode.startswith("cell_type:"):
            _m = mode.split(':', 1)[1]
            with core._annot_lock:
                primary = core._annot_state.get(f"labels_{_m}")
            parts = []
            for di, edf in enumerate(dfs):
                lbl = primary if di == 0 else dataview._extra_cell_type_labels(extras[di - 1], _m)[0]
                parts.append(lbl.reindex(edf.index).fillna("Unknown").astype(str).values
                             if lbl is not None else np.full(len(edf), "Unknown", dtype=object))
            return self._categorical(np.concatenate(parts), "Cell type")

        if mode == "gene":
            parts = []
            # corrected counts only when every sample has them (as make_spatial_fig)
            mc = dataview._multi_use_corrected(corrected)
            for di, edf in enumerate(dfs):
                vals = (dataview._xenium_gene_values(gene, mc) if di == 0
                        else dataview._extra_gene_values(extras[di - 1], gene, mc))
                parts.append(np.asarray(vals, dtype=float) if vals is not None and len(vals) == len(edf)
                             else np.zeros(len(edf)))
            return self._continuous(np.concatenate(parts),
                                    f"{gene} (log1p{', SPLIT-corrected' if mc else ''})")

        if mode.startswith("roi_") and any(mode in edf.columns for edf in dfs):
            vals = pd.concat([
                (edf[mode] if mode in edf.columns else pd.Series(pd.NA, index=edf.index, dtype=object))
                for edf in dfs]).fillna("(none)").astype(str)
            named = [v for v in sorted(vals.unique()) if v != "(none)"]
            return self._categorical(vals.values, mode[4:] or "ROI", order=named + ["(none)"],
                                     colors=[regions._roi_color(i) for i in range(len(named))] + ["#666666"])

        col = mode if any(mode in edf.columns for edf in dfs) else "transcript_counts"
        raw = np.concatenate([
            pd.to_numeric(edf[col], errors="coerce").values.astype(np.float64) if col in edf.columns
            else np.zeros(len(edf)) for edf in dfs])
        if col == "transcript_counts":
            return self._continuous(np.log1p(raw), "Transcript Counts (log1p)")
        return self._continuous(raw, core.QC_METRICS.get(col, col))

    def boundaries(self, source, kind, x0, y0, x1, y1):
        """Polygons from the in-memory boundary dicts, or None to let the route read the
        bundle's cells.zarr.zip (possible only when DATA["df"] holds exactly the bundle's
        cells in file order, i.e. Xenium boundaries and no active subset)."""
        # The kind picks the polygon set whatever the active source is, as the four
        # checkboxes do in the Plotly view: "cell" / "nucleus" are the original Xenium
        # boundaries, "baysor" / "proseg" that tool's loaded run.
        if kind in ("cell", "nucleus"):
            df, slot = core.DATA["df"], ("xenium", kind)
            d = core.DATA.get("data_dir")
            if (d and "_df_original" not in core.DATA
                    and os.path.exists(os.path.join(d, "cells.zarr.zip"))
                    and self._zip_matches(d, df, kind)):
                return None
            bounds = core.DATA.get("cell_bounds" if kind == "cell" else "nucleus_bounds") or {}
        elif kind in core._SEG_TOOL_LABELS:
            alt = self._alt_res(kind)
            if alt is None:
                return {"n": 0, "truncated": False, "available": False}, {}
            df, slot = alt["cells_df"], (kind, "cell")
            bounds = alt.get("cell_bounds") or {}
        else:
            return {"n": 0, "truncated": False, "available": False}, {}
        if not bounds:
            return {"n": 0, "truncated": False, "available": False}, {}
        pad = self._BOUNDARY_PAD
        cx, cy = df["x_centroid"].values, df["y_centroid"].values
        rows = np.flatnonzero((cx >= x0 - pad) & (cx <= x1 + pad) & (cy >= y0 - pad) & (cy <= y1 + pad))
        if len(rows) > self._BOUNDARY_LIMIT:
            return {"n": int(len(rows)), "truncated": True, "available": True}, {}
        xy_all, ring_start, poly_ring, row_poly = self._packed_bounds(slot, bounds, df)
        # rings of the rows in view, gathered by index arithmetic (no per-cell loop)
        ent = row_poly[rows]
        rows, ent = rows[ent >= 0], ent[ent >= 0]
        n_rings = poly_ring[ent + 1] - poly_ring[ent]
        total = int(n_rings.sum())
        if total == 0:
            return {"n": 0, "truncated": False, "available": True}, {}
        rings = np.repeat(poly_ring[ent] - (np.cumsum(n_rings) - n_rings), n_rings) + np.arange(total)
        n_vert = ring_start[rings + 1] - ring_start[rings]
        starts = np.zeros(total + 1, dtype=np.int64)
        np.cumsum(n_vert, out=starts[1:])
        verts = np.repeat(ring_start[rings] - starts[:-1], n_vert) + np.arange(int(starts[-1]))
        return ({"n": total, "truncated": False, "available": True},
                {"start": starts.astype(np.uint32), "xy": xy_all[verts].ravel(),
                 "cell": np.repeat(rows, n_rings).astype(np.uint32)})

    def forget(self):
        """Drop what was derived from the previous dataset (called on Load Sample)."""
        self._packed.clear()
        self._zip_ok.clear()

    def _zip_matches(self, d, df, kind):
        """True when cells.zarr.zip can stand in for the in-memory polygons: it has readable
        polygons of *kind*, and DATA["df"] holds exactly its cells in file order (a
        spatialdata table written or filtered outside the app may not). Checked once per
        dataset and frame."""
        key = (jobs._dataset_gen, d, len(df), kind)
        hit = self._zip_ok.get(kind)
        if hit is not None and hit[0] == key and hit[1] is df:
            return hit[2]
        ok = False
        try:
            import xenium_viewer_api as _va
            store = _va.tile_store(d)
            ids = store.cell_ids()
            if ids is not None and len(ids) == len(df) and store._polygons(kind) is not None:
                pick = np.unique(np.linspace(0, len(df) - 1, min(len(df), 512)).astype(int))
                if ids.ndim == 2:   # (prefix, dataset suffix) -> "aaaaljij-1"
                    names = ["".join(chr(97 + int(c, 16)) for c in f"{int(p):08x}") + f"-{int(s)}"
                             for p, s in ids[pick]]
                else:               # older bundles: integer cell ids
                    names = [str(int(v)) for v in ids[pick]]
                ok = names == [str(v) for v in df.index[pick]]
        except Exception:
            ok = False
        self._zip_ok[kind] = (key, df, ok)
        return ok

    def _packed_bounds(self, slot, bounds, df):
        """A {cell id: (xs, ys)} boundary dict packed once into flat arrays:
        (xy (V, 2) float32, ring_start (R + 1), poly_ring (P + 1), row_poly (len(df))).
        Multi-ring cells (several nuclei) are stored NaN-separated and become one ring
        each; rings with fewer than 3 vertices are dropped. row_poly maps a row of *df* to
        its dict entry (-1 = none). Rebuilt when the dict or the frame is replaced."""
        hit = self._packed.get(slot)
        if hit is None or hit["bounds"] is not bounds or hit["n"] != len(bounds):
            polys = [p if p is not None else ((), ()) for p in bounds.values()]
            n_pts = np.fromiter((len(p[0]) for p in polys), dtype=np.int64, count=len(polys))
            if n_pts.sum():
                x = np.concatenate([p[0] for p in polys], dtype=np.float32, casting="unsafe")
                y = np.concatenate([p[1] for p in polys], dtype=np.float32, casting="unsafe")
            else:
                x = y = np.zeros(0, dtype=np.float32)
            poly_of = np.repeat(np.arange(len(polys)), n_pts)
            gap = np.isnan(x)
            first = np.zeros(len(x), dtype=bool)          # vertex that opens a ring
            first[(np.cumsum(n_pts) - n_pts)[n_pts > 0]] = True
            first[1:] |= gap[:-1]
            ring_of = np.cumsum(first) - 1
            ring_len = np.bincount(ring_of[~gap], minlength=int(first.sum()))
            keep = ~gap & (ring_len[ring_of] >= 3)
            good = ring_len >= 3
            ring_poly = np.zeros(len(ring_len), dtype=np.int64)
            ring_poly[ring_of] = poly_of
            ring_start = np.zeros(int(good.sum()) + 1, dtype=np.int64)
            np.cumsum(ring_len[good], out=ring_start[1:])
            poly_ring = np.searchsorted(ring_poly[good], np.arange(len(polys) + 1))
            hit = {"bounds": bounds, "n": len(bounds), "df": None, "row_poly": None,
                   "arrays": (np.stack([x[keep], y[keep]], axis=1), ring_start, poly_ring)}
            self._packed[slot] = hit
        if hit["df"] is not df:
            keys = pd.Index(list(bounds.keys()))
            row_poly = keys.get_indexer(df.index)
            miss = row_poly < 0
            if miss.any():      # ids kept as strings in the dict, numbers in the frame
                row_poly[miss] = keys.get_indexer(df.index[miss].astype(str))
            hit["df"], hit["row_poly"] = df, row_poly
        return (*hit["arrays"], hit["row_poly"])

    def overlays(self, source, args=None):
        """Outlines for the viewer, in µm with y down: saved ROIs (stored in plot space,
        y = -µm), the SpatialData tissue ROI / patch grid when asked for with sdata_roi=1 /
        sdata_patches=1, and the separators between samples."""
        args = args or {}
        with core._roi_lock:
            rois = list(core._roi_state["rois"]) if core._roi_state.get("show", True) else []
        polys, lines = figures._provided_overlays(figures._overlay_polygon_providers, source), []
        for r in rois:
            pts = [[float(x), -float(y)] for x, y in r.get("polygon_xy") or []]
            if len(pts) >= 3:
                # line width and fill opacity (0.18) of _roi_shapes_for_fig
                polys.append({"points": pts, "color": r.get("color", "#ffffff"), "width": 1.5,
                              "fill": r.get("color", "#ffffff"), "fillAlpha": 46,
                              "label": f"{r.get('name', '')} ({r.get('cls', '')})"})
        show_roi = args.get("sdata_roi") == "1"
        show_patches = args.get("sdata_patches") == "1"
        if show_roi or show_patches:        # same geometry as _sdata_overlays_to_shapes
            with core._sdata_lock:
                roi = core._sdata_state.get("roi")
                patches = core._sdata_state.get("patches")
            if show_roi and roi is not None:
                try:
                    for geom in sdata_store._sdata_transform_to_um(roi).geometry:
                        polys.append({"points": [[float(c[0]), float(c[1])] for c in geom.exterior.coords],
                                      "color": "#52b788", "width": 2,
                                      "fill": "#52b788", "fillAlpha": 13})
                except Exception as exc:
                    print(f"  ROI overlay error: {exc}", flush=True)
            if show_patches and patches is not None:
                try:
                    for geom in sdata_store._sdata_transform_to_um(patches).geometry:
                        bx0, by0, bx1, by1 = (float(v) for v in geom.bounds)
                        polys.append({"points": [[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]],
                                      "color": "#ffaa00", "width": 1, "alpha": 153,   # line opacity 0.6
                                      "fill": "#ffaa00", "fillAlpha": 8})
                except Exception as exc:
                    print(f"  Patch overlay error: {exc}", flush=True)
        # middle of each gap, full height, as in Plotly
        for sep_x in (dataview._sample_separator_xs() if self._extras(self._alt_res(source)) else []):
            lines.append({"from": [sep_x, -1e6], "to": [sep_x, 1e6], "color": core.MUTED, "width": 1})
        return {"polygons": polys, "lines": lines}

    def cell_at(self, source, index):
        df, alt = self._frame(source)
        offset, row, sample = 0.0, int(index), 0
        for si, eds in enumerate(self._extras(alt), start=1):   # rows past the primary sample belong to the extra ones
            if index < len(df):
                break
            index -= len(df)
            df, offset, sample = eds["df"], float(eds["x_offset"]), si
        if index < 0 or index >= len(df):
            return {}
        # "sample": 0 = primary, n = n-th extra sample (ids are unique within a sample only)
        return {"index": row, "id": str(df.index[index]), "sample": sample,
                "x": float(df["x_centroid"].iloc[index]) + offset,
                "y": float(df["y_centroid"].iloc[index])}

    # ── app-specific binary payloads: /viewer/extra/<name>.bin ─────────────────
    def extra(self, name, args):
        if name == "pies":
            return self._pies(args)
        if name == "cellnest":
            return self._cellnest(args)
        if name == "density":
            return self._density(args)
        return None

    def _density(self, args):
        """Transcript-density grid of one gene for the box (µm, y down): float32 values
        (transcripts per bin, row-major, `h` rows of `w`) and where they lie. `ready`
        is False while the gene's grid has not been computed (the Compute button)."""
        gene = args.get("gene") or ""
        try:
            bin_um = float(args.get("bin") or 10)
            box = [float(args[k]) for k in ("x0", "y0", "x1", "y1")]
        except (KeyError, TypeError, ValueError):
            return {"available": False, "ready": False, "w": 0, "h": 0}, {}
        # One grid per sample of the view (the primary one and the extra samples beside
        # it), each at its sample's offset: `parts` lists them, `values` holds them one
        # after the other. w / h / x0 / y0 / cell repeat the first non-empty part (all
        # there is with a single sample).
        have_dir = any(s["data_dir"] for s in analysis._density_samples())
        if not gene or not have_dir:
            return {"available": have_dir, "ready": False, "w": 0, "h": 0}, {}
        wins = analysis._density_windows(gene, bin_um, *box)
        if wins is None:
            return {"available": True, "ready": False, "w": 0, "h": 0}, {}
        parts, vals, pos = [], [], 0
        for win in wins["parts"]:
            v = win["values"]
            parts.append({"h": int(v.shape[0]), "w": int(v.shape[1]), "x0": float(win["x0"]),
                          "y0": float(win["y0"]), "cell": float(win["cell"]), "offset": pos,
                          "label": win["label"]})
            vals.append(v.ravel())
            pos += v.size
        first = next((p for p in parts if p["w"] and p["h"]), parts[0])
        return ({"available": True, "ready": True, "h": first["h"], "w": first["w"],
                 "x0": first["x0"], "y0": first["y0"], "cell": first["cell"], "parts": parts,
                 "bin": float(wins["bin"]), "vmax": float(wins["vmax"]), "total": int(wins["total"]),
                 "coarsened": bool(wins["coarsened"]),
                 # (zooming in then fetches the bins themselves, see RegionLoader.refinable)
                 "subsampled": bool(wins["coarsened"]),
                 "label": analysis._density_label(gene, wins["bin"]),
                 # a categorical layer (values 1 … n index the palette, 0 = empty)
                 **({"palette": list(wins["palette"]), "names": list(wins.get("names") or [])}
                    if wins.get("palette") else {})},
                {"values": np.concatenate(vals) if len(vals) > 1 else vals[0]})

    def _pies(self, args):
        """RCTD weights of the cells in the box, for the weight pies the Plotly view draws
        (_rctd_pie_shapes_for_view): single-sample Xenium view, a cell_type colour mode with
        stored weights, at most PIE_THRESHOLD cells. The browser builds the wedges."""
        none = ({"n": 0, "available": False}, {})
        mode = args.get("mode") or ""
        if (core._seg_tool(args.get("source", "xenium")) != "xenium" or self._extras()
                or not mode.startswith("cell_type:")):
            return none
        lkey = f"labels_{mode.split(':')[1]}"
        with core._annot_lock:
            weights = core._annot_state.get(f"rctd_weights_{lkey}")
        if weights is None:
            return none
        try:
            x0, y0, x1, y1 = (float(args[k]) for k in ("x0", "y0", "x1", "y1"))
        except (KeyError, ValueError):
            return none
        df = core.DATA["df"]
        cx, cy = df["x_centroid"].values, df["y_centroid"].values
        rows = np.flatnonzero((cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1))
        if len(rows) > core.PIE_THRESHOLD:
            return {"n": int(len(rows)), "available": True, "truncated": True}, {}
        w = weights.reindex(df.index[rows].astype(str)).fillna(0).values.astype(float)
        total = w.sum(axis=1) if w.size else np.zeros(len(rows))
        keep = total > 0
        rows, w = rows[keep], w[keep] / total[keep, None]
        cmap = figures._cell_type_color_map(lkey, df)     # colours of the full frame, as the cells have
        xy = np.empty(2 * len(rows), dtype=np.float32)
        xy[0::2] = cx[rows]
        xy[1::2] = cy[rows]
        return ({"n": int(len(rows)), "available": True, "truncated": False,
                 "types": [str(c) for c in weights.columns],
                 "colors": [cmap.get(c, "#888888") for c in weights.columns],
                 "radius": core.RCTD_PIE_RADIUS_UM, "min_frac": 0.02},
                {"xy": xy, "frac": w.astype(np.float32).ravel()})

    def _cellnest(self, args):
        """CellNEST edges for the map as line segments (shaft plus two arrowhead strokes
        per edge), selected and coloured as _cellnest_overlay_traces does. The controls
        arrive as JSON in ``q``. With the gray-out option the rows of the participating
        cells in the viewer's table are returned as ``cells``."""
        none = ({"n": 0, "edges": 0, "available": False}, {})
        try:
            q = json.loads(args.get("q") or "{}")
        except ValueError:
            return none
        if not isinstance(q, dict):
            return none
        with core._cellnest_lock:
            data = core._cellnest_state.get("data") if core._cellnest_state.get("status") == "done" else None
        pairs = q.get("pairs") or []
        pairs = pairs if isinstance(pairs, list) else [pairs]
        if data is None or not pairs:
            return none
        opts = q.get("options") or []
        color_by = q.get("color_by") or "lr_pair"
        try:
            top_n, edge_width = int(q.get("top_n") or 100), float(q.get("edge_width") or 1)
        except (TypeError, ValueError):
            top_n, edge_width = 100, 1.0
        ccc_df = cellnest._apply_cellnest_filters(data["ccc_df"], q.get("ligand"), q.get("receptor"),
                                         q.get("sender"), q.get("receiver"), q.get("annot_col"))

        def _rgb(color):
            if str(color).startswith("#"):
                return [int(color[i:i + 2], 16) for i in (1, 3, 5)]
            return [int(round(float(v))) for v in re.findall(r"[\d.]+", str(color))[:3]]

        palette = qualitative.Plotly
        lig_cmap = rec_cmap = viridis = None
        if color_by == "ligand":
            lig_cmap = {l: palette[i % len(palette)] for i, l in enumerate(sorted(ccc_df["ligand"].unique()))}
        elif color_by == "receptor":
            rec_cmap = {r: palette[i % len(palette)] for i, r in enumerate(sorted(ccc_df["receptor"].unique()))}
        elif color_by == "attention":
            import plotly.colors as _pc
            viridis = np.array([_rgb(c) for c in _pc.sample_colorscale("Viridis", list(np.linspace(0, 1, 10)))])
        seg_xy, seg_rgba, seg_w, ids, n_edges = [], [], [], set(), 0
        for pi, pair in enumerate(pairs):
            ligand, receptor = cellnest._cellnest_split_pair(pair, ccc_df)
            sub = ccc_df[(ccc_df["ligand"] == ligand) & (ccc_df["receptor"] == receptor)]
            sub = sub.nlargest(top_n, "attention_score")
            if sub.empty:
                continue
            n = len(sub)
            n_edges += n
            ids.update(sub["from_app_id"].astype(str))
            ids.update(sub["to_app_id"].astype(str))
            score = sub["attention_score"].values.astype(float)
            s01 = (score - score.min()) / ((score.max() - score.min()) or 1.0)
            if viridis is not None:
                rgb = viridis[np.minimum(9, (s01 * 10).astype(int))]
            else:
                col = palette[pi % len(palette)]
                if lig_cmap is not None:
                    col = lig_cmap.get(ligand, col)
                elif rec_cmap is not None:
                    col = rec_cmap.get(receptor, col)
                rgb = np.tile(_rgb(col), (n, 1))
            width = edge_width * (0.5 + 1.5 * s01) if "edge_weight" in opts else np.full(n, edge_width)
            a = sub[["from_x", "from_y"]].values.astype(float)
            b = sub[["to_x", "to_y"]].values.astype(float)
            # arrowhead: two strokes at the target end, 30 % of the edge long, 25° off it
            d = (a - b) * 0.3
            cs, sn = np.cos(np.radians(25)), np.sin(np.radians(25))
            h1 = b + np.stack([d[:, 0] * cs - d[:, 1] * sn, d[:, 0] * sn + d[:, 1] * cs], axis=1)
            h2 = b + np.stack([d[:, 0] * cs + d[:, 1] * sn, -d[:, 0] * sn + d[:, 1] * cs], axis=1)
            for p0, p1, alpha in ((a, b, 102), (b, h1, 178), (b, h2, 178)):   # opacity 0.4 / 0.7
                seg_xy.append(np.hstack([p0, p1]))
                seg_rgba.append(np.hstack([rgb, np.full((n, 1), alpha)]))
                seg_w.append(width)
        if not seg_xy:
            return {"n": 0, "edges": 0, "available": True}, {}
        arrays = {"xy": np.concatenate(seg_xy).astype(np.float32).ravel(),
                  "rgba": np.concatenate(seg_rgba).astype(np.uint8).ravel(),
                  "width": np.concatenate(seg_w).astype(np.float32)}
        gray = "gray_bg" in opts
        if gray:
            df, _ = self._frame(args.get("source", "xenium"))
            arrays["cells"] = np.flatnonzero(df.index.astype(str).isin(ids)).astype(np.uint32)
        return ({"n": int(len(arrays["width"])), "edges": int(n_edges), "available": True,
                 "gray": gray}, arrays)


class _UmapViewerProvider(_ViewerProvider):
    """Same cells and colours as the spatial provider, positioned by their UMAP embedding.
    The viewer's source carries the counts mode as '<seg-source>|corr' so the corrected
    embedding (split_umap_*) is served when corrected counts are active. Like the Plotly
    UMAP it shows the primary sample only when extra samples are loaded."""

    _multi_sample = False

    @staticmethod
    def _split(source):
        parts = (source or "xenium").split("|")
        return parts[0] or "xenium", "corr" in parts[1:]

    @classmethod
    def _joint(cls, source):
        """The joint embedding of all samples when the source names one ('…|emb:<cluster
        set>', sent while a joint cluster set is selected) and it exists; else None."""
        for part in (source or "").split("|")[1:]:
            if part.startswith("emb:"):
                return joint._jc_umap_view(part[4:], cls._split(source)[0])
        return None

    def _umap_xy(self, source):
        base, corrected = self._split(source)
        df, alt = self._frame(base)
        n = len(df)
        x = np.full(n, np.nan)
        y = np.full(n, np.nan)
        stored = None
        if alt is not None:
            # a "Rerun UMAP" result belongs to one run and one counts mode
            with core._umap_reseg_lock:
                if (core._umap_reseg_state.get("status") == "done"
                        and core._umap_reseg_state.get("owner") == alt.get("out_dir")
                        and (core._umap_reseg_state.get("counts_mode", "original") == "corrected") == corrected):
                    stored = core._umap_reseg_state.get("result")
        cols = None
        frame = df
        if stored is not None and "umap_1" in getattr(stored, "columns", []):
            frame = stored.copy()
            frame.index = frame.index.astype(str)
            frame = frame[~frame.index.duplicated()].reindex(df.index.astype(str))
            cols = ("split_umap_1", "split_umap_2") if (corrected and "split_umap_1" in frame.columns
                                                         and frame["split_umap_1"].notna().any()) \
                else ("umap_1", "umap_2")
        elif corrected and "split_umap_1" in df.columns and df["split_umap_1"].notna().any():
            cols = ("split_umap_1", "split_umap_2")
        elif "umap_1" in df.columns and "umap_2" in df.columns:
            cols = ("umap_1", "umap_2")
        if cols is not None:
            x = pd.to_numeric(frame[cols[0]], errors="coerce").to_numpy(dtype=float)
            y = -pd.to_numeric(frame[cols[1]], errors="coerce").to_numpy(dtype=float)   # viewer y is down
        return x, y

    @staticmethod
    def _xy_bounds(x, y):
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.any():
            return [0.0, 0.0, 1.0, 1.0]
        return [float(x[ok].min()), float(y[ok].min()), float(x[ok].max()), float(y[ok].max())]

    def data_dir(self):
        return None            # no tiles, boundaries or images in embedding space

    def image_files(self):
        return []

    def meta(self):
        x, y = self._umap_xy("xenium")
        return {"dataset": caching._dataset_name(), "dataset_gen": jobs._dataset_gen, "sources": ["xenium"],
                "bounds": self._xy_bounds(x, y), "pixel_size": 1.0, "has_boundaries": False,
                "has_transcripts": False, "images": [], "embedding": "umap"}

    def cells(self, source):
        jv = self._joint(source)
        if jv is not None:       # all samples, in the order of the spatial view
            x, y = np.concatenate(jv["x"]), -np.concatenate(jv["y"])
        else:
            x, y = self._umap_xy(source)
        xy = np.empty(2 * len(x), dtype=np.float32)
        xy[0::2] = x
        xy[1::2] = y
        return ({"n": int(len(x)), "bounds": self._xy_bounds(x, y),
                 "n_embedded": int(np.isfinite(x).sum()), "samples": len(jv["tables"]) if jv is not None else 1,
                 "token": f"umap:{jobs._dataset_gen}:{core._subset_version}:{source}:{len(x)}"}, {"xy": xy})

    def colors(self, args):
        args = dict(args)
        joint_view = self._joint(args.get("source")) is not None
        base, corrected = self._split(args.get("source"))
        args["source"] = base
        if joint_view:           # the colours of every sample, as the spatial view has them
            return _viewer_provider.colors(args)
        return super().colors(args)

    def boundaries(self, source, kind, x0, y0, x1, y1):
        return {"n": 0, "truncated": False, "available": False}, {}

    def overlays(self, source, args=None):
        return {"polygons": [], "lines": []}

    def extra(self, name, args):
        return None

    def cell_at(self, source, index):
        jv = self._joint(source)
        if jv is not None:
            pos = int(index)
            for t in jv["tables"]:
                if pos < len(t["df"]):
                    return ({"index": int(index), "id": str(t["df"].index[pos]), "sample": t["label"]}
                            if pos >= 0 else {})
                pos -= len(t["df"])
            return {}
        df, _ = self._frame(self._split(source)[0])
        if index < 0 or index >= len(df):
            return {}
        return {"index": int(index), "id": str(df.index[index])}


_viewer_provider = _ViewerProvider()
try:
    import xenium_viewer_api as _viewer_api
    _viewer_api.register(app_layout.app.server, _viewer_provider)
    _viewer_api.register(app_layout.app.server, _UmapViewerProvider(), prefix="/umap")
except Exception as _viewer_exc:   # the Plotly view keeps working without the WebGL viewer
    print(f"  WebGL viewer endpoints unavailable: {_viewer_exc}", flush=True)


# ─── Health check endpoint (for Cloud Run / load balancers) ──────────────────
@app_layout.app.server.route("/health")
def health():
    return "ok"
