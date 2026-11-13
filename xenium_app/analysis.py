"""Differential expression, gene-set scores and transcript density (compute + tab layouts)."""

import os
import re
import json
import threading
import hashlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from dash import dcc, html
from dash import dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import time as _time
from collections import OrderedDict as _OrderedDict

from . import core, jobs, caching, ui, dataview, annotation, reseg, figures, regions


# ─── Analysis: differential expression, gene-set scores, transcript density ───
# All three read the cells / counts of the *active view*: the active segmentation
# (Xenium, or the loaded Baysor / Proseg run), the active counts mode (original or
# SPLIT-corrected), the current subset and, for the Xenium source, the extra samples
# shown beside the primary one. _an_samples() describes that view.

_AN_SCALE = 1e4            # library-size normalisation target (counts per 10,000)

_de_state: dict = {"status": "idle", "message": "", "result": None,
                   "groups": [],          # saved groups: [{name, polygon_xy}] (plot space)
                   "selection": None}     # last lasso / box outline: {polygon_xy, n, t}
_de_lock = threading.Lock()
_DE_MIN_CELLS = 10         # smallest group a test is run for
# Detection rule used wherever "expressed" / "% of cells" is computed: a cell detects a
# gene when its count is at least one whole transcript. For integer counts this is the
# usual count > 0; for SPLIT-corrected (fractional) counts a residue below one transcript
# is not a detection. The depth-robust companion is the rate Σ counts / Σ depth.
_DETECT_MIN = 1.0
_DETECT_RULE = ("detected = count ≥ 1 transcript (the same as > 0 for integer counts; a fractional "
                "SPLIT-corrected residue below 1 is not a detection); rate = Σ counts / Σ total "
                "counts of the group, per 10,000")
_DE_DEPTH_RATIO = 1.5      # median counts per cell of the two groups differing by more is flagged
_DE_RATE_MIN_COUNT = 100   # Σ counts of a gene over both groups below which its rate is unstable
_DE_PB_KINDS = ("pseudobulk", "contact")     # results fitted on pseudobulks: same table and volcano rules
_DE_FDR_ONLY_KINDS = _DE_PB_KINDS + ("proximity",)   # volcano colour by FDR alone
# functions(res, ctx) that add columns / notes to a finished DE result before it is shown
# (the leak gate of the controls module registers here); ctx: samples, masks_a, masks_b,
# seg_source, method, color_by, use_corrected, options
_de_annotators: list = []
# other per-cell values offered under Color By through the gene-set path: name prefix ->
# function(name, seg_source, use_corrected) -> (values over the cells of the view, (lo, hi)
# or None for a diverging scale centred at 0, label) or None
_view_color_providers: dict = {}
_view_color_seq: dict = {}  # label -> (lo, hi) of a sequential (not diverging) colour value
_view_color_options: list = []   # functions() -> [{"label", "value"}] of such Color By entries


def _view_color_extra_options() -> list:
    """Color By options of the registered per-cell values (after the gene sets)."""
    out = []
    for fn in list(_view_color_options):
        try:
            out += list(fn())
        except Exception as exc:
            print(f"  Color By options error: {exc}", flush=True)
    return out

_geneset_state: dict = {"status": "idle", "message": "",
                        "sets": {},       # name -> [genes as entered]
                        "version": 0}     # bumped when a set is added / removed / rescored
_geneset_lock = threading.Lock()
_geneset_cache: "_OrderedDict" = _OrderedDict()     # (view key, name, genes) -> (arrays, info)
_geneset_compute_lock = threading.Lock()
_an_means_cache: "_OrderedDict" = _OrderedDict()    # view key -> (genes, mean log-norm expr)

_density_state: dict = {"status": "idle", "message": "", "version": 0}
_density_lock = threading.Lock()
_DENSITY_BASE_UM = 5.0     # grids are stored at this bin size; coarser bins are exact sums
_DENSITY_BINS = (5, 10, 20, 50)
_DENSITY_FILTER_VERSION = "qv20_genes_v1"   # part of the cache file name
_DENSITY_ALL = "__all__"   # "gene" value of the all-gene-transcripts grid
_DENSITY_MAX_CELLS = 360_000   # heat-layer cells sent per request (coarsened above this)
_density_mem: "_OrderedDict" = _OrderedDict()       # (cache file, bin) -> dense grid
_density_mem_lock = threading.Lock()                # (read by request threads)
# Heat layers that are not the transcript grid of a gene (e.g. the transcript factors):
# "gene" prefix -> {"grid": f(gene, ds) -> grid dict as _density_grid returns (plus
# "palette" / "names" for a categorical layer whose values are 1 … n, 0 = empty) or None,
# "label": f(gene, bin) -> str, "options": f() -> dropdown options, "hint": f(gene) -> str}
_density_virtual_providers: dict = {}


def _density_virtual(gene):
    """The provider of a heat-layer "gene" that is not a gene, or None."""
    g = str(gene or "")
    for prefix, prov in list(_density_virtual_providers.items()):
        if g.startswith(prefix):
            return prov
    return None


def _an_dataset_tag() -> str:
    return hashlib.md5(caching._dataset_key().encode()).hexdigest()[:8]


def _an_alt_res(seg_source):
    """The loaded Baysor / Proseg / image-segmentation result when *seg_source* names that tool, else None."""
    return core._seg_result(seg_source)


def _an_seg_missing(seg_source) -> bool:
    """*seg_source* names a Baysor / Proseg / image-segmentation run whose result is not
    in memory (still loading from the cache, or the load failed)."""
    tool = core._seg_tool(seg_source)
    return tool != "xenium" and core._seg_state(tool)[0] is not None and _an_alt_res(seg_source) is None


def _an_require_seg(seg_source) -> None:
    """ValueError when *seg_source* names a segmentation run that is not loaded (a job
    then stops instead of computing on the Xenium cells under that run's name)."""
    if _an_seg_missing(seg_source):
        raise ValueError(_an_no_matrix_msg(seg_source))


def _an_no_matrix_msg(seg_source=None) -> str:
    """Why _an_samples() gave nothing."""
    if seg_source is not None and _an_seg_missing(seg_source):
        return (f"The selected segmentation ({seg_source}) is not loaded (it is still loading, or its "
                f"load failed): wait for it or pick it again. Nothing was computed on the Xenium cells "
                f"in its place.")
    return "The active view has no expression matrix."


def _ds_panel_genes(ds: dict) -> list:
    """Measured (not SpaGE-imputed) genes of a dataset dict (DATA or an extra sample)."""
    names = list(ds.get("gene_names") or [])
    gv = ds.get("gene_var")
    if gv is not None and "is_imputed" in gv.columns:
        imp = set(gv.index[gv["is_imputed"].astype(bool)])
        return [g for g in names if g not in imp]
    return names


def _an_samples(seg_source="xenium", use_corrected: bool = False) -> list:
    """The cell tables of the active view with the count matrix behind each.

    One dict per sample: df (cells, as drawn), csc (cells × genes counts, CSC), rows
    (matrix row of each df row, -1: none - with the SPLIT-corrected matrix also a cell
    that has no corrected counts; n_uncorrected counts those), gni (gene → matrix column), genes (measured
    panel genes present in the matrix; imputed genes are left out, their values are not
    counts), x_offset (µm the sample is shifted by in the view), corrected (the matrix
    is the SPLIT-corrected one), label, ds (the dataset dict the sample comes from: DATA
    or an extra sample), key / name (its identity and display name).

    With several samples, SPLIT-corrected counts are used only when every sample has
    them (dataview._corrected_status); otherwise all samples give their original counts,
    so one test or colour scale never mixes the two.

    Empty when *seg_source* names a segmentation run that is not loaded: the Xenium cells
    are never returned in its place (_an_no_matrix_msg says so)."""
    alt = _an_alt_res(seg_source)
    out = []
    if alt is None and _an_seg_missing(seg_source):
        # never the Xenium cells under the name of another segmentation
        return out

    def _add(label, df, holder, key, rows, gni, panel, x_off, ds=None):
        m = holder.get(key)
        if m is None or not sp.issparse(m):
            return
        if holder is core.DATA and key == "expr" and core.DATA.get("expr_csc") is not None:
            csc = core.DATA["expr_csc"]
        else:
            csc = figures._csc_view(holder, key)
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows) != len(df) or (len(rows) and rows.max() >= csc.shape[0]):
            return
        genes = [g for g in panel if g in gni and gni[g] < csc.shape[1]]
        ds = core.DATA if ds is None else ds
        n_unc = 0
        if key == "split_corrected_expr":
            # Cells without corrected counts (dropped by RCTD, or outside the subset SPLIT
            # ran on) have an empty padding row: they are not in this view of the matrix.
            cov = dataview._corrected_covered(holder)
            if cov is not None and len(cov) == csc.shape[0]:
                drop = (rows >= 0) & ~cov[np.clip(rows, 0, None)]
                n_unc = int(drop.sum())
                if n_unc:
                    rows = np.where(drop, -1, rows)
        out.append({"label": label, "df": df, "csc": csc, "rows": rows, "gni": gni,
                    "genes": genes, "x_offset": float(x_off or 0.0),
                    "corrected": key == "split_corrected_expr", "n_uncorrected": n_unc,
                    "ds": ds, "key": dataview._ds_key(ds), "name": dataview._ds_name(ds)})

    if alt is not None:
        df = alt["cells_df"]
        key = ("split_corrected_expr"
               if use_corrected and alt.get("split_corrected_expr") is not None else "expr")
        gni = alt.get("gene_name_to_idx") or core.DATA.get("gene_name_to_idx", {})
        _add(core._seg_tool(seg_source), df, alt, key, np.arange(len(df)), gni,
             dataview._panel_gene_names(), 0.0)
        return out

    df = core.DATA["df"]
    extras = list(dataview.EXTRA_DATASETS)
    # several samples side by side: corrected counts only when every sample has them
    use_corr = (dataview._multi_use_corrected(use_corrected) if extras
                else bool(use_corrected) and core.DATA.get("split_corrected_expr") is not None)
    if core.DATA.get("expr") is not None:
        key = "split_corrected_expr" if use_corr else "expr"
        d2e = core.DATA.get("df_to_expr")
        d2e = np.arange(len(df)) if d2e is None else np.asarray(d2e)
        rows = d2e
        if key != "expr" and core.DATA[key].shape[0] != core.DATA["expr"].shape[0]:
            rows = np.arange(len(df)) if core.DATA[key].shape[0] == len(df) else np.full(len(df) + 1, -1)
        _add("S0" if extras else "xenium", df, core.DATA, key, rows,
             core.DATA.get("gene_name_to_idx", {}), dataview._panel_gene_names(), 0.0)
    for i, eds in enumerate(extras):
        if eds.get("expr") is None:
            continue
        d2e = eds.get("df_to_expr")
        _add(f"S{i + 1}", eds["df"], eds, "split_corrected_expr" if use_corr else "expr",
             np.arange(len(eds["df"])) if d2e is None else d2e,
             eds.get("gene_name_to_idx", {}), _ds_panel_genes(eds), eds.get("x_offset", 0.0),
             ds=eds)
    return out


def _an_uncorrected(samples) -> int:
    """Cells of the view left out because they have no SPLIT-corrected counts."""
    return int(sum(int(s.get("n_uncorrected") or 0) for s in samples or []))


def _an_uncorrected_note(n) -> str:
    """Sentence for status lines and CSV headers ("" when no cell was left out)."""
    n = int(n or 0)
    if not n:
        return ""
    return (f"{n:,} cell{'s' if n != 1 else ''} without SPLIT-corrected counts (dropped by the "
            f"correction or outside the cells it was run on) {'were' if n != 1 else 'was'} left out")


def _an_matrix_digest(m) -> str:
    """Content fingerprint of a count matrix for disk-cache keys: shape, number of stored
    values and a CRC of the values and of their positions. Two matrices of the same cells
    with different counts (SPLIT run again with other settings) never share it; shape and
    nnz alone can coincide."""
    import zlib
    parts = [str(m.shape), str(int(m.nnz))]
    for a in (m.data, m.indices, m.indptr):
        parts.append(format(zlib.crc32(memoryview(np.ascontiguousarray(a)).cast("B")) & 0xFFFFFFFF, "08x"))
    return "|".join(parts)


def _an_view_key(seg_source, use_corrected, samples) -> tuple:
    """Identifies the cells and matrices of a view (for the caches below)."""
    return (jobs._dataset_gen, core._subset_version, str(seg_source or "xenium"), bool(use_corrected),
            tuple((id(s["csc"]), len(s["df"]), s["corrected"]) for s in samples))


def _an_common_genes(samples) -> list:
    """Panel genes every sample of the view has, in the first sample's order."""
    if not samples:
        return []
    genes = samples[0]["genes"]
    for s in samples[1:]:
        have = set(s["genes"])
        genes = [g for g in genes if g in have]
    return genes


def _an_size_factors(s: dict, genes=None):
    """(sf, in_view) per matrix row of sample *s*: sf = _AN_SCALE / library size, where
    the library size is the cell's total count over the measured panel genes of the
    active matrix; 0 for rows that are not in the view (subset) or have no counts."""
    csc = s["csc"]
    cols = np.array([s["gni"][g] for g in (s["genes"] if genes is None else genes)], dtype=np.int64)
    if len(cols) == csc.shape[1]:
        lib = np.asarray(csc.sum(axis=1), dtype=np.float64).ravel()
    else:
        lib = np.asarray(csc[:, cols].sum(axis=1), dtype=np.float64).ravel()
    in_view = np.zeros(csc.shape[0], dtype=bool)
    rows = s["rows"]
    in_view[rows[rows >= 0]] = True
    sf = np.zeros(csc.shape[0], dtype=np.float32)
    ok = in_view & (lib > 0)
    sf[ok] = (_AN_SCALE / lib[ok]).astype(np.float32)
    return sf, in_view


def _an_lognorm_col_sums(csc, sf, cols, power: int = 1, chunk_nnz: int = 8_000_000) -> np.ndarray:
    """Σ over cells of log1p(count · sf)**power for each matrix column in *cols*."""
    cols = np.asarray(cols, dtype=np.int64)
    out = np.zeros(len(cols), dtype=np.float64)
    ip = csc.indptr
    i = 0
    while i < len(cols):
        # a run of consecutive columns is one slice of the CSC arrays
        j = i + 1
        while (j < len(cols) and cols[j] == cols[j - 1] + 1
               and ip[cols[j] + 1] - ip[cols[i]] <= chunk_nnz):
            j += 1
        s, e = int(ip[cols[i]]), int(ip[cols[j - 1] + 1])
        if e > s:
            v = np.log1p(csc.data[s:e].astype(np.float32, copy=False) * sf[csc.indices[s:e]])
            if power == 2:
                v = v * v
            cs = np.concatenate([[0.0], np.cumsum(v, dtype=np.float64)])
            out[i:j] = cs[ip[cols[i:j] + 1] - s] - cs[ip[cols[i:j]] - s]
        i = j
    return out


# ── differential expression ───────────────────────────────────────────────────
def _an_polygon_mask(polygon_xy, s: dict) -> np.ndarray:
    """Cells of sample *s* inside an outline given in plot space (x incl. the sample's
    offset, y = -µm), as the ROI code does."""
    df = s["df"]
    if s["x_offset"]:
        df = pd.DataFrame({"x_centroid": df["x_centroid"].to_numpy(dtype=float) + s["x_offset"],
                           "y_centroid": df["y_centroid"].to_numpy(dtype=float)})
    return regions._roi_mask_in_polygon(polygon_xy, df)


def _an_cluster_labels(samples, method, use_corrected, alt) -> list:
    """Per sample: the active cluster set's label of each cell as str (None: no such
    column). With several samples the labels are per sample ("S0:C3"), as in the view."""
    from . import clustersets          # (clustersets imports this module)
    multi = len(samples) > 1
    out = []
    for di, s in enumerate(samples):
        df = s["df"]
        own = False                        # a set installed by the app: 0 = no label
        shared = False                     # ... computed jointly: the same labels in every sample
        if alt is not None:
            cands = reseg._reseg_cluster_cols(df.columns)
            pref = [c for c in cands if ("split" in c) == bool(use_corrected)]
            col = clustersets._cs_reseg_col(df.columns, method)
            own = col is not None
            col = col or (pref or cands or [None])[0]
        else:
            ds = s.get("ds") or core.DATA
            cm = ds.get("cluster_methods") or []
            m_use = method if method in cm else (cm or [method])[0]
            col = f"clust__{m_use}"
            own = m_use in clustersets._cs_registry
            shared = own and m_use == method and clustersets._cs_is_shared(m_use)
        if col is None or col not in df.columns:
            out.append(None)
            continue
        v = df[col]
        lab = v.astype(str).to_numpy(dtype=object)
        lab[v.isna().to_numpy()] = None
        if own:
            lab[lab == str(clustersets._CS_UNASSIGNED)] = None
        if multi:
            pre = "C" if shared else f"S{di}:C"
            lab = np.array([None if x is None else f"{pre}{x}" for x in lab], dtype=object)
        out.append(lab)
    return out


def _an_annot_method(color_by, samples, alt):
    """Annotation the cell-type groups come from: the one Color By shows, else the first
    method that has labels for the active segmentation."""
    cands = []
    if str(color_by or "").startswith("cell_type:"):
        cands.append(color_by.split(":", 1)[1])
    cands += [m for m in core._ANNOT_METHODS if m not in cands]
    for m in cands:
        labs = _an_celltype_labels(samples, m, alt)
        if any(l is not None for l in labs):
            return m
    return None


def _an_celltype_labels(samples, annot_method, alt) -> list:
    """Per sample: cell-type label of each cell (None: unlabelled / no annotation)."""
    out = []
    for di, s in enumerate(samples):
        df = s["df"]
        lab = None
        if not annot_method:
            pass
        elif alt is not None:
            lkey = annotation._labels_key_for_method(annot_method, alt)
            with core._annot_lock:
                labels = core._annot_state.get(lkey)
            if labels is not None:
                lab = labels.reindex(pd.Index(df.index.astype(str)))
        elif s.get("ds", core.DATA) is core.DATA:
            with core._annot_lock:
                labels = core._annot_state.get(f"labels_{annot_method}")
            if labels is not None:
                lab = labels.reindex(df.index.astype(str))
        else:
            labels, _src = dataview._extra_cell_type_labels(s["ds"], annot_method)
            if labels is not None:
                lab = labels.reindex(df.index)
                if lab.isna().all() and len(lab):      # labels keyed by str ids
                    lab = pd.Series(labels.reindex(df.index.astype(str)).to_numpy(), index=df.index)
        if lab is None:
            out.append(None)
        else:
            arr = lab.astype(object).to_numpy()
            arr[pd.isna(arr)] = None
            out.append(arr)
    return out


def _de_group_options(seg_source, method, color_by, use_corrected) -> list:
    """Dropdown options of the two group pickers for the active view."""
    samples = _an_samples(seg_source, use_corrected)
    alt = _an_alt_res(seg_source)
    opts = []
    with _de_lock:
        sel = _de_state.get("selection")
        groups = list(_de_state["groups"])
    if sel and sel.get("polygon_xy"):
        opts.append({"label": f"Current selection ({int(sel.get('n') or 0):,} cells)",
                     "value": "sel:current"})
    opts.append({"label": "All other cells", "value": "rest"})     # (as _DE_REST_OPTION)
    for g in groups:
        opts.append({"label": f"Group: {g['name']}", "value": f"grp:{g['name']}"})
    with core._roi_lock:
        rois = list(core._roi_state["rois"])
    for r in rois:
        opts.append({"label": f"ROI: {r['name']} ({r['cls']})",
                     "value": f"roi:{r['cls']}|{r['name']}"})

    def _key(v):
        t = str(v).split(":C")[-1]
        return (0, int(t), str(v)) if t.isdigit() else (1, 0, str(v))

    seen = set()
    for lab in _an_cluster_labels(samples, method, use_corrected, alt):
        if lab is not None:
            seen.update(x for x in pd.unique(lab) if x is not None)
    for c in sorted(seen, key=_key):
        opts.append({"label": f"Cluster {c}", "value": f"cluster:{c}"})
    am = _an_annot_method(color_by, samples, alt)
    seen = set()
    for lab in _an_celltype_labels(samples, am, alt):
        if lab is not None:
            seen.update(x for x in pd.unique(lab) if x is not None)
    for c in sorted(seen, key=str):
        opts.append({"label": f"Cell type ({core._ANNOT_METHODS.get(am, am)}): {c}",
                     "value": f"celltype:{c}"})
    return opts


def _de_group_masks(spec, samples, seg_source, method, color_by, use_corrected):
    """(per-sample boolean masks over the cells, label) of one group picker value.
    "rest" gives (None, label): the complement of the other group. Raises ValueError
    with a message for the user when the group cannot be resolved."""
    alt = _an_alt_res(seg_source)
    spec = str(spec or "")
    if spec == "rest":
        return None, "all other cells"
    if spec == "sel:current":
        with _de_lock:
            sel = _de_state.get("selection")
        if not sel or not sel.get("polygon_xy"):
            raise ValueError("There is no current selection: draw a lasso or box on the tissue first.")
        return [_an_polygon_mask(sel["polygon_xy"], s) for s in samples], "current selection"
    if spec.startswith("grp:"):
        name = spec[4:]
        with _de_lock:
            g = next((g for g in _de_state["groups"] if g["name"] == name), None)
        if g is None:
            raise ValueError(f"Saved group '{name}' no longer exists.")
        return [_an_polygon_mask(g["polygon_xy"], s) for s in samples], name
    if spec.startswith("roi:"):
        cls, _, name = spec[4:].partition("|")
        with core._roi_lock:
            r = next((r for r in core._roi_state["rois"] if r["cls"] == cls and r["name"] == name), None)
        if r is None:
            raise ValueError(f"ROI '{name}' no longer exists.")
        return [_an_polygon_mask(r["polygon_xy"], s) for s in samples], f"ROI {name}"
    if spec.startswith("cluster:"):
        want = spec[8:]
        labs = _an_cluster_labels(samples, method, use_corrected, alt)
        if all(l is None for l in labs):
            raise ValueError("The active view has no cluster set.")
        return ([np.zeros(len(s["df"]), dtype=bool) if l is None else (l == want)
                 for s, l in zip(samples, labs)], f"cluster {want}")
    if spec.startswith("celltype:"):
        want = spec[9:]
        labs = _an_celltype_labels(samples, _an_annot_method(color_by, samples, alt), alt)
        if all(l is None for l in labs):
            raise ValueError("The active segmentation has no cell-type annotation.")
        return ([np.zeros(len(s["df"]), dtype=bool) if l is None else (l == want)
                 for s, l in zip(samples, labs)], want)
    raise ValueError("Pick a group for A and for B.")


def _de_rank_sums(parts, n_genes: int, progress=None, chunk_nnz: int = 6_000_000) -> dict:
    """Sparse Wilcoxon rank sums of log1p(count · sf) for two groups of cells.

    parts: one dict per count matrix with csc, cols (matrix column of each of the
    n_genes genes, -1: absent), grp (int8 per matrix row: 1 = group A, 2 = group B,
    0 = neither) and sf (float32 size factor per matrix row).

    Only the non-zero entries are sorted: all zeros of a gene share one average rank.
    Per chunk of genes one uint64 key per entry (gene | float32 bits of the value |
    is-A) is sorted, which orders the values within each gene and keeps ties adjacent.
    Returns per gene: rank_sum_a (average ranks for ties), tie_term (Σ t³ − t over the
    tie groups, zeros included), nnz_a / nnz_b (cells with a non-zero count), det_a /
    det_b (cells with a count ≥ _DETECT_MIN), sum_a / sum_b (Σ of the normalised, not
    log-transformed values) and raw_a / raw_b (Σ of the counts), plus n_a and n_b."""
    n_a = int(sum(int((p["grp"] == 1).sum()) for p in parts))
    n_b = int(sum(int((p["grp"] == 2).sum()) for p in parts))
    n = n_a + n_b
    nnz_g = np.zeros(n_genes, dtype=np.int64)
    for p in parts:
        ip, c = p["csc"].indptr, p["cols"]
        ok = c >= 0
        nnz_g[ok] += (ip[c[ok] + 1] - ip[c[ok]]).astype(np.int64)
    edges, acc = [0], 0
    for j in range(n_genes):
        acc += int(nnz_g[j])
        if acc >= chunk_nnz:
            edges.append(j + 1)
            acc = 0
    if edges[-1] != n_genes:
        edges.append(n_genes)
    rank_a = np.zeros(n_genes); tie = np.zeros(n_genes)
    nnz_a = np.zeros(n_genes); nnz_b = np.zeros(n_genes)
    sum_a = np.zeros(n_genes); sum_b = np.zeros(n_genes)
    det_a = np.zeros(n_genes); det_b = np.zeros(n_genes)
    raw_a = np.zeros(n_genes); raw_b = np.zeros(n_genes)
    _1, _33 = np.uint64(1), np.uint64(33)
    for g0, g1 in zip(edges[:-1], edges[1:]):
        if progress is not None:
            progress(g0 / max(1, n_genes))
        ng = g1 - g0
        keys = []
        sums = np.zeros(2 * ng); cnts = np.zeros(2 * ng)
        dets = np.zeros(2 * ng); raws = np.zeros(2 * ng)
        for p in parts:
            csc, ip = p["csc"], p["csc"].indptr
            c = p["cols"][g0:g1]
            valid = np.flatnonzero(c >= 0)
            if not len(valid):
                continue
            cv = c[valid]
            if cv[-1] - cv[0] + 1 == len(cv) and (len(cv) == 1 or (np.diff(cv) == 1).all()):
                s, e = int(ip[cv[0]]), int(ip[cv[-1] + 1])
                rows, dat = csc.indices[s:e], csc.data[s:e]
                lens = np.diff(ip[cv[0]:cv[-1] + 2])
            else:
                segs = [(int(ip[x]), int(ip[x + 1])) for x in cv]
                rows = np.concatenate([csc.indices[s:e] for s, e in segs])
                dat = np.concatenate([csc.data[s:e] for s, e in segs])
                lens = np.array([e - s for s, e in segs], dtype=np.int64)
            col_local = np.repeat(valid, lens)
            g = p["grp"][rows]
            keep = (g > 0) & (dat > 0)
            if not keep.any():
                continue
            rows = rows[keep]; col_local = col_local[keep]
            isa = (g[keep] == 1)
            norm = dat[keep].astype(np.float32, copy=False) * p["sf"][rows]
            idx2 = col_local * 2 + isa
            sums += np.bincount(idx2, weights=norm.astype(np.float64), minlength=2 * ng)
            cnts += np.bincount(idx2, minlength=2 * ng)
            dets += np.bincount(idx2, weights=(dat[keep] >= _DETECT_MIN).astype(np.float64),
                                minlength=2 * ng)
            raws += np.bincount(idx2, weights=dat[keep].astype(np.float64), minlength=2 * ng)
            v = np.ascontiguousarray(np.log1p(norm), dtype=np.float32)
            keys.append((col_local.astype(np.uint64) << _33)
                        | (v.view(np.uint32).astype(np.uint64) << _1)
                        | isa.astype(np.uint64))
        sum_b[g0:g1], sum_a[g0:g1] = sums[0::2], sums[1::2]
        nnz_b[g0:g1], nnz_a[g0:g1] = cnts[0::2], cnts[1::2]
        det_b[g0:g1], det_a[g0:g1] = dets[0::2], dets[1::2]
        raw_b[g0:g1], raw_a[g0:g1] = raws[0::2], raws[1::2]
        if not keys:
            continue
        key = keys[0] if len(keys) == 1 else np.concatenate(keys)
        del keys
        key.sort()
        m = len(key)
        col = (key >> _33).astype(np.int64)
        isa = (key & _1).astype(np.int64)
        cv = key >> _1
        start = np.empty(m, dtype=bool)
        start[0] = True
        np.not_equal(cv[1:], cv[:-1], out=start[1:])
        rs = np.flatnonzero(start)                       # first entry of each tie group
        t = np.diff(np.append(rs, m)).astype(np.float64)  # size of each tie group
        col_run = col[rs]
        nnz_col = np.bincount(col, minlength=ng)
        col_start = np.concatenate([[0], np.cumsum(nnz_col)[:-1]])
        nz = n - nnz_col                                  # zeros of the gene: ranks 1..nz
        rank_run = nz[col_run] + (rs - col_start[col_run]) + (t + 1.0) / 2.0
        a_run = np.add.reduceat(isa, rs)
        rank_a[g0:g1] = np.bincount(col_run, weights=rank_run * a_run, minlength=ng)
        tie[g0:g1] = np.bincount(col_run, weights=t ** 3 - t, minlength=ng)
    nz = n - (nnz_a + nnz_b)
    rank_a += (n_a - nnz_a) * (nz + 1.0) / 2.0
    tie += nz ** 3 - nz
    if progress is not None:
        progress(1.0)
    return {"n_a": n_a, "n_b": n_b, "rank_sum_a": rank_a, "tie_term": tie,
            "nnz_a": nnz_a, "nnz_b": nnz_b, "sum_a": sum_a, "sum_b": sum_b,
            "det_a": det_a, "det_b": det_b, "raw_a": raw_a, "raw_b": raw_b}


def _de_test_stats(rank_sum_a, tie_term, n_a: int, n_b: int):
    """Mann-Whitney U of group A, its z-score and the two-sided p-value as log10 (normal
    approximation with tie and continuity correction, as scipy.stats.mannwhitneyu
    method="asymptotic")."""
    from scipy.stats import norm as _norm
    n = n_a + n_b
    u = np.asarray(rank_sum_a, dtype=np.float64) - n_a * (n_a + 1) / 2.0
    mu = n_a * n_b / 2.0
    var = n_a * n_b / 12.0 * ((n + 1) - np.asarray(tie_term, dtype=np.float64) / (n * (n - 1.0)))
    d = u - mu
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(var > 0, np.sign(d) * np.maximum(np.abs(d) - 0.5, 0.0) / np.sqrt(np.maximum(var, 1e-300)), 0.0)
    log10p = np.minimum((_norm.logsf(np.abs(z)) + np.log(2.0)) / np.log(10.0), 0.0)
    return u, z, log10p


_P_FLOOR = 1e-300          # p-values / FDRs below this are written as this value (never 0)
_P_FLOOR_NOTE = ("pval / padj below 1e-300 are written as 1e-300 (they are below the float range); "
                 "neg_log10_pval / neg_log10_padj hold the exact values")


def _p_from_log10(log10p) -> np.ndarray:
    """p-values from their log10: 10**x underflows to exactly 0 for x below about -324
    (|z| above about 38), which would read as "p = 0" and tie every such gene. The value
    is floored at _P_FLOOR; the log-scale column next to it keeps the exact number."""
    lp = np.asarray(log10p, dtype=np.float64)
    with np.errstate(under="ignore"):
        return np.maximum(10.0 ** np.maximum(lp, -400.0), _P_FLOOR)


def _de_sorted(t: pd.DataFrame) -> pd.DataFrame:
    """A result table, most significant first: by FDR, then p (both on the log scale where
    the table has it, so that values below the float range still order), then |statistic|."""
    k = pd.DataFrame(index=t.index)
    k["a"] = -t["neg_log10_padj"] if "neg_log10_padj" in t.columns else t["padj"]
    k["b"] = -t["neg_log10_pval"] if "neg_log10_pval" in t.columns else t["pval"]
    st = next((c for c in ("z", "stat") if c in t.columns), None)
    k["c"] = -pd.to_numeric(t[st], errors="coerce").abs() if st else 0.0
    return t.loc[k.sort_values(["a", "b", "c"], kind="stable", na_position="last").index]


def _bh_log10(log10p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, computed and returned as log10 (so that a
    p-value below the float range still gets an FDR)."""
    lp = np.asarray(log10p, dtype=np.float64)
    m = len(lp)
    if m == 0:
        return lp.copy()
    order = np.argsort(lp, kind="stable")
    adj = lp[order] + np.log10(m / np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 0.0)
    return out


def _de_compute(samples, masks_a, masks_b, progress=None) -> dict:
    """Wilcoxon rank-sum test per gene between two groups of cells of the view.

    masks_a / masks_b: per-sample boolean masks over the cells (None: every cell that is
    not in the other group). A cell in both groups is counted in A only. Expression is
    log1p(count / library size × 10,000); the log2 fold change is of the group means of
    the normalised (not log) values with pseudocount 1; cells are the observations."""
    if not samples:
        raise ValueError("The active view has no expression matrix.")
    if masks_a is None and masks_b is None:
        raise ValueError("Only one of the two groups can be 'All other cells'.")
    genes = _an_common_genes(samples)
    if not genes:
        raise ValueError("The samples of this view share no measured genes.")
    parts, n_overlap, n_nomat = [], 0, 0
    lib_a, lib_b = [], []
    for i, s in enumerate(samples):
        nrow = len(s["df"])
        a = None if masks_a is None else np.asarray(masks_a[i], dtype=bool)
        b = None if masks_b is None else np.asarray(masks_b[i], dtype=bool)
        if a is None:
            a = ~b
        if b is None:
            b = ~a
        if len(a) != nrow or len(b) != nrow:
            raise ValueError("The cell table changed while the groups were collected; run again.")
        n_overlap += int((a & b).sum())
        b = b & ~a
        rows = s["rows"]
        ok = rows >= 0
        n_nomat += int(((a | b) & ~ok).sum())
        sf, _in_view = _an_size_factors(s)
        grp = np.zeros(s["csc"].shape[0], dtype=np.int8)
        grp[rows[ok & b]] = 2
        grp[rows[ok & a]] = 1
        with np.errstate(divide="ignore"):
            lib = np.where(sf > 0, _AN_SCALE / np.where(sf > 0, sf, 1.0), 0.0)
        lib_a.append(lib[grp == 1])
        lib_b.append(lib[grp == 2])
        parts.append({"csc": s["csc"], "grp": grp, "sf": sf,
                      "cols": np.array([s["gni"][g] for g in genes], dtype=np.int64)})
    n_a = int(sum(int((p["grp"] == 1).sum()) for p in parts))
    n_b = int(sum(int((p["grp"] == 2).sum()) for p in parts))
    for nm, k in (("A", n_a), ("B", n_b)):
        if k < _DE_MIN_CELLS:
            _ov = (f" after the {n_overlap:,} cells it shares with A were removed"
                   if nm == "B" and n_overlap else "")
            raise ValueError(f"Group {nm} has {k:,} cells{_ov}; at least {_DE_MIN_CELLS} "
                             f"are needed in each group.")
    r = _de_rank_sums(parts, len(genes), progress=progress)
    u, z, log10p = _de_test_stats(r["rank_sum_a"], r["tie_term"], n_a, n_b)
    log10padj = _bh_log10(log10p)
    mean_a, mean_b = r["sum_a"] / n_a, r["sum_b"] / n_b
    lib_a, lib_b = np.concatenate(lib_a), np.concatenate(lib_b)
    dep_a, dep_b = float(lib_a.sum()), float(lib_b.sum())
    rate_a = _AN_SCALE * r["raw_a"] / max(dep_a, 1e-12)
    rate_b = _AN_SCALE * r["raw_b"] / max(dep_b, 1e-12)
    med_a = float(np.median(lib_a)) if len(lib_a) else float("nan")
    med_b = float(np.median(lib_b)) if len(lib_b) else float("nan")
    table = pd.DataFrame({
        "gene": genes,
        "log2fc": np.log2((mean_a + 1.0) / (mean_b + 1.0)),
        "pct_a": 100.0 * r["det_a"] / n_a,
        "pct_b": 100.0 * r["det_b"] / n_b,
        "rate_a": rate_a, "rate_b": rate_b,
        "low_count": (r["raw_a"] + r["raw_b"]) < _DE_RATE_MIN_COUNT,
        "mean_a": mean_a, "mean_b": mean_b,
        "U": u, "z": z,
        "pval": _p_from_log10(log10p), "padj": _p_from_log10(log10padj),
        "neg_log10_pval": -log10p, "neg_log10_padj": -log10padj,
    })
    return {"table": table, "n_a": n_a, "n_b": n_b, "n_overlap": n_overlap,
            "n_no_matrix": n_nomat, "n_genes": len(genes), "n_samples": len(samples),
            "corrected": bool(samples[0]["corrected"]),
            "median_counts_a": med_a, "median_counts_b": med_b,
            "depth_ratio": _depth_ratio(med_a, med_b)}


def _depth_ratio(med_a, med_b) -> float:
    """max / min of the two groups' median counts per cell (NaN when one is 0 or unknown)."""
    if not (np.isfinite(med_a) and np.isfinite(med_b)) or min(med_a, med_b) <= 0:
        return float("nan")
    return float(max(med_a, med_b) / min(med_a, med_b))


def _depth_note(med_a, med_b, ratio=None) -> str:
    """Sentence on the depth of the two groups of a DE result, with the warning when the
    medians differ by more than _DE_DEPTH_RATIO."""
    ratio = _depth_ratio(med_a, med_b) if ratio is None else ratio
    txt = f"Median counts per cell: A {med_a:,.0f}, B {med_b:,.0f}"
    if np.isfinite(ratio):
        txt += f" (ratio {ratio:.2f})."
        if ratio > _DE_DEPTH_RATIO:
            txt += (f"\n⚠ The groups differ in depth by more than {_DE_DEPTH_RATIO:g}×: detection "
                    f"(% of cells) rises with depth for every gene, and library-size normalisation "
                    f"does not remove that for sparse counts. Read the rate columns (Σ counts / "
                    f"Σ depth) and the log2 fold change, not the percentages.")
    else:
        txt += "."
    return txt


def _de_annotate(res: dict, ctx: dict) -> None:
    """Run the registered annotators on a finished DE result; one that fails leaves a
    note instead of stopping the result from being shown."""
    for fn in list(_de_annotators):
        try:
            fn(res, ctx)
        except jobs._DatasetChanged:
            raise
        except Exception as exc:
            import traceback
            traceback.print_exc()
            res.setdefault("notes", []).append(f"{getattr(fn, '__name__', 'annotator')} failed: {str(exc)[:200]}")


def _de_groups_path() -> str:
    os.makedirs(core.CACHE_BASE_DIR, exist_ok=True)
    return os.path.join(core.CACHE_BASE_DIR, f"de_groups_{_an_dataset_tag()}.json")


def _de_groups_save() -> None:
    """Write the saved groups of this dataset. Call inside _de_lock."""
    try:
        with open(_de_groups_path(), "w") as f:
            json.dump(_de_state["groups"], f)
    except Exception as e:
        print(f"  DE groups cache write error: {e}", flush=True)


def _run_de(spec_a, spec_b, seg_source, method, color_by, use_corrected, options=None) -> None:
    """Background job: differential expression between the two picked groups."""
    _gen = jobs._dataset_gen

    def _set(status, message, result=None):
        with _de_lock:
            if jobs._stale_job(_gen, status, message):
                return
            _de_state["status"] = status
            _de_state["message"] = message
            if result is not None:
                _de_state["result"] = result

    try:
        _t0 = _time.time()
        _set("running", "Collecting the two groups…")
        samples = _an_samples(seg_source, use_corrected)
        if not samples:
            _set("error", _an_no_matrix_msg(seg_source))
            return
        ma, la = _de_group_masks(spec_a, samples, seg_source, method, color_by, use_corrected)
        mb, lb = _de_group_masks(spec_b, samples, seg_source, method, color_by, use_corrected)
        _set("running", "Ranking expression per gene…")

        def _prog(f):
            jobs._job_check()
            jobs._job_progress(fraction=0.05 + 0.9 * f)

        res = _de_compute(samples, ma, mb, progress=_prog)
        res["n_uncorrected"] = _an_uncorrected(samples)
        res.update({"label_a": la, "label_b": lb, "seg_source": seg_source or "xenium",
                    "units_note": f"n = {res['n_a'] + res['n_b']:,} cells, treated as independent",
                    "detect_rule": _DETECT_RULE, "notes": []})
        _set("running", "Gene ownership and leak gate…")
        _de_annotate(res, {"samples": samples, "masks_a": ma, "masks_b": mb, "seg_source": seg_source,
                           "method": method, "color_by": color_by, "use_corrected": use_corrected,
                           "options": dict(options or {})})
        res["seconds"] = _time.time() - _t0
        jobs._check_dataset_gen(_gen)
        _msg = (f"{la} (A, {res['n_a']:,} cells) vs {lb} (B, {res['n_b']:,} cells): "
                f"{res['n_genes']:,} genes, "
                f"{int((res['table']['padj'] < 0.05).sum()):,} with FDR < 0.05 "
                f"({'SPLIT-corrected' if res['corrected'] else 'original'} counts, "
                f"{res['seconds']:.1f} s).")
        if res["n_overlap"]:
            _msg += (f" {res['n_overlap']:,} cells were in both groups and were removed from B.")
        if res["n_samples"] > 1:
            _msg += f" Cells of {res['n_samples']} samples were pooled."
        if res.get("n_uncorrected"):
            _msg += " " + _an_uncorrected_note(res["n_uncorrected"]) + "."
        _msg += (f"\nUnit of the p-values: {res['units_note']} (they are not, when the groups come "
                 f"from few samples: the p-values are then far too small).")
        _msg += "\n" + _depth_note(res["median_counts_a"], res["median_counts_b"], res["depth_ratio"])
        _msg += f"\n% A / % B: {_DETECT_RULE}."
        if res.get("leak_summary"):
            _msg += "\n" + res["leak_summary"]
        for nt in res.get("notes") or []:
            _msg += f"\nNote: {nt}."
        _set("done", _msg, result=res)
        print(f"  DE: {_msg}", flush=True)
    except jobs._JobCancelled:
        raise
    except jobs._DatasetChanged:
        raise
    except ValueError as exc:
        _set("error", str(exc))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set("error", str(exc)[:300])


def make_de_volcano_fig(width=None, height=None, lfc_axis=None):
    """Volcano plot of the last differential-expression result.

    lfc_axis ("mle" / "shrunk"): the fold change on the x axis of a pseudobulk result -
    the unshrunken maximum-likelihood estimate (default) or the shrunken one. A point's
    colour never depends on that choice: for a pseudobulk result it is FDR < 0.05 alone
    (a shrunken estimate near 0 can belong to a significant gene), for the cell-level
    test FDR < 0.05 and |log2FC| ≥ 0.5. Genes the leak gate flagged are drawn as open
    orange diamonds whatever their significance."""
    with _de_lock:
        res = _de_state.get("result")
    if not res:
        return figures._make_placeholder_fig("No differential expression computed yet.")
    t = res["table"]
    pb = res.get("kind") in _DE_PB_KINDS
    fdr_only = res.get("kind") in _DE_FDR_ONLY_KINDS
    lfc, x_title = "log2fc", res.get("lfc_title") or "log2 fold change (A / B)"
    if pb:
        if lfc_axis == "shrunk" and "log2fc_shrunk" in t.columns:
            lfc = "log2fc_shrunk"
            x_title = f"log2 fold change (A / B), {res.get('shrink')}-shrunken"
        else:
            x_title = res.get("lfc_title") or "log2 fold change (A / B), unshrunken (MLE)"
            if lfc_axis == "shrunk":
                x_title += " — no shrunken estimate in this result"
    # genes without an adjusted p-value (DESeq2 independent filtering) are not drawn
    y = np.minimum(t["neg_log10_padj"].to_numpy(dtype=float), 300.0)
    x = t[lfc].to_numpy(dtype=float)
    direction = t["log2fc"].to_numpy(dtype=float)       # the side never depends on shrinkage
    fdr_ok = t["padj"].to_numpy(dtype=float) < 0.05
    sig = fdr_ok if fdr_only else (fdr_ok & (np.abs(x) >= 0.5))
    flagged = (t["leak_flag"].fillna(False).to_numpy(dtype=bool) if "leak_flag" in t.columns
               else np.zeros(len(t), dtype=bool))
    up = sig & (direction > 0) & ~flagged
    dn = sig & ~(direction > 0) & ~flagged
    rule = ((res.get("fdr_name") or "FDR") + " < 0.05") if fdr_only else "FDR < 0.05, |log2FC| ≥ 0.5"
    traces = []
    groups = [(~sig & ~flagged, "not significant", "#6e7681", "circle", 0.8),
              (up, f"{res.get('up_name') or 'higher in A'} ({int(up.sum()):,}; {rule})", "#f85149", "circle", 0.8),
              (dn, f"{res.get('down_name') or 'higher in B'} ({int(dn.sum()):,})", "#58a6ff", "circle", 0.8)]
    if flagged.any():
        groups.append((flagged, f"leak-gate flagged ({int(flagged.sum()):,}; {int((flagged & sig).sum()):,} "
                                f"significant)", "#d29922", "diamond-open", 0.95))
    for mask, name, color, symbol, op in groups:
        sub = t[mask]
        extra = ""
        cd = sub["gene"]
        if "leak_fraction" in sub.columns and "owner" in sub.columns:
            cd = np.column_stack([sub["gene"], sub["owner"].fillna("").astype(str),
                                  sub["leak_fraction"].round(2).astype(object).where(sub["leak_fraction"].notna(), "–")])
            extra = "<br>owner %{customdata[1]}<br>leak fraction %{customdata[2]}"
        traces.append(go.Scattergl(
            x=x[mask], y=y[mask], mode="markers", name=name,
            marker=dict(size=7 if symbol != "circle" else 5, color=color, opacity=op, symbol=symbol,
                        line=dict(width=1.2, color=color) if symbol != "circle" else None),
            text=sub["gene"], customdata=cd,
            hovertemplate="<b>%{text}</b><br>log2FC %{x:.2f}<br>−log10 FDR %{y:.1f}" + extra + "<extra></extra>"))
    # label the strongest genes of each side
    lab = pd.concat([t[up].nlargest(8, "neg_log10_padj"), t[dn].nlargest(8, "neg_log10_padj")])
    fig = go.Figure(traces)
    for _, r in lab.iterrows():
        fig.add_annotation(x=r[lfc], y=min(r["neg_log10_padj"], 300.0), text=r["gene"],
                           showarrow=False, yshift=9, font=dict(size=10, color=core.TEXT))
    fig.add_hline(y=-np.log10(0.05), line=dict(color=core.MUTED, width=1, dash="dot"))
    if not fdr_only:
        for xv in (-0.5, 0.5):
            fig.add_vline(x=xv, line=dict(color=core.MUTED, width=1, dash="dot"))
    fig.update_layout(
        paper_bgcolor=core.CARD_BG, plot_bgcolor=core.CARD_BG, font=dict(color=core.TEXT, size=11),
        margin=dict(l=50, r=10, t=30, b=40),
        title=dict(text=res.get("title") or
                   f"A: {res['label_a']} ({res['n_a']:,})  vs  B: {res['label_b']} ({res['n_b']:,})",
                   font=dict(size=12), x=0.01),
        xaxis=dict(title=x_title, gridcolor=core.BORDER, zerolinecolor=core.BORDER),
        yaxis=dict(title=f"−log10 {res.get('fdr_name') or 'FDR'}", gridcolor=core.BORDER, zerolinecolor=core.BORDER),
        legend=dict(orientation="h", y=-0.2, font=dict(size=10)),
        uirevision=f"de-{res.get('seconds')}-{lfc}",
    )
    if width:
        fig.update_layout(width=width)
    if height:
        fig.update_layout(height=height)
    return fig


def _de_leak_columns(t, out) -> None:
    """Copy the ownership / leak-gate columns of a DE table into its display frame."""
    if "owner" in t.columns:
        out["owner"] = t["owner"].fillna("").astype(str).to_numpy()
    if "leak_fraction" in t.columns:
        out["leak_fraction"] = t["leak_fraction"].round(3).to_numpy()
        out["leak_flag"] = np.where(t["leak_flag"].isna(), "", np.where(t["leak_flag"] == True, "FLAG", "ok"))  # noqa: E712


def _de_table_records(res) -> list:
    """Rows of the result table, most significant first (numbers rounded for display)."""
    if not res:
        return []
    t = _de_sorted(res["table"])
    _g3 = lambda v: None if not np.isfinite(v) else float(f"{v:.3g}")
    if res.get("columns"):
        # a result that names its own columns: [(column, header, "text" | "num")]
        out = pd.DataFrame({"gene": t["gene"].to_numpy()})
        for cid, _name, kind in res["columns"]:
            if cid in ("gene", "leak_flag") or cid not in t.columns:
                continue
            v = t[cid]
            if v.dtype == bool:
                out[cid] = np.where(v.to_numpy(), "yes", "no")
            elif kind == "num":
                out[cid] = pd.to_numeric(v, errors="coerce").map(_g3).to_numpy()
            else:
                out[cid] = v.fillna("").astype(str).to_numpy()
        if "leak_flag" in t.columns:
            out["leak_flag"] = np.where(t["leak_flag"].isna(), "", np.where(t["leak_flag"] == True, "FLAG", "ok"))  # noqa: E712
        return out.astype(object).where(out.notna(), None).to_dict("records")
    if res.get("kind") in _DE_PB_KINDS:
        out = pd.DataFrame({"gene": t["gene"].to_numpy(), "log2fc": t["log2fc"].round(3).to_numpy()})
        if "log2fc_shrunk" in t.columns:
            out["log2fc_shrunk"] = t["log2fc_shrunk"].round(3).to_numpy()
        out["base_mean"] = t["base_mean"].round(1).to_numpy()
        out["mean_a"], out["mean_b"] = t["mean_a"].round(1).to_numpy(), t["mean_b"].round(1).to_numpy()
        out["stat"] = t["stat"].round(2).to_numpy()
        out["pval"], out["padj"] = t["pval"].map(_g3).to_numpy(), t["padj"].map(_g3).to_numpy()
        _de_leak_columns(t, out)
        return out.astype(object).where(out.notna(), None).to_dict("records")
    out = pd.DataFrame({
        "gene": t["gene"].to_numpy(),
        "log2fc": t["log2fc"].round(3).to_numpy(),
        "pct_a": t["pct_a"].round(1).to_numpy(), "pct_b": t["pct_b"].round(1).to_numpy(),
        "mean_a": t["mean_a"].round(2).to_numpy(), "mean_b": t["mean_b"].round(2).to_numpy(),
        "z": t["z"].round(2).to_numpy(),
        "pval": t["pval"].map(lambda v: float(f"{v:.3g}")).to_numpy(),
        "padj": t["padj"].map(lambda v: float(f"{v:.3g}")).to_numpy(),
    })
    if "rate_a" in t.columns:
        out.insert(4, "rate_a", t["rate_a"].map(_g3).to_numpy())
        out.insert(5, "rate_b", t["rate_b"].map(_g3).to_numpy())
    _de_leak_columns(t, out)
    return out.astype(object).where(out.notna(), None).to_dict("records")


def _de_table_columns(res=None) -> list:
    """Columns of the result table: of the cell-level Wilcoxon test, or of a pseudobulk
    result (which names its test statistic and its fold-change columns); the gene
    ownership / leak-gate columns when the result has them."""
    _text = ("gene", "owner", "leak_flag")
    _col = lambda cid, name: {"id": cid, "name": name, "type": "text" if cid in _text else "numeric"}
    leak = []
    if res is not None and "owner" in res["table"].columns:
        leak.append(_col("owner", "Owner"))
    if res is not None and "leak_fraction" in res["table"].columns:
        leak += [_col("leak_fraction", "Leak fraction"), _col("leak_flag", "Leak gate")]
    if res and res.get("columns"):
        return [{"id": cid, "name": name, "type": "text" if kind == "text" else "numeric"}
                for cid, name, kind in res["columns"] if cid in res["table"].columns]
    if res and res.get("kind") in _DE_PB_KINDS:
        cols = [_col("gene", "Gene"), _col("log2fc", "log2FC (MLE)")]
        if "log2fc_shrunk" in res["table"].columns:
            cols.append(_col("log2fc_shrunk", f"log2FC ({res.get('shrink') or 'shrunk'})"))
        return cols + [_col("base_mean", "Base mean"), _col("mean_a", "Mean A"), _col("mean_b", "Mean B"),
                       _col("stat", res.get("stat_name") or "stat"), _col("pval", "p"), _col("padj", "FDR")] + leak
    rate = ([_col("rate_a", "Rate A /10k"), _col("rate_b", "Rate B /10k")]
            if res is None or "rate_a" in res["table"].columns else [])
    return [_col("gene", "Gene"), _col("log2fc", "log2FC"), _col("pct_a", "% A (≥1)"),
            _col("pct_b", "% B (≥1)")] + rate + [_col("mean_a", "Mean A"), _col("mean_b", "Mean B"),
            _col("z", "z"), _col("pval", "p"), _col("padj", "FDR")] + leak


# ── gene-set scores ───────────────────────────────────────────────────────────
def _gs_path() -> str:
    os.makedirs(core.CACHE_BASE_DIR, exist_ok=True)
    return os.path.join(core.CACHE_BASE_DIR, f"genesets_{_an_dataset_tag()}.json")


def _gs_save() -> None:
    """Write the saved gene sets of this dataset. Call inside _geneset_lock."""
    try:
        with open(_gs_path(), "w") as f:
            json.dump(_geneset_state["sets"], f)
    except Exception as e:
        print(f"  gene-set cache write error: {e}", flush=True)


def _an_autoload() -> None:
    """Load the saved gene sets and DE groups of the current dataset from the cache dir
    and drop every result computed for another dataset."""
    sets, groups = {}, []
    try:
        if os.path.exists(_gs_path()):
            with open(_gs_path()) as f:
                sets = {str(k): [str(g) for g in v] for k, v in json.load(f).items()}
    except Exception as e:
        print(f"  gene sets: could not read {_gs_path()}: {e}", flush=True)
    try:
        if os.path.exists(_de_groups_path()):
            with open(_de_groups_path()) as f:
                groups = [g for g in json.load(f) if g.get("name") and g.get("polygon_xy")]
    except Exception as e:
        print(f"  DE groups: could not read {_de_groups_path()}: {e}", flush=True)
    with _geneset_lock:
        _geneset_state.update({"status": "idle", "message": "", "sets": sets})
        _geneset_state["version"] += 1
    with _de_lock:
        _de_state.update({"status": "idle", "message": "", "result": None,
                          "groups": groups, "selection": None})
    with _density_lock:
        _density_state.update({"status": "idle", "message": ""})
        _density_state["version"] += 1
    _geneset_cache.clear()
    _an_means_cache.clear()
    with _density_mem_lock:
        _density_mem.clear()
    if sets or groups:
        print(f"  Loaded {len(sets)} gene set(s) and {len(groups)} DE group(s) from cache", flush=True)


def _gs_parse(text) -> list:
    """Gene names of a pasted list (newline / comma / semicolon / whitespace separated),
    in order, without repeats."""
    out, seen = [], set()
    for g in re.split(r"[\s,;]+", str(text or "")):
        g = g.strip().strip("\"'")
        if g and g not in seen:
            seen.add(g)
            out.append(g)
    return out


def _gs_match(genes, available) -> tuple:
    """(found, missing): *genes* matched to the names in *available*, exactly or, failing
    that, ignoring case (found holds the dataset's spelling, without repeats)."""
    avail = set(available)
    lower = {}
    for g in available:
        lower.setdefault(g.lower(), g)
    found, missing = [], []
    for g in genes:
        hit = g if g in avail else lower.get(g.lower())
        if hit is None:
            missing.append(g)
        elif hit not in found:
            found.append(hit)
    return found, missing


def _an_gene_means(samples, genes, key=None) -> np.ndarray:
    """Mean log-normalised expression of each of *genes* over all cells of the view."""
    if key is not None:
        hit = _an_means_cache.get(key)
        if hit is not None and hit[0] == genes:
            return hit[1]
    tot, n = np.zeros(len(genes)), 0
    for s in samples:
        sf, in_view = _an_size_factors(s, genes)
        tot += _an_lognorm_col_sums(s["csc"], sf, [s["gni"][g] for g in genes])
        n += int(in_view.sum())
    means = tot / max(1, n)
    if key is not None:
        _an_means_cache[key] = (genes, means)
        while len(_an_means_cache) > 4:
            _an_means_cache.popitem(last=False)
    return means


def _gs_score(samples, gene_list, ctrl_size: int = 50, n_bins: int = 25, seed: int = 0,
              zscore: bool = True, means_key=None) -> tuple:
    """Gene-set score per cell: mean expression of the set minus the mean expression of
    a control set drawn from the same expression bins (Seurat AddModuleScore /
    scanpy.tl.score_genes).

    Expression is log1p(count / library size × 10,000). All measured genes are ranked
    by their mean expression over the cells of the view and cut into n_bins bins; for
    every bin a set gene falls into, ctrl_size genes of that bin are drawn (fixed
    *seed*), set genes excluded. With *zscore* each gene is first standardised over the
    cells of the view (mean 0, sd 1), so that a highly expressed gene does not dominate
    the mean; genes with no variance are left out.

    Returns (arrays, info): one float32 array per sample aligned to its cells (NaN for
    a cell without a matrix row), and info with found / missing / used / n_control."""
    from scipy.stats import rankdata as _rankdata
    genes = _an_common_genes(samples)
    found, missing = _gs_match(gene_list, genes)
    info = {"found": found, "missing": missing, "used": [], "control": [], "n_control": 0,
            "n_cells": int(sum(len(s["df"]) for s in samples))}
    if len(found) < 2:
        return None, info
    means = _an_gene_means(samples, genes, key=means_key)
    pos = {g: i for i, g in enumerate(genes)}
    set_idx = np.array([pos[g] for g in found], dtype=np.int64)
    n_items = max(1, int(np.round(len(genes) / (n_bins - 1))))
    cut = _rankdata(means, method="min").astype(np.int64) // n_items
    rng = np.random.default_rng(seed)
    in_set = np.zeros(len(genes), dtype=bool)
    in_set[set_idx] = True
    ctrl = []
    for c in np.unique(cut[set_idx]):
        pool = np.flatnonzero((cut == c) & ~in_set)
        if len(pool):
            ctrl.append(rng.choice(pool, size=min(ctrl_size, len(pool)), replace=False))
    ctrl_idx = np.sort(np.concatenate(ctrl)) if ctrl else np.zeros(0, dtype=np.int64)
    sel = np.concatenate([set_idx, ctrl_idx])
    mu = means[sel]
    if zscore:
        sq, n = np.zeros(len(sel)), 0
        for s in samples:
            sf, in_view = _an_size_factors(s, genes)
            sq += _an_lognorm_col_sums(s["csc"], sf, [s["gni"][genes[i]] for i in sel], power=2)
            n += int(in_view.sum())
        sd = np.sqrt(np.maximum(sq / max(1, n) - mu ** 2, 0.0))
    else:
        sd = np.ones(len(sel))
    ok = sd > 1e-12
    is_set = np.arange(len(sel)) < len(set_idx)
    k_set, k_ctrl = int((ok & is_set).sum()), int((ok & ~is_set).sum())
    info["used"] = [genes[i] for i, o in zip(set_idx, ok[:len(set_idx)]) if o]
    info["control"] = [genes[i] for i, o in zip(ctrl_idx, ok[len(set_idx):]) if o]
    info["n_control"] = k_ctrl
    if k_set < 1:
        return None, info
    w = np.zeros(len(sel))
    w[ok & is_set] = 1.0 / (sd[ok & is_set] * k_set)
    if k_ctrl:
        w[ok & ~is_set] = -1.0 / (sd[ok & ~is_set] * k_ctrl)
    offset = -float((w * mu).sum()) if zscore else 0.0
    arrays = []
    for s in samples:
        csc, ip = s["csc"], s["csc"].indptr
        sf, _in_view = _an_size_factors(s, genes)
        rr, vv = [], []
        for j, wj in zip(sel, w):
            if wj == 0.0:
                continue
            c = s["gni"][genes[j]]
            a, b = int(ip[c]), int(ip[c + 1])
            if b > a:
                r_ = csc.indices[a:b]
                rr.append(r_)
                vv.append(np.log1p(csc.data[a:b].astype(np.float32, copy=False) * sf[r_]) * wj)
        per_row = (np.bincount(np.concatenate(rr), weights=np.concatenate(vv),
                               minlength=csc.shape[0]) if rr else np.zeros(csc.shape[0]))
        rows = s["rows"]
        vals = np.where(rows >= 0, per_row[np.clip(rows, 0, None)] + offset, np.nan)
        arrays.append(vals.astype(np.float32))
    return arrays, info


def _geneset_names() -> list:
    with _geneset_lock:
        return list(_geneset_state["sets"])


def _geneset_values(name, seg_source="xenium", use_corrected: bool = False):
    """(arrays, info) of the saved gene set *name* for the active view (one array per
    sample, aligned to its cells), computed on first use and kept until the view
    changes; (None, info) when the set is unknown or fewer than 2 of its genes exist."""
    with _geneset_lock:
        gene_list = _geneset_state["sets"].get(name)
    if not gene_list:
        return None, {"found": [], "missing": [], "used": [], "n_control": 0}
    with _geneset_compute_lock:
        samples = _an_samples(seg_source, use_corrected)
        if not samples:
            return None, {"found": [], "missing": list(gene_list), "used": [], "n_control": 0}
        vkey = _an_view_key(seg_source, use_corrected, samples)
        ckey = (vkey, name, tuple(gene_list))
        hit = _geneset_cache.get(ckey)
        if hit is not None:
            return hit
        _t0 = _time.time()
        out = _gs_score(samples, gene_list, means_key=vkey)
        _geneset_cache[ckey] = out
        while len(_geneset_cache) > 12:
            _geneset_cache.popitem(last=False)
        print(f"  Gene set '{name}': scored {out[1].get('n_cells', 0):,} cells in "
              f"{_time.time() - _t0:.1f} s ({len(out[1]['used'])} genes, "
              f"{out[1]['n_control']} control genes)", flush=True)
        return out


def _geneset_color(color_by, seg_source="xenium", use_corrected: bool = False):
    """For a Color By value "geneset:<name>": (values over the cells of the view, the
    samples one after the other; symmetric colour limit; label), or None when the set
    cannot be scored (the views then draw transcript counts)."""
    name = str(color_by or "")[len("geneset:"):]
    for prefix, fn in list(_view_color_providers.items()):
        if name.startswith(prefix):
            got = fn(name, seg_source, use_corrected)
            if got is None:
                return None
            vals, rng, label = got
            if rng is None:
                _view_color_seq.pop(label, None)
                fin = np.asarray(vals)[np.isfinite(vals)]
                lim = float(np.abs(fin).max()) if len(fin) else 1.0
                return vals, (lim if lim > 0 else 1.0), label
            _view_color_seq[label] = (float(rng[0]), float(rng[1]))
            return vals, float(rng[1]), label
    arrays, _info = _geneset_values(name, seg_source, use_corrected)
    if arrays is None:
        return None
    vals = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
    fin = vals[np.isfinite(vals)]
    lim = float(np.abs(fin).max()) if len(fin) else 1.0
    return vals, (lim if lim > 0 else 1.0), f"Gene set: {name}"


def _geneset_marker(vals, lim, label, cbar_range=None, **marker) -> dict:
    """Plotly marker of a gene-set score: diverging scale centred at 0 (the range inputs
    override either end)."""
    lo, hi = -lim, lim
    seq = _view_color_seq.get(label)
    if seq is not None:                    # a sequential value (distance, density, score ≥ 0)
        lo, hi = seq
    if cbar_range:
        lo = cbar_range[0] if cbar_range[0] is not None else lo
        hi = cbar_range[1] if cbar_range[1] is not None else hi
    return dict(color=vals, colorscale="Viridis" if seq is not None else "RdBu_r",
                cmin=lo, cmax=hi, showscale=True,
                colorbar=dict(title=label.replace(": ", ":<br>"), thickness=12, len=0.5, x=1.02),
                **marker)


def _run_geneset_score(name, seg_source, use_corrected) -> None:
    """Background job: score a saved gene set for the active view."""
    _gen = jobs._dataset_gen

    def _set(status, message):
        with _geneset_lock:
            if jobs._stale_job(_gen, status, message):
                return
            _geneset_state["status"] = status
            _geneset_state["message"] = message

    try:
        _set("running", f"Scoring gene set '{name}'…")
        arrays, info = _geneset_values(name, seg_source, use_corrected)
        jobs._check_dataset_gen(_gen)
        _miss = (f" Not in this dataset ({len(info['missing'])}): "
                 f"{', '.join(info['missing'][:30])}{'…' if len(info['missing']) > 30 else ''}."
                 if info["missing"] else "")
        if arrays is None:
            _set("error", f"Gene set '{name}': only {len(info['found'])} of its genes are measured "
                          f"in this dataset; at least 2 are needed.{_miss}")
            return
        with _geneset_lock:
            _geneset_state["version"] += 1
        _dropped = len(info["found"]) - len(info["used"])
        _unc_note = _an_uncorrected_note(_an_uncorrected(_an_samples(seg_source, use_corrected)))
        _set("done", f"Gene set '{name}': {len(info['used'])} genes scored against "
                     f"{info['n_control']} control genes."
                     + (f" {_dropped} gene(s) without any variance left out." if _dropped else "")
                     + _miss
                     + (f" {_unc_note}; they have no score." if _unc_note else "")
                     + f" Pick 'Gene set: {name}' under Color By.")
    except jobs._JobCancelled:
        raise
    except jobs._DatasetChanged:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set("error", str(exc)[:300])


# ── transcript density ────────────────────────────────────────────────────────
def _density_samples() -> list:
    """The samples of the Xenium view a density grid is drawn for: the primary one and the
    extra samples beside it, each with the µm it is shifted by. One dict per sample:
    label, ds (dataset dict), data_dir (None: loaded from a .zarr, no transcripts),
    x_offset."""
    extras = list(dataview.EXTRA_DATASETS)
    out = [{"label": "S0" if extras else "xenium", "ds": core.DATA,
            "data_dir": core.DATA.get("data_dir"), "x_offset": 0.0}]
    for i, eds in enumerate(extras):
        out.append({"label": f"S{i + 1}", "ds": eds, "data_dir": eds.get("data_dir"),
                    "x_offset": float(eds.get("x_offset", 0.0))})
    return out


def _density_cache_file(gene: str, ds=None) -> str:
    """Cache file of a gene's base grid: <cache>/density_<dataset>/<gene>__b<bin>um__<filter>.npz
    (*ds*: the dataset dict of an extra sample; each sample has its own directory, so a
    grid computed for a sample is found again whether it is loaded as primary or extra)."""
    tag = (_an_dataset_tag() if ds is None or ds is core.DATA
           else hashlib.md5(dataview._ds_key(ds).encode()).hexdigest()[:8])
    d = os.path.join(core.CACHE_BASE_DIR, f"density_{tag}")
    if gene == _DENSITY_ALL:
        stem = "ALL_GENES"
    else:
        stem = (re.sub(r"[^A-Za-z0-9_.-]", "_", gene)[:60] + "_"
                + hashlib.md5(gene.encode()).hexdigest()[:6])
    return os.path.join(d, f"{stem}__b{_DENSITY_BASE_UM:g}um__{_DENSITY_FILTER_VERSION}.npz")


def _density_tx_batches_parquet(path: str, gene: str, qv_min=20):
    """Yield (x, y, fraction done) of the transcripts.parquet rows that pass the app's
    transcript filters (Q ≥ qv_min, gene features only; one gene when *gene* is not
    _DENSITY_ALL). One row group at a time, two or three columns, filter pushed down."""
    import pyarrow as _pa
    import pyarrow.compute as _pc
    import pyarrow.dataset as _ds
    import pyarrow.parquet as _pq
    dataset = _ds.dataset(path, format="parquet")
    names = dataset.schema.names
    flt = reseg._tx_gene_filters(names, qv_min)
    expr = _pq.filters_to_expression(flt) if flt else None
    by_name = "is_gene" not in names and "feature_name" in names
    if gene != _DENSITY_ALL:
        _ftype = dataset.schema.field("feature_name").type
        _val = gene.encode() if (_pa.types.is_binary(_ftype) or _pa.types.is_large_binary(_ftype)) else gene
        _g = _pc.field("feature_name") == _val
        expr = _g if expr is None else (expr & _g)
        by_name = False          # a panel gene is not a control feature
    cols = ["x_location", "y_location"] + (["feature_name"] if by_name else [])
    groups = [rg for frag in dataset.get_fragments() for rg in frag.split_by_row_group()]
    for i, rg in enumerate(groups):
        for batch in rg.to_batches(columns=cols, filter=expr, batch_size=1 << 20):
            if not batch.num_rows:
                continue
            if by_name:
                bad = reseg._tx_control_feature_mask(batch.column("feature_name").to_pandas())
                if bad.any():
                    batch = batch.filter(_pa.array(~bad))
            yield (batch.column("x_location").to_numpy(zero_copy_only=False),
                   batch.column("y_location").to_numpy(zero_copy_only=False), i / len(groups))
        yield np.zeros(0, np.float32), np.zeros(0, np.float32), (i + 1) / len(groups)


def _density_tx_batches_zarr(data_dir: str, gene: str, qv_min=20):
    """As _density_tx_batches_parquet, from the full-resolution tiles of
    transcripts.zarr.zip (bundles without transcripts.parquet): one tile at a time.
    Gene features are told from control features by name."""
    import xenium_viewer_api as _va
    store = _va.tile_store(data_dir)
    m = store._tx_open()
    if not m:
        raise ValueError("This bundle has no transcripts.parquet and no transcripts.zarr.zip.")
    names = [n for n, _ in sorted(m["gene_index"].items(), key=lambda kv: kv[1])]
    if gene == _DENSITY_ALL:
        ok_gene = ~reseg._tx_control_feature_mask(names)
    else:
        ok_gene = np.array([n == gene for n in names], dtype=bool)
    keys = m["keys"][0]
    for i, key in enumerate(keys):
        grp = store._tx[f"grids/0/{key}"]
        gid = np.asarray(grp["gene_identity"][:]).reshape(-1).astype(np.int64)
        keep = (gid >= 0) & (gid < len(ok_gene))
        keep[keep] = ok_gene[gid[keep]]
        if "quality_score" in grp and qv_min is not None:
            keep &= np.asarray(grp["quality_score"][:], dtype=np.float32).reshape(-1) >= qv_min
        if keep.any():
            loc = np.asarray(grp["location"][:], dtype=np.float32)
            yield loc[keep, 0], loc[keep, 1], (i + 1) / len(keys)
        else:
            yield np.zeros(0, np.float32), np.zeros(0, np.float32), (i + 1) / len(keys)


def _density_bin_index(v, base=_DENSITY_BASE_UM) -> np.ndarray:
    """Base-grid bin of a coordinate in µm: bin i covers [i·base, (i+1)·base)."""
    return np.floor(np.asarray(v, dtype=np.float64) / base).astype(np.int64)


def _density_scan(data_dir: str, gene: str, progress=None, source: str = "auto") -> dict:
    """Count the filtered transcripts of *gene* (or of all genes) per base bin.
    Memory stays at one batch plus the non-empty bins. Returns {iy, ix, n (int64 arrays
    of the non-empty bins), total, source}."""
    pq_path = os.path.join(data_dir, "transcripts.parquet")
    if source in ("auto", "parquet") and os.path.exists(pq_path):
        batches, source = _density_tx_batches_parquet(pq_path, gene), "parquet"
    elif source in ("auto", "zarr") and os.path.exists(os.path.join(data_dir, "transcripts.zarr.zip")):
        batches, source = _density_tx_batches_zarr(data_dir, gene), "zarr"
    else:
        raise ValueError("This bundle has no transcripts.parquet and no transcripts.zarr.zip.")
    _OFF, _W = 1 << 20, 1 << 22          # bin indices from -2**20 to 3·2**20 fit one key
    keys, cnts, pending, total = [], [], 0, 0

    def _merge():
        k = np.concatenate(keys)
        uk, inv = np.unique(k, return_inverse=True)
        c = np.bincount(inv, weights=np.concatenate(cnts), minlength=len(uk)).astype(np.int64)
        keys[:], cnts[:] = [uk], [c]
        return len(uk)

    for x, y, frac in batches:
        jobs._job_check()
        if len(x):
            ix, iy = _density_bin_index(x), _density_bin_index(y)
            total += len(x)
            x0, y0 = int(ix.min()), int(iy.min())
            w, h = int(ix.max()) - x0 + 1, int(iy.max()) - y0 + 1
            if w * h <= 8 * len(x) + 1024:
                # a batch is local in space: count on its own window
                c = np.bincount((iy - y0) * w + (ix - x0), minlength=w * h)
                nzi = np.flatnonzero(c)
                k = (nzi // w + y0 + _OFF) * _W + (nzi % w + x0 + _OFF)
                c = c[nzi]
            else:
                k, c = np.unique((iy + _OFF) * _W + (ix + _OFF), return_counts=True)
            keys.append(k.astype(np.int64)); cnts.append(c.astype(np.int64))
            pending += len(k)
            if pending > 16_000_000:
                pending = _merge()
        if progress is not None:
            progress(frac)
    if not keys:
        z = np.zeros(0, dtype=np.int64)
        return {"iy": z, "ix": z.copy(), "n": z.copy(), "total": 0, "source": source}
    _merge()
    uk, c = keys[0], cnts[0]
    return {"iy": uk // _W - _OFF, "ix": uk % _W - _OFF, "n": c, "total": int(total),
            "source": source}


def _density_write(path: str, scan: dict, gene: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, iy=scan["iy"].astype(np.int32), ix=scan["ix"].astype(np.int32),
                        n=scan["n"].astype(np.int64), total=np.int64(scan["total"]),
                        base=np.float64(_DENSITY_BASE_UM), gene=np.array(gene),
                        filter=np.array(_DENSITY_FILTER_VERSION), source=np.array(scan["source"]))
    os.replace(tmp, path)


def _density_ready(gene) -> bool:
    return bool(gene) and core.DATA.get("data_dir") is not None and os.path.exists(_density_cache_file(gene))


def _density_status(gene) -> dict:
    """Per sample of the view, whether *gene*'s grid exists: {ready, missing (has
    transcripts, no grid yet), unavailable (loaded from a .zarr)} as lists of labels."""
    out = {"ready": [], "missing": [], "unavailable": []}
    prov = _density_virtual(gene)
    for smp in _density_samples():
        if not smp["data_dir"]:
            out["unavailable"].append(smp["label"])
        elif gene and (prov["grid"](gene, smp["ds"]) is not None if prov is not None
                       else os.path.exists(_density_cache_file(gene, smp["ds"]))):
            out["ready"].append(smp["label"])
        else:
            out["missing"].append(smp["label"])
    return out


def _density_grid(gene: str, bin_um, ds=None) -> "dict | None":
    """Dense grid of a cached gene at *bin_um* (a multiple of the base bin): {grid
    (int64, rows = y bins, cols = x bins), ix0, iy0 (bin index of grid[0, 0]), bin,
    total, vmax (99.5th percentile of the non-empty bins: default colour maximum)}.
    None when the gene's grid has not been computed yet. *ds*: an extra sample (its
    grid is in that sample's own µm, without the view offset)."""
    prov = _density_virtual(gene)
    if prov is not None:
        return prov["grid"](gene, ds)       # its own bins; *bin_um* does not apply
    path = _density_cache_file(gene, ds)
    k = max(1, int(round(float(bin_um) / _DENSITY_BASE_UM)))
    mkey = (path, k)
    with _density_mem_lock:
        hit = _density_mem.get(mkey)
        if hit is not None:
            _density_mem.move_to_end(mkey)
            return hit
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        iy, ix, n, total = z["iy"].astype(np.int64), z["ix"].astype(np.int64), z["n"], int(z["total"])
    out = {"bin": k * _DENSITY_BASE_UM, "total": total, "ix0": 0, "iy0": 0,
           "grid": np.zeros((0, 0), dtype=np.int64), "vmax": 1.0}
    if len(n):
        # floor division keeps bin edges at multiples of the bin size, also below 0
        jy, jx = iy // k, ix // k
        y0, x0 = int(jy.min()), int(jx.min())
        h, w = int(jy.max()) - y0 + 1, int(jx.max()) - x0 + 1
        grid = np.bincount((jy - y0) * w + (jx - x0), weights=n, minlength=h * w)
        grid = np.rint(grid).astype(np.int64).reshape(h, w)
        nzv = grid[grid > 0]
        out.update({"grid": grid, "ix0": x0, "iy0": y0,
                    "vmax": float(max(1.0, np.percentile(nzv, 99.5))) if len(nzv) else 1.0})
    with _density_mem_lock:
        _density_mem[mkey] = out
        _density_mem_trim()
    return out


def _density_mem_trim() -> None:
    """Keep the grids of a few (gene, bin) pairs for every sample of the view. Call
    inside _density_mem_lock."""
    keep = max(6, 3 * (1 + len(dataview.EXTRA_DATASETS)) + 2)
    while len(_density_mem) > keep:
        _density_mem.popitem(last=False)


def _density_box_slice(g: dict, x0=None, y0=None, x1=None, y1=None) -> tuple:
    """(r0, r1, c0, c1): the rows / columns of grid *g* that cover the box (µm of the
    grid's own sample; None: everything)."""
    b = g["bin"]
    h, w = g["grid"].shape
    c0, r0, c1, r1 = 0, 0, w, h
    if x0 is not None and w:
        c0 = int(np.clip(np.floor(x0 / b) - g["ix0"], 0, w))
        c1 = int(np.clip(np.floor(x1 / b) - g["ix0"] + 1, c0, w))
        r0 = int(np.clip(np.floor(y0 / b) - g["iy0"], 0, h))
        r1 = int(np.clip(np.floor(y1 / b) - g["iy0"] + 1, r0, h))
    return r0, r1, c0, c1


def _density_window(gene, bin_um, x0=None, y0=None, x1=None, y1=None,
                    max_cells: int = _DENSITY_MAX_CELLS, ds=None, x_offset: float = 0.0,
                    factor=None) -> "dict | None":
    """The part of a gene's grid that covers the box (µm; None: everything), for a heat
    layer: {values (float32 2-D, transcripts per bin), x0, y0 (µm of the corner of
    values[0, 0]), cell (µm per value), bin, vmax, total, coarsened}. When the box
    holds more than *max_cells* bins, f × f bins are averaged into one value, so the
    values stay "transcripts per bin" and the colour scale does not move with the zoom.

    *ds* / *x_offset*: the grid of an extra sample, drawn *x_offset* µm to the right; the
    box and the returned x0 are in view coordinates (the grid itself is not resampled:
    its bin edges stay multiples of the bin size in the sample's own µm). *factor*
    fixes f (used to coarsen all samples of a view alike)."""
    g = _density_grid(gene, bin_um, ds)
    if g is None:
        return None
    b, grid = g["bin"], g["grid"]
    off = float(x_offset or 0.0)
    r0, r1, c0, c1 = _density_box_slice(g, None if x0 is None else x0 - off, y0,
                                        None if x1 is None else x1 - off, y1)
    sub = grid[r0:r1, c0:c1]
    f = int(factor or 1)
    if factor is None:
        while (sub.shape[0] // f + 1) * (sub.shape[1] // f + 1) > max_cells:
            f += 1
    if f > 1 and sub.size and g.get("palette"):
        sub = sub[::f, ::f]                 # categories are not averaged: every f-th bin
    elif f > 1 and sub.size:
        ph, pw = -sub.shape[0] % f, -sub.shape[1] % f
        sub = np.pad(sub, ((0, ph), (0, pw)))
        sub = sub.reshape(sub.shape[0] // f, f, sub.shape[1] // f, f).sum(axis=(1, 3)) / float(f * f)
    return {"values": np.ascontiguousarray(sub, dtype=np.float32),
            "x0": (g["ix0"] + c0) * b + off, "y0": (g["iy0"] + r0) * b, "cell": b * f, "bin": b,
            "vmax": g["vmax"], "total": g["total"], "coarsened": f > 1}


def _density_windows(gene, bin_um, x0=None, y0=None, x1=None, y1=None,
                     max_cells: int = _DENSITY_MAX_CELLS) -> "dict | None":
    """_density_window for every sample of the view that has a grid of *gene*: {parts
    (one window per sample, each at its sample's offset, with its label), bin, vmax,
    total, coarsened, labels, missing, unavailable}. All parts share one coarsening
    factor and one colour maximum (99.5th percentile of the non-empty bins of all
    samples together), so the samples can be compared by eye. None when no sample has
    a grid. With a single sample this is that sample's _density_window."""
    smps = _density_samples()
    st = _density_status(gene)
    grids = []
    for smp in smps:
        if smp["label"] in st["ready"]:
            g = _density_grid(gene, bin_um, smp["ds"])
            if g is not None:
                grids.append((smp, g))
    if not grids:
        return None
    f = 1
    while True:
        n = 0
        for smp, g in grids:
            off = smp["x_offset"]
            r0, r1, c0, c1 = _density_box_slice(g, None if x0 is None else x0 - off, y0,
                                                None if x1 is None else x1 - off, y1)
            if r1 > r0 and c1 > c0:
                n += ((r1 - r0) // f + 1) * ((c1 - c0) // f + 1)
        if n <= max_cells:
            break
        f += 1
    parts = []
    for smp, g in grids:
        win = _density_window(gene, bin_um, x0, y0, x1, y1, ds=smp["ds"],
                              x_offset=smp["x_offset"], factor=f)
        win["label"] = smp["label"]
        parts.append(win)
    if len(grids) == 1 or _density_virtual(gene) is not None:
        vmax = max(g["vmax"] for _, g in grids)
    else:
        vkey = ("vmax", tuple(_density_cache_file(gene, smp["ds"]) for smp, _ in grids), grids[0][1]["bin"])
        with _density_mem_lock:
            vmax = _density_mem.get(vkey)
        if vmax is None:
            nz = np.concatenate([g["grid"][g["grid"] > 0] for _, g in grids])
            vmax = float(max(1.0, np.percentile(nz, 99.5))) if len(nz) else 1.0
            with _density_mem_lock:
                _density_mem[vkey] = vmax
                _density_mem_trim()
    return {"parts": parts, "bin": grids[0][1]["bin"], "vmax": vmax,
            "total": int(sum(g["total"] for _, g in grids)), "coarsened": f > 1,
            "labels": [smp["label"] for smp, _ in grids],
            "missing": st["missing"], "unavailable": st["unavailable"],
            "palette": grids[0][1].get("palette"), "names": grids[0][1].get("names")}


def _density_label(gene, bin_um) -> str:
    prov = _density_virtual(gene)
    if prov is not None:
        return prov["label"](gene, bin_um)
    return (f"{'All genes' if gene == _DENSITY_ALL else gene}: transcripts per "
            f"{float(bin_um):g} µm bin")


def _density_heatmap_trace(spec: dict, relayout=None):
    """The first of _density_heatmap_traces (the only one with a single sample), or None."""
    traces = _density_heatmap_traces(spec, relayout)
    return traces[0] if traces else None


def _density_heatmap_traces(spec: dict, relayout=None) -> list:
    """Plotly heat layers of the transcript density for the current viewport (drawn below
    the WebGL cell markers): one per sample that has a grid, each at its sample's offset,
    all on one colour scale. Empty when the layer is off / not computed."""
    if not spec or not spec.get("on") or not spec.get("gene"):
        return []
    box = [None] * 4
    r = relayout or {}
    try:
        xa, xb = float(r["xaxis.range[0]"]), float(r["xaxis.range[1]"])
        ya, yb = float(r["yaxis.range[0]"]), float(r["yaxis.range[1]"])
        px, py = 0.25 * abs(xb - xa), 0.25 * abs(yb - ya)
        box = [min(xa, xb) - px, min(-ya, -yb) - py, max(xa, xb) + px, max(-ya, -yb) + py]
    except (KeyError, TypeError, ValueError):
        pass
    wins = _density_windows(spec["gene"], spec.get("bin") or 10, *box, max_cells=160_000)
    if wins is None:
        return []
    vmax = spec.get("max") or wins["vmax"]
    pal = wins.get("palette")
    scale, zmin = "Magma", 0
    if pal:                                     # categorical layer: value i is colour i - 1
        n = len(pal)
        scale = [[min(1.0, (i + j) / n), pal[i]] for i in range(n) for j in (0, 1)]
        zmin, vmax = 0.5, n + 0.5
    traces = []
    for win in wins["parts"]:
        if not win["values"].size:
            continue
        z = win["values"].astype(float)
        z[z <= 0] = np.nan                      # empty bins stay transparent
        z = z[::-1]                             # plot y = -µm: last row first
        c = win["cell"]
        traces.append(go.Heatmap(
            z=z, x0=win["x0"] + c / 2, dx=c, y0=-(win["y0"] + (z.shape[0] - 0.5) * c), dy=c,
            colorscale=scale, zmin=zmin, zmax=vmax,
            opacity=float(spec.get("opacity") if spec.get("opacity") is not None else 0.7),
            hoverinfo="skip", name="transcript density", showlegend=False,
            showscale=not traces,               # one colour bar for all samples
            colorbar=dict(title=_density_label(spec["gene"], win["bin"]).replace(": ", ":<br>")
                          .replace(" per ", "<br>per "), thickness=12, len=0.4, x=1.02, y=0.2,
                          **(dict(tickmode="array", tickvals=list(range(1, len(pal) + 1)),
                                  ticktext=list(wins.get("names") or [])) if pal else {})),
        ))
    return traces


def _run_density(genes, _unused=None) -> None:
    """Background job: bin the transcripts of each of *genes* and cache the grids."""
    _gen = jobs._dataset_gen

    def _set(status, message):
        with _density_lock:
            if jobs._stale_job(_gen, status, message):
                return
            _density_state["status"] = status
            _density_state["message"] = message

    try:
        # every sample of the view that has its transcripts (a sample loaded from a .zarr
        # has none): one grid per sample and gene, in that sample's own cache directory
        smps = _density_samples()
        todo = [s for s in smps if s["data_dir"]]
        skipped = [s["label"] for s in smps if not s["data_dir"]]
        if not todo:
            _set("error", "Transcript density needs the Xenium output directory "
                          "(transcripts.parquet); this dataset was loaded from a .zarr.")
            return
        multi = len(smps) > 1
        _t0 = _time.time()
        msgs = []
        steps = [(smp, gene) for smp in todo for gene in genes]
        for si, (smp, gene) in enumerate(steps):
            name = "all genes" if gene == _DENSITY_ALL else gene
            path = _density_cache_file(gene, smp["ds"])
            if os.path.exists(path):
                continue
            _set("running", f"Binning transcripts of {name}" + (f" ({smp['label']})…" if multi else "…"))
            scan = _density_scan(
                smp["data_dir"], gene,
                progress=lambda f, _si=si: jobs._job_progress(fraction=(_si + f) / len(steps)))
            jobs._check_dataset_gen(_gen)
            _density_write(path, scan, gene)
            msgs.append(f"{name}{' (' + smp['label'] + ')' if multi else ''}: {scan['total']:,} transcripts")
        with _density_lock:
            _density_state["version"] += 1
        _skip = (f" No transcripts for {', '.join(skipped)} (loaded from a .zarr): no grid there."
                 if skipped else "")
        _set("done", ("Binned " + "; ".join(msgs) + f" ({_time.time() - _t0:.1f} s)." if msgs
                      else "Grid loaded from the cache.") + _skip)
    except jobs._JobCancelled:
        raise
    except jobs._DatasetChanged:
        raise
    except ValueError as exc:
        _set("error", str(exc))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set("error", str(exc)[:300])


# ── layout of the three analysis tabs ─────────────────────────────────────────
_AN_NOTE_STYLE = {"fontSize": "10px", "color": core.MUTED, "lineHeight": "1.35", "padding": "2px 8px"}
_AN_ROW_STYLE = {"display": "flex", "alignItems": "center", "gap": "6px", "padding": "4px 8px",
                 "flexShrink": "0", "flexWrap": "wrap"}
_AN_INPUT_STYLE = {"backgroundColor": core.CARD_BG, "color": core.TEXT, "border": f"1px solid {core.BORDER}",
                   "borderRadius": "4px", "padding": "3px 6px", "fontSize": "11px"}
_AN_LABEL_STYLE = {"fontSize": "11px", "color": core.MUTED}
_AN_STATUS_STYLE = {"fontSize": "11px", "color": core.TEXT, "padding": "2px 8px", "minHeight": "16px",
                    "flexShrink": "0"}

_DE_CAVEAT = ("Wilcoxon rank-sum test per gene on log1p(count / library size × 10,000); "
              "log2 fold change of the group means of the normalised counts (pseudocount 1); "
              "Benjamini–Hochberg FDR over all tested genes. Every cell is treated as an "
              "independent observation, so p-values are anticonservative when the groups come "
              "from different samples or replicates (pseudobulk per sample is the proper test "
              "there) and when the groups were defined from the same expression data (clusters). "
              "Imputed genes are not tested. % A / % B count a cell when the gene has at least "
              "one transcript in it (≥ 1; a fractional SPLIT-corrected residue is not a "
              "detection); Rate = Σ counts / Σ total counts of the group per 10,000, which does "
              "not move with depth the way a percentage does. Owner / Leak fraction / Leak gate: "
              "see the Confound Controls tab; a flagged gene's signal or change is predicted to "
              "be more than half explained by transcripts of neighbouring cell types.")
_GS_NOTE = ("Score per cell = mean of the z-scored log-normalised expression of the set minus the "
            "mean of a control set drawn from the same expression bins (as Seurat AddModuleScore / "
            "scanpy score_genes; 25 bins, 50 control genes per bin, fixed seed). Computed for the "
            "active segmentation, counts mode and subset; recomputed when these change.")
_DENSITY_NOTE = ("Transcripts (not cells) per bin over the whole tissue: quality Q ≥ 20, gene "
                 "features only, the same filter the cell-feature matrix uses. The grid is read "
                 "from transcripts.parquet once per gene and sample and kept in the cache directory. "
                 "It does not depend on the segmentation or on a cell subset. With several samples "
                 "each sample's grid is drawn at that sample's position, on one colour scale; a "
                 "sample loaded from a .zarr has no transcripts and no grid.")


_DE_REST_OPTION = {"label": "All other cells", "value": "rest"}


def _de_tab(pseudobulk_controls=None, extra_modes=None, extra_controls=None):
    """The Differential Expression tab. *pseudobulk_controls*: the rows shown instead of
    the two group pickers in the "Pseudobulk across samples" mode (stats._pb_controls).
    *extra_modes* / *extra_controls*: further tests (radio options) and the rows shown for
    them (proximity._PROX_DE_MODES, proximity._prox_controls)."""
    return dbc.Tab(label="Differential Expression", tab_id="tab-de",
                   label_style={"fontSize": "12px", "padding": "4px 12px"},
                   children=[html.Div([
        html.Div([
            html.Span("Test:", style=_AN_LABEL_STYLE),
            dcc.RadioItems(id="de-mode", value="cells", inline=True,
                           options=[{"label": " Cells (Wilcoxon)", "value": "cells"},
                                    {"label": " Pseudobulk across samples", "value": "pseudobulk"}]
                                   + [dict(o) for o in (extra_modes or [])],
                           inputStyle={"marginRight": "3px"},
                           labelStyle={"marginRight": "12px", "cursor": "pointer", "fontSize": "11px",
                                       "color": core.TEXT},
                           style={"flex": "1"}),
            dbc.Button("Run", id="de-run-btn", color="primary", size="sm", n_clicks=0,
                       style={"fontSize": "11px", "padding": "2px 12px"}),
        ], style=_AN_ROW_STYLE),
        html.Div([
            html.Div([
                html.Span("A:", style=_AN_LABEL_STYLE),
                dcc.Dropdown(id="de-group-a", options=[dict(_DE_REST_OPTION)], value=None, placeholder="Group A…",
                             style={"flex": "1", "minWidth": "150px", "fontSize": "11px"}),
                html.Span("B:", style=_AN_LABEL_STYLE),
                dcc.Dropdown(id="de-group-b", options=[dict(_DE_REST_OPTION)], value="rest", placeholder="Group B…",
                             style={"flex": "1", "minWidth": "150px", "fontSize": "11px"}),
            ], style=_AN_ROW_STYLE),
            html.Div([
                html.Span(id="de-selection-info", children="No current selection.", style=_AN_LABEL_STYLE),
                dcc.Input(id="de-group-name", type="text", placeholder="Group name", debounce=False,
                          style={**_AN_INPUT_STYLE, "width": "130px"}),
                dbc.Button("Save current selection as group", id="de-save-group-btn", color="secondary",
                           size="sm", n_clicks=0, style={"fontSize": "11px", "padding": "2px 8px"}),
                dbc.Button("Delete group in A", id="de-delete-group-btn", color="secondary", outline=True,
                           size="sm", n_clicks=0, style={"fontSize": "11px", "padding": "2px 8px"}),
            ], style=_AN_ROW_STYLE),
        ], id="de-cells-controls", style={"flexShrink": "0"}),
        html.Div(pseudobulk_controls or [], id="de-pb-controls",
                 style={"display": "none", "flexShrink": "0"}),
        html.Div(extra_controls or [], id="de-prox-controls",
                 style={"display": "none", "flexShrink": "0"}),
        html.Div([
            dcc.Checklist(id="de-leak-gate",
                          options=[{"label": html.Span(" Gene ownership + leak gate (needs a cell-type "
                                                       "annotation)", style={"fontSize": "11px", "color": core.TEXT}),
                                    "value": "on"}],
                          value=["on"], inputStyle={"marginRight": "3px"}, style={"display": "inline-flex"}),
            html.Span("Volcano x axis:", style={**_AN_LABEL_STYLE, "marginLeft": "10px"}),
            dcc.RadioItems(id="de-lfc-axis", value="mle", inline=True,
                           options=[{"label": " log2FC (MLE)", "value": "mle"},
                                    {"label": " shrunken log2FC (pseudobulk)", "value": "shrunk"}],
                           inputStyle={"marginRight": "3px"},
                           labelStyle={"marginRight": "10px", "cursor": "pointer", "fontSize": "11px",
                                       "color": core.TEXT}),
        ], style=_AN_ROW_STYLE),
        html.Div(id="de-status", style={**_AN_STATUS_STYLE, "whiteSpace": "pre-wrap"}),
        html.Div(id="de-pb-samples", style={"padding": "0 8px", "flexShrink": "0"}),
        html.Div(_DE_CAVEAT, id="de-caveat", style=_AN_NOTE_STYLE),
        html.Div([
            html.Div(dcc.Graph(id="de-volcano-plot", responsive=True,
                               figure=figures._make_placeholder_fig("No differential expression computed yet."),
                               config={"displayModeBar": "hover",
                                       "toImageButtonOptions": {"format": "svg", "filename": "xenium_de_volcano"}},
                               style={"width": "100%", "height": "100%"}),
                     id="de-plot-container", style={"flex": "1", "minWidth": "0", "minHeight": "260px"}),
            html.Div([
                dash_table.DataTable(
                    id="de-table", data=[], columns=_de_table_columns(),
                    sort_action="native", filter_action="native", page_size=12,
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": core.CARD_BG, "color": core.TEXT, "fontWeight": "600",
                                  "border": f"1px solid {core.BORDER}", "fontSize": "11px"},
                    style_cell={"backgroundColor": "#0d1117", "color": core.TEXT, "fontSize": "11px",
                                "border": f"1px solid {core.BORDER}", "padding": "2px 5px",
                                "fontFamily": "inherit", "minWidth": "38px", "maxWidth": "90px"},
                    style_filter={"backgroundColor": core.CARD_BG, "color": core.TEXT},
                    style_data_conditional=[{"if": {"filter_query": '{leak_flag} = "FLAG"'},
                                             "color": "#d29922"}],
                ),
                html.Div([
                    dbc.Button("Download CSV", id="de-download-btn", color="secondary", size="sm",
                               n_clicks=0, style={"fontSize": "11px", "padding": "2px 8px"}),
                    html.Span("Click a point or a table cell to select the gene.",
                              style={**_AN_LABEL_STYLE, "marginLeft": "8px"}),
                    dcc.Download(id="de-download"),
                ], style={"padding": "4px 0"}),
            ], style={"flex": "1.25", "minWidth": "0", "overflow": "auto", "padding": "0 6px"}),
        ], style={"display": "flex", "flex": "1", "minHeight": "320px", "gap": "6px"}),
        ui._plot_dim_controls("de"),
    ], style={"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0",
              "overflowY": "auto"})])


def _geneset_tab():
    return dbc.Tab(label="Gene Sets", tab_id="tab-geneset",
                   label_style={"fontSize": "12px", "padding": "4px 12px"},
                   children=[html.Div([
        html.Div([
            html.Span("Saved:", style=_AN_LABEL_STYLE),
            dcc.Dropdown(id="geneset-select", options=[], value=None, placeholder="Saved gene sets…",
                         style={"flex": "1", "minWidth": "160px", "fontSize": "11px"}),
            html.Span("Name:", style=_AN_LABEL_STYLE),
            dcc.Input(id="geneset-name", type="text", placeholder="Gene set name",
                      style={**_AN_INPUT_STYLE, "width": "160px"}),
            dbc.Button("Save & score", id="geneset-run-btn", color="primary", size="sm", n_clicks=0,
                       style={"fontSize": "11px", "padding": "2px 12px"}),
            dbc.Button("Delete", id="geneset-delete-btn", color="secondary", outline=True, size="sm",
                       n_clicks=0, style={"fontSize": "11px", "padding": "2px 8px"}),
        ], style=_AN_ROW_STYLE),
        dcc.Textarea(id="geneset-genes", value="",
                     placeholder="Paste gene names, separated by new lines or commas…",
                     style={**_AN_INPUT_STYLE, "margin": "4px 8px", "minHeight": "120px",
                            "flex": "1", "fontFamily": "monospace", "resize": "vertical"}),
        html.Div(id="geneset-status", style={**_AN_STATUS_STYLE, "whiteSpace": "pre-wrap"}),
        html.Div(_GS_NOTE, id="geneset-note", style=_AN_NOTE_STYLE),
    ], style={"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0",
              "overflowY": "auto"})])


def _density_tab():
    return dbc.Tab(label="Transcript Density", tab_id="tab-density",
                   label_style={"fontSize": "12px", "padding": "4px 12px"},
                   children=[html.Div([
        html.Div([
            html.Span("Gene:", style=_AN_LABEL_STYLE),
            dcc.Dropdown(id="density-gene", options=[], value=None, placeholder="Gene…",
                         style={"flex": "1", "minWidth": "160px", "fontSize": "11px"}),
            html.Span("Bin (µm):", style=_AN_LABEL_STYLE),
            dcc.Dropdown(id="density-bin", clearable=False, value=10,
                         options=[{"label": str(b), "value": b} for b in _DENSITY_BINS],
                         style={"width": "70px", "fontSize": "11px"}),
            dbc.Button("Compute", id="density-run-btn", color="primary", size="sm", n_clicks=0,
                       style={"fontSize": "11px", "padding": "2px 12px"}),
        ], style=_AN_ROW_STYLE),
        html.Div([
            dcc.Checklist(id="density-enable",
                          options=[{"label": html.Span(" Show heat layer under the cells",
                                                       style={"fontSize": "11px", "color": core.TEXT}),
                                    "value": "on"}],
                          value=["on"], inputStyle={"marginRight": "4px"},
                          labelStyle={"display": "flex", "alignItems": "center", "cursor": "pointer"},
                          style={"display": "inline-flex"}),
            html.Span("Opacity:", style={**_AN_LABEL_STYLE, "marginLeft": "10px"}),
            html.Div(dcc.Slider(id="density-opacity", min=0, max=1, step=0.05, value=0.7,
                                marks=None, tooltip={"placement": "bottom"}),
                     style={"width": "160px"}),
            html.Span("Colour max:", style={**_AN_LABEL_STYLE, "marginLeft": "10px"}),
            dcc.Input(id="density-max", type="number", placeholder="auto", min=0, debounce=True,
                      style=ui._PLOT_CTRL_INPUT),
        ], style=_AN_ROW_STYLE),
        html.Div(id="density-status", style=_AN_STATUS_STYLE),
        html.Div(id="density-hint", style=_AN_STATUS_STYLE),
        html.Div(_DENSITY_NOTE, id="density-note", style=_AN_NOTE_STYLE),
    ], style={"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0",
              "overflowY": "auto"})])
