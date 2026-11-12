"""Statistics across samples and in space: the sample → condition design, pseudobulk
differential expression (DESeq2 / edgeR through rpy2), composition tests, Moran's I and
ligand-receptor scores (compute + layouts)."""

import re
import threading
import numpy as np
import pandas as pd
import scipy.sparse as sp
from dash import dcc, html
from dash import dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import time as _time

from . import core, jobs, dataview, ui, figures, analysis, spatial


# ─── Samples and their conditions ─────────────────────────────────────────────
# One assignment for every test that compares samples (pseudobulk DE, composition):
# each loaded sample is in condition A, in condition B, or excluded, with an optional
# pairing / batch label. Kept per sample identity (its directory / .zarr path), so it
# survives removing and re-adding a sample; cleared when another dataset is loaded.

_design_state: dict = {"assign": {},                    # sample key -> {"cond": "A"|"B"|"", "batch": str}
                       "names": {"A": "A", "B": "B"},   # display names of the two conditions
                       "version": 0}
_design_lock = threading.Lock()

_PB_MIN_CELLS = 10         # a sample contributing fewer cells is dropped
_PB_MIN_COUNT = 10         # a gene with a smaller total count over the used samples is not tested
_COMP_MIN_SAMPLES = 3      # samples per condition below which no composition test is run

_comp_state: dict = {"result": None, "message": "", "version": 0}
_comp_lock = threading.Lock()

_moran_state: dict = {"status": "idle", "message": "", "result": None}
_moran_lock = threading.Lock()
_MORAN_K = 6               # neighbours per cell of the spatial graph
_MORAN_CHUNK_BYTES = 64e6  # dense block of expression values handled at a time

_lr_state: dict = {"status": "idle", "message": "", "result": None}
_lr_lock = threading.Lock()
_LR_MIN_CELLS = 10         # a cell type with fewer cells is not scored
_LR_MIN_EDGES = 20         # adjacent cell pairs two types need to count as neighbours
_LR_MIN_FRAC = 0.10        # fraction of cells that must express the ligand / the receptor
_LR_MAX_PERM = 10_000


def _stats_reset() -> None:
    """Forget the design and every result (another dataset was loaded; tests)."""
    with _design_lock:
        _design_state.update({"assign": {}, "names": {"A": "A", "B": "B"}})
        _design_state["version"] += 1
    with _comp_lock:
        _comp_state.update({"result": None, "message": ""})
        _comp_state["version"] += 1
    with _moran_lock:
        _moran_state.update({"status": "idle", "message": "", "result": None})
    with _lr_lock:
        _lr_state.update({"status": "idle", "message": "", "result": None})


def _design_samples() -> list:
    """The loaded samples (the primary one and the extra ones) with their assignment:
    [{index, label (S0…), key, name, n_cells, cond ("A" / "B" / ""), batch}]."""
    dsets = [core.DATA] + list(dataview.EXTRA_DATASETS)
    with _design_lock:
        assign = {k: dict(v) for k, v in _design_state["assign"].items()}
    out = []
    for i, ds in enumerate(dsets):
        key = dataview._ds_key(ds)
        a = assign.get(key, {})
        out.append({"index": i, "label": f"S{i}", "key": key, "name": dataview._ds_name(ds),
                    "n_cells": int(len(ds["df"])), "cond": a.get("cond") or "",
                    "batch": a.get("batch") or ""})
    return out


def _design_set(rows, names=None) -> None:
    """Store the assignment: rows = [{key, cond, batch}], names = {"A": …, "B": …}."""
    with _design_lock:
        for r in rows:
            cond = r.get("cond") if r.get("cond") in ("A", "B") else ""
            batch = re.sub(r"\s+", " ", str(r.get("batch") or "").strip())
            _design_state["assign"][str(r["key"])] = {"cond": cond, "batch": batch}
        if names:
            for c in ("A", "B"):
                nm = re.sub(r"\s+", " ", str(names.get(c) or "").strip())
                _design_state["names"][c] = nm or c
        _design_state["version"] += 1


def _design_names() -> dict:
    with _design_lock:
        return dict(_design_state["names"])


def _design_of(samples) -> tuple:
    """(conds, batches) of analysis samples / view tables, by their key."""
    with _design_lock:
        assign = {k: dict(v) for k, v in _design_state["assign"].items()}
    conds = [(assign.get(s["key"], {}).get("cond") or "") for s in samples]
    batches = [(assign.get(s["key"], {}).get("batch") or "") for s in samples]
    return conds, batches


def _design_summary() -> str:
    """One line for the tabs that use the design."""
    rows = _design_samples()
    names = _design_names()
    if len(rows) < 2:
        return ("One sample is loaded. Tests between conditions need several samples: add them "
                "with '+ Add sample' and assign each to a condition.")
    parts = []
    for c in ("A", "B"):
        labs = [r["label"] for r in rows if r["cond"] == c]
        nm = f"{c} ({names[c]})" if names[c] != c else c
        parts.append(f"{nm}: {', '.join(labs) if labs else 'no samples'}")
    ex = [r["label"] for r in rows if not r["cond"]]
    if ex:
        parts.append(f"excluded: {', '.join(ex)}")
    if any(r["batch"] for r in rows if r["cond"]):
        parts.append("pairing / batch: " + ", ".join(f"{r['label']}={r['batch'] or '?'}"
                                                      for r in rows if r["cond"]))
    return " · ".join(parts)


def _view_tables(seg_source="xenium") -> list:
    """The cell tables of the active view, with or without a count matrix: one dict per
    sample (label, df, ds, key, name, x_offset), in the order of analysis._an_samples."""
    alt = analysis._an_alt_res(seg_source)
    if alt is not None:
        return [{"label": core._seg_tool(seg_source), "df": alt["cells_df"], "ds": core.DATA,
                 "key": dataview._ds_key(core.DATA), "name": dataview._ds_name(core.DATA),
                 "x_offset": 0.0}]
    extras = list(dataview.EXTRA_DATASETS)
    out = [{"label": "S0" if extras else "xenium", "df": core.DATA["df"], "ds": core.DATA,
            "key": dataview._ds_key(core.DATA), "name": dataview._ds_name(core.DATA), "x_offset": 0.0}]
    for i, eds in enumerate(extras):
        out.append({"label": f"S{i + 1}", "df": eds["df"], "ds": eds, "key": dataview._ds_key(eds),
                    "name": dataview._ds_name(eds), "x_offset": float(eds.get("x_offset", 0.0))})
    return out


def _raw_cluster_labels(samples, method, alt) -> list:
    """Per sample: the active cluster set's label of each cell without the sample prefix
    the multi-sample view adds ("3", not "S1:C3"); None for a sample without the set.
    Whether cluster 3 of one sample is cluster 3 of another depends on how the samples
    were clustered: the ids are comparable only when they were clustered together."""
    labs = analysis._an_cluster_labels(samples, method, False, alt)
    out = []
    for lab in labs:
        if lab is None:
            out.append(None)
            continue
        out.append(np.array([None if x is None else re.sub(r"^S\d+:C", "", str(x)) for x in lab],
                            dtype=object))
    return out


def _niche_labels(samples) -> list:
    """Per sample: the niche of each cell (None: no niche / no niche column)."""
    out = []
    for s in samples:
        df = s["df"]
        if "niche" not in df.columns:
            out.append(None)
            continue
        lab = spatial._niche_str(df["niche"]).to_numpy(dtype=object)
        lab[lab == spatial._NICHE_NONE] = None
        out.append(lab)
    return out


def _bh(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values; NaN entries are left out and stay NaN."""
    p = np.asarray(p, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m:
        v = p[ok]
        order = np.argsort(v, kind="stable")
        adj = v[order] * m / np.arange(1, m + 1)
        adj = np.minimum(np.minimum.accumulate(adj[::-1])[::-1], 1.0)
        res = np.empty(m)
        res[order] = adj
        out[ok] = res
    return out


# ─── Pseudobulk differential expression ───────────────────────────────────────
_PB_CAVEAT = ("Pseudobulk: the raw integer counts of the chosen cells are summed per sample and "
              "the samples (not the cells) are the observations. DESeq2 (Wald test, design "
              "~ condition, or ~ batch + condition with pairing / batch labels; independent "
              "filtering at alpha 0.05; log2 fold change A / B, shrunken with apeglm when that "
              "package is installed, else DESeq2's normal prior, else unshrunken and labelled so); "
              "edgeR's quasi-likelihood F-test is used, and named, when DESeq2 is not installed. "
              "Always the original counts: SPLIT-corrected counts are modified, non-integer values "
              "and imputed genes are not counts. A condition needs at least 2 samples; with fewer "
              "than 3 the dispersion estimates are unreliable. Samples from one tissue section "
              "are not biological replicates.")


def _pb_population_options(seg_source, method, color_by) -> list:
    """Cell populations the counts can be summed over."""
    samples = _view_tables(seg_source)
    alt = analysis._an_alt_res(seg_source)
    opts = [{"label": "All cells", "value": "all"}]
    am = analysis._an_annot_method(color_by, samples, alt)
    seen = set()
    for lab in analysis._an_celltype_labels(samples, am, alt):
        if lab is not None:
            seen.update(x for x in pd.unique(lab) if x is not None)
    for c in sorted(seen, key=str):
        opts.append({"label": f"Cell type ({core._ANNOT_METHODS.get(am, am)}): {c}",
                     "value": f"celltype:{c}"})
    seen = set()
    for lab in _raw_cluster_labels(samples, method, alt):
        if lab is not None:
            seen.update(x for x in pd.unique(lab) if x is not None)
    for c in sorted(seen, key=lambda v: (0, int(v), "") if str(v).isdigit() else (1, 0, str(v))):
        opts.append({"label": f"Cluster {c} (same id in every sample)", "value": f"cluster:{c}"})
    with core._roi_lock:
        rois = list(core._roi_state["rois"])
    for cls in dict.fromkeys(r["cls"] for r in rois):
        opts.append({"label": f"ROI class: {cls} (all its ROIs)", "value": f"roicls:{cls}"})
    for r in rois:
        opts.append({"label": f"ROI: {r['name']} ({r['cls']})", "value": f"roi:{r['cls']}|{r['name']}"})
    with analysis._de_lock:
        groups = list(analysis._de_state["groups"])
        sel = analysis._de_state.get("selection")
    for g in groups:
        opts.append({"label": f"Group: {g['name']}", "value": f"grp:{g['name']}"})
    if sel and sel.get("polygon_xy"):
        opts.append({"label": f"Current selection ({int(sel.get('n') or 0):,} cells)",
                     "value": "sel:current"})
    return opts


def _pb_population_masks(spec, samples, seg_source, method, color_by) -> tuple:
    """(per-sample boolean masks over the cells, label) of a population picker value.
    Raises ValueError with a message for the user."""
    spec = str(spec or "")
    alt = analysis._an_alt_res(seg_source)
    if spec == "all":
        return [np.ones(len(s["df"]), dtype=bool) for s in samples], "all cells"
    if spec.startswith("cluster:"):
        want = spec[8:]
        labs = _raw_cluster_labels(samples, method, alt)
        if all(l is None for l in labs):
            raise ValueError("The active view has no cluster set.")
        return ([np.zeros(len(s["df"]), dtype=bool) if l is None else (l == want)
                 for s, l in zip(samples, labs)], f"cluster {want}")
    if spec.startswith("roicls:"):
        cls = spec[7:]
        with core._roi_lock:
            polys = [r["polygon_xy"] for r in core._roi_state["rois"] if r["cls"] == cls]
        if not polys:
            raise ValueError(f"ROI class '{cls}' no longer exists.")
        masks = []
        for s in samples:
            m = np.zeros(len(s["df"]), dtype=bool)
            for p in polys:
                m |= analysis._an_polygon_mask(p, s)
            masks.append(m)
        return masks, f"ROI class {cls}"
    if spec == "rest" or not spec:
        raise ValueError("Pick the cell population to sum the counts over.")
    return analysis._de_group_masks(spec, samples, seg_source, method, color_by, False)


def _pb_sum_counts(s: dict, mask) -> tuple:
    """(count per matrix column summed over the masked cells of sample *s*, number of
    cells summed). The counts must be whole numbers."""
    csc = s["csc"]
    rows = np.asarray(s["rows"])
    sel = rows[np.asarray(mask, dtype=bool) & (rows >= 0)]
    data = csc.data
    if data.dtype.kind not in "iu" and len(data) and not np.array_equal(data, np.rint(data)):
        raise ValueError(f"The count matrix of {s['label']} holds non-integer values: pseudobulk "
                         f"differential expression needs raw counts.")
    w = np.zeros(csc.shape[0], dtype=np.float64)
    w[sel] = 1.0
    tot = np.asarray(csc.T @ w, dtype=np.float64).ravel()
    return np.rint(tot).astype(np.int64), int(len(sel))


def _pb_design_matrix(conds, batches) -> np.ndarray:
    """Model matrix of ~ condition or ~ batch + condition: intercept, batch dummies
    (first level dropped), condition (A = 1) last."""
    n = len(conds)
    cols = [np.ones(n)]
    if batches is not None:
        levels = sorted(set(batches))
        for lv in levels[1:]:
            cols.append(np.array([1.0 if b == lv else 0.0 for b in batches]))
    cols.append(np.array([1.0 if c == "A" else 0.0 for c in conds]))
    return np.column_stack(cols)


def _design_check(conds, batches, min_per_cond=2) -> tuple:
    """(batches or None, warnings) for the samples of a test, after checking that it can
    be run: at least *min_per_cond* samples per condition, batch labels on all samples
    or on none, and a batch term that is not confounded with the condition. Raises
    ValueError with the reason otherwise."""
    names = _design_names()
    warn = []
    for c in ("A", "B"):
        k = sum(1 for x in conds if x == c)
        nm = f"{c} ({names[c]})" if names[c] != c else c
        if k < min_per_cond:
            raise ValueError(f"Condition {nm} has {k} usable sample{'s' if k != 1 else ''}; at "
                             f"least {min_per_cond} per condition are needed.")
        if k < 3:
            warn.append(f"condition {nm} has only {k} samples: variance estimates from fewer "
                        f"than 3 replicates are unreliable")
    given = [bool(b) for b in batches]
    if not any(given):
        return None, warn
    if not all(given):
        raise ValueError("Pairing / batch labels are set for some samples only: give every "
                         "included sample a label, or none.")
    X = _pb_design_matrix(conds, batches)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("The pairing / batch labels are confounded with the condition (each "
                         "label occurs in one condition only, or every sample has its own "
                         "label): ~ batch + condition cannot be estimated. Remove the labels or "
                         "pair samples across the conditions.")
    if X.shape[0] <= X.shape[1]:
        raise ValueError("No residual degrees of freedom are left for ~ batch + condition with "
                         "these samples.")
    return list(batches), warn


def _pb_prepare(samples, masks, conds, batches, min_cells=_PB_MIN_CELLS,
                min_count=_PB_MIN_COUNT) -> dict:
    """The pseudobulk matrix and its design.

    samples / masks: the analysis samples (original counts) and the cells of the
    population in each; conds / batches: the assignment of each sample. Samples that
    are excluded, have no usable matrix rows or contribute fewer than *min_cells* cells
    are left out (and listed); genes whose total count over the used samples is below
    *min_count* are not tested. Returns counts (genes × used samples, int64), genes,
    cond, batch (None: no batch term), design, labels, sample_rows (every sample with
    cells, library size and whether / why not it was used), warnings, n_filtered."""
    min_cells = max(1, int(min_cells if min_cells is not None else _PB_MIN_CELLS))
    min_count = max(0, int(min_count if min_count is not None else _PB_MIN_COUNT))
    genes = analysis._an_common_genes(samples)
    if not genes:
        raise ValueError("The samples of this view share no measured genes.")
    rows, cols, used = [], [], []
    for i, s in enumerate(samples):
        tot, n = _pb_sum_counts(s, masks[i])
        v = tot[np.array([s["gni"][g] for g in genes], dtype=np.int64)]
        row = {"label": s["label"], "name": s["name"], "cond": conds[i], "batch": batches[i],
               "n_cells": n, "lib_size": int(v.sum()), "used": False, "note": ""}
        if conds[i] not in ("A", "B"):
            row["note"] = "excluded"
        elif n < min_cells:
            row["note"] = f"dropped: fewer than {min_cells} cells"
        elif row["lib_size"] <= 0:
            row["note"] = "dropped: no counts"
        else:
            row["used"] = True
            used.append(i)
            cols.append(v)
        rows.append(row)
    u_cond = [conds[i] for i in used]
    try:
        u_batch, warn = _design_check(u_cond, [batches[i] for i in used])
    except ValueError as exc:
        dropped = [r for r in rows if r["note"].startswith("dropped")]
        extra = (" Dropped: " + "; ".join(f"{r['label']} ({r['n_cells']} cells)" for r in dropped) + "."
                 if dropped else "")
        raise ValueError(str(exc) + extra) from None
    counts = np.column_stack(cols).astype(np.int64)
    keep = counts.sum(axis=1) >= min_count
    if int(keep.sum()) < 2:
        raise ValueError(f"Fewer than 2 genes have a total count of at least {min_count} over "
                         f"the used samples.")
    if counts.max() >= 2 ** 31 - 1:
        raise ValueError("A pseudobulk count exceeds the 32-bit integer range R works with.")
    return {"counts": counts[keep], "genes": [g for g, k in zip(genes, keep) if k],
            "cond": u_cond, "batch": u_batch,
            "design": "~ batch + condition" if u_batch else "~ condition",
            "labels": [samples[i]["label"] for i in used], "sample_rows": rows,
            "warnings": warn, "n_filtered": int((~keep).sum()), "n_genes_all": len(genes),
            "min_cells": min_cells, "min_count": min_count}


_PB_R_FUNCS = r"""
.xe_pb_deseq2 <- function(counts, genes, samples, cond, batch, shrink) {
  suppressPackageStartupMessages(library(DESeq2))
  storage.mode(counts) <- "integer"
  dimnames(counts) <- list(genes, samples)
  cd <- data.frame(condition = factor(cond, levels = c("B", "A")), row.names = samples)
  if (!is.null(batch)) { cd$batch <- factor(batch); dsg <- ~ batch + condition } else dsg <- ~ condition
  dds <- suppressMessages(DESeqDataSetFromMatrix(countData = counts, colData = cd, design = dsg))
  note <- ""
  dds <- tryCatch(suppressMessages(DESeq(dds, quiet = TRUE)), error = function(e) {
    note <<- paste0("DESeq() stopped (", conditionMessage(e), "): gene-wise dispersion estimates ",
                    "were used without the trend fit")
    d <- tryCatch(estimateSizeFactors(dds), error = function(e2) estimateSizeFactors(dds, type = "poscounts"))
    d <- suppressMessages(estimateDispersionsGeneEst(d))
    dispersions(d) <- mcols(d)$dispGeneEst
    suppressMessages(nbinomWaldTest(d))
  })
  res <- results(dds, name = "condition_A_vs_B", alpha = 0.05)
  shr <- rep(NA_real_, nrow(res)); shr_type <- "none"
  if (isTRUE(shrink)) {
    for (ty in c(if (requireNamespace("apeglm", quietly = TRUE)) "apeglm", "normal")) {
      s <- tryCatch(suppressMessages(lfcShrink(dds, coef = "condition_A_vs_B", type = ty, quiet = TRUE)),
                    error = function(e) NULL)
      if (!is.null(s)) { shr <- s$log2FoldChange; shr_type <- ty; break }
    }
  }
  list(baseMean = as.numeric(res$baseMean), log2FoldChange = as.numeric(res$log2FoldChange),
       lfcSE = as.numeric(res$lfcSE), stat = as.numeric(res$stat), pvalue = as.numeric(res$pvalue),
       padj = as.numeric(res$padj), shrunk = as.numeric(shr), shrink_type = shr_type,
       size_factors = as.numeric(sizeFactors(dds)), note = note,
       version = as.character(packageVersion("DESeq2")))
}
.xe_pb_edger <- function(counts, genes, samples, cond, batch) {
  suppressPackageStartupMessages(library(edgeR))
  dimnames(counts) <- list(genes, samples)
  condition <- factor(cond, levels = c("B", "A"))
  design <- if (!is.null(batch)) model.matrix(~ factor(batch) + condition) else model.matrix(~ condition)
  y <- DGEList(counts = counts)
  y <- calcNormFactors(y)
  y <- estimateDisp(y, design)
  fit <- glmQLFit(y, design)
  qlf <- glmQLFTest(fit, coef = ncol(design))
  tt <- qlf$table
  list(log2FoldChange = as.numeric(tt$logFC), logCPM = as.numeric(tt$logCPM), stat = as.numeric(tt$F),
       pvalue = as.numeric(tt$PValue), padj = as.numeric(p.adjust(tt$PValue, method = "BH")),
       size_factors = as.numeric(y$samples$lib.size * y$samples$norm.factors / 1e6), note = "",
       version = as.character(packageVersion("edgeR")))
}
"""

_PB_INSTALL_HINT = ('Pseudobulk differential expression needs the R package DESeq2 (or edgeR): in R, '
                    'run  install.packages("BiocManager"); BiocManager::install(c("DESeq2", "apeglm")).')


def _pb_fit(prep: dict, shrink: bool = True, engine: str = "auto") -> dict:
    """Fit the pseudobulk model of _pb_prepare() in R through rpy2.

    engine "auto": DESeq2 when installed, else edgeR (named in the result), else a
    ValueError with the install hint - never a test that ignores the count nature of
    the data. Returns table (gene, log2fc [unshrunken MLE], log2fc_shrunk [when a
    shrinkage estimator ran], lfc_se, base_mean, mean_a, mean_b, stat, pval, padj),
    engine, method_label, shrink, stat_name, size_factors, notes.

    Call from a thread whose rpy2 conversion context was set (see _run_pseudobulk_de)."""
    try:
        import rpy2.robjects as ro
        import rpy2.robjects.conversion as _rconv
        from rpy2.robjects.packages import isinstalled
    except Exception as exc:
        raise ValueError(f"rpy2 / R is not available ({exc}). " + _PB_INSTALL_HINT) from None
    _rconv.set_conversion(ro.default_converter)          # rpy2 thread rule: unconditional
    counts = np.asarray(prep["counts"], dtype=np.int64)
    n_g, n_s = counts.shape
    cond = list(prep["cond"])
    with _rconv.localconverter(ro.default_converter):
        have = {p: bool(isinstalled(p)) for p in ("DESeq2", "edgeR")}
        if engine == "auto":
            engine = "deseq2" if have["DESeq2"] else ("edger" if have["edgeR"] else None)
        if engine is None or not have["DESeq2" if engine == "deseq2" else "edgeR"]:
            raise ValueError(_PB_INSTALL_HINT)
        ro.r(_PB_R_FUNCS)
        r_counts = ro.r["matrix"](ro.IntVector(counts.ravel(order="F").tolist()), nrow=n_g, ncol=n_s)
        r_batch = ro.NULL if prep.get("batch") is None else ro.StrVector([str(b) for b in prep["batch"]])
        args = (r_counts, ro.StrVector(list(prep["genes"])), ro.StrVector(list(prep["labels"])),
                ro.StrVector(cond), r_batch)
        if engine == "deseq2":
            res = ro.globalenv[".xe_pb_deseq2"](*args, bool(shrink))
        else:
            res = ro.globalenv[".xe_pb_edger"](*args)
        get = lambda k: np.asarray(res.rx2(k), dtype=np.float64)
        txt = lambda k: str(res.rx2(k)[0])
        sf = get("size_factors")
        out = {"log2fc": get("log2FoldChange"), "stat": get("stat"), "pval": get("pvalue"),
               "padj": get("padj")}
        note, version = txt("note"), txt("version")
        if engine == "deseq2":
            out["lfc_se"], out["base_mean"] = get("lfcSE"), get("baseMean")
            shrink_type = txt("shrink_type")
            if shrink_type != "none":
                out["log2fc_shrunk"] = get("shrunk")
        else:
            shrink_type = "none"
    norm = counts / np.where(sf > 0, sf, np.nan)[None, :]
    a = np.array([c == "A" for c in cond])
    table = pd.DataFrame({"gene": list(prep["genes"]), "log2fc": out["log2fc"]})
    if "log2fc_shrunk" in out:
        table["log2fc_shrunk"] = out["log2fc_shrunk"]
    if "lfc_se" in out:
        table["lfc_se"] = out["lfc_se"]
    table["base_mean"] = out.get("base_mean", np.nanmean(norm, axis=1))
    table["mean_a"], table["mean_b"] = np.nanmean(norm[:, a], axis=1), np.nanmean(norm[:, ~a], axis=1)
    table["stat"], table["pval"], table["padj"] = out["stat"], out["pval"], out["padj"]
    with np.errstate(divide="ignore", invalid="ignore"):
        table["neg_log10_padj"] = -np.log10(np.maximum(table["padj"].to_numpy(), 1e-300))
    if engine == "deseq2":
        label = f"DESeq2 {version}, Wald test"
        stat_name, mean_unit = "Wald", "size-factor normalised counts"
    else:
        label = f"edgeR {version}, quasi-likelihood F-test (DESeq2 is not installed)"
        stat_name, mean_unit = "F", "counts per million (TMM)"
    return {"table": table, "engine": engine, "method_label": label, "shrink": shrink_type,
            "stat_name": stat_name, "mean_unit": mean_unit, "size_factors": sf,
            "notes": [note] if note else []}


def _run_pseudobulk_de(population, seg_source, method, color_by, counts_corrected=False,
                       min_cells=_PB_MIN_CELLS, min_count=_PB_MIN_COUNT, shrink=True,
                       engine="auto") -> None:
    """Background job: pseudobulk differential expression between the two conditions of
    the sample design, for one cell population. Publishes into the DE state, so the
    volcano, the table and the CSV of the Differential Expression tab show it."""
    _gen = jobs._dataset_gen
    try:                                   # rpy2 thread rule: before anything else
        import rpy2.robjects as _ro_mod
        import rpy2.robjects.conversion as _rconv
        _rconv.set_conversion(_ro_mod.default_converter)
    except Exception:
        pass                               # reported by _pb_fit with the install hint

    def _set(status, message, result=None):
        with analysis._de_lock:
            if jobs._stale_job(_gen, status, message):
                return
            analysis._de_state["status"] = status
            analysis._de_state["message"] = message
            if result is not None:
                analysis._de_state["result"] = result

    try:
        _t0 = _time.time()
        _set("running", "Summing counts per sample…")
        jobs._job_progress("Summing counts per sample…", 0.05)
        # always the original counts (see _PB_CAVEAT)
        samples = analysis._an_samples(seg_source, False)
        if not samples:
            _set("error", "The active view has no expression matrix.")
            return
        if analysis._an_alt_res(seg_source) is not None:
            _set("error", "A Baysor / Proseg run is one sample: pseudobulk differential "
                          "expression compares the samples loaded with the Xenium segmentation "
                          "('+ Add sample'). Switch Segmentation to Xenium.")
            return
        masks, pop_label = _pb_population_masks(population, samples, seg_source, method, color_by)
        conds, batches = _design_of(samples)
        prep = _pb_prepare(samples, masks, conds, batches, min_cells, min_count)
        jobs._job_check()
        _set("running", f"Fitting {prep['design']} on {len(prep['labels'])} samples × "
                        f"{len(prep['genes']):,} genes in R…")
        jobs._job_progress("Fitting the model in R…", 0.3)
        fit = _pb_fit(prep, shrink=bool(shrink), engine=engine)
        jobs._check_dataset_gen(_gen)
        jobs._job_check()
        names = _design_names()
        table = fit["table"]
        for r, lab in ((r, r["label"]) for r in prep["sample_rows"]):
            r["size_factor"] = (float(fit["size_factors"][prep["labels"].index(lab)])
                                if r["used"] else None)
        n_a, n_b = prep["cond"].count("A"), prep["cond"].count("B")
        shrunk = "log2fc_shrunk" in table.columns
        lfc_col = "log2fc_shrunk" if shrunk else "log2fc"
        lfc_note = (f"{fit['shrink']}-shrunken" if shrunk else "unshrunken (MLE)")
        n_na = int(table["padj"].isna().sum())
        n_sig = int((table["padj"] < 0.05).sum())
        notes = list(fit["notes"])
        if counts_corrected:
            notes.append("the counts mode is SPLIT-corrected, but pseudobulk uses the original "
                         "integer counts")
        if bool(shrink) and not shrunk and fit["engine"] == "deseq2":
            notes.append("no shrinkage estimator ran: log2 fold changes are unshrunken")
        res = {"kind": "pseudobulk", "table": table, "population": pop_label,
               "label_a": names["A"], "label_b": names["B"], "n_a": n_a, "n_b": n_b,
               "title": (f"Pseudobulk, {pop_label}: A {names['A']} ({n_a} samples) vs "
                         f"B {names['B']} ({n_b} samples) — {fit['method_label']}, {prep['design']}"),
               "lfc_col": lfc_col, "lfc_title": f"log2 fold change (A / B), {lfc_note}",
               "stat_name": fit["stat_name"], "shrink": fit["shrink"], "engine": fit["engine"],
               "method_label": fit["method_label"], "mean_unit": fit["mean_unit"],
               "design": prep["design"], "samples": prep["sample_rows"],
               "n_genes": len(table), "n_filtered": prep["n_filtered"], "n_padj_na": n_na,
               "min_count": prep["min_count"], "min_cells": prep["min_cells"],
               "warnings": prep["warnings"], "notes": notes,
               "seg_source": seg_source or "xenium", "corrected": False,
               "n_overlap": 0, "n_samples": len(prep["labels"]), "seconds": _time.time() - _t0}
        used = [r for r in prep["sample_rows"] if r["used"]]
        dropped = [r for r in prep["sample_rows"] if r["note"].startswith("dropped")]
        msg = (f"Pseudobulk ({pop_label}): A {names['A']} [{', '.join(r['label'] for r in used if r['cond'] == 'A')}] "
               f"vs B {names['B']} [{', '.join(r['label'] for r in used if r['cond'] == 'B')}]; "
               f"{fit['method_label']}, design {prep['design']}, log2FC {lfc_note}. "
               f"{len(table):,} genes tested ({prep['n_filtered']:,} with a total count below "
               f"{prep['min_count']} not tested"
               + (f"; {n_na:,} without an adjusted p-value after independent filtering / outlier "
                  f"flagging" if n_na else "")
               + f"), {n_sig:,} with FDR < 0.05 ({res['seconds']:.1f} s). Original counts.")
        if dropped:
            msg += (" Dropped (fewer than " + str(prep["min_cells"]) + " cells or no counts): "
                    + ", ".join(f"{r['label']} ({r['n_cells']} cells)" for r in dropped) + ".")
        for w in prep["warnings"]:
            msg += f"\n⚠ {w[0].upper() + w[1:]}."
        for nt in notes:
            msg += f"\nNote: {nt}."
        _set("done", msg, result=res)
        print(f"  Pseudobulk DE: {msg}", flush=True)
    except jobs._JobCancelled:
        raise
    except jobs._DatasetChanged:
        raise
    except ValueError as exc:
        _set("error", str(exc))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set("error", f"Pseudobulk differential expression failed: {str(exc)[:400]}")


def _pb_csv_header(res: dict) -> str:
    """Comment lines of the CSV download of a pseudobulk result."""
    lines = [f"# Pseudobulk differential expression; population: {res['population']}",
             f"# A: {res['label_a']} ({res['n_a']} samples); B: {res['label_b']} ({res['n_b']} samples); "
             f"log2fc = A / B",
             f"# method: {res['method_label']}; design: {res['design']}; "
             f"log2fc: unshrunken MLE" + (f"; log2fc_shrunk: {res['shrink']}" if res["shrink"] != "none" else ""),
             f"# counts: original integer counts summed per sample; segmentation: {res['seg_source']}; "
             f"means: {res['mean_unit']}",
             f"# genes: {res['n_genes']} tested, {res['n_filtered']} with total count < {res['min_count']} "
             f"not tested, {res['n_padj_na']} without adjusted p-value"]
    for r in res["samples"]:
        sf = "" if r.get("size_factor") is None else f"; size factor {r['size_factor']:.4g}"
        lines.append(f"# sample {r['label']} ({r['name']}): condition {r['cond'] or '-'}"
                     + (f"; batch {r['batch']}" if r["batch"] else "")
                     + f"; {r['n_cells']} cells; library size {r['lib_size']}{sf}"
                     + (f"; {r['note']}" if r["note"] else ""))
    for w in list(res.get("warnings") or []) + list(res.get("notes") or []):
        lines.append(f"# note: {w}")
    return "\n".join(lines) + "\n"


def _samples_table(rows, columns) -> "html.Table":
    """A small table of per-sample facts: columns = [(key, header, format or None)]."""
    th = {"padding": "1px 8px", "textAlign": "left", "color": core.MUTED, "fontWeight": "600",
          "borderBottom": f"1px solid {core.BORDER}"}
    td = {"padding": "1px 8px", "color": core.TEXT}

    def _fmt(v, f):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "–"
        return f(v) if f else str(v)

    return html.Table([
        html.Thead(html.Tr([html.Th(h, style=th) for _, h, _ in columns])),
        html.Tbody([html.Tr([html.Td(_fmt(r.get(k), f), style={**td, **({} if r.get("used", True)
                                                                         else {"color": core.MUTED})})
                             for k, _, f in columns]) for r in rows]),
    ], style={"fontSize": "10.5px", "borderCollapse": "collapse", "margin": "2px 0"})


def _pb_samples_view(res):
    """Per-sample cells / library sizes / size factors of a pseudobulk result."""
    if not res or res.get("kind") != "pseudobulk":
        return []
    cols = [("label", "Sample", None), ("name", "Name", lambda v: str(v)[:38]),
            ("cond", "Condition", lambda v: v or "–"), ("batch", "Batch", lambda v: v or "–"),
            ("n_cells", "Cells summed", lambda v: f"{v:,}"),
            ("lib_size", "Library size", lambda v: f"{v:,}"),
            ("size_factor", "Size factor", lambda v: f"{v:.3g}"), ("note", "", None)]
    return [_samples_table(res["samples"], cols)]


def _pb_controls() -> list:
    """Rows of the Differential Expression tab in the pseudobulk mode."""
    S, L = analysis._AN_ROW_STYLE, analysis._AN_LABEL_STYLE
    return [
        html.Div([
            html.Span("Cells:", style=L),
            dcc.Dropdown(id="de-pb-population", options=[{"label": "All cells", "value": "all"}],
                         value="all", clearable=False, placeholder="Cell population…",
                         style={"flex": "1", "minWidth": "200px", "fontSize": "11px"}),
            html.Span("Min. cells / sample:", style=L),
            dcc.Input(id="de-pb-min-cells", type="number", min=1, step=1, value=_PB_MIN_CELLS,
                      debounce=True, style=ui._PLOT_CTRL_INPUT),
            html.Span("Min. total count / gene:", style=L),
            dcc.Input(id="de-pb-min-count", type="number", min=0, step=1, value=_PB_MIN_COUNT,
                      debounce=True, style=ui._PLOT_CTRL_INPUT),
            dcc.Checklist(id="de-pb-shrink",
                          options=[{"label": html.Span(" Shrink log2FC", style={"fontSize": "11px",
                                                                                 "color": core.TEXT}),
                                    "value": "on"}],
                          value=["on"], inputStyle={"marginRight": "3px"},
                          style={"display": "inline-flex"}),
        ], style=S),
        html.Div([
            dbc.Button("Samples & conditions…", id="de-design-open-btn", color="secondary", size="sm",
                       n_clicks=0, style={"fontSize": "11px", "padding": "2px 8px"}),
            html.Span(id="de-design-summary", style=L),
        ], style=S),
    ]


def _design_modal():
    """The one editor of the sample → condition assignment (opened from the Differential
    Expression tab and from the Cell Type Abundance tab)."""
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Samples and conditions", style={"fontSize": "15px"}),
                        close_button=False),
        dbc.ModalBody([
            html.Div("Assign each loaded sample to condition A, to condition B, or leave it out. "
                     "The optional pairing / batch label (a patient, a slide, a run) adds a "
                     "batch term to the model; give it to every included sample or to none. "
                     "The assignment is shared by pseudobulk differential expression and the "
                     "composition test.",
                     style={"fontSize": "11px", "color": core.MUTED, "marginBottom": "8px"}),
            html.Div([
                html.Span("Name of A:", style=analysis._AN_LABEL_STYLE),
                dcc.Input(id="design-name-a", type="text", value="A", debounce=True,
                          style={**analysis._AN_INPUT_STYLE, "width": "140px"}),
                html.Span("Name of B:", style={**analysis._AN_LABEL_STYLE, "marginLeft": "10px"}),
                dcc.Input(id="design-name-b", type="text", value="B", debounce=True,
                          style={**analysis._AN_INPUT_STYLE, "width": "140px"}),
            ], style={"display": "flex", "alignItems": "center", "gap": "6px", "marginBottom": "8px"}),
            html.Div(id="design-rows"),
            html.Div(id="design-status", style={**analysis._AN_STATUS_STYLE, "padding": "6px 0 0 0"}),
        ]),
        dbc.ModalFooter([
            dbc.Button("Apply", id="design-apply-btn", color="primary", size="sm", n_clicks=0),
            dbc.Button("Close", id="design-close-btn", color="secondary", size="sm", n_clicks=0),
        ]),
    ], id="design-modal", is_open=False, size="lg", centered=True)


def _design_rows_children() -> list:
    """One row per loaded sample: label, name, cells, condition picker, batch input."""
    rows = _design_samples()
    L = analysis._AN_LABEL_STYLE
    head = html.Div([html.Span("Sample", style={**L, "width": "44px"}),
                     html.Span("Name", style={**L, "flex": "1"}),
                     html.Span("Cells", style={**L, "width": "70px", "textAlign": "right"}),
                     html.Span("Condition", style={**L, "width": "130px"}),
                     html.Span("Pairing / batch", style={**L, "width": "130px"})],
                    style={"display": "flex", "gap": "8px", "alignItems": "center",
                           "borderBottom": f"1px solid {core.BORDER}", "paddingBottom": "3px"})
    out = [head]
    for r in rows:
        out.append(html.Div([
            html.Span(r["label"], style={"fontSize": "11px", "color": core.TEXT, "width": "44px"}),
            html.Span(r["name"], title=r["key"],
                      style={"fontSize": "11px", "color": core.TEXT, "flex": "1", "overflow": "hidden",
                             "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
            html.Span(f"{r['n_cells']:,}", style={"fontSize": "11px", "color": core.MUTED,
                                                  "width": "70px", "textAlign": "right"}),
            dcc.Dropdown(id={"type": "design-cond", "index": r["index"]}, clearable=False,
                         options=[{"label": "A", "value": "A"}, {"label": "B", "value": "B"},
                                  {"label": "excluded", "value": ""}],
                         value=r["cond"], style={"width": "130px", "fontSize": "11px"}),
            dcc.Input(id={"type": "design-batch", "index": r["index"]}, type="text",
                      value=r["batch"], placeholder="(none)",
                      style={**analysis._AN_INPUT_STYLE, "width": "130px"}),
        ], style={"display": "flex", "gap": "8px", "alignItems": "center", "padding": "3px 0"}))
    return out


# ─── Composition between conditions ───────────────────────────────────────────
_COMP_NOTE = ("Proportions are compositional: they sum to 1 within a sample, so a real increase "
              "of one category lowers the proportions of all others, and the tests of the "
              "categories are not independent. Method (propeller-style, Phipson et al. 2022): "
              "per-sample proportions are logit-transformed (count + 0.5 pseudocount) or arcsine-"
              "square-root-transformed, a linear model (~ condition, or ~ batch + condition) is "
              "fitted per category and tested with a moderated t-test (limma-style empirical "
              "Bayes shrinkage of the variances, non-robust); Welch's t-test on the same values "
              "(ignores batch labels) is given beside it; Benjamini–Hochberg FDR over the "
              f"categories. With fewer than {_COMP_MIN_SAMPLES} samples in a condition no test is "
              "run: only the proportions are shown. Cluster and niche ids are comparable across "
              "samples only when the samples were clustered together.")


def _comp_category_options(seg_source, color_by) -> list:
    """What the proportions can be computed of."""
    tables = _view_tables(seg_source)
    alt = analysis._an_alt_res(seg_source)
    opts = []
    for m in core._ANNOT_METHODS:
        if any(l is not None for l in analysis._an_celltype_labels(tables, m, alt)):
            opts.append({"label": f"Cell type ({core._ANNOT_METHODS[m]})", "value": f"celltype:{m}"})
    opts.append({"label": "Cluster (active cluster set)", "value": "cluster"})
    if any("niche" in t["df"].columns for t in tables):
        opts.append({"label": "Spatial niche", "value": "niche"})
    return opts


def _comp_labels(kind, tables, seg_source, method) -> tuple:
    """(per-sample label arrays, name of the categories) for a category picker value."""
    alt = analysis._an_alt_res(seg_source)
    kind = str(kind or "")
    if kind.startswith("celltype:"):
        m = kind.split(":", 1)[1]
        return analysis._an_celltype_labels(tables, m, alt), f"cell type ({core._ANNOT_METHODS.get(m, m)})"
    if kind == "cluster":
        return _raw_cluster_labels(tables, method, alt), "cluster"
    if kind == "niche":
        return _niche_labels(tables), "niche"
    raise ValueError("Pick what the proportions are computed of.")


def _trigamma_inverse(x: float) -> float:
    """Inverse of the trigamma function (Newton iterations, as limma::trigammaInverse)."""
    from scipy.special import polygamma
    if not np.isfinite(x) or x <= 0:
        return np.nan
    if x > 1e7:
        return 1.0 / np.sqrt(x)
    if x < 1e-6:
        return 1.0 / x
    y = 0.5 + 1.0 / x
    for _ in range(50):
        tri = float(polygamma(1, y))
        dif = tri * (1.0 - tri / x) / float(polygamma(2, y))
        y += dif
        if -dif / y < 1e-8:
            break
    return y


def _squeeze_var(s2, df: float) -> tuple:
    """Empirical Bayes shrinkage of sample variances towards a common prior (limma's
    squeezeVar / fitFDist without trend or robust estimation): (posterior variances,
    prior variance s0², prior degrees of freedom; inf when the variances are no more
    variable than chance). With fewer than 2 usable variances nothing is shrunk
    (prior df 0)."""
    from scipy.special import digamma, polygamma
    s2 = np.asarray(s2, dtype=np.float64)
    ok = np.isfinite(s2) & (s2 > -1e-15)
    if df <= 0 or int(ok.sum()) < 2:
        return s2.copy(), np.nan, 0.0
    x = np.maximum(s2[ok], 0.0)
    m = np.median(x)
    if m == 0:
        m = 1.0
    x = np.maximum(x, 1e-5 * m)
    e = np.log(x) - digamma(df / 2.0) + np.log(df / 2.0)
    emean = e.mean()
    evar = ((e - emean) ** 2).sum() / (len(e) - 1) - float(polygamma(1, df / 2.0))
    if evar > 0:
        df0 = 2.0 * _trigamma_inverse(evar)
        s20 = float(np.exp(emean + digamma(df0 / 2.0) - np.log(df0 / 2.0)))
        post = (df * s2 + df0 * s20) / (df + df0)
    else:
        df0, s20 = np.inf, float(np.exp(emean))
        post = np.full(s2.shape, s20)
    return post, s20, df0


def _comp_transform(counts, transform: str = "logit") -> np.ndarray:
    """Transformed proportions (categories × samples) as speckle::getTransformedProps:
    logit of (count + 0.5) / Σ(count + 0.5), or asin(sqrt(count / Σ count))."""
    counts = np.asarray(counts, dtype=np.float64)
    if transform == "asin":
        return np.arcsin(np.sqrt(counts / counts.sum(axis=0, keepdims=True)))
    pp = (counts + 0.5) / (counts + 0.5).sum(axis=0, keepdims=True)
    return np.log(pp / (1.0 - pp))


def _comp_test(counts, conds, batches=None, transform: str = "logit") -> pd.DataFrame:
    """Test each category's proportion between conditions A and B.

    counts: categories × samples cell counts; conds: "A" / "B" per sample; batches:
    labels per sample or None. Returns per category: mean_prop_a, mean_prop_b,
    prop_ratio (A / B), coef (difference A − B of the transformed proportions, batch-
    adjusted), t_mod / df_mod / p_mod / fdr_mod (moderated t-test) and t_welch /
    df_welch / p_welch / fdr_welch (Welch's t-test on the transformed proportions)."""
    from scipy import stats as _st
    counts = np.asarray(counts, dtype=np.float64)
    a = np.array([c == "A" for c in conds])
    props = counts / counts.sum(axis=0, keepdims=True)
    y = _comp_transform(counts, transform)
    X = _pb_design_matrix(conds, batches)
    n, p = X.shape
    df = n - p
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = y @ X @ xtx_inv                              # categories × coefficients
    resid = y - beta @ X.T
    s2 = (resid ** 2).sum(axis=1) / df
    post, s20, df0 = _squeeze_var(s2, df)
    unscaled = np.sqrt(xtx_inv[-1, -1])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_mod = beta[:, -1] / (unscaled * np.sqrt(post))
    df_tot = min(df + df0, df * len(s2)) if np.isfinite(df0) else float(df * len(s2))
    p_mod = 2.0 * _st.t.sf(np.abs(t_mod), df_tot)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = _st.ttest_ind(y[:, a], y[:, ~a], axis=1, equal_var=False)
        va, vb = y[:, a].var(axis=1, ddof=1) / a.sum(), y[:, ~a].var(axis=1, ddof=1) / (~a).sum()
        df_w = (va + vb) ** 2 / (va ** 2 / (a.sum() - 1) + vb ** 2 / ((~a).sum() - 1))
        ratio = props[:, a].mean(axis=1) / props[:, ~a].mean(axis=1)
    return pd.DataFrame({
        "mean_prop_a": props[:, a].mean(axis=1), "mean_prop_b": props[:, ~a].mean(axis=1),
        "prop_ratio": ratio, "coef": beta[:, -1],
        "t_mod": t_mod, "df_mod": df_tot, "p_mod": p_mod, "fdr_mod": _bh(p_mod),
        "t_welch": np.asarray(w.statistic), "df_welch": df_w, "p_welch": np.asarray(w.pvalue),
        "fdr_welch": _bh(np.asarray(w.pvalue)),
        "prior_df": df0, "prior_var": s20,
    })


def _comp_compute(kind, seg_source="xenium", method=None, transform="logit") -> dict:
    """Per-sample proportions of the categories of *kind* and, when the design allows
    it, the test of each category between the two conditions. Raises ValueError with a
    message when nothing can be shown."""
    tables = _view_tables(seg_source)
    labels, cat_name = _comp_labels(kind, tables, seg_source, method)
    conds, batches = _design_of(tables)
    names = _design_names()
    rows, cols, used = [], {}, []
    for i, (t, lab) in enumerate(zip(tables, labels)):
        row = {"label": t["label"], "name": t["name"], "cond": conds[i], "batch": batches[i],
               "n_cells": 0, "n_unlabelled": 0, "used": False, "note": ""}
        if lab is None:
            row["note"] = f"no {cat_name} labels"
        else:
            s = pd.Series(lab, dtype=object)
            row["n_unlabelled"] = int(s.isna().sum())
            vc = s.dropna().astype(str).value_counts()
            row["n_cells"] = int(vc.sum())
            if not row["n_cells"]:
                row["note"] = f"no {cat_name} labels"
            else:
                row["used"] = True
                cols[t["label"]] = vc
                used.append(i)
        rows.append(row)
    if not cols:
        raise ValueError(f"No sample of the active view has {cat_name} labels.")
    counts = pd.DataFrame(cols).fillna(0).astype(np.int64)
    counts = counts.loc[sorted(counts.index, key=lambda v: (0, int(v), "") if str(v).isdigit() else (1, 0, str(v)))]
    props = counts / counts.sum(axis=0)
    res = {"kind": kind, "category": cat_name, "counts": counts, "props": props, "samples": rows,
           "names": names, "transform": transform, "tests": None, "design": None,
           "seg_source": seg_source or "xenium", "warnings": []}
    for r in rows:
        if not r["used"] and r["cond"]:
            res["warnings"].append(f"{r['label']} has {r['note']} and is left out")
    t_idx = [i for i in used if conds[i] in ("A", "B")]
    t_cond = [conds[i] for i in t_idx]
    n_a, n_b = t_cond.count("A"), t_cond.count("B")
    res["n_a"], res["n_b"] = n_a, n_b
    if len(tables) < 2:
        res["test_note"] = ("One sample: its proportions are shown; a test between conditions "
                            "needs several samples.")
        return res
    if n_a < _COMP_MIN_SAMPLES or n_b < _COMP_MIN_SAMPLES:
        res["test_note"] = (f"No test was run: condition A has {n_a} and condition B has {n_b} "
                            f"samples with {cat_name} labels; at least {_COMP_MIN_SAMPLES} per "
                            f"condition are needed. The proportions are shown.")
        return res
    try:
        t_batch, _warn = _design_check(t_cond, [batches[i] for i in t_idx], _COMP_MIN_SAMPLES)
    except ValueError as exc:
        res["test_note"] = f"No test was run: {exc}"
        return res
    sub = counts[[tables[i]["label"] for i in t_idx]]
    sub = sub.loc[sub.sum(axis=1) > 0]
    if len(sub) < 2:
        res["test_note"] = "No test was run: fewer than 2 categories have cells in the tested samples."
        return res
    tests = _comp_test(sub.to_numpy(), t_cond, t_batch, transform)
    tests.insert(0, "category", list(sub.index))
    tests.insert(1, "n_cells", sub.sum(axis=1).to_numpy())
    res["tests"] = tests
    res["design"] = "~ batch + condition" if t_batch else "~ condition"
    df0 = float(tests["prior_df"].iloc[0])
    res["test_note"] = (
        f"{len(tests)} categories tested between A {names['A']} ({n_a} samples) and B "
        f"{names['B']} ({n_b} samples): {'logit' if transform != 'asin' else 'arcsine-square-root'}"
        f"-transformed proportions, {res['design']}, moderated t-test (prior df "
        f"{'∞' if not np.isfinite(df0) else f'{df0:.1f}'}"
        f"{'; fewer than 2 usable variances: ordinary t-test' if df0 == 0 else ''}), BH FDR. "
        f"{int((tests['fdr_mod'] < 0.05).sum())} with FDR < 0.05."
        + (" Welch's t-test ignores the batch labels." if t_batch else ""))
    return res


def make_composition_fig(width=None, height=None):
    """Per-sample proportions of every category: one point per sample, grouped and
    coloured by condition, with the condition means."""
    with _comp_lock:
        res = _comp_state.get("result")
    if not res:
        return figures._make_placeholder_fig("No composition computed yet.")
    props = res["props"]
    cats = [str(c) for c in props.index]
    xpos = np.arange(len(cats), dtype=float)
    names = res["names"]
    groups = (("A", f"A: {names['A']}", "#f85149", -0.2), ("B", f"B: {names['B']}", "#58a6ff", 0.2),
              ("", "not assigned", "#8b949e", 0.0))
    fig = go.Figure()
    rng = np.random.default_rng(0)
    for cond, name, color, shift in groups:
        smp = [r for r in res["samples"] if r["used"] and (r["cond"] or "") == cond]
        if not smp:
            continue
        xs, ys, txt = [], [], []
        for r in smp:
            jit = rng.uniform(-0.06, 0.06, size=len(cats))
            xs.extend(xpos + shift + jit)
            ys.extend(100.0 * props[r["label"]].to_numpy())
            txt.extend([f"{r['label']} ({r['name']})"] * len(cats))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name=name, text=txt,
                                 marker=dict(size=7, color=color, opacity=0.85,
                                             line=dict(width=0.5, color="#0d1117")),
                                 hovertemplate="%{text}<br>%{y:.2f} %<extra></extra>"))
        if cond and len(smp) > 1:
            mean = 100.0 * props[[r["label"] for r in smp]].mean(axis=1).to_numpy()
            mx, my = [], []
            for xv, mv in zip(xpos + shift, mean):
                mx += [xv - 0.14, xv + 0.14, None]
                my += [mv, mv, None]
            fig.add_trace(go.Scatter(x=mx, y=my, mode="lines", line=dict(color=color, width=2),
                                     hoverinfo="skip", showlegend=False))
    tests = res.get("tests")
    if tests is not None:
        top = (100.0 * props.max(axis=1)).reindex(tests["category"]).to_numpy()
        pos = {c: i for i, c in enumerate(cats)}
        for c, q, yv in zip(tests["category"], tests["fdr_mod"], top):
            if np.isfinite(q) and q < 0.05:
                fig.add_annotation(x=pos[str(c)], y=yv, text="*", showarrow=False, yshift=10,
                                   font=dict(size=16, color=core.TEXT))
    fig.update_layout(
        paper_bgcolor=core.CARD_BG, plot_bgcolor=core.CARD_BG, font=dict(color=core.TEXT, size=11),
        margin=dict(l=50, r=10, t=52, b=80),
        title=dict(text=f"Proportion of each {res['category']} per sample"
                        + (" (* FDR < 0.05)" if tests is not None else ""),
                   font=dict(size=12), x=0.01, y=0.98, yanchor="top"),
        xaxis=dict(tickmode="array", tickvals=list(xpos), ticktext=cats, tickangle=-40,
                   gridcolor=core.BORDER, zeroline=False),
        yaxis=dict(title="% of the sample's labelled cells", gridcolor=core.BORDER,
                   zerolinecolor=core.BORDER, rangemode="tozero"),
        legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0.0, xanchor="left", font=dict(size=10)),
    )
    if width:
        fig.update_layout(width=width)
    if height:
        fig.update_layout(height=height)
    return fig


def _comp_table(res) -> tuple:
    """(columns, records) of the composition table: the test per category when one was
    run, and the proportion of every sample."""
    if not res:
        return [], []
    _col = lambda cid, name: {"id": cid, "name": name, "type": "text" if cid == "category" else "numeric"}
    props, tests = res["props"], res.get("tests")
    out = pd.DataFrame({"category": [str(c) for c in props.index]})
    cols = [_col("category", res["category"].capitalize())]
    if tests is not None:
        t = tests.set_index(tests["category"].astype(str)).reindex(out["category"])
        _g3 = lambda v: None if not np.isfinite(v) else float(f"{v:.3g}")
        for cid, name, f in (("mean_prop_a", "% A", lambda v: None if not np.isfinite(v) else round(100 * v, 2)),
                             ("mean_prop_b", "% B", lambda v: None if not np.isfinite(v) else round(100 * v, 2)),
                             ("prop_ratio", "Ratio A/B", _g3), ("t_mod", "t (mod.)", _g3),
                             ("p_mod", "p", _g3), ("fdr_mod", "FDR", _g3),
                             ("p_welch", "p (Welch)", _g3), ("fdr_welch", "FDR (Welch)", _g3)):
            out[cid] = [f(float(v)) for v in t[cid].to_numpy(dtype=float)]
            cols.append(_col(cid, name))
    for r in res["samples"]:
        if r["used"]:
            cid = f"prop_{r['label']}"
            out[cid] = (100.0 * props[r["label"]].to_numpy()).round(2)
            cols.append(_col(cid, f"% {r['label']}" + (f" ({r['cond']})" if r["cond"] else "")))
    return cols, out.astype(object).where(out.notna(), None).to_dict("records")


def _comp_csv(res) -> str:
    """The composition result as CSV text: design and samples as comment lines, then one
    row per category with the test and every sample's count and proportion."""
    names = res["names"]
    lines = [f"# Composition: proportions of each {res['category']} per sample; segmentation: "
             f"{res['seg_source']}",
             f"# A: {names['A']}; B: {names['B']}; {res.get('test_note', '')}"]
    for r in res["samples"]:
        lines.append(f"# sample {r['label']} ({r['name']}): condition {r['cond'] or '-'}"
                     + (f"; batch {r['batch']}" if r["batch"] else "")
                     + f"; {r['n_cells']} labelled cells; {r['n_unlabelled']} unlabelled"
                     + (f"; {r['note']}" if r["note"] else ""))
    lines.append("# proportions are compositional (they sum to 1 per sample): the tests of the "
                 "categories are not independent")
    out = pd.DataFrame({"category": [str(c) for c in res["props"].index]})
    tests = res.get("tests")
    if tests is not None:
        t = tests.drop(columns=["n_cells"]).set_index(tests["category"].astype(str)).drop(columns=["category"])
        out = out.join(t.reindex(out["category"]).reset_index(drop=True))
    for r in res["samples"]:
        if r["used"]:
            out[f"n_{r['label']}"] = res["counts"][r["label"]].to_numpy()
            out[f"prop_{r['label']}"] = res["props"][r["label"]].to_numpy()
    return "\n".join(lines) + "\n" + out.to_csv(index=False)


def _dark_table_style(page_size: int = 12) -> dict:
    """DataTable keyword arguments giving the style of the DE result table."""
    return dict(
        sort_action="native", filter_action="native", page_size=page_size,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": core.CARD_BG, "color": core.TEXT, "fontWeight": "600",
                      "border": f"1px solid {core.BORDER}", "fontSize": "11px"},
        style_cell={"backgroundColor": "#0d1117", "color": core.TEXT, "fontSize": "11px",
                    "border": f"1px solid {core.BORDER}", "padding": "2px 5px",
                    "fontFamily": "inherit", "minWidth": "38px", "maxWidth": "160px",
                    "overflow": "hidden", "textOverflow": "ellipsis"},
        style_filter={"backgroundColor": core.CARD_BG, "color": core.TEXT})


def _composition_section():
    """The "Composition between conditions" block of the Cell Type Abundance tab."""
    S, L = analysis._AN_ROW_STYLE, analysis._AN_LABEL_STYLE
    return html.Div([
        html.Div("Composition between conditions",
                 style={"fontSize": "12px", "color": core.TEXT, "fontWeight": "600", "padding": "6px 8px 0 8px",
                        "borderTop": f"1px solid {core.BORDER}"}),
        html.Div([
            html.Span("Proportions of:", style=L),
            dcc.Dropdown(id="comp-category", options=[{"label": "Cluster (active cluster set)", "value": "cluster"}],
                         value=None, placeholder="Cell type / cluster / niche…",
                         style={"flex": "1", "minWidth": "180px", "fontSize": "11px"}),
            html.Span("Transform:", style=L),
            dcc.Dropdown(id="comp-transform", clearable=False, value="logit",
                         options=[{"label": "logit", "value": "logit"},
                                  {"label": "arcsine √", "value": "asin"}],
                         style={"width": "110px", "fontSize": "11px"}),
            dbc.Button("Compute", id="comp-run-btn", color="primary", size="sm", n_clicks=0,
                       style={"fontSize": "11px", "padding": "2px 12px"}),
        ], style=S),
        html.Div([
            dbc.Button("Samples & conditions…", id="comp-design-open-btn", color="secondary", size="sm",
                       n_clicks=0, style={"fontSize": "11px", "padding": "2px 8px"}),
            html.Span(id="comp-design-summary", style=L),
        ], style=S),
        html.Div(id="comp-status", style={**analysis._AN_STATUS_STYLE, "whiteSpace": "pre-wrap"}),
        html.Div(_COMP_NOTE, id="comp-note", style=analysis._AN_NOTE_STYLE),
        html.Div([
            html.Div(dcc.Graph(id="comp-plot", responsive=True,
                               figure=figures._make_placeholder_fig("No composition computed yet."),
                               config={"displayModeBar": "hover",
                                       "toImageButtonOptions": {"format": "svg", "filename": "xenium_composition"}},
                               style={"width": "100%", "height": "100%"}),
                     id="comp-plot-container", style={"flex": "1", "minWidth": "0", "minHeight": "280px"}),
            html.Div([
                dash_table.DataTable(id="comp-table", data=[], columns=[], **_dark_table_style(14)),
                html.Div([
                    dbc.Button("Download CSV", id="comp-download-btn", color="secondary", size="sm",
                               n_clicks=0, style={"fontSize": "11px", "padding": "2px 8px"}),
                    dcc.Download(id="comp-download"),
                ], style={"padding": "4px 0"}),
            ], style={"flex": "1.25", "minWidth": "0", "overflow": "auto", "padding": "0 6px"}),
        ], style={"display": "flex", "minHeight": "300px", "gap": "6px"}),
        ui._plot_dim_controls("comp"),
    ], id="comp-section", style={"display": "flex", "flexDirection": "column", "flexShrink": "0"})


# ─── Spatially variable genes: Moran's I ──────────────────────────────────────
_MORAN_NOTE = ("Moran's I per gene on log1p(count / library size × 10,000), with a k-nearest-"
               "neighbour graph of the cell centroids (row-standardised weights: each cell's k "
               "neighbours weigh 1/k), computed per sample for the active segmentation, counts "
               "mode and subset. z-score and two-sided p-value from the analytic mean and "
               "variance of I under the normality assumption (Cliff & Ord), Benjamini–Hochberg "
               "FDR over the genes of a sample. With tens of thousands of cells almost every "
               "expressed gene is significant: rank by I, not by the p-value. I measures spatial "
               "autocorrelation at the scale of the k neighbours; a gene restricted to a cell "
               "type that clusters in space scores high whatever the reason. Imputed genes are "
               "not tested.")


def _xy_of(df) -> np.ndarray:
    """Cell centroids (µm) of a cell table of any segmentation source."""
    xc = "x_centroid" if "x_centroid" in df.columns else "centroid_x"
    yc = "y_centroid" if "y_centroid" in df.columns else "centroid_y"
    return np.column_stack([df[xc].to_numpy(dtype=np.float64), df[yc].to_numpy(dtype=np.float64)])


def _knn_indices(xy, k: int) -> np.ndarray:
    """Indices (n × k) of each point's k nearest other points (a point is never its own
    neighbour, also when several points share its coordinates)."""
    from sklearn.neighbors import NearestNeighbors
    n = len(xy)
    k = int(min(k, n - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(xy)
    idx = nn.kneighbors(xy, return_distance=False)
    self_col = idx == np.arange(n)[:, None]
    # move the point itself to the end of its row (the last column when it is not in
    # the row at all: k + 1 points at the same place), keep the first k
    last = np.zeros_like(self_col)
    last[:, -1] = ~self_col.any(axis=1)
    order = np.argsort(self_col | last, axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1)[:, :k]


def _knn_weights(xy, k: int = _MORAN_K) -> "sp.csr_matrix":
    """Row-standardised spatial weights of the k-nearest-neighbour graph (n × n CSR: w_ij
    = 1/k when j is one of i's k nearest neighbours)."""
    idx = _knn_indices(xy, k)
    n, kk = idx.shape
    return sp.csr_matrix((np.full(n * kk, 1.0 / kk), idx.ravel(), np.arange(0, n * kk + 1, kk)),
                         shape=(n, n))


def _moran_moments(W) -> tuple:
    """(E[I], Var[I]) of Moran's I under the normality assumption for weights *W*."""
    n = W.shape[0]
    s0 = float(W.sum())
    m = (W + W.T).tocsr()
    s1 = 0.5 * float((m.data ** 2).sum())
    s2 = float(((np.asarray(W.sum(axis=1)).ravel() + np.asarray(W.sum(axis=0)).ravel()) ** 2).sum())
    ei = -1.0 / (n - 1)
    var = (n * n * s1 - n * s2 + 3.0 * s0 * s0) / (s0 * s0 * (n * n - 1.0)) - ei * ei
    return ei, var


def _moran_sample(s: dict, k: int = _MORAN_K, progress=None, chunk_bytes: float = _MORAN_CHUNK_BYTES) -> dict:
    """Moran's I of every measured gene of one analysis sample.

    Cells: the cells of the view that have a matrix row. The expression matrix stays
    sparse; genes are handled in dense blocks of at most *chunk_bytes* (cells × genes
    float64), so memory does not grow with the panel. Returns {table (gene, moran_i, z,
    pval, padj, pct_cells, mean), n_cells, k, expected, variance}."""
    from scipy.stats import norm as _norm
    rows = np.asarray(s["rows"])
    ok = rows >= 0
    sel = rows[ok]
    n = int(len(sel))
    genes = list(s["genes"])
    if n < 3:
        raise ValueError(f"{s['label']}: fewer than 3 cells with counts.")
    W = _knn_weights(_xy_of(s["df"])[ok], k)
    kk = int(W.indptr[1] - W.indptr[0])
    ei, var = _moran_moments(W)
    s0 = float(W.sum())
    sf, _in_view = analysis._an_size_factors(s)
    sf_sel = sf[sel].astype(np.float64)
    csc = s["csc"]
    cols = np.array([s["gni"][g] for g in genes], dtype=np.int64)
    identity = csc.shape[0] == n and np.array_equal(sel, np.arange(n))
    step = int(max(1, min(len(cols), chunk_bytes // (8 * max(n, csc.shape[0])))))
    I = np.full(len(cols), np.nan)
    pct = np.zeros(len(cols))
    mean = np.zeros(len(cols))
    for c0 in range(0, len(cols), step):
        jobs._job_check()
        if progress is not None:
            progress(c0 / len(cols))
        cc = cols[c0:c0 + step]
        block = csc[:, cc].toarray().astype(np.float64, copy=False)
        if not identity:
            block = block[sel]
        pct[c0:c0 + step] = 100.0 * (block > 0).sum(axis=0) / n
        block *= sf_sel[:, None]
        np.log1p(block, out=block)
        mu = block.mean(axis=0)
        mean[c0:c0 + step] = mu
        block -= mu
        den = np.einsum("ij,ij->j", block, block)
        num = np.einsum("ij,ij->j", block, W @ block)
        with np.errstate(divide="ignore", invalid="ignore"):
            I[c0:c0 + step] = np.where(den > 0, (n / s0) * num / den, np.nan)
    if progress is not None:
        progress(1.0)
    z = (I - ei) / np.sqrt(var)
    log10p = np.minimum((_norm.logsf(np.abs(z)) + np.log(2.0)) / np.log(10.0), 0.0)
    okg = np.isfinite(I)
    log10padj = np.full(len(I), np.nan)
    log10padj[okg] = analysis._bh_log10(log10p[okg])
    table = pd.DataFrame({"gene": genes, "moran_i": I, "z": z,
                          "pval": np.where(okg, 10.0 ** log10p, np.nan),
                          "padj": 10.0 ** log10padj, "neg_log10_padj": -log10padj,
                          "pct_cells": pct, "mean": mean})
    return {"table": table, "n_cells": n, "k": kk, "expected": ei, "variance": var}


def _run_moran(seg_source, use_corrected, k=_MORAN_K) -> None:
    """Background job: Moran's I of every measured gene, per sample of the active view."""
    _gen = jobs._dataset_gen

    def _set(status, message, result=None):
        with _moran_lock:
            if jobs._stale_job(_gen, status, message):
                return
            _moran_state["status"] = status
            _moran_state["message"] = message
            if result is not None:
                _moran_state["result"] = result

    try:
        _t0 = _time.time()
        k = int(max(1, min(50, k or _MORAN_K)))
        _set("running", "Building the spatial graph…")
        samples = analysis._an_samples(seg_source, use_corrected)
        if not samples:
            _set("error", "The active view has no expression matrix.")
            return
        parts, infos, skipped = [], [], []
        for si, s in enumerate(samples):
            _set("running", f"Moran's I: {s['label']} ({len(s['df']):,} cells × {len(s['genes']):,} genes)…")
            try:
                r = _moran_sample(s, k, progress=lambda f, _si=si: jobs._job_progress(
                    fraction=(_si + f) / len(samples)))
            except ValueError as exc:
                skipped.append(str(exc))
                continue
            t = r["table"]
            t.insert(0, "sample", s["label"])
            parts.append(t)
            infos.append({"label": s["label"], "name": s["name"], "n_cells": r["n_cells"],
                          "k": r["k"], "expected": r["expected"], "variance": r["variance"],
                          "n_tested": int(t["moran_i"].notna().sum()),
                          "n_sig": int((t["padj"] < 0.05).sum())})
        jobs._check_dataset_gen(_gen)
        if not parts:
            _set("error", "No sample could be analysed. " + " ".join(skipped))
            return
        table = pd.concat(parts, ignore_index=True)
        corrected = bool(samples[0]["corrected"])
        res = {"table": table, "samples": infos, "k": k, "corrected": corrected,
               "seg_source": seg_source or "xenium", "seconds": _time.time() - _t0}
        msg = ("Moran's I, kNN graph (k = " + str(k) + "), "
               + "; ".join(f"{i['label']}: {i['n_cells']:,} cells, {i['n_tested']:,} genes, "
                           f"{i['n_sig']:,} with FDR < 0.05" for i in infos)
               + f" ({'SPLIT-corrected' if corrected else 'original'} counts, {res['seconds']:.1f} s).")
        if use_corrected and not corrected:
            msg += " Original counts were used: not every sample has SPLIT-corrected counts."
        if skipped:
            msg += " Skipped: " + " ".join(skipped)
        _set("done", msg, result=res)
        print(f"  {msg}", flush=True)
    except jobs._JobCancelled:
        raise
    except jobs._DatasetChanged:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set("error", str(exc)[:300])


def _moran_table_records(res) -> list:
    """Rows of the result table, highest I first (numbers rounded for display)."""
    if not res:
        return []
    t = res["table"].dropna(subset=["moran_i"]).sort_values("moran_i", ascending=False, kind="stable")
    _g3 = lambda v: None if not np.isfinite(v) else float(f"{v:.3g}")
    out = pd.DataFrame({"sample": t["sample"], "gene": t["gene"], "moran_i": t["moran_i"].round(4),
                        "z": t["z"].round(1), "pval": t["pval"].map(_g3), "padj": t["padj"].map(_g3),
                        "pct_cells": t["pct_cells"].round(1), "mean": t["mean"].round(3)})
    return out.to_dict("records")


def _moran_tab():
    S, L = analysis._AN_ROW_STYLE, analysis._AN_LABEL_STYLE
    _col = lambda cid, name: {"id": cid, "name": name,
                              "type": "text" if cid in ("gene", "sample") else "numeric"}
    return dbc.Tab(label="Spatial Genes", tab_id="tab-moran",
                   label_style={"fontSize": "12px", "padding": "4px 12px"},
                   children=[html.Div([
        html.Div([
            html.Span("Spatially variable genes (Moran's I)", style={**L, "color": core.TEXT, "flex": "1"}),
            html.Span("Neighbours (k):", style=L),
            dcc.Input(id="moran-k", type="number", min=1, max=50, step=1, value=_MORAN_K,
                      debounce=True, style=ui._PLOT_CTRL_INPUT),
            dbc.Button("Run", id="moran-run-btn", color="primary", size="sm", n_clicks=0,
                       style={"fontSize": "11px", "padding": "2px 12px"}),
        ], style=S),
        html.Div(id="moran-status", style={**analysis._AN_STATUS_STYLE, "whiteSpace": "pre-wrap"}),
        html.Div(_MORAN_NOTE, id="moran-note", style=analysis._AN_NOTE_STYLE),
        html.Div([
            dash_table.DataTable(id="moran-table", data=[], columns=[_col("sample", "Sample"), _col("gene", "Gene"), _col("moran_i", "Moran's I"),
                         _col("z", "z"), _col("pval", "p"), _col("padj", "FDR"),
                         _col("pct_cells", "% cells"), _col("mean", "Mean")],
                                 **_dark_table_style(15)),
            html.Div([
                dbc.Button("Download CSV", id="moran-download-btn", color="secondary", size="sm",
                           n_clicks=0, style={"fontSize": "11px", "padding": "2px 8px"}),
                html.Span("Click a table cell to select the gene.", style={**L, "marginLeft": "8px"}),
                dcc.Download(id="moran-download"),
            ], style={"padding": "4px 0"}),
        ], style={"flex": "1", "minHeight": "0", "overflow": "auto", "padding": "0 8px"}),
    ], style={"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0",
              "overflowY": "auto"})])


# ─── Ligand-receptor scores between cell types ────────────────────────────────
_LR_NOTE = ("Score of a ligand–receptor pair for sender type S and receiver type R = mean of the "
            "mean log-normalised expression of the ligand in S and of the receptor in R "
            "(CellPhoneDB-style), computed only for type pairs that are spatial neighbours (at "
            f"least {_LR_MIN_EDGES} adjacent cell pairs in the k-nearest-neighbour graph of the "
            "centroids; a type needs "
            f"{_LR_MIN_CELLS} cells) and only when the ligand and the receptor are detected in at "
            f"least {int(_LR_MIN_FRAC * 100)} % of the cells of their type. p-value: the cell-type "
            "labels are permuted (within each sample, fixed seed) and the score recomputed; p = "
            "(1 + permutations with a score ≥ observed) / (1 + permutations), so the smallest "
            "possible p is 1 / (1 + permutations); Benjamini–Hochberg FDR over all tested "
            "combinations. No database is bundled or downloaded: paste or upload a list of "
            "ligand,receptor gene pairs (or take the pairs of loaded CellNEST results). Only "
            "pairs with both genes measured on the panel are scored; complexes are not modelled. "
            "A significant score means both genes are expressed specifically in neighbouring "
            "cell types - expression-based inference, not evidence of signalling; segmentation "
            "spill-over between adjacent cells inflates it.")


def _lr_parse_pairs(text) -> tuple:
    """(pairs, n_skipped): ligand-receptor pairs of a pasted / uploaded list. One pair
    per line, the two genes separated by a comma, tab, semicolon or white space; a
    header line naming a ligand and a receptor column picks those columns of a wider
    table; repeated pairs and lines without two names are skipped."""
    pairs, seen, skipped = [], set(), 0
    li, ri = 0, 1
    first = True
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sep = "," if "," in line else ("\t" if "\t" in line else (";" if ";" in line else None))
        tok = [t.strip().strip("\"'") for t in (line.split(sep) if sep else line.split())]
        if first:
            first = False
            low = [t.lower() for t in tok]
            lcol = next((i for i, t in enumerate(low) if t.startswith("ligand")), None)
            rcol = next((i for i, t in enumerate(low) if t.startswith("receptor")), None)
            if lcol is not None and rcol is not None:
                li, ri = lcol, rcol
                continue
        if len(tok) <= max(li, ri) or not tok[li] or not tok[ri]:
            skipped += 1
            continue
        pr = (tok[li], tok[ri])
        if pr in seen:
            skipped += 1
            continue
        seen.add(pr)
        pairs.append(pr)
    return pairs, skipped


def _lr_pairs_from_cellnest() -> list:
    """The distinct ligand-receptor pairs of the loaded CellNEST results ([] when none
    are loaded)."""
    with core._cellnest_lock:
        data = core._cellnest_state.get("data")
    ccc = (data or {}).get("ccc_df")
    if ccc is None or "ligand" not in ccc.columns or "receptor" not in ccc.columns:
        return []
    d = ccc[["ligand", "receptor"]].dropna().astype(str).drop_duplicates()
    return list(zip(d["ligand"], d["receptor"]))


def _lr_label_options(seg_source, color_by) -> list:
    """Cell groupings the scores can be computed between."""
    tables = _view_tables(seg_source)
    alt = analysis._an_alt_res(seg_source)
    opts = []
    for m in core._ANNOT_METHODS:
        if any(l is not None for l in analysis._an_celltype_labels(tables, m, alt)):
            opts.append({"label": f"Cell type ({core._ANNOT_METHODS[m]})", "value": f"celltype:{m}"})
    opts.append({"label": "Cluster (active cluster set)", "value": "cluster"})
    return opts


def _lr_compute(samples, labels, pairs, k: int = _MORAN_K, n_perm: int = 200, seed: int = 0,
                min_cells: int = _LR_MIN_CELLS, min_edges: int = _LR_MIN_EDGES,
                min_frac: float = _LR_MIN_FRAC, progress=None) -> dict:
    """Ligand-receptor scores between neighbouring cell types with a label-permutation
    p-value.

    samples: analysis samples; labels: per sample the type of each cell (None entries /
    None arrays: unlabelled, left out); pairs: [(ligand, receptor)]. Expression is
    log1p(count / library size × 10,000). The cells of all samples are pooled; the
    neighbour graph (k nearest neighbours of the centroids) is built within each sample
    and the labels are permuted within each sample. Returns {table (one row per tested
    sender / receiver / pair: score, mean_ligand, mean_receptor, frac_ligand,
    frac_receptor, n_edges, pval, padj), info}."""
    genes_all = analysis._an_common_genes(samples)
    info = {"n_pairs_input": len(pairs), "n_perm": int(n_perm), "seed": int(seed), "k": int(k)}
    names = list(dict.fromkeys([g for pr in pairs for g in pr]))
    found, _missing = analysis._gs_match(names, genes_all)
    lower = {g.lower(): g for g in found}
    have = set(found)
    _hit = lambda g: g if g in have else lower.get(g.lower())
    kept = []
    for lg, rc in pairs:
        a, b = _hit(lg), _hit(rc)
        if a is not None and b is not None and (a, b) not in kept:
            kept.append((a, b))
    info["n_pairs_on_panel"] = len(kept)
    info["n_pairs_missing"] = len(pairs) - len(kept)
    if not kept:
        raise ValueError(f"None of the {len(pairs)} ligand–receptor pairs has both genes among "
                         f"the {len(genes_all):,} measured genes of this view.")
    G = list(dict.fromkeys([g for pr in kept for g in pr]))
    gpos = {g: i for i, g in enumerate(G)}
    L = np.array([gpos[a] for a, _ in kept], dtype=np.int64)
    R = np.array([gpos[b] for _, b in kept], dtype=np.int64)
    # pooled cells: expression of the genes of the pairs, labels, neighbour graph
    xs, labs, blocks, adj = [], [], [], []
    n0 = 0
    for s, lab in zip(samples, labels):
        if lab is None:
            continue
        rows = np.asarray(s["rows"])
        lab = np.asarray(lab, dtype=object)
        ok = (rows >= 0) & np.array([x is not None for x in lab], dtype=bool)
        n = int(ok.sum())
        if n < 3:
            continue
        sf, _in_view = analysis._an_size_factors(s)
        sub = s["csc"][:, np.array([s["gni"][g] for g in G], dtype=np.int64)].tocsr()[rows[ok]]
        sub = sub.astype(np.float64)
        sub.data = np.log1p(sub.data * np.repeat(sf[rows[ok]].astype(np.float64), np.diff(sub.indptr)))
        xs.append(sub)
        labs.append(lab[ok].astype(str))
        idx = _knn_indices(_xy_of(s["df"])[ok], k)
        kk = idx.shape[1]
        adj.append(sp.csr_matrix((np.ones(n * kk, dtype=np.float32), idx.ravel(),
                                  np.arange(0, n * kk + 1, kk)), shape=(n, n)))
        blocks.append((n0, n0 + n))
        n0 += n
        jobs._job_check()
    if not xs:
        raise ValueError("No sample of the active view has labelled cells with counts.")
    X = sp.vstack(xs).tocsc()
    X.eliminate_zeros()
    lab_all = np.concatenate(labs)
    n = X.shape[0]
    vc = pd.Series(lab_all).value_counts()
    types = sorted(vc.index[vc >= min_cells], key=str)
    T = len(types)
    info.update({"n_cells": n, "n_types": T, "types": types,
                 "types_dropped": sorted(vc.index[vc < min_cells], key=str)})
    if T < 1:
        raise ValueError(f"No cell type has at least {min_cells} cells.")
    code = pd.Series(lab_all).map({t: i for i, t in enumerate(types)}).fillna(T).to_numpy(dtype=np.int64)
    n_t = np.bincount(code, minlength=T + 1).astype(np.float64)
    # adjacent cell pairs between the types (an unordered pair counts once)
    A = sp.block_diag(adj, format="csr")
    U = ((A + A.T) > 0).astype(np.float32)
    H = sp.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), code)), shape=(n, T + 1))
    E = np.asarray((H.T @ U @ H).todense(), dtype=np.float64)[:T, :T]
    E[np.diag_indices(T)] /= 2.0
    neigh = E >= min_edges
    info["n_type_pairs"] = int(T * T)
    info["n_type_pairs_neighbouring"] = int(neigh.sum())
    n_g = len(G)
    col_of_nnz = np.repeat(np.arange(n_g, dtype=np.int64), np.diff(X.indptr))
    xi, xd = X.indices, X.data

    def _sums(c):
        return np.bincount(c[xi] * n_g + col_of_nnz, weights=xd, minlength=(T + 1) * n_g).reshape(T + 1, n_g)

    inv_n = (1.0 / n_t[:T])[:, None]                       # label counts do not change under permutation
    means = _sums(code)[:T] * inv_n
    frac = (np.bincount(code[xi] * n_g + col_of_nnz, minlength=(T + 1) * n_g)
            .reshape(T + 1, n_g)[:T] * inv_n)
    mL, mR = means[:, L], means[:, R]                      # T × P
    score = 0.5 * (mL[:, None, :] + mR[None, :, :])        # sender × receiver × pair
    tested = (neigh[:, :, None] & (frac[:, L] >= min_frac)[:, None, :]
              & (frac[:, R] >= min_frac)[None, :, :])
    info["n_tested"] = int(tested.sum())
    cnt = np.zeros(score.shape, dtype=np.int32)
    rng = np.random.default_rng(seed)
    thr = score - 1e-12
    perm = code.copy()
    for b in range(int(n_perm)):
        jobs._job_check()
        if progress is not None:
            progress(b / max(1, n_perm))
        for a0, a1 in blocks:
            perm[a0:a1] = rng.permutation(code[a0:a1])
        pm = _sums(perm)[:T] * inv_n
        cnt += (0.5 * (pm[:, L][:, None, :] + pm[:, R][None, :, :]) >= thr)
    if progress is not None:
        progress(1.0)
    pval = (cnt + 1.0) / (n_perm + 1.0)
    si, ri, pi = np.nonzero(tested)
    p = pval[si, ri, pi]
    table = pd.DataFrame({
        "sender": [types[i] for i in si], "receiver": [types[i] for i in ri],
        "ligand": [kept[i][0] for i in pi], "receptor": [kept[i][1] for i in pi],
        "score": score[si, ri, pi], "mean_ligand": mL[si, pi], "mean_receptor": mR[ri, pi],
        "frac_ligand": frac[:, L][si, pi], "frac_receptor": frac[:, R][ri, pi],
        "n_edges": E[si, ri].astype(np.int64), "pval": p, "padj": _bh(p),
    })
    return {"table": table, "info": info}


def _run_lr(pairs_text, label_kind, seg_source, method, use_corrected, k, n_perm, seed=0) -> None:
    """Background job: ligand-receptor scores for the active view."""
    _gen = jobs._dataset_gen

    def _set(status, message, result=None):
        with _lr_lock:
            if jobs._stale_job(_gen, status, message):
                return
            _lr_state["status"] = status
            _lr_state["message"] = message
            if result is not None:
                _lr_state["result"] = result

    try:
        _t0 = _time.time()
        _set("running", "Collecting cells, labels and the neighbour graph…")
        pairs, n_skip = _lr_parse_pairs(pairs_text)
        if not pairs:
            _set("error", "No ligand–receptor pairs: paste or upload lines of 'ligand,receptor'.")
            return
        samples = analysis._an_samples(seg_source, use_corrected)
        if not samples:
            _set("error", "The active view has no expression matrix.")
            return
        labels, cat_name = _comp_labels(label_kind, samples, seg_source, method)
        if all(l is None for l in labels):
            _set("error", f"The active view has no {cat_name} labels: run Cell Type Annotation "
                          f"first, or score between clusters.")
            return
        k = int(max(1, min(50, k or _MORAN_K)))
        n_perm = int(max(10, min(_LR_MAX_PERM, n_perm or 200)))

        def _prog(f):
            jobs._job_progress(f"Permuting labels ({int(f * n_perm)} / {n_perm})…", 0.1 + 0.9 * f)

        _set("running", f"Scoring {len(pairs):,} pairs with {n_perm} label permutations…")
        res = _lr_compute(samples, labels, pairs, k=k, n_perm=n_perm, seed=seed, progress=_prog)
        jobs._check_dataset_gen(_gen)
        info = res["info"]
        corrected = bool(samples[0]["corrected"])
        res.update({"category": cat_name, "label_kind": label_kind, "corrected": corrected,
                    "seg_source": seg_source or "xenium", "n_samples": len(samples),
                    "n_skipped_lines": n_skip, "seconds": _time.time() - _t0})
        t = res["table"]
        msg = (f"{info['n_pairs_on_panel']:,} of {info['n_pairs_input']:,} ligand–receptor pairs "
               f"have both genes on the panel ({info['n_pairs_missing']:,} not scored). "
               f"{info['n_cells']:,} labelled cells in {info['n_types']} groups ({cat_name}); "
               f"{info['n_type_pairs_neighbouring']} of {info['n_type_pairs']} directed type pairs "
               f"are neighbours (kNN graph, k = {k}). {len(t):,} combinations tested, "
               f"{int((t['padj'] < 0.05).sum()):,} with FDR < 0.05 ({n_perm} permutations, seed "
               f"{seed}; smallest possible p {1.0 / (n_perm + 1):.2g}; "
               f"{'SPLIT-corrected' if corrected else 'original'} counts; {res['seconds']:.1f} s).")
        if use_corrected and not corrected:
            msg += " Original counts were used: not every sample has SPLIT-corrected counts."
        if len(samples) > 1:
            msg += f" Cells of {len(samples)} samples were pooled (graph and permutations within each sample)."
        if info["types_dropped"]:
            msg += (f" Not scored (fewer than {_LR_MIN_CELLS} cells): "
                    f"{', '.join(map(str, info['types_dropped'][:12]))}"
                    f"{'…' if len(info['types_dropped']) > 12 else ''}.")
        if label_kind == "cluster" and len(samples) > 1:
            msg += " Cluster ids were taken as the same clusters in every sample."
        _set("done", msg, result=res)
        print(f"  LR: {msg}", flush=True)
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


def make_lr_dotplot(top_n=25, width=None, height=None):
    """Dot plot of the last ligand-receptor result: sender → receiver (columns) × pair
    (rows) for the *top_n* pairs with the smallest FDR; dot size = −log10 FDR, colour =
    score."""
    with _lr_lock:
        res = _lr_state.get("result")
    if not res:
        return figures._make_placeholder_fig("No ligand–receptor scores computed yet.")
    t = res["table"]
    if not len(t):
        return figures._make_placeholder_fig("No combination could be tested.")
    t = t.assign(pair=t["ligand"] + " → " + t["receptor"], types=t["sender"] + " → " + t["receiver"])
    top_n = int(max(1, top_n or 25))
    best = (t.sort_values(["padj", "pval", "score"], ascending=[True, True, False], kind="stable")
             .groupby("pair", sort=False).head(1).head(top_n))
    pairs = list(best["pair"])
    sub = t[t["pair"].isin(pairs)]
    sig = sub[sub["padj"] < 0.05]
    shown = sig if len(sig) else sub.nsmallest(min(len(sub), 60), "pval")
    cols = list(shown.groupby("types")["padj"].agg(["count", "min"])
                .sort_values(["count", "min"], ascending=[False, True]).head(40).index)
    shown = shown[shown["types"].isin(cols)]
    floor = 1.0 / (res["info"]["n_perm"] + 1.0)
    nlq = -np.log10(np.maximum(shown["padj"].to_numpy(), floor / 10))
    fig = go.Figure(go.Scatter(
        x=shown["types"], y=shown["pair"], mode="markers",
        marker=dict(size=5 + 6 * nlq / max(1e-9, nlq.max() if len(nlq) else 1), color=shown["score"],
                    colorscale="Viridis", showscale=True,
                    colorbar=dict(title="score", thickness=12, len=0.6),
                    line=dict(width=0.5, color="#0d1117")),
        customdata=np.column_stack([shown["pval"], shown["padj"], shown["n_edges"],
                                    shown["ligand"], shown["receptor"]]),
        hovertemplate=("<b>%{y}</b><br>%{x}<br>score %{marker.color:.3f}<br>p %{customdata[0]:.3g}"
                       "<br>FDR %{customdata[1]:.3g}<br>%{customdata[2]} adjacent cell pairs<extra></extra>")))
    fig.update_layout(
        paper_bgcolor=core.CARD_BG, plot_bgcolor=core.CARD_BG, font=dict(color=core.TEXT, size=10),
        margin=dict(l=130, r=10, t=30, b=40),
        title=dict(text=(f"Top {len(pairs)} pairs "
                         f"({'FDR < 0.05' if len(sig) else 'none with FDR < 0.05: smallest p'}); "
                         f"size −log10 FDR"), font=dict(size=12), x=0.01),
        xaxis=dict(title=dict(text="sender → receiver", standoff=6), tickangle=-60, gridcolor=core.BORDER,
                   automargin=True, tickfont=dict(size=9), categoryorder="array", categoryarray=cols),
        yaxis=dict(gridcolor=core.BORDER, categoryorder="array", categoryarray=pairs[::-1]),
    )
    if width:
        fig.update_layout(width=width)
    if height:
        fig.update_layout(height=height)
    return fig


def _lr_table_records(res) -> list:
    if not res:
        return []
    t = res["table"].sort_values(["padj", "pval", "score"], ascending=[True, True, False], kind="stable")
    _g3 = lambda v: None if not np.isfinite(v) else float(f"{v:.3g}")
    out = pd.DataFrame({"sender": t["sender"], "receiver": t["receiver"], "ligand": t["ligand"],
                        "receptor": t["receptor"], "score": t["score"].round(3),
                        "mean_ligand": t["mean_ligand"].round(3),
                        "mean_receptor": t["mean_receptor"].round(3),
                        "frac_ligand": (100 * t["frac_ligand"]).round(1),
                        "frac_receptor": (100 * t["frac_receptor"]).round(1),
                        "n_edges": t["n_edges"], "pval": t["pval"].map(_g3), "padj": t["padj"].map(_g3)})
    return out.head(5000).to_dict("records")


def _lr_tab():
    S, L = analysis._AN_ROW_STYLE, analysis._AN_LABEL_STYLE
    _text = ("sender", "receiver", "ligand", "receptor")
    _col = lambda cid, name: {"id": cid, "name": name, "type": "text" if cid in _text else "numeric"}
    return dbc.Tab(label="Ligand–Receptor", tab_id="tab-lr",
                   label_style={"fontSize": "12px", "padding": "4px 12px"},
                   children=[html.Div([
        html.Div([
            html.Span("Between:", style=L),
            dcc.Dropdown(id="lr-labels", options=[{"label": "Cluster (active cluster set)", "value": "cluster"}],
                         value=None, placeholder="Cell type annotation / cluster…",
                         style={"flex": "1", "minWidth": "170px", "fontSize": "11px"}),
            html.Span("k:", style=L),
            dcc.Input(id="lr-k", type="number", min=1, max=50, step=1, value=_MORAN_K, debounce=True,
                      style=ui._PLOT_CTRL_INPUT),
            html.Span("Permutations:", style=L),
            dcc.Input(id="lr-nperm", type="number", min=10, max=_LR_MAX_PERM, step=10, value=200,
                      debounce=True, style=ui._PLOT_CTRL_INPUT),
            dbc.Button("Run", id="lr-run-btn", color="primary", size="sm", n_clicks=0,
                       style={"fontSize": "11px", "padding": "2px 12px"}),
        ], style=S),
        html.Div([
            dcc.Upload(dbc.Button("Upload CSV of ligand,receptor pairs", color="secondary", size="sm",
                                  style={"fontSize": "11px", "padding": "2px 8px"}),
                       id="lr-upload", multiple=False, accept=".csv,.tsv,.txt"),
            dbc.Button("Use pairs of loaded CellNEST results", id="lr-cellnest-btn", color="secondary",
                       outline=True, size="sm", n_clicks=0, style={"fontSize": "11px", "padding": "2px 8px"}),
            html.Span(id="lr-pairs-info", style=L),
        ], style=S),
        dcc.Textarea(id="lr-pairs", value="",
                     placeholder="ligand,receptor — one pair per line (a header line 'ligand,receptor' is allowed)…",
                     style={**analysis._AN_INPUT_STYLE, "margin": "2px 8px", "height": "64px",
                            "flexShrink": "0", "fontFamily": "monospace", "resize": "vertical"}),
        html.Div(id="lr-status", style={**analysis._AN_STATUS_STYLE, "whiteSpace": "pre-wrap"}),
        html.Div(_LR_NOTE, id="lr-note", style=analysis._AN_NOTE_STYLE),
        html.Div([
            html.Div(dcc.Graph(id="lr-plot", responsive=True,
                               figure=figures._make_placeholder_fig("No ligand–receptor scores computed yet."),
                               config={"displayModeBar": "hover",
                                       "toImageButtonOptions": {"format": "svg", "filename": "xenium_lr_dotplot"}},
                               style={"width": "100%", "height": "100%"}),
                     id="lr-plot-container", style={"flex": "1", "minWidth": "0", "minHeight": "320px"}),
            html.Div([
                dash_table.DataTable(id="lr-table", data=[], columns=[_col("sender", "Sender"), _col("receiver", "Receiver"), _col("ligand", "Ligand"),
                             _col("receptor", "Receptor"), _col("score", "Score"),
                             _col("mean_ligand", "Mean L"), _col("mean_receptor", "Mean R"),
                             _col("frac_ligand", "% L"), _col("frac_receptor", "% R"),
                             _col("n_edges", "Adjacent pairs"), _col("pval", "p"), _col("padj", "FDR")],
                                     **_dark_table_style(12)),
                html.Div([
                    dbc.Button("Download CSV", id="lr-download-btn", color="secondary", size="sm",
                               n_clicks=0, style={"fontSize": "11px", "padding": "2px 8px"}),
                    html.Span("Top pairs shown:", style={**L, "marginLeft": "10px"}),
                    dcc.Input(id="lr-top-n", type="number", min=1, max=200, step=1, value=25,
                              debounce=True, style=ui._PLOT_CTRL_INPUT),
                    dcc.Download(id="lr-download"),
                ], style={"padding": "4px 0", "display": "flex", "alignItems": "center", "gap": "4px"}),
            ], style={"flex": "1.25", "minWidth": "0", "overflow": "auto", "padding": "0 6px"}),
        ], style={"display": "flex", "flex": "1", "minHeight": "0", "gap": "6px"}),
        ui._plot_dim_controls("lr"),
    ], style={"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0",
              "overflowY": "auto"})])
