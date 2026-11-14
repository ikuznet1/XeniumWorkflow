"""Methods and provenance record: one structured record per analysis the app ran.

Every analysis job (and the few analyses that run inside a callback) reports here with ONE
call, ``_prov_record(kind, ...)``, right after it has published its result. The record
states what was run, when, on which data, with which parameters, the statistical unit as
the job's own status line states it, the design, headline numbers, the caveats the job
raised, seeds, cache hit or computed, run time and software versions.

Records are kept per dataset as append-only JSON lines under ``<cache>/provenance/`` (the
file is trimmed to the newest records once it is larger than _PROV_MAX_BYTES), are listed
in the Provenance tab and are exported as JSON and as a methods text (Markdown, one
template per analysis kind, ``_prov_methods_md``). The id of the record is written into the
result (``res["prov_id"]``) and from there into the header of every CSV the app writes
(``_prov_csv_line``).
"""

import os
import re
import sys
import json
import time as _time
import uuid
import hashlib
import datetime as _dt
import threading

import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table

from . import core, jobs, caching, compat, dataview, stats, analysis
from . import rworker


_PROV_SCHEMA      = 1
_PROV_MAX_BYTES   = 4_000_000     # the file of a dataset is trimmed above this size ...
_PROV_MAX_RECORDS = 2000          # ... or above this many records ...
_PROV_KEEP        = 1500          # ... to its newest _PROV_KEEP records
_PROV_STR_MAX     = 6000          # longest string kept in a record (status lines are long)
_PROV_LIST_MAX    = 60            # longest list kept in a record

_prov_lock = threading.RLock()
_prov_state: dict = {"version": 0, "last_id": None}
_prov_recent: list = []            # the last records written by this process, whatever the cache directory
_prov_once: set = set()            # (dataset, kind, identity) recorded once per session (see _prov_record)
_PROV_RECENT_MAX = 400

# Analysis kinds and their titles. Every kind a job is registered under (jobs._job_start)
# has to be listed here or in _PROV_NOT_ANALYSES; tests/test_provenance.py checks both
# that and that the job's function calls _prov_record().
_PROV_KINDS = {
    "baysor": "Baysor segmentation", "proseg": "Proseg segmentation",
    "imgseg": "Image-based segmentation", "patches": "SpatialData store, tissue ROI and patches",
    "annotation": "Cell type annotation", "split": "SPLIT correction", "spage": "SpaGE imputation",
    "umap": "Clustering and UMAP",
    "spatial": "Spatial analysis", "niches": "Spatial niches",
    "de": "Differential expression (cells as observations)",
    "pseudobulk": "Pseudobulk differential expression",
    "proximity": "Proximity differential expression",
    "contact": "Contact-group pseudobulk differential expression",
    "geneset": "Gene-set score", "density": "Transcript density",
    "composition": "Composition between conditions",
    "moran": "Spatially variable genes (Moran's I)",
    "lr": "Ligand-receptor scores", "lrdecomp": "Ligand-receptor exposure decomposition",
    "compartments": "Tissue compartments", "spillover": "Spillover score and gene ownership",
    "segqc": "Segmentation QC", "coloc": "Stratified co-localisation",
    "aggregates": "Aggregates", "segfree": "Segmentation-free abundance",
    "banksy": "Banksy-style spatial domains", "joint": "Joint clustering across samples",
    "txfactor": "Transcript factors", "h5ad": "AnnData (.h5ad) conversion",
    "save": "Save as SpatialData", "cellnest": "CellNEST results loaded",
    "session": "Session saved or restored",
}
# job kinds that are not analyses (nothing a methods section would report)
_PROV_NOT_ANALYSES: dict = {}

# fields every record has (tests check them)
_PROV_REQUIRED = ("id", "schema", "kind", "title", "time", "data", "params", "unit", "design",
                  "headline", "warnings", "seeds", "cache", "runtime_s", "software", "status_line")

# sentences of a job's status line that are caveats (kept verbatim in "warnings")
_PROV_CAVEAT_RE = re.compile(
    r"smallest possible p|minimum achievable|smallest achievable|cannot be smaller|"
    r"non-integer|not raw counts|leak gate|far too small|pseudo-?replic|were pooled|"
    r"original counts were used|have no SPLIT-corrected|left out|not tested|skipped|dropped|"
    r"fallback|confound|not validated|too few|fewer than|no test|not a test|warning|caution|"
    r"could not|unavailable|not installed|approximate|were removed|not (fully )?testable|"
    r"validity check|upper bound|single-sample|do not generalise|describes? th(is|ese) section", re.I)


# ─── Storage ──────────────────────────────────────────────────────────────────
def _prov_dir() -> str:
    return os.path.join(core.CACHE_BASE_DIR, "provenance")


def _prov_dataset_id(key=None) -> str:
    """File stem of a dataset's records. Not the md5 tag of the other cache entries, so
    'Clean cache for this dataset' leaves the record of what was run in place."""
    key = caching._dataset_key() if key is None else key
    return hashlib.sha1(("prov:" + str(key)).encode()).hexdigest()[:16]


def _prov_path(key=None) -> str:
    return os.path.join(_prov_dir(), _prov_dataset_id(key) + ".jsonl")


def _prov_read(key=None) -> list:
    """The records of the current dataset, oldest first (unreadable lines are skipped)."""
    path = _prov_path(key)
    out = []
    with _prov_lock:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(rec, dict) and rec.get("id"):
                        out.append(rec)
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"  provenance: cannot read {os.path.basename(path)}: {exc}", flush=True)
    return out


def _prov_append(rec: dict, key=None) -> None:
    """Append one record; trim the file to its newest records when it has grown too large."""
    path = _prov_path(key)
    line = json.dumps(rec, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    with _prov_lock:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        try:
            too_big = os.path.getsize(path) > _PROV_MAX_BYTES
        except OSError:
            too_big = False
        if too_big or _prov_state.get("n_since_check", 0) >= 200:
            _prov_state["n_since_check"] = 0
            recs = _prov_read(key)
            if too_big or len(recs) > _PROV_MAX_RECORDS:
                keep = recs[-_PROV_KEEP:]
                while len(keep) > 1 and sum(len(json.dumps(r, ensure_ascii=False)) + 1
                                            for r in keep) > _PROV_MAX_BYTES:
                    keep = keep[len(keep) // 4 or 1:]
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    for r in keep:
                        fh.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
                os.replace(tmp, path)
        else:
            _prov_state["n_since_check"] = _prov_state.get("n_since_check", 0) + 1
        _prov_state["version"] += 1
        _prov_state["last_id"] = rec.get("id")


def _prov_get(prov_id: str):
    for rec in reversed(_prov_read()):
        if rec.get("id") == prov_id:
            return rec
    return None


# ─── Turning values into JSON ─────────────────────────────────────────────────
def _prov_plain(v, depth: int = 0):
    """*v* as a JSON value: scalars, short strings, short lists and small dicts are kept;
    tables and arrays are replaced by their shape (a table with a padj column also by the
    number of tests and of FDR < 0.05); anything else by its type name."""
    if v is None or isinstance(v, (bool, str)):
        return v if not isinstance(v, str) or len(v) <= _PROV_STR_MAX else v[:_PROV_STR_MAX] + "…"
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, pd.DataFrame):
        out = {"table_rows": int(len(v)), "table_columns": [str(c) for c in list(v.columns)[:40]]}
        if "padj" in v.columns:
            try:
                padj = pd.to_numeric(v["padj"], errors="coerce")
                out["n_tested"] = int(padj.notna().sum())
                out["n_fdr_below_0.05"] = int((padj < 0.05).sum())
            except Exception:
                pass
        return out
    if isinstance(v, pd.Series):
        return {"series_length": int(len(v))}
    if isinstance(v, np.ndarray):
        if v.ndim <= 1 and v.size <= _PROV_LIST_MAX and v.dtype.kind in "biufUO":
            return [_prov_plain(x, depth + 1) for x in v.tolist()]
        return {"array_shape": [int(s) for s in v.shape]}
    if hasattr(v, "shape") and hasattr(v, "nnz"):                       # scipy sparse
        return {"matrix_shape": [int(s) for s in v.shape]}
    if isinstance(v, dict):
        if depth >= 3:
            return {"keys": [str(k) for k in list(v)[:20]]}
        out = {}
        for k, x in list(v.items())[:80]:
            if str(k).startswith("_"):
                continue
            px = _prov_plain(x, depth + 1)
            if px is not _PROV_SKIP:
                out[str(k)] = px
        return out
    if isinstance(v, (list, tuple, set, frozenset)):
        seq = list(v)
        if depth >= 3:
            return {"list_length": len(seq)}
        out = [_prov_plain(x, depth + 1) for x in seq[:_PROV_LIST_MAX]]
        out = [x for x in out if x is not _PROV_SKIP]
        if len(seq) > _PROV_LIST_MAX:
            out.append(f"… ({len(seq)} in all)")
        return out
    if isinstance(v, (_dt.datetime, _dt.date)):
        return v.isoformat()
    return _PROV_SKIP


_PROV_SKIP = object()

# keys of a result dict that are bookkeeping, not headline numbers
_PROV_RESULT_DROP = {"gen", "uid", "key", "cache_tag", "out_dir", "sdata_path", "prov_id",
                     "cells_df", "cell_bounds", "expr", "outlines", "take", "emb", "pcs", "umap",
                     "labels", "sample", "H", "heat", "split_corrected_expr"}


def _prov_headline(result) -> dict:
    if result is None:
        return {}
    if not isinstance(result, dict):
        p = _prov_plain(result)
        return {} if p is _PROV_SKIP else {"result": p}
    out = {}
    if isinstance(result.get("cells_df"), pd.DataFrame):
        out["n_cells"] = int(len(result["cells_df"]))
    if hasattr(result.get("expr"), "shape"):
        out["n_genes_in_matrix"] = int(result["expr"].shape[1])
    for k, v in result.items():
        if k in _PROV_RESULT_DROP or str(k).startswith("_") or k in ("params", "notes", "warnings"):
            continue
        p = _prov_plain(v, 1)
        if p is not _PROV_SKIP:
            out[str(k)] = p
    return out


def _prov_sentences(text: str) -> list:
    out = []
    for line in str(text or "").splitlines():
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z⚠(])", line.strip()):
            s = s.strip()
            if s:
                out.append(s)
    return out


def _prov_warnings(message, result, extra=None) -> list:
    """The caveats a job raised, verbatim: its "⚠" and "Note:" lines, result["notes"] /
    ["warnings"] / ["leak_summary"], and the sentences of its status line that name a known
    caveat (minimum achievable p, non-integer counts, pooled samples, cells left out ...)."""
    seen, out = set(), []

    def add(s):
        s = re.sub(r"\s+", " ", str(s or "")).strip()
        s = re.sub(r"^(⚠|Note:)\s*", "", s).strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s[:1500])

    for w in extra or []:
        add(w)
    if isinstance(result, dict):
        for k in ("warnings", "notes"):
            for w in (result.get(k) or []) if isinstance(result.get(k), (list, tuple)) else []:
                add(w[0] if isinstance(w, (list, tuple)) and w else w)
        if result.get("leak_summary"):
            for ln in str(result["leak_summary"]).splitlines():
                add(ln)
    for line in str(message or "").splitlines():
        if line.strip().startswith(("⚠", "Note:")):
            add(line)
    for s in _prov_sentences(message):
        if _PROV_CAVEAT_RE.search(s) and not s.startswith("Unit of the p-values"):
            key = re.sub(r"\s+", " ", re.sub(r"^(⚠|Note:)\s*", "", s)).strip().lower().rstrip(".")
            if not any(key in w.lower() for w in out):      # already part of a "⚠" line
                add(s)
    return out


def _prov_unit(message, result) -> dict:
    """The statistical unit and n exactly as the job states them: the "Unit of the
    p-values: ..." line of its status line (the CSV headers print the same text), else the
    units_note of its result. Nothing is derived here."""
    stmt, src = "", ""
    m = re.search(r"Unit of the p-values:\s*(.+)", str(message or ""))
    if m:
        stmt, src = m.group(1).strip(), "status line"
    else:
        # no such line: every sentence of the status line that speaks of the unit(s), verbatim
        said = [re.sub(r"^⚠\s*", "", x) for x in _prov_sentences(message) if re.search(r"\bunits?\s*:|\bunit\(s\)|\bunit of\b|\bindependent units?\b", x, re.I)]
        if said:
            stmt, src = " ".join(said), "status line"
        elif isinstance(result, dict) and result.get("units_note"):
            stmt, src = str(result["units_note"]), "result"
    n = {}
    if isinstance(result, dict):
        for k in ("n_a", "n_b", "n_cells", "n_samples", "n_donors", "n_genes", "n_pairs", "n_tested"):
            v = result.get(k)
            if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                n[k] = int(v)
        if result.get("unit_word"):
            n["unit_word"] = str(result["unit_word"])
    return {"statement": stmt, "source": src, "n": n}


# ─── Context: data, design, software ──────────────────────────────────────────
def _prov_counts_used(seg_source, corrected, result) -> "bool | None":
    """The counts an analysis used (True: SPLIT-corrected, False: original, None: it read
    no count matrix or did not say). What the job reports wins (result["corrected"]). A
    job that reports nothing is never taken at its word for "corrected": the requested
    mode is only what was asked for, and the view falls back to the original counts when
    corrected ones do not exist for every sample (analysis._an_samples) - so the fallback
    rule is applied here as well."""
    if isinstance(result, dict) and "corrected" in result:
        return None if result["corrected"] is None else bool(result["corrected"])
    if corrected is None:
        return None
    if not corrected:
        return False
    try:
        seg = seg_source or "xenium"
        if core._seg_tool(seg) != "xenium":
            alt = core._seg_result(seg)
            return bool(alt is not None and alt.get("split_corrected_expr") is not None)
        if dataview.EXTRA_DATASETS:
            return bool(dataview._multi_use_corrected(True))
        return core.DATA.get("split_corrected_expr") is not None
    except Exception:
        return None


def _prov_data(seg_source=None, corrected=None, result=None, labels=None) -> dict:
    dsets = [core.DATA] + list(dataview.EXTRA_DATASETS)
    samples = []
    for i, ds in enumerate(dsets):
        try:
            samples.append({"label": f"S{i}", "key": dataview._ds_key(ds), "name": dataview._ds_name(ds),
                            "n_cells": int(len(ds["df"])),
                            "n_genes": int(len(ds.get("gene_names") or []))})
        except Exception:
            continue
    seg = seg_source or "xenium"
    out = {"dataset_key": caching._dataset_key(), "dataset_name": dataview._ds_name(core.DATA),
           "n_samples": len(samples), "samples": samples,
           "n_cells": int(sum(s["n_cells"] for s in samples)),
           "n_genes": samples[0]["n_genes"] if samples else 0,
           "seg_source": seg, "seg_tool": core._seg_tool(seg),
           "seg_tag": seg.split(":", 1)[1] if ":" in seg else ""}
    alt = None
    try:
        alt = core._seg_result(seg) if out["seg_tool"] != "xenium" else None
    except Exception:
        alt = None
    if alt is not None:
        try:
            out["seg_n_cells"] = int(len(alt["cells_df"]))
            out["seg_run"] = os.path.basename(str(alt.get("out_dir") or "").rstrip("/"))
            pj = os.path.join(str(alt.get("out_dir") or ""), "params.json")
            if os.path.exists(pj):
                with open(pj) as fh:
                    out["seg_params"] = _prov_plain(json.load(fh), 1)
        except Exception:
            pass
    if "_df_original" in core.DATA:
        try:
            out["subset"] = {"n_cells": int(len(core.DATA["df"])),
                             "of": int(len(core.DATA["_df_original"])),
                             "filters": _prov_plain(core.DATA.get("_subset_desc") or "not recorded")}
        except Exception:
            out["subset"] = {"filters": "not recorded"}
    else:
        out["subset"] = None
    used = _prov_counts_used(seg, corrected, result)
    out["counts"] = {
        "requested": None if corrected is None else ("corrected" if corrected else "original"),
        "used": None if used is None else ("SPLIT-corrected" if used else "original"),
        "n_cells_left_out": (int(result["n_uncorrected"]) if isinstance(result, dict)
                             and isinstance(result.get("n_uncorrected"), (int, np.integer)) else None)}
    if labels is not None:
        kind, method = (tuple(labels) + (None, None))[:2] if isinstance(labels, (list, tuple)) else (labels, None)
        out["labels"] = {"label_kind": None if kind is None else str(kind),
                         "cluster_method": None if method is None else str(method),
                         "category": (str(result.get("category")) if isinstance(result, dict)
                                      and result.get("category") else None)}
    else:
        out["labels"] = None
    return out


def _prov_design():
    """The sample design as the user set it (their own labels), None with one sample."""
    try:
        rows = stats._design_samples()
    except Exception:
        return None
    if len(rows) < 2:
        return None
    names = stats._design_names()
    donors = [r["donor"] for r in rows if r["cond"] and r["donor"]]
    # paired by the rule the tests use (stats._sl_units): every included sample has a donor
    # and at least 2 donors occur in both conditions - not "some donor occurs in both"
    _u = stats._sl_units([r["cond"] for r in rows], [r["donor"] for r in rows])
    paired = bool(_u["paired"])
    _both = sorted({d for d in set(donors)
                    if {x["cond"] for x in rows if x["donor"] == d and x["cond"]} == {"A", "B"}})
    return {"condition_names": {"A": names.get("A", "A"), "B": names.get("B", "B")},
            "samples": [{"label": r["label"], "name": r["name"], "condition": r["cond"] or "excluded",
                         "donor": r["donor"], "batch": r["batch"], "n_cells": r["n_cells"]} for r in rows],
            "n_a": sum(r["cond"] == "A" for r in rows), "n_b": sum(r["cond"] == "B" for r in rows),
            "donors_given": bool(donors), "paired_donors": paired,
            "donors_in_both": _both,
            "donors_on_all": bool(donors) and all(r["donor"] for r in rows if r["cond"] in ("A", "B")),
            "units_note": _u["note"],
            "summary": stats._design_summary()}


_prov_sw_cache: dict = {}


def _prov_git_commit() -> str:
    """Commit of the app's checkout, read from .git without starting git ('' if none)."""
    if "commit" in _prov_sw_cache:
        return _prov_sw_cache["commit"]
    commit = ""
    try:
        gd = os.path.join(compat.REPO_DIR, ".git")
        if os.path.isfile(gd):                                  # a worktree: "gitdir: <path>"
            with open(gd) as fh:
                gd = fh.read().split(":", 1)[1].strip()
        with open(os.path.join(gd, "HEAD")) as fh:
            head = fh.read().strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            common = gd
            if os.path.exists(os.path.join(gd, "commondir")):
                with open(os.path.join(gd, "commondir")) as fh:
                    common = os.path.normpath(os.path.join(gd, fh.read().strip()))
            p = os.path.join(common, ref)
            if os.path.exists(p):
                with open(p) as fh:
                    commit = fh.read().strip()
            else:
                with open(os.path.join(common, "packed-refs")) as fh:
                    for ln in fh:
                        if ln.strip().endswith(" " + ref):
                            commit = ln.split()[0]
        else:
            commit = head
    except Exception:
        commit = ""
    _prov_sw_cache["commit"] = commit[:12]
    return _prov_sw_cache["commit"]


def _prov_pkg_version(name: str) -> str:
    if name not in _prov_sw_cache:
        try:
            from importlib import metadata as _md
            _prov_sw_cache[name] = _md.version(name)
        except Exception:
            mod = sys.modules.get(name.replace("-", "_"))
            _prov_sw_cache[name] = str(getattr(mod, "__version__", "")) if mod else ""
    return _prov_sw_cache[name]


def _prov_r_versions(packages) -> dict:
    """R and R package versions, asked of the R that the job has just used: the R worker
    process of the job (it reported them with its result, see rworker), else the R of
    this process (rpy2 already imported here); {} when R was not started."""
    return rworker._rw_versions_seen(packages) or rworker._rw_r_versions(packages)


def _prov_software(packages=(), r_packages=(), tools=None) -> dict:
    out = {"app_commit": _prov_git_commit(), "python": sys.version.split()[0]}
    for p in ("numpy", "scipy", "pandas", "scikit-learn") + tuple(packages or ()):
        v = _prov_pkg_version(p)
        if v:
            out[p] = v
    if r_packages:
        r = _prov_r_versions(r_packages)
        if r:
            out["r"] = r
    if tools:
        out["tools"] = {str(k): str(v) for k, v in tools.items() if v}
    return out


# ─── The one call ─────────────────────────────────────────────────────────────
def _prov_record(kind: str, title: str = None, *, seg_source=None, corrected=None, params=None,
                 result=None, message=None, labels=None, seeds=None, cache=None,
                 packages=(), r_packages=(), tools=None, warnings=None, outputs=None,
                 t0: float = None, uses_design: bool = False, once=None):
    """Write the provenance record of an analysis that has just published its result.

    kind: a key of _PROV_KINDS. params: every parameter of the run, user-set ones and the
    defaults actually used. result: the job's result dict (scalars and table shapes become
    the headline numbers; ``result["prov_id"]`` is set to the record id). message: the
    job's status line (the unit of the p-values and the caveats are taken from it,
    verbatim). labels: (label kind, cluster method) of the cell labels used. cache: the
    result came from the cache. once: a hashable identity of something that is recorded once
    per session (a cached segmentation run that is loaded: picking it again in the
    dropdown reads the same files and is not another event of the analysis); a second
    call with the same identity writes nothing. Never raises: a failure is printed and
    None returned, the analysis itself is not affected."""
    try:
        if once is not None:
            with _prov_lock:
                okey = (caching._dataset_key(), kind, once)
                if okey in _prov_once:
                    return None
                _prov_once.add(okey)
        now = _time.time()
        job = jobs._job_current()
        started = t0 if t0 is not None else (job.started if job is not None else None)
        rid = "prov-" + _dt.datetime.fromtimestamp(now).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        prm = _prov_plain(dict(params or {}), 0)
        if isinstance(result, dict) and isinstance(result.get("params"), dict):
            prm.setdefault("resolved", _prov_plain(result["params"], 1))
        sd = dict(seeds or {})
        for src in (params or {}), (result if isinstance(result, dict) else {}):
            for k in ("seed", "random_state"):
                if k in src and isinstance(src[k], (int, np.integer)) and not isinstance(src[k], bool):
                    sd.setdefault(k, int(src[k]))
        rec = {
            "id": rid, "schema": _PROV_SCHEMA, "kind": kind,
            "title": title or _PROV_KINDS.get(kind, kind),
            "time": _dt.datetime.fromtimestamp(now, _dt.timezone.utc).isoformat(timespec="seconds"),
            "time_local": _dt.datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "job_id": job.id if job is not None else None,
            "data": _prov_data(seg_source, corrected, result, labels),
            "params": prm,
            "unit": _prov_unit(message, result),
            "design": _prov_design() if (uses_design or len(dataview.EXTRA_DATASETS)) else None,
            "design_used": bool(uses_design),
            "headline": _prov_headline(result),
            "warnings": _prov_warnings(message, result, warnings),
            "seeds": sd,
            "cache": "cache" if _prov_from_cache(cache, result, message) else "computed",
            "runtime_s": None if started is None else round(max(0.0, now - started), 2),
            "software": _prov_software(packages, r_packages, tools),
            "status_line": _prov_plain(str(message or "")),
            "outputs": _prov_plain(list(outputs or [])),
        }
        json.dumps(rec, allow_nan=False)            # a record that cannot be written is a bug: say so
        _prov_append(rec)
        with _prov_lock:
            _prov_recent.append(rec)
            del _prov_recent[:-_PROV_RECENT_MAX]
        if isinstance(result, dict):
            result["prov_id"] = rid
        return rec
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"  provenance: the record of '{kind}' could not be written: {exc}", flush=True)
        return None


_PROV_CACHE_RE = re.compile(r"\(cached|^Done \(cached|from the cache|^Loaded —|^Auto-loaded", re.I)


def _prov_from_cache(cache, result, message) -> bool:
    """Cache hit or computed: what the job says (cache=True / False), else the flag of its
    result (from_cache / cached), else its status line ("Done (cached) ...")."""
    if cache is not None:
        return bool(cache)
    if isinstance(result, dict):
        # the job's own flag wins, also when it says "computed": a status line may mention
        # a cached input ("binning 0 s (cached), fit 35 s") of a freshly computed result
        flags = [result[k] for k in ("from_cache", "cached") if k in result and result[k] is not None]
        if flags:
            return any(bool(f) for f in flags)
    return bool(_PROV_CACHE_RE.search(str(message or "")))


def _prov_labels_source(labels_key) -> str:
    """Segmentation an annotation's labels belong to, from its key in the annotation
    state: "labels" -> xenium, "labels_celltypist" -> xenium,
    "labels_baysor" / "labels_rctd_baysor" -> baysor ..."""
    parts = str(labels_key or "labels").split("_")
    tools = [p for p in parts if p in ("baysor", "proseg", "imgseg")]
    return tools[0] if tools else "xenium"


def _prov_labels_summary(labels):
    if labels is None:
        return None
    try:
        s = pd.Series(labels)
        return {"n_cells_labelled": int(s.notna().sum()), "n_cell_types": int(s.dropna().nunique())}
    except Exception:
        return None


def _prov_run_source(tool: str, result) -> str:
    """seg-source value ("tool:tag") of the run a result dict belongs to."""
    try:
        out_dir = str((result or {}).get("out_dir") or "")
        tag = caching._seg_run_tag(os.path.basename(out_dir.rstrip("/")), tool)
        return f"{tool}:{tag}" if tag and isinstance(tag, str) else tool
    except Exception:
        return tool


def _prov_tool_seen(tool: str, version_output) -> None:
    """Remember what `<tool> --version` printed (the run functions probe the binary anyway)."""
    try:
        if isinstance(version_output, bytes):
            version_output = version_output.decode("utf-8", "replace")
        txt = " ".join(str(version_output or "").split())[:120]
        if txt:
            _prov_sw_cache["tool:" + tool] = txt
    except Exception:
        pass


def _prov_tool_version(tool: str, binary=None) -> str:
    """Version string of an external tool: what its probe printed (_prov_tool_seen), else
    `<binary> --version` asked once with a short limit; "" when unknown."""
    key = "tool:" + tool
    if key not in _prov_sw_cache and binary:
        try:
            import subprocess
            r = subprocess.run([binary, "--version"], capture_output=True, timeout=10)
            _prov_tool_seen(tool, r.stdout or r.stderr)
        except Exception:
            pass
    return _prov_sw_cache.get(key, "")


def _prov_csv_line(res) -> str:
    """Comment line for a CSV header naming the provenance record the table came from."""
    pid = res.get("prov_id") if isinstance(res, dict) else None
    return (f"# provenance record: {pid} (Provenance tab; exported with the methods text)\n" if pid else
            "# provenance record: none (this result was restored from the cache of an earlier session "
            "or computed before records were kept)\n")


def _csv_comments(lines) -> str:
    """The comment block of a CSV download: *lines* (a list of lines, or one text) with
    every physical line starting with '#', and a final newline. A note that carries a line
    break of its own (a warning on a second line) would otherwise put a prose line without
    '#' into the header block, which a reader with comment='#' takes for the column header.
    Every CSV builder passes its header through here."""
    if isinstance(lines, str):
        lines = [lines]
    out = []
    for ln in lines:
        for part in str(ln).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            if not part.strip():
                continue
            out.append(part if part.startswith("#") else "# " + part)
    return "\n".join(out) + "\n" if out else ""


def _prov_csv_stamp(res, text: str) -> str:
    """*text* (a CSV with leading '#' comment lines, or only such lines) with the
    provenance line added as the last line of that leading comment block."""
    lines = str(text).split("\n")
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        i += 1
    lines[i:i] = [_prov_csv_line(res).rstrip("\n")]
    out = "\n".join(lines)
    return out if i < len(lines) - 1 or out.endswith("\n") else out + "\n"


# ─── Export ───────────────────────────────────────────────────────────────────
def _prov_export_json(records=None) -> str:
    recs = _prov_read() if records is None else list(records)
    doc = {"schema": _PROV_SCHEMA, "exported": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "dataset": dataview._ds_name(core.DATA), "n_records": len(recs), "records": recs}
    return json.dumps(doc, ensure_ascii=False, indent=1)


# ─── The Provenance tab ───────────────────────────────────────────────────────
def _prov_rows(kind=None) -> list:
    rows = []
    for r in reversed(_prov_read()):
        if kind and r.get("kind") != kind:
            continue
        d = r.get("data") or {}
        rows.append({"id": r["id"], "time": str(r.get("time_local") or r.get("time") or "").replace("T", " "),
                     "kind": r.get("kind"), "title": r.get("title"),
                     "seg": d.get("seg_source") or "", "counts": (d.get("counts") or {}).get("used") or "",
                     "cache": r.get("cache"), "runtime_s": r.get("runtime_s"),
                     "n_warnings": len(r.get("warnings") or [])})
    return rows


def _prov_kind_options() -> list:
    kinds = sorted({r.get("kind") for r in _prov_read() if r.get("kind")})
    return [{"label": _PROV_KINDS.get(k, k), "value": k} for k in kinds]


def _prov_detail_text(rec) -> str:
    if not rec:
        return "Click a row to see the full record."
    return json.dumps(rec, ensure_ascii=False, indent=1)


_PROV_NOTE = ("One record per analysis run on this dataset (kept under the cache directory, newest first): "
              "what was run, on which data, parameters, the statistical unit as the analysis stated it, "
              "caveats raised, seeds, cache hit or computed, run time and software versions. "
              "'Methods text' writes a Markdown methods section from the newest record of each analysis; "
              "'Full record' saves every record as JSON. CSV downloads name their record in the header.")


def _prov_tab():
    S, L = analysis._AN_ROW_STYLE, analysis._AN_LABEL_STYLE
    _col = lambda cid, name, num=False: {"id": cid, "name": name, "type": "numeric" if num else "text"}
    _btn = {"fontSize": "11px", "padding": "2px 8px"}
    return dbc.Tab(label="Provenance", tab_id="tab-provenance",
                   label_style={"fontSize": "12px", "padding": "4px 12px"},
                   children=[html.Div([
        html.Div([
            html.Span("Methods and provenance record", style={**L, "color": core.TEXT, "flex": "1"}),
            html.Span("Kind:", style=L),
            dcc.Dropdown(id="prov-kind", options=[], value=None, placeholder="all", clearable=True,
                         style={"width": "220px", "fontSize": "11px"}),
            dbc.Button("Refresh", id="prov-refresh-btn", color="primary", size="sm", n_clicks=0,
                       style={"fontSize": "11px", "padding": "2px 12px"}),
        ], style=S),
        html.Div(id="prov-status", style={**analysis._AN_STATUS_STYLE, "whiteSpace": "pre-wrap"}),
        html.Div(_PROV_NOTE, id="prov-note", style=analysis._AN_NOTE_STYLE),
        html.Div([
            dash_table.DataTable(
                id="prov-table", data=[], row_selectable=False,
                columns=[_col("time", "When"), _col("kind", "Kind"), _col("title", "Analysis"),
                         _col("seg", "Segmentation"), _col("counts", "Counts"), _col("cache", "Source"),
                         _col("runtime_s", "Seconds", True), _col("n_warnings", "Caveats", True),
                         _col("id", "Record id")],
                **{**stats._dark_table_style(10),
                   "style_cell": {**stats._dark_table_style(10)["style_cell"], "maxWidth": "260px"}}),
            html.Div([
                dbc.Button("Methods text (.md)", id="prov-export-md-btn", color="secondary", size="sm",
                           n_clicks=0, style=_btn),
                dbc.Button("Full record (.json)", id="prov-export-json-btn", color="secondary", size="sm",
                           n_clicks=0, style={**_btn, "marginLeft": "6px"}),
                html.Span("Click a row for the full record.", style={**L, "marginLeft": "8px"}),
                dcc.Download(id="prov-download"),
            ], style={"padding": "4px 0"}),
            html.Pre("Click a row to see the full record.", id="prov-detail",
                     style={"fontSize": "10px", "color": core.TEXT, "backgroundColor": "#0d1117",
                            "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                            "padding": "6px", "whiteSpace": "pre-wrap", "wordBreak": "break-word",
                            "maxHeight": "320px", "overflowY": "auto", "margin": "4px 0"}),
        ], style={"flex": "1", "minHeight": "0", "overflow": "auto", "padding": "0 8px"}),
    ], style={"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0",
              "overflowY": "auto"})])



# ─── Methods text ─────────────────────────────────────────────────────────────
# References of the published methods the app implements or calls. Entries whose details
# could not be confirmed from memory when this list was written carry "[verify]".
_PROV_REFS = {
    "baysor": "Petukhov V, Xu RJ, Soldatov RA, et al. Cell segmentation in imaging-based spatial "
              "transcriptomics. Nat Biotechnol 40, 345-354 (2022).",
    "proseg": "Jones DC, Elz AE, Hadadianpour A, et al. Cell simulation as cell segmentation. "
              "Nat Methods (2025). [verify volume and pages]",
    "rctd": "Cable DM, Murray E, Zou LS, et al. Robust decomposition of cell type mixtures in spatial "
            "transcriptomics. Nat Biotechnol 40, 517-526 (2022).",
    "split": "SPLIT (R package, github.com/bdsc-tds/SPLIT): purification of cell-level counts in "
             "imaging-based spatial transcriptomics using RCTD doublet-mode results. [verify authors, "
             "title and venue of the accompanying publication]",
    "spage": "Abdelaal T, Mourragui S, Mahfouz A, Reinders MJT. SpaGE: Spatial Gene Enhancement using "
             "scRNA-seq. Nucleic Acids Res 48, e107 (2020).",
    "celltypist": "Dominguez Conde C, Xu C, Jarvis LB, et al. Cross-tissue immune cell analysis reveals "
                  "tissue-specific features in humans. Science 376, eabl5197 (2022).",
    "seurat": "Hao Y, Stuart T, Kowalski MH, et al. Dictionary learning for integrative, multimodal and "
              "scalable single-cell analysis. Nat Biotechnol 42, 293-304 (2024).",
    "deseq2": "Love MI, Huber W, Anders S. Moderated estimation of fold change and dispersion for "
              "RNA-seq data with DESeq2. Genome Biol 15, 550 (2014).",
    "edger": "Robinson MD, McCarthy DJ, Smyth GK. edgeR: a Bioconductor package for differential "
             "expression analysis of digital gene expression data. Bioinformatics 26, 139-140 (2010).",
    "limma": "Ritchie ME, Phipson B, Wu D, et al. limma powers differential expression analyses for "
             "RNA-sequencing and microarray studies. Nucleic Acids Res 43, e47 (2015).",
    "propeller": "Phipson B, Sim CB, Porrello ER, et al. propeller: testing for differences in cell type "
                 "proportions in single cell data. Bioinformatics 38, 4720-4726 (2022).",
    "harmony": "Korsunsky I, Millard N, Fan J, et al. Fast, sensitive and accurate integration of "
               "single-cell data with Harmony. Nat Methods 16, 1289-1296 (2019).",
    "banksy": "Singhal V, Chou N, Lee J, et al. BANKSY unifies cell typing and tissue domain "
              "segmentation for scalable spatial omics data analysis. Nat Genet 56, 431-441 (2024).",
    "cellpose": "Stringer C, Wang T, Michaelos M, Pachitariu M. Cellpose: a generalist algorithm for "
                "cellular segmentation. Nat Methods 18, 100-106 (2021).",
    "sopa": "Blampey Q, Mulder K, Gardet M, et al. Sopa: a technology-invariant pipeline for analyses "
            "of image-based spatial omics. Nat Commun 15, 4981 (2024).",
    "spatialdata": "Marconato L, Palla G, Yamauchi KA, et al. SpatialData: an open and universal data "
                   "framework for spatial omics. Nat Methods 22, 58-62 (2025).",
    "moran": "Moran PAP. Notes on continuous stochastic phenomena. Biometrika 37, 17-23 (1950).",
    "cliff_ord": "Cliff AD, Ord JK. Spatial Processes: Models and Applications. Pion, London (1981).",
    "bh": "Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful "
          "approach to multiple testing. J R Stat Soc B 57, 289-300 (1995).",
    "leiden": "Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing well-connected "
              "communities. Sci Rep 9, 5233 (2019).",
    "umap": "McInnes L, Healy J, Melville J. UMAP: Uniform Manifold Approximation and Projection for "
            "dimension reduction. arXiv:1802.03426 (2018).",
    "cellphonedb": "Efremova M, Vento-Tormo M, Teichmann SA, Vento-Tormo R. CellPhoneDB: inferring "
                   "cell-cell communication from combined expression of multi-subunit ligand-receptor "
                   "complexes. Nat Protoc 15, 1484-1506 (2020).",
    "ficture": "Si Y, Lee C, Hwang Y, et al. FICTURE: scalable segmentation-free analysis of "
               "submicron-resolution spatial transcriptomics. Nat Methods 21, 1843-1854 (2024). "
               "[verify volume and pages]",
    "cellnest": "Zohora FT, Paliwal D, Flores-Figueroa E, et al. CellNEST reveals cell-cell relay "
                "networks using attention mechanisms on spatial transcriptomics. Nat Methods (2025). "
                "[verify authors, volume and pages]",
    "modulescore": "Tirosh I, Izar B, Prakadan SM, et al. Dissecting the multicellular ecosystem of "
                   "metastatic melanoma by single-cell RNA-seq. Science 352, 189-196 (2016).",
    "anndata": "Virshup I, Rybakov S, Theis FJ, Angerer P, Wolf FA. anndata: Access and store annotated "
               "data matrices. J Open Source Softw 9, 4371 (2024). [verify volume and article number]",
}


def _pm_v(d, key, default="not recorded"):
    v = (d or {}).get(key)
    return default if v is None or v == "" else v


def _pm_num(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _pm_params(params: dict, skip=()) -> str:
    """'name = value; ...' of a record's parameters (None left out, long text shortened)."""
    out = []

    def walk(prefix, d):
        for k, v in d.items():
            if k in skip or v is None or v == "" or v == [] or v == {}:
                continue
            name = f"{prefix}{k}"
            if isinstance(v, dict):
                walk(name + ".", v)
            elif isinstance(v, list):
                txt = ", ".join(_pm_num(x) if x is not None else "-" for x in v)
                out.append(f"{name} = [{txt if len(txt) <= 160 else txt[:160] + '…'}]")
            else:
                txt = " ".join(_pm_num(v).split())
                out.append(f"{name} = {txt if len(txt) <= 160 else txt[:160] + f'… ({len(txt)} characters)'}")
    walk("", dict(params or {}))
    return "; ".join(out)


def _pm_counts(r) -> str:
    c = (r.get("data") or {}).get("counts") or {}
    used = c.get("used")
    if not used:
        return ""
    txt = f"{used} counts were used"
    if c.get("requested") == "corrected" and used != "SPLIT-corrected":
        txt = "SPLIT-corrected counts were requested but original counts were used"
    if c.get("n_cells_left_out"):
        txt += f"; {c['n_cells_left_out']:,} cells without SPLIT-corrected counts were left out"
    return txt[0].upper() + txt[1:] + "."


def _pm_labels(r) -> str:
    lab = (r.get("data") or {}).get("labels") or {}
    if not lab.get("label_kind"):
        return ""
    txt = f"Cell labels: {lab['label_kind']}"
    if lab.get("category"):
        txt += f" ({lab['category']})"
    if lab.get("cluster_method") and "cluster" in str(lab["label_kind"]):
        txt += f", cluster set {lab['cluster_method']}"
    return txt + "."


def _pm_seg(r) -> str:
    d = r.get("data") or {}
    tool = d.get("seg_tool") or "xenium"
    if tool == "xenium":
        return "the Xenium segmentation"
    return f"the {tool} segmentation" + (f" (run {d['seg_tag']})" if d.get("seg_tag") else "")


def _pm_design(r) -> str:
    de = r.get("design")
    if not de or not r.get("design_used"):
        return ""
    nm = de.get("condition_names") or {}
    parts = []
    for c in ("A", "B"):
        labs = [s["label"] + (f" (donor {s['donor']})" if s.get("donor") else "")
                + (f" (batch {s['batch']})" if s.get("batch") else "")
                for s in de.get("samples", []) if s.get("condition") == c]
        parts.append(f"condition {c}" + (f" ('{nm.get(c)}')" if nm.get(c) and nm.get(c) != c else "")
                     + ": " + (", ".join(labs) if labs else "no samples"))
    ex = [s["label"] for s in de.get("samples", []) if s.get("condition") == "excluded"]
    txt = "Sample design as set in the app: " + "; ".join(parts)
    if ex:
        txt += "; excluded: " + ", ".join(ex)
    if not de.get("donors_given"):
        return txt + ". No donor labels were given: each sample counted as one unit."
    both = de.get("donors_in_both")
    if de.get("paired_donors"):
        txt += (". Donor labels were given for every included sample and at least 2 donors occur in both "
                "conditions: the design is paired by donor.")
    elif both:
        txt += (f". Donor labels were given; {', '.join(map(str, both))} occur{'s' if len(both) == 1 else ''} "
                f"in both conditions, but the design is not paired (pairing needs a donor on every included "
                f"sample and at least 2 donors in both conditions): the samples of a donor within a condition "
                f"were combined into one unit and the units compared as independent.")
    elif both is None:
        # a record written before the rule was recorded with the design
        txt += ". Donor labels were given; the design is not paired by donor."
    else:
        txt += (". Donor labels were given; no donor occurs in both conditions (not paired): the samples of a "
                "donor within a condition were combined into one unit.")
    if de.get("donors_on_all") is False:
        txt += " Some included samples have no donor label; each of them is its own unit."
    return txt


# One function per analysis kind: (sentences, reference keys). The sentences state the
# method as the app implements it; numbers come from the record. Nothing is claimed that
# the record does not hold - the job's own status line follows every paragraph verbatim.
def _pm_de(r):
    H, P = r["headline"], r["params"]
    return ([f"Differential expression between two groups of cells of {_pm_seg(r)} "
             f"(A: {_pm_v(H, 'label_a')}, n = {_pm_num(_pm_v(H, 'n_a'))} cells; "
             f"B: {_pm_v(H, 'label_b')}, n = {_pm_num(_pm_v(H, 'n_b'))} cells) was tested for every measured gene "
             f"({_pm_num(_pm_v(H, 'n_genes'))} genes) with the two-sided Wilcoxon rank-sum test on "
             f"log1p(counts per 10,000)"
             + (f" (the pooled samples have different panels: the library size of a cell is its total over "
                f"the {_pm_num(H['lib_shared_genes'])} shared genes that were tested)"
                if H.get("lib_shared_genes") else "")
             + " and p-values were adjusted with the Benjamini-Hochberg procedure "
             f"over the tested genes.",
             "Cells were treated as independent observations; this test does not use the sample design, "
             "so with cells from few samples its p-values are too small (pseudo-replication) and rank genes "
             "rather than establish significance.",
             "A gene counted as detected in a cell with a count of at least one transcript.",
             ("The leak gate (a model of transcripts mis-assigned from neighbouring cell types, with gene "
              "ownership per cell type) was " + ("switched off for this run."
                                                 if (P.get("options") or {}).get("leak_gate") is False else
                                                 "requested; what it did is in the app's report below "
                                                 "(it is computed only when group A is one annotated cell "
                                                 "type)."))], ["bh"])


def _pm_pseudobulk(r):
    H, P = r["headline"], r["params"]
    uw = _pm_v(H, "unit_word", "samples")
    refs = ["bh"]
    ml = str(_pm_v(H, "method_label", ""))
    refs += ["deseq2"] if "DESeq2" in ml else []
    refs += ["edger"] if "edgeR" in ml else []
    s = [f"Pseudobulk differential expression of the population '{_pm_v(H, 'population', _pm_v(P, 'population'))}' "
         f"between the two conditions of the sample design: the counts of the population's cells were summed "
         f"per {'donor and condition' if uw == 'donors' else 'sample'} "
         f"(A: {_pm_num(_pm_v(H, 'n_a'))} {uw}; B: {_pm_num(_pm_v(H, 'n_b'))} {uw}); a sample with fewer than "
         f"{_pm_num(_pm_v(P, 'min_cells'))} cells of the population was dropped and a gene with a total count "
         f"below {_pm_num(_pm_v(H, 'min_count', _pm_v(P, 'min_count')))} was not tested.",
         f"Model: {ml or 'not recorded'}, design {_pm_v(H, 'design')}; log2 fold changes are A / B "
         f"(unshrunken maximum-likelihood estimates"
         + (f"; shrunken estimates: {H['shrink']}" if H.get("shrink") and H.get("shrink") != "none" else "")
         + "); adjusted p-values are those of the fitting package (Benjamini-Hochberg).",
         f"{_pm_num(_pm_v(H, 'n_genes'))} genes were tested, {_pm_num(_pm_v(H, 'n_filtered'))} were not "
         f"(low count), and {_pm_num(_pm_v(H, 'n_padj_na', 0))} have no adjusted p-value.",
         ]
    if H.get("counts_rounded"):
        # stated plainly, in place of "original counts": the option is off by default
        s.append("The count matrix of this data does NOT hold raw counts (it has non-integer values), and "
                 "noninteger_counts = round_sums was set when it was loaded: the values were summed per "
                 "pseudobulk column (per sample, or per donor and condition) and each sum was rounded to a "
                 "whole number. The count model of DESeq2 / edgeR is then an approximation, and these results "
                 "are not from raw counts.")
    else:
        s.append("Original (uncorrected) counts were summed; SPLIT-corrected counts are not used for count "
                 "models.")
    return s, refs


def _pm_proximity(r):
    """Written from what the run recorded: the estimator the p-values are of (near − far,
    or the slope), the adjustment behind padj (empirical, Benjamini-Hochberg or none),
    whether a density covariate was used, and whether the test is across units."""
    H, P = r["headline"], r["params"]
    info = H.get("info") if isinstance(H.get("info"), dict) else {}
    est = str(H.get("estimator") or info.get("estimator") or "")
    fdr = str(H.get("fdr") or info.get("fdr") or "")
    mode = str(H.get("mode") or info.get("mode") or "")
    # the density-covariate type the run USED (the result's "source": the type "auto" was
    # resolved to, None when no covariate was used), not the request, which may be "auto"
    src_unrecorded = False
    if "source" in H:
        src = H.get("source") if H.get("source") not in ("", "none", "auto") else None
    else:
        src = P.get("source") if P.get("source") not in ("", "none", "auto") else None
        src_unrecorded = P.get("source") == "auto"
    rp = P.get("resolved") if isinstance(P.get("resolved"), dict) else {}
    near, far = rp.get("near", P.get("near")), rp.get("far", P.get("far"))
    lr = rp.get("local_r", 50)
    s = [f"Proximity differential expression: genes of the cells of type '{_pm_v(P, 'target')}' that "
         f"change near cells of the anchor type '{_pm_v(P, 'anchor')}', on {_pm_seg(r)}."]
    resid = ("Per sample, log1p(counts per 10,000) of every target cell was residualised (ordinary least "
             "squares) on log1p(library size)")
    if src:
        resid += (f" and on quantile bins of the local fraction of '{src}' cells within {_pm_num(lr)} µm "
                  f"(the density covariate)")
    elif src_unrecorded:
        resid += ("; a density-covariate cell type was to be chosen automatically, and which one (or none) "
                  "was used was not recorded for this run")
    else:
        resid += "; no density-covariate cell type was used"
    s.append(resid + ".")
    nf = ("the mean residual of the target cells whose nearest anchor cell is within the near distance"
          + (f" ({_pm_num(near)} µm)" if near is not None else "")
          + " minus that of the target cells with no anchor cell closer than the far distance"
          + (f" ({_pm_num(far)} µm)" if far is not None else ""))
    sl = "the ordinary-least-squares slope of the residual on −log(d + 1), d the distance to the nearest anchor cell"
    if est == "slope":
        s.append(f"The statistic the p-values and the FDR are of is {sl} (too few random anchor sets left "
                 f"far cells for the near − far difference, which is reported beside it: {nf}).")
    elif est == "nearfar":
        s.append(f"The effect of a gene is {nf}; the slope ({sl}) is reported beside it.")
    else:
        s.append(f"Two statistics were computed per gene: {nf}; and {sl}. Which of the two the p-values "
                 f"are of was not recorded for this run (see its status line).")
    rnd = ("Matched random anchors (sets of as many non-anchor cells, drawn within the same compartment x "
           "edge-distance stratum) went through the same computation")
    refs = []
    if fdr == "empirical":
        s.append(rnd + "; the empirical FDR of a gene is the number of genes the random sets yield at its "
                 "threshold divided by the number observed.")
    elif fdr == "bh":
        s.append(rnd + ", but too few of them gave an effect in every unit for an empirical FDR: the FDR "
                 "reported is Benjamini-Hochberg over the sign-flip p-values.")
        refs.append("bh")
    elif fdr == "none":
        s.append(rnd + "; no FDR was computed for this run (no usable random anchor set).")
    else:
        s.append(rnd + "; the adjustment behind the reported FDR (empirical from the random sets, or "
                 "Benjamini-Hochberg) was not recorded for this run (see its status line).")
    if mode == "cohort":
        s.append("The p-value is the exact sign-flip test of the per-unit effects across the independent "
                 "units (donors, or samples).")
    elif mode == "single":
        s.append("With fewer than 3 independent units the p-value is the share of random anchor sets at "
                 "least as extreme, which describes these sections only.")
    else:
        s.append("With at least 3 independent units (donors, or samples) the p-value is the exact sign-flip "
                 "test of the per-unit effects; with fewer it is the share of random anchor sets at least as "
                 "extreme, which describes these sections only.")
    s.append("A gradient with distance is an association with position, not an effect of the anchor type.")
    return s, refs


def _pm_contact(r):
    H, P = r["headline"], r["params"]
    by_donor = bool(H.get("by_donor"))
    mixed = by_donor and str(H.get("unit_word") or "donors") != "donors"
    unit = ("independent unit (a donor, or a sample without a donor label)" if mixed else
            "donor" if by_donor else "sample")
    return ([f"Contact-group pseudobulk differential expression: cells of type '{_pm_v(P, 'target')}' with a "
             f"cell of type '{_pm_v(P, 'anchor')}' within the radius were compared with those without, on "
             f"{_pm_seg(r)}.",
             "Within each sample the comparator cells were weighted so that their mix of compartment x "
             "edge-distance bin x density quintile equals that of the contact cells. Weighted counts were "
             + ("summed per independent unit and group (the sections of one donor pooled, a sample without a "
                "donor label on its own; rounded to whole counts) and fitted with ~ unit + group" if mixed else
                "summed per donor and group (the sections of one donor pooled; rounded to whole counts) and "
                "fitted with ~ donor + group" if by_donor else
                "summed per sample and group (rounded to whole counts) and fitted with ~ sample + group")
             + f" in R (DESeq2, or edgeR when DESeq2 is not installed); the unit of the p-values is the {unit}.",
             "The contact / comparator labels were also permuted within sample x stratum and the fit "
             "repeated, to compare the number of genes passing with the observed number.",
             ("The count matrix of this data does NOT hold raw counts (non-integer values; noninteger_counts = "
              "round_sums was set when it was loaded): the weighted sums of these values were rounded, and the "
              "results are not from raw counts." if H.get("counts_rounded") else
              "Original (uncorrected) counts were summed; SPLIT-corrected counts are not used for count "
              "models.")],
            ["deseq2", "edger", "bh"])


def _pm_geneset(r):
    P = r["params"]
    return ([f"Gene-set score '{_pm_v(P, 'gene_set')}': per cell, the mean of the z-scored log-normalised "
             f"expression of the set minus the mean of a control set drawn from the same expression bins "
             f"(25 bins, 50 control genes per bin, fixed seed), as in Seurat AddModuleScore / scanpy "
             f"score_genes; computed on {_pm_seg(r)}. No test is attached to the score."], ["modulescore"])


def _pm_density(r):
    return (["Transcript density: transcripts (not cells) per bin over the whole tissue, read from the "
             "transcript file with quality Q >= 20 and gene features only. It does not depend on the "
             "segmentation or on a cell subset. No test is attached."], [])


def _pm_composition(r):
    H, P = r["headline"], r["params"]
    tr = {"logit": "logit (count + 0.5 pseudocount)", "asin": "arcsine square root"}.get(
        str(P.get("transform")), str(P.get("transform")))
    s = [f"Composition between conditions: per-sample proportions of each {_pm_v(H, 'category', 'category')} "
         f"on {_pm_seg(r)}, transformed with the {tr} transformation."]
    if str(P.get("test")) == "exact":
        s.append("Test 'exact (sample-level)': the transformed proportions were tested without a variance "
                 "model - the paired sign-flip test over donors when every included sample has a donor and at "
                 "least 2 donors occur in both conditions, otherwise the label-permutation test over the "
                 "independent units, enumerating every assignment when feasible; Benjamini-Hochberg FDR over "
                 "the categories.")
    else:
        s.append("Propeller-style test: a linear model per category (~ condition, or ~ batch + condition; "
                 "~ donor + condition when the design is paired) with a moderated t-test (limma-style "
                 "empirical Bayes shrinkage of the variances, implemented in the app, not by calling limma or "
                 "propeller); Welch's t-test is given beside it; Benjamini-Hochberg FDR over the categories.")
    s.append("Proportions are compositional (they sum to 1 per sample), so the tests of the categories are "
             "not independent. With fewer than 3 units (samples, or donors) in a condition no test is run.")
    return s, ["propeller", "limma", "bh"]


def _pm_moran(r):
    H, P = r["headline"], r["params"]
    return ([f"Spatially variable genes: Moran's I of every measured gene on log1p(counts per 10,000), with a "
             f"k-nearest-neighbour graph of the cell centroids (k = {_pm_num(_pm_v(H, 'k', _pm_v(P, 'k')))}, "
             f"row-standardised weights), computed per sample on {_pm_seg(r)}.",
             "z-scores and two-sided p-values come from the analytic mean and variance of I under the "
             "normality assumption (Cliff and Ord); Benjamini-Hochberg FDR over the genes of a sample.",
             "The unit of the p-values is the cell: the cells of a sample were treated as independent "
             "observations. With tens of thousands of cells almost every expressed gene is significant, so "
             "genes should be ranked by I, not by the p-value."], ["moran", "cliff_ord", "bh"])


def _pm_lr(r):
    H, P = r["headline"], r["params"]
    return ([f"Ligand-receptor scores between cell groups on {_pm_seg(r)}: the score of a pair for sender "
             f"type S and receiver type R is the mean of the mean log-normalised expression of the ligand in "
             f"S and of the receptor in R (CellPhoneDB-style), computed only for type pairs that are spatial "
             f"neighbours in the k-nearest-neighbour graph of the centroids (k = {_pm_num(_pm_v(P, 'k'))}) "
             f"and only when ligand and receptor are detected in their type. The list of pairs was supplied "
             f"by the user; no database is bundled.",
             f"p-values: the cell-type labels were permuted within each sample "
             f"({_pm_num(_pm_v(P, 'n_perm'))} permutations, seed {_pm_num(_pm_v(P, 'seed'))}); "
             f"p = (1 + permutations with a score >= observed) / (1 + permutations), so the smallest possible "
             f"p is 1 / (1 + permutations); Benjamini-Hochberg FDR over all tested combinations.",
             "The unit of the p-values is the cell (labels exchangeable within a sample); it is not a test "
             "across samples or donors. A significant score is expression-based inference, not evidence of "
             "signalling."], ["cellphonedb", "bh"])


def _pm_lib_shared(r) -> list:
    """Sentence for a pooled analysis whose samples have different panels."""
    n = (r.get("headline") or {}).get("lib_shared_genes")
    return ([f"The pooled samples have different panels: the depth (library size) of a cell is its total "
             f"over the {_pm_num(n)} genes they share."] if n else [])


def _pm_lrdecomp(r):
    return (["Ligand-receptor exposure decomposition: for each sender type, receiver type, ligand and "
             "receptor, the exposure E (sender contacts per receiver x the contact-weighted product of "
             "receptor and ligand rates) was written as the exact product of five terms - contacts per "
             "receiver (K), the type-wide receptor rate (R), the type-wide ligand rate (L), tile co-location "
             "(gamma) and contact preference (pi) - so that the log2 ratio of E between two conditions is the "
             "sum of the log2 ratios of the terms.",
             "Tests are across the independent units of the sample design (donors, or samples): the exact "
             "sign-flip test of the per-donor log2 ratios when paired, exact label permutation otherwise; "
             "with one condition, one sample or fewer than 3 units per condition the decomposition is "
             "descriptive, without p-values."] + _pm_lib_shared(r) + [
             "Like every expression-based score this is not evidence of signalling."], [])


def _pm_compartments(r):
    P = r["params"]
    q = P.get("resolved") or P
    H = r.get("headline") or {}
    if "dominant" not in H:
        # a record without the result's field: say what a core is, not that one was defined
        core_txt = (f"Where cell-type labels were used, the dense core of the dominant cell type is: at least "
                    f"{_pm_num(_pm_v(q, 'min_nb'))} other cells within that radius, at least "
                    f"{_pm_num(_pm_v(q, 'min_frac'))} of them of the type, and further than "
                    f"{_pm_num(_pm_v(q, 'min_edge'))} µm from the edge (see the status line for whether a core "
                    f"was defined).")
    elif H.get("dominant"):
        core_txt = (f"Dense core of the cell type '{H['dominant']}'"
                    + (" (the most frequent type of the view)" if P.get("dominant") in (None, "", "auto") else "")
                    + f": at least {_pm_num(_pm_v(q, 'min_nb'))} other cells within "
                    f"that radius, at least {_pm_num(_pm_v(q, 'min_frac'))} of them of the type, and further than "
                    f"{_pm_num(_pm_v(q, 'min_edge'))} µm from the edge.")
    else:
        core_txt = (f"No cell-type labels were used, so no core was defined: cells within "
                    f"{_pm_num(_pm_v(q, 'min_edge'))} µm of the tissue edge are 'edge', all others 'other'.")
    return ([f"Tissue compartments, per sample, on the cells of {_pm_seg(r)}: cell positions were rasterised "
             f"into {_pm_num(_pm_v(q, 'bin_um'))} µm bins, occupied bins taken as tissue (morphological closing "
             f"x {_pm_num(_pm_v(q, 'close_bins'))}; interior holes of at most {_pm_num(_pm_v(q, 'hole_max_bins'))} "
             f"bins filled) and a Euclidean distance transform gave each cell its distance to the tissue edge. "
             f"Local density is the number of other cells within {_pm_num(_pm_v(q, 'radius'))} µm.",
             core_txt + " No test is attached; the columns describe where a cell sits, not what it is."], [])


def _pm_marker_rule(H) -> "str | None":
    """The marker rule a run recorded ("user", "specific", "fallback"), None when it did not."""
    mi = H.get("marker_info")
    rule = H.get("marker_rule") or (mi.get("rule") if isinstance(mi, dict) else None)
    return str(rule) if rule in ("user", "specific", "fallback") else None


def _pm_markers_sentence(H, of: str) -> str:
    """How the markers of a run were chosen, from the recorded rule: listed by the user,
    the specificity rule, or its fallback (never the specificity claim for a fallback set)."""
    spec = ("at least 50 counts per million in the type and at least 10x the level of every other type")
    rule = _pm_marker_rule(H)
    if rule == "user":
        return "Markers were the genes listed by the user."
    if rule == "specific":
        return f"Markers were derived from the annotation: the most specific genes of {of} ({spec})."
    if rule == "fallback":
        return (f"Markers were derived from the annotation, but fewer than 3 genes of {of} met the specificity "
                f"rule ({spec}), so the genes with the highest specificity ratio among those at 50 counts per "
                f"million or more were used whatever their ratio: these markers are not exclusive to the type "
                f"and what is built on them is an upper bound.")
    return (f"Markers were the most specific genes of {of} in the annotation ({spec}; when fewer than 3 genes "
            f"meet this, the genes with the highest ratio) or the genes listed by the user; which applied "
            f"was not recorded for this run (see its status line).")


def _pm_spillover(r):
    H = r.get("headline") or {}
    kind = H.get("reference_kind")
    ref = ("the median of that share inside source cells" if kind == "median" else
           "the mean of that share inside source cells (its median was 0 - the markers are sparse - so the "
           "mean was the reference and the per-cell score is coarse)" if kind == "mean" else
           "a reference share inside source cells (its median, or its mean when the median is 0; which one "
           "was not recorded for this run)")
    return ([f"Spillover score and gene ownership on {_pm_seg(r)}: the spillover score of a cell for a source "
             "cell type is the share of the cell's counts in the source type's marker genes divided by "
             + ref + ". " + _pm_markers_sentence(H, "the source type"),
             "Built-in validity checks: the marker share of non-source cells must fall with the distance to "
             "the nearest source cell and rise with the fraction of source cells nearby; when a check fails "
             "the score is reported as not usable as a contamination mask.",
             "Gene ownership: per gene, the cell type with the highest rate (sum of counts / sum of depth) "
             "and every type at 20 % of it or more (threshold as recorded in the parameters)."]
            + _pm_lib_shared(r), [])


def _pm_segqc(r):
    return (["Segmentation QC: one row of metrics per segmentation source (Xenium and every cached "
             "re-segmentation run of the primary sample) and counts mode. Identity marker sets (the most "
             "specific genes of every cell type, derived from the Xenium segmentation's original counts and "
             "annotation, or supplied) were applied to every source: exclusion-rule failure, marker "
             "co-expression, spillover slope against the number of source-type neighbours within the radius, "
             "own-identity retention for corrected counts, and the share of gene transcripts assigned to "
             "cells.",
             "A run without its own annotation received the Xenium label of the nearest Xenium centroid "
             "(within 10 µm), which favours segmentations close to Xenium's. The metrics are descriptive; "
             "no test is attached."], [])


def _pm_coloc(r):
    P = r["params"]
    return ([f"Stratified co-localisation on {_pm_seg(r)}: for every ordered pair of cell types (focal A, "
             f"target B), the number of B cells within {_pm_num(_pm_v(P, 'radius'))} µm of each A cell (O) was "
             f"compared with the mean of that count around pool cells (labelled cells that are neither A nor "
             f"B) of the same stratum - sample x compartment x edge-distance bin x local-density quintile; "
             f"effect per sample = log2((sum O + 0.5) / (sum E + 0.5)).",
             f"Within-sample p-value: the A label was permuted among A and pool cells within each stratum "
             f"({_pm_num(_pm_v(P, 'n_perm'))} permutations, seed {_pm_num(_pm_v(P, 'seed'))}); "
             f"p = (1 + permutations at least as extreme) / (1 + permutations). That p treats the cells of "
             f"one section as exchangeable and says nothing about other sections or donors.",
             "With at least 3 independent units (samples, or donors) the effect was tested across them with "
             "the exact sign-flip test and, between conditions, with the paired sign-flip or "
             "label-permutation test of the sample design."], ["bh"])


def _pm_aggregates(r):
    return ([f"Aggregates on {_pm_seg(r)}: connected groups of cells where the chosen class of cell types is "
             "locally denser than label permutations allow. Per sample, the class cells within the radius "
             "of every cell were counted and the class label permuted among the cells of the same stratum "
             "(compartment x edge-distance bin x local-density quintile); hotspot cells "
             "(p = (k + 1) / (permutations + 1) at most alpha) joined by Delaunay edges no longer than the "
             "link length form components, and components with at least the minimum number of class cells "
             "are the aggregates.",
             "The per-cell p-values are a detection threshold with the cells of one section as their unit, "
             "not a test across samples. Between conditions, aggregates per mm² and area fraction were "
             "compared with the exact sample-level test when the design has at least 3 units per condition."],
            [])


def _pm_segfree(r):
    P = r["params"]
    return ([f"Segmentation-free abundance of '{_pm_v(P, 'cell_type')}': marker transcripts per mm² of "
             "tissue, counted from the transcript file (quality Q >= 20, gene features only), not from the "
             "cells. " + _pm_markers_sentence(r.get("headline") or {}, "the cell type")
             + " The tissue area was taken from a square grid at every tile size of the sweep.",
             "Concentration: Gini coefficient of the marker count over the tissue tiles and the share of "
             "the count in the top 10 % of tiles, each minus its mean under a null that redistributes the "
             f"same number of transcripts over the same tiles ({_pm_num(_pm_v(P, 'n_null'))} redistributions, "
             f"seed {_pm_num(_pm_v(P, 'seed'))}); one-sided p = (1 + redistributions with a Gini >= observed) "
             "/ (1 + redistributions). Its unit is the tiles of one sample, so it describes that section "
             "only. Transcript density does not tell cell number from expression per cell."], [])


def _pm_banksy(r):
    return ([f"Banksy-style spatial domains on {_pm_seg(r)}: a native implementation that follows the "
             "construction of the published BANKSY method - each cell's own expression, the Gaussian-weighted "
             "mean expression of its k nearest cells in space and the azimuthal Gabor filter of its 2k "
             "nearest cells, each block z-scaled per gene and mixed by lambda - followed by PCA, a "
             "nearest-neighbour graph in PC space and Leiden clustering (KMeans when Leiden is not "
             "installed; the record's status line says which).",
             "It is not the Banksy package and has not been validated against the reference "
             "implementation. The clusters are an exploratory partition, not a test."],
            ["banksy", "leiden"])


def _pm_joint(r):
    H = r["headline"]
    s = [f"Joint clustering across samples on {_pm_seg(r)}: all cells of the view were put in one matrix over "
         "the panel genes every sample has (counts per 10,000, log1p); genes were ranked by dispersion within "
         "each sample and the top genes kept; each gene was scaled, centred and capped at 10; PCA over all "
         "cells; then one neighbour graph, clustering (Leiden when installed) and UMAP on all cells."]
    refs = ["leiden", "umap"]
    if H.get("harmony"):
        s.append("The principal components were corrected with Harmony (harmonypy) for the integration "
                 "variable recorded in the parameters. Integration removes differences between the levels of "
                 "that variable; when the condition is confounded with it, real differences between "
                 "conditions can be removed as well.")
        refs.append("harmony")
    else:
        s.append("No integration was applied unless the parameters below say so.")
    s.append("Mixing diagnostics (entropy of the integration variable per cluster, a simplified LISI) were "
             "computed before and after the correction. No test is attached.")
    return s, refs


def _pm_txfactor(r):
    P = r["params"]
    m = str(P.get("method") or "nmf")
    how = {"nmf": "NMF with the Kullback-Leibler loss (multiplicative updates)", "lda": "online LDA",
           "reference": "the mean expression profiles of the annotated cell types (not fitted)"}.get(m, m)
    return (["Transcript factors: a simplified, segmentation-free transcript-binning factorisation in the "
             "spirit of FICTURE (not that tool: no anchors, no pixel-level decoding). Gene transcripts "
             "(Q >= 20, gene features) were counted per bin of a square or hexagonal grid, each sample on "
             f"its own grid; the gene loadings were {how}, fitted on a random subsample of the bins (fixed "
             "seed), and every bin was then projected onto the loadings.",
             "Factors come from one random initialisation / subsample. No test is attached."], ["ficture"])


def _pm_annotation(r):
    P, H = r["params"], r["headline"]
    m = str(P.get("method"))
    n = (f" {_pm_num(_pm_v(H, 'n_cells_labelled'))} cells received one of "
         f"{_pm_num(_pm_v(H, 'n_cell_types'))} labels.")
    if m == "celltypist":
        return ([f"Cell types of {_pm_seg(r)} were predicted with CellTypist (model {_pm_v(P, 'model')}) on "
                 f"counts normalised to 10,000 per cell and log1p-transformed, with majority voting." + n],
                ["celltypist"])
    if m == "rctd":
        return ([f"Cell types of {_pm_seg(r)} were assigned with RCTD (spacexr; mode {_pm_v(P, 'mode')}; "
                 f"UMI_min {_pm_num(_pm_v(P, 'umi_min'))}, UMI_min_sigma {_pm_num(_pm_v(P, 'umi_min_sigma'))}) "
                 f"against the reference '{_pm_v(P, 'reference')}' (label column {_pm_v(P, 'label_col')})." + n],
                ["rctd", "seurat"])
    return ([f"Cell types of {_pm_seg(r)} were transferred from the reference '{_pm_v(P, 'reference')}' (label "
             f"column {_pm_v(P, 'label_col')}; read through Seurat in R): both data sets were normalised to "
             f"10,000 counts per cell and log1p-transformed, the cells were projected into the reference's "
             f"PCA space and each cell received the majority label of its 50 nearest reference cells "
             f"(cosine distance)." + n], ["seurat"])


def _pm_reseg(tool):
    def f(r):
        P, H = r["params"], r["headline"]
        ver = ((r.get("software") or {}).get("tools") or {}).get(tool)
        name = {"baysor": "Baysor", "proseg": "Proseg"}[tool]
        if r.get("cache") == "cache" and "run" in P:
            sp = (r.get("data") or {}).get("seg_params")
            return ([f"A {name} segmentation computed in an earlier session (run {_pm_v(P, 'run')}) was loaded "
                     f"from the cache ({_pm_num(_pm_v(H, 'n_cells'))} cells)."
                     + (f" Parameters stored with the run: {_pm_params(sp)}." if isinstance(sp, dict) else "")],
                    [tool])
        return ([f"Cells were re-segmented from the transcript coordinates with {name}"
                 + (f" ({ver})" if ver else " (version not recorded)")
                 + (", run per transcript patch (sopa) and merged" if P.get("use_patches") else "")
                 + f"; the run produced {_pm_num(_pm_v(H, 'n_cells'))} cells. A cell x gene count matrix was "
                   f"built from the tool's transcript assignments."], [tool, "sopa", "spatialdata"])
    return f


def _pm_imgseg(r):
    P, H = r["params"], r["headline"]
    if r.get("cache") == "cache" and "run" in P:
        return ([f"An image-based segmentation computed in an earlier session (run {_pm_v(P, 'run')}) was "
                 f"loaded from the cache ({_pm_num(_pm_v(H, 'n_cells'))} cells)."], ["sopa"])
    # the recorded backend of a Cellpose run is "cellpose:<model>"
    _kind, _, _model = str(P.get("backend")).partition(":")
    cp = _kind == "cellpose"
    return (["Cells were segmented on the morphology image, patch by patch through sopa (cells cut by a "
             "patch edge merged), with "
             + ((f"Cellpose (model '{_model}', local weights)" if _model else "Cellpose (local weights)") if cp else
                "the built-in classical backend (Gaussian smoothing, Otsu threshold per patch, "
                "distance-transform watershed, optional fixed-distance expansion of every nucleus - a simple "
                "baseline, not a state-of-the-art segmenter)")
             + f". Gene transcripts (Q >= {_pm_num(_pm_v(P, 'qv_min'))}, gene features only) were assigned to "
               f"the polygon they fall in, in 2D: {_pm_num(_pm_v(H, 'n_transcripts_assigned'))} of "
               f"{_pm_num(_pm_v(H, 'n_transcripts'))} transcripts, {_pm_num(_pm_v(H, 'n_cells'))} cells."],
            ["sopa", "spatialdata"] + (["cellpose"] if cp else []))


def _pm_patches(r):
    return ([f"{r.get('title')}: preparation step of the re-segmentation (SpatialData store of the dataset "
             f"through sopa, tissue region of interest, transcript patches)."], ["sopa", "spatialdata"])


def _pm_split(r):
    P, H = r["params"], r["headline"]
    return ([f"Counts of {_pm_seg(r)} were corrected with SPLIT: RCTD (spacexr) in doublet mode against the "
             f"reference '{_pm_v(P, 'reference')}' (label column {_pm_v(P, 'label_col')}; UMI_min "
             f"{_pm_num(_pm_v(P, 'min_umi'))}, UMI_min_sigma {_pm_num(_pm_v(P, 'min_umi_sigma'))}), then "
             f"SPLIT::purify() (purification of singlets: {_pm_num(bool(P.get('purify_singlets')))}). "
             f"{_pm_num(_pm_v(H, 'n_returned'))} of {_pm_num(_pm_v(H, 'n_cells'))} submitted cells were "
             f"returned with corrected counts; cells without corrected counts are left out of analyses in "
             f"corrected-counts mode."], ["split", "rctd"])


def _pm_spage(r):
    P = r["params"]
    return ([f"Unmeasured genes were imputed with SpaGE from the reference '{_pm_v(P, 'reference')}' "
             f"({_pm_num(_pm_v(P, 'n_pv'))} principal vectors requested; 50 nearest reference cells, cosine "
             f"distance) on the original counts of {_pm_seg(r)}. Imputed values are predictions, not counts; "
             f"they are not used in the count-based tests."], ["spage"])


def _pm_umap(r):
    P, H = r.get("params") or {}, r.get("headline") or {}
    how = str(P.get("method") or "")
    if how == "t-SNE":
        emb = "t-SNE (umap-learn was not available)"
    elif how == "UMAP":
        emb = (f"UMAP ({_pm_num(P.get('n_neighbors', 15))} neighbours, min_dist "
               f"{_pm_num(P.get('min_dist', 0.1))})")
    else:
        emb = "UMAP (15 neighbours, min_dist 0.1; t-SNE when UMAP is not available)"
    ncomp = f"{_pm_num(P['n_components'])} components" if P.get("n_components") else "up to 50 components"
    cells = (f" of {_pm_num(H['n_cells'])} cells" if isinstance(H.get("n_cells"), int) else "")
    return ([f"Embedding{cells} of {_pm_seg(r)}: counts per 10,000, log1p, truncated SVD ({ncomp}) and "
             f"{emb}; clusters of a re-segmentation run come from KMeans on the SVD components. "
             f"Exploratory; no test."], ["umap"] if how != "t-SNE" else [])


def _pm_spatial(r):
    P = r["params"]
    return ([f"Spatial analysis of the annotated cell types: a {_pm_num(_pm_v(P, 'n_neighbors'))}-nearest-"
             "neighbour graph of the cell centroids; co-occurrence = log2 of the observed number of directed "
             "neighbour pairs of two types over the number expected from the type frequencies; neighbourhood "
             "enrichment = log2 of P(neighbour is b | cell is a) over the frequency of b. Descriptive; no "
             "test and no permutation null (the stratified co-localisation analysis provides those)."], [])


def _pm_niches(r):
    P = r["params"]
    feat = str(P.get("features") or "")
    if feat.startswith("normalised spatial coordinates"):
        # no labels: the job clustered the coordinates, not a neighbourhood composition
        return ([f"Spatial blocks: no cell-type labels were available, so the normalised cell centroid "
                 f"coordinates were clustered with KMeans into {_pm_num(_pm_v(P, 'n_clusters'))} groups "
                 f"(random_state 42, 10 initialisations). These are contiguous spatial blocks, not niches of "
                 f"the neighbourhood composition. Exploratory; no test."], [])
    if not feat.startswith("cell-type composition"):
        # an older record that does not say what was clustered
        return ([f"Spatial niches: KMeans into {_pm_num(_pm_v(P, 'n_clusters'))} groups (random_state 42, 10 "
                 f"initialisations) on the features named in the status line of the run (the cell-type "
                 f"composition of the {_pm_num(_pm_v(P, 'n_neighbors'))} nearest neighbours when cell-type "
                 f"labels existed, otherwise the cell coordinates). Exploratory; no test."], [])
    return ([f"Spatial niches: for every cell the cell-type composition of its "
             f"{_pm_num(_pm_v(P, 'n_neighbors'))} nearest neighbours was clustered with KMeans into "
             f"{_pm_num(_pm_v(P, 'n_clusters'))} niches (random_state 42, 10 initialisations). Exploratory; "
             f"no test."], [])


def _pm_h5ad(r):
    P = r["params"]
    H = r.get("headline") or {}
    s = [f"The AnnData file '{_pm_v(P, 'file')}' was converted once into one store per section with the "
         "column mapping recorded in the parameters; the values of its count matrix were taken as given."]
    mode = str(H.get("noninteger_counts") or ((P.get("mapping") or {}).get("noninteger_counts")
                                              if isinstance(P.get("mapping"), dict) else "") or "")
    if H.get("counts_integer") is False:
        s.append(f"The count matrix is NOT integer-valued ({_pm_num(H.get('n_noninteger', 0))} stored values "
                 f"are not whole numbers, {_pm_num(H.get('n_negative', 0))} are negative): it does not hold "
                 f"raw counts.")
        if mode == "round_sums":
            s.append("The mapping set noninteger_counts = round_sums: the sample-level count models "
                     "(pseudobulk DESeq2 / edgeR, contact-group pseudobulk) were allowed to sum these values "
                     "per sample or donor and round the sums to whole numbers. Results obtained this way are "
                     "not from raw counts and say so in their own record.")
        else:
            s.append("The sample-level count models (pseudobulk DESeq2 / edgeR) refuse this matrix "
                     "(noninteger_counts = refuse, the default).")
    elif mode == "round_sums":
        s.append("The mapping set noninteger_counts = round_sums; the count matrix is integer-valued, so "
                 "the setting had no effect.")
    return s, ["anndata"]


def _pm_save(r):
    P = r["params"]
    return ([f"The analysis state of {_pm_seg(r)} was written to the SpatialData store "
             f"'{_pm_v(P, 'file')}'."], ["spatialdata"])


def _pm_cellnest(r):
    return (["Results of CellNEST computed outside the app were loaded for display; the app does not run "
             "CellNEST."], ["cellnest"])


def _pm_session(r):
    P = r["params"]
    what = ("control values, the sample design, regions of interest, gene sets, saved groups and subset "
            "filters; no results")
    if P.get("action") == "restore":
        part = ("" if P.get("mode") != "independent" else
                " The session had been saved for other data, so only the settings that do not depend on the "
                "data were applied.")
        return ([f"The settings of a saved session ({what}) were restored from the {_pm_v(P, 'from')} "
                 f"(saved {_pm_v(P, 'saved_at')}); {_pm_v(P, 'n_controls_set')} control values were set and "
                 f"{_pm_v(P, 'n_missing')} values that named data no longer present were left out.{part} "
                 f"Analyses run afterwards are recorded on their own."], [])
    return ([f"The settings of the session ({what}) were saved as a JSON file ({_pm_v(P, 'to')})."], [])


_PROV_TEMPLATES = {
    "de": _pm_de, "pseudobulk": _pm_pseudobulk, "proximity": _pm_proximity, "contact": _pm_contact,
    "geneset": _pm_geneset, "density": _pm_density, "composition": _pm_composition, "moran": _pm_moran,
    "lr": _pm_lr, "lrdecomp": _pm_lrdecomp, "compartments": _pm_compartments, "spillover": _pm_spillover,
    "segqc": _pm_segqc, "coloc": _pm_coloc, "aggregates": _pm_aggregates, "segfree": _pm_segfree,
    "banksy": _pm_banksy, "joint": _pm_joint, "txfactor": _pm_txfactor, "annotation": _pm_annotation,
    "baysor": _pm_reseg("baysor"), "proseg": _pm_reseg("proseg"), "imgseg": _pm_imgseg,
    "patches": _pm_patches, "split": _pm_split, "spage": _pm_spage, "umap": _pm_umap,
    "spatial": _pm_spatial, "niches": _pm_niches, "h5ad": _pm_h5ad, "save": _pm_save,
    "cellnest": _pm_cellnest, "session": _pm_session,
}
# order of the sections of the methods text
_PROV_ORDER = ["h5ad", "patches", "baysor", "proseg", "imgseg", "umap", "annotation", "split", "spage",
               "segqc", "compartments", "spillover", "de", "pseudobulk", "proximity", "contact",
               "geneset", "density", "moran", "composition", "spatial", "niches", "coloc", "aggregates",
               "segfree", "txfactor", "banksy", "joint", "lr", "lrdecomp", "cellnest", "save",
               "session"]


def _pm_software(sw: dict) -> str:
    sw = sw or {}
    parts = [f"Python {sw.get('python', '?')}"]
    parts += [f"{k} {v}" for k, v in sw.items() if k not in ("python", "app_commit", "r", "tools")]
    if sw.get("r"):
        parts += [("R " + str(v)) if k == "R" else f"{k} {v} (R)" for k, v in sw["r"].items()]
    if sw.get("tools"):
        parts += [f"{k}: {v}" for k, v in sw["tools"].items()]
    return "; ".join(parts)


def _pm_variant(r) -> tuple:
    """Records with the same variant describe the same analysis; the newest one is written up."""
    P = r.get("params") or {}
    d = r.get("data") or {}
    extra = {"annotation": P.get("method"), "geneset": P.get("gene_set"), "patches": r.get("title"),
             "proximity": (P.get("target"), P.get("anchor")), "contact": (P.get("target"), P.get("anchor")),
             "pseudobulk": P.get("population"), "segfree": P.get("cell_type"),
             "de": (str(P.get("group_a")), str(P.get("group_b"))),
             "composition": (P.get("proportions_of"), P.get("test"))}.get(r.get("kind"))
    return (r.get("kind"), d.get("seg_source"), (d.get("counts") or {}).get("used"), json.dumps(extra, default=str))


_PM_RUN_KINDS = ("baysor", "proseg", "imgseg")


def _pm_is_reload_of(new, old) -> bool:
    """*new* (same variant as *old*, written later) only reports that the result *old*
    computed was read back from the cache: a cached segmentation run loaded again (the
    variant names the run), or a cache hit of an analysis with the same parameters."""
    if new.get("cache") != "cache" or old.get("cache") == "cache":
        return False
    if new.get("kind") in _PM_RUN_KINDS:
        return True
    strip = lambda p: {k: v for k, v in (p or {}).items() if k != "resolved"}
    return strip(new.get("params")) == strip(old.get("params"))


def _prov_methods_md(records=None, latest_only: bool = True) -> str:
    """Methods text (Markdown) written from the provenance records: a data paragraph, one
    section per analysis from its template (method, parameters, unit and n as the app
    stated them, multiple-testing procedure, caveats the app raised, software), the
    references of the published methods, and a table of every record."""
    recs = _prov_read() if records is None else list(records)
    out = ["# Methods", ""]
    if not recs:
        return "\n".join(out + ["No analysis has been recorded for this dataset.", ""])
    newest = recs[-1]
    commit = (newest.get("software") or {}).get("app_commit") or "unknown"
    out += [f"*Generated by Xenium Visualizer (commit {commit}) from {len(recs)} provenance record(s) on "
            f"{_dt.datetime.now().strftime('%Y-%m-%d')}. "
            + ("Each section describes the newest run of an analysis; every run is listed in the table at "
               "the end. " if latest_only else "Every recorded run has its own section. ")
            + "Sentences state what the app computed; the line 'As reported by the app' is the analysis' own "
              "status line, verbatim. Check each paragraph against your study before use.*", ""]
    d = newest.get("data") or {}
    smp = d.get("samples") or []
    out += ["## Data", ""]
    out.append(f"{len(smp)} sample(s) were loaded ("
               + "; ".join(f"{s['label']}: {s['name']}, {s['n_cells']:,} cells" for s in smp[:30])
               + ("; …" if len(smp) > 30 else "")
               + f"), {d.get('n_cells', 0):,} cells and {d.get('n_genes', 0):,} genes in the primary sample's "
                 f"panel." + (f" A cell subset was active in the newest record ({d['subset'].get('n_cells', '?'):,} "
                              f"of {d['subset'].get('of', '?'):,} cells; filters: {d['subset'].get('filters')})."
                              if isinstance(d.get("subset"), dict) and isinstance(d["subset"].get("n_cells"), int)
                              else ""))
    out.append("")
    chosen, reloaded = {}, {}
    if latest_only:
        for r in recs:
            v = _pm_variant(r)
            cur = chosen.get(v)
            if cur is not None and _pm_is_reload_of(r, cur):
                # the same result read back from the cache: the section stays that of the
                # run that computed it (its parameters, tool versions and software)
                reloaded[cur.get("id")] = r
                continue
            chosen[v] = r
        todo = list(chosen.values())
    else:
        todo = list(recs)
    rank = {k: i for i, k in enumerate(_PROV_ORDER)}
    todo.sort(key=lambda r: (rank.get(r.get("kind"), len(rank)), r.get("time", "")))
    refs_used = []
    for r in todo:
        fn = _PROV_TEMPLATES.get(r.get("kind"))
        try:
            sentences, refs = fn(r) if fn else ([f"{r.get('title')}: no template; see the status line."], [])
        except Exception as exc:                       # a record of an older schema: never fail the export
            sentences, refs = [f"{r.get('title')} (the template could not be filled: {exc})."], []
        for k in refs:
            if k not in refs_used:
                refs_used.append(k)
        out += [f"## {r.get('title')}", ""]
        cite = " ".join(f"[{refs_used.index(k) + 1}]" for k in refs)
        out.append(" ".join(sentences) + (f" {cite}" if cite else ""))
        out.append("")
        extra = [x for x in (_pm_labels(r), _pm_counts(r), _pm_design(r)) if x]
        if extra:
            out += [" ".join(extra), ""]
        prm = _pm_params(r.get("params") or {})
        if prm:
            out += [f"Parameters: {prm}.", ""]
        if r.get("seeds"):
            out += ["Random seeds: " + ", ".join(f"{k} = {v}" for k, v in r["seeds"].items()) + ".", ""]
        u = r.get("unit") or {}
        if u.get("statement"):
            out += [f"Unit of the p-values, as stated by the app: {u['statement']}", ""]
        if r.get("status_line"):
            out += ["As reported by the app:", ""]
            out += ["> " + ln for ln in str(r["status_line"]).splitlines() if ln.strip()]
            out.append("")
        if r.get("warnings"):
            out += ["Caveats raised by the app:", ""] + [f"- {w}" for w in r["warnings"]] + [""]
        out += [f"Software: {_pm_software(r.get('software'))}. Record {r.get('id')} "
                f"({'loaded from the cache' if r.get('cache') == 'cache' else 'computed'}"
                + (f", {r['runtime_s']:g} s" if isinstance(r.get("runtime_s"), (int, float)) else "") + ")."
                + (f" The result was read back from the cache later (newest: record {reloaded[r.get('id')].get('id')}, "
                   f"{str(reloaded[r.get('id')].get('time_local') or reloaded[r.get('id')].get('time') or '')[:10]})."
                   if r.get("id") in reloaded else ""), ""]
    if refs_used:
        out += ["## References", ""]
        out += [f"{i}. {_PROV_REFS[k]}" for i, k in enumerate(refs_used, 1)]
        out += ["", "Entries marked [verify] could not be confirmed when the app's reference list was "
                    "written: check them against the publication before citing.", ""]
    out += ["## All records", "", "| When | Analysis | Segmentation | Counts | Source | Seconds | Record |",
            "|---|---|---|---|---|---|---|"]
    for r in recs:
        dd = r.get("data") or {}
        out.append(f"| {str(r.get('time_local') or r.get('time') or '').replace('T', ' ')} | {r.get('title')} | "
                   f"{dd.get('seg_source') or ''} | {(dd.get('counts') or {}).get('used') or ''} | "
                   f"{r.get('cache')} | {r.get('runtime_s') if r.get('runtime_s') is not None else ''} | "
                   f"{r.get('id')} |")
    return "\n".join(out) + "\n"
