"""Headless batch mode: run analyses from a config file, without a browser.

    python -m xenium_app.batch config.yaml [--dry-run] [--list-steps]

The config names the data, the cache and output directories and an ordered list of steps.
A step is run by calling the SAME function the step's Run button calls (the Dash callback,
as a plain Python function): it validates its inputs, assembles the arguments and starts
the job through the job registry exactly as in the app. The batch waits for the job,
prints its progress and then writes what the tab's Download button would write, the
tables of the result, the figures of the tab and the provenance record. There is no
second implementation of any analysis, of its argument handling or of its defaults: a
parameter the config leaves out takes the default value of its control in the app layout.

No web server is started and no port is bound: the Dash app object is built on import
(about 0.4 s, measured on the test bundle) because the callbacks are its functions, but
``app.run`` is never called.

The package loads its dataset while it is imported, so ``xenium_app/__init__.py`` reads the
config's ``data.path`` and ``cache_dir`` before anything else when Python was started as
``python -m xenium_app.batch``. From Python (tests, notebooks) call ``_batch_run(config)``
in a process whose loaded dataset is the config's.
"""

import os
import re
import sys
import json
import time as _time
import inspect
import argparse
import traceback

import numpy as np
import pandas as pd

from . import core, jobs, caching, dataview, stats, analysis, controls, provenance, compat


class _BatchError(Exception):
    """A config or step error with a message for the user."""


# ─── Steps ────────────────────────────────────────────────────────────────────
# kind → run: (module, callback) of the Run button; state: (module, state dict, lock) of
# the job kind; download: (module, callback) of the Download button; figures: (file stem,
# module, figure function); fixed: values of controls the step kind pins; sync: the
# callback computes in place (no job).
_BATCH_STEPS = {
    "annotation":   {"run": ("cb_tools", "start_annotation"), "state": ("core", "_annot_state", "_annot_lock"),
                     "figures": [("cell_type_abundance", "figures", "make_cell_type_abundance_fig")]},
    "clustering":   {"run": ("cb_reseg", "start_reseg_umap"),
                     "state": ("core", "_umap_reseg_state", "_umap_reseg_lock")},
    "compartments": {"run": ("cb_controls", "run_compartments"),
                     "state": ("controls", "_comp_ctl_state", "_comp_ctl_lock")},
    "spillover":    {"run": ("cb_controls", "run_spillover"), "state": ("controls", "_spill_state", "_spill_lock"),
                     "download": ("cb_controls", "download_ownership"),
                     "figures": [("spillover", "controls", "make_spill_fig")]},
    "segqc":        {"run": ("cb_controls", "run_segqc"), "state": ("controls", "_segqc_state", "_segqc_lock"),
                     "download": ("cb_controls", "download_segqc"),
                     "figures": [("segmentation_qc", "controls", "make_segqc_fig")]},
    "coloc":        {"run": ("cb_controls", "run_coloc"), "state": ("controls", "_coloc_state", "_coloc_lock"),
                     "download": ("cb_controls", "download_coloc"),
                     "figures": [("colocalisation", "controls", "make_coloc_fig")]},
    "de":           {"run": ("cb_analysis", "run_de"), "state": ("analysis", "_de_state", "_de_lock"),
                     "download": ("cb_analysis", "download_de_table"), "fixed": {"de-mode": "cells"},
                     "figures": [("volcano", "analysis", "make_de_volcano_fig")]},
    "pseudobulk":   {"run": ("cb_analysis", "run_de"), "state": ("analysis", "_de_state", "_de_lock"),
                     "download": ("cb_analysis", "download_de_table"), "fixed": {"de-mode": "pseudobulk"},
                     "figures": [("volcano", "analysis", "make_de_volcano_fig")], "design": True},
    "proximity":    {"run": ("cb_analysis", "run_de"), "state": ("analysis", "_de_state", "_de_lock"),
                     "download": ("cb_analysis", "download_de_table"), "fixed": {"de-mode": "proximity"},
                     "figures": [("volcano", "analysis", "make_de_volcano_fig")]},
    "contact":      {"run": ("cb_analysis", "run_de"), "state": ("analysis", "_de_state", "_de_lock"),
                     "download": ("cb_analysis", "download_de_table"), "fixed": {"de-mode": "contact"},
                     "figures": [("volcano", "analysis", "make_de_volcano_fig")], "design": True},
    "geneset":      {"run": ("cb_analysis", "run_geneset_score"),
                     "state": ("analysis", "_geneset_state", "_geneset_lock")},
    "density":      {"run": ("cb_analysis", "run_density"), "state": ("analysis", "_density_state", "_density_lock"),
                     "transcripts": True},
    "composition":  {"run": ("cb_stats", "run_composition"), "state": ("stats", "_comp_state", "_comp_lock"),
                     "download": ("cb_stats", "download_composition"), "sync": True,
                     "figures": [("composition", "stats", "make_composition_fig")]},
    "moran":        {"run": ("cb_stats", "run_moran"), "state": ("stats", "_moran_state", "_moran_lock"),
                     "download": ("cb_stats", "download_moran_table")},
    "lr":           {"run": ("cb_stats", "run_lr"), "state": ("stats", "_lr_state", "_lr_lock"),
                     "download": ("cb_stats", "download_lr_table")},
    "lrdecomp":     {"run": ("cb_lrdecomp", "run_lrdecomp"), "state": ("lrdecomp", "_lrd_state", "_lrd_lock"),
                     "download": ("cb_lrdecomp", "download_lrd_table"),
                     "figures": [("lr_decomposition", "lrdecomp", "make_lrd_fig")]},
    "aggregates":   {"run": ("cb_aggregates", "run_aggregates"), "state": ("aggregates", "_agg_state", "_agg_lock"),
                     "download": ("cb_aggregates", "download_aggregates"),
                     "figures": [("aggregates", "aggregates", "make_agg_fig")]},
    "segfree":      {"run": ("cb_segfree", "run_segfree"), "state": ("segfree", "_sfree_state", "_sfree_lock"),
                     "download": ("cb_segfree", "download_segfree"), "transcripts": True,
                     "figures": [("segmentation_free_abundance", "segfree", "make_sfree_fig")]},
    "banksy":       {"run": ("cb_banksy", "run_banksy"), "state": ("banksy", "_bk_state", "_bk_lock"),
                     "download": ("cb_banksy", "download_banksy"),
                     "figures": [("banksy_domains", "banksy", "make_bk_fig")]},
    "joint":        {"run": ("cb_joint", "run_joint"), "state": ("joint", "_jc_state", "_jc_lock"),
                     "download": ("cb_joint", "download_joint"), "multi": True,
                     "figures": [("joint_clusters", "joint", "make_jc_fig")]},
    "txfactor":     {"run": ("cb_txfactor", "run_txfactor"), "state": ("txfactor", "_txf_state", "_txf_lock"),
                     "download": ("cb_txfactor", "download_txfactor"), "transcripts": True,
                     "figures": [("transcript_factors", "txfactor", "make_txf_fig")]},
    "spatial":      {"run": ("cb_spatial", "run_spatial_analysis"),
                     "state": ("core", "_spatial_state", "_spatial_lock"),
                     "figures": [("cooccurrence", "figures", "make_cooccurrence_fig"),
                                 ("neighbourhood_enrichment", "figures", "make_enrichment_fig"),
                                 ("interaction_graph", "figures", "make_interaction_graph_fig")]},
    "niches":       {"run": ("cb_spatial", "run_niche"), "state": ("core", "_niche_state", "_niche_lock")},
    "export_spatialdata": {"run": ("cb_data", "start_save_sdata"),
                           "state": ("core", "_save_sdata_state", "_save_sdata_lock")},
}
# need: controls that must have a value; uses: the controls of the shared DE callback that
# belong to this mode (the others cannot be set and are not validated)
_BATCH_NEED = {
    "spillover": ["ctl-labels"], "coloc": ["coloc-labels"], "segqc": ["segqc-method"],
    "de": ["de-group-a", "de-group-b"], "pseudobulk": ["de-pb-population"],
    "proximity": ["prox-labels", "prox-target", "prox-anchor"],
    "contact": ["prox-labels", "prox-target", "prox-anchor"],
    "lr": ["lr-labels", "lr-pairs"], "lrdecomp": ["lr-labels", "lr-pairs"],
    "aggregates": ["agg-labels", "agg-classes"], "segfree": ["sfree-labels", "sfree-type"],
    "composition": ["comp-category"], "geneset": ["geneset-name", "geneset-genes"],
    "density": ["density-gene"],
}
_BATCH_USES = {
    "de": ["de-group-a", "de-group-b", "de-leak-gate"],
    "pseudobulk": ["de-pb-population", "de-pb-min-cells", "de-pb-min-count", "de-pb-shrink", "de-leak-gate"],
    "proximity": ["prox-labels", "prox-target", "prox-anchor", "prox-near", "prox-far", "prox-nrand",
                  "prox-source", "prox-length", "de-leak-gate"],
    "contact": ["prox-labels", "prox-target", "prox-anchor", "contact-radius", "contact-min-cells",
                "contact-nshuffle", "de-leak-gate"],
}
_BATCH_ALIASES = {"umap": "clustering", "segmentation_qc": "segqc", "colocalisation": "coloc",
                  "colocalization": "coloc", "gene_set": "geneset", "save": "export_spatialdata",
                  "lr_scores": "lr", "lr_decomposition": "lrdecomp", "transcript_factors": "txfactor",
                  "segmentation_free": "segfree", "joint_clustering": "joint", "de_cells": "de"}

# controls every step shares; their value comes from the config's top level
_BATCH_CONTEXT = {"seg-source": "seg_source", "counts-mode-store": "counts", "cluster-method": "cluster_method",
                  "color-by": "color_by"}
# what a control's value has to be, for --dry-run: component id → (role, id of the control
# holding the label source for a cell-type role)
_BATCH_ROLES = {
    "ctl-labels": ("label_kind", None), "coloc-labels": ("label_kind", None), "prox-labels": ("label_kind", None),
    "lr-labels": ("label_kind", None), "agg-labels": ("label_kind", None), "sfree-labels": ("label_kind", None),
    "txf-labels": ("label_kind_or_none", None), "comp-category": ("label_kind", None),
    "ctl-dominant": ("cell_type", "ctl-labels"), "spill-source": ("cell_type", "ctl-labels"),
    "coloc-placebo": ("cell_type", "coloc-labels"), "prox-target": ("cell_type", "prox-labels"),
    "prox-anchor": ("cell_type", "prox-labels"), "sfree-type": ("cell_type", "sfree-labels"),
    "agg-classes": ("cell_types", "agg-labels"),
    "density-gene": ("gene", None), "geneset-genes": ("genes_text", None), "spill-markers": ("genes_text", None),
    "sfree-markers": ("genes_text", None), "lr-pairs": ("pairs_text", None),
    "de-group-a": ("de_group", None), "de-group-b": ("de_group", None),
    "de-pb-population": ("population", None),
    "annot-rds-path": ("file", None), "save-sdata-dir": ("dir", None),
}
def _batch_unset(v) -> bool:
    """A control left at "nothing chosen" / "auto"."""
    return v is None or (isinstance(v, str) and v in ("", "auto")) or (isinstance(v, (list, tuple)) and not v)


def _batch_kind(kind) -> str:
    k = str(kind or "").strip().lower().replace("-", "_")
    return _BATCH_ALIASES.get(k, k)


def _batch_module(name):
    return sys.modules[f"{__package__}.{name}"]


def _batch_app():
    return _batch_module("app_layout").app


_batch_memo: dict = {}


def _batch_io(func) -> list:
    """[(argument name, component id, property, is_input)] of a callback, in the order of
    its arguments (Dash passes the Inputs, then the States)."""
    if func in _batch_memo:
        return _batch_memo[func]
    for entry in _batch_app().callback_map.values():
        cb = entry.get("callback")
        if getattr(cb, "__wrapped__", None) is func or cb is func:
            io = [(i["id"], i["property"], True) for i in entry["inputs"]] + \
                 [(s["id"], s["property"], False) for s in entry["state"]]
            names = list(inspect.signature(func).parameters)
            if len(names) < len(io):
                raise _BatchError(f"{func.__name__}: {len(io)} callback inputs, {len(names)} arguments")
            out = [(names[i],) + io[i] for i in range(len(io))]
            _batch_memo[func] = out
            return out
    raise _BatchError(f"{func.__name__} is not a registered callback")


def _batch_components() -> dict:
    """Component id → component of the app layout (the defaults of the controls)."""
    if "components" not in _batch_memo:
        out = {}

        def walk(node):
            cid = getattr(node, "id", None)
            if isinstance(cid, str):
                out[cid] = node
            kids = getattr(node, "children", None)
            if isinstance(kids, (list, tuple)):
                for k in kids:
                    walk(k)
            elif kids is not None and hasattr(kids, "to_plotly_json"):
                walk(kids)
        walk(_batch_app().layout)
        _batch_memo["components"] = out
    return _batch_memo["components"]


def _batch_default(cid, prop):
    comp = _batch_components().get(cid)
    return getattr(comp, prop, None) if comp is not None else None


def _batch_coerce(cid, prop, value):
    """A config value as the control holds it: true / false for a checklist becomes its
    list value."""
    comp = _batch_components().get(cid)
    if isinstance(value, bool) and comp is not None and type(comp).__name__ == "Checklist":
        opts = getattr(comp, "options", None) or []
        first = opts[0]["value"] if opts and isinstance(opts[0], dict) else (opts[0] if opts else "on")
        return [first] if value else []
    return value


def _batch_context(cfg) -> dict:
    counts = str(cfg.get("counts") or "original").lower()
    if counts not in ("original", "corrected"):
        raise _BatchError(f"counts: '{counts}' is not 'original' or 'corrected'")
    core._active_counts_mode = counts
    methods = list(core.DATA.get("cluster_methods") or [])
    return {"seg-source": cfg.get("seg_source") or "xenium", "counts-mode-store": counts,
            "cluster-method": cfg.get("cluster_method") or _batch_default("cluster-method", "value")
            or (methods[0] if methods else None),
            "color-by": cfg.get("color_by") or _batch_default("color-by", "value")}


def _batch_step_args(spec, params, ctx, kind=None) -> tuple:
    """(positional arguments of the Run callback, {component id: value} as resolved,
    names the config may use). Unknown parameter names raise."""
    func = getattr(_batch_module(spec["run"][0]), spec["run"][1])
    io = _batch_io(func)
    fixed = spec.get("fixed") or {}
    allowed = {}
    uses = _BATCH_USES.get(kind)
    for name, cid, prop, is_input in io:
        if not is_input and cid not in fixed and cid not in _BATCH_CONTEXT and (uses is None or cid in uses):
            allowed[name] = cid
            allowed[cid] = cid
            if _BATCH_ROLES.get(cid, (None,))[0] in ("label_kind", "label_kind_or_none"):
                allowed["labels"] = cid            # one name for "where the cell types come from"
    for cid in _BATCH_CONTEXT:                      # a step may override the shared context
        if any(c == cid for _n, c, _p, _i in io):
            allowed[_BATCH_CONTEXT[cid]] = cid
    unknown = [k for k in (params or {}) if k not in allowed]
    if unknown:
        raise _BatchError(f"unknown parameter(s) {', '.join(map(str, unknown))}; this step takes: "
                          + ", ".join(sorted(n for n, c in allowed.items() if n != c)))
    given = {allowed[k]: v for k, v in (params or {}).items()}
    args, resolved = [], {}
    # label lookups without a counts argument follow the mode of the step (core mirror)
    core._active_counts_mode = ctx.get("counts-mode-store") or "original"
    for name, cid, prop, is_input in io:
        if is_input:
            val = 1 if prop in ("n_clicks", "n_intervals", "data") else None
        elif cid in fixed:
            val = fixed[cid]
        elif cid in given:
            val = _batch_coerce(cid, prop, given[cid])
            if cid == "counts-mode-store":
                val = "corrected" if str(val).lower() == "corrected" else "original"
                core._active_counts_mode = val
        elif cid in ctx:
            val = ctx[cid]
        else:
            val = _batch_default(cid, prop)
        args.append(val)
        if not is_input and (uses is None or cid in uses or cid in fixed or cid in _BATCH_CONTEXT):
            resolved[cid] = val
    return args, resolved, allowed


def _batch_list_steps() -> str:
    """Every step kind with its parameters (argument name, control, default)."""
    lines = []
    for kind, spec in _BATCH_STEPS.items():
        func = getattr(_batch_module(spec["run"][0]), spec["run"][1])
        lines.append(f"{kind}  ({spec['run'][0]}.{spec['run'][1]})")
        for name, cid, prop, is_input in _batch_io(func):
            if is_input or cid in (spec.get("fixed") or {}):
                continue
            if kind in _BATCH_USES and cid not in _BATCH_USES[kind] and cid not in _BATCH_CONTEXT:
                continue
            if cid in _BATCH_CONTEXT:
                lines.append(f"    {_BATCH_CONTEXT[cid]:<22} shared ({cid})")
                continue
            d = _batch_default(cid, prop)
            d = (repr(d) if not isinstance(d, str) or len(d) < 50 else repr(d[:47] + "…"))
            also = " (or: labels)" if _BATCH_ROLES.get(cid, (None,))[0] in ("label_kind", "label_kind_or_none") else ""
            req = "  REQUIRED" if cid in _BATCH_NEED.get(kind, ()) else ""
            lines.append(f"    {name + also:<22} {cid:<26} default {d}{req}")
    return "\n".join(lines)


# ─── Config ───────────────────────────────────────────────────────────────────
def _batch_read_config(path: str) -> dict:
    """The config file as a dict: JSON, or YAML when PyYAML is installed."""
    return sys.modules[__package__]._batch_config_file(path)


def _batch_norm_config(cfg) -> dict:
    if not isinstance(cfg, dict):
        raise _BatchError("the config must be a mapping (data, output_dir, steps, …)")
    known = {"data", "cache_dir", "output_dir", "steps", "continue_on_error", "seg_source", "counts",
             "cluster_method", "color_by", "figures", "tables"}
    extra = set(cfg) - known
    if extra:
        raise _BatchError(f"unknown top-level key(s): {', '.join(sorted(map(str, extra)))} "
                          f"(known: {', '.join(sorted(known))})")
    data = cfg.get("data")
    if isinstance(data, str):
        data = {"path": data}
    if not isinstance(data, dict) or not data.get("path"):
        raise _BatchError("data.path is missing (a Xenium bundle directory, a .zarr store or an .h5ad file)")
    extra = set(data) - {"path", "extra_samples", "design", "h5ad_mapping"}
    if extra:
        raise _BatchError(f"unknown key(s) under data: {', '.join(sorted(map(str, extra)))}")
    steps = cfg.get("steps")
    if not isinstance(steps, list) or not steps:
        raise _BatchError("steps is missing or empty")
    out_steps, names = [], set()
    for i, st in enumerate(steps, 1):
        if isinstance(st, str):
            st = {"kind": st}
        if not isinstance(st, dict) or not st.get("kind"):
            raise _BatchError(f"step {i}: needs a 'kind'")
        bad = set(st) - {"kind", "name", "params"}
        if bad:
            raise _BatchError(f"step {i}: unknown key(s) {', '.join(sorted(map(str, bad)))} "
                              f"(a step has kind, name, params)")
        kind = _batch_kind(st["kind"])
        if kind not in _BATCH_STEPS:
            raise _BatchError(f"step {i}: unknown kind '{st['kind']}' (known: {', '.join(_BATCH_STEPS)})")
        params = st.get("params") or {}
        if not isinstance(params, dict):
            raise _BatchError(f"step {i} ({kind}): params must be a mapping")
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(st.get("name") or kind))[:60]
        if name in names:
            name = f"{name}_{i}"
        names.add(name)
        out_steps.append({"index": i, "kind": kind, "name": name, "params": dict(params)})
    return {**cfg, "data": dict(data), "steps": out_steps,
            "continue_on_error": bool(cfg.get("continue_on_error", False)),
            "figures": cfg.get("figures", True), "tables": cfg.get("tables", True)}


def _batch_same_path(a, b) -> bool:
    try:
        return os.path.realpath(str(a).rstrip("/")) == os.path.realpath(str(b).rstrip("/"))
    except Exception:
        return False


def _batch_loaded_source() -> str:
    """What the running process loaded as its primary dataset (the .h5ad for a dataset
    converted from one)."""
    h = core.DATA.get("h5ad") or {}
    return str(h.get("source") or h.get("path") or caching._dataset_key())


def _batch_check_data(cfg) -> None:
    want = cfg["data"]["path"]
    have = [_batch_loaded_source(), caching._dataset_key(), core.DATA.get("data_dir") or "",
            core.DATA.get("sdata_path") or ""]
    if not any(h and _batch_same_path(want, h) for h in have):
        raise _BatchError(
            f"data.path is {want}, but this process has loaded {have[0]}. The dataset is loaded when the "
            f"package is imported: run 'python -m xenium_app.batch <config>' (a new process per config).")


def _batch_setup_samples(cfg, log) -> None:
    """Extra samples and the design, through the functions the app's dialogs use."""
    data = cfg["data"]
    cb_data = _batch_module("cb_data")
    for p in data.get("extra_samples") or []:
        p = os.path.abspath(os.path.expanduser(str(p)))
        if any(_batch_same_path(p, dataview._ds_key(ds)) for ds in [core.DATA] + list(dataview.EXTRA_DATASETS)):
            continue
        out = cb_data.add_sample(1, None, p, len(dataview.EXTRA_DATASETS))
        if not str(out[0]).startswith("Added"):
            raise _BatchError(f"extra sample {p}: {out[0]}")
        log(f"  added sample {dataview._ds_name(dataview.EXTRA_DATASETS[-1])}: {out[0]}")
    design = data.get("design")
    if design:
        rows = stats._design_samples()
        by = {}
        for r in rows:
            for k in (r["label"], r["name"], r["key"], os.path.basename(str(r["key"]).rstrip("/"))):
                by.setdefault(str(k), r)
        new = []
        names = design.get("names") or {}
        # names as text: YAML reads `A: 1` / `B: no` as a number / a boolean
        names = {c: str(names[c]).strip() for c in ("A", "B") if names.get(c) is not None and str(names[c]).strip()}
        for d in design.get("samples") or []:
            if not isinstance(d, dict) or "sample" not in d:
                raise _BatchError("data.design.samples: each entry needs 'sample' (S0, S1, …, a name or a path)")
            r = by.get(str(d["sample"])) or next((x for x in rows if _batch_same_path(d["sample"], x["key"])), None)
            if r is None:
                raise _BatchError(f"data.design: no loaded sample '{d['sample']}' "
                                  f"(loaded: {', '.join(x['label'] + '=' + x['name'] for x in rows)})")
            # presence, not truthiness: a condition written as 0 or false is a condition
            raw = d.get("condition")
            asked = "" if raw is None else str(raw).strip()
            cond = next((c for c in ("A", "B") if asked and asked in (c, names.get(c))), "")
            if asked and not cond:
                raise _BatchError(f"data.design: sample '{d['sample']}' has condition '{raw}', "
                                  f"which is neither A / B nor one of the names {names}")
            _txt = lambda v: "" if v is None else str(v).strip()
            new.append({"key": r["key"], "cond": cond, "batch": _txt(d.get("batch")),
                        "donor": _txt(d.get("donor"))})
        # The config IS the design of the run: a sample it does not list is unassigned and
        # a listed sample without donor / batch has none, whatever an earlier session or
        # run left in the design file of these samples.
        listed = {row["key"] for row in new}
        new += [{"key": r["key"], "cond": "", "batch": "", "donor": ""} for r in rows if r["key"] not in listed]
        with stats._design_lock:
            stats._design_state["names"].update({"A": "A", "B": "B"})
        stats._design_set(new, names)
        log("  design: " + stats._design_summary())


def _batch_wait(job, kind, log) -> None:
    """Wait for a job of the registry; print its step text and fraction as they change."""
    last = None
    while True:
        job.thread.join(timeout=0.2)
        snap = (job.step, None if job.fraction is None else int(job.fraction * 20))
        if snap != last and job.step:
            pct = "" if job.fraction is None else f"{100 * job.fraction:3.0f} % "
            log(f"    [{kind}] {pct}{str(job.step).splitlines()[0][:200]}")
            last = snap
        if not job.thread.is_alive():
            return


def _batch_run_callback(module, name, args, kind, log, state=None, wait_status=("running", "starting", "loading")):
    """Call a Run callback and wait for the job it started. Returns (job or None, what
    the callback returned)."""
    func = getattr(_batch_module(module), name)
    before = {j.id for j in jobs._jobs_list()}
    try:
        out = func(*args)
    except Exception as exc:
        if type(exc).__name__ == "PreventUpdate":
            raise _BatchError("the step's inputs were not accepted (PreventUpdate)") from None
        raise
    new = [j for j in jobs._jobs_list() if j.id not in before]
    for job in new:
        _batch_wait(job, kind, log)
    return (new[-1] if new else None), out


def _batch_text(out) -> str:
    parts = out if isinstance(out, (list, tuple)) else [out]
    return " ".join(str(p) for p in parts if isinstance(p, str) and p.strip())


# ─── Outputs of a step ────────────────────────────────────────────────────────
def _batch_write_tables(res, step_dir, files) -> None:
    if not isinstance(res, dict):
        return
    for key, val in res.items():
        if isinstance(val, pd.DataFrame) and len(val.columns):
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(key))
            df = val.reset_index() if not isinstance(val.index, pd.RangeIndex) else val
            df = df.copy()
            df.columns = [str(c) for c in df.columns]
            try:
                p = os.path.join(step_dir, f"{stem}.parquet")
                df.to_parquet(p, index=False)
            except Exception:
                p = os.path.join(step_dir, f"{stem}.csv")
                df.to_csv(p, index=False)
            files.append(p)


def _batch_cell_table(kind, resolved, ctx) -> "pd.DataFrame | None":
    """Per-cell output of the steps whose result is columns of the cells, not a table."""
    seg = resolved.get("seg-source") or ctx["seg-source"]
    tables = stats._view_tables(seg)
    parts = []
    for t in tables:
        df = t["df"]
        out = pd.DataFrame({"sample": t["label"], "cell_id": df.index.astype(str)})
        if kind == "compartments":
            vals = {c: controls._ctl_get(t, seg, c) for c in controls._COMP_COLUMNS}
            vals = {c: v for c, v in vals.items() if v is not None}
            if not vals:
                continue
            for c, v in vals.items():
                out[c] = v
            parts.append(out)
            continue
        elif kind == "niches":
            cols = [c for c in df.columns if str(c) == "niche"]
        elif kind == "clustering":
            cols = [c for c in df.columns if str(c) in ("umap_1", "umap_2") or str(c).startswith("cluster")]
        else:
            cols = []
        if not cols:
            return None
        for c in cols:
            out[str(c)] = df[c].to_numpy()
        parts.append(out)
    return pd.concat(parts, ignore_index=True) if parts else None


def _batch_write_outputs(step, spec, resolved, ctx, res, step_dir, cfg, log) -> list:
    files = []
    kind = step["kind"]
    if cfg.get("tables", True):
        if spec.get("download"):
            try:
                out = getattr(_batch_module(spec["download"][0]), spec["download"][1])(1)
                p = os.path.join(step_dir, re.sub(r"[^A-Za-z0-9_.-]+", "_", str(out["filename"])))
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write(out["content"])
                files.append(p)
            except Exception as exc:
                if type(exc).__name__ != "PreventUpdate":
                    raise
        _batch_write_tables(res, step_dir, files)
        cells = None
        if kind in ("compartments", "niches", "clustering"):
            cells = _batch_cell_table(kind, resolved, ctx)
        elif kind == "annotation":
            src = str(resolved.get("annot-source") or "celltypist")
            seg = resolved.get("seg-source") or ctx["seg-source"]
            # the key the annotation job wrote: per method and, for a run, per run
            lk = _batch_module("annotation")._labels_key_for_method(
                src if src in core._ANNOT_METHODS else "celltypist", core._seg_result(seg, done_only=True))
            with core._annot_lock:
                lab = core._annot_state.get(lk)
            if lab is not None:
                cells = pd.DataFrame({"cell_id": pd.Series(lab).index.astype(str),
                                      f"cell_type_{src}": pd.Series(lab).astype(object).to_numpy()})
        elif kind == "geneset":
            # the name as the callback stored it, the segmentation and counts of the step
            name = re.sub(r"\s+", " ", str(resolved.get("geneset-name") or "").strip())
            seg = resolved.get("seg-source") or ctx["seg-source"]
            corr = (resolved.get("counts-mode-store") or ctx["counts-mode-store"]) == "corrected"
            arrays, _info = analysis._geneset_values(name, seg, corr)
            tabs = stats._view_tables(seg)
            if arrays is not None and len(arrays) == len(tabs):
                cells = pd.concat([pd.DataFrame({"sample": t["label"], "cell_id": t["df"].index.astype(str),
                                                 f"geneset_{name}": np.asarray(a)})
                                   for t, a in zip(tabs, arrays)], ignore_index=True)
        if cells is not None:
            try:
                p = os.path.join(step_dir, "cells.parquet")
                cells.to_parquet(p, index=False)
            except Exception:
                p = os.path.join(step_dir, "cells.csv")
                cells.to_csv(p, index=False)
            files.append(p)
    if cfg.get("figures", True):
        for stem, mod, fn in spec.get("figures") or []:
            try:
                fig = getattr(_batch_module(mod), fn)()
            except Exception as exc:
                log(f"    figure {stem}: not written ({exc})")
                continue
            files += _batch_write_figure(fig, os.path.join(step_dir, stem))
    return files


def _batch_kaleido() -> bool:
    try:
        import kaleido  # noqa: F401
        return True
    except Exception:
        return False


def _batch_write_figure(fig, stem) -> list:
    """A static file of a Plotly figure: PNG through kaleido when it is installed, else the
    figure as self-contained-data HTML (plotly.js from the CDN) and as Plotly JSON."""
    out = []
    if _batch_kaleido():
        try:
            fig.write_image(stem + ".png", width=1200, height=800, scale=2)
            out.append(stem + ".png")
        except Exception as exc:
            print(f"    figure {os.path.basename(stem)}: PNG not written ({exc})", flush=True)
    with open(stem + ".plotly.json", "w", encoding="utf-8") as fh:
        fh.write(fig.to_json())
    out.append(stem + ".plotly.json")
    if not any(p.endswith(".png") for p in out):
        fig.write_html(stem + ".html", include_plotlyjs="cdn", full_html=True)
        out.append(stem + ".html")
    return out


# ─── Validation (--dry-run) ───────────────────────────────────────────────────
def _batch_seg_run(seg_source) -> "dict | None":
    """The cached run *seg_source* ("<tool>:<tag>") names, from the listing; None when
    there is none."""
    return next((r for r in caching._list_cached_seg_runs() if r["value"] == seg_source), None)


def _batch_seg_ready(seg_source) -> bool:
    """True when the cells a step on *seg_source* would use are those it names: the
    Xenium cells, or the loaded result of the tool when that IS the named run (a tool
    holds one run at a time, and core._seg_result() answers for the tool alone)."""
    tool = core._seg_tool(seg_source)
    if tool == "xenium":
        return True
    res = core._seg_result(seg_source, done_only=True)
    if res is None:
        return False
    if ":" not in str(seg_source):
        return True                              # "<tool>": whichever run is loaded
    run = _batch_seg_run(seg_source)
    return run is not None and _batch_same_path(run.get("out_dir") or "", res.get("out_dir") or "")


def _batch_ensure_seg(seg_source, log) -> None:
    """Load the cached run *seg_source* names unless it is the loaded one (nothing is
    computed). _BatchError when it does not exist or cannot be loaded."""
    tool = core._seg_tool(seg_source)
    if _batch_seg_ready(seg_source):
        return
    if tool not in core._SEG_TOOL_LABELS:
        raise _BatchError(f"seg_source '{seg_source}': unknown segmentation (xenium, or <tool>:<run tag> with "
                          f"tool one of {', '.join(core._SEG_TOOL_LABELS)})")
    if ":" not in str(seg_source) or _batch_seg_run(seg_source) is None:
        runs = sorted(o["value"] for o in caching._list_cached_seg_runs())
        raise _BatchError(f"seg_source '{seg_source}': no such cached run (cached: {', '.join(runs) or 'none'})")
    log(f"Loading the cached run {seg_source}…")
    try:
        if tool == "imgseg":
            _batch_run_callback("cb_reseg", "load_imgseg_run_from_cache", [seg_source], "load", log)
        else:
            _batch_run_callback("cb_reseg", "load_seg_run_from_cache", [seg_source, 0, 0], "load", log)
    except _BatchError:
        pass                                     # PreventUpdate: judged by the result below
    if not _batch_seg_ready(seg_source):
        raise _BatchError(f"seg_source '{seg_source}' could not be loaded from the cache")


def _batch_label_kinds(ctx) -> set:
    try:
        return {o["value"] for o in stats._comp_category_options(ctx["seg-source"], ctx["color-by"])}
    except Exception:
        return set()


def _batch_cell_types(kind, ctx) -> "set | None":
    try:
        tables = stats._view_tables(ctx["seg-source"])
        labels, _cat = stats._comp_labels(kind, tables, ctx["seg-source"], ctx["cluster-method"],
                                           ctx["counts-mode-store"] == "corrected")
        return {str(t) for t in controls._pooled_types(labels).index}
    except Exception:
        return None


def _batch_validate(cfg, log=None) -> tuple:
    """(errors, warnings) of the config against the loaded data, without computing. The
    checks of a step are made on the segmentation the step names: a cached run is loaded
    for them (also in a dry run - loading computes nothing)."""
    errors, warns = [], []
    try:
        ctx = _batch_context(cfg)
    except _BatchError as exc:
        return [str(exc)], []
    genes = set(map(str, core.DATA.get("gene_names") or []))
    methods = list(core.DATA.get("cluster_methods") or [])
    seg_tool = core._seg_tool(ctx["seg-source"])
    if not _batch_seg_ready(ctx["seg-source"]):
        try:
            _batch_ensure_seg(ctx["seg-source"], log or (lambda s: None))
        except _BatchError as exc:
            return [str(exc)], warns
    if cfg.get("cluster_method") and methods and cfg["cluster_method"] not in methods:
        try:
            from . import clustersets
            known = {o["value"] for o in clustersets._cs_options(ctx["seg-source"])}
        except Exception:
            known = set(methods)
        if cfg["cluster_method"] not in known:
            errors.append(f"cluster_method '{cfg['cluster_method']}' is not a cluster set of the data "
                          f"({', '.join(map(str, sorted(known)))})")
    if ctx["counts-mode-store"] == "corrected" and seg_tool == "xenium":
        st = dataview._corrected_status()
        if not st.get("all"):
            warns.append("counts: corrected - not every loaded sample has SPLIT-corrected counts; the "
                         "analyses will say that original counts were used")
    # counts that are not raw counts (an .h5ad whose matrix is not integer-valued): what the
    # mapping says the count models may do with them is said before anything runs
    _ni = {str(ds.get("counts_noninteger")) for ds in [core.DATA] + list(dataview.EXTRA_DATASETS)
           if ds.get("counts_noninteger")}
    if "round_sums" in _ni:
        warns.append("noninteger_counts = round_sums: the count matrix of this data is NOT integer-valued "
                     "(corrected, not raw, counts). Pseudobulk steps will sum these values per sample or donor "
                     "and round the sums; their results are not from raw counts, and the provenance record "
                     "and methods text say so")
    rows = stats._design_samples()
    n_a, n_b = sum(r["cond"] == "A" for r in rows), sum(r["cond"] == "B" for r in rows)
    have_tx = bool(core.DATA.get("data_dir")) and os.path.exists(
        os.path.join(str(core.DATA.get("data_dir")), "transcripts.parquet"))
    label_kinds = _batch_label_kinds(ctx)
    _lk_seg = ctx["seg-source"]              # the segmentation label_kinds was read for
    annotated_later = False
    for st in cfg["steps"]:
        spec, where = _BATCH_STEPS[st["kind"]], f"step {st['index']} ({st['name']})"
        try:
            _args, resolved, allowed = _batch_step_args(spec, st["params"], ctx, st["kind"])
        except _BatchError as exc:
            errors.append(f"{where}: {exc}")
            continue
        for cid in _BATCH_NEED.get(st["kind"], ()):
            if resolved.get(cid) in (None, "", []):
                nm = sorted(n for n, c in allowed.items() if c == cid and n != cid)
                errors.append(f"{where}: parameter '{nm[0] if nm else cid}' is required")
        sctx = {**ctx, **{k: v for k, v in resolved.items() if k in _BATCH_CONTEXT}}
        if sctx["seg-source"] != ctx["seg-source"]:
            # a step on another segmentation: its labels, cell types and groups are those
            # of that run, which is loaded for the checks (and again before the step runs)
            try:
                _batch_ensure_seg(sctx["seg-source"], log or (lambda s: None))
            except _BatchError as exc:
                errors.append(f"{where}: {exc}")
                continue
            label_kinds = _batch_label_kinds(sctx)
        elif not _batch_seg_ready(ctx["seg-source"]):
            try:
                _batch_ensure_seg(ctx["seg-source"], log or (lambda s: None))
            except _BatchError as exc:
                errors.append(f"{where}: {exc}")
                continue
            label_kinds = _batch_label_kinds(ctx)
        else:
            label_kinds = _batch_label_kinds(ctx) if _lk_seg != ctx["seg-source"] else label_kinds
        _lk_seg = sctx["seg-source"]
        if spec.get("design") and (n_a < 1 or n_b < 1):
            errors.append(f"{where}: needs the sample design (samples in condition A and in B); "
                          f"the design has {n_a} in A and {n_b} in B")
        if st["kind"] == "pseudobulk" and "refuse" in _ni:
            errors.append(f"{where}: the count matrix is not integer-valued (not raw counts) and pseudobulk "
                          f"refuses it. Name the matrix with the raw counts in the h5ad mapping, or - when the "
                          f"file only has corrected counts - set \"noninteger_counts\": \"round_sums\" there")
        elif st["kind"] == "pseudobulk" and "round_sums" in _ni:
            warns.append(f"{where}: runs on non-integer counts (sums rounded, noninteger_counts = round_sums)")
        if spec.get("multi") and len(rows) < 2:
            errors.append(f"{where}: needs at least two loaded samples (data.extra_samples)")
        if spec.get("transcripts") and not have_tx:
            errors.append(f"{where}: needs the transcript file (transcripts.parquet) of a Xenium bundle; "
                          f"this dataset was loaded without one")
        for cid, val in resolved.items():
            role, ref = _BATCH_ROLES.get(cid, (None, None))
            if role is None or _batch_unset(val):
                continue
            if role in ("label_kind", "label_kind_or_none"):
                if val not in label_kinds:
                    msg = (f"{where}: label source '{val}' is not available "
                           f"(available: {', '.join(sorted(label_kinds)) or 'none'})")
                    (warns if annotated_later and str(val).startswith("celltype:") else errors).append(
                        msg + (" - an earlier annotation step may provide it" if annotated_later else ""))
            elif role in ("cell_type", "cell_types"):
                kind_val = resolved.get(ref)
                types = _batch_cell_types(kind_val, sctx) if kind_val in label_kinds else None
                if types is None:
                    continue                      # reported with the label source, or not checkable yet
                vals = val if isinstance(val, (list, tuple)) else [val]
                missing = [str(v) for v in vals if str(v) not in types and not _batch_unset(v)]
                if missing:
                    errors.append(f"{where}: no cell is labelled {', '.join(repr(m) for m in missing)} in "
                                  f"'{kind_val}' (labels: {', '.join(sorted(types)[:30])})")
            elif role == "gene":
                if str(val) not in genes and str(val) != str(getattr(analysis, "_DENSITY_ALL", "")):
                    errors.append(f"{where}: gene '{val}' is not on the panel")
            elif role == "genes_text":
                listed = [g for g in re.split(r"[\s,;]+", str(val)) if g]
                missing = [g for g in listed if g not in genes]
                if listed and len(missing) == len(listed):
                    errors.append(f"{where}: none of the listed genes is on the panel ({', '.join(missing[:10])})")
                elif missing:
                    warns.append(f"{where}: not on the panel: {', '.join(missing[:20])}")
            elif role == "pairs_text":
                pairs = [re.split(r"[\s,;]+", ln.strip()) for ln in str(val).splitlines() if ln.strip()]
                pairs = [p for p in pairs if len(p) >= 2]
                ok = [p for p in pairs if p[0] in genes and p[1] in genes]
                if not pairs:
                    errors.append(f"{where}: no ligand,receptor pairs given")
                elif not ok:
                    errors.append(f"{where}: no pair has both genes on the panel")
                elif len(ok) < len(pairs):
                    warns.append(f"{where}: {len(pairs) - len(ok)} of {len(pairs)} pairs are not on the panel")
            elif role == "de_group":
                try:
                    opts = {o["value"] for o in analysis._de_group_options(
                        sctx["seg-source"], sctx["cluster-method"], sctx["color-by"],
                        sctx["counts-mode-store"] == "corrected")}
                except Exception:
                    opts = None
                if st["kind"] == "de" and opts is not None and val not in opts:
                    msg = f"{where}: group '{val}' is not available (e.g. {', '.join(sorted(opts)[:12])})"
                    (warns if annotated_later else errors).append(msg)
            elif role == "population":
                try:
                    opts = {o["value"] for o in stats._pb_population_options(
                        sctx["seg-source"], sctx["cluster-method"], sctx["color-by"])}
                except Exception:
                    opts = None
                if opts is not None and val not in opts:
                    msg = (f"{where}: population '{val}' is not available (e.g. "
                           f"{', '.join(sorted(map(str, opts))[:12])})")
                    (warns if annotated_later else errors).append(msg)
            elif role == "file" and val and not os.path.exists(os.path.expanduser(str(val))):
                if st["kind"] != "annotation" or str(resolved.get("annot-source")) != "celltypist":
                    errors.append(f"{where}: file not found: {val}")
            elif role == "dir" and val and not os.path.isdir(os.path.expanduser(str(val))):
                errors.append(f"{where}: directory not found: {val}")
        if st["kind"] == "annotation":
            annotated_later = True
    return errors, warns


# ─── Run ──────────────────────────────────────────────────────────────────────
def _batch_step_result(spec):
    mod, sname, lname = spec["state"]
    state, lock = getattr(_batch_module(mod), sname), getattr(_batch_module(mod), lname)
    with lock:
        return state.get("status"), state.get("message"), state.get("result")


def _batch_run_step(step, cfg, ctx, out_dir, log) -> dict:
    spec = _BATCH_STEPS[step["kind"]]
    step_dir = os.path.join(out_dir, f"{step['index']:02d}_{step['name']}")
    os.makedirs(step_dir, exist_ok=True)
    args, resolved, _allowed = _batch_step_args(spec, step["params"], ctx, step["kind"])
    if step["kind"] == "export_spatialdata" and not resolved.get("save-sdata-dir"):
        io = _batch_io(getattr(_batch_module(spec["run"][0]), spec["run"][1]))
        args[[c for _n, c, _p, _i in io].index("save-sdata-dir")] = step_dir
        resolved["save-sdata-dir"] = step_dir
    # the step runs on the cells it names: the named run is (re)loaded when another run of
    # the tool - or none - is in memory
    _batch_ensure_seg(resolved.get("seg-source") or ctx["seg-source"], log)
    n_before = len(provenance._prov_recent)
    last_id = provenance._prov_recent[-1]["id"] if provenance._prov_recent else None
    t0 = _time.time()
    _res_before = _batch_step_result(spec)[2]
    job, out = _batch_run_callback(spec["run"][0], spec["run"][1], args, step["kind"], log)
    status, message, res = _batch_step_result(spec)
    if spec.get("sync") and res is _res_before:
        # the callback returned without computing (a missing input): the result in the
        # state is an earlier step's, not this one's
        raise _BatchError(_batch_text(out) or "the step produced no result")
    if job is None and not spec.get("sync"):
        raise _BatchError(_batch_text(out) or "the step did not start")
    if job is not None and job.state != "done":
        raise _BatchError(str(message or job.error or job.step or "the job failed")[:2000])
    if spec.get("sync") and res is None:
        raise _BatchError(str(message or _batch_text(out) or "no result")[:2000])
    if job is not None and status == "error":
        raise _BatchError(str(message)[:2000])
    recs = list(provenance._prov_recent)
    if last_id is not None and any(r["id"] == last_id for r in recs):
        recs = recs[[r["id"] for r in recs].index(last_id) + 1:]
    elif last_id is not None:
        recs = recs[-max(0, len(recs) - n_before):] if len(recs) > n_before else recs[-1:]
    files = _batch_write_outputs(step, spec, resolved, ctx, res, step_dir, cfg, log)
    with open(os.path.join(step_dir, "status.txt"), "w", encoding="utf-8") as fh:
        fh.write(str(message or _batch_text(out) or "") + "\n")
    with open(os.path.join(step_dir, "provenance.json"), "w", encoding="utf-8") as fh:
        json.dump(recs, fh, ensure_ascii=False, indent=1)
    first_line = str(message or _batch_text(out) or "").splitlines()[0][:300] if (message or out) else ""
    return {"index": step["index"], "kind": step["kind"], "name": step["name"], "status": "done",
            "seconds": round(_time.time() - t0, 2), "message": first_line, "dir": step_dir,
            "files": [os.path.relpath(f, out_dir) for f in files] + [
                os.path.relpath(os.path.join(step_dir, f), out_dir) for f in ("status.txt", "provenance.json")],
            "provenance_ids": [r["id"] for r in recs], "_records": recs,
            "resolved": {k: provenance._prov_plain(v) for k, v in resolved.items()}}


def _batch_run(cfg: dict, dry_run: bool = False, log=None) -> dict:
    """Run (or, with dry_run, only validate) a config in this process. Returns the run
    summary; summary["ok"] is False when a step failed or the validation found errors."""
    log = log or (lambda s: print(s, flush=True))
    cfg = _batch_norm_config(cfg)
    _batch_check_data(cfg)
    _batch_setup_samples(cfg, log)
    ctx = _batch_context(cfg)
    summary = {"ok": True, "dry_run": bool(dry_run), "data": cfg["data"]["path"], "steps": [],
               "errors": [], "warnings": [], "kaleido": _batch_kaleido()}
    # The run seg_source names is loaded before anything is checked or computed - in a dry
    # run too: its labels, clusters and groups are what the checks have to be made on.
    _batch_ensure_seg(ctx["seg-source"], log)
    errors, warns = _batch_validate(cfg, log)
    _batch_ensure_seg(ctx["seg-source"], log)      # the checks of a step may have loaded another run
    summary["errors"], summary["warnings"] = errors, warns
    for w in warns:
        log(f"  warning: {w}")
    for e in errors:
        log(f"  error: {e}")
    if dry_run:
        summary["ok"] = not errors
        log(f"Dry run: {len(cfg['steps'])} step(s), {len(errors)} error(s), {len(warns)} warning(s). "
            "Nothing was computed.")
        return summary
    if errors:
        # the steps still run (each fails on its own, see continue_on_error), but a run
        # whose config did not pass its checks is never reported as ok
        summary["ok"] = False
    out_dir = os.path.abspath(os.path.expanduser(str(cfg.get("output_dir") or "batch_output")))
    os.makedirs(out_dir, exist_ok=True)
    summary["output_dir"] = out_dir
    records = []
    for step in cfg["steps"]:
        log(f"[{step['index']}/{len(cfg['steps'])}] {step['name']} ({step['kind']})")
        t0 = _time.time()
        try:
            info = _batch_run_step(step, cfg, ctx, out_dir, log)
            records += info.pop("_records")
            log(f"    done in {info['seconds']:.1f} s: {info['message']}")
        except Exception as exc:
            if not isinstance(exc, _BatchError):
                traceback.print_exc()
            msg = str(exc) or type(exc).__name__
            info = {"index": step["index"], "kind": step["kind"], "name": step["name"], "status": "failed",
                    "seconds": round(_time.time() - t0, 2), "message": msg, "files": [], "provenance_ids": []}
            summary["ok"] = False
            log(f"    FAILED: step {step['index']} ({step['name']}): {msg}")
            summary["steps"].append(info)
            if not cfg["continue_on_error"]:
                log(f"Stopped at step {step['index']} of {len(cfg['steps'])} "
                    f"(continue_on_error: true runs the remaining steps).")
                break
            continue
        summary["steps"].append(info)
    # the record of the run: provenance JSON and methods text of what this run did
    with open(os.path.join(out_dir, "provenance.json"), "w", encoding="utf-8") as fh:
        fh.write(provenance._prov_export_json(records))
    with open(os.path.join(out_dir, "methods.md"), "w", encoding="utf-8") as fh:
        fh.write(provenance._prov_methods_md(records))
    summary["n_failed"] = sum(s["status"] == "failed" for s in summary["steps"])
    summary["n_done"] = sum(s["status"] == "done" for s in summary["steps"])
    summary["n_not_run"] = len(cfg["steps"]) - len(summary["steps"])
    if not summary["kaleido"] and cfg.get("figures", True):
        summary["figures_note"] = ("kaleido is not installed: figures were written as .html and "
                                   ".plotly.json, not as images")
    with open(os.path.join(out_dir, "batch_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
    log(f"Batch finished: {summary['n_done']} step(s) done, {summary['n_failed']} failed, "
        f"{summary['n_not_run']} not run. Output: {out_dir} (provenance.json, methods.md, batch_summary.json)."
        + (f" {summary['figures_note']}." if summary.get("figures_note") else ""))
    return summary


def _batch_main(argv=None) -> int:
    """Entry point of ``python -m xenium_app.batch``. Exit code 0: every step done (or
    the dry run found no error); 1: a step failed or the dry run found errors (with
    continue_on_error the remaining steps still run; the exit code stays 1); 2: the config
    could not be used."""
    ap = argparse.ArgumentParser(
        prog="python -m xenium_app.batch",
        description="Run analyses of the Xenium app from a config file, without a browser.")
    ap.add_argument("config", nargs="?",
                    help="config file (.json, or .yaml / .yml when PyYAML is installed)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the config against the data (columns, cell types, genes) without computing")
    ap.add_argument("--list-steps", action="store_true",
                    help="print every step kind with its parameters and their defaults, then exit")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if args.list_steps:
        print(_batch_list_steps(), flush=True)
        return 0
    if not args.config:
        ap.print_usage()
        return 2
    try:
        cfg = _batch_read_config(args.config)
        summary = _batch_run(cfg, dry_run=args.dry_run)
    except _BatchError as exc:
        print(f"Batch error: {exc}", flush=True)
        return 2
    return 0 if summary["ok"] else 1


# Started with ``python -m xenium_app.batch``: this file then runs as the main module while
# its import spec keeps its real name (written this way, not as the usual name test, so the
# server's entry point in xenium_explorer.py stays the only such block of the app). The
# package did not import this module under its own name in that case (see
# xenium_app/__init__.py): register it, so the one namespace of the app (compat) sees its
# globals, then run.
if __spec__ is not None and __name__ != __spec__.name:
    sys.modules.setdefault(__spec__.name, sys.modules[__name__])
    sys.exit(_batch_main())
