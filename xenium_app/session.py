"""Session save / restore: the settings and choices of a session as ONE JSON file.

A session file holds settings, never results: an analysis that is run again after a
restore reads its result back from the caches the app already keeps. Restore never
unpickles, evaluates or imports anything from the file, never opens a path the file
names, and applies a value only after it passed the allow-list below (known key, type,
range / fixed choices, and for data-dependent values: present in the loaded data now).

# ─── State inventory ──────────────────────────────────────────────────────────
# What defines a session, where it lives, and whether the session file holds it.
#
# | State                              | Owner / where                              | Saved | How restored / why not                                        |
# |------------------------------------|--------------------------------------------|-------|---------------------------------------------------------------|
# | Primary dataset                    | core.DATA (data_dir / sdata_path)          | id    | identity only (key, name, cell count): compared, never loaded |
# | Extra samples                      | dataview.EXTRA_DATASETS                    | id    | key + name + cell count; reloaded through cb_data.add_sample   |
# |                                    |                                            |       | only when the path is one the Load Sample picker itself lists, |
# |                                    |                                            |       | else reported as "needs loading"                              |
# | Sample design (conditions, donors, | stats._design_state; already persisted     | yes   | embedded; applied with stats._design_set (which rewrites the  |
# |   batches, condition names)        |   per sample set (design_<tag>.json)       |       | app's own design file, so both agree)                         |
# | Segmentation source                | control seg-source; core._*_state results  | yes   | the value and the run id; the cached run is loaded through the |
# |                                    |                                            |       | callbacks of the Segmentation dropdown; missing run: reported |
# | Counts mode                        | control counts-mode, core._active_counts_mode | yes | cb_spatial.sync_counts_mode; "corrected" only when available  |
# | Active cluster set                 | control cluster-method                     | yes   | must be a set of clustersets._cs_options now; else reported   |
# | Colour by, gene, view settings     | controls color-by, gene-selector, point-*, | yes   | control values (colour by is checked against what is loaded)  |
# |                                    |   morph-*, boundary-toggles, renderer-mode |       |                                                               |
# | Annotation label choice            | controls annot-source, annot-model, …,     | yes   | control values; label sources / cell types are applied only   |
# |                                    |   *-labels, cell-type pickers              |       | when the labels are loaded now (else "needs the labels")      |
# | Per-analysis parameters            | the numeric / choice / text controls of    | yes   | control values, each checked against the range / the options  |
# |                                    |   every tab and dialog (_SESSION_NUM, …)   |       | of its control in the layout                                  |
# | ROIs                               | core._roi_state; already persisted per     | yes   | embedded; added through regions (cache file + cell metadata + |
# |                                    |   dataset (rois_<tag>.json)                |       | store write, as Save ROI does); existing ROIs are kept        |
# | ROI overlay visible                | control roi-show-toggle, _roi_state.show   | yes   | cb_spatial.toggle_roi_show                                    |
# | Gene sets                          | analysis._geneset_state; persisted         | yes   | embedded; merged, written with analysis._gs_save              |
# | Saved DE groups (outlines)         | analysis._de_state["groups"]; persisted    | yes   | embedded; merged, written with analysis._de_groups_save       |
# | Subset filters                     | core.DATA["_subset_desc"] (subset())       | yes*  | re-applied with subsetting.subset(); *not a subset given by   |
# |                                    |                                            |       | explicit cell ids or by more than 49 values (the app records  |
# |                                    |                                            |       | only their number): reported when saving                      |
# | Cluster sets computed in the app   | clustersets (cached per sample / run)      | no    | results; clustersets._cs_restore brings them back from cache  |
# | Cell-type labels, SPLIT counts,    | core._annot_state, DATA[...], *_state      | no    | results; the app autoloads them from its caches               |
# |   imputed genes, niches, UMAP      |                                            |       |                                                               |
# | Analysis results and status lines  | *_state dicts, *-done stores               | no    | results; re-run (cache hit)                                   |
# | Version / trigger stores           | dcc.Store *-version, *-done, *-request     | no    | counters of this browser page, meaningless in another         |
# | Selection, lasso, measure, pending | stores selected-cell, webgl-*, roi-pending,| no    | transient interaction state                                   |
# |   ROI outline, viewport            |   measure-store, spatial-relayout          |       |                                                               |
# | Density overlay on/off             | store density-store                        | no    | follows the density controls, which are saved                 |
# | Open tab / group                   | plots-tabs.active_tab, store nav-last      | no    | navigation; nav-last is browser session storage already       |
# | Path fields (reference .rds,       | annot-rds-path, spage-rds-path,            | no    | a path taken from a file must not reach a control that a      |
# |   CellNEST folder, sample path,    |   split-rds-path, cellnest-folder-path,    |       | later click reads or writes                                   |
# |   SpatialData output)              |   sample-custom-path, save-sdata-*         |       |                                                               |
# | CellNEST results and their filters | core._cellnest_state, cellnest-* pickers   | no    | loaded from a folder path (see above); display numbers saved  |
# | .h5ad mapping dialog               | h5ad-* controls                            | no    | part of how a dataset is converted, not of a session          |
# | Dialog scratch fields              | roi-name-input, roi-op-*, de-group-name,   | no    | text of a dialog that is not an action yet                    |
# |                                    |   geneset-name, sample-picker, prov-kind   |       |                                                               |
# | Image-segmentation back end /      | iseg-backend, iseg-channel(-cyto),         | no    | options are built from the images / installed back ends when  |
# |   channels, patch overlays, layer  |   sdata-show-*, txf-layer                  |       | the dialog opens or a job finishes                            |
# | REPL input, REPL gate, host bind,  | repl-input; XENIUM_ENABLE_REPL,            | never | environment of the server process; no key of a session file   |
# |   port, cache dir, data dir        |   XENIUM_HOST, PORT, XENIUM_CACHE_DIR      |       | reaches them                                                  |
#
# Controls have no Dash ``persistence``: the values above live in the browser page and in
# the module globals named, which is why they are lost with the server (or the page).
"""

import os
import re
import sys
import json
import math
import base64
import secrets
import threading
import datetime as _dt

import dash_bootstrap_components as dbc
from dash import dcc, html

from . import core, jobs, caching, dataview, regions, subsetting, analysis, stats, controls
from . import clustersets, sdata_store, provenance


_SESSION_FORMAT = "xenium-explorer-session"
_SESSION_VERSION = 1
_SESSION_MAX_BYTES = 8_000_000       # largest file read (ROI outlines are the bulk of a session)
_SESSION_TEXT_MAX = 20_000           # longest text-area value
_SESSION_AUTOSAVE_MS = 60_000        # the page sends its control values this often
_SESSION_PROCESS = secrets.token_hex(6)     # tells the autosave of this server run from an earlier one
_session_lock = threading.RLock()
_session_auto: dict = {"baseline": None, "written": None}   # signatures of the autosave (see _session_autosave)
_SESSION_AUTO_PAGES = 32          # page loads whose starting state is remembered
_session_memo: dict = {}


class _SessionError(Exception):
    """A session file that cannot be used at all (message for the user)."""


# ─── The allow-list of controls ───────────────────────────────────────────────
# Numbers: range taken from the control (min / max of the input or slider).
_SESSION_NUM = """
    sdata-qv sdata-patch-width sdata-patch-min-tx
    baysor-scale baysor-scale-std baysor-min-mol baysor-n_clusters baysor-prior-conf
    proseg-voxel-size proseg-nthreads proseg-samples proseg-recorded-samples
    proseg-nuclear-reassign-prob proseg-prior-seg-prob
    iseg-diameter iseg-expand iseg-min-tx iseg-patch iseg-thr iseg-flow iseg-cellprob
    annot-rctd-cores annot-rctd-umi-min annot-rctd-umi-min-sigma spage-npv
    niche-n-clusters niche-n-neighbors split-max-cores split-min-umi split-min-umi-sigma
    point-size point-opacity colorbar-min colorbar-max morph-brightness morph-opacity
    umap-plot-w umap-plot-h abundance-plot-w abundance-plot-h comp-plot-w comp-plot-h
    sfree-n-markers sfree-hole sfree-nnull sfree-plot-w sfree-plot-h
    cooccurrence-cbar-min cooccurrence-cbar-max cooccurrence-plot-w cooccurrence-plot-h
    enrichment-cbar-min enrichment-cbar-max enrichment-plot-w enrichment-plot-h
    interaction-min-freq interaction-plot-w interaction-plot-h
    chord-min-freq chord-plot-w chord-plot-h niche-umap-plot-w niche-umap-plot-h
    cellnest-top-n cellnest-edge-width cellnest-plot-w cellnest-plot-h
    cellnest-nichenest-top-n cellnest-downstream-hops roi-area-plot-w roi-area-plot-h
    de-pb-min-cells de-pb-min-count prox-near prox-far prox-nrand
    contact-radius contact-min-cells contact-nshuffle de-plot-w de-plot-h
    density-opacity density-max moran-k
    lrd-radius lrd-tile lrd-own lrd-floor lrd-min-cells lr-k lr-nperm lrd-top-n
    lrd-plot-w lrd-plot-h lr-top-n lr-plot-w lr-plot-h
    ctl-bin ctl-close ctl-hole ctl-radius ctl-core-nb ctl-core-frac ctl-core-edge
    spill-n-markers spill-own-frac spill-plot-w spill-plot-h segqc-radius segqc-plot-w segqc-plot-h
    coloc-radius coloc-nperm coloc-plot-w coloc-plot-h
    agg-radius agg-nmin agg-link agg-nperm agg-alpha agg-minnb agg-plot-w agg-plot-h
    txf-k txf-bin txf-mintx txf-maxfit txf-seed txf-plot-w txf-plot-h
    bk-lambda bk-resolution bk-kgeom bk-maxgenes bk-npcs bk-kexpr bk-seed bk-plot-w bk-plot-h
    jc-res jc-nhvg jc-npcs jc-nn jc-seed jc-plot-w jc-plot-h
""".split()
# Numbers that are coordinates of the loaded data (the region of a resegmentation run).
_SESSION_BOX = """
    baysor-xmin baysor-xmax baysor-ymin baysor-ymax proseg-xmin proseg-xmax proseg-ymin proseg-ymax
    iseg-xmin iseg-xmax iseg-ymin iseg-ymax
""".split()
# One of the fixed options of the control.
_SESSION_CHOICE = """
    baysor-prior-type proseg-prior-type annot-source annot-model annot-rctd-mode
    morph-zlevel renderer-mode comp-transform comp-test sfree-area cellnest-color-by
    de-mode de-lfc-axis density-bin lr-mode lrd-rates lrd-level agg-hull
    txf-method txf-shape bk-preset jc-integrate
""".split()
# A list out of the fixed options of the control (checklists).
_SESSION_MULTI = """
    baysor-use-patches baysor-use-prior iseg-gpu split-purify-singlets boundary-toggles
    morph-enable morph-channels roi-show-toggle interaction-normalize chord-normalize
    cellnest-options de-pb-shrink prox-length de-leak-gate density-enable agg-outline-toggle
    bk-agf bk-log1p jc-umap
""".split()
# Text: control -> (longest value, pattern the whole value has to match or None for
# "printable text and line breaks").
_SESSION_TEXT = {
    "proseg-schedule": (200, r"[0-9, ]*"), "sfree-bins": (200, r"[0-9., ]*"),
    "annot-label-col": (64, r"[A-Za-z0-9_. -]*"), "split-label-col": (64, r"[A-Za-z0-9_. -]*"),
    "spage-genes": (_SESSION_TEXT_MAX, None), "geneset-genes": (_SESSION_TEXT_MAX, None),
    "sfree-markers": (_SESSION_TEXT_MAX, None), "spill-markers": (_SESSION_TEXT_MAX, None),
    "segqc-sets": (_SESSION_TEXT_MAX, None), "lr-pairs": (_SESSION_TEXT_MAX, None),
}
# Values that name something of the loaded data: control -> (role, control holding the
# label source for a cell-type role, extra values that are always valid).
_SESSION_DYNAMIC = {
    "seg-source": ("seg", None, ()), "counts-mode": ("counts", None, ()),
    "cluster-method": ("cluster_set", None, ()), "color-by": ("color_by", None, ()),
    "gene-selector": ("gene", None, ()), "webgl-tx-genes": ("genes", None, ()),
    "density-gene": ("gene", None, ("__all__",)),
    "comp-category": ("label_kind", None, ()), "sfree-labels": ("label_kind", None, ()),
    "prox-labels": ("label_kind", None, ()), "lr-labels": ("label_kind", None, ()),
    "ctl-labels": ("label_kind", None, ()), "coloc-labels": ("label_kind", None, ()),
    "agg-labels": ("label_kind", None, ()), "txf-labels": ("label_kind", None, ()),
    "segqc-method": ("annot_method", None, ()),
    "sfree-type": ("cell_type", "sfree-labels", ()),
    "prox-target": ("cell_type", "prox-labels", ()), "prox-anchor": ("cell_type", "prox-labels", ()),
    "prox-source": ("cell_type", "prox-labels", ("auto", "none")),
    "ctl-dominant": ("cell_type", "ctl-labels", ("auto",)),
    "spill-source": ("cell_type", "ctl-labels", ("auto",)),
    "coloc-placebo": ("cell_type", "coloc-labels", ()),
    "agg-classes": ("cell_types", "agg-labels", ()),
    "de-group-a": ("de_group", None, ()), "de-group-b": ("de_group", None, ()),
    "de-pb-population": ("population", None, ()), "geneset-select": ("geneset", None, ()),
}
# Fixed-option controls whose options still belong to the loaded data (its morphology
# images, its size): not applied when the session was saved for other data.
_SESSION_BOUND_VIEW = ("morph-enable", "morph-channels", "morph-zlevel", "renderer-mode")
# Controls with a value that a session file never holds, each with the reason (shown when
# a file names one). tests/test_session.py checks that every value control of the layout
# is either allowed above or listed here.
_SESSION_PATH_WHY = "a path is never taken from a session file"
_SESSION_DIALOG_WHY = "scratch field of a dialog, not a setting"
_SESSION_BUILT_WHY = "its options are built from the loaded images / results when it is used"
_SESSION_NOT_SAVED = {
    **{c: _SESSION_PATH_WHY for c in (
        "annot-rds-path", "spage-rds-path", "split-rds-path", "cellnest-folder-path",
        "sample-custom-path", "sample-picker", "save-sdata-dir", "save-sdata-name")},
    **{c: _SESSION_DIALOG_WHY for c in (
        "roi-name-input", "roi-class-input", "roi-op-a", "roi-op", "roi-op-b", "roi-op-name",
        "roi-op-class", "de-group-name", "geneset-name", "prov-kind", "save-sdata-roi-only",
        "design-name-a", "design-name-b")},
    **{c: _SESSION_BUILT_WHY for c in (
        "iseg-backend", "iseg-channel", "iseg-channel-cyto", "sdata-show-roi", "sdata-show-patches",
        "txf-layer", "niche-label-method", "cellnest-dataset-dropdown", "cellnest-strip-suffix",
        "cellnest-lr-pair", "cellnest-annot-col", "cellnest-filter-ligand", "cellnest-filter-receptor",
        "cellnest-filter-sender", "cellnest-filter-receiver", "cellnest-downstream-receptor",
        "cellnest-downstream-tf-select", "cellnest-overlay-enable", "segqc-source")},
    **{c: "part of converting an .h5ad into a dataset, not of a session" for c in (
        "h5ad-x", "h5ad-y", "h5ad-counts", "h5ad-section", "h5ad-celltypes", "h5ad-clusters",
        "h5ad-condition", "h5ad-donor", "h5ad-cond-a", "h5ad-cond-b", "h5ad-noninteger")},
    "webgl-tool": "the pointer tool of the moment",
    "repl-input": "the REPL is never reachable from a session file",
}
# Names a file may try that touch the server process: answered by name, never applied.
_SESSION_FORBIDDEN_RE = re.compile(r"repl|host|port|bind|env|cache_dir|data_dir|xenium_", re.I)

_SESSION_TOP_KEYS = ("format", "version", "saved_at", "kind", "app", "process", "dataset", "state", "notes")
_SESSION_STATE_KEYS = ("controls", "seg_run_id", "design", "rois", "gene_sets", "de_groups", "subset")
_SESSION_SUBSET_NUM = ("min_expr", "max_expr", "min_transcripts", "max_transcripts", "min_cell_area",
                       "max_cell_area", "min_nucleus_area", "max_nucleus_area")
_SESSION_SEG_RE = re.compile(r"xenium|(baysor|proseg|imgseg)(:[A-Za-z0-9]{1,32})?")
_SESSION_TIME_RE = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\+00:00|Z)?")


def _session_controls() -> dict:
    """control id -> (kind, bound to the loaded data) of every control a session holds."""
    if "controls" not in _session_memo:
        out = {c: ("num", False) for c in _SESSION_NUM}
        out.update({c: ("num", True) for c in _SESSION_BOX})
        out.update({c: ("choice", False) for c in _SESSION_CHOICE})
        out.update({c: ("multi", False) for c in _SESSION_MULTI})
        out.update({c: ("text", False) for c in _SESSION_TEXT})
        out.update({c: ("dynamic", True) for c in _SESSION_DYNAMIC})
        out.update({c: (out[c][0], True) for c in _SESSION_BOUND_VIEW})
        _session_memo["controls"] = out
    return _session_memo["controls"]


def _session_module(name):
    """A module of the app that this library module must not import (the callbacks)."""
    return sys.modules.get(f"{__package__}.{name}")


def _session_components() -> dict:
    """Component id -> component of the app layout (ranges, options and defaults)."""
    if "components" not in _session_memo:
        layout_mod = _session_module("app_layout")
        if layout_mod is None or getattr(layout_mod, "app", None) is None:
            return {}
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
        walk(layout_mod.app.layout)
        _session_memo["components"] = out
    return _session_memo["components"]


def _session_default(cid):
    return getattr(_session_components().get(cid), "value", None)


def _session_options(cid) -> list:
    opts = getattr(_session_components().get(cid), "options", None) or []
    return [o.get("value") if isinstance(o, dict) else o for o in opts]


# ─── Checks of single values (no data, no file system) ────────────────────────
def _session_show(v, n: int = 60) -> str:
    """A value of a file as it is quoted in a message: short, one line, printable."""
    s = v if isinstance(v, str) else json.dumps(v, default=str)[:n + 20]
    s = "".join(ch if ch.isprintable() else "?" for ch in s)
    return repr(s[:n] + ("…" if len(s) > n else ""))


def _session_gene_lines(text: str) -> list:
    """The genes of a gene-list text, one per non-empty line (as fill_geneset_form writes
    a stored set)."""
    return [ln.strip() for ln in str(text or "").replace("\r", "\n").split("\n") if ln.strip()]


def _session_is_num(v) -> bool:
    """A finite JSON number. Total: an integer of any size is answered, never raised on
    (json accepts integers of hundreds of digits, which math.isfinite cannot convert)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    if isinstance(v, int):
        return abs(v) < _SESSION_NUM_MAX
    return math.isfinite(v)


_SESSION_NUM_MAX = 10 ** 300     # no control, coordinate or threshold of the app is near it


def _session_is_text(v, n: int, multiline: bool = False) -> bool:
    return (isinstance(v, str) and len(v) <= n
            and all(ch.isprintable() or (multiline and ch in "\n\r\t") for ch in v))


def _session_name_ok(v, n: int = 100) -> bool:
    return _session_is_text(v, n) and bool(v.strip())


def _session_polygon(v, max_points: int = 50_000):
    """[[x, y], …] of finite numbers, or None."""
    if not isinstance(v, list) or not (3 <= len(v) <= max_points):
        return None
    out = []
    for p in v:
        if not (isinstance(p, (list, tuple)) and len(p) == 2 and all(_session_is_num(c) and abs(c) < 1e9 for c in p)):
            return None
        out.append([float(p[0]), float(p[1])])
    return out


def _session_check_control(cid, value) -> tuple:
    """(True, the value as it is applied) or (False, why not). Type, range, pattern and
    fixed options only; whether a data-dependent value exists is checked at restore."""
    spec = _session_controls().get(cid)
    if spec is None:
        return False, _SESSION_NOT_SAVED.get(cid) or "not a control a session holds"
    kind = spec[0]
    comp = _session_components().get(cid)
    if kind == "num":
        if value is None or value == "":
            return True, None
        if not _session_is_num(value):
            return False, "not a number"
        lo, hi = getattr(comp, "min", None), getattr(comp, "max", None)
        lo = lo if _session_is_num(lo) else -1e9
        hi = hi if _session_is_num(hi) else 1e9
        if not (lo <= value <= hi):
            return False, f"outside {lo:g} … {hi:g}"
        return True, value
    if kind == "choice":
        opts = _session_options(cid)
        if any(type(o) is type(value) and o == value for o in opts) or (value is None and _session_default(cid) is None):
            return True, value
        return False, "not one of the options of this control"
    if kind == "multi":
        opts = _session_options(cid)
        if value is None:
            return True, []
        if not isinstance(value, list) or len(value) > len(opts) or len(set(map(str, value))) != len(value):
            return False, "not a list of options"
        if not all(any(type(o) is type(v) and o == v for o in opts) for v in value):
            return False, "holds a value that is not an option of this control"
        return True, list(value)
    if kind == "text":
        n, pattern = _SESSION_TEXT[cid]
        if value is None:
            return True, None
        if not _session_is_text(value, n, multiline=pattern is None):
            return False, f"not text of at most {n} characters"
        if pattern is not None and not re.fullmatch(pattern, value):
            return False, "holds characters this field does not take"
        return True, value
    role = _SESSION_DYNAMIC[cid][0]
    if role in ("genes", "cell_types"):
        if value is None:
            return True, []
        if not isinstance(value, list) or len(value) > 500 or not all(_session_name_ok(v, 200) for v in value):
            return False, "not a list of names"
        return True, list(value)
    if value is None or value == "":
        return True, None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return False, "not a name"
    if isinstance(value, str) and not _session_name_ok(value, 200):
        return False, "not a name"
    if role == "seg" and not _SESSION_SEG_RE.fullmatch(str(value)):
        return False, "not a segmentation source (xenium, or <tool>:<run tag>)"
    if role == "counts" and value not in ("original", "corrected"):
        return False, "not 'original' or 'corrected'"
    return True, value


# ─── Reading and validating a file ────────────────────────────────────────────
def _session_no_constant(name):
    raise ValueError(f"{name} is not JSON")


def _session_no_duplicates(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"the key {_session_show(k)} appears twice")
        out[k] = v
    return out


def _session_parse(text) -> dict:
    """The JSON document of a session file (text or bytes). _SessionError when it is not
    a session file this version of the app reads."""
    if isinstance(text, (bytes, bytearray)):
        if len(text) > _SESSION_MAX_BYTES:
            raise _SessionError(f"the file is larger than {_SESSION_MAX_BYTES // 1_000_000} MB: not a session file")
        try:
            text = bytes(text).decode("utf-8")
        except UnicodeDecodeError:
            raise _SessionError("the file is not UTF-8 text: not a session file") from None
    if not isinstance(text, str) or len(text) > _SESSION_MAX_BYTES:
        raise _SessionError("not a session file (too large or not text)")
    try:
        doc = json.loads(text, parse_constant=_session_no_constant, object_pairs_hook=_session_no_duplicates)
    except RecursionError:
        raise _SessionError("the file is nested too deeply: not a session file") from None
    except ValueError as exc:
        raise _SessionError(f"the file is not valid JSON ({str(exc)[:120]})") from None
    if not isinstance(doc, dict) or doc.get("format") != _SESSION_FORMAT:
        raise _SessionError(f"not a session file of this app (no \"format\": \"{_SESSION_FORMAT}\")")
    version = doc.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise _SessionError("the session file has no format version")
    if version > _SESSION_VERSION:
        raise _SessionError(f"the session file has format version {version}; this app reads version "
                            f"{_SESSION_VERSION} and older. Restore it with the newer app that wrote it.")
    if version < 1:
        raise _SessionError(f"the session file has the unknown format version {version}")
    return doc


def _session_validate(doc: dict) -> tuple:
    """(clean, ignored, invalid): *clean* holds only allow-listed keys with values of the
    right type and range; *ignored* names the unknown keys, *invalid* the known keys whose
    value was refused (both as text for the report). Nothing is applied here."""
    ignored, invalid = [], []

    def unknown(where, d, known):
        for k in d:
            if k not in known:
                tail = " (the server's settings are never read from a session file)" \
                    if _SESSION_FORBIDDEN_RE.search(str(k)) else ""
                ignored.append(f"{where}{_session_show(str(k))}{tail}")

    unknown("", doc, _SESSION_TOP_KEYS)
    clean = {"version": doc["version"], "kind": doc.get("kind") if doc.get("kind") in
             ("saved", "auto", "download", "batch") else "file",
             "saved_at": doc["saved_at"] if isinstance(doc.get("saved_at"), str)
             and _SESSION_TIME_RE.fullmatch(doc["saved_at"]) else "",
             "process": doc["process"] if isinstance(doc.get("process"), str)
             and re.fullmatch(r"[0-9a-f]{1,32}", doc["process"]) else ""}
    # dataset identity: text and counts that are only ever compared
    ds = doc.get("dataset")
    cds = {"key": "", "name": "", "n_cells": None, "samples": []}
    if isinstance(ds, dict):
        unknown("dataset.", ds, ("key", "name", "n_cells", "samples"))
        if _session_is_text(ds.get("key"), 4096):
            cds["key"] = ds["key"]
        if _session_is_text(ds.get("name"), 300):
            cds["name"] = ds["name"]
        if isinstance(ds.get("n_cells"), int) and not isinstance(ds.get("n_cells"), bool) and ds["n_cells"] >= 0:
            cds["n_cells"] = ds["n_cells"]
        rows = ds.get("samples")
        if isinstance(rows, list) and len(rows) <= 1000:
            for r in rows:
                if (isinstance(r, dict) and _session_is_text(r.get("key"), 4096) and _session_is_text(r.get("name"), 300)
                        and isinstance(r.get("n_cells"), int) and not isinstance(r.get("n_cells"), bool)
                        and r["n_cells"] >= 0):
                    cds["samples"].append({"key": r["key"], "name": r["name"], "n_cells": r["n_cells"]})
                else:
                    invalid.append("dataset.samples: an entry is not {key, name, n_cells}")
                    cds["samples"] = None
                    break
        elif rows is not None:
            invalid.append("dataset.samples: not a list")
            cds["samples"] = None
    else:
        invalid.append("dataset: missing (the session cannot be matched to the loaded data)")
        cds["samples"] = None
    clean["dataset"] = cds

    st = doc.get("state")
    if not isinstance(st, dict):
        invalid.append("state: missing")
        st = {}
    unknown("state.", st, _SESSION_STATE_KEYS)
    # controls
    cc = {}
    ctl = st.get("controls")
    if isinstance(ctl, dict):
        for cid, val in ctl.items():
            ok, res = _session_check_control(cid, val)
            if ok:
                cc[cid] = res
            elif cid in _session_controls():
                invalid.append(f"control {cid}: {_session_show(val, 30)} {res}")
            else:
                tail = "" if cid in _SESSION_NOT_SAVED else \
                    (" (the server's settings are never read from a session file)"
                     if _SESSION_FORBIDDEN_RE.search(str(cid)) else "")
                ignored.append(f"control {_session_show(str(cid))}: {res}{tail}")
    elif ctl is not None:
        invalid.append("state.controls: not a mapping")
    clean["controls"] = cc
    clean["seg_run_id"] = st["seg_run_id"] if isinstance(st.get("seg_run_id"), str) \
        and re.fullmatch(r"[A-Za-z0-9_.-]{0,64}", st["seg_run_id"]) else ""
    # design
    clean["design"] = None
    d = st.get("design")
    if isinstance(d, dict):
        unknown("state.design.", d, ("names", "samples"))
        names = d.get("names") if isinstance(d.get("names"), dict) else {}
        cn = {c: names[c].strip() for c in ("A", "B") if _session_name_ok(names.get(c), 60)}
        rows, bad = [], 0
        for r in (d.get("samples") if isinstance(d.get("samples"), list) else [])[:1000]:
            if (isinstance(r, dict) and _session_is_text(r.get("key"), 4096) and r.get("cond") in ("A", "B", "")
                    and _session_is_text(r.get("donor", ""), 60) and _session_is_text(r.get("batch", ""), 60)):
                rows.append({"key": r["key"], "cond": r["cond"], "donor": r.get("donor", ""),
                             "batch": r.get("batch", "")})
            else:
                bad += 1
        if bad:
            invalid.append(f"state.design: {bad} sample row(s) are not {{key, cond A / B / \"\", donor, batch}}")
        if len(cn) == 2 and cn["A"] == cn["B"]:
            invalid.append("state.design.names: the two conditions have the same name")
            cn = {}
        clean["design"] = {"names": cn, "samples": rows}
    elif d is not None:
        invalid.append("state.design: not a mapping")
    # ROIs
    rois = []
    raw = st.get("rois")
    if isinstance(raw, list) and len(raw) <= 500:
        for r in raw:
            poly = _session_polygon(r.get("polygon_xy")) if isinstance(r, dict) else None
            if poly is None or not _session_name_ok(r.get("name")) or not _session_name_ok(r.get("cls")):
                invalid.append("state.rois: an entry is not {name, cls, polygon_xy of at least 3 finite points}")
                continue
            color = r.get("color")
            rois.append({"name": r["name"].strip(), "cls": r["cls"].strip(), "polygon_xy": poly,
                         "color": color if isinstance(color, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", color) else None})
    elif raw is not None:
        invalid.append("state.rois: not a list of at most 500 ROIs")
    clean["rois"] = rois
    # gene sets
    sets = {}
    raw = st.get("gene_sets")
    if isinstance(raw, dict) and len(raw) <= 500:
        for name, genes in raw.items():
            if (_session_name_ok(name) and isinstance(genes, list) and 1 <= len(genes) <= 5000
                    and all(_session_name_ok(g, 64) for g in genes)):
                sets[name] = [str(g) for g in genes]
            else:
                invalid.append(f"state.gene_sets: {_session_show(str(name), 30)} is not a named list of gene names")
    elif raw is not None:
        invalid.append("state.gene_sets: not a mapping of at most 500 sets")
    clean["gene_sets"] = sets
    # DE groups
    groups = []
    raw = st.get("de_groups")
    if isinstance(raw, list) and len(raw) <= 200:
        for g in raw:
            poly = _session_polygon(g.get("polygon_xy")) if isinstance(g, dict) else None
            if poly is None or not _session_name_ok(g.get("name")):
                invalid.append("state.de_groups: an entry is not {name, polygon_xy}")
                continue
            groups.append({"name": g["name"], "polygon_xy": poly})
    elif raw is not None:
        invalid.append("state.de_groups: not a list of at most 200 groups")
    clean["de_groups"] = groups
    # subset
    clean["subset"] = None
    sub = st.get("subset")
    if isinstance(sub, dict):
        unknown("state.subset.", sub, ("filters", "n_cells", "restorable", "reason"))
        if sub.get("restorable") is False:
            clean["subset"] = {"restorable": False}
        else:
            f, cf, ok = sub.get("filters"), {}, isinstance(sub.get("filters"), dict)
            for k, v in (f.items() if ok else ()):
                if k in _SESSION_SUBSET_NUM and _session_is_num(v):
                    cf[k] = float(v)
                elif k in ("gene", "method") and _session_name_ok(v, 200):
                    cf[k] = v
                elif k == "cluster" and (isinstance(v, int) and not isinstance(v, bool) or isinstance(v, list)
                                         and 1 <= len(v) < 50 and all(re.fullmatch(r"-?\d{1,9}", str(x)) for x in v)):
                    cf[k] = v if isinstance(v, int) else [int(str(x)) for x in v]
                elif k == "cell_type" and (_session_name_ok(v, 200) or isinstance(v, list) and 1 <= len(v) < 50
                                           and all(_session_name_ok(x, 200) for x in v)):
                    cf[k] = v
                else:
                    ok = False
                    invalid.append(f"state.subset.filters.{_session_show(str(k), 30)}: not a filter of subset() "
                                   f"with a valid value")
            n = sub.get("n_cells")
            if ok and cf:
                clean["subset"] = {"filters": cf, "n_cells": n if isinstance(n, int) and not isinstance(n, bool) else None}
            elif not ok:
                clean["subset"] = {"restorable": False}
    elif sub is not None:
        invalid.append("state.subset: not a mapping")
    return clean, ignored, invalid


# ─── The loaded data, as a session sees it ────────────────────────────────────
def _session_dataset() -> dict:
    """Identity of the loaded data: the key the provenance log uses, and per sample its
    key, name and number of cells (of the whole sample, whatever subset is active)."""
    samples = []
    for i, ds in enumerate([core.DATA] + list(dataview.EXTRA_DATASETS)):
        df = ds.get("_df_original") if ds.get("_df_original") is not None else ds["df"]
        samples.append({"label": f"S{i}", "key": dataview._ds_key(ds), "name": dataview._ds_name(ds),
                        "n_cells": int(len(df))})
    return {"key": caching._dataset_key(), "name": dataview._ds_name(core.DATA),
            "n_cells": int(sum(s["n_cells"] for s in samples)), "samples": samples}


def _session_same_key(a, b) -> bool:
    """Two dataset keys as text (nothing is looked up on disk for a key from a file)."""
    norm = lambda s: os.path.normpath(str(s)).rstrip("/") if s else ""
    return bool(a) and bool(b) and norm(a) == norm(b)


def _session_dataset_diff(saved: dict, now: dict) -> list:
    """Why the session's data is not the loaded data ([] when it is)."""
    if saved.get("samples") is None or not saved.get("key"):
        return ["the session does not say which data it was saved for"]
    out = []
    if not _session_same_key(saved["key"], now["key"]):
        return [f"it was saved for another dataset ({_session_show(saved.get('name') or saved['key'], 80)}; "
                f"loaded: {now['name']})"]
    ss, ns = saved["samples"], now["samples"]
    if len(ss) != len(ns):
        out.append(f"it was saved with {len(ss)} sample(s), {len(ns)} are loaded")
    for i, (a, b) in enumerate(zip(ss, ns)):
        if not _session_same_key(a["key"], b["key"]):
            out.append(f"sample S{i} was {_session_show(a['name'], 60)}, it is {b['name']}")
        elif a["n_cells"] != b["n_cells"]:
            out.append(f"sample S{i} ({b['name']}) had {a['n_cells']:,} cells, it has {b['n_cells']:,}")
    return out


def _session_busy() -> list:
    """Names of the jobs running now (the list Load Sample refuses on)."""
    cbd = _session_module("cb_data")
    if cbd is not None:
        return list(cbd._running_jobs())
    return [j.name for j in jobs._jobs_list(active_only=True)]


def _session_subset_now():
    """The active subset as a session holds it: None, {"filters", "n_cells"} or
    {"restorable": False, "reason"}."""
    if "_df_original" not in core.DATA:
        return None
    desc = (core.DATA.get("_subset_desc") or [{}])[0]
    f = {k: v for k, v in desc.items() if k != "skipped"}
    if "n_cell_ids" in f:
        return {"restorable": False, "reason": "the subset was made from an explicit list of cell ids, "
                                               "which the app does not keep"}
    if any(isinstance(v, list) and len(v) >= 50 for v in f.values()):
        return {"restorable": False, "reason": "the subset names 50 or more values, of which the app "
                                               "records the first 50 only"}
    if not f:
        return {"restorable": False, "reason": "the filters of the subset were not recorded"}
    try:        # the app records a numpy integer as text: numbers as numbers, cluster ids as integers
        for k in list(f):
            if k in _SESSION_SUBSET_NUM:
                f[k] = float(f[k])
            elif k == "cluster":
                f[k] = [int(str(x)) for x in f[k]] if isinstance(f[k], list) else int(str(f[k]))
    except (TypeError, ValueError):
        return {"restorable": False, "reason": "the filters of the subset were not recorded as numbers"}
    return {"filters": f, "n_cells": int(len(core.DATA["df"]))}


# ─── Save ─────────────────────────────────────────────────────────────────────
def _session_snapshot(controls=None, kind: str = "saved") -> dict:
    """The session as a JSON document. *controls*: {control id: value} as the page holds
    them (None: the defaults of the layout, with the counts mode and the ROI switch as the
    server knows them). Values that do not pass the allow-list are left out and named
    under "notes"."""
    notes = []
    given = dict(controls) if isinstance(controls, dict) else None
    out_ctl = {}
    for cid in _session_controls():
        if given is not None and cid in given:
            val = given[cid]
        else:
            val = _session_default(cid)
            if given is None and cid == "counts-mode":
                val = core._active_counts_mode
            if given is None and cid == "roi-show-toggle":
                with core._roi_lock:
                    val = ["show"] if core._roi_state.get("show", True) else []
        ok, res = _session_check_control(cid, val)
        if ok:
            out_ctl[cid] = res
        else:
            notes.append(f"{cid} was not saved: {res}")
    seg = out_ctl.get("seg-source") or "xenium"
    rows = stats._design_samples()
    with core._roi_lock:
        rois = [{"name": str(r["name"]), "cls": str(r["cls"]), "color": r.get("color"),
                 "polygon_xy": [[float(p[0]), float(p[1])] for p in r["polygon_xy"]]}
                for r in core._roi_state["rois"]]
    with analysis._geneset_lock:
        sets = {str(k): [str(g) for g in v] for k, v in analysis._geneset_state["sets"].items()}
    with analysis._de_lock:
        groups = [{"name": str(g["name"]), "polygon_xy": [[float(p[0]), float(p[1])] for p in g["polygon_xy"]]}
                  for g in analysis._de_state["groups"]]
    subset = _session_subset_now()
    if subset is not None and subset.get("restorable") is False:
        notes.append(f"the active subset was not saved: {subset['reason']}")
    doc = {
        "format": _SESSION_FORMAT, "version": _SESSION_VERSION,
        "saved_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "kind": kind if kind in ("saved", "auto", "download", "batch") else "saved",
        "app": {"name": "Xenium Explorer", "commit": provenance._prov_git_commit()},
        "process": _SESSION_PROCESS,
        "dataset": _session_dataset(),
        "state": {
            "controls": out_ctl,
            "seg_run_id": core._seg_run_id(core._seg_result(seg)) if core._seg_tool(seg) != "xenium" else "",
            "design": {"names": stats._design_names(),
                       "samples": [{"key": r["key"], "cond": r["cond"], "donor": r["donor"], "batch": r["batch"]}
                                   for r in rows]},
            "rois": rois, "gene_sets": sets, "de_groups": groups, "subset": subset,
        },
        "notes": notes,
    }
    json.dumps(doc, allow_nan=False)          # a document that is not plain JSON is a bug: say so here
    return doc


def _session_text(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=False, allow_nan=False, indent=1)


def _session_dir() -> str:
    return os.path.join(core.CACHE_BASE_DIR, "sessions")


def _session_path(slot: str) -> str:
    """File of a slot of the loaded dataset: "saved" (Save session), "last" (the
    automatic save of a server run) or "prev" (that of the run before). The stem is the
    one of the dataset's provenance file, so 'Clean cache' leaves both in place."""
    if slot not in ("saved", "last", "prev"):
        raise ValueError(slot)
    return os.path.join(_session_dir(), f"{provenance._prov_dataset_id()}.{slot}.json")


def _session_write(doc: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(_session_text(doc))
    os.replace(tmp, path)


def _session_read_slot(slot: str):
    """The document of a slot, or None when there is none (or it is unreadable)."""
    try:
        with open(_session_path(slot), "rb") as fh:
            return _session_parse(fh.read(_SESSION_MAX_BYTES + 1))
    except (OSError, _SessionError):
        return None


def _session_summary(doc: dict) -> str:
    st = doc["state"]
    design = [r for r in st["design"]["samples"] if r["cond"]]
    parts = [f"{len(st['controls'])} control values", f"{len(doc['dataset']['samples'])} sample(s)",
             f"design: {len(design)} sample(s) assigned", f"{len(st['rois'])} ROI(s)",
             f"{len(st['gene_sets'])} gene set(s)", f"{len(st['de_groups'])} DE group(s)",
             "a subset" if (st["subset"] or {}).get("filters") else "no subset"]
    return ", ".join(parts)


def _session_save(controls=None, kind: str = "saved", record: bool = True) -> tuple:
    """Save the session. kind "saved": written to the dataset's slot under the cache
    directory; "download": only returned. Returns (document, status line)."""
    doc = _session_snapshot(controls, kind)
    where = "for download"
    if kind == "saved":
        if not caching._dataset_key():
            raise _SessionError("this dataset has no identity on disk; use Download instead")
        _session_write(doc, _session_path("saved"))
        where = "in the cache directory"
    msg = f"Session saved {where} ({_session_summary(doc)}). Settings only: no result is in the file."
    if doc["notes"]:
        msg += " Not saved: " + "; ".join(doc["notes"][:8]) + ("…" if len(doc["notes"]) > 8 else "") + "."
    if record:
        provenance._prov_record(
            "session", "Session saved", seg_source=doc["state"]["controls"].get("seg-source") or "xenium",
            params={"action": "save", "to": "cache directory" if kind == "saved" else "download",
                    "format_version": _SESSION_VERSION, "n_controls": len(doc["state"]["controls"]),
                    "n_rois": len(doc["state"]["rois"]), "n_gene_sets": len(doc["state"]["gene_sets"]),
                    "n_de_groups": len(doc["state"]["de_groups"]),
                    "subset": bool((doc["state"]["subset"] or {}).get("filters"))},
            message=msg, warnings=list(doc["notes"]), uses_design=True)
    return doc, msg


def _session_signature(doc: dict) -> str:
    return json.dumps({"dataset": doc["dataset"], "state": doc["state"]}, sort_keys=True, default=str)


def _session_autosave(controls, page=None) -> str:
    """The automatic save of the "last session" of the loaded dataset (called with the
    control values the page sends every _SESSION_AUTOSAVE_MS). *page*: the id of the page
    load that sends them (one per browser page load; None: one unnamed page).

    Nothing is written while a page still shows the state of its first call - a page that
    was only opened, reloaded or reopened sends the default control values, and must
    replace neither the session of the run before nor what this run has already saved -
    and nothing when the state did not change. Once a page has shown another state, all
    its states are saved (also a return to the first one). The first write of a run keeps
    the file of the run before as the "prev" slot. Never raises; returns what it did (""
    for nothing)."""
    try:
        if not caching._dataset_key():
            return ""
        doc = _session_snapshot(controls, "auto")
        sig = (doc["dataset"]["key"], _session_signature(doc))
        page = str(page)[:64] if page is not None else None
        with _session_lock:
            if _session_auto["baseline"] is None or _session_auto["baseline"][0] != sig[0]:
                # first call of this server run for this dataset
                _session_auto.update({"baseline": sig, "written": None, "pages": {page: [sig, False]}})
                return ""
            pages = _session_auto.setdefault("pages", {None: [_session_auto["baseline"], False]})
            mine = pages.get(page)
            if mine is None:
                # a page that was just opened: what it shows now is its starting point
                while len(pages) >= _SESSION_AUTO_PAGES:
                    pages.pop(next(iter(pages)))
                pages[page] = [sig, False]
                return ""
            if not mine[1]:
                if sig == mine[0]:
                    return ""                     # still as opened: nothing of this page to save
                mine[1] = True
            if sig == _session_auto["written"]:
                return ""
            last = _session_path("last")
            old = _session_read_slot("last")
            if old is not None and old.get("process") != _SESSION_PROCESS:
                os.replace(last, _session_path("prev"))
            _session_write(doc, last)
            _session_auto["written"] = sig
        return "saved"
    except Exception as exc:
        print(f"  session: the automatic save failed: {exc}", flush=True)
        return ""


def _session_last_slot():
    """The slot "Restore last session" reads: the automatic save of the server run before
    this one ("last" until this run has written its own, then "prev"); None without one."""
    doc = _session_read_slot("last")
    if doc is not None and doc.get("process") != _SESSION_PROCESS:
        return "last"
    return "prev" if _session_read_slot("prev") is not None else None


def _session_slots_text() -> str:
    """One line per slot of the loaded dataset, for the dialog."""
    lines = []
    for slot, label in (("saved", "Saved session"), (_session_last_slot(), "Last session (automatic)")):
        doc = _session_read_slot(slot) if slot else None
        when = str((doc or {}).get("saved_at") or "")
        when = when if _SESSION_TIME_RE.fullmatch(when) else "time not recorded"
        lines.append(f"{label}: " + (f"{when.replace('T', ' ')[:16]} UTC" if doc is not None else "none"))
    return "\n".join(lines)


# ─── Restore ──────────────────────────────────────────────────────────────────
def _session_seg_load(seg) -> "str | None":
    """Make *seg* the loaded segmentation through the callbacks of the Segmentation
    dropdown (a cached run is read, nothing is computed). None, or why it is missing."""
    if core._seg_tool(seg) == "xenium":
        return None
    batch = _session_module("batch") or sys.modules.get("__main__")
    try:
        if batch._batch_seg_ready(seg):
            return None
        batch._batch_ensure_seg(seg, lambda s: print(f"  session: {s}", flush=True))
        return None
    except Exception as exc:
        return str(exc)[:300]


def _session_has_corrected(seg) -> bool:
    alt = core._seg_result(seg, done_only=True)
    if alt is not None:
        return alt.get("split_corrected_expr") is not None
    if core.DATA.get("split_corrected_expr") is None:
        return False
    return bool(dataview._corrected_status()["all"]) if dataview.EXTRA_DATASETS else True


def _session_color_by_values(seg) -> set:
    """The Color By values that can be shown now (as the dropdown enables them)."""
    with core._annot_lock:
        with_labels = {m for m in core._ANNOT_METHODS
                       if any(k.startswith(f"labels_{m}") and v is not None for k, v in core._annot_state.items())}
    out = set()
    for v in _session_options("color-by"):
        v = str(v)
        if v.startswith("cell_type:"):
            continue
        if v == "niche" and not any("niche" in t["df"].columns for t in stats._view_tables(seg)):
            continue
        out.add(v)
    out |= {f"cell_type:{m}" for m in with_labels}
    with core._roi_lock:
        out |= {f"roi_{r['cls']}" for r in core._roi_state["rois"]}
    out |= {f"geneset:{n}" for n in analysis._geneset_names()}
    out |= {str(o["value"]) for o in analysis._view_color_extra_options()}
    return out


def _session_genes() -> set:
    return set(map(str, core.DATA.get("gene_name_to_idx") or ())) | set(map(str, dataview._panel_gene_names()))


def _session_dynamic_problem(cid, value, ctx, applied, current, memo) -> "str | None":
    """Why the data-dependent value of *cid* cannot be applied now (None: it can)."""
    role, ref, extra = _SESSION_DYNAMIC[cid]
    seg = ctx["seg-source"]
    if value is None or value == [] or (not isinstance(value, list) and value in extra):
        return None
    if role == "cluster_set":
        clustersets._cs_restore(seg)             # as the Clustering dropdown does before it lists them
        have = {str(o["value"]) for o in clustersets._cs_options(seg)}
        return None if str(value) in have else "this cluster set does not exist any more"
    if role == "color_by":
        return None if str(value) in _session_color_by_values(seg) else \
            "this colouring is not available (its labels or result are not loaded)"
    if role in ("gene", "genes"):
        genes = memo.setdefault("genes", _session_genes())
        vals = value if isinstance(value, list) else [value]
        bad = [v for v in vals if core._gene_base(v) not in genes]
        return None if not bad else f"not on the gene panel: {', '.join(_session_show(b, 20) for b in bad[:5])}"
    if role == "label_kind":
        kinds = memo.setdefault(("kinds", seg), {o["value"] for o in stats._comp_category_options(seg, ctx["color-by"])})
        return None if value in kinds else "this label source is not loaded (run or reload the annotation)"
    if role == "annot_method":
        with core._annot_lock:
            ok = value in core._ANNOT_METHODS and core._annot_state.get(f"labels_{value}") is not None
        return None if ok else "the labels of this annotation method are not loaded"
    if role in ("cell_type", "cell_types"):
        kind = applied.get(ref, current.get(ref))
        if not kind:
            return "its label source is not set"
        key = ("types", seg, kind, ctx["cluster-method"], ctx["counts-mode"])
        if key not in memo:
            try:
                labels, _cat = stats._comp_labels(kind, stats._view_tables(seg), seg, ctx["cluster-method"],
                                                  ctx["counts-mode"] == "corrected")
                memo[key] = {str(t) for t in controls._pooled_types(labels).index}
            except Exception:
                memo[key] = None
        if memo[key] is None:
            return "its label source is not loaded"
        vals = value if isinstance(value, list) else [value]
        bad = [v for v in vals if str(v) not in memo[key] and v not in extra]
        return None if not bad else f"no cell is labelled {', '.join(_session_show(b, 30) for b in bad[:5])} now"
    if role == "de_group":
        try:
            have = {o["value"] for o in analysis._de_group_options(seg, ctx["cluster-method"], ctx["color-by"],
                                                                   ctx["counts-mode"] == "corrected")}
        except Exception:
            return "the groups cannot be listed now"
        return None if value in have else "this group does not exist now"
    if role == "population":
        try:
            have = {o["value"] for o in stats._pb_population_options(seg, ctx["cluster-method"], ctx["color-by"])}
        except Exception:
            return "the populations cannot be listed now"
        return None if value in have else "this population does not exist now"
    if role == "geneset":
        return None if value in analysis._geneset_names() else "this gene set does not exist"
    return "not restorable"


def _session_apply_samples(saved: dict, report: dict) -> None:
    """Samples the session had beside the loaded ones: added through the function of the
    '+ Add sample' button when the Load Sample picker itself lists their path, reported as
    "needs loading" otherwise. A path of the file is only compared with the picker's list."""
    want = saved.get("samples") or []
    now = _session_dataset()["samples"]
    if len(want) <= len(now) or not all(_session_same_key(a["key"], b["key"]) for a, b in zip(want, now)):
        return
    cbd = _session_module("cb_data")
    listed = {o["value"] for o in sdata_store._available_samples()}
    for s in want[len(now):]:
        path = next((p for p in listed if p == s["key"]), None)
        if path is None or cbd is None:
            report["needs_loading"].append(f"{_session_show(s['name'], 60)} ({s['n_cells']:,} cells)")
            break                                   # the order of the samples is part of the view
        out = cbd.add_sample(1, path, None, 0)
        if not str(out[0]).startswith("Added"):
            report["needs_loading"].append(f"{_session_show(s['name'], 60)}: {str(out[0])[:200]}")
            break
        report["applied"].append(f"sample {dataview._ds_name(dataview.EXTRA_DATASETS[-1])} added")
        report["bump"].add("extra-datasets-version")


def _session_apply_design(design: dict, report: dict, persist: bool = True) -> None:
    rows_now = stats._design_samples()
    by_key = {}
    for r in design["samples"]:
        hit = next((x for x in rows_now if _session_same_key(r["key"], x["key"])), None)
        if hit is not None:
            by_key[hit["key"]] = r
    if not by_key:
        return
    names = {**stats._design_names(), **design["names"]}
    same = all(x["key"] not in by_key or (x["cond"], x["donor"], x["batch"]) ==
               (by_key[x["key"]]["cond"], by_key[x["key"]]["donor"].strip(), by_key[x["key"]]["batch"].strip())
               for x in rows_now) and names == stats._design_names()
    if same:
        report["unchanged"].append("sample design")
        return
    stats._design_set([{"key": k, "cond": r["cond"], "donor": r["donor"], "batch": r["batch"]}
                       for k, r in by_key.items()], names, save=persist)
    report["applied"].append("sample design: " + stats._design_summary()[:200])
    report["bump"].add("design-version")


def _session_apply_rois(rois: list, report: dict, persist: bool = True) -> None:
    """ROIs of the session that are not loaded are added, one with the same name and class
    but another outline is replaced, the other loaded ROIs stay (the ROI cache of the
    dataset is newer than any session). Same steps as Save ROI: state and cache file
    under the lock, cell metadata, the SpatialData stores in the background."""
    if not rois:
        return
    added, replaced = 0, []
    with core._roi_lock:
        cur = core._roi_state["rois"]
        for r in rois:
            hit = next((x for x in cur if x["name"] == r["name"] and x["cls"] == r["cls"]), None)
            if hit is None:
                cur.append({"name": r["name"], "cls": r["cls"], "polygon_xy": r["polygon_xy"],
                            "color": r["color"] or regions._roi_color(len(cur))})
                added += 1
            elif [list(map(float, p)) for p in hit["polygon_xy"]] != r["polygon_xy"]:
                hit["polygon_xy"] = r["polygon_xy"]
                replaced.append(r["name"])
        kept = len(cur) - len(rois)
        if added or replaced:
            if persist:
                regions._roi_save_cache()
            core._roi_state["pending_hull"] = None
        now = list(cur)
    if not (added or replaced):
        report["unchanged"].append("ROIs")
        return
    regions._roi_apply_everywhere(now)
    if persist:
        threading.Thread(target=regions._roi_write_all_zarrs_bg, daemon=True).start()
    report["applied"].append(f"ROIs: {added} added" + (f", outline replaced for {', '.join(replaced[:5])}" if replaced else "")
                             + (f"; {kept} loaded ROI(s) that are not in the session were kept" if kept > 0 else ""))
    report["bump"].add("roi-done")


def _session_apply_sets(sets: dict, groups: list, report: dict, persist: bool = True) -> None:
    if sets:
        with analysis._geneset_lock:
            cur = analysis._geneset_state["sets"]
            changed = [n for n, g in sets.items() if cur.get(n) != g]
            if changed:
                cur.update({n: list(sets[n]) for n in changed})
                analysis._geneset_state["version"] += 1
                if persist:
                    analysis._gs_save()
        if changed:
            report["applied"].append(f"gene sets: {len(changed)} added or replaced ({', '.join(changed[:5])}"
                                     f"{'…' if len(changed) > 5 else ''}); score them again to colour by them")
            report["bump"].add("geneset-version")
        else:
            report["unchanged"].append("gene sets")
    if groups:
        with analysis._de_lock:
            cur = analysis._de_state["groups"]
            n = 0
            for g in groups:
                hit = next((x for x in cur if x["name"] == g["name"]), None)
                if hit is None:
                    cur.append({"name": g["name"], "polygon_xy": g["polygon_xy"]})
                    n += 1
                elif [list(map(float, p)) for p in hit["polygon_xy"]] != g["polygon_xy"]:
                    hit["polygon_xy"] = g["polygon_xy"]
                    n += 1
            if n and persist:
                analysis._de_groups_save()
        if n:
            report["applied"].append(f"DE groups: {n} added or replaced")
            report["bump"].add("de-groups-version")
        else:
            report["unchanged"].append("DE groups")


def _session_apply_subset(sub, report: dict) -> None:
    now = _session_subset_now()
    if sub is None:
        if now is not None:
            subsetting.unsubset()
            report["applied"].append("subset cleared (the session had none)")
        return
    if sub.get("restorable") is False:
        report["skipped"].append("subset: the session's subset could not be saved (explicit cell ids or too "
                                 "many values); the cells in view were left as they are")
        return
    f = dict(sub["filters"])
    if now is not None and now.get("filters") == f:
        report["unchanged"].append("subset")
        return
    if f.get("method") and f["method"] not in (core.DATA.get("cluster_methods") or []):
        report["missing"].append(f"subset: the cluster set {_session_show(f['method'], 60)} it filters on does not exist")
        return
    if f.get("gene") and core._gene_base(f["gene"]) not in _session_genes():
        report["missing"].append(f"subset: the gene {_session_show(f['gene'], 30)} it filters on is not on the panel")
        return
    try:
        subsetting.subset(**f)
    except Exception as exc:
        report["missing"].append(f"subset: not applied ({str(exc)[:200]})")
        return
    got = _session_subset_now()
    if got is None or got.get("filters") != f:
        if got is not None:                      # a filter was skipped: not the session's subset
            subsetting.unsubset()
        report["missing"].append("subset: its filters cannot all be applied now (cell-type labels not loaded?); "
                                 "no subset is active")
        return
    n = int(len(core.DATA["df"]))
    note = "" if sub.get("n_cells") in (None, n) else f" (it selected {sub['n_cells']:,} when it was saved)"
    report["applied"].append(f"subset: {n:,} cells{note}")


def _session_restore(doc: dict, current=None, source: str = "file", record: bool = True, echo: bool = True,
                     persist: bool = True) -> dict:
    """Apply a parsed session document. *current*: {control id: value} of the page now
    (for the values the session leaves alone). Returns the report: ok, refused (why
    nothing was done), mode ("full" / "independent"), applied, unchanged, skipped, missing,
    needs_loading, ignored, invalid, controls ({control id: value} to set in the page),
    bump (version stores to increase), text. persist=False (the dry run of a batch config):
    the state is applied in memory for the checks; no cache file, store or provenance
    record is written."""
    report = {"ok": False, "refused": None, "mode": None, "applied": [], "unchanged": [], "skipped": [],
              "missing": [], "needs_loading": [], "ignored": [], "invalid": [], "controls": {}, "bump": set(),
              "text": ""}
    current = dict(current) if isinstance(current, dict) else {}
    try:
        busy = _session_busy()
        if busy:
            report["refused"] = (f"Still running: {', '.join(busy)}. A session is restored when no job is "
                                 f"running: wait for it to finish (or cancel it) and restore again.")
            return _session_finish(report, None, source, record=False, echo=echo)
        clean, report["ignored"], report["invalid"] = _session_validate(doc)
        _session_apply_samples(clean["dataset"], report)
        diff = _session_dataset_diff(clean["dataset"], _session_dataset())
        report["mode"] = "independent" if diff else "full"
        ctl = clean["controls"]
        spec = _session_controls()
        applied = {}
        if diff:
            report["skipped"].append("The session does not match the loaded data: " + "; ".join(diff) + ".")
            bound = sorted(c for c in ctl if spec[c][1])
            parts = [f"{len(bound)} data-dependent control value(s) (segmentation, cluster set, colouring, labels, "
                     f"cell types, genes, regions)" if bound else "",
                     "the sample design" if clean["design"] and clean["design"]["samples"] else "",
                     f"{len(clean['rois'])} ROI(s)" if clean["rois"] else "",
                     f"{len(clean['gene_sets'])} gene set(s)" if clean["gene_sets"] else "",
                     f"{len(clean['de_groups'])} DE group(s)" if clean["de_groups"] else "",
                     "the subset" if clean["subset"] else ""]
            report["skipped"].append("Not applied: " + ", ".join(p for p in parts if p) + "." if any(parts)
                                     else "Nothing data-dependent was in the session.")
            for cid, val in ctl.items():
                if not spec[cid][1]:
                    applied[cid] = val
        else:
            ctx = {"seg-source": current.get("seg-source") or "xenium",
                   "counts-mode": current.get("counts-mode") or core._active_counts_mode or "original",
                   "cluster-method": current.get("cluster-method") or _session_default("cluster-method"),
                   "color-by": current.get("color-by") or _session_default("color-by")}
            memo = {}
            # 1. the segmentation everything else is read on
            seg = ctl.get("seg-source")
            if seg:
                why = _session_seg_load(seg)
                if why is None:
                    applied["seg-source"] = ctx["seg-source"] = seg
                    rid = core._seg_run_id(core._seg_result(seg)) if core._seg_tool(seg) != "xenium" else ""
                    if clean["seg_run_id"] and rid and rid != clean["seg_run_id"]:
                        report["skipped"].append(f"Note: the segmentation run {seg} was computed again since the "
                                                 f"session was saved (same parameters, another run id).")
                else:
                    report["missing"].append(f"segmentation {seg}: {why}; the view stays on {ctx['seg-source']}")
            # 2. what the app persists itself (merged, never removed), then the subset
            if clean["design"]:
                _session_apply_design(clean["design"], report, persist)
            _session_apply_rois(clean["rois"], report, persist)
            _session_apply_sets(clean["gene_sets"], clean["de_groups"], report, persist)
            if not persist:
                report["skipped"].append("Validation only: the design, ROIs, gene sets and groups of the session "
                                         "are in memory, no cache file or store was written.")
            _session_apply_subset(clean["subset"], report)
            # 3. the shared context, in the order the pickers depend on each other
            mode = ctl.get("counts-mode")
            if mode == "corrected" and not _session_has_corrected(ctx["seg-source"]):
                report["missing"].append("counts mode 'corrected': SPLIT-corrected counts are not loaded for this view")
            elif mode:
                applied["counts-mode"] = ctx["counts-mode"] = mode
            for cid in ("cluster-method", "color-by"):
                if ctl.get(cid) is None:
                    continue
                why = _session_dynamic_problem(cid, ctl[cid], ctx, applied, current, memo)
                if why is None:
                    applied[cid] = ctx[cid] = ctl[cid]
                else:
                    report["missing"].append(f"{cid} {_session_show(ctl[cid], 60)}: {why}")
            # 4. every other control: label sources before the cell types read from them
            rest = [c for c in ctl if c not in ("seg-source", "counts-mode", "cluster-method", "color-by")]
            rest.sort(key=lambda c: (spec[c][0] == "dynamic", _SESSION_DYNAMIC.get(c, ("",))[0] in ("cell_type", "cell_types")))
            for cid in rest:
                if spec[cid][0] != "dynamic":
                    applied[cid] = ctl[cid]
                    continue
                try:
                    why = _session_dynamic_problem(cid, ctl[cid], ctx, applied, current, memo)
                except Exception as exc:
                    why = f"could not be checked ({type(exc).__name__})"
                if why is None:
                    applied[cid] = ctl[cid]
                else:
                    report["missing"].append(f"{cid} {_session_show(ctl[cid], 60)}: {why}")
        # a preset that would overwrite the restored λ / resolution in the page
        if applied.get("bk-preset") in ("domains", "celltype"):
            banksy = _session_module("banksy")
            want = ((banksy._BK_LAMBDA_DOMAINS, banksy._BK_RESOLUTION) if applied["bk-preset"] == "domains"
                    else (banksy._BK_LAMBDA_CELLTYPE, banksy._BK_RESOLUTION_CELLTYPE)) if banksy else None
            if want and (applied.get("bk-lambda", want[0]), applied.get("bk-resolution", want[1])) != want:
                applied["bk-preset"] = "custom"
        # the gene-set picker fills the gene text from the stored set when it changes in the
        # page (cb_analysis.fill_geneset_form): a restored text that differs from that set
        # (it was edited and not saved as a set) would be overwritten. The text wins; the
        # picker is left empty and the report says so.
        _gs_pick, _gs_text = applied.get("geneset-select"), applied.get("geneset-genes")
        if _gs_pick and isinstance(_gs_text, str):
            _an = _session_module("analysis")
            _stored = None
            if _an is not None:
                with _an._geneset_lock:
                    _stored = _an._geneset_state["sets"].get(_gs_pick)
            _lines = _session_gene_lines(_gs_text)
            # (an empty text is not an edit: the picker then fills it, as it does in the page)
            if _stored is not None and _lines and _lines != [str(g) for g in _stored]:
                applied["geneset-select"] = None
                report["skipped"].append(
                    f"The gene-set picker was left empty: the gene list of the session differs from the stored "
                    f"set {_session_show(_gs_pick, 40)} (it was edited after the set was saved), and selecting "
                    f"the set would replace the text.")
        # the two switches the server mirrors, through the callbacks that own the mirror
        cbs = _session_module("cb_spatial")
        if cbs is not None:
            if "counts-mode" in applied and applied["counts-mode"] != core._active_counts_mode:
                cbs.sync_counts_mode(applied["counts-mode"])
            if "roi-show-toggle" in applied:
                with core._roi_lock:
                    shown = bool(core._roi_state.get("show", True))
                if shown != ("show" in applied["roi-show-toggle"]):
                    cbs.toggle_roi_show(applied["roi-show-toggle"], 0)
                    report["bump"].add("roi-done")
        changed = {c: v for c, v in applied.items() if c not in current or current[c] != v}
        report["controls"] = changed
        n_same = len(applied) - len(changed)
        report["applied"].insert(0, f"{len(changed)} control value(s) set"
                                 + (f" ({n_same} more already had the session's value)" if n_same and current else ""))
        report["ok"] = True
        return _session_finish(report, clean, source, record and persist, echo)
    except _SessionError as exc:
        report["refused"] = str(exc)
        return _session_finish(report, None, source, record=False, echo=echo)


def _session_finish(report: dict, clean, source: str, record: bool, echo: bool = True) -> dict:
    lines = []
    if report["refused"]:
        lines.append("Not restored. " + report["refused"])
    else:
        when = (clean or {}).get("saved_at") or ""
        head = "Session restored" if report["mode"] == "full" else \
            "Session restored in part (only the settings that do not depend on the data)"
        lines.append(f"{head}{' — saved ' + when.replace('T', ' ')[:16] + ' UTC' if when else ''}.")
        for title, key in (("Applied", "applied"), ("Already as in the session", "unchanged"),
                           ("Needs loading (not loaded from a path in the file)", "needs_loading"),
                           ("Missing now (the rest was restored)", "missing"), ("Skipped", "skipped"),
                           ("Refused values (not applied)", "invalid"), ("Unknown keys (ignored)", "ignored")):
            items = report[key]
            if items:
                shown = items[:40] + ([f"… and {len(items) - 40} more"] if len(items) > 40 else [])
                lines.append(f"{title}:\n" + "\n".join(f"  - {x}" for x in shown))
        if report["needs_loading"]:
            lines.append("Add the samples named above with '+ Add sample' and restore again.")
    report["text"] = "\n".join(lines)
    if echo:
        print("  session: " + report["text"].replace("\n", "\n  "), flush=True)
    if record and report["ok"]:
        provenance._prov_record(
            "session", "Session restored", seg_source=(report["controls"].get("seg-source") or None),
            params={"action": "restore", "from": source, "mode": report["mode"],
                    "format_version": (clean or {}).get("version"), "saved_at": (clean or {}).get("saved_at") or None,
                    "n_controls_set": len(report["controls"]), "n_missing": len(report["missing"]),
                    "n_refused_values": len(report["invalid"]), "n_unknown_keys": len(report["ignored"]),
                    "n_needs_loading": len(report["needs_loading"])},
            message=report["text"].splitlines()[0],
            warnings=(report["missing"] + report["skipped"] + report["needs_loading"])[:30], uses_design=True)
    return report


def _session_restore_text(text, current=None, source: str = "file", record: bool = True) -> dict:
    """Parse and restore the text / bytes of a session file (an upload, a slot)."""
    try:
        doc = _session_parse(text)
    except _SessionError as exc:
        report = {"ok": False, "refused": str(exc), "mode": None, "applied": [], "unchanged": [], "skipped": [],
                  "missing": [], "needs_loading": [], "ignored": [], "invalid": [], "controls": {}, "bump": set(),
                  "text": ""}
        return _session_finish(report, None, source, record=False)
    return _session_restore(doc, current, source, record)


def _session_restore_slot(slot: str, current=None, record: bool = True) -> dict:
    """Restore a slot of the cache directory: "saved", or "last" (the automatic save of
    the server run before this one)."""
    real = _session_last_slot() if slot == "last" else ("saved" if slot == "saved" else None)
    text = None
    if real is not None:
        try:
            with open(_session_path(real), "rb") as fh:
                text = fh.read(_SESSION_MAX_BYTES + 1)
        except OSError:
            text = None
    if text is None:
        what = "No session of an earlier run of the app is saved for this dataset yet (the last session is " \
               "saved automatically while the app is open)." if slot == "last" else \
               "No session was saved for this dataset with 'Save session'."
        report = {"ok": False, "refused": what, "mode": None, "applied": [], "unchanged": [], "skipped": [],
                  "missing": [], "needs_loading": [], "ignored": [], "invalid": [], "controls": {}, "bump": set(),
                  "text": ""}
        return _session_finish(report, None, slot, record=False)
    return _session_restore_text(text, current, "last session (automatic)" if slot == "last" else "saved session",
                                 record)


def _session_upload_bytes(contents) -> bytes:
    """The bytes of a dcc.Upload value ("data:<type>;base64,<data>")."""
    if not isinstance(contents, str) or "," not in contents:
        raise _SessionError("the upload is empty")
    head, data = contents.split(",", 1)
    if "base64" not in head or len(data) > _SESSION_MAX_BYTES * 4 // 3 + 16:
        raise _SessionError(f"the file is larger than {_SESSION_MAX_BYTES // 1_000_000} MB or not a file upload")
    try:
        return base64.b64decode(data, validate=True)
    except ValueError:
        raise _SessionError("the upload could not be decoded") from None


# ─── Layout ───────────────────────────────────────────────────────────────────
_SESSION_BTN_STYLE = {"padding": "4px 10px", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                      "border": f"1px solid {core.BORDER}", "borderRadius": "4px", "cursor": "pointer",
                      "fontSize": "12px"}


def _session_sidebar_entry():
    """The one entry in the sidebar, under the dataset block (Load / Save as SpatialData)."""
    return html.Div([
        html.Button("Session: save / restore…", id="session-open-btn", n_clicks=0,
                    title="Save the settings of this session to a file, or restore a saved session",
                    style={**_SESSION_BTN_STYLE, "width": "100%", "fontSize": "11px", "padding": "5px 0"}),
        # always in the page (the content of a closed dialog is not): the timer of the
        # automatic save and the stores the callbacks of cb_session.py pass values through
        dcc.Download(id="session-download"),
        dcc.Store(id="session-request", data=None),
        dcc.Store(id="session-apply", data=None),
        dcc.Store(id="session-autosave-note", data=None),
        dcc.Interval(id="session-autosave", interval=_SESSION_AUTOSAVE_MS, n_intervals=0),
    ], id="session-entry", style={"marginBottom": "10px"})


def _session_modal():
    note = {"fontSize": "12px", "color": core.MUTED, "lineHeight": "1.4", "marginBottom": "10px"}
    row = {"display": "flex", "flexWrap": "wrap", "gap": "8px", "alignItems": "center", "marginBottom": "10px"}
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Session"), close_button=True),
        dbc.ModalBody([
            html.Div("A session file holds the settings of this page (controls, sample design, ROIs, gene sets, "
                     "DE groups, subset filters) for the loaded dataset. It holds no results: run an analysis "
                     "again after a restore and its result comes back from the cache. Path fields are not saved.",
                     style=note),
            html.Div([
                html.Button("Save session", id="session-save-btn", n_clicks=0, style=_SESSION_BTN_STYLE,
                            title="Write the session of this dataset under the cache directory"),
                html.Button("Download…", id="session-download-btn", n_clicks=0, style=_SESSION_BTN_STYLE,
                            title="Download the session as a JSON file"),
            ], style=row),
            html.Div([
                html.Button("Restore saved session", id="session-restore-saved-btn", n_clicks=0,
                            style=_SESSION_BTN_STYLE, title="Restore what 'Save session' wrote for this dataset"),
                html.Button("Restore last session", id="session-restore-last-btn", n_clicks=0,
                            style=_SESSION_BTN_STYLE,
                            title="Restore the automatic save of the last run of the app on this dataset"),
                dcc.Upload(html.Button("Restore from file…", style=_SESSION_BTN_STYLE, n_clicks=0,
                                       title="Upload a downloaded session file (.json)"),
                           id="session-upload", multiple=False, accept=".json,application/json",
                           max_size=_SESSION_MAX_BYTES),
            ], style=row),
            html.Pre(id="session-slots", style={**note, "whiteSpace": "pre-wrap", "margin": "0 0 8px 0",
                                                 "fontFamily": "inherit"}),
            html.Pre("", id="session-status",
                     style={"fontSize": "12px", "color": core.TEXT, "whiteSpace": "pre-wrap", "maxHeight": "320px",
                            "overflowY": "auto", "margin": "0", "fontFamily": "inherit"}),
        ]),
        dbc.ModalFooter(dbc.Button("Close", id="session-close-btn", color="secondary", size="sm")),
    ], id="session-modal", is_open=False, size="lg", style={"color": core.TEXT},
        content_style={"backgroundColor": core.DARK_BG, "color": core.TEXT}, backdrop=True)
