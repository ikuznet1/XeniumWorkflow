"""Session save / restore (xenium_app/session.py, cb_session.py).

A session file is settings, never results, and restore is an input path that takes a file
from outside: the tests here are mostly about what restore refuses. "The page" is played
by a dict of control values, as the browser sends it in the ``session-request`` store."""
from __future__ import annotations

import ast
import base64
import copy
import json
import os
import threading

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import xe_testlib as tl


@pytest.fixture(autouse=True)
def _own_cache(xe, tmp_path, monkeypatch):
    """Every file a test writes (session slots, ROI / gene-set / design caches, provenance)
    goes to its own cache root; the state those files feed is reset afterwards."""
    monkeypatch.setattr(xe, "CACHE_BASE_DIR", str(tmp_path))
    before = list(xe._prov_recent)
    xe._an_autoload()            # gene sets / DE groups an earlier test module left in memory
    xe._stats_reset()
    xe._session_auto.update({"baseline": None, "written": None})
    yield
    xe._prov_recent[:] = before
    monkeypatch.undo()
    xe._an_autoload()
    xe._stats_reset()
    xe._session_auto.update({"baseline": None, "written": None})


def _defaults(xe) -> dict:
    return {cid: xe._session_default(cid) for cid in xe._session_controls()}


def _fab(D, n, seed):
    """A small fabricated extra sample (as tests/test_stats.py builds them)."""
    rng = np.random.default_rng(seed)
    genes = list(D["gene_names"])
    m = sp.lil_matrix((n, len(genes)), dtype=np.float32)
    m[:, :20] = rng.poisson(2.0, size=(n, 20))
    df = pd.DataFrame({"x_centroid": rng.uniform(0, 400, n), "y_centroid": rng.uniform(0, 400, n),
                       "transcript_counts": np.asarray(m.sum(axis=1)).ravel()},
                      index=[f"s{seed}_{i}" for i in range(n)])
    return {"expr": m.tocsr(), "df_to_expr": np.arange(n), "gene_name_to_idx": D["gene_name_to_idx"],
            "gene_names": genes, "df": df, "x_offset": 2000.0 * (seed + 1), "cluster_methods": []}


def _rect(D, lo=0.0, hi=0.6):
    x, y = D["df"]["x_centroid"].to_numpy(float), -D["df"]["y_centroid"].to_numpy(float)
    x0, x1 = np.quantile(x, lo) - 1.0, np.quantile(x, hi) + 1.0
    return [[float(x0), float(y.min() - 5)], [float(x1), float(y.min() - 5)],
            [float(x1), float(y.max() + 5)], [float(x0), float(y.max() + 5)]]


def _doc(xe, controls=None, **state):
    """A valid session document of the loaded data with some values replaced."""
    doc = xe._session_snapshot({**_defaults(xe), **(controls or {})}, "download")
    doc["state"].update(state)
    return doc


def _restore(xe, doc, current=None):
    return xe._session_restore_text(json.dumps(doc), _defaults(xe) if current is None else current, "test")


# ── the allow-list ───────────────────────────────────────────────────────────
def test_every_value_control_of_the_layout_is_saved_or_listed_as_not_saved(xe):
    comps = xe._session_components()
    valued = {cid for cid, c in comps.items() if "value" in getattr(c, "_prop_names", ())
              and type(c).__name__ not in ("Tabs", "Button", "Store")}
    allowed, excluded = set(xe._session_controls()), set(xe._SESSION_NOT_SAVED)
    assert not allowed & excluded
    assert not allowed - set(comps), "allow-listed controls that are not in the layout"
    unlisted = valued - allowed - excluded
    assert not unlisted, (f"controls with a value that session.py neither saves nor lists in _SESSION_NOT_SAVED "
                          f"(add each to one of the two): {sorted(unlisted)}")
    # no path field, and nothing of the REPL, is ever saved
    assert not [c for c in allowed if "path" in c or c.endswith("-dir") or "repl" in c]
    assert "repl-input" in excluded and all(len(why) > 10 for why in xe._SESSION_NOT_SAVED.values())
    # every default of the layout passes its own check (a control whose default is refused
    # could never be restored)
    # view controls whose options belong to the loaded images / data size are bound to the data
    assert all(xe._session_controls()[c][1] for c in ("morph-enable", "morph-channels", "morph-zlevel", "renderer-mode"))
    bad = {c: xe._session_check_control(c, xe._session_default(c)) for c in allowed}
    assert not {c: r for c, r in bad.items() if not r[0]}


def test_inventory_table_is_in_the_module_docstring():
    text = ast.get_docstring(ast.parse(tl.app_source_of("_session_restore").read_text()))
    for word in ("State inventory", "| Sample design", "| ROIs", "| Subset filters", "| Path fields", "never",
                 "XENIUM_ENABLE_REPL"):
        assert word in text


# ── round trip ───────────────────────────────────────────────────────────────
def test_round_trip_restores_design_rois_sets_subset_and_controls(xe, D, labels, expressed_gene):
    extras = [_fab(D, 40, 0), _fab(D, 50, 1)]
    methods = list(D["cluster_methods"])
    roi, box = _rect(D), _rect(D, 0.2, 0.9)

    def build():
        xe.EXTRA_DATASETS.extend(extras)
        with xe._annot_lock:
            xe._annot_state["labels_celltypist"] = labels

    build()
    xe._design_set([{"key": xe._ds_key(D), "cond": "A", "donor": "d1", "batch": "b1"},
                    {"key": xe._ds_key(extras[0]), "cond": "B", "donor": "d2", "batch": ""},
                    {"key": xe._ds_key(extras[1]), "cond": "", "donor": "", "batch": ""}],
                   {"A": "treated", "B": "control"})
    out = xe.save_roi(1, "left", "region", roi, 0, "xenium")
    assert out[1] == ""
    with xe._geneset_lock:
        xe._geneset_state["sets"]["mine"] = list(D["gene_names"][:4])
        xe._gs_save()
    with xe._de_lock:
        xe._de_state["groups"].append({"name": "box", "polygon_xy": box})
        xe._de_groups_save()
    cut = float(np.median(D["df"]["transcript_counts"]))
    xe.subset(min_transcripts=cut)
    n_sub = len(D["df"])
    assert 0 < n_sub < len(D["_df_original"])
    changed = {"point-size": 5, "point-opacity": 0.4, "moran-k": 9, "de-mode": "pseudobulk", "bk-preset": "custom",
               "bk-lambda": 0.33, "lr-pairs": f"{D['gene_names'][0]},{D['gene_names'][1]}",
               "boundary-toggles": ["cell"], "cluster-method": methods[1], "gene-selector": expressed_gene,
               "color-by": "cell_type:celltypist", "prox-labels": "celltype:celltypist", "prox-target": "Alpha",
               "prox-anchor": "Beta", "agg-labels": "celltype:celltypist", "agg-classes": ["Alpha", "Gamma"],
               "segqc-method": "celltypist", "geneset-select": "mine", "de-group-a": "grp:box",
               "roi-show-toggle": [], "baysor-xmin": 12.5, "txf-seed": 7, "jc-integrate": "donor",
               "annot-label-col": "cell_type", "sfree-bins": "20, 40", "webgl-tx-genes": [expressed_gene],
               "morph-channels": ["dapi"]}
    xe.toggle_roi_show([], 0)
    doc, msg = xe._session_save({**_defaults(xe), **changed}, "download")
    text = xe._session_text(doc)
    assert "Settings only" in msg and doc["notes"] == []
    saved = json.loads(text)
    assert saved["format"] == xe._SESSION_FORMAT and saved["version"] == 1 and saved["saved_at"].endswith("+00:00")
    assert saved["dataset"]["key"] == xe._dataset_key() == xe._prov_data()["dataset_key"]
    assert [s["n_cells"] for s in saved["dataset"]["samples"]] == [len(D["_df_original"]), 40, 50]
    assert saved["dataset"]["samples"][0]["name"] == xe._ds_name(D) and "commit" in saved["app"]
    assert saved["state"]["subset"] == {"filters": {"min_transcripts": cut}, "n_cells": n_sub}
    # settings, not results: no matrix, no labels, no per-cell list
    assert len(text) < 60_000 and "labels_celltypist" not in text and str(D["df"].index[0]) not in text

    # ── a new "server run": everything back to the freshly loaded app ──
    xe._test_baseline.reset()
    for path in (xe._gs_path(), xe._de_groups_path()):
        os.remove(path)
    xe._an_autoload()
    xe._stats_reset()
    assert not xe._roi_state["rois"] and not xe._geneset_names() and "_df_original" not in D
    assert all(not r["cond"] for r in xe._design_samples())
    build()

    report = _restore(xe, saved)
    assert report["ok"] and report["mode"] == "full", report["text"]
    assert not report["missing"] and not report["invalid"] and not report["ignored"], report["text"]
    rows = xe._design_samples()
    assert [(r["cond"], r["donor"], r["batch"]) for r in rows] == [("A", "d1", "b1"), ("B", "d2", ""), ("", "", "")]
    assert xe._design_names() == {"A": "treated", "B": "control"}
    assert [(r["name"], r["cls"], r["polygon_xy"]) for r in xe._roi_state["rois"]] == [("left", "region", roi)]
    assert "roi_region" in D["_df_original"].columns and (D["_df_original"]["roi_region"] == "left").any()
    assert xe._roi_state["show"] is False
    assert xe._geneset_state["sets"] == {"mine": list(D["gene_names"][:4])}
    assert xe._de_state["groups"] == [{"name": "box", "polygon_xy": box}]
    assert len(D["df"]) == n_sub and D["_subset_desc"][0]["min_transcripts"] == cut
    for path in (xe._roi_cache_path(), xe._gs_path(), xe._de_groups_path()):
        assert os.path.exists(path), "the app's own cache file is written by the restore"
    # the page: exactly the controls whose value differs from what it shows
    assert report["controls"] == changed
    assert set(report["bump"]) == {"design-version", "roi-done", "geneset-version", "de-groups-version"}
    # … and saving again gives the same session
    again = xe._session_snapshot({**_defaults(xe), **report["controls"]}, "download")
    assert again["state"] == saved["state"] and again["dataset"] == saved["dataset"]
    # a second restore changes nothing
    second = _restore(xe, saved, {**_defaults(xe), **report["controls"]})
    assert second["controls"] == {} and not second["bump"]
    assert {"sample design", "ROIs", "gene sets", "DE groups", "subset"} <= set(second["unchanged"])


def test_restore_keeps_what_the_app_persisted_after_the_session_was_saved(xe, D):
    """ROIs, gene sets and DE groups have their own cache files, which are newer than any
    session: restore adds and replaces, it never deletes."""
    xe.save_roi(1, "old", "region", _rect(D, 0.0, 0.4), 0, "xenium")
    saved = _doc(xe)
    xe.save_roi(1, "newer", "region", _rect(D, 0.5, 1.0), 0, "xenium")
    with xe._roi_lock:
        xe._roi_state["rois"][0]["polygon_xy"] = _rect(D, 0.0, 0.3)       # "old" redrawn since
    report = _restore(xe, saved)
    rois = {r["name"]: r["polygon_xy"] for r in xe._roi_state["rois"]}
    assert set(rois) == {"old", "newer"} and rois["old"] == _rect(D, 0.0, 0.4)
    assert "1 loaded ROI(s) that are not in the session were kept" in report["text"]
    assert "outline replaced for old" in report["text"]


def test_labels_and_sets_that_are_not_loaded_are_reported_and_the_rest_is_restored(xe, D):
    saved = _doc(xe, {"color-by": "cell_type:celltypist", "prox-labels": "celltype:celltypist",
                      "prox-target": "Alpha", "segqc-method": "celltypist", "geneset-select": "gone",
                      "de-group-a": "grp:gone", "gene-selector": "NOT_A_GENE", "point-size": 6,
                      "counts-mode": "corrected"})
    report = _restore(xe, saved)
    assert report["ok"] and report["mode"] == "full"
    missing = "\n".join(report["missing"])
    for cid in ("color-by", "prox-labels", "prox-target", "segqc-method", "geneset-select", "de-group-a",
                "gene-selector", "counts mode 'corrected'"):
        assert cid in missing, cid
    assert report["controls"] == {"point-size": 6}
    assert xe._active_counts_mode == "original"
    assert "Missing now (the rest was restored)" in report["text"]


def test_missing_segmentation_run_and_cluster_set_are_named_and_the_rest_is_restored(xe, D):
    saved = _doc(xe, {"seg-source": "baysor:0123abcd", "cluster-method": "banksy_domains_gone",
                      "moran-k": 11, "density-bin": 20})
    saved["state"]["seg_run_id"] = "run-1"
    report = _restore(xe, saved)
    assert report["ok"] and report["mode"] == "full"
    assert report["controls"] == {"moran-k": 11, "density-bin": 20}
    missing = "\n".join(report["missing"])
    assert "segmentation baysor:0123abcd" in missing and "no such cached run" in missing
    assert "the view stays on xenium" in missing
    assert "cluster-method 'banksy_domains_gone': this cluster set does not exist any more" in missing
    assert xe._baysor_state["status"] == "idle" and xe._baysor_state["result"] is None


# ── validation ───────────────────────────────────────────────────────────────
def test_unknown_keys_are_reported_and_ignored(xe):
    saved = _doc(xe, {"point-size": 4})
    saved["plugins"] = ["x"]
    saved["state"]["globals"] = {"CACHE_BASE_DIR": "/elsewhere"}
    saved["state"]["controls"]["no-such-control"] = 1
    saved["dataset"]["loader"] = "os.system"
    saved["state"]["design"]["hook"] = "x"
    before = xe.CACHE_BASE_DIR
    report = _restore(xe, saved)
    assert report["ok"] and report["controls"] == {"point-size": 4}
    ignored = "\n".join(report["ignored"])
    for key in ("'plugins'", "state.'globals'", "control 'no-such-control'", "dataset.'loader'", "state.design.'hook'"):
        assert key in ignored, key
    assert "Unknown keys (ignored)" in report["text"] and xe.CACHE_BASE_DIR == before


@pytest.mark.parametrize("cid, value, why", [
    ("point-size", "5", "not a number"), ("point-size", True, "not a number"),
    ("moran-k", 9999, "outside 1 … 50"), ("moran-k", -3, "outside"), ("agg-alpha", 0.9, "outside"),
    ("txf-seed", float("1e300"), "outside"), ("de-mode", "nonsense", "not one of the options"),
    ("density-bin", "10", "not one of the options"), ("boundary-toggles", ["cell", "evil"], "not an option"),
    ("boundary-toggles", "cell", "not a list"), ("lr-pairs", "A,B\x00", "not text"),
    ("lr-pairs", "x" * 20_001, "at most 20000"), ("annot-label-col", 'x"); system("id"); #', "characters"),
    ("sfree-bins", "25; rm", "characters"), ("gene-selector", {"a": 1}, "not a name"),
    ("webgl-tx-genes", "Gene", "not a list"), ("counts-mode", "raw", "not 'original' or 'corrected'"),
    ("seg-source", "baysor:../../x", "not a segmentation source"), ("seg-source", "evil:abc", "not a segmentation"),
    ("cluster-method", ["a"], "not a name"), ("agg-classes", [1, 2], "not a list of names"),
])
def test_values_of_the_wrong_type_or_out_of_range_are_refused(xe, cid, value, why):
    saved = _doc(xe, {"point-opacity": 0.5})
    saved["state"]["controls"][cid] = value
    report = _restore(xe, saved)
    assert report["ok"] and cid not in report["controls"] and report["controls"] == {"point-opacity": 0.5}
    hit = [m for m in report["invalid"] if m.startswith(f"control {cid}:")]
    assert len(hit) == 1 and why in hit[0], report["invalid"]
    assert "Refused values (not applied)" in report["text"]


def test_malformed_state_sections_are_refused_one_by_one(xe, D):
    saved = _doc(xe, {"point-size": 3},
                 rois=[{"name": "ok", "cls": "region", "polygon_xy": _rect(D)},
                       {"name": "two points", "cls": "region", "polygon_xy": [[0, 0], [1, 1]]},
                       {"name": "nan", "cls": "region", "polygon_xy": [[0, 0], [1, "x"], [2, 2]]},
                       {"name": "", "cls": "region", "polygon_xy": _rect(D)}, "text"],
                 gene_sets={"good": ["A", "B"], "bad": "A,B", "worse": [1, 2]},
                 de_groups=[{"name": "g", "polygon_xy": [[0, 0]]}],
                 subset={"filters": {"min_transcripts": "many", "cell_ids": ["a"]}},
                 design={"names": {"A": "same", "B": "same"}, "samples": [{"key": xe._ds_key(D), "cond": "C"}]})
    report = _restore(xe, saved)
    assert report["ok"]
    assert [r["name"] for r in xe._roi_state["rois"]] == ["ok"]
    assert xe._geneset_state["sets"] == {"good": ["A", "B"]} and not xe._de_state["groups"]
    assert "_df_original" not in D and all(not r["cond"] for r in xe._design_samples())
    assert xe._design_names() == {"A": "A", "B": "B"}
    invalid = "\n".join(report["invalid"])
    for part in ("state.rois", "state.gene_sets: 'bad'", "state.gene_sets: 'worse'", "state.de_groups",
                 "state.subset.filters.'min_transcripts'", "state.subset.filters.'cell_ids'", "state.design"):
        assert part in invalid, part
    assert "subset: the session's subset could not be saved" in "\n".join(report["skipped"])


def test_paths_in_a_file_are_never_followed(xe, D, tmp_path, monkeypatch):
    outside = tmp_path / "outside" / "output-evil"
    outside.mkdir(parents=True)
    (outside / "experiment.xenium").write_text("{}")
    opened = []
    real_open = open

    def spy(path, *a, **k):
        opened.append(str(path))
        return real_open(path, *a, **k)

    def no_load(*a, **k):
        raise AssertionError("a sample was loaded from a path of the session file")

    monkeypatch.setattr(xe, "add_sample", no_load)
    monkeypatch.setattr(xe, "load_sample", no_load)
    monkeypatch.setattr(xe, "load_xenium_data", no_load)
    monkeypatch.setattr(xe, "_available_samples", lambda: [{"label": "listed", "value": str(tmp_path / "listed")}])
    saved = _doc(xe, {"point-size": 7})
    for cid, val in (("annot-rds-path", "../../etc/passwd"), ("spage-rds-path", str(outside / "ref.rds")),
                     ("split-rds-path", "~/x.rds"), ("save-sdata-dir", str(outside)), ("save-sdata-name", "../x"),
                     ("sample-custom-path", str(outside)), ("cellnest-folder-path", str(outside)),
                     ("sample-picker", str(outside))):
        saved["state"]["controls"][cid] = val
    saved["dataset"]["samples"].append({"key": str(outside), "name": "output-evil", "n_cells": 5})
    saved["dataset"]["samples"].append({"key": "../../../etc", "name": "etc", "n_cells": 1})
    with monkeypatch.context() as m:
        m.setattr("builtins.open", spy)
        report = _restore(xe, saved)
    assert report["ok"] and report["mode"] == "independent"
    assert not [p for p in opened if "outside" in p or "etc" in p], opened
    assert len(xe.EXTRA_DATASETS) == 0
    assert len(report["needs_loading"]) == 1 and "output-evil" in report["needs_loading"][0]
    assert "Needs loading (not loaded from a path in the file)" in report["text"]
    ignored = "\n".join(report["ignored"])
    for cid in ("annot-rds-path", "spage-rds-path", "split-rds-path", "save-sdata-dir", "sample-custom-path",
                "cellnest-folder-path", "sample-picker"):
        assert f"control '{cid}': a path is never taken from a session file" in ignored
    assert set(report["controls"]) == {"point-size"}


def test_a_sample_is_added_only_when_the_load_sample_picker_lists_its_path(xe, D, monkeypatch):
    extra = _fab(D, 30, 3)
    listed = "/data/root/output-listed"
    calls = []

    def fake_add(n_clicks, picked, custom, extra_ver):
        calls.append((picked, custom))
        xe.EXTRA_DATASETS.append({**extra, "data_dir": picked})
        return "Added (x_offset=0 µm).", False, 1, "2 samples loaded"

    monkeypatch.setattr(xe, "add_sample", fake_add)
    monkeypatch.setattr(xe, "_available_samples", lambda: [{"label": "output-listed", "value": listed}])
    saved = _doc(xe, {"moran-k": 8})
    saved["dataset"]["samples"].append({"key": listed, "name": "output-listed", "n_cells": 30})
    saved["dataset"]["samples"].append({"key": listed + "/../output-other", "name": "output-other", "n_cells": 9})
    report = _restore(xe, saved)
    assert calls == [(listed, None)], "only the picker's own value is passed on, and only the listed sample"
    assert "sample output-listed added" in report["text"] and "extra-datasets-version" in report["bump"]
    assert report["mode"] == "independent" and "output-other" in report["needs_loading"][0]


def test_repl_gate_host_and_environment_cannot_be_set_from_a_file(xe, monkeypatch):
    env_before = dict(os.environ)
    repl_before = (xe._REPL_ENABLED, xe._REPL_MODE)
    saved = _doc(xe, {"point-size": 3})
    saved.update({"XENIUM_ENABLE_REPL": "remote", "XENIUM_HOST": "0.0.0.0", "host": "0.0.0.0", "port": 80,
                  "repl": True, "env": {"XENIUM_ENABLE_REPL": "1"}})
    saved["state"].update({"_REPL_ENABLED": True, "repl_enabled": True, "XENIUM_HOST": "0.0.0.0",
                           "environment": {"XENIUM_CACHE_DIR": "/elsewhere"}})
    saved["state"]["controls"].update({"repl-input": "import os; os.system('id')", "XENIUM_ENABLE_REPL": "1",
                                       "_REPL_ENABLED": True, "repl-run": 1})
    ran = []
    monkeypatch.setattr(xe, "run_repl", lambda *a, **k: ran.append(a))
    report = _restore(xe, saved)
    assert report["ok"] and report["controls"] == {"point-size": 3}
    assert dict(os.environ) == env_before and (xe._REPL_ENABLED, xe._REPL_MODE) == repl_before and not ran
    assert not xe._REPL_ENABLED, "the test session runs without the REPL"
    ignored = "\n".join(report["ignored"])
    for key in ("'XENIUM_ENABLE_REPL'", "'XENIUM_HOST'", "'host'", "'port'", "'repl'", "'env'", "state.'_REPL_ENABLED'",
                "state.'environment'", "control 'repl-input'", "control '_REPL_ENABLED'", "control 'repl-run'"):
        assert key in ignored, key
    assert "the REPL is never reachable from a session file" in ignored
    assert ignored.count("the server's settings are never read from a session file") >= 10
    # and nothing of a snapshot names them either
    text = xe._session_text(xe._session_snapshot(_defaults(xe)))
    assert "repl" not in text.lower() and "XENIUM_" not in text and "host" not in json.loads(text)


@pytest.mark.parametrize("text, message", [
    ("not json at all", "not valid JSON"), ("[1, 2]", "not a session file of this app"),
    ('{"format": "something-else", "version": 1}', "not a session file of this app"),
    ('{"format": "xenium-explorer-session", "version": 1, "state": {"controls": {"moran-k": NaN}}}', "not valid JSON"),
    ('{"format": "xenium-explorer-session", "version": 1, "version": 1}', "appears twice"),
    ("[" * 100_000, "nested too deeply"),
])
def test_a_file_that_is_not_a_session_file_is_refused(xe, text, message):
    report = xe._session_restore_text(text, _defaults(xe), "test")
    assert not report["ok"] and message in report["refused"] and report["text"].startswith("Not restored.")
    assert report["controls"] == {}


def test_a_file_that_is_too_large_is_refused_before_it_is_parsed(xe, monkeypatch):
    monkeypatch.setattr(xe, "_SESSION_MAX_BYTES", 2000)
    report = xe._session_restore_text(json.dumps(_doc(xe)).encode() + b" " * 2000, None, "test")
    assert not report["ok"] and "larger than" in report["refused"]
    big = "data:application/json;base64," + base64.b64encode(b"x" * 4000).decode()
    with pytest.raises(xe._SessionError, match="larger than"):
        xe._session_upload_bytes(big)
    assert xe._session_parse(json.dumps({"format": xe._SESSION_FORMAT, "version": 1}).encode())["version"] == 1
    with pytest.raises(xe._SessionError, match="not UTF-8"):
        xe._session_parse(b"\xff\xfe\x00")


# ── format version ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("version, message", [
    (2, "format version 2; this app reads version 1 and older"), (99, "format version 99"),
    (0, "unknown format version 0"), (-1, "unknown format version -1"), ("1", "no format version"),
    (1.0, "no format version"), (True, "no format version"), (None, "no format version"),
])
def test_format_version_is_checked_before_anything_is_applied(xe, D, version, message):
    saved = _doc(xe, {"point-size": 6}, rois=[{"name": "r", "cls": "region", "polygon_xy": _rect(D)}])
    saved["version"] = version
    if version is None:
        del saved["version"]
    report = _restore(xe, saved)
    assert not report["ok"] and message in report["refused"], report["refused"]
    assert report["controls"] == {} and not xe._roi_state["rois"]


def test_current_format_version_is_written_and_read(xe):
    doc = xe._session_snapshot(_defaults(xe))
    assert doc["version"] == xe._SESSION_VERSION == 1
    assert xe._session_parse(xe._session_text(doc))["version"] == 1
    assert _restore(xe, doc)["ok"]


# ── another dataset ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("change", ["key", "n_cells", "extra_sample", "no_dataset"])
def test_a_session_of_other_data_only_applies_what_does_not_depend_on_the_data(xe, D, change):
    methods = list(D["cluster_methods"])
    saved = _doc(xe, {"point-size": 6, "moran-k": 12, "lr-pairs": "A,B", "cluster-method": methods[1],
                      "gene-selector": D["gene_names"][3], "color-by": "gene", "baysor-xmin": 10.0,
                      "prox-labels": "cluster", "seg-source": "xenium",
                      # options of the loaded images / data size: bound to the data as well
                      "morph-channels": ["dapi"], "morph-enable": ["show"], "morph-zlevel": 1,
                      "renderer-mode": "webgl"},
                 rois=[{"name": "r", "cls": "region", "polygon_xy": _rect(D)}],
                 gene_sets={"s": list(D["gene_names"][:3])},
                 de_groups=[{"name": "g", "polygon_xy": _rect(D)}],
                 subset={"filters": {"min_transcripts": 1.0}, "n_cells": 3},
                 design={"names": {"A": "x", "B": "y"},
                         "samples": [{"key": xe._ds_key(D), "cond": "A", "donor": "", "batch": ""}]})
    if change == "key":
        saved["dataset"]["key"] = saved["dataset"]["samples"][0]["key"] = "/another/place/output-XETG-other"
        expect = "it was saved for another dataset"
    elif change == "n_cells":
        saved["dataset"]["samples"][0]["n_cells"] += 7
        expect = f"had {len(D['df']) + 7:,} cells, it has {len(D['df']):,}"
    elif change == "extra_sample":
        saved["dataset"]["samples"].append({"key": "/data/output-second", "name": "output-second", "n_cells": 11})
        expect = "it was saved with 2 sample(s), 1 are loaded"
    else:
        del saved["dataset"]
        expect = "does not say which data it was saved for"
    report = _restore(xe, saved)
    assert report["ok"] and report["mode"] == "independent"
    assert report["controls"] == {"point-size": 6, "moran-k": 12, "lr-pairs": "A,B"}
    text = report["text"]
    assert "restored in part" in text and expect in text, text
    for part in ("data-dependent control value(s)", "the sample design", "1 ROI(s)", "1 gene set(s)",
                 "1 DE group(s)", "the subset"):
        assert part in text, part
    assert not xe._roi_state["rois"] and not xe._geneset_names() and not xe._de_state["groups"]
    assert "_df_original" not in D and all(not r["cond"] for r in xe._design_samples())
    assert not report["bump"]
    if change == "extra_sample":
        assert "output-second" in report["needs_loading"][0] and "+ Add sample" in text


# ── jobs ─────────────────────────────────────────────────────────────────────
def test_restore_is_refused_while_a_job_runs(xe, D):
    saved = _doc(xe, {"point-size": 6}, rois=[{"name": "r", "cls": "region", "polygon_xy": _rect(D)}])
    state, lock = {"status": "running", "message": "", "result": None}, threading.Lock()
    release = threading.Event()
    job = xe._job_start("umap", "stub", lambda: release.wait(30), state=state, lock=lock)
    try:
        report = _restore(xe, saved)
        assert not report["ok"] and report["refused"].startswith("Still running: UMAP")
        assert report["controls"] == {} and not xe._roi_state["rois"]
        out = xe.session_action({"action": "upload", "controls": _defaults(xe)},
                                "data:application/json;base64," + base64.b64encode(json.dumps(saved).encode()).decode(),
                                0, 0, 0, 0, 0)
        assert out[0].startswith("Not restored. Still running: UMAP") and out[2] is xe.no_update
    finally:
        release.set()
    job.thread.join(10)
    # a state dict that is claimed before its job is registered counts too
    with xe._split_lock:
        xe._split_state["status"] = "running"
    try:
        assert "SPLIT" in _restore(xe, saved)["refused"]
    finally:
        with xe._split_lock:
            xe._split_state["status"] = "idle"
    assert _restore(xe, saved)["ok"] and len(xe._roi_state["rois"]) == 1


def test_restore_changes_state_through_the_functions_of_the_ui(xe, D, monkeypatch):
    """Design, ROI metadata, subset, counts mode and the ROI switch are set by the same
    functions the dialogs call (so caches, versions and cache files follow)."""
    called = []
    for name in ("_design_set", "_roi_apply_everywhere", "_roi_save_cache", "_gs_save", "_de_groups_save",
                 "subset", "toggle_roi_show"):
        real = getattr(xe, name)
        monkeypatch.setattr(xe, name, lambda *a, _n=name, _r=real, **k: (called.append(_n), _r(*a, **k))[1])
    extra = _fab(D, 30, 5)
    xe.EXTRA_DATASETS.append(extra)
    saved = _doc(xe, {"roi-show-toggle": []},
                 rois=[{"name": "r", "cls": "region", "polygon_xy": _rect(D)}],
                 gene_sets={"s": list(D["gene_names"][:3])}, de_groups=[{"name": "g", "polygon_xy": _rect(D)}],
                 subset={"filters": {"min_transcripts": 1.0}, "n_cells": None},
                 design={"names": {"A": "x", "B": "y"},
                         "samples": [{"key": xe._ds_key(D), "cond": "A", "donor": "", "batch": ""},
                                     {"key": xe._ds_key(extra), "cond": "B", "donor": "", "batch": ""}]})
    version, sub_version = xe._design_state["version"], xe._subset_version
    report = _restore(xe, saved)
    assert report["ok"] and report["mode"] == "full", report["text"]
    assert set(called) == {"_design_set", "_roi_apply_everywhere", "_roi_save_cache", "_gs_save", "_de_groups_save",
                           "subset", "toggle_roi_show"}
    assert xe._design_state["version"] > version and xe._subset_version > sub_version
    assert "x: S0" in xe._design_summary() or "A (x): S0" in xe._design_summary()


# ── slots, the automatic save, the callbacks ─────────────────────────────────
def test_automatic_save_keeps_the_session_of_the_run_before(xe, D, monkeypatch):
    page = _defaults(xe)
    assert xe._session_last_slot() is None
    assert "Last session (automatic): none" in xe._session_slots_text()
    assert not _restore_slot(xe, "last")["ok"]
    # run 1: the first tick only notes what an untouched page looks like
    assert xe._session_autosave(page) == "" and not os.path.exists(xe._session_path("last"))
    assert xe._session_autosave(page) == ""
    assert xe._session_autosave({**page, "moran-k": 9}) == "saved"
    assert xe._session_autosave({**page, "moran-k": 9}) == "", "unchanged: not written again"
    assert json.load(open(xe._session_path("last")))["kind"] == "auto"
    assert xe._session_last_slot() is None, "the save of this run is not 'the last session'"
    # run 2 (another server process): nothing is restored on its own …
    monkeypatch.setattr(xe, "_SESSION_PROCESS", "feedfacefeed")
    xe._session_auto.update({"baseline": None, "written": None})
    assert xe._session_last_slot() == "last"
    assert xe._session_autosave(page) == "", "an opened page does not replace the last session"
    # … until the user asks
    report = _restore_slot(xe, "last", page)
    assert report["ok"] and report["controls"] == {"moran-k": 9}
    # its own first save moves run 1's file aside; "last session" stays run 1's
    assert xe._session_autosave({**page, "moran-k": 4}) == "saved"
    assert xe._session_last_slot() == "prev"
    assert _restore_slot(xe, "last", page)["controls"] == {"moran-k": 9}
    assert json.load(open(xe._session_path("last")))["state"]["controls"]["moran-k"] == 4
    assert "Last session (automatic): 20" in xe._session_slots_text()
    # the automatic save writes no provenance record and survives 'Clean cache'
    xe._clean_cache_for_dataset()
    assert os.path.exists(xe._session_path("last")) and os.path.exists(xe._session_path("prev"))


def _restore_slot(xe, slot, current=None):
    return xe._session_restore_slot(slot, current)


def test_nothing_restores_a_session_unless_asked(xe):
    """Restore is reached from the dialog's callback and from the batch key only: no
    import-time code and no other callback calls it."""
    callers = {}
    for path in tl.app_sources():
        tree = ast.parse(path.read_text())
        for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.Module))]:
            body = fn.body if isinstance(fn, ast.Module) else [fn]
            for node in body:
                if isinstance(fn, ast.Module) and isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    continue
                for call in ast.walk(node):
                    name = getattr(getattr(call, "func", None), "attr", None) or getattr(getattr(call, "func", None), "id", "")
                    if isinstance(call, ast.Call) and str(name).startswith("_session_restore"):
                        callers.setdefault(path.stem, set()).add(getattr(fn, "name", "<module>"))
    assert set(callers) == {"session", "cb_session", "batch"}, callers
    assert callers["cb_session"] == {"session_action"} and callers["batch"] == {"_batch_apply_session"}
    assert "<module>" not in callers["session"]


def test_dialog_callbacks_save_download_upload_and_restore(xe, D):
    page = {**_defaults(xe), "moran-k": 7, "repl-input": "1+1", "annot-rds-path": "/x/ref.rds"}
    nu = xe.no_update
    is_open, slots = xe.open_session_modal(1)
    assert is_open is True and "Saved session: none" in slots and xe.close_session_modal(1) is False
    out = xe.session_action({"action": "save", "controls": page}, None, 0, 0, 0, 0, 0)
    assert len(out) == 11 and out[0].startswith("Session saved in the cache directory") and out[2] is nu
    assert "Saved session: 20" in out[1]
    on_disk = json.load(open(xe._session_path("saved")))
    assert on_disk["state"]["controls"]["moran-k"] == 7
    assert "repl-input" not in on_disk["state"]["controls"] and "annot-rds-path" not in on_disk["state"]["controls"]
    out = xe.session_action({"action": "download", "controls": page}, None, 0, 0, 0, 0, 0)
    assert out[3]["filename"].startswith("xenium_session_") and out[3]["filename"].endswith(".json")
    upload = "data:application/json;base64," + base64.b64encode(out[3]["content"].encode()).decode()
    doc = json.loads(out[3]["content"])
    doc["state"]["rois"] = [{"name": "r", "cls": "region", "polygon_xy": _rect(D)}]
    doc["state"]["controls"]["point-size"] = 4
    upload = "data:application/json;base64," + base64.b64encode(json.dumps(doc).encode()).decode()
    out = xe.session_action({"action": "upload", "t": 5, "controls": page}, upload, 3, 10, 0, 0, 2)
    assert out[0].startswith("Session restored") and out[2] == {"t": 5, "controls": {"point-size": 4}}
    assert out[4] is None, "the upload is cleared so the same file can be chosen again"
    assert out[6] is nu and out[7] == 11 and out[8] is nu and out[10] is nu
    out = xe.session_action({"action": "restore-saved", "controls": {**page, "moran-k": 3}}, None, 0, 0, 0, 0, 0)
    assert out[2]["controls"] == {"moran-k": 7}
    out = xe.session_action({"action": "upload", "controls": page}, "data:application/json;base64,!!!", 0, 0, 0, 0, 0)
    assert out[0].startswith("Not done. the upload could not be decoded") and out[2] is nu
    out = xe.session_action({"action": "auto", "controls": page}, None, 0, 0, 0, 0, 0)
    assert all(o is nu for o in out)
    with pytest.raises(Exception):
        xe.session_action(None)


def test_save_and_restore_are_in_the_provenance_log(xe, D):
    n = len(xe._prov_read())
    doc, _msg = xe._session_save(_defaults(xe), "saved")
    xe._session_autosave(_defaults(xe))
    xe._session_autosave({**_defaults(xe), "moran-k": 3})
    report = _restore_slot(xe, "saved", {**_defaults(xe), "moran-k": 3})
    refused = xe._session_restore_text("nope", None, "test")
    recs = xe._prov_read()[n:]
    assert [(r["kind"], r["title"], r["params"]["action"]) for r in recs] == [
        ("session", "Session saved", "save"), ("session", "Session restored", "restore")]
    assert report["ok"] and not refused["ok"]
    for r in recs:
        assert all(k in r for k in xe._PROV_REQUIRED) and json.loads(json.dumps(r, allow_nan=False)) == r
        assert r["data"]["dataset_key"] == doc["dataset"]["key"] == xe._dataset_key()
    assert recs[1]["params"]["from"] == "saved session" and recs[1]["params"]["mode"] == "full"
    md = xe._prov_methods_md(recs, latest_only=False)
    assert "## Session saved" in md and "## Session restored" in md and "could not be filled" not in md
    assert "no results" in md


# ── batch ────────────────────────────────────────────────────────────────────
def test_batch_config_applies_a_session_before_the_steps(xe, D, bundle_dir, tmp_path):
    saved = _doc(xe, {"moran-k": 5, "cluster-method": list(D["cluster_methods"])[1], "lr-nperm": 50},
                 rois=[{"name": "r", "cls": "region", "polygon_xy": _rect(D)}])
    path = tmp_path / "session.json"
    path.write_text(json.dumps(saved))
    cfg = {"data": {"path": str(bundle_dir)}, "output_dir": str(tmp_path / "out"), "session": str(path),
           "steps": [{"kind": "moran"}, {"kind": "moran", "name": "own", "params": {"k": 4}}]}
    lines = []
    try:
        out = xe._batch_run(cfg, dry_run=True, log=lines.append)
        assert out["ok"], out["errors"]
        assert any("session: Session restored" in ln for ln in lines)
        assert [r["name"] for r in xe._roi_state["rois"]] == ["r"]
        ctx = xe._batch_context(xe._batch_norm_config(cfg))
        assert ctx["cluster-method"] == list(D["cluster_methods"])[1]
        spec = xe._BATCH_STEPS["moran"]
        assert xe._batch_step_args(spec, {}, ctx, "moran")[1]["moran-k"] == 5            # the session's value
        assert xe._batch_step_args(spec, {"k": 4}, ctx, "moran")[1]["moran-k"] == 4      # a step's own comes first
        assert xe._batch_context(xe._batch_norm_config({**cfg, "cluster_method": "x"}))["cluster-method"] == "x"
        # without the key the next run starts from the layout's defaults again
        xe._batch_run({k: v for k, v in cfg.items() if k != "session"}, dry_run=True, log=lines.append)
        assert xe._batch_step_args(spec, {}, xe._batch_context(xe._batch_norm_config(cfg)), "moran")[1]["moran-k"] == 6
    finally:
        xe._batch_memo.pop("session", None)
    for bad, message in ((tmp_path / "missing.json", "session: cannot read"), (None, "not valid JSON")):
        if bad is None:
            bad = tmp_path / "bad.json"
            bad.write_text("{")
        with pytest.raises(xe._BatchError, match=message):
            xe._batch_run({**cfg, "session": str(bad)}, dry_run=True, log=lines.append)
    saved["version"] = 3
    path.write_text(json.dumps(saved))
    with pytest.raises(xe._BatchError, match="format version 3"):
        xe._batch_run(cfg, dry_run=True, log=lines.append)


def _files(root) -> dict:
    return {os.path.join(d, f): os.path.getsize(os.path.join(d, f)) for d, _s, fs in os.walk(root) for f in fs}


def test_batch_dry_run_with_a_session_writes_nothing(xe, D, bundle_dir, tmp_path, monkeypatch):
    """A dry run validates on the session's state in memory: no design file, ROI cache,
    gene-set or group file, no store write, no provenance record."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(xe, "CACHE_BASE_DIR", str(cache))
    saved = _doc(xe, {"moran-k": 5},
                 rois=[{"name": "r", "cls": "region", "polygon_xy": _rect(D)}],
                 gene_sets={"s": list(D["gene_names"][:3])}, de_groups=[{"name": "g", "polygon_xy": _rect(D)}],
                 design={"names": {"A": "x", "B": "y"},
                         "samples": [{"key": xe._ds_key(D), "cond": "A", "donor": "d", "batch": ""}]})
    path = tmp_path / "session.json"
    path.write_text(json.dumps(saved))
    cfg = {"data": {"path": str(bundle_dir)}, "output_dir": str(tmp_path / "out"), "session": str(path),
           "steps": [{"kind": "moran"}]}
    stores = []
    monkeypatch.setattr(xe, "_roi_write_all_zarrs_bg", lambda: stores.append(1))
    bundle_before, n_recent = tl.bundle_snapshot(bundle_dir), len(xe._prov_recent)
    lines = []
    try:
        out = xe._batch_run(cfg, dry_run=True, log=lines.append)
        assert out["ok"] and out["dry_run"], out["errors"]
        # the state is there for the checks …
        assert [r["name"] for r in xe._roi_state["rois"]] == ["r"] and xe._geneset_names() == ["s"]
        assert xe._design_samples()[0]["cond"] == "A" and xe._design_names()["A"] == "x"
        assert xe._batch_step_args(xe._BATCH_STEPS["moran"], {}, xe._batch_context(xe._batch_norm_config(cfg)),
                                   "moran")[1]["moran-k"] == 5
        assert any("Validation only" in ln for ln in lines)
        # … and nothing was written
        assert _files(cache) == {}, sorted(_files(cache))
        assert not (tmp_path / "out").exists() and not stores
        assert tl.bundle_snapshot(bundle_dir) == bundle_before and len(xe._prov_recent) == n_recent
        # the same call outside a dry run does write (so the check above can fail)
        xe._test_baseline.reset()
        xe._an_autoload()
        xe._stats_reset()
        xe._batch_apply_session(xe._batch_norm_config(cfg), lines.append, save=True)
        xe._test_baseline.settle()
        names = {os.path.basename(f).split("_")[0].split(".")[0] for f in _files(cache)}
        assert {"rois", "genesets", "de", "design"} <= names, names
        assert stores == [1] and len(xe._prov_recent) == n_recent + 1
    finally:
        xe._batch_memo.pop("session", None)


# ── the source ───────────────────────────────────────────────────────────────
_FORBIDDEN_MODULES = {"pickle", "cPickle", "_pickle", "dill", "cloudpickle", "joblib", "marshal", "shelve",
                      "importlib", "runpy", "subprocess", "ctypes", "yaml", "code", "codeop", "imp", "pkgutil",
                      "zipimport", "shutil", "glob", "urllib", "requests", "socket"}
_FORBIDDEN_BUILTINS = {"eval", "exec", "compile", "__import__", "execfile", "globals", "locals", "vars", "setattr",
                       "delattr", "input", "breakpoint"}
_FORBIDDEN_METHODS = {"system", "popen", "spawn", "load_module", "import_module", "read_pickle", "unpack_archive",
                      "literal_eval", "listdir", "walk", "scandir", "remove", "unlink", "rmtree", "chmod",
                      "symlink", "putenv", "eval", "exec", "loads_unsafe", "load"}


@pytest.mark.parametrize("module", ["session", "cb_session"])
def test_restore_path_has_no_pickle_eval_exec_or_import(module):
    path = tl.APP_PACKAGE / f"{module}.py"
    tree = ast.parse(path.read_text())
    opens = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = {a.name.split(".")[0] for a in node.names}
            assert not names & _FORBIDDEN_MODULES, f"{path.name}:{node.lineno} imports {names & _FORBIDDEN_MODULES}"
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                assert (node.module or "").split(".")[0] not in _FORBIDDEN_MODULES, f"{path.name}:{node.lineno}"
        elif isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            bad = _FORBIDDEN_METHODS if isinstance(node.func, ast.Attribute) else _FORBIDDEN_BUILTINS
            assert name not in bad, f"{path.name}:{node.lineno} calls {name}()"
            if name == "getattr":       # only ever a fixed attribute name, never one read from a file
                assert isinstance(node.args[1], ast.Constant), f"{path.name}:{node.lineno}"
            if name == "open":
                opens.append(node)
        elif isinstance(node, ast.Attribute) and node.attr in ("environ", "__dict__", "__globals__", "__builtins__"):
            raise AssertionError(f"{path.name}:{node.lineno} touches {node.attr}")
    if module == "cb_session":
        assert not opens
        return
    # the only files session.py opens are its own slots under the cache directory
    by_fn = {fn.name for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)
             for n in ast.walk(fn) if n in opens}
    assert by_fn == {"_session_write", "_session_read_slot", "_session_restore_slot"}, by_fn
    for node in opens:
        arg = node.args[0]
        assert isinstance(arg, ast.Name) and arg.id == "tmp" or (
            isinstance(arg, ast.Call) and arg.func.id == "_session_path"), ast.dump(arg)[:80]
    slot = tl.function_node(tree, "_session_path")
    assert "_prov_dataset_id" in ast.unparse(slot) and "doc" not in {a.arg for a in slot.args.args}


def test_json_is_the_only_parser_and_refuses_non_json_constants(xe):
    for const in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(xe._SessionError, match="not valid JSON"):
            xe._session_parse('{"format": "xenium-explorer-session", "version": 1, "x": %s}' % const)
    # what comes back from a parse is plain data
    doc = xe._session_parse(xe._session_text(xe._session_snapshot(_defaults(xe))))

    def plain(v):
        if isinstance(v, dict):
            return all(isinstance(k, str) and plain(x) for k, x in v.items())
        if isinstance(v, list):
            return all(plain(x) for x in v)
        return v is None or isinstance(v, (str, int, float, bool))
    assert plain(doc) and copy.deepcopy(doc) == doc
