"""Methods and provenance record (xenium_app/provenance.py).

* the wiring, in the source: every job the registry starts belongs to a known analysis
  kind and its function calls ``provenance._prov_record`` (so a new job cannot forget it);
* the record: required fields, JSON, per-dataset append-only file, bounded size, survives
  a restart;
* real runs on the tiny bundle write a record through the job's own code path;
* the methods text: every kind has a template; snapshots for a few kinds;
* the id of the record is in the CSV headers.

tests/test_zz_provenance_coverage.py checks, at the end of a full run of the default
tier, the records that the other test modules' jobs wrote.
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import re

import numpy as np
import pandas as pd
import pytest

import xe_testlib as tl

SNAP = pathlib.Path(__file__).resolve().parent / "snapshots"


@pytest.fixture(autouse=True)
def _keep_session_records(xe):
    """The records this module fabricates stay out of the session's list, which
    tests/test_zz_provenance_coverage.py reads for the records of real job runs."""
    before = list(xe._prov_recent)
    yield
    xe._prov_recent[:] = before


# ── 1. the wiring, in the source ─────────────────────────────────────────────
def _functions(tree) -> dict:
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(n.name, []).append(n)
    return out


def _records_called(fn_node) -> list:
    """Kinds of the provenance._prov_record(...) calls inside a function ('?' when the
    kind is not a literal)."""
    out = []
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_prov_record":
            a = n.args[0] if n.args else None
            out.append(a.value if isinstance(a, ast.Constant) else "?")
    return out


def _job_starts() -> list:
    """(file, line, kind or None, target expression) of every jobs._job_start(...) call."""
    out = []
    for path in tl.app_sources():
        tree = ast.parse(path.read_text())
        for n in ast.walk(tree):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_job_start"
                    and len(n.args) >= 3):
                kind = n.args[0].value if isinstance(n.args[0], ast.Constant) else None
                out.append((path, n.lineno, kind, n.args[2]))
    return out


# a job whose record is written by another function, after its result is installed:
# job function -> (module of the writer, writer, module of its caller, caller)
_RECORDED_AFTER_INSTALL = {"_h5ad_job": ("h5ad", "_h5ad_record", "cb_data", "_h5ad_install")}

# a job whose target is a variable: the functions it can be
_DYNAMIC_TARGETS = {"loader": ["_load_cached_baysor", "_load_cached_proseg"]}


def test_every_registered_job_kind_writes_a_provenance_record(xe):
    starts = _job_starts()
    assert len(starts) >= 35, "the scan no longer finds the jobs._job_start calls"
    trees = {p: ast.parse(p.read_text()) for p in tl.app_sources()}
    by_module = {p.stem: _functions(t) for p, t in trees.items()}
    missing, kinds_seen = [], set()
    for path, line, kind, target in starts:
        where = f"{path.name}:{line}"
        if kind is not None:
            kinds_seen.add(kind)
            assert kind in xe._JOB_KIND_NAMES, f"{where}: job kind {kind!r} has no name"
            assert kind in xe._PROV_KINDS or kind in xe._PROV_NOT_ANALYSES, \
                f"{where}: job kind {kind!r} is neither an analysis kind of the provenance record " \
                f"(_PROV_KINDS) nor listed as not an analysis (_PROV_NOT_ANALYSES)"
            if kind in xe._PROV_NOT_ANALYSES:
                continue
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            cands = by_module.get(target.value.id, {}).get(target.attr, [])
        elif isinstance(target, ast.Name):
            names = _DYNAMIC_TARGETS.get(target.id, [target.id])
            cands = [f for nm in names for f in by_module[path.stem].get(nm, [])
                     + [g for m in by_module.values() for g in m.get(nm, []) if nm.startswith("_load_cached")]]
            if target.id == "_run":          # a nested _run: the one defined just above the call
                cands = [f for f in by_module[path.stem].get("_run", []) if f.end_lineno < line][-1:]
        else:
            cands = []
        assert cands, f"{where}: cannot find the function the job runs ({ast.dump(target)[:80]})"
        for fn in cands:
            got = _records_called(fn)
            if not got and fn.name in _RECORDED_AFTER_INSTALL:
                # the job only prepares the data; the record is written once the data is
                # installed (written by the job it would describe the previous dataset)
                mod, writer, caller_mod, caller = _RECORDED_AFTER_INSTALL[fn.name]
                got = [k for f in by_module[mod].get(writer, []) for k in _records_called(f)]
                src = ast.get_source_segment(trees[next(p for p in trees if p.stem == caller_mod)].body and
                                             next(p for p in trees if p.stem == caller_mod).read_text(),
                                             by_module[caller_mod][caller][0]) or ""
                assert f"{writer}(" in src, f"{caller_mod}.{caller}() no longer calls {writer}"
            if not got:
                missing.append(f"{where}: {fn.name}() never calls provenance._prov_record")
            elif kind is not None:
                rec_kinds = set(got) - {"?"}
                # a job registered as "de" may record a more specific kind (pseudobulk)
                ok = kind in rec_kinds or (kind == "de" and rec_kinds <= {"de", "pseudobulk"})
                if not ok:
                    missing.append(f"{where}: {fn.name}() records {sorted(rec_kinds)}, registered as {kind!r}")
    assert not missing, "\n".join(missing)
    # every job kind the app names is started somewhere, so the check above saw it
    assert set(xe._JOB_KIND_NAMES) <= kinds_seen | {"baysor", "proseg"}, set(xe._JOB_KIND_NAMES) - kinds_seen


def test_every_record_call_names_a_known_kind_and_every_kind_is_recorded_and_has_a_template(xe):
    called = set()
    for path in tl.app_sources():
        for kind in _records_called(ast.parse(path.read_text())):
            assert kind != "?", f"{path.name}: _prov_record with a non-literal kind"
            assert kind in xe._PROV_KINDS, f"{path.name}: unknown provenance kind {kind!r}"
            called.add(kind)
    assert called == set(xe._PROV_KINDS), set(xe._PROV_KINDS) ^ called
    assert set(xe._PROV_TEMPLATES) == set(xe._PROV_KINDS)
    assert set(xe._PROV_ORDER) == set(xe._PROV_KINDS) and len(xe._PROV_ORDER) == len(set(xe._PROV_ORDER))


def test_every_csv_the_app_writes_names_its_record(xe):
    """Every function that builds a result CSV goes through _prov_csv_stamp."""
    stamped, writers = 0, 0
    for path in tl.app_sources():
        tree = ast.parse(path.read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            src = ast.get_source_segment(path.read_text(), fn) or ""
            is_writer = (re.fullmatch(r"_\w+_csv(_header)?", fn.name) and fn.name != "_prov_csv_stamp"
                         and not fn.name.startswith("_prov")) or \
                        (fn.name.startswith("download_") and ".to_csv(" in src and "header" in src)
            if is_writer:
                writers += 1
                if "_prov_csv_stamp(" in src or re.search(r"_\w+_csv(_header)?\(res\)", src):
                    stamped += 1
                else:
                    raise AssertionError(f"{path.name}: {fn.name}() writes a CSV without the provenance line")
    assert writers >= 14 and stamped == writers


# ── 2. the record ────────────────────────────────────────────────────────────
@pytest.fixture()
def prov_dir(xe, tmp_path, monkeypatch):
    """Records of the test go to their own cache root."""
    monkeypatch.setattr(xe, "CACHE_BASE_DIR", str(tmp_path))
    return tmp_path / "provenance"


def _fake_result():
    return {"table": pd.DataFrame({"gene": list("abcd"), "padj": [0.001, 0.2, np.nan, 0.04]}),
            "n_cells": 23, "k": 6, "corrected": False, "n_uncorrected": 0, "seg_source": "xenium",
            "seconds": 0.5, "samples": [{"label": "S0", "n_cells": 23}], "expr": np.zeros((3, 3)),
            "params": {"alpha": 0.05}, "notes": ["a note of the job"], "seed": 3}


def test_record_has_the_required_fields_and_is_plain_json(xe, prov_dir):
    res = _fake_result()
    msg = ("Moran's I, kNN graph (k = 6), S0: 23 cells, 4 genes (original counts, 0.5 s). "
           "Cells of 2 samples were pooled. The smallest possible p is 0.001.\n"
           "Unit of the p-values: n = 23 cells, treated as independent.\n⚠ Something to know.")
    rec = xe._prov_record("moran", seg_source="xenium", corrected=False, params={"k": 6, "none": None},
                          result=res, message=msg, labels=("cluster", "graphclust"))
    assert rec is not None and all(k in rec for k in xe._PROV_REQUIRED)
    assert rec["kind"] == "moran" and rec["title"] == xe._PROV_KINDS["moran"]
    assert res["prov_id"] == rec["id"] and re.fullmatch(r"prov-\d{8}-\d{6}-[0-9a-f]{6}", rec["id"])
    assert json.loads(json.dumps(rec)) == rec
    d = rec["data"]
    assert d["n_cells"] == len(xe.DATA["df"]) and d["n_genes"] == len(xe.DATA["gene_names"])
    assert d["seg_source"] == "xenium" and d["seg_tool"] == "xenium" and d["n_samples"] == 1
    assert d["counts"] == {"requested": "original", "used": "original", "n_cells_left_out": 0}
    assert d["labels"]["label_kind"] == "cluster" and d["subset"] is None
    # parameters: user-set ones and the defaults the job resolved
    assert rec["params"]["k"] == 6 and rec["params"]["resolved"] == {"alpha": 0.05}
    # the unit is the job's own sentence, not a re-derivation
    assert rec["unit"]["statement"] == "n = 23 cells, treated as independent."
    assert rec["unit"]["source"] == "status line" and rec["unit"]["n"]["n_cells"] == 23
    # headline numbers: scalars kept, tables summarised, arrays dropped
    h = rec["headline"]
    assert h["table"]["table_rows"] == 4 and h["table"]["n_tested"] == 3 and h["table"]["n_fdr_below_0.05"] == 2
    assert h["k"] == 6 and "expr" not in h and h["n_genes_in_matrix"] == 3
    # caveats, verbatim
    assert "a note of the job" in rec["warnings"] and "Something to know." in rec["warnings"]
    assert "Cells of 2 samples were pooled." in rec["warnings"]
    assert "The smallest possible p is 0.001." in rec["warnings"]
    assert rec["seeds"] == {"seed": 3} and rec["cache"] == "computed"
    sw = rec["software"]
    assert sw["python"] and sw["numpy"] == np.__version__ and sw["pandas"] == pd.__version__
    assert "scipy" in sw and "scikit-learn" in sw and "app_commit" in sw
    assert rec["status_line"] == msg and rec["design"] is None
    # stored: one JSON line in the dataset's file
    files = list(prov_dir.glob("*.jsonl"))
    assert [f.name for f in files] == [xe._prov_dataset_id() + ".jsonl"]
    assert [json.loads(l)["id"] for l in files[0].read_text().splitlines()] == [rec["id"]]
    assert xe._prov_get(rec["id"]) == rec and xe._prov_read() == [rec]


def test_record_describes_an_active_subset(xe, prov_dir):
    med = int(xe.DATA["df"]["transcript_counts"].median())
    n_all = len(xe.DATA["df"])
    try:
        xe.subset(min_transcripts=med)
        rec = xe._prov_record("moran", params={"k": 6}, message="done")
        sub = rec["data"]["subset"]
        assert sub["of"] == n_all and 0 < sub["n_cells"] == len(xe.DATA["df"]) < n_all
        assert sub["filters"] == [{"min_transcripts": med}]
        assert f"A cell subset was active in the newest record ({sub['n_cells']:,} of {n_all:,} cells" \
            in xe._prov_methods_md([rec])
    finally:
        xe.unsubset()
    assert xe._prov_record("moran", message="done")["data"]["subset"] is None


def test_cache_hits_and_failures(xe, prov_dir, capsys):
    assert xe._prov_record("annotation", params={"method": "celltypist"}, message="Done (cached) — 4 cell types",
                           result=xe._prov_labels_summary(pd.Series(["a", "b", None])))["cache"] == "cache"
    assert xe._prov_record("banksy", result={"cached": True}, message="x")["cache"] == "cache"
    assert xe._prov_record("joint", result={"from_cache": False}, message="x")["cache"] == "computed"
    assert xe._prov_record("baysor", message="Done — 5 cells", cache=True)["cache"] == "cache"
    r = xe._prov_read()[0]
    assert r["headline"] == {"n_cells_labelled": 2, "n_cell_types": 2}
    # a record that cannot be written never breaks the analysis
    class Bad:
        def __iter__(self):
            raise RuntimeError("boom")
    assert xe._prov_record("moran", params=Bad()) is None
    assert "could not be written" in capsys.readouterr().out
    assert len(xe._prov_read()) == 4


def test_file_is_append_only_and_bounded(xe, prov_dir, monkeypatch):
    monkeypatch.setattr(xe, "_PROV_MAX_BYTES", 30_000)
    monkeypatch.setattr(xe, "_PROV_KEEP", 12)
    ids = []
    for i in range(60):
        ids.append(xe._prov_record("geneset", params={"gene_set": f"set{i}"}, message="m" * 500)["id"])
        if i == 3:
            first = (prov_dir / (xe._prov_dataset_id() + ".jsonl")).read_text()
        if i == 5:      # appended to, not rewritten
            assert (prov_dir / (xe._prov_dataset_id() + ".jsonl")).read_text().startswith(first)
    path = prov_dir / (xe._prov_dataset_id() + ".jsonl")
    kept = [r["id"] for r in xe._prov_read()]
    assert os.path.getsize(path) <= 30_000 + 2_000
    assert 0 < len(kept) < 60 and kept == ids[-len(kept):]        # the newest ones, in order
    # another dataset has another file; unreadable lines are skipped
    other = xe._prov_path("some/other/dataset")
    assert other != str(path)
    with open(path, "a") as fh:
        fh.write("{not json\n\n")
    assert [r["id"] for r in xe._prov_read()] == kept


def test_records_survive_a_restart_and_clean_cache(xe, bundle_dir, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    code = """
    rec = xe._prov_record("compartments", params={"bin_um": 20}, message="first run")
    RESULT["id"] = rec["id"]
    RESULT["n"] = len(xe._prov_read())
    RESULT["clean"] = xe._clean_cache_for_dataset()
    RESULT["n_after_clean"] = len(xe._prov_read())
    """
    a = tl.run_in_fresh_app(code, bundle_dir, cache)
    assert a["n"] == 1 and a["n_after_clean"] == 1          # Clean Cache leaves the record of what was run
    b = tl.run_in_fresh_app("""
    RESULT["ids"] = [r["id"] for r in xe._prov_read()]
    RESULT["rows"] = xe._prov_rows()
    RESULT["rows_kind"] = xe._prov_rows("moran")
    RESULT["opts"] = xe._prov_kind_options()
    RESULT["md"] = xe._prov_methods_md()
    """, bundle_dir, cache)
    assert b["ids"] == [a["id"]]
    assert b["rows"][0]["id"] == a["id"] and b["rows"][0]["kind"] == "compartments" and b["rows_kind"] == []
    assert b["opts"] == [{"label": "Tissue compartments", "value": "compartments"}]
    assert "## Tissue compartments" in b["md"] and a["id"] in b["md"]


# ── 3. real runs on the tiny bundle ──────────────────────────────────────────
def _last(xe, kind):
    recs = [r for r in xe._prov_read() if r["kind"] == kind]
    assert recs, f"no '{kind}' record was written"
    return recs[-1]


def _check(xe, rec):
    assert all(k in rec for k in xe._PROV_REQUIRED), set(xe._PROV_REQUIRED) - set(rec)
    assert json.loads(json.dumps(rec, allow_nan=False)) == rec
    assert rec["status_line"] and rec["software"]["python"] and rec["data"]["n_cells"] > 0
    md = xe._prov_methods_md([rec])
    assert f"## {rec['title']}" in md and rec["id"] in md and "could not be filled" not in md
    return md


def test_moran_job_writes_its_record(xe, prov_dir):
    xe._run_moran("xenium", False, 6)
    with xe._moran_lock:
        res, msg = xe._moran_state["result"], xe._moran_state["message"]
    try:
        rec = _last(xe, "moran")
        md = _check(xe, rec)
        assert rec["params"] == {"k": 6} and rec["status_line"] == msg and rec["cache"] == "computed"
        assert res["prov_id"] == rec["id"]
        assert rec["unit"]["statement"].startswith("the cells of each sample (")
        assert rec["headline"]["table"]["n_tested"] == int(res["table"]["padj"].notna().sum())
        assert rec["data"]["counts"]["used"] == "original"
        assert "treated as independent observations" in md and "k = 6" in md
        # ... and the CSV download names it
        out = xe.download_moran_table(1)["content"]
        assert f"# provenance record: {rec['id']}" in out
        assert pd.read_csv(pd.io.common.StringIO(out), comment="#").shape[0] == len(res["table"])
    finally:
        xe._stats_reset()


def test_compartments_job_and_composition_write_their_records(xe, prov_dir):
    try:
        xe._run_compartments("xenium", None, None, None, {"bin_um": 25})
        rec = _last(xe, "compartments")
        md = _check(xe, rec)
        assert rec["params"]["bin_um"] == 25 and rec["params"]["resolved"]["bin_um"] == 25
        assert rec["params"]["resolved"]["radius"] == xe._DENSITY_R_UM          # a default that was used
        assert "25 µm bins" in md
        method = xe.DATA["cluster_methods"][0]
        res, msg = xe._comp_run("cluster", "xenium", method, "logit", "moderated")
        rec = _last(xe, "composition")
        _check(xe, rec)
        assert res["prov_id"] == rec["id"] and rec["params"]["transform"] == "logit"
        assert rec["design_used"] is True and rec["design"] is None                # one sample: no design
        assert f"# provenance record: {rec['id']}" in xe._comp_csv(res)
        assert xe._comp_csv(res).startswith("# Composition")
    finally:
        xe._controls_reset()
        xe._stats_reset()


def test_spatial_analysis_and_niches_write_their_records(xe, prov_dir):
    n = len(xe.DATA["df"])
    labels = pd.Series(np.where(np.arange(n) % 2 == 0, "even", "odd"), index=xe.DATA["df"].index.astype(str))
    with xe._annot_lock:
        had = xe._annot_state.get("labels_provtest")
        xe._annot_state["labels_provtest"] = labels
    try:
        xe._run_spatial_analysis("labels_provtest", 4, None)
        rec = _last(xe, "spatial")
        _check(xe, rec)
        assert rec["params"]["n_neighbors"] == 4 and rec["headline"]["n_cell_types"] == 2
        xe._run_spatial_niches(2, 4, "labels_provtest", None)
        rec = _last(xe, "niches")
        md = _check(xe, rec)
        assert rec["params"]["n_clusters"] == 2 and "KMeans into 2 niches" in md
    finally:
        with xe._annot_lock:
            xe._annot_state.pop("labels_provtest", None)
            xe._spatial_state.update({"status": "idle", "result": None})
            xe._niche_state.update({"status": "idle"})
        for col in ("niche",):
            if col in xe.DATA["df"].columns:
                xe.DATA["df"].drop(columns=[col], inplace=True)


def test_a_failed_or_stale_job_writes_no_record(xe, prov_dir):
    xe._run_spatial_analysis("labels_that_do_not_exist", 4, None)            # reports an error
    assert xe._spatial_state["status"] == "error" and xe._prov_read() == []
    gen0 = xe._dataset_gen
    real = xe._moran_sample

    def swap_then_compute(*a, **k):
        xe._dataset_gen += 1                       # Load Sample while the job computes
        return real(*a, **k)

    try:
        xe._moran_sample = swap_then_compute
        with pytest.raises(xe._DatasetChanged):
            xe._run_moran("xenium", False, 6)
        assert xe._prov_read() == []                 # nothing was published, nothing recorded
    finally:
        xe._moran_sample = real
        xe._dataset_gen = gen0
        xe._stats_reset()
        with xe._spatial_lock:
            xe._spatial_state.update({"status": "idle", "message": "", "result": None})


# ── 4. the Provenance tab ────────────────────────────────────────────────────
def test_tab_lists_filters_shows_and_exports(xe, prov_dir):
    a = xe._prov_record("moran", params={"k": 6}, message="Moran done.\nUnit of the p-values: cells.")
    b = xe._prov_record("lr", params={"k": 5, "n_perm": 99, "seed": 1}, message="LR done.")
    rows, opts, msg = xe.update_prov_table("tab-provenance", 0, None, 0)
    assert [r["id"] for r in rows] == [b["id"], a["id"]]                    # newest first
    assert {o["value"] for o in opts} == {"moran", "lr"} and "2 record(s)" in msg
    rows, _, msg = xe.update_prov_table("tab-provenance", 0, "lr", 0)
    assert [r["id"] for r in rows] == [b["id"]]
    with pytest.raises(Exception):                                          # another tab: nothing is read
        xe.update_prov_table("tab-umap", 0, None, 0)
    detail = xe.show_prov_detail({"row": 0, "column": 0}, rows)
    assert json.loads(detail)["id"] == b["id"]
    doc = json.loads(xe._prov_export_json())
    assert doc["n_records"] == 2 and [r["id"] for r in doc["records"]] == [a["id"], b["id"]]
    md = xe._prov_methods_md()
    assert md.index("## Spatially variable genes") < md.index("## Ligand-receptor scores")
    assert "## References" in md and "Moran PAP" in md and "## All records" in md
    layout_ids = {c.id for c in tl_walk(xe.app.layout) if getattr(c, "id", None)}
    assert {"prov-table", "prov-kind", "prov-detail", "prov-export-md-btn", "prov-export-json-btn",
            "prov-download", "prov-refresh-btn", "prov-status"} <= layout_ids


def tl_walk(node):
    yield node
    kids = getattr(node, "children", None)
    if isinstance(kids, (list, tuple)):
        for k in kids:
            yield from tl_walk(k)
    elif kids is not None and hasattr(kids, "to_plotly_json"):
        yield from tl_walk(kids)


# ── 5. the methods text ──────────────────────────────────────────────────────
def _snap_record(kind, title=None, **over):
    rec = {"id": f"prov-20260102-030405-{kind[:6]:0<6}", "schema": 1, "kind": kind, "title": title or kind,
           "time": "2026-01-02T03:04:05+00:00", "time_local": "2026-01-02T03:04:05", "job_id": None,
           "data": {"dataset_key": "/data/sectionA", "dataset_name": "sectionA", "n_samples": 1,
                    "samples": [{"label": "S0", "key": "/data/sectionA", "name": "sectionA",
                                 "n_cells": 1200, "n_genes": 300}],
                    "n_cells": 1200, "n_genes": 300, "seg_source": "xenium", "seg_tool": "xenium",
                    "seg_tag": "", "subset": None,
                    "counts": {"requested": "original", "used": "original", "n_cells_left_out": None},
                    "labels": None},
           "params": {}, "unit": {"statement": "", "source": "", "n": {}}, "design": None,
           "design_used": False, "headline": {}, "warnings": [], "seeds": {}, "cache": "computed",
           "runtime_s": 1.5,
           "software": {"app_commit": "abc123", "python": "3.11.6", "numpy": "1.26.0", "scipy": "1.11.0",
                        "pandas": "2.1.0", "scikit-learn": "1.3.0"},
           "status_line": "", "outputs": []}
    rec.update(over)
    return rec


def _snap_records(xe) -> dict:
    two = {"dataset_key": "/data/sectionA", "dataset_name": "sectionA", "n_samples": 1,
           "samples": [{"label": "S0", "key": "/data/sectionA", "name": "sectionA", "n_cells": 1200, "n_genes": 300}],
           "n_cells": 1200, "n_genes": 300, "seg_source": "xenium", "seg_tool": "xenium", "seg_tag": "",
           "subset": None, "labels": {"label_kind": "celltype:celltypist", "cluster_method": None,
                                      "category": "cell type"}}
    de = _snap_record(
        "de", xe._PROV_KINDS["de"],
        data={**two, "counts": {"requested": "corrected", "used": "SPLIT-corrected", "n_cells_left_out": 40}},
        params={"group_a": {"kind": "celltype", "value": "T cell"}, "group_b": None, "color_by": "cell_type",
                "options": {"leak_gate": True}},
        headline={"label_a": "T cell", "label_b": "all other cells", "n_a": 200, "n_b": 960, "n_genes": 300,
                  "corrected": True, "n_uncorrected": 40},
        unit={"statement": "n = 1,160 cells, treated as independent (they are not, when the groups come "
                           "from few samples: the p-values are then far too small).",
              "source": "status line", "n": {"n_a": 200, "n_b": 960}},
        warnings=["Leak gate: 12 of 300 genes flagged.", "40 cells have no SPLIT-corrected counts and were left out."],
        status_line="T cell (A, 200 cells) vs all other cells (B, 960 cells): 300 genes, 25 with FDR < 0.05 "
                    "(SPLIT-corrected counts, 0.4 s).\nUnit of the p-values: n = 1,160 cells, treated as "
                    "independent (they are not, when the groups come from few samples: the p-values are then "
                    "far too small).")
    design = {"condition_names": {"A": "treated", "B": "control"},
              "samples": [{"label": f"S{i}", "name": f"s{i}", "condition": "A" if i < 3 else "B",
                           "donor": f"d{i % 3}", "batch": "", "n_cells": 1000} for i in range(6)],
              "n_a": 3, "n_b": 3, "donors_given": True, "paired_donors": True, "summary": "x"}
    pb = _snap_record(
        "pseudobulk", xe._PROV_KINDS["pseudobulk"],
        data={**two, "counts": {"requested": "original", "used": "original", "n_cells_left_out": None}},
        params={"population": "T cell", "min_cells": 10, "min_count": 10, "shrink": True, "engine": "auto"},
        headline={"population": "T cell", "n_a": 3, "n_b": 3, "unit_word": "donors", "n_genes": 280,
                  "n_filtered": 20, "n_padj_na": 0, "min_count": 10, "design": "~ donor + condition",
                  "method_label": "DESeq2 (Wald test)", "shrink": "apeglm", "counts_rounded": False},
        design=design, design_used=True,
        unit={"statement": "n = 3 donors per condition (the donors: not their sections, not the cells).",
              "source": "status line", "n": {"n_a": 3, "n_b": 3, "unit_word": "donors"}},
        software={"app_commit": "abc123", "python": "3.11.6", "numpy": "1.26.0",
                  "r": {"R": "4.3.1 (2023-06-16)", "DESeq2": "1.40.2", "edgeR": "not found"}},
        status_line="Pseudobulk (T cell): 280 genes tested, 4 with FDR < 0.05 (3.2 s).")
    lr = _snap_record(
        "lr", xe._PROV_KINDS["lr"],
        data={**two, "counts": {"requested": "original", "used": "original", "n_cells_left_out": None}},
        params={"pairs": "CXCL12,CXCR4\nCCL19,CCR7", "k": 6, "n_perm": 999, "seed": 0}, seeds={"seed": 0},
        unit={"statement": "n = 1,100 cells, treated as exchangeable within their sample (label permutation); "
                           "not a test across samples.", "source": "status line", "n": {}},
        warnings=["999 permutations, seed 0; smallest possible p 0.001."],
        status_line="2 of 2 ligand–receptor pairs have both genes on the panel.")
    bk = _snap_record(
        "banksy", xe._PROV_KINDS["banksy"],
        data={**two, "labels": None, "counts": {"requested": "original", "used": "original", "n_cells_left_out": None}},
        params={"lam": 0.8, "k": 15, "resolution": 1.0, "seed": 0, "resolved": {"n_pcs": 20}}, seeds={"seed": 0},
        cache="cache", status_line="Banksy-style λ=0.8: 7 clusters (from the cache).")
    return {"de": de, "pseudobulk": pb, "lr": lr, "banksy": bk}


def _stable(md: str) -> str:
    return "\n".join(l for l in md.splitlines() if not l.startswith("*Generated by")) + "\n"


@pytest.mark.parametrize("kind", ["de", "pseudobulk", "lr", "banksy"])
def test_methods_text_snapshot(xe, kind):
    md = _stable(xe._prov_methods_md([_snap_records(xe)[kind]]))
    path = SNAP / f"methods_{kind}.md"
    if os.environ.get("XENIUM_UPDATE_SNAPSHOTS") == "1":
        path.write_text(md)
    assert md == path.read_text(), f"methods text of '{kind}' changed (XENIUM_UPDATE_SNAPSHOTS=1 rewrites {path.name})"


def test_methods_text_does_not_overclaim(xe):
    recs = _snap_records(xe)
    md = xe._prov_methods_md(list(recs.values()))
    # cell-level tests say so; the unit is the app's sentence
    assert "Cells were treated as independent observations" in md
    assert "Unit of the p-values, as stated by the app: n = 3 donors per condition" in md
    assert "The unit of the p-values is the cell (labels exchangeable within a sample)" in md
    # corrected counts and cells left out
    assert "SPLIT-corrected counts were used; 40 cells without SPLIT-corrected counts were left out." in md
    # the design, with the user's own labels
    assert "condition A ('treated'): S0 (donor d0), S1 (donor d1), S2 (donor d2)" in md
    # Banksy: named as a re-implementation, not validated; cache hit said
    assert "has not been validated against the reference" in md and "loaded from the cache" in md
    # each caveat the app raised
    assert "- Leak gate: 12 of 300 genes flagged." in md and "- 999 permutations, seed 0; smallest possible p 0.001." in md
    # multiple testing and software versions
    assert md.count("Benjamini-Hochberg") >= 3 and "DESeq2 1.40.2 (R)" in md and "R 4.3.1" in md
    # references: numbered, only those used, uncertain ones marked
    refs = md.split("## References")[1].split("## All records")[0]
    assert "Love MI" in refs and "Singhal V" in refs and "Petukhov" not in refs
    for key, text in xe._PROV_REFS.items():
        assert re.search(r"\(\d{4}\)|\[verify", text), key
    # latest_only: an older run of the same analysis is listed but not written up twice
    older = dict(recs["lr"], id="prov-20250101-000000-older0", time="2025-01-01T00:00:00+00:00")
    md2 = xe._prov_methods_md([older, recs["lr"]])
    assert md2.count("## Ligand-receptor scores") == 1 and "prov-20250101-000000-older0" in md2.split("## All records")[1]
    assert xe._prov_methods_md([older, recs["lr"]], latest_only=False).count("## Ligand-receptor scores") == 2


@pytest.mark.parametrize("kind", sorted(
    ["baysor", "proseg", "imgseg", "patches", "annotation", "split", "spage", "umap", "spatial", "niches", "de",
     "pseudobulk", "proximity", "contact", "geneset", "density", "composition", "moran", "lr", "lrdecomp",
     "compartments", "spillover", "segqc", "coloc", "aggregates", "segfree", "banksy", "joint", "txfactor",
     "h5ad", "save", "cellnest"]))
def test_every_kind_gives_a_record_and_a_methods_paragraph(xe, prov_dir, kind):
    """Fabricated call of the one helper for each analysis kind: required fields, and a
    template that renders from what the record holds."""
    assert kind in xe._PROV_KINDS
    rec = xe._prov_record(kind, params={"method": "celltypist", "model": "m.pkl", "target": "T", "anchor": "B",
                                        "k": 5, "n_perm": 99, "seed": 2, "radius": 30, "transform": "logit"},
                          result={"n_cells": 10, "table": pd.DataFrame({"padj": [0.01, 0.5]})},
                          message="Done.\nUnit of the p-values: the unit the job states.", seg_source="xenium")
    md = _check(xe, rec)
    assert rec["title"] == xe._PROV_KINDS[kind] and rec["unit"]["statement"] == "the unit the job states."
    assert "Unit of the p-values, as stated by the app: the unit the job states." in md


def test_parametrised_kinds_are_all_kinds(xe):
    mark = [m for m in test_every_kind_gives_a_record_and_a_methods_paragraph.pytestmark
            if m.name == "parametrize"][0]
    assert set(mark.args[1]) == set(xe._PROV_KINDS)
