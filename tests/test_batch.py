"""Headless batch mode (xenium_app/batch.py): config validation, a run, failure handling,
and the same tables as the app's own Run buttons give for the same parameters and seed.

A step is run by calling the callback of the step's Run button, so "the UI path" in these
tests is that callback called the way Dash calls it, and "batch" is the config-driven run
- in this process (``_batch_run``) and as ``python -m xenium_app.batch`` in a new one."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pandas as pd
import pytest

import xe_testlib as tl


@pytest.fixture(autouse=True)
def _clean(xe):
    before = list(xe._prov_recent)
    yield
    xe._prov_recent[:] = before
    xe._stats_reset()
    xe._controls_reset()
    with xe._txf_lock:
        xe._txf_state.update({"status": "idle", "message": "", "result": None})


@pytest.fixture()
def cfg(xe, bundle_dir, tmp_path):
    return {"data": {"path": str(bundle_dir)}, "output_dir": str(tmp_path / "out"),
            "steps": [{"kind": "moran", "params": {"k": 6}}]}


def _quiet(_line):
    pass


# ── config validation ────────────────────────────────────────────────────────
@pytest.mark.parametrize("change, message", [
    ({"steps": []}, "steps is missing or empty"),
    ({"steps": [{"kind": "nonsense"}]}, "unknown kind 'nonsense'"),
    ({"steps": [{"kind": "moran", "parameters": {}}]}, "unknown key(s) parameters"),
    ({"data": {}}, "data.path is missing"),
    ({"outdir": "x"}, "unknown top-level key(s): outdir"),
    ({"counts": "raw"}, "'raw' is not 'original' or 'corrected'"),
    ({"data": {"path": "/somewhere/else"}}, "this process has loaded"),
])
def test_a_config_that_cannot_be_used_is_refused_with_a_message(xe, cfg, change, message):
    with pytest.raises(xe._BatchError) as err:
        xe._batch_run({**cfg, **change}, dry_run=True, log=_quiet)
    assert message in str(err.value)


def test_dry_run_checks_parameters_labels_cell_types_and_genes_without_computing(xe, cfg):
    genes = list(xe.DATA["gene_names"])
    cfg["steps"] = [
        {"kind": "moran", "params": {"k": 6, "neighbours": 3}},                     # unknown parameter
        {"kind": "density", "params": {"gene": "NOT_A_GENE"}},                      # not on the panel
        {"kind": "coloc", "params": {"labels": "celltype:nothing"}},                # no such label source
        {"kind": "segfree", "params": {"labels": "cluster", "cell_type": "no such cluster"}},
        {"kind": "lr", "params": {"labels": "cluster", "pairs_text": f"{genes[0]},{genes[1]}\nXX,YY"}},
        {"kind": "pseudobulk", "params": {"population": "all"}},                    # one sample: no design
        {"kind": "joint"},                                                          # one sample
        {"kind": "aggregates", "params": {"labels": "cluster"}},                    # classes missing
        {"kind": "de", "params": {"group_a": "cluster:1", "group_b": "rest"}},      # fine
        {"kind": "geneset", "params": {"name": "s", "text": f"{genes[0]} NOPE1"}},  # one gene missing: warning
    ]
    lines = []
    out = xe._batch_run(cfg, dry_run=True, log=lines.append)
    err = "\n".join(out["errors"])
    assert out["ok"] is False and out["dry_run"] is True and out["steps"] == []
    assert "step 1 (moran): unknown parameter(s) neighbours; this step takes: counts, k, seg_source" in err
    assert "step 2 (density): gene 'NOT_A_GENE' is not on the panel" in err
    assert "step 3 (coloc): label source 'celltype:nothing' is not available (available: cluster" in err
    assert "step 4 (segfree): no cell is labelled 'no such cluster' in 'cluster'" in err
    assert "step 6 (pseudobulk): needs the sample design" in err
    assert "step 7 (joint): needs at least two loaded samples" in err
    assert "step 8 (aggregates): parameter 'classes' is required" in err
    assert "step 9" not in err and "step 5" not in err and "step 10" not in err
    assert any("step 5 (lr): 1 of 2 pairs are not on the panel" in w for w in out["warnings"])
    assert any("step 10 (geneset): not on the panel: NOPE1" in w for w in out["warnings"])
    assert lines[-1].startswith("Dry run: 10 step(s), 7 error(s), 2 warning(s). Nothing was computed.")
    # nothing ran, nothing was written
    assert xe._moran_state["result"] is None and not os.path.exists(cfg["output_dir"])
    # a valid config passes
    cfg["steps"] = [{"kind": "moran", "params": {"k": 6}}, {"kind": "compartments"}]
    assert xe._batch_run(cfg, dry_run=True, log=_quiet)["ok"] is True


def test_every_step_kind_resolves_its_callback_and_defaults(xe):
    """The step table against the app: each Run / Download callback and figure function
    exists, every control named in the tables is an input of its callback, and a step
    without parameters takes the defaults of the controls in the layout."""
    ctx = xe._batch_context({})
    assert ctx["seg-source"] == "xenium" and ctx["counts-mode-store"] == "original"
    comps = xe._batch_components()
    for kind, spec in xe._BATCH_STEPS.items():
        func = getattr(sys.modules[f"xenium_app.{spec['run'][0]}"], spec["run"][1])
        ids = [cid for _n, cid, _p, _i in xe._batch_io(func)]
        args, resolved, allowed = xe._batch_step_args(spec, {}, ctx, kind)
        assert len(args) == len(ids) and args[0] == 1
        for cid in list(spec.get("fixed") or {}) + xe._BATCH_NEED.get(kind, []) + xe._BATCH_USES.get(kind, []):
            assert cid in ids, f"{kind}: {cid} is not an input of {spec['run'][1]}"
        for cid, val in resolved.items():
            if cid not in xe._BATCH_CONTEXT and cid not in (spec.get("fixed") or {}):
                assert cid in comps, f"{kind}: no control '{cid}' in the layout"
                prop = [p for _n, c, p, _i in xe._batch_io(func) if c == cid][0]
                assert val == getattr(comps[cid], prop, None)
        if spec.get("download"):
            assert callable(getattr(sys.modules[f"xenium_app.{spec['download'][0]}"], spec["download"][1]))
        for _stem, mod, fn in spec.get("figures") or []:
            assert callable(getattr(sys.modules[f"xenium_app.{mod}"], fn))
        mod, sname, lname = spec["state"]
        assert isinstance(getattr(sys.modules[f"xenium_app.{mod}"], sname), dict)
    for cid in xe._BATCH_ROLES:
        assert cid in comps, cid
    assert set(xe._BATCH_ALIASES.values()) <= set(xe._BATCH_STEPS)
    # true / false for a checklist
    assert xe._batch_coerce("de-leak-gate", "value", False) == []
    assert xe._batch_coerce("de-leak-gate", "value", True) == comps["de-leak-gate"].value
    listing = xe._batch_list_steps()
    assert "moran  (cb_stats.run_moran)" in listing and "k " in listing and "REQUIRED" in listing


# ── a run, in this process ───────────────────────────────────────────────────
def test_two_step_run_writes_tables_figures_provenance_and_methods(xe, cfg):
    cfg["steps"] = [{"kind": "moran", "params": {"k": 6}},
                    {"kind": "composition", "name": "clusters", "params": {"labels": "cluster"}}]
    lines = []
    out = xe._batch_run(cfg, log=lines.append)
    root = pathlib.Path(cfg["output_dir"])
    assert out["ok"] and out["n_done"] == 2 and out["n_failed"] == 0 and out["n_not_run"] == 0
    assert [s["status"] for s in out["steps"]] == ["done", "done"]
    assert any(l.startswith("[1/2] moran (moran)") for l in lines)
    assert any("[moran]" in l for l in lines), "no progress line of the job"          # progress to stdout
    # step 1: the Download button's CSV (with the record id), the table, the record
    csv = (root / "01_moran" / "spatially_variable_genes_moran.csv").read_text()
    rec = json.loads((root / "01_moran" / "provenance.json").read_text())
    assert len(rec) == 1 and rec[0]["kind"] == "moran" and rec[0]["params"] == {"k": 6}
    assert f"# provenance record: {rec[0]['id']}" in csv
    table = pd.read_parquet(root / "01_moran" / "table.parquet")
    pd.testing.assert_frame_equal(table, xe._moran_state["result"]["table"].reset_index(drop=True))
    assert (root / "01_moran" / "status.txt").read_text().startswith("Moran's I, kNN graph (k = 6)")
    # step 2: figure as HTML + Plotly JSON (PNG as well when kaleido is installed)
    two = root / "02_clusters"
    assert (two / "composition.plotly.json").exists() and (two / "props.parquet").exists()
    assert (two / "composition.png").exists() or (two / "composition.html").exists()
    assert json.loads((two / "composition.plotly.json").read_text())["data"]
    # the run's record and methods text
    doc = json.loads((root / "provenance.json").read_text())
    assert [r["kind"] for r in doc["records"]] == ["moran", "composition"]
    md = (root / "methods.md").read_text()
    assert "## Spatially variable genes (Moran's I)" in md and "## Composition between conditions" in md
    summ = json.loads((root / "batch_summary.json").read_text())
    assert summ["steps"][0]["provenance_ids"] == [rec[0]["id"]] and summ["steps"][0]["resolved"]["moran-k"] == 6
    assert summ["kaleido"] in (True, False) and ("figures_note" in summ) == (not summ["kaleido"])


# ── failure handling ─────────────────────────────────────────────────────────
def test_a_failed_step_stops_the_run_unless_continue_on_error(xe, cfg, tmp_path, capsys):
    steps = [{"kind": "density", "params": {"gene": "NOT_A_GENE"}}, {"kind": "moran", "params": {"k": 6}}]
    lines = []
    out = xe._batch_run({**cfg, "steps": steps}, log=lines.append)
    assert out["ok"] is False and out["n_failed"] == 1 and out["n_done"] == 0 and out["n_not_run"] == 1
    assert out["steps"][0]["status"] == "failed" and "NOT_A_GENE" in out["steps"][0]["message"]
    assert any(l.strip().startswith("FAILED: step 1 (density)") and "not a measured gene" in l for l in lines)
    assert any(l.startswith("Stopped at step 1 of 2") for l in lines)
    assert xe._moran_state["result"] is None                                   # step 2 did not run
    summ = json.loads((pathlib.Path(cfg["output_dir"]) / "batch_summary.json").read_text())
    assert summ["ok"] is False and summ["n_not_run"] == 1
    # continue_on_error: the next step runs; the failure is still reported
    out = xe._batch_run({**cfg, "steps": steps, "continue_on_error": True,
                         "output_dir": str(tmp_path / "out2")}, log=_quiet)
    assert [s["status"] for s in out["steps"]] == ["failed", "done"] and out["ok"] is False
    assert xe._moran_state["result"] is not None
    # the command's exit code: 1 for a failed step, 2 for an unusable config, 0 otherwise
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({**cfg, "steps": steps, "output_dir": str(tmp_path / "out3")}))
    assert xe._batch_main([str(path)]) == 1
    assert "FAILED: step 1 (density)" in capsys.readouterr().out
    path.write_text(json.dumps({**cfg, "steps": [{"kind": "moran"}], "output_dir": str(tmp_path / "out4")}))
    assert xe._batch_main([str(path)]) == 0
    assert xe._batch_main([str(path), "--dry-run"]) == 0
    path.write_text("{not json")
    assert xe._batch_main([str(path)]) == 2
    assert "Batch error:" in capsys.readouterr().out and "not valid JSON" not in ""
    path.write_text(json.dumps({**cfg, "steps": [{"kind": "moran", "params": {"kk": 1}}]}))
    assert xe._batch_main([str(path), "--dry-run"]) == 1


# ── the same numbers as the app ──────────────────────────────────────────────
def _wait(xe, state, lock):
    tl.wait_state(state, lock, timeout=120)
    for j in xe._jobs_list():                 # the job's thread has ended (record written)
        if j.thread is not None:
            j.thread.join(timeout=60)


def test_command_line_run_gives_the_tables_of_the_run_buttons(xe, bundle_dir, tmp_path):
    """Moran's I and the seeded transcript factors: once through the Run callbacks in this
    process, once through ``python -m xenium_app.batch`` in a new process on its own copy
    of the bundle - the result tables are identical."""
    # the UI path
    xe.run_moran(1, 5, "xenium", "original")
    _wait(xe, xe._moran_state, xe._moran_lock)
    ui_moran = xe._moran_state["result"]["table"].reset_index(drop=True)
    comps = xe._batch_components()
    xe.run_txfactor(1, "nmf", 3, 10, comps["txf-shape"].value, comps["txf-mintx"].value,
                    comps["txf-maxfit"].value, 7, None, "xenium", xe.DATA["cluster_methods"][0], "original")
    _wait(xe, xe._txf_state, xe._txf_lock)
    assert xe._txf_state["status"] == "done", xe._txf_state["message"]
    ui_factors = xe._txf_state["result"]["factors"].reset_index(drop=True)
    ui_csv = xe.download_txfactor(1)["content"]
    # batch, as a command, on a copy
    data = tl.make_working_copy(tl.TINY_BUNDLE, tmp_path / "bundle")
    cfg = {"data": {"path": str(data)}, "cache_dir": str(tmp_path / "cache"), "output_dir": str(tmp_path / "out"),
           "steps": [{"kind": "moran", "params": {"k": 5}},
                     {"kind": "transcript_factors", "name": "factors",
                      "params": {"method": "nmf", "k": 3, "bin_um": 10, "seed": 7}}]}
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    env = {k: v for k, v in os.environ.items() if k not in ("DATA_DIR", "XENIUM_CACHE_DIR", "PORT",
                                                            "XENIUM_ENABLE_REPL", "XENIUM_RENDERER")}
    env["PYTHONWARNINGS"] = "error::RuntimeWarning:runpy"          # the module is not imported twice
    proc = subprocess.run([sys.executable, "-m", "xenium_app.batch", str(path)], cwd=str(tl.REPO), env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "Batch finished: 2 step(s) done, 0 failed" in proc.stdout
    assert "running at http" not in proc.stdout and "Dash is running" not in proc.stdout + proc.stderr
    out = tmp_path / "out"
    b_moran = pd.read_parquet(out / "01_moran" / "table.parquet")
    b_factors = pd.read_parquet(out / "02_factors" / "factors.parquet")
    pd.testing.assert_frame_equal(b_moran, ui_moran, check_exact=True)
    pd.testing.assert_frame_equal(b_factors, ui_factors, check_exact=True)
    # ... and the downloaded CSV is the same text, apart from its timing and record id lines
    strip = lambda t: [l for l in t.splitlines() if not l.startswith("# provenance record") and " s" not in l[:0]]
    b_csv = (out / "02_factors" / "transcript_factors.csv").read_text()
    same = [(a, b) for a, b in zip(strip(ui_csv), strip(b_csv)) if a != b]
    assert all(a.startswith("#") for a, _b in same) and len(same) <= 2, same[:3]
    # the record states the seed and that it was computed
    rec = json.loads((out / "02_factors" / "provenance.json").read_text())[0]
    assert rec["kind"] == "txfactor" and rec["cache"] == "computed" and rec["params"]["method"] == "nmf"
    assert "7" in json.dumps(rec["params"]) and (out / "methods.md").exists()
    # the cache of the run holds the records too (the Provenance tab of a later session shows them)
    assert list((tmp_path / "cache" / "provenance").glob("*.jsonl"))


def test_example_config_is_valid_against_the_fixture(xe, bundle_dir, tmp_path):
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((tl.REPO / "examples" / "batch_tiny.yaml").read_text())
    assert cfg["data"]["path"] == "/tmp/xenium_tiny"
    cfg["data"]["path"] = str(bundle_dir)
    cfg["output_dir"] = str(tmp_path / "out")
    cfg.pop("cache_dir")
    out = xe._batch_run(cfg, dry_run=True, log=_quiet)
    assert out["ok"] is True and out["errors"] == []
