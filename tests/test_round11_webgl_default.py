"""Round 11: WebGL default spatial renderer and Plotly fallback coverage.

Covers:
1. Default renderer (`_DEFAULT_RENDERER == "webgl"`) when `XENIUM_RENDERER` is unset or
   invalid, with no false oversized-Plotly cell-limit warning at startup.
2. Fresh-process `XENIUM_RENDERER` overrides (`plotly`, case/whitespace normalisation,
   oversized dataset under `XENIUM_PLOTLY_MAX_CELLS`, and invalid values).
3. In-app `renderer-mode` callback switching (`update_plots`, `toggle_morph_controls`,
   `draw_measurement`).
4. Session save / restore / autosave (`_session_default("renderer-mode")`, untouched-page
   autosave baseline, `"plotly"` <-> `"webgl"` same-dataset restore, and cross-dataset skip).
5. Headless `batch.py` spatial figure export while `_DEFAULT_RENDERER == "webgl"`, both
   in-process (`_batch_run`) and via `python -m xenium_app.batch`.
6. `README.md` documentation of the WebGL default, `XENIUM_RENDERER=plotly`, and untested
   Firefox / Safari browser note.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
import types

import pytest

import xe_testlib as tl


def _update_plots_args(xe, renderer_mode: str | None, gene: str | None = None) -> list:
    D = xe.DATA
    method = D["cluster_methods"][0] if D["cluster_methods"] else None
    chosen_gene = gene or D["gene_names"][0]
    return [
        "cluster" if method else "transcript_counts",  # color_by
        method,                                        # method
        chosen_gene,                                   # gene
        3,                                             # size
        0.8,                                           # opacity
        [],                                            # boundary_toggles
        0,                                             # _annot_done
        [],                                            # morph_enable
        3,                                             # morph_zlevel
        [],                                            # morph_channels
        1.0,                                           # morph_brightness
        0.7,                                           # morph_opacity
        None,                                          # relayout
        0,                                             # _baysor_done
        0,                                             # _proseg_done
        "xenium",                                      # seg_source
        0,                                             # _spage_done
        0,                                             # _subset_ver
        1,                                             # _startup
        0,                                             # _ds_version
        0,                                             # _extra_ds_version
        0,                                             # _split_done
        0,                                             # _roi_done
        "original",                                    # counts_mode
        0,                                             # _niche_done
        None,                                          # cbar_min
        None,                                          # cbar_max
        renderer_mode,                                 # renderer_mode
        None,                                          # density_spec
        0,                                             # _geneset_ver
        None,                                          # cn_lr_pair
        50,                                            # cn_top_n
        2,                                             # cn_edge_width
        [],                                            # cn_overlay_enable
        [],                                            # cn_options
        "pair",                                        # cn_color_by
        None,                                          # cn_f_ligand
        None,                                          # cn_f_receptor
        None,                                          # cn_f_sender
        None,                                          # cn_f_receiver
        None,                                          # cn_annot_col
        [],                                            # sdata_show_roi
        [],                                            # sdata_show_patches
        None,                                          # measure_state
    ]


# ── 1. In-process default (XENIUM_RENDERER unset) ────────────────────────────
def test_default_renderer_is_webgl_and_prints_no_cell_limit_warning(xe):
    assert xe._RENDERER_ENV == ""
    assert xe._DEFAULT_RENDERER == "webgl"

    comp = xe._session_components()["renderer-mode"]
    assert comp.value == "webgl"
    assert list(comp.options) == [
        {"label": " Plotly", "value": "plotly"},
        {"label": " WebGL", "value": "webgl"},
    ]

    startup_log = "\n".join(xe._log_buffer)
    assert "exceed the Plotly limit" not in startup_log
    assert "too many for the Plotly renderer" not in startup_log
    assert "switching to WebGL" not in startup_log.lower()


# ── 2. Fresh-process XENIUM_RENDERER startup matrix ──────────────────────────
_FRESH_RENDERER_PROBE = r"""
import types
from xenium_app import cb_plots

cb_plots.callback_context = types.SimpleNamespace(
    triggered_id="renderer-mode",
    triggered=[{"prop_id": "renderer-mode.value"}],
)
comp = xe._session_components()["renderer-mode"]
method = xe.DATA["cluster_methods"][0] if xe.DATA["cluster_methods"] else None
gene = xe.DATA["gene_names"][0]
color_by = "cluster" if method else "transcript_counts"

def _call(rmode):
    return xe.update_plots(
        color_by, method, gene, 3, 0.8, [], 0, [], 3, [], 1.0, 0.7, None,
        0, 0, "xenium", 0, 0, 1, 0, 0, 0, 0, "original", 0, None, None,
        rmode, None, 0, None, 50, 2, [], [], "pair", None, None, None, None, None,
        [], [], None,
    )

sp_w, um_w = _call("webgl")
sp_p, um_p = _call("plotly")
defaults = {cid: xe._session_default(cid) for cid in xe._session_controls()}
autosave_first = xe._session_autosave(defaults)
client = xe.app.server.test_client()

def _fig_info(fig):
    if fig is xe.no_update:
        return {"no_update": True}
    d = fig.to_plotly_json()
    ann = " ".join(str(a.get("text", "")) for a in (d.get("layout", {}).get("annotations") or []))
    n_pts = sum(len(t.get("x", [])) for t in d.get("data", []) if t.get("x") is not None)
    return {"no_update": False, "n_pts": int(n_pts), "annotations": ann}

RESULT.update({
    "renderer_env": xe._RENDERER_ENV,
    "default_renderer": xe._DEFAULT_RENDERER,
    "layout_value": comp.value,
    "session_default": xe._session_default("renderer-mode"),
    "autosave_untouched": autosave_first,
    "last_exists": os.path.exists(xe._session_path("last")),
    "too_large": xe._plotly_too_large(),
    "log_text": "\n".join(xe._log_buffer),
    "n_cells": int(len(xe.DATA["df"])),
    "viewer_meta_status": int(client.get("/viewer/meta").status_code),
    "webgl_spatial": _fig_info(sp_w),
    "webgl_umap": _fig_info(um_w),
    "plotly_spatial": _fig_info(sp_p),
    "plotly_umap": _fig_info(um_p),
})
"""


@pytest.mark.parametrize(
    "name, env, expected_renderer, expect_limit_notice",
    [
        ("unset", {}, "webgl", False),
        ("plotly_normal", {"XENIUM_RENDERER": "  PLOTLY "}, "plotly", False),
        ("plotly_oversized", {"XENIUM_RENDERER": "plotly", "XENIUM_PLOTLY_MAX_CELLS": "5"}, "plotly", True),
        ("invalid_oversized", {"XENIUM_RENDERER": "bogus", "XENIUM_PLOTLY_MAX_CELLS": "5"}, "webgl", False),
    ],
)
def test_fresh_app_renderer_env_overrides(bundle_dir, tmp_path, name, env, expected_renderer, expect_limit_notice):
    cache_dir = tmp_path / f"cache_{name}"
    cache_dir.mkdir()
    r = tl.run_in_fresh_app(_FRESH_RENDERER_PROBE, bundle_dir, cache_dir, env=env)

    assert r["default_renderer"] == expected_renderer
    assert r["layout_value"] == expected_renderer
    assert r["session_default"] == expected_renderer
    assert r["autosave_untouched"] == ""
    assert r["last_exists"] is False
    assert r["viewer_meta_status"] == 200

    # WebGL mode never builds Plotly figures; switching to Plotly builds figures or placeholders.
    assert r["webgl_spatial"] == {"no_update": True}
    assert r["webgl_umap"] == {"no_update": True}

    has_notice = "exceed the Plotly limit" in r["log_text"]
    assert has_notice is expect_limit_notice, r["log_text"]

    if expect_limit_notice:
        assert "placeholder until switched to WebGL" in r["log_text"]
        assert r["too_large"] and "limit 5" in r["too_large"]
        assert r["plotly_spatial"]["n_pts"] == 0
        assert "too many for the Plotly renderer (limit 5)" in r["plotly_spatial"]["annotations"]
        assert "WebGL view" in r["plotly_umap"]["annotations"]
    elif "XENIUM_PLOTLY_MAX_CELLS" not in env:
        assert r["too_large"] is None
        assert r["plotly_spatial"]["n_pts"] >= r["n_cells"]
        assert r["plotly_umap"]["n_pts"] > 0


# ── 3. In-app Plotly <-> WebGL callback switching ────────────────────────────
def test_in_app_plotly_and_webgl_callbacks(xe, D, expressed_gene, monkeypatch):
    from xenium_app import cb_plots

    monkeypatch.setattr(
        cb_plots,
        "callback_context",
        types.SimpleNamespace(
            triggered_id="renderer-mode",
            triggered=[{"prop_id": "renderer-mode.value"}],
        ),
    )

    # WebGL view skips building both Plotly figures
    sp_w, um_w = xe.update_plots(*_update_plots_args(xe, "webgl", expressed_gene))
    assert sp_w is xe.no_update and um_w is xe.no_update

    # Switching to Plotly builds full spatial and UMAP figures matching the dataset cells
    sp_p, um_p = xe.update_plots(*_update_plots_args(xe, "plotly", expressed_gene))
    assert sp_p is not xe.no_update and um_p is not xe.no_update
    drawn = tl.fig_cells(sp_p)
    assert set(drawn.index) == set(D["df"].index)
    if D.get("umap_df") is not None and not D["umap_df"].empty:
        umap_drawn = tl.fig_cells(um_p)
        assert set(umap_drawn.index) == set(D["umap_df"].index)

    # None renderer_mode also builds Plotly figures cleanly
    sp_n, um_n = xe.update_plots(*_update_plots_args(xe, None, expressed_gene))
    assert sp_n is not xe.no_update and um_n is not xe.no_update

    # Morphology hi-res poll is disabled in WebGL and enabled in Plotly when morphology is shown
    assert xe.toggle_morph_controls(["show"], "webgl") == ({}, True)
    assert xe.toggle_morph_controls(["show"], "plotly") == ({}, False)
    assert xe.toggle_morph_controls([], "plotly") == ({"display": "none"}, True)

    # Distance ruler: WebGL returns no_update for the figure + computed distance label (no infinite recursion)
    ruler = {"active": True, "points": [[0.0, 0.0], [30.0, -40.0]]}
    base_fig = {"data": [{"name": "cells", "x": [1.0], "y": [2.0]}]}
    fig_w, label_w, style_w = xe.draw_measurement(ruler, dict(data=list(base_fig["data"])), "webgl")
    assert fig_w is xe.no_update
    assert label_w == "50.0 µm"
    assert style_w["display"] == "inline"

    fig_p, label_p, style_p = xe.draw_measurement(ruler, {"data": list(base_fig["data"])}, "plotly")
    assert fig_p is not xe.no_update
    assert label_p == "50.0 µm"
    assert style_p["display"] == "inline"
    assert any(str(t.get("name", "")).startswith("measure_") for t in fig_p["data"])


# ── 4. Session defaults, autosave baseline, and Plotly <-> WebGL restore ─────
def test_session_defaults_autosave_and_renderer_restore(xe, D, tmp_path, monkeypatch):
    monkeypatch.setattr(xe, "CACHE_BASE_DIR", str(tmp_path))
    before_prov = list(xe._prov_recent)
    xe._session_auto.update({"baseline": None, "written": None})
    try:
        assert xe._session_default("renderer-mode") == "webgl"
        assert "renderer-mode" in xe._SESSION_BOUND_VIEW
        defaults = {cid: xe._session_default(cid) for cid in xe._session_controls()}
        assert defaults["renderer-mode"] == "webgl"

        # Untouched WebGL default page matches server default_sig: no autosave written
        assert xe._session_autosave(defaults) == ""
        assert not os.path.exists(xe._session_path("last"))

        # User switches to Plotly: autosave records "plotly"
        plotly_page = {**defaults, "renderer-mode": "plotly"}
        assert xe._session_autosave(plotly_page) == "saved"
        saved_auto = json.loads(pathlib.Path(xe._session_path("last")).read_text())
        assert saved_auto["state"]["controls"]["renderer-mode"] == "plotly"

        # Explicit save in Plotly mode -> restore onto a default WebGL page restores "plotly"
        # and updates the autosave baseline so subsequent ticks do not re-save
        out_save = xe.session_action({"action": "save", "controls": plotly_page}, 0, 0, 0, 0, 0)
        assert out_save[0].startswith("Session saved")
        out_load = xe.session_action({"action": "load_saved", "controls": defaults}, 0, 0, 0, 0, 0)
        assert out_load[0].startswith("Session restored")
        assert out_load[2] == {"renderer-mode": "plotly"}
        assert xe._session_autosave(plotly_page) == ""

        # Explicit save in WebGL mode -> restore onto a page currently on "plotly" restores "webgl"
        webgl_doc = xe._session_snapshot(defaults, "download")
        rep_to_webgl = xe._session_restore_text(json.dumps(webgl_doc), plotly_page, "test")
        assert rep_to_webgl["ok"] and rep_to_webgl["mode"] == "full"
        assert rep_to_webgl["controls"] == {"renderer-mode": "webgl"}

        # Cross-dataset restore skips "renderer-mode" because it is bound to the dataset view
        foreign_doc = xe._session_snapshot({**plotly_page, "point-size": 7}, "download")
        foreign_doc["dataset"]["key"] = foreign_doc["dataset"]["samples"][0]["key"] = "/other/dataset"
        rep_foreign = xe._session_restore_text(json.dumps(foreign_doc), defaults, "test")
        assert rep_foreign["ok"] and rep_foreign["mode"] == "independent"
        assert rep_foreign["controls"] == {"point-size": 7}
        assert "renderer-mode" not in rep_foreign["controls"]
    finally:
        xe._prov_recent[:] = before_prov
        xe._session_auto.update({"baseline": None, "written": None})


# ── 5. Headless batch figure export under the WebGL default ──────────────────
def test_headless_batch_exports_spatial_figures_under_webgl_default(xe, D, bundle_dir, labels, tmp_path, monkeypatch):
    assert xe._DEFAULT_RENDERER == "webgl"
    cache_dir = tmp_path / "batch_cache"
    cache_dir.mkdir()
    monkeypatch.setattr(xe, "CACHE_BASE_DIR", str(cache_dir))
    before_prov = list(xe._prov_recent)

    method = D["cluster_methods"][0]
    first_cluster = str(D["df"][f"cluster_{method}"].dropna().iloc[0])
    out_dir = tmp_path / "batch_out"
    cfg = {
        "data": {"path": str(bundle_dir)},
        "output_dir": str(out_dir),
        "steps": [
            {"kind": "spatial", "name": "spatial_figs",
             "params": {"label_key": "labels_celltypist", "n_neighbors": 5}},
            {"kind": "aggregates", "name": "agg_figs",
             "params": {"labels": "cluster", "classes": first_cluster}},
            {"kind": "composition", "name": "comp_figs",
             "params": {"labels": "cluster"}},
        ],
    }
    try:
        res = xe._batch_run(cfg, log=lambda _line: None)
        assert res["ok"] is True and res["n_done"] == 3 and res["n_failed"] == 0

        expected_figs = [
            (out_dir / "01_spatial_figs", "cooccurrence"),
            (out_dir / "01_spatial_figs", "neighbourhood_enrichment"),
            (out_dir / "01_spatial_figs", "interaction_graph"),
            (out_dir / "02_agg_figs", "aggregates"),
            (out_dir / "03_comp_figs", "composition"),
        ]
        for step_dir, stem in expected_figs:
            pj = step_dir / f"{stem}.plotly.json"
            assert pj.is_file(), f"missing {pj}"
            fig_doc = json.loads(pj.read_text())
            assert fig_doc.get("data"), f"empty figure data in {pj}"
            assert (step_dir / f"{stem}.html").is_file() or (step_dir / f"{stem}.png").is_file()
    finally:
        xe._prov_recent[:] = before_prov
        xe._stats_reset()
        xe._controls_reset()
        with xe._agg_lock:
            xe._agg_state.update({"status": "idle", "message": "", "result": None})

    # Subprocess check: `python -m xenium_app.batch` strips XENIUM_RENDERER in __init__.py
    # and still exports valid figures cleanly without starting a server.
    sub_bundle = tl.make_working_copy(tl.TINY_BUNDLE, tmp_path / "sub_bundle")
    sub_out = tmp_path / "sub_out"
    sub_cfg = {
        "data": {"path": str(sub_bundle)},
        "cache_dir": str(tmp_path / "sub_cache"),
        "output_dir": str(sub_out),
        "steps": [{"kind": "composition", "name": "clusters", "params": {"labels": "cluster"}}],
    }
    cfg_path = tmp_path / "batch_cfg.json"
    cfg_path.write_text(json.dumps(sub_cfg))
    env = {k: v for k, v in os.environ.items()
           if k not in ("DATA_DIR", "XENIUM_CACHE_DIR", "PORT", "XENIUM_ENABLE_REPL")}
    env["XENIUM_RENDERER"] = "plotly"
    proc = subprocess.run(
        [sys.executable, "-m", "xenium_app.batch", str(cfg_path)],
        cwd=str(tl.REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "Batch finished: 1 step(s) done, 0 failed" in proc.stdout
    sub_fig = sub_out / "01_clusters" / "composition.plotly.json"
    assert sub_fig.is_file() and json.loads(sub_fig.read_text()).get("data")
    assert (sub_out / "01_clusters" / "composition.html").is_file() or (
        sub_out / "01_clusters" / "composition.png"
    ).is_file()


# ── 6. Dash clientside glue: WebGL start, Plotly <-> WebGL switch, fallback ──
def test_dash_glue_webgl_default_plotly_switch_and_fallback_in_node():
    node = shutil.which("node") or "/opt/homebrew/bin/node"
    if not os.path.exists(node):
        pytest.skip("node binary not available")

    glue_js = str(tl.REPO / "assets" / "xenium_viewer_dash.js")
    driver = textwrap.dedent("""
        const fs = require("fs");
        const vm = require("vm");

        function runCase(hasWebgl2) {
          function stub() {
            return new Proxy(function () {}, {
              get(t, k) {
                if (k === Symbol.toPrimitive) return () => "";
                if (k === "then") return undefined;
                if (!(k in t)) t[k] = stub();
                return t[k];
              },
              set(t, k, v) { t[k] = v; return true; },
              apply() { return stub(); },
              construct() { return stub(); },
            });
          }
          const props = [];
          const viewers = [];
          const viewports = [];
          let fitCount = 0;

          function viewerRecord(el, opts) {
            const rec = { baseUrl: opts.baseUrl || "", states: [] };
            viewers.push(rec);
            const api = stub();
            api.setState = (s) => { rec.states.push(JSON.parse(JSON.stringify(s))); };
            api.setViewport = (vp) => { viewports.push(vp); };
            api.fitView = () => { fitCount += 1; };
            api.resize = () => {};
            api.destroy = () => {};
            return api;
          }

          const document = stub();
          document.getElementById = (id) => {
            if (id === "_dash-config") return { textContent: "{}" };
            const el = stub();
            el.id = id;
            return el;
          };
          document.createElement = (tag) => {
            if (tag === "canvas") {
              return {
                getContext: (kind) => {
                  if (kind === "webgl2") return hasWebgl2 ? { getExtension: () => null } : null;
                  if (kind === "webgl") return { getExtension: () => null };
                  return null;
                },
              };
            }
            return stub();
          };
          const window = stub();
          window.document = document;
          window.XeniumViewer = { mount: viewerRecord };
          window.dash_clientside = { no_update: "NO_UPDATE", set_props: (id, p) => props.push([id, p]) };
          const sandbox = {
            window, document, console: { warn: () => {}, log: () => {}, error: () => {} },
            JSON, Math, Date, Object, Array, Number, String, Boolean, RegExp,
            encodeURIComponent, isFinite, parseFloat, parseInt, Promise, Event: function () {},
            setTimeout: (fn) => { try { fn(); } catch (e) {} return 0; },
            clearTimeout: () => {}, setInterval: () => 0, clearInterval: () => {},
            requestAnimationFrame: () => 0, navigator: { userAgent: "node" }, URL: stub(),
            fetch: () => Promise.resolve({ json: () => Promise.resolve({}) }),
          };
          vm.createContext(sandbox);
          vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
          const sync = (mode, relayout) => window.dash_clientside.xenium_viewer.sync(
            mode, "cluster", "graphclust", "", 2, 0.8, [], "xenium", "original", null, null,
            [], [], [], 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, "pan", {}, false, 1, relayout || {}
          );
          const s1 = sync("webgl", {});
          const s2 = sync("plotly", {});
          const s3 = sync("webgl", {
            "xaxis.range[0]": 10, "xaxis.range[1]": 90,
            "yaxis.range[0]": -80, "yaxis.range[1]": -20,
          });
          const s4 = sync("plotly", {});
          const s5 = sync("webgl", {});
          return { s1, s2, s3, viewers: viewers.length, viewports, fitCount, props };
        }

        console.log(JSON.stringify({ ok: runCase(true), noGl2: runCase(false) }));
    """)
    proc = subprocess.run([node, "-e", driver, glue_js], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)

    # Normal WebGL2 browser: starts in WebGL, switches cleanly to Plotly and back to WebGL
    ok = out["ok"]
    assert ok["s1"][0]["display"] == "none" and ok["s1"][1]["display"] == "block" and ok["s1"][3] is True
    assert ok["s2"][0]["display"] == "block" and ok["s2"][1]["display"] == "none" and ok["s2"][3] is False
    assert ok["s3"][0]["display"] == "none" and ok["s3"][1]["display"] == "block" and ok["s3"][3] is True
    assert ok["viewports"] == [{"x0": 10, "x1": 90, "y0": 20, "y1": 80}]
    assert ok["fitCount"] == 1

    # Browser without WebGL2: starting in default WebGL falls back automatically to Plotly
    fb = out["noGl2"]
    assert fb["s1"][0]["display"] == "block" and fb["s1"][1]["display"] == "none" and fb["s1"][3] is False
    rmode_updates = [p[1] for p in fb["props"] if p[0] == "renderer-mode"]
    assert rmode_updates and rmode_updates[0]["value"] == "plotly"
    assert rmode_updates[0]["options"][1]["disabled"] is True


# ── 7. Documentation check ───────────────────────────────────────────────────
def test_readme_documents_webgl_default_plotly_fallback_and_untested_browsers():
    readme = (tl.REPO / "README.md").read_text(encoding="utf-8")
    assert "XENIUM_RENDERER=plotly" in readme
    assert "Plotly / WebGL" in readme
    assert "Firefox" in readme and "Safari" in readme
    assert "untested" in readme
