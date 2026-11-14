"""Round-11 regression and adversarial stress tests for Slice 3:
viewer API / bounds NaN hygiene (U2), WebGL figure export & array-colour overlays (U3),
batch --dry-run / --list-steps zero-write preconfigure & validator parity (U1, U4),
duplicate-ID / NaN cell inspector (U5), CellNEST clust__ join & sender_type colouring (U6),
multi-sample WebGL/Plotly colourmap parity + SPLIT mask + safe log1p (U7, S5, D6b),
provenance methods variant deduplication (U8), and session restore validation (U9).
"""
from __future__ import annotations

import copy
import json
import os
import pathlib
import shutil
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _r11_isolate_state(xe, tmp_path, monkeypatch):
    """Isolate mutable caches, extra samples, annotations, and segmentation states per test."""
    monkeypatch.setattr(xe, "CACHE_BASE_DIR", str(tmp_path / "cache"))
    os.makedirs(xe.CACHE_BASE_DIR, exist_ok=True)
    orig_extras = list(xe.EXTRA_DATASETS)
    orig_annot = copy.copy(xe._annot_state)
    orig_spage = copy.copy(xe._spage_state)
    orig_baysor = copy.copy(xe._baysor_state)
    orig_proseg = copy.copy(xe._proseg_state)
    orig_cellnest = copy.copy(xe._cellnest_state)
    orig_prov = list(xe._prov_recent)
    orig_cm = list(xe.DATA.get("cluster_methods") or [])
    orig_split_expr = xe.DATA.get("split_corrected_expr", "__ABSENT__")
    orig_split_ids = xe.DATA.get("split_cell_ids", "__ABSENT__")
    orig_split_imp = xe.DATA.get("split_corrected_imputed_genes", "__ABSENT__")
    yield
    xe.EXTRA_DATASETS[:] = orig_extras
    with xe._annot_lock:
        xe._annot_state.clear()
        xe._annot_state.update(orig_annot)
    with xe._spage_lock:
        xe._spage_state.clear()
        xe._spage_state.update(orig_spage)
    with xe._baysor_lock:
        xe._baysor_state.clear()
        xe._baysor_state.update(orig_baysor)
    with xe._proseg_lock:
        xe._proseg_state.clear()
        xe._proseg_state.update(orig_proseg)
    with xe._cellnest_lock:
        xe._cellnest_state.clear()
        xe._cellnest_state.update(orig_cellnest)
    xe._prov_recent[:] = orig_prov
    xe.DATA["cluster_methods"] = orig_cm
    for k, v in (
        ("split_corrected_expr", orig_split_expr),
        ("split_cell_ids", orig_split_ids),
        ("split_corrected_imputed_genes", orig_split_imp),
    ):
        if v == "__ABSENT__":
            xe.DATA.pop(k, None)
        else:
            xe.DATA[k] = v
    xe._gene_expr_cache.clear()
    xe._umap_df_cache.clear()


def _write_cached_baysor_run(xe, D, tag: str = "a11ba001", n_cells: int = 16) -> str:
    import hashlib
    out_dir = xe._seg_run_dir("baysor", tag)
    os.makedirs(out_dir, exist_ok=True)
    dhash = hashlib.md5(xe._dataset_key().encode()).hexdigest()[:8]
    with open(os.path.join(out_dir, "params.json"), "w", encoding="utf-8") as fh:
        json.dump({"dataset_hash": dhash, "min_molecules_per_cell": 5}, fh)
    with open(os.path.join(out_dir, "run_id"), "w", encoding="utf-8") as fh:
        fh.write(f"run_{tag}\n")
    rng = np.random.default_rng(11)
    genes = list(D["gene_names"][:8])
    rows = []
    for ci in range(n_cells):
        cx = float(40.0 + (ci % 4) * 25.0)
        cy = float(40.0 + (ci // 4) * 25.0)
        for ti in range(6):
            rows.append({
                "x": cx + float(rng.normal(0, 1.0)),
                "y": cy + float(rng.normal(0, 1.0)),
                "gene": genes[(ci + ti) % len(genes)],
                "cell": f"bcell_{ci}",
            })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "segmentation.csv"), index=False)
    return out_dir


# ── U1: Batch validator parity & per-segmentation/per-method producer scoping ──
def test_r11_u1_batch_validate_spatial_segqc_txfactor_and_scoped_producers(xe, D, bundle_dir, tmp_path):
    base = {"data": {"path": str(bundle_dir)}, "output_dir": str(tmp_path / "out")}
    genes = list(D["gene_names"][:2])
    _write_cached_baysor_run(xe, D, tag="a11ba001")

    with xe._annot_lock:
        for k in list(xe._annot_state):
            if k == "labels" or k.startswith("labels_"):
                xe._annot_state[k] = None

    # 1. spatial requires label_method and validates it against loaded or matching earlier annotation
    r_sp_missing = xe._batch_run({**base, "steps": [{"kind": "spatial"}]}, dry_run=True, log=lambda _s: None)
    assert r_sp_missing["ok"] is False
    assert any("step 1 (spatial): parameter" in e and "is required" in e for e in r_sp_missing["errors"])

    r_sp_unloaded = xe._batch_run(
        {**base, "steps": [{"kind": "spatial", "params": {"label_key": "labels_celltypist"}}]},
        dry_run=True, log=lambda _s: None,
    )
    assert r_sp_unloaded["ok"] is False
    assert any("labels_celltypist" in e for e in r_sp_unloaded["errors"])

    r_sp_ok = xe._batch_run({
        **base,
        "steps": [
            {"kind": "annotation", "params": {"source": "celltypist"}},
            {"kind": "spatial", "params": {"label_key": "labels_celltypist"}},
        ],
    }, dry_run=True, log=lambda _s: None)
    assert r_sp_ok["ok"] is True, r_sp_ok["errors"]
    assert any("earlier annotation step may provide it" in w for w in r_sp_ok["warnings"])

    # Method mismatch: annotation produces celltypist, spatial asks for labels_rctd -> error!
    r_sp_wrong_method = xe._batch_run({
        **base,
        "steps": [
            {"kind": "annotation", "params": {"source": "celltypist"}},
            {"kind": "spatial", "params": {"label_key": "labels_rctd"}},
        ],
    }, dry_run=True, log=lambda _s: None)
    assert r_sp_wrong_method["ok"] is False

    # Segmentation mismatch: annotation on baysor does NOT satisfy composition/spatial/segqc on xenium!
    r_cross_seg_annot = xe._batch_run({
        **base,
        "steps": [
            {"kind": "annotation", "params": {"seg_source": "baysor:a11ba001", "source": "celltypist"}},
            {"kind": "composition", "params": {"seg_source": "xenium", "labels": "celltype:celltypist"}},
            {"kind": "segqc", "params": {"method": "celltypist"}},
        ],
    }, dry_run=True, log=lambda _s: None)
    assert r_cross_seg_annot["ok"] is False
    assert sum("step 2" in e for e in r_cross_seg_annot["errors"]) >= 1
    assert sum("step 3" in e for e in r_cross_seg_annot["errors"]) >= 1

    # 2. segqc validates method and source against Xenium labels
    r_segqc_unloaded = xe._batch_run(
        {**base, "steps": [{"kind": "segqc", "params": {"method": "celltypist"}}]},
        dry_run=True, log=lambda _s: None,
    )
    assert r_segqc_unloaded["ok"] is False

    with xe._annot_lock:
        xe._annot_state["labels_celltypist"] = pd.Series(
            ["T cell" if i % 2 == 0 else "B cell" for i in range(len(D["df"]))],
            index=D["df"].index.astype(str),
        )
    r_segqc_bad_src = xe._batch_run(
        {**base, "steps": [{"kind": "segqc", "params": {"method": "celltypist", "source": "MartianCell"}}]},
        dry_run=True, log=lambda _s: None,
    )
    assert r_segqc_bad_src["ok"] is False
    assert any("MartianCell" in e for e in r_segqc_bad_src["errors"])

    r_segqc_good = xe._batch_run(
        {**base, "steps": [{"kind": "segqc", "params": {"method": "celltypist", "source": "T cell"}}]},
        dry_run=True, log=lambda _s: None,
    )
    assert r_segqc_good["ok"] is True, r_segqc_good["errors"]

    # 3. txfactor reference requires labels and rejects niche
    r_txf_ref = xe._batch_run(
        {**base, "steps": [{"kind": "txfactor", "params": {"method": "reference"}}]},
        dry_run=True, log=lambda _s: None,
    )
    assert r_txf_ref["ok"] is False
    assert any("parameter 'labels' is required when method is 'reference'" in e for e in r_txf_ref["errors"])

    r_txf_niche = xe._batch_run(
        {**base, "steps": [{"kind": "txfactor", "params": {"method": "reference", "labels": "niche"}}]},
        dry_run=True, log=lambda _s: None,
    )
    assert r_txf_niche["ok"] is False

    r_txf_nmf = xe._batch_run(
        {**base, "steps": [{"kind": "txfactor", "params": {"method": "nmf", "k": 4}}]},
        dry_run=True, log=lambda _s: None,
    )
    assert r_txf_nmf["ok"] is True, r_txf_nmf["errors"]

    # 4. clustering on xenium does NOT mask step-overridden nonexistent cluster_method or cross-seg
    r_xen_override = xe._batch_run({
        **base,
        "steps": [
            {"kind": "clustering"},
            {"kind": "composition", "params": {"labels": "cluster", "cluster_method": "no_such_set"}},
        ],
    }, dry_run=True, log=lambda _s: None)
    assert r_xen_override["ok"] is False
    assert any("no_such_set" in e for e in r_xen_override["errors"])

    r_cross_seg_clust = xe._batch_run({
        **base,
        "steps": [
            {"kind": "clustering", "params": {"seg_source": "baysor:a11ba001"}},
            {"kind": "composition", "params": {"seg_source": "xenium", "labels": "cluster",
                                               "cluster_method": "no_such_set"}},
        ],
    }, dry_run=True, log=lambda _s: None)
    assert r_cross_seg_clust["ok"] is False

    # Preserve Round-10 top-level cluster_method creator warning contract on clustered Xenium
    for creator in ("banksy", "clustering"):
        r_r10 = xe._batch_run({
            **base,
            "cluster_method": f"{creator}_future_domains",
            "steps": [
                {"kind": creator},
                {"kind": "composition", "params": {"labels": "cluster"}},
                {"kind": "de", "params": {"group_a": "cluster:0", "group_b": "rest"}},
                {"kind": "segfree", "params": {"labels": "cluster", "cell_type": "0",
                                               "markers": f"{genes[0]},{genes[1]}"}},
            ],
        }, dry_run=True, log=lambda _s: None)
        assert r_r10["ok"] is True, r_r10["errors"]


# ── U2: Viewer API strict JSON NaN/Inf sanitization & _bounds finite guard ──
def test_r11_u2_viewer_api_and_bounds_nan_inf_hygiene(xe, D, monkeypatch):
    import xenium_viewer_api as va

    def _forbid_nan_consts(tok):
        raise ValueError(f"Non-standard JSON token emitted: {tok}")

    blob = va.pack({
        "min": float("nan"),
        "max": float("inf"),
        "neg": float("-inf"),
        "arr": [np.float32("nan"), np.float64("inf"), 2.5],
    }, {"x": np.array([1.0, 2.0], dtype=np.float32)})
    hlen = int.from_bytes(blob[:4], "little")
    header = json.loads(blob[4:4 + hlen].decode("utf-8"), parse_constant=_forbid_nan_consts)
    assert header["min"] is None and header["max"] is None and header["neg"] is None
    assert header["arr"] == [None, None, 2.5]

    vp = xe._viewer_provider
    # Corrupt centroids & QC column with NaN / Inf and test _bounds + Flask endpoints
    bad_df = D["df"].copy()
    bad_df.iloc[0, bad_df.columns.get_loc("x_centroid")] = np.nan
    bad_df.iloc[1, bad_df.columns.get_loc("y_centroid")] = np.inf
    if "cell_area" in bad_df.columns:
        bad_df.iloc[0, bad_df.columns.get_loc("cell_area")] = np.nan
    monkeypatch.setitem(xe.DATA, "df", bad_df)

    b = vp._bounds(bad_df)
    assert len(b) == 4 and all(np.isfinite(v) for v in b)

    all_nan_df = bad_df.iloc[:3].copy()
    all_nan_df["x_centroid"] = np.nan
    all_nan_df["y_centroid"] = np.nan
    assert vp._bounds(all_nan_df) == [0.0, 0.0, 1.0, 1.0]

    client = xe.app.server.test_client()
    for url in ("/viewer/meta", "/viewer/overlays?source=xenium", "/viewer/cell/0?source=xenium"):
        resp = client.get(url)
        assert resp.status_code == 200, (url, resp.status_code)
        parsed = json.loads(resp.data.decode("utf-8"), parse_constant=_forbid_nan_consts)
        assert isinstance(parsed, dict)

    cells_resp = client.get("/viewer/cells.bin?source=xenium")
    assert cells_resp.status_code == 200
    chlen = int.from_bytes(cells_resp.data[:4], "little")
    chead = json.loads(cells_resp.data[4:4 + chlen].decode("utf-8"), parse_constant=_forbid_nan_consts)
    assert all(np.isfinite(v) for v in chead["bounds"])

    colors_resp = client.get("/viewer/colors.bin?source=xenium&mode=cell_area")
    assert colors_resp.status_code == 200
    col_hlen = int.from_bytes(colors_resp.data[:4], "little")
    col_head = json.loads(colors_resp.data[4:4 + col_hlen].decode("utf-8"), parse_constant=_forbid_nan_consts)
    assert isinstance(col_head, dict)


# ── U3: Node verification of WebGL figure export & [r,g,b] overlay colours ──
def test_r11_u3_webgl_figure_export_and_array_colors_node():
    node = shutil.which("node") or ("/opt/homebrew/bin/node" if os.path.exists("/opt/homebrew/bin/node") else None)
    if not node:
        pytest.skip("node binary not available")
    script = ROOT / "viewer_src" / "test" / "round11_figure_checks.mjs"
    proc = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 0, f"round11_figure_checks failed:\n{proc.stdout}\n{proc.stderr}"
    assert "0 failed" in proc.stdout


# ── U4: _batch_preconfigure zero disk writes on --dry-run and --list-steps ──
def test_r11_u4_batch_preconfigure_dry_run_list_steps_zero_disk_writes(bundle_dir, tmp_path, monkeypatch):
    import xenium_app

    for flag in ("--dry-run", "--list-steps"):
        ghost_cache = tmp_path / f"never_created_{flag.strip('-')}"
        cfg_file = tmp_path / f"cfg_{flag.strip('-')}.json"
        cfg_file.write_text(json.dumps({
            "data": {
                "path": str(bundle_dir),
                "h5ad_mapping": {"x": "spatial_x", "y": "spatial_y"},
            },
            "cache_dir": str(ghost_cache),
            "steps": [{"kind": "moran"}],
        }), encoding="utf-8")

        env_clean = {k: v for k, v in os.environ.items()
                     if k not in ("DATA_DIR", "XENIUM_CACHE_DIR", "XENIUM_H5AD_MAPPING")}
        monkeypatch.setattr(os, "environ", env_clean)
        monkeypatch.setattr(sys, "argv", ["-m", str(cfg_file), flag])
        monkeypatch.setattr(sys, "orig_argv", ["python", "-m", "xenium_app.batch", str(cfg_file), flag], raising=False)

        assert xenium_app._batch_preconfigure() is True
        assert not ghost_cache.exists(), f"{flag} created cache_dir {ghost_cache}!"
        assert os.environ.get("XENIUM_CACHE_DIR") == str(ghost_cache)
        h5_map_path = os.environ.get("XENIUM_H5AD_MAPPING")
        assert h5_map_path and os.path.isfile(h5_map_path)
        assert not pathlib.Path(h5_map_path).is_relative_to(ghost_cache)
        assert json.loads(pathlib.Path(h5_map_path).read_text(encoding="utf-8"))["x"] == "spatial_x"
        os.unlink(h5_map_path)


# ── U5: select_cell duplicate reseg cell IDs & NaN transcript_counts/centroids ──
def test_r11_u5_select_cell_reseg_duplicate_id_and_nan_counts(xe, D):
    genes = list(D["gene_names"][:4])
    gni = {g: i for i, g in enumerate(genes)}
    cells_df = pd.DataFrame(
        {
            "x_centroid": [np.nan, 20.0, 30.0],
            "y_centroid": [15.0, 25.0, 35.0],
            "transcript_counts": [np.nan, 12.0, 8.0],
            "cell_area": [np.nan, 50.0, 60.0],
        },
        index=pd.Index(["dup_cell", "dup_cell", "unique_cell"]),
    )
    expr = sp.csr_matrix(np.array([[3.0, 0.0, 1.0, 0.0],
                                   [0.0, 2.0, 0.0, 0.0],
                                   [1.0, 1.0, 0.0, 0.0]], dtype=np.float32))
    with xe._proseg_lock:
        xe._proseg_state["status"] = "done"
        xe._proseg_state["result"] = {
            "cells_df": cells_df,
            "expr": expr,
            "gene_name_to_idx": gni,
            "cell_bounds": {},
        }
    out = xe.show_cell_info(
        {"id": "dup_cell", "source": "proseg"},
        D["cluster_methods"][0] if D["cluster_methods"] else None,
    )

    def _collect_strings(node):
        if isinstance(node, str):
            return [node]
        if isinstance(node, (list, tuple)):
            return [s for item in node for s in _collect_strings(item)]
        if hasattr(node, "children"):
            return _collect_strings(node.children)
        return []

    rendered = " | ".join(_collect_strings(out))
    assert "dup_cell" in rendered
    assert "—" in rendered
    assert "NaN" not in rendered and "nan" not in rendered


# ── U6: CellNEST clust__* column join & sender_type colouring across all 3 paths ──
def test_r11_u6_cellnest_clust_join_and_sender_type_coloring(xe, D, tmp_path):
    cell_ids = list(D["df"].index.astype(str)[:6])
    folder = tmp_path / "cellnest_run"
    ds_name = "test_ds"
    (folder / "output" / ds_name).mkdir(parents=True)
    (folder / "metadata" / ds_name).mkdir(parents=True)

    # Write minimal CellNEST CSV + metadata so _load_cellnest_data exercises the real join
    coords = D["df"].iloc[:6][["x_centroid", "y_centroid"]].reset_index(drop=True)
    coords.to_csv(folder / "metadata" / ds_name / f"coordinates_{ds_name}.csv", header=False, index=False)
    pd.DataFrame({"barcode": cell_ids}).to_csv(
        folder / "metadata" / ds_name / f"cell_barcode_{ds_name}.csv", header=False, index=False
    )
    ccc_raw = pd.DataFrame({
        "from_id": [0, 1, 2],
        "to_id": [3, 4, 5],
        "ligand": ["LIG1", "LIG1", "LIG1"],
        "receptor": ["REC1", "REC1", "REC1"],
        "attention_score": [0.9, 0.6, 0.3],
    })
    ccc_raw.to_csv(folder / "output" / ds_name / f"CellNEST_{ds_name}_top20percent.csv", index=False)
    pd.DataFrame({"Ligand-Receptor Pairs": ["LIG1-REC1"], "Total Count": [3]}).to_csv(
        folder / "output" / ds_name / f"CellNEST_{ds_name}_histogram_byFrequency_table_top1500.csv", index=False
    )

    xe._load_cellnest_data(str(folder), ds_name, strip_suffix="")
    with xe._cellnest_lock:
        assert xe._cellnest_state["status"] == "done", xe._cellnest_state["message"]
        loaded = xe._cellnest_state["data"]
    assert loaded is not None
    expected_clust_cols = [c for c in D["df"].columns if c.startswith("clust__")]
    if expected_clust_cols:
        assert any(c.startswith("clust__") for c in loaded["annot_cols"]), loaded["annot_cols"]

    # Inject deterministic sender/receiver types and verify all 3 spatial paths on sender_type
    loaded["ccc_df"]["from_cell_type_test"] = ["TypeA", "TypeB", "TypeA"]
    loaded["ccc_df"]["to_cell_type_test"] = ["TypeB", "TypeA", "TypeB"]
    if "cell_type_test" not in loaded["annot_cols"]:
        loaded["annot_cols"].append("cell_type_test")
    with xe._cellnest_lock:
        xe._cellnest_state["data"] = loaded

    pair = "LIG1-REC1"
    fig_sp = xe._cellnest_make_spatial([pair], top_n=10, color_by="sender_type",
                                       annot_col="cell_type_test", edge_width=1.5)
    line_colors_sp = {t.line.color for t in fig_sp.data if getattr(t, "mode", "") == "lines" and t.line.color}
    assert len(line_colors_sp) >= 2, f"Expected distinct sender colours in spatial fig, got {line_colors_sp}"

    ov_traces = xe._cellnest_overlay_traces(
        [pair], top_n=10, edge_width=1.5, color_by="sender_type",
        options=["gray_bg"], annot_col="cell_type_test",
    )
    ov_line_colors = {t.line.color for t in ov_traces if getattr(t, "mode", "") == "lines" and t.line.color}
    assert len(ov_line_colors) >= 2, f"Expected distinct sender colours in overlay traces, got {ov_line_colors}"

    meta_cn, arrs_cn = xe._viewer_provider._cellnest({
        "source": "xenium",
        "q": json.dumps({
            "pairs": [pair],
            "top_n": 10,
            "color_by": "sender_type",
            "annot_col": "cell_type_test",
        }),
    })
    assert meta_cn["edges"] == 3
    rgba = arrs_cn["rgba"].reshape(-1, 4)
    # First 3 rows are the main edge segments for the 3 edges (TypeA, TypeB, TypeA)
    assert not np.array_equal(rgba[0, :3], rgba[1, :3]), f"Expected distinct sender RGBs, got {rgba[:3, :3]}"
    assert np.array_equal(rgba[0, :3], rgba[2, :3])


# ── U7 + S5 + D6b: Multi-sample parity, SPLIT uncovered NaN mask, safe log1p ──
def test_r11_u7_s5_d6b_multi_sample_parity_split_mask_and_safe_log1p(xe, D, monkeypatch):
    vp = xe._viewer_provider
    gene = D["gene_names"][0]

    # Fabricate an extra dataset for multi-sample tests
    extra_df = D["df"].iloc[:8].copy()
    extra_df.index = [f"ex_{i}" for i in range(len(extra_df))]
    extra = {
        "df": extra_df,
        "expr": D["expr"][:8],
        "expr_csc": D["expr"][:8].tocsc(),
        "df_to_expr": np.arange(len(extra_df)),
        "gene_names": list(D["gene_names"]),
        "gene_name_to_idx": dict(D["gene_name_to_idx"]),
        "cluster_methods": list(D.get("cluster_methods") or []),
        "x_offset": 1500.0,
    }
    xe.EXTRA_DATASETS.append(extra)

    # U7: WebGL _colors_multi colormap parity
    m_gene, _ = vp.colors({"source": "xenium", "mode": "gene", "gene": gene})
    assert m_gene.get("colormap") == "plasma"
    m_tx, _ = vp.colors({"source": "xenium", "mode": "transcript_counts"})
    assert m_tx.get("colormap") == "viridis"
    for qc_col, qc_cs in xe.COLORSCALES.items():
        if qc_col in D["df"].columns and qc_col != "transcript_counts":
            m_qc, _ = vp.colors({"source": "xenium", "mode": qc_col})
            assert m_qc.get("colormap") == qc_cs.lower()

    # U7: Plotly multi-sample missing column falls back to Transcript Counts (log1p) + Viridis
    fig_ms = xe.make_spatial_fig(
        color_by="nonexistent_qc_column", method=None, gene=gene,
        size=4, opacity=0.8, boundary_toggles=[], relayout=None,
    )
    assert fig_ms.data[0].name == "Transcript Counts (log1p)"

    # S5: SPLIT-corrected imputed gene masks uncovered cells to NaN (not 0.0)
    xe.EXTRA_DATASETS.clear()
    n_expr = D["expr"].shape[0]
    n_cov = max(1, n_expr // 2)
    raw_dense = np.maximum(D["expr"].toarray(), 1.0).astype(np.float32)  # every cell has raw counts > 0
    corr_dense = raw_dense.copy()
    corr_dense[n_cov:, :] = 0.0                                          # SPLIT dropped rows n_cov..n_expr-1
    imp_col = np.full((n_expr, 1), 5.0, dtype=np.float32)
    imp_col[n_cov:, :] = 0.0
    aug_csr = sp.csr_matrix(np.hstack([raw_dense, imp_col]))
    aug_csc = aug_csr.tocsc()
    aug_gni = {**D["gene_name_to_idx"], "R11_IMP_GENE": aug_csc.shape[1] - 1}
    monkeypatch.setitem(xe.DATA, "expr", aug_csr)
    monkeypatch.setitem(xe.DATA, "expr_csc", aug_csc)
    monkeypatch.setitem(xe.DATA, "gene_name_to_idx", aug_gni)
    monkeypatch.setitem(xe.DATA, "imputed_genes", ["R11_IMP_GENE"])
    monkeypatch.setitem(xe.DATA, "split_corrected_expr", sp.csc_matrix(corr_dense))
    monkeypatch.setitem(xe.DATA, "_split_cov_src", None)
    monkeypatch.setitem(xe.DATA, "split_corrected_imputed_genes", ["R11_IMP_GENE"])

    vals_s5 = xe._get_expr_values("R11_IMP_GENE [imp]", None, use_corrected=True)
    assert vals_s5 is not None and len(vals_s5) == len(D["df"])
    assert np.all(np.isfinite(vals_s5[:n_cov]))
    assert np.all(np.isnan(vals_s5[n_cov:])), "Uncovered SPLIT cells must be NaN, not 0!"

    # D6b: Negative finite values clamped to 0 without RuntimeWarning while preserving NaNs
    neg_arr = np.array([-4.0, -0.5, np.nan, 0.0, 3.0], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out_safe = xe._fig_safe_log1p(neg_arr)
        assert out_safe.dtype == np.float32
        assert out_safe[0] == 0.0 and out_safe[1] == 0.0
        assert np.isnan(out_safe[2])
        assert np.isclose(out_safe[4], np.log1p(3.0))

        neg_df = D["df"].copy()
        neg_df["transcript_counts"] = np.linspace(-5.0, 10.0, len(neg_df))
        monkeypatch.setitem(xe.DATA, "df", neg_df)
        m_neg, a_neg = vp.colors({"source": "xenium", "mode": "transcript_counts"})
        assert np.all(np.isfinite(a_neg["values"])) and a_neg["values"].min() >= 0.0


# ── U8: _prov_methods_md variant deduplication preserves distinct runs & snapshots ──
def test_r11_u8_prov_methods_md_preserves_distinct_variants_and_snapshots(xe):
    def _mk(rec_id, kind, params, headline=None, labels=None):
        return {
            "id": rec_id,
            "kind": kind,
            "title": kind,
            "created_at": f"2026-11-14T00:00:0{rec_id}+00:00",
            "cache": "computed",
            "data": {
                "dataset": "bundle",
                "seg_source": "xenium",
                "n_cells": 100,
                "counts": {"requested": "original", "used": "original"},
                **({"labels": labels} if labels else {}),
            },
            "params": params,
            "headline": headline or {},
        }

    # Distinct variants of banksy, spillover, txfactor, and aggregates must ALL survive
    b1 = _mk(1, "banksy", {"lam": 0.2, "k_geom": 15, "res": 0.5}, {"name": "banksy_02", "n_clusters": 4})
    b2 = _mk(2, "banksy", {"lam": 0.8, "k_geom": 15, "res": 0.5}, {"name": "banksy_08", "n_clusters": 6})
    s1 = _mk(3, "spillover", {"marker_source": "panel"}, labels={"label_kind": "cluster", "cluster_method": "graphclust"})
    s2 = _mk(4, "spillover", {"marker_source": "reference"}, labels={"label_kind": "cluster", "cluster_method": "graphclust"})
    t1 = _mk(5, "txfactor", {"method": "nmf", "k": 5, "bin": 10})
    t2 = _mk(6, "txfactor", {"method": "nmf", "k": 12, "bin": 10})
    t3 = _mk(7, "txfactor", {"method": "reference", "bin": 10}, labels={"label_kind": "celltype:celltypist", "cluster_method": None})
    a1 = _mk(8, "aggregates", {"classes": ["B cell"], "radius": 30, "min_cells": 10},
             labels={"label_kind": "celltype:celltypist", "cluster_method": None})
    a2 = _mk(9, "aggregates", {"classes": ["B cell", "T cell"], "radius": 30, "min_cells": 10},
             labels={"label_kind": "celltype:celltypist", "cluster_method": None})

    assert xe._pm_variant(b1) != xe._pm_variant(b2)
    assert xe._pm_variant(s1) != xe._pm_variant(s2)
    assert xe._pm_variant(t1) != xe._pm_variant(t2)
    assert xe._pm_variant(t1) != xe._pm_variant(t3)
    assert xe._pm_variant(a1) != xe._pm_variant(a2)

    # Identical variant still deduplicates in latest_only=True
    t1_repeat = _mk(10, "txfactor", {"method": "nmf", "k": 5, "bin": 10})
    assert xe._pm_variant(t1) == xe._pm_variant(t1_repeat)
    md = xe._prov_methods_md([b1, b2, t1, t2, t1_repeat], latest_only=True)
    assert "0.2" in md and "0.8" in md
    assert f"Record {t1['id']} (" not in md and f"Record {t1_repeat['id']} (" in md


# ── U9: Session restore per-seg cell_type:<m> & strict [imp] gene validation ──
def test_r11_u9_session_restore_reseg_celltype_and_imp_gene_validation(xe, D):
    _write_cached_baysor_run(xe, D, tag="a11ba099")
    gene = str(D["gene_names"][0])
    defaults = {cid: xe._session_default(cid) for cid in xe._session_controls()}

    with xe._annot_lock:
        for k in list(xe._annot_state):
            if k == "labels" or k.startswith("labels_"):
                xe._annot_state[k] = None
        # Only Xenium has celltypist labels — Baysor does NOT
        xe._annot_state["labels_celltypist"] = pd.Series(
            ["T cell"] * len(D["df"]), index=D["df"].index.astype(str)
        )

    # 1. On baysor:a11ba099, cell_type:celltypist must NOT be accepted from Xenium's labels
    doc_bay = xe._session_snapshot({**defaults}, "download")
    doc_bay["state"]["controls"]["seg-source"] = "baysor:a11ba099"
    doc_bay["state"]["controls"]["color-by"] = "cell_type:celltypist"
    rep_bay = xe._session_restore_text(json.dumps(doc_bay), dict(defaults), "test", record=False)
    assert rep_bay["controls"].get("color-by") != "cell_type:celltypist"
    assert any("color-by" in n for n in rep_bay["missing"])

    # Conversely, stale reseg labels alone must NOT enable cell_type:rctd on xenium
    with xe._annot_lock:
        xe._annot_state["labels_rctd_baysor_a11ba099"] = pd.Series(["B cell"], index=["bcell_0"])
    assert "cell_type:rctd" not in xe._session_color_by_values("xenium")

    # 2. [imp] gene suffixes rejected on webgl-tx-genes & density-gene, and gated on gene-selector
    doc_genes = xe._session_snapshot({**defaults}, "download")
    doc_genes["state"]["controls"]["seg-source"] = "xenium"
    doc_genes["state"]["controls"]["webgl-tx-genes"] = [f"{gene} [imp]"]
    doc_genes["state"]["controls"]["density-gene"] = f"{gene} [imp]"
    doc_genes["state"]["controls"]["gene-selector"] = f"{gene} [imp]"
    rep_g = xe._session_restore_text(json.dumps(doc_genes), dict(defaults), "test", record=False)
    assert rep_g["controls"].get("webgl-tx-genes") != [f"{gene} [imp]"]
    assert rep_g["controls"].get("density-gene") != f"{gene} [imp]"
    assert rep_g["controls"].get("gene-selector") != f"{gene} [imp]"
    assert sum("not on the gene panel" in m for m in rep_g["missing"]) == 3

    # Once SpaGE imputes an unmeasured gene on Xenium, gene-selector accepts "<gene> [imp]"
    # while webgl-tx-genes and density-gene STILL reject "<gene> [imp]"!
    with xe._spage_lock:
        xe._spage_state["result_genes"] = ["R11_SPAGE_ONLY"]
    doc_imp = xe._session_snapshot({**defaults}, "download")
    doc_imp["state"]["controls"]["seg-source"] = "xenium"
    doc_imp["state"]["controls"]["gene-selector"] = "R11_SPAGE_ONLY [imp]"
    doc_imp["state"]["controls"]["webgl-tx-genes"] = ["R11_SPAGE_ONLY [imp]"]
    doc_imp["state"]["controls"]["density-gene"] = "R11_SPAGE_ONLY [imp]"
    rep_imp = xe._session_restore_text(json.dumps(doc_imp), dict(defaults), "test", record=False)
    assert rep_imp["controls"].get("gene-selector") == "R11_SPAGE_ONLY [imp]"
    assert rep_imp["controls"].get("webgl-tx-genes") != ["R11_SPAGE_ONLY [imp]"]
    assert rep_imp["controls"].get("density-gene") != "R11_SPAGE_ONLY [imp]"
