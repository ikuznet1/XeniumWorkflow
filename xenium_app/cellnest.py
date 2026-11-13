"""CellNEST cell-cell communication: loading, filters, plots, downstream TFs."""

import os
import threading
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative

from . import provenance
from . import core, jobs


# ─── CellNEST helper functions ────────────────────────────────────────────────

def _load_cellnest_data(folder: str, dataset_name: str, strip_suffix: str = ""):
    """Load CellNEST output files in a background thread and store in _cellnest_state."""
    import os, traceback
    _gen = jobs._dataset_gen
    try:
        with core._cellnest_lock:
            core._cellnest_state["status"] = "loading"
            core._cellnest_state["message"] = "Loading CellNEST data…"

        ccc_path = os.path.join(folder, "output", dataset_name,
                                f"CellNEST_{dataset_name}_top20percent.csv")
        freq_path = os.path.join(folder, "output", dataset_name,
                                 f"CellNEST_{dataset_name}_histogram_byFrequency_table_top1500.csv")
        bc_path  = os.path.join(folder, "metadata", dataset_name,
                                f"cell_barcode_{dataset_name}.csv")

        jobs._job_progress("Reading CellNEST tables…")
        ccc_df  = pd.read_csv(ccc_path)
        lr_freq = pd.read_csv(freq_path)
        barcodes = pd.read_csv(bc_path, header=None)[0].tolist()
        jobs._job_check()
        jobs._job_progress("Matching cells and annotations…")

        # Load CellNEST's own coordinate file (one row per cell, no header)
        coord_path = os.path.join(folder, "metadata", dataset_name,
                                  f"coordinates_{dataset_name}.csv")
        coords = pd.read_csv(coord_path, header=None, names=["x", "y"])
        n_coords = len(coords)
        print(f"[CellNEST] Loaded {n_coords} cell coordinates", flush=True)

        # from_id/to_id are 0-based row indices into barcodes & coordinates
        def _lookup(idx):
            try:
                idx = int(idx)
                if 0 <= idx < n_coords:
                    return float(coords.iat[idx, 0]), float(coords.iat[idx, 1])
            except Exception:
                pass
            return None, None

        from_xy = [_lookup(i) for i in ccc_df["from_id"]]
        to_xy   = [_lookup(i) for i in ccc_df["to_id"]]
        ccc_df["from_x"] = [v[0] for v in from_xy]
        ccc_df["from_y"] = [v[1] for v in from_xy]
        ccc_df["to_x"]   = [v[0] for v in to_xy]
        ccc_df["to_y"]   = [v[1] for v in to_xy]

        # Drop rows with missing coordinates
        ccc_df = ccc_df.dropna(subset=["from_x", "from_y", "to_x", "to_y"])

        # ── Map CellNEST barcodes → Proseg/app cell IDs ──────────────────────
        # Barcode format: "{cell_id}_4" — strip suffix to get the app cell ID directly.
        # Build barcode → app_cell_id map from the barcodes list.
        def _strip(barcode_str):
            s = str(barcode_str)
            if strip_suffix and s.endswith(strip_suffix):
                return s[:-len(strip_suffix)]
            return s

        # from_id/to_id index into the barcodes list
        bc_arr = np.array(barcodes)
        from_barcodes = bc_arr[ccc_df["from_id"].values.astype(int)]
        to_barcodes   = bc_arr[ccc_df["to_id"].values.astype(int)]
        ccc_df["from_app_id"] = [_strip(b) for b in from_barcodes]
        ccc_df["to_app_id"]   = [_strip(b) for b in to_barcodes]

        print(f"[CellNEST] Direct barcode→ID mapping; "
              f"sample from_app_id={ccc_df['from_app_id'].head(3).tolist()}", flush=True)

        # ── Join Proseg/app annotations onto ccc_df ───────────────────────────
        annot_cols = []
        with core._proseg_lock:
            _pr = core._proseg_state.get("result")
        with core._baysor_lock:
            _br = core._baysor_state.get("result")
        seg_df = None
        if _pr is not None and "cells_df" in _pr:
            seg_df = _pr["cells_df"]
        elif _br is not None and "cells_df" in _br:
            seg_df = _br["cells_df"]
        else:
            seg_df = core.DATA["df"]

        if seg_df is not None:
            cand_cols = [c for c in seg_df.columns
                         if c.startswith("cluster") or c.startswith("cell_type")]
            if cand_cols:
                seg_sub = seg_df[cand_cols].copy()
                seg_sub.index = seg_sub.index.astype(str)
                for col in cand_cols:
                    ccc_df[f"from_{col}"] = ccc_df["from_app_id"].map(seg_sub[col])
                    ccc_df[f"to_{col}"]   = ccc_df["to_app_id"].map(seg_sub[col])
                    annot_cols.append(col)
                print(f"[CellNEST] Joined {len(annot_cols)} annotation columns: {annot_cols[:5]}", flush=True)
            else:
                print("[CellNEST] No cluster/cell_type columns found in seg df", flush=True)

        # ── Also join annotation labels from _annot_state (Seurat/RCTD/CellTypist) ──
        with core._annot_lock:
            annot_series = [(k, v.copy()) for k, v in core._annot_state.items()
                            if isinstance(v, pd.Series) and len(v) > 0]
        for key, labels in annot_series:
            # key like "labels_proseg", "labels_seurat_proseg:abc12345"
            # Make a short human-readable column name
            col_name = key.replace("labels_", "celltype_", 1)
            labels.index = labels.index.astype(str)
            ccc_df[f"from_{col_name}"] = ccc_df["from_app_id"].map(labels)
            ccc_df[f"to_{col_name}"]   = ccc_df["to_app_id"].map(labels)
            annot_cols.append(col_name)
        if annot_series:
            print(f"[CellNEST] Joined annotation-state labels: {[k for k,_ in annot_series]}", flush=True)

        jobs._job_check()   # last cancellation point
        with core._cellnest_lock:
            if _gen != jobs._dataset_gen:
                # another sample was loaded meanwhile: these results are not its results
                print("[CellNEST] dataset was replaced while loading - result discarded", flush=True)
                return
            core._cellnest_state["data"] = {
                "ccc_df": ccc_df,
                "lr_freq": lr_freq,
                "barcodes": barcodes,
                "dataset_name": dataset_name,
                "folder": folder,
                "coords": coords,
                "annot_cols": annot_cols,
            }
            core._cellnest_state["status"] = "done"
            core._cellnest_state["message"] = f"Loaded {len(ccc_df):,} interactions"
        provenance._prov_record(
            "cellnest", params={"dataset_name": dataset_name, "strip_suffix": strip_suffix},
            result={"n_interactions": int(len(ccc_df))},
            message=f"Loaded {len(ccc_df):,} interactions", cache=False)
    except jobs._JobCancelled:
        raise
    except Exception as exc:
        with core._cellnest_lock:
            if _gen == jobs._dataset_gen:
                core._cellnest_state["status"] = "error"
                core._cellnest_state["message"] = f"Error: {exc}"
        print(f"[CellNEST] load error: {traceback.format_exc()}", flush=True)


def _apply_cellnest_filters(ccc_df, f_ligand=None, f_receptor=None,
                             f_sender=None, f_receiver=None, annot_col=None):
    """Apply ligand/receptor/sender/receiver filters to a CellNEST interactions DataFrame."""
    sub = ccc_df
    if f_ligand:
        vals = f_ligand if isinstance(f_ligand, list) else [f_ligand]
        sub = sub[sub["ligand"].isin(vals)]
    if f_receptor:
        vals = f_receptor if isinstance(f_receptor, list) else [f_receptor]
        sub = sub[sub["receptor"].isin(vals)]
    if annot_col and f_sender:
        col = f"from_{annot_col}"
        if col in sub.columns:
            vals = f_sender if isinstance(f_sender, list) else [f_sender]
            sub = sub[sub[col].isin(vals)]
    if annot_col and f_receiver:
        col = f"to_{annot_col}"
        if col in sub.columns:
            vals = f_receiver if isinstance(f_receiver, list) else [f_receiver]
            sub = sub[sub[col].isin(vals)]
    return sub


def _cellnest_split_pair(pair: str, ccc_df) -> tuple:
    """Split a 'LIGAND-RECEPTOR' label. Symbols may themselves contain '-'
    (HLA-DRA, MT-CO2), so with more than one hyphen pick the split that exists
    in ccc_df; otherwise split on the first hyphen as before."""
    parts = str(pair).split("-")
    if len(parts) > 2:
        known = set(zip(ccc_df["ligand"], ccc_df["receptor"]))
        for i in range(1, len(parts)):
            cand = ("-".join(parts[:i]), "-".join(parts[i:]))
            if cand in known:
                return cand
    return tuple(str(pair).split("-", 1))


def _cellnest_make_bar(f_ligand=None, f_receptor=None,
                       f_sender=None, f_receiver=None, annot_col=None):
    """Return a Plotly bar chart using the pre-computed CellNEST histogram CSV."""
    with core._cellnest_lock:
        data = core._cellnest_state.get("data")
    if data is None:
        return go.Figure()
    lr_freq = data.get("lr_freq")
    if lr_freq is None or lr_freq.empty:
        return go.Figure(layout=go.Layout(paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG))

    # lr_freq columns: "Ligand-Receptor Pairs", "Total Count"
    freq = lr_freq.copy()
    freq.columns = [c.strip() for c in freq.columns]
    pair_col  = freq.columns[0]   # "Ligand-Receptor Pairs"
    count_col = freq.columns[1]   # "Total Count"

    # Apply ligand/receptor name filters if set
    # (the filter dropdowns are multi-select, so these arrive as lists)
    if f_ligand:
        _lv = f_ligand if isinstance(f_ligand, list) else [f_ligand]
        freq = freq[freq[pair_col].astype(str).str.startswith(tuple(f"{v}-" for v in _lv))]
    if f_receptor:
        _rv = f_receptor if isinstance(f_receptor, list) else [f_receptor]
        freq = freq[freq[pair_col].astype(str).str.endswith(tuple(f"-{v}" for v in _rv))]

    freq = freq.head(30)
    if freq.empty:
        return go.Figure(layout=go.Layout(paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG))

    fig = go.Figure(go.Bar(
        x=freq[count_col].tolist(),
        y=freq[pair_col].tolist(),
        orientation="h",
        marker_color="#0d9488",
    ))
    fig.update_layout(
        paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG,
        font={"color": core.TEXT, "size": 10},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        title={"text": "CellNEST LR pair frequency (top 30)", "font": {"size": 12}},
        xaxis={"gridcolor": core.BORDER, "title": "Total Count"},
        yaxis={"autorange": "reversed"},
    )
    return fig


def _cellnest_make_lr_heatmap(f_ligand=None, f_receptor=None, f_sender=None,
                               f_receiver=None, annot_col=None, top_n=20):
    """Ligand × Receptor heatmap colored by mean attention score."""
    with core._cellnest_lock:
        data = core._cellnest_state.get("data")
    if data is None:
        return go.Figure(layout=go.Layout(paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG))
    sub = _apply_cellnest_filters(data["ccc_df"], f_ligand, f_receptor, f_sender, f_receiver, annot_col)
    if sub.empty:
        return go.Figure(layout=go.Layout(paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG))
    top_lig = sub.groupby("ligand")["attention_score"].sum().nlargest(top_n).index.tolist()
    top_rec = sub.groupby("receptor")["attention_score"].sum().nlargest(top_n).index.tolist()
    sub2 = sub[sub["ligand"].isin(top_lig) & sub["receptor"].isin(top_rec)]
    pivot = (sub2.groupby(["ligand", "receptor"])["attention_score"].mean()
               .unstack(fill_value=0)
               .reindex(index=top_lig, columns=top_rec, fill_value=0))
    fig = go.Figure(go.Heatmap(
        z=pivot.values.tolist(), x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale="Viridis", colorbar={"title": "Mean attn", "thickness": 10, "len": 0.6},
        hovertemplate="Ligand: %{y}<br>Receptor: %{x}<br>Score: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG, font={"color": core.TEXT, "size": 10},
        margin={"l": 10, "r": 60, "t": 30, "b": 80},
        title={"text": f"Ligand × Receptor (mean attention, top {top_n})", "font": {"size": 11}},
        xaxis={"tickangle": -45, "title": "Receptor", "gridcolor": core.BORDER},
        yaxis={"autorange": "reversed", "title": "Ligand"},
    )
    return fig


def _cellnest_make_ligtarget_heatmap(annot_col=None, f_ligand=None, f_receptor=None,
                                      f_sender=None, f_receiver=None, top_n=20):
    """Ligand (rows) × Receiver cell type (cols), color = mean attention score.
    Falls back to LR heatmap when no annotation is selected."""
    with core._cellnest_lock:
        data = core._cellnest_state.get("data")
    if data is None:
        return go.Figure(layout=go.Layout(paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG))
    sub = _apply_cellnest_filters(data["ccc_df"], f_ligand, f_receptor, f_sender, f_receiver, annot_col)
    if sub.empty:
        return go.Figure(layout=go.Layout(paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG))
    if not annot_col:
        return _cellnest_make_lr_heatmap(f_ligand=f_ligand, f_receptor=f_receptor,
                                         f_sender=f_sender, f_receiver=f_receiver, top_n=top_n)
    to_col = f"to_{annot_col}"
    if to_col not in sub.columns:
        return go.Figure(layout=go.Layout(paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG))
    sub = sub.dropna(subset=[to_col])
    if sub.empty:
        return go.Figure(layout=go.Layout(paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG))
    top_lig = sub.groupby("ligand")["attention_score"].sum().nlargest(top_n).index.tolist()
    sub2 = sub[sub["ligand"].isin(top_lig)]
    pivot = (sub2.groupby(["ligand", to_col])["attention_score"].mean()
               .unstack(fill_value=0)
               .reindex(index=top_lig, fill_value=0))
    # Sort receiver cols by total attention
    col_order = pivot.sum(axis=0).sort_values(ascending=False).index.tolist()
    pivot = pivot[col_order]
    fig = go.Figure(go.Heatmap(
        z=pivot.values.tolist(), x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale="Purples", colorbar={"title": "Mean attn", "thickness": 10, "len": 0.6},
        hovertemplate="Ligand: %{y}<br>Receiver: %{x}<br>Score: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG, font={"color": core.TEXT, "size": 10},
        margin={"l": 10, "r": 60, "t": 30, "b": 80},
        title={"text": f"Ligand → Receiver cell type (mean attention, top {top_n})", "font": {"size": 11}},
        xaxis={"tickangle": -45, "title": "Receiver cell type", "gridcolor": core.BORDER},
        yaxis={"autorange": "reversed", "title": "Ligand"},
    )
    return fig


def _cellnest_make_dotplot(annot_col=None, f_ligand=None, f_receptor=None,
                            f_sender=None, f_receiver=None, top_n=20):
    """DotPlot: top ligands (y) × sender cell types (x).
    Dot size = fraction of sender-type cells sending this ligand.
    Dot color = mean attention score."""
    with core._cellnest_lock:
        data = core._cellnest_state.get("data")
    if data is None:
        return go.Figure(layout=go.Layout(paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG))
    ccc_df = data["ccc_df"]
    sub = _apply_cellnest_filters(ccc_df, f_ligand, f_receptor, f_sender, f_receiver, annot_col)
    if sub.empty or not annot_col:
        return go.Figure(layout=go.Layout(
            paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG,
            annotations=[{"text": "Select an Annotation column to enable DotPlot",
                           "showarrow": False, "font": {"color": core.MUTED, "size": 12},
                           "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5}]))
    from_col = f"from_{annot_col}"
    if from_col not in sub.columns:
        return go.Figure(layout=go.Layout(paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG))
    sub = sub.dropna(subset=[from_col])
    if sub.empty:
        return go.Figure(layout=go.Layout(paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG))
    top_lig = sub.groupby("ligand")["attention_score"].sum().nlargest(top_n).index.tolist()
    sub2 = sub[sub["ligand"].isin(top_lig)]
    sender_types = sorted(sub2[from_col].dropna().unique(), key=str)
    # Total unique sender cell IDs per type (denominator for fraction)
    if from_col in ccc_df.columns:
        total_per_type = (ccc_df.dropna(subset=[from_col])
                          .groupby(from_col)["from_id"].nunique())
    else:
        total_per_type = pd.Series(dtype=float)
    rows = []
    for lig in top_lig:
        lig_sub = sub2[sub2["ligand"] == lig]
        for st in sender_types:
            st_sub = lig_sub[lig_sub[from_col] == st]
            if st_sub.empty:
                continue
            n_send = st_sub["from_id"].nunique()
            n_total = int(total_per_type.get(st, n_send))
            rows.append({"ligand": lig, "sender": str(st),
                         "fraction": n_send / max(n_total, 1),
                         "mean_attn": st_sub["attention_score"].mean(),
                         "n_send": n_send})
    if not rows:
        return go.Figure(layout=go.Layout(paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG))
    df_dot = pd.DataFrame(rows)
    fig = go.Figure(go.Scatter(
        x=df_dot["sender"].tolist(),
        y=df_dot["ligand"].tolist(),
        mode="markers",
        marker={
            "size": (df_dot["fraction"] * 30).clip(lower=2).tolist(),
            "color": df_dot["mean_attn"].tolist(),
            "colorscale": "Teal",
            "showscale": True,
            "colorbar": {"title": "Mean<br>attn", "thickness": 10, "len": 0.6},
            "line": {"width": 0},
        },
        hovertemplate=(
            "Ligand: %{y}<br>Sender: %{x}<br>"
            "Fraction: %{customdata[0]:.1%}<br>"
            "Mean attn: %{customdata[1]:.4f}<br>"
            "N sending: %{customdata[2]}<extra></extra>"
        ),
        customdata=list(zip(df_dot["fraction"], df_dot["mean_attn"], df_dot["n_send"])),
    ))
    fig.update_layout(
        paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG, font={"color": core.TEXT, "size": 10},
        margin={"l": 10, "r": 60, "t": 30, "b": 80},
        title={"text": f"DotPlot: ligand × sender type (top {top_n})", "font": {"size": 11}},
        xaxis={"title": "Sender cell type", "tickangle": -45, "gridcolor": core.BORDER},
        yaxis={"title": "Ligand", "autorange": "reversed", "gridcolor": core.BORDER},
    )
    return fig


def _cellnest_make_spatial(lr_pair, top_n=100, color_by="_default",
                           annot_col=None, f_ligand=None, f_receptor=None,
                           f_sender=None, f_receiver=None,
                           edge_weight=False, show_autocrine=False, node_degree=False,
                           edge_width=1):
    """Return a Plotly figure showing communicating cell pairs for an LR pair."""
    with core._cellnest_lock:
        data = core._cellnest_state.get("data")
    if data is None:
        return go.Figure()
    ccc_df = data["ccc_df"]

    # Background dots from CellNEST coordinates file
    coord_path = os.path.join(data["folder"], "metadata", data["dataset_name"],
                              f"coordinates_{data['dataset_name']}.csv")
    try:
        bg_coords = pd.read_csv(coord_path, header=None, names=["x", "y"])
    except Exception:
        bg_coords = core.DATA["df"][["x_centroid", "y_centroid"]].rename(
            columns={"x_centroid": "x", "y_centroid": "y"})
    # Normalise lr_pair to a list early so we can compute degree
    lr_pairs = lr_pair if isinstance(lr_pair, list) else ([lr_pair] if lr_pair else [])

    # Node-degree background coloring
    if node_degree and lr_pairs:
        all_sub = []
        for pair in lr_pairs:
            lig, rec = _cellnest_split_pair(pair, ccc_df)
            all_sub.append(ccc_df[(ccc_df["ligand"] == lig) & (ccc_df["receptor"] == rec)])
        combined = pd.concat(all_sub) if all_sub else pd.DataFrame()
        if not combined.empty:
            from_counts = combined.groupby("from_id").size()
            to_counts   = combined.groupby("to_id").size()
            degree = (from_counts.add(to_counts, fill_value=0)
                      .reindex(range(len(bg_coords)), fill_value=0).values)
            bg_marker = {"size": 3, "color": degree, "colorscale": "Viridis",
                         "showscale": True, "colorbar": {"title": "Degree", "thickness": 10,
                                                          "len": 0.4, "x": 1.02}}
        else:
            bg_marker = {"size": 2, "color": "rgba(150,150,150,0.3)"}
    else:
        bg_marker = {"size": 2, "color": "rgba(150,150,150,0.3)"}

    fig = go.Figure([go.Scattergl(
        x=bg_coords["x"].tolist(), y=(-bg_coords["y"]).tolist(),
        mode="markers", marker=bg_marker,
        name="cells", hoverinfo="skip",
    )])
    pair_palette = qualitative.Plotly
    # Pre-compute color maps for non-lr_pair edge coloring
    if color_by == "ligand":
        _lig_cmap = {l: pair_palette[i % len(pair_palette)]
                     for i, l in enumerate(sorted(ccc_df["ligand"].unique()))}
    elif color_by == "receptor":
        _rec_cmap = {r: pair_palette[i % len(pair_palette)]
                     for i, r in enumerate(sorted(ccc_df["receptor"].unique()))}

    for pi, pair in enumerate(lr_pairs):
        ligand, receptor = _cellnest_split_pair(pair, ccc_df)
        sub = ccc_df[(ccc_df["ligand"] == ligand) & (ccc_df["receptor"] == receptor)]
        sub = _apply_cellnest_filters(sub, f_ligand, f_receptor, f_sender, f_receiver, annot_col)
        sub = sub.nlargest(int(top_n or 100), "attention_score")
        if sub.empty:
            continue

        # Determine edge color for this pair
        if color_by == "lr_pair":
            edge_color_hex = pair_palette[pi % len(pair_palette)]
        elif color_by == "ligand":
            edge_color_hex = _lig_cmap.get(ligand, "#888888")
        elif color_by == "receptor":
            edge_color_hex = _rec_cmap.get(receptor, "#888888")
        else:
            edge_color_hex = None  # attention: handled per-edge below
        # Solid per-pair color for markers (arrows, autocrine rings, sender/receiver dots)
        pair_color_hex = edge_color_hex or pair_palette[pi % len(pair_palette)]

        # ── Edge lines ──────────────────────────────────────────────────────
        if color_by == "attention":
            import plotly.colors as _pc
            scores = sub["attention_score"].values
            s_min, s_max = scores.min(), scores.max()
            s_range = s_max - s_min if s_max > s_min else 1
            viridis = _pc.sample_colorscale("Viridis", list(np.linspace(0, 1, 10)))
            for _, row in sub.iterrows():
                bin_idx = int(min(9, (row["attention_score"] - s_min) / s_range * 10))
                w = (1 + 4 * (row["attention_score"] - s_min) / s_range) * edge_width
                fig.add_trace(go.Scattergl(
                    x=[row["from_x"], row["to_x"], None],
                    y=[-row["from_y"], -row["to_y"], None],
                    mode="lines", line={"color": viridis[bin_idx], "width": w},
                    opacity=0.5, showlegend=False, legendgroup=pair, hoverinfo="skip",
                ))
        elif edge_weight:
            scores = sub["attention_score"].values
            w_min, w_max2 = scores.min(), scores.max()
            w_range = w_max2 - w_min if w_max2 > w_min else 1
            for _, row in sub.iterrows():
                w = (1 + 4 * (row["attention_score"] - w_min) / w_range) * edge_width
                fig.add_trace(go.Scattergl(
                    x=[row["from_x"], row["to_x"], None],
                    y=[-row["from_y"], -row["to_y"], None],
                    mode="lines", line={"color": edge_color_hex, "width": w},
                    opacity=0.5, showlegend=False, legendgroup=pair, hoverinfo="skip",
                ))
        else:
            xs, ys = [], []
            for _, row in sub.iterrows():
                xs += [row["from_x"], row["to_x"], None]
                ys += [-row["from_y"], -row["to_y"], None]
            fig.add_trace(go.Scattergl(
                x=xs, y=ys, mode="lines",
                line={"color": edge_color_hex, "width": edge_width},
                opacity=0.4,
                name=f"edges:{pair}", legendgroup=pair, showlegend=False,
                hoverinfo="skip",
            ))

        # ── Autocrine / self-loop highlight ─────────────────────────────────
        if show_autocrine:
            auto = sub[sub["from_id"] == sub["to_id"]]
            if not auto.empty:
                fig.add_trace(go.Scattergl(
                    x=auto["from_x"].tolist(), y=(-auto["from_y"]).tolist(),
                    mode="markers",
                    marker={"size": 14, "color": "rgba(0,0,0,0)", "symbol": "circle",
                            "line": {"color": pair_color_hex, "width": 2}},
                    name=f"autocrine:{pair}", legendgroup=pair,
                    hovertemplate=f"Autocrine {pair}<extra></extra>",
                ))

        # ── Arrowheads at receiver (to) end ─────────────────────────────────
        dx = (sub["to_x"] - sub["from_x"]).values
        dy = (sub["from_y"] - sub["to_y"]).values   # note: plot y = -real_y
        angles = np.degrees(np.arctan2(dx, dy))     # clockwise from north
        fig.add_trace(go.Scatter(
            x=sub["to_x"].tolist(), y=(-sub["to_y"]).tolist(),
            mode="markers",
            marker={"symbol": "arrow", "size": 8, "color": pair_color_hex,
                    "angle": angles.tolist(), "line": {"width": 0}},
            name=f"arrows:{pair}", legendgroup=pair, showlegend=False,
            hoverinfo="skip",
        ))

        # ── Sender / receiver dots ───────────────────────────────────────────
        from_col = f"from_{color_by}" if color_by != "_default" else None
        to_col   = f"to_{color_by}"   if color_by != "_default" else None

        if from_col and from_col in sub.columns:
            _cats = pd.concat([sub[from_col], sub[to_col]]).dropna().unique()
            _palette2 = qualitative.Plotly
            _cmap = {c: _palette2[i % len(_palette2)] for i, c in enumerate(sorted(_cats, key=str))}
            for cat in sorted(_cats, key=str):
                s_mask = sub[from_col] == cat
                r_mask = sub[to_col]   == cat
                col_hex = _cmap[cat]
                if s_mask.any():
                    fig.add_trace(go.Scattergl(
                        x=sub.loc[s_mask, "from_x"].tolist(),
                        y=(-sub.loc[s_mask, "from_y"]).tolist(),
                        mode="markers", marker={"size": 5, "color": col_hex},
                        name=f"{pair} sender:{cat}", legendgroup=pair,
                        hovertemplate=f"Sender [{cat}] {pair}<extra></extra>",
                    ))
                if r_mask.any():
                    fig.add_trace(go.Scattergl(
                        x=sub.loc[r_mask, "to_x"].tolist(),
                        y=(-sub.loc[r_mask, "to_y"]).tolist(),
                        mode="markers", marker={"size": 5, "color": col_hex,
                                                "symbol": "diamond"},
                        name=f"{pair} receiver:{cat}", legendgroup=pair,
                        hovertemplate=f"Receiver [{cat}] {pair}<extra></extra>",
                    ))
        else:
            fig.add_trace(go.Scattergl(
                x=sub["from_x"].tolist(), y=(-sub["from_y"]).tolist(),
                mode="markers", marker={"size": 5, "color": pair_color_hex},
                name=f"{pair} senders", legendgroup=pair,
                hovertemplate=f"Sender {pair}<extra></extra>",
            ))
            fig.add_trace(go.Scattergl(
                x=sub["to_x"].tolist(), y=(-sub["to_y"]).tolist(),
                mode="markers", marker={"size": 5, "color": pair_color_hex,
                                        "symbol": "diamond"},
                name=f"{pair} receivers", legendgroup=pair,
                hovertemplate=f"Receiver {pair}<extra></extra>",
            ))

    fig.update_layout(
        paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG,
        font={"color": core.TEXT, "size": 10},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        title={"text": f"CCC Spatial: {lr_pair or 'no pair selected'}", "font": {"size": 12}},
        xaxis={"gridcolor": core.BORDER, "showticklabels": False},
        yaxis={"gridcolor": core.BORDER, "showticklabels": False, "scaleanchor": "x"},
        showlegend=color_by != "_default",
        legend={"font": {"size": 9}, "itemsizing": "constant"},
    )
    return fig


def _cellnest_make_circos(lr_pairs, annot_col, top_n=500, color_by="lr_pair",
                          f_ligand=None, f_receptor=None,
                          f_sender=None, f_receiver=None):
    """Build a chord/circos diagram of cell-type interactions for selected LR pairs."""
    with core._cellnest_lock:
        data = core._cellnest_state.get("data")
    if data is None:
        return go.Figure(layout=go.Layout(paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG))
    pairs = lr_pairs if isinstance(lr_pairs, list) else ([lr_pairs] if lr_pairs else [])
    if not pairs:
        return go.Figure(layout=go.Layout(paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG))

    ccc_df = data["ccc_df"]
    from_col = f"from_{annot_col}" if annot_col else None
    to_col   = f"to_{annot_col}"   if annot_col else None

    _no_annot_fig = go.Figure(layout=go.Layout(
        paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG,
        annotations=[{"text": "Select an Annotation column<br>to show the circos plot",
                      "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,
                      "showarrow": False, "font": {"color": core.MUTED, "size": 12}}]))

    if not from_col or from_col not in ccc_df.columns:
        return _no_annot_fig

    # Aggregate interactions by cell-type pair across all selected LR pairs
    palette = qualitative.Plotly
    # Pre-compute color maps for chord coloring
    if color_by == "ligand":
        all_ligs = sorted(ccc_df["ligand"].unique())
        _chord_cmap = {l: palette[i % len(palette)] for i, l in enumerate(all_ligs)}
    elif color_by == "receptor":
        all_recs = sorted(ccc_df["receptor"].unique())
        _chord_cmap = {r: palette[i % len(palette)] for i, r in enumerate(all_recs)}
    else:
        _chord_cmap = {p: palette[i % len(palette)] for i, p in enumerate(pairs)}

    subs = []
    for pair in pairs:
        ligand, receptor = _cellnest_split_pair(pair, ccc_df)
        sub = ccc_df[(ccc_df["ligand"] == ligand) & (ccc_df["receptor"] == receptor)]
        sub = _apply_cellnest_filters(sub, f_ligand, f_receptor, f_sender, f_receiver, annot_col)
        sub = sub.nlargest(int(top_n or 500), "attention_score")
        if not sub.empty:
            subs.append((pair, sub))

    if not subs:
        return _no_annot_fig

    # Build per-pair aggregation: (pair, src_type, tgt_type) → sum weight
    records = []
    for pair, sub in subs:
        ligand, receptor = _cellnest_split_pair(pair, ccc_df)
        agg = (sub.groupby([from_col, to_col])["attention_score"]
                  .sum().reset_index())
        agg.columns = ["source", "target", "weight"]
        agg["pair"] = pair
        agg["ligand"] = ligand
        agg["receptor"] = receptor
        records.append(agg)
    agg_df = pd.concat(records).dropna(subset=["source", "target"])

    types = sorted(set(agg_df["source"]) | set(agg_df["target"]), key=str)
    n = len(types)
    if n == 0:
        return _no_annot_fig

    type_palette = qualitative.Plotly
    type_color = {t: type_palette[i % len(type_palette)] for i, t in enumerate(types)}
    # Node positions: evenly spaced on unit circle; gap between first and last
    gap = 0.05
    arc_span = (2 * np.pi - gap * n) / n
    angles = {}   # centre angle per type
    arc_start = {}
    arc_end   = {}
    theta = gap / 2
    for t in types:
        arc_start[t] = theta
        angles[t]    = theta + arc_span / 2
        arc_end[t]   = theta + arc_span
        theta += arc_span + gap

    fig = go.Figure()

    # Draw arc segments for each cell type
    for t in types:
        th = np.linspace(arc_start[t], arc_end[t], 40)
        r_outer, r_inner = 1.0, 0.85
        x_arc = (np.cos(th) * r_outer).tolist() + (np.cos(th[::-1]) * r_inner).tolist()
        y_arc = (np.sin(th) * r_outer).tolist() + (np.sin(th[::-1]) * r_inner).tolist()
        fig.add_trace(go.Scatter(
            x=x_arc + [x_arc[0]], y=y_arc + [y_arc[0]],
            mode="lines", fill="toself",
            fillcolor=type_color[t], line={"width": 0},
            name=t, hovertemplate=f"{t}<extra></extra>",
        ))
        # Label
        lx = np.cos(angles[t]) * 1.12
        ly = np.sin(angles[t]) * 1.12
        fig.add_annotation(x=lx, y=ly, text=t,
                           font={"size": 8, "color": core.TEXT},
                           showarrow=False, xanchor="center", yanchor="middle")

    # Draw filled ribbon chords (SVG path shapes for constant visual width)
    def _col_rgba(col, alpha=0.55):
        if col.startswith("#"):
            h = col.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"
        if col.startswith("rgb("):
            parts = col[4:-1].split(",")
            return f"rgba({parts[0].strip()},{parts[1].strip()},{parts[2].strip()},{alpha})"
        return col

    w_max = agg_df["weight"].max() or 1
    if color_by == "attention":
        import plotly.colors as _pc
        _viridis = _pc.sample_colorscale("Viridis", list(np.linspace(0, 1, 10)))
    for _, row in agg_df.iterrows():
        src, tgt, w = row["source"], row["target"], row["weight"]
        pair, ligand, receptor = row["pair"], row["ligand"], row["receptor"]
        if src not in angles or tgt not in angles:
            continue
        if color_by == "attention":
            bin_idx = int(min(9, w / w_max * 10))
            chord_col = _viridis[bin_idx]
        elif color_by == "sender_type":
            chord_col = type_color.get(src, "#888888")
        elif color_by == "ligand":
            chord_col = _chord_cmap.get(ligand, "#888888")
        elif color_by == "receptor":
            chord_col = _chord_cmap.get(receptor, "#888888")
        else:
            chord_col = _chord_cmap.get(pair, "#888888")
        # Ribbon half-width proportional to sqrt(weight) — sqrt gives better perceptual contrast
        half_w = max(0.008, 0.18 * np.sqrt(w / w_max))
        a1, a2 = angles[src], angles[tgt]
        r_in = 0.83
        x0a = np.cos(a1 - half_w) * r_in;  y0a = np.sin(a1 - half_w) * r_in
        x0b = np.cos(a1 + half_w) * r_in;  y0b = np.sin(a1 + half_w) * r_in
        x1a = np.cos(a2 - half_w) * r_in;  y1a = np.sin(a2 - half_w) * r_in
        x1b = np.cos(a2 + half_w) * r_in;  y1b = np.sin(a2 + half_w) * r_in
        # SVG cubic bezier ribbon: forward + backward bezier, closed
        path = (f"M {x0a:.4f},{y0a:.4f} "
                f"C 0,0 0,0 {x1a:.4f},{y1a:.4f} "
                f"L {x1b:.4f},{y1b:.4f} "
                f"C 0,0 0,0 {x0b:.4f},{y0b:.4f} Z")
        fig.add_shape(type="path", path=path,
                      fillcolor=_col_rgba(chord_col, 0.55),
                      line_width=0, layer="below")
        # Invisible hover marker at chord midpoint
        mx = (x0a + x0b + x1a + x1b) / 4
        my = (y0a + y0b + y1a + y1b) / 4
        fig.add_trace(go.Scatter(
            x=[mx], y=[my], mode="markers",
            marker={"size": max(8, int(18 * w / w_max)), "opacity": 0, "color": chord_col},
            showlegend=False,
            hovertemplate=f"{pair}: {src} → {tgt} ({w:.0f})<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor=core.DARK_BG, plot_bgcolor=core.DARK_BG,
        font={"color": core.TEXT, "size": 9},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        title={"text": "Cell-Cell Communication (Circos)", "font": {"size": 11}},
        xaxis={"showticklabels": False, "showgrid": False, "zeroline": False,
               "range": [-1.35, 1.35]},
        yaxis={"showticklabels": False, "showgrid": False, "zeroline": False,
               "range": [-1.35, 1.35], "scaleanchor": "x"},
        showlegend=False,
    )
    return fig


def _cellnest_overlay_traces(lr_pair, top_n=100, edge_width=1, color_by="lr_pair", options=None,
                             f_ligand=None, f_receptor=None, f_sender=None, f_receiver=None,
                             annot_col=None):
    """Return list of Plotly trace dicts to overlay on the main spatial plot."""
    with core._cellnest_lock:
        data = core._cellnest_state.get("data")
    if data is None or not lr_pair:
        return []
    ccc_df = data["ccc_df"]
    coords = data.get("coords")
    pairs = lr_pair if isinstance(lr_pair, list) else [lr_pair]
    opts = options or []
    # Apply global filters (ligand/receptor/sender/receiver) before per-pair processing
    ccc_df = _apply_cellnest_filters(ccc_df, f_ligand, f_receptor, f_sender, f_receiver, annot_col)
    palette = qualitative.Plotly
    # Pre-compute color maps
    if color_by == "ligand":
        _lig_cmap = {l: palette[i % len(palette)]
                     for i, l in enumerate(sorted(ccc_df["ligand"].unique()))}
    elif color_by == "receptor":
        _rec_cmap = {r: palette[i % len(palette)]
                     for i, r in enumerate(sorted(ccc_df["receptor"].unique()))}
    if color_by == "attention":
        import plotly.colors as _pc
        _viridis = _pc.sample_colorscale("Viridis", list(np.linspace(0, 1, 10)))

    traces = []
    participating_ids = set()  # track from_id/to_id indices for gray-out exclusion

    for pi, pair in enumerate(pairs):
        ligand, receptor = _cellnest_split_pair(pair, ccc_df)
        sub = ccc_df[(ccc_df["ligand"] == ligand) & (ccc_df["receptor"] == receptor)]
        sub = sub.nlargest(int(top_n or 100), "attention_score")
        if sub.empty:
            continue

        # Collect participating cell indices (for gray-out)
        participating_ids.update(sub["from_id"].astype(int).tolist())
        participating_ids.update(sub["to_id"].astype(int).tolist())

        # Node color always per-pair
        node_col = palette[pi % len(palette)]
        dx = (sub["to_x"] - sub["from_x"]).values
        dy = (sub["from_y"] - sub["to_y"]).values
        arrow_angles = np.degrees(np.arctan2(dx, dy))

        # Edge coloring
        arrow_size = max(6, int(edge_width * 3))
        hover_size = max(10, int(edge_width * 5))
        head_size = max(8, int(edge_width * 5))
        mid_x = ((sub["from_x"] + sub["to_x"]) / 2).tolist()
        mid_y = ((-sub["from_y"] + (-sub["to_y"])) / 2).tolist()

        if color_by == "attention":
            scores = sub["attention_score"].values
            s_min, s_max = scores.min(), scores.max()
            s_range = s_max - s_min if s_max > s_min else 1
            # Collect arrowhead data across all rows for a single marker trace
            head_x, head_y, head_colors, head_angles = [], [], [], []
            for i, (_, row) in enumerate(sub.iterrows()):
                bin_idx = int(min(9, (row["attention_score"] - s_min) / s_range * 10))
                ew = edge_width * (0.5 + 1.5 * (row["attention_score"] - s_min) / s_range) if "edge_weight" in opts else edge_width
                traces.append(go.Scattergl(
                    x=[row["from_x"], row["to_x"], None],
                    y=[-row["from_y"], -row["to_y"], None],
                    mode="lines",
                    line={"color": _viridis[bin_idx], "width": ew},
                    opacity=0.4, name=f"cellnest_lines_{pair}",
                    hoverinfo="skip"))
                head_x.append(row["to_x"])
                head_y.append(-row["to_y"])
                head_colors.append(_viridis[bin_idx])
                head_angles.append(arrow_angles[i])
            if head_x:
                traces.append(go.Scattergl(
                    x=head_x, y=head_y, mode="markers",
                    marker={"symbol": "arrow", "size": head_size,
                            "angle": head_angles, "color": head_colors,
                            "line": {"width": 0}},
                    opacity=0.7, name=f"cellnest_heads_{pair}",
                    hoverinfo="skip"))
        else:
            if color_by == "ligand":
                edge_col = _lig_cmap.get(ligand, node_col)
            elif color_by == "receptor":
                edge_col = _rec_cmap.get(receptor, node_col)
            else:
                edge_col = node_col  # lr_pair
            if "edge_weight" in opts:
                scores = sub["attention_score"].values
                w_min_s, w_max_s = scores.min(), scores.max()
                w_range_s = w_max_s - w_min_s if w_max_s > w_min_s else 1.0
                for i, (_, row) in enumerate(sub.iterrows()):
                    ew = edge_width * (0.5 + 1.5 * (row["attention_score"] - w_min_s) / w_range_s)
                    ah = max(3, int(ew * 3))
                    traces.append(go.Scattergl(
                        x=[row["from_x"], row["to_x"], None],
                        y=[-row["from_y"], -row["to_y"], None],
                        mode="lines",
                        line={"color": edge_col, "width": ew}, opacity=0.4,
                        name=f"cellnest_lines_{pair}", hoverinfo="skip"))
                    traces.append(go.Scattergl(
                        x=[row["to_x"]], y=[-row["to_y"]], mode="markers",
                        marker={"symbol": "arrow", "size": ah,
                                "angle": arrow_angles[i], "color": edge_col,
                                "line": {"width": 0}},
                        opacity=0.7, name=f"cellnest_heads_{pair}", hoverinfo="skip"))
            else:
                xs, ys = [], []
                for _, row in sub.iterrows():
                    xs += [row["from_x"], row["to_x"], None]
                    ys += [-row["from_y"], -row["to_y"], None]
                traces.append(go.Scattergl(
                    x=xs, y=ys, mode="lines",
                    line={"color": edge_col, "width": edge_width}, opacity=0.4,
                    name=f"cellnest_lines_{pair}",
                    hoverinfo="skip"))
                # Arrowheads at the target end, scaled with edge_width
                traces.append(go.Scattergl(
                    x=sub["to_x"].tolist(), y=(-sub["to_y"]).tolist(),
                    mode="markers",
                    marker={"symbol": "arrow", "size": head_size,
                            "angle": arrow_angles.tolist(), "color": edge_col,
                            "line": {"width": 0}},
                    opacity=0.7, name=f"cellnest_heads_{pair}",
                    hoverinfo="skip"))

        # Invisible midpoint markers for reliable hover hit area
        traces.append(go.Scattergl(
            x=mid_x, y=mid_y, mode="markers",
            marker={"size": hover_size, "opacity": 0, "color": node_col, "line": {"width": 0}},
            name=f"cellnest_hover_{pair}",
            hovertemplate=f"{pair}<extra></extra>",
        ))

    # Gray-out: add dark overlay for non-participating cells (prepend so it's beneath edges)
    if "gray_bg" in opts and coords is not None and participating_ids:
        all_idx = np.arange(len(coords))
        mask = ~np.isin(all_idx, list(participating_ids))
        gray_x = coords["x"].values[mask].tolist()
        gray_y = (-coords["y"].values[mask]).tolist()
        traces.insert(0, go.Scattergl(
            x=gray_x, y=gray_y, mode="markers",
            marker={"size": 5, "color": "rgba(28,28,36,0.92)",
                    "line": {"width": 1, "color": "rgba(28,28,36,0.92)"}},
            name="cellnest_gray_bg", hoverinfo="skip",
        ))

    return traces


# ─── CellNEST downstream TF helpers ──────────────────────────────────────────

_CELLNEST_DB_CACHE = os.path.join(core.CACHE_BASE_DIR, "cellnest_db")
_CELLNEST_DB_URLS = {
    "ppi":    "https://raw.githubusercontent.com/schwartzlab-methods/CellNEST/main/database/human_signaling_ppi.csv",
    "tf":     "https://raw.githubusercontent.com/schwartzlab-methods/CellNEST/main/database/human_tf_target.csv",
}
_cellnest_db: dict = {}   # loaded lazily: {"ppi": df, "tf": df}
_cellnest_db_lock = threading.Lock()


def _ensure_cellnest_db():
    """Download and cache PPI + TF databases if not already loaded. Returns (ppi_df, tf_df)."""
    import urllib.request
    with _cellnest_db_lock:
        if "ppi" in _cellnest_db and "tf" in _cellnest_db:
            return _cellnest_db["ppi"], _cellnest_db["tf"]
    os.makedirs(_CELLNEST_DB_CACHE, exist_ok=True)
    result = {}
    for key, url in _CELLNEST_DB_URLS.items():
        local = os.path.join(_CELLNEST_DB_CACHE, os.path.basename(url))
        if not os.path.isfile(local):
            print(f"[CellNEST-DB] Downloading {url} …", flush=True)
            # download to a temp name so an interrupted transfer is not taken as complete
            _tmp = local + ".part"
            try:
                urllib.request.urlretrieve(url, _tmp)
                os.replace(_tmp, local)
            except Exception:
                if os.path.exists(_tmp):
                    os.remove(_tmp)
                raise
            print(f"[CellNEST-DB] Saved to {local}", flush=True)
        result[key] = pd.read_csv(local)
    with _cellnest_db_lock:
        _cellnest_db.update(result)
    return result["ppi"], result["tf"]


def _downstream_tfs(receptor_gene: str, n_hops: int = 2) -> list[str]:
    """Return TF genes downstream of receptor_gene via PPI→TF network."""
    ppi_df, tf_df = _ensure_cellnest_db()
    # Build adjacency: source → set of targets
    ppi_fwd = ppi_df.groupby(ppi_df.columns[0])[ppi_df.columns[1]].apply(set).to_dict()
    # BFS from receptor up to n_hops
    visited = {receptor_gene}
    frontier = {receptor_gene}
    for _ in range(n_hops):
        next_frontier = set()
        for node in frontier:
            next_frontier |= ppi_fwd.get(node, set())
        next_frontier -= visited
        visited |= next_frontier
        frontier = next_frontier
    # TFs = proteins in visited that appear as source in tf_df
    tf_genes = set(tf_df[tf_df.columns[0]].unique())
    downstream = sorted(visited & tf_genes - {receptor_gene})
    return downstream


def _downstream_tf_targets(tf_genes: list[str], panel_genes: set) -> dict:
    """Return {tf: [target_genes_in_panel]} for each TF in tf_genes."""
    _, tf_df = _ensure_cellnest_db()
    src_col = tf_df.columns[0]
    tgt_col = tf_df.columns[1]
    result = {}
    for tf in tf_genes:
        targets = set(tf_df.loc[tf_df[src_col] == tf, tgt_col].tolist()) & panel_genes
        if targets:
            result[tf] = sorted(targets)
    return result
