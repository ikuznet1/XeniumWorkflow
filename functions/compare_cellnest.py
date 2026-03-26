#!/usr/bin/env python3
"""
compare_cellnest.py — Cross-condition statistical comparison of CellNEST CCC

Compares cell-cell communication (LR pairs) across experimental conditions using
permutation tests, bootstrap CIs, and Wilcoxon rank-sum. Supports cell-type-
specific breakdown when annotation CSVs are available.

Usage
-----
  python compare_cellnest.py \\
      --cellnest_dir functions/cellnest_output \\
      --samples name1:condA name2:condA name3:condB name4:condB \\
      --output_dir results/ \\
      --annotation_dir functions/cellnest_input \\
      --annotation_col cell_type

  # Single-sample metrics only (no comparison):
  python compare_cellnest.py \\
      --cellnest_dir functions/cellnest_output \\
      --samples Xenium_resegmented_imputed_final_1343:test \\
      --output_dir /tmp/cellnest_test

Importable
----------
  from compare_cellnest import compare_cellnest
  compare_cellnest(
      cellnest_dir="functions/cellnest_output",
      samples_config=["name1:condA", "name2:condB"],
      output_dir="results/",
  )

Dependencies
------------
  pandas, numpy, scipy, matplotlib, seaborn
"""

import argparse
import itertools
import os
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# ---------------------------------------------------------------------------
# Step 1: Data loading
# ---------------------------------------------------------------------------

def load_sample_ccc(cellnest_dir, name, use_all, attn_thresh,
                    annot_dir=None, annot_col="cell_type"):
    """Load CCC edges + metadata for a single sample.

    Returns dict with keys: edges (DataFrame), n_cells (int), name (str).
    """
    suffix = "allCCC" if use_all else "top20percent"
    ccc_path = os.path.join(
        cellnest_dir, "output", name, f"CellNEST_{name}_{suffix}.csv"
    )
    barcode_path = os.path.join(
        cellnest_dir, "metadata", name, f"cell_barcode_{name}.csv"
    )

    if not os.path.isfile(ccc_path):
        raise FileNotFoundError(f"CCC file not found: {ccc_path}")
    if not os.path.isfile(barcode_path):
        raise FileNotFoundError(f"Barcode file not found: {barcode_path}")

    edges = pd.read_csv(ccc_path)
    barcodes = pd.read_csv(barcode_path, header=None, names=["barcode"])
    n_cells = len(barcodes)

    # Apply attention threshold
    if attn_thresh is not None and attn_thresh > 0:
        edges = edges[edges["attention_score"] >= attn_thresh].copy()

    # Build lr_pair column
    edges["lr_pair"] = edges["ligand"] + "-" + edges["receptor"]

    # Join annotation if available
    if annot_dir is not None:
        annot_path = os.path.join(annot_dir, f"{name}_annotation.csv")
        if os.path.isfile(annot_path):
            annot = pd.read_csv(annot_path, index_col=0)
            col = annot_col if annot_col in annot.columns else annot.columns[0]
            barcode_to_type = annot[col].to_dict()
            edges["from_type"] = edges["from_cell"].map(barcode_to_type)
            edges["to_type"] = edges["to_cell"].map(barcode_to_type)
        else:
            print(f"  Warning: annotation not found: {annot_path}")

    return {"edges": edges, "n_cells": n_cells, "name": name}


def load_all_samples(cellnest_dir, samples_config, use_all, attn_thresh,
                     annot_dir=None, annot_col="cell_type"):
    """Load all samples. Returns (edges_df, sample_meta dict).

    samples_config: list of "name:condition" strings.
    """
    all_edges = []
    sample_meta = {}  # name -> {condition, n_cells}

    for entry in samples_config:
        parts = entry.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid sample spec '{entry}'. Expected 'name:condition'."
            )
        name, condition = parts[0].strip(), parts[1].strip()
        print(f"  Loading {name} (condition={condition})...")

        data = load_sample_ccc(
            cellnest_dir, name, use_all, attn_thresh, annot_dir, annot_col
        )
        data["edges"]["sample"] = name
        data["edges"]["condition"] = condition
        all_edges.append(data["edges"])
        sample_meta[name] = {"condition": condition, "n_cells": data["n_cells"]}

    edges_df = pd.concat(all_edges, ignore_index=True)
    return edges_df, sample_meta


# ---------------------------------------------------------------------------
# Step 2: Per-sample LR-pair metrics
# ---------------------------------------------------------------------------

def compute_sample_metrics(edges_df, sample_meta):
    """Compute per-sample, per-LR-pair metrics. Returns long-format DataFrame."""
    total_per_sample = edges_df.groupby("sample").size()
    agg = edges_df.groupby(["sample", "lr_pair"]).agg(
        n_edges=("attention_score", "size"),
        mean_attention=("attention_score", "mean"),
        n_unique_senders=("from_cell", "nunique"),
    ).reset_index()
    if agg.empty:
        return pd.DataFrame()

    cond_map = {n: m["condition"] for n, m in sample_meta.items()}
    ncells_map = {n: m["n_cells"] for n, m in sample_meta.items()}

    agg["condition"] = agg["sample"].map(cond_map)
    agg["normalized_freq"] = agg["n_edges"] / agg["sample"].map(total_per_sample)
    agg["sender_fraction"] = agg["n_unique_senders"] / agg["sample"].map(ncells_map)
    agg["presence"] = 1

    return agg[["sample", "condition", "lr_pair", "n_edges",
                "normalized_freq", "mean_attention", "sender_fraction", "presence"]]


# ---------------------------------------------------------------------------
# Step 3: Cell-type-specific metrics
# ---------------------------------------------------------------------------

def compute_celltype_metrics(edges_df, sample_meta, min_edges_per_triplet=10):
    """Compute per-sample, per (LR pair, sender_type, receiver_type) metrics."""
    if "from_type" not in edges_df.columns or "to_type" not in edges_df.columns:
        print("  Skipping cell-type metrics: no annotation columns found.")
        return pd.DataFrame()

    adf = edges_df.dropna(subset=["from_type", "to_type"])
    if adf.empty:
        return pd.DataFrame()

    total_per_sample = adf.groupby("sample").size()
    agg = adf.groupby(["sample", "lr_pair", "from_type", "to_type"]).agg(
        n_edges=("attention_score", "size"),
        mean_attention=("attention_score", "mean"),
        n_unique_senders=("from_cell", "nunique"),
    ).reset_index()

    # Filter by minimum edge count
    agg = agg[agg["n_edges"] >= min_edges_per_triplet]
    if agg.empty:
        return pd.DataFrame()

    cond_map = {n: m["condition"] for n, m in sample_meta.items()}
    ncells_map = {n: m["n_cells"] for n, m in sample_meta.items()}

    agg["condition"] = agg["sample"].map(cond_map)
    agg["normalized_freq"] = agg["n_edges"] / agg["sample"].map(total_per_sample)
    agg["sender_fraction"] = agg["n_unique_senders"] / agg["sample"].map(ncells_map)
    agg["presence"] = 1

    return agg.rename(columns={"from_type": "sender_type", "to_type": "receiver_type"})[
        ["sample", "condition", "lr_pair", "sender_type", "receiver_type",
         "n_edges", "normalized_freq", "mean_attention", "sender_fraction", "presence"]]


# ---------------------------------------------------------------------------
# Step 4: Statistical comparisons
# ---------------------------------------------------------------------------

def _build_wide_matrix(metrics_df, metric, group_cols, sample_names):
    """Pivot metrics to (n_features, n_samples) matrix.

    group_cols: columns that define a feature (e.g., ["lr_pair"]).
    Returns: matrix (ndarray), feature_labels (list), sample_order (list).
    """
    pivot = metrics_df.pivot_table(
        index=group_cols, columns="sample", values=metric, fill_value=0.0
    )
    # Ensure all samples present
    for s in sample_names:
        if s not in pivot.columns:
            pivot[s] = 0.0
    pivot = pivot[sample_names]  # reorder
    return pivot.values, pivot.index.tolist(), sample_names


def permutation_test_vectorized(matrix, idx_a, idx_b, n_perm=10000, rng=None):
    """Vectorized permutation test on (n_features, n_samples) matrix.

    Uses exact enumeration when C(n_total, n_a) <= n_perm (e.g. 20 for n=3v3),
    otherwise falls back to Monte Carlo.

    Returns observed_diffs (n_features,) and pvals (n_features,).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_a, n_b = len(idx_a), len(idx_b)
    combined = np.array(idx_a + idx_b)
    n_total = n_a + n_b

    vals = matrix[:, combined]  # (n_features, n_total)
    obs_diff = vals[:, :n_a].mean(axis=1) - vals[:, n_a:].mean(axis=1)
    abs_obs = np.abs(obs_diff)

    n_combos = comb(n_total, n_a)

    if n_combos <= n_perm:
        # Exact permutation test — enumerate all partitions
        from itertools import combinations as _combs
        count_extreme = np.zeros(matrix.shape[0], dtype=np.int64)
        all_idx = set(range(n_total))
        for combo in _combs(range(n_total), n_a):
            rest = sorted(all_idx - set(combo))
            perm_diff = vals[:, list(combo)].mean(axis=1) - vals[:, rest].mean(axis=1)
            count_extreme += (np.abs(perm_diff) >= abs_obs)
        pvals = count_extreme / n_combos
        # Clamp minimum p to 1/n_combos (avoid p=0 from float precision)
        pvals = np.maximum(pvals, 1.0 / n_combos)
    else:
        # Monte Carlo permutation test
        count_extreme = np.zeros(matrix.shape[0], dtype=np.int64)
        for _ in range(n_perm):
            perm = rng.permutation(n_total)
            perm_diff = vals[:, perm[:n_a]].mean(axis=1) - vals[:, perm[n_a:]].mean(axis=1)
            count_extreme += (np.abs(perm_diff) >= abs_obs)
        pvals = (count_extreme + 1) / (n_perm + 1)  # +1 to avoid p=0

    return obs_diff, pvals


def bootstrap_ci(matrix, idx_a, idx_b, n_boot=2000, alpha=0.05, rng=None):
    """Bootstrap 95% CI on mean difference. Returns (mean_diff, ci_lo, ci_hi)."""
    if rng is None:
        rng = np.random.default_rng(42)

    n_a, n_b = len(idx_a), len(idx_b)
    combined = np.array(idx_a + idx_b)
    vals = matrix[:, combined]
    vals_a = vals[:, :n_a]
    vals_b = vals[:, n_a:]

    boot_diffs = np.zeros((matrix.shape[0], n_boot))
    for i in range(n_boot):
        ba = rng.choice(n_a, size=n_a, replace=True)
        bb = rng.choice(n_b, size=n_b, replace=True)
        boot_diffs[:, i] = vals_a[:, ba].mean(axis=1) - vals_b[:, bb].mean(axis=1)

    mean_diff = boot_diffs.mean(axis=1)
    ci_lo = np.percentile(boot_diffs, 100 * alpha / 2, axis=1)
    ci_hi = np.percentile(boot_diffs, 100 * (1 - alpha / 2), axis=1)
    return mean_diff, ci_lo, ci_hi


def _compute_log2fc(mean_a, mean_b, eps=None):
    """log2 fold change: log2((mean_B + eps) / (mean_A + eps))."""
    if eps is None:
        all_vals = np.concatenate([mean_a, mean_b])
        nonzero = all_vals[all_vals > 0]
        eps = nonzero.min() / 10 if len(nonzero) > 0 else 1e-10
    return np.log2((mean_b + eps) / (mean_a + eps))


def _bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    if n == 0:
        return np.array([])
    order = np.argsort(pvals)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    fdr = pvals * n / ranks
    # Enforce monotonicity (step-up)
    fdr_sorted = fdr[order]
    for i in range(n - 2, -1, -1):
        fdr_sorted[i] = min(fdr_sorted[i], fdr_sorted[i + 1])
    fdr[order] = fdr_sorted
    return np.clip(fdr, 0, 1)


def run_pairwise_comparison(metrics_df, cond_a, cond_b, metric="normalized_freq",
                            group_cols=None, n_perm=10000, n_boot=2000,
                            fdr_alpha=0.05, min_presence=2):
    """Run full statistical comparison between two conditions.

    Returns DataFrame with one row per feature (LR pair or triplet).
    """
    if group_cols is None:
        group_cols = ["lr_pair"]

    samples_a = sorted(
        metrics_df[metrics_df["condition"] == cond_a]["sample"].unique()
    )
    samples_b = sorted(
        metrics_df[metrics_df["condition"] == cond_b]["sample"].unique()
    )
    all_samples = samples_a + samples_b

    if len(samples_a) == 0 or len(samples_b) == 0:
        print(f"  Warning: empty condition group ({cond_a}: {len(samples_a)}, "
              f"{cond_b}: {len(samples_b)}). Skipping.")
        return pd.DataFrame()

    # Build wide matrix
    matrix, feature_labels, _ = _build_wide_matrix(
        metrics_df, metric, group_cols, all_samples
    )
    idx_a = list(range(len(samples_a)))
    idx_b = list(range(len(samples_a), len(all_samples)))

    # Filter by min_presence
    presence_a = (matrix[:, idx_a] > 0).sum(axis=1)
    presence_b = (matrix[:, idx_b] > 0).sum(axis=1)
    keep = (presence_a >= min(min_presence, len(samples_a))) & \
           (presence_b >= min(min_presence, len(samples_b)))

    if keep.sum() == 0:
        print(f"  No features pass min_presence filter for {cond_a} vs {cond_b}.")
        return pd.DataFrame()

    matrix_filt = matrix[keep]
    labels_filt = [feature_labels[i] for i in range(len(feature_labels)) if keep[i]]

    rng = np.random.default_rng(42)

    # Permutation test
    obs_diff, perm_pvals = permutation_test_vectorized(
        matrix_filt, idx_a, idx_b, n_perm=n_perm, rng=rng
    )

    # Bootstrap CI
    _, ci_lo, ci_hi = bootstrap_ci(
        matrix_filt, idx_a, idx_b, n_boot=n_boot, alpha=fdr_alpha, rng=rng
    )

    # Mean per group
    mean_a = matrix_filt[:, idx_a].mean(axis=1)
    mean_b = matrix_filt[:, idx_b].mean(axis=1)

    # log2 fold change
    log2fc = _compute_log2fc(mean_a, mean_b)

    # Wilcoxon rank-sum (per feature)
    wilcox_pvals = np.ones(len(labels_filt))
    for i in range(len(labels_filt)):
        va = matrix_filt[i, idx_a]
        vb = matrix_filt[i, idx_b]
        if np.all(va == vb):
            continue
        try:
            _, p = stats.ranksums(va, vb)
            wilcox_pvals[i] = p
        except Exception:
            pass

    # FDR
    perm_fdr = _bh_fdr(perm_pvals)
    wilcox_fdr = _bh_fdr(wilcox_pvals)

    # Build results
    results = []
    for i, label in enumerate(labels_filt):
        row = {}
        if isinstance(label, tuple):
            for j, col in enumerate(group_cols):
                row[col] = label[j]
        else:
            row[group_cols[0]] = label
        row.update({
            f"mean_{cond_a}": mean_a[i],
            f"mean_{cond_b}": mean_b[i],
            "log2FC": log2fc[i],
            "obs_diff": obs_diff[i],
            "perm_pval": perm_pvals[i],
            "perm_FDR": perm_fdr[i],
            "bootstrap_ci_lo": ci_lo[i],
            "bootstrap_ci_hi": ci_hi[i],
            "wilcox_pval": wilcox_pvals[i],
            "wilcox_FDR": wilcox_fdr[i],
            f"presence_{cond_a}": int(
                (matrix_filt[i, idx_a] > 0).sum()
            ),
            f"presence_{cond_b}": int(
                (matrix_filt[i, idx_b] > 0).sum()
            ),
        })
        results.append(row)

    return pd.DataFrame(results).sort_values("perm_pval")


def run_all_comparisons(metrics_df, conditions, metric="normalized_freq",
                        group_cols=None, n_perm=10000, n_boot=2000,
                        fdr_alpha=0.05, min_presence=2):
    """Run all pairwise comparisons. Returns dict of (condA, condB) -> DataFrame."""
    results = {}
    pairs = list(itertools.combinations(sorted(conditions), 2))

    for cond_a, cond_b in pairs:
        print(f"  Comparing {cond_a} vs {cond_b}...")
        res = run_pairwise_comparison(
            metrics_df, cond_a, cond_b, metric=metric,
            group_cols=group_cols, n_perm=n_perm, n_boot=n_boot,
            fdr_alpha=fdr_alpha, min_presence=min_presence,
        )
        results[(cond_a, cond_b)] = res

    # Combined FDR across all comparisons
    if len(pairs) > 1:
        all_pvals = []
        all_indices = []  # (pair_key, row_idx)
        for key, df in results.items():
            if df.empty:
                continue
            for idx in df.index:
                all_pvals.append(df.loc[idx, "perm_pval"])
                all_indices.append((key, idx))
        if all_pvals:
            combined_fdr = _bh_fdr(np.array(all_pvals))
            for i, (key, idx) in enumerate(all_indices):
                results[key].loc[idx, "combined_FDR"] = combined_fdr[i]

    return results


def run_celltype_comparisons(ct_metrics_df, cond_a, cond_b, n_perm=10000,
                             n_boot=2000, fdr_alpha=0.05, min_presence=2):
    """Run cell-type-specific comparison for one condition pair."""
    if ct_metrics_df.empty:
        return pd.DataFrame()

    return run_pairwise_comparison(
        ct_metrics_df, cond_a, cond_b,
        metric="normalized_freq",
        group_cols=["lr_pair", "sender_type", "receiver_type"],
        n_perm=n_perm, n_boot=n_boot,
        fdr_alpha=fdr_alpha, min_presence=min_presence,
    )


# ---------------------------------------------------------------------------
# Step 7: Figures
# ---------------------------------------------------------------------------

def plot_volcano(results_df, output_path, cond_a, cond_b, top_n_label=15):
    """Volcano plot: log2FC vs -log10(perm_pval)."""
    if results_df.empty:
        return
    df = results_df.copy()
    df["-log10p"] = -np.log10(df["perm_pval"].clip(lower=1e-300))

    fig, ax = plt.subplots(figsize=(10, 7))

    # Color by significance
    sig = df["perm_FDR"] < 0.05
    ax.scatter(
        df.loc[~sig, "log2FC"], df.loc[~sig, "-log10p"],
        c="gray", alpha=0.5, s=30, label="NS",
    )
    ax.scatter(
        df.loc[sig, "log2FC"], df.loc[sig, "-log10p"],
        c="crimson", alpha=0.7, s=40, label="FDR < 0.05",
    )

    # Label top hits
    top = df.nsmallest(top_n_label, "perm_pval")
    for _, row in top.iterrows():
        ax.annotate(
            row["lr_pair"],
            (row["log2FC"], row["-log10p"]),
            fontsize=7, ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )

    ax.set_xlabel(f"log2 FC ({cond_b} / {cond_a})")
    ax.set_ylabel("-log10(permutation p-value)")
    ax.set_title(f"Volcano: {cond_a} vs {cond_b}")
    ax.legend()
    ax.axhline(-np.log10(0.05), ls="--", color="gray", lw=0.8)
    ax.axvline(0, ls="--", color="gray", lw=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {output_path}")


def plot_heatmap(metrics_df, output_path, top_n=30):
    """Heatmap: LR pairs × samples, grouped by condition."""
    if metrics_df.empty:
        return

    # Pick top LR pairs by variance across samples
    pivot = metrics_df.pivot_table(
        index="lr_pair", columns="sample", values="normalized_freq", fill_value=0
    )
    if len(pivot) == 0:
        return
    variances = pivot.var(axis=1).sort_values(ascending=False)
    top_lr = variances.head(top_n).index.tolist()
    pivot = pivot.loc[top_lr]

    # Order columns by condition
    sample_cond = metrics_df[["sample", "condition"]].drop_duplicates()
    sample_order = sample_cond.sort_values("condition")["sample"].tolist()
    pivot = pivot[[s for s in sample_order if s in pivot.columns]]

    # Condition color bar
    cond_map = dict(zip(sample_cond["sample"], sample_cond["condition"]))
    conditions = sorted(set(cond_map.values()))
    palette = sns.color_palette("Set2", len(conditions))
    cond_colors = {c: palette[i] for i, c in enumerate(conditions)}
    col_colors = [cond_colors[cond_map[s]] for s in pivot.columns]

    g = sns.clustermap(
        pivot, col_cluster=False, row_cluster=True,
        cmap="YlOrRd", figsize=(max(8, len(pivot.columns) * 0.8), max(6, top_n * 0.3)),
        col_colors=col_colors, linewidths=0.5,
        xticklabels=True, yticklabels=True,
        dendrogram_ratio=(0.1, 0.05),
    )
    g.ax_heatmap.set_xlabel("Sample")
    g.ax_heatmap.set_ylabel("LR Pair")
    g.fig.suptitle("Normalized Frequency: LR Pairs × Samples", y=1.02)

    # Legend for conditions
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=cond_colors[c], label=c) for c in conditions]
    g.ax_heatmap.legend(
        handles=legend_handles, loc="upper left",
        bbox_to_anchor=(1.05, 1), title="Condition",
    )

    g.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    print(f"    Saved {output_path}")


def plot_condition_dotplot(metrics_df, results_dict, output_path, top_n=30):
    """Dotplot: LR pairs × conditions (size=freq, color=mean_attention)."""
    if metrics_df.empty:
        return

    # Collect significant LR pairs from all comparisons
    sig_lr = set()
    for key, df in results_dict.items():
        if df.empty:
            continue
        sig = df[df["perm_FDR"] < 0.05]
        sig_lr.update(sig["lr_pair"].tolist())

    if not sig_lr:
        # Fall back to top by variance
        agg = metrics_df.groupby("lr_pair")["normalized_freq"].var().nlargest(top_n)
        sig_lr = set(agg.index)

    # Aggregate per condition
    cond_agg = metrics_df[metrics_df["lr_pair"].isin(sig_lr)].groupby(
        ["lr_pair", "condition"]
    ).agg(
        mean_freq=("normalized_freq", "mean"),
        mean_attn=("mean_attention", "mean"),
    ).reset_index()

    lr_order = (
        cond_agg.groupby("lr_pair")["mean_freq"]
        .max().nlargest(top_n).index.tolist()
    )
    cond_agg = cond_agg[cond_agg["lr_pair"].isin(lr_order)]

    if cond_agg.empty:
        return

    conditions = sorted(cond_agg["condition"].unique())
    fig, ax = plt.subplots(
        figsize=(max(6, len(conditions) * 2), max(6, len(lr_order) * 0.35))
    )

    # Map conditions to x positions
    cond_pos = {c: i for i, c in enumerate(conditions)}

    scatter = ax.scatter(
        cond_agg["condition"].map(cond_pos),
        cond_agg["lr_pair"].map({lr: i for i, lr in enumerate(lr_order)}),
        s=cond_agg["mean_freq"] / cond_agg["mean_freq"].max() * 300 + 10,
        c=cond_agg["mean_attn"],
        cmap="viridis", edgecolors="black", linewidths=0.5, alpha=0.8,
    )

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions)
    ax.set_yticks(range(len(lr_order)))
    ax.set_yticklabels(lr_order, fontsize=8)
    ax.set_xlabel("Condition")
    ax.set_ylabel("LR Pair")
    ax.set_title("Condition Dotplot (size=freq, color=attention)")
    plt.colorbar(scatter, ax=ax, label="Mean Attention")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {output_path}")


def plot_barplot_top_hits(metrics_df, results_dict, output_path, top_n=15):
    """Bar plot of top significant LR pairs with replicate dots."""
    # Collect top significant across all comparisons
    all_sig = []
    for key, df in results_dict.items():
        if df.empty:
            continue
        sig = df[df["perm_FDR"] < 0.05].copy()
        sig["comparison"] = f"{key[0]}_vs_{key[1]}"
        all_sig.append(sig)

    if not all_sig:
        return

    sig_df = pd.concat(all_sig)
    top_lr = sig_df.nsmallest(top_n, "perm_pval")["lr_pair"].unique()

    plot_data = metrics_df[metrics_df["lr_pair"].isin(top_lr)]
    if plot_data.empty:
        return

    conditions = sorted(plot_data["condition"].unique())
    fig, ax = plt.subplots(figsize=(max(10, top_n * 0.8), 6))

    # Bar = mean, dots = replicates
    bar_agg = plot_data.groupby(["lr_pair", "condition"])["normalized_freq"].mean().reset_index()
    bar_agg = bar_agg[bar_agg["lr_pair"].isin(top_lr)]

    x_pos = np.arange(len(top_lr))
    width = 0.8 / len(conditions)
    palette = sns.color_palette("Set2", len(conditions))

    for j, cond in enumerate(conditions):
        cond_data = bar_agg[bar_agg["condition"] == cond]
        # Ensure all LR pairs present
        heights = []
        for lr in top_lr:
            match = cond_data[cond_data["lr_pair"] == lr]["normalized_freq"]
            heights.append(match.values[0] if len(match) > 0 else 0)

        offset = (j - len(conditions) / 2 + 0.5) * width
        bars = ax.bar(
            x_pos + offset, heights, width,
            label=cond, color=palette[j], alpha=0.7, edgecolor="black", linewidth=0.5,
        )

        # Overlay replicate dots
        rep_data = plot_data[plot_data["condition"] == cond]
        for i, lr in enumerate(top_lr):
            reps = rep_data[rep_data["lr_pair"] == lr]["normalized_freq"]
            ax.scatter(
                [x_pos[i] + offset] * len(reps), reps,
                color="black", s=15, zorder=5, alpha=0.7,
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_lr, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Normalized Frequency")
    ax.set_title("Top Significant LR Pairs")
    ax.legend(title="Condition")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {output_path}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def compare_cellnest(cellnest_dir, samples_config, output_dir,
                     annotation_dir=None, annotation_col="cell_type",
                     use_all=False, attention_threshold=None,
                     n_permutations=10000, fdr_alpha=0.05,
                     min_presence=2, top_n_plot=30):
    """Main orchestrator for cross-condition CellNEST comparison."""

    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ---- Step 1: Load ----
    print("[1/7] Loading CCC data from all samples...")
    edges_df, sample_meta = load_all_samples(
        cellnest_dir, samples_config, use_all, attention_threshold,
        annotation_dir, annotation_col,
    )
    n_samples = len(sample_meta)
    n_edges = len(edges_df)
    n_lr = edges_df["lr_pair"].nunique()
    print(f"  Loaded {n_edges:,} edges across {n_samples} samples, {n_lr} LR pairs.")

    conditions = sorted(set(m["condition"] for m in sample_meta.values()))
    print(f"  Conditions: {conditions}")
    for cond in conditions:
        names = [n for n, m in sample_meta.items() if m["condition"] == cond]
        print(f"    {cond}: {names}")

    # ---- Step 2: Sample metrics ----
    print("[2/7] Computing per-sample LR-pair metrics...")
    metrics_df = compute_sample_metrics(edges_df, sample_meta)
    print(f"  {len(metrics_df)} metric rows.")

    # ---- Step 3: Cell-type metrics ----
    print("[3/7] Computing cell-type-specific metrics...")
    ct_metrics_df = compute_celltype_metrics(
        edges_df, sample_meta, min_edges_per_triplet=10
    )
    if not ct_metrics_df.empty:
        print(f"  {len(ct_metrics_df)} cell-type metric rows.")
    else:
        print("  No cell-type metrics (annotation missing or insufficient edges).")

    # ---- Step 4: Pairwise comparisons ----
    print("[4/7] Running pairwise statistical comparisons...")
    results_dict = {}
    if len(conditions) >= 2:
        results_dict = run_all_comparisons(
            metrics_df, conditions,
            metric="normalized_freq",
            n_perm=n_permutations, n_boot=2000,
            fdr_alpha=fdr_alpha, min_presence=min_presence,
        )
        for key, df in results_dict.items():
            n_sig = (df["perm_FDR"] < 0.05).sum() if not df.empty else 0
            print(f"  {key[0]} vs {key[1]}: {len(df)} tested, {n_sig} significant (FDR<0.05).")
    else:
        print("  Only 1 condition — skipping comparisons (metrics-only mode).")

    # ---- Step 5: Cell-type comparisons ----
    print("[5/7] Running cell-type-specific comparisons...")
    ct_results_dict = {}
    if len(conditions) >= 2 and not ct_metrics_df.empty:
        pairs = list(itertools.combinations(conditions, 2))
        for cond_a, cond_b in pairs:
            print(f"  Cell-type comparison: {cond_a} vs {cond_b}...")
            ct_res = run_celltype_comparisons(
                ct_metrics_df, cond_a, cond_b,
                n_perm=n_permutations, n_boot=2000,
                fdr_alpha=fdr_alpha, min_presence=min_presence,
            )
            ct_results_dict[(cond_a, cond_b)] = ct_res
            if not ct_res.empty:
                n_sig = (ct_res["perm_FDR"] < 0.05).sum()
                print(f"    {len(ct_res)} triplets tested, {n_sig} significant.")
            else:
                print(f"    No testable triplets.")
    else:
        print("  Skipped (single condition or no annotations).")

    # ---- Step 6: Write tables ----
    print("[6/7] Writing result tables...")
    metrics_path = os.path.join(output_dir, "sample_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  {metrics_path}")

    if not ct_metrics_df.empty:
        ct_path = os.path.join(output_dir, "celltype_metrics.csv")
        ct_metrics_df.to_csv(ct_path, index=False)
        print(f"  {ct_path}")

    all_summary_rows = []
    for (cond_a, cond_b), df in results_dict.items():
        if df.empty:
            continue
        fname = f"{cond_a}_vs_{cond_b}_freq.csv"
        fpath = os.path.join(output_dir, fname)
        df.to_csv(fpath, index=False)
        print(f"  {fpath}")
        # Collect significant for summary
        sig = df[df["perm_FDR"] < 0.05].copy()
        if not sig.empty:
            sig["comparison"] = f"{cond_a}_vs_{cond_b}"
            all_summary_rows.append(sig)

    for (cond_a, cond_b), df in ct_results_dict.items():
        if df.empty:
            continue
        fname = f"{cond_a}_vs_{cond_b}_celltype.csv"
        fpath = os.path.join(output_dir, fname)
        df.to_csv(fpath, index=False)
        print(f"  {fpath}")

    if all_summary_rows:
        summary = pd.concat(all_summary_rows, ignore_index=True)
        summary_path = os.path.join(output_dir, "summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"  {summary_path}")

    # ---- Step 7: Figures ----
    print("[7/7] Generating figures...")
    for (cond_a, cond_b), df in results_dict.items():
        if df.empty:
            continue
        plot_volcano(
            df, os.path.join(fig_dir, f"volcano_{cond_a}_vs_{cond_b}.pdf"),
            cond_a, cond_b, top_n_label=15,
        )

    plot_heatmap(
        metrics_df, os.path.join(fig_dir, "heatmap_samples.pdf"),
        top_n=top_n_plot,
    )

    if results_dict:
        plot_condition_dotplot(
            metrics_df, results_dict,
            os.path.join(fig_dir, "dotplot_conditions.pdf"),
            top_n=top_n_plot,
        )
        plot_barplot_top_hits(
            metrics_df, results_dict,
            os.path.join(fig_dir, "barplot_top_hits.pdf"),
            top_n=15,
        )

    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-condition statistical comparison of CellNEST CCC results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Single sample (metrics only, no comparison):
  python compare_cellnest.py \\
      --cellnest_dir functions/cellnest_output \\
      --samples sample1:condA \\
      --output_dir /tmp/test

  # Two conditions, 3 replicates each:
  python compare_cellnest.py \\
      --cellnest_dir functions/cellnest_output \\
      --samples s1:ctrl s2:ctrl s3:ctrl s4:treat s5:treat s6:treat \\
      --annotation_dir functions/cellnest_input \\
      --output_dir results/
""",
    )
    parser.add_argument(
        "--cellnest_dir", required=True,
        help="Root CellNEST directory (contains output/ and metadata/ subdirs)",
    )
    parser.add_argument(
        "--samples", required=True, nargs="+",
        help="Sample specs as name:condition pairs (e.g. sample1:ctrl sample2:treat)",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to write results",
    )
    parser.add_argument(
        "--annotation_dir", default=None,
        help="Directory containing {name}_annotation.csv files",
    )
    parser.add_argument(
        "--annotation_col", default="cell_type",
        help="Column name in annotation CSV (default: cell_type)",
    )
    parser.add_argument(
        "--use_all", action="store_true",
        help="Use allCCC.csv instead of top20percent.csv",
    )
    parser.add_argument(
        "--attention_threshold", type=float, default=None,
        help="Uniform attention score cutoff (edges below are dropped)",
    )
    parser.add_argument(
        "--n_permutations", type=int, default=10000,
        help="Number of permutations for permutation test (default: 10000)",
    )
    parser.add_argument(
        "--fdr_alpha", type=float, default=0.05,
        help="FDR significance threshold (default: 0.05)",
    )
    parser.add_argument(
        "--min_presence", type=int, default=2,
        help="Min samples per condition with LR pair present to test (default: 2)",
    )
    parser.add_argument(
        "--top_n_plot", type=int, default=30,
        help="Number of top features to show in plots (default: 30)",
    )

    args = parser.parse_args()

    compare_cellnest(
        cellnest_dir=args.cellnest_dir,
        samples_config=args.samples,
        output_dir=args.output_dir,
        annotation_dir=args.annotation_dir,
        annotation_col=args.annotation_col,
        use_all=args.use_all,
        attention_threshold=args.attention_threshold,
        n_permutations=args.n_permutations,
        fdr_alpha=args.fdr_alpha,
        min_presence=args.min_presence,
        top_n_plot=args.top_n_plot,
    )


if __name__ == "__main__":
    main()
