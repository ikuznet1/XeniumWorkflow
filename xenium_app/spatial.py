"""Spatial analysis (neighbourhood enrichment, co-occurrence) and spatial niches."""

import numpy as np
import pandas as pd

from . import core, jobs, dataview
from . import provenance


def _build_raw_neighbor_matrix(label_arr, neighbor_indices, n_types):
    """Build raw count matrix of type-type neighbor pairs (vectorized)."""
    n_cells, k = neighbor_indices.shape
    src = np.repeat(label_arr, k)
    tgt = label_arr[neighbor_indices.ravel()]
    # Remove self-neighbors (where neighbor_indices[i,j] == i)
    rows_idx = np.repeat(np.arange(n_cells), k)
    valid = neighbor_indices.ravel() != rows_idx
    src, tgt = src[valid], tgt[valid]
    matrix = np.zeros((n_types, n_types), dtype=np.float64)
    np.add.at(matrix, (src, tgt), 1)
    return matrix


def _compute_cooccurrence_from_raw(raw_matrix, freqs):
    """Compute log2 co-occurrence score matrix from pre-built raw counts."""
    total_pairs = raw_matrix.sum()
    if total_pairs == 0:
        return raw_matrix.copy()
    # raw_matrix counts directed (cell, neighbour) pairs, so no factor 2 here
    expected = np.outer(freqs, freqs) * total_pairs
    return np.log2(raw_matrix / (expected + 1e-9) + 1e-9)


def _compute_enrichment_from_raw(raw_matrix, row_counts, freqs):
    """Compute log2 neighborhood enrichment matrix from pre-built raw counts:
    log2( P(neighbour is b | cell is a) / freq_b ), which is 0 under random mixing.
    (row_counts, the number of cells per source type, is kept for the signature only.)"""
    matrix = raw_matrix.copy()
    # Normalize each row by its own number of (cell, neighbour) pairs - not by the
    # number of source cells, which left every value too high by log2(k)
    row_pairs = raw_matrix.sum(axis=1)
    safe_row = np.where(row_pairs > 0, row_pairs, 1)
    matrix /= safe_row[:, None]
    # Normalize cols by global frequency of target type
    safe_freq = np.where(freqs > 0, freqs, 1e-9)
    return np.log2(matrix / safe_freq[None, :] + 1e-9)


def _run_spatial_analysis(label_key, n_neighbors, alt_res):
    """Background thread: compute spatial analysis plots."""
    _gen = jobs._dataset_gen
    try:
        with core._spatial_lock:
            core._spatial_state["status"] = "running"
            core._spatial_state["message"] = "Building neighbor graph…"
            core._spatial_state["result"] = None
        jobs._job_progress("Building neighbor graph…", 0.0)

        # Get cells and coordinates
        if alt_res is not None:
            cells_df = alt_res["cells_df"].copy()
            x_col = "centroid_x" if "centroid_x" in cells_df.columns else "x_centroid"
            y_col = "centroid_y" if "centroid_y" in cells_df.columns else "y_centroid"
        else:
            cells_df = core.DATA["df"].copy()
            x_col, y_col = "x_centroid", "y_centroid"

        coords = cells_df[[x_col, y_col]].values

        # Get labels
        with core._annot_lock:
            labels = core._annot_state.get(label_key)
        if labels is None:
            with core._spatial_lock:
                core._spatial_state["status"] = "error"
                core._spatial_state["message"] = f"No annotation found for key '{label_key}'. Run annotation first."
            return

        # Align labels to cells_df index
        if hasattr(labels, "index"):
            labels = labels.reindex(cells_df.index)
        else:
            if len(labels) != len(cells_df):
                with core._spatial_lock:
                    core._spatial_state["status"] = "error"
                    core._spatial_state["message"] = "Label length mismatch."
                return

        # Drop NaN labels
        valid_mask = ~pd.isna(labels.values if hasattr(labels, "values") else labels)
        labels_arr = labels.values if hasattr(labels, "values") else np.array(labels)
        labels_arr = labels_arr[valid_mask]
        coords_valid = coords[valid_mask]

        cell_types = sorted(set(labels_arr))
        label_to_idx = {ct: i for i, ct in enumerate(cell_types)}
        label_int = np.array([label_to_idx[l] for l in labels_arr], dtype=np.int32)
        n_types = len(cell_types)

        jobs._job_step(core._spatial_state, core._spatial_lock,
                  f"Computing nearest neighbors (k={n_neighbors})…", 1 / 5)

        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(coords_valid)), algorithm="ball_tree")
        nn.fit(coords_valid)
        _, neighbor_indices = nn.kneighbors(coords_valid)
        # Remove self (first neighbor is usually self)
        neighbor_indices = neighbor_indices[:, 1:]

        jobs._job_step(core._spatial_state, core._spatial_lock, "Computing neighbor type matrix…", 2 / 5)

        # Build raw count matrix ONCE (vectorized), reuse for all analyses
        raw_matrix = _build_raw_neighbor_matrix(label_int, neighbor_indices, n_types)
        counts = np.bincount(label_int, minlength=n_types).astype(np.float64)
        freqs = counts / counts.sum()

        jobs._job_step(core._spatial_state, core._spatial_lock, "Computing co-occurrence…", 3 / 5)

        cooccurrence = _compute_cooccurrence_from_raw(raw_matrix, freqs)

        jobs._job_step(core._spatial_state, core._spatial_lock, "Computing enrichment…", 4 / 5)

        enrichment = _compute_enrichment_from_raw(raw_matrix, counts, freqs)

        adjacency = raw_matrix

        # Niche labels and UMAP coords if available
        niche_labels = None
        umap_coords = None
        if "niche" in cells_df.columns:
            niche_labels = cells_df["niche"]
        if "umap_1" in cells_df.columns and "umap_2" in cells_df.columns:
            umap_coords = (cells_df["umap_1"].values, cells_df["umap_2"].values)
        elif core.DATA.get("umap") is not None:
            u = core.DATA["umap"]
            if len(u) == len(cells_df):
                umap_coords = (u[:, 0], u[:, 1])

        with core._spatial_lock:
            jobs._check_dataset_gen(_gen)
            core._spatial_state["status"] = "done"
            core._spatial_state["message"] = f"Done. {n_types} cell types, {len(coords_valid):,} cells."
            core._spatial_state["result"] = {
                "cell_types": cell_types,
                "cooccurrence": cooccurrence,
                "enrichment": enrichment,
                "adjacency": adjacency,
                "freqs": freqs,
                "niche_labels": niche_labels,
                "umap_coords": umap_coords,
            }
            _done_msg = core._spatial_state["message"]
        provenance._prov_record(
            "spatial", seg_source=provenance._prov_labels_source(label_key),
            params={"n_neighbors": n_neighbors, "labels_key": label_key},
            result={"n_cell_types": n_types, "n_cells": int(len(coords_valid)),
                    "n_cells_unlabelled": int(len(coords) - len(coords_valid))},
            message=_done_msg, labels=(label_key, None))
    except Exception as e:
        import traceback
        with core._spatial_lock:
            if not jobs._stale_job(_gen, "error"):
                core._spatial_state["status"] = "error"
                core._spatial_state["message"] = f"Error: {e}\n{traceback.format_exc()}"


_NICHE_NONE = "N/A"     # a cell without a niche (outside the subset niches were computed on)


def _niche_str(values) -> "pd.Series":
    """The niche column as strings for display; cells without a niche (NaN after
    unsubset(), or a store written while some cells had none) become _NICHE_NONE
    instead of the string "nan"."""
    s = values if isinstance(values, pd.Series) else pd.Series(values)
    out = s.astype(object).where(s.notna(), _NICHE_NONE).astype(str)
    return out.where(~out.isin(["nan", "NaN", "None", "<NA>", ""]), _NICHE_NONE)


def _niche_name(niche: str) -> str:
    """Legend entry of a niche value returned by _niche_str()."""
    return "No niche" if niche == _NICHE_NONE else f"Niche {niche}"


# what the niche KMeans was run on (recorded, and read by the methods text)
_NICHE_FEATURES_COMPOSITION = "cell-type composition of the neighbours"
_NICHE_FEATURES_COORDINATES = "normalised spatial coordinates (no cell-type labels available)"


def _run_spatial_niches(n_clusters: int, n_neighbors: int, label_key: str | None,
                        alt_res) -> None:
    """Background thread: compute spatial niches via BuildNicheAssay-style KMeans."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import KMeans
    _gen = jobs._dataset_gen
    try:
        with core._niche_lock:
            core._niche_state["status"] = "running"
            core._niche_state["message"] = "Computing niches…"

        # Get cells_df and coordinates
        if alt_res is not None:
            cells_df = alt_res["cells_df"]
        else:
            cells_df = core.DATA["df"]

        x_col = "centroid_x" if "centroid_x" in cells_df.columns else "x_centroid"
        y_col = "centroid_y" if "centroid_y" in cells_df.columns else "y_centroid"
        coords = cells_df[[x_col, y_col]].values
        n_cells = len(coords)
        k = min(n_neighbors + 1, n_cells)

        jobs._job_step(core._niche_state, core._niche_lock,
                  f"Finding {n_neighbors} neighbors for {n_cells:,} cells…", 0.0)

        nn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", n_jobs=-1)
        nn.fit(coords)
        _, indices = nn.kneighbors(coords)
        # Exclude self (first column)
        neighbor_indices = indices[:, 1:]

        if label_key is not None:
            with core._annot_lock:
                labels_series = core._annot_state.get(label_key)
        else:
            labels_series = None

        if labels_series is not None:
            # Align labels to cells_df index
            aligned = labels_series.reindex(cells_df.index)
            if aligned.isna().all():
                # label ids are stored as str; a reseg cells_df index may be int
                aligned = pd.Series(labels_series.reindex(cells_df.index.astype(str)).values,
                                    index=cells_df.index)
            if aligned.isna().all():
                raise ValueError("The selected annotation has no labels for the cells of the "
                                 "active segmentation")
            aligned = aligned.fillna("Unknown")
            cell_types = sorted(aligned.unique())
            ct_to_idx = {ct: i for i, ct in enumerate(cell_types)}
            label_arr = aligned.map(ct_to_idx).values  # int array

            jobs._job_step(core._niche_state, core._niche_lock,
                      "Building neighborhood composition matrix…", 1 / 3)

            n_types = len(cell_types)
            composition = np.zeros((n_cells, n_types), dtype=np.float32)
            k = neighbor_indices.shape[1]
            flat_labels = label_arr[neighbor_indices.ravel()]
            valid = flat_labels >= 0
            src = np.repeat(np.arange(n_cells), k)[valid]
            tgt = flat_labels[valid]
            np.add.at(composition, (src, tgt), 1)
            # Normalize to proportions
            row_sums = composition.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            composition /= row_sums
        else:
            # Fall back: KMeans on raw spatial coordinates (normalized)
            composition = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)
            cell_types = None

        jobs._job_step(core._niche_state, core._niche_lock,
                  f"KMeans clustering into {n_clusters} niches…", 2 / 3)

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        niche_labels = km.fit_predict(composition).astype(str)

        # Write niche column to cells_df
        jobs._check_dataset_gen(_gen)
        jobs._job_shield()   # last cancellation point: the niche column is written below
        # The labels are assigned by cell id to the frame that is live NOW: subset() /
        # unsubset() may have replaced DATA["df"] since this job took its snapshot, and
        # writing the snapshot back would undo that (a full frame beside a subset-sized
        # df_to_expr). Cells the job did not see get no niche.
        niche = pd.Series(niche_labels, index=cells_df.index, dtype=object)
        if not niche.index.is_unique:
            niche = niche[~niche.index.duplicated()]

        def _with_niche(df):
            out = df.copy()
            out["niche"] = niche.reindex(out.index).values
            return out

        def _niche_into_full(full_df):
            # subset() active: unsubset() restores this frame, so it gets the labels
            # too (by cell id; cells outside the subset have no niche)
            if full_df is not None:
                full_df["niche"] = niche.reindex(full_df.index).values

        if alt_res is not None:
            alt_res["cells_df"] = _with_niche(alt_res["cells_df"])
            _niche_into_full(alt_res.get("_cells_df_original"))
        else:
            if "_df_original" in core.DATA:
                _niche_into_full(core.DATA["_df_original"])
                core.DATA["df"] = _with_niche(core.DATA["df"])   # the subset slice itself is not written to
            else:
                core.DATA["df"]["niche"] = niche.reindex(core.DATA["df"].index).values
            dataview._umap_df_cache.clear()   # cached dropna() copies lack the new column

        # Print proportion table if cell types available
        if cell_types is not None and labels_series is not None:
            import pandas as _pd_inner
            niche_arr = km.labels_
            rows = {}
            for niche_id in range(n_clusters):
                mask = niche_arr == niche_id
                if mask.sum() == 0:
                    continue
                ct_counts = np.bincount(label_arr[mask], minlength=len(cell_types))
                rows[f"Niche {niche_id}"] = ct_counts / max(ct_counts.sum(), 1)
            prop_df = _pd_inner.DataFrame(rows, index=cell_types).T
            print("\n── Spatial Niche × Cell Type proportions ──", flush=True)
            print(prop_df.to_string(float_format=lambda v: f"{v:.2f}"), flush=True)
            print("", flush=True)

        with core._niche_lock:
            jobs._check_dataset_gen(_gen)
            core._niche_state["status"] = "done"
            _niche_msg = (f"Done — {n_clusters} niches computed for {n_cells:,} cells"
                          + ("" if labels_series is not None else
                             " from the normalised cell coordinates: no cell-type labels were available, so "
                             "these are spatial blocks, not composition niches (run Cell Type Annotation "
                             "first for niches of the neighbourhood composition)"))
            core._niche_state["message"] = _niche_msg
        provenance._prov_record(
            "niches", seg_source=("xenium" if alt_res is None else provenance._prov_run_source(
                str(alt_res.get("source") or "resegmentation"), alt_res)),
            params={"n_clusters": n_clusters, "n_neighbors": n_neighbors, "labels_key": label_key,
                    "features": (_NICHE_FEATURES_COMPOSITION if labels_series is not None
                                 else _NICHE_FEATURES_COORDINATES)},
            result={"n_niches": int(n_clusters), "n_cells": int(n_cells)},
            message=_niche_msg, labels=(label_key, None))
    except jobs._JobCancelled:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        with core._niche_lock:
            if not jobs._stale_job(_gen, "error"):
                core._niche_state["status"] = "error"
                core._niche_state["message"] = str(exc)
