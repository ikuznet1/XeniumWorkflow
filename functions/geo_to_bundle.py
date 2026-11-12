#!/usr/bin/env python3
"""Assemble a Xenium-Explorer-loadable bundle from a GEO-style per-section deposit.

GEO deposits of Xenium data usually hold a few processed files per section rather
than the full instrument output, for example (GSE345643):

    GSM..._<section>_cell_feature_matrix.h5          10x-format counts (cells x genes)
    GSM..._<section>_cells_stats.csv.gz              per-cell table: cell_id, x_centroid, y_centroid, ...
    GSM..._<section>_transcripts.parquet.gz          Xenium transcripts table
    GSM..._<section>_ambient_corrected_matrix.h5     optional
    GSM..._<section>_morphology.ome.tif.gz           optional, large

xenium_explorer.py needs at least experiment.xenium, cells.parquet and
cell_feature_matrix.h5. This script writes those (plus transcripts.parquet when
present) into an output directory that can be opened with

    python xenium_explorer.py <out_dir>

Usage:
    # from files already on disk (a directory holding one section's files)
    python functions/geo_to_bundle.py --input_dir geo_files/ --out_dir bundle/

    # download one sample's supplementary files from GEO first
    python functions/geo_to_bundle.py --gsm GSM10012173 --out_dir bundle/ [--with_image]

Every extra column of the per-cell table (cell types, patient, group, ...) is kept in
cells.parquet and so appears in the app's cell table. The bundle has no cell
boundaries, clustering or UMAP unless the deposit provides them; the app starts
without them.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import shutil
import sys
import urllib.request

_UA = {"User-Agent": "geo_to_bundle/1.0"}
_ROLES = {
    "matrix": re.compile(r"(?<!corrected_)cell_feature_matrix\.h5$"),
    "cells": re.compile(r"cells?(_stats)?\.(csv|tsv|parquet)(\.gz)?$"),
    "transcripts": re.compile(r"transcripts\.parquet(\.gz)?$"),
    "corrected": re.compile(r"corrected_matrix\.h5$"),
    "image": re.compile(r"morphology(_focus)?\.ome\.tiff?(\.gz)?$"),
}


def _find(input_dir: str) -> dict:
    """Map each role to the single file in *input_dir* that matches it."""
    found = {}
    for path in sorted(glob.glob(os.path.join(input_dir, "*"))):
        name = os.path.basename(path)
        for role, pat in _ROLES.items():
            if pat.search(name):
                if role in found:
                    sys.exit(f"Two files look like the {role} file ({os.path.basename(found[role])}, {name}); "
                             f"put one section per directory.")
                found[role] = path
    return found


def _download_gsm(gsm: str, dest: str, with_image: bool) -> None:
    """Fetch a GEO sample's supplementary files over HTTPS into *dest*."""
    if not re.fullmatch(r"GSM\d+", gsm):
        sys.exit(f"Not a GEO sample accession: {gsm}")
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm[:-3]}nnn/{gsm}/suppl/"
    listing = urllib.request.urlopen(urllib.request.Request(base, headers=_UA), timeout=60).read().decode()
    names = re.findall(r'<a href="([^"?/][^"]*)">', listing)
    if not names:
        sys.exit(f"No supplementary files listed at {base}")
    os.makedirs(dest, exist_ok=True)
    for name in names:
        if _ROLES["image"].search(name) and not with_image:
            print(f"  skipping {name} (pass --with_image to download it)")
            continue
        if not any(p.search(name) for p in _ROLES.values()):
            continue
        out = os.path.join(dest, name)
        if os.path.exists(out):
            print(f"  have {name}")
            continue
        print(f"  downloading {name} ...", flush=True)
        tmp = out + ".part"
        with urllib.request.urlopen(urllib.request.Request(base + name, headers=_UA), timeout=120) as r, \
                open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=4 * 1024 * 1024)
        os.replace(tmp, out)


def _gunzip_to(src: str, dst: str) -> None:
    if src.endswith(".gz"):
        tmp = dst + ".part"
        with gzip.open(src, "rb") as f, open(tmp, "wb") as o:
            shutil.copyfileobj(f, o, length=4 * 1024 * 1024)
        os.replace(tmp, dst)
    else:
        shutil.copyfile(src, dst)


def build_bundle(input_dir: str, out_dir: str, pixel_size: float = 0.2125, name: str | None = None) -> str:
    import h5py
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    files = _find(input_dir)
    for role in ("matrix", "cells"):
        if role not in files:
            sys.exit(f"No {role} file found in {input_dir} (looked for {_ROLES[role].pattern}).")
    os.makedirs(out_dir, exist_ok=True)

    # ── per-cell table ────────────────────────────────────────────────────────
    cpath = files["cells"]
    if ".parquet" in os.path.basename(cpath):
        cells = pd.read_parquet(cpath)
    else:
        cells = pd.read_csv(cpath, sep="\t" if ".tsv" in os.path.basename(cpath) else ",")
    need = {"cell_id", "x_centroid", "y_centroid"}
    if not need <= set(cells.columns):
        sys.exit(f"{os.path.basename(cpath)} must have columns {sorted(need)}; found {list(cells.columns)}")
    cells["cell_id"] = cells["cell_id"].astype(str)
    if cells["cell_id"].duplicated().any():
        sys.exit("cell_id values are not unique in the per-cell table.")

    # ── counts: only gene-expression features, columns = cells ────────────────
    with h5py.File(files["matrix"], "r") as h:
        barcodes = [b.decode() for b in h["matrix/barcodes"][:]]
        shape = tuple(int(v) for v in h["matrix/shape"][:])
        mat = sp.csc_matrix((h["matrix/data"][:], h["matrix/indices"][:], h["matrix/indptr"][:]),
                            shape=shape)
        ftype = ([t.decode() for t in h["matrix/features/feature_type"][:]]
                 if "matrix/features/feature_type" in h else None)
    if shape[1] != len(barcodes):
        sys.exit(f"Matrix has {shape[1]} columns but {len(barcodes)} barcodes.")
    gene_rows = (np.array([t == "Gene Expression" for t in ftype]) if ftype
                 else np.ones(shape[0], dtype=bool))
    totals = pd.Series(np.asarray(mat[gene_rows].sum(axis=0)).ravel(), index=barcodes)

    missing = ~cells["cell_id"].isin(totals.index)
    if missing.all():
        sys.exit("No cell_id in the per-cell table matches a matrix barcode.")
    if missing.any():
        print(f"  note: {int(missing.sum()):,} cells in the table have no matrix column (kept, zero counts)")
    extra = len(set(barcodes) - set(cells["cell_id"]))
    if extra:
        print(f"  note: {extra:,} matrix barcodes are not in the per-cell table (ignored by the app)")

    if "transcript_counts" not in cells.columns:
        cells["transcript_counts"] = totals.reindex(cells["cell_id"]).fillna(0).values
    if "total_counts" not in cells.columns:
        cells["total_counts"] = cells["transcript_counts"]
    cells.to_parquet(os.path.join(out_dir, "cells.parquet"), index=False)
    shutil.copyfile(files["matrix"], os.path.join(out_dir, "cell_feature_matrix.h5"))

    # ── optional files ────────────────────────────────────────────────────────
    if "transcripts" in files:
        _gunzip_to(files["transcripts"], os.path.join(out_dir, "transcripts.parquet"))
    if "corrected" in files:
        shutil.copyfile(files["corrected"], os.path.join(out_dir, "ambient_corrected_matrix.h5"))
    images = {}
    if "image" in files:
        img_name = ("morphology_focus.ome.tif" if "focus" in os.path.basename(files["image"])
                    else "morphology.ome.tif")
        _gunzip_to(files["image"], os.path.join(out_dir, img_name))
        images = {("morphology_focus_filepath" if "focus" in img_name else "morphology_filepath"): img_name}

    meta = {
        "major_version": 0, "minor_version": 0, "patch_version": 0,
        "run_name": name or os.path.basename(os.path.abspath(input_dir)),
        "region_name": name or os.path.basename(os.path.abspath(input_dir)),
        "num_cells": int(len(cells)),
        "pixel_size": float(pixel_size),
        "analysis_sw_version": "assembled by geo_to_bundle.py",
        "images": images,
    }
    with open(os.path.join(out_dir, "experiment.xenium"), "w") as f:
        json.dump(meta, f, indent=1)

    print(f"  bundle written: {out_dir}")
    print(f"    {len(cells):,} cells, {int(gene_rows.sum()):,} genes, "
          f"{'transcripts, ' if 'transcripts' in files else ''}"
          f"{len(cells.columns) - 3} per-cell columns")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_dir", help="Directory holding one section's deposited files")
    src.add_argument("--gsm", help="GEO sample accession to download first (e.g. GSM10012173)")
    ap.add_argument("--out_dir", required=True, help="Bundle directory to create")
    ap.add_argument("--download_dir", help="Where to keep downloaded files (default: <out_dir>_geo_files)")
    ap.add_argument("--with_image", action="store_true", help="Also download the morphology image (large)")
    ap.add_argument("--pixel_size", type=float, default=0.2125, help="Image pixel size in µm (default 0.2125)")
    ap.add_argument("--name", help="Run / region name recorded in experiment.xenium")
    args = ap.parse_args()

    input_dir = args.input_dir
    if args.gsm:
        input_dir = args.download_dir or (args.out_dir.rstrip("/") + "_geo_files")
        _download_gsm(args.gsm, input_dir, args.with_image)
    build_bundle(input_dir, args.out_dir, pixel_size=args.pixel_size, name=args.name or args.gsm)


if __name__ == "__main__":
    main()
