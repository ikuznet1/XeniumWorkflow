"""Image-based resegmentation: cells from the morphology images (Cellpose or a built-in
classical watershed), run on sopa image patches; transcripts are then assigned to the
polygons and the run is stored and loaded like a Baysor / Proseg run.

A segmenter is a callable ``method(image) -> mask``: image of shape (C, Y, X) holding the
chosen channels (the nuclear channel LAST), mask of shape (Y, X) with 0 for background and
one positive integer per cell. Two backends build such a callable (_iseg_make_method):

* "classical": smoothing, Otsu threshold, distance-transform watershed on the nuclear
  channel, optional expansion. numpy / scipy only (scikit-image's watershed is used when
  installed, a nearest-seed split otherwise). A simple baseline that needs no model.
* "cellpose": sopa.segmentation.methods.cellpose_patch, used only when cellpose is
  installed AND the model's weights are already on disk; the weights file is passed by
  its path, which is the one way cellpose never tries to download anything.
"""

import os
import json
import hashlib
import time as _time
import numpy as np
import pandas as pd
from dash import dcc, html

from . import provenance
from . import core, jobs, caching, dataview, clustering, reseg, regions, ui


_ISEG_TOOL = "imgseg"
_ISEG_PATCH_UM = 500.0         # image patch width (µm)
_ISEG_OVERLAP_UM = 30.0        # overlap between image patches (µm); cells cut by a patch edge are merged
_ISEG_DIAMETER_UM = 8.0        # expected nucleus diameter
_ISEG_EXPAND_UM = 5.0          # classical backend: every nucleus grows by this much (0: nuclei only)
_ISEG_MIN_TX = 5               # cells with fewer assigned gene transcripts are dropped
_ISEG_TX_TILE_UM = 1000.0      # transcripts are read and assigned one tile at a time
_ISEG_BOUNDARY_COLOR = "#ff6b9d"

# cellpose model name → weights file in cellpose's model directory (cellpose < 4: the
# "torch_0" suffix of the first-generation models; "cpsam" is the cellpose ≥ 4 model)
_ISEG_CELLPOSE_FILES = {"nuclei": "nucleitorch_0", "cyto3": "cyto3", "cyto2": "cyto2torch_0",
                        "cyto": "cytotorch_0", "cpsam": "cpsam"}
_ISEG_CELLPOSE_HINT = (
    "Cellpose is not installed. Install it with: pip install 'sopa[cellpose]' (or pip install "
    "cellpose). Its model weights are not part of the package: the first run downloads them from "
    "the Cellpose project (cellpose.org) into ~/.cellpose/models. This app never starts that "
    "download - run Cellpose once yourself, e.g. python -c \"from cellpose import models; "
    "models.CellposeModel(model_type='nuclei')\".")

_ISEG_NOTE = (
    "Cells are segmented on the morphology image (the nuclear channel; Cellpose can also take a "
    "membrane / cytoplasm channel), patch by patch through sopa; cells cut by a patch edge are "
    "merged. Gene transcripts (Q ≥ 20, gene features only - the filter of the cell-feature "
    "matrix) are then assigned to the polygon they fall in, in 2D. Built-in classical backend: "
    "Gaussian smoothing, Otsu threshold per patch, distance-transform watershed, optional "
    "expansion of every nucleus by a fixed distance - a simple baseline that needs no model, not "
    "a state-of-the-art segmenter: it merges touching nuclei it cannot split, misses dim nuclei, "
    "and the expansion is a geometric guess, not a membrane. Cellpose (Stringer et al., Nat "
    "Methods 2021) is used only when it is installed and its weights are already on this "
    "machine.")


# ─── Backends ─────────────────────────────────────────────────────────────────
def _iseg_cellpose_status() -> dict:
    """What is known about Cellpose on this machine, without downloading anything:
    {available, version, models: {name: weights path}, model_dir, message}.
    available = the package imports AND at least one known model's weights file exists."""
    out = {"available": False, "version": None, "models": {}, "model_dir": None, "message": ""}
    try:
        import cellpose
        from cellpose import models as _cp_models
    except Exception as exc:                       # not installed, or a broken install
        out["message"] = _ISEG_CELLPOSE_HINT + (
            "" if isinstance(exc, ImportError) else f" (import failed: {str(exc)[:120]})")
        return out
    out["version"] = str(getattr(cellpose, "version", "") or getattr(cellpose, "__version__", "") or "?")
    model_dir = str(getattr(_cp_models, "MODEL_DIR", os.path.join(os.path.expanduser("~"), ".cellpose", "models")))
    out["model_dir"] = model_dir
    try:
        major = int(str(out["version"]).split(".")[0])
    except ValueError:
        major = 3
    names = ["cpsam"] if major >= 4 else [n for n in _ISEG_CELLPOSE_FILES if n != "cpsam"]
    for name in names:
        path = os.path.join(model_dir, _ISEG_CELLPOSE_FILES[name])
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            out["models"][name] = path
    if out["models"]:
        out["available"] = True
        out["message"] = (f"Cellpose {out['version']} with local weights: {', '.join(out['models'])}. "
                          f"Nothing is downloaded.")
    else:
        out["message"] = (
            f"Cellpose {out['version']} is installed but none of its model weights "
            f"({', '.join(_ISEG_CELLPOSE_FILES[n] for n in names)}) are in its model directory. The first "
            f"Cellpose run downloads them from the Cellpose project (cellpose.org); this app never starts "
            f"that download - run Cellpose once yourself, then come back.")
    return out


def _iseg_backend_options() -> list:
    """Dropdown options of the backends; Cellpose is listed but disabled when unusable."""
    st = _iseg_cellpose_status()
    opts = [{"label": "Classical watershed (built-in baseline, no model)", "value": "classical"}]
    if st["available"]:
        for name in st["models"]:
            opts.append({"label": f"Cellpose {st['version']} - {name} (local weights)", "value": f"cellpose:{name}"})
    else:
        opts.append({"label": "Cellpose (not available - see the note below)", "value": "cellpose", "disabled": True})
    return opts


def _iseg_otsu(values) -> float:
    """Otsu's threshold of an array (256-bin histogram)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    lo, hi = float(v.min()), float(v.max())
    if not hi > lo:
        return hi
    hist, edges = np.histogram(v, bins=256, range=(lo, hi))
    mid = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist).astype(np.float64)
    w1 = w0[-1] - w0
    s0 = np.cumsum(hist * mid)
    m0 = s0 / np.maximum(w0, 1.0)
    m1 = (s0[-1] - s0) / np.maximum(w1, 1.0)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(mid[int(np.argmax(between[:-1]))]) if len(between) > 1 else lo


def _iseg_expand_labels(mask, distance: float) -> np.ndarray:
    """Grow every label by up to *distance* pixels without overlap: a background pixel
    takes the label of the nearest labelled pixel when that is within the distance."""
    from scipy import ndimage as ndi
    mask = np.asarray(mask)
    if distance <= 0 or not mask.any():
        return mask
    d, (iy, ix) = ndi.distance_transform_edt(mask == 0, return_indices=True)
    out = mask.copy()
    grow = (d <= float(distance)) & (mask == 0)
    out[grow] = mask[iy[grow], ix[grow]]
    return out


def _iseg_classical_mask(image, diameter_px: float, expand_px: float = 0.0, thr_scale: float = 1.0,
                         min_area_px=None) -> np.ndarray:
    """The classical backend on one patch. image: (C, Y, X), the nuclear channel last (only
    that channel is used). Returns an int32 label mask (0 = background).

    1. Gaussian smoothing (σ = diameter / 10, at least 1 px).
    2. Otsu threshold of the smoothed patch × *thr_scale*. A patch whose two Otsu classes
       are not clearly apart (foreground mean < 1.5 × background mean, or less than 3
       robust SDs above it) is taken as empty.
    3. Holes filled; Euclidean distance transform of the foreground, smoothed; its local
       maxima (window ≈ 0.6 × diameter) seed a watershed of the inverted distance within
       the foreground - touching round nuclei are split where the distance map necks.
    4. Objects below *min_area_px* (default: 20 % of a disc of the given diameter) removed.
    5. Optional expansion of every label by *expand_px* (nearest label wins, no overlap)."""
    from scipy import ndimage as ndi
    img = np.asarray(image)
    img = img[-1] if img.ndim == 3 else img
    img = img.astype(np.float32, copy=False)
    d = float(max(2.0, diameter_px))
    empty = np.zeros(img.shape, dtype=np.int32)
    if img.size == 0 or not np.isfinite(img).any():
        return empty
    sm = ndi.gaussian_filter(img, sigma=max(1.0, d / 10.0))
    thr = _iseg_otsu(sm[:: max(1, sm.shape[0] // 2048), :: max(1, sm.shape[1] // 2048)])
    sub = sm[:: max(1, sm.shape[0] // 1024), :: max(1, sm.shape[1] // 1024)]
    bg, fgv = sub[sub <= thr], sub[sub > thr]
    if not len(bg) or not len(fgv):
        return empty
    m_bg, m_fg = float(bg.mean()), float(fgv.mean())
    sd_bg = 1.4826 * float(np.median(np.abs(bg - np.median(bg)))) + 1e-9
    if m_fg < 1.5 * max(m_bg, 1e-9) or (m_fg - m_bg) < 3.0 * sd_bg:
        return empty
    fg = ndi.binary_fill_holes(sm > thr * float(thr_scale))
    if not fg.any():
        return empty
    dist = ndi.distance_transform_edt(fg)
    dist = ndi.gaussian_filter(dist.astype(np.float32), sigma=max(1.0, d / 8.0))
    win = int(max(3, round(0.6 * d))) | 1
    peaks = (ndi.maximum_filter(dist, size=win, mode="nearest") == dist) & fg & (dist > 0.1 * d)
    markers, n_mark = ndi.label(peaks, structure=np.ones((3, 3), dtype=bool))
    if not n_mark:
        return empty
    try:
        from skimage.segmentation import watershed
        lab = watershed(-dist, markers, mask=fg)
    except ImportError:
        # scipy only: every foreground pixel goes to the nearest seed (for convex nuclei
        # close to the watershed of the distance map)
        _d, (sy, sx) = ndi.distance_transform_edt(markers == 0, return_indices=True)
        lab = np.where(fg, markers[sy, sx], 0)
    lab = np.asarray(lab, dtype=np.int32)
    min_area = float(0.2 * np.pi * (d / 2.0) ** 2 if min_area_px is None else min_area_px)
    area = np.bincount(lab.ravel())
    small = np.flatnonzero(area < min_area)
    small = small[small > 0]
    if len(small):
        lab[np.isin(lab, small)] = 0
    keep = np.unique(lab)
    keep = keep[keep > 0]
    if not len(keep):
        return empty
    lut = np.zeros(int(lab.max()) + 1, dtype=np.int32)
    lut[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)
    lab = lut[lab]
    return _iseg_expand_labels(lab, expand_px) if expand_px and expand_px > 0 else lab


def _iseg_cellpose_model(backend, n_channels: int = 1, models=None) -> "str | None":
    """The Cellpose model a backend value resolves to ("cellpose:<model>", or "cellpose"
    alone: nuclei for one channel when it is on disk, else the first model found)."""
    kind, _, model = str(backend or "").partition(":")
    if kind != "cellpose":
        return None
    if model:
        return model
    models = _iseg_cellpose_status()["models"] if models is None else models
    if not models:
        return None
    return "nuclei" if "nuclei" in models and n_channels == 1 else next(iter(models))


def _iseg_uses_second_channel(backend, models=None) -> bool:
    """Whether the backend reads a membrane / cytoplasm channel beside the nuclear one: the
    Cellpose cyto-type models (and any model that is not a nuclear one) do; the nuclei model
    and the classical backend segment the nuclear channel alone. With two channels sopa
    names them [cyto, nucleus] and Cellpose takes the first as its primary channel - for the
    nuclei model that would be the membrane stain."""
    model = _iseg_cellpose_model(backend, n_channels=2, models=models)
    return bool(model) and not str(model).lower().startswith("nuclei")


def _iseg_make_method(backend: str, diameter_px: float, n_channels: int = 1, expand_px: float = 0.0,
                      thr_scale: float = 1.0, gpu: bool = False, flow_threshold: float = 0.4,
                      cellprob_threshold: float = 0.0) -> tuple:
    """(method, info) for a backend: "classical", or "cellpose:<model>".

    method(image (C, Y, X)) -> label mask (Y, X). info: {backend, label, model, weights,
    preprocess: {clip_limit, gaussian_sigma} for sopa's StainingSegmentation}. Raises
    RuntimeError with the install note when the Cellpose backend cannot run here; never
    downloads anything."""
    kind, _, model = str(backend or "classical").partition(":")
    if kind == "classical":
        def method(image):
            return _iseg_classical_mask(image, diameter_px, expand_px=expand_px, thr_scale=thr_scale)
        return method, {"backend": "classical", "label": "classical watershed", "model": None, "weights": None,
                        "preprocess": {"clip_limit": 0.0, "gaussian_sigma": 0.0}}
    if kind != "cellpose":
        raise RuntimeError(f"Unknown image-segmentation backend '{backend}'.")
    st = _iseg_cellpose_status()
    if not st["available"]:
        raise RuntimeError(st["message"])
    model = model or ("nuclei" if "nuclei" in st["models"] and n_channels == 1 else next(iter(st["models"])))
    weights = st["models"].get(model)
    # the file is checked again right before use: an absent path would make cellpose fall
    # back to downloading its default model
    if not weights or not os.path.isfile(weights):
        raise RuntimeError(f"The weights of the Cellpose model '{model}' are not on this machine. " + st["message"])
    from sopa.segmentation.methods import cellpose_patch
    # sopa convention: one channel = the nuclear channel; two = [cytoplasm, nucleus]
    names = ["nucleus"] if n_channels == 1 else ["cyto", "nucleus"]
    base = cellpose_patch(diameter=float(diameter_px), channels=names, pretrained_model=weights, gpu=bool(gpu),
                          flow_threshold=float(flow_threshold), cellprob_threshold=float(cellprob_threshold))

    def method(image):
        if not os.path.isfile(weights):
            raise RuntimeError("Cellpose weights disappeared during the run.")
        mask = np.asarray(base(image))
        return _iseg_expand_labels(mask, expand_px) if expand_px and expand_px > 0 else mask
    return method, {"backend": "cellpose", "label": f"Cellpose {st['version']} ({model})", "model": model,
                    "weights": os.path.basename(weights), "preprocess": {"clip_limit": 0.2, "gaussian_sigma": 1.0}}


# ─── Image, patches, polygons ─────────────────────────────────────────────────
def _iseg_image(sdata=None) -> tuple:
    """(image key, full-resolution image DataArray (c, y, x)) of the dataset's SpatialData
    store, (None, None) when it is not loaded or has no image."""
    if sdata is None:
        with core._sdata_lock:
            sdata = core._sdata_state.get("sdata")
    if sdata is None or not len(getattr(sdata, "images", {})):
        return None, None
    try:
        from sopa.utils import get_spatial_image
        img, key = None, None
        got = get_spatial_image(sdata, return_key=True)
        key, img = got
        return key, img
    except Exception as exc:
        print(f"  Image segmentation: no usable image in the SpatialData store ({exc})", flush=True)
        return None, None


def _iseg_channels(sdata=None) -> list:
    """Channel names of the morphology image in the SpatialData store ([] when not loaded)."""
    _key, img = _iseg_image(sdata)
    if img is None:
        return []
    return [str(c) for c in img.coords["c"].values]


def _iseg_default_channel(channels) -> "str | None":
    for c in channels:
        if "dapi" in c.lower():
            return c
    return channels[0] if channels else None


def _iseg_patches(sdata, image_key, width_px: float, overlap_px: float, region_px=None):
    """A private in-memory SpatialData holding the image (and the tissue ROI when the store
    has one) with sopa image patches; the dataset's own store is not modified. region_px:
    (x0, y0, x1, y1) in image pixels - patches are clipped to it. Returns (mini sdata,
    number of patches)."""
    import sopa
    import spatialdata as sd
    from sopa.constants import SopaKeys
    from shapely.geometry import box
    shapes = {}
    if SopaKeys.ROI in sdata.shapes:
        shapes[SopaKeys.ROI] = sdata.shapes[SopaKeys.ROI]
    mini = sd.SpatialData(images={image_key: sdata.images[image_key]}, shapes=shapes)
    sopa.make_image_patches(mini, patch_width=int(width_px), patch_overlap=int(overlap_px), image_key=image_key)
    gdf = mini.shapes[SopaKeys.PATCHES]
    if region_px is not None:
        from spatialdata.models import ShapesModel
        from spatialdata.transformations import get_transformation
        rb = box(*region_px)
        g2 = gdf.copy()
        g2.geometry = g2.geometry.intersection(rb)
        g2 = g2[~g2.geometry.is_empty & (g2.geometry.area > 0)].reset_index(drop=True)
        if not len(g2):
            raise ValueError("The chosen region does not overlap the image (or the tissue ROI).")
        tf = get_transformation(gdf, get_all=True)
        del mini.shapes[SopaKeys.PATCHES]
        import geopandas as gpd
        mini.shapes[SopaKeys.PATCHES] = ShapesModel.parse(gpd.GeoDataFrame(geometry=list(g2.geometry)),
                                                          transformations=tf)
        gdf = mini.shapes[SopaKeys.PATCHES]
    return mini, int(len(gdf))


def _iseg_ring(geom):
    """Exterior ring (x array, y array) of a polygon; the largest part of a multi-polygon."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    if geom.geom_type != "Polygon":
        return None
    xs, ys = geom.exterior.coords.xy
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _iseg_assign_transcripts(polys, tx_path: str, qv_min=20, tile_um: float = _ISEG_TX_TILE_UM,
                             bounds=None, progress=None) -> tuple:
    """Assign the gene transcripts of transcripts.parquet to the polygons they fall in.

    polys: shapely polygons in µm. The file is read one *tile_um* tile at a time over
    *bounds* (default: the polygons' bounding box) with the filters of the cell-feature
    matrix (Q ≥ qv_min, gene features only); a transcript inside two overlapping polygons
    goes to the first. Returns (counts DataFrame [cell (polygon index), gene, n],
    transcripts read in the bounds, transcripts assigned)."""
    import shapely
    import pyarrow as pa
    import pyarrow.parquet as pq
    polys = list(polys)
    cols_out = ["cell", "gene", "n"]
    if not polys:
        return pd.DataFrame(columns=cols_out), 0, 0
    tree = shapely.STRtree(polys)
    if bounds is None:
        bounds = shapely.total_bounds(np.asarray(polys, dtype=object))
    x0, y0, x1, y1 = [float(v) for v in bounds]
    schema = pq.read_schema(tx_path)
    filters = reseg._tx_gene_filters(schema.names, qv_min)
    ft = schema.field("feature_name").type
    dict_cols = (["feature_name"] if (pa.types.is_string(ft) or pa.types.is_large_string(ft)
                                      or pa.types.is_binary(ft)) else None)
    need_name_filter = "is_gene" not in schema.names
    xs = np.arange(x0, x1, float(tile_um))
    ys = np.arange(y0, y1, float(tile_um))
    xs = xs if len(xs) else np.array([x0])
    ys = ys if len(ys) else np.array([y0])
    parts, n_read, n_assigned, k, n_tiles = [], 0, 0, 0, len(xs) * len(ys)
    for tx0 in xs:
        for ty0 in ys:
            jobs._job_check()
            k += 1
            tx1, ty1 = min(tx0 + tile_um, x1), min(ty0 + tile_um, y1)
            tx = reseg._read_tx_region(tx_path, ["x_location", "y_location", "feature_name"],
                                       tx0, tx1, ty0, ty1, extra_filters=filters, read_dictionary=dict_cols)
            if need_name_filter:
                tx = reseg._tx_drop_control_features(tx)
            # tiles are half-open so that a transcript on a tile edge is read once
            px, py = tx["x_location"].to_numpy(), tx["y_location"].to_numpy()
            keep = ((px < tx1) | (tx1 >= x1)) & ((py < ty1) | (ty1 >= y1))
            if not keep.all():
                tx, px, py = tx[keep], px[keep], py[keep]
            n_read += int(len(tx))
            if progress is not None:
                progress(k / n_tiles)
            if not len(tx):
                continue
            hit = tree.query(shapely.points(px, py), predicate="within")
            if not hit.shape[1]:
                continue
            pt, first = np.unique(hit[0], return_index=True)
            cell = hit[1][first]
            n_assigned += int(len(pt))
            gene = tx["feature_name"].iloc[pt]
            gene = gene.map(lambda v: v.decode() if isinstance(v, bytes) else v) if gene.dtype == object else gene
            part = (pd.DataFrame({"cell": cell, "gene": np.asarray(gene.astype(str))})
                    .groupby(["cell", "gene"], sort=False).size().rename("n").reset_index())
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=cols_out), n_read, 0
    out = pd.concat(parts, ignore_index=True).groupby(["cell", "gene"], sort=False)["n"].sum().reset_index()
    return out, n_read, n_assigned


def _iseg_param_tag(p: dict, n_patches: int) -> tuple:
    """(parameter string, 8-character cache tag) of a run."""
    _A = lambda v: "A" if v is None else v
    s = (f"b{p['backend']}_ch{'+'.join(p['channels'])}_d{p['diameter_um']}_e{p['expand_um']}_t{p['thr_scale']}"
         f"_m{p['min_tx']}_w{p['patch_um']}_o{p['overlap_um']}_x{_A(p['x_min'])}-{_A(p['x_max'])}"
         f"_y{_A(p['y_min'])}-{_A(p['y_max'])}_np{n_patches}")
    if p["qv_min"] != 20:
        s += f"_qv{p['qv_min']}"
    if str(p["backend"]).startswith("cellpose"):
        s += f"_f{p['flow_threshold']}_c{p['cellprob_threshold']}"
    return s, hashlib.md5(s.encode()).hexdigest()[:8]


# ─── The run ──────────────────────────────────────────────────────────────────
def _iseg_set_factory(_gen):
    def _set(status, message, result=None):
        with core._imgseg_lock:
            if jobs._stale_job(_gen, status, message):
                return
            core._imgseg_state["status"] = status
            core._imgseg_state["message"] = message
            if result is not None:
                core._imgseg_state["result"] = result
            if status != "running":
                core._imgseg_state["loading_out_dir"] = None
    return _set


def _run_imgseg(backend: str = "classical", channels=None, diameter_um: float = _ISEG_DIAMETER_UM,
                expand_um: float = _ISEG_EXPAND_UM, x_min=None, x_max=None, y_min=None, y_max=None,
                patch_um: float = _ISEG_PATCH_UM, overlap_um: float = _ISEG_OVERLAP_UM,
                thr_scale: float = 1.0, min_tx: int = _ISEG_MIN_TX, qv_min: float = 20,
                gpu: bool = False, flow_threshold: float = 0.4, cellprob_threshold: float = 0.0) -> None:
    """Background job: segment the morphology image patch by patch, assign transcripts to
    the polygons, build the count matrix and store the run like a Baysor / Proseg run.

    channels: channel names of the image, the nuclear channel LAST (one name: nuclear only).
    Cancellation is checked between patches and between transcript tiles."""
    _gen = jobs._dataset_gen
    _set = _iseg_set_factory(_gen)
    out_dir = None
    _t0 = _time.time()
    try:
        data_dir = core.DATA.get("data_dir")
        tx_path = os.path.join(data_dir, "transcripts.parquet") if data_dir else None
        if not tx_path or not os.path.exists(tx_path):
            _set("error", "Resegmentation needs the original Xenium output directory "
                          "(transcripts.parquet); this dataset was " + dataview._origin_phrase() + ".")
            return
        with core._sdata_lock:
            sdata = core._sdata_state.get("sdata")
        image_key, image = _iseg_image(sdata)
        if image is None:
            _set("error", "The morphology image is read from the dataset's SpatialData store: run "
                          "Step 1 (Prepare Patches) first.")
            return
        avail = [str(c) for c in image.coords["c"].values]
        channels = [str(c) for c in (channels or []) if c]
        if not channels:
            channels = [_iseg_default_channel(avail)]
        missing = [c for c in channels if c not in avail]
        if missing:
            _set("error", f"Channel(s) not in the morphology image: {', '.join(missing)}. "
                          f"Available: {', '.join(avail)}.")
            return
        kind = str(backend or "classical").partition(":")[0]
        if kind == "classical":
            channels = channels[-1:]               # the classical backend reads the nuclear channel only
        elif len(channels) > 1 and not _iseg_uses_second_channel(backend):
            channels = channels[-1:]               # … and so does the Cellpose nuclei model
        px = float(core.PIXEL_SIZE_UM)
        p = {"backend": str(backend or "classical"), "channels": channels,
             "diameter_um": float(diameter_um or _ISEG_DIAMETER_UM),
             "expand_um": float(expand_um or 0.0), "thr_scale": float(thr_scale or 1.0),
             "min_tx": int(max(0, min_tx if min_tx is not None else _ISEG_MIN_TX)),
             "patch_um": float(patch_um or _ISEG_PATCH_UM), "overlap_um": float(overlap_um or _ISEG_OVERLAP_UM),
             "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "qv_min": qv_min,
             "flow_threshold": float(flow_threshold), "cellprob_threshold": float(cellprob_threshold)}
        if p["diameter_um"] <= 0 or p["patch_um"] <= 2 * p["overlap_um"]:
            _set("error", "The diameter must be positive and the patch width more than twice the overlap.")
            return
        method, info = _iseg_make_method(p["backend"], p["diameter_um"] / px, n_channels=len(channels),
                                         expand_px=p["expand_um"] / px, thr_scale=p["thr_scale"], gpu=gpu,
                                         flow_threshold=p["flow_threshold"],
                                         cellprob_threshold=p["cellprob_threshold"])

        # ── image patches (sopa) ──────────────────────────────────────────
        _set("running", "Making image patches…")
        ny, nx = int(image.sizes["y"]), int(image.sizes["x"])
        region_um = None
        region_px = None
        if any(v is not None for v in (x_min, x_max, y_min, y_max)):
            region_um = (0.0 if x_min is None else float(x_min), 0.0 if y_min is None else float(y_min),
                         nx * px if x_max is None else float(x_max), ny * px if y_max is None else float(y_max))
            region_px = tuple(v / px for v in region_um)
        mini, n_patches = _iseg_patches(sdata, image_key, p["patch_um"] / px, p["overlap_um"] / px, region_px)
        _param_str, _param_tag = _iseg_param_tag(p, n_patches)
        final_dir = caching._seg_run_dir(_ISEG_TOOL, _param_tag)
        out_dir = caching._seg_stage_dir(final_dir)
        print(f"  Image segmentation: cache dir = {os.path.basename(final_dir)} ({_param_str}); "
              f"{info['label']}, channel(s) {', '.join(channels)}, {n_patches} image patch(es)", flush=True)

        # ── segmentation, one patch at a time ─────────────────────────────
        from sopa.segmentation import StainingSegmentation, solve_conflicts
        seg = StainingSegmentation(mini, method, channels, image_key=image_key, min_area=0,
                                   clip_limit=info["preprocess"]["clip_limit"],
                                   gaussian_sigma=info["preprocess"]["gaussian_sigma"])
        patch_dir = os.path.join(out_dir, "patches")
        n_failed = 0
        for pi in range(n_patches):
            _set("running", f"Image segmentation ({info['label']}): patch {pi + 1}/{n_patches}…")
            jobs._job_progress(fraction=0.7 * pi / n_patches)
            try:
                seg.write_patch_cells(patch_dir, pi)
            except jobs._DatasetChanged:
                raise
            except Exception as pe:
                n_failed += 1
                print(f"  Image segmentation: patch {pi + 1} failed: {str(pe)[:300]}", flush=True)
        if n_failed == n_patches:
            _set("error", f"Segmentation failed on every patch (see the server log). Backend: {info['label']}.")
            return
        _set("running", "Merging cells across patches…")
        jobs._job_progress(fraction=0.7)
        import geopandas as gpd
        from shapely import affinity
        cells = StainingSegmentation.read_patches_cells(patch_dir)
        cells = cells[~cells.geometry.is_empty & cells.geometry.is_valid] if len(cells) else cells
        if not len(cells):
            _set("error", "No cell was found in the image. Check the channel, the diameter and the region.")
            return
        cells = solve_conflicts(cells)
        geoms = [affinity.scale(g, xfact=px, yfact=px, origin=(0, 0)) for g in cells.geometry]
        if region_um is not None:
            cen = np.array([[g.centroid.x, g.centroid.y] for g in geoms])
            inside = ((cen[:, 0] >= region_um[0]) & (cen[:, 0] <= region_um[2])
                      & (cen[:, 1] >= region_um[1]) & (cen[:, 1] <= region_um[3]))
            geoms = [g for g, ok in zip(geoms, inside) if ok]
        n_poly = len(geoms)
        print(f"  Image segmentation: {n_poly:,} polygon(s) after merging the patch overlaps", flush=True)

        # ── transcripts → polygons → counts ───────────────────────────────
        _set("running", f"Assigning transcripts to {n_poly:,} cells…")
        # the denominator of "% of transcripts assigned" is the area that was segmented (the
        # region, or the extent of the image patches of a whole-tissue run) - not the
        # bounding box of the cells that happened to be found
        seg_bounds = region_um
        if seg_bounds is None:
            try:
                from sopa.constants import SopaKeys
                _b = np.asarray(mini.shapes[SopaKeys.PATCHES].total_bounds, dtype=np.float64) * px
                seg_bounds = (max(0.0, _b[0]), max(0.0, _b[1]), min(nx * px, _b[2]), min(ny * px, _b[3]))
            except Exception:
                seg_bounds = (0.0, 0.0, nx * px, ny * px)
        counts, n_tx, n_assigned = _iseg_assign_transcripts(
            geoms, tx_path, qv_min=qv_min, bounds=seg_bounds,
            progress=lambda f: jobs._job_progress(fraction=0.72 + 0.18 * f))
        per_cell = np.zeros(n_poly, dtype=np.int64)
        if len(counts):
            np.add.at(per_cell, counts["cell"].to_numpy(dtype=np.int64), counts["n"].to_numpy(dtype=np.int64))
        keep = np.flatnonzero(per_cell >= p["min_tx"])
        if not len(keep):
            _set("error", f"No segmented cell holds {p['min_tx']} or more gene transcripts "
                          f"({n_poly:,} polygons, {n_assigned:,} of {n_tx:,} transcripts inside one).")
            return
        ids = np.array([f"img-{i + 1}" for i in range(len(keep))], dtype=object)
        cen = np.array([[geoms[i].centroid.x, geoms[i].centroid.y] for i in keep])
        cells_df = pd.DataFrame({"x_centroid": cen[:, 0], "y_centroid": cen[:, 1],
                                 "transcript_counts": per_cell[keep],
                                 "cell_area": np.array([geoms[i].area for i in keep])},
                                index=pd.Index(ids, name="cell_id"))
        cell_bounds = {}
        for cid, i in zip(ids, keep):
            ring = _iseg_ring(geoms[i])
            if ring is not None:
                cell_bounds[str(cid)] = ring
        lut = np.full(n_poly, -1, dtype=np.int64)
        lut[keep] = np.arange(len(keep))
        row = lut[counts["cell"].to_numpy(dtype=np.int64)]
        seg_df = pd.DataFrame({"cell": ids[np.clip(row, 0, None)], "gene": counts["gene"].to_numpy(),
                               "n": counts["n"].to_numpy()})[row >= 0]
        _set("running", "Building expression matrix…")
        jobs._job_progress(fraction=0.9)
        expr_mat = reseg._build_baysor_expr(seg_df, cells_df)       # the path of a Baysor run
        n_cells = len(cells_df)
        n_kept_tx = int(per_cell[keep].sum())
        jobs._check_dataset_gen(_gen)
        params_out = {
            "tool": _ISEG_TOOL, **p, "backend_label": info["label"], "model": info["model"],
            "weights_file": info["weights"], "image_key": image_key, "pixel_size_um": px,
            "n_patches": n_patches, "n_patches_failed": n_failed, "n_polygons": n_poly,
            "n_cells": n_cells, "n_transcripts": int(n_tx), "n_assigned": n_kept_tx,
            "n_in_dropped_cells": int(n_assigned - n_kept_tx),
            "assigned_fraction": float(n_kept_tx / n_tx) if n_tx else None,
            "seconds_segmentation": round(_time.time() - _t0, 1),
            "param_tag": _param_tag,
            "dataset_hash": hashlib.md5(caching._dataset_key().encode()).hexdigest()[:8],
        }
        with open(os.path.join(out_dir, "params.json"), "w") as f:
            json.dump(params_out, f)
        import shutil
        shutil.rmtree(patch_dir, ignore_errors=True)           # per-patch polygons: merged above
        _set("running", "Computing UMAP and clusters…")
        jobs._job_progress(fraction=0.92)
        clustering._compute_reseg_clusters_umap(cells_df, expr_mat, progress_fn=lambda msg: _set("running", msg))
        _set("running", "Writing SpatialData zarr…")
        jobs._check_dataset_gen(_gen)
        jobs._job_shield()                 # last cancellation point: the run is stored and published
        jobs._job_progress(fraction=0.97)
        zarr_path = reseg._build_reseg_sdata(cells_df, cell_bounds, expr_mat, core.DATA["gene_names"],
                                             _ISEG_TOOL, out_dir)
        if not zarr_path:
            _set("error", "The SpatialData store of the run could not be written (see the server log).")
            return
        caching._seg_stage_commit(out_dir, final_dir)
        zarr_path = os.path.join(final_dir, os.path.basename(zarr_path))
        out_dir = final_dir
        with core._roi_lock:
            regions._roi_apply_metadata_to_df(cells_df, core._roi_state["rois"])
        frac = f"{100.0 * n_kept_tx / n_tx:.1f} %" if n_tx else "n/a"
        msg = (f"Done — {n_cells:,} cells ({info['label']}); {frac} of {n_tx:,} gene transcripts assigned"
               + (f"; {n_failed} patch(es) failed" if n_failed else ""))
        _set("done", msg, result={"cells_df": cells_df, "cell_bounds": cell_bounds, "out_dir": out_dir,
                                  "expr": expr_mat, "source": _ISEG_TOOL, "sdata_path": zarr_path})
        print(f"  Image segmentation: {msg} ({_time.time() - _t0:.0f} s)", flush=True)
        provenance._prov_record(
            "imgseg", f"Image-based segmentation ({info['label']})",
            seg_source=provenance._prov_run_source(_ISEG_TOOL, {"out_dir": out_dir}),
            params={"backend": backend, "channels": channels, "diameter_um": diameter_um,
                    "expand_um": expand_um, "patch_um": patch_um, "overlap_um": overlap_um,
                    "threshold_scale": thr_scale, "min_transcripts": min_tx, "qv_min": qv_min,
                    "gpu": gpu, "flow_threshold": flow_threshold,
                    "cellprob_threshold": cellprob_threshold, "region": [x_min, x_max, y_min, y_max],
                    "resolved": p},
            result={"n_cells": int(n_cells), "n_transcripts": int(n_tx),
                    "n_transcripts_assigned": int(n_kept_tx), "n_patches_failed": int(n_failed)},
            message=msg, cache=False, t0=_t0,
            packages=("sopa", "spatialdata", "scikit-image") + (("cellpose", "torch") if info["backend"] == "cellpose" else ()))
    except jobs._JobCancelled:
        raise
    except jobs._DatasetChanged:
        raise
    except (RuntimeError, ValueError, AssertionError) as exc:
        _set("error", str(exc)[:600])
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set("error", str(exc)[:300])
    finally:
        caching._seg_stage_discard(out_dir)


def _load_cached_imgseg(out_dir: str) -> None:
    """Load a cached image-segmentation run (its SpatialData store) into core._imgseg_state."""
    _gen = jobs._dataset_gen
    _set = _iseg_set_factory(_gen)
    print(f"Loading cached image segmentation: {os.path.basename(out_dir)}…", flush=True)
    try:
        _set("running", "Loading cached image segmentation…")
        zarr_path = os.path.join(out_dir, f"spatialdata_{_ISEG_TOOL}.zarr")
        if not os.path.isdir(zarr_path):
            _set("error", f"No SpatialData store in {os.path.basename(out_dir)}: run the segmentation again.")
            return
        spage_path, spage_genes = None, None
        ref = os.path.join(out_dir, "spage_result.json")
        if os.path.exists(ref):
            try:
                with open(ref) as f:
                    sref = json.load(f)
                if os.path.exists(sref.get("path", "")):      # zarr (streamed) or parquet
                    spage_path, spage_genes = sref["path"], sref.get("genes", [])
            except Exception:
                pass
        cells_df, expr_mat, corr, corr_imp, cell_bounds = reseg._fast_load_reseg_zarr(zarr_path)
        n_cells = len(cells_df)
        if (("umap_1" not in cells_df.columns or not reseg._reseg_cluster_cols(cells_df.columns))
                and expr_mat is not None):
            _set("running", f"Computing UMAP and clusters for {n_cells:,} cells (one-time)…")
            clustering._compute_reseg_clusters_umap(cells_df, expr_mat, progress_fn=lambda msg: _set("running", msg))
            reseg._update_reseg_zarr_obs(zarr_path, cells_df)
        with core._roi_lock:
            regions._roi_apply_metadata_to_df(cells_df, core._roi_state["rois"])
        _set("done", f"Loaded — {n_cells:,} cells", result={
            "cells_df": cells_df, "cell_bounds": cell_bounds, "out_dir": out_dir, "expr": expr_mat,
            "source": _ISEG_TOOL, "sdata_path": zarr_path, "split_corrected_expr": corr,
            "split_corrected_imputed_genes": corr_imp, "spage_result_path": spage_path,
            "spage_result_genes": spage_genes,
            **(reseg._reseg_zarr_gene_index(zarr_path, expr_mat.shape[1]) if expr_mat is not None else {})})
        print(f"Loaded cached image segmentation: {n_cells:,} cells, {len(cell_bounds):,} boundaries", flush=True)
        provenance._prov_record(
            "imgseg", "Image-based segmentation (cached run loaded)",
            seg_source=provenance._prov_run_source(_ISEG_TOOL, {"out_dir": out_dir}),
            params={"run": os.path.basename(str(out_dir).rstrip("/"))},
            result={"n_cells": int(n_cells)}, message=f"Loaded — {n_cells:,} cells", cache=True,
            once=("load", str(out_dir), core._seg_run_id({"out_dir": out_dir})))
    except jobs._JobCancelled:
        raise
    except jobs._DatasetChanged:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set("error", str(exc)[:300])


# ─── Layout: the third panel of the Resegmentation modal ──────────────────────
def _iseg_panel():
    inp = {"width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
           "border": f"1px solid {core.BORDER}", "borderRadius": "4px", "padding": "3px 6px",
           "fontSize": "11px", "boxSizing": "border-box"}
    small = {"fontSize": "10px", "color": core.MUTED, "marginBottom": "2px"}

    def box(label, comp):
        return html.Div([html.Div(label, style=small), comp], style={"flex": "1"})

    def num(id, placeholder=None, value=None, **kw):
        return dcc.Input(id=id, type="number", placeholder=placeholder, value=value, style=inp, **kw)

    row = {"display": "flex", "gap": "6px", "marginBottom": "6px"}
    dd = {"fontSize": "12px", "marginBottom": "8px"}
    return html.Div(id="modal-imgseg-panel", children=[
        html.Div("Segment cells on the morphology image instead of the transcripts. Built-in classical "
                 "watershed (a simple baseline, no model) or Cellpose when it is installed with its weights.",
                 style={"fontSize": "10px", "color": core.MUTED, "marginBottom": "8px", "lineHeight": "1.4"}),
        ui.ctrl_label("Segmenter"),
        dcc.Dropdown(id="iseg-backend", options=[{"label": "Classical watershed (built-in baseline, no model)",
                                                  "value": "classical"}],
                     value="classical", clearable=False, style=dd),
        html.Div(id="iseg-backend-note", style={"fontSize": "10px", "color": core.MUTED, "marginBottom": "8px",
                                                "lineHeight": "1.4", "whiteSpace": "pre-wrap"}),
        html.Div([
            box("Nuclear channel", dcc.Dropdown(id="iseg-channel", options=[], value=None, clearable=False,
                                                placeholder="Run Step 1 to read the image…",
                                                style={"fontSize": "12px"})),
            box("Membrane / cytoplasm channel (Cellpose cyto models only)",
                dcc.Dropdown(id="iseg-channel-cyto", options=[], value=None, placeholder="none",
                             style={"fontSize": "12px"})),
        ], style={**row, "marginBottom": "8px"}),
        ui.ctrl_label("Spatial Region (µm) — leave blank for the whole tissue"),
        html.Div([box("X min", num(id="iseg-xmin", placeholder="auto")), box("X max", num(id="iseg-xmax", placeholder="auto")),
                  box("Y min", num(id="iseg-ymin", placeholder="auto")), box("Y max", num(id="iseg-ymax", placeholder="auto"))], style=row),
        html.Button("Use Current Viewport", id="iseg-use-viewport", n_clicks=0,
                    style={"width": "100%", "padding": "4px 0", "backgroundColor": core.CARD_BG,
                           "color": core.MUTED, "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                           "cursor": "pointer", "fontSize": "11px", "marginBottom": "10px"}),
        html.Div([
            box("Nucleus diameter (µm)", num(id="iseg-diameter", value=_ISEG_DIAMETER_UM, min=1, max=100, step=0.5)),
            box("Expansion (µm; 0 = nuclei only)", num(id="iseg-expand", value=_ISEG_EXPAND_UM, min=0, max=50, step=0.5)),
            box("Min transcripts per cell", num(id="iseg-min-tx", value=_ISEG_MIN_TX, min=0, max=10000, step=1)),
        ], style=row),
        html.Div([
            box("Image patch width (µm)", num(id="iseg-patch", value=_ISEG_PATCH_UM, min=100, max=5000, step=50)),
            box("Classical: threshold × Otsu", num(id="iseg-thr", value=1.0, min=0.2, max=5, step=0.05)),
            box("Cellpose: flow threshold", num(id="iseg-flow", value=0.4, min=0, max=3, step=0.1)),
            box("Cellpose: cell probability", num(id="iseg-cellprob", value=0.0, min=-6, max=6, step=0.5)),
        ], style=row),
        dcc.Checklist(id="iseg-gpu",
                      options=[{"label": html.Span(" Cellpose on the GPU when one is available",
                                                   style={"fontSize": "12px", "color": core.TEXT}), "value": "yes"}],
                      value=[], inputStyle={"marginRight": "6px"}, style={"marginBottom": "8px"}),
        html.Div(_ISEG_NOTE, style={"fontSize": "10px", "color": core.MUTED, "lineHeight": "1.4",
                                    "marginBottom": "6px"}),
    ], style={"display": "none"})
