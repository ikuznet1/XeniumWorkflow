"""Morphology image overlay for the Plotly view: image handles, overviews, tiles."""

import os
import io
import base64
import threading
import hashlib
import numpy as np
from PIL import Image

from . import core


# ─── Morphology image overlay ─────────────────────────────────────────────────
# Max viewport width (µm) for which image tiles will load automatically
MORPH_MAX_UM = 5000

# Pseudocolors matching standard fluorescence conventions
MORPH_CHANNELS = [
    {"label": "DAPI",                   "value": "dapi",     "color": (70,  130, 255), "ch": 0},
    {"label": "ATP1A1/CD45/E-Cadherin", "value": "boundary", "color": (50,  255, 80),  "ch": 1},
    {"label": "18S",                    "value": "18s",      "color": (255, 60,  0),   "ch": 2},
    {"label": "αSMA/Vimentin",          "value": "sma",      "color": (255, 0,   220), "ch": 3},
]
MORPH_MAX_UM = 5000

# ── Change 1: reduced prefetch — only 15% extra on each side ──────────────────
_PREFETCH_FRAC = 0.15   # was 0.50; smaller = faster first render, less pan buffer

# ── Change 2 & 3: per-channel handles with LRU store cache ────────────────────
# One TiffFile + LRU-backed zarr per channel per z-level → parallel reads,
# no shared-lock needed, LRU avoids re-reading the same tiles from disk.
_morph_handles: dict = {}    # {(data_dir, z_level): {"handles": [(tif, arr)×n_ch], "H", "W", "n_ch"} | None}
_morph_init_lock = threading.Lock()   # only held during one-time initialisation

# Low-resolution overviews cached in memory and on disk
# Coarse (stride ~26): for viewport > ~4,400 µm  — fast zoomed-out view
# Medium (stride  ~8): for viewport 1,300–4,400 µm — eliminates mid-zoom tile reads
_morph_overview: dict = {}        # {(data_dir, z_level): ov_dict, (data_dir, "med", z_level): ov_dict}
_morph_overview_lock  = threading.Lock()
_overview_generating  = set()     # keys currently being generated in background

# Render cache: stores raw float32 channel data (brightness-independent)
# so brightness/opacity changes are instant without re-reading tiles.
_morph_render_cache: dict = {}
_morph_render_lock  = threading.Lock()

# ── Change 4: progressive rendering queue ─────────────────────────────────────
# Background hires fetches append here; the Dash callback polls and pushes a Patch.
_morph_hires_queue: list = []   # list of image-dict results; GIL protects appends/pops


def _reset_morph_caches() -> None:
    """Drop all in-memory morphology state (called when a different dataset is loaded)."""
    with _morph_init_lock:
        for _info in _morph_handles.values():
            if _info:
                for _tif, _ in _info["handles"]:
                    try:
                        _tif.close()
                    except Exception:
                        pass
        _morph_handles.clear()
    with _morph_overview_lock:
        _morph_overview.clear()
    with _morph_render_lock:
        _morph_render_cache.clear()
    _morph_hires_queue.clear()


class _ZarrTiffArray:
    """Thin array adapter over tifffile's ZarrTiffStore.

    Zarr ≥ 3 no longer accepts ZarrTiffStore (a plain MutableMapping) in
    zarr.open().  This class wraps the store directly, assembling tiles on
    demand and applying any stride requested via slice notation.

    Supports indexing of the form  arr[channel_int, y_slice, x_slice]
    which is all that _read_one_channel and _gen_overviews require.
    """

    def __init__(self, store):
        import json
        meta = json.loads(store[".zarray"])
        self.shape   = tuple(meta["shape"])   # (n_ch, H, W)
        self._csize  = meta["chunks"]         # [1, tile_h, tile_w]
        # Single-channel images (e.g. DAPI-only morphology_focus) are 2-D (H, W):
        # present them as (1, H, W) so callers keep the (C, H, W) contract.
        self._2d     = len(self.shape) == 2
        if self._2d:
            self.shape  = (1, *self.shape)
            self._csize = [1, *self._csize]
        self._dtype  = np.dtype(meta["dtype"])
        self._store  = store
        self._pyr    = {}                     # c_idx -> [(factor, store, (Hl, Wl), (cy, cx))]

    def __getitem__(self, idx):
        if not isinstance(idx, tuple) or len(idx) != 3:
            raise IndexError(f"_ZarrTiffArray: expected (ch, y_slice, x_slice), got {idx!r}")
        c_idx, ys, xs = idx
        _, H, W = self.shape
        cy, cx   = self._csize[1], self._csize[2]

        # Resolve slices to (start, stop, step)
        y0, y1, dy = ys.indices(H) if isinstance(ys, slice) else (int(ys), int(ys) + 1, 1)
        x0, x1, dx = xs.indices(W) if isinstance(xs, slice) else (int(xs), int(xs) + 1, 1)
        y0 = max(0, y0);  y1 = min(H, y1)
        x0 = max(0, x0);  x1 = min(W, x1)
        h, w = y1 - y0, x1 - x0
        if h <= 0 or w <= 0:
            out_h = len(range(y0, y1, dy)) if h > 0 else 0
            out_w = len(range(x0, x1, dx)) if w > 0 else 0
            return np.zeros((out_h, out_w), dtype=self._dtype)

        # Fill the strided output directly (same pixels as out[::dy, ::dx] of the
        # full-resolution region, without allocating that region).
        ys_idx = np.arange(y0, y1, dy)
        xs_idx = np.arange(x0, x1, dx)

        # Zoomed-out reads: take the pixels from the coarsest pyramid level that is still
        # at least as fine as the requested stride, so far fewer tiles are decoded.
        levels = self._pyr.get(c_idx) if min(dy, dx) >= 2 else None
        if levels:
            best = None
            for lv in levels:                       # (factor, store, (Hl, Wl), (cyl, cxl))
                if lv[0] <= min(dy, dx) * 1.02 and (best is None or lv[0] > best[0]):   # levels are ~2^k
                    best = lv
            if best is not None:
                _f, lstore, (Hl, Wl), (cyl, cxl) = best
                ys_l = np.minimum(ys_idx * Hl // H, Hl - 1)
                xs_l = np.minimum(xs_idx * Wl // W, Wl - 1)
                return self._gather(lstore, lambda by, bx: f"{by}.{bx}", cyl, cxl, ys_l, xs_l)

        key_fn = ((lambda by, bx: f"{by}.{bx}") if self._2d
                  else (lambda by, bx: f"{c_idx}.{by}.{bx}"))
        return self._gather(self._store, key_fn, cy, cx, ys_idx, xs_idx)

    def _gather(self, store, key_fn, cy, cx, ys_idx, xs_idx):
        """Assemble out[i, j] = image[ys_idx[i], xs_idx[j]] from the tiles of *store*
        (ys_idx / xs_idx are sorted pixel indices; only tiles they touch are decoded)."""
        out = np.zeros((len(ys_idx), len(xs_idx)), dtype=self._dtype)
        for bcy in range(int(ys_idx[0]) // cy, int(ys_idx[-1]) // cy + 1):
            ylo, yhi = np.searchsorted(ys_idx, [bcy * cy, (bcy + 1) * cy])
            if ylo == yhi:
                continue
            for bcx in range(int(xs_idx[0]) // cx, int(xs_idx[-1]) // cx + 1):
                xlo, xhi = np.searchsorted(xs_idx, [bcx * cx, (bcx + 1) * cx])
                if xlo == xhi:
                    continue   # stride skips this tile entirely
                raw = store.get(key_fn(bcy, bcx))
                if raw is None:
                    continue
                tile = np.asarray(raw).reshape(cy, cx)
                out[ylo:yhi, xlo:xhi] = tile[np.ix_(ys_idx[ylo:yhi] - bcy * cy,
                                                    xs_idx[xlo:xhi] - bcx * cx)]
        return out

    def attach_pyramid(self, c_idx, tif):
        """Register the reduced-resolution levels of *tif* (a TiffFile whose first series is
        the 2-D plane of channel *c_idx*) for use by zoomed-out reads. No-op if it has none."""
        import json
        try:
            series = tif.series[0]
            _, H, W = self.shape
            if len(series.levels) < 2 or tuple(series.levels[0].shape[-2:]) != (H, W):
                return
            levels = []
            for li in range(1, len(series.levels)):
                lstore = tif.aszarr(level=li)
                meta = json.loads(lstore[".zarray"])
                if len(meta["shape"]) != 2 or np.dtype(meta["dtype"]) != self._dtype:
                    return
                Hl, Wl = meta["shape"]
                levels.append((H / Hl, lstore, (Hl, Wl), tuple(meta["chunks"])))
            self._pyr[c_idx] = levels
        except Exception:
            self._pyr.pop(c_idx, None)


def _open_tiff_as_array(tif):
    """Open a TiffFile as an array-like, falling back to _ZarrTiffArray for zarr v3."""
    import warnings
    warnings.filterwarnings("ignore", message=".*OME series cannot read multi-file pyramids.*")
    # level=0: a single-file pyramid would otherwise yield a multiscale group
    store = tif.aszarr(level=0)
    try:
        import zarr as zarr_mod
        arr = zarr_mod.open(store, mode="r")
        if getattr(arr, "ndim", None) != 3:
            return _ZarrTiffArray(store)   # 2-D image → (1, H, W) adapter
        return arr
    except (TypeError, AttributeError):
        return _ZarrTiffArray(store)


def _morph_focus_path(data_dir: str, z_level: int):
    """Locate the morphology focus image for a file index, or None if absent."""
    import glob
    d = os.path.join(data_dir, "morphology_focus")
    p = os.path.join(d, f"morphology_focus_{z_level:04d}.ome.tif")     # XOA 2.x/3.x
    if os.path.exists(p):
        return p
    hits = sorted(h for h in glob.glob(os.path.join(d, f"ch{z_level:04d}_*.ome.tif"))
                  if not os.path.basename(h).startswith("._"))         # XOA 4.x
    if hits:
        return hits[0]
    if z_level == 0:
        try:   # path recorded by the instrument, then the XOA 1.x single-file layout
            rel = core.DATA.get("metadata", {}).get("images", {}).get("morphology_focus_filepath")
        except Exception:
            rel = None
        root = os.path.abspath(data_dir)
        for cand in (rel, "morphology_focus.ome.tif"):
            if not cand or not isinstance(cand, str):
                continue
            # the recorded path comes from the bundle's metadata: it must name a TIFF
            # inside the bundle (the WebGL viewer serves this file to the browser)
            full = os.path.abspath(os.path.join(root, cand))
            if (full.startswith(root + os.sep) and full.lower().endswith((".tif", ".tiff"))
                    and os.path.isfile(full)):
                return os.path.join(data_dir, os.path.relpath(full, root))
    return None


def _get_morph_handles(data_dir: str, z_level: int):
    """Lazily open one TiffFile per channel with a direct-tile array adapter."""
    hk = (data_dir, z_level)   # keyed by dataset so a switched sample never reuses handles
    with _morph_init_lock:
        if hk in _morph_handles:
            return _morph_handles[hk]

        import warnings, logging, tifffile
        warnings.filterwarnings("ignore", message=".*OME series cannot read multi-file pyramids.*")
        logging.getLogger("tifffile").setLevel(logging.ERROR)

        path  = _morph_focus_path(data_dir, z_level)

        if path is None:
            _morph_handles[hk] = None
            return None
        fname = os.path.basename(path)

        try:
            # Probe shape
            tif_probe = tifffile.TiffFile(path)
            arr_probe = _open_tiff_as_array(tif_probe)
            n_ch, H, W = arr_probe.shape
            tif_probe.close()

            # Open one handle per channel
            handles = []
            for _ci in range(n_ch):
                tif_ch = tifffile.TiffFile(path)
                arr    = _open_tiff_as_array(tif_ch)
                handles.append((tif_ch, arr))
                # Pyramid levels: tifffile exposes them for a single-file 2-D image, but not
                # through a multi-file OME series, so open the channel's own file directly.
                if isinstance(arr, _ZarrTiffArray):
                    try:
                        if arr._2d:
                            arr.attach_pyramid(0, tif_ch)
                        else:
                            _pg    = tif_ch.series[0].pages[_ci]
                            _cpath = _pg.parent.filehandle.path
                            _tif_p = tifffile.TiffFile(_cpath, is_ome=False)
                            arr.attach_pyramid(_ci, _tif_p)
                            if _ci in arr._pyr:
                                arr._pyr_tifs = getattr(arr, "_pyr_tifs", []) + [_tif_p]  # keep open
                            else:
                                _tif_p.close()
                    except Exception:
                        pass

            _morph_handles[hk] = {"handles": handles, "H": H, "W": W, "n_ch": n_ch}
            adapter = type(handles[0][1]).__name__
            print(f"  Morphology z{z_level}: {n_ch}ch × {H}×{W} px [{adapter}]", flush=True)
        except Exception as exc:
            print(f"  Warning: could not open {fname}: {exc}", flush=True)
            _morph_handles[hk] = None

    return _morph_handles.get(hk)


def _overview_cache_path(data_dir: str, z_level: int, medium: bool = False) -> str:
    tag = hashlib.md5(data_dir.encode()).hexdigest()[:8]
    cache_dir = core.CACHE_BASE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    suffix = "med_" if medium else ""
    return os.path.join(cache_dir, f"morph_overview_{suffix}{tag}_z{z_level}.npz")


def _load_overview_from_disk(path: str):
    """Load an NPZ overview file. Returns dict or None."""
    try:
        data = np.load(path)
        n_ch = sum(1 for k in data.files if k.startswith("ch"))
        return {
            "channels": [data[f"ch{i}"] for i in range(n_ch)],
            "stride":   int(data["stride"]),
        }
    except Exception:
        return None


def _load_or_generate_overview(data_dir: str, z_level: int):
    """Return coarse (stride ~26) overview dict, or None if not yet ready."""
    ov_key = (data_dir, z_level)
    with _morph_overview_lock:
        if ov_key in _morph_overview:
            return _morph_overview[ov_key]

    ov_path = _overview_cache_path(data_dir, z_level, medium=False)
    if os.path.exists(ov_path):
        ov = _load_overview_from_disk(ov_path)
        if ov:
            with _morph_overview_lock:
                _morph_overview[ov_key] = ov
            print(f"  Overview z{z_level} (coarse): loaded from cache", flush=True)
            return ov

    # Generate coarse + medium together in one background pass
    with _morph_overview_lock:
        if ("coarse", data_dir, z_level) in _overview_generating:
            return None
        _overview_generating.add(("coarse", data_dir, z_level))

    def _gen_overviews():
        try:
            import tifffile
            path  = _morph_focus_path(data_dir, z_level)
            if path is None:
                return

            tif_ov  = tifffile.TiffFile(path)
            zarr_ov = _open_tiff_as_array(tif_ov)
            H, W    = zarr_ov.shape[1], zarr_ov.shape[2]
            n_ch    = zarr_ov.shape[0]

            # ── Change 5: read at medium stride, derive coarse from it ────
            # Both reads touch all tiles anyway; reading medium first lets us
            # subsample to coarse without a second zarr pass.
            med_stride   = 8
            coarse_factor = max(1, max(H, W) // 4096) // med_stride or 3
            # Compute actual coarse stride as a multiple of med_stride
            coarse_stride = med_stride * coarse_factor

            print(f"  Generating overviews z{z_level} (medium stride={med_stride}, "
                  f"coarse stride≈{coarse_stride})…", flush=True)

            med_channels    = []
            coarse_channels = []
            for ch in range(n_ch):
                arr_med = zarr_ov[ch, ::med_stride, ::med_stride].astype(np.uint16)
                med_channels.append(arr_med)
                coarse_channels.append(arr_med[::coarse_factor, ::coarse_factor])
                print(f"    ch{ch}: medium {arr_med.shape}", flush=True)
            tif_ov.close()

            # Save medium overview
            med_path = _overview_cache_path(data_dir, z_level, medium=True)
            np.savez_compressed(med_path,
                                **{f"ch{i}": med_channels[i] for i in range(n_ch)},
                                stride=np.array(med_stride))
            ov_med = {"channels": med_channels, "stride": med_stride}

            # Save coarse overview
            ov_path_c = _overview_cache_path(data_dir, z_level, medium=False)
            np.savez_compressed(ov_path_c,
                                **{f"ch{i}": coarse_channels[i] for i in range(n_ch)},
                                stride=np.array(coarse_stride))
            ov_coarse = {"channels": coarse_channels, "stride": coarse_stride}

            with _morph_overview_lock:
                # Skip the in-memory copy if another dataset was loaded meanwhile
                if data_dir == core.DATA.get("data_dir"):
                    _morph_overview[ov_key]                     = ov_coarse
                    _morph_overview[(data_dir, "med", z_level)] = ov_med
            print(f"  Overviews z{z_level}: generated and cached", flush=True)
        except Exception as exc:
            print(f"  Overview generation error z{z_level}: {exc}", flush=True)
        finally:
            with _morph_overview_lock:
                _overview_generating.discard(("coarse", data_dir, z_level))

    threading.Thread(target=_gen_overviews, daemon=True).start()
    return None


def _load_or_get_medium_overview(data_dir: str, z_level: int):
    """Return medium (stride ~8) overview dict, or None if not yet ready."""
    key = (data_dir, "med", z_level)
    with _morph_overview_lock:
        if key in _morph_overview:
            return _morph_overview[key]

    med_path = _overview_cache_path(data_dir, z_level, medium=True)
    if os.path.exists(med_path):
        ov = _load_overview_from_disk(med_path)
        if ov:
            with _morph_overview_lock:
                _morph_overview[key] = ov
            print(f"  Overview z{z_level} (medium): loaded from cache", flush=True)
            return ov
    return None


def _compose_rgb(layers, p1p99_list, brightness, out_h, out_w, channels, crop=None):
    """Composite raw float32 layers into an RGB array, applying brightness."""
    ch_map = {c["value"]: c for c in MORPH_CHANNELS}
    rgb = np.zeros((out_h, out_w, 3), dtype=np.float32)
    for i, ch_val in enumerate(channels):
        info = ch_map.get(ch_val)
        if info is None or layers[i] is None:
            continue
        raw = layers[i]
        if crop:
            raw = raw[crop[0]:crop[1], crop[2]:crop[3]]
        p1, p99 = p1p99_list[i]
        if p99 > p1:
            norm = np.clip((raw - p1) / (p99 - p1) * brightness, 0, 1)
        else:
            norm = np.zeros_like(raw)
        color_f = np.array(info["color"], dtype=np.float32) / 255.0
        rgb += norm[..., np.newaxis] * color_f
    return rgb


def _overview_to_image(ov, channels, vp_px_x0, vp_px_y0, vp_px_x1, vp_px_y1,
                        rw_vp, rh_vp, brightness, img_opacity):
    """Crop and composite an overview dict for the given pixel viewport."""
    ov_stride = ov["stride"]
    ch_map    = {c["value"]: c for c in MORPH_CHANNELS}
    ov_y0 = vp_px_y0 // ov_stride
    ov_y1 = min(ov["channels"][0].shape[0], vp_px_y1 // ov_stride + 1)
    ov_x0 = vp_px_x0 // ov_stride
    ov_x1 = min(ov["channels"][0].shape[1], vp_px_x1 // ov_stride + 1)
    oh, ow = ov_y1 - ov_y0, ov_x1 - ov_x0
    if oh <= 0 or ow <= 0:
        return None
    rgb = np.zeros((oh, ow, 3), dtype=np.float32)
    for ch_val in channels:
        info = ch_map.get(ch_val)
        if info is None or info["ch"] >= len(ov["channels"]):
            continue
        region  = ov["channels"][info["ch"]][ov_y0:ov_y1, ov_x0:ov_x1].astype(np.float32)
        nonzero = region[region > 0]
        if nonzero.size > 50:
            p1, p99 = np.percentile(nonzero, [1, 99])
            norm = np.clip((region - p1) / (p99 - p1) * brightness, 0, 1) if p99 > p1 \
                   else np.zeros_like(region)
        else:
            norm = np.zeros_like(region)
        rgb += norm[..., np.newaxis] * (np.array(info["color"], dtype=np.float32) / 255.0)
    return _encode_overlay_jpeg(rgb, vp_px_x0, vp_px_y0, rw_vp, rh_vp, img_opacity)


def _encode_overlay_jpeg(rgb, vp_px_x0, vp_px_y0, rw_vp, rh_vp, img_opacity):
    """Clip, encode to JPEG base64, return plotly layout.images dict."""
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    img_pil   = Image.fromarray(rgb_uint8, mode="RGB")
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return dict(
        source=f"data:image/jpeg;base64,{b64}",
        xref="x", yref="y",
        x=vp_px_x0 * core.PIXEL_SIZE_UM,
        y=-(vp_px_y0 * core.PIXEL_SIZE_UM),
        sizex=rw_vp * core.PIXEL_SIZE_UM,
        sizey=rh_vp * core.PIXEL_SIZE_UM,
        xanchor="left", yanchor="top",
        layer="below", opacity=float(img_opacity),
    )


def _read_one_channel(ch_val, handles_info, py0, py1, px0, px1, stride, out_h, out_w):
    """Read one channel from its dedicated zarr handle (no lock needed)."""
    ch_map = {c["value"]: c for c in MORPH_CHANNELS}
    info   = ch_map.get(ch_val)
    if info is None or handles_info is None:
        return None, (0.0, 0.0)
    ch_idx = info["ch"]
    if ch_idx >= handles_info["n_ch"]:
        return None, (0.0, 0.0)
    _, arr = handles_info["handles"][ch_idx]
    # Each arr covers all channels; read only the one we need
    region  = arr[ch_idx, py0:py1:stride, px0:px1:stride].astype(np.float32)
    region  = region[:out_h, :out_w]
    nonzero = region[region > 0]
    if nonzero.size > 100:
        p1, p99 = float(np.percentile(nonzero, 1)), float(np.percentile(nonzero, 99))
    else:
        p1, p99 = 0.0, 0.0
    return region, (p1, p99)


def _read_channels_parallel(channels, handles_info, py0, py1, px0, px1, stride, out_h, out_w):
    """Read all requested channels in parallel threads (GIL released during JPEG2000 decode)."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(channels), 4)) as ex:
        futs = {
            ex.submit(_read_one_channel, ch, handles_info, py0, py1, px0, px1,
                      stride, out_h, out_w): ch
            for ch in channels
        }
        ordered = [concurrent.futures.as_completed  # silence linter
                   and futs[f].result()
                   for f in [next(f for f, c in futs.items() if c == ch)
                              for ch in channels]]
    layers = [r[0] for r in ordered]
    p1p99  = [r[1] for r in ordered]
    return layers, p1p99


def make_morphology_overlay(data_dir, relayout, z_level, channels, brightness, img_opacity):
    """
    Composite morphology tiles for the current viewport.

    Performance optimisations (all five changes):
      1. Prefetch fraction reduced 0.50 → 0.15 (3× fewer cold tiles).
      2. Per-channel zarr handles: parallel reads without a shared lock.
      3. LRU store cache: 50 MB per channel keeps compressed tiles in RAM.
      4. Progressive rendering: overview returned instantly; hires queued in bg.
      5. Medium-stride (stride~8) overview covers the 1,300–4,400 µm range.
    """
    if not data_dir:
        return None, ""
    if not channels or not relayout:
        return None, ""
    # Checklist values arrive in click order; the render cache key is order-free,
    # so keep one canonical order for reads, cached layers and compositing.
    channels = sorted(channels)

    handles_info = _get_morph_handles(data_dir, z_level)
    if handles_info is None:
        return None, " · morphology images not found"

    H, W = handles_info["H"], handles_info["W"]
    full_w_um = W * core.PIXEL_SIZE_UM
    full_h_um = H * core.PIXEL_SIZE_UM

    # ── Viewport bounds (µm, Y-flipped plot coords) ───────────────────────
    x0_um = float(relayout.get("xaxis.range[0]", 0))
    x1_um = float(relayout.get("xaxis.range[1]", full_w_um))
    y0_um = float(relayout.get("yaxis.range[0]", -full_h_um))
    y1_um = float(relayout.get("yaxis.range[1]", 0))

    vp_w = x1_um - x0_um
    vp_h = abs(y1_um - y0_um)
    if max(vp_w, vp_h) > MORPH_MAX_UM:
        return None, f" · zoom in to ≤{MORPH_MAX_UM:,} µm for image overlay"

    # ── Viewport in image pixels ──────────────────────────────────────────
    vp_px_x0 = max(0, int(x0_um / core.PIXEL_SIZE_UM))
    vp_px_x1 = min(W, int(x1_um / core.PIXEL_SIZE_UM) + 1)
    vp_px_y0 = max(0, int(-y1_um / core.PIXEL_SIZE_UM))
    vp_px_y1 = min(H, int(-y0_um / core.PIXEL_SIZE_UM) + 1)
    if vp_px_x1 <= vp_px_x0 or vp_px_y1 <= vp_px_y0:
        return None, ""

    rw_vp  = vp_px_x1 - vp_px_x0
    rh_vp  = vp_px_y1 - vp_px_y0
    stride = max(1, max(rw_vp, rh_vp) // 800)

    # ── Path 1: coarse overview (instant, stride ≥ coarse_stride) ────────
    ov_coarse = _load_or_generate_overview(data_dir, z_level)
    if ov_coarse and stride >= ov_coarse["stride"]:
        img = _overview_to_image(ov_coarse, channels, vp_px_x0, vp_px_y0,
                                  vp_px_x1, vp_px_y1, rw_vp, rh_vp,
                                  brightness, img_opacity)
        return img, " (overview)"

    # ── Path 1b: medium overview (instant, stride ≥ 8) ───────────────────
    ov_med = _load_or_get_medium_overview(data_dir, z_level)
    if ov_med and stride >= ov_med["stride"]:
        img = _overview_to_image(ov_med, channels, vp_px_x0, vp_px_y0,
                                  vp_px_x1, vp_px_y1, rw_vp, rh_vp,
                                  brightness, img_opacity)
        if img:
            return img, " (medium overview)"

    sorted_ch  = tuple(sorted(channels))
    cache_key  = (data_dir, z_level, sorted_ch, stride)

    # ── Path 2: render cache hit (brightness-independent raw layers) ──────
    with _morph_render_lock:
        entry = _morph_render_cache.get(cache_key)

    if entry is not None:
        cpx_x0, cpx_y0, cpx_x1, cpx_y1 = entry["px"]
        if (vp_px_x0 >= cpx_x0 and vp_px_y0 >= cpx_y0 and
                vp_px_x1 <= cpx_x1 and vp_px_y1 <= cpx_y1):
            cy0 = (vp_px_y0 - cpx_y0) // stride
            cy1 = cy0 + (rh_vp - 1) // stride + 1
            cx0 = (vp_px_x0 - cpx_x0) // stride
            cx1 = cx0 + (rw_vp - 1) // stride + 1
            rgb = _compose_rgb(entry["layers"], entry["p1p99"],
                               brightness, cy1 - cy0, cx1 - cx0, channels,
                               crop=(cy0, cy1, cx0, cx1))
            return _encode_overlay_jpeg(rgb, vp_px_x0, vp_px_y0,
                                        rw_vp, rh_vp, img_opacity), " (cached)"

    # ── Path 3: tile read ─────────────────────────────────────────────────
    # Prefetch region (Change 1: 15% instead of 50%)
    pad_x = int(rw_vp * _PREFETCH_FRAC)
    pad_y = int(rh_vp * _PREFETCH_FRAC)
    px_x0 = max(0, vp_px_x0 - pad_x);  px_x1 = min(W, vp_px_x1 + pad_x)
    px_y0 = max(0, vp_px_y0 - pad_y);  px_y1 = min(H, vp_px_y1 + pad_y)
    rw, rh  = px_x1 - px_x0, px_y1 - px_y0
    out_h   = (rh - 1) // stride + 1
    out_w   = (rw - 1) // stride + 1

    def _do_tile_read():
        """Read tiles for the prefetch region, update render cache, queue hires result."""
        # Change 2: parallel channel reads (each uses its own zarr handle)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(channels), 4)) as ex:
            futs    = {ex.submit(_read_one_channel, ch, handles_info,
                                 px_y0, px_y1, px_x0, px_x1,
                                 stride, out_h, out_w): ch
                       for ch in channels}
            results = {futs[f]: f.result() for f in concurrent.futures.as_completed(futs)}

        layers = [results[ch][0] for ch in channels]
        p1p99  = [results[ch][1] for ch in channels]

        with _morph_render_lock:
            _morph_render_cache[cache_key] = {
                "px": (px_x0, px_y0, px_x1, px_y1),
                "layers": layers, "p1p99": p1p99,
            }
            if len(_morph_render_cache) > 8:
                del _morph_render_cache[next(iter(_morph_render_cache))]

        # Crop to exact viewport and encode
        cy0 = (vp_px_y0 - px_y0) // stride;  cy1 = cy0 + (rh_vp - 1) // stride + 1
        cx0 = (vp_px_x0 - px_x0) // stride;  cx1 = cx0 + (rw_vp - 1) // stride + 1
        rgb = _compose_rgb(layers, p1p99, brightness, cy1 - cy0, cx1 - cx0, channels,
                           crop=(cy0, cy1, cx0, cx1))
        img = _encode_overlay_jpeg(rgb, vp_px_x0, vp_px_y0, rw_vp, rh_vp, img_opacity)
        if data_dir != core.DATA.get("data_dir"):
            return   # dataset was switched while reading — do not show the old slide
        _morph_hires_queue.append(img)   # picked up by push_hires_overlay callback

    # ── Change 4: progressive rendering ───────────────────────────────────
    # If any overview is available, return it immediately (< 20 ms) and
    # start a background thread to fetch the full-resolution tiles.
    preview = None
    if ov_med:
        preview = _overview_to_image(ov_med, channels, vp_px_x0, vp_px_y0,
                                     vp_px_x1, vp_px_y1, rw_vp, rh_vp,
                                     brightness, img_opacity)
    elif ov_coarse:
        preview = _overview_to_image(ov_coarse, channels, vp_px_x0, vp_px_y0,
                                     vp_px_x1, vp_px_y1, rw_vp, rh_vp,
                                     brightness, img_opacity)

    if preview:
        threading.Thread(target=_do_tile_read, daemon=True).start()
        return preview, " (preview → loading…)"

    # No overview available yet — read tiles synchronously (first-ever render)
    _do_tile_read()
    # The result was queued by _do_tile_read; pop it and return directly
    if _morph_hires_queue:
        img = _morph_hires_queue.pop()
        return img, ""
    return None, ""
