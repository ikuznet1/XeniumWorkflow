"""Binary data endpoints for the WebGL (deck.gl) spatial viewer.

The browser keeps cell coordinates on the GPU and asks the server only for what
changes: one value per cell on a recolour, and boundaries / transcripts for the
current viewport. Boundaries and transcripts are read lazily from the tiled stores
that Xenium Onboard Analysis writes for its own viewer (cells.zarr.zip and
transcripts.zarr.zip), so nothing here loads a whole table into memory.

Wire format of every ``*.bin`` response (little-endian):

    uint32  header_len
    bytes   header JSON (utf-8), padded with spaces to a multiple of 4 bytes
    bytes   the arrays listed in header["arrays"], back to back, each padded to 4 bytes

``header["arrays"]`` is a list of ``{"name", "dtype", "length"}``; everything else in
the header is endpoint-specific metadata.
"""
from __future__ import annotations

import json
import math
import os
import threading
import zlib
from collections import OrderedDict

import numpy as np

_DTYPES = {"float32": "<f4", "uint32": "<u4", "uint16": "<u2", "uint8": "|u1", "int32": "<i4"}


def pack(meta: dict, arrays: dict) -> bytes:
    """Serialise *meta* plus named numpy arrays into the wire format above."""
    listing, chunks = [], []
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        dt = arr.dtype.name
        if dt not in _DTYPES:
            raise TypeError(f"unsupported dtype for {name}: {dt}")
        raw = arr.astype(_DTYPES[dt], copy=False).tobytes()
        listing.append({"name": name, "dtype": dt, "length": int(arr.size)})
        chunks.append(raw + b"\0" * (-len(raw) % 4))
    header = json.dumps({**meta, "arrays": listing}, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 4)
    return b"".join([np.uint32(len(header)).tobytes(), header, *chunks])


class XeniumTileStore:
    """Lazy reader over one Xenium output bundle's cells.zarr.zip / transcripts.zarr.zip."""

    POLYGON_SET = {"nucleus": 0, "cell": 1}
    TILE_CACHE = 24          # decoded transcript tiles kept in memory (at most)
    TILE_CACHE_BYTES = 192 * 1024 * 1024   # ... and at most this many bytes of them
    MAX_TILES = 12           # tiles read per transcript request (sets the level of detail)

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._lock = threading.Lock()
        self._cells = None
        self._tx = None
        self._tx_meta = None
        self._poly = {}
        self._tiles: OrderedDict = OrderedDict()
        self._tile_bytes = 0
        self._cell_ids = False

    # ── stores ────────────────────────────────────────────────────────────────
    def _open(self, name):
        import zarr
        path = os.path.join(self.data_dir, name)
        if not os.path.exists(path):
            return None
        return zarr.open_group(zarr.storage.ZipStore(path, mode="r"), mode="r")

    @property
    def has_boundaries(self) -> bool:
        return os.path.exists(os.path.join(self.data_dir, "cells.zarr.zip"))

    @property
    def has_transcripts(self) -> bool:
        return os.path.exists(os.path.join(self.data_dir, "transcripts.zarr.zip"))

    # ── boundaries ────────────────────────────────────────────────────────────
    def _polygons(self, kind: str):
        """(vertices (n, V, 2) float32, n_vertices int32, cell_index uint32, bbox (n, 4)) for
        one polygon set, read once per bundle (about 200 bytes per cell)."""
        with self._lock:
            if kind in self._poly:
                return self._poly[kind]
            if self._cells is None:
                self._cells = self._open("cells.zarr.zip")
            if self._cells is None:
                self._poly[kind] = None
                return None
            try:
                names = list(self._cells.attrs.get("polygon_set_names", ["nucleus", "cell"]))
                si = names.index(kind) if kind in names else self.POLYGON_SET[kind]
                if "polygon_sets" in self._cells:
                    grp = self._cells[f"polygon_sets/{si}"]
                    verts = np.asarray(grp["vertices"][:], dtype=np.float32)
                    nv = np.asarray(grp["num_vertices"][:], dtype=np.int32).reshape(-1)
                    cidx = np.asarray(grp["cell_index"][:], dtype=np.uint32).reshape(-1)
                else:
                    # older cells.zarr.zip: root arrays (sets, cells, 2 V) / (sets, cells),
                    # one polygon per cell in cell order
                    verts = np.asarray(self._cells["polygon_vertices"][si], dtype=np.float32)
                    nv = np.asarray(self._cells["polygon_num_vertices"][si], dtype=np.int32).reshape(-1)
                    cidx = np.arange(len(verts), dtype=np.uint32)
                verts = verts.reshape(len(verts), -1, 2)
                if len(nv) != len(verts) or len(cidx) != len(verts):
                    raise ValueError("inconsistent polygon arrays")
            except Exception as exc:   # unknown layout: the provider's in-memory polygons are used
                print(f"  viewer: no {kind} polygons readable from cells.zarr.zip ({exc!r})", flush=True)
                self._poly[kind] = None
                return None
            nv = np.clip(nv, 0, verts.shape[1])
            valid = np.arange(verts.shape[1])[None, :] < nv[:, None]
            big = np.float32(np.inf)
            xs, ys = verts[..., 0], verts[..., 1]
            bbox = np.stack([np.where(valid, xs, big).min(1), np.where(valid, ys, big).min(1),
                             np.where(valid, xs, -big).max(1), np.where(valid, ys, -big).max(1)], 1)
            self._poly[kind] = (verts, nv, cidx, bbox)
            return self._poly[kind]

    def cell_ids(self):
        """The bundle's cell_id array in file order ((n, 2) uint32 prefix / dataset suffix,
        or (n,) integers in older bundles), or None when it cannot be read."""
        with self._lock:
            if self._cell_ids is False:
                try:
                    if self._cells is None:
                        self._cells = self._open("cells.zarr.zip")
                    self._cell_ids = (np.asarray(self._cells["cell_id"][:])
                                      if self._cells is not None else None)
                except Exception:
                    self._cell_ids = None
            return self._cell_ids

    def boundaries_bbox(self, kind, x0, y0, x1, y1, limit=20000):
        """Polygons of *kind* intersecting the box. Returns (meta, arrays) for pack()."""
        poly = self._polygons(kind)
        if poly is None:
            return {"n": 0, "truncated": False, "available": False}, {}
        verts, nv, cidx, bbox = poly
        x0, y0, x1, y1 = (float(min(max(v, -3e38), 3e38)) for v in (x0, y0, x1, y1))
        hit =np.flatnonzero((bbox[:, 2] >= x0) & (bbox[:, 0] <= x1) &
                             (bbox[:, 3] >= y0) & (bbox[:, 1] <= y1) & (nv >= 3))
        truncated = len(hit) > limit
        if truncated:
            return {"n": int(len(hit)), "truncated": True, "available": True}, {}
        nvh = nv[hit]
        mask = np.arange(verts.shape[1])[None, :] < nvh[:, None]
        flat = verts[hit][mask]                                   # (sum nv, 2), row-major
        starts = np.zeros(len(hit) + 1, dtype=np.uint32)
        np.cumsum(nvh, out=starts[1:])
        return ({"n": int(len(hit)), "truncated": False, "available": True},
                {"start": starts, "xy": flat.ravel(), "cell": cidx[hit]})

    # ── transcripts ───────────────────────────────────────────────────────────
    def _tx_open(self):
        with self._lock:
            if self._tx_meta is not None:
                return self._tx_meta
            self._tx = self._open("transcripts.zarr.zip")
            if self._tx is None:
                self._tx_meta = {}
                return self._tx_meta
            g = dict(self._tx["grids"].attrs)
            names = list(self._tx.attrs.get("gene_names", []))
            levels = int(g["number_levels"])
            # per level: the tiles that exist, as key strings and as (tx, ty) numbers
            # and how many points each holds (0 when the bundle does not say)
            keys, index, counts = [], [], []
            sizes = g.get("grid_number_objects") or []
            for lv in range(levels):
                all_keys = list(g["grid_keys"][lv])
                n_pts = list(sizes[lv]) if lv < len(sizes) and len(sizes[lv]) == len(all_keys) \
                    else [0] * len(all_keys)
                ok = [i for i, k in enumerate(all_keys) if len(str(k).split(",")) == 2]
                ks = [all_keys[i] for i in ok]
                keys.append(ks)
                index.append(np.array([[float(v) for v in k.split(",")] for k in ks],
                                      dtype=np.float64).reshape(-1, 2))
                counts.append(np.array([n_pts[i] for i in ok], dtype=np.float64))
            self._tx_meta = {
                "tile": float(g["grid_size"][0]),
                "levels": levels,
                "keys": keys,
                "index": index,
                "counts": counts,
                "gene_index": {n: i for i, n in enumerate(names)},
            }
            return self._tx_meta

    def _tiles_in(self, m, level, x0, y0, x1, y1):
        """(keys, total points) of the existing tiles of *level* that the box touches. Looks
        only at the tiles that exist, so the work does not grow with the size of the box."""
        size = m["tile"] * (2 ** level)
        t = m["index"][level] * size
        hit = np.flatnonzero((t[:, 0] <= x1) & (t[:, 0] + size > x0) &
                             (t[:, 1] <= y1) & (t[:, 1] + size > y0))
        return [m["keys"][level][i] for i in hit], float(m["counts"][level][hit].sum())

    def _tile(self, level: int, key: str):
        ck = (level, key)
        with self._lock:
            if ck in self._tiles:
                self._tiles.move_to_end(ck)
                return self._tiles[ck]
        grp = self._tx[f"grids/{level}/{key}"]
        loc = np.asarray(grp["location"][:], dtype=np.float32)
        gene = np.asarray(grp["gene_identity"][:]).reshape(-1).astype(np.uint16, copy=False)
        # Only the full-resolution level carries per-transcript quality scores.
        qv = (np.asarray(grp["quality_score"][:], dtype=np.float32).reshape(-1)
              if "quality_score" in grp else None)
        # Coarser levels hold one point per cluster of same-gene transcripts.
        cnt = (np.asarray(grp["cluster_count"][:]).reshape(-1).astype(np.uint32, copy=False)
               if "cluster_count" in grp else None)
        # Kept sorted by gene, so a request slices out its genes instead of scanning the tile.
        order = np.argsort(gene, kind="stable")
        val = (loc[order, 0], loc[order, 1], gene[order],
               None if qv is None else qv[order], None if cnt is None else cnt[order])
        nbytes = sum(a.nbytes for a in val if a is not None)
        with self._lock:
            if ck not in self._tiles:
                self._tiles[ck] = val
                self._tile_bytes += nbytes
            while len(self._tiles) > 1 and (len(self._tiles) > self.TILE_CACHE
                                            or self._tile_bytes > self.TILE_CACHE_BYTES):
                _, old = self._tiles.popitem(last=False)
                self._tile_bytes -= sum(a.nbytes for a in old if a is not None)
        return val

    def transcripts_bbox(self, x0, y0, x1, y1, genes, qv_min=20.0, max_points=400_000,
                         weights=False):
        """Transcripts of the named *genes* inside the box, at the finest level of detail
        that needs at most MAX_TILES existing tiles which together fit the tile cache.
        Returns (meta, arrays) for pack().

        Level 0 holds the transcripts themselves and is filtered by *qv_min*. A coarser
        level holds one point per cluster of same-gene transcripts of any quality; the
        header says so with ``clustered`` / ``filtered``, and with *weights* the number
        of transcripts behind each point is returned as the array ``weight``."""
        m = self._tx_open()
        if not m:
            return {"n": 0, "available": False}, {}
        wanted = [(i, m["gene_index"][g]) for i, g in enumerate(genes) if g in m["gene_index"]]
        if not wanted:
            return {"n": 0, "available": True, "level": None, "subsampled": False}, {}
        # float32 range, so comparing with the stored coordinates cannot overflow
        x0, y0, x1, y1 = (float(min(max(v, -3e38), 3e38)) for v in (x0, y0, x1, y1))
        level = m["levels"] - 1
        keys = None
        for lv in range(m["levels"]):
            cand, n_pts = self._tiles_in(m, lv, x0, y0, x1, y1)
            # few enough tiles, and small enough together to stay in the tile cache
            if len(cand) <= self.MAX_TILES and n_pts * 16 <= self.TILE_CACHE_BYTES:
                level, keys = lv, cand
                break
        if keys is None:
            keys = self._tiles_in(m, level, x0, y0, x1, y1)[0][:self.MAX_TILES]
        info = {"available": True, "level": level, "clustered": level > 0, "filtered": level == 0}
        src = np.array([s for _, s in wanted], dtype=np.int64)
        xs, ys, gs, ws = [], [], [], []
        for key in keys:
            x, y, gene, qv, cnt = self._tile(level, key)
            lo = np.searchsorted(gene, src, side="left")
            hi = np.searchsorted(gene, src, side="right")
            for (out_i, _), a, b in zip(wanted, lo, hi):
                if b <= a:
                    continue
                xg, yg = x[a:b], y[a:b]
                keep = (xg >= x0) & (xg <= x1) & (yg >= y0) & (yg <= y1)
                if qv is not None:
                    keep &= qv[a:b] >= qv_min
                k = int(np.count_nonzero(keep))
                if k:
                    xs.append(xg[keep]); ys.append(yg[keep])
                    gs.append(np.full(k, out_i, dtype=np.uint16))
                    if weights:
                        ws.append(cnt[a:b][keep] if cnt is not None else np.ones(k, dtype=np.uint32))
        if not xs:
            return {"n": 0, **info, "subsampled": level > 0}, {}
        x = np.concatenate(xs); y = np.concatenate(ys); g = np.concatenate(gs)
        w = np.concatenate(ws) if weights else None
        capped = len(x) > max_points
        if capped:
            sel = np.random.default_rng(0).choice(len(x), max_points, replace=False)
            x, y, g = x[sel], y[sel], g[sel]
            if w is not None:
                w = w[sel]
        xy = np.empty(2 * len(x), dtype=np.float32)
        xy[0::2] = x; xy[1::2] = y
        arrays = {"xy": xy, "gene": g}
        if w is not None:
            arrays["weight"] = w.astype(np.uint32, copy=False)
        return {"n": int(len(x)), **info, "subsampled": bool(level > 0 or capped)}, arrays


_stores: dict = {}
_stores_lock = threading.Lock()


def tile_store(data_dir: str) -> XeniumTileStore:
    with _stores_lock:
        if data_dir not in _stores:
            _stores[data_dir] = XeniumTileStore(data_dir)
        return _stores[data_dir]


def reset(keep=None) -> None:
    """Forget the stores of every bundle except *keep* (call when the dataset changes), so
    their tile caches, polygon arrays and open zip files are released. Nothing is closed
    here: a request still reading from a dropped store finishes, and the files close when
    the last reference goes."""
    with _stores_lock:
        for d in [d for d in _stores if d != keep]:
            _stores.pop(d, None)


def image_token(path: str) -> str:
    """Identifies one image file's contents (path, size, mtime) for use in its URL."""
    try:
        st = os.stat(path)
    except OSError:
        return ""
    return f"{st.st_mtime_ns:x}-{st.st_size:x}-{zlib.adler32(os.fsencode(path)):x}"


def register(server, provider, prefix: str = "") -> None:
    """Attach the viewer routes to a Flask *server* under ``<prefix>/viewer/*``.
    A second provider (e.g. the UMAP embedding of the same cells) can be registered
    under its own prefix; the browser viewer is pointed at it with ``baseUrl``.

    *provider* supplies the app-specific pieces:
      provider.data_dir()              -> str | None   (current Xenium bundle)
      provider.meta()                  -> dict         (JSON for /viewer/meta)
      provider.cells(source)           -> (meta dict, {"xy": float32[2n]})
      provider.colors(args)            -> (meta dict, {"values"|"codes": array})
      provider.boundaries(source, kind, x0, y0, x1, y1) -> (meta, arrays) | None
                                          (None = use the bundle's cells.zarr.zip)
      provider.image_files()           -> list[str]    (absolute paths the browser may read;
                                          meta()["images"][i] may carry "v": image_token(path),
                                          which the browser sends back as ?v= to get a
                                          response it is allowed to cache)
      provider.cell_at(source, index)  -> dict         (JSON for a clicked cell)
      provider.overlays(source, args)  -> dict         ({"polygons": [...], "lines": [...]} in µm, y down;
                                          *args* is the query string as a dict)
      provider.extra(name, args)       -> (meta, arrays) | None   (optional: app-specific binary
                                          payloads served as /viewer/extra/<name>.bin; None = 404)
    """
    from flask import Response, abort, jsonify, request, send_file

    def _bin(meta, arrays):
        resp = Response(pack(meta, arrays), mimetype="application/octet-stream")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    def _bbox():
        try:
            box = [float(request.args[k]) for k in ("x0", "y0", "x1", "y1")]
        except (KeyError, ValueError):
            abort(400)
        if not all(math.isfinite(v) for v in box) or box[2] < box[0] or box[3] < box[1]:
            abort(400)
        return box

    @server.route(prefix + "/viewer/meta", endpoint=f"{prefix}:viewer_meta")
    def viewer_meta():
        return jsonify(provider.meta())

    @server.route(prefix + "/viewer/cells.bin", endpoint=f"{prefix}:viewer_cells")
    def viewer_cells():
        return _bin(*provider.cells(request.args.get("source", "xenium")))

    @server.route(prefix + "/viewer/colors.bin", endpoint=f"{prefix}:viewer_colors")
    def viewer_colors():
        return _bin(*provider.colors(request.args.to_dict()))

    @server.route(prefix + "/viewer/boundaries.bin", endpoint=f"{prefix}:viewer_boundaries")
    def viewer_boundaries():
        kind = request.args.get("kind", "cell")
        source = request.args.get("source", "xenium")
        x0, y0, x1, y1 = _bbox()
        out = provider.boundaries(source, kind, x0, y0, x1, y1)
        if out is None:
            if kind not in XeniumTileStore.POLYGON_SET or not provider.data_dir():
                return _bin({"n": 0, "truncated": False, "available": False}, {})
            out = tile_store(provider.data_dir()).boundaries_bbox(kind, x0, y0, x1, y1)
        return _bin(*out)

    @server.route(prefix + "/viewer/transcripts.bin", endpoint=f"{prefix}:viewer_transcripts")
    def viewer_transcripts():
        x0, y0, x1, y1 = _bbox()
        genes = [g for g in request.args.get("genes", "").split(",") if g][:64]
        if not provider.data_dir() or not genes:
            return _bin({"n": 0, "available": bool(provider.data_dir())}, {})
        try:
            qv = float(request.args.get("qv", 20))
        except ValueError:
            qv = 20.0
        weights = request.args.get("weights", "0").lower() in ("1", "true")
        return _bin(*tile_store(provider.data_dir()).transcripts_bbox(x0, y0, x1, y1, genes, qv,
                                                                      weights=weights))

    @server.route(prefix + "/viewer/overlays", endpoint=f"{prefix}:viewer_overlays")
    def viewer_overlays():
        return jsonify(provider.overlays(request.args.get("source", "xenium"), request.args.to_dict()))

    @server.route(prefix + "/viewer/extra/<name>.bin", endpoint=f"{prefix}:viewer_extra")
    def viewer_extra(name):
        fn = getattr(provider, "extra", None)
        out = fn(name, request.args.to_dict()) if fn else None
        if out is None:
            abort(404)
        return _bin(*out)

    @server.route(prefix + "/viewer/cell/<int:index>", endpoint=f"{prefix}:viewer_cell")
    def viewer_cell(index):
        return jsonify(provider.cell_at(request.args.get("source", "xenium"), index))

    @server.route(prefix + "/viewer/image/<int:index>", endpoint=f"{prefix}:viewer_image")
    def viewer_image(index):
        # Only files the provider lists can be read; the browser fetches byte ranges.
        files = provider.image_files()
        if index < 0 or index >= len(files) or not os.path.isfile(files[index]):
            abort(404)
        # /viewer/image/<index> names a different file after another dataset is loaded, so
        # by default the browser must revalidate (ETag) before reusing cached bytes. Only a
        # URL that names the file's contents (?v=<image_token>) may be cached without asking.
        v = request.args.get("v")
        if v is not None and v != image_token(files[index]):
            abort(404)
        resp = send_file(files[index], mimetype="image/tiff", conditional=True,
                         max_age=3600 if v else 0)
        if not v:
            resp.headers["Cache-Control"] = "no-cache"
        return resp
