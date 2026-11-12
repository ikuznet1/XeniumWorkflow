// Morphology-image layer for the WebGL Xenium viewer.
//
// The browser reads Xenium's pyramidal, JPEG 2000 compressed OME-TIFFs directly over
// HTTP Range requests (headers via geotiff.js, tile payloads via plain fetch), decodes
// tiles in a Web Worker pool (OpenJPEG) and renders them with Viv's MultiscaleImageLayer
// (GPU channel mixing with per-channel colour and contrast limits).
//
//   const ctl = await openImages({ baseUrl: "", images: meta.images, pixelSize: meta.pixel_size });
//   viewer.setExtraLayers(() => ctl.layers(state));

import { Matrix4 } from "@math.gl/core";
import { MultiscaleImageLayer } from "@vivjs/layers";
import { ColorPaletteExtension } from "@vivjs/extensions";
import { DecoderPool, defaultPoolSize } from "./pool.js";
import { openPyramid } from "./tiff.js";
import { TileCache, XeniumPixelSource } from "./source.js";

export const WORKER_FILE = "xenium_image_worker.js";
export const MAX_CHANNELS = 6; // rendered at once (Viv allows more; four exist in Xenium data)

// URL of the script this module was bundled into (the worker and the .wasm sit next to it).
const SCRIPT_URL = (typeof document !== "undefined" && document.currentScript && document.currentScript.src) || null;

function defaultWorkerUrl() {
  const base = SCRIPT_URL || (typeof location !== "undefined" ? new URL("assets/", location.href).href : "");
  return new URL(WORKER_FILE, base).href;
}

// Largest level read whole (default contrast, Viv's backdrop); bigger ones are sampled / skipped.
const RASTER_BUDGET = 4e6;
const SAMPLE_AT = [1 / 6, 1 / 2, 5 / 6]; // 3 x 3 tiles

const coarsePixels = (loader) => loader[loader.length - 1].width * loader[loader.length - 1].height;

function percentileLimits(data, lo = 0.01, hi = 0.999) {
  // Exact histogram over the uint16 range; zeros are "no data" padding in Xenium images.
  const hist = new Uint32Array(65536);
  for (let i = 0; i < data.length; i++) hist[data[i]]++;
  const n = data.length - hist[0];
  if (n <= 0) return { min: 0, max: 1 };
  const find = (q) => {
    const target = q * n;
    let acc = 0;
    for (let v = 1; v < 65536; v++) {
      acc += hist[v];
      if (acc >= target) return v;
    }
    return 65535;
  };
  const min = find(lo);
  const max = Math.max(find(hi), min + 1);
  return { min, max };
}

class ImageController {
  constructor({ groups, channels, pool, cache, stats, pixelSize, maxRequests, id }) {
    this._groups = groups; // [{ loader: XeniumPixelSource[], members: [channelIndex...] }]
    this._pool = pool;
    this._cache = cache;
    this._pixelSize = pixelSize;
    this._maxRequests = maxRequests;
    this._id = id;
    this._modelMatrix = new Matrix4().scale([pixelSize, pixelSize, 1]);
    this._extensions = [new ColorPaletteExtension()];
    this._contrast = new Map();
    this._destroyed = false;
    // Per group: the channels that are on right now. Only these are fetched for new tiles;
    // switching one back on reloads the tiles, and its decoded ones come from the tile cache.
    this._selKey = groups.map(() => "");
    this._selections = groups.map(() => []);
    this.channels = channels;
    this.stats = stats;
  }

  /** "wasm" | "js" once the first worker is up, else null. */
  get decoderBackend() {
    return this._pool.backend;
  }

  /** Codec-only decode times (ms) of the most recent tiles. */
  get decodeTimes() {
    return this._pool.decodeTimes;
  }

  get poolSize() {
    return this._pool.size;
  }

  /**
   * deck.gl layers for the current display state; put them before (beneath) the cells.
   * state = { visible, opacity, channels: [{ index, on, color:[r,g,b], min, max }] }
   */
  layers(state) {
    if (this._destroyed || !state || state.visible === false) return [];
    const opacity = Math.max(0, Math.min(1, state.opacity === undefined ? 1 : Number(state.opacity)));
    const want = new Map();
    for (const ch of state.channels || []) want.set(ch.index, ch);
    const out = [];
    let shown = 0;
    this._groups.forEach((group, gi) => {
      const active = [];
      for (let local = 0; local < group.members.length && active.length < MAX_CHANNELS; local++) {
        const st = want.get(group.members[local]);
        if (st && st.on) active.push(local);
      }
      if (!active.length) return;
      const key = active.join(",");
      if (key !== this._selKey[gi]) {
        // New array identity only when the set changes: Viv reloads tiles when it does
        // (already decoded channels come straight back from the tile cache).
        this._selKey[gi] = key;
        this._selections[gi] = active.map((c) => ({ c }));
      }
      const colors = [];
      const contrastLimits = [];
      const channelsVisible = [];
      for (const local of active) {
        const st = want.get(group.members[local]);
        const min = Number.isFinite(st.min) ? st.min : 0;
        const max = Number.isFinite(st.max) ? st.max : 65535;
        colors.push(st.color ? st.color.slice(0, 3) : [255, 255, 255]);
        contrastLimits.push([min, Math.max(max, min + 1e-6)]);
        channelsVisible.push(true);
      }
      shown++;
      out.push(new MultiscaleImageLayer({
        id: `${this._id}-${gi}`,
        loader: group.loader,
        selections: this._selections[gi],
        colors,
        contrastLimits,
        channelsVisible,
        dtype: "Uint16",
        opacity,
        modelMatrix: this._modelMatrix,
        extensions: this._extensions,
        pickable: false,
        maxRequests: this._maxRequests,
        // Viv draws the coarsest level as a backdrop under the tiles; with a translucent
        // layer that backdrop would blend through the sharp tiles, so drop it then.
        // A file without a small overview would make that backdrop a read of every tile.
        excludeBackground: opacity < 1 || coarsePixels(group.loader) > RASTER_BUDGET,
      }));
    });
    return shown ? out : [];
  }

  /** Robust display range from a low-resolution level: 1st - 99.9th percentile of non-zero pixels. */
  defaultContrast(index) {
    if (this._contrast.has(index)) return this._contrast.get(index);
    const p = (async () => {
      for (const group of this._groups) {
        const local = group.members.indexOf(index);
        if (local < 0) continue;
        // Finest level with at most ~1.5 Mpx (two tiles or fewer for a Xenium slide).
        let L = group.loader.length - 1;
        for (let i = 0; i < group.loader.length; i++) {
          if (group.loader[i].width * group.loader[i].height <= 1.5e6) {
            L = i;
            break;
          }
        }
        const lv = group.loader[L];
        if (lv.width * lv.height <= RASTER_BUDGET) {
          const raster = await lv.getRaster({ selection: { c: local } });
          return percentileLimits(raster.data);
        }
        // No small level (a file without a pyramid): estimate from a few tiles spread over
        // the image instead of reading all of it.
        const picks = new Map();
        for (const fy of SAMPLE_AT) {
          for (const fx of SAMPLE_AT) {
            const x = Math.min(lv.cols - 1, Math.floor(fx * lv.cols));
            const y = Math.min(lv.rows - 1, Math.floor(fy * lv.rows));
            picks.set(`${x}/${y}`, { x, y });
          }
        }
        const tiles = await Promise.all([...picks.values()].map(({ x, y }) => lv.getTile({ x, y, selection: { c: local } })));
        const sample = new Uint16Array(tiles.reduce((a, t) => a + t.data.length, 0));
        let at = 0;
        for (const t of tiles) {
          sample.set(t.data, at);
          at += t.data.length;
        }
        return percentileLimits(sample);
      }
      throw new Error(`no such image channel: ${index}`);
    })();
    p.catch(() => this._contrast.delete(index));
    this._contrast.set(index, p);
    return p;
  }

  destroy() {
    this._destroyed = true;
    if (this._preloadTimer) clearTimeout(this._preloadTimer);
    this._pool.destroy();
    this._cache.clear();
  }
}

/**
 * Open the image pyramids (headers only) and return a controller.
 *
 * @param {object} o
 * @param {string} [o.baseUrl=""]   prefix for `/viewer/image/<index>`
 * @param {{index:number,name:string}[]} o.images   from GET /viewer/meta
 * @param {number} [o.pixelSize]    µm per full-resolution pixel (default: OME PhysicalSizeX, else 0.2125)
 * @param {string} [o.workerUrl]    URL of xenium_image_worker.js (default: next to the current script)
 * @param {string} [o.wasmUrl]      URL of openjpegwasm_decode.wasm (default: next to the worker)
 * @param {number} [o.poolSize]     decode workers (default: hardwareConcurrency - 1, 1..6)
 * @param {number} [o.cacheBytes]   decoded-tile LRU size in bytes (default 128 MB)
 * @param {number} [o.maxRequests]  concurrent tile loads before deck.gl aborts off-screen ones (default 8)
 * @param {(image:{index:number,name:string}) => string} [o.imageUrl]  override the URL scheme
 * @param {boolean} [o.eager=false] read the headers of every file before resolving. By default, when
 *                                  the first file's OME-XML describes all files as channels of one image
 *                                  (Xenium's multi-file layout), only the first file is opened up front and
 *                                  the others on first use / in the background after `preloadDelay` ms.
 * @param {number|null} [o.preloadDelay=1500]  null: never open unused channels in the background
 * @param {string} [o.id]           deck.gl layer id prefix
 */
export async function openImages({
  baseUrl = "", images, pixelSize, workerUrl, wasmUrl, poolSize, cacheBytes = 128 * 1024 * 1024,
  maxRequests = 8, imageUrl, forceJs = false, eager = false, preloadDelay = 1500, id = "xenium-morphology",
} = {}) {
  if (!images || !images.length) throw new Error("openImages: no images");
  const root = String(baseUrl || "").replace(/\/+$/, "");
  const urlOf = imageUrl || ((im) => `${root}/viewer/image/${im.index}`);
  const stats = {
    requests: 0, bytes: 0, tiles: 0, aborted: 0, errors: 0, cacheHits: 0, inFlight: 0, fetchMs: [],
    openMs: images.map(() => null),
  };

  // One entry per file / channel; open() reads its headers once (retried after a failure).
  let reference = null; // level sizes every lazily opened file must match
  const files = images.map((im, pos) => {
    const f = {
      image: im,
      // Absolute URL: the same string is used by fetch() here and by geotiff.
      url: new URL(urlOf(im), typeof location !== "undefined" ? location.href : undefined).href,
      levels: null,
      ome: null,
      promise: null,
    };
    f.open = () => {
      if (!f.promise) {
        f.promise = openPyramid(f.url).then((p) => {
          const sig = p.levels.map((l) => `${l.width}x${l.height}`).join(",");
          if (reference !== null && sig !== reference) {
            throw new Error(`${im.name}: pyramid ${sig} differs from the first image (${reference})`);
          }
          f.levels = p.levels;
          f.ome = p.ome;
          stats.openMs[pos] = p.openMs;
          return f;
        });
        f.promise.catch(() => { f.promise = null; });
      }
      return f.promise;
    };
    return f;
  });

  await files[0].open();
  const ome0 = files[0].ome;
  // Xenium multi-file layout: every file's OME-XML lists all channels and which file holds
  // which; all channels of an OME image share one pixel grid, so one header is enough to start.
  const chanOf = images.map((im) => ome0.fileChannel[im.name]);
  const lazy = !eager && files.length > 1 && ome0.sizeC === files.length
    && chanOf.every((c) => Number.isInteger(c) && c >= 0 && c < files.length)
    && new Set(chanOf).size === files.length;
  if (lazy) reference = files[0].levels.map((l) => `${l.width}x${l.height}`).join(",");
  else await Promise.all(files.map((f) => f.open()));

  const pool = new DecoderPool({
    workerUrl: workerUrl || defaultWorkerUrl(), wasmUrl, size: poolSize || defaultPoolSize(), forceJs,
  });
  const cache = new TileCache(cacheBytes);

  const fallbackLabel = (f) => String(f.image.name || `image ${f.image.index}`).replace(/\.ome\.tiff?$/i, "");
  const labelOf = (f, pos) => {
    if (lazy) return ome0.channelNames[chanOf[pos]] || fallbackLabel(f);
    const { channelNames, fileChannel } = f.ome;
    let c = fileChannel[f.image.name];
    if (c === undefined) c = channelNames.length === files.length ? pos : 0;
    return channelNames[c] || fallbackLabel(f);
  };

  // Files that share a pixel grid are mixed in one layer (normally: all of them).
  const groupMap = new Map();
  files.forEach((f, pos) => {
    const key = (f.levels || files[0].levels).map((l) => `${l.width}x${l.height}`).join(",");
    if (!groupMap.has(key)) groupMap.set(key, []);
    groupMap.get(key).push(pos);
  });
  const groups = [];
  const levelsOf = [];
  let gi = 0;
  for (const members of groupMap.values()) {
    const gfiles = members.map((pos) => files[pos]);
    const dims = (gfiles[0].levels || files[0].levels);
    const loader = dims.map((d, level) => new XeniumPixelSource({
      files: gfiles, dims: d, level, pool, cache, stats, cachePrefix: `${gi}:`,
    }));
    members.forEach((pos) => { levelsOf[pos] = dims; });
    groups.push({ loader, members: members.map((pos) => files[pos].image.index) });
    gi++;
  }

  const channels = files.map((f, pos) => ({
    index: f.image.index,
    name: f.image.name,
    label: labelOf(f, pos),
    dtype: "Uint16",
    width: levelsOf[pos][0].width,
    height: levelsOf[pos][0].height,
    levels: levelsOf[pos].length,
  }));

  const px = Number(pixelSize) || ome0.physicalSizeX || 0.2125;
  const ctl = new ImageController({ groups, channels, pool, cache, stats, pixelSize: px, maxRequests, id });
  if (lazy && preloadDelay !== null && typeof setTimeout !== "undefined") {
    ctl._preloadTimer = setTimeout(() => {
      if (!ctl._destroyed) files.forEach((f) => f.open().catch(() => {}));
    }, preloadDelay);
  }
  return ctl;
}
