// Viv PixelSource for one pyramid level of a set of single-channel Xenium OME-TIFFs
// that share a pixel grid. Each channel is its own file; `selection.c` picks the file.
// A tile is one HTTP Range request for the compressed JPEG 2000 codestream, decoded in
// the worker pool, cropped to the image extent and kept in a byte-bounded LRU cache.

import { ABORTED } from "./pool.js";

/** Byte-bounded LRU for decoded tiles (Map keeps insertion order). */
export class TileCache {
  constructor(maxBytes) {
    this.maxBytes = maxBytes;
    this.bytes = 0;
    this.map = new Map();
  }

  get(key) {
    const v = this.map.get(key);
    if (v) {
      this.map.delete(key);
      this.map.set(key, v);
    }
    return v;
  }

  set(key, tile) {
    const size = tile.data.byteLength;
    if (size > this.maxBytes) return;
    const old = this.map.get(key);
    if (old) {
      this.bytes -= old.data.byteLength;
      this.map.delete(key);
    }
    this.map.set(key, tile);
    this.bytes += size;
    for (const [k, v] of this.map) {
      if (this.bytes <= this.maxBytes) break;
      this.map.delete(k);
      this.bytes -= v.data.byteLength;
    }
  }

  clear() {
    this.map.clear();
    this.bytes = 0;
  }
}

const isAbort = (err, signal) => err === ABORTED
  || (err && err.name === "AbortError")
  || (signal && signal.aborted);

export class XeniumPixelSource {
  /**
   * @param {object} o
   * @param {object[]} o.files   per channel: { url, levels, open() }; `levels` is filled by
   *                              open() (headers of a channel are read on first use)
   * @param {object} o.dims      { width, height, tileWidth, cols, rows } of this level
   * @param {number} o.level     pyramid level this source serves
   * @param {import('./pool.js').DecoderPool} o.pool
   * @param {TileCache} o.cache
   * @param {object} o.stats     shared counters (mutated)
   * @param {string} o.cachePrefix
   */
  constructor({ files, dims, level, pool, cache, stats, cachePrefix = "" }) {
    this.files = files;
    this.level = level;
    this.pool = pool;
    this.cache = cache;
    this.stats = stats;
    this.cachePrefix = cachePrefix;
    const lv = dims;
    this.cols = lv.cols;
    this.rows = lv.rows;
    this.width = lv.width;
    this.height = lv.height;
    // --- Viv PixelSource interface ---
    this.tileSize = lv.tileWidth;
    this.dtype = "Uint16";
    this.labels = ["c", "y", "x"];
    this.shape = [files.length, lv.height, lv.width];
    this.meta = { physicalSizes: {} };
  }

  onTileError(err) {
    // eslint-disable-next-line no-console
    console.error("[xenium-image] tile failed:", err);
  }

  /** Clipped pixel size of tile (x, y); zero when the tile lies outside this level. */
  tileExtent(x, y) {
    const w = Math.min(this.tileSize, this.width - x * this.tileSize);
    const h = Math.min(this.tileSize, this.height - y * this.tileSize);
    return (x < 0 || y < 0 || w <= 0 || h <= 0) ? { width: 0, height: 0 } : { width: w, height: h };
  }

  async getTile({ x, y, selection, signal }) {
    const c = (selection && selection.c) || 0;
    if (!this.files[c]) throw new Error(`no such image channel: ${c}`);
    const { width, height } = this.tileExtent(x, y);
    // deck.gl derives tile indices from the full-resolution extent; because level sizes
    // are floor-halved it can ask for a tile just past the edge of a coarse level.
    if (!width || !height) return { data: new Uint16Array(0), width: 0, height: 0 };
    const key = `${this.cachePrefix}${c}/${this.level}/${x}/${y}`;
    const hit = this.cache.get(key);
    if (hit) {
      this.stats.cacheHits++;
      return hit;
    }
    if (signal && signal.aborted) {
      this.stats.aborted++;
      throw ABORTED;
    }
    const file = await this.files[c].open(); // no-op once the channel's headers are in
    if (signal && signal.aborted) {
      this.stats.aborted++;
      throw ABORTED;
    }
    const lv = file.levels[this.level];
    const idx = y * lv.cols + x;
    const offset = lv.offsets[idx];
    const length = lv.byteCounts[idx];
    let tile;
    if (!length) {
      tile = { data: new Uint16Array(width * height), width, height }; // sparse tile
    } else {
      try {
        const t0 = performance.now();
        this.stats.requests++;
        this.stats.inFlight++;
        let buffer;
        try {
          const resp = await fetch(file.url, {
            headers: { Range: `bytes=${offset}-${offset + length - 1}` },
            signal,
          });
          if (resp.status !== 206) {
            // A 200 here would be the whole (multi-hundred-MB) file: refuse to read it.
            if (resp.body) resp.body.cancel().catch(() => {});
            throw new Error(`image server ignored the Range request (HTTP ${resp.status}) for ${file.url}`);
          }
          buffer = await resp.arrayBuffer();
        } finally {
          this.stats.inFlight--;
        }
        if (buffer.byteLength !== length) {
          throw new Error(`short read for ${file.url}: ${buffer.byteLength} of ${length} bytes`);
        }
        this.stats.bytes += buffer.byteLength;
        this.stats.fetchMs.push(performance.now() - t0);
        if (this.stats.fetchMs.length > 500) this.stats.fetchMs.shift();
        const dec = await this.pool.decode(buffer, { cropWidth: width, cropHeight: height, signal });
        if (dec.width !== width || dec.height !== height) {
          throw new Error(`decoded tile is ${dec.width}x${dec.height}, expected ${width}x${height}`);
        }
        tile = { data: dec.data, width, height };
        this.stats.tiles++;
      } catch (err) {
        if (isAbort(err, signal)) {
          this.stats.aborted++;
          throw ABORTED; // Viv turns this into an empty (re-loadable) tile
        }
        this.stats.errors++;
        throw err;
      }
    }
    this.cache.set(key, tile);
    return tile;
  }

  /** Whole level as one array (Viv uses it for the coarsest level as a backdrop). */
  async getRaster({ selection, signal } = {}) {
    const lv = this;
    const { width, height, tileSize } = this;
    const out = new Uint16Array(width * height);
    const jobs = [];
    for (let ty = 0; ty < lv.rows; ty++) {
      for (let tx = 0; tx < lv.cols; tx++) {
        jobs.push(this.getTile({ x: tx, y: ty, selection, signal }).then((t) => {
          for (let r = 0; r < t.height; r++) {
            out.set(t.data.subarray(r * t.width, (r + 1) * t.width), (ty * tileSize + r) * width + tx * tileSize);
          }
        }));
      }
    }
    await Promise.all(jobs);
    return { data: out, width, height };
  }
}
