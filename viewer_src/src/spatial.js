// CPU-side spatial queries over the interleaved Float32 cell positions:
// polygon selection (lasso / rectangle) and nearest-cell lookup for click picking.

/**
 * Indices of points inside a polygon (even-odd rule, same edge test as the classic
 * ray-casting loop). Edges are bucketed into horizontal bands so each point only tests
 * the few edges that can cross its scanline; cost is ~O(n) regardless of lasso detail.
 *
 * @param {Float32Array} xy interleaved positions
 * @param {number} n number of points
 * @param {number[][]} polygon [[x, y], ...] (not closed)
 * @param {Uint8Array} [rgba] optional RGBA buffer; points with alpha 0 are skipped
 * @returns {Uint32Array}
 */
export function pointsInPolygon(xy, n, polygon, rgba) {
  const m = polygon.length;
  if (m < 3 || n === 0) return new Uint32Array(0);
  const px = new Float64Array(m);
  const py = new Float64Array(m);
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < m; i++) {
    const x = (px[i] = polygon[i][0]);
    const y = (py[i] = polygon[i][1]);
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const bands = Math.max(1, Math.min(1024, m));
  const height = maxY - minY;
  const inv = height > 0 ? bands / height : 0;
  const bandOf = (y) => {
    const b = Math.floor((y - minY) * inv);
    return b < 0 ? 0 : b >= bands ? bands - 1 : b;
  };
  // CSR listing of edges per band.
  const start = new Uint32Array(bands + 1);
  for (let i = 0, j = m - 1; i < m; j = i++) {
    const lo = bandOf(Math.min(py[i], py[j]));
    const hi = bandOf(Math.max(py[i], py[j]));
    for (let b = lo; b <= hi; b++) start[b + 1]++;
  }
  for (let b = 0; b < bands; b++) start[b + 1] += start[b];
  const fill = start.slice(0, bands);
  const ei = new Uint32Array(start[bands]);
  const ej = new Uint32Array(start[bands]);
  for (let i = 0, j = m - 1; i < m; j = i++) {
    const lo = bandOf(Math.min(py[i], py[j]));
    const hi = bandOf(Math.max(py[i], py[j]));
    for (let b = lo; b <= hi; b++) {
      ei[fill[b]] = i;
      ej[fill[b]++] = j;
    }
  }
  const out = new Uint32Array(n);
  let count = 0;
  for (let p = 0; p < n; p++) {
    const x = xy[2 * p];
    const y = xy[2 * p + 1];
    if (x < minX || x > maxX || y < minY || y > maxY) continue;
    if (rgba && rgba[4 * p + 3] === 0) continue;
    const b = bandOf(y);
    let inside = false;
    for (let e = start[b], end = start[b + 1]; e < end; e++) {
      const i = ei[e], j = ej[e];
      const yi = py[i], yj = py[j];
      if ((yi > y) !== (yj > y) && x < ((px[j] - px[i]) * (y - yi)) / (yj - yi) + px[i]) inside = !inside;
    }
    if (inside) out[count++] = p;
  }
  return out.slice(0, count);
}

/** Uniform grid over the points for nearest-neighbour lookups (built once per cells.bin). */
export class PointGrid {
  constructor(xy, n, bounds) {
    this.xy = xy;
    this.n = n;
    const [x0, y0, x1, y1] = bounds;
    const w = Math.max(x1 - x0, 1e-6);
    const h = Math.max(y1 - y0, 1e-6);
    // ~4 points per bucket on average, capped so the tables stay small.
    const cell = Math.max(Math.sqrt((w * h * 4) / Math.max(n, 1)), Math.max(w, h) / 2048);
    this.x0 = x0;
    this.y0 = y0;
    this.cell = cell;
    this.nx = Math.max(1, Math.ceil(w / cell));
    this.ny = Math.max(1, Math.ceil(h / cell));
    const nb = this.nx * this.ny;
    const start = new Uint32Array(nb + 1);
    const bucket = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      const b = this._bucket(xy[2 * i], xy[2 * i + 1]);
      bucket[i] = b;
      start[b + 1]++;
    }
    for (let b = 0; b < nb; b++) start[b + 1] += start[b];
    const fill = start.slice(0, nb);
    const order = new Uint32Array(n);
    for (let i = 0; i < n; i++) order[fill[bucket[i]]++] = i;
    this.start = start;
    this.order = order;
  }

  _bucket(x, y) {
    let bx = Math.floor((x - this.x0) / this.cell);
    let by = Math.floor((y - this.y0) / this.cell);
    bx = bx < 0 ? 0 : bx >= this.nx ? this.nx - 1 : bx;
    by = by < 0 ? 0 : by >= this.ny ? this.ny - 1 : by;
    return by * this.nx + bx;
  }

  /** Index of the nearest point within `radius` of (x, y), or -1. Alpha-0 points are skipped. */
  nearest(x, y, radius, rgba) {
    const {xy, cell, nx, ny, start, order} = this;
    const bx0 = Math.max(0, Math.floor((x - radius - this.x0) / cell));
    const bx1 = Math.min(nx - 1, Math.floor((x + radius - this.x0) / cell));
    const by0 = Math.max(0, Math.floor((y - radius - this.y0) / cell));
    const by1 = Math.min(ny - 1, Math.floor((y + radius - this.y0) / cell));
    let best = -1;
    let bestD = radius * radius;
    for (let by = by0; by <= by1; by++) {
      for (let bx = bx0; bx <= bx1; bx++) {
        const b = by * nx + bx;
        for (let k = start[b], end = start[b + 1]; k < end; k++) {
          const i = order[k];
          if (rgba && rgba[4 * i + 3] === 0) continue;
          const dx = xy[2 * i] - x;
          const dy = xy[2 * i + 1] - y;
          const d = dx * dx + dy * dy;
          if (d <= bestD) {
            bestD = d;
            best = i;
          }
        }
      }
    }
    return best;
  }
}
