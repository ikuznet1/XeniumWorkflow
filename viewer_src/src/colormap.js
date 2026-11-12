// Colour helpers: perceptual colormaps as 256-entry LUTs, hex parsing, RGBA buffer fill.

const STOPS = {
  // 11 evenly spaced anchors of the matplotlib maps; linearly interpolated to 256 entries.
  viridis: ['#440154', '#482576', '#414487', '#35608d', '#2a788e', '#21908c', '#22a884', '#43bf71',
    '#7ad151', '#bbdf27', '#fde725'],
  plasma: ['#0d0887', '#41049d', '#6a00a8', '#8f0da4', '#b12a90', '#cc4678', '#e16462', '#f1844b',
    '#fca636', '#fcce25', '#f0f921'],
  magma: ['#000004', '#140e36', '#3b0f70', '#641a80', '#8c2981', '#b73779', '#de4968', '#f7705c',
    '#fe9f6d', '#fecf92', '#fcfdbf'],
  cividis: ['#00224e', '#083370', '#35456c', '#4f576c', '#666970', '#7d7c78', '#948e77', '#aea371',
    '#c8b866', '#e5cf52', '#fee838'],
  // diverging (ColorBrewer RdBu reversed: low blue, middle white, high red), for values centred at 0
  rdbu: ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', '#fddbc7', '#f4a582',
    '#d6604d', '#b2182b', '#67001f'],
};

export const MISSING_RGB = [128, 128, 128];

export const TX_PALETTE = ['#ff4d4d', '#33d6ff', '#ffe14d', '#7dff5c', '#ff7df2', '#ff9a3d', '#9d8cff',
  '#3dffc5', '#ffffff', '#c7ff3d', '#ff6fa3', '#5c9dff', '#ffc9a3', '#b3ffe6', '#d9a3ff', '#a3a3a3'];

const RGB_FN = /^rgba?\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)/i;

/** '#rgb', '#rrggbb' or 'rgb(r, g, b)' / 'rgba(…)' (as in Plotly's qualitative palettes) → [r, g, b]. */
export function hexToRgb(hex) {
  let h = String(hex || '').trim();
  const m = RGB_FN.exec(h);
  if (m) return [m[1], m[2], m[3]].map((v) => Math.max(0, Math.min(255, Math.round(+v))));
  h = h.replace(/^#/, '');
  if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
  const v = parseInt(h, 16);
  if (h.length !== 6 || Number.isNaN(v)) return [128, 128, 128];
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

const lutCache = {};

/** @returns {Uint8Array} 256*3 RGB */
export function colormapLut(name) {
  const key = STOPS[name] ? name : 'viridis';
  if (lutCache[key]) return lutCache[key];
  const stops = STOPS[key].map(hexToRgb);
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (stops.length - 1);
    const a = Math.min(stops.length - 2, Math.floor(t));
    const f = t - a;
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(stops[a][c] + (stops[a + 1][c] - stops[a][c]) * f);
  }
  lutCache[key] = lut;
  return lut;
}

export function colormapCss(name) {
  const key = STOPS[name] ? name : 'viridis';
  return `linear-gradient(to right, ${STOPS[key].join(',')})`;
}

// Typed arrays are little-endian on every platform browsers run on, so a Uint32 view of an
// RGBA byte buffer packs as 0xAABBGGRR.
const pack = (r, g, b, a) => ((a << 24) | (b << 16) | (g << 8) | r) >>> 0;

/**
 * Fill `rgba` (Uint8Array 4n) from continuous values. NaN → grey.
 */
export function fillContinuous(rgba, values, n, min, max, cmap) {
  const out = new Uint32Array(rgba.buffer, rgba.byteOffset, n);
  const lut = colormapLut(cmap);
  const lut32 = new Uint32Array(256);
  for (let i = 0; i < 256; i++) lut32[i] = pack(lut[i * 3], lut[i * 3 + 1], lut[i * 3 + 2], 255);
  const missing = pack(MISSING_RGB[0], MISSING_RGB[1], MISSING_RGB[2], 255);
  const span = max - min;
  const k = span > 0 ? 255.999 / span : 0;
  for (let i = 0; i < n; i++) {
    const v = values[i];
    if (v !== v) {
      out[i] = missing;
    } else {
      const t = (v - min) * k;
      out[i] = lut32[t <= 0 ? 0 : t >= 255 ? 255 : t | 0];
    }
  }
}

/**
 * Fill `rgba` from categorical codes. 65535 / out of range → grey; hidden categories → alpha 0.
 * @param {Uint8Array} hiddenMask one byte per category (1 = hidden)
 */
export function fillCategorical(rgba, codes, n, colors, hiddenMask) {
  const out = new Uint32Array(rgba.buffer, rgba.byteOffset, n);
  const k = colors.length;
  const pal = new Uint32Array(k);
  for (let c = 0; c < k; c++) {
    const [r, g, b] = hexToRgb(colors[c]);
    pal[c] = hiddenMask && hiddenMask[c] ? 0 : pack(r, g, b, 255);
  }
  const missing = pack(MISSING_RGB[0], MISSING_RGB[1], MISSING_RGB[2], 255);
  for (let i = 0; i < n; i++) {
    const c = codes[i];
    out[i] = c < k ? pal[c] : missing;
  }
}

/**
 * RGBA image of a density grid: values in (0, max] through `cmap`, 0 / NaN transparent.
 * @param {Uint8ClampedArray|Uint8Array} rgba 4n bytes
 */
export function fillDensity(rgba, values, n, max, cmap = 'magma') {
  const out = new Uint32Array(rgba.buffer, rgba.byteOffset, n);
  const lut = colormapLut(cmap);
  const lut32 = new Uint32Array(256);
  for (let i = 0; i < 256; i++) lut32[i] = pack(lut[i * 3], lut[i * 3 + 1], lut[i * 3 + 2], 255);
  const k = max > 0 ? 255.999 / max : 0;
  for (let i = 0; i < n; i++) {
    const v = values[i];
    if (!(v > 0)) {
      out[i] = 0;
    } else {
      const t = v * k;
      out[i] = lut32[t >= 255 ? 255 : t | 0];
    }
  }
}

export function fillUniform(rgba, n, rgb) {
  new Uint32Array(rgba.buffer, rgba.byteOffset, n).fill(pack(rgb[0], rgb[1], rgb[2], 255));
}

/** Multiply down the alpha of every visible cell that is not in `mask`. */
export function dimUnselected(rgba, n, mask, alpha) {
  const out = new Uint32Array(rgba.buffer, rgba.byteOffset, n);
  const a = (alpha & 255) << 24;
  for (let i = 0; i < n; i++) {
    if (!mask[i] && out[i] >>> 24) out[i] = ((out[i] & 0x00ffffff) | a) >>> 0;
  }
}
