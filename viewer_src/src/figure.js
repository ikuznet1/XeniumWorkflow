// Publication figure of the current view: a display list ("scene") built from plain view
// data, and two writers for it, SVG and PDF. Nothing here touches the DOM or WebGL, so the
// whole module also runs under node (see test/export_checks.mjs); the viewer supplies the
// view data and the raster images (Viewer.exportFigure in index.js).
//
// Figure units are the CSS pixels of the view: the map occupies [0, width] x [0, height]
// and a point (x, y) in µm lands at ((x - target.x) * scale + width / 2, ...), exactly as
// on screen. The legend is laid out in a margin to the right of the map.
import {colormapLut, hexToRgb} from './colormap.js';

export const DEFAULT_MAX_VECTOR = 50000;

// ── text metrics (Helvetica, characters 32 … 126, 1/1000 em) ──────────────────────────────
const HELVETICA = [
  278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
  556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
  1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
  667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
  333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
  556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584,
];
const BOLD_FACTOR = 1.08; // Helvetica-Bold is about this much wider on average

/** Width of `text` set in Helvetica at `size` (approximate for bold and non-ASCII text). */
export function textWidth(text, size, bold = false) {
  let w = 0;
  const s = String(text);
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    w += c >= 32 && c <= 126 ? HELVETICA[c - 32] : 556;
  }
  return (w / 1000) * size * (bold ? BOLD_FACTOR : 1);
}

function clipText(text, width, size, bold) {
  text = String(text);
  if (textWidth(text, size, bold) <= width) return text;
  while (text.length > 1 && textWidth(text + '…', size, bold) > width) text = text.slice(0, -1);
  return text + '…';
}

export function fmtNum(v) {
  if (v === null || v === undefined || !Number.isFinite(v)) return '';
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.01 || a >= 1e5)) return v.toExponential(2);
  return String(Math.round(v * 100) / 100);
}

// ── scale bar ─────────────────────────────────────────────────────────────────────────────
/**
 * A round bar length (1, 2 or 5 x 10^k µm) at most `fraction` of the view's width.
 * @param {number} umPerPx µm per figure pixel
 * @param {number} widthPx width of the map
 * @returns {{um: number, px: number, label: string}}
 */
export function scaleBarLength(umPerPx, widthPx, fraction = 0.2) {
  const raw = umPerPx * widthPx * fraction;
  if (!(raw > 0) || !Number.isFinite(raw)) return {um: 0, px: 0, label: ''};
  const exp = Math.floor(Math.log10(raw));
  const mag = Math.pow(10, exp);
  // 1, 2 or 5 times a power of ten, rounded so that 10^-1 * 2 is 0.2 and not 0.2000…04
  const round = (v) => (exp < 0 ? +v.toFixed(-exp) : v);
  const um = round([5, 2, 1].map((m) => m * mag).find((v) => v <= raw * (1 + 1e-9)) || mag);
  const label = um >= 1000 ? `${fmtNum(um / 1000)} mm` : `${fmtNum(um)} µm`;
  return {um, px: um / umPerPx, label};
}

// ── what goes in as vectors ───────────────────────────────────────────────────────────────
const cellRadiusPx = (view) => {
  const c = view.cells;
  return Math.max(c.minPx, Math.min(c.maxPx, c.radiusUm * view.scale));
};

/**
 * Counts what is in view and decides, per layer, between vector shapes and an embedded
 * image: a layer with more than `maxVector` elements in view is listed in `raster`.
 * view = {width, height, scale (px per µm), target: [x, y], background,
 *   cells: {n, xy, rgba, radiusUm, minPx, maxPx, opacity} | null,
 *   boundaries: [{kind, label, color: [r, g, b, a], n, start, xy, closed}],
 *   transcripts: {n, xy, rgba, radiusPx} | null,
 *   overlays: {lines, polygons, points}, labels: [{x, y, text, color}],
 *   legend: [{kind, title, rows: [{name, color}]} | {kind, title, range: [min, max], cmap}]}
 */
export function planFigure(view, {maxVector = DEFAULT_MAX_VECTOR} = {}) {
  const lim = Number.isFinite(+maxVector) && +maxVector >= 0 ? Math.floor(+maxVector) : DEFAULT_MAX_VECTOR;
  const {width: w, height: h, scale: s} = view;
  const tx = view.target[0], ty = view.target[1];
  const plan = {maxVector: lim, raster: [], counts: {cells: 0, transcripts: 0, boundaries: {}}, cells: null, transcripts: null, boundaries: {}};
  const inView = (xy, n, pad, rgba) => {
    const idx = new Uint32Array(n);
    let k = 0;
    const x0 = tx - (w / 2 + pad) / s, x1 = tx + (w / 2 + pad) / s;
    const y0 = ty - (h / 2 + pad) / s, y1 = ty + (h / 2 + pad) / s;
    for (let i = 0; i < n; i++) {
      const x = xy[2 * i], y = xy[2 * i + 1];
      if (x >= x0 && x <= x1 && y >= y0 && y <= y1 && (!rgba || rgba[4 * i + 3] > 0)) idx[k++] = i;
    }
    return idx.subarray(0, k);
  };
  if (view.cells && view.cells.n) {
    const idx = inView(view.cells.xy, view.cells.n, cellRadiusPx(view), view.cells.rgba);
    plan.counts.cells = idx.length;
    if (idx.length > lim) plan.raster.push('cells');
    else plan.cells = idx;
  }
  for (const b of view.boundaries || []) {
    const idx = new Uint32Array(b.n);
    let k = 0;
    const x0 = tx - w / 2 / s, x1 = tx + w / 2 / s, y0 = ty - h / 2 / s, y1 = ty + h / 2 / s;
    for (let p = 0; p < b.n; p++) {
      let bx0 = Infinity, by0 = Infinity, bx1 = -Infinity, by1 = -Infinity;
      for (let v = b.start[p]; v < b.start[p + 1]; v++) {
        const x = b.xy[2 * v], y = b.xy[2 * v + 1];
        if (x < bx0) bx0 = x;
        if (x > bx1) bx1 = x;
        if (y < by0) by0 = y;
        if (y > by1) by1 = y;
      }
      if (bx1 >= x0 && bx0 <= x1 && by1 >= y0 && by0 <= y1) idx[k++] = p;
    }
    plan.counts.boundaries[b.kind] = k;
    if (k > lim) plan.raster.push(`boundary:${b.kind}`);
    else plan.boundaries[b.kind] = idx.subarray(0, k);
  }
  if (view.transcripts && view.transcripts.n) {
    const idx = inView(view.transcripts.xy, view.transcripts.n, view.transcripts.radiusPx, null);
    plan.counts.transcripts = idx.length;
    if (idx.length > lim) plan.raster.push('transcripts');
    else plan.transcripts = idx;
  }
  return plan;
}

// ── scene ─────────────────────────────────────────────────────────────────────────────────
const isColor = (c) => /^\s*(#([0-9a-f]{3}|[0-9a-f]{6})|rgba?\()/i.test(String(c || ''));
const rgbOf = (c) => (Array.isArray(c) ? [c[0], c[1], c[2]] : hexToRgb(c));
const LAYER_NAMES = {cells: 'cells', transcripts: 'transcripts'};
const layerName = (key, view) => {
  if (LAYER_NAMES[key]) return LAYER_NAMES[key];
  const kind = key.replace(/^boundary:/, '');
  const b = (view.boundaries || []).find((x) => x.kind === kind);
  return (b && b.label) || `${kind} boundaries`;
};

/**
 * The figure as a display list.
 * @param view see planFigure
 * @param plan the result of planFigure(view)
 * @param rasters {underlay, cells, transcripts, 'boundary:<kind>'}: images of the map area,
 *   each {href (data URL), pw, ph, getRgba() → Uint8ClampedArray}; one is required for every
 *   entry of plan.raster, `underlay` (image layer, density) is drawn beneath everything
 * @returns {{width, height, mapWidth, mapHeight, title, desc, info, groups}}
 */
export function buildScene(view, plan, rasters = {}, {title = 'Xenium spatial map', background} = {}) {
  const {width: w, height: h, scale: s} = view;
  const tx = view.target[0], ty = view.target[1];
  const px = (x) => (x - tx) * s + w / 2;
  const py = (y) => (y - ty) * s + h / 2;
  const bgCss = background !== undefined ? background : view.background;
  const bg = isColor(bgCss) ? hexToRgb(bgCss) : null;
  const dark = bg ? (0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]) / 255 < 0.5 : false;
  const fg = dark ? [240, 243, 246] : [20, 20, 20];
  const image = (r) => ({t: 'image', x: 0, y: 0, w, h, href: r.href, pw: r.pw, ph: r.ph, getRgba: r.getRgba});
  const rasterized = [];
  const map = [];

  if (rasters.underlay) map.push({t: 'g', attrs: {id: 'underlay'}, items: [image(rasters.underlay)]});

  // cells
  if (view.cells && view.cells.n) {
    const c = view.cells;
    if (plan.raster.includes('cells')) {
      if (!rasters.cells) throw new Error('buildScene: no raster for the cell layer');
      rasterized.push('cells');
      map.push({t: 'g', attrs: {id: 'cells', 'data-raster': 'true', 'data-count': plan.counts.cells}, items: [image(rasters.cells)]});
    } else {
      const idx = plan.cells;
      const n = idx.length;
      const xy = new Float32Array(2 * n), rgb = new Uint8Array(3 * n), a = new Uint8Array(n);
      for (let k = 0; k < n; k++) {
        const i = idx[k];
        xy[2 * k] = px(c.xy[2 * i]);
        xy[2 * k + 1] = py(c.xy[2 * i + 1]);
        rgb[3 * k] = c.rgba[4 * i];
        rgb[3 * k + 1] = c.rgba[4 * i + 1];
        rgb[3 * k + 2] = c.rgba[4 * i + 2];
        a[k] = c.rgba[4 * i + 3];
      }
      map.push({t: 'g', attrs: {id: 'cells', 'data-count': n}, items: [{t: 'circles', n, xy, rgb, a, r: cellRadiusPx(view), opacity: c.opacity}]});
    }
  }

  // boundaries, in the order they are drawn
  for (const b of view.boundaries || []) {
    const key = `boundary:${b.kind}`;
    const id = `boundaries-${b.kind}`;
    if (plan.raster.includes(key)) {
      if (!rasters[key]) throw new Error(`buildScene: no raster for ${key}`);
      rasterized.push(key);
      map.push({t: 'g', attrs: {id, 'data-raster': 'true', 'data-count': plan.counts.boundaries[b.kind]}, items: [image(rasters[key])]});
      continue;
    }
    const idx = plan.boundaries[b.kind] || new Uint32Array(0);
    let total = 0;
    for (let k = 0; k < idx.length; k++) total += b.start[idx[k] + 1] - b.start[idx[k]];
    const xy = new Float32Array(2 * total);
    const start = new Uint32Array(idx.length + 1);
    let o = 0;
    for (let k = 0; k < idx.length; k++) {
      let v0 = b.start[idx[k]], v1 = b.start[idx[k] + 1];
      // rings that repeat their first vertex are closed by the path operator instead
      if (v1 - v0 > 1 && b.xy[2 * v0] === b.xy[2 * (v1 - 1)] && b.xy[2 * v0 + 1] === b.xy[2 * (v1 - 1) + 1]) v1--;
      for (let v = v0; v < v1; v++) {
        xy[2 * o] = px(b.xy[2 * v]);
        xy[2 * o + 1] = py(b.xy[2 * v + 1]);
        o++;
      }
      start[k + 1] = o;
    }
    map.push({t: 'g', attrs: {id, 'data-count': idx.length}, items: [{
      t: 'rings', xy: xy.subarray(0, 2 * o), start, closed: true,
      stroke: [b.color[0], b.color[1], b.color[2]], opacity: (b.color[3] === undefined ? 255 : b.color[3]) / 255, width: 1,
    }]});
  }

  // transcripts
  if (view.transcripts && view.transcripts.n) {
    const t = view.transcripts;
    if (plan.raster.includes('transcripts')) {
      if (!rasters.transcripts) throw new Error('buildScene: no raster for the transcripts');
      rasterized.push('transcripts');
      map.push({t: 'g', attrs: {id: 'transcripts', 'data-raster': 'true', 'data-count': plan.counts.transcripts}, items: [image(rasters.transcripts)]});
    } else {
      const idx = plan.transcripts;
      const n = idx.length;
      const xy = new Float32Array(2 * n), rgb = new Uint8Array(3 * n);
      for (let k = 0; k < n; k++) {
        const i = idx[k];
        xy[2 * k] = px(t.xy[2 * i]);
        xy[2 * k + 1] = py(t.xy[2 * i + 1]);
        rgb[3 * k] = t.rgba[4 * i];
        rgb[3 * k + 1] = t.rgba[4 * i + 1];
        rgb[3 * k + 2] = t.rgba[4 * i + 2];
      }
      map.push({t: 'g', attrs: {id: 'transcripts', 'data-count': n}, items: [{t: 'circles', n, xy, rgb, a: null, r: t.radiusPx, opacity: 1}]});
    }
  }

  // overlays (always vectors: they are few)
  const ov = view.overlays || {};
  const ovItems = [];
  for (const d of ov.lines || []) {
    ovItems.push({t: 'line', x1: px(d.from[0]), y1: py(d.from[1]), x2: px(d.to[0]), y2: py(d.to[1]),
      stroke: rgbOf(d.color || '#ffffff'), opacity: (d.alpha === undefined ? 255 : d.alpha) / 255, width: d.width || 1});
  }
  for (const d of ov.polygons || []) {
    if (!d.points || !d.points.length) continue;
    const width = d.width === 0 ? 0 : d.width || 2;
    ovItems.push({t: 'poly', pts: d.points.map((p) => [px(p[0]), py(p[1])]), closed: true,
      fill: d.fill ? hexToRgb(d.fill) : null, fillOpacity: (d.fillAlpha === undefined ? 64 : d.fillAlpha) / 255,
      stroke: width ? hexToRgb(d.color || '#ffffff') : null, strokeOpacity: (d.alpha === undefined ? 255 : d.alpha) / 255, strokeWidth: width});
  }
  for (const d of ov.points || []) {
    ovItems.push({t: 'circles', n: 1, xy: new Float32Array([px(d.at[0]), py(d.at[1])]), rgb: new Uint8Array(hexToRgb(d.color || '#ffffff')),
      a: null, r: (d.size || 8) / 2, opacity: 1});
  }
  for (const l of view.labels || []) {
    ovItems.push({t: 'text', x: px(l.x), y: py(l.y) - 6, text: l.text, size: 12, bold: true, anchor: 'middle',
      fill: rgbOf(l.color || '#ffffff'), halo: [0, 0, 0]});
  }
  if (ovItems.length) map.push({t: 'g', attrs: {id: 'overlays'}, items: ovItems});

  // scale bar (bottom left of the map)
  const umPerPx = 1 / s;
  const bar = scaleBarLength(umPerPx, w);
  const barItems = [];
  if (bar.px > 0) {
    const x0 = 16, y0 = h - 18;
    const tw = textWidth(bar.label, 12, true);
    if (bg) barItems.push({t: 'rect', x: x0 - 6, y: y0 - 22, w: Math.max(bar.px, tw) + 12, h: 32, fill: bg, fillOpacity: 0.75});
    barItems.push({t: 'rect', x: x0, y: y0, w: bar.px, h: 3, fill: fg, attrs: {id: 'scale-bar', 'data-length-um': bar.um}});
    barItems.push({t: 'text', x: x0, y: y0 - 6, text: bar.label, size: 12, bold: true, anchor: 'start', fill: fg, attrs: {id: 'scale-bar-label'}});
  }

  // legend (right margin; further columns when a list is longer than the map is tall)
  const FS = 11, ROW = 15, PAD = 12, GAP = 10;
  const boxes = view.legend || [];
  let widest = 0;
  for (const b of boxes) {
    widest = Math.max(widest, textWidth(b.title || '', FS, true), b.range ? 180 : 0); // 180: the colour bar
    for (const r of b.rows || []) widest = Math.max(widest, textWidth(r.name, FS) + 18);
  }
  const colW = Math.max(110, Math.min(240, Math.ceil(widest)));
  let lx = w + PAD, ly = PAD, cols = 1;
  const newCol = () => {
    lx += colW + PAD;
    ly = PAD;
    cols++;
  };
  const legendItems = [];
  for (const b of boxes) {
    if (ly + 18 + (b.range ? 30 : ROW) > h - PAD && ly > PAD) newCol();
    const items = [{t: 'text', x: lx, y: ly + FS, text: clipText(b.title || '', colW, FS, true), size: FS, bold: true, anchor: 'start', fill: fg,
      attrs: {class: 'legend-title'}}];
    ly += 20;
    if (b.range) {
      const bw = Math.min(colW, 180);
      items.push({t: 'gradient', x: lx, y: ly, w: bw, h: 10, lut: colormapLut(b.cmap), attrs: {class: 'legend-bar', 'data-colormap': b.cmap}});
      items.push({t: 'text', x: lx, y: ly + 10 + FS + 2, text: fmtNum(b.range[0]), size: FS, anchor: 'start', fill: fg, attrs: {class: 'legend-min'}});
      items.push({t: 'text', x: lx + bw, y: ly + 10 + FS + 2, text: fmtNum(b.range[1]), size: FS, anchor: 'end', fill: fg, attrs: {class: 'legend-max'}});
      ly += 10 + FS + 6;
    }
    for (const r of b.rows || []) {
      if (ly + ROW > h - PAD && ly > PAD) newCol();
      items.push({t: 'g', attrs: {class: 'legend-entry'}, items: [
        {t: 'circles', n: 1, xy: new Float32Array([lx + 5, ly + ROW / 2]), rgb: new Uint8Array(hexToRgb(r.color)), a: null, r: 5, opacity: 1},
        {t: 'text', x: lx + 16, y: ly + ROW / 2 + 0.35 * FS, text: clipText(r.name, colW - 16, FS), size: FS, anchor: 'start', fill: fg},
      ]});
      ly += ROW;
    }
    ly += GAP;
    legendItems.push({t: 'g', attrs: {class: 'legend-box', 'data-kind': b.kind || ''}, items});
  }
  const legendW = boxes.length ? PAD + cols * (colW + PAD) : 0;
  const width = w + legendW;

  const x0 = tx - w / 2 / s, y0 = ty - h / 2 / s, x1 = tx + w / 2 / s, y1 = ty + h / 2 / s;
  const notes = [
    `Viewport ${fmtNum(x0)}, ${fmtNum(y0)} to ${fmtNum(x1)}, ${fmtNum(y1)} µm; ${+umPerPx.toPrecision(6)} µm per figure unit; scale bar ${bar.label}.`,
  ];
  if (view.cells && view.cells.n) notes.push(`${plan.counts.cells} cells in view.`);
  for (const key of rasterized) {
    const n = key === 'cells' ? plan.counts.cells : key === 'transcripts' ? plan.counts.transcripts : plan.counts.boundaries[key.slice(9)];
    notes.push(`The ${layerName(key, view)} layer (${n} in view, more than the vector limit of ${plan.maxVector}) is embedded as an image.`);
  }
  if (rasters.underlay) notes.push('The image / density layers are embedded as an image.');
  const info = {
    umPerPx, viewport: [x0, y0, x1, y1], scaleBarUm: bar.um, scaleBarPx: bar.px, scaleBarLabel: bar.label,
    counts: plan.counts, rasterized, underlay: !!rasters.underlay, maxVector: plan.maxVector, notes,
  };
  const groups = [];
  if (bg) groups.push({id: 'background', clip: false, items: [{t: 'rect', x: 0, y: 0, w: width, h, fill: bg}]});
  groups.push({id: 'map', clip: true, items: map});
  if (barItems.length) groups.push({id: 'scale', clip: false, items: barItems});
  if (legendItems.length) groups.push({id: 'legend', clip: false, items: legendItems});
  const fullTitle = rasterized.length ? `${title} (${rasterized.map((k) => layerName(k, view)).join(', ')} rasterized)` : title;
  return {width, height: h, mapWidth: w, mapHeight: h, title: fullTitle, desc: notes.join(' '), info, groups};
}

// ── SVG ───────────────────────────────────────────────────────────────────────────────────
const n2 = (v) => String(Math.round(v * 100) / 100);
const n3 = (v) => String(Math.round(v * 1000) / 1000);
const hex = (c) => '#' + ((1 << 24) | (c[0] << 16) | (c[1] << 8) | c[2]).toString(16).slice(1);
/**
 * Text for an XML attribute or element. Every code point XML 1.0 does not allow is dropped -
 * the C0 controls other than tab, LF and CR, U+FFFE, U+FFFF and unpaired surrogates (one
 * such character makes the whole SVG ill-formed: no viewer opens it, and the PNG export,
 * which loads the SVG as an image, fails) - then the markup characters are escaped.
 */
export function xmlText(s) {
  // eslint-disable-next-line no-control-regex
  return String(s).replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\ufffe\uffff]/g, '')
    .replace(/[\ud800-\udbff][\udc00-\udfff]|[\ud800-\udfff]/g, (m) => (m.length === 2 ? m : ''))
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
const xml = xmlText;
const attrs = (a) => (a ? Object.keys(a).map((k) => ` ${k}="${xml(a[k])}"`).join('') : '');
const FONT = 'Helvetica, Arial, sans-serif';

function svgItems(items, out, defs) {
  for (const it of items) {
    switch (it.t) {
      case 'g':
        out.push(`<g${attrs(it.attrs)}>`);
        svgItems(it.items, out, defs);
        out.push('</g>');
        break;
      case 'image':
        out.push(`<image x="${n2(it.x)}" y="${n2(it.y)}" width="${n2(it.w)}" height="${n2(it.h)}" preserveAspectRatio="none" xlink:href="${it.href}"/>`);
        break;
      case 'circles': {
        const r = n2(it.r);
        const op = it.opacity === undefined ? 1 : it.opacity;
        out.push(op < 1 ? `<g fill-opacity="${n3(op)}">` : '<g>');
        for (let k = 0; k < it.n; k++) {
          const a = it.a ? it.a[k] : 255;
          const fo = a < 255 ? ` fill-opacity="${n3((a / 255) * op)}"` : '';
          out.push(`<circle cx="${n2(it.xy[2 * k])}" cy="${n2(it.xy[2 * k + 1])}" r="${r}" fill="${hex([it.rgb[3 * k], it.rgb[3 * k + 1], it.rgb[3 * k + 2]])}"${fo}/>`);
        }
        out.push('</g>');
        break;
      }
      case 'rings': {
        out.push(`<g fill="none" stroke="${hex(it.stroke)}" stroke-width="${n2(it.width)}"${it.opacity < 1 ? ` stroke-opacity="${n3(it.opacity)}"` : ''} stroke-linejoin="round">`);
        for (let p = 0; p + 1 < it.start.length; p++) {
          let d = '';
          for (let v = it.start[p]; v < it.start[p + 1]; v++) d += `${v === it.start[p] ? 'M' : 'L'}${n2(it.xy[2 * v])} ${n2(it.xy[2 * v + 1])}`;
          if (d) out.push(`<path d="${d}${it.closed ? 'Z' : ''}"/>`);
        }
        out.push('</g>');
        break;
      }
      case 'poly': {
        const d = it.pts.map((p, i) => `${i ? 'L' : 'M'}${n2(p[0])} ${n2(p[1])}`).join('') + (it.closed ? 'Z' : '');
        const fill = it.fill ? ` fill="${hex(it.fill)}" fill-opacity="${n3(it.fillOpacity)}"` : ' fill="none"';
        const stroke = it.stroke ? ` stroke="${hex(it.stroke)}" stroke-width="${n2(it.strokeWidth)}"${it.strokeOpacity < 1 ? ` stroke-opacity="${n3(it.strokeOpacity)}"` : ''}` : '';
        out.push(`<path d="${d}"${fill}${stroke}/>`);
        break;
      }
      case 'line':
        out.push(`<line x1="${n2(it.x1)}" y1="${n2(it.y1)}" x2="${n2(it.x2)}" y2="${n2(it.y2)}" stroke="${hex(it.stroke)}" stroke-width="${n2(it.width)}"${it.opacity < 1 ? ` stroke-opacity="${n3(it.opacity)}"` : ''}/>`);
        break;
      case 'rect':
        out.push(`<rect${attrs(it.attrs)} x="${n2(it.x)}" y="${n2(it.y)}" width="${n2(it.w)}" height="${n2(it.h)}" fill="${hex(it.fill)}"${it.fillOpacity !== undefined && it.fillOpacity < 1 ? ` fill-opacity="${n3(it.fillOpacity)}"` : ''}/>`);
        break;
      case 'gradient': {
        const id = `grad${defs.filter((d) => d.startsWith('<linearGradient')).length + 1}`;
        let stops = '';
        for (let i = 0; i <= 255; i += 5) {
          stops += `<stop offset="${n3(i / 255)}" stop-color="${hex([it.lut[3 * i], it.lut[3 * i + 1], it.lut[3 * i + 2]])}"/>`;
        }
        defs.push(`<linearGradient id="${id}" x1="0" y1="0" x2="1" y2="0">${stops}</linearGradient>`);
        out.push(`<rect${attrs(it.attrs)} x="${n2(it.x)}" y="${n2(it.y)}" width="${n2(it.w)}" height="${n2(it.h)}" fill="url(#${id})"/>`);
        break;
      }
      case 'text': {
        const anchor = it.anchor && it.anchor !== 'start' ? ` text-anchor="${it.anchor}"` : '';
        const halo = it.halo ? ` stroke="${hex(it.halo)}" stroke-width="3" stroke-linejoin="round" paint-order="stroke"` : '';
        out.push(`<text${attrs(it.attrs)} x="${n2(it.x)}" y="${n2(it.y)}" font-size="${it.size}"${it.bold ? ' font-weight="bold"' : ''}${anchor} fill="${hex(it.fill)}"${halo}>${xml(it.text)}</text>`);
        break;
      }
      default:
        throw new Error(`figure: unknown item ${it.t}`);
    }
  }
}

/** The scene as an SVG document (a string). */
export function sceneToSvg(scene) {
  const body = [];
  const defs = [`<clipPath id="map-clip"><rect x="0" y="0" width="${n2(scene.mapWidth)}" height="${n2(scene.mapHeight)}"/></clipPath>`];
  for (const g of scene.groups) {
    body.push(`<g id="${g.id}"${g.clip ? ' clip-path="url(#map-clip)"' : ''}>`);
    svgItems(g.items, body, defs);
    body.push('</g>');
  }
  const i = scene.info;
  const head = `<?xml version="1.0" encoding="UTF-8"?>\n` +
    `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="${n2(scene.width)}" height="${n2(scene.height)}" ` +
    `viewBox="0 0 ${n2(scene.width)} ${n2(scene.height)}" font-family="${FONT}" ` +
    `data-um-per-px="${+i.umPerPx.toPrecision(9)}" data-viewport="${i.viewport.map((v) => +v.toFixed(3)).join(' ')}" ` +
    `data-cells-in-view="${i.counts.cells}" data-max-vector="${i.maxVector}" data-rasterized="${xml(i.rasterized.join(','))}">\n` +
    `<title>${xml(scene.title)}</title>\n<desc>${xml(scene.desc)}</desc>\n<defs>`;
  return head + defs.join('') + '</defs>\n' + body.join('\n') + '\n</svg>\n';
}

// ── PDF ───────────────────────────────────────────────────────────────────────────────────
// A minimal PDF 1.4 writer: one page in figure units (1 unit = 1 pt), filled and stroked
// paths, Flate-compressed RGB images with a soft mask, axial shadings, and text in the
// standard Helvetica fonts (not embedded; WinAnsi characters only, others print as '?').
// Windows-1252 bytes 0x80-0x9F: all 27 characters that sit above U+00FF in Unicode (the rest
// of the code page equals Latin-1), plus two look-alikes that have no code of their own
// there (Greek mu → micro sign, minus sign → hyphen-minus).
const WINANSI = {0x20ac: 0x80, 0x201a: 0x82, 0x0192: 0x83, 0x201e: 0x84, 0x2026: 0x85, 0x2020: 0x86, 0x2021: 0x87,
  0x02c6: 0x88, 0x2030: 0x89, 0x0160: 0x8a, 0x2039: 0x8b, 0x0152: 0x8c, 0x017d: 0x8e, 0x2018: 0x91, 0x2019: 0x92,
  0x201c: 0x93, 0x201d: 0x94, 0x2022: 0x95, 0x2013: 0x96, 0x2014: 0x97, 0x02dc: 0x98, 0x2122: 0x99, 0x0161: 0x9a,
  0x203a: 0x9b, 0x0153: 0x9c, 0x017e: 0x9e, 0x0178: 0x9f, 0x03bc: 0xb5, 0x2212: 0x2d};

/** A PDF literal string in WinAnsiEncoding; a character outside Windows-1252 becomes '?'. */
export function pdfText(text) {
  let out = '(';
  for (const ch of String(text)) {
    let c = ch.codePointAt(0);
    if (WINANSI[c]) c = WINANSI[c];
    else if (c > 0xff || (c >= 0x7f && c < 0xa0) || c < 0x20) c = 0x3f;
    if (c === 0x28 || c === 0x29 || c === 0x5c) out += '\\' + String.fromCharCode(c);
    else if (c > 0x7e) out += '\\' + c.toString(8);
    else out += String.fromCharCode(c);
  }
  return out + ')';
}

/** A text string of the document information dictionary (UTF-16BE, any character). */
function pdfInfoText(text) {
  let out = '<FEFF';
  const s = String(text);
  for (let i = 0; i < s.length; i++) out += s.charCodeAt(i).toString(16).padStart(4, '0').toUpperCase();
  return out + '>';
}

const ascii = (s) => {
  const b = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i) & 255;
  return b;
};

async function deflate(bytes) {
  if (typeof CompressionStream === 'undefined') return null;
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

const K = 0.5522847498; // control-point offset of a quarter circle

function pdfItems(items, out, res) {
  const gs = (alpha) => {
    const a = Math.max(0, Math.min(1, Math.round(alpha * 1000) / 1000));
    if (!res.gs.has(a)) res.gs.set(a, `GS${res.gs.size + 1}`);
    return `/${res.gs.get(a)} gs`;
  };
  const rg = (c, op) => `${n3(c[0] / 255)} ${n3(c[1] / 255)} ${n3(c[2] / 255)} ${op}`;
  for (const it of items) {
    switch (it.t) {
      case 'g':
        pdfItems(it.items, out, res);
        break;
      case 'image': {
        const name = `Im${res.images.length + 1}`;
        res.images.push({name, item: it});
        out.push(`q ${n2(it.w)} 0 0 ${n2(-it.h)} ${n2(it.x)} ${n2(it.y + it.h)} cm /${name} Do Q`);
        break;
      }
      case 'circles': {
        const r = it.r, k = r * K;
        const op = it.opacity === undefined ? 1 : it.opacity;
        out.push('q');
        let lastA = -1, lastC = -1;
        for (let i = 0; i < it.n; i++) {
          const a = it.a ? it.a[i] : 255;
          if (a !== lastA) {
            out.push(gs((a / 255) * op));
            lastA = a;
          }
          const c = (it.rgb[3 * i] << 16) | (it.rgb[3 * i + 1] << 8) | it.rgb[3 * i + 2];
          if (c !== lastC) {
            out.push(rg([it.rgb[3 * i], it.rgb[3 * i + 1], it.rgb[3 * i + 2]], 'rg'));
            lastC = c;
          }
          const x = it.xy[2 * i], y = it.xy[2 * i + 1];
          out.push(`${n2(x + r)} ${n2(y)} m ${n2(x + r)} ${n2(y + k)} ${n2(x + k)} ${n2(y + r)} ${n2(x)} ${n2(y + r)} c ` +
            `${n2(x - k)} ${n2(y + r)} ${n2(x - r)} ${n2(y + k)} ${n2(x - r)} ${n2(y)} c ` +
            `${n2(x - r)} ${n2(y - k)} ${n2(x - k)} ${n2(y - r)} ${n2(x)} ${n2(y - r)} c ` +
            `${n2(x + k)} ${n2(y - r)} ${n2(x + r)} ${n2(y - k)} ${n2(x + r)} ${n2(y)} c f`);
        }
        out.push('Q');
        break;
      }
      case 'rings':
        out.push(`q ${gs(it.opacity)} ${rg(it.stroke, 'RG')} ${n2(it.width)} w 1 j`);
        for (let p = 0; p + 1 < it.start.length; p++) {
          let d = '';
          for (let v = it.start[p]; v < it.start[p + 1]; v++) d += `${n2(it.xy[2 * v])} ${n2(it.xy[2 * v + 1])} ${v === it.start[p] ? 'm' : 'l'} `;
          if (d) out.push(d + (it.closed ? 's' : 'S'));
        }
        out.push('Q');
        break;
      case 'poly': {
        const d = it.pts.map((p, i) => `${n2(p[0])} ${n2(p[1])} ${i ? 'l' : 'm'}`).join(' ') + (it.closed ? ' h' : '');
        if (it.fill) out.push(`q ${gs(it.fillOpacity)} ${rg(it.fill, 'rg')} ${d} f Q`);
        if (it.stroke) out.push(`q ${gs(it.strokeOpacity)} ${rg(it.stroke, 'RG')} ${n2(it.strokeWidth)} w ${d} S Q`);
        break;
      }
      case 'line':
        out.push(`q ${gs(it.opacity)} ${rg(it.stroke, 'RG')} ${n2(it.width)} w ${n2(it.x1)} ${n2(it.y1)} m ${n2(it.x2)} ${n2(it.y2)} l S Q`);
        break;
      case 'rect':
        out.push(`q ${gs(it.fillOpacity === undefined ? 1 : it.fillOpacity)} ${rg(it.fill, 'rg')} ${n2(it.x)} ${n2(it.y)} ${n2(it.w)} ${n2(it.h)} re f Q`);
        break;
      case 'gradient': {
        const name = `Sh${res.shadings.length + 1}`;
        res.shadings.push({name, item: it});
        out.push(`q ${gs(1)} ${n2(it.x)} ${n2(it.y)} ${n2(it.w)} ${n2(it.h)} re W n /${name} sh Q`);
        break;
      }
      case 'text': {
        const w = textWidth(it.text, it.size, it.bold);
        const x = it.anchor === 'end' ? it.x - w : it.anchor === 'middle' ? it.x - w / 2 : it.x;
        const font = `/${it.bold ? 'F2' : 'F1'} ${it.size} Tf 1 0 0 -1 ${n2(x)} ${n2(it.y)} Tm`;
        const str = pdfText(it.text);
        out.push(`q ${gs(1)}`);
        if (it.halo) out.push(`BT ${font} 1 Tr 3 w 1 j ${rg(it.halo, 'RG')} ${str} Tj ET`);
        out.push(`BT ${font} 0 Tr ${rg(it.fill, 'rg')} ${str} Tj ET Q`);
        break;
      }
      default:
        throw new Error(`figure: unknown item ${it.t}`);
    }
  }
}

/**
 * The scene as a PDF file.
 * @returns {Promise<Uint8Array>}
 */
export async function sceneToPdf(scene) {
  const W = scene.width, H = scene.height;
  const res = {gs: new Map(), images: [], shadings: []};
  const content = [`1 0 0 -1 0 ${n2(H)} cm`]; // y down, as in the view
  for (const g of scene.groups) {
    content.push(g.clip ? `q 0 0 ${n2(scene.mapWidth)} ${n2(scene.mapHeight)} re W n` : 'q');
    pdfItems(g.items, content, res);
    content.push('Q');
  }

  const objects = []; // {dict, stream}
  const add = (dict, stream = null) => objects.push({dict, stream}) && objects.length; // → object number
  const addStream = async (dict, bytes) => {
    const z = await deflate(bytes);
    return add(`${dict}${z ? ' /Filter /FlateDecode' : ''} /Length ${(z || bytes).length}`, z || bytes);
  };
  const catalog = add(''), pages = add(''), page = add('');
  const f1 = add('/Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding');
  const f2 = add('/Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding');
  const contents = await addStream('', ascii(content.join('\n') + '\n'));
  const xobjects = [];
  for (const im of res.images) {
    const rgba = im.item.getRgba();
    const n = im.item.pw * im.item.ph;
    const rgb = new Uint8Array(3 * n), alpha = new Uint8Array(n);
    let opaque = true;
    for (let i = 0; i < n; i++) {
      rgb[3 * i] = rgba[4 * i];
      rgb[3 * i + 1] = rgba[4 * i + 1];
      rgb[3 * i + 2] = rgba[4 * i + 2];
      if ((alpha[i] = rgba[4 * i + 3]) !== 255) opaque = false;
    }
    const base = `/Type /XObject /Subtype /Image /Width ${im.item.pw} /Height ${im.item.ph} /BitsPerComponent 8`;
    const mask = opaque ? 0 : await addStream(`${base} /ColorSpace /DeviceGray`, alpha);
    xobjects.push(`/${im.name} ${await addStream(`${base} /ColorSpace /DeviceRGB${mask ? ` /SMask ${mask} 0 R` : ''}`, rgb)} 0 R`);
  }
  const shadings = [];
  for (const sh of res.shadings) {
    const fn = await addStream('/FunctionType 0 /Domain [0 1] /Size [256] /BitsPerSample 8 /Range [0 1 0 1 0 1]', sh.item.lut);
    const it = sh.item;
    shadings.push(`/${sh.name} ${add(`/ShadingType 2 /ColorSpace /DeviceRGB /Coords [${n2(it.x)} 0 ${n2(it.x + it.w)} 0] /Function ${fn} 0 R /Extend [true true]`)} 0 R`);
  }
  const info = add(`/Title ${pdfInfoText(scene.title)} /Subject ${pdfInfoText(scene.desc)} /Creator ${pdfInfoText('Xenium viewer figure export')}`);
  const gsDict = [...res.gs].map(([a, name]) => `/${name} << /ca ${a} /CA ${a} >>`).join(' ');
  objects[catalog - 1].dict = `/Type /Catalog /Pages ${pages} 0 R`;
  objects[pages - 1].dict = `/Type /Pages /Kids [${page} 0 R] /Count 1`;
  objects[page - 1].dict = `/Type /Page /Parent ${pages} 0 R /MediaBox [0 0 ${n2(W)} ${n2(H)}] /Contents ${contents} 0 R ` +
    `/Group << /S /Transparency /CS /DeviceRGB >> ` +
    `/Resources << /Font << /F1 ${f1} 0 R /F2 ${f2} 0 R >> /ExtGState << ${gsDict} >> /XObject << ${xobjects.join(' ')} >> /Shading << ${shadings.join(' ')} >> >>`;

  const parts = [];
  let size = 0;
  const push = (bytes) => {
    parts.push(bytes);
    size += bytes.length;
  };
  push(ascii('%PDF-1.4\n%\xe2\xe3\xcf\xd3\n'));
  const offsets = [];
  objects.forEach((o, i) => {
    offsets.push(size);
    if (o.stream) {
      push(ascii(`${i + 1} 0 obj\n<< ${o.dict} >>\nstream\n`));
      push(o.stream);
      push(ascii('\nendstream\nendobj\n'));
    } else {
      push(ascii(`${i + 1} 0 obj\n<< ${o.dict} >>\nendobj\n`));
    }
  });
  const xref = size;
  let tail = `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  for (const off of offsets) tail += `${String(off).padStart(10, '0')} 00000 n \n`;
  tail += `trailer\n<< /Size ${objects.length + 1} /Root ${catalog} 0 R /Info ${info} 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  push(ascii(tail));
  const out = new Uint8Array(size);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}
