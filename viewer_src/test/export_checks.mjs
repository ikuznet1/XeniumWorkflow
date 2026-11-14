// Checks of the figure export (SVG / PDF / PNG of the WebGL view; src/figure.js,
// Viewer.exportFigure, the "Export figure" control of the Dash glue).
//   node test/export_checks.mjs                       pure functions + the viewer on the dev server
//   node test/export_checks.mjs http://127.0.0.1:8201 ... + the control inside the running app
// Part 1 needs nothing but node (a synthetic view through the scene, SVG and PDF writers).
// Part 2 starts viewer_src/dev_server.py on the human skin bundle (XV_SKIN / XV_TEST_DATA, see
// lib.mjs; skipped when the bundle is absent) on port XV_EXPORT_PORT (default 8074) and
// compares the exported SVG with the cells the server sends.
// Part 3 runs when an app URL is given: the app must have been started by hand with
// XENIUM_RENDERER=webgl on a real bundle (never on tests/data), as for app_smoke.mjs.
// Exported files are written to a temporary folder that is removed at the end
// (XV_EXPORT_KEEP=1 keeps it and prints its path). Exit code is non-zero on any failure.
import {execFileSync} from 'node:child_process';
import {mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {resolve} from 'node:path';
import {inflateSync} from 'node:zlib';
import {Checks, DATA, fetchBin, hasBundle, launch, newPage, startServer} from './lib.mjs';
import {buildScene, DEFAULT_MAX_VECTOR, pdfText, planFigure, scaleBarLength, sceneToPdf, sceneToSvg, textWidth, xmlText} from '../src/figure.js';
import {colormapLut} from '../src/colormap.js';

const T = new Checks();
const OUT = mkdtempSync(resolve(tmpdir(), 'xv-export-'));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const hex = (r, g, b) => '#' + ((1 << 24) | (r << 16) | (g << 8) | b).toString(16).slice(1);
const count = (text, re) => (text.match(re) || []).length;
/** The text of the top-level group `id` of an exported SVG (groups are written one per line). */
const group = (svg, id) => {
  const m = new RegExp(`\\n<g id="${id}"[^>]*>\\n([\\s\\S]*?)\\n</g>(?=\\n<g id="(?:background|map|scale|legend)"|\\n</svg>)`).exec(svg);
  return m ? m[1] : null;
};
const attr = (tag, name) => {
  const m = new RegExp(`\\s${name}="([^"]*)"`).exec(tag || '');
  return m ? m[1] : null;
};
const circles = (text) => [...(text || '').matchAll(/<circle [^>]*>/g)].map((m) => ({
  cx: +attr(m[0], 'cx'), cy: +attr(m[0], 'cy'), r: +attr(m[0], 'r'), fill: attr(m[0], 'fill'), fo: attr(m[0], 'fill-opacity'),
}));

/** Structure of a PDF written by sceneToPdf: objects by number, checked against the xref table. */
function readPdf(bytes) {
  const text = Buffer.from(bytes).toString('latin1');
  const startxref = +/startxref\n(\d+)\n%%EOF\n$/.exec(text)[1];
  const xref = text.slice(startxref);
  const n = +/^xref\n0 (\d+)\n/.exec(xref)[1];
  const entries = [...xref.matchAll(/(\d{10}) (\d{5}) ([nf]) \n/g)].slice(0, n);
  let offsetsOk = entries.length === n && text.startsWith('%PDF-1.4\n');
  const objects = {};
  entries.forEach((e, i) => {
    if (i === 0) return;
    const off = +e[1];
    const head = `${i} 0 obj\n<< `;
    if (text.substr(off, head.length) !== head) { offsetsOk = false; return; }
    const dictEnd = text.indexOf(' >>\n', off);
    const dict = text.slice(off + head.length, dictEnd);
    let stream = null;
    if (text.substr(dictEnd + 4, 7) === 'stream\n') {
      const len = +/\/Length (\d+)/.exec(dict)[1];
      const raw = Buffer.from(bytes.buffer, bytes.byteOffset + dictEnd + 11, len);
      if (text.substr(dictEnd + 11 + len, 18) !== '\nendstream\nendobj\n') offsetsOk = false;
      stream = /\/FlateDecode/.test(dict) ? inflateSync(raw) : raw;
    }
    objects[i] = {dict, stream};
  });
  const trailer = /trailer\n<< (.*) >>/.exec(xref)[1];
  const page = Object.values(objects).find((o) => /\/Type \/Page /.test(o.dict));
  const content = objects[+/\/Contents (\d+) 0 R/.exec(page.dict)[1]].stream.toString('latin1');
  return {offsetsOk, objects, trailer, page, content, text};
}

/** Renders a PDF with macOS Quartz (sips) as an independent reader; null where sips is absent. */
function quartzRender(pdfPath) {
  if (process.platform !== 'darwin') return null;
  const png = pdfPath.replace(/\.pdf$/, '.quartz.png');
  try {
    execFileSync('/usr/bin/sips', ['-s', 'format', 'png', pdfPath, '--out', png], {stdio: 'pipe'});
    const b = readFileSync(png);
    return {width: b.readUInt32BE(16), height: b.readUInt32BE(20), bytes: b.length, path: png};
  } catch (err) {
    return {error: String(err.message || err).slice(0, 200)};
  }
}

// ── 1. pure functions on a synthetic view ────────────────────────────────────────────────
{
  // scale bar: a round length (1, 2, 5 x 10^k) of at most a fifth of the width
  const cases = [[0.5, 800, 50, 100, '50 µm'], [1, 500, 100, 100, '100 µm'], [12.5, 1000, 2000, 160, '2 mm'], [0.001, 1000, 0.2, 200, '0.2 µm'],
    [0.26, 1280, 50, 50 / 0.26, '50 µm'], [3, 333, 100, 100 / 3, '100 µm']];
  const bad = cases.filter(([um, w, len, px, label]) => {
    const b = scaleBarLength(um, w);
    return b.um !== len || Math.abs(b.px - px) > 1e-9 || b.label !== label || b.px > 0.2 * w + 1e-9;
  });
  T.ok(bad.length === 0, '1 scale bar: round lengths, never longer than a fifth of the view', JSON.stringify(bad));
  let roundOk = true;
  for (let k = -3; k < 4; k += 0.037) {
    const b = scaleBarLength(Math.pow(10, k), 1000);
    const m = b.um / Math.pow(10, Math.floor(Math.log10(b.um) + 1e-9));
    if (![1, 2, 5].some((v) => Math.abs(m - v) < 1e-6) || b.px > 200 + 1e-6 || b.px < 200 / 2.5 - 1e-6) roundOk = false;
  }
  T.ok(roundOk, '1 scale bar: 1 / 2 / 5 x 10^k over 7 decades of zoom, between 8 % and 20 % of the width');
  const zero = scaleBarLength(0, 100);
  T.ok(zero.um === 0 && zero.px === 0, '1 scale bar: no bar for a degenerate view');

  // a view of 800 x 600 px at 2 px per µm centred on (100, 100): x from -100 to 300, y from -50 to 250;
  // a point (x, y) µm lands at (2 x + 200, 2 y + 100)
  const cellXY = [[0, 0], [100, 0], [100, 100], [290, 240], [50, 50], [20, 200], [500, 500], [-300, 0]];
  const cellRGBA = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [18, 52, 86, 255], [9, 9, 9, 0], [200, 100, 50, 38],
    [1, 1, 1, 255], [2, 2, 2, 255]];
  const mkView = (legend) => ({
    width: 800, height: 600, scale: 2, target: [100, 100], background: '#0d1117',
    cells: {n: cellXY.length, xy: new Float32Array(cellXY.flat()), rgba: new Uint8Array(cellRGBA.flat()), radiusUm: 3, minPx: 1.5, maxPx: 7.5, opacity: 0.8},
    boundaries: [{kind: 'cell', label: 'cell boundaries', color: [0, 212, 255, 220], n: 3,
      // a ring that repeats its first vertex, an open one, and one outside the view
      start: new Uint32Array([0, 4, 7, 10]),
      xy: new Float32Array([0, 0, 10, 0, 10, 10, 0, 0, 100, 100, 110, 100, 110, 110, 900, 900, 910, 900, 910, 910])}],
    transcripts: {n: 4, xy: new Float32Array([10, 10, 20, 20, 30, 30, 999, 999]),
      rgba: new Uint8Array([255, 77, 77, 255, 51, 214, 255, 255, 255, 77, 77, 255, 0, 0, 0, 255]), radiusPx: 1},
    overlays: {lines: [{from: [0, 0], to: [100, 0], color: '#ffffff', width: 2}],
      polygons: [{points: [[0, 0], [50, 0], [50, 50]], color: '#ff00ff', fill: '#00ff00', label: 'ROI <1> & "x"'}],
      points: [{at: [100, 100], color: '#f0c040', size: 9}]},
    labels: [{x: 33, y: 0, text: 'ROI <1> & "x"', color: 'rgb(255, 0, 255)'}],
    legend,
  });
  const catLegend = [{kind: 'categorical', title: 'cluster', rows: [{name: 'T cells', color: '#ff0000'}, {name: 'B & NK', color: '#00ff00'},
    {name: 'Keratinocytes', color: '#0000ff'}]},
  {kind: 'transcripts', title: 'Transcripts (3)', rows: [{name: 'POSTN', color: '#ff4d4d'}]}];
  const view = mkView(catLegend);
  const plan = planFigure(view);
  T.ok(plan.maxVector === DEFAULT_MAX_VECTOR && DEFAULT_MAX_VECTOR === 50000 && plan.counts.cells === 5 && plan.raster.length === 0 &&
    plan.counts.boundaries.cell === 2 && plan.counts.transcripts === 3,
    '1 plan: counts what is in view (hidden and outside cells left out), all vector below the limit', JSON.stringify({counts: plan.counts, raster: plan.raster}));
  const scene = buildScene(view, plan);
  const svg = sceneToSvg(scene);
  writeFileSync(resolve(OUT, 'synthetic.svg'), svg);
  const cells = circles(group(svg, 'map').split('<g id="boundaries-cell"')[0]);
  const expectXY = [[200, 100], [400, 100], [400, 300], [780, 580], [240, 500]];
  const expectFill = ['#ff0000', '#00ff00', '#0000ff', '#123456', '#c86432'];
  T.ok(cells.length === 5 && cells.every((c, i) => c.cx === expectXY[i][0] && c.cy === expectXY[i][1] && c.r === 6 && c.fill === expectFill[i]),
    '1 SVG: one circle per visible cell in view, at its screen position, radius and colour', JSON.stringify(cells));
  T.ok(cells[4].fo === String(Math.round((38 / 255) * 0.8 * 1000) / 1000) && cells[0].fo === null && /<g id="cells" data-count="5">\n<g fill-opacity="0.8">/.test(svg),
    '1 SVG: layer opacity on the group, a dimmed cell carries its own', JSON.stringify({dim: cells[4].fo}));
  const bar = /<rect id="scale-bar"[^>]*>/.exec(svg)[0];
  const barPx = +attr(bar, 'width'), barUm = +attr(bar, 'data-length-um');
  // cells 0 and 1 are 100 µm apart
  const pxPerUm = (cells[1].cx - cells[0].cx) / 100;
  T.ok(barUm === 50 && barPx === 100 && Math.abs(barPx / barUm - pxPerUm) < 1e-9 && />50 µm<\/text>/.test(svg) && attr(svg, 'data-um-per-px') === '0.5',
    '1 SVG: the scale bar is as long as its label says, measured against two cells 100 µm apart', JSON.stringify({barPx, barUm, pxPerUm}));
  const rings = [...group(svg, 'map').matchAll(/<path d="([^"]*)"\/>/g)].map((m) => m[1]);
  T.ok(rings.length === 2 && rings[0] === 'M200 100L220 100L220 120Z' && rings[1] === 'M400 300L420 300L420 320Z' &&
    /<g id="boundaries-cell" data-count="2">\n<g fill="none" stroke="#00d4ff" stroke-width="1" stroke-opacity="0.863"/.test(svg),
    '1 SVG: boundary rings in view as closed paths in the boundary colour (repeated first vertex dropped)', JSON.stringify(rings));
  const txs = circles(/<g id="transcripts"[^>]*>\n([\s\S]*?)\n<\/g>\n<\/g>/.exec(svg)[1]);
  T.ok(txs.length === 3 && txs.every((c) => c.r === 1) && txs[0].fill === '#ff4d4d' && txs[1].fill === '#33d6ff' && txs[0].cx === 220,
    '1 SVG: transcripts in view as dots in their gene colours', JSON.stringify(txs));
  const ovl = svg.slice(svg.indexOf('<g id="overlays">'), svg.indexOf('<g id="scale">'));
  T.ok(/<line x1="200" y1="100" x2="400" y2="100" stroke="#ffffff" stroke-width="2"\/>/.test(ovl) &&
    /<path d="M200 100L300 100L300 200Z" fill="#00ff00" fill-opacity="0.251" stroke="#ff00ff" stroke-width="2"\/>/.test(ovl) &&
    /<circle cx="400" cy="300" r="4.5" fill="#f0c040"\/>/.test(ovl) && />ROI &lt;1&gt; &amp; &quot;x&quot;<\/text>/.test(ovl) && /fill="#ff00ff" stroke="#000000"/.test(ovl),
    '1 SVG: overlay line, polygon, marker and label (text escaped)');
  const legend = group(svg, 'legend');
  const entries = [...legend.matchAll(/<g class="legend-entry">\n<g>\n(<circle [^>]*>)\n<\/g>\n<text [^>]*>([^<]*)<\/text>/g)].map((m) => [attr(m[1], 'fill'), m[2]]);
  const titles = [...legend.matchAll(/<text class="legend-title"[^>]*>([^<]*)<\/text>/g)].map((m) => m[1]);
  T.ok(JSON.stringify(entries) === JSON.stringify([['#ff0000', 'T cells'], ['#00ff00', 'B &amp; NK'], ['#0000ff', 'Keratinocytes'], ['#ff4d4d', 'POSTN']]) &&
    JSON.stringify(titles) === JSON.stringify(['cluster', 'Transcripts (3)']) && count(legend, /data-kind="categorical"/g) === 1,
    '1 SVG: categorical legend lists the categories with their colours', JSON.stringify({entries, titles}));
  const legendX = [...legend.matchAll(/ (?:x|cx)="([\d.]+)"/g)].map((m) => +m[1]);
  T.ok(Math.min(...legendX) >= 800 && +attr(svg, 'width') > 800 + Math.max(...legendX) - 800 && +attr(svg, 'height') === 600 &&
    /<g id="map" clip-path="url\(#map-clip\)">/.test(svg) && /<clipPath id="map-clip"><rect x="0" y="0" width="800" height="600"\/>/.test(svg),
    '1 SVG: legend in the margin right of the map, map clipped to the view', JSON.stringify({min: Math.min(...legendX), width: attr(svg, 'width')}));
  T.ok(svg.startsWith('<?xml') && /<title>Xenium spatial map<\/title>/.test(svg) && /<desc>Viewport -100, -50 to 300, 250 µm; 0.5 µm per figure unit; scale bar 50 µm\. 5 cells in view\.<\/desc>/.test(svg) &&
    attr(svg, 'data-rasterized') === '' && attr(svg, 'data-cells-in-view') === '5' && attr(svg, 'data-max-vector') === '50000' && !/<image/.test(svg),
    '1 SVG: title, description and data attributes describe the view; no image in an all-vector figure', /<desc>[^<]*/.exec(svg)[0]);
  T.ok(/<g id="background">\n<rect x="0" y="0" width="[\d.]+" height="600" fill="#0d1117"\/>/.test(svg) && /id="scale-bar" [^>]*fill="#f0f3f6"/.test(svg),
    '1 SVG: background of the view, light text on a dark background');
  const light = sceneToSvg(buildScene(view, plan, {}, {background: '#ffffff'}));
  const none = sceneToSvg(buildScene({...view, background: 'transparent'}, plan));
  T.ok(/id="scale-bar" [^>]*fill="#141414"/.test(light) && /<g id="background">\n<rect [^>]*fill="#ffffff"/.test(light) && !/<g id="background">/.test(none) && /id="scale-bar" [^>]*fill="#141414"/.test(none),
    '1 SVG: dark text on a light or absent background');

  // continuous legend: colour bar with min / max and the name
  const contView = mkView([{kind: 'continuous', title: 'EEF1G (log1p)', range: [0, 4.25], cmap: 'viridis'}]);
  const contSvg = sceneToSvg(buildScene(contView, planFigure(contView)));
  const stops = [...contSvg.matchAll(/<stop offset="([\d.]+)" stop-color="(#[0-9a-f]{6})"\/>/g)].map((m) => [+m[1], m[2]]);
  const lut = colormapLut('viridis');
  const stopsOk = stops.length === 52 && stops.every(([off, col]) => {
    const i = Math.round(off * 255);
    return col === hex(lut[3 * i], lut[3 * i + 1], lut[3 * i + 2]);
  });
  T.ok(stopsOk && stops[0][1] === '#440154' && stops[51][1] === '#fde725' && /<rect class="legend-bar" data-colormap="viridis"[^>]*fill="url\(#grad1\)"\/>/.test(contSvg) &&
    /class="legend-title"[^>]*>EEF1G \(log1p\)<\/text>/.test(contSvg) && /class="legend-min"[^>]*>0<\/text>/.test(contSvg) &&
    /class="legend-max"[^>]*text-anchor="end"[^>]*>4\.25<\/text>/.test(contSvg) && count(contSvg, /legend-entry/g) === 0,
    '1 SVG: continuous legend is a colour bar of the colour map with min, max and the name', JSON.stringify({stops: stops.length}));

  // a long categorical legend flows into further columns instead of leaving the figure
  const many = mkView([{kind: 'categorical', title: 'cluster', rows: Array.from({length: 90}, (_, i) => ({name: `cluster ${i}`, color: '#336699'}))}]);
  const manyScene = buildScene(many, planFigure(many));
  const manySvg = sceneToSvg(manyScene);
  const ys = [...group(manySvg, 'legend').matchAll(/<circle cx="([\d.]+)" cy="([\d.]+)"/g)].map((m) => [+m[1], +m[2]]);
  T.ok(ys.length === 90 && ys.every(([x, y]) => y > 0 && y < 600 && x < manyScene.width) && new Set(ys.map((p) => p[0])).size === 3,
    '1 SVG: 90 categories in 3 columns inside the figure', JSON.stringify({width: manyScene.width, columns: [...new Set(ys.map((p) => p[0]))]}));

  // raster fallback above the limit
  const low = planFigure(view, {maxVector: 4});
  let threw = false;
  try { buildScene(view, low); } catch { threw = true; }
  const px = new Uint8ClampedArray(2 * 2 * 4).fill(255);
  px[3] = 0; // one transparent pixel: the image needs a soft mask
  const fake = {href: 'data:image/png;base64,AAAA', pw: 2, ph: 2, getRgba: () => px};
  const rScene = buildScene(view, low, {cells: fake});
  const rSvg = sceneToSvg(rScene);
  const rMap = group(rSvg, 'map');
  T.ok(JSON.stringify(low.raster) === '["cells"]' && low.cells === null && threw && circles(rMap.split('<g id="boundaries-cell"')[0]).length === 0 &&
    /<g id="cells" data-raster="true" data-count="5">\n<image x="0" y="0" width="800" height="600" preserveAspectRatio="none" xlink:href="data:image\/png;base64,AAAA"\/>/.test(rSvg) &&
    count(rSvg, /<image /g) === 1 && attr(rSvg, 'data-rasterized') === 'cells' && attr(rSvg, 'data-max-vector') === '4',
    '1 raster fallback: above the limit the cell layer is one embedded image and no circles', JSON.stringify({raster: low.raster, threw}));
  T.ok(/<title>Xenium spatial map \(cells rasterized\)<\/title>/.test(rSvg) && /The cells layer \(5 in view, more than the vector limit of 4\) is embedded as an image\./.test(rSvg) &&
    rScene.info.rasterized[0] === 'cells' && /<rect id="scale-bar"/.test(rSvg) && count(rSvg, /legend-entry/g) === 4 && count(rMap, /<path /g) === 3,
    '1 raster fallback: said in title and description; scale bar, legend, boundaries and overlays stay vector');
  const per = planFigure(view, {maxVector: 2});
  T.ok(JSON.stringify(per.raster) === '["cells","transcripts"]' && per.counts.boundaries.cell === 2 && planFigure(view, {maxVector: 1}).raster.includes('boundary:cell') &&
    planFigure(view, {maxVector: 'x'}).maxVector === 50000 && planFigure(view, {maxVector: 0}).raster.length === 3,
    '1 raster fallback: the limit applies to each layer (cells, outlines, transcripts)', JSON.stringify(per.raster));
  // bounded output: 200,000 cells in view stay a small file apart from the image
  {
    const n = 200000;
    const xy = new Float32Array(2 * n), rgba = new Uint8Array(4 * n).fill(255);
    for (let i = 0; i < n; i++) { xy[2 * i] = (i % 400) * 0.9 - 80; xy[2 * i + 1] = Math.floor(i / 400) * 0.5 - 40; }
    const big = {...view, cells: {...view.cells, n, xy, rgba}};
    const bp = planFigure(big);
    const bs = sceneToSvg(buildScene(big, bp, {cells: fake}));
    const atLimit = planFigure({...big, cells: {...big.cells, n: 50000}});
    const vec = sceneToSvg(buildScene({...big, cells: {...big.cells, n: 50000}}, atLimit));
    T.ok(bp.counts.cells === n && bp.raster[0] === 'cells' && bs.length < 20000 && atLimit.raster.length === 0 && count(vec, /<circle /g) === 50000 + 3 + 1 + 4 &&
      vec.length < 6e6, '1 bounded output: 200,000 cells in view give a small file with an image, exactly 50,000 stay vector',
    JSON.stringify({rasterSvgBytes: bs.length, vectorSvgBytes: vec.length}));
  }

  // PDF
  const pdfBytes = await sceneToPdf(scene);
  const pdfPath = resolve(OUT, 'synthetic.pdf');
  writeFileSync(pdfPath, pdfBytes);
  const pdf = readPdf(pdfBytes);
  T.ok(pdf.offsetsOk && /\/Root 1 0 R/.test(pdf.trailer) && /\/Info \d+ 0 R/.test(pdf.trailer) && /\/MediaBox \[0 0 [\d.]+ 600\]/.test(pdf.page.dict) &&
    /\/BaseFont \/Helvetica /.test(pdf.text) && Buffer.from(pdfBytes).toString('latin1').endsWith('%%EOF\n'),
    '1 PDF: header, objects at the offsets of the xref table, trailer, one page of the figure size', pdf.page.dict.slice(0, 120));
  const nCircles = 5 + 3 + 1 + 4; // cells, transcripts, overlay marker, legend swatches
  T.ok(count(pdf.content, / c f\n/g) === nCircles && count(pdf.content, / s\n/g) === 2 && pdf.content.startsWith('1 0 0 -1 0 600 cm\n') &&
    /\n1 0 0 rg\n206 100 m /.test(pdf.content) && /16 582 100 3 re f/.test(pdf.content) && /\(50 \\265m\) Tj/.test(pdf.content) &&
    /\(B & NK\) Tj/.test(pdf.content) && /\(ROI <1> & "x"\) Tj/.test(pdf.content) && /q 0 0 800 600 re W n/.test(pdf.content),
    '1 PDF: the same shapes as the SVG (circles, closed rings, 100-unit scale bar, text), y flipped once', JSON.stringify({circles: count(pdf.content, / c f\n/g)}));
  // every character of Windows-1252 has its byte (the fonts are WinAnsiEncoding); only what is
  // outside the code page prints as '?'
  const cp1252 = new TextDecoder('windows-1252');
  const high = [];
  for (let b = 0x80; b <= 0xff; b++) {
    const ch = cp1252.decode(Uint8Array.of(b));
    if (ch.codePointAt(0) >= 0xa0 || ch.codePointAt(0) > 0xff) high.push([b, ch]);
  }
  const wrong = high.filter(([b, ch]) => pdfText(ch) !== `(\\${b.toString(8)})`).map(([b, ch]) => `${ch}:${pdfText(ch)}`);
  T.ok(high.length === 123 && wrong.length === 0 && pdfText('Cellules œsophagiennes') === '(Cellules \\234sophagiennes)' &&
    pdfText('αβ 細胞 (x)') === '(?? ?? \\(x\\))',
    '1 PDF: all of Windows-1252 is written with its own byte; other characters as ?', wrong.slice(0, 6).join(' '));
  // a name with a code point XML 1.0 forbids must not make the SVG ill-formed
  const dirty = 'T\uffffcell\ufffe \u0001a\ud83d b \ud83e\udda0 <&>';
  T.ok(xmlText(dirty) === 'Tcell a b \ud83e\udda0 &lt;&amp;&gt;' && xmlText('tab\there\nline') === 'tab\there\nline',
    '1 SVG: code points XML does not allow (controls, U+FFFE/FFFF, lone surrogates) are dropped from text', JSON.stringify(xmlText(dirty)));
  const gsNames = [...pdf.page.dict.matchAll(/\/(GS\d+) << \/ca ([\d.]+) \/CA [\d.]+ >>/g)].map((m) => +m[2]);
  T.ok(gsNames.includes(0.8) && gsNames.includes(0.119) && gsNames.includes(1) && [...pdf.content.matchAll(/\/(GS\d+) gs/g)].every((m) => pdf.page.dict.includes(`/${m[1]} <<`)),
    '1 PDF: opacities as graphics states that the page declares', JSON.stringify(gsNames));
  const info = Object.values(pdf.objects).find((o) => /\/Title <FEFF/.test(o.dict));
  const utf16 = (h) => Buffer.from(h, 'hex').swap16().toString('utf16le');
  T.ok(info && utf16(/\/Title <FEFF([0-9A-F]*)>/.exec(info.dict)[1]) === 'Xenium spatial map' && /µm per figure unit/.test(utf16(/\/Subject <FEFF([0-9A-F]*)>/.exec(info.dict)[1])),
    '1 PDF: title and description in the document information');
  const rPdf = readPdf(await sceneToPdf(rScene));
  const imgs = Object.values(rPdf.objects).filter((o) => /\/Subtype \/Image/.test(o.dict));
  const rgbImg = imgs.find((o) => /DeviceRGB/.test(o.dict)), maskImg = imgs.find((o) => /DeviceGray/.test(o.dict));
  T.ok(rPdf.offsetsOk && imgs.length === 2 && rgbImg.stream.length === 12 && maskImg.stream.length === 4 && maskImg.stream[0] === 0 && maskImg.stream[1] === 255 &&
    /\/SMask \d+ 0 R/.test(rgbImg.dict) && /q 800 0 0 -600 0 600 cm \/Im1 Do Q/.test(rPdf.content) && count(rPdf.content, / c f\n/g) === nCircles - 5 &&
    /Xenium spatial map \(cells rasterized\)/.test(utf16(/\/Title <FEFF([0-9A-F]*)>/.exec(rPdf.text)[1])),
    '1 PDF raster fallback: the cell layer is an RGB image with a soft mask, upright over the map; said in the title');
  const cPdfBytes = await sceneToPdf(buildScene(contView, planFigure(contView)));
  writeFileSync(resolve(OUT, 'synthetic_continuous.pdf'), cPdfBytes);
  const cPdf = readPdf(cPdfBytes);
  const fn = Object.values(cPdf.objects).find((o) => /\/FunctionType 0/.test(o.dict));
  T.ok(cPdf.offsetsOk && /\/Sh1 sh/.test(cPdf.content) && /\/ShadingType 2 [^>]*\/Coords \[812 0 992 0\]/.test(cPdf.text) && fn && Buffer.compare(fn.stream, Buffer.from(lut)) === 0 &&
    /\(4\.25\) Tj/.test(cPdf.content) && /\(EEF1G \\\(log1p\\\)\) Tj/.test(cPdf.content),
    '1 PDF: continuous legend as an axial shading through the colour map table, with min / max / name');
  // right-aligned text ends where the bar ends (Helvetica metrics)
  const tm = /1 0 0 -1 ([\d.]+) [\d.]+ Tm 0 Tr [\d. ]+ rg \(4\.25\) Tj/.exec(cPdf.content);
  T.ok(tm && Math.abs(+tm[1] + textWidth('4.25', 11) - 992) < 0.011 && Math.abs(textWidth('4.25', 11) - 11 * (3 * 556 + 278) / 1000) < 1e-9,
    '1 PDF: right-aligned text is placed with the Helvetica widths', tm && tm[1]);
  for (const name of ['synthetic.pdf', 'synthetic_continuous.pdf']) {
    const q = quartzRender(resolve(OUT, name));
    if (q === null) console.log(`SKIP  1 ${name}: no macOS Quartz (sips) to render it with`);
    else T.ok(!q.error && q.width > 800 && q.height === 600 && q.bytes > 3000, `1 PDF: ${name} is rendered by an independent reader (macOS Quartz)`, JSON.stringify(q));
  }
}

// ── 2. the viewer on the dev server (human skin bundle) ──────────────────────────────────
let browser = null;
let server = null;
try {
  if (!hasBundle(DATA.skin)) {
    console.log(`SKIP  2 viewer export: no Xenium bundle at ${DATA.skin} (set XV_SKIN or XV_TEST_DATA)`);
  } else {
    server = await startServer(DATA.skin, +(process.env.XV_EXPORT_PORT || 8074));
    browser = await launch();
    const {page, consoleIssues, close} = await newPage(browser, {width: 1280, height: 800});
    const real = () => consoleIssues.filter((m) => !/GL Driver Message .*GPU stall due to ReadPixels/.test(m));
    const idle = () => page.evaluate(() => window.viewer.whenIdle());
    const setState = async (p) => { await page.evaluate((q) => window.viewer.setState(q), p); await idle(); };
    const setViewport = async (vp) => { await page.evaluate((v) => window.viewer.setViewport(v), vp); await sleep(250); await idle(); };
    /** exportFigure in the page → {info, text (svg) | b64 (pdf, png)} */
    const exportFigure = (opts) => page.evaluate(async (o) => {
      const r = await window.viewer.exportFigure(o);
      const out = {info: r.info, filename: r.filename, type: r.blob.type};
      if (o.format === 'svg' || !o.format) out.text = await r.blob.text();
      else {
        const b = new Uint8Array(await r.blob.arrayBuffer());
        let s = '';
        for (let i = 0; i < b.length; i += 32768) s += String.fromCharCode.apply(null, b.subarray(i, i + 32768));
        out.b64 = btoa(s);
      }
      return out;
    }, opts);
    /** Parses an SVG in the page (DOMParser) and reports what the checks need. */
    const inspect = (svg) => page.evaluate((text) => {
      const doc = new DOMParser().parseFromString(text, 'image/svg+xml');
      const q = (sel) => [...doc.querySelectorAll(sel)];
      const bar = doc.querySelector('#scale-bar');
      return {
        parseError: !!doc.querySelector('parsererror'),
        root: Object.fromEntries([...doc.documentElement.attributes].map((a) => [a.name, a.value])),
        title: doc.querySelector('svg > title').textContent, desc: doc.querySelector('svg > desc').textContent,
        cells: q('#cells circle').map((c) => [+c.getAttribute('cx'), +c.getAttribute('cy'), +c.getAttribute('r'), c.getAttribute('fill')]),
        cellImages: q('#cells image').length, images: q('image').map((i) => [i.parentNode.id, +i.getAttribute('width'), +i.getAttribute('height'), i.getAttribute('xlink:href').length]),
        rings: q('#boundaries-cell path').length, ringStroke: (doc.querySelector('#boundaries-cell > g') || {getAttribute: () => null}).getAttribute('stroke'),
        tx: q('#transcripts circle').length, txFills: [...new Set(q('#transcripts circle').map((c) => c.getAttribute('fill')))],
        bar: bar && {x: +bar.getAttribute('x'), width: +bar.getAttribute('width'), um: +bar.getAttribute('data-length-um')},
        barLabel: (doc.querySelector('#scale-bar-label') || {}).textContent,
        boxes: q('#legend .legend-box').map((b) => ({kind: b.getAttribute('data-kind'), title: b.querySelector('.legend-title').textContent,
          entries: [...b.querySelectorAll('.legend-entry')].map((e) => [e.querySelector('circle').getAttribute('fill'), e.querySelector('text').textContent]),
          min: (b.querySelector('.legend-min') || {}).textContent, max: (b.querySelector('.legend-max') || {}).textContent,
          bar: (b.querySelector('.legend-bar') || {getAttribute: () => null}).getAttribute('data-colormap')})),
        texts: q('text').length,
      };
    }, svg);

    await page.goto(server.url + '/');
    await page.waitForFunction(() => window.viewer && window.viewer.stats.firstRenderMs !== undefined, null, {timeout: 120000});
    await idle();
    const cellsBin = await fetchBin(`${server.url}/viewer/cells.bin?source=xenium`);
    const xy = cellsBin.arrays.xy, N = cellsBin.header.n;
    const colorsBin = await fetchBin(`${server.url}/viewer/colors.bin?source=xenium&mode=cluster`);
    const cats = colorsBin.header.categories, palette = colorsBin.header.colors, codes = colorsBin.arrays.codes;
    await setState({color: {mode: 'cluster'}});

    // a zoomed view around a cell in the middle of the tissue: about 300 µm wide
    const mid = Math.floor(N / 2);
    const cx = xy[2 * mid], cy = xy[2 * mid + 1];
    const vp = {x0: cx - 160, x1: cx + 160, y0: cy - 100, y1: cy + 100};
    await setViewport(vp);
    const got = await page.evaluate(() => ({vp: window.viewer.getViewport(), state: window.viewer.getState()}));
    const s = 1280 / (got.vp.x1 - got.vp.x0); // px per µm
    const a = await exportFigure({format: 'svg'});
    writeFileSync(resolve(OUT, 'skin_zoom.svg'), a.text);
    const A = await inspect(a.text);
    // the cells the server sends, projected the way the view does
    const size = Math.max(0.5, got.state.pointSize), r = Math.max(size / 2, Math.min(size * 2.5, 3 * s));
    const expect = [];
    for (let i = 0; i < N; i++) {
      const px = (xy[2 * i] - got.vp.x0) * s, py = (xy[2 * i + 1] - got.vp.y0) * s;
      if (px >= -r && px <= 1280 + r && py >= -r && py <= 800 + r) expect.push([px, py, palette[codes[i]].toLowerCase()]);
    }
    let worst = 0, colourMismatch = 0;
    if (A.cells.length === expect.length) {
      A.cells.forEach((c, k) => {
        worst = Math.max(worst, Math.abs(c[0] - expect[k][0]), Math.abs(c[1] - expect[k][1]));
        if (c[3] !== expect[k][2]) colourMismatch++;
      });
    }
    T.ok(!A.parseError && a.type === 'image/svg+xml' && a.filename === 'xenium_figure.svg' && A.cells.length === expect.length && expect.length > 50 && worst < 0.02 &&
      a.info.counts.cells === expect.length, '2 SVG: one circle per cell of cells.bin in view, at the projected position',
    JSON.stringify({circles: A.cells.length, expected: expect.length, worstPx: +worst.toFixed(4)}));
    T.ok(colourMismatch === 0 && A.cells.every((c) => Math.abs(c[2] - r) < 0.006), '2 SVG: every circle has the colour of its cluster and the radius drawn on screen',
      JSON.stringify({colourMismatch, r}));
    // scale bar against known cell coordinates: the two cells farthest apart in x among the exported ones
    let i0 = 0, i1 = 0;
    expect.forEach((e, k) => { if (e[0] < expect[i0][0]) i0 = k; if (e[0] > expect[i1][0]) i1 = k; });
    const inViewIdx = [];
    for (let i = 0; i < N; i++) {
      const px = (xy[2 * i] - got.vp.x0) * s, py = (xy[2 * i + 1] - got.vp.y0) * s;
      if (px >= -r && px <= 1280 + r && py >= -r && py <= 800 + r) inViewIdx.push(i);
    }
    const umApart = xy[2 * inViewIdx[i1]] - xy[2 * inViewIdx[i0]];
    const svgApart = A.cells[i1][0] - A.cells[i0][0];
    const svgPerUm = svgApart / umApart;
    T.ok(A.bar && [1, 2, 5].includes(+String(A.bar.um).replace(/0+$/, '')) && Math.abs(A.bar.width / A.bar.um / svgPerUm - 1) < 1e-3 &&
      A.bar.width <= 0.2 * 1280 + 0.01 && A.barLabel === `${A.bar.um} µm` && Math.abs(+A.root['data-um-per-px'] * svgPerUm - 1) < 1e-3,
    '2 scale bar: its length in SVG units over its µm equals the SVG units per µm between two known cells',
    JSON.stringify({bar: A.bar, label: A.barLabel, umApart: +umApart.toFixed(2), svgApart: +svgApart.toFixed(2), svgPerUm: +svgPerUm.toFixed(5)}));
    const box = A.boxes[0];
    T.ok(A.boxes.length === 1 && box.kind === 'categorical' && box.entries.length === cats.length &&
      box.entries.every((e, k) => e[0] === palette[k].toLowerCase() && e[1] === cats[k]) && A.images.length === 0 && A.root['data-rasterized'] === '',
    '2 legend: every cluster with its colour, as text; an all-vector figure has no image', JSON.stringify({title: box.title, n: box.entries.length}));

    // hidden categories leave the figure and the legend
    const hide = cats[codes[mid]];
    await setState({hiddenCategories: [hide]});
    const h = await inspect((await exportFigure({format: 'svg'})).text);
    const expectVisible = expect.filter((e) => e[2] !== palette[cats.indexOf(hide)].toLowerCase()).length;
    T.ok(h.cells.length === expectVisible && expectVisible < expect.length && h.boxes[0].entries.length === cats.length - 1 && !h.boxes[0].entries.some((e) => e[1] === hide),
      '2 hidden category: its cells and its legend entry are left out', JSON.stringify({circles: h.cells.length, expectVisible, hide}));
    await setState({hiddenCategories: []});

    // boundaries and transcripts as vectors
    await setState({boundaries: {cell: true}, transcripts: {genes: ['POSTN', 'LDHA']}});
    await sleep(400);
    await idle();
    const b = await exportFigure({format: 'svg'});
    writeFileSync(resolve(OUT, 'skin_zoom_boundaries.svg'), b.text);
    const B = await inspect(b.text);
    const loaded = await page.evaluate(() => {
      const v = window.viewer._viewer;
      return {rings: v._loaders.cell.data ? v._loaders.cell.data.header.n : 0, tx: v._loaders.tx.data ? v._loaders.tx.data.header.n : 0, status: window.viewer.getStatus()};
    });
    T.ok(B.rings > 50 && B.rings === b.info.counts.boundaries.cell && B.rings <= loaded.rings && B.ringStroke === '#00d4ff' && B.tx > 0 && B.tx === b.info.counts.transcripts &&
      B.tx <= loaded.tx && B.txFills.every((f) => ['#ff4d4d', '#33d6ff'].includes(f)) && B.boxes.length === 2 && B.boxes[1].kind === 'transcripts' &&
      JSON.stringify(B.boxes[1].entries) === JSON.stringify([['#ff4d4d', 'POSTN'], ['#33d6ff', 'LDHA']]),
    '2 boundaries and transcripts: outlines as paths in the boundary colour, transcripts as dots in the gene colours, with a legend box',
    JSON.stringify({rings: B.rings, loaded, tx: B.tx, fills: B.txFills}));

    // PDF and PNG of the same view
    const p = await exportFigure({format: 'pdf'});
    const pBytes = Buffer.from(p.b64, 'base64');
    const pPath = resolve(OUT, 'skin_zoom.pdf');
    writeFileSync(pPath, pBytes);
    const P = readPdf(new Uint8Array(pBytes));
    const nDots = B.cells.length + B.tx + cats.length + 2;
    T.ok(p.type === 'application/pdf' && p.filename === 'xenium_figure.pdf' && P.offsetsOk && count(P.content, / c f\n/g) === nDots && count(P.content, / s\n/g) === B.rings &&
      P.content.includes(`(${B.barLabel.replace('µ', '\\265')}) Tj`) && P.content.includes(`16 782 ${Math.round(B.bar.width * 100) / 100} 3 re f`) && /FlateDecode/.test(P.text),
    '2 PDF: the same cells, outlines, transcripts, legend swatches and scale bar as the SVG, compressed', JSON.stringify({bytes: pBytes.length, dots: count(P.content, / c f\n/g), nDots}));
    const q = quartzRender(pPath);
    if (q === null) console.log('SKIP  2 PDF rendered by macOS Quartz (sips): not on macOS');
    else T.ok(!q.error && q.width === Math.round(p.info.width) && q.height === 800 && q.bytes > 20000, '2 PDF: rendered by an independent reader (macOS Quartz)', JSON.stringify(q));
    for (const scale of [1, 3]) {
      const g = await exportFigure({format: 'png', scale});
      const png = Buffer.from(g.b64, 'base64');
      writeFileSync(resolve(OUT, `skin_zoom_${scale}x.png`), png);
      const dims = [png.readUInt32BE(16), png.readUInt32BE(20)];
      T.ok(g.type === 'image/png' && dims[0] === Math.round(g.info.width * scale) && dims[1] === 800 * scale && g.info.width > 1280 && png.length > 20000,
        `2 PNG at ${scale}x: map plus legend margin at the chosen scale`, JSON.stringify({dims, bytes: png.length}));
    }
    await setState({boundaries: {cell: false}, transcripts: {genes: []}});

    // continuous colours: colour bar with the gene name and the range on screen
    await setState({color: {mode: 'gene', gene: 'EEF1G'}});
    const c = await exportFigure({format: 'svg'});
    const C = await inspect(c.text);
    const dom = await page.evaluate(() => ({title: document.querySelector('.xv-legend .xv-title').textContent, min: document.querySelector('.xv-min').textContent,
      max: document.querySelector('.xv-max').textContent, rgba: Array.from(window.viewer._viewer._rgba.subarray(0, 0))}));
    T.ok(C.boxes.length === 1 && C.boxes[0].kind === 'continuous' && C.boxes[0].title === dom.title && /EEF1G/.test(dom.title) && C.boxes[0].min === dom.min &&
      C.boxes[0].max === dom.max && C.boxes[0].bar === 'viridis' && C.boxes[0].entries.length === 0 && C.cells.length === expect.length &&
      new Set(C.cells.map((x) => x[3])).size > 5, '2 continuous legend: colour bar with the name, min and max shown on screen',
    JSON.stringify({...C.boxes[0], dom, colours: new Set(C.cells.map((x) => x[3])).size}));
    await setState({color: {mode: 'cluster'}});

    // raster fallback: whole tissue in view
    await page.evaluate(() => window.viewer.fitView());
    await sleep(300);
    await idle();
    const f = await exportFigure({format: 'svg'});
    writeFileSync(resolve(OUT, 'skin_fit.svg'), f.text);
    const F = await inspect(f.text);
    T.ok(N > 50000 && f.info.counts.cells > 50000 && F.cells.length === 0 && F.cellImages === 1 && F.images.length === 1 && F.images[0][1] === 1280 && F.images[0][2] === 800 &&
      F.root['data-rasterized'] === 'cells' && /cells rasterized/.test(F.title) && new RegExp(`The cells layer \\(${f.info.counts.cells} in view, more than the vector limit of 50000\\) is embedded as an image`).test(F.desc) &&
      f.info.rasterized[0] === 'cells', '2 raster fallback: with more than 50,000 cells in view the cell layer is one embedded image, and the file says so',
    JSON.stringify({inView: f.info.counts.cells, bytes: f.info.bytes, title: F.title}));
    T.ok(f.info.bytes < 12e6 && F.bar && F.bar.um >= 100 && F.boxes[0].entries.length === cats.length && F.texts === cats.length + 2,
      '2 raster fallback: bounded size; scale bar and legend stay vector', JSON.stringify({bytes: f.info.bytes, bar: F.bar, texts: F.texts}));
    // the embedded image shows the cells: decode it and compare with a screenshot-free measure (non-transparent pixels)
    const img = await page.evaluate(async (text) => {
      const doc = new DOMParser().parseFromString(text, 'image/svg+xml');
      const im = new Image();
      im.src = doc.querySelector('#cells image').getAttribute('xlink:href');
      await im.decode();
      const cv = document.createElement('canvas');
      cv.width = im.width;
      cv.height = im.height;
      const g = cv.getContext('2d');
      g.drawImage(im, 0, 0);
      const d = g.getImageData(0, 0, cv.width, cv.height).data;
      let drawn = 0, clear = 0;
      const colours = new Set();
      for (let i = 0; i < d.length; i += 4) {
        if (d[i + 3] > 200) { drawn++; if (colours.size < 4000) colours.add((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]); } else if (d[i + 3] === 0) clear++;
      }
      return {w: im.width, h: im.height, drawn: drawn / (d.length / 4), clear: clear / (d.length / 4), colours: colours.size};
    }, f.text);
    T.ok(img.w === 2560 && img.h === 1600 && img.drawn > 0.02 && img.clear > 0.2 && img.colours > 5,
      '2 raster fallback: the image is the cell layer alone at 2x on a transparent background', JSON.stringify(img));
    // the limit is a parameter: raised above the cell count, the same view is all vector
    const v = await exportFigure({format: 'svg', maxVector: N + 1});
    const V = await inspect(v.text);
    T.ok(V.cells.length === f.info.counts.cells && V.images.length === 0 && v.info.rasterized.length === 0 && v.info.maxVector === N + 1,
      '2 limit raised above the cell count: every cell in view is a circle again', JSON.stringify({circles: V.cells.length, bytes: v.info.bytes}));
    const fp = await exportFigure({format: 'pdf'});
    const fpBytes = Buffer.from(fp.b64, 'base64');
    writeFileSync(resolve(OUT, 'skin_fit.pdf'), fpBytes);
    const FP = readPdf(new Uint8Array(fpBytes));
    const fpImgs = Object.values(FP.objects).filter((o) => /\/Subtype \/Image/.test(o.dict));
    const fq = quartzRender(resolve(OUT, 'skin_fit.pdf'));
    T.ok(FP.offsetsOk && fpImgs.length === 2 && fpImgs.some((o) => /\/Width 2560 \/Height 1600 [^>]*DeviceRGB[^>]*\/SMask/.test(o.dict)) && count(FP.content, / c f\n/g) === cats.length &&
      fpBytes.length < 12e6 && (fq === null || (!fq.error && fq.height === 800)),
    '2 raster fallback in PDF: one image with a soft mask, legend swatches and text as vectors', JSON.stringify({bytes: fpBytes.length, quartz: fq}));

    // the view is left as it was: frame, device pixel ratio and layer filter restored
    const after = await page.evaluate(() => ({filter: window.viewer.deck.props.layerFilter, dpr: window.viewer.deck.props.useDevicePixels,
      canvas: [document.querySelector('canvas.xv-canvas').width, document.querySelector('canvas.xv-canvas').height]}));
    T.ok(after.filter === null && after.dpr === true && after.canvas[0] === 1280 && after.canvas[1] === 800, '2 the view is restored after an export', JSON.stringify(after));
    let rejected = '';
    try { await exportFigure({format: 'tiff'}); } catch (err) { rejected = String(err.message); }
    T.ok(/unknown format/.test(rejected), '2 an unknown format is refused');
    T.ok(real().length === 0, '2 no console errors or warnings', JSON.stringify(real().slice(0, 3)));
    await close();
  }

  // ── 3. the control in the running app ──────────────────────────────────────────────────
  const base = process.argv[2];
  if (!base) {
    console.log('SKIP  3 "Export figure" control: no app URL given (see the header)');
  } else {
    browser = browser || await launch();
    const ctx = await browser.newContext({viewport: {width: 1500, height: 900}, deviceScaleFactor: 1, acceptDownloads: true});
    const page = await ctx.newPage();
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto(base, {waitUntil: 'domcontentloaded'});
    await page.waitForFunction(() => document.querySelector('#webgl-view canvas'), null, {timeout: 120000});
    await page.waitForFunction(() => performance.getEntriesByType('resource').some((e) => e.name.includes('/viewer/colors.bin')), null, {timeout: 120000});
    await page.waitForTimeout(3000);
    const ui = await page.evaluate(() => {
      const w = document.querySelector('#webgl-export');
      return w && {after: w.previousSibling.id, shown: getComputedStyle(w).display !== 'none', panel: getComputedStyle(document.querySelector('#webgl-export-panel')).display,
        button: document.querySelector('#webgl-export-btn').textContent, formats: [...document.querySelectorAll('#webgl-export-format option')].map((o) => o.value),
        limit: document.querySelector('#webgl-export-limit').value, text: document.querySelector('#webgl-export-panel').textContent};
    });
    T.ok(ui && ui.after === 'webgl-png-btn' && ui.shown && ui.panel === 'none' && ui.button === 'Export figure' && ui.formats.join() === 'svg,pdf,png' && ui.limit === '50000' &&
      /embedded as an image/.test(ui.text), '3 control: "Export figure" beside the PNG button, panel closed, limit of 50000 shown', JSON.stringify(ui));
    await page.click('#webgl-export-btn');
    const open = await page.evaluate(() => {
      const r = document.querySelector('#webgl-export-panel').getBoundingClientRect();
      return {display: getComputedStyle(document.querySelector('#webgl-export-panel')).display, inside: r.top >= 0 && r.left >= 0 && r.right <= innerWidth && r.bottom <= innerHeight, width: r.width};
    });
    T.ok(open.display === 'block' && open.inside, '3 control: the button opens the panel inside the window', JSON.stringify(open));
    const save = async (format, name, limit) => {
      await page.selectOption('#webgl-export-format', format);
      if (limit !== undefined) await page.fill('#webgl-export-limit', String(limit));
      const [dl] = await Promise.all([page.waitForEvent('download', {timeout: 60000}), page.click('#webgl-export-go')]);
      const file = resolve(OUT, name);
      await dl.saveAs(file);
      await page.waitForFunction(() => /saved/.test(document.querySelector('#webgl-export-msg').textContent), null, {timeout: 20000});
      return {name: dl.suggestedFilename(), bytes: readFileSync(file), msg: await page.evaluate(() => document.querySelector('#webgl-export-msg').textContent),
        status: await page.evaluate(() => document.querySelector('#webgl-status').textContent)};
    };
    const inView = await page.evaluate(() => window.XeniumViewer.instance && window.XeniumViewer.instance.getState ? 1 : 0);
    const s1 = await save('svg', 'app.svg');
    const svg = s1.bytes.toString('utf8');
    const nCells = +attr(svg, 'data-cells-in-view');
    const rasterized = attr(svg, 'data-rasterized') === 'cells';
    T.ok(inView === 1 && s1.name === 'xenium_figure.svg' && svg.startsWith('<?xml') && /<rect id="scale-bar"/.test(svg) && count(svg, /class="legend-box"/g) >= 1 &&
      (nCells > 50000) === rasterized && (rasterized ? /vector limit of 50000/.test(s1.msg) && /figure export: The cells layer/.test(s1.status) : count(svg, /<circle /g) >= nCells),
    '3 SVG download: scale bar and legend; above 50,000 cells in view the message and the status line say the layer is an image',
    JSON.stringify({nCells, rasterized, bytes: s1.bytes.length, msg: s1.msg.slice(0, 160)}));
    // a limit typed into the field decides: 0 forces the image, a large one forces vectors
    const s0 = await save('svg', 'app_limit0.svg', 0);
    const t0 = s0.bytes.toString('utf8');
    T.ok(attr(t0, 'data-rasterized') === 'cells' && attr(t0, 'data-max-vector') === '0' && /<g id="cells" data-raster="true"/.test(t0) && /vector limit of 0/.test(s0.msg),
      '3 the limit field is used: 0 embeds the cell layer as an image', s0.msg.slice(0, 160));
    const sp = await save('pdf', 'app.pdf', 50000);
    const P = readPdf(new Uint8Array(sp.bytes));
    const q = quartzRender(resolve(OUT, 'app.pdf'));
    T.ok(sp.name === 'xenium_figure.pdf' && P.offsetsOk && /Tj/.test(P.content) && (q === null || (!q.error && q.bytes > 10000)), '3 PDF download: a well-formed file',
      JSON.stringify({bytes: sp.bytes.length, quartz: q}));
    const sg = await save('png', 'app.png');
    const vb = await page.locator('#webgl-view').boundingBox();
    const dims = [sg.bytes.readUInt32BE(16), sg.bytes.readUInt32BE(20)];
    T.ok(sg.name === 'xenium_figure.png' && dims[1] === 2 * Math.round(vb.height) && dims[0] > 2 * Math.round(vb.width), '3 PNG download at 2x: map plus legend margin',
      JSON.stringify({dims, view: [vb.width, vb.height]}));
    // the morphology image, when shown, is embedded beneath the cells
    const setProp = (id, props) => page.evaluate(([i, q]) => window.dash_clientside.set_props(i, q), [id, props]);
    await setProp('morph-channels', {value: ['dapi']});
    await setProp('morph-enable', {value: ['show']});
    let imageShown = true;
    try {
      await page.waitForFunction(() => (window.XeniumViewer.instance.deck.props.layers || []).flat(3).some((l) => l && /^xenium-morphology/.test(l.id)), null, {timeout: 60000});
    } catch { imageShown = false; }
    if (!imageShown) {
      console.log('SKIP  3 image layer: the bundle shows no morphology image');
    } else {
      await page.waitForTimeout(6000); // tiles
      await page.screenshot({path: resolve(OUT, 'app_panel.png')});
      const si = await save('svg', 'app_image.svg', 50000);
      const ti = si.bytes.toString('utf8');
      const under = await page.evaluate(async (text) => {
        const doc = new DOMParser().parseFromString(text, 'image/svg+xml');
        const node = doc.querySelector('#map > #underlay > image');
        if (!node) return null;
        const im = new Image();
        im.src = node.getAttribute('xlink:href');
        await im.decode();
        const cv = document.createElement('canvas');
        cv.width = im.width;
        cv.height = im.height;
        const g = cv.getContext('2d');
        g.drawImage(im, 0, 0);
        const d = g.getImageData(0, 0, cv.width, cv.height).data;
        let lit = 0;
        for (let i = 0; i < d.length; i += 4) if (d[i + 3] > 0 && d[i] + d[i + 1] + d[i + 2] > 30) lit++;
        return {first: doc.querySelector('#map').firstElementChild.id, w: im.width, h: im.height, lit: lit / (d.length / 4), view: [+node.getAttribute('width'), +node.getAttribute('height')]};
      }, ti);
      const vbox = await page.locator('#webgl-view').boundingBox();
      T.ok(under && under.first === 'underlay' && under.w === 2 * Math.round(vbox.width) && under.h === 2 * Math.round(vbox.height) && under.view[0] === Math.round(vbox.width) && under.lit > 0.01 &&
        /The image \/ density layers are embedded as an image\./.test(ti) && /<rect id="scale-bar"/.test(ti),
      '3 image layer: embedded at 2x beneath the cell layer, with pixels of the image in it', JSON.stringify(under));
      const pi = await save('pdf', 'app_image.pdf');
      const PI = readPdf(new Uint8Array(pi.bytes));
      const qi = quartzRender(resolve(OUT, 'app_image.pdf'));
      T.ok(PI.offsetsOk && /\/Im1 Do Q[\s\S]*\/Im2 Do Q/.test(PI.content) && (qi === null || !qi.error), '3 image layer in PDF: image and cell layer as two images in order',
        JSON.stringify({bytes: pi.bytes.length, quartz: qi}));
      await setProp('morph-enable', {value: []});
      await page.waitForTimeout(1000);
    }
    // a click elsewhere closes the panel; the control is hidden in the Plotly renderer
    await page.mouse.click(700, 300);
    const closed = await page.evaluate(() => getComputedStyle(document.querySelector('#webgl-export-panel')).display);
    await page.evaluate(() => window.dash_clientside.set_props('renderer-mode', {value: 'plotly'}));
    await page.waitForFunction(() => !document.body.classList.contains('xv-webgl'), null, {timeout: 20000});
    const hidden = await page.evaluate(() => getComputedStyle(document.querySelector('#webgl-export')).display);
    await page.evaluate(() => window.dash_clientside.set_props('renderer-mode', {value: 'webgl'}));
    await page.waitForFunction(() => document.body.classList.contains('xv-webgl'), null, {timeout: 20000});
    await page.waitForTimeout(1500);
    const back = await page.evaluate(() => ({n: document.querySelectorAll('#webgl-export').length, shown: getComputedStyle(document.querySelector('#webgl-export')).display !== 'none'}));
    T.ok(closed === 'none' && hidden === 'none' && back.n === 1 && back.shown, '3 control: closes on an outside click, hidden in the Plotly renderer, one copy after switching back',
      JSON.stringify({closed, hidden, back}));
    T.ok(errors.length === 0, '3 no page errors', JSON.stringify(errors.slice(0, 3)));
    await ctx.close();
  }
} finally {
  if (browser) await browser.close();
  if (server) server.stop();
  if (process.env.XV_EXPORT_KEEP) console.log(`exported files kept in ${OUT}`);
  else rmSync(OUT, {recursive: true, force: true});
}

console.log(`\n${T.results.length - T.failed.length}/${T.results.length} checks passed`);
process.exit(T.failed.length ? 1 : 0);
