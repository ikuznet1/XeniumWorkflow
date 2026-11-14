// Round-11 regression & stress checks for WebGL figure export and overlay/label guards
// (viewer_src/src/figure.js, viewer_src/src/index.js, rebuilt assets/xenium_viewer.js).
//   node viewer_src/test/round11_figure_checks.mjs
import {inflateSync} from 'node:zlib';
import {readFileSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {buildScene, planFigure, sceneToPdf, sceneToSvg} from '../src/figure.js';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
let passed = 0;
let failed = 0;

function check(cond, msg) {
  if (cond) {
    passed += 1;
  } else {
    failed += 1;
    console.error('FAIL:', msg);
  }
}

function pdfContentText(pdfBytes) {
  const raw = Buffer.from(pdfBytes).toString('latin1');
  const streams = [];
  let pos = 0;
  while (true) {
    const idx = raw.indexOf('>>\nstream\n', pos);
    if (idx < 0) break;
    const dictStart = raw.lastIndexOf('<< ', idx);
    const dict = raw.slice(dictStart, idx + 2);
    const lenMatch = /\/Length (\d+)/.exec(dict);
    if (!lenMatch) break;
    const len = +lenMatch[1];
    const startOff = idx + 10;
    const chunk = Buffer.from(pdfBytes.buffer, pdfBytes.byteOffset + startOff, len);
    const decoded = /\/FlateDecode/.test(dict) ? inflateSync(chunk) : chunk;
    streams.push(decoded.toString('latin1'));
    pos = startOff + len;
  }
  return streams.join('\n');
}

// 1. Non-finite boundary vertices + degenerate rings + non-finite overlays/labels + [r,g,b] arrays
{
  const xy = new Float32Array([
    10, 20,
    50, 60,
    Number.NaN, 40,
    30, Number.POSITIVE_INFINITY,
  ]);
  const rgba = new Uint8Array([
    255, 0, 0, 220,
    0, 255, 0, 220,
    0, 0, 255, 220,
    255, 255, 0, 220,
  ]);

  // Ring 0: 4 vertices, 1 NaN -> 3 finite vertices remain (kept)
  // Ring 1: 3 vertices, 2 non-finite -> only 1 finite vertex remains (dropped)
  const bXY = new Float32Array([
    10, 20,   30, 20,   Number.NaN, 40,   20, 40,
    50, 50,   Number.NaN, Number.NaN,     Number.POSITIVE_INFINITY, 60,
  ]);
  const bStart = new Uint32Array([0, 4, 7]);

  const view = {
    width: 500,
    height: 500,
    scale: 2,
    target: [50, 50],
    background: '#0d1117',
    cells: {n: 4, xy, rgba, radiusUm: 3, minPx: 1.5, maxPx: 7.5, opacity: 0.8},
    boundaries: [
      {kind: 'cell', label: 'cell boundaries', n: 2, xy: bXY, start: bStart, color: [0, 212, 255, 220]},
    ],
    overlays: {
      lines: [
        {from: [10, 10], to: [80, 80], color: '#ff0000', width: 2, opacity: 0.8},
        {from: [Number.NaN, 10], to: [80, 80], color: '#ff0000', width: 2},
        {from: [10, 10], to: [80, Number.NEGATIVE_INFINITY], color: '#ff0000', width: 2},
      ],
      polygons: [
        // Aggregate-style [r, g, b] fill & stroke + 1 non-finite vertex mixed in
        {
          points: [[15, 15], [45, 15], [Number.NaN, 30], [45, 45], [15, 45]],
          fill: [248, 81, 73],
          fillOpacity: 0.25,
          color: [46, 213, 115],
          width: 1.8,
        },
        // Degenerate polygon (<3 finite vertices) -> must be dropped
        {
          points: [[10, 10], [Number.NaN, 20], [30, Number.POSITIVE_INFINITY]],
          fill: [255, 0, 0],
          color: [255, 0, 0],
        },
      ],
      points: [
        {at: [25, 35], color: [63, 185, 80], size: 6},
        {at: [Number.NaN, 35], color: [63, 185, 80], size: 6},
      ],
    },
    labels: [
      {x: 30, y: 30, text: 'Aggregate A', color: [248, 81, 73]},
      {x: Number.NaN, y: 30, text: 'Broken NaN Label', color: '#ffffff'},
    ],
  };

  const plan = planFigure(view);
  check(plan.counts.cells === 2, `planFigure finite cells count expected 2, got ${plan.counts.cells}`);
  check(plan.counts.boundaries.cell === 1,
    `planFigure finite boundary rings expected 1, got ${JSON.stringify(plan.counts.boundaries)}`);

  const scene = buildScene(view, plan);
  const mapGroup = scene.groups.find((g) => g.id === 'map');
  check(Boolean(mapGroup), 'scene missing map group');
  const cellGroup = mapGroup.items.find((it) => it.attrs?.id === 'cells');
  check(cellGroup && cellGroup.items[0]?.n === 2,
    `cells finite count expected 2, got ${cellGroup?.items?.[0]?.n}`);
  const bndGroup = mapGroup.items.find((it) => it.attrs?.id === 'boundaries-cell');
  check(bndGroup && bndGroup.items[0]?.start.length === 2,
    `boundaries-cell expected 1 ring (start len 2), got ${bndGroup?.items?.[0]?.start?.length}`);
  const ovGroup = mapGroup.items.find((it) => it.attrs?.id === 'overlays');
  check(Boolean(ovGroup), 'overlays group missing');
  check(ovGroup.items.filter((it) => it.t === 'line').length === 1, 'expected 1 finite overlay line');
  check(ovGroup.items.filter((it) => it.t === 'poly').length === 1, 'expected 1 finite overlay polygon');
  check(ovGroup.items.filter((it) => it.t === 'circles').length === 1, 'expected 1 finite overlay point');
  check(ovGroup.items.filter((it) => it.t === 'text').length === 1, 'expected 1 finite label');

  const svg = sceneToSvg(scene);
  check(!/\bNaN\b|\bInfinity\b/i.test(svg), 'SVG contains NaN or Infinity token');
  check(svg.includes('#f85149'), 'SVG missing polygon fill #f85149 from [248, 81, 73]');
  check(svg.includes('#2ed573'), 'SVG missing polygon stroke #2ed573 from [46, 213, 115]');
  check(svg.includes('#3fb950'), 'SVG missing point color #3fb950 from [63, 185, 80]');
  check(svg.includes('Aggregate A') && !svg.includes('Broken NaN Label'), 'SVG label filtering failed');

  const pdf = await sceneToPdf(scene);
  const pdfStreams = pdfContentText(pdf);
  check(!/\bNaN\b|\bInfinity\b/i.test(pdfStreams), 'PDF content stream contains NaN or Infinity token');
  check(pdfStreams.includes('0.973 0.318 0.286 rg'), 'PDF stream missing [248,81,73] fill rg');
  check(pdfStreams.includes('0.18 0.835 0.451 RG'), 'PDF stream missing [46,213,115] stroke RG');
}

// 2. Stress test: all-NaN boundary layer is omitted cleanly without empty starts buffer
{
  const view = {
    width: 300,
    height: 300,
    scale: 2,
    target: [25, 25],
    background: '#0d1117',
    cells: {
      n: 1,
      xy: new Float32Array([10, 10]),
      rgba: new Uint8Array([100, 150, 200, 255]),
      radiusUm: 2,
      minPx: 1,
      maxPx: 5,
      opacity: 1,
    },
    boundaries: [
      {
        kind: 'cell',
        label: 'cell boundaries',
        n: 1,
        xy: new Float32Array([Number.NaN, 1, 2, Number.NaN, Number.POSITIVE_INFINITY, 3]),
        start: new Uint32Array([0, 3]),
        color: [255, 255, 255, 220],
      },
    ],
  };
  const scene = buildScene(view, planFigure(view));
  const mapGroup = scene.groups.find((g) => g.id === 'map');
  check(!mapGroup.items.some((it) => it.attrs?.id === 'boundaries-cell'),
    'all-NaN boundary layer should be omitted');
  const svg = sceneToSvg(scene);
  check(!/\bNaN\b|\bInfinity\b/i.test(svg), 'all-NaN boundary SVG contains NaN/Infinity');
}

// 3. Built bundle sanity check: assets/xenium_viewer.js includes Viv image support + array-color handling
{
  const bundle = readFileSync(resolve(ROOT, 'assets/xenium_viewer.js'), 'utf8');
  check(bundle.length > 1_000_000, `assets/xenium_viewer.js unexpectedly small (${bundle.length} B; missing --with-image?)`);
  check(bundle.includes('rgbOf(p.fill)') || bundle.includes('Array.isArray'),
    'assets/xenium_viewer.js does not contain updated array-color handling');
}

if (failed > 0) {
  console.error(`round11_figure_checks: ${failed} failed, ${passed} passed`);
  process.exit(1);
}
console.log(`round11_figure_checks: ${passed} passed, 0 failed`);
