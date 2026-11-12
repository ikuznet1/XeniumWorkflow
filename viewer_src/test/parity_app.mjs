// Parity overlays of the WebGL view in the real Dash app, on fabricated results
// (a Proseg run, SpatialData ROI / patches, CellNEST edges, RCTD weights, a second sample).
//   DATA_DIR=<skin bundle> XENIUM_RENDERER=webgl PORT=8098 python test/launch_fab.py
//   node test/parity_app.mjs http://127.0.0.1:8098 /path/to/screenshot_dir [extra sample dir]
// launch_fab.py is the app plus /_fab/* routes that put the fabricated state in place.
import {chromium} from 'playwright';
import {readFileSync} from 'node:fs';

const base = process.argv[2] || 'http://127.0.0.1:8098';
const out = process.argv[3] || '.';
const extraDir = process.argv[4] || '/Users/ikuz/Documents/XeniumTestData/prime_mouse_ileum_tiny';
const gl = process.env.XV_GL || (process.platform === 'darwin' ? 'metal' : 'swiftshader');
const args = gl === 'metal' ? ['--use-angle=metal', '--ignore-gpu-blocklist']
  : ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'];
const browser = await chromium.launch({args});
const ctx = await browser.newContext({viewport: {width: 1500, height: 900}, acceptDownloads: true});
const page = await ctx.newPage();
const logs = [], reqs = [];
page.on('console', (m) => { if (m.type() === 'error') logs.push(`error: ${m.text().slice(0, 300)}`); });
page.on('pageerror', (e) => logs.push('pageerror: ' + e.message.slice(0, 300)));
page.on('request', (r) => { const u = new URL(r.url()); if (u.pathname.includes('/viewer/')) reqs.push(u.pathname + u.search); });
let failures = 0;
const check = (name, ok, detail = '') => { if (!ok) failures += 1; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`); };
const setProp = (id, props) => page.evaluate(([i, p]) => window.dash_clientside.set_props(i, p), [id, props]);
const fab = (what, q = '') => page.evaluate(async (u) => (await fetch(u)).json(), `/_fab/${what}${q}`);
const V = (fn, arg) => page.evaluate(fn, arg);
const state = () => V(() => { const s = window.XeniumViewer.instance.getState(); return {
  polys: s.overlays.polygons.length, lines: s.overlays.lines.length, points: (s.overlays.points || []).length,
  selected: s.selected ? s.selected.length : 0, boundaries: s.boundaries,
  layers: window.XeniumViewer.instance.deck.props.layers.map((l) => l.id), status: document.querySelector('#webgl-status').textContent}; });
const wait = (ms) => page.waitForTimeout(ms);
const idle = () => V(() => window.XeniumViewer.instance.whenIdle());
const pixels = async (name, tests) => {
  const buf = await page.locator('#webgl-view').screenshot({path: `${out}/${name}`});
  return V(async ({b64, tests}) => {
    const img = new Image(); img.src = `data:image/png;base64,${b64}`; await img.decode();
    const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
    const g = c.getContext('2d', {willReadFrequently: true}); g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height).data;
    const res = {};
    for (const [k, rgb, tol] of tests) { let n = 0; for (let i = 0; i < d.length; i += 4) if (Math.abs(d[i] - rgb[0]) + Math.abs(d[i + 1] - rgb[1]) + Math.abs(d[i + 2] - rgb[2]) <= tol) n++; res[k] = n; }
    return res;
  }, {b64: buf.toString('base64'), tests});
};

await page.goto(base, {waitUntil: 'domcontentloaded'});
await page.waitForFunction(() => document.querySelector('#webgl-view canvas'), null, {timeout: 120000});
await page.waitForFunction(() => window.XeniumViewer.instance && window.XeniumViewer.instance.stats.firstRenderMs !== undefined, null, {timeout: 120000});
await wait(2500);
const meta = await V(async () => (await fetch('/viewer/meta')).json());
const B = meta.bounds;
const cx = B[0] + (B[2] - B[0]) * 0.72, cy = B[1] + (B[3] - B[1]) * 0.45;
const view = (hw, hh) => V(([x, y, a, b]) => window.XeniumViewer.instance.setViewport({x0: x - a, x1: x + a, y0: y - b, y1: y + b}), [cx, cy, hw, hh]);
await setProp('point-opacity', {value: 0.35});
await view(150, 90);
await wait(1500);

// ── 1. per-source boundary toggles ──────────────────────────────────────────
await fab('proseg', `?x=${cx}&y=${cy}`);
await setProp('proseg-done', {data: 1});
let n = reqs.length;
await setProp('boundary-toggles', {value: ['cell', 'nucleus', 'proseg']});
await wait(3000); await idle();
let st = await state();
let kinds = reqs.slice(n).filter((u) => u.includes('boundaries.bin')).map((u) => new URL(base + u).searchParams.get('kind'));
let px = await pixels('parity_1_boundaries.png', [['cyan', [2, 185, 223], 60], ['green', [41, 186, 102], 60], ['orange', [222, 139, 61], 60]]);
check('1 Xenium cell + nucleus and Proseg boundaries drawn together under the Xenium source',
  ['cell', 'nucleus', 'proseg'].every((k) => kinds.includes(k) && st.layers.includes(`xv-boundary-${k}`)) && px.cyan > 1500 && px.green > 1500 && px.orange > 800,
  JSON.stringify({kinds: [...new Set(kinds)], px, status: st.status}));
n = reqs.length;
await setProp('boundary-toggles', {value: ['baysor']});
await wait(2000);
st = await state();
check('1 Baysor checkbox asks for the Baysor set only and says it is not available', reqs.slice(n).filter((u) => u.includes('boundaries.bin')).every((u) => u.includes('kind=baysor')) &&
  /Baysor boundaries not available/.test(st.status) && !st.layers.some((l) => /boundary-(cell|nucleus|proseg)/.test(l)), JSON.stringify(st.status));
await setProp('boundary-toggles', {value: []});

// ── 2. SpatialData tissue ROI / patch grid ──────────────────────────────────
await fab('sdata', `?x=${cx}&y=${cy}`);
const base0 = (await state()).polys;
n = reqs.length;
await setProp('sdata-show-roi', {value: ['yes']});
await wait(1500);
const roiOnly = (await state()).polys;
await setProp('sdata-show-patches', {value: ['yes']});
await wait(1500); await idle();
st = await state();
px = await pixels('parity_2_sdata.png', [['roi', [82, 183, 136], 50], ['patch', [255, 170, 0], 60]]);
check('2 sdata ROI and patch outlines are fetched and drawn', roiOnly === base0 + 1 && st.polys === base0 + 3 && px.roi > 300 && px.patch > 300 &&
  reqs.slice(n).some((u) => u.includes('sdata_roi=1') && u.includes('sdata_patches=1')), JSON.stringify({base0, roiOnly, polys: st.polys, px}));
await setProp('sdata-show-roi', {value: []}); await setProp('sdata-show-patches', {value: []});
await wait(1200);
check('2 outlines go when unticked', (await state()).polys === base0);

// ── 3. ruler ────────────────────────────────────────────────────────────────
const box = await page.locator('#webgl-view').boundingBox();
const panel0 = await V(() => document.querySelector('#cell-info-panel')?.innerText);
await page.click('#measure-btn');
await wait(800);
n = reqs.length;
const p1 = [box.x + 400, box.y + 300], p2 = [box.x + 900, box.y + 520];
const w1 = await V(([x, y]) => window.XeniumViewer.instance.unproject(x, y), [400, 300]);
const w2 = await V(([x, y]) => window.XeniumViewer.instance.unproject(x, y), [900, 520]);
await page.mouse.click(...p1);
await wait(1200);
const mid = {st: await state(), label: await V(() => document.querySelector('#measure-dist-display').textContent)};
await page.mouse.click(...p2);
await wait(1500);
st = await state();
const label = await V(() => ({text: document.querySelector('#measure-dist-display').textContent, shown: getComputedStyle(document.querySelector('#measure-dist-display')).display,
  store: window.dash_clientside.callback_context ? null : null}));
const dist = parseFloat(label.text);
const free = Math.hypot(w2[0] - w1[0], w2[1] - w1[1]);
px = await pixels('parity_3_ruler.png', [['ruler', [240, 192, 64], 40]]);
check('3 ruler: two clicks draw the segment and show the distance', mid.st.points === 1 && /second point/.test(mid.label) && st.points === 2 && st.lines === 1 &&
  label.shown !== 'none' && Math.abs(dist - free) < 25 && px.ruler > 500, JSON.stringify({label: label.text, clickDistance: free.toFixed(1), px, mid: mid.label}));
check('3 ruler clicks select no cell', !reqs.slice(n).some((u) => u.includes('/viewer/cell/')) && (await V(() => document.querySelector('#cell-info-panel')?.innerText)) === panel0);
await page.click('#measure-btn');
await wait(1000);
st = await state();
check('3 ruler off: drawing and label removed', st.points === 0 && st.lines === 0 && (await V(() => getComputedStyle(document.querySelector('#measure-dist-display')).display)) === 'none');

// ── 4. CellNEST edges ───────────────────────────────────────────────────────
await fab('cellnest', `?x=${cx}&y=${cy}`);
await view(110, 66);
await wait(1000);
n = reqs.length;
await setProp('cellnest-lr-pair', {options: ['LA-RA', 'LB-RB', 'LA-RB'], value: ['LA-RA', 'LB-RB']});   // (the dropdown drops values it has no option for)
await setProp('cellnest-overlay-enable', {value: ['on']});
await wait(2500); await idle();
st = await state();
const hdr = async (q) => V(async (q) => { const r = await fetch('/viewer/extra/cellnest.bin?source=xenium&q=' + encodeURIComponent(JSON.stringify(q)));
  return window.XeniumViewer.util.parseBin(await r.arrayBuffer()).header; }, q);
let h = await hdr({pairs: ['LA-RA', 'LB-RB'], top_n: 100});
px = await pixels('parity_4_cellnest.png', [['any', [13, 17, 23], 0]]);
check('4 CellNEST edges drawn as lines (shaft + arrowhead strokes)', h.edges > 20 && st.lines === 3 * h.edges && reqs.slice(n).some((u) => u.includes('/viewer/extra/cellnest.bin')),
  JSON.stringify({edges: h.edges, lines: st.lines}));
await setProp('cellnest-top-n', {value: 10});
await wait(1500);
check('4 top-N limits the edges per pair', (await state()).lines === 3 * 20, String((await state()).lines));
await setProp('cellnest-color-by', {value: 'attention'});
await setProp('cellnest-options', {value: ['edge_weight', 'gray_bg']});
await setProp('cellnest-edge-width', {value: 3});
await wait(2000); await idle();
st = await state();
const lineInfo = await V(() => { const l = window.XeniumViewer.instance.getState().overlays.lines; return {colors: new Set(l.map((d) => d.color.join())).size, widths: new Set(l.map((d) => d.width.toFixed(2))).size}; });
await page.locator('#webgl-view').screenshot({path: `${out}/parity_4b_cellnest_gray.png`});
check('4 colour by attention, width by attention and gray-out apply', lineInfo.colors > 2 && lineInfo.widths > 3 && st.selected > 5 && st.selected <= 40, JSON.stringify({lineInfo, selected: st.selected}));
await setProp('cellnest-overlay-enable', {value: []});
await wait(1200);
st = await state();
check('4 overlay off: edges and gray-out removed', st.lines === 0 && st.selected === 0);
await setProp('cellnest-options', {value: []});

// ── 5. RCTD weight pies ─────────────────────────────────────────────────────
await fab('rctd');
await view(60, 36);
n = reqs.length;
await setProp('color-by', {value: 'cell_type:rctd'});
await setProp('annot-done', {data: 99});
await wait(3500); await idle();
st = await state();
const pie = await V(() => { const p = window.XeniumViewer.instance.getState().overlays.polygons.filter((d) => d.width === 0); return {n: p.length, alpha: p[0] && p[0].fillAlpha, fills: new Set(p.map((d) => d.fill)).size}; });
await page.locator('#webgl-view').screenshot({path: `${out}/parity_5_pies.png`});
check('5 RCTD pies drawn for the cells in view', pie.n > 20 && pie.alpha === 217 && pie.fills >= 3 && reqs.slice(n).some((u) => u.includes('/viewer/extra/pies.bin')), JSON.stringify(pie));
await V(() => window.XeniumViewer.instance.fitView());
await wait(2500);
const pieOut = await V(() => window.XeniumViewer.instance.getState().overlays.polygons.filter((d) => d.width === 0).length);
check('5 no pies above PIE_THRESHOLD cells in view', pieOut === 0, String(pieOut));
n = reqs.length;
await setProp('color-by', {value: 'cluster'});
await wait(1500);
await view(60, 36); await wait(1200);
check('5 no pie requests for other colour modes', !reqs.slice(n).some((u) => u.includes('pies.bin')));

// ── 7. PNG export ───────────────────────────────────────────────────────────
const btn = await V(() => { const b = document.querySelector('#webgl-png-btn'); return {shown: getComputedStyle(b).display !== 'none', title: b.title}; });
const [dl] = await Promise.all([page.waitForEvent('download', {timeout: 20000}), page.click('#webgl-png-btn')]);
const file = `${out}/parity_7_export.png`;
await dl.saveAs(file);
const png = readFileSync(file);
const dims = [png.readUInt32BE(16), png.readUInt32BE(20)];
const vbox = await page.locator('#webgl-view').boundingBox();
check('7 PNG button downloads the view at 2x', btn.shown && /2x/.test(btn.title) && dl.suggestedFilename() === 'xenium_spatial.png' &&
  dims[0] === 2 * Math.round(vbox.width) && dims[1] === 2 * Math.round(vbox.height) && png.length > 50000, JSON.stringify({dims, view: [vbox.width, vbox.height], bytes: png.length}));

// ── 8. Z-level control ──────────────────────────────────────────────────────
await setProp('morph-enable', {value: ['show']});
await wait(1500);
const z1 = await V(() => ({disabled: !!document.querySelector('#morph-zlevel .Select.is-disabled, #morph-zlevel.dash-dropdown-disabled, #morph-zlevel [disabled], #morph-zlevel[disabled], #morph-zlevel [aria-disabled="true"]'),
  title: document.querySelector('#morph-zlevel-wrap').title, html: document.querySelector('#morph-zlevel').outerHTML.slice(0, 300)}));
await setProp('renderer-mode', {value: 'plotly'});
await wait(2500);
const z2 = await V(() => ({disabled: !!document.querySelector('#morph-zlevel .Select.is-disabled, #morph-zlevel.dash-dropdown-disabled, #morph-zlevel [disabled], #morph-zlevel[disabled], #morph-zlevel [aria-disabled="true"]'),
  title: document.querySelector('#morph-zlevel-wrap').title, png: getComputedStyle(document.querySelector('#webgl-png-btn')).display}));
await setProp('morph-enable', {value: []});
await setProp('renderer-mode', {value: 'webgl'});
await wait(2500);
check('8 Z-Level is disabled with a tooltip in WebGL mode, and normal in Plotly mode (PNG button hidden there)',
  z1.disabled && /WebGL/.test(z1.title) && !z2.disabled && z2.title === '' && z2.png === 'none', JSON.stringify({z1, z2}));

// ── 6. a second sample ──────────────────────────────────────────────────────
const n0 = await V(() => window.XeniumViewer.instance._viewer._cells.n);
const ex = await fab('extra', `?dir=${encodeURIComponent(extraDir)}`);
await setProp('extra-datasets-version', {data: 1});
await wait(5000); await idle();
const multi = await V(() => { const v = window.XeniumViewer.instance; return {n: v._viewer._cells.n, bounds: v._viewer._cells.bounds, lines: v.getState().overlays.lines,
  cats: (v._viewer._colors.header.categories || []).filter((c) => c.startsWith('S1:')).length, umapN: window.XeniumViewer.umapInstance ? window.XeniumViewer.umapInstance._viewer._cells.n : null}; });
await V(([x0, x1, y0, y1]) => window.XeniumViewer.instance.setViewport({x0, x1, y0, y1}), [B[2] - 600, ex.offset + 900, 0, 900]);
await wait(1500);
await page.locator('#webgl-view').screenshot({path: `${out}/parity_6_multi.png`});
const hit = await V(async (i) => (await fetch('/viewer/cell/' + i)).json(), n0 + 2);
check('6 second sample: cells, per-sample cluster names and separator', multi.n === n0 + ex.n && multi.bounds[2] > B[2] + 400 && multi.lines.length === 1 &&
  Math.abs(multi.lines[0].from[0] - (ex.offset - 250)) < 1 && multi.cats > 0 && hit.index === n0 + 2 && hit.x > B[2] + 400,
  JSON.stringify({n0, n: multi.n, extra: ex, lines: multi.lines.length, s1cats: multi.cats, umapN: multi.umapN, hit}));
await fab('extra_clear');
await setProp('extra-datasets-version', {data: 2});
await wait(4000);
check('6 removing the sample restores the single-sample view', (await V(() => window.XeniumViewer.instance._viewer._cells.n)) === n0 && (await state()).lines === 0);

check('no page errors', logs.length === 0, JSON.stringify(logs.slice(0, 6)));
console.log(failures === 0 ? 'ALL PASS' : `FAILURES: ${failures}`);
await browser.close();
process.exit(failures === 0 ? 0 : 1);
