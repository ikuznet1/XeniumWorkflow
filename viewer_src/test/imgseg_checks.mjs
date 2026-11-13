// Drives the image-based resegmentation in the real app, in both renderers: the third tab of
// the Resegmentation modal, a classical-watershed run on a region, the new Segmentation
// Source entry, its boundary checkbox (WebGL and Plotly), and its row in Segmentation QC.
//   XENIUM_RENDERER=webgl PORT=8161 DATA_DIR=<bundle> XENIUM_CACHE_DIR=<dir> python xenium_explorer.py
//   [XI_REGION=x0,x1,y0,y1] [XI_BACKEND=classical|cellpose:nuclei] [XI_QC=celltypist] [XI_KEEP=1]
//   node test/imgseg_checks.mjs http://127.0.0.1:8161 /path/to/screenshot_dir
// Needs the SpatialData store and patches of the bundle in the cache (Prepare Patches then
// only reloads them). The run is deleted from the cache at the end unless XI_KEEP=1.
// XI_QC: annotation method for the Segmentation QC step (skipped when not given).
import {chromium} from 'playwright';

const base = process.argv[2] || 'http://127.0.0.1:8050';
const out = process.argv[3] || '.';
const gl = process.env.XV_GL || (process.platform === 'darwin' ? 'metal' : 'swiftshader');
const args = gl === 'metal'
  ? ['--use-angle=metal', '--ignore-gpu-blocklist']
  : ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'];
const browser = await chromium.launch({args});
let failures = 0;
const check = (name, ok, detail = '') => {
  if (!ok) failures += 1;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
};
const page = await browser.newPage({viewport: {width: 1700, height: 1000}});
const logs = [];
page.on('pageerror', (e) => logs.push('pageerror: ' + e.message.slice(0, 300)));
page.on('console', (m) => { if (m.type() === 'error') logs.push('console: ' + m.text().slice(0, 300)); });
await page.goto(base, {waitUntil: 'domcontentloaded'});
await page.waitForFunction(() => document.querySelector('#webgl-view canvas'), null, {timeout: 600000});
await page.waitForFunction(() => window.XeniumViewer.instance && window.XeniumViewer.instance._viewer._cells, null, {timeout: 600000});
await page.waitForTimeout(1500);

const setProp = (id, props) => page.evaluate(([i, p]) => window.dash_clientside.set_props(i, p), [id, props]);
const text = (sel) => page.evaluate((s) => document.querySelector(s)?.innerText || '', sel);
const click = (sel) => page.evaluate((s) => document.querySelector(s).click(), sel);
const shot = (name) => page.screenshot({path: `${out}/${name}.png`});
const reactProp = (id, name) => page.evaluate(([i, n]) => {
  const el = document.getElementById(i);
  const key = Object.keys(el).find((k) => k.startsWith('__reactFiber'));
  let f = el[key];
  for (let d = 0; f && d < 30; d += 1, f = f.return) {
    const p = f.memoizedProps;
    if (p && p[n] !== undefined && (p.id === i || n === 'options')) return p[n];
  }
  return null;
}, [id, name]);
const options = async (id) => (await reactProp(id, 'options') || []).map((o) => (o && o.value !== undefined ? o : {value: o}));

const region = (process.env.XI_REGION || '').split(',').map(Number).filter((v) => Number.isFinite(v));
const backend = process.env.XI_BACKEND || 'classical';

// ── 1. the modal and its third tab ───────────────────────────────────────────
await click('#reseg-modal-open-btn');
await page.waitForSelector('#reseg-modal-run-btn', {state: 'visible', timeout: 20000});
await page.locator('#sdata-prepare-btn').click();
await page.waitForSelector('#reseg-confirm-patches-btn', {state: 'visible', timeout: 1800000});
await page.locator('#reseg-confirm-patches-btn').click();
await page.waitForTimeout(800);
await setProp('reseg-algo-tabs', {value: 'imgseg'});
await page.waitForFunction(() => getComputedStyle(document.querySelector('#modal-imgseg-panel')).display !== 'none', null, {timeout: 20000});
await page.waitForTimeout(1200);
const panels = await page.evaluate(() => ['modal-baysor-panel', 'modal-proseg-panel', 'modal-imgseg-panel']
  .map((i) => getComputedStyle(document.getElementById(i)).display));
check('1 the Image tab shows its panel only', panels[0] === 'none' && panels[1] === 'none' && panels[2] !== 'none', panels.join(','));
const backends = await options('iseg-backend');
const chans = (await options('iseg-channel')).map((o) => o.value);
const note = await text('#iseg-backend-note');
check('1 the classical backend is offered and Cellpose only as far as it can run here',
  backends[0] && backends[0].value === 'classical' && backends.slice(1).every((o) => /^cellpose/.test(o.value)) && /Cellpose/.test(note),
  `${backends.map((o) => o.value + (o.disabled ? ' (disabled)' : '')).join(', ')} | ${note.slice(0, 160)}`);
check('1 the channels of the morphology image are listed, the nuclear one preselected',
  chans.length >= 1 && chans.includes(await reactProp('iseg-channel', 'value')), `${chans.join(', ')}; chosen ${await reactProp('iseg-channel', 'value')}`);
if (region.length === 4) {
  await setProp('iseg-xmin', {value: region[0]});
  await setProp('iseg-xmax', {value: region[1]});
  await setProp('iseg-ymin', {value: region[2]});
  await setProp('iseg-ymax', {value: region[3]});
}
if (backend !== 'classical') await setProp('iseg-backend', {value: backend});
await page.waitForTimeout(500);
await shot('iseg_1_modal_tab');

// ── 2. run ───────────────────────────────────────────────────────────────────
const before = (await options('seg-source')).map((o) => o.value);
let t0 = Date.now();
await page.locator('#reseg-modal-run-btn').click();
await page.waitForFunction(() => /Image segmentation/.test(document.querySelector('#reseg-status')?.innerText || '')
  || /^imgseg:/.test(document.querySelector('#seg-source input')?.value || ''), null, {timeout: 60000}).catch(() => {});
const jobRow = await page.evaluate(() => [...document.querySelectorAll('#jobs-panel .job-row')].map((r) => r.innerText.replace(/\s+/g, ' ')).join(' || '));
check('2 the run is a job in the jobs panel with Cancel', /Image segmentation/i.test(jobRow) && /Cancel/i.test(jobRow), jobRow.slice(0, 200));
await shot('iseg_2_running');
await page.waitForFunction(() => /^[✓✗]/.test(document.querySelector('#imgseg-status')?.innerText || ''), null, {timeout: 3600000});
const status = await text('#imgseg-status');
check('2 the run finished', /^✓ Done — [\d,]+ cells/.test(status), `${((Date.now() - t0) / 1000).toFixed(0)} s: ${status}`);
await page.waitForTimeout(4000);
const seg = await reactProp('seg-source', 'value');
const after = await options('seg-source');
const mine = after.find((o) => o.value === seg) || {};
check('2 the Segmentation Source switched to the new run, listed as a cached run',
  /^imgseg:[0-9a-f]{8}$/.test(String(seg)) && /Image:/.test(String(mine.label)), `${seg}: ${mine.label}; ${after.length - before.length} new entr${after.length - before.length === 1 ? 'y' : 'ies'}`);
const nCells = Number((status.match(/([\d,]+) cells/) || [])[1]?.replace(/,/g, ''));

// ── 3. WebGL: cells of the run and its boundaries ────────────────────────────
await page.waitForFunction((n) => {
  const v = window.XeniumViewer.instance && window.XeniumViewer.instance._viewer;
  return v && v._cells && v._cells.n === n;
}, nCells, {timeout: 120000}).catch(() => {});
const glN = await page.evaluate(() => window.XeniumViewer.instance._viewer._cells.n);
check('3 WebGL draws the cells of the run', glN === nCells, `${glN} drawn, ${nCells} in the run`);
const bopts = await options('boundary-toggles');
check('3 the boundary checkbox of the run is enabled', bopts.some((o) => o.value === 'imgseg' && !o.disabled), JSON.stringify(bopts.map((o) => [o.value, !!o.disabled])));
const vb = await page.evaluate(() => {
  const c = window.XeniumViewer.instance._viewer._cells;
  let sx = 0, sy = 0;
  const n = Math.min(c.n, 200);
  for (let i = 0; i < n; i += 1) { sx += c.xy[2 * i]; sy += c.xy[2 * i + 1]; }
  return {x: sx / n, y: sy / n};
});
const cx = region.length === 4 ? (region[0] + region[1]) / 2 : vb.x;
const cy = region.length === 4 ? (region[2] + region[3]) / 2 : vb.y;
await page.evaluate(([x, y]) => window.XeniumViewer.instance.setViewport({x0: x - 60, x1: x + 60, y0: y - 40, y1: y + 40}), [cx, cy]);
await setProp('boundary-toggles', {value: ['imgseg', 'nucleus']});
await page.waitForFunction(() => window.XeniumViewer.instance._viewer.deck.props.layers.some((l) => l.id === 'xv-boundary-imgseg'), null, {timeout: 60000}).catch(() => {});
await page.waitForTimeout(2500);
const layers = await page.evaluate(() => window.XeniumViewer.instance._viewer.deck.props.layers.map((l) => l.id));
check('3 WebGL draws the outlines of the run (next to the Xenium nuclei)', layers.includes('xv-boundary-imgseg') && layers.includes('xv-boundary-nucleus'), layers.join(','));
await shot('iseg_3_webgl_boundaries');

// ── 4. Plotly: the same ──────────────────────────────────────────────────────
await setProp('renderer-mode', {value: 'plotly'});
await page.waitForFunction(() => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  return gd && gd.layout && /Image segmentation/.test((gd.layout.title && gd.layout.title.text) || '');
}, null, {timeout: 180000}).catch(() => {});
await page.evaluate(([x, y]) => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  return window.Plotly.relayout(gd, {'xaxis.range[0]': x - 60, 'xaxis.range[1]': x + 60, 'yaxis.range[0]': -y - 40, 'yaxis.range[1]': -y + 40});
}, [cx, cy]);
await page.waitForFunction(() => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  return gd && gd.data && gd.data.some((t) => t.name === 'Image segmentation Boundaries');
}, null, {timeout: 180000}).catch(() => {});
await page.waitForTimeout(1500);
const pl = await page.evaluate(() => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  return {title: gd.layout.title.text, traces: gd.data.map((t) => t.name || ''), n: gd.data.filter((t) => t.type === 'scattergl' && t.mode === 'markers').reduce((s, t) => s + (t.x.length ?? (t.x.bdata ? t.x.bdata.length : 0)), 0)};
});
check('4 Plotly draws the run and its outlines', /\[Image segmentation\]/.test(pl.title) && pl.traces.includes('Image segmentation Boundaries') && pl.n > 0,
  `${pl.title.slice(0, 80)}; ${pl.n} cells in view; traces ${[...new Set(pl.traces)].slice(0, 8).join(', ')}`);
await shot('iseg_4_plotly_boundaries');
await setProp('renderer-mode', {value: 'webgl'});
await setProp('boundary-toggles', {value: []});
await page.waitForTimeout(1500);

// ── 5. Segmentation QC row ───────────────────────────────────────────────────
if (process.env.XI_QC) {
  await setProp('seg-source', {value: 'xenium'});
  await page.waitForTimeout(3000);
  if (await page.evaluate(() => getComputedStyle(document.querySelector('#umap-panel')).display === 'none')) await click('#umap-toggle');
  await setProp('plots-tabs', {active_tab: 'tab-segqc'});
  await page.waitForTimeout(2000);
  await setProp('segqc-method', {value: process.env.XI_QC});
  await page.waitForTimeout(800);
  t0 = Date.now();
  await click('#segqc-run-btn');
  await page.waitForFunction(() => /rows: \d+ segmentation source|^✗|rror|No |not /.test(document.querySelector('#segqc-status')?.innerText || '')
    && !/Running|…$/.test(document.querySelector('#segqc-status')?.innerText || ''), null, {timeout: 3600000});
  await page.waitForTimeout(2500);
  const qc = await text('#segqc-status');
  const rows = await page.evaluate(() => [...document.querySelectorAll('#segqc-table tbody tr')].map((r) => r.innerText.replace(/\s+/g, ' ').trim()).filter(Boolean));
  const row = rows.find((r) => /^Image \d/.test(r) || / Image: /.test(r));
  check('5 Segmentation QC has a row for the image-segmentation run', !!row, `${((Date.now() - t0) / 1000).toFixed(0)} s; ${row ? row.slice(0, 260) : qc.slice(0, 300)}`);
  await shot('iseg_5_segqc');
}

// ── 6. clean up ──────────────────────────────────────────────────────────────
if (!process.env.XI_KEEP) {
  await setProp('seg-source', {value: seg});
  await page.waitForTimeout(3000);
  await click('#seg-delete-btn');
  await page.waitForTimeout(800);
  await click('#seg-delete-confirm-btn');
  await page.waitForTimeout(3000);
  const left = (await options('seg-source')).map((o) => o.value);
  check('6 deleting the run removes it from the dropdown and returns to Xenium', !left.includes(seg) && (await reactProp('seg-source', 'value')) === 'xenium', left.filter((v) => /^imgseg/.test(v)).join(',') || 'no image runs left');
}
check('no page errors', logs.length === 0, logs.slice(0, 5).join(' | '));
console.log(failures ? `FAILS: ${failures}` : 'FAILS: 0');
await browser.close();
process.exit(failures ? 1 : 0);
