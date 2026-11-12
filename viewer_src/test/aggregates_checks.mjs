// Drives the Aggregates tab in the real app, in both renderers: run the detection for a
// class of cell types, the result table / plot, the aggregate id as Color By and the
// outlines on the map (WebGL overlay polygons, Plotly layout shapes), the outline toggle.
//   XENIUM_RENDERER=webgl PORT=8153 DATA_DIR=<bundle> XENIUM_CACHE_DIR=<dir> python xenium_explorer.py
//   [XA_LABELS=celltype:celltypist] [XA_CLASS="B cells|Plasma cells"] node test/aggregates_checks.mjs http://127.0.0.1:8153 /path/to/screenshot_dir
// Needs a cell-type annotation of the Xenium segmentation in the cache. Without XA_CLASS
// the third and fourth most abundant types are the class.
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
await page.waitForFunction(() => document.querySelector('#webgl-view canvas'), null, {timeout: 180000});
await page.waitForFunction(() => window.XeniumViewer.instance && window.XeniumViewer.instance._viewer._cells, null, {timeout: 180000});
await page.waitForTimeout(1500);

const setProp = (id, props) => page.evaluate(([i, p]) => window.dash_clientside.set_props(i, p), [id, props]);
const text = (sel) => page.evaluate((s) => document.querySelector(s)?.innerText || '', sel);
const waitText = async (sel, re, timeout = 600000) => {
  try {
    await page.waitForFunction(([s, r]) => new RegExp(r, 's').test(document.querySelector(s)?.innerText || ''), [sel, re.source], {timeout});
  } catch (err) {
    throw new Error(`${sel} never matched ${re}: it shows "${(await text(sel)).slice(0, 400)}"`);
  }
};
const shot = (name) => page.screenshot({path: `${out}/${name}.png`});
const click = (sel) => page.evaluate((s) => document.querySelector(s).click(), sel);
const tableRows = (id) => page.evaluate((i) => Math.max(0, document.querySelectorAll(`#${i} tbody tr`).length - 2), id);
const tab = async (t) => { await setProp('plots-tabs', {active_tab: t}); await page.waitForTimeout(1200); };
const dropdownOptions = (id) => page.evaluate((i) => {
  const el = document.getElementById(i);
  const key = Object.keys(el).find((k) => k.startsWith('__reactFiber'));
  let f = el[key];
  for (let d = 0; f && d < 30; d += 1, f = f.return) {
    const p = f.memoizedProps;
    if (p && Array.isArray(p.options)) return p.options.map((o) => (o && o.value !== undefined ? o.value : o));
  }
  return [];
}, id);
const overlayLabels = () => page.evaluate(() => {
  const o = window.XeniumViewer.instance._viewer.state.overlays || {};
  return (o.polygons || []).map((p) => p.label || '').filter((l) => /^A\d+$/.test(l));
});
const plotlyAggShapes = () => page.evaluate(() => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  return ((gd && gd.layout && gd.layout.shapes) || []).filter((s) => s.line && s.line.color === '#f0c040').length;
});
const plotlyReady = () => page.waitForFunction(() => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  return gd && gd.data && gd.data.length > 0;
}, null, {timeout: 180000});

if (await page.evaluate(() => getComputedStyle(document.querySelector('#umap-panel')).display === 'none')) {
  await click('#umap-toggle');
}

// ── run ──────────────────────────────────────────────────────────────────────
await tab('tab-agg');
await page.waitForTimeout(1500);
const labOpts = await dropdownOptions('agg-labels');
const labels = process.env.XA_LABELS || labOpts.find((v) => String(v).startsWith('celltype:')) || labOpts[0];
check('Aggregates tab offers the label sources of the view', labOpts.length > 0, JSON.stringify(labOpts));
await setProp('agg-labels', {value: labels});
await page.waitForTimeout(1500);
const typeOpts = await dropdownOptions('agg-classes');
const cls = process.env.XA_CLASS ? process.env.XA_CLASS.split('|') : typeOpts.slice(2, 4);
check('class picker lists the cell types', typeOpts.length > 2, `${typeOpts.length} types; class ${cls.join(' + ')}`);
await shot('agg_0_empty_state');
check('empty state before the first run', /No aggregates computed yet/.test(await text('#agg-plot')) && (await tableRows('agg-table')) === 0);
await click('#agg-run-btn');                       // no class picked: refused with a message, no job
await waitText('#agg-status', /Pick the cell types/);
check('a run without a class is refused with a message', true, await text('#agg-status'));
await setProp('agg-classes', {value: cls});
await page.waitForTimeout(500);
let t0 = Date.now();
await click('#agg-run-btn');
let jobsTxt = '';
for (let i = 0; i < 20 && !/Aggregates/.test(jobsTxt); i += 1) {
  await page.waitForTimeout(300);
  jobsTxt = await text('#jobs-panel');
}
check('the job is listed in the jobs panel with Cancel', /Aggregates/.test(jobsTxt) && /Cancel/i.test(jobsTxt), jobsTxt.replace(/\s+/g, ' ').slice(0, 120));
await waitText('#agg-status', /aggregate\(s\) of ≥/);
const secs = (Date.now() - t0) / 1000;
const status = await text('#agg-status');
check('aggregates computed', true, `${secs.toFixed(1)} s in the app: ${status.slice(0, 330)}`);
check('status states the unit of the p-values and the single-sample limit or the sample-level test',
      /Unit of the hotspot p-values: the cells of one section/.test(status) && /(SINGLE-SAMPLE RESULT|Between conditions)/.test(status));
await page.waitForTimeout(2500);
const nAgg = Number((status.match(/^([\d,]+) aggregate/) || [0, '0'])[1].replace(/,/g, ''));
const rows = await tableRows('agg-table');
check('aggregate table is filled', nAgg > 0 && rows > 0, `${nAgg} aggregates, ${rows} rows on the first page`);
check('per-sample table is filled', (await tableRows('agg-sample-table')) >= 1);
const traces = await page.evaluate(() => {
  const gd = document.querySelector('#agg-plot .js-plotly-plot');
  return gd && gd.data ? gd.data.map((t) => t.type + ':' + (t.x || []).length) : [];
});
check('plot shows density per sample and the pooled composition', traces.length >= 2 && traces.every((t) => !t.endsWith(':0')), traces.join(', '));
await shot('agg_1_tab_result');

// ── WebGL: outlines and Color By ─────────────────────────────────────────────
let labs = await overlayLabels();
check('WebGL: an outline polygon labelled with the id for every aggregate', labs.length === nAgg && new Set(labs).size === nAgg, `${labs.length} labelled polygons, ${new Set(labs).size} ids`);
const cbOpts = await dropdownOptions('color-by');
check('Color By lists the aggregate id', cbOpts.includes('geneset:agg#id'));
await setProp('color-by', {value: 'geneset:agg#id'});
await page.waitForTimeout(4000);
const legend = await text('#webgl-view');
check('WebGL: colour legend names the aggregate id', /Aggregate id/.test(legend), legend.replace(/\s+/g, ' ').slice(0, 100));
await shot('agg_2_webgl_color_outlines');
// zoom onto the largest aggregate
const first = await page.evaluate(() => {
  const v = window.XeniumViewer.instance._viewer;
  const p = (v.state.overlays.polygons || []).find((q) => q.label === 'A1');
  const xs = p.points.map((q) => q[0]), ys = p.points.map((q) => q[1]);
  const box = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  const pad = 0.6 * Math.max(box[2] - box[0], box[3] - box[1]);
  window.XeniumViewer.instance.setViewport({x0: box[0] - 1.6 * pad, x1: box[2] + 1.6 * pad, y0: box[1] - pad, y1: box[3] + pad});
  return {box};
});
await page.waitForTimeout(3500);
await shot('agg_3_webgl_zoom_A1');
console.log('INFO  A1 box (µm)', JSON.stringify(first));
await setProp('agg-outline-toggle', {value: []});
await page.waitForFunction(() => !((window.XeniumViewer.instance._viewer.state.overlays || {}).polygons || []).some((p) => p.color === '#f0c040'), null, {timeout: 30000});
check('WebGL: the toggle hides the outlines', (await overlayLabels()).length === 0);
await setProp('agg-outline-toggle', {value: ['on']});
await page.waitForFunction(() => ((window.XeniumViewer.instance._viewer.state.overlays || {}).polygons || []).some((p) => /^A\d+$/.test(p.label || '')), null, {timeout: 30000});
check('WebGL: and shows them again', (await overlayLabels()).length >= nAgg);

// ── Plotly: outlines and Color By ────────────────────────────────────────────
await setProp('renderer-mode', {value: 'plotly'});
await plotlyReady();
await page.waitForFunction(() => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  return gd && gd.data && gd.data.some((t) => t.marker && t.marker.colorbar && /Aggregate id/.test(JSON.stringify(t.marker.colorbar.title || '')));
}, null, {timeout: 180000});
await page.waitForTimeout(3000);
let nShapes = await plotlyAggShapes();
check('Plotly: outline shapes on the map', nShapes >= nAgg, `${nShapes} shapes`);
const pc = await page.evaluate(() => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  const t = gd.data.find((q) => q.marker && q.marker.colorbar);
  const full = gd._fullData.find((q) => q.marker && q.marker.colorbar);
  const c = Array.from((full && full.marker.color) || t.marker.color);
  const fin = c.filter((v) => Number.isFinite(v));
  return {n: c.length, inside: fin.length, max: fin.reduce((a, b) => Math.max(a, b), 0), cmax: t.marker.cmax};
});
check('Plotly: cells coloured by aggregate id (NaN outside)', pc.inside > 0 && pc.inside < pc.n && pc.max === nAgg, JSON.stringify(pc));
await shot('agg_4_plotly_color_outlines');
await page.evaluate((b) => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  const pad = 0.6 * Math.max(b[2] - b[0], b[3] - b[1]);
  return window.Plotly.relayout(gd, {'xaxis.range[0]': b[0] - pad, 'xaxis.range[1]': b[2] + pad,
                                     'yaxis.range[0]': -(b[3] + pad), 'yaxis.range[1]': -(b[1] - pad)});
}, first.box);
await page.waitForTimeout(4000);
check('Plotly: outlines stay after a zoom', (await plotlyAggShapes()) >= nAgg);
await shot('agg_5_plotly_zoom_A1');
await setProp('agg-outline-toggle', {value: []});
await page.waitForFunction(() => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  return !((gd.layout.shapes || []).some((s) => s.line && s.line.color === '#f0c040'));
}, null, {timeout: 60000});
check('Plotly: the toggle hides the outlines', (await plotlyAggShapes()) === 0);
await setProp('agg-outline-toggle', {value: ['on']});
await page.waitForFunction(() => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  return (gd.layout.shapes || []).some((s) => s.line && s.line.color === '#f0c040');
}, null, {timeout: 60000});

// ── alpha hull rerun, CSV ────────────────────────────────────────────────────
await setProp('agg-hull', {value: 'alpha'});
await setProp('agg-radius', {value: 30});
await page.waitForTimeout(400);
t0 = Date.now();
await click('#agg-run-btn');
await waitText('#agg-status', /radius 30 µm/);
check('rerun with radius 30 µm and alpha hulls', true, `${((Date.now() - t0) / 1000).toFixed(1)} s: ${(await text('#agg-status')).slice(0, 160)}`);
await page.waitForTimeout(5000);
nShapes = await plotlyAggShapes();
check('Plotly: outlines follow the new result', nShapes > 0, `${nShapes} shapes`);
await shot('agg_6_plotly_alpha_hull');
const [dl] = await Promise.all([page.waitForEvent('download', {timeout: 30000}), click('#agg-download-btn')]);
const csvPath = `${out}/aggregates.csv`;
await dl.saveAs(csvPath);
const fs = await import('node:fs');
const csv = fs.readFileSync(csvPath, 'utf8');
check('CSV has the samples, the aggregates and the composition', /# samples/.test(csv) && /# aggregates/.test(csv) && /# composition per aggregate/.test(csv), `${csv.split('\n').length} lines`);
await setProp('renderer-mode', {value: 'webgl'});
await page.waitForTimeout(5000);
labs = await overlayLabels();
check('WebGL: outlines of the new result after switching back', labs.length > 0, `${labs.length} polygons`);
await shot('agg_7_webgl_alpha_hull');

check('no page errors', logs.length === 0, logs.slice(0, 5).join(' | '));
console.log(failures ? `FAILURES: ${failures}` : 'ALL PASS');
await browser.close();
process.exit(failures ? 1 : 0);
