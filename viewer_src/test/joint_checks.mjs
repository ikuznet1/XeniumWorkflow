// Drives the Joint clustering tab in the real app, in both renderers: the refusal with one
// sample, adding samples, a run (Harmony on the sample when harmonypy is installed), the
// diagnostics plot / table, the shared cluster set under Cluster Set, the map coloured by it
// in WebGL and Plotly, and the joint UMAP of all samples in the UMAP panel of both.
//   XENIUM_RENDERER=webgl PORT=8171 DATA_DIR=<sample 0> XENIUM_CACHE_DIR=<dir> python xenium_explorer.py
//   XJ_SAMPLES=/path/s1,/path/s2 [XJ_INTEGRATE=sample|none|donor|batch] [XJ_UMAP=0] \
//     node test/joint_checks.mjs http://127.0.0.1:8171 /path/to/screenshot_dir
// XJ_SAMPLES: at least one further Xenium bundle, added through the "+ Add sample" dialog.
import {chromium} from 'playwright';
import {openTab} from './lib.mjs';

const base = process.argv[2] || 'http://127.0.0.1:8050';
const out = process.argv[3] || '.';
const extra = (process.env.XJ_SAMPLES || '').split(',').filter(Boolean);
const integrate = process.env.XJ_INTEGRATE || 'sample';
const wantUmap = process.env.XJ_UMAP !== '0';
if (extra.length < 1) {
  console.log('XJ_SAMPLES must name at least 1 further bundle');
  process.exit(2);
}
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
const waitText = async (sel, re, timeout = 1800000) => {
  try {
    await page.waitForFunction(([s, r]) => new RegExp(r, 's').test(document.querySelector(s)?.innerText || ''), [sel, re.source], {timeout});
  } catch (err) {
    throw new Error(`${sel} never matched ${re}: it shows "${(await text(sel)).slice(0, 400)}"`);
  }
};
const shot = (name) => page.screenshot({path: `${out}/${name}.png`});
const click = (sel) => page.evaluate((s) => document.querySelector(s).click(), sel);
const tableRows = (id) => page.evaluate((i) => Math.max(0, document.querySelectorAll(`#${i} tbody tr`).length - 2), id);
const tab = (t) => openTab(page, t);   // by tab id, through the two-level navigation (lib.mjs)
const dropdown = (id) => page.evaluate((i) => {
  const el = document.getElementById(i);
  const key = Object.keys(el).find((k) => k.startsWith('__reactFiber'));
  let f = el[key];
  for (let d = 0; f && d < 30; d += 1, f = f.return) {
    const p = f.memoizedProps;
    if (p && Array.isArray(p.options)) return {options: p.options.map((o) => ({value: o.value, label: String(o.label)})), value: p.value};
  }
  return {options: [], value: null};
}, id);
const glColours = () => page.evaluate(() => {
  const v = window.XeniumViewer.instance._viewer;
  const col = v._colors;
  if (!v._cells || !col) return null;
  const h = col.header || {};
  return {n: v._cells.n, kind: h.kind, label: h.label, categories: (h.categories || col.categories || []).map(String)};
});
const umapFig = () => page.evaluate(() => {
  const gd = document.querySelector('#umap-plot .js-plotly-plot') || document.querySelector('#umap-plot');
  if (!gd || !gd.data) return null;
  const title = gd.layout && gd.layout.title ? (gd.layout.title.text || String(gd.layout.title)) : '';
  return {title, traces: gd.data.length, points: gd.data.reduce((a, t) => a + (t.x ? (t.x.length ?? (t.x.bdata ? t.x.bdata.length : 0)) : 0), 0),
          names: gd.data.slice(0, 40).map((t) => t.name)};
});

if (await page.evaluate(() => getComputedStyle(document.querySelector('#umap-panel')).display === 'none')) {
  await click('#umap-toggle');
}

// ── one sample: refused ──────────────────────────────────────────────────────
await tab('tab-joint');
await page.waitForTimeout(1500);
await shot('joint_0_empty_state');
check('empty state before the first run', /No joint clustering computed yet/.test(await text('#jc-plot')) && (await tableRows('jc-table')) === 0);
check('with one sample the tab says what is needed', /One sample loaded: joint clustering needs at least two/.test(await text('#jc-design-summary')), (await text('#jc-design-summary')).slice(0, 140));
await click('#jc-run-btn');
await waitText('#jc-status', /at least two loaded samples/, 30000);
check('a run with one sample is refused with a message', true, await text('#jc-status'));

// ── add the samples ──────────────────────────────────────────────────────────
for (let i = 0; i < extra.length; i += 1) {
  await click('#sample-modal-open-btn');
  await page.waitForSelector('#sample-custom-path', {state: 'visible', timeout: 30000});
  await page.fill('#sample-custom-path', extra[i]);
  await page.click('#sample-add-btn');
  await waitText('#extra-samples-status', new RegExp(`${i + 2} samples loaded`), 900000);
  await page.waitForTimeout(800);
}
const nS = extra.length + 1;
await page.waitForTimeout(3000);
await page.evaluate(() => window.XeniumViewer.instance.fitView());
await tab('tab-joint');
await waitText('#jc-design-summary', new RegExp(`${nS} samples loaded`), 60000);
const intOpts = await dropdown('jc-integrate');
const noHarmony = intOpts.options.some((o) => /needs harmonypy/.test(o.label));
check('integration choices: none, sample, donor, batch', intOpts.options.map((o) => o.value).join(',') === 'none,sample,donor,batch', intOpts.options.map((o) => o.label).join(' | '));

// ── run ──────────────────────────────────────────────────────────────────────
await setProp('jc-integrate', {value: integrate});
if (!wantUmap) await setProp('jc-umap', {value: []});
await page.waitForTimeout(500);
const methodsBefore = (await dropdown('cluster-method')).options.map((o) => o.value);
const t0 = Date.now();
await click('#jc-run-btn');
let jobsTxt = '';
for (let i = 0; i < 30 && !/Joint clustering/.test(jobsTxt); i += 1) {
  await page.waitForTimeout(300);
  jobsTxt = await text('#jobs-panel');
}
check('the job is listed in the jobs panel with Cancel', /Joint clustering/.test(jobsTxt) && /Cancel/i.test(jobsTxt), jobsTxt.replace(/\s+/g, ' ').slice(0, 140));
await waitText('#jc-status', /clusters of [\d,]+ cells in \d+ samples|^(?!Running)(?!.*…$).{30,}/);
const secs = (Date.now() - t0) / 1000;
const status = await text('#jc-status');
const expectName = integrate === 'none' ? 'Joint clusters' : (noHarmony ? 'Joint clusters (uncorrected)' : 'Joint clusters (Harmony)');
check('joint clustering finished', new RegExp(`'${expectName.replace(/[()]/g, '\\$&')}': \\d+ clusters of [\\d,]+ cells in ${nS} samples`).test(status), `${secs.toFixed(1)} s in the app: ${status.slice(0, 420)}`);
if (integrate !== 'none' && !noHarmony) {
  check('status reports the mixing before and after Harmony', /before Harmony: \d+ clusters, mean entropy [\d.]+.*after: \d+ clusters, mean entropy [\d.]+/s.test(status), (status.match(/Mixing of the[^\n]*/) || [''])[0].slice(0, 330));
  check('status carries the confounding warning', /Integration removes differences between the levels/.test(status));
} else if (integrate !== 'none') {
  check('status gives the harmonypy install hint', /pip install harmonypy/.test(status));
}
await page.waitForTimeout(2500);
const k = Number((status.match(/: (\d+) clusters of/) || [0, '0'])[1]);
const rows = await tableRows('jc-table');
check('cluster table is filled', k > 0 && rows > 0, `${k} clusters, ${rows} rows on the first page`);
const traces = await page.evaluate(() => {
  const gd = document.querySelector('#jc-plot .js-plotly-plot');
  return gd && gd.data ? gd.data.map((t) => t.type + ':' + (t.x || []).length) : [];
});
check('plot: cluster shares by sample and the entropy curves', traces.filter((t) => t.startsWith('bar')).length === nS && traces.some((t) => t.startsWith('scatter')), traces.join(', '));
await shot('joint_1_tab_result');

// ── the cluster set, WebGL ───────────────────────────────────────────────────
await page.waitForTimeout(1500);
const methods = await dropdown('cluster-method');
const added = methods.options.find((o) => o.label === expectName);
check('Cluster Set lists the joint set', !!added, `${methods.options.length} sets (${methodsBefore.length} before); labels: ${methods.options.slice(-3).map((o) => o.label).join(' | ')}`);
const key = added ? added.value : null;
await setProp('color-by', {value: 'cluster'});
await setProp('cluster-method', {value: key});
await page.waitForTimeout(6000);
const gc = await glColours();
check('WebGL: cells of all samples are coloured by the joint clusters', !!gc && gc.kind !== 'continuous' && gc.n > 0, JSON.stringify(gc && {n: gc.n, kind: gc.kind, label: gc.label, categories: gc.categories.slice(0, 8), nCategories: gc.categories.length}));
if (gc && gc.categories.length) {
  const shared = gc.categories.every((c) => !/^S\d+:/.test(c));
  check('WebGL: the categories are shared across samples (no per-sample prefix)', shared && gc.categories.length <= k + 1, `${gc.categories.length} categories for ${k} clusters`);
}
await shot('joint_2_webgl_clusters');

// ── the joint UMAP in the UMAP panel, WebGL ──────────────────────────────────
const glUmap = () => page.evaluate(() => {
  const u = window.XeniumViewer.umapInstance;
  const v = u && u._viewer;
  if (!v || !v._cells) return null;
  const h = (v._colors && v._colors.header) || {};
  return {n: v._cells.n, source: v.state.source, kind: h.kind, categories: (h.categories || []).length};
});
const haveUmap = wantUmap && !/no joint UMAP/.test(status);
await tab('tab-umap');
await page.waitForTimeout(6000);
const nAll = gc ? gc.n : 0;
let gu = await glUmap();
if (haveUmap) {
  await page.waitForFunction((n) => {
    const u = window.XeniumViewer.umapInstance;
    return u && u._viewer && u._viewer._cells && u._viewer._cells.n === n;
  }, nAll, {timeout: 120000}).catch(() => {});
  await page.waitForTimeout(2500);
  gu = await glUmap();
  check('WebGL UMAP panel: the joint embedding holds the cells of all samples', !!gu && gu.n === nAll && /\|emb:/.test(gu.source), JSON.stringify(gu));
  check('WebGL UMAP panel: coloured by the joint clusters', !!gu && gu.kind === 'categorical' && gu.categories >= k && gu.categories <= k + 1, gu ? `${gu.categories} categories for ${k} clusters` : '');
  const png = (await page.locator('#webgl-umap').screenshot({path: `${out}/joint_3_webgl_umap_panel.png`})).length;
  check('WebGL UMAP panel: cells drawn', png > 20000, `PNG ${png} bytes`);
  await shot('joint_3_webgl_umap');
} else {
  console.log('SKIP  joint UMAP (not computed)');
}
// another cluster set: the panel goes back to the primary sample's own UMAP
if (methodsBefore.length && haveUmap) {
  await setProp('cluster-method', {value: methodsBefore[0]});
  await page.waitForFunction((n) => {
    const u = window.XeniumViewer.umapInstance;
    return u && u._viewer && u._viewer._cells && u._viewer._cells.n !== n;
  }, nAll, {timeout: 120000}).catch(() => {});
  gu = await glUmap();
  check('WebGL UMAP panel: another cluster set shows the sample\'s own UMAP again', !!gu && gu.n < nAll && !/emb:/.test(gu.source), JSON.stringify(gu));
  await setProp('cluster-method', {value: key});
  await page.waitForTimeout(3000);
}

// ── Plotly ───────────────────────────────────────────────────────────────────
await setProp('renderer-mode', {value: 'plotly'});
await page.waitForFunction(() => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  return gd && gd.data && gd.data.length > 0;
}, null, {timeout: 300000});
await page.waitForTimeout(6000);
const pl = await page.evaluate(() => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  const names = gd.data.filter((t) => t.x && (t.x.length || t.x.bdata) && t.name).map((t) => String(t.name));
  return {traces: gd.data.length, points: gd.data.reduce((a, t) => a + (t.x ? (t.x.length ?? (t.x.bdata ? t.x.bdata.length : 0)) : 0), 0), names: names.slice(0, 60)};
});
check('Plotly: the map draws the joint clusters without a per-sample prefix', pl.points > 0 && pl.names.length > 0 && pl.names.every((n) => !/^S\d+:C/.test(n)), `${pl.traces} traces, ${pl.points} points; names ${pl.names.slice(0, 8).join(', ')}`);
await shot('joint_5_plotly_clusters');
if (haveUmap) {
  await page.waitForFunction(() => {
    const gd = document.querySelector('#umap-plot .js-plotly-plot') || document.querySelector('#umap-plot');
    return gd && gd.layout && gd.layout.title && /Joint UMAP/.test(gd.layout.title.text || '');
  }, null, {timeout: 180000}).catch(() => {});
  await tab('tab-umap');
  await page.waitForTimeout(2500);
  let um = await umapFig();
  check('Plotly UMAP panel shows the joint embedding of all samples', !!um && /Joint UMAP/.test(um.title) && new RegExp(`${nS} samples`).test(um.title) && um.points > 0, um ? `${um.title}; ${um.traces} traces, ${um.points} points` : 'no figure');
  check('Plotly UMAP panel: one trace per joint cluster', !!um && um.traces === k, um ? um.names.join(',') : '');
  await shot('joint_5b_plotly_umap_clusters');
  await setProp('color-by', {value: 'transcript_counts'});
  await page.waitForFunction(() => {
    const gd = document.querySelector('#umap-plot .js-plotly-plot') || document.querySelector('#umap-plot');
    return gd && gd.layout && gd.layout.title && /· sample/.test(gd.layout.title.text || '');
  }, null, {timeout: 180000}).catch(() => {});
  um = await umapFig();
  check('Plotly UMAP panel: another Color By value colours the joint embedding by sample', !!um && um.traces === nS && /sample/.test(um.title), um ? `${um.title}; ${um.names.join(',')}` : '');
  await shot('joint_5c_plotly_umap_by_sample');
  await setProp('color-by', {value: 'cluster'});
  await page.waitForTimeout(4000);
}

// ── composition by the shared clusters, CSV ──────────────────────────────────
await tab('tab-joint');
const [dl] = await Promise.all([page.waitForEvent('download', {timeout: 30000}), click('#jc-download-btn')]);
const csvPath = `${out}/joint_clusters.csv`;
await dl.saveAs(csvPath);
const fs = await import('node:fs');
const csv = fs.readFileSync(csvPath, 'utf8');
check('CSV: header with the method and one row per cluster', /^# Joint clusters/.test(csv) && csv.split('\n').filter((l) => /^\d+,/.test(l)).length === k, `${csv.length} bytes`);
await click('#jc-design-open-btn');
await page.waitForSelector('#design-modal', {state: 'visible', timeout: 15000}).catch(() => {});
check('Samples & conditions opens the shared editor', await page.evaluate(() => !!document.querySelector('#design-rows') && document.querySelector('#design-rows').children.length > 0));
await shot('joint_6_design_editor');
await click('#design-close-btn');
await setProp('renderer-mode', {value: 'webgl'});
await page.waitForTimeout(1500);

const errs = logs.filter((l) => !/favicon|ResizeObserver/.test(l));
check('no page errors', errs.length === 0, errs.slice(0, 3).join(' | '));
console.log(`\n${failures === 0 ? 'ALL PASS' : failures + ' FAILED'}`);
await browser.close();
process.exit(failures ? 1 : 0);
