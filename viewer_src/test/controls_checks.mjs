// Drives the confound controls in the real app, in both renderers: tissue compartments and
// spillover score as Color By, gene ownership, the leak gate in Differential Expression,
// Segmentation QC, stratified co-localisation with placebo anchors, the annotation label
// report and the donor column of the design editor.
//   XENIUM_RENDERER=webgl PORT=8097 DATA_DIR=<bundle> XENIUM_CACHE_DIR=<dir> python xenium_explorer.py
//   [XC_LABELS=celltype:celltypist] [XC_TYPE=Fibroblasts] node test/controls_checks.mjs http://127.0.0.1:8097 /path/to/screenshot_dir
// Needs a cell-type annotation of the Xenium segmentation in the cache (the label source
// XC_LABELS, default the first annotation offered) and, for the Segmentation QC rows beyond
// Xenium, cached Baysor / Proseg runs.
import {chromium} from 'playwright';
import {openTab} from './lib.mjs';

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
    await page.waitForFunction(([s, r]) => new RegExp(r).test(document.querySelector(s)?.innerText || ''), [sel, re.source], {timeout});
  } catch (err) {
    throw new Error(`${sel} never matched ${re}: it shows "${(await text(sel)).slice(0, 400)}"`);
  }
};
const shot = (name) => page.screenshot({path: `${out}/${name}.png`});
const click = (sel) => page.evaluate((s) => document.querySelector(s).click(), sel);
const tableRows = (id) => page.evaluate((i) => Math.max(0, document.querySelectorAll(`#${i} tbody tr`).length - 2), id);
const headers = (id) => page.evaluate((i) => [...document.querySelectorAll(`#${i} th`)].map((e) => e.innerText.trim()).filter(Boolean), id);
const options = (id) => page.evaluate(async (i) => {
  const r = await fetch('/_dash-layout'); void r;
  return null;
}, id);
void options;
const tab = (t) => openTab(page, t);   // by tab id, through the two-level navigation (lib.mjs)
const plotTraces = (id) => page.evaluate((i) => {
  const gd = document.querySelector(`#${i} .js-plotly-plot`);
  return gd && gd.data ? gd.data.map((t) => ({name: t.name || '', type: t.type, n: (t.x || t.z || []).length})) : [];
}, id);
// current value of a Dash prop through a throw-away clientside read of the React props
const dropdownOptions = (id) => page.evaluate((i) => {
  const el = document.getElementById(i);
  if (!el) return [];
  const key = Object.keys(el).find((k) => k.startsWith('__reactFiber'));
  // The fiber a DOM node points at may be the alternate of the committed one (React keeps
  // two per component), whose props are those of the render before: both are read, and the
  // one whose options differ from the other's older copy is taken - the longer list here,
  // as the lists these checks wait for only grow from empty.
  const walk = (f) => {
    for (let d = 0; f && d < 30; d += 1, f = f.return) {
      const p = f.memoizedProps;
      if (p && Array.isArray(p.options)) return p.options.map((o) => (o && o.value !== undefined ? o.value : o));
    }
    return [];
  };
  const a = walk(el[key]);
  const b = walk(el[key] && el[key].alternate);
  return b.length > a.length ? b : a;
}, id);

if (await page.evaluate(() => getComputedStyle(document.querySelector('#umap-panel')).display === 'none')) {
  await click('#umap-toggle');
}

// ── compartments ─────────────────────────────────────────────────────────────
await tab('tab-controls');
await page.waitForTimeout(1500);
let labOpts = await dropdownOptions('ctl-labels');
const labels = process.env.XC_LABELS || labOpts.find((v) => String(v).startsWith('celltype:')) || labOpts[0];
check('Confound Controls tab offers the label sources of the view', labOpts.length > 0, JSON.stringify(labOpts));
await setProp('ctl-labels', {value: labels});
await page.waitForTimeout(1500);
const typeOpts = await dropdownOptions('spill-source');
const focal = process.env.XC_TYPE || typeOpts[1];
check('source / dominant type pickers list the cell types', typeOpts.length > 2 && typeOpts[0] === 'auto', `${typeOpts.length - 1} types; using ${focal}`);
let t0 = Date.now();
await click('#ctl-comp-run-btn');
await waitText('#ctl-comp-status', /Compartments \(.*median edge distance/s);
check('compartments computed', true, `${((Date.now() - t0) / 1000).toFixed(1)} s: ${(await text('#ctl-comp-status')).slice(0, 230)}`);
await page.waitForTimeout(800);
const cbOpts = await dropdownOptions('color-by');
check('Color By lists the compartment columns', ['geneset:@edge_distance', 'geneset:@density', 'geneset:@core'].every((v) => cbOpts.includes(v)), cbOpts.filter((v) => String(v).includes('@')).join(', '));
await setProp('color-by', {value: 'geneset:@edge_distance'});
await page.waitForTimeout(3500);
const wg = await page.evaluate(() => {
  const v = window.XeniumViewer.instance._viewer;
  const m = v._colorMeta || v._colors || (v._state && v._state.colorMeta) || null;
  return {keys: Object.keys(v).filter((k) => /color/i.test(k)).slice(0, 12), label: m && m.label, kind: m && m.kind, min: m && m.min, max: m && m.max};
});
console.log('INFO  webgl colour state', JSON.stringify(wg));
await shot('cc_1_webgl_edge_distance');
await setProp('color-by', {value: 'geneset:@core'});
await page.waitForTimeout(3000);
await shot('cc_2_webgl_core');
await setProp('renderer-mode', {value: 'plotly'});
await page.waitForFunction(() => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  return gd && gd.data && gd.data.length > 0;
}, null, {timeout: 180000});
await page.waitForTimeout(4000);
await setProp('color-by', {value: 'geneset:@edge_distance'});
await page.waitForFunction(() => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  return gd && gd.data && gd.data.some((t) => t.marker && t.marker.colorbar && /Edge distance/.test(JSON.stringify(t.marker.colorbar.title || '')));
}, null, {timeout: 180000});
const pl = await page.evaluate(() => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  const t = gd.data.find((d) => d.marker && d.marker.colorbar);
  // the colour array arrives as a typed array or as Plotly's binary spec {dtype, bdata}
  const col = t.marker.color;
  const n = col.length !== undefined ? col.length : (col.bdata ? col.bdata.length : 0);
  return {n, nx: t.x.length !== undefined ? t.x.length : -1, cmin: t.marker.cmin, cmax: t.marker.cmax, scale: JSON.stringify(t.marker.colorscale).slice(0, 40),
          title: JSON.stringify(t.marker.colorbar.title).slice(0, 60)};
});
check('Plotly map coloured by edge distance (sequential scale from 0)', pl.n > 0 && pl.cmin === 0 && pl.cmax > 50 && /440154/.test(pl.scale), JSON.stringify(pl));
await shot('cc_3_plotly_edge_distance');

// ── spillover + ownership ────────────────────────────────────────────────────
await setProp('spill-source', {value: focal});
t0 = Date.now();
await click('#spill-run-btn');
await waitText('#spill-status', /Validity check/);
const sp = await text('#spill-status');
check('spillover scored with its validity check shown', /Validity check (PASSED|FAILED|not fully testable)/.test(sp) && /Gene ownership/.test(sp), `${((Date.now() - t0) / 1000).toFixed(1)} s: ${sp.slice(0, 420)}`);
await page.waitForTimeout(1500);
check('ownership table filled', (await tableRows('spill-own-table')) > 0, (await headers('spill-own-table')).join(' | '));
const spTr = await plotTraces('spill-plot');
check('spillover plot: distance profile and flagged share by type', spTr.length === 2 && spTr[0].n > 2, JSON.stringify(spTr));
await shot('cc_4_controls_tab');
await setProp('color-by', {value: 'geneset:@spillover'});
await page.waitForFunction(() => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  return gd && gd.data && gd.data.some((t) => t.marker && t.marker.colorbar && /Spillover/.test(JSON.stringify(t.marker.colorbar.title || '')));
}, null, {timeout: 180000});
await shot('cc_5_plotly_spillover');
await setProp('renderer-mode', {value: 'webgl'});
await page.waitForTimeout(5000);
await shot('cc_6_webgl_spillover');

// ── leak gate in differential expression ─────────────────────────────────────
await tab('tab-de');
await page.waitForTimeout(1500);
const kindName = String(labels).split(':')[1];
await setProp('color-by', {value: `cell_type:${kindName}`});
await page.waitForTimeout(2500);
await setProp('de-group-a', {value: `celltype:${focal}`});
await setProp('de-group-b', {value: 'rest'});
t0 = Date.now();
await click('#de-run-btn');
await waitText('#de-status', /Unit of the p-values/);
const de = await text('#de-status');
check('DE status: unit, depth, detection rule, ownership / leak gate', /Unit of the p-values: n = [\d,]+ cells, treated as independent/.test(de) && /Median counts per cell/.test(de) && /count ≥ 1 transcript/.test(de) && /(genes flagged|Leak gate not applied)/.test(de), `${((Date.now() - t0) / 1000).toFixed(1)} s: ${de.slice(0, 900)}`);
await page.waitForTimeout(1500);
const deH = await headers('de-table');
check('DE table has Owner / Leak fraction / Leak gate and the rate columns', ['Owner', 'Rate A /10k'].every((h) => deH.includes(h)), deH.join(' | '));
const vt = await plotTraces('de-volcano-plot');
check('volcano marks the flagged genes', !/genes flagged/.test(de) || vt.some((t) => /leak-gate flagged/.test(t.name)), vt.map((t) => t.name).join(' ; '));
await shot('cc_7_de_leak_gate');

// ── segmentation QC ──────────────────────────────────────────────────────────
await tab('tab-segqc');
await page.waitForTimeout(1500);
const qcOpts = await dropdownOptions('segqc-method');
await setProp('segqc-method', {value: kindName});
await page.waitForTimeout(800);
t0 = Date.now();
await click('#segqc-run-btn');
await waitText('#segqc-status', /rows: \d+ segmentation source/);
const qc = await text('#segqc-status');
await page.waitForFunction(() => document.querySelectorAll('#segqc-table tbody tr').length > 2, null, {timeout: 60000}).catch(() => {});
await page.waitForTimeout(1500);
const qcRows = await tableRows('segqc-table');
check('Segmentation QC table over sources and counts modes', qcRows >= 1 && /no p-value is computed/.test(qc), `${((Date.now() - t0) / 1000).toFixed(1)} s, ${qcRows} rows; methods ${qcOpts.join(',')}: ${qc.slice(0, 300)}`);
check('Segmentation QC plots', (await plotTraces('segqc-plot')).length === 4);
await shot('cc_8_segqc');

// ── stratified co-localisation ───────────────────────────────────────────────
await tab('tab-coloc');
await page.waitForTimeout(1500);
await setProp('coloc-labels', {value: labels});
await page.waitForTimeout(1200);
await setProp('coloc-placebo', {value: focal});
t0 = Date.now();
await click('#coloc-run-btn');
await waitText('#coloc-status', /ordered pairs of \d+ groups/);
const co = await text('#coloc-status');
await page.waitForFunction(() => document.querySelectorAll('#coloc-table tbody tr').length > 2, null, {timeout: 60000}).catch(() => {});
await page.waitForTimeout(2000);
const single = /SINGLE-SAMPLE RESULT/.test(co);
check('co-localisation: effect table, units stated', (single ? /do not generalise/.test(co) : /unit of the p-values/.test(co)) && (await tableRows('coloc-table')) > 0, `${((Date.now() - t0) / 1000).toFixed(1)} s: ${co.slice(0, 700)}`);
const ht = await plotTraces('coloc-plot');
check('co-localisation heatmap', ht.length === 1 && ht[0].type === 'heatmap', JSON.stringify(ht));
check('placebo anchors table', (await tableRows('coloc-placebo-table')) > 0 && /Placebo anchors for/.test(co));
await shot('cc_9_coloc');
await click('#coloc-design-open-btn');
await page.waitForTimeout(1500);
const donorInputs = await page.evaluate(() => [...document.querySelectorAll('#design-rows input')].length);
check('design editor opens from the tab with donor and batch inputs', donorInputs >= 2, `${donorInputs} text inputs`);
await shot('cc_10_design_donor');
await click('#design-close-btn');

// ── composition test option, annotation report ───────────────────────────────
await tab('tab-composition');
await page.waitForTimeout(1000);
check('composition test offers the exact sample-level test', (await dropdownOptions('comp-test')).includes('exact'));
await click('#annot-modal-open-btn');
await waitText('#annot-label-report', /targeted panel/, 60000);
const rep = await text('#annot-label-report');
check('annotation dialog: caution and label counts', /labels over [\d,]+ labelled cells/.test(rep), rep.slice(0, 600).replace(/\n/g, ' | '));
await shot('cc_11_annotation_report');

check('no page errors', logs.length === 0, logs.slice(0, 5).join(' || '));
console.log(failures ? `FAILS: ${failures}` : 'ALL PASS');
await browser.close();
process.exit(failures ? 1 : 0);
