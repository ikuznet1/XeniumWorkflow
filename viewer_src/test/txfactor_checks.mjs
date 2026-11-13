// Drives the Transcript Factors tab in the real app: a fit, the table and the cross-tab, the
// map layer (dominant factor, one factor) in both renderers, the Color By values, the
// reference mode and the empty states.
//   PORT=8154 DATA_DIR=<bundle> XENIUM_CACHE_DIR=<dir> python xenium_explorer.py
//   [TXF_K=8] [TXF_BIN=10] [TXF_SHAPE=square|hex] [TXF_LABELS=celltype:celltypist] [TXF_REFERENCE=0]
//   node test/txfactor_checks.mjs http://127.0.0.1:8154 /path/to/screenshot_dir
// Needs a bundle with transcripts.parquet (or transcripts.zarr.zip). The first run on a
// bundle bins all transcripts (minutes on tens of millions); the counts are then cached.
import {chromium} from 'playwright';
import {openTab} from './lib.mjs';

const base = process.argv[2] || 'http://127.0.0.1:8050';
const out = process.argv[3] || '.';
const K = Number(process.env.TXF_K || 8);
const BIN = Number(process.env.TXF_BIN || 10);
const SHAPE = process.env.TXF_SHAPE || 'square';
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
const page = await browser.newPage({viewport: {width: 1700, height: 1100}});
const logs = [];
page.on('pageerror', (e) => logs.push('pageerror: ' + e.message.slice(0, 300)));
page.on('console', (m) => { if (m.type() === 'error') logs.push('console: ' + m.text().slice(0, 300)); });
await page.goto(base, {waitUntil: 'domcontentloaded'});
await page.waitForFunction(() => document.querySelector('#plots-tabs') && window.dash_clientside
  && window.dash_clientside.set_props, null, {timeout: 180000});
await page.waitForTimeout(4000);

const setProp = (id, props) => page.evaluate(([i, p]) => window.dash_clientside.set_props(i, p), [id, props]);
const text = (sel) => page.evaluate((s) => document.querySelector(s)?.innerText || '', sel);
const waitText = async (sel, re, timeout = 1800000) => {
  try {
    await page.waitForFunction(([s, r]) => new RegExp(r).test(document.querySelector(s)?.innerText || ''), [sel, re.source], {timeout});
  } catch (err) {
    throw new Error(`${sel} never matched ${re}: it shows "${(await text(sel)).slice(0, 400)}"`);
  }
};
const shot = (name) => page.screenshot({path: `${out}/${name}.png`});
const click = (sel) => page.evaluate((s) => document.querySelector(s).click(), sel);
const tableRows = (id) => page.evaluate((i) => [...document.querySelectorAll(`#${i} tbody tr`)].slice(2)
  .map((r) => [...r.querySelectorAll('td')].map((c) => c.innerText.trim())), id);
const headers = (id) => page.evaluate((i) => [...document.querySelectorAll(`#${i} th`)].map((e) => e.innerText.trim()).filter(Boolean), id);
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
const run = async (re) => {
  const t0 = Date.now();
  await click('#txf-run-btn');
  await page.waitForTimeout(300);
  await waitText('#txf-status', re);
  await page.waitForFunction(() => !document.querySelector('#txf-run-btn').disabled, null, {timeout: 1800000});
  await page.waitForTimeout(2000);      // the results callback redraws the plot and the table
  return (Date.now() - t0) / 1000;
};
const spatial = () => page.evaluate(() => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  const t = gd && gd._fullData ? gd._fullData[0] : null;
  if (!t) return null;
  const z = t.type === 'heatmap' ? [].concat(...Array.from(t.z, (r) => Array.from(r))).filter((v) => Number.isFinite(v)) : [];
  return {type: t.type, zmin: t.zmin, zmax: t.zmax, n: z.length, distinct: [...new Set(z)].length,
          colors: t.colorscale ? [...new Set(t.colorscale.map((c) => c[1]))].length : 0,
          ticks: (t.colorbar && t.colorbar.ticktext) || [], nTraces: gd._fullData.length,
          second: gd._fullData[1] ? gd._fullData[1].type : ''};
});
const webgl = () => page.evaluate(() => {
  const v = window.XeniumViewer.instance._viewer, d = v._loaders.density.data;
  const ids = (window.XeniumViewer.instance.deck.props.layers || []).flat(3).filter(Boolean).map((l) => l.id);
  const lg = document.querySelector('#webgl-view .xv-box[data-kind="density"]');
  return {h: d ? d.header : null, ids, title: lg?.querySelector('.xv-title')?.textContent || '',
          items: lg ? [...lg.querySelectorAll('.xv-item .xv-name')].map((e) => e.textContent) : [],
          swatches: lg ? [...lg.querySelectorAll('.xv-item .xv-swatch')].map((e) => e.style.background) : [],
          bar: !!lg?.querySelector('.xv-bar'),
          distinct: d && d.arrays.values ? [...new Set(Array.from(d.arrays.values))].length : 0};
});

if (await page.evaluate(() => getComputedStyle(document.querySelector('#umap-panel')).display === 'none')) {
  await click('#umap-toggle');
}
await setProp('renderer-mode', {value: 'plotly'});
await openTab(page, 'tab-txf');
await page.waitForSelector('#txf-run-btn', {state: 'visible'});
await page.waitForTimeout(2500);
check('the tab says what this is and what it is not',
  /in the spirit of FICTURE/.test(await text('#txf-note')) && /not that tool/.test(await text('#txf-note')));
const fresh = /No transcript factors computed yet/.test(await text('#txf-plot'));
check('empty state: placeholder figure, empty table, no layer to choose (or the result of an earlier run)',
  fresh ? (await tableRows('txf-table')).length === 0 && (await dropdownOptions('txf-layer')).length === 0 : true,
  fresh ? 'fresh server' : 'server already holds a result');
await shot('txf_0_empty');

// 1. guards
await setProp('txf-method', {value: 'reference'});
await setProp('txf-labels', {value: null});
await page.waitForTimeout(500);
await click('#txf-run-btn');
await waitText('#txf-status', /pick an annotation/);
check('reference mode without labels: asks for them, no job', /pick an annotation \(or clusters\)/.test(await text('#txf-status')));

// 2. a fit
const kinds = await dropdownOptions('txf-labels');
const kind = process.env.TXF_LABELS || kinds.find((k) => k.startsWith('celltype:')) || 'cluster';
check('label sources are offered', kinds.includes(kind), kinds.join(', '));
await setProp('txf-labels', {value: kind});
await setProp('txf-method', {value: 'nmf'});
await setProp('txf-k', {value: K});
await setProp('txf-bin', {value: BIN});
await setProp('txf-shape', {value: SHAPE});
await page.waitForTimeout(800);
const secs = await run(/factors \(NMF|Only .* bins hold|need the Xenium output/);
const status = await text('#txf-status');
console.log(status.split('\n').map((l) => '    | ' + l).join('\n'));
check('status: factors, bins, transcripts, the fit, the FICTURE statement, the cells',
  new RegExp(`^${K} factors \\(NMF \\(KL loss\\)\\) on [\\d,]+ ${BIN} µm ${SHAPE} bins holding [\\d,]+ gene transcripts \\(Q ≥ 20\\)`).test(status)
  && /in the spirit of FICTURE; not that tool/.test(status) && /lie over a bin with transcripts/.test(status),
  `${secs.toFixed(1)} s in the browser`);
check('status: the cross-tab with the labels (purity, adjusted Rand index)', /Dominant factor × .*purity \d\.\d\d.*adjusted Rand index/.test(status));
const rows = await tableRows('txf-table');
const hd = await headers('txf-table');
check('table: one row per factor with its share, most similar type and top genes',
  rows.length === Math.min(K, 12) && ['Factor', 'Share of tx', 'Most similar type', 'Cosine', 'Top genes'].every((h) => hd.includes(h))
  && rows[0][hd.indexOf('Factor')] === 'F1' && rows[0][hd.indexOf('Top genes')].split(',').length >= 5,
  rows.slice(0, 3).map((r) => `${r[0]}: ${r[hd.indexOf('Most similar type')]} (${r[hd.indexOf('Cosine')]}) ${r[hd.indexOf('Top genes')].slice(0, 50)}`).join(' | '));
const plot = await page.evaluate(() => {
  const gd = document.querySelector('#txf-plot .js-plotly-plot');
  return gd && gd._fullData ? {type: gd._fullData[0].type, nx: gd._fullData[0].x.length, ny: (gd._fullData[0].y || []).length} : null;
});
check('plot: cell types × dominant factor', plot && plot.type === 'heatmap' && plot.nx === K && plot.ny >= 2, JSON.stringify(plot));
await shot('txf_1_fit');

// 3. the map layer, Plotly
// the layer list is filled by a callback that follows the result: wait for it (condition, not a fixed time)
let layers = [];
for (let i = 0; i < 120 && layers.length !== K + 1; i += 1) {
  if (i) await page.waitForTimeout(500);
  layers = await dropdownOptions('txf-layer');
}
check('map layers: dominant factor and one per factor', layers.length === K + 1 && layers[0] === '@txf:dominant', layers.slice(0, 3).join(', '));
await setProp('txf-layer', {value: '@txf:dominant'});
await page.waitForFunction(() => document.querySelector('#spatial-plot .js-plotly-plot')?._fullData?.[0]?.type === 'heatmap', null, {timeout: 180000});
await page.waitForTimeout(1500);
let sp = await spatial();
check('Plotly: dominant-factor layer under the cells, one colour per factor',
  sp.type === 'heatmap' && sp.zmin === 0.5 && sp.zmax === K + 0.5 && sp.distinct >= 2 && sp.distinct <= K && sp.colors === K
  && sp.ticks.length === K && sp.nTraces >= 2, JSON.stringify({...sp, ticks: sp.ticks.slice(0, 3)}));
check('the density tab shows the same layer and its hint', /Dominant factor per/.test(await text('#density-hint')));
await setProp('point-opacity', {value: 0.15});
await page.waitForTimeout(2500);
await shot('txf_2_plotly_dominant');
await setProp('txf-layer', {value: '@txf:0'});
await page.waitForFunction(() => document.querySelector('#spatial-plot .js-plotly-plot')?._fullData?.[0]?.zmin === 0, null, {timeout: 180000});
await page.waitForTimeout(1500);
sp = await spatial();
check('Plotly: one factor as a continuous layer', sp.type === 'heatmap' && sp.zmin === 0 && sp.zmax > 0 && sp.distinct > K, JSON.stringify({...sp, ticks: []}));
await shot('txf_3_plotly_factor1');

// 4. the map layer, WebGL
await setProp('renderer-mode', {value: 'webgl'});
await page.waitForFunction(() => window.XeniumViewer && window.XeniumViewer.instance && window.XeniumViewer.instance._viewer, null, {timeout: 180000});
await page.waitForFunction(() => (window.XeniumViewer.instance.deck.props.layers || []).flat(3).some((l) => l && l.id === 'xv-density'), null, {timeout: 180000});
await page.waitForTimeout(2500);
let wg = await webgl();
check('WebGL: one factor as a heat layer under the cells with a colour bar',
  wg.ids.indexOf('xv-density') >= 0 && wg.ids.indexOf('xv-density') < wg.ids.indexOf('xv-cells') && wg.bar
  && /^Factor F1: transcripts per/.test(wg.title) && !wg.h.palette, `${wg.title}; layers ${wg.ids.join(',')}`);
await shot('txf_4_webgl_factor1');
await setProp('txf-layer', {value: '@txf:dominant'});
await page.waitForFunction(() => Array.isArray(window.XeniumViewer.instance._viewer._loaders.density.data?.header.palette), null, {timeout: 180000});
await page.waitForTimeout(2500);
wg = await webgl();
check('WebGL: dominant-factor layer with the factor colours in the legend',
  wg.h.palette.length === K && wg.items.length === K && wg.items[0] === 'F1' && !wg.bar && wg.distinct >= 2 && wg.distinct <= K + 1
  && new Set(wg.swatches).size === Math.min(K, new Set(wg.h.palette).size) && /^Dominant factor per/.test(wg.title),
  `${wg.title}; ${wg.items.slice(0, 4).join(', ')}…; grid ${wg.h.w}×${wg.h.h} of ${wg.h.cell} µm`);
await shot('txf_5_webgl_dominant');
// zoomed in: the bins themselves
const zoomed = await page.evaluate(() => {
  const v = window.XeniumViewer.instance._viewer;
  const vs = v.deck.viewState || v._viewState || null;
  return !!vs;
});
if (zoomed) {
  await page.mouse.move(700, 500);
  for (let i = 0; i < 8; i += 1) { await page.mouse.wheel(0, -400); await page.waitForTimeout(250); }
  await page.waitForTimeout(4000);
  wg = await webgl();
  check('WebGL, zoomed in: the layer is served at the bin size', wg.h && (wg.h.coarsened === false), `cell ${wg.h && wg.h.cell} µm`);
  await shot('txf_6_webgl_dominant_zoom');
}

// 5. Color By: the mixed score
const cb = await dropdownOptions('color-by');
check('Color By offers the mixed score and the factor shares',
  cb.includes('geneset:txf#mixed') && cb.includes('geneset:txf#top') && cb.includes('geneset:txf#share:0'));
await setProp('point-opacity', {value: 0.9});
await setProp('color-by', {value: 'geneset:txf#mixed'});
await page.waitForFunction(() => /mixed score/.test(document.querySelector('#webgl-view .xv-box[data-kind="continuous"] .xv-title')?.textContent || ''), null, {timeout: 120000});
await page.waitForTimeout(2000);
check('WebGL: cells coloured by the mixed score', true, await text('#webgl-view .xv-box[data-kind="continuous"] .xv-title'));
await shot('txf_7_webgl_mixed');
await setProp('renderer-mode', {value: 'plotly'});
await page.waitForFunction(() => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  return gd && gd._fullData && gd._fullData.some((t) => t.marker && t.marker.colorbar && /mixed score/.test(JSON.stringify(t.marker.colorbar.title || '')));
}, null, {timeout: 180000}).catch(() => {});
await page.waitForTimeout(2500);
const pm = await page.evaluate(() => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  return gd._fullData.map((t) => `${t.type}:${JSON.stringify((t.marker && t.marker.colorbar && t.marker.colorbar.title) || '').slice(0, 60)}`);
});
check('Plotly: cells coloured by the mixed score over the layer', pm.some((s) => /mixed score/.test(s)) && pm[0].startsWith('heatmap'), pm.join(' | '));
await shot('txf_8_plotly_mixed');

// 6. layer off
await setProp('txf-layer', {value: null});
await page.waitForFunction(() => document.querySelector('#spatial-plot .js-plotly-plot')?._fullData?.[0]?.type !== 'heatmap', null, {timeout: 180000});
check('clearing the layer removes it from the map', (await spatial()).type !== 'heatmap');
await setProp('color-by', {value: 'cluster'});

// 7. reference mode
if (process.env.TXF_REFERENCE !== '0' && kind !== 'cluster') {
  await setProp('txf-method', {value: 'reference'});
  await page.waitForTimeout(500);
  const rs = await run(/factors \(Reference|Fewer than 2 cell types|needs the count matrix/);
  const st = await text('#txf-status');
  console.log(st.split('\n').map((l) => '    | ' + l).join('\n'));
  const rr = await tableRows('txf-table');
  check('reference mode: the factors are the cell types, nothing is fitted',
    /Reference components: the mean profiles of \d+ cell types/.test(st) && /not fitted/.test(st)
    && /have their own type as the dominant factor/.test(st) && rr.length >= 2 && rr[0][0] !== 'F1',
    `${rs.toFixed(1)} s; ${rr.slice(0, 4).map((r) => r[0]).join(', ')}…`);
  await shot('txf_9_reference');
}

check('no page errors', logs.length === 0, logs.slice(0, 3).join(' || '));
console.log(failures ? `FAILURES: ${failures}` : 'ALL PASS');
await browser.close();
process.exit(failures ? 1 : 0);
