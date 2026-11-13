// Drives the three analysis tabs (Differential Expression, Gene Sets, Transcript Density)
// in the real app, in both renderers:
//   XENIUM_RENDERER=webgl PORT=8097 python xenium_explorer.py <bundle>
//   node test/analysis_checks.mjs http://127.0.0.1:8097 /path/to/screenshot_dir
// It lassos two regions, saves them as groups, runs the test, clicks a volcano point,
// scores a gene set made of the top genes, shows a density grid at several bin sizes, and
// repeats the drawing part in the Plotly view. The groups and the gene set it creates are
// deleted again at the end (the density grids stay in the cache directory).
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
const TAG = 'xvtest_' + Date.now().toString(36);

const page = await browser.newPage({viewport: {width: 1700, height: 1000}});
const logs = [];
page.on('pageerror', (e) => logs.push('pageerror: ' + e.message.slice(0, 300)));
page.on('console', (m) => { if (m.type() === 'error') logs.push('console: ' + m.text().slice(0, 300)); });
await page.goto(base, {waitUntil: 'domcontentloaded'});
await page.waitForFunction(() => document.querySelector('#webgl-view canvas'), null, {timeout: 120000});
await page.waitForFunction(() => window.XeniumViewer.instance && window.XeniumViewer.instance._viewer._cells, null, {timeout: 120000});
await page.waitForTimeout(1500);

const setProp = (id, props) => page.evaluate(([i, p]) => window.dash_clientside.set_props(i, p), [id, props]);
const text = (sel) => page.evaluate((s) => document.querySelector(s)?.innerText || '', sel);
const waitText = async (sel, re, timeout = 180000) => {
  try {
    await page.waitForFunction(([s, r]) => new RegExp(r).test(document.querySelector(s)?.innerText || ''), [sel, re.source], {timeout});
  } catch (err) {
    throw new Error(`${sel} never matched ${re}: it shows "${await text(sel)}"`);
  }
};
const shot = (name) => page.screenshot({path: `${out}/${name}.png`});
const viewerState = () => page.evaluate(() => window.XeniumViewer.instance.getState());
const layerIds = () => page.evaluate(() => (window.XeniumViewer.instance.deck.props.layers || []).flat(3).filter(Boolean).map((l) => l.id));
const closeRoiDialog = async () => {
  await page.waitForFunction(() => [...document.querySelectorAll('.modal.show')].some((e) => /Save ROI/.test(e.innerText)), null, {timeout: 30000});
  await page.click('#roi-save-cancel-btn');
  await page.waitForFunction(() => ![...document.querySelectorAll('.modal.show')].some((e) => /Save ROI/.test(e.innerText)), null, {timeout: 30000});
};

// open the plots panel on the Differential Expression tab
if (await page.evaluate(() => getComputedStyle(document.querySelector('#umap-panel')).display === 'none')) {
  await page.evaluate(() => document.querySelector('#umap-toggle').click());   // (sidebar: may be scrolled out of view)
}
await openTab(page, 'tab-de');
await page.waitForSelector('#de-run-btn', {state: 'visible'});
check('DE tab: empty state (on a fresh server) and the statistical caveat',
  (/No differential expression computed yet/.test(await text('#de-plot-container')) || /FDR/.test(await text('#de-plot-container')))
  && /independent observation/.test(await text('#de-caveat')) && /pseudobulk/.test(await text('#de-caveat')));

const bounds = await page.evaluate(async () => (await (await fetch('/viewer/meta')).json()).bounds);
const at = (fx, fy) => [bounds[0] + (bounds[2] - bounds[0]) * fx, bounds[1] + (bounds[3] - bounds[1]) * fy];

// ── WebGL: two lasso regions → two saved groups ──────────────────────────────
async function lassoGl(center, halfUm, name) {
  await page.evaluate(([x, y, h]) => window.XeniumViewer.instance.setViewport({x0: x - h * 1.6, x1: x + h * 1.6, y0: y - h, y1: y + h}), [...center, halfUm]);
  await page.waitForTimeout(1200);
  const box = await page.locator('#webgl-view').boundingBox();
  const px = box.x + box.width / 2, py = box.y + box.height / 2;
  const r = Math.min(box.width, box.height) * 0.3;
  await setProp('webgl-tool', {value: 'lasso'});
  await page.waitForTimeout(600);
  await page.mouse.move(px - r, py - r * 0.7);
  await page.mouse.down();
  for (const [dx, dy] of [[r, -r], [r * 1.1, r * 0.5], [0.1 * r, r], [-r, r * 0.8], [-r, -r * 0.7]]) {
    await page.mouse.move(px + dx, py + dy, {steps: 6});
  }
  await page.mouse.up();
  await closeRoiDialog();
  await waitText('#de-selection-info', /Current selection: [\d,]+ cells/);
  const info = await text('#de-selection-info');
  await page.fill('#de-group-name', name);
  await page.click('#de-save-group-btn');
  await waitText('#de-status', new RegExp(`Saved the current selection as group '${name}'`));
  return Number(info.replace(/[^\d]/g, ''));
}
const nA = await lassoGl(at(0.72, 0.45), 260, TAG + '_A');
const nB = await lassoGl(at(0.30, 0.50), 260, TAG + '_B');
check('WebGL lasso → current selection → saved group (twice)', nA >= 10 && nB >= 10, `A ${nA} cells, B ${nB} cells`);
await page.evaluate(() => window.XeniumViewer.instance.fitView());

// ── run A vs B ───────────────────────────────────────────────────────────────
// (the pickers are refilled after a save; a value set before its option exists is dropped)
const pick = async (id, value, label) => {
  await page.click(`#${id}`);
  await page.locator(`text="${label}"`).last().click();
  await page.waitForFunction(([i, l]) => document.querySelector('#' + i).innerText.includes(l), [id, label], {timeout: 15000});
};
await page.waitForTimeout(1500);
await pick('de-group-a', `grp:${TAG}_A`, `Group: ${TAG}_A`);
await pick('de-group-b', `grp:${TAG}_B`, `Group: ${TAG}_B`);
let t0 = Date.now();
await page.click('#de-run-btn');
await waitText('#de-status', /genes, [\d,]+ with FDR < 0\.05/);
const deMs = Date.now() - t0;
const deStatus = await text('#de-status');
await page.waitForFunction(() => document.querySelectorAll('#de-table td').length > 20, null, {timeout: 60000});
await page.waitForTimeout(1200);
const de = await page.evaluate(() => {
  const gd = document.querySelector('#de-volcano-plot .js-plotly-plot');
  const rows = [...document.querySelectorAll('#de-table tr')].map((tr) => [...tr.querySelectorAll('td')].map((td) => td.innerText.trim())).filter((r) => r.length > 3 && r[0] && !/^filter/i.test(r[0]));
  return {traces: gd._fullData.map((t) => ({name: t.name, n: t.x.length})), title: gd.layout.title.text, rows: rows.slice(0, 12)};
});
check('DE ran: status with group sizes, volcano with 3 traces, table filled',
  new RegExp(`${nA.toLocaleString('en-US')} cells`).test(deStatus) && new RegExp(`${nB.toLocaleString('en-US')} cells`).test(deStatus)
  && de.traces.length === 3 && de.rows.length >= 10, `${deStatus}  [click-to-result ${deMs} ms]`);
await shot('an_1_webgl_de');

// sortable table: click the log2FC header twice (ascending, descending)
const colOf = async (i) => page.evaluate((k) => [...document.querySelectorAll('#de-table tr')].map((tr) => tr.querySelectorAll('td')[k]?.innerText.trim()).filter((v) => v !== undefined && v !== '').slice(0, 8).map(Number), i);
await page.click('#de-table th[data-dash-column="log2fc"] .column-header--sort');
await page.waitForTimeout(500);
const asc = await colOf(1);
await page.click('#de-table th[data-dash-column="log2fc"] .column-header--sort');
await page.waitForTimeout(500);
const desc = await colOf(1);
check('DE table sorts by a column', asc.length > 3 && asc.every((v, i) => i === 0 || v >= asc[i - 1]) && desc.every((v, i) => i === 0 || v <= desc[i - 1]),
  `asc ${asc.slice(0, 4)} desc ${desc.slice(0, 4)}`);
const topUp = await page.evaluate(() => [...document.querySelectorAll('#de-table tr')].map((tr) => tr.querySelectorAll('td')[0]?.innerText.trim()).filter(Boolean).slice(0, 12));

// click a volcano point → gene selector
const pt = await page.evaluate(() => {
  const gd = document.querySelector('#de-volcano-plot .js-plotly-plot');
  const t = gd._fullData[1].x.length ? gd._fullData[1] : gd._fullData[2];   // (decoded arrays)
  let k = 0;
  for (let i = 1; i < t.y.length; i++) if (t.y[i] > t.y[k]) k = i;
  const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis, r = gd.getBoundingClientRect();
  return {gene: t.customdata[k], x: r.left + xa._offset + xa.l2p(t.x[k]), y: r.top + ya._offset + ya.l2p(t.y[k])};
});
await page.mouse.move(pt.x, pt.y);
await page.waitForTimeout(300);
await page.mouse.click(pt.x, pt.y);
await page.waitForTimeout(1500);
const picked = await page.evaluate(() => document.querySelector('#gene-selector')?.innerText || '');
check('click on a volcano point sets the gene selector', picked.includes(pt.gene), `clicked ${pt.gene}, selector shows "${picked.trim().slice(0, 30)}"`);

// CSV download
const [dl] = await Promise.all([page.waitForEvent('download', {timeout: 30000}), page.click('#de-download-btn')]);
const csvPath = `${out}/an_de_table.csv`;
await dl.saveAs(csvPath);
const csv = (await import('node:fs')).readFileSync(csvPath, 'utf8').split('\n');
// comment lines (groups, unit of the p-values, depth, ownership / leak gate), then the table
const csvHead = csv.findIndex((l) => !l.startsWith('#'));
check('CSV download', /^# A: /.test(csv[0]) && csv.slice(0, csvHead).some((l) => /unit of the p-values/.test(l))
  && /^gene,log2fc,pct_a,pct_b/.test(csv[csvHead]) && csv.length > 100, `${dl.suggestedFilename()}, ${csv.length - csvHead - 2} genes, ${csvHead} header lines`);

// a group that is too small is refused
await page.evaluate(([x, y]) => window.XeniumViewer.instance.setViewport({x0: x - 20, x1: x + 20, y0: y - 13, y1: y + 13}), at(0.72, 0.45));
await page.waitForTimeout(1200);
{
  const box = await page.locator('#webgl-view').boundingBox();
  await setProp('webgl-tool', {value: 'rect'});
  await page.waitForTimeout(500);
  await page.mouse.move(box.x + box.width * 0.35, box.y + box.height * 0.35);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.65, box.y + box.height * 0.65, {steps: 6});
  await page.mouse.up();
  await page.waitForTimeout(1500);
  const modal = await page.evaluate(() => [...document.querySelectorAll('.modal.show')].some((e) => /Save ROI/.test(e.innerText)));
  if (modal) await page.click('#roi-save-cancel-btn');
  await page.waitForTimeout(800);
  await page.waitForTimeout(1500);
  await setProp('de-group-a', {value: 'sel:current'});
  await setProp('de-group-b', {value: 'rest'});
  await page.waitForTimeout(500);
  await page.click('#de-run-btn');
  await waitText('#de-status', /at least 10 are needed|current selection \(A, /);
  const st = await text('#de-status');
  check('a group under 10 cells is refused with a message', /Group A has \d+ cells; at least 10 are needed/.test(st),
    `${await text('#de-selection-info')} → ${st}`);
}
await setProp('webgl-tool', {value: 'pan'});
await page.evaluate(() => window.XeniumViewer.instance.fitView());

// ── gene set from the top genes ──────────────────────────────────────────────
await openTab(page, 'tab-geneset');
await page.waitForSelector('#geneset-run-btn', {state: 'visible'});
await page.fill('#geneset-name', TAG);
await page.fill('#geneset-genes', topUp.join('\n') + '\nNOT_A_GENE_1, not_a_gene_2');
t0 = Date.now();
await page.click('#geneset-run-btn');
await waitText('#geneset-status', /Pick 'Gene set: /);
const gsMs = Date.now() - t0;
const gsStatus = await text('#geneset-status');
await page.waitForFunction((t) => window.XeniumViewer.instance.getState().color.mode === 'geneset:' + t, TAG, {timeout: 30000});
await page.waitForFunction(() => /Gene set/.test(document.querySelector('#webgl-view .xv-box[data-kind="continuous"] .xv-title')?.textContent || ''), null, {timeout: 60000});
await page.waitForTimeout(1500);
const gs = await page.evaluate(() => {
  const v = window.XeniumViewer.instance._viewer, u = window.XeniumViewer.umapInstance && window.XeniumViewer.umapInstance._viewer;
  const lg = document.querySelector('#webgl-view .xv-box[data-kind="continuous"]');
  return {header: v._colors.header, cmap: v._colormap(), n: v._colors.values.length, cells: v._cells.n,
          min: lg.querySelector('.xv-min').textContent, max: lg.querySelector('.xv-max').textContent,
          bar: lg.querySelector('.xv-bar').style.background.slice(0, 60),
          umap: u && u._colors ? {label: u._colors.header.label, cmap: u._colormap()} : null,
          ctrl: getComputedStyle(document.querySelector('#colorbar-ctrl')).display};
});
check('gene set scored; missing genes reported; Color By switched to it', /Not in this dataset \(2\): NOT_A_GENE_1, not_a_gene_2/.test(gsStatus), `${gsStatus}  [click-to-result ${gsMs} ms]`);
check('WebGL: diverging scale centred at 0, one value per cell, range controls shown',
  gs.header.colormap === 'rdbu' && gs.cmap === 'rdbu' && Math.abs(gs.header.min + gs.header.max) < 1e-6 && gs.n === gs.cells && gs.ctrl !== 'none',
  `label "${gs.header.label}", range ${gs.min} … ${gs.max}`);
await setProp('colorbar-max', {value: 1});
await setProp('colorbar-min', {value: -1});
await page.waitForTimeout(900);
const rng = await page.evaluate(() => [document.querySelector('#webgl-view .xv-min').textContent, document.querySelector('#webgl-view .xv-max').textContent]);
check('WebGL: colour range inputs apply to the score', rng[0] === '-1' && rng[1] === '1', rng.join(' … '));
await shot('an_2_webgl_geneset');
await openTab(page, 'tab-umap');
await page.waitForTimeout(2500);
const uGs = await page.evaluate(() => {
  const u = window.XeniumViewer.umapInstance && window.XeniumViewer.umapInstance._viewer;
  return u && u._colors ? {label: u._colors.header.label, cmap: u._colormap(), n: u._colors.values.length} : null;
});
check('WebGL UMAP is coloured by the score', !!uGs && /Gene set/.test(uGs.label) && uGs.cmap === 'rdbu', JSON.stringify(uGs));
await shot('an_3_webgl_geneset_umap');
await page.evaluate(() => document.querySelector('#colorbar-reset-btn').click());

// ── transcript density ───────────────────────────────────────────────────────
await openTab(page, 'tab-density');
await page.waitForSelector('#density-run-btn', {state: 'visible'});
const gene = pt.gene;
await setProp('density-gene', {value: gene});
await page.waitForTimeout(800);
const hint0 = await text('#density-hint');
t0 = Date.now();
await page.click('#density-run-btn');
await waitText('#density-status', /Binned|Grid loaded from the cache/);
const dnMs = Date.now() - t0;
const dnStatus = await text('#density-status');
await page.waitForFunction(() => (window.XeniumViewer.instance.deck.props.layers || []).flat(3).some((l) => l && l.id === 'xv-density'), null, {timeout: 60000});
await page.waitForTimeout(1500);
const dens = async () => page.evaluate(() => {
  const v = window.XeniumViewer.instance._viewer, d = v._loaders.density.data;
  const ids = (window.XeniumViewer.instance.deck.props.layers || []).flat(3).filter(Boolean).map((l) => l.id);
  const lg = document.querySelector('#webgl-view .xv-box[data-kind="density"]');
  let sum = 0;
  for (const x of d.arrays.values) sum += x;
  return {h: d.header, sum, ids, title: lg?.querySelector('.xv-title').textContent, max: lg?.querySelector('.xv-max').textContent,
          opacity: (window.XeniumViewer.instance.deck.props.layers || []).flat(3).find((l) => l && l.id === 'xv-density').props.opacity};
});
let d10 = await dens();
check('density computed as a job; heat layer under the cells with its colour bar',
  (hint0.includes('press Compute') || /Grid loaded/.test(dnStatus)) && d10.ids.indexOf('xv-density') >= 0 && d10.ids.indexOf('xv-density') < d10.ids.indexOf('xv-cells')
  && d10.title === `${gene}: transcripts per 10 µm bin` && Number(d10.max) === d10.h.vmax,
  `${dnStatus}  [click-to-result ${dnMs} ms]  grid ${d10.h.w}×${d10.h.h} of ${d10.h.cell} µm, layers ${d10.ids.join(',')}`);
check('whole-slide grid sums to the transcript count', Math.abs(d10.sum * (d10.h.cell / 10) ** 2 - d10.h.total) / d10.h.total < 1e-4, `${Math.round(d10.sum * (d10.h.cell / 10) ** 2)} vs ${d10.h.total}`);
await setProp('point-opacity', {value: 0.25});
await page.waitForTimeout(800);
await shot('an_4_webgl_density_10um_full');
await setProp('density-bin', {value: 50});
await page.waitForFunction(() => window.XeniumViewer.instance._viewer._loaders.density.data?.header.bin === 50, null, {timeout: 30000});
await page.waitForTimeout(1000);
const d50 = await dens();
check('bin size 50 µm', d50.h.cell === 50 && d50.title.endsWith('per 50 µm bin') && Math.abs(d50.sum - d50.h.total) < 1, `grid ${d50.h.w}×${d50.h.h}, vmax ${d50.h.vmax}`);
await shot('an_5_webgl_density_50um_full');
// zoom in at 5 µm: the bins of the viewport, lined up with the cells
await setProp('density-bin', {value: 5});
await page.evaluate(([x, y]) => window.XeniumViewer.instance.setViewport({x0: x - 160, x1: x + 160, y0: y - 100, y1: y + 100}), at(0.72, 0.45));
await page.waitForFunction(() => { const h = window.XeniumViewer.instance._viewer._loaders.density.data?.header; return h && h.bin === 5 && h.cell === 5; }, null, {timeout: 30000});
await setProp('point-opacity', {value: 0.5});
await page.waitForTimeout(1500);
const d5 = await dens();
const vp = await page.evaluate(() => window.XeniumViewer.instance.getViewport());
check('zoomed in: 5 µm bins for the viewport only', d5.h.cell === 5 && d5.h.x0 <= vp.x0 && d5.h.x0 + d5.h.w * 5 >= vp.x1 && d5.h.w * 5 < 3 * (vp.x1 - vp.x0),
  `grid ${d5.h.w}×${d5.h.h} at (${d5.h.x0}, ${d5.h.y0})`);
await shot('an_6_webgl_density_5um_zoom');
// the same view with the gene's transcripts drawn as points: they must fall on the bins
await setProp('webgl-tx-genes', {value: [gene]});
await setProp('point-opacity', {value: 0.05});
await page.evaluate(([x, y]) => window.XeniumViewer.instance.setViewport({x0: x - 40, x1: x + 40, y0: y - 25, y1: y + 25}), at(0.72, 0.45));
await page.waitForFunction(() => window.XeniumViewer.instance._viewer._loaders.tx.data?.header.n > 0, null, {timeout: 30000});
await page.waitForTimeout(1500);
const align = await page.evaluate(() => {
  const v = window.XeniumViewer.instance._viewer, d = v._loaders.density.data, t = v._loaders.tx.data;
  const {x0, y0, w, h, cell} = d.header;
  const cnt = new Float64Array(w * h);
  let inside = 0;
  for (let i = 0; i < t.header.n; i++) {
    const c = Math.floor((t.arrays.xy[2 * i] - x0) / cell), r = Math.floor((t.arrays.xy[2 * i + 1] - y0) / cell);
    if (c >= 0 && c < w && r >= 0 && r < h) { cnt[r * w + c] += 1; inside += 1; }
  }
  // every bin that holds drawn transcript points has at least that many in the grid
  let bad = 0, used = 0;
  for (let k = 0; k < w * h; k++) if (cnt[k] > 0) { used += 1; if (d.arrays.values[k] < cnt[k]) bad += 1; }
  return {n: t.header.n, inside, used, bad, level: t.header.level};
});
check('transcript points of the gene fall into the bins that count them', align.inside > 20 && align.bad === 0, JSON.stringify(align));
await shot('an_7_webgl_density_5um_with_transcripts');
await setProp('webgl-tx-genes', {value: []});
await setProp('density-opacity', {value: 0.3});
await page.waitForTimeout(700);
check('opacity control', Math.abs((await dens()).opacity - 0.3) < 1e-6);
await setProp('density-max', {value: 3});
await page.waitForTimeout(700);
check('colour maximum override', (await dens()).max === '3');
await setProp('density-max', {value: null});
await setProp('density-opacity', {value: 0.7});
await setProp('point-opacity', {value: 0.4});
await page.evaluate(() => window.XeniumViewer.instance.fitView());
await page.waitForTimeout(1200);

// ── Plotly renderer ──────────────────────────────────────────────────────────
await setProp('density-bin', {value: 20});
await setProp('renderer-mode', {value: 'plotly'});
await page.waitForFunction(() => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  return gd && gd._fullData && gd._fullData.length >= 2 && gd._fullData[0].type === 'heatmap';
}, null, {timeout: 180000});
await page.waitForTimeout(2500);
const pl = await page.evaluate(() => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  const hm = gd._fullData[0], sc = gd._fullData[1];
  let sum = 0;
  for (const row of hm.z) for (const v of row) if (v > 0) sum += v;
  return {dx: hm.dx, zmax: hm.zmax, sum, title: hm.colorbar.title.text, scale: sc.marker.colorscale[0][1] + '…' + sc.marker.colorscale.at(-1)[1],
          cmin: sc.marker.cmin, cmax: sc.marker.cmax, name: sc.name, n: sc.x.length, cbx: [hm.colorbar.x, sc.marker.colorbar.x]};
});
check('Plotly: heat layer below the cells (20 µm bins, coarsened for the whole slide) and the score as colour',
  pl.dx >= 20 && Math.abs(pl.sum * (pl.dx / 20) ** 2 - d10.h.total) / d10.h.total < 1e-3 && /Gene set/.test(pl.name) && Math.abs(pl.cmin + pl.cmax) < 1e-6 && pl.cbx[0] > pl.cbx[1],
  JSON.stringify(pl));
await shot('an_8_plotly_density_geneset_full');
// zoom: the figure is rebuilt with the bins of the viewport
const c = at(0.72, 0.45);
await page.evaluate(([x, y]) => window.Plotly.relayout(document.querySelector('#spatial-plot .js-plotly-plot'),
  {'xaxis.range[0]': x - 160, 'xaxis.range[1]': x + 160, 'yaxis.range[0]': -y - 100, 'yaxis.range[1]': -y + 100}), c);
await page.waitForFunction(() => document.querySelector('#spatial-plot .js-plotly-plot')._fullData[0].dx === 20, null, {timeout: 120000});
await page.waitForTimeout(2500);
await shot('an_9_plotly_density_20um_zoom');
// Plotly lasso → current selection → group → DE against the rest
await setProp('density-enable', {value: []});
await openTab(page, 'tab-de');
await page.waitForFunction(() => document.querySelector('#spatial-plot .js-plotly-plot')._fullData[0].type !== 'heatmap', null, {timeout: 120000});
await page.evaluate(() => window.Plotly.relayout(document.querySelector('#spatial-plot .js-plotly-plot'), {dragmode: 'lasso'}));
await page.waitForTimeout(1500);
{
  const r = await page.evaluate(() => {
    const gd = document.querySelector('#spatial-plot .js-plotly-plot');
    const b = gd.getBoundingClientRect(), fl = gd._fullLayout;
    return {x: b.left + fl.xaxis._offset, y: b.top + fl.yaxis._offset, w: fl.xaxis._length, h: fl.yaxis._length};
  });
  const px = r.x + r.w / 2, py = r.y + r.h / 2, k = Math.min(r.w, r.h) * 0.3;
  await page.mouse.move(px - k, py - k * 0.7);
  await page.mouse.down();
  for (const [dx, dy] of [[k, -k], [k * 1.1, k * 0.5], [0.1 * k, k], [-k, k * 0.8], [-k, -k * 0.7]]) {
    await page.mouse.move(px + dx, py + dy, {steps: 8});
  }
  await page.mouse.up();
}
await closeRoiDialog();
await waitText('#de-selection-info', /Current selection: [\d,]+ cells/);
const nP = Number((await text('#de-selection-info')).replace(/[^\d]/g, ''));
await page.fill('#de-group-name', TAG + '_P');
await page.click('#de-save-group-btn');
await waitText('#de-status', new RegExp(`Saved the current selection as group '${TAG}_P'`));
await page.waitForTimeout(1500);
await pick('de-group-a', `grp:${TAG}_P`, `Group: ${TAG}_P`);
await setProp('de-group-b', {value: 'rest'});
await page.waitForTimeout(500);
t0 = Date.now();
await page.click('#de-run-btn');
await waitText('#de-status', new RegExp(`${TAG}_P \\(A, [\\d,]+ cells\\) vs all other cells`));
const dePMs = Date.now() - t0;
const stP = await text('#de-status');
await page.waitForTimeout(1500);
check('Plotly lasso → saved group → DE against all other cells', nP >= 10 && stP.includes(`${nP.toLocaleString('en-US')} cells`), `${stP}  [click-to-result ${dePMs} ms]`);
await shot('an_10_plotly_de');

// ── clean up what this run saved ─────────────────────────────────────────────
for (const g of ['_A', '_B', '_P']) {
  await page.waitForTimeout(1500);
  await setProp('de-group-a', {value: `grp:${TAG}${g}`});
  await page.waitForFunction((l) => document.querySelector('#de-group-a').innerText.includes(l), `Group: ${TAG}${g}`, {timeout: 15000});
  await page.click('#de-delete-group-btn');
  await waitText('#de-status', new RegExp(`Deleted group '${TAG}${g}'`));
}
await openTab(page, 'tab-geneset');
await setProp('geneset-select', {value: TAG});
await page.waitForTimeout(500);
await page.click('#geneset-delete-btn');
await waitText('#geneset-status', /Deleted gene set/);
await page.waitForTimeout(1500);
const after = await page.evaluate(async () => ({
  colorBy: document.querySelector('#color-by')?.innerText.trim(),
  groups: (await (await fetch('/viewer/meta')).json()).dataset,
}));
check('deleting the gene set takes Color By off it', !/Gene set/.test(after.colorBy), after.colorBy);
check('no page errors', logs.length === 0, logs.slice(0, 5).join(' | '));

await browser.close();
console.log(failures ? `FAILS: ${failures}` : 'ALL PASSED');
process.exit(failures ? 1 : 0);
