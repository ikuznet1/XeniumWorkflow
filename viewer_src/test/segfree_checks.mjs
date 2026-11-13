// Drives the segmentation-free abundance block of the Cell Type Abundance tab in the real
// app: derived markers of a cell type, the tile-size sweep, the table, the plot, the CSV
// link, the area from all transcripts, user-listed markers and the empty states.
//   PORT=8154 DATA_DIR=<bundle> XENIUM_CACHE_DIR=<dir> python xenium_explorer.py
//   [SF_LABELS=celltype:celltypist] [SF_TYPE="Mast cells"] [SF_GENES="CPA3, KIT"] [SF_ALLTX=1]
//   node test/segfree_checks.mjs http://127.0.0.1:8154 /path/to/screenshot_dir
// Needs a bundle with transcripts.parquet (or transcripts.zarr.zip). With a cell-type
// annotation in the cache the markers are derived from it; otherwise from the clusters.
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
const plotTraces = (id) => page.evaluate((i) => {
  const gd = document.querySelector(`#${i} .js-plotly-plot`);
  return gd && gd._fullData ? gd._fullData.map((t) => ({name: t.name || '', type: t.type, x: Array.from(t.x || []), y: Array.from(t.y || [])})) : [];
}, id);
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
const firstRow = (id) => page.evaluate((i) => {
  const rows = [...document.querySelectorAll(`#${i} tbody tr`)].slice(2);
  return rows.map((r) => [...r.querySelectorAll('td')].map((c) => c.innerText.trim()));
}, id);
const run = async (re = /Markers \(|Pick a cell type|not a tile size|None of the listed|needs the Xenium output/) => {
  const t0 = Date.now();
  await click('#sfree-run-btn');
  await page.waitForTimeout(300);
  await waitText('#sfree-status', re);
  await page.waitForFunction(() => !document.querySelector('#sfree-run-btn').disabled, null, {timeout: 600000});
  await page.waitForTimeout(1500);      // the results callback redraws the plot and the table
  return (Date.now() - t0) / 1000;
};

if (await page.evaluate(() => getComputedStyle(document.querySelector('#umap-panel')).display === 'none')) {
  await click('#umap-toggle');
}
await setProp('plots-tabs', {active_tab: 'tab-abundance'});
await page.waitForTimeout(2500);
await page.evaluate(() => document.querySelector('#sfree-section').scrollIntoView());
check('the block sits in the Cell Type Abundance tab, below the composition block',
  await page.evaluate(() => {
    const s = document.querySelector('#sfree-section'), c = document.querySelector('#comp-section');
    return !!s && !!c && s.getBoundingClientRect().top > c.getBoundingClientRect().top
      && getComputedStyle(s).display !== 'none';
  }));
const fresh = /No segmentation-free abundance computed yet/.test(await text('#sfree-plot'));
check('empty state: placeholder figure and empty table (or the result of an earlier run of this server)',
  fresh ? (await tableRows('sfree-table')) === 0 : (await tableRows('sfree-table')) > 0, fresh ? 'fresh server' : 'server already holds a result');
await shot('sfree_0_empty');

// 1. nothing picked -> a clear message, no job
await setProp('sfree-type', {value: null});
await setProp('sfree-markers', {value: ''});
await run(/Pick a cell type/);
check('no type and no markers: asks for one of them', /Pick a cell type .* or list marker genes/.test(await text('#sfree-status')));

// 2. derived markers of a cell type, default sweep
const kinds = await dropdownOptions('sfree-labels');
const kind = process.env.SF_LABELS || kinds.find((k) => k.startsWith('celltype:')) || 'cluster';
check('label sources are offered', kinds.includes(kind), kinds.join(', '));
await setProp('sfree-labels', {value: kind});
await page.waitForTimeout(1500);
const types = await dropdownOptions('sfree-type');
const type = process.env.SF_TYPE && types.includes(process.env.SF_TYPE) ? process.env.SF_TYPE
  : (types.find((t) => /mast/i.test(t)) || types[Math.min(4, types.length - 1)]);
check('cell types of the label source are offered', types.length >= 2 && !!type, `${types.length} types; using "${type}"`);
await setProp('sfree-type', {value: type});
const secs = await run(/Markers \(derived/);
const status = await text('#sfree-status');
console.log(status.split('\n').map((l) => '    | ' + l).join('\n'));
check('status names the markers, the area rule, the sweep and the unit of the concentration p',
  /Markers \(derived from/.test(status) && /Tissue area: tiles with a cell/.test(status)
  && /tile sizes 25, 50, 100, 200 µm/.test(status) && /unit: the tiles of one sample \(n = /.test(status),
  `${secs.toFixed(1)} s in the browser`);
check('single sample is said to be descriptive', /Single sample: the densities are descriptive/.test(status));
const rows = await firstRow('sfree-table');
const hd = await headers('sfree-table');
check('table: one row per tile size with the density columns', rows.length === 4
  && ['Tile µm', 'Area mm²', 'Tx / mm²', 'Cells / mm²', 'Gini − null', 'p (tiles)'].every((h) => hd.includes(h)),
  hd.join(' | '));
const ci = (h) => hd.indexOf(h);
const area = rows.map((r) => parseFloat(r[ci('Area mm²')]));
const txd = rows.map((r) => parseFloat(r[ci('Tx / mm²')]));
check('area grows and transcript density falls along the sweep', area.every((a, i) => i === 0 || a >= area[i - 1])
  && txd[0] >= txd[txd.length - 1], `area ${area.join(' → ')} mm²; tx/mm² ${txd.join(' → ')}`);
const tr = await plotTraces('sfree-plot');
check('plot: three panels (transcripts, segmented cells, Gini − null) over the four tile sizes',
  tr.length === 3 && tr.every((t) => t.x.length === 4) && Math.abs(tr[0].y[0] - txd[0]) <= Math.abs(txd[0]) * 0.01 + 1);
await page.evaluate(() => document.querySelector('#sfree-section').scrollIntoView());
await page.waitForTimeout(800);
await shot('sfree_1_derived');
await page.evaluate(() => document.querySelector('#sfree-table').scrollIntoView({block: 'center'}));
await page.waitForTimeout(500);
await shot('sfree_1_table');

// 3. CSV
const [dl] = await Promise.all([page.waitForEvent('download', {timeout: 30000}), click('#sfree-download-btn')]);
const csvPath = `${out}/segmentation_free_abundance.csv`;
await dl.saveAs(csvPath);
const fs = await import('node:fs');
const csv = fs.readFileSync(csvPath, 'utf8');
check('CSV: header comments, sweep rows and the per-gene table',
  /^# Segmentation-free abundance; cell type: /.test(csv) && /# markers \(/.test(csv)
  && /\nsample,name,bin_um,/.test(csv) && /# per marker gene/.test(csv), `${csv.length} bytes`);

// 4. user-listed markers and a custom sweep
const genes = process.env.SF_GENES || status.match(/: ([A-Za-z0-9_.-]+), ([A-Za-z0-9_.-]+)/)?.slice(1, 3).join(', ') || '';
await setProp('sfree-markers', {value: genes + ', NOT_A_GENE'});
await setProp('sfree-bins', {value: '50, 150'});
await setProp('sfree-nnull', {value: 50});
await run(/Markers \(listed\)/);
const st2 = await text('#sfree-status');
check('listed markers are used, names not on the panel are reported', /Markers \(listed\): /.test(st2)
  && /Not on the panel: NOT_A_GENE/.test(st2) && /tile sizes 50, 150 µm/.test(st2) && (await tableRows('sfree-table')) === 2);
await setProp('sfree-bins', {value: 'wide'});
await run(/not a tile size/);
check('a bad tile size is refused before a job starts', /'wide' is not a tile size/.test(await text('#sfree-status')));
await setProp('sfree-bins', {value: '25, 50, 100, 200'});
await setProp('sfree-markers', {value: 'NOT_A_GENE'});
await run(/None of the listed/);
check('only unknown genes: clear error', /None of the listed marker genes is on the panel/.test(await text('#sfree-status')));
await setProp('sfree-markers', {value: ''});

// 5. area from all transcripts (scans every transcript once; cached afterwards)
if (process.env.SF_ALLTX !== '0') {
  await setProp('sfree-area', {value: 'transcripts'});
  const s3 = await run(/Markers \(derived/);
  const st3 = await text('#sfree-status');
  const r3 = await firstRow('sfree-table');
  const h3 = await headers('sfree-table');
  const a3 = r3.map((r) => parseFloat(r[h3.indexOf('Area mm²')]));
  const p10 = r3.map((r) => parseFloat(r[h3.indexOf('Tx / 10k tx')]));
  check('area from all transcripts: said so, area ≥ the cells-based one, share of all transcripts filled in',
    /tiles with ≥ 1 gene transcript \(segmentation-free/.test(st3) && a3[0] >= area[0] && p10.every((v) => v > 0),
    `${s3.toFixed(1)} s; area ${a3.join(' → ')} mm²; tx / 10k ${p10[0]}`);
  await page.evaluate(() => document.querySelector('#sfree-section').scrollIntoView());
  await page.waitForTimeout(800);
  await shot('sfree_2_alltx');
  await setProp('sfree-area', {value: 'cells'});
}

// 6. the jobs panel showed the job; nothing broke
check('no page or console errors', logs.length === 0, logs.slice(0, 3).join(' || '));
console.log(failures ? `\n${failures} FAILED` : '\nall segmentation-free checks passed');
await browser.close();
process.exit(failures ? 1 : 0);
