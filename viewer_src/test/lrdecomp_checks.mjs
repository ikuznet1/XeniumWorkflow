// Drives the ligand-receptor exposure decomposition (second mode of the Ligand–Receptor
// tab) in the real app. Nothing is drawn on the map, so either renderer will do.
//   PORT=8152 DATA_DIR=<sample 0> XENIUM_CACHE_DIR=<dir> python xenium_explorer.py
//   XL_PAIRS=/path/pairs.csv [XL_SAMPLES=/path/s1,/path/s2,...] [XL_FLOOR=100] \
//     node test/lrdecomp_checks.mjs http://127.0.0.1:8152 /path/to/screenshot_dir
// XL_PAIRS: a CSV of ligand,receptor lines (no database is bundled). XL_SAMPLES: further
// Xenium bundles added through "+ Add sample"; with at least 5 (6 samples) the design
// part runs: even samples to condition A, odd ones to B, first unpaired, then paired by
// donor. XL_FLOOR: the transcript floor to type in (default: the app's default).
import {chromium} from 'playwright';
import {openTab} from './lib.mjs';
import fs from 'node:fs';

const base = process.argv[2] || 'http://127.0.0.1:8050';
const out = process.argv[3] || '.';
const extra = (process.env.XL_SAMPLES || '').split(',').filter(Boolean);
const pairsFile = process.env.XL_PAIRS;
if (!pairsFile) {
  console.log('XL_PAIRS must name a CSV of ligand,receptor pairs');
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
await page.waitForSelector('#plots-tabs', {state: 'attached', timeout: 180000});
await page.waitForFunction(() => window.dash_clientside && window.dash_clientside.set_props, null, {timeout: 120000});
await page.waitForTimeout(4000);

const setProp = (id, props) => page.evaluate(([i, p]) => window.dash_clientside.set_props(i, p), [id, props]);
const text = (sel) => page.evaluate((s) => document.querySelector(s)?.innerText || '', sel);
const waitText = async (sel, re, timeout = 900000) => {
  try {
    await page.waitForFunction(([s, r]) => new RegExp(r).test(document.querySelector(s)?.innerText || ''), [sel, re.source], {timeout});
  } catch (err) {
    throw new Error(`${sel} never matched ${re}: it shows "${await text(sel)}"`);
  }
};
const shot = async (name) => {
  await page.screenshot({path: `${out}/${name}.png`});
  const el = await page.$('#lrd-plot');
  if (el && await el.isVisible()) {
    await el.scrollIntoViewIfNeeded();
    await page.waitForTimeout(300);
    await el.screenshot({path: `${out}/${name}_plot.png`});
  }
};
const click = (sel) => page.evaluate((s) => document.querySelector(s).click(), sel);
const tableRows = (id) => page.evaluate((i) => document.querySelectorAll(`#${i} tbody tr`).length - 2, id);
const headers = (id) => page.evaluate((i) => [...document.querySelectorAll(`#${i} th`)].map((e) => e.innerText.trim()).filter(Boolean), id);
const plot = () => page.evaluate(() => {
  const gd = document.querySelector('#lrd-plot .js-plotly-plot');
  if (!gd || !gd.data) return null;
  const fd = gd._fullData;                       // decoded arrays (gd.data may hold base64 typed arrays)
  return {names: fd.map((t) => t.name), types: fd.map((t) => t.type), n: (fd[0].y || []).length,
          barmode: gd.layout.barmode, title: gd.layout.title?.text || '',
          sums: Array.from(fd[0].y || []).map((_, k) => fd.filter((t) => t.type === 'bar').reduce((a, t) => a + t.x[k], 0)),
          total: fd.length > 5 ? Array.from(fd[fd.length - 1].x) : null};
});

if (await page.evaluate(() => getComputedStyle(document.querySelector('#umap-panel')).display === 'none')) {
  await click('#umap-toggle');
}

// ── add the samples (optional) ───────────────────────────────────────────────
for (let i = 0; i < extra.length; i++) {
  await click('#sample-modal-open-btn');
  await page.waitForSelector('#sample-custom-path', {state: 'visible', timeout: 30000});
  await page.fill('#sample-custom-path', extra[i]);
  await page.click('#sample-add-btn');
  await waitText('#extra-samples-status', new RegExp(`${i + 2} samples loaded`), 900000);
  await page.waitForTimeout(800);
}
const nS = extra.length + 1;

// ── the mode switch ──────────────────────────────────────────────────────────
await openTab(page, 'tab-lr');
await page.waitForSelector('#lr-run-btn', {state: 'visible'});
check('scores mode is the default: its controls and results are shown, the decomposition is hidden',
  await page.isVisible('#lr-k') && await page.isVisible('#lr-plot') && !(await page.isVisible('#lrd-run-btn')) && !(await page.isVisible('#lrd-plot')));
await setProp('lr-mode', {value: 'decomp'});
await page.waitForSelector('#lrd-run-btn', {state: 'visible'});
check('decomposition mode swaps the controls and the results; pair list and labels stay',
  !(await page.isVisible('#lr-k')) && !(await page.isVisible('#lr-plot')) && await page.isVisible('#lrd-plot') && await page.isVisible('#lr-pairs') && await page.isVisible('#lr-labels'));
check('the note says what it explains and what it is not',
  /WHY an exposure differs/.test(await text('#lrd-note')) && /not evidence of signalling/.test(await text('#lrd-note')) && /1 − \(1 − p_L\)\^k/.test(await text('#lrd-note')));
check('nothing ran on load', (await text('#lrd-status')) === '' && /No exposure decomposition computed yet/.test(await text('#lrd-plot')));
await shot('lrd_0_empty');

await page.click('#lrd-run-btn');
await waitText('#lrd-status', /No ligand–receptor pairs/);
check('without a pair list nothing runs', /no database is bundled or downloaded/.test(await text('#lrd-status')));

await page.setInputFiles('#lr-upload input[type=file]', {name: 'pairs.csv', mimeType: 'text/csv', buffer: fs.readFileSync(pairsFile)});
await waitText('#lr-pairs-info', /pairs from pairs\.csv/);
if (process.env.XL_FLOOR) {
  await page.fill('#lrd-floor', process.env.XL_FLOOR);
  await page.keyboard.press('Tab');
  await page.waitForTimeout(500);
}
const labelKind = (await text('#lr-labels')).trim();

const run = async () => {
  const t0 = Date.now();
  await page.click('#lrd-run-btn');
  await page.waitForFunction(() => /Running|Collecting|Summing|Starting/.test(document.querySelector('#lrd-status')?.innerText || ''), null, {timeout: 30000}).catch(() => {});
  await waitText('#lrd-status', /Exposure = K × R × L × γ × π exactly|None of the|no .* labels|No cell type/);
  await page.waitForTimeout(1500);
  return (Date.now() - t0) / 1000;
};

const design = async (assign, donors) => {
  await click('#lrd-design-open-btn');
  await page.waitForFunction((n) => document.querySelectorAll('#design-rows > div').length === n + 1, nS, {timeout: 30000});
  for (let i = 0; i < nS; i++) {
    await setProp({type: 'design-cond', index: i}, {value: assign[i]});
    await setProp({type: 'design-batch', index: i}, {value: ''});
    await setProp({type: 'design-donor', index: i}, {value: donors ? donors[i] : ''});
  }
  await page.fill('#design-name-a', 'even');
  await page.fill('#design-name-b', 'odd');
  await page.keyboard.press('Tab');
  await page.waitForTimeout(400);
  await page.click('#design-apply-btn');
  await waitText('#design-status', /Saved:|Give the two|changed while/);
  const msg = await text('#design-status');
  await page.click('#design-close-btn');
  await page.waitForFunction(() => !document.querySelector('.modal.show #design-rows'), null, {timeout: 30000});
  await page.waitForTimeout(500);
  return msg;
};
if (nS > 1) await design(Array.from({length: nS}, () => ''), null);   // a design kept from an earlier session

// ── without a design: descriptive ────────────────────────────────────────────
let secs = await run();
let st = await text('#lrd-status');
console.log(`INFO  in-app run (${labelKind}; click → result shown): ${secs.toFixed(1)} s`);
console.log('INFO  ' + st.replace(/\n/g, '\nINFO  '));
let p = await plot();
let hd = await headers('lrd-table');
check('one condition: the decomposition is shown descriptively, with the reason for no p-values',
  /No p-values: /.test(st) && /not evidence of signalling/.test(st) && !hd.some((h) => /^p /.test(h)) && hd.includes('γ') && hd.includes('π') && hd.includes('O/E'),
  hd.join(','));
check('descriptive plot: stacked bars of the two geometry factors, one row per interaction',
  !!p && p.names.join('|') === 'Tile co-location (γ)|Contact preference (π)' && p.barmode === 'relative' && p.n > 0 && /descriptive, no p-values/.test(p.title),
  p ? `${p.n} rows; ${p.title}` : 'no plot');
check('the table has the interactions', (await tableRows('lrd-table')) > 0, `${await tableRows('lrd-table')} rows on the page`);
check('the job went through the jobs registry and the Run button is enabled again', !(await page.isDisabled('#lrd-run-btn')));
await shot('lrd_1_descriptive');
const dl = page.waitForEvent('download', {timeout: 30000});
await page.click('#lrd-download-btn');
const file = await dl;
const csv = fs.readFileSync(await file.path(), 'utf8');
check('CSV: header with the definition, the mode and the caveat, then the rows',
  /exposure = contact \(K/.test(csv) && /# mode: descriptive/.test(csv) && /not evidence of signalling/.test(csv) && /\nsender,receiver,ligand,receptor,/.test(csv),
  `${file.suggestedFilename()}, ${csv.split('\n').length} lines`);

// ── with a design ────────────────────────────────────────────────────────────
if (nS >= 6) {
  const alternate = Array.from({length: nS}, (_, i) => (i % 2 ? 'B' : 'A'));
  let saved = await design(alternate, null);
  check('the sample design opens from the decomposition controls and its summary is shown',
    /Saved:/.test(saved) && /even/.test(await text('#lrd-design-summary')), (await text('#lrd-design-summary')).slice(0, 160));
  secs = await run();
  st = await text('#lrd-status');
  console.log(`INFO  in-app run, unpaired (${secs.toFixed(1)} s):\nINFO  ` + st.replace(/\n/g, '\nINFO  '));
  p = await plot();
  hd = await headers('lrd-table');
  check('two conditions, unpaired: exact label permutation per term, unit and n stated, omnibus reported',
    /exact unpaired label permutation per term; unit of the p-values: n = \d+ samples \(not cells\); smallest achievable p/.test(st)
    && (/Omnibus over the between-type pairs/.test(st) || /No interaction passed/.test(st)), st.split('\n')[2] || '');
  if (p && p.total) {
    const worst = Math.max(...p.sums.map((s, k) => Math.abs(s - p.total[k])));
    check('plot: five stacked term bars per interaction that sum to the exposure change (diamond)',
      p.names.length === 6 && p.names[5] === 'Exposure (sum)' && p.barmode === 'relative' && worst < 1e-6 && /log2 A \(even\) \/ B \(odd\)/.test(p.title),
      `${p.n} rows; max |Σ bars − ◆| = ${worst.toExponential(1)}; ${p.title}`);
    check('table: Δ per term and p-values', hd.includes('Δ exposure') && hd.includes('Δ K') && hd.includes('p (exposure)') && hd.includes('Δ O/E'), hd.join(','));
    await shot('lrd_2_unpaired');
    await setProp('lrd-level', {value: 'pairs'});
    await page.waitForTimeout(1500);
    const pp = await plot();
    const hp = await headers('lrd-table');
    check('level "sender → receiver pairs": one row per type pair', !!pp && /type pairs/.test(pp.title) && hp.includes('Interactions') && !hp.includes('Ligand'), pp ? pp.title : '');
    await shot('lrd_3_unpaired_type_pairs');
    await setProp('lrd-level', {value: 'interactions'});
  } else {
    check('two conditions: a clear message when no interaction passes the filters', /No interaction passed/.test(st), st.slice(0, 200));
  }
  saved = await design(alternate, Array.from({length: nS}, (_, i) => `d${Math.floor(i / 2)}`));
  secs = await run();
  st = await text('#lrd-status');
  console.log(`INFO  in-app run, paired (${secs.toFixed(1)} s):\nINFO  ` + st.replace(/\n/g, '\nINFO  '));
  check('donors in both conditions: exact paired sign-flip, donors are the units',
    /exact paired sign-flip per term; unit of the p-values: n = \d+ samples \(\d+ donors, paired\) \(not cells\)/.test(st), st.split('\n')[2] || '');
  await shot('lrd_4_paired');
  // fewer than 3 units per condition: ratios, no p-values
  saved = await design(alternate.map((c, i) => (i < 4 ? c : '')), null);
  secs = await run();
  st = await text('#lrd-status');
  hd = await headers('lrd-table');
  check('2 v 2 samples: log2 ratios of the pooled conditions without p-values, and it says why',
    /No p-values: fewer than 3 independent units per condition/.test(st) && !hd.some((h) => /^p /.test(h)), st.split('\n')[2] || '');
  await shot('lrd_5_too_few_units');
  await design(Array.from({length: nS}, () => ''), null);
} else {
  console.log('SKIP  design checks  (XL_SAMPLES names fewer than 5 further bundles)');
}

// back to the scores mode: its results come back untouched
await setProp('lr-mode', {value: 'scores'});
await page.waitForSelector('#lr-run-btn', {state: 'visible'});
check('switching back shows the score controls again', await page.isVisible('#lr-plot') && !(await page.isVisible('#lrd-plot')));
check('no page errors', logs.length === 0, logs.slice(0, 5).join(' | '));

await browser.close();
console.log(failures ? `FAILS: ${failures}` : 'ALL PASSED');
process.exit(failures ? 1 : 0);
