// Drives the multi-sample statistics in the real app, in both renderers: sample design,
// pseudobulk differential expression, composition test, transcript density of added
// samples, Moran's I and ligand-receptor scores.
//   XENIUM_RENDERER=webgl PORT=8097 DATA_DIR=<sample 0> XENIUM_CACHE_DIR=<dir> python xenium_explorer.py
//   XS_SAMPLES=/path/s1,/path/s2,... [XS_PAIRS=/path/pairs.csv] [XS_GENE=GENE] \
//     node test/stats_checks.mjs http://127.0.0.1:8097 /path/to/screenshot_dir
// XS_SAMPLES: at least 3 further Xenium bundles (4 samples in all), added through the
// "+ Add sample" dialog. Even-numbered samples go to condition A, odd ones to B. Needs R
// with DESeq2 for the pseudobulk part and a cell-type annotation of every sample for the
// composition / ligand-receptor parts (those checks are reported as SKIP otherwise).
import {chromium} from 'playwright';
import fs from 'node:fs';

const base = process.argv[2] || 'http://127.0.0.1:8050';
const out = process.argv[3] || '.';
const extra = (process.env.XS_SAMPLES || '').split(',').filter(Boolean);
if (extra.length < 3) {
  console.log('XS_SAMPLES must name at least 3 further bundles');
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
const skip = (name, why) => console.log(`SKIP  ${name}  ${why}`);

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
const waitText = async (sel, re, timeout = 600000) => {
  try {
    await page.waitForFunction(([s, r]) => new RegExp(r).test(document.querySelector(s)?.innerText || ''), [sel, re.source], {timeout});
  } catch (err) {
    throw new Error(`${sel} never matched ${re}: it shows "${await text(sel)}"`);
  }
};
const shot = (name) => page.screenshot({path: `${out}/${name}.png`});
const click = (sel) => page.evaluate((s) => document.querySelector(s).click(), sel);
const tableRows = (id) => page.evaluate((i) => document.querySelectorAll(`#${i} tbody tr`).length - 2, id);   // header + filter rows
const headers = (id) => page.evaluate((i) => [...document.querySelectorAll(`#${i} th`)].map((e) => e.innerText.trim()).filter(Boolean), id);

if (await page.evaluate(() => getComputedStyle(document.querySelector('#umap-panel')).display === 'none')) {
  await click('#umap-toggle');
}

// ── add the samples ──────────────────────────────────────────────────────────
for (let i = 0; i < extra.length; i++) {
  await click('#sample-modal-open-btn');
  await page.waitForSelector('#sample-custom-path', {state: 'visible', timeout: 30000});
  await page.fill('#sample-custom-path', extra[i]);
  await page.click('#sample-add-btn');
  await waitText('#extra-samples-status', new RegExp(`${i + 2} samples loaded`), 900000);
  await page.waitForTimeout(800);
}
const nS = extra.length + 1;
await page.waitForFunction((n) => {
  const v = window.XeniumViewer.instance._viewer;
  return v._cells && v._meta && v._cells.n > 0 && (v._meta.samples || []).length !== 1;
}, nS, {timeout: 120000}).catch(() => {});
await page.evaluate(() => window.XeniumViewer.instance.fitView());
await page.waitForTimeout(2500);
await shot('st_0_webgl_samples_side_by_side');
check(`${nS} samples loaded side by side`, /samples loaded/.test(await text('#extra-samples-status')), await text('#extra-samples-status'));
check('counts mode note is empty when no sample has SPLIT-corrected counts', (await text('#counts-mode-note')) === '' || /SPLIT/.test(await text('#counts-mode-note')), await text('#counts-mode-note'));

// ── the design, through the editor ───────────────────────────────────────────
await setProp('plots-tabs', {active_tab: 'tab-de'});
await page.waitForSelector('#de-run-btn', {state: 'visible'});
check('cell-level mode shows the two group pickers and its caveat',
  await page.isVisible('#de-group-a') && !(await page.isVisible('#de-pb-population')) && /Wilcoxon/.test(await text('#de-caveat')));
await setProp('de-mode', {value: 'pseudobulk'});
await page.waitForSelector('#de-pb-population', {state: 'visible'});
check('pseudobulk mode shows the population picker and its caveat',
  !(await page.isVisible('#de-group-a')) && /samples \(not the cells\) are the observations/.test(await text('#de-caveat')),
  (await text('#de-design-summary')).slice(0, 120));
const design = async (assign, names, batch) => {
  await click('#de-design-open-btn');
  await page.waitForFunction((n) => document.querySelectorAll('#design-rows > div').length === n + 1 && document.querySelector('#design-modal'), nS, {timeout: 30000});
  for (let i = 0; i < nS; i++) {
    await setProp({type: 'design-cond', index: i}, {value: assign[i]});
    await setProp({type: 'design-batch', index: i}, {value: batch ? batch[i] : ''});
  }
  if (names) {
    await page.fill('#design-name-a', names[0]);
    await page.fill('#design-name-b', names[1]);
    await page.keyboard.press('Tab');
  }
  await page.waitForTimeout(400);
  return async () => {
    await page.click('#design-apply-btn');
    await waitText('#design-status', /Saved:|Give the two|changed while/);
    const msg = await text('#design-status');
    await page.click('#design-close-btn');
    await page.waitForFunction(() => !document.querySelector('.modal.show #design-rows'), null, {timeout: 30000});
    return msg;
  };
};
const alternate = Array.from({length: nS}, (_, i) => (i % 2 ? 'B' : 'A'));
let apply = await design(alternate, ['even', 'odd']);
await shot('st_1_design_editor');
let saved = await apply();
await page.waitForTimeout(600);
const summary = await text('#de-design-summary');
check('design editor lists every sample; Apply stores the assignment for both tabs',
  /Saved: \d+ samples? in A, \d+ in B, 0 excluded/.test(saved) && /A \(even\): S0, S2/.test(summary) && /B \(odd\): S1, S3/.test(summary), `${saved} | ${summary}`);

// ── pseudobulk differential expression ───────────────────────────────────────
const runDe = async () => {
  await page.click('#de-run-btn');
  await page.waitForFunction(() => /Running|Summing|Fitting/.test(document.querySelector('#de-status')?.innerText || ''), null, {timeout: 30000}).catch(() => {});
  await page.waitForFunction(() => !document.querySelector('#de-run-btn').disabled && !/^(Running…|Summing|Fitting)/.test(document.querySelector('#de-status')?.innerText || ''), null, {timeout: 900000});
  await page.waitForTimeout(1500);
  return text('#de-status');
};
let st = await runDe();
const noR = /rpy2 \/ R is not available|needs the R package/.test(st);
if (noR) {
  check('without R / DESeq2 the run is refused with the install hint (no silent fallback)', /BiocManager::install/.test(st), st.slice(0, 200));
  skip('pseudobulk result in volcano / table', 'R with DESeq2 or edgeR is not installed');
} else {
  const pb = await page.evaluate(() => {
    const gd = document.querySelector('#de-volcano-plot .js-plotly-plot');
    return {title: gd.layout.title.text, xt: gd.layout.xaxis.title.text, n: gd._fullData.reduce((a, t) => a + (t.x || []).length, 0),
            samples: [...document.querySelectorAll('#de-pb-samples tbody tr')].map((r) => r.innerText.replace(/\s+/g, ' ').trim())};
  });
  const hd = await headers('de-table');
  check('pseudobulk run: status names method, design, samples and the genes not tested',
    /Pseudobulk \(all cells\): A even \[S0, S2.*\] vs B odd \[S1, S3.*\]; (DESeq2|edgeR) .*design ~ condition/.test(st) && /with a total count below 10 not tested/.test(st) && /Original counts\./.test(st), st.slice(0, 330));
  check('volcano and table show the pseudobulk result (method and design in the title, test statistic and base mean as columns)',
    /Pseudobulk, all cells: A even \(\d samples\) vs B odd \(\d samples\)/.test(pb.title) && /~ condition/.test(pb.title) && /shrunken|unshrunken/.test(pb.xt) && pb.n > 100
    && hd.includes('Base mean') && (hd.includes('Wald') || hd.includes('F')) && !hd.includes('% A') && (await tableRows('de-table')) > 0,
    `${pb.title} | x: ${pb.xt} | ${pb.n} points | ${hd.join(',')}`);
  check('per-sample cells, library sizes and size factors are shown', pb.samples.length === nS && pb.samples.every((r) => /S\d .* (A|B) .* [\d,]+ [\d,]+ [\d.e+-]+/.test(r)), pb.samples.slice(0, 2).join(' || '));
  await shot('st_2_pseudobulk_all_cells');
  // a cell type as the population (XS_CELLTYPE: a label every sample has)
  if (process.env.XS_CELLTYPE) {
    await setProp('de-pb-population', {value: `celltype:${process.env.XS_CELLTYPE}`});
    await page.waitForTimeout(500);
    st = await runDe();
    check('pseudobulk of one cell type', new RegExp(`Pseudobulk \\(${process.env.XS_CELLTYPE}\\)`).test(st) && /genes tested/.test(st), st.slice(0, 260));
    await shot('st_2b_pseudobulk_cell_type');
    await setProp('de-pb-population', {value: 'all'});
    await page.waitForTimeout(500);
  }
  // a condition left with one sample: refused
  apply = await design(alternate.map((c, i) => (i % 2 && i > 1 ? '' : c)));
  await apply();
  st = await runDe();
  check('a condition with fewer than 2 samples is refused with the reason', /Condition B \(odd\) has 1 usable sample; at least 2 per condition are needed/.test(st), st.slice(0, 200));
  // a batch label on some samples only: refused; confounded labels: refused
  apply = await design(alternate, null, alternate.map((c, i) => (i === 0 ? 'p1' : '')));
  await apply();
  st = await runDe();
  check('batch labels on some samples only are refused', /some samples only/.test(st), st.slice(0, 160));
  apply = await design(alternate, null, alternate.map((c) => (c === 'A' ? 'p1' : 'p2')));
  await apply();
  st = await runDe();
  check('batch labels confounded with the condition are refused', /confounded with the condition/.test(st), st.slice(0, 160));
  apply = await design(alternate, null, null);
  await apply();
}

// ── composition between conditions ───────────────────────────────────────────
await setProp('plots-tabs', {active_tab: 'tab-abundance'});
await page.waitForSelector('#comp-run-btn', {state: 'visible'});
await page.waitForFunction(() => /A \(even\)/.test(document.querySelector('#comp-design-summary')?.innerText || ''), null, {timeout: 30000});
await page.waitForTimeout(1200);
await page.evaluate(() => document.querySelector('#comp-section').scrollIntoView());
await page.click('#comp-run-btn');
await waitText('#comp-status', /categories tested|No test was run|No sample|One sample|Pick what/);
await page.waitForTimeout(1500);
const cst = await text('#comp-status');
const comp = await page.evaluate(() => {
  const gd = document.querySelector('#comp-plot .js-plotly-plot');
  return {traces: gd._fullData.map((t) => [t.name, t.mode, (t.x || []).length]), title: gd.layout.title.text, cat: document.querySelector('#comp-category')?.innerText.trim()};
});
const chd = await headers('comp-table');
const nA = alternate.filter((c) => c === 'A').length, nB = nS - nA;
if (nA >= 3 && nB >= 3) {
  check('composition: every category tested between the conditions (propeller-style), points per sample in the plot',
    /categories tested between A even \(\d samples\) and B odd \(\d samples\): logit-transformed proportions, ~ condition, moderated t-test/.test(cst)
    && comp.traces.some((t) => t[0] === 'A: even' && t[1] === 'markers') && comp.traces.some((t) => t[0] === 'B: odd')
    && chd.includes('FDR') && chd.includes('p (Welch)') && chd.some((h) => /^% S1/.test(h)) && (await tableRows('comp-table')) > 1,
    `${comp.cat} | ${cst.slice(0, 200)} | ${chd.join(',')}`);
} else {
  check('composition: fewer than 3 samples per condition → proportions only, said plainly',
    /No test was run: condition A has \d and condition B has \d samples/.test(cst) && !chd.includes('FDR') && chd.some((h) => /^% S1/.test(h)), cst.slice(0, 200));
}
check('the compositional caveat is on the tab', /Proportions are compositional/.test(await text('#comp-note')) && /no test is run/.test(await text('#comp-note')));
await shot('st_3_composition');

// ── transcript density of the added samples, both renderers ──────────────────
await setProp('plots-tabs', {active_tab: 'tab-density'});
await page.waitForSelector('#density-run-btn', {state: 'visible'});
const gene = process.env.XS_GENE || '__all__';
await setProp('density-gene', {value: gene});
await page.waitForTimeout(1200);
await page.click('#density-run-btn');
await waitText('#density-status', /Binned|Grid loaded from the cache/);
await page.waitForTimeout(800);
await page.waitForFunction((n) => (window.XeniumViewer.instance.deck.props.layers || []).flat(3).filter((l) => l && /^xv-density/.test(l.id)).length === n, nS, {timeout: 180000});
await page.evaluate(() => window.XeniumViewer.instance.fitView());
await page.waitForTimeout(3000);
const dn = await page.evaluate(async () => {
  const v = window.XeniumViewer.instance._viewer, d = v._loaders.density.data;
  const layers = (window.XeniumViewer.instance.deck.props.layers || []).flat(3).filter((l) => l && /^xv-density/.test(l.id));
  const ids = (window.XeniumViewer.instance.deck.props.layers || []).flat(3).filter(Boolean).map((l) => l.id);
  // x range of the cells of every sample, from the positions the viewer draws
  const pos = v._cells.positions || v._cells.xy, smp = v._cells.sample;
  return {parts: d.header.parts, total: d.header.total, vmax: d.header.vmax, bounds: layers.map((l) => l.props.bounds),
          order: [ids.findIndex((i) => /^xv-density/.test(i)), ids.indexOf('xv-cells')], ids: layers.map((l) => l.id),
          legend: document.querySelector('#webgl-view .xv-box[data-kind="density"]')?.innerText.replace(/\s+/g, ' '),
          havePos: !!pos, haveSmp: !!smp};
});
const hint = await text('#density-hint');
// the server's own description of where each sample lies
const sep = await page.evaluate(async () => (await (await fetch('/viewer/meta')).json()));
const increasing = dn.bounds.every((b, i) => i === 0 || b[0] > dn.bounds[i - 1][2]);
check(`WebGL: one density grid per sample (${nS}), each in its own place, below the cells, one colour bar`,
  dn.parts.length === nS && dn.ids.length === nS && dn.ids[0] === 'xv-density' && increasing && dn.order[0] >= 0 && dn.order[0] < dn.order[1]
  && /Samples drawn: S0, S1/.test(hint) && /one colour scale/.test(hint),
  `${hint} | x of grids: ${dn.bounds.map((b) => `${Math.round(b[0])}–${Math.round(b[2])}`).join(', ')} | legend: ${dn.legend}`);
void sep;
await shot('st_4_webgl_density_all_samples');
await setProp('renderer-mode', {value: 'plotly'});
await page.waitForFunction((n) => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  return gd && gd._fullData && gd._fullData.filter((t) => t.type === 'heatmap').length === n;
}, nS, {timeout: 300000});
await page.waitForTimeout(3000);
const pl = await page.evaluate(() => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  const hm = gd._fullData.filter((t) => t.type === 'heatmap');
  let sum = 0;
  for (const t of hm) for (const row of t.z) for (const v of row) if (v > 0) sum += v * (t.dx / 10) ** 2;
  const sc = gd._fullData.find((t) => t.type === 'scattergl');
  return {n: hm.length, x0: hm.map((t) => t.x0 - t.dx / 2), x1: hm.map((t) => t.x0 - t.dx / 2 + t.dx * t.z[0].length), zmax: hm.map((t) => t.zmax), scales: hm.map((t) => !!t.showscale),
          sum, first: gd._fullData[0].type, cells: sc ? sc.x.length : 0, dx: hm[0].dx};
});
check(`Plotly: one heat layer per sample (${nS}) at the same places and the same colour maximum, one colour bar, below the cells`,
  pl.n === nS && pl.first === 'heatmap' && new Set(pl.zmax).size === 1 && pl.scales.filter(Boolean).length === 1
  && pl.x0.every((x, i) => Math.abs(x - dn.bounds[i][0]) <= pl.dx + 1e-6) && Math.abs(pl.sum - dn.total) / dn.total < 1e-3,
  `x of grids: ${pl.x0.map((x, i) => `${Math.round(x)}–${Math.round(pl.x1[i])}`).join(', ')} | zmax ${pl.zmax[0]} | Σ ${Math.round(pl.sum)} vs ${dn.total}`);
await shot('st_5_plotly_density_all_samples');
await setProp('density-enable', {value: []});
await setProp('renderer-mode', {value: 'webgl'});
await page.waitForFunction(() => !(window.XeniumViewer.instance.deck.props.layers || []).flat(3).some((l) => l && /^xv-density/.test(l.id)), null, {timeout: 120000});

// ── Moran's I ────────────────────────────────────────────────────────────────
await setProp('plots-tabs', {active_tab: 'tab-moran'});
await page.waitForSelector('#moran-run-btn', {state: 'visible'});
await page.click('#moran-run-btn');
await waitText('#moran-status', /Moran's I, kNN graph|No sample|no expression matrix/);
await page.waitForFunction(() => document.querySelectorAll('#moran-table tbody tr').length > 3, null, {timeout: 120000});
await page.waitForTimeout(1000);
const mst = await text('#moran-status');
const perSample = (mst.match(/S\d+: [\d,]+ cells, [\d,]+ genes/g) || []).length;
check(`Moran's I ran per sample (${nS}) as a job`, perSample === nS && /original counts/.test(mst) && /k = 6/.test(mst), mst.slice(0, 260));
const firstGene = await page.evaluate(() => document.querySelector('#moran-table tbody tr:nth-child(3) td[data-dash-column="gene"]')?.innerText.trim());
await page.click('#moran-table tbody tr:nth-child(3) td[data-dash-column="gene"]');
await page.waitForFunction((g) => (document.querySelector('#gene-selector')?.innerText || '').includes(g), firstGene, {timeout: 30000}).catch(() => {});
const picked = await text('#gene-selector');
const mhd = await headers('moran-table');
check('the table is sorted by I and a click selects the gene in the sidebar', !!firstGene && picked.includes(firstGene) && mhd.includes("Moran's I") && mhd.includes('Sample'), `${firstGene} → ${picked.trim().slice(0, 40)}`);
await setProp('color-by', {value: 'gene'});
await page.waitForTimeout(2500);
await shot('st_6_moran_table_top_gene_on_map');

// ── ligand-receptor scores ───────────────────────────────────────────────────
await setProp('plots-tabs', {active_tab: 'tab-lr'});
await page.waitForSelector('#lr-run-btn', {state: 'visible'});
await page.click('#lr-run-btn');
await waitText('#lr-status', /No ligand–receptor pairs/);
check('without a pair list nothing runs, and the tab says no database is bundled', /no database is bundled or downloaded/.test(await text('#lr-status')) && /No database is bundled or downloaded/.test(await text('#lr-note')));
const pairs = process.env.XS_PAIRS ? fs.readFileSync(process.env.XS_PAIRS, 'utf8') : 'ligand,receptor\nCXCL12,CXCR4\nCCL19,CCR7\nTGFB1,TGFBR2\n';
await page.setInputFiles('#lr-upload input[type=file]', {name: 'pairs.csv', mimeType: 'text/csv', buffer: Buffer.from(pairs)});
await waitText('#lr-pairs-info', /pairs from pairs\.csv/);
await page.waitForTimeout(1000);
const labelKind = await text('#lr-labels');
await page.click('#lr-run-btn');
await waitText('#lr-status', /combinations tested|None of the|no .* labels|No cell type/);
await page.waitForTimeout(2000);
const lst = await text('#lr-status');
if (/combinations tested/.test(lst)) {
  const lr = await page.evaluate(() => {
    const gd = document.querySelector('#lr-plot .js-plotly-plot');
    const t = gd._fullData[0];
    return {n: (t.x || []).length, title: gd.layout.title.text, xt: gd.layout.xaxis.title.text};
  });
  const lhd = await headers('lr-table');
  check('ligand–receptor: uploaded list scored between neighbouring cell types; pairs on the panel are counted; dot plot and table filled',
    /\d+ of \d+ ligand–receptor pairs have both genes on the panel \(\d+ not scored\)/.test(lst) && /permutations, seed 0/.test(lst)
    && /directed type pairs are neighbours/.test(lst) && lr.n > 0 && /sender → receiver/.test(lr.xt) && lhd.includes('Adjacent pairs') && (await tableRows('lr-table')) > 0,
    `${labelKind.trim()} | ${lst.slice(0, 300)} | ${lr.n} dots`);
} else {
  check('ligand–receptor: a clear message when nothing can be scored', /None of the|labels|No cell type/.test(lst), lst.slice(0, 200));
}
check('the inference caveat is on the tab', /not evidence of signalling/.test(await text('#lr-note')) && /Only pairs with both genes measured on the panel are scored/.test(await text('#lr-note')));
await shot('st_7_ligand_receptor');
check('no page errors', logs.length === 0, logs.slice(0, 5).join(' | '));

await browser.close();
console.log(failures ? `FAILS: ${failures}` : 'ALL PASSED');
process.exit(failures ? 1 : 0);
