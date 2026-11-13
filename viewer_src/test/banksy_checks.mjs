// Drives the Banksy tab in the real app, in both renderers: run the clustering, the
// cluster table / plot, the new cluster set in the Clustering Method dropdown, and the
// set as Color By -> Cluster on the map (WebGL colours endpoint as the view reads it,
// Plotly traces), a second run from the cache.
//   XENIUM_RENDERER=webgl PORT=8141 DATA_DIR=<bundle> XENIUM_CACHE_DIR=<dir> python xenium_explorer.py
//   [XB_LAMBDA=0.8] [XB_GENES=2000] [XB_RES=0.5] node test/banksy_checks.mjs http://127.0.0.1:8141 /path/to/screenshot_dir
// Writes clustersets_<dataset>/ and banksy_<dataset>/ into the cache directory.
import {chromium} from 'playwright';
import {fetchBin} from './lib.mjs';

const base = process.argv[2] || 'http://127.0.0.1:8050';
const out = process.argv[3] || '.';
const lam = Number(process.env.XB_LAMBDA || 0.8);
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
const tab = async (t) => { await setProp('plots-tabs', {active_tab: t}); await page.waitForTimeout(1200); };
const dropdownOptions = (id) => page.evaluate((i) => {
  const el = document.getElementById(i);
  const key = Object.keys(el).find((k) => k.startsWith('__reactFiber'));
  let f = el[key];
  for (let d = 0; f && d < 30; d += 1, f = f.return) {
    const p = f.memoizedProps;
    if (p && Array.isArray(p.options)) return p.options.map((o) => (o && o.value !== undefined ? [o.value, o.label] : [o, o]));
  }
  return [];
}, id);
const plotlyReady = () => page.waitForFunction(() => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  return gd && gd.data && gd.data.length > 0;
}, null, {timeout: 180000});

if (await page.evaluate(() => getComputedStyle(document.querySelector('#umap-panel')).display === 'none')) {
  await click('#umap-toggle');
}

// ── run ──────────────────────────────────────────────────────────────────────
await tab('tab-banksy');
await shot('banksy_0_empty_state');
check('empty state before the first run', /No Banksy clusters computed yet/.test(await text('#bk-plot')) && (await tableRows('bk-table')) === 0);
check('the tab says what the method is and that it is a native implementation',
      /Singhal et al\., Nat Genet 2024/.test(await text('#bk-note')) && /not the Banksy package/.test(await text('#bk-note')));
const before = (await dropdownOptions('cluster-method')).map((o) => o[0]);
await setProp('bk-preset', {value: 'custom'});
await setProp('bk-lambda', {value: lam});
if (process.env.XB_GENES) await setProp('bk-maxgenes', {value: Number(process.env.XB_GENES)});
if (process.env.XB_RES) await setProp('bk-resolution', {value: Number(process.env.XB_RES)});
await page.waitForTimeout(600);
let t0 = Date.now();
await click('#bk-run-btn');
let jobsTxt = '';
for (let i = 0; i < 30 && !/Banksy/.test(jobsTxt); i += 1) {
  await page.waitForTimeout(300);
  jobsTxt = await text('#jobs-panel');
}
check('the job is listed in the jobs panel with Cancel', /Banksy/.test(jobsTxt) && /Cancel/i.test(jobsTxt), jobsTxt.replace(/\s+/g, ' ').slice(0, 120));
await waitText('#bk-status', /cluster\(s\) of [\d,]+ cells/);
let secs = (Date.now() - t0) / 1000;
const status = await text('#bk-status');
check('Banksy clusters computed', true, `${secs.toFixed(1)} s in the app: ${status.slice(0, 700)}`);
check('status names the features, the neighbours within the sample and the backend',
      /own expression/.test(status) && /spatial neighbours within the sample/.test(status) && /(leiden|kmeans)/.test(status));
check('status gives the spatial coherence against scattered labels', /Spatial coherence: on average \d+ %/.test(status));
await page.waitForTimeout(2500);
const K = Number((status.match(/: (\d+) cluster\(s\)/) || [0, '0'])[1]);
check('cluster table is filled', K > 0 && (await tableRows('bk-table')) > 0, `${K} clusters`);
const traces = await page.evaluate(() => {
  const gd = document.querySelector('#bk-plot .js-plotly-plot');
  return gd && gd.data ? gd.data.map((t) => t.type) : [];
});
check('plot has the cluster sizes and the top-gene heat map', traces.includes('bar') && traces.includes('heatmap'), traces.join(','));
await shot('banksy_1_result_tab');

// ── the cluster set ──────────────────────────────────────────────────────────
const after = await dropdownOptions('cluster-method');
const added = after.filter((o) => !before.includes(o[0]));
const want = after.find((o) => String(o[1]) === `Banksy λ=${lam}`);
check('Clustering Method lists the new cluster set', !!want, JSON.stringify(added.length ? added : after.slice(-2)));
const key = want ? want[0] : null;
await setProp('color-by', {value: 'cluster'});
await setProp('cluster-method', {value: key});
await page.waitForTimeout(4000);
const col = await fetchBin(`${base}/viewer/colors.bin?mode=cluster&method=${encodeURIComponent(key)}`);
const cats = col.header.categories || [];
const counts = new Array(cats.length).fill(0);
for (const c of col.arrays.codes) if (c < counts.length) counts[c] += 1;
check('WebGL colours: one category per Banksy cluster', col.header.kind === 'categorical' && cats.filter((c) => c !== 'Cluster 0').length === K,
      `${cats.length} categories: ${cats.slice(0, 6).join(', ')}…; largest ${Math.max(...counts)} cells`);
const legend = await page.evaluate(() => {
  const v = window.XeniumViewer.instance;
  const c = v && v._viewer && v._viewer.state && v._viewer.state.colors;
  return c && c.categories ? c.categories.length : (document.querySelector('#webgl-view')?.innerText || '').match(/Cluster \d+/g)?.length || 0;
});
check('WebGL view shows the cluster set', legend >= K, `${legend} legend entries`);
await shot('banksy_2_webgl_map');

await setProp('renderer-mode', {value: 'plotly'});
await plotlyReady();
await page.waitForTimeout(4000);
const names = await page.evaluate(() => {
  const gd = document.querySelector('#spatial-plot .js-plotly-plot');
  return gd.data.filter((t) => /^(Cluster |C)\d+$/.test(t.name || '')).map((t) => t.name);
});
check('Plotly map: one trace per Banksy cluster', names.filter((n) => !/^(Cluster |C)0$/.test(n)).length === K, `${names.length} traces`);
await shot('banksy_3_plotly_map');
await setProp('renderer-mode', {value: 'webgl'});
await page.waitForTimeout(3000);

// ── the group pickers see it ─────────────────────────────────────────────────
await tab('tab-de');
await page.waitForTimeout(2500);
const groups = (await dropdownOptions('de-group-a')).map((o) => String(o[1]));
check('Differential Expression offers the Banksy clusters as groups', groups.filter((g) => /Cluster|^C?\d+$|C\d+/.test(g)).length >= Math.min(K, 2),
      groups.slice(0, 8).join(' | '));

// ── second run: from the cache ───────────────────────────────────────────────
await tab('tab-banksy');
t0 = Date.now();
await click('#bk-run-btn');
await waitText('#bk-status', /read from the cache/);
secs = (Date.now() - t0) / 1000;
check('the same request again is read from the cache', secs < 60, `${secs.toFixed(1)} s`);

const bad = logs.filter((l) => !/favicon|ResizeObserver/.test(l));
check('no page errors', bad.length === 0, bad.slice(0, 3).join(' | '));
console.log(`\n${failures === 0 ? 'ALL PASS' : failures + ' FAILED'}`);
await browser.close();
process.exit(failures ? 1 : 0);
