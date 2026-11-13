// Drives the two spatial modes of the Differential Expression tab in the real app:
// "Proximity to a cell type" and "Contact groups (pseudobulk)".
//   PORT=8151 DATA_DIR=<bundle> XENIUM_CACHE_DIR=<dir> python xenium_explorer.py
//   [XP_LABELS=celltype:celltypist] [XP_TARGET=Fibroblasts] [XP_ANCHOR="Mast cells"] [XP_NRAND=30] \
//   [XS_SAMPLES=/path/s1,/path/s2,...] [XP_SHUFFLES=3] \
//     node test/proximity_checks.mjs http://127.0.0.1:8151 /path/to/screenshot_dir
// Needs a cell-type annotation (or uses the clusters). Without XS_SAMPLES the contact mode
// is checked for its refusal of a single sample; with further bundles (added through
// "+ Add sample") it is run with R (DESeq2 or edgeR) and XP_SHUFFLES label shuffles.
import {chromium} from 'playwright';

const base = process.argv[2] || 'http://127.0.0.1:8050';
const out = process.argv[3] || '.';
const extra = (process.env.XS_SAMPLES || '').split(',').filter(Boolean);
const browser = await chromium.launch();
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
await page.waitForSelector('#de-run-btn', {state: 'attached', timeout: 180000});
await page.waitForFunction(() => window.dash_clientside && window.dash_clientside.set_props, null, {timeout: 180000});
await page.waitForTimeout(4000);

const setProp = (id, props) => page.evaluate(([i, p]) => window.dash_clientside.set_props(i, p), [id, props]);
const text = (sel) => page.evaluate((s) => document.querySelector(s)?.innerText || '', sel);
const waitText = async (sel, re, timeout = 1800000) => {
  try {
    await page.waitForFunction(([s, r, f]) => new RegExp(r, f).test(document.querySelector(s)?.innerText || ''), [sel, re.source, re.flags], {timeout});
  } catch (err) {
    throw new Error(`${sel} never matched ${re}: it shows "${(await text(sel)).slice(0, 400)}"`);
  }
};
const shot = (name) => page.screenshot({path: `${out}/${name}.png`});
const click = (sel) => page.evaluate((s) => document.querySelector(s).click(), sel);
const visible = (sel) => page.evaluate((s) => {
  const e = document.querySelector(s);
  return !!e && e.offsetParent !== null;
}, sel);
const headers = (id) => page.evaluate((i) => [...document.querySelectorAll(`#${i} th`)].map((e) => e.innerText.trim()).filter(Boolean), id);
const plotTraces = (id) => page.evaluate((i) => {
  const gd = document.querySelector(`#${i} .js-plotly-plot`) || document.querySelector(`#${i}.js-plotly-plot`);
  return gd && gd.data ? gd.data.map((t) => ({name: t.name || '', n: (t.x || []).length})) : [];
}, id);
const axisTitle = (id) => page.evaluate((i) => {
  const gd = document.querySelector(`#${i} .js-plotly-plot`) || document.querySelector(`#${i}.js-plotly-plot`);
  const t = gd && gd.layout && gd.layout.xaxis && gd.layout.xaxis.title;
  return (t && (t.text || t)) || '';
}, id);
const dropdownOptions = (id) => page.evaluate((i) => {
  const el = document.getElementById(i);
  if (!el) return [];
  const key = Object.keys(el).find((k) => k.startsWith('__reactFiber'));
  // The fiber a DOM node points at may be the alternate of the committed one (React keeps
  // two per component), whose props are those of the render before: both are read and the
  // longer list is the current one in this flow (the lists only grow from empty).
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
await setProp('plots-tabs', {active_tab: 'tab-de'});
await page.waitForFunction(() => { const e = document.querySelector('#de-group-a'); return e && e.getClientRects().length > 0; }, null, {timeout: 30000}).catch(() => {});
await page.waitForTimeout(1500);
check('nothing runs on load (n_clicks = 0)', (await text('#de-status')) === '', await text('#de-status'));
check('cells mode shows the group pickers, not the proximity rows', (await visible('#de-group-a')) && !(await visible('#prox-target')));

// ── proximity ────────────────────────────────────────────────────────────────
await setProp('de-mode', {value: 'proximity'});
await page.waitForFunction(() => /Matched random anchors/.test((document.querySelector('#de-caveat') || {}).textContent || ''), null, {timeout: 30000}).catch(() => {});
await page.waitForTimeout(1000);
check('proximity mode shows its pickers and parameters, hides the others',
  (await visible('#prox-target')) && (await visible('#prox-near')) && !(await visible('#contact-radius')) && !(await visible('#de-group-a')));
check('the caveat explains the mode', /Matched random anchors/.test(await text('#de-caveat')));
const labOpts = await dropdownOptions('prox-labels');
const labels = process.env.XP_LABELS || labOpts.find((v) => String(v).startsWith('celltype:')) || labOpts[0];
check('label sources are offered', labOpts.length > 0 && labOpts.includes(labels), JSON.stringify(labOpts));
await setProp('prox-labels', {value: labels});
// the type lists are filled by a callback that queues behind the others (on a busy machine
// for a long time): wait for the condition - the lists are there - not for a fixed time
let types = [];
for (let i = 0; i < 360 && types.length < 2; i += 1) {
  await page.waitForTimeout(500);
  types = await dropdownOptions('prox-target');
}
const target = process.env.XP_TARGET || types[1] || types[0];
const anchor = process.env.XP_ANCHOR || types[types.length - 1];
check('cell types are listed for T and X', types.length >= 2 && types.includes(target) && types.includes(anchor), `${types.length} types; T = ${target}, X = ${anchor}`);
await click('#de-run-btn');
await page.waitForTimeout(800);
check('Run without T / X asks for them', /Pick the cells to test/.test(await text('#de-status')), await text('#de-status'));
await setProp('prox-target', {value: target});
await setProp('prox-anchor', {value: anchor});
await setProp('prox-nrand', {value: Number(process.env.XP_NRAND || 30)});
await setProp('prox-length', {value: ['on']});
await page.waitForTimeout(500);
let t0 = Date.now();
await click('#de-run-btn');
await page.waitForTimeout(700);
await shot('px_0_running');
await waitText('#de-status', /Proximity of .*Unit of the p-values|needs|No sample|No cell|Only \d+ of/s);
let status = await text('#de-status');
const secs = ((Date.now() - t0) / 1000).toFixed(1);
check('proximity DE finished', /Proximity of .* genes tested/.test(status), `${secs} s in the app: ${status.slice(0, 600)}`);
check('status states the unit and n of the p-values', /Unit of the p-values: n = [\d,]+ (cells|samples|donors)/.test(status), (status.match(/Unit of the p-values[^\n]*/) || [''])[0].slice(0, 260));
check('single-sample result is announced as such (or a cohort test is named)', /SINGLE-SAMPLE RESULT|exact sign-flip/.test(status));
check('ownership and leak gate are reported', /Ownership: [\d,]+ tested genes are owned by/.test(status) && /Leak gate/.test(status));
await page.waitForTimeout(2500);
const hp = await headers('de-table');
check('table has the proximity columns', ['Gene', 'Near − far', 'Slope', 'p', 'Emp. FDR', 'Specific', 'Owner', 'Leak gate', 'L (µm)'].every((h) => hp.includes(h)), hp.join(' | '));
const tr = await plotTraces('de-volcano-plot');
const xt = await axisTitle('de-volcano-plot');
check('volcano shows the result with the proximity axis', tr.length >= 3 && tr.some((t) => /near/.test(t.name)) && /near − far/.test(xt), `${JSON.stringify(tr)} | x: ${xt}`);
check('per-sample table lists T / X / near / far cells', /T cells.*X cells.*Near.*Far/s.test((await text('#de-pb-samples')).replace(/\n/g, ' ')), (await text('#de-pb-samples')).replace(/\s+/g, ' ').slice(0, 200));
await shot('px_1_proximity_result');
await page.evaluate(() => document.querySelector('#de-table').scrollIntoView());
await page.waitForTimeout(500);
await shot('px_2_proximity_table');

// ── contact groups ───────────────────────────────────────────────────────────
await setProp('de-mode', {value: 'contact'});
await page.waitForTimeout(1500);
check('contact mode shows its parameters, hides the proximity ones', (await visible('#contact-radius')) && !(await visible('#prox-near')) && (await visible('#prox-target')));
check('the caveat explains the contact test', /Contact groups: do T cells/.test(await text('#de-caveat')));
if (!extra.length) {
  await click('#de-run-btn');
  await waitText('#de-status', /needs at least 2 samples|Contact groups \(/s, 600000);
  status = await text('#de-status');
  check('one sample: the contact test refuses and points to the proximity mode',
    /needs at least 2 samples/.test(status) && /Proximity to a cell type/.test(status), status.slice(0, 420));
  await shot('px_3_contact_refusal');
} else {
  for (let i = 0; i < extra.length; i += 1) {
    await click('#sample-modal-open-btn');
    await page.waitForSelector('#sample-custom-path', {state: 'visible', timeout: 30000});
    await page.fill('#sample-custom-path', extra[i]);
    await page.click('#sample-add-btn');
    await waitText('#extra-samples-status', new RegExp(`${i + 2} samples loaded`), 900000);
  }
  await page.keyboard.press('Escape');
  await page.waitForTimeout(3000);
  await setProp('plots-tabs', {active_tab: 'tab-de'});
  await setProp('prox-labels', {value: labels});
  await page.waitForTimeout(2500);
  const types2 = await dropdownOptions('prox-target');
  const t2 = process.env.XP_TARGET2 || target;
  const a2 = process.env.XP_ANCHOR2 || anchor;
  check('types of the multi-sample view are listed', types2.includes(t2) && types2.includes(a2), `${types2.length} types; T = ${t2}, X = ${a2}`);
  await setProp('prox-target', {value: t2});
  await setProp('prox-anchor', {value: a2});
  await setProp('contact-nshuffle', {value: Number(process.env.XP_SHUFFLES || 3)});
  await page.waitForTimeout(500);
  check('design summary is shown beside the button', (await text('#prox-design-summary')).length > 10, await text('#prox-design-summary'));
  t0 = Date.now();
  await click('#de-run-btn');
  await page.waitForTimeout(1500);
  await shot('px_3_contact_running');
  await waitText('#de-status', /Contact groups \(.*Unit of the p-values|needs at least 2 samples|failed|not available/s);
  status = await text('#de-status');
  check('contact-group DE finished', /Contact groups \(.* genes tested/s.test(status), `${((Date.now() - t0) / 1000).toFixed(1)} s in the app: ${status.slice(0, 700)}`);
  check('unit of the p-values is the sample', /Unit of the p-values: n = \d+ (samples|donors), each with a contact and a matched comparator pseudobulk/.test(status));
  check('label-shuffle null is reported', /Label shuffle within sample × stratum \(\d+ shuffles/.test(status), (status.match(/Label shuffle[^\n]*/) || [''])[0].slice(0, 300));
  await page.waitForTimeout(2500);
  const hc = await headers('de-table');
  check('table has the pseudobulk columns and the leak gate', ['Gene', 'log2FC (MLE)', 'p', 'FDR', 'Owner', 'Leak gate'].every((h) => hc.includes(h)), hc.join(' | '));
  check('volcano axis is contact / comparator', /contact \/ comparator/.test(await axisTitle('de-volcano-plot')), await axisTitle('de-volcano-plot'));
  check('per-sample table lists contact and comparator cells', /Contact.*Comparator/s.test((await text('#de-pb-samples')).replace(/\n/g, ' ')));
  await shot('px_4_contact_result');
  // proximity across the samples
  await setProp('de-mode', {value: 'proximity'});
  await setProp('prox-nrand', {value: 20});
  await page.waitForTimeout(800);
  t0 = Date.now();
  await click('#de-run-btn');
  await waitText('#de-status', /Proximity of .*Unit of the p-values|No sample|Only \d+ of/s);
  status = await text('#de-status');
  check('proximity across samples finished', /Proximity of .* genes tested/.test(status), `${((Date.now() - t0) / 1000).toFixed(1)} s: ${(status.match(/Unit of the p-values[^\n]*/) || [''])[0].slice(0, 200)}`);
  await page.waitForTimeout(2000);
  await shot('px_5_proximity_multi');
}
check('no page errors', logs.length === 0, logs.slice(0, 4).join(' || '));
console.log(failures ? `FAILS: ${failures}` : 'FAILS: 0');
await browser.close();
process.exit(failures ? 1 : 0);
