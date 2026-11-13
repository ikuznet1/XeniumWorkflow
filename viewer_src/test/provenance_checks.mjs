// The Provenance tab in the real app (either renderer): the record list, the row click that
// opens the full record, the kind filter and the two exports. Needs at least one recorded
// analysis for the loaded dataset (run any analysis first, or use a cache that has records).
//   PORT=8171 DATA_DIR=<bundle copy> XENIUM_CACHE_DIR=<dir> python xenium_explorer.py
//   node test/provenance_checks.mjs http://127.0.0.1:8171 /path/to/screenshot_dir
import {chromium} from 'playwright';
import {openTab, GL_ARGS} from './lib.mjs';

const base = process.argv[2] || 'http://127.0.0.1:8050';
const out = process.argv[3] || '.';
const browser = await chromium.launch({args: GL_ARGS});
let failures = 0;
const check = (name, ok, detail = '') => {
  if (!ok) failures += 1;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
};
const page = await browser.newPage({viewport: {width: 1440, height: 900}, acceptDownloads: true});
const logs = [];
page.on('pageerror', (e) => logs.push('pageerror: ' + e.message.slice(0, 300)));
await page.goto(base, {waitUntil: 'domcontentloaded'});
await page.waitForSelector('#plots-nav', {state: 'attached', timeout: 180000});
await page.waitForFunction(() => window.dash_clientside && window.dash_clientside.set_props, null, {timeout: 60000});
await page.waitForTimeout(3000);
const text = (sel) => page.evaluate((s) => document.querySelector(s)?.innerText || '', sel);

await openTab(page, 'tab-provenance');
await page.waitForFunction(() => (document.querySelector('#prov-status')?.innerText || '').length > 10, null, {timeout: 60000});
const status = await text('#prov-status');
const rowCount = () => page.evaluate(() => document.querySelectorAll('#prov-table td[data-dash-column="id"]').length);
const n = await rowCount();
if (!n) {
  console.log(`SKIP  no provenance record for this dataset (${status.slice(0, 120)}): run an analysis first`);
  await browser.close();
  process.exit(0);
}
check('the table lists the records', n > 0 && /record\(s\) for/.test(status), `${n} row(s) on the page; ${status.slice(0, 100)}`);
check('before a click the detail says what to do', /Click a row/.test(await text('#prov-detail')));

// click a cell of the first row, then of another row: the detail is that row's full record
const idOf = (row) => page.evaluate((r) => {
  const cells = document.querySelectorAll('#prov-table td[data-dash-column="id"]');
  return cells[r] ? cells[r].innerText.trim() : null;
}, row);
const clickRow = async (row, column) => {
  const cell = page.locator(`#prov-table td[data-dash-column="${column}"]`).nth(row);
  await cell.scrollIntoViewIfNeeded();
  await cell.click();
};
const id0 = await idOf(0);
await clickRow(0, 'title');
await page.waitForFunction((id) => (document.querySelector('#prov-detail')?.innerText || '').includes(id), id0, {timeout: 20000}).catch(() => {});
let detail = await text('#prov-detail');
let rec = null;
try { rec = JSON.parse(detail); } catch { /* reported below */ }
check('row click: the detail shows the full record of that row', !!rec && rec.id === id0, `row id ${id0}; detail starts: ${detail.slice(0, 80).replace(/\s+/g, ' ')}`);
check('row click: the record has its data, parameters, software and status line',
      !!rec && !!rec.data && 'params' in rec && !!rec.software && 'status_line' in rec && !!rec.kind,
      rec ? `kind ${rec.kind}; counts ${JSON.stringify(rec.data.counts)}; cache ${rec.cache}` : '');
if (n > 1) {
  const last = Math.min(n - 1, 3);
  const id1 = await idOf(last);
  await clickRow(last, 'time');
  await page.waitForFunction((id) => (document.querySelector('#prov-detail')?.innerText || '').includes(id), id1, {timeout: 20000}).catch(() => {});
  detail = await text('#prov-detail');
  check('row click on another row (another column): the detail follows', detail.includes(id1) && id1 !== id0, `row ${last}: ${id1}`);
}
await page.screenshot({path: `${out}/provenance_1_detail.png`});

// the kind filter narrows the list, and a click still opens the row that is shown
const kind = rec ? rec.kind : null;
if (kind) {
  await page.evaluate((k) => window.dash_clientside.set_props('prov-kind', {value: k}), kind);
  await page.waitForFunction(() => /of kind/.test(document.querySelector('#prov-status')?.innerText || ''), null, {timeout: 20000}).catch(() => {});
  await page.waitForTimeout(800);
  const m = await rowCount();
  const idf = await idOf(0);
  await clickRow(0, 'kind');
  await page.waitForFunction((id) => (document.querySelector('#prov-detail')?.innerText || '').includes(id), idf, {timeout: 20000}).catch(() => {});
  detail = await text('#prov-detail');
  let r2 = null;
  try { r2 = JSON.parse(detail); } catch { /* below */ }
  check('kind filter: the list is narrowed and a row click opens a record of that kind', m > 0 && m <= n && !!r2 && r2.id === idf && r2.kind === kind, `${m} of ${n} row(s), kind ${kind}`);
  await page.evaluate(() => window.dash_clientside.set_props('prov-kind', {value: null}));
}

// the exports
for (const [btn, name, re] of [['#prov-export-md-btn', 'methods.md', /^# Methods/], ['#prov-export-json-btn', 'provenance.json', /"records"/]]) {
  const [dl] = await Promise.all([page.waitForEvent('download', {timeout: 30000}), page.click(btn)]);
  const file = await dl.path();
  const fs = await import('fs');
  const body = fs.readFileSync(file, 'utf8');
  check(`export ${name}`, dl.suggestedFilename() === name && re.test(body), `${body.length} characters`);
}
check('no page errors', logs.length === 0, logs.slice(0, 3).join(' | '));
console.log(failures ? `${failures} FAILED` : 'ALL PASS');
await browser.close();
process.exit(failures ? 1 : 0);
