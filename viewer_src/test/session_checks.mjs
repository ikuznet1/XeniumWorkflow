// Session save / restore in the real app, across a restart of the server.
//
// The script starts the app itself, twice (the point is the restart), on 127.0.0.1:
//   run 1: set control values, Save session, Download, let the automatic save write;
//   run 2: a fresh server process and page - nothing is restored on its own; then
//          "Restore last session", "Restore saved session" and uploads of edited files
//          (a newer format version, another dataset, keys a file must never set).
// It creates no ROI and runs no analysis, so nothing is written into the bundle; the
// session slots, the provenance log and whatever else the app caches go to XS_CACHE.
//
//   XS_CACHE=/some/empty/scratch/dir node test/session_checks.mjs [screenshot_dir]
//   XS_DATA   bundle (default: the human skin bundle of lib.mjs; skipped when absent)
//   XS_PORT   port (default 8211)        XV_PYTHON   python to start the app with
import {chromium} from 'playwright';
import {spawn} from 'node:child_process';
import {existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {resolve} from 'node:path';
import {DATA, GL_ARGS, ROOT, hasBundle} from './lib.mjs';

const bundle = process.env.XS_DATA || DATA.skin;
const cache = process.env.XS_CACHE;
const port = Number(process.env.XS_PORT || 8211);
const out = process.argv[2] || '.';
if (!hasBundle(bundle)) {
  console.log(`SKIP  no bundle at ${bundle}`);
  process.exit(0);
}
if (!cache) {
  console.log('usage: XS_CACHE=<empty scratch dir> node test/session_checks.mjs [screenshot_dir]');
  process.exit(2);
}
mkdirSync(cache, {recursive: true});
const base = `http://127.0.0.1:${port}`;
let failures = 0;
const check = (name, ok, detail = '') => {
  if (!ok) failures += 1;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
};

async function startApp() {
  const env = {...process.env, DATA_DIR: bundle, XENIUM_CACHE_DIR: cache, PORT: String(port)};
  delete env.XENIUM_ENABLE_REPL;
  delete env.XENIUM_HOST;
  const proc = spawn(process.env.XV_PYTHON || 'python', [resolve(ROOT, 'xenium_explorer.py')],
    {env, cwd: ROOT, stdio: ['ignore', 'ignore', 'pipe']});
  let err = '';
  proc.stderr.on('data', (d) => { err = (err + d).slice(-4000); });
  const t0 = Date.now();
  for (;;) {
    if (proc.exitCode !== null) throw new Error(`the app exited: ${err}`);
    try {
      if ((await fetch(`${base}/_dash-layout`)).ok) break;
    } catch { /* not up yet */ }
    if (Date.now() - t0 > 600000) {
      proc.kill('SIGTERM');
      throw new Error(`the app did not start: ${err}`);
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  console.log(`      app up (pid ${proc.pid}) after ${Math.round((Date.now() - t0) / 1000)} s`);
  const stop = () => new Promise((done) => {
    if (proc.exitCode !== null) return done();
    proc.on('exit', () => done());
    proc.kill('SIGTERM');
    setTimeout(() => { if (proc.exitCode === null) proc.kill('SIGKILL'); }, 8000);
  });
  return {stop, pid: proc.pid};
}

/** {id: props} of the components of the layout the server sends (their initial props). */
async function layoutProps() {
  const layout = await (await fetch(`${base}/_dash-layout`)).json();
  const byId = {};
  const walk = (n) => {
    if (Array.isArray(n)) return n.forEach(walk);
    if (!n || typeof n !== 'object') return;
    if (n.props) {
      if (typeof n.props.id === 'string') byId[n.props.id] = n.props;
      walk(n.props.children);
    }
  };
  walk(layout);
  return byId;
}

async function openPage(browser) {
  const page = await browser.newPage({viewport: {width: 1440, height: 900}, acceptDownloads: true});
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message.slice(0, 300)));
  // Dash reports a callback dependency cycle on the console, not as a page error.
  page.on('console', (m) => { if (m.type() === 'error' && /Dependency Cycle/.test(m.text())) errors.push('console: ' + m.text().slice(0, 300)); });
  await page.goto(base, {waitUntil: 'domcontentloaded'});
  await page.waitForSelector('#session-open-btn', {state: 'attached', timeout: 180000});
  await page.waitForFunction(() => window.dash_clientside && window.dash_clientside.set_props, null, {timeout: 60000});
  await page.waitForTimeout(4000);
  return {page, errors};
}
const setProp = (page, id, props) => page.evaluate(([i, p]) => window.dash_clientside.set_props(i, p), [id, props]);
const text = (page, sel) => page.evaluate((s) => document.querySelector(s)?.innerText || '', sel);
const openDialog = async (page) => {
  await page.evaluate(() => document.querySelector('#session-open-btn').click());
  await page.waitForSelector('#session-status', {state: 'attached', timeout: 20000});
  await page.waitForTimeout(400);
};
const status = (page) => text(page, '#session-status');
const waitStatus = (page, re) => page.waitForFunction(
  (src) => new RegExp(src).test(document.querySelector('#session-status')?.innerText || ''), re.source, {timeout: 120000});
/** The session of the page as the Download button gives it (what the page really holds). */
async function download(page) {
  const [dl] = await Promise.all([page.waitForEvent('download', {timeout: 60000}),
    page.evaluate(() => document.querySelector('#session-download-btn').click())]);
  return {name: dl.suggestedFilename(), doc: JSON.parse(readFileSync(await dl.path(), 'utf8'))};
}
async function upload(page, doc, name = 'session.json') {
  await page.setInputFiles('#session-upload input[type=file]',
    {name, mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(doc))});
}
const slotFile = (slot) => {
  const dir = resolve(cache, 'sessions');
  const f = existsSync(dir) ? readdirSync(dir).find((n) => n.endsWith(`.${slot}.json`)) : null;
  return f ? JSON.parse(readFileSync(resolve(dir, f), 'utf8')) : null;
};

const browser = await chromium.launch({args: GL_ARGS});
let app = await startApp();
let want = {};
try {
  // ── run 1 ────────────────────────────────────────────────────────────────
  const props = await layoutProps();
  const optionValues = (id) => (props[id]?.options || []).map((o) => (o && typeof o === 'object' ? o.value : o));
  const methods = optionValues('cluster-method'), genes = optionValues('gene-selector');
  let {page, errors} = await openPage(browser);
  check('the Session entry is in the sidebar, the dialog is closed', await page.evaluate(() =>
    !!document.querySelector('#session-entry #session-open-btn') && !document.querySelector('#session-status')));
  await setProp(page, 'session-autosave', {n_intervals: 1000});      // the tick that only notes the untouched page
  await page.waitForTimeout(2500);
  check('an untouched page writes no last session', slotFile('last') === null);
  want = {'point-size': 5, 'moran-k': 9, 'de-mode': 'pseudobulk', 'lr-pairs': 'GENE1,GENE2', 'lr-nperm': 300,
    'bk-preset': 'custom', 'bk-lambda': 0.33, 'color-by': 'gene', 'boundary-toggles': ['cell']};
  if (methods.length > 1) want['cluster-method'] = methods[1];
  if (genes.length > 5) want['gene-selector'] = genes[5];
  for (const [id, value] of Object.entries(want)) await setProp(page, id, {value});
  await page.waitForTimeout(6000);
  await openDialog(page);
  check('dialog: no slot yet', /Saved session: none/.test(await text(page, '#session-slots'))
    && /Last session \(automatic\): none/.test(await text(page, '#session-slots')), (await text(page, '#session-slots')).replace(/\n/g, ' | '));
  await page.evaluate(() => document.querySelector('#session-save-btn').click());
  await waitStatus(page, /Session saved in the cache directory/);
  check('Save session: written under the cache directory', slotFile('saved') !== null, (await status(page)).slice(0, 110));
  const first = await download(page);
  const c1 = first.doc.state.controls;
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  const wrong = (c) => Object.entries(want).filter(([k, v]) => !same(c[k], v)).map(([k, v]) => `${k}: ${JSON.stringify(c[k])} != ${JSON.stringify(v)}`);
  check('Download: a JSON file with the format, version, dataset and the values set in the page',
    /^xenium_session_\d{8}T\d{6}\.json$/.test(first.name) && first.doc.format === 'xenium-explorer-session'
    && first.doc.version === 1 && first.doc.dataset.key === bundle && first.doc.dataset.samples.length === 1
    && first.doc.dataset.n_cells > 0 && wrong(c1).length === 0,
    `${first.name}; ${Object.keys(c1).length} controls; ${first.doc.dataset.n_cells} cells; ${wrong(c1).join('; ')}`);
  check('the file holds no path field, nothing of the REPL and no result',
    !Object.keys(c1).some((k) => /path|repl|-dir$/.test(k)) && !/repl|XENIUM_/i.test(JSON.stringify(first.doc.state))
    && JSON.stringify(first.doc).length < 200000, `${JSON.stringify(first.doc).length} characters`);
  await setProp(page, 'session-autosave', {n_intervals: 2000});
  await page.waitForFunction(() => true, null, {timeout: 1000});
  for (let i = 0; i < 40 && slotFile('last') === null; i++) await page.waitForTimeout(250);
  const last1 = slotFile('last');
  check('automatic save: the changed page is written as the last session',
    !!last1 && last1.kind === 'auto' && wrong(last1.state.controls).length === 0, last1 ? last1.saved_at : 'no file');
  await page.screenshot({path: `${out}/session_1_saved.png`});
  check('run 1: no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  await page.close();
  await app.stop();

  // ── run 2: another server process ────────────────────────────────────────
  app = await startApp();
  ({page, errors} = await openPage(browser));
  await openDialog(page);
  const fresh = (await download(page)).doc.state.controls;
  check('after the restart nothing is restored on its own',
    fresh['moran-k'] === props['moran-k'].value && fresh['point-size'] === props['point-size'].value
    && fresh['lr-pairs'] !== want['lr-pairs'] && fresh['de-mode'] === props['de-mode'].value,
    `moran-k ${fresh['moran-k']}, point-size ${fresh['point-size']}, de-mode ${fresh['de-mode']}`);
  const slots = await text(page, '#session-slots');
  check('dialog: both slots of the run before are offered', /Saved session: 20\d\d-/.test(slots)
    && /Last session \(automatic\): 20\d\d-/.test(slots), slots.replace(/\n/g, ' | '));
  await page.evaluate(() => document.querySelector('#session-restore-last-btn').click());
  await waitStatus(page, /^Session restored/);
  await page.waitForTimeout(6000);
  let st = await status(page);
  const after = (await download(page)).doc.state.controls;
  check('Restore last session: the page holds the values of run 1', wrong(after).length === 0,
    `${st.split('\n')[0]} ${wrong(after).join('; ')}`);
  check('… and reports what it applied', new RegExp(`${Object.keys(want).length} control value\\(s\\) set`).test(st)
    && !/Missing now|Refused values|Unknown keys/.test(st), st.replace(/\n/g, ' | ').slice(0, 200));
  const dom = await page.evaluate(() => document.querySelector('#moran-k')?.value);
  check('the control in the page shows the restored value', dom === undefined || dom === '9', `#moran-k = ${dom}`);
  check('the server run of the restore has not replaced the last session', slotFile('last')?.state.controls['moran-k'] === 9);
  await page.screenshot({path: `${out}/session_2_restored.png`});

  // a file from a newer app is refused whole
  await upload(page, {...first.doc, version: 2});
  await waitStatus(page, /Not restored\. the session file has format version 2/);
  check('upload of a newer format version is refused', true, (await status(page)).slice(0, 120));

  // the SAME file chosen again must fire again (the page empties the upload component after
  // reading it; a trigger on the file's modification time alone would not change). A file on
  // disk, chosen twice: same name, same content, same modification time.
  const sameFile = resolve(mkdtempSync(resolve(tmpdir(), 'xs-upload-')), 'same.json');
  writeFileSync(sameFile, JSON.stringify({...first.doc, version: 2}));
  // As a user does it: the component empties its file input before it opens the dialog
  // (Upload.open: fileInputEl.value = null), then the file is chosen. Without that step
  // React itself drops the second change event of an input whose value did not change.
  const choose = async (path) => {
    await page.evaluate(() => { document.querySelector('#session-upload input[type=file]').value = null; });
    await page.setInputFiles('#session-upload input[type=file]', path);
  };
  await page.evaluate(() => document.querySelector('#session-save-btn').click());
  await waitStatus(page, /^Session saved/);
  await choose(sameFile);
  await waitStatus(page, /Not restored\. the session file has format version 2/);
  await page.evaluate(() => document.querySelector('#session-save-btn').click());
  await waitStatus(page, /^Session saved/);
  let again = true;
  try {
    await choose(sameFile);
    await page.waitForFunction(() => /Not restored\. the session file has format version 2/
      .test(document.querySelector('#session-status')?.innerText || ''), null, {timeout: 20000});
  } catch { again = false; }
  check('the same file uploaded a second time is read again', again, (await status(page)).slice(0, 80));
  check('no dependency cycle is reported after the uploads', !errors.some((e) => /Dependency Cycle/.test(e)),
    errors.filter((e) => /Dependency Cycle/.test(e)).slice(0, 1).join(''));

  // another dataset: only what does not depend on the data; server settings are never read
  const other = JSON.parse(JSON.stringify(first.doc));
  other.dataset.key = other.dataset.samples[0].key = '/somewhere/else/output-other';
  other.state.controls['moran-k'] = 12;
  other.state.controls['cluster-method'] = methods[0];
  other.state.controls['repl-input'] = 'import os';
  other.state.controls['annot-rds-path'] = '../../evil.rds';
  other.state.controls['point-size'] = 'huge';
  other.XENIUM_ENABLE_REPL = 'remote';
  other.host = '0.0.0.0';
  await upload(page, other);
  await waitStatus(page, /restored in part/);
  await page.waitForTimeout(4000);
  st = await status(page);
  const part = (await download(page)).doc.state.controls;
  check('upload for another dataset: restored in part, with what was skipped',
    /saved for another dataset/.test(st) && /data-dependent control value/.test(st) && part['moran-k'] === 12
    && same(part['cluster-method'], after['cluster-method']), st.split('\n').slice(0, 3).join(' | ').slice(0, 220));
  check('… unknown and forbidden keys are named and ignored, a wrong value is refused',
    /'XENIUM_ENABLE_REPL'/.test(st) && /'host'/.test(st) && /control 'repl-input'/.test(st)
    && /control 'annot-rds-path': a path is never taken/.test(st) && /control point-size: 'huge' not a number/.test(st)
    && part['point-size'] === 5);
  const repl = await page.evaluate(() => document.querySelector('#repl-input')?.value || '');
  check('the REPL field is untouched and the REPL stays off', repl === '' && !/import os/.test(repl));

  await page.evaluate(() => document.querySelector('#session-restore-saved-btn').click());
  await waitStatus(page, /^Session restored — saved/);
  await page.waitForTimeout(3000);
  const back = (await download(page)).doc.state.controls;
  check('Restore saved session: back to the saved values', wrong(back).length === 0, wrong(back).join('; '));
  const prov = resolve(cache, 'provenance');
  const records = existsSync(prov) ? readdirSync(prov).flatMap((f) => readFileSync(resolve(prov, f), 'utf8').trim().split('\n').map((l) => JSON.parse(l))) : [];
  const titles = records.filter((r) => r.kind === 'session').map((r) => `${r.title}/${r.params.to || r.params.from}`);
  check('saves and restores are in the provenance log (the automatic save is not)',
    titles.filter((t) => t.startsWith('Session saved')).length >= 2 && titles.filter((t) => t.startsWith('Session restored')).length === 3
    && !records.some((r) => /auto/.test(JSON.stringify(r.params.to || ''))), titles.join(', '));
  check('run 2: no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  await page.close();
} catch (e) {
  failures += 1;
  console.log(`FAIL  the script stopped: ${e.message.split('\n')[0]}`);
} finally {
  await app.stop();
  await browser.close();
}
console.log(failures ? `${failures} FAILED` : 'ALL PASS');
process.exit(failures ? 1 : 0);
