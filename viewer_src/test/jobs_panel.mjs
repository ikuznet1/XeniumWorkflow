// Jobs panel in the real Dash app: a running Proseg job is shown with progress, and
// clicking Cancel stops it. Needs a bundle whose SpatialData store and patches are cached
// (Prepare Patches then only reloads them) and Proseg on the server.
//   DATA_DIR=<bundle> XENIUM_CACHE_DIR=<cache> PORT=8064 python xenium_explorer.py
//   node test/jobs_panel.mjs http://127.0.0.1:8064 /path/to/screenshot_dir
// The run is cancelled, so nothing is added to the cache. XV_GL as in app_smoke.mjs.
import {chromium} from 'playwright';

const base = process.argv[2] || 'http://127.0.0.1:8050';
const out = process.argv[3] || '.';
const gl = process.env.XV_GL || (process.platform === 'darwin' ? 'metal' : 'swiftshader');
const args = gl === 'metal'
  ? ['--use-angle=metal', '--ignore-gpu-blocklist']
  : ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'];

const browser = await chromium.launch({args});
const page = await browser.newPage({viewport: {width: 1500, height: 900}});
const logs = [];
const panelPolls = [];
page.on('console', (m) => { if (m.type() === 'error') logs.push(`error: ${m.text().slice(0, 300)}`); });
page.on('pageerror', (e) => logs.push('pageerror: ' + e.message.slice(0, 300)));
page.on('request', (r) => {
  if (!r.url().endsWith('/_dash-update-component')) return;
  const body = r.postData() || '';
  if (body.includes('"jobs-poll.n_intervals"') || body.includes('"id":"jobs-poll","property":"n_intervals"')) panelPolls.push(Date.now());
});

let failures = 0;
const check = (name, ok, detail = '') => {
  if (!ok) failures += 1;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
};
const setProp = (id, props) => page.evaluate(([i, p]) => window.dash_clientside.set_props(i, p), [id, props]);
const click = (sel) => page.evaluate((s) => document.querySelector(s).click(), sel);   // sidebar is hover-reveal
const text = (sel) => page.evaluate((s) => (document.querySelector(s)?.innerText || '').trim(), sel);
const panel = () => page.evaluate(() => {
  const p = document.querySelector('#jobs-panel');
  return {
    display: getComputedStyle(p).display,
    rows: [...p.querySelectorAll('.job-row')].map((r) => ({
      job: r.dataset.job, state: r.dataset.state,
      step: r.querySelector('.job-step')?.innerText || '',
      elapsed: r.querySelector('.job-elapsed')?.innerText || '',
      fraction: r.querySelector('.job-progress') ? Number(r.querySelector('.job-progress').dataset.fraction) : null,
      barWidth: r.querySelector('.job-progress-fill')?.style.width || null,
      cancel: !!r.querySelector('button'),
      text: r.innerText.replace(/\n+/g, ' | '),
    })),
  };
});
const waitPanel = async (pred, timeout, what) => {
  const t0 = Date.now();
  for (;;) {
    const p = await panel();
    if (pred(p)) return p;
    if (Date.now() - t0 > timeout) throw new Error(`timed out waiting for ${what}: ${JSON.stringify(p)}`);
    await page.waitForTimeout(250);
  }
};

await page.goto(base, {waitUntil: 'domcontentloaded'});
await page.waitForFunction(() => document.querySelector('#spatial-plot .main-svg, #webgl-view canvas'), null, {timeout: 180000});
await page.waitForTimeout(3000);

// 0. idle: hidden, and its interval is off
let p = await panel();
check('0 no job: panel hidden', p.display === 'none' && p.rows.length === 0, JSON.stringify(p));
let n0 = panelPolls.length;
await page.waitForTimeout(3500);
check('0 no job: the panel interval is not ticking', panelPolls.length === n0, `${panelPolls.length - n0} requests in 3.5 s`);

// 1. Prepare Patches (cached): the three steps appear as jobs
await click('#reseg-modal-open-btn');
await page.waitForSelector('#reseg-modal-run-btn', {state: 'visible', timeout: 20000});
await page.locator('#sdata-prepare-btn').click();
// (no job when this app instance has the patches in memory already, e.g. on a second run)
let prep = null;
try {
  prep = await waitPanel((q) => q.display !== 'none' && q.rows.some((r) => /patches/.test(r.text)), 10000, 'a patches job in the panel');
} catch { /* checked below */ }
check('1 Prepare Patches is listed as a job', prep !== null || await page.locator('#reseg-confirm-patches-btn').isVisible(),
  prep ? prep.rows.map((r) => r.text).join(' || ') : 'patches were already loaded: nothing to run');
await page.screenshot({path: `${out}/jobs_1_prepare_patches.png`});
await page.waitForSelector('#reseg-confirm-patches-btn', {state: 'visible', timeout: 900000});
console.log(`   sdata status: ${await text('#sdata-status')}`);
await page.locator('#reseg-confirm-patches-btn').click();
await page.waitForTimeout(800);

// 2. start Proseg on the patches
await setProp('reseg-algo-tabs', {value: 'proseg'});
await page.waitForTimeout(500);
await page.locator('#reseg-modal-run-btn').click();
p = await waitPanel((q) => q.rows.some((r) => r.state === 'running' && /^proseg/i.test(r.text)), 60000, 'a running Proseg job');
check('2 running Proseg job is listed with a Cancel button',
  p.rows.filter((r) => r.state === 'running' && /proseg/i.test(r.text)).every((r) => r.cancel), JSON.stringify(p.rows));
const jobId = p.rows.find((r) => r.state === 'running' && /^proseg/i.test(r.text)).job;
const row = (q) => q.rows.find((r) => r.job === jobId);

// 3. progress: a fraction from Proseg's own output, and a clock that runs
p = await waitPanel((q) => row(q) && row(q).fraction !== null && row(q).fraction > 0.004 && /sampling/.test(row(q).step), 1200000, 'Proseg sampler progress');
const first = row(p);
check('3 progress bar with a fraction and the step text', first.fraction > 0 && first.fraction < 1 && /Proseg patch \d+\/\d+: sampling \d+%/.test(first.step)
  && Math.abs(parseFloat(first.barWidth) - 100 * first.fraction) < 0.06, JSON.stringify(first));
check('3 the dialog status line still works', /patch|Proseg/i.test(await text('#proseg-status')), await text('#proseg-status'));
// the panel is above an open dialog (the Run button closed it: open it again): the Cancel
// button is what a click at its centre hits
await click('#reseg-modal-open-btn');
await page.waitForSelector('.modal.show #reseg-modal-close-btn', {state: 'visible', timeout: 20000});
await page.waitForTimeout(600);
const hit = await page.evaluate((id) => {
  const b = document.querySelector(`.job-row[data-job="${id}"] button`);
  const r = b.getBoundingClientRect();
  const el = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
  return {isButton: el === b, modalOpen: !!document.querySelector('.modal.show'), tag: el && el.tagName};
}, jobId);
check('3 panel is on top of the open dialog', hit.isButton && hit.modalOpen, JSON.stringify(hit));
await page.screenshot({path: `${out}/jobs_2_running_modal.png`});
await page.locator('#jobs-panel').screenshot({path: `${out}/jobs_2_panel.png`});
// the list can be folded away (it covers the corner of the log), and stays folded while it updates
await page.locator('#jobs-panel-summary').click();
await page.waitForTimeout(2500);
const folded = await page.evaluate(() => ({open: document.querySelector('#jobs-panel').open,
  height: document.querySelector('#jobs-panel').getBoundingClientRect().height,
  summary: document.querySelector('#jobs-panel-summary').innerText.replace(/\n+/g, ' ')}));
check('3 the job list folds to its header line', folded.open === false && folded.height < 45 && /Jobs\s*1 running/.test(folded.summary), JSON.stringify(folded));
await page.locator('#jobs-panel').screenshot({path: `${out}/jobs_2_panel_folded.png`});
await page.locator('#jobs-panel-summary').click();
await page.waitForTimeout(1500);
const later = row(await panel());
check('3 progress and elapsed time advance', later.fraction >= first.fraction && later.elapsed !== first.elapsed, JSON.stringify({first: [first.fraction, first.elapsed], later: [later.fraction, later.elapsed]}));

// 4. same panel with the other renderer, dialog closed
const renderer0 = await page.evaluate(() => getComputedStyle(document.querySelector('#webgl-view')).display === 'none' ? 'plotly' : 'webgl');
await page.locator('#reseg-modal-close-btn').click();
await page.waitForTimeout(600);
await page.screenshot({path: `${out}/jobs_3_${renderer0}.png`});
const other = renderer0 === 'plotly' ? 'webgl' : 'plotly';
await setProp('renderer-mode', {value: other});
if (other === 'webgl') await page.waitForFunction(() => document.querySelector('#webgl-view canvas'), null, {timeout: 120000});
await page.waitForTimeout(5000);
p = await panel();
const shown = await page.evaluate(() => ({
  webgl: getComputedStyle(document.querySelector('#webgl-view')).display,
  plotly: getComputedStyle(document.querySelector('#spatial-plot')).display}));
check(`4 panel still shown and updating after switching ${renderer0} -> ${other}`,
  p.display !== 'none' && row(p) && row(p).state === 'running' && row(p).cancel && (other === 'webgl' ? shown.webgl !== 'none' : shown.plotly !== 'none'),
  JSON.stringify({shown, row: row(p)}));
await page.screenshot({path: `${out}/jobs_4_${other}.png`});

// 5. Cancel (a real mouse click on the button in the panel)
const tCancel = Date.now();
await page.locator(`.job-row[data-job="${jobId}"] button`).click();
p = await waitPanel((q) => row(q) && row(q).state !== 'running', 15000, 'the job to leave the running state');
console.log(`   after the click: ${JSON.stringify(row(p))}`);
p = await waitPanel((q) => row(q) && row(q).state === 'cancelled', 60000, 'the job to be cancelled');
const took = (Date.now() - tCancel) / 1000;
check('5 Cancel stops the job', /cancelled/.test(row(p).text) && !row(p).cancel, `${took.toFixed(1)} s  ${row(p).text}`);
await page.locator('#jobs-panel').screenshot({path: `${out}/jobs_5_cancelled.png`});
await page.waitForFunction(() => /Cancelled/.test(document.querySelector('#proseg-status')?.innerText || ''), null, {timeout: 20000}).catch(() => {});
check('5 the tool status line reports the cancel', /✗ Cancelled/.test(await text('#proseg-status')), await text('#proseg-status'));

// 6. the panel goes away and stops polling
p = await waitPanel((q) => q.display === 'none', 30000, 'the panel to hide');
await page.waitForTimeout(1500);
n0 = panelPolls.length;
await page.waitForTimeout(4000);
check('6 panel hidden again and its interval is off', p.rows.length === 0 && panelPolls.length === n0, `${panelPolls.length - n0} requests in 4 s`);

check('no console errors', logs.length === 0, logs.slice(0, 5).join(' ; '));
console.log(failures ? `${failures} FAILED` : 'ALL PASSED');
await browser.close();
process.exit(failures ? 1 : 0);
