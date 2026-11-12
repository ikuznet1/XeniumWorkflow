// The app must never show an empty spatial panel when the WebGL view cannot run: it goes
// back to the Plotly renderer by itself and says why in the status line.
//   XENIUM_RENDERER=webgl PORT=8062 python xenium_explorer.py <bundle>
//   node test/webgl_fallback.mjs http://127.0.0.1:8062 [screenshot dir]
// Four browsers are started against the same app:
//   1. Chromium without WebGL2 but with WebGL 1 (--disable-webgl2): Plotly takes over and draws
//   2. Chromium with WebGL switched off altogether (--disable-gpu --disable-webgl
//      --disable-webgl2): Plotly's point layer needs WebGL 1, so no renderer can draw the
//      cells; the status line says so and Plotly shows its own notice (no empty panel)
//   3. Chromium with WebGL, where the WebGL2 context for the viewer's own canvas is refused
//      (the probe succeeds, the device creation fails)
//   4. Chromium with WebGL: no fallback, the WebGL view stays
// XV_BROWSER=firefox|webkit runs case 4 only, in that browser (if Playwright has it).
import {chromium, firefox, webkit} from 'playwright';
import {mkdirSync} from 'node:fs';
import {resolve} from 'node:path';
import {GL_ARGS} from './lib.mjs';

const base = process.argv[2] || 'http://127.0.0.1:8050';
const out = process.argv[3] || '';
if (out) mkdirSync(out, {recursive: true});
const other = {firefox, webkit}[process.env.XV_BROWSER || ''];

let failures = 0;
const check = (name, ok, detail = '') => {
  if (!ok) failures += 1;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
};
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function open(type, args, init) {
  const browser = await type.launch({headless: true, args});
  const page = await browser.newPage({viewport: {width: 1400, height: 850}});
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message.slice(0, 200)));
  if (init) await page.addInitScript(init);
  await page.goto(base, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.dash_clientside && document.querySelector('#renderer-mode') && document.querySelector('#spatial-plot'), null, {timeout: 120000});
  return {browser, page, errors};
}

const state = (page) => page.evaluate(() => {
  const shown = (sel) => { const e = document.querySelector(sel); return !!e && getComputedStyle(e).display !== 'none' && e.getBoundingClientRect().height > 50; };
  const g = document.querySelector('#spatial-plot .js-plotly-plot');
  const radios = [...document.querySelectorAll('#renderer-mode input')];
  const probe = document.createElement('canvas').getContext('webgl2');
  const st = document.querySelector('#webgl-status');
  const bar = st.parentElement.getBoundingClientRect();
  const top = document.elementFromPoint(bar.x + 12, bar.y + bar.height / 2);
  const sr = st.getBoundingClientRect();
  return {
    webgl2: !!probe,
    webgl1: !!document.createElement('canvas').getContext('webgl'),
    // the renderer bar with the status line is on top of whatever the panel shows
    barOnTop: !!top && st.parentElement.contains(top),
    statusLines: sr.height > 0 ? Math.round(sr.height / parseFloat(getComputedStyle(st).lineHeight || 14)) || 1 : 0,
    plotlyNotice: !!document.querySelector('#spatial-plot .no-webgl'),
    plotlyCanvas: !!document.querySelector('#spatial-plot canvas'),
    mode: (radios.find((r) => r.checked) || {}).value || (window.dash_component_api && window.dash_component_api.getLayout('renderer-mode').props.value),
    webglOptionDisabled: radios.some((r) => r.disabled),
    labels: [...document.querySelectorAll('#renderer-mode label')].map((l) => l.textContent.trim()),
    plotlyShown: shown('#spatial-plot'),
    plotlyCells: g && g._fullData ? g._fullData.reduce((n, t) => n + (t.mode === 'markers' && t.x ? t.x.length : 0), 0) : 0,
    glShown: shown('#webgl-view'),
    glCanvas: !!document.querySelector('#webgl-view canvas'),
    status: document.querySelector('#webgl-status').textContent,
  };
});
const fellBack = async (page) => {
  try {
    await page.waitForFunction(() => {
      const g = document.querySelector('#spatial-plot .js-plotly-plot');
      return /WebGL (view unavailable|is switched off)/.test(document.querySelector('#webgl-status').textContent) && g && g._fullData && g._fullData.some((t) => t.mode === 'markers' && t.x && t.x.length);
    }, null, {timeout: 120000});
  } catch { /* reported by the checks */ }
  await wait(1500);
  return state(page);
};
const fallbackOk = (s) => s.mode === 'plotly' && s.plotlyShown && s.plotlyCells > 0 && s.plotlyCanvas && !s.plotlyNotice && !s.glShown && s.webglOptionDisabled &&
  s.barOnTop && /WebGL view unavailable: .+ - showing the Plotly view/.test(s.status) && !s.status.includes('\n');

if (!other) {
  // 1. no WebGL2, WebGL 1 present
  let b = await open(chromium, [...GL_ARGS, '--disable-webgl2']);
  let s = await fellBack(b.page);
  if (out) await b.page.screenshot({path: resolve(out, 'fallback_no_webgl2.png')});
  check('WebGL2 disabled in the browser: no WebGL2, WebGL 1 present', s.webgl2 === false && s.webgl1 === true, JSON.stringify({webgl2: s.webgl2, webgl1: s.webgl1}));
  check('WebGL2 disabled: the app draws the cells with Plotly, the WebGL option is disabled and one line says why', fallbackOk(s), JSON.stringify(s));
  check('WebGL2 disabled: no page errors', b.errors.length === 0, JSON.stringify(b.errors.slice(0, 3)));
  await b.browser.close();

  // 2. no WebGL at all
  b = await open(chromium, ['--disable-gpu', '--disable-webgl', '--disable-webgl2']);
  s = await fellBack(b.page);
  if (out) await b.page.screenshot({path: resolve(out, 'fallback_no_webgl.png')});
  check('WebGL off altogether: Plotly renderer selected, its notice shown (no empty panel), the status line on top says neither view can draw',
    s.webgl2 === false && s.webgl1 === false && s.mode === 'plotly' && s.plotlyShown && s.plotlyNotice && !s.glShown && s.webglOptionDisabled && s.barOnTop &&
    /^WebGL is switched off in this browser: neither .+ can draw cells/.test(s.status), JSON.stringify(s));
  check('WebGL off altogether: no page errors', b.errors.length === 0, JSON.stringify(b.errors.slice(0, 3)));
  await b.browser.close();

  // 3. WebGL2 present for the probe, refused for the viewer's canvas (WebGL 1 stays, for Plotly)
  b = await open(chromium, GL_ARGS, () => {
    const orig = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type, ...rest) {
      if (type === 'webgl2' && this.isConnected) return null;   // the glue's detached probe canvas gets one
      return orig.call(this, type, ...rest);
    };
  });
  s = await fellBack(b.page);
  if (out) await b.page.screenshot({path: resolve(out, 'fallback_context_refused.png')});
  check('context refused for the viewer: the app falls back to Plotly with one line of explanation', fallbackOk(s), JSON.stringify(s));
  await b.browser.close();
}

// 4. WebGL available: nothing changes
const b = await open(other || chromium, other ? [] : GL_ARGS);
let ok = true;
try {
  await b.page.waitForFunction(() => window.XeniumViewer && window.XeniumViewer.instance && window.XeniumViewer.instance.stats.firstRenderMs !== undefined, null, {timeout: 120000});
} catch { ok = false; }
await wait(1500);
const s = await state(b.page);
if (out) await b.page.screenshot({path: resolve(out, `webgl_available_${process.env.XV_BROWSER || 'chromium'}.png`)});
check(`WebGL available (${process.env.XV_BROWSER || 'chromium'}): the WebGL view renders, no fallback message`,
  ok && s.mode === 'webgl' && s.glShown && s.glCanvas && !s.plotlyShown && !s.webglOptionDisabled && !/unavailable/.test(s.status), JSON.stringify(s));
await b.browser.close();

console.log(failures === 0 ? 'ALL PASS' : `FAILURES: ${failures}`);
process.exit(failures === 0 ? 0 : 1);
