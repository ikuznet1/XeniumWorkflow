// The WebGL view behind a proxy that serves the app under a path prefix and strips it.
//   DASH_REQUESTS_PATHNAME_PREFIX=/proxy/ XENIUM_RENDERER=webgl PORT=8097 python xenium_explorer.py <bundle with images>
//   node test/prefix_check.mjs http://127.0.0.1:8097 /proxy
// The "proxy" is a Playwright route: /proxy/x is fetched from /x, anything else under the origin is a 404.
import {chromium} from 'playwright';

const base = process.argv[2] || 'http://127.0.0.1:8097';
const prefix = process.argv[3] || '/proxy';
const gl = process.env.XV_GL || (process.platform === 'darwin' ? 'metal' : 'swiftshader');
const args = gl === 'metal'
  ? ['--use-angle=metal', '--ignore-gpu-blocklist']
  : ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'];

const browser = await chromium.launch({args});
const ctx = await browser.newContext({viewport: {width: 1300, height: 850}});
const outside = [];
const seen = [];
await ctx.route(`${base}/**`, async (route) => {
  const u = new URL(route.request().url());
  if (!u.pathname.startsWith(prefix + '/')) {
    outside.push(u.pathname);
    return route.fulfill({status: 404, body: 'outside the proxied prefix'});
  }
  seen.push(u.pathname);
  u.pathname = u.pathname.slice(prefix.length);
  return route.continue({url: u.href});
});
const page = await ctx.newPage();
const logs = [];
page.on('pageerror', (e) => logs.push(e.message.slice(0, 200)));

let failures = 0;
const check = (name, ok, detail = '') => {
  if (!ok) failures += 1;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
};

await page.goto(`${base}${prefix}/`, {waitUntil: 'domcontentloaded'});
let mounted = true;
await page.waitForFunction(() => window.XeniumViewer && window.XeniumViewer.instance
  && window.XeniumViewer.instance._viewer._cells, null, {timeout: 120000}).catch(() => { mounted = false; });
check('cells load through the prefix', mounted, JSON.stringify(seen.filter((p) => p.includes('/viewer/')).slice(0, 4)));
if (mounted) {
  await page.evaluate(() => {
    window.dash_clientside.set_props('morph-channels', {value: ['dapi']});
    window.dash_clientside.set_props('morph-enable', {value: ['show']});
  });
  let drawn = true;
  await page.waitForFunction(() => (window.XeniumViewer.instance.deck.props.layers || []).flat(3)
    .some((l) => l && /^xenium-morphology/.test(l.id)), null, {timeout: 60000}).catch(() => { drawn = false; });
  await page.waitForTimeout(3000);
  const status = await page.evaluate(() => document.querySelector('#webgl-status')?.textContent || '');
  check('image layer (worker, wasm, tiles) loads through the prefix', drawn && !/unavailable|failed/i.test(status),
    JSON.stringify({status, image: seen.filter((p) => /image|worker|wasm/.test(p)).length}));
}
check('nothing was requested outside the prefix', outside.length === 0, JSON.stringify([...new Set(outside)].slice(0, 8)));
check('no page errors', logs.length === 0, JSON.stringify(logs.slice(0, 4)));
console.log(failures === 0 ? 'ALL PASS' : `FAILURES: ${failures}`);
await browser.close();
process.exit(failures === 0 ? 0 : 1);
