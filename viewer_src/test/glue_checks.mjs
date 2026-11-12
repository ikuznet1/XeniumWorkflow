// Checks of the Dash glue (assets/xenium_viewer_dash.js) in the real app: renderer
// switches, image channel state, status line, context loss, pending ROI outline.
//   XENIUM_RENDERER=webgl PORT=8096 python xenium_explorer.py <skin bundle>
//   node test/glue_checks.mjs http://127.0.0.1:8096 /path/to/screenshot_dir
// It saves nothing in the app's cache (the ROI dialog is cancelled).
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
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function open(setup) {
  const page = await browser.newPage({viewport: {width: 1500, height: 900}});
  const logs = [];
  const reqs = [];
  page.on('pageerror', (e) => logs.push('pageerror: ' + e.message.slice(0, 300)));
  page.on('request', (r) => {
    const u = new URL(r.url());
    if (u.pathname.includes('/viewer/')) reqs.push(u.pathname + u.search);
  });
  if (setup) await setup(page);
  await page.goto(base, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => document.querySelector('#webgl-view canvas'), null, {timeout: 120000});
  await page.waitForFunction(() => window.XeniumViewer.instance && window.XeniumViewer.instance.stats
    && window.XeniumViewer.instance.getState && window.XeniumViewer.instance._viewer._cells, null, {timeout: 120000});
  await page.waitForTimeout(2000);
  const setProp = (id, props) => page.evaluate(([i, p]) => window.dash_clientside.set_props(i, p), [id, props]);
  const status = () => page.evaluate(() => document.querySelector('#webgl-status')?.textContent || '');
  const imageLayer = () => page.evaluate(() => {
    const ls = (window.XeniumViewer.instance.deck.props.layers || []).flat(3).filter((l) => l && /^xenium-morphology/.test(l.id));
    return ls.map((l) => ({sel: l.props.selections.map((s) => s.c), vis: l.props.channelsVisible.slice()}));
  });
  return {page, logs, reqs, setProp, status, imageLayer};
}

// ───────────────────────── page 1: colour maps, zoom carry, image channels, lasso ─────────────────────────
{
  const {page, logs, reqs, setProp, status, imageLayer} = await open();
  const bounds = await page.evaluate(async () => (await (await fetch('/viewer/meta')).json()).bounds);

  // colour scale per colour mode (Plotly: gene Plasma, COLORSCALES otherwise)
  const maps = {};
  for (const mode of ['gene', 'cell_area', 'nucleus_area', 'total_counts', 'transcript_counts']) {
    await setProp('color-by', {value: mode});
    await page.waitForTimeout(700);
    maps[mode] = await page.evaluate(() => window.XeniumViewer.instance.getState().colormap);
  }
  check('colormap follows the Plotly colour scales',
    maps.gene === 'plasma' && maps.cell_area === 'plasma' && maps.nucleus_area === 'cividis'
    && maps.total_counts === 'magma' && maps.transcript_counts === 'viridis', JSON.stringify(maps));

  // Plotly zoom -> WebGL
  await setProp('renderer-mode', {value: 'plotly'});
  await page.waitForFunction(() => {
    const g = document.querySelector('#spatial-plot .js-plotly-plot');
    return g && g.data && g.data.length && g.layout && g.layout.xaxis && g.layout.xaxis.range
      && getComputedStyle(document.querySelector('#spatial-plot')).display !== 'none';
  }, null, {timeout: 180000});
  await page.waitForTimeout(3000);
  const cx = bounds[0] + (bounds[2] - bounds[0]) * 0.6, cy = bounds[1] + (bounds[3] - bounds[1]) * 0.4;
  const want = {x0: cx - 200, x1: cx + 200, y0: cy - 120, y1: cy + 120};
  await page.evaluate((w) => window.Plotly.relayout(document.querySelector('#spatial-plot .js-plotly-plot'),
    {'xaxis.range[0]': w.x0, 'xaxis.range[1]': w.x1, 'yaxis.range[0]': -w.y1, 'yaxis.range[1]': -w.y0}), want);
  await page.waitForTimeout(6000);   // relayoutData -> spatial-relayout store (server round trip)
  await setProp('renderer-mode', {value: 'webgl'});
  await page.waitForTimeout(2500);
  const got = await page.evaluate(() => window.XeniumViewer.instance.getViewport());
  const mid = got && [(got.x0 + got.x1) / 2, (got.y0 + got.y1) / 2];
  check('Plotly zoom is carried into the WebGL view',
    !!got && Math.abs(mid[0] - cx) < 5 && Math.abs(mid[1] - cy) < 5 && (got.x1 - got.x0) < 1200
    && got.x0 <= want.x0 + 1 && got.x1 >= want.x1 - 1 && got.y0 <= want.y0 + 1 && got.y1 >= want.y1 - 1,
    JSON.stringify({want, got}));
  await page.screenshot({path: `${out}/glue_1_zoom.png`});

  // image channels
  await setProp('morph-channels', {value: ['dapi']});
  await setProp('morph-enable', {value: ['show']});
  await page.waitForFunction(() => (window.XeniumViewer.instance.deck.props.layers || []).flat(3)
    .some((l) => l && /^xenium-morphology/.test(l.id)), null, {timeout: 60000});
  await page.waitForTimeout(4000);
  // (a) a channel unticked while its contrast is loading must not be drawn
  await page.route('**/viewer/image/**', async (route) => { await sleep(1500); await route.continue(); });
  await setProp('morph-channels', {value: ['dapi', 'boundary']});
  await page.waitForTimeout(60);
  await setProp('morph-channels', {value: ['dapi']});
  await page.waitForTimeout(9000);
  let il = await imageLayer();
  check('channel unticked while its contrast loads is not drawn',
    il.length === 1 && il[0].vis.filter(Boolean).length === 1, JSON.stringify(il));
  await page.unroute('**/viewer/image/**');
  // (b) channels that were on once are not fetched after being unticked
  await setProp('morph-channels', {value: ['dapi', 'boundary', '18s', 'sma']});
  await page.waitForTimeout(9000);
  il = await imageLayer();
  const allOn = il.length === 1 && il[0].vis.filter(Boolean).length === 4;
  await setProp('morph-channels', {value: ['dapi']});
  await page.waitForTimeout(2500);
  let n = reqs.length;
  await page.evaluate(([x, y]) => window.XeniumViewer.instance.setViewport({x0: x + 900, x1: x + 1300, y0: y + 500, y1: y + 740}), [cx, cy]);
  await page.waitForTimeout(7000);
  const perFile = [0, 1, 2, 3].map((i) => reqs.slice(n).filter((p) => p.startsWith(`/viewer/image/${i}`)).length);
  il = await imageLayer();
  check('only the ticked channel is fetched after a pan', allOn && perFile[0] > 0 && perFile[1] + perFile[2] + perFile[3] === 0,
    JSON.stringify({allOn, perFile, il}));
  await page.screenshot({path: `${out}/glue_2_image.png`});
  // (c) a dataset-version bump with the same image files keeps the image controller
  n = reqs.length;
  await page.evaluate(() => {
    // what poll_reseg_umap returns after "Rerun UMAP": dataset-version + 1, same bundle
    window.__dv = (window.__dv || 1000) + 1;
    window.dash_clientside.set_props('dataset-version', {data: window.__dv});
  });
  await page.waitForTimeout(8000);
  const imgReq = reqs.slice(n).filter((p) => p.startsWith('/viewer/image/')).length;
  il = await imageLayer();
  check('dataset-version bump with unchanged image files keeps the image layer', imgReq === 0 && il.length === 1,
    JSON.stringify({imgReq, il}));
  check('no image error in the status line', !/unavailable|failed/i.test(await status()), JSON.stringify(await status()));
  await setProp('morph-enable', {value: []});
  await page.waitForTimeout(500);

  // lasso: selection highlight and pending outline stay while the Save ROI dialog is open
  const lx = bounds[0] + (bounds[2] - bounds[0]) * 0.72, ly = bounds[1] + (bounds[3] - bounds[1]) * 0.45;   // dense tissue
  await page.evaluate(([x, y]) => window.XeniumViewer.instance.setViewport({x0: x - 150, x1: x + 150, y0: y - 100, y1: y + 100}), [lx, ly]);
  await page.waitForTimeout(1500);
  const before = await page.evaluate(() => (window.XeniumViewer.instance.getState().overlays.polygons || []).length);
  const box = await page.locator('#webgl-view').boundingBox();
  const px = box.x + box.width / 2, py = box.y + box.height / 2;
  await setProp('webgl-tool', {value: 'lasso'});
  await page.waitForTimeout(800);
  await page.mouse.move(px - 120, py - 80);
  await page.mouse.down();
  for (const [dx, dy] of [[120, -110], [150, 60], [20, 130], [-130, 90], [-120, -80]]) {
    await page.mouse.move(px + dx, py + dy, {steps: 6});
  }
  await page.mouse.up();
  await page.waitForFunction(() => [...document.querySelectorAll('.modal.show')].some((e) => /Save ROI/.test(e.innerText)), null, {timeout: 30000});
  await page.waitForTimeout(1500);
  const during = await page.evaluate(() => {
    const s = window.XeniumViewer.instance.getState();
    return {selected: s.selected ? s.selected.length : 0, polygons: (s.overlays.polygons || []).length, tool: s.tool,
            preview: document.querySelector('#roi-save-preview')?.innerText};
  });
  check('lasso highlight and pending outline are shown behind the Save ROI dialog',
    during.selected > 0 && during.polygons === before + 1 && during.tool === 'pan', JSON.stringify({before, during}));
  await page.screenshot({path: `${out}/glue_3_pending_roi.png`});
  // an unrelated control change must not clear it
  await setProp('point-size', {value: 3});
  await page.waitForTimeout(800);
  const still = await page.evaluate(() => (window.XeniumViewer.instance.getState().selected || []).length);
  check('an unrelated control change keeps the selection', still > 0, String(still));
  await page.click('#roi-save-cancel-btn');
  await page.waitForTimeout(4000);
  const after = await page.evaluate(() => {
    const s = window.XeniumViewer.instance.getState();
    return {selected: s.selected ? s.selected.length : 0, polygons: (s.overlays.polygons || []).length};
  });
  check('cancelling the dialog removes the highlight and the pending outline',
    after.selected === 0 && after.polygons === before, JSON.stringify({before, after}));
  check('page 1: no page errors', logs.length === 0, JSON.stringify(logs.slice(0, 5)));
  await page.close();
}

// ───────────────────────── page 2: status line, bundle without images, context loss ─────────────────────────
{
  const {page, logs, reqs, setProp, status} = await open(async (page) => {
    // pretend the bundle has no morphology files
    await page.route('**/viewer/meta', async (route) => {
      const resp = await route.fetch();
      const meta = await resp.json();
      meta.images = [];
      await route.fulfill({response: resp, json: meta});
    });
  });
  const metaCount = () => reqs.filter((p) => p.startsWith('/viewer/meta')).length;
  // a viewer status that stays: colours fail to load
  await page.route('**/viewer/colors.bin*', (route) => route.fulfill({status: 500, body: 'boom'}));
  await setProp('color-by', {value: 'cell_area'});
  await page.waitForTimeout(2500);
  const s0 = await status();
  check('viewer status is shown', /failed to load colours/.test(s0), JSON.stringify(s0));
  await setProp('morph-channels', {value: ['dapi']});
  await setProp('morph-enable', {value: ['show']});
  await page.waitForTimeout(3000);
  const s1 = await status();
  check('image error is shown next to the viewer status', /no morphology images/.test(s1) && /failed to load colours/.test(s1), JSON.stringify(s1));
  const m0 = metaCount();
  await setProp('point-size', {value: 3});
  await page.waitForTimeout(800);
  await setProp('point-size', {value: 2});
  await page.waitForTimeout(1500);
  check('/viewer/meta is not refetched after a failed image open', metaCount() === m0, `${metaCount() - m0} extra requests`);
  await setProp('morph-enable', {value: []});
  await page.waitForTimeout(1000);
  const s2 = await status();
  check('image error is cleared when the overlay is switched off', !/no morphology images/.test(s2) && /failed to load colours/.test(s2), JSON.stringify(s2));
  await setProp('renderer-mode', {value: 'plotly'});
  await page.waitForTimeout(1500);
  const s3 = await status();
  await setProp('renderer-mode', {value: 'webgl'});
  await page.waitForTimeout(2500);
  const s4 = await status();
  const vs = await page.evaluate(() => window.XeniumViewer.instance.getStatus());
  check('viewer status is restored after a Plotly round trip', s3 === '' && s4 === vs && /failed to load colours/.test(s4), JSON.stringify({s3, s4, vs}));
  await page.unroute('**/viewer/colors.bin*');
  await setProp('color-by', {value: 'transcript_counts'});
  await page.waitForTimeout(3000);

  // WebGL context loss: status, then a remount on the next control change
  await page.evaluate(() => {
    window.__oldViewer = window.XeniumViewer.instance;
    const c = document.querySelector('#webgl-view canvas');
    const g = c.getContext('webgl2');
    g.getExtension('WEBGL_lose_context').loseContext();
  });
  await page.waitForTimeout(1500);
  const s5 = await status();
  check('context loss is reported in the status line', /context lost/i.test(s5), JSON.stringify(s5));
  await setProp('point-size', {value: 3});
  await page.waitForFunction(() => window.XeniumViewer.instance !== window.__oldViewer
    && window.XeniumViewer.instance._viewer._cells, null, {timeout: 60000}).catch(() => {});
  await page.waitForTimeout(3000);
  const re = await page.evaluate(() => {
    const c = document.querySelector('#webgl-view canvas');
    const g = c && c.getContext('webgl2');
    return {remounted: window.XeniumViewer.instance !== window.__oldViewer, lost: g ? g.isContextLost() : null,
            canvases: document.querySelectorAll('#webgl-view canvas').length};
  });
  const png = (await page.locator('#webgl-view').screenshot()).length;
  const s6 = await status();
  check('viewer is remounted after a context loss', re.remounted && re.lost === false && re.canvases === 1 && png > 60000 && !/context lost/i.test(s6),
    JSON.stringify({re, png, s6}));
  await page.screenshot({path: `${out}/glue_4_remount.png`});
  check('page 2: no page errors', logs.length === 0, JSON.stringify(logs.slice(0, 5)));
  await page.close();
}

console.log(failures === 0 ? 'ALL PASS' : `FAILURES: ${failures}`);
await browser.close();
process.exit(failures === 0 ? 0 : 1);
