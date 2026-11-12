// End-to-end check of the WebGL view inside the real Dash app.
//   XENIUM_RENDERER=webgl PORT=8062 python xenium_explorer.py <bundle>
//   node test/app_smoke.mjs http://127.0.0.1:8062 /path/to/screenshot_dir
// XV_GL=swiftshader|metal picks the WebGL backend (default: metal on macOS).
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
const reqs = [];
page.on('console', (m) => { if (m.type() === 'error') logs.push(`error: ${m.text().slice(0, 300)}`); });
page.on('pageerror', (e) => logs.push('pageerror: ' + e.message.slice(0, 300)));
page.on('requestfinished', async (r) => {
  const u = new URL(r.url());
  if (!u.pathname.startsWith('/viewer/') && !u.pathname.startsWith('/umap/') && u.pathname !== '/_dash-update-component') return;
  let size = 0;
  try { size = (await r.sizes()).responseBodySize; } catch { /* aborted */ }
  reqs.push({path: u.pathname + u.search.slice(0, 60), size, status: (await r.response())?.status()});
});

let failures = 0;
const check = (name, ok, detail = '') => {
  if (!ok) failures += 1;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
};
const setProp = (id, props) => page.evaluate(([i, p]) => window.dash_clientside.set_props(i, p), [id, props]);
const since = (n) => reqs.slice(n);
const viewerSum = (list, part) => list.filter((r) => r.path.includes(part));

const t0 = Date.now();
await page.goto(base, {waitUntil: 'domcontentloaded'});
await page.waitForFunction(() => document.querySelector('#webgl-view canvas'), null, {timeout: 90000});
await page.waitForFunction(() => performance.getEntriesByType('resource').some((e) => e.name.includes('/viewer/colors.bin')), null, {timeout: 90000});
await page.waitForTimeout(3000);
console.log(`first view after ${Date.now() - t0} ms (backend ${gl})`);

// 1. initial render
const canvasStats = async () => {
  const buf = await page.locator('#webgl-view').screenshot();
  return buf.length;   // PNG size grows with drawn detail; a blank canvas compresses to a few kB
};
const layout = await page.evaluate(() => ({
  gl: getComputedStyle(document.querySelector('#webgl-view')).display,
  plotly: getComputedStyle(document.querySelector('#spatial-plot')).display,
  legend: (document.querySelector('.xv-legend')?.innerText || '').slice(0, 40).replace(/\n/g, ' '),
}));
check('WebGL view visible, Plotly hidden', layout.gl === 'block' && layout.plotly === 'none', JSON.stringify(layout));
const png0 = await canvasStats();
check('cells drawn', png0 > 60000, `viewer PNG ${png0} bytes`);
await page.screenshot({path: `${out}/app_1_initial.png`});

// open the UMAP panel so the second viewer is fed
await page.evaluate(() => document.querySelector('#umap-toggle').click());   // lives in the hover-reveal sidebar
await page.waitForTimeout(3000);

// 2. recolour: only colors.bin is fetched by the viewer
let n = reqs.length;
await setProp('color-by', {value: 'transcript_counts'});
await page.waitForTimeout(2500);
let got = since(n);
check('recolour fetches colors.bin once and no cells.bin',
  got.filter((r) => r.path.startsWith('/viewer/colors.bin')).length === 1 && got.filter((r) => r.path.startsWith('/viewer/cells.bin')).length === 0,
  JSON.stringify(viewerSum(got, '/viewer/').map((r) => [r.path.slice(0, 40), r.size])));
const dashMax = Math.max(0, ...got.filter((r) => r.path.startsWith('/_dash')).map((r) => r.size));
check('no Plotly figure is shipped on a recolour in WebGL mode', dashMax < 200000, `largest Dash response ${(dashMax / 1e3).toFixed(1)} kB`);
check('UMAP recolour uses its own colors.bin', viewerSum(got, '/umap/viewer/colors.bin').length === 1);
const umap = await page.evaluate(() => {
  const el = document.querySelector('#webgl-umap');
  const st = window.XeniumViewer.umapInstance && window.XeniumViewer.umapInstance.stats;
  return {display: getComputedStyle(el).display, canvas: !!el.querySelector('canvas'),
          plotlyHidden: getComputedStyle(document.querySelector('#umap-plot')).display === 'none',
          n: window.XeniumViewer.umapInstance.getState().color.mode,
          w: Math.round(el.getBoundingClientRect().width), h: Math.round(el.getBoundingClientRect().height)};
});
check('WebGL UMAP mounted and Plotly UMAP hidden', umap.canvas && umap.display !== 'none' && umap.plotlyHidden && umap.w > 100 && umap.h > 100, JSON.stringify(umap));
const umapPng = (await page.locator('#webgl-umap').screenshot()).length;
check('UMAP cells drawn', umapPng > 30000, `UMAP PNG ${umapPng} bytes`);
await page.screenshot({path: `${out}/app_2_counts.png`});

// 3. zoom to ~300 µm, boundaries + transcripts
const vp = await page.evaluate(async () => {
  const meta = await (await fetch('/viewer/meta')).json();
  return meta.bounds;
});
// zoom to a 300 µm view in dense tissue through the viewer handle
const cx = vp[0] + (vp[2] - vp[0]) * 0.72, cy = vp[1] + (vp[3] - vp[1]) * 0.45;
await page.evaluate(([x, y]) => window.XeniumViewer.instance.setViewport({x0: x - 150, x1: x + 150, y0: y - 100, y1: y + 100}), [cx, cy]);
await page.waitForTimeout(1500);
const box = await page.locator('#webgl-view').boundingBox();
const px = box.x + box.width / 2, py = box.y + box.height / 2;
n = reqs.length;
await setProp('boundary-toggles', {value: ['cell', 'nucleus']});
await page.waitForTimeout(2500);
got = since(n);
const bReq = viewerSum(got, 'boundaries.bin');
check('boundaries fetched for the viewport', bReq.length >= 2 && bReq.every((r) => r.status === 200),
  JSON.stringify(bReq.map((r) => r.size)));
const status1 = await page.evaluate(() => document.querySelector('#webgl-status')?.textContent);
console.log('      status:', JSON.stringify(status1));
await page.screenshot({path: `${out}/app_3_boundaries.png`});

n = reqs.length;
await setProp('webgl-tx-genes', {value: ['EEF1G', 'POSTN', 'LDHA']});
await page.waitForTimeout(3000);
got = since(n);
const tReq = viewerSum(got, 'transcripts.bin');
check('transcripts fetched for the viewport', tReq.length >= 1 && tReq[0].size > 1000, JSON.stringify(tReq.map((r) => r.size)));
await page.screenshot({path: `${out}/app_4_transcripts.png`});
await setProp('webgl-tx-genes', {value: []});

// 4. morphology image
n = reqs.length;
await setProp('morph-channels', {value: ['dapi']});
await setProp('morph-enable', {value: ['show']});
await page.waitForTimeout(9000);
got = since(n);
const iReq = viewerSum(got, '/viewer/image/');
check('image tiles fetched by byte range', iReq.length > 3 && iReq.every((r) => r.status === 206),
  `${iReq.length} requests, ${(iReq.reduce((a, r) => a + r.size, 0) / 1e6).toFixed(2)} MB, statuses ${[...new Set(iReq.map((r) => r.status))]}`);
const status2 = await page.evaluate(() => document.querySelector('#webgl-status')?.textContent);
check('no image error reported', !/unavailable|failed/i.test(status2 || ''), JSON.stringify(status2));
await page.screenshot({path: `${out}/app_5_image.png`});

// 5. click a cell -> cell info panel
const before = await page.evaluate(() => document.querySelector('#cell-info-panel')?.innerText.slice(0, 60));
await setProp('boundary-toggles', {value: []});
await page.waitForTimeout(800);
let clicked = false;
// centre the view on a known cell so the click lands on it
const target = await page.evaluate(async () => (await fetch('/viewer/cell/50000')).json());
await page.evaluate((c) => window.XeniumViewer.instance.setViewport({x0: c.x - 60, x1: c.x + 60, y0: c.y - 40, y1: c.y + 40}), target);
await page.waitForTimeout(1200);
const cbox = await page.locator('#webgl-view').boundingBox();
const hit = await page.evaluate((c) => window.XeniumViewer.instance.project(c.x, c.y), target);
for (const [dx, dy] of [[0, 0], [2, 2], [-2, 1]]) {
  await page.mouse.click(cbox.x + hit[0] + dx, cbox.y + hit[1] + dy);
  await page.waitForTimeout(1500);
  const txt = await page.evaluate(() => document.querySelector('#cell-info-panel')?.innerText.slice(0, 200));
  if (txt && txt !== before && txt.includes(target.id)) { clicked = true; console.log('      panel:', JSON.stringify(txt.replace(/\n/g, ' | ').slice(0, 110))); break; }
}
check('clicking a cell fills the cell-info panel', clicked);
// hover: the tooltip starts with the row index and shows the cell id once the pointer rests
await page.mouse.move(cbox.x + hit[0] + 1, cbox.y + hit[1]);
await page.mouse.move(cbox.x + hit[0], cbox.y + hit[1]);
await page.waitForTimeout(1500);
const tipText = await page.evaluate(() => { const t = document.querySelector('#webgl-view .xv-tip'); return t && t.style.display === 'block' ? t.textContent : null; });
check('hover tooltip shows the cell id', !!tipText && tipText.startsWith(target.id), JSON.stringify(tipText));
await page.screenshot({path: `${out}/app_5b_hover.png`});

// 5b2. continuous colour scale follows the Plotly one for the colour mode (COLORSCALES)
await setProp('color-by', {value: 'cell_area'});
await page.waitForTimeout(1500);
const cmap = await page.evaluate(() => [window.XeniumViewer.instance.getState().colormap,
  window.XeniumViewer.umapInstance && window.XeniumViewer.umapInstance.getState().colormap]);
check('cell_area is drawn with the plasma scale in both viewers', cmap[0] === 'plasma' && (cmap[1] === undefined || cmap[1] === 'plasma'), JSON.stringify(cmap));

// 5c. categorical legend: double click isolates one category, double click again shows all
await setProp('color-by', {value: 'cluster'});
await page.waitForTimeout(2500);
const legendItems = await page.locator('#webgl-view .xv-item[data-category]').count();
if (legendItems > 1) {
  await page.locator('#webgl-view .xv-item[data-category]').first().dblclick();
  await page.waitForTimeout(800);
  const iso = await page.evaluate(() => ({hidden: window.XeniumViewer.instance.getState().hiddenCategories.length,
    off: document.querySelectorAll('#webgl-view .xv-item.xv-off').length}));
  await page.screenshot({path: `${out}/app_5c_isolated.png`});
  await page.locator('#webgl-view .xv-item[data-category]').first().dblclick();
  await page.waitForTimeout(800);
  const all = await page.evaluate(() => window.XeniumViewer.instance.getState().hiddenCategories.length);
  check('legend double click isolates a category and restores all', iso.hidden === legendItems - 1 && iso.off === legendItems - 1 && all === 0,
    JSON.stringify({legendItems, iso, all}));
} else {
  console.log(`      (legend isolate skipped: ${legendItems} categories)`);
}

// 5d. ruler: two clicks measure a distance and select no cell
{
  const panelBefore = await page.evaluate(() => document.querySelector('#cell-info-panel')?.innerText);
  await page.click('#measure-btn');
  await page.waitForTimeout(800);
  n = reqs.length;
  const rbox = await page.locator('#webgl-view').boundingBox();
  const ends = await page.evaluate(() => [window.XeniumViewer.instance.unproject(300, 250), window.XeniumViewer.instance.unproject(700, 400)]);
  await page.mouse.click(rbox.x + 300, rbox.y + 250);
  await page.waitForTimeout(1200);
  await page.mouse.click(rbox.x + 700, rbox.y + 400);
  await page.waitForTimeout(1800);
  const ruler = await page.evaluate(() => { const o = window.XeniumViewer.instance.getState().overlays; const d = document.querySelector('#measure-dist-display');
    return {lines: o.lines.length, points: (o.points || []).length, text: d.textContent, shown: getComputedStyle(d).display !== 'none'}; });
  const free = Math.hypot(ends[1][0] - ends[0][0], ends[1][1] - ends[0][1]);   // end points snap to a cell under the click
  check('ruler draws its segment and shows the distance', ruler.lines === 1 && ruler.points === 2 && ruler.shown && Math.abs(parseFloat(ruler.text) - free) < 25,
    JSON.stringify({...ruler, clickDistance: free.toFixed(1)}));
  check('ruler clicks select no cell', since(n).filter((r) => r.path.startsWith('/viewer/cell/')).length === 0 &&
    (await page.evaluate(() => document.querySelector('#cell-info-panel')?.innerText)) === panelBefore);
  await page.screenshot({path: `${out}/app_5d_ruler.png`});
  await page.click('#measure-btn');
  await page.waitForTimeout(800);
  check('ruler off removes the drawing', await page.evaluate(() => { const o = window.XeniumViewer.instance.getState().overlays; return o.lines.length === 0 && (o.points || []).length === 0; }));
}

// 5e. Xenium boundary checkboxes map to their own sets; the Proseg one asks for the Proseg set
await setProp('boundary-toggles', {value: ['cell', 'proseg']});
await page.waitForTimeout(2000);
const kindState = await page.evaluate(() => ({b: window.XeniumViewer.instance.getState().boundaries, status: document.querySelector('#webgl-status').textContent}));
check('boundary checkboxes drive one set each', kindState.b.cell && kindState.b.proseg && !kindState.b.nucleus && !kindState.b.baysor && /Proseg boundaries not available/.test(kindState.status), JSON.stringify(kindState));
await setProp('boundary-toggles', {value: []});

// 5f. PNG export button and the Z-Level control
{
  const page2 = page;
  const [dl] = await Promise.all([page2.waitForEvent('download', {timeout: 20000}), page2.click('#webgl-png-btn')]);
  const file = `${out}/app_5f_export.png`;
  await dl.saveAs(file);
  const png = (await import('node:fs')).readFileSync(file);
  const vb = await page.locator('#webgl-view').boundingBox();
  const dims = [png.readUInt32BE(16), png.readUInt32BE(20)];
  check('PNG button downloads the view at 2x', dl.suggestedFilename() === 'xenium_spatial.png' && dims[0] === 2 * Math.round(vb.width) && dims[1] === 2 * Math.round(vb.height) && png.length > 50000,
    JSON.stringify({dims, bytes: png.length}));
  const z = await page.evaluate(() => ({disabled: document.querySelector('#morph-zlevel').disabled, title: document.querySelector('#morph-zlevel-wrap').title}));
  check('Z-Level is disabled with an explanation in WebGL mode', z.disabled === true && /WebGL/.test(z.title), JSON.stringify(z));
}

// 6. lasso -> ROI save modal
await setProp('webgl-tool', {value: 'lasso'});
await page.waitForTimeout(600);
await page.mouse.move(px - 120, py - 80);
await page.mouse.down();
for (const [dx, dy] of [[120, -110], [150, 60], [20, 130], [-130, 90], [-120, -80]]) {
  await page.mouse.move(px + dx, py + dy, {steps: 6});
}
await page.mouse.up();
await page.waitForTimeout(2000);
const modal = await page.evaluate(() => {
  const m = [...document.querySelectorAll('.modal.show')].find((e) => /Save ROI/.test(e.innerText));
  return {open: !!m,
          preview: document.querySelector('#roi-save-preview')?.innerText,
          err: document.querySelector('#roi-save-error')?.innerText,
          tool: document.querySelector('#webgl-tool input:checked')?.value};
});
check('lasso opens the ROI save dialog with a cell count', modal.open && /cells selected/.test(modal.preview || ''), JSON.stringify(modal));
// the drawn outline and the highlight of the selected cells stay behind the dialog
const pend = await page.evaluate(() => {
  const st = window.XeniumViewer.instance.getState();
  return {selected: (st.selected || []).length, pending: (st.overlays.polygons || []).filter((p) => !p.label).length};
});
check('pending ROI outline and selection are shown while the dialog is open', pend.selected > 0 && pend.pending === 1, JSON.stringify(pend));
await page.screenshot({path: `${out}/app_6_lasso.png`});

// 7. save the ROI and confirm the viewer receives it as an overlay
await page.fill('#roi-name-input', 'webgl_test_roi');
await page.click('#roi-save-btn');
await page.waitForTimeout(3000);
const ov = await page.evaluate(async () => (await (await fetch('/viewer/overlays')).json()).polygons.map((p) => [p.label, p.points.length]));
const drawn = await page.evaluate(() => (window.XeniumViewer.instance.getState().overlays.polygons || []).length);
check('saved ROI is served and drawn as an overlay', ov.some((o) => /webgl_test_roi/.test(o[0])) && drawn >= 1, JSON.stringify({ov, drawn}));
await page.screenshot({path: `${out}/app_7_roi.png`});

check('no page errors', logs.length === 0, JSON.stringify(logs.slice(0, 6)));
console.log(failures === 0 ? 'ALL PASS' : `FAILURES: ${failures}`);
await browser.close();
process.exit(failures === 0 ? 0 : 1);
