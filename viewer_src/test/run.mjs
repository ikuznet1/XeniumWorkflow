// End-to-end tests for the viewer bundle in headless chromium (WebGL backend: see GL_MODE in lib.mjs).
//   node test/run.mjs            (ports 8072 = skin, 8073 = tiny)
// Screenshots are written to test/screenshots/. Exit code is non-zero on any failure.
import {pathToFileURL} from 'node:url';
import {resolve} from 'node:path';
import {writeFileSync} from 'node:fs';
import {
  Checks, DATA, GL_MODE, HERE, ROOT, diffFraction, fetchBin, launch, newPage, parseBin, pointInPolygon, shot, startServer, hasBundle,
} from './lib.mjs';

const T = new Checks();
const numbers = {}; // measured values, printed as JSON at the end
// Chromium's software compositor reads the WebGL canvas back every frame under SwiftShader and logs
// this driver message for ANY WebGL page (see test/gl_noise.mjs, which reproduces it without the bundle).
const GPU_NOISE = /GL Driver Message .*GPU stall due to ReadPixels/;
const real = (issues) => issues.filter((m) => !GPU_NOISE.test(m));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const VIEW = {width: 1280, height: 800};

const idle = (page) => page.evaluate(() => window.viewer.whenIdle());
const ready = async (page) => {
  await page.waitForFunction(() => window.viewer && window.viewer.stats.firstRenderMs !== undefined, null, {timeout: 120000});
  await idle(page);
};
const setState = async (page, partial) => {
  await page.evaluate((p) => window.viewer.setState(p), partial);
  await idle(page);
};
const setViewport = async (page, vp) => {
  await page.evaluate((v) => window.viewer.setViewport(v), vp);
  await sleep(200); // let the 150 ms settle debounce fire
  await idle(page);
};
const viewerRequests = (requests, name) => requests.filter((u) => u.includes(`/viewer/${name}`));
const eventsOf = (page, kind) => page.evaluate((k) => window.events.filter((e) => e.kind === k).map((e) => e.payload), kind);

function nearestBrute(xy, n, x, y) {
  let best = -1, bestD = Infinity;
  for (let i = 0; i < n; i++) {
    const dx = xy[2 * i] - x, dy = xy[2 * i + 1] - y;
    const d = dx * dx + dy * dy;
    if (d < bestD) { bestD = d; best = i; }
  }
  return {index: best, dist: Math.sqrt(bestD)};
}

async function drawPath(page, pts) {
  await page.mouse.move(pts[0][0], pts[0][1]);
  await page.mouse.down();
  for (let i = 1; i < pts.length; i++) await page.mouse.move(pts[i][0], pts[i][1]);
  await page.mouse.up();
}

for (const dir of [DATA.skin, DATA.tiny]) {
  if (!hasBundle(dir)) {
    console.log(`SKIP  viewer tests: no Xenium bundle at ${dir} (set XV_SKIN / XV_TINY or XV_TEST_DATA)`);
    process.exit(0);
  }
}
const skin = await startServer(DATA.skin, 8072);
const tiny = await startServer(DATA.tiny, 8073);
const browser = await launch();

try {
  // ── 0. bundle is inert when loaded without mount() ─────────────────────────
  {
    const {page, consoleIssues, requests, close} = await newPage(browser, VIEW);
    await page.setContent('<!doctype html><html><body><div id="other">hello</div></body></html>');
    await page.addScriptTag({path: resolve(ROOT, 'assets', 'xenium_viewer.js')});
    await sleep(500);
    const info = await page.evaluate(() => ({
      api: typeof window.XeniumViewer.mount,
      canvases: document.querySelectorAll('canvas').length,
      bodyChildren: [...document.body.children].map((c) => c.tagName).join(','),
      styles: document.querySelectorAll('style').length,
    }));
    T.ok(info.api === 'function' && info.canvases === 0 && info.styles === 0 && info.bodyChildren === 'DIV',
      '0 bundle load without mount is inert', JSON.stringify(info));
    T.ok(consoleIssues.length === 0 && requests.length === 0, '0 inert load: no console output, no requests',
      JSON.stringify({consoleIssues, requests}));
    await close();
  }

  const cells = await fetchBin(`${skin.url}/viewer/cells.bin?source=xenium`);
  const N = cells.header.n;
  const XY = cells.arrays.xy;
  const B = cells.header.bounds;

  // Densest 300 µm block of the tissue, used for the zoomed-in tests.
  let focus;
  {
    const bin = 300, nx = Math.ceil((B[2] - B[0]) / bin), ny = Math.ceil((B[3] - B[1]) / bin);
    const h = new Uint32Array(nx * ny);
    for (let i = 0; i < N; i++) h[Math.floor((XY[2 * i + 1] - B[1]) / bin) * nx + Math.floor((XY[2 * i] - B[0]) / bin)]++;
    let best = 0;
    for (let k = 1; k < h.length; k++) if (h[k] > h[best]) best = k;
    const cx = B[0] + ((best % nx) + 0.5) * bin, cy = B[1] + (Math.floor(best / nx) + 0.5) * bin;
    const hh = (150 * VIEW.height) / VIEW.width;
    focus = {x0: cx - 150, y0: cy - hh, x1: cx + 150, y1: cy + hh, count: h[best]};
  }

  // ── 0b. a session without any screenshot must produce no console output at all ──
  {
    const {page, consoleIssues, close} = await newPage(browser, VIEW);
    await page.goto(skin.url + '/');
    await ready(page);
    await setState(page, {color: {mode: 'cluster'}, boundaries: {cell: true}, transcripts: {genes: ['POSTN']}});
    await setViewport(page, focus);
    await setState(page, {tool: 'lasso'});
    await drawPath(page, [[500, 300], [700, 300], [700, 500], [500, 500]]);
    await idle(page);
    T.ok(real(consoleIssues).length === 0, '0b no console errors/warnings from the bundle (interactive session)', JSON.stringify(real(consoleIssues)));
    await close();
  }

  const {page, consoleIssues, requests, close} = await newPage(browser, VIEW);
  const responses = [];
  page.on('response', (r) => { if (r.url().includes('/viewer/')) responses.push(r); });
  await page.goto(skin.url + '/');
  await ready(page);

  // ── 1. cells render ────────────────────────────────────────────────────────
  {
    const s1 = await shot(page, '01_cells_transcript_counts.png');
    const st = await page.evaluate(() => ({stats: window.viewer.stats, vp: window.viewer.getViewport(),
      layer: window.viewer.deck.props.layers.map((l) => [l.id, l.props.data.length])}));
    numbers.skin = {n: N, firstRenderMs: st.stats.firstRenderMs, cellsFetchMs: st.stats.cells.fetchMs,
      gridIndexMs: st.stats.cells.indexMs, recolorMs: st.stats.recolorMs, drawnFraction: s1.drawnFraction};
    T.ok(st.layer.length === 1 && st.layer[0][0] === 'xv-cells' && st.layer[0][1] === N, '1 cells layer holds every cell', JSON.stringify(st.layer));
    T.ok(s1.drawnFraction > 0.1 && s1.drawnFraction < 0.8 && s1.distinctColors > 200, '1 screenshot is not blank',
      `drawn=${s1.drawnFraction.toFixed(3)} distinctColors=${s1.distinctColors}`);
    // Tissue shape: rasterise cells.bin through the reported viewport and compare with drawn pixels.
    const vp = st.vp;
    T.ok(vp.x0 <= B[0] && vp.y0 <= B[1] && vp.x1 >= B[2] && vp.y1 >= B[3], '1 initial view fits data bounds', JSON.stringify(vp));
    numbers.skin.shapeAgreement = await shapeAgreement(page, s1.path, vp, XY, N);
    T.ok(numbers.skin.shapeAgreement.iou > 0.6, '1 drawn pixels match the tissue shape rasterised from cells.bin',
      JSON.stringify(numbers.skin.shapeAgreement));
  }

  // ── 2. recolour refetches colours only; categorical legend toggles ─────────
  {
    const before = {cells: viewerRequests(requests, 'cells.bin').length, colors: viewerRequests(requests, 'colors.bin').length};
    await setState(page, {color: {mode: 'gene', gene: 'EEF1G'}});
    const s2 = await shot(page, '02_gene_EEF1G.png');
    const after = {cells: viewerRequests(requests, 'cells.bin').length, colors: viewerRequests(requests, 'colors.bin').length};
    const d = await diffFraction(page, resolve(HERE, 'screenshots', '01_cells_transcript_counts.png'), s2.path);
    T.ok(before.cells === 1 && after.cells === 1 && after.colors === before.colors + 1,
      '2 recolour fetched colors.bin once and cells.bin not at all', JSON.stringify({before, after}));
    T.ok(d > 0.05, '2 picture changed after recolour', `changed pixels=${(d * 100).toFixed(1)}%`);
    const legend = await page.evaluate(() => ({kind: document.querySelector('.xv-box').dataset.kind,
      title: document.querySelector('.xv-title').textContent, min: document.querySelector('.xv-min').textContent,
      max: document.querySelector('.xv-max').textContent}));
    T.ok(legend.kind === 'continuous' && /EEF1G/.test(legend.title), '2 continuous legend shows label and range', JSON.stringify(legend));
    numbers.skin.recolor = await page.evaluate(() => ({fetchMs: window.viewer.stats.colorFetchMs,
      computeMs: window.viewer.stats.recolorMs, toFrameMs: window.viewer.stats.recolorRenderMs}));

    // min/max override: no request, picture changes
    const c0 = viewerRequests(requests, 'colors.bin').length;
    await setState(page, {color: {min: 0, max: 1}});
    const s2b = await shot(page, '02b_gene_range_0_1.png');
    const d2 = await diffFraction(page, s2.path, s2b.path);
    T.ok(viewerRequests(requests, 'colors.bin').length === c0 && d2 > 0.02, '2 min/max override recolours without a request',
      `changed pixels=${(d2 * 100).toFixed(1)}%`);
    await setState(page, {color: {min: null, max: null}});

    await setState(page, {color: {mode: 'cluster'}});
    const s2c = await shot(page, '02c_cluster.png');
    const items = await page.evaluate(() => [...document.querySelectorAll('.xv-item[data-category]')].map((e) => e.dataset.category));
    const colors = parseBin(Buffer.from(await (await fetch(`${skin.url}/viewer/colors.bin?source=xenium&mode=cluster`)).arrayBuffer()));
    T.ok(items.length === colors.header.categories.length && items.length > 1, '2 categorical legend lists every category', items.join(','));
    T.ok(viewerRequests(requests, 'cells.bin').length === 1, '2 still a single cells.bin request');
    const target = items[items.length - 1];
    const code = colors.header.categories.indexOf(target);
    let expectHidden = 0;
    for (let i = 0; i < N; i++) if (colors.arrays.codes[i] === code) expectHidden++;
    await page.click(`.xv-item[data-category="${target}"]`);
    await idle(page);
    const s2d = await shot(page, '02d_cluster_hidden.png');
    const hid = await page.evaluate(() => {
      const v = window.viewer._viewer;
      let zero = 0;
      for (let i = 0; i < v._cells.n; i++) if (v._rgba[4 * i + 3] === 0) zero++;
      return {state: window.viewer.getState().hiddenCategories, zero,
        stateEvents: window.events.filter((e) => e.kind === 'state').length,
        off: document.querySelectorAll('.xv-item.xv-off').length};
    });
    const d3 = await diffFraction(page, s2c.path, s2d.path);
    T.ok(hid.state.length === 1 && hid.state[0] === target && hid.stateEvents === 1 && hid.off === 1,
      '2 legend click updates hiddenCategories and notifies the host', JSON.stringify(hid));
    T.ok(hid.zero === expectHidden && expectHidden > 0, '2 hidden category cells get alpha 0', `${hid.zero} cells of "${target}" (expected ${expectHidden})`);
    T.ok(d3 > 0.01, '2 picture changed after hiding a category', `changed pixels=${(d3 * 100).toFixed(1)}%`);
    // hidden cells are not pickable
    const hiddenIdx = colors.arrays.codes.findIndex((c) => c === code);
    const picked = await page.evaluate((i) => {
      const v = window.viewer._viewer;
      return v._pick(v._toScreen(v._cells.xy[2 * i], v._cells.xy[2 * i + 1]));
    }, hiddenIdx);
    T.ok(picked !== hiddenIdx, '2 hidden cells are not pickable', `cell ${hiddenIdx} -> pick ${picked}`);
    await page.click(`.xv-item[data-category="${target}"]`);
    await idle(page);
    T.ok((await page.evaluate(() => window.viewer.getState().hiddenCategories.length)) === 0, '2 second click shows the category again');
    // double click isolates the category (Plotly legend behaviour); double click again shows all
    await page.dblclick(`.xv-item[data-category="${target}"]`);
    await idle(page);
    const iso = await page.evaluate(() => {
      const v = window.viewer._viewer;
      let shown = 0;
      for (let i = 0; i < v._cells.n; i++) if (v._rgba[4 * i + 3] !== 0) shown++;
      return {hidden: window.viewer.getState().hiddenCategories, shown, off: document.querySelectorAll('.xv-item.xv-off').length};
    });
    await shot(page, '02e_cluster_isolated.png');
    T.ok(iso.hidden.length === items.length - 1 && !iso.hidden.includes(target) && iso.off === items.length - 1 && iso.shown === expectHidden,
      '2 double click on a legend entry isolates that category', `${iso.shown} cells shown, ${iso.off} entries struck out`);
    await page.dblclick(`.xv-item[data-category="${target}"]`);
    await idle(page);
    T.ok((await page.evaluate(() => window.viewer.getState().hiddenCategories.length)) === 0, '2 double click on the isolated entry shows all again');
    // extra colormaps
    await setState(page, {color: {mode: 'transcript_counts'}});
    const maps = {};
    for (const name of ['magma', 'cividis', 'no-such-map', 'viridis']) {
      await setState(page, {colormap: name});
      maps[name] = await page.evaluate(() => {
        const v = window.viewer._viewer;
        let lo = 0;
        for (let i = 1; i < v._cells.n; i++) if (v._colors.values[i] < v._colors.values[lo]) lo = i;
        return {low: Array.from(v._rgba.slice(4 * lo, 4 * lo + 3)).join(','), bar: document.querySelector('.xv-bar').style.background.slice(0, 60)};
      });
    }
    T.ok(maps.magma.low === '0,0,4' && maps.cividis.low === '0,34,78' && maps.viridis.low === '68,1,84' && maps['no-such-map'].low === '68,1,84' &&
      /rgb\(0, 0, 4\)/.test(maps.magma.bar) && /rgb\(0, 34, 78\)/.test(maps.cividis.bar),
      '2 colormap accepts magma and cividis (unknown names fall back to viridis)', JSON.stringify(maps));
    await setState(page, {color: {mode: 'cluster'}});
  }

  // ── 3. boundaries ──────────────────────────────────────────────────────────
  {
    await page.evaluate(() => window.viewer.fitView());
    await sleep(200);
    await idle(page);
    await setState(page, {boundaries: {cell: true}});
    const out = await page.evaluate(() => ({status: window.viewer.getStatus(),
      layers: window.viewer.deck.props.layers.map((l) => l.id), info: window.viewer._viewer._loaders.cell.info,
      statusEvents: window.events.filter((e) => e.kind === 'status').map((e) => e.payload)}));
    T.ok(/zoom in to see cell boundaries/.test(out.status) && out.statusEvents.some((s) => /zoom in/.test(s)) &&
      out.info.truncated === true && !out.layers.includes('xv-boundary-cell'),
      '3 zoomed-out view reports truncation via onStatus and draws no outlines', JSON.stringify({status: out.status, layers: out.layers, n: out.info.n}));
    await shot(page, '03a_boundaries_truncated.png');

    await setState(page, {opacity: 0.6});
    await setViewport(page, focus);
    const s3 = await shot(page, '03b_boundaries_300um.png');
    const b = await page.evaluate(() => {
      const v = window.viewer._viewer;
      const d = v._loaders.cell.data;
      const {start, xy, cell} = d.arrays;
      // distance between each polygon's vertex centroid and the centroid of the cell it belongs to
      const dist = [];
      for (let p = 0; p < d.header.n; p++) {
        let sx = 0, sy = 0;
        for (let k = start[p]; k < start[p + 1]; k++) { sx += xy[2 * k]; sy += xy[2 * k + 1]; }
        const m = start[p + 1] - start[p];
        dist.push(Math.hypot(sx / m - v._cells.xy[2 * cell[p]], sy / m - v._cells.xy[2 * cell[p] + 1]));
      }
      dist.sort((a, c) => a - c);
      const layer = window.viewer.deck.props.layers.find((l) => l.id === 'xv-boundary-cell');
      return {n: d.header.n, median: dist[dist.length >> 1], p95: dist[Math.floor(dist.length * 0.95)], max: dist[dist.length - 1],
        layerLength: layer && layer.props.data.length, pathType: layer && layer.props._pathType,
        binary: layer && layer.props.data.attributes.getPath.value instanceof Float32Array,
        status: window.viewer.getStatus(), box: v._loaders.cell.box, vp: window.viewer.getViewport()};
    });
    numbers.boundaries = {polygons: b.n, centroidOffsetMedianUm: b.median, p95Um: b.p95, maxUm: b.max, pathType: b.pathType};
    T.ok(b.n > 50 && b.layerLength === b.n && b.binary && b.status === '', '3 boundaries drawn at ~300 µm from binary arrays',
      `${b.n} polygons, status="${b.status}"`);
    T.ok(b.median < 1.5 && b.p95 < 4, '3 each outline is centred on its own cell (data alignment)',
      `median ${b.median.toFixed(2)} µm, p95 ${b.p95.toFixed(2)} µm`);
    T.ok(b.box[0] < b.vp.x0 && b.box[2] > b.vp.x1, '3 fetched region is padded beyond the viewport', JSON.stringify(b.box.map(Math.round)));
    // outline pixels: near-white pixels exist only when the layer is on
    const white = await countColor(page, s3.path, [2, 185, 223], 40); // #00d4ff at alpha 220 over the background
    await setState(page, {boundaries: {cell: false}});
    const s3off = await shot(page, '03c_boundaries_off.png');
    const whiteOff = await countColor(page, s3off.path, [2, 185, 223], 40);
    T.ok(white > 5000 && whiteOff < white / 20, '3 outline pixels present only with boundaries on', `outline-coloured px on=${white} off=${whiteOff}`);

    // a small pan inside the padded region must not refetch; a large one must
    await setState(page, {boundaries: {cell: true, nucleus: true}});
    const r0 = viewerRequests(requests, 'boundaries.bin').length;
    await setViewport(page, {x0: focus.x0 + 20, y0: focus.y0 + 10, x1: focus.x1 + 20, y1: focus.y1 + 10});
    const r1 = viewerRequests(requests, 'boundaries.bin').length;
    await setViewport(page, {x0: focus.x0 + 400, y0: focus.y0, x1: focus.x1 + 400, y1: focus.y1});
    const r2 = viewerRequests(requests, 'boundaries.bin').length;
    T.ok(r1 === r0 && r2 === r0 + 2, '3 small pan reuses the padded region, large pan refetches (cell + nucleus)', JSON.stringify({r0, r1, r2}));
    await setViewport(page, focus);
    await shot(page, '03d_cell_and_nucleus.png');

    // stale responses: delay the first request, issue a second; the first must be aborted
    let delayed = 0;
    const failed = [];
    page.on('requestfailed', (r) => failed.push(r.url()));
    await page.route('**/viewer/boundaries.bin*', async (route) => {
      if (delayed++ < 2) await sleep(1500);
      try { await route.continue(); } catch {}
    });
    const vpA = {x0: focus.x0 - 900, y0: focus.y0, x1: focus.x1 - 900, y1: focus.y1};
    const vpB = {x0: focus.x0 + 900, y0: focus.y0, x1: focus.x1 + 900, y1: focus.y1};
    await page.evaluate((v) => window.viewer.setViewport(v), vpA);
    await sleep(400);
    await page.evaluate((v) => window.viewer.setViewport(v), vpB);
    await sleep(2500);
    await idle(page);
    await page.unroute('**/viewer/boundaries.bin*');
    const box = await page.evaluate(() => window.viewer._viewer._loaders.cell.box);
    T.ok(failed.filter((u) => u.includes('boundaries.bin')).length >= 2 && box[0] < vpB.x0 && box[2] > vpB.x1,
      '3 superseded boundary requests are aborted; final data belongs to the latest viewport',
      `aborted=${failed.length} box=${JSON.stringify(box.map(Math.round))}`);
    await setState(page, {boundaries: {cell: false, nucleus: false}});

    // every kind has its own loader, request and status: a Baysor / Proseg set can be asked
    // for next to the Xenium ones whatever the source is
    const k0 = viewerRequests(requests, 'boundaries.bin').length;
    await setState(page, {boundaries: {cell: true, proseg: true, baysor: true}});
    const kinds = viewerRequests(requests, 'boundaries.bin').slice(k0).map((u) => new URL(u).searchParams.get('kind')).sort();
    const kst = await page.evaluate(() => ({status: window.viewer.getStatus(), layers: window.viewer.deck.props.layers.map((l) => l.id),
      colors: ['cell', 'nucleus', 'proseg', 'baysor'].map((k) => (window.viewer.deck.props.layers.find((l) => l.id === `xv-boundary-${k}`) || {props: {}}).props.getColor)}));
    T.ok(kinds.join() === 'baysor,cell,proseg' && /Proseg boundaries not available/.test(kst.status) && /Baysor boundaries not available/.test(kst.status) &&
      kst.layers.includes('xv-boundary-cell') && !kst.layers.includes('xv-boundary-proseg') && String(kst.colors[0]) === '0,212,255,220',
      '3 boundary kinds are independent: one request and one status per ticked set', JSON.stringify({kinds, status: kst.status, cell: kst.colors[0]}));
    await setState(page, {boundaries: {cell: false, proseg: false, baysor: false}});
    T.ok(await page.evaluate(() => window.viewer.getStatus()) === '', '3 statuses clear when the sets are switched off');
  }

  // ── 4. transcripts ─────────────────────────────────────────────────────────
  {
    await setViewport(page, focus);
    const genes = ['POSTN', 'LDHA', 'EEF1G'];
    const mark = responses.length;
    await setState(page, {transcripts: {genes}});
    const resp = responses.slice(mark).filter((r) => r.url().includes('transcripts.bin')).pop();
    const wire = parseBin(Buffer.from(await resp.body()));
    const s4 = await shot(page, '04_transcripts_300um.png');
    const tx = await page.evaluate(() => {
      const layer = window.viewer.deck.props.layers.find((l) => l.id === 'xv-transcripts');
      const d = window.viewer._viewer._loaders.tx.data;
      const perGene = {};
      for (let i = 0; i < d.header.n; i++) perGene[d.arrays.gene[i]] = (perGene[d.arrays.gene[i]] || 0) + 1;
      return {layerLength: layer && layer.props.data.length, header: d.header, perGene,
        legend: [...document.querySelectorAll('.xv-box[data-kind="transcripts"] .xv-name')].map((e) => e.textContent),
        title: document.querySelector('.xv-box[data-kind="transcripts"] .xv-title').textContent,
        status: window.viewer.getStatus()};
    });
    numbers.transcripts = {headerN: wire.header.n, layerLength: tx.layerLength, level: wire.header.level,
      subsampled: wire.header.subsampled, perGene: tx.perGene};
    T.ok(wire.header.n > 100 && tx.layerLength === wire.header.n && wire.arrays.xy.length === 2 * wire.header.n,
      '4 transcript point count equals the response header', `header n=${wire.header.n}, layer length=${tx.layerLength}`);
    T.ok(tx.legend.join(',') === genes.join(',') && Object.keys(tx.perGene).length === 3, '4 transcript legend lists the genes',
      `${tx.title}: ${JSON.stringify(tx.perGene)}`);
    T.ok(wire.header.subsampled === false && tx.status === '', '4 full-resolution transcripts at 300 µm (not subsampled)', `level=${wire.header.level}`);
    const red = await countColor(page, s4.path, [255, 77, 77], 40);
    T.ok(red > 200, '4 transcript-coloured pixels on screen', `pixels near gene-0 colour: ${red}`);
    // zoomed out: subsampled status
    await page.evaluate(() => window.viewer.fitView());
    await sleep(200);
    await idle(page);
    const st = await page.evaluate(() => ({status: window.viewer.getStatus(), header: window.viewer._viewer._loaders.tx.info}));
    T.ok(/subsampled/.test(st.status) && st.header.subsampled === true, '4 zoomed-out transcripts report subsampling via onStatus', st.status);
    await shot(page, '04b_transcripts_zoomed_out.png');
    await setState(page, {transcripts: {genes: []}});
    T.ok((await page.evaluate(() => window.viewer.deck.props.layers.map((l) => l.id).join(','))) === 'xv-cells', '4 clearing genes removes the layer');
  }

  // ── 5. lasso + rectangle selection ─────────────────────────────────────────
  {
    await setState(page, {tool: 'lasso', color: {mode: 'transcript_counts'}});
    const vpBefore = await page.evaluate(() => window.viewer.getViewport());
    // irregular star-shaped lasso (concave), 72 vertices, in the right half of the page
    const pts = [];
    for (let k = 0; k < 72; k++) {
      const a = (k / 72) * 2 * Math.PI;
      const r = 150 + 90 * Math.sin(5 * a);
      pts.push([820 + r * Math.cos(a) * 1.3, 430 + r * Math.sin(a)]);
    }
    await drawPath(page, pts);
    await idle(page);
    const sel = await page.evaluate(() => {
      const e = window.events.filter((e) => e.kind === 'select').pop().payload;
      return {indices: Array.from(e.indices), polygon: e.polygon, isU32: e.indices instanceof Uint32Array,
        ms: window.viewer.stats.selectMs, selected: window.viewer.getState().selected.length};
    });
    const vpAfter = await page.evaluate(() => window.viewer.getViewport());
    const brute = [];
    for (let i = 0; i < N; i++) if (pointInPolygon(XY[2 * i], XY[2 * i + 1], sel.polygon)) brute.push(i);
    const same = brute.length === sel.indices.length && brute.every((v, i) => v === sel.indices[i]);
    numbers.lassoSkin = {selected: sel.indices.length, bruteForce: brute.length, vertices: sel.polygon.length, ms: sel.ms};
    T.ok(same && brute.length > 1000 && sel.isU32, '5 lasso indices equal brute-force point-in-polygon over cells.bin',
      `${sel.indices.length} vs ${brute.length} cells, ${sel.polygon.length} vertices, ${sel.ms.toFixed(1)} ms`);
    // polygon is in µm: its first vertex must unproject from the first mouse position
    const exp0 = [vpBefore.x0 + (pts[0][0] / VIEW.width) * (vpBefore.x1 - vpBefore.x0), vpBefore.y0 + (pts[0][1] / VIEW.height) * (vpBefore.y1 - vpBefore.y0)];
    T.ok(Math.hypot(sel.polygon[0][0] - exp0[0], sel.polygon[0][1] - exp0[1]) < 0.01, '5 lasso polygon is reported in µm',
      `first vertex ${sel.polygon[0].map((v) => v.toFixed(2))} expected ${exp0.map((v) => v.toFixed(2))}`);
    T.ok(Math.abs(vpAfter.x0 - vpBefore.x0) < 1e-9 && sel.selected === sel.indices.length, '5 dragging in lasso mode does not pan; selection is highlighted');
    await shot(page, '05a_lasso_selected.png');

    // in-progress shape is drawn while dragging
    await page.mouse.move(700, 250);
    await page.mouse.down();
    await page.mouse.move(800, 300);
    await page.mouse.move(900, 500);
    await page.mouse.move(750, 600);
    const dAttr = await page.evaluate(() => document.querySelector('.xv-draw path').getAttribute('d'));
    await shot(page, '05b_lasso_in_progress.png');
    await page.mouse.up();
    await idle(page);
    T.ok(/^M700\.0 250\.0L800\.0 300\.0L900\.0 500\.0L750\.0 600\.0Z$/.test(dAttr), '5 in-progress lasso shape is drawn', dAttr);
    T.ok((await page.evaluate(() => document.querySelector('.xv-draw path').getAttribute('d'))) === '', '5 shape cleared on release');

    await setState(page, {tool: 'rect'});
    await drawPath(page, [[600, 300], [700, 380], [900, 560]]);
    await idle(page);
    const rect = await page.evaluate(() => {
      const e = window.events.filter((e) => e.kind === 'select').pop().payload;
      return {indices: Array.from(e.indices), polygon: e.polygon};
    });
    const xs = rect.polygon.map((p) => p[0]), ys = rect.polygon.map((p) => p[1]);
    const [rx0, rx1, ry0, ry1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const bruteR = [];
    for (let i = 0; i < N; i++) if (pointInPolygon(XY[2 * i], XY[2 * i + 1], rect.polygon)) bruteR.push(i);
    let inBox = 0;
    for (let i = 0; i < N; i++) if (XY[2 * i] > rx0 && XY[2 * i] < rx1 && XY[2 * i + 1] > ry0 && XY[2 * i + 1] < ry1) inBox++;
    numbers.rectSkin = {selected: rect.indices.length, bruteForce: bruteR.length, strictlyInsideBox: inBox};
    T.ok(rect.polygon.length === 4 && bruteR.length === rect.indices.length && bruteR.every((v, i) => v === rect.indices[i]) &&
      Math.abs(inBox - rect.indices.length) <= 2 && rect.indices.length > 100,
      '5 rectangle selection equals brute force', `${rect.indices.length} vs ${bruteR.length} (box test ${inBox})`);
    await shot(page, '05c_rect_selected.png');
    await setState(page, {tool: 'pan', selected: null});
  }

  // ── 6. click picking, pan, wheel zoom, viewport callbacks ──────────────────
  {
    await setViewport(page, focus);
    const vp = await page.evaluate(() => window.viewer.getViewport());
    const toWorld = (sx, sy) => [vp.x0 + (sx / VIEW.width) * (vp.x1 - vp.x0), vp.y0 + (sy / VIEW.height) * (vp.y1 - vp.y0)];
    const toScreen = (x, y) => [((x - vp.x0) / (vp.x1 - vp.x0)) * VIEW.width, ((y - vp.y0) / (vp.y1 - vp.y0)) * VIEW.height];
    // five cells spread over the right part of the view (clear of the dev panel and the legend)
    const inView = [];
    for (let i = 0; i < N; i++) {
      const [sx, sy] = toScreen(XY[2 * i], XY[2 * i + 1]);
      if (sx > 320 && sx < 1000 && sy > 120 && sy < 760) inView.push(i);
    }
    const results = [];
    for (let k = 0; k < 5; k++) {
      const i = inView[Math.floor(((k + 0.5) / 5) * inView.length)];
      const [sx, sy] = toScreen(XY[2 * i], XY[2 * i + 1]);
      const click = [Math.round(sx) + 1, Math.round(sy) - 1];
      const mark = (await eventsOf(page, 'click')).length;
      await page.mouse.click(click[0], click[1]);
      await sleep(50);
      const got = (await eventsOf(page, 'click'))[mark];
      const exp = nearestBrute(XY, N, ...toWorld(click[0], click[1]));
      results.push({click, got: got && got.index, expected: exp.index, distUm: +exp.dist.toFixed(3),
        xyOk: got && Math.abs(got.x - XY[2 * exp.index]) < 1e-3 && got.source === 'xenium'});
    }
    numbers.picking = results;
    T.ok(results.every((r) => r.got === r.expected && r.xyOk), '6 click returns the nearest cell (5 clicks, brute-force check)',
      results.map((r) => `${r.got}==${r.expected}`).join(' '));
    await sleep(300);
    // responses can arrive out of click order on a busy machine: look for the last click's cell
    const cellJson = (await eventsOf(page, 'cell')).find((c) => c.index === results[4].expected);
    T.ok(cellJson && cellJson.index === results[4].expected, '6 dev page fetched /viewer/cell/<index> for the picked cell', JSON.stringify(cellJson));

    // hover tooltip
    const last = results[4].click;
    await page.mouse.move(last[0] + 1, last[1]);
    await page.mouse.move(last[0], last[1]);
    await sleep(100);
    const tip = await page.evaluate(() => ({text: document.querySelector('.xv-tip').textContent, shown: document.querySelector('.xv-tip').style.display}));
    const hoverId = (await (await fetch(`${skin.url}/viewer/cell/${results[4].expected}?source=xenium`)).json()).id;
    T.ok(tip.shown === 'block' && (tip.text.startsWith(`cell ${results[4].expected}`) || tip.text.startsWith(hoverId)),
      '6 hover tooltip appears at once (row index until the id is known)', tip.text);
    await sleep(700);
    const tip2 = await page.evaluate(() => document.querySelector('.xv-tip').textContent);
    const idReqs = requests.filter((u) => u.includes(`/viewer/cell/${results[4].expected}?`)).length;
    T.ok(tip2.startsWith(`${hoverId} · `) && /: [-\d.e+]+$/.test(tip2), '6 resting on a cell swaps in its id and labels the value', tip2);
    await page.mouse.move(last[0] + 1, last[1]);
    await page.mouse.move(last[0], last[1]);
    await sleep(500);
    T.ok(requests.filter((u) => u.includes(`/viewer/cell/${results[4].expected}?`)).length === idReqs,
      '6 the id of a cell is looked up once and remembered', `${idReqs} request(s) so far (incl. the click lookup of the dev page)`);
    await shot(page, '06_pick_hover_300um.png');

    // background click -> null
    await page.evaluate(() => window.viewer.fitView());
    await sleep(200);
    await idle(page);
    const fit = await page.evaluate(() => window.viewer.getViewport());
    const bg = [640, 780];
    const w = [fit.x0 + (bg[0] / VIEW.width) * (fit.x1 - fit.x0), fit.y0 + (bg[1] / VIEW.height) * (fit.y1 - fit.y0)];
    const nearest = nearestBrute(XY, N, w[0], w[1]);
    const mark = (await eventsOf(page, 'click')).length;
    await page.mouse.click(bg[0], bg[1]);
    await sleep(50);
    const ev = await eventsOf(page, 'click');
    T.ok(ev.length === mark + 1 && ev[mark] === null && nearest.dist > 50, '6 background click reports null', `nearest cell ${nearest.dist.toFixed(0)} µm away`);

    // wheel zoom keeps the point under the cursor fixed
    const cur = [800, 400];
    await page.mouse.move(cur[0], cur[1]);
    const w0 = await page.evaluate((c) => window.viewer.unproject(c[0], c[1]), cur);
    const vpMark = (await eventsOf(page, 'viewport')).length;
    await page.mouse.wheel(0, -400);
    await sleep(400);
    await idle(page);
    const w1 = await page.evaluate((c) => window.viewer.unproject(c[0], c[1]), cur);
    const zoomed = await page.evaluate(() => window.viewer.getViewport());
    const ratio = (fit.x1 - fit.x0) / (zoomed.x1 - zoomed.x0);
    T.ok(ratio > 1.5 && Math.hypot(w1[0] - w0[0], w1[1] - w0[1]) < 0.5, '6 wheel zooms about the cursor',
      `zoom x${ratio.toFixed(2)}, cursor point moved ${Math.hypot(w1[0] - w0[0], w1[1] - w0[1]).toFixed(4)} µm`);
    const vpEvents = (await eventsOf(page, 'viewport')).slice(vpMark);
    T.ok(vpEvents.length >= 1 && vpEvents.length <= 3 && Math.abs(vpEvents[vpEvents.length - 1].x0 - zoomed.x0) < 1e-6,
      '6 onViewport fires debounced with the final viewport in µm', `${vpEvents.length} event(s)`);

    // drag pans
    await page.mouse.move(700, 400);
    await page.mouse.down();
    await page.mouse.move(650, 380, {steps: 4});
    await page.mouse.move(600, 360, {steps: 4});
    await page.mouse.up();
    await sleep(300);
    const panned = await page.evaluate(() => window.viewer.getViewport());
    const scale = VIEW.width / (zoomed.x1 - zoomed.x0);
    const dx = (panned.x0 - zoomed.x0) * scale, dy = (panned.y0 - zoomed.y0) * scale;
    // deck.gl's pan recogniser only starts after a small movement threshold, so the first step (12.5, 5 px) is not applied
    T.ok(dx > 80 && dx <= 101 && dy > 30 && dy <= 41, '6 drag pans the view by the drag distance (minus the recogniser threshold)', `moved ${dx.toFixed(1)}, ${dy.toFixed(1)} px for a 100, 40 px drag`);
    const clicksAfterDrag = (await eventsOf(page, 'click')).length;
    T.ok(clicksAfterDrag === mark + 1, '6 a drag does not count as a click');

    // double click fits; setViewport restores
    await page.mouse.dblclick(700, 500);
    await sleep(300);
    await idle(page);
    const refit = await page.evaluate(() => window.viewer.getViewport());
    T.ok(Math.abs(refit.x0 - fit.x0) < 1e-6 && Math.abs(refit.y1 - fit.y1) < 1e-6, '6 double-click fits the data bounds');
    await setViewport(page, focus);
    const restored = await page.evaluate(() => window.viewer.getViewport());
    T.ok(Math.abs(restored.x0 - focus.x0) < 1e-6 && Math.abs(restored.x1 - focus.x1) < 1e-6 && Math.abs(restored.y0 - focus.y0) < 1e-6,
      '6 setViewport restores an exact viewport', JSON.stringify(restored));

    // overlays + resize
    await setState(page, {overlays: {
      polygons: [{points: [[focus.x0 + 60, focus.y0 + 40], [focus.x0 + 160, focus.y0 + 40], [focus.x0 + 180, focus.y0 + 120], [focus.x0 + 70, focus.y0 + 140]],
        color: '#ff3366', width: 3, fill: '#ff3366', label: 'ROI 1'}],
      lines: [{from: [focus.x0 + 100, focus.y0 + 60], to: [focus.x0 + 260, focus.y0 + 150], color: '#00ffcc', width: 3}],
    }});
    const s6 = await shot(page, '06b_overlays.png');
    const pink = await countColor(page, s6.path, [255, 51, 102], 30);
    const teal = await countColor(page, s6.path, [0, 255, 204], 30);
    const label = await page.evaluate(() => { const l = document.querySelector('.xv-label'); return l && {text: l.textContent, left: l.style.left, top: l.style.top}; });
    T.ok(pink > 500 && teal > 200 && label && label.text === 'ROI 1', '6 overlay polygon, line and label are drawn', `pink=${pink} teal=${teal} label=${JSON.stringify(label)}`);
    await setState(page, {overlays: {polygons: [], lines: []}});

    await page.setViewportSize({width: 1000, height: 600});
    await sleep(400);
    await idle(page);
    const size = await page.evaluate(() => { const c = document.querySelector('canvas.xv-canvas'); return [c.width, c.height, c.clientWidth, c.clientHeight]; });
    const vpR = await page.evaluate(() => window.viewer.getViewport());
    T.ok(size[0] === 1000 && size[1] === 600 && Math.abs((vpR.x0 + vpR.x1) / 2 - (focus.x0 + focus.x1) / 2) < 1e-6,
      '6 canvas tracks its container on resize, centre preserved', JSON.stringify(size));
    await page.setViewportSize(VIEW);
    await sleep(300);
    await idle(page);

    // setExtraLayers: host layers are drawn below the cells and get the view state
    const extra = await page.evaluate(async () => {
      const v = window.viewer;
      const vp = v.getViewport();
      let seen = null;
      v.setExtraLayers((viewState) => {
        seen = viewState;
        return [new window.XeniumViewer.layers.ScatterplotLayer({id: 'host-extra', data: [{p: [(vp.x0 + vp.x1) / 2, (vp.y0 + vp.y1) / 2]}],
          getPosition: (d) => d.p, getRadius: 40, radiusUnits: 'common', getFillColor: [255, 0, 255, 255]})];
      });
      await v.whenIdle();
      return {ids: v.deck.props.layers.map((l) => l.id), seen};
    });
    const s6c = await shot(page, '06c_extra_layer_below_cells.png');
    const magenta = await countColor(page, s6c.path, [255, 0, 255], 30);
    // the disc has radius 40 um at 1280 px / 300 um; cells drawn on top must cover part of it
    const disc = Math.PI * ((40 * VIEW.width) / 300) ** 2;
    T.ok(extra.ids[0] === 'host-extra' && extra.ids[1] === 'xv-cells' && extra.seen && extra.seen.target.length === 3 && Number.isFinite(extra.seen.zoom) &&
      magenta > 0.5 * disc && magenta < 0.97 * disc, '6 setExtraLayers layers are drawn below the cells',
      `layers=${extra.ids.join(',')} magenta px=${magenta} of ${Math.round(disc)} disc px (rest covered by cells)`);
    await page.evaluate(() => window.viewer.setExtraLayers(null));

    // overlay options: opaque fill without outline, [r, g, b] line colour, markers of a fixed pixel size
    await setState(page, {opacity: 0.05, overlays: {
      polygons: [{points: [[focus.x0 + 150, focus.y0 + 30], [focus.x0 + 210, focus.y0 + 30], [focus.x0 + 210, focus.y0 + 90], [focus.x0 + 150, focus.y0 + 90]],
        fill: '#00ff00', fillAlpha: 255, width: 0, color: '#ff0000'}],
      lines: [{from: [focus.x0 + 120, focus.y0 + 150], to: [focus.x0 + 250, focus.y0 + 150], color: [255, 0, 255], alpha: 255, width: 6}],
      points: [{at: [focus.x0 + 250, focus.y0 + 60], color: '#ffff00', size: 30}],
    }});
    const s6e = await shot(page, '06e_overlay_options.png');
    const side = (60 * VIEW.width) / 300;
    const green = await countColor(page, s6e.path, [0, 255, 0], 30);
    const redEdge = await countColor(page, s6e.path, [255, 0, 0], 30);
    const mag = await countColor(page, s6e.path, [255, 0, 255], 30);
    const yellow = await countColor(page, s6e.path, [255, 255, 0], 40);
    T.ok(green > 0.95 * side * side && redEdge < 20 && mag > 0.8 * 6 * (130 * VIEW.width) / 300 && yellow > 0.75 * Math.PI * 225 && yellow < 1.25 * Math.PI * 225,
      '6 overlays: fillAlpha, width 0, array line colour and pixel-sized point markers',
      `green=${green} of ${Math.round(side * side)}, outline px=${redEdge}, magenta=${mag}, yellow=${yellow} (disc ${Math.round(Math.PI * 225)})`);

    // snapshot(): 2x the on-screen size, the same frame plus legend and scale bar; the canvas goes back to its size
    await setState(page, {color: {mode: 'cluster'}});
    const snap = await page.evaluate(async () => {
      const v = window.viewer;
      const c = v.snapshot({scale: 2});
      const el = document.querySelector('canvas.xv-canvas');
      const during = [c.width, c.height];
      await v.whenIdle();
      return {during, after: [el.width, el.height], css: [el.clientWidth, el.clientHeight], png: c.toDataURL('image/png').split(',')[1],
        legendRows: document.querySelectorAll('.xv-item[data-category]').length};
    });
    const snapPath = resolve(HERE, 'screenshots', '06f_snapshot.png');
    writeFileSync(snapPath, Buffer.from(snap.png, 'base64'));
    const s6f = await shot(page, '06f_screen.png');
    const px = await page.evaluate(async ({b64}) => {
      const img = new Image(); img.src = `data:image/png;base64,${b64}`; await img.decode();
      const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
      const g = c.getContext('2d', {willReadFrequently: true}); g.drawImage(img, 0, 0);
      const count = (x, y, w, h, test) => { const d = g.getImageData(x, y, w, h).data; let n = 0; for (let i = 0; i < d.length; i += 4) if (test(d[i], d[i + 1], d[i + 2])) n++; return n; };
      const bright = (r, gg, b) => r > 235 && gg > 235 && b > 235;
      return {
        green: count(0, 0, c.width, c.height, (r, gg, b) => r < 30 && gg > 225 && b < 30),
        // scale bar: a white bar and its label near the bottom left
        bar: count(20, c.height - 90, 700, 80, bright),
        // legend box (top right): light text on the dark box
        legendText: count(c.width - 420, 16, 400, 300, (r, gg, b) => r > 200 && gg > 210 && b > 215),
        legendSwatch: count(c.width - 400, 50, 22, 260, (r, gg, b) => Math.max(r, gg, b) - Math.min(r, gg, b) > 60),
      };
    }, {b64: snap.png});
    T.ok(snap.during[0] === 2 * snap.css[0] && snap.during[1] === 2 * snap.css[1] && snap.after[0] === snap.css[0] && snap.after[1] === snap.css[1],
      '6 snapshot is 2x the on-screen size and the canvas returns to its own', JSON.stringify({snapshot: snap.during, after: snap.after, css: snap.css}));
    T.ok(px.green > 3.6 * 0.95 * side * side && px.green < 4.4 * side * side, '6 snapshot holds the WebGL frame at 2x (overlay square has 4x the pixels)',
      `green=${px.green}, on screen ${green}`);
    T.ok(px.bar > 400 && px.legendText > 300 && px.legendSwatch > 200 && snap.legendRows > 1, '6 snapshot includes legend and scale bar',
      JSON.stringify({...px, legendRows: snap.legendRows, file: snapPath, screen: s6f.path}));
    await setState(page, {opacity: 0.6, overlays: {polygons: [], lines: [], points: []}});

    // onClick gets the world position as its second argument, also for a click that hits no cell
    {
      await setViewport(page, {x0: B[0] - 700, x1: B[0] - 400, y0: focus.y0, y1: focus.y0 + 187.5});
      await page.evaluate(() => { window.clicks2 = []; window.viewer._viewer.opts.onClick = (info, at) => window.clicks2.push({info, at}); });
      await page.mouse.click(640, 400);
      await sleep(100);
      const c2 = await page.evaluate(() => ({clicks: window.clicks2, w: window.viewer.unproject(640, 400)}));
      T.ok(c2.clicks.length === 1 && c2.clicks[0].info === null && Math.abs(c2.clicks[0].at.x - c2.w[0]) < 1e-6 && Math.abs(c2.clicks[0].at.y - c2.w[1]) < 1e-6,
        '6 a click off any cell reports its world position', JSON.stringify(c2));
      await setViewport(page, focus);
    }

    // dataVersion bump refetches cells and colours together, keeps the viewport
    {
      const c0 = [viewerRequests(requests, 'cells.bin').length, viewerRequests(requests, 'colors.bin').length];
      const vp0 = await page.evaluate(() => window.viewer.getViewport());
      await setState(page, {dataVersion: 1});
      const c1 = [viewerRequests(requests, 'cells.bin').length, viewerRequests(requests, 'colors.bin').length];
      const vp1 = await page.evaluate(() => window.viewer.getViewport());
      T.ok(c1[0] === c0[0] + 1 && c1[1] === c0[1] + 1 && Math.abs(vp1.x0 - vp0.x0) < 1e-9,
        '6 dataVersion bump refetches cells.bin + colors.bin once each and keeps the viewport', JSON.stringify({c0, c1}));
    }

    // destroy
    const gone = await page.evaluate(() => { window.viewer.destroy(); return document.querySelectorAll('.xv-wrap, canvas').length; });
    T.ok(gone === 0, '6 destroy() removes the viewer DOM');
  }
  T.ok(real(consoleIssues).length === 0, 'skin session: no console errors/warnings from the bundle',
    JSON.stringify({real: real(consoleIssues), screenshotReadbackWarnings: consoleIssues.length - real(consoleIssues).length}));
  await close();

  // ── 8. tiny dataset ────────────────────────────────────────────────────────
  {
    const {page, consoleIssues, close} = await newPage(browser, VIEW);
    await page.goto(tiny.url + '/');
    await ready(page);
    await setState(page, {pointSize: 8});
    const tc = await fetchBin(`${tiny.url}/viewer/cells.bin?source=xenium`);
    const s8 = await shot(page, '08_tiny.png');
    const st = await page.evaluate(() => ({vp: window.viewer.getViewport(), n: window.viewer.deck.props.layers[0].props.data.length}));
    const tb = tc.header.bounds;
    const fracX = (tb[2] - tb[0]) / (st.vp.x1 - st.vp.x0), fracY = (tb[3] - tb[1]) / (st.vp.y1 - st.vp.y0);
    numbers.tiny = {n: st.n, bounds: tb, viewport: st.vp, boundsFractionOfView: [fracX, fracY], drawnFraction: s8.drawnFraction};
    T.ok(st.n === tc.header.n && tc.header.n === 23, '8 tiny dataset: all 23 cells in the layer');
    T.ok(st.vp.x0 <= tb[0] && st.vp.x1 >= tb[2] && st.vp.y0 <= tb[1] && st.vp.y1 >= tb[3] && Math.max(fracX, fracY) > 0.9,
      '8 tiny dataset fits its bounds', `bounds cover ${(fracX * 100).toFixed(0)}% x ${(fracY * 100).toFixed(0)}% of the view`);
    // every cell must be visible as a dot: check a non-background pixel at each projected centroid
    const dots = await page.evaluate(async ({b64, pts}) => {
      const img = new Image(); img.src = `data:image/png;base64,${b64}`; await img.decode();
      const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
      const g = c.getContext('2d'); g.drawImage(img, 0, 0);
      const all = g.getImageData(0, 0, c.width, c.height).data;
      return pts.filter(([x, y]) => { const i = 4 * (Math.round(y) * c.width + Math.round(x)); return Math.abs(all[i] - 13) + Math.abs(all[i + 1] - 17) + Math.abs(all[i + 2] - 23) > 24; }).length;
    }, {b64: (await import('node:fs')).readFileSync(s8.path).toString('base64'),
      pts: Array.from({length: 23}, (_, i) => [((tc.arrays.xy[2 * i] - st.vp.x0) / (st.vp.x1 - st.vp.x0)) * VIEW.width,
        ((tc.arrays.xy[2 * i + 1] - st.vp.y0) / (st.vp.y1 - st.vp.y0)) * VIEW.height])});
    numbers.tiny.dotsFound = dots;
    T.ok(dots >= 21, '8 a dot is drawn at the projected position of the cells', `${dots}/23 (remainder sit under the dev panel/legend)`);
    await setState(page, {boundaries: {cell: true, nucleus: true}, transcripts: {genes: ['Epcam', 'Lyz1', 'Muc2']}});
    await shot(page, '08b_tiny_boundaries.png');
    const tb2 = await page.evaluate(() => ({cell: window.viewer._viewer._loaders.cell.info, tx: window.viewer._viewer._loaders.tx.info, status: window.viewer.getStatus()}));
    numbers.tiny.boundaries = tb2;
    T.ok(tb2.cell && tb2.cell.n === 23 && !tb2.cell.truncated, '8 tiny dataset boundaries load at the fitted view', JSON.stringify(tb2));
    T.ok(real(consoleIssues).length === 0, '8 tiny session: no console errors/warnings', JSON.stringify(real(consoleIssues)));
    await close();
  }

  // ── 7. one million synthetic points ────────────────────────────────────────
  {
    const {page, consoleIssues, close} = await newPage(browser, VIEW);
    await page.goto(pathToFileURL(resolve(HERE, 'synth.html')).href);
    await page.waitForFunction(() => window.viewer && window.viewer.stats.firstRenderMs !== undefined, null, {timeout: 300000});
    await idle(page);
    const first = await page.evaluate(() => ({n: window.synth.N, genMs: window.synth.genMs, ...window.viewer.stats}));
    const s7 = await shot(page, '07a_synth_1M.png');
    // recolour (continuous -> continuous, then categorical), three times each
    const recolor = [];
    for (const mode of ['x', 'cat', 'rand', 'x', 'cat']) {
      const r = await page.evaluate(async (mode) => {
        const v = window.viewer;
        const t0 = performance.now();
        v.setState({color: {mode}});
        await new Promise((res) => {
          const prev = v.stats.recolorRenderMs;
          v._viewer._afterRender.push(() => 0);
          const poll = () => (v._viewer._colors && v._viewer._colors.key.includes(`|${mode}|`) && v._viewer._afterRender.length === 0 ? res() : setTimeout(poll, 1));
          void prev;
          poll();
        });
        return {mode, totalToFrameMs: performance.now() - t0, computeMs: v.stats.recolorMs, fakeFetchAndParseMs: v.stats.colorFetchMs};
      }, mode);
      recolor.push(r);
    }
    await idle(page);
    const s7b = await shot(page, '07b_synth_1M_categorical.png');
    // Frame cost while panning 1M points: (a) rAF-driven pan, frame interval as the browser delivers it
    // (headless chromium caps at 60 Hz); (b) redraw + 1-pixel readPixels, which blocks until the GPU
    // has really finished the frame (upper bound, includes the readback round trip).
    const frames = await page.evaluate(async () => {
      const v = window.viewer._viewer;
      const base = v.viewState.target.slice();
      const stamps = [];
      await new Promise((res) => {
        let i = 0;
        const step = (t) => {
          stamps.push(t);
          v._setViewState({...v.viewState, target: [base[0] + 300 * Math.sin(i / 10), base[1] + 200 * Math.cos(i / 13), 0]});
          if (++i < 120) requestAnimationFrame(step); else res();
        };
        requestAnimationFrame(step);
      });
      const iv = [];
      for (let i = 1; i < stamps.length; i++) iv.push(stamps[i] - stamps[i - 1]);
      iv.sort((a, b) => a - b);
      const gl = v.deck.canvas.getContext('webgl2');
      const px = new Uint8Array(4);
      const sync = [];
      for (let i = 0; i < 10; i++) {
        v._setViewState({...v.viewState, target: [base[0] + 37 * i, base[1], 0]});
        const t = performance.now();
        v.deck.redraw('test');
        gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
        sync.push(performance.now() - t);
      }
      sync.sort((a, b) => a - b);
      v._setViewState({...v.viewState, target: base});
      const e = gl.getExtension('WEBGL_debug_renderer_info');
      return {rafIntervalMedianMs: iv[iv.length >> 1], rafIntervalP95Ms: iv[Math.floor(iv.length * 0.95)], rafIntervalMaxMs: iv[iv.length - 1],
        redrawPlusReadbackMedianMs: sync[5], redrawPlusReadbackMaxMs: sync[9], canvas: [v.deck.canvas.width, v.deck.canvas.height],
        renderer: e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : 'unknown'};
    });
    // JS-side cost of one pan step (view state update + layer diff, excluding rasterisation)
    const panJs = await page.evaluate(() => {
      const v = window.viewer._viewer;
      const out = [];
      for (let i = 0; i < 20; i++) {
        const t = performance.now();
        v._setViewState({...v.viewState, target: [v.viewState.target[0] + 5, v.viewState.target[1], 0]});
        out.push(performance.now() - t);
      }
      return out.reduce((a, b) => a + b, 0) / out.length;
    });
    // lasso over a large part of the data
    await page.evaluate(() => window.viewer.setState({tool: 'lasso'}));
    await idle(page);
    const pts = [];
    for (let k = 0; k < 120; k++) {
      const a = (k / 120) * 2 * Math.PI;
      const r = 230 + 110 * Math.sin(7 * a);
      pts.push([640 + r * Math.cos(a) * 1.5, 400 + r * Math.sin(a)]);
    }
    await page.mouse.move(pts[0][0], pts[0][1]);
    await page.mouse.down();
    for (let i = 1; i < pts.length; i++) await page.mouse.move(pts[i][0], pts[i][1]);
    const tUp = await page.evaluate(() => performance.now());
    await page.mouse.up();
    await page.waitForFunction(() => window.events.some((e) => e.kind === 'select'), null, {timeout: 60000});
    const lasso = await page.evaluate((tUp) => {
      const e = window.events.find((e) => e.kind === 'select');
      const poly = e.payload.polygon;
      const {xy, N} = window.synth;
      const t0 = performance.now();
      let brute = 0, mismatch = 0, j = 0;
      const idx = e.payload.indices;
      for (let i = 0; i < N; i++) {
        const x = xy[2 * i], y = xy[2 * i + 1];
        let inside = false;
        for (let a = 0, b = poly.length - 1; a < poly.length; b = a++) {
          const xi = poly[a][0], yi = poly[a][1], xj = poly[b][0], yj = poly[b][1];
          if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
        }
        if (inside) { brute++; if (idx[j] !== i) mismatch++; j++; }
      }
      return {selected: idx.length, brute, mismatch, vertices: poly.length, selectMs: window.viewer.stats.selectMs,
        bruteMs: performance.now() - t0, mouseupToCallbackMs: e.t - tUp, recolorAfterSelectMs: window.viewer.stats.recolorMs};
    }, tUp);
    await idle(page);
    await shot(page, '07c_synth_1M_lasso.png');
    const pick = await page.evaluate(() => {
      const v = window.viewer._viewer;
      const t = performance.now();
      let hits = 0;
      for (let i = 0; i < 1000; i++) if (v._pick([200 + (i % 40) * 20, 100 + Math.floor(i / 40) * 20]) >= 0) hits++;
      return {per: (performance.now() - t) / 1000, hits};
    });
    numbers.synth1M = {
      n: first.n, firstRenderMsFromMount: first.firstRenderMs, fakeFetchParseMs: first.cells.fetchMs, gridIndexMs: first.cells.indexMs,
      firstColorComputeMs: first.recolorMs, recolor, panFrames: frames, panStepJsMs: panJs, lasso, pickMsPerLookup: pick.per,
      drawnFraction: s7.drawnFraction,
    };
    T.ok(first.n === 1000000 && s7.drawnFraction > 0.1, '7 1M points render', `first render ${first.firstRenderMs.toFixed(0)} ms after mount, drawn=${s7.drawnFraction.toFixed(2)}`);
    T.ok(s7.hash !== s7b.hash, '7 1M recolour changes the picture');
    T.ok(Math.max(...recolor.map((r) => r.computeMs)) < 100, '7 1M recolour compute < 100 ms', recolor.map((r) => `${r.mode}:${r.computeMs.toFixed(1)}ms`).join(' '));
    T.ok(lasso.selected === lasso.brute && lasso.mismatch === 0 && lasso.selected > 100000 && lasso.selectMs < 500,
      '7 1M lasso correct and well under a second', `${lasso.selected} cells in ${lasso.selectMs.toFixed(1)} ms (brute force ${lasso.bruteMs.toFixed(0)} ms)`);
    T.ok(real(consoleIssues).length === 0, '7 synthetic session: no console errors/warnings', JSON.stringify(real(consoleIssues)));
    await close();
  }

  // ── 9. mock backend: load ordering, failures, robustness ───────────────────
  {
    const {page, consoleIssues, close} = await newPage(browser, VIEW);
    await page.goto(pathToFileURL(resolve(HERE, 'synth.html')).href + '?n=1000');
    await page.waitForFunction(() => window.viewer && window.viewer.stats.firstRenderMs !== undefined);
    await page.evaluate(() => {
      window.viewer.destroy();
      document.getElementById('viewer').remove();
      const {packBin} = window.XeniumViewer.util;
      const wait = (ms) => new Promise((r) => setTimeout(r, ms));
      window.wait = wait;
      // cfg: {n, token, offset, delay: {name: ms}, fail: {name: count}, colorN, cats, palette, badSource}
      window.mk = (cfg) => {
        const log = [];
        const events = [];
        const fetch = async (url, init) => {
          const u = new URL(url, 'http://x');
          const name = u.pathname.includes('/cell/') ? 'cell' : u.pathname.split('/').pop();
          log.push(name + u.search);
          const d = cfg.delay && cfg.delay[name];
          if (d) await wait(d);
          if (init && init.signal && init.signal.aborted) throw new DOMException('aborted', 'AbortError');
          if (cfg.fail && cfg.fail[name] > 0) { cfg.fail[name]--; return new Response('boom', {status: 500}); }
          if (cfg.badSource && u.searchParams.get('source') === cfg.badSource && name === 'cells.bin') return new Response('boom', {status: 500});
          const n = cfg.n;
          if (name === 'cells.bin') {
            const xy = new Float32Array(2 * n);
            const off = cfg.offset || 0;
            for (let i = 0; i < n; i++) { xy[2 * i] = off + (i % 40) * 10; xy[2 * i + 1] = off + Math.floor(i / 40) * 10; }
            return new Response(packBin({n, token: cfg.token || '1:0:xenium:' + n, bounds: [off, off, off + 390, off + Math.floor((n - 1) / 40) * 10]}, {xy}));
          }
          if (name === 'colors.bin') {
            const mode = u.searchParams.get('mode');
            const m = cfg.colorN === undefined ? n : cfg.colorN;
            if (mode.startsWith('cat')) {
              const k = cfg.cats || 3;
              const codes = new Uint16Array(m);
              for (let i = 0; i < m; i++) codes[i] = i % k;
              return new Response(packBin({kind: 'categorical', label: mode, categories: Array.from({length: k}, (_, i) => 'c' + i),
                colors: cfg.palette || Array.from({length: k}, (_, i) => '#' + (0x102030 * (i + 1) & 0xffffff).toString(16).padStart(6, '0'))}, {codes}));
            }
            const values = new Float32Array(m);
            for (let i = 0; i < m; i++) values[i] = i;
            return new Response(packBin({kind: 'continuous', label: mode, min: 0, max: m - 1}, {values}));
          }
          if (name === 'transcripts.bin') {
            const x0 = +u.searchParams.get('x0'), y0 = +u.searchParams.get('y0');
            return new Response(packBin({n: 2, truncated: false, subsampled: false},
              {xy: new Float32Array([x0 + 1, y0 + 1, x0 + 2, y0 + 2]), gene: new Uint16Array([0, 0])}));
          }
          if (name === 'cell') return new Response(JSON.stringify({id: 'id_' + u.pathname.split('/').pop()}));
          return new Response(packBin({n: 0, available: false, truncated: false}, {}));
        };
        const host = document.createElement('div');
        host.style.cssText = 'position:absolute;inset:0';
        document.body.appendChild(host);
        const h = window.XeniumViewer.mount(host, {
          fetch,
          onStatus: (t) => events.push(['status', t]),
          onSelect: (sel) => events.push(['select', sel.indices.length, sel.polygon.length]),
          onClick: (info) => events.push(['click', info]),
        });
        return {h, v: h._viewer, log, events, cfg, count: (name) => log.filter((l) => l.startsWith(name)).length,
          done: () => { h.destroy(); host.remove(); }};
      };
      window.rgb = (m, i) => Array.from(m.v._rgba.slice(4 * i, 4 * i + 3)).join(',');
    });
    const run = (fn, arg) => page.evaluate(fn, arg);

    // a slow cells response must not overwrite colours chosen while it was in flight
    const order = await run(async () => {
      const out = {};
      let m = window.mk({n: 1000, delay: {'cells.bin': 400, 'colors.bin': 20}});
      m.h.setState({color: {mode: 'A'}});
      await window.wait(50);
      m.h.setState({color: {mode: 'B'}});
      await window.wait(700);
      await m.h.whenIdle();
      out.case1 = {state: m.h.getState().color.mode, shown: m.v._colors && m.v._colors.header.label, status: m.h.getStatus(),
        legend: m.v.legend.querySelector('.xv-title').textContent, last: window.rgb(m, 999), colorReqs: m.count('colors.bin')};
      m.done();
      m = window.mk({n: 1000, delay: {'cells.bin': 400, 'colors.bin': 20}});
      await window.wait(50);
      m.h.setState({color: {mode: 'B'}});
      await window.wait(700);
      await m.h.whenIdle();
      out.case2 = {shown: m.v._colors && m.v._colors.header.label, last: window.rgb(m, 999), colorReqs: m.count('colors.bin')};
      // colours slower than cells: the colour-only request in flight is not asked for twice
      m.done();
      m = window.mk({n: 1000, delay: {'cells.bin': 200, 'colors.bin': 500}});
      await window.wait(50);
      m.h.setState({color: {mode: 'B'}});
      await window.wait(900);
      await m.h.whenIdle();
      out.case3 = {shown: m.v._colors && m.v._colors.header.label, colorReqs: m.count('colors.bin')};
      m.done();
      return out;
    });
    T.ok(order.case1.shown === 'B' && order.case1.legend === 'B' && order.case1.status === '' && order.case1.last === '253,231,37',
      '9 colour changed during a cells load: the newer colours win', JSON.stringify(order.case1));
    T.ok(order.case2.shown === 'B' && order.case2.last === '253,231,37' && order.case2.colorReqs === 1,
      '9 colour set during a colourless cells load is kept', JSON.stringify(order.case2));
    T.ok(order.case3.shown === 'B' && order.case3.colorReqs === 1, '9 a colour request still in flight at the cells commit is not repeated', JSON.stringify(order.case3));

    // failures: cells retried, colours failure does not block cells, errors are not erased
    const fail = await run(async () => {
      const out = {};
      let m = window.mk({n: 1000, fail: {'cells.bin': 1}});
      m.h.setState({color: {mode: 'A'}});
      await window.wait(300);
      out.first = {status: m.h.getStatus(), cells: !!m.v._cells, reqs: m.count('cells.bin')};
      m.h.setState({color: {mode: 'B'}});   // state change while failed: asks for cells again, not only colours
      await window.wait(300);
      out.afterChange = {status: m.h.getStatus(), n: m.v._cells && m.v._cells.n, reqs: m.count('cells.bin'), shown: m.v._colors && m.v._colors.header.label};
      m.done();
      m = window.mk({n: 1000, fail: {'cells.bin': 1}});
      await window.wait(300);
      const t0 = performance.now();
      await m.h.whenIdle();   // waits for the automatic retry
      out.auto = {status: m.h.getStatus(), n: m.v._cells && m.v._cells.n, reqs: m.count('cells.bin'), waitedMs: Math.round(performance.now() - t0)};
      m.done();
      m = window.mk({n: 1000, fail: {'cells.bin': 99}});
      await window.wait(300);
      await m.h.whenIdle();
      out.givesUp = {status: m.h.getStatus(), reqs: m.count('cells.bin')};
      m.done();
      m = window.mk({n: 1000, fail: {'colors.bin': 1}});
      m.h.setState({color: {mode: 'A'}});
      await window.wait(300);
      out.colFail = {status: m.h.getStatus(), n: m.v._cells && m.v._cells.n, layers: m.h.deck.props.layers.map((l) => l.id).join(',')};
      await m.h.whenIdle();   // waits for the automatic retry: no state change is needed
      out.colRetry = {status: m.h.getStatus(), shown: m.v._colors && m.v._colors.header.label, reqs: m.count('colors.bin')};
      m.done();
      // a colours-only request (the cells are already there) that fails is retried as well
      const cfgCol = {n: 1000, fail: {}};
      m = window.mk(cfgCol);
      m.h.setState({color: {mode: 'A'}});
      await window.wait(300);
      await m.h.whenIdle();
      cfgCol.fail['colors.bin'] = 1;
      m.h.setState({color: {mode: 'B'}});
      await window.wait(300);
      out.colOnlyFail = {status: m.h.getStatus(), shown: m.v._colors && m.v._colors.header.label};
      await m.h.whenIdle();
      out.colOnlyRetry = {status: m.h.getStatus(), shown: m.v._colors && m.v._colors.header.label};
      m.done();
      m = window.mk({n: 1000, fail: {'colors.bin': 99}});
      m.h.setState({color: {mode: 'A'}});
      await window.wait(300);
      await m.h.whenIdle();
      out.colGivesUp = {status: m.h.getStatus(), reqs: m.count('colors.bin'), n: m.v._cells && m.v._cells.n};
      m.h.setState({pointSize: 4});   // after that, any later state change asks again
      await window.wait(300);
      out.colGivesUp.reqsAfterChange = m.count('colors.bin');
      m.done();
      // render errors stay visible through later loads
      m = window.mk({n: 1000});
      m.h.setState({color: {mode: 'A'}});
      await window.wait(300);
      await m.h.whenIdle();
      m.h.deck.props.onError(new Error('Failed to create WebGL context'));
      m.h.setState({color: {mode: 'B'}});
      await window.wait(300);
      await m.h.whenIdle();
      out.render = {status: m.h.getStatus(), shown: m.v._colors.header.label};
      m.done();
      // failed source switch: the old table is dropped, nothing of it is pickable
      m = window.mk({n: 1000, badSource: 'baysor:x'});
      m.h.setState({color: {mode: 'cat'}});
      await window.wait(300);
      await m.h.whenIdle();
      m.v._click(m.v._toScreen(m.v._cells.xy[2 * 900], m.v._cells.xy[2 * 900 + 1]));
      m.h.setState({source: 'baysor:x', dataVersion: 2});
      await window.wait(300);
      m.h.setState({color: {mode: 'cat2'}});
      await window.wait(300);
      m.v._click([640, 400]);
      out.sw = {status: m.h.getStatus(), cells: !!m.v._cells, colors: !!m.v._colors, legend: m.v.legend.children.length,
        layers: m.h.deck.props.layers.map((l) => l.id).join(','), clicks: m.events.filter((e) => e[0] === 'click').map((e) => e[1] && [e[1].index, e[1].source])};
      m.h.setState({source: 'xenium', dataVersion: 3});
      await window.wait(300);
      await m.h.whenIdle();
      out.back = {status: m.h.getStatus(), n: m.v._cells && m.v._cells.n, shown: m.v._colors && m.v._colors.header.label};
      m.done();
      return out;
    });
    T.ok(/failed to load cells: HTTP 500/.test(fail.first.status) && !fail.first.cells && fail.afterChange.n === 1000 && fail.afterChange.reqs === 2 &&
      fail.afterChange.status === '' && fail.afterChange.shown === 'B', '9 a failed cells load is asked for again on the next state change', JSON.stringify([fail.first, fail.afterChange]));
    T.ok(fail.auto.n === 1000 && fail.auto.reqs === 2 && fail.auto.status === '', '9 a failed cells load is retried automatically', JSON.stringify(fail.auto));
    T.ok(fail.givesUp.reqs === 3 && /failed to load cells/.test(fail.givesUp.status), '9 automatic retries are bounded (2), the error stays', JSON.stringify(fail.givesUp));
    T.ok(fail.colFail.n === 1000 && fail.colFail.layers === 'xv-cells' && /^failed to load colours/.test(fail.colFail.status) &&
      fail.colRetry.status === '' && fail.colRetry.shown === 'A' && fail.colRetry.reqs === 2,
      '9 a failing colours request does not block the cells and is retried automatically', JSON.stringify([fail.colFail, fail.colRetry]));
    T.ok(/^failed to load colours/.test(fail.colOnlyFail.status) && fail.colOnlyRetry.status === '' && fail.colOnlyRetry.shown === 'B',
      '9 a failed colours-only request (Active Counts, Color By) is retried automatically', JSON.stringify([fail.colOnlyFail, fail.colOnlyRetry]));
    T.ok(fail.colGivesUp.reqs === 3 && /^failed to load colours/.test(fail.colGivesUp.status) && fail.colGivesUp.n === 1000 &&
      fail.colGivesUp.reqsAfterChange === 4, '9 automatic colour retries are bounded (2); the next state change asks again', JSON.stringify(fail.colGivesUp));
    T.ok(/^render error: Failed to create WebGL context$/.test(fail.render.status) && fail.render.shown === 'B',
      '9 a render error is not cleared by later successful loads', JSON.stringify(fail.render));
    T.ok(!fail.sw.cells && !fail.sw.colors && fail.sw.legend === 0 && fail.sw.layers === '' && /failed to load cells/.test(fail.sw.status) &&
      JSON.stringify(fail.sw.clicks) === '[[900,"xenium"],null]' && fail.back.n === 1000 && fail.back.status === '' && fail.back.shown === 'cat2',
      '9 failed source switch drops the old table (no stale picks); switching back recovers', JSON.stringify([fail.sw, fail.back]));

    // colours: rgb() palette entries, wrong-length arrays, inverted range
    const col = await run(async () => {
      const out = {};
      let m = window.mk({n: 1000, cats: 2, palette: ['#ff0000', 'rgb(102, 197, 204)']});
      m.h.setState({color: {mode: 'cat'}});
      await window.wait(200);
      await m.h.whenIdle();
      out.rgbFn = [window.rgb(m, 0), window.rgb(m, 1)];
      m.done();
      m = window.mk({n: 1000, colorN: 900});
      m.h.setState({color: {mode: 'cat'}});
      await window.wait(300);
      await m.h.whenIdle();
      out.stale = {status: m.h.getStatus(), legend: m.v.legend.children.length, cellsReqs: m.count('cells.bin'), c0: window.rgb(m, 0), tip: m.v._tipText(5)};
      m.cfg.colorN = undefined;   // the server is consistent again
      m.h.setState({color: {mode: 'cat2'}});
      await window.wait(300);
      await m.h.whenIdle();
      out.fresh = {status: m.h.getStatus(), legend: m.v.legend.children.length, tip: m.v._tipText(5)};
      m.h.setState({color: {mode: 'v', min: 800, max: 100}});
      await window.wait(300);
      await m.h.whenIdle();
      out.inv = {c0: window.rgb(m, 0), c999: window.rgb(m, 999), min: m.v.legend.querySelector('.xv-min').textContent, max: m.v.legend.querySelector('.xv-max').textContent};
      // null for a nested key resets it instead of breaking every later sync
      m.h.setState({color: null, boundaries: null, transcripts: 5, overlays: null});
      m.h.setState({pointSize: 5});
      await window.wait(200);
      await m.h.whenIdle();
      out.nul = {color: m.h.getState().color, legend: m.v.legend.children.length, c0: window.rgb(m, 0), size: m.h.getState().pointSize};
      m.done();
      return out;
    });
    T.ok(col.rgbFn[0] === '255,0,0' && col.rgbFn[1] === '102,197,204', "9 'rgb(r, g, b)' palette entries are drawn in their colour", JSON.stringify(col.rgbFn));
    T.ok(/^colours out of date \(900 values for 1000 cells\)$/.test(col.stale.status) && col.stale.legend === 0 && col.stale.cellsReqs === 2 &&
      col.stale.c0 === '128,128,128' && col.stale.tip === 'cell 5' && col.fresh.status === '' && col.fresh.legend === 1 && col.fresh.tip === 'cell 5 · c2',
      '9 colours of the wrong length: status, one reload, no legend / hover value', JSON.stringify([col.stale, col.fresh]));
    T.ok(col.inv.c0 === '68,1,84' && col.inv.c999 === '253,231,37' && col.inv.min === '100' && col.inv.max === '800',
      '9 min > max override is read as the swapped range', JSON.stringify(col.inv));
    T.ok(col.nul.color.mode === null && col.nul.legend === 0 && col.nul.c0 === '128,128,128' && col.nul.size === 5,
      '9 null for a nested state key resets it; later state changes still apply', JSON.stringify(col.nul));

    // viewport: clamping, non-finite boxes, refit on a new dataset, transcripts follow dataVersion, legend kept
    const view = await run(async () => {
      const out = {};
      const m = window.mk({n: 1000, cats: 300});
      m.h.setState({color: {mode: 'cat'}, transcripts: {genes: ['G1']}});
      await window.wait(400);
      await m.h.whenIdle();
      const fit = m.h.getViewport();
      m.h.setViewport({x0: 0, y0: 0, x1: Infinity, y1: 10});
      m.h.setViewport({x0: NaN, y0: 0, x1: 5, y1: 10});
      out.nonFinite = JSON.stringify(m.h.getViewport()) === JSON.stringify(fit);
      m.h.setViewport({x0: 0, y0: 0, x1: 1e-9, y1: 1e-9});
      out.zoomIn = m.v.viewState.zoom;
      m.h.setViewport({x0: -1e12, y0: -1e12, x1: 1e12, y1: 1e12});
      out.zoomOut = m.v.viewState.zoom;
      m.h.setViewport({x0: 100, y0: 50, x1: 200, y1: 100});
      await window.wait(300);
      await m.h.whenIdle();
      // legend scroll position survives transcript updates and range changes
      const list = m.v.legend.querySelector('.xv-box[data-kind="categorical"] .xv-list');
      list.scrollTop = 200;
      const tx0 = m.count('transcripts.bin');
      m.h.setViewport({x0: 2000, y0: 50, x1: 2100, y1: 100});   // leaves the padded transcript region
      await window.wait(300);
      await m.h.whenIdle();
      const list2 = m.v.legend.querySelector('.xv-box[data-kind="categorical"] .xv-list');
      out.legend = {txRequests: m.count('transcripts.bin') - tx0, sameNode: list2 === list, scrollTop: list2.scrollTop,
        title: m.v.legend.querySelector('.xv-box[data-kind="transcripts"] .xv-title').textContent};
      m.h.setState({transcripts: {size: 3}});
      await m.h.whenIdle();
      out.legend.afterRebuild = m.v.legend.querySelector('.xv-box[data-kind="categorical"] .xv-list').scrollTop;
      m.h.setViewport({x0: 100, y0: 50, x1: 200, y1: 100});
      await window.wait(300);
      await m.h.whenIdle();
      // same dataset, new version: viewport kept, transcripts refetched
      const vp0 = m.h.getViewport();
      const tx1 = m.count('transcripts.bin');
      m.h.setState({dataVersion: 1});
      await window.wait(300);
      await m.h.whenIdle();
      out.sameDataset = {kept: JSON.stringify(m.h.getViewport()) === JSON.stringify(vp0), tx: m.count('transcripts.bin') - tx1, cells: m.count('cells.bin')};
      // another dataset (token generation changes), overlapping coordinates: refit
      m.cfg.token = '2:0:xenium:1000';
      m.cfg.offset = 50;
      m.h.setState({dataVersion: 2});
      await window.wait(300);
      await m.h.whenIdle();
      const vp2 = m.h.getViewport();
      out.newDataset = {vp: vp2, bounds: m.v._cells.bounds, status: m.h.getStatus()};
      m.done();
      // single cell: the fit is clamped to maxZoom, so the first wheel tick does not jump
      const one = window.mk({n: 1});
      await window.wait(300);
      await one.h.whenIdle();
      out.single = one.v.viewState.zoom;
      one.done();
      return out;
    });
    T.ok(view.nonFinite && view.zoomIn === 8 && view.zoomOut === -12 && view.single <= 8,
      '9 setViewport ignores non-finite boxes and clamps the zoom', JSON.stringify({nonFinite: view.nonFinite, zoomIn: view.zoomIn, zoomOut: view.zoomOut, single: view.single}));
    T.ok(view.legend.txRequests === 1 && view.legend.sameNode && view.legend.scrollTop === 200 && view.legend.afterRebuild === 200 && /^Transcripts \(2\)$/.test(view.legend.title),
      '9 transcript updates leave the category list (and its scroll position) alone', JSON.stringify(view.legend));
    T.ok(view.sameDataset.kept && view.sameDataset.tx === 1 && view.sameDataset.cells === 2,
      '9 dataVersion bump on the same dataset keeps the viewport and refetches transcripts', JSON.stringify(view.sameDataset));
    T.ok(view.newDataset.vp.x0 < view.newDataset.bounds[0] && view.newDataset.vp.x1 > view.newDataset.bounds[2] &&
      view.newDataset.vp.y0 < view.newDataset.bounds[1] && view.newDataset.vp.y1 > view.newDataset.bounds[3],
      '9 a new dataset (token generation) refits the view', JSON.stringify(view.newDataset));

    // lasso: pointer cancel, lost release, second finger
    const ptr = await run(async () => {
      const out = {};
      const m = window.mk({n: 1000});
      await window.wait(300);
      await m.h.whenIdle();
      m.h.setState({tool: 'lasso'});
      await m.h.whenIdle();
      const c = m.v.canvas;
      const ev = (target, type, id, x, y, buttons, button = 0) => target.dispatchEvent(
        new PointerEvent(type, {pointerId: id, clientX: x, clientY: y, buttons, button, bubbles: true, cancelable: true}));
      const selects = () => m.events.filter((e) => e[0] === 'select').length;
      // (a) pointercancel ends the shape; later hovering does not extend it
      ev(c, 'pointerdown', 1, 500, 300, 1);
      ev(window, 'pointermove', 1, 700, 300, 1);
      ev(window, 'pointermove', 1, 700, 500, 1);
      ev(window, 'pointercancel', 1, 700, 500, 0);
      ev(window, 'pointermove', 1, 600, 600, 0);
      ev(window, 'pointerup', 1, 600, 600, 0, 2);
      out.cancel = {draw: !!m.v._draw, selects: selects(), path: m.v.drawPath.getAttribute('d')};
      // (b) the release never arrives (context menu): the next button-less move drops the shape
      ev(c, 'pointerdown', 1, 500, 300, 1);
      ev(window, 'pointermove', 1, 700, 300, 1);
      ev(window, 'pointermove', 1, 700, 500, 1);
      ev(window, 'pointermove', 1, 650, 550, 0);
      out.lost = {draw: !!m.v._draw, selects: selects()};
      // (c) a second finger neither restarts nor feeds the shape, and its release does not commit it
      ev(c, 'pointerdown', 1, 500, 300, 1);
      ev(window, 'pointermove', 1, 700, 300, 1);
      ev(c, 'pointerdown', 2, 900, 700, 1);
      ev(window, 'pointermove', 2, 950, 750, 1);
      ev(window, 'pointermove', 1, 700, 500, 1);
      ev(window, 'pointerup', 2, 950, 750, 0);
      out.two = {draw: !!m.v._draw, pts: m.v._draw && m.v._draw.pts.length, selects: selects()};
      ev(window, 'pointermove', 1, 500, 500, 1);
      ev(window, 'pointerup', 1, 500, 500, 0);
      await m.h.whenIdle();
      out.finish = {draw: !!m.v._draw, events: m.events.filter((e) => e[0] === 'select')};
      m.done();
      return out;
    });
    T.ok(!ptr.cancel.draw && ptr.cancel.selects === 0 && ptr.cancel.path === '', '9 pointercancel drops the lasso (no sticky shape, no selection)', JSON.stringify(ptr.cancel));
    T.ok(!ptr.lost.draw && ptr.lost.selects === 0, '9 a lasso whose release was lost ends on the next button-less move', JSON.stringify(ptr.lost));
    T.ok(ptr.two.draw && ptr.two.pts === 3 && ptr.two.selects === 0 && !ptr.finish.draw && ptr.finish.events.length === 1 && ptr.finish.events[0][2] === 4,
      '9 a second pointer does not corrupt or commit the lasso', JSON.stringify([ptr.two, ptr.finish]));

    // overlay labels: one layout read per view change
    const lab = await run(async () => {
      const m = window.mk({n: 1000});
      await window.wait(300);
      await m.h.whenIdle();
      const polygons = [];
      for (let i = 0; i < 1000; i++) {
        const x = (i % 40) * 10, y = Math.floor(i / 40) * 10;
        polygons.push({points: [[x, y], [x + 5, y], [x + 5, y + 5]], label: 'r' + i});
      }
      m.h.setState({overlays: {polygons}});
      await m.h.whenIdle();
      const base = m.v.viewState.target.slice();
      const stamps = [];
      await new Promise((res) => {
        let i = 0;
        const step = (t) => {
          stamps.push(t);
          m.v._setViewState({...m.v.viewState, target: [base[0] + i, base[1], 0]});
          if (++i < 30) requestAnimationFrame(step); else res();
        };
        requestAnimationFrame(step);
      });
      const iv = [];
      for (let i = 1; i < stamps.length; i++) iv.push(stamps[i] - stamps[i - 1]);
      iv.sort((a, b) => a - b);
      const node = m.v._labelItems[41].node;
      const want = m.h.project(m.v._labelItems[41].x, m.v._labelItems[41].y);
      const out = {medianFrameMs: iv[iv.length >> 1], labels: m.v._labelItems.length,
        dx: Math.abs(parseFloat(node.style.left) - want[0]), dy: Math.abs(parseFloat(node.style.top) - (want[1] - 2))};
      m.done();
      return out;
    });
    numbers.labels1000 = lab;
    T.ok(lab.labels === 1000 && lab.medianFrameMs < 100 && lab.dx < 0.06 && lab.dy < 0.06, // CSS keeps 0.1 px
      '9 1000 overlay labels follow the view without layout thrash', `median frame ${lab.medianFrameMs.toFixed(1)} ms (was ~590 ms)`);

    // destroy gives the WebGL context back
    const mark = consoleIssues.length;
    const ctx = await run(async () => {
      let lost = 0;
      for (let i = 0; i < 24; i++) {
        const m = window.mk({n: 100});
        await window.wait(60);
        await m.h.whenIdle();
        const canvas = m.v.canvas;
        const gl = canvas.getContext('webgl2');
        m.done();
        if (gl && gl.isContextLost() && canvas.width === 0) lost++;
      }
      const last = window.mk({n: 100});
      await window.wait(100);
      await last.h.whenIdle();
      const alive = !last.v.canvas.getContext('webgl2').isContextLost() && last.h.deck.props.layers.length === 1;
      last.done();
      return {lost, alive};
    });
    await sleep(300);
    const tooMany = consoleIssues.slice(mark).filter((t) => /Too many active WebGL contexts/.test(t)).length;
    T.ok(ctx.lost === 24 && ctx.alive && tooMany === 0, '9 destroy() releases its WebGL context (24 mount/destroy cycles)', JSON.stringify({...ctx, tooMany}));
    numbers.mockConsole = real(consoleIssues).filter((t) => !/HTTP 500|Failed to create WebGL context|\[XeniumViewer\]/.test(t));
    T.ok(numbers.mockConsole.length === 0, '9 mock session: only the provoked errors were logged', JSON.stringify(numbers.mockConsole));
    await close();
  }
} catch (err) {
  T.ok(false, 'test run crashed', err.stack || String(err));
} finally {
  await browser.close();
  skin.stop();
  tiny.stop();
}

console.log('\nGL backend: ' + GL_MODE);
console.log('\nNUMBERS ' + JSON.stringify(numbers, (k, v) => (typeof v === 'number' && !Number.isInteger(v) ? Math.round(v * 1000) / 1000 : v), 1));
console.log(`\n${T.results.length - T.failed.length}/${T.results.length} checks passed`);
process.exit(T.failed.length ? 1 : 0);

// ── helpers that need a page ───────────────────────────────────────────────────
/** Count pixels within `tol` (sum of abs channel differences) of an RGB colour. */
async function countColor(page, path, rgb, tol) {
  const {readFileSync} = await import('node:fs');
  return page.evaluate(async ({b64, rgb, tol}) => {
    const img = new Image(); img.src = `data:image/png;base64,${b64}`; await img.decode();
    const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
    const g = c.getContext('2d'); g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 0; i < d.length; i += 4) if (Math.abs(d[i] - rgb[0]) + Math.abs(d[i + 1] - rgb[1]) + Math.abs(d[i + 2] - rgb[2]) <= tol) n++;
    return n;
  }, {b64: readFileSync(path).toString('base64'), rgb, tol});
}

/**
 * Compare the screenshot with an occupancy mask rasterised independently from cells.bin
 * (8 px blocks, right of the dev panel and below the legend). Returns intersection-over-union.
 */
async function shapeAgreement(page, path, vp, XY, N) {
  const {readFileSync} = await import('node:fs');
  const block = 8, gw = VIEW.width / block, gh = VIEW.height / block;
  const expect = new Uint8Array(gw * gh);
  for (let i = 0; i < N; i++) {
    const sx = ((XY[2 * i] - vp.x0) / (vp.x1 - vp.x0)) * VIEW.width, sy = ((XY[2 * i + 1] - vp.y0) / (vp.y1 - vp.y0)) * VIEW.height;
    if (sx >= 0 && sx < VIEW.width && sy >= 0 && sy < VIEW.height) expect[Math.floor(sy / block) * gw + Math.floor(sx / block)] = 1;
  }
  const drawn = await page.evaluate(async ({b64, block, gw, gh}) => {
    const img = new Image(); img.src = `data:image/png;base64,${b64}`; await img.decode();
    const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
    const g = c.getContext('2d'); g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height).data;
    const out = new Array(gw * gh).fill(0);
    for (let y = 0; y < c.height; y++) for (let x = 0; x < c.width; x++) {
      const i = 4 * (y * c.width + x);
      if (Math.abs(d[i] - 13) + Math.abs(d[i + 1] - 17) + Math.abs(d[i + 2] - 23) > 24) out[Math.floor(y / block) * gw + Math.floor(x / block)] = 1;
    }
    return out;
  }, {b64: readFileSync(path).toString('base64'), block, gw, gh});
  let inter = 0, union = 0, blocks = 0;
  for (let by = 0; by < gh; by++) for (let bx = 0; bx < gw; bx++) {
    if (bx * block < 270 && by * block < 420) continue; // dev panel
    if (bx * block > 1060 && by * block < 80) continue; // legend
    const a = expect[by * gw + bx], b = drawn[by * gw + bx];
    blocks++;
    if (a && b) inter++;
    if (a || b) union++;
  }
  return {iou: inter / union, occupiedBlocks: union, comparedBlocks: blocks};
}
