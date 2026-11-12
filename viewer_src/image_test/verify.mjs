// End-to-end verification of the morphology-image layer in headless chromium.
//
//   python image_test/serve.py <skin bundle> 8085 &
//   python image_test/serve.py <tiny bundle> 8086 &
//   node image_test/verify.mjs [--skin http://127.0.0.1:8085 --tiny http://127.0.0.1:8086]
//   python image_test/reference.py tiles image_test/out/tiles     # pixel comparison with tifffile
//
// Writes screenshots, out/tiles/* (raw uint16 tiles + manifest) and out/results.json.
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const out = join(here, "out");
mkdirSync(join(out, "tiles"), { recursive: true });
const arg = (name, def) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : def;
};
const SKIN = arg("skin", "http://127.0.0.1:8085");
const TINY = arg("tiny", "http://127.0.0.1:8086");
const SKIN_DIR = arg("skin-dir", "/Users/ikuz/Documents/XeniumTestData/prime_human_skin_ffpe");
const TINY_DIR = arg("tiny-dir", "/Users/ikuz/Documents/XeniumTestData/prime_mouse_ileum_tiny");
const VIEW = { width: 1200, height: 800 };
const SPOT = [6800, 1700]; // µm, dense tissue in the skin dataset

const results = { launchArgs: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] };
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : null; };
const r1 = (v) => (v === null || v === undefined ? v : Math.round(v * 10) / 10);

const browser = await chromium.launch({ args: results.launchArgs });

async function open(url, { throttle } = {}) {
  const context = await browser.newContext({ viewport: VIEW });
  const page = await context.newPage();
  const net = { requests: [], failed: [], console: [] };
  page.on("console", (m) => {
    if (m.type() === "error") net.console.push(m.text().slice(0, 500));
  });
  page.on("pageerror", (e) => net.console.push(`pageerror: ${String(e).slice(0, 500)}`));
  page.on("requestfinished", async (req) => {
    if (!req.url().includes("/viewer/image/")) return;
    const resp = await req.response();
    const sizes = await req.sizes().catch(() => null);
    net.requests.push({
      url: req.url().replace(/^.*\/viewer\/image\//, ""),
      range: req.headers().range || null,
      status: resp ? resp.status() : null,
      bytes: sizes ? sizes.responseBodySize : Number(resp && resp.headers()["content-length"]),
    });
  });
  page.on("requestfailed", (req) => {
    if (req.url().includes("/viewer/image/")) net.failed.push({ range: req.headers().range || null, why: req.failure() && req.failure().errorText });
  });
  if (throttle) {
    const cdp = await context.newCDPSession(page);
    await cdp.send("Network.enable");
    await cdp.send("Network.emulateNetworkConditions", { offline: false, latency: throttle.latency, downloadThroughput: throttle.bps / 8, uploadThroughput: throttle.bps / 8 });
  }
  await page.goto(url);
  await page.waitForFunction(() => window.__t && (window.__t.error || window.__t.isIdle), null, { timeout: 120000 });
  const err = await page.evaluate(() => window.__t.error);
  if (err) throw new Error(err);
  return { page, context, net };
}

const idle = (page) => page.evaluate(async () => { await window.__t.whenIdle(); return performance.now(); });
const settle = async (page) => { await idle(page); await page.waitForTimeout(400); await idle(page); };
const netSummary = (net) => ({
  requests: net.requests.length,
  bytes: net.requests.reduce((s, r) => s + (r.bytes || 0), 0),
  nonRange: net.requests.filter((r) => !r.range || r.status !== 206).length,
  largest: Math.max(0, ...net.requests.map((r) => r.bytes || 0)),
  aborted: net.failed.length,
});
const shot = async (page, name) => { const p = join(out, name); await page.screenshot({ path: p }); return p; };
const longTasks = (page, since = 0) => page.evaluate((s) => window.__t.longTasks.filter((l) => l.start >= s).map((l) => Math.round(l.duration)), since);

async function dumpTiles(page, dataset, bundle, list, manifest) {
  for (const tl of list) {
    const res = await page.evaluate(async ({ channel, level, x, y }) => {
      const src = window.__t.ctl._groups[0].loader[level];
      const tile = await src.getTile({ x, y, selection: { c: channel } });
      const u8 = new Uint8Array(tile.data.buffer, tile.data.byteOffset, tile.data.byteLength);
      let s = "";
      for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
      return { width: tile.width, height: tile.height, b64: btoa(s) };
    }, tl);
    const file = `${dataset}_c${tl.channel}_L${tl.level}_${tl.x}_${tl.y}.u16`;
    writeFileSync(join(out, "tiles", file), Buffer.from(res.b64, "base64"));
    manifest.push({ dataset, bundle, ...tl, width: res.width, height: res.height, file });
  }
}

// Decode `n` distinct full-resolution tiles straight through the pixel source (no rendering
// of them) and report codec time inside the worker plus what the main thread saw meanwhile.
async function decodeBench(page, n = 20) {
  return page.evaluate(async (count) => {
    const { ctl } = window.__t;
    const src = ctl._groups[0].loader[0];
    const before = ctl.decodeTimes.length;
    const t0 = performance.now();
    let maxGap = 0;
    let last = t0;
    const timer = setInterval(() => { const now = performance.now(); maxGap = Math.max(maxGap, now - last); last = now; }, 5);
    const jobs = [];
    for (let i = 0; i < count; i++) jobs.push(src.getTile({ x: 28 + (i % 5), y: 5 + Math.floor(i / 5), selection: { c: 0 } }));
    const tiles = await Promise.all(jobs);
    clearInterval(timer);
    const wall = performance.now() - t0;
    const times = ctl.decodeTimes.slice(before);
    return {
      backend: ctl.decoderBackend,
      poolSize: ctl.poolSize,
      tiles: tiles.length,
      codecMs: times.map((v) => Math.round(v * 10) / 10),
      wallMs: wall,
      mainThreadMaxTimerGapMs: maxGap,
      longTasksDuring: window.__t.longTasks.filter((l) => l.start >= t0).map((l) => Math.round(l.duration)),
    };
  }, n);
}

const manifest = [];

// ------------------------------------------------------------------ skin: whole slide
{
  const { page, context, net } = await open(`${SKIN}/`);
  const tFirst = await idle(page);
  const info = await page.evaluate(() => ({
    channels: window.__t.ctl.channels, openMs: window.__t.ctl.stats.openMs, openedAt: window.__t.openedAt,
    backend: window.__t.ctl.decoderBackend, pool: window.__t.ctl.poolSize, contrast0: window.__t.state.channels[0],
    view: window.__t.getView(), tiles: window.__t.tileInfo(), hc: navigator.hardwareConcurrency,
  }));
  results.skin = { channels: info.channels, hardwareConcurrency: info.hc, poolSize: info.pool, backend: info.backend };
  results.skinWhole = {
    headerOpenMsPerFile: info.openMs.map(r1), allHeadersOpenAtMs: r1(info.openedAt), firstCompleteFrameMs: r1(tFirst),
    net: netSummary(net), view: info.view, dapiAutoContrast: [info.contrast0.min, info.contrast0.max],
    tilesDrawn: info.tiles.map((l) => l.tiles.filter((x) => x.selected).map((x) => `${x.z}/${x.x}/${x.y}`)),
    longTasks: await longTasks(page), screenshot: await shot(page, "skin_whole_dapi.png"),
  };
  results.skinWhole.headerRequests = net.requests.filter((r) => r.bytes < 20000).length;

  // 4 channels, each with its own colour and auto contrast.
  const before4 = net.requests.length;
  const limits = await page.evaluate(async () => {
    const lims = [];
    for (let i = 1; i < 4; i++) { window.__t.state.channels[i].on = true; lims.push(await window.__t.autoContrast(i)); }
    return lims;
  });
  await settle(page);
  results.skinWhole4 = {
    autoContrast: limits, extraRequests: net.requests.length - before4,
    extraBytes: net.requests.slice(before4).reduce((s, r) => s + r.bytes, 0),
    screenshot: await shot(page, "skin_whole_4ch.png"),
  };
  // Channel toggled off again: must not hit the network.
  const beforeToggle = net.requests.length;
  await page.evaluate(() => window.__t.setState({ channels: [{ index: 0, on: false }, { index: 2, on: false }] }));
  await settle(page);
  results.skinWhole4.toggleOffRequests = net.requests.length - beforeToggle;
  results.skinWhole4.screenshotTwoOff = await shot(page, "skin_whole_ch1_ch3_only.png");

  // Pixel dumps for comparison with tifffile.
  await dumpTiles(page, "skin", SKIN_DIR, [
    { channel: 0, level: 0, x: 30, y: 6 },
    { channel: 0, level: 0, x: 41, y: 19 }, // bottom-right corner tile (764 x 838)
    { channel: 1, level: 0, x: 3, y: 15 },
    { channel: 1, level: 2, x: 7, y: 1 },
    { channel: 3, level: 2, x: 10, y: 4 }, // corner tile of a floor-halved level (447 x 977)
    { channel: 2, level: 5, x: 1, y: 0 },
    { channel: 3, level: 7, x: 0, y: 0 },
  ], manifest);
  results.skinWhole.consoleErrors = [...net.console, ...(await page.evaluate(() => window.__t.consoleErrors))];
  await context.close();
}

// ------------------------------------------------------------------ skin: 500 µm view, fresh navigation
{
  const zoom = Math.log2(VIEW.width / 500);
  const { page, context, net } = await open(`${SKIN}/?x=${SPOT[0]}&y=${SPOT[1]}&zoom=${zoom}`);
  const tFirst = await idle(page);
  const info = await page.evaluate(() => ({ tiles: window.__t.tileInfo(), dec: window.__t.ctl.decodeTimes.slice(), fetchMs: window.__t.ctl.stats.fetchMs.slice(), stats: { ...window.__t.ctl.stats, fetchMs: undefined } }));
  results.skin500 = {
    zoom, firstCompleteFrameMs: r1(tFirst), net: netSummary(net),
    tilesDrawn: info.tiles.map((l) => l.tiles.filter((x) => x.selected).map((x) => `${x.z}/${x.x}/${x.y}`)),
    codecMsMedian: r1(median(info.dec)), fetchMsMedian: r1(median(info.fetchMs)),
    longTasks: await longTasks(page), screenshot: await shot(page, "skin_500um_dapi.png"),
  };
  // Contrast change.
  const c0 = await page.evaluate(() => window.__t.state.channels[0]);
  await page.evaluate((mx) => window.__t.setState({ channels: [{ index: 0, min: 0, max: mx }] }), Math.round(c0.max / 5));
  await settle(page);
  results.skin500.contrast = { from: [c0.min, c0.max], to: [0, Math.round(c0.max / 5)], screenshot: await shot(page, "skin_500um_dapi_contrast_max_div5.png") };
  await page.evaluate((c) => window.__t.setState({ channels: [{ index: 0, min: c.min, max: c.max }] }), c0);
  // Opacity.
  await page.evaluate(() => window.__t.setState({ opacity: 0.35 }));
  await settle(page);
  results.skin500.opacityScreenshot = await shot(page, "skin_500um_dapi_opacity_035.png");
  await page.evaluate(() => window.__t.setState({ opacity: 1 }));
  // 4 channels at full resolution.
  await page.evaluate(async () => { for (let i = 1; i < 4; i++) { window.__t.state.channels[i].on = true; await window.__t.autoContrast(i); } });
  await settle(page);
  results.skin500.screenshot4ch = await shot(page, "skin_500um_4ch.png");
  await page.evaluate(() => window.__t.setState({ channels: [1, 2, 3].map((index) => ({ index, on: false })) }));

  // Alignment: 20 cell centroids over DAPI in a ~200 µm view.
  const z200 = Math.log2(VIEW.width / 200);
  const half = [100 * 0.8, (100 * VIEW.height / VIEW.width) * 0.8];
  const cells = await (await fetch(`${SKIN}/test/cells?n=20&x0=${SPOT[0] - half[0]}&x1=${SPOT[0] + half[0]}&y0=${SPOT[1] - half[1]}&y1=${SPOT[1] + half[1]}`)).json();
  await page.evaluate(({ target, zoom: z, pts }) => { window.__t.setView({ target, zoom: z }); window.__t.setDots(pts.map((c) => c.centroid), pts.map((c) => c.nucleus).filter(Boolean)); document.body.classList.add("nopanel"); }, { target: SPOT, zoom: z200, pts: cells });
  await settle(page);
  results.alignment = { cells: cells.length, nucleusOutlines: cells.filter((c) => c.nucleus).length, zoom: z200, screenshot: await shot(page, "skin_200um_alignment.png") };
  await page.evaluate(() => { window.__t.setDots([]); document.body.classList.remove("nopanel"); });

  // Decode benchmark (wasm) + main-thread responsiveness while 20 tiles decode.
  await settle(page);
  const b = await decodeBench(page, 20);
  results.decodeWasm = { ...b, codecMsMedian: r1(median(b.codecMs)), wallMs: r1(b.wallMs), mainThreadMaxTimerGapMs: r1(b.mainThreadMaxTimerGapMs) };
  results.skin500.consoleErrors = [...net.console, ...(await page.evaluate(() => window.__t.consoleErrors))];
  await context.close();
}

// ------------------------------------------------------------------ skin: asm.js codec for comparison
{
  const { page, context, net } = await open(`${SKIN}/?js=1&worker=/assets/bench/xenium_image_worker.js`);
  await idle(page);
  const b = await decodeBench(page, 20);
  results.decodeJs = { ...b, codecMsMedian: r1(median(b.codecMs)), wallMs: r1(b.wallMs), mainThreadMaxTimerGapMs: r1(b.mainThreadMaxTimerGapMs), consoleErrors: net.console };
  await context.close();
}

// ------------------------------------------------------------------ skin: panning cancels work (throttled network)
{
  const zoom = Math.log2(VIEW.width / 500);
  // ~40 Mbit/s, 30 ms latency: closer to a remote deployment than loopback.
  const { page, context, net } = await open(`${SKIN}/?x=2000&y=3500&zoom=${zoom}`, { throttle: { latency: 30, bps: 40e6 } });
  await settle(page);
  await page.evaluate(async () => { for (let i = 1; i < 4; i++) { window.__t.state.channels[i].on = true; await window.__t.autoContrast(i); } });
  await settle(page);
  const mark = { req: net.requests.length, failed: net.failed.length };
  const s0 = await page.evaluate(() => ({ ...window.__t.ctl.stats, fetchMs: undefined }));
  // Jump to A (far away, all tiles new), then 250 ms later to B (no overlap with A).
  const A = [6800, 1700];
  const B = [8200, 900];
  await page.evaluate(({ target, zoom: z }) => window.__t.setView({ target, zoom: z }), { target: A, zoom });
  await page.waitForTimeout(250);
  const mid = await page.evaluate(() => ({ ...window.__t.ctl.stats, fetchMs: undefined }));
  await page.evaluate(({ target, zoom: z }) => window.__t.setView({ target, zoom: z }), { target: B, zoom });
  await settle(page);
  const s1 = await page.evaluate(() => ({ ...window.__t.ctl.stats, fetchMs: undefined }));
  const tiles = await page.evaluate(() => window.__t.tileInfo());
  // Which tile columns/rows belong to A and to B (full-resolution level, 1024 px tiles)?
  const tileBox = (c) => {
    const px = 0.2125 * 1024;
    return [Math.floor((c[0] - 250) / px), Math.floor((c[1] - 250 * VIEW.height / VIEW.width) / px), Math.floor((c[0] + 250) / px), Math.floor((c[1] + 250 * VIEW.height / VIEW.width) / px)];
  };
  const boxA = tileBox(A);
  const boxB = tileBox(B);
  const nTiles = (b) => (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  results.pan = {
    network: "throttled to 40 Mbit/s, 30 ms latency",
    tilesNeededA: nTiles(boxA) * 4, tilesNeededB: nTiles(boxB) * 4,
    requestsStartedByViewA_after250ms: mid.requests - s0.requests,
    requestsTotalAfterBoth: s1.requests - s0.requests,
    requestsCompleted: net.requests.length - mark.req,
    bytesCompleted: net.requests.slice(mark.req).reduce((s, r) => s + r.bytes, 0),
    requestsAbortedOnNetwork: net.failed.length - mark.failed,
    abortReasons: [...new Set(net.failed.slice(mark.failed).map((f) => f.why))],
    tileLoadsAbortedInSource: s1.aborted - s0.aborted,
    errors: s1.errors,
    finalTiles: tiles.map((l) => l.tiles.filter((x) => x.selected && x.loaded).length),
    screenshot: await shot(page, "skin_after_pan.png"),
    consoleErrors: [...net.console, ...(await page.evaluate(() => window.__t.consoleErrors))],
  };
  await context.close();
}

// ------------------------------------------------------------------ tiny single-file dataset
{
  const { page, context, net } = await open(`${TINY}/`);
  const tFirst = await idle(page);
  const info = await page.evaluate(() => ({ channels: window.__t.ctl.channels, openMs: window.__t.ctl.stats.openMs, c0: window.__t.state.channels[0] }));
  results.tiny = {
    channels: info.channels, headerOpenMs: info.openMs.map(r1), firstCompleteFrameMs: r1(tFirst), net: netSummary(net),
    autoContrast: [info.c0.min, info.c0.max], screenshot: await shot(page, "tiny_whole.png"),
  };
  // Zoom on the middle patch (about 612 µm, 374 µm) at full resolution.
  await page.evaluate(() => window.__t.setView({ target: [612, 374], zoom: 3 }));
  await settle(page);
  results.tiny.screenshotZoom = await shot(page, "tiny_zoom_patch.png");
  // Bottom-right corner of the image at full resolution (edge tiles 669 x 450 px).
  await page.evaluate(() => window.__t.setView({ target: [5789 * 0.2125 - 60, 3522 * 0.2125 - 40], zoom: 2.5 }));
  await settle(page);
  results.tiny.cornerTiles = (await page.evaluate(() => window.__t.tileInfo())).map((l) => l.tiles.filter((x) => x.selected).map((x) => `${x.z}/${x.x}/${x.y}:${x.loaded}`));
  await dumpTiles(page, "tiny", TINY_DIR, [
    { channel: 0, level: 0, x: 2, y: 1 },
    { channel: 0, level: 0, x: 5, y: 3 }, // corner tile (669 x 450)
    { channel: 0, level: 1, x: 1, y: 0 },
    { channel: 0, level: 1, x: 2, y: 1 }, // corner tile (846 x 737)
    { channel: 0, level: 4, x: 0, y: 0 },
  ], manifest);
  results.tiny.consoleErrors = [...net.console, ...(await page.evaluate(() => window.__t.consoleErrors))];
  await context.close();
}

// ------------------------------------------------------------------ worker file loaded as a plain page script
{
  const context = await browser.newContext();
  const page = await context.newPage();
  const reqs = [];
  const errs = [];
  page.on("request", (r) => reqs.push(r.url().replace(/^https?:\/\/[^/]+/, "")));
  page.on("pageerror", (e) => errs.push(String(e)));
  page.on("console", (m) => { if (m.type() === "error") errs.push(m.text()); });
  await page.goto(`${SKIN}/viewer/meta`);
  const globalsBefore = await page.evaluate(() => Object.keys(window).length);
  await page.addScriptTag({ url: `${SKIN}/assets/xenium_image_worker.js` });
  await page.waitForTimeout(500);
  results.workerAsPageScript = {
    errors: errs, requests: reqs, newGlobals: (await page.evaluate(() => Object.keys(window).length)) - globalsBefore,
    wasmFetched: reqs.some((u) => u.includes(".wasm")),
  };
  await context.close();
}

writeFileSync(join(out, "tiles", "manifest.json"), JSON.stringify(manifest, null, 1));
writeFileSync(join(out, "results.json"), JSON.stringify(results, null, 1));
await browser.close();
console.log(JSON.stringify(results, (k, v) => (k === "channels" && Array.isArray(v) ? v.map((c) => `${c.index}:${c.label}:${c.width}x${c.height}:${c.levels}`) : v), 1));
