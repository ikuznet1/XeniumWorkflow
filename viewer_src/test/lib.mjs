// Shared helpers for the Playwright tests: dev-server lifecycle, a WebGL-capable headless
// chromium, wire-format parsing in node, and screenshot pixel statistics.
import {chromium} from 'playwright';
import {spawn} from 'node:child_process';
import {existsSync, mkdirSync} from 'node:fs';
import {homedir} from 'node:os';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

export const HERE = dirname(fileURLToPath(import.meta.url));
export const ROOT = resolve(HERE, '..', '..');
export const SHOTS = resolve(HERE, 'screenshots');
mkdirSync(SHOTS, {recursive: true});

// Test bundles. XV_SKIN / XV_TINY name one bundle each; XV_TEST_DATA names the folder that
// holds them (default ~/Documents/XeniumTestData). Without either, the tiny bundle falls
// back to the fixture in the repository; tests that need a bundle that is absent skip.
export const TEST_DATA = process.env.XV_TEST_DATA || resolve(homedir(), 'Documents', 'XeniumTestData');
const firstExisting = (...paths) => paths.find((p) => existsSync(p)) || paths[0];
export const DATA = {
  skin: process.env.XV_SKIN || resolve(TEST_DATA, 'prime_human_skin_ffpe'),
  tiny: process.env.XV_TINY || firstExisting(resolve(TEST_DATA, 'prime_mouse_ileum_tiny'),
                                             resolve(ROOT, 'tests', 'data', 'prime_mouse_ileum_tiny')),
};
export const hasBundle = (dir) => existsSync(resolve(dir, 'experiment.xenium'));

// XV_GL=metal (default on macOS): the real GPU through ANGLE-Metal in headless chromium.
// XV_GL=swiftshader (default elsewhere): software WebGL2. Usable for small data only: in this
// chromium build requestAnimationFrame stops being serviced after the second large instanced
// draw (>= ~100k instances), with or without the viewer bundle (see test/gl_noise.mjs).
export const GL_MODE = process.env.XV_GL || (process.platform === 'darwin' ? 'metal' : 'swiftshader');
export const GL_ARGS = GL_MODE === 'metal'
  ? ['--use-angle=metal', '--ignore-gpu-blocklist']
  : ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'];

export async function startServer(bundle, port) {
  const proc = spawn(process.env.XV_PYTHON || 'python', [resolve(ROOT, 'viewer_src', 'dev_server.py'), bundle, String(port)], {
    stdio: ['ignore', 'ignore', 'pipe'],
  });
  let err = '';
  proc.stderr.on('data', (d) => (err += d));
  const url = `http://127.0.0.1:${port}`;
  const t0 = Date.now();
  for (;;) {
    if (proc.exitCode !== null) throw new Error(`dev server exited: ${err}`);
    try {
      const r = await fetch(`${url}/viewer/meta`);
      if (r.ok) break;
    } catch {}
    if (Date.now() - t0 > 60000) {
      proc.kill();
      throw new Error(`dev server did not start: ${err}`);
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  return {url, stop: () => proc.kill('SIGKILL')};
}

export async function launch() {
  return chromium.launch({headless: true, args: GL_ARGS});
}

/** New page that records console errors/warnings and every request URL. */
export async function newPage(browser, viewport = {width: 1280, height: 800}) {
  const ctx = await browser.newContext({viewport, deviceScaleFactor: 1});
  const page = await ctx.newPage();
  const consoleIssues = [];
  const requests = [];
  page.on('console', (m) => {
    if (m.type() === 'error' || m.type() === 'warning') consoleIssues.push(`${m.type()}: ${m.text()}`);
  });
  page.on('pageerror', (e) => consoleIssues.push(`pageerror: ${e.message}`));
  page.on('request', (r) => requests.push(r.url()));
  return {page, consoleIssues, requests, close: () => ctx.close()};
}

export function parseBin(buf) {
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const hl = new DataView(ab).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(ab, 4, hl)));
  const T = {float32: Float32Array, uint32: Uint32Array, int32: Int32Array, uint16: Uint16Array, uint8: Uint8Array};
  const arrays = {};
  let off = 4 + hl;
  for (const a of header.arrays || []) {
    const bytes = a.length * T[a.dtype].BYTES_PER_ELEMENT;
    arrays[a.name] = new T[a.dtype](ab.slice(off, off + bytes));
    off += bytes + ((4 - (bytes % 4)) % 4);
  }
  return {header, arrays};
}

export async function fetchBin(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return parseBin(Buffer.from(await r.arrayBuffer()));
}

/**
 * Screenshot a clip of the page and return pixel statistics computed in the browser
 * (decodes the PNG on a 2D canvas, so no node image dependency is needed).
 * `bg` is the background colour [r,g,b]; a pixel counts as "drawn" if it differs by > 24.
 */
export async function shot(page, name, {clip, bg = [13, 17, 23]} = {}) {
  const path = resolve(SHOTS, name);
  const png = await page.screenshot({path, clip});
  const stats = await page.evaluate(async ({b64, bg}) => {
    const img = new Image();
    img.src = `data:image/png;base64,${b64}`;
    await img.decode();
    const c = document.createElement('canvas');
    c.width = img.width;
    c.height = img.height;
    const g = c.getContext('2d');
    g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height).data;
    let drawn = 0, sr = 0, sg = 0, sb = 0, hash = 0;
    const colors = new Set();
    for (let i = 0; i < d.length; i += 4) {
      const dr = Math.abs(d[i] - bg[0]), dg = Math.abs(d[i + 1] - bg[1]), db = Math.abs(d[i + 2] - bg[2]);
      if (dr + dg + db > 24) drawn++;
      sr += d[i]; sg += d[i + 1]; sb += d[i + 2];
      hash = (hash * 31 + d[i] * 7 + d[i + 1] * 3 + d[i + 2]) >>> 0;
      if (colors.size < 5000) colors.add((d[i] >> 3) << 10 | (d[i + 1] >> 3) << 5 | (d[i + 2] >> 3));
    }
    const n = d.length / 4;
    return {width: c.width, height: c.height, drawnFraction: drawn / n,
      mean: [sr / n, sg / n, sb / n].map((v) => Math.round(v * 10) / 10), distinctColors: colors.size, hash};
  }, {b64: png.toString('base64'), bg});
  return {path, ...stats};
}

/** Fraction of pixels that differ between two screenshots of identical size (computed in page). */
export async function diffFraction(page, pathA, pathB) {
  const {readFileSync} = await import('node:fs');
  const [a, b] = [pathA, pathB].map((p) => readFileSync(p).toString('base64'));
  return page.evaluate(async ({a, b}) => {
    const load = async (b64) => {
      const img = new Image();
      img.src = `data:image/png;base64,${b64}`;
      await img.decode();
      const c = document.createElement('canvas');
      c.width = img.width;
      c.height = img.height;
      const g = c.getContext('2d');
      g.drawImage(img, 0, 0);
      return g.getImageData(0, 0, c.width, c.height).data;
    };
    const [da, db] = [await load(a), await load(b)];
    if (da.length !== db.length) return 1;
    let diff = 0;
    for (let i = 0; i < da.length; i += 4) {
      if (Math.abs(da[i] - db[i]) + Math.abs(da[i + 1] - db[i + 1]) + Math.abs(da[i + 2] - db[i + 2]) > 24) diff++;
    }
    return diff / (da.length / 4);
  }, {a, b});
}

export function pointInPolygon(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export class Checks {
  constructor() {
    this.results = [];
  }
  ok(cond, name, detail = '') {
    this.results.push({pass: !!cond, name, detail});
    console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
    return !!cond;
  }
  get failed() {
    return this.results.filter((r) => !r.pass);
  }
}
