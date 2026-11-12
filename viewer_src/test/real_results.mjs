// WebGL view against the Plotly view on REAL analysis results (no fabricated state).
//
// Needs a Xenium bundle and a cache directory that holds real runs for it: Baysor / Proseg
// runs, annotation label caches (Seurat / CellTypist / RCTD with weights), SPLIT-corrected
// counts, SpaGE-imputed genes and the sopa store with a tissue ROI and patches.
//
//   XV_REAL_DATA   bundle             (default …/XeniumTestData/skin_pipeline_copy)
//   XV_REAL_CACHE  XENIUM_CACHE_DIR   (default …/XeniumTestData/cache_pipelines)
//   XV_REAL_EXTRA  second sample(s) for the multi-sample checks, comma-separated
//                  (default …/prime_mouse_ileum_tiny and …/geo_rv_0038290_R2/bundle)
//   XV_REAL_URL    use an app that is already running there instead of starting one
//   XV_REAL_ONLY   comma-separated sections: reseg,counts,pies,sdata,click,lasso,subset,multi,umap
//   XV_REAL_ROI=0  skip the lasso check (section 6). It saves an ROI of class "rrtest" and
//                  deletes it again through the ROI manager; as with any ROI saved in the
//                  app, the stores of the loaded runs keep an (empty) roi_rrtest column.
//
//   node test/real_results.mjs [screenshot dir]
//
// The script starts the real app (XENIUM_RENDERER=webgl), selects runs / labels / counts
// through the Dash controls, and for every state reads what the WebGL viewer holds
// (cells.bin, colors.bin, boundaries.bin, the pie / outline overlays) and what the Plotly
// figure of the same state contains (switching renderer-mode to "plotly" and reading the
// figure in the browser), and compares the two cell by cell. Skips (exit 0) when the
// bundle or the cache is not on this machine. Each section also skips what the cache does
// not hold (no stored RCTD weights - the app keeps them for "full" mode runs only -, no
// recorded patches, no second bundle). There are no CellNEST results in that cache:
// CellNEST edges stay on the fabricated test (parity_app.mjs).
//
// Where the Plotly figure itself is at fault the script says so instead of failing: a
// second sample without cluster sets makes the Plotly figure callback raise in cluster mode
// (reported with a NOTE; the WebGL view draws that state). The RCTD pie wedges are compared
// with the figure's layout shapes, i.e. with what Plotly is asked to draw: plotly.js itself
// drops the arc command of those paths, so its own picture of the pies is not usable.
import {chromium} from 'playwright';
import {spawn} from 'node:child_process';
import {existsSync, mkdirSync, readFileSync, readdirSync} from 'node:fs';
import {createServer} from 'node:net';
import {resolve} from 'node:path';
import {GL_ARGS, GL_MODE, ROOT, SHOTS} from './lib.mjs';

const TD = '/Users/ikuz/Documents/XeniumTestData';
const DATA_DIR = process.env.XV_REAL_DATA || `${TD}/skin_pipeline_copy`;
const CACHE_DIR = process.env.XV_REAL_CACHE || `${TD}/cache_pipelines`;
const EXTRA_DIR = process.env.XV_REAL_EXTRA || `${TD}/prime_mouse_ileum_tiny,${TD}/geo_rv_0038290_R2/bundle`;   // comma-separated
const OUT = process.argv[2] || resolve(SHOTS, 'real');
const ONLY = (process.env.XV_REAL_ONLY || '').split(',').filter(Boolean);
const want = (s) => !ONLY.length || ONLY.includes(s);

const cacheRuns = existsSync(CACHE_DIR) ? readdirSync(CACHE_DIR).filter((f) => /^(baysor|proseg)_/.test(f)) : [];
if (!existsSync(resolve(DATA_DIR, 'experiment.xenium')) || !cacheRuns.length) {
  console.log(`SKIP  real-results checks: no bundle at ${DATA_DIR} or no Baysor / Proseg runs in ${CACHE_DIR}`);
  process.exit(0);
}
mkdirSync(OUT, {recursive: true});

// ── the app ────────────────────────────────────────────────────────────────────
let app = null;
let base = process.env.XV_REAL_URL || '';
if (!base) {
  const port = await new Promise((res) => { const s = createServer(); s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => res(p)); }); });
  app = spawn(process.env.XV_PYTHON || 'python', [resolve(ROOT, 'xenium_explorer.py'), DATA_DIR], {
    cwd: ROOT, stdio: ['ignore', 'ignore', 'ignore'],
    // (the in-app REPL, localhost only, is how subset() / unsubset() are called in section 7)
    env: {...process.env, XENIUM_RENDERER: 'webgl', PORT: String(port), XENIUM_CACHE_DIR: CACHE_DIR, XENIUM_ENABLE_REPL: '1'},
  });
  base = `http://127.0.0.1:${port}`;
  const t0 = Date.now();
  for (;;) {
    if (app.exitCode !== null) { console.log('FAIL  the app exited during start-up'); process.exit(1); }
    try { if ((await fetch(`${base}/viewer/meta`)).ok) break; } catch { /* not up yet */ }
    if (Date.now() - t0 > 600000) { app.kill('SIGKILL'); console.log('FAIL  the app did not start within 10 min'); process.exit(1); }
    await new Promise((r) => setTimeout(r, 1000));
  }
  console.log(`app started on ${base} after ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
const stopApp = () => { if (app) { try { app.kill('SIGKILL'); } catch { /* gone */ } app = null; } };
process.on('exit', stopApp);
process.on('SIGINT', () => { stopApp(); process.exit(130); });

// ── the browser ────────────────────────────────────────────────────────────────
const browser = await chromium.launch({args: GL_ARGS});
const page = await browser.newPage({viewport: {width: 1500, height: 900}});
const logs = [], reqs = [];
page.on('console', (m) => { if (m.type() === 'error') logs.push(`error: ${m.text().slice(0, 300)}`); });
page.on('pageerror', (e) => logs.push('pageerror: ' + e.message.slice(0, 300)));
page.on('request', (r) => { const u = new URL(r.url()); if (u.pathname.includes('/viewer/')) reqs.push(u.pathname + u.search); });
let failures = 0;
const check = (name, ok, detail = '') => { if (!ok) failures += 1; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`); return ok; };
const note = (text) => console.log(`      ${text}`);
const J = (o) => JSON.stringify(o);
const V = (fn, arg) => page.evaluate(fn, arg);
const wait = (ms) => page.waitForTimeout(ms);
const setProp = (id, props) => V(([i, p]) => window.dash_clientside.set_props(i, p), [id, props]);
const prop = (id, key = 'value') => V(([i, k]) => { const n = window.dash_component_api.getLayout(i); return n && n.props ? n.props[k] : undefined; }, [id, key]);
const options = async (id) => ((await prop(id, 'options')) || []).map((o) => (o && typeof o === 'object' ? o : {value: o}));
const hasOption = async (id, value) => (await options(id)).some((o) => o.value === value && !o.disabled);
const until = async (fn, ms = 120000, step = 500) => {
  const t0 = Date.now();
  for (;;) {
    const v = await fn();
    if (v) return v;
    if (Date.now() - t0 > ms) return null;
    await wait(step);
  }
};
const idle = async () => { await wait(350); await V(() => window.XeniumViewer.instance.whenIdle()); await wait(250); };
const shot = (name) => page.screenshot({path: `${OUT}/${name}`});

// Helpers that run in the page. window.__rr.plotly() reads the Plotly spatial figure,
// window.__rr.gl() the WebGL viewer's tables; the cmp* functions compare the two.
const HELPERS = () => {
  const f32 = Math.fround;
  const rgbOf = (c) => {
    const s = String(c).trim();
    const m = /^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)/.exec(s);
    if (m) return [Math.round(+m[1]), Math.round(+m[2]), Math.round(+m[3])];
    let h = s.replace(/^#/, '');
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    const v = parseInt(h, 16);
    return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
  };
  const gd = (sel = '#spatial-plot') => document.querySelector(sel + ' .js-plotly-plot') || document.querySelector(sel);
  const rr = (window.__rr = {});
  // which: 'instance' (the spatial viewer) or 'umapInstance' (the UMAP viewer, served from /umap/viewer/*)
  rr.viewer = (which = 'instance') => window.XeniumViewer[which]._viewer;

  // Cells of the Plotly figure: one entry per marker of the cell traces.
  rr.plotly = (sel) => {
    const g = gd(sel);
    const fd = g._fullData || [];
    const x = [], y = [], id = [], rgb = [], val = [], name = [];
    const info = {title: (g.layout.title && g.layout.title.text) || '', traces: fd.length, cellTraces: 0, names: [],
      continuous: null, lines: [], shapes: (g.layout.shapes || []).length, ghost: false};
    for (const t of fd) {
      const n = t.x ? t.x.length : 0;
      if (t.mode === 'lines') { info.lines.push(t.name); continue; }
      if (t.mode !== 'markers' || !t.customdata || t.customdata.length !== n || !n || t.x[0] === null) continue;
      info.cellTraces += 1;
      const mc = t.marker.color;
      const perPoint = mc && typeof mc !== 'string' && mc.length === n;
      const numeric = perPoint && typeof mc[0] === 'number';
      if (t.marker.opacity === 0) info.ghost = true;
      if (numeric) {
        const cs = t.marker.colorscale || [];
        info.continuous = {cmin: t.marker.cmin, cmax: t.marker.cmax, first: cs.length ? rgbOf(cs[0][1]) : null,
          last: cs.length ? rgbOf(cs[cs.length - 1][1]) : null,
          label: ((t.marker.colorbar && t.marker.colorbar.title && t.marker.colorbar.title.text) || '').replace(/<br>/g, ' ')};
      } else if (!perPoint) {
        info.names.push(t.name);
      }
      const fixed = !perPoint ? rgbOf(mc) : null;
      for (let i = 0; i < n; i++) {
        x.push(f32(t.x[i])); y.push(f32(-t.y[i])); id.push(String(t.customdata[i]));
        val.push(numeric ? mc[i] : NaN);
        rgb.push(numeric ? null : (fixed || rgbOf(mc[i])));
        name.push(perPoint ? null : t.name);
      }
    }
    rr.pl = {x, y, id, rgb, val, name, info};
    return {n: x.length, ...info};
  };

  // Compare rr.pl with the WebGL viewer's cell table and colours.
  rr.cmpCells = (which) => {
    const v = rr.viewer(which), pl = rr.pl;
    const cells = v._cells, col = v._colors, rgba = v._rgba;
    const out = {plotly: pl.x.length, gl: cells ? cells.n : 0, kind: col && col.header.kind, label: col && col.header.label,
      source: cells && cells.source};
    if (!cells || !col) return out;
    const rows = new Map();
    let placed = 0;
    for (let i = 0; i < cells.n; i++) {
      if (cells.xy[2 * i] !== cells.xy[2 * i]) continue;   // no position (a cell without an embedding)
      placed += 1;
      const k = cells.xy[2 * i] + ',' + cells.xy[2 * i + 1];
      const a = rows.get(k);
      if (a) a.push(i); else rows.set(k, [i]);
    }
    const used = new Uint8Array(cells.n);
    const row = new Int32Array(pl.x.length).fill(-1);
    let unmatched = 0, sameOrder = 0;
    for (let j = 0; j < pl.x.length; j++) {
      const a = rows.get(pl.x[j] + ',' + pl.y[j]);
      let r = -1;
      if (a) {
        // cells sharing one position: prefer the row whose colour agrees
        for (const i of a) if (!used[i]) { r = i; if (!pl.rgb[j] || (rgba[4 * i] === pl.rgb[j][0] && rgba[4 * i + 1] === pl.rgb[j][1] && rgba[4 * i + 2] === pl.rgb[j][2])) break; }
      }
      if (r < 0) { unmatched += 1; continue; }
      used[r] = 1; row[j] = r;
      if (r === j) sameOrder += 1;
    }
    out.placed = placed;
    out.positionsUnmatched = unmatched;
    out.sameOrder = sameOrder;
    rr.row = row;
    if (col.header.kind === 'categorical') {
      const cats = col.header.categories, codes = col.codes;
      let colour = 0, nameDiff = 0, compared = 0;
      const ex = [];
      for (let j = 0; j < row.length; j++) {
        const r = row[j];
        if (r < 0 || !pl.rgb[j]) continue;
        compared += 1;
        const c = pl.rgb[j];
        const bad = rgba[4 * r] !== c[0] || rgba[4 * r + 1] !== c[1] || rgba[4 * r + 2] !== c[2];
        if (bad) { colour += 1; if (ex.length < 3) ex.push({id: pl.id[j], plotly: c, gl: [rgba[4 * r], rgba[4 * r + 1], rgba[4 * r + 2]], plName: pl.name[j], glName: cats[codes[r]]}); }
        if (pl.name[j] !== null && pl.name[j] !== cats[codes[r]]) nameDiff += 1;
      }
      out.compared = compared; out.colourDiff = colour; out.nameDiff = nameDiff; out.examples = ex;
      out.categories = cats.length;
      out.plotlyCategories = new Set(pl.name.filter((s) => s !== null)).size;
    } else {
      const vals = col.values;
      let maxDiff = 0, nanDiff = 0, compared = 0;
      for (let j = 0; j < row.length; j++) {
        const r = row[j];
        if (r < 0 || pl.rgb[j]) continue;
        compared += 1;
        const a = pl.val[j], b = vals[r];
        const an = a === null || a !== a, bn = b !== b;
        if (an || bn) { if (an !== bn) nanDiff += 1; continue; }
        const d = Math.abs(a - b);
        if (d > maxDiff) maxDiff = d;
      }
      const range = v._colorRange();
      // first and last entry of the viewer's colour maps (viewer_src/src/colormap.js)
      const lut = {viridis: [68, 1, 84, 253, 231, 37], plasma: [13, 8, 135, 240, 249, 33], magma: [0, 0, 4, 252, 253, 191],
        cividis: [0, 34, 78, 254, 232, 56]}[v._colormap()] || null;
      out.compared = compared; out.maxDiff = maxDiff; out.nanDiff = nanDiff;
      out.range = range; out.plotlyRange = pl.info.continuous && [pl.info.continuous.cmin, pl.info.continuous.cmax];
      out.colormap = v._colormap(); out.plotlyLabel = pl.info.continuous && pl.info.continuous.label;
      out.plotlyScaleEnds = pl.info.continuous && [pl.info.continuous.first, pl.info.continuous.last];
      out.glScaleEnds = lut ? [lut.slice(0, 3), lut.slice(3)] : null;
    }
    return out;
  };

  // Cell ids of a sample of rows: /viewer/cell/<row> against the Plotly customdata.
  rr.cmpIds = async (k = 40, which = 'instance') => {
    const v = rr.viewer(which), pl = rr.pl, row = rr.row;
    const picks = [];
    for (let s = 0; s < k; s++) { const j = Math.floor(((s + 0.5) / k) * row.length); if (row[j] >= 0) picks.push(j); }
    let bad = 0;
    for (const j of picks) {
      const c = await (await fetch((which === 'umapInstance' ? '/umap' : '') + '/viewer/cell/' + row[j] + '?source=' + encodeURIComponent(v._cells.source))).json();
      if (String(c.id) !== pl.id[j]) bad += 1;
    }
    return {checked: picks.length, bad};
  };

  const ringsKey = (xs, ys) => {
    let n = xs.length;
    if (n > 1 && xs[0] === xs[n - 1] && ys[0] === ys[n - 1]) n -= 1;   // closing vertex repeated
    const p = [];
    for (let i = 0; i < n; i++) p.push(xs[i].toFixed(3) + ' ' + ys[i].toFixed(3));
    return p.join('|');
  };
  // Rings of a Plotly boundary trace against the rings the viewer holds for a kind.
  rr.cmpBoundaries = (kind, traceName, vp) => {
    const v = rr.viewer();
    const t = (gd()._fullData || []).find((d) => d.mode === 'lines' && d.name === traceName);
    const d = v._loaders[kind].data;
    const out = {plotly: 0, gl: d && d.header.n ? d.header.n : 0, common: 0, glInView: 0, glInViewMissing: 0, closedRings: 0,
      header: d && {n: d.header.n, truncated: d.header.truncated, available: d.header.available}};
    const glKeys = new Map();
    if (d && d.arrays.xy) {
      const {start, xy, cell} = d.arrays;
      for (let r = 0; r < d.header.n; r++) {
        const xs = [], ys = [];
        for (let i = start[r]; i < start[r + 1]; i++) { xs.push(xy[2 * i]); ys.push(xy[2 * i + 1]); }
        if (xs.length > 1 && xs[0] === xs[xs.length - 1] && ys[0] === ys[ys.length - 1]) out.closedRings += 1;
        glKeys.set(ringsKey(xs, ys), cell ? cell[r] : -1);
      }
      out.glDistinct = glKeys.size;
    }
    const plKeys = new Set();
    if (t) {
      let xs = [], ys = [];
      const flush = () => { if (xs.length >= 3) plKeys.add(ringsKey(xs, ys)); xs = []; ys = []; };
      for (let i = 0; i < t.x.length; i++) {
        const a = t.x[i], b = t.y[i];
        if (a === null || b === null || a !== a || b !== b) flush();
        else { xs.push(f32(a)); ys.push(f32(-b)); }
      }
      flush();
    }
    out.plotly = plKeys.size;
    for (const k of plKeys) if (glKeys.has(k)) out.common += 1;
    // rings of the viewer whose cell centre is inside the viewport (vp, as it was in the WebGL
    // view) must all be in the Plotly trace; the row -> cell lookup needs the kind's own table
    const cells = v._cells;
    if (vp && cells && kind === cells.source.split(':')[0]) {
      for (const [k, c] of glKeys) {
        const x = cells.xy[2 * c], y = cells.xy[2 * c + 1];
        if (x >= vp.x0 && x <= vp.x1 && y >= vp.y0 && y <= vp.y1) { out.glInView += 1; if (!plKeys.has(k)) out.glInViewMissing += 1; }
      }
    }
    return out;
  };

  const area = (pts) => { let a = 0; for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) a += pts[j][0] * pts[i][1] - pts[i][0] * pts[j][1]; return Math.abs(a) / 2; };
  // RCTD pies: wedges in the Plotly layout shapes against the viewer's pie polygons,
  // matched on (cell centre, fill colour) and compared by area.
  // A Plotly wedge is the SVG path "M cx,cy L x1,y1 A r,r,0,large,1,x2,y2 Z" in plot space
  // (y = -µm); the viewer's is a polygon [centre, start, …arc…, end] in µm. A wedge is
  // matched on (centre, start point, fill colour) and compared by its end point and by
  // whether it spans more than half the circle.
  rr.cmpPies = (vp) => {
    const v = rr.viewer();
    const wedge = new Map();
    let plotly = 0, plotlyInView = 0;
    const key = (c, s, fill) => c[0].toFixed(2) + ',' + c[1].toFixed(2) + ',' + s[0].toFixed(2) + ',' + s[1].toFixed(2) + ',' + rgbOf(fill).join();
    for (const s of gd().layout.shapes || []) {
      if (s.type !== 'path' || !s.fillcolor || !(s.line && s.line.width === 0)) continue;
      const m = (s.path.match(/-?\d+(?:\.\d+)?(?:e[-+]?\d+)?/gi) || []).map(Number);
      if (m.length !== 11) continue;
      plotly += 1;
      const c = [m[0], -m[1]];
      const inView = !vp || (c[0] >= vp.x0 && c[0] <= vp.x1 && c[1] >= vp.y0 && c[1] <= vp.y1);
      if (inView) plotlyInView += 1;
      wedge.set(key(c, [m[2], -m[3]], s.fillcolor), {end: [m[9], -m[10]], large: m[7] === 1, r: m[4], opacity: s.opacity, inView, hit: false});
    }
    const polys = (v.state.overlays.polygons || []).filter((p) => p.width === 0);
    let common = 0, maxEnd = 0, largeDiff = 0, maxRadius = 0;
    for (const p of polys) {
      const n = p.points.length;
      const w = wedge.get(key(p.points[0], p.points[1], p.fill));
      if (!w) continue;
      common += 1;
      w.hit = true;
      maxEnd = Math.max(maxEnd, Math.hypot(p.points[n - 1][0] - w.end[0], p.points[n - 1][1] - w.end[1]));
      maxRadius = Math.max(maxRadius, Math.abs(Math.hypot(p.points[1][0] - p.points[0][0], p.points[1][1] - p.points[0][1]) - w.r));
      // n - 2 arc steps of at most 15 degrees each: more than 12 means more than half a turn
      if ((n - 2 > 12) !== w.large) largeDiff += 1;
    }
    const cellsOf = (keys) => new Set(keys.map((k) => k.split(',').slice(0, 2).join())).size;
    return {plotly, plotlyInView, plotlyInViewMissing: [...wedge.values()].filter((w) => w.inView && !w.hit).length,
      gl: polys.length, common, maxEnd, maxRadius, largeDiff, cells: cellsOf(polys.map((p) => key(p.points[0], p.points[1], p.fill))),
      glAlpha: polys[0] && polys[0].fillAlpha, plotlyOpacity: wedge.size ? [...wedge.values()][0].opacity : null};
  };

  // Outline shapes of the Plotly layout (ROIs, sdata ROI / patches) as vertex lists, y down.
  rr.plotlyOutlines = () => (gd().layout.shapes || []).filter((s) => !(s.type === 'path' && s.line && s.line.width === 0)).map((s) => {
    let pts = [];
    if (s.type === 'path') {
      const nums = (s.path.match(/-?\d+(?:\.\d+)?(?:e-?\d+)?/gi) || []).map(Number);
      for (let i = 0; i + 1 < nums.length; i += 2) pts.push([nums[i], -nums[i + 1]]);
    } else if (s.type === 'rect') {
      pts = [[s.x0, -s.y0], [s.x1, -s.y0], [s.x1, -s.y1], [s.x0, -s.y1]];
    } else if (s.type === 'line') {
      pts = [[s.x0, -s.y0], [s.x1, -s.y1]];
    }
    return {type: s.type, color: s.line && s.line.color ? rgbOf(s.line.color).join() : null, n: pts.length, pts};
  });
};

// ── state changes and the two renderers ────────────────────────────────────────
const glState = () => V(() => { const v = window.__rr.viewer(); return {source: v._cells && v._cells.source, n: v._cells ? v._cells.n : 0,
  mode: v.state.color.mode, kind: v._colors && v._colors.header.kind, label: v._colors && v._colors.header.label,
  status: document.querySelector('#webgl-status').textContent, layers: v.deck.props.layers.map((l) => l.id)}; });

/** Switch to the Plotly renderer, wait for its figure of the current state, run fn, switch back. */
async function withPlotly(fn, shotName) {
  const resp = page.waitForResponse(async (r) => r.url().includes('_dash-update-component') && (r.request().postData() || '').includes('"output":"..spatial-plot.figure'), {timeout: 240000});
  const mark = logs.length;
  await setProp('renderer-mode', {value: 'plotly'});
  const failed = (await resp).status() !== 200;
  await until(() => V(() => { const g = document.querySelector('#spatial-plot .js-plotly-plot'); return !!(g && g._fullData && g._fullData.length && getComputedStyle(document.querySelector('#spatial-plot')).display !== 'none'); }), 60000);
  await wait(2500);
  let res;
  try {
    // A figure callback that raised leaves the previous figure on screen: nothing to compare.
    res = failed ? {plotlyError: true} : await fn();
    if (failed) logs.length = mark;     // (the browser's report of that 500 is not a WebGL error)
    if (shotName && !failed) await shot(shotName);
  } finally {
    await setProp('renderer-mode', {value: 'webgl'});
    await wait(1200);
    await idle();
  }
  return res;
}

/** Colours of the current state in both renderers: {gl, plotly, cmp, ids}. */
async function compareColours(tag, shots = true) {
  await idle();
  if (shots) await shot(`${tag}_webgl.png`);
  const gl = await glState();
  const res = await withPlotly(async () => {
    const plotly = await V(() => window.__rr.plotly());
    const cmp = await V(() => window.__rr.cmpCells());
    const ids = await V(() => window.__rr.cmpIds());
    return {plotly, cmp, ids};
  }, shots ? `${tag}_plotly.png` : null);
  return {gl, ...res};
}
/** Plotly's figure callback raised for this state (reported, since the WebGL view has nothing to agree with). */
const plotlyFailed = (r, what) => { if (r && r.plotlyError) console.log(`NOTE  ${what}: the Plotly figure callback raises an error for this state; WebGL view not compared`); return !!(r && r.plotlyError); };
/** Colours and positions of the UMAP panel in both renderers (cells without an embedding are not drawn). */
async function compareUmap(tag) {
  await idle();
  await V(() => window.XeniumViewer.umapInstance.whenIdle());
  if (tag) await shot(`${tag}_webgl.png`);
  return withPlotly(async () => {
    const plotly = await V(() => window.__rr.plotly('#umap-plot'));
    const cmp = await V(() => window.__rr.cmpCells('umapInstance'));
    const ids = await V(() => window.__rr.cmpIds(40, 'umapInstance'));
    return {plotly, cmp, ids};
  }, tag ? `${tag}_plotly.png` : null);
}
// names: false for the UMAP panel, whose Plotly traces carry no legend names
const coloursAgree = (r, names = true) => {
  const c = r.cmp;
  if (!c || c.plotly !== c.placed || !c.placed || c.positionsUnmatched !== 0 || r.ids.bad !== 0 || !r.ids.checked) return false;
  if (c.kind === 'categorical') return c.compared === c.placed && c.colourDiff === 0 && (!names || c.nameDiff === 0);
  const close = (a, b) => Math.abs(a - b) <= 1e-4 * Math.max(1, Math.abs(a), Math.abs(b));
  const ends = c.plotlyScaleEnds && c.glScaleEnds && [0, 1].every((e) => c.plotlyScaleEnds[e].every((p, i) => Math.abs(p - c.glScaleEnds[e][i]) <= 12));
  return c.compared === c.placed && c.maxDiff < 1e-4 && c.nanDiff === 0 && !!c.plotlyRange && close(c.range[0], c.plotlyRange[0]) && close(c.range[1], c.plotlyRange[1]) && ends;
};
const brief = (r) => { if (r.plotlyError) return 'Plotly callback error'; const c = r.cmp || {}; return J(c.kind === 'categorical'
  ? {cells: [c.plotly, c.placed, c.gl], unmatched: c.positionsUnmatched, colourDiff: c.colourDiff, nameDiff: c.nameDiff, categories: [c.plotlyCategories, c.categories], ids: r.ids, ex: c.examples && c.examples[0]}
  : {cells: [c.plotly, c.placed, c.gl], unmatched: c.positionsUnmatched, maxDiff: c.maxDiff, nanDiff: c.nanDiff, range: c.range, plotlyRange: c.plotlyRange, colormap: c.colormap, ends: [c.plotlyScaleEnds, c.glScaleEnds], label: [c.label, c.plotlyLabel], ids: r.ids}); };

/** Select a segmentation source through the dropdown and wait until the viewer shows it. */
async function selectSource(value) {
  const tool = value.split(':')[0];
  if ((await prop('seg-source')) === value) return true;
  if (!(await hasOption('seg-source', value))) return false;
  const doneId = tool === 'xenium' ? null : `${tool}-done`;
  const before = doneId ? await prop(doneId, 'data') : null;
  await setProp('seg-source', {value});
  if (doneId) {
    // A cached run that is not in memory is loaded in the background: its poll is switched
    // on and <tool>-done is bumped when the run is in place. (A run that is still loaded
    // from an earlier selection is shown at once.)
    await wait(1500);
    if ((await prop(`${tool}-poll`, 'disabled')) === false) {
      await until(async () => (await prop(doneId, 'data')) !== before, 300000, 1000);
    }
  }
  await wait(1500);
  await until(async () => (await glState()).source === value, 60000);
  await idle();
  return (await prop('seg-source')) === value;
}

const view = (x0, y0, x1, y1) => V((b) => window.XeniumViewer.instance.setViewport({x0: b[0], y0: b[1], x1: b[2], y1: b[3]}), [x0, y0, x1, y1]);
const fit = () => V(() => window.XeniumViewer.instance.fitView());

// ── open the app ───────────────────────────────────────────────────────────────
await page.goto(base, {waitUntil: 'domcontentloaded'});
await page.waitForFunction(() => document.querySelector('#webgl-view canvas'), null, {timeout: 180000});
await page.waitForFunction(() => window.XeniumViewer.instance && window.XeniumViewer.instance.stats.firstRenderMs !== undefined, null, {timeout: 180000});
await V(HELPERS);
await wait(3000);
await idle();
console.log(`bundle ${DATA_DIR}\ncache  ${CACHE_DIR}\nWebGL backend ${GL_MODE}`);
const segOptions = (await options('seg-source')).map((o) => o.value);
const baysorRuns = segOptions.filter((v) => v.startsWith('baysor:'));
const prosegRuns = segOptions.filter((v) => v.startsWith('proseg:'));
const nXenium = (await glState()).n;
note(`Xenium cells ${nXenium}; cached runs: ${segOptions.filter((v) => v !== 'xenium').join(', ')}`);
check('0 cached Baysor and Proseg runs are offered in the segmentation dropdown', baysorRuns.length > 0 && prosegRuns.length > 0, J(segOptions));

const colourModes = async () => (await options('color-by')).filter((o) => !o.disabled).map((o) => o.value);
const BOUNDARY_TRACE = {cell: 'Xenium Cell Boundaries', nucleus: 'Xenium Nuclei Boundaries', baysor: 'Baysor Boundaries', proseg: 'Proseg Boundaries'};
/** Zoom to the middle of the cells on screen (a box of w x h µm). */
const zoomToCells = async (w, h, fx = 0.5, fy = 0.5) => {
  const b = await V(() => window.__rr.viewer()._cells.bounds);
  const cx = b[0] + (b[2] - b[0]) * fx, cy = b[1] + (b[3] - b[1]) * fy;
  await view(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2);
  await idle();
  return [cx, cy];
};

// ── 1. Baysor / Proseg runs: cells, colours, outlines ──────────────────────────
if (want('reseg')) {
  // one whole-ROI run and one patch run (seam de-duplicated) per tool, when the cache has them
  const pick = (runs, prefer) => [...new Set([...prefer.filter((t) => runs.some((r) => r.endsWith(t))).map((t) => runs.find((r) => r.endsWith(t))), runs[0]])].slice(0, 2);
  const runs = [...pick(baysorRuns, ['b7f1eed9', 'aa1530b9']), ...pick(prosegRuns, ['3dc382ab', '4a964e6e'])];
  for (const run of runs) {
    const tool = run.split(':')[0], tag = run.replace(':', '_');
    const ok = await selectSource(run);
    const st = await glState();
    if (!check(`1 ${run}: selected through the dropdown, the viewer shows that run's cells`, ok && st.source === run && st.n > 0 && st.n !== nXenium, J({n: st.n, status: st.status}))) continue;
    await fit();
    const modes = (await colourModes()).filter((m) => ['cluster', 'transcript_counts', 'cell_area', 'nucleus_area', 'niche'].includes(m) || m.startsWith('cell_type:'));
    for (const mode of modes) {
      await setProp('color-by', {value: mode});
      await wait(600);
      const r = await compareColours(`1_${tag}_${mode.replace(/[^a-z0-9]+/gi, '-')}`, mode === 'cluster' || mode.startsWith('cell_type:rctd'));
      check(`1 ${run}: colour by ${mode} agrees with Plotly for every cell`, coloursAgree(r), brief(r));
    }
    // outlines of the run next to the Xenium ones, zoomed in
    await setProp('color-by', {value: 'cluster'});
    await zoomToCells(120, 72);
    await setProp('boundary-toggles', {value: ['cell', 'nucleus', tool]});
    await wait(1500);
    await idle();
    await shot(`1_${tag}_outlines_webgl.png`);
    const vp = await V(() => window.XeniumViewer.instance.getViewport());
    const b = await withPlotly(() => V(([kinds, names, vp]) => Object.fromEntries(kinds.map((k) => [k, window.__rr.cmpBoundaries(k, names[k], vp)])), [['cell', 'nucleus', tool], BOUNDARY_TRACE, vp]), `1_${tag}_outlines_plotly.png`);
    const own = b[tool];
    check(`1 ${run}: ${tool} outlines are the run's polygons (every ring of the Plotly trace, vertex for vertex)`,
      own.plotly > 0 && own.common === own.plotly && own.glInView > 0 && own.glInViewMissing === 0 && own.gl >= own.plotly, J(own));
    check(`1 ${run}: Xenium cell and nucleus outlines stay the original ones under the ${tool} source`,
      ['cell', 'nucleus'].every((k) => b[k].plotly > 0 && b[k].common === b[k].plotly), J({cell: b.cell, nucleus: b.nucleus}));
    await setProp('boundary-toggles', {value: []});
    await wait(500);
  }
  await selectSource('xenium');
  await fit();
}

// ── 2. Active Counts (SPLIT-corrected) and SpaGE-imputed genes ─────────────────
const glValues = () => V(() => { const c = window.__rr.viewer()._colors; return c && c.values ? Array.from(c.values) : null; });
const nDiffer = (a, b) => { if (!a || !b || a.length !== b.length) return -1; let n = 0; for (let i = 0; i < a.length; i++) if (!(Math.abs(a[i] - b[i]) <= 1e-6) && !(Number.isNaN(a[i]) && Number.isNaN(b[i]))) n++; return n; };
const setCounts = async (mode) => {
  if ((await prop('counts-mode-store', 'data')) === mode) return true;
  if (!(await hasOption('counts-mode', mode))) return false;
  await setProp('counts-mode', {value: mode});
  await until(async () => (await prop('counts-mode-store', 'data')) === mode, 20000);
  await wait(800);
  await idle();
  return true;
};
const coloursReq = (list) => list.filter((u) => u.startsWith('/viewer/colors.bin')).map((u) => Object.fromEntries(new URL(base + u).searchParams));
async function countsChecks(label, tag) {
  const genes = (await options('gene-selector')).map((o) => String(o.value));
  const imputed = genes.filter((g) => / \[imp\]$/.test(g));
  const corrImp = genes.filter((g) => / \[corr\+imp\]$/.test(g));
  // a measured gene with counts in this source (in the corrected matrix too, when there is one)
  const hasCorr = await hasOption('counts-mode', 'corrected');
  const cand = [...['KRT14', 'KRT5', 'KRT10', 'COL1A1', 'COL1A2', 'VIM', 'PTPRC', 'ACTA2', 'DCN', 'LUM'].filter((g) => genes.includes(g)),
    ...genes.filter((g) => !g.includes('[')).slice(0, 40)];
  const panel = (await V(async ([src, cand, corr]) => {
    for (const g of cand) {
      let ok = true;
      for (const c of corr) {
        const r = await fetch(`/viewer/colors.bin?source=${encodeURIComponent(src)}&mode=gene&gene=${encodeURIComponent(g)}&corrected=${c}`);
        const a = window.XeniumViewer.util.parseBin(await r.arrayBuffer()).arrays.values;
        let nz = 0;
        for (let i = 0; i < a.length; i++) if (a[i] > 0) nz++;
        if (nz < 50) { ok = false; break; }
      }
      if (ok) return g;
    }
    return null;
  }, [await prop('seg-source'), cand, hasCorr ? [0, 1] : [0]])) || (await prop('gene-selector'));
  note(`${label}: panel gene ${panel}; imputed genes offered: ${J(imputed.slice(0, 4))}${imputed.length > 4 ? '…' : ''} ${J(corrImp.slice(0, 3))}`);
  await setCounts('original');
  await setProp('color-by', {value: 'gene'});
  await setProp('gene-selector', {value: panel});
  await wait(800);
  let r = await compareColours(`2_${tag}_gene_original`);
  check(`2 ${label}: ${panel}, original counts, agrees with Plotly`, coloursAgree(r), brief(r));
  const orig = await glValues();
  if (!(await hasOption('counts-mode', 'corrected'))) {
    note(`${label}: no SPLIT-corrected counts stored for this source`);
  } else {
    let n = reqs.length;
    await setCounts('corrected');
    const got = reqs.slice(n);
    const q = coloursReq(got);
    const corr = await glValues();
    check(`2 ${label}: switching Active Counts to corrected refetches the colours (corrected=1), not the cells`,
      q.length >= 1 && q.every((p) => p.corrected === '1' && p.mode === 'gene') && !got.some((u) => u.startsWith('/viewer/cells.bin')), J({colours: q.length, cells: got.filter((u) => u.startsWith('/viewer/cells.bin')).length}));
    r = await compareColours(`2_${tag}_gene_corrected`);
    check(`2 ${label}: ${panel}, SPLIT-corrected counts, agrees with Plotly`, coloursAgree(r), brief(r));
    check(`2 ${label}: corrected values differ from the original ones`, nDiffer(orig, corr) > 0, `${nDiffer(orig, corr)} of ${orig ? orig.length : 0} cells`);
    await setProp('color-by', {value: 'cluster'});
    await wait(800);
    r = await compareColours(`2_${tag}_cluster_corrected`, false);
    check(`2 ${label}: clusters under corrected counts agree with Plotly`, coloursAgree(r), brief(r));
    for (const g of [imputed[0], corrImp[0]].filter(Boolean)) {
      await setProp('color-by', {value: 'gene'});
      await setProp('gene-selector', {value: g});
      await wait(800);
      r = await compareColours(`2_${tag}_${g.replace(/[^a-z0-9]+/gi, '-')}_corrected`, false);
      check(`2 ${label}: ${g} under corrected counts agrees with Plotly`, coloursAgree(r), brief(r));
    }
    n = reqs.length;
    await setCounts('original');
    check(`2 ${label}: switching back to original refetches the colours (corrected=0)`, coloursReq(reqs.slice(n)).some((p) => p.corrected === '0'));
  }
  for (const g of imputed.slice(0, 2)) {
    await setProp('color-by', {value: 'gene'});
    await setProp('gene-selector', {value: g});
    await wait(800);
    r = await compareColours(`2_${tag}_${g.replace(/[^a-z0-9]+/gi, '-')}`, g === imputed[0]);
    const vals = await glValues();
    check(`2 ${label}: SpaGE-imputed ${g} agrees with Plotly`, coloursAgree(r) && vals.some((v) => v > 0), brief(r));
  }
  await setProp('gene-selector', {value: panel});
  await setProp('color-by', {value: 'cluster'});
  await wait(500);
}
if (want('counts')) {
  await selectSource('xenium');
  await fit();
  await countsChecks('Xenium', 'xenium');
  const run = baysorRuns.find((r) => r.endsWith('b7f1eed9')) || baysorRuns[0];
  if (await selectSource(run)) {
    await fit();
    await countsChecks(run, run.replace(':', '_'));
  }
  await selectSource('xenium');
  await fit();
}

// ── 3. RCTD weight pies ─────────────────────────────────────────────────────────
if (want('pies')) {
  await selectSource('xenium');
  const modes = (await colourModes()).filter((m) => m.startsWith('cell_type:'));
  let withPies = 0;
  for (const mode of modes) {
    await setProp('color-by', {value: mode});
    await wait(800);
    await idle();
    const probe = await V(async (m) => { const vp = {x0: 0, y0: 0, x1: 1, y1: 1};
      const r = await fetch(`/viewer/extra/pies.bin?source=xenium&mode=${encodeURIComponent(m)}&x0=0&y0=0&x1=1&y1=1`);
      return window.XeniumViewer.util.parseBin(await r.arrayBuffer()).header.available; }, mode);
    if (!probe) { note(`${mode}: no RCTD weights stored`); continue; }
    withPies += 1;
    await fit();
    const r = await compareColours(`3_${mode.replace(/[^a-z0-9]+/gi, '-')}_all`, false);
    check(`3 ${mode}: cell-type colours agree with Plotly for every cell`, coloursAgree(r), brief(r));
    const n0 = reqs.length;
    await zoomToCells(150, 90, 0.72, 0.45);
    await wait(1500);
    await idle();
    await shot(`3_${mode.replace(/[^a-z0-9]+/gi, '-')}_pies_webgl.png`);
    const vp = await V(() => window.XeniumViewer.instance.getViewport());
    const p = await withPlotly(() => V((vp) => window.__rr.cmpPies(vp), vp), `3_${mode.replace(/[^a-z0-9]+/gi, '-')}_pies_plotly.png`);
    // the wedges of every cell in the WebGL viewport, each with the Plotly wedge's geometry
    check(`3 ${mode}: RCTD weight pies of the cells in view equal the Plotly wedges`,
      p.gl > 20 && p.common === p.gl && p.plotlyInViewMissing === 0 && p.maxEnd < 0.01 && p.maxRadius < 0.01 && p.largeDiff === 0 && p.glAlpha === 217 && p.plotlyOpacity === 0.85 &&
      reqs.slice(n0).some((u) => u.includes('/viewer/extra/pies.bin')), J(p));
    // the weights themselves: normalised rows of the stored RCTD weights
    const w = await V(async ([m, vp]) => {
      const r = await fetch(`/viewer/extra/pies.bin?source=xenium&mode=${encodeURIComponent(m)}&x0=${vp.x0}&y0=${vp.y0}&x1=${vp.x1}&y1=${vp.y1}`);
      const {header, arrays} = window.XeniumViewer.util.parseBin(await r.arrayBuffer());
      const k = header.types.length; let maxDev = 0;
      for (let i = 0; i < header.n; i++) { let s = 0; for (let j = 0; j < k; j++) s += arrays.frac[i * k + j]; maxDev = Math.max(maxDev, Math.abs(s - 1)); }
      return {cells: header.n, types: k, maxDev, radius: header.radius};
    }, [mode, vp]);
    check(`3 ${mode}: one pie per cell in view, weights sum to 1`, w.cells === p.cells && w.maxDev < 1e-4 && w.radius === 8, J(w));
    await fit();
    await wait(1500);
    await idle();
    check(`3 ${mode}: no pies when more than PIE_THRESHOLD cells are in view`, (await V(() => window.__rr.viewer().state.overlays.polygons.filter((d) => d.width === 0).length)) === 0);
  }
  // (the app stores weights for RCTD "full" mode only; doublet / multi runs have labels alone)
  if (!withPies) console.log('SKIP  3 RCTD weight pies: no annotation in this cache has stored weights (needs an RCTD run in "full" mode)');
  // a resegmentation source has no pies in either renderer
  const run = baysorRuns.find((r) => r.endsWith('b7f1eed9')) || baysorRuns[0];
  if (modes.includes('cell_type:rctd') && (await selectSource(run))) {
    await setProp('color-by', {value: 'cell_type:rctd'});
    await zoomToCells(120, 72);
    await wait(1200);
    await idle();
    const glPies = await V(() => window.__rr.viewer().state.overlays.polygons.filter((d) => d.width === 0).length);
    const plPies = await withPlotly(() => V(() => window.__rr.cmpPies(null).plotly));
    check(`3 ${run}: no pies under a resegmentation source in either renderer`, glPies === 0 && plPies === 0, J({glPies, plPies}));
    await selectSource('xenium');
  }
  await setProp('color-by', {value: 'cluster'});
  await fit();
}

// ── 4. SpatialData tissue ROI and patch grid ───────────────────────────────────
// The outlines exist in the app once "Prepare Patches" (resegmentation dialog) has run. With
// the settings recorded in the cache store the app restores the stored ROI and patches
// instead of computing them again; without such a record this section is skipped, so that
// a test never recomputes (and rewrites) the patches of the cache.
const sdataRecord = () => {
  const store = readdirSync(CACHE_DIR).find((f) => /^spatialdata_[0-9a-f]+\.zarr$/.test(f));
  const file = store && resolve(CACHE_DIR, store, '.xe_patch_params.json');
  try { return file && existsSync(file) ? JSON.parse(readFileSync(file, 'utf8')) : null; } catch { return null; }
};
if (want('sdata') && !sdataRecord()) note('sdata checks skipped: the cache has no SpatialData store with recorded patches');
if (want('sdata') && sdataRecord()) {
  const rec = sdataRecord();
  await selectSource('xenium');
  await fit();
  const polys = () => V(() => window.__rr.viewer().state.overlays.polygons.filter((p) => p.width !== 0 && !p.label).map((p) => ({color: p.color, n: p.points.length, pts: p.points})));   // colours: '#52b788' ROI, '#ffaa00' patches
  await setProp('sdata-show-roi', {value: []});
  await setProp('sdata-show-patches', {value: []});
  await wait(1200);
  const base0 = (await polys()).length;
  const done0 = await prop('sdata-done', 'data');
  await V(() => document.querySelector('#reseg-modal-open-btn').click());
  await wait(1500);
  await setProp('sdata-qv', {value: rec.qv_threshold === null ? 20 : rec.qv_threshold});
  await setProp('sdata-patch-width', {value: rec.patch_width});
  await setProp('sdata-patch-min-tx', {value: rec.min_transcripts});
  await wait(500);
  await V(() => document.querySelector('#sdata-prepare-btn').click());
  // (patches already in memory: the button ticks the two outline boxes at once)
  const ready = await until(async () => (await prop('sdata-done', 'data')) !== done0 || ((await prop('sdata-show-patches')) || []).includes('yes'), 900000, 2000);
  const confirm = await V(() => document.querySelector('#reseg-patch-confirm-msg')?.innerText || '');
  await V(() => { const b = document.querySelector('#reseg-modal-close-btn'); if (b) b.click(); });
  check('4 "Prepare Patches" restores the stored tissue ROI and patches', !!ready && new RegExp(`${rec.n_patches} patches ready`).test(confirm), J({confirm, recorded: rec.n_patches}));
  await setProp('sdata-show-roi', {value: ['yes']});
  await setProp('sdata-show-patches', {value: ['yes']});
  await wait(2500);
  await idle();
  await shot('4_sdata_webgl.png');
  const gl = await polys();
  const pl = await withPlotly(() => V(() => window.__rr.plotlyOutlines()), '4_sdata_plotly.png');
  const bbox = (pts) => { const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]); return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]; };
  const sameRing = (a, b) => a.length === b.length && a.every((p, i) => Math.hypot(p[0] - b[i][0], p[1] - b[i][1]) < 0.01);
  const sameBox = (a, b) => bbox(a).every((v, i) => Math.abs(v - bbox(b)[i]) < 0.01);
  const glRoi = gl.filter((p) => p.color === '#52b788'), glPatch = gl.filter((p) => p.color === '#ffaa00');
  const plRoi = pl.filter((p) => p.color === '82,183,136'), plPatch = pl.filter((p) => p.color === '255,170,0');
  const drop = (pts) => (pts.length > 1 && pts[0][0] === pts[pts.length - 1][0] && pts[0][1] === pts[pts.length - 1][1] ? pts.slice(0, -1) : pts);
  const roiOk = glRoi.length > 0 && glRoi.length === plRoi.length && glRoi.every((g) => plRoi.some((p) => sameRing(drop(g.pts), drop(p.pts))));
  const patchOk = glPatch.length > 0 && glPatch.length === plPatch.length && glPatch.every((g) => plPatch.some((p) => sameBox(g.pts, p.pts)));
  check('4 tissue ROI outline equals the Plotly shape (same vertices)', roiOk, J({gl: glRoi.map((p) => p.n), plotly: plRoi.map((p) => p.n)}));
  check('4 patch grid equals the Plotly rectangles', patchOk && glPatch.length === rec.n_patches, J({gl: glPatch.length, plotly: plPatch.length, recorded: rec.n_patches}));
  note(`sdata outlines: ${glRoi.length} ROI polygon(s), ${glPatch.length} patches (before: ${base0} outlines)`);
  await setProp('sdata-show-roi', {value: []});
  await setProp('sdata-show-patches', {value: []});
  await wait(1500);
  check('4 outlines go when unticked', (await polys()).length === base0);
}

// ── 5. clicking a cell under a resegmentation source ───────────────────────────
const clickRow = async (row) => {
  const src = await V(() => window.__rr.viewer()._cells.source);
  const target = await V(async ([i, s]) => (await fetch(`/viewer/cell/${i}?source=${encodeURIComponent(s)}`)).json(), [row, src]);
  await view(target.x - 40, target.y - 24, target.x + 40, target.y + 24);
  await idle();
  const box = await page.locator('#webgl-view').boundingBox();
  const hit = await V((c) => window.XeniumViewer.instance.project(c.x, c.y), target);
  await page.mouse.click(box.x + hit[0], box.y + hit[1]);
  const ok = await until(async () => ((await V(() => document.querySelector('#cell-info-panel')?.innerText)) || '').includes(target.id), 8000, 300);
  return {target, panel: ((await V(() => document.querySelector('#cell-info-panel')?.innerText)) || '').replace(/\n+/g, ' | ').slice(0, 220), ok: !!ok};
};
if (want('click')) {
  for (const run of [baysorRuns.find((r) => r.endsWith('b7f1eed9')) || baysorRuns[0], prosegRuns.find((r) => r.endsWith('3dc382ab')) || prosegRuns[0]]) {
    if (!(await selectSource(run))) continue;
    const tool = run.split(':')[0];
    await setProp('color-by', {value: 'transcript_counts'});
    await wait(600);
    await idle();
    const n = (await glState()).n;
    const row = Math.floor(n * 0.37);
    const tc = await V((i) => Math.expm1(window.__rr.viewer()._colors.values[i]), row);
    const c = await clickRow(row);
    await shot(`5_${run.replace(':', '_')}_click.png`);
    const counts = (c.panel.match(/Transcripts?[^0-9]*([\d,]+)/i) || [])[1];
    check(`5 ${run}: clicking a cell fills the cell-info panel with that run's cell`,
      c.ok && new RegExp(`\\[${tool}\\]`, 'i').test(c.panel), J({id: c.target.id, panel: c.panel.slice(0, 150)}));
    check(`5 ${run}: the panel's transcript count is the clicked cell's`, counts !== undefined && Math.abs(Number(counts.replace(/,/g, '')) - tc) < 0.51, J({panel: counts, colours: tc}));
  }
  await setProp('color-by', {value: 'cluster'});
  await selectSource('xenium');
  await fit();
}

// ── 6. lasso selection under a resegmentation source ───────────────────────────
if (want('lasso') && process.env.XV_REAL_ROI !== '0') {
  const run = baysorRuns.find((r) => r.endsWith('b7f1eed9')) || baysorRuns[0];
  const NAME = `rr_${Date.now() % 100000}`, CLS = 'rrtest';
  if (await selectSource(run)) {
    await setProp('color-by', {value: 'cluster'});
    await zoomToCells(260, 156);
    await setProp('webgl-tool', {value: 'lasso'});
    await wait(600);
    const box = await page.locator('#webgl-view').boundingBox();
    const px = box.x + box.width / 2, py = box.y + box.height / 2;
    await page.mouse.move(px - 220, py - 150);
    await page.mouse.down();
    for (const [dx, dy] of [[180, -190], [260, 40], [60, 200], [-240, 150], [-220, -150]]) await page.mouse.move(px + dx, py + dy, {steps: 8});
    await page.mouse.up();
    await until(() => V(() => [...document.querySelectorAll('.modal.show')].some((e) => /Save ROI/.test(e.innerText))), 10000, 300);
    await wait(500);
    // rows of the run's cells inside the drawn outline, recomputed here from the viewer's table
    const sel = await V(() => {
      const v = window.__rr.viewer(), poly = window.dash_component_api.getLayout('webgl-select').props.data.polygon;
      const inside = (x, y) => { let c = false; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        const xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
        if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c; } return c; };
      const rows = [];
      for (let i = 0; i < v._cells.n; i++) if (inside(v._cells.xy[2 * i], v._cells.xy[2 * i + 1])) rows.push(i);
      const shown = Array.from(v.state.selected || []).sort((a, b) => a - b);
      return {rows, same: shown.length === rows.length && shown.every((r, i) => r === rows[i]), n: v._cells.n,
        preview: document.querySelector('#roi-save-preview')?.innerText || ''};
    });
    await shot('6_lasso_webgl.png');
    check(`6 ${run}: the lasso selects the run's cells inside the outline and the dialog reports their number`,
      sel.rows.length >= 3 && sel.rows.length < sel.n && sel.same && sel.preview.startsWith(`${sel.rows.length.toLocaleString('en-US')} cells`), J({inside: sel.rows.length, of: sel.n, preview: sel.preview}));
    await page.fill('#roi-name-input', NAME);
    await page.fill('#roi-class-input', CLS);
    const done0 = await prop('roi-done', 'data');
    await page.click('#roi-save-btn');
    await until(async () => (await prop('roi-done', 'data')) !== done0, 20000);
    await wait(1500);
    const saved = await until(() => hasOption('color-by', `roi_${CLS}`), 15000);
    check(`6 ${run}: saving adds the ROI class to Color By`, !!saved);
    if (saved) {
      await setProp('color-by', {value: `roi_${CLS}`});
      await wait(800);
      await idle();
      const labelled = await V((name) => { const c = window.__rr.viewer()._colors; const k = c.header.categories.indexOf(name); const rows = [];
        for (let i = 0; i < c.codes.length; i++) if (c.codes[i] === k) rows.push(i); return rows; }, NAME);
      // (a cell exactly on the outline may fall on either side of the server's polygon test)
      const a = new Set(sel.rows), b = new Set(labelled);
      const onlyA = sel.rows.filter((r) => !b.has(r)).length, onlyB = labelled.filter((r) => !a.has(r)).length;
      check(`6 ${run}: the saved ROI labels exactly the cells the lasso selected`, labelled.length > 0 && onlyA === 0 && onlyB === 0, J({selected: sel.rows.length, labelled: labelled.length, onlyA, onlyB}));
      let r = await compareColours(`6_roi_${run.replace(':', '_')}`);
      check(`6 ${run}: colour by the ROI class agrees with Plotly`, coloursAgree(r), brief(r));
      const ov = await withPlotly(() => V(() => window.__rr.plotlyOutlines().filter((s) => s.type === 'path').map((s) => s.pts)));
      const glRoi = await V((name) => window.__rr.viewer().state.overlays.polygons.filter((p) => (p.label || '').startsWith(name)).map((p) => p.points), NAME);
      const close = (p, q) => { const d = (pts) => (pts.length > 1 && pts[0][0] === pts[pts.length - 1][0] && pts[0][1] === pts[pts.length - 1][1] ? pts.slice(0, -1) : pts);
        const x = d(p), y = d(q); return x.length === y.length && x.every((pt, i) => Math.hypot(pt[0] - y[i][0], pt[1] - y[i][1]) < 0.01); };
      check(`6 ${run}: the ROI outline drawn in WebGL is the one Plotly draws`, glRoi.length === 1 && ov.some((p) => close(p, glRoi[0])), J({gl: glRoi.map((p) => p.length), plotly: ov.map((p) => p.length)}));
      // the same ROI under the Xenium source: its cells inside the same outline
      await selectSource('xenium');
      await setProp('color-by', {value: `roi_${CLS}`});
      await wait(800);
      r = await compareColours('6_roi_xenium', false);
      check('6 Xenium source: colour by the ROI class agrees with Plotly', coloursAgree(r), brief(r));
    }
    // remove the test ROI again (ROI manager -> delete), so the cache is left as it was
    await setProp('color-by', {value: 'cluster'});
    await V(() => document.querySelector('#roi-manage-btn').click());
    await wait(1500);
    const delDone = await prop('roi-done', 'data');
    const deleted = await V((name) => { const row = [...document.querySelectorAll('#roi-list-div tr')].find((tr) => tr.innerText.includes(name));
      const btn = row && row.querySelector('button'); if (btn) btn.click(); return !!btn; }, NAME);
    await until(async () => (await prop('roi-done', 'data')) !== delDone, 20000);
    await wait(1000);
    await V(() => { const b = document.querySelector('#roi-manage-close-btn'); if (b) b.click(); });
    const left = await V(async (name) => (await (await fetch('/viewer/overlays')).json()).polygons.filter((p) => (p.label || '').startsWith(name)).length, NAME);
    check('6 the test ROI is deleted again', deleted && left === 0, J({deleted, left}));
    await wait(3000);   // let the background store update finish
    await selectSource('xenium');
    await fit();
  }
}

// ── 7. subset() while in WebGL ─────────────────────────────────────────────────
if (want('subset')) {
  const replOff = await V(() => { const e = document.querySelector('#repl-input'); return !e || e.disabled || /disabled/i.test(e.placeholder || ''); });
  if (replOff) {
    note('subset checks skipped: the in-app REPL is off (start the app with XENIUM_ENABLE_REPL=1)');
  } else {
    const repl = async (code) => { await page.fill('#repl-input', code); await page.press('#repl-input', 'Enter'); };
    const subsetTo = async (code, expectChange = true) => {
      const v0 = await prop('subset-version', 'data');
      const n = reqs.length;
      await repl(code);
      if (expectChange) await until(async () => (await prop('subset-version', 'data')) !== v0, 60000);
      await wait(1500);
      await idle();
      return reqs.slice(n);
    };
    await selectSource('xenium');
    await setProp('color-by', {value: 'cluster'});
    await fit();
    const method = await prop('cluster-method');
    let got = await subsetTo(`subset(cluster=[1, 2], method=${J(method)})`);
    let st = await glState();
    check('7 subset(): the WebGL view reloads cells and colours for the subset', st.n > 0 && st.n < nXenium && got.some((u) => u.startsWith('/viewer/cells.bin')) && got.some((u) => u.startsWith('/viewer/colors.bin')), J({n: st.n, of: nXenium}));
    let r = await compareColours('7_subset_cluster');
    check('7 subset: clusters agree with Plotly for every remaining cell', coloursAgree(r) && r.cmp.categories === 2, brief(r));
    await setProp('color-by', {value: 'gene'});
    await wait(800);
    r = await compareColours('7_subset_gene', false);
    check('7 subset: gene expression agrees with Plotly', coloursAgree(r), brief(r));
    // outlines of the subset come from the in-memory polygons (not the bundle's tile store)
    await setProp('color-by', {value: 'cluster'});
    await zoomToCells(150, 90, 0.72, 0.45);
    await setProp('boundary-toggles', {value: ['cell']});
    await wait(1500);
    await idle();
    await shot('7_subset_outlines_webgl.png');
    const vp = await V(() => window.XeniumViewer.instance.getViewport());
    const b = await withPlotly(() => V(([name, vp]) => window.__rr.cmpBoundaries('cell', name, vp), [BOUNDARY_TRACE.cell, vp]), '7_subset_outlines_plotly.png');
    check('7 subset: cell outlines are those of the remaining cells, as in Plotly', b.plotly > 0 && b.common === b.plotly && b.glInViewMissing === 0, J(b));
    await setProp('boundary-toggles', {value: []});
    got = await subsetTo('unsubset()');
    st = await glState();
    check('7 unsubset(): all cells are back', st.n === nXenium && got.some((u) => u.startsWith('/viewer/cells.bin')), J({n: st.n}));
    await fit();
  }
}

// ── 8. a second sample beside the first ────────────────────────────────────────
if (want('multi')) {
  const extras = EXTRA_DIR.split(',').filter((d) => existsSync(resolve(d, 'experiment.xenium')));
  if (!extras.length) note(`multi-sample checks skipped: no second bundle at ${EXTRA_DIR}`);
  for (const dir of extras) {
    const tag = dir.replace(/\/+$/, '').split('/').slice(-2).join('_').replace(/[^a-z0-9]+/gi, '-');
    await selectSource('xenium');
    await setProp('color-by', {value: 'cluster'});
    await fit();
    const v0 = await prop('extra-datasets-version', 'data');
    await V(() => document.querySelector('#sample-modal-open-btn').click());
    await wait(1200);
    await page.fill('#sample-custom-path', dir);
    await page.click('#sample-add-btn');
    const added = await until(async () => (await prop('extra-datasets-version', 'data')) !== v0, 300000, 1000);
    await wait(2500);
    await idle();
    const st = await glState();
    const sample = await V(() => document.querySelector('#sample-load-status')?.innerText || '');
    if (!check(`8 ${tag}: added through the sample dialog, the viewer holds both samples`, !!added && st.n > nXenium, J({n: st.n, status: sample}))) continue;
    await fit();
    await wait(800);
    for (const mode of ['cluster', 'transcript_counts', 'gene', 'cell_type:seurat']) {
      if (!(await hasOption('color-by', mode))) continue;
      await setProp('color-by', {value: mode});
      await wait(800);
      const r = await compareColours(`8_${tag}_${mode.replace(/[^a-z0-9]+/gi, '-')}`, mode === 'cluster');
      if (plotlyFailed(r, `8 ${tag}: ${mode}`)) {
        // (seen with a second sample that has no cluster sets: KeyError 'S1:C0' in make_spatial_fig)
        check(`8 ${tag}: ${mode}: the WebGL view still draws both samples`, r.gl.n === st.n && !!r.gl.kind, J({n: r.gl.n, kind: r.gl.kind}));
        continue;
      }
      check(`8 ${tag}: ${mode} agrees with Plotly for the cells of both samples (count and positions)`, coloursAgree(r) && r.cmp.gl === st.n, brief(r));
    }
    // separator between the samples and the id / position of a cell of the second one
    const sep = await V(() => window.__rr.viewer().state.overlays.lines.map((l) => l.from[0]));
    const plSep = await withPlotly(() => V(() => window.__rr.plotlyOutlines().filter((s) => s.type === 'line').map((s) => s.pts[0][0])));
    check(`8 ${tag}: the separator is where Plotly draws it`, sep.length === 1 && plSep.length === 1 && Math.abs(sep[0] - plSep[0]) < 0.01, J({gl: sep, plotly: plSep}));
    // boundaries stay those of the first sample, pies are off, as in the Plotly view
    await setProp('color-by', {value: 'cluster'});
    const c = await clickRow(nXenium + Math.floor((st.n - nXenium) / 2));
    note(`click on a cell of the second sample: ${J({id: c.target.id, ok: c.ok, panel: c.panel.slice(0, 80)})}`);
    await fit();
    // a resegmentation source shows that run only, in both renderers
    const run = baysorRuns.find((r) => r.endsWith('b7f1eed9')) || baysorRuns[0];
    if (await selectSource(run)) {
      const r = await compareColours(`8_${tag}_under_${run.replace(':', '_')}`, false);
      check(`8 ${tag}: under ${run} only the run's cells are shown, as in Plotly`, coloursAgree(r) && r.cmp.gl < nXenium, brief(r));
      await selectSource('xenium');
    }
    const v1 = await prop('extra-datasets-version', 'data');
    await V(() => document.querySelector('#remove-extra-samples-btn').click());
    await until(async () => (await prop('extra-datasets-version', 'data')) !== v1, 30000);
    await wait(2500);
    await idle();
    const back = await glState();
    check(`8 ${tag}: removing the sample restores the single-sample view`, back.n === nXenium && (await V(() => window.__rr.viewer().state.overlays.lines.length)) === 0, J({n: back.n}));
    await fit();
  }
}

// ── 9. the UMAP panel (second viewer, /umap/viewer/*) ───────────────────────────
if (want('umap')) {
  await selectSource('xenium');
  await setCounts('original');
  await setProp('color-by', {value: 'cluster'});
  const wasOpen = await V(() => getComputedStyle(document.querySelector('#umap-panel')).display !== 'none');
  if (!wasOpen) await V(() => document.querySelector('#umap-toggle').click());
  await until(() => V(() => !!(window.XeniumViewer.umapInstance && window.XeniumViewer.umapInstance._viewer._cells)), 60000);
  await wait(1500);
  const umapXY = () => V(() => Array.from(window.XeniumViewer.umapInstance._viewer._cells.xy.slice(0, 4000)));
  let r = await compareUmap('9_umap_xenium_cluster');
  check('9 Xenium: UMAP positions and cluster colours agree with Plotly', coloursAgree(r, false), brief(r));
  if (await hasOption('counts-mode', 'corrected')) {
    const xy0 = await umapXY();
    await setCounts('corrected');
    await V(() => window.XeniumViewer.umapInstance.whenIdle());
    const xy1 = await umapXY();
    r = await compareUmap('9_umap_xenium_cluster_corrected');
    check('9 Xenium, corrected counts: the UMAP viewer shows the embedding Plotly shows', coloursAgree(r, false), brief(r));
    check('9 Xenium: Active Counts switches the UMAP viewer to the corrected embedding', nDiffer(xy0, xy1) > 0, `${nDiffer(xy0, xy1)} of the first 4000 coordinates differ`);
    await setCounts('original');
  }
  const run = baysorRuns.find((r) => r.endsWith('b7f1eed9')) || baysorRuns[0];
  if (await selectSource(run)) {
    await V(() => window.XeniumViewer.umapInstance.whenIdle());
    for (const mode of ['cluster', 'cell_type:seurat', 'transcript_counts']) {
      if (!(await hasOption('color-by', mode))) continue;
      await setProp('color-by', {value: mode});
      await wait(800);
      r = await compareUmap(mode === 'cluster' ? `9_umap_${run.replace(':', '_')}_cluster` : null);
      check(`9 ${run}: UMAP coloured by ${mode} agrees with Plotly`, coloursAgree(r, false), brief(r));
    }
    if (await hasOption('counts-mode', 'corrected')) {
      await setProp('color-by', {value: 'cluster'});
      await setCounts('corrected');
      await V(() => window.XeniumViewer.umapInstance.whenIdle());
      r = await compareUmap(`9_umap_${run.replace(':', '_')}_cluster_corrected`);
      check(`9 ${run}, corrected counts: UMAP embedding and clusters agree with Plotly`, coloursAgree(r, false), brief(r));
      await setCounts('original');
    }
    await selectSource('xenium');
  }
  await setProp('color-by', {value: 'cluster'});
  if (!wasOpen) await V(() => document.querySelector('#umap-toggle').click());
  await wait(1000);
}

check('no page errors', logs.length === 0, J(logs.slice(0, 6)));
console.log(failures === 0 ? 'ALL PASS' : `FAILURES: ${failures}`);
await browser.close();
stopApp();
process.exit(failures === 0 ? 0 : 1);
