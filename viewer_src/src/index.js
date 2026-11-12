// Xenium spatial viewer core (deck.gl 9, framework-free).
// The bundle only defines window.XeniumViewer; nothing touches the DOM until mount().
import {Deck, OrthographicView} from '@deck.gl/core';
import {ScatterplotLayer, PathLayer, PolygonLayer, LineLayer} from '@deck.gl/layers';
import {parseBin, packBin} from './wire.js';
import {
  colormapCss, dimUnselected, fillCategorical, fillContinuous, fillUniform, hexToRgb,
  MISSING_RGB, TX_PALETTE,
} from './colormap.js';
import {pointsInPolygon, PointGrid} from './spatial.js';
import {RegionLoader} from './region.js';

const VERSION = '0.1.0';
const SETTLE_MS = 150;
const RETRY_MS = 2000; // delay before a failed cells load is retried
const CELLS_RETRIES = 2; // automatic retries; after that the next state change retries
const HOVER_ID_MS = 200; // hover dwell before the cell id is looked up
const CELL_RADIUS_UM = 3; // world-space radius; clamped in pixels by pointSize (see _cellsLayer)
const NESTED = ['color', 'boundaries', 'transcripts', 'overlays'];
// One polygon set per kind, each with its own toggle; colours as in the Plotly view.
// 'cell' / 'nucleus' are always the original Xenium sets, whatever state.source is.
const BOUNDARY_STYLE = {
  cell: {color: [0, 212, 255, 220], label: 'cell boundaries'},
  nucleus: {color: [255, 159, 67, 220], label: 'nucleus boundaries'},
  proseg: {color: [46, 213, 115, 220], label: 'Proseg boundaries'},
  baysor: {color: [162, 155, 254, 220], label: 'Baysor boundaries'},
};
const BOUNDARY_KINDS = Object.keys(BOUNDARY_STYLE);

const CSS = `
.xv-wrap{position:absolute;inset:0;overflow:hidden;font:12px/1.35 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;color:#e6edf3;user-select:none;-webkit-user-select:none}
.xv-wrap canvas.xv-canvas{position:absolute;left:0;top:0;width:100%;height:100%;outline:none}
.xv-draw,.xv-labels{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;overflow:hidden}
.xv-label{position:absolute;transform:translate(-50%,-100%);padding:0 4px;white-space:nowrap;text-shadow:0 0 3px #000,0 0 3px #000;font-weight:600}
.xv-legend{position:absolute;top:8px;right:8px;max-width:220px;max-height:calc(100% - 16px);display:flex;flex-direction:column;gap:6px;pointer-events:none}
.xv-box{background:rgba(13,17,23,.82);border:1px solid rgba(255,255,255,.14);border-radius:6px;padding:6px 8px;pointer-events:auto;min-height:0;display:flex;flex-direction:column}
.xv-title{font-weight:600;margin-bottom:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.xv-bar{height:10px;width:180px;border-radius:2px}
.xv-range{display:flex;justify-content:space-between;margin-top:2px;font-variant-numeric:tabular-nums}
.xv-list{overflow-y:auto;min-height:0;max-height:320px}
.xv-item{display:flex;align-items:center;gap:6px;padding:1px 0;cursor:pointer;white-space:nowrap}
.xv-item.xv-static{cursor:default}
.xv-item.xv-off{opacity:.38;text-decoration:line-through}
.xv-swatch{flex:0 0 10px;width:10px;height:10px;border-radius:50%}
.xv-name{overflow:hidden;text-overflow:ellipsis}
.xv-tip{position:absolute;pointer-events:none;background:rgba(13,17,23,.9);border:1px solid rgba(255,255,255,.2);border-radius:4px;padding:1px 6px;display:none;white-space:nowrap}
`;

let cssInjected = false;
function injectCss() {
  if (cssInjected) return;
  const style = document.createElement('style');
  style.setAttribute('data-xenium-viewer', '');
  style.textContent = CSS;
  document.head.appendChild(style);
  cssInjected = true;
}

const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

const fmt = (v) => {
  if (v === null || v === undefined || !Number.isFinite(v)) return '';
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.01 || a >= 1e5)) return v.toExponential(2);
  return String(Math.round(v * 100) / 100);
};

const sameList = (a, b) => {
  if (a === b) return true;
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
};

function defaultState() {
  return {
    source: 'xenium',
    dataVersion: 0,
    color: {mode: null, method: null, gene: null, corrected: false, min: null, max: null, version: 0},
    colormap: 'viridis',
    hiddenCategories: [],
    pointSize: 3,
    opacity: 0.8,
    boundaries: {cell: false, nucleus: false, proseg: false, baysor: false},
    transcripts: {genes: [], qv: 20, size: 2},
    tool: 'pan',
    selected: null,
    overlays: {polygons: [], lines: [], points: []},
    background: '#0d1117',
  };
}

class Viewer {
  constructor(element, opts) {
    this.opts = opts || {};
    this.baseUrl = (this.opts.baseUrl || '').replace(/\/$/, '');
    this._fetch = this.opts.fetch || ((url, init) => window.fetch(url, init));
    this.state = defaultState();
    this.stats = {mountedAt: performance.now()};
    this.requestLog = []; // paths of every request issued, in order (tests / debugging)
    this._destroyed = false;
    this._pending = 0;
    this._status = {};
    this._statusText = '';
    this._layers = {};
    this._dirty = {};
    this._cells = null; // {n, xy, bounds, key, posAttr, grid}
    this._colors = null; // {key, header, values|codes}
    this._rgba = null;
    this._colAttr = null;
    this._cellData = null;
    this._cellsSeq = 0;
    this._colorSeq = 0;
    this._cellsKey = null;
    this._colorKey = null;
    this._viewSet = false; // has the view been fitted / set at least once?
    this._pendingViewport = null;
    this._extraFn = null;
    this._afterRender = [];
    this._draw = null;

    this._buildDom(element);
    this.viewState = {target: [0, 0, 0], zoom: 0, minZoom: -12, maxZoom: 8};

    const mkLoader = (name) => new RegionLoader(
      (url, signal) => this._fetchBin(url, signal),
      () => this._onRegionChange(name)
    );
    this._loaders = {tx: mkLoader('tx')};
    for (const kind of BOUNDARY_KINDS) this._loaders[kind] = mkLoader(kind);

    this.deck = new Deck({
      canvas: this.canvas,
      width: '100%',
      height: '100%',
      views: new OrthographicView({id: 'ortho', flipY: true}),
      viewState: this.viewState,
      controller: this._controller(),
      layers: [],
      parameters: {depthWriteEnabled: false, depthCompare: 'always'},
      pickingRadius: 0,
      getCursor: ({isDragging}) => (this.state.tool !== 'pan' ? 'crosshair' : isDragging ? 'grabbing' : 'grab'),
      onViewStateChange: ({viewState}) => this._setViewState(viewState),
      onAfterRender: () => this._onAfterRender(),
      onDeviceInitialized: () => {
        this._deviceReady = true;
      },
      onError: (err) => {
        console.error('[XeniumViewer]', err);
        // Own key: a successful data load must not hide that nothing can be drawn.
        this._setStatus('render', `render error: ${(err && err.message) || err}`);
        // No WebGL2 device could be created at all: nothing will ever be drawn here, the
        // host can fall back to another renderer.
        if (!this._deviceReady && !this._fatalSent && this.opts.onFatal) {
          this._fatalSent = true;
          this.opts.onFatal('the WebGL2 context could not be created');   // details: console
        }
      },
    });

    this._bindEvents();
    this._applyBackground();
    this._scheduleSync(['*']);
  }

  // ── DOM ─────────────────────────────────────────────────────────────────────
  _buildDom(element) {
    injectCss();
    this.root = element;
    if (getComputedStyle(element).position === 'static') element.style.position = 'relative';
    const wrap = (this.wrap = el('div', 'xv-wrap'));
    this.canvas = el('canvas', 'xv-canvas');
    const ns = 'http://www.w3.org/2000/svg';
    this.svg = document.createElementNS(ns, 'svg');
    this.svg.setAttribute('class', 'xv-draw');
    this.drawPath = document.createElementNS(ns, 'path');
    this.drawPath.setAttribute('fill', 'rgba(255,255,255,0.12)');
    this.drawPath.setAttribute('stroke', '#ffffff');
    this.drawPath.setAttribute('stroke-width', '1.5');
    this.drawPath.setAttribute('stroke-dasharray', '5 3');
    this.svg.appendChild(this.drawPath);
    this.labels = el('div', 'xv-labels');
    this.legend = el('div', 'xv-legend');
    this.tip = el('div', 'xv-tip');
    wrap.append(this.canvas, this.svg, this.labels, this.legend, this.tip);
    element.appendChild(wrap);
  }

  _size() {
    return [this.wrap.clientWidth, this.wrap.clientHeight];
  }

  _applyBackground() {
    this.wrap.style.background = this.state.background || 'transparent';
  }

  // ── coordinates ─────────────────────────────────────────────────────────────
  // World units are µm with y growing downward (image rows); screen = (world - target) * 2^zoom + size / 2.
  _scale() {
    return Math.pow(2, this.viewState.zoom);
  }

  _toWorld(sx, sy) {
    const [w, h] = this._size();
    const s = this._scale();
    const t = this.viewState.target;
    return [t[0] + (sx - w / 2) / s, t[1] + (sy - h / 2) / s];
  }

  _toScreen(x, y) {
    const [w, h] = this._size();
    const s = this._scale();
    const t = this.viewState.target;
    return [(x - t[0]) * s + w / 2, (y - t[1]) * s + h / 2];
  }

  getViewport() {
    const [w, h] = this._size();
    if (!w || !h) return null;
    const [x0, y0] = this._toWorld(0, 0);
    const [x1, y1] = this._toWorld(w, h);
    return {x0, y0, x1, y1};
  }

  setViewport(vp) {
    if (!vp || ![vp.x0, vp.y0, vp.x1, vp.y1].every(Number.isFinite)) return;
    if (!(vp.x1 > vp.x0) || !(vp.y1 > vp.y0)) return;
    const [w, h] = this._size();
    this._viewSet = true;
    if (!w || !h) {
      this._pendingViewport = vp; // applied once the container has a size
      return;
    }
    this._pendingViewport = null;
    const fit = Math.log2(Math.min(w / (vp.x1 - vp.x0), h / (vp.y1 - vp.y0)));
    const zoom = Math.max(this.viewState.minZoom, Math.min(this.viewState.maxZoom, fit));
    this._setViewState({...this.viewState, target: [(vp.x0 + vp.x1) / 2, (vp.y0 + vp.y1) / 2, 0], zoom});
  }

  fitView() {
    const b = this._cells && this._cells.bounds;
    if (!b) {
      this._viewSet = false; // fit as soon as cells arrive
      return;
    }
    const mx = Math.max((b[2] - b[0]) * 0.03, 1);
    const my = Math.max((b[3] - b[1]) * 0.03, 1);
    this.setViewport({x0: b[0] - mx, y0: b[1] - my, x1: b[2] + mx, y1: b[3] + my});
  }

  _setViewState(vs) {
    const t = vs.target || [0, 0, 0];
    this.viewState = {
      target: [t[0], t[1], 0],
      zoom: Array.isArray(vs.zoom) ? vs.zoom[0] : vs.zoom,
      minZoom: this.viewState.minZoom,
      maxZoom: this.viewState.maxZoom,
    };
    this._render();
    this._updateLabels();
    this._redrawShape();
    this._hideTip();
    this._scheduleSettle();
  }

  _scheduleSettle() {
    clearTimeout(this._settleTimer);
    this._settleTimer = setTimeout(() => {
      this._settleTimer = null;
      if (this._destroyed) return;
      const vp = this.getViewport();
      if (!vp) return;
      this._updateRegions();
      if (this.opts.onViewport) this.opts.onViewport(vp);
    }, SETTLE_MS);
  }

  resize() {
    if (this._destroyed) return;
    const [w, h] = this._size();
    if (w && h && this._pendingViewport) this.setViewport(this._pendingViewport);
    else if (w && h && !this._viewSet && this._cells) this.fitView();
    this.deck.redraw();
    this._updateLabels();
    this._scheduleSettle();
  }

  // ── state ───────────────────────────────────────────────────────────────────
  getState() {
    const s = this.state;
    return {
      ...s,
      color: {...s.color},
      boundaries: {...s.boundaries},
      transcripts: {...s.transcripts},
      overlays: {...s.overlays},
      hiddenCategories: s.hiddenCategories.slice(),
    };
  }

  setState(partial) {
    if (!partial || this._destroyed) return;
    const changed = [];
    for (const key of Object.keys(partial)) {
      if (!(key in this.state)) continue;
      let next = partial[key];
      // null / a primitive for a nested key means "back to the defaults", never a stored null
      if (NESTED.includes(key) && (!next || typeof next !== 'object')) next = defaultState()[key];
      if (NESTED.includes(key)) {
        const cur = this.state[key];
        let diff = false;
        for (const k of Object.keys(next)) {
          const same = Array.isArray(next[k]) && k === 'genes' ? sameList(cur[k], next[k]) : cur[k] === next[k];
          if (!same) diff = true;
        }
        if (diff) {
          this.state[key] = {...cur, ...next};
          changed.push(key);
        }
      } else if (key === 'hiddenCategories') {
        const list = Array.isArray(next) ? next.slice() : [];
        if (!sameList(list, this.state.hiddenCategories)) {
          this.state.hiddenCategories = list;
          changed.push(key);
        }
      } else if (this.state[key] !== next) {
        this.state[key] = next;
        changed.push(key);
      }
    }
    if (changed.length) this._scheduleSync(changed);
  }

  // Changes are applied in a microtask so several setState calls in a row cost one pass.
  _scheduleSync(changed) {
    this._changed = (this._changed || new Set());
    for (const c of changed) this._changed.add(c);
    if (this._syncQueued) return;
    this._syncQueued = true;
    Promise.resolve().then(() => {
      this._syncQueued = false;
      if (this._destroyed) return;
      const set = this._changed;
      this._changed = null;
      this._sync(set);
    });
  }

  _sync(changed) {
    const s = this.state;
    const has = (k) => changed.has(k) || changed.has('*');
    const c = s.color;
    const cellsKey = `${s.source}|${s.dataVersion}`;
    const colorKey = c.mode
      ? [cellsKey, c.mode, c.method, c.gene, c.corrected ? 1 : 0, c.version].join('|')
      : null;

    if (cellsKey !== this._cellsKey) {
      this._cellsKey = cellsKey;
      this._colorKey = colorKey;
      this._loadCells(cellsKey, colorKey);
    } else if (colorKey !== this._colorKey) {
      this._colorKey = colorKey;
      this._loadColors(colorKey);
    } else if (has('color') || has('hiddenCategories') || has('selected') || has('colormap')) {
      this._recolor();
      if (has('color') || has('colormap')) this._buildLegend(); // range / gradient may have changed
    }
    if (has('pointSize') || has('opacity')) this._dirty.cells = true;
    if (has('overlays')) {
      this._dirty.overlays = true;
      this._buildLabels();
    }
    if (has('background')) this._applyBackground();
    if (has('tool')) {
      this.deck.setProps({controller: this._controller()});
      this._cancelShape();
    }
    if (has('transcripts')) this._dirty.tx = true;
    if (has('boundaries') || has('transcripts') || has('source') || has('dataVersion')) this._updateRegions();
    if (has('transcripts')) this._buildLegend();
    this._render();
  }

  _controller() {
    return {
      dragPan: this.state.tool === 'pan',
      dragRotate: false,
      touchRotate: false,
      doubleClickZoom: false,
      keyboard: false,
      inertia: false,
      scrollZoom: {speed: 0.01, smooth: false},
    };
  }

  // ── network ─────────────────────────────────────────────────────────────────
  _url(path, params) {
    const q = new URLSearchParams();
    for (const k of Object.keys(params)) {
      const v = params[k];
      if (v !== null && v !== undefined && v !== '') q.set(k, String(v));
    }
    return `${this.baseUrl}${path}?${q.toString()}`;
  }

  async _fetchBin(url, signal) {
    this.requestLog.push(url);
    if (this.requestLog.length > 500) this.requestLog.splice(0, 250);
    this._pending++;
    try {
      const resp = await this._fetch(url, {signal});
      if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url.split('?')[0]}`);
      return parseBin(await resp.arrayBuffer());
    } finally {
      this._pending--;
    }
  }

  _colorRequest(signal) {
    const s = this.state;
    const c = s.color;
    return this._fetchBin(this._url('/viewer/colors.bin', {
      source: s.source, mode: c.mode, method: c.method, gene: c.gene, corrected: c.corrected ? 1 : 0,
    }), signal);
  }

  async _loadCells(cellsKey, colorKey) {
    const seq = ++this._cellsSeq;
    ++this._colorSeq; // supersede any colour-only request
    if (this._cellsCtrl) this._cellsCtrl.abort();
    if (this._colorCtrl) this._colorCtrl.abort();
    this._colorPendingKey = undefined;
    clearTimeout(this._retryTimer);
    this._retryTimer = null;
    const ctrl = (this._cellsCtrl = new AbortController());
    const source = this.state.source;
    this._cellsLoading = true;
    this._setStatus('cells', 'loading cells…');
    const t0 = performance.now();
    try {
      // Coordinates and colours are requested together and committed together, so a
      // dataset / subset switch never shows new positions with old colours.
      const [cellsRes, colorsRes] = await Promise.allSettled([
        this._fetchBin(this._url('/viewer/cells.bin', {source}), ctrl.signal),
        colorKey ? this._colorRequest(ctrl.signal) : null,
      ]);
      if (seq !== this._cellsSeq || this._destroyed) return;
      if (cellsRes.status === 'rejected') throw cellsRes.reason;
      const cells = cellsRes.value;
      const t1 = performance.now();
      const xy = cells.arrays.xy || new Float32Array(0);
      const n = Math.min(cells.header.n | 0, xy.length >> 1);
      let bounds = cells.header.bounds;
      if (!bounds || bounds.some((v) => !Number.isFinite(v))) bounds = n ? boundsOf(xy, n) : [0, 0, 1, 1];
      const prevGen = this._cells ? datasetGen(this._cells.token) : null;
      this._cells = {
        n, xy, bounds, key: cellsKey, token: cells.header.token, source,
        posAttr: {value: xy, size: 2}, // same descriptor object for the lifetime of this dataset => uploaded once
        grid: new PointGrid(xy, n, bounds),
        ids: new Map(), // index -> cell id, filled lazily by the hover tooltip
      };
      const t2 = performance.now();
      this._cellsLoading = false;
      this._cellsRetries = 0;
      // The colour selection may have changed while cells.bin was in flight: only take the
      // paired response if it is still the one asked for, and never drop newer colours.
      let reloadColors = false;
      if (colorKey === this._colorKey) {
        const failed = colorsRes.status === 'rejected';
        const colors = failed ? null : colorsRes.value;
        this._colors = colors ? {key: colorKey, header: colors.header, ...colors.arrays} : null;
        if (failed) {
          console.error('[XeniumViewer] colors', colorsRes.reason);
          this._colorKey = undefined; // retried by the next _sync
          this._setStatus('colorError', `failed to load colours: ${colorsRes.reason.message || colorsRes.reason}`);
        } else {
          this._setStatus('colorError', null);
        }
      } else if (!(this._colors && this._colors.key === this._colorKey)) {
        this._colors = null;
        reloadColors = this._colorPendingKey !== this._colorKey;
      }
      this._rgba = null;
      this.stats.cells = {n, fetchMs: t1 - t0, indexMs: t2 - t1};
      this._setStatus('cells', null);
      this._setStatus('cellsError', null);
      this._recolor();
      this._buildLegend();
      const vp = this.getViewport();
      const outside = vp && (vp.x1 < bounds[0] || vp.x0 > bounds[2] || vp.y1 < bounds[1] || vp.y0 > bounds[3]);
      // A different dataset gets a fresh fit (like Plotly's per-dataset uirevision).
      const newDataset = prevGen !== null && datasetGen(cells.header.token) !== prevGen;
      if (!this._viewSet || outside || newDataset) this.fitView();
      this._render();
      if (this.stats.firstRenderMs === undefined) {
        this._afterRender.push(() => {
          this.stats.firstRenderMs = performance.now() - this.stats.mountedAt;
        });
      }
      if (reloadColors) this._loadColors(this._colorKey);
    } catch (err) {
      if ((err && err.name === 'AbortError') || seq !== this._cellsSeq) return;
      console.error('[XeniumViewer] cells', err);
      this._cellsLoading = false;
      // Forget the keys so the next state change asks again, and try again a couple of times
      // on our own (a transient 5xx while the server is busy must not leave the view empty).
      this._cellsKey = null;
      this._colorKey = null;
      ++this._colorSeq;
      if (this._colorCtrl) this._colorCtrl.abort();
      this._colorPendingKey = undefined;
      if (!this._cells || this._cells.key !== cellsKey) {
        // The table on screen belongs to another source / version: drop it, so nothing of
        // it can be picked or coloured under the new source.
        this._cells = this._colors = this._rgba = this._cellData = null;
        this._dirty.cells = true;
        this._hideTip();
        this._buildLegend();
        this._render();
      }
      this._setStatus('cells', null);
      this._setStatus('cellsError', `failed to load cells: ${err.message || err}`);
      this._cellsRetries = (this._cellsRetries || 0) + 1;
      if (this._cellsRetries <= CELLS_RETRIES) {
        this._retryTimer = setTimeout(() => {
          this._retryTimer = null;
          if (!this._destroyed) this._scheduleSync(['dataVersion']);
        }, RETRY_MS);
      }
    }
  }

  async _loadColors(colorKey) {
    const seq = ++this._colorSeq;
    if (this._colorCtrl) this._colorCtrl.abort();
    this._colorPendingKey = undefined;
    if (!colorKey) {
      this._colors = null;
      this._setStatus('colorError', null);
      this._recolor();
      this._buildLegend();
      return;
    }
    const ctrl = (this._colorCtrl = new AbortController());
    this._colorPendingKey = colorKey;
    const t0 = performance.now();
    try {
      const colors = await this._colorRequest(ctrl.signal);
      if (seq !== this._colorSeq || this._destroyed) return;
      this._colorPendingKey = undefined;
      this._colors = {key: colorKey, header: colors.header, ...colors.arrays};
      this.stats.colorFetchMs = performance.now() - t0;
      this._setStatus('colorError', null);
      if (this._cellsLoading) return; // belongs to the cells in flight; their commit paints it
      this._recolor();
      this._buildLegend();
      this._render();
    } catch (err) {
      if ((err && err.name === 'AbortError') || seq !== this._colorSeq) return;
      console.error('[XeniumViewer] colors', err);
      this._colorPendingKey = undefined;
      this._colorKey = undefined; // retried by the next _sync
      this._setStatus('colorError', `failed to load colours: ${err.message || err}`);
    }
  }

  // ── colours ─────────────────────────────────────────────────────────────────
  _colorRange() {
    const h = this._colors && this._colors.header;
    if (!h || h.kind !== 'continuous') return null;
    const c = this.state.color;
    const min = Number.isFinite(c.min) ? c.min : h.min;
    const max = Number.isFinite(c.max) ? c.max : h.max;
    return max < min ? [max, min] : [min, max]; // e.g. while a new Min is being typed
  }

  /**
   * Colour map of the continuous values on screen: the one the colours response names
   * (the server may answer a mode with other values, e.g. transcript counts where a column
   * is missing), else state.colormap.
   */
  _colormap() {
    const h = this._colors && this._colors.header;
    return (h && h.kind === 'continuous' && h.colormap) || this.state.colormap;
  }

  /** Rebuild the per-cell RGBA attribute from the last colours response (no network). */
  _recolor() {
    const cells = this._cells;
    if (!cells) return;
    const t0 = performance.now();
    const n = cells.n;
    if (!this._rgba || this._rgba.length !== 4 * n) this._rgba = new Uint8Array(4 * n);
    const rgba = this._rgba;
    const col = this._colors;
    const data = col && (col.header.kind === 'categorical' ? col.codes : col.values);
    const stale = !!(col && data && data.length !== n);
    if (this._cellsLoading) {
      // a new table is on its way; whatever is on screen is repainted when it commits
    } else if (stale) {
      // Cells and colours were computed from different tables (e.g. a subset swapped in
      // between the two requests): say so, and ask for both again once.
      this._setStatus('colors', `colours out of date (${data.length} values for ${n} cells)`);
      if (!this._staleRetried) {
        this._staleRetried = true;
        this._cellsKey = null;
        this._scheduleSync(['dataVersion']);
      }
    } else {
      this._setStatus('colors', null);
      if (col && data) this._staleRetried = false;
    }
    if (!col || !data || data.length !== n) {
      fillUniform(rgba, n, MISSING_RGB);
    } else if (col.header.kind === 'categorical') {
      const cats = col.header.categories || [];
      const hidden = new Set(this.state.hiddenCategories);
      const mask = new Uint8Array(cats.length);
      for (let i = 0; i < cats.length; i++) mask[i] = hidden.has(cats[i]) ? 1 : 0;
      fillCategorical(rgba, data, n, col.header.colors || [], mask);
    } else {
      const [min, max] = this._colorRange();
      fillContinuous(rgba, data, n, min, max, this._colormap());
    }
    const sel = this.state.selected;
    if (sel && sel.length) {
      const mask = new Uint8Array(n);
      for (let i = 0; i < sel.length; i++) if (sel[i] < n) mask[sel[i]] = 1;
      dimUnselected(rgba, n, mask, 38);
    }
    // A new descriptor object makes deck.gl re-upload this one attribute; positions keep theirs.
    this._colAttr = {value: rgba, size: 4, normalized: true};
    this._cellData = {length: n, attributes: {getPosition: cells.posAttr, getFillColor: this._colAttr}};
    this._dirty.cells = true;
    this.stats.recolorMs = performance.now() - t0;
    const tr = performance.now();
    this._afterRender.push(() => {
      this.stats.recolorRenderMs = performance.now() - tr;
    });
  }

  // ── viewport-bound data ─────────────────────────────────────────────────────
  _updateRegions() {
    const v = this.getViewport();
    const vp = v ? [v.x0, v.y0, v.x1, v.y1] : null;
    const s = this.state;
    for (const kind of BOUNDARY_KINDS) {
      this._loaders[kind].update(vp, {
        enabled: !!s.boundaries[kind],
        key: `${s.source}|${s.dataVersion}`,
        url: (b) => this._url('/viewer/boundaries.bin', {source: s.source, kind, x0: b[0], y0: b[1], x1: b[2], y1: b[3]}),
      });
    }
    const genes = (s.transcripts.genes || []).filter(Boolean);
    this._loaders.tx.update(vp, {
      enabled: genes.length > 0,
      key: `${s.source}|${s.dataVersion}|${genes.join(',')}|${s.transcripts.qv}`,
      url: (b) => this._url('/viewer/transcripts.bin', {
        x0: b[0], y0: b[1], x1: b[2], y1: b[3], genes: genes.join(','), qv: s.transcripts.qv,
      }),
    });
  }

  _onRegionChange(name) {
    if (this._destroyed) return;
    const L = this._loaders[name];
    this._dirty[name] = true;
    let text = null;
    const what = name === 'tx' ? 'transcripts' : BOUNDARY_STYLE[name].label;
    if (L.active) {
      const h = L.info;
      if (L.error) text = `failed to load ${what}: ${L.error}`;
      else if (L.loading && !L.data) text = `loading ${what}…`;
      else if (h && h.available === false) text = `${what} not available`;
      else if (h && h.truncated) text = `zoom in to see ${what} (${h.n.toLocaleString()} in view)`;
      else if (h && h.subsampled) text = `${what} subsampled (${h.n.toLocaleString()} shown), zoom in for all`;
    }
    this._setStatus(name, text);
    if (name === 'tx') this._updateTxLegend();
    this._render();
  }

  // ── layers ──────────────────────────────────────────────────────────────────
  _cellsLayer() {
    const s = this.state;
    if (!this._cellData || !this._cells || !this._cells.n) return null;
    const size = Math.max(0.5, +s.pointSize || 3);
    return new ScatterplotLayer({
      id: 'xv-cells',
      data: this._cellData,
      // Radius is 3 µm in world space, clamped so the dot is `pointSize` px wide when zoomed
      // out and at most 5x that when zoomed in.
      radiusUnits: 'common',
      getRadius: CELL_RADIUS_UM,
      radiusMinPixels: size / 2,
      radiusMaxPixels: size * 2.5,
      opacity: Math.max(0, Math.min(1, +s.opacity)),
      stroked: false,
      pickable: false,
    });
  }

  _boundaryLayer(kind) {
    const d = this._loaders[kind].data;
    if (!d || !d.header.n || !d.arrays.xy) return null;
    const {start, xy} = d.arrays;
    // Xenium rings usually repeat their first vertex; draw those as open paths.
    const e = 2 * (start[1] - 1);
    const closed = start[1] > 1 && xy[0] === xy[e] && xy[1] === xy[e + 1];
    return new PathLayer({
      id: `xv-boundary-${kind}`,
      data: {length: d.header.n, startIndices: start, attributes: {getPath: {value: xy, size: 2}}},
      positionFormat: 'XY',
      _pathType: closed ? 'open' : 'loop',
      getColor: BOUNDARY_STYLE[kind].color,
      getWidth: 1,
      widthUnits: 'pixels',
      widthMinPixels: 1,
      jointRounded: false,
      pickable: false,
    });
  }

  _txLayer() {
    const d = this._loaders.tx.data;
    if (!d || !d.header.n || !d.arrays.xy) return null;
    const n = d.header.n;
    if (!d.rgba) {
      const pal = TX_PALETTE.map(hexToRgb);
      const g = d.arrays.gene;
      const rgba = (d.rgba = new Uint8Array(4 * n));
      for (let i = 0; i < n; i++) {
        const c = pal[g[i] % pal.length];
        rgba[4 * i] = c[0];
        rgba[4 * i + 1] = c[1];
        rgba[4 * i + 2] = c[2];
        rgba[4 * i + 3] = 255;
      }
      d.layerData = {
        length: n,
        attributes: {getPosition: {value: d.arrays.xy, size: 2}, getFillColor: {value: rgba, size: 4, normalized: true}},
      };
    }
    const size = Math.max(0.5, +this.state.transcripts.size || 2);
    return new ScatterplotLayer({
      id: 'xv-transcripts',
      data: d.layerData,
      radiusUnits: 'pixels',
      getRadius: size / 2,
      stroked: false,
      pickable: false,
    });
  }

  _overlayLayers() {
    const o = this.state.overlays || {};
    const out = [];
    if (o.lines && o.lines.length) {
      out.push(new LineLayer({
        id: 'xv-overlay-lines',
        data: o.lines,
        getSourcePosition: (d) => d.from,
        getTargetPosition: (d) => d.to,
        // color: '#rrggbb' or [r, g, b]; alpha: 0-255 (default opaque)
        getColor: (d) => [...(Array.isArray(d.color) ? d.color : hexToRgb(d.color || '#ffffff')), d.alpha === undefined ? 255 : d.alpha],
        getWidth: (d) => d.width || 1,
        widthUnits: 'pixels',
        pickable: false,
      }));
    }
    if (o.polygons && o.polygons.length) {
      out.push(new PolygonLayer({
        id: 'xv-overlay-polygons',
        data: o.polygons,
        positionFormat: 'XY',
        getPolygon: (d) => d.points,
        stroked: true,
        filled: true,
        // fillAlpha: 0-255 (default 64); width 0 draws no outline
        getFillColor: (d) => (d.fill ? [...hexToRgb(d.fill), d.fillAlpha === undefined ? 64 : d.fillAlpha] : [0, 0, 0, 0]),
        // alpha: 0-255 opacity of the outline (default opaque)
        getLineColor: (d) => [...hexToRgb(d.color || '#ffffff'), d.alpha === undefined ? 255 : d.alpha],
        getLineWidth: (d) => (d.width === 0 ? 0 : d.width || 2),
        lineWidthUnits: 'pixels',
        pickable: false,
      }));
    }
    if (o.points && o.points.length) {
      // markers of a fixed size on screen: {at: [x, y], color, size (px, default 8)}
      out.push(new ScatterplotLayer({
        id: 'xv-overlay-points',
        data: o.points,
        getPosition: (d) => d.at,
        getFillColor: (d) => hexToRgb(d.color || '#ffffff'),
        getRadius: (d) => (d.size || 8) / 2,
        radiusUnits: 'pixels',
        stroked: false,
        pickable: false,
      }));
    }
    return out;
  }

  _render() {
    if (this._destroyed) return;
    const L = this._layers;
    const D = this._dirty;
    if (D.cells) L.cells = this._cellsLayer();
    for (const kind of BOUNDARY_KINDS) if (D[kind]) L[kind] = this._boundaryLayer(kind);
    if (D.tx) L.tx = this._txLayer();
    if (D.overlays) L.overlays = this._overlayLayers();
    this._dirty = {};
    let extra = [];
    if (this._extraFn) {
      try {
        extra = this._extraFn(this.viewState) || [];
      } catch (err) {
        console.error('[XeniumViewer] extra layers', err);
      }
    }
    const layers = [...extra, L.cells, L.nucleus, L.cell, L.proseg, L.baysor, L.tx, ...(L.overlays || [])].filter(Boolean);
    this.deck.setProps({layers, viewState: this.viewState});
  }

  setExtraLayers(fn) {
    this._extraFn = typeof fn === 'function' ? fn : null;
    this._render();
  }

  _onAfterRender() {
    if (!this._afterRender.length) return;
    const list = this._afterRender;
    this._afterRender = [];
    for (const fn of list) fn();
  }

  /** Resolves once no request, pending state change or settle timer is outstanding and a frame was drawn. */
  whenIdle() {
    return new Promise((resolve) => {
      let quiet = 0;
      const tick = () => {
        if (this._destroyed) return resolve();
        const busy = this._pending > 0 || this._syncQueued || this._settleTimer || this._retryTimer ||
          this.deck.needsRedraw({clearRedrawFlags: false});
        quiet = busy ? 0 : quiet + 1;
        if (quiet >= 2) resolve();
        else requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
  }

  // ── status ──────────────────────────────────────────────────────────────────
  _setStatus(key, text) {
    if (text) this._status[key] = text;
    else delete this._status[key];
    const joined = Object.values(this._status).join(' · ');
    if (joined === this._statusText) return;
    this._statusText = joined;
    if (this.opts.onStatus) this.opts.onStatus(joined);
  }

  getStatus() {
    return this._statusText;
  }

  // ── legend ──────────────────────────────────────────────────────────────────
  /** The colours response, unless it does not belong to the cells on screen. */
  _shownColors() {
    const col = this._colors;
    if (!col || !this._cells) return col;
    const data = col.header.kind === 'categorical' ? col.codes : col.values;
    return data && data.length !== this._cells.n ? null : col;
  }

  _buildLegend() {
    const legend = this.legend;
    // Rebuilding must not lose the user's place in a long category list.
    const oldList = legend.querySelector('.xv-box[data-kind="categorical"] .xv-list');
    const keep = oldList && this._legendKey ? {key: this._legendKey, top: oldList.scrollTop} : null;
    this._legendKey = null;
    legend.textContent = '';
    const col = this._shownColors();
    if (col && col.header.kind === 'continuous') {
      const box = el('div', 'xv-box');
      box.dataset.kind = 'continuous';
      box.appendChild(el('div', 'xv-title', col.header.label || ''));
      const bar = el('div', 'xv-bar');
      bar.style.background = colormapCss(this._colormap());
      box.appendChild(bar);
      const [min, max] = this._colorRange();
      const range = el('div', 'xv-range');
      range.append(el('span', 'xv-min', fmt(min)), el('span', 'xv-max', fmt(max)));
      box.appendChild(range);
      legend.appendChild(box);
    } else if (col && col.header.kind === 'categorical') {
      const box = el('div', 'xv-box');
      box.dataset.kind = 'categorical';
      box.appendChild(el('div', 'xv-title', col.header.label || ''));
      const list = el('div', 'xv-list');
      const hidden = new Set(this.state.hiddenCategories);
      (col.header.categories || []).forEach((name, i) => {
        const item = el('div', 'xv-item' + (hidden.has(name) ? ' xv-off' : ''));
        item.dataset.category = name;
        const sw = el('span', 'xv-swatch');
        sw.style.background = (col.header.colors || [])[i] || '#808080';
        item.append(sw, el('span', 'xv-name', name));
        item.title = name;
        item.addEventListener('click', () => this._toggleCategory(name));
        item.addEventListener('dblclick', () => this._isolateCategory(name));
        list.appendChild(item);
      });
      box.appendChild(list);
      legend.appendChild(box);
      this._legendKey = col.key;
      if (keep && keep.key === col.key) list.scrollTop = keep.top;
    }
    const genes = (this.state.transcripts.genes || []).filter(Boolean);
    if (genes.length) {
      const box = el('div', 'xv-box');
      box.dataset.kind = 'transcripts';
      box.appendChild(el('div', 'xv-title', this._txTitle()));
      const list = el('div', 'xv-list');
      genes.forEach((g, i) => {
        const item = el('div', 'xv-item xv-static');
        const sw = el('span', 'xv-swatch');
        sw.style.background = TX_PALETTE[i % TX_PALETTE.length];
        item.append(sw, el('span', 'xv-name', g));
        list.appendChild(item);
      });
      box.appendChild(list);
      legend.appendChild(box);
    }
  }

  _txTitle() {
    const info = this._loaders.tx.info;
    const count = this._loaders.tx.data && info ? ` (${info.n.toLocaleString()})` : '';
    return `Transcripts${count}`;
  }

  /** A transcript region update only changes the count in the box title. */
  _updateTxLegend() {
    const title = this.legend.querySelector('.xv-box[data-kind="transcripts"] .xv-title');
    if (title) title.textContent = this._txTitle();
    else this._buildLegend();
  }

  _toggleCategory(name) {
    const cur = this.state.hiddenCategories;
    this._setHidden(cur.includes(name) ? cur.filter((c) => c !== name) : [...cur, name]);
  }

  /** Double click, as in a Plotly legend: show only this category; again: show all. */
  _isolateCategory(name) {
    const col = this._shownColors();
    const cats = (col && col.header.categories) || [];
    const others = cats.filter((c) => c !== name);
    const hidden = new Set(this.state.hiddenCategories);
    const alone = !hidden.has(name) && others.every((c) => hidden.has(c));
    this._setHidden(alone ? [] : others);
  }

  _setHidden(next) {
    this.state.hiddenCategories = next;
    // Update the legend rows in place (keeps the list's scroll position).
    const hidden = new Set(next);
    for (const item of this.legend.querySelectorAll('.xv-item[data-category]')) {
      item.classList.toggle('xv-off', hidden.has(item.dataset.category));
    }
    this._recolor();
    this._render();
    if (this.opts.onStateChange) this.opts.onStateChange(this.getState(), ['hiddenCategories']);
  }

  // ── overlay labels (plain DOM, repositioned on view changes) ────────────────
  _buildLabels() {
    this.labels.textContent = '';
    this._labelItems = [];
    for (const p of (this.state.overlays && this.state.overlays.polygons) || []) {
      if (!p.label || !p.points || !p.points.length) continue;
      let sx = 0, top = Infinity;
      for (const pt of p.points) {
        sx += pt[0];
        if (pt[1] < top) top = pt[1];
      }
      const node = el('div', 'xv-label', p.label);
      node.style.color = p.color || '#ffffff';
      this.labels.appendChild(node);
      this._labelItems.push({node, x: sx / p.points.length, y: top});
    }
    this._updateLabels();
  }

  _updateLabels() {
    if (!this._labelItems || !this._labelItems.length) return;
    // Read the layout once: a size read between two style writes would reflow every label.
    const [w, h] = this._size();
    const s = this._scale();
    const t = this.viewState.target;
    for (const it of this._labelItems) {
      it.node.style.left = `${(it.x - t[0]) * s + w / 2}px`;
      it.node.style.top = `${(it.y - t[1]) * s + h / 2 - 2}px`;
    }
  }

  // ── pointer interaction: click picking, hover tooltip, lasso / rectangle ────
  _bindEvents() {
    const c = this.canvas;
    const local = (e) => {
      const r = this.wrap.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    };
    this._onDown = (e) => {
      if (e.button !== 0 || this._draw) return; // a second finger must not restart the shape
      const p = local(e);
      this._down = {p, t: performance.now()};
      if (this.state.tool === 'lasso' || this.state.tool === 'rect') {
        this._draw = {tool: this.state.tool, pts: [this._toWorld(p[0], p[1])], last: p, id: e.pointerId};
        try {
          c.setPointerCapture(e.pointerId);
        } catch (err) { /* synthetic or already released pointer */ }
        this._hideTip();
        e.preventDefault();
      }
    };
    this._onMove = (e) => {
      const p = local(e);
      const d = this._draw;
      if (d) {
        if (e.pointerId !== d.id) return;
        if (e.buttons === 0) return this._cancelShape(); // the release never reached us
        if (Math.hypot(p[0] - d.last[0], p[1] - d.last[1]) < 2) return;
        d.last = p;
        const w = this._toWorld(p[0], p[1]);
        if (d.tool === 'rect') d.pts[1] = w;
        else d.pts.push(w);
        this._redrawShape();
        return;
      }
      if (e.buttons === 0 && e.target === c) this._hover(p);
    };
    this._onUp = (e) => {
      const d = this._draw;
      if (d && e.pointerId !== d.id) return;
      const down = this._down;
      this._down = null;
      if (d) {
        this._draw = null;
        this._redrawShape();
        this._finishShape(d);
        return;
      }
      if (!down || e.button !== 0) return;
      const p = local(e);
      if (Math.hypot(p[0] - down.p[0], p[1] - down.p[1]) > 4) return; // it was a drag
      if (this.state.tool !== 'pan') return;
      this._click(p);
    };
    this._onDbl = () => {
      if (this.state.tool === 'pan') this.fitView();
    };
    this._onLeave = () => this._hideTip();
    this._onCancel = (e) => {
      if (this._draw && e.pointerId === this._draw.id) this._cancelShape();
      this._down = null;
    };
    c.addEventListener('pointerdown', this._onDown);
    c.addEventListener('dblclick', this._onDbl);
    c.addEventListener('pointerleave', this._onLeave);
    window.addEventListener('pointermove', this._onMove);
    window.addEventListener('pointerup', this._onUp);
    window.addEventListener('pointercancel', this._onCancel);
    c.addEventListener('lostpointercapture', this._onCancel);
    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(this.wrap);
  }

  /** Nearest visible cell under a screen position, or -1. Pure CPU lookup, no GPU readback. */
  _pick(p) {
    const cells = this._cells;
    if (!cells || !cells.n) return -1;
    const s = this._scale();
    const size = Math.max(0.5, +this.state.pointSize || 3);
    const rPx = Math.max(size / 2, Math.min(size * 2.5, CELL_RADIUS_UM * s)) + 3;
    const [x, y] = this._toWorld(p[0], p[1]);
    return cells.grid.nearest(x, y, rPx / s, this._rgba);
  }

  _click(p) {
    const index = this._pick(p);
    if (!this.opts.onClick) return;
    // second argument: where the click landed in world coordinates (also off any cell)
    const [wx, wy] = this._toWorld(p[0], p[1]);
    const at = {x: wx, y: wy};
    if (index < 0) return this.opts.onClick(null, at);
    const xy = this._cells.xy;
    // the source the table was loaded for (state.source may already name the next one)
    this.opts.onClick({index, source: this._cells.source, x: xy[2 * index], y: xy[2 * index + 1]}, at);
  }

  _hover(p) {
    if (this._hoverQueued) {
      this._hoverPos = p;
      return;
    }
    this._hoverPos = p;
    this._hoverQueued = true;
    requestAnimationFrame(() => {
      this._hoverQueued = false;
      if (this._destroyed || this._draw) return;
      const pos = this._hoverPos;
      const index = this._pick(pos);
      if (index < 0) return this._hideTip();
      this._tipIndex = index;
      this.tip.textContent = this._tipText(index);
      this._lookupId(index);
      this.tip.style.display = 'block';
      this.tip.style.left = `${pos[0] + 12}px`;
      this.tip.style.top = `${pos[1] + 12}px`;
    });
  }

  _tipText(index) {
    const id = this._cells.ids.get(index);
    let text = id !== undefined ? id : `cell ${index}`;
    const col = this._shownColors();
    if (col && col.header.kind === 'categorical' && col.codes) {
      const name = (col.header.categories || [])[col.codes[index]];
      if (name !== undefined) text += ` · ${name}`;
    } else if (col && col.values && Number.isFinite(col.values[index])) {
      text += ` · ${col.header.label ? col.header.label + ': ' : ''}${fmt(col.values[index])}`;
    }
    return text;
  }

  /**
   * The tooltip starts with the row index; once the pointer has rested on a cell for
   * HOVER_ID_MS its id is fetched (one small request per cell, remembered) and swapped in.
   */
  _lookupId(index) {
    clearTimeout(this._hoverIdTimer);
    const cells = this._cells;
    if (cells.ids.has(index)) return;
    this._hoverIdTimer = setTimeout(async () => {
      if (this._destroyed || this._cells !== cells || this._tipIndex !== index) return;
      try {
        const resp = await this._fetch(this._url(`/viewer/cell/${index}`, {source: cells.source}), {});
        const info = resp.ok ? await resp.json() : null;
        if (!info || info.id === undefined || info.id === null) return;
        if (cells.ids.size > 5000) cells.ids.clear();
        cells.ids.set(index, String(info.id));
        if (this._cells === cells && this._tipIndex === index && this.tip.style.display === 'block') {
          this.tip.textContent = this._tipText(index);
        }
      } catch (err) { /* the index stays in the tooltip */ }
    }, HOVER_ID_MS);
  }

  _hideTip() {
    this._tipIndex = -1;
    clearTimeout(this._hoverIdTimer);
    if (this.tip.style.display !== 'none') this.tip.style.display = 'none';
  }

  _shapePolygon(d) {
    if (d.tool === 'rect') {
      if (d.pts.length < 2) return null;
      const [a, b] = d.pts;
      const x0 = Math.min(a[0], b[0]), x1 = Math.max(a[0], b[0]);
      const y0 = Math.min(a[1], b[1]), y1 = Math.max(a[1], b[1]);
      if (!(x1 > x0) || !(y1 > y0)) return null;
      return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    }
    return d.pts.length >= 3 ? d.pts : null;
  }

  _redrawShape() {
    const d = this._draw;
    const poly = d && (this._shapePolygon(d) || (d.pts.length > 1 ? d.pts : null));
    if (!poly) {
      this.drawPath.setAttribute('d', '');
      return;
    }
    let path = '';
    for (let i = 0; i < poly.length; i++) {
      const [x, y] = this._toScreen(poly[i][0], poly[i][1]);
      path += `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`;
    }
    this.drawPath.setAttribute('d', path + 'Z');
  }

  _cancelShape() {
    this._draw = null;
    this._redrawShape();
  }

  _finishShape(d) {
    const polygon = this._shapePolygon(d);
    const cells = this._cells;
    if (!polygon || !cells) return;
    const t0 = performance.now();
    const indices = pointsInPolygon(cells.xy, cells.n, polygon, this._rgba);
    this.stats.selectMs = performance.now() - t0;
    this.stats.selectCount = indices.length;
    // Highlight straight away; the host may override with setState({selected}).
    this.state.selected = indices.length ? indices : null;
    this._recolor();
    this._render();
    if (this.opts.onSelect) this.opts.onSelect({indices, polygon: polygon.map((p) => [p[0], p[1]])});
  }

  // ── snapshot ────────────────────────────────────────────────────────────────
  /**
   * The current view as a 2D canvas at `scale` x the on-screen (CSS) resolution: the WebGL
   * frame, then overlay labels, the legend and (scaleBar) a scale bar drawn on top.
   * Raster only: the cells are drawn by the GPU, there is no vector form of the frame.
   */
  snapshot({scale = 2, scaleBar = true, unit = 'µm'} = {}) {
    const [w, h] = this._size();
    if (this._destroyed || !w || !h) return null;
    scale = Math.max(1, Math.min(4, +scale || 2));
    const out = document.createElement('canvas');
    out.width = Math.round(w * scale);
    out.height = Math.round(h * scale);
    const g = out.getContext('2d');
    g.fillStyle = this.state.background || '#000000';
    g.fillRect(0, 0, out.width, out.height);
    // Render once at the requested pixel ratio and copy in the same task (the drawing
    // buffer is not preserved), then go back to the screen's ratio.
    this.deck.setProps({useDevicePixels: scale});
    try {
      this.deck.redraw('snapshot');
      g.drawImage(this.canvas, 0, 0, out.width, out.height);
    } finally {
      this.deck.setProps({useDevicePixels: true});
      this.deck.redraw('snapshot');
    }
    g.scale(scale, scale); // from here on in CSS pixels, like the DOM overlays
    const font = (weight) => `${weight} 12px -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif`;
    g.textBaseline = 'middle';
    // overlay labels
    g.font = font(600);
    g.textAlign = 'center';
    for (const it of this._labelItems || []) {
      const [sx, sy] = this._toScreen(it.x, it.y);
      g.strokeStyle = '#000000';
      g.lineWidth = 3;
      g.strokeText(it.node.textContent, sx, sy - 10);
      g.fillStyle = it.node.style.color || '#ffffff';
      g.fillText(it.node.textContent, sx, sy - 10);
    }
    g.textAlign = 'left';
    this._drawLegend(g, w, h, font);
    if (scaleBar) {
      // a round length close to a fifth of the width
      const perPx = 1 / this._scale();
      const raw = perPx * w * 0.2;
      const mag = Math.pow(10, Math.floor(Math.log10(raw)));
      const len = [5, 2, 1].map((m) => m * mag).find((v) => v <= raw) || mag;
      const px = len / perPx;
      const x0 = 16, y0 = h - 18;
      const text = unit === 'µm' && len >= 1000 ? `${fmt(len / 1000)} mm` : `${fmt(len)} ${unit}`.trim();
      g.font = font(600);
      g.fillStyle = 'rgba(13,17,23,.75)';
      g.fillRect(x0 - 6, y0 - 22, Math.max(px, g.measureText(text).width) + 12, 32);
      g.fillStyle = '#ffffff';
      g.fillRect(x0, y0, px, 3);
      g.fillText(text, x0, y0 - 10);
    }
    return out;
  }

  /** The legend boxes as on screen (top right), drawn on a 2D context in CSS pixels. */
  _drawLegend(g, w, h, font) {
    const boxes = [];
    const col = this._shownColors();
    if (col && col.header.kind === 'continuous') {
      boxes.push({title: col.header.label || '', range: this._colorRange()});
    } else if (col && col.header.kind === 'categorical') {
      const hidden = new Set(this.state.hiddenCategories);
      const rows = [];
      (col.header.categories || []).forEach((name, i) => {
        if (!hidden.has(name)) rows.push({name, color: (col.header.colors || [])[i] || '#808080'});
      });
      boxes.push({title: col.header.label || '', rows});
    }
    const genes = (this.state.transcripts.genes || []).filter(Boolean);
    if (genes.length) {
      boxes.push({title: this._txTitle(), rows: genes.map((name, i) => ({name, color: TX_PALETTE[i % TX_PALETTE.length]}))});
    }
    const W = 200, ROW = 16, PAD = 8;
    let y = 8;
    for (const box of boxes) {
      let rows = box.rows || [];
      const room = Math.max(1, Math.floor((h - y - 8 - 2 * PAD - 20) / ROW));
      const more = rows.length > room ? rows.length - (room - 1) : 0;
      if (more) rows = rows.slice(0, room - 1);
      const bh = 2 * PAD + 18 + (box.range ? 30 : ROW * (rows.length + (more ? 1 : 0)));
      const x = w - 8 - W;
      g.fillStyle = 'rgba(13,17,23,.82)';
      g.fillRect(x, y, W, bh);
      g.strokeStyle = 'rgba(255,255,255,.14)';
      g.lineWidth = 1;
      g.strokeRect(x + 0.5, y + 0.5, W - 1, bh - 1);
      g.fillStyle = '#e6edf3';
      g.font = font(600);
      g.fillText(clip(g, box.title, W - 2 * PAD), x + PAD, y + PAD + 7);
      g.font = font(400);
      let ry = y + PAD + 18;
      if (box.range) {
        const n = 32, vals = new Float32Array(n), rgba = new Uint8Array(4 * n);
        for (let i = 0; i < n; i++) vals[i] = i / (n - 1);
        fillContinuous(rgba, vals, n, 0, 1, this._colormap());
        const grad = g.createLinearGradient(x + PAD, 0, x + W - PAD, 0);
        for (let i = 0; i < n; i++) grad.addColorStop(i / (n - 1), `rgb(${rgba[4 * i]},${rgba[4 * i + 1]},${rgba[4 * i + 2]})`);
        g.fillStyle = grad;
        g.fillRect(x + PAD, ry, W - 2 * PAD, 10);
        g.fillStyle = '#e6edf3';
        g.fillText(fmt(box.range[0]), x + PAD, ry + 21);
        g.textAlign = 'right';
        g.fillText(fmt(box.range[1]), x + W - PAD, ry + 21);
        g.textAlign = 'left';
      }
      for (const r of rows) {
        g.fillStyle = r.color;
        g.beginPath();
        g.arc(x + PAD + 5, ry + ROW / 2, 5, 0, 2 * Math.PI);
        g.fill();
        g.fillStyle = '#e6edf3';
        g.fillText(clip(g, r.name, W - 2 * PAD - 16), x + PAD + 16, ry + ROW / 2);
        ry += ROW;
      }
      if (more) g.fillText(`… ${more} more`, x + PAD + 16, ry + ROW / 2);
      y += bh + 6;
    }
  }

  // ── teardown ────────────────────────────────────────────────────────────────
  destroy() {
    if (this._destroyed) return;
    this._destroyed = true;
    clearTimeout(this._settleTimer);
    this._settleTimer = null;
    if (this._cellsCtrl) this._cellsCtrl.abort();
    if (this._colorCtrl) this._colorCtrl.abort();
    for (const k of Object.keys(this._loaders)) this._loaders[k].destroy();
    this._ro.disconnect();
    this.canvas.removeEventListener('pointerdown', this._onDown);
    this.canvas.removeEventListener('dblclick', this._onDbl);
    this.canvas.removeEventListener('pointerleave', this._onLeave);
    window.removeEventListener('pointermove', this._onMove);
    window.removeEventListener('pointerup', this._onUp);
    window.removeEventListener('pointercancel', this._onCancel);
    this.canvas.removeEventListener('lostpointercapture', this._onCancel);
    clearTimeout(this._retryTimer);
    clearTimeout(this._hoverIdTimer);
    this._retryTimer = null;
    // Chrome keeps at most 16 live WebGL contexts per page and evicts the oldest (which may be
    // a Plotly plot), so give this one back now instead of waiting for garbage collection.
    let gl = null;
    try {
      gl = (this.deck.device && this.deck.device.gl) || null;
    } catch (err) { /* device was never created */ }
    this.deck.finalize();
    this.wrap.remove();
    try {
      const ext = gl && gl.getExtension('WEBGL_lose_context');
      if (ext) ext.loseContext();
    } catch (err) { /* already lost */ }
    this.canvas.width = this.canvas.height = 0;
    this._cells = this._colors = this._rgba = this._cellData = null;
  }
}

// cells.bin tokens start with the dataset generation: "<gen>:<subset>:…" ("umap:<gen>:…" for the UMAP provider).
function datasetGen(token) {
  const parts = String(token || '').split(':');
  return parts[0] === 'umap' ? parts[1] : parts[0];
}

/** `text` shortened with an ellipsis to fit `width` on the 2D context `g`. */
function clip(g, text, width) {
  text = String(text);
  if (g.measureText(text).width <= width) return text;
  while (text.length > 1 && g.measureText(text + '…').width > width) text = text.slice(0, -1);
  return text + '…';
}

function boundsOf(xy, n) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = xy[2 * i], y = xy[2 * i + 1];
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
  }
  return [x0, y0, x1, y1];
}

function mount(element, opts) {
  if (!element) throw new Error('XeniumViewer.mount: element is required');
  const v = new Viewer(element, opts);
  // Public handle: the documented API plus a few read-only extras for tests / integration.
  return {
    setState: (partial) => v.setState(partial),
    getState: () => v.getState(),
    getViewport: () => v.getViewport(),
    fitView: () => v.fitView(),
    setViewport: (vp) => v.setViewport(vp),
    resize: () => v.resize(),
    destroy: () => v.destroy(),
    setExtraLayers: (fn) => v.setExtraLayers(fn),
    snapshot: (opts) => v.snapshot(opts),
    // extras
    whenIdle: () => v.whenIdle(),
    getStatus: () => v.getStatus(),
    project: (x, y) => v._toScreen(x, y),
    unproject: (sx, sy) => v._toWorld(sx, sy),
    get stats() { return v.stats; },
    get requestLog() { return v.requestLog.slice(); },
    get deck() { return v.deck; },
    _viewer: v,
  };
}

if (typeof window !== 'undefined') {
  window.XeniumViewer = {
    mount,
    version: VERSION,
    util: {parseBin, packBin, pointsInPolygon},
    // The layer classes of THIS bundle's deck.gl copy, for building setExtraLayers() layers.
    // Layers created from another copy of deck.gl must not be mixed in.
    layers: {ScatterplotLayer, PathLayer, PolygonLayer, LineLayer},
  };
}
