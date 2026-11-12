// Minimal test page: a bare deck.gl Deck (OrthographicView, y down) with the morphology
// layers and a few controls. Everything the Playwright tests need hangs off window.__t.
//
// URL parameters: ?worker=<url> (worker bundle, default /assets/xenium_image_worker.js),
//                 ?js=1 (force the asm.js codec; needs the bench worker), ?pool=<n>, ?panel=0
import { Deck, OrthographicView } from "@deck.gl/core";
import { PathLayer, ScatterplotLayer } from "@deck.gl/layers";
import { openImages } from "../src/image/index.js";

const DEFAULT_COLORS = [[0, 120, 255], [0, 255, 0], [255, 0, 255], [255, 160, 0], [255, 255, 255], [255, 0, 0]];
const q = new URLSearchParams(location.search);
if (q.get("panel") === "0") document.body.classList.add("nopanel");

const t = {
  ready: null, error: null, consoleErrors: [], longTasks: [], dots: [], outlines: [],
};
window.__t = t;
window.addEventListener("error", (e) => t.consoleErrors.push(String(e.message)));
window.addEventListener("unhandledrejection", (e) => t.consoleErrors.push(`unhandled: ${e.reason && (e.reason.stack || e.reason)}`));
try {
  new PerformanceObserver((list) => {
    for (const e of list.getEntries()) t.longTasks.push({ start: e.startTime, duration: e.duration });
  }).observe({ entryTypes: ["longtask"] });
} catch (e) { /* longtask not supported */ }

const hex = (c) => `#${c.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
const unhex = (s) => [1, 3, 5].map((i) => parseInt(s.slice(i, i + 2), 16));

t.ready = (async () => {
  const meta = await (await fetch("/viewer/meta")).json();
  const ctl = await openImages({
    baseUrl: "",
    images: meta.images,
    pixelSize: meta.pixel_size,
    workerUrl: q.get("worker") || undefined,
    forceJs: q.get("js") === "1",
    poolSize: q.get("pool") ? Number(q.get("pool")) : undefined,
  });
  t.openedAt = performance.now();
  const ch0 = ctl.channels[0];
  const px = meta.pixel_size;
  const size = [ch0.width * px, ch0.height * px]; // µm

  const state = {
    visible: true,
    opacity: 1,
    channels: ctl.channels.map((c, i) => ({ index: c.index, on: i === 0, color: DEFAULT_COLORS[i % 6], min: 0, max: 65535 })),
  };

  const el = document.getElementById("map");
  const fit = () => {
    const zoom = Math.log2(Math.min(el.clientWidth / size[0], el.clientHeight / size[1])) - 0.05;
    return { target: [size[0] / 2, size[1] / 2, 0], zoom };
  };
  let viewState = fit();
  if (q.get("x")) viewState = { target: [Number(q.get("x")), Number(q.get("y")), 0], zoom: Number(q.get("zoom")) };

  const layers = () => [
    ...ctl.layers(state),
    t.dots.length && new ScatterplotLayer({
      id: "dots",
      data: t.dots,
      getPosition: (d) => d,
      getRadius: 4,
      radiusUnits: "pixels",
      stroked: true,
      filled: false,
      getLineColor: [255, 40, 40],
      getLineWidth: 1.5,
      lineWidthUnits: "pixels",
    }),
    t.outlines.length && new PathLayer({
      id: "outlines",
      data: t.outlines,
      getPath: (d) => d,
      getColor: [255, 230, 0],
      getWidth: 1,
      widthUnits: "pixels",
    }),
  ].filter(Boolean);

  const deck = new Deck({
    parent: el,
    views: new OrthographicView({ id: "ortho", flipY: true }),
    initialViewState: viewState,
    controller: true,
    layers: layers(),
    onViewStateChange: ({ viewState: vs }) => { viewState = vs; },
    onError: (err) => { t.consoleErrors.push(`deck: ${err && (err.stack || err.message || err)}`); console.error(err); },
  });
  const render = () => deck.setProps({ layers: layers() });

  // ---- controls ----
  const box = document.getElementById("channels");
  const ui = [];
  ctl.channels.forEach((c, i) => {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<input type="checkbox" class="on"><input type="color" class="col"><label title="${c.label}">${c.label}</label>`
      + '<input type="range" class="min" min="0" max="65535" step="1"><input type="range" class="max" min="0" max="65535" step="1">';
    box.appendChild(row);
    const g = (s) => row.querySelector(s);
    ui.push({ on: g(".on"), col: g(".col"), min: g(".min"), max: g(".max") });
    row.addEventListener("input", () => {
      const s = state.channels[i];
      s.on = g(".on").checked;
      s.color = unhex(g(".col").value);
      s.min = Number(g(".min").value);
      s.max = Number(g(".max").value);
      if (s.on && !s.auto) autoContrast(i);
      render();
    });
  });
  const syncUi = () => {
    state.channels.forEach((s, i) => {
      ui[i].on.checked = s.on;
      ui[i].col.value = hex(s.color);
      ui[i].min.value = s.min;
      ui[i].max.value = s.max;
    });
    document.getElementById("opacity").value = state.opacity;
  };
  document.getElementById("opacity").addEventListener("input", (e) => { state.opacity = Number(e.target.value); render(); });

  // Auto contrast the first time a channel is shown.
  const autoContrast = async (i) => {
    const s = state.channels[i];
    s.auto = true;
    const lim = await ctl.defaultContrast(s.index);
    s.min = lim.min;
    s.max = lim.max;
    syncUi();
    render();
    return lim;
  };

  const tileLayers = () => deck.layerManager.getLayers().filter((l) => l.id.startsWith("Tiled-Image-"));
  const isIdle = () => {
    const tl = tileLayers();
    const wantTiles = ctl.layers(state).length > 0;
    return (!wantTiles || (tl.length > 0 && tl.every((l) => l.isLoaded))) && ctl.stats.inFlight === 0;
  };

  Object.assign(t, {
    meta, ctl, deck, state, size, render, syncUi, autoContrast, isIdle, fit,
    setState(patch) {
      if (patch.opacity !== undefined) state.opacity = patch.opacity;
      if (patch.visible !== undefined) state.visible = patch.visible;
      for (const p of patch.channels || []) Object.assign(state.channels.find((s) => s.index === p.index), p);
      syncUi();
      render();
    },
    setView(vs) {
      viewState = { ...vs, target: [vs.target[0], vs.target[1], 0] };
      deck.setProps({ initialViewState: { ...viewState } });
    },
    getView: () => viewState,
    setDots(pts, outlines = []) { t.dots = pts; t.outlines = outlines; render(); },
    // Resolves (with elapsed ms) once every visible tile is loaded and drawn.
    whenIdle(timeout = 120000) {
      const t0 = performance.now();
      return new Promise((resolve, reject) => {
        let stable = 0;
        const tick = () => {
          stable = isIdle() ? stable + 1 : 0;
          if (stable >= 3) return requestAnimationFrame(() => requestAnimationFrame(() => resolve(performance.now() - t0)));
          if (performance.now() - t0 > timeout) return reject(new Error("whenIdle timeout"));
          return setTimeout(tick, 30);
        };
        tick();
      });
    },
    tileInfo: () => tileLayers().map((l) => ({
      id: l.id,
      tiles: l.state.tileset.tiles.map((x) => ({ ...x.index, loaded: x.isLoaded, visible: x.isVisible, selected: x.isSelected })),
    })),
  });

  syncUi();
  await autoContrast(0);
  const status = document.getElementById("status");
  setInterval(() => {
    const s = ctl.stats;
    status.textContent = `${ctl.channels[0].width}×${ctl.channels[0].height}px  ${ctl.channels[0].levels} levels  codec ${ctl.decoderBackend} ×${ctl.poolSize}\n`
      + `req ${s.requests}  ${(s.bytes / 1e6).toFixed(1)} MB  aborted ${s.aborted}  errors ${s.errors}  zoom ${viewState.zoom.toFixed(2)}`;
  }, 250);
  return true;
})().catch((err) => {
  t.error = String((err && err.stack) || err);
  document.getElementById("status").textContent = t.error;
  console.error(err);
  throw err;
});
