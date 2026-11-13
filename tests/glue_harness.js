// Runs assets/xenium_viewer_dash.js outside a browser: a stand-in DOM, a recording
// XeniumViewer, and fetch answered from a table. Used by the default-tier tests of the
// glue (node only; no browser). Usage: node glue_harness.js <glue.js> <scenario>
const fs = require("fs");
const vm = require("vm");

function stub() {
  return new Proxy(function () {}, {
    get(t, k) {
      if (k === Symbol.toPrimitive) return () => "";
      if (k === "then") return undefined;
      if (!(k in t)) t[k] = stub();
      return t[k];
    },
    set(t, k, v) { t[k] = v; return true; },
    apply() { return stub(); },
    construct() { return stub(); },
  });
}

const props = [];              // every set_props call: [id, props]
const viewers = [];            // one record per mount: {baseUrl, opts, states: []}
const fetches = {};            // url prefix -> JSON answer

function viewerRecord(el, opts) {
  const rec = { baseUrl: opts.baseUrl || "", opts: opts, states: [] };
  viewers.push(rec);
  const api = stub();
  api.setState = (s) => { rec.states.push(JSON.parse(JSON.stringify(s))); };
  api.resize = () => {};
  api.destroy = () => {};
  return api;
}

const document = stub();
document.getElementById = (id) => {
  if (id === "_dash-config") return { textContent: "{}" };
  const el = stub();
  el.id = id;
  return el;
};
const window = stub();
window.document = document;
window.XeniumViewer = { mount: viewerRecord };
window.dash_clientside = { no_update: "NO_UPDATE", set_props: (id, p) => props.push([id, p]) };
const sandbox = {
  window, document, console, JSON, Math, Date, Object, Array, Number, String, Boolean, RegExp,
  encodeURIComponent, isFinite, parseFloat, parseInt, Promise, Event: function () {},
  setTimeout: (fn) => { try { fn(); } catch (e) { /* layout only */ } return 0; },
  clearTimeout: () => {}, setInterval: () => 0, clearInterval: () => {},
  requestAnimationFrame: () => 0, navigator: { userAgent: "node" }, URL: stub(),
  fetch: (url) => {
    const key = Object.keys(fetches).find((k) => String(url).indexOf(k) >= 0);
    return Promise.resolve({ json: () => Promise.resolve(key ? fetches[key] : {}) });
  },
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), sandbox);
const glue = window.dash_clientside.xenium_viewer;

// sync(mode, colorBy, method, gene, size, opacity, toggles, segSource, countsMode, cmin, cmax,
//      txGenes, morphEnable, morphChannels, morphOpacity, morphBrightness, datasetVer, subsetVer,
//      baysorDone, prosegDone, annotDone, spageDone, splitDone, roiDone, nicheDone, tool,
//      umapPanelStyle, roiModalOpen, extraVer, relayout)
function sync(method) {
  return glue.sync("webgl", "cluster", method, "", 2, 0.8, [], "xenium", "original", null, null,
                   [], [], [], 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, "pan", {}, false, 1, {});
}
const umap = () => viewers.filter((v) => /\/umap$/.test(v.baseUrl)).pop();
const lastWith = (rec, key) => rec.states.filter((s) => key in s).pop();

async function main() {
  const out = {};
  const scenario = process.argv[3];
  if (scenario === "joint_rerun") {
    sync("joint_clusters");
    glue.analysis("webgl", null, 1);
    out.before = lastWith(umap(), "dataVersion").dataVersion;
    const n = umap().states.length;
    glue.analysis("webgl", null, 2);                 // what a finished joint re-run sends
    const after = umap().states.slice(n);
    out.after = (after.filter((s) => "dataVersion" in s).pop() || {}).dataVersion || null;
    out.color = (after.filter((s) => s.color).pop() || { color: {} }).color.version || null;
    // not a joint embedding: the coordinates are not refetched for a gene-set change
    sync("graphclust");
    const m = umap().states.length;
    glue.analysis("webgl", null, 3);
    out.plain = umap().states.slice(m).some((s) => "dataVersion" in s);
  } else if (scenario === "umap_click") {
    sync("joint_clusters");
    fetches["/umap/viewer/cell/7"] = { index: 7, id: "cellX", sample: 1 };
    fetches["/umap/viewer/cell/3"] = { index: 3, id: "cellY", sample: 0 };
    umap().opts.onClick({ index: 7, source: "xenium|emb:joint_clusters" });
    umap().opts.onClick({ index: 3, source: "xenium|emb:joint_clusters" });
    await new Promise((r) => setImmediate(r));
    await new Promise((r) => setImmediate(r));
    out.selected = props.filter((p) => p[0] === "selected-cell").map((p) => p[1].data);
  }
  console.log(JSON.stringify(out));
}
main().catch((e) => { console.error(e); process.exit(1); });
