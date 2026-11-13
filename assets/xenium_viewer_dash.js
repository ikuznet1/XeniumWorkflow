// Glue between the Dash sidebar controls and window.XeniumViewer (assets/xenium_viewer.js).
// Registered as the clientside callback "xenium_viewer.sync" in xenium_explorer.py.
// Only control state crosses this boundary; the viewer fetches its own data from /viewer/*.
(function () {
  "use strict";

  // Sidebar channel keys -> morphology file index and display colour (matches MORPH_CHANNELS).
  var CHANNELS = {
    dapi: { index: 0, color: [70, 130, 255], label: "DAPI" },
    boundary: { index: 1, color: [50, 255, 80], label: "ATP1A1/CD45/E-Cadherin" },
    "18s": { index: 2, color: [255, 60, 0], label: "18S" },
    sma: { index: 3, color: [255, 0, 220], label: "αSMA/Vimentin" },
  };

  // Continuous colour scale per colour mode, as in the Plotly figures (COLORSCALES in
  // xenium_explorer.py; gene expression is drawn with Plasma there). Everything else: viridis.
  var COLORMAPS = { gene: "plasma", cell_area: "plasma", nucleus_area: "cividis", total_counts: "magma" };

  // Path prefix the app is reached under (a proxy that strips it, DASH_REQUESTS_PATHNAME_PREFIX).
  // With url_base_pathname Dash's own routes move as well, but /viewer/* stays at the root.
  var PREFIX = (function () {
    try {
      var cfg = JSON.parse(document.getElementById("_dash-config").textContent);
      if (cfg.url_base_pathname) return "";
      return String(cfg.requests_pathname_prefix || "/").replace(/\/+$/, "");
    } catch (e) {
      return "";
    }
  })();

  var viewer = null;        // mounted XeniumViewer instance
  var mountedOn = null;     // the element it is mounted on (Dash may re-create the node)
  // { key, sig, ctl, contrast: {index: {min, max}}, contrastFailed: {index: true}, failed, checking }
  var images = null;
  var imageKey = null;      // dataset key of the latest sync
  var imageState = { visible: false, opacity: 0.85, channels: [] };
  var lastMode = null;
  var lost = { spatial: false, umap: false };   // WebGL context lost: remount on the next sync

  // Status line: the viewer's own text plus the glue's messages (image layer, missing bundle).
  var viewerStatus = "";
  var glueMsg = {};
  function renderStatus() {
    var parts = [];
    if (glBroken) {
      // Plotly draws the cells with WebGL 1: without any WebGL it shows its own notice.
      parts.push(glNone ? "WebGL is switched off in this browser: neither the WebGL view nor the Plotly view can " +
                          "draw cells - enable WebGL / hardware acceleration and reload"
                        : "WebGL view unavailable: " + glBroken + " - showing the Plotly view");
    }
    if (lastMode === "webgl" && !glBroken) {
      parts.push(lost.spatial ? "WebGL context lost - change any setting or switch the renderer to redraw"
                              : viewerStatus);
      Object.keys(glueMsg).forEach(function (k) { parts.push(glueMsg[k]); });
    }
    setProps("webgl-status", { children: parts.filter(Boolean).join(" · ") });
  }
  function setMsg(key, text) {
    if ((glueMsg[key] || null) === (text || null)) return;
    if (text) glueMsg[key] = text;
    else delete glueMsg[key];
    renderStatus();
  }

  // The WebGL view needs a WebGL2 context. Without one (old browser, GPU blocklisted,
  // hardware acceleration off, context creation refused) the app goes back to the Plotly
  // view by itself and says why in the status line, instead of leaving an empty panel.
  var glBroken = null;      // reason the WebGL view cannot be used in this page, or null
  var glNone = false;       // not even WebGL 1: the Plotly view's point layer cannot draw either
  var glChecked = false;
  function probeGl(type) {
    try {
      var gl = document.createElement("canvas").getContext(type);
      if (!gl) return false;
      var lose = gl.getExtension("WEBGL_lose_context");
      if (lose) lose.loseContext();   // give the probe context back
      return true;
    } catch (e) {
      return false;
    }
  }
  function webglProblem() {
    if (glBroken || glChecked) return glBroken;
    var why = null;
    if (!probeGl("webgl2")) {
      why = "this browser has no WebGL2 (unsupported, or disabled / hardware acceleration off)";
      glNone = !probeGl("webgl");
    }
    if (!why && !window.XeniumViewer) why = "the viewer bundle (assets/xenium_viewer.js) did not load";
    glChecked = !why;
    return why;
  }
  function fallBackToPlotly(why) {
    if (glBroken) return;
    glBroken = why;
    console.warn("[XeniumViewer] " + why + " - falling back to the Plotly renderer");
    renderStatus();
    // Outside the running callback: renderer-mode is one of its own inputs.
    setTimeout(function () {
      setProps("renderer-mode", {
        value: "plotly",
        options: [{ label: " Plotly", value: "plotly" },
                  { label: " WebGL (unavailable)", value: "webgl", disabled: true }],
      });
    }, 0);
  }

  // Hides the Plotly UMAP graph while the WebGL one is shown.
  var hideCss = document.createElement("style");
  hideCss.textContent = "body.xv-webgl #umap-plot{display:none !important}" +
    // Plotly's "WebGL is not supported" notice must not cover the renderer bar and its status line
    "#spatial-plot .no-webgl{z-index:1 !important}" +
    "body:not(.xv-webgl) #webgl-png-btn{display:none !important}";   // PNG export: WebGL view only
  document.head.appendChild(hideCss);

  function setProps(id, props) {
    if (window.dash_clientside && window.dash_clientside.set_props) {
      window.dash_clientside.set_props(id, props);
    }
  }

  // A lost context (GPU reset, sleep / wake, too many contexts) leaves a black canvas.
  function watchContext(el, which, isCurrent) {
    var canvas = el.querySelector("canvas");
    if (!canvas) return;
    canvas.addEventListener("webglcontextlost", function () {
      if (!isCurrent()) return;   // destroy() gives its context back on purpose
      lost[which] = true;
      renderStatus();
    });
  }

  function mount() {
    var el = document.getElementById("webgl-view");
    if (!el || !window.XeniumViewer) return null;
    if (viewer && mountedOn === el && !lost.spatial) return viewer;
    var keep = null;
    if (viewer) {
      if (lost.spatial && mountedOn === el) {
        try { keep = viewer.getViewport(); } catch (e) { /* no size */ }
      }
      closeImages();
      try { viewer.destroy(); } catch (e) { /* element already gone */ }
    }
    lost.spatial = false;
    overlayKey = null;      // the new viewer has no overlays yet
    cn.key = null;
    pies.polys = [];
    viewerStatus = "";
    viewer = window.XeniumViewer.mount(el, {
      baseUrl: PREFIX,
      onViewport: function (vp) {
        // Same shape as Plotly relayoutData (plot y is -µm), so server-side
        // "use viewport" features keep working.
        setProps("spatial-relayout", {
          data: {
            "xaxis.range[0]": vp.x0, "xaxis.range[1]": vp.x1,
            "yaxis.range[0]": -vp.y1, "yaxis.range[1]": -vp.y0,
          },
        });
        syncPies(vp);
      },
      onClick: function (info, at) {
        if (ex.measure) {
          // Ruler: the click sets an end point (the cell's centre, or the spot itself off any
          // cell) and selects nothing. Points are kept in plot space (y = -µm) like Plotly's.
          var p = info ? [info.x, -info.y] : (at ? [at.x, -at.y] : null);
          if (!p) return;
          var pts = ex.measure.points || [];
          ex.measure = { active: true, points: pts.length >= 2 ? [p] : pts.concat([p]) };
          setProps("measure-store", { data: ex.measure });   // the server fills the distance label
          pushOverlays();
          return;
        }
        setProps("webgl-click", { data: info });
        if (!info) return;
        // Resolve the clicked index to a cell id and feed the existing cell-info panel.
        fetch(PREFIX + "/viewer/cell/" + info.index + "?source=" + encodeURIComponent(info.source || "xenium"))
          .then(function (r) { return r.json(); })
          .then(function (cell) {
            if (!cell || !cell.id) return;
            // a cell of a sample shown beside the primary one is named with its sample
            setProps("selected-cell", { data: cell.sample ? { id: cell.id, sample: cell.sample } : cell.id });
          })
          .catch(function () { /* cell info is optional */ });
      },
      onSelect: function (sel) {
        // The outline is what defines an ROI; the indices stay in the browser.
        setProps("webgl-select", {
          data: { count: sel.indices.length, polygon: sel.polygon, t: Date.now() },
        });
        // Keep the drawn outline (and the viewer's highlight of the selected cells) on
        // screen while the Save ROI dialog is open; sync() removes both when it closes.
        pendingRoi = { points: sel.polygon, color: "#aaaaaa", fill: "#c8c8c8", width: 1.5 };
        pushOverlays();
        if (viewer) viewer.setState({ tool: "pan" });
        setProps("webgl-tool", { value: "pan" });
      },
      onStatus: function (text) {
        viewerStatus = text || "";
        renderStatus();
      },
      onFatal: fallBackToPlotly,   // no WebGL2 device after all
    });
    mountedOn = el;
    window.XeniumViewer.instance = viewer;   // handle for the browser console and tests
    var handle = viewer;
    watchContext(el, "spatial", function () { return viewer === handle; });
    if (keep) viewer.setViewport(keep);
    return viewer;
  }

  var umapViewer = null;
  var umapMountedOn = null;
  function mountUmap() {
    var el = document.getElementById("webgl-umap");
    if (!el || !window.XeniumViewer) return null;
    if (umapViewer && umapMountedOn === el && !lost.umap) return umapViewer;
    if (umapViewer) {
      try { umapViewer.destroy(); } catch (e) { /* element already gone */ }
    }
    lost.umap = false;
    umapViewer = window.XeniumViewer.mount(el, {
      baseUrl: PREFIX + "/umap",
      onClick: function (info) {
        if (!info || ex.measure) return;   // no cell selection while the ruler is active
        fetch(PREFIX + "/umap/viewer/cell/" + info.index + "?source=" + encodeURIComponent(info.source || "xenium"))
          .then(function (r) { return r.json(); })
          .then(function (cell) { if (cell && cell.id) setProps("selected-cell", { data: cell.id }); })
          .catch(function () { /* cell info is optional */ });
      },
    });
    umapMountedOn = el;
    window.XeniumViewer.umapInstance = umapViewer;
    var handle = umapViewer;
    watchContext(el, "umap", function () { return umapViewer === handle; });
    return umapViewer;
  }

  var overlayKey = null;
  var serverOverlays = { polygons: [], lines: [] };   // saved ROIs, sdata outlines, sample separators
  var pendingRoi = null;                              // outline drawn for the open Save ROI dialog
  var lastRoiDone, lastRoiOpen = false, lastTool = null;
  // What the latest sync() selected (the fetches below are keyed on it).
  var cur = { source: "xenium", tool: "xenium", colorBy: "", dataVersion: "", colorVersion: "", overlayBase: "" };
  // Controls handled by extras(): ruler state, sdata outline toggles, CellNEST query (JSON) or null.
  var ex = { measure: null, sdataRoi: false, sdataPatches: false, sdataDone: 0, cellnest: null, cnVersion: 0 };
  var cn = { key: null, seq: 0, lines: [], cells: null };   // CellNEST edges / participating rows
  var pies = { seq: 0, polys: [], off: null };              // RCTD weight pies of the cells in view
  var joined = { a: null, b: null, out: [] };

  function parseBin(buf) { return window.XeniumViewer.util.parseBin(buf); }

  function pushOverlays() {
    if (!viewer) return;
    var polygons = serverOverlays.polygons;
    if (pendingRoi) polygons = polygons.concat([pendingRoi]);
    if (pies.polys.length) polygons = polygons.concat(pies.polys);
    // (the same array while nothing changed, so the viewer does not rebuild its line layer)
    if (joined.a !== serverOverlays.lines || joined.b !== cn.lines) {
      joined = { a: serverOverlays.lines, b: cn.lines, out: serverOverlays.lines.concat(cn.lines) };
    }
    var lines = joined.out, points = [];
    var m = ex.measure;
    if (m && m.points && m.points.length) {
      points = m.points.map(function (p) { return { at: [p[0], -p[1]], color: "#f0c040", size: 9 }; });
      if (m.points.length === 2) {
        lines = lines.concat([{ from: points[0].at, to: points[1].at, color: "#f0c040", width: 2 }]);
      }
    }
    viewer.setState({ overlays: { polygons: polygons, lines: lines, points: points } });
  }

  // Outlines from the server: refetched only when what they are built from changes.
  function syncOverlays() {
    // (an.gsVersion: per-cell results drawn as outlines - aggregates - bump the same version)
    var key = [cur.overlayBase, ex.sdataRoi, ex.sdataPatches, ex.sdataDone, an.gsVersion].join("|");
    if (!viewer || key === overlayKey) return;
    overlayKey = key;
    fetch(PREFIX + "/viewer/overlays?source=" + encodeURIComponent(cur.source) +
          (ex.sdataRoi ? "&sdata_roi=1" : "") + (ex.sdataPatches ? "&sdata_patches=1" : ""))
      .then(function (r) { return r.json(); })
      .then(function (ov) {
        if (overlayKey !== key || !viewer) return;
        serverOverlays = { polygons: ov.polygons || [], lines: ov.lines || [] };
        pushOverlays();
      })
      .catch(function () { overlayKey = null; });
  }

  // CellNEST "gray-out" dims every cell that takes no part in the drawn edges; a lasso
  // selection takes the highlight over while its dialog is open.
  function applySelection() {
    if (viewer && !pendingRoi) viewer.setState({ selected: cn.cells || null });
  }

  // CellNEST edges for the selected LR pairs, as line segments from the server.
  function syncCellnest() {
    if (!viewer) return;
    var q = ex.cellnest;
    var key = q ? [cur.source, cur.dataVersion, ex.cnVersion, q].join("|") : null;
    if (key === cn.key) return;
    cn.key = key;
    var seq = ++cn.seq;
    if (!q) {
      cn.lines = [];
      cn.cells = null;
      setMsg("cellnest", null);
      applySelection();
      pushOverlays();
      return;
    }
    // Row indices (gray-out) and edges belong to one cell table: after a source / subset /
    // dataset change the old ones must not be applied to the new table while (or if) the
    // new answer is missing. A change of the CellNEST controls alone keeps them until the
    // answer arrives, so the edges do not flicker.
    var table = cur.source + "|" + cur.dataVersion;
    if (cn.table !== table) {
      cn.table = table;
      if (cn.cells || cn.lines.length) {
        cn.lines = [];
        cn.cells = null;
        applySelection();
        pushOverlays();
      }
    }
    fetch(PREFIX + "/viewer/extra/cellnest.bin?source=" + encodeURIComponent(cur.source) +
          "&q=" + encodeURIComponent(q))
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.arrayBuffer();
      })
      .then(function (buf) {
        if (seq !== cn.seq) return;
        var res = parseBin(buf), h = res.header, a = res.arrays;
        var n = a.width ? a.width.length : 0;
        var lines = new Array(n);
        for (var i = 0; i < n; i++) {
          lines[i] = {
            from: [a.xy[4 * i], a.xy[4 * i + 1]], to: [a.xy[4 * i + 2], a.xy[4 * i + 3]],
            color: [a.rgba[4 * i], a.rgba[4 * i + 1], a.rgba[4 * i + 2]], alpha: a.rgba[4 * i + 3],
            width: a.width[i],
          };
        }
        cn.lines = lines;
        cn.cells = h.gray && a.cells && a.cells.length ? a.cells : null;
        setMsg("cellnest", h.available === false ? "CellNEST: no results loaded"
          : !n ? "CellNEST: no edges for this selection"
          : h.gray && !cn.cells ? "CellNEST gray-out: no cell of this segmentation matches the results"
          : null);
        applySelection();
        pushOverlays();
      })
      .catch(function (err) {
        if (seq !== cn.seq) return;
        cn.key = null;   // asked again on the next change
        cn.lines = [];   // nothing of the previous answer stays on screen
        cn.cells = null;
        setMsg("cellnest", "CellNEST edges failed to load: " + ((err && err.message) || err));
        applySelection();
        pushOverlays();
      });
  }

  // RCTD weight pies (as the Plotly view draws them) for the cells in view, when the colour
  // mode has stored weights and few enough cells are visible. One small request per settled
  // viewport; a colour mode without weights is asked about once.
  function syncPies(vp) {
    if (!viewer) return;
    var mode = cur.colorBy || "";
    var key = cur.colorVersion + "|" + mode;
    if (cur.tool !== "xenium" || mode.indexOf("cell_type:") !== 0 || pies.off === key) {
      ++pies.seq;
      if (pies.polys.length) {
        pies.polys = [];
        pushOverlays();
      }
      return;
    }
    vp = vp || viewer.getViewport();
    if (!vp) return;
    var seq = ++pies.seq;
    fetch(PREFIX + "/viewer/extra/pies.bin?source=" + encodeURIComponent(cur.source) +
          "&mode=" + encodeURIComponent(mode) +
          "&x0=" + vp.x0 + "&y0=" + vp.y0 + "&x1=" + vp.x1 + "&y1=" + vp.y1)
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.arrayBuffer();
      })
      .then(function (buf) {
        if (seq !== pies.seq) return;
        var res = parseBin(buf);
        if (res.header.available === false) pies.off = key;
        var polys = buildPies(res);
        if (!polys.length && !pies.polys.length) return;
        pies.polys = polys;
        pushOverlays();
      })
      .catch(function () { /* pies are optional */ });
  }

  // One wedge polygon per cell type with at least min_frac of a cell's weight, starting at
  // angle -90° in plot space (y up) and going round as in _build_rctd_pie_shapes.
  function buildPies(res) {
    var h = res.header, xy = res.arrays.xy, frac = res.arrays.frac;
    var k = (h.types || []).length, n = h.n | 0, out = [];
    if (!xy || !frac || !k) return out;
    var r = h.radius || 8, minFrac = h.min_frac || 0.02;
    for (var i = 0; i < n; i++) {
      var cx = xy[2 * i], cy = xy[2 * i + 1], theta = -Math.PI / 2;
      for (var j = 0; j < k; j++) {
        var d = frac[i * k + j] * 2 * Math.PI;
        if (frac[i * k + j] >= minFrac) {
          var steps = Math.max(2, Math.ceil(d / (Math.PI / 12)));
          var pts = [[cx, cy]];
          for (var t = 0; t <= steps; t++) {
            var ang = theta + (d * t) / steps;
            pts.push([cx + r * Math.cos(ang), cy - r * Math.sin(ang)]);   // viewer y is down
          }
          out.push({ points: pts, fill: h.colors[j], fillAlpha: 217, width: 0 });
        }
        theta += d;
      }
    }
    return out;
  }

  function closeImages() {
    if (images && images.ctl) {
      try { images.ctl.destroy(); } catch (e) { /* workers already gone */ }
    }
    images = null;
    if (viewer) {
      try { viewer.setExtraLayers(null); } catch (e) { /* viewer already gone */ }
    }
  }

  function applyImageLayers() {
    if (!viewer) return;
    if (!images || !images.ctl || images.checking || !imageState.visible) {
      viewer.setExtraLayers(null);
      return;
    }
    var ctl = images.ctl;
    var contrast = images.contrast;
    // Drawn from the checklist as it is now: channels beyond the available files, or still
    // waiting for their contrast limits, are left out (ensureContrast redraws when they arrive).
    var channels = imageState.channels.filter(function (c) {
      return c.index < ctl.channels.length && contrast[c.index];
    }).map(function (c) {
      var lim = contrast[c.index];
      // Sidebar brightness (default 2) scales the upper contrast limit.
      var max = lim.min + (lim.max - lim.min) * 2 / Math.max(0.1, imageState.brightness || 2);
      return { index: c.index, on: true, color: c.color, min: lim.min, max: max };
    });
    var state = { visible: true, opacity: imageState.opacity, channels: channels };
    viewer.setExtraLayers(function () { return ctl.layers(state); });
  }

  function fetchMeta() {
    return fetch(PREFIX + "/viewer/meta").then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status + " for /viewer/meta");
      return r.json();
    });
  }

  // Identifies the image files: dataset-version alone also changes when they do not (Rerun UMAP).
  function imageSig(meta) {
    return meta.dataset_gen + "|" + (meta.images || []).map(function (i) {
      return i.name + ":" + (i.v || "");
    }).join(",");
  }

  function syncImages(datasetKey) {
    if (!viewer) return;
    imageKey = datasetKey;
    if (!imageState.visible || !window.XeniumViewer.image) {
      // Switching the overlay off and on again retries a failed open / contrast.
      if (images && images.failed) images = null;
      if (images) images.contrastFailed = {};
      setMsg("image", null);
      setMsg("channels", null);
      applyImageLayers();
      return;
    }
    if (images && images.checking) return;
    if (images && images.key !== datasetKey) {
      if (!images.ctl) {
        closeImages();
      } else {
        // Keep the controller (worker pool, decoded tiles) when the files are the same;
        // nothing is drawn or fetched from it until that is known.
        var cur = images;
        cur.checking = true;
        applyImageLayers();
        fetchMeta().then(function (meta) {
          if (images !== cur) return;
          cur.checking = false;
          if (imageSig(meta) === cur.sig) cur.key = datasetKey;
          else closeImages();
          syncImages(imageKey);
        }).catch(function () {
          if (images === cur) closeImages();
        });
        return;
      }
    }
    if (!images) {
      var mine = (images = { key: datasetKey, sig: null, ctl: null, contrast: {}, contrastFailed: {}, failed: null });
      fetchMeta()
        .then(function (meta) {
          if (!meta.images || !meta.images.length) throw new Error("no morphology images");
          mine.sig = imageSig(meta);
          // The worker and the .wasm are looked up next to the viewer bundle (its script URL).
          return window.XeniumViewer.image.openImages({
            baseUrl: PREFIX, images: meta.images, pixelSize: meta.pixel_size,
          });
        })
        .then(function (ctl) {
          if (images !== mine) { ctl.destroy(); return; }   // dataset changed meanwhile
          mine.ctl = ctl;
          setMsg("image", null);
          ensureContrast();
        })
        .catch(function (err) {
          if (images !== mine) return;
          // Remembered until the dataset changes or the overlay is switched off and on:
          // not retried on every control change.
          mine.failed = "Image unavailable: " + ((err && err.message) || err);
          setMsg("image", mine.failed);
        });
      return;
    }
    if (images.failed) {
      setMsg("image", images.failed);
      return;
    }
    if (images.ctl) ensureContrast();
  }

  // Draw the ticked channels that have contrast limits and fetch default limits for those
  // switched on for the first time (each is drawn when its limits arrive).
  function ensureContrast() {
    var mine = images;
    var available = mine.ctl.channels.length;
    var wanted = imageState.channels.filter(function (c) { return c.index < available; });
    // A ticked channel the bundle has no file for cannot be drawn: say so.
    var absent = imageState.channels.filter(function (c) { return c.index >= available; })
      .map(function (c) { return c.label; });
    setMsg("channels", absent.length ? "no image file in this bundle for: " + absent.join(", ") : null);
    if (!wanted.some(function (c) { return mine.contrastFailed[c.index]; })) setMsg("image", null);
    applyImageLayers();
    wanted.forEach(function (c) {
      if (mine.contrast[c.index] || mine.contrastFailed[c.index]) return;
      mine.ctl.defaultContrast(c.index).then(function (lim) {
        if (images !== mine) return;
        mine.contrast[c.index] = lim;
        applyImageLayers();
      }, function (err) {
        if (images !== mine) return;   // closed meanwhile: the rejection is the pool shutting down
        mine.contrastFailed[c.index] = true;
        setMsg("image", "Image contrast failed: " + ((err && err.message) || err));
      });
    });
  }

  function sync(mode, colorBy, method, gene, size, opacity, toggles, segSource, countsMode,
                cmin, cmax, txGenes, morphEnable, morphChannels, morphOpacity, morphBrightness,
                datasetVer, subsetVer, baysorDone, prosegDone,
                annotDone, spageDone, splitDone, roiDone, nicheDone, tool_, umapPanelStyle,
                roiModalOpen, extraVer, relayout) {
    if (mode === "webgl") {
      var why = webglProblem();
      if (why) {
        fallBackToPlotly(why);
        mode = "plotly";   // this call already lays the panels out for Plotly
      }
    }
    var webgl = mode === "webgl";
    var plotStyle = { height: "100%", display: webgl ? "none" : "block" };
    var glStyle = { position: "absolute", inset: "0", display: webgl ? "block" : "none" };
    document.body.classList.toggle("xv-webgl", webgl);   // its style prop belongs to the plot-size controls
    var umapGlStyle = { flex: "1", minHeight: "420px", position: "relative", display: webgl ? "block" : "none" };
    // Z-Level picks the file the Plotly overlay opens its channels through; the WebGL image
    // layer reads one file per ticked channel, so the control has nothing to select here.
    var styles = [plotStyle, glStyle, umapGlStyle, webgl,
                  webgl ? "Not used by the WebGL view: it reads each ticked channel from its own morphology_focus file" : ""];

    var entered = false, fromPlotly = false;
    if (mode !== lastMode) {
      entered = webgl;
      fromPlotly = lastMode === "plotly";
      lastMode = mode;
      // Let Plotly / deck.gl pick up the size of the panel that just became visible.
      setTimeout(function () {
        window.dispatchEvent(new Event("resize"));
        if (viewer) viewer.resize();
        if (umapViewer) umapViewer.resize();
      }, 60);
      renderStatus();   // blank outside WebGL mode, the current statuses on the way back in
    }
    if (!webgl) return styles;

    var v = null;
    try {
      v = mount();
    } catch (e) {
      fallBackToPlotly("the viewer could not start (" + ((e && e.message) || e) + ")");
      styles[0] = { height: "100%", display: "block" };
      styles[1] = { position: "absolute", inset: "0", display: "none" };
      styles[2] = Object.assign({}, umapGlStyle, { display: "none" });
      styles[3] = false;
      styles[4] = "";
      document.body.classList.remove("xv-webgl");
      return styles;
    }
    if (!v) {
      setMsg("bundle", "WebGL viewer bundle not loaded");
      return styles;
    }
    setMsg("bundle", null);

    if (entered) {
      // Open at the view the Plotly figure had (spatial-relayout holds its ranges, plot y = -µm).
      // The element is still hidden here: the viewer applies this on the resize above.
      var r = relayout || {};
      var x0 = r["xaxis.range[0]"], x1 = r["xaxis.range[1]"];
      var y0 = r["yaxis.range[0]"], y1 = r["yaxis.range[1]"];
      if ([x0, x1, y0, y1].every(function (n) { return typeof n === "number" && isFinite(n); })) {
        v.setViewport({
          x0: Math.min(x0, x1), x1: Math.max(x0, x1),
          y0: Math.min(-y0, -y1), y1: Math.max(-y0, -y1),
        });
      } else if (fromPlotly) {
        v.fitView();   // Plotly was autoranged
      }
    }

    var source = segSource || "xenium";
    var tool = source.split(":")[0];
    toggles = toggles || [];
    var has = function (k) { return toggles.indexOf(k) >= 0; };
    var dataVersion = [datasetVer, subsetVer, baysorDone, prosegDone, source, extraVer || 0].join("|");
    var colorVersion = [dataVersion, annotDone, spageDone, splitDone, roiDone, nicheDone, an.gsVersion].join("|");
    var colormap = COLORMAPS[colorBy] || "viridis";

    v.setState({
      source: source,
      dataVersion: dataVersion,
      color: {
        mode: colorBy || "cluster",
        method: method || "",
        gene: gene || "",
        corrected: countsMode === "corrected" ? 1 : 0,
        min: cmin === undefined || cmin === null || cmin === "" ? null : Number(cmin),
        max: cmax === undefined || cmax === null || cmax === "" ? null : Number(cmax),
        version: colorVersion,
      },
      colormap: colormap,
      pointSize: size || 2,
      opacity: opacity === undefined || opacity === null ? 0.85 : opacity,
      // one polygon set per checkbox, whatever the active source (as in the Plotly view)
      boundaries: { cell: has("cell"), nucleus: has("nucleus"), baysor: has("baysor"), proseg: has("proseg"), imgseg: has("imgseg") },
      transcripts: { genes: tool === "xenium" ? (txGenes || []) : [], qv: 20, size: 2 },
      tool: tool_ || "pan",
      density: densityState(),
    });
    // The lasso highlight and the pending outline go when the Save ROI dialog closes (save,
    // cancel or dismissed), when the ROI set changes, or when a new outline is started.
    var drawTool = !!tool_ && tool_ !== "pan";
    if ((lastRoiOpen && !roiModalOpen) || roiDone !== lastRoiDone || (drawTool && tool_ !== lastTool)) {
      pendingRoi = null;
      v.setState({ selected: cn.cells || null });
      pushOverlays();
    }
    lastRoiOpen = !!roiModalOpen;
    lastRoiDone = roiDone;
    lastTool = tool_;
    cur = { source: source, tool: tool, colorBy: colorBy || "cluster", dataVersion: dataVersion,
            colorVersion: colorVersion,
            overlayBase: [datasetVer, roiDone, source, extraVer || 0].join("|") };
    syncOverlays();
    syncCellnest();
    syncPies();

    // UMAP of the same cells, drawn by a second viewer from /umap/viewer/*.
    var umapOpen = !(umapPanelStyle && umapPanelStyle.display === "none");
    var u = umapOpen ? mountUmap() : null;
    if (u) {
      setTimeout(function () { u.resize(); }, 60);
      var corrected = countsMode === "corrected";
      var jointEmb = tool === "xenium" && /^joint_clusters/.test(method || "");
      u.setState({
        // a joint cluster set brings its own embedding of all samples (served when it exists)
        source: source + (corrected ? "|corr" : "") + (jointEmb ? "|emb:" + method : ""),
        // a joint re-run replaces the coordinates under the same source: its version (the
        // one the poll bumps for the colours) is part of the cells key, so they are refetched
        dataVersion: [dataVersion, splitDone, corrected ? 1 : 0, jointEmb ? an.gsVersion : 0].join("|"),
        color: {
          mode: colorBy || "cluster", method: method || "", gene: gene || "",
          corrected: corrected ? 1 : 0,
          min: cmin === undefined || cmin === null || cmin === "" ? null : Number(cmin),
          max: cmax === undefined || cmax === null || cmax === "" ? null : Number(cmax),
          version: colorVersion,
        },
        colormap: colormap,
        pointSize: Math.max(2, (size || 2) * 1.2),
        opacity: opacity === undefined || opacity === null ? 0.85 : opacity,
      });
    }

    imageState.visible = (morphEnable || []).indexOf("show") >= 0;
    imageState.opacity = morphOpacity === undefined || morphOpacity === null ? 0.85 : morphOpacity;
    imageState.brightness = morphBrightness || 2;
    imageState.channels = (morphChannels || [])
      .filter(function (k) { return CHANNELS[k]; })
      .map(function (k) { return { index: CHANNELS[k].index, color: CHANNELS[k].color, label: CHANNELS[k].label }; });
    syncImages(String(datasetVer));

    return styles;
  }

  // Second callback: controls whose overlays the Plotly view draws on its own figure.
  function extras(mode, measure, sdataRoi, sdataPatches, sdataDone, cnVersion, cnEnable, cnPairs,
                  cnTopN, cnWidth, cnColorBy, cnOptions, fLigand, fReceptor, fSender, fReceiver,
                  annotCol) {
    var on = function (v, k) { return (v || []).indexOf(k) >= 0; };
    ex.measure = measure && measure.active ? measure : null;
    ex.sdataRoi = on(sdataRoi, "yes");
    ex.sdataPatches = on(sdataPatches, "yes");
    ex.sdataDone = sdataDone || 0;
    ex.cnVersion = cnVersion || 0;
    var pairs = cnPairs ? (Array.isArray(cnPairs) ? cnPairs : [cnPairs]) : [];
    ex.cellnest = on(cnEnable, "on") && pairs.length ? JSON.stringify({
      pairs: pairs, top_n: cnTopN || 100, edge_width: cnWidth || 1, color_by: cnColorBy || "lr_pair",
      options: cnOptions || [], ligand: fLigand || null, receptor: fReceptor || null,
      sender: fSender || null, receiver: fReceiver || null, annot_col: annotCol || null,
    }) : null;
    if (mode === "webgl" && viewer && lastMode === "webgl") {
      syncOverlays();
      syncCellnest();
      pushOverlays();
    }
    return window.dash_clientside.no_update;
  }

  // Third callback: the transcript-density heat layer (density-store) and the version of
  // the saved gene sets (a rescored set must be fetched again although Color By is unchanged).
  var an = { density: null, gsVersion: 0 };
  function densityState() {
    var d = an.density || {};
    return {
      on: !!(d.on && d.gene), gene: d.gene || null, bin: d.bin || 10,
      opacity: d.opacity === undefined || d.opacity === null ? 0.7 : d.opacity,
      max: d.max === undefined || d.max === null || d.max === "" ? null : Number(d.max),
      version: d.version || 0,
    };
  }
  function analysis(mode, density, gsVersion) {
    var gsChanged = (gsVersion || 0) !== an.gsVersion;
    an.density = density || null;
    an.gsVersion = gsVersion || 0;
    if (mode === "webgl" && viewer && lastMode === "webgl") {
      viewer.setState({ density: densityState() });
      if (gsChanged) syncOverlays();
      if (gsChanged && cur.colorVersion) {
        var cv = cur.colorVersion.split("|").slice(0, -1).concat([an.gsVersion]).join("|");
        cur.colorVersion = cv;
        viewer.setState({ color: { version: cv } });
        if (umapViewer) umapViewer.setState({ color: { version: cv } });
      }
    }
    return window.dash_clientside.no_update;
  }

  // "PNG" button: the current view at twice the on-screen resolution, with legend and scale
  // bar. Raster only; a vector figure needs the Plotly renderer (its camera button saves SVG).
  function exportPng() {
    var canvas = viewer && lastMode === "webgl" && viewer.snapshot
      ? viewer.snapshot({ scale: 2, scaleBar: true }) : null;
    if (canvas) {
      canvas.toBlob(function (blob) {
        if (!blob) return;
        var a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "xenium_spatial.png";
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(function () { URL.revokeObjectURL(a.href); }, 5000);
      }, "image/png");
    }
    return window.dash_clientside.no_update;
  }

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    xenium_viewer: { sync: sync, extras: extras, exportPng: exportPng, analysis: analysis },
  });
})();
