// A morphology OME-TIFF without pyramid levels must not be read whole for the default
// contrast (or for Viv's backdrop): a few tiles are sampled instead.
//   node image_test/single_level.mjs [port]      (default 8097; needs tifffile + imagecodecs)
// Builds a 3000 x 2400 single-level JPEG 2000 file in a temporary bundle, serves it with
// dev_server.py and opens it through assets/xenium_viewer.js.
import { chromium } from "playwright";
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const port = Number(process.argv[2] || 8097);
// the bundle: XV_TINY, else <XV_TEST_DATA>/prime_mouse_ileum_tiny (as test/lib.mjs), else the repository fixture
const testData = process.env.XV_TEST_DATA || resolve(homedir(), "Documents", "XeniumTestData");
const tinyIn = resolve(testData, "prime_mouse_ileum_tiny");
const tiny = process.env.XV_TINY || (existsSync(tinyIn) ? tinyIn
  : resolve(here, "..", "..", "tests", "data", "prime_mouse_ileum_tiny"));
const dir = mkdtempSync(join(process.env.XV_TMP || tmpdir(), "xv-single-"));
const py = process.env.XV_PYTHON || "python";
const made = spawnSync(py, ["-c", `
import os, shutil, sys, numpy as np, tifffile
d, tiny = sys.argv[1], sys.argv[2]
shutil.copy(os.path.join(tiny, "cells.parquet"), d)
os.makedirs(os.path.join(d, "morphology_focus"))
rng = np.random.default_rng(0)
img = (rng.integers(100, 3000, (2400, 3000))).astype(np.uint16)
tifffile.imwrite(os.path.join(d, "morphology_focus", "single.ome.tif"), img, tile=(512, 512), compression="jpeg2000")
`, dir, tiny], { encoding: "utf8" });
if (made.status !== 0) { console.error(made.stderr); process.exit(2); }

const server = spawn(py, [resolve(here, "..", "dev_server.py"), dir, String(port)], { stdio: "ignore" });
const url = `http://127.0.0.1:${port}`;
let failures = 0;
const check = (name, ok, detail = "") => { if (!ok) failures++; console.log(`${ok ? "PASS" : "FAIL"}  ${name}  ${detail}`); };
let browser;
try {
  for (let i = 0; ; i++) {
    try { if ((await fetch(`${url}/viewer/image/0`, { headers: { Range: "bytes=0-1" } })).status === 206) break; } catch { /* not up yet */ }
    if (i > 240) throw new Error("dev server did not start");
    await new Promise((r) => setTimeout(r, 250));
  }
  browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
  const page = await browser.newPage();
  await page.goto(`${url}/`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.XeniumViewer && window.XeniumViewer.image);
  const res = await page.evaluate(async () => {
    const ctl = await window.XeniumViewer.image.openImages({ baseUrl: "", images: [{ index: 0, name: "single.ome.tif" }] });
    const lim = await ctl.defaultContrast(0);
    const layers = ctl.layers({ visible: true, opacity: 1, channels: [{ index: 0, on: true, min: lim.min, max: lim.max }] });
    const out = { levels: ctl.channels[0].levels, size: [ctl.channels[0].width, ctl.channels[0].height], lim,
      requests: ctl.stats.requests, excludeBackground: layers[0].props.excludeBackground };
    ctl.destroy();
    return out;
  });
  check("single-level file opens", res.levels === 1 && res.size[0] === 3000 && res.size[1] === 2400, JSON.stringify(res));
  check("default contrast samples at most 9 of the 30 tiles", res.requests > 0 && res.requests <= 9, `${res.requests} tile requests`);
  check("sampled limits are sensible", res.lim.min >= 100 && res.lim.min < 300 && res.lim.max > 2800 && res.lim.max <= 3000, JSON.stringify(res.lim));
  check("no full-raster backdrop for a file without an overview", res.excludeBackground === true);
} finally {
  if (browser) await browser.close();
  server.kill("SIGKILL");
  rmSync(dir, { recursive: true, force: true });
}
console.log(failures === 0 ? "ALL PASS" : `FAILURES: ${failures}`);
process.exit(failures === 0 ? 0 : 1);
