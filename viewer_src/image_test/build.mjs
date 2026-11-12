// Build the morphology-image layer for testing. Output goes to image_test/dist/ only.
//
//   node image_test/build.mjs
//
// dist/xenium_image_worker.js      JPEG 2000 decode worker (classic script; ship as an asset)
// dist/openjpegwasm_decode.wasm    OpenJPEG decoder (ship as an asset next to the worker)
// dist/image_test_app.js           test page bundle (deck.gl + Viv + src/image); NOT for shipping
// dist/bench/xenium_image_worker.js  worker that also contains the asm.js codec (timing comparison only)
import { build } from "esbuild";
import { copyFileSync, mkdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const dist = join(here, "dist");
mkdirSync(join(dist, "bench"), { recursive: true });

const common = {
  bundle: true,
  format: "iife",
  platform: "browser",
  target: "es2020",
  minify: true,
  legalComments: "none",
  logLevel: "warning",
  absWorkingDir: root,
};

// The emscripten glue has a Node branch with require("fs"/"path"); never taken in a browser.
const worker = (outfile, jsFallback) => build({
  ...common,
  entryPoints: [join(root, "src/image/jp2k.worker.js")],
  outfile,
  external: ["fs", "path"],
  define: { __XV_INCLUDE_JS_FALLBACK__: jsFallback ? "true" : "false" },
});

await worker(join(dist, "xenium_image_worker.js"), false);
await worker(join(dist, "bench/xenium_image_worker.js"), true);
for (const d of [dist, join(dist, "bench")]) {
  copyFileSync(
    join(root, "node_modules/@cornerstonejs/codec-openjpeg/dist/openjpegwasm_decode.wasm"),
    join(d, "openjpegwasm_decode.wasm"),
  );
}

const meta = await build({
  ...common,
  entryPoints: [join(here, "app.js")],
  outfile: join(dist, "image_test_app.js"),
  metafile: true,
  sourcemap: false,
  // Optional but recommended for the real bundle too: saves ~0.2 MB (minified) of unused Viv loader code.
  alias: process.argv.includes("--no-shim") ? {} : { "@vivjs/loaders": join(root, "src/image/viv-loaders-shim.js") },
});

for (const f of ["xenium_image_worker.js", "openjpegwasm_decode.wasm", "image_test_app.js", "bench/xenium_image_worker.js"]) {
  console.log(f.padEnd(34), statSync(join(dist, f)).size, "bytes");
}
// How much of the test bundle is this module's own dependency closure (Viv + geotiff)?
const by = {};
for (const [file, info] of Object.entries(meta.metafile.outputs[Object.keys(meta.metafile.outputs)[0]].inputs)) {
  const m = /node_modules\/((?:@[^/]+\/)?[^/]+)/.exec(file);
  const k = m ? m[1] : (file.startsWith("src/image") ? "src/image" : "test page");
  by[k] = (by[k] || 0) + info.bytesInOutput;
}
console.log(Object.entries(by).sort((a, b) => b[1] - a[1]).slice(0, 14).map(([k, v]) => `  ${k}: ${v}`).join("\n"));
