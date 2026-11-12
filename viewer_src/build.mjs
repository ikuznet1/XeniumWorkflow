// Builds the viewer bundle into ../assets/xenium_viewer.js (Dash auto-loads assets/*.js).
//   node build.mjs            one-off minified build
//   node build.mjs --watch    rebuild on change
//   node build.mjs --dev      unminified, with inline source map
//   node build.mjs --with-image   also bundle src/image (Viv morphology layer) as XeniumViewer.image,
//                                 sharing this bundle's deck.gl; default is the core alone
//   node build.mjs --out=path     write somewhere else (e.g. to try a build without touching assets/)
import * as esbuild from 'esbuild';
import {existsSync, statSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const outArg = process.argv.find((a) => a.startsWith('--out='));
const outfile = outArg ? resolve(outArg.slice(6)) : resolve(here, '..', 'assets', 'xenium_viewer.js');
const dev = process.argv.includes('--dev');
const watch = process.argv.includes('--watch');
const withImage = process.argv.includes('--with-image');
// The image module ships a shim that drops unused Viv loader code; use it when present.
const vivShim = resolve(here, 'src', 'image', 'viv-loaders-shim.js');

const options = {
  entryPoints: [resolve(here, 'src', withImage ? 'index_with_image.js' : 'index.js')],
  alias: withImage && existsSync(vivShim) ? {'@vivjs/loaders': vivShim} : {},
  outfile,
  bundle: true,
  format: 'iife',
  platform: 'browser',
  target: ['es2020'],
  minify: !dev,
  sourcemap: dev ? 'inline' : false,
  legalComments: 'none',
  define: {'process.env.NODE_ENV': '"production"'},
  logLevel: 'info',
};

if (watch) {
  const ctx = await esbuild.context(options);
  await ctx.watch();
  console.log('watching src/ ...');
} else {
  await esbuild.build(options);
  const kb = statSync(outfile).size / 1024;
  console.log(`${outfile}: ${kb.toFixed(1)} kB`);
}
