// Control experiments for the headless SwiftShader backend, using plain WebGL2 only
// (no deck.gl, no viewer bundle):
//  1. the "GPU stall due to ReadPixels" console warning appears for any WebGL page;
//  2. requestAnimationFrame stops being serviced after a second large instanced draw
//     (100k instances), which is why the suite's default backend on macOS is ANGLE-Metal.
// Usage: XV_GL=swiftshader node test/gl_noise.mjs
import {launch, newPage, GL_MODE} from './lib.mjs';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
for (const instances of [2000, 100000]) {
  const browser = await launch();
  const {page, consoleIssues} = await newPage(browser);
  await page.setContent(`<canvas id=c width=1280 height=800></canvas><script>
  window.raf = 0; (function f(){ window.raf++; requestAnimationFrame(f); })();
  const gl = document.getElementById('c').getContext('webgl2');
  const sh = (t, s) => { const o = gl.createShader(t); gl.shaderSource(o, s); gl.compileShader(o); return o; };
  const p = gl.createProgram();
  gl.attachShader(p, sh(gl.VERTEX_SHADER, '#version 300 es\\nin vec2 q; in vec2 pos; void main(){ gl_Position = vec4(pos + q * 0.004, 0., 1.); }'));
  gl.attachShader(p, sh(gl.FRAGMENT_SHADER, '#version 300 es\\nprecision mediump float; out vec4 c; uniform float t; void main(){ c = vec4(t, 0.6, 0.3, 1.); }'));
  gl.linkProgram(p); gl.useProgram(p);
  const N = ${instances};
  const quad = new Float32Array([-1,-1, 1,-1, -1,1, 1,1]);
  const pos = new Float32Array(2 * N); for (let i = 0; i < 2 * N; i++) pos[i] = Math.random() * 2 - 1;
  const bind = (name, data, div) => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    const l = gl.getAttribLocation(p, name); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, 2, gl.FLOAT, false, 0, 0); gl.vertexAttribDivisor(l, div); };
  bind('q', quad, 0); bind('pos', pos, 1);
  window.draw = (t) => { gl.uniform1f(gl.getUniformLocation(p, 't'), t); gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT); gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, N); };
  window.draw(0.2);
  </script>`);
  await sleep(1500);
  const before = await page.evaluate(() => window.raf);
  await page.evaluate(() => window.draw(0.9)); // second frame
  await sleep(3000);
  const after = await page.evaluate(() => window.raf);
  console.log(JSON.stringify({backend: GL_MODE, instances, rafIn1500msBeforeSecondDraw: before, rafIn3000msAfterSecondDraw: after - before,
    readPixelsWarnings: consoleIssues.filter((m) => /ReadPixels/.test(m)).length}));
  await Promise.race([browser.close(), sleep(5000)]);
}
process.exit(0);
