// Quick smoke test: does the dev page render cells? Usage: node test/smoke.mjs [skin|tiny] [port]
import {DATA, launch, newPage, shot, startServer} from './lib.mjs';

const which = process.argv[2] || 'skin';
const port = +(process.argv[3] || 8071);
const server = await startServer(DATA[which], port);
const browser = await launch();
try {
  const {page, consoleIssues, requests} = await newPage(browser);
  await page.goto(server.url + '/');
  await page.waitForFunction(() => window.viewer && window.viewer.stats.firstRenderMs !== undefined, null, {timeout: 60000});
  await page.evaluate(() => window.viewer.whenIdle());
  const info = await page.evaluate(() => ({
    gl: (() => { const c = document.createElement('canvas').getContext('webgl2'); return c ? c.getParameter(c.VERSION) : null; })(),
    stats: window.viewer.stats, vp: window.viewer.getViewport(), status: window.viewer.getStatus(),
  }));
  console.log(JSON.stringify(info));
  console.log(JSON.stringify(await shot(page, `smoke_${which}.png`)));
  console.log('requests', requests.filter((u) => u.includes('/viewer/')).map((u) => u.replace(server.url, '')));
  console.log('console issues', consoleIssues);
} finally {
  await browser.close();
  server.stop();
}
