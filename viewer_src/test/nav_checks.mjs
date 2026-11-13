// The two-level navigation of the plots panel, in the real app (either renderer):
// groups and tabs, the tab remembered per group, plots-tabs.active_tab staying the source
// of truth, the shared "Samples & conditions…" entry, the in-place availability notes,
// figures of hidden tabs taking their size when shown, the jobs panel staying clear of
// the Run buttons, no horizontal overflow at a laptop viewport, and one screenshot per
// tab (1440 x 900) for a visual check.
//   [XENIUM_RENDERER=webgl] PORT=8171 DATA_DIR=<bundle copy> XENIUM_CACHE_DIR=<dir> python xenium_explorer.py
//   node test/nav_checks.mjs http://127.0.0.1:8171 /path/to/screenshot_dir [prefix]
import {chromium} from 'playwright';
import {openTab, activeTab, showPlotsPanel, GL_ARGS} from './lib.mjs';

const base = process.argv[2] || 'http://127.0.0.1:8050';
const out = process.argv[3] || '.';
const prefix = process.argv[4] || 'nav';
const browser = await chromium.launch({args: GL_ARGS});
let failures = 0, n = 0;
const check = (name, ok, detail = '') => {
  n += 1;
  if (!ok) failures += 1;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
};
const page = await browser.newPage({viewport: {width: 1440, height: 900}});
const logs = [];
page.on('pageerror', (e) => logs.push('pageerror: ' + e.message.slice(0, 300)));
page.on('console', (m) => { if (m.type() === 'error') logs.push('console: ' + m.text().slice(0, 300)); });
const posts = [];           // _dash-update-component requests: which callbacks a tab switch fires
const slow = [];            // [output, ms] of the callbacks a tab switch triggered
page.on('requestfinished', (r) => {
  if (!r.url().includes('_dash-update-component')) return;
  try {
    const body = JSON.parse(r.postData());
    if (!(body.changedPropIds || []).includes('plots-tabs.active_tab')) return;   // polls etc. are not tab switches
    posts.push(body.output);
    slow.push([body.output, Math.round(r.timing().responseEnd)]);
  } catch { /* ignore */ }
});
await page.goto(base, {waitUntil: 'domcontentloaded'});
await page.waitForSelector('#plots-nav', {state: 'attached', timeout: 180000});
await page.waitForFunction(() => window.dash_clientside && window.dash_clientside.set_props, null, {timeout: 60000});
await page.waitForTimeout(4000);
const text = (sel) => page.evaluate((s) => document.querySelector(s)?.innerText || '', sel);
const setProp = (id, props) => page.evaluate(([i, p]) => window.dash_clientside.set_props(i, p), [id, props]);

const structure = await page.evaluate(() => ({
  groups: [...document.querySelectorAll('#nav-groups .nav-group')].map((e) => [e.id.replace('nav-group-', ''), e.innerText.trim()]),
  tabs: [...document.querySelectorAll('#nav-subtabs .nav-sub')].map((e) => e.id.replace('nav-tab-', '')),
  panes: document.querySelectorAll('#umap-panel .tab-content > .tab-pane').length,
  flatStrip: getComputedStyle(document.querySelector('#plots-tabs')).display,
}));
check('six groups: Overview, QC, Expression, Neighbourhoods, Communication, Provenance',
  structure.groups.map((g) => g[1]).join('|') === 'Overview|QC|Expression|Neighbourhoods|Communication|Provenance',
  structure.groups.map((g) => g[1]).join('|'));
check('every tab of plots-tabs has a button in the navigation and the flat strip is hidden',
  structure.tabs.length === structure.panes && structure.flatStrip === 'none', `${structure.tabs.length} buttons, ${structure.panes} panes`);

await showPlotsPanel(page);
check('first view: Overview / UMAP', (await activeTab(page)) === 'tab-umap');

// a group click opens its first tab; the tab last used in a group is remembered
await page.click('#nav-group-expression');
await page.waitForFunction(() => document.querySelector('#nav-tab-tab-de')?.classList.contains('active'), null, {timeout: 10000});
check('a group opens on its first tab', (await activeTab(page)) === 'tab-de');
await page.click('#nav-tab-tab-moran');
await page.waitForTimeout(400);
await page.click('#nav-group-qc');
await page.waitForTimeout(400);
const qcFirst = await activeTab(page);
await page.click('#nav-group-expression');
await page.waitForTimeout(400);
check('the last tab of a group is remembered', qcFirst === 'tab-segqc' && (await activeTab(page)) === 'tab-moran', `${qcFirst}, ${await activeTab(page)}`);
const onlyOne = await page.evaluate(() => [...document.querySelectorAll('#umap-panel .tab-content > .tab-pane')]
  .filter((p) => getComputedStyle(p).display !== 'none').length);
check('exactly one tab body is shown', onlyOne === 1, String(onlyOne));

// plots-tabs.active_tab set from outside (another callback, a test) moves the bar
await setProp('plots-tabs', {active_tab: 'tab-coloc'});
await page.waitForFunction(() => document.querySelector('#nav-tab-tab-coloc')?.classList.contains('active'), null, {timeout: 10000}).catch(() => {});
check('the bar follows plots-tabs.active_tab', (await activeTab(page)) === 'tab-coloc'
  && await page.evaluate(() => document.querySelector('#nav-group-neighbourhoods').classList.contains('active')));

// the shared design entry
check('the design entry is shown on a tab that uses the design, with the summary',
  await page.isVisible('#nav-design-open-btn') && (await text('#nav-design-summary')).length > 10, (await text('#nav-design-summary')).slice(0, 90));
check('the per-tab copies of the button are not shown', !(await page.isVisible('#coloc-design-open-btn')));
await page.click('#nav-design-open-btn');
await page.waitForSelector('#design-modal', {state: 'visible', timeout: 15000}).catch(() => {});
check('it opens the design editor', await page.isVisible('#design-modal'));
await page.click('#design-close-btn');
await page.waitForSelector('#design-modal', {state: 'hidden', timeout: 15000}).catch(() => {});
await openTab(page, 'tab-umap');
check('no design entry on a tab that does not use the design', !(await page.isVisible('#nav-design-open-btn')));

// every tab: opens, no horizontal overflow of the panel, Run button not under the jobs panel
const allTabs = structure.tabs;
const report = [];
for (const t of allTabs) {
  posts.length = 0;
  await openTab(page, t, {options: false, settle: 1500});
  const info = await page.evaluate((tab) => {
    const panel = document.querySelector('#umap-panel');
    const pane = [...document.querySelectorAll('#umap-panel .tab-content > .tab-pane')].find((p) => getComputedStyle(p).display !== 'none');
    const pr = panel.getBoundingClientRect();
    let overflowX = 0;
    const offenders = [];
    pane.querySelectorAll('*').forEach((el) => {
      const r = el.getBoundingClientRect();
      if (!r.width || !r.height) return;
      // inside a scroller of its own (a table that scrolls sideways) is fine
      let p = el.parentElement, scrolled = false;
      while (p && p !== pane) {
        const ox = getComputedStyle(p).overflowX;
        if (ox === 'auto' || ox === 'scroll' || ox === 'hidden') { scrolled = true; break; }
        p = p.parentElement;
      }
      if (!scrolled && r.right > pr.right + 2) {
        overflowX = Math.max(overflowX, r.right - pr.right);
        if (offenders.length < 3) offenders.push(el.id || el.className || el.tagName);
      }
    });
    const jobs = document.querySelector('#jobs-panel');
    const jr = jobs && getComputedStyle(jobs).display !== 'none' ? jobs.getBoundingClientRect() : null;
    const notice = document.querySelector('#nav-notice');
    const plot = [...pane.querySelectorAll('.js-plotly-plot')].find((p) => p.offsetParent !== null);   // the shown one
    const plotBox = plot ? plot.getBoundingClientRect() : null;
    const svg = plot ? plot.querySelector('.main-svg') : null;
    const svgBox = svg ? svg.getBoundingClientRect() : null;
    // the tab body ends inside the panel and, when it is taller, scrolls there
    const body = pane.firstElementChild && pane.firstElementChild.firstElementChild;
    const paneBelow = Math.round(pane.getBoundingClientRect().bottom - pr.bottom);
    const scrollers = body ? [body, ...body.querySelectorAll('*')].filter((e) => e.scrollHeight > e.clientHeight + 2
      && ['auto', 'scroll'].includes(getComputedStyle(e).overflowY)) : [];
    const cut = body ? [...body.children].filter((c) => c.getBoundingClientRect().bottom > pr.bottom + 2).length : 0;
    return {
      tab, overflowX: Math.round(overflowX), offenders, jobsShown: !!jr, paneBelow, cut, canScroll: scrollers.length > 0,
      notice: getComputedStyle(notice).display !== 'none' ? notice.innerText.slice(0, 80) : '',
      docOverflow: document.documentElement.scrollWidth - window.innerWidth,
      plot: plotBox && svgBox ? [Math.round(plotBox.width), Math.round(plotBox.height), Math.round(svgBox.width), Math.round(svgBox.height)] : null,
    };
  }, t);
  info.posts = posts.slice();
  report.push(info);
  await page.screenshot({path: `${out}/${prefix}_${t}.png`});
}
const over = report.filter((r) => r.overflowX > 0 || r.docOverflow > 0);
check('no tab overflows the panel or the page sideways (1440 x 900)', over.length === 0,
  over.map((r) => `${r.tab}: +${r.overflowX}px ${r.offenders.join(',')} doc+${r.docOverflow}`).join('; '));
const clipped = report.filter((r) => r.paneBelow > 1 || (r.cut > 0 && !r.canScroll));
check('every tab body ends inside the panel; what does not fit scrolls there (nothing is cut off out of reach)', clipped.length === 0,
  clipped.map((r) => `${r.tab}: pane +${r.paneBelow}px, ${r.cut} blocks below the edge, scroll ${r.canScroll}`).join('; '));
// a figure that was built while its tab was hidden fills its container once shown
const sized = report.filter((r) => r.plot);
const badSize = sized.filter((r) => r.plot[0] < 50 || r.plot[1] < 50 || Math.abs(r.plot[0] - r.plot[2]) > 4 || Math.abs(r.plot[1] - r.plot[3]) > 4);
check(`figures of tabs that were hidden take their container's size when shown (${sized.length} tabs with a figure)`,
  sized.length > 0 && badSize.length === 0, badSize.map((r) => `${r.tab}: box ${r.plot.slice(0, 2)} svg ${r.plot.slice(2)}`).join('; '));
// a tab switch asks the server only for option lists / summaries / notes, never for a result
const heavy = /figure|\.data\b|plot/;
const fired = [...new Set(report.flatMap((r) => r.posts))];
const suspicious = fired.filter((o) => heavy.test(o) && !/prov-table\.data/.test(o));   // the record list is read from a small file
check('a tab switch requests no figure or result table from the server (option lists, summaries and notes only)',
  suspicious.length === 0, suspicious.join(' | ').slice(0, 300));
const worst = slow.slice().sort((x, y) => y[1] - x[1])[0] || ['-', 0];
const limit = Number(process.env.XN_MAX_MS || 3000);
check(`the slowest callback of a tab switch answers within ${limit} ms`, worst[1] < limit, `${worst[1]} ms: ${String(worst[0]).slice(0, 80)}`);
console.log('INFO  callbacks fired by tab switches: ' + fired.map((o) => o.slice(0, 70)).join(' | '));
console.log('INFO  availability notes: ' + report.filter((r) => r.notice).map((r) => `${r.tab}: ${r.notice}`).join(' || '));

// a figure that is REPLACED while its tab is hidden (a job finishing in the background) is
// laid out for a hidden box; it must take the real size when its tab is opened
await openTab(page, 'tab-umap', {settle: 300});
for (const [tabId, graph] of [['tab-composition', 'comp-plot'], ['tab-coloc', 'coloc-plot'], ['tab-segqc', 'segqc-plot']]) {
  await setProp(graph, {figure: {data: [{type: 'bar', x: ['a', 'b', 'c'], y: [1, 3, 2]}],
    layout: {autosize: true, margin: {l: 30, r: 10, t: 10, b: 30}, paper_bgcolor: '#0d1117', plot_bgcolor: '#0d1117'}}});
  await page.waitForTimeout(600);
  await openTab(page, tabId, {options: false, settle: 1500});
  const sz = await page.evaluate((g) => {
    const box = document.getElementById(g).getBoundingClientRect();
    const svg = document.querySelector(`#${g} .main-svg`)?.getBoundingClientRect();
    const bars = document.querySelectorAll(`#${g} .bars .point`).length;
    return {box: [Math.round(box.width), Math.round(box.height)], svg: svg ? [Math.round(svg.width), Math.round(svg.height)] : null, bars};
  }, graph);
  check(`${graph}: a figure set while its tab was hidden fills its box when the tab is opened`,
    sz.svg && sz.bars === 3 && sz.box[0] > 100 && sz.box[1] > 100 && Math.abs(sz.box[0] - sz.svg[0]) <= 4 && Math.abs(sz.box[1] - sz.svg[1]) <= 4,
    JSON.stringify(sz));
  await openTab(page, 'tab-umap', {settle: 300});
}

// jobs panel: when it shows, no Run button of the open tab is underneath it
await setProp('jobs-panel', {style: {position: 'fixed', right: '12px', bottom: '12px', width: '340px', maxHeight: '60vh', zIndex: 2000, display: 'block'}});
void 0;
const covered = [];
for (const t of allTabs) {
  await openTab(page, t, {options: false, settle: 250});
  const hit = await page.evaluate(() => {
    const jobs = document.querySelector('#jobs-panel');
    if (!jobs || getComputedStyle(jobs).display === 'none') return null;
    const jr = jobs.getBoundingClientRect();
    const pane = [...document.querySelectorAll('#umap-panel .tab-content > .tab-pane')].find((p) => getComputedStyle(p).display !== 'none');
    const bad = [];
    pane.querySelectorAll('button[id$="-run-btn"]').forEach((b) => {
      const r = b.getBoundingClientRect();
      if (!r.width) return;
      if (r.left < jr.right && r.right > jr.left && r.top < jr.bottom && r.bottom > jr.top) bad.push(b.id);
    });
    return bad;
  });
  if (hit && hit.length) covered.push(`${t}: ${hit.join(',')}`);
}
check('the jobs panel does not cover a Run button on any tab', covered.length === 0, covered.join('; '));

check('no page errors', logs.length === 0, logs.slice(0, 3).join(' | '));
console.log(`${n - failures}/${n} passed`);
await browser.close();
process.exit(failures ? 1 : 0);
