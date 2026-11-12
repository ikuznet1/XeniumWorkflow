// Diagnostic: CPU-profile the main thread during first load and attribute long tasks.
import { chromium } from "playwright";
const url = process.argv[2] || "http://127.0.0.1:8085/";
const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
const context = await browser.newContext({ viewport: { width: 1200, height: 800 } });
const page = await context.newPage();
const warns = [];
page.on("console", (m) => { if (m.type() !== "log") warns.push(`${m.type()}: ${m.text().slice(0, 160)}`); });
const cdp = await context.newCDPSession(page);
await cdp.send("Profiler.enable");
await cdp.send("Profiler.setSamplingInterval", { interval: 500 });
await cdp.send("Profiler.start");
await page.goto(url);
await page.waitForFunction(() => window.__t && (window.__t.error || window.__t.isIdle), null, { timeout: 120000 });
const t = await page.evaluate(async () => { await window.__t.whenIdle(); return performance.now(); });
const { profile } = await cdp.send("Profiler.stop");
const lt = await page.evaluate(() => window.__t.longTasks.map((l) => [Math.round(l.start), Math.round(l.duration)]));
console.log("first frame ms", Math.round(t), "long tasks [start,dur]", JSON.stringify(lt));
console.log("console (before any screenshot):", JSON.stringify([...new Set(warns)], null, 1));
// self time per function
const dt = profile.timeDeltas; const self = new Map(); const byId = new Map(profile.nodes.map((n) => [n.id, n]));
profile.samples.forEach((id, i) => { const n = byId.get(id); const k = `${n.callFrame.functionName || "(anon)"} @${n.callFrame.url.split("/").pop()}:${n.callFrame.lineNumber}:${n.callFrame.columnNumber}`; self.set(k, (self.get(k) || 0) + dt[i] / 1000); });
console.log([...self.entries()].sort((a, b) => b[1] - a[1]).slice(0, 14).map(([k, v]) => `${v.toFixed(0).padStart(6)} ms  ${k}`).join("\n"));
await browser.close();
