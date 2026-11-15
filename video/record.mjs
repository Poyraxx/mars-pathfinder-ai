// Real-time screen recordings of the running app with a visible cursor (no stills, no mock-ups).
// Usage: node record.mjs [plan|conditions|replay ...]  — dev server :5173 and backend :8000 must be running.
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';

const W = 1920, H = 1080;
const FFMPEG = process.env.FFMPEG || 'C:/Users/osura/AppData/Local/mars-pathfinder/venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v9.0.2.exe';
const OUT = 'public/rec';
const sleep = ms => new Promise(r => setTimeout(r, ms));
const which = process.argv.slice(2).length ? process.argv.slice(2) : ['plan', 'conditions', 'replay'];
fs.mkdirSync(OUT, {recursive: true});

const browser = await puppeteer.launch({executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: 'new',
  args: [`--window-size=${W},${H}`, '--use-angle=d3d11', '--enable-webgl', '--ignore-gpu-blocklist', '--hide-scrollbars'], defaultViewport: {width: W, height: H}});
const page = await browser.newPage();

async function fresh(state) {
  await page.goto('http://127.0.0.1:5173/', {waitUntil: 'networkidle2', timeout: 120000});
  await page.addStyleTag({content: '#rec-cursor{position:fixed;z-index:99999;left:0;top:0;width:26px;height:26px;pointer-events:none;transform:translate(-3px,-2px);transition:transform .08s} #rec-cursor svg{filter:drop-shadow(0 2px 3px rgba(0,0,0,.6))} #rec-click{position:fixed;z-index:99998;width:34px;height:34px;margin:-17px 0 0 -17px;border-radius:50%;border:2px solid #3ce1ff;pointer-events:none;opacity:0}'});
  await page.evaluate(() => {
    const c = document.createElement('div'); c.id = 'rec-cursor';
    c.innerHTML = '<svg width="26" height="26" viewBox="0 0 26 26"><path d="M3 2 L3 21 L8.5 16 L12.5 24 L15.5 22.6 L11.6 14.8 L19 14.6 Z" fill="#fff" stroke="#111" stroke-width="1.4" stroke-linejoin="round"/></svg>';
    document.body.appendChild(c);
    const r = document.createElement('div'); r.id = 'rec-click'; document.body.appendChild(r);
    document.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
    document.addEventListener('mousedown', e => { r.style.left = e.clientX + 'px'; r.style.top = e.clientY + 'px'; r.animate([{opacity: .9, transform: 'scale(.4)'}, {opacity: 0, transform: 'scale(1.4)'}], {duration: 450}); }, true);
  });
  await page.mouse.move(W * .55, H * .55);
  await sleep(4000);
  if (state) await page.evaluate(state);
  await sleep(1500);
}
async function waitTiles(max = 20000) {
  const t0 = Date.now();
  while (Date.now() - t0 < max) { if (await page.evaluate(() => window.__viewer?.scene.globe.tilesLoaded)) break; await sleep(300); }
}
async function moveTo(sel, {text, dx = 0, dy = 0, steps = 28} = {}) {
  const box = await page.evaluate(({sel, text}) => {
    const el = [...document.querySelectorAll(sel)].find(e => !text || e.textContent.includes(text));
    if (!el) return null; el.scrollIntoView({block: 'nearest'}); const b = el.getBoundingClientRect(); return {x: b.x + b.width / 2, y: b.y + b.height / 2};
  }, {sel, text});
  if (!box) throw new Error(`not found: ${sel} ${text || ''}`);
  await page.mouse.move(box.x + dx, box.y + dy, {steps});
  return box;
}
async function click(sel, opts = {}) { await moveTo(sel, opts); await sleep(250); await page.mouse.down(); await sleep(70); await page.mouse.up(); await sleep(350); }
async function record(name, fn) {
  const rec = await page.screencast({path: `${OUT}/${name}.webm`, ffmpegPath: FFMPEG});
  await fn();
  await rec.stop();
  console.log('recorded', name);
}

if (which.includes('plan')) {
  await fresh(() => { const m = window.__mission, s = m.getState(); s.set({mode: 'plan', showAI: false, relief: false, layer: 'viking', overlay: null}); });
  await page.evaluate(() => { const m = window.__mission; m.getState().selectRegion(m.getState().catalog.regions.find(r => r.id === 'gale')); m.getState().set({view: 'terrain', showAI: false}); });
  await sleep(4000); await waitTiles();
  await record('plan', async () => {
    await sleep(600);
    if (!(await page.evaluate(() => document.querySelector('.science-details')?.open))) await click('.science-details summary');
    else await moveTo('.science-details summary');
    await sleep(500);
    await click('.science-list button', {text: '30 min'});
    await sleep(400);
    await click('.science-list button', {text: '45 min'});
    await sleep(400);
    await click('button', {text: 'Calculate route'});
    for (let i = 0; i < 40; i++) { if (await page.evaluate(() => window.__mission.getState().mission?.eva?.status === 'ready')) break; await sleep(250); }
    await sleep(2200);
    await page.mouse.move(W * .62, H * .55, {steps: 30});
    await sleep(1800);
  });
}

if (which.includes('conditions')) {
  await fresh(() => { const s = window.__mission.getState(); s.set({mode: 'explore', relief: true, layer: 'mola', overlay: null, showAI: false, exploreTarget: {lat: -5, lon: 137.5, alt: 2400000}, view: 'orbital'}); });
  await sleep(5000); await waitTiles(30000);
  await record('conditions', async () => {
    await sleep(1200);
    await click('.overlay-switch>button');
    await sleep(500);
    await click('.overlay-menu button', {text: 'Dust cover'});
    await sleep(500); await waitTiles(8000); await sleep(1800);
    await click('.overlay-switch>button');
    await sleep(400);
    await click('.overlay-menu button', {text: 'Geology'});
    await sleep(2600);
  });
}

if (which.includes('replay')) {
  await fresh(() => { window.__mission.getState().set({mode: 'replay', showAI: false}); });
  for (let i = 0; i < 60; i++) { if (await page.evaluate(() => !!window.__mission.getState().localBatch)) break; await sleep(500); }
  await page.evaluate(() => { const s = window.__mission.getState(), r = s.replay; s.set({replayStep: 2, frameId: r.steps[2].frame_id}); });
  await sleep(4000);
  await record('replay', async () => {
    await sleep(800);
    await click('button[aria-label="Play"]');
    await sleep(5200);
    await click('button[aria-label="Pause"]');
    await sleep(1500);
  });
}
await browser.close();
