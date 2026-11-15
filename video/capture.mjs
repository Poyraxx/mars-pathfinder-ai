// Captures real app footage (frame sequences) from the local dev server for the demo video.
// Usage: node capture.mjs   (dev server on http://127.0.0.1:5173, backend on :8000)
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';

const OUT = 'public/capture';
const W = 1920, H = 1080;
const chrome = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const sleep = ms => new Promise(r => setTimeout(r, ms));

const browser = await puppeteer.launch({executablePath: chrome, headless: 'new', args: [`--window-size=${W},${H}`, '--use-angle=d3d11', '--enable-webgl', '--ignore-gpu-blocklist'], defaultViewport: {width: W, height: H, deviceScaleFactor: 1}});
const page = await browser.newPage();
await page.goto('http://127.0.0.1:5173/', {waitUntil: 'networkidle2', timeout: 120000});
await sleep(6000);

async function hideChrome(hide) {
  await page.evaluate(h => {
    let s = document.getElementById('cap-style');
    if (!s) { s = document.createElement('style'); s.id = 'cap-style'; document.head.appendChild(s); }
    s.textContent = h ? '.topbar,.sidebar,.analysis,.details,.map-tools,.globe-source,.intro,.relief-legend,.relief-switch,.overlay-switch,.overlay-legend,.cesium-viewer-bottom{display:none!important}.app{grid-template-columns:0 1fr!important;grid-template-rows:0 1fr!important}.stage{grid-column:1/-1!important;grid-row:1/-1!important}' : '';
    window.dispatchEvent(new Event('resize'));
  }, hide);
  await sleep(800);
}
async function waitTiles(max = 25000) {
  const t0 = Date.now();
  while (Date.now() - t0 < max) {
    const ok = await page.evaluate(() => window.__viewer?.scene.globe.tilesLoaded);
    if (ok) break; await sleep(400);
  }
  await sleep(600);
}
async function orbit(dir, {lat, lon, h, pitch, heading0, heading1, frames, back = 1.3}) {
  fs.mkdirSync(`${OUT}/${dir}`, {recursive: true});
  for (let i = 0; i < frames; i++) {
    const t = i / (frames - 1);
    await page.evaluate(({lat, lon, h, pitch, heading, back}) => {
      const v = window.__viewer, M = v.scene.globe.ellipsoid, R = 3396190;
      const hd = heading * Math.PI / 180, d = h * back / R;
      const la = lat * Math.PI / 180 - d * Math.cos(hd), lo = lon * Math.PI / 180 - d * Math.sin(hd) / Math.cos(lat * Math.PI / 180);
      v.camera.setView({destination: M.cartographicToCartesian({longitude: lo, latitude: la, height: h}), orientation: {heading: hd, pitch: pitch * Math.PI / 180, roll: 0}});
      v.scene.requestRender();
    }, {lat, lon, h, pitch, heading: heading0 + (heading1 - heading0) * t, back});
    await waitTiles(i === 0 ? 40000 : 6000);
    await page.screenshot({path: `${OUT}/${dir}/${String(i).padStart(3, '0')}.jpg`, type: 'jpeg', quality: 90});
  }
  console.log(dir, frames);
}

// 1) Relief globe orbit (Explore, relief on).
await page.evaluate(() => window.__mission.getState().set({mode: 'explore', relief: true, layer: 'mola', overlay: null, showAI: false}));
await hideChrome(true); await sleep(3000);
if(!process.env.SKIP_ORBIT){
await orbit('globe', {lat: -5, lon: 137.8, h: 2600000, pitch: -62, heading0: -25, heading1: 10, frames: 60, back: .55});
// 2) Gale crater close relief.
await orbit('gale', {lat: -5.2, lon: 137.6, h: 260000, pitch: -38, heading0: -10, heading1: 12, frames: 50, back: 1.4});
}

// 3) Plan with science stops at Gale (stills of the full UI).
await hideChrome(false);
await page.evaluate(() => { const s = window.__mission.getState(); s.set({relief: false, layer: 'viking', overlay: null}); });
await page.evaluate(async () => {
  const m = window.__mission; const s = m.getState();
  s.set({mode: 'plan'}); await new Promise(r => setTimeout(r, 500));
  const gale = s.catalog.regions.find(r => r.id === 'gale'); m.getState().selectRegion(gale);
  await new Promise(r => setTimeout(r, 1500));
  const t = await (await fetch('/api/science-targets')).json();
  const a = t.gale[0], b = t.gale[2];
  m.getState().set({showAI: false, stops: [{lat: a.lat, lon: a.lon}, {lat: b.lat, lon: b.lon}], stopWork: [a.work_min, b.work_min], stopNames: [a.name, b.name], view: 'terrain'});
  await new Promise(r => setTimeout(r, 800));
  [...document.querySelectorAll('button')].find(x => x.textContent.includes('Calculate route')).click();
});
await sleep(9000); await waitTiles(30000);
fs.mkdirSync(`${OUT}/plan`, {recursive: true});
await page.screenshot({path: `${OUT}/plan/route.jpg`, type: 'jpeg', quality: 92});
await page.evaluate(() => window.__mission.getState().set({overlay: 'dust', view: 'orbital', exploreTarget: {lat: -4.8, lon: 137.6, alt: 2200000}}));
await sleep(6000); await waitTiles(30000);
await page.screenshot({path: `${OUT}/plan/dust.jpg`, type: 'jpeg', quality: 92});
await page.evaluate(() => window.__mission.getState().set({overlay: 'geology', exploreTarget: {lat: 0, lon: 120, alt: 9000000}}));
await sleep(6000); await waitTiles(30000);
await page.screenshot({path: `${OUT}/plan/geology.jpg`, type: 'jpeg', quality: 92});
const eva = await page.$('.analysis');
if (eva) await eva.screenshot({path: `${OUT}/plan/eva.png`});
console.log('plan stills');
await browser.close();
