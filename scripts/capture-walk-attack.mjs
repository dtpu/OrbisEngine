// R9-style attack on ?walk=1, on the presets exactly as they ship: from the start pose hold W for
// SECS at each of 8 yaws (turned through the drag path), and report how far the walker got, how
// many samples stood on unknown floor, how far from known floor it ended (grid cells), and whether
// it ever stepped further into the void than it stood. Then stairs2 straight ahead (the climb),
// R after the climb (walk.floor must return to the start floor), the elevator right strafe, and
// the atrium start (must move). Coverage per grid is printed.
//   bun scripts/capture-walk-attack.mjs [--clips elevator,lobby,stairs2,atrium,hpwide] [--secs 6] [--out dir]
import { mkdir } from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright-core';

const rest = process.argv.slice(2);
const flag = (name, d) => { const i = rest.indexOf('--' + name); return i < 0 ? d : rest[i + 1]; };
const OUT = flag('out', path.resolve('.context/evidence/walk'));
const BASE = flag('base', 'http://127.0.0.1:5399');
const CLIPS = flag('clips', 'elevator,lobby,stairs2,atrium,hpwide').split(',');
const SECS = +flag('secs', 6), SAMPLE_MS = 150, LOOK_RAD_PER_PX = 0.003, W = 1280, H = 720;

await mkdir(OUT, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-angle=metal', '--ignore-gpu-blocklist'] });
const state = () => { const w = window.wander, p = w.camera.position, c = w.walk.cellAt(p.x, p.z);
  return { x: +p.x.toFixed(3), y: +p.y.toFixed(3), z: +p.z.toFixed(3), floor: +w.walk.floor.toFixed(3), cellFloor: Number.isNaN(c.floor) ? null : +c.floor.toFixed(3), dist: c.dist, inside: c.inside, blockedUnder: w.walk.blockedAt(p.x, p.z), edge: +w.edge.toFixed(2) }; };
async function hold(page, code, ms) {
  const tr = []; await page.keyboard.down(code); const t0 = performance.now();
  while (performance.now() - t0 < ms) { await page.waitForTimeout(SAMPLE_MS); tr.push({ t: +((performance.now() - t0) / 1000).toFixed(2), ...await page.evaluate(state) }); }
  await page.keyboard.up(code); return tr;
}
async function turn(page, deg) {   // drag path, as a touch would; headless Chrome grants pointer lock on the first click, so release it first
  await page.evaluate(() => document.exitPointerLock?.());
  const px = (deg * Math.PI / 180) / LOOK_RAD_PER_PX;
  await page.mouse.move(W / 2, H / 2); await page.mouse.down();
  for (let i = 1; i <= 20; i++) { await page.mouse.move(W / 2 + px * i / 20, H / 2); await page.waitForTimeout(10); }
  await page.mouse.up();
}
const summarise = (tr, start) => {
  const far = Math.max(...tr.map(s => Math.hypot(s.x - start.x, s.z - start.z)));
  const onNaN = tr.filter(s => s.cellFloor === null).length;
  let deeper = 0; for (let i = 1; i < tr.length; i++) if (!tr[i].inside && tr[i].dist > tr[i - 1].dist) deeper++;
  const end = tr[tr.length - 1];
  return { unitsMoved: +far.toFixed(3), samples: tr.length, samplesOnUnknownFloor: onNaN, samplesOutside: tr.filter(s => !s.inside).length, endDist: end.dist, endInside: end.inside, endFloor: end.floor, endY: end.y, endEdge: end.edge, stepsDeeperIntoVoid: deeper };
};
let hull = 0;
const out = {};
try {
  const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  for (const clip of CLIPS) {
    const page = await ctx.newPage(); const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(`${BASE}/fourd.html?demo=${clip}&walk=1`, { waitUntil: 'load' });
    await page.waitForFunction(() => window.wander?.ready, null, { timeout: 240000 });
    await page.waitForTimeout(1500); await page.keyboard.press('Space');
    const r = { errors, grid: await page.evaluate(() => window.wander.walk.grid), upm: await page.evaluate(() => +window.wander.upm.toFixed(4)),
      clampOn: await page.evaluate(() => window.wander.clampOn), box: await page.evaluate(() => window.wander.clampBox.min.toArray().concat(window.wander.clampBox.max.toArray()).map(v => +v.toFixed(2))),
      start: await page.evaluate(state), yaws: {} };
    hull = r.grid?.hullCells ?? 0;
    for (const yaw of [0, 45, 90, 135, 180, 225, 270, 315]) {
      await page.keyboard.press('KeyR'); await page.waitForTimeout(100);
      if (yaw) await turn(page, yaw);
      const got = await page.evaluate(() => +window.wander.THREE.MathUtils.radToDeg(new window.wander.THREE.Euler().setFromQuaternion(window.wander.camera.quaternion, 'YXZ').y).toFixed(0));
      const tr = await hold(page, 'KeyW', SECS * 1000);
      r.yaws[yaw] = { yawReached: got, ...summarise(tr, r.start) };
      if (clip === 'stairs2' && yaw === 0) { r.climb = { trace: tr.filter((_, i) => i % 4 === 0).map(s => ({ t: s.t, z: s.z, floor: s.floor, y: s.y })), floors: [...new Set(tr.map(s => s.floor.toFixed(2)))].length, endFloor: tr[tr.length - 1].floor, largestJump: +Math.max(...tr.slice(1).map((s, i) => s.floor - tr[i].floor)).toFixed(3) };
        await page.screenshot({ path: path.join(OUT, 'stairs2-climb-shipped-end.png') });
        await page.keyboard.press('KeyR'); await page.waitForTimeout(200); r.afterR = await page.evaluate(state); }
      if (clip === 'elevator' && yaw === 45) await page.screenshot({ path: path.join(OUT, 'elevator-attack-yaw45-end.png') });
      if (clip === 'hpwide' && yaw === 180) await page.screenshot({ path: path.join(OUT, 'hpwide-attack-yaw180-end.png') });
      if (clip === 'atrium' && yaw === 0) await page.screenshot({ path: path.join(OUT, 'atrium-attack-yaw0-end.png') });
    }
    if (clip === 'elevator') {   // right strafe from 2 s in: where does it stop, and against what
      await page.keyboard.press('KeyR'); await page.waitForTimeout(100); await turn(page, 0);
      await hold(page, 'KeyW', 2500); const tr = await hold(page, 'KeyD', 8000);
      r.rightStrafe = { ...summarise(tr, tr[0]), endX: tr[tr.length - 1].x, endBlockedUnder: tr[tr.length - 1].blockedUnder, movingOverLastSecond: +(Math.max(...tr.slice(-7).map(s => s.x)) - Math.min(...tr.slice(-7).map(s => s.x))).toFixed(4) };
      await page.screenshot({ path: path.join(OUT, 'elevator-attack-right-strafe-end.png') });
    }
    // replay the yaw that spent most samples on unknown floor and keep its last frame: that is what the closing let through
    r.worst = Object.entries(r.yaws).sort((a, b) => b[1].samplesOnUnknownFloor - a[1].samplesOnUnknownFloor)[0];
    await page.keyboard.press('KeyR'); await page.waitForTimeout(100);
    if (+r.worst[0]) await turn(page, +r.worst[0]);
    await hold(page, 'KeyW', SECS * 1000);
    r.worstShot = `${clip}-attack-worst-yaw${r.worst[0]}-end.png`;
    await page.screenshot({ path: path.join(OUT, r.worstShot) });
    out[clip] = r; await page.close();
  }
} finally { await browser.close(); }
console.log(JSON.stringify(out, null, 1));
