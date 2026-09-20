// Real scene collision data through simulated XR; this is not physical-headset evidence.
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright-core';
import type {} from './viewer-types.ts';

const browser = await chromium.launch({ channel: 'chrome', headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.addInitScript({ path: new URL('./fake-webxr.js', import.meta.url).pathname });
  await page.goto(
    'http://127.0.0.1:5399/fourd.html?demo=stairs2&walk=1&personsize=0.9&xr=1&fakexr=1&xrmove=smooth&fakew=512&fakeh=512&xradapt=0',
    { waitUntil: 'load', timeout: 120000 },
  );
  await page.waitForFunction(() => window.wander?.ready, null, { timeout: 180000 });
  await page.click('#xrBtn');
  async function advance(count = 8) {
    const start = await page.evaluate(() => window.__fakeXR.frames.length);
    await page.waitForFunction((end) => window.__fakeXR.frames.length >= end, start + count);
  }
  async function snapshot() {
    return page.evaluate(() => {
      const w = window.wander;
      const xr = w.spark.renderer.xr;
      const eyes = xr.getCamera().cameras;
      const head = new w.THREE.Vector3()
        .setFromMatrixPosition(eyes[0].matrixWorld)
        .add(new w.THREE.Vector3().setFromMatrixPosition(eyes[1].matrixWorld))
        .multiplyScalar(0.5);
      const floor = (window.__xr as { groundFloor?: number }).groundFloor ?? w.walk!.floor;
      return {
        head: head.toArray(),
        floor,
        upm: w.upm,
        rigY: w.camera.parent!.position.y,
        rawHeight: window.__fakeXR.head.y,
        blocked: w.walk!.blockedAt(head.x, head.z, floor),
      };
    });
  }
  await advance();
  await page.evaluate(() => window.wander.play(false));
  const start = await snapshot();
  await page.evaluate(() => {
    window.__fakeXR.axes.left = [0, 0, 0, -1];
  });
  const trace = [start];
  for (let i = 0; i < 12; i++) {
    await advance(15);
    trace.push(await snapshot());
  }
  await page.evaluate(() => {
    window.__fakeXR.axes.left = [0, 0, 0, 0];
  });
  await advance(20);
  const end = await snapshot();
  trace.push(end);
  await mkdir('.context/evidence/quest-debug', { recursive: true });
  await Bun.write(
    '.context/evidence/quest-debug/xr-stairs-ground.json',
    JSON.stringify(trace, null, 2),
  );
  await page.screenshot({ path: '.context/evidence/quest-debug/xr-stairs-ground.png' });
  console.log('XR stairs ground', { start, end, maxFloor: Math.max(...trace.map((s) => s.floor)) });
  assert.ok(end.head[2] < start.head[2] - 0.1 * start.upm, 'joystick must move forward');
  assert.ok(end.floor > start.floor + 0.1 * start.upm, 'known stair support must raise the rig');
  for (const sample of trace) {
    assert.ok(
      Math.abs(sample.head[1] - sample.rigY - sample.rawHeight * sample.upm) < 0.0002,
      'stairs move the rig without changing tracked eye height',
    );
    assert.ok(sample.blocked <= start.blocked, 'joystick must not enter more blocked columns');
  }
  assert.deepEqual(errors, []);
  console.log('XR stairs checks passed: forward walking, support rise, tracked height, obstacles.');
} finally {
  await browser.close();
}
