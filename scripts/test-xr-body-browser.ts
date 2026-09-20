// Real viewer with scripted XR poses. Visual captures here are simulated, not headset evidence.
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright-core';
import type {} from './viewer-types.ts';

const browser = await chromium.launch({ channel: 'chrome', headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error' && /shader|WebGLProgram/i.test(message.text()))
      errors.push(message.text());
  });
  await page.addInitScript({ path: new URL('./fake-webxr.js', import.meta.url).pathname });
  await page.goto(
    'http://127.0.0.1:5399/fourd.html?demo=stairs2&walk=1&personsize=0.9&xr=1&fakexr=1&xrmove=smooth&fakew=768&fakeh=768&xradapt=0',
    { waitUntil: 'load', timeout: 120000 },
  );
  await page.waitForFunction(() => window.wander?.ready, null, { timeout: 180000 });
  await page.click('#xrBtn');
  async function advance(count = 10) {
    const start = await page.evaluate(() => window.__fakeXR.frames.length);
    await page.waitForFunction((end) => window.__fakeXR.frames.length >= end, start + count);
  }
  async function state() {
    return page.evaluate(() => {
      const w = window.wander;
      const body = w.camera.parent?.getObjectByName('avatar-body');
      const meshes: { name: string; position: number[]; scale: number[] }[] = [];
      body?.traverse((part) => {
        if ((part as { isMesh?: boolean }).isMesh)
          meshes.push({
            name: part.name,
            position: part.position.toArray(),
            scale: part.scale.toArray(),
          });
      });
      return {
        bodyVisible: body?.visible,
        meshes,
        report: (window.__xr as { body?: unknown }).body,
        head: { ...window.__fakeXR.head },
        hands: (window.__xr as { hands: { visibleCount: number } }).hands,
      };
    });
  }
  await advance();
  await page.evaluate(() => {
    window.wander.play(false);
    window.__fakeXR.head.pitch = -1.4;
  });
  await advance();
  const standing = await state();
  assert.equal(standing.bodyVisible, true);
  assert.ok(standing.meshes.length >= 10, 'torso and paired arm/leg segments must render');
  assert.equal(standing.hands.visibleCount, 2);
  for (const part of standing.meshes) {
    assert.ok(part.position.every(Number.isFinite), 'body positions must be finite');
    assert.ok(
      part.scale.every((value) => Number.isFinite(value) && value > 0),
      'valid body scale',
    );
  }
  await mkdir('.context/evidence/quest-debug', { recursive: true });
  await page.screenshot({ path: '.context/evidence/quest-debug/body-look-down-simulated.png' });
  await advance(15);
  const still = await state();
  assert.deepEqual(still.meshes, standing.meshes, 'stationary body must not walk or drift');
  await page.evaluate(() => {
    window.__fakeXR.axes.left = [0, 0, 0, -0.7];
  });
  await advance(22);
  const walking = await state();
  assert.notDeepEqual(walking.meshes, standing.meshes, 'joystick movement must animate the legs');
  await page.screenshot({ path: '.context/evidence/quest-debug/body-walking-simulated.png' });
  await page.evaluate(() => {
    window.__fakeXR.axes.left = [0, 0, 0, 0];
  });
  await advance(40);
  const beforeCrouch = await state();
  await page.evaluate(() => {
    window.__fakeXR.head.y = 1.1;
  });
  await advance(15);
  const crouching = await state();
  assert.notDeepEqual(
    crouching.meshes,
    beforeCrouch.meshes,
    'crouching must lower and bend the body',
  );
  await page.screenshot({ path: '.context/evidence/quest-debug/body-crouching-simulated.png' });
  await Bun.write(
    '.context/evidence/quest-debug/avatar-body-browser.json',
    JSON.stringify({ standing, walking, crouching }, null, 2),
  );
  await page.evaluate(async () => {
    await window.wander.spark.renderer.xr.getSession()!.end();
  });
  assert.equal(
    await page.evaluate(() => {
      // The camera leaves its rig on exit, so inspect the renderer scene via the published body report.
      return (window.__xr as { body: { visible: boolean } }).body.visible;
    }),
    false,
  );
  assert.deepEqual(errors, []);
  console.log(
    'XR body browser checks passed: visible parts, stationary pose, walking, crouching, session hide.',
  );
} finally {
  await browser.close();
}
