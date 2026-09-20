// Real viewer + synthetic WebXR input; this does not certify headset comfort or performance.
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright-core';
import { Quaternion, Vector3 } from 'three';

type FakeDriver = {
  head: { x: number; y: number; z: number; yaw: number };
  axes: { left: number[]; right: number[] };
  buttons: { left: number[]; right: number[] };
  frames: number[];
};

type HandReport = {
  visibleCount: number;
  left: { mode: string; trigger: number; squeeze: number };
  right: { mode: string; trigger: number; squeeze: number };
};

const browser = await chromium.launch({ channel: 'chrome', headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error' && /shader|WebGLProgram/i.test(message.text()))
      errors.push(message.text());
  });
  await page.addInitScript({ path: new URL('./fake-webxr.js', import.meta.url).pathname });
  // This suite covers snap pivots; test-xr-turning-browser covers smooth mode's continuous default.
  await page.goto(
    'http://127.0.0.1:5399/fourd.html?demo=stairs2&xr=1&fakexr=1&xrmove=smooth&xrturnmode=snap&xrwalkgain=1.3&clamp=0&fakew=512&fakeh=512&xradapt=0',
    { waitUntil: 'load', timeout: 120000 },
  );
  await page.waitForFunction(() => window.wander?.ready, null, { timeout: 180000 });
  await page.evaluate(() => window.wander.play(false));
  await page.click('#xrBtn');

  async function advance(count = 8) {
    const before = await page.evaluate(() => window.__fakeXR.frames.length);
    await page
      .waitForFunction((end) => window.__fakeXR.frames.length >= end, before + count)
      .catch(async (error) => {
        console.error('XR frame wait failed', {
          before,
          errors,
          state: await page.evaluate(() => ({
            frames: window.__fakeXR.frames.length,
            presenting: window.wander.spark.renderer.xr.isPresenting,
            hidden: document.hidden,
            xr: window.__xr,
            status: document.getElementById('st')?.textContent,
          })),
        });
        throw error;
      });
  }
  async function snapshot() {
    return page.evaluate(() => {
      const w = window.wander;
      const xr = w.spark.renderer.xr;
      const rig = w.camera.parent!;
      // The culling camera is shifted by Three's stereo frustum union. The rendered eye
      // midpoint measures the actual head center without that artificial offset.
      const eyes = xr
        .getCamera()
        .cameras.map((eye) => new w.THREE.Vector3().setFromMatrixPosition(eye.matrixWorld));
      return {
        head: eyes[0].clone().add(eyes[1]).multiplyScalar(0.5).toArray(),
        eyeSeparation: eyes[0].distanceTo(eyes[1]),
        rig: rig.position.toArray(),
        rotation: rig.quaternion.toArray(),
        scale: rig.scale.toArray(),
        upm: w.upm,
        presenting: xr.isPresenting,
      };
    });
  }
  function close(actual: number, expected: number, label: string, tolerance = 0.0002) {
    assert.ok(Math.abs(actual - expected) < tolerance, `${label}: ${actual} != ${expected}`);
  }

  await advance();
  console.log('XR locomotion: session started');
  const before = await snapshot();
  assert.equal(before.presenting, true);
  assert.equal(await page.evaluate(() => (window.__xr as { walkGain: number }).walkGain), 1.3);

  // Grip spaces must connect through Three's real input-source lifecycle before gloves appear.
  const hands = await page.evaluate(() => (window.__xr as { hands: HandReport }).hands);
  assert.equal(hands.visibleCount, 2);
  assert.equal(hands.left.mode, 'controller');
  assert.equal(hands.right.mode, 'controller');
  await page.evaluate(() => {
    (window.__fakeXR as FakeDriver).buttons.left = [0.75, 1];
    (window.__fakeXR as FakeDriver).buttons.right = [0, 0.4];
    const sources = window.wander.spark.renderer.xr.getSession()!.inputSources;
    for (const source of sources) {
      const grip = source.gripSpace as unknown as { _matrix: number[] };
      grip._matrix[13] = 1.4;
      grip._matrix[14] = -0.45;
    }
  });
  await advance();
  const curled = await page.evaluate(() => (window.__xr as { hands: HandReport }).hands);
  close(curled.left.trigger, 0.75, 'left trigger curl');
  close(curled.left.squeeze, 1, 'left grip curl');
  close(curled.right.squeeze, 0.4, 'right grip curl');
  await mkdir('.context/evidence/quest-debug', { recursive: true });
  await page.screenshot({ path: '.context/evidence/quest-debug/hands-simulated.png' });
  await page.evaluate(() => {
    const head = (window.__fakeXR as FakeDriver).head;
    head.x += 0.25;
    head.y += 0.2;
    head.z -= 0.15;
  });
  await advance();
  const moved = await snapshot();
  console.log('XR locomotion: physical step sampled');
  const expected = new Vector3(0.25 * 1.3, 0.2, -0.15 * 1.3)
    .multiplyScalar(before.upm)
    .applyQuaternion(new Quaternion().fromArray(before.rotation));
  expected
    .toArray()
    .forEach((value, i) => close(moved.head[i] - before.head[i], value, 'head gain'));
  assert.deepEqual(moved.scale, before.scale, 'gain must not change rig scale or eye separation');
  close(moved.eyeSeparation, before.eyeSeparation, 'eye separation');

  await advance(20);
  const stationary = await snapshot();
  stationary.head.forEach((value, i) => close(value, moved.head[i], 'stationary head'));
  await page.evaluate(() => ((window.__fakeXR as FakeDriver).head.yaw = 0.4));
  await advance();
  const looked = await snapshot();
  looked.head.forEach((value, i) => close(value, stationary.head[i], 'head turn adds no travel'));

  // A recenter rebases the extra gain instead of interpreting the new origin as a large step.
  await page.evaluate(() => {
    const driver = window.__fakeXR as FakeDriver;
    driver.head.x += 2;
    driver.head.z -= 1;
    const ref = window.wander.spark.renderer.xr.getReferenceSpace()!;
    ref.dispatchEvent(new Event('reset'));
  });
  await advance();
  const reset = await snapshot();
  console.log('XR locomotion: reference reset sampled');
  reset.rig.forEach((value, i) => close(value, stationary.rig[i], 'reference reset gain'));

  // Snap turning rotates around the head, without a second amplified translation.
  await page.evaluate(() => ((window.__fakeXR as FakeDriver).axes.right = [0, 0, 1, 0]));
  await advance();
  const turned = await snapshot();
  console.log('XR locomotion: snap turn sampled');
  turned.head.forEach((value, i) => close(value, reset.head[i], 'snap-turn head pivot'));
  assert.notDeepEqual(turned.rotation, reset.rotation);
  await page.evaluate(() => ((window.__fakeXR as FakeDriver).axes.right = [0, 0, 0, 0]));
  await advance();

  await page.evaluate(() => ((window.__fakeXR as FakeDriver).axes.left = [0, 0, 0, -0.5]));
  await advance(24);
  const walked = await snapshot();
  console.log('XR locomotion: joystick walk sampled');
  assert.ok(Math.hypot(walked.head[0] - turned.head[0], walked.head[2] - turned.head[2]) > 0.01);
  close(walked.head[1], turned.head[1], 'joystick stays level');
  await page.evaluate(() => {
    (window.__fakeXR as FakeDriver).axes.left = [0, 0, 0, 0];
    (window.__fakeXR as FakeDriver).axes.right = [0, 0, 0, -1];
  });
  await advance(24);
  const noFlight = await snapshot();
  close(noFlight.head[1], walked.head[1], 'right stick does not fly');

  await page.evaluate(async () => {
    await window.wander.spark.renderer.xr.getSession()!.end();
    (window.__fakeXR as FakeDriver).axes.right = [0, 0, 0, 0];
  });
  assert.equal(
    await page.evaluate(() => (window.__xr as { hands: HandReport }).hands.visibleCount),
    0,
    'gloves hide outside VR',
  );
  await page.click('#xrBtn');
  await advance();
  const reentered = await snapshot();
  console.log('XR locomotion: session re-entry sampled');
  before.head.forEach((value, i) =>
    close(reentered.head[i], value + (i === 1 ? 0.2 * before.upm : 0), 'session re-entry anchor'),
  );
  assert.equal(
    await page.evaluate(() => (window.__xr as { hands: HandReport }).hands.visibleCount),
    2,
    'gloves return with the new session',
  );
  assert.deepEqual(errors, []);
  console.log(
    'XR browser checks passed: hands, gain, height, drift, reset, turning, walking, re-entry.',
  );
} finally {
  await browser.close();
}
