// Real stereo rendering with synthetic input; this is not headset comfort/performance evidence.
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { Quaternion, Vector3 } from 'three';

type BodySample = {
  visible: boolean;
  moving: boolean;
  steps: number;
  feet: { left: number[]; right: number[] };
  torso: number[];
  torsoRotation: number[];
};
type Sample = {
  time: number;
  yaw: number;
  head: number[];
  rotation: number[];
  localEye: number[];
  separation: number;
  vignette: boolean;
  body: BodySample;
};
const out = '.context/evidence/quest-debug';
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const records: Record<string, Sample[]> = {};
const close = (a: number, b: number, label: string, tolerance = 0.0002) =>
  assert.ok(Math.abs(a - b) < tolerance, `${label}: ${a} != ${b}`);
const angle = (a: number, b: number) => Math.atan2(Math.sin(a - b), Math.cos(a - b));
try {
  await mkdir(out, { recursive: true });
  const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
  // Exercise this checkout's production build while the existing server supplies private media.
  // This avoids testing unrelated work from another checkout serving the shared dev port.
  if (process.env.XR_TEST_DIST) {
    const dist = path.resolve(process.env.XR_TEST_DIST);
    await page.route('http://127.0.0.1:5399/fourd.html?*', (route) =>
      route.fulfill({
        path: path.join(dist, 'fourd.html'),
        contentType: 'text/html',
        headers: {
          'Cross-Origin-Opener-Policy': 'same-origin',
          'Cross-Origin-Embedder-Policy': 'credentialless',
        },
      }),
    );
    await page.route('http://127.0.0.1:5399/assets/*', (route) =>
      route.fulfill({
        path: path.join(dist, 'assets', path.basename(new URL(route.request().url()).pathname)),
      }),
    );
  }
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.addInitScript({ path: new URL('./fake-webxr.js', import.meta.url).pathname });
  async function open(options = '') {
    await page.goto(
      `http://127.0.0.1:5399/fourd.html?demo=stairs2&xr=1&fakexr=1&xrmove=smooth&clamp=0&fakew=384&fakeh=384&xradapt=0${options}`,
      { waitUntil: 'load', timeout: 120000 },
    );
    await page.waitForFunction(() => window.wander?.ready, null, { timeout: 180000 });
    await page.evaluate(() => {
      Object.assign(window.__fakeXR.head, { x: 0.35, z: -0.25, yaw: 0.2, pitch: -0.3 });
    });
    await page.click('#xrBtn');
    await page.waitForFunction(() => window.__fakeXR.frames.length > 4);
  }
  // Observe after actual rendering: the eye matrices include this frame's rig motion.
  // Input changes and the first sampled render share a JS task boundary, avoiding wallclock sleeps.
  async function collect(right: number, left = 0, count = 12): Promise<Sample[]> {
    return page.evaluate(
      ({ right, left, count }) =>
        new Promise<Sample[]>((resolve) => {
          const w = window.wander;
          const renderer = w.spark.renderer;
          const original = renderer.render;
          const samples: Sample[] = [];
          window.__fakeXR.axes.right = [0, 0, right, 0];
          window.__fakeXR.axes.left = [0, 0, 0, left];
          renderer.render = function (scene, camera) {
            const time = performance.now();
            original.call(this, scene, camera);
            if (camera !== w.camera || !renderer.xr.isPresenting) return;
            const rig = w.camera.parent!;
            const eyes = renderer.xr.getCamera().cameras;
            const positions = eyes.map((eye) =>
              new w.THREE.Vector3().setFromMatrixPosition(eye.matrixWorld),
            );
            const forward = new w.THREE.Vector3(0, 0, -1).applyQuaternion(rig.quaternion);
            const body = (window.__xr as { body: Omit<BodySample, 'torso' | 'torsoRotation'> })
              .body;
            const torso = rig.getObjectByName('avatar-body-torso')!;
            const vignette = w.camera.children.find((child) => {
              const material = (child as import('three').Mesh).material;
              return (
                material &&
                !Array.isArray(material) &&
                (material as import('three').ShaderMaterial).uniforms?.uAmount
              );
            });
            samples.push({
              time,
              yaw: Math.atan2(-forward.x, -forward.z),
              head: positions[0].clone().add(positions[1]).multiplyScalar(0.5).toArray(),
              rotation: rig.quaternion.toArray(),
              localEye: eyes[0].quaternion.toArray(),
              separation: positions[0].distanceTo(positions[1]),
              vignette: vignette?.visible ?? false,
              // The avatar uses reference-space coordinates under the rig. Copy the mutable
              // diagnostics now so later frames cannot overwrite evidence of foot movement.
              body: {
                visible: body.visible,
                moving: body.moving,
                steps: body.steps,
                feet: { left: [...body.feet.left], right: [...body.feet.right] },
                torso: torso.position.toArray(),
                torsoRotation: torso.quaternion.toArray(),
              },
            });
            if (samples.length === count) {
              // Release at the last observed render, before Node checks the samples. Otherwise
              // unobserved held-input frames can advance yaw before the release assertion.
              window.__fakeXR.axes.right = [0, 0, 0, 0];
              window.__fakeXR.axes.left = [0, 0, 0, 0];
              renderer.render = original;
              resolve(samples);
            }
          };
        }),
      { right, left, count },
    );
  }
  function rate(samples: Sample[], expected: number) {
    for (let i = 1; i < samples.length; i++) {
      const dt = Math.min(0.1, (samples[i].time - samples[i - 1].time) / 1000);
      close(
        angle(samples[i].yaw, samples[i - 1].yaw),
        expected * dt,
        'per-render angular speed',
        0.0015,
      );
    }
  }
  function fixedHead(samples: Sample[], baseline: Sample) {
    for (const sample of samples) {
      sample.head.forEach((value, i) => close(value, baseline.head[i], 'tracked head pivot'));
      close(sample.separation, baseline.separation, 'stereo separation');
      sample.localEye.forEach((value, i) => close(value, baseline.localEye[i], 'tracked eye pose'));
      assert.equal(sample.vignette, false, 'smooth turn must not blink');
    }
  }
  function fixedBody(samples: Sample[], baseline: Sample) {
    assert.equal(baseline.body.visible, true, 'stationary avatar is visible');
    assert.equal(baseline.body.moving, false, 'stationary avatar starts with planted feet');
    for (const sample of samples) {
      assert.equal(sample.body.visible, true, 'turning keeps the avatar visible');
      assert.equal(sample.body.moving, false, 'turning must not fabricate walking motion');
      assert.equal(sample.body.steps, baseline.body.steps, 'turning must not fabricate steps');
      for (const side of ['left', 'right'] as const)
        sample.body.feet[side].forEach((value, i) =>
          close(value, baseline.body.feet[side][i], `${side} foot stays planted relative to rig`),
        );
      sample.body.torso.forEach((value, i) =>
        close(value, baseline.body.torso[i], 'torso stays stationary relative to rig'),
      );
      sample.body.torsoRotation.forEach((value, i) =>
        close(value, baseline.body.torsoRotation[i], 'torso keeps tracked local orientation'),
      );
    }
  }
  await open();
  assert.deepEqual(
    await page.evaluate(() => {
      const xr = window.__xr as { turnMode: string; turnSpeed: number };
      return [xr.turnMode, xr.turnSpeed];
    }),
    ['smooth', 90],
  );
  const baseline = (await collect(0))[0];
  records.baseline = [baseline];
  await page.screenshot({ path: `${out}/turning-before.png` });
  for (const [name, stick] of [
    ['full-right', 1],
    ['half-right', 0.5],
    ['left', -1],
    ['deadzone', 0.1],
  ] as const) {
    const samples = await collect(stick);
    records[name] = samples;
    const analog =
      Math.abs(stick) <= 0.15 ? 0 : (Math.sign(stick) * (Math.abs(stick) - 0.15)) / 0.85;
    rate(samples, (-Math.PI / 2) * analog);
    fixedHead(samples, baseline);
    // The nonzero tracked X/Z offset makes turnAroundHead translate the rig origin.
    // That pivot compensation must not become artificial walking in the avatar.
    fixedBody(samples, baseline);
    const stopped = await collect(0);
    records[`${name}-released`] = stopped;
    stopped.forEach((sample) =>
      close(angle(sample.yaw, samples.at(-1)!.yaw), 0, 'neutral stops immediately'),
    );
    fixedHead(stopped, baseline);
    fixedBody(stopped, baseline);
  }
  await page.screenshot({ path: `${out}/turning-after.png` });
  // On the first movement frame velocity starts from zero, so its direction must match
  // the newly turned head, independent of the translation acceleration constant.
  const beforeWalk = (await collect(0)).at(-1)!;
  const walking = await collect(1, -1, 1);
  const delta = new Vector3()
    .fromArray(walking[0].head)
    .sub(new Vector3().fromArray(beforeWalk.head));
  const forward = new Vector3(0, 0, -1).applyQuaternion(
    new Quaternion()
      .fromArray(walking[0].rotation)
      .multiply(new Quaternion().fromArray(walking[0].localEye)),
  );
  forward.y = 0;
  delta.y = 0;
  assert.ok(delta.length() > 0, 'forward input moves on the first frame');
  close(
    delta.normalize().dot(forward.normalize()),
    1,
    'walking follows this frame turning',
    0.00001,
  );
  await collect(0);
  await page.evaluate(async () => {
    await window.wander.spark.renderer.xr.getSession()!.end();
  });
  await page.click('#xrBtn');
  await collect(0);
  const reentry = await collect(-1);
  rate(reentry, Math.PI / 2);
  records.reentry = reentry;
  // Disabled turning must suppress both snap and smooth input; custom speed remains observable.
  await open('&xrturn=0&xrturnspeed=120');
  const disabled = (await collect(0)).at(-1)!;
  const held = await collect(1);
  held.forEach((sample) => close(angle(sample.yaw, disabled.yaw), 0, 'xrturn=0 disables turning'));
  assert.equal(await page.evaluate(() => (window.__xr as { turnSpeed: number }).turnSpeed), 120);
  // The explicit legacy mode still turns once per deflection and rearms on release.
  await open('&xrturnmode=snap');
  assert.equal(await page.evaluate(() => (window.__xr as { turnMode: string }).turnMode), 'snap');
  const beforeSnap = (await collect(0)).at(-1)!;
  const firstSnap = await collect(1);
  firstSnap.forEach((sample) =>
    close(angle(sample.yaw, beforeSnap.yaw), -Math.PI / 6, 'snap turns once while held'),
  );
  await collect(0);
  const secondSnap = await collect(-1);
  secondSnap.forEach((sample) =>
    close(angle(sample.yaw, beforeSnap.yaw), 0, 'snap rearms when centered'),
  );
  for (const sample of [...firstSnap, ...secondSnap])
    sample.head.forEach((value, i) => close(value, beforeSnap.head[i], 'snap head pivot'));
  records.snap = [...firstSnap, ...secondSnap];
  assert.deepEqual(errors, []);
  await writeFile(`${out}/turning-measurements.json`, JSON.stringify(records, null, 2));
  console.log(
    'XR turning passed: analog rate, both directions, immediate stop, head pivot, stationary avatar and planted feet, tracked pose, stereo, no blink, same-frame walking, re-entry, disabled turns, legacy snap.',
  );
} finally {
  await browser.close();
}
