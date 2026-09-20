// Real desktop input and private scene geometry; screenshots require visual review.
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright-core';
import type { ViewerDiagnostics, WalkDiagnostics } from './viewer-types.ts';

type WalkViewer = ViewerDiagnostics & {
  world: { numSplats: number };
  walk: WalkDiagnostics & { stand(x: number, z: number): boolean };
};
type Sample = {
  position: number[];
  floor: number;
  stature: number;
  blocked: number;
};
const out = path.resolve(
  process.env.WALK_TEST_OUT || '.context/evidence/walk-clearance/kitchen-regression',
);
const base = 'http://127.0.0.1:5399';
// This is the existing legacy kitchen preset, including its measured placement.
const query = new URLSearchParams({
  world: '/marble-kitchen-cooking-image.spz',
  people: '/reviews/gym-kitchen/kitchen/people.json',
  video: '/reviews/gym-kitchen/kitchen/source.mp4',
  scale: '0.36765256685552467',
  floor: '-1.07197589556301',
  placement: '/reviews/gym-kitchen/kitchen/placement.json',
  place: '1',
  feetmode: 'sfm',
  feetlock: '0',
  stance: '0',
  camdrift: '0',
  walk: '1',
  cam: '0,0,0,0,-0.3,-2.5',
  fov: '63',
  worldfallback: 'none',
  audio: '0',
});

await mkdir(out, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  if (process.env.WALK_TEST_DIST) {
    const dist = path.resolve(process.env.WALK_TEST_DIST);
    await page.route(`${base}/fourd.html?*`, (route) =>
      route.fulfill({
        path: path.join(dist, 'fourd.html'),
        contentType: 'text/html',
        headers: {
          'Cross-Origin-Opener-Policy': 'same-origin',
          'Cross-Origin-Embedder-Policy': 'credentialless',
        },
      }),
    );
    await page.route(`${base}/assets/*`, (route) =>
      route.fulfill({
        path: path.join(dist, 'assets', path.basename(new URL(route.request().url()).pathname)),
      }),
    );
  }
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(`${base}/fourd.html?${query}`, { waitUntil: 'load', timeout: 120000 });
  // Collision is available before all 559 animation samples have streamed.
  await page.waitForFunction(() => window.wander?.walk, null, { timeout: 120000 });
  await page.evaluate(() => window.wander.play(false));
  // A collision-only pass with a black, unloaded world is not visual evidence.
  await page.waitForFunction(
    () =>
      (window.wander as WalkViewer).world.numSplats > 1000 && window.wander.video.readyState >= 2,
    null,
    { timeout: 120000 },
  );

  async function position(yaw: number) {
    assert.equal(
      await page.evaluate((yaw) => {
        const w = window.wander as WalkViewer;
        if (!w.walk.stand(1.13, -0.04)) return false;
        w.camera.rotation.set(-0.25, yaw, 0, 'YXZ');
        return true;
      }, yaw),
      true,
      'The reported starting spot must remain a supported, unblocked standing position',
    );
    await page.evaluate(
      () =>
        new Promise<void>((resolve) =>
          requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
        ),
    );
  }

  async function walkForward() {
    // Keyboard events exercise desktop walkMove, floor settling, and camera clamping together.
    const start = await page.evaluate(() => window.wander.camera.position.toArray());
    await page.keyboard.down('w');
    try {
      return await page.evaluate(
        (start) =>
          new Promise<{ start: number[]; samples: Sample[]; distance: number }>((resolve) => {
            const w = window.wander as WalkViewer;
            const samples: Sample[] = [];
            let stationary = 0;
            function sample() {
              const p = w.camera.position;
              const previous = samples.at(-1);
              stationary =
                previous &&
                Math.hypot(p.x - previous.position[0], p.z - previous.position[2]) < 1e-6
                  ? stationary + 1
                  : 0;
              samples.push({
                position: p.toArray(),
                floor: w.walk.floor,
                stature: w.walk.stature,
                blocked: w.walk.blockedAt(p.x, p.z),
              });
              const distance = Math.hypot(p.x - start[0], p.z - start[2]) / w.walk.stature;
              if (distance >= 0.4 || stationary >= 24 || samples.length >= 180) {
                resolve({ start, samples, distance });
              } else requestAnimationFrame(sample);
            }
            requestAnimationFrame(sample);
          }),
        start,
      );
    } finally {
      await page.keyboard.up('w');
    }
  }

  await position(-Math.PI / 4);
  await page.screenshot({ path: path.join(out, 'aisle-start.png') });
  const source = await page.evaluate(() => {
    const video = window.wander.video;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d')!.drawImage(video, 0, 0);
    return { time: video.currentTime, png: canvas.toDataURL('image/png').split(',')[1] };
  });
  await Bun.write(path.join(out, 'source.png'), Buffer.from(source.png, 'base64'));
  const aisle = await walkForward();
  await page.screenshot({ path: path.join(out, 'aisle-end.png') });

  await position(-Math.PI / 2);
  await page.screenshot({ path: path.join(out, 'counter-start.png') });
  const counter = await walkForward();
  await page.screenshot({ path: path.join(out, 'counter-stop.png') });
  const beyondCounter = await page.evaluate(() => {
    const w = window.wander as WalkViewer;
    const p = w.camera.position;
    return w.walk.blockedAt(p.x + 0.04 * w.walk.stature, p.z);
  });
  const report = {
    sourceTime: source.time,
    dist: process.env.WALK_TEST_DIST || 'live server',
    aisle,
    counter,
    beyondCounter,
    errors,
  };
  await Bun.write(path.join(out, 'report.json'), JSON.stringify(report, null, 2));
  console.log('Kitchen clearance, distances in body-heights', {
    aisle: aisle.distance,
    aisleEnd: aisle.samples.at(-1)?.position,
    counter: counter.distance,
    counterEnd: counter.samples.at(-1)?.position,
    beyondCounter,
    evidence: out,
  });

  assert.ok(
    aisle.distance >= 0.35,
    'The visible aisle must permit at least 0.35 body-heights of travel',
  );
  assert.ok(aisle.distance < 0.6, 'The aisle check must remain inside the reviewed local route');
  assert.ok(counter.distance > 0.005, 'The countertop approach must exercise actual movement');
  assert.ok(counter.distance < 0.1, 'The solid countertop must stop the approaching walker');
  assert.ok(beyondCounter > 0, 'The stop must retain occupied countertop cells immediately ahead');
  for (const sample of [...aisle.samples, ...counter.samples]) {
    assert.equal(sample.blocked, 0, 'Both routes must stay outside occupied footprint cells');
  }
  assert.deepEqual(errors, []);
} finally {
  await browser.close();
}
