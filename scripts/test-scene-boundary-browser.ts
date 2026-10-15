import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright-core';
import type * as THREE from 'three';
import type { ViewerDiagnostics } from './viewer-types.ts';

type BoundaryViewer = ViewerDiagnostics & {
  hardBox: THREE.Box3;
  world: { numSplats: number };
  place(position: number[], look?: number[]): void;
};
const output = process.env.BOUNDARY_TEST_OUTPUT || '.context/evidence/scene-boundary';
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const errors: string[] = [];
page.on('pageerror', (error) => errors.push(error.message));
let edge;
try {
  await page.goto('http://127.0.0.1:5399/fourd.html?demo=elevator&pause=1', {
    waitUntil: 'domcontentloaded',
  });
  await page.waitForFunction(() => window.wander?.ready, null, { timeout: 120000 });
  const home = await page.evaluate(() => {
    const w = window.wander as BoundaryViewer;
    w.play(false);
    w.setTime(9.8);
    const position = w.camera.position.toArray();
    w.place([w.hardBox.max.x + 1, position[1], position[2]], [0, 0, -2]);
    return { position, time: w.t };
  });
  await page.waitForFunction(() => window.wander.edge > 0.9);
  await page.waitForFunction(
    () =>
      (window.wander as BoundaryViewer).world.numSplats > 0 &&
      Math.abs(window.wander.video.currentTime - 9.8) < 0.02,
    null,
    { timeout: 30000 },
  );
  await page.evaluate(async () => {
    for (let i = 0; i < 20; i++) await new Promise(requestAnimationFrame);
  });
  edge = await page.evaluate(() => {
    const w = window.wander as BoundaryViewer;
    return {
      edge: w.edge,
      position: w.camera.position.toArray(),
      hardMax: w.hardBox.max.toArray(),
      worldSplats: w.world.numSplats,
      filter: w.spark.renderer.domElement.style.filter,
      overlay: document.querySelector<HTMLElement>('#edgefade')!.style.opacity,
    };
  });
  await page.screenshot({ path: `${output}/edge.png` });
  assert.ok(edge.position[0] <= edge.hardMax[0], 'Movement boundary must remain enforced');
  assert.equal(edge.filter, '', 'The boundary darkened the whole scene');
  assert.equal(Number(edge.overlay), 0, 'The boundary overlaid the scene with darkness');
  await page.getByRole('button', { name: 'Back to scene', exact: true }).click();
  const returned = await page.evaluate(() => ({
    position: window.wander.camera.position.toArray(),
    time: window.wander.t,
  }));
  assert.deepEqual(
    returned,
    home,
    'Return should restore only the camera and preserve playback time',
  );
  await page.getByText('Scene boundary', { exact: true }).waitFor({ state: 'hidden' });
  await page.screenshot({ path: `${output}/returned.png` });
  await page.evaluate(() => {
    const w = window.wander as BoundaryViewer;
    w.place([w.hardBox.max.x, w.camera.position.y, w.camera.position.z]);
  });
  await page.getByRole('button', { name: 'Back to scene', exact: true }).focus();
  await page.keyboard.press('Space');
  assert.deepEqual(
    await page.evaluate(() => ({
      position: window.wander.camera.position.toArray(),
      time: window.wander.t,
      playing: window.wander.playing,
    })),
    { ...home, playing: false },
    'Keyboard return must activate the button without toggling playback',
  );
  assert.deepEqual(errors, []);
  console.log(
    'Scene stays visible at its movement boundary; return restores the camera without seeking.',
  );
} finally {
  await writeFile(`${output}/report.json`, JSON.stringify({ edge, errors }, null, 2));
  await browser.close();
}
