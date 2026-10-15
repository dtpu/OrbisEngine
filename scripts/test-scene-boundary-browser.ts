import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright-core';
import type * as THREE from 'three';
import type { ViewerDiagnostics } from './viewer-types.ts';

type ExplorationViewer = ViewerDiagnostics & {
  hardBox: THREE.Box3;
  world: { numSplats: number };
  place(position: number[], look?: number[]): void;
};
const url = process.env.EXPLORATION_TEST_URL || 'http://127.0.0.1:5399/fourd.html?demo=elevator';
const output = process.env.EXPLORATION_TEST_OUTPUT || '.context/evidence/free-exploration';
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const errors: string[] = [];
const checks: unknown[] = [];
page.on('pageerror', (error) => errors.push(error.message));
async function settle() {
  await page.evaluate(async () => {
    for (let i = 0; i < 10; i++) await new Promise(requestAnimationFrame);
  });
}
try {
  // Old saved scenes may explicitly request limits. Only a current URL choice opts back in.
  const manifestUrl = new URL('/exploration-legacy-scene.json', url).href;
  await page.route(manifestUrl, (route) =>
    route.fulfill({
      json: { schema: 'wander.viewer-scene/1', params: { freedom: 'tight', clamp: '1' } },
    }),
  );
  for (const legacyManifest of [false, true]) {
    const target = new URL(url);
    target.searchParams.set('pause', '1');
    target.searchParams.set('path', '0,0,0;0,0,-2');
    target.searchParams.set('pathr', '0.2');
    if (legacyManifest) {
      target.searchParams.delete('scene');
      target.searchParams.set('demo', 'elevator');
      target.searchParams.set('scene', new URL(manifestUrl).pathname);
    }
    await page.goto(target.href, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => window.wander?.ready, null, { timeout: 120000 });
    const home = await page.evaluate(() => {
      const w = window.wander as ExplorationViewer;
      w.play(false);
      w.setTime(9.8);
      return { position: w.camera.position.toArray(), time: w.t, playing: w.playing };
    });
    assert.equal(await page.evaluate(() => window.wander.clampOn), false);
    assert.equal(await page.locator('#scene-boundary, #boundary-return').count(), 0);
    // Each face of the old hard box can be crossed without correction on subsequent frames.
    for (const axis of [0, 1, 2]) {
      for (const side of [-1, 1]) {
        const requested = await page.evaluate(
          ({ axis, side, home }) => {
            const w = window.wander as ExplorationViewer;
            const position = [...home];
            position[axis] =
              (side < 0 ? w.hardBox.min : w.hardBox.max).getComponent(axis) + side * 2;
            w.place(position);
            return position;
          },
          { axis, side, home: home.position },
        );
        await settle();
        assert.deepEqual(
          await page.evaluate(() => window.wander.camera.position.toArray()),
          requested,
        );
      }
    }
    const startX = await page.evaluate(() => {
      const w = window.wander as ExplorationViewer;
      const p = [w.hardBox.max.x + 0.1, w.camera.position.y, w.camera.position.z];
      w.place(p, [p[0], p[1], p[2] - 1]);
      return p[0];
    });
    await page.keyboard.down('d');
    try {
      await page.waitForFunction((x) => window.wander.camera.position.x > x + 0.2, startX, {
        timeout: 5000,
      });
    } finally {
      await page.keyboard.up('d');
    }
    await page.evaluate(({ position }) => {
      const w = window.wander as ExplorationViewer;
      w.place([w.hardBox.max.x + 0.3, position[1], position[2]], [0, 0, -2]);
    }, home);
    await settle();
    const beyond = await page.evaluate(() => ({
      position: window.wander.camera.position.toArray(),
      filter: window.wander.spark.renderer.domElement.style.filter,
      overlay: document.querySelector<HTMLElement>('#edgefade')!.style.opacity,
      time: window.wander.t,
    }));
    assert.equal(beyond.filter, '');
    assert.equal(Number(beyond.overlay), 0);
    assert.equal(beyond.time, home.time);
    checks.push({ legacyManifest, beyond });
    await page.screenshot({
      path: `${output}/${legacyManifest ? 'legacy' : 'current'}-beyond.png`,
    });
    // Unlimited placement no longer substitutes a larger, but still finite, artificial box.
    assert.equal(
      await page.evaluate(() => {
        const w = window.wander as ExplorationViewer;
        w.place([2000000, 0, 0]);
        return w.camera.position.x;
      }),
      2000000,
    );
    await page.keyboard.press('r');
    assert.deepEqual(
      await page.evaluate(() => ({
        position: window.wander.camera.position.toArray(),
        time: window.wander.t,
        playing: window.wander.playing,
      })),
      home,
      'Home must remain available without changing playback time',
    );
    await page.screenshot({ path: `${output}/${legacyManifest ? 'legacy' : 'current'}-home.png` });
  }
  const diagnostic = new URL(url);
  diagnostic.searchParams.set('pause', '1');
  diagnostic.searchParams.set('freedom', 'tight');
  await page.goto(diagnostic.href, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.wander?.ready, null, { timeout: 120000 });
  assert.equal(await page.evaluate(() => window.wander.clampOn), true);
  assert.deepEqual(errors, []);
  console.log(
    'Default exploration crosses every scene-box face and the path leash without darkening; home and explicit diagnostic limits still work.',
  );
} finally {
  await writeFile(`${output}/report.json`, JSON.stringify({ url, checks, errors }, null, 2));
  await browser.close();
}
