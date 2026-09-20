// Real viewer with synthetic XR input; timings here are not native headset measurements.
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright-core';
import type {} from './viewer-types';

const base = process.env.XR_TEST_URL || 'http://127.0.0.1:5399';
const output = '.context/evidence/xr-animation';
const browser = await chromium.launch({ channel: 'chrome', headless: true });
try {
  await mkdir(output, { recursive: true });
  const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.addInitScript({ path: new URL('./fake-webxr.js', import.meta.url).pathname });
  const results = [];
  for (const fps of [36, 0]) {
    await page.goto(
      `${base}/fourd.html?demo=gym-accepted&xr=1&fakexr=1&fakew=384&fakeh=384&xrmove=smooth&xrview=0&xranimfps=${fps}`,
      { waitUntil: 'domcontentloaded' },
    );
    await page.waitForFunction(() => window.wander?.ready, null, { timeout: 180000 });
    await page.click('#xrBtn');
    await page.waitForFunction(() => window.wander.spark.renderer.xr.isPresenting);
    const result = await page.evaluate(async () => {
      const w = window.wander as typeof window.wander & {
        stats: { frames: number };
      };
      const person = w.people[0] as (typeof w.people)[number] & {
        step(time: number, snap: boolean): number;
      };
      const original = person.step;
      let updates = 0;
      person.step = function (...args) {
        updates++;
        return original.apply(this, args);
      };
      try {
        w.play(false);
        w.setTime(1);
        const seekUpdates = updates;
        const seekTime = w.t;
        updates = 0;
        w.play(true);
        const start = performance.now(),
          initialTime = w.t,
          initialFrames = w.stats.frames;
        await new Promise((resolve) => setTimeout(resolve, 3000));
        const result = {
          seconds: (performance.now() - start) / 1000,
          updates,
          frames: w.stats.frames - initialFrames,
          mediaSeconds: w.t - initialTime,
          seekUpdates,
          seekTime,
        };
        w.play(false);
        return result;
      } finally {
        person.step = original;
      }
    });
    assert.equal(result.seekUpdates, 1, 'Explicit seeking bypasses the cadence');
    assert.equal(result.seekTime, 1);
    assert.ok(result.mediaSeconds > result.seconds * 0.8, 'The media clock keeps normal speed');
    assert.ok(result.frames > 60, 'XR continues rendering throughout the sample');
    if (fps) {
      assert.ok(
        result.updates <= Math.ceil(result.seconds * fps) + 1,
        'Recorded animation respects the cap',
      );
      assert.ok(
        result.frames > result.updates * 1.2,
        'Headset rendering continues between animation updates',
      );
    } else
      assert.ok(
        Math.abs(result.frames - result.updates) <= 1,
        'Opt-out updates every rendered frame',
      );
    await page.screenshot({ path: `${output}/gym-${fps}.png` });
    results.push({ fps, ...result });
  }
  assert.deepEqual(errors, []);
  await writeFile(`${output}/results.json`, JSON.stringify(results, null, 2));
  console.log(JSON.stringify(results));
} finally {
  await browser.close();
}
