// Real video transport with simulated XR; this does not measure headset playback quality.
import assert from 'node:assert/strict';
import { chromium } from 'playwright-core';
import type {} from './viewer-types.ts';

const browser = await chromium.launch({ channel: 'chrome', headless: true });
const base = process.env.XR_TEST_URL || 'http://127.0.0.1:5399';
try {
  const page = await browser.newPage({ viewport: { width: 1000, height: 700 } });
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.addInitScript({ path: new URL('./fake-webxr.js', import.meta.url).pathname });
  await page.goto(
    `${base}/fourd.html?demo=stairs2&xr=1&fakexr=1&pause=1&fakew=256&fakeh=256&xradapt=0`,
    { waitUntil: 'load', timeout: 120000 },
  );
  await page.waitForFunction(
    () => window.wander?.ready && window.wander.video.readyState >= 3,
    null,
    {
      timeout: 180000,
    },
  );
  assert.equal(await page.evaluate(() => window.wander.playing), false);
  await page.click('#xrBtn');
  await page.waitForFunction(
    () =>
      window.wander.spark.renderer.xr.isPresenting &&
      window.wander.playing &&
      !window.wander.video.paused,
    null,
    { timeout: 10000 },
  );
  console.log('XR playback: paused scene starts on entry');

  const interval = await page.evaluate(() => {
    const w = window.wander;
    return Math.min(w.dur, w.video.duration);
  });
  assert.ok(Number.isFinite(interval) && interval > 1);
  await page.evaluate((end) => window.wander.setTime(end - 0.3), interval);
  for (let wrap = 0; wrap < 2; wrap++) {
    await page.waitForFunction((end) => {
      const w = window.wander;
      return !w.video.seeking && w.video.currentTime > end - 0.4;
    }, interval);
    await page.waitForFunction(() => {
      const w = window.wander;
      return w.playing && !w.video.paused && !w.video.seeking && w.video.currentTime < 1 && w.t < 1;
    });
    const state = await page.evaluate(() => ({
      scene: window.wander.t,
      video: window.wander.video.currentTime,
      loop: window.wander.video.loop,
    }));
    assert.equal(state.loop, true);
    assert.ok(Math.abs(state.scene - state.video) < 0.2, JSON.stringify(state));
  }
  console.log('XR playback: two source/scene loop wraps stay synchronized');

  await page.evaluate(() => window.wander.play(false));
  const paused = await page.evaluate(() => ({
    t: window.wander.t,
    frames: window.__fakeXR.frames.length,
  }));
  await page.waitForFunction((count) => window.__fakeXR.frames.length > count + 12, paused.frames);
  assert.deepEqual(
    await page.evaluate(() => ({
      playing: window.wander.playing,
      paused: window.wander.video.paused,
      t: window.wander.t,
    })),
    { playing: false, paused: true, t: paused.t },
    'Explicit pause must persist while VR frames continue',
  );

  await page.evaluate(() => window.wander.play(true));
  await page.waitForFunction(() => window.wander.playing && !window.wander.video.paused);
  await page.evaluate(async () => {
    await window.wander.spark.renderer.xr.getSession()!.end();
  });
  assert.equal(await page.evaluate(() => window.wander.playing), false);
  assert.equal(await page.evaluate(() => window.wander.video.paused), true);
  await page.click('#xrBtn');
  await page.waitForFunction(() => window.wander.playing && !window.wander.video.paused);
  await page.waitForFunction((time) => window.wander.t !== time, paused.t);
  console.log('XR playback: user pause persists; exit pauses; re-entry resumes');
  assert.deepEqual(errors, []);
} finally {
  await browser.close();
}
