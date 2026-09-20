// Real desktop wrapper + scene runtimes. Requires the private stairs2 and lobby assets.
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright-core';
import type * as THREE from 'three';
import type {} from './viewer-types';

declare global {
  interface Window {
    __demoSwitchRenderer: THREE.WebGLRenderer;
  }
}

const base = process.env.DEMO_TEST_URL || 'http://127.0.0.1:5399';
const browser = await chromium.launch({ channel: 'chrome', headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(`${base}/demo.html?clip=stairs2`, { timeout: 120000 });
  const element = await page.locator('#frame').elementHandle();
  const frame = await element?.contentFrame();
  assert.ok(frame);
  async function ready(clip: string) {
    await frame!.waitForFunction(
      (clip) => window.wander?.demo === clip && window.wander.ready,
      clip,
      { timeout: 180000 },
    );
    await page.waitForFunction(
      () =>
        document.querySelector('#loading')?.classList.contains('gone') &&
        !(document.querySelector('#play') as HTMLButtonElement).disabled,
    );
    assert.equal(new URL(page.url()).searchParams.get('clip'), clip);
    assert.equal(await page.locator('.card.on').getAttribute('data-id'), clip);
  }
  await ready('stairs2');
  assert.equal(await page.locator('#railbtn').getAttribute('aria-expanded'), 'false');
  await page.click('#helpbtn');
  assert.equal(await page.locator('#helppop').isVisible(), true);
  await page.click('#helpclose');
  await frame.evaluate(() => {
    window.__demoSwitchRenderer = window.wander.spark.renderer;
  });
  await page.waitForFunction(
    () => !(document.querySelector('#soundbtn') as HTMLButtonElement).disabled,
  );
  await page.click('#soundbtn');
  await frame.waitForFunction(() => window.wander.audioState.unlocked);
  await page.click('#railbtn');
  await page.click('.card[data-id="lobby"]');
  await ready('lobby');
  await frame.waitForFunction(
    () => window.wander.audioState.unlocked && !window.wander.audioState.muted,
  );
  assert.equal(
    await frame.evaluate(() => window.__demoSwitchRenderer === window.wander.spark.renderer),
    true,
    'The desktop picker must reuse the existing viewer and renderer',
  );
  // A source inset is recreated with the scene; header controls must follow the new root.
  const sourceHidden = await frame
    .locator('#pip')
    .evaluate((video) => video.classList.contains('hidden'));
  await page.click('#srcbtn');
  assert.equal(
    await frame.locator('#pip').evaluate((video) => video.classList.contains('hidden')),
    !sourceHidden,
  );
  await page.click('#srcbtn');
  assert.equal(
    await frame.locator('#pip').evaluate((video) => video.classList.contains('hidden')),
    sourceHidden,
  );
  await page.click('#soundbtn');
  await frame.waitForFunction(() => window.wander.audioState.muted);
  await page.click('.card[data-id="stairs2"]');
  await ready('stairs2');
  assert.equal(await frame.evaluate(() => window.wander.audioState.muted), true);
  assert.equal(
    await frame.evaluate(() => {
      const state = (window as unknown as { __sceneCache: { lastLoad: { reused: boolean } } })
        .__sceneCache;
      return state.lastLoad.reused;
    }),
    true,
  );
  assert.equal(
    await frame.evaluate(() => window.__demoSwitchRenderer === window.wander.spark.renderer),
    true,
  );
  await page.waitForFunction(
    () =>
      getComputedStyle(document.querySelector('#loading')!).opacity === '0' &&
      document.querySelector('#soundbtn')?.getAttribute('aria-pressed') === 'false',
  );
  assert.deepEqual(errors, []);
  await mkdir('.context/evidence/xr-scene-sidebar', { recursive: true });
  await page.screenshot({ path: '.context/evidence/xr-scene-sidebar/merged-desktop-shell.png' });
  console.log(
    'Desktop shell: header controls, live scene switching, cached return and mute persistence pass.',
  );
} finally {
  await browser.close();
}
