// Build first. Uses the running viewer's media, without restarting its server.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright-core';

const origin = process.env.VIEWER_ORIGIN || 'http://127.0.0.1:5399';
const useServedBuild = process.env.HUD_USE_SERVED === '1';
const query = new URLSearchParams({
  world: '/reviews/kitchen-continuation/pan/environment.ply',
  people: '/reviews/kitchen-continuation/actor-repair/people.json',
  video: '/reviews/gym-kitchen/kitchen/source.mp4',
  placement: '/reviews/kitchen-continuation/actor-repair/floor-placement.json',
  worldup: 'y',
  worldscale: '1',
  scale: '1',
  floor: '-3.118813392508837',
  place: '1',
  feetmode: 'sfm',
  stance: '0',
  feetlock: '0',
  camdrift: '0',
  audio: '0',
  walk: '1',
  lod: '1',
});
const browser = await chromium.launch({ channel: 'chrome', headless: true });
try {
  const context = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  const html = await readFile('dist/fourd.html', 'utf8');
  if (!useServedBuild) {
    await context.route(`${origin}/__hud-regression.html*`, (route) =>
      route.fulfill({ contentType: 'text/html', body: html }),
    );
    await context.route(`${origin}/assets/**`, async (route) => {
      const name = path.basename(new URL(route.request().url()).pathname);
      await route.fulfill({ path: path.resolve('dist/assets', name) });
    });
  }
  const page = await context.newPage();
  const entry = useServedBuild ? '/fourd.html' : '/__hud-regression.html';
  await page.goto(`${origin}${entry}?${query}`);
  await page.waitForFunction(() => Reflect.get(window, 'wander')?.ready, null, { timeout: 120000 });
  await page.evaluate(() => {
    const viewer = Reflect.get(window, 'wander');
    viewer.play(false);
    viewer.setTime(0);
  });
  const initial = await page.locator('#st').textContent();
  await page.evaluate(() => {
    Reflect.set(window, 'hudTestErrors', []);
    for (const type of ['error', 'unhandledrejection']) {
      window.addEventListener(type, () => Reflect.get(window, 'hudTestErrors').push(type));
    }
    window.dispatchEvent(new ErrorEvent('error', { message: 'HUD regression late error' }));
    window.dispatchEvent(
      new PromiseRejectionEvent('unhandledrejection', {
        promise: Promise.resolve(),
        reason: new Error('HUD regression late rejection'),
      }),
    );
  });
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => Reflect.get(window, 'wander').t > 0.5);
  const samples = [];
  for (let i = 0; i < 5; i++) {
    await page.waitForTimeout(1000);
    samples.push(
      await page.evaluate(() => {
        const text = document.getElementById('st')!.textContent || '';
        return {
          text,
          lines: text.split('\n').length,
          height: document.getElementById('hud')!.getBoundingClientRect().height,
          fatal: Reflect.get(window, '__wanderStartupError'),
          errors: Reflect.get(window, 'hudTestErrors'),
          time: Reflect.get(window, 'wander').t,
        };
      }),
    );
  }
  assert.ok(samples.at(-1)!.time > samples[0].time + 1, 'Playback must advance during the check');
  console.log(JSON.stringify({ samples: samples.map(({ text, ...rest }) => rest) }));
  for (const sample of samples) {
    assert.equal(sample.fatal, false);
    assert.deepEqual(sample.errors, ['error', 'unhandledrejection']);
    assert.equal((sample.text.match(/person pos=/g) || []).length, 1);
    assert.ok(sample.lines < 25, `Unbounded status lines: ${sample.lines}`);
    assert.ok(sample.height <= 324, `Unbounded HUD height: ${sample.height}`);
    assert.ok(!sample.text.includes('Scene could not load'));
  }
  assert.notEqual(samples.at(-1)!.text, initial, 'Playback diagnostics must keep updating');
  // Any latched fatal diagnostic must survive later animation-frame completions unchanged.
  await page.evaluate(() => {
    Reflect.set(window, '__wanderStartupError', true);
    document.getElementById('st')!.textContent = 'Scene could not load. Reload to retry.';
  });
  await page.waitForTimeout(1000);
  assert.equal(await page.locator('#st').textContent(), 'Scene could not load. Reload to retry.');
  // A real failed module import must still display the fatal startup diagnostic.
  const startup = await context.newPage();
  await startup.route(`${origin}/assets/**`, (route) => route.abort());
  await startup.goto(`${origin}${entry}?${query}`);
  await startup.waitForFunction(() => Reflect.get(window, '__wanderStartupError') === true);
  await startup.waitForTimeout(1000);
  assert.equal(
    await startup.locator('#st').textContent(),
    'Scene could not load. Reload to retry.',
  );
  assert.equal(await startup.evaluate(() => !!Reflect.get(window, 'wander')), false);
  console.log(
    JSON.stringify({ lateErrors: samples.map(({ text, ...rest }) => rest), startup: 'pass' }),
  );
} finally {
  await browser.close();
}
