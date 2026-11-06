// Read-only browser proof using an existing real run asset through the same-origin proxy.
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright-core';
import { worldPreviewAssetUrl } from '../src/world-preview-path';

const base = process.env.WORLD_PREVIEW_TEST_URL || 'http://127.0.0.1:5399';
const run = process.env.WORLD_PREVIEW_TEST_RUN || 'local-e2e-delivery-20261105';
const response = await fetch(
  `${base}/api/pipeline/runs/${encodeURIComponent(run)}/viewer/scene.json`,
);
assert(response.ok, 'Existing local run scene must be readable');
const scene = (await response.json()) as { params: { world: string } };
const world = worldPreviewAssetUrl(scene.params.world);
const browser = await chromium.launch(
  process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : { channel: 'chrome' },
);
try {
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();
  const errors: string[] = [];
  const requests: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  page.on('requestfailed', (request) => errors.push(`Failed request: ${request.url()}`));
  page.on('response', (result) => {
    if (result.status() >= 400) errors.push(`HTTP ${result.status()}: ${result.url()}`);
  });
  page.on('request', (request) => requests.push(new URL(request.url()).pathname));
  await page.goto(`${base}/world-preview.html?world=${encodeURIComponent(world)}`);
  await page.waitForFunction(() => document.body.dataset.state === 'ready', null, {
    timeout: 120000,
  });
  assert.match(await page.locator('#status').innerText(), /splats · preview only/);
  assert.equal(await page.locator('a[href="/"]').count(), 1);
  assert.match(
    await page.locator('header').innerText(),
    /People and recorded audio are not included/,
  );
  assert.match(await page.locator('header').innerText(), /Scale and alignment are unverified/);
  assert.equal(await page.locator('video,audio').count(), 0);
  assert(requests.includes(world), 'World must load through its actual same-origin route');
  assert(
    !requests.some((path) => /people\.json|sequence\.json|\.mp4$|\.wav$/.test(path)),
    'World preview must not request people or audio/video',
  );
  await page.waitForTimeout(500);
  const canvas = page.locator('#viewport canvas');
  const before = await canvas.screenshot();
  await page.mouse.move(960, 600);
  await page.mouse.down();
  await page.mouse.move(1140, 630, { steps: 15 });
  await page.mouse.up();
  await page.waitForTimeout(500);
  const after = await canvas.screenshot();
  assert(!before.equals(after), 'Orbit interaction must change the rendered view');
  await page.locator('#reset').click();
  await page.waitForTimeout(500);
  await mkdir('.context/evidence/world-preview', { recursive: true });
  await page.screenshot({ path: '.context/evidence/world-preview/retained-world.png' });
  assert.deepEqual(errors, []);
  await page.setViewportSize({ width: 320, height: 568 });
  await page.waitForTimeout(100);
  const overflow = await page.evaluate(() => ({
    page: document.documentElement.scrollWidth > innerWidth,
    header:
      document.querySelector('header')!.scrollWidth > document.querySelector('header')!.clientWidth,
    status: document.querySelector('#status')!.getBoundingClientRect().right > innerWidth,
  }));
  assert.deepEqual(
    overflow,
    { page: false, header: false, status: false },
    'Long asset paths must wrap on narrow phones',
  );
  await page.screenshot({ path: '.context/evidence/world-preview/retained-world-mobile.png' });
  await context.close();

  const invalidContext = await browser.newContext();
  const invalidPage = await invalidContext.newPage();
  const assetRequests: string[] = [];
  invalidPage.on('request', (request) => {
    if (new URL(request.url()).pathname.endsWith('.spz')) assetRequests.push(request.url());
  });
  await invalidPage.goto(
    `${base}/world-preview.html?world=${encodeURIComponent('https://example.com/world.spz')}`,
  );
  await invalidPage.waitForFunction(() => document.body.dataset.state === 'error');
  assert.match(await invalidPage.locator('#status').innerText(), /Could not load/);
  assert.deepEqual(assetRequests, [], 'Invalid asset URLs fail before fetching');
  await invalidPage.goto(
    `${base}/world-preview.html?world=${encodeURIComponent('/api/pipeline/runs/does-not-exist/viewer/assets/missing.spz')}`,
  );
  await invalidPage.waitForFunction(() => document.body.dataset.state === 'error');
  assert.match(await invalidPage.locator('#status').innerText(), /Could not load/);
  assert.equal(
    await invalidPage.locator('canvas').count(),
    0,
    'Missing worlds must not masquerade as a ready scene',
  );
  const corruptWorld = '/api/pipeline/runs/corrupt-preview-fixture/viewer/assets/broken.spz';
  await invalidPage.route(`${base}${corruptWorld}`, (route) =>
    route.fulfill({
      contentType: 'application/octet-stream',
      body: Buffer.from('not-an-spz'),
    }),
  );
  await invalidPage.goto(`${base}/world-preview.html?world=${encodeURIComponent(corruptWorld)}`);
  await invalidPage.waitForFunction(() => document.body.dataset.state === 'error');
  assert.equal(
    await invalidPage.locator('canvas').count(),
    0,
    'Failed decoding must remove its canvas',
  );
  assert(await invalidPage.locator('#reset').isDisabled());
  assert(await invalidPage.locator('#overview').isDisabled());
  await invalidContext.close();
  console.log(
    'PASS actual retained SPZ rendered, orbit changed view, disclosures/back link present, no people/audio requests; invalid URL fails before fetch',
  );
} finally {
  await browser.close();
}
