// Isolated browser fixture: verifies UI transport, not physical headset capture quality.
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright-core';

const source = await Bun.file(new URL('../src/quest-view-panel.ts', import.meta.url)).text();
const module = new Bun.Transpiler({ loader: 'ts' }).transformSync(source);
let active = false;
let stale = false;
let failure = false;
let emptyFrame = false;
let autoFrames = false;
let sequence = 1;
let session = 'quest-a';
let requests = 0;
let delay = 0;
let jpeg = Buffer.alloc(0);
const server = Bun.serve({
  hostname: '127.0.0.1',
  port: 0,
  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname === '/panel.js')
      return new Response(module, { headers: { 'Content-Type': 'application/javascript' } });
    if (url.pathname === '/api/quest-view') {
      requests++;
      if (failure) return new Response('Unavailable', { status: 503 });
      if (autoFrames) sequence++;
      return Response.json({
        active,
        sessionId: session,
        clip: 'fixture',
        frame: sequence,
        updatedAt: Date.now() - (stale ? 3000 : 0),
        viewers: 1,
      });
    }
    if (url.pathname === '/api/quest-view/frame') {
      if (emptyFrame) return new Response(null, { status: 204 });
      if (delay) await Bun.sleep(delay);
      return new Response(jpeg, {
        headers: { 'Content-Type': 'image/jpeg', 'X-Quest-Frame': String(sequence) },
      });
    }
    return new Response(
      `<!doctype html><html><head><style>body{margin:0;background:#06090c;color:white;font:14px system-ui}#stage{position:relative;height:calc(100dvh - 45px);width:100%}#toggle{height:40px}#bar{position:absolute;bottom:15px;height:45px;left:16px;right:16px;background:#252a2d;border-radius:8px}</style></head><body><button id="toggle">Quest view</button><main id="stage"><div id="bar">Playback controls</div></main><script type="module">import {mountQuestViewPanel} from '/panel.js';window.presenting=false;window.disposePanel=mountQuestViewPanel({stage:document.querySelector('#stage'),toggle:document.querySelector('#toggle'),isLocalPresenting:()=>window.presenting});</script></body></html>`,
      { headers: { 'Content-Type': 'text/html' } },
    );
  },
});
const browser = await chromium.launch({ channel: 'chrome', headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${server.port}`);
  const encoded = await page.evaluate(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 240;
    const context = canvas.getContext('2d')!;
    context.fillStyle = '#314d58';
    context.fillRect(0, 0, 320, 240);
    context.fillStyle = '#dfbe70';
    context.fillRect(30, 35, 70, 160);
    context.fillStyle = '#f6f0df';
    context.font = '20px sans-serif';
    context.fillText('Quest fixture', 115, 120);
    return canvas.toDataURL('image/jpeg').split(',')[1];
  });
  jpeg = Buffer.from(encoded, 'base64');
  const panel = page.locator('.quest-view-panel');
  const status = panel.locator('[role=status]');
  const picture = panel.locator('img');
  async function expectStatus(value: string) {
    await page.waitForFunction(
      (text) => document.querySelector('[role=status]')?.textContent === text,
      value,
    );
  }
  async function live() {
    await expectStatus('Live');
    await page.waitForFunction(() => {
      const image = document.querySelector('img')!;
      return (
        !image.hidden && image.complete && image.naturalWidth === 320 && image.naturalHeight === 240
      );
    });
  }
  await expectStatus('Waiting for Quest');
  assert.equal(await page.locator('#toggle').getAttribute('aria-expanded'), 'true');
  assert.equal(
    await page.locator('#toggle').getAttribute('aria-controls'),
    await panel.getAttribute('id'),
  );
  assert.equal(await status.getAttribute('aria-live'), 'polite');
  active = true;
  await live();
  assert.equal(await picture.evaluate((image) => getComputedStyle(image).objectFit), 'contain');
  const desktopBounds = await panel.boundingBox();
  assert.ok(desktopBounds && desktopBounds.width >= 370 && desktopBounds.width <= 382);
  const desktopPicture = await panel.locator('.quest-view-picture').boundingBox();
  assert.ok(desktopPicture);
  assert.ok(Math.abs(desktopPicture!.width / desktopPicture!.height - 4 / 3) < 0.03);
  await page.getByRole('button', { name: 'Expand Quest view' }).click();
  const expandedBounds = await panel.boundingBox();
  assert.ok(
    expandedBounds && expandedBounds.width > desktopBounds!.width && expandedBounds.width <= 542,
  );
  await page.getByRole('button', { name: 'Reduce Quest view' }).click();
  assert.equal(await page.getByRole('button', { name: 'Expand Quest view' }).count(), 1);
  autoFrames = true;
  const cadence = await page.evaluate(async () => {
    const image = document.querySelector('.quest-view-panel img')!;
    let updates = 0;
    let previous = image.getAttribute('src');
    const observer = new MutationObserver(() => {
      const next = image.getAttribute('src');
      if (next && next !== previous) {
        updates++;
        previous = next;
      }
    });
    observer.observe(image, { attributes: true, attributeFilter: ['src'] });
    await new Promise((resolve) => setTimeout(resolve, 1000));
    observer.disconnect();
    return updates;
  });
  autoFrames = false;
  assert.ok(cadence > 30, `active fixture should publish over 30 frames/sec, got ${cadence}`);
  await page.waitForFunction(() =>
    /^\d+ fps$/.test(document.querySelector('[data-role="rate"]')?.textContent ?? ''),
  );
  const displayedRate = Number.parseInt(
    (await page.locator('[data-role="rate"]').textContent()) ?? '',
    10,
  );
  assert.ok(
    displayedRate > 30 && displayedRate <= 75,
    `displayed FPS should be sane, got ${displayedRate}`,
  );
  await mkdir('.context/evidence/quest-debug', { recursive: true });
  await page.screenshot({ path: '.context/evidence/quest-debug/quest-view-panel-desktop.png' });
  emptyFrame = true;
  sequence++;
  await expectStatus('Waiting for Quest');
  assert.equal(await picture.getAttribute('src'), null);
  emptyFrame = false;
  await live();
  stale = true;
  await expectStatus('Connection lost');
  assert.equal(await picture.getAttribute('src'), null);
  assert.equal(await page.locator('[data-role="rate"]').getAttribute('hidden'), '');
  stale = false;
  sequence++;
  await live();
  active = false;
  await expectStatus('Waiting for Quest');
  assert.equal(await picture.getAttribute('src'), null);
  active = true;
  sequence++;
  await live();
  failure = true;
  await expectStatus('Connection lost');
  assert.equal(await picture.getAttribute('src'), null);
  failure = false;
  sequence++;
  await live();
  await page.getByRole('button', { name: 'Minimize Quest view' }).click();
  assert.equal(await page.locator('#toggle').getAttribute('aria-expanded'), 'false');
  const closedRequests = requests;
  await page.waitForTimeout(450);
  assert.equal(requests, closedRequests, 'closed panel must stop requests');
  await page.locator('#toggle').click();
  await live();
  await page.evaluate(() => {
    Object.defineProperty(document, 'hidden', { configurable: true, value: true });
    document.dispatchEvent(new Event('visibilitychange'));
  });
  const hiddenRequests = requests;
  await page.waitForTimeout(450);
  assert.equal(requests, hiddenRequests, 'hidden tab must stop requests');
  assert.equal(await picture.getAttribute('src'), null);
  await page.evaluate(() => {
    Object.defineProperty(document, 'hidden', { configurable: true, value: false });
    document.dispatchEvent(new Event('visibilitychange'));
  });
  await live();
  await page.evaluate(() => {
    (window as unknown as { presenting: boolean }).presenting = true;
  });
  await expectStatus('Viewing on this Quest');
  const localRequests = requests;
  await page.waitForTimeout(450);
  assert.equal(requests, localRequests, 'local headset must not receive its own image');
  assert.equal(await picture.getAttribute('src'), null);
  await page.evaluate(() => {
    (window as unknown as { presenting: boolean }).presenting = false;
  });
  await live();
  session = 'quest-b';
  sequence = 1;
  delay = 400;
  await page.waitForTimeout(210);
  await page.locator('#toggle').click();
  await page.waitForTimeout(500);
  assert.equal(
    await picture.getAttribute('src'),
    null,
    'late frame cannot repopulate a closed panel',
  );
  delay = 0;
  await page.locator('#toggle').click();
  await live();
  await page.setViewportSize({ width: 320, height: 568 });
  let bounds = await panel.boundingBox();
  assert.ok(
    bounds &&
      bounds.x >= 0 &&
      bounds.y >= 0 &&
      bounds.x + bounds.width <= 320 &&
      bounds.y + bounds.height <= 568,
  );
  await page.screenshot({ path: '.context/evidence/quest-debug/quest-view-panel-mobile.png' });
  await page.setViewportSize({ width: 320, height: 180 });
  await page.evaluate(
    () =>
      new Promise<void>((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
      ),
  );
  bounds = await panel.boundingBox();
  assert.ok(
    bounds &&
      bounds.width >= 220 &&
      bounds.x >= 0 &&
      bounds.y >= 0 &&
      bounds.x + bounds.width <= 320 &&
      bounds.y + bounds.height <= 180,
    'panel must remain readable and fit a short stage',
  );
  for (const name of ['Expand Quest view', 'Minimize Quest view']) {
    const button = page.getByRole('button', { name });
    const buttonBounds = await button.boundingBox();
    assert.ok(
      buttonBounds &&
        bounds &&
        buttonBounds.x >= bounds.x &&
        buttonBounds.y >= bounds.y &&
        buttonBounds.x + buttonBounds.width <= bounds.x + bounds.width &&
        buttonBounds.y + buttonBounds.height <= bounds.y + bounds.height,
      `${name} must be inside short panel`,
    );
  }
  await page.getByRole('button', { name: 'Expand Quest view' }).click();
  await page.getByRole('button', { name: 'Reduce Quest view' }).click();
  await page.screenshot({ path: '.context/evidence/quest-debug/quest-view-panel-short.png' });
  await page.evaluate(() => {
    (window as unknown as { disposePanel: () => void }).disposePanel();
  });
  const disposedRequests = requests;
  await page.waitForTimeout(450);
  assert.equal(requests, disposedRequests);
  assert.equal(await panel.count(), 0);
  assert.deepEqual(errors, []);
  console.log(
    `Quest view panel: waiting, JPEG render, ${cadence} applied updates/sec, displayed ${displayedRate} fps, stale/stop, errors, close, hidden, local XR, late response, responsive bounds, disposal passed.`,
  );
} finally {
  await browser.close();
  server.stop(true);
}
