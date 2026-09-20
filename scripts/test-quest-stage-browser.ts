// Isolated browser fixture: verifies UI transport, not physical headset capture quality.
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright-core';

const source = await Bun.file(new URL('../src/quest-stage.ts', import.meta.url)).text();
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
    if (url.pathname === '/stage.js')
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
      `<!doctype html><html><head><style>body{margin:0;background:#06090c;color:white;font:14px system-ui}#stage{position:relative;height:calc(100dvh - 45px);width:100%;background:#345}</style></head><body><div style="height:45px">Header</div><main id="stage"><p id="viewer">Desktop viewer</p></main><script type="module">import {mountQuestStage} from '/stage.js';window.presenting=false;window.disposeStage=mountQuestStage({stage:document.querySelector('#stage'),isLocalPresenting:()=>window.presenting});</script></body></html>`,
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
  const view = page.locator('.quest-stage');
  const status = view.locator('[role=status]');
  const picture = view.locator('img');
  const shown = () => view.evaluate((element) => !(element as HTMLElement).hidden);
  // Playwright's isVisible ignores overlap, so ask what is actually on top at the stage centre.
  const onTop = () =>
    page.evaluate(() => {
      const rect = document.querySelector('#stage')!.getBoundingClientRect();
      const hit = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
      return hit?.closest('.quest-stage') ? 'quest' : hit?.id || hit?.tagName || 'none';
    });
  async function expectStatus(value: string) {
    await page.waitForFunction(
      (text) => document.querySelector('[role=status]')?.textContent === text,
      value,
    );
  }
  async function live() {
    await expectStatus('Live');
    await page.waitForFunction(() => {
      const image = document.querySelector<HTMLImageElement>('.quest-stage img')!;
      return image.complete && image.naturalWidth === 320 && image.naturalHeight === 240;
    });
    assert.ok(await shown(), 'a live image must replace the desktop viewer');
  }
  await expectStatus('Waiting for Quest');
  assert.ok(!(await shown()), 'nothing to show means the desktop viewer stays visible');
  assert.notEqual(await onTop(), 'quest');
  assert.equal(await status.getAttribute('aria-live'), 'polite');
  active = true;
  await live();
  assert.equal(await picture.evaluate((image) => getComputedStyle(image).objectFit), 'contain');
  const stageBounds = await page.locator('#stage').boundingBox();
  const viewBounds = await view.boundingBox();
  assert.ok(stageBounds && viewBounds);
  assert.deepEqual(
    [viewBounds!.x, viewBounds!.y, viewBounds!.width, viewBounds!.height],
    [stageBounds!.x, stageBounds!.y, stageBounds!.width, stageBounds!.height],
    'the live view fills the stage',
  );
  assert.equal(await onTop(), 'quest', 'the live view covers the desktop viewer');
  autoFrames = true;
  const cadence = await page.evaluate(async () => {
    const image = document.querySelector('.quest-stage img')!;
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
  await page.screenshot({ path: '.context/evidence/quest-debug/quest-stage-desktop.png' });
  emptyFrame = true;
  sequence++;
  await expectStatus('Waiting for Quest');
  assert.equal(await picture.getAttribute('src'), null);
  assert.ok(!(await shown()), 'an empty frame hands the stage back to the desktop viewer');
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
  // A new headset session invalidates the image from the old one before its frame arrives.
  session = 'quest-b';
  sequence = 1;
  delay = 400;
  await page.waitForFunction(
    () => !document.querySelector<HTMLImageElement>('.quest-stage img')?.getAttribute('src'),
    null,
    { timeout: 2000 },
  );
  assert.ok(!(await shown()), 'a session change clears the old image');
  delay = 0;
  await live();
  await page.setViewportSize({ width: 320, height: 568 });
  await page.evaluate(
    () =>
      new Promise<void>((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
      ),
  );
  const mobileStage = await page.locator('#stage').boundingBox();
  const mobileView = await view.boundingBox();
  assert.ok(mobileStage && mobileView);
  assert.deepEqual(
    [mobileView!.width, mobileView!.height],
    [mobileStage!.width, mobileStage!.height],
    'the live view keeps filling a narrow stage',
  );
  await page.screenshot({ path: '.context/evidence/quest-debug/quest-stage-mobile.png' });
  await page.evaluate(() => {
    (window as unknown as { disposeStage: () => void }).disposeStage();
  });
  const disposedRequests = requests;
  await page.waitForTimeout(450);
  assert.equal(requests, disposedRequests);
  assert.equal(await view.count(), 0);
  assert.notEqual(await onTop(), 'quest');
  assert.deepEqual(errors, []);
  console.log(
    `Quest stage: waiting, JPEG render, ${cadence} applied updates/sec, displayed ${displayedRate} fps, stale/stop, errors, hidden, local XR, session change, narrow stage, disposal passed.`,
  );
} finally {
  await browser.close();
  server.stop(true);
}
