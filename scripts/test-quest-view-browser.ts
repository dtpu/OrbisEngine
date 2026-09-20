// Production viewer + relay, with synthetic XR input. This is not native headset evidence.
// Run `bun run build` first. An isolated HTTP fixture leaves the shared Vite server untouched.
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { createReadStream } from 'node:fs';
import { mkdir, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { loadEnv } from 'vite';
import { createQuestViewMiddleware } from '../server/quest-view.ts';
import { sharedAssets } from '../server/shared-assets.ts';

const relay = createQuestViewMiddleware();
let uploadDelay = 0;
let activeUploads = 0;
let maxUploads = 0;
const uploads: { start: number; end?: number }[] = [];
const assets = sharedAssets({
  ...loadEnv('development', process.cwd(), 'WANDER_'),
  ...process.env,
});
const server = createServer((req, res) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'credentialless');
  const serve = () =>
    relay(req, res, () => {
      const serveBuilt = async () => {
        const pathname = new URL(req.url!, 'http://localhost').pathname;
        const root = path.resolve('dist');
        const file = path.resolve(root, '.' + pathname);
        if (!file.startsWith(root + path.sep)) {
          res.writeHead(403).end();
          return;
        }
        try {
          const info = await stat(file);
          if (!info.isFile()) throw new Error('Not a file');
          const contentType = file.endsWith('.html')
            ? 'text/html'
            : file.endsWith('.css')
              ? 'text/css'
              : file.endsWith('.woff2')
                ? 'font/woff2'
                : file.endsWith('.woff')
                  ? 'font/woff'
                  : file.endsWith('.js')
                    ? 'text/javascript'
                    : 'application/octet-stream';
          res.setHeader('Content-Type', contentType);
          res.setHeader('Content-Length', info.size);
          createReadStream(file).pipe(res);
        } catch {
          res.writeHead(404).end();
        }
      };
      if (req.url?.startsWith('/assets/') || req.url?.split('?')[0].endsWith('.html'))
        void serveBuilt();
      else
        void assets.middleware(req, res, () => {
          res.writeHead(404).end();
        });
    });
  if (req.method === 'POST' && req.url?.startsWith('/api/quest-view/frame')) {
    const sample = { start: Date.now(), end: undefined as number | undefined };
    uploads.push(sample);
    activeUploads++;
    maxUploads = Math.max(maxUploads, activeUploads);
    res.once('close', () => {
      activeUploads--;
      sample.end = Date.now();
    });
    if (uploadDelay) setTimeout(() => void serve(), uploadDelay);
    else void serve();
  } else void serve();
});
await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const out = '.context/evidence/quest-debug';
try {
  const producer = await browser.newPage({ viewport: { width: 1000, height: 800 } });
  const viewer = await browser.newPage({ viewport: { width: 1440, height: 960 } });
  const errors: string[] = [];
  for (const page of [producer, viewer])
    page.on('pageerror', (error) => errors.push(error.message));
  await producer.addInitScript({ path: new URL('./fake-webxr.js', import.meta.url).pathname });
  await Promise.all([
    producer.goto(
      `${base}/fourd.html?demo=stairs2&xr=1&fakexr=1&fakew=512&fakeh=512&xrfbscale=1&xradapt=0&clamp=0&xrmove=smooth`,
      { timeout: 120000 },
    ),
    viewer.goto(`${base}/demo.html?clip=stairs2`, { timeout: 120000 }),
  ]);
  await producer.waitForFunction(() => window.wander?.ready, null, { timeout: 180000 });
  await viewer.waitForFunction(
    () => (window.frames[0] as Window & { wander?: { ready?: boolean } })?.wander?.ready,
    null,
    { timeout: 180000 },
  );
  await viewer.waitForFunction(
    () => document.querySelector('#loading')?.classList.contains('gone'),
    null,
    { timeout: 180000 },
  );
  await viewer.evaluate(() => document.fonts?.ready);
  const panel = viewer.locator('.quest-view-panel');
  const live = () =>
    viewer.waitForFunction(
      () => document.querySelector('.quest-view-panel [role=status]')?.textContent === 'Live',
      null,
      { timeout: 20000 },
    );
  await producer.click('#xrBtn');
  await producer.waitForFunction(() => window.wander.spark.renderer.xr.isPresenting);
  await producer.evaluate(() => window.wander.play(false));
  try {
    await live();
  } catch (error) {
    console.error('Sender:', await producer.evaluate(() => window.__xr));
    throw error;
  }
  await mkdir(out, { recursive: true });

  async function compareEye() {
    // Freeze head/playback while Spark settles, then compare JPEG to the actual rendered left eye.
    await producer.waitForTimeout(800);
    return producer.evaluate(async () => {
      const renderer = window.wander.spark.renderer;
      const viewport = renderer.xr.getCamera().cameras[0].viewport;
      const state = await (await fetch('/api/quest-view')).json();
      const jpeg = await (
        await fetch(`/api/quest-view/frame?session=${state.sessionId}&seq=${state.frame}`)
      ).blob();
      const decoded = await createImageBitmap(jpeg);
      const canvas = document.createElement('canvas');
      canvas.width = decoded.width;
      canvas.height = decoded.height;
      const context = canvas.getContext('2d')!;
      context.drawImage(decoded, 0, 0);
      const actual = context.getImageData(0, 0, canvas.width, canvas.height).data;
      const input = renderer.domElement;
      context.drawImage(
        input,
        viewport.x,
        input.height - viewport.y - viewport.w,
        viewport.z,
        viewport.w,
        0,
        0,
        canvas.width,
        canvas.height,
      );
      const expected = context.getImageData(0, 0, canvas.width, canvas.height).data;
      let error = 0;
      let min = 255;
      let max = 0;
      let total = 0;
      for (let i = 0; i < actual.length; i++) {
        if (i % 4 === 3) continue;
        error += Math.abs(actual[i] - expected[i]);
        min = Math.min(min, actual[i]);
        max = Math.max(max, actual[i]);
        total += actual[i];
      }
      decoded.close();
      return {
        error: error / (actual.length * 0.75),
        span: max - min,
        brightness: total / (actual.length * 0.75),
        width: canvas.width,
        height: canvas.height,
        glError: renderer.getContext().getError(),
        frame: state.frame,
      };
    });
  }
  const before = await compareEye();
  assert.equal(before.width, 512);
  assert.equal(before.height, 512);
  assert.ok(before.span > 100, `image must contain scene detail: ${JSON.stringify(before)}`);
  assert.ok(
    before.error < 15,
    `left-eye JPEG must match rendered pixels: ${JSON.stringify(before)}`,
  );
  assert.equal(before.glError, 0);
  const cadence = await viewer.evaluate(
    () =>
      new Promise<{ frames: number; elapsed: number }>((resolve) => {
        const picture = document.querySelector('.quest-view-panel img')!;
        let frames = 0;
        const started = performance.now();
        const observer = new MutationObserver((records) => {
          frames += records.length;
        });
        observer.observe(picture, { attributes: true, attributeFilter: ['src'] });
        setTimeout(() => {
          observer.disconnect();
          resolve({ frames, elapsed: performance.now() - started });
        }, 2000);
      }),
  );
  const sender = await producer.evaluate(() => (window.__xr as { questView: unknown }).questView);
  console.log('Quest preview cadence:', JSON.stringify({ cadence, sender }));
  assert.ok(
    cadence.frames >= 60,
    `preview should present at least 30 fps on the local fixture: ${JSON.stringify(cadence)}`,
  );
  await viewer.screenshot({ path: `${out}/quest-view-integrated.png` });
  await producer.evaluate(() => {
    window.__fakeXR.head.yaw = 0.6;
    window.__fakeXR.head.pitch = -0.35;
  });
  const after = await compareEye();
  assert.ok(after.frame > before.frame);
  assert.ok(after.error < 15, `preview follows head rotation: ${JSON.stringify(after)}`);
  assert.equal(after.glError, 0);
  assert.ok(Math.abs(after.brightness - before.brightness) > 1, 'turning changes the preview');
  await viewer.screenshot({ path: `${out}/quest-view-turned.png` });
  // Delayed uploads must overlap capture without adding an unbounded queue or concurrent POSTs.
  const slowStart = uploads.length;
  uploadDelay = 140;
  await producer.evaluate(() => {
    const gl = window.wander.spark.renderer.getContext() as WebGL2RenderingContext;
    const original = gl.blitFramebuffer;
    const samples: number[] = [];
    Object.assign(window, {
      __previewCopies: samples,
      __restorePreviewCopy: () => {
        gl.blitFramebuffer = original;
      },
    });
    gl.blitFramebuffer = function (...args) {
      samples.push(Date.now());
      return original.apply(this, args);
    };
  });
  await producer.waitForTimeout(900);
  const copies = await producer.evaluate(() => {
    const diagnostic = window as unknown as {
      __previewCopies: number[];
      __restorePreviewCopy: () => void;
    };
    diagnostic.__restorePreviewCopy();
    return diagnostic.__previewCopies;
  });
  const slowUploads = uploads.slice(slowStart);
  assert.equal(maxUploads, 1, 'only one frame upload may be in flight');
  assert.ok(
    copies.length <= slowUploads.length + 2,
    'capture queue is bounded while uploads are slow',
  );
  assert.ok(
    copies.some((time) =>
      slowUploads.some((upload) => upload.end && time > upload.start && time < upload.end),
    ),
    'capture should overlap the previous upload',
  );
  uploadDelay = 0;
  await live();
  await panel.getByRole('button', { name: 'Minimize Quest view' }).click();
  await producer.waitForFunction(
    () => (window.__xr as { questView: { viewers: number } }).questView.viewers === 0,
    null,
    { timeout: 10000 },
  );
  const stopped = await producer.evaluate(() => ({
    count: (window.__xr as { questView: { frames: number } }).questView.frames,
    xrFrames: window.__fakeXR.frames.length,
  }));
  await producer.waitForTimeout(600);
  assert.equal(
    await producer.evaluate(
      () => (window.__xr as { questView: { frames: number } }).questView.frames,
    ),
    stopped.count,
    'no capture when nobody watches',
  );
  assert.ok(
    (await producer.evaluate(() => window.__fakeXR.frames.length)) > stopped.xrFrames,
    'headset keeps rendering',
  );
  await viewer.locator('#questviewbtn').click();
  await live();
  await producer.evaluate(async () => {
    await window.wander.spark.renderer.xr.getSession()!.end();
  });
  await viewer.waitForFunction(
    () =>
      document.querySelector('.quest-view-panel [role=status]')?.textContent ===
      'Waiting for Quest',
  );
  assert.equal(await panel.locator('img').getAttribute('src'), null);
  await producer.click('#xrBtn');
  await live();
  assert.deepEqual(errors, []);
  await writeFile(
    `${out}/quest-view-measurements.json`,
    JSON.stringify(
      { before, after, cadence, sender, slowUploads, copies, maxUploads, errors },
      null,
      2,
    ),
  );
  console.log(
    'Quest view integration passed: rendered left eye, JPEG orientation, head rotation, GL state, live panel, idle capture pause, session end/re-entry. Synthetic XR only.',
  );
} finally {
  await browser.close();
  server.closeAllConnections();
  await new Promise<void>((resolve) => server.close(() => resolve()));
}
