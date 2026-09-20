// Uses the real renderer and private lobby assets. Prepare /marble-lobby-clean.spz first.
// These desktop measurements are not headset performance evidence.
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright-core';
import type { SplatMesh } from '@sparkjsdev/spark';
import type {} from './viewer-types';

const base = process.env.PREPARED_WORLD_TEST_URL || 'http://127.0.0.1:5399';
const output = '.context/evidence/prepared-world';
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const context = await browser.newContext({ viewport: { width: 1200, height: 800 } });
const records: Record<string, unknown> = {};
let expected: unknown;
try {
  await mkdir(output, { recursive: true });
  for (const mode of ['original', 'prepared', 'corrupt-fallback']) {
    const page = await context.newPage();
    const errors: string[] = [];
    const warnings: string[] = [];
    const requests: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));
    page.on('console', (message) => {
      if (message.type() === 'warning') warnings.push(message.text());
    });
    page.on('request', (request) => {
      if (request.method() === 'GET') requests.push(new URL(request.url()).pathname);
    });
    if (mode === 'corrupt-fallback')
      await page.route('**/api/prepared-world/**', (route) =>
        route.fulfill({ contentType: 'application/octet-stream', body: 'invalid cache' }),
      );
    const started = performance.now();
    await page.goto(
      `${base}/fourd.html?demo=lobby&pause=1&xrview=0&xrworldcache=${mode === 'original' ? 0 : 1}`,
      { waitUntil: 'load', timeout: 120000 },
    );
    await page.waitForFunction(() => window.wander?.ready, null, { timeout: 180000 });
    const loadMilliseconds = Math.round(performance.now() - started);
    await page.waitForFunction(
      () => {
        const w = window.wander as typeof window.wander & { world: SplatMesh };
        return w.world.numSplats > 100000 && w.spark.renderer.info.render.frame > 30;
      },
      null,
      { timeout: 30000 },
    );
    const renderedMilliseconds = Math.round(performance.now() - started);
    await page.evaluate(() => window.wander.setTime(0));
    await page.waitForFunction(() => !window.wander.video.seeking);
    const result = await page.evaluate(async () => {
      const w = window.wander as typeof window.wander & { world: SplatMesh };
      const stages = (window as unknown as { __stages: Record<string, number | boolean> }).__stages;
      const packed = w.world.packedSplats!;
      const lod = packed.lodSplats!;
      async function hash(array: Uint32Array) {
        const bytes = array.slice().buffer;
        return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (byte) =>
          byte.toString(16).padStart(2, '0'),
        ).join('');
      }
      const arrays: Record<string, string> = { packed: await hash(lod.packedArray!) };
      for (const [key, value] of Object.entries(lod.extra))
        if (value instanceof Uint32Array) arrays[key] = await hash(value);
      return {
        prepared: stages.worldPrepared,
        worldMilliseconds: Math.round((Number(stages.worldReady) - Number(stages.world)) * 1000),
        geometry: {
          count: lod.numSplats,
          capacity: lod.maxSplats,
          encoding: lod.splatEncoding,
          rootEncoding: packed.splatEncoding,
          arrays,
          camera: w.camera.position.toArray(),
        },
      };
    });
    assert.equal(result.prepared, mode === 'prepared');
    assert.ok(result.geometry.count > 1000000);
    if (mode === 'original') expected = result.geometry;
    else assert.deepEqual(result.geometry, expected, 'Prepared geometry must preserve every word');
    assert.equal(requests.includes('/marble-lobby-clean.spz'), mode !== 'prepared');
    if (mode === 'corrupt-fallback')
      assert.ok(warnings.some((warning) => warning.includes('Prepared world unavailable')));
    assert.deepEqual(errors, []);
    await page.screenshot({ path: `${output}/${mode}.png` });
    records[mode] = { loadMilliseconds, renderedMilliseconds, ...result };
    console.log(mode, JSON.stringify({ loadMilliseconds, renderedMilliseconds, ...result }));
    await page.close();
  }
  await writeFile(`${output}/browser.json`, JSON.stringify(records, null, 2));
} finally {
  await browser.close();
}
