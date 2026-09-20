// Build first. Supply a scene fixture with a reviewed route and expected map probes.
// WALK_TEST_CONFIG=... WALK_TEST_DIST=dist bun scripts/test-walk-support-browser.ts
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright-core';
import type { ViewerDiagnostics } from './viewer-types.ts';
type SupportViewer = Omit<ViewerDiagnostics, 'walk'> & {
  walk: {
    floor: number;
    canStand(x: number, z: number): boolean;
    blockedAt(x: number, z: number, floor?: number): number;
    cellAt(x: number, z: number): { floor: number; inside: number; dist: number };
  };
};

interface Fixture {
  url: string;
  waypoints: [number, number][];
  probes: { x: number; z: number; placeable: boolean; blocked: boolean }[];
  time?: number;
}
const config = process.env.WALK_TEST_CONFIG;
assert.ok(config, 'WALK_TEST_CONFIG must name a JSON scene fixture');
const fixture: Fixture = JSON.parse(await readFile(config, 'utf8'));
const url = new URL(fixture.url);
const out = path.resolve(process.env.WALK_TEST_OUT || '.context/evidence/walk-support-browser');
await mkdir(out, { recursive: true });
const report: { errors: string[]; route: unknown[]; probes?: unknown; failure?: string } = {
  errors: [],
  route: [],
};
const browser = await chromium.launch({ channel: 'chrome', headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  page.on('pageerror', (error) => report.errors.push(error.message));
  if (process.env.WALK_TEST_DIST) {
    const dist = path.resolve(process.env.WALK_TEST_DIST);
    await page.route(`${url.origin}${url.pathname}?*`, (route) =>
      route.fulfill({ path: path.join(dist, 'fourd.html'), contentType: 'text/html' }),
    );
    await page.route(`${url.origin}/assets/*`, (route) =>
      route.fulfill({
        path: path.join(dist, 'assets', path.basename(new URL(route.request().url()).pathname)),
      }),
    );
  }
  await page.goto(url.href);
  await page.waitForFunction(() => Reflect.get(window, 'wander')?.ready, null, { timeout: 180000 });
  await page.evaluate((time) => {
    const w = Reflect.get(window, 'wander') as unknown as SupportViewer;
    w.play(false);
    w.setTime(time);
  }, fixture.time || 0);
  for (const target of fixture.waypoints) {
    const samples = [];
    for (let i = 0; i < 100; i++) {
      const state = await page.evaluate(([x, z]) => {
        const w = Reflect.get(window, 'wander') as unknown as SupportViewer,
          p = w.camera.position;
        const dx = x - p.x,
          dz = z - p.z;
        w.camera.rotation.set(-0.18, Math.atan2(-dx, -dz), 0, 'YXZ');
        return {
          position: p.toArray(),
          distance: Math.hypot(dx, dz),
          blocked: w.walk.blockedAt(p.x, p.z),
          inside: w.walk.cellAt(p.x, p.z).inside,
        };
      }, target);
      samples.push(state);
      assert.equal(state.blocked, 0, 'Route must preserve solid collisions');
      assert.equal(state.inside, 1, 'Route must remain supported or inside the sampled hull');
      if (state.distance < 0.12) break;
      await page.keyboard.down('w');
      await page.waitForTimeout(100);
      await page.keyboard.up('w');
    }
    report.route.push({ target, samples });
    assert.ok(samples.at(-1)!.distance < 0.15, `Could not reach waypoint ${target}`);
  }
  const probes = await page.evaluate((input) => {
    const w = Reflect.get(window, 'wander') as unknown as SupportViewer;
    return input.map(({ x, z }) => {
      const cell = w.walk.cellAt(x, z);
      return {
        x,
        z,
        cell,
        placeable: w.walk.canStand(x, z),
        blocked:
          w.walk.blockedAt(x, z, Number.isFinite(cell.floor) ? cell.floor : w.walk.floor) > 0,
      };
    });
  }, fixture.probes);
  report.probes = probes;
  for (let i = 0; i < probes.length; i++) {
    assert.equal(probes[i].placeable, fixture.probes[i].placeable, `Map placement at probe ${i}`);
    assert.equal(probes[i].blocked, fixture.probes[i].blocked, `Solid collision at probe ${i}`);
  }
  assert.deepEqual(report.errors, []);
  console.log(
    `Walk support passed: ${fixture.waypoints.length} waypoints, ${probes.length} probes`,
  );
} catch (error) {
  report.failure = String(error);
  throw error;
} finally {
  await writeFile(path.join(out, 'report.json'), JSON.stringify(report, null, 2));
  await browser.close();
}
