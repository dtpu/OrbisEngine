// Build first. Supply a scene fixture with a reviewed route and expected map probes.
// WALK_TEST_CONFIG=... WALK_TEST_DIST=dist bun scripts/test-walk-support-browser.ts
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { chromium, type Frame } from 'playwright-core';
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
  blockedApproach?: { target: [number, number]; maxTravel: number };
}
const config = process.env.WALK_TEST_CONFIG;
assert.ok(config, 'WALK_TEST_CONFIG must name a JSON scene fixture');
const fixture: Fixture = JSON.parse(await readFile(config, 'utf8'));
const url = new URL(fixture.url);
const out = path.resolve(process.env.WALK_TEST_OUT || '.context/evidence/walk-support-browser');
await mkdir(out, { recursive: true });
const report: {
  errors: string[];
  route: unknown[];
  probes?: unknown;
  approach?: unknown;
  failure?: string;
} = {
  errors: [],
  route: [],
};
const browser = await chromium.launch({
  channel: 'chrome',
  headless: true,
  args: ['--use-angle=metal', '--ignore-gpu-blocklist'],
});
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  page.on('pageerror', (error) => report.errors.push(error.message));
  if (process.env.WALK_TEST_DIST) {
    const dist = path.resolve(process.env.WALK_TEST_DIST);
    await page.route(`${url.origin}/**/*.html?*`, (route) =>
      route.fulfill({
        path: path.join(dist, path.basename(new URL(route.request().url()).pathname)),
        contentType: 'text/html',
      }),
    );
    await page.route(`${url.origin}/assets/*`, (route) =>
      route.fulfill({
        path: path.join(dist, 'assets', path.basename(new URL(route.request().url()).pathname)),
      }),
    );
  }
  await page.goto(url.href, { timeout: 120000 });
  const iframe = await page.$('#frame');
  const viewer: Frame = iframe ? (await iframe.contentFrame())! : page.mainFrame();
  await viewer.waitForFunction(() => Reflect.get(window, 'wander')?.ready, null, {
    timeout: 180000,
  });
  await viewer.evaluate((time) => {
    const w = Reflect.get(window, 'wander') as unknown as SupportViewer;
    w.play(false);
    w.setTime(time);
  }, fixture.time || 0);
  await viewer.evaluate(() => window.focus());
  for (const [index, target] of fixture.waypoints.entries()) {
    const samples = [];
    for (let i = 0; i < 100; i++) {
      const state = await viewer.evaluate(([x, z]) => {
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
    await page.screenshot({ path: path.join(out, `waypoint-${index}.png`) });
    assert.ok(samples.at(-1)!.distance < 0.15, `Could not reach waypoint ${target}`);
  }
  if (fixture.blockedApproach) {
    const start = await viewer.evaluate(([x, z]) => {
      const w = Reflect.get(window, 'wander') as unknown as SupportViewer;
      w.camera.lookAt(x, w.camera.position.y, z);
      return w.camera.position.toArray();
    }, fixture.blockedApproach.target);
    await page.keyboard.down('w');
    await page.waitForTimeout(2000);
    await page.keyboard.up('w');
    const approach = await viewer.evaluate((start) => {
      const w = Reflect.get(window, 'wander') as unknown as SupportViewer;
      const p = w.camera.position;
      return {
        start,
        end: p.toArray(),
        travel: Math.hypot(p.x - start[0], p.z - start[2]),
        blocked: w.walk.blockedAt(p.x, p.z),
      };
    }, start);
    report.approach = approach;
    await page.screenshot({ path: path.join(out, 'counter-stop.png') });
    assert.ok(approach.travel > 0.01, 'The counter approach must exercise keyboard movement');
    assert.ok(
      approach.travel < fixture.blockedApproach.maxTravel,
      'The counter must stop forward movement',
    );
    assert.equal(approach.blocked, 0, 'The walker must stop outside occupied counter cells');
  }
  const probes = await viewer.evaluate((input) => {
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
