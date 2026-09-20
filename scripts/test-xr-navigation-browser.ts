// Real stair collision data and rendered eye/body transforms with synthetic WebXR input.
// These regressions exercise navigation contracts; they are not physical-headset evidence.
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium, type Page } from 'playwright-core';
import { Quaternion, Vector3 } from 'three';
import type { WalkDiagnostics } from './viewer-types.ts';

type Point = { x: number; z: number; floor: number };
type Report = {
  mode: string;
  groundFloor: number;
  teleportTarget: number[] | null;
  teleportValid: boolean;
  body: { floorY: number; visible: boolean };
};
declare global {
  interface Window {
    __navigationAdvance: { deltas: number[][]; original: WalkDiagnostics['advance'] };
  }
}

function close(actual: number, expected: number, label: string) {
  assert.ok(Math.abs(actual - expected) < 0.0003, `${label}: ${actual} != ${expected}`);
}
function vectorClose(actual: number[], expected: number[], label: string) {
  actual.forEach((value, i) => close(value, expected[i], `${label}[${i}]`));
}
async function frames(page: Page, count = 8) {
  const start = await page.evaluate(() => window.__fakeXR.frames.length);
  await page.waitForFunction((end) => window.__fakeXR.frames.length >= end, start + count);
}
async function desktop(page: Page) {
  return page.evaluate(() => ({
    position: window.wander.camera.position.toArray(),
    rotation: window.wander.camera.quaternion.toArray(),
    floor: window.wander.walk!.floor,
  }));
}
async function state(page: Page) {
  return page.evaluate(() => {
    const w = window.wander;
    const rig = w.camera.parent!;
    const eyes = w.spark.renderer.xr.getCamera().cameras;
    const head = new w.THREE.Vector3()
      .setFromMatrixPosition(eyes[0].matrixWorld)
      .add(new w.THREE.Vector3().setFromMatrixPosition(eyes[1].matrixWorld))
      .multiplyScalar(0.5);
    const report = window.__xr as Report;
    return {
      head: head.toArray(),
      rig: rig.position.toArray(),
      rotation: rig.quaternion.toArray(),
      upm: w.upm,
      floor: report.groundFloor,
      desktopFloor: w.walk!.floor,
      target: report.teleportTarget,
      valid: report.teleportValid,
      mode: report.mode,
      bodyVisible: report.body.visible,
      bodyWorldFloor: report.body.floorY * w.upm + rig.position.y,
      soles: ['left', 'right'].map((side) => {
        const sole = rig.getObjectByName(`avatar-body-${side}-sole`)!;
        return new w.THREE.Box3().setFromObject(sole).min.y;
      }),
    };
  });
}
async function aim(page: Page, destination: Point, upward = false) {
  await page.evaluate(
    ({ destination, upward }) => {
      const w = window.wander;
      const rig = w.camera.parent!;
      // Place the synthetic controller above the measured cell and point straight down.
      // A vertical ray isolates support selection from neighboring stair risers.
      const origin = new w.THREE.Vector3(
        destination.x,
        Math.max(destination.floor, (window.__xr as Report).groundFloor) + 2 * w.upm,
        destination.z,
      );
      rig.worldToLocal(origin);
      const direction = new w.THREE.Vector3(0, upward ? 1 : -1, 0).applyQuaternion(
        rig.quaternion.clone().invert(),
      );
      const rotation = new w.THREE.Quaternion().setFromUnitVectors(
        new w.THREE.Vector3(0, 0, -1),
        direction,
      );
      const matrix = new w.THREE.Matrix4().compose(origin, rotation, new w.THREE.Vector3(1, 1, 1));
      const source = w.spark.renderer.xr.getSession()!.inputSources[0];
      (source.targetRaySpace as unknown as { _matrix: number[] })._matrix = matrix.toArray();
      window.__fakeXR.axes.left = [0, 0, 0, -1];
    },
    { destination, upward },
  );
  await frames(page);
}
async function release(page: Page) {
  await page.evaluate(() => {
    window.__fakeXR.axes.left = [0, 0, 0, 0];
  });
  await frames(page, 12);
}

const browser = await chromium.launch({ channel: 'chrome', headless: true });
const evidence: unknown[] = [];
try {
  for (const localFloor of [true, false]) {
    const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.addInitScript({ path: new URL('./fake-webxr.js', import.meta.url).pathname });
    await page.goto(
      `http://127.0.0.1:5399/fourd.html?demo=stairs2&walk=1&xr=1&fakexr=1&fakefloor=${localFloor ? 1 : 0}&xrwalkgain=1.5&fakew=512&fakeh=512&xradapt=0`,
      { waitUntil: 'load', timeout: 120000 },
    );
    await page.waitForFunction(() => window.wander?.ready, null, { timeout: 180000 });
    await page.evaluate(() => window.wander.play(false));
    const initialDesktop = await desktop(page);
    const supports = await page.evaluate(() => {
      const w = window.wander;
      const walk = w.walk! as WalkDiagnostics & { canStand(x: number, z: number): boolean };
      const navigation = w as typeof w & { pathPts: import('three').Vector3[]; pathR: number };
      const cell = walk.grid!.cell;
      const points: Point[] = [];
      for (let i = Math.floor(w.clampBox.min.x / cell); i * cell < w.clampBox.max.x; i++) {
        for (let j = Math.floor(w.clampBox.min.z / cell); j * cell < w.clampBox.max.z; j++) {
          const x = (i + 0.5) * cell;
          const z = (j + 0.5) * cell;
          const support = walk.cellAt(x, z, Infinity);
          if (!support.inside || !Number.isFinite(support.floor)) continue;
          if (walk.blockedAt(x, z, support.floor) || !walk.canStand(x, z)) continue;
          const point = new w.THREE.Vector3(x, support.floor, z);
          if (!w.clampBox.containsPoint(point)) continue;
          if (navigation.pathPts.length >= 2 && navigation.pathR > 0) {
            let distance = Infinity;
            for (let k = 1; k < navigation.pathPts.length; k++) {
              const line = new w.THREE.Line3(navigation.pathPts[k - 1], navigation.pathPts[k]);
              distance = Math.min(
                distance,
                line.closestPointToPoint(point, true, new w.THREE.Vector3()).distanceTo(point),
              );
            }
            if (distance > navigation.pathR) continue;
          }
          points.push({ x, z, floor: support.floor });
        }
      }
      points.sort((a, b) => a.floor - b.floor);
      const low = points[0];
      const high = points.at(-1)!;
      const middle = points.find(
        (point) => point.floor > low.floor + 0.1 * w.upm && point.floor < high.floor - 0.1 * w.upm,
      );
      return { low, high, middle, count: points.length };
    });
    assert.ok(
      supports.middle,
      'real stair data must provide lower, middle and higher safe supports',
    );
    assert.equal(
      await page.evaluate((point) => {
        const walk = window.wander.walk as WalkDiagnostics & {
          stand(x: number, z: number): boolean;
        };
        return walk.stand(point.x, point.z);
      }, supports.middle),
      true,
    );
    const saved = await desktop(page);
    assert.notDeepEqual(
      saved.position,
      initialDesktop.position,
      'entry starts from a moved desktop pose',
    );
    await page.click('#xrBtn');
    await frames(page);
    await page.evaluate(() => window.wander.play(false));
    const start = await state(page);
    assert.equal(start.mode, 'teleport', 'omitting xrmove must exercise default teleport');
    close(start.floor, saved.floor, 'session starts at the current desktop floor');
    const floorFeatures = await page.evaluate(() =>
      Array.from(window.wander.spark.renderer.xr.getSession()!.enabledFeatures ?? []),
    );
    assert.equal(floorFeatures.includes('local-floor'), localFloor);

    // Desktop keyboard movement/settling must not mutate its saved floor from tracked XR coordinates.
    await page.keyboard.down('w');
    await page.evaluate(() => {
      window.__fakeXR.head.x += 0.08;
      window.__fakeXR.head.z -= 0.06;
    });
    await frames(page, 18);
    await page.keyboard.up('w');
    await page.keyboard.press('KeyR');
    close(
      (await desktop(page)).floor,
      saved.floor,
      'desktop floor is frozen during XR tracking, WASD and desktop reset',
    );

    const teleports = [];
    for (const destination of [supports.high, supports.low]) {
      await aim(page, destination);
      const aimed = await state(page);
      assert.equal(aimed.valid, true, 'known clear measured support must be offered');
      assert.ok(aimed.target);
      vectorClose(
        aimed.target,
        [destination.x, destination.floor, destination.z],
        'landing marker',
      );
      const before = await state(page);
      await release(page);
      const landed = await state(page);
      close(landed.floor, destination.floor, 'teleport adopts destination support');
      close(landed.head[0], destination.x, 'teleport head x');
      close(landed.head[2], destination.z, 'teleport head z');
      close(
        landed.rig[1] - before.rig[1],
        destination.floor - before.floor,
        'rig follows support delta',
      );
      close(
        landed.head[1] - landed.rig[1],
        before.head[1] - before.rig[1],
        'tracked height is preserved',
      );
      assert.equal(landed.bodyVisible, true);
      close(landed.bodyWorldFloor, destination.floor, 'avatar floor follows landing');
      landed.soles.forEach((y) => close(y, destination.floor, 'avatar sole world floor'));
      close(landed.desktopFloor, saved.floor, 'teleport does not mutate saved desktop floor');
      await mkdir('.context/evidence/vr-ui-pr', { recursive: true });
      await page.screenshot({
        path: `.context/evidence/vr-ui-pr/teleport-${localFloor ? 'floor' : 'fallback'}-${teleports.length === 0 ? 'up' : 'down'}-simulated.png`,
      });
      teleports.push(landed);
    }
    assert.ok(teleports[0].floor > start.floor && teleports[1].floor < start.floor);

    // Both a failed ray direction and an unsupported/out-of-bounds destination clear old validity.
    for (const upward of [true, false]) {
      await aim(page, supports.high);
      assert.equal((await state(page)).valid, true);
      const invalid = upward
        ? supports.high
        : { ...supports.high, x: supports.high.x + 1000 * start.upm };
      await aim(page, invalid, upward);
      assert.equal(
        (await state(page)).valid,
        false,
        'invalid aim must clear previous valid target',
      );
      const before = await state(page);
      await release(page);
      const after = await state(page);
      vectorClose(after.rig, before.rig, 'invalid aim release cannot teleport to stale target');
      close(after.floor, before.floor, 'invalid aim keeps support');
    }

    // Deterministically reject artificial gain through the public collision API. Raw tracking must survive.
    await page.evaluate(() => {
      const walk = window.wander.walk!;
      window.__navigationAdvance = { original: walk.advance, deltas: [] };
      walk.advance = (from, delta) => {
        window.__navigationAdvance.deltas.push(delta.toArray());
        return from.clone();
      };
    });
    const gainBefore = await state(page);
    await page.evaluate(() => {
      window.__fakeXR.head.x += 0.08;
      window.__fakeXR.head.y -= 0.03;
      window.__fakeXR.head.z += 0.04;
    });
    await frames(page);
    const gainAfter = await state(page);
    const calls = await page.evaluate(() => window.__navigationAdvance.deltas);
    assert.ok(
      calls.some((delta) => Math.hypot(delta[0], delta[2]) > 0),
      'gain must use walk.advance',
    );
    vectorClose(gainAfter.rig, gainBefore.rig, 'rejected artificial gain cannot move rig');
    const rawStep = new Vector3(0.08, -0.03, 0.04)
      .multiplyScalar(start.upm)
      .applyQuaternion(new Quaternion().fromArray(gainBefore.rotation));
    vectorClose(
      gainAfter.head,
      rawStep.add(new Vector3().fromArray(gainBefore.head)).toArray(),
      'raw tracked step',
    );

    // Isolate the measured-region clamp from collision: permit all API movement, move the physical
    // head outside the boundary, then take an outward step. Only the extra gain must be suppressed.
    await page.evaluate(() => {
      const w = window.wander;
      w.walk!.advance = (from, delta) => from.clone().add(delta);
      const rig = w.camera.parent!;
      const desired = new w.THREE.Vector3(
        w.clampBox.max.x + 2 * w.clampBox.getSize(new w.THREE.Vector3()).x,
        0,
        rig.position.z,
      );
      rig.worldToLocal(desired);
      window.__fakeXR.head.x = desired.x;
      window.__fakeXR.head.z = desired.z;
      w.spark.renderer.xr.getReferenceSpace()!.dispatchEvent(new Event('reset'));
    });
    await frames(page);
    const boundaryBefore = await state(page);
    await page.evaluate(() => {
      const w = window.wander;
      const step = new w.THREE.Vector3(0.08, 0, 0).applyQuaternion(
        w.camera.parent!.quaternion.clone().invert(),
      );
      window.__fakeXR.head.x += step.x;
      window.__fakeXR.head.z += step.z;
    });
    await frames(page);
    const boundaryAfter = await state(page);
    vectorClose(
      boundaryAfter.rig,
      boundaryBefore.rig,
      'boundary suppresses outward artificial gain',
    );
    close(
      boundaryAfter.head[0] - boundaryBefore.head[0],
      0.08 * start.upm,
      'boundary preserves physical lean',
    );
    await page.evaluate(() => {
      window.wander.walk!.advance = window.__navigationAdvance.original;
    });

    await page.evaluate(() => window.wander.spark.renderer.xr.getSession()!.end());
    const restored = await desktop(page);
    vectorClose(restored.position, saved.position, 'exit restores moved desktop position');
    vectorClose(restored.rotation, saved.rotation, 'exit restores desktop rotation');
    close(restored.floor, saved.floor, 'exit restores desktop floor');
    await page.click('#xrBtn');
    await frames(page);
    const reentered = await state(page);
    close(reentered.head[0], saved.position[0], 're-entry anchors at saved desktop x');
    close(reentered.head[2], saved.position[2], 're-entry anchors at saved desktop z');
    close(reentered.floor, saved.floor, 're-entry uses saved desktop floor');
    await page.evaluate(() => window.wander.spark.renderer.xr.getSession()!.end());
    assert.deepEqual(errors, []);
    evidence.push({
      localFloor,
      supports,
      saved,
      start,
      teleports,
      gainBefore,
      gainAfter,
      boundaryBefore,
      boundaryAfter,
      restored,
      reentered,
    });
    console.log(`XR navigation passed: ${localFloor ? 'local-floor' : 'fallback reference space'}`);
    await page.close();
  }
  await mkdir('.context/evidence/vr-ui-pr', { recursive: true });
  await Bun.write(
    '.context/evidence/vr-ui-pr/xr-navigation.json',
    JSON.stringify(evidence, null, 2),
  );
  console.log(
    'XR navigation checks passed: stairs, desktop isolation/restoration, stale aim, gain collisions and clamp.',
  );
} finally {
  await browser.close();
}
