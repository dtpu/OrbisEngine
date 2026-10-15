// Retained replay regression: geometry must not rewind before its source video loops.
import assert from 'node:assert/strict';
import { chromium } from 'playwright-core';
import { mkdir, writeFile } from 'node:fs/promises';
import type * as THREE from 'three';
import type { ViewerDiagnostics } from './viewer-types.ts';

type TimingViewer = Omit<ViewerDiagnostics, 'people'> & {
  snap: boolean;
  people: {
    id: string;
    ts: number[];
    nF: number;
    loaded: number;
    frame: number;
    group: THREE.Group;
    pmesh: { packedSplats: { packedArray: Uint32Array } };
  }[];
};
const url = process.env.TIMING_TEST_URL || 'http://127.0.0.1:5399/fourd.html?demo=elevator';
const output = process.env.TIMING_TEST_OUTPUT || '.context/evidence/person-timing';
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const page = await browser.newPage({ viewport: { width: 1200, height: 750 } });
const errors: string[] = [];
page.on('pageerror', (error) => errors.push(error.message));
const checks: unknown[] = [];
try {
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.wander?.ready, null, { timeout: 120000 });
  const end = await page.evaluate(() => {
    const w = window.wander as TimingViewer;
    w.play(false);
    return { last: Math.max(...w.people.map((p) => p.ts.at(-1)!)), duration: w.dur };
  });
  assert.ok(end.last < end.duration, 'Fixture must include video after the last pose');
  for (const snap of [false, true]) {
    const states = [];
    for (const time of [
      0,
      end.last,
      (end.last + end.duration) / 2,
      end.duration - 1e-5,
      end.duration,
    ]) {
      states.push(
        await page.evaluate(
          async ({ time, snap }) => {
            const w = window.wander as TimingViewer;
            w.snap = snap;
            w.setTime(time);
            const people = await Promise.all(
              w.people.map(async (p) => {
                // Copy the actual rendered packed geometry; this includes all centres and rotations.
                const packed = new Uint32Array(p.pmesh.packedSplats.packedArray);
                const digest = await crypto.subtle.digest('SHA-256', packed);
                return {
                  id: p.id,
                  frame: p.frame,
                  last: p.nF - 1,
                  geometry: Array.from(new Uint8Array(digest), (v) =>
                    v.toString(16).padStart(2, '0'),
                  ).join(''),
                  position: p.group.position.toArray(),
                };
              }),
            );
            return { time: w.t, people };
          },
          { time, snap },
        ),
      );
    }
    checks.push({ snap, states });
    for (const state of states.slice(2, 4)) {
      assert.deepEqual(
        state.people,
        states[1].people,
        'The final pose or its placement rewound early',
      );
    }
    assert.deepEqual(states[4].people, states[0].people, 'Geometry did not follow the clock loop');
  }
  // Seeks and normal playback still advance the real source video.
  await page.evaluate(() => {
    const w = window.wander as TimingViewer;
    w.snap = false;
    w.setTime(0);
    w.play(true);
  });
  await page.waitForFunction(() => window.wander.video.currentTime > 0.2, null, { timeout: 15000 });
  await page.evaluate(() => window.wander.play(false));
  assert.deepEqual(errors, []);
  console.log(
    'Recorded people hold their final pose through the video tail, then loop with the source.',
  );
} finally {
  await writeFile(`${output}/report.json`, JSON.stringify({ url, errors, checks }, null, 2));
  await browser.close();
}
