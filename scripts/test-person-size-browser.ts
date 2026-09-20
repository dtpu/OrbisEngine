// Real viewer authoring controls and source-video transport; no headset claims.
import assert from 'node:assert/strict';
import { chromium, type Page } from 'playwright-core';
import type * as THREE from 'three';
import type { ViewerDiagnostics } from './viewer-types.ts';

type Transform = { pos: number[]; rot: number[]; scale: number };
type Person = {
  loaded: number;
  nF: number;
  group: THREE.Group;
  registeredScale: THREE.Vector3;
  basePos: THREE.Vector3;
  bodyH: number;
  ts: number[];
  dur: number;
  feet: number[][];
  measureFeet(index: number): void;
  standingHeight(): number;
};
type AuthorViewer = Omit<ViewerDiagnostics, 'people'> & {
  primary: Person;
  people: Person[];
  getTransform(): Transform;
  setTransform(value: Partial<Transform>): void;
  url(): string;
  unlockAudio(): Promise<void>;
};
type TestWindow = Window & { __personSizeAudio?: AudioContext };
// evaluate serializes its callback, so this function deliberately has no module dependencies.
async function snapshot(page: Page) {
  return page.evaluate(() => {
    const w = window.wander as AuthorViewer;
    const p = w.primary;
    let i = 0;
    while (i + 1 < p.ts.length && p.ts[i + 1] <= w.t) i++;
    const j = (i + 1) % p.ts.length;
    const next = i + 1 < p.ts.length ? p.ts[i + 1] : p.dur;
    const u = next > p.ts[i] ? Math.min(1, (w.t - p.ts[i]) / (next - p.ts[i])) : 0;
    p.measureFeet(i);
    p.measureFeet(j);
    const a = p.feet[i];
    const b = p.feet[j];
    const anchor = new w.THREE.Vector3(
      a[1] + (b[1] - a[1]) * u,
      a[0] + (b[0] - a[0]) * u,
      a[2] + (b[2] - a[2]) * u,
    );
    p.group.updateMatrixWorld(true);
    const foot = p.group.localToWorld(anchor.clone());
    const head = p.group.localToWorld(anchor.clone().add(new w.THREE.Vector3(0, p.bodyH, 0)));
    const listener = (window as TestWindow).__personSizeAudio?.listener;
    return {
      t: w.t,
      transform: w.getTransform(),
      url: w.url(),
      scales: w.people.map((person) => ({
        visual: person.group.scale.toArray(),
        registered: person.registeredScale.toArray(),
      })),
      height: p.standingHeight(),
      unit: p.bodyH * p.registeredScale.x,
      camera: w.camera.position.toArray(),
      listener: listener && [
        listener.positionX.value,
        listener.positionY.value,
        listener.positionZ.value,
      ],
      foot: foot.toArray(),
      head: head.toArray(),
    };
  });
}
function near(a: number[], b: number[], tolerance = 1e-7) {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i] - b[i]) < tolerance, `${a} != ${b}`);
}
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const results = new Map<number, Awaited<ReturnType<typeof snapshot>>[]>();
try {
  for (const factor of [1, 0.9]) {
    const page = await browser.newPage({ viewport: { width: 1000, height: 700 } });
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.addInitScript(() => {
      const NativeAudioContext = window.AudioContext;
      window.AudioContext = class extends NativeAudioContext {
        constructor(options?: AudioContextOptions) {
          super(options);
          (window as TestWindow).__personSizeAudio = this;
        }
      };
    });
    await page.goto(
      `http://127.0.0.1:5399/fourd.html?demo=stairs2&walk=0&pause=1&personsize=${factor}`,
      { waitUntil: 'load', timeout: 120000 },
    );
    await page.waitForFunction(
      () => window.wander?.ready && window.wander.video.readyState >= 3,
      null,
      {
        timeout: 180000,
      },
    );
    await page.evaluate(async () => {
      const w = window.wander as AuthorViewer;
      w.play(false);
      w.setTime(0.75);
      await w.unlockAudio();
    });
    const home = await snapshot(page);
    const stages = [home];
    const edited: Transform = {
      pos: home.transform.pos.map((v, i) => v + [0.25, 0.1, -0.2][i]),
      rot: home.transform.rot.map((v, i) => v + (i === 1 ? 13 : 0)),
      scale: home.transform.scale * 1.3,
    };
    await page.evaluate(
      (transform) => (window.wander as AuthorViewer).setTransform(transform),
      edited,
    );
    const checkScale = async (expected: number) => {
      const state = await snapshot(page);
      for (const scale of state.scales) {
        near(scale.registered, [expected, expected, expected]);
        near(scale.visual, [expected * factor, expected * factor, expected * factor]);
      }
      near(state.transform.pos, edited.pos);
      near(state.transform.rot, edited.rot);
      assert.ok(Math.abs(state.transform.scale - expected) < 1e-12);
      assert.equal(new URL(state.url).searchParams.get('scale'), expected.toFixed(4));
      return state;
    };
    stages.push(await checkScale(edited.scale));
    // Real input events must update a paused body immediately and survive its next animation sample.
    await page.keyboard.press('Equal');
    stages.push(await checkScale(edited.scale * 1.05));
    await page.evaluate(() => window.wander.setTime(1.25));
    stages.push(await checkScale(edited.scale * 1.05));
    await page.keyboard.press('Minus');
    stages.push(await checkScale(edited.scale));
    for (const time of [0.25, 1.75, 0.75]) {
      await page.evaluate((t) => window.wander.setTime(t), time);
      stages.push(await checkScale(edited.scale));
    }
    // A registration edit changes the canonical audio distance unit even at personsize=1.
    await page.waitForFunction(() => {
      const w = window.wander as AuthorViewer;
      const listener = (window as TestWindow).__personSizeAudio?.listener;
      const unit = w.primary.bodyH * w.primary.registeredScale.x;
      return listener && Math.abs(listener.positionZ.value - w.camera.position.z / unit) < 1e-5;
    });
    const audio = await snapshot(page);
    assert.ok(audio.listener);
    near(
      audio.listener,
      audio.camera.map((v) => v / audio.unit),
      1e-5,
    );
    await page.evaluate(() => window.wander.play(true));
    await page.waitForFunction(() => window.wander.t > 1 && !window.wander.video.paused);
    await page.evaluate(() => window.wander.play(false));
    await checkScale(edited.scale);
    await page.evaluate(() => window.wander.setTime(1.25));
    await page.keyboard.press('KeyR');
    const reset = await snapshot(page);
    near(reset.transform.pos, home.transform.pos);
    near(reset.transform.rot, home.transform.rot);
    assert.equal(reset.transform.scale, home.transform.scale);
    stages.push(reset);
    await page.evaluate(() => window.wander.setTime(0.75));
    const resetSeek = await snapshot(page);
    near(resetSeek.foot, home.foot);
    near(resetSeek.head, home.head);
    assert.equal(resetSeek.height, home.height);
    stages.push(resetSeek);
    results.set(factor, stages);
    assert.deepEqual(errors, []);
    console.log(
      `personsize=${factor}: API, scale keys, paused edits, seeks, playback, audio ruler and reset passed`,
    );
    await page.close();
  }
  const full = results.get(1)!;
  const small = results.get(0.9)!;
  for (let i = 0; i < full.length; i++) {
    assert.equal(full[i].t, small[i].t);
    assert.deepEqual(full[i].transform, small[i].transform);
    near(small[i].foot, full[i].foot);
    near(
      small[i].head,
      full[i].foot.map((v, axis) => v + 0.9 * (full[i].head[axis] - v)),
    );
    assert.equal(full[i].height, small[i].height);
    assert.equal(full[i].unit, small[i].unit);
  }
  console.log(
    'personsize=0.9/1: identical moving foot anchors and canonical rulers; body scales exactly 0.9',
  );
} finally {
  await browser.close();
}
