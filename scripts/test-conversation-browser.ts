// Real scene assets and GPU rendering; conversation events are synthetic and make no provider calls.
import assert from 'node:assert/strict';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { chromium } from 'playwright-core';
import type * as THREE from 'three';
import type { SplatMesh } from '@sparkjsdev/spark';
import type { BottleScene, InteractionPerson } from '../src/interaction/bottle-scene';
import type { ConversationSplats } from '../src/interaction/conversation-splats';
import type { ConversationMotion } from '../src/interaction/conversation-motion';
import type { ViewerDiagnostics } from './viewer-types';

type Diagnostics = Omit<ViewerDiagnostics, 'people'> & {
  scene: THREE.Scene;
  people: Array<InteractionPerson & { pmesh: SplatMesh; loaded: number; nF: number }>;
  interaction: BottleScene;
};
type Internals = {
  anchor(person: InteractionPerson): THREE.Vector3;
  client: { options: { speaking(value: boolean): void } };
  animations: Map<string, { motion: ConversationMotion; splats: ConversationSplats }>;
};
type Probe = {
  head: THREE.Vector3;
  rotation: THREE.Quaternion;
  before: Uint8Array;
};
declare global {
  interface Window {
    conversationProbe: Probe;
  }
}

const personIndex = process.env.CONVERSATION_PERSON === '1' ? 1 : 0;
const output = resolve(`.context/evidence/conversation-animation/person-${personIndex}`);
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const errors: string[] = [];
try {
  const page = await browser.newPage({ viewport: { width: 1100, height: 900 } });
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error' && /shader|GL_INVALID|WebGLProgram/.test(message.text()))
      errors.push(message.text());
  });
  const build = process.env.VIEWER_BUILD_DIR;
  if (build) {
    const root = resolve(build);
    await page.route('http://127.0.0.1:5399/**', async (route) => {
      const path = new URL(route.request().url()).pathname;
      const file = resolve(root, `.${path}`);
      if (
        file.startsWith(`${root}/`) &&
        (path.endsWith('.html') || path.startsWith('/assets/')) &&
        (await stat(file).catch(() => null))?.isFile()
      ) {
        await route.fulfill({
          body: await readFile(file),
          contentType: file.endsWith('.js')
            ? 'text/javascript'
            : file.endsWith('.css')
              ? 'text/css'
              : file.endsWith('.html')
                ? 'text/html'
                : 'application/octet-stream',
        });
      } else await route.fallback();
    });
  }
  await page.route('**/api/bottle-agent/session', (route) => route.abort());
  await page.goto(
    'http://127.0.0.1:5399/fourd.html?demo=elevator&interact=1&walk=0&clamp=0&shadow=0',
    { timeout: 120000 },
  );
  await page.waitForFunction(
    () => window.wander?.ready && window.wander.video.readyState >= 3,
    null,
    { timeout: 180000 },
  );
  const initial = await page.evaluate((personIndex) => {
    const w = window.wander as Diagnostics;
    w.interaction.sessionStart();
    w.interaction.replay(false);
    w.setTime(0.5);
    const internals = w.interaction as unknown as Internals;
    const head = internals.anchor(w.people[personIndex]);
    const stature = (w.interaction as unknown as { host: { stature: number } }).host.stature;
    const visitor = head.clone().add(new w.THREE.Vector3(0, 0, 0.7 * stature));
    const rotation = new w.THREE.Quaternion();
    window.conversationProbe = { head: visitor, rotation, before: new Uint8Array() };
    w.interaction.frame(visitor, rotation, [], 0);
    const accepted = w.interaction.interrupt('speech', w.people[personIndex].id);
    // Settle the existing whole-body facing separately from the new head animation.
    for (let i = 0; i < 240; i++) w.interaction.frame(visitor, rotation, [], 1 / 60);
    w.camera.position.copy(head).add(new w.THREE.Vector3(0, -0.2 * stature, 1.35 * stature));
    w.camera.lookAt(head.clone().add(new w.THREE.Vector3(0, -0.35 * stature, 0)));
    w.camera.updateMatrixWorld(true);
    return { accepted, state: w.interaction.snapshot() };
  }, personIndex);
  assert.equal(initial.accepted, true);
  assert.equal(initial.state.playing, false);
  assert.equal(initial.state.animations[personIndex].pose?.mode, 'listening');
  assert.equal(initial.state.animations[1 - personIndex].pose, null);
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${output}/listening.png` });
  const source = await page.evaluate(() => {
    const video = window.wander.video;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d')!.drawImage(video, 0, 0);
    return canvas.toDataURL();
  });
  await writeFile(`${output}/source.png`, Buffer.from(source.split(',')[1], 'base64'));
  console.log('Real elevator loads and selected person listens with recording paused');

  const advance = async (speaking: boolean, frames: number) =>
    page.evaluate(
      ({ speaking, frames }) => {
        const w = window.wander as Diagnostics;
        (w.interaction as unknown as Internals).client.options.speaking(speaking);
        const probe = window.conversationProbe;
        for (let i = 0; i < frames; i++)
          w.interaction.frame(probe.head, probe.rotation, [], 1 / 60);
        return w.interaction.snapshot();
      },
      { speaking, frames },
    );
  const speaking = await advance(true, 45);
  assert.equal(speaking.animations[personIndex].pose?.mode, 'speaking');
  assert.equal(speaking.recordingTime, initial.state.recordingTime);
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${output}/speaking.png` });

  // Isolate the actual actor in the same renderer. No moving markers, HUD, source inset or
  // room can account for the pixel difference. The world-space lower-body region must stay put.
  const region = await page.evaluate((personIndex) => {
    const w = window.wander as Diagnostics;
    const p = w.people[personIndex];
    const root = p.group.parent!;
    for (const object of w.scene.children)
      object.visible = object === root || object === (w.spark as unknown as THREE.Object3D);
    for (const person of w.people) person.group.visible = person === p;
    const head = (w.interaction as unknown as Internals).anchor(p);
    const size = (w.interaction as unknown as { host: { stature: number } }).host.stature;
    const bottomOfChest = head
      .clone()
      .add(new w.THREE.Vector3(0, -0.48 * size, 0))
      .project(w.camera);
    return {
      lowerRows: Math.floor((bottomOfChest.y + 1) * 0.5 * w.spark.renderer.domElement.height),
    };
  }, personIndex);
  const pixels = async (save: boolean) =>
    page.evaluate(
      ({ save, lowerRows }) => {
        const w = window.wander as Diagnostics;
        w.scene.getObjectByName('agent-speaking')!.visible = false;
        w.scene.getObjectByName('bottle-interaction-visual')!.visible = false;
        w.spark.renderer.render(w.scene, w.camera);
        const gl = w.spark.renderer.getContext();
        const width = gl.drawingBufferWidth,
          height = gl.drawingBufferHeight;
        const rgba = new Uint8Array(width * height * 4);
        gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
        if (save) {
          window.conversationProbe.before = rgba;
          return null;
        }
        const before = window.conversationProbe.before;
        let changed = 0,
          exactChanged = 0,
          lowerChanged = 0,
          nonBlack = 0;
        for (let pixel = 0; pixel < width * height; pixel++) {
          const i = pixel * 4;
          if (rgba[i] + rgba[i + 1] + rgba[i + 2] > 30) nonBlack++;
          const difference = Math.max(
            ...[0, 1, 2].map((c) => Math.abs(rgba[i + c] - before[i + c])),
          );
          if (difference > 0) exactChanged++;
          if (difference > 4) {
            changed++;
            if (pixel < width * Math.max(0, lowerRows - 12)) lowerChanged++;
          }
        }
        return { changed, exactChanged, lowerChanged, nonBlack, width, height, lowerRows };
      },
      { save, lowerRows: region.lowerRows },
    );

  // Neutral baseline for exact restoration checks, including Gaussian orientation.
  await page.evaluate(() => {
    const w = window.wander as Diagnostics;
    const animations = (w.interaction as unknown as Internals).animations;
    for (const animation of animations.values()) animation.splats.reset();
  });
  await page.waitForTimeout(250);
  await pixels(true);
  await advance(true, 40);
  await page.waitForTimeout(250);
  const changed = await pixels(false);
  assert.ok(changed && changed.nonBlack > 1000, JSON.stringify(changed));
  assert.ok(
    changed.changed > 50,
    'Speaking must change the real rendered person, not only metadata',
  );
  assert.equal(changed.lowerChanged, 0, 'Feet and lower body must remain in their recorded pose');
  await page.screenshot({ path: `${output}/isolated-speaking.png` });
  await page.evaluate(() => {
    for (const animation of (
      (window.wander as Diagnostics).interaction as unknown as Internals
    ).animations.values())
      animation.splats.reset();
  });
  await page.waitForTimeout(250);
  const restored = await pixels(false);
  assert.equal(restored?.exactChanged, 0, 'Disabling the overlay must restore original pixels');
  const replay = await page.evaluate(() => {
    const w = window.wander as Diagnostics;
    w.interaction.replay(false);
    return {
      state: w.interaction.snapshot(),
      modifiers: w.people.map((p) => p.pmesh.objectModifiers?.length ?? 0),
    };
  });
  assert.deepEqual(replay.modifiers, [0, 0]);
  assert.ok(replay.state.animations.every((a) => a.pose === null));
  assert.deepEqual(errors, []);
  await writeFile(
    `${output}/results.json`,
    JSON.stringify({ initial, speaking, changed, restored, replay, errors }, null, 2),
  );
  console.log(
    'GPU head/chest motion changes pixels, preserves the lower body, and restores exactly',
  );
} finally {
  await browser.close();
}
