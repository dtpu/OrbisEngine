// Actual stereo renderer and scene loads driven by synthetic XR. This is not headset evidence.
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright-core';
import type * as THREE from 'three';
import type { ViewerDiagnostics } from './viewer-types';

type Sidebar = {
  open: boolean;
  selected: string | null;
  busy: boolean;
  error: string;
  focus: number;
  hover: number;
  scroll: number;
  clipCount: number;
  position: number[];
  quaternion: number[];
};
type Report = { sidebar: Sidebar; hands: { visibleCount: number } };
declare global {
  interface Window {
    __sidebarTest: {
      renderer: THREE.WebGLRenderer;
      session: XRSession;
      prior: ViewerDiagnostics;
      retiredScene?: THREE.Object3D;
    };
  }
}
const base = process.env.XR_SIDEBAR_URL || 'http://127.0.0.1:5400';
const out = '.context/evidence/xr-scene-sidebar';
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const records: Record<string, unknown> = {};
try {
  await mkdir(out, { recursive: true });
  const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
  page.setDefaultTimeout(30000);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const pendingRequests = new Set<string>();
  const assetRequests: string[] = [];
  page.on('request', (request) => {
    pendingRequests.add(request.url());
    if (/\.(ply|spz|bin|json)(\?|$)/.test(request.url())) assetRequests.push(request.url());
  });
  page.on('requestfinished', (request) => pendingRequests.delete(request.url()));
  page.on('requestfailed', (request) => {
    pendingRequests.delete(request.url());
    if (request.failure()?.errorText !== 'net::ERR_ABORTED')
      console.log('XR request failed', request.url(), request.failure()?.errorText);
  });
  page.on('console', (message) => {
    if (message.type() === 'error') console.log('XR console error', message.text());
  });
  await page.addInitScript({ path: new URL('./fake-webxr.js', import.meta.url).pathname });
  await page.goto(
    `${base}/fourd.html?demo=elevator&xr=1&fakexr=1&xrmove=smooth&xrwalkgain=1.5&fakew=384&fakeh=384&xradapt=0&xrview=0`,
    { waitUntil: 'load', timeout: 120000 },
  );
  await page.waitForFunction(() => window.wander?.ready, null, { timeout: 180000 });
  console.log('XR sidebar: initial scene ready');
  await page.click('#xrBtn');
  await page.waitForFunction(
    () => window.wander.spark.renderer.xr.isPresenting && window.wander.playing,
  );
  await page.evaluate(() => {
    const renderer = window.wander.spark.renderer;
    window.__sidebarTest = { renderer, session: renderer.xr.getSession()!, prior: window.wander };
  });
  async function frames(count = 4) {
    const before = await page.evaluate(() => window.__fakeXR.frames.length);
    await page.waitForFunction(
      ({ before, count }) => window.__fakeXR.frames.length >= before + count,
      { before, count },
    );
  }
  let audioCheck = 0;
  async function assertAudio(muted: boolean) {
    await page.waitForFunction((muted) => {
      const audio = window.wander.audioState;
      return (
        audio.unlocked &&
        audio.context === 'running' &&
        !audio.loading &&
        audio.muted === muted &&
        (muted ? audio.activeSources === 0 : audio.activeSources > 0)
      );
    }, muted);
    records[`${audioCheck++}-audio`] = await page.evaluate(() => ({
      demo: window.wander.demo,
      audio: window.wander.audioState,
    }));
  }
  await assertAudio(false);
  async function button(index: number) {
    await page.evaluate((index) => {
      window.__fakeXR.buttons.right[index] = 1;
    }, index);
    await frames(2);
    await page.evaluate((index) => {
      window.__fakeXR.buttons.right[index] = 0;
    }, index);
    await frames(2);
  }
  async function state() {
    return page.evaluate(() => {
      const w = window.wander;
      const rig = w.camera.parent!;
      const eyes = w.spark.renderer.xr.getCamera().cameras;
      return {
        demo: w.demo,
        playing: w.playing,
        upm: w.upm,
        rig: rig.position.toArray(),
        rotation: rig.quaternion.toArray(),
        head: eyes
          .map((eye) => new w.THREE.Vector3().setFromMatrixPosition(eye.matrixWorld))
          .reduce((sum, value) => sum.add(value), new w.THREE.Vector3())
          .multiplyScalar(1 / eyes.length)
          .toArray(),
        sidebar: structuredClone((window.__xr as Report).sidebar),
      };
    });
  }
  async function focus(index: number) {
    for (let attempt = 0; attempt < 12; attempt++) {
      const current = (await state()).sidebar.focus;
      if (current === index) return;
      await page.evaluate(
        (direction) => {
          window.__fakeXR.axes.left = [0, 0, 0, direction];
        },
        Math.sign(index - current),
      );
      await page.waitForFunction(
        (current) => (window.__xr as Report).sidebar.focus !== current,
        current,
      );
      await page.evaluate(() => {
        window.__fakeXR.axes.left = [0, 0, 0, 0];
      });
      await frames(2);
    }
    assert.fail(`Could not focus sidebar row ${index}`);
  }
  async function assertRuntime(demo: string) {
    const immediate = await page.evaluate(() => ({
      current: window.wander.demo,
      pending: (window as unknown as { __loadingScene?: { id: string } }).__loadingScene?.id,
      sidebar: structuredClone((window.__xr as Report).sidebar),
      roots: Array.from(document.querySelectorAll('.scene-root')).map((root) => ({
        hidden: (root as HTMLElement).hidden,
        status: root.querySelector('#st')?.textContent,
      })),
    }));
    console.log('XR switch immediate', JSON.stringify({ expected: demo, ...immediate }));
    assert.ok(
      immediate.current === demo || immediate.pending === demo,
      `Trigger selected an unexpected scene: ${JSON.stringify(immediate)}`,
    );

    const diagnosticTimer = setTimeout(() => {
      void page
        .evaluate(() => ({
          current: window.wander.demo,
          sidebar: structuredClone((window.__xr as Report).sidebar),
          roots: Array.from(document.querySelectorAll('.scene-root')).map((root) => ({
            hidden: (root as HTMLElement).hidden,
            status: root.querySelector('#st')?.textContent,
          })),
        }))
        .then((result) =>
          console.log(
            'XR switch after 10s',
            JSON.stringify({
              expected: demo,
              ...result,
              pendingRequests: [...pendingRequests],
              errors,
            }),
          ),
        )
        .catch(() => {});
    }, 10000);
    try {
      await page.waitForFunction(
        (demo) => window.wander.demo === demo && window.wander.ready,
        demo,
        { timeout: 180000 },
      );
    } catch (error) {
      const diagnostics = await page.evaluate(() => {
        const pending = (
          window as unknown as {
            __loadingScene?: {
              id: string;
              api?: {
                ready: boolean;
                people: { loaded: number; nF: number; loadError?: unknown }[];
              };
            };
          }
        ).__loadingScene;
        return {
          current: window.wander.demo,
          pending: pending
            ? {
                id: pending.id,
                ready: pending.api?.ready,
                people: pending.api?.people.map((person) => ({
                  loaded: person.loaded,
                  nF: person.nF,
                  loadError: String(person.loadError || ''),
                })),
              }
            : null,
          roots: Array.from(document.querySelectorAll('.scene-root')).map((root) => ({
            hidden: (root as HTMLElement).hidden,
            status: root.querySelector('#st')?.textContent,
          })),
          xr: window.__xr,
        };
      });
      console.error('XR scene switch stalled', JSON.stringify({ demo, diagnostics, errors }));
      await writeFile(
        `${out}/failure-diagnostics.json`,
        JSON.stringify({ demo, diagnostics, errors }, null, 2),
      );
      await page.screenshot({ path: `${out}/switch-stalled.png` });
      throw error;
    } finally {
      clearTimeout(diagnosticTimer);
    }
    await frames(6);
    const result = await page.evaluate(() => {
      const w = window.wander;
      const renderer = w.spark.renderer;
      const scene = w.camera.parent!.parent!;
      let rigs = 0;
      let menus = 0;
      scene.traverse((node) => {
        if (node.name === 'xr-rig') rigs++;
        if (node.name === 'xr-scene-sidebar') menus++;
      });
      return {
        sameRenderer: renderer === window.__sidebarTest.renderer,
        sameSession: renderer.xr.getSession() === window.__sidebarTest.session,
        presenting: renderer.xr.isPresenting,
        rigs,
        menus,
        audioControls: document.querySelectorAll('#audio-controls').length,
        videos: document.querySelectorAll('#pip').length,
        buttons: document.querySelectorAll('#xrBtn').length,
        hands: (window.__xr as Report).hands.visibleCount,
        selected: (window.__xr as Report).sidebar.selected,
      };
    });
    assert.deepEqual(result, {
      sameRenderer: true,
      sameSession: true,
      presenting: true,
      rigs: 1,
      menus: 1,
      audioControls: 1,
      videos: 1,
      buttons: 1,
      hands: 2,
      selected: demo,
    });
    records[`${demo}-runtime`] = result;
    console.log(`XR sidebar: ${demo} retains session and resources`);
  }

  await frames();
  // Aim forward at chest height, as with a controller raised to browse the sidebar.
  await page.evaluate(() => {
    const session = window.wander.spark.renderer.xr.getSession()!;
    const source = Array.from(session.inputSources).find(
      (source) => source.handedness === 'right',
    )!;
    (source.targetRaySpace as XRSpace & { _matrix: Float32Array })._matrix = new Float32Array([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.2, 1.4, -0.2, 1,
    ]);
  });
  await frames();
  await button(4);
  assert.equal((await state()).sidebar.open, false, 'A selects clips; it does not open the menu');
  await button(5);
  const opened = await state();
  assert.equal(opened.sidebar.open, true);
  assert.equal(opened.playing, false);
  assert.equal(opened.sidebar.selected, 'elevator');
  assert.equal(opened.sidebar.clipCount, 4);
  const aim = await page.evaluate(() => {
    const w = window.wander;
    const controller = w.spark.renderer.xr.getController(1);
    const origin = controller.getWorldPosition(new w.THREE.Vector3());
    const direction = new w.THREE.Vector3(0, 0, -1).applyQuaternion(
      controller.getWorldQuaternion(new w.THREE.Quaternion()),
    );
    const panel = w.camera.parent!.parent!.getObjectByName('xr-scene-sidebar')!;
    const hit = new w.THREE.Raycaster(origin, direction).intersectObject(panel)[0];
    return hit?.uv?.toArray();
  });
  assert.ok(aim && Math.abs(aim[0] - 0.5) < 1e-5 && Math.abs(aim[1] - 0.5) < 1e-5);
  records.openingAim = aim;
  await page.screenshot({ path: `${out}/menu-open.png` });
  await page.evaluate(() => {
    window.__fakeXR.head.x += 0.15;
    window.__fakeXR.head.y -= 0.12;
    window.__fakeXR.head.yaw += 0.2;
    window.__fakeXR.axes.left = [0, 0, 1, 1];
    window.__fakeXR.axes.right = [0, 0, 1, 0];
  });
  await frames(12);
  await page.evaluate(() => {
    window.__fakeXR.axes.left = window.__fakeXR.axes.right = [0, 0, 0, 0];
  });
  const tracked = await state();
  assert.deepEqual(tracked.rig, opened.rig, 'Menu blocks walking and extra physical gain');
  assert.deepEqual(tracked.rotation, opened.rotation, 'Menu blocks artificial turning');
  assert.ok(
    Math.hypot(...tracked.head.map((value, i) => value - opened.head[i])) > opened.upm * 0.1,
    'Physical head tracking continues',
  );
  assert.deepEqual(
    tracked.sidebar.position,
    opened.sidebar.position,
    'Panel remains anchored while head moves',
  );
  assert.deepEqual(tracked.sidebar.quaternion, opened.sidebar.quaternion);
  await focus(0);
  const browsed = await state();
  assert.equal(browsed.sidebar.selected, 'elevator', 'Focus does not change loaded selection');
  await frames(10);
  assert.equal((await state()).sidebar.focus, 0, 'Focus stays stable after joystick release');
  records.menu = { opened, tracked, browsed };
  await page.screenshot({ path: `${out}/menu-browsed.png` });

  console.log('XR sidebar: movement and browsing passed');
  await focus(2);
  await button(4);
  await assertRuntime('plaza');
  await assertAudio(false);
  const parked = await page.evaluate(() => {
    const previous = window.__sidebarTest.prior;
    window.__sidebarTest.retiredScene = window.wander.camera.parent!.parent!;
    return {
      paused: previous.video.paused,
      attached: previous.video.isConnected,
      sources: previous.audioState.activeSources,
      cache: (window as unknown as { __sceneCache: unknown }).__sceneCache,
    };
  });
  assert.equal(parked.paused, true, 'The cached clip must not keep playing');
  assert.equal(parked.attached, false, 'Cached HUD and media are detached');
  assert.equal(parked.sources, 0, 'Cached scene must not keep audio voices alive');
  records.parked = parked;
  await page.screenshot({ path: `${out}/plaza-switched.png` });
  await button(5);
  await focus(3);
  const requestsBeforeReturn = assetRequests.length;
  await button(0);
  await assertRuntime('elevator');
  await assertAudio(false);
  assert.deepEqual(
    assetRequests.slice(requestsBeforeReturn),
    [],
    'Returning to the cached scene must not fetch its world, people or manifests again',
  );
  const cachedReturn = await page.evaluate(() => ({
    sameRuntime: window.wander === window.__sidebarTest.prior,
    cache: (
      window as unknown as {
        __sceneCache: {
          retained: string[];
          hits: number;
          lastLoad: { id: string; reused: boolean; milliseconds: number };
        };
      }
    ).__sceneCache,
  }));
  assert.equal(cachedReturn.sameRuntime, true, 'Reuse the prepared scene, not just cached files');
  assert.equal(cachedReturn.cache.hits, 1);
  assert.equal(cachedReturn.cache.lastLoad.reused, true);
  assert.deepEqual(cachedReturn.cache.retained, ['plaza']);
  records.cachedReturn = cachedReturn;
  console.log('XR cached return', JSON.stringify(cachedReturn.cache));
  await page.evaluate(() => {
    (window.wander as ViewerDiagnostics & { setMuted(value: boolean): void }).setMuted(true);
  });
  for (const [demo, row] of [
    ['plaza', 2],
    ['elevator', 3],
  ] as const) {
    await button(5);
    await focus(row);
    const before = assetRequests.length;
    await button(4);
    await assertRuntime(demo);
    await assertAudio(true);
    assert.deepEqual(
      assetRequests.slice(before),
      [],
      'Repeated cache swaps make no asset requests',
    );
  }

  // Hold a required manifest indefinitely: a second controller choice must not wait for it.
  const gymManifest = '**/reviews/gym-repair/person-contact-candidate/sequence.json*';
  let releaseStall!: () => void;
  let sawStall!: () => void;
  const stalled = new Promise<void>((resolve) => (sawStall = resolve));
  const released = new Promise<void>((resolve) => (releaseStall = resolve));
  await page.route(gymManifest, async (route) => {
    sawStall();
    await released;
    await route.continue().catch(() => {}); // Cancellation can close the underlying request.
  });
  await button(5);
  await focus(1);
  await button(4);
  await stalled;
  assert.equal((await state()).sidebar.busy, true);
  await focus(2);
  await button(4);
  await assertRuntime('plaza');
  assert.equal((await state()).sidebar.error, '');
  releaseStall();
  await page.unroute(gymManifest);
  await frames(8);
  assert.equal((await state()).demo, 'plaza', 'A late cancelled load cannot replace the choice');
  records.cancelledLoad = await state();
  await button(5);
  await focus(3);
  await button(4);
  await assertRuntime('elevator');

  // Fail a required manifest to exercise rollback instead of a cache hit.
  await page.route(gymManifest, (route) =>
    route.fulfill({
      status: 500,
      contentType: 'application/json',
      body: '{"error":"Injected sidebar retry failure"}',
    }),
  );
  await page.evaluate(() => {
    window.__sidebarTest.prior = window.wander;
  });
  await button(5);
  await focus(1);
  await button(4);
  await page.waitForFunction(
    () => {
      const sidebar = (window.__xr as Report).sidebar;
      return !!sidebar.error && !sidebar.busy;
    },
    null,
    { timeout: 180000 },
  );
  assert.equal(
    await page.evaluate(() => window.wander === window.__sidebarTest.prior),
    true,
    'Failed scene keeps prior runtime',
  );
  const failed = await state();
  assert.equal(
    await page.evaluate(() => window.__sidebarTest.retiredScene!.children.length),
    0,
    'A third scene disposes the old spare before loading, even if the new load fails',
  );
  assert.equal(failed.sidebar.open, true);
  assert.equal(failed.playing, false);
  assert.ok(failed.sidebar.error.length > 0);
  await assertRuntime('elevator');
  records.failure = failed;
  await page.screenshot({ path: `${out}/menu-load-error.png` });
  await page.unroute(gymManifest);
  // Retry the same focused row with A and prove the error did not strand the session.
  await button(4);
  await assertRuntime('gym-accepted');
  await assertAudio(true);
  await page.evaluate(() => {
    (window.wander as ViewerDiagnostics & { setMuted(value: boolean): void }).setMuted(false);
  });
  await assertAudio(false);

  // Throw after the old XR binding is disposed. The reactivated binding must show the error.
  await page.evaluate(() => {
    const renderer = window.wander.spark.renderer;
    const setAnimationLoop = renderer.setAnimationLoop;
    renderer.setAnimationLoop = function (callback) {
      if (callback) {
        renderer.setAnimationLoop = setAnimationLoop;
        throw new Error('Injected activation failure');
      }
      return setAnimationLoop.call(this, callback);
    };
    window.__sidebarTest.prior = window.wander;
  });
  await button(5);
  await focus(3);
  await button(4);
  await page.waitForFunction(() => {
    const sidebar = (window.__xr as Report).sidebar;
    return sidebar.open && sidebar.error.includes('Injected activation failure') && !sidebar.busy;
  });
  assert.equal(await page.evaluate(() => window.wander === window.__sidebarTest.prior), true);
  assert.equal((await state()).playing, false, 'Restored error sidebar pauses playback');
  await assertRuntime('gym-accepted');
  records.activationFailure = await state();
  await page.screenshot({ path: `${out}/menu-activation-error.png` });
  await button(5);
  assert.equal((await state()).playing, true, 'Closing the error restores prior playback intent');
  await assertAudio(false);
  assert.deepEqual(errors, []);
  await writeFile(`${out}/checks.json`, JSON.stringify(records, null, 2));
  console.log(
    'XR sidebar: tracking, controls, cache/session preservation, stalled-load replacement, failure/retry and activation rollback passed.',
  );
} finally {
  await browser.close();
}
