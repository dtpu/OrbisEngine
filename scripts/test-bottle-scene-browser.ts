// Exercises the real viewer with recorded assets and synthetic XR input, not headset evidence.
import assert from 'node:assert/strict';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createServer } from 'node:http';
import { chromium } from 'playwright-core';
import type * as THREE from 'three';
import type { BottleScene, InteractionPerson } from '../src/interaction/bottle-scene';
import type { ViewerDiagnostics } from './viewer-types';
import { createBottleAgentMiddleware } from '../server/bottle-agent';

type SceneDiagnostics = Omit<ViewerDiagnostics, 'people'> & {
  interaction: BottleScene;
  scene: THREE.Scene;
  people: Array<InteractionPerson & { loaded: number; nF: number }>;
};
type SyntheticSession = XRSession & {
  inputSources: Array<XRInputSource & { targetRaySpace: XRSpace & { _matrix: Float32Array } }>;
};
type VoiceProbeWindow = Window & { __voiceChecks: Array<Record<string, unknown>> };
const output = resolve('.context/evidence/bottle-agent');
await mkdir(output, { recursive: true });
const liveVoice = process.env.WANDER_TEST_LIVE_VOICE === '1';
if (liveVoice && !process.env.OPENAI_API_KEY) throw new Error('Live voice requires a server key.');
const providerChecks: Array<{ status: number; code?: string; param?: string }> = [];
// The default empty environment proves Enter VR remains usable when voice setup cannot mint a
// secret. The metered provider call exists only behind WANDER_TEST_LIVE_VOICE=1.
const middleware = createBottleAgentMiddleware(liveVoice ? process.env : {}, {
  fetch: async (url, init) => {
    const response = await fetch(url, init);
    const error = !response.ok
      ? (
          await response
            .clone()
            .json()
            .catch(() => ({}))
        ).error
      : null;
    const safe = (value: unknown) =>
      typeof value === 'string' && /^[a-zA-Z0-9_.-]{1,100}$/.test(value) ? value : undefined;
    providerChecks.push({
      status: response.status,
      code: safe(error?.code),
      param: safe(error?.param),
    });
    return response;
  },
});
const backend = createServer(
  (req, res) =>
    void middleware(req, res, () => {
      res.writeHead(404);
      res.end();
    }),
);
await new Promise<void>((resolve) => backend.listen(0, '127.0.0.1', resolve));
const address = backend.address();
if (!address || typeof address === 'string') throw new Error('No middleware test port');
const backendOrigin = `http://127.0.0.1:${address.port}`;
let voiceSessionRequests = 0;
const browser = await chromium.launch({
  channel: 'chrome',
  headless: true,
  args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
});
const errors: string[] = [];
try {
  const page = await browser.newPage({ viewport: { width: 1100, height: 760 } });
  page.on('pageerror', (error) => errors.push(error.message));
  if (liveVoice) {
    await page.addInitScript(() => {
      const probe = window as unknown as VoiceProbeWindow;
      probe.__voiceChecks = [];
      const Original = window.RTCPeerConnection;
      window.RTCPeerConnection = class extends Original {
        constructor(configuration?: RTCConfiguration) {
          super(configuration);
          this.addEventListener('connectionstatechange', () =>
            probe.__voiceChecks.push({
              connection: this.connectionState,
              ice: this.iceConnectionState,
            }),
          );
        }
        override createDataChannel(label: string, options?: RTCDataChannelInit) {
          const channel = super.createDataChannel(label, options);
          channel.addEventListener('message', (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'error')
              probe.__voiceChecks.push({
                type: data.type,
                code: data.error?.code,
                param: data.error?.param,
              });
            else if (data.type.startsWith('session.') || data.type === 'response.done')
              probe.__voiceChecks.push({ type: data.type });
          });
          return channel;
        }
      };
    });
    page.on('response', async (response) => {
      if (response.url().startsWith('https://api.openai.com/v1/realtime/calls')) {
        const data = response.ok() ? null : await response.json().catch(() => null);
        providerChecks.push({
          status: response.status(),
          code: data?.error?.code,
          param: data?.error?.param,
        });
      }
    });
  }
  // Optionally test this worktree's build while an existing Vite serves shared media.
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
        const contentType = file.endsWith('.js')
          ? 'text/javascript'
          : file.endsWith('.css')
            ? 'text/css'
            : file.endsWith('.html')
              ? 'text/html'
              : 'application/octet-stream';
        await route.fulfill({ body: await readFile(file), contentType });
      } else await route.continue();
    });
  }
  await page.route('**/api/bottle-agent/*', async (route) => {
    const request = route.request();
    if (new URL(request.url()).pathname.endsWith('/session')) voiceSessionRequests++;
    const response = await fetch(`${backendOrigin}${new URL(request.url()).pathname}`, {
      method: request.method(),
      headers: { Origin: backendOrigin, 'Content-Type': 'application/json' },
      body: request.postData(),
    });
    await route.fulfill({
      status: response.status,
      contentType: 'application/json',
      body: await response.text(),
    });
  });
  await page.addInitScript({ path: new URL('./fake-webxr.js', import.meta.url).pathname });
  await page.goto(
    `http://127.0.0.1:5399/fourd.html?demo=elevator&xr=1&interact=1&walk=1&fakexr=1&fakew=512&fakeh=512&xradapt=0&xrmove=${process.env.WANDER_TEST_XR_MOVE === 'smooth' ? 'smooth' : 'teleport'}`,
    { timeout: 120000 },
  );
  await page.waitForFunction(
    () => {
      const w = window.wander as SceneDiagnostics;
      return w?.ready && w.interaction && w.video.readyState >= 3;
    },
    null,
    { timeout: 180000 },
  );
  await page.click('#xrBtn');
  await page.waitForFunction(() => window.wander.spark.renderer.xr.isPresenting);
  const frames = async (count = 12) => {
    const start = await page.evaluate(() => window.__fakeXR.frames.length);
    await page.waitForFunction((end) => window.__fakeXR.frames.length >= end, start + count);
  };
  await frames();
  const initial = await page.evaluate(() => {
    const w = window.wander as SceneDiagnostics;
    return {
      state: w.interaction.snapshot(),
      upm: w.upm,
      stature: w.walk?.stature,
      head: w.spark.renderer.xr.getCamera().getWorldPosition(new w.THREE.Vector3()).toArray(),
      people: w.people.map((p) => ({
        id: p.id,
        position: p.group.getWorldPosition(new w.THREE.Vector3()).toArray(),
      })),
      sceneChildren: w.scene.children.map((o) => ({ name: o.name, type: o.type })),
    };
  });
  await writeFile(`${output}/initial.json`, JSON.stringify(initial, null, 2));
  console.log('Normal initial playback, isolated opt-in controls, and single-pass source');
  await page.screenshot({ path: `${output}/initial-xr.png` });
  await page
    .evaluate(() => {
      const canvas = document.createElement('canvas');
      canvas.width = window.wander.video.videoWidth;
      canvas.height = window.wander.video.videoHeight;
      canvas.getContext('2d')!.drawImage(window.wander.video, 0, 0);
      return canvas.toDataURL();
    })
    .then((url) => writeFile(`${output}/source.png`, Buffer.from(url.split(',')[1], 'base64')));
  assert.equal(initial.state.interrupted, false);
  assert.equal(initial.state.playing, true);
  assert.equal(initial.state.sceneId, 'elevator');
  assert.equal(await page.evaluate(() => window.wander.video.loop), false);
  assert.equal(
    await page.evaluate(
      () =>
        (window.wander as SceneDiagnostics).scene.getObjectByName('interaction-controls') ===
        undefined,
    ),
    true,
    'The floating interaction toolbar must not be present',
  );
  const snapshot = () =>
    page.evaluate(() => (window.wander as SceneDiagnostics).interaction.snapshot());
  const visualState = () =>
    page.evaluate(() => {
      const w = window.wander as SceneDiagnostics;
      const scene = w.scene;
      const interaction = w.interaction as unknown as {
        bottle: { mesh: THREE.Object3D };
      };
      const group = scene.getObjectByName('bottle-interaction-visual')!;
      const proxy = scene.getObjectByName('interaction-bottle-proxy')!;
      const locator = scene.getObjectByName('interaction-bottle-locator')!;
      return {
        group: { present: !!group, visible: group?.visible },
        proxy: {
          present: !!proxy,
          visible: proxy?.visible,
          position: proxy?.getWorldPosition(new w.THREE.Vector3()).toArray(),
          quaternion: proxy?.getWorldQuaternion(new w.THREE.Quaternion()).toArray(),
        },
        locator: { present: !!locator, visible: locator?.visible },
        source: { visible: interaction.bottle.mesh.visible },
      };
    });
  await page.waitForFunction(
    (liveVoice) => {
      const state = (window.wander as SceneDiagnostics).interaction.snapshot();
      return liveVoice
        ? state.voiceConnected
        : !state.voiceStatus.startsWith('Connecting') && state.voiceStatus !== 'Mic off';
    },
    liveVoice,
    { timeout: 30000 },
  );
  assert.equal((await snapshot()).voiceConnected, liveVoice);
  assert.equal(voiceSessionRequests, 1, 'Enter VR must start one local microphone/session attempt');
  await page.evaluate(() => (window.wander as SceneDiagnostics).interaction.control('pause'));
  const paused = await snapshot();
  assert.equal(paused.playing, false, 'Diagnostic pause setup must stop the recording');
  await frames(12);
  assert.equal((await snapshot()).recordingTime, paused.recordingTime);
  await page.evaluate(() => window.wander.play(true));
  await page.waitForFunction(() => window.wander.playing);
  console.log('Missing voice configuration does not block Enter VR or initial playback');

  await page.evaluate(() => window.wander.setTime(window.wander.dur - 0.25));
  await page.waitForFunction(() => !window.wander.playing);
  const ended = await snapshot();
  await frames(20);
  assert.equal((await snapshot()).recordingTime, ended.recordingTime);
  assert.ok(ended.recordingTime > 9);
  console.log('Recording reaches its end once and remains stopped');

  await page.evaluate(() => (window.wander as SceneDiagnostics).interaction.replay(false));
  await page.evaluate(() => window.wander.setTime(2));
  // Anchor independently from the packaged head sidecar, transformed by the rendered person.
  const anchor = await page.evaluate(async () => {
    const w = window.wander as SceneDiagnostics;
    const person = w.people[0];
    const track = await (
      await fetch(new URL('head.json', new URL(person.seqUrl, location.href)))
    ).json();
    const index = Math.round(w.t * track.fps);
    return new w.THREE.Vector3(...track.positions[index])
      .applyMatrix4(person.group.matrixWorld)
      .toArray();
  });
  const headAt = async (distance: number) => {
    await page.evaluate(
      ({ anchor, distance }) => {
        const w = window.wander as SceneDiagnostics;
        const rig = w.scene.getObjectByName('xr-rig')!;
        const target = new w.THREE.Vector3(...anchor);
        const point = target.clone().add(new w.THREE.Vector3(0, 0, distance * w.walk!.stature));
        rig.worldToLocal(point);
        Object.assign(window.__fakeXR.head, {
          x: point.x,
          y: point.y,
          z: point.z,
          yaw: 0,
          pitch: 0,
        });
      },
      { anchor, distance },
    );
    await frames(6);
  };
  const armLiveAudio = async () => {
    if (!liveVoice) return;
    await page.waitForFunction(() => {
      const internals = (window.wander as SceneDiagnostics).interaction as unknown as {
        client: { audio: AudioContext | null; gain: GainNode | null; session: unknown };
      };
      return !!internals.client.audio && !!internals.client.gain && !!internals.client.session;
    });
    await page.evaluate(() => {
      const internals = (window.wander as SceneDiagnostics).interaction as unknown as {
        client: {
          audio: AudioContext;
          gain: GainNode;
          session: {
            transport: {
              on(name: string, callback: (event: Record<string, unknown>) => void): void;
            };
          };
        };
      };
      const analyzer = internals.client.audio.createAnalyser();
      const measurement = {
        analyzer,
        gain: internals.client.gain,
        completed: false,
        reply: '',
        peak: 0,
        timer: 0,
      };
      measurement.gain.connect(measurement.analyzer);
      const samples = new Float32Array(analyzer.fftSize);
      measurement.timer = window.setInterval(() => {
        analyzer.getFloatTimeDomainData(samples);
        for (const sample of samples)
          measurement.peak = Math.max(measurement.peak, Math.abs(sample));
      }, 20);
      internals.client.session.transport.on('*', (event) => {
        if (
          ['response.output_audio_transcript.done', 'response.audio_transcript.done'].includes(
            String(event.type),
          ) &&
          typeof event.transcript === 'string'
        )
          measurement.reply += event.transcript;
        if (event.type === 'response.done') {
          const response = event.response as {
            status?: string;
            output?: Array<{ type?: string; content?: Array<{ transcript?: string }> }>;
          };
          measurement.completed ||=
            response.status === 'completed' &&
            !!response.output?.some((item) => item.type === 'message');
          const reply =
            response.output
              ?.flatMap((item) => item.content ?? [])
              .map((content) => content.transcript ?? '')
              .filter(Boolean)
              .join(' ') ?? '';
          if (reply) measurement.reply = reply;
        }
      });
      (window as unknown as { __liveVoiceAudio?: typeof measurement }).__liveVoiceAudio =
        measurement;
    });
  };
  const measureLiveAudio = async () => {
    if (!liveVoice) return null;
    return page.evaluate(async () => {
      const measurement = (
        window as unknown as {
          __liveVoiceAudio: {
            analyzer: AnalyserNode;
            gain: GainNode;
            completed: boolean;
            reply: string;
            peak: number;
            timer: number;
          };
        }
      ).__liveVoiceAudio;
      const start = performance.now();
      while (
        performance.now() - start < 12000 &&
        (!measurement.completed || measurement.peak < 0.0001)
      ) {
        await new Promise((resolve) => setTimeout(resolve, 20));
      }
      clearInterval(measurement.timer);
      measurement.gain.disconnect(measurement.analyzer);
      Reflect.deleteProperty(window, '__liveVoiceAudio');
      return { completed: measurement.completed, peak: measurement.peak, reply: measurement.reply };
    });
  };
  await headAt(1.15);
  await armLiveAudio();
  await page.evaluate(() => window.wander.play(true));
  await headAt(1.02);
  assert.equal(
    (await snapshot()).interrupted,
    false,
    'Passing outside the close zone must not pause',
  );
  await headAt(0.88);
  await page.waitForFunction(
    () => (window.wander as SceneDiagnostics).interaction.snapshot().interrupted,
  );
  const approached = await snapshot();
  assert.equal(approached.lastEvent, 'approached');
  assert.equal(approached.activePersonId, 'person');
  assert.equal(approached.playing, false);
  assert.ok(approached.recordingTime > 2, 'Approach must interrupt an advancing recording');
  assert.equal(await page.evaluate(() => window.wander.video.paused), true);
  const beforeAutomaticFacing = await page.evaluate(() => {
    const w = window.wander as SceneDiagnostics;
    const person = w.people.find(
      (person) => person.id === w.interaction.snapshot().activePersonId,
    )!;
    return person.group.getWorldQuaternion(new w.THREE.Quaternion()).toArray();
  });
  await frames(30);
  const automaticFacing = await page.evaluate(() => {
    const w = window.wander as SceneDiagnostics;
    const person = w.people.find(
      (person) => person.id === w.interaction.snapshot().activePersonId,
    )!;
    const pivot = w.scene.getObjectByName(`interaction-facing-${person.id}`);
    return {
      pivot: !!pivot,
      after: person.group.getWorldQuaternion(new w.THREE.Quaternion()).toArray(),
    };
  });
  assert.equal(automaticFacing.pivot, true, 'Interruption must begin smooth local facing');
  assert.notDeepEqual(
    beforeAutomaticFacing,
    automaticFacing.after,
    'Automatic facing must advance while the visitor remains addressable',
  );
  const facing = await page.evaluate(() => {
    const w = window.wander as SceneDiagnostics;
    const person = w.people.find(
      (person) => person.id === w.interaction.snapshot().activePersonId,
    )!;
    const before = person.group.getWorldQuaternion(new w.THREE.Quaternion()).toArray();
    const response = (
      w.interaction as unknown as { agentAction(name: string, args: unknown): string }
    ).agentAction('face_player', {});
    const after = person.group.getWorldQuaternion(new w.THREE.Quaternion()).toArray();
    return { before, after, response };
  });
  assert.equal(facing.response, 'Character turned toward visitor.');
  assert.equal((await snapshot()).playing, false, 'Facing must not resume the recording');
  await page.screenshot({ path: `${output}/approached-xr.png` });
  console.log('Close approach interrupts playback; smooth local facing follows the paused body');

  if (liveVoice) {
    assert.equal(
      (await snapshot()).voiceConnected,
      true,
      JSON.stringify({
        voiceStatus: (await snapshot()).voiceStatus,
        providerChecks,
        rtc: await page.evaluate(() => (window as unknown as VoiceProbeWindow).__voiceChecks),
      }),
    );
    const voice = await measureLiveAudio();
    await writeFile(`${output}/live-voice.json`, JSON.stringify(voice, null, 2));
    assert.ok(
      voice && voice.completed && voice.peak > 0.0001 && voice.reply,
      JSON.stringify(voice),
    );
    console.log('Live provider: auto-started voice and generated audio through the spatial graph');
  } else {
    assert.equal((await snapshot()).voiceConnected, false);
    console.log('Missing voice configuration is visible without blocking local interaction');
  }

  const setXrButton = async (hand: 'left' | 'right', index: number, pressed: boolean) => {
    await page.evaluate(
      ({ hand, index, pressed }) => {
        window.__fakeXR.buttons[hand][index] = pressed ? 1 : 0;
      },
      { hand, index, pressed },
    );
    await frames(4);
  };
  await page.evaluate(() => {
    for (const hand of ['left', 'right'] as const) {
      for (let index = 0; index < 6; index++) window.__fakeXR.buttons[hand][index] ??= 0;
    }
  });
  await setXrButton('left', 4, false);
  const beforeReplay = await snapshot();
  await setXrButton('left', 4, true);
  const replayed = await snapshot();
  assert.equal(replayed.lastEvent, 'replayed');
  assert.equal(replayed.epoch, beforeReplay.epoch + 1, 'Left X must replay once on press');
  assert.equal(replayed.playing, true);
  assert.ok(replayed.recordingTime < 1);
  const restoredVisual = await visualState();
  assert.deepEqual(restoredVisual.group, { present: true, visible: true });
  assert.equal(
    restoredVisual.source.visible,
    true,
    'Replay must restore the recorded bottle source',
  );
  assert.equal(restoredVisual.proxy.visible, false, 'Replay must hide the interaction proxy');
  assert.equal(restoredVisual.locator.visible, true, 'Replay must restore the bottle locator');
  assert.equal(
    await page.evaluate(
      () =>
        (window.wander as SceneDiagnostics).scene.getObjectByName('interaction-facing-person') ===
        undefined,
    ),
    true,
    'Replay must restore the paused-facing transform',
  );
  await frames(12);
  const heldX = await snapshot();
  assert.equal(heldX.epoch, replayed.epoch, 'Holding X must not repeat Replay');
  await setXrButton('left', 4, false);
  await setXrButton('right', 4, true);
  assert.equal((await snapshot()).epoch, replayed.epoch, 'Right A must not replay');
  await setXrButton('right', 4, false);
  await setXrButton('left', 4, true);
  assert.equal((await snapshot()).epoch, replayed.epoch + 1, 'X must rearm after release');
  await setXrButton('left', 4, false);
  console.log('Left X replays once per press; right A is ignored; replay clears paused facing');

  await page.evaluate(() => (window.wander as SceneDiagnostics).interaction.replay(false));
  assert.equal((await snapshot()).activePersonId, null);
  assert.equal(
    await page.evaluate(
      () =>
        (window.wander as SceneDiagnostics).scene.getObjectByName('interaction-facing-person') ===
        undefined,
    ),
    true,
  );
  await page.evaluate(() => window.wander.setTime(0.4));
  const beforeGripVisual = await visualState();
  assert.deepEqual(beforeGripVisual.group, { present: true, visible: true });
  assert.equal(
    beforeGripVisual.locator.visible,
    true,
    'The loose bottle must have a visible locator',
  );
  assert.equal(
    beforeGripVisual.proxy.visible,
    false,
    'The proxy is only for interrupted interaction',
  );
  const handAt = async (
    position: number[],
    squeeze: boolean,
    count = 3,
    rotation: [number, number, number, number] = [0, 0, 0, 1],
  ) => {
    await page.evaluate(
      ({ position, squeeze, rotation }) => {
        const w = window.wander as SceneDiagnostics;
        const rig = w.scene.getObjectByName('xr-rig')!;
        const point = rig.worldToLocal(new w.THREE.Vector3(...position));
        const source = (w.spark.renderer.xr.getSession() as SyntheticSession).inputSources[1];
        source.targetRaySpace._matrix = new Float32Array(
          new w.THREE.Matrix4().compose(
            point,
            new w.THREE.Quaternion(...rotation),
            new w.THREE.Vector3(1, 1, 1),
          ).elements,
        );
        window.__fakeXR.buttons.right[1] = squeeze ? 1 : 0;
      },
      { position, squeeze, rotation },
    );
    await frames(count);
  };
  await frames(3);
  await handAt((await snapshot()).bottle.position, true, 3, [
    Math.sin(Math.PI / 4),
    0,
    0,
    Math.cos(Math.PI / 4),
  ]);
  let held = await snapshot();
  assert.equal(held.bottle.mode, 'held', JSON.stringify(held));
  assert.equal(held.playing, false);
  assert.ok(held.returnTarget, 'Manifest must yield an available assisted return target');
  const heldVisual = await page.evaluate(() => {
    const w = window.wander as SceneDiagnostics;
    const scene = w.scene;
    const proxy = scene.getObjectByName('interaction-bottle-proxy')!;
    const locator = scene.getObjectByName('interaction-bottle-locator')!;
    const source = (w.interaction as unknown as { bottle: { mesh: THREE.Object3D } }).bottle.mesh;
    const rig = scene.getObjectByName('xr-rig')!;
    const input = (w.spark.renderer.xr.getSession() as SyntheticSession).inputSources[1];
    const raw = new w.THREE.Matrix4().fromArray(input.targetRaySpace._matrix);
    const rawPosition = new w.THREE.Vector3().setFromMatrixPosition(raw);
    const rawRotation = new w.THREE.Quaternion().setFromRotationMatrix(raw);
    const expected = rawPosition
      .add(new w.THREE.Vector3(0, 0.012, -0.025).applyQuaternion(rawRotation))
      .applyMatrix4(rig.matrixWorld);
    return {
      proxyVisible: proxy.visible,
      locatorVisible: locator.visible,
      sourceVisible: source.visible,
      proxyPosition: proxy.getWorldPosition(new w.THREE.Vector3()).toArray(),
      expectedPosition: expected.toArray(),
      proxyQuaternion: proxy.getWorldQuaternion(new w.THREE.Quaternion()).toArray(),
      gripQuaternion: w.spark.renderer.xr
        .getControllerGrip(1)
        .getWorldQuaternion(new w.THREE.Quaternion())
        .toArray(),
    };
  });
  assert.equal(heldVisual.proxyVisible, true, 'A held bottle must use the interaction proxy');
  assert.equal(heldVisual.locatorVisible, false, 'The locator must hide while the bottle is held');
  assert.equal(heldVisual.sourceVisible, false, 'The recorded source bottle must hide while held');
  const distance = (a: number[], b: number[]) => Math.hypot(...a.map((value, i) => value - b[i]!));
  assert.ok(
    distance(heldVisual.proxyPosition, heldVisual.expectedPosition) < 0.004 * initial.upm,
    JSON.stringify(heldVisual),
  );
  assert.ok(
    Math.abs(
      heldVisual.proxyQuaternion.reduce(
        (dot, value, index) => dot + value * heldVisual.gripQuaternion[index]!,
        0,
      ),
    ) > 0.999,
    JSON.stringify(heldVisual),
  );
  // Bring the held bottle away from the recorded actor so the screenshot can assess the grip
  // without the source person's hand covering it. The rotation assertion above covers tilting.
  await handAt(
    held.bottle.position.map((value, i) => value + [0.25, 0.2, 0.3][i]! * initial.stature!),
    true,
  );
  held = await snapshot();
  assert.equal(held.bottle.mode, 'held');
  await page.evaluate((bottle) => {
    const w = window.wander as SceneDiagnostics;
    const rig = w.scene.getObjectByName('xr-rig')!;
    const target = new w.THREE.Vector3(...bottle);
    const origin = target
      .clone()
      .add(new w.THREE.Vector3(0, 0.2, 0.5).multiplyScalar(w.walk!.stature));
    const facing = new w.THREE.Quaternion().setFromRotationMatrix(
      new w.THREE.Matrix4().lookAt(origin, target, new w.THREE.Vector3(0, 1, 0)),
    );
    facing.premultiply(rig.getWorldQuaternion(new w.THREE.Quaternion()).invert());
    const euler = new w.THREE.Euler().setFromQuaternion(facing, 'YXZ');
    const local = rig.worldToLocal(origin);
    Object.assign(window.__fakeXR.head, {
      x: local.x,
      y: local.y,
      z: local.z,
      yaw: euler.y,
      pitch: euler.x,
    });
  }, held.bottle.position);
  await frames(4);
  await page.screenshot({ path: `${output}/held-xr.png` });
  const stature = initial.stature!;
  const target = held.returnTarget!;
  await handAt([target[0], target[1] + 0.04 * stature, target[2] + 0.3 * stature], true);
  await handAt([target[0], target[1] + 0.04 * stature, target[2] + 0.25 * stature], false, 1);
  await page.waitForFunction(
    () => (window.wander as SceneDiagnostics).interaction.snapshot().bottle.mode === 'returned',
  );
  assert.equal((await snapshot()).playing, false);
  console.log('Controller grip picks up the bottle; a short throw reaches the assisted target');

  await page.evaluate(async () => window.wander.spark.renderer.xr.getSession()!.end());
  await page.click('#xrBtn');
  await frames(6);
  assert.equal((await snapshot()).playing, false);
  assert.equal((await snapshot()).bottle.mode, 'returned');
  console.log('Exiting and re-entering VR preserves the interrupted scene');

  // The same build must keep the established looping viewer outside the experiment.
  await page.evaluate(async () => window.wander.spark.renderer.xr.getSession()!.end());
  await page.goto(
    'http://127.0.0.1:5399/fourd.html?demo=elevator&xr=1&fakexr=1&fakew=256&fakeh=256&xradapt=0&pause=1',
    { timeout: 120000 },
  );
  await page.waitForFunction(
    () => window.wander?.ready && window.wander.video.readyState >= 3,
    null,
    { timeout: 180000 },
  );
  assert.equal(
    await page.evaluate(() => Boolean((window.wander as SceneDiagnostics).interaction)),
    false,
  );
  assert.equal(await page.evaluate(() => window.wander.video.loop), true);
  await page.click('#xrBtn');
  await page.waitForFunction(() => window.wander.playing && !window.wander.video.paused);
  await page.evaluate(() => window.wander.setTime(window.wander.dur - 0.25));
  await page.waitForFunction(
    () => window.wander.video.currentTime < 1 && window.wander.t < 1 && window.wander.playing,
  );
  console.log('Standard viewer still loops and starts playback on VR entry');
  assert.deepEqual(errors, []);
} finally {
  await browser.close();
  backend.closeAllConnections();
  await new Promise<void>((resolve) => backend.close(() => resolve()));
}
