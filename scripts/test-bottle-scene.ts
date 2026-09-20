import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import * as THREE from 'three';
import type { BottleAgentClient } from '../src/interaction/bottle-agent-client';
import {
  BottleScene,
  type BottleSceneHost,
  type InteractionHand,
  type InteractionObject,
  type InteractionPerson,
} from '../src/interaction/bottle-scene';

// Only browser services are stubbed. Scene ownership, approach detection, transforms,
// and bottle physics all use the production implementation; no SDK module is replaced.
const originals = new Map<string, PropertyDescriptor | undefined>();
const instances: BottleScene[] = [];
function replace(name: string, value: unknown) {
  originals.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
  Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
}

beforeEach(() => {
  const events = new EventTarget();
  replace('addEventListener', events.addEventListener.bind(events));
  replace('removeEventListener', events.removeEventListener.bind(events));
  replace('location', { href: 'http://127.0.0.1:5399/fourd.html' });
  replace('document', {
    hidden: false,
    addEventListener: events.addEventListener.bind(events),
    removeEventListener: events.removeEventListener.bind(events),
    createElement(tag: string) {
      if (tag !== 'canvas') throw new Error(`Unexpected browser element: ${tag}`);
      return {
        width: 0,
        height: 0,
        getContext: () => ({ fillRect() {}, strokeRect() {}, fillText() {} }),
      };
    },
  });
  replace('fetch', async (input: string | URL) => {
    if (!String(input).endsWith('/head.json')) throw new Error('Unexpected network request');
    // Exercise the documented visible-body fallback rather than loading scene assets.
    return new Response(null, { status: 404 });
  });
});

afterEach(() => {
  for (const instance of instances.splice(0)) instance.dispose();
  for (const [name, descriptor] of originals) {
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else Reflect.deleteProperty(globalThis, name);
  }
  originals.clear();
});

function fixture(startTime = 1) {
  const scene = new THREE.Scene();
  const people: InteractionPerson[] = ['thrower', 'receiver'].map((id, track) => {
    const group = new THREE.Group();
    group.position.set(track ? 2 : 0, 0, -2);
    const pmesh = Object.assign(new THREE.Object3D(), {
      getBoundingBox: () =>
        new THREE.Box3(new THREE.Vector3(-0.1, 0, -0.1), new THREE.Vector3(0.1, 1.08, 0.1)),
    });
    group.add(pmesh);
    scene.add(group);
    return {
      id,
      label: id,
      meta: { track },
      seqUrl: `http://127.0.0.1:5399/people/${id}/seq.json`,
      group,
      pmesh,
      frame: 0,
      ts: [0, 1, 2, 3, 4],
    };
  });
  const bottle: InteractionObject = {
    id: 'bottle',
    label: 'Bottle',
    meta: {
      objectClass: 'thrown',
      appearance: { sizeWorldUnits: [0.04, 0.04, 0.04] },
      pose: {
        segments: [
          { kind: 'attached', fromSourceFrame: 0, toSourceFrame: 9, parent: 'thrower' },
          {
            kind: 'free',
            fromSourceFrame: 10,
            toSourceFrame: 29,
            throwerTrack: 0,
            catcherTrack: 1,
          },
          { kind: 'attached', fromSourceFrame: 30, toSourceFrame: 40, parent: 'receiver' },
        ],
      },
    },
    track: { fps: 10 },
    group: new THREE.Group(),
    mesh: new THREE.Object3D(),
    interactionOwned: false,
    samplePosition: (time) => new THREE.Vector3(2 * (time - 1.5), 1, -0.2),
  };
  bottle.group.add(bottle.mesh);
  scene.add(bottle.group);
  let time = startTime;
  let playing = false;
  let runtime: BottleScene;
  const setTime = (next: number) => {
    time = next;
    if (!bottle.interactionOwned) bottle.mesh.position.copy(bottle.samplePosition(time));
    scene.updateMatrixWorld(true);
  };
  setTime(time);
  const host: BottleSceneHost = {
    scene,
    people,
    objects: [bottle],
    stature: 1,
    sceneId: 'fixture',
    params: new URLSearchParams(),
    time: () => time,
    playing: () => playing,
    play: (next) => {
      playing = next;
      runtime?.transportChanged(next);
    },
    seek: setTime,
    floorAt: () => 0,
    blockedAt: () => false,
  };
  runtime = new BottleScene(host, bottle);
  instances.push(runtime);
  runtime.sessionStart();
  const head = new THREE.Vector3(0, 1, 0);
  const rotation = new THREE.Quaternion();
  const frame = (hands: InteractionHand[] = [], dt = 1 / 72) =>
    runtime.frame(head, rotation, hands, dt);
  return { runtime, bottle, people, head, rotation, frame, setTime, host };
}

const hand = (position: THREE.Vector3, squeeze: boolean): InteractionHand => ({
  id: 'right',
  position: position.clone(),
  rotation: new THREE.Quaternion(),
  squeeze,
});

// Invoke the same local acceptance callback used by the microphone's VAD event,
// without opening capture, creating credentials, or replacing the voice client.
const speechStarted = (runtime: BottleScene) =>
  (runtime as unknown as { onSpeech(): boolean }).onSpeech();

function voiceFixture(runtime: BottleScene) {
  const client = (runtime as unknown as { client: BottleAgentClient }).client;
  const options = (client as unknown as { options: { expired(): void } }).options;
  let connected = false;
  let attempts = 0;
  const pending: Array<() => void> = [];
  Object.defineProperty(client, 'connected', { get: () => connected });
  client.connect = () => {
    attempts++;
    return new Promise<void>((resolve) => pending.push(resolve));
  };
  client.disconnect = () => {
    connected = false;
  };
  return {
    attempts: () => attempts,
    async settle(success = true) {
      connected = success;
      pending.shift()?.();
      await Promise.resolve();
      await Promise.resolve();
    },
    expire() {
      connected = false;
      options.expired();
    },
  };
}

describe('BottleScene automatic voice ownership', () => {
  test('normal expiration renews while visible; replay does not reconnect and exit cancels renewal', async () => {
    const { runtime } = fixture();
    const voice = voiceFixture(runtime);
    runtime.startVoice();
    runtime.startVoice();
    expect(voice.attempts()).toBe(1);
    await voice.settle();
    runtime.replay();
    expect(voice.attempts()).toBe(1);
    voice.expire();
    expect(voice.attempts()).toBe(2);
    await voice.settle();
    runtime.sessionEnd();
    voice.expire();
    expect(voice.attempts()).toBe(2);
  });

  test('visibility resumes a healthy session once but never retries a failed connection', async () => {
    const { runtime, host } = fixture();
    const voice = voiceFixture(runtime);
    runtime.startVoice();
    await voice.settle();
    runtime.visibilityChanged(false);
    runtime.visibilityChanged(false);
    voice.expire();
    expect(voice.attempts()).toBe(1);
    expect(host.playing()).toBe(false);
    runtime.visibilityChanged(true);
    expect(voice.attempts()).toBe(2);
    await voice.settle(false);
    runtime.visibilityChanged(false);
    runtime.visibilityChanged(true);
    expect(voice.attempts()).toBe(2);
    expect(host.playing()).toBe(false);
  });

  test('a cancelled connection cannot clear a newer entry attempt', async () => {
    const { runtime } = fixture();
    const voice = voiceFixture(runtime);
    runtime.startVoice();
    runtime.sessionEnd();
    runtime.startVoice();
    runtime.sessionStart();
    await voice.settle(false);
    runtime.startVoice();
    expect(voice.attempts()).toBe(2);
    await voice.settle();
    runtime.stopVoice();
    voice.expire();
    expect(voice.attempts()).toBe(2);
  });
});

describe('BottleScene real-physics interaction integration', () => {
  test('a bottle crossing an armed stationary hand is caught during recorded-flight interruption', () => {
    const { runtime, frame, setTime, host } = fixture(1);
    const palm = new THREE.Vector3(0, 1, -0.2);
    frame([hand(palm, false)]);
    setTime(2);
    frame([hand(palm, true)]);
    const state = runtime.snapshot();
    expect(state.interrupted).toBe(true);
    expect(state.activePersonId).toBe('thrower');
    expect(state.bottle.mode).toBe('held');
    expect(state.bottle.holder).toBe('right');
    expect(state.bottle.position).toEqual(palm.toArray());
    expect(host.playing()).toBe(false);
  });

  test('Replay and Reset suppress an already-held grip until release rearms a fresh grab', () => {
    for (const play of [true, false]) {
      const { runtime, bottle, frame, host } = fixture(0.5);
      const firstPalm = bottle.mesh.getWorldPosition(new THREE.Vector3());
      frame([hand(firstPalm, false)]);
      frame([hand(firstPalm, true)]);
      expect(runtime.snapshot().bottle.holder).toBe('right');
      runtime.replay(play);
      const restoredPalm = bottle.mesh.getWorldPosition(new THREE.Vector3());
      frame([hand(restoredPalm, true)]);
      frame([hand(restoredPalm, true)]);
      expect(runtime.snapshot().interrupted).toBe(false);
      expect(runtime.snapshot().bottle.holder).toBe(null);
      expect(runtime.snapshot().bottle.mode).toBe('recorded');
      expect(host.playing()).toBe(play);
      frame([hand(restoredPalm, false)]);
      frame([hand(restoredPalm, true)]);
      expect(runtime.snapshot().bottle.holder).toBe('right');
      expect(runtime.snapshot().interrupted).toBe(true);
      expect(host.playing()).toBe(false);
    }
  });

  test('the active character remains addressable when another eligible person is nearer', () => {
    const { runtime, people, frame, head, rotation } = fixture(0.5);
    people[0].group.position.set(0, 0, -0.6);
    people[1].group.position.set(0.05, 0, -0.3);
    frame();
    expect(runtime.interrupt('approached', 'thrower')).toBe(true);
    expect(speechStarted(runtime)).toBe(true);
    expect(runtime.snapshot().activePersonId).toBe('thrower');
    // Locked selection must still reject a hidden, distant, or unaddressed target.
    people[0].group.visible = false;
    expect(speechStarted(runtime)).toBe(false);
    people[0].group.visible = true;
    head.z = 1;
    frame();
    expect(speechStarted(runtime)).toBe(false);
    head.z = 0;
    rotation.setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI);
    frame();
    expect(speechStarted(runtime)).toBe(false);
  });

  test('Replay restores a lost mesh, recorded ownership, and a fresh character selection', () => {
    const { runtime, bottle, frame } = fixture(1.5);
    expect(runtime.interrupt('approached', 'thrower')).toBe(true);
    expect(runtime.physics.startFlight([100, 1, 0], [0, 0, 0])).toBe(true);
    frame();
    expect(runtime.snapshot().bottle.lost).toBe(true);
    expect(bottle.mesh.visible).toBe(false);
    expect(runtime.snapshot().activePersonId).toBe('thrower');
    runtime.replay();
    expect(bottle.mesh.visible).toBe(true);
    expect(bottle.interactionOwned).toBe(false);
    expect(runtime.snapshot()).toMatchObject({
      activePersonId: null,
      interrupted: false,
      playing: true,
      recordingTime: 0,
      bottle: { mode: 'recorded', holder: null, personOwner: null, lost: false },
    });
    expect(bottle.mesh.getWorldPosition(new THREE.Vector3()).toArray()).toEqual(
      bottle.samplePosition(0).toArray(),
    );
  });

  test('a close approach hands airborne source velocity to physics and pauses the recording', () => {
    const { runtime, people, head, frame, bottle, host } = fixture(1.5);
    people[0].group.position.set(0, 0, -0.6);
    head.z = 0.3;
    frame([], 0.1); // Outside the exit zone, so a visitor approach may arm.
    head.z = -0.35;
    frame([], 0.2);
    expect(runtime.snapshot().interrupted).toBe(false);
    frame([], 0.2);
    const state = runtime.snapshot();
    expect(state.interrupted).toBe(true);
    expect(state.activePersonId).toBe('thrower');
    expect(host.playing()).toBe(false);
    expect(bottle.interactionOwned).toBe(true);
    expect(state.bottle.mode).toBe('free');
    expect(state.bottle.velocity[0]).toBeCloseTo(2, 8);
    expect(state.bottle.velocity[1]).toBeLessThan(0);
    expect(state.bottle.velocity[2]).toBeCloseTo(0, 8);
    expect(state.bottle.position[0]).toBeGreaterThan(bottle.samplePosition(1.5).x);
    expect(state.bottle.position[1]).toBeLessThan(bottle.samplePosition(1.5).y);
  });
});
