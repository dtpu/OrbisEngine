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

function fixture(startTime = 1, params = new URLSearchParams()) {
  const scene = new THREE.Scene();
  const meshUpdates = new Map<string, { generator: number; version: number }>();
  const people: InteractionPerson[] = ['thrower', 'receiver'].map((id, track) => {
    const updates = { generator: 0, version: 0 };
    meshUpdates.set(id, updates);
    const group = new THREE.Group();
    group.position.set(track ? 2 : 0, 0, -2);
    const pmesh = Object.assign(new THREE.Object3D(), {
      updateGenerator() {
        updates.generator++;
      },
      updateVersion() {
        updates.version++;
      },
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
  let blocked = false;
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
    params,
    time: () => time,
    playing: () => playing,
    play: (next) => {
      playing = next;
      runtime?.transportChanged(next);
    },
    seek: setTime,
    floorAt: () => 0,
    blockedAt: () => blocked,
  };
  runtime = new BottleScene(host, bottle);
  instances.push(runtime);
  runtime.sessionStart();
  const head = new THREE.Vector3(0, 1, 0);
  const rotation = new THREE.Quaternion();
  const frame = (hands: InteractionHand[] = [], dt = 1 / 72) =>
    runtime.frame(head, rotation, hands, dt);
  return {
    runtime,
    bottle,
    people,
    head,
    rotation,
    frame,
    setTime,
    host,
    meshUpdates,
    setBlocked: (value: boolean) => {
      blocked = value;
    },
  };
}

const hand = (
  position: THREE.Vector3,
  squeeze: boolean,
  rotation = new THREE.Quaternion(),
): InteractionHand => ({
  id: 'right',
  position: position.clone(),
  rotation: rotation.clone(),
  squeeze,
});

// Invoke the same local acceptance callback used by the microphone's VAD event,
// without opening capture, creating credentials, or replacing the voice client.
const speechStarted = (runtime: BottleScene) =>
  (runtime as unknown as { onSpeech(): boolean }).onSpeech();

function voiceFixture(runtime: BottleScene) {
  const client = (runtime as unknown as { client: BottleAgentClient }).client;
  const options = (
    client as unknown as { options: { expired(): void; speaking(value: boolean): void } }
  ).options;
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
    speaking: (value: boolean) => options.speaking(value),
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

  test('initial voice needs a visible, nearby, unobstructed person within 1.8 body-heights', () => {
    const { runtime, people, frame, rotation, setBlocked, host } = fixture(0.5);
    people[0].group.position.set(0, 0, -1.2);
    people[1].group.position.set(4, 0, -4);
    frame();
    people[0].group.visible = false;
    expect(speechStarted(runtime)).toBe(false);
    people[0].group.visible = true;
    people[0].group.position.z = -1.81;
    frame();
    expect(speechStarted(runtime)).toBe(false);
    people[0].group.position.z = -1.2;
    setBlocked(true);
    frame();
    expect(speechStarted(runtime)).toBe(false);
    setBlocked(false);
    rotation.setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI);
    frame();
    expect(speechStarted(runtime)).toBe(false);
    rotation.identity();
    frame();
    expect(host.playing()).toBe(true);
    expect(speechStarted(runtime)).toBe(true);
    expect(runtime.snapshot().activePersonId).toBe('thrower');
    expect(host.playing()).toBe(false);
    expect(runtime.snapshot().lastEvent).toBe('speech');
  });

  test('continuing speech permits looking away to 2.52 body-heights and can switch to a clearly addressed person', () => {
    const { runtime, bottle, people, frame, head, rotation } = fixture(0.5);
    people[0].group.position.set(1.5, 0, -1);
    people[1].group.position.set(0, 0, -1);
    frame();
    expect(runtime.interrupt('approached', 'thrower')).toBe(true);
    const firstCharacter = runtime.snapshot().character;
    const beforeSwitch = runtime.snapshot().bottle.position;
    expect(speechStarted(runtime)).toBe(true);
    expect(runtime.snapshot().activePersonId).toBe('receiver');
    expect(runtime.snapshot().bottle.position).toEqual(beforeSwitch);
    expect(runtime.snapshot().bottle.personOwner).toBe('thrower');
    expect(runtime.snapshot().character?.role).toBe('receiver');
    expect(runtime.snapshot().character?.style).not.toBe(firstCharacter?.style);

    head.z = 1.4;
    rotation.setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI);
    frame();
    expect(speechStarted(runtime)).toBe(true);
    head.z = 1.53;
    frame();
    expect(speechStarted(runtime)).toBe(false);
    expect(bottle.interactionOwned).toBe(true);
  });

  test('a closer person in the same direction does not steal the current conversation', () => {
    const { runtime, people, frame } = fixture(0.5);
    people[0].group.position.set(0, 0, -0.6);
    people[1].group.position.set(0.05, 0, -0.3);
    frame();
    runtime.interrupt('approached', 'thrower');
    expect(speechStarted(runtime)).toBe(true);
    expect(runtime.snapshot().activePersonId).toBe('thrower');
  });

  test('automatic facing turns the paused body smoothly and replay restores its recorded transform', () => {
    const { runtime, people, head, frame } = fixture(0.5);
    people[0].group.position.set(0, 0, -1.2);
    head.set(1.2, 1, 0);
    frame();
    const before = people[0].group.getWorldQuaternion(new THREE.Quaternion()).toArray();
    const beforePosition = people[0].group.getWorldPosition(new THREE.Vector3());
    expect(runtime.interrupt('approached', 'thrower')).toBe(true);
    frame([], 0.1);
    const pivot = people[0].group.parent;
    const after = people[0].group.getWorldQuaternion(new THREE.Quaternion()).toArray();
    expect(pivot?.name).toBe('interaction-facing-thrower');
    expect(after).not.toEqual(before);
    expect(new THREE.Quaternion(...before).angleTo(new THREE.Quaternion(...after))).toBeLessThan(
      0.161,
    );
    expect(
      people[0].group.getWorldPosition(new THREE.Vector3()).distanceTo(beforePosition),
    ).toBeLessThan(1e-8);
    expect(runtime.snapshot().playing).toBe(false);
    runtime.replay();
    expect(people[0].group.parent?.name).not.toBe('interaction-facing-thrower');
    expect(people[0].group.getWorldQuaternion(new THREE.Quaternion()).toArray()).toEqual(before);
  });

  test('the visible locator becomes a rotated held proxy and replay restores the source bottle', () => {
    const { runtime, bottle, frame } = fixture(0.5);
    frame();
    const scene = bottle.group.parent!;
    const visual = scene.getObjectByName('bottle-interaction-visual')!;
    const proxy = scene.getObjectByName('interaction-bottle-proxy')!;
    const locator = scene.getObjectByName('interaction-bottle-locator')!;
    expect(visual.visible).toBe(true);
    expect(locator.visible).toBe(true);
    expect(proxy.visible).toBe(false);
    expect(bottle.mesh.visible).toBe(true);

    const grip = bottle.mesh.getWorldPosition(new THREE.Vector3());
    const rotation = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(1, 0, 0),
      Math.PI / 2,
    );
    frame([hand(grip, false, rotation)]);
    frame([hand(grip, true, rotation)]);
    expect(runtime.snapshot().bottle.mode).toBe('held');
    expect(proxy.visible).toBe(true);
    expect(locator.visible).toBe(false);
    expect(bottle.mesh.visible).toBe(false);
    expect(proxy.getWorldPosition(new THREE.Vector3()).distanceTo(grip)).toBeLessThan(1e-8);
    expect(
      Math.abs(proxy.getWorldQuaternion(new THREE.Quaternion()).dot(rotation)),
    ).toBeGreaterThan(0.9999);

    runtime.replay();
    expect(bottle.mesh.visible).toBe(true);
    expect(proxy.visible).toBe(false);
    frame();
    expect(bottle.mesh.visible).toBe(true);
    expect(proxy.visible).toBe(false);
    expect(locator.visible).toBe(true);
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
    head.z = 1;
    frame([], 0.1); // Outside the exit zone, so a visitor approach may arm.
    head.z = 0.15; // Still 0.75 body-heights away; the old close-only threshold rejected this.
    frame([], 0.1);
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

const animationPose = (runtime: BottleScene, personId = 'thrower') =>
  runtime.snapshot().animations.find((item) => item.personId === personId)?.pose;
const zeroAnimation = {
  pitch: 0,
  yaw: 0,
  roll: 0,
  breath: 0,
  leftShoulder: 0,
  rightShoulder: 0,
  leftElbow: 0,
  rightElbow: 0,
  weight: 0,
  mode: 'off' as const,
};
const modifiers = (person: InteractionPerson) =>
  (person.pmesh as unknown as { objectModifiers?: unknown[] }).objectModifiers ?? [];

function conversationFixture(params = new URLSearchParams()) {
  const setup = fixture(0.5, params);
  setup.people[0].group.position.set(0, 0, -1.2);
  setup.people[1].group.position.set(4, 0, -4);
  setup.frame();
  const tick = (count = 72) => {
    for (let i = 0; i < count; i++) setup.frame();
  };
  return { ...setup, tick };
}

const armFreedom = (runtime: BottleScene, personId = 'thrower') =>
  runtime.snapshot().animations.find((item) => item.personId === personId)!.armFreedom!;

function placeRecordedBottle(setup: ReturnType<typeof conversationFixture>, x: number) {
  setup.bottle.samplePosition = () => new THREE.Vector3(x, 0.8, -1.1);
  setup.setTime(setup.host.time());
}

describe('BottleScene held-prop arm ownership', () => {
  test('a recorded held bottle pins its side, with both arms pinned for a central attachment', () => {
    for (const x of [-0.18, 0, 0.18]) {
      const setup = conversationFixture();
      placeRecordedBottle(setup, x);
      expect(setup.runtime.interrupt('approached', 'thrower')).toBe(true);
      voiceFixture(setup.runtime).speaking(true);
      setup.tick();
      expect(setup.runtime.snapshot().bottle.personOwner).toBe('thrower');
      const [left, right] = armFreedom(setup.runtime);
      if (x <= 0) expect(left).toBe(0);
      else expect(left).toBeGreaterThan(0.99);
      if (x >= 0) expect(right).toBe(0);
      else expect(right).toBeGreaterThan(0.99);
    }
  });

  test('a visitor grab gradually frees the holding arm while the source stays paused', () => {
    const setup = conversationFixture();
    const { runtime, frame, tick, host, bottle } = setup;
    const voice = voiceFixture(runtime);
    placeRecordedBottle(setup, -0.18);
    expect(runtime.interrupt('approached', 'thrower')).toBe(true);
    voice.speaking(true);
    tick();
    expect(armFreedom(runtime)[0]).toBe(0);
    const palm = bottle.mesh.getWorldPosition(new THREE.Vector3());
    frame([hand(palm, false)]);
    frame([hand(palm, true)]);
    expect(runtime.snapshot().bottle).toMatchObject({ mode: 'held', holder: 'right' });
    const released = armFreedom(runtime)[0];
    expect(released).toBeGreaterThan(0);
    expect(released).toBeLessThan(0.2);
    // The grab invalidates stale generated output; a fresh reply may now gesture.
    voice.speaking(true);
    for (let i = 0; i < 72; i++) frame([hand(palm, true)]);
    expect(armFreedom(runtime)[0]).toBeGreaterThan(0.99);
    expect(animationPose(runtime)!.mode).toBe('speaking');
    expect(runtime.snapshot().bottle.holder).toBe('right');
    expect(host.time()).toBe(0.5);
    expect(host.playing()).toBe(false);
  });

  test('a bottle held by the other person leaves the active speaker free to gesture', () => {
    const setup = conversationFixture();
    setup.setTime(3.5);
    placeRecordedBottle(setup, 0);
    expect(setup.runtime.interrupt('approached', 'thrower')).toBe(true);
    setup.tick();
    expect(setup.runtime.snapshot().bottle.personOwner).toBe('receiver');
    expect(setup.runtime.snapshot().activePersonId).toBe('thrower');
    for (const freedom of armFreedom(setup.runtime)) expect(freedom).toBeGreaterThan(0.99);
  });

  test('non-thrown wrist props pin their owner, while spine props and another owner do not', () => {
    for (const attachment of [
      { id: 'cup', parent: 'thrower', jointName: 'rightWrist', pinsRight: true },
      { id: 'backpack', parent: 'thrower', jointName: 'spine', pinsRight: false },
      { id: 'other-cup', parent: 'receiver', jointName: 'rightWrist', pinsRight: false },
    ]) {
      const setup = conversationFixture();
      placeRecordedBottle(setup, -0.18);
      const group = new THREE.Group();
      const mesh = new THREE.Object3D();
      mesh.position.set(0.18, 0.8, -1.1);
      group.add(mesh);
      setup.host.scene.add(group);
      setup.host.objects.push({
        id: attachment.id,
        label: attachment.id,
        meta: {
          objectClass: 'attached',
          pose: {
            segments: [
              {
                kind: 'attached',
                fromSourceFrame: 0,
                toSourceFrame: 40,
                parent: attachment.parent,
                jointName: attachment.jointName,
              },
            ],
          },
        },
        track: { fps: 10 },
        group,
        mesh,
        interactionOwned: false,
        samplePosition: () => new THREE.Vector3(0.18, 0.8, -1.1),
      });
      setup.host.scene.updateMatrixWorld(true);
      expect(setup.runtime.interrupt('approached', 'thrower')).toBe(true);
      setup.tick();
      const [left, right] = armFreedom(setup.runtime);
      expect(left).toBe(0);
      if (attachment.pinsRight) expect(right).toBe(0);
      else expect(right).toBeGreaterThan(0.99);
    }
  });
});

describe('BottleScene conversation animation lifecycle', () => {
  test('normal playback and the explicit animation opt-out leave source meshes alone', () => {
    for (const disabled of [false, true]) {
      const setup = conversationFixture(new URLSearchParams(disabled ? 'interactAnimation=0' : ''));
      setup.tick();
      expect(setup.host.playing()).toBe(true);
      expect(setup.runtime.snapshot().animations.every((item) => item.pose === null)).toBe(true);
      expect(setup.meshUpdates.get('thrower')).toEqual({ generator: 0, version: 0 });
      if (disabled) {
        expect(setup.runtime.interrupt('approached', 'thrower')).toBe(true);
        setup.tick();
        expect(setup.host.playing()).toBe(false);
        expect(animationPose(setup.runtime)).toBe(null);
        expect(modifiers(setup.people[0])).toHaveLength(0);
        expect(setup.meshUpdates.get('thrower')).toEqual({ generator: 0, version: 0 });
      }
    }
  });

  test('selected paused person listens, speaks and listens again without moving recorded state', () => {
    const { runtime, people, frame, tick, host, meshUpdates, bottle } = conversationFixture();
    const voice = voiceFixture(runtime);
    expect(runtime.interrupt('approached', 'thrower')).toBe(true);
    tick();
    expect(animationPose(runtime)).toMatchObject({ mode: 'listening', weight: 1 });
    expect(animationPose(runtime, 'receiver')).toBe(null);
    expect(modifiers(people[0])).toHaveLength(1);
    expect(meshUpdates.get('thrower')!.generator).toBe(1);
    expect(meshUpdates.get('thrower')!.version).toBeGreaterThan(0);
    const pausedTime = host.time();
    const groupPosition = people[0].group.getWorldPosition(new THREE.Vector3()).toArray();
    const groupRotation = people[0].group.getWorldQuaternion(new THREE.Quaternion()).toArray();
    const bodyScale = people[0].group.scale.toArray();
    const bottleState = runtime.snapshot().bottle;
    const listening = animationPose(runtime)!;
    voice.speaking(true);
    frame([], 0);
    expect(animationPose(runtime)).toEqual({ ...listening, mode: 'speaking' });
    tick();
    const speaking = animationPose(runtime)!;
    expect(speaking.mode).toBe('speaking');
    expect(speaking.pitch).not.toBe(listening.pitch);
    voice.speaking(false);
    frame([], 0);
    expect(animationPose(runtime)).toEqual({ ...speaking, mode: 'listening' });
    tick();
    expect(animationPose(runtime)).toMatchObject({ mode: 'listening', weight: 1 });
    expect(host.time()).toBe(pausedTime);
    expect(host.playing()).toBe(false);
    expect(people[0].group.getWorldPosition(new THREE.Vector3()).toArray()).toEqual(groupPosition);
    expect(people[0].group.getWorldQuaternion(new THREE.Quaternion()).toArray()).toEqual(
      groupRotation,
    );
    expect(people[0].group.scale.toArray()).toEqual(bodyScale);
    expect(people[0].frame).toBe(0);
    expect(runtime.snapshot().bottle).toEqual(bottleState);
    expect(bottle.interactionOwned).toBe(true);
  });

  test('addressing another person fades the previous overlay out without transferring the bottle', () => {
    const { runtime, people, tick, frame } = conversationFixture();
    people[0].group.position.set(1.5, 0, -1);
    people[1].group.position.set(0, 0, -1);
    frame();
    expect(runtime.interrupt('approached', 'thrower')).toBe(true);
    tick();
    const owner = runtime.snapshot().bottle.personOwner;
    expect(animationPose(runtime)!.weight).toBe(1);
    expect(speechStarted(runtime)).toBe(true);
    expect(runtime.snapshot().activePersonId).toBe('receiver');
    frame();
    expect(animationPose(runtime)!.mode).toBe('off');
    expect(animationPose(runtime)!.weight).toBeGreaterThan(0);
    expect(animationPose(runtime)!.weight).toBeLessThan(1);
    expect(animationPose(runtime, 'receiver')!.mode).toBe('listening');
    tick();
    expect(animationPose(runtime)).toEqual(zeroAnimation);
    expect(animationPose(runtime, 'receiver')!.weight).toBe(1);
    expect(runtime.snapshot().bottle.personOwner).toBe(owner);
  });

  test('replay removes the overlay and exactly restores the recorded transform', () => {
    const { runtime, people, tick, frame, host } = conversationFixture();
    const position = people[0].group.position.toArray();
    const rotation = people[0].group.quaternion.toArray();
    expect(runtime.interrupt('approached', 'thrower')).toBe(true);
    tick();
    expect(modifiers(people[0])).toHaveLength(1);
    runtime.replay();
    expect(animationPose(runtime)).toBe(null);
    expect(modifiers(people[0])).toHaveLength(0);
    expect(people[0].group.position.toArray()).toEqual(position);
    expect(people[0].group.quaternion.toArray()).toEqual(rotation);
    expect(host.time()).toBe(0);
    expect(host.playing()).toBe(true);
    frame();
    expect(animationPose(runtime)).toBe(null);
    expect(modifiers(people[0])).toHaveLength(0);
  });

  test('starting source transport clears the overlay before a new recorded frame renders', () => {
    const { runtime, tick, frame, host } = conversationFixture();
    expect(runtime.interrupt('approached', 'thrower')).toBe(true);
    tick();
    expect(animationPose(runtime)!.weight).toBe(1);
    host.play(true);
    expect(animationPose(runtime)).toEqual(zeroAnimation);
    frame();
    expect(animationPose(runtime)).toEqual(zeroAnimation);
  });

  test('session exit and hidden XR/document reset immediately, then fade in from zero on return', () => {
    for (const lifecycle of ['session', 'visibility', 'document'] as const) {
      const { runtime, tick, frame, host, meshUpdates } = conversationFixture();
      expect(runtime.interrupt('approached', 'thrower')).toBe(true);
      tick();
      const updates = meshUpdates.get('thrower')!.version;
      if (lifecycle === 'session') runtime.sessionEnd();
      else if (lifecycle === 'document') {
        Object.defineProperty(document, 'hidden', { configurable: true, value: true });
        runtime.visibilityChanged(true);
      } else runtime.visibilityChanged(false);
      expect(animationPose(runtime)).toEqual(zeroAnimation);
      expect(meshUpdates.get('thrower')!.version).toBeGreaterThan(updates);
      tick();
      expect(animationPose(runtime)).toEqual(zeroAnimation);
      if (lifecycle === 'session') runtime.sessionStart();
      else {
        Object.defineProperty(document, 'hidden', { configurable: true, value: false });
        runtime.visibilityChanged(true);
      }
      expect(animationPose(runtime)).toEqual(zeroAnimation);
      frame();
      expect(animationPose(runtime)!.mode).toBe('listening');
      expect(animationPose(runtime)!.weight).toBeGreaterThan(0);
      expect(animationPose(runtime)!.weight).toBeLessThan(0.02);
      expect(host.playing()).toBe(false);
      expect(host.time()).toBe(0.5);
    }
  });

  test('a hidden person resets immediately and reappears from a neutral pose', () => {
    const { runtime, people, tick, frame } = conversationFixture();
    expect(runtime.interrupt('approached', 'thrower')).toBe(true);
    tick();
    people[0].group.visible = false;
    frame();
    expect(animationPose(runtime)).toEqual(zeroAnimation);
    people[0].group.visible = true;
    frame();
    expect(animationPose(runtime)!.weight).toBeGreaterThan(0);
    expect(animationPose(runtime)!.weight).toBeLessThan(0.02);
  });

  test('leaving conversation range fades out without changing the paused recording or owner', () => {
    const { runtime, head, tick, frame, host } = conversationFixture();
    expect(runtime.interrupt('approached', 'thrower')).toBe(true);
    tick();
    const owner = runtime.snapshot().bottle.personOwner;
    head.z = 3;
    frame();
    expect(animationPose(runtime)!.mode).toBe('off');
    expect(animationPose(runtime)!.weight).toBeGreaterThan(0);
    expect(animationPose(runtime)!.weight).toBeLessThan(1);
    tick();
    expect(animationPose(runtime)).toEqual(zeroAnimation);
    expect(host.time()).toBe(0.5);
    expect(host.playing()).toBe(false);
    expect(runtime.snapshot().bottle.personOwner).toBe(owner);
  });
});
