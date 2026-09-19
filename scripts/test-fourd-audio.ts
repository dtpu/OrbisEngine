import { afterEach, test } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {
  validateManifest,
  validateAnchor,
  sampleAnchor,
  FourDAudio,
  type AudioManifest,
  type Track,
} from '../src/audio/fourd-audio.ts';
// Restore browser mocks even when an assertion fails, so other suites use the real fetch.
const browserGlobals = ['document', 'location', 'AudioContext', 'fetch'] as const;
const originalGlobals = new Map(
  browserGlobals.map((name) => [name, Object.getOwnPropertyDescriptor(globalThis, name)]),
);
afterEach(() => {
  for (const name of browserGlobals) {
    const descriptor = originalGlobals.get(name);
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else Reflect.deleteProperty(globalThis, name);
  }
});

const manifest = (): AudioManifest & { tracks: Track[] } => ({
  schema: 'wander.audio/1',
  source: { hasAudio: false },
  timeline: { durationSeconds: 10 },
  original: { url: 'original.wav' },
  spatialMixComplete: true,
  defaultMode: 'spatial',
  tracks: [
    {
      id: 'a',
      kind: 'dialogue',
      url: 'a.wav',
      personId: 'a',
      reviewed: true,
      provenance: 'owned-synthetic-fixture',
      anchor: { url: 'a.json', space: 'person-local' },
    },
    {
      id: 'b',
      kind: 'dialogue',
      url: 'b.wav',
      personId: 'b',
      reviewed: true,
      provenance: 'owned-synthetic-fixture',
      anchor: { url: 'b.json', space: 'person-local' },
    },
    { id: 'bed', kind: 'ambience', url: 'bed.wav' },
  ],
});
test('rejects mismatched duration, unreviewed identities and malformed anchors', () => {
  assert.throws(() => validateManifest(manifest(), 8, ['a', 'b']));
  const m = manifest();
  m.tracks[0].reviewed = false;
  assert.throws(() => validateManifest(m, 10, ['a', 'b']));
  assert.throws(() => validateManifest(manifest(), 10, ['a']));
  assert.throws(() =>
    validateAnchor(
      {
        positions: [
          [0, 1, 0],
          [0, 1, 1],
        ],
        times: [1, 0],
      },
      [],
    ),
  );
  assert.throws(() => validateAnchor({ positions: [[NaN, 1, 0]] }, [0]));
});
test('head anchor interpolates in the character coordinate frame', () => {
  const out = sampleAnchor(
    [
      [0, 1, 0],
      [2, 1, 0],
    ],
    [0, 2],
    1,
    new THREE.Vector3(),
  );
  const group = new THREE.Group();
  group.position.set(4, 0, 2);
  group.scale.setScalar(2);
  group.updateMatrixWorld(true);
  out.applyMatrix4(group.matrixWorld);
  assert.deepEqual(out.toArray(), [6, 2, 2]);
});
class Element extends EventTarget {
  style = {};
  children: Element[] = [];
  hidden = false;
  parentElement: Element | null = null;
  id = '';
  textContent = '';
  href?: string;
  append(...children: Element[]) {
    this.children.push(...children);
    for (const child of children) child.parentElement = this;
  }
  setAttribute() {}
  remove() {}
}
class Param {
  value = 0;
  cancelScheduledValues() {}
  setTargetAtTime(value: number) {
    this.value = value;
  }
}
class Node {
  connections: Node[] = [];
  startAt?: number;
  offset?: number;
  stopped = false;
  gain = new Param();
  playbackRate = new Param();
  positionX = new Param();
  positionY = new Param();
  positionZ = new Param();
  connect(node: Node) {
    this.connections.push(node);
    return node;
  }
  disconnect() {
    this.connections = [];
  }
  start(when: number, offset: number) {
    this.startAt = when;
    this.offset = offset;
  }
  stop() {
    this.stopped = true;
  }
}
class Context {
  currentTime = 0;
  state = 'suspended';
  destination = new Node();
  sources: Node[] = [];
  media: Node[] = [];
  listener = Object.fromEntries(
    ['position', 'forward', 'up'].flatMap((k) => ['X', 'Y', 'Z'].map((a) => [k + a, new Param()])),
  );
  createGain() {
    return new Node();
  }
  createPanner() {
    return new Node();
  }
  createMediaElementSource() {
    const n = new Node();
    this.media.push(n);
    return n;
  }
  createBufferSource() {
    const n = new Node();
    this.sources.push(n);
    return n;
  }
  async decodeAudioData() {
    return { duration: 10, numberOfChannels: 1 };
  }
  async resume() {
    this.state = 'running';
  }
  async close() {
    this.state = 'closed';
  }
}
function setup(m: AudioManifest = manifest(), fail = '') {
  const document = { createElement: () => new Element(), body: new Element() };
  Reflect.set(globalThis, 'document', document);
  Reflect.set(globalThis, 'location', { href: 'http://localhost/fourd.html' });
  let context!: Context;
  Reflect.set(
    globalThis,
    'AudioContext',
    class extends Context {
      constructor() {
        super();
        context = this;
      }
    },
  );
  Reflect.set(globalThis, 'fetch', async (url: string | URL) => {
    const name = String(url);
    return {
      ok: !name.includes(fail || 'never-fail'),
      status: 200,
      json: async () =>
        name.endsWith('audio.json')
          ? m
          : {
              positions: [
                [0, 1, 0],
                [2, 1, 0],
              ],
              times: [0, 10],
            },
      arrayBuffer: async () => new ArrayBuffer(32),
    };
  });
  const video = Object.assign(new Element(), {
    paused: false,
    muted: true,
    seeking: false,
    readyState: 4,
    playbackRate: 1,
    currentTime: 0,
    pause() {
      this.paused = true;
    },
  });
  const people = ['a', 'b'].map((id, i) => {
    const group = new THREE.Group();
    group.position.x = i * 2;
    return { id, group, ts: [0, 10], bodyH: 1 };
  });
  const audio = new FourDAudio(video as unknown as HTMLVideoElement, 10, people, () => 1);
  return { audio, video, document, context: () => context };
}
test('overlapping speakers share a schedule, follow moving heads, and replace original', async () => {
  const s = setup();
  await s.audio.load('/audio.json');
  await s.audio.unlock();
  s.audio.tick(0, true, new THREE.Vector3(3, 1, 0), new THREE.Quaternion());
  const c = s.context();
  assert.equal(s.audio.state.activeSources, 3);
  assert.equal(s.audio.state.mode, 'spatial');
  assert.equal(c.media.length, 1);
  assert.equal(c.media[0].connections[0].gain.value, 0);
  assert.equal(c.sources[0].startAt, c.sources[1].startAt);
  assert.equal(c.sources[0].connections[0].positionX.value, 0);
  assert.equal(c.sources[1].connections[0].positionX.value, 2);
  c.currentTime = 5;
  s.video.currentTime = 5;
  s.audio.tick(5, true, new THREE.Vector3(), new THREE.Quaternion());
  assert.equal(c.sources[0].connections[0].positionX.value, 1);
  assert.equal(c.sources[1].connections[0].positionX.value, 3);
  s.audio.setMuted(true);
  await s.audio.unlock();
  assert.equal(s.audio.state.muted, true, 'later gestures must preserve user mute');
  s.audio.dispose();
  assert.equal(c.state, 'closed');
});
test('pause, seek, loop and rate cancel stale scheduling without a duplicate original', async () => {
  const s = setup();
  await s.audio.load('/audio.json');
  await s.audio.unlock();
  const c = s.context();
  const tick = (t: number) => {
    s.video.currentTime = t;
    s.audio.tick(t, true, new THREE.Vector3(), new THREE.Quaternion());
  };
  tick(2);
  const first = [...c.sources];
  s.video.dispatchEvent(new Event('seeking'));
  assert.equal(s.audio.state.activeSources, 0);
  assert.ok(first.every((n) => n.stopped));
  tick(8);
  c.currentTime += 2;
  tick(0);
  assert.ok(c.sources.slice(3, 6).every((n) => n.stopped));
  s.video.playbackRate = 2;
  s.video.dispatchEvent(new Event('ratechange'));
  tick(1);
  assert.equal(s.audio.state.mode, 'original');
  assert.equal(s.audio.state.activeSources, 1);
  assert.equal(c.media[0].connections[0].gain.value, 0, 'external original excludes embedded mix');
  s.video.paused = true;
  s.video.dispatchEvent(new Event('pause'));
  tick(1);
  assert.equal(s.audio.state.activeSources, 0);
  s.audio.dispose();
});
test('failed stem falls back to authentic original; silent manifest exposes no soundtrack', async () => {
  const s = setup(manifest(), 'b.wav');
  await s.audio.load('/audio.json');
  await s.audio.unlock();
  assert.equal(s.audio.state.spatialReady, false);
  assert.equal(s.audio.state.mode, 'original');
  s.audio.tick(0, true, new THREE.Vector3(), new THREE.Quaternion());
  assert.equal(s.audio.state.activeSources, 1);
  s.audio.dispose();
  const malformed = manifest();
  malformed.tracks[0].personId = 'unknown';
  const fallback = setup(malformed);
  await fallback.audio.load('/audio.json');
  await fallback.audio.unlock();
  fallback.audio.tick(0, true, new THREE.Vector3(), new THREE.Quaternion());
  assert.equal(fallback.audio.state.activeSources, 1);
  assert.equal(fallback.audio.state.spatialReady, false);
  fallback.audio.dispose();
  const silent = setup({
    schema: 'wander.audio/1',
    source: { hasAudio: false },
    timeline: { durationSeconds: 10 },
  });
  await silent.audio.load('/audio.json');
  assert.equal(silent.audio.state.hasAudio, false);
  assert.equal(silent.audio.button.disabled, true);
  silent.audio.dispose();
});

test('recovered silent-video manifest enables original audio and displays safe source credit', async () => {
  const m: AudioManifest & { provenance: NonNullable<AudioManifest['provenance']> } = {
    schema: 'wander.audio/1',
    source: { hasAudio: false },
    timeline: { durationSeconds: 10 },
    original: { url: 'audio/original.wav', offsetSeconds: 0 },
    defaultMode: 'original',
    provenance: {
      attribution: '(CC) Blender Foundation | mango.blender.org',
      license: 'CC BY 3.0',
      licenseUrl: 'https://creativecommons.org/licenses/by/3.0/',
    },
  };
  const s = setup(m);
  await s.audio.load('/audio.json');
  await s.audio.unlock();
  assert.equal(s.audio.state.hasAudio, true);
  assert.equal(s.audio.button.disabled, false);
  assert.equal(s.audio.state.mode, 'original');
  assert.equal(s.audio.state.spatialReady, false);
  const credit = s.document.body.children.find((e) => e.id === 'audio-credit')!;
  assert.equal(credit.hidden, false);
  assert.match(credit.textContent, /Blender Foundation/);
  assert.equal(credit.children[0].href, m.provenance.licenseUrl);
  s.audio.dispose();
  m.provenance.licenseUrl = 'javascript:alert(1)';
  const unsafe = setup(m);
  await unsafe.audio.load('/audio.json');
  const safeCredit = unsafe.document.body.children.find((e) => e.id === 'audio-credit')!;
  assert.equal(safeCredit.children[0].href, undefined);
  unsafe.audio.dispose();
});

test('a listener without AudioParams is positioned through the legacy calls, not thrown at', async () => {
  const s = setup();
  await s.audio.load('/audio.json');
  await s.audio.unlock();
  const c = s.context();
  // Firefox exposes neither the listener nor the panner AudioParams, only these two calls.
  const calls: { position?: number[]; orientation?: number[]; panner?: number[] } = {};
  c.listener = {
    setPosition: (...a: number[]) => {
      calls.position = a;
    },
    setOrientation: (...a: number[]) => {
      calls.orientation = a;
    },
  } as never;
  const created = c.createPanner.bind(c);
  c.createPanner = () => {
    const panner = created() as unknown as Record<string, unknown>;
    delete panner.positionX;
    delete panner.positionY;
    delete panner.positionZ;
    panner.setPosition = (...a: number[]) => {
      calls.panner = a;
    };
    return panner as never;
  };
  s.audio.tick(0, true, new THREE.Vector3(3, 1, 0), new THREE.Quaternion());
  assert.deepEqual(calls.position, [3, 1, 0]);
  assert.deepEqual(calls.orientation, [0, 0, -1, 0, 1, 0]);
  assert.deepEqual(calls.panner, [2, 1, 0]);
  // a second frame must not throw either, which is what made the viewer crawl
  s.audio.tick(1, true, new THREE.Vector3(0, 1, 2), new THREE.Quaternion());
  assert.deepEqual(calls.position, [0, 1, 2]);
  assert.deepEqual(calls.panner, [2.2, 1, 0]);
  assert.equal(s.audio.state.mode, 'spatial');
  s.audio.dispose();
});
