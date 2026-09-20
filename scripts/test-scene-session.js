import { afterEach, beforeEach, expect, test } from 'bun:test';
import { createSceneSession } from '../src/scene-session.js';

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
};
const flush = async () => {
  for (let i = 0; i < 30; i++) await Promise.resolve();
};
class Element {
  children = [];
  style = {};
  hidden = false;
  append(child) {
    child.remove();
    this.children.push(child);
    child.parent = this;
  }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter((node) => node !== this);
    this.parent = null;
  }
  querySelectorAll() {
    return [];
  }
  setAttribute(name, value) {
    this[name] = value;
  }
}
let originals;
let sessions;
beforeEach(() => {
  originals = new Map(
    ['window', 'document', 'location', 'history'].map((key) => [
      key,
      Object.getOwnPropertyDescriptor(globalThis, key),
    ]),
  );
  sessions = [];
  const body = new Element();
  Object.assign(globalThis, {
    window: new EventTarget(),
    document: {
      body,
      createElement: () => new Element(),
      getElementById: (id) => body.children.find((node) => node.id === id),
    },
    location: { href: 'http://fixture/fourd.html?demo=A' },
    history: { replaceState: (_state, _unused, url) => (location.href = String(url)) },
  });
});
afterEach(() => {
  for (const session of sessions) session.dispose();
  for (const [key, descriptor] of originals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  }
});
async function fixture(playing = true, options = {}) {
  const requests = [];
  const events = [];
  const live = new Set();
  let maxLive = 0;
  window.addEventListener('wander:scenechange', (event) => events.push(event.detail));
  const session = createSceneSession({
    renderer: {},
    template: { content: { cloneNode: () => new Element() } },
    initial: new URLSearchParams({ demo: 'A', ...options }),
    clips: ['A', 'B', 'C', 'D'].map((id) => ({ id })),
    createRuntime({ search, scope }) {
      const id = search.get('demo');
      const preparation = deferred();
      const activations = [];
      const runtime = {
        api: {
          demo: id,
          ready: true,
          people: [],
          playing: false,
          audioState: { muted: true },
          play(value) {
            this.playing = value;
          },
          setTime() {},
          setMuted() {},
        },
        stages: { id },
        activateCalls: 0,
        deactivateCalls: 0,
        async activate() {
          runtime.activateCalls++;
          const action = activations.shift();
          if (action) await action();
        },
        deactivate() {
          runtime.deactivateCalls++;
        },
      };
      const request = { id, search, scope, preparation, runtime, activations };
      requests.push(request);
      live.add(request);
      maxLive = Math.max(maxLive, live.size);
      scope.onDispose(() => live.delete(request));
      if (id === 'A') preparation.resolve(runtime);
      return preparation.promise;
    },
  });
  sessions.push(session);
  await session.ready;
  window.wander.play(playing);
  return {
    session,
    requests,
    events,
    live,
    maxLive: () => maxLive,
    latest: () => requests.at(-1),
    resolve: () => requests.at(-1).preparation.resolve(requests.at(-1).runtime),
  };
}

test('scene switches retain the interaction opt-in without leaking scene-specific placement', async () => {
  const f = await fixture(true, { interact: '1', interactObject: 'prop', pos: '1,2,3' });
  const switched = f.session.select('B');
  expect(f.latest().search.get('interact')).toBe('1');
  expect(f.latest().search.get('interactObject')).toBe('prop');
  expect(f.latest().search.has('pos')).toBe(false);
  f.resolve();
  await switched;
});

for (const outcome of ['resolve', 'reject']) {
  test(`superseding stalled preparation disposes immediately and ignores late ${outcome}`, async () => {
    const f = await fixture();
    const bDone = f.session.select('B');
    const b = f.latest();
    const cDone = f.session.select('C');
    const c = f.latest();
    expect(b.scope.abort.signal.aborted).toBe(true);
    expect(f.live.has(b)).toBe(false);
    expect(window.__loadingScene.id).toBe('C');
    const count = f.events.length;
    if (outcome === 'resolve') b.preparation.resolve(b.runtime);
    else b.preparation.reject(new Error('obsolete failure'));
    await bDone;
    await flush();
    expect(window.wander.demo).toBe('A');
    expect(window.__loadingScene.scope).toBe(c.scope);
    expect(f.events.length).toBe(count);
    expect(window.__wanderStartupError).toBe(false);
    f.resolve();
    await cDone;
    expect(window.wander.demo).toBe('C');
    expect(f.maxLive()).toBe(2);
  });
}

test('selecting current cancels a stalled load and restores playback', async () => {
  const f = await fixture();
  const pending = f.session.select('B');
  const b = f.latest();
  await f.session.select('A');
  await pending;
  expect(b.scope.disposed).toBe(true);
  expect(window.wander.demo).toBe('A');
  expect(window.wander.playing).toBe(true);
  expect(window.__loadingScene).toBe(null);
  expect(f.events.at(-1).cancelled).toBe(true);
});

for (const playing of [true, false]) {
  test(`preparation failure restores prior playing=${playing}`, async () => {
    const f = await fixture(playing);
    const done = f.session.select('B');
    f.latest().preparation.reject(new Error('prepare failed'));
    await expect(done).rejects.toThrow('prepare failed');
    expect(window.wander.demo).toBe('A');
    expect(window.wander.playing).toBe(playing);
    expect(window.__loadingScene).toBe(null);
    expect(f.events.at(-1).error).toBe('prepare failed');
    expect(f.live.size).toBe(1);
  });
  test(`activation failure rolls back and restores prior playing=${playing}`, async () => {
    const f = await fixture(playing);
    const a = f.latest();
    const done = f.session.select('B');
    f.latest().activations.push(() => Promise.reject(new Error('activate failed')));
    f.resolve();
    await expect(done).rejects.toThrow('activate failed');
    expect(window.wander).toBe(a.runtime.api);
    expect(window.wander.playing).toBe(playing);
    expect(a.runtime.activateCalls).toBe(2);
    expect(window.__loadingScene).toBe(null);
    expect(f.live.size).toBe(1);
  });
}

test('failed rollback clears pending, reports error, and permits retrying current', async () => {
  const f = await fixture();
  const a = f.latest();
  a.activations.push(() => Promise.reject(new Error('restore failed')));
  const done = f.session.select('B');
  f.latest().activations.push(() => Promise.reject(new Error('activate failed')));
  f.resolve();
  await expect(done).rejects.toThrow('previous scene could not resume');
  expect(window.__loadingScene).toBe(null);
  expect(f.events.at(-1).error).toContain('previous scene could not resume');
  const retry = f.session.select('A');
  f.resolve();
  await retry;
  expect(window.wander.demo).toBe('A');
  expect(window.__loadingScene).toBe(null);
  expect(f.events.at(-1).error).toBeUndefined();
});

test('cancellation during activation waits for rollback before latest activation', async () => {
  const f = await fixture();
  const bDone = f.session.select('B');
  const b = f.latest();
  const gate = deferred();
  b.activations.push(() => gate.promise);
  f.resolve();
  await flush();
  expect(b.runtime.activateCalls).toBe(1);
  const cDone = f.session.select('C');
  const c = f.latest();
  f.resolve();
  await flush();
  expect(b.scope.disposed).toBe(true);
  expect(c.runtime.activateCalls).toBe(0);
  expect(window.__loadingScene.id).toBe('C');
  gate.resolve();
  await Promise.all([bDone, cDone]);
  expect(c.runtime.activateCalls).toBe(1);
  expect(window.wander.demo).toBe('C');
  expect(f.events.filter((event) => event.id === 'B' && !event.loading)).toEqual([]);
  expect(window.__loadingScene).toBe(null);
});

test('choosing current during a failed rollback rebuilds it instead of reporting a false recovery', async () => {
  const f = await fixture();
  const a = f.latest();
  a.activations.push(() => Promise.reject(new Error('restore failed')));
  const bDone = f.session.select('B');
  const gate = deferred();
  f.latest().activations.push(() => gate.promise);
  f.resolve();
  await flush();
  const back = f.session.select('A');
  gate.resolve();
  await Promise.all([bDone, back]);
  expect(f.requests.map((request) => request.id)).toEqual(['A', 'B', 'A']);
  expect(window.wander).not.toBe(a.runtime.api);
  expect(window.wander.demo).toBe('A');
  expect(window.__loadingScene).toBe(null);
});

test('cache return reuses runtime and third scene never exceeds current plus spare', async () => {
  const f = await fixture();
  const a = f.latest();
  let done = f.session.select('B');
  f.resolve();
  await done;
  expect(window.__sceneCache.retained).toEqual(['A']);
  await f.session.select('A');
  expect(window.wander).toBe(a.runtime.api);
  expect(f.requests.length).toBe(2);
  expect(window.__sceneCache.hits).toBe(1);
  expect(window.__sceneCache.retained).toEqual(['B']);
  const b = f.requests[1];
  done = f.session.select('C');
  expect(b.scope.disposed).toBe(true);
  f.resolve();
  await done;
  expect(window.__sceneCache.retained).toEqual(['A']);
  expect(f.live.size).toBe(2);
  expect(f.maxLive()).toBe(2);
});

for (const phase of ['preparation', 'activation']) {
  test(`dispose during ${phase} cancels resources and prevents late publication`, async () => {
    const f = await fixture();
    const done = f.session.select('B');
    const b = f.latest();
    const gate = deferred();
    if (phase === 'activation') {
      b.activations.push(() => gate.promise);
      f.resolve();
      await flush();
      expect(b.runtime.activateCalls).toBe(1);
    }
    const count = f.events.length;
    const url = location.href;
    f.session.dispose();
    expect(f.live.size).toBe(0);
    expect(window.__loadingScene).toBe(null);
    expect(b.scope.abort.signal.aborted).toBe(true);
    b.preparation.resolve(b.runtime);
    gate.resolve();
    await done;
    await flush();
    expect(f.events.length).toBe(count);
    expect(location.href).toBe(url);
    expect(window.wander.demo).toBe('A');
    await f.session.select('C');
    expect(f.requests.length).toBe(2);
  });
}
