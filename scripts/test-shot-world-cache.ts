import { expect, test } from 'bun:test';
import { createShotWorldCache } from '../src/shot-world-cache.ts';

interface Fake {
  index: number;
  released: boolean;
}

function harness(limit = 4, delay = 0) {
  const loads: number[] = [];
  const releases: number[] = [];
  const cache = createShotWorldCache<Fake>({
    limit,
    load: async (index) => {
      loads.push(index);
      if (delay) await new Promise((r) => setTimeout(r, delay));
      return { index, released: false };
    },
    release: (value, index) => {
      value.released = true;
      releases.push(index);
    },
  });
  return { cache, loads, releases };
}

test('a shot`s state is built once and kept, keyed by its index', async () => {
  const { cache, loads } = harness();
  const first = await cache.ensure(2);
  expect(first.index).toBe(2);
  expect(await cache.ensure(2)).toBe(first);
  expect(loads).toEqual([2]);
  expect(cache.peek(2)).toBe(first);
  expect(cache.peek(3)).toBeUndefined();
});

test('concurrent activations of the same shot share one load', async () => {
  const { cache, loads } = harness(4, 5);
  const [a, b, c] = await Promise.all([cache.ensure(1), cache.ensure(1), cache.keep(1, 0)]);
  expect(loads).toEqual([1]);
  expect(b).toBe(a);
  expect(c).toBe(a);
});

test('residency never exceeds the limit and the active shot is never evicted', async () => {
  const { cache, releases } = harness(2);
  for (const index of [0, 1, 2, 3]) {
    await cache.keep(index, 0, 4);
    expect(cache.resident().length).toBeLessThanOrEqual(2);
    expect(cache.peek(index)).toBeDefined(); // the world being rendered stays in memory
  }
  expect(releases).toEqual([0, 1]); // least recently active first
});

test('an evicted shot is released exactly once and rebuilt when it comes back', async () => {
  const { cache, loads, releases } = harness(2);
  const zero = await cache.keep(0, 0, 3);
  await cache.keep(1, 0, 3);
  await cache.keep(2, 0, 3);
  expect(zero.released).toBe(true);
  expect(releases).toEqual([0]);
  const again = await cache.keep(0, 0, 3);
  expect(again).not.toBe(zero);
  expect(again.released).toBe(false);
  expect(loads).toEqual([0, 1, 2, 0]);
});

test('keep starts the lookahead without waiting for it', async () => {
  const { cache, loads } = harness(4, 5);
  const active = await cache.keep(0, 1, 5);
  expect(active.index).toBe(0);
  expect(loads).toEqual([0, 1]); // the next shot is already loading at the moment of the cut
  expect(cache.pending(1)).toBe(true);
  expect(cache.peek(1)).toBeUndefined();
  await cache.ensure(1);
  expect(cache.pending(1)).toBe(false);
});

test('the lookahead stops at the end of the sequence and inside the limit', async () => {
  const { cache, loads } = harness(2);
  await cache.keep(4, 2, 5); // shot 4 is the last of five
  expect(loads).toEqual([4]);
  const second = harness(2);
  await second.cache.keep(0, 3, 10); // limit 2: one active plus one lookahead
  expect(second.loads).toEqual([0, 1]);
});

test('seeking backwards over a cut reuses a resident world', async () => {
  const { cache, loads } = harness(4);
  await cache.keep(0, 1, 3);
  await cache.keep(1, 1, 3);
  await cache.keep(0, 1, 3); // the viewer seeks back into shot 0
  expect(loads.filter((i) => i === 0)).toHaveLength(1);
  expect(cache.resident()[0]).toBe(0); // most recently used first
});

test('clear releases everything, including a load still in flight', async () => {
  const { cache, releases } = harness(4, 10);
  await cache.keep(0, 0, 4);
  const inFlight = cache.ensure(1);
  cache.clear();
  expect(releases).toEqual([0]);
  await inFlight;
  expect(releases).toEqual([0, 1]);
  expect(cache.peek(1)).toBeUndefined();
  expect(cache.resident()).toEqual([]);
});

test('a failed world load is not cached and does not wedge the cache', async () => {
  let attempts = 0;
  const cache = createShotWorldCache<Fake>({
    limit: 2,
    load: async (index) => {
      attempts++;
      if (attempts === 1) throw new Error('spz did not load');
      return { index, released: false };
    },
  });
  await expect(cache.ensure(0)).rejects.toThrow(/did not load/);
  expect(cache.peek(0)).toBeUndefined();
  expect(cache.pending(0)).toBe(false);
  expect((await cache.ensure(0)).index).toBe(0);
});
