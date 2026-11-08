import { expect, test } from 'bun:test';
import { Vector3 } from 'three';
import { applyPlacementOffset, stanceCorrectionEnabled } from '../src/person-placement.ts';

test('per-person offsets use sequence indices and world units independently of body scale', () => {
  const base = new Vector3(2, -1, 4);
  const offsets = [
    [0.1, 0.2, -0.3],
    [0.5, 0.4, -0.7],
  ];
  const expected = [2.2, -0.75, 3.6];
  const position = base.clone();
  applyPlacementOffset(position, offsets, [[99, 99, 99]], 40, 0, 1, 0.25);
  position.toArray().forEach((value, i) => expect(value).toBeCloseTo(expected[i], 10));
  // A paused refresh starts from registration; offsets cannot accumulate.
  position.copy(base);
  applyPlacementOffset(position, offsets, null, 40, 0, 1, 0.25);
  position.toArray().forEach((value, i) => expect(value).toBeCloseTo(expected[i], 10));
});

test('discrete frames and legacy shared offsets retain their own sample origins', () => {
  const shared = [
    [9, 9, 9],
    [8, 8, 8],
    [1, 2, 3],
    [4, 5, 6],
  ];
  const position = new Vector3();
  applyPlacementOffset(position, undefined, shared, 2, 1, 1, 0);
  expect(position.toArray()).toEqual([4, 5, 6]);
  position.set(0, 0, 0);
  applyPlacementOffset(position, undefined, shared, 2, 0, 1, 0.5);
  expect(position.toArray()).toEqual([2.5, 3.5, 4.5]);
});

test('an absent next sample holds its recorded offset and no placement preserves registration', () => {
  const position = new Vector3(1, 1, 1);
  applyPlacementOffset(position, [[2, 3, 4]], null, 0, 0, 1, 0.9);
  expect(position.toArray()).toEqual([3, 4, 5]);
  applyPlacementOffset(position, undefined, null, 0, 0, 1, 0.5);
  expect(position.toArray()).toEqual([3, 4, 5]);
});

test('measured trajectories avoid additional stance shifts unless explicitly requested', () => {
  expect(stanceCorrectionEnabled(null, true, false, true)).toBe(false);
  expect(stanceCorrectionEnabled('1', true, false, true)).toBe(true);
  expect(stanceCorrectionEnabled('0', true, false, false)).toBe(false);
  expect(stanceCorrectionEnabled(null, true, false, false)).toBe(true);
  expect(stanceCorrectionEnabled('1', false, false, true)).toBe(false);
  expect(stanceCorrectionEnabled('1', true, true, true)).toBe(false);
});
