import test from 'node:test';
import assert from 'node:assert/strict';
import { createStanceResolver } from '../src/person-stance.ts';

// Small drift while planted, a swing, then another planted interval.
const feet = [0, 0.01, 0.02, 0.08, 0.09, 0.1].map((x) => [0, x, 0]);

test('direct and backward seeks match fully measured sequential playback', () => {
  const sequential = createStanceResolver((frame) => feet[frame]);
  const expected = feet.map((_, frame) => sequential(frame, 1, 1));
  assert.deepEqual(expected.slice(0, 3), [
    [0, 0],
    [-0.01, 0],
    [-0.02, 0],
  ]);
  assert.equal(expected[3][0], -0.012); // Preserve the existing swing decay.
  const measured: number[] = [];
  const seek = createStanceResolver((frame) => {
    measured.push(frame);
    return feet[frame];
  });
  for (const frame of [5, 2, 4, 0, 3, 1]) assert.deepEqual(seek(frame, 1, 1), expected[frame]);
  assert.deepEqual([...new Set(measured)], [0, 1, 2, 3, 4, 5]);
});

test('out-of-order arrivals cannot permanently cache missing predecessor feet', () => {
  const complete = createStanceResolver((frame) => feet[frame]);
  const expected = feet.map((_, frame) => complete(frame, 1, 1));
  const available = new Set([0, 4, 5]);
  const streaming = createStanceResolver((frame) =>
    available.has(frame) ? feet[frame] : undefined,
  );
  assert.deepEqual(streaming(5, 1, 1), [0, 0]);
  available.add(1);
  available.add(3);
  assert.deepEqual(streaming(4, 1, 1), [0, 0]);
  assert.deepEqual(streaming(1, 1, 1), expected[1]);
  available.add(2);
  for (const frame of [5, 3, 2, 4, 1, 0]) assert.deepEqual(streaming(frame, 1, 1), expected[frame]);
});

test('stance keeps its clamp and recomputes when placement units change', () => {
  const drift = Array.from({ length: 15 }, (_, frame) => [0, frame * 0.02, 0]);
  const resolve = createStanceResolver((frame) => drift[frame]);
  assert.equal(resolve(14, 1, 1)[0], -0.1);
  const rescaled = resolve(14, 2, 1);
  assert.deepEqual(rescaled, [0, 0]); // This speed is now above the planted threshold.
  assert.deepEqual(rescaled, createStanceResolver((frame) => drift[frame])(14, 2, 1));
});
