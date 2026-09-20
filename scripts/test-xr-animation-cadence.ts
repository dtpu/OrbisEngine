import assert from 'node:assert/strict';
import test from 'node:test';
import { createAnimationCadence } from '../src/xr/animation-cadence';

test('recorded animation runs at 36 Hz while headset frames run at 72 or 90 Hz', () => {
  for (const hz of [72, 90]) {
    const due = createAnimationCadence();
    let count = 0;
    for (let i = 0; i < hz * 10; i++) if (due((i * 1000) / hz, true)) count++;
    assert.ok(Math.abs(count - 360) <= 1, `${hz} Hz produced ${count} animation updates`);
  }
});

test('stalls update immediately once without a burst of stale animation work', () => {
  const due = createAnimationCadence();
  assert.equal(due(0, true), true);
  assert.equal(due(10, true), false);
  assert.equal(due(5000, true), true);
  assert.equal(due(5001, true), false);
  assert.equal(due(5030, true), true);
});

test('desktop updates every frame and VR re-entry starts immediately', () => {
  const due = createAnimationCadence();
  assert.equal(due(0, true), true);
  assert.equal(due(1, false), true);
  assert.equal(due(2, false), true);
  assert.equal(due(3, true), true);
  assert.equal(due(4, true), false);
});

test('zero opts out of the animation cap', () => {
  const due = createAnimationCadence(0);
  assert.equal(due(1, true), true);
  assert.equal(due(2, true), true);
});
