import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizePlaybackTime, sampleAnimationFrame } from '../src/playback-time.ts';

test('preserves exact nonnegative in-range timestamps and adjacent boundaries', () => {
  const duration = 9;
  const before = 1 / 15;
  const after = 2 / 15;
  assert.equal(normalizePlaybackTime(before, duration), before);
  assert.equal(normalizePlaybackTime(after, duration), after);
  const timestamps = [0, before, after];
  const frameAt = (value: number) => timestamps.filter((time) => time <= value).length - 1;
  assert.equal(frameAt(normalizePlaybackTime(before, duration)), 1);
  assert.equal(frameAt(normalizePlaybackTime(before - Number.EPSILON, duration)), 0);
  assert.equal(normalizePlaybackTime(0, duration), 0);
  assert.equal(normalizePlaybackTime(duration, duration), 0);
});

test('wraps out-of-range positive and negative times without negative zero', () => {
  assert.equal(normalizePlaybackTime(10, 9), 1);
  assert.equal(normalizePlaybackTime(-1, 9), 8);
  assert.equal(normalizePlaybackTime(18, 9), 0);
  assert.equal(Object.is(normalizePlaybackTime(-9, 9), -0), false);
  assert.equal(Object.is(normalizePlaybackTime(-0, 9), -0), false);
  assert.equal(normalizePlaybackTime(-Number.MIN_VALUE, 9), 0);
});

test('maps invalid values and invalid durations to a safe origin', () => {
  assert.equal(normalizePlaybackTime(Number.NaN, 9), 0);
  assert.equal(normalizePlaybackTime(Infinity, 9), 0);
  assert.equal(normalizePlaybackTime(2, 0), 0);
  assert.equal(normalizePlaybackTime(2, Number.NaN), 0);
});

test('holds the last recorded pose until the playback clock actually loops', () => {
  const times = [0, 2 / 30, 5 / 30, 8 / 30];
  const duration = 10 / 30;
  for (const time of [times.at(-1)!, 9 / 30, duration - 1e-8]) {
    assert.deepEqual(sampleAnimationFrame(time, times), [3, 0]);
  }
  assert.deepEqual(sampleAnimationFrame(normalizePlaybackTime(duration, duration), times), [0, 0]);
});

test('interpolates actual sample times and clamps before the first observation', () => {
  const times = [1, 1.1, 1.4];
  assert.deepEqual(sampleAnimationFrame(0.5, times), [0, 0]);
  assert.deepEqual(sampleAnimationFrame(1.1, times), [1, 0]);
  const [i, u] = sampleAnimationFrame(1.25, times);
  assert.equal(i, 1);
  assert.ok(Math.abs(u - 0.5) < 1e-12);
  assert.deepEqual(sampleAnimationFrame(1.8, times), [2, 0]);
  assert.deepEqual(sampleAnimationFrame(1.8, [1.2]), [0, 0]);
});
