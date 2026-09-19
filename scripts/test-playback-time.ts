import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizePlaybackTime } from '../src/playback-time.ts';

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
