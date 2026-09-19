import test from 'node:test';
import assert from 'node:assert/strict';
import { validateBakedTrack } from '../src/object-track.ts';

function track() {
  return {
    fps: 30,
    sourceFrames: [4, 5, 6],
    sampleIndex: [0, 0.1, 0.2],
    positions: [
      [0, 1, 2],
      [1, 2, 3],
      [2, 3, 4],
    ],
    quaternionsXYZW: [
      [0, 0, 0, 1],
      [0, 0, 0, 1],
      [0, 0, 0, 1],
    ],
    visible: [true, false, true],
  };
}

test('accepts a finite contiguous baked track', () => {
  const result = validateBakedTrack(track());
  assert.equal(result.ok, true);
});

test('rejects gaps, mismatched arrays, nonfinite values, and invalid visibility', () => {
  const cases = [
    { sourceFrames: [4, 6, 7], reason: /contiguous/ },
    { sourceFrames: [-1, 0, 1], reason: /nonnegative/ },
    { positions: [[0, 1, 2]], reason: /matching lengths/ },
    {
      positions: [
        [0, NaN, 2],
        [1, 2, 3],
        [2, 3, 4],
      ],
      reason: /finite xyz/,
    },
    {
      quaternionsXYZW: [
        [0, 0, 0, Infinity],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
      ],
      reason: /finite xyzw/,
    },
    {
      quaternionsXYZW: [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
      ],
      reason: /near-unit/,
    },
    { visible: [true, 1, true], reason: /booleans/ },
    { fps: 0, reason: /positive/ },
  ];
  for (const item of cases) {
    const candidate = { ...track(), ...item };
    const result = validateBakedTrack(candidate);
    assert.equal(result.ok, false);
    if (!result.ok) assert.match(result.reason, item.reason);
  }
});

test('rejects sample indices outside the active drift table', () => {
  const candidate = track();
  candidate.sampleIndex = [0, 1, 2];
  const result = validateBakedTrack(candidate, 2);
  assert.equal(result.ok, false);
  if (!result.ok) assert.match(result.reason, /camera drift/);
});

test('optional orientation and visibility may be omitted', () => {
  const { quaternionsXYZW: _quaternions, visible: _visible, ...candidate } = track();
  assert.equal(validateBakedTrack(candidate).ok, true);
});

test('permits packaged rounding but rejects scaled quaternion rotations', () => {
  const rounded = track();
  rounded.quaternionsXYZW[0] = [0, 0.70711, 0, 0.70711];
  assert.equal(validateBakedTrack(rounded).ok, true);
  rounded.quaternionsXYZW[0] = [0, 0, 0, 1.01];
  assert.equal(validateBakedTrack(rounded).ok, false);
});
