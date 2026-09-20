import assert from 'node:assert/strict';
import test from 'node:test';
import { Vector3 } from 'three';
import {
  nearestHeldTime,
  parseHeadTrack,
  recordedState,
  sampleVector,
  segments,
  yawBetween,
  type ObjectMetadata,
} from '../src/interaction/recorded-object';

const metadata: ObjectMetadata = {
  pose: {
    segments: [
      { kind: 'attached', fromSourceFrame: 0, toSourceFrame: 9, parent: 'thrower' },
      { kind: 'free', fromSourceFrame: 10, toSourceFrame: 19, throwerTrack: 4, catcherTrack: 8 },
      { kind: 'attached', fromSourceFrame: 20, toSourceFrame: 29, parent: 'receiver' },
    ],
  },
};

test('recorded spans change ownership at held-to-flight and flight-to-held boundaries', () => {
  const people = [
    { id: 'thrower', track: 4 },
    { id: 'receiver', track: 8 },
  ];
  assert.deepEqual(recordedState(metadata, 10, 0.9, people), {
    free: false,
    owner: 'thrower',
    thrower: 'thrower',
    recipient: null,
  });
  assert.deepEqual(recordedState(metadata, 10, 1, people), {
    free: true,
    owner: null,
    thrower: 'thrower',
    recipient: 'receiver',
  });
  assert.deepEqual(recordedState(metadata, 10, 1.9, people), {
    free: true,
    owner: null,
    thrower: 'thrower',
    recipient: 'receiver',
  });
  assert.deepEqual(recordedState(metadata, 10, 2, people), {
    free: false,
    owner: 'receiver',
    thrower: 'receiver',
    recipient: null,
  });
  assert.deepEqual(recordedState(metadata, 10, 3.1, people), {
    free: false,
    owner: 'receiver',
    thrower: 'receiver',
    recipient: null,
  });
});

test('unmapped thrower and catcher tracks remain null rather than selecting a different person', () => {
  const state = recordedState(metadata, 10, 1.2, [{ id: 'different-track', track: 7 }]);
  assert.deepEqual(state, { free: true, owner: null, thrower: null, recipient: null });
});

test('nearest held time clamps to the nearest receiving span and rejects unknown people', () => {
  assert.equal(nearestHeldTime(metadata, 10, 0.1, 'receiver'), 2);
  assert.equal(nearestHeldTime(metadata, 10, 2.4, 'receiver'), 2.4);
  assert.equal(nearestHeldTime(metadata, 10, 9, 'receiver'), 2.9);
  assert.equal(nearestHeldTime(metadata, 10, 2.4, 'missing'), null);
});

test('head tracks validate their clock and vectors, and sample linearly with endpoint clamping', () => {
  const parsed = parseHeadTrack(
    {
      times: [1, 2, 4],
      positions: [
        [0, 1, 0],
        [2, 3, 4],
        [6, 7, 8],
      ],
      forward: [
        [0, 0, -1],
        [0, 0, -1],
        [0, 0, -1],
      ],
    },
    [0, 1, 2],
  );
  assert.ok(parsed);
  assert.deepEqual(sampleVector(parsed.positions, parsed.times, 1.5).toArray(), [1, 2, 2]);
  assert.deepEqual(sampleVector(parsed.positions, parsed.times, -3).toArray(), [0, 1, 0]);
  assert.deepEqual(sampleVector(parsed.positions, parsed.times, 9).toArray(), [6, 7, 8]);
  assert.deepEqual(parsed.forward, [
    [0, 0, -1],
    [0, 0, -1],
    [0, 0, -1],
  ]);

  for (const value of [
    null,
    {
      positions: [
        [0, 0, 0],
        [1, 1, 1],
      ],
    },
    {
      positions: [
        [0, 0, 0],
        [1, 1, 1],
      ],
      times: [0],
    },
    {
      positions: [
        [0, 0, 0],
        [1, 1, 1],
      ],
      times: [1, 1],
    },
    { positions: [[0, 0, Infinity]], times: [0] },
  ]) {
    assert.equal(parseHeadTrack(value, [0]), null);
  }
  const withoutForward = parseHeadTrack(
    { positions: [[0, 0, 0]], times: [0], forward: [[0, 0]] },
    [],
  );
  assert.ok(withoutForward);
  assert.equal(withoutForward.forward, undefined);
});

test('segment parsing ignores malformed individual entries and malformed JSON metadata safely', () => {
  const partial = {
    pose: {
      segments: [
        { kind: 'attached', fromSourceFrame: 2, toSourceFrame: 1, parent: 'bad-range' },
        { kind: 'unknown', fromSourceFrame: 0, toSourceFrame: 1 },
        { kind: 'free', fromSourceFrame: 3, toSourceFrame: 4 },
        null,
      ],
    },
  } as ObjectMetadata;
  assert.deepEqual(segments(partial), [{ kind: 'free', fromSourceFrame: 3, toSourceFrame: 4 }]);
  assert.deepEqual(
    segments({ pose: { segments: { not: 'an array' } } } as unknown as ObjectMetadata),
    [],
  );
  assert.deepEqual(
    segments({ pose: { segments: 'not an array' } } as unknown as ObjectMetadata),
    [],
  );
});

test('yaw maps a recorded horizontal forward vector onto a desired visitor direction', () => {
  const from = new Vector3(0, 0, 1);
  const right = new Vector3(1, 0, 0);
  const rotated = from.clone().applyQuaternion(yawBetween(from, right));
  assert.ok(rotated.distanceTo(right) < 1e-10);

  const unchanged = new Vector3(0, 0, -1).applyQuaternion(
    yawBetween(new Vector3(0, 0, -2), new Vector3(0, 0, -1)),
  );
  assert.ok(unchanged.distanceTo(new Vector3(0, 0, -1)) < 1e-10);
});
