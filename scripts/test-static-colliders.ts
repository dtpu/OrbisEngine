import assert from 'node:assert/strict';
import test from 'node:test';
import { Quaternion, Vector3 } from 'three';
import {
  parseStaticColliders,
  supportHeight,
  supportPlane,
  highestSupport,
  capsuleIntersects,
  capsuleSweepBlocked,
} from '../src/static-colliders.ts';

function manifest(overrides = {}) {
  return {
    schema: 'wander.colliders/1',
    coordinates: 'viewer-world',
    world: 'room.spz',
    bodies: [
      {
        id: 'seat',
        role: 'seat',
        shape: 'box',
        center: [0, 0.45, 0],
        halfExtents: [0.4, 0.05, 0.3],
        quaternionXYZW: [0, 0, 0, 1],
        walkable: true,
        provenance: 'Synthetic test geometry',
        ...overrides,
      },
    ],
  };
}
const boxes = (overrides = {}) => parseStaticColliders(manifest(overrides), '/room.spz');

test('explicit geometry validates world, coordinates, extents, rotation and identity', () => {
  for (const value of [
    null,
    {},
    { ...manifest(), world: 'other.spz' },
    { ...manifest(), coordinates: 'source-world' },
    { ...manifest(), bodies: [] },
    manifest({ halfExtents: [1, 0, 1] }),
    manifest({ center: [0, NaN, 0] }),
    manifest({ quaternionXYZW: [0, 0, 0, 0] }),
    manifest({ provenance: '' }),
    { ...manifest(), bodies: [...manifest().bodies, ...manifest().bodies] },
  ])
    assert.throws(() => parseStaticColliders(value, '/room.spz'));
  assert.equal(
    parseStaticColliders(manifest(), 'http://localhost/@fs/public/room.spz')[0].id,
    'seat',
  );
});

test('finite support stays on the top face and cannot replace unrelated floor', () => {
  const b = boxes()[0];
  assert.equal(supportHeight(b, 0, 0), 0.5);
  assert.equal(supportHeight(b, 0.5, 0), undefined);
  assert.equal(highestSupport([b], 0, 0, 0.2), undefined);
  assert.equal(highestSupport([b], 0, 0, 0.6), 0.5);
  assert.equal(highestSupport(boxes({ walkable: false }), 0, 0), undefined);
  assert.equal(highestSupport([], 0, 0), undefined);
});

test('inclined top is a finite rotated plane, with capsule contact height', () => {
  const q = new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), Math.PI / 6);
  const b = boxes({ center: [0, 0, 0], quaternionXYZW: q.toArray() })[0];
  const a = supportHeight(b, 0, 0)!;
  const c = supportHeight(b, 0.1, 0)!;
  assert.ok(Math.abs(c - a - 0.1 * Math.tan(Math.PI / 6)) < 1e-10);
  const p = supportPlane(b);
  assert.ok(Math.abs(p.normal.dot(p.point) - p.d) < 1e-10);
  const h = highestSupport([b], 0, 0, Infinity, 0.08)!;
  assert.equal(capsuleIntersects(b, new Vector3(0, h, 0), 1.7, 0.08), false);
});

test('solid seat blocks standing below it but supports standing on its top', () => {
  const b = boxes()[0];
  assert.equal(capsuleIntersects(b, new Vector3(0, 0, 0), 1.7, 0.1), true);
  assert.equal(capsuleIntersects(b, new Vector3(0, 0.5, 0), 1.7, 0.1), false);
  assert.equal(capsuleIntersects(b, new Vector3(0, -2, 0), 1.7, 0.1), false);
  assert.equal(capsuleIntersects(b, new Vector3(0.7, 0, 0), 1.7, 0.1), false);
});

test('a continuous sweep cannot tunnel through a thin or rotated obstacle', () => {
  const b = boxes({ halfExtents: [0.002, 0.4, 0.3], walkable: false });
  const from = new Vector3(-2, 0, 0),
    to = new Vector3(2, 0, 0);
  assert.equal(capsuleIntersects(b[0], from, 1.7, 0.1), false);
  assert.equal(capsuleIntersects(b[0], to, 1.7, 0.1), false);
  assert.equal(capsuleSweepBlocked(b, from, to, 1.7, 0.1), true);
  const q = new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), Math.PI / 4);
  assert.equal(
    capsuleSweepBlocked(boxes({ quaternionXYZW: q.toArray() }), from, to, 1.7, 0.1),
    true,
  );
  assert.equal(
    capsuleSweepBlocked(b, new Vector3(-2, 1, 0), new Vector3(2, 1, 0), 1.7, 0.1),
    false,
  );
  assert.equal(capsuleSweepBlocked([], from, to, 1.7, 0.1), false);
});

test('movement along a supported horizontal top does not collide with itself', () => {
  assert.equal(
    capsuleSweepBlocked(boxes(), new Vector3(-0.2, 0.5, 0), new Vector3(0.2, 0.5, 0), 1.7, 0.1),
    false,
  );
  assert.throws(() =>
    capsuleSweepBlocked(boxes(), new Vector3(NaN, 0, 0), new Vector3(), 1.7, 0.1),
  );
  assert.throws(() => capsuleIntersects(boxes()[0], new Vector3(), 0, 0.1));
});

test('a valid capsule near a box corner can move away or pass outside the rounded contact', () => {
  const b = boxes({ center: [0, 0, 0], halfExtents: [0.3, 0.3, 0.3] });
  const from = new Vector3(0.45, -0.2, 0.45);
  const away = new Vector3(0.8, -0.2, 0.8);
  assert.equal(capsuleIntersects(b[0], from, 1.7, 0.2), false);
  assert.equal(capsuleIntersects(b[0], away, 1.7, 0.2), false);
  assert.equal(capsuleSweepBlocked(b, from, away, 1.7, 0.2), false);
  assert.equal(capsuleSweepBlocked(b, away, from, 1.7, 0.2), false);
  assert.equal(
    capsuleSweepBlocked(b, new Vector3(0.6, -0.2, 0.4), new Vector3(0.4, -0.2, 0.6), 1.7, 0.2),
    false,
  );
  // A closer diagonal still passes through the actual rounded contact region.
  assert.equal(
    capsuleSweepBlocked(b, new Vector3(0.6, -0.2, 0.2), new Vector3(0.2, -0.2, 0.6), 1.7, 0.2),
    true,
  );
});

test('a capsule can follow an inclined top in either direction at its support height', () => {
  const q = new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), Math.PI / 6);
  const b = boxes({ center: [0, 0, 0], quaternionXYZW: q.toArray() });
  const low = new Vector3(0, highestSupport(b, 0, 0, Infinity, 0.08), 0);
  const high = new Vector3(0.1, highestSupport(b, 0.1, 0, Infinity, 0.08), 0);
  assert.equal(capsuleIntersects(b[0], low, 1.7, 0.08), false);
  assert.equal(capsuleIntersects(b[0], high, 1.7, 0.08), false);
  assert.equal(capsuleSweepBlocked(b, low, high, 1.7, 0.08), false);
  assert.equal(capsuleSweepBlocked(b, high, low, 1.7, 0.08), false);
  assert.equal(capsuleSweepBlocked(b, low, new Vector3(high.x, low.y, high.z), 1.7, 0.08), true);
});

test('inclined support requires the capsule tangent to lie on the finite top face', () => {
  const q = new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), Math.PI / 6);
  const b = boxes({ center: [0, 0, 0], quaternionXYZW: q.toArray() });
  assert.notEqual(supportHeight(b[0], 0.32, 0), undefined);
  assert.equal(highestSupport(b, 0.32, 0, Infinity, 0.08), undefined);
  assert.notEqual(highestSupport(b, 0, 0, Infinity, 0.08), undefined);
});
