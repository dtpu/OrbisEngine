import assert from 'node:assert/strict';
import test from 'node:test';
import { Group, Quaternion, Vector3 } from 'three';
import { PhysicalWalk, parseWalkGain } from '../src/xr/physical-walk.ts';

function close(actual: Vector3, expected: Vector3) {
  assert.ok(actual.distanceTo(expected) < 1e-10, `${actual.toArray()} != ${expected.toArray()}`);
}

function walking(gain: number, upm = 1) {
  const walk = new PhysicalWalk(gain);
  const rig = new Group();
  rig.scale.setScalar(upm);
  const extra = new Vector3();
  return {
    walk,
    rig,
    sample(head: Vector3, active = true) {
      rig.position.add(walk.update(head, rig.quaternion, upm, extra, active));
      rig.updateMatrixWorld(true);
      return head.clone().applyMatrix4(rig.matrixWorld);
    },
  };
}

test('gain defaults to one, rejects nonfinite values and clamps finite values to 1..2', () => {
  for (const value of [null, '', ' ', 'NaN', 'Infinity', '-Infinity', 'abc', '-5', '0']) {
    assert.equal(parseWalkGain(value), 1);
  }
  assert.equal(parseWalkGain('1.3'), 1.3);
  assert.equal(parseWalkGain('2'), 2);
  assert.equal(parseWalkGain('100'), 2);
});

test('gain one leaves physical tracking and rig translation unchanged', () => {
  const { rig, sample } = walking(1, 2.5);
  rig.position.set(3, 4, 5);
  const from = sample(new Vector3(1, 1.7, -1));
  const to = sample(new Vector3(1.4, 1.4, -1.2));
  close(to.sub(from), new Vector3(1, -0.75, -0.5));
  close(rig.position, new Vector3(3, 4, 5));
});

test('gain 1.3 amplifies horizontal travel but preserves crouching, rotation and scale', () => {
  const { rig, sample } = walking(1.3, 2);
  const rotation = rig.quaternion.clone();
  const from = sample(new Vector3(0.6, 1.7, -0.4));
  close(rig.position, new Vector3()); // First pose is a baseline, not a step.
  const to = sample(new Vector3(1.6, 1.2, -0.9));
  close(to.sub(from), new Vector3(2.6, -1, -1.3));
  close(rig.scale, new Vector3(2, 2, 2));
  assert.deepEqual(rig.quaternion.toArray(), rotation.toArray());
});

test('stationary poses never accumulate drift, including after a teleport or snap turn', () => {
  const { rig, sample } = walking(1.3);
  const head = new Vector3(0.2, 1.7, -0.5);
  sample(head);
  sample(head.clone().add(new Vector3(1, 0, 0)));
  sample(head);
  rig.position.set(12, 3, 9); // Teleport changes the rig, not the tracked baseline.
  rig.quaternion.setFromAxisAngle(new Vector3(0, 1, 0), Math.PI / 2);
  const landed = sample(head);
  for (let i = 0; i < 100; i++) close(sample(head), landed);
  close(rig.position, new Vector3(12, 3, 9));
});

test('physical steps follow the current rig yaw after a turn', () => {
  const { rig, sample } = walking(1.3, 2);
  const head = new Vector3(0.2, 1.7, 0.5);
  sample(head);
  rig.quaternion.setFromAxisAngle(new Vector3(0, 1, 0), Math.PI / 2);
  const from = sample(head);
  const to = sample(head.clone().add(new Vector3(0, 0, -1)));
  close(to.sub(from), new Vector3(-2.6, 0, 0));
});

test('session re-entry and reference-space recenter establish fresh baselines', () => {
  const { walk, rig, sample } = walking(1.3);
  sample(new Vector3(1, 1.7, 2));
  sample(new Vector3(2, 1.7, 2));
  const previous = rig.position.clone();
  walk.reset(); // Reference-space reset: its coordinate jump must not receive extra gain.
  sample(new Vector3(-10, 1.6, 5));
  close(rig.position, previous);
  sample(new Vector3(-9, 1.6, 5));
  close(rig.position, previous.clone().add(new Vector3(0.3, 0, 0)));
  walk.reset(); // The next session also begins with an unrelated reference-space pose.
  rig.position.set(0, 0, 0);
  sample(new Vector3(20, 1.7, -20));
  close(rig.position, new Vector3());
});

test('possession suspends gain and rebases when free walking resumes', () => {
  const { rig, sample } = walking(1.3);
  sample(new Vector3(1, 1.7, 2));
  sample(new Vector3(2, 1.7, 2), false);
  sample(new Vector3(8, 1.7, 4), false);
  sample(new Vector3(9, 1.7, 5));
  close(rig.position, new Vector3());
  sample(new Vector3(10, 1.7, 5));
  close(rig.position, new Vector3(0.3, 0, 0));
});

test('invalid tracked positions rebase, while valid physical travel is never clamped', () => {
  const walk = new PhysicalWalk(1.3);
  const out = new Vector3();
  const rotation = new Quaternion();
  walk.update(new Vector3(0, 1.7, 0), rotation, 1, out);
  walk.update(new Vector3(NaN, 1.7, 0), rotation, 1, out);
  close(out, new Vector3());
  walk.update(new Vector3(5, 1.7, 0), rotation, 1, out);
  close(out, new Vector3());
  walk.update(new Vector3(105, 1.7, 0), rotation, 1, out);
  close(out, new Vector3(30, 0, 0));
});
