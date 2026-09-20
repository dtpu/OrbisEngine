import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';
import { applyPersonSizeAtAnchor, parsePersonSize } from '../src/person-size';

function near(actual: THREE.Vector3, expected: THREE.Vector3) {
  expect(actual.distanceTo(expected)).toBeLessThan(1e-12);
}

describe('visual person size', () => {
  test('defaults to unchanged size and rejects invalid factors', () => {
    for (const value of [null, '', ' ', 'no', 'NaN', 'Infinity', '0', '-0.9']) {
      expect(parsePersonSize(value)).toBe(1);
    }
    expect(parsePersonSize('0.9')).toBe(0.9);
    expect(parsePersonSize('1.2')).toBe(1.2);
    const person = new THREE.Group();
    person.position.set(2, 3, 4);
    person.scale.set(0.7, 0.8, 0.9);
    person.rotation.set(0.2, -0.3, 0.4);
    person.updateMatrix();
    const original = person.matrix.clone();
    applyPersonSizeAtAnchor(
      person,
      person.position.clone(),
      person.scale.clone(),
      new THREE.Vector3(12, -4, 7),
      1,
    );
    person.updateMatrix();
    expect(person.matrix.equals(original)).toBe(true);
  });

  test('holds the measured foot under rotated, scaled registration while body and mouth shrink', () => {
    const parent = new THREE.Group();
    parent.position.set(4, -2, 7);
    parent.rotation.set(-0.2, 0.7, 0.1);
    parent.scale.setScalar(1.8);
    const person = new THREE.Group();
    parent.add(person);
    const position = new THREE.Vector3(-3, 1.2, -5);
    const scale = new THREE.Vector3(0.4, 0.6, 0.5);
    const anchor = new THREE.Vector3(5, -0.9, -7);
    const mouth = anchor.clone().add(new THREE.Vector3(0.1, 1.55, 0.15));
    person.quaternion.setFromEuler(new THREE.Euler(0.31, -0.9, 0.21));
    person.position.copy(position);
    person.scale.copy(scale);
    parent.updateMatrixWorld(true);
    const oldFoot = person.localToWorld(anchor.clone());
    const oldMouth = person.localToWorld(mouth.clone());
    applyPersonSizeAtAnchor(person, position, scale, anchor, 0.9);
    parent.updateMatrixWorld(true);
    near(person.localToWorld(anchor.clone()), oldFoot);
    near(person.localToWorld(mouth.clone()), oldFoot.clone().lerp(oldMouth, 0.9));
    near(scale, new THREE.Vector3(0.4, 0.6, 0.5));
  });

  test('moving/interpolated anchors and repeated backwards seeks never accumulate scale or drift', () => {
    const person = new THREE.Group();
    person.rotation.set(0.2, 0.5, -0.1);
    const scale = new THREE.Vector3(0.75, 0.75, 0.75);
    const a = new THREE.Vector3(0.4, -1, -2);
    const b = new THREE.Vector3(1.2, -1.4, -4);
    const expected = new Map<number, THREE.Matrix4>();
    const sample = (t: number) => {
      const anchor = a.clone().lerp(b, t);
      const placement = new THREE.Vector3(0.1 * t, 0.4, -0.2 * t);
      const foot = anchor.clone().multiply(scale).applyQuaternion(person.quaternion).add(placement);
      applyPersonSizeAtAnchor(person, placement, scale, anchor, 0.9);
      person.updateMatrixWorld(true);
      near(person.localToWorld(anchor.clone()), foot);
      return person.matrix.clone();
    };
    for (const t of [0, 0.25, 0.5, 1]) expected.set(t, sample(t));
    for (const t of [1, 0.25, 0, 0.5, 0.25, 1, 1, 0]) {
      expect(sample(t).equals(expected.get(t)!)).toBe(true);
      near(person.scale, new THREE.Vector3(0.675, 0.675, 0.675));
    }
  });
});
