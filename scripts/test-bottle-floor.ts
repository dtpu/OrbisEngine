import { expect, test } from 'bun:test';
import * as THREE from 'three';
import { interpolateBottleFloor } from '../src/interaction/bottle-floor';
import { BottlePhysics } from '../src/interaction/bottle-physics';
import { BottleVisual } from '../src/interaction/bottle-visual';

const unknownOneCell = { floor: NaN, inside: true, distance: 1 };

test('one-cell omissions recover only agreeing measured neighbouring supports', () => {
  expect(interpolateBottleFloor({ floor: -0.63, inside: true, distance: 0 }, [], 0.05)).toBe(-0.63);
  expect(interpolateBottleFloor(unknownOneCell, [-0.6511, -0.637], 0.058)).toBe(-0.637);
  expect(interpolateBottleFloor(unknownOneCell, [-0.7046, -0.6951, -0.6887], 0.058)).toBe(-0.6887);
  expect(
    interpolateBottleFloor({ floor: NaN, inside: true, distance: 3 }, [-0.65, -0.64], 0.058),
  ).toBeNull();
  expect(interpolateBottleFloor(unknownOneCell, [-0.65], 0.058)).toBeNull();
  expect(interpolateBottleFloor(unknownOneCell, [-0.8, -0.6], 0.058)).toBeNull();
  expect(
    interpolateBottleFloor({ floor: NaN, inside: false, distance: 1 }, [-0.65, -0.64], 0.058),
  ).toBeNull();
});

test('the reported elevator throw settles on its bounded measured floor holes', () => {
  // Actual cached-elevator collision samples at the Quest release and later lost coordinates.
  const stature = 0.91527837485075;
  const visual = new BottleVisual(
    stature,
    new THREE.Vector3(
      0.032560366967145135,
      0.06512073393429027,
      0.032560366967145135,
    ).multiplyScalar(1.08),
  );
  const floorRadius = new THREE.Box3()
    .setFromObject(visual.proxy)
    .getBoundingSphere(new THREE.Sphere()).radius;
  const supports = [
    { x: 0.7, z: -0.8, floors: [-0.6511, -0.637] },
    { x: -0.1308783, z: -4.1531556, floors: [-0.7046, -0.6951, -0.6887] },
  ];
  const floorAt = (x: number, z: number) => {
    const support = supports.find(
      (candidate) => Math.hypot(x - candidate.x, z - candidate.z) < 0.01,
    );
    return support ? interpolateBottleFloor(unknownOneCell, support.floors, 0.058528) : null;
  };
  const drop = (x: number, z: number, floor: number) => {
    const physics = new BottlePhysics({
      radius: 0.03516519632451675,
      floorRadius,
      gravity: 5.77 * stature,
      maxSpeed: 30,
      floorAt,
      blockedAt: () => false,
      bounceSpeed: 100,
      settleSpeed: 1,
    });
    expect(physics.startFlight([x, 1, z], [0, -30, 0])).toBe(true);
    expect(physics.step(0.1)).toEqual([{ type: 'dropped', position: [x, floor + floorRadius, z] }]);
  };
  drop(0.7, -0.8, -0.637);
  drop(-0.1308783, -4.1531556, -0.6887);
  visual.dispose();
});

test('unknown floor callbacks remain unsupported', () => {
  const physics = new BottlePhysics({
    radius: 0.035,
    gravity: 5,
    maxSpeed: 30,
    floorAt: () => null,
    blockedAt: () => false,
  });
  expect(physics.startFlight([0, 1, 0], [0, -30, 0])).toBe(true);
  physics.step(0.1);
  expect(physics.snapshot().position[1]).toBeLessThan(0);
  expect(physics.snapshot().mode).toBe('free');
});
