import { expect, test } from 'bun:test';
import { Vector3 } from 'three';
import { raycastWalkFloor } from '../src/xr/teleport';

const origin = new Vector3(0, 2, 0);
const direction = new Vector3(0, -1, -1).normalize();

test('teleport ray lands on the first supported level, above or below the starting floor', () => {
  for (const height of [0, 1, -1]) {
    const target = new Vector3();
    expect(
      raycastWalkFloor(origin, direction, (_x, z) => (z < -0.5 ? height : 0), 0.025, 10, target),
    ).toBe(true);
    expect(target.y).toBe(height);
    expect(target.z).toBeCloseTo(height - 2, 4);
  }
});

test('a raised tread at a discontinuity is intersected instead of the flat plane behind it', () => {
  const target = new Vector3();
  expect(
    raycastWalkFloor(origin, direction, (_x, z) => (z <= -1.5 ? 1 : 0), 0.025, 10, target),
  ).toBe(true);
  expect(target.y).toBe(1);
  expect(target.z).toBeCloseTo(-1.5, 4);
});

test('unknown cells and overhead geometry cannot become teleport support', () => {
  const target = new Vector3(9, 9, 9);
  for (const height of [NaN, Infinity, 3]) {
    expect(raycastWalkFloor(origin, direction, () => height, 0.025, 10, target)).toBe(false);
    expect(target.toArray()).toEqual([9, 9, 9]);
  }
  expect(
    raycastWalkFloor(origin, direction, (_x, z) => (z < -2.5 ? 0 : NaN), 0.025, 10, target),
  ).toBe(true);
  expect(target.z).toBeCloseTo(-2.5, 4);
  expect(target.y).toBe(0);
});

test('upward rays, invalid samples and destinations beyond range have no target', () => {
  const target = new Vector3();
  expect(raycastWalkFloor(origin, new Vector3(0, 1, 0), () => 0, 0.025, 10, target)).toBe(false);
  expect(raycastWalkFloor(origin, direction, () => 0, 0.025, 1, target)).toBe(false);
  expect(raycastWalkFloor(origin, direction, () => 0, 0, 10, target)).toBe(false);
  expect(raycastWalkFloor(new Vector3(NaN, 2, 0), direction, () => 0, 0.025, 10, target)).toBe(
    false,
  );
});
