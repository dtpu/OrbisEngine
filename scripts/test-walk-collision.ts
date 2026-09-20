import assert from 'node:assert/strict';
import test from 'node:test';
import { countGridFootprintContacts, type WalkCollisionGrid } from '../src/walk-collision.ts';

const grid: WalkCollisionGrid = { cell: 0.1, width: 8, height: 8, ox: -4, oz: -4 };
const column = (i: number, j: number) => (index: number) => index === j * grid.width + i;

test('a column outside the circular corner is clear despite the old square footprint', () => {
  // Cell [0.2, 0.3] on both axes is sampled at (0.26, 0.26) by the old square.
  // Its closest point to the walker is about 0.269 away, beyond the 0.25 radius.
  assert.equal(countGridFootprintContacts(grid, 0.01, 0.01, 0.25, column(6, 6)), 0);
  assert.equal(countGridFootprintContacts(grid, 0.04, 0.04, 0.25, column(6, 6)), 1);
});

test('direct wall approaches and an isolated one-cell column retain collision', () => {
  const wall = (index: number) => index % grid.width === 6;
  assert.equal(countGridFootprintContacts(grid, -0.06, 0.05, 0.25, wall), 0);
  assert.ok(countGridFootprintContacts(grid, 0, 0.05, 0.25, wall) > 0);
  assert.equal(countGridFootprintContacts(grid, 0, 0.05, 0.25, column(6, 4)), 1);
  assert.equal(countGridFootprintContacts(grid, 0.25, 0.05, 0.25, column(6, 4)), 1);
});

test('cell-edge slivers collide even when the occupied cell centre is outside the circle', () => {
  // The cell starts 0.249 away, although its centre is 0.299 away.
  assert.equal(countGridFootprintContacts(grid, -0.049, 0.05, 0.25, column(6, 4)), 1);
  assert.equal(countGridFootprintContacts(grid, -0.051, 0.05, 0.25, column(6, 4)), 0);
});

test('exact tangent contact is clear while a small penetration blocks', () => {
  const unit: WalkCollisionGrid = { cell: 1, width: 1, height: 1, ox: 0, oz: 0 };
  assert.equal(
    countGridFootprintContacts(unit, -0.25, 0.5, 0.25, () => true),
    0,
  );
  assert.equal(
    countGridFootprintContacts(unit, -0.249, 0.5, 0.25, () => true),
    1,
  );
});

test('negative positions and negative cell origins preserve the same edge geometry', () => {
  assert.equal(countGridFootprintContacts(grid, -0.35, -0.35, 0.08, column(0, 0)), 1);
  assert.equal(countGridFootprintContacts(grid, -0.55, -0.35, 0.08, column(0, 0)), 0);
  assert.equal(countGridFootprintContacts(grid, -0.475, -0.35, 0.08, column(0, 0)), 1);
});

test('coarse cells larger than the body diameter and nonintegral radius ratios work', () => {
  const coarse: WalkCollisionGrid = { cell: 2, width: 2, height: 2, ox: -1, oz: -1 };
  const seen: number[] = [];
  assert.equal(
    countGridFootprintContacts(coarse, -1, -1, 0.1, (index) => {
      seen.push(index);
      return true;
    }),
    1,
  );
  assert.deepEqual(seen, [0]);
  assert.equal(
    countGridFootprintContacts(coarse, 0, 0, 0.13, () => true),
    4,
  );
  assert.equal(countGridFootprintContacts(grid, 0.05, 0.05, 0.13, column(6, 4)), 0);
  assert.equal(countGridFootprintContacts(grid, 0.05, 0.05, 0.17, column(6, 4)), 1);
});

test('grid clipping visits only valid indices once and supports partial edge overlap', () => {
  const small: WalkCollisionGrid = { cell: 1, width: 2, height: 2, ox: 0, oz: 0 };
  const seen: number[] = [];
  assert.equal(
    countGridFootprintContacts(small, 0.5, 0.5, 10, (index) => {
      seen.push(index);
      return index !== 2;
    }),
    3,
  );
  assert.deepEqual(seen, [0, 1, 2, 3]);
  assert.equal(
    countGridFootprintContacts(small, -0.1, 0.5, 0.2, () => true),
    1,
  );
  assert.equal(
    countGridFootprintContacts(small, 2.1, 0.5, 0.2, () => true),
    1,
  );
  assert.equal(
    countGridFootprintContacts(small, 0.5, -0.1, 0.2, () => true),
    1,
  );
  assert.equal(
    countGridFootprintContacts(small, 0.5, 2.1, 0.2, () => true),
    1,
  );
  assert.equal(
    countGridFootprintContacts(small, -10, -10, 0.2, () => {
      assert.fail('A footprint outside the grid must not query occupancy');
    }),
    0,
  );
});

test('translating the grid and walker together preserves all contacted indices', () => {
  const original: WalkCollisionGrid = { cell: 0.125, width: 9, height: 7, ox: -4, oz: -3 };
  const shifted = { ...original, ox: original.ox + 21, oz: original.oz - 17 };
  for (const radius of [0.03125, 0.1875, 0.3125]) {
    const contacts = (g: WalkCollisionGrid, x: number, z: number) => {
      const indices: number[] = [];
      countGridFootprintContacts(g, x, z, radius, (index) => {
        indices.push(index);
        return true;
      });
      return indices;
    };
    assert.deepEqual(
      contacts(original, -0.0625, 0.0625),
      contacts(shifted, -0.0625 + 21 * original.cell, 0.0625 - 17 * original.cell),
    );
  }
});
