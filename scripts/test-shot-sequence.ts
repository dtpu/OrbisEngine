import { expect, test } from 'bun:test';
import {
  activeShotIndex,
  createShotLookup,
  parseShots,
  shotAssetUrl,
  shotGaps,
} from '../src/shot-sequence.ts';

// scripts/package_shot_sequence.py writes these: source-clock windows, one world per shot, and a
// gap wherever no candidate covers the clip.
const raw = [
  {
    index: 0,
    candidate: 'clip-shot01',
    world: '/marble-clip-shot01-clean.spz',
    sourceStart: 0,
    sourceEnd: 4.233,
    placement: 'shots/00/placement.json',
    cameras: 'shots/00/cameras.json',
    primary: 's00-person',
    people: ['s00-person', 's00-person_01'],
    sharedPlacement: { bodyHeightUnits: 0.85 },
    sharedScale: { mode: 'auto' },
    floorFit: null,
  },
  {
    index: 1,
    candidate: 'clip-shot02',
    world: '/marble-clip-shot02-clean.spz',
    sourceStart: 6.0,
    sourceEnd: 9.5,
    placement: null,
    cameras: 'shots/01/cameras.json',
    primary: 's01-person',
    people: ['s01-person'],
  },
];

test('parseShots reads the packager block and keeps every per-shot key', () => {
  const shots = parseShots(raw)!;
  expect(shots).toHaveLength(2);
  expect(shots[0].world).toBe('/marble-clip-shot01-clean.spz');
  expect(shots[0].cameras).toBe('shots/00/cameras.json');
  expect(shots[0].primary).toBe('s00-person');
  expect(shots[0].people).toEqual(['s00-person', 's00-person_01']);
  expect(shots[1].placement).toBeNull(); // a shot whose solve wrote none
  expect(shots[1].sharedPlacement).toBeNull();
});

test('parseShots leaves every scene that has no shots alone', () => {
  expect(parseShots(undefined)).toBeNull();
  expect(parseShots(null)).toBeNull();
});

test('parseShots refuses a package it cannot play straight through', () => {
  expect(() => parseShots([])).toThrow(/empty/);
  expect(() => parseShots({} as unknown)).toThrow(/must be a list/);
  expect(() => parseShots([{ ...raw[0], sourceEnd: 0 }])).toThrow(/forward window/);
  expect(() => parseShots([{ ...raw[0], world: '' }])).toThrow(/no world/);
  // overlapping windows would make two worlds active at once
  expect(() => parseShots([raw[0], { ...raw[1], sourceStart: 4.0 }])).toThrow(/must not overlap/);
  // and so would running backwards
  expect(() => parseShots([raw[1], raw[0]])).toThrow(/must not overlap/);
});

test('activeShotIndex is closed-open at every boundary', () => {
  const shots = parseShots(raw)!;
  expect(activeShotIndex(shots, 0)).toBe(0); // the first frame belongs to shot 0
  expect(activeShotIndex(shots, 4.2329)).toBe(0);
  expect(activeShotIndex(shots, 4.233)).toBe(-1); // the cut belongs to the NEXT window, not this one
  expect(activeShotIndex(shots, 6.0)).toBe(1);
  expect(activeShotIndex(shots, 9.4999)).toBe(1);
  expect(activeShotIndex(shots, 9.5)).toBe(-1);
});

test('a time in a gap, before the first shot or past the last is no shot at all', () => {
  const shots = parseShots(raw)!;
  expect(activeShotIndex(shots, 5)).toBe(-1);
  expect(activeShotIndex(shots, -1)).toBe(-1);
  expect(activeShotIndex(shots, 99)).toBe(-1);
  expect(activeShotIndex(shots, NaN)).toBe(-1);
  expect(activeShotIndex([], 0)).toBe(-1);
  expect(activeShotIndex(null, 0)).toBe(-1);
});

test('back-to-back shots leave no gap and no ambiguity at the cut', () => {
  const shots = parseShots([
    { ...raw[0], sourceEnd: 4 },
    { ...raw[1], sourceStart: 4, sourceEnd: 8 },
  ])!;
  expect(activeShotIndex(shots, 3.999)).toBe(0);
  expect(activeShotIndex(shots, 4)).toBe(1);
  expect(shotGaps(shots, 8)).toEqual([]);
});

test('a single shot that does not cover the clip still answers everywhere', () => {
  const shots = parseShots([{ ...raw[0], sourceStart: 1, sourceEnd: 3 }])!;
  expect(activeShotIndex(shots, 0.99)).toBe(-1);
  expect(activeShotIndex(shots, 1)).toBe(0);
  expect(activeShotIndex(shots, 3)).toBe(-1);
  expect(shotGaps(shots, 5)).toEqual([
    { start: 0, end: 1 },
    { start: 3, end: 5 },
  ]);
});

test('the lookup answers the same as the search, however the viewer seeks', () => {
  const shots = parseShots(raw)!;
  const lookup = createShotLookup(shots);
  // play forwards, then seek backwards over a cut, then jump into a gap and out again
  const times = [0, 1, 2, 4.2, 4.233, 5, 6, 7, 9.49, 9.5, 2, 0, 6.5, 4.9, 0.1];
  for (const t of times) expect(lookup(t)).toBe(activeShotIndex(shots, t));
  const empty = createShotLookup([]);
  expect(empty(0)).toBe(-1);
});

test('shotGaps mirrors the packager`s own gap list', () => {
  const shots = parseShots(raw)!;
  expect(shotGaps(shots, 12)).toEqual([
    { start: 4.233, end: 6.0 },
    { start: 9.5, end: 12 },
  ]);
  expect(shotGaps(shots, 9.5)).toEqual([{ start: 4.233, end: 6.0 }]);
  expect(shotGaps(null, 10)).toEqual([]);
  expect(shotGaps(shots, 0)).toEqual([]);
});

test('shotAssetUrl resolves package-relative paths beside people.json', () => {
  const base = '/worlds/clip-sequence-4d/';
  expect(shotAssetUrl(base, 'shots/00/cameras.json')).toBe(
    '/worlds/clip-sequence-4d/shots/00/cameras.json',
  );
  expect(shotAssetUrl(base, '/absolute/cameras.json')).toBe('/absolute/cameras.json');
  expect(shotAssetUrl(base, null)).toBeNull();
});
