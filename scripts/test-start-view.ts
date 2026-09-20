import { expect, test } from 'bun:test';
import {
  cameraWorldPose,
  chooseStartPose,
  firstSourceCamera,
  pointsExtent,
  robustExtent,
} from '../src/start-view.ts';

// a 2 x 2 x 2 room of points, plus the handful of enormous sky splats a Marble world always carries
function room(): number[][] {
  const points: number[][] = [];
  for (let i = 0; i <= 20; i++)
    for (let j = 0; j <= 20; j++) {
      points.push([-1 + i * 0.1, -1 + j * 0.1, -1]);
      points.push([-1 + i * 0.1, -1 + j * 0.1, 1]);
    }
  return points;
}

test('robustExtent ignores outliers the raw bounding box is made of', () => {
  const points = room();
  const sky = [
    [-400, 300, -500],
    [420, 260, 480],
    [0, 900, 0],
  ];
  const extent = robustExtent([...points, ...sky])!;
  expect(extent).not.toBeNull();
  for (let axis = 0; axis < 3; axis++) {
    expect(extent.min[axis]).toBeGreaterThanOrEqual(-1.001);
    expect(extent.max[axis]).toBeLessThanOrEqual(1.001);
  }
  expect(extent.diagonal).toBeLessThan(4);
  // the rule this replaces would have opened the scene hundreds of units away
  const raw = pointsExtent([...points, ...sky])!;
  expect(raw.diagonal).toBeGreaterThan(1000);
});

test('robustExtent percentiles are per axis and symmetric', () => {
  const points: number[][] = [];
  for (let i = 0; i < 100; i++) points.push([i, 10 * i, -i]);
  const extent = robustExtent(points, { low: 0.1, high: 0.9 })!;
  expect(extent.min[0]).toBeCloseTo(9.9, 0);
  expect(extent.max[0]).toBeCloseTo(89.1, 0);
  expect(extent.min[2]).toBeCloseTo(-89.1, 0);
  expect(extent.center[1]).toBeCloseTo(495, 0);
});

test('robustExtent refuses a cloud too small to have percentiles, and bad fractions', () => {
  expect(robustExtent([[0, 0, 0]])).toBeNull();
  expect(robustExtent([], { minimumPoints: 0 })).toBeNull();
  expect(() => robustExtent(room(), { low: 0.9, high: 0.1 })).toThrow();
});

test('robustExtent skips non-finite points instead of poisoning the box', () => {
  const extent = robustExtent([...room(), [NaN, 0, 0], [0, Infinity, 0], [0, 0]] as number[][])!;
  expect(Number.isFinite(extent.diagonal)).toBe(true);
  expect(extent.max[1]).toBeLessThanOrEqual(1.001);
});

// cameras.json as scripts/package_person_sequence.py writes it: rows of a camera_to_world, camera 0
// at the origin of the SfM frame looking -z
const identityCamera = {
  sourceIndex: 0,
  time: 0,
  camera_to_world: [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ],
};
const laterCamera = {
  sourceIndex: 12,
  time: 1,
  camera_to_world: [
    [1, 0, 0, 5],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ],
};

test('firstSourceCamera takes the earliest solved camera, whatever the file order', () => {
  expect(firstSourceCamera({ cameras: [laterCamera, identityCamera] })).toBe(identityCamera);
  expect(firstSourceCamera([laterCamera, identityCamera])).toBe(identityCamera);
  expect(firstSourceCamera({ cameras: [] })).toBeNull();
  expect(firstSourceCamera(null)).toBeNull();
  expect(firstSourceCamera({ cameras: [{ camera_to_world: [[1, 0, 0, 0]] }] })).toBeNull();
});

test('cameraWorldPose carries the SfM camera through the cast placement', () => {
  const pose = cameraWorldPose(identityCamera, { scale: 2, offset: [0, -0.5, 0] })!;
  expect(pose.position).toEqual([0, -0.5, 0]);
  expect(pose.forward[2]).toBeCloseTo(-1, 9);
  expect(pose.back[2]).toBeCloseTo(1, 9);
  const moved = cameraWorldPose(laterCamera, { scale: 2, offset: [1, -0.5, 3] })!;
  expect(moved.position).toEqual([11, -0.5, 3]);
});

test('cameraWorldPose applies the ?rotfix= rotation to position and axes', () => {
  // 90 deg about +y: native -z (the look direction) becomes world -x
  const half = Math.SQRT1_2;
  const pose = cameraWorldPose(laterCamera, {
    scale: 1,
    offset: [0, 0, 0],
    rotationXYZW: [0, half, 0, half],
  })!;
  expect(pose.position[0]).toBeCloseTo(0, 9);
  expect(pose.position[2]).toBeCloseTo(-5, 9);
  expect(pose.forward[0]).toBeCloseTo(-1, 9);
  expect(pose.forward[2]).toBeCloseTo(0, 9);
});

test('cameraWorldPose reads a flat 16-number row-major matrix too', () => {
  const pose = cameraWorldPose(
    { camera_to_world: [1, 0, 0, 2, 0, 1, 0, 3, 0, 0, 1, 4, 0, 0, 0, 1] },
    { scale: 1, offset: [0, 0, 0] },
  )!;
  expect(pose.position).toEqual([2, 3, 4]);
});

test('the source camera route stands behind the phone and looks at the person', () => {
  const pose = chooseStartPose({
    cameras: { cameras: [identityCamera, laterCamera] },
    placement: { scale: 0.75, offset: [0, -0.001, 0] },
    subject: [0.2, 0.3, -2],
    stepBack: 0.25,
  })!;
  expect(pose.source).toBe('source-camera');
  // camera 0 is the origin of the frame, so the start pose is that point stepped back along +z
  expect(pose.position[2]).toBeCloseTo(0.25, 9);
  expect(pose.position[0]).toBeCloseTo(0, 9);
  expect(pose.target).toEqual([0.2, 0.3, -2]);
});

test('without a subject the source camera route looks the way the clip was shot', () => {
  const pose = chooseStartPose({
    cameras: [identityCamera],
    placement: { scale: 1, offset: [0, 0, 0] },
    stepBack: 0.5,
    minimumDistance: 2,
  })!;
  expect(pose.position[2]).toBeCloseTo(0.5, 9);
  expect(pose.target[2]).toBeLessThan(pose.position[2]);
});

test('a scene with no cameras falls back to the people, then the splats, then the box', () => {
  const peopleExtent = pointsExtent(
    [
      [0, 0, -2],
      [0.5, 0.9, -2.4],
    ],
    0.2,
  )!;
  const splatExtent = robustExtent(room())!;
  const boundingBox = { center: [0, 0, 0] as const, size: [900, 900, 900] as const };

  const people = chooseStartPose({ peopleExtent, splatExtent, boundingBox, minimumDistance: 1.5 })!;
  expect(people.source).toBe('people');
  expect(people.position[2] - people.target[2]).toBeCloseTo(1.5, 9); // the floor, not the tiny diagonal

  const splats = chooseStartPose({ splatExtent, boundingBox })!;
  expect(splats.source).toBe('splat-extent');
  expect(splats.position[2]).toBeLessThan(2); // not 0.55 x 1558 units away

  const box = chooseStartPose({ boundingBox })!;
  expect(box.source).toBe('bounding-box');
  expect(box.position[2]).toBeCloseTo(Math.hypot(900, 900, 900) * 0.55, 6);

  expect(chooseStartPose({})).toBeNull();
});

test('a malformed cameras.json falls through rather than throwing', () => {
  const splatExtent = robustExtent(room())!;
  const pose = chooseStartPose({
    cameras: { cameras: [{ camera_to_world: 'nonsense' }] },
    placement: { scale: 1, offset: [0, 0, 0] },
    splatExtent,
  })!;
  expect(pose.source).toBe('splat-extent');
  const broken = chooseStartPose({
    cameras: { cameras: [identityCamera] },
    placement: { scale: NaN, offset: [0, 0, 0] },
    splatExtent,
  })!;
  expect(broken.source).toBe('splat-extent');
});

test('the framing distance is clamped between its floor and its ceiling', () => {
  const splatExtent = robustExtent(room())!;
  const near = chooseStartPose({ splatExtent, minimumDistance: 10 })!;
  expect(near.position[2] - near.target[2]).toBeCloseTo(10, 9);
  const far = chooseStartPose({ splatExtent, maximumDistance: 0.4 })!;
  expect(far.position[2] - far.target[2]).toBeCloseTo(0.4, 9);
});
