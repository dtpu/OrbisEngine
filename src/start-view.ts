// Where a scene opens when the URL carries no ?cam=.
//
// The viewer's original rule was "world bounding-box centre, stepped back by 0.55 x the box
// diagonal". A Marble world holds a handful of enormous sky/background splats, so that box can be
// tens of units across even when the reconstructed room is two units wide, and a fresh scene opens
// far away, staring at a small bright blob. Nothing here changes a URL that already says where the
// camera goes: every preset passes ?cam=, so this module only ever decides the FIRST frame of a
// scene that never had a fitted start pose.
//
// Three routes, best first:
//   1. the recorded source camera. people.json names a cameras.json solved in the SAME frame as the
//      people, so the first solved camera - the place the phone actually stood when the clip began -
//      maps into the world with the very similarity the cast is placed with (scale, offset and the
//      optional ?rotfix= rotation). Stepping back along that camera's own backward axis and looking
//      at the primary person at t=0 opens the scene on the recorded subject.
//   2. the people's own region, when there is a cast but no solved camera.
//   3. a ROBUST extent of the world's splat centres (a low/high percentile per axis), which ignores
//      the sky splats the raw bounding box is made of.
// The raw bounding box remains the last resort, so a world with no usable geometry behaves as before.
//
// Pure arithmetic on plain arrays: no three.js, no DOM, so scripts/test-start-view.ts can check it.

export type Vec3 = readonly [number, number, number];
export type Quat = readonly [number, number, number, number];

export interface Extent {
  min: [number, number, number];
  max: [number, number, number];
  center: [number, number, number];
  size: [number, number, number];
  /** Length of the extent's diagonal, the distance the framing rules are scaled by. */
  diagonal: number;
}

/** How a start pose was arrived at, so the HUD and the tests can tell the routes apart. */
export type StartPoseSource = 'source-camera' | 'people' | 'splat-extent' | 'bounding-box';

export interface StartPose {
  position: [number, number, number];
  target: [number, number, number];
  source: StartPoseSource;
}

/** The similarity that carries SfM-frame (camera/person) coordinates into this world. */
export interface WorldPlacement {
  /** The cast's shared scale (`scale0` in fourd.html). */
  scale: number;
  /** The cast's shared base position (`pos0` in fourd.html). */
  offset: Vec3;
  /** ?rotfix=1's SfM-frame -> world-frame rotation, when one is applied. */
  rotationXYZW?: Quat | null;
}

/** One entry of a `wander` cameras.json: a row-major 4x4 camera_to_world, nested rows or flat. */
export interface SourceCameraEntry {
  sourceIndex?: number;
  time?: number;
  camera_to_world: readonly (readonly number[])[] | readonly number[];
}

const FINITE = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

function percentile(sorted: Float64Array, fraction: number): number {
  const i = Math.round(fraction * (sorted.length - 1));
  return sorted[Math.min(sorted.length - 1, Math.max(0, i))];
}

function extentOf(min: [number, number, number], max: [number, number, number]): Extent {
  const size: [number, number, number] = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  return {
    min,
    max,
    center: [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2],
    size,
    diagonal: Math.hypot(size[0], size[1], size[2]),
  };
}

/**
 * Per-axis percentile box of a point cloud. `low`/`high` are fractions (0.05/0.95 by default), so a
 * handful of sky splats a thousand units away cannot set the framing distance. Non-finite points are
 * skipped; fewer than `minimumPoints` usable points returns null so the caller keeps its old rule.
 */
export function robustExtent(
  points: Iterable<readonly number[]>,
  options: { low?: number; high?: number; minimumPoints?: number } = {},
): Extent | null {
  const low = options.low ?? 0.05;
  const high = options.high ?? 0.95;
  const minimumPoints = Math.max(1, options.minimumPoints ?? 8);
  if (!(low >= 0) || !(high <= 1) || !(low < high)) {
    throw new Error('robustExtent needs 0 <= low < high <= 1');
  }
  const xs: number[] = [],
    ys: number[] = [],
    zs: number[] = [];
  for (const p of points) {
    if (!p || p.length < 3) continue;
    if (!FINITE(p[0]) || !FINITE(p[1]) || !FINITE(p[2])) continue;
    xs.push(p[0]);
    ys.push(p[1]);
    zs.push(p[2]);
  }
  if (xs.length < minimumPoints) return null;
  const axes = [xs, ys, zs].map((values) => Float64Array.from(values).sort());
  const min = axes.map((a) => percentile(a, low)) as [number, number, number];
  const max = axes.map((a) => percentile(a, high)) as [number, number, number];
  return extentOf(min, max);
}

/** The smallest box holding every point, grown by `pad` on each side. Used for the cast's region. */
export function pointsExtent(points: Iterable<readonly number[]>, pad = 0): Extent | null {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  let n = 0;
  for (const p of points) {
    if (!p || p.length < 3) continue;
    if (!FINITE(p[0]) || !FINITE(p[1]) || !FINITE(p[2])) continue;
    n++;
    for (let i = 0; i < 3; i++) {
      min[i] = Math.min(min[i], p[i]);
      max[i] = Math.max(max[i], p[i]);
    }
  }
  if (!n) return null;
  for (let i = 0; i < 3; i++) {
    min[i] -= pad;
    max[i] += pad;
  }
  return extentOf(min, max);
}

function rotate(q: Quat | null | undefined, v: Vec3): [number, number, number] {
  if (!q) return [v[0], v[1], v[2]];
  const [x, y, z, w] = q;
  // t = 2 * (q_vec x v); v' = v + w * t + q_vec x t
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [
    v[0] + w * tx + (y * tz - z * ty),
    v[1] + w * ty + (z * tx - x * tz),
    v[2] + w * tz + (x * ty - y * tx),
  ];
}

function normalize(v: Vec3): [number, number, number] | null {
  const n = Math.hypot(v[0], v[1], v[2]);
  if (!(n > 1e-12)) return null;
  return [v[0] / n, v[1] / n, v[2] / n];
}

function rows(matrix: SourceCameraEntry['camera_to_world']): number[][] | null {
  if (!Array.isArray(matrix)) return null;
  if (matrix.length === 16 && matrix.every((v) => FINITE(v))) {
    const flat = matrix as readonly number[];
    return [0, 1, 2, 3].map((r) => [
      flat[4 * r],
      flat[4 * r + 1],
      flat[4 * r + 2],
      flat[4 * r + 3],
    ]);
  }
  if (matrix.length !== 4) return null;
  const out: number[][] = [];
  for (const row of matrix as readonly (readonly number[])[]) {
    if (!Array.isArray(row) || row.length !== 4 || !row.every((v) => FINITE(v))) return null;
    out.push([row[0], row[1], row[2], row[3]]);
  }
  return out;
}

export interface WorldCameraPose {
  position: [number, number, number];
  /** Unit vector the camera looks along (OpenGL: -Z of its own basis). */
  forward: [number, number, number];
  /** Unit vector pointing behind the camera, which a start pose steps back along. */
  back: [number, number, number];
  up: [number, number, number];
}

/**
 * A row-major camera_to_world from cameras.json, carried into world space by the same similarity
 * the cast is placed with: world = offset + scale * (rotation * native). Directions take the
 * rotation only. Returns null for a matrix this viewer cannot read.
 */
export function cameraWorldPose(
  entry: SourceCameraEntry | null | undefined,
  placement: WorldPlacement,
): WorldCameraPose | null {
  const m = entry ? rows(entry.camera_to_world) : null;
  if (!m) return null;
  if (!FINITE(placement.scale) || placement.scale === 0) return null;
  const offset = placement.offset;
  if (!FINITE(offset[0]) || !FINITE(offset[1]) || !FINITE(offset[2])) return null;
  const q = placement.rotationXYZW ?? null;
  const native: Vec3 = [m[0][3], m[1][3], m[2][3]];
  const rotated = rotate(q, native);
  const position: [number, number, number] = [
    offset[0] + placement.scale * rotated[0],
    offset[1] + placement.scale * rotated[1],
    offset[2] + placement.scale * rotated[2],
  ];
  // columns of the rotation part: x right, y up, z backward (the camera looks down -z)
  const back = normalize(rotate(q, [m[0][2], m[1][2], m[2][2]]));
  const up = normalize(rotate(q, [m[0][1], m[1][1], m[2][1]]));
  if (!back || !up) return null;
  if (!position.every((v) => Number.isFinite(v))) return null;
  return { position, forward: [-back[0], -back[1], -back[2]], back, up };
}

/** The first solved camera of a cameras.json document (bare array or `{ cameras: [...] }`). */
export function firstSourceCamera(document: unknown): SourceCameraEntry | null {
  const list = Array.isArray(document)
    ? document
    : document && typeof document === 'object' && Array.isArray((document as any).cameras)
      ? ((document as any).cameras as unknown[])
      : null;
  if (!list?.length) return null;
  let best: SourceCameraEntry | null = null;
  let bestKey = Infinity;
  for (const raw of list) {
    if (!raw || typeof raw !== 'object') continue;
    const entry = raw as SourceCameraEntry;
    if (!rows(entry.camera_to_world)) continue;
    const key = FINITE(entry.time)
      ? entry.time
      : FINITE(entry.sourceIndex)
        ? entry.sourceIndex
        : Infinity;
    if (best === null || key < bestKey) {
      best = entry;
      bestKey = key;
    }
  }
  return best;
}

export interface StartPoseInput {
  /** cameras.json as fetched, or its `cameras` array, or null when the scene has none. */
  cameras?: unknown;
  /** The similarity the cast is placed with. Required for the source-camera route. */
  placement?: WorldPlacement | null;
  /** Where the primary person is at t = 0, in world units. */
  subject?: Vec3 | null;
  /** The cast's world-space region, when one can be measured. */
  peopleExtent?: Extent | null;
  /** A robust extent of the world's splat centres. */
  splatExtent?: Extent | null;
  /** The world's raw bounding box centre and size, the rule this module replaces. */
  boundingBox?: { center: Vec3; size: Vec3 } | null;
  /** How far behind the source camera to stand, in world units. */
  stepBack?: number;
  /** Framing distance floor and ceiling, in world units. */
  minimumDistance?: number;
  maximumDistance?: number;
  /** Fraction of an extent's diagonal used as the framing distance. */
  distanceFactor?: number;
}

function frame(extent: Extent, source: StartPoseSource, input: StartPoseInput): StartPose {
  const factor = input.distanceFactor ?? 0.55;
  const minimum = input.minimumDistance ?? 0;
  const maximum = input.maximumDistance ?? Infinity;
  const distance = Math.min(maximum, Math.max(minimum, extent.diagonal * factor));
  const target: [number, number, number] = input.subject
    ? [input.subject[0], input.subject[1], input.subject[2]]
    : [...extent.center];
  // +z of the reconstruction frame is where the phone stood: every person PLY, cameras.json and the
  // fitted placement put camera 0 at the origin looking -z, so backing off along +z looks the way
  // the clip was shot rather than through a wall.
  return {
    position: [target[0], target[1], target[2] + distance],
    target,
    source,
  };
}

/**
 * Pick the start pose for a scene with no ?cam=. Routes are tried best first and every one of them
 * is checked for finite output, so a malformed cameras.json falls through to the framing rules
 * instead of throwing the viewer's start-up.
 */
export function chooseStartPose(input: StartPoseInput): StartPose | null {
  const stepBack = input.stepBack ?? 0;
  if (input.placement) {
    const pose = cameraWorldPose(firstSourceCamera(input.cameras), input.placement);
    if (pose) {
      const position: [number, number, number] = [
        pose.position[0] + pose.back[0] * stepBack,
        pose.position[1] + pose.back[1] * stepBack,
        pose.position[2] + pose.back[2] * stepBack,
      ];
      const distance = Math.max(input.minimumDistance ?? 0, 1);
      const target: [number, number, number] = input.subject
        ? [input.subject[0], input.subject[1], input.subject[2]]
        : [
            pose.position[0] + pose.forward[0] * distance,
            pose.position[1] + pose.forward[1] * distance,
            pose.position[2] + pose.forward[2] * distance,
          ];
      const apart = Math.hypot(
        target[0] - position[0],
        target[1] - position[1],
        target[2] - position[2],
      );
      if (position.every(Number.isFinite) && target.every(Number.isFinite) && apart > 1e-6) {
        return { position, target, source: 'source-camera' };
      }
    }
  }
  if (input.peopleExtent) return frame(input.peopleExtent, 'people', input);
  if (input.splatExtent) return frame(input.splatExtent, 'splat-extent', input);
  if (input.boundingBox) {
    const { center, size } = input.boundingBox;
    if ([...center, ...size].every((v) => FINITE(v))) {
      const diagonal = Math.hypot(size[0], size[1], size[2]);
      // the rule this module replaces, kept verbatim as the last resort
      return {
        position: [center[0], center[1], center[2] + diagonal * (input.distanceFactor ?? 0.55)],
        target: [center[0], center[1], center[2]],
        source: 'bounding-box',
      };
    }
  }
  return null;
}
