import { Matrix4, Quaternion, Vector3 } from 'three';

export interface StaticCollider {
  id: string;
  role: string;
  center: Vector3;
  halfExtents: Vector3;
  quaternion: Quaternion;
  localToWorld: Matrix4;
  worldToLocal: Matrix4;
  walkable: boolean;
  provenance: string;
}

const finiteVector = (value: unknown, length: number): value is number[] =>
  Array.isArray(value) &&
  value.length === length &&
  value.every((v) => typeof v === 'number' && Number.isFinite(v));

export function parseStaticColliders(value: unknown, world: string): StaticCollider[] {
  if (!value || typeof value !== 'object') throw new Error('Invalid collider manifest');
  const manifest = value as Record<string, unknown>;
  const filename = (path: string) => new URL(path, 'http://localhost').pathname.split('/').at(-1);
  if (
    manifest.schema !== 'wander.colliders/1' ||
    manifest.coordinates !== 'viewer-world' ||
    typeof manifest.world !== 'string' ||
    filename(manifest.world) !== filename(world) ||
    !Array.isArray(manifest.bodies) ||
    manifest.bodies.length === 0
  )
    throw new Error('Collider schema, world, coordinates, or bodies do not match');
  const ids = new Set<string>();
  return manifest.bodies.map((entry: unknown) => {
    if (!entry || typeof entry !== 'object') throw new Error('Invalid collider body');
    const b = entry as Record<string, unknown>;
    if (
      typeof b.id !== 'string' ||
      !b.id ||
      ids.has(b.id) ||
      typeof b.role !== 'string' ||
      b.shape !== 'box' ||
      !finiteVector(b.center, 3) ||
      !finiteVector(b.halfExtents, 3) ||
      b.halfExtents.some((v) => v <= 0) ||
      !finiteVector(b.quaternionXYZW, 4) ||
      Math.abs(Math.hypot(...b.quaternionXYZW) - 1) > 0.01 ||
      typeof b.walkable !== 'boolean' ||
      typeof b.provenance !== 'string' ||
      !b.provenance.trim()
    )
      throw new Error('Invalid or duplicate collider geometry/provenance');
    ids.add(b.id);
    const center = new Vector3(...b.center);
    const halfExtents = new Vector3(...b.halfExtents);
    const quaternion = new Quaternion(...b.quaternionXYZW).normalize();
    const localToWorld = new Matrix4().compose(center, quaternion, new Vector3(1, 1, 1));
    return {
      id: b.id,
      role: b.role,
      center,
      halfExtents,
      quaternion,
      localToWorld,
      worldToLocal: localToWorld.clone().invert(),
      walkable: b.walkable,
      provenance: b.provenance,
    };
  });
}

/** The finite box's local +Y face is also the offline actor's support surface. */
export function supportPlane(box: StaticCollider) {
  const normal = new Vector3(0, 1, 0).applyQuaternion(box.quaternion);
  const point = new Vector3(0, box.halfExtents.y, 0).applyMatrix4(box.localToWorld);
  return { normal, point, d: normal.dot(point) };
}

export function supportHeight(box: StaticCollider, x: number, z: number): number | undefined {
  if (!box.walkable || !Number.isFinite(x) || !Number.isFinite(z)) return;
  const { normal, d } = supportPlane(box);
  if (normal.y <= 0.1) return;
  const y = (d - normal.x * x - normal.z * z) / normal.y;
  const local = new Vector3(x, y, z).applyMatrix4(box.worldToLocal);
  const eps = 1e-7;
  if (Math.abs(local.x) <= box.halfExtents.x + eps && Math.abs(local.z) <= box.halfExtents.z + eps)
    return y;
}

export function highestSupport(
  boxes: StaticCollider[],
  x: number,
  z: number,
  maximum = Infinity,
  radius = 0,
): number | undefined {
  let best: number | undefined;
  for (const box of boxes) {
    const surface = supportHeight(box, x, z);
    // A vertical capsule touches a tilted plane on its lower hemisphere.
    const normal = supportPlane(box).normal;
    const y = surface === undefined ? undefined : surface + radius * (1 / normal.y - 1);
    if (y === undefined) continue;
    const tangent = new Vector3(x, y + radius, z)
      .addScaledVector(normal, -radius)
      .applyMatrix4(box.worldToLocal);
    if (
      Math.abs(tangent.x) <= box.halfExtents.x + 1e-7 &&
      Math.abs(tangent.z) <= box.halfExtents.z + 1e-7 &&
      y <= maximum &&
      (best === undefined || y > best)
    )
      best = y;
  }
  return best;
}

function segmentBoxDistanceSquared(a: Vector3, b: Vector3, half: Vector3) {
  const start = a.toArray(),
    delta = b.clone().sub(a).toArray(),
    size = half.toArray();
  const cuts = [0, 1];
  for (let k = 0; k < 3; k++) {
    if (Math.abs(delta[k]) < 1e-15) continue;
    for (const face of [-size[k], size[k]]) {
      const t = (face - start[k]) / delta[k];
      if (t > 0 && t < 1) cuts.push(t);
    }
  }
  cuts.sort((a, b) => a - b);
  const distance = (t: number) =>
    start.reduce((sum, v, k) => {
      const outside = Math.max(0, Math.abs(v + delta[k] * t) - size[k]);
      return sum + outside * outside;
    }, 0);
  let best = Math.min(distance(0), distance(1));
  for (let i = 1; i < cuts.length; i++) {
    const lo = cuts[i - 1],
      hi = cuts[i],
      mid = (lo + hi) / 2;
    let numerator = 0,
      denominator = 0;
    for (let k = 0; k < 3; k++) {
      const v = start[k] + delta[k] * mid;
      if (Math.abs(v) <= size[k]) continue;
      numerator += delta[k] * (Math.sign(v) * size[k] - start[k]);
      denominator += delta[k] ** 2;
    }
    const t = denominator ? Math.max(lo, Math.min(hi, numerator / denominator)) : mid;
    best = Math.min(best, distance(t));
  }
  return best;
}

export function capsuleIntersects(
  box: StaticCollider,
  foot: Vector3,
  height: number,
  radius: number,
) {
  if (!finiteVector(foot.toArray(), 3)) throw new Error('Invalid walker position');
  if (!(height > 0 && radius > 0 && Number.isFinite(height) && Number.isFinite(radius)))
    throw new Error('Invalid walker capsule');
  radius = Math.min(radius, height / 2);
  const a = foot
    .clone()
    .add(new Vector3(0, radius, 0))
    .applyMatrix4(box.worldToLocal);
  const b = foot
    .clone()
    .add(new Vector3(0, height - radius, 0))
    .applyMatrix4(box.worldToLocal);
  return segmentBoxDistanceSquared(a, b, box.halfExtents) < (radius - 1e-7) ** 2;
}

/** Expanded-box broad phase, then minimum distance of the moving capsule axis.
 * Distance to a convex box under translation is convex in time. Golden-section
 * refinement avoids treating the broad phase's square corners as real contacts.
 * A Lipschitz bound keeps an unresolved near-contact interval conservative.
 */
export function capsuleSweepBlocked(
  boxes: StaticCollider[],
  from: Vector3,
  to: Vector3,
  height: number,
  radius: number,
) {
  if (!boxes.length) return false;
  if (!finiteVector(from.toArray(), 3) || !finiteVector(to.toArray(), 3))
    throw new Error('Invalid walker sweep');
  if (!(height > 0 && radius > 0 && Number.isFinite(height) && Number.isFinite(radius)))
    throw new Error('Invalid walker capsule');
  radius = Math.min(radius, height / 2);
  const offset = new Vector3(0, height / 2, 0);
  for (const box of boxes) {
    const a = from.clone().add(offset).applyMatrix4(box.worldToLocal);
    const b = to.clone().add(offset).applyMatrix4(box.worldToLocal);
    const vertical = new Vector3(0, 1, 0).applyQuaternion(box.quaternion.clone().invert());
    const half = box.halfExtents.clone();
    for (const k of ['x', 'y', 'z'] as const)
      half[k] += Math.abs(vertical[k]) * (height / 2 - radius) + radius - 1e-7;
    const delta = b.sub(a);
    let enter = 0,
      leave = 1;
    for (const k of ['x', 'y', 'z'] as const) {
      if (Math.abs(delta[k]) < 1e-15) {
        if (Math.abs(a[k]) >= half[k]) {
          enter = 2;
          break;
        }
      } else {
        const t0 = (-half[k] - a[k]) / delta[k],
          t1 = (half[k] - a[k]) / delta[k];
        enter = Math.max(enter, Math.min(t0, t1));
        leave = Math.min(leave, Math.max(t0, t1));
      }
    }
    if (enter > leave || leave < 0 || enter > 1) continue;
    const axisHalf = vertical.multiplyScalar(height / 2 - radius);
    const distance = (t: number) => {
      const midpoint = a.clone().addScaledVector(delta, t);
      return Math.sqrt(
        segmentBoxDistanceSquared(
          midpoint.clone().sub(axisHalf),
          midpoint.add(axisHalf),
          box.halfExtents,
        ),
      );
    };
    const threshold = Math.max(0, radius - 1e-7);
    let lo = Math.max(0, enter),
      hi = Math.min(1, leave);
    if (distance(lo) < threshold || distance(hi) < threshold) return true;
    const ratio = (Math.sqrt(5) - 1) / 2;
    let left = hi - ratio * (hi - lo),
      right = lo + ratio * (hi - lo),
      dl = distance(left),
      dr = distance(right);
    for (let iteration = 0; iteration < 48; iteration++) {
      if (Math.min(dl, dr) < threshold) return true;
      if (dl < dr) {
        hi = right;
        right = left;
        dr = dl;
        left = hi - ratio * (hi - lo);
        dl = distance(left);
      } else {
        lo = left;
        left = right;
        dl = dr;
        right = lo + ratio * (hi - lo);
        dr = distance(right);
      }
    }
    const lowerBound = distance((lo + hi) / 2) - (delta.length() * (hi - lo)) / 2;
    if (lowerBound < threshold) return true;
  }
  return false;
}
