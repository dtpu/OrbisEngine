import { Vector3 } from 'three';

/** First supported floor crossed by a downward ray. Unknown cells cannot be landing surfaces.
 * Sample at most half a navigation cell apart, then refine the first crossing, including risers.
 * Heights come from the existing navigation data; this does not reconstruct missing geometry.
 */
export function raycastWalkFloor(
  origin: Vector3,
  direction: Vector3,
  floorAt: (x: number, z: number) => number,
  step: number,
  maxDistance: number,
  out: Vector3,
): boolean {
  if (
    !origin.toArray().every(Number.isFinite) ||
    !direction.toArray().every(Number.isFinite) ||
    direction.y >= -1e-3 ||
    !Number.isFinite(step) ||
    step <= 0 ||
    !Number.isFinite(maxDistance) ||
    maxDistance <= 0
  )
    return false;
  const point = new Vector3();
  const sample = (distance: number) => {
    point.copy(origin).addScaledVector(direction, distance);
    const floor = floorAt(point.x, point.z);
    return Number.isFinite(floor) && floor <= origin.y && point.y <= floor ? floor : NaN;
  };
  let previous = 0;
  for (
    let distance = Math.min(step, maxDistance);
    ;
    distance = Math.min(distance + step, maxDistance)
  ) {
    let floor = sample(distance);
    if (Number.isFinite(floor)) {
      let low = previous,
        high = distance;
      for (let i = 0; i < 12; i++) {
        const middle = (low + high) / 2;
        const candidate = sample(middle);
        if (Number.isFinite(candidate)) {
          high = middle;
          floor = candidate;
        } else low = middle;
      }
      out.copy(origin).addScaledVector(direction, high).setY(floor);
      return true;
    }
    if (distance >= maxDistance) return false;
    previous = distance;
  }
}
