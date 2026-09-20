export type Vec3 = [number, number, number];
export type BottleMode = 'recorded' | 'held' | 'free' | 'returned' | 'resting';

export type BottlePhysicsOptions = {
  /** Horizontal radius used for walls and hand reach checks. */
  radius: number;
  /**
   * Distance from the centre to the lowest visible point when resting on a horizontal support.
   * Defaults to `radius` for spherical props. A nonspherical visual can use its bounding sphere
   * for conservative floor clearance without making its wall collision artificially wide.
   */
  floorRadius?: number;
  gravity: number;
  maxSpeed: number;
  floorAt: (x: number, z: number) => number | null;
  blockedAt: (position: Vec3, radius: number) => boolean;
  /** Speeds below this on a supported surface become a resting bottle. */
  settleSpeed?: number;
  /** Downward speeds below this on a supported surface do not bounce. */
  bounceSpeed?: number;
  /** Horizontal speed decay while the bottle is supported. */
  groundDampingPerSecond?: number;
  /** Largest simulated duration accepted from one render frame. */
  maxFrameSeconds?: number;
};
type ReturnTarget = { position: Vec3; radius: number };
type BottleEvent = { type: 'returned' | 'dropped'; position: Vec3 };

const finite = (v: Vec3) => v.length === 3 && v.every(Number.isFinite);
const distance = (a: Vec3, b: Vec3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const lerp = (a: Vec3, b: Vec3, t: number): Vec3 => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t,
  a[2] + (b[2] - a[2]) * t,
];

const closestSegmentPoints = (a0: Vec3, a1: Vec3, b0: Vec3, b1: Vec3): [Vec3, Vec3] => {
  const u = a1.map((value, axis) => value - a0[axis]) as Vec3;
  const v = b1.map((value, axis) => value - b0[axis]) as Vec3;
  const w = a0.map((value, axis) => value - b0[axis]) as Vec3;
  const dot = (left: Vec3, right: Vec3) =>
    left.reduce((sum, value, i) => sum + value * right[i], 0);
  const uu = dot(u, u);
  const uv = dot(u, v);
  const vv = dot(v, v);
  const uw = dot(u, w);
  const vw = dot(v, w);
  const denominator = uu * vv - uv * uv;
  let s = denominator > Number.EPSILON ? (uv * vw - vv * uw) / denominator : 0;
  let t = denominator > Number.EPSILON ? (uu * vw - uv * uw) / denominator : vv > 0 ? vw / vv : 0;
  s = Math.max(0, Math.min(1, s));
  t = Math.max(0, Math.min(1, t));
  if (uu > 0) s = Math.max(0, Math.min(1, (uv * t - uw) / uu));
  if (vv > 0) t = Math.max(0, Math.min(1, (uv * s + vw) / vv));
  return [lerp(a0, a1, s), lerp(b0, b1, t)];
};

/** Small sphere approximation. Scene callbacks supply all supported geometry. */
export class BottlePhysics {
  private mode: BottleMode = 'recorded';
  private position: Vec3 = [0, 0, 0];
  private previousPosition: Vec3 = [0, 0, 0];
  private velocity: Vec3 = [0, 0, 0];
  private holder: string | null = null;
  private readonly options: BottlePhysicsOptions;

  constructor(options: BottlePhysicsOptions) {
    if (
      !Number.isFinite(options.radius) ||
      options.radius <= 0 ||
      (options.floorRadius !== undefined &&
        (!Number.isFinite(options.floorRadius) || options.floorRadius <= 0)) ||
      !Number.isFinite(options.gravity) ||
      options.gravity < 0 ||
      !Number.isFinite(options.maxSpeed) ||
      options.maxSpeed <= 0 ||
      (options.settleSpeed !== undefined &&
        (!Number.isFinite(options.settleSpeed) || options.settleSpeed < 0)) ||
      (options.bounceSpeed !== undefined &&
        (!Number.isFinite(options.bounceSpeed) || options.bounceSpeed < 0)) ||
      (options.groundDampingPerSecond !== undefined &&
        (!Number.isFinite(options.groundDampingPerSecond) || options.groundDampingPerSecond < 0)) ||
      (options.maxFrameSeconds !== undefined &&
        (!Number.isFinite(options.maxFrameSeconds) || options.maxFrameSeconds <= 0))
    ) {
      throw new Error('Bottle physics requires positive radius/speed and nonnegative gravity');
    }
    this.options = { ...options };
  }

  snapshot(): { mode: BottleMode; position: Vec3; velocity: Vec3; holder: string | null } {
    return {
      mode: this.mode,
      position: [...this.position],
      velocity: [...this.velocity],
      holder: this.holder,
    };
  }

  reset(position: Vec3): void {
    if (!finite(position)) return;
    this.mode = 'recorded';
    this.position = [...position];
    this.previousPosition = [...position];
    this.velocity = [0, 0, 0];
    this.holder = null;
  }

  setRecordedPosition(position: Vec3): void {
    if (this.mode === 'recorded' && finite(position)) {
      this.previousPosition = [...this.position];
      this.position = [...position];
    }
  }

  /** Synchronizes a recorded or accepted bottle to its character attachment without changing mode. */
  setAttachedPosition(position: Vec3): boolean {
    if ((this.mode !== 'recorded' && this.mode !== 'returned') || !finite(position)) return false;
    this.previousPosition = [...position];
    this.position = [...position];
    return true;
  }

  /**
   * Takes over an airborne recorded pose at a playback pause. A held bottle is owned by the
   * visitor and must not be replaced by the recorded track.
   */
  startFlight(position: Vec3, velocity: Vec3): boolean {
    if (this.mode === 'held' || !finite(position) || !finite(velocity)) return false;
    this.position = [...position];
    this.previousPosition = [...position];
    this.velocity = this.capped(velocity);
    this.holder = null;
    this.mode = 'free';
    return true;
  }

  /**
   * Checks a hand movement and the bottle movement from their preceding samples. This keeps a
   * fast bottle or controller from slipping through a one-frame reach test.
   */
  canGrabSegment(handPosition: Vec3, reach: number, previousHandPosition = handPosition): boolean {
    if (
      this.mode === 'held' ||
      !finite(handPosition) ||
      !finite(previousHandPosition) ||
      !Number.isFinite(reach) ||
      reach < 0
    )
      return false;
    const [bottlePoint, handPoint] = closestSegmentPoints(
      this.previousPosition,
      this.position,
      previousHandPosition,
      handPosition,
    );
    if (distance(bottlePoint, handPoint) > reach) return false;
    return (
      this.clearPath(this.previousPosition, bottlePoint) &&
      this.clearPath(previousHandPosition, handPoint) &&
      this.clearPath(bottlePoint, handPoint)
    );
  }

  grab(handId: string, handPosition: Vec3, reach: number, previousHandPosition?: Vec3): boolean {
    if (this.mode === 'held') return false;
    const validGrab =
      previousHandPosition === undefined
        ? finite(handPosition) &&
          Number.isFinite(reach) &&
          reach >= 0 &&
          distance(this.position, handPosition) <= reach &&
          this.clearPath(this.position, handPosition)
        : this.canGrabSegment(handPosition, reach, previousHandPosition);
    if (!handId || !validGrab) return false;
    this.mode = 'held';
    this.holder = handId;
    this.previousPosition = [...handPosition];
    this.position = [...handPosition];
    this.velocity = [0, 0, 0];
    return true;
  }

  moveHand(handId: string, position: Vec3, deltaSeconds: number): void {
    if (
      this.mode !== 'held' ||
      this.holder !== handId ||
      !finite(position) ||
      !Number.isFinite(deltaSeconds) ||
      deltaSeconds <= 0
    )
      return;
    const velocity: Vec3 = [
      (position[0] - this.position[0]) / deltaSeconds,
      (position[1] - this.position[1]) / deltaSeconds,
      (position[2] - this.position[2]) / deltaSeconds,
    ];
    if (!finite(velocity)) return;
    this.velocity = this.capped(velocity);
    this.previousPosition = [...this.position];
    this.position = [...position];
  }

  release(handId: string, velocity?: Vec3): boolean {
    if (this.mode !== 'held' || this.holder !== handId || (velocity && !finite(velocity)))
      return false;
    this.velocity = this.capped(velocity ?? this.velocity);
    this.mode = 'free';
    this.holder = null;
    return true;
  }

  private capped(velocity: Vec3): Vec3 {
    const largest = Math.max(...velocity.map(Math.abs));
    if (largest === 0) return [0, 0, 0];
    // Normalize first so even finite components near Number.MAX_VALUE cannot overflow.
    const unit = velocity.map((v) => v / largest) as Vec3;
    const norm = Math.hypot(...unit);
    if (largest <= this.options.maxSpeed / norm) return [...velocity];
    return unit.map((v) => (v / norm) * this.options.maxSpeed) as Vec3;
  }

  private blocked(position: Vec3): boolean {
    return this.options.blockedAt([...position], this.options.radius);
  }

  private get floorRadius(): number {
    return this.options.floorRadius ?? this.options.radius;
  }

  private clearPath(start: Vec3, end: Vec3): boolean {
    const stepLength = this.options.radius * 0.5;
    const steps = Math.ceil(distance(start, end) / stepLength);
    if (!Number.isFinite(steps) || steps > 4096) return false;
    for (let i = 0; i <= Math.max(1, steps); i++) {
      const point = lerp(start, end, i / Math.max(1, steps));
      const floor = this.options.floorAt(point[0], point[2]);
      if (
        this.blocked(point) ||
        (floor !== null && Number.isFinite(floor) && point[1] < floor + this.floorRadius)
      )
        return false;
    }
    return true;
  }

  private catchTarget(start: Vec3, end: Vec3, target?: ReturnTarget): boolean {
    if (!target || !finite(target.position) || !Number.isFinite(target.radius) || target.radius < 0)
      return false;
    const delta = end.map((v, i) => v - start[i]) as Vec3;
    const lengthSquared = delta.reduce((sum, v) => sum + v * v, 0);
    const dot = delta.reduce((sum, v, i) => sum + v * (target.position[i] - start[i]), 0);
    const t = lengthSquared ? Math.max(0, Math.min(1, dot / lengthSquared)) : 0;
    const nearest = lerp(start, end, t);
    if (distance(nearest, target.position) > target.radius + this.options.radius) return false;
    // A large catch volume must not pull the object through a wall to its center.
    if (!this.clearPath(nearest, target.position)) return false;
    this.mode = 'returned';
    this.position = [...target.position];
    this.velocity = [0, 0, 0];
    return true;
  }

  step(deltaSeconds: number, returnTarget?: ReturnTarget): BottleEvent[] {
    if (this.mode !== 'free' || !Number.isFinite(deltaSeconds) || deltaSeconds <= 0) return [];
    const { radius, gravity, maxSpeed, floorAt } = this.options;
    // Bound displacement per substep, including gravity. Extreme configurations lose time
    // instead of making an unbounded loop or skipping thin occupied surfaces.
    const duration = Math.min(
      deltaSeconds,
      this.options.maxFrameSeconds ?? 0.1,
      (radius * 0.5 * 4096) / maxSpeed,
    );
    const steps = Math.max(1, Math.ceil((maxSpeed * duration) / (radius * 0.5)));
    const dt = duration / steps;
    this.previousPosition = [...this.position];
    for (let i = 0; i < steps; i++) {
      this.velocity = this.capped([
        this.velocity[0],
        this.velocity[1] - gravity * dt,
        this.velocity[2],
      ]);
      const start: Vec3 = [...this.position];
      const next = start.map((v, axis) => v + this.velocity[axis] * dt) as Vec3;
      if (!finite(next)) return [];
      const floor = floorAt(next[0], next[2]);
      const grounded =
        floor !== null && Number.isFinite(floor) && next[1] <= floor + this.floorRadius;
      if (grounded) next[1] = floor + this.floorRadius;
      if (this.blocked(next)) {
        let bounced = false;
        for (let axis = 0; axis < 3; axis++) {
          const probe: Vec3 = [...start];
          probe[axis] = next[axis];
          if (this.blocked(probe)) {
            this.velocity[axis] *= -0.25;
            bounced = true;
          }
        }
        if (!bounced) this.velocity = this.velocity.map((v) => -v * 0.25) as Vec3;
        continue;
      }
      if (this.catchTarget(start, next, returnTarget))
        return [{ type: 'returned', position: [...this.position] }];
      this.position = next;
      if (grounded) {
        const scaleSpeed = Math.sqrt(gravity * radius);
        const bounceSpeed = this.options.bounceSpeed ?? scaleSpeed * 0.5;
        const settleSpeed = this.options.settleSpeed ?? scaleSpeed * 0.12;
        this.velocity[1] =
          Math.abs(this.velocity[1]) < bounceSpeed ? 0 : Math.abs(this.velocity[1]) * 0.25;
        const friction = Math.exp(-(this.options.groundDampingPerSecond ?? 12) * dt);
        this.velocity[0] *= friction;
        this.velocity[2] *= friction;
        if (Math.hypot(...this.velocity) < settleSpeed) {
          this.velocity = [0, 0, 0];
          this.mode = 'resting';
          return [{ type: 'dropped', position: [...this.position] }];
        }
      }
    }
    return [];
  }
}
