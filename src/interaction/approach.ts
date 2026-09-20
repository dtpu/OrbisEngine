import type { Vec3 } from './bottle-physics';

export type ApproachCandidate = { id: string; position: Vec3; eligible: boolean };

export type ApproachDetectorOptions = {
  enterDistance: number;
  exitDistance: number;
  dwellSeconds: number;
  /** Horizontal visitor movement required after a target has been armed. */
  minApproachDistance?: number;
  /** Minimum horizontal cosine between the visitor forward vector and the target direction. */
  facingCos?: number;
  /** Allow a deliberate approach from the spawn/reset position without first backing out. */
  allowInitialApproach?: boolean;
};

type CandidateState = {
  armed: boolean;
  armPosition: Vec3;
  armDistance: number;
};

type ActiveTarget = {
  id: string;
  dwellSeconds: number;
  armPosition: Vec3;
  armDistance: number;
};

const finite = (value: Vec3) => value.length === 3 && value.every(Number.isFinite);
const horizontalDistance = (left: Vec3, right: Vec3) =>
  Math.hypot(left[0] - right[0], left[2] - right[2]);
const horizontalDirection = (from: Vec3, to: Vec3): Vec3 | null => {
  const x = to[0] - from[0];
  const z = to[2] - from[2];
  const length = Math.hypot(x, z);
  return length > Number.EPSILON ? [x / length, 0, z / length] : null;
};

/**
 * Pure proximity trigger. Callers determine vertical compatibility and occlusion through each
 * candidate's `eligible` flag; this class only considers horizontal movement and facing.
 */
export class ApproachDetector {
  private readonly options: Required<ApproachDetectorOptions>;
  private readonly states = new Map<string, CandidateState>();
  private active: ActiveTarget | null = null;

  constructor(options: ApproachDetectorOptions) {
    const minApproachDistance = options.minApproachDistance ?? options.enterDistance * 0.08;
    const facingCos = options.facingCos ?? 0;
    if (
      !Number.isFinite(options.enterDistance) ||
      options.enterDistance <= 0 ||
      !Number.isFinite(options.exitDistance) ||
      options.exitDistance < options.enterDistance ||
      !Number.isFinite(options.dwellSeconds) ||
      options.dwellSeconds < 0 ||
      !Number.isFinite(minApproachDistance) ||
      minApproachDistance < 0 ||
      !Number.isFinite(facingCos) ||
      facingCos < -1 ||
      facingCos > 1
    ) {
      throw new Error(
        'Approach detection requires finite hysteresis, dwell, movement, and facing values',
      );
    }
    this.options = {
      ...options,
      minApproachDistance,
      facingCos,
      allowInitialApproach: options.allowInitialApproach ?? false,
    };
  }

  reset(_position?: Vec3): void {
    this.states.clear();
    this.active = null;
  }

  update(
    {
      position,
      forward,
      candidates,
    }: { position: Vec3; forward: Vec3; candidates: ApproachCandidate[] },
    deltaSeconds: number,
  ): string | null {
    if (!finite(position) || !finite(forward) || !Number.isFinite(deltaSeconds) || deltaSeconds < 0)
      return null;
    const unique = new Map<string, ApproachCandidate>();
    for (const candidate of candidates) {
      if (!candidate.id || !finite(candidate.position)) continue;
      const current = unique.get(candidate.id);
      if (
        !current ||
        horizontalDistance(position, candidate.position) <
          horizontalDistance(position, current.position)
      )
        unique.set(candidate.id, candidate);
    }
    const measured = [...unique.values()].map((candidate) => ({
      candidate,
      distance: horizontalDistance(position, candidate.position),
    }));

    for (const { candidate, distance } of measured) {
      const state = this.states.get(candidate.id);
      if (!state) {
        this.states.set(candidate.id, {
          armed: this.options.allowInitialApproach || distance > this.options.exitDistance,
          armPosition: [...position],
          armDistance: distance,
        });
      } else if (distance > this.options.exitDistance && !state.armed) {
        state.armed = true;
        state.armPosition = [...position];
        state.armDistance = distance;
      }
    }

    const active = this.active;
    if (active) {
      const current = measured.find(({ candidate }) => candidate.id === active.id);
      if (
        !current ||
        !this.canDwell(position, forward, current.candidate, current.distance) ||
        !this.hasApproached(position, current.distance, active)
      ) {
        this.active = null;
      } else {
        active.dwellSeconds += deltaSeconds;
        if (this.completedApproach(position, current.distance, active)) {
          const state = this.states.get(active.id)!;
          state.armed = false;
          this.active = null;
          return active.id;
        }
        return null;
      }
    }

    const next = measured
      .filter(({ candidate, distance }) => {
        const state = this.states.get(candidate.id)!;
        return (
          state.armed &&
          this.hasApproached(position, distance, state) &&
          this.canDwell(position, forward, candidate, distance)
        );
      })
      .sort(
        (left, right) =>
          left.distance - right.distance || left.candidate.id.localeCompare(right.candidate.id),
      )[0];
    if (!next) return null;
    const state = this.states.get(next.candidate.id)!;
    this.active = {
      id: next.candidate.id,
      dwellSeconds: deltaSeconds,
      armPosition: [...state.armPosition],
      armDistance: state.armDistance,
    };
    if (this.completedApproach(position, next.distance, this.active)) {
      state.armed = false;
      const result = this.active.id;
      this.active = null;
      return result;
    }
    return null;
  }

  private canDwell(
    position: Vec3,
    forward: Vec3,
    candidate: ApproachCandidate,
    distance: number,
  ): boolean {
    if (!candidate.eligible || distance > this.options.enterDistance) return false;
    const targetDirection = horizontalDirection(position, candidate.position);
    const visitorDirection = horizontalDirection([0, 0, 0], forward);
    return (
      targetDirection !== null &&
      visitorDirection !== null &&
      targetDirection[0] * visitorDirection[0] + targetDirection[2] * visitorDirection[2] >=
        this.options.facingCos
    );
  }

  private completedApproach(position: Vec3, distance: number, active: ActiveTarget): boolean {
    return (
      active.dwellSeconds >= this.options.dwellSeconds &&
      this.hasApproached(position, distance, active)
    );
  }

  private hasApproached(
    position: Vec3,
    distance: number,
    armed: Pick<CandidateState, 'armPosition' | 'armDistance'>,
  ) {
    return (
      horizontalDistance(position, armed.armPosition) >= this.options.minApproachDistance &&
      armed.armDistance - distance >= this.options.minApproachDistance
    );
  }
}
