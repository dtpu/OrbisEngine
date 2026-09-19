export interface BakedObjectTrack {
  fps: number;
  sourceFrames: number[];
  sampleIndex?: number[];
  positions: number[][];
  quaternionsXYZW?: number[][];
  visible?: boolean[];
  [key: string]: unknown;
}

export type BakedTrackValidation =
  { ok: true; track: BakedObjectTrack } | { ok: false; reason: string };

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function finiteVector(value: unknown, length: number): value is number[] {
  return (
    Array.isArray(value) &&
    value.length === length &&
    value.every((component) => typeof component === 'number' && Number.isFinite(component))
  );
}

/** Validate the baked track fields used by the viewer's interpolator. */
export function validateBakedTrack(value: unknown, driftLength?: number): BakedTrackValidation {
  if (!record(value)) return { ok: false, reason: 'track is not an object' };
  if (typeof value.fps !== 'number' || !Number.isFinite(value.fps) || value.fps <= 0)
    return { ok: false, reason: 'fps must be finite and positive' };
  if (!Array.isArray(value.sourceFrames) || value.sourceFrames.length === 0)
    return { ok: false, reason: 'sourceFrames must be a nonempty array' };
  for (let i = 0; i < value.sourceFrames.length; i++) {
    const frame = value.sourceFrames[i];
    if (!Number.isSafeInteger(frame) || frame < 0)
      return { ok: false, reason: 'sourceFrames must be nonnegative integers' };
    if (i > 0 && frame !== value.sourceFrames[i - 1] + 1)
      return { ok: false, reason: 'sourceFrames must be contiguous' };
  }
  if (!Array.isArray(value.positions) || value.positions.length !== value.sourceFrames.length)
    return { ok: false, reason: 'positions and sourceFrames must have matching lengths' };
  if (!value.positions.every((position) => finiteVector(position, 3)))
    return { ok: false, reason: 'positions must contain finite xyz vectors' };
  for (const field of ['sampleIndex'] as const) {
    const samples = value[field];
    if (samples !== undefined) {
      if (!Array.isArray(samples) || samples.length !== value.positions.length)
        return { ok: false, reason: `${field} and positions must have matching lengths` };
      if (!samples.every((sample) => typeof sample === 'number' && Number.isFinite(sample)))
        return { ok: false, reason: `${field} must contain finite numbers` };
      if (
        driftLength !== undefined &&
        (!Number.isSafeInteger(driftLength) ||
          driftLength <= 0 ||
          samples.some((sample) => sample < 0 || sample > driftLength - 1))
      )
        return { ok: false, reason: `${field} must stay within the camera drift table` };
    }
  }
  if (value.quaternionsXYZW !== undefined) {
    if (
      !Array.isArray(value.quaternionsXYZW) ||
      value.quaternionsXYZW.length !== value.positions.length
    )
      return { ok: false, reason: 'quaternionsXYZW and positions must have matching lengths' };
    if (!value.quaternionsXYZW.every((quaternion) => finiteVector(quaternion, 4)))
      return { ok: false, reason: 'quaternionsXYZW must contain finite xyzw vectors' };
    if (
      value.quaternionsXYZW.some((quaternion) => {
        const norm = Math.hypot(...quaternion);
        // Five-decimal packaged quaternions can have small rounding error. Slerp
        // assumes unit rotations; accepting materially scaled inputs distorts it.
        return Math.abs(norm - 1) > 1e-3;
      })
    )
      return { ok: false, reason: 'quaternionsXYZW must have near-unit nonzero norms' };
  }
  if (value.visible !== undefined) {
    if (!Array.isArray(value.visible) || value.visible.length !== value.positions.length)
      return { ok: false, reason: 'visible and positions must have matching lengths' };
    if (!value.visible.every((shown) => typeof shown === 'boolean'))
      return { ok: false, reason: 'visible must contain booleans' };
  }
  return { ok: true, track: value as BakedObjectTrack };
}
