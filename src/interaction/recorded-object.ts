import { Quaternion, Vector3 } from 'three';

export type RecordedSegment = {
  kind: string;
  fromSourceFrame: number;
  toSourceFrame: number;
  parent?: string;
  jointName?: string;
  throwerTrack?: number;
  catcherTrack?: number;
};
export type ObjectMetadata = {
  objectClass?: string;
  appearance?: { sizeWorldUnits?: number[] };
  pose?: { segments?: RecordedSegment[] };
};
export type HeadTrack = { positions: number[][]; forward?: number[][]; times: number[] };
const finiteVector = (v: unknown): v is number[] =>
  Array.isArray(v) && v.length === 3 && v.every(Number.isFinite);

export function parseHeadTrack(value: unknown, times: number[]): HeadTrack | null {
  if (!value || typeof value !== 'object') return null;
  const data = value as { positions?: unknown; forward?: unknown; times?: unknown };
  const clock = Array.isArray(data.times) ? data.times : times;
  if (
    !Array.isArray(data.positions) ||
    data.positions.length !== clock.length ||
    !clock.length ||
    !clock.every((t, i) => Number.isFinite(t) && (!i || t > clock[i - 1])) ||
    !data.positions.every(finiteVector)
  )
    return null;
  const forward =
    Array.isArray(data.forward) &&
    data.forward.length === clock.length &&
    data.forward.every(finiteVector)
      ? data.forward
      : undefined;
  return { positions: data.positions, times: clock, forward };
}

export function sampleVector(values: number[][], times: number[], time: number): Vector3 {
  let i = 0;
  while (i + 1 < times.length && times[i + 1] <= time) i++;
  const j = Math.min(i + 1, times.length - 1);
  const u = i === j ? 0 : Math.max(0, Math.min(1, (time - times[i]) / (times[j] - times[i])));
  return new Vector3().fromArray(values[i]).lerp(new Vector3().fromArray(values[j]), u);
}

export function segments(meta: ObjectMetadata): RecordedSegment[] {
  return (Array.isArray(meta.pose?.segments) ? meta.pose.segments : []).filter(
    (s) =>
      s &&
      Number.isFinite(s.fromSourceFrame) &&
      Number.isFinite(s.toSourceFrame) &&
      s.toSourceFrame >= s.fromSourceFrame &&
      ['attached', 'free', 'blend'].includes(s.kind),
  );
}

export function recordedState(
  meta: ObjectMetadata,
  fps: number,
  time: number,
  people: Array<{ id: string; track?: number }>,
) {
  const frame = time * fps;
  const spans = segments(meta);
  const flight = spans.find(
    (s) => s.kind === 'free' && frame >= s.fromSourceFrame && frame <= s.toSourceFrame,
  );
  if (flight)
    return {
      free: true,
      owner: null,
      thrower: people.find((p) => p.track === flight.throwerTrack)?.id ?? null,
      recipient: people.find((p) => p.track === flight.catcherTrack)?.id ?? null,
    };
  const held = spans
    .filter((s) => s.kind === 'attached' && s.parent)
    .sort((a, b) => distanceToSpan(a, frame) - distanceToSpan(b, frame))[0];
  return {
    free: false,
    owner: held?.parent ?? null,
    thrower: held?.parent ?? null,
    recipient: null,
  };
}

function distanceToSpan(s: RecordedSegment, frame: number) {
  return Math.max(s.fromSourceFrame - frame, 0, frame - s.toSourceFrame);
}

export function nearestHeldTime(
  meta: ObjectMetadata,
  fps: number,
  time: number,
  personId: string,
): number | null {
  const span = segments(meta)
    .filter((s) => s.kind === 'attached' && s.parent === personId)
    .sort((a, b) => distanceToSpan(a, time * fps) - distanceToSpan(b, time * fps))[0];
  return span
    ? Math.max(span.fromSourceFrame, Math.min(span.toSourceFrame, time * fps)) / fps
    : null;
}

export function yawBetween(from: Vector3, to: Vector3): Quaternion {
  const a = Math.atan2(from.x, from.z),
    b = Math.atan2(to.x, to.z);
  return new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), b - a);
}
