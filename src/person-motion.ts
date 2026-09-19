// Decoder for the compact person motion track written by scripts/package_person_motion.py.
// The track holds, for every frame, the seven per-splat values that change between frames
// (centre x y z, rotation w x y z) in the frame PLYs' own order; frame 0's PLY stays the
// appearance keyframe. The viewer keeps one Float32Array(nS * 7) per frame ("keys"), possibly for
// a subset of splats (`sel`), so the decoder produces exactly those arrays.
export const MOTION_CHANNELS = 'x,y,z,rot_0,rot_1,rot_2,rot_3';

export type MotionRecord = {
  file: string;
  dtype: 'uint16' | 'float32';
  frames: number;
  splats: number;
  channels: string[];
  min?: number[];
  scale?: number[];
};

/** True when `record` describes a track for a sequence of `frames` PLYs holding `splats` each. */
export function motionTrackUsable(
  record: unknown,
  frames: number,
  splats: number,
): record is MotionRecord {
  const r = record as MotionRecord | null;
  return (
    !!r &&
    typeof r.file === 'string' &&
    (r.dtype === 'uint16' || r.dtype === 'float32') &&
    r.frames === frames &&
    r.splats === splats &&
    Array.isArray(r.channels) &&
    r.channels.join() === MOTION_CHANNELS &&
    (r.dtype === 'float32' ||
      (Array.isArray(r.min) &&
        r.min.length === 7 &&
        Array.isArray(r.scale) &&
        r.scale.length === 7))
  );
}

/** Byte length a well-formed track of this record must have. */
export function motionTrackBytes(record: MotionRecord): number {
  return record.frames * record.splats * 7 * (record.dtype === 'uint16' ? 2 : 4);
}

/**
 * Decodes frame `frame` of the track into a fresh Float32Array(nS * 7). `sel` maps each kept
 * splat to its index in the PLY (null keeps every splat, in order).
 */
export function decodeMotionFrame(
  buf: ArrayBuffer,
  record: MotionRecord,
  frame: number,
  sel: ArrayLike<number> | null,
  nS: number,
): Float32Array {
  if (buf.byteLength !== motionTrackBytes(record))
    throw new Error(
      `motion track is ${buf.byteLength} bytes, expected ${motionTrackBytes(record)}`,
    );
  if (frame < 0 || frame >= record.frames) throw new RangeError(`frame ${frame} not in track`);
  const raw = record.dtype === 'uint16' ? new Uint16Array(buf) : new Float32Array(buf);
  const lo = record.dtype === 'uint16' ? record.min! : null;
  const sc = record.dtype === 'uint16' ? record.scale! : null;
  const off = frame * record.splats * 7;
  const out = new Float32Array(nS * 7);
  for (let s = 0; s < nS; s++) {
    const src = off + (sel ? sel[s] : s) * 7,
      d = s * 7;
    for (let c = 0; c < 7; c++) out[d + c] = lo ? raw[src + c] * sc![c] + lo[c] : raw[src + c];
  }
  return out;
}
