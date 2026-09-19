// Compact motion retains every frame/splat in PLY order; the first PLY supplies appearance.
// See docs/person-motion.md and scripts/package_person_motion.py for the cross-language contract.
export const MOTION_CHANNELS = 'x,y,z,rot_0,rot_1,rot_2,rot_3';
export const MOTION_SCHEMA = 'wander.person-motion/1';

export type MotionRecord = {
  schema: typeof MOTION_SCHEMA;
  file: string;
  dtype: 'uint16' | 'float32';
  frames: number;
  splats: number;
  channels: string[];
  frameFiles: string[];
  base: { file: string; bytes: number; sha256: string };
  bytes: number;
  sha256: string;
  min?: number[];
  scale?: number[];
};

const hashPattern = /^[a-f0-9]{64}$/;
const localFile = (v: unknown): v is string =>
  typeof v === 'string' && /^[a-zA-Z0-9_-][a-zA-Z0-9_.-]*$/.test(v);
const finiteSeven = (v: unknown): v is number[] =>
  Array.isArray(v) && v.length === 7 && v.every(Number.isFinite);

/** Old unbound tracks are deliberately refused; their original PLYs remain usable. */
export function motionTrackUsable(
  record: unknown,
  frames: number,
  splats: number,
): record is MotionRecord {
  const r = record as MotionRecord | null;
  return (
    !!r &&
    r.schema === MOTION_SCHEMA &&
    localFile(r.file) &&
    (r.dtype === 'uint16' || r.dtype === 'float32') &&
    Number.isSafeInteger(frames) &&
    frames > 0 &&
    Number.isSafeInteger(splats) &&
    splats > 0 &&
    r.frames === frames &&
    r.splats === splats &&
    Array.isArray(r.channels) &&
    r.channels.join() === MOTION_CHANNELS &&
    Array.isArray(r.frameFiles) &&
    r.frameFiles.length === frames &&
    r.frameFiles.every(localFile) &&
    !!r.base &&
    r.base.file === r.frameFiles[0] &&
    Number.isSafeInteger(r.base.bytes) &&
    r.base.bytes > 0 &&
    typeof r.base.sha256 === 'string' &&
    hashPattern.test(r.base.sha256) &&
    typeof r.sha256 === 'string' &&
    hashPattern.test(r.sha256) &&
    Number.isSafeInteger(r.bytes) &&
    r.bytes === motionTrackBytes(r) &&
    (r.dtype === 'float32' ||
      (finiteSeven(r.min) && finiteSeven(r.scale) && r.scale.every((v) => v >= 0)))
  );
}

export function motionTrackBytes(record: MotionRecord): number {
  return record.frames * record.splats * 7 * (record.dtype === 'uint16' ? 2 : 4);
}

export async function verifySha256(buf: ArrayBuffer, expected: string, label: string) {
  if (!hashPattern.test(expected)) throw new Error(`${label}: missing or invalid SHA-256`);
  const digest = await crypto.subtle.digest('SHA-256', buf);
  const actual = Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join(
    '',
  );
  if (actual !== expected) throw new Error(`${label}: SHA-256 mismatch`);
}

/** Decodes into the same float32 keys used by PLY playback, including signed zero. */
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
  if (!Number.isInteger(frame) || frame < 0 || frame >= record.frames)
    throw new RangeError(`frame ${frame} not in track`);
  if (!Number.isInteger(nS) || nS < 0 || nS > record.splats || (sel && sel.length !== nS))
    throw new RangeError('invalid splat selection');
  const raw = new DataView(buf);
  const quantized = record.dtype === 'uint16';
  const off = frame * record.splats * 7;
  const out = new Float32Array(nS * 7);
  for (let s = 0; s < nS; s++) {
    const index = sel ? sel[s] : s;
    if (!Number.isInteger(index) || index < 0 || index >= record.splats)
      throw new RangeError(`splat ${index} not in track`);
    const src = off + index * 7;
    for (let c = 0; c < 7; c++) {
      const value = quantized
        ? raw.getUint16((src + c) * 2, true) * record.scale![c] + record.min![c]
        : raw.getFloat32((src + c) * 4, true);
      out[s * 7 + c] = value;
      if (!Number.isFinite(out[s * 7 + c]))
        throw new Error('motion track contains nonfinite values');
    }
  }
  return out;
}

type LoadOptions = {
  record: unknown;
  frameFiles: string[];
  splats: number;
  basePly: ArrayBuffer;
  selection: ArrayLike<number> | null;
  selectedSplats: number;
  fetchPayload: (file: string) => Promise<ArrayBuffer>;
  installFrame: (frame: number, values: Float32Array) => void;
  fallback: () => Promise<void>;
  onFallback: (error: Error) => void;
  allowQuantized?: boolean;
};

/** Verify/decode everything before installing any compact key; fallback errors propagate. */
export async function loadMotionOrPlys(options: LoadOptions): Promise<'motion' | 'ply'> {
  const { record: r, frameFiles, splats } = options;
  let decoded: Float32Array[];
  try {
    if (!motionTrackUsable(r, frameFiles.length, splats))
      throw new Error('motion descriptor is missing required format or integrity metadata');
    if (r.dtype !== 'float32' && !options.allowQuantized)
      throw new Error('quantized motion requires explicit opt-in');
    if (r.frameFiles.some((file, i) => file !== frameFiles[i]))
      throw new Error('motion frame order does not match sequence');
    if (options.basePly.byteLength !== r.base.bytes)
      throw new Error('motion appearance keyframe size mismatch');
    await verifySha256(options.basePly, r.base.sha256, 'motion appearance keyframe');
    const buf = await options.fetchPayload(r.file);
    if (buf.byteLength !== r.bytes) throw new Error('motion payload size mismatch');
    await verifySha256(buf, r.sha256, 'motion payload');
    decoded = Array.from({ length: r.frames }, (_, frame) =>
      decodeMotionFrame(buf, r, frame, options.selection, options.selectedSplats),
    );
  } catch (cause) {
    options.onFallback(cause instanceof Error ? cause : new Error(String(cause)));
    await options.fallback();
    return 'ply';
  }
  // No partial compact state can survive a failed verification/decode.
  for (let i = 1; i < decoded.length; i++) options.installFrame(i, decoded[i]);
  return 'motion';
}
