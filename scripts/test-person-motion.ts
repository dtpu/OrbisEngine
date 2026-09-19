// Synthetic tracks only; checks the viewer-side decoder against the packager's written contract
// (scripts/package_person_motion.py: frame-major uint16 or float32, value = code * scale + min).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  decodeMotionFrame,
  motionTrackBytes,
  motionTrackUsable,
  type MotionRecord,
} from '../src/person-motion.ts';

const channels = ['x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];

function track(frames: number, splats: number, dtype: 'uint16' | 'float32') {
  const truth = new Float32Array(frames * splats * 7);
  for (let i = 0; i < truth.length; i++) truth[i] = Math.sin(i * 0.37) * (1 + (i % 7));
  const min = Array.from({ length: 7 }, (_, c) => -(1 + c));
  const scale = Array.from({ length: 7 }, (_, c) => (2 * (1 + c)) / 65535);
  const record: MotionRecord = {
    schema: 'wander.person-motion/1',
    file: 'motion.u16',
    dtype,
    frames,
    splats,
    channels,
    min,
    scale,
    frameFiles: Array.from({ length: frames }, (_, i) => `frame_${i}.ply`),
    base: { file: 'frame_0.ply', bytes: 1024, sha256: 'a'.repeat(64) },
    bytes: frames * splats * 7 * (dtype === 'float32' ? 4 : 2),
    sha256: 'b'.repeat(64),
  };
  let buf: ArrayBuffer;
  if (dtype === 'float32') {
    record.file = 'motion.f32';
    buf = truth.buffer.slice(0);
  } else {
    const codes = new Uint16Array(truth.length);
    for (let i = 0; i < truth.length; i++) {
      const c = i % 7;
      codes[i] = Math.round((truth[i] - min[c]) / scale[c]);
    }
    buf = codes.buffer.slice(0);
  }
  return { truth, record, buf };
}

test('uint16 frames decode to the packed values within the channel step', () => {
  const { truth, record, buf } = track(5, 11, 'uint16');
  assert.equal(motionTrackBytes(record), buf.byteLength);
  for (let f = 0; f < 5; f++) {
    const out = decodeMotionFrame(buf, record, f, null, 11);
    assert.equal(out.length, 11 * 7);
    for (let i = 0; i < out.length; i++) {
      const c = i % 7;
      assert.ok(
        Math.abs(out[i] - truth[f * 77 + i]) <= record.scale![c] / 2 + 1e-6,
        `frame ${f} value ${i}`,
      );
    }
  }
});

test('float32 frames decode bit-exactly and honour a splat selection', () => {
  const { truth, record, buf } = track(3, 8, 'float32');
  const sel = Int32Array.from([7, 2, 5]);
  const out = decodeMotionFrame(buf, record, 2, sel, 3);
  for (let s = 0; s < 3; s++)
    for (let c = 0; c < 7; c++) assert.equal(out[s * 7 + c], truth[2 * 56 + sel[s] * 7 + c]);
});

test('a record that does not match the sequence or a short buffer is refused', () => {
  const { record, buf } = track(4, 6, 'uint16');
  // a fresh object each time: the type predicate would otherwise narrow `record` away
  const rec = (patch: Partial<MotionRecord> = {}): unknown => ({ ...record, ...patch });
  assert.ok(motionTrackUsable(rec(), 4, 6));
  assert.ok(!motionTrackUsable(rec(), 5, 6));
  assert.ok(!motionTrackUsable(rec(), 4, 7));
  assert.ok(!motionTrackUsable(rec({ channels: channels.slice(1) }), 4, 6));
  assert.ok(!motionTrackUsable(rec({ min: undefined }), 4, 6));
  assert.ok(!motionTrackUsable(null, 4, 6));
  assert.ok(!motionTrackUsable(rec({ sha256: undefined }), 4, 6));
  assert.ok(!motionTrackUsable(rec({ base: undefined }), 4, 6));
  assert.ok(!motionTrackUsable(rec({ scale: [NaN, 0, 0, 0, 0, 0, 0] }), 4, 6));
  assert.ok(!motionTrackUsable(rec({ file: '../wrong.f32' }), 4, 6));
  assert.throws(() => decodeMotionFrame(buf.slice(0, 100), record, 0, null, 6), /100 bytes/);
  assert.throws(() => decodeMotionFrame(buf, record, 4, null, 6), RangeError);
  assert.throws(() => decodeMotionFrame(buf, record, 0, [-1], 1), RangeError);
  assert.throws(() => decodeMotionFrame(buf, record, 0, [6], 1), RangeError);
});
