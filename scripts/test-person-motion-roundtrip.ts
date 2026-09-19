// Cross-toolchain check: frames packed by scripts/package_person_motion.py (Python) decode in the
// viewer's TypeScript decoder to the values the PLYs hold. Synthetic frames only, written to a
// temporary directory; this is the contract the other two suites each assume from their own side.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { parseGaussianPly } from '../src/gaussian-ply.ts';
import {
  decodeMotionFrame,
  motionTrackBytes,
  motionTrackUsable,
  type MotionRecord,
} from '../src/person-motion.ts';

const FIELDS = [
  'x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
  'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3',
]; // prettier-ignore
const CHANNELS = ['x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];
const FRAMES = 5,
  SPLATS = 40;

function plyBytes(rows: Float32Array): Uint8Array {
  const header =
    `ply\nformat binary_little_endian 1.0\nelement vertex ${SPLATS}\n` +
    FIELDS.map((p) => `property float ${p}\n`).join('') +
    'end_header\n';
  const head = new TextEncoder().encode(header);
  const out = new Uint8Array(head.length + rows.byteLength);
  out.set(head);
  out.set(new Uint8Array(rows.buffer, rows.byteOffset, rows.byteLength), head.length);
  return out;
}

/** A person directory whose frames share appearance and differ only in centre and rotation. */
function writePerson(): string {
  const dir = mkdtempSync(join(tmpdir(), 'wander-motion-'));
  const base = new Float32Array(SPLATS * 17);
  for (let i = 0; i < base.length; i++) base[i] = Math.sin(i * 0.731) * 3;
  for (let s = 0; s < SPLATS; s++) for (let c = 3; c < 6; c++) base[s * 17 + c] = 0;
  const names: string[] = [];
  for (let f = 0; f < FRAMES; f++) {
    const rows = base.slice();
    for (let s = 0; s < SPLATS; s++) {
      for (let c = 0; c < 3; c++) rows[s * 17 + c] += 0.02 * f * Math.cos(s + c);
      const q = [0, 1, 2, 3].map((k) => rows[s * 17 + 13 + k] + 0.01 * f * (k + 1));
      const n = Math.hypot(...q);
      for (let k = 0; k < 4; k++) rows[s * 17 + 13 + k] = q[k] / n;
    }
    const name = `frame_${String(f).padStart(3, '0')}.ply`;
    writeFileSync(join(dir, name), plyBytes(rows));
    names.push(name);
  }
  writeFileSync(join(dir, 'sequence.json'), JSON.stringify({ frames: names, fps: 12 }));
  return dir;
}

function pack(dir: string, lossless: boolean) {
  const args = ['run', '--locked', 'scripts/package_person_motion.py', dir];
  if (lossless) args.push('--lossless');
  const run = spawnSync('uv', args, { encoding: 'utf8' });
  assert.equal(run.status, 0, `packager failed:\n${run.stdout}\n${run.stderr}`);
  const seq = JSON.parse(readFileSync(join(dir, 'sequence.json'), 'utf8'));
  assert.ok(motionTrackUsable(seq.motion, FRAMES, SPLATS), JSON.stringify(seq.motion));
  const record = seq.motion as MotionRecord & { maxAbsError: number[] };
  const raw = readFileSync(join(dir, record.file));
  const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  assert.equal(buf.byteLength, motionTrackBytes(record));
  return { record, buf };
}

function plyChannels(dir: string, frame: number): Float32Array {
  const name = `frame_${String(frame).padStart(3, '0')}.ply`;
  const raw = readFileSync(join(dir, name));
  const ply = parseGaussianPly(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
  const cols = CHANNELS.map((c) => ply.props.indexOf(c));
  const out = new Float32Array(ply.n * 7);
  for (let s = 0; s < ply.n; s++)
    for (let c = 0; c < 7; c++) out[s * 7 + c] = ply.data[s * ply.props.length + cols[c]];
  return out;
}

for (const lossless of [false, true]) {
  test(`frames packed by the Python packager decode in the viewer (${lossless ? 'float32' : 'uint16'})`, () => {
    const dir = writePerson();
    try {
      const { record, buf } = pack(dir, lossless);
      assert.equal(record.dtype, lossless ? 'float32' : 'uint16');
      const worst = new Array(7).fill(0); // per channel, like the record's maxAbsError
      for (let f = 0; f < FRAMES; f++) {
        const got = decodeMotionFrame(buf, record, f, null, SPLATS);
        const want = plyChannels(dir, f);
        for (let i = 0; i < want.length; i++)
          worst[i % 7] = Math.max(worst[i % 7], Math.abs(got[i] - want[i]));
      }
      if (lossless) assert.deepEqual(worst, new Array(7).fill(0));
      else
        worst.forEach((w, c) => {
          // the decoder works in float32 where the packager measured in float64: allow a few ulp
          assert.ok(
            w <= record.maxAbsError[c] + 1e-6,
            `channel ${c}: ${w} > ${record.maxAbsError[c]}`,
          );
          assert.ok(w < 2e-4, `channel ${c}: quantisation error ${w} too large for a 6 u range`);
        });
      // a splat selection reads the same rows the PLY holds at those indices
      const sel = [3, 17, SPLATS - 1];
      const part = decodeMotionFrame(buf, record, FRAMES - 1, sel, sel.length);
      const full = decodeMotionFrame(buf, record, FRAMES - 1, null, SPLATS);
      sel.forEach((s, i) => {
        for (let c = 0; c < 7; c++) assert.equal(part[i * 7 + c], full[s * 7 + c]);
      });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
}
