// Synthetic PLY bytes only; the same layout scripts/package_person_sequence.py writes.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseGaussianPly } from '../src/gaussian-ply.ts';

const FIELDS = [
  'x',
  'y',
  'z',
  'nx',
  'ny',
  'nz',
  'f_dc_0',
  'f_dc_1',
  'f_dc_2',
  'opacity',
  'scale_0',
  'scale_1',
  'scale_2',
  'rot_0',
  'rot_1',
  'rot_2',
  'rot_3',
];

function ply(n: number, props = FIELDS, format = 'binary_little_endian 1.0', type = 'float') {
  const header =
    `ply\nformat ${format}\ncomment synthetic\nelement vertex ${n}\n` +
    props.map((p) => `property ${type} ${p}\n`).join('') +
    'end_header\n';
  const head = new TextEncoder().encode(header);
  const data = new Float32Array(n * props.length);
  for (let i = 0; i < data.length; i++) data[i] = i * 0.25 - 3;
  const out = new Uint8Array(head.length + data.byteLength);
  out.set(head);
  out.set(new Uint8Array(data.buffer), head.length);
  return { buf: out.buffer, data };
}

test('reads the property list, count and floats of a packaged frame', () => {
  const { buf, data } = ply(7);
  const r = parseGaussianPly(buf);
  assert.deepEqual(r.props, FIELDS);
  assert.equal(r.n, 7);
  assert.deepEqual(Array.from(r.data), Array.from(data));
});

test('rejects an HTML answer, a text PLY, non-float properties and truncated data', () => {
  const html = new TextEncoder().encode('<!doctype html><html></html>').buffer;
  assert.throws(() => parseGaussianPly(html, 'frame_005.ply'), /frame_005\.ply: not a PLY/);
  assert.throws(() => parseGaussianPly(ply(2, FIELDS, 'ascii 1.0').buf), /little-endian/);
  assert.throws(() => parseGaussianPly(ply(2, FIELDS, undefined, 'uchar').buf), /not float/);
  const { buf } = ply(4);
  assert.throws(() => parseGaussianPly(buf.slice(0, buf.byteLength - 8)), /header promises/);
});

test('rejects ambiguous element layouts, duplicate or missing properties, and nonfinite values', () => {
  assert.throws(() => parseGaussianPly(ply(2, [...FIELDS, 'opacity']).buf), /malformed/);
  assert.throws(() => parseGaussianPly(ply(2, FIELDS.slice(1)).buf), /Gaussian properties/);
  assert.throws(() => parseGaussianPly(ply(0).buf), /malformed/);
  const { buf } = ply(2);
  const extra = new Uint8Array(buf.byteLength + 4);
  extra.set(new Uint8Array(buf));
  assert.throws(() => parseGaussianPly(extra.buffer), /header promises/);
  new DataView(buf).setFloat32(buf.byteLength - 4, NaN, true);
  assert.throws(() => parseGaussianPly(buf), /nonfinite/);
});
