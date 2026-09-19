// Reader for the packaged person frames: binary little-endian PLY with one `vertex` element whose
// properties are all float32 (x y z nx ny nz f_dc_0..2 opacity scale_0..2 rot_0..3). The viewer
// needs the raw floats, not a mesh, so this is deliberately narrower than Spark's loader.
export type GaussianPly = { props: string[]; n: number; data: Float32Array };

const HEADER_END = 'end_header\n';

/** Parses a person frame PLY, rejecting anything that is not the packaged binary float layout. */
export function parseGaussianPly(buf: ArrayBuffer, label = 'PLY'): GaussianPly {
  const head = new TextDecoder().decode(new Uint8Array(buf, 0, Math.min(4096, buf.byteLength)));
  const hi = head.indexOf(HEADER_END);
  // the dev server answers a missing file with index.html and status 200
  if (!head.startsWith('ply\n') || hi < 0) throw new Error(`${label}: not a PLY`);
  const lines = head.slice(0, hi).split('\n');
  if (!lines.includes('format binary_little_endian 1.0'))
    throw new Error(`${label}: not binary little-endian`);
  const elements = lines.filter((line) => line.startsWith('element '));
  if (elements.length !== 1 || !elements[0].startsWith('element vertex '))
    throw new Error(`${label}: expected only a vertex element`);
  const vertex = elements[0];
  if (!vertex) throw new Error(`${label}: no vertex element`);
  const n = Number(vertex.split(' ')[2]);
  const props: string[] = [];
  for (const line of lines) {
    if (!line.startsWith('property ')) continue;
    const [, type, name] = line.split(' ');
    if (type !== 'float') throw new Error(`${label}: property ${name} is ${type}, not float`);
    props.push(name);
  }
  if (!Number.isSafeInteger(n) || n <= 0 || !props.length || new Set(props).size !== props.length)
    throw new Error(`${label}: malformed vertex element`);
  for (const group of [
    ['x', 'y', 'z'],
    ['rot_0', 'rot_1', 'rot_2', 'rot_3'],
    ['scale_0', 'scale_1', 'scale_2'],
    ['f_dc_0', 'f_dc_1', 'f_dc_2'],
    ['opacity'],
  ]) {
    const start = props.indexOf(group[0]);
    if (start < 0 || group.some((name, i) => props[start + i] !== name))
      throw new Error(`${label}: missing or noncontiguous Gaussian properties ${group.join(',')}`);
  }
  const headerLen = new TextEncoder().encode(head.slice(0, hi + HEADER_END.length)).length;
  const bytes = n * props.length * 4;
  if (!Number.isSafeInteger(bytes) || buf.byteLength !== headerLen + bytes)
    throw new Error(`${label}: ${buf.byteLength} bytes, header promises ${headerLen + bytes}`);
  const data = new Float32Array(buf.slice(headerLen, headerLen + bytes));
  if (!data.every(Number.isFinite)) throw new Error(`${label}: nonfinite Gaussian attributes`);
  return { props, n, data };
}
