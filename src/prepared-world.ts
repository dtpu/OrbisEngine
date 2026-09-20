import { PackedSplats, type PackedSplatsOptions } from '@sparkjsdev/spark';

// Versioned cache of Spark's existing packed representation, never a re-encoding.
const MAGIC = 0x31575057;
const SCHEMA = 1;
const MAX_HEADER = 1024 * 1024;
const MAX_BYTES = 1024 * 1024 * 1024;
const MAX_DEPTH = 8;
const ARRAY_KEYS = ['sh1', 'sh2', 'sh3', 'sh1Codes', 'sh2Codes', 'sh3Codes', 'lodTree'] as const;
type Identity = { sourceHash: string; buildId: string };
type Range = { offset: number; length: number };
type Saved = {
  packedArray: Range | null;
  numSplats: number;
  maxSplats: number;
  maxSh: number;
  lod?: boolean | 'quality';
  nonLod?: boolean;
  splatEncoding?: PackedSplatsOptions['splatEncoding'];
  extra: Partial<Record<(typeof ARRAY_KEYS)[number], Range>> & { radMeta?: unknown };
  lodSplats?: Saved;
};
type Header = Identity & { schema: number; hash: string; payloadBytes: number; root: Saved };

function check(ok: unknown, message: string): asserts ok {
  if (!ok) throw new Error(`Invalid prepared world: ${message}`);
}
function integer(value: unknown, maximum = MAX_BYTES): asserts value is number {
  check(
    Number.isSafeInteger(value) && (value as number) >= 0 && (value as number) <= maximum,
    'dimension',
  );
}
function identity(value: Identity) {
  check(/^[a-f0-9]{64}$/.test(value.sourceHash), 'source hash');
  check(
    typeof value.buildId === 'string' && value.buildId.length > 0 && value.buildId.length <= 512,
    'build ID',
  );
}
function plain(value: unknown, depth = 0): void {
  check(depth <= 16, 'metadata depth');
  if (typeof value === 'number') check(Number.isFinite(value), 'nonfinite metadata');
  else if (value && typeof value === 'object') {
    check(
      Array.isArray(value) || Object.getPrototypeOf(value) === Object.prototype,
      'metadata object',
    );
    for (const child of Object.values(value)) plain(child, depth + 1);
  } else
    check(
      value == null || ['string', 'boolean', 'undefined'].includes(typeof value),
      'metadata value',
    );
}
async function digest(bytes: Uint8Array<ArrayBuffer>) {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (v) =>
    v.toString(16).padStart(2, '0'),
  ).join('');
}
function validate(header: Header, expected: Identity, payloadBytes: number) {
  identity(expected);
  check(header.schema === SCHEMA, 'schema');
  check(
    header.sourceHash === expected.sourceHash && header.buildId === expected.buildId,
    'identity mismatch',
  );
  check(typeof header.hash === 'string' && /^[a-f0-9]{64}$/.test(header.hash), 'payload hash');
  check(header.payloadBytes === payloadBytes, 'payload size');
  plain(header);
  const ranges: Range[] = [];
  function range(value: Range) {
    check(value && typeof value === 'object', 'array range');
    integer(value.offset, payloadBytes);
    integer(value.length, payloadBytes / 4);
    check(
      value.offset % 4 === 0 && value.offset + value.length * 4 <= payloadBytes,
      'array bounds',
    );
    if (value.length) ranges.push(value);
  }
  function node(value: Saved, depth: number) {
    check(value && typeof value === 'object' && depth <= MAX_DEPTH, 'splat nesting');
    for (const key of Object.keys(value))
      check(
        [
          'packedArray',
          'numSplats',
          'maxSplats',
          'maxSh',
          'lod',
          'nonLod',
          'splatEncoding',
          'extra',
          'lodSplats',
        ].includes(key),
        'splat field',
      );
    integer(value.numSplats, MAX_BYTES / 16);
    integer(value.maxSplats, MAX_BYTES / 16);
    integer(value.maxSh, 3);
    check(
      value.lod === undefined || typeof value.lod === 'boolean' || value.lod === 'quality',
      'lod',
    );
    check(value.nonLod === undefined || typeof value.nonLod === 'boolean', 'nonLod');
    if (value.splatEncoding !== undefined) {
      check(value.splatEncoding !== null && typeof value.splatEncoding === 'object', 'encoding');
      for (const [key, entry] of Object.entries(value.splatEncoding)) {
        check(
          key === 'lodOpacity'
            ? typeof entry === 'boolean'
            : [
                'rgbMin',
                'rgbMax',
                'lnScaleMin',
                'lnScaleMax',
                'sh1Max',
                'sh2Max',
                'sh3Max',
              ].includes(key) &&
                typeof entry === 'number' &&
                Number.isFinite(entry),
          'encoding field',
        );
      }
    }
    check(value.numSplats <= value.maxSplats, 'splat count');
    if (value.packedArray !== null) {
      range(value.packedArray);
      check(
        value.packedArray.length % 4 === 0 &&
          Math.floor(value.packedArray.length / 4 / 2048) * 2048 === value.maxSplats,
        'packed capacity',
      );
    } else check(value.numSplats === 0, 'missing packed array');
    check(value.extra && typeof value.extra === 'object', 'extras');
    for (const key of Object.keys(value.extra))
      check(
        key === 'radMeta' || ARRAY_KEYS.includes(key as (typeof ARRAY_KEYS)[number]),
        'extra field',
      );
    for (const key of ARRAY_KEYS) if (value.extra[key] !== undefined) range(value.extra[key]!);
    for (const [key, width] of [
      ['sh1', 2],
      ['sh2', 4],
      ['sh3', 4],
    ] as const) {
      const entry = value.extra[key];
      if (entry)
        check(entry.length % width === 0 && entry.length >= value.numSplats * width, 'SH capacity');
    }
    if (value.lodSplats !== undefined) node(value.lodSplats, depth + 1);
  }
  node(header.root, 0);
  ranges.sort((a, b) => a.offset - b.offset);
  let end = 0;
  for (const entry of ranges) {
    check(entry.offset === end, 'overlapping or unclaimed payload');
    end += entry.length * 4;
  }
  check(end === payloadBytes, 'unclaimed payload');
}

export async function encodePreparedWorld(
  packed: PackedSplats,
  ids: Identity,
): Promise<ArrayBuffer> {
  identity(ids);
  const arrays: Uint32Array[] = [];
  let payloadBytes = 0;
  const visited = new Set<PackedSplats>();
  function saveArray(array: Uint32Array): Range {
    check(array instanceof Uint32Array, 'array type');
    const entry = { offset: payloadBytes, length: array.length };
    payloadBytes += array.byteLength;
    check(payloadBytes <= MAX_BYTES, 'payload too large');
    arrays.push(array);
    return entry;
  }
  function save(value: PackedSplats, depth: number): Saved {
    check(depth <= MAX_DEPTH && !visited.has(value), 'splat nesting');
    visited.add(value);
    const extra: Saved['extra'] = {};
    const packedArray = value.packedArray ? saveArray(value.packedArray) : null;
    for (const key of ARRAY_KEYS)
      if (value.extra[key] !== undefined) extra[key] = saveArray(value.extra[key] as Uint32Array);
    if (value.extra.radMeta !== undefined) extra.radMeta = value.extra.radMeta;
    return {
      packedArray,
      numSplats: value.numSplats,
      maxSplats: value.maxSplats,
      maxSh: value.maxSh,
      lod: value.lod,
      nonLod: value.nonLod,
      splatEncoding: value.splatEncoding,
      extra,
      lodSplats: value.lodSplats ? save(value.lodSplats, depth + 1) : undefined,
    };
  }
  const root = save(packed, 0);
  const payload = new Uint8Array(payloadBytes);
  let offset = 0;
  for (const array of arrays) {
    payload.set(new Uint8Array(array.buffer, array.byteOffset, array.byteLength), offset);
    offset += array.byteLength;
  }
  const header: Header = {
    ...ids,
    schema: SCHEMA,
    hash: await digest(payload),
    payloadBytes,
    root,
  };
  validate(header, ids, payloadBytes);
  const json = new TextEncoder().encode(JSON.stringify(header));
  check(json.length <= MAX_HEADER, 'header too large');
  const start = (8 + json.length + 3) & ~3;
  check(start + payloadBytes <= MAX_BYTES, 'container too large');
  const buffer = new ArrayBuffer(start + payloadBytes);
  const view = new DataView(buffer);
  view.setUint32(0, MAGIC, true);
  view.setUint32(4, json.length, true);
  new Uint8Array(buffer, 8, json.length).set(json);
  new Uint8Array(buffer, start).set(payload);
  return buffer;
}

export async function decodePreparedWorld(
  buffer: ArrayBuffer,
  ids: Identity,
): Promise<PackedSplats> {
  check(
    buffer instanceof ArrayBuffer && buffer.byteLength >= 8 && buffer.byteLength <= MAX_BYTES,
    'container size',
  );
  const view = new DataView(buffer);
  check(view.getUint32(0, true) === MAGIC, 'magic');
  const length = view.getUint32(4, true);
  check(length > 0 && length <= MAX_HEADER && 8 + length <= buffer.byteLength, 'header size');
  const start = (8 + length + 3) & ~3;
  check(start <= buffer.byteLength && (buffer.byteLength - start) % 4 === 0, 'payload alignment');
  const header = JSON.parse(
    new TextDecoder('utf-8', { fatal: true }).decode(new Uint8Array(buffer, 8, length)),
  ) as Header;
  check(header && typeof header === 'object', 'header');
  validate(header, ids, buffer.byteLength - start);
  check((await digest(new Uint8Array(buffer, start))) === header.hash, 'payload hash mismatch');
  const created: PackedSplats[] = [];
  const array = (entry: Range) => new Uint32Array(buffer, start + entry.offset, entry.length);
  function restore(value: Saved): PackedSplats {
    const lodSplats = value.lodSplats ? restore(value.lodSplats) : undefined;
    const extra: Record<string, unknown> = {};
    for (const key of ARRAY_KEYS) if (value.extra[key]) extra[key] = array(value.extra[key]!);
    if (value.extra.radMeta !== undefined) extra.radMeta = value.extra.radMeta;
    const result = new PackedSplats({
      numSplats: value.numSplats,
      maxSplats: value.maxSplats,
      splatEncoding: value.splatEncoding,
      lod: value.lod,
      nonLod: value.nonLod,
      packedArray: value.packedArray ? array(value.packedArray) : undefined,
      extra,
      lodSplats,
    });
    created.push(result);
    result.setMaxSh(value.maxSh);
    return result;
  }
  try {
    return restore(header.root);
  } catch (error) {
    for (const value of created) value.lodSplats = undefined;
    for (const value of created) value.dispose();
    throw error;
  }
}
