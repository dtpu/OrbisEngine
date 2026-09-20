import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PackedSplats } from '@sparkjsdev/spark';
import { decodePreparedWorld, encodePreparedWorld } from '../src/prepared-world.ts';

const ids = { sourceHash: 'a'.repeat(64), buildId: 'spark-2.2.0-artifact-sha256' };
function fixture() {
  const words = (length: number) =>
    Uint32Array.from({ length }, (_, index) => (index * 2654435761) >>> 0);
  const lod = new PackedSplats({
    packedArray: words(4096 * 4),
    numSplats: 2051,
    splatEncoding: {
      rgbMin: -0.2,
      rgbMax: 1.3,
      lnScaleMin: -12,
      lnScaleMax: 9,
      sh1Max: 2,
      sh2Max: 3,
      sh3Max: 4,
      lodOpacity: true,
    },
    extra: {
      sh1: words(4096 * 2),
      sh2: words(4096 * 4),
      sh3: words(4096 * 4),
      sh1Codes: words(8),
      sh2Codes: words(16),
      sh3Codes: words(16),
      lodTree: words(4096 * 4),
      radMeta: { version: 1, type: 'test', count: 2051, chunks: [] },
    },
  });
  lod.setMaxSh(2);
  const root = new PackedSplats({ lod: true, nonLod: false, lodSplats: lod });
  root.extra.sh1Texture = { ignoredGpuResource: true };
  return root;
}
function rewriteHeader(buffer: ArrayBuffer, change: (header: any) => void) {
  const view = new DataView(buffer);
  const length = view.getUint32(4, true);
  const oldStart = (8 + length + 3) & ~3;
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, length)));
  change(header);
  const json = new TextEncoder().encode(JSON.stringify(header));
  const start = (8 + json.length + 3) & ~3;
  const result = new ArrayBuffer(start + buffer.byteLength - oldStart);
  new DataView(result).setUint32(0, view.getUint32(0, true), true);
  new DataView(result).setUint32(4, json.length, true);
  new Uint8Array(result, 8, json.length).set(json);
  new Uint8Array(result, start).set(new Uint8Array(buffer, oldStart));
  return result;
}

test('roundtrips empty root and complete padded LoD data with zero-copy views', async () => {
  const original = fixture();
  const buffer = await encodePreparedWorld(original, ids);
  const decoded = await decodePreparedWorld(buffer, ids);
  assert.equal(decoded.numSplats, 0);
  assert.equal(decoded.packedArray, null);
  assert.equal(decoded.lod, true);
  assert.equal(decoded.nonLod, false);
  assert.equal(decoded.extra.sh1Texture, undefined);
  const actual = decoded.lodSplats!;
  const expected = original.lodSplats!;
  assert.equal(actual.numSplats, 2051);
  assert.equal(actual.maxSplats, 4096);
  assert.equal(actual.maxSh, 2);
  assert.equal(actual.packedArray!.buffer, buffer);
  assert.deepEqual(actual.packedArray, expected.packedArray);
  assert.deepEqual(actual.extra, expected.extra);
  assert.deepEqual(actual.splatEncoding, expected.splatEncoding);
  assert.deepEqual(new Uint8Array(await encodePreparedWorld(decoded, ids)), new Uint8Array(buffer));
  original.dispose();
  decoded.dispose();
});

test('preserves a populated original alongside nested LoD and encoding flags', async () => {
  const original = fixture();
  const root = new PackedSplats({
    packedArray: new Uint32Array(8192),
    numSplats: 7,
    lod: 'quality',
    nonLod: true,
    lodSplats: original.lodSplats,
    splatEncoding: { lodOpacity: false, rgbMax: 1 },
  });
  original.lodSplats = undefined;
  const decoded = await decodePreparedWorld(await encodePreparedWorld(root, ids), ids);
  assert.equal(decoded.numSplats, 7);
  assert.equal(decoded.lod, 'quality');
  assert.equal(decoded.nonLod, true);
  assert.deepEqual(decoded.splatEncoding, root.splatEncoding);
  assert.deepEqual(decoded.packedArray, root.packedArray);
  decoded.dispose();
  root.dispose();
  original.dispose();
});

test('rejects mismatched identity, corrupt bytes, truncation and unsupported schema', async () => {
  const original = fixture();
  const buffer = await encodePreparedWorld(original, ids);
  await assert.rejects(
    decodePreparedWorld(buffer, { ...ids, sourceHash: 'b'.repeat(64) }),
    /identity mismatch/,
  );
  await assert.rejects(
    decodePreparedWorld(buffer, { ...ids, buildId: 'different' }),
    /identity mismatch/,
  );
  const corrupt = buffer.slice(0);
  new Uint8Array(corrupt)[corrupt.byteLength - 1] ^= 1;
  await assert.rejects(decodePreparedWorld(corrupt, ids), /hash mismatch/);
  await assert.rejects(decodePreparedWorld(buffer.slice(0, -4), ids), /payload size/);
  await assert.rejects(decodePreparedWorld(buffer.slice(0, 9), ids), /header size/);
  await assert.rejects(
    decodePreparedWorld(
      rewriteHeader(buffer, (h) => {
        h.schema = 2;
      }),
      ids,
    ),
    /schema/,
  );
  original.dispose();
});

test('rejects overlapping arrays, invalid capacities, nonfinite metadata and excessive nesting', async () => {
  const original = fixture();
  const buffer = await encodePreparedWorld(original, ids);
  for (const change of [
    (h: any) => {
      h.root.url = 'https://unexpected.invalid/world.spz';
    },
    (h: any) => {
      h.root.lodSplats.extra.sh1.offset = 0;
    },
    (h: any) => {
      h.root.lodSplats.packedArray.offset = 3;
    },
    (h: any) => {
      h.root.lodSplats.numSplats = 4097;
    },
    (h: any) => {
      h.root.lodSplats.maxSplats = 2048;
    },
    (h: any) => {
      h.root.lodSplats.extra.sh1.length = 1;
    },
    (h: any) => {
      h.root.lodSplats.packedArray.length = -1;
    },
    (h: any) => {
      h.root.lodSplats.splatEncoding.rgbMin = null;
    },
    (h: any) => {
      let node = h.root;
      for (let i = 0; i < 10; i++) {
        node.lodSplats = { ...h.root, lodSplats: undefined };
        node = node.lodSplats;
      }
    },
  ])
    await assert.rejects(
      decodePreparedWorld(rewriteHeader(buffer, change), ids),
      /Invalid prepared world/,
    );
  original.lodSplats!.splatEncoding!.rgbMax = Infinity;
  await assert.rejects(encodePreparedWorld(original, ids), /nonfinite metadata/);
  original.dispose();
});

test('disposes restored children if a later Spark constructor fails', async () => {
  const original = fixture();
  const buffer = await encodePreparedWorld(original, ids);
  original.dispose();
  const initialize = PackedSplats.prototype.initialize;
  const dispose = PackedSplats.prototype.dispose;
  const children: PackedSplats[] = [];
  const disposed: PackedSplats[] = [];
  PackedSplats.prototype.initialize = function (options) {
    if (options.lodSplats) throw new Error('synthetic constructor failure');
    initialize.call(this, options);
    children.push(this);
  };
  PackedSplats.prototype.dispose = function () {
    disposed.push(this);
    dispose.call(this);
  };
  try {
    await assert.rejects(decodePreparedWorld(buffer, ids), /synthetic constructor failure/);
    assert.equal(children.length, 1);
    assert.deepEqual(disposed, children);
    assert.equal(children[0].packedArray, null);
    assert.deepEqual(children[0].extra, {});
  } finally {
    PackedSplats.prototype.initialize = initialize;
    PackedSplats.prototype.dispose = dispose;
  }
});
