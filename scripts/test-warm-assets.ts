import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { demoSceneIds, presetPaths, selectPaths, warmAssets } from './warm-assets.ts';
import { ROOT } from './lib/shared-storage.ts';
import type { AssetCatalog, AssetEntry } from './lib/shared-storage.ts';

const entry = (sha: string, size = 10): AssetEntry => ({
  key: `viewer/blobs/${sha}`,
  sha256: sha,
  size,
  contentType: 'application/octet-stream',
});
const sha = (n: number) => String(n).padStart(64, '0');

test('the demo scene list and viewer presets resolve without a fixed asset list', async () => {
  const demoHtml = await readFile(path.join(ROOT, 'demo.html'), 'utf8');
  const fourdHtml = await readFile(path.join(ROOT, 'fourd.html'), 'utf8');
  const ids = demoSceneIds(demoHtml);
  assert.ok(ids.length >= 5, `expected the picker's scenes, got ${ids.length}`);
  assert.ok(ids.includes('lobby') && ids.includes('elevator'));
  const presets = presetPaths(fourdHtml, ids);
  assert.deepEqual([...presets.keys()].sort(), [...ids].sort());
  for (const [id, paths] of presets) {
    assert.ok(
      paths.some((p) => p.startsWith('/worlds/')),
      `${id} names no world directory`,
    );
    assert.ok(
      paths.some((p) => p.endsWith('.spz') || p.endsWith('.ply')),
      `${id} names no world asset`,
    );
  }
  assert.throws(() => presetPaths(fourdHtml, ['not-a-scene']), /No viewer preset/);
});

test('a named world directory is selected whole and unrelated paths are left alone', () => {
  const catalog = {
    schema: 'wander.shared/1',
    files: {
      '/worlds/a-4d/person/sequence.json': entry(sha(1)),
      '/worlds/a-4d/person/frame_000.ply': entry(sha(2)),
      '/worlds/a-4d/collision.json': entry(sha(3)),
      '/worlds/b-4d/person/frame_000.ply': entry(sha(4)),
      '/clips/a.mp4': entry(sha(5)),
      '/clips/b.mp4': entry(sha(6)),
      '/marble-a.spz': entry(sha(7)),
    },
  } as unknown as AssetCatalog;
  assert.deepEqual(
    selectPaths(catalog, ['/worlds/a-4d/person/sequence.json', '/clips/a.mp4', '/marble-a.spz']),
    [
      '/clips/a.mp4',
      '/marble-a.spz',
      '/worlds/a-4d/collision.json',
      '/worlds/a-4d/person/frame_000.ply',
      '/worlds/a-4d/person/sequence.json',
    ],
  );
  // A preset can name a file this snapshot does not carry; that is not a warming failure.
  assert.deepEqual(selectPaths(catalog, ['/marble-gone.spz']), []);
});

test('warming downloads each blob once and reports the pinned snapshot', async () => {
  const shared = entry(sha(9));
  const catalog = {
    schema: 'wander.shared/1',
    snapshot: 'viewer/snapshots/pinned.json',
    files: {
      '/worlds/a-4d/person/frame_000.ply': shared,
      '/worlds/a-4d/person/frame_001.ply': shared, // identical frames share one blob
      '/worlds/a-4d/person/frame_002.ply': entry(sha(10), 30),
      '/clips/unrelated.mp4': entry(sha(11)),
    },
  } as unknown as AssetCatalog & { snapshot: string };
  const downloaded: string[] = [];
  const cache = {
    catalog: async () => catalog,
    cached: async (f: AssetEntry) => {
      downloaded.push(f.sha256);
      return f.sha256;
    },
  };
  const result = await warmAssets(cache, { paths: ['/worlds/a-4d/person/frame_000.ply'] });
  assert.deepEqual(result, {
    snapshot: 'viewer/snapshots/pinned.json',
    paths: 3,
    blobs: 2,
    bytes: 40,
  });
  assert.deepEqual(downloaded.sort(), [sha(10), sha(9)].sort());
  assert.equal(downloaded.filter((x) => x === sha(9)).length, 1);

  const everything = await warmAssets(cache, { all: true });
  assert.equal(everything.paths, 4);
  assert.equal(everything.blobs, 3);
  await assert.rejects(warmAssets(cache, { paths: ['/nothing'] }), /Nothing selected/);
});
