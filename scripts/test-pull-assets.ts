import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, readdir, rm, symlink } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { Readable } from 'node:stream';
import os from 'node:os';
import path from 'node:path';
import { GetObjectCommand } from '@aws-sdk/client-s3';
import { pullAssets } from './pull-assets.ts';
import { config, type AssetEntry, type ObjectReader } from './lib/shared-storage.ts';

function entry(data: string, namespace = 'viewer'): AssetEntry {
  const sha256 = createHash('sha256').update(data).digest('hex');
  return {
    key: `${namespace}/blobs/${sha256}`,
    size: Buffer.byteLength(data),
    sha256,
    contentType: 'video/mp4',
  };
}
async function fixture(run: (root: string) => Promise<void>) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'wander-pull-'));
  try {
    await run(root);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}
function storage(
  files: Record<string, AssetEntry>,
  bodies: Record<string, string>,
  archive = false,
) {
  const snapshot = `${archive ? 'archive' : 'viewer'}/snapshots/one.json`;
  const objects: Record<string, unknown> = {
    [config.catalogKey]: {
      snapshot: archive ? 'viewer/snapshots/view.json' : snapshot,
      archive: archive ? snapshot : undefined,
    },
    [snapshot]: { schema: 'wander.shared/1', files },
  };
  const calls: string[] = [];
  const s3: ObjectReader = {
    async send(command: GetObjectCommand) {
      const key = command.input.Key!;
      calls.push(key);
      if (key in bodies) return { Body: Readable.from(Buffer.from(bodies[key])) };
      assert.ok(key in objects, `Unexpected S3 read: ${key}`);
      return { Body: { transformToString: async () => JSON.stringify(objects[key]) } };
    },
    destroy() {},
  };
  return { s3, snapshot, calls, objects, bodies };
}

test('pins snapshot across resumes and verifies existing bytes without fetching the blob again', async () =>
  fixture(async (root) => {
    const asset = entry('recorded clip');
    const store = storage({ '/clips/example.mp4': asset }, { [asset.key]: 'recorded clip' });
    const options = { out: '.context/inputs', paths: ['/clips/example.mp4'] };
    const first = await pullAssets(options, store.s3, root);
    assert.equal(first.files[0].status, 'downloaded');
    assert.equal(
      await readFile(path.join(root, '.context/inputs/clips/example.mp4'), 'utf8'),
      'recorded clip',
    );
    store.objects[config.catalogKey] = { snapshot: 'viewer/snapshots/newer.json' };
    const second = await pullAssets(options, store.s3, root);
    assert.equal(second.snapshot, store.snapshot);
    assert.equal(second.files[0].status, 'verified');
    assert.equal(store.calls.filter((key) => key === config.catalogKey).length, 1);
    assert.equal(store.calls.filter((key) => key === asset.key).length, 1);
    await assert.rejects(
      pullAssets(
        { ...options, snapshot: 'viewer/snapshots/newer.json', overwrite: true },
        store.s3,
        root,
      ),
      /pinned/,
    );
    const receipt = JSON.parse(
      await readFile(path.join(root, '.context/inputs/.wander-pull.json'), 'utf8'),
    );
    assert.equal(receipt.files['/clips/example.mp4'].sha256, asset.sha256);
  }));

test('does not overwrite different local data unless explicitly requested', async () =>
  fixture(async (root) => {
    const asset = entry('new clip');
    const store = storage({ '/clips/example.mp4': asset }, { [asset.key]: 'new clip' });
    const target = path.join(root, '.context/inputs/clips/example.mp4');
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, 'local work');
    const options = { out: '.context/inputs', paths: ['/clips/example.mp4'] };
    await assert.rejects(pullAssets(options, store.s3, root), /Local file differs/);
    assert.equal(await readFile(target, 'utf8'), 'local work');
    assert.ok(!store.calls.includes(asset.key));
    await pullAssets({ ...options, overwrite: true }, store.s3, root);
    assert.equal(await readFile(target, 'utf8'), 'new clip');
  }));

test('bad size/hash never installs partial bytes or damages an overwritten file', async () =>
  fixture(async (root) => {
    const asset = entry('good');
    const store = storage({ '/clips/example.mp4': asset }, { [asset.key]: 'evil' });
    const target = path.join(root, '.context/inputs/clips/example.mp4');
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, 'old');
    const options = { out: '.context/inputs', paths: ['/clips/example.mp4'], overwrite: true };
    await assert.rejects(pullAssets(options, store.s3, root), /checksum\/size/);
    assert.equal(await readFile(target, 'utf8'), 'old');
    store.bodies[asset.key] = 'wrong size';
    await assert.rejects(pullAssets(options, store.s3, root), /checksum\/size/);
    assert.deepEqual(await readdir(path.dirname(target)), ['example.mp4']);
    const receipt = JSON.parse(
      await readFile(path.join(root, '.context/inputs/.wander-pull.json'), 'utf8'),
    );
    assert.deepEqual(receipt.files, {});
  }));

test('rejects unsafe output and logical paths before any remote read', async () =>
  fixture(async (root) => {
    const store = storage({}, {});
    for (const out of ['public', '.context', '../outside', '.context/../../outside']) {
      await assert.rejects(pullAssets({ out, list: true }, store.s3, root), /--out/);
    }
    for (const name of [
      '/../outside',
      '/clips/../../outside',
      '/clips/%2e%2e/out',
      '/clips/a\\b',
      '/clips//a',
      '/.wander-pull.json',
      '/.env',
    ]) {
      await assert.rejects(pullAssets({ out: '.context/inputs', paths: [name] }, store.s3, root));
    }
    assert.deepEqual(store.calls, []);
  }));

test('rejects symlinked output directories, ancestors, receipt, and destination even with overwrite', async () =>
  fixture(async (root) => {
    const asset = entry('data');
    const store = storage({ '/clips/example.mp4': asset }, { [asset.key]: 'data' });
    const outside = path.join(root, 'outside');
    await mkdir(outside);
    await mkdir(path.join(root, '.context'));
    await symlink(outside, path.join(root, '.context/link'));
    await assert.rejects(
      pullAssets({ out: '.context/link', list: true }, store.s3, root),
      /symlink/,
    );
    const out = path.join(root, '.context/inputs');
    await mkdir(out);
    await symlink(outside, path.join(out, 'clips'));
    await assert.rejects(
      pullAssets({ out: '.context/inputs', paths: ['/clips/example.mp4'] }, store.s3, root),
      /symlink/,
    );
    await rm(path.join(out, 'clips'));
    await mkdir(path.join(out, 'clips'));
    await writeFile(path.join(outside, 'keep'), 'keep');
    await symlink(path.join(outside, 'keep'), path.join(out, 'clips/example.mp4'));
    await assert.rejects(
      pullAssets(
        { out: '.context/inputs', paths: ['/clips/example.mp4'], overwrite: true },
        store.s3,
        root,
      ),
      /symlink/,
    );
    assert.equal(await readFile(path.join(outside, 'keep'), 'utf8'), 'keep');
    await rm(path.join(out, '.wander-pull.json'));
    await symlink(path.join(outside, 'keep'), path.join(out, '.wander-pull.json'));
    await assert.rejects(
      pullAssets({ out: '.context/inputs', list: true }, store.s3, root),
      /symlink/,
    );
  }));

test('lists input videos and requires explicit cinematic selection', async () =>
  fixture(async (root) => {
    const asset = entry('clip');
    const names = [
      '/clips/example.mp4',
      '/clips/cinematic/generated.mp4',
      '/clips/audio/original.wav',
      '/worlds/example/data.json',
    ];
    const store = storage(Object.fromEntries(names.map((name) => [name, asset])), {
      [asset.key]: 'clip',
    });
    const base = { out: '.context/inputs', list: true };
    assert.deepEqual(
      (await pullAssets(base, store.s3, root)).files.map((f) => f.path),
      ['/clips/example.mp4'],
    );
    assert.equal(
      (await pullAssets({ ...base, includeCinematic: true }, store.s3, root)).files.length,
      2,
    );
    assert.equal(
      (await pullAssets({ ...base, prefixes: ['/clips/'] }, store.s3, root)).files.length,
      2,
    );
    assert.equal(
      (await pullAssets({ ...base, prefixes: ['/clips/cinematic/'] }, store.s3, root)).files[0]
        .path,
      names[1],
    );
    assert.equal(
      (await pullAssets({ ...base, paths: [names[1]], list: false }, store.s3, root)).files[0]
        .status,
      'downloaded',
    );
  }));

test('archive mode selects archive pointer, preserves run hierarchy, and supports historical pins', async () =>
  fixture(async (root) => {
    const asset = entry('run record', 'archive');
    const store = storage(
      { 'runs/example/state.json': asset },
      { [asset.key]: 'run record' },
      true,
    );
    const result = await pullAssets(
      { out: '.context/archive', archive: true, paths: ['runs/example/state.json'] },
      store.s3,
      root,
    );
    assert.equal(result.snapshot, store.snapshot);
    assert.equal(
      await readFile(path.join(root, '.context/archive/runs/example/state.json'), 'utf8'),
      'run record',
    );
    store.calls.length = 0;
    await pullAssets(
      { out: '.context/history', archive: true, list: true, snapshot: store.snapshot },
      store.s3,
      root,
    );
    assert.deepEqual(store.calls, [store.snapshot]);
    await assert.rejects(
      pullAssets({ out: '.context/archive', list: true }, store.s3, root),
      /bucket\/mode/,
    );
  }));

test('rejects malformed catalogs, unknown paths, and changes to an already pinned manifest', async () =>
  fixture(async (root) => {
    const asset = entry('clip');
    const store = storage({ '/clips/example.mp4': asset }, { [asset.key]: 'clip' });
    await pullAssets({ out: '.context/inputs', list: true }, store.s3, root);
    await assert.rejects(
      pullAssets({ out: '.context/inputs', paths: ['/clips/missing.mp4'] }, store.s3, root),
      /not in/,
    );
    store.objects[store.snapshot] = {
      schema: 'wander.shared/1',
      files: { '/clips/other.mp4': asset },
    };
    await assert.rejects(
      pullAssets({ out: '.context/inputs', list: true }, store.s3, root),
      /manifest changed/,
    );
    store.objects[store.snapshot] = { schema: 'wander.shared/1', files: { '/../escape': asset } };
    await assert.rejects(pullAssets({ out: '.context/bad', list: true }, store.s3, root), /Unsafe/);
    store.objects[store.snapshot] = {
      schema: 'wander.shared/1',
      files: { '/clips/example.mp4': { ...asset, key: 'elsewhere' } },
    };
    await assert.rejects(
      pullAssets({ out: '.context/bad-key', list: true }, store.s3, root),
      /blob key/,
    );
  }));
