import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, mkdir, rm, symlink, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import path from 'node:path';
import {
  publishInventory,
  scanPublishInventory,
  publishMetadata,
  writePublishMetadata,
} from './lib/publish-policy.ts';

const previous = {
  schema: 'wander.shared/1',
  snapshot: 'viewer/snapshots/accepted.json',
  archive: 'archive/snapshots/old.json',
  createdAt: 'previous-viewer-time',
  extra: 'preserved',
};
const input = {
  archiveOnly: true,
  previous,
  etag: 'prior-etag',
  archiveKey: 'archive/snapshots/new.json',
  snapshot: 'viewer/snapshots/new.json',
  catalogKey: 'viewer/latest.json',
  schema: 'wander.shared/1',
  createdAt: 'new-time',
  files: { '/unreviewed.ply': { key: 'must-not-be-published' } },
  archives: { 'public/unreviewed.ply': { key: 'archive/blobs/hash' } },
  exclusions: { public: ['excluded-file'] },
};

test('archive-only inventories public outputs, runs and evidence exclusively as author archives', () => {
  const roots = publishInventory('/fixture', { archiveOnly: true }, '/evidence', {});
  assert.deepEqual(
    roots.map((root) => root.directory),
    ['/fixture/public', '/fixture/.context/runs', '/fixture/.context/run', '/evidence'],
  );
  assert.deepEqual(
    roots.map((root) => root.logicalPrefix),
    ['public/', 'runs/', 'runs/', 'evidence/'],
  );
  assert.ok(roots.every((root) => root.kind === 'archives' && root.prefix === 'archive'));
});

test('explicit full and assets-only publication retain their inventory behavior', () => {
  const full = publishInventory('/fixture', {}, '/evidence', {});
  assert.equal(full[0].kind, 'files');
  assert.equal(full[0].prefix, 'viewer');
  assert.equal(full[0].logicalPrefix, '/');
  assert.equal(full.length, 4);
  const assets = publishInventory('/fixture', { assetsOnly: true }, '/evidence', {});
  assert.deepEqual(assets, [full[0]]);
  assert.throws(
    () => publishInventory('/fixture', { archiveOnly: true, assetsOnly: true }),
    /cannot be combined/,
  );
});

test('configured roots resolve relative to the repo, expand home, and deduplicate defaults', () => {
  const roots = publishInventory('/fixture', {}, undefined, {
    WANDER_RUNS_ROOT: 'custom/runs',
    WANDER_RUNS_DIR: '~/old-runs',
  });
  assert.deepEqual(
    roots.slice(1).map((root) => root.directory),
    [
      '/fixture/.context/runs',
      '/fixture/.context/run',
      '/fixture/custom/runs',
      path.join(homedir(), 'old-runs'),
    ],
  );
  assert.equal(
    publishInventory('/fixture', {}, undefined, {
      WANDER_RUNS_ROOT: '.context/runs',
      WANDER_RUNS_DIR: '/fixture/.context/run',
    }).length,
    3,
  );
});

async function fixture(t: { after: (fn: () => Promise<void>) => void }) {
  const root = await mkdtemp(path.join(tmpdir(), 'publish-inventory-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const write = async (relative: string, contents = 'fixture output') => {
    const file = path.join(root, relative);
    await mkdir(path.dirname(file), { recursive: true });
    await writeFile(file, contents);
  };
  return { root, write };
}

for (const archiveOnly of [false, true]) {
  test(`${archiveOnly ? 'automatic archive' : 'manual publish'} scans new, legacy, and custom history`, async (t) => {
    const { root, write } = await fixture(t);
    await write('public/scene.json');
    await write('.context/runs/new-run/result.json');
    await write('.context/run/old-run/result.json');
    await write('configured/api-run/result.json');
    await write('legacy-configured/clip-run/result.json');
    await write('.context/runs/new-run/.env', 'private');
    await write('.context/run/old-run/.env', 'private');
    const { tasks, exclusions } = await scanPublishInventory(
      publishInventory(root, { archiveOnly }, undefined, {
        WANDER_RUNS_ROOT: 'configured',
        WANDER_RUNS_DIR: path.join(root, 'legacy-configured'),
      }),
    );
    assert.deepEqual(
      tasks.map((task) => task.path).sort(),
      [
        archiveOnly ? 'public/scene.json' : '/scene.json',
        'runs/new-run/result.json',
        'runs/old-run/result.json',
        'runs/api-run/result.json',
        'runs/clip-run/result.json',
      ].sort(),
    );
    assert.ok(
      tasks
        .filter((task) => task.path.startsWith('runs/'))
        .every((task) => task.kind === 'archives' && task.prefix === 'archive'),
    );
    assert.equal(tasks[0].kind, archiveOnly ? 'archives' : 'files');
    assert.deepEqual(exclusions.runs, ['new-run/.env']);
    assert.deepEqual(exclusions['runs-legacy'], ['old-run/.env']);
  });
}

test('missing roots are harmless and aliases of the same root are scanned once', async (t) => {
  const { root, write } = await fixture(t);
  assert.deepEqual(
    (await scanPublishInventory(publishInventory(root, {}, undefined, {}))).tasks,
    [],
  );
  await write('.context/runs/example/result.json');
  await symlink(path.join(root, '.context/runs'), path.join(root, '.context/run'), 'dir');
  const { tasks } = await scanPublishInventory(publishInventory(root, {}, undefined, {}));
  assert.deepEqual(
    tasks.map((task) => task.path),
    ['runs/example/result.json'],
  );
});

for (const secondFile of ['result.json', 'other-stage.json']) {
  test(`duplicate run names fail before upload even with ${secondFile}`, async (t) => {
    const { root, write } = await fixture(t);
    await write('.context/runs/example/result.json', 'new run');
    await write(`.context/run/example/${secondFile}`, 'legacy run');
    for (const archiveOnly of [false, true]) {
      await assert.rejects(
        scanPublishInventory(publishInventory(root, { archiveOnly }, undefined, {})),
        /Archive run collision at runs\/example/,
      );
    }
  });
}

test('archive-only writes an immutable archive then changes only the existing archive reference', () => {
  const writes = publishMetadata(input);
  assert.deepEqual(
    writes.map((write) => write.key),
    [input.archiveKey, input.catalogKey],
  );
  assert.deepEqual(writes[0], {
    key: input.archiveKey,
    value: {
      schema: input.schema,
      createdAt: input.createdAt,
      files: input.archives,
      exclusions: input.exclusions,
    },
    condition: { IfNoneMatch: '*' },
  });
  assert.deepEqual(writes[1].value, { ...previous, archive: input.archiveKey });
  assert.deepEqual(writes[1].condition, { IfMatch: input.etag });
  assert.equal(previous.archive, 'archive/snapshots/old.json');
});

test('archive-only with no existing pointer creates no viewer metadata', () => {
  const writes = publishMetadata({ ...input, previous: {}, etag: null });
  assert.deepEqual(
    writes.map((write) => write.key),
    [input.archiveKey],
  );
});

test('full publication retains archive, viewer snapshot, then conditional pointer ordering', () => {
  const writes = publishMetadata({ ...input, archiveOnly: false });
  assert.deepEqual(
    writes.map((write) => write.key),
    [input.archiveKey, input.snapshot, input.catalogKey],
  );
  assert.deepEqual(writes[1], {
    key: input.snapshot,
    value: { schema: input.schema, createdAt: input.createdAt, files: input.files },
    condition: { IfNoneMatch: '*' },
  });
  assert.deepEqual(writes[2].condition, { IfMatch: input.etag });
  assert.deepEqual(publishMetadata({ ...input, archiveOnly: false, etag: null })[2].condition, {
    IfNoneMatch: '*',
  });
});

test('archive failure prevents any pointer attempt', async () => {
  const calls: string[] = [];
  const failure = new Error('archive upload rejected');
  await assert.rejects(
    writePublishMetadata(publishMetadata(input), async (write) => {
      calls.push(write.key);
      throw failure;
    }),
    (error: unknown) => error === failure,
  );
  assert.deepEqual(calls, [input.archiveKey]);
});

test('archive-only CAS conflict propagates without overwriting a concurrent viewer snapshot', async () => {
  const concurrent = { ...previous, snapshot: 'viewer/snapshots/concurrent.json' };
  const stored = new Map<string, unknown>([[input.catalogKey, concurrent]]);
  const conflict = Object.assign(new Error('PreconditionFailed'), { statusCode: 412 });
  const calls: string[] = [];
  await assert.rejects(
    writePublishMetadata(publishMetadata(input), async (write) => {
      calls.push(write.key);
      if (write.key === input.catalogKey) {
        assert.equal(write.condition.IfMatch, 'prior-etag');
        throw conflict;
      }
      stored.set(write.key, write.value);
    }),
    (error: unknown) => error === conflict,
  );
  assert.deepEqual(calls, [input.archiveKey, input.catalogKey]);
  assert.equal(stored.get(input.catalogKey), concurrent);
  assert.ok(stored.has(input.archiveKey));
  assert.equal(stored.has(input.snapshot), false);
});
