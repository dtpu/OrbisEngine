import assert from 'node:assert/strict';
import test from 'node:test';
import { publishInventory, publishMetadata, writePublishMetadata } from './lib/publish-policy.ts';

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
  const roots = publishInventory('/fixture', { archiveOnly: true }, '/evidence');
  assert.deepEqual(
    roots.map((root) => root.directory),
    ['/fixture/public', '/fixture/.context/run', '/evidence'],
  );
  assert.deepEqual(
    roots.map((root) => root.logicalPrefix),
    ['public/', 'runs/', 'evidence/'],
  );
  assert.ok(roots.every((root) => root.kind === 'archives' && root.prefix === 'archive'));
});

test('explicit full and assets-only publication retain their inventory behavior', () => {
  const full = publishInventory('/fixture', {}, '/evidence');
  assert.equal(full[0].kind, 'files');
  assert.equal(full[0].prefix, 'viewer');
  assert.equal(full[0].logicalPrefix, '/');
  assert.equal(full.length, 3);
  const assets = publishInventory('/fixture', { assetsOnly: true }, '/evidence');
  assert.deepEqual(assets, [full[0]]);
  assert.throws(
    () => publishInventory('/fixture', { archiveOnly: true, assetsOnly: true }),
    /cannot be combined/,
  );
});

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
