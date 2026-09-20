import test from 'node:test';
import type { AddressInfo } from 'node:net';
import type { GetObjectCommand } from '@aws-sdk/client-s3';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { Readable } from 'node:stream';
import { createHash } from 'node:crypto';
import { mkdtemp, rm, writeFile, mkdir, symlink } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { sharedAssets, parseRange, assetPath } from '../server/shared-assets.ts';
import { containsSecret, forbidden, filesUnder } from './lib/shared-storage.ts';
test('range parsing handles suffixes, clamping, invalid and multiple ranges', () => {
  assert.deepEqual(parseRange('bytes=2-5', 10), { start: 2, end: 5 });
  assert.deepEqual(parseRange('bytes=-3', 10), { start: 7, end: 9 });
  assert.deepEqual(parseRange('bytes=8-50', 10), { start: 8, end: 9 });
  for (const h of ['bytes=10-', 'bytes=5-2', 'bytes=-0', 'bytes=0-1,3-4', 'bytes=-', 'bogus'])
    assert.equal(parseRange(h, 10), false, h);
  assert.equal(parseRange('bytes=0-', 0), false);
});
test('asset paths cannot escape the public namespace', () => {
  for (const p of ['/../.env', '/%2e%2e/.env', '/a%5cb', '/%00x', '/%zz'])
    assert.equal(assetPath(p), null);
  assert.equal(assetPath('/worlds/a%20b/frame.ply?v=1'), '/worlds/a b/frame.ply');
});
test('credential exclusions inspect names and text without exposing values', async () => {
  assert.ok(forbidden('foo/.env.local'));
  assert.ok(forbidden('foo/bar.pem'));
  assert.ok(forbidden('credentials.json'));
  assert.ok(containsSecret('AWS_SECRET_ACCESS_KEY=' + 'x'.repeat(40)));
  assert.ok(containsSecret('AKIA' + 'X'.repeat(16)));
  assert.ok(containsSecret(JSON.stringify({ WLT_API_KEY: 'x'.repeat(40) })));
  assert.equal(containsSecret('See ~/.openai-env for your key; 0.5724 scale'), false);
  const dir = await mkdtemp(path.join(os.tmpdir(), 'wander-scan-'));
  try {
    await writeFile(path.join(dir, 'state.json'), '{}');
    await writeFile(path.join(dir, 'stage.log'), 'OPENAI_API_KEY=sk-' + 'x'.repeat(30));
    await writeFile(path.join(dir, '.env.local'), 'hidden');
    await writeFile(path.join(dir, 'unsafe.ts'), 'OPENAI_API_KEY=sk-' + 'x'.repeat(30));
    await symlink('/etc', path.join(dir, 'outside'));
    const scan = await filesUnder(dir);
    assert.deepEqual(
      scan.files.map((x) => x.path),
      ['state.json'],
    );
    assert.deepEqual(scan.excluded.sort(), ['.env.local', 'outside', 'stage.log', 'unsafe.ts']);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
test('private proxy pins snapshot, coalesces downloads, caches and serves video ranges', async () => {
  const data = Buffer.from('0123456789');
  const sha = createHash('sha256').update(data).digest('hex');
  const snapshot = 'viewer/snapshots/test.json';
  const key = `viewer/blobs/${sha}`;
  const calls: string[] = [];
  const s3 = {
    async send(command: GetObjectCommand) {
      calls.push(command.input.Key!);
      if (command.input.Key! === key) {
        await new Promise((r) => setTimeout(r, 20));
        return { Body: Readable.from(data) };
      }
      const value = command.input.Key!.endsWith('latest.json')
        ? { snapshot }
        : {
            schema: 'wander.shared/1',
            files: {
              '/clips/test.mp4': { key, size: data.length, sha256: sha, contentType: 'video/mp4' },
            },
          };
      return { ETag: '"catalog"', Body: { transformToString: async () => JSON.stringify(value) } };
    },
    destroy() {},
  };
  const dir = await mkdtemp(path.join(os.tmpdir(), 'wander-cache-'));
  const plugin = sharedAssets({}, { s3, cacheDir: dir });
  const server = createServer((req, res) =>
    plugin.middleware(req, res, () => {
      res.statusCode = 418;
      res.end('application');
    }),
  );
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
  const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  try {
    const responses = await Promise.all([
      fetch(url + '/clips/test.mp4'),
      fetch(url + '/clips/test.mp4'),
    ]);
    for (const r of responses) {
      assert.equal(r.status, 200);
      assert.equal(r.headers.get('x-wander-asset-source'), 's3');
      assert.equal(await r.text(), '0123456789');
    }
    assert.equal(calls.filter((x) => x === key).length, 1);
    const partial = await fetch(url + '/clips/test.mp4', { headers: { Range: 'bytes=3-6' } });
    assert.equal(partial.status, 206);
    assert.equal(partial.headers.get('content-range'), 'bytes 3-6/10');
    assert.equal(await partial.text(), '3456');
    const invalid = await fetch(url + '/clips/test.mp4', { headers: { Range: 'bytes=99-' } });
    assert.equal(invalid.status, 416);
    const unchanged = await fetch(url + '/clips/test.mp4', {
      headers: { 'If-None-Match': `"${sha}"` },
    });
    assert.equal(unchanged.status, 304);
    const head = await fetch(url + '/clips/test.mp4', { method: 'HEAD' });
    assert.equal(head.headers.get('content-length'), '10');
    assert.equal(await head.text(), '');
    const ifRange = await fetch(url + '/clips/test.mp4', {
      headers: { Range: 'bytes=1-2', 'If-Range': '"stale"' },
    });
    assert.equal(ifRange.status, 200);
    await ifRange.text();
    assert.equal((await fetch(url + '/clips/missing.mp4')).status, 404);
    assert.equal((await fetch(url + '/clips/test.mp4', { method: 'POST' })).status, 405);
    assert.equal((await fetch(url + '/src/xr/fourd-xr.ts')).status, 418);
    const status = await (await fetch(url + '/api/shared-assets')).json();
    assert.equal(status.snapshot, snapshot);
    assert.equal(status.assets, 1);
    assert.equal(calls.filter((x) => x.endsWith('latest.json')).length, 1);
    assert.equal(calls.filter((x) => x === key).length, 1);
  } finally {
    await new Promise<void>((resolve, reject) =>
      server.close((error) => (error ? reject(error) : resolve())),
    );
    await rm(dir, { recursive: true, force: true });
  }
});
test('failed S3 auth never falls back to local media or exposes the error details', async () => {
  const s3 = {
    send: async () => {
      throw Object.assign(new Error('SECRET_SENTINEL'), { name: 'AccessDenied' });
    },
    destroy() {},
  };
  const plugin = sharedAssets({}, { s3 });
  const server = createServer((req, res) =>
    plugin.middleware(req, res, () => {
      throw new Error('unexpected local fallback');
    }),
  );
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
  try {
    const r = await fetch(`http://127.0.0.1:${(server.address() as AddressInfo).port}/clips/a.mp4`);
    assert.equal(r.status, 503);
    assert.ok(!(await r.text()).includes('SECRET_SENTINEL'));
  } finally {
    await new Promise<void>((resolve, reject) =>
      server.close((error) => (error ? reject(error) : resolve())),
    );
  }
});
test('a stalled blob download is abandoned and fetched again', async () => {
  const data = Buffer.from('0123456789');
  const sha = createHash('sha256').update(data).digest('hex');
  const key = `viewer/blobs/${sha}`;
  let blobCalls = 0;
  const s3 = {
    async send(command: GetObjectCommand) {
      if (command.input.Key! === key) {
        blobCalls++;
        if (blobCalls > 1) return { Body: Readable.from(data) };
        // first attempt sends half the file and then goes quiet for good
        let sent = false;
        return {
          Body: new Readable({
            read() {
              if (!sent) this.push(data.subarray(0, 5));
              sent = true;
            },
          }),
        };
      }
      const value = command.input.Key!.endsWith('latest.json')
        ? { snapshot: 'viewer/snapshots/test.json' }
        : {
            schema: 'wander.shared/1',
            files: {
              '/worlds/a/frame_000.ply': {
                key,
                size: data.length,
                sha256: sha,
                contentType: 'application/octet-stream',
              },
            },
          };
      return { ETag: '"catalog"', Body: { transformToString: async () => JSON.stringify(value) } };
    },
    destroy() {},
  };
  const dir = await mkdtemp(path.join(os.tmpdir(), 'wander-cache-'));
  const plugin = sharedAssets({}, { s3, cacheDir: dir, stallMs: 50 });
  const server = createServer((req, res) => plugin.middleware(req, res, () => res.end()));
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
  const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  try {
    const response = await fetch(url + '/worlds/a/frame_000.ply', {
      signal: AbortSignal.timeout(5000),
    });
    assert.equal(response.status, 200);
    assert.equal(await response.text(), '0123456789');
    assert.equal(blobCalls, 2);
  } finally {
    await new Promise<void>((resolve, reject) =>
      server.close((error) => (error ? reject(error) : resolve())),
    );
    await rm(dir, { recursive: true, force: true });
  }
});
