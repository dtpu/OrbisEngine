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
import {
  sharedAssets,
  parseRange,
  assetPath,
  assetQuery,
  appPath,
  cacheControl,
  snapshotId,
  versionShim,
} from '../server/shared-assets.ts';
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
const IMMUTABLE = 'private, max-age=31536000, immutable';
const SESSION = 'private, max-age=600';
test('only a request bound to its content is cacheable without revalidation', () => {
  const sha = 'a'.repeat(64);
  const snapshot = 'viewer/snapshots/pinned-1.json';
  assert.equal(snapshotId(snapshot), 'pinned-1');
  const control = (query: string) => cacheControl(assetQuery(query), sha, snapshot);
  assert.equal(control('/x.ply?snap=pinned-1'), IMMUTABLE);
  assert.equal(control(`/x.ply?v=${sha.slice(0, 12)}`), IMMUTABLE);
  assert.equal(control(`/x.ply?v=${sha}&extra=1`), IMMUTABLE);
  // A bare path, a stale snapshot or a hash that is not this blob's must stay revalidatable.
  assert.equal(control('/x.ply'), SESSION);
  assert.equal(control('/x.ply?snap=pinned-0'), SESSION);
  assert.equal(control('/x.ply?v=' + 'b'.repeat(64)), SESSION);
  assert.equal(control('/x.ply?v=aaa'), SESSION, 'too short to identify a blob');
  assert.equal(control('/x.ply?v=AAAAAAAA'), SESSION, 'hashes are lowercase hex');
  assert.equal(control('/x.ply?snap='), SESSION);
  for (const p of ['/', '/demo.html', '/fourd.html', '/src/xr/fourd-xr.ts', '/@vite/client'])
    assert.ok(appPath(p), p);
  for (const p of ['/clips/a.mp4', '/worlds/a/frame_000.ply', '/marble-a.spz'])
    assert.equal(appPath(p), false, p);
});
test('a new snapshot changes every asset URL, so no stale bytes can be reused', async () => {
  const asset = '/worlds/a/frame_000.ply';
  const dirs: string[] = [];
  const servers: ReturnType<typeof createServer>[] = [];
  async function serve(snapshot: string, body: Buffer) {
    const sha = createHash('sha256').update(body).digest('hex');
    const s3 = {
      async send(command: GetObjectCommand) {
        if (command.input.Key! === `viewer/blobs/${sha}`) return { Body: Readable.from(body) };
        const value = command.input.Key!.endsWith('latest.json')
          ? { snapshot }
          : {
              schema: 'wander.shared/1',
              files: {
                [asset]: {
                  key: `viewer/blobs/${sha}`,
                  size: body.length,
                  sha256: sha,
                  contentType: 'application/octet-stream',
                },
              },
            };
        return {
          ETag: '"catalog"',
          Body: { transformToString: async () => JSON.stringify(value) },
        };
      },
      destroy() {},
    };
    const dir = await mkdtemp(path.join(os.tmpdir(), 'wander-cache-'));
    dirs.push(dir);
    const plugin = sharedAssets({}, { s3, cacheDir: dir });
    const server = createServer((req, res) =>
      plugin.middleware(req, res, () => {
        res.statusCode = 418;
        res.end('application');
      }),
    );
    await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
    servers.push(server);
    return { plugin, sha, url: `http://127.0.0.1:${(server.address() as AddressInfo).port}` };
  }
  try {
    const one = await serve('viewer/snapshots/one.json', Buffer.from('first content'));
    const two = await serve('viewer/snapshots/two.json', Buffer.from('second content!'));
    // Each run hands its page the id it pinned, so two runs never share an asset URL.
    const injected = async (plugin: { transformIndexHtml?: unknown }) => {
      const hook = plugin.transformIndexHtml as {
        handler(html: string): Promise<{ tags: { children: string }[] }>;
      };
      return (await hook.handler('<html><head></head><body></body></html>')).tags[0].children;
    };
    assert.match(await injected(one.plugin), /var snap = "one";/);
    assert.match(await injected(two.plugin), /var snap = "two";/);

    const fresh = await fetch(`${one.url}${asset}?snap=one`);
    assert.equal(fresh.headers.get('cache-control'), IMMUTABLE);
    assert.equal(await fresh.text(), 'first content');
    // The same versioned URL against the newer snapshot serves the NEW bytes, and because the
    // version no longer matches it is revalidatable rather than immutable.
    const carried = await fetch(`${two.url}${asset}?snap=one`);
    assert.equal(carried.headers.get('cache-control'), SESSION);
    assert.equal(carried.headers.get('etag'), `"${two.sha}"`);
    assert.equal(await carried.text(), 'second content!');
    const stale = await fetch(`${two.url}${asset}?snap=one`, {
      headers: { 'If-None-Match': `"${one.sha}"` },
    });
    assert.equal(stale.status, 200, 'an old validator must never be answered with 304');
    await stale.text();
    const plain = await fetch(`${two.url}${asset}`);
    assert.equal(plain.headers.get('cache-control'), SESSION);
    await plain.text();
    // Documents keep Vite's own no-cache handling even when a version query is present.
    const page = await fetch(`${two.url}/demo.html?snap=one`);
    assert.equal(page.status, 418);
    assert.equal(page.headers.get('cache-control'), null);
    await page.text();
  } finally {
    for (const server of servers)
      await new Promise<void>((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve())),
      );
    for (const dir of dirs) await rm(dir, { recursive: true, force: true });
  }
});
test('the injected shell script versions asset fetches and leaves app code alone', async () => {
  const sent: string[] = [];
  const window = { fetch: async (input: unknown) => void sent.push(String(input)) };
  const location = { search: '', href: 'http://host/demo.html', origin: 'http://host' };
  new Function('window', 'location', 'URLSearchParams', 'URL', versionShim('pinned-1'))(
    window,
    location,
    URLSearchParams,
    URL,
  );
  const rewritten = async (url: string) => {
    await window.fetch(url);
    return sent.at(-1)!;
  };
  assert.equal(
    await rewritten('/worlds/a/frame_000.ply'),
    'http://host/worlds/a/frame_000.ply?snap=pinned-1',
  );
  assert.equal(await rewritten('/clips/a.mp4?t=1'), 'http://host/clips/a.mp4?t=1&snap=pinned-1');
  for (const untouched of [
    '/src/xr/fourd-xr.ts',
    '/demo.html',
    'https://elsewhere.example/x.ply',
    '/worlds/a/frame_000.ply?snap=other',
  ])
    assert.equal(await rewritten(untouched), untouched);
  // ?assetver=0 leaves fetch exactly as it found it.
  const bare = { fetch: async (input: unknown) => void sent.push('bare:' + String(input)) };
  new Function('window', 'location', 'URLSearchParams', 'URL', versionShim('pinned-1'))(
    bare,
    { ...location, search: '?assetver=0' },
    URLSearchParams,
    URL,
  );
  await bare.fetch('/worlds/a/frame_000.ply');
  assert.equal(sent.at(-1), 'bare:/worlds/a/frame_000.ply');
});
