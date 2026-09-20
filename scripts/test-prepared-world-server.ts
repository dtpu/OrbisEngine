import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer, request } from 'node:http';
import type { AddressInfo } from 'node:net';
import { mkdtemp, mkdir, rename, rm, stat, symlink, utimes, writeFile } from 'node:fs/promises';
import os from 'node:os';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { preparedWorldBuildId, preparedWorlds, SPARK_BUILD_ID } from '../server/prepared-worlds.ts';

const source = 'a'.repeat(64);
const assetPath = (hash = source, build = SPARK_BUILD_ID) =>
  `/api/prepared-world/v1/${build}/${hash}.bin`;

async function fixture(
  run: (ctx: {
    directory: string;
    parent: string;
    get: (
      pathname: string,
      method?: string,
      headers?: Record<string, string>,
    ) => Promise<{
      status: number;
      headers: import('node:http').IncomingHttpHeaders;
      body: Buffer;
    }>;
  }) => Promise<void>,
) {
  const parent = await mkdtemp(path.join(os.tmpdir(), 'wander-prepared-world-'));
  const directory = path.join(parent, 'cache');
  await mkdir(path.join(directory, SPARK_BUILD_ID), { recursive: true });
  const middleware = preparedWorlds(directory).middleware;
  const server = createServer((req, res) =>
    middleware(req, res, () => res.writeHead(418).end('next')),
  );
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const port = (server.address() as AddressInfo).port;
  // Raw HTTP keeps adversarial path bytes intact instead of URL normalization.
  const get = (pathname: string, method = 'GET', headers: Record<string, string> = {}) =>
    new Promise<{ status: number; headers: import('node:http').IncomingHttpHeaders; body: Buffer }>(
      (resolve, reject) => {
        const req = request(
          { hostname: '127.0.0.1', port, path: pathname, method, headers },
          (res) => {
            const chunks: Buffer[] = [];
            res.on('data', (chunk) => chunks.push(Buffer.from(chunk)));
            res.on('end', () =>
              resolve({
                status: res.statusCode!,
                headers: res.headers,
                body: Buffer.concat(chunks),
              }),
            );
            res.on('error', reject);
          },
        );
        req.on('error', reject);
        req.end();
      },
    );
  try {
    await run({ directory, parent, get });
  } finally {
    await new Promise<void>((resolve, reject) => {
      server.close((error) => (error ? reject(error) : resolve()));
      server.closeAllConnections();
    });
    await rm(parent, { recursive: true, force: true });
  }
}

test('GET and HEAD serve exact identity bytes with private revalidation headers', async () => {
  await fixture(async ({ directory, get }) => {
    const bytes = Buffer.from([0, 255, 1, 128, 17, 19]);
    await writeFile(path.join(directory, SPARK_BUILD_ID, `${source}.bin`), bytes);
    const response = await get(assetPath() + '?ignored=1');
    assert.equal(response.status, 200);
    assert.deepEqual(response.body, bytes);
    assert.equal(response.headers['content-type'], 'application/octet-stream');
    assert.equal(response.headers['content-length'], String(bytes.length));
    assert.equal(response.headers['cache-control'], 'private, no-cache');
    assert.equal(response.headers.etag, `"${createHash('sha256').update(bytes).digest('hex')}"`);
    const head = await get(assetPath(), 'HEAD');
    assert.equal(head.status, 200);
    assert.equal(head.body.length, 0);
    for (const name of ['etag', 'content-length', 'content-type', 'cache-control'])
      assert.equal(head.headers[name], response.headers[name]);
    const second = await get(assetPath());
    assert.deepEqual(second.body, bytes);
    const cached = await get(assetPath(), 'GET', { 'If-None-Match': response.headers.etag! });
    assert.equal(cached.status, 304);
    assert.equal(cached.body.length, 0);
    assert.equal(cached.headers.etag, response.headers.etag);
    assert.equal(cached.headers['cache-control'], response.headers['cache-control']);
    const cachedHead = await get(assetPath(), 'HEAD', {
      'If-None-Match': response.headers.etag!,
    });
    assert.equal(cachedHead.status, 304);
    assert.equal(cachedHead.body.length, 0);
    const changed = await get(assetPath(), 'GET', { 'If-None-Match': '"other"' });
    assert.equal(changed.status, 200);
    assert.deepEqual(changed.body, bytes);
  });
});

test('missing or stale identities return uncached 404 so preparation can later succeed', async () => {
  await fixture(async ({ directory, get }) => {
    const staleBuild = 'b'.repeat(64);
    await mkdir(path.join(directory, staleBuild));
    await writeFile(path.join(directory, staleBuild, `${source}.bin`), 'stale prepared bytes');
    for (const [pathname, method] of [
      [assetPath(), 'GET'],
      [assetPath(), 'HEAD'],
      [assetPath(source, 'b'.repeat(64)), 'GET'],
      ['/api/prepared-world/v1/bad/bad.bin', 'GET'],
    ]) {
      const response = await get(pathname, method);
      assert.equal(response.status, 404);
      assert.equal(response.headers['cache-control'], 'no-store');
      assert.equal(response.body.length, 0);
    }
    await writeFile(path.join(directory, SPARK_BUILD_ID, `${source}.bin`), 'newly prepared');
    const response = await get(assetPath());
    assert.equal(response.status, 200);
    assert.equal(response.body.toString(), 'newly prepared');
  });
});

test('path confinement rejects traversal, external symlinks and non-files', async () => {
  await fixture(async ({ directory, parent, get }) => {
    const outside = path.join(parent, 'outside.bin');
    await writeFile(outside, 'must remain private');
    await symlink(outside, path.join(directory, SPARK_BUILD_ID, `${source}.bin`));
    await mkdir(path.join(directory, SPARK_BUILD_ID, `${'b'.repeat(64)}.bin`));
    for (const pathname of [
      assetPath(),
      assetPath('b'.repeat(64)),
      `/api/prepared-world/v1/${SPARK_BUILD_ID}/../../outside.bin`,
      `/api/prepared-world/v1/${SPARK_BUILD_ID}/%2e%2e/outside.bin`,
      `/api/prepared-world/v1/${SPARK_BUILD_ID}/%2foutside.bin`,
      assetPath() + '/extra',
    ]) {
      const response = await get(pathname);
      assert.equal(response.status, 404, pathname);
      assert.equal(response.headers['cache-control'], 'no-store');
      assert.equal(response.body.length, 0);
    }
  });
});

test('unsupported methods return 405 while unrelated URLs reach the next middleware', async () => {
  await fixture(async ({ get }) => {
    for (const method of ['POST', 'PUT', 'DELETE']) {
      const response = await get(assetPath(), method);
      assert.equal(response.status, 405);
      assert.equal(response.headers['cache-control'], 'no-store');
      assert.equal(response.body.length, 0);
    }
    const response = await get('/some-viewer-path');
    assert.equal(response.status, 418);
    assert.equal(response.body.toString(), 'next');
  });
});

test('same-size force replacements and rewrites invalidate the content validator', async () => {
  await fixture(async ({ directory, get }) => {
    const file = path.join(directory, SPARK_BUILD_ID, `${source}.bin`);
    await writeFile(file, 'first bytes');
    const first = await get(assetPath());
    const original = await stat(file);
    // Match prepare:worlds --force publication, including an adversarial preserved mtime.
    await writeFile(file + '.part', 'other bytes');
    await utimes(file + '.part', original.atime, original.mtime);
    await rename(file + '.part', file);
    const replacement = await get(assetPath(), 'GET', { 'If-None-Match': first.headers.etag! });
    assert.equal(replacement.status, 200);
    assert.equal(replacement.body.toString(), 'other bytes');
    assert.notEqual(replacement.headers.etag, first.headers.etag);
    assert.equal(replacement.headers['cache-control'], 'private, no-cache');
    const cached = await get(assetPath(), 'GET', { 'If-None-Match': replacement.headers.etag! });
    assert.equal(cached.status, 304);
    // Also invalidate same-inode edits, even with the old mtime restored.
    await writeFile(file, 'third bytes');
    await utimes(file, original.atime, original.mtime);
    const rewritten = await get(assetPath(), 'GET', {
      'If-None-Match': replacement.headers.etag!,
    });
    assert.equal(rewritten.status, 200);
    assert.equal(rewritten.body.toString(), 'third bytes');
    assert.notEqual(rewritten.headers.etag, replacement.headers.etag);
  });
});

test('build identity includes installed Spark, preparation policy and codec contents', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'wander-prepared-build-'));
  const inputs = [
    'node_modules/@sparkjsdev/spark/dist/spark.module.js',
    'scripts/prepare-world-client.ts',
    'src/prepared-world.ts',
  ];
  try {
    for (const input of inputs) {
      await mkdir(path.dirname(path.join(directory, input)), { recursive: true });
      await writeFile(path.join(directory, input), input);
    }
    const initial = preparedWorldBuildId(directory);
    assert.match(initial, /^[a-f0-9]{64}$/);
    assert.equal(preparedWorldBuildId(directory), initial);
    for (const input of inputs) {
      await writeFile(path.join(directory, input), input + ' changed');
      assert.notEqual(preparedWorldBuildId(directory), initial, input);
      await writeFile(path.join(directory, input), input);
      assert.equal(preparedWorldBuildId(directory), initial);
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
