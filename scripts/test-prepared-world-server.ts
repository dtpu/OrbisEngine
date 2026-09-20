import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer, request } from 'node:http';
import type { AddressInfo } from 'node:net';
import { mkdtemp, mkdir, rm, symlink, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { preparedWorlds, SPARK_BUILD_ID } from '../server/prepared-worlds.ts';

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

test('GET and HEAD serve exact identity bytes with private immutable cache headers', async () => {
  await fixture(async ({ directory, get }) => {
    const bytes = Buffer.from([0, 255, 1, 128, 17, 19]);
    await writeFile(path.join(directory, SPARK_BUILD_ID, `${source}.bin`), bytes);
    const response = await get(assetPath() + '?ignored=1');
    assert.equal(response.status, 200);
    assert.deepEqual(response.body, bytes);
    assert.equal(response.headers['content-type'], 'application/octet-stream');
    assert.equal(response.headers['content-length'], String(bytes.length));
    assert.equal(response.headers['cache-control'], 'private, max-age=31536000, immutable');
    assert.equal(response.headers.etag, `"${SPARK_BUILD_ID}-${source}"`);
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
