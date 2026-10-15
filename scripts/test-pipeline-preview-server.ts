import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer, request } from 'node:http';
import type { AddressInfo } from 'node:net';
import { pipelinePreview } from '../server/pipeline-preview.ts';

test('preview proxy keeps credentials server-side, preserves range/HEAD and refuses writes/traversal', async () => {
  const calls: { method?: string; authorization?: string; range?: string; url?: string }[] = [];
  const upstream = createServer((req, res) => {
    calls.push({
      method: req.method,
      authorization: req.headers.authorization,
      range: req.headers.range,
      url: req.url,
    });
    res.writeHead(req.headers.range ? 206 : 200, {
      'Content-Type': 'application/octet-stream',
      'Content-Length': '3',
      ...(req.headers.range ? { 'Content-Range': 'bytes 1-3/10' } : {}),
    });
    res.end(req.method === 'HEAD' ? undefined : 'abc');
  });
  await new Promise<void>((resolve) => upstream.listen(0, '127.0.0.1', resolve));
  const middleware = pipelinePreview({
    WANDER_PIPELINE_API: `http://127.0.0.1:${(upstream.address() as AddressInfo).port}`,
    WANDER_API_TOKEN: 'server-only-token',
  }).middleware;
  const server = createServer((req, res) => middleware(req, res, () => res.writeHead(418).end()));
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const get = (path: string, method = 'GET', headers = {}) =>
    new Promise<{ status: number; body: string; headers: import('node:http').IncomingHttpHeaders }>(
      (resolve, reject) => {
        const req = request(
          {
            hostname: '127.0.0.1',
            port: (server.address() as AddressInfo).port,
            path,
            method,
            headers,
          },
          (res) => {
            let body = '';
            res.on('data', (chunk) => (body += chunk));
            res.on('end', () => resolve({ status: res.statusCode!, body, headers: res.headers }));
          },
        );
        req.on('error', reject);
        req.end();
      },
    );
  const prefix = '/api/pipeline/runs/run-1/viewer/';
  try {
    const ranged = await get(prefix + 'assets/clips/source.mp4', 'GET', {
      range: 'bytes=1-3',
      authorization: 'attacker',
    });
    assert.equal(ranged.status, 206);
    assert.equal(ranged.body, 'abc');
    assert.equal(ranged.headers['content-range'], 'bytes 1-3/10');
    assert.equal(ranged.headers.authorization, undefined);
    const head = await get(prefix + 'scene.json', 'HEAD');
    assert.equal(head.status, 200);
    assert.equal(head.body, '');
    const dottedRun = '/api/pipeline/runs/run..1/viewer/scene.json';
    assert.equal((await get(dottedRun)).status, 200);
    const dottedFile = prefix + 'assets/worlds/run-1-4d/person/frame..001.ply';
    assert.equal((await get(dottedFile)).status, 200);
    assert.deepEqual(calls, [
      {
        method: 'GET',
        authorization: 'Bearer server-only-token',
        range: 'bytes=1-3',
        url: prefix + 'assets/clips/source.mp4',
      },
      {
        method: 'HEAD',
        authorization: 'Bearer server-only-token',
        range: undefined,
        url: prefix + 'scene.json',
      },
      {
        method: 'GET',
        authorization: 'Bearer server-only-token',
        range: undefined,
        url: dottedRun,
      },
      {
        method: 'GET',
        authorization: 'Bearer server-only-token',
        range: undefined,
        url: dottedFile,
      },
    ]);
    assert.equal((await get(prefix + 'scene.json', 'POST')).status, 405);
    for (const path of [
      'assets/../run.json',
      'assets/%2e%2e/run.json',
      'assets/%2E./run.json',
      'assets/.%2e/run.json',
      'assets/%2fetc/passwd',
      'assets/%5cetc/passwd',
      'assets/%ZZ/run.json',
    ])
      assert.equal((await get(prefix + path)).status, 404, path);
    assert.equal((await get('/api/pipeline/runs/run-1/cancel', 'POST')).status, 418);
    assert.equal(calls.length, 4);
  } finally {
    for (const service of [server, upstream]) {
      service.closeAllConnections();
      await new Promise<void>((resolve) => service.close(() => resolve()));
    }
  }
});

test('unconfigured viewer preview fails explicitly', async () => {
  const middleware = pipelinePreview({}).middleware;
  let status = 0;
  let message = '';
  const response = {
    setHeader() {},
    writeHead(code: number) {
      status = code;
      return this;
    },
    end(body: string) {
      message = body;
    },
  };
  middleware(
    { url: '/api/pipeline/runs/run-1/viewer/scene.json', method: 'GET' } as never,
    response as never,
    () => assert.fail('must intercept preview'),
  );
  assert.equal(status, 503);
  assert.match(message, /WANDER_API_TOKEN/);
});
