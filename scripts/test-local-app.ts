import { test, expect } from 'bun:test';
import { createServer, request } from 'node:http';
import { once } from 'node:events';
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, writeFile, symlink, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { localApp } from '../server/local-app.ts';

test('app routes streaming uploads and byte ranges to the dashboard, while leaving viewer routes alone', async () => {
  let forwarded = 0;
  const upstream = createServer(async (req, res) => {
    forwarded++;
    if (req.url === '/api/pipeline/chunks') {
      res.setHeader('Connection', 'X-Upstream-Hop');
      res.setHeader('X-Upstream-Hop', 'private');
      res.write('first');
      await new Promise((resolve) => setTimeout(resolve, 20));
      res.end('second');
      return;
    }
    let data = '';
    for await (const chunk of req) data += chunk;
    if (req.headers.range) res.writeHead(206, { 'Content-Range': 'bytes 0-3/10' }).end('clip');
    else res.end(JSON.stringify({ url: req.url, method: req.method, data }));
  }).listen(0, '127.0.0.1');
  await once(upstream, 'listening');
  const port = (upstream.address() as { port: number }).port;
  const plugin = localApp(`http://127.0.0.1:${port}`, '/missing');
  const server = createServer((req, res) =>
    plugin.middleware(req, res, () => res.end('viewer')),
  ).listen(0, '127.0.0.1');
  await once(server, 'listening');
  const base = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  try {
    const rejected = await fetch(`${base}/api/pipeline/runs/upload`, {
      method: 'POST',
      body: 'cross-origin upload',
      headers: {
        Origin: 'https://untrusted.example',
        'Content-Type': 'multipart/form-data; boundary=x',
      },
    });
    expect(rejected.status).toBe(403);
    expect(forwarded).toBe(0);
    const trusted = await fetch(`${base}/api/pipeline/runs/upload`, {
      method: 'POST',
      body: 'local browser upload',
      headers: { Origin: base },
    });
    expect(trusted.status).toBe(200);
    const upload = await fetch(`${base}/api/pipeline/runs/upload?keep=1`, {
      method: 'POST',
      body: 'source bytes',
    });
    expect(await upload.json()).toEqual({
      url: '/api/pipeline/runs/upload?keep=1',
      method: 'POST',
      data: 'source bytes',
    });
    const range = await fetch(`${base}/api/pipeline/runs/run/artifacts/source`, {
      headers: { Range: 'bytes=0-3' },
    });
    expect(range.status).toBe(206);
    expect(range.headers.get('content-range')).toBe('bytes 0-3/10');
    expect(await range.text()).toBe('clip');
    // Bun's fetch tolerates duplicate chunked encoding; Node's HTTP parser detects it.
    // Use a separate process so this also checks the actual wire response from the gateway.
    const client = spawn(
      'node',
      [
        '--input-type=module',
        '-e',
        `
      const response = await fetch(process.argv[1]);
      if (await response.text() !== 'firstsecond') throw new Error('Corrupt stream');
      if (response.headers.has('x-upstream-hop')) throw new Error('Leaked hop header');
    `,
        `${base}/api/pipeline/chunks`,
      ],
      { stdio: ['ignore', 'ignore', 'pipe'] },
    );
    let clientError = '';
    client.stderr.on('data', (chunk) => {
      clientError += chunk;
    });
    const [code] = await once(client, 'exit');
    expect(clientError).toBe('');
    expect(code).toBe(0);
    for (const url of ['/demo.html', '/fourd.html', '/api/prepared-world/v1/a/b', '/src/fourd.ts'])
      expect(await (await fetch(base + url)).text()).toBe('viewer');
  } finally {
    server.closeAllConnections();
    server.close();
    upstream.closeAllConnections();
    upstream.close();
  }
});

test('about serves only built site files and refuses hidden files, traversal and symlinks outside the site', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'orbis-site-'));
  await mkdir(path.join(root, 'dist'));
  await writeFile(path.join(root, 'dist/index.html'), '<h1>Orbis</h1>');
  await writeFile(path.join(root, 'private'), 'secret');
  await symlink(path.join(root, 'private'), path.join(root, 'dist/leak'));
  const plugin = localApp('http://127.0.0.1:5401', path.join(root, 'dist'));
  const server = createServer((req, res) =>
    plugin.middleware(req, res, () => res.writeHead(404).end()),
  ).listen(0, '127.0.0.1');
  await once(server, 'listening');
  const port = (server.address() as { port: number }).port;
  try {
    const response = await fetch(`http://127.0.0.1:${port}/about/`);
    expect(await response.text()).toBe('<h1>Orbis</h1>');
    for (const target of [
      '/about/leak',
      '/about/%2e%2e/private',
      '/about/.env',
      '/about/%2f..%2fprivate',
    ]) {
      const status = await new Promise<number>((resolve, reject) => {
        request({ host: '127.0.0.1', port, path: target }, (res) => {
          res.resume();
          resolve(res.statusCode!);
        })
          .on('error', reject)
          .end();
      });
      expect(status).toBe(404);
    }
  } finally {
    server.closeAllConnections();
    server.close();
    await rm(root, { recursive: true, force: true });
  }
});
