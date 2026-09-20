// CPU preparation on this Mac only. Outputs remain private under .context; nothing is published.
import { chromium } from 'playwright-core';
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, rename, rm, stat } from 'node:fs/promises';
import path from 'node:path';
import { PREPARED_WORLD_DIRECTORY, SPARK_BUILD_ID } from '../server/prepared-worlds';
import { decodePreparedWorld } from '../src/prepared-world';

let base = 'http://127.0.0.1:5399';
let force = false;
const worlds: string[] = [];
for (let i = 2; i < Bun.argv.length; i++) {
  const argument = Bun.argv[i];
  if (argument === '--url') base = Bun.argv[++i];
  else if (argument === '--force') force = true;
  else if (argument.startsWith('/') && /\.(spz|ply|splat|sog|rad)$/.test(argument))
    worlds.push(argument);
  else throw new Error(`Unknown argument: ${argument}`);
}
if (!worlds.length)
  throw new Error('Usage: bun run prepare:worlds --url http://127.0.0.1:5399 /world.spz ...');
const baseUrl = new URL(base);
if (!['127.0.0.1', 'localhost', '[::1]'].includes(baseUrl.hostname))
  throw new Error('Use the local viewer as the source server');

const built = await Bun.build({
  entrypoints: [path.resolve(import.meta.dirname, 'prepare-world-client.ts')],
  target: 'browser',
  format: 'esm',
});
if (!built.success) throw new Error(built.logs.map(String).join('\n'));
const script = await built.outputs[0].text();
const directory = path.join(PREPARED_WORLD_DIRECTORY, SPARK_BUILD_ID);
await mkdir(directory, { recursive: true });
let sourceBytes: Uint8Array<ArrayBuffer> | null = null;
let sourceHash = '';
let outputFile = '';
let saved = false;
const token = randomUUID();
const headers = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
};
const server = Bun.serve({
  hostname: '127.0.0.1',
  port: 0,
  maxRequestBodySize: 512 * 1024 * 1024,
  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname === `/${token}/`)
      return new Response('<!doctype html><script type="module" src="/prepare.js"></script>', {
        headers: { ...headers, 'Content-Type': 'text/html' },
      });
    if (url.pathname === '/prepare.js')
      return new Response(script, { headers: { ...headers, 'Content-Type': 'text/javascript' } });
    if (/^\/source\.(spz|ply|splat|sog|rad)$/.test(url.pathname) && sourceBytes)
      return new Response(sourceBytes, {
        headers: { ...headers, 'Content-Type': 'application/octet-stream' },
      });
    if (
      url.pathname === '/prepared' &&
      request.method === 'POST' &&
      request.headers.get('origin') === server.url.origin &&
      sourceHash &&
      outputFile
    ) {
      let temporary = '';
      try {
        const buffer = await request.arrayBuffer();
        const decoded = await decodePreparedWorld(buffer, { sourceHash, buildId: SPARK_BUILD_ID });
        decoded.dispose();
        temporary = `${outputFile}.${randomUUID()}.part`;
        await Bun.write(temporary, buffer);
        await rename(temporary, outputFile);
        saved = true;
        return new Response('Saved', { headers });
      } catch (error) {
        console.error('Prepared world verification failed:', error);
        return new Response('Preparation failed', { status: 500, headers });
      } finally {
        if (temporary) await rm(temporary, { force: true });
      }
    }
    return new Response('Not found', { status: 404, headers });
  },
});
const browser = await chromium.launch({ channel: 'chrome', headless: true });
try {
  for (const world of worlds) {
    const url = new URL(world, baseUrl);
    if (url.origin !== baseUrl.origin) throw new Error('World must belong to the local viewer');
    const head = await fetch(url, { method: 'HEAD' });
    if (!head.ok) throw new Error(`${world}: HTTP ${head.status}`);
    const expectedHash = /^"([a-f0-9]{64})"$/.exec(head.headers.get('etag') || '')?.[1];
    if (!expectedHash)
      throw new Error(`${world}: preparation requires the shared-assets server's SHA-256 ETag`);
    const existing = path.join(directory, `${expectedHash}.bin`);
    if (!force && (await stat(existing).catch(() => null))?.isFile()) {
      console.log(`${world}: already prepared`);
      continue;
    }
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${world}: HTTP ${response.status}`);
    sourceBytes = new Uint8Array(await response.arrayBuffer());
    sourceHash = createHash('sha256').update(sourceBytes).digest('hex');
    if (sourceHash !== expectedHash) throw new Error(`${world}: source hash mismatch`);
    outputFile = path.join(directory, `${sourceHash}.bin`);
    saved = false;
    const page = await browser.newPage();
    try {
      await page.goto(`${server.url.origin}/${token}/`);
      await page.waitForFunction(() => typeof window.prepareWorld === 'function');
      const result = await page.evaluate(
        ({ source, sourceHash, buildId }) => window.prepareWorld(source, sourceHash, buildId),
        { source: `/source${path.extname(world)}`, sourceHash, buildId: SPARK_BUILD_ID },
      );
      if (!saved) throw new Error(`${world}: no verified output was saved`);
      console.log(
        `${world}: ${result.splats.toLocaleString()} prepared splats, ${(result.bytes / 1048576).toFixed(1)} MiB, ${(result.milliseconds / 1000).toFixed(1)} seconds`,
      );
    } finally {
      await page.close();
      sourceBytes = null;
      sourceHash = outputFile = '';
    }
  }
} finally {
  await browser.close();
  server.stop(true);
}
