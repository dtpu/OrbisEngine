// Server-only S3 credentials. Browser requests retain their ordinary asset paths.
import { mkdir, stat, rename, unlink } from 'node:fs/promises';
import { createReadStream, createWriteStream } from 'node:fs';
import { pipeline } from 'node:stream/promises';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { GetObjectCommand } from '@aws-sdk/client-s3';
import { ROOT, config, client, getJSON, hashFile } from '../scripts/lib/shared-storage.ts';
import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Connect, Plugin, ViteDevServer } from 'vite';
import type {
  AssetEntry,
  AssetCatalog,
  CatalogPointer,
  ObjectReader,
  StorageEnvironment,
} from '../scripts/lib/shared-storage.ts';
const DOWNLOAD_ATTEMPTS = 2;
export interface ByteRange {
  start: number;
  end: number;
}
export interface SharedAssetDependencies {
  s3?: ObjectReader;
  cacheDir?: string;
  stallMs?: number;
}
export type AssetMiddleware = (
  req: IncomingMessage,
  res: ServerResponse,
  next: Connect.NextFunction,
) => Promise<void>;
export type SharedAssetsPlugin = Plugin & { middleware: AssetMiddleware };
type PinnedCatalog = AssetCatalog & { snapshot: string };
export function parseRange(header: string | undefined, size: number): ByteRange | false | null {
  if (!header) return null;
  const m = /^bytes=(\d*)-(\d*)$/.exec(header);
  if (!m || (!m[1] && !m[2]) || size === 0) return false;
  let start = m[1] ? Number(m[1]) : Math.max(0, size - Number(m[2]));
  let end = m[1] ? (m[2] ? Math.min(Number(m[2]), size - 1) : size - 1) : size - 1;
  if (
    !Number.isSafeInteger(start) ||
    !Number.isSafeInteger(end) ||
    start < 0 ||
    start >= size ||
    end < start
  )
    return false;
  return { start, end };
}
export function assetPath(url: string) {
  try {
    const raw = decodeURIComponent(url.split('?')[0]);
    if (
      !raw.startsWith('/') ||
      raw.includes('\\') ||
      raw.includes('\0') ||
      raw.split('/').some((x) => x === '.' || x === '..')
    )
      return null;
    return raw;
  } catch {
    return null;
  }
}
export function sharedAssets(
  env: StorageEnvironment = {},
  dependencies: SharedAssetDependencies = {},
): SharedAssetsPlugin {
  const local = env.WANDER_ASSETS_MODE === 'local';
  const s3 = dependencies.s3 || client(env);
  const cacheDir = dependencies.cacheDir || path.join(ROOT, '.context/shared-assets/blobs');
  const stallMs = dependencies.stallMs ?? 30000;
  let pinned: Promise<PinnedCatalog> | null | undefined;
  const downloads = new Map<string, Promise<string>>();
  async function catalog(): Promise<PinnedCatalog> {
    if (!pinned)
      pinned = (async () => {
        const pointer = (await getJSON<CatalogPointer>(s3, config.catalogKey)).value;
        if (!/^viewer\/snapshots\/[A-Za-z0-9.-]+\.json$/.test(pointer.snapshot))
          throw new Error('Invalid snapshot pointer');
        const value = (await getJSON<AssetCatalog>(s3, pointer.snapshot)).value;
        if (value.schema !== 'wander.shared/1') throw new Error('Unknown shared catalog schema');
        for (const f of Object.values(value.files)) {
          if (
            !/^viewer\/blobs\/[a-f0-9]{64}$/.test(f.key) ||
            f.key.split('/').at(-1) !== f.sha256 ||
            !Number.isSafeInteger(f.size) ||
            f.size < 0
          )
            throw new Error('Invalid asset entry');
        }
        return { ...value, snapshot: pointer.snapshot };
      })().catch((e) => {
        pinned = null;
        throw e;
      });
    return pinned;
  }
  async function cached(f: AssetEntry) {
    const dest = path.join(cacheDir, f.sha256);
    try {
      if ((await stat(dest)).size === f.size) return dest;
    } catch {}
    if (!downloads.has(f.sha256))
      downloads.set(
        f.sha256,
        (async () => {
          await mkdir(cacheDir, { recursive: true });
          // A stalled S3 stream never ends on its own, and every later request for the blob joins
          // the same promise: the viewer then sits a few frames short until the server restarts.
          for (let attempt = 1; ; attempt++) {
            const temp = `${dest}.${randomUUID()}.part`;
            try {
              const object = await s3.send(
                new GetObjectCommand({ Bucket: config.bucket, Key: f.key }),
              );
              const body = object.Body as NodeJS.ReadableStream & { destroy(e?: Error): void };
              const stalled = () => body.destroy(new Error('Asset download stalled'));
              let watchdog = setTimeout(stalled, stallMs);
              body.on('data', () => {
                clearTimeout(watchdog);
                watchdog = setTimeout(stalled, stallMs);
              });
              try {
                await pipeline(body, createWriteStream(temp, { flags: 'wx', mode: 0o600 }));
              } finally {
                clearTimeout(watchdog);
              }
              if ((await stat(temp)).size !== f.size || (await hashFile(temp)) !== f.sha256)
                throw new Error('Asset checksum mismatch');
              await rename(temp, dest);
              return dest;
            } catch (e) {
              if (attempt >= DOWNLOAD_ATTEMPTS) throw e;
            } finally {
              await unlink(temp).catch(() => {});
            }
          }
        })().finally(() => downloads.delete(f.sha256)),
      );
    return downloads.get(f.sha256)!;
  }
  async function middleware(req: IncomingMessage, res: ServerResponse, next: Connect.NextFunction) {
    if (local) return next();
    const pathname = assetPath(req.url || '/');
    if (!pathname) {
      res.statusCode = 400;
      res.end('Invalid asset path');
      return;
    }
    const status = pathname === '/api/shared-assets';
    // These are application code requests; all public assets are looked up in the catalog.
    if (
      !status &&
      (/^\/(?:src|node_modules|@[^/]+|api)(?:\/|$)/.test(pathname) ||
        pathname === '/' ||
        /^\/[^/]+\.html$/.test(pathname))
    )
      return next();
    if (!['GET', 'HEAD'].includes(req.method!)) {
      res.statusCode = 405;
      res.end();
      return;
    }
    try {
      const c = await catalog();
      if (status) {
        res.setHeader('Content-Type', 'application/json');
        res.setHeader('Cache-Control', 'no-store');
        res.end(
          JSON.stringify({
            mode: 's3',
            snapshot: c.snapshot,
            createdAt: c.createdAt,
            assets: Object.keys(c.files).length,
          }),
        );
        return;
      }
      const f = c.files[pathname];
      if (!f) {
        res.statusCode = 404;
        res.end(
          'Asset is not in the shared snapshot. Authors: publish it with bun run runs:publish.',
        );
        return;
      }
      const etag = `"${f.sha256}"`;
      res.setHeader('ETag', etag);
      res.setHeader('Cache-Control', 'private, max-age=0, must-revalidate');
      res.setHeader('X-Wander-Asset-Source', 's3');
      res.setHeader('X-Wander-Snapshot', c.snapshot);
      res.setHeader('Accept-Ranges', 'bytes');
      res.setHeader('Content-Type', f.contentType);
      if (
        req.headers['if-none-match']
          ?.split(/,\s*/)
          .some((x) => x === etag || x === `W/${etag}` || x === '*')
      ) {
        res.statusCode = 304;
        res.end();
        return;
      }
      const range = parseRange(
        !req.headers['if-range'] || req.headers['if-range'] === etag
          ? req.headers.range
          : undefined,
        f.size,
      );
      if (range === false) {
        res.statusCode = 416;
        res.setHeader('Content-Range', `bytes */${f.size}`);
        res.end();
        return;
      }
      if (range) {
        res.statusCode = 206;
        res.setHeader('Content-Range', `bytes ${range.start}-${range.end}/${f.size}`);
      }
      res.setHeader('Content-Length', range ? range.end - range.start + 1 : f.size);
      if (req.method === 'HEAD') {
        res.end();
        return;
      }
      const file = await cached(f);
      if (res.destroyed) return;
      await pipeline(createReadStream(file, range || {}), res);
    } catch (e) {
      if (res.destroyed) return;
      if (res.headersSent) {
        res.destroy();
        return;
      }
      res.removeHeader('Content-Length');
      res.statusCode = 503;
      res.setHeader('Content-Type', 'text/plain');
      res.end(
        'Shared assets unavailable. Set the teammate credentials in .env.local and restart Vite. See docs/shared-assets.md.',
      );
      console.error(
        `[shared-assets] ${(e as Error).name || 'Error'}; asset request failed (credentials are never logged).`,
      );
    }
  }
  return {
    name: 'private-shared-assets',
    configureServer(server: ViteDevServer) {
      server.middlewares.use(middleware);
      server.httpServer?.once('close', () => s3.destroy());
    },
    middleware,
  };
}
