import { createReadStream, readFileSync } from 'node:fs';
import { realpath, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Connect, Plugin } from 'vite';

const root = path.resolve(import.meta.dirname, '..');
export const PREPARED_WORLD_DIRECTORY = path.join(root, '.context/prepared-worlds');
// Cache files are tied to the installed decoder, LoD builder and binary schema.
export const SPARK_BUILD_ID = createHash('sha256')
  .update('wander.prepared-world/1\n')
  .update(readFileSync(path.join(root, 'node_modules/@sparkjsdev/spark/dist/spark.module.js')))
  .digest('hex');

export function preparedWorlds(directory = PREPARED_WORLD_DIRECTORY): Plugin & {
  middleware: (req: IncomingMessage, res: ServerResponse, next: Connect.NextFunction) => void;
} {
  const middleware = (req: IncomingMessage, res: ServerResponse, next: Connect.NextFunction) => {
    const pathname = (req.url || '').split('?')[0];
    if (!pathname.startsWith('/api/prepared-world/')) return next();
    res.setHeader('Cache-Control', 'no-store');
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.writeHead(405).end();
      return;
    }
    const match = /^\/api\/prepared-world\/v1\/([a-f0-9]{64})\/([a-f0-9]{64})\.bin$/.exec(pathname);
    if (!match || match[1] !== SPARK_BUILD_ID) {
      res.writeHead(404).end();
      return;
    }
    void (async () => {
      const [build, source] = match.slice(1);
      const file = path.join(directory, build, `${source}.bin`);
      const [resolvedRoot, resolvedFile] = await Promise.all([realpath(directory), realpath(file)]);
      if (!resolvedFile.startsWith(resolvedRoot + path.sep)) throw new Error('Outside cache');
      const info = await stat(resolvedFile);
      if (!info.isFile()) throw new Error('Not a cache file');
      res.setHeader('Content-Type', 'application/octet-stream');
      res.setHeader('Cache-Control', 'private, max-age=31536000, immutable');
      const etag = `"${build}-${source}"`;
      res.setHeader('ETag', etag);
      if (req.headers['if-none-match'] === etag) {
        res.writeHead(304).end();
        return;
      }
      res.setHeader('Content-Length', info.size);
      if (req.method === 'HEAD') res.end();
      else {
        const stream = createReadStream(resolvedFile);
        stream.on('error', () => res.destroy());
        res.on('close', () => stream.destroy());
        stream.pipe(res);
      }
    })().catch(() => {
      if (!res.headersSent) res.writeHead(404).end();
      else res.destroy();
    });
  };
  return {
    name: 'wander-prepared-worlds',
    middleware,
    configureServer(server) {
      server.middlewares.use(middleware);
    },
    configurePreviewServer(server) {
      server.middlewares.use(middleware);
    },
  };
}
