import { readFileSync, type BigIntStats } from 'node:fs';
import { open, realpath } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Connect, Plugin } from 'vite';

const root = path.resolve(import.meta.dirname, '..');
export const PREPARED_WORLD_DIRECTORY = path.join(root, '.context/prepared-worlds');
// Include preparation policy and codec changes even when the schema number stays unchanged.
export function preparedWorldBuildId(directory = root): string {
  const hash = createHash('sha256').update('wander.prepared-world/1\n');
  for (const filename of [
    'node_modules/@sparkjsdev/spark/dist/spark.module.js',
    'scripts/prepare-world-client.ts',
    'src/prepared-world.ts',
  ]) {
    const contents = readFileSync(path.join(directory, filename));
    hash.update(`${filename}\0${contents.length}\0`).update(contents);
  }
  return hash.digest('hex');
}
export const SPARK_BUILD_ID = preparedWorldBuildId();

function fileIdentity(info: BigIntStats): string {
  return [info.dev, info.ino, info.size, info.mtimeNs, info.ctimeNs].join(':');
}

export function preparedWorlds(directory = PREPARED_WORLD_DIRECTORY): Plugin & {
  middleware: (req: IncomingMessage, res: ServerResponse, next: Connect.NextFunction) => void;
} {
  const validators = new Map<string, { identity: string; etag: string }>();
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
      const handle = await open(resolvedFile, 'r');
      let streaming = false;
      try {
        const info = await handle.stat({ bigint: true });
        if (!info.isFile()) throw new Error('Not a cache file');
        const identity = fileIdentity(info);
        let validator = validators.get(resolvedFile);
        if (validator?.identity !== identity) {
          const hash = createHash('sha256');
          for await (const chunk of handle.createReadStream({ start: 0, autoClose: false }))
            hash.update(chunk);
          if (fileIdentity(await handle.stat({ bigint: true })) !== identity)
            throw new Error('Cache changed while hashing');
          validator = { identity, etag: `"${hash.digest('hex')}"` };
          validators.set(resolvedFile, validator);
        }
        res.setHeader('Content-Type', 'application/octet-stream');
        // --force can replace this route. Revalidate without downloading unchanged bytes.
        res.setHeader('Cache-Control', 'private, no-cache');
        res.setHeader('ETag', validator.etag);
        if (
          req.headers['if-none-match']?.split(',').some((value) => {
            const tag = value.trim().replace(/^W\//, '');
            return tag === '*' || tag === validator.etag;
          })
        ) {
          res.writeHead(304).end();
          return;
        }
        res.setHeader('Content-Length', String(info.size));
        if (req.method === 'HEAD') res.end();
        else {
          // Keep the descriptor used for hashing: --force publishes by atomic rename.
          const stream = handle.createReadStream({ start: 0 });
          streaming = true;
          stream.on('error', () => res.destroy());
          res.on('close', () => stream.destroy());
          stream.pipe(res);
        }
      } finally {
        if (!streaming) await handle.close();
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
