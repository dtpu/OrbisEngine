import { request } from 'node:http';
import { createReadStream } from 'node:fs';
import { realpath, stat } from 'node:fs/promises';
import path from 'node:path';
import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Connect, Plugin } from 'vite';

const MIME: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.svg': 'image/svg+xml',
  '.jpg': 'image/jpeg',
  '.png': 'image/png',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.mp4': 'video/mp4',
};

/** Join the local operator UI and viewer without exposing the repository or a bearer token. */
export function localApp(
  adminUrl: string | undefined,
  siteDirectory: string,
): Plugin & {
  middleware: (req: IncomingMessage, res: ServerResponse, next: Connect.NextFunction) => void;
} {
  const admin = adminUrl ? new URL(adminUrl) : null;
  if (admin && (admin.protocol !== 'http:' || admin.hostname !== '127.0.0.1'))
    throw new Error('WANDER_ADMIN_URL must be a local http://127.0.0.1 address');

  const middleware = (req: IncomingMessage, res: ServerResponse, next: Connect.NextFunction) => {
    if (!admin) return next();
    const raw = (req.url || '/').split('?')[0];
    if (raw === '/about') {
      res.writeHead(308, { Location: '/about/' }).end();
      return;
    }
    if (raw.startsWith('/about/')) {
      if (req.method !== 'GET' && req.method !== 'HEAD') {
        res.writeHead(405).end();
        return;
      }
      void (async () => {
        const relative = decodeURIComponent(raw.slice('/about/'.length)) || 'index.html';
        if (relative.split('/').some((part) => part.startsWith('.')) || /[\\\0]/.test(relative))
          throw new Error('Invalid path');
        const root = await realpath(siteDirectory);
        const file = await realpath(path.resolve(root, relative));
        if (!file.startsWith(root + path.sep)) throw new Error('Invalid path');
        const info = await stat(file);
        if (!info.isFile()) throw new Error('Not a file');
        res.setHeader('Content-Type', MIME[path.extname(file)] || 'application/octet-stream');
        res.setHeader('Content-Length', info.size);
        res.setHeader('X-Content-Type-Options', 'nosniff');
        if (req.method === 'HEAD') res.end();
        else {
          const stream = createReadStream(file);
          stream.on('error', () => res.destroy());
          res.once('close', () => stream.destroy());
          stream.pipe(res);
        }
      })().catch(() => {
        if (!res.headersSent) res.writeHead(404).end('Page or media not found.');
        else res.destroy();
      });
      return;
    }
    if (raw !== '/' && raw !== '/favicon.ico' && !/^\/(runs|_next|api\/pipeline)(\/|$)/.test(raw))
      return next();

    if (!['GET', 'HEAD', 'OPTIONS'].includes(req.method || 'GET')) {
      const origin = req.headers.origin;
      let trusted = req.headers['sec-fetch-site'] !== 'cross-site';
      try {
        if (origin) {
          const target = new URL(`http://${req.headers.host}`);
          trusted &&=
            ['127.0.0.1', 'localhost', '[::1]'].includes(target.hostname) &&
            origin === target.origin;
        }
      } catch {
        trusted = false;
      }
      if (!trusted) {
        res
          .writeHead(403, { 'Content-Type': 'text/plain' })
          .end('Cross-origin mutations are not allowed.');
        return;
      }
    }

    // Streaming preserves upload progress, SSE, video ranges and cancellation. Do not buffer clips.
    const upstream = request(
      {
        hostname: admin.hostname,
        port: admin.port,
        method: req.method,
        path: req.url,
        headers: { ...req.headers, host: admin.host },
      },
      (response) => {
        res.writeHead(response.statusCode || 502, response.headers);
        response.on('error', () => res.destroy());
        response.pipe(res);
      },
    );
    upstream.on('error', () => {
      if (!res.headersSent)
        res.writeHead(502, { 'Content-Type': 'text/plain' }).end('The dashboard is unavailable.');
      else res.destroy();
    });
    req.once('aborted', () => upstream.destroy());
    res.once('close', () => upstream.destroy());
    req.pipe(upstream);
  };
  return {
    name: 'orbis-local-app',
    middleware,
    configureServer(server) {
      server.middlewares.use(middleware);
    },
  };
}
