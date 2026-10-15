import { Readable } from 'node:stream';
import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Connect, Plugin } from 'vite';

/** The local viewer gets only read-only run previews; credentials stay on this server. */
export function pipelinePreview(environment: Record<string, string | undefined>): Plugin & {
  middleware: (req: IncomingMessage, res: ServerResponse, next: Connect.NextFunction) => void;
} {
  const middleware = (req: IncomingMessage, res: ServerResponse, next: Connect.NextFunction) => {
    const pathname = (req.url || '').split('?')[0];
    if (!/^\/api\/pipeline\/runs\/[^/]+\/viewer\//.test(pathname)) return next();
    res.setHeader('Cache-Control', 'no-store');
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.writeHead(405).end();
      return;
    }
    const invalidSegment = pathname.split('/').some((segment) => {
      try {
        const decoded = decodeURIComponent(segment);
        return decoded === '.' || decoded === '..' || /[/\\]/.test(decoded);
      } catch {
        return true;
      }
    });
    if (
      !/^\/api\/pipeline\/runs\/[A-Za-z0-9_.-]+\/viewer\/(scene\.json|assets\/.+)$/.test(
        pathname,
      ) ||
      invalidSegment
    ) {
      res.writeHead(404).end();
      return;
    }
    if (!environment.WANDER_PIPELINE_API || !environment.WANDER_API_TOKEN) {
      res
        .writeHead(503, { 'Content-Type': 'text/plain' })
        .end(
          'Set WANDER_PIPELINE_API and WANDER_API_TOKEN on the local viewer server to preview runs.',
        );
      return;
    }
    const controller = new AbortController();
    res.once('close', () => controller.abort());
    void (async () => {
      const headers = new Headers({ authorization: `Bearer ${environment.WANDER_API_TOKEN}` });
      for (const key of ['range', 'if-range']) {
        const value = req.headers[key];
        if (typeof value === 'string') headers.set(key, value);
      }
      const upstream = await fetch(
        `${environment.WANDER_PIPELINE_API!.replace(/\/$/, '')}${pathname}`,
        {
          method: req.method,
          headers,
          signal: controller.signal,
          redirect: 'manual',
        },
      );
      for (const key of [
        'content-type',
        'content-length',
        'content-range',
        'accept-ranges',
        'etag',
        'last-modified',
        'x-content-type-options',
      ]) {
        const value = upstream.headers.get(key);
        if (value) res.setHeader(key, value);
      }
      res.statusCode = upstream.status;
      if (req.method === 'HEAD' || !upstream.body) res.end();
      else {
        const stream = Readable.fromWeb(
          upstream.body as unknown as Parameters<typeof Readable.fromWeb>[0],
        );
        stream.on('error', () => res.destroy());
        stream.pipe(res);
      }
    })().catch(() => {
      if (!res.headersSent) res.writeHead(502).end('Pipeline preview unavailable.');
      else res.destroy();
    });
  };
  return {
    name: 'wander-pipeline-preview',
    configureServer(server) {
      server.middlewares.use(middleware);
    },
    middleware,
  };
}
