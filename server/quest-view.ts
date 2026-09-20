import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Connect, Plugin } from 'vite';

const MAX_FRAME_BYTES = 256 * 1024;
const ID = /^[A-Za-z0-9-]{1,80}$/;
const CLIP = /^[A-Za-z0-9_-]{1,80}$/;

class RequestError extends Error {
  constructor(readonly status: number) {
    super(String(status));
  }
}

// Allocate once per upload; never retain chunks or an unbounded streamed body.
function readBody(req: IncomingMessage, limit: number, timeout: number): Promise<Buffer> {
  const length = req.headers['content-length'];
  if (length !== undefined && (!/^\d+$/.test(length) || Number(length) > limit)) {
    return Promise.reject(new RequestError(413));
  }
  return new Promise((resolve, reject) => {
    const buffer = Buffer.allocUnsafe(length === undefined ? limit : Number(length));
    let used = 0;
    const timer = setTimeout(() => finish(new RequestError(408)), timeout);
    const finish = (error?: RequestError) => {
      clearTimeout(timer);
      req.off('data', data);
      req.off('end', end);
      req.off('aborted', aborted);
      req.off('error', aborted);
      if (error) {
        req.resume();
        reject(error);
      } else resolve(buffer.subarray(0, used));
    };
    const data = (chunk: Buffer) => {
      if (used + chunk.length > buffer.length) return finish(new RequestError(413));
      chunk.copy(buffer, used);
      used += chunk.length;
    };
    const end = () => finish();
    const aborted = () => finish(new RequestError(400));
    req.on('data', data);
    req.once('end', end);
    req.once('aborted', aborted);
    req.once('error', aborted);
  });
}

export interface QuestViewOptions {
  now?: () => number;
  bodyTimeoutMs?: number;
}
export type QuestViewMiddleware = (
  req: IncomingMessage,
  res: ServerResponse,
  next: Connect.NextFunction,
) => Promise<void>;

export function createQuestViewMiddleware(options: QuestViewOptions = {}): QuestViewMiddleware {
  const now = options.now ?? Date.now;
  const timeout = options.bodyTimeoutMs ?? 2000;
  let publisher: { id: string; clip: string; heartbeat: number } | null = null;
  let frame: { data: Buffer; sequence: number; updatedAt: number } | null = null;
  let uploading = false;
  let readingStatus = false;
  const viewers = new Map<string, number>();
  function expire() {
    const time = now();
    if (publisher && time - publisher.heartbeat >= 4000) {
      publisher = null;
      frame = null;
    }
    for (const [id, seen] of viewers) if (time - seen >= 3000) viewers.delete(id);
  }
  function status() {
    expire();
    return {
      active: publisher !== null,
      sessionId: publisher?.id ?? null,
      clip: publisher?.clip ?? null,
      frame: frame?.sequence ?? 0,
      updatedAt: frame?.updatedAt ?? null,
      viewers: viewers.size,
    };
  }
  return async (req, res, next) => {
    const path = req.url?.split('?')[0];
    if (path !== '/api/quest-view' && path !== '/api/quest-view/frame') return next();
    res.setHeader('Cache-Control', 'no-store');
    try {
      const url = new URL(req.url!, 'http://localhost');
      expire();
      if (req.method !== 'GET' && req.method !== 'POST') {
        res.setHeader('Allow', 'GET, POST');
        throw new RequestError(405);
      }
      if (req.method === 'POST' && req.headers.origin !== undefined) {
        // Compare to the actual connection origin, never forwarded headers.
        const protocol = 'encrypted' in req.socket && req.socket.encrypted ? 'https' : 'http';
        if (req.headers.origin !== `${protocol}://${req.headers.host}`) throw new RequestError(403);
      }
      if (path === '/api/quest-view' && req.method === 'GET') {
        const viewer = url.searchParams.get('viewer');
        if (viewer !== null) {
          if (!ID.test(viewer)) throw new RequestError(400);
          if (!viewers.has(viewer) && viewers.size >= 64) throw new RequestError(429);
          viewers.set(viewer, now());
        }
      } else if (path === '/api/quest-view') {
        if (req.headers['content-type']?.split(';')[0] !== 'application/json')
          throw new RequestError(415);
        if (readingStatus) throw new RequestError(409);
        readingStatus = true;
        let payload: unknown;
        try {
          payload = JSON.parse((await readBody(req, 1024, timeout)).toString('utf8'));
        } catch (error) {
          throw error instanceof RequestError ? error : new RequestError(400);
        } finally {
          readingStatus = false;
        }
        if (!payload || typeof payload !== 'object') throw new RequestError(400);
        const { sessionId, clip, active } = payload as Record<string, unknown>;
        if (
          typeof sessionId !== 'string' ||
          !ID.test(sessionId) ||
          typeof clip !== 'string' ||
          !CLIP.test(clip) ||
          typeof active !== 'boolean'
        )
          throw new RequestError(400);
        expire();
        if (active) {
          if (publisher && publisher.id !== sessionId) throw new RequestError(409);
          if (!publisher || publisher.clip !== clip) {
            frame = null;
            publisher = { id: sessionId, clip, heartbeat: now() };
          } else publisher.heartbeat = now();
        } else if (publisher?.id === sessionId) {
          publisher = null;
          frame = null;
        }
      } else {
        const session = url.searchParams.get('session');
        const seq = url.searchParams.get('seq') ?? '';
        if (
          !session ||
          !ID.test(session) ||
          !/^\d+$/.test(seq) ||
          !Number.isSafeInteger(Number(seq))
        )
          throw new RequestError(400);
        if (req.method === 'GET') {
          if (publisher?.id === session && frame && now() - frame.updatedAt <= 2500) {
            res.setHeader('Content-Type', 'image/jpeg');
            res.setHeader('X-Quest-Frame', frame.sequence);
            res.end(frame.data);
          } else {
            res.statusCode = 204;
            res.end();
          }
          return;
        }
        if (Number(seq) <= 0) throw new RequestError(400);
        if (req.headers['content-type'] !== 'image/jpeg') throw new RequestError(415);
        if (publisher?.id !== session || Number(seq) <= (frame?.sequence ?? 0))
          throw new RequestError(409);
        if (uploading) throw new RequestError(409);
        const owner = publisher;
        uploading = true;
        try {
          const data = await readBody(req, MAX_FRAME_BYTES, timeout);
          expire();
          // A stopped/restarted session must not accept an older in-flight upload.
          if (publisher !== owner || Number(seq) <= (frame?.sequence ?? 0))
            throw new RequestError(409);
          if (
            data.length < 4 ||
            data[0] !== 0xff ||
            data[1] !== 0xd8 ||
            data[2] !== 0xff ||
            data.at(-2) !== 0xff ||
            data.at(-1) !== 0xd9
          )
            throw new RequestError(400);
          frame = { data, sequence: Number(seq), updatedAt: now() };
        } finally {
          uploading = false;
        }
        res.statusCode = 204;
        res.end();
        return;
      }
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify(status()));
    } catch (error) {
      res.statusCode = error instanceof RequestError ? error.status : 500;
      req.resume();
      res.end();
    }
  };
}

export function questView(
  options: QuestViewOptions = {},
): Plugin & { middleware: QuestViewMiddleware } {
  const middleware = createQuestViewMiddleware(options);
  return {
    name: 'wander-quest-view',
    middleware,
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        void middleware(req, res, next);
      });
    },
  };
}
