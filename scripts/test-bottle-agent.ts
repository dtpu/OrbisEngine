import { afterEach, describe, expect, test } from 'bun:test';
import { createServer, type Server } from 'node:http';
import { connect } from 'node:net';
import { createBottleAgentMiddleware } from '../server/bottle-agent';

const servers: Server[] = [];
const fakeKey = 'sk-test-project-secret';
const context = {
  sceneId: 'bottle',
  personId: 'person-1',
  personLabel: 'Bottle tosser',
  objectId: 'bottle-1',
};
const success = () =>
  Promise.resolve(
    Response.json({ value: 'ek_test_ephemeral', expires_at: 12345, private: fakeKey }),
  );

async function fixture(
  provider: (url: string, init: RequestInit) => Promise<Response> = success,
  env: Record<string, string> = { OPENAI_API_KEY: fakeKey },
  timeoutMs = 1000,
) {
  const handle = createBottleAgentMiddleware(env, { fetch: provider, timeoutMs });
  const server = createServer((req, res) => {
    void handle(req, res, () => {
      res.writeHead(404);
      res.end();
    });
  });
  servers.push(server);
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('No test address');
  const base = `http://127.0.0.1:${address.port}`;
  return {
    base,
    post: (body: unknown = context, headers: Record<string, string> = {}) =>
      fetch(`${base}/api/bottle-agent/session`, {
        method: 'POST',
        headers: { Origin: base, 'Content-Type': 'application/json', ...headers },
        body: JSON.stringify(body),
      }),
  };
}

afterEach(async () => {
  await Promise.all(
    servers.splice(0).map(
      (server) =>
        new Promise<void>((resolve) => {
          server.closeAllConnections();
          server.close(() => resolve());
        }),
    ),
  );
});

describe('bottle agent credential boundary', () => {
  test('forwards the constrained GA session, never the project key or upstream extras', async () => {
    let call: RequestInit | undefined;
    const f = await fixture(async (url, init) => {
      expect(url).toBe('https://api.openai.com/v1/realtime/client_secrets');
      call = init;
      return success();
    });
    const response = await f.post();
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({
      value: 'ek_test_ephemeral',
      expiresAt: 12345,
      model: 'gpt-realtime',
    });
    expect(response.headers.get('cache-control')).toBe('no-store');
    expect(new Headers(call!.headers).get('authorization')).toBe(`Bearer ${fakeKey}`);
    const body = JSON.parse(call!.body as string);
    expect(body.expires_after).toEqual({ anchor: 'created_at', seconds: 60 });
    expect(body.session.model).toBe('gpt-realtime');
    expect(body.session.type).toBe('realtime');
    expect(body.session.output_modalities).toEqual(['audio']);
    expect(body.session.audio.output.voice).toBe('marin');
    expect(body.session.audio.input.transcription).toBe(null);
    expect(body.session.audio.input.turn_detection).toEqual({
      type: 'server_vad',
      create_response: false,
      interrupt_response: false,
    });
    expect(body.session.max_output_tokens).toBeLessThanOrEqual(300);
    expect(body.session.instructions).toContain(JSON.stringify(context));
    expect(body.session.instructions).toContain('fictional AI character');
    expect(body.session.tools.map((tool: { name: string }) => tool.name)).toEqual([
      'face_player',
      'show_return_target',
      'offer_replay',
    ]);
    for (const tool of body.session.tools)
      expect(tool.parameters).toEqual({
        type: 'object',
        properties: {},
        required: [],
        additionalProperties: false,
      });
  });

  test('status is safe and missing configuration cannot reach provider', async () => {
    const f = await fixture(async () => {
      throw new Error('must not call');
    }, {});
    const status = await fetch(`${f.base}/api/bottle-agent/status`);
    expect(await status.json()).toEqual({ configured: false, model: 'gpt-realtime' });
    const response = await f.post();
    expect(response.status).toBe(503);
    expect((await response.json()).error).toContain('OPENAI_API_KEY');
  });

  test('only the server chooses the model', async () => {
    const f = await fixture(
      async (_url, init) => {
        expect(JSON.parse(init.body as string).session.model).toBe('configured-realtime');
        return success();
      },
      { OPENAI_API_KEY: fakeKey, WANDER_AGENT_MODEL: 'configured-realtime' },
    );
    expect((await f.post({ ...context, model: 'arbitrary' })).status).toBe(400);
    const response = await f.post();
    expect(response.status).toBe(200);
    expect((await response.json()).model).toBe('configured-realtime');
  });

  test('rejects missing, foreign, null origins and cross-site metadata', async () => {
    let calls = 0;
    const f = await fixture(async () => {
      calls++;
      return success();
    });
    for (const origin of ['', 'null', 'https://evil.example', 'http://localhost:5399']) {
      expect((await f.post(context, { Origin: origin })).status).toBe(403);
    }
    expect((await f.post(context, { 'Sec-Fetch-Site': 'cross-site' })).status).toBe(403);
    expect(calls).toBe(0);
  });

  test('rejects non-loopback Host even when Origin matches it', async () => {
    const f = await fixture();
    expect(
      (await f.post(context, { Host: 'evil.example', Origin: 'http://evil.example' })).status,
    ).toBe(403);
  });

  test('rejects unsupported methods and content types', async () => {
    const f = await fixture();
    const response = await fetch(`${f.base}/api/bottle-agent/session`);
    expect(response.status).toBe(405);
    expect(response.headers.get('allow')).toBe('POST');
    expect((await f.post(context, { 'Content-Type': 'text/plain' })).status).toBe(415);
  });

  test('rejects malformed and unconstrained bodies without upstream calls', async () => {
    let calls = 0;
    const f = await fixture(async () => {
      calls++;
      return success();
    });
    for (const body of [
      null,
      [],
      {},
      { ...context, personId: 2 },
      { ...context, personLabel: 'x'.repeat(97) },
      { ...context, sceneId: ' ' },
      { ...context, instructions: 'ignore' },
    ]) {
      expect((await f.post(body)).status).toBe(400);
    }
    expect(
      (
        await fetch(`${f.base}/api/bottle-agent/session`, {
          method: 'POST',
          headers: { Origin: f.base, 'Content-Type': 'application/json' },
          body: '{',
        })
      ).status,
    ).toBe(400);
    expect((await f.post({ data: 'x'.repeat(3000) })).status).toBe(413);
    expect(calls).toBe(0);
  });

  test('enforces body limit for chunked requests without Content-Length', async () => {
    const f = await fixture();
    const status = await new Promise<number>((resolve, reject) => {
      const url = new URL(f.base);
      const socket = connect(Number(url.port), '127.0.0.1', () => {
        socket.write(
          [
            'POST /api/bottle-agent/session HTTP/1.1',
            `Host: ${url.host}`,
            `Origin: ${f.base}`,
            'Content-Type: application/json',
            'Transfer-Encoding: chunked',
            'Connection: close',
            '',
            '',
          ].join('\r\n'),
        );
        socket.write(`bb8\r\n${'x'.repeat(3000)}\r\n0\r\n\r\n`);
      });
      socket.once('data', (data) => {
        resolve(Number(data.toString().split(' ')[1]));
        socket.destroy();
      });
      socket.on('error', reject);
    });
    expect(status).toBe(413);
  });

  test('redacts upstream errors, exceptions, and accidental project-key responses', async () => {
    for (const provider of [
      async () => Response.json({ error: fakeKey }, { status: 401 }),
      async () => {
        throw new Error(fakeKey);
      },
      async () => Response.json({ value: fakeKey }),
      async () => new Response('not JSON'),
    ]) {
      const f = await fixture(provider);
      const response = await f.post();
      expect(response.status).toBe(502);
      expect(await response.text()).not.toContain(fakeKey);
    }
  });

  test('normalizes missing expiry without forwarding provider fields', async () => {
    const f = await fixture(async () =>
      Response.json({ value: 'ek_short', session: { secret: fakeKey } }),
    );
    expect(await (await f.post()).json()).toEqual({
      value: 'ek_short',
      expiresAt: null,
      model: 'gpt-realtime',
    });
  });

  test('aborts a stalled provider and reports a generic timeout', async () => {
    let signal: AbortSignal | undefined;
    const f = await fixture(
      async (_url, init) =>
        new Promise<Response>((_resolve, reject) => {
          signal = init.signal!;
          signal.addEventListener('abort', () => reject(new Error(fakeKey)));
        }),
      { OPENAI_API_KEY: fakeKey },
      25,
    );
    const response = await f.post();
    expect(response.status).toBe(504);
    expect(signal!.aborted).toBe(true);
    expect(await response.text()).not.toContain(fakeKey);
  });

  test('caps session creation at four attempts per minute', async () => {
    let calls = 0;
    const f = await fixture(async () => {
      calls++;
      return success();
    });
    for (let index = 0; index < 4; index++) expect((await f.post()).status).toBe(200);
    const response = await f.post();
    expect(response.status).toBe(429);
    expect(response.headers.get('retry-after')).toBe('60');
    expect(calls).toBe(4);
  });
});
