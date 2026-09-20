import { test } from 'bun:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { connect } from 'node:net';
import type { AddressInfo } from 'node:net';
import { createQuestViewMiddleware } from '../server/quest-view.ts';

const jpeg = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0x01, 0xff, 0xd9]);
async function fixture(
  run: (f: {
    base: string;
    advance: (ms: number) => void;
    status: (sessionId?: string, active?: boolean) => Promise<Response>;
    upload: (seq: number, body?: Buffer, session?: string) => Promise<Response>;
  }) => Promise<void>,
) {
  let clock = 10000;
  const middleware = createQuestViewMiddleware({ now: () => clock, bodyTimeoutMs: 80 });
  const server = createServer((req, res) => {
    void middleware(req, res, () => {
      res.statusCode = 404;
      res.end();
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}/api/quest-view`;
  try {
    await run({
      base,
      advance: (ms) => {
        clock += ms;
      },
      status: (sessionId = 'publisher-1', active = true) =>
        fetch(base, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sessionId, clip: 'elevator', active }),
        }),
      upload: (seq, body = jpeg, session = 'publisher-1') =>
        fetch(`${base}/frame?session=${session}&seq=${seq}`, {
          method: 'POST',
          headers: { 'Content-Type': 'image/jpeg' },
          body: new Uint8Array(body),
        }),
    });
  } finally {
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
}

test('publisher ownership, latest-only delivery, stop and expiry', async () => {
  await fixture(async ({ base, status, upload, advance }) => {
    assert.deepEqual(await (await fetch(base)).json(), {
      active: false,
      sessionId: null,
      clip: null,
      frame: 0,
      updatedAt: null,
      viewers: 0,
    });
    assert.equal((await status()).status, 200);
    assert.equal((await status('publisher-2')).status, 409);
    assert.equal((await upload(1)).status, 204);
    const newer = Buffer.concat([jpeg.subarray(0, -2), Buffer.from([42]), jpeg.subarray(-2)]);
    assert.equal((await upload(3, newer)).status, 204);
    assert.equal((await upload(2)).status, 409);
    const frame = await fetch(`${base}/frame?session=publisher-1&seq=1`);
    assert.equal(frame.headers.get('x-quest-frame'), '3');
    assert.equal(frame.headers.get('cache-control'), 'no-store');
    assert.deepEqual(Buffer.from(await frame.arrayBuffer()), newer);
    assert.equal((await fetch(`${base}/frame?session=publisher-2&seq=3`)).status, 204);
    await status('publisher-2', false);
    assert.equal((await (await fetch(base)).json()).active, true);
    advance(2501);
    assert.equal((await fetch(`${base}/frame?session=publisher-1&seq=3`)).status, 204);
    assert.equal((await (await fetch(base)).json()).frame, 3);
    advance(1500);
    assert.equal((await (await fetch(base)).json()).active, false);
    assert.equal((await upload(4)).status, 409);
    await status('publisher-2');
    assert.equal((await (await fetch(base)).json()).frame, 0);
    await status('publisher-2', false);
    assert.equal((await (await fetch(base)).json()).sessionId, null);
  });
});

test('viewer leases expire and cap at 64 without read-only observers adding leases', async () => {
  await fixture(async ({ base, advance, status }) => {
    for (let i = 0; i < 64; i++)
      assert.equal((await fetch(`${base}?viewer=viewer-${i}`)).status, 200);
    assert.equal((await fetch(`${base}?viewer=overflow`)).status, 429);
    assert.equal((await fetch(`${base}?viewer=viewer-0`)).status, 200);
    assert.equal((await (await status()).json()).viewers, 64);
    advance(3000);
    assert.equal((await (await fetch(base)).json()).viewers, 0);
    assert.equal((await fetch(`${base}?viewer=overflow`)).status, 200);
  });
});

test('invalid input, size, methods and origins cannot replace a good frame', async () => {
  await fixture(async ({ base, status, upload }) => {
    await status();
    await upload(1);
    assert.equal((await upload(2, Buffer.from('invalid'))).status, 400);
    assert.equal((await upload(2, Buffer.alloc(256 * 1024 + 1))).status, 413);
    assert.equal((await upload(0)).status, 400);
    assert.equal(
      (
        await fetch(`${base}/frame?session=publisher-1&seq=2`, {
          method: 'POST',
          headers: { 'Content-Type': 'text/plain' },
          body: jpeg,
        })
      ).status,
      415,
    );
    assert.equal((await fetch(base, { method: 'DELETE' })).status, 405);
    assert.equal((await fetch(`${base}-other`)).status, 404);
    for (const body of ['{', 'null', '{"sessionId":"../x","clip":"elevator","active":true}']) {
      assert.equal(
        (
          await fetch(base, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body,
          })
        ).status,
        400,
      );
    }
    assert.equal(
      (
        await fetch(base, {
          method: 'POST',
          headers: { Origin: 'http://evil.example', 'Content-Type': 'application/json' },
          body: '{}',
        })
      ).status,
      403,
    );
    assert.equal(
      (
        await fetch(base, {
          method: 'POST',
          headers: { Origin: new URL(base).origin, 'Content-Type': 'application/json' },
          body: JSON.stringify({ sessionId: 'publisher-1', clip: 'elevator', active: true }),
        })
      ).status,
      200,
    );
    assert.equal((await (await fetch(base)).json()).frame, 1);
  });
});

async function streaming(base: string) {
  const url = new URL(base);
  const req = connect(Number(url.port), url.hostname);
  await new Promise<void>((resolve) => req.once('connect', resolve));
  let resolve!: (code: number) => void;
  const done = new Promise<number>((r) => {
    resolve = r;
  });
  req.on('data', (chunk) => {
    const match = /^HTTP\/1\.1 (\d+)/.exec(chunk.toString());
    if (match) resolve(Number(match[1]));
  });
  req.on('error', () => {});
  req.write(
    `POST /api/quest-view/frame?session=publisher-1&seq=2 HTTP/1.1\r\nHost: ${url.host}\r\nContent-Type: image/jpeg\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n`,
  );
  function chunk(data: Buffer) {
    req.write(data.length.toString(16) + '\r\n');
    req.write(data);
    req.write('\r\n');
  }
  chunk(jpeg.subarray(0, 3));
  return {
    req,
    done,
    end(data: Buffer) {
      chunk(data);
      req.write('0\r\n\r\n');
    },
  };
}

test('chunked oversize, body timeout and stopped-session uploads are rejected', async () => {
  await fixture(async ({ base, status, upload }) => {
    await status();
    await upload(1);
    const oversized = await streaming(base);
    oversized.end(Buffer.alloc(256 * 1024));
    assert.equal(await oversized.done, 413);
    const stalled = await streaming(base);
    assert.equal(await stalled.done, 408);
    stalled.req.destroy();
    assert.equal((await (await fetch(base)).json()).frame, 1);
    const old = await streaming(base);
    // A status read ensures the server has started receiving the partial frame.
    await fetch(base);
    await status('publisher-1', false);
    await status('publisher-1');
    old.end(jpeg.subarray(3));
    assert.equal(await old.done, 409);
    assert.equal((await (await fetch(base)).json()).frame, 0);
  });
});
