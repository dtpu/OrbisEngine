import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mutationOriginAllowed } from '../lib/request-origin';

const admin = 'http://127.0.0.1:5401';
const gateway = 'http://127.0.0.1:5399';
const request = (headers: HeadersInit = {}, method = 'POST') =>
  new Request(`${admin}/api/pipeline/runs/upload`, { method, headers });

test('mutations allow the request origin, explicitly configured gateway, and local CLI tools', () => {
  assert.equal(mutationOriginAllowed(request({ Origin: admin })), true);
  assert.equal(mutationOriginAllowed(request({ Origin: gateway }), gateway), true);
  assert.equal(mutationOriginAllowed(request()), true);
  assert.equal(mutationOriginAllowed(request({ Origin: gateway })), false);
  assert.equal(
    mutationOriginAllowed(request({ Origin: 'http://127.0.0.1:53990' }), gateway),
    false,
  );
});

test('production host matching handles Next localhost URLs without trusting forwarded-host', () => {
  assert.equal(
    mutationOriginAllowed(
      new Request('http://localhost:5401/api/pipeline/runs/upload', {
        method: 'POST',
        headers: { Origin: admin, Host: '127.0.0.1:5401' },
      }),
    ),
    true,
  );
  assert.equal(
    mutationOriginAllowed(request({ Origin: gateway, 'X-Forwarded-Host': '127.0.0.1:5399' })),
    false,
  );
});

test('arbitrary request URL and Host domains require explicit app-origin configuration', () => {
  const origin = 'http://rebind.attacker.example:5401';
  for (const url of ['http://localhost:5401', origin]) {
    const candidate = new Request(`${url}/api/pipeline/runs/upload`, {
      method: 'POST',
      headers: {
        Origin: origin,
        Host: 'rebind.attacker.example:5401',
        'Sec-Fetch-Site': 'same-origin',
      },
    });
    assert.equal(mutationOriginAllowed(candidate), false);
    assert.equal(mutationOriginAllowed(candidate, origin), true);
  }
});

test('cross-site metadata and malformed or opaque origins cannot authorize mutations', () => {
  for (const origin of [
    'null',
    'https://untrusted.example',
    `${admin}.untrusted.example`,
    `${admin}/path/..`,
    'http://user:secret@127.0.0.1:5401',
  ]) {
    assert.equal(mutationOriginAllowed(request({ Origin: origin }), gateway), false);
  }
  assert.equal(mutationOriginAllowed(request({ 'Sec-Fetch-Site': 'cross-site' }), gateway), false);
  assert.equal(
    mutationOriginAllowed(request({ Origin: gateway, 'Sec-Fetch-Site': 'cross-site' }), gateway),
    false,
  );
  assert.equal(
    mutationOriginAllowed(request({ Origin: 'https://untrusted.example' }, 'GET')),
    true,
  );
});

test('the configured app origin cannot contain credentials, paths, wildcard hosts or extra URL parts', () => {
  for (const origin of [
    '*',
    'https://*.example',
    'http://user:secret@127.0.0.1:5399',
    `${gateway}/path`,
    `${gateway}/path/..`,
    `${gateway}?token=value`,
    `${gateway}#fragment`,
  ]) {
    assert.throws(() => mutationOriginAllowed(request(), origin), /WANDER_APP_ORIGIN/);
  }
});

test('a malicious multipart upload is rejected by the route before any upstream forwarding', async () => {
  const { POST } = await import('../app/api/pipeline/[[...path]]/route');
  const originalFetch = globalThis.fetch;
  let forwards = 0;
  globalThis.fetch = (async () => {
    forwards += 1;
    return Response.json({ id: 'unexpected' }, { status: 201 });
  }) as unknown as typeof fetch;
  try {
    const body = new FormData();
    body.set('source', new Blob(['source bytes']), 'source.mp4');
    body.set('options', JSON.stringify({ hands_off: true }));
    const response = await POST(
      new Request(`${admin}/api/pipeline/runs/upload`, {
        method: 'POST',
        headers: { Origin: 'https://untrusted.example' },
        body,
      }),
      { params: Promise.resolve({ path: ['runs', 'upload'] }) },
    );
    assert.equal(response.status, 403);
    assert.deepEqual(await response.json(), { detail: 'Untrusted request origin' });
    assert.equal(forwards, 0);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
