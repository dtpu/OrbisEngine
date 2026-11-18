import assert from 'node:assert/strict';
import { test } from 'node:test';
import { checkSetup } from '../lib/client';

for (const [name, response, expected] of [
  [
    'plain server error',
    new Response('Internal Server Error', { status: 500, statusText: 'Internal Server Error' }),
    '500 Internal Server Error',
  ],
  [
    'HTML proxy error',
    new Response('<html>private diagnostics</html>', { status: 502, statusText: 'Bad Gateway' }),
    '502 Bad Gateway',
  ],
  [
    'structured error',
    Response.json({ detail: 'Worker unavailable' }, { status: 503 }),
    'Worker unavailable',
  ],
  [
    'malformed success',
    new Response('not JSON'),
    'Pipeline API returned an invalid JSON response.',
  ],
] as const) {
  test(`setup reports ${name} without losing the actionable error`, async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async () => response.clone()) as unknown as typeof fetch;
    try {
      await assert.rejects(
        checkSetup({
          marble: 'video',
          people: 1,
          all_people: false,
          objects: false,
          finetune: false,
        }),
        { message: expected },
      );
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
}
