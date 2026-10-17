import assert from 'node:assert/strict';
import { test } from 'node:test';
import { checkSavedInputs, prepareSavedInputRun } from '../lib/client';

test('saved input requests use the authenticated same-origin proxy and encode run ids', async () => {
  const originalFetch = globalThis.fetch;
  const requests: Array<{ input: RequestInfo | URL; init?: RequestInit }> = [];
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    requests.push({ input, init });
    return Response.json({
      available: true,
      sourceRunId: 'source/one',
      fileCount: 3,
      byteCount: 12,
    });
  }) as typeof fetch;
  try {
    const result = await checkSavedInputs('run/one');
    assert.deepEqual(result, {
      available: true,
      sourceRunId: 'source/one',
      fileCount: 3,
      byteCount: 12,
    });
    assert.equal(String(requests[0]?.input), '/api/pipeline/runs/run%2Fone/saved-inputs');
    assert.equal(requests[0]?.init?.cache, 'no-store');

    await prepareSavedInputRun('run/one', 'delivery-1');
    assert.equal(String(requests[1]?.input), '/api/pipeline/runs/run%2Fone/saved-inputs');
    assert.equal(requests[1]?.init?.method, 'POST');
    assert.deepEqual(JSON.parse(String(requests[1]?.init?.body)), { id: 'delivery-1' });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('saved input requests surface proxy detail for retry messaging', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () =>
    Response.json(
      { detail: 'source run is still active' },
      { status: 409 },
    )) as unknown as typeof fetch;
  try {
    await assert.rejects(prepareSavedInputRun('source'), /source run is still active/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
