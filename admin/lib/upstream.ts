const BASE = process.env.WANDER_PIPELINE_API ?? 'http://127.0.0.1:8000';
const TOKEN = process.env.WANDER_API_TOKEN;

/**
 * Hop-by-hop and length headers must not be copied onto a re-issued request or response.
 * `expect` is included because undici rejects forwarding it outright (UND_ERR_NOT_SUPPORTED),
 * and CLI clients send `Expect: 100-continue` on large uploads.
 */
const STRIPPED = new Set([
  'connection',
  'keep-alive',
  'transfer-encoding',
  'upgrade',
  'content-length',
  'content-encoding',
  'expect',
  'host',
]);

export function upstreamUrl(segments: string[], search: string): string {
  const suffix = segments.map(encodeURIComponent).join('/');
  return `${BASE}/api/pipeline${suffix ? `/${suffix}` : ''}${search}`;
}

export function forwardHeaders(source: Headers): Headers {
  const headers = new Headers();
  source.forEach((value, key) => {
    if (!STRIPPED.has(key.toLowerCase())) headers.set(key, value);
  });
  // The operator's browser never sees the pipeline token; it is attached here, server-side.
  if (TOKEN) headers.set('authorization', `Bearer ${TOKEN}`);
  else headers.delete('authorization');
  return headers;
}

export function responseHeaders(source: Headers): Headers {
  const headers = new Headers();
  source.forEach((value, key) => {
    if (!STRIPPED.has(key.toLowerCase())) headers.set(key, value);
  });
  return headers;
}

export function tokenConfigured(): boolean {
  return Boolean(TOKEN);
}
