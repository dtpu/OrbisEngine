import { forwardHeaders, responseHeaders, tokenConfigured, upstreamUrl } from '@/lib/upstream';

// The panel talks only to this proxy. It adds the bearer token server-side and streams the
// response through untouched, so SSE (/events) and multipart uploads both pass unbuffered.
export const dynamic = 'force-dynamic';

type Context = { params: Promise<{ path?: string[] }> };

async function proxy(request: Request, context: Context): Promise<Response> {
  if (!tokenConfigured()) {
    return Response.json(
      { detail: 'WANDER_API_TOKEN is not set for the admin server' },
      { status: 503 },
    );
  }
  const { path } = await context.params;
  const { search } = new URL(request.url);
  const method = request.method;
  const hasBody = method !== 'GET' && method !== 'HEAD';
  let upstream: Response;
  try {
    upstream = await fetch(upstreamUrl(path ?? [], search), {
      method,
      headers: forwardHeaders(request.headers),
      body: hasBody ? request.body : undefined,
      // Required by undici whenever a streaming body is forwarded.
      ...(hasBody ? { duplex: 'half' } : {}),
      redirect: 'manual',
      cache: 'no-store',
    } as RequestInit);
  } catch (error) {
    const cause = (error as { cause?: unknown }).cause;
    console.error('pipeline proxy failed', error, cause);
    return Response.json(
      {
        detail: `pipeline API unreachable: ${(error as Error).message}${
          cause ? ` (${String((cause as Error).message ?? cause)})` : ''
        }`,
      },
      { status: 502 },
    );
  }
  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: responseHeaders(upstream.headers),
  });
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
