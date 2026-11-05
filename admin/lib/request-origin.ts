function httpOrigin(value: string): string | null {
  if (!/^https?:\/\/[^/?#\\]+\/?$/.test(value)) return null;
  try {
    const url = new URL(value);
    if (
      !['http:', 'https:'].includes(url.protocol) ||
      url.hostname.includes('*') ||
      url.username ||
      url.password ||
      url.pathname !== '/' ||
      url.search ||
      url.hash
    )
      return null;
    return url.origin;
  } catch {
    return null;
  }
}

function loopbackOrigin(origin: string | null): string | null {
  return origin && ['localhost', '127.0.0.1', '[::1]'].includes(new URL(origin).hostname)
    ? origin
    : null;
}

/** This server attaches API credentials, so browser mutations must come from a trusted origin. */
export function mutationOriginAllowed(request: Request, configuredOrigin?: string): boolean {
  if (['GET', 'HEAD', 'OPTIONS'].includes(request.method)) return true;
  if (request.headers.get('sec-fetch-site')?.toLowerCase() === 'cross-site') return false;
  let appOrigin: string | null = null;
  if (configuredOrigin?.trim()) {
    appOrigin = httpOrigin(configuredOrigin.trim());
    if (!appOrigin)
      throw new Error('WANDER_APP_ORIGIN must be an HTTP(S) origin without credentials or a path');
  }
  const source = request.headers.get('origin');
  // Local CLI tools do not send Origin. Browser opaque origins ("null") remain untrusted.
  if (source === null) return true;
  const origin = httpOrigin(source);
  const url = new URL(request.url);
  // Next's production Request URL can use localhost even when the browser addresses 127.0.0.1.
  // Host is the actual HTTP destination; do not trust client-supplied forwarded-host headers.
  const host = request.headers.get('host');
  const destinationOrigin = loopbackOrigin(host ? httpOrigin(`${url.protocol}//${host}`) : null);
  return (
    origin !== null &&
    (origin === loopbackOrigin(url.origin) || origin === destinationOrigin || origin === appOrigin)
  );
}
