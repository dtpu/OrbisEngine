/** Only an origin is public configuration; never serialize credentials or other URL parts. */
export function parseViewerOrigin(value: string | undefined): string {
  if (!value?.trim()) return '';
  let url: URL;
  try {
    url = new URL(value.trim());
  } catch {
    throw new Error('WANDER_VIEWER_ORIGIN must be an HTTP(S) origin');
  }
  if (
    !['http:', 'https:'].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.pathname !== '/' ||
    url.search ||
    url.hash
  ) {
    throw new Error('WANDER_VIEWER_ORIGIN must be an HTTP(S) origin without credentials or a path');
  }
  return url.origin;
}

/** Viewer paths come from run metadata, and must remain on the configured viewer origin. */
export function viewerUrl(path: string, origin = ''): string {
  if (!/^\/(?:demo|fourd)\.html(?:[?#]|$)/.test(path) || /[\\\r\n\t]/.test(path)) {
    throw new Error('Invalid viewer path');
  }
  return `${parseViewerOrigin(origin)}${path}`;
}
