type SceneEntry = { id: string; sceneManifest?: string };
type SceneConfigScope = {
  fetch: (url: string) => Promise<Response>;
  abort: AbortController;
};

/** Load private scene defaults without replacing explicit URL choices or the scene identity. */
export async function resolveSceneConfig(
  query: URLSearchParams,
  catalog: readonly SceneEntry[],
  scope: SceneConfigScope,
): Promise<URLSearchParams> {
  const resolved = new URLSearchParams(query);
  const path = query.has('scene')
    ? query.get('scene')!
    : catalog.find((clip) => clip.id === query.get('demo'))?.sceneManifest;
  if (path === undefined) return resolved;
  // Root-relative paths keep manifests on the same private asset server.
  if (!/^\/(?!\/)/.test(path) || /[\\\u0000-\u0020\u007f]/.test(path)) {
    throw new Error('The scene manifest must use a root-relative asset path.');
  }
  scope.abort.signal.throwIfAborted();
  const response = await scope.fetch(path);
  scope.abort.signal.throwIfAborted();
  if (!response.ok)
    throw new Error('The scene manifest could not download. Choose the clip again to retry.');
  let manifest: unknown;
  try {
    manifest = await response.json();
  } catch (error) {
    scope.abort.signal.throwIfAborted();
    if (error instanceof Error && error.name === 'AbortError') throw error;
    throw new Error('The scene manifest could not be read. Choose the clip again to retry.');
  }
  scope.abort.signal.throwIfAborted();
  if (!manifest || typeof manifest !== 'object' || Array.isArray(manifest)) {
    throw new Error('Invalid scene manifest.');
  }
  const { schema, params } = manifest as { schema?: unknown; params?: unknown };
  if (
    schema !== 'wander.viewer-scene/1' ||
    !params ||
    typeof params !== 'object' ||
    Array.isArray(params)
  ) {
    throw new Error('Invalid scene manifest schema or parameters.');
  }
  for (const [key, value] of Object.entries(params)) {
    if (
      !/^[a-z][a-z0-9]*$/.test(key) ||
      key === 'demo' ||
      key === 'scene' ||
      typeof value !== 'string' ||
      /[\u0000-\u001f\u007f]/.test(value)
    ) {
      throw new Error(`Invalid scene manifest parameter: ${key}.`);
    }
    if (!resolved.has(key)) resolved.set(key, value);
  }
  return resolved;
}
