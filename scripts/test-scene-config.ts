import { describe, expect, test } from 'bun:test';
import { resolveSceneConfig } from '../src/scene-config.ts';

const catalog = [{ id: 'reviewed-scene', sceneManifest: '/scenes/reviewed.json' }];
const manifest = (params: unknown) => ({ schema: 'wander.viewer-scene/1', params });
function fixture(body: unknown = manifest({ world: '/world.spz' }), status = 200) {
  const requests: string[] = [];
  const scope = {
    abort: new AbortController(),
    fetch: async (path: string) => {
      requests.push(path);
      return Response.json(body, { status });
    },
  };
  return { scope, requests };
}

describe('private scene configuration', () => {
  test('loads catalog defaults and retains identity and explicit viewing choices', async () => {
    const { scope, requests } = fixture(
      manifest({ world: '/world.spz', walk: '0', audio: '0', xr: '0' }),
    );
    const query = new URLSearchParams('demo=reviewed-scene&walk=1&audio=1&xr=1');
    const result = await resolveSceneConfig(query, catalog, scope);
    expect(requests).toEqual(['/scenes/reviewed.json']);
    expect(Object.fromEntries(result)).toEqual({
      demo: 'reviewed-scene',
      walk: '1',
      audio: '1',
      xr: '1',
      world: '/world.spz',
    });
    expect(query.has('world')).toBe(false);
  });

  test('explicit scene path overrides catalog selection; historical ids need no manifest', async () => {
    const { scope, requests } = fixture();
    await resolveSceneConfig(
      new URLSearchParams('demo=reviewed-scene&scene=/scenes/other.json'),
      catalog,
      scope,
    );
    const old = await resolveSceneConfig(new URLSearchParams('demo=historical'), catalog, scope);
    expect(requests).toEqual(['/scenes/other.json']);
    expect(old.get('demo')).toBe('historical');
  });

  test.each([
    'https://other.test/scene.json',
    '//other.test/scene.json',
    '/\\other.test/scene.json',
    '',
    '/\n/other.test',
  ])('rejects unsafe manifest path %j', async (path) => {
    const { scope, requests } = fixture();
    await expect(
      resolveSceneConfig(new URLSearchParams({ scene: path }), catalog, scope),
    ).rejects.toThrow('root-relative');
    expect(requests).toEqual([]);
  });

  test.each([
    null,
    [[]],
    {},
    { schema: 'wrong', params: {} },
    manifest([]),
    manifest({ world: 42 }),
    manifest({ demo: 'other' }),
    manifest({ scene: '/recursive.json' }),
    manifest({ world: '/bad\npath' }),
  ])('rejects malformed or recursive configuration %j', async (body) => {
    const { scope } = fixture(body);
    await expect(
      resolveSceneConfig(new URLSearchParams('demo=reviewed-scene'), catalog, scope),
    ).rejects.toThrow('Invalid scene manifest');
  });

  test('propagates HTTP and offline failures without silently using a different scene', async () => {
    const { scope } = fixture({}, 404);
    await expect(
      resolveSceneConfig(new URLSearchParams('demo=reviewed-scene'), catalog, scope),
    ).rejects.toThrow('could not download');
    const offline = new TypeError('offline');
    scope.fetch = async () => {
      throw offline;
    };
    await expect(
      resolveSceneConfig(new URLSearchParams('demo=reviewed-scene'), catalog, scope),
    ).rejects.toBe(offline);
  });

  test('reports invalid JSON and preserves body cancellation', async () => {
    const { scope } = fixture();
    scope.fetch = async () => new Response('{');
    await expect(
      resolveSceneConfig(new URLSearchParams('demo=reviewed-scene'), catalog, scope),
    ).rejects.toThrow('could not be read');
    scope.fetch = async () =>
      ({
        ok: true,
        json: async () => {
          scope.abort.abort();
          throw scope.abort.signal.reason;
        },
      }) as unknown as Response;
    await expect(
      resolveSceneConfig(new URLSearchParams('demo=reviewed-scene'), catalog, scope),
    ).rejects.toHaveProperty('name', 'AbortError');
  });

  test('cancellation before and during fetch cannot apply defaults', async () => {
    const before = fixture();
    before.scope.abort.abort();
    await expect(
      resolveSceneConfig(new URLSearchParams('demo=reviewed-scene'), catalog, before.scope),
    ).rejects.toHaveProperty('name', 'AbortError');
    expect(before.requests).toEqual([]);
    const during = fixture();
    during.scope.fetch = async () => {
      during.scope.abort.abort();
      return Response.json(manifest({ world: '/world.spz' }));
    };
    await expect(
      resolveSceneConfig(new URLSearchParams('demo=reviewed-scene'), catalog, during.scope),
    ).rejects.toHaveProperty('name', 'AbortError');
  });
});
