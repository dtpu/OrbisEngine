import { expect, test } from 'bun:test';
import { worldPreviewAssetUrl } from '../src/world-preview-path';

test('world preview preserves existing same-origin local and proxied run SPZ routes', () => {
  for (const path of [
    '/marble-lobby-e2e-20261105-clean.spz',
    '/api/pipeline/runs/local-e2e-delivery-20261105/viewer/assets/marble-local-e2e-delivery-20261105-clean.spz',
  ])
    expect(worldPreviewAssetUrl(path)).toBe(path);
});

test('world preview refuses arbitrary origins, traversal, code, private files and other API routes', () => {
  for (const path of [
    null,
    '',
    'https://example.com/world.spz',
    '//example.com/world.spz',
    '/api/pipeline/runs/../viewer/assets/world.spz',
    '/api/pipeline/runs/a/viewer/assets/../world.spz',
    '/api/pipeline/runs/a/viewer/assets/%2e%2e/world.spz',
    '/api/pipeline/runs/a/viewer/assets/world.spz?key=private',
    '/.context/secret.spz',
    '/@fs/tmp/world.spz',
    '/src/world.spz',
    '/api/secret.spz',
    '/world.js',
  ])
    expect(() => worldPreviewAssetUrl(path)).toThrow('same-origin SPZ');
});
