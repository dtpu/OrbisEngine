import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseViewerOrigin, viewerUrl } from '../lib/viewer-url';

test('viewer links default to same origin and preserve packaged-scene query parameters', () => {
  assert.equal(parseViewerOrigin(undefined), '');
  assert.equal(parseViewerOrigin('  '), '');
  assert.equal(viewerUrl('/demo.html'), '/demo.html');
  const path = '/fourd.html?runPreview=run%2Fone&scene=scene.json#preview';
  assert.equal(viewerUrl(path), path);
  assert.equal(viewerUrl(path, 'https://viewer.example/'), `https://viewer.example${path}`);
});

test('standalone viewer configuration accepts only a public HTTP origin', () => {
  assert.equal(parseViewerOrigin(' http://127.0.0.1:5399/ '), 'http://127.0.0.1:5399');
  for (const origin of [
    'invalid',
    'javascript:alert(1)',
    'ftp://viewer.example',
    'https://user:secret@viewer.example',
    'https://viewer.example/path',
    'https://viewer.example/?token=secret',
    'https://viewer.example/#secret',
  ]) {
    assert.throws(() => parseViewerOrigin(origin), /WANDER_VIEWER_ORIGIN/);
  }
});

test('run metadata cannot replace the viewer origin or link to another page', () => {
  for (const path of [
    '//evil.example/fourd.html',
    'https://evil.example/fourd.html',
    'javascript:alert(1)',
    '/api/pipeline/runs',
    '/demo.html/other',
    '/fourd.html?run=one\n',
    '/fourd.html?run=one\\two',
  ]) {
    assert.throws(() => viewerUrl(path), /Invalid viewer path/);
  }
});
