import assert from 'node:assert/strict';
import test from 'node:test';
import { sceneUsesLod, xrVignetteAmount } from '../src/xr/render-policy.ts';

const usesLod = (query: string) => sceneUsesLod(new URLSearchParams(query));

test('desktop keeps scene LoD preferences', () => {
  assert.equal(usesLod(''), true);
  assert.equal(usesLod('lod=1'), true);
  assert.equal(usesLod('lod=0'), false);
  assert.equal(usesLod('xr=0&lod=0'), false);
});

test('XR enables a budget for a scene that disables desktop LoD', () => {
  assert.equal(usesLod('xr=1&lod=0'), true);
  assert.equal(usesLod('xr=1'), true);
  assert.equal(usesLod('xr=1&lod=0&xrworldlod=0'), false);
  assert.equal(usesLod('xr=1&xrworldlod=0'), true);
});

test('automatic LoD preserves CPU colour paths in desktop and XR scenes', () => {
  for (const prefix of ['', 'xr=1&', 'xr=1&lod=0&']) {
    assert.equal(usesLod(`${prefix}bakedweights=weights.bin`), false);
    assert.equal(usesLod(`${prefix}obs=observations.bin&obsfade=0.5`), false);
  }
  for (const suffix of ['obs=observations.bin', 'obs=observations.bin&obsfade=0', 'obsfade=1']) {
    assert.equal(usesLod(`xr=1&lod=0&${suffix}`), true);
  }
});

test('explicit LoD remains an override for CPU colour and XR preferences', () => {
  assert.equal(usesLod('lod=1&bakedweights=weights.bin'), true);
  assert.equal(usesLod('lod=1&obs=observations.bin&obsfade=1'), true);
  assert.equal(usesLod('xr=1&lod=1&xrworldlod=0'), true);
});

test('disabled boundary fade stays clear outside the measured bounds', () => {
  assert.equal(xrVignetteAmount(1, 0, 0), 0);
  assert.equal(xrVignetteAmount(10, 0, 0), 0);
  assert.equal(xrVignetteAmount(1, 0, -1), 0);
});

test('teleport blink survives disabled fade and combines with boundary fade', () => {
  assert.equal(xrVignetteAmount(1, 0.7, 0), 0.7);
  assert.equal(xrVignetteAmount(0, 1, 0), 1);
  assert.equal(xrVignetteAmount(0.5, 0.7, 1), 0.7);
  assert.equal(xrVignetteAmount(1, 0.7, 1), 1);
});

test('fade scales the edge before smoothing and clamps the resulting amount', () => {
  assert.equal(xrVignetteAmount(0.5, 0, 1), 0.5);
  assert.equal(xrVignetteAmount(0.5, 0, 0.5), 0.15625);
  assert.equal(xrVignetteAmount(0.5, 0, 2), 1);
  assert.equal(xrVignetteAmount(-1, -1, 1), 0);
  assert.equal(xrVignetteAmount(2, 0, 1), 1);
  assert.equal(xrVignetteAmount(0, 2, 1), 1);
});

test('nonfinite inputs use a clear edge and blink and normal fade strength', () => {
  for (const value of [NaN, Infinity, -Infinity]) {
    assert.equal(xrVignetteAmount(value, 0, 1), 0);
    assert.equal(xrVignetteAmount(0, value, 1), 0);
    assert.equal(xrVignetteAmount(0.5, 0, value), 0.5);
  }
});
