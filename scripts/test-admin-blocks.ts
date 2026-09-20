import assert from 'node:assert/strict';
import test from 'node:test';
import { classify, extensionOf, leadArtifact, plyFlavour } from '../admin/lib/blocks.ts';
import type { RunArtifact } from '../admin/lib/types.ts';

function artifact(name: string, mediaType: string, size = 1024): RunArtifact {
  return {
    id: name,
    role: 'output',
    sha256: 'a'.repeat(64),
    size,
    mediaType,
    attemptId: 'attempt-1',
    previewArtifactId: null,
    name,
  };
}

test('the file name decides where the media type cannot', () => {
  // Every 3D output leaves the pipeline as an opaque octet-stream.
  const opaque = 'application/octet-stream';
  assert.equal(classify(artifact('world.spz', opaque)).type, 'splat');
  // `.ply` is ambiguous in this pipeline, so it gets a block that reads the header.
  assert.equal(classify(artifact('outputs/shape/object.ply', opaque)).type, 'points');
  assert.equal(classify(artifact('frame_000.ply', opaque)).type, 'points');
  assert.equal(classify(artifact('masks.npz', opaque)).type, 'tensor');
  assert.equal(classify(artifact('scene.glb', opaque)).type, 'mesh');
  assert.equal(classify(artifact('source.bin', opaque)).type, 'binary');
});

test('media types classify the things a browser can already draw', () => {
  assert.equal(classify(artifact('world-thumb.png', 'image/png')).type, 'image');
  assert.equal(classify(artifact('clean.mp4', 'video/mp4')).type, 'video');
  assert.equal(classify(artifact('speech.wav', 'audio/wav')).type, 'audio');
  assert.equal(classify(artifact('shots.json', 'application/json')).type, 'data');
  assert.equal(classify(artifact('log.txt', 'text/plain; charset=utf-8')).type, 'text');
});

test('readable files the API downgrades to octet-stream still read as text', () => {
  // Only a short allowlist is served inline, so a stage log leaves the API looking binary.
  const opaque = 'application/octet-stream';
  assert.equal(classify(artifact('stage.log', opaque)).type, 'text');
  assert.equal(classify(artifact('stage.log', opaque)).noun, 'log');
  assert.equal(classify(artifact('notes.md', opaque)).type, 'text');
  assert.equal(classify(artifact('rows.csv', opaque)).type, 'text');
  assert.equal(classify(artifact('workspace.tar.gz', 'application/x-tar')).type, 'archive');
});

test('SVG is markup, not a picture, so it is never drawn inline', () => {
  assert.equal(classify(artifact('diagram.svg', 'image/svg+xml')).type, 'binary');
});

test('a PLY header, not its extension, says which renderer it needs', () => {
  // The shape stages write Gaussians.
  const gaussian = [
    'ply',
    'format binary_little_endian 1.0',
    'element vertex 4',
    ...['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity'].map((p) => `property float ${p}`),
    ...['scale_0', 'scale_1', 'scale_2'].map((p) => `property float ${p}`),
    ...['rot_0', 'rot_1', 'rot_2', 'rot_3'].map((p) => `property float ${p}`),
    'end_header',
  ].join('\n');
  assert.equal(plyFlavour(gaussian), 'gaussian');

  // The person frames this pipeline actually wrote: a bare coloured point cloud.
  const cloud = [
    'ply',
    'format binary_little_endian 1.0',
    'element vertex 23873',
    ...['x', 'y', 'z'].map((p) => `property float ${p}`),
    ...['red', 'green', 'blue'].map((p) => `property uchar ${p}`),
    'end_header',
  ].join('\n');
  assert.equal(plyFlavour(cloud), 'points');

  const mesh = [
    'ply',
    'element vertex 8',
    'property float x',
    'element face 12',
    'property list uchar int vertex_index',
    'end_header',
  ].join('\n');
  assert.equal(plyFlavour(mesh), 'mesh');

  // A face element declared empty is still a point cloud.
  assert.equal(plyFlavour(mesh.replace('element face 12', 'element face 0')), 'points');
});

test('only the 3D blocks claim a canvas', () => {
  const canvas = ['world.spz', 'object.ply', 'scene.glb', 'scan.stl'];
  for (const name of canvas) {
    assert.equal(classify(artifact(name, 'application/octet-stream')).canvas, true, name);
  }
  assert.equal(classify(artifact('frame.png', 'image/png')).canvas, false);
  assert.equal(classify(artifact('report.json', 'application/json')).canvas, false);
});

test('every block carries a noun and a mark an operator can read', () => {
  for (const sample of [
    artifact('world.spz', 'application/octet-stream'),
    artifact('frame.png', 'image/png'),
    artifact('mystery', 'application/x-thing'),
  ]) {
    const kind = classify(sample);
    assert.ok(kind.noun.length > 0 && kind.mark.length > 0);
  }
});

test('a stage leads with the thing an operator judges by eye', () => {
  const outputs = [
    artifact('receipt.json', 'application/json'),
    artifact('world-thumb.png', 'image/png'),
    artifact('world.spz', 'application/octet-stream', 40_000_000),
  ];
  assert.equal(leadArtifact(outputs)?.name, 'world.spz');
  // With no world, the still beats the report.
  assert.equal(leadArtifact(outputs.slice(0, 2))?.name, 'world-thumb.png');
  assert.equal(leadArtifact([]), undefined);
});

test('extensions come from the written file name, not the role', () => {
  assert.equal(extensionOf(artifact('outputs/finetuned.spz', 'application/octet-stream')), 'spz');
  assert.equal(extensionOf(artifact('LICENSE', 'text/plain')), '');
  const unnamed = { ...artifact('x', 'text/plain'), name: undefined };
  assert.equal(extensionOf(unnamed), '');
});
