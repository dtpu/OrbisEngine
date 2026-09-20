import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import { BottleVisual } from '../src/interaction/bottle-visual';

function bounds(object: THREE.Object3D) {
  object.updateMatrixWorld(true);
  return new THREE.Box3().setFromObject(object);
}

test('procedural bottle is finite, upright, and constrained by measured dimensions', () => {
  const visual = new BottleVisual(1, new THREE.Vector3(0.06, 0.18, 0.05));
  const box = bounds(visual.proxy);
  const size = box.getSize(new THREE.Vector3());

  assert.ok(box.min.toArray().every(Number.isFinite));
  assert.ok(box.max.toArray().every(Number.isFinite));
  assert.ok(size.y > size.x * 1.8, `expected an upright bottle, got ${size.toArray()}`);
  assert.ok(size.y >= 0.11 && size.y <= 0.13, `unexpected height ${size.y}`);
  assert.ok(size.x < 0.05 && size.z < 0.05, `unexpected width ${size.toArray()}`);
  assert.equal(visual.group.name, 'bottle-interaction-visual');
  assert.equal(visual.proxy.name, 'interaction-bottle-proxy');
  assert.equal(visual.locator.name, 'interaction-bottle-locator');
  assert.deepEqual(
    visual.proxy.children.map((child) => child.name),
    [
      'bottle-proxy-body',
      'bottle-proxy-shoulder',
      'bottle-proxy-neck',
      'bottle-proxy-cap',
      'bottle-proxy-label',
      'bottle-proxy-label-band',
    ],
  );
  visual.dispose();
});

test('fallback is stature-scaled and remains visible beyond a centred grip', () => {
  const small = new BottleVisual(0.8);
  const large = new BottleVisual(1.2);
  const smallHeight = bounds(small.proxy).getSize(new THREE.Vector3()).y;
  const largeHeight = bounds(large.proxy).getSize(new THREE.Vector3()).y;

  assert.ok(smallHeight > 0.07);
  assert.ok(Math.abs(largeHeight / smallHeight - 1.5) < 0.03);
  assert.ok(bounds(small.proxy).max.y > 0.035, 'top must protrude above a grip at origin');
  small.dispose();
  large.dispose();
});

test('controller-scale bottle leaves a visible fist silhouette around the held profile', () => {
  const stature = 0.915278;
  const unitsPerMetre = 0.585282;
  const visual = new BottleVisual(stature, new THREE.Vector3(0.0432, 0.0432, 0.0432));
  const sizeMetres = bounds(visual.proxy).getSize(new THREE.Vector3()).divideScalar(unitsPerMetre);

  // The controller glove palm is 7.5 cm long in its rotated grip axis. Keep the label's outer
  // diameter below 5.5 cm so its finger joints are still visible around a closed controller grip.
  assert.ok(sizeMetres.y >= 0.13 && sizeMetres.y <= 0.19, `unexpected held height ${sizeMetres.y}`);
  assert.ok(sizeMetres.x < 0.055 && sizeMetres.z < 0.055, `unexpected held width ${sizeMetres}`);
  assert.ok(sizeMetres.y > sizeMetres.x * 2.5, `held form became too squat ${sizeMetres}`);
  visual.dispose();
});

test('updates copy world transforms, visibility, and locator pulse without mutating inputs', () => {
  const visual = new BottleVisual(1);
  const scene = new THREE.Scene();
  const sourceParent = new THREE.Group();
  sourceParent.scale.setScalar(3);
  scene.add(sourceParent);
  scene.add(visual.group);
  const position = new THREE.Vector3(2, 1, -3);
  const quaternion = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), 0.6);
  visual.update({
    position,
    quaternion,
    proxyVisible: true,
    locatorVisible: true,
    elapsed: 0,
  });
  const firstOpacity = (
    (visual.locator.children[0] as THREE.Mesh).material as THREE.MeshBasicMaterial
  ).opacity;
  visual.update({
    position,
    quaternion,
    proxyVisible: false,
    locatorVisible: true,
    nearHand: true,
    elapsed: 0.2,
  });
  const secondOpacity = (
    (visual.locator.children[0] as THREE.Mesh).material as THREE.MeshBasicMaterial
  ).opacity;

  assert.deepEqual(visual.group.position.toArray(), [2, 1, -3]);
  assert.ok(visual.group.quaternion.angleTo(quaternion) < 1e-12);
  assert.ok(
    visual.locator.getWorldQuaternion(new THREE.Quaternion()).angleTo(new THREE.Quaternion()) <
      1e-12,
  );
  assert.equal(visual.proxy.visible, false);
  assert.equal(visual.locator.visible, true);
  assert.notEqual(firstOpacity, secondOpacity);
  assert.deepEqual(position.toArray(), [2, 1, -3]);
  assert.ok(quaternion.angleTo(visual.group.quaternion) < 1e-12);
  assert.equal(visual.group.parent, scene);
  assert.notEqual(visual.group.parent, sourceParent);
  visual.dispose();
});

test('disposal releases visual resources and detaches the helper', () => {
  const visual = new BottleVisual(1);
  const scene = new THREE.Scene();
  scene.add(visual.group);
  const disposed: string[] = [];
  visual.group.traverse((object) => {
    const mesh = object as THREE.Mesh;
    mesh.geometry?.addEventListener('dispose', () => disposed.push(`geometry:${object.name}`));
    const material = mesh.material;
    if (!Array.isArray(material))
      material?.addEventListener('dispose', () => disposed.push(`material:${object.name}`));
  });

  visual.dispose();
  visual.dispose();
  assert.equal(visual.group.parent, null);
  assert.equal(visual.group.children.length, 0);
  assert.equal(disposed.length, 16);
});
