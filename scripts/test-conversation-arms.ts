import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';
import type { SplatMesh } from '@sparkjsdev/spark';
import { ConversationSplats, freeConversationArms } from '../src/interaction/conversation-splats';
import type { ConversationPose } from '../src/interaction/conversation-motion';

const pose: ConversationPose = {
  pitch: 0,
  yaw: 0,
  roll: 0,
  breath: 0,
  weight: 1,
  mode: 'speaking',
  leftShoulder: 0.16,
  rightShoulder: 0.16,
  leftElbow: 0.38,
  rightElbow: 0.38,
};

describe('held-prop arm protection', () => {
  test('free hands gesture; each attachment pins its side and a central one pins both', () => {
    expect(freeConversationArms([])).toEqual([1, 1]);
    expect(freeConversationArms([new THREE.Vector3(-0.12, -0.4, 0)])).toEqual([0, 1]);
    expect(freeConversationArms([new THREE.Vector3(0.12, -0.4, 0)])).toEqual([1, 0]);
    expect(freeConversationArms([new THREE.Vector3(0, -0.4, 0)])).toEqual([0, 0]);
    expect(
      freeConversationArms([new THREE.Vector3(-0.12, -0.4, 0), new THREE.Vector3(0.12, -0.4, 0)]),
    ).toEqual([0, 0]);
  });

  test('protection follows body orientation and scale; a released arm blends in and reset clears it', () => {
    for (const yaw of [0, Math.PI / 2, -1.1]) {
      const mesh = Object.assign(new THREE.Object3D(), {
        updateGenerator() {},
        updateVersion() {},
      }) as unknown as SplatMesh;
      const overlay = new ConversationSplats(mesh);
      const head = new THREE.Vector3(3, 2, -5);
      const rotation = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
      const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(rotation);
      const held = new THREE.Vector3(-0.12, -0.4, 0)
        .multiplyScalar(2)
        .applyQuaternion(rotation)
        .add(head);
      for (let i = 0; i < 90; i++) overlay.update(pose, head, forward, 2, [held]);
      expect(overlay.armFreedom[0]).toBe(0);
      expect(overlay.armFreedom[1]).toBeGreaterThan(0.99);
      overlay.update(pose, head, forward, 2, []);
      expect(overlay.armFreedom[0]).toBeGreaterThan(0);
      expect(overlay.armFreedom[0]).toBeLessThan(0.1);
      overlay.update(pose, head, forward, 2, [held]);
      expect(overlay.armFreedom[0]).toBe(0);
      overlay.reset();
      expect(overlay.armFreedom).toEqual([0, 0]);
      overlay.dispose();
      expect(mesh.objectModifiers).toHaveLength(0);
    }
  });
});
