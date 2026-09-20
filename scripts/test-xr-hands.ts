import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';
import { createAvatarHands } from '../src/xr/avatar-hands';

function fixture() {
  const grips = [new THREE.Group(), new THREE.Group()];
  const hands = [0, 1].map(() =>
    Object.assign(new THREE.Group(), { joints: {} as Record<string, THREE.Group> }),
  );
  const xr = Object.assign(new THREE.EventDispatcher(), {
    isPresenting: false,
    getControllerGrip: (i: number) => grips[i],
    getHand: (i: number) => hands[i],
  });
  const rig = new THREE.Group();
  rig.scale.setScalar(3.7);
  const avatar = createAvatarHands({ xr } as unknown as THREE.WebGLRenderer, rig);
  function connect(index: number, source: Partial<XRInputSource>) {
    // Simulate Three's connection event while retaining real Object3D transforms/meshes.
    grips[index].dispatchEvent({ type: 'connected', data: source } as never);
  }
  return { avatar, grips, hands, rig, connect };
}

describe('XR avatar hands', () => {
  test('follows handedness, curls on input, and leaves runtime visibility and poses untouched', () => {
    const { avatar, grips, connect, rig } = fixture();
    const buttons = [{ value: 0 }, { value: 0 }];
    connect(0, { handedness: 'right', gamepad: { buttons } as unknown as Gamepad });
    grips[0].position.set(0.2, 1.2, -0.4);
    avatar.update();
    expect(avatar.state.visibleCount).toBe(0);
    avatar.setSessionActive(true);
    avatar.update();
    expect(avatar.state.right.mode).toBe('controller');
    expect(avatar.state.left.mode).toBe('hidden');
    const glove = grips[0].children[0];
    const joints = glove.children[0] as THREE.InstancedMesh;
    expect(joints.count).toBe(20);
    const relaxed = new THREE.Matrix4();
    const curled = new THREE.Matrix4();
    joints.getMatrixAt(7, relaxed);
    buttons[0].value = buttons[1].value = 1;
    avatar.update();
    joints.getMatrixAt(7, curled);
    expect(relaxed.equals(curled)).toBe(false);
    expect(avatar.state.right.trigger).toBe(1);
    expect(avatar.state.right.squeeze).toBe(1);
    expect(grips[0].parent).toBe(rig);
    expect(grips[0].position.toArray()).toEqual([0.2, 1.2, -0.4]);
    expect(grips[0].scale.toArray()).toEqual([1, 1, 1]);
    grips[0].visible = false;
    avatar.update();
    expect(avatar.state.visibleCount).toBe(0);
    expect(grips[0].visible).toBe(false);
    grips[0].visible = true;
    avatar.update();
    avatar.setSessionActive(false);
    expect(glove.visible).toBe(false);
    expect(grips[0].visible).toBe(true);
    expect(avatar.state.visibleCount).toBe(0);
    avatar.dispose();
    avatar.dispose();
  });

  test('renders tracked joints in hand space and hides missing/stale poses without controller doubles', () => {
    const { avatar, hands, grips, connect } = fixture();
    avatar.setSessionActive(true);
    connect(0, { handedness: 'left', hand: {} as XRHand });
    avatar.update();
    expect(avatar.state.visibleCount).toBe(0);
    function joint(name: string, x: number, y: number, z: number) {
      const node = new THREE.Group();
      node.position.set(x, y, z);
      hands[0].joints[name] = node;
    }
    joint('wrist', 0, 1.1, -0.2);
    for (const [index, finger] of ['index', 'middle', 'ring', 'pinky'].entries()) {
      const x = 0.027 - index * 0.018;
      joint(`${finger}-finger-phalanx-proximal`, x, 1.1, -0.28);
      joint(`${finger}-finger-phalanx-intermediate`, x, 1.1, -0.31);
      joint(`${finger}-finger-phalanx-distal`, x, 1.1, -0.33);
      joint(`${finger}-finger-tip`, x, 1.1, -0.35);
    }
    avatar.update();
    expect(avatar.state.left.mode).toBe('joints');
    expect(avatar.state.left.joints).toBe(17);
    expect(avatar.state.visibleCount).toBe(1);
    expect(grips[0].children[0].visible).toBe(false);
    const tracked = hands[0].children[0];
    const spheres = tracked.children[0] as THREE.InstancedMesh;
    expect(spheres.count).toBe(16);
    const matrix = new THREE.Matrix4();
    spheres.getMatrixAt(0, matrix);
    expect(
      new THREE.Vector3()
        .setFromMatrixPosition(matrix)
        .distanceTo(new THREE.Vector3(0.027, 1.1, -0.28)),
    ).toBeLessThan(1e-6);
    hands[0].joints['index-finger-tip'].visible = false;
    avatar.update();
    expect(spheres.count).toBe(15);
    hands[0].joints.wrist.visible = false;
    avatar.update();
    expect(tracked.visible).toBe(false);
    expect(hands[0].visible).toBe(true);
    expect(avatar.state.visibleCount).toBe(0);
    hands[0].joints.wrist.visible = true;
    avatar.update();
    grips[0].dispatchEvent({ type: 'disconnected' } as never);
    avatar.update();
    expect(tracked.visible).toBe(false);
    expect(avatar.state.left.mode).toBe('hidden');
    avatar.dispose();
  });
});
