import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';
import { createAvatarBody, solveBodyLimb } from '../src/xr/avatar-body';

function fixture() {
  const grips = [new THREE.Group(), new THREE.Group()];
  const hands = [0, 1].map(() =>
    Object.assign(new THREE.Group(), { joints: {} as Record<string, THREE.Group> }),
  );
  const xr = Object.assign(new THREE.EventDispatcher(), {
    isPresenting: false,
    getControllerGrip: (index: number) => grips[index],
    getHand: (index: number) => hands[index],
  });
  const rig = new THREE.Group();
  const body = createAvatarBody({ xr } as unknown as THREE.WebGLRenderer, rig);
  const pose = {
    headLocal: new THREE.Vector3(0, 1.65, 0),
    headQuaternion: new THREE.Quaternion(),
    floorY: 0,
    dt: 1 / 72,
    travel: new THREE.Vector3(),
  };
  return { body, xr, grips, hands, pose };
}

describe('estimated XR body', () => {
  test('limb solver preserves lengths for overlapping, reachable and unreachable targets', () => {
    const root = new THREE.Vector3(0.2, 1.4, -0.1);
    for (const offset of [
      new THREE.Vector3(),
      new THREE.Vector3(0.1, -0.3, 0.1),
      new THREE.Vector3(3, 0, 0),
    ]) {
      const limb = solveBodyLimb(
        root,
        root.clone().add(offset),
        new THREE.Vector3(0, -1, 0),
        0.29,
        0.27,
      );
      expect(limb.joint.distanceTo(root)).toBeCloseTo(0.29, 10);
      expect(limb.end.distanceTo(limb.joint)).toBeCloseTo(0.27, 10);
      expect(limb.end.toArray().every(Number.isFinite)).toBe(true);
    }
  });

  test('planted feet compensate rig translation and sustained travel produces steps', () => {
    const { body, pose } = fixture();
    body.setSessionActive(true);
    body.update(pose);
    const foot = [...body.state.feet.left];
    pose.travel.x = 0.01;
    body.update(pose);
    expect(body.state.feet.left[0] + pose.travel.x).toBeCloseTo(foot[0], 12);
    for (let i = 0; i < 100; i++) body.update(pose);
    expect(body.state.steps).toBeGreaterThan(0);
    expect(body.state.moving).toBe(true);
    body.dispose();
  });

  test('tracks controller and wrist targets without a hands renderer', () => {
    const { body, grips, hands, pose } = fixture();
    body.setSessionActive(true);
    grips[0].position.set(-0.25, 1.1, -0.3);
    grips[0].dispatchEvent({ type: 'connected', data: { handedness: 'left' } } as never);
    body.update(pose);
    expect(body.state.arms.left).toBe('controller');
    grips[0].dispatchEvent({ type: 'disconnected' } as never);
    const wrist = new THREE.Group();
    wrist.position.set(0.25, 1.1, -0.3);
    hands[1].joints.wrist = wrist;
    grips[1].dispatchEvent({ type: 'connected', data: { handedness: 'right', hand: {} } } as never);
    body.update(pose);
    expect(body.state.arms.left).toBe('rest');
    expect(body.state.arms.right).toBe('joints');
    wrist.visible = false;
    body.update(pose);
    expect(body.state.arms.right).toBe('rest');
    body.dispose();
  });

  test('local fallback floor, crouching, reset and session re-entry keep valid geometry', () => {
    const { body, xr, pose } = fixture();
    pose.headLocal.y = 0;
    pose.floorY = -1.65;
    xr.dispatchEvent({ type: 'sessionstart' } as never);
    body.update(pose);
    expect(body.state.visible).toBe(true);
    expect(body.state.feet.left[1]).toBeCloseTo(-1.575, 10);
    const torso = body.group.getObjectByName('avatar-body-torso')!;
    const standingY = torso.position.y;
    const standingWidth = torso.scale.x;
    pose.headLocal.y -= 0.5;
    body.update(pose);
    expect(torso.position.y).toBeLessThan(standingY);
    for (const part of body.group.children) {
      expect(part.position.toArray().every(Number.isFinite)).toBe(true);
      expect(part.scale.toArray().every((value) => Number.isFinite(value) && value > 0)).toBe(true);
    }
    body.reset();
    expect(body.state.visible).toBe(false);
    body.update(pose);
    expect(body.state.visible).toBe(true);
    expect(torso.scale.x).toBe(standingWidth);
    xr.dispatchEvent({ type: 'sessionend' } as never);
    body.update(pose);
    expect(body.state.visible).toBe(false);
    xr.dispatchEvent({ type: 'sessionstart' } as never);
    body.update(pose);
    expect(body.state.visible).toBe(true);
    body.dispose();
    expect(body.group.parent).toBe(null);
  });
});
