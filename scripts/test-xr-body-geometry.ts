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
  rig.scale.setScalar(3.7);
  const body = createAvatarBody({ xr } as unknown as THREE.WebGLRenderer, rig);
  const pose = {
    headLocal: new THREE.Vector3(2, 1.65, -3),
    headQuaternion: new THREE.Quaternion(),
    floorY: 0,
    dt: 1 / 72,
    travel: new THREE.Vector3(),
  };
  const update = () => body.update(pose);
  const connect = (index: number, source: Partial<XRInputSource>) =>
    grips[index].dispatchEvent({ type: 'connected', data: source } as never);
  return { body, pose, update, grips, hands, connect, rig };
}

describe('estimated XR body', () => {
  test('fixed-length two-bone limbs remain finite for reachable, far and coincident targets', () => {
    const root = new THREE.Vector3(0.2, 1.3, 0);
    for (const target of [
      new THREE.Vector3(0.3, 1, -0.3),
      new THREE.Vector3(100, 100, 100),
      root,
    ]) {
      const solved = solveBodyLimb(root, target, new THREE.Vector3(0, -1, 0), 0.29, 0.27);
      expect(solved.joint.distanceTo(root)).toBeCloseTo(0.29, 8);
      expect(solved.end.distanceTo(solved.joint)).toBeCloseTo(0.27, 8);
      expect(solved.end.distanceTo(root)).toBeLessThan(0.560001);
      expect([...solved.joint.toArray(), ...solved.end.toArray()].every(Number.isFinite)).toBe(
        true,
      );
    }
  });

  test('first valid pose appears beneath viewer with grounded shoes, no head and no scale changes', () => {
    const { body, update, rig } = fixture();
    update();
    expect(body.state.visible).toBe(false);
    body.setSessionActive(true);
    update();
    expect(body.state.estimatedPose).toBe(true);
    expect(body.state.visible).toBe(true);
    expect(body.state.parts).toBe(20);
    expect(body.state.steps).toBe(0);
    expect(body.state.moving).toBe(false);
    expect(body.group.getObjectByName('avatar-body-head')).toBeUndefined();
    expect(body.state.feet.left[0]).toBeCloseTo(1.895);
    expect(body.state.feet.right[0]).toBeCloseTo(2.105);
    for (const side of ['left', 'right']) {
      const sole = body.group.getObjectByName(`avatar-body-${side}-sole`)!;
      expect(sole.position.y - sole.scale.y / 2).toBeCloseTo(0, 8);
    }
    expect(rig.scale.toArray()).toEqual([3.7, 3.7, 3.7]);
    body.dispose();
  });

  test('stationary poses and repeated activation do not slide feet or restart gait', () => {
    const { body, update } = fixture();
    body.setSessionActive(true);
    update();
    const feet = structuredClone(body.state.feet);
    for (let i = 0; i < 180; i++) {
      body.setSessionActive(true);
      update();
    }
    expect(body.state.feet).toEqual(feet);
    expect(body.state.steps).toBe(0);
    expect(body.state.moving).toBe(false);
    body.dispose();
  });

  test('torso stays behind the eye and leaves clear eye-to-shoe sight lines when looking down', () => {
    for (const yaw of [0, 0.8]) {
      const { body, pose, update, rig } = fixture();
      pose.headQuaternion.setFromEuler(new THREE.Euler(-1.4, yaw, 0, 'YXZ'));
      body.setSessionActive(true);
      update();
      rig.updateMatrixWorld(true);
      const torso = body.group.getObjectByName('avatar-body-torso')!;
      const inverse = torso.matrixWorld.clone().invert();
      const eye = rig.localToWorld(pose.headLocal.clone()).applyMatrix4(inverse);
      const chestBox = new THREE.Box3(
        new THREE.Vector3(-0.5, -0.5, -0.5),
        new THREE.Vector3(0.5, 0.5, 0.5),
      );
      expect(eye.z).toBeLessThan(chestBox.min.z);
      for (const side of ['left', 'right']) {
        const shoe = body.group.getObjectByName(`avatar-body-${side}-shoe`)!;
        const shoeCenter = shoe.getWorldPosition(new THREE.Vector3()).applyMatrix4(inverse);
        const sightLine = new THREE.Ray(eye, shoeCenter.sub(eye).normalize());
        expect(sightLine.intersectBox(chestBox, new THREE.Vector3())).toBeNull();
      }
      body.dispose();
    }
  });

  test('physical walking creates alternating steps and settles without endless foot drift', () => {
    const { body, pose, update } = fixture();
    body.setSessionActive(true);
    update();
    const lifts = { left: 0, right: 0 };
    for (let i = 0; i < 140; i++) {
      pose.headLocal.z -= 0.008;
      update();
      for (const side of ['left', 'right'] as const)
        lifts[side] = Math.max(lifts[side], body.state.footLift[side]);
      expect(body.state.footLift.left === 0 || body.state.footLift.right === 0).toBe(true);
    }
    expect(body.state.steps).toBeGreaterThan(2);
    expect(lifts.left).toBeGreaterThan(0.03);
    expect(lifts.right).toBeGreaterThan(0.03);
    for (let i = 0; i < 150; i++) update();
    const planted = structuredClone(body.state.feet);
    const steps = body.state.steps;
    for (let i = 0; i < 150; i++) update();
    expect(body.state.feet).toEqual(planted);
    expect(body.state.steps).toBe(steps);
    expect(body.state.moving).toBe(false);
    body.dispose();
  });

  test('joystick travel compensates planted foot world position and drives gait once', () => {
    const { body, pose, update } = fixture();
    body.setSessionActive(true);
    update();
    const foot = [...body.state.feet.left];
    pose.travel.set(0.01, 0, -0.02);
    update();
    expect(body.state.feet.left[0] + pose.travel.x).toBeCloseTo(foot[0]);
    expect(body.state.feet.left[2] + pose.travel.z).toBeCloseTo(foot[2]);
    for (let i = 0; i < 80; i++) update();
    expect(body.state.steps).toBeGreaterThan(1);
    expect(body.state.moving).toBe(true);
    body.dispose();
  });

  test('crouching lowers torso and bends knees while feet remain planted on supplied floor', () => {
    const { body, pose, update } = fixture();
    pose.floorY = -0.4;
    pose.headLocal.y = 1.25;
    body.setSessionActive(true);
    update();
    const feet = structuredClone(body.state.feet);
    const pelvis = body.group.getObjectByName('avatar-body-pelvis')!;
    const height = pelvis.position.y;
    pose.headLocal.y -= 0.45;
    update();
    expect(pelvis.position.y).toBeLessThan(height - 0.3);
    expect(body.state.feet).toEqual(feet);
    const knee = body.group.getObjectByName('avatar-body-left-knee')!;
    expect(knee.position.z).toBeLessThan(feet.left[2] - 0.1);
    const torso = body.group.getObjectByName('avatar-body-torso')!;
    expect(torso.position.y + torso.scale.y / 2).toBeLessThan(pose.headLocal.y - 0.2);
    const width = torso.scale.x;
    body.reset();
    update();
    expect(torso.scale.x).toBe(width);
    expect(body.state.feet).toEqual(feet);
    body.dispose();
  });

  test('arms use handedness and visible wrist/grip poses without changing runtime transforms', () => {
    const { body, pose, update, grips, hands, connect } = fixture();
    body.setSessionActive(true);
    connect(0, { handedness: 'right' });
    grips[0].position.copy(pose.headLocal).add(new THREE.Vector3(0.2, -0.45, -0.35));
    const original = grips[0].position.toArray();
    update();
    expect(body.state.arms.right).toBe('controller');
    expect(body.state.arms.left).toBe('rest');
    expect(grips[0].position.toArray()).toEqual(original);
    expect(grips[0].parent).toBeNull();
    connect(1, { handedness: 'left', hand: {} as XRHand });
    hands[1].joints.wrist = new THREE.Group();
    hands[1].joints.wrist.position.copy(pose.headLocal).add(new THREE.Vector3(-0.2, -0.5, -0.3));
    update();
    expect(body.state.arms.left).toBe('joints');
    const wrist = hands[1].joints.wrist.position.toArray();
    hands[1].joints.wrist.visible = false;
    grips[0].visible = false;
    update();
    expect(body.state.arms).toEqual({ left: 'rest', right: 'rest' });
    expect(grips[0].visible).toBe(false);
    expect(hands[1].joints.wrist.position.toArray()).toEqual(wrist);
    body.dispose();
  });

  test('reset, invalid pose, re-entry and large jumps rebase without starting a remote step', () => {
    const { body, pose, update } = fixture();
    body.setSessionActive(true);
    update();
    pose.headLocal.x += 20;
    pose.dt = 100;
    update();
    expect(body.state.steps).toBe(0);
    expect(body.state.moving).toBe(false);
    expect(body.state.feet.left[0]).toBeCloseTo(pose.headLocal.x - 0.105);
    body.reset();
    expect(body.state.visible).toBe(false);
    update();
    pose.headLocal.x = NaN;
    update();
    expect(body.state.visible).toBe(false);
    pose.headLocal.x = 1;
    update();
    expect(body.state.visible).toBe(true);
    body.setSessionActive(false);
    update();
    expect(body.state.visible).toBe(false);
    body.setSessionActive(true);
    update();
    expect(body.state.steps).toBe(0);
    body.dispose();
    body.dispose();
    expect(body.group.parent).toBeNull();
    update();
    expect(body.state.visible).toBe(false);
  });
});
