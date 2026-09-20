import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

const UP = new THREE.Vector3(0, 1, 0);
const finite = (v: THREE.Vector3) =>
  Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.z);
const angleDelta = (a: number, b: number) => Math.atan2(Math.sin(a - b), Math.cos(a - b));

/** Estimated two-bone pose with fixed limb lengths, including unreachable/overlapping targets. */
export function solveBodyLimb(
  root: THREE.Vector3,
  target: THREE.Vector3,
  pole: THREE.Vector3,
  upper: number,
  lower: number,
) {
  const direction = target.clone().sub(root);
  const requested = direction.length();
  if (requested < 1e-8) direction.set(0, -1, 0);
  else direction.divideScalar(requested);
  const distance = THREE.MathUtils.clamp(
    requested,
    Math.abs(upper - lower) + 1e-5,
    upper + lower - 1e-5,
  );
  const along = (upper * upper - lower * lower + distance * distance) / (2 * distance);
  const bend = pole.clone().addScaledVector(direction, -pole.dot(direction));
  if (bend.lengthSq() < 1e-8) {
    bend.set(Math.abs(direction.x) < 0.8 ? 1 : 0, Math.abs(direction.x) < 0.8 ? 0 : 1, 0);
    bend.addScaledVector(direction, -bend.dot(direction));
  }
  bend.normalize();
  return {
    joint: root
      .clone()
      .addScaledVector(direction, along)
      .addScaledVector(bend, Math.sqrt(Math.max(0, upper * upper - along * along))),
    end: root.clone().addScaledVector(direction, distance),
  };
}

export type AvatarBodyPose = {
  headLocal: THREE.Vector3;
  headQuaternion: THREE.Quaternion;
  floorY: number;
  dt: number;
  /** Artificial rig travel this frame, already expressed in rig-local metres. */
  travel: THREE.Vector3;
};

function bodyMaterial(color: number) {
  return new THREE.ShaderMaterial({
    uniforms: { bodyColor: { value: new THREE.Color(color) } },
    toneMapped: false,
    vertexShader: `
      varying vec3 bodyNormal;
      #include <common>
      void main() {
        #include <beginnormal_vertex>
        #include <defaultnormal_vertex>
        bodyNormal = normalize(transformedNormal);
        #include <begin_vertex>
        #include <project_vertex>
      }
    `,
    fragmentShader: `
      uniform vec3 bodyColor;
      varying vec3 bodyNormal;
      void main() {
        vec3 n = normalize(bodyNormal);
        float key = max(0.0, dot(n, normalize(vec3(-0.4, 0.8, 0.9))));
        float fill = max(0.0, dot(n, normalize(vec3(0.6, -0.2, 0.4))));
        gl_FragColor = vec4(bodyColor * (0.42 + 0.45 * key + 0.13 * fill), 1.0);
        #include <colorspace_fragment>
      }
    `,
  });
}

/** Illustrative body inferred from head/hands, not tracked legs or a measured body reconstruction. */
export function createAvatarBody(renderer: THREE.WebGLRenderer, rig: THREE.Group) {
  const group = new THREE.Group();
  group.name = 'avatar-body';
  group.visible = false;
  rig.add(group);
  const geometry = {
    block: new RoundedBoxGeometry(1, 1, 1, 2, 0.16),
    limb: new THREE.CylinderGeometry(0.88, 1, 1, 10),
    joint: new THREE.SphereGeometry(1, 10, 6),
  };
  const materials = {
    shirt: bodyMaterial(0x34545c),
    trousers: bodyMaterial(0x26333b),
    cuff: bodyMaterial(0x567b80),
    sole: bodyMaterial(0xe8dfcc),
  };
  const state = {
    visible: false,
    estimatedPose: true as const,
    moving: false,
    steps: 0,
    parts: 0,
    floorY: 0,
    feet: { left: [0, 0, 0], right: [0, 0, 0] },
    footLift: { left: 0, right: 0 },
    arms: { left: 'rest', right: 'rest' },
  };
  let active = renderer.xr.isPresenting;
  let disposed = false;
  let initialized = false;
  let calibrated = false;
  let bodyYaw = 0;
  let size = 1;
  let previousFloor = 0;
  let nextFoot = 0;
  const previousHead = new THREE.Vector3();
  const motion = new THREE.Vector3();
  const center = new THREE.Vector3();
  const forward = new THREE.Vector3();
  const bodyQ = new THREE.Quaternion();

  function mesh(name: string, shape: keyof typeof geometry, material: keyof typeof materials) {
    const part = new THREE.Mesh(geometry[shape], materials[material]);
    part.name = `avatar-body-${name}`;
    group.add(part);
    state.parts++;
    return part;
  }
  const torso = mesh('torso', 'block', 'shirt');
  const pelvis = mesh('pelvis', 'block', 'trousers');
  const sides = (['left', 'right'] as const).map((side, index) => ({
    side,
    sign: index === 0 ? -1 : 1,
    upperArm: mesh(`${side}-upper-arm`, 'limb', 'shirt'),
    forearm: mesh(`${side}-forearm`, 'limb', 'shirt'),
    elbow: mesh(`${side}-elbow`, 'joint', 'shirt'),
    cuff: mesh(`${side}-cuff`, 'joint', 'cuff'),
    thigh: mesh(`${side}-thigh`, 'limb', 'trousers'),
    shin: mesh(`${side}-shin`, 'limb', 'trousers'),
    knee: mesh(`${side}-knee`, 'joint', 'trousers'),
    shoe: mesh(`${side}-shoe`, 'block', 'shirt'),
    sole: mesh(`${side}-sole`, 'block', 'sole'),
    foot: new THREE.Vector3(),
    start: new THREE.Vector3(),
    target: new THREE.Vector3(),
    desired: new THREE.Vector3(),
    progress: -1,
    yaw: 0,
    startYaw: 0,
  }));

  // Read Three's reference-space poses directly. The hands renderer owns attachment of these
  // spaces to the rig; the body never changes runtime position, orientation or visibility.
  const inputs = [0, 1].map((index) => {
    const grip = renderer.xr.getControllerGrip(index);
    const hand = renderer.xr.getHand(index);
    const controllerWrist = new THREE.Vector3();
    let source: XRInputSource | null = null;
    const connected = (event: { data: XRInputSource }) => {
      source = event.data;
    };
    const disconnected = () => {
      source = null;
    };
    grip.addEventListener('connected', connected);
    grip.addEventListener('disconnected', disconnected);
    return {
      target(side: string) {
        if (source?.handedness !== side) return null;
        const wrist = hand.joints.wrist;
        if (source.hand && hand.visible && wrist?.visible && finite(wrist.position))
          return { position: wrist.position, mode: 'joints' };
        if (!source.hand && grip.visible && finite(grip.position)) {
          // Meet the cuff of the procedural controller glove rather than the handle center.
          controllerWrist
            .set(side === 'left' ? -0.012 : 0.012, 0, 0.057)
            .applyQuaternion(grip.quaternion)
            .add(grip.position);
          return { position: controllerWrist, mode: 'controller' };
        }
        return null;
      },
      dispose() {
        grip.removeEventListener('connected', connected);
        grip.removeEventListener('disconnected', disconnected);
      },
    };
  });

  function reset() {
    initialized = false;
    group.visible = state.visible = state.moving = false;
    state.steps = 0;
    state.footLift.left = state.footLift.right = 0;
    state.arms.left = state.arms.right = 'rest';
    for (const side of sides) side.progress = -1;
    nextFoot = 0;
  }
  function setSessionActive(value: boolean) {
    if (value === active) return;
    active = value;
    reset();
  }
  const start = () => {
    calibrated = false;
    setSessionActive(true);
    reset();
  };
  const end = () => setSessionActive(false);
  renderer.xr.addEventListener('sessionstart', start);
  renderer.xr.addEventListener('sessionend', end);

  const direction = new THREE.Vector3();
  function segment(part: THREE.Mesh, a: THREE.Vector3, b: THREE.Vector3, radius: number) {
    direction.subVectors(b, a);
    const length = direction.length();
    part.position.copy(a).add(b).multiplyScalar(0.5);
    part.quaternion.setFromUnitVectors(UP, direction.multiplyScalar(1 / Math.max(length, 1e-8)));
    part.scale.set(radius, length, radius);
  }
  function local(x: number, y: number, z: number) {
    return new THREE.Vector3(x * size, y, z * size).applyQuaternion(bodyQ).add(center);
  }

  function update({ headLocal, headQuaternion, floorY, dt, travel }: AvatarBodyPose) {
    if (disposed || !active) return;
    if (
      !finite(headLocal) ||
      !finite(travel) ||
      !Number.isFinite(floorY) ||
      !Number.isFinite(headQuaternion.x) ||
      !Number.isFinite(headQuaternion.y) ||
      !Number.isFinite(headQuaternion.z) ||
      !Number.isFinite(headQuaternion.w)
    ) {
      reset();
      return;
    }
    const elapsed = Number.isFinite(dt) ? THREE.MathUtils.clamp(dt, 0, 0.05) : 0;
    forward.set(0, 0, -1).applyQuaternion(headQuaternion);
    const yaw =
      Math.hypot(forward.x, forward.z) > 0.05 ? Math.atan2(-forward.x, -forward.z) : bodyYaw;
    motion.copy(headLocal).sub(previousHead).add(travel);
    motion.y = 0;
    if (
      initialized &&
      (motion.length() > 0.75 * size || Math.abs(floorY - previousFloor) > 0.3 * size)
    ) {
      reset();
    }
    const first = !initialized;
    if (first) {
      // Recenter/teleport/tracking gaps replant the feet without resizing a crouching viewer.
      if (!calibrated) {
        size = THREE.MathUtils.clamp((headLocal.y - floorY) / 1.65, 0.75, 1.25);
        calibrated = true;
      }
      bodyYaw = yaw;
      motion.set(0, 0, 0);
    } else bodyYaw += angleDelta(yaw, bodyYaw) * (1 - Math.exp(-5 * elapsed));
    bodyQ.setFromAxisAngle(UP, bodyYaw);
    // Eyes sit ahead of the torso. Keep all geometry below the neck and out of forward vision.
    center
      .set(0, 0, 0.14 * size)
      .applyQuaternion(bodyQ)
      .add(headLocal);
    center.y = 0;
    const ankleY = floorY + 0.075 * size;
    for (const side of sides) {
      side.desired.copy(local(side.sign * 0.105, ankleY, -0.15));
      if (first) {
        side.foot.copy(side.desired);
        side.yaw = bodyYaw;
      } else {
        // Compensate artificial rig movement so each planted foot stays fixed in world space.
        for (const point of [side.foot, side.start, side.target]) {
          point.x -= travel.x;
          point.z -= travel.z;
        }
      }
    }
    initialized = true;
    previousHead.copy(headLocal);
    previousFloor = floorY;
    state.floorY = floorY;

    if (!sides.some((side) => side.progress >= 0)) {
      const error = sides.map((side) =>
        Math.hypot(side.desired.x - side.foot.x, side.desired.z - side.foot.z),
      );
      if (Math.max(...error) > 0.14 * size) {
        const index = error[nextFoot] > 0.08 * size ? nextFoot : 1 - nextFoot;
        const side = sides[index];
        side.start.copy(side.foot);
        side.target.copy(side.desired);
        if (motion.lengthSq() > 1e-8)
          side.target.addScaledVector(motion.clone().normalize(), 0.09 * size);
        side.startYaw = side.yaw;
        side.progress = 0;
        nextFoot = 1 - index;
      }
    }
    for (const side of sides) {
      let lift = 0;
      if (side.progress >= 0) {
        side.progress = Math.min(1, side.progress + elapsed / 0.3);
        const blend = THREE.MathUtils.smoothstep(side.progress, 0, 1);
        side.foot.copy(side.start).lerp(side.target, blend);
        side.yaw = side.startYaw + angleDelta(bodyYaw, side.startYaw) * blend;
        lift = Math.sin(Math.PI * side.progress) * 0.055 * size;
        if (side.progress >= 1) {
          side.progress = -1;
          state.steps++;
        }
      }
      side.foot.y = ankleY + lift;
      side.foot.toArray(state.feet[side.side]);
      state.footLift[side.side] = lift;
    }
    state.moving = sides.some((side) => side.progress >= 0) || motion.lengthSq() > 1e-6;

    const hipY = THREE.MathUtils.clamp(
      headLocal.y - 0.68 * size,
      floorY + 0.18 * size,
      floorY + 0.96 * size,
    );
    const shoulderY = Math.max(hipY + 0.16 * size, headLocal.y - 0.24 * size);
    torso.position.copy(center).setY((shoulderY + hipY) / 2);
    torso.quaternion.copy(bodyQ);
    torso.scale.set(0.37 * size, shoulderY - hipY, 0.17 * size);
    pelvis.position.copy(center).setY(hipY);
    pelvis.quaternion.copy(bodyQ);
    pelvis.scale.set(0.3 * size, 0.16 * size, 0.19 * size);

    for (const side of sides) {
      const shoulder = local(side.sign * 0.205, shoulderY - 0.035 * size, 0);
      const tracked = inputs.map((input) => input.target(side.side)).find(Boolean);
      state.arms[side.side] = tracked?.mode ?? 'rest';
      const wrist = tracked?.position ?? local(side.sign * 0.23, hipY + 0.04 * size, -0.025);
      const elbowPole = new THREE.Vector3(side.sign * 0.5, -0.2, 0.65).applyQuaternion(bodyQ);
      const arm = solveBodyLimb(shoulder, wrist, elbowPole, 0.29 * size, 0.27 * size);
      segment(side.upperArm, shoulder, arm.joint, 0.062 * size);
      segment(side.forearm, arm.joint, arm.end, 0.05 * size);
      side.elbow.position.copy(arm.joint);
      side.elbow.scale.setScalar(0.054 * size);
      side.cuff.position.copy(arm.end);
      side.cuff.scale.setScalar(0.043 * size);

      const hip = local(side.sign * 0.095, hipY, 0);
      const kneePole = new THREE.Vector3(0, 0, -1).applyQuaternion(bodyQ);
      const leg = solveBodyLimb(hip, side.foot, kneePole, 0.47 * size, 0.455 * size);
      segment(side.thigh, hip, leg.joint, 0.083 * size);
      segment(side.shin, leg.joint, leg.end, 0.063 * size);
      side.knee.position.copy(leg.joint);
      side.knee.scale.setScalar(0.065 * size);
      side.shoe.quaternion.setFromAxisAngle(UP, side.yaw);
      side.sole.quaternion.copy(side.shoe.quaternion);
      side.shoe.position
        .set(0, -0.025 * size, -0.04 * size)
        .applyQuaternion(side.shoe.quaternion)
        .add(side.foot);
      side.sole.position
        .copy(side.shoe.position)
        .setY(floorY + state.footLift[side.side] + 0.0125 * size);
      side.shoe.scale.set(0.115 * size, 0.075 * size, 0.245 * size);
      side.sole.scale.set(0.12 * size, 0.025 * size, 0.25 * size);
    }
    group.visible = state.visible = true;
  }

  return {
    group,
    state,
    reset,
    setSessionActive,
    update,
    dispose() {
      if (disposed) return;
      disposed = true;
      reset();
      renderer.xr.removeEventListener('sessionstart', start);
      renderer.xr.removeEventListener('sessionend', end);
      inputs.forEach((input) => input.dispose());
      group.removeFromParent();
      Object.values(geometry).forEach((item) => item.dispose());
      Object.values(materials).forEach((item) => item.dispose());
    },
  };
}
