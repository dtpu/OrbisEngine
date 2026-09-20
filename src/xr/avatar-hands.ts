import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

type HandMode = 'hidden' | 'controller' | 'joints';
type HandState = { mode: HandMode; joints: number; trigger: number; squeeze: number };
type Joint = { position: THREE.Vector3; visible: boolean; jointRadius?: number };

const FINGERS: XRHandJoint[][] = [
  ['thumb-metacarpal', 'thumb-phalanx-proximal', 'thumb-phalanx-distal', 'thumb-tip'],
  [
    'index-finger-phalanx-proximal',
    'index-finger-phalanx-intermediate',
    'index-finger-phalanx-distal',
    'index-finger-tip',
  ],
  [
    'middle-finger-phalanx-proximal',
    'middle-finger-phalanx-intermediate',
    'middle-finger-phalanx-distal',
    'middle-finger-tip',
  ],
  [
    'ring-finger-phalanx-proximal',
    'ring-finger-phalanx-intermediate',
    'ring-finger-phalanx-distal',
    'ring-finger-tip',
  ],
  [
    'pinky-finger-phalanx-proximal',
    'pinky-finger-phalanx-intermediate',
    'pinky-finger-phalanx-distal',
    'pinky-finger-tip',
  ],
];

const UP = new THREE.Vector3(0, 1, 0);
const idleState = (): HandState => ({ mode: 'hidden', joints: 0, trigger: 0, squeeze: 0 });

// Splat scenes may contain no Three lights. Soft view-space shading keeps the glove legible
// without adding lights or making its appearance depend on a scene's object-track inventory.
function gloveMaterial(color: number) {
  return new THREE.ShaderMaterial({
    uniforms: { gloveColor: { value: new THREE.Color(color) } },
    toneMapped: false,
    vertexShader: `
      varying vec3 handNormal;
      #include <common>
      void main() {
        #include <beginnormal_vertex>
        #include <defaultnormal_vertex>
        handNormal = normalize(transformedNormal);
        #include <begin_vertex>
        #include <project_vertex>
      }
    `,
    fragmentShader: `
      uniform vec3 gloveColor;
      varying vec3 handNormal;
      void main() {
        vec3 n = normalize(handNormal);
        float key = max(0.0, dot(n, normalize(vec3(-0.4, 0.8, 0.9))));
        float fill = max(0.0, dot(n, normalize(vec3(0.6, -0.2, 0.4))));
        gl_FragColor = vec4(gloveColor * (0.42 + 0.45 * key + 0.13 * fill), 1.0);
        #include <colorspace_fragment>
      }
    `,
  });
}

/** Procedural gloves in XR metres; the parent rig supplies the scene's units-per-metre scale.
 * Call update after Three has updated XR poses, and setSessionActive(false) when leaving XR.
 * Runtime grip/hand/joint transforms and visibility always remain owned by Three/WebXR.
 */
export function createAvatarHands(renderer: THREE.WebGLRenderer, rig: THREE.Group) {
  const geometry = {
    joint: new THREE.SphereGeometry(1, 10, 6),
    bone: new THREE.CylinderGeometry(1, 1, 1, 8, 1, true),
    palm: new RoundedBoxGeometry(1, 1, 1, 2, 0.2),
  };
  const glove = gloveMaterial(0xe8dfcc);
  const cuffMaterial = gloveMaterial(0x567b80);
  const state = { visibleCount: 0, left: idleState(), right: idleState() };
  let active = renderer.xr.isPresenting;
  let disposed = false;

  // Both tracked and controller gloves use four draw calls, independent of finger count.
  function makeGlove(name: string) {
    const group = new THREE.Group();
    group.name = name;
    group.visible = false;
    const joints = new THREE.InstancedMesh(geometry.joint, glove, 20);
    const bones = new THREE.InstancedMesh(geometry.bone, glove, 15);
    const palm = new THREE.Mesh(geometry.palm, glove);
    const cuff = new THREE.Mesh(geometry.palm, cuffMaterial);
    joints.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    bones.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    joints.frustumCulled = bones.frustumCulled = false;
    group.add(joints, bones, palm, cuff);
    const pose = new THREE.Object3D();
    const direction = new THREE.Vector3();
    let jointCount = 0;
    let boneCount = 0;
    return {
      group,
      palm,
      cuff,
      reset() {
        jointCount = boneCount = 0;
      },
      joint(position: THREE.Vector3, radius: number) {
        pose.position.copy(position);
        pose.quaternion.identity();
        pose.scale.setScalar(radius);
        pose.updateMatrix();
        joints.setMatrixAt(jointCount++, pose.matrix);
      },
      bone(a: THREE.Vector3, b: THREE.Vector3, radius: number) {
        direction.subVectors(b, a);
        const length = direction.length();
        if (length < 0.0001) return;
        pose.position.copy(a).add(b).multiplyScalar(0.5);
        pose.quaternion.setFromUnitVectors(UP, direction.multiplyScalar(1 / length));
        pose.scale.set(radius, length, radius);
        pose.updateMatrix();
        bones.setMatrixAt(boneCount++, pose.matrix);
      },
      finish() {
        joints.count = jointCount;
        bones.count = boneCount;
        joints.instanceMatrix.needsUpdate = bones.instanceMatrix.needsUpdate = true;
      },
      dispose() {
        group.removeFromParent();
        joints.dispose();
        bones.dispose();
      },
    };
  }

  const inputs = [0, 1].map((index) => {
    const grip = renderer.xr.getControllerGrip(index);
    const hand = renderer.xr.getHand(index);
    const controllerModel = makeGlove(`avatar-controller-hand-${index}`);
    const trackedModel = makeGlove(`avatar-tracked-hand-${index}`);
    grip.add(controllerModel.group);
    hand.add(trackedModel.group);
    rig.add(grip, hand);
    let source: XRInputSource | null = null;
    const connected = (event: { data: XRInputSource }) => {
      source = event.data;
    };
    const disconnected = () => {
      source = null;
      controllerModel.group.visible = trackedModel.group.visible = false;
    };
    grip.addEventListener('connected', connected);
    grip.addEventListener('disconnected', disconnected);

    const fingerPoints = Array.from({ length: 4 }, () => new THREE.Vector3());
    function updateController(side: 'left' | 'right', trigger: number, squeeze: number) {
      const model = controllerModel;
      const mirror = side === 'left' ? 1 : -1;
      // Grip space points forward along -Z. The palm wraps the handle, thumb uppermost.
      model.group.rotation.z = mirror * Math.PI * 0.5;
      model.palm.scale.set(0.075, 0.027, 0.087);
      model.palm.position.set(0, 0.012, 0.005);
      model.cuff.scale.set(0.062, 0.034, 0.025);
      model.cuff.position.set(0, 0.012, 0.057);
      model.reset();
      for (let finger = 0; finger < 5; finger++) {
        const thumb = finger === 0;
        const radius = finger === 4 ? 0.0065 : thumb ? 0.0085 : 0.0078;
        const curl = thumb ? 0.4 + squeeze * 0.5 : 0.3 + (finger === 1 ? trigger : squeeze) * 1.2;
        const length = [0.052, 0.071, 0.08, 0.074, 0.06][finger];
        fingerPoints[0].set(
          mirror * (thumb ? 0.038 : 0.027 - (finger - 1) * 0.018),
          0.011,
          thumb ? 0.007 : -0.034,
        );
        for (let segment = 0; segment < 3; segment++) {
          const angle = curl * (0.5 + segment * 0.72);
          const segmentLength = length * [0.45, 0.32, 0.23][segment];
          fingerPoints[segment + 1]
            .copy(fingerPoints[segment])
            .addScaledVector(
              direction
                .set(
                  thumb ? mirror * (0.48 - segment * 0.3) : 0,
                  -Math.sin(angle),
                  -Math.cos(angle),
                )
                .normalize(),
              segmentLength,
            );
        }
        for (let segment = 0; segment < 4; segment++) {
          const r = radius * (1 - segment * 0.065);
          model.joint(fingerPoints[segment], r);
          if (segment < 3) model.bone(fingerPoints[segment], fingerPoints[segment + 1], r * 0.94);
        }
      }
      model.finish();
    }

    const direction = new THREE.Vector3();
    const across = new THREE.Vector3();
    const normal = new THREE.Vector3();
    const basis = new THREE.Matrix4();
    const valid = (joint: Joint | undefined): joint is Joint =>
      !!joint?.visible &&
      Number.isFinite(joint.position.x) &&
      Number.isFinite(joint.position.y) &&
      Number.isFinite(joint.position.z);
    const radiusOf = (joint: Joint) =>
      THREE.MathUtils.clamp(joint.jointRadius || 0.008, 0.004, 0.013);

    function updateTracked() {
      const joints = hand.joints;
      const wrist = joints.wrist;
      const indexBase = joints['index-finger-phalanx-proximal'];
      const middleBase = joints['middle-finger-phalanx-proximal'];
      const pinkyBase = joints['pinky-finger-phalanx-proximal'];
      const count = Object.values(joints).filter(valid).length;
      if (
        !valid(wrist) ||
        !valid(indexBase) ||
        !valid(middleBase) ||
        !valid(pinkyBase) ||
        count < 10
      ) {
        return 0;
      }
      // Three's joint coordinates are already in the hand group's reference-space frame.
      // Derive the palm from measured knuckles; never apply the controller grip pose again.
      direction.subVectors(wrist.position, middleBase.position);
      const length = direction.length();
      across.subVectors(indexBase.position, pinkyBase.position);
      const width = across.length();
      if (length < 0.015 || width < 0.015) return 0;
      direction.normalize();
      normal.crossVectors(direction, across).normalize();
      if (normal.lengthSq() < 0.5) return 0;
      across.crossVectors(normal, direction).normalize();
      basis.makeBasis(across, normal, direction);
      const model = trackedModel;
      model.palm.quaternion.setFromRotationMatrix(basis);
      model.palm.position.copy(wrist.position).lerp(middleBase.position, 0.52);
      model.palm.scale.set(width + 0.017, 0.025, length * 0.96);
      model.cuff.quaternion.copy(model.palm.quaternion);
      model.cuff.position.copy(wrist.position).addScaledVector(direction, 0.01);
      model.cuff.scale.set(width * 0.86, 0.031, 0.024);
      model.reset();
      for (const finger of FINGERS) {
        for (let i = 0; i < finger.length; i++) {
          const joint = joints[finger[i]];
          if (!valid(joint)) continue;
          model.joint(joint.position, radiusOf(joint));
          const next = joints[finger[i + 1]];
          if (valid(next)) {
            model.bone(joint.position, next.position, Math.min(radiusOf(joint), radiusOf(next)));
          }
        }
      }
      model.finish();
      return count;
    }

    return {
      update() {
        controllerModel.group.visible = trackedModel.group.visible = false;
        if (!active || !source || (source.handedness !== 'left' && source.handedness !== 'right'))
          return;
        const diagnostic = state[source.handedness];
        if (source.hand && hand.visible) {
          diagnostic.joints = updateTracked();
          if (diagnostic.joints) {
            trackedModel.group.visible = true;
            diagnostic.mode = 'joints';
            state.visibleCount++;
          }
        } else if (!source.hand && grip.visible) {
          const value = (button: number) => {
            const v = source?.gamepad?.buttons[button]?.value ?? 0;
            return Number.isFinite(v) ? THREE.MathUtils.clamp(v, 0, 1) : 0;
          };
          diagnostic.trigger = value(0);
          diagnostic.squeeze = value(1);
          updateController(source.handedness, diagnostic.trigger, diagnostic.squeeze);
          controllerModel.group.visible = true;
          diagnostic.mode = 'controller';
          state.visibleCount++;
        }
      },
      hide() {
        controllerModel.group.visible = trackedModel.group.visible = false;
      },
      dispose() {
        grip.removeEventListener('connected', connected);
        grip.removeEventListener('disconnected', disconnected);
        controllerModel.dispose();
        trackedModel.dispose();
        // These spaces were attached here, but do not remove any unrelated runtime children.
        if (grip.parent === rig && !grip.children.length) rig.remove(grip);
        if (
          hand.parent === rig &&
          hand.children.every((child) =>
            Object.values(hand.joints).some((joint) => joint === child),
          )
        ) {
          rig.remove(hand);
        }
      },
    };
  });

  function resetState() {
    state.visibleCount = 0;
    Object.assign(state.left, idleState());
    Object.assign(state.right, idleState());
  }
  function setSessionActive(value: boolean) {
    active = value;
    if (!active) {
      inputs.forEach((input) => input.hide());
      resetState();
    }
  }
  const start = () => setSessionActive(true);
  const end = () => setSessionActive(false);
  renderer.xr.addEventListener('sessionstart', start);
  renderer.xr.addEventListener('sessionend', end);
  return {
    state,
    setSessionActive,
    update() {
      if (disposed) return;
      resetState();
      inputs.forEach((input) => input.update());
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      setSessionActive(false);
      renderer.xr.removeEventListener('sessionstart', start);
      renderer.xr.removeEventListener('sessionend', end);
      inputs.forEach((input) => input.dispose());
      Object.values(geometry).forEach((item) => item.dispose());
      glove.dispose();
      cuffMaterial.dispose();
    },
  };
}
