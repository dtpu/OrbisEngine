import * as THREE from 'three';
import { dyno, type SplatMesh } from '@sparkjsdev/spark';
import type { ConversationPose } from './conversation-motion';

const makeArm = () => ({
  shoulder: dyno.dynoVec3(new THREE.Vector3()),
  elbow: dyno.dynoVec3(new THREE.Vector3()),
  upper: dyno.dynoVec4(new THREE.Vector4(0, 0, 0, 1)),
  lower: dyno.dynoVec4(new THREE.Vector4(0, 0, 0, 1)),
  freedom: 0,
});

/** A held prop pins its side; an ambiguous central attachment conservatively pins both. */
export function freeConversationArms(bodyPositions: THREE.Vector3[]): [number, number] {
  let left = 1,
    right = 1;
  for (const point of bodyPositions) {
    if (point.x < 0.04) left = 0;
    if (point.x > -0.04) right = 0;
  }
  return [left, right];
}

/** Invented soft articulation for baked splats, not recovered skeletal weights. */
export class ConversationSplats {
  private readonly neck = dyno.dynoVec3(new THREE.Vector3());
  private readonly localToBody = dyno.dynoMat4(new THREE.Matrix4());
  private readonly rotation = dyno.dynoVec4(new THREE.Vector4(0, 0, 0, 1));
  private readonly breathing = dyno.dynoVec3(new THREE.Vector3());
  private readonly enabled = dyno.dynoFloat(0);
  private readonly left = makeArm();
  private readonly rightArm = makeArm();
  private readonly modifier;
  private readonly basis = new THREE.Matrix4();
  private readonly inverse = new THREE.Matrix4();
  private readonly right = new THREE.Vector3();
  private readonly up = new THREE.Vector3(0, 1, 0);
  private readonly forward = new THREE.Vector3();
  private readonly pivot = new THREE.Vector3();
  private readonly localRotation = new THREE.Quaternion();
  private readonly bodyRotation = new THREE.Quaternion();
  private readonly inverseRotation = new THREE.Quaternion();
  private readonly delta = new THREE.Quaternion();
  private readonly euler = new THREE.Euler();
  private disposed = false;

  constructor(private readonly mesh: SplatMesh) {
    this.modifier = dyno.dynoBlock(
      { gsplat: dyno.Gsplat },
      { gsplat: dyno.Gsplat },
      ({ gsplat }) => ({
        gsplat: new dyno.Dyno({
          inTypes: {
            gsplat: dyno.Gsplat,
            neck: 'vec3',
            body: 'mat4',
            rotation: 'vec4',
            breathing: 'vec3',
            enabled: 'float',
            leftShoulder: 'vec3',
            leftElbow: 'vec3',
            leftUpper: 'vec4',
            leftLower: 'vec4',
            rightShoulder: 'vec3',
            rightElbow: 'vec3',
            rightUpper: 'vec4',
            rightLower: 'vec4',
          },
          outTypes: { gsplat: dyno.Gsplat },
          inputs: {
            gsplat,
            neck: this.neck,
            body: this.localToBody,
            rotation: this.rotation,
            breathing: this.breathing,
            enabled: this.enabled,
            leftShoulder: this.left.shoulder,
            leftElbow: this.left.elbow,
            leftUpper: this.left.upper,
            leftLower: this.left.lower,
            rightShoulder: this.rightArm.shoulder,
            rightElbow: this.rightArm.elbow,
            rightUpper: this.rightArm.upper,
            rightLower: this.rightArm.lower,
          },
          globals: () => [
            `vec3 wanderConversationRotate(vec4 q, vec3 v) {
              return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
            }
            vec4 wanderConversationMultiply(vec4 q, vec4 r) {
              return vec4(q.w * r.xyz + r.w * q.xyz + cross(q.xyz, r.xyz),
                q.w * r.w - dot(q.xyz, r.xyz));
            }`,
          ],
          statements: ({ inputs: i, outputs: o }) => [
            `${o.gsplat} = ${i.gsplat};
            if (${i.enabled} > 0.0) {
              // Body coordinates are in body-heights, with the approximate neck as origin.
              vec3 p = (${i.body} * vec4(${i.gsplat}.center, 1.0)).xyz;
              float head = smoothstep(-0.015, 0.045, p.y)
                * (1.0 - smoothstep(0.09, 0.15, length(p.xz)));
              vec4 q = normalize(mix(vec4(0.0, 0.0, 0.0, 1.0), ${i.rotation}, head));
              vec3 v = ${i.gsplat}.center - ${i.neck};
              ${o.gsplat}.center += 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
              vec4 r = ${i.gsplat}.quaternion;
              ${o.gsplat}.quaternion = vec4(q.w * r.xyz + r.w * q.xyz + cross(q.xyz, r.xyz),
                q.w * r.w - dot(q.xyz, r.xyz));
              float chest = smoothstep(-0.40, -0.24, p.y)
                * (1.0 - smoothstep(-0.10, 0.015, p.y))
                * (1.0 - smoothstep(0.10, 0.19, abs(p.x)))
                * (1.0 - smoothstep(0.12, 0.22, abs(p.z)));
              ${o.gsplat}.center += ${i.breathing} * chest;
              ${(['left', 'right'] as const)
                .map((side) => {
                  const sign = side === 'left' ? '-1.0' : '1.0';
                  return `{
                  // Approximate arm regions on upright poses. Taper out above the hips:
                  // a sideways lean can put an upper thigh directly below the hand.
                  float arm = smoothstep(0.08, 0.155, ${sign} * p.x)
                    * smoothstep(-0.40, -0.32, p.y)
                    * (1.0 - smoothstep(-0.075, 0.015, p.y))
                    * (1.0 - smoothstep(0.22, 0.34, abs(p.z)));
                  float forearm = arm * (1.0 - smoothstep(-0.30, -0.20, p.y));
                  vec4 lower = normalize(mix(vec4(0.0, 0.0, 0.0, 1.0), ${i[`${side}Lower`]}, forearm));
                  vec4 upper = normalize(mix(vec4(0.0, 0.0, 0.0, 1.0), ${i[`${side}Upper`]}, arm));
                  if (dot(lower.xyz, lower.xyz) + dot(upper.xyz, upper.xyz) > 0.0) {
                    vec3 c = ${o.gsplat}.center;
                    c = ${i[`${side}Elbow`]} + wanderConversationRotate(lower, c - ${i[`${side}Elbow`]});
                    ${o.gsplat}.center = ${i[`${side}Shoulder`]} + wanderConversationRotate(upper, c - ${i[`${side}Shoulder`]});
                    ${o.gsplat}.quaternion = wanderConversationMultiply(upper,
                      wanderConversationMultiply(lower, ${o.gsplat}.quaternion));
                  }
                }`;
                })
                .join('\n')}
            }`,
          ],
        }).outputs.gsplat,
      }),
    );
    // Preserve existing modifiers and source-video projection. Only this overlay is removed later.
    mesh.objectModifiers = [...(mesh.objectModifiers ?? []), this.modifier];
    mesh.updateGenerator();
  }

  update(
    pose: ConversationPose,
    head: THREE.Vector3,
    forward: THREE.Vector3,
    stature: number,
    heldPositions: THREE.Vector3[] = [],
    deltaSeconds = 1 / 60,
  ) {
    if (this.disposed) return;
    if (!(stature > 0) || !Number.isFinite(stature) || pose.weight === 0) {
      this.reset();
      return;
    }
    this.forward.copy(forward).setY(0);
    if (this.forward.lengthSq() < 1e-8) {
      this.reset();
      return;
    }
    this.forward.normalize();
    this.right.crossVectors(this.up, this.forward).normalize();
    this.basis.makeBasis(this.right, this.up, this.forward);
    this.bodyRotation.setFromRotationMatrix(this.basis);
    this.pivot.copy(head).addScaledVector(this.up, -0.07 * stature);
    this.basis.scale(new THREE.Vector3(stature, stature, stature)).setPosition(this.pivot);
    this.mesh.updateWorldMatrix(true, false);
    this.inverse.copy(this.mesh.matrixWorld).invert();
    this.neck.value.copy(this.pivot).applyMatrix4(this.inverse);
    this.localToBody.value.copy(this.basis).invert().multiply(this.mesh.matrixWorld);

    this.mesh.getWorldQuaternion(this.localRotation).invert().multiply(this.bodyRotation);
    this.inverseRotation.copy(this.localRotation).invert();
    this.delta.setFromEuler(this.euler.set(pose.pitch, pose.yaw, pose.roll));
    this.delta.premultiply(this.localRotation).multiply(this.inverseRotation);
    this.rotation.value.set(this.delta.x, this.delta.y, this.delta.z, this.delta.w);
    const bodyInverse = this.basis.clone().invert();
    const freedom = freeConversationArms(
      heldPositions.map((p) => p.clone().applyMatrix4(bodyInverse)),
    );
    const dt = Number.isFinite(deltaSeconds) ? Math.max(0, Math.min(0.1, deltaSeconds)) : 0;
    for (const [index, arm] of [this.left, this.rightArm].entries()) {
      // Pin immediately on a return; blend back to gesturing after the visitor takes the prop.
      arm.freedom =
        freedom[index] === 0 ? 0 : arm.freedom + (1 - arm.freedom) * (1 - Math.exp(-6 * dt));
      const side = index === 0 ? -1 : 1;
      arm.shoulder.value
        .set(side * 0.115, -0.055, 0)
        .applyMatrix4(this.basis)
        .applyMatrix4(this.inverse);
      arm.elbow.value
        .set(side * 0.145, -0.255, 0.01)
        .applyMatrix4(this.basis)
        .applyMatrix4(this.inverse);
      const shoulder = (index === 0 ? pose.leftShoulder : pose.rightShoulder) * arm.freedom;
      const elbow = (index === 0 ? pose.leftElbow : pose.rightElbow) * arm.freedom;
      this.delta
        .setFromEuler(this.euler.set(-shoulder, 0, side * shoulder * 0.5))
        .premultiply(this.localRotation)
        .multiply(this.inverseRotation);
      arm.upper.value.set(this.delta.x, this.delta.y, this.delta.z, this.delta.w);
      this.delta
        .setFromEuler(this.euler.set(-elbow, 0, 0))
        .premultiply(this.localRotation)
        .multiply(this.inverseRotation);
      arm.lower.value.set(this.delta.x, this.delta.y, this.delta.z, this.delta.w);
    }
    // Translate only the central chest; feet and separately held props keep their recorded pose.
    this.breathing.value
      .copy(this.forward)
      .addScaledVector(this.up, 0.3)
      .multiplyScalar(pose.breath * stature)
      .add(this.pivot)
      .applyMatrix4(this.inverse)
      .sub(this.neck.value);
    this.enabled.value = 1;
    this.mesh.updateVersion();
  }

  reset() {
    this.left.freedom = this.rightArm.freedom = 0;
    if (this.enabled.value === 0) return;
    this.enabled.value = 0;
    this.mesh.updateVersion();
  }

  get armFreedom() {
    return [this.left.freedom, this.rightArm.freedom];
  }

  dispose() {
    if (this.disposed) return;
    this.reset();
    this.disposed = true;
    this.mesh.objectModifiers = this.mesh.objectModifiers?.filter((m) => m !== this.modifier);
    this.mesh.updateGenerator();
  }
}
