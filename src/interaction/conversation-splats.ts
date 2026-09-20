import * as THREE from 'three';
import { dyno, type SplatMesh } from '@sparkjsdev/spark';
import type { ConversationPose } from './conversation-motion';

/** Invented soft neck/chest articulation for baked splats, not recovered skeletal weights. */
export class ConversationSplats {
  private readonly neck = dyno.dynoVec3(new THREE.Vector3());
  private readonly localToBody = dyno.dynoMat4(new THREE.Matrix4());
  private readonly rotation = dyno.dynoVec4(new THREE.Vector4(0, 0, 0, 1));
  private readonly breathing = dyno.dynoVec3(new THREE.Vector3());
  private readonly enabled = dyno.dynoFloat(0);
  private readonly modifier;
  private readonly basis = new THREE.Matrix4();
  private readonly inverse = new THREE.Matrix4();
  private readonly right = new THREE.Vector3();
  private readonly up = new THREE.Vector3(0, 1, 0);
  private readonly forward = new THREE.Vector3();
  private readonly pivot = new THREE.Vector3();
  private readonly localRotation = new THREE.Quaternion();
  private readonly bodyRotation = new THREE.Quaternion();
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
          },
          outTypes: { gsplat: dyno.Gsplat },
          inputs: {
            gsplat,
            neck: this.neck,
            body: this.localToBody,
            rotation: this.rotation,
            breathing: this.breathing,
            enabled: this.enabled,
          },
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
            }`,
          ],
        }).outputs.gsplat,
      }),
    );
    // Preserve existing modifiers and source-video projection. Only this overlay is removed later.
    mesh.objectModifiers = [...(mesh.objectModifiers ?? []), this.modifier];
    mesh.updateGenerator();
  }

  update(pose: ConversationPose, head: THREE.Vector3, forward: THREE.Vector3, stature: number) {
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
    this.delta.setFromEuler(this.euler.set(pose.pitch, pose.yaw, pose.roll));
    this.delta.premultiply(this.localRotation).multiply(this.localRotation.invert());
    this.rotation.value.set(this.delta.x, this.delta.y, this.delta.z, this.delta.w);
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
    if (this.enabled.value === 0) return;
    this.enabled.value = 0;
    this.mesh.updateVersion();
  }

  dispose() {
    if (this.disposed) return;
    this.reset();
    this.disposed = true;
    this.mesh.objectModifiers = this.mesh.objectModifiers?.filter((m) => m !== this.modifier);
    this.mesh.updateGenerator();
  }
}
