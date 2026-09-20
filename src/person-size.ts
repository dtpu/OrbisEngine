import * as THREE from 'three';

/** A visual body-size adjustment, independent of the scene's physical ruler. */
export function parsePersonSize(value: string | null): number {
  if (value === null || value.trim() === '') return 1;
  const factor = Number(value);
  return Number.isFinite(factor) && factor > 0 ? factor : 1;
}

const offset = new THREE.Vector3();

/** Rebuild from the unsized placement each frame; never feed back the last sized position.
 * The measured local foot anchor has exactly the same world position before and after sizing,
 * including under rotated/nonuniform registration transforms. Children (including audio
 * anchors) follow the visual body; the caller retains referenceScale for scene calibration.
 */
export function applyPersonSizeAtAnchor(
  person: THREE.Object3D,
  referencePosition: THREE.Vector3,
  referenceScale: THREE.Vector3,
  localFoot: THREE.Vector3,
  factor: number,
): void {
  offset.copy(localFoot).multiply(referenceScale).applyQuaternion(person.quaternion);
  person.position.copy(referencePosition).addScaledVector(offset, 1 - factor);
  person.scale.copy(referenceScale).multiplyScalar(factor);
}
