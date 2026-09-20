import { Quaternion, Vector3 } from 'three';

export function parseWalkGain(value: string | null): number {
  const gain = Number(value);
  return Number.isFinite(gain) ? Math.min(2, Math.max(1, gain)) : 1;
}

/** Extra rig travel only; the XR runtime still supplies the unmodified tracked pose. */
export class PhysicalWalk {
  private previous = new Vector3();
  private hasBaseline = false;

  constructor(readonly gain: number) {}

  reset(): void {
    this.hasBaseline = false;
  }

  update(
    headLocal: Vector3,
    rigQuaternion: Quaternion,
    unitsPerMetre: number,
    out: Vector3,
    active = true,
  ): Vector3 {
    out.set(0, 0, 0);
    if (
      !active ||
      !Number.isFinite(headLocal.x) ||
      !Number.isFinite(headLocal.y) ||
      !Number.isFinite(headLocal.z)
    ) {
      this.reset();
      return out;
    }
    if (this.hasBaseline) {
      out
        .set(headLocal.x - this.previous.x, 0, headLocal.z - this.previous.z)
        .multiplyScalar((this.gain - 1) * unitsPerMetre)
        .applyQuaternion(rigQuaternion);
    }
    this.previous.copy(headLocal);
    this.hasBaseline = true;
    return out;
  }
}
