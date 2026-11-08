import type { Vector3 } from 'three';

/** Recorded offsets are world-space registration, independent of optional foot effects. */
export function applyPlacementOffset(
  position: Vector3,
  own: readonly (readonly number[])[] | undefined,
  shared: readonly (readonly number[])[] | null,
  first: number,
  i: number,
  j: number,
  u: number,
): void {
  const offsets = own || shared;
  if (!offsets) return;
  // Per-person tables follow that sequence; legacy shared tables follow the full clip.
  const start = own ? 0 : first;
  const a = offsets[start + i] || [0, 0, 0];
  const b = offsets[start + j] || a;
  position.x += a[0] + (b[0] - a[0]) * u;
  position.y += a[1] + (b[1] - a[1]) * u;
  position.z += a[2] + (b[2] - a[2]) * u;
}

export function stanceCorrectionEnabled(
  value: string | null,
  feetOn: boolean,
  lockOn: boolean,
  hasPlacement: boolean,
): boolean {
  return feetOn && !lockOn && (value === null ? !hasPlacement : value !== '0');
}
