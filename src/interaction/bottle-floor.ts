/** A floor sample taken from the walk collision grid. */
export type BottleFloorCell = {
  floor: number;
  inside: boolean;
  distance: number;
};

/**
 * Fills only a one-cell omission inside measured collision-grid floor coverage.
 *
 * The walker can traverse these sparse cells at its carried height. A loose prop needs a concrete
 * support height, but must not turn a wider unobserved area into a floor. Two agreeing neighbouring
 * floor samples distinguish the omission from a lone noisy splat; the higher one is conservative
 * for an object settling onto the surface.
 */
export function interpolateBottleFloor(
  cell: BottleFloorCell,
  neighbouringFloors: readonly number[],
  maximumSpread: number,
): number | null {
  if (Number.isFinite(cell.floor)) return cell.floor;
  if (!cell.inside || cell.distance !== 1 || !Number.isFinite(maximumSpread) || maximumSpread <= 0)
    return null;
  const floors = neighbouringFloors.filter(Number.isFinite);
  if (floors.length < 2) return null;
  const low = Math.min(...floors);
  const high = Math.max(...floors);
  return high - low <= maximumSpread ? high : null;
}
