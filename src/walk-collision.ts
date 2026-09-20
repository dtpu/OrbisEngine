export interface WalkCollisionGrid {
  cell: number;
  width: number;
  height: number;
  /** Integer cell coordinates of the grid's lower X/Z corner. */
  ox: number;
  oz: number;
}

/** Count occupied cells intersecting a circular footprint, excluding tangent contact.
 * Cell rectangles, rather than their centres, preserve contact with thin edge slivers.
 * The caller decides which cells are solid at the walker's current height.
 */
export function countGridFootprintContacts(
  grid: WalkCollisionGrid,
  x: number,
  z: number,
  radius: number,
  isBlocked: (index: number) => boolean,
): number {
  const minI = Math.max(0, Math.floor((x - radius) / grid.cell) - grid.ox);
  const maxI = Math.min(grid.width - 1, Math.floor((x + radius) / grid.cell) - grid.ox);
  const minJ = Math.max(0, Math.floor((z - radius) / grid.cell) - grid.oz);
  const maxJ = Math.min(grid.height - 1, Math.floor((z + radius) / grid.cell) - grid.oz);
  const radiusSquared = radius * radius;
  let contacts = 0;
  for (let j = minJ; j <= maxJ; j++) {
    const nearZ = (grid.oz + j) * grid.cell;
    const farZ = (grid.oz + j + 1) * grid.cell;
    const dz = Math.max(nearZ - z, 0, z - farZ);
    for (let i = minI; i <= maxI; i++) {
      const nearX = (grid.ox + i) * grid.cell;
      const farX = (grid.ox + i + 1) * grid.cell;
      const dx = Math.max(nearX - x, 0, x - farX);
      if (dx * dx + dz * dz < radiusSquared && isBlocked(j * grid.width + i)) contacts++;
    }
  }
  return contacts;
}
