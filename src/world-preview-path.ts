/** Accept same-origin SPZ assets only; private run assets remain behind their existing proxy. */
export function worldPreviewAssetUrl(world: string | null): string {
  if (
    !world ||
    !/^\/(?!\/)[A-Za-z0-9_./-]+\.spz$/.test(world) ||
    world.split('/').some((segment) => segment === '.' || segment === '..') ||
    !(
      /^\/[A-Za-z0-9][A-Za-z0-9._-]*\.spz$/.test(world) ||
      /^\/api\/pipeline\/runs\/[A-Za-z0-9_.-]+\/viewer\/assets\/.+\.spz$/.test(world)
    )
  ) {
    throw new Error('Choose a same-origin SPZ asset path such as /marble-example-clean.spz.');
  }
  return world;
}
