/** Keep recorded animation work off some XR frames without slowing the media clock. */
export function createAnimationCadence(fps = 36) {
  const interval = Number.isFinite(fps) && fps > 0 ? 1000 / fps : 0;
  let next = -Infinity;
  return (now: number, presenting: boolean): boolean => {
    if (!presenting || !interval) {
      next = -Infinity;
      return true;
    }
    if (now < next) return false;
    // Preserve cadence through ordinary frame jitter, but never queue catch-up work after a stall.
    next = now - next >= interval ? now + interval : next + interval;
    return true;
  };
}
