/** Normalize a playback time without perturbing timestamps already inside the clip. */
export function normalizePlaybackTime(value: number, duration: number): number {
  if (!Number.isFinite(duration) || duration <= 0 || !Number.isFinite(value)) return 0;
  if (value === 0) return 0;
  if (value > 0 && value < duration) return value;
  const wrapped = value % duration;
  const normalized = wrapped < 0 ? wrapped + duration : wrapped;
  return normalized === 0 || normalized === duration ? 0 : normalized;
}

/** Sample recorded poses without blending the last observation back to the first. */
export function sampleAnimationFrame(
  time: number,
  timestamps: readonly number[],
): [number, number] {
  let i = 0;
  while (i + 1 < timestamps.length && timestamps[i + 1] <= time) i++;
  if (i + 1 >= timestamps.length || time <= timestamps[i]) return [i, 0];
  const span = timestamps[i + 1] - timestamps[i];
  return [i, span > 0 ? Math.min(1, (time - timestamps[i]) / span) : 0];
}
