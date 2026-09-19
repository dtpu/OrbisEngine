/** Normalize a playback time without perturbing timestamps already inside the clip. */
export function normalizePlaybackTime(value: number, duration: number): number {
  if (!Number.isFinite(duration) || duration <= 0 || !Number.isFinite(value)) return 0;
  if (value === 0) return 0;
  if (value > 0 && value < duration) return value;
  const wrapped = value % duration;
  const normalized = wrapped < 0 ? wrapped + duration : wrapped;
  return normalized === 0 || normalized === duration ? 0 : normalized;
}
