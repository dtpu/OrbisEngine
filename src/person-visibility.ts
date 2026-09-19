// Visibility runs refer to the shared source sample grid, not compact PLY indices.
export function personVisibility(
  runs: number[][] | null | undefined,
  timestamps: number[] | undefined,
  duration: number,
): (time: number) => boolean {
  if (!runs) return () => true;
  if (
    !timestamps?.length ||
    !Number.isFinite(duration) ||
    timestamps.some((time, i) => !Number.isFinite(time) || (i > 0 && time <= timestamps[i - 1])) ||
    duration <= timestamps[timestamps.length - 1]
  ) {
    throw new Error('Person visibility requires a valid shared source timeline');
  }
  let previousEnd = -1;
  const intervals = runs.map((run) => {
    if (
      run.length !== 2 ||
      run.some((index) => !Number.isInteger(index)) ||
      run[0] < 0 ||
      run[1] < run[0] ||
      run[1] >= timestamps.length ||
      run[0] <= previousEnd
    ) {
      throw new Error('Person visibility runs must be ordered source sample ranges');
    }
    previousEnd = run[1];
    return [timestamps[run[0]], timestamps[run[1] + 1] ?? duration];
  });
  return (time) =>
    Number.isFinite(time) && intervals.some(([start, end]) => time >= start && time < end);
}
