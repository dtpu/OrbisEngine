/**
 * A thin progress bar. With a fraction it fills to that point; without one it shows a moving
 * band, because the pipeline reports no percentage and pretending otherwise would be a lie.
 */
export function Progress({
  fraction,
  label,
  tone = 'active',
}: {
  fraction?: number | null;
  label: string;
  tone?: 'active' | 'good' | 'attention';
}) {
  const determinate = typeof fraction === 'number' && Number.isFinite(fraction);
  const width = determinate
    ? `${Math.round(Math.min(1, Math.max(0, fraction)) * 100)}%`
    : undefined;
  return (
    <div
      className={`progress progress--${tone}${determinate ? '' : ' progress--indeterminate'}`}
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={determinate ? Math.round(fraction * 100) : undefined}
      aria-valuetext={label}
    >
      <span style={width ? { width } : undefined} />
    </div>
  );
}
