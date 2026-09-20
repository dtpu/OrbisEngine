import { statusLabel, statusTone } from '@/lib/format';

/**
 * A small status word coloured by tone. The text is always present; colour only reinforces it.
 * `blockedReason` distinguishes a stage that never ran from one that broke.
 */
export function Status({
  status,
  blockedReason,
  className = '',
}: {
  status: string;
  blockedReason?: string | null;
  className?: string;
}) {
  return (
    <span
      className={`status status--${statusTone(status, blockedReason)} ${className}`.trim()}
      title={blockedReason || status}
    >
      {statusLabel(status, blockedReason)}
    </span>
  );
}
