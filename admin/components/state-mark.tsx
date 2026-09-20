/**
 * What a stage looks like before it has produced anything.
 *
 * Every stage in the graph gets a tile, but most of a run's stages have no output yet, and a row
 * of identical grey placeholders hides the one thing the operator wants to see: where the run
 * actually is. This draws the stage's state instead, in the frame-and-lens shape the product mark
 * uses, so an empty tile still carries information and still looks like this dashboard.
 */
export function StateMark({
  tone,
  label,
}: {
  tone: 'idle' | 'active' | 'attention' | 'good' | 'bad' | 'muted';
  label: string;
}) {
  return (
    <span className={`mark mark--${tone}`}>
      <svg viewBox="0 0 48 34" role="img" aria-label={label}>
        <rect
          className="mark__frame"
          x="1.5"
          y="1.5"
          width="45"
          height="31"
          rx="3"
          strokeDasharray={tone === 'muted' || tone === 'idle' ? '3 3' : undefined}
        />
        {tone === 'good' ? <circle className="mark__lens" cx="24" cy="17" r="6.5" /> : null}
        {tone === 'bad' ? (
          <>
            <circle className="mark__lens mark__lens--hollow" cx="24" cy="17" r="6.5" />
            <line className="mark__slash" x1="17" y1="24" x2="31" y2="10" />
          </>
        ) : null}
        {tone === 'attention' ? (
          <circle
            className="mark__lens mark__lens--hollow"
            cx="24"
            cy="17"
            r="6.5"
            strokeDasharray="28 13"
          />
        ) : null}
        {tone === 'active' ? (
          <circle
            className="mark__lens mark__lens--sweep"
            cx="24"
            cy="17"
            r="6.5"
            strokeDasharray="14 27"
          />
        ) : null}
        {tone === 'idle' || tone === 'muted' ? (
          <circle className="mark__lens mark__lens--faint" cx="24" cy="17" r="2" />
        ) : null}
      </svg>
    </span>
  );
}
