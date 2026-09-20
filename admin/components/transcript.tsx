import type { TranscriptEvent } from '@/lib/types';

const OUTPUT_LIMIT = 4000;

/**
 * Codex emits `item.started` then `item.completed` for the same item id; the operator wants one
 * row per command that updates in place, not two.
 */
export function collapse(events: TranscriptEvent[]): TranscriptEvent[] {
  const rows: TranscriptEvent[] = [];
  const index = new Map<string, number>();
  for (const event of events) {
    const id = event.item?.id;
    if (id && (event.type === 'item.started' || event.type === 'item.completed')) {
      const at = index.get(id);
      if (at !== undefined) {
        rows[at] = event;
        continue;
      }
      index.set(id, rows.length);
    }
    rows.push(event);
  }
  return rows;
}

function trim(text: string): string {
  return text.length > OUTPUT_LIMIT
    ? `${text.slice(0, OUTPUT_LIMIT)}\n… (${text.length - OUTPUT_LIMIT} more characters)`
    : text;
}

/** One line of a `codex exec --json` transcript, rendered the way an operator reads it. */
export function TranscriptRow({ event }: { event: TranscriptEvent }) {
  const item = event.item;
  switch (event.type) {
    case 'thread.started':
      return <p className="tail__system">Review started</p>;
    case 'turn.started':
      return null;
    case 'turn.completed': {
      const usage = event.usage as Record<string, number> | undefined;
      const tokens = usage
        ? Object.entries(usage)
            .filter(([, value]) => typeof value === 'number')
            .map(([key, value]) => `${key.replaceAll('_', ' ')} ${value}`)
            .join(', ')
        : '';
      return <p className="tail__system">Agent finished{tokens ? ` (${tokens})` : ''}</p>;
    }
    case 'item.started':
    case 'item.completed':
      break;
    case 'raw':
      return <p className="tail__raw mono">{event.text}</p>;
    default:
      return <p className="tail__raw mono">{event.type}</p>;
  }
  if (!item) return null;
  switch (item.type) {
    case 'agent_message':
      return <div className="tail__say">{item.text}</div>;
    case 'reasoning':
      return <p className="tail__think">{item.text}</p>;
    case 'command_execution': {
      const running = event.type === 'item.started' || item.status === 'in_progress';
      const failed = typeof item.exit_code === 'number' && item.exit_code !== 0;
      const output = (item.aggregated_output ?? '').trim();
      return (
        <details className={`tail__cmd${failed ? ' tail__cmd--failed' : ''}`}>
          <summary>
            <span className="mono">{item.command}</span>
            <span className="tail__exit">
              {running ? 'running' : failed ? `exit ${item.exit_code}` : 'ok'}
            </span>
          </summary>
          {output ? <pre className="mono">{trim(output)}</pre> : null}
        </details>
      );
    }
    case 'file_change': {
      const changes = Array.isArray(item.changes)
        ? (item.changes as Array<Record<string, unknown>>)
        : [];
      const names = changes
        .map((change) => String(change.path ?? change.file ?? ''))
        .filter(Boolean)
        .map((path) => path.split('/').slice(-2).join('/'));
      return (
        <p className="tail__system">
          Edited{' '}
          {names.length
            ? names.map((name, index) => (
                <span key={`${name}-${index}`}>
                  {index > 0 ? ', ' : ''}
                  <span className="mono">{name}</span>
                </span>
              ))
            : 'files'}
        </p>
      );
    }
    default:
      return (
        <p className="tail__raw mono">
          {item.type}: {JSON.stringify(item).slice(0, 300)}
        </p>
      );
  }
}
