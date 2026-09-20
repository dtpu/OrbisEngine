import type { GraphOptions } from './types';

/** A stage blocked because an upstream stage failed never ran; it is not itself broken. */
export function isWaitingOnUpstream(status: string, blockedReason?: string | null): boolean {
  return status === 'blocked' && /^dependency did not finish/i.test(blockedReason ?? '');
}

/** Plain words for a status; the raw enum is what the API speaks, not what an operator reads. */
export function statusLabel(status: string, blockedReason?: string | null): string {
  if (isWaitingOnUpstream(status, blockedReason)) return 'never ran';
  switch (status) {
    case 'waiting_human':
      return 'needs you';
    case 'waiting_agent':
      return 'agent reviewing';
    case 'succeeded':
      return 'done';
    default:
      return status.replaceAll('_', ' ');
  }
}

/** Collapse the twelve stage statuses into the six tones the palette distinguishes. */
export function statusTone(
  status: string,
  blockedReason?: string | null,
): 'idle' | 'active' | 'attention' | 'good' | 'bad' | 'muted' {
  if (isWaitingOnUpstream(status, blockedReason)) return 'muted';
  switch (status) {
    case 'running':
    case 'ready':
    case 'waiting_agent':
      return 'active';
    case 'waiting_human':
    case 'paused':
      return 'attention';
    case 'succeeded':
      return 'good';
    case 'failed':
    case 'blocked':
    case 'unknown':
      return 'bad';
    case 'canceled':
    case 'skipped':
      return 'muted';
    default:
      return 'idle';
  }
}

export function formatBytes(bytes: number): string {
  if (bytes === 0) return 'empty';
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB'];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}

function parse(value?: string | null): number | null {
  if (!value) return null;
  // The API emits naive ISO timestamps that are UTC; pin them so the browser does not read local.
  const iso = /[zZ]|[+-]\d\d:\d\d$/.test(value) ? value : `${value}Z`;
  const time = Date.parse(iso);
  return Number.isFinite(time) ? time : null;
}

export function formatDuration(
  start?: string | null,
  end?: string | null,
  now = Date.now(),
): string {
  const from = parse(start);
  if (from === null) return '';
  const to = parse(end) ?? now;
  const seconds = Math.max(0, Math.round((to - from) / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${String(seconds % 60).padStart(2, '0')}s`;
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, '0')}m`;
}

const clock = new Intl.DateTimeFormat(undefined, {
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
});
const day = new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric' });

/** "14:05" today, "Sep 19 14:05" otherwise. Operators glance at this dozens of times an hour. */
export function formatWhen(value?: string | null, now = Date.now()): string {
  const time = parse(value);
  if (time === null) return '—';
  const date = new Date(time);
  const sameDay = new Date(now).toDateString() === date.toDateString();
  return sameDay ? clock.format(date) : `${day.format(date)} ${clock.format(date)}`;
}

const exactClock = new Intl.DateTimeFormat(undefined, {
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hour12: false,
});

/** "01:03:12" today, "Sep 19 01:03:12" otherwise: the precision an attempt timeline needs. */
export function formatWhenExact(value?: string | null, now = Date.now()): string {
  const time = parse(value);
  if (time === null) return '—';
  const date = new Date(time);
  const sameDay = new Date(now).toDateString() === date.toDateString();
  return sameDay ? exactClock.format(date) : `${day.format(date)} ${exactClock.format(date)}`;
}

/** Milliseconds between two timestamps, or null when either is missing or unparsable. */
export function elapsedMs(
  start?: string | null,
  end?: string | null,
  now = Date.now(),
): number | null {
  const from = parse(start);
  if (from === null) return null;
  return Math.max(0, (parse(end) ?? now) - from);
}

export function formatMs(ms: number): string {
  const seconds = Math.round(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${String(seconds % 60).padStart(2, '0')}s`;
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, '0')}m`;
}

/** Statuses during which work is happening and a clock should tick. */
export function isInFlight(status: string): boolean {
  return status === 'running' || status === 'waiting_agent' || status === 'ready';
}

export function formatAgo(value?: string | null, now = Date.now()): string {
  const time = parse(value);
  if (time === null) return '';
  const seconds = Math.max(0, Math.round((now - time) / 1000));
  if (seconds < 45) return 'just now';
  if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)} h ago`;
  return `${Math.round(seconds / 86400)} d ago`;
}

export const WORLD_MODES: Record<GraphOptions['marble'], string> = {
  video: 'World from cleaned video',
  image: 'World from cleaned first frame',
  multi: 'World from selected frames',
  both: 'World from video and first frame',
  none: 'No generated world',
};

/** One sentence describing what a run was asked to do. */
export function describeOptions(options: GraphOptions): string {
  const parts: string[] = [];
  const people = options.all_people || options.people > 1 ? options.people : 1;
  parts.push(
    options.all_people ? `up to ${people} people` : people === 1 ? '1 person' : `${people} people`,
  );
  parts.push(
    {
      video: 'world from video',
      image: 'world from first frame',
      multi: 'world from selected frames',
      both: 'world from video and frame',
      none: 'no world',
    }[options.marble],
  );
  if (options.objects) parts.push('objects');
  if (options.reviewed_audio) parts.push('reviewed audio');
  if (options.finetune) parts.push('fine-tune');
  return parts.join(', ');
}

/** How a reviewing agent's verdict reads to the operator. */
export function verdictLabel(verdict: string): string {
  switch (verdict) {
    case 'pass':
      return 'agent passed it';
    case 'retry':
      return 'agent asked for a retry';
    case 'needs_human':
      return 'agent asked you';
    default:
      return `agent: ${verdict.replaceAll('_', ' ')}`;
  }
}

export function verdictTone(verdict: string): 'good' | 'active' | 'attention' | 'idle' {
  switch (verdict) {
    case 'pass':
      return 'good';
    case 'retry':
      return 'active';
    case 'needs_human':
      return 'attention';
    default:
      return 'idle';
  }
}

/**
 * Who an automated decision came from. `decidedBy` is `agent:<kind>:<model>:<sandbox>`, e.g.
 * `agent:codex:gpt-6-astra:danger-full-access`. The sandbox segment tells the operator how much
 * host access the reviewing agent had while it judged (and possibly edited) the outputs.
 */
export function describeDecider(decidedBy: string): {
  who: string;
  sandbox: string | null;
  unsandboxed: boolean;
} {
  const [kind, harness, model, sandbox] = decidedBy.split(':');
  if (kind !== 'agent' || !harness) return { who: decidedBy, sandbox: null, unsandboxed: false };
  const who = model ? `${harness} (${model})` : harness;
  if (!sandbox) return { who, sandbox: null, unsandboxed: false };
  const label =
    {
      'danger-full-access': 'full host access, no sandbox',
      'workspace-write': 'sandboxed, workspace write',
      'read-only': 'sandboxed, read-only',
    }[sandbox] ?? sandbox.replaceAll('-', ' ');
  return { who, sandbox: label, unsandboxed: sandbox === 'danger-full-access' };
}

const RETRY_CAP = /^agent asked for retry #(\d+), over the cap:\s*/i;

/** Split a gate's blockedReason into the agent's question and whether it hit its retry cap. */
export function parseAgentQuestion(reason: string): { question: string; overCap: number | null } {
  const match = reason.match(RETRY_CAP);
  if (!match) return { question: reason, overCap: null };
  return { question: reason.slice(match[0].length), overCap: Number(match[1]) };
}

export function shortHash(value: string, length = 12): string {
  return value.length > length ? value.slice(0, length) : value;
}

/** Artifact ids are `artifact:<64 hex>`; the hash tail is noise on screen. */
export function shortId(value: string): string {
  const [head, ...rest] = value.split(':');
  const tail = rest.join(':');
  if (!tail) return value;
  return /^[0-9a-f]{24,}$/.test(tail) ? `${head}:${tail.slice(0, 8)}` : value;
}
