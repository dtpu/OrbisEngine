/**
 * Headline facts pulled out of a stage's JSON report.
 *
 * A report tile that says `{ }` tells an operator nothing. The same tile saying "1 shot, 14.9s,
 * 1280x720" answers the question they opened the graph to ask. The pipeline's reports are small
 * and flat, so the facts are read straight from the bytes rather than from a schema: a short list
 * of keys worth showing, in the order they read best, and nothing invented when a key is absent.
 */

export interface Fact {
  value: string;
  label: string;
}

function round(value: number, places = 1): string {
  const rounded = Number(value.toFixed(places));
  return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(places);
}

function seconds(value: number): string {
  if (value < 60) return `${round(value)}s`;
  const minutes = Math.floor(value / 60);
  return `${minutes}m ${String(Math.round(value % 60)).padStart(2, '0')}s`;
}

function count(value: number, singular: string, plural = `${singular}s`): Fact {
  return { value: String(value), label: value === 1 ? singular : plural };
}

function bytes(value: number): string {
  if (value < 1024) return `${value} B`;
  const units = ['KB', 'MB', 'GB'];
  let scaled = value / 1024;
  let unit = 0;
  while (scaled >= 1024 && unit < units.length - 1) {
    scaled /= 1024;
    unit += 1;
  }
  return `${scaled < 10 ? scaled.toFixed(1) : Math.round(scaled)} ${units[unit]}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** Keys worth a tile, in reading order. Each returns a fact only when its data is really there. */
const READERS: Array<(data: Record<string, unknown>) => Fact | null> = [
  // Shot selection: how much of the clip survived and whether it is one continuous take.
  (data) => {
    const shots = num(data.shotCount) ?? (Array.isArray(data.shots) ? data.shots.length : null);
    return shots === null ? null : count(shots, 'shot');
  },
  // Frames a stage actually processed, not the frames it was handed.
  (data) => {
    const inpainted = num(data.inpainted);
    const frames = num(data.frames);
    if (inpainted !== null && frames !== null && inpainted !== frames) {
      return { value: `${inpainted}/${frames}`, label: 'frames filled' };
    }
    return frames === null ? null : count(frames, 'frame');
  },
  (data) => {
    const source = isRecord(data.source) ? data.source : data;
    const duration = num(source.duration) ?? num(data.durationSeconds);
    return duration === null ? null : { value: seconds(duration), label: 'of footage' };
  },
  (data) => {
    const source = isRecord(data.source) ? data.source : data;
    const width = num(source.width);
    const height = num(source.height);
    return width === null || height === null
      ? null
      : { value: `${width}x${height}`, label: 'pixels' };
  },
  (data) => {
    const people =
      num(data.peopleCount) ?? (Array.isArray(data.people) ? data.people.length : null);
    return people === null ? null : count(people, 'person', 'people');
  },
  (data) => {
    const tracks = num(data.trackCount) ?? (Array.isArray(data.tracks) ? data.tracks.length : null);
    return tracks === null ? null : count(tracks, 'track');
  },
  (data) => {
    const objects =
      num(data.objectCount) ?? (Array.isArray(data.objects) ? data.objects.length : null);
    return objects === null ? null : count(objects, 'object');
  },
  (data) => {
    const cameras =
      num(data.cameraCount) ?? (Array.isArray(data.cameras) ? data.cameras.length : null);
    return cameras === null ? null : count(cameras, 'camera');
  },
  (data) => {
    const coverage = num(data.maskCoverage) ?? num(data.averageMaskCoverage);
    return coverage === null
      ? null
      : { value: `${round(coverage <= 1 ? coverage * 100 : coverage)}%`, label: 'masked' };
  },
  (data) => {
    const scale = num(data.scale) ?? num(data.metresPerUnit);
    return scale === null ? null : { value: round(scale, 3), label: 'scale' };
  },
  // How long the stage's own work took, as the stage measured it.
  (data) => {
    const taken = num(data.seconds) ?? num(data.elapsed);
    return taken === null ? null : { value: seconds(taken), label: 'compute' };
  },
  (data) => (typeof data.gpu === 'string' ? { value: data.gpu, label: 'GPU' } : null),
  (data) => {
    const size = num(data.bytes);
    return size === null ? null : { value: bytes(size), label: 'source' };
  },
  (data) => {
    const points = num(data.pointCount) ?? num(data.points);
    return points === null
      ? null
      : { value: points >= 1000 ? `${round(points / 1000)}k` : String(points), label: 'points' };
  },
];

const MAX_FACTS = 3;

/** Keys whose prose is the point of the report, tried before any other string value. */
const DESCRIPTIVE = ['space', 'summary', 'description', 'prompt', 'caption', 'title', 'appearance'];
const DESCRIPTIVE_MIN = 40;

function humanise(key: string): string {
  return key
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replaceAll('_', ' ')
    .toLowerCase();
}

function scalarFact(key: string, value: unknown): Fact | null {
  const label = humanise(key);
  const asNumber = num(value);
  if (asNumber !== null) {
    const shown = Number.isInteger(asNumber) ? String(asNumber) : round(asNumber, 2);
    return { value: shown, label };
  }
  if (typeof value === 'boolean') return { value: value ? 'yes' : 'no', label };
  if (typeof value === 'string' && value.length > 0 && value.length <= 24) {
    return { value, label };
  }
  return null;
}

function known(data: Record<string, unknown>): Fact[] {
  const facts: Fact[] = [];
  for (const read of READERS) {
    const fact = read(data);
    if (fact) facts.push(fact);
    if (facts.length === MAX_FACTS) break;
  }
  return facts;
}

/** The prose a report is really made of: a world prompt is paragraphs, not measurements. */
function prose(data: Record<string, unknown>): string | null {
  for (const key of DESCRIPTIVE) {
    const value = data[key];
    if (typeof value === 'string' && value.trim().length >= DESCRIPTIVE_MIN) return value;
  }
  for (const value of Object.values(data)) {
    if (typeof value === 'string' && value.trim().length >= DESCRIPTIVE_MIN * 2) return value;
  }
  return null;
}

/** Anything scalar at the top level, so an unfamiliar report still shows its own numbers. */
function anything(data: Record<string, unknown>): Fact[] {
  const facts: Fact[] = [];
  for (const [key, value] of Object.entries(data)) {
    if (key === 'schema' || key === 'sha256') continue;
    const fact = scalarFact(key, value);
    if (fact) facts.push(fact);
    if (facts.length === MAX_FACTS) break;
  }
  return facts;
}

export type Summary = { kind: 'facts'; facts: Fact[] } | { kind: 'text'; text: string };

/**
 * What a report tile shows. Measurements when the report has them, its own prose when it is
 * written rather than measured, and failing both, whatever scalars it does carry. An `error`
 * field wins outright: a report that recorded a failure should say so before anything else.
 */
export function summarize(parsed: unknown): Summary | null {
  if (!isRecord(parsed)) return null;
  if (typeof parsed.error === 'string' && parsed.error.trim()) {
    return { kind: 'facts', facts: [{ value: 'error', label: parsed.error.trim().slice(0, 60) }] };
  }
  const facts = known(parsed);
  if (facts.length >= 2) return { kind: 'facts', facts };
  const text = prose(parsed);
  if (text) return { kind: 'text', text: excerpt(text) };
  if (facts.length === 1) return { kind: 'facts', facts };
  const rest = anything(parsed);
  return rest.length ? { kind: 'facts', facts: rest } : null;
}

/** The opening of a text artifact, collapsed to one readable run of words. */
export function excerpt(text: string, limit = 220): string {
  const trimmed = text.replace(/\s+/g, ' ').trim();
  return trimmed.length > limit ? `${trimmed.slice(0, limit)}…` : trimmed;
}
