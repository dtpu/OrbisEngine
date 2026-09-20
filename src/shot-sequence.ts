// A clip with cuts is reconstructed shot by shot: each shot is its own candidate world on its own
// local clock. scripts/package_shot_sequence.py stitches those candidates back onto the ORIGINAL
// source clock and writes ONE people.json carrying a top-level `shots` block, so the viewer plays
// the clip straight through and swaps world + cast at each cut.
//
// This module is the time half of that contract, and nothing else: which shot owns a source time.
// From the packager's docstring (scripts/package_shot_sequence.py, "VIEWER CONTRACT"):
//
//   `shots` is ordered, non-overlapping and closed-open: shot k is active for `t` in
//   [shots[k].sourceStart, shots[k].sourceEnd). Outside every window NO shot is active: that source
//   time was never reconstructed. Show the gap as such - do not stretch a neighbour over it.
//
// Pure arithmetic and validation: no three.js, no DOM, no fetch, so scripts/test-shot-sequence.ts
// can check the boundaries a viewer would otherwise only reach by seeking a real package.

/** One entry of the manifest's top-level `shots` list, as the packager writes it. */
export interface ShotWindow {
  index: number;
  /** Viewer URL of this shot's own `.spz`. */
  world: string;
  /** Seconds on the ORIGINAL clip; the window is [sourceStart, sourceEnd). */
  sourceStart: number;
  sourceEnd: number;
  /** Package-relative path of this shot's placement.json, or null when the solve has none. */
  placement: string | null;
  /** Package-relative path of this shot's cameras.json. Each shot is its own SfM solve. */
  cameras: string | null;
  /** Merged id of this shot's primary person, or null when the shot has no cast. */
  primary: string | null;
  /** Merged ids of the people confined to this shot by their `visibleSampleRuns`. */
  people: string[];
  candidate: string | null;
  sharedPlacement: { bodyHeightUnits?: number; feetY?: number } | null;
  sharedScale: unknown;
  floorFit: unknown;
}

export interface ShotGap {
  start: number;
  end: number;
}

const FINITE = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
/** Windows are compared with this slack so a packaged boundary is not split by float noise. */
export const SHOT_TIME_EPS = 1e-6;

/**
 * Read and check a manifest's `shots`. A package whose windows overlap, run backwards or name no
 * world is a packaging fault, not something to render half of: this throws rather than guessing.
 * Returns null when the manifest carries no `shots` at all, which is every scene that exists today.
 */
export function parseShots(raw: unknown): ShotWindow[] | null {
  if (raw == null) return null;
  if (!Array.isArray(raw)) throw new Error('manifest `shots` must be a list');
  if (!raw.length) throw new Error('manifest `shots` is empty; a sequence needs at least one shot');
  const shots: ShotWindow[] = [];
  raw.forEach((entry, i) => {
    if (!entry || typeof entry !== 'object') throw new Error(`shot ${i}: not an object`);
    const s = entry as Record<string, unknown>;
    const start = s.sourceStart,
      end = s.sourceEnd;
    if (!FINITE(start) || !FINITE(end) || !(end > start)) {
      throw new Error(
        `shot ${i}: sourceStart ${start} .. sourceEnd ${end} is not a forward window`,
      );
    }
    if (typeof s.world !== 'string' || !s.world) throw new Error(`shot ${i}: no world URL`);
    const previous = shots[shots.length - 1];
    if (previous && start < previous.sourceEnd - SHOT_TIME_EPS) {
      throw new Error(
        `shot ${i}: starts at ${start} s, inside shot ${previous.index} which ends at ` +
          `${previous.sourceEnd} s; shots must be ordered and must not overlap`,
      );
    }
    shots.push({
      index: FINITE(s.index) ? s.index : i,
      world: s.world,
      sourceStart: start,
      sourceEnd: end,
      placement: typeof s.placement === 'string' ? s.placement : null,
      cameras: typeof s.cameras === 'string' ? s.cameras : null,
      primary: typeof s.primary === 'string' ? s.primary : null,
      people: Array.isArray(s.people)
        ? s.people.filter((p): p is string => typeof p === 'string')
        : [],
      candidate: typeof s.candidate === 'string' ? s.candidate : null,
      sharedPlacement:
        s.sharedPlacement && typeof s.sharedPlacement === 'object'
          ? (s.sharedPlacement as ShotWindow['sharedPlacement'])
          : null,
      sharedScale: s.sharedScale ?? null,
      floorFit: s.floorFit ?? null,
    });
  });
  return shots;
}

/**
 * Index of the shot that owns source time `t`, or -1 when none does. Closed-open: a cut belongs to
 * the shot that starts at it, never to the one that ends there. Binary search, so a seek anywhere on
 * the clip costs the same as a frame step.
 */
export function activeShotIndex(
  shots: readonly ShotWindow[] | null | undefined,
  t: number,
): number {
  if (!shots?.length || !FINITE(t)) return -1;
  let lo = 0,
    hi = shots.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const shot = shots[mid];
    if (t < shot.sourceStart) hi = mid - 1;
    else if (t >= shot.sourceEnd) lo = mid + 1;
    else return mid;
  }
  return -1;
}

/**
 * A lookup that remembers where it was. applyTime() calls it every frame, so the common answer -
 * "the same shot as last frame" - costs one comparison; a seek, forwards or backwards, falls back to
 * the binary search. The memory is an optimisation only: the answer never depends on it.
 */
export function createShotLookup(
  shots: readonly ShotWindow[] | null | undefined,
): (t: number) => number {
  let last = -1;
  return (t: number) => {
    if (!shots?.length || !FINITE(t)) return -1;
    if (last >= 0 && last < shots.length) {
      const shot = shots[last];
      if (t >= shot.sourceStart && t < shot.sourceEnd) return last;
    }
    last = activeShotIndex(shots, t);
    return last;
  };
}

/**
 * The source times no shot covers, on a clip of `duration` seconds. These are the packager's own
 * `shotSequence.gaps`: stretches that were never reconstructed, which the viewer must show as such.
 */
export function shotGaps(
  shots: readonly ShotWindow[] | null | undefined,
  duration: number,
): ShotGap[] {
  const gaps: ShotGap[] = [];
  // no shots at all is not a sequence with one enormous gap: it is an ordinary scene
  if (!shots?.length || !FINITE(duration) || duration <= 0) return gaps;
  let cursor = 0;
  for (const shot of shots) {
    if (shot.sourceStart - cursor > SHOT_TIME_EPS)
      gaps.push({ start: cursor, end: shot.sourceStart });
    cursor = Math.max(cursor, shot.sourceEnd);
  }
  if (duration - cursor > SHOT_TIME_EPS) gaps.push({ start: cursor, end: duration });
  return gaps;
}

/** Resolve a package-relative path (`shots/00/cameras.json`) against the people.json's own folder. */
export function shotAssetUrl(base: string, relative: string | null | undefined): string | null {
  if (!relative) return null;
  if (/^(https?:)?\/\//.test(relative) || relative.startsWith('/')) return relative;
  return (base || '') + relative;
}
