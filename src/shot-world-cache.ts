// Residency for a shot sequence's worlds and the per-world state derived from them.
//
// Each shot in a package has its own Marble `.spz`, and everything the viewer derives from a world -
// its floor height, its floor map, its walk grid, its clamp box, the cast scale that shot's people
// were solved with, its own start pose - is state of THAT world, not of the scene. SplatMesh
// construction is asynchronous and applyTime() is not, so a cut cannot be the moment a world starts
// loading: the active shot and a small lookahead are kept resident and everything else is released.
//
// Keyed by shot index, least-recently-active evicted first, never evicting the active shot or a load
// still in flight. Nothing here knows about three.js or the DOM: `load` and `release` are the
// caller's, so scripts/test-shot-world-cache.ts drives it with plain objects.

export interface ShotWorldCacheOptions<T> {
  /** How many shots may stay resident at once. The viewer keeps about four worlds in memory. */
  limit?: number;
  /** Build the state for a shot. May be async (a SplatMesh has to finish initialising). */
  load: (index: number) => T | Promise<T>;
  /** Free an evicted shot's state. Called once per evicted entry. */
  release?: (value: T, index: number) => void;
}

export interface ShotWorldCache<T> {
  /** Load `index` if needed, mark it most recently used, and return its state. */
  ensure(index: number): Promise<T>;
  /** The state for `index` if it is already resident and finished loading, else undefined. */
  peek(index: number): T | undefined;
  /** True while `index` is loading. */
  pending(index: number): boolean;
  /**
   * Make `index` active and start the next `lookahead` shots loading, then release everything the
   * limit has no room for. Resolves with the active shot's state.
   */
  keep(index: number, lookahead?: number, total?: number): Promise<T>;
  /** Resident shot indices, most recently used first. */
  resident(): number[];
  /** Release everything. */
  clear(): void;
}

export function createShotWorldCache<T>(options: ShotWorldCacheOptions<T>): ShotWorldCache<T> {
  const limit = Math.max(1, Math.floor(options.limit ?? 4));
  const ready = new Map<number, T>();
  const loading = new Map<number, Promise<T>>();
  // loads that were in flight when clear() ran: their results are released, never stored
  const cleared = new Set<Promise<T>>();
  // most recently used LAST, so the head of this list is the first candidate for eviction
  const used: number[] = [];
  let active = -1;

  const touch = (index: number) => {
    const at = used.indexOf(index);
    if (at >= 0) used.splice(at, 1);
    used.push(index);
  };

  function start(index: number): Promise<T> {
    const already = loading.get(index);
    if (already) return already;
    if (ready.has(index)) return Promise.resolve(ready.get(index)!);
    const promise = (async () => options.load(index))().then(
      (value) => {
        loading.delete(index);
        // a clear() while this was in flight must not resurrect the entry
        if (!cleared.has(promise)) {
          ready.set(index, value);
          touch(index);
        } else options.release?.(value, index);
        return value;
      },
      (error) => {
        loading.delete(index);
        throw error;
      },
    );
    loading.set(index, promise);
    touch(index);
    return promise;
  }

  function evict() {
    for (let i = 0; i < used.length && ready.size + loading.size > limit;) {
      const index = used[i];
      if (index === active || loading.has(index)) {
        i++;
        continue;
      }
      used.splice(i, 1);
      const value = ready.get(index);
      ready.delete(index);
      if (value !== undefined) options.release?.(value, index);
    }
  }

  return {
    async ensure(index) {
      if (ready.has(index)) {
        touch(index);
        return ready.get(index)!;
      }
      const value = await start(index);
      evict();
      return value;
    },
    peek(index) {
      return ready.get(index);
    },
    pending(index) {
      return loading.has(index);
    },
    async keep(index, lookahead = 1, total = Infinity) {
      active = index;
      const value = ready.has(index) ? (touch(index), ready.get(index)!) : await start(index);
      active = index;
      evict(); // make room before the lookahead, so a stale world cannot crowd out the next one
      // the lookahead is fire-and-forget: a cut must never wait on the shot after next
      for (let i = 1; i <= Math.max(0, lookahead); i++) {
        const next = index + i;
        if (next >= total) break;
        if (ready.size + loading.size >= limit) break;
        void start(next).catch(() => {});
      }
      evict();
      return value;
    },
    resident() {
      return [...used].reverse().filter((i) => ready.has(i));
    },
    clear() {
      for (const promise of loading.values()) cleared.add(promise);
      for (const [index, value] of ready) options.release?.(value, index);
      ready.clear();
      used.length = 0;
      active = -1;
    },
  };
}
