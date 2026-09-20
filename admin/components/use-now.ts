'use client';

import { useEffect, useState } from 'react';

/**
 * False during the server render and the first client render, true afterwards.
 *
 * A subtree that reads localStorage, polls the API or formats a time in the viewer's locale
 * cannot produce the same HTML on both sides, so it waits for this rather than hydrating a
 * guess. React refuses to patch up a mismatch, which is how one ends up with a control that is
 * disabled in the markup and enabled in the DOM.
 */
export function useMounted(): boolean {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return mounted;
}

/** A clock that re-renders on an interval so elapsed times tick while a stage runs. */
export function useNow(intervalMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(timer);
  }, [intervalMs]);
  return now;
}
