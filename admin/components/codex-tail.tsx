'use client';

import Link from 'next/link';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useMounted } from '@/components/use-now';
import { collapse, TranscriptRow } from '@/components/transcript';
import { listReviewTranscripts, readTranscript, sendMessage } from '@/lib/client';
import { formatWhen } from '@/lib/format';
import type { ReviewTranscript, TranscriptEvent } from '@/lib/types';

const TAIL_INTERVAL = 1500;
const LIST_INTERVAL = 5000;
const OPEN_KEY = 'wander-admin-codex-open';

interface TailState {
  key: string;
  events: TranscriptEvent[];
  finished: boolean;
  error?: string;
}

function keyOf(item: ReviewTranscript): string {
  return `${item.runId ?? ''}/${item.nodeId}/${item.attemptId}`;
}

/**
 * One run's Codex log. It follows whatever review the agent is writing for this run right now,
 * falls back to the most recent finished one, and lets the operator drop the agent a message
 * that it reads at its next review. Scoped to the run whose page it sits on: a log belongs to
 * the run you are looking at, and answering it sends a message about one of that run's stages.
 */
export function CodexTail({ runId }: { runId: string }) {
  const mounted = useMounted();
  const [open, setOpen] = useState(true);
  const [transcripts, setTranscripts] = useState<ReviewTranscript[]>([]);
  const [available, setAvailable] = useState(true);
  const [chosen, setChosen] = useState<string | null>(null);
  const [tail, setTail] = useState<TailState | null>(null);
  const [draft, setDraft] = useState('');
  const [sending, setSending] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [follow, setFollow] = useState(true);
  const stream = useRef<HTMLDivElement>(null);

  // Below 1100px the rail floats over the page instead of sitting beside it, so it starts
  // collapsed there. A stored choice still wins: only the default depends on the width.
  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(OPEN_KEY);
      if (stored) setOpen(stored !== 'closed');
      else setOpen(window.innerWidth > 1100);
    } catch {
      // Storage may be blocked; the sidebar simply starts open.
    }
  }, []);
  function toggle() {
    setOpen((current) => {
      try {
        window.localStorage.setItem(OPEN_KEY, current ? 'closed' : 'open');
      } catch {
        // Fine to forget.
      }
      return !current;
    });
  }

  const reload = useCallback(async () => {
    try {
      const listing = await listReviewTranscripts(runId);
      setTranscripts(listing.transcripts);
      setAvailable(listing.available);
    } catch {
      // The API may be down; keep whatever we last saw.
    }
  }, [runId]);

  useEffect(() => {
    void reload();
    const timer = setInterval(reload, LIST_INTERVAL);
    return () => clearInterval(timer);
  }, [reload]);

  const current = useMemo(() => {
    if (chosen) return transcripts.find((item) => keyOf(item) === chosen) ?? null;
    return transcripts.find((item) => !item.finished) ?? transcripts[0] ?? null;
  }, [chosen, transcripts]);
  const currentKey = current ? keyOf(current) : null;

  // Tail the chosen transcript; a shrunken file (re-review) comes back with reset and restarts.
  useEffect(() => {
    if (!current || !current.runId) {
      setTail(null);
      return;
    }
    const key = keyOf(current);
    const { runId, nodeId, attemptId } = current;
    let cancelled = false;
    let offset = 0;
    let done = false;
    setTail({ key, events: [], finished: false });
    const pull = async () => {
      if (cancelled || done) return;
      try {
        const chunk = await readTranscript(runId, nodeId, attemptId, offset);
        if (cancelled) return;
        offset = chunk.offset;
        setTail((state) => {
          const base = state && state.key === key && !chunk.reset ? state.events : [];
          return { key, events: [...base, ...chunk.events], finished: chunk.finished };
        });
        if (chunk.finished && chunk.offset >= chunk.size) done = true;
      } catch (cause) {
        if (!cancelled) {
          setTail((state) => (state ? { ...state, error: (cause as Error).message } : state));
        }
      }
    };
    void pull();
    const timer = setInterval(pull, TAIL_INTERVAL);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [currentKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const rows = useMemo(() => collapse(tail?.events ?? []), [tail?.events]);

  useEffect(() => {
    if (follow && stream.current) stream.current.scrollTop = stream.current.scrollHeight;
  }, [rows.length, follow, open]);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || sending || !current?.runId) return;
    setSending(true);
    setNote(null);
    try {
      await sendMessage(current.runId, { message: text, node_id: current.nodeId });
      setDraft('');
      setNote(`Sent. The agent reads it at its next review of ${current.nodeId}.`);
    } catch (cause) {
      setNote((cause as Error).message);
    } finally {
      setSending(false);
    }
  }

  const live = mounted && current ? !current.finished : false;
  // Until the mount, the server and the client agree on exactly one thing: nothing is loaded.
  const target = mounted ? current : null;

  return (
    <aside className={`tail${open ? '' : ' tail--closed'}`} aria-label="Codex log">
      <button
        type="button"
        className="tail__toggle"
        onClick={toggle}
        aria-expanded={open}
        title={open ? 'Hide the Codex log' : 'Show the Codex log'}
      >
        <span className={`tail__dot${live ? ' tail__dot--live' : ''}`} aria-hidden="true" />
        <span className="tail__toggle-label">Codex</span>
      </button>

      {open ? (
        <>
          <div className="tail__head">
            {target?.runId ? (
              <>
                <span className={`status status--${live ? 'active' : 'muted'}`}>
                  {live ? 'reviewing' : 'last review'}
                </span>
                <Link
                  className="mono"
                  href={`/runs/${encodeURIComponent(target.runId)}?stage=${encodeURIComponent(target.nodeId)}`}
                >
                  {target.nodeId}
                </Link>
                {live ? null : <span className="tail__where">{formatWhen(target.updatedAt)}</span>}
              </>
            ) : (
              <span className="tail__where">
                {!mounted
                  ? 'Loading…'
                  : available
                    ? 'No Codex review yet.'
                    : 'Transcripts are not available from this API.'}
              </span>
            )}
          </div>

          {mounted && transcripts.length > 1 ? (
            <select
              className="tail__pick"
              aria-label="Which review to follow"
              value={(mounted ? currentKey : null) ?? ''}
              onChange={(event) => setChosen(event.target.value || null)}
            >
              {transcripts.map((item) => (
                <option key={keyOf(item)} value={keyOf(item)}>
                  {item.finished ? '' : 'live: '}
                  {item.nodeId}, {formatWhen(item.updatedAt)}
                </option>
              ))}
            </select>
          ) : null}

          <div
            className="tail__stream"
            ref={stream}
            onScroll={(event) => {
              const element = event.currentTarget;
              setFollow(element.scrollHeight - element.scrollTop - element.clientHeight < 40);
            }}
          >
            {mounted && tail?.error ? <p className="tail__system">{tail.error}</p> : null}
            {mounted
              ? rows.map((event, index) => (
                  <TranscriptRow key={event.item?.id ?? `${event.type}-${index}`} event={event} />
                ))
              : null}
            {live && rows.length > 0 ? <p className="tail__system tail__cursor">▍</p> : null}
          </div>

          <form className="tail__compose" onSubmit={submit}>
            <input
              type="text"
              value={draft}
              disabled={!target?.runId}
              aria-label="Message the agent"
              title={note ?? 'The agent reads this at its next review.'}
              placeholder={
                target ? `Message the agent about ${target.nodeId}` : 'Message the agent'
              }
              onChange={(event) => setDraft(event.target.value)}
            />
            <button
              className="button button--small"
              type="submit"
              disabled={sending || !draft.trim() || !target?.runId}
            >
              Send
            </button>
            {note ? <span className="tail__note">{note}</span> : null}
          </form>
        </>
      ) : null}
    </aside>
  );
}
