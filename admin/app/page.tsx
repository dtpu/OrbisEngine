'use client';

import Link from 'next/link';
import { useCallback, useEffect, useRef, useState } from 'react';
import { StageStrip } from '@/components/stage-strip';
import { Status } from '@/components/status';
import { useNow } from '@/components/use-now';
import { listRuns, uploadRun } from '@/lib/client';
import {
  describeOptions,
  formatBytes,
  formatMs,
  formatWhen,
  isWaitingOnUpstream,
} from '@/lib/format';
import { activeMs, countByStatus, isExecuting, stagesWaitingOnYou } from '@/lib/graph';
import type { GraphOptions, PipelineRun } from '@/lib/types';

/**
 * Every clip goes through the whole pipeline at full effort; the reviewing agent steers from
 * there. Reviewed audio stays off because it needs a `reviewed_audio_config` run input that an
 * upload does not carry, so the stage would wait forever.
 */
const FULL_EFFORT: GraphOptions = {
  marble: 'video',
  people: 16,
  all_people: true,
  objects: true,
  reviewed_audio: false,
  finetune: true,
};

function CreateRun({ onCreated }: { onCreated: (runId: string) => void }) {
  const [requestedId, setRequestedId] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [over, setOver] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!file || progress !== null) return;
    setError(null);
    setProgress(0);
    try {
      const run = await uploadRun({
        file,
        options: FULL_EFFORT,
        requestedId: requestedId.trim() || undefined,
        onProgress: setProgress,
      });
      setFile(null);
      setRequestedId('');
      if (fileInput.current) fileInput.current.value = '';
      onCreated(run.id);
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setProgress(null);
    }
  }

  const busy = progress !== null;
  const percent = Math.round((progress ?? 0) * 100);

  return (
    <form className="create" onSubmit={submit} aria-labelledby="create-title">
      <h2 className="create__title" id="create-title">
        Start a run
      </h2>

      <label
        className={`drop${over ? ' drop--over' : ''}`}
        onDragOver={(event) => {
          event.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(event) => {
          event.preventDefault();
          setOver(false);
          const dropped = event.dataTransfer.files?.[0];
          if (dropped && !busy) setFile(dropped);
        }}
      >
        <input
          ref={fileInput}
          type="file"
          accept="video/*"
          required={!file}
          disabled={busy}
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
        />
        {file ? (
          <>
            <span className="drop__file">{file.name}</span>
            <span className="drop__hint">{formatBytes(file.size)}. Click to change.</span>
          </>
        ) : (
          <>
            <strong>Choose a source clip</strong>
            <span className="drop__hint">or drop a video file here</span>
          </>
        )}
      </label>

      <div className="field">
        <label htmlFor="requested-id">Run name</label>
        <input
          id="requested-id"
          type="text"
          className="mono"
          placeholder="generated when blank"
          value={requestedId}
          disabled={busy}
          onChange={(event) => setRequestedId(event.target.value)}
        />
      </div>

      {error ? (
        <p className="notice notice--error" role="alert">
          {error}
        </p>
      ) : null}

      {busy ? (
        <div className="field" aria-live="polite">
          <div className="bar">
            <span style={{ width: `${percent}%` }} />
          </div>
          <span className="field__hint">Uploading {percent}%</span>
        </div>
      ) : null}

      <button className="button button--primary" type="submit" disabled={!file || busy}>
        {busy ? 'Uploading…' : 'Create run'}
      </button>
      <p className="field__hint">
        Every clip gets the full pipeline: everyone in the shot, moving objects, a generated world
        and a fine-tune pass. The reviewing agent steers; you approve the cleaned input and the
        finished world.
      </p>
    </form>
  );
}

function RunRow({ run, shot, now }: { run: PipelineRun; shot: boolean; now: number }) {
  const counts = countByStatus(run);
  const executing = isExecuting(run);
  const ran = activeMs(run, now);
  const done = counts.succeeded ?? 0;
  const running = counts.running ?? 0;
  const waiting = stagesWaitingOnYou(run).length;
  const neverRan = run.nodes.filter((node) =>
    isWaitingOnUpstream(node.status, node.blockedReason),
  ).length;
  const blocked = (counts.blocked ?? 0) - neverRan;
  const tally = [
    `${done} of ${run.nodes.length} done`,
    running ? `${running} running` : null,
    counts.failed ? `${counts.failed} failed` : null,
    blocked > 0 ? `${blocked} blocked` : null,
    neverRan ? `${neverRan} never ran` : null,
  ]
    .filter(Boolean)
    .join(', ');

  return (
    <Link className={`run${shot ? ' run--shot' : ''}`} href={`/runs/${encodeURIComponent(run.id)}`}>
      <span className="run__name">
        <span className="run__id">
          {shot ? (
            <>
              <span className="shot">shot </span>
              {run.branchKey}
            </>
          ) : (
            run.id
          )}
        </span>
        {shot ? null : (
          <span className="run__what">{describeOptions(run.configuration.options)}</span>
        )}
      </span>
      <span className="run__progress">
        <StageStrip nodes={run.nodes} compact />
        <span className="run__tally">{tally}</span>
      </span>
      <span className="run__when">
        <span>{formatWhen(run.createdAt, now)}</span>
        {ran !== null ? (
          <span className={executing ? 'live' : undefined} title="Time stages spent executing">
            ran {formatMs(ran)}
          </span>
        ) : null}
      </span>
      <span className="run__state">
        <Status status={run.status} />
        {waiting > 0 ? (
          <span className="run__needs">
            {waiting === 1 ? '1 gate' : `${waiting} gates`} waiting on you
          </span>
        ) : null}
      </span>
    </Link>
  );
}

export default function Page() {
  const [runs, setRuns] = useState<PipelineRun[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const now = useNow(1000);

  const refresh = useCallback(async () => {
    try {
      setRuns(await listRuns());
      setError(null);
    } catch (cause) {
      setError((cause as Error).message);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = setInterval(refresh, 3000);
    return () => clearInterval(timer);
  }, [refresh]);

  const gates = (runs ?? []).flatMap((run) =>
    stagesWaitingOnYou(run).map((node) => ({ run, node })),
  );
  const parents = (runs ?? []).filter((run) => !run.parentRunId);

  return (
    <main className="page home" id="main">
      <section aria-labelledby="runs-title">
        {error ? (
          <p className="notice notice--error" role="alert" style={{ marginBottom: 20 }}>
            {error}
          </p>
        ) : null}

        {gates.length > 0 ? (
          <div className="gates" role="region" aria-label="Waiting on you">
            {gates.map(({ run, node }) => (
              <Link
                key={`${run.id}/${node.id}`}
                className="gate"
                href={`/runs/${encodeURIComponent(run.id)}?stage=${encodeURIComponent(node.id)}`}
              >
                <Status status={node.status} blockedReason={node.blockedReason} />
                <span className="gate__what">
                  <strong>{node.definition.title}</strong>
                  <span className="gate__where">
                    {' '}
                    in <span className="mono">{run.id}</span>
                  </span>
                </span>
                <span className="gate__cta">Review</span>
              </Link>
            ))}
          </div>
        ) : null}

        <div className="section-head">
          <h1 className="section-title" id="runs-title">
            Runs
          </h1>
          {runs ? <span className="count">{runs.length}</span> : null}
          <span className="live">Refreshes every 3 s</span>
        </div>

        {runs === null ? (
          <p className="empty">Loading runs…</p>
        ) : runs.length === 0 ? (
          <p className="empty">No runs yet. Choose a clip on the right to start one.</p>
        ) : (
          <div className="runs">
            <div className="runs__head" aria-hidden="true">
              <span>Run</span>
              <span>Stages</span>
              <span style={{ textAlign: 'right' }}>Started</span>
              <span style={{ textAlign: 'right' }}>State</span>
            </div>
            {parents.map((parent) => (
              <div key={parent.id}>
                <RunRow run={parent} shot={false} now={now} />
                {(runs ?? [])
                  .filter((run) => run.parentRunId === parent.id)
                  .map((child) => (
                    <RunRow key={child.id} run={child} shot now={now} />
                  ))}
              </div>
            ))}
          </div>
        )}
      </section>

      <CreateRun onCreated={refresh} />
    </main>
  );
}
