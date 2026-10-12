'use client';

import Link from 'next/link';
import { useCallback, useEffect, useState } from 'react';
import { StageStrip } from '@/components/stage-strip';
import { Status } from '@/components/status';
import { useNow } from '@/components/use-now';
import { listRuns } from '@/lib/client';
import { CreateRun } from '@/components/create-run';
import { describeOptions, formatMs, formatWhen, isWaitingOnUpstream } from '@/lib/format';
import { activeMs, countByStatus, isExecuting, stagesWaitingOnYou } from '@/lib/graph';
import type { PipelineRun } from '@/lib/types';

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
  const tally: { key: string; tone: string; text: string }[] = [
    { key: 'done', tone: 'done', text: `${done}/${run.nodes.length} done` },
    ...(running ? [{ key: 'running', tone: 'active', text: `${running} running` }] : []),
    ...(counts.failed ? [{ key: 'failed', tone: 'bad', text: `${counts.failed} failed` }] : []),
    ...(blocked > 0 ? [{ key: 'blocked', tone: 'bad', text: `${blocked} blocked` }] : []),
    ...(neverRan ? [{ key: 'never', tone: 'idle', text: `${neverRan} never ran` }] : []),
  ];

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
        <span className="run__tally">
          {tally.map((count) => (
            <span key={count.key} className={`run__count run__count--${count.tone}`}>
              {count.text}
            </span>
          ))}
        </span>
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
          <section className="waiting-on-you" aria-labelledby="gates-title">
            <div className="section-head">
              <h2 className="section-title" id="gates-title">
                Waiting on you
              </h2>
              <span className="count">{gates.length}</span>
            </div>
            <div className="gates">
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
          </section>
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
              <span className="right">Started</span>
              <span>State</span>
            </div>
            {parents.map((parent) => (
              <div className="runs__group" key={parent.id}>
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
