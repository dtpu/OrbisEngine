'use client';

import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { ArtifactViewer } from '@/components/artifact-viewer';
import { leadArtifact } from '@/lib/blocks';
import { GraphCanvas } from '@/components/graph-canvas';
import { Progress } from '@/components/progress';
import { StageStrip } from '@/components/stage-strip';
import { Status } from '@/components/status';
import { useNow } from '@/components/use-now';
import {
  approvalArtifacts,
  approve,
  command,
  listRuns,
  loadRun,
  reconcileClaim,
} from '@/lib/client';
import {
  describeDecider,
  describeOptions,
  elapsedMs,
  formatDuration,
  formatMs,
  formatWhen,
  formatWhenExact,
  isInFlight,
  isWaitingOnUpstream,
  parseAgentQuestion,
  shortHash,
  statusLabel,
  statusTone,
  verdictLabel,
  verdictTone,
} from '@/lib/format';
import {
  activeMs,
  dependenciesOf,
  isExecuting,
  orderStages,
  stageNeedingAttention,
  typicalDurations,
  unrecordedClaims,
} from '@/lib/graph';
import type {
  PipelineRun,
  QualityReview,
  ReconcileStatus,
  RunAttempt,
  RunCommand,
  RunNode,
} from '@/lib/types';

const UPSTREAM_NOTE = 'This stage never started: an earlier stage it depends on did not finish.';

const KIND_LABEL: Record<string, string> = {
  compute: 'compute stage',
  agent: 'agent stage',
  human: 'human gate',
  expand: 'fan-out stage',
  join: 'join stage',
};

function attemptsFor(run: PipelineRun, nodeId: string): RunAttempt[] {
  return (run.attempts ?? []).filter((attempt) => attempt.nodeId === nodeId);
}

/** The attempt an operator decision applies to: the selected one, else the newest. */
function decisionAttempt(run: PipelineRun, node: RunNode): RunAttempt | undefined {
  const attempts = attemptsFor(run, node.id);
  return (
    attempts.find((attempt) => attempt.id === node.selectedAttemptId) ??
    attempts[attempts.length - 1]
  );
}

const TERMINAL_RUN = new Set(['succeeded', 'failed', 'canceled']);

type View = 'ladder' | 'graph';
const VIEW_KEY = 'wander-admin-run-view';

function storedView(): View {
  try {
    return window.localStorage.getItem(VIEW_KEY) === 'graph' ? 'graph' : 'ladder';
  } catch {
    return 'ladder';
  }
}

/** The attempt whose clock a stage is currently on: the newest one that has started. */
function runningAttempt(run: PipelineRun, nodeId: string): RunAttempt | undefined {
  return [...attemptsFor(run, nodeId)].reverse().find((attempt) => attempt.startedAt);
}

/**
 * How far along an in-flight attempt is against the stage's typical duration, capped below
 * full so a bar never claims "done" before the pipeline does. Null when nothing has finished
 * this stage before; the bar is then indeterminate.
 */
function progressFraction(elapsed: number | null, typical: number | undefined): number | null {
  if (elapsed === null || typical === undefined || typical <= 0) return null;
  return Math.min(0.96, elapsed / typical);
}

/** Reviews for a stage, newest first. */
function reviewsFor(run: PipelineRun, nodeId: string): QualityReview[] {
  return (run.reviews ?? [])
    .filter((review) => review.nodeId === nodeId)
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

/** Attempts the reviewing agent produced by editing a stage's outputs end in ":agent". */
function isAgentRevision(attempt: RunAttempt): boolean {
  return attempt.id.endsWith(':agent');
}

function Review({ review, now }: { review: QualityReview; now: number }) {
  const parameters = review.proposedParameters;
  const hasParameters = parameters && Object.keys(parameters).length > 0;
  const decider = describeDecider(review.decidedBy);
  return (
    <div className={`review review--${verdictTone(review.verdict)}`}>
      <div className="review__head">
        <span className={`status status--${verdictTone(review.verdict)}`}>
          {verdictLabel(review.verdict)}
        </span>
        <span>{decider.who}</span>
        {decider.sandbox ? (
          <span className={decider.unsandboxed ? 'exposed' : undefined} title={review.decidedBy}>
            {decider.sandbox}
          </span>
        ) : null}
        <span className="mono">{formatWhen(review.createdAt, now)}</span>
        <span>
          on attempt <span className="mono">{review.attemptId.split(':').slice(-2).join(':')}</span>
        </span>
      </div>
      <p className="review__body">{review.rationale}</p>
      {review.hypothesis ? (
        <p className="review__extra">
          Next hypothesis: <span className="mono">{review.hypothesis}</span>
        </p>
      ) : null}
      {hasParameters ? (
        <p className="review__extra">
          Proposed parameters: <span className="mono">{JSON.stringify(parameters)}</span>
        </p>
      ) : null}
    </div>
  );
}

function Ladder({
  run,
  selectedId,
  onSelect,
  now,
  typical,
}: {
  run: PipelineRun;
  selectedId: string | null;
  onSelect: (id: string) => void;
  now: number;
  typical: Map<string, number>;
}) {
  const ordered = orderStages(run.nodes);
  const tiers = new Map<number, typeof ordered>();
  for (const entry of ordered) tiers.set(entry.depth, [...(tiers.get(entry.depth) ?? []), entry]);

  return (
    <nav className="ladder" aria-label="Stages in dependency order">
      <ol>
        {[...tiers.entries()].map(([depth, entries]) => (
          <li className="ladder__tier" key={depth}>
            <ol>
              {entries.map(({ node }) => {
                const tone = statusTone(node.status, node.blockedReason);
                const latest = reviewsFor(run, node.id)[0];
                const inFlight = isInFlight(node.status);
                const clock = inFlight ? runningAttempt(run, node.id) : undefined;
                const elapsed = clock ? elapsedMs(clock.startedAt, clock.finishedAt, now) : null;
                return (
                  <li key={node.id}>
                    <button
                      type="button"
                      className={`stage${node.id === selectedId ? ' stage--selected' : ''}`}
                      aria-current={node.id === selectedId ? 'true' : undefined}
                      onClick={() => onSelect(node.id)}
                    >
                      <span className={`stage__dot stage__dot--${tone}`} aria-hidden="true" />
                      <span className="stage__id">
                        {node.id}
                        <span className={`stage__state stage__state--${tone}`}>
                          {statusLabel(node.status, node.blockedReason)}
                        </span>
                        {elapsed !== null ? (
                          <span className="stage__elapsed">{formatMs(elapsed)}</span>
                        ) : null}
                      </span>
                      <span className="stage__title">{node.definition.title}</span>
                      {inFlight ? (
                        <Progress
                          fraction={progressFraction(elapsed, typical.get(node.id))}
                          label={`${node.id} in progress`}
                        />
                      ) : null}
                      {node.definition.kind === 'human' || latest ? (
                        <span className="stage__flags">
                          {node.definition.kind === 'human' ? <span>needs approval</span> : null}
                          {latest ? (
                            <span className="stage__agent">{verdictLabel(latest.verdict)}</span>
                          ) : null}
                        </span>
                      ) : null}
                    </button>
                  </li>
                );
              })}
            </ol>
          </li>
        ))}
      </ol>
    </nav>
  );
}

export default function RunConsole({ runId }: { runId: string }) {
  const params = useSearchParams();
  const [run, setRun] = useState<PipelineRun | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(params.get('stage'));
  const [attemptChoice, setAttemptChoice] = useState<Record<string, string>>({});
  const [artifactChoice, setArtifactChoice] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [rationale, setRationale] = useState('');
  const [hypothesis, setHypothesis] = useState('');
  const [reconcileStatus, setReconcileStatus] = useState<Record<string, ReconcileStatus>>({});
  const [reconcileWhy, setReconcileWhy] = useState<Record<string, string>>({});
  const [history, setHistory] = useState<PipelineRun[]>([]);
  const [view, setView] = useState<View>('ladder');
  const now = useNow(1000);

  // The chosen view is a per-browser convenience, read after mount so the server render matches;
  // `?view=graph` in the URL wins so a link can open the graph directly.
  useEffect(() => {
    const fromUrl = params.get('view');
    setView(fromUrl === 'graph' || fromUrl === 'ladder' ? fromUrl : storedView());
  }, [params]);
  function chooseView(next: View) {
    setView(next);
    try {
      window.localStorage.setItem(VIEW_KEY, next);
    } catch {
      // Private windows and blocked storage just forget the choice.
    }
  }

  // Every run the operator can see, refreshed slowly: the source of "typical" stage durations.
  useEffect(() => {
    let cancelled = false;
    const load = () =>
      listRuns()
        .then((runs) => {
          if (!cancelled) setHistory(runs);
        })
        .catch(() => undefined);
    void load();
    const timer = setInterval(load, 30_000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);
  const typical = useMemo(() => typicalDurations(history), [history]);

  const refresh = useCallback(async () => {
    try {
      setRun(await loadRun(runId));
      setError(null);
    } catch (cause) {
      setError((cause as Error).message);
    }
  }, [runId]);

  useEffect(() => {
    void refresh();
    const timer = setInterval(refresh, 3000);
    return () => clearInterval(timer);
  }, [refresh]);

  // Land on whatever needs the operator first, unless the URL named a stage.
  useEffect(() => {
    if (!run || selectedId) return;
    const first = stageNeedingAttention(run) ?? orderStages(run.nodes)[0]?.node;
    if (first) setSelectedId(first.id);
  }, [run, selectedId]);

  const selected = run?.nodes.find((node) => node.id === selectedId) ?? null;
  const attempts = run && selected ? attemptsFor(run, selected.id) : [];
  const attempt = useMemo(() => {
    if (!run || !selected) return undefined;
    const chosen = attempts.find((item) => item.id === attemptChoice[selected.id]);
    return chosen ?? decisionAttempt(run, selected);
  }, [run, selected, attempts, attemptChoice]);
  const artifacts = useMemo(
    () =>
      run && attempt ? (run.artifacts ?? []).filter((item) => item.attemptId === attempt.id) : [],
    [run, attempt],
  );
  const artifactId = attempt
    ? (artifactChoice[attempt.id] ?? leadArtifact(artifacts)?.id ?? null)
    : null;

  const gates = run?.nodes.filter((node) => node.status === 'waiting_human') ?? [];

  async function act(work: () => Promise<unknown>, done: string) {
    if (busy) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await work();
      setNotice(done);
      setRationale('');
      setHypothesis('');
      await refresh();
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setBusy(false);
    }
  }

  function onApprove() {
    if (!run || !selected || !attempt) return;
    const text = rationale.trim();
    if (!text) {
      setError('Approval needs a rationale.');
      return;
    }
    const grouped = approvalArtifacts(
      (run.artifacts ?? []).filter((item) => item.attemptId === attempt.id),
    );
    void act(
      () =>
        approve(run.id, {
          node_id: selected.id,
          attempt_id: attempt.id,
          artifacts: grouped,
          rationale: text,
        }),
      `Approved ${selected.id}.`,
    );
  }

  function onCommand(name: RunCommand) {
    if (!run || !selected) return;
    const text = rationale.trim();
    if (!text) {
      setError(`${name[0].toUpperCase()}${name.slice(1)} needs a rationale.`);
      return;
    }
    const why = hypothesis.trim();
    if (name === 'retry' && !why) {
      setError('A retry needs a hypothesis: what will be different this time?');
      return;
    }
    void act(
      () =>
        command(run.id, name, {
          node_id: selected.id,
          attempt_id: attempt?.id,
          rationale: text,
          ...(name === 'retry' && why ? { parameters: { hypothesis: why } } : {}),
        }),
      `Sent ${name} for ${selected.id}.`,
    );
  }

  function onReconcile(claimId: string) {
    if (!run) return;
    const why = (reconcileWhy[claimId] ?? '').trim();
    if (!why) {
      setError('Reconciling a claim needs a rationale: what evidence shows the outcome?');
      return;
    }
    void act(
      () =>
        reconcileClaim(run.id, claimId, {
          status: reconcileStatus[claimId] ?? 'failed',
          rationale: why,
        }),
      'Recorded.',
    );
  }

  if (!run) {
    return (
      <main className="page" id="main">
        <p className="crumbs">
          <Link href="/">Runs</Link>
        </p>
        {error ? (
          <p className="notice notice--error" role="alert">
            {error}
          </p>
        ) : (
          <p className="empty">Loading run…</p>
        )}
      </main>
    );
  }

  const isGate = selected?.status === 'waiting_human';
  const reviews = run && selected ? reviewsFor(run, selected.id) : [];
  // When the reviewing agent defers to a human, its question travels in blockedReason.
  const agentQuestion =
    isGate && selected?.blockedReason ? parseAgentQuestion(selected.blockedReason) : null;
  const canRetry = selected
    ? ['failed', 'blocked', 'succeeded', 'canceled'].includes(selected.status)
    : false;
  const paused = run.live?.paused === true || run.status === 'paused';
  const claims = run.providerClaims ?? [];

  return (
    <main className="page" id="main">
      <p className="crumbs">
        <Link href="/">Runs</Link>
        {run.parentRunId ? (
          <>
            {' / '}
            <Link href={`/runs/${encodeURIComponent(run.parentRunId)}`} className="mono">
              {run.parentRunId}
            </Link>
          </>
        ) : null}
      </p>

      <header className="run-head">
        <div>
          <h1>{run.parentRunId && run.branchKey ? `shot ${run.branchKey}` : run.id}</h1>
          <p className="run-head__what">{describeOptions(run.configuration.options)}</p>
          <div className="run-head__facts">
            <span>
              started <span className="mono">{formatWhen(run.createdAt, now)}</span>
            </span>
            {TERMINAL_RUN.has(run.status) ? (
              <span>
                finished <span className="mono">{formatWhen(run.updatedAt, now)}</span>
              </span>
            ) : null}
            <span title="Time during which a stage was executing; waiting and blocked time is not counted">
              ran for{' '}
              <span className={`mono${isExecuting(run) ? ' live' : ''}`}>
                {activeMs(run, now) === null ? 'nothing yet' : formatMs(activeMs(run, now) ?? 0)}
              </span>
            </span>
            <span>
              source{' '}
              <span className="mono" title={run.sourceSha256}>
                {shortHash(run.sourceSha256)}
              </span>
            </span>
            <span>
              graph <span className="mono">{run.graphVersion}</span>
            </span>
            {run.parentRunId ? (
              <span>
                run <span className="mono">{run.id}</span>
              </span>
            ) : null}
          </div>
        </div>
        <div className="run-head__state">
          <Status status={run.status} />
        </div>
        <StageStrip nodes={run.nodes} />
      </header>

      {run.liveError ? (
        <p className="notice notice--warn" style={{ marginBottom: 16 }}>
          The workflow engine could not be queried for this run ({run.liveError}); stage state below
          is what the database recorded.
        </p>
      ) : null}
      {paused ? (
        <p className="notice notice--warn" style={{ marginBottom: 16 }}>
          This run is paused. Nothing new starts until you resume it.
        </p>
      ) : null}
      {gates.filter((gate) => gate.id !== selectedId).length > 0 ? (
        <div className="gates">
          {gates
            .filter((gate) => gate.id !== selectedId)
            .map((gate) => (
              <button
                key={gate.id}
                type="button"
                className="gate"
                onClick={() => setSelectedId(gate.id)}
              >
                <Status status={gate.status} />
                <span className="gate__what">
                  <strong>{gate.definition.title}</strong>
                  <span className="gate__where">
                    {' '}
                    <span className="mono">{gate.id}</span>
                  </span>
                </span>
                <span className="gate__cta">Review</span>
              </button>
            ))}
        </div>
      ) : null}
      {notice ? (
        <p className="notice notice--ok" role="status" style={{ marginBottom: 16 }}>
          {notice}
        </p>
      ) : null}
      {error ? (
        <p className="notice notice--error" role="alert" style={{ marginBottom: 16 }}>
          {error}
        </p>
      ) : null}

      <div className="views" role="tablist" aria-label="Stage view">
        <button
          type="button"
          role="tab"
          aria-selected={view === 'ladder'}
          className={`views__tab${view === 'ladder' ? ' views__tab--current' : ''}`}
          onClick={() => chooseView('ladder')}
        >
          List
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={view === 'graph'}
          className={`views__tab${view === 'graph' ? ' views__tab--current' : ''}`}
          onClick={() => chooseView('graph')}
        >
          Graph
        </button>
      </div>

      <div className={`console${view === 'graph' ? ' console--graph' : ''}`}>
        {view === 'graph' ? (
          <GraphCanvas
            run={run}
            selectedId={selectedId}
            onSelect={setSelectedId}
            now={now}
            typical={typical}
          />
        ) : (
          <Ladder
            run={run}
            selectedId={selectedId}
            onSelect={setSelectedId}
            now={now}
            typical={typical}
          />
        )}

        <section className="inspect" aria-live="polite">
          {!selected ? (
            <p className="empty">Choose a stage to see its files and act on it.</p>
          ) : (
            <>
              <div className="inspect__head">
                <h2>
                  <span className="mono">{selected.id}</span>
                  <Status status={selected.status} blockedReason={selected.blockedReason} />
                </h2>
                <p>{selected.definition.title}</p>
                <div className="inspect__facts">
                  <span>{KIND_LABEL[selected.definition.kind] ?? selected.definition.kind}</span>
                  {selected.definition.executor ? (
                    <span>
                      runs <span className="mono">{selected.definition.executor}</span>
                    </span>
                  ) : null}
                  {selected.definition.resources?.task_queue ? (
                    <span>
                      on <span className="mono">{selected.definition.resources.task_queue}</span>
                    </span>
                  ) : null}
                  {dependenciesOf(selected).length > 0 ? (
                    <span>
                      after{' '}
                      {dependenciesOf(selected).map((dependency, index) => (
                        <span key={dependency}>
                          {index > 0 ? ', ' : ''}
                          <button
                            type="button"
                            className="linkish mono"
                            onClick={() => setSelectedId(dependency)}
                          >
                            {dependency}
                          </button>
                        </span>
                      ))}
                    </span>
                  ) : null}
                </div>
                {selected.blockedReason && !agentQuestion ? (
                  isWaitingOnUpstream(selected.status, selected.blockedReason) ? (
                    <p className="waiting">{UPSTREAM_NOTE}</p>
                  ) : (
                    <p className="blocked">{selected.blockedReason}</p>
                  )
                ) : null}
              </div>

              {unrecordedClaims(run, selected.id).map((claim) => (
                <div className="unblock" key={claim.id}>
                  <p>
                    A <span className="mono">{claim.provider}</span> call for this stage ended
                    without a recorded outcome. Nothing is waiting on this; recording it keeps the
                    ledger accurate.
                  </p>
                  <div className="unblock__row">
                    <select
                      aria-label={`Observed outcome for ${claim.operation}`}
                      value={reconcileStatus[claim.id] ?? 'failed'}
                      onChange={(event) =>
                        setReconcileStatus((current) => ({
                          ...current,
                          [claim.id]: event.target.value as ReconcileStatus,
                        }))
                      }
                    >
                      <option value="failed">It failed, nothing was produced</option>
                      <option value="completed">It completed and was charged</option>
                    </select>
                    <input
                      type="text"
                      aria-label={`Evidence for ${claim.operation}`}
                      placeholder="What the provider's logs show"
                      value={reconcileWhy[claim.id] ?? ''}
                      onChange={(event) =>
                        setReconcileWhy((current) => ({
                          ...current,
                          [claim.id]: event.target.value,
                        }))
                      }
                    />
                    <button
                      type="button"
                      className="button button--primary button--small"
                      disabled={busy}
                      onClick={() => onReconcile(claim.id)}
                    >
                      Record outcome
                    </button>
                  </div>
                </div>
              ))}

              {reviews.length > 0 ? (
                <div className="review-list">
                  <Review review={reviews[0]} now={now} />
                  {reviews.length > 1 ? (
                    <details className="review__earlier">
                      <summary>
                        {reviews.length - 1 === 1
                          ? '1 earlier review'
                          : `${reviews.length - 1} earlier reviews`}
                      </summary>
                      <ol>
                        {reviews.slice(1).map((review) => (
                          <li key={review.id}>
                            <Review review={review} now={now} />
                          </li>
                        ))}
                      </ol>
                    </details>
                  ) : null}
                </div>
              ) : null}

              {attempts.length > 0 ? (
                <div>
                  <div className="attempts" role="tablist" aria-label="Attempts">
                    {attempts.map((item) => (
                      <button
                        key={item.id}
                        type="button"
                        role="tab"
                        aria-selected={item.id === attempt?.id}
                        className={`attempt${item.id === attempt?.id ? ' attempt--current' : ''}`}
                        onClick={() =>
                          setAttemptChoice((current) => ({ ...current, [selected.id]: item.id }))
                        }
                      >
                        <span className="attempt__n">
                          #{item.number}
                          {isAgentRevision(item) ? ' agent revision' : ''}
                        </span>
                        <span>{statusLabel(item.status)}</span>
                        {item.startedAt ? (
                          <span className="attempt__time">
                            {formatDuration(item.startedAt, item.finishedAt, now)}
                          </span>
                        ) : null}
                      </button>
                    ))}
                  </div>
                  {attempt?.startedAt ? (
                    <div className="timeline">
                      <div className="timeline__row">
                        <span>
                          started{' '}
                          <span className="mono">{formatWhenExact(attempt.startedAt, now)}</span>
                        </span>
                        {attempt.finishedAt ? (
                          <span>
                            finished{' '}
                            <span className="mono">{formatWhenExact(attempt.finishedAt, now)}</span>
                          </span>
                        ) : null}
                        <span>
                          {attempt.finishedAt
                            ? 'took'
                            : isInFlight(attempt.status)
                              ? 'running for'
                              : 'open for'}{' '}
                          <span className={`mono${attempt.finishedAt ? '' : ' live'}`}>
                            {formatMs(elapsedMs(attempt.startedAt, attempt.finishedAt, now) ?? 0)}
                          </span>
                        </span>
                        {isInFlight(attempt.status) ? (
                          <span>
                            {typical.get(selected.id) !== undefined
                              ? `typically about ${formatMs(typical.get(selected.id) ?? 0)}`
                              : 'no earlier finish of this stage to estimate from'}
                          </span>
                        ) : null}
                      </div>
                      {isInFlight(attempt.status) ? (
                        <Progress
                          fraction={progressFraction(
                            elapsedMs(attempt.startedAt, attempt.finishedAt, now),
                            typical.get(selected.id),
                          )}
                          label={`${selected.id} attempt ${attempt.number} in progress`}
                        />
                      ) : null}
                    </div>
                  ) : null}
                  {attempt?.hypothesis ? (
                    <p className="attempt__note">
                      Hypothesis: <span className="mono">{attempt.hypothesis}</span>
                    </p>
                  ) : null}
                  {attempt?.providerOperationId ? (
                    <p className="attempt__note">
                      Provider operation <span className="mono">{attempt.providerOperationId}</span>
                    </p>
                  ) : null}
                </div>
              ) : null}

              {attempt?.error ? (
                <p className="notice notice--error">
                  Attempt #{attempt.number}: <code>{attempt.error}</code>
                </p>
              ) : null}

              <ArtifactViewer
                runId={run.id}
                artifacts={artifacts}
                selectedId={artifactId}
                onSelect={(id) =>
                  attempt && setArtifactChoice((current) => ({ ...current, [attempt.id]: id }))
                }
                empty={
                  attempts.length === 0
                    ? 'This stage has not run yet, so there is nothing to look at.'
                    : 'This attempt produced no files.'
                }
              />

              <div className={`decide${isGate ? ' decide--gate' : ''}`}>
                <h3 className="decide__title">
                  {isGate ? 'Your decision' : 'Steer this stage'}
                  <small>recorded on the run with your reason</small>
                </h3>

                {agentQuestion ? (
                  <p className="ask">
                    <span className="ask__label">
                      {agentQuestion.overCap === null
                        ? 'The reviewing agent asks'
                        : `The reviewing agent used its ${agentQuestion.overCap - 1} allowed retries and now asks`}
                    </span>
                    {agentQuestion.question}
                  </p>
                ) : null}

                {canRetry ? (
                  <div className="field">
                    <label htmlFor="hypothesis">What will be different this retry</label>
                    <input
                      id="hypothesis"
                      type="text"
                      value={hypothesis}
                      placeholder="Rerunning a stage unchanged is refused; say what changed"
                      onChange={(event) => setHypothesis(event.target.value)}
                    />
                  </div>
                ) : null}

                <div className="field">
                  <label htmlFor="rationale">Reason</label>
                  <textarea
                    id="rationale"
                    value={rationale}
                    placeholder={
                      isGate
                        ? 'What you checked in the files above and why they pass or fail.'
                        : 'Why this action, in a sentence.'
                    }
                    onChange={(event) => setRationale(event.target.value)}
                  />
                </div>

                <div className="decide__actions">
                  {isGate || selected.definition.kind === 'human' ? (
                    <button
                      className="button button--approve"
                      type="button"
                      disabled={busy || !attempt}
                      onClick={onApprove}
                    >
                      Approve
                    </button>
                  ) : null}
                  <button
                    className={`button${!isGate && canRetry ? ' button--primary' : ''}`}
                    type="button"
                    disabled={busy}
                    onClick={() => onCommand('retry')}
                  >
                    Retry stage
                  </button>
                  {paused ? (
                    <button
                      className="button"
                      type="button"
                      disabled={busy}
                      onClick={() => onCommand('resume')}
                    >
                      Resume run
                    </button>
                  ) : (
                    <button
                      className="button button--quiet"
                      type="button"
                      disabled={busy}
                      onClick={() => onCommand('pause')}
                    >
                      Pause run
                    </button>
                  )}
                  <span className="gap" />
                  {isGate || selected.definition.kind === 'human' ? (
                    <button
                      className="button button--danger"
                      type="button"
                      disabled={busy}
                      onClick={() => onCommand('reject')}
                    >
                      Reject
                    </button>
                  ) : null}
                  <button
                    className="button button--danger"
                    type="button"
                    disabled={busy}
                    onClick={() => onCommand('cancel')}
                  >
                    Cancel run
                  </button>
                </div>
              </div>
            </>
          )}
        </section>
      </div>
    </main>
  );
}
