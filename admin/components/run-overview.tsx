'use client';

import { useEffect, useState } from 'react';
import { Progress } from '@/components/progress';
import { Status } from '@/components/status';
import { loadRunProgress } from '@/lib/client';
import { formatBytes } from '@/lib/format';
import type { RunProgress } from '@/lib/types';

function progressTitle(progress: RunProgress): string {
  if (progress.finished) return `Run finished: ${progress.finished}`;
  if (progress.paused) return 'Run paused';
  if (progress.question) return 'Waiting for your answer';
  if (progress.stopped.length) return 'Processing stopped';
  if (progress.currentStages.length) return `Running ${progress.currentStages.join(', ')}`;
  return 'Waiting for the supervisor';
}

export function RunOverview({
  runId,
  onSelectStage,
}: {
  runId: string;
  onSelectStage: (name: string) => void;
}) {
  const [progress, setProgress] = useState<RunProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setProgress(null);
    const refresh = async () => {
      try {
        const next = await loadRunProgress(runId);
        if (!cancelled) {
          setProgress(next);
          setError(null);
        }
      } catch (cause) {
        if (!cancelled) setError((cause as Error).message);
      }
    };
    void refresh();
    const timer = setInterval(refresh, 3000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [runId]);

  if (!progress)
    return (
      <p className="field__hint run-overview__loading" role="status">
        {error
          ? `Run files unavailable: ${error}. Retrying.`
          : 'Loading run progress and previews…'}
      </p>
    );

  const done = progress.stages.filter((stage) =>
    ['succeeded', 'skipped'].includes(stage.status),
  ).length;
  const reused = progress.stages.filter(
    (stage) => stage.freshInference === false && ['succeeded', 'skipped'].includes(stage.status),
  );
  const videos = progress.previews.filter((preview) => preview.kind === 'video');
  const images = progress.previews.filter((preview) => preview.kind === 'image');
  const reports = progress.previews.filter((preview) => preview.kind === 'report');
  const active = progress.stages.filter((stage) => progress.currentStages.includes(stage.name));
  const ready = progress.stages.find(
    (stage) => stage.status === 'pending' && !stage.dependencies.length,
  );
  const title = progressTitle(progress);

  return (
    <section className="run-overview" aria-labelledby="run-progress-title">
      <div className="section-head">
        <h2 className="section-title" id="run-progress-title">
          {title}
        </h2>
        <span className="field__hint">
          {done}/{progress.stages.length} stages completed
        </span>
      </div>
      <Progress
        fraction={progress.stages.length ? done / progress.stages.length : 0}
        label={`${done} of ${progress.stages.length} stages completed`}
        tone={
          progress.stopped.length || progress.question
            ? 'attention'
            : progress.finished === 'succeeded'
              ? 'good'
              : 'active'
        }
      />
      {error ? (
        <p className="field__hint">Progress refresh failed: {error}. Showing the last update.</p>
      ) : null}
      {progress.question ? <p className="notice notice--warn">{progress.question}</p> : null}
      {reused.length ? (
        <p className="field__hint">
          {reused.length} completed stages use saved inputs:{' '}
          {reused.map((stage) => stage.name).join(', ')}. No fresh inference was performed for these
          stages.
        </p>
      ) : null}
      {progress.viewerPath ? (
        <p className="field__hint">
          <a href={`http://127.0.0.1:5399${progress.viewerPath}`} target="_blank" rel="noreferrer">
            Preview packaged scene
          </a>{' '}
          — reconstruction quality is unverified.
        </p>
      ) : null}
      {progress.paused ? (
        <p className="field__hint">Resume this run to allow new work to start.</p>
      ) : null}
      {active.length || progress.stopped.length ? (
        <div className="run-overview__stages">
          {[...progress.stopped, ...active].map((stage) => (
            <article className="run-overview__stage" key={stage.name}>
              <div className="run-overview__stage-head">
                <button
                  type="button"
                  className="text-link mono"
                  onClick={() => onSelectStage(stage.name)}
                >
                  {stage.name}
                </button>
                <Status status={stage.status} />
              </div>
              <p>{stage.summary}</p>
              {stage.reason ? <p className="run-overview__reason">{stage.reason}</p> : null}
              <p className="field__hint">{stage.nextAction}</p>
            </article>
          ))}
        </div>
      ) : ready && !progress.finished && !progress.paused && !progress.question ? (
        <p className="field__hint">
          Next: {ready.name}. {ready.nextAction}
        </p>
      ) : null}
      {progress.previews.length ? (
        <details className="run-previews" open>
          <summary>
            Source and outputs <span className="count">{progress.previews.length}</span>
          </summary>
          {videos.length ? (
            <div className="run-previews__media">
              {videos.map((preview) => (
                <figure key={preview.path}>
                  <video controls preload="metadata" src={preview.url} aria-label={preview.label} />
                  <figcaption>
                    <a href={preview.url} target="_blank" rel="noreferrer">
                      {preview.label}
                    </a>
                    <span>{formatBytes(preview.bytes)}</span>
                  </figcaption>
                </figure>
              ))}
            </div>
          ) : null}
          {images.length ? (
            <div className="run-previews__images">
              {images.map((preview) => (
                <figure key={preview.path}>
                  <a href={preview.url} target="_blank" rel="noreferrer">
                    <img
                      loading="lazy"
                      src={preview.url}
                      alt={`${preview.label}: ${preview.path}`}
                    />
                  </a>
                  <figcaption>{preview.path}</figcaption>
                </figure>
              ))}
            </div>
          ) : null}
          {reports.length ? (
            <ul className="run-previews__reports">
              {reports.map((preview) => (
                <li key={preview.path}>
                  <a href={preview.url} target="_blank" rel="noreferrer">
                    {preview.label}: <span className="mono">{preview.path}</span>
                  </a>{' '}
                  <span className="field__hint">{formatBytes(preview.bytes)}</span>
                </li>
              ))}
            </ul>
          ) : null}
        </details>
      ) : (
        <p className="field__hint">Previews appear as each stage produces them.</p>
      )}
    </section>
  );
}
