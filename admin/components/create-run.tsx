'use client';

import Link from 'next/link';
import { useCallback, useEffect, useRef, useState } from 'react';
import { checkSetup, loadOptionsSchema, uploadRun } from '@/lib/client';
import { formatBytes } from '@/lib/format';
import { automaticUploadOptions, optionFields, setupBlocked } from '@/lib/options';
import type { GraphOptions, OptionsSchema, SetupReport } from '@/lib/types';

const WORLD_LABELS: Record<string, string> = {
  video: 'Video',
  image: 'Single image',
  multi: 'Multiple views',
  both: 'Image and video (two worlds)',
  none: 'No world generation',
};

export function SetupChecks({ report }: { report: SetupReport }) {
  const priority: Record<string, number> = { blocked: 0, unverified: 1, okay: 2, ok: 2 };
  const checks = [...report.checks].sort(
    (a, b) => (priority[a.status] ?? 2) - (priority[b.status] ?? 2),
  );
  return (
    <section className="setup-checks" aria-label="Pipeline setup checks" aria-live="polite">
      <h3>{report.ready ? 'Local setup checks passed' : 'Setup needs attention'}</h3>
      <p className="field__hint">
        Checks describe this server. Provider access and worker readiness may still need
        verification.
      </p>
      <ul>
        {checks.map((check) => (
          <li key={check.name} className={`setup-check setup-check--${check.status}`}>
            <div className="setup-check__heading">
              <strong>{check.name}</strong>
              <span>{check.status === 'ok' ? 'okay' : check.status}</span>
            </div>
            <p>{check.detail}</p>
            {check.action ? <p className="setup-check__action">{check.action}</p> : null}
          </li>
        ))}
      </ul>
    </section>
  );
}

export function CreateRun({ onCreated }: { onCreated: (runId: string) => void }) {
  const [requestedId, setRequestedId] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [schema, setSchema] = useState<OptionsSchema | null>(null);
  const [options, setOptions] = useState<GraphOptions | null>(null);
  const [setup, setSetup] = useState<SetupReport | null>(null);
  const [checking, setChecking] = useState(false);
  const [canQueueLater, setCanQueueLater] = useState(false);
  const [createdId, setCreatedId] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const loadOptions = useCallback(async () => {
    try {
      const description = await loadOptionsSchema();
      const defaults = automaticUploadOptions(description);
      setSchema(description);
      setOptions(defaults);
      setError(null);
    } catch (cause) {
      setError(`Could not load supported options: ${(cause as Error).message}`);
    }
  }, []);

  useEffect(() => {
    void loadOptions();
  }, [loadOptions]);

  function updateOption(name: keyof GraphOptions, value: string | number | boolean) {
    setOptions((current) => (current ? { ...current, [name]: value } : current));
    setSetup(null);
    setCanQueueLater(false);
  }

  async function inspect(): Promise<SetupReport | null> {
    if (!options) return null;
    setChecking(true);
    setError(null);
    setCanQueueLater(false);
    try {
      const report = await checkSetup(options);
      setSetup(report);
      return report;
    } catch (cause) {
      setError(`Setup check could not finish: ${(cause as Error).message}`);
      // The source stays selected, so a temporarily unavailable check can be retried.
      return null;
    } finally {
      setChecking(false);
    }
  }

  async function upload() {
    if (!file || !options || progress !== null) return;
    setError(null);
    setCreatedId(null);
    setProgress(0);
    try {
      const run = await uploadRun({
        file,
        options,
        requestedId: requestedId.trim() || undefined,
        onProgress: setProgress,
      });
      setCreatedId(run.id);
      setFile(null);
      setRequestedId('');
      setCanQueueLater(false);
      if (fileInput.current) fileInput.current.value = '';
      onCreated(run.id);
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setProgress(null);
    }
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!file || !options || checking || progress !== null) return;
    const report = await inspect();
    if (!report) return;
    if (setupBlocked(report)) {
      setCanQueueLater(true);
      return;
    }
    await upload();
  }

  const busy = checking || progress !== null;
  const percent = Math.round((progress ?? 0) * 100);

  return (
    <form className="create" onSubmit={submit} aria-labelledby="create-title">
      <h2 className="create__title" id="create-title">
        Start a run
      </h2>
      <p className="field__hint">
        Upload once. Processing continues automatically and returns a scene preview or a failure
        report. Failed checks remain visible; there are no automatic paid retries.
      </p>
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
          pattern="[A-Za-z0-9][A-Za-z0-9_.-]{0,127}"
          maxLength={128}
          value={requestedId}
          disabled={busy}
          onChange={(event) => setRequestedId(event.target.value)}
        />
      </div>
      {schema && options ? (
        <fieldset className="run-options" disabled={busy}>
          <legend>Processing options</legend>
          {optionFields(schema).map((field) => (
            <div className="field" key={field.name}>
              {field.type === 'boolean' ? (
                <label className="field__toggle" htmlFor={`option-${field.name}`}>
                  <input
                    id={`option-${field.name}`}
                    type="checkbox"
                    checked={Boolean(options[field.name])}
                    onChange={(event) => updateOption(field.name, event.target.checked)}
                  />
                  {field.label}
                </label>
              ) : (
                <>
                  <label htmlFor={`option-${field.name}`}>{field.label}</label>
                  {field.enum ? (
                    <select
                      id={`option-${field.name}`}
                      value={String(options[field.name])}
                      onChange={(event) => updateOption(field.name, event.target.value)}
                    >
                      {field.enum.map((value) => (
                        <option value={value} key={value}>
                          {WORLD_LABELS[value] ?? value}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      id={`option-${field.name}`}
                      type="number"
                      required
                      min={field.minimum}
                      max={field.maximum}
                      step={field.type === 'integer' ? 1 : 'any'}
                      value={
                        Number.isFinite(Number(options[field.name]))
                          ? Number(options[field.name])
                          : ''
                      }
                      onChange={(event) => updateOption(field.name, event.target.valueAsNumber)}
                    />
                  )}
                </>
              )}
              {field.description ? <p className="field__hint">{field.description}</p> : null}
            </div>
          ))}
        </fieldset>
      ) : (
        <p className="field__hint">Loading supported processing options…</p>
      )}
      {!schema && error ? (
        <button type="button" className="button button--quiet" onClick={loadOptions}>
          Retry loading options
        </button>
      ) : null}
      {setup ? <SetupChecks report={setup} /> : null}
      {error ? (
        <p className="notice notice--error" role="alert">
          {error}
        </p>
      ) : null}
      {progress !== null ? (
        <div className="field" aria-live="polite">
          <div className="bar">
            <span style={{ width: `${percent}%` }} />
          </div>
          <span className="field__hint">
            {percent === 100 ? 'Preparing the run…' : `Uploading ${percent}%`}
          </span>
        </div>
      ) : null}
      <div className="create__actions">
        <button
          className="button button--primary"
          type="submit"
          disabled={!file || !options || busy}
        >
          {checking ? 'Checking setup…' : progress !== null ? 'Uploading…' : 'Upload and process'}
        </button>
        <button
          className="button button--quiet"
          type="button"
          disabled={!options || busy}
          onClick={() => void inspect()}
        >
          Check setup
        </button>
      </div>
      {canQueueLater ? (
        <div className="field">
          <p className="field__hint">
            Resolve the checks before processing, or queue the clip while the worker is being set
            up.
          </p>
          <button
            className="button button--quiet"
            type="button"
            disabled={busy || !file}
            onClick={() => void upload()}
          >
            Queue for later
          </button>
        </div>
      ) : null}
      {createdId ? (
        <p className="notice notice--ok" role="status">
          Queued <Link href={`/runs/${encodeURIComponent(createdId)}`}>{createdId}</Link>. Open the
          run to follow its stages and previews.
        </p>
      ) : null}
      <p className="field__hint">
        Generated worlds, tracking and fine-tuning can use paid providers. Inspect the cleaned
        frames when the run requests review. Original recorded audio is preserved.
      </p>
    </form>
  );
}
