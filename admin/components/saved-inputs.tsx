'use client';

import Link from 'next/link';
import { useRef, useState } from 'react';
import { checkSavedInputs, prepareSavedInputRun } from '@/lib/client';
import { formatBytes } from '@/lib/format';
import type { SavedInputsAvailability } from '@/lib/types';

export function SavedInputs({ runId }: { runId: string }) {
  const [availability, setAvailability] = useState<SavedInputsAvailability | null>(null);
  const [checking, setChecking] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [name, setName] = useState('');
  const [preparedId, setPreparedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const busyRef = useRef(false);

  async function check() {
    if (busyRef.current) return;
    busyRef.current = true;
    setChecking(true);
    setError(null);
    setPreparedId(null);
    try {
      setAvailability(await checkSavedInputs(runId));
    } catch (cause) {
      setError(`Saved files could not be checked: ${(cause as Error).message}`);
    } finally {
      busyRef.current = false;
      setChecking(false);
    }
  }

  async function submitPreparation() {
    if (!availability?.available || busyRef.current) return;
    busyRef.current = true;
    setSubmitting(true);
    setError(null);
    try {
      const result = await prepareSavedInputRun(runId, name.trim() || undefined);
      setPreparedId(result.id);
    } catch (cause) {
      setError(`Paused run could not be prepared: ${(cause as Error).message}`);
    } finally {
      busyRef.current = false;
      setSubmitting(false);
    }
  }

  function prepare(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submitPreparation();
  }

  return (
    <section className="saved-inputs" aria-labelledby="saved-inputs-title">
      <div>
        <h2 className="create__title" id="saved-inputs-title">
          Prepare a delivery run
        </h2>
        <p className="field__hint">
          Reuses saved reconstruction. Local packaging only; no new inference or quality approval.
        </p>
      </div>
      {!availability ? (
        <button className="button button--small" type="button" onClick={check} disabled={checking}>
          {checking ? 'Checking saved files…' : 'Use saved inputs'}
        </button>
      ) : availability.available ? (
        preparedId ? null : (
          <>
            <p className="notice notice--ok" role="status">
              {availability.fileCount ?? 0} saved files · {formatBytes(availability.byteCount ?? 0)}
            </p>
            <form className="saved-inputs__form" onSubmit={prepare}>
              <div className="field">
                <label htmlFor="saved-input-run-name">Run name (optional)</label>
                <input
                  id="saved-input-run-name"
                  type="text"
                  className="mono"
                  placeholder="generated when blank"
                  pattern="[A-Za-z0-9][A-Za-z0-9_.-]{0,127}"
                  maxLength={128}
                  value={name}
                  disabled={submitting}
                  onChange={(event) => setName(event.target.value)}
                />
              </div>
              <button
                className="button button--small button--primary"
                type="submit"
                disabled={submitting}
              >
                {submitting ? 'Preparing…' : 'Prepare paused run'}
              </button>
            </form>
          </>
        )
      ) : (
        <div className="saved-inputs__result" aria-live="polite">
          <p className="notice notice--warn">
            {availability.reason || 'Saved inputs are unavailable.'}
          </p>
          <button
            className="button button--small"
            type="button"
            onClick={check}
            disabled={checking}
          >
            {checking ? 'Checking saved files…' : 'Check again'}
          </button>
        </div>
      )}
      {error ? (
        <div className="saved-inputs__result" role="alert">
          <p className="notice notice--error">{error}</p>
          {availability?.available && !preparedId ? (
            <button
              className="button button--small"
              type="button"
              onClick={() => void submitPreparation()}
              disabled={submitting}
            >
              Retry preparation
            </button>
          ) : (
            <button
              className="button button--small"
              type="button"
              onClick={check}
              disabled={checking}
            >
              Retry check
            </button>
          )}
        </div>
      ) : null}
      {preparedId ? (
        <p className="notice notice--ok" role="status">
          Paused run prepared.{' '}
          <Link href={`/runs/${encodeURIComponent(preparedId)}`}>Open prepared run</Link>
        </p>
      ) : null}
    </section>
  );
}
