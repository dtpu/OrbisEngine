'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { ArtifactBlock } from '@/components/blocks';
import { classify } from '@/lib/blocks';
import { artifactUrl } from '@/lib/client';
import { formatBytes, shortHash } from '@/lib/format';
import type { RunArtifact } from '@/lib/types';

function displayName(artifact: RunArtifact): string {
  return artifact.name ?? artifact.role;
}

export function ArtifactViewer({
  runId,
  artifacts,
  selectedId,
  onSelect,
  empty = 'No files yet.',
}: {
  runId: string;
  artifacts: RunArtifact[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  empty?: string;
}) {
  const selected = artifacts.find((artifact) => artifact.id === selectedId);
  const filmRef = useRef<HTMLDivElement>(null);
  const [copied, setCopied] = useState(false);

  const grouped = useMemo(() => {
    // Outputs the stage contract names come first; incidental attempt files (logs, staged inputs)
    // follow so the operator's eye lands on the result.
    const outputs = artifacts.filter((artifact) => artifact.role !== 'attempt_file');
    const rest = artifacts.filter((artifact) => artifact.role === 'attempt_file');
    return [...outputs, ...rest];
  }, [artifacts]);

  useEffect(() => {
    setCopied(false);
  }, [selectedId]);

  function moveSelection(delta: number) {
    if (!selected) return;
    const index = grouped.findIndex((artifact) => artifact.id === selected.id);
    const next = grouped[index + delta];
    if (next) {
      onSelect(next.id);
      filmRef.current
        ?.querySelector<HTMLButtonElement>(`[data-artifact="${CSS.escape(next.id)}"]`)
        ?.focus();
    }
  }

  if (artifacts.length === 0) {
    return (
      <div className="viewer viewer--empty">
        <p className="viewer__note">{empty}</p>
      </div>
    );
  }

  const url = selected ? artifactUrl(runId, selected.id) : '';
  const kind = selected ? classify(selected) : null;

  return (
    <div className="viewer">
      <div className={`viewer__stage${kind ? ` viewer__stage--${kind.type}` : ''}`}>
        {selected ? <ArtifactBlock runId={runId} artifact={selected} density="full" /> : null}
      </div>

      {selected && kind ? (
        <div className="viewer__meta">
          <div className="viewer__title">
            <strong className="mono">{displayName(selected)}</strong>
            <span className="viewer__role">{selected.role.replaceAll('_', ' ')}</span>
          </div>
          <dl className="viewer__facts">
            <div>
              <dt>Kind</dt>
              <dd>{kind.noun}</dd>
            </div>
            <div>
              <dt>Type</dt>
              <dd className="mono">{selected.mediaType}</dd>
            </div>
            <div>
              <dt>Size</dt>
              <dd className="mono">{formatBytes(selected.size)}</dd>
            </div>
            <div>
              <dt>sha256</dt>
              <dd>
                <button
                  type="button"
                  className="linkish mono"
                  title={selected.sha256}
                  onClick={() => {
                    void navigator.clipboard
                      ?.writeText(selected.sha256)
                      .then(() => setCopied(true));
                  }}
                >
                  {shortHash(selected.sha256)}
                  {copied ? ' copied' : ''}
                </button>
              </dd>
            </div>
          </dl>
          <div className="viewer__actions">
            <a className="button button--quiet" href={url} target="_blank" rel="noreferrer">
              Open in new tab
            </a>
            <a className="button button--quiet" href={`${url}?download=1`}>
              Download
            </a>
          </div>
        </div>
      ) : null}

      <div
        className="film"
        ref={filmRef}
        role="listbox"
        aria-label="Files from this attempt"
        onKeyDown={(event) => {
          if (event.key === 'ArrowRight') {
            event.preventDefault();
            moveSelection(1);
          } else if (event.key === 'ArrowLeft') {
            event.preventDefault();
            moveSelection(-1);
          }
        }}
      >
        {grouped.map((artifact) => {
          const active = artifact.id === selected?.id;
          return (
            <button
              key={artifact.id}
              type="button"
              role="option"
              aria-selected={active}
              data-artifact={artifact.id}
              tabIndex={active || (!selected && artifact === grouped[0]) ? 0 : -1}
              className={`film__frame${active ? ' film__frame--active' : ''}`}
              onClick={() => onSelect(artifact.id)}
            >
              <ArtifactBlock runId={runId} artifact={artifact} density="thumb" />
              <span className="film__label">
                <span className="film__name">{displayName(artifact)}</span>
                <span className="film__size mono">
                  {classify(artifact).noun} · {formatBytes(artifact.size)}
                </span>
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
