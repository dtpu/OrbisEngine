'use client';

import type {
  GraphOptions,
  OptionsSchema,
  RunProgress,
  SetupReport,
  PipelineRun,
  ReconcileStatus,
  ReviewTranscript,
  RunArtifact,
  RunCommand,
  PreparedSavedInputRun,
  SavedInputsAvailability,
  TranscriptTail,
} from './types';

const ROOT = '/api/pipeline';

async function unwrap(response: Response): Promise<unknown> {
  const text = await response.text();
  const body = text ? JSON.parse(text) : null;
  if (!response.ok) {
    const detail =
      body && typeof body === 'object' && 'detail' in body
        ? String((body as { detail: unknown }).detail)
        : `${response.status} ${response.statusText}`;
    throw new Error(detail);
  }
  return body;
}

async function send(path: string, payload: unknown): Promise<unknown> {
  return unwrap(
    await fetch(`${ROOT}${path}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  );
}

export async function listRuns(): Promise<PipelineRun[]> {
  const body = (await unwrap(await fetch(ROOT, { cache: 'no-store' }))) as { runs: PipelineRun[] };
  return body.runs;
}

export async function loadRun(runId: string): Promise<PipelineRun> {
  return (await unwrap(
    await fetch(`${ROOT}/runs/${encodeURIComponent(runId)}`, { cache: 'no-store' }),
  )) as PipelineRun;
}

export async function checkSavedInputs(runId: string): Promise<SavedInputsAvailability> {
  return (await unwrap(
    await fetch(`${ROOT}/runs/${encodeURIComponent(runId)}/saved-inputs`, { cache: 'no-store' }),
  )) as SavedInputsAvailability;
}

export async function prepareSavedInputRun(
  runId: string,
  id?: string,
): Promise<PreparedSavedInputRun> {
  return (await send(`/runs/${encodeURIComponent(runId)}/saved-inputs`, {
    id,
  })) as PreparedSavedInputRun;
}

export async function uploadRun(input: {
  file: File;
  options: GraphOptions;
  requestedId?: string;
  onProgress?: (fraction: number) => void;
}): Promise<{ id: string }> {
  const form = new FormData();
  form.set('source', input.file);
  form.set('options', JSON.stringify(input.options));
  form.set('budget', JSON.stringify({}));
  if (input.requestedId) form.set('requested_id', input.requestedId);

  // XHR rather than fetch: it is the only browser API that reports upload progress, and clips
  // are large enough that a silent multi-minute POST looks like a hang.
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('POST', `${ROOT}/runs/upload`);
    request.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable) input.onProgress?.(event.loaded / event.total);
    });
    request.addEventListener('load', () => {
      let body: unknown = null;
      try {
        body = request.responseText ? JSON.parse(request.responseText) : null;
      } catch {
        body = null;
      }
      if (request.status >= 200 && request.status < 300) {
        resolve(body as { id: string });
        return;
      }
      const detail =
        body && typeof body === 'object' && 'detail' in body
          ? String((body as { detail: unknown }).detail)
          : `${request.status} ${request.statusText}`;
      reject(new Error(detail));
    });
    request.addEventListener('error', () => reject(new Error('upload failed')));
    request.addEventListener('abort', () => reject(new Error('upload canceled')));
    request.send(form);
  });
}

/** One run's transcripts, newest first. A run writes one, so this is usually a single row. */
export async function listReviewTranscripts(
  runId: string,
): Promise<{ transcripts: ReviewTranscript[]; available: boolean }> {
  const body = (await unwrap(
    await fetch(`${ROOT}/runs/${encodeURIComponent(runId)}/reviews`, { cache: 'no-store' }),
  )) as { transcripts: ReviewTranscript[]; available: boolean };
  return { ...body, transcripts: [...body.transcripts].reverse() };
}

export async function readTranscript(
  runId: string,
  nodeId: string,
  attemptId: string,
  after: number,
): Promise<TranscriptTail> {
  const path = `${ROOT}/runs/${encodeURIComponent(runId)}/reviews/${encodeURIComponent(nodeId)}/${encodeURIComponent(attemptId)}/transcript?after=${after}`;
  return (await unwrap(await fetch(path, { cache: 'no-store' }))) as TranscriptTail;
}

/** Same-origin URL for an artifact's bytes; the proxy adds the bearer token server-side. */
export function artifactUrl(runId: string, artifactId: string): string {
  return `${ROOT}/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(artifactId)}`;
}

/** Group an attempt's artifacts by role, the shape the approve endpoint expects. */
export function approvalArtifacts(artifacts: RunArtifact[]): Record<string, string[]> {
  return artifacts.reduce<Record<string, string[]>>((groups, artifact) => {
    const role = artifact.role;
    return { ...groups, [role]: [...(groups[role] ?? []), artifact.id] };
  }, {});
}

export function approve(
  runId: string,
  body: {
    node_id: string;
    attempt_id: string;
    artifacts: Record<string, string[]>;
    rationale: string;
  },
): Promise<unknown> {
  return send(`/runs/${encodeURIComponent(runId)}/approve`, body);
}

export function command(
  runId: string,
  name: RunCommand,
  body: {
    node_id: string;
    attempt_id?: string;
    rationale: string;
    parameters?: Record<string, unknown>;
  },
): Promise<unknown> {
  return send(`/runs/${encodeURIComponent(runId)}/${name}`, body);
}

export function sendMessage(
  runId: string,
  body: { message: string; node_id?: string; attempt_id?: string },
): Promise<unknown> {
  return send(`/runs/${encodeURIComponent(runId)}/messages`, body);
}

/**
 * Record the observed outcome of a provider claim the orchestrator lost track of. This is the
 * operator asserting what the provider actually did, so it always carries a rationale.
 */
export function reconcileClaim(
  runId: string,
  claimId: string,
  body: { status: ReconcileStatus; rationale: string; provider_operation_id?: string },
): Promise<unknown> {
  return send(
    `/runs/${encodeURIComponent(runId)}/provider-claims/${encodeURIComponent(claimId)}/reconcile`,
    body,
  );
}

export async function loadOptionsSchema(): Promise<OptionsSchema> {
  return (await unwrap(await fetch(`${ROOT}/options`, { cache: 'no-store' }))) as OptionsSchema;
}

export async function checkSetup(options: GraphOptions): Promise<SetupReport> {
  return (await send('/doctor', options)) as SetupReport;
}

export async function loadRunProgress(runId: string): Promise<RunProgress> {
  return (await unwrap(
    await fetch(`${ROOT}/runs/${encodeURIComponent(runId)}/progress`, { cache: 'no-store' }),
  )) as RunProgress;
}
