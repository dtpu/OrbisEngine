import { decodePipelineEvent, normalizeRuns } from './state.ts';
import type { PipelineCommand, PipelineEvent, PipelineRun } from './types.ts';

let apiToken: string | undefined;

export function requestApiToken(): boolean {
  if (apiToken) return true;
  if (typeof window === 'undefined') return false;
  apiToken = window.sessionStorage.getItem('wander.pipeline.token') ?? undefined;
  if (!apiToken) {
    apiToken = window.prompt('Pipeline access token')?.trim() || undefined;
    if (apiToken) window.sessionStorage.setItem('wander.pipeline.token', apiToken);
  }
  return Boolean(apiToken);
}

function headers(accept: string): Record<string, string> {
  return {
    Accept: accept,
    ...(apiToken ? { Authorization: `Bearer ${apiToken}` } : {}),
  };
}

export interface SseMessage {
  id?: string;
  event: string;
  data: string;
  retry?: number;
}

export function parseSseBlock(block: string): SseMessage | undefined {
  let event = 'message';
  let id: string | undefined;
  let retry: number | undefined;
  const data: string[] = [];
  for (const sourceLine of block.split(/\r?\n/)) {
    const line = sourceLine.endsWith('\r') ? sourceLine.slice(0, -1) : sourceLine;
    if (!line || line.startsWith(':')) continue;
    const separator = line.indexOf(':');
    const field = separator < 0 ? line : line.slice(0, separator);
    let value = separator < 0 ? '' : line.slice(separator + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'event') event = value || 'message';
    else if (field === 'data') data.push(value);
    else if (field === 'id' && !value.includes('\0')) id = value;
    else if (field === 'retry' && /^\d+$/.test(value)) retry = Number(value);
  }
  return data.length ? { id, event, data: data.join('\n'), retry } : undefined;
}

async function responseJson(response: Response): Promise<unknown> {
  const body = await response.text();
  if (response.ok) return body ? JSON.parse(body) : undefined;
  let message = body;
  try {
    const parsed = JSON.parse(body) as { detail?: unknown; error?: unknown; message?: unknown };
    message =
      typeof parsed.error === 'string'
        ? parsed.error
        : typeof parsed.message === 'string'
          ? parsed.message
          : typeof parsed.detail === 'string'
            ? parsed.detail
            : body;
  } catch {
    // The response was intentionally kept as plain text.
  }
  throw new Error(message || `Request failed (${response.status})`);
}

async function post(path: string, body: unknown): Promise<unknown> {
  return responseJson(
    await fetch(path, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { ...headers('application/json'), 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  );
}

export async function loadRuns(): Promise<PipelineRun[]> {
  const response = await fetch('/api/pipeline', {
    credentials: 'same-origin',
    headers: headers('application/json'),
  });
  return normalizeRuns(await responseJson(response));
}

export async function loadRun(runId: string): Promise<PipelineRun> {
  const response = await fetch(`/api/pipeline/runs/${encodeURIComponent(runId)}`, {
    credentials: 'same-origin',
    headers: headers('application/json'),
  });
  const run = normalizeRuns(await responseJson(response))[0];
  if (!run) throw new Error('Pipeline returned an empty run');
  return run;
}

export function sendCommand(
  runId: string,
  command: PipelineCommand,
  stageId?: string,
  attemptId?: string,
  artifacts: ArtifactApproval = {},
): Promise<unknown> {
  if (!stageId) throw new Error('Select a stage before sending a command');
  const body = {
    node_id: stageId,
    ...(attemptId ? { attempt_id: attemptId } : {}),
    rationale: `${command} requested from the operator dashboard`,
    ...(command === 'approve' ? { artifacts } : { parameters: {} }),
  };
  return post(`/api/pipeline/runs/${encodeURIComponent(runId)}/${command}`, body);
}

type ArtifactApproval = Record<string, string[]>;

export function approvalArtifacts(
  artifacts: Array<{ id: string; role?: string; name: string }>,
): ArtifactApproval {
  return artifacts.reduce<ArtifactApproval>((groups, artifact) => {
    const role = artifact.role ?? artifact.name;
    return { ...groups, [role]: [...(groups[role] ?? []), artifact.id] };
  }, {});
}

export function sendMessage(runId: string, message: string, stageId?: string): Promise<unknown> {
  return post(`/api/pipeline/runs/${encodeURIComponent(runId)}/messages`, {
    message,
    ...(stageId ? { node_id: stageId } : {}),
  });
}

export interface EventStreamHandlers {
  onConnection(state: 'live' | 'reconnecting' | 'offline'): void;
  onEvent(event: PipelineEvent, eventId?: string): void;
  onError(message: string): void;
}

export class PipelineEventStream {
  private controller?: AbortController;
  private stopped = true;
  private retryMs = 1_000;
  private lastEventId?: string;

  constructor(
    private readonly runId: string,
    private readonly handlers: EventStreamHandlers,
    lastEventId?: string,
  ) {
    this.lastEventId = lastEventId;
  }

  start(): void {
    if (!this.stopped) return;
    this.stopped = false;
    void this.connect();
  }

  stop(): void {
    this.stopped = true;
    this.controller?.abort();
    this.handlers.onConnection('offline');
  }

  private async connect(): Promise<void> {
    this.controller = new AbortController();
    try {
      const requestHeaders = headers('text/event-stream');
      if (this.lastEventId) requestHeaders['Last-Event-ID'] = this.lastEventId;
      const response = await fetch(`/api/pipeline/runs/${encodeURIComponent(this.runId)}/events`, {
        credentials: 'same-origin',
        headers: requestHeaders,
        signal: this.controller.signal,
      });
      if (!response.ok || !response.body) {
        throw new Error(`Event stream failed (${response.status})`);
      }
      this.handlers.onConnection('live');
      this.retryMs = 1_000;
      await this.consume(response.body);
      if (!this.stopped) throw new Error('Event stream closed');
    } catch (error) {
      if (this.stopped || (error instanceof DOMException && error.name === 'AbortError')) return;
      this.handlers.onError(error instanceof Error ? error.message : 'Event stream failed');
      this.handlers.onConnection('reconnecting');
      window.setTimeout(() => {
        if (!this.stopped) void this.connect();
      }, this.retryMs);
      this.retryMs = Math.min(this.retryMs * 2, 15_000);
    }
  }

  private async consume(body: ReadableStream<Uint8Array>): Promise<void> {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (!this.stopped) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let boundary = buffer.indexOf('\n\n');
      while (boundary >= 0) {
        const message = parseSseBlock(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
        if (message) this.dispatch(message);
        boundary = buffer.indexOf('\n\n');
      }
    }
  }

  private dispatch(message: SseMessage): void {
    if (message.retry !== undefined) this.retryMs = Math.max(250, message.retry);
    if (message.id !== undefined) this.lastEventId = message.id;
    try {
      const event = decodePipelineEvent(message.event, JSON.parse(message.data));
      if (event) this.handlers.onEvent(event, message.id);
    } catch {
      this.handlers.onError('Ignored a malformed pipeline event');
    }
  }
}
