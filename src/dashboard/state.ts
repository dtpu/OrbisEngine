import type {
  Artifact,
  Budget,
  DashboardAction,
  DashboardState,
  PipelineEvent,
  PipelineRun,
  PipelineStage,
  RunStatus,
  StageAttempt,
  StageStatus,
} from './types.ts';

const RUN_STATUSES = new Set<RunStatus>([
  'queued',
  'running',
  'waiting_human',
  'paused',
  'blocked',
  'succeeded',
  'failed',
  'canceled',
]);
const STAGE_STATUSES = new Set<StageStatus>([
  'pending',
  'queued',
  'ready',
  'running',
  'waiting_agent',
  'waiting_human',
  'blocked',
  'unknown',
  'succeeded',
  'failed',
  'canceled',
  'skipped',
]);

export const initialState: DashboardState = {
  runs: [],
  connection: 'connecting',
};

function object(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' ? (value as Record<string, unknown>) : {};
}

function text(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback;
}

function optionalText(value: unknown): string | undefined {
  return typeof value === 'string' && value ? value : undefined;
}

function finite(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function runStatus(value: unknown): RunStatus {
  const candidate =
    value === 'awaiting_approval' ? 'waiting_human' : value === 'cancelled' ? 'canceled' : value;
  return RUN_STATUSES.has(candidate as RunStatus) ? (candidate as RunStatus) : 'queued';
}

function stageStatus(value: unknown): StageStatus {
  const candidate =
    value === 'awaiting_approval' ? 'waiting_human' : value === 'cancelled' ? 'canceled' : value;
  return STAGE_STATUSES.has(candidate as StageStatus) ? (candidate as StageStatus) : 'pending';
}

function artifact(value: unknown, index: number): Artifact {
  const raw = object(value);
  const contentType = optionalText(raw.contentType) ?? optionalText(raw.mediaType);
  const rawKind = text(raw.kind, contentType?.split('/')[0] ?? 'file');
  const kind = ['image', 'video', 'audio', 'text', 'json', 'file'].includes(rawKind)
    ? (rawKind as Artifact['kind'])
    : 'file';
  return {
    id: text(raw.id, `artifact-${index}`),
    name: text(raw.name, text(raw.role, `Artifact ${index + 1}`)),
    role: optionalText(raw.role),
    kind,
    url: optionalText(raw.url),
    preview: optionalText(raw.preview),
    contentType,
    bytes:
      typeof (raw.bytes ?? raw.size) === 'number' && finite(raw.bytes ?? raw.size) >= 0
        ? finite(raw.bytes ?? raw.size)
        : undefined,
  };
}

function attempt(value: unknown, index: number): StageAttempt {
  const raw = object(value);
  return {
    id: text(raw.id, `attempt-${index + 1}`),
    number: finite(raw.number, index + 1),
    status: stageStatus(raw.status),
    startedAt: optionalText(raw.startedAt) ?? optionalText(raw.started_at),
    finishedAt: optionalText(raw.finishedAt) ?? optionalText(raw.finished_at),
    summary: optionalText(raw.summary) ?? optionalText(raw.error),
    logs: Array.isArray(raw.logs)
      ? raw.logs.filter((line): line is string => typeof line === 'string')
      : [],
    artifacts: Array.isArray(raw.artifacts)
      ? raw.artifacts.map((item, artifactIndex) => artifact(item, artifactIndex))
      : [],
  };
}

function stage(value: unknown, index: number): PipelineStage {
  const raw = object(value);
  const definition = object(raw.definition);
  const dependencies = raw.dependsOn ?? raw.dependencies;
  return {
    id: text(raw.id, `stage-${index + 1}`),
    name: text(raw.name, text(definition.title, text(raw.id, `Stage ${index + 1}`))),
    status: stageStatus(raw.status),
    dependsOn: Array.isArray(dependencies)
      ? dependencies.filter((id): id is string => typeof id === 'string')
      : [],
    attempts: Array.isArray(raw.attempts)
      ? raw.attempts.map((item, attemptIndex) => attempt(item, attemptIndex))
      : [],
  };
}

function budget(value: unknown, spentOverride?: number): Budget | undefined {
  const raw = object(value);
  const limit = raw.limit ?? raw.maximum_cost;
  if (typeof limit !== 'number' || !Number.isFinite(limit)) return undefined;
  return {
    spent: Math.max(0, spentOverride ?? finite(raw.spent)),
    limit: Math.max(0, limit),
    currency: text(raw.currency, 'USD'),
  };
}

export function normalizeRun(value: unknown, index = 0): PipelineRun {
  const raw = object(value);
  const rawAttempts = Array.isArray(raw.attempts) ? raw.attempts : [];
  const rawArtifacts = Array.isArray(raw.artifacts) ? raw.artifacts : [];
  const attempts = rawAttempts.map((item, attemptIndex) => ({
    ...attempt(item, attemptIndex),
    nodeId: text(object(item).nodeId),
  }));
  const artifacts = rawArtifacts.map((item, artifactIndex) => ({
    ...artifact(item, artifactIndex),
    attemptId: text(object(item).attemptId),
  }));
  const summaryStages = Array.isArray(raw.nodes)
    ? raw.nodes.map((item, stageIndex) => {
        const normalized = stage(item, stageIndex);
        return {
          ...normalized,
          attempts: attempts
            .filter((current) => current.nodeId === normalized.id)
            .map((current) => ({
              ...current,
              artifacts: artifacts.filter((output) => output.attemptId === current.id),
            })),
        };
      })
    : undefined;
  const spent = rawAttempts.reduce((total, item) => {
    const costs = object(item).costs;
    if (!Array.isArray(costs)) return total;
    return total + costs.reduce((sum, cost) => sum + Math.max(0, finite(object(cost).amount)), 0);
  }, 0);
  return {
    id: text(raw.id, `run-${index + 1}`),
    name: text(raw.name, text(raw.id, `Run ${index + 1}`)),
    status: runStatus(raw.status),
    createdAt: optionalText(raw.createdAt),
    updatedAt: optionalText(raw.updatedAt),
    budget: budget(raw.budget, spent),
    stages:
      summaryStages ??
      (Array.isArray(raw.stages)
        ? raw.stages.map((item, stageIndex) => stage(item, stageIndex))
        : []),
  };
}

export function normalizeRuns(value: unknown): PipelineRun[] {
  const raw = object(value);
  const runs = Array.isArray(value)
    ? value
    : Array.isArray(raw.runs)
      ? raw.runs
      : raw.run
        ? [raw.run]
        : raw.id
          ? [raw]
          : [];
  return runs.map((run, index) => normalizeRun(run, index));
}

function replaceById<T extends { id: string }>(items: T[], replacement: T): T[] {
  const found = items.some((item) => item.id === replacement.id);
  return found
    ? items.map((item) => (item.id === replacement.id ? replacement : item))
    : [...items, replacement];
}

function applyEvent(runs: PipelineRun[], event: PipelineEvent): PipelineRun[] {
  if (event.type === 'refresh') return runs;
  if (event.type === 'snapshot') return event.runs;
  if (event.type === 'run.upsert') return replaceById(runs, event.run);
  if (event.type === 'run.remove') return runs.filter((run) => run.id !== event.runId);
  return runs.map((run) => {
    if (run.id !== event.runId) return run;
    if (event.type === 'stage.upsert') {
      return { ...run, stages: replaceById(run.stages, event.stage) };
    }
    return {
      ...run,
      stages: run.stages.map((currentStage) => {
        if (currentStage.id !== event.stageId) return currentStage;
        if (event.type === 'attempt.upsert') {
          return { ...currentStage, attempts: replaceById(currentStage.attempts, event.attempt) };
        }
        return {
          ...currentStage,
          attempts: currentStage.attempts.map((currentAttempt) =>
            currentAttempt.id === event.attemptId
              ? { ...currentAttempt, logs: [...currentAttempt.logs, event.line] }
              : currentAttempt,
          ),
        };
      }),
    };
  });
}

function reconcileSelection(state: DashboardState): DashboardState {
  const selectedRun = state.runs.find((run) => run.id === state.selectedRunId) ?? state.runs[0];
  const selectedStage =
    selectedRun?.stages.find((item) => item.id === state.selectedStageId) ?? selectedRun?.stages[0];
  const selectedAttempt =
    selectedStage?.attempts.find((item) => item.id === state.selectedAttemptId) ??
    selectedStage?.attempts.at(-1);
  return {
    ...state,
    selectedRunId: selectedRun?.id,
    selectedStageId: selectedStage?.id,
    selectedAttemptId: selectedAttempt?.id,
  };
}

export function dashboardReducer(state: DashboardState, action: DashboardAction): DashboardState {
  if (action.type === 'load')
    return reconcileSelection({ ...state, runs: action.runs, error: undefined });
  if (action.type === 'event') {
    return reconcileSelection({
      ...state,
      runs: applyEvent(state.runs, action.event),
      lastEventId: action.eventId ?? state.lastEventId,
      error: undefined,
    });
  }
  if (action.type === 'select-run') {
    return reconcileSelection({
      ...state,
      selectedRunId: action.runId,
      selectedStageId: undefined,
      selectedAttemptId: undefined,
      lastEventId: undefined,
    });
  }
  if (action.type === 'select-stage') {
    return reconcileSelection({
      ...state,
      selectedStageId: action.stageId,
      selectedAttemptId: undefined,
    });
  }
  if (action.type === 'select-attempt') return { ...state, selectedAttemptId: action.attemptId };
  if (action.type === 'connection') return { ...state, connection: action.connection };
  return { ...state, error: action.message };
}

export function decodePipelineEvent(eventName: string, value: unknown): PipelineEvent | undefined {
  const raw = object(value);
  const type = eventName === 'message' ? text(raw.type) : eventName;
  if (type === 'snapshot') return { type, runs: normalizeRuns(raw.runs ?? value) };
  if (type === 'run.upsert' && raw.run) return { type, run: normalizeRun(raw.run) };
  if (type === 'run.remove' && typeof raw.runId === 'string') return { type, runId: raw.runId };
  if (type === 'stage.upsert' && typeof raw.runId === 'string' && raw.stage) {
    return { type, runId: raw.runId, stage: stage(raw.stage, 0) };
  }
  if (
    type === 'attempt.upsert' &&
    typeof raw.runId === 'string' &&
    typeof raw.stageId === 'string' &&
    raw.attempt
  ) {
    return {
      type,
      runId: raw.runId,
      stageId: raw.stageId,
      attempt: attempt(raw.attempt, 0),
    };
  }
  if (
    type === 'attempt.log' &&
    typeof raw.runId === 'string' &&
    typeof raw.stageId === 'string' &&
    typeof raw.attemptId === 'string' &&
    typeof raw.line === 'string'
  ) {
    return {
      type,
      runId: raw.runId,
      stageId: raw.stageId,
      attemptId: raw.attemptId,
      line: raw.line,
    };
  }
  if (
    [
      'run',
      'node',
      'attempt',
      'log',
      'artifact',
      'quality',
      'agent',
      'operator_message',
      'approval',
      'cost',
    ].includes(type) &&
    typeof raw.run_id === 'string'
  ) {
    return { type: 'refresh', runId: raw.run_id };
  }
  return undefined;
}

export function selectedEntities(state: DashboardState) {
  const run = state.runs.find((item) => item.id === state.selectedRunId);
  const stage = run?.stages.find((item) => item.id === state.selectedStageId);
  const attempt = stage?.attempts.find((item) => item.id === state.selectedAttemptId);
  return { run, stage, attempt };
}
