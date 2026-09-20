export type RunStatus =
  | 'queued'
  | 'running'
  | 'waiting_human'
  | 'paused'
  | 'blocked'
  | 'succeeded'
  | 'failed'
  | 'canceled';

export type StageStatus =
  | 'pending'
  | 'queued'
  | 'ready'
  | 'running'
  | 'waiting_agent'
  | 'waiting_human'
  | 'blocked'
  | 'unknown'
  | 'succeeded'
  | 'failed'
  | 'canceled'
  | 'skipped';

export interface Artifact {
  id: string;
  name: string;
  role?: string;
  kind: 'image' | 'video' | 'audio' | 'text' | 'json' | 'file';
  url?: string;
  preview?: string;
  contentType?: string;
  bytes?: number;
}

export interface StageAttempt {
  id: string;
  number: number;
  status: StageStatus;
  startedAt?: string;
  finishedAt?: string;
  summary?: string;
  logs: string[];
  artifacts: Artifact[];
}

export interface PipelineStage {
  id: string;
  name: string;
  status: StageStatus;
  dependsOn: string[];
  attempts: StageAttempt[];
}

export interface Budget {
  spent: number;
  limit: number;
  currency: string;
}

export interface PipelineRun {
  id: string;
  name: string;
  status: RunStatus;
  createdAt?: string;
  updatedAt?: string;
  budget?: Budget;
  stages: PipelineStage[];
}

export interface DashboardState {
  runs: PipelineRun[];
  selectedRunId?: string;
  selectedStageId?: string;
  selectedAttemptId?: string;
  connection: 'connecting' | 'live' | 'reconnecting' | 'offline';
  lastEventId?: string;
  error?: string;
}

export type PipelineEvent =
  | { type: 'snapshot'; runs: PipelineRun[] }
  | { type: 'run.upsert'; run: PipelineRun }
  | { type: 'run.remove'; runId: string }
  | { type: 'stage.upsert'; runId: string; stage: PipelineStage }
  | { type: 'attempt.upsert'; runId: string; stageId: string; attempt: StageAttempt }
  | { type: 'attempt.log'; runId: string; stageId: string; attemptId: string; line: string }
  | { type: 'refresh'; runId: string };

export type DashboardAction =
  | { type: 'load'; runs: PipelineRun[] }
  | { type: 'event'; event: PipelineEvent; eventId?: string }
  | { type: 'select-run'; runId: string }
  | { type: 'select-stage'; stageId: string }
  | { type: 'select-attempt'; attemptId: string }
  | { type: 'connection'; connection: DashboardState['connection'] }
  | { type: 'error'; message?: string };

export type PipelineCommand = 'approve' | 'reject' | 'retry' | 'pause' | 'resume' | 'cancel';
