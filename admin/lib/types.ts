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

export interface StageDefinition {
  id: string;
  title: string;
  kind: 'compute' | 'agent' | 'human' | 'expand';
  executor: string | null;
  resources?: { task_queue?: string };
  quality?: { human_approval?: boolean; agent_rubric?: string | null };
}

export interface RunNode {
  id: string;
  status: StageStatus;
  dependencies: string[];
  selectedAttemptId: string | null;
  blockedReason: string | null;
  definition: StageDefinition;
}

export interface AttemptCost {
  provider?: string;
  amount?: number;
  currency?: string;
  estimated?: boolean;
}

export interface RunAttempt {
  id: string;
  nodeId: string;
  number: number;
  status: StageStatus;
  parameters: Record<string, unknown>;
  costs: AttemptCost[];
  error: string | null;
  hypothesis?: string | null;
  retryOf?: string | null;
  providerOperationId?: string | null;
  inputArtifactIds?: string[];
  outputArtifactIds?: string[];
  startedAt?: string | null;
  finishedAt?: string | null;
}

export interface RunArtifact {
  id: string;
  role: string;
  sha256: string;
  size: number;
  mediaType: string;
  attemptId: string | null;
  previewArtifactId: string | null;
  /** File name the stage wrote (`clean_frame.png`); older API builds omit it. */
  name?: string;
  contract?: string | null;
  createdAt?: string;
}

export interface ProviderClaim {
  id: string;
  provider: string;
  operation: string;
  status: string;
  providerOperationId: string | null;
  estimatedCost: number | null;
  createdAt: string;
  updatedAt: string;
}

/** A reviewing agent's verdict on one attempt; `rationale` is its evidence. */
export interface QualityReview {
  id: string;
  nodeId: string;
  attemptId: string;
  verdict: 'pass' | 'retry' | 'needs_human' | string;
  rationale: string;
  hypothesis?: string | null;
  proposedParameters?: Record<string, unknown> | null;
  decidedBy: string;
  createdAt: string;
}

/** One `transcript.jsonl` the reviewing agent wrote (or is writing) for an attempt. */
export interface ReviewTranscript {
  /** Present in the cross-run listing; the per-run listing implies it. */
  runId?: string;
  nodeId: string;
  attemptId: string;
  bytes: number;
  updatedAt: string;
  finished: boolean;
}

/** A line of `codex exec --json` output; anything that was not JSON arrives as `raw`. */
export interface TranscriptEvent {
  type: string;
  text?: string;
  thread_id?: string;
  usage?: Record<string, unknown>;
  item?: {
    id?: string;
    type?: string;
    text?: string;
    command?: string;
    aggregated_output?: string;
    exit_code?: number | null;
    status?: string;
    changes?: unknown;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

export interface TranscriptTail {
  offset: number;
  size: number;
  reset: boolean;
  finished: boolean;
  updatedAt: string;
  events: TranscriptEvent[];
}

export interface GraphOptions {
  marble: 'video' | 'image' | 'multi' | 'both' | 'none';
  people: number;
  all_people: boolean;
  objects: boolean;
  finetune: boolean;
}

export interface PipelineRun {
  id: string;
  status: RunStatus;
  /** What the run row says; `status` is derived live from the nodes until the run finishes. */
  recordedStatus?: RunStatus;
  parentRunId?: string | null;
  branchKey?: string | null;
  graphVersion: string;
  sourceSha256: string;
  configuration: { options: GraphOptions };
  budget?: { currency: string; maximum_cost: number | null };
  createdAt?: string;
  updatedAt?: string;
  nodes: RunNode[];
  attempts?: RunAttempt[];
  artifacts?: RunArtifact[];
  providerClaims?: ProviderClaim[];
  reviews?: QualityReview[];
  /** Live run metadata derived from the journal. */
  live?: { paused?: boolean; canceled?: boolean } | null;
  liveError?: string | null;
}

export type RunCommand = 'reject' | 'retry' | 'pause' | 'resume' | 'cancel';

/** The outcomes an operator may record for a provider claim that ended unobserved. */
export type ReconcileStatus = 'completed' | 'failed';

/** The control plane publishes these defaults and supported values from RunOptions. */
export interface OptionProperty {
  title?: string;
  description?: string;
  type: 'string' | 'integer' | 'number' | 'boolean';
  default?: string | number | boolean;
  enum?: string[];
  minimum?: number;
  maximum?: number;
  deprecated?: boolean;
}

export interface OptionsSchema {
  properties: Record<string, OptionProperty>;
  additionalProperties: boolean;
}

export interface SetupCheck {
  name: string;
  status: 'okay' | 'ok' | 'blocked' | 'unverified';
  detail: string;
  action?: string;
}

export interface SetupReport {
  ready: boolean;
  status: 'okay' | 'blocked' | 'unverified';
  checks: SetupCheck[];
}

export interface StageProgress {
  name: string;
  summary: string;
  status: StageStatus;
  reason: string | null;
  nextAction: string;
  dependencies: string[];
}

export interface RunPreview {
  label: string;
  path: string;
  kind: 'video' | 'image' | 'report';
  bytes: number;
  url: string;
}

export interface RunProgress {
  runId: string;
  finished: string | null;
  paused: boolean;
  question: string | null;
  currentStages: string[];
  stopped: StageProgress[];
  stages: StageProgress[];
  previews: RunPreview[];
}
