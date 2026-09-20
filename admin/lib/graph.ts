import type { PipelineRun, RunNode, StageStatus } from './types';

export interface OrderedNode {
  node: RunNode;
  /** Longest dependency chain above this stage: stages at the same depth can run together. */
  depth: number;
}

function dependencyIds(node: RunNode): string[] {
  return (node.dependencies as unknown[]).flatMap((dependency) => {
    if (typeof dependency === 'string') return [dependency];
    if (dependency && typeof dependency === 'object') {
      const record = dependency as Record<string, unknown>;
      const id = record.node_id ?? record.id ?? record.stage_id;
      return typeof id === 'string' ? [id] : [];
    }
    return [];
  });
}

/**
 * Stages in the order work actually flows: by dependency depth, then by the API's order. The API
 * lists nodes alphabetically, which puts `marble_video` above `pi3x` and hides the structure.
 */
export function orderStages(nodes: RunNode[]): OrderedNode[] {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const depth = new Map<string, number>();
  const visiting = new Set<string>();

  function visit(id: string): number {
    const known = depth.get(id);
    if (known !== undefined) return known;
    if (visiting.has(id)) return 0;
    visiting.add(id);
    const node = byId.get(id);
    const parents = node ? dependencyIds(node).filter((parent) => byId.has(parent)) : [];
    const value = parents.length ? Math.max(...parents.map(visit)) + 1 : 0;
    visiting.delete(id);
    depth.set(id, value);
    return value;
  }

  return nodes
    .map((node, index) => ({ node, depth: visit(node.id), index }))
    .sort((a, b) => a.depth - b.depth || a.index - b.index)
    .map(({ node, depth: value }) => ({ node, depth: value }));
}

/**
 * Median wall time of every attempt of each stage that has finished before, across all runs the
 * operator can see. It is the only honest basis for a progress bar: the pipeline reports no
 * percentage, but it does report when stages started and finished.
 */
export function typicalDurations(runs: PipelineRun[]): Map<string, number> {
  const samples = new Map<string, number[]>();
  for (const run of runs) {
    for (const attempt of run.attempts ?? []) {
      if (attempt.status !== 'succeeded' || !attempt.startedAt || !attempt.finishedAt) continue;
      const start = Date.parse(pin(attempt.startedAt));
      const end = Date.parse(pin(attempt.finishedAt));
      if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) continue;
      samples.set(attempt.nodeId, [...(samples.get(attempt.nodeId) ?? []), end - start]);
    }
  }
  const typical = new Map<string, number>();
  for (const [nodeId, values] of samples) {
    const sorted = [...values].sort((a, b) => a - b);
    typical.set(nodeId, sorted[Math.floor(sorted.length / 2)]);
  }
  return typical;
}

/** Attempt statuses during which a stage was actually executing. Blocked and waiting do not count. */
const RAN = new Set(['running', 'succeeded', 'failed', 'canceled', 'timed_out']);

/**
 * Wall time during which at least one stage of the run was executing: the union of every
 * attempt's [started, finished] interval, open intervals ending now. Hours spent parked on the
 * spend guard or waiting for a person are not "running", so they are left out.
 */
export function activeMs(run: PipelineRun, now = Date.now()): number | null {
  const intervals: Array<[number, number]> = [];
  for (const attempt of run.attempts ?? []) {
    if (!RAN.has(attempt.status) || !attempt.startedAt) continue;
    const start = Date.parse(pin(attempt.startedAt));
    const end = attempt.finishedAt ? Date.parse(pin(attempt.finishedAt)) : now;
    if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) continue;
    intervals.push([start, end]);
  }
  if (intervals.length === 0) return null;
  intervals.sort((a, b) => a[0] - b[0]);
  let total = 0;
  let [from, to] = intervals[0];
  for (const [start, end] of intervals.slice(1)) {
    if (start <= to) {
      to = Math.max(to, end);
    } else {
      total += to - from;
      [from, to] = [start, end];
    }
  }
  return total + (to - from);
}

const UNRESOLVED_CLAIM = new Set(['pending', 'unknown']);
/** While the stage is still working, its claim is simply open, not orphaned. */
const IN_FLIGHT_NODE = new Set(['running', 'ready', 'waiting_agent', 'queued']);

/**
 * Claims worth showing an operator: a provider call with no recorded outcome whose stage is not
 * still working. A claim is written before the call and resolved when it returns, so "pending"
 * on a running stage is the normal in-flight state.
 */
export function unrecordedClaims(run: PipelineRun, nodeId?: string) {
  const status = new Map(run.nodes.map((node) => [node.id, node.status]));
  return (run.providerClaims ?? []).filter((claim) => {
    if (!UNRESOLVED_CLAIM.has(claim.status)) return false;
    if (nodeId && claim.operation !== nodeId) return false;
    const node = status.get(claim.operation);
    return node === undefined || !IN_FLIGHT_NODE.has(node);
  });
}

/** True while any attempt is executing, so the active clock should tick. */
export function isExecuting(run: PipelineRun): boolean {
  return (run.attempts ?? []).some((attempt) => attempt.status === 'running' && attempt.startedAt);
}

function pin(value: string): string {
  return /[zZ]|[+-]\d\d:\d\d$/.test(value) ? value : `${value}Z`;
}

export function dependenciesOf(node: RunNode): string[] {
  return dependencyIds(node);
}

export function stagesWaitingOnYou(run: PipelineRun): RunNode[] {
  return run.nodes.filter((node) => node.status === 'waiting_human');
}

export function countByStatus(run: PipelineRun): Partial<Record<StageStatus, number>> {
  const counts: Partial<Record<StageStatus, number>> = {};
  for (const node of run.nodes) counts[node.status] = (counts[node.status] ?? 0) + 1;
  return counts;
}

/** The next stage an operator should look at: a gate first, then a failure, then a block. */
export function stageNeedingAttention(run: PipelineRun): RunNode | undefined {
  const order: StageStatus[] = ['waiting_human', 'failed', 'blocked', 'running'];
  for (const status of order) {
    const match = run.nodes.find((node) => node.status === status);
    if (match) return match;
  }
  return undefined;
}
