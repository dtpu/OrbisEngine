import assert from 'node:assert/strict';
import test from 'node:test';
import { approvalArtifacts, parseSseBlock } from '../src/dashboard/api.ts';
import { layoutDag } from '../src/dashboard/dag.ts';
import { redactSecrets } from '../src/dashboard/redact.ts';
import {
  dashboardReducer,
  decodePipelineEvent,
  initialState,
  normalizeRuns,
} from '../src/dashboard/state.ts';
import type { PipelineRun, PipelineStage } from '../src/dashboard/types.ts';

const stages: PipelineStage[] = [
  { id: 'ingest', name: 'Ingest', status: 'succeeded', dependsOn: [], attempts: [] },
  {
    id: 'camera',
    name: 'Camera',
    status: 'running',
    dependsOn: ['ingest'],
    attempts: [
      {
        id: 'camera-1',
        number: 1,
        status: 'running',
        logs: ['starting'],
        artifacts: [],
      },
    ],
  },
  {
    id: 'world',
    name: 'World',
    status: 'pending',
    dependsOn: ['camera'],
    attempts: [],
  },
];

const run: PipelineRun = {
  id: 'run-1',
  name: 'Lobby',
  status: 'running',
  budget: { spent: 2.5, limit: 10, currency: 'USD' },
  stages,
};

test('normalizes untrusted snapshots and rejects unknown statuses', () => {
  const runs = normalizeRuns({
    runs: [
      {
        id: 'run-1',
        name: 'Lobby',
        status: 'not-a-status',
        budget: { spent: 4, limit: 8, currency: 'USD' },
        stages: [{ id: 'world', status: 'also-invalid', logs: 'not-an-array' }],
      },
    ],
  });
  assert.equal(runs[0].status, 'queued');
  assert.equal(runs[0].stages[0].status, 'pending');
  assert.deepEqual(runs[0].stages[0].attempts, []);
  assert.deepEqual(normalizeRuns(null), []);
});

test('normalizes the orchestrator run summary contract', () => {
  const normalized = normalizeRuns({
    id: 'run-2',
    status: 'waiting_human',
    budget: { currency: 'USD', maximum_cost: 12 },
    nodes: [
      {
        id: 'clean',
        status: 'waiting_human',
        dependencies: ['ingest'],
        definition: { title: 'Clean frames' },
      },
    ],
    attempts: [
      {
        id: 'attempt:clean:1',
        nodeId: 'clean',
        number: 1,
        status: 'succeeded',
        costs: [{ amount: 3.25 }],
      },
    ],
    artifacts: [
      {
        id: 'artifact:frames',
        role: 'clean_frames',
        attemptId: 'attempt:clean:1',
        mediaType: 'image/png',
        size: 120,
      },
    ],
  })[0];
  assert.equal(normalized.status, 'waiting_human');
  assert.equal(normalized.stages[0].name, 'Clean frames');
  assert.equal(normalized.stages[0].attempts[0].artifacts[0].kind, 'image');
  assert.deepEqual(normalized.budget, { spent: 3.25, limit: 12, currency: 'USD' });
  assert.deepEqual(approvalArtifacts(normalized.stages[0].attempts[0].artifacts), {
    clean_frames: ['artifact:frames'],
  });
});

test('reducer selects defaults and updates an attempt log immutably', () => {
  const loaded = dashboardReducer(initialState, { type: 'load', runs: [run] });
  assert.equal(loaded.selectedRunId, 'run-1');
  assert.equal(loaded.selectedStageId, 'ingest');
  const selected = dashboardReducer(loaded, { type: 'select-stage', stageId: 'camera' });
  assert.equal(selected.selectedAttemptId, 'camera-1');
  const updated = dashboardReducer(selected, {
    type: 'event',
    eventId: '42',
    event: {
      type: 'attempt.log',
      runId: 'run-1',
      stageId: 'camera',
      attemptId: 'camera-1',
      line: 'finished',
    },
  });
  assert.deepEqual(updated.runs[0].stages[1].attempts[0].logs, ['starting', 'finished']);
  assert.deepEqual(run.stages[1].attempts[0].logs, ['starting']);
  assert.equal(updated.lastEventId, '42');
});

test('event decoding accepts explicit and generic SSE event names', () => {
  assert.deepEqual(
    decodePipelineEvent('attempt.log', {
      runId: 'r',
      stageId: 's',
      attemptId: 'a',
      line: 'hello',
    }),
    {
      type: 'attempt.log',
      runId: 'r',
      stageId: 's',
      attemptId: 'a',
      line: 'hello',
    },
  );
  assert.equal(
    decodePipelineEvent('message', { type: 'run.remove', runId: 'r' })?.type,
    'run.remove',
  );
  assert.deepEqual(decodePipelineEvent('node', { run_id: 'r', subject_id: 'clean', payload: {} }), {
    type: 'refresh',
    runId: 'r',
  });
  assert.equal(decodePipelineEvent('unknown', {}), undefined);
});

test('SSE parsing preserves IDs, multiline data, event names, and retry hints', () => {
  assert.deepEqual(
    parseSseBlock(
      ': heartbeat\nid: 17\nevent: attempt.log\nretry: 2500\ndata: {"line":"one",\ndata: "more":"two"}',
    ),
    {
      id: '17',
      event: 'attempt.log',
      retry: 2500,
      data: '{"line":"one",\n"more":"two"}',
    },
  );
  assert.equal(parseSseBlock(': heartbeat'), undefined);
});

test('DAG layout is deterministic and orders dependencies left to right', () => {
  const first = layoutDag(stages);
  const second = layoutDag(stages);
  assert.deepEqual(first, second);
  const positions = new Map(first.nodes.map((node) => [node.id, node]));
  assert.ok(positions.get('ingest')!.x < positions.get('camera')!.x);
  assert.ok(positions.get('camera')!.x < positions.get('world')!.x);

  const cyclic = [
    { ...stages[0], id: 'a', dependsOn: ['b'] },
    { ...stages[0], id: 'b', dependsOn: ['a'] },
  ];
  assert.equal(layoutDag(cyclic).nodes.length, 2);
});

test('operator-visible text redacts common credentials', () => {
  const value =
    'Authorization: Bearer abc.def-123 api_key=super-secret AKIA1234567890ABCDEF https://x.test/?token=hidden';
  const redacted = redactSecrets(value);
  assert.doesNotMatch(redacted, /abc\.def|super-secret|AKIA123|token=hidden/);
  assert.match(redacted, /\[REDACTED/);
});
