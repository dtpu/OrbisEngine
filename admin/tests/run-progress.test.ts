import assert from 'node:assert/strict';
import { test } from 'node:test';
import { progressTitle, workerNotice } from '../lib/run-progress';
import type { RunProgress } from '../lib/types';

const progress: RunProgress = {
  runId: 'test',
  finished: null,
  paused: false,
  question: null,
  currentStages: [],
  stopped: [],
  stages: [],
  previews: [],
};

test('a playable rejected preview is not presented as a successful scene', () => {
  assert.equal(
    progressTitle({ ...progress, finished: 'failed', viewerPath: '/fourd.html?scene=test' }),
    'Preview available — checks did not pass',
  );
  assert.equal(progressTitle({ ...progress, finished: 'failed' }), 'Processing could not finish');
  assert.equal(progressTitle({ ...progress, finished: 'succeeded' }), 'Scene processing complete');
});

test('cancellation stays visible despite a preview or late active stage', () => {
  assert.equal(
    progressTitle({
      ...progress,
      finished: 'canceled',
      viewerPath: '/preview',
      currentStages: ['clean'],
    }),
    'Run canceled',
  );
});

test('working and explicitly paused uploads have distinct progress', () => {
  assert.equal(progressTitle({ ...progress, currentStages: ['clean'] }), 'Running clean');
  assert.equal(progressTitle({ ...progress, paused: true }), 'Run paused');
});

const offline: RunProgress = {
  ...progress,
  handsOff: true,
  worker: { status: 'unavailable', matchingWorkers: 0, busyWorkers: 0, detail: 'No live worker.' },
};

test('a saved upload with no worker never claims it is processing automatically', () => {
  assert.match(workerNotice(offline)!, /Waiting for a worker.*upload is saved/);
  assert.doesNotMatch(workerNotice(offline)!, /Processing automatically/);
  assert.match(
    workerNotice({ ...offline, currentStages: ['clean'] })!,
    /Started steps may still finish/,
  );
  assert.equal(progressTitle({ ...offline, currentStages: ['clean'] }), 'Running clean');
});

test('worker reconnect updates the message without masking terminal or manual holds', () => {
  const connected: RunProgress = {
    ...offline,
    worker: { status: 'available', matchingWorkers: 1, busyWorkers: 1, detail: 'Connected.' },
  };
  assert.match(workerNotice(connected)!, /Worker connected.*Processing automatically/);
  for (const state of [{ finished: 'failed' }, { paused: true }, { question: 'Review output' }]) {
    assert.equal(workerNotice({ ...offline, ...state }), null);
    assert.equal(workerNotice({ ...connected, ...state }), null);
  }
  assert.equal(workerNotice({ ...connected, handsOff: false }), null);
  assert.equal(workerNotice(progress), 'This server has not reported worker availability.');
});

test('admission limits keep queued uploads distinct from disconnected workers', () => {
  const exhausted: RunProgress = {
    ...offline,
    worker: { ...offline.worker!, reason: 'admission_exhausted' },
  };
  const busy: RunProgress = {
    ...offline,
    worker: { ...offline.worker!, reason: 'admission_busy' },
  };
  assert.equal(progressTitle(exhausted), 'Upload saved — processing limit reached');
  assert.match(workerNotice(exhausted)!, /upload is saved.*limit has been reached.*operator/);
  assert.doesNotMatch(workerNotice(exhausted)!, /connect|Processing automatically/);
  assert.equal(progressTitle(busy), 'Upload queued');
  assert.match(workerNotice(busy)!, /worker is handling another upload.*remains queued/);
  for (const limited of [exhausted, busy]) {
    assert.equal(progressTitle({ ...limited, currentStages: ['clean'] }), 'Running clean');
    assert.equal(progressTitle({ ...limited, paused: true }), 'Run paused');
    assert.equal(progressTitle({ ...limited, finished: 'canceled' }), 'Run canceled');
    assert.equal(workerNotice({ ...limited, finished: 'failed' }), null);
    assert.equal(workerNotice({ ...limited, paused: true }), null);
  }
});
