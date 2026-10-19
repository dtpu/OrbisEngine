import assert from 'node:assert/strict';
import { test } from 'node:test';
import { progressTitle } from '../lib/run-progress';
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
