import type { RunProgress } from './types';

export function progressTitle(progress: RunProgress): string {
  if (progress.finished === 'canceled') return 'Run canceled';
  if (progress.finished === 'succeeded') return 'Scene processing complete';
  if (progress.finished) {
    return progress.viewerPath && progress.result?.delivery?.status !== 'unavailable'
      ? 'Preview available — checks did not pass'
      : 'Processing could not finish';
  }
  if (progress.paused) return 'Run paused';
  if (progress.question) return 'Waiting for your answer';
  if (progress.currentStages.length) return `Running ${progress.currentStages.join(', ')}`;
  if (progress.stopped.length) return 'Processing stopped';
  return 'Waiting for the worker';
}

export function workerNotice(progress: RunProgress): string | null {
  if (progress.finished || progress.paused || progress.question) return null;
  if (!progress.worker) return 'This server has not reported worker availability.';
  if (progress.worker.status === 'unavailable') {
    return progress.currentStages.length
      ? 'No worker is connected for this run. Started steps may still finish; processing will continue when a worker reconnects.'
      : 'Waiting for a worker. Your upload is saved and will be picked up when a worker connects.';
  }
  return progress.handsOff
    ? 'Worker connected. Processing automatically; you can leave this page.'
    : null;
}
