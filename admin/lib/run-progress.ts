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
