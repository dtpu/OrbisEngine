import assert from 'node:assert/strict';
import { test } from 'node:test';
import { loadWorldReview, saveWorldReview } from '../lib/client';
import {
  blankWorldReview,
  reviewCriteria,
  worldReviewReady,
  type ManualWorldReview,
  type WorldReviewBundle,
} from '../lib/world-review';

function fixture(): WorldReviewBundle {
  const diagnosticId = 'diagnostic-retained';
  const planSha256 = 'a'.repeat(64);
  const samples = [0, 50, 100].map((sourceFrame, index) => ({
    id: `checkpoint-${index}`,
    sourceFrame,
    timeSeconds: sourceFrame / 25,
    critical: index === 1,
    pairSha256: String(index + 1).repeat(64),
    pairUrl: `/api/pipeline/runs/run-one/world-review/${diagnosticId}/pairs/${sourceFrame}?planSha256=${planSha256}`,
  }));
  return {
    diagnosticId,
    planSha256,
    scope: 'static-world-only',
    samples,
    canSubmit: true,
    holdReason: null,
    reviewTemplate: {
      schema: 'wander.static-world-review/1',
      planSha256,
      judgeKind: 'manual',
      reviewer: 'Prior reviewer',
      acknowledgements: {
        estimatedCameraCorrespondence: true,
        unobservedGeometryNotVerified: true,
        staticWorldOnly: true,
      },
      samples: samples.map(({ id, pairSha256 }) => ({
        id,
        pairSha256,
        criteria: {
          room_layout: { status: 'pass', criticalFailure: false, reason: 'Prior layout' },
          scene_coverage: { status: 'pass', criticalFailure: false, reason: 'Prior coverage' },
          appearance: { status: 'pass', criticalFailure: false, reason: 'Prior appearance' },
        },
      })),
    },
  };
}

function completedUncertainReview(bundle: WorldReviewBundle): ManualWorldReview {
  const result = blankWorldReview(bundle);
  result.reviewer = 'Operator';
  for (const key of Object.keys(
    result.acknowledgements,
  ) as (keyof typeof result.acknowledgements)[]) {
    result.acknowledgements[key] = true;
  }
  for (const sample of result.samples) {
    for (const criterion of reviewCriteria) {
      sample.criteria[criterion].reason = 'I cannot verify the surface from this angle.';
    }
  }
  return result;
}

test('loading evidence binds a fresh manual review to the exact plan and pairs', () => {
  const bundle = fixture();
  const result = blankWorldReview(bundle);
  assert.equal(result.schema, 'wander.static-world-review/1');
  assert.equal(result.judgeKind, 'manual');
  assert.equal(result.planSha256, bundle.planSha256);
  assert.equal(result.reviewer, '');
  assert.deepEqual(Object.values(result.acknowledgements), [false, false, false]);
  assert.deepEqual(
    result.samples.map(({ id, pairSha256 }) => ({ id, pairSha256 })),
    bundle.samples.map(({ id, pairSha256 }) => ({ id, pairSha256 })),
  );
  for (const sample of result.samples) {
    assert.deepEqual(Object.keys(sample.criteria), [...reviewCriteria]);
    for (const judgment of Object.values(sample.criteria)) {
      assert.deepEqual(judgment, { status: 'unknown', criticalFailure: false, reason: '' });
    }
  }
  assert.equal(worldReviewReady(result), false);
});

test('readiness requires an explicit reviewer, each acknowledgement and every reason', () => {
  const bundle = fixture();
  const valid = completedUncertainReview(bundle);
  assert.equal(worldReviewReady(valid), true);
  assert.equal(worldReviewReady({ ...valid, reviewer: ' \t\n' }), false);
  for (const key of Object.keys(valid.acknowledgements)) {
    assert.equal(
      worldReviewReady({
        ...valid,
        acknowledgements: { ...valid.acknowledgements, [key]: false },
      }),
      false,
      key,
    );
  }
  const missingReason = structuredClone(valid);
  missingReason.samples[2].criteria.appearance.reason = '\n ';
  assert.equal(worldReviewReady(missingReason), false);
  assert.equal(worldReviewReady({ ...valid, samples: valid.samples.slice(0, 2) }), false);
});

test('client submits uncertainty with plan and pair bindings for server adjudication', async () => {
  const bundle = fixture();
  const review = completedUncertainReview(bundle);
  const originalFetch = globalThis.fetch;
  const requests: { url: string; init?: RequestInit }[] = [];
  const blocked = {
    reviewStatus: 'blocked',
    runStatus: 'failed',
    errors: ['checkpoint-0/room_layout: uncertainty remains'],
    completionReason: null,
  };
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    requests.push({ url: String(input), init });
    return Response.json(init?.method === 'POST' ? blocked : bundle);
  }) as typeof fetch;
  try {
    assert.equal(worldReviewReady(review), true);
    assert.deepEqual(await loadWorldReview('run/one'), bundle);
    assert.deepEqual(await saveWorldReview('run/one', bundle, review), blocked);
    assert.equal(requests[0].url, '/api/pipeline/runs/run%2Fone/world-review');
    assert.equal(requests[0].init?.cache, 'no-store');
    assert.equal(requests[1].url, requests[0].url);
    assert.equal(requests[1].init?.method, 'POST');
    assert.deepEqual(JSON.parse(String(requests[1].init?.body)), {
      diagnosticId: bundle.diagnosticId,
      planSha256: bundle.planSha256,
      result: review,
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});
