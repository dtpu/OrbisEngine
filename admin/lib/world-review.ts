export const reviewCriteria = ['room_layout', 'scene_coverage', 'appearance'] as const;
export type ReviewCriterion = (typeof reviewCriteria)[number];
export type ReviewJudgment = {
  status: 'pass' | 'fail' | 'unknown';
  criticalFailure: boolean;
  reason: string;
};
export type ManualWorldReview = {
  schema: 'wander.static-world-review/1';
  planSha256: string;
  judgeKind: 'manual';
  reviewer: string;
  acknowledgements: {
    estimatedCameraCorrespondence: boolean;
    unobservedGeometryNotVerified: boolean;
    staticWorldOnly: boolean;
  };
  samples: {
    id: string;
    pairSha256: string;
    criteria: Record<ReviewCriterion, ReviewJudgment>;
  }[];
};
export type WorldReviewBundle = {
  diagnosticId: string;
  planSha256: string;
  scope: string;
  samples: {
    id: string;
    sourceFrame: number;
    timeSeconds: number;
    critical: boolean;
    pairSha256: string;
    pairUrl: string;
  }[];
  reviewTemplate: ManualWorldReview;
  canSubmit: boolean;
  holdReason: string | null;
};
export type WorldReviewOutcome = {
  reviewStatus: 'passed' | 'failed' | 'blocked';
  runStatus: string;
  errors: string[];
  completionReason: string | null;
};

/** New evidence always needs new judgments, even if the server has an older result. */
export function blankWorldReview(bundle: WorldReviewBundle): ManualWorldReview {
  return {
    schema: 'wander.static-world-review/1',
    judgeKind: 'manual',
    planSha256: bundle.planSha256,
    reviewer: '',
    acknowledgements: {
      estimatedCameraCorrespondence: false,
      unobservedGeometryNotVerified: false,
      staticWorldOnly: false,
    },
    samples: bundle.samples.map((sample) => ({
      id: sample.id,
      pairSha256: sample.pairSha256,
      criteria: Object.fromEntries(
        reviewCriteria.map((key) => [
          key,
          { status: 'unknown', criticalFailure: false, reason: '' },
        ]),
      ) as Record<ReviewCriterion, ReviewJudgment>,
    })),
  };
}

export function worldReviewReady(result: ManualWorldReview): boolean {
  return (
    !!result.reviewer.trim() &&
    Object.values(result.acknowledgements).every((value) => value === true) &&
    result.samples.length >= 3 &&
    result.samples.every((sample) =>
      reviewCriteria.every((key) => !!sample.criteria[key].reason.trim()),
    )
  );
}
