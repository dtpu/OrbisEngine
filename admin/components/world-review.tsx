'use client';

import Image from 'next/image';
import { useRef, useState } from 'react';
import { loadWorldReview, saveWorldReview } from '@/lib/client';
import {
  blankWorldReview,
  reviewCriteria,
  worldReviewReady,
  type ManualWorldReview,
  type ReviewCriterion,
  type ReviewJudgment,
  type WorldReviewBundle,
  type WorldReviewOutcome,
} from '@/lib/world-review';

const criteriaLabels: Record<ReviewCriterion, string> = {
  room_layout: 'Room layout',
  scene_coverage: 'Scene coverage',
  appearance: 'Appearance',
};
const acknowledgements = {
  estimatedCameraCorrespondence: 'I understand that camera alignment is estimated.',
  unobservedGeometryNotVerified: 'I am not approving surfaces that the video never showed.',
  staticWorldOnly: 'This review covers the static room only, not people, objects or walking.',
} as const;

export function WorldReview({ runId }: { runId: string }) {
  const [bundle, setBundle] = useState<WorldReviewBundle | null>(null);
  const [review, setReview] = useState<ManualWorldReview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<WorldReviewOutcome | null>(null);
  const [loadedImages, setLoadedImages] = useState<Set<string>>(new Set());
  const [imageGeneration, setImageGeneration] = useState(0);
  const currentImageGeneration = useRef(0);

  async function load() {
    currentImageGeneration.current += 1;
    setBundle(null);
    setReview(null);
    setLoadedImages(new Set());
    setBusy(true);
    setError(null);
    setOutcome(null);
    try {
      const next = await loadWorldReview(runId);
      setBundle(next);
      setReview(blankWorldReview(next));
      setLoadedImages(new Set());
      setImageGeneration(currentImageGeneration.current);
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setBusy(false);
    }
  }

  function changeJudgment(id: string, key: ReviewCriterion, change: Partial<ReviewJudgment>) {
    setReview((current) =>
      current
        ? {
            ...current,
            samples: current.samples.map((sample) =>
              sample.id === id
                ? {
                    ...sample,
                    criteria: {
                      ...sample.criteria,
                      [key]: { ...sample.criteria[key], ...change },
                    },
                  }
                : sample,
            ),
          }
        : null,
    );
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (
      busy ||
      outcome ||
      !bundle ||
      !review ||
      !bundle.canSubmit ||
      !worldReviewReady(review) ||
      loadedImages.size !== bundle.samples.length
    )
      return;
    setBusy(true);
    setError(null);
    try {
      setOutcome(await saveWorldReview(runId, bundle, review));
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="world-review" aria-labelledby="world-review-title">
      <div className="section-head">
        <h2 className="section-title" id="world-review-title">
          Static world review
        </h2>
        <button type="button" className="button" onClick={load} disabled={busy}>
          {busy ? 'Working…' : bundle ? 'Reload review images' : 'Load review images'}
        </button>
      </div>
      <p className="field__hint">
        Compare every retained source and render pair. Record layout, coverage and appearance
        separately. Saving this review does not regenerate the scene or rerun paid stages.
      </p>
      {error ? (
        <p className="notice notice--warn" role="alert">
          {error}
        </p>
      ) : null}
      {outcome ? (
        <div className="notice" role="status">
          <p>
            Static world review: {outcome.reviewStatus}. Run status: {outcome.runStatus}.
          </p>
          {outcome.completionReason ? <p>{outcome.completionReason}</p> : null}
          {outcome.errors.length ? (
            <ul>
              {outcome.errors.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}
      {bundle && review ? (
        <form onSubmit={submit}>
          {bundle.holdReason ? <p className="notice notice--warn">{bundle.holdReason}</p> : null}
          <p className="field__hint">
            Reloading clears unsaved judgments. A missing or uncertain judgment keeps acceptance
            blocked; a critical failure cannot be averaged away.
          </p>
          <fieldset disabled={busy || !bundle.canSubmit || !!outcome}>
            <div className="field">
              <label htmlFor="world-reviewer">Reviewer name</label>
              <input
                id="world-reviewer"
                type="text"
                autoComplete="name"
                required
                value={review.reviewer}
                onChange={(event) => setReview({ ...review, reviewer: event.target.value })}
              />
            </div>
            {bundle.samples.map((sample, index) => (
              <article className="world-review__sample" key={sample.id}>
                <h3>
                  Frame {sample.sourceFrame} · {sample.timeSeconds.toFixed(2)}s
                  {sample.critical ? ' · Critical checkpoint' : ''}
                </h3>
                <a href={sample.pairUrl} target="_blank" rel="noreferrer">
                  <Image
                    key={`${imageGeneration}:${sample.id}`}
                    src={sample.pairUrl}
                    alt={`Source and rendered room at ${sample.timeSeconds.toFixed(2)} seconds`}
                    width={1600}
                    height={540}
                    unoptimized
                    className="world-review__pair"
                    onLoad={() => {
                      if (currentImageGeneration.current === imageGeneration)
                        setLoadedImages((current) => new Set([...current, sample.id]));
                    }}
                    onError={() => {
                      if (currentImageGeneration.current !== imageGeneration) return;
                      setLoadedImages((current) => {
                        const next = new Set(current);
                        next.delete(sample.id);
                        return next;
                      });
                      setError(
                        'A comparison image could not load. Reload the review before submitting.',
                      );
                    }}
                  />
                </a>
                <div className="world-review__criteria">
                  {reviewCriteria.map((key) => {
                    const value = review.samples[index].criteria[key];
                    const id = `review-${index}-${key}`;
                    return (
                      <div className="world-review__criterion" key={key}>
                        <div className="field">
                          <label htmlFor={id}>{criteriaLabels[key]}</label>
                          <select
                            id={id}
                            value={value.status}
                            onChange={(event) =>
                              changeJudgment(sample.id, key, {
                                status: event.target.value as ReviewJudgment['status'],
                              })
                            }
                          >
                            <option value="unknown">Uncertain / not reviewed</option>
                            <option value="pass">Pass</option>
                            <option value="fail">Fail</option>
                          </select>
                        </div>
                        <div className="field">
                          <label htmlFor={`${id}-reason`}>
                            Reason — {criteriaLabels[key].toLowerCase()}
                          </label>
                          <textarea
                            id={`${id}-reason`}
                            required
                            rows={2}
                            value={value.reason}
                            onChange={(event) =>
                              changeJudgment(sample.id, key, { reason: event.target.value })
                            }
                          />
                        </div>
                        <label className="field__toggle">
                          <input
                            type="checkbox"
                            checked={value.criticalFailure}
                            onChange={(event) =>
                              changeJudgment(sample.id, key, {
                                criticalFailure: event.target.checked,
                              })
                            }
                          />
                          Critical {criteriaLabels[key].toLowerCase()} failure
                        </label>
                      </div>
                    );
                  })}
                </div>
              </article>
            ))}
            <div className="world-review__acknowledgements">
              {(Object.keys(acknowledgements) as (keyof typeof acknowledgements)[]).map((key) => (
                <label className="field__toggle" key={key}>
                  <input
                    type="checkbox"
                    required
                    checked={review.acknowledgements[key]}
                    onChange={(event) =>
                      setReview({
                        ...review,
                        acknowledgements: {
                          ...review.acknowledgements,
                          [key]: event.target.checked,
                        },
                      })
                    }
                  />
                  {acknowledgements[key]}
                </label>
              ))}
            </div>
            <button
              className="button"
              type="submit"
              disabled={!worldReviewReady(review) || loadedImages.size !== bundle.samples.length}
            >
              Save static world review
            </button>
          </fieldset>
        </form>
      ) : null}
    </section>
  );
}
