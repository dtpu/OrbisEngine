# Joker stairs and Rocky scene preparation

Branch: `austin/joker-rocky-scenes`. The two supplied movie excerpts contain edits; each
continuous shot is a separate reconstruction candidate. Joker is restricted to the stairs
dance. The original recordings remain unchanged in private, ignored storage.

Local preparation produced 15 Joker stairs shots covering source frames `[491,1806)`
(20.478791667–75.325250000 seconds), and 19 Rocky shots covering 161.035875 seconds.
Rocky's four subsecond inserts and 30.681375-second promotional tail are catalogued as
excluded. These counts describe prepared inputs, not completed worlds. Source frame ordinals,
integer PTS, hashes and decoded audio sample ranges are retained per shot under
`.context/evidence/film-clips/` in the author checkout. Detector boundaries were visually
reviewed; missed edits and false positives were corrected before splitting.

## Authorized first candidates

The user authorized $10 total additional compute/API spending and 3,200 existing Marble
credits, with no top-ups. At 1,600 credits per generation this permits two worlds, not every
prepared shot. Start one candidate from each film; preserve the others for later selection.

| Candidate | Original source SHA-256 | Source interval | Frames |
| --- | --- | --- | ---: |
| Joker stairs shot 005 | `d2f50a98c948074e5c0c8627d2aaf7d331a4d432b215ecf108ef400285b14108` | 33.575208333–37.078708333 s, `[805,889)` | 84 |
| Rocky museum stairs | `eef2cea148c8bfac5e8f645d61d292fd45293a1d9d0d65df6a5312feca60cad4` | 120.495375–144.894750 s, `[2889,3474)` | 585 |

The Joker derivative retains exactly matching decoded pixels and PCM audio. Rocky's H.264/AAC
derivative retains every selected video frame and shifted PTS, with a companion WAV that
matches original decoded PCM samples. The supplied files are actually 640×348 and 960×540;
their filenames do not establish higher resolution or HDR. Neither candidate supports
recovering fine facial details or unobserved surfaces exactly.

## Execution bounds and review

Use the existing absolute author ledger `/Users/austinjian/htn2026/.context/pipeline-attempts.json`
and the original canonical hashes across derivatives and aliases. No new allowance is created
by a shot name. Recover uncertain outputs before considering another submission.

The five GPU functions now cap CPU and memory at their existing requests: four physical
cores, 32 GiB for cleaning/camera reconstruction and 64 GiB for person inference. Existing
provider timeouts and zero retries are unchanged. With the published base resource rates,
the planned single-person Joker graph plus one tracked Rocky person reserves approximately
$8.78 at their complete execution-timeout ceilings, leaving $1.22 for startup/build overhead.
This is a reservation estimate, not a verified bill or a universal cross-provider budget tool.
Paid setup, extra people, paid descriptions and retries are not part of that reservation.
Run bounded stage subsets and inspect costs before advancing. Cached model files must be
checked before launching; preserve all other sessions' jobs.

Inspect people masks against stair treads and rails before either video-first world submission.
Keep the static-world manual review gate intact. Test actor continuity, source timing,
displaced views and walking in the real viewer before describing a candidate as delivered.
Stair support must be measured; a flat floor is insufficient. Prepared clips and completed
subprocesses do not establish visual quality. Formal acceptance, publication and generation
status will be recorded separately as evidence becomes available.

## Current review state

The selected Joker shot is 3.5035 seconds and Rocky museum stairs is 24.399375 seconds.
Both browser inputs preserve the selected source timeline and audio. A local review page at
`http://127.0.0.1:5401/reviews/joker-rocky-v1/index.html` exposes those inputs and an explicitly
failed/experimental Joker preview. It is not a completed two-world delivery.

Joker recovered 41 of 43 LHM animation samples; samples 17 and 24 are missing. That
candidate interpolates those gaps, but its generated appearance failed source comparison.
The current rough preview instead displays 43 independent source-colored depth samples,
with incomplete silhouettes and missing backs. The observed stair surface has holes and
residual actor fragments. Neither is an accepted scene.

The user authorized one additional Joker cleanup execution after the ordinary three-attempt
limit, retaining the $10 cap. Its approval is hash-bound to the original source and cleaning
stage in the existing attempt ledger; it grants no fifth attempt or exception for Rocky.
All 42 final Joker frames were reviewed. Recognizable person fragments are removed, but
stair lines are offset across fill patches, and late frames contain disconnected lamp
sections and damaged rails/building edges. The fourth cleanup therefore remains failed.

Rocky's third camera solve uses 16 shared anchors and batches of eight local samples. All
293 camera samples retain exact source indices matching the pose tracker, and the existing
camera-jump guard passes. Its worst measured step is about 1.88 scene depths per second,
19.54 times the median; thresholds are unchanged. This is not geometric or walking acceptance.
CPU frame alignment now consumes the saved 16 anchor IDs instead of assuming eight. It
checks anchor-array counts and source-camera correspondence before using them.

Rocky's tracked person covers samples 31–292 (262 of 293). The first 11 samples show empty
stairs; the partially visible entrance in samples 11–30 lacks supported poses. Camera and
pose samples end at decoded source index 583, while the separately bound cleaning grid ends
at 584. These grids are not interchangeable merely because both contain 293 samples.

The third and final Rocky cleanup improves most body removal, including the previously
reported late floating arm, but all-frame review found detached shoes/hands at samples
101, 141–142 and 173–174. The distant monument also loses detail or disappears around
224–252. These failures block world generation. No Marble generation has been submitted
for either candidate; all 3,200 authorized credits remain unspent. Retain failed outputs and
their evidence without promoting them or silently buying further cleanup attempts.

The camera guard now measures static depth from the recorded supported anchor in the same
normalized units as exported camera positions. It retains the existing rejection thresholds.
The corrected calculation still rejects the second Rocky solve: camera 127 to 128 moves about 3.34 scene
depths per second, with a step about 59.76 times the median. A successful GPU exit does not
promote that camera path.
