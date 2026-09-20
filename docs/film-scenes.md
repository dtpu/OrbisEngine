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

Joker recovered 41 of 43 independent animation samples; samples 17 and 24 are missing.
Playback interpolates those gaps. The real-viewer inspection also rejected the character's
appearance and observed holes/residual actor fragments in the untrained stair surface.
The cleaned-video revision retains floating head and limb fragments, so Marble submission
has not been approved. Rocky's revised camera export is finite but contains a real batch
boundary teleport; its animation and world remain unfinished. No Marble generation has
been submitted for either candidate.

The camera guard now measures static depth from the recorded supported anchor in the same
normalized units as exported camera positions. It retains the existing rejection thresholds.
The corrected calculation still rejects Rocky: camera 127 to 128 moves about 3.34 scene
depths per second, with a step about 59.76 times the median. A successful GPU exit does not
promote that camera path.
