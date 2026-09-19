# Gym and kitchen delivery

Branch: `austin/gym-kitchen`, based on current consolidated `origin/main` (`7c24a2d`).
The accepted gym is recovered unchanged. Kitchen is a full-duration **partial reconstruction**:
animated people and original sound work, but its room registration, moving fixtures and objects
have not passed acceptance. No new paid generation, cleaning or inference was submitted.

## Open the scenes

[Delivery page](http://127.0.0.1:5399/reviews/gym-kitchen/index.html) contains both scenes and videos.
It includes the current compiled viewer, so the visibility correction travels with the media.
A teammate server must adopt the delivery snapshot; a running server pins its older snapshot.
Only restart a server you own. See [shared assets](shared-assets.md).

On Austin's current machine, the existing server belongs to another checkout and was preserved.
The [immediate local delivery page](http://127.0.0.1:5399/@fs/Users/austinjian/htn2026/public/reviews/gym-kitchen/index.html)
uses its already permitted filesystem serving to load this checkout's compiled viewer and verified
media without restarting that server. This machine-specific link is separate from private S3 access.

| Scene | Coverage | Preview |
| --- | --- | --- |
| [Accepted gym comparison](http://127.0.0.1:5399/reviews/gym-repair/index.html?quality=detail) | 213 original PLY samples, 18.0667 s; two 542-sample dumbbell tracks; three bench colliders | [Unchanged gym video](http://127.0.0.1:5399/reviews/gym-repair/gym-contact-wip.mp4), silent, 18.051 s |
| Kitchen, from delivery page | 609 pose samples across 0–50.73 s; one recorded cook represented by two appearance fragments | [Kitchen video](http://127.0.0.1:5399/reviews/gym-kitchen/kitchen-preview.mp4), 1280×720, 609 frames at 12 fps, 50.730 s, original AAC |

The final kitchen pose is at 50.663333 s and is held for the remaining 0.066667 s. This is full
playback coverage, not a claim of 60 independently reconstructed poses per second. The preview's
last video sample is shortened to the recording endpoint, rounded to its 1/12288 s timebase.

## Recovery and source timing

All 226 accepted gym dependencies matched the accepted snapshot's SHA-256 and size. The person,
dumbbells, placement, frame alignment and colliders were not repacked or repaired. See
[gym delivery](gym-delivery.md) for its acceptance and preserved defects.

The kitchen original is SHA-256
`15084804fb92ee6bcf37aa9433452aa534c4ae7f8d43dac3b6dbaf29389d8598`:
3,040 decoded HEVC frames, variable source timing, 50.73 seconds. A separate, later recovered
`kitchen-cooking` run supplies the delivered people and image-generated world; the older
`test1.mov` poses and failed multi-image-world transform are **not reused**.

The later run's browser derivative is SHA-256
`a18bc6e275be65705999bfc73df52e4baf4ef4de59093f52a1d5478dc6295669`:
3,041 CFR frames at 60000/1001 fps. Every decoded derivative frame was compared photometrically
with nearby original decoded frames. The resulting 3,041 correspondences are monotonic, with
maximum timestamp difference 0.009584 s. This is a measured image match, not byte identity or
proof of pose accuracy. Original native PTS are authoritative in the delivery manifests;
derivative indices and the mapping remain available as provenance.

People, cameras and placement use the same mapped timeline. One duplicate late detection at
sample 553 was suppressed because it overlaps the same cook's existing track. The early fragment
retains 559 samples; the late fragment retains 50. Their original geometry remains unchanged.
Both tracks use verified lossless float32 motion (all errors zero), with original PLYs retained.
A general viewer correction interprets visibility runs on shared source timestamps instead of
compact PLY indices, preventing an occlusion gap from delaying a later fragment.

Cleaning uses a different decode schedule from the pose workers. Reconstructing the newer
historical FFmpeg schedule exposes 609 incorrect guessed labels: the first resampled source
ordinal is 2, not 0. These old cleaned frames are not relabeled as camera-aligned inputs.
The older archived framehash pair independently proves 609 pixel matches and 305 incorrect labels.
New pipeline cleaning binds actual decoder selections and PTS, image hashes and source hashes;
multi-image selection/submission checks these bindings before spending. Existing operation
recovery remains available without a new submission. See [kitchen review](kitchen-review.md).

The recovered first-image clean is a separate source-frame-0 operation. Its saved camera RGB
matches that source view (resized RGB mean absolute difference 1.375/255; ray reprojection RMSE
0.245 pixels). Direct inspection confirms that inpainting also damaged visible sink/counter
geometry. Source-camera rays contain no depth. A source-coloured alternative was not promoted:
the required anchor download failed, and rays alone cannot determine a surface.

## Sound, objects and visual limits

The delivered source video reuses the recovered browser video encoding and remuxes the original
AAC stream. All **2,380 AAC packet payload hashes, timestamps and durations** match the original,
including priming metadata. The preview preserves the same packets. No words, timing, speaker
stems or other sound were synthesized. The browser source duration differs by 4 ms because of its
CFR frame grid; the scene and preview end at the original 50.730 s.

Kitchen limits remain material:

- The sink/counter was damaged by cleaning, and generated structural geometry and person/world
  registration remain unaccepted. A recognizable opening view does not validate displaced views.
- Refrigerator and dishwasher motion, cookware, the bag and other handled objects are absent.
  Recovered detector candidates were false floor/background tracklets; no validated supported
  3D object trajectory was recovered. The published object manifest explicitly contains none.
- The same cook changes inferred appearance near 46.58 s. Hidden body geometry and generated
  room appearance are inferred. The late pose/appearance is visibly less accurate.
- Opposing views reveal missing/smeared surfaces and can enter generated geometry. Walking works
  within an estimated room grid; it is not proof of source-accurate kitchen collision.

Gym retains broad/flattened body shape, imperfect feet/upper-back contact, inferred dumbbells,
and a late-frame mismatch where hands lower while dumbbells remain overhead. The accepted preview
is shorter than the scene by about 15 ms. These are preserved acceptance limitations.

## Validation and spending

Actual desktop Chrome tested both scenes at port 5399. Gym passed full-detail comparison controls,
213-frame playback/seeking and looping, both props, three colliders, four walking approaches,
stationary seat support and blocked furniture map selection. Seat eye height was 0.93 body-heights.
Kitchen passed all 609 samples, 52 seconds of playback including a loop with original audio enabled,
one visible cook at every captured preview time, handover/last-frame seeks and four walking inputs.
Front/profile/opposing views and source imagery were inspected. These are desktop checks, not
headset evidence or cold-network benchmarks.

`bun run build`, `bun run format:check`, source-timing tests, person-motion packaging/round-trip/
visibility tests and relevant collider/audio regressions passed. Tests do not certify visual quality.
Private reports, screenshots, source mappings and receipts are kept under ignored
`.context/evidence/gym-kitchen/`; the curated publication evidence uses the archive prefix
`evidence/gym-kitchen/`.

The recovered pipeline ledger retains all 31 prior claims and source aliases. Four historical
resource ledgers and both known kitchen Marble operation receipts are preserved. The known
operations are `23c32eb3-603a-4dff-92dd-035b0d10705e` and
`642f830b-3410-44a6-957f-21b412b64727`. Existing cleaning/world allowances have no remaining
submissions. Shared account activity prevents a reliable remaining-dollar balance; it was not
reset or topped up. Delivery inference/API spend is zero; S3 transfer/storage costs are unmeasured.
Other Modal jobs, caches and the existing Vite server were preserved.

Publication receipt and independently checked snapshot identity follow after upload verification.
