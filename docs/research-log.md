# Research log

What we tried on the way to the current pipeline, and how each one went. Numbers come from the
experiment records kept on `archived-main`. Most were measured on one clip, an 11-second forward
walk down a corridor, so read them as "what happened to us", not as benchmarks of these projects.
[Known limits](known-limits.md) is the short list of dead ends to check before trying something new.

## Rebuilding the room from the footage

| Approach                                                               | Used for                          | Result                                                                                    |
| ---------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) | Depth and cameras from frames     | Worked. 20 frames in about 12 s on an M3 Pro, 1.7 GB. Our first backbone.                 |
| [VGGT](https://github.com/facebookresearch/vggt)                       | Same job                          | Worked but too heavy: 164 s and 15 GB for the same 20 frames.                             |
| [Brush](https://github.com/ArthurBrussee/brush)                        | Training Gaussian splats on a Mac | Worked. A real splat of the corridor trained on Metal in minutes.                         |
| [gsplat](https://github.com/nerfstudio-project/gsplat)                 | CUDA splat training               | Worked on GPU only. Later became our fine-tuning trainer.                                 |
| Fixed-focal SfM                                                        | Camera calibration                | Unconstrained SfM registered 3 of 550 frames. Fixing the focal length registered all 550. |
| TSDF fusion of masked depth                                            | Filling low-texture holes         | Failed. Roof and stairwell holes stayed.                                                  |
| Extending the fitted floor                                             | Filling the surroundings          | Failed. Turn around and you see a flat rectangle ending.                                  |
| Hand-built architecture mesh                                           | An explicit inferred room         | Partial. Looked painted: every surface was textured from a few crops of frame 0.          |

The wall we hit: a camera walking forward only ever records about 41° off its own path. We measured
recorded coverage at 60°, 90° and 180° and it was 0%. Three different depth and SfM sources agreed.
Seeing a wall side-on needs you to have stood well back from it, and an 11-second walk never does.
No reconstruction method recovers pixels that were never filmed.

## Generating the missing views

| Approach                                                                       | Result                                                                                                                                                                  |
| ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Stable Virtual Camera](https://github.com/Stability-AI/stable-virtual-camera) | Produced an outdoor courtyard and melted the subject's arms. We later realised we ran it on a single frame, its worst case, so this is not a fair verdict on the model. |
| [GeoNVS](https://github.com/MinJunKang/GeoNVS)                                 | Collapsed to a grey smear at exactly the angle where recorded coverage ends. Also rewrote a real shop sign into invented letters.                                       |
| FLUX panorama outpainting with a 360 LoRA                                      | Partial. 4 of 16 held-out seeds passed. The trick that made it work at all was drawing the known structure as luminance inside the masked region.                       |
| [GEN3C](https://github.com/nv-tlabs/GEN3C)                                     | Lost to the FLUX fill. Held-out quality dropped and it left tens of thousands of stray Gaussians in the walkable space.                                                 |
| [Lyra](https://github.com/nv-tlabs/lyra)                                       | Ran end to end but gave three static snapshots at 6 to 7 fps in the viewer, with sideways streaking. Dropped for time, not a finding that Lyra fails in general.        |
| Observation-count opacity falloff ([GA-GS](https://arxiv.org/abs/2604.04331))  | Did nothing on our data. Almost every splat had plenty of observations, so there was nothing to fade.                                                                   |

## Generating the whole room

| Approach                                                          | Result                                                                                                                                                                                                                     |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [World Labs Marble](https://docs.worldlabs.ai/api)                | Adopted. Generates a full splat world in 6 to 7 minutes. On its own it invents materials and gets scale wrong by around 25%.                                                                                               |
| Fine-tuning the Marble world against the posed frames with gsplat | The biggest single jump. Held-out PSNR on the corridor went from 9.96 to 33.09 dB and the invented materials became the recorded ones. Buys fidelity on the camera path plus a short step either side, not free viewpoint. |
| Difix3D+ views as extra supervision off the path                  | Helped once the weights actually loaded (the first run was RGB noise). Reduced the melting a couple of steps off the path.                                                                                                 |
| LaMa vs SDXL for the strip the walker always covered              | LaMa shipped. SDXL fills looked better one at a time, but 20 fills that disagree average back into haze.                                                                                                                   |
| ReCamMaster camera-controlled sweeps                              | Did not beat a smooth LaMa fill.                                                                                                                                                                                           |
| Synthesising one floor sheet directly in 3D                       | Shipped. Copying real floor splats across gave debris; generating the geometry and carrying only colour worked.                                                                                                            |

Marble notes: video input is caption, then panorama, then world. It is not reconstruction. One
clean, person-free frame in image mode is the most faithful input. Multiple images help a clip that
orbits and hurt a straight dolly. Always fit the scale against real depth.

## People

| Approach                                                                              | Result                                                                                                             |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| [V-DPM](https://github.com/eldar/vdpm)                                                | Rejected. Many more points, almost no extra depth: the person stayed paper-thin.                                   |
| [PIFuHD](https://github.com/facebookresearch/pifuhd)                                  | Failed. A closed body, but malformed neck, limbs and shoes.                                                        |
| [Hunyuan3D](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) texture pass on that body | Rejected. Colour improved, anatomy did not.                                                                        |
| [LHM](https://huggingface.co/3DAIGC/LHM-500M-HF)                                      | Worked. The first avatar that held up from behind and from the side. 40,000 Gaussians per person.                  |
| [LHM 1B](https://huggingface.co/3DAIGC/LHM-1B-HF)                                     | Rejected. More invented facial detail is not more likeness.                                                        |
| Refitting body pose against the footage                                               | Made it worse on every clip. One camera cannot say how far away a hand is, so the optimiser bends the body to fit. |
| Refitting the head's appearance per video                                             | Failed. The loss went down while the head shrank out of frame.                                                     |
| LHM++ with 8 views and GVHMR poses                                                    | Partial. Better face, but the feet sank steadily through the floor over ten seconds.                               |
| [Pi3X](https://github.com/yyfz/Pi3)                                                   | Too thin to be the person, but it became our camera and depth solver.                                              |

Two rules that held: a moving person cannot live in the static world, because the solver smears
them into it, so remove them first and add them back as their own layer. And faces under about
64 px tall in the source are unreadable, so shot choice matters more than any later fix.
