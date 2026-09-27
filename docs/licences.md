# Licences

Orbis's own code is Apache-2.0 (see `LICENSE`). That covers this repository only. The pipeline
downloads and runs third-party models and libraries under their own terms, and several of them
allow research use only. A world produced by the full pipeline is therefore fine for research and
demos, not for commercial use, until the non-commercial pieces are replaced.

Terms below are as published upstream when this was written; check the upstream licence before
relying on a row. "Verify" means we have not confirmed the terms.

## Pipeline components

| Component                                                  | Used for                                                | Where                                                                             | Terms                                                                                   |
| ---------------------------------------------------------- | ------------------------------------------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Pi3X weights (`yyfz233/Pi3X`)                              | Cameras and depth                                       | `worker/stages/dense_pi3x.py`                                                     | Non-commercial (CC BY-NC)                                                               |
| SegFormer-b0 (`nvidia/segformer-b0-finetuned-ade-512-512`) | People masks for room cleaning, every run               | `worker/wander_worker/masks.py`                                                   | Non-commercial (NVIDIA Source Code License)                                             |
| Mask R-CNN (torchvision)                                   | Person prep and tracking                                | `worker/stages/track_people.py`                                                   | BSD-3-Clause                                                                            |
| LaMa via simple-lama-inpainting                            | Filling where people stood                              | `worker/modal_clean_video.py`                                                     | Apache-2.0                                                                              |
| World Labs Marble                                          | Room generation                                         | `scripts/marble_world.py`                                                         | Paid API; World Labs terms govern outputs                                               |
| OpenAI API                                                 | World prompt, VLM judge, character voice                | `scripts/world_prompt.py`, `worker/stages/vlm_judge.py`, `server/bottle-agent.ts` | Paid API                                                                                |
| LHM (`3DAIGC/LHM-500M-HF`)                                 | Animated person avatars                                 | `worker/modal_lhm.py`                                                             | Code Apache-2.0; weights and bundled priors: verify (likely includes Sapiens, CC BY-NC) |
| SMPL-X body model                                          | Person pose and shape                                   | `worker/modal_lhm.py`, `worker/stages/lhm_animate.py`                             | Non-commercial (MPI licence)                                                            |
| Multi-HMR                                                  | Per-person pose                                         | `worker/stages/track_people.py`                                                   | Non-commercial (CC BY-NC-SA 4.0)                                                        |
| GVHMR outputs                                              | World-space body motion                                 | `scripts/package_multiperson.py`                                                  | Verify                                                                                  |
| GFPGAN, facexlib, BasicSR                                  | Face restoration for avatars                            | `worker/modal_lhm.py`                                                             | Apache-2.0 / MIT / Apache-2.0; verify weights                                           |
| diff-gaussian-rasterization, simple-knn                    | Gaussian rendering in LHM and image-to-3D               | `worker/modal_lhm.py`, `worker/modal_image_to_3d.py`                              | Non-commercial (Inria Gaussian-Splatting licence)                                       |
| nvdiffrast                                                 | Rasterization in image-to-3D                            | `worker/modal_image_to_3d.py`                                                     | Non-commercial (NVIDIA Source Code License)                                             |
| mip-splatting                                              | Image-to-3D                                             | `worker/modal_image_to_3d.py`                                                     | Non-commercial (Inria-derived); verify                                                  |
| TRELLIS                                                    | Object image-to-3D                                      | `worker/modal_image_to_3d.py`                                                     | MIT                                                                                     |
| Stable Diffusion XL base                                   | Object image-to-3D                                      | `worker/modal_image_to_3d.py`                                                     | CreativeML OpenRAIL++-M (use restrictions)                                              |
| PyTorch3D                                                  | Geometry utilities                                      | worker images                                                                     | BSD-3-Clause                                                                            |
| gsplat                                                     | Optional fine-tune against real frames                  | `scripts/finetune_*.py`                                                           | Apache-2.0                                                                              |
| VGGT, V-DPM                                                | Abandoned experiments still in `worker/modal_motion.py` | `worker/stages/vdpm_person.py`                                                    | VGGT non-commercial; V-DPM verify. Slated for removal                                   |

## Viewer

| Component                   | Terms      |
| --------------------------- | ---------- |
| three.js                    | MIT        |
| Spark (`@sparkjsdev/spark`) | MIT        |
| webxr-polyfill              | Apache-2.0 |
| OpenAI Agents SDK           | MIT        |

## Footage

Tears of Steel excerpts are (CC) Blender Foundation, CC BY 3.0. Phone clips are the team's own
footage and are not redistributed. Movie excerpts used in testing are not redistributed. Anyone
rebuilt as an avatar from a clip should have agreed to it.

## Commercial-safe replacements under consideration

Mask R-CNN or SAM for SegFormer, MapAnything or Depth Anything 3 (Apache checkpoints) for Pi3X,
and SAM 3D Body + MHR for SMPL-X.
