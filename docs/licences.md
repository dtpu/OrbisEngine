# Licences

Our code is Apache-2.0. The models and libraries below keep their own terms, and several are
research-only, so worlds from the full pipeline can't be used commercially yet.

"Verify" means we haven't confirmed the terms. Check upstream before relying on any row.

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

Tears of Steel: (CC) Blender Foundation, CC BY 3.0. Phone clips are our own. Neither phone clips
nor movie excerpts are redistributed. Get consent from anyone rebuilt as an avatar.

## Possible replacements

- SegFormer: Mask R-CNN or SAM
- Pi3X: MapAnything or Depth Anything 3 (Apache checkpoints)
- SMPL-X: SAM 3D Body + MHR
