#!/usr/bin/env python3
"""Compare an official LHM checkpoint with identical saved source pose and crops."""

import argparse, hashlib, json, os, shutil, sys, time
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    for name in ("prepared", "out", "model", "model-id", "model-revision"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--repo", default="/opt/lhm")
    a = ap.parse_args()
    prepared = Path(a.prepared)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(a.repo)
    sys.path.insert(0, a.repo)
    started = time.time()
    import torch
    from accelerate import Accelerator
    from lhm_person import preprocess, pose_parameters
    from LHM.models import model_dict
    from LHM.utils.hf_hub import wrap_model_hub

    torch._dynamo.config.disable = True
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    Accelerator()
    pose = torch.load(prepared / "source-pose.pt", map_location="cpu", weights_only=False)
    pose_record = json.loads((prepared / "source-pose.json").read_text())
    mask = np.asarray(Image.open(prepared / "mask.png").convert("L"))
    image = preprocess(Path(a.repo), prepared / "source.png", mask)
    head = np.asarray(Image.open(prepared / "head-input.png").convert("RGB"))
    head_image = torch.from_numpy(head / 255.0).float().permute(2, 0, 1)[None, None]
    Image.fromarray((image[0].permute(1, 2, 0).numpy() * 255).astype("uint8")).save(
        out / "model-input.png"
    )
    for name in ("source-pose.pt", "source-pose.json", "head-input.png"):
        shutil.copyfile(prepared / name, out / name)
    print("Loading pinned model with exact saved source pose:", a.model_id, flush=True)
    model = wrap_model_hub(model_dict["human_lrm_sapdino_bh_sd3_5"]).from_pretrained(a.model)
    model.eval()
    model.cuda()
    params = pose_parameters(pose)
    with torch.inference_mode():
        attrs, query, neutral = model.infer_single_view(
            image[None].cuda(), head_image.cuda(), None, None, None, None, None, smplx_params=params
        )
        params["transform_mat_neutral_pose"] = neutral
        gaussians = model.animation_infer_gs(attrs, query, params)
        gaussians.save_ply(str(out / "person-posed.ply"))
        torch.save(
            dict(attrs=attrs, query=query, neutral=neutral, params=params),
            out / "canonical-state.pt",
        )
    torch.cuda.synchronize()
    xyz = gaussians.xyz.detach().float().cpu().numpy()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    result = dict(
        model=a.model_id,
        modelRevision=a.model_revision,
        codeRevision="4f88aaeb3629249fbbddb4d0784a06962d9e1338",
        seconds=time.time() - started,
        peakVRAMGB=torch.cuda.max_memory_allocated() / 1e9,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        points=len(xyz),
        bounds=np.percentile(xyz, [2, 50, 98], axis=0).tolist(),
        sourceIntrinsics=pose_record["sourceIntrinsics"],
        prepared=json.loads((prepared / "prepared.json").read_text()),
        sourceImageSha256=sha(prepared / "source.png"),
        maskSha256=sha(prepared / "mask.png"),
        sourcePoseSha256=sha(prepared / "source-pose.pt"),
        headInputSha256=sha(prepared / "head-input.png"),
        modelInputPreviewSha256=sha(out / "model-input.png"),
        modelInputTensorSha256=hashlib.sha256(image.numpy().tobytes()).hexdigest(),
        headDetected=bool(np.any(head)),
        poseMethod="Exact saved MultiHMR source-pose tensors reused; no redetection. Same saved head crop and deterministic source/mask preprocessing.",
        nativeCoordinates="OpenCV source camera: x right, y down, z forward.",
        coverage="Learned body prior; hidden identity remains unverified inference.",
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
