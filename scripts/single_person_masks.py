#!/usr/bin/env python3
"""Source-indexed, undilated single-person observations; unavailable never means background.

Extraction uses only an already cached Mask R-CNN checkpoint. No tracker, guessed identity,
union, crop seed, dilation, or model download is involved. The archive is deliberately distinct
from legacy tight/clean archives, whose zero masks do not encode measurement availability.
"""

import argparse
import hashlib
import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "wander.single-person-masks/1"
MODEL = "torchvision.maskrcnn_resnet50_fpn_v2.COCO_V1"
CHECKPOINT_SHA256 = "73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e"
SCORE = 0.7
PROBABILITY = 0.5
SELECTION = "exactly-one-confident-person"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def valid_sha(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def source_contract(sequence, cameras):
    indices = sequence.get("sourceIndices")
    if (
        not isinstance(indices, list)
        or not indices
        or any(type(i) is not int or i < 0 for i in indices)
        or len(set(indices)) != len(indices)
        or len(sequence.get("frames", indices)) != len(indices)
    ):
        raise ValueError(
            "single-person sequence needs unique nonnegative sourceIndices matching frames"
        )
    source_sha = sequence.get("sourceSha256")
    if not valid_sha(source_sha):
        raise ValueError("single-person sequence needs sourceSha256 binding")
    by_source = {}
    for camera in cameras:
        index = camera.get("sourceIndex")
        if type(index) is not int or index < 0 or index in by_source:
            raise ValueError("camera sourceIndices must be unique nonnegative integers")
        by_source[index] = camera
    dimensions = None
    for index in indices:
        if index not in by_source:
            raise ValueError(f"no camera for single-person sourceIndex {index}")
        size = by_source[index].get("source_image_size")
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(type(v) is not int or v < 1 for v in size)
        ):
            raise ValueError(f"invalid source image grid for camera sourceIndex {index}")
        if dimensions is not None and size != dimensions:
            raise ValueError("single-person cameras have inconsistent source image grids")
        dimensions = size
    return indices, source_sha, dimensions


def load_archive(path, sequence, cameras, *, expected_identity=None):
    """Return only known source masks plus provenance; reject unrelated/clean/stale inputs."""
    indices, source_sha, size = source_contract(sequence, cameras)
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if not isinstance(metadata, dict) or not isinstance(metadata.get("identity"), dict):
                raise ValueError("single-person metadata and identity must be objects")
            identity = metadata["identity"]
            shape = archive["shape"]
            stored_indices = archive["sourceIndices"]
            packed = archive["packedMasks"]
            if archive["status"].shape != (len(indices),):
                raise ValueError("invalid single-person observation status shape")
            statuses = archive["status"].tolist()
            expected_shape = [len(indices), size[1], size[0]]
            if metadata.get("schema") != SCHEMA:
                raise ValueError("expected explicit single-person tight-mask schema")
            if identity.get("sourceSha256") != source_sha:
                raise ValueError("single-person mask source hash does not match sequence")
            if stored_indices.dtype.kind not in "iu" or stored_indices.tolist() != indices:
                raise ValueError("single-person mask source indices do not exactly match sequence")
            if shape.dtype.kind not in "iu" or shape.tolist() != expected_shape:
                raise ValueError("single-person mask grid does not match source cameras")
            if (
                identity.get("sourceIndices") != indices
                or identity.get("sourceImageSize") != size
                or identity.get("model") != MODEL
                or identity.get("selection") != SELECTION
                or identity.get("dilationPixels") != 0
                or identity.get("scoreThreshold") != SCORE
                or identity.get("maskThreshold") != PROBABILITY
                or identity.get("checkpointSha256") != CHECKPOINT_SHA256
                or not valid_sha(identity.get("extractorSha256"))
                or not valid_sha(identity.get("decoderSha256"))
            ):
                raise ValueError(
                    "invalid single-person model/selection/grid provenance; masks must be undilated"
                )
            if expected_identity is not None and identity != expected_identity:
                raise ValueError("single-person mask cache identity changed")
            if packed.dtype != np.uint8 or packed.shape != (
                len(indices),
                size[1],
                (size[0] + 7) // 8,
            ):
                raise ValueError("invalid packed single-person mask shape")
            if len(statuses) != len(indices) or any(
                s not in ("known", "missing", "ambiguous", "empty") for s in statuses
            ):
                raise ValueError("invalid single-person observation status")
            masks = np.unpackbits(packed, axis=-1)[..., : size[0]].astype(bool)
            records = metadata["frames"]
            if (
                not isinstance(records, list)
                or len(records) != len(indices)
                or any(not isinstance(record, dict) for record in records)
            ):
                raise ValueError("missing single-person observation provenance")
            known = {}
            for i, (index, status, record) in enumerate(zip(indices, statuses, records)):
                count = record.get("confidentPersonCount")
                scores = np.asarray(record.get("personScores"), dtype=float)
                if (
                    scores.ndim != 1
                    or not np.isfinite(scores).all()
                    or np.any((scores < 0) | (scores > 1))
                    or int(np.sum(scores > SCORE)) != count
                ):
                    raise ValueError(
                        f"invalid person score/count provenance at sourceIndex {index}"
                    )
                if (
                    record.get("sourceIndex") != index
                    or record.get("status") != status
                    or type(count) is not int
                    or count < 0
                    or not valid_sha(record.get("rgbSha256"))
                    or (status == "known" and (count != 1 or not masks[i].any()))
                    or (status == "empty" and count != 1)
                    or (status == "missing" and count != 0)
                    or (status == "ambiguous" and count < 2)
                    or (status != "known" and masks[i].any())
                ):
                    raise ValueError(f"invalid single-person observation at sourceIndex {index}")
                if status == "known":
                    known[index] = masks[i]
            return known, metadata
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        raise ValueError(f"invalid single-person masks {path}: {error}") from error


def select_instance(prediction):
    """Input NumPy arrays; preserve ambiguity instead of selecting or merging identities."""
    labels, scores = np.asarray(prediction["labels"]), np.asarray(prediction["scores"])
    masks = np.asarray(prediction["masks"])
    if (
        labels.ndim != 1
        or scores.shape != labels.shape
        or masks.ndim != 4
        or masks.shape[:2] != (len(labels), 1)
    ):
        raise ValueError("invalid instance prediction shape")
    if not np.isfinite(scores).all() or not np.isfinite(masks).all():
        raise ValueError("nonfinite instance prediction")
    keep = np.flatnonzero((labels == 1) & (scores > SCORE))
    count = len(keep)
    record = dict(confidentPersonCount=count, personScores=scores[labels == 1].tolist())
    selected = np.zeros(masks.shape[2:], bool)
    if count != 1:
        return selected, dict(record, status="missing" if count == 0 else "ambiguous")
    selected = masks[keep[0], 0] > PROBABILITY
    return selected, dict(record, status="known" if selected.any() else "empty")


def cached_checkpoint():
    import torch
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights

    path = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / Path(MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.url).name
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"cached Mask R-CNN checkpoint required (downloads disabled): {path}"
        )
    if sha256(path) != CHECKPOINT_SHA256:
        raise ValueError(f"cached Mask R-CNN checkpoint hash does not match COCO_V1: {path}")
    return path


def extraction_identity(source_sha, indices, size, checkpoint):
    import torch
    import torchvision

    return dict(
        sourceSha256=source_sha,
        sourceIndices=indices,
        sourceImageSize=size,
        model=MODEL,
        checkpointSha256=sha256(checkpoint),
        scoreThreshold=SCORE,
        maskThreshold=PROBABILITY,
        selection=SELECTION,
        dilationPixels=0,
        extractorSha256=sha256(__file__),
        decoderSha256=sha256(ROOT / "worker/wander_worker/source_timing.py"),
        torchVersion=torch.__version__,
        torchvisionVersion=torchvision.__version__,
        device="mps" if torch.backends.mps.is_available() else "cpu",
        ffmpegVersion=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
    )


def cached_predictor(checkpoint, device):
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("cached Mask R-CNN checkpoint hash does not match COCO_V1")
    # Both weights arguments are None: even a missing backbone cannot trigger a download.
    model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval().to(device)

    def predict(rgb):
        with torch.inference_mode():
            tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255).to(device)
            return {k: v.cpu().numpy() for k, v in model([tensor])[0].items()}

    return predict


def write_archive(path, packed, metadata):
    """Atomic replacement, with distinct unavailable statuses and explicit source ordinals."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = metadata["identity"]
    width, height = identity["sourceImageSize"]
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            np.savez_compressed(
                temporary,
                packedMasks=np.stack(packed),
                shape=np.array([len(packed), height, width]),
                sourceIndices=np.array(identity["sourceIndices"]),
                status=np.array([record["status"] for record in metadata["frames"]]),
                metadata=np.array(json.dumps(metadata)),
            )
            temporary.flush()
            os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)


def prepare(source, sequence_path, cameras_path, out):
    sequence = json.loads(Path(sequence_path).read_text())
    camera_doc = json.loads(Path(cameras_path).read_text())
    cameras = camera_doc["cameras"]
    indices, source_sha, size = source_contract(sequence, cameras)
    if sha256(source) != source_sha or camera_doc.get("sourceSha256", source_sha) != source_sha:
        raise ValueError("source video/cameras hash does not match packaged sequence")
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(source),
            ],
            text=True,
        )
    )
    stream = probe["streams"][0]
    if [stream["width"], stream["height"]] != size:
        raise ValueError("source video grid does not match packaged cameras")
    checkpoint = cached_checkpoint()
    identity = extraction_identity(source_sha, indices, size, checkpoint)
    if Path(out).exists():
        try:
            known, _ = load_archive(out, sequence, cameras, expected_identity=identity)
            print(
                f"single-person masks: exact cache match ({len(known)}/{len(indices)} known)",
                flush=True,
            )
            return
        except ValueError as error:
            print(f"single-person masks: rebuilding stale local archive: {error}", flush=True)
    sys.path.insert(0, str(ROOT))
    from worker.wander_worker.source_timing import select_source_frames

    rgb, provenance = select_source_frames(
        source, source_indices=sorted(indices), width=size[0], height=size[1]
    )
    if provenance["sourceSha256"] != source_sha or [
        f["sourceIndex"] for f in provenance["frames"]
    ] != sorted(indices):
        raise ValueError("decoded frames do not match requested source hash/indices")
    frames = np.frombuffer(rgb, np.uint8).reshape(len(indices), size[1], size[0], 3)
    by_source = {index: frame for index, frame in zip(sorted(indices), frames)}
    predict = cached_predictor(checkpoint, identity["device"])
    packed, records = [], []
    for index in indices:
        frame = by_source[index]
        mask, record = select_instance(predict(frame))
        if mask.shape != (size[1], size[0]):
            raise ValueError(f"prediction grid mismatch at sourceIndex {index}")
        packed.append(np.packbits(mask, axis=-1))
        records.append(
            dict(record, sourceIndex=index, rgbSha256=hashlib.sha256(frame.tobytes()).hexdigest())
        )
        print(f"single-person sourceIndex {index}: {record['status']}", flush=True)
    if (
        sha256(source) != source_sha
        or extraction_identity(source_sha, indices, size, checkpoint) != identity
    ):
        raise ValueError("source/model/extractor changed during mask extraction")
    metadata = dict(schema=SCHEMA, identity=identity, frames=records, decoder=provenance)
    write_archive(out, packed, metadata)
    load_archive(out, sequence, cameras, expected_identity=identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.sequence, args.cameras, args.out)


if __name__ == "__main__":
    main()
