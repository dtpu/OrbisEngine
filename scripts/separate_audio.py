#!/usr/bin/env python3
"""Estimate a voice track and a residual bed from one mixed soundtrack, on CPU.

This is a *model estimate*, never a recorded stem. It extracts the original programme audio
from a source video at its native sample rate, runs Demucs htdemucs (two stems: vocals /
no-vocals) on CPU, and writes:

  original.wav  exact PCM of the chosen source audio stream, native rate and channels
  voice.wav     mono vocals estimate, for positional playback
  bed.wav       original minus the vocals estimate, native channel layout
  receipt.json  hashes, stream choice, model, durations, residuals, voice activity

`bed` is defined as `original - voice_estimate` so that voice + bed reconstruct the original
sample for sample: nothing recorded is dropped and nothing is invented. The independent
model round-trip residual (original - vocals - no_vocals) is measured and reported as well.

  uv run --locked --group audio scripts/separate_audio.py \
      --video /path/to/clip.MOV --out .context/evidence/new-clips/audio/clip

Optionally (Phase 2) it also splits `voice.wav` into per-speaker stems plus `segments.json`:

  uv run --locked --group audio scripts/separate_audio.py \
      --video /path/to/clip.MOV --out DIR --segment-speakers 2

Those speaker stems partition the same samples - every sample lands in exactly one stem, and
the written integers are checked to sum back to `voice.wav`. The grouping is weak acoustic
clustering, never an identity: `segments.json` leaves `attribution.personId` null for a judge
or an operator to fill in, and nothing here is ever marked reviewed.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import wave

import numpy as np

PROVENANCE = "separated-estimate:demucs-htdemucs"
NOTE = (
    "Model-separated estimates of the recorded sound, not recorded stems. "
    "No words, timing, or sample positions are altered; bed = original - voice."
)
SAMPLE_WIDTH = 2
FULL_SCALE = 32767.0


def fail(message):
    raise ValueError(message)


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        fail(f"{command[0]} failed: {result.stderr.strip()[-400:]}")
    return result.stdout


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_audio_streams(video):
    """List every audio stream with the fields the stream choice is justified by."""
    data = json.loads(
        run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index,codec_name,codec_tag_string,channels,channel_layout,sample_rate,duration",
                "-of",
                "json",
                str(video),
            ]
        )
    )
    streams = []
    for stream in data.get("streams", []):
        codec = stream.get("codec_name")
        streams.append(
            {
                "index": int(stream["index"]),
                "codecName": codec,
                "codecTag": stream.get("codec_tag_string"),
                "channels": int(stream.get("channels") or 0),
                "channelLayout": stream.get("channel_layout"),
                "sampleRate": int(stream.get("sample_rate") or 0),
                "durationSeconds": float(stream["duration"]) if stream.get("duration") else None,
                "decodable": bool(codec) and codec not in ("none", "unknown"),
            }
        )
    return streams


def choose_stream(streams):
    """Prefer the decodable 2-channel stream; phone clips also carry an undecodable 4-ch stream."""
    if not streams:
        fail("The source video has no audio stream")
    decodable = [s for s in streams if s["decodable"] and s["sampleRate"]]
    if not decodable:
        fail(
            "No decodable audio stream: "
            + ", ".join(f"#{s['index']} tag {s['codecTag']}" for s in streams)
        )
    stereo = [s for s in decodable if s["channels"] == 2]
    chosen = (stereo or decodable)[0]
    others = [s for s in streams if s["index"] != chosen["index"]]
    reason = (
        f"stream #{chosen['index']} is the "
        + ("2-channel " if chosen["channels"] == 2 else f"{chosen['channels']}-channel ")
        + f"{chosen['codecName']} programme mix at {chosen['sampleRate']} Hz"
    )
    if others:
        reason += "; rejected " + ", ".join(
            f"#{s['index']} ({s['codecTag']}, {s['channels']} ch, "
            + ("not decodable by this ffmpeg" if not s["decodable"] else "not stereo")
            + ")"
            for s in others
        )
    return chosen, reason


def extract_original(video, stream_index, destination):
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(video),
            "-map",
            f"0:{stream_index}",
            "-vn",
            "-sn",
            "-dn",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )


def read_wav(path):
    with wave.open(str(path), "rb") as handle:
        if handle.getcomptype() != "NONE" or handle.getsampwidth() != SAMPLE_WIDTH:
            fail(f"{path.name}: expected 16-bit PCM WAV")
        channels, rate, frames = handle.getnchannels(), handle.getframerate(), handle.getnframes()
        raw = handle.readframes(frames)
    samples = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).T.astype(np.float64)
    return samples / FULL_SCALE, rate


def write_wav(path, samples, rate):
    """samples: (channels, n) float in [-1, 1]. Returns the clipped-sample count."""
    scaled = np.rint(samples * FULL_SCALE)
    clipped = int(np.count_nonzero((scaled > FULL_SCALE) | (scaled < -FULL_SCALE - 1)))
    data = np.clip(scaled, -FULL_SCALE - 1, FULL_SCALE).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(samples.shape[0])
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(rate)
        handle.writeframes(data.T.reshape(-1).tobytes())
    return clipped


def rms(samples):
    return float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0


def dbfs(value):
    return float(round(20 * math.log10(value), 2)) if value > 0 else None


def resample(tensor, source_rate, target_rate):
    import julius

    if source_rate == target_rate:
        return tensor
    return julius.resample_frac(tensor, source_rate, target_rate)


def fit_length(tensor, length):
    """Rational resampling changes the length by a sample or two; never shift the start."""
    import torch

    if tensor.shape[-1] > length:
        return tensor[..., :length]
    if tensor.shape[-1] < length:
        pad = torch.zeros(*tensor.shape[:-1], length - tensor.shape[-1], dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=-1)
    return tensor


def separate(mix, rate, model_name, shifts, overlap, jobs):
    """Return (vocals, non_vocals) at the original rate, channels and sample count."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    if jobs:
        torch.set_num_threads(jobs)
    jobs = torch.get_num_threads()
    model = get_model(model_name)
    model.cpu().eval()
    version = getattr(model, "sig", None) or model_name
    length, channels = mix.shape[1], mix.shape[0]
    wav = torch.from_numpy(mix).float()
    if channels == 1:
        wav = wav.repeat(model.audio_channels, 1)
    elif channels > model.audio_channels:
        wav = wav[: model.audio_channels]
    wav = resample(wav, rate, model.samplerate)
    # Demucs' own normalisation (demucs.separate): centre and scale by the mono reference.
    reference = wav.mean(0)
    mean, std = reference.mean(), reference.std()
    normalised = (wav - mean) / (std if float(std) > 0 else 1.0)
    started = time.monotonic()
    with torch.no_grad():
        stems = apply_model(
            model,
            normalised[None],
            device="cpu",
            shifts=shifts,
            split=True,
            overlap=overlap,
            progress=False,
            num_workers=0,
        )[0]
    elapsed = time.monotonic() - started
    stems = stems * (std if float(std) > 0 else 1.0) + mean
    index = model.sources.index("vocals")
    vocals = stems[index]
    non_vocals = stems.sum(0) - vocals
    back = []
    for stem in (vocals, non_vocals):
        stem = fit_length(resample(stem, model.samplerate, rate), length)
        if channels == 1:
            stem = stem.mean(0, keepdim=True)
        elif channels > model.audio_channels:
            stem = torch.cat([stem, torch.zeros(channels - stem.shape[0], length)], 0)
        back.append(stem.double().numpy())
    return (
        back[0],
        back[1],
        {
            "name": model_name,
            "signature": version,
            "sources": list(model.sources),
            "modelSampleRate": int(model.samplerate),
            "modelChannels": int(model.audio_channels),
            "shifts": shifts,
            "overlap": overlap,
            "torchThreads": jobs,
            "applyModelSeconds": round(elapsed, 2),
        },
    )


def activity(voice, bed, rate, threshold_dbfs, window):
    """Per-second RMS of both signals, plus the fraction of short frames above a threshold."""
    hop = max(1, int(round(rate * window)))
    threshold = 10 ** (threshold_dbfs / 20)
    frames = [voice[i : i + hop] for i in range(0, voice.shape[0], hop)]
    levels = np.array([rms(frame) for frame in frames if frame.size == hop] or [0.0])
    per_second = max(1, int(round(rate)))

    def seconds(signal):
        chunks = [signal[i : i + per_second] for i in range(0, signal.shape[0], per_second)]
        return [round(rms(chunk), 6) for chunk in chunks if chunk.size]

    return {
        "thresholdDbFs": threshold_dbfs,
        "frameSeconds": window,
        "voiceActiveFraction": round(float(np.mean(levels > threshold)), 4),
        "voiceFrameRmsMedian": round(float(np.median(levels)), 6),
        "voiceFrameRmsPeak": round(float(levels.max()), 6),
        "voiceRmsPerSecond": seconds(voice),
        "bedRmsPerSecond": seconds(bed),
        "note": (
            "Energy only. It shows when the vocals estimate carries signal, "
            "not who speaks, whether words are intelligible, or whether the bed still leaks speech."
        ),
    }


SEGMENT_PROVENANCE = "segmented-by:energy-vad;clustered-by:mfcc-agglomerative"


def video_fps(video):
    """Frame rate of the first video stream, so segment times can be quoted as frame numbers."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "csv=p=0",
        str(video),
    ]
    try:
        text = run(command).strip()
        text = text.splitlines()[0].strip(", ") if text else ""
        if "/" in text:
            num, den = text.split("/")
            return round(float(num) / float(den), 6) if float(den) else None
        return float(text) if text else None
    except ValueError:
        # Frame numbers are a convenience for the judge; audio-only input is still separable.
        return None


def voice_segments(voice, rate, threshold_dbfs, frame_seconds, min_speech, min_gap, pad):
    """Energy gate on the vocals estimate, smoothed into speech spans. Energy, never phonetics."""
    hop = max(1, int(round(rate * frame_seconds)))
    count = voice.shape[0] // hop
    if not count:
        return []
    levels = np.sqrt(np.mean(np.square(voice[: count * hop].reshape(count, hop)), axis=1))
    floor = float(np.percentile(levels, 10))
    threshold = max(10 ** (threshold_dbfs / 20), floor * 3.0)
    active = levels > threshold
    spans, start = [], None
    for i, flag in enumerate(list(active) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append([start * hop, i * hop])
            start = None
    merged = []
    for span in spans:
        if merged and span[0] - merged[-1][1] < min_gap * rate:
            merged[-1][1] = span[1]
        else:
            merged.append(span)
    padding = int(round(pad * rate))
    out = []
    for begin, end in merged:
        if end - begin < min_speech * rate:
            continue
        begin = max(0, begin - padding)
        end = min(voice.shape[0], end + padding)
        if out and begin <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([begin, end])
    return [(int(a), int(b)) for a, b in out]


def cepstral_features(signal, rate, bands=26, coefficients=13):
    """Mean and spread of mel-like log-spectral cepstra: a weak, dependency-free voice print."""
    from scipy.fft import dct
    from scipy.signal import stft

    window = min(1024, max(256, 1 << int(math.log2(max(rate // 40, 256)))))
    _, _, spectra = stft(signal, fs=rate, nperseg=window, noverlap=window // 2)
    power = np.abs(spectra) ** 2
    freqs = np.linspace(0, rate / 2, power.shape[0])
    mel = 2595 * np.log10(1 + freqs / 700)
    edges = np.linspace(mel[1], mel.max(), bands + 2)
    filtered = []
    for lo, mid, hi in zip(edges[:-2], edges[1:-1], edges[2:]):
        weights = np.clip(
            np.minimum((mel - lo) / max(mid - lo, 1e-9), (hi - mel) / max(hi - mid, 1e-9)), 0, None
        )
        filtered.append(weights @ power)
    energies = np.log(np.array(filtered) + 1e-10)
    cepstra = dct(energies, axis=0, norm="ortho")[1:coefficients]
    return np.concatenate([cepstra.mean(axis=1), cepstra.std(axis=1)])


def cluster_segments(features, max_speakers):
    """Agglomerative clustering on cosine distance. Weak evidence, labelled as such."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    if len(features) < 2 or max_speakers < 2:
        return [1] * len(features), {"method": "single-cluster", "clusters": 1}
    matrix = np.array(features)
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    distances = pdist(matrix, metric="cosine")
    tree = linkage(distances, method="average")
    heights = tree[:, 2]
    counts = min(max_speakers, len(features))
    # Cut where the merge heights jump most, within the allowed speaker count.
    gaps = np.diff(heights)
    best = int(np.argmax(gaps[-(counts - 1) :])) if counts > 1 and gaps.size else 0
    clusters = min(counts, len(heights) - (len(heights) - counts + best))
    labels = fcluster(tree, t=max(1, clusters), criterion="maxclust")
    return [int(v) for v in labels], {
        "method": "average-linkage cosine on 24-d cepstral statistics",
        "clusters": int(len(set(labels))),
        "mergeHeights": [round(float(h), 4) for h in heights],
        "confidence": "low: short segments and a separation estimate; not a verified diarizer",
    }


def segment_speakers(
    out,
    max_speakers=2,
    threshold_dbfs=-45.0,
    frame_seconds=0.02,
    min_speech=0.25,
    min_gap=0.3,
    pad=0.1,
):
    """Split voice.wav into per-speaker stems that sum back to it exactly, plus segments.json.

    Every sample belongs to exactly one output: a speaker stem inside a detected span, or
    `unassigned.wav` elsewhere. Nothing is cross-faded, resampled, dropped or invented.
    """
    out = Path(out)
    receipt_path = out / "receipt.json"
    if not receipt_path.is_file():
        fail("Run the separation first; segments are derived from its voice.wav and receipt")
    receipt = json.loads(receipt_path.read_text())
    voice, rate = read_wav(out / "voice.wav")
    bed, _ = read_wav(out / "bed.wav")
    signal = voice[0]
    spans = voice_segments(signal, rate, threshold_dbfs, frame_seconds, min_speech, min_gap, pad)
    features = [cepstral_features(signal[a:b], rate) for a, b in spans]
    labels, clustering = (
        cluster_segments(features, max_speakers) if spans else ([], {"clusters": 0})
    )
    names = {label: f"speaker-{i + 1}" for i, label in enumerate(sorted(set(labels)))}
    masks = {name: np.zeros(signal.shape[0], dtype=bool) for name in names.values()}
    for (begin, end), label in zip(spans, labels):
        masks[names[label]][begin:end] = True
    assigned = np.zeros(signal.shape[0], dtype=bool)
    for name in masks:
        masks[name] &= ~assigned
        assigned |= masks[name]
    stems = {name: signal * mask for name, mask in masks.items()}
    stems["unassigned"] = signal * ~assigned
    for name, stem in stems.items():
        write_wav(out / f"{name}.wav", stem[None, :], rate)
    # Exactness is checked on the written integers, not on the float intermediates.
    written = sum(read_wav(out / f"{name}.wav")[0][0] for name in stems)
    if not np.array_equal(np.rint(written * FULL_SCALE), np.rint(signal * FULL_SCALE)):
        fail("Per-speaker stems do not sum back to voice.wav; refusing to write segments")
    fps = receipt.get("audio", {}).get("videoFps")
    bed_mono = bed.mean(axis=0)
    segments = []
    for index, ((begin, end), label) in enumerate(zip(spans, labels)):
        segments.append(
            {
                "index": index,
                "speaker": names[label],
                "startSeconds": round(begin / rate, 3),
                "endSeconds": round(end / rate, 3),
                "startSample": begin,
                "endSample": end,
                "startFrame": int(begin / rate * fps) if fps else None,
                "endFrame": int(math.ceil(end / rate * fps)) if fps else None,
                "voiceRms": round(rms(signal[begin:end]), 6),
                "bedRms": round(rms(bed_mono[begin:end]), 6),
                "attribution": {
                    "personId": None,
                    "method": None,
                    "confidence": None,
                    "offScreen": None,
                },
            }
        )
    document = {
        "schema": "wander.speaker-segments/1",
        "voice": {"file": "voice.wav", "sha256": receipt["outputs"]["voice"]["sha256"]},
        "source": {"path": receipt["source"]["path"], "sha256": receipt["source"]["sha256"]},
        "audio": {
            "sampleRate": rate,
            "samples": int(signal.shape[0]),
            "durationSeconds": round(signal.shape[0] / rate, 6),
            "videoFps": fps,
        },
        "provenance": f"{receipt['provenance']};{SEGMENT_PROVENANCE}",
        "provenanceNote": (
            "Segment boundaries are energy gates on a separated voice estimate and the speaker "
            "grouping is weak acoustic clustering. Neither identifies a person."
        ),
        "clustering": clustering,
        "speakers": [
            {
                "id": name,
                "file": f"{name}.wav",
                "segments": sum(1 for s in segments if s["speaker"] == name),
                "speechSeconds": round(float(masks[name].sum()) / rate, 3),
            }
            for name in sorted(masks)
        ]
        + [
            {
                "id": "unassigned",
                "file": "unassigned.wav",
                "segments": 0,
                "speechSeconds": round(float((~assigned).sum()) / rate, 3),
                "note": "Everything outside a detected speech span; keeps the sum exact.",
            }
        ],
        "segments": segments,
        "sumsToVoiceExactly": True,
        "judgeRequest": {
            "task": (
                "For each segment, decide which tracked person is speaking, or that the speaker "
                "is not visible. Answer only from the frames; do not guess from the audio."
            ),
            "inputsTheJudgeNeeds": [
                "A frame strip for each segment, sampled between startFrame and endFrame",
                "Each tracked person's id and 2D box drawn on those frames (people.json + tracks)",
                "The segment's startSeconds/endSeconds and the clip duration",
            ],
            "answerSchema": {
                "index": "segment index",
                "personId": "a people.json id, or null",
                "offScreen": "true when no visible person is speaking (narration, commentary)",
                "confidence": "0..1",
                "reason": "one short sentence citing what is visible",
            },
            "rules": [
                "A segment with offScreen true, or personId null, stays in the ambience bed.",
                "Never split one segment across two people; leave it unattributed instead.",
                "Attribution is recorded as attributed-by:<method>; it is never a human review.",
            ],
        },
        "reviewed": False,
        "reviewedBy": None,
    }
    (out / "segments.json").write_text(json.dumps(document, indent=2) + "\n")
    return document


def separate_audio(
    video,
    out,
    model_name="htdemucs",
    shifts=0,
    overlap=0.25,
    jobs=0,
    threshold_dbfs=-45.0,
    window=0.02,
):
    video, out = Path(video).resolve(), Path(out)
    if not video.is_file():
        fail(f"Source video not found: {video}")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        fail("ffmpeg and ffprobe are required")
    out.mkdir(parents=True, exist_ok=True)
    streams = probe_audio_streams(video)
    chosen, reason = choose_stream(streams)
    original_path = out / "original.wav"
    extract_original(video, chosen["index"], original_path)
    mix, rate = read_wav(original_path)
    if rate != chosen["sampleRate"]:
        fail("ffmpeg resampled the source; the native rate must be preserved")
    started = time.monotonic()
    vocals, non_vocals, model = separate(mix, rate, model_name, shifts, overlap, jobs)
    elapsed = time.monotonic() - started
    if vocals.shape != mix.shape:
        fail("Separated stems lost sample alignment with the original")
    voice = vocals.mean(axis=0, keepdims=True)
    bed = mix - vocals
    voice_clipped = write_wav(out / "voice.wav", voice, rate)
    bed_clipped = write_wav(out / "bed.wav", bed, rate)
    read_voice, voice_rate = read_wav(out / "voice.wav")
    read_bed, bed_rate = read_wav(out / "bed.wav")
    if read_voice.shape[1] != mix.shape[1] or read_bed.shape[1] != mix.shape[1]:
        fail("Written stems do not have the original sample count")
    if voice_rate != rate or bed_rate != rate:
        fail("Written stems do not have the original sample rate")
    written_residual = rms(mix - (read_bed + np.repeat(read_voice, mix.shape[0], axis=0)))
    receipt = {
        "schema": "wander.audio-separation/1",
        "source": {
            "path": video.name,
            "sha256": sha256_file(video),
            "audioStreams": streams,
            "chosenStreamIndex": chosen["index"],
            "streamChoiceReason": reason,
        },
        "model": model,
        "audio": {
            "sampleRate": rate,
            "channels": mix.shape[0],
            "samples": mix.shape[1],
            "durationSeconds": round(mix.shape[1] / rate, 6),
            "containerAudioDurationSeconds": chosen["durationSeconds"],
            "videoFps": video_fps(video),
        },
        "outputs": {
            "original": {"file": "original.wav", "channels": mix.shape[0]},
            "voice": {"file": "voice.wav", "channels": 1, "clippedSamples": voice_clipped},
            "bed": {"file": "bed.wav", "channels": bed.shape[0], "clippedSamples": bed_clipped},
        },
        "reconstruction": {
            "bedDefinition": "bed = original - vocals estimate, sample for sample",
            "sampleCountsMatch": True,
            "originalRms": round(rms(mix), 6),
            "originalRmsDbFs": dbfs(rms(mix)),
            "voiceRms": round(rms(voice), 6),
            "voiceRmsDbFs": dbfs(rms(voice)),
            "bedRms": round(rms(bed), 6),
            "bedRmsDbFs": dbfs(rms(bed)),
            "modelRoundTripResidualRms": round(rms(mix - (vocals + non_vocals)), 8),
            "modelRoundTripResidualDbFs": dbfs(rms(mix - (vocals + non_vocals))),
            "writtenVoicePlusBedResidualRms": round(written_residual, 8),
            "writtenVoicePlusBedResidualDbFs": dbfs(written_residual),
            "monoFoldRms": round(rms(vocals - np.repeat(voice, mix.shape[0], axis=0)), 8),
            "residualNote": (
                "modelRoundTrip measures vocals + no_vocals against the original (model and "
                "resampling loss). writtenVoicePlusBed measures the delivered files and therefore "
                "also carries the vocal stereo image folded to mono, plus 16-bit quantisation."
            ),
        },
        "voiceActivity": activity(voice[0], bed.mean(axis=0), rate, threshold_dbfs, window),
        "provenance": PROVENANCE,
        "provenanceNote": NOTE,
        "reviewed": False,
        "reviewedBy": None,
        "runtime": {
            "separationSeconds": round(elapsed, 2),
            "realtimeFactor": round(elapsed / (mix.shape[1] / rate), 3),
        },
    }
    receipt["outputs"]["voice"]["sha256"] = sha256_file(out / "voice.wav")
    receipt["outputs"]["bed"]["sha256"] = sha256_file(out / "bed.wav")
    receipt["outputs"]["original"]["sha256"] = sha256_file(original_path)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="Source video; omit with --segment-only")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument("--shifts", type=int, default=0, help="Demucs random shifts; 0 is fastest")
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument(
        "--jobs", type=int, default=0, help="torch CPU threads; 0 keeps the default"
    )
    parser.add_argument("--voice-threshold-dbfs", type=float, default=-45.0)
    parser.add_argument("--frame-seconds", type=float, default=0.02)
    parser.add_argument(
        "--segment-speakers",
        type=int,
        metavar="MAX",
        help="Also split voice.wav into at most MAX speaker stems that sum back to it exactly",
    )
    parser.add_argument(
        "--segment-only",
        action="store_true",
        help="Skip separation and segment an existing --out directory",
    )
    parser.add_argument("--min-speech-seconds", type=float, default=0.25)
    parser.add_argument("--min-gap-seconds", type=float, default=0.3)
    parser.add_argument("--pad-seconds", type=float, default=0.1)
    args = parser.parse_args()
    summary = {"out": str(args.out), "reviewed": False}
    try:
        if not args.segment_only:
            if not args.video:
                fail("--video is required unless --segment-only is passed")
            receipt = separate_audio(
                args.video,
                args.out,
                args.model,
                args.shifts,
                args.overlap,
                args.jobs,
                args.voice_threshold_dbfs,
                args.frame_seconds,
            )
            summary.update(
                stream=receipt["source"]["streamChoiceReason"],
                sampleRate=receipt["audio"]["sampleRate"],
                samples=receipt["audio"]["samples"],
                seconds=receipt["audio"]["durationSeconds"],
                separationSeconds=receipt["runtime"]["separationSeconds"],
                realtimeFactor=receipt["runtime"]["realtimeFactor"],
                voiceActiveFraction=receipt["voiceActivity"]["voiceActiveFraction"],
                modelRoundTripResidualDbFs=receipt["reconstruction"]["modelRoundTripResidualDbFs"],
                writtenVoicePlusBedResidualDbFs=receipt["reconstruction"][
                    "writtenVoicePlusBedResidualDbFs"
                ],
                provenance=receipt["provenance"],
            )
        if args.segment_speakers or args.segment_only:
            document = segment_speakers(
                args.out,
                args.segment_speakers or 2,
                args.voice_threshold_dbfs,
                args.frame_seconds,
                args.min_speech_seconds,
                args.min_gap_seconds,
                args.pad_seconds,
            )
            summary.update(
                segments=len(document["segments"]),
                speakers=[s["id"] for s in document["speakers"]],
                speechSeconds={s["id"]: s["speechSeconds"] for s in document["speakers"]},
                clustering=document["clustering"].get("method"),
                clusterConfidence=document["clustering"].get("confidence"),
                sumsToVoiceExactly=document["sumsToVoiceExactly"],
                provenance=document["provenance"],
            )
    except (ValueError, OSError, wave.Error) as error:
        parser.exit(2, f"Separation failed: {error}\n")
    print(json.dumps(summary, indent=2), file=sys.stdout)


if __name__ == "__main__":
    main()
