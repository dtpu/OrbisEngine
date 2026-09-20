#!/usr/bin/env python3
"""Owned synthetic audio only. No network, no real media; the separator itself is mocked.

One optional slow case runs the real Demucs model on 3 s of synthetic audio and is skipped
unless the `audio` dependency group is installed (`uv sync --locked --group audio`).
"""

import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import wave

import numpy as np

import package_audio
import separate_audio

HAS_DEMUCS = importlib.util.find_spec("demucs") is not None
HAS_FFMPEG = separate_audio.shutil.which("ffmpeg") is not None


def synthetic_mix(seconds=3.0, rate=8000):
    """Sine bursts standing in for speech over a steady pseudo-random bed, both owned."""
    n = int(seconds * rate)
    t = np.arange(n) / rate
    gate = ((t % 1.0) < 0.45).astype(float)
    voice = 0.35 * gate * np.sin(2 * math.pi * 220 * t)
    rng = np.random.default_rng(7)
    bed = 0.08 * rng.standard_normal(n)
    mix = np.stack([voice + bed, voice + 0.9 * bed])
    return mix, voice[None, :], rate


def write_wav(path, samples, rate):
    data = np.clip(np.rint(samples * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(samples.shape[0])
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.T.reshape(-1).tobytes())


def tone(path, channels=1, seconds=1, frequency=440, rate=8000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(
            b"".join(
                struct.pack("<h", round(math.sin(i * frequency / rate * math.tau) * 3000))
                * channels
                for i in range(round(rate * seconds))
            )
        )


class StreamChoice(unittest.TestCase):
    def test_prefers_the_decodable_stereo_stream_and_records_the_rejects(self):
        streams = [
            {
                "index": 1,
                "codecName": "aac",
                "codecTag": "mp4a",
                "channels": 2,
                "sampleRate": 48000,
                "decodable": True,
            },
            {
                "index": 2,
                "codecName": None,
                "codecTag": "apac",
                "channels": 4,
                "sampleRate": 48000,
                "decodable": False,
            },
        ]
        chosen, reason = separate_audio.choose_stream(streams)
        self.assertEqual(chosen["index"], 1)
        self.assertIn("2-channel aac", reason)
        self.assertIn("apac", reason)
        self.assertIn("not decodable", reason)

    def test_rejects_a_file_whose_only_audio_cannot_be_decoded(self):
        with self.assertRaisesRegex(ValueError, "No decodable audio stream"):
            separate_audio.choose_stream(
                [
                    {
                        "index": 2,
                        "codecName": None,
                        "codecTag": "apac",
                        "channels": 4,
                        "sampleRate": 48000,
                        "decodable": False,
                    }
                ]
            )

    def test_rejects_a_silent_container(self):
        with self.assertRaisesRegex(ValueError, "no audio stream"):
            separate_audio.choose_stream([])


class AlignmentAndReceipt(unittest.TestCase):
    """The separator is mocked: only alignment, bed arithmetic and the receipt are under test."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.mix, self.voice, self.rate = synthetic_mix()
        self.out = self.root / "out"
        self.out.mkdir()
        write_wav(self.out / "original.wav", self.mix, self.rate)
        self.video = self.root / "clip.mov"
        self.video.write_bytes(b"not a real container")
        self.streams = [
            {
                "index": 1,
                "codecName": "aac",
                "codecTag": "mp4a",
                "channels": 2,
                "channelLayout": "stereo",
                "sampleRate": self.rate,
                "durationSeconds": self.mix.shape[1] / self.rate,
                "decodable": True,
            }
        ]

    def run_separation(self, vocals=None, **kwargs):
        """Patch probe/extract/separate so no ffmpeg or model is needed."""
        estimate = self.voice if vocals is None else vocals
        stereo = np.repeat(estimate, 2, axis=0) if estimate.shape[0] == 1 else estimate
        original = separate_audio.probe_audio_streams, separate_audio.extract_original
        model = separate_audio.separate
        separate_audio.probe_audio_streams = lambda video: self.streams
        separate_audio.extract_original = lambda video, index, destination: None
        separate_audio.separate = lambda mix, rate, name, shifts, overlap, jobs: (
            stereo,
            mix - stereo if stereo.shape == mix.shape else np.zeros_like(stereo),
            {"name": name, "signature": "mocked", "modelSampleRate": rate},
        )
        try:
            return separate_audio.separate_audio(self.video, self.out, **kwargs)
        finally:
            separate_audio.probe_audio_streams, separate_audio.extract_original = original
            separate_audio.separate = model

    def test_voice_and_bed_preserve_the_original_sample_grid(self):
        receipt = self.run_separation()
        self.assertEqual(receipt["audio"]["samples"], self.mix.shape[1])
        self.assertEqual(receipt["audio"]["sampleRate"], self.rate)
        voice, voice_rate = separate_audio.read_wav(self.out / "voice.wav")
        bed, bed_rate = separate_audio.read_wav(self.out / "bed.wav")
        self.assertEqual(voice.shape, (1, self.mix.shape[1]))
        self.assertEqual(bed.shape, self.mix.shape)
        self.assertEqual((voice_rate, bed_rate), (self.rate, self.rate))
        self.assertTrue(receipt["reconstruction"]["sampleCountsMatch"])

    def test_voice_plus_bed_reconstructs_the_original(self):
        receipt = self.run_separation()
        voice, _ = separate_audio.read_wav(self.out / "voice.wav")
        bed, _ = separate_audio.read_wav(self.out / "bed.wav")
        original, _ = separate_audio.read_wav(self.out / "original.wav")
        residual = separate_audio.rms(original - (bed + np.repeat(voice, 2, axis=0)))
        # The mocked vocals are identical in both channels, so only quantisation remains.
        self.assertLess(residual, 1e-4)
        self.assertAlmostEqual(
            residual, receipt["reconstruction"]["writtenVoicePlusBedResidualRms"], places=6
        )

    def test_stereo_vocals_folded_to_mono_are_reported_not_hidden(self):
        wide = np.stack([self.voice[0], -self.voice[0]])
        receipt = self.run_separation(vocals=wide)
        self.assertGreater(receipt["reconstruction"]["monoFoldRms"], 0.01)
        # The written residual is the mono fold plus 16-bit quantisation only.
        self.assertAlmostEqual(
            receipt["reconstruction"]["monoFoldRms"],
            receipt["reconstruction"]["writtenVoicePlusBedResidualRms"],
            places=4,
        )

    def test_receipt_records_provenance_stream_and_activity(self):
        receipt = self.run_separation()
        self.assertEqual(receipt["provenance"], "separated-estimate:demucs-htdemucs")
        self.assertIn("not recorded stems", receipt["provenanceNote"])
        self.assertIs(receipt["reviewed"], False)
        self.assertIsNone(receipt["reviewedBy"])
        self.assertEqual(receipt["source"]["chosenStreamIndex"], 1)
        self.assertEqual(len(receipt["source"]["sha256"]), 64)
        activity = receipt["voiceActivity"]
        # The synthetic gate is on 45% of the time.
        self.assertAlmostEqual(activity["voiceActiveFraction"], 0.45, delta=0.05)
        self.assertEqual(len(activity["voiceRmsPerSecond"]), 3)
        self.assertEqual(len(activity["bedRmsPerSecond"]), 3)
        self.assertEqual(json.loads((self.out / "receipt.json").read_text()), receipt)

    def test_silent_voice_estimate_reports_no_activity(self):
        receipt = self.run_separation(vocals=np.zeros_like(self.voice))
        self.assertEqual(receipt["voiceActivity"]["voiceActiveFraction"], 0.0)

    def test_misaligned_stems_are_refused(self):
        with self.assertRaisesRegex(ValueError, "sample alignment"):
            self.run_separation(vocals=self.voice[:, :-10])


class SpeakerSegments(unittest.TestCase):
    """Phase 2: per-speaker stems must partition voice.wav, never resynthesise it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        rate = 8000
        n = 6 * rate
        t = np.arange(n) / rate
        # Two alternating "speakers": different pitch and formant-like partials, 1 s turns.
        turn = (t.astype(int) % 2) == 0
        low = 0.4 * (np.sin(2 * math.pi * 130 * t) + 0.5 * np.sin(2 * math.pi * 520 * t))
        high = 0.4 * (np.sin(2 * math.pi * 240 * t) + 0.5 * np.sin(2 * math.pi * 1450 * t))
        gate = ((t % 1.0) < 0.7).astype(float)
        voice = gate * np.where(turn, low, high)
        bed = 0.02 * np.random.default_rng(3).standard_normal(n)
        write_wav(self.out / "voice.wav", voice[None, :], rate)
        write_wav(self.out / "bed.wav", np.stack([bed, bed]), rate)
        (self.out / "receipt.json").write_text(
            json.dumps(
                {
                    "schema": "wander.audio-separation/1",
                    "provenance": "separated-estimate:demucs-htdemucs",
                    "source": {"path": "synthetic.wav", "sha256": "0" * 64},
                    "outputs": {"voice": {"sha256": "1" * 64}},
                    "audio": {"videoFps": 30.0},
                }
            )
        )

    def test_stems_sum_back_to_the_voice_track_exactly(self):
        document = separate_audio.segment_speakers(self.out, max_speakers=2)
        voice, _ = separate_audio.read_wav(self.out / "voice.wav")
        total = np.zeros_like(voice[0])
        for speaker in document["speakers"]:
            stem, _ = separate_audio.read_wav(self.out / speaker["file"])
            self.assertEqual(stem.shape, voice.shape)
            total += stem[0]
        np.testing.assert_array_equal(np.rint(total * 32767), np.rint(voice[0] * 32767))
        self.assertTrue(document["sumsToVoiceExactly"])

    def test_segments_carry_times_frames_and_an_empty_attribution(self):
        document = separate_audio.segment_speakers(self.out, max_speakers=2)
        self.assertGreaterEqual(len(document["segments"]), 4)
        previous = -1.0
        for segment in document["segments"]:
            self.assertGreater(segment["endSeconds"], segment["startSeconds"])
            self.assertGreaterEqual(segment["startSeconds"], previous)
            previous = segment["startSeconds"]
            self.assertEqual(segment["startFrame"], int(segment["startSample"] / 8000 * 30))
            self.assertIsNone(segment["attribution"]["personId"])
            self.assertIsNone(segment["attribution"]["offScreen"])
        self.assertIn("segmented-by:energy-vad", document["provenance"])
        self.assertIs(document["reviewed"], False)
        self.assertIn("judgeRequest", document)
        self.assertEqual(json.loads((self.out / "segments.json").read_text()), document)

    def test_two_voices_produce_two_clusters(self):
        document = separate_audio.segment_speakers(self.out, max_speakers=2)
        speakers = {s["id"] for s in document["speakers"] if s["id"] != "unassigned"}
        self.assertEqual(len(speakers), 2)
        self.assertIn("low", document["clustering"]["confidence"])

    def test_one_speaker_limit_keeps_a_single_stem(self):
        document = separate_audio.segment_speakers(self.out, max_speakers=1)
        speakers = [s["id"] for s in document["speakers"] if s["id"] != "unassigned"]
        self.assertEqual(speakers, ["speaker-1"])

    def test_silence_yields_no_segments_and_a_full_unassigned_stem(self):
        write_wav(self.out / "voice.wav", np.zeros((1, 8000)), 8000)
        write_wav(self.out / "bed.wav", np.zeros((2, 8000)), 8000)
        document = separate_audio.segment_speakers(self.out)
        self.assertEqual(document["segments"], [])
        self.assertEqual(document["speakers"][-1]["id"], "unassigned")
        self.assertAlmostEqual(document["speakers"][-1]["speechSeconds"], 1.0, places=3)

    def test_segmenting_without_a_separation_receipt_is_refused(self):
        (self.out / "receipt.json").unlink()
        with self.assertRaisesRegex(ValueError, "Run the separation first"):
            separate_audio.segment_speakers(self.out)


class ManifestFromSeparation(unittest.TestCase):
    """The separated pair must package into a manifest the viewer's own rules accept."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.world = self.root / "world"
        self.world.mkdir()
        mix, voice, rate = synthetic_mix(seconds=1.0)
        write_wav(self.world / "original.wav", mix, rate)
        self.base = {
            "schema": "wander.audio/1",
            "source": {"hasAudio": False},
            "timeline": {"durationSeconds": 1},
            "original": {"url": "original.wav", "offsetSeconds": 0},
        }
        (self.world / "audio.json").write_text(json.dumps(self.base))
        self.sep = self.root / "sep"
        self.sep.mkdir()
        write_wav(self.sep / "voice.wav", voice, rate)
        write_wav(self.sep / "bed.wav", mix - np.repeat(voice, 2, axis=0), rate)
        (self.sep / "receipt.json").write_text(
            json.dumps(
                {
                    "schema": "wander.audio-separation/1",
                    "provenance": "separated-estimate:demucs-htdemucs",
                    "outputs": {"voice": {"file": "voice.wav"}, "bed": {"file": "bed.wav"}},
                }
            )
        )
        self.anchor = self.root / "head.json"
        self.anchor.write_text(json.dumps({"positions": [[0, 1, 0], [0.1, 1, 0]], "times": [0, 1]}))

    def package(self, **kwargs):
        config, base, _ = package_audio.separation_config(
            self.sep, kwargs.pop("person_id", "person-1"), self.anchor, **kwargs
        )
        return package_audio.prepare(
            base, self.world, reviewed=True, config=config, reviewed_by="tester"
        )

    def test_manifest_satisfies_the_viewer_track_rules(self):
        manifest = self.package(attributed_by="vlm-judge")
        self.assertEqual(manifest["schema"], "wander.audio/1")
        self.assertTrue(manifest["spatialMixComplete"])
        self.assertEqual(manifest["original"], self.base["original"])
        self.assertEqual(manifest["review"]["reviewedBy"], "tester")
        kinds = {t["kind"] for t in manifest["tracks"]}
        self.assertEqual(kinds, {"dialogue", "ambience"})
        ids = set()
        for track in manifest["tracks"]:
            # docs/audio.md and src/audio/fourd-audio.ts:52-74
            self.assertIsInstance(track["id"], str)
            self.assertTrue(track["id"] and track["id"] not in ids)
            ids.add(track["id"])
            self.assertIn(track["kind"], ("dialogue", "ambience"))
            self.assertTrue(track["url"] and not track["url"].startswith("/"))
            self.assertNotIn("://", track["url"])
            self.assertEqual(track["offsetSeconds"], 0)
            if track["kind"] != "dialogue":
                continue
            self.assertIs(track["reviewed"], True)
            self.assertEqual(track["channels"], 1)
            self.assertEqual(track["personId"], "person-1")
            self.assertEqual(track["anchor"]["space"], "person-local")
            self.assertEqual(track["anchor"]["offsetBodyHeights"], [0, 0, 0])
            self.assertTrue(track["provenance"].startswith("separated-estimate:demucs-htdemucs"))
            self.assertIn("attributed-by:vlm-judge", track["provenance"])
            anchor = json.loads((self.world / track["anchor"]["url"]).read_text())
            self.assertEqual(len(anchor["positions"]), len(anchor["times"]))
            self.assertEqual(anchor["times"][0], 0)

    def test_packaged_audio_is_byte_identical_to_the_separated_files(self):
        manifest = self.package()
        for track in manifest["tracks"]:
            name = "voice.wav" if track["kind"] == "dialogue" else "bed.wav"
            self.assertEqual(
                (self.world / track["url"]).read_bytes(), (self.sep / name).read_bytes()
            )

    def test_unknown_receipt_schema_is_refused(self):
        (self.sep / "receipt.json").write_text(json.dumps({"schema": "something/9"}))
        with self.assertRaisesRegex(ValueError, "receipt schema"):
            self.package()

    def test_review_gate_still_refuses_an_unattested_package(self):
        config, base, _ = package_audio.separation_config(self.sep, "person-1", self.anchor)
        with self.assertRaisesRegex(ValueError, "Human review"):
            package_audio.prepare(base, self.world, reviewed=False, config=config)
        self.assertEqual(json.loads((self.world / "audio.json").read_text()), self.base)

    def test_cli_refuses_separation_without_the_operator_flag(self):
        result = subprocess.run(
            [
                separate_audio.sys.executable,
                str(Path(__file__).with_name("package_audio.py")),
                "--separation",
                str(self.sep),
                "--world",
                str(self.world),
                "--person-id",
                "person-1",
                "--anchor-file",
                str(self.anchor),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--voice-reviewed-by", result.stderr)
        self.assertEqual(json.loads((self.world / "audio.json").read_text()), self.base)


@unittest.skipUnless(HAS_DEMUCS and HAS_FFMPEG, "install the optional `audio` group to run")
class RealModel(unittest.TestCase):
    """Slow: the real htdemucs weights on 3 s of owned synthetic audio."""

    def test_real_demucs_keeps_the_sample_grid(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            mix, _, rate = synthetic_mix(seconds=3.0, rate=44100)
            source = root / "mix.wav"
            write_wav(source, mix, rate)
            video = root / "clip.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=64x64:r=10:d=3",
                    "-i",
                    str(source),
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    "-shortest",
                    str(video),
                ],
                check=True,
                capture_output=True,
            )
            receipt = separate_audio.separate_audio(video, root / "out")
            self.assertEqual(receipt["model"]["name"], "htdemucs")
            voice, voice_rate = separate_audio.read_wav(root / "out" / "voice.wav")
            bed, _ = separate_audio.read_wav(root / "out" / "bed.wav")
            self.assertEqual(voice_rate, receipt["audio"]["sampleRate"])
            self.assertEqual(voice.shape[1], receipt["audio"]["samples"])
            self.assertEqual(bed.shape[1], receipt["audio"]["samples"])
            self.assertLess(receipt["reconstruction"]["modelRoundTripResidualRms"], 0.02)


if __name__ == "__main__":
    unittest.main()
