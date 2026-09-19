#!/usr/bin/env python3
"""Owned synthetic PCM fixtures only; no real media or services required."""

import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
import wave
from package_audio import prepare


def tone(path, channels=1, seconds=1, frequency=440):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(
            b"".join(
                struct.pack("<h", round(math.sin(i * frequency / 8000 * math.tau) * 3000))
                * channels
                for i in range(round(8000 * seconds))
            )
        )


class Packaging(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.world = self.root / "world"
        self.world.mkdir()
        tone(self.world / "original.wav", channels=2)
        self.base = {
            "schema": "wander.audio/1",
            "source": {"hasAudio": False},
            "timeline": {"durationSeconds": 1},
            "original": {"url": "original.wav", "offsetSeconds": 0},
            "provenance": {"attribution": "Owned fixture"},
        }
        (self.world / "audio.json").write_text(json.dumps(self.base))
        tone(self.root / "voice.wav")
        tone(self.root / "bed.wav", channels=2, frequency=220)
        (self.root / "head.json").write_text(
            json.dumps({"positions": [[0, 1, 0], [1, 1, 0]], "times": [0, 1]})
        )
        self.config = {
            "personIds": ["speaker"],
            "tracks": [
                {
                    "id": "voice",
                    "kind": "dialogue",
                    "file": "voice.wav",
                    "personId": "speaker",
                    "reviewed": True,
                    "anchorFile": "head.json",
                },
                {"id": "bed", "kind": "ambience", "file": "bed.wav"},
            ],
        }
        self.file = self.root / "config.json"

    def run_package(self, **kwargs):
        self.file.write_text(json.dumps(self.config))
        return prepare(self.file, self.world, **kwargs)

    def test_review_gate_leaves_original_untouched(self):
        before = (self.world / "audio.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "Human review"):
            self.run_package()
        self.assertEqual(before, (self.world / "audio.json").read_bytes())
        self.assertFalse((self.world / "audio-reviewed").exists())

    def test_package_retains_fallback_attribution_and_exact_samples(self):
        result = self.run_package(reviewed=True)
        self.assertEqual(result["original"], self.base["original"])
        self.assertEqual(result["provenance"], self.base["provenance"])
        self.assertEqual(result["defaultMode"], "original")
        self.assertTrue(result["spatialMixComplete"])
        self.assertEqual(
            (self.world / result["tracks"][0]["url"]).read_bytes(),
            (self.root / "voice.wav").read_bytes(),
        )
        anchor = json.loads((self.world / result["tracks"][0]["anchor"]["url"]).read_text())
        self.assertEqual(anchor["times"], [0, 1])
        self.assertEqual(self.run_package(reviewed=True)["tracks"], result["tracks"])

    def test_rejects_stereo_voice_and_mismatched_duration(self):
        tone(self.root / "voice.wav", channels=2)
        with self.assertRaisesRegex(ValueError, "mono"):
            self.run_package(reviewed=True)
        tone(self.root / "voice.wav", seconds=0.5)
        with self.assertRaisesRegex(ValueError, "full clip length"):
            self.run_package(reviewed=True)

    def test_rejects_unreviewed_unknown_identity_and_bad_head(self):
        self.config["tracks"][0]["personId"] = "guess"
        with self.assertRaisesRegex(ValueError, "personId"):
            self.run_package(reviewed=True)
        self.config["tracks"][0]["personId"] = "speaker"
        (self.root / "head.json").write_text(
            json.dumps({"positions": [[0, 1, 0], [1, 1, 0]], "times": [1, 0]})
        )
        with self.assertRaisesRegex(ValueError, "head times"):
            self.run_package(reviewed=True)
        self.assertEqual(json.loads((self.world / "audio.json").read_text()), self.base)

    def test_check_does_not_write_and_rejects_duplicating_original_mix(self):
        self.run_package(reviewed=True, check=True)
        self.assertFalse((self.world / "audio-reviewed").exists())
        self.config["tracks"][1]["file"] = "world/original.wav"
        with self.assertRaisesRegex(ValueError, "original mix"):
            self.run_package(reviewed=True)


if __name__ == "__main__":
    unittest.main()
