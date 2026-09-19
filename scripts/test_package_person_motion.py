#!/usr/bin/env python3
"""Synthetic gaussian PLY sequences only; no real media, GPU, or services required."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from plyfile import PlyData, PlyElement

from package_person_motion import CHANNELS, load_frames, package

FIELDS = (
    "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
)  # fmt: skip


def write_ply(path: Path, rows: np.ndarray):
    data = np.core.records.fromarrays(rows.T, names=",".join(FIELDS), formats=",".join(["f4"] * 17))
    PlyData([PlyElement.describe(data, "vertex")], text=False, byte_order="<").write(path)


def sequence(person: Path, frames=4, splats=64, drift_field=None, seed=1):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((splats, 17)).astype(np.float32)
    base[:, 3:6] = 0  # normals are zero in every packaged frame
    names = []
    for f in range(frames):
        rows = base.copy()
        rows[:, 0:3] += 0.05 * f * rng.standard_normal((splats, 3)).astype(np.float32)
        q = rows[:, 13:17] + 0.01 * f
        rows[:, 13:17] = q / np.linalg.norm(q, axis=1, keepdims=True)
        if drift_field is not None and f == frames - 1:
            rows[:, FIELDS.index(drift_field)] += 0.5
        name = f"frame_{f:03d}.ply"
        write_ply(person / name, rows)
        names.append(name)
    (person / "sequence.json").write_text(json.dumps({"frames": names, "fps": 12}))
    return names


class MotionTrack(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.person = Path(self.tmp.name) / "person"
        self.person.mkdir()

    def read_back(self, record):
        raw = (self.person / record["file"]).read_bytes()
        shape = (record["frames"], record["splats"], 7)
        if record["dtype"] == "float32":
            return np.frombuffer(raw, "<f4").reshape(shape).astype(np.float64)
        codes = np.frombuffer(raw, "<u2").reshape(shape).astype(np.float64)
        return codes * np.array(record["scale"]) + np.array(record["min"])

    def test_uint16_track_matches_plys_within_recorded_error(self):
        sequence(self.person)
        record = package(self.person)
        truth = load_frames(self.person, json.loads((self.person / "sequence.json").read_text()))
        back = self.read_back(record)
        err = np.abs(back - truth).max(axis=(0, 1))
        self.assertTrue(np.all(err <= np.array(record["maxAbsError"]) + 1e-12))
        self.assertLess(max(record["maxAbsError"]), 1e-3)
        self.assertEqual(record["bytes"], 4 * 64 * 7 * 2)
        self.assertGreater(record["plyBytes"], record["bytes"] * 4)
        seq = json.loads((self.person / "sequence.json").read_text())
        self.assertEqual(seq["motion"]["channels"], list(CHANNELS))
        self.assertEqual(seq["frames"], [f"frame_{f:03d}.ply" for f in range(4)])

    def test_lossless_track_is_bit_identical(self):
        sequence(self.person, frames=3, splats=16)
        record = package(self.person, lossless=True)
        truth = load_frames(self.person, json.loads((self.person / "sequence.json").read_text()))
        self.assertTrue(np.array_equal(self.read_back(record).astype(np.float32), truth))
        self.assertEqual(record["maxAbsError"], [0.0] * 7)
        self.assertEqual(record["file"], "motion.f32")

    def test_static_attribute_drift_is_refused(self):
        sequence(self.person, drift_field="opacity")
        with self.assertRaisesRegex(ValueError, "opacity differs from frame 0"):
            package(self.person)
        self.assertFalse((self.person / "motion.u16").exists())
        self.assertNotIn("motion", json.loads((self.person / "sequence.json").read_text()))

    def test_splat_count_mismatch_is_refused(self):
        names = sequence(self.person, frames=2, splats=8)
        rows = np.zeros((9, 17), np.float32)
        write_ply(self.person / names[1], rows)
        with self.assertRaisesRegex(ValueError, "has 9 splats"):
            package(self.person)

    def test_constant_channel_quantises_to_itself(self):
        sequence(self.person, frames=2, splats=8)
        for name in ("frame_000.ply", "frame_001.ply"):
            v = PlyData.read(self.person / name)["vertex"].data.copy()
            v["z"] = 2.5
            PlyData([PlyElement.describe(v, "vertex")], text=False, byte_order="<").write(
                self.person / name
            )
        record = package(self.person)
        back = self.read_back(record)
        self.assertTrue(np.all(back[:, :, 2] == 2.5))
        self.assertEqual(record["maxAbsError"][2], 0.0)


if __name__ == "__main__":
    unittest.main()
