"""A sample with nobody in it is not a sample that failed to reconstruct.

LHM animated every frame of a clip and the stage then failed, after the GPU had been paid
for, because one of 180 requested samples had no pose: the runner enters 83 ms in. The stage
records why each sample is missing, and "the track is not in this one" is an ordinary fact
about a clip rather than a partial reconstruction.
"""

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker" / "stages"))

from lhm_recovery import output_problems

ABSENT = "Track absent from this sample (out of frame, undetected, or occluded)"
COVERAGE = "source motion coverage is partial or unverified"


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="lhm-coverage-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def motion(self, frames: int, requested: int, missing: list[dict]) -> list[str]:
        """A finished motion export with `frames` PLYs out of `requested` samples."""
        names = [f"frame_{index:03d}.ply" for index in range(frames)]
        digests = []
        for name in names:
            (self.root / name).write_bytes(name.encode())
            digests.append(hashlib.sha256(name.encode()).hexdigest())
        sequence = {
            "frames": names,
            "frame_sha256": digests,
            "count": frames,
            "timestamps": [index / 12 for index in range(frames)],
            "sourceIndices": list(range(frames)),
            "requestedSamples": requested,
            "allRequestedSamplesReconstructed": not missing,
            "missingPoseSamples": missing,
        }
        (self.root / "sequence.json").write_text(json.dumps(sequence))
        (self.root / "motion.json").write_text(json.dumps({"frames": names, "missing": missing}))
        (self.root / "missing-poses.json").write_text(json.dumps(missing))
        for name in ("modal-run.json", "inference.log", "source-poses.pt"):
            (self.root / name).write_bytes(b"{}" if name.endswith(".json") else b"x")
        files = {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in self.root.iterdir()
        }
        return output_problems(self.root, "motion", files)

    def test_an_actor_who_arrives_late_is_not_a_failure(self):
        """179 frames of 180 requested, the missing one explained: the rocky clip."""
        problems = self.motion(179, 180, [{"sample": 0, "sourceIndex": 0, "reason": ABSENT}])
        self.assertNotIn(COVERAGE, problems, problems)

    def test_a_complete_clip_is_still_complete(self):
        self.assertNotIn(COVERAGE, self.motion(180, 180, []))

    def test_frames_that_vanished_with_no_explanation_still_fail(self):
        """What the guard is for: outputs short of the plan and nothing saying why."""
        self.assertIn(COVERAGE, self.motion(179, 180, []))

    def test_a_sample_missing_for_another_reason_still_fails(self):
        problems = self.motion(179, 180, [{"sample": 42, "reason": "inference failed here"}])
        self.assertIn(COVERAGE, problems)

    def test_an_absence_that_does_not_add_up_still_fails(self):
        """Two explained absences cannot account for three missing frames."""
        missing = [{"sample": 0, "reason": ABSENT}, {"sample": 1, "reason": ABSENT}]
        self.assertIn(COVERAGE, self.motion(177, 180, missing))


if __name__ == "__main__":
    unittest.main()
