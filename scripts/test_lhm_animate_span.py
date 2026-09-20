"""Motion is planned over the span that decodes, not the one the container advertises.

The frozen person was reconstructed on the GPU and the animation then stopped dead on
"Camera records must correspond exactly to sampled source indices": rocky advertises 448
frames over 18.685 s, decodes 358 over 14.944 s, and the schedule every stage derives from
that claim therefore ran 46 samples past the end of the video. The dense solve and the
tracker both truncate to what they can read -- their cameras and seed poses cover the real
clip -- and this stage did not, so nothing lined up with them.
"""

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_policy():
    """Extract the pure-python span rules; the module itself needs torch and CUDA."""
    source = (ROOT / "worker/stages/lhm_animate.py").read_text()
    match = re.search(r"^MINIMUM_DECODED_FRAMES = .*?^def main\(", source, re.S | re.M)
    if match is None:  # pragma: no cover - the file would have to be restructured
        raise AssertionError("lhm_animate.py no longer defines its decode-span policy")
    namespace: dict = {}
    exec(match.group(0).rsplit("def main(", 1)[0], namespace)  # noqa: S102
    return namespace


class DecodableSpanTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy()
        self.reads = []

    def clip(self, decodable: int):
        """A file whose frames read up to `decodable` and fail after it."""

        def readable(position: int) -> bool:
            self.reads.append(position)
            return position < decodable

        return readable

    def span(self, decodable: int, planned: int) -> int:
        return self.policy["decodable_span"](self.clip(decodable), planned)

    def test_a_clip_that_decodes_to_the_end_is_untouched(self):
        self.assertEqual(self.span(225, 225), 225)

    def test_the_rocky_tail_is_found_exactly(self):
        """225 samples planned from a 448-frame claim; 179 cameras exist, and 179 decode."""
        self.assertEqual(self.span(179, 225), 179)

    def test_every_boundary_is_found_exactly(self):
        for decodable in range(0, 40):
            with self.subTest(decodable=decodable):
                self.reads.clear()
                self.assertEqual(self.span(decodable, 39), decodable)

    def test_a_file_that_opens_but_reads_nothing_is_zero(self):
        self.assertEqual(self.span(0, 225), 0)

    def test_an_empty_schedule_is_zero_without_touching_the_file(self):
        self.assertEqual(self.span(225, 0), 0)
        self.assertEqual(self.reads, [])

    def test_the_clip_is_not_decoded_twice_to_find_the_edge(self):
        """Probing every sample would double the decode cost of the stage."""
        self.span(179, 225)
        self.assertLess(len(self.reads), 16, f"{len(self.reads)} probes to halve 225 samples")

    def test_losing_most_of_the_clip_is_a_broken_source_not_a_short_one(self):
        usable = self.policy["decoded_span_usable"]
        self.assertTrue(usable(179, 225), "losing a short tail is ordinary")
        self.assertFalse(usable(20, 225), "losing most of the clip is a broken source")
        self.assertFalse(usable(0, 0), "nothing requested is nothing to animate")


if __name__ == "__main__":
    unittest.main()
