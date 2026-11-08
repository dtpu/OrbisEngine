import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from worker.stages.lhm_animate import source_pose_provenance

FRESH = {"sample": 0}
RETAINED = {"sample": 1, "retainedOriginalPose": True}


@pytest.mark.parametrize(
    ("records", "supplied_intrinsics", "fresh", "retained", "reestimated"),
    [
        pytest.param([FRESH], True, 1, 0, True, id="supplied-intrinsics"),
        pytest.param([FRESH], False, 1, 0, False, id="default-intrinsics"),
        pytest.param([RETAINED], True, 0, 1, False, id="retained-track"),
        pytest.param([FRESH, RETAINED, None], True, 1, 1, True, id="mixed-recovery"),
        pytest.param([None], True, 0, 0, False, id="missing-pose"),
    ],
)
def test_pose_provenance_counts_successful_samples(
    records, supplied_intrinsics, fresh, retained, reestimated
):
    assert source_pose_provenance(records, supplied_intrinsics=supplied_intrinsics) == {
        "freshSourcePoseSamples": fresh,
        "retainedSourcePoseSamples": retained,
        "sourcePosesReestimatedWithSuppliedIntrinsics": reestimated,
    }
