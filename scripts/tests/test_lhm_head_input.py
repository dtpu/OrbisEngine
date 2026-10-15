"""Head conditioning excludes bystander pixels outside the supplied person mask."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worker"))

from stages.lhm_person import prepare_head_input


def test_head_selection_excludes_larger_bystander():
    raw = np.full((60, 100, 3), 255, dtype=np.uint8)
    raw[10:20, 10:20] = [220, 30, 40]
    raw[5:30, 60:85] = [20, 40, 220]
    mask = np.zeros(raw.shape[:2], np.uint8)
    mask[10:20, 10:20] = 255

    def largest_visible_head(image):
        # The pinned detector selects the largest detected head; emulate that choice
        # without loading weights. It must only receive the chosen person's pixels.
        rgb = image.permute(1, 2, 0).numpy()
        boxes = []
        for color in ([220, 30, 40], [20, 40, 220]):
            ys, xs = np.where(np.all(rgb == color, axis=2))
            if len(xs):
                boxes.append((len(xs), [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]))
        return torch.tensor(max(boxes)[1])

    head = prepare_head_input(raw, mask, largest_visible_head)

    assert head.shape == (112, 112, 3)
    assert np.all(head == [220, 30, 40])
    # Masking must not mutate the retained source.
    assert np.all(raw[5:30, 60:85] == [20, 40, 220])


def test_expanded_head_crop_keeps_other_people_masked_out():
    raw = np.full((30, 40, 3), [20, 40, 220], dtype=np.uint8)
    raw[10:20, 10:20] = [220, 30, 40]
    mask = np.zeros(raw.shape[:2], np.uint8)
    mask[10:20, 10:20] = 255
    expected = np.full_like(raw, 255)
    expected[mask > 127] = raw[mask > 127]

    def expanded_box(image):
        assert np.array_equal(image.permute(1, 2, 0).numpy(), expected)
        return torch.tensor([-2, -3, 43, 34])

    head = prepare_head_input(raw, mask, expanded_box)

    assert np.array_equal(head, cv2.resize(expected, (112, 112), interpolation=cv2.INTER_AREA))


def test_full_person_mask_preserves_original_head_pixels():
    raw = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
    mask = np.full(raw.shape[:2], 255, dtype=np.uint8)
    head = prepare_head_input(raw, mask, lambda _: torch.tensor([2, 3, 18, 19]))
    assert np.array_equal(
        head, cv2.resize(raw[3:19, 2:18], (112, 112), interpolation=cv2.INTER_AREA)
    )


@pytest.mark.parametrize("box", [[0, 0, 0, 0], [50, 50, 60, 60], [0, 0, 5, 5]])
def test_head_box_without_selected_person_is_unusable(box):
    raw = np.zeros((30, 40, 3), np.uint8)
    mask = np.zeros(raw.shape[:2], np.uint8)
    mask[10:20, 10:20] = 255
    with pytest.raises(ValueError, match="selected person"):
        prepare_head_input(raw, mask, lambda _: torch.tensor(box))
