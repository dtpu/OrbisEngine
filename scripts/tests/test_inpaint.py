"""Signal-isolation and legacy compatibility tests with deterministic PIL stubs only."""

from pathlib import Path
import sys
from unittest.mock import Mock

import cv2
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from worker.wander_worker.inpaint import inpaint_frame


def sample(height=108, width=192):
    rng = np.random.default_rng(114)
    image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), bool)
    mask[height // 3 : height * 2 // 3, width // 4 : width * 3 // 4] = True
    return image, mask


def patterned_fill(rgb, mask):
    pixels = np.array(rgb)
    support = np.array(mask) // 255
    return Image.fromarray(
        ((pixels.astype(np.uint16) * 3 + support[..., None] * 71) % 256).astype(np.uint8)
    )


def legacy_reference(image, mask, model, lama_px, passes):
    # Literal pre-extraction worker loop, held as a compatibility oracle.
    height, width = image.shape[:2]
    sw, sh = lama_px // 8 * 8, int(lama_px * height / width) // 8 * 8
    im = image.copy()
    if mask.any():
        for _ in range(passes):
            small = Image.fromarray(im).resize((sw, sh), Image.LANCZOS)
            sm = Image.fromarray(mask.astype(np.uint8) * 255).resize((sw, sh), Image.NEAREST)
            fill = np.asarray(model(small, sm).resize((width, height), Image.BICUBIC))
            alpha = np.clip(
                cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3)[..., None] * 1.5, 0, 1
            )
            im = (im * (1 - alpha) + fill * alpha).astype(np.uint8)
    return im


@pytest.mark.parametrize("shape,passes", [((108, 192), 1), ((79, 117), 3), ((96, 40), 2)])
def test_legacy_is_pixel_identical_to_existing_worker_loop(shape, passes):
    image, mask = sample(*shape)
    image_before, mask_before = image.copy(), mask.copy()
    expected = legacy_reference(image, mask, patterned_fill, 64, passes)
    actual = inpaint_frame(image, mask, patterned_fill, lama_px=64, passes=passes)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, image_before)
    np.testing.assert_array_equal(mask, mask_before)
    assert actual.dtype == np.uint8


@pytest.mark.parametrize("passes", [1, 3])
def test_isolated_foreground_mutations_cannot_change_model_input_or_result(passes):
    image, mask = sample()
    dark, light = image.copy(), image.copy()
    dark[mask], light[mask] = 0, 255
    seen = []

    def input_sensitive_stub(rgb, small_mask):
        pixels = np.array(rgb)
        seen.append((pixels.copy(), np.array(small_mask).copy()))
        # Depend on every input pixel: any escaped contamination changes this fill.
        value = int(pixels.astype(np.uint64).sum() % 251)
        return Image.new("RGB", rgb.size, (value, 255 - value, value // 2))

    output_dark = inpaint_frame(
        dark, mask, input_sensitive_stub, lama_px=64, passes=passes, mode="isolated"
    )
    output_light = inpaint_frame(
        light, mask, input_sensitive_stub, lama_px=64, passes=passes, mode="isolated"
    )
    assert len(seen) == 2 * passes
    for rgb, support in seen:
        np.testing.assert_array_equal(rgb, seen[0][0])
        np.testing.assert_array_equal(support, seen[0][1])
    np.testing.assert_array_equal(output_dark, output_light)
    np.testing.assert_array_equal(output_dark[~mask], image[~mask])


def test_isolated_composite_uses_identical_supplied_fill_fully_inside_mask():
    image, mask = sample()
    changed = image.copy()
    changed[mask] = 255 - changed[mask]
    model = Mock(side_effect=lambda rgb, mask: Image.new("RGB", rgb.size, (37, 81, 229)))
    actual = inpaint_frame(image, mask, model, lama_px=64, mode="isolated")
    second = inpaint_frame(changed, mask, model, lama_px=64, mode="isolated")
    np.testing.assert_array_equal(actual[mask], np.tile([37, 81, 229], (mask.sum(), 1)))
    np.testing.assert_array_equal(actual, second)
    np.testing.assert_array_equal(actual[~mask], image[~mask])


@pytest.mark.parametrize("kind", ["corner-pixel", "thin-row", "thin-column"])
def test_subpixel_masks_survive_and_cover_actual_lanczos_contamination(kind):
    image = np.zeros((108, 192, 3), np.uint8)
    mask = np.zeros(image.shape[:2], bool)
    if kind == "corner-pixel":
        mask[0, 0] = True
    elif kind == "thin-row":
        mask[0, :] = True
    else:
        mask[:, 0] = True
    # These exact thin supports disappear under the old nearest resize.
    nearest = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((64, 32), Image.NEAREST))
    assert not nearest.any()
    mutated = image.copy()
    mutated[mask] = 255
    changed_low_rgb = np.any(
        np.array(Image.fromarray(mutated).resize((64, 32), Image.LANCZOS)) != 0, axis=-1
    )
    observed = []

    def stub(rgb, support):
        assert rgb.mode == "RGB" and support.mode == "L"
        assert rgb.size == support.size == (64, 32)
        observed.append(np.array(support) > 0)
        # Foreground was neutralized before RGB filtering.
        assert not np.array(rgb).any()
        return Image.new("RGB", rgb.size, "white")

    actual = inpaint_frame(mutated, mask, stub, lama_px=64, mode="isolated")
    assert observed[0].any()
    assert not (changed_low_rgb & ~observed[0]).any()
    assert np.all(actual[mask] == 255)
    np.testing.assert_array_equal(actual[~mask], image[~mask])


@pytest.mark.parametrize("shape,lama_px", [((16, 16), 960), ((11, 17), 64), ((5, 37), 128)])
@pytest.mark.parametrize("position", ["corner", "edge", "interior"])
def test_upsampled_single_pixel_mask_covers_actual_lanczos_filter_support(shape, lama_px, position):
    height, width = shape
    image = np.zeros((height, width, 3), np.uint8)
    mask = np.zeros(shape, bool)
    point = {
        "corner": (0, 0),
        "edge": (height - 1, width // 2),
        "interior": (height // 2, width // 2),
    }[position]
    mask[point] = True
    mutated = image.copy()
    mutated[mask] = 255
    target = (lama_px // 8 * 8, int(lama_px * height / width) // 8 * 8)
    changed_rgb = np.any(
        np.array(Image.fromarray(mutated).resize(target, Image.LANCZOS)) != 0, axis=-1
    )
    area = cv2.resize(mask.astype(np.float32), target, interpolation=cv2.INTER_AREA) > 0
    old_support = cv2.dilate(area.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    assert (changed_rgb & ~old_support).any(), (
        "Fixture must expose the old fixed-radius upsample gap"
    )
    observed = []

    def stub(rgb, support):
        assert rgb.size == support.size == target
        assert not np.array(rgb).any(), "Masked source RGB must still be neutralized"
        observed.append(np.array(support) > 0)
        return Image.new("RGB", target, "white")

    result = inpaint_frame(mutated, mask, stub, lama_px=lama_px, mode="isolated")
    assert not (changed_rgb & ~observed[0]).any()
    assert observed[0].any() and not observed[0].all()
    np.testing.assert_array_equal(result[~mask], image[~mask])
    assert np.all(result[mask] == 255)


def test_standard_worker_downsample_keeps_three_target_pixel_growth():
    image = np.zeros((1080, 1920, 3), np.uint8)
    mask = np.zeros((1080, 1920), bool)
    mask[700, 1000] = True
    area = cv2.resize(mask.astype(np.float32), (960, 536), interpolation=cv2.INTER_AREA) > 0
    expected = cv2.dilate(area.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0

    def stub(rgb, support):
        np.testing.assert_array_equal(np.array(support) > 0, expected)
        return Image.new("RGB", rgb.size, "black")

    inpaint_frame(image, mask, stub, mode="isolated")


@pytest.mark.parametrize("mode", ["legacy", "isolated"])
def test_empty_mask_returns_independent_copy_and_never_calls_model(mode):
    image, mask = sample()
    mask[:] = False
    model = Mock(side_effect=AssertionError("must not call"))
    result = inpaint_frame(image, mask, model, lama_px=64, mode=mode)
    np.testing.assert_array_equal(result, image)
    assert not np.shares_memory(result, image)
    model.assert_not_called()


def test_empty_mask_does_not_require_loading_or_constructing_a_model():
    image = np.zeros((16, 16, 3), np.uint8)
    mask = np.zeros((16, 16), bool)
    np.testing.assert_array_equal(inpaint_frame(image, mask, None), image)


@pytest.mark.parametrize("mode", ["legacy", "isolated"])
def test_full_mask_and_multiple_passes(mode):
    image, mask = sample(60, 96)
    mask[:] = True
    seen = []

    def stub(rgb, support):
        seen.append(np.array(rgb))
        assert np.all(np.array(support) == 255)
        return Image.new("RGB", rgb.size, (12, 34, 56))

    result = inpaint_frame(image, mask, stub, lama_px=64, passes=2, mode=mode)
    assert len(seen) == 2
    if mode == "isolated":
        assert all(not pixels.any() for pixels in seen)
    np.testing.assert_array_equal(result, np.broadcast_to([12, 34, 56], image.shape))


def test_single_boundary_pixel_writes_only_that_pixel_at_original_resolution():
    image, mask = sample(30, 90)
    mask[:] = False
    mask[-1, -1] = True
    result = inpaint_frame(
        image,
        mask,
        lambda rgb, mask: Image.new("RGB", rgb.size, (1, 2, 3)),
        lama_px=48,
        mode="isolated",
    )
    np.testing.assert_array_equal(result[-1, -1], [1, 2, 3])
    np.testing.assert_array_equal(result[~mask], image[~mask])


@pytest.mark.parametrize(
    "bad",
    [
        "image-dtype",
        "image-shape",
        "image-empty",
        "mask-dtype",
        "mask-shape",
        "mode",
        "passes-zero",
        "passes-bool",
        "passes-float",
        "lama-zero",
        "lama-bool",
        "lama-float",
        "rounded-width-zero",
        "rounded-height-zero",
    ],
)
def test_invalid_inputs_are_rejected_before_model_call_even_for_empty_masks(bad):
    image, mask = sample()
    mask[:] = False
    kwargs = {"lama_px": 64, "passes": 1, "mode": "isolated"}
    if bad == "image-dtype":
        image = image.astype(np.float32)
    elif bad == "image-shape":
        image = image[:, :, 0]
    elif bad == "image-empty":
        image, mask = image[:0], mask[:0]
    elif bad == "mask-dtype":
        mask = mask.astype(np.uint8)
    elif bad == "mask-shape":
        mask = mask[:-1]
    elif bad == "mode":
        kwargs["mode"] = "unknown"
    elif bad == "passes-zero":
        kwargs["passes"] = 0
    elif bad == "passes-bool":
        kwargs["passes"] = True
    elif bad == "passes-float":
        kwargs["passes"] = 1.5
    elif bad == "lama-zero":
        kwargs["lama_px"] = 0
    elif bad == "lama-bool":
        kwargs["lama_px"] = True
    elif bad == "lama-float":
        kwargs["lama_px"] = 64.5
    elif bad == "rounded-width-zero":
        kwargs["lama_px"] = 7
    else:
        image, mask = image[:1], mask[:1]
    model = Mock(side_effect=AssertionError("must not call"))
    with pytest.raises(ValueError):
        inpaint_frame(image, mask, model, **kwargs)
    model.assert_not_called()


def test_noncallable_model_and_non_rgb_model_result_are_rejected():
    image, mask = sample()
    with pytest.raises(ValueError, match="callable"):
        inpaint_frame(image, mask, None)
    with pytest.raises(ValueError, match="PIL RGB"):
        inpaint_frame(image, mask, lambda rgb, mask: Image.new("L", rgb.size), lama_px=64)
