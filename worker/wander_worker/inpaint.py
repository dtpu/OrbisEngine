"""Deterministic frame preparation/compositing around an injected inpainter.

The default preserves the existing worker. ``isolated`` is an experimental signal
isolation mode, not a claim that a mask or generated fill meets visual quality gates.
No model, weights, device or provider is selected or loaded here.
"""

from collections.abc import Callable
import math

import numpy as np
from PIL import Image


def inpaint_frame(
    image: np.ndarray,
    mask: np.ndarray,
    model: Callable[[Image.Image, Image.Image], Image.Image] | None,
    *,
    lama_px: int = 960,
    passes: int = 1,
    mode: str = "legacy",
) -> np.ndarray:
    """Return RGB uint8 pixels without modifying the supplied image or boolean mask.

    Isolated mode neutralizes removal pixels before RGB filtering, covers the
    low-resolution filter support, and writes only inside the approved full-resolution
    mask. Mask semantics and the quality of the callable's fill remain the caller's job.
    """
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] == 0
        or image.shape[1] == 0
    ):
        raise ValueError("Image must be a nonempty uint8 HxWx3 RGB array")
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != image.shape[:2]:
        raise ValueError("Mask must be a boolean HxW array matching the image")
    if mode not in ("legacy", "isolated"):
        raise ValueError("Inpainting mode must be legacy or isolated")
    if type(passes) is not int or passes <= 0:
        raise ValueError("Passes must be a positive integer")
    if type(lama_px) is not int or lama_px <= 0:
        raise ValueError("LaMa size must be a positive integer")
    height, width = image.shape[:2]
    try:
        sw = lama_px // 8 * 8
        sh = int(lama_px * height / width) // 8 * 8
    except OverflowError as error:
        raise ValueError("LaMa target dimensions are not representable") from error
    if sw <= 0 or sh <= 0:
        raise ValueError("LaMa target dimensions must both be positive after rounding to eight")

    im = image.copy()
    if not mask.any():
        return im
    if not callable(model):
        raise ValueError("Model must be callable")

    import cv2

    if mode == "isolated":
        # Area coverage retains subpixel/thin support that nearest sampling can lose.
        small_mask = cv2.resize(mask.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA) > 0
        # During downsampling LANCZOS has three target pixels of support. During
        # upsampling its three source pixels cover more target pixels on each axis.
        rx = math.ceil(3 * max(1, sw / width))
        ry = math.ceil(3 * max(1, sh / height))
        small_mask = (
            cv2.dilate(small_mask.astype(np.uint8), np.ones((2 * ry + 1, 2 * rx + 1), np.uint8)) > 0
        )

    for _ in range(passes):
        if mode == "legacy":
            small = Image.fromarray(im).resize((sw, sh), Image.LANCZOS)
            sm = Image.fromarray(mask.astype(np.uint8) * 255).resize((sw, sh), Image.NEAREST)
        else:
            # Reapply on every pass: even the next model input must not depend on the
            # previous fill or original appearance inside the approved removal mask.
            neutral = im.copy()
            neutral[mask] = 0
            small = Image.fromarray(neutral).resize((sw, sh), Image.LANCZOS)
            sm = Image.fromarray(small_mask.astype(np.uint8) * 255)
        result = model(small, sm)
        if not isinstance(result, Image.Image) or result.mode != "RGB":
            raise ValueError("Model must return a PIL RGB image")
        fill = np.asarray(result.resize((width, height), Image.BICUBIC))
        if mode == "legacy":
            alpha = np.clip(
                cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3)[..., None] * 1.5, 0, 1
            )
            im = (im * (1 - alpha) + fill * alpha).astype(np.uint8)
        else:
            # No original-image contribution within the mask, no writes outside it.
            im[mask] = fill[mask]
    return im
