"""Deterministic presentation-only image operations for the demo matrix."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


TARGET_DEPTH_BACKGROUND = np.array([163, 160, 146], dtype=np.float32)


def normalize_depth_background(image: Image.Image, character: str) -> Image.Image:
    """Place the generated subject on the shared demo background.

    This changes only the presentation background. The raw Forge PNG remains
    untouched in outputs/raw and is the source of truth for API evidence.
    """

    array = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
    saturation_threshold = 65 if character == "Black Widow" else 38
    value_threshold = 110 if character == "Black Widow" else 118
    seed = (
        (hsv[:, :, 1] > saturation_threshold) | (hsv[:, :, 2] < value_threshold)
    ).astype(np.uint8) * 255
    seed = cv2.morphologyEx(
        seed,
        cv2.MORPH_CLOSE,
        np.ones((13, 13), dtype=np.uint8),
    )
    contours, _ = cv2.findContours(
        seed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        raise RuntimeError(f"Could not isolate Depth subject: {character}")
    subject = np.zeros(seed.shape, dtype=np.uint8)
    cv2.drawContours(
        subject,
        [max(contours, key=cv2.contourArea)],
        -1,
        255,
        cv2.FILLED,
    )
    alpha = cv2.GaussianBlur(
        subject.astype(np.float32) / 255.0,
        (0, 0),
        1.8,
    )[:, :, None]
    result = np.clip(
        array.astype(np.float32) * alpha + TARGET_DEPTH_BACKGROUND * (1.0 - alpha),
        0,
        255,
    ).astype(np.uint8)
    return Image.fromarray(result, mode="RGB")
