"""Conservative colour-based control-source detection for the Forge UI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .images import normalize_image


PHOTO = "photo"
DEPTH_MAP = "depth"
OPENPOSE_MAP = "openpose"


@dataclass(frozen=True)
class ControlDetection:
    kind: str
    confidence: float
    reason: str


def _thumbnail(image) -> np.ndarray:
    source = Image.fromarray(normalize_image(image), mode="RGB")
    source.thumbnail((256, 256), Image.Resampling.BILINEAR)
    return np.asarray(source, dtype=np.float32)


def detect_control_kind(image) -> ControlDetection:
    """Guess photo, grayscale depth, or sparse OpenPose without hiding doubt."""

    rgb = _thumbnail(image)
    highest = rgb.max(axis=2)
    lowest = rgb.min(axis=2)
    spread = highest - lowest
    luminance = rgb.mean(axis=2)
    saturation = spread / np.maximum(highest, 1.0)

    dark_fraction = float(np.mean(highest < 32))
    lit_fraction = float(np.mean(highest > 64))
    coloured = (saturation > 0.42) & (highest > 72)
    coloured_fraction = float(np.mean(coloured))
    gray_fraction = float(np.mean(spread < 9))

    grad_x = np.abs(np.diff(luminance, axis=1))
    grad_y = np.abs(np.diff(luminance, axis=0))
    gradients = np.concatenate((grad_x.ravel(), grad_y.ravel()))
    edge_fraction = float(np.mean(gradients > 22)) if gradients.size else 0.0
    smooth_fraction = float(np.mean(gradients < 7)) if gradients.size else 1.0

    hue_bins = 0
    if np.any(coloured):
        selected = rgb[coloured]
        maximum = selected.max(axis=1)
        minimum = selected.min(axis=1)
        delta = maximum - minimum
        hue = np.zeros(len(selected), dtype=np.float32)
        nonzero = delta > 0
        red = nonzero & (selected[:, 0] == maximum)
        green = nonzero & (selected[:, 1] == maximum)
        blue = nonzero & (selected[:, 2] == maximum)
        hue[red] = ((selected[red, 1] - selected[red, 2]) / delta[red]) % 6
        hue[green] = (selected[green, 2] - selected[green, 0]) / delta[green] + 2
        hue[blue] = (selected[blue, 0] - selected[blue, 1]) / delta[blue] + 4
        hue_bins = int(np.count_nonzero(np.histogram(hue / 6, bins=12, range=(0, 1))[0]))

    sparse_on_black = dark_fraction > 0.67 and 0.002 < lit_fraction < 0.34
    coloured_skeleton = coloured_fraction > 0.0015 and hue_bins >= 3
    monochrome_skeleton = gray_fraction > 0.93 and edge_fraction > 0.018
    if sparse_on_black and (coloured_skeleton or monochrome_skeleton):
        confidence = min(
            0.98,
            0.70
            + min(0.14, dark_fraction * 0.12)
            + min(0.10, hue_bins * 0.015)
            + min(0.08, edge_fraction),
        )
        return ControlDetection(
            OPENPOSE_MAP,
            confidence,
            "sparse coloured/bright lines on a predominantly black canvas",
        )

    luminance_std = float(luminance.std())
    tone_count = int(np.unique((luminance / 4).astype(np.uint8)).size)
    depth_like = (
        gray_fraction > 0.92
        and luminance_std > 7
        and tone_count >= 12
        and smooth_fraction > 0.55
        and not (dark_fraction > 0.96 or lit_fraction < 0.01)
    )
    if depth_like:
        confidence = min(
            0.96,
            0.62
            + (gray_fraction - 0.92) * 1.8
            + min(0.12, smooth_fraction * 0.12)
            + min(0.08, tone_count / 512),
        )
        return ControlDetection(
            DEPTH_MAP,
            confidence,
            "near-grayscale continuous tones with predominantly smooth gradients",
        )

    colour_evidence = min(0.22, (1.0 - gray_fraction) * 0.35)
    texture_evidence = min(0.12, edge_fraction * 0.8)
    return ControlDetection(
        PHOTO,
        min(0.94, 0.58 + colour_evidence + texture_evidence),
        "ordinary image statistics; no reliable control-map signature",
    )
