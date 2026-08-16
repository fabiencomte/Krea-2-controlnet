from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forge_krea2_depth.detection import (  # noqa: E402
    DEPTH_MAP,
    OPENPOSE_MAP,
    PHOTO,
    detect_control_kind,
)


def test_detects_coloured_openpose_lines_on_black():
    image = Image.new("RGB", (256, 256), "black")
    drawing = ImageDraw.Draw(image)
    points = [(128, 25), (128, 70), (80, 120), (45, 180), (176, 120), (211, 180)]
    colours = ["red", "lime", "blue", "yellow", "magenta"]
    for start, end, colour in zip(points, points[1:], colours):
        drawing.line((start, end), fill=colour, width=5)

    detection = detect_control_kind(image)

    assert detection.kind == OPENPOSE_MAP
    assert detection.confidence >= 0.75


def test_keeps_monochrome_lines_and_blank_canvas_conservative():
    image = Image.new("RGB", (256, 256), "black")
    drawing = ImageDraw.Draw(image)
    drawing.line((128, 20, 128, 100, 60, 220), fill="white", width=5)
    drawing.line((128, 100, 210, 220), fill="white", width=5)

    assert detect_control_kind(image).kind == PHOTO
    assert detect_control_kind(np.zeros((256, 256, 3), dtype=np.uint8)).kind == PHOTO


def test_detects_smooth_grayscale_depth_map():
    horizontal = np.linspace(12, 244, 256, dtype=np.uint8)
    depth = np.repeat(horizontal[None, :], 192, axis=0)

    detection = detect_control_kind(depth)

    assert detection.kind == DEPTH_MAP
    assert detection.confidence >= 0.84


def test_does_not_treat_colour_or_textured_grayscale_image_as_control_map():
    y, x = np.mgrid[:192, :256]
    colour = np.stack((x % 256, y % 256, (x + y) % 256), axis=2).astype(np.uint8)
    checker = (((x // 4 + y // 4) % 2) * 255).astype(np.uint8)

    assert detect_control_kind(colour).kind == PHOTO
    assert detect_control_kind(checker).kind == PHOTO
