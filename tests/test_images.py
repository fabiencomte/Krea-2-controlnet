from __future__ import annotations

import base64
import io
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forge_krea2_depth.images import (  # noqa: E402
    create_depth_map,
    fit_control_map,
    generation_dimensions,
    normalize_image,
)


def test_normalize_accepts_grayscale_float_and_editor_dict():
    source = np.array([[0.0, 0.5], [1.0, np.nan]], dtype=np.float32)
    image = normalize_image({"background": source})
    assert image.shape == (2, 2, 3)
    assert image.dtype == np.uint8
    assert image[1, 0].tolist() == [255, 255, 255]


def test_normalize_accepts_pil_rgba():
    image = normalize_image(Image.new("RGBA", (3, 2), (10, 20, 30, 40)))
    assert image.shape == (2, 3, 3)
    assert image[0, 0].tolist() == [10, 20, 30]


def test_normalize_accepts_api_data_url():
    buffer = io.BytesIO()
    Image.new("RGB", (2, 1), (7, 8, 9)).save(buffer, format="PNG")
    value = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    image = normalize_image(value)
    assert image.shape == (1, 2, 3)
    assert image[0, 0].tolist() == [7, 8, 9]


def test_normalize_rejects_missing_image():
    with pytest.raises(ValueError, match="Choose a control image"):
        normalize_image(None)


def test_fit_control_map_uses_exact_generation_dimensions():
    source = np.zeros((30, 60, 3), dtype=np.uint8)
    assert fit_control_map(source, 96, 64).shape == (64, 96, 3)


def test_hires_pass_uses_its_own_dimensions_and_aspect_ratio():
    class Process:
        width = 512
        height = 512
        is_hr_pass = True
        hr_upscale_to_x = 768
        hr_upscale_to_y = 512

    assert generation_dimensions(Process()) == (768, 512)


def test_first_pass_ignores_future_hires_dimensions():
    class Process:
        width = 512
        height = 512
        is_hr_pass = False
        hr_upscale_to_x = 768
        hr_upscale_to_y = 512

    assert generation_dimensions(Process()) == (512, 512)


def test_none_preprocessor_uses_supplied_depth_map(monkeypatch):
    fake_shared = types.ModuleType("modules_forge.shared")
    fake_shared.supported_preprocessors = {
        "depth_anything_v2": lambda **_kwargs: pytest.fail("preprocessor was called")
    }
    monkeypatch.setitem(sys.modules, "modules_forge.shared", fake_shared)
    source = np.full((4, 8, 3), 31, dtype=np.uint8)

    result = create_depth_map(
        source, "None (already a depth map)", 768, 8, 4, False
    )

    assert result.shape == (4, 8, 3)
    assert np.all(result == 31)


def test_depth_preprocessor_receives_resolution_and_can_invert(monkeypatch):
    calls = []

    def preprocessor(**kwargs):
        calls.append(kwargs)
        return np.full((2, 4, 3), 10, dtype=np.uint8)

    fake_shared = types.ModuleType("modules_forge.shared")
    fake_shared.supported_preprocessors = {"depth_anything_v2": preprocessor}
    monkeypatch.setitem(sys.modules, "modules_forge.shared", fake_shared)

    result = create_depth_map(
        np.zeros((3, 5, 3), dtype=np.uint8),
        "depth_anything_v2",
        1024,
        4,
        2,
        True,
    )

    assert calls[0]["resolution"] == 1024
    assert calls[0]["slider_1"] is None
    assert calls[0]["slider_2"] is None
    assert np.all(result == 245)


def test_unknown_preprocessor_is_rejected(monkeypatch):
    fake_shared = types.ModuleType("modules_forge.shared")
    fake_shared.supported_preprocessors = {}
    monkeypatch.setitem(sys.modules, "modules_forge.shared", fake_shared)

    with pytest.raises(ValueError, match="preprocessor is unavailable"):
        create_depth_map(
            np.zeros((2, 2, 3), dtype=np.uint8), "missing", 768, 2, 2, False
        )
