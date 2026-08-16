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
    alternating_image_indices,
    create_depth_map,
    fit_control_map,
    generation_dimensions,
    normalize_image,
    normalize_images,
    preview_dimensions,
    ratio_warning,
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


def test_normalize_accepts_an_existing_filepath(tmp_path):
    path = tmp_path / "control.png"
    Image.new("RGB", (3, 2), (11, 22, 33)).save(path)

    image = normalize_image(str(path))

    assert image.shape == (2, 3, 3)
    assert image[0, 0].tolist() == [11, 22, 33]


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


def test_normalize_gallery_accepts_different_sizes_and_captions():
    first = np.zeros((2, 4, 3), dtype=np.uint8)
    second = np.ones((3, 2, 3), dtype=np.uint8)

    images = normalize_images([(first, "wide"), (second, None)])

    assert [image.shape for image in images] == [(2, 4, 3), (3, 2, 3)]


def test_normalize_gallery_rejects_an_empty_selection():
    with pytest.raises(ValueError, match="at least one control image"):
        normalize_images([])


def test_fit_control_map_uses_exact_generation_dimensions():
    source = np.zeros((30, 60, 3), dtype=np.uint8)
    assert fit_control_map(source, 96, 64).shape == (64, 96, 3)


def test_fit_control_map_resizes_without_cropping_the_left_and_right_edges():
    source = np.zeros((2, 6, 3), dtype=np.uint8)
    source[:, :2] = (255, 0, 0)
    source[:, 2:4] = (0, 255, 0)
    source[:, 4:] = (0, 0, 255)

    fitted = fit_control_map(source, 3, 3)

    assert np.all(fitted[0] == 0)
    assert np.all(fitted[-1] == 0)
    assert fitted[1, 0, 0] > fitted[1, 0, 1]
    assert fitted[1, 0, 0] > fitted[1, 0, 2]
    assert fitted[1, -1, 2] > fitted[1, -1, 0]
    assert fitted[1, -1, 2] > fitted[1, -1, 1]


def test_fit_control_map_adds_centered_side_bars_for_a_tall_source():
    source = np.full((6, 2, 3), 200, dtype=np.uint8)

    fitted = fit_control_map(source, 6, 6)

    assert np.all(fitted[:, :2] == 0)
    assert np.all(fitted[:, 4:] == 0)
    assert np.all(fitted[:, 2:4] == 200)


def test_ratio_difference_warns_but_matching_ratio_does_not():
    wide = np.zeros((4, 8, 3), dtype=np.uint8)
    square = np.zeros((4, 4, 3), dtype=np.uint8)

    assert ratio_warning([wide], 1024, 512) == ""
    warning = ratio_warning([wide, square], 1024, 512)
    assert "#2 (4×4" in warning
    assert "without cropping" in warning
    assert "black bars" in warning


def test_preview_dimensions_follow_forge_hires_target_rules():
    assert preview_dimensions(512, 768) == (512, 768)
    assert preview_dimensions(512, 768, True, 1.5, 0, 0, 64) == (768, 1152)
    assert preview_dimensions(512, 768, True, 1.0, 1024, 0, 64) == (
        1024,
        1536,
    )
    assert preview_dimensions(512, 768, True, 1.0, 0, 1024, 64) == (704, 1024)


def test_alternating_indices_continue_across_batch_iterations():
    assert alternating_image_indices(3, 2, 0) == [0, 1]
    assert alternating_image_indices(3, 2, 1) == [2, 0]
    assert alternating_image_indices(3, 2, 2) == [1, 2]


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


def test_inversion_keeps_letterbox_bars_black(monkeypatch):
    fake_shared = types.ModuleType("modules_forge.shared")
    fake_shared.supported_preprocessors = {}
    monkeypatch.setitem(sys.modules, "modules_forge.shared", fake_shared)

    result = create_depth_map(
        np.full((2, 4, 3), 10, dtype=np.uint8),
        "None (already a depth map)",
        768,
        4,
        4,
        True,
    )

    assert np.all(result[0] == 0)
    assert np.all(result[-1] == 0)
    assert np.all(result[1:3] == 245)


def test_unknown_preprocessor_is_rejected(monkeypatch):
    fake_shared = types.ModuleType("modules_forge.shared")
    fake_shared.supported_preprocessors = {}
    monkeypatch.setitem(sys.modules, "modules_forge.shared", fake_shared)

    with pytest.raises(ValueError, match="preprocessor is unavailable"):
        create_depth_map(
            np.zeros((2, 2, 3), dtype=np.uint8), "missing", 768, 2, 2, False
        )
