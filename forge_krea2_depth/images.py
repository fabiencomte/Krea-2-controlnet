"""Image and Forge preprocessor helpers kept separate for unit testing."""

from __future__ import annotations

import base64
import io
import math
import os
from typing import Any

import numpy as np
from PIL import Image, ImageOps


def normalize_image(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("background", value.get("image"))
    if isinstance(value, str):
        if os.path.isfile(value):
            try:
                value = Image.open(value)
            except Exception as exc:
                raise ValueError(f"The control image file cannot be read: {value}.") from exc
        else:
            encoded = value.split(",", 1)[1] if value.startswith("data:image/") else value
            try:
                value = Image.open(io.BytesIO(base64.b64decode(encoded, validate=True)))
            except Exception as exc:
                raise ValueError("The API control image is not valid base64 image data.") from exc
    if isinstance(value, Image.Image):
        value = np.asarray(value.convert("RGB"))
    if value is None:
        raise ValueError("Choose a control image before generating.")
    image = np.asarray(value)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 1:
        raise ValueError(f"Unsupported control image shape: {image.shape}.")
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    image = image[:, :, :3]
    if np.issubdtype(image.dtype, np.floating):
        maximum = float(np.nanmax(image)) if image.size else 0.0
        if maximum <= 1.0:
            image = image * 255.0
    return np.clip(np.nan_to_num(image), 0, 255).astype(np.uint8)


def normalize_images(value: Any) -> list[np.ndarray]:
    """Normalize a legacy single image or a Gradio/API image gallery."""

    if isinstance(value, list):
        try:
            array = np.asarray(value)
        except ValueError:
            array = None
        if (
            array is not None
            and array.ndim in (2, 3)
            and np.issubdtype(array.dtype, np.number)
        ):
            items = [array]
        else:
            items = value
    else:
        items = [value]

    images = []
    for item in items:
        if isinstance(item, tuple) and len(item) == 2:
            item = item[0]
        elif isinstance(item, dict) and "image" in item and "background" not in item:
            item = item["image"]
        images.append(normalize_image(item))
    if not images:
        raise ValueError("Choose at least one control image before generating.")
    return images


def fit_control_map(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Letterbox a control map to the sampling canvas without crop or distortion."""

    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("Krea 2 Control target dimensions must be positive.")
    fitted = ImageOps.pad(
        Image.fromarray(normalize_image(image), mode="RGB"),
        (width, height),
        method=Image.Resampling.LANCZOS,
        color=(0, 0, 0),
        centering=(0.5, 0.5),
    )
    return np.asarray(fitted)


def ratio_warning(images: Any, width: int, height: int) -> str:
    """Return a non-blocking warning when a source ratio differs from the target."""

    sources = normalize_images(images)
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("Krea 2 Control target dimensions must be positive.")
    mismatches = []
    for index, source in enumerate(sources):
        source_height, source_width = source.shape[:2]
        if source_width * height != source_height * width:
            mismatches.append(
                f"#{index + 1} ({source_width}×{source_height}, "
                f"{source_width / source_height:.3f}:1)"
            )
    if not mismatches:
        return ""
    return (
        "Aspect ratio differs for source "
        + ("image " if len(mismatches) == 1 else "images ")
        + ", ".join(mismatches)
        + f" versus target {width}×{height} ({width / height:.3f}:1). "
        "Control maps will be fitted to the exact target without cropping or "
        "distortion; black bars will fill the unused area."
    )


def alternating_image_indices(
    image_count: int, batch_size: int, iteration: int
) -> list[int]:
    """Select A, B, C, A... across both batch positions and iterations."""

    image_count, batch_size, iteration = (
        int(image_count),
        int(batch_size),
        int(iteration),
    )
    if image_count <= 0:
        raise ValueError("Choose at least one control image before generating.")
    if batch_size <= 0 or iteration < 0:
        raise ValueError("Invalid Forge batch position for Krea 2 control.")
    start = iteration * batch_size
    return [(start + offset) % image_count for offset in range(batch_size)]


def preview_dimensions(
    width: int,
    height: int,
    enable_hr: bool = False,
    hr_scale: float = 1.0,
    hr_resize_x: int = 0,
    hr_resize_y: int = 0,
    resolution_step: int = 8,
) -> tuple[int, int]:
    """Mirror Forge's requested final canvas dimensions for depth previews."""

    width, height = int(width), int(height)
    if not enable_hr:
        return width, height

    resize_x, resize_y = int(hr_resize_x or 0), int(hr_resize_y or 0)
    if resize_x == 0 and resize_y == 0:
        target_width = width * float(hr_scale)
        target_height = height * float(hr_scale)
    elif resize_y == 0:
        target_width = resize_x
        target_height = resize_x * height / width
    elif resize_x == 0:
        target_width = resize_y * width / height
        target_height = resize_y
    else:
        target_width, target_height = resize_x, resize_y

    step = max(1, int(resolution_step))

    def rounded(value: float) -> int:
        return math.floor(value / step + 0.5) * step

    return rounded(target_width), rounded(target_height)


def generation_dimensions(process: Any) -> tuple[int, int]:
    """Return the pixel dimensions of the sampling pass currently being built."""

    if bool(getattr(process, "is_hr_pass", False)):
        width = int(getattr(process, "hr_upscale_to_x", 0) or 0)
        height = int(getattr(process, "hr_upscale_to_y", 0) or 0)
        if width > 0 and height > 0:
            return width, height
    return int(process.width), int(process.height)


def preprocess_depth_map(
    image: Any,
    preprocessor_name: str,
    resolution: int,
    invert: bool = False,
) -> np.ndarray:
    """Run the expensive per-source preprocessor without target letterboxing."""

    source = normalize_image(image)
    if preprocessor_name and preprocessor_name != "None (already a depth map)":
        from modules_forge.shared import supported_preprocessors

        preprocessor = supported_preprocessors.get(preprocessor_name)
        if preprocessor is None:
            raise ValueError(f"Forge preprocessor is unavailable: {preprocessor_name}.")
        source = normalize_image(
            preprocessor(
                input_image=source,
                resolution=int(resolution),
                slider_1=None,
                slider_2=None,
            )
        )
    if invert:
        source = 255 - source
    return np.ascontiguousarray(source)


def create_depth_map(
    image: Any,
    preprocessor_name: str,
    resolution: int,
    width: int,
    height: int,
    invert: bool = False,
) -> np.ndarray:
    source = preprocess_depth_map(image, preprocessor_name, resolution, invert)
    return np.ascontiguousarray(fit_control_map(source, width, height))
