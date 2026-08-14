"""Image and Forge preprocessor helpers kept separate for unit testing."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image, ImageOps


def normalize_image(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("background", value.get("image"))
    if isinstance(value, str):
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


def fit_control_map(image: np.ndarray, width: int, height: int) -> np.ndarray:
    fitted = ImageOps.fit(
        Image.fromarray(normalize_image(image), mode="RGB"),
        (int(width), int(height)),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    return np.asarray(fitted)


def create_depth_map(
    image: Any,
    preprocessor_name: str,
    resolution: int,
    width: int,
    height: int,
    invert: bool = False,
) -> np.ndarray:
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
    result = fit_control_map(source, width, height)
    if invert:
        result = 255 - result
    return np.ascontiguousarray(result)
