"""Forge Classic integration for the Krea 2 depth ControlNet-LoRA."""

from .adapter import (
    CONTROL_MODEL_FILENAME,
    CONTROL_MODEL_REPO,
    ControlProjection,
    apply_depth_control,
    build_lora_patches,
    control_model_path,
    download_control_model,
    install_failure_guard,
    load_control_state_dict,
    make_control_tokens,
)

__all__ = [
    "CONTROL_MODEL_FILENAME",
    "CONTROL_MODEL_REPO",
    "ControlProjection",
    "apply_depth_control",
    "build_lora_patches",
    "control_model_path",
    "download_control_model",
    "install_failure_guard",
    "load_control_state_dict",
    "make_control_tokens",
]
