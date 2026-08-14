"""Generation-scoped Krea 2 depth ControlNet-LoRA adapter for Forge.

The upstream checkpoint stores an expanded first projection and rank-64 LoRA
pairs for every Krea block.  Forge already owns model loading, quantisation and
sampling, so this module only translates that checkpoint into a cloned
``UnetPatcher``.  The live base model is never modified permanently.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

CONTROL_MODEL_REPO = "Patil/Krea-2-depth-controlnet"
CONTROL_MODEL_FILENAME = "depth-control-lora.safetensors"
LORA_TARGETS = (
    "attn.wq",
    "attn.wk",
    "attn.wv",
    "attn.wo",
    "attn.gate",
    "mlp.gate",
    "mlp.up",
    "mlp.down",
)
EXPECTED_BLOCKS = 28


def control_model_path(models_path: str | os.PathLike[str]) -> Path:
    return Path(models_path) / "ControlNet" / "Krea2" / CONTROL_MODEL_FILENAME


@lru_cache(maxsize=2)
def _load_cached(path: str, modified_ns: int) -> Mapping[str, torch.Tensor]:
    del modified_ns
    from backend.utils import load_torch_file

    return load_torch_file(path, safe_load=True)


def load_control_state_dict(path: str | os.PathLike[str]) -> Mapping[str, torch.Tensor]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Krea 2 depth control model not found: {resolved}. "
            f"Download {CONTROL_MODEL_REPO}/{CONTROL_MODEL_FILENAME}."
        )
    stat = resolved.stat()
    return _load_cached(str(resolved), stat.st_mtime_ns)


def download_control_model(models_path: str | os.PathLike[str]) -> Path:
    """Download the official checkpoint without placing it in the Git repo."""

    destination = control_model_path(models_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=CONTROL_MODEL_REPO,
        filename=CONTROL_MODEL_FILENAME,
        local_dir=str(destination.parent),
    )
    result = Path(downloaded).resolve()
    load_control_state_dict(result)
    return result


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "tensor_shape", None)
    if shape is None:
        shape = getattr(value, "shape", None)
    return tuple(shape) if shape is not None else None


def _projection_parts(
    state_dict: Mapping[str, torch.Tensor], image_features: int, out_features: int
) -> torch.Tensor:
    weight = state_dict.get("first.weight")
    expected = (out_features, image_features * 2)
    if not torch.is_tensor(weight) or tuple(weight.shape) != expected:
        actual = None if weight is None else tuple(weight.shape)
        raise ValueError(
            f"Invalid Krea 2 control first.weight: expected {expected}, got {actual}."
        )
    return weight[:, image_features:].detach().to(device="cpu").contiguous()


class ControlProjection(nn.Module):
    """Adds the learned depth half to the active Krea input projection.

    Keeping the active projection for image tokens is important for quantised
    checkpoints and for any regular LoRA already selected in Forge.
    """

    def __init__(self, base_projection: nn.Module, control_weight: torch.Tensor):
        super().__init__()
        base_shape = _shape(getattr(base_projection, "weight", None))
        if base_shape is None or len(base_shape) != 2:
            raise ValueError("The active Krea input projection has no 2D weight.")
        if tuple(control_weight.shape) != (base_shape[0], base_shape[1]):
            raise ValueError(
                "Control projection shape does not match the active Krea model: "
                f"{tuple(control_weight.shape)} != {(base_shape[0], base_shape[1])}."
            )

        self.in_features = int(base_shape[1])
        self.out_features = int(base_shape[0])
        self.control_weight = nn.Parameter(control_weight, requires_grad=False)
        self.control_tokens: torch.Tensor | None = None
        object.__setattr__(self, "_base_projection", base_projection)

    @property
    def base_projection(self) -> nn.Module:
        return object.__getattribute__(self, "_base_projection")

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        if image_tokens.shape[-1] != self.in_features:
            raise RuntimeError(
                f"Krea image tokens have {image_tokens.shape[-1]} features; "
                f"expected {self.in_features}."
            )
        control = self.control_tokens
        if control is None:
            raise RuntimeError("Krea depth control tokens were not attached for this denoising step.")
        if control.shape[1] != image_tokens.shape[1]:
            raise RuntimeError(
                f"Krea token count mismatch: image={image_tokens.shape[1]}, "
                f"depth={control.shape[1]}."
            )
        control = _repeat_batch(control, image_tokens.shape[0]).to(
            device=image_tokens.device, dtype=image_tokens.dtype
        )
        from backend.memory_management import cast_to

        weight = cast_to(
            self.control_weight,
            device=image_tokens.device,
            dtype=image_tokens.dtype,
        )
        return self.base_projection(image_tokens) + F.linear(control, weight, None)


def _repeat_batch(tensor: torch.Tensor, batch: int) -> torch.Tensor:
    current = tensor.shape[0]
    if current == batch:
        return tensor
    if current <= 0 or batch % current != 0:
        raise RuntimeError(
            f"Cannot align Krea depth batch {current} with sampling batch {batch}."
        )
    repeats = [batch // current] + [1] * (tensor.ndim - 1)
    return tensor.repeat(*repeats)


def make_control_tokens(
    control_latent: torch.Tensor,
    sample: torch.Tensor,
    patch_size: int,
    expected_features: int,
) -> torch.Tensor:
    """Resize and patchify a Krea/Qwen latent for the current sampling call."""

    if control_latent.ndim == 5:
        if control_latent.shape[2] != 1:
            raise RuntimeError("Krea 2 depth control only supports one latent frame.")
        control_latent = control_latent[:, :, 0]
    if control_latent.ndim != 4:
        raise RuntimeError(
            f"Krea depth latent must be BCHW or BC1HW, got {tuple(control_latent.shape)}."
        )

    if sample.ndim == 5:
        if sample.shape[2] != 1:
            raise RuntimeError("Krea 2 image sampling expected a single temporal frame.")
        batch, _, _, height, width = sample.shape
    elif sample.ndim == 4:
        batch, _, height, width = sample.shape
    else:
        raise RuntimeError(f"Krea sample must be BCHW or BC1HW, got {tuple(sample.shape)}.")

    control = _repeat_batch(control_latent, batch).to(
        device=sample.device, dtype=sample.dtype
    )
    if control.shape[-2:] != (height, width):
        control = F.interpolate(
            control, size=(height, width), mode="bilinear", align_corners=False
        )

    pad_h = (-height) % patch_size
    pad_w = (-width) % patch_size
    if pad_h or pad_w:
        control = F.pad(control, (0, pad_w, 0, pad_h), mode="replicate")
    batch, channels, height, width = control.shape
    features = channels * patch_size * patch_size
    if features != expected_features:
        raise RuntimeError(
            f"Krea depth latent produces {features} features per token; "
            f"expected {expected_features}. Check that the Krea/Qwen image VAE is loaded."
        )

    control = control.reshape(
        batch,
        channels,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    return control.permute(0, 2, 4, 1, 3, 5).reshape(
        batch, (height // patch_size) * (width // patch_size), features
    )


def build_lora_patches(
    state_dict: Mapping[str, torch.Tensor], model_state: Mapping[str, Any]
) -> dict[str, Any]:
    """Translate the checkpoint's ``.A/.B`` pairs into Forge LoRA adapters."""

    from modules_forge.packages.comfy.weight_adapter.lora import LoRAAdapter

    patches: dict[str, Any] = {}
    errors: list[str] = []
    for block in range(EXPECTED_BLOCKS):
        for target in LORA_TARGETS:
            source = f"blocks.{block}.{target}"
            key_a, key_b = f"{source}.A", f"{source}.B"
            model_key = f"diffusion_model.{source}.weight"
            down, up = state_dict.get(key_a), state_dict.get(key_b)
            target_shape = _shape(model_state.get(model_key))
            if not torch.is_tensor(down) or not torch.is_tensor(up):
                errors.append(f"missing {key_a}/{key_b}")
                continue
            if down.ndim != 2 or up.ndim != 2 or target_shape is None:
                errors.append(f"invalid tensors for {source}")
                continue
            expected = (int(up.shape[0]), int(down.shape[1]))
            if tuple(target_shape) != expected or up.shape[1] != down.shape[0]:
                errors.append(
                    f"shape mismatch for {source}: LoRA {expected}, model {target_shape}"
                )
                continue
            rank = int(down.shape[0])
            patches[model_key] = LoRAAdapter(
                {key_a, key_b},
                (up, down, float(rank), None, None, None),
            )

    expected_count = EXPECTED_BLOCKS * len(LORA_TARGETS)
    if errors or len(patches) != expected_count:
        detail = "; ".join(errors[:4])
        raise ValueError(
            f"Incomplete or incompatible Krea 2 depth LoRA: {len(patches)}/"
            f"{expected_count} layers accepted. {detail}"
        )
    return patches


def _compose_control_wrapper(
    projection: ControlProjection,
    control_latent: torch.Tensor,
    patch_size: int,
    previous_wrapper: Any,
):
    def wrapper(model_function, call: dict[str, Any]):
        previous_tokens = projection.control_tokens
        try:
            projection.control_tokens = make_control_tokens(
                control_latent,
                call["input"],
                patch_size,
                projection.in_features,
            )
            if previous_wrapper is not None:
                return previous_wrapper(model_function, call)
            return model_function(call["input"], call["timestep"], **call["c"])
        finally:
            projection.control_tokens = previous_tokens

    return wrapper


def install_failure_guard(process, error: Exception) -> None:
    """Make sampling fail hard when Forge has swallowed a script-hook error."""

    message = f"Krea 2 Depth ControlNet-LoRA could not be applied: {error}"
    unet = process.sd_model.forge_objects.unet.clone()

    def fail_sampling(model_function, call):
        del model_function, call
        raise RuntimeError(message)

    unet.set_model_unet_function_wrapper(fail_sampling)
    process.sd_model.forge_objects.unet = unet


def apply_depth_control(
    unet,
    control_latent: torch.Tensor,
    state_dict: Mapping[str, torch.Tensor],
    strength: float,
):
    """Return a generation-local Forge UNet clone with depth control attached."""

    diffusion = unet.model.diffusion_model
    required = ("first", "blocks", "patch", "channels")
    if not all(hasattr(diffusion, name) for name in required):
        raise TypeError("Krea 2 Depth ControlNet-LoRA requires a native Krea 2 model.")
    base_first = unet.get_model_object("diffusion_model.first")
    base_shape = _shape(getattr(base_first, "weight", None))
    if base_shape is None or len(base_shape) != 2:
        raise TypeError("The selected Krea 2 model has an unsupported input projection.")
    if int(diffusion.channels) * int(diffusion.patch) ** 2 != int(base_shape[1]):
        raise TypeError("The selected model is not compatible with the Krea 2 depth checkpoint.")

    projection = ControlProjection(
        base_first,
        _projection_parts(state_dict, int(base_shape[1]), int(base_shape[0])),
    )
    new_unet = unet.clone()
    patches = build_lora_patches(state_dict, unet.model.state_dict())
    accepted = new_unet.add_patches(
        patches,
        strength_patch=float(strength),
        strength_model=1.0,
        filename=CONTROL_MODEL_FILENAME,
    )
    if len(accepted) != len(patches):
        raise ValueError(
            f"Forge accepted only {len(accepted)}/{len(patches)} Krea depth LoRA layers."
        )

    previous_wrapper = new_unet.model_options.get("model_function_wrapper")
    new_unet.add_object_patch("diffusion_model.first", projection)
    new_unet.set_model_unet_function_wrapper(
        _compose_control_wrapper(
            projection,
            control_latent.detach(),
            int(diffusion.patch),
            previous_wrapper,
        )
    )
    return new_unet
