"""Generation-scoped Krea 2 depth ControlNet-LoRA adapter for Forge.

The upstream checkpoint stores an expanded first projection and rank-64 LoRA
pairs for every Krea block.  Forge already owns model loading, quantisation and
sampling, so this module only translates that checkpoint into a cloned
``UnetPatcher``.  The live base model is never modified permanently.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

CONTROL_MODEL_REPO = "Patil/Krea-2-depth-controlnet"
CONTROL_MODEL_FILENAME = "depth-control-lora.safetensors"
CONTROL_MODEL_REVISION = "2e3ed7331854063f71853ab7213d2a2837a2be08"
CONTROL_MODEL_SHA256 = "fb80547ed79b47c1e3fea7bb9d36297e3917b2115fab6700ca1501350f9f483c"
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
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
        revision=CONTROL_MODEL_REVISION,
        local_dir=str(destination.parent),
    )
    result = Path(downloaded).resolve()
    digest = _sha256(result)
    if digest != CONTROL_MODEL_SHA256:
        raise ValueError(
            f"Invalid {CONTROL_MODEL_FILENAME} SHA-256: {digest}; "
            f"expected {CONTROL_MODEL_SHA256}."
        )
    _load_cached.cache_clear()
    load_control_state_dict(result)
    return result


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "tensor_shape", None)
    if shape is None:
        shape = getattr(value, "shape", None)
    return tuple(shape) if shape is not None else None


def _projection_tensors(
    state_dict: Mapping[str, torch.Tensor], image_features: int, out_features: int
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = state_dict.get("first.weight")
    bias = state_dict.get("first.bias")
    expected = (out_features, image_features * 2)
    if not torch.is_tensor(weight) or tuple(weight.shape) != expected:
        actual = None if weight is None else tuple(weight.shape)
        raise ValueError(
            f"Invalid Krea 2 control first.weight: expected {expected}, got {actual}."
        )
    expected_bias = (out_features,)
    if not torch.is_tensor(bias) or tuple(bias.shape) != expected_bias:
        actual = None if bias is None else tuple(bias.shape)
        raise ValueError(
            f"Invalid Krea 2 control first.bias: expected {expected_bias}, got {actual}."
        )
    return (
        weight.detach().to(device="cpu").contiguous(),
        bias.detach().to(device="cpu").contiguous(),
    )


class ControlProjection:
    """Official expanded Krea projection without replacing the live Forge layer.

    The tensors intentionally remain plain CPU tensors.  A temporary forward
    hook casts them for each model call, while Forge retains ownership of the
    real ``diffusion_model.first`` module and its low-VRAM lifecycle.
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, image_features: int):
        self.weight = weight
        self.bias = bias
        self.in_features = int(image_features)

    def __call__(
        self, image_tokens: torch.Tensor, control_tokens: torch.Tensor
    ) -> torch.Tensor:
        if image_tokens.shape[-1] != self.in_features:
            raise RuntimeError(
                f"Krea image tokens have {image_tokens.shape[-1]} features; "
                f"expected {self.in_features}."
            )
        if control_tokens.shape[1] != image_tokens.shape[1]:
            raise RuntimeError(
                f"Krea token count mismatch: image={image_tokens.shape[1]}, "
                f"depth={control_tokens.shape[1]}."
            )
        control_tokens = _repeat_batch(control_tokens, image_tokens.shape[0]).to(
            device=image_tokens.device, dtype=image_tokens.dtype
        )
        from backend.memory_management import cast_to

        weight = cast_to(
            self.weight, device=image_tokens.device, dtype=image_tokens.dtype
        )
        bias = cast_to(
            self.bias, device=image_tokens.device, dtype=image_tokens.dtype
        )
        combined = torch.cat((image_tokens, control_tokens), dim=-1)
        return F.linear(combined, weight, bias)


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
    first_module: Any,
    projection: ControlProjection,
    control_latent: torch.Tensor,
    patch_size: int,
    previous_wrapper: Any,
):
    def wrapper(model_function, call: dict[str, Any]):
        def controlled_model_function(input_, timestep, **conditioning):
            control_tokens = make_control_tokens(
                control_latent,
                input_,
                patch_size,
                projection.in_features,
            )
            used = False

            def project_first(_module, args, output):
                nonlocal used
                if used:
                    return output
                used = True
                return projection(args[0], control_tokens)

            handle = first_module.register_forward_hook(project_first)
            try:
                return model_function(input_, timestep, **conditioning)
            finally:
                handle.remove()

        if previous_wrapper is not None:
            return previous_wrapper(controlled_model_function, call)
        return controlled_model_function(
            call["input"], call["timestep"], **call["c"]
        )

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
        *_projection_tensors(state_dict, int(base_shape[1]), int(base_shape[0])),
        image_features=int(base_shape[1]),
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
    new_unet.set_model_unet_function_wrapper(
        _compose_control_wrapper(
            base_first,
            projection,
            control_latent.detach(),
            int(diffusion.patch),
            previous_wrapper,
        )
    )
    return new_unet
