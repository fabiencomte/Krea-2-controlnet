"""Krea 2 Pose reference conditioning and verified DWPose preprocessing.

The pose adapter was trained with ai-toolkit's Krea 2 edit/reference path, not
with the expanded depth input projection.  Pose references are consequently
appended as clean (t=0) image tokens and the adapter is applied as an ordinary
Krea LoRA on a generation-local Forge ``UnetPatcher`` clone.

The index-timestep-zero reference path is adapted from Ostris' MIT-licensed
ComfyUI-Krea2-Ostris-Edit. See ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import hashlib
import math
import os
from copy import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from forge_krea2_depth.images import fit_control_map, normalize_image


POSE_MODEL_REPO = "thedeoxen/Krea-2-pose-controlnet"
POSE_MODEL_FILENAME = "krea2_turbo_openpose_controlnet.safetensors"
POSE_MODEL_REVISION = "517163d935a10c100095b47e443269cb766a157d"
POSE_MODEL_SHA256 = "0ddc3aafce4abdf7af3309b2f00c1bacdf15df1f2b4fb7adc9ff71795da90ecf"
POSE_MODEL_SIZE = 228587504
POSE_EXPECTED_LAYERS = 256

DWPOSE_REPO = "RedHash/DWPose"
DWPOSE_REVISION = "c9a7bafe8eba39dd6e6ca243fe4a3ff7b194fc2a"
DWPOSE_FILES = {
    "dw-ll_ucoco_384.onnx": (
        134399116,
        "724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843",
    ),
    "yolox_l.onnx": (
        216746733,
        "7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411",
    ),
}

VLM_MAX_PIXELS = 384 * 384
REF_LATENT_MAX_PIXELS = 1024 * 1024
REF_SNAP = 16


def pose_model_path(models_path: str | os.PathLike[str]) -> Path:
    return Path(models_path) / "ControlNet" / "Krea2" / POSE_MODEL_FILENAME


def dwpose_model_dir(models_path: str | os.PathLike[str]) -> Path:
    return Path(models_path) / "ControlNet" / "DWPose"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _verify_cached(
    path: str, modified_ns: int, size: int, expected_size: int, expected_sha256: str
) -> None:
    del modified_ns
    if size != expected_size:
        raise ValueError(
            f"Invalid model size for {Path(path).name}: {size}; expected {expected_size}."
        )
    digest = _sha256(Path(path))
    if digest != expected_sha256:
        raise ValueError(
            f"Invalid {Path(path).name} SHA-256: {digest}; expected {expected_sha256}."
        )


def _verify(path: Path, expected_size: int, expected_sha256: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Krea 2 Pose model not found: {path.resolve()}.")
    stat = path.stat()
    _verify_cached(
        str(path.resolve()),
        stat.st_mtime_ns,
        stat.st_size,
        expected_size,
        expected_sha256,
    )
    return path.resolve()


@lru_cache(maxsize=1)
def _load_pose_cached(
    path: str, modified_ns: int, size: int
) -> Mapping[str, torch.Tensor]:
    del modified_ns, size
    from safetensors.torch import load_file

    return load_file(path, device="cpu")


def load_pose_state_dict(
    path: str | os.PathLike[str],
) -> Mapping[str, torch.Tensor]:
    resolved = Path(path).resolve()
    _verify(resolved, POSE_MODEL_SIZE, POSE_MODEL_SHA256)
    stat = resolved.stat()
    return _load_pose_cached(str(resolved), stat.st_mtime_ns, stat.st_size)


def _download_verified(
    repo_id: str,
    filename: str,
    revision: str,
    destination: Path,
    expected_size: int,
    expected_sha256: str,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        try:
            return _verify(destination, expected_size, expected_sha256)
        except ValueError:
            pass

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=str(destination.parent),
        force_download=destination.is_file(),
    )
    _verify_cached.cache_clear()
    return _verify(Path(downloaded), expected_size, expected_sha256)


def download_pose_model(models_path: str | os.PathLike[str]) -> Path:
    result = _download_verified(
        POSE_MODEL_REPO,
        POSE_MODEL_FILENAME,
        POSE_MODEL_REVISION,
        pose_model_path(models_path),
        POSE_MODEL_SIZE,
        POSE_MODEL_SHA256,
    )
    _load_pose_cached.cache_clear()
    load_pose_state_dict(result)
    return result


def download_dwpose_models(models_path: str | os.PathLike[str]) -> tuple[Path, Path]:
    directory = dwpose_model_dir(models_path)
    results = []
    for filename, (size, digest) in DWPOSE_FILES.items():
        results.append(
            _download_verified(
                DWPOSE_REPO,
                filename,
                DWPOSE_REVISION,
                directory / filename,
                size,
                digest,
            )
        )
    _dwpose_detector_cached.cache_clear()
    return tuple(results)


def verify_dwpose_models(models_path: str | os.PathLike[str]) -> tuple[Path, Path]:
    directory = dwpose_model_dir(models_path)
    return tuple(
        _verify(directory / filename, size, digest)
        for filename, (size, digest) in DWPOSE_FILES.items()
    )


@lru_cache(maxsize=2)
def _dwpose_detector_cached(detector_path: str, pose_path: str, device: str):
    try:
        from easy_dwpose.body_estimation import Wholebody
    except ImportError as exc:
        raise RuntimeError(
            "DWPose needs easy-dwpose 1.0.2. Restart Forge once so the extension "
            "installer can add it, or install easy-dwpose==1.0.2 with --no-deps."
        ) from exc
    return Wholebody(model_det=detector_path, model_pose=pose_path, device=device)


def _dwpose_device() -> str:
    try:
        import onnxruntime

        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def load_dwpose_detector(models_path: str | os.PathLike[str]):
    try:
        pose_path, detector_path = verify_dwpose_models(models_path)
    except (FileNotFoundError, ValueError):
        pose_path, detector_path = download_dwpose_models(models_path)
    return _dwpose_detector_cached(
        str(detector_path), str(pose_path), _dwpose_device()
    )


def _format_dwpose(candidates, scores, width: int, height: int) -> dict[str, Any]:
    candidates = candidates.copy()
    scores = scores.copy()
    count, _, locations = candidates.shape
    candidates[..., 0] /= float(width)
    candidates[..., 1] /= float(height)
    bodies = candidates[:, :18].copy().reshape(count * 18, locations)
    body_scores = scores[:, :18]
    for person in range(len(body_scores)):
        for joint in range(len(body_scores[person])):
            body_scores[person][joint] = (
                int(18 * person + joint) if body_scores[person][joint] > 0.3 else -1
            )
    return {
        "bodies": bodies,
        "body_scores": body_scores,
        "hands": np.vstack([candidates[:, 92:113], candidates[:, 113:]]),
        "hands_scores": np.vstack([scores[:, 92:113], scores[:, 113:]]),
        "faces": candidates[:, 24:92],
        "faces_scores": scores[:, 24:92],
    }


def preprocess_pose_map(
    image: Any,
    preprocessor_name: str,
    resolution: int,
    models_path: str | os.PathLike[str],
) -> np.ndarray:
    """Create the expensive full body/hand/face map before target fitting."""

    source = normalize_image(image)
    if preprocessor_name == "DWPose (photo to pose)":
        from easy_dwpose.body_estimation import resize_image
        from easy_dwpose.draw import draw_openpose

        detector = load_dwpose_detector(models_path)
        detected = resize_image(source.copy(), target_resolution=int(resolution))
        candidates, scores = detector(detected)
        pose = _format_dwpose(candidates, scores, detected.shape[1], detected.shape[0])
        source = normalize_image(
            draw_openpose(pose, height=detected.shape[0], width=detected.shape[1])
        )
    elif preprocessor_name != "None (already an OpenPose map)":
        raise ValueError(f"Unsupported Krea 2 Pose preprocessor: {preprocessor_name}.")
    return np.ascontiguousarray(source)


def create_pose_map(
    image: Any,
    preprocessor_name: str,
    resolution: int,
    width: int,
    height: int,
    models_path: str | os.PathLike[str],
) -> np.ndarray:
    """Create a full body/hand/face OpenPose map, then letterbox it exactly."""

    source = preprocess_pose_map(image, preprocessor_name, resolution, models_path)
    return np.ascontiguousarray(fit_control_map(source, width, height))


def _fit_area(samples: torch.Tensor, max_pixels: int, snap: int = 1) -> torch.Tensor:
    height, width = samples.shape[-2:]
    scale = min(1.0, math.sqrt(max_pixels / (width * height)))
    new_width = max(round(width * scale / snap) * snap, snap)
    new_height = max(round(height * scale / snap) * snap, snap)
    if (new_height, new_width) == (height, width):
        return samples
    return F.interpolate(samples, size=(new_height, new_width), mode="area")


@torch.inference_mode()
def encode_pose_references(engine, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode BHWC pose maps using the exact ai-toolkit/Ostris size contract."""

    samples = images.movedim(-1, 1)
    vision = _fit_area(samples, VLM_MAX_PIXELS).movedim(1, -1)[..., :3]
    refs = _fit_area(samples, REF_LATENT_MAX_PIXELS, REF_SNAP)
    encoded = engine.forge_objects.vae.encode(refs.movedim(1, -1)[..., :3])
    latent = engine.forge_objects.vae.first_stage_model.process_in(encoded)
    if latent.ndim == 5 and latent.shape[2] == 1:
        latent = latent[:, :, 0]
    return vision, latent


class _PoseTextModel:
    def __init__(self, engine, vision: torch.Tensor):
        self.engine = engine
        self.vision = vision

    def get_learned_conditioning(self, prompt):
        from backend import memory_management

        memory_management.load_model_gpu(self.engine.forge_objects.clip.patcher)
        processor = copy(self.engine.text_processing_engine_qwen)
        # ai-toolkit/Ostris training places the vision marker after a numbered
        # label.  Forge's generic Krea reference path omits that label, so use a
        # shallow processor copy to reproduce the trained prompt without
        # mutating the engine shared by other requests.
        processor.vision_block = f"Picture 1: {processor.vision_block}"
        return processor(prompt, images=[self.vision])


@torch.inference_mode()
def build_pose_prompt_conditioning(
    engine,
    prompts: Sequence[str],
    vision: torch.Tensor,
    steps: int,
    *,
    width: int,
    height: int,
    negative: bool = False,
    multicond: bool = False,
    hires_steps: int | None = None,
    distilled_cfg_scale: float | None = None,
):
    """Build per-sample visual conditioning without Forge's prompt cache aliasing."""

    from modules import prompt_parser

    if len(prompts) != vision.shape[0]:
        raise RuntimeError(
            f"Krea Pose prompt/image batch mismatch: {len(prompts)} vs {vision.shape[0]}."
        )
    if multicond:
        combined = []
        for index, text in enumerate(prompts):
            conditioning = prompt_parser.SdConditioning(
                [text],
                width=width,
                height=height,
                distilled_cfg_scale=distilled_cfg_scale,
            )
            result = prompt_parser.get_multicond_learned_conditioning(
                _PoseTextModel(engine, vision[index : index + 1]),
                conditioning,
                steps,
                hires_steps,
            )
            combined.extend(result.batch)
        return prompt_parser.MulticondLearnedConditioning(
            shape=(len(prompts),), batch=combined
        )

    combined = []
    for index, text in enumerate(prompts):
        conditioning = prompt_parser.SdConditioning(
            [text],
            width=width,
            height=height,
            is_negative_prompt=negative,
            distilled_cfg_scale=distilled_cfg_scale,
        )
        result = prompt_parser.get_learned_conditioning(
            _PoseTextModel(engine, vision[index : index + 1]),
            conditioning,
            steps,
            hires_steps,
        )
        combined.extend(result)
    return combined


def _repeat_batch(tensor: torch.Tensor, batch: int) -> torch.Tensor:
    current = int(tensor.shape[0])
    if current == batch:
        return tensor
    if current <= 0 or batch % current != 0:
        raise RuntimeError(
            f"Cannot align Krea Pose batch {current} with sampling batch {batch}."
        )
    return tensor.repeat(batch // current, *([1] * (tensor.ndim - 1)))


def _pad_to_patch(tensor: torch.Tensor, patch: int) -> torch.Tensor:
    pad_height = (-tensor.shape[-2]) % patch
    pad_width = (-tensor.shape[-1]) % patch
    if pad_height or pad_width:
        tensor = F.pad(tensor, (0, pad_width, 0, pad_height), mode="replicate")
    return tensor


def _pack_refs(dit, ref_latents, batch: int, device, dtype):
    patch = int(dit.patch)
    tokens, positions = [], []

    for index, ref in enumerate(ref_latents):
        if ref.ndim == 5:
            if ref.shape[2] != 1:
                raise RuntimeError("Krea 2 Pose references must contain one frame.")
            ref = ref[:, :, 0]
        ref = _repeat_batch(ref, batch).to(device=device, dtype=dtype)
        ref = _pad_to_patch(ref, patch)
        ref_height, ref_width = ref.shape[-2] // patch, ref.shape[-1] // patch
        tokens.append(
            rearrange(
                ref,
                "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                ph=patch,
                pw=patch,
            )
        )
        ids = torch.zeros(
            ref_height, ref_width, 3, device=device, dtype=torch.float32
        )
        ids[..., 0] = index + 1.0
        ids[..., 1] = torch.arange(ref_height, device=device)[:, None]
        ids[..., 2] = torch.arange(ref_width, device=device)[None, :]
        positions.append(
            ids.reshape(1, ref_height * ref_width, 3).repeat(batch, 1, 1)
        )
    return torch.cat(tokens, dim=1), torch.cat(positions, dim=1)


def _block_ref_forward(
    block, value, real_vector, zero_vector, split, frequencies, transformer_options
):
    real = block.mod(real_vector)
    zero = block.mod(zero_vector)

    def mod(hidden, scale: int, shift: int):
        return torch.cat(
            (
                (1 + real[scale]) * hidden[:, :split] + real[shift],
                (1 + zero[scale]) * hidden[:, split:] + zero[shift],
            ),
            dim=1,
        )

    def gate(hidden, index: int):
        return torch.cat(
            (real[index] * hidden[:, :split], zero[index] * hidden[:, split:]),
            dim=1,
        )

    value = value + gate(
        block.attn(
            mod(block.prenorm(value), 0, 1),
            frequencies,
            None,
            transformer_options=transformer_options,
        ),
        2,
    )
    value = value + gate(block.mlp(mod(block.postnorm(value), 3, 4)), 5)
    return value


def _forward_with_pose_refs(
    dit, x, timesteps, context, ref_latents, transformer_options
):
    """Forge port of Ostris' ai-toolkit-compatible index-timestep-zero path."""

    from backend.nn.flux import timestep_embedding
    if x.ndim == 5:
        if x.shape[2] != 1:
            raise RuntimeError("Krea 2 Pose control supports image generation only.")
        x = x[:, :, 0]
    batch, _, original_height, original_width = x.shape
    patch = int(dit.patch)
    x = _pad_to_patch(x, patch)
    height, width = x.shape[-2:]
    grid_height, grid_width = height // patch, width // patch
    image = rearrange(
        x,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        ph=patch,
        pw=patch,
    )
    refs, ref_positions = _pack_refs(
        dit, ref_latents, batch, x.device, x.dtype
    )
    reference_length = refs.shape[1]
    image = dit.first(torch.cat((image, refs), dim=1))

    time = dit.tmlp(
        timestep_embedding(timesteps, dit.tdim).unsqueeze(1).to(image.dtype)
    )
    time_vector = dit.tproj(time)
    zero_time = dit.tmlp(
        timestep_embedding(torch.zeros_like(timesteps), dit.tdim)
        .unsqueeze(1)
        .to(image.dtype)
    )
    zero_vector = dit.tproj(zero_time)

    context = dit.txtfusion(
        context, mask=None, transformer_options=transformer_options
    )
    context = dit.txtmlp(context)
    text_length = context.shape[1]
    image_length = image.shape[1]
    combined = torch.cat((context, image), dim=1)
    split = text_length + image_length - reference_length

    text_positions = torch.zeros(
        batch, text_length, 3, device=x.device, dtype=torch.float32
    )
    image_ids = torch.zeros(
        grid_height, grid_width, 3, device=x.device, dtype=torch.float32
    )
    image_ids[..., 1] = torch.arange(grid_height, device=x.device)[:, None]
    image_ids[..., 2] = torch.arange(grid_width, device=x.device)[None, :]
    image_positions = image_ids.reshape(
        1, grid_height * grid_width, 3
    ).repeat(batch, 1, 1)
    frequencies = dit.pe_embedder(
        torch.cat((text_positions, image_positions, ref_positions), dim=1)
    )

    for block in dit.blocks:
        combined = _block_ref_forward(
            block,
            combined,
            time_vector,
            zero_vector,
            split,
            frequencies,
            transformer_options,
        )
    final = dit.last(combined, time)
    output = final[:, text_length:split]
    output = rearrange(
        output,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=grid_height,
        w=grid_width,
        ph=patch,
        pw=patch,
        c=dit.channels,
    )
    return output[:, :, :original_height, :original_width].unsqueeze(2)


def build_pose_lora_patches(state_dict, model) -> dict[str, Any]:
    from backend.patcher.lora import load_lora, model_lora_keys_unet

    key_map = model_lora_keys_unet(model, {})
    patches, remaining = load_lora(dict(state_dict), key_map)
    tensor_remaining = [key for key, value in remaining.items() if torch.is_tensor(value)]
    if len(patches) != POSE_EXPECTED_LAYERS or tensor_remaining:
        detail = ", ".join(tensor_remaining[:4])
        raise ValueError(
            f"Incomplete or incompatible Krea 2 Pose LoRA: {len(patches)}/"
            f"{POSE_EXPECTED_LAYERS} layers accepted; remaining tensors: {detail or 'none'}."
        )
    return patches


def apply_pose_control(unet, ref_latent, state_dict, strength: float):
    """Return a generation-local Forge UNet clone with Pose control attached."""

    diffusion = unet.model.diffusion_model
    required = ("first", "blocks", "patch", "channels", "txtfusion")
    if not all(hasattr(diffusion, name) for name in required):
        raise TypeError("Krea 2 Pose ControlNet-LoRA requires a native Krea 2 model.")
    new_unet = unet.clone()
    patches = build_pose_lora_patches(state_dict, unet.model)
    from backend.args import dynamic_args

    accepted = new_unet.add_patches(
        patches,
        strength_patch=float(strength),
        strength_model=1.0,
        filename=POSE_MODEL_FILENAME,
        online_mode=bool(dynamic_args.online_lora),
    )
    if len(accepted) != len(patches):
        raise ValueError(
            f"Forge accepted only {len(accepted)}/{len(patches)} Krea Pose layers."
        )

    refs = [ref_latent.detach()]

    def pose_forward(
        x,
        timesteps,
        context,
        attention_mask=None,
        transformer_options=None,
        **kwargs,
    ):
        del attention_mask, kwargs
        return _forward_with_pose_refs(
            diffusion,
            x,
            timesteps,
            context,
            refs,
            transformer_options or {},
        )

    new_unet.add_object_patch("diffusion_model.forward", pose_forward)
    return new_unet
