from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import forge_krea2_depth.pose as pose  # noqa: E402


def test_pose_checkpoint_contract_is_revision_and_hash_pinned():
    assert len(pose.POSE_MODEL_REVISION) == 40
    assert len(pose.POSE_MODEL_SHA256) == 64
    assert pose.POSE_MODEL_SIZE == 228587504
    assert pose.POSE_EXPECTED_LAYERS == 256


def test_pose_download_reuses_a_verified_file(tmp_path, monkeypatch):
    payload = b"verified pose"
    target = tmp_path / "ControlNet" / "Krea2" / pose.POSE_MODEL_FILENAME
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    monkeypatch.setattr(pose, "POSE_MODEL_SIZE", len(payload))
    monkeypatch.setattr(pose, "POSE_MODEL_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(pose, "load_pose_state_dict", lambda _path: {})
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = lambda **_kwargs: pytest.fail("network was used")
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    pose._verify_cached.cache_clear()

    assert pose.download_pose_model(tmp_path) == target.resolve()


def test_dwpose_downloads_both_files_at_one_pinned_revision(tmp_path, monkeypatch):
    payloads = {"pose.onnx": b"pose", "det.onnx": b"detector"}
    monkeypatch.setattr(
        pose,
        "DWPOSE_FILES",
        {
            name: (len(value), hashlib.sha256(value).hexdigest())
            for name, value in payloads.items()
        },
    )
    calls = []
    fake_hub = types.ModuleType("huggingface_hub")

    def fake_download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.write_bytes(payloads[kwargs["filename"]])
        return str(target)

    fake_hub.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    pose._verify_cached.cache_clear()

    results = pose.download_dwpose_models(tmp_path)

    assert len(results) == 2
    assert {call["revision"] for call in calls} == {pose.DWPOSE_REVISION}


def test_direct_openpose_map_is_letterboxed_without_crop():
    source = np.full((8, 4, 3), 200, dtype=np.uint8)

    result = pose.create_pose_map(
        source,
        "None (already an OpenPose map)",
        512,
        8,
        8,
        ".",
    )

    assert result.shape == (8, 8, 3)
    assert np.all(result[:, :2] == 0)
    assert np.all(result[:, 6:] == 0)
    assert np.all(result[:, 2:6] == 200)


def test_dwpose_photo_path_renders_then_letterboxes(monkeypatch):
    body = types.ModuleType("easy_dwpose.body_estimation")
    body.resize_image = lambda image, target_resolution: image
    drawing = types.ModuleType("easy_dwpose.draw")
    drawing.draw_openpose = lambda _pose, height, width: np.pad(
        np.full((height, width - 2, 3), 255, dtype=np.uint8),
        ((0, 0), (1, 1), (0, 0)),
    )
    monkeypatch.setitem(sys.modules, "easy_dwpose.body_estimation", body)
    monkeypatch.setitem(sys.modules, "easy_dwpose.draw", drawing)
    candidates = np.ones((1, 134, 2), dtype=np.float32)
    scores = np.ones((1, 134), dtype=np.float32)
    monkeypatch.setattr(pose, "load_dwpose_detector", lambda _path: lambda _image: (candidates, scores))

    result = pose.create_pose_map(
        np.zeros((4, 8, 3), dtype=np.uint8),
        "DWPose (photo to pose)",
        512,
        8,
        8,
        ".",
    )

    assert result.shape == (8, 8, 3)
    assert np.count_nonzero(result) > 0
    assert np.all(result[0] == 0)
    assert np.all(result[-1] == 0)


def test_fit_area_never_upscales_and_snaps_reference_dimensions():
    small = torch.zeros(1, 3, 100, 200)
    assert pose._fit_area(small, pose.VLM_MAX_PIXELS).shape == small.shape
    large = torch.zeros(1, 3, 1600, 800)
    fitted = pose._fit_area(large, pose.REF_LATENT_MAX_PIXELS, pose.REF_SNAP)
    assert fitted.shape[-2] % 16 == 0
    assert fitted.shape[-1] % 16 == 0
    assert fitted.shape[-2] * fitted.shape[-1] <= pose.REF_LATENT_MAX_PIXELS


def test_reference_batch_repeats_ab_for_cfg_without_crossing_images():
    dit = types.SimpleNamespace(patch=1)
    references = torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1)

    tokens, positions = pose._pack_refs(
        dit, [references], 4, torch.device("cpu"), torch.float32
    )

    assert tokens[:, 0, 0].tolist() == [1.0, 2.0, 1.0, 2.0]
    assert positions.shape == (4, 1, 3)


def test_reference_span_receives_zero_timestep_modulation():
    class Block:
        prenorm = nn.Identity()
        postnorm = nn.Identity()

        def mod(self, vector):
            return (vector,) * 6

        def attn(self, value, *_args, **_kwargs):
            self.attention_input = value.detach().clone()
            return torch.zeros_like(value)

        def mlp(self, value):
            self.mlp_input = value.detach().clone()
            return torch.zeros_like(value)

    block = Block()
    values = torch.zeros(1, 4, 1)
    real = torch.full((1, 1, 1), 2.0)
    zero = torch.full((1, 1, 1), 5.0)

    pose._block_ref_forward(block, values, real, zero, 2, None, {})

    assert block.attention_input[0, :2, 0].tolist() == [2.0, 2.0]
    assert block.attention_input[0, 2:, 0].tolist() == [5.0, 5.0]


def test_visual_prompt_conditioning_keeps_duplicate_prompts_per_image(
    monkeypatch,
):
    fake_backend = types.ModuleType("backend")
    fake_backend.memory_management = types.SimpleNamespace(
        load_model_gpu=lambda _model: None
    )
    monkeypatch.setitem(sys.modules, "backend", fake_backend)
    parser = types.ModuleType("modules.prompt_parser")

    class SdConditioning(list):
        def __init__(self, values, **kwargs):
            super().__init__(values)
            self.kwargs = kwargs

    class Multi:
        def __init__(self, shape, batch):
            self.shape = shape
            self.batch = batch

    parser.SdConditioning = SdConditioning
    parser.MulticondLearnedConditioning = Multi
    parser.get_multicond_learned_conditioning = (
        lambda model, conditioning, *_args: types.SimpleNamespace(
            batch=[[model.get_learned_conditioning(conditioning)[0]]]
        )
    )
    fake_modules = types.ModuleType("modules")
    fake_modules.prompt_parser = parser
    monkeypatch.setitem(sys.modules, "modules", fake_modules)
    monkeypatch.setitem(sys.modules, "modules.prompt_parser", parser)

    class TextProcessor:
        vision_block = "<vision>"

        def __call__(self, _prompt, images):
            conditioned_blocks.append(self.vision_block)
            return [float(images[0].mean())]

    conditioned_blocks = []
    engine = types.SimpleNamespace(
        forge_objects=types.SimpleNamespace(
            clip=types.SimpleNamespace(patcher=object())
        ),
        text_processing_engine_qwen=TextProcessor(),
    )
    vision = torch.stack((torch.ones(2, 2, 3), torch.full((2, 2, 3), 2.0)))

    result = pose.build_pose_prompt_conditioning(
        engine,
        ["same prompt", "same prompt"],
        vision,
        10,
        width=64,
        height=64,
        multicond=True,
    )

    assert result.batch == [[1.0], [2.0]]
    assert conditioned_blocks == ["Picture 1: <vision>", "Picture 1: <vision>"]


def test_pose_lora_loader_rejects_unmapped_tensors(monkeypatch):
    fake_lora = types.ModuleType("backend.patcher.lora")
    fake_lora.model_lora_keys_unet = lambda _model, _keys: {"layer": "weight"}
    fake_lora.load_lora = lambda state, _mapping: (
        {"weight": object()},
        {key: value for key, value in state.items() if key != "layer.lora_A.weight"},
    )
    monkeypatch.setitem(sys.modules, "backend.patcher.lora", fake_lora)
    monkeypatch.setattr(pose, "POSE_EXPECTED_LAYERS", 1)

    with pytest.raises(ValueError, match="remaining tensors"):
        pose.build_pose_lora_patches(
            {
                "layer.lora_A.weight": torch.ones(1),
                "unexpected": torch.ones(1),
            },
            object(),
        )


def test_apply_pose_control_is_generation_local_and_strength_scales_lora(
    monkeypatch,
):
    dynamic = types.SimpleNamespace(online_lora=True)
    fake_args = types.ModuleType("backend.args")
    fake_args.dynamic_args = dynamic
    monkeypatch.setitem(sys.modules, "backend.args", fake_args)
    monkeypatch.setattr(pose, "build_pose_lora_patches", lambda *_args: {"weight": object()})

    diffusion = types.SimpleNamespace(
        first=object(), blocks=[], patch=2, channels=16, txtfusion=object()
    )

    class FakeUnet:
        def __init__(self):
            self.model = types.SimpleNamespace(diffusion_model=diffusion)
            self.object_patches = {}

        def clone(self):
            return FakeUnet()

        def add_patches(self, patches, **kwargs):
            self.patch_kwargs = kwargs
            return list(patches)

        def add_object_patch(self, name, value):
            self.object_patches[name] = value

    original = FakeUnet()
    controlled = pose.apply_pose_control(
        original, torch.zeros(1, 16, 4, 4), {}, 0.85
    )

    assert controlled is not original
    assert original.object_patches == {}
    assert controlled.patch_kwargs["strength_patch"] == 0.85
    assert controlled.patch_kwargs["online_mode"] is True
    assert "diffusion_model.forward" in controlled.object_patches
