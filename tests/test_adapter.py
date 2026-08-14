from __future__ import annotations

import sys
import types
import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FakeLoRAAdapter:
    def __init__(self, loaded_keys, weights):
        self.loaded_keys = loaded_keys
        self.weights = weights


fake_lora = types.ModuleType("modules_forge.packages.comfy.weight_adapter.lora")
fake_lora.LoRAAdapter = FakeLoRAAdapter
sys.modules.setdefault(
    "modules_forge.packages.comfy.weight_adapter.lora", fake_lora
)

import forge_krea2_depth.adapter as adapter  # noqa: E402
from forge_krea2_depth.adapter import (  # noqa: E402
    EXPECTED_BLOCKS,
    LORA_TARGETS,
    ControlProjection,
    _compose_control_wrapper,
    build_lora_patches,
    install_failure_guard,
    make_control_tokens,
)


def make_state(features=4, rank=2):
    state = {
        "first.weight": torch.zeros(6, features * 2),
        "first.bias": torch.zeros(6),
    }
    model = {}
    for block in range(EXPECTED_BLOCKS):
        for target in LORA_TARGETS:
            name = f"blocks.{block}.{target}"
            state[f"{name}.A"] = torch.ones(rank, features)
            state[f"{name}.B"] = torch.ones(6, rank)
            model[f"diffusion_model.{name}.weight"] = torch.zeros(6, features)
    return state, model


def test_builds_every_expected_lora_pair():
    state, model = make_state()
    patches = build_lora_patches(state, model)
    assert len(patches) == EXPECTED_BLOCKS * len(LORA_TARGETS)
    sample = patches["diffusion_model.blocks.0.attn.wq.weight"]
    assert sample.weights[0].shape == (6, 2)
    assert sample.weights[1].shape == (2, 4)
    assert sample.weights[2] == 2.0


def test_rejects_partial_checkpoint_instead_of_silently_degrading():
    state, model = make_state()
    del state["blocks.7.mlp.up.B"]
    with pytest.raises(ValueError, match="223/224"):
        build_lora_patches(state, model)


def test_projection_uses_official_full_weight_and_bias(monkeypatch):
    backend_mm = types.ModuleType("backend.memory_management")
    backend_mm.cast_to = lambda value, device, dtype: value.to(device=device, dtype=dtype)
    monkeypatch.setitem(sys.modules, "backend.memory_management", backend_mm)
    weight = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    bias = torch.tensor([1.0, 2.0, 3.0])
    projection = ControlProjection(weight, bias, image_features=4)
    image = torch.randn(2, 5, 4)
    control = torch.randn(1, 5, 4)
    combined = torch.cat((image, control.repeat(2, 1, 1)), dim=-1)
    expected = torch.nn.functional.linear(combined, weight, bias)
    torch.testing.assert_close(projection(image, control), expected)


def test_projection_casts_cpu_weights_to_the_live_low_vram_dtype(monkeypatch):
    casts = []
    backend_mm = types.ModuleType("backend.memory_management")

    def cast_to(value, device, dtype):
        casts.append((value.device.type, device.type, dtype))
        return value.to(device=device, dtype=dtype)

    backend_mm.cast_to = cast_to
    monkeypatch.setitem(sys.modules, "backend.memory_management", backend_mm)
    projection = ControlProjection(
        torch.ones(3, 8, dtype=torch.float32),
        torch.zeros(3, dtype=torch.float32),
        image_features=4,
    )
    image = torch.ones(1, 2, 4, dtype=torch.float64)
    control = torch.ones(1, 2, 4, dtype=torch.float32)

    result = projection(image, control)

    assert result.dtype == torch.float64
    assert casts == [
        ("cpu", "cpu", torch.float64),
        ("cpu", "cpu", torch.float64),
    ]


def test_control_tokens_resize_repeat_and_patchify():
    latent = torch.arange(16.0).reshape(1, 1, 4, 4)
    sample = torch.zeros(2, 1, 1, 4, 4)
    tokens = make_control_tokens(latent, sample, patch_size=2, expected_features=4)
    assert tokens.shape == (2, 4, 4)
    torch.testing.assert_close(tokens[0], tokens[1])
    assert tokens[0, 0].tolist() == [0.0, 1.0, 4.0, 5.0]


def test_control_tokens_reject_wrong_vae_channels():
    latent = torch.zeros(1, 2, 4, 4)
    sample = torch.zeros(1, 2, 1, 4, 4)
    with pytest.raises(RuntimeError, match="8 features"):
        make_control_tokens(latent, sample, patch_size=2, expected_features=4)


def _hook_fixture(monkeypatch):
    backend_mm = types.ModuleType("backend.memory_management")
    backend_mm.cast_to = lambda value, device, dtype: value.to(device=device, dtype=dtype)
    monkeypatch.setitem(sys.modules, "backend.memory_management", backend_mm)
    first = nn.Linear(4, 3)
    weight = torch.zeros(3, 8)
    weight[:, 4:] = 1.0
    projection = ControlProjection(weight, torch.zeros(3), image_features=4)
    call = {
        "input": torch.zeros(1, 1, 1, 4, 4),
        "timestep": torch.tensor([1.0]),
        "c": {},
    }
    return first, projection, call


def test_wrapper_composes_previous_wrapper_and_is_fresh_for_each_call(monkeypatch):
    first, projection, call = _hook_fixture(monkeypatch)
    seen = []

    def previous(model_function, call):
        seen.append(model_function(call["input"], call["timestep"], **call["c"]))
        seen.append(model_function(call["input"], call["timestep"], **call["c"]))
        return "composed"

    image = torch.ones(1, 4, 4)
    reference = torch.full((1, 4, 4), 2.0)

    def model_function(*_args, **_kwargs):
        return first(image), first(reference)

    wrapper = _compose_control_wrapper(
        first, projection, torch.ones(1, 1, 4, 4), 2, previous
    )
    assert wrapper(model_function, call) == "composed"
    assert len(seen) == 2
    for controlled_image, untouched_reference in seen:
        torch.testing.assert_close(controlled_image, torch.full((1, 4, 3), 4.0))
        torch.testing.assert_close(untouched_reference, first(reference))
    assert len(first._forward_hooks) == 0


def test_wrapper_removes_hook_after_interrupt_like_exception(monkeypatch):
    first, projection, call = _hook_fixture(monkeypatch)
    wrapper = _compose_control_wrapper(
        first, projection, torch.zeros(1, 1, 4, 4), 2, None
    )

    def interrupted(*args, **kwargs):
        first(torch.ones(1, 4, 4))
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        wrapper(interrupted, call)
    assert len(first._forward_hooks) == 0


def test_apply_control_keeps_first_registered_and_avoids_object_patch(monkeypatch):
    class FakeUnet:
        def __init__(self, model):
            self.model = model
            self.model_options = {}
            self.object_patches = {}

        def clone(self):
            return FakeUnet(self.model)

        def get_model_object(self, name):
            assert name == "diffusion_model.first"
            return self.model.diffusion_model.first

        def add_patches(self, patches, **_kwargs):
            self.patches = patches
            self.patch_kwargs = _kwargs
            return list(patches)

        def set_model_unet_function_wrapper(self, wrapper):
            self.wrapper = wrapper

    diffusion = nn.Module()
    diffusion.first = nn.Linear(4, 6)
    diffusion.blocks = nn.ModuleList()
    diffusion.patch = 2
    diffusion.channels = 1
    model = nn.Module()
    model.diffusion_model = diffusion
    unet = FakeUnet(model)
    state = {
        "first.weight": torch.zeros(6, 8),
        "first.bias": torch.zeros(6),
    }
    monkeypatch.setattr(adapter, "build_lora_patches", lambda *_: {"layer": object()})

    controlled = adapter.apply_depth_control(
        unet, torch.zeros(1, 1, 4, 4), state, 1.0
    )

    assert controlled.object_patches == {}
    assert "diffusion_model.first.weight" in dict(model.named_parameters())
    assert controlled.model.diffusion_model.first is diffusion.first
    assert controlled.patch_kwargs["strength_patch"] == 1.0
    assert controlled.patch_kwargs["strength_model"] == 1.0


def test_download_is_revision_pinned_and_sha_verified(tmp_path, monkeypatch):
    payload = b"verified control model"
    downloaded = tmp_path / adapter.CONTROL_MODEL_FILENAME
    downloaded.write_bytes(payload)
    calls = {}
    fake_hub = types.ModuleType("huggingface_hub")

    def fake_download(**kwargs):
        calls.update(kwargs)
        return str(downloaded)

    fake_hub.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setattr(adapter, "CONTROL_MODEL_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(adapter, "load_control_state_dict", lambda path: {})

    assert adapter.download_control_model(tmp_path) == downloaded.resolve()
    assert calls["revision"] == adapter.CONTROL_MODEL_REVISION


def test_download_rejects_wrong_sha(tmp_path, monkeypatch):
    downloaded = tmp_path / adapter.CONTROL_MODEL_FILENAME
    downloaded.write_bytes(b"corrupt")
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = lambda **_kwargs: str(downloaded)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    with pytest.raises(ValueError, match="Invalid depth-control-lora.safetensors SHA-256"):
        adapter.download_control_model(tmp_path)


def test_ordinary_model_load_rejects_wrong_sha(tmp_path, monkeypatch):
    checkpoint = tmp_path / adapter.CONTROL_MODEL_FILENAME
    checkpoint.write_bytes(b"same-shaped but untrusted control model")
    backend_utils = types.ModuleType("backend.utils")
    backend_utils.load_torch_file = lambda *_args, **_kwargs: {"loaded": True}
    monkeypatch.setitem(sys.modules, "backend.utils", backend_utils)

    with pytest.raises(ValueError, match="Invalid depth-control-lora.safetensors SHA-256"):
        adapter.load_control_state_dict(checkpoint)


def test_download_uses_verified_existing_file_without_network(tmp_path, monkeypatch):
    payload = b"already verified"
    checkpoint = tmp_path / "ControlNet" / "Krea2" / adapter.CONTROL_MODEL_FILENAME
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(payload)
    monkeypatch.setattr(adapter, "CONTROL_MODEL_SHA256", hashlib.sha256(payload).hexdigest())
    backend_utils = types.ModuleType("backend.utils")
    backend_utils.load_torch_file = lambda *_args, **_kwargs: {"loaded": True}
    monkeypatch.setitem(sys.modules, "backend.utils", backend_utils)
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = lambda **_kwargs: pytest.fail("network was used")
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    assert adapter.download_control_model(tmp_path) == checkpoint.resolve()


def test_download_replaces_a_corrupt_existing_file(tmp_path, monkeypatch):
    payload = b"fresh verified model"
    checkpoint = tmp_path / "ControlNet" / "Krea2" / adapter.CONTROL_MODEL_FILENAME
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"corrupt existing model")
    monkeypatch.setattr(adapter, "CONTROL_MODEL_SHA256", hashlib.sha256(payload).hexdigest())
    backend_utils = types.ModuleType("backend.utils")
    backend_utils.load_torch_file = lambda *_args, **_kwargs: {"loaded": True}
    monkeypatch.setitem(sys.modules, "backend.utils", backend_utils)
    calls = {}
    fake_hub = types.ModuleType("huggingface_hub")

    def fake_download(**kwargs):
        calls.update(kwargs)
        checkpoint.write_bytes(payload)
        return str(checkpoint)

    fake_hub.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    assert adapter.download_control_model(tmp_path) == checkpoint.resolve()
    assert calls["force_download"] is True


def test_failure_guard_prevents_silent_uncontrolled_sampling():
    class FakeUnet:
        def clone(self):
            return FakeUnet()

        def set_model_unet_function_wrapper(self, wrapper):
            self.wrapper = wrapper

    process = types.SimpleNamespace(
        sd_model=types.SimpleNamespace(
            forge_objects=types.SimpleNamespace(unet=FakeUnet())
        )
    )
    install_failure_guard(process, ValueError("missing control image"))
    with pytest.raises(RuntimeError, match="missing control image"):
        process.sd_model.forge_objects.unet.wrapper(None, {})
