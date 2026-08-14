from __future__ import annotations

import sys
import types
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
    state = {"first.weight": torch.zeros(6, features * 2)}
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


def test_projection_preserves_base_and_adds_depth_branch(monkeypatch):
    backend_mm = types.ModuleType("backend.memory_management")
    backend_mm.cast_to = lambda value, device, dtype: value.to(device=device, dtype=dtype)
    monkeypatch.setitem(sys.modules, "backend.memory_management", backend_mm)
    base = nn.Linear(4, 3)
    control_weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    projection = ControlProjection(base, control_weight)
    image = torch.randn(2, 5, 4)
    control = torch.randn(1, 5, 4)
    projection.control_tokens = control
    expected = base(image) + torch.nn.functional.linear(control.repeat(2, 1, 1), control_weight)
    torch.testing.assert_close(projection(image), expected)


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


def test_wrapper_restores_tokens_and_composes_existing_wrapper():
    projection = types.SimpleNamespace(control_tokens="old", in_features=4)
    seen = []

    def previous(model_function, call):
        seen.append(projection.control_tokens.shape)
        return model_function(call["input"], call["timestep"], **call["c"]) + 1

    wrapper = _compose_control_wrapper(
        projection, torch.zeros(1, 1, 4, 4), 2, previous
    )
    call = {
        "input": torch.zeros(1, 1, 1, 4, 4),
        "timestep": torch.tensor([1.0]),
        "c": {},
    }
    result = wrapper(lambda x, t, **c: torch.tensor(4), call)
    assert result.item() == 5
    assert seen == [(1, 4, 4)]
    assert projection.control_tokens == "old"


def test_wrapper_restores_tokens_after_interrupt_like_exception():
    projection = types.SimpleNamespace(control_tokens=None, in_features=4)
    wrapper = _compose_control_wrapper(
        projection, torch.zeros(1, 1, 4, 4), 2, None
    )
    call = {
        "input": torch.zeros(1, 1, 1, 4, 4),
        "timestep": torch.tensor([1.0]),
        "c": {},
    }

    def interrupted(*args, **kwargs):
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        wrapper(interrupted, call)
    assert projection.control_tokens is None


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
