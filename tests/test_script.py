from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


class FakeComponent:
    registry: list["FakeComponent"] = []

    def __init__(self, value=None, *args, kind="component", **kwargs):
        del args
        self.value = value
        self.kind = kind
        self.kwargs = kwargs
        self.do_not_save_to_config = False
        self.events = []
        self.registry.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def click(self, **kwargs):
        self.events.append(kwargs)
        return self


def _component(kind):
    return lambda value=None, *args, **kwargs: FakeComponent(
        value, *args, kind=kind, **kwargs
    )


def load_script(monkeypatch, tmp_path, preprocessors=None):
    FakeComponent.registry = []
    fake_gradio = types.ModuleType("gradio")
    for name in ("Markdown", "Row", "Image", "Dropdown", "Slider", "Checkbox", "Button"):
        setattr(fake_gradio, name, _component(name))

    fake_scripts = types.ModuleType("modules.scripts")
    fake_scripts.ScriptBuiltinUI = object
    fake_scripts.AlwaysVisible = object()
    fake_paths = types.SimpleNamespace(models_path=str(tmp_path))
    fake_modules = types.ModuleType("modules")
    fake_modules.paths = fake_paths
    fake_modules.scripts = fake_scripts

    class FakeInputAccordion(FakeComponent):
        def __init__(self, value=None, **kwargs):
            super().__init__(value, kind="InputAccordion", **kwargs)

    fake_ui_components = types.ModuleType("modules.ui_components")
    fake_ui_components.InputAccordion = FakeInputAccordion
    fake_forge_shared = types.ModuleType("modules_forge.shared")
    fake_forge_shared.supported_preprocessors = preprocessors or {}

    monkeypatch.setitem(sys.modules, "gradio", fake_gradio)
    monkeypatch.setitem(sys.modules, "modules", fake_modules)
    monkeypatch.setitem(sys.modules, "modules.scripts", fake_scripts)
    monkeypatch.setitem(sys.modules, "modules.ui_components", fake_ui_components)
    monkeypatch.setitem(sys.modules, "modules_forge.shared", fake_forge_shared)

    path = ROOT / "scripts" / "krea2_depth_controlnet.py"
    name = f"krea2_depth_controlnet_test_{id(monkeypatch)}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_process():
    class Krea2:
        def __init__(self):
            self.forge_objects = types.SimpleNamespace(unet="base-unet")

        def encode_vision(self, image):
            self.encoded = image
            return None, torch.zeros(1, 1, 1, 2, 2)

    return types.SimpleNamespace(
        sd_model=Krea2(),
        width=64,
        height=32,
        is_hr_pass=False,
        extra_generation_params={},
    )


def install_processing_fakes(module, monkeypatch):
    calls = types.SimpleNamespace(depth=[], applied=[], guarded=[])

    def create_depth(image, preprocessor, resolution, width, height, invert):
        calls.depth.append(
            (image, preprocessor, resolution, width, height, invert)
        )
        if image == "bad-image":
            raise ValueError("invalid control image")
        return np.zeros((height, width, 3), dtype=np.uint8)

    def apply(unet, latent, state_dict, strength):
        calls.applied.append((unet, latent.shape, state_dict, strength))
        return "controlled-unet"

    def guard(process, error):
        calls.guarded.append((process, str(error)))

    monkeypatch.setattr(module, "create_depth_map", create_depth)
    monkeypatch.setattr(module, "load_control_state_dict", lambda _path: {"model": True})
    monkeypatch.setattr(module, "apply_depth_control", apply)
    monkeypatch.setattr(module, "install_failure_guard", guard)
    return calls


def test_panel_contract_and_preprocessor_choices(monkeypatch, tmp_path):
    module = load_script(
        monkeypatch,
        tmp_path,
        preprocessors={"depth_anything_v2": object(), "unrelated": object()},
    )
    script = module.Krea2DepthControlScript()

    controls = script.ui(False)

    assert script.title() == "Krea 2 Depth ControlNet-LoRA"
    assert script.show(False) is module.scripts.AlwaysVisible
    assert len(controls) == 6
    assert [component.kind for component in controls] == [
        "InputAccordion",
        "Image",
        "Dropdown",
        "Slider",
        "Checkbox",
        "Slider",
    ]
    assert [component.value for component in controls] == [
        False,
        None,
        "depth_anything_v2",
        768,
        False,
        1.0,
    ]
    assert controls[1].kwargs["type"] == "numpy"
    assert controls[2].kwargs["choices"] == [
        "None (already a depth map)",
        "depth_anything_v2",
    ]
    button_labels = [
        component.value
        for component in FakeComponent.registry
        if component.kind == "Button"
    ]
    assert button_labels == ["Preview depth", "Download / verify model (862 MB)"]
    assert not any(label in button_labels for label in ("Generate", "Skip", "Interrupt"))


def test_existing_model_status_defers_verification_until_loading(
    monkeypatch, tmp_path
):
    model_path = (
        tmp_path / "ControlNet" / "Krea2" / "depth-control-lora.safetensors"
    )
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"not verified yet")
    module = load_script(monkeypatch, tmp_path)

    module.Krea2DepthControlScript().ui(False)

    statuses = [
        component.value
        for component in FakeComponent.registry
        if component.kind == "Markdown" and str(component.value).startswith("Model:")
    ]
    assert len(statuses) == 1
    assert "present" in statuses[0]
    assert "verified automatically before loading" in statuses[0]
    assert "— ready" not in statuses[0]


@pytest.mark.parametrize(
    ("enabled", "resolution", "invert", "strength", "message"),
    [
        ("false", 768, False, 1.0, "enabled"),
        (True, 768, "false", 1.0, "invert"),
        (True, 768.5, False, 1.0, "resolution"),
        (True, 768, False, math.nan, "strength"),
        (True, 768, False, -0.1, "strength"),
        (True, 768, False, 2.1, "strength"),
    ],
)
def test_api_rejects_ambiguous_or_out_of_range_controls(
    monkeypatch, tmp_path, enabled, resolution, invert, strength, message
):
    module = load_script(monkeypatch, tmp_path)
    calls = install_processing_fakes(module, monkeypatch)
    process = make_process()

    with pytest.raises(ValueError, match=message):
        module.Krea2DepthControlScript().process_before_every_sampling(
            process,
            enabled,
            "image",
            "None (already a depth map)",
            resolution,
            invert,
            strength,
        )

    assert not calls.depth
    assert not calls.applied
    assert len(calls.guarded) == 1


@pytest.mark.parametrize("enabled,strength", [(False, 1.0), (True, 0.0)])
def test_disabled_and_zero_strength_are_true_noops(
    monkeypatch, tmp_path, enabled, strength
):
    module = load_script(monkeypatch, tmp_path)
    calls = install_processing_fakes(module, monkeypatch)
    process = make_process()
    original = process.sd_model.forge_objects.unet

    module.Krea2DepthControlScript().process_before_every_sampling(
        process,
        enabled,
        None,
        None,
        None,
        None,
        strength,
    )

    assert process.sd_model.forge_objects.unet == original
    assert not process.extra_generation_params
    assert not calls.depth
    assert not calls.applied
    assert not calls.guarded


def test_valid_api_values_apply_control_without_touching_forge_state(
    monkeypatch, tmp_path
):
    module = load_script(monkeypatch, tmp_path)
    calls = install_processing_fakes(module, monkeypatch)
    process = make_process()

    module.Krea2DepthControlScript().process_before_every_sampling(
        process,
        True,
        "image",
        "None (already a depth map)",
        768,
        False,
        0.75,
    )

    assert calls.depth == [
        ("image", "None (already a depth map)", 768, 64, 32, False)
    ]
    assert calls.applied[0][0] == "base-unet"
    assert calls.applied[0][1] == (1, 1, 2, 2)
    assert calls.applied[0][3] == 0.75
    assert process.sd_model.forge_objects.unet == "controlled-unet"
    assert process.extra_generation_params["Krea 2 Depth Strength"] == 0.75
    assert not calls.guarded


def test_missing_preprocessor_fails_closed_before_processing(monkeypatch, tmp_path):
    module = load_script(monkeypatch, tmp_path)
    calls = install_processing_fakes(module, monkeypatch)
    process = make_process()

    with pytest.raises(ValueError, match="preprocessor"):
        module.Krea2DepthControlScript().process_before_every_sampling(
            process, True, "image", None, 768, False, 1.0
        )

    assert not calls.depth
    assert len(calls.guarded) == 1


def test_incompatible_checkpoint_fails_closed_before_processing(
    monkeypatch, tmp_path
):
    module = load_script(monkeypatch, tmp_path)
    calls = install_processing_fakes(module, monkeypatch)
    process = make_process()
    process.sd_model = types.SimpleNamespace(
        forge_objects=types.SimpleNamespace(unet="base-unet")
    )

    with pytest.raises(TypeError, match="requires a Krea 2 checkpoint"):
        module.Krea2DepthControlScript().process_before_every_sampling(
            process,
            True,
            "image",
            "None (already a depth map)",
            768,
            False,
            1.0,
        )

    assert not calls.depth
    assert len(calls.guarded) == 1


def test_invalid_image_fails_closed_and_a_new_request_can_succeed(
    monkeypatch, tmp_path
):
    module = load_script(monkeypatch, tmp_path)
    calls = install_processing_fakes(module, monkeypatch)
    script = module.Krea2DepthControlScript()
    failed = make_process()

    with pytest.raises(ValueError, match="invalid control image"):
        script.process_before_every_sampling(
            failed,
            True,
            "bad-image",
            "None (already a depth map)",
            768,
            False,
            1.0,
        )

    assert len(calls.guarded) == 1
    resumed = make_process()
    script.process_before_every_sampling(
        resumed,
        True,
        "image",
        "None (already a depth map)",
        768,
        False,
        1.0,
    )
    assert resumed.sd_model.forge_objects.unet == "controlled-unet"


def test_hires_pass_applies_the_exact_runtime_dimensions(monkeypatch, tmp_path):
    module = load_script(monkeypatch, tmp_path)
    calls = install_processing_fakes(module, monkeypatch)
    process = make_process()
    process.is_hr_pass = True
    process.hr_upscale_to_x = 96
    process.hr_upscale_to_y = 48

    module.Krea2DepthControlScript().process_before_every_sampling(
        process,
        True,
        "image",
        "None (already a depth map)",
        768,
        False,
        1.0,
    )

    assert calls.depth[0][3:5] == (96, 48)
