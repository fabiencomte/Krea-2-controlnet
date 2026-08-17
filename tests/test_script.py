from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class FakeComponent:
    registry: list["FakeComponent"] = []

    def __init__(self, value=None, *args, kind="component", **kwargs):
        del args
        self.value = value
        self.kind = kind
        self.kwargs = kwargs
        self.elem_id = kwargs.get("elem_id")
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

    def change(self, **kwargs):
        self.events.append(kwargs)
        return self

    def input(self, **kwargs):
        self.events.append(kwargs)
        return self

    def upload(self, **kwargs):
        self.events.append(kwargs)
        return self

    def select(self, **kwargs):
        self.events.append(kwargs)
        return self


def _component(kind):
    return lambda value=None, *args, **kwargs: FakeComponent(
        value, *args, kind=kind, **kwargs
    )


def load_script(monkeypatch, tmp_path, preprocessors=None):
    FakeComponent.registry = []
    fake_gradio = types.ModuleType("gradio")
    for name in (
        "Markdown",
        "Row",
        "Column",
        "Image",
        "Gallery",
        "UploadButton",
        "Dataframe",
        "State",
        "Dropdown",
        "Slider",
        "Number",
        "Checkbox",
        "Button",
    ):
        setattr(fake_gradio, name, _component(name))
    fake_gradio.warnings = []
    fake_gradio.Warning = lambda message: fake_gradio.warnings.append(message)
    fake_gradio.update = lambda **kwargs: kwargs
    fake_gradio.SelectData = type("SelectData", (), {})

    fake_scripts = types.ModuleType("modules.scripts")

    class FakeScriptBuiltinUI:
        is_txt2img = True
        is_img2img = False
        tabname = "txt2img"

        def elem_id(self, item_id):
            return f"txt2img_{item_id}"

    fake_scripts.ScriptBuiltinUI = FakeScriptBuiltinUI
    fake_scripts.AlwaysVisible = object()
    fake_paths = types.SimpleNamespace(models_path=str(tmp_path))
    fake_modules = types.ModuleType("modules")
    fake_modules.paths = fake_paths
    fake_modules.scripts = fake_scripts
    assigned_previews = []
    fake_modules.shared = types.SimpleNamespace(
        opts=types.SimpleNamespace(
            res_step=64,
            CLIP_stop_at_last_layers=1,
            live_previews_enable=True,
        ),
        state=types.SimpleNamespace(
            assign_current_image=assigned_previews.append,
            textinfo="",
            assigned_previews=assigned_previews,
        ),
    )

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
            return None, torch.zeros(image.shape[0], 1, 1, 2, 2)

    return types.SimpleNamespace(
        sd_model=Krea2(),
        width=64,
        height=32,
        is_hr_pass=False,
        batch_size=1,
        iteration=0,
        extra_generation_params={},
    )


def install_processing_fakes(module, monkeypatch):
    calls = types.SimpleNamespace(depth=[], applied=[], guarded=[])

    def preprocess_depth(image, preprocessor, resolution, invert):
        calls.depth.append((image, preprocessor, resolution, invert))
        if image == "bad-image":
            raise ValueError("invalid control image")
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def apply(unet, latent, state_dict, strength):
        calls.applied.append((unet, latent.shape, state_dict, strength))
        return "controlled-unet"

    def guard(process, error):
        calls.guarded.append((process, str(error)))

    def normalize_many(images):
        return images if isinstance(images, list) else [images]

    monkeypatch.setattr(module, "normalize_images", normalize_many)
    monkeypatch.setattr(module, "ratio_warning", lambda *_args: "")
    monkeypatch.setattr(module, "preprocess_depth_map", preprocess_depth)
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

    assert script.title() == "Krea 2 Depth / Pose ControlNet-LoRA"
    assert script.show(False) is module.scripts.AlwaysVisible
    assert len(controls) == 7
    assert [component.kind for component in controls] == [
        "InputAccordion",
        "Dropdown",
        "State",
        "Dropdown",
        "Slider",
        "Checkbox",
        "Slider",
    ]
    assert [component.value for component in controls] == [
        False,
        "Depth",
        [],
        "depth_anything_v2",
        768,
        False,
        1.0,
    ]
    assert controls[3].kwargs["choices"] == [
        "None (already a depth map)",
        "depth_anything_v2",
    ]
    button_labels = [
        component.value
        for component in FakeComponent.registry
        if component.kind == "Button"
    ]
    assert button_labels == [
        "Remove selected",
        "Move up",
        "Move down",
        "Clear list",
        "Re-detect selected",
        "Preview selected",
        "Cache all previews",
        "Download / verify models needed by the list",
    ]
    assert not any(label in button_labels for label in ("Generate", "Skip", "Interrupt"))

    preview_events = [
        component.events[0]
        for component in FakeComponent.registry
        if component.kind == "Button"
        and component.value == "Preview selected"
        and component.events
    ]
    assert len(preview_events) == 1
    assert len(preview_events[0]["outputs"]) == 3


def test_preview_builds_a_grid_at_hires_final_size_and_warns_on_ratio(
    monkeypatch, tmp_path
):
    module = load_script(monkeypatch, tmp_path)
    wide = np.zeros((4, 8, 3), dtype=np.uint8)
    square = np.zeros((4, 4, 3), dtype=np.uint8)

    previews, status = module._preview(
        module.DEPTH_MODE,
        [(wide, None), (square, "square")],
        "None (already a depth map)",
        768,
        False,
        64,
        32,
        True,
        2.0,
        0,
        0,
    )

    assert len(previews) == 2
    assert [preview[0].shape for preview in previews] == [
        (64, 128, 3),
        (64, 128, 3),
    ]
    assert all("128×64" in preview[1] for preview in previews)
    assert status.startswith("⚠️")
    assert len(module.gr.warnings) == 1


@pytest.mark.parametrize(
    ("mode", "preprocessor"),
    [
        ("Depth", "None (already a depth map)"),
        ("Depth", "depth_anything_v2"),
        ("Pose / OpenPose", "None (already an OpenPose map)"),
        ("Pose / OpenPose", "DWPose (photo to pose)"),
    ],
)
@pytest.mark.parametrize("source_count", [1, 3])
def test_preview_routes_every_single_or_multi_source_through_all_four_paths(
    monkeypatch, tmp_path, mode, preprocessor, source_count
):
    module = load_script(
        monkeypatch, tmp_path, preprocessors={"depth_anything_v2": object()}
    )
    calls = []

    def create(selected_mode, source, selected_preprocessor, *_args):
        calls.append((selected_mode, int(source[0, 0, 0]), selected_preprocessor))
        return np.full((32, 64, 3), int(source[0, 0, 0]), dtype=np.uint8)

    monkeypatch.setattr(module, "_create_control_map", create)
    sources = [
        np.full((4, 8, 3), index + 1, dtype=np.uint8)
        for index in range(source_count)
    ]

    previews, _status = module._preview(
        mode, sources, preprocessor, 768, False, 64, 32
    )

    assert len(previews) == source_count
    assert calls == [
        (mode, index + 1, preprocessor) for index in range(source_count)
    ]


def test_active_preview_grid_labels_the_exact_current_batch(monkeypatch, tmp_path):
    module = load_script(monkeypatch, tmp_path)
    maps = [
        np.full((40, 80, 3), 20, dtype=np.uint8),
        np.full((40, 80, 3), 200, dtype=np.uint8),
    ]

    module._publish_active_control_preview(maps, [2, 0], "Depth")

    assigned = module.shared.state.assigned_previews
    assert len(assigned) == 1
    assert assigned[0].mode == "RGB"
    assert assigned[0].size == (160, 66)
    assert module.shared.state.textinfo.endswith("3, 1")


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
        if component.kind == "Markdown" and str(component.value).startswith("Depth:")
    ]
    assert len(statuses) == 1
    assert "present" in statuses[0]
    assert "Every file is hash-verified before use" in statuses[0]
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
            module.DEPTH_MODE,
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
        module.DEPTH_MODE,
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


@pytest.mark.parametrize(
    "preprocessor", ["None (already a depth map)", "depth_anything_v2"]
)
def test_valid_api_values_apply_control_without_touching_forge_state(
    monkeypatch, tmp_path, preprocessor
):
    module = load_script(
        monkeypatch, tmp_path, preprocessors={"depth_anything_v2": object()}
    )
    calls = install_processing_fakes(module, monkeypatch)
    process = make_process()

    module.Krea2DepthControlScript().process_before_every_sampling(
        process,
        True,
        module.DEPTH_MODE,
        "image",
        preprocessor,
        768,
        False,
        0.75,
    )

    assert calls.depth == [("image", preprocessor, 768, False)]
    assert calls.applied[0][0] == "base-unet"
    assert calls.applied[0][1] == (1, 1, 2, 2)
    assert calls.applied[0][3] == 0.75
    assert process.sd_model.forge_objects.unet == "controlled-unet"
    assert process.extra_generation_params["Krea 2 Depth Strength"] == 0.75
    assert not calls.guarded


@pytest.mark.parametrize(
    "preprocessor", ["None (already a depth map)", "depth_anything_v2"]
)
def test_multiple_images_alternate_inside_and_across_batches(
    monkeypatch, tmp_path, preprocessor
):
    module = load_script(
        monkeypatch, tmp_path, preprocessors={"depth_anything_v2": object()}
    )
    calls = install_processing_fakes(module, monkeypatch)
    process = make_process()
    process.batch_size = 2
    process.n_iter = 2
    process.iteration = 1

    module.Krea2DepthControlScript().process_before_every_sampling(
        process,
        True,
        module.DEPTH_MODE,
        ["A", "B", "C"],
        preprocessor,
        768,
        False,
        1.0,
        x=torch.zeros(2, 1, 1, 1, 1),
    )

    assert [call[0] for call in calls.depth] == ["A", "B", "C"]
    assert {call[1] for call in calls.depth} == {preprocessor}
    assert calls.applied[0][1][0] == 2
    assert process.extra_generation_params["Krea 2 Control Files"] == 3
    assert process.extra_generation_params["Krea 2 Control Sequence"] == "A, B, C, A…"


def test_ratio_warning_does_not_block_generation(monkeypatch, tmp_path):
    module = load_script(monkeypatch, tmp_path)
    calls = install_processing_fakes(module, monkeypatch)
    monkeypatch.setattr(module, "ratio_warning", lambda *_args: "ratio differs")
    process = make_process()

    module.Krea2DepthControlScript().process_before_every_sampling(
        process,
        True,
        module.DEPTH_MODE,
        "image",
        "None (already a depth map)",
        768,
        False,
        1.0,
    )

    assert process.sd_model.forge_objects.unet == "controlled-unet"
    assert module.gr.warnings == ["ratio differs"]
    assert not calls.guarded


def test_missing_preprocessor_fails_closed_before_processing(monkeypatch, tmp_path):
    module = load_script(monkeypatch, tmp_path)
    calls = install_processing_fakes(module, monkeypatch)
    process = make_process()

    with pytest.raises(ValueError, match="preprocessor"):
        module.Krea2DepthControlScript().process_before_every_sampling(
            process, True, module.DEPTH_MODE, "image", None, 768, False, 1.0
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
            module.DEPTH_MODE,
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
            module.DEPTH_MODE,
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
        module.DEPTH_MODE,
        "image",
        "None (already a depth map)",
        768,
        False,
        1.0,
    )
    assert resumed.sd_model.forge_objects.unet == "controlled-unet"


def test_hires_pass_applies_the_exact_runtime_dimensions(monkeypatch, tmp_path):
    module = load_script(monkeypatch, tmp_path)
    install_processing_fakes(module, monkeypatch)
    process = make_process()
    process.is_hr_pass = True
    process.hr_upscale_to_x = 96
    process.hr_upscale_to_y = 48

    module.Krea2DepthControlScript().process_before_every_sampling(
        process,
        True,
        module.DEPTH_MODE,
        "image",
        "None (already a depth map)",
        768,
        False,
        1.0,
    )

    assert process.sd_model.encoded.shape[1:3] == (48, 96)


@pytest.mark.parametrize(
    "preprocessor",
    ["None (already an OpenPose map)", "DWPose (photo to pose)"],
)
def test_pose_batch_alternates_refs_conditions_each_image_and_cleans_up(
    monkeypatch, tmp_path, preprocessor
):
    module = load_script(monkeypatch, tmp_path)
    selected = []
    conditioned = []
    values = {"A": 1, "B": 2, "C": 3}

    def preprocess_pose(source, _preprocessor, _resolution, _models):
        selected.append(source)
        return np.full((8, 8, 3), values[source], dtype=np.uint8)

    monkeypatch.setattr(module, "normalize_images", lambda images: images)
    monkeypatch.setattr(module, "ratio_warning", lambda *_args: "")
    monkeypatch.setattr(module, "preprocess_pose_map", preprocess_pose)
    monkeypatch.setattr(
        module,
        "encode_pose_references",
        lambda _engine, image: (
            image.amax(dim=(1, 2, 3)).reshape(image.shape[0], 1, 1, 1),
            torch.ones(image.shape[0], 16, 2, 2),
        ),
    )
    monkeypatch.setattr(module, "load_pose_state_dict", lambda _path: {"pose": True})
    monkeypatch.setattr(
        module,
        "build_pose_prompt_conditioning",
        lambda _engine, prompts, vision, *_args, **kwargs: conditioned.append(
            (list(prompts), vision[:, 0, 0, 0].tolist(), kwargs.get("multicond"))
        )
        or ("conditioning" if kwargs.get("multicond") else "negative"),
    )
    monkeypatch.setattr(
        module,
        "apply_pose_control",
        lambda unet, latent, state, strength: (
            "pose-unet",
            unet,
            tuple(latent.shape),
            state,
            strength,
        ),
    )
    sys.modules["modules"].sd_samplers = types.SimpleNamespace(
        find_sampler_config=lambda _name: None
    )

    class Krea2:
        def __init__(self):
            self.forge_objects = types.SimpleNamespace(unet="base-unet")

        def set_clip_skip(self, _value):
            pass

    original_setup = lambda: None
    process = types.SimpleNamespace(
        sd_model=Krea2(),
        width=64,
        height=32,
        enable_hr=False,
        batch_size=2,
        n_iter=2,
        iteration=1,
        setup_conds=original_setup,
        prompts=["one", "two"],
        negative_prompts=["", ""],
        sampler_name="Euler",
        steps=10,
        cfg_scale=2,
        distilled_cfg_scale=1.0,
        cached_c=[None, None, None],
        cached_uc=[None, None, None],
        extra_generation_params={},
    )
    script = module.Krea2DepthControlScript()

    script.process_batch(
        process,
        True,
        module.POSE_MODE,
        ["A", "B", "C"],
        preprocessor,
        512,
        False,
        0.85,
    )
    process.setup_conds()
    script.process_before_every_sampling(
        process,
        True,
        module.POSE_MODE,
        ["A", "B", "C"],
        preprocessor,
        512,
        False,
        0.85,
    )

    assert set(selected) == {"A", "B", "C"}
    assert conditioned[0][0] == ["", ""]
    assert conditioned[0][1] == pytest.approx([3 / 255, 1 / 255])
    assert conditioned[0][2] is not True
    assert conditioned[1][0] == ["one", "two"]
    assert conditioned[1][1] == pytest.approx([3 / 255, 1 / 255])
    assert conditioned[1][2] is True
    assert process.sd_model.forge_objects.unet[0] == "pose-unet"
    assert process.sd_model.forge_objects.unet[-1] == 0.85
    assert process.extra_generation_params["Krea 2 Control Sequence"] == "A, B, C, A…"

    script.postprocess_batch(process)
    assert process.setup_conds is original_setup
    assert not hasattr(process, "_krea2_pose_context")


def test_list_auto_detects_each_file_and_inherits_current_settings(
    monkeypatch, tmp_path
):
    module = load_script(
        monkeypatch, tmp_path, preprocessors={"depth_anything_v2": object()}
    )
    detections = {
        "pose.png": types.SimpleNamespace(
            kind=module.OPENPOSE_MAP, confidence=0.91, reason="coloured skeleton"
        ),
        "depth.png": types.SimpleNamespace(
            kind=module.DEPTH_MAP, confidence=0.93, reason="smooth grayscale"
        ),
        "photo.png": types.SimpleNamespace(
            kind=module.PHOTO, confidence=0.88, reason="ordinary image"
        ),
    }
    monkeypatch.setattr(module, "detect_control_kind", detections.__getitem__)

    result = module._add_entries(
        list(detections),
        [],
        module.POSE_MODE,
        module.DWPOSE,
        1024,
        False,
        0.65,
        {},
    )
    entries, selected_index = result[:2]

    assert selected_index == 2
    assert [(entry["mode"], entry["preprocessor"]) for entry in entries] == [
        (module.POSE_MODE, module.POSE_DIRECT),
        (module.DEPTH_MODE, module.DEPTH_DIRECT),
        (module.POSE_MODE, module.DWPOSE),
    ]
    assert all(entry["resolution"] == 1024 for entry in entries)
    assert all(entry["strength"] == 0.65 for entry in entries)
    assert len({entry["id"] for entry in entries}) == 3


def test_low_confidence_detection_never_overrides_current_settings(
    monkeypatch, tmp_path
):
    module = load_script(monkeypatch, tmp_path)
    monkeypatch.setattr(
        module,
        "detect_control_kind",
        lambda _source: types.SimpleNamespace(
            kind=module.DEPTH_MAP, confidence=0.7, reason="ambiguous grayscale"
        ),
    )

    entry = module._new_entry(
        "ambiguous.png", module.POSE_MODE, module.DWPOSE, 768, False, 1.0
    )

    assert entry["mode"] == module.POSE_MODE
    assert entry["preprocessor"] == module.DWPOSE
    assert entry["detection_applied"] is False


def test_per_file_edit_select_reorder_remove_and_cache_invalidation(
    monkeypatch, tmp_path
):
    module = load_script(
        monkeypatch, tmp_path, preprocessors={"depth_anything_v2": object()}
    )
    monkeypatch.setattr(
        module,
        "detect_control_kind",
        lambda _source: types.SimpleNamespace(
            kind=module.PHOTO, confidence=0.9, reason="photo"
        ),
    )
    entries = [
        module._new_entry(name, module.DEPTH_MODE, "depth_anything_v2", 768, False, 1.0)
        for name in ("A.png", "B.png", "C.png")
    ]
    cache = {
        entries[1]["id"]: {
            "signature": module._preview_signature(entries[1]),
            "image": np.ones((2, 2, 3), dtype=np.uint8),
        }
    }
    event = types.SimpleNamespace(index=(1, 4))

    selected = module._select_entry(entries, cache, event)
    assert selected[0] == 1
    assert selected[3]["value"].shape == (2, 2, 3)

    edited, _table, cache, preview, preview_status = module._change_entry_settings(
        entries, 1, "depth_anything_v2", 1024, True, 0.55, cache
    )
    assert edited[0]["resolution"] == 768
    assert edited[1]["resolution"] == 1024
    assert edited[1]["strength"] == 0.55
    assert entries[1]["resolution"] == 768
    assert edited[1]["id"] not in cache
    assert preview is None
    assert "cleared" in preview_status

    moved = module._move_entry(edited, 1, cache, -1)
    moved_entries, moved_index = moved[:2]
    assert moved_index == 0
    assert [entry["name"] for entry in moved_entries] == ["B.png", "A.png", "C.png"]

    removed = module._remove_entry(moved_entries, moved_index, cache)
    remaining, remaining_index = removed[:2]
    assert remaining_index == 0
    assert [entry["name"] for entry in remaining] == ["A.png", "C.png"]

    cleared_cache, cleared_preview, status = module._invalidate_all_previews(
        remaining, remaining_index
    )
    assert cleared_cache == {}
    assert cleared_preview is None
    assert "dimensions changed" in status


def test_strength_only_edit_keeps_the_existing_preview(monkeypatch, tmp_path):
    module = load_script(monkeypatch, tmp_path)
    entry = {
        "id": "one",
        "source": "one.png",
        "name": "one",
        "mode": module.DEPTH_MODE,
        "preprocessor": module.DEPTH_DIRECT,
        "resolution": 768,
        "invert": False,
        "strength": 1.0,
    }
    image = np.ones((2, 2, 3), dtype=np.uint8)
    cache = {
        "one": {
            "signature": module._preview_signature(entry),
            "image": image,
            "dimensions": (512, 512),
        }
    }

    entries, _table, updated_cache, preview, _status = module._change_entry_settings(
        [entry], 0, module.DEPTH_DIRECT, 768, False, 0.4, cache
    )

    assert entries[0]["strength"] == 0.4
    assert updated_cache is cache
    assert preview is image


def test_batch_validation_accepts_per_file_settings_sequentially_only(
    monkeypatch, tmp_path
):
    module = load_script(monkeypatch, tmp_path)
    entries = [
        {"mode": module.DEPTH_MODE, "strength": 0.6},
        {"mode": module.POSE_MODE, "strength": 1.1},
    ]

    assert module._validate_schedule(entries, 1, 2) == [[0], [1]]
    with pytest.raises(ValueError, match="Batch size to 1"):
        module._validate_schedule(entries, 2, 1)


def test_generation_cache_reuses_duplicate_processing_and_cleans_after_generation(
    monkeypatch, tmp_path
):
    module = load_script(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(
        module,
        "preprocess_depth_map",
        lambda source, preprocessor, resolution, invert: calls.append(
            (source, preprocessor, resolution, invert)
        )
        or np.full((4, 8, 3), 90, dtype=np.uint8),
    )
    monkeypatch.setattr(module, "load_control_state_dict", lambda _path: {"ok": True})
    entries = [
        {
            "id": f"id-{index}",
            "source": "same.png",
            "name": f"copy {index}",
            "mode": module.DEPTH_MODE,
            "preprocessor": module.DEPTH_DIRECT,
            "resolution": 768,
            "invert": False,
            "strength": 1.0,
        }
        for index in range(2)
    ]
    process = make_process()
    process.n_iter = 2

    cache = module._prepare_generation_cache(process, entries)

    assert len(calls) == 1
    assert set(cache["processed"]) == {"id-0", "id-1"}
    assert cache["previews"]["id-0"].shape == (32, 64, 3)
    module.Krea2DepthControlScript().postprocess(process, None)
    assert not hasattr(process, "_krea2_control_cache")


def test_preprocessor_cache_survives_generation_cleanup(monkeypatch, tmp_path):
    module = load_script(monkeypatch, tmp_path)
    calls = []
    source = np.full((5, 7, 3), 42, dtype=np.uint8)
    monkeypatch.setattr(
        module,
        "preprocess_depth_map",
        lambda *_args: calls.append(True)
        or np.full((4, 8, 3), 90, dtype=np.uint8),
    )
    monkeypatch.setattr(module, "load_control_state_dict", lambda _path: {"ok": True})
    entry = {
        "id": "same-entry",
        "source": source,
        "name": "same image",
        "mode": module.DEPTH_MODE,
        "preprocessor": module.DEPTH_DIRECT,
        "resolution": 768,
        "invert": False,
        "strength": 1.0,
    }

    first = make_process()
    module._prepare_generation_cache(first, [entry])
    module._cleanup_control_cache(first)
    second = make_process()
    module._prepare_generation_cache(second, [entry])

    assert len(calls) == 1
    assert module._preprocessor_cache_info()["hits"] == 1
    assert second._krea2_control_cache["processed"]["same-entry"].shape == (4, 8, 3)


def test_preview_primes_the_generation_preprocessor_cache(monkeypatch, tmp_path):
    module = load_script(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(
        module,
        "preprocess_depth_map",
        lambda *_args: calls.append(True)
        or np.full((4, 8, 3), 90, dtype=np.uint8),
    )
    monkeypatch.setattr(module, "load_control_state_dict", lambda _path: {"ok": True})
    entry = {
        "id": "previewed",
        "source": np.full((5, 7, 3), 42, dtype=np.uint8),
        "name": "previewed image",
        "mode": module.DEPTH_MODE,
        "preprocessor": module.DEPTH_DIRECT,
        "resolution": 768,
        "invert": False,
        "strength": 1.0,
    }

    preview, _warning, _width, _height = module._preview_entry(entry, 64, 32)
    process = make_process()
    cache = module._prepare_generation_cache(process, [entry])

    assert preview.shape == (32, 64, 3)
    assert cache["previews"]["previewed"].shape == (32, 64, 3)
    assert len(calls) == 1


def test_changed_pixels_or_settings_do_not_reuse_stale_preprocessing(
    monkeypatch, tmp_path
):
    module = load_script(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(
        module,
        "preprocess_depth_map",
        lambda image, _preprocessor, resolution, invert: calls.append(
            (int(np.asarray(image)[0, 0, 0]), resolution, invert)
        )
        or np.full((4, 8, 3), len(calls), dtype=np.uint8),
    )
    monkeypatch.setattr(module, "load_control_state_dict", lambda _path: {"ok": True})

    def entry(pixels, resolution=768, invert=False):
        return {
            "id": "stable-id",
            "source": np.full((5, 7, 3), pixels, dtype=np.uint8),
            "name": "changing image",
            "mode": module.DEPTH_MODE,
            "preprocessor": module.DEPTH_DIRECT,
            "resolution": resolution,
            "invert": invert,
            "strength": 1.0,
        }

    for item in (entry(1), entry(2), entry(2, 1024), entry(2, 1024, True)):
        module._prepare_generation_cache(make_process(), [item])

    assert calls == [(1, 768, False), (2, 768, False), (2, 1024, False), (2, 1024, True)]


def test_file_rewritten_at_same_path_invalidates_preview_signature(
    monkeypatch, tmp_path
):
    module = load_script(monkeypatch, tmp_path)
    path = tmp_path / "control.png"
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(path)
    entry = {
        "id": "file",
        "source": str(path),
        "mode": module.DEPTH_MODE,
        "preprocessor": module.DEPTH_DIRECT,
        "resolution": 768,
        "invert": False,
    }
    cache = {
        "file": {
            "signature": module._preview_signature(entry),
            "image": np.ones((2, 2, 3), dtype=np.uint8),
        }
    }

    Image.fromarray(np.full((3, 4, 3), 255, dtype=np.uint8)).save(path)

    assert module._cache_for_entry(cache, entry) is None


def test_pose_preprocessor_cache_survives_generate_cleanup(monkeypatch, tmp_path):
    module = load_script(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(
        module,
        "preprocess_pose_map",
        lambda *_args: calls.append(True)
        or np.full((6, 4, 3), 120, dtype=np.uint8),
    )
    monkeypatch.setattr(module, "load_pose_state_dict", lambda _path: {"pose": True})
    entry = {
        "id": "pose",
        "source": np.full((8, 5, 3), 33, dtype=np.uint8),
        "name": "pose photo",
        "mode": module.POSE_MODE,
        "preprocessor": module.DWPOSE,
        "resolution": 768,
        "invert": False,
        "strength": 1.0,
    }

    first = make_process()
    module._prepare_generation_cache(first, [entry])
    module._cleanup_control_cache(first)
    module._prepare_generation_cache(make_process(), [entry])

    assert len(calls) == 1


def test_sequence_cache_reuses_content_duplicates_and_next_generate(
    monkeypatch, tmp_path
):
    module = load_script(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(
        module,
        "preprocess_depth_map",
        lambda image, *_args: calls.append(int(np.asarray(image)[0, 0, 0]))
        or np.asarray(image),
    )
    monkeypatch.setattr(module, "load_control_state_dict", lambda _path: {"ok": True})

    def entry(entry_id, value):
        return {
            "id": entry_id,
            "source": np.full((5, 7, 3), value, dtype=np.uint8),
            "name": entry_id,
            "mode": module.DEPTH_MODE,
            "preprocessor": module.DEPTH_DIRECT,
            "resolution": 768,
            "invert": False,
            "strength": 1.0,
        }

    entries = [entry("A", 10), entry("B", 20), entry("A-copy", 10)]
    for _ in range(2):
        process = make_process()
        process.n_iter = 3
        module._prepare_generation_cache(process, entries)
        module._cleanup_control_cache(process)

    assert calls == [10, 20]


def test_cache_all_previews_reuses_valid_results(monkeypatch, tmp_path):
    module = load_script(monkeypatch, tmp_path)
    calls = []
    entry = {
        "id": "one",
        "source": "one.png",
        "name": "one",
        "mode": module.DEPTH_MODE,
        "preprocessor": module.DEPTH_DIRECT,
        "resolution": 768,
        "invert": False,
        "strength": 1.0,
    }
    monkeypatch.setattr(
        module,
        "_preview_entry",
        lambda *_args, **_kwargs: calls.append(True)
        or (np.ones((32, 64, 3), dtype=np.uint8), "", 64, 32),
    )

    _selected, _status, cache = module._cache_all_previews(
        [entry], 0, {}, 64, 32
    )
    selected, _status, cache = module._cache_all_previews(
        [entry], 0, cache, 64, 32
    )

    assert len(calls) == 1
    assert selected.shape == (32, 64, 3)


def test_depth_preprocessor_is_cached_across_first_and_hires_sampling_passes(
    monkeypatch, tmp_path
):
    module = load_script(monkeypatch, tmp_path)
    calls = install_processing_fakes(module, monkeypatch)
    process = make_process()
    process.enable_hr = True
    process.hr_scale = 2.0
    process.hr_resize_x = 0
    process.hr_resize_y = 0
    process.hr_upscale_to_x = 128
    process.hr_upscale_to_y = 64
    script = module.Krea2DepthControlScript()

    script.process(
        process,
        True,
        module.DEPTH_MODE,
        "image",
        module.DEPTH_DIRECT,
        768,
        False,
        1.0,
    )
    script.process_batch(
        process,
        True,
        module.DEPTH_MODE,
        "image",
        module.DEPTH_DIRECT,
        768,
        False,
        1.0,
    )
    script.process_before_every_sampling(
        process,
        True,
        module.DEPTH_MODE,
        "image",
        module.DEPTH_DIRECT,
        768,
        False,
        1.0,
    )
    assert process.sd_model.encoded.shape[1:3] == (32, 64)

    process.is_hr_pass = True
    script.process_before_every_sampling(
        process,
        True,
        module.DEPTH_MODE,
        "image",
        module.DEPTH_DIRECT,
        768,
        False,
        1.0,
    )

    assert process.sd_model.encoded.shape[1:3] == (64, 128)
    assert len(calls.depth) == 1
    assert len(calls.applied) == 2
    script.postprocess(process, None)
    assert not hasattr(process, "_krea2_control_cache")


def test_interruption_during_cache_build_skips_control_and_is_cleanable(
    monkeypatch, tmp_path
):
    module = load_script(monkeypatch, tmp_path)
    module.shared.state.interrupted = True
    monkeypatch.setattr(
        module,
        "preprocess_depth_map",
        lambda *_args: pytest.fail("preprocessing should stop on interruption"),
    )
    entry = {
        "id": "one",
        "source": "one.png",
        "name": "one",
        "mode": module.DEPTH_MODE,
        "preprocessor": module.DEPTH_DIRECT,
        "resolution": 768,
        "invert": False,
        "strength": 1.0,
    }
    process = make_process()

    cache = module._prepare_generation_cache(process, [entry])
    assert cache["cancelled"] is True
    module.Krea2DepthControlScript().process_batch(
        process,
        True,
        module.DEPTH_MODE,
        [entry],
        module.DEPTH_DIRECT,
        768,
        False,
        1.0,
    )
    assert not hasattr(process, "_krea2_control_active")
    module.Krea2DepthControlScript().postprocess(process, None)
    assert not hasattr(process, "_krea2_control_cache")


def test_duplicate_api_ids_are_made_unique(monkeypatch, tmp_path):
    module = load_script(monkeypatch, tmp_path)
    records = [
        {
            "id": "duplicate",
            "source": np.zeros((2, 2, 3), dtype=np.uint8),
            "mode": module.DEPTH_MODE,
            "preprocessor": module.DEPTH_DIRECT,
            "resolution": 768,
            "invert": False,
            "strength": 1.0,
        }
        for _ in range(2)
    ]

    entries = module._generation_entries(
        records, module.DEPTH_MODE, module.DEPTH_DIRECT, 768, False, 1.0
    )

    assert [entry["id"] for entry in entries] == ["duplicate", "duplicate-1"]
