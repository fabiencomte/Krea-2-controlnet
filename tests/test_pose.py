from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

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
    monkeypatch.setattr(
        pose, "load_dwpose_detector", lambda _path: lambda _image: (candidates, scores)
    )

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
    parser.get_multicond_learned_conditioning = lambda model, conditioning, *_args: (
        types.SimpleNamespace(batch=[[model.get_learned_conditioning(conditioning)[0]]])
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
    monkeypatch.setattr(
        pose, "build_pose_lora_patches", lambda *_args: {"weight": object()}
    )

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
    controlled = pose.apply_pose_control(original, torch.zeros(1, 16, 4, 4), {}, 0.85)

    assert controlled is not original
    assert original.object_patches == {}
    assert controlled.patch_kwargs["strength_patch"] == 0.85
    assert controlled.patch_kwargs["online_mode"] is True
    assert "diffusion_model.forward" in controlled.object_patches


def test_kv_attention_appends_reference_keys_without_live_reference_queries(
    monkeypatch,
):
    observed = []
    fake_attention = types.ModuleType("backend.attention")

    def attention_function(q, k, v, _heads, **_kwargs):
        observed.append((q.detach().clone(), k.detach().clone(), v.detach().clone()))
        return q.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)

    fake_attention.attention_function = attention_function
    fake_quant = types.ModuleType("backend.quant_ops")
    rope_calls = []

    def apply_rope(query, key, frequencies):
        rope_calls.append((query.shape, key.shape, frequencies))
        return query, key

    fake_quant.ck = types.SimpleNamespace(apply_rope=apply_rope)
    monkeypatch.setitem(sys.modules, "backend.attention", fake_attention)
    monkeypatch.setitem(sys.modules, "backend.quant_ops", fake_quant)

    identity = lambda value: value
    attn = types.SimpleNamespace(
        wq=identity,
        wk=identity,
        wv=identity,
        gate=lambda value: torch.zeros_like(value),
        qknorm=lambda q, k: (q, k),
        wo=identity,
        heads=2,
        kvheads=1,
    )
    captured = []
    pose._attention_with_pose_kv(
        attn,
        torch.ones(1, 2, 4),
        "rope",
        kv_capture=captured,
    )
    result = pose._attention_with_pose_kv(
        attn,
        torch.full((1, 3, 4), 2.0),
        "rope",
        kv_cache=captured[0],
    )

    first_q, first_k, first_v = observed[0]
    live_q, combined_k, combined_v = observed[1]
    assert first_q.shape[2] == first_k.shape[2] == first_v.shape[2] == 2
    assert live_q.shape[2] == 3
    assert combined_k.shape[1] == combined_v.shape[1] == 2
    assert combined_k.shape[2] == combined_v.shape[2] == 5
    assert [call[2] for call in rope_calls] == ["rope", "rope"]
    assert torch.all(live_q == 2)
    assert torch.all(combined_k[:, :, :3] == 2)
    assert torch.all(combined_k[:, :, 3:] == 1)
    assert torch.all(combined_v[:, :, :3] == 2)
    assert torch.all(combined_v[:, :, 3:] == 1)
    assert result.shape == (1, 3, 4)


def test_pose_control_caches_isolated_reference_kv_and_resets_for_a_new_run(
    monkeypatch,
):
    fake_args = types.ModuleType("backend.args")
    fake_args.dynamic_args = types.SimpleNamespace(online_lora=True)
    monkeypatch.setitem(sys.modules, "backend.args", fake_args)
    monkeypatch.setattr(
        pose, "build_pose_lora_patches", lambda *_args: {"weight": object()}
    )
    precomputed = []
    forwarded = []

    def precompute(_dit, _x, timesteps, _refs, _options):
        precomputed.append(float(timesteps.max()))
        return [("cached-k", "cached-v")]

    def forward(_dit, x, _timesteps, _context, ref_kv, _options):
        forwarded.append(ref_kv)
        return x

    monkeypatch.setattr(pose, "_precompute_pose_ref_kv", precompute)
    monkeypatch.setattr(pose, "_forward_with_cached_pose_refs", forward)

    diffusion = types.SimpleNamespace(
        first=object(), blocks=[], patch=2, channels=16, txtfusion=object()
    )

    class FakeUnet:
        def __init__(self):
            self.model = types.SimpleNamespace(diffusion_model=diffusion)
            self.object_patches = {}

        def clone(self):
            return FakeUnet()

        def add_patches(self, patches, **_kwargs):
            return list(patches)

        def add_object_patch(self, name, value):
            self.object_patches[name] = value

    controlled = pose.apply_pose_control(FakeUnet(), torch.zeros(1, 16, 4, 4), {}, 0.85)
    model_forward = controlled.object_patches["diffusion_model.forward"]
    sample = torch.zeros(1, 16, 1, 4, 4)
    model_forward(sample, torch.tensor([1.0]), None)
    model_forward(sample, torch.tensor([0.5]), None)
    model_forward(sample, torch.tensor([0.9]), None)

    assert precomputed == [1.0, pytest.approx(0.9)]
    assert len(forwarded) == 3
    assert all(item == [("cached-k", "cached-v")] for item in forwarded)


def test_pose_reference_vae_encodes_batch_items_as_independent_images():
    calls = []

    class FakeVae:
        first_stage_model = types.SimpleNamespace(process_in=lambda value: value)

        def encode(self, pixels):
            calls.append(int(pixels.shape[0]))
            # Reproduce Forge's WanVAE behaviour: a multi-image call is treated
            # as one temporal sample, so only one batch item comes back.
            return pixels[:1, ..., :1].movedim(-1, 1).unsqueeze(2)

    engine = types.SimpleNamespace(forge_objects=types.SimpleNamespace(vae=FakeVae()))
    images = torch.stack(
        (
            torch.zeros(8, 8, 3),
            torch.ones(8, 8, 3),
        )
    )

    vision, latent = pose.encode_pose_references(engine, images)

    assert calls == [1, 1]
    assert vision.shape[0] == latent.shape[0] == 2
    assert latent[:, 0, 0, 0].tolist() == [0.0, 1.0]
    with pytest.raises(ValueError, match="At least one pose reference"):
        pose.encode_pose_references(engine, torch.empty(0, 8, 8, 3))


def test_pose_kv_block_applies_modulation_gates_and_preserves_cache_contract(
    monkeypatch,
):
    observed = {}

    def attention(_attn, value, frequencies, **kwargs):
        observed["attention_input"] = value.detach().clone()
        observed["frequencies"] = frequencies
        observed["capture"] = kwargs["kv_capture"]
        observed["cache"] = kwargs["kv_cache"]
        observed["options"] = kwargs["transformer_options"]
        return torch.full_like(value, 10.0)

    monkeypatch.setattr(pose, "_attention_with_pose_kv", attention)
    constants = tuple(torch.tensor(float(value)) for value in (1, 2, 3, 4, 5, 6))
    block = types.SimpleNamespace(
        mod=lambda _vector: constants,
        prenorm=lambda value: value,
        postnorm=lambda value: value,
        attn=object(),
        mlp=lambda value: value,
    )
    capture = []
    cache = ("reference-k", "reference-v")
    options = {"patches": "kept"}

    result = pose._block_with_pose_kv(
        block,
        torch.ones(1, 2, 1),
        torch.zeros(1, 1, 1),
        "rope",
        kv_capture=capture,
        kv_cache=cache,
        transformer_options=options,
    )

    assert torch.all(observed["attention_input"] == 4)
    assert observed["frequencies"] == "rope"
    assert observed["capture"] is capture
    assert observed["cache"] is cache
    assert observed["options"] is options
    assert torch.all(result == 991)


def test_pose_reference_kv_precompute_is_t0_per_block_and_batch_aligned(
    monkeypatch,
):
    fake_backend = types.ModuleType("backend")
    fake_backend_nn = types.ModuleType("backend.nn")
    fake_flux = types.ModuleType("backend.nn.flux")
    embedded = []

    def timestep_embedding(timesteps, dimension):
        embedded.append((timesteps.detach().clone(), dimension))
        return timesteps[:, None]

    fake_flux.timestep_embedding = timestep_embedding
    monkeypatch.setitem(sys.modules, "backend", fake_backend)
    monkeypatch.setitem(sys.modules, "backend.nn", fake_backend_nn)
    monkeypatch.setitem(sys.modules, "backend.nn.flux", fake_flux)

    calls = []
    blocks = [object(), object()]

    def block_forward(
        block,
        hidden,
        vector,
        frequencies,
        *,
        kv_capture,
        transformer_options,
        **_kwargs,
    ):
        index = blocks.index(block)
        calls.append(
            {
                "index": index,
                "hidden": hidden.detach().clone(),
                "vector": vector.detach().clone(),
                "frequencies": frequencies.detach().clone(),
                "options": transformer_options,
            }
        )
        kv_capture.append(
            (
                torch.full((hidden.shape[0], 1, 1, 1), 10.0 + index),
                torch.full((hidden.shape[0], 1, 1, 1), 20.0 + index),
            )
        )
        return hidden + 1

    monkeypatch.setattr(pose, "_block_with_pose_kv", block_forward)
    dit = types.SimpleNamespace(
        patch=1,
        first=lambda value: value + 5,
        tdim=1,
        tmlp=lambda value: value,
        tproj=lambda value: value,
        pe_embedder=lambda value: value + 7,
        blocks=blocks,
    )
    sample = torch.zeros(2, 1, 1, 1, 1)
    references = [torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1)]
    timesteps = torch.tensor([0.8, 0.6])
    options = {"runtime": "preserved"}

    cached = pose._precompute_pose_ref_kv(dit, sample, timesteps, references, options)

    assert len(embedded) == 1
    assert embedded[0][1] == 1
    assert torch.equal(embedded[0][0], torch.zeros_like(timesteps))
    assert [call["index"] for call in calls] == [0, 1]
    assert all(torch.count_nonzero(call["vector"]) == 0 for call in calls)
    assert all(call["options"] is options for call in calls)
    assert calls[0]["hidden"][:, 0, 0].tolist() == [6.0, 7.0]
    assert calls[1]["hidden"][:, 0, 0].tolist() == [7.0, 8.0]
    assert calls[0]["frequencies"][:, 0, 0].tolist() == [8.0, 8.0]
    assert [pair[0][0, 0, 0, 0].item() for pair in cached] == [10.0, 11.0]
    assert [pair[1][0, 0, 0, 0].item() for pair in cached] == [20.0, 21.0]


def test_cached_pose_forward_never_adds_reference_queries_and_restores_shape(
    monkeypatch,
):
    fake_backend = types.ModuleType("backend")
    fake_backend_nn = types.ModuleType("backend.nn")
    fake_flux = types.ModuleType("backend.nn.flux")
    fake_flux.timestep_embedding = lambda timesteps, _dimension: timesteps[:, None]
    monkeypatch.setitem(sys.modules, "backend", fake_backend)
    monkeypatch.setitem(sys.modules, "backend.nn", fake_backend_nn)
    monkeypatch.setitem(sys.modules, "backend.nn.flux", fake_flux)

    blocks = [object(), object()]
    calls = []
    positions = []

    def block_forward(
        block,
        hidden,
        _vector,
        frequencies,
        *,
        kv_cache,
        transformer_options,
        **_kwargs,
    ):
        calls.append(
            (
                blocks.index(block),
                int(hidden.shape[1]),
                kv_cache,
                frequencies,
                transformer_options,
            )
        )
        return hidden + 1

    monkeypatch.setattr(pose, "_block_with_pose_kv", block_forward)

    def embed(value):
        positions.append(value.detach().clone())
        return value

    dit = types.SimpleNamespace(
        patch=1,
        channels=1,
        first=lambda value: value,
        tdim=1,
        tmlp=lambda value: value,
        tproj=lambda value: value,
        txtfusion=lambda value, **_kwargs: value,
        txtmlp=lambda value: value,
        pe_embedder=embed,
        blocks=blocks,
        last=lambda value, _time: value,
    )
    image = torch.arange(6, dtype=torch.float32).reshape(1, 1, 1, 2, 3)
    context = torch.tensor([[[100.0], [200.0]]])
    cache = [("k0", "v0"), ("k1", "v1")]
    options = {"transformer": "kept"}

    result = pose._forward_with_cached_pose_refs(
        dit,
        image,
        torch.tensor([0.5]),
        context,
        cache,
        options,
    )

    assert result.shape == image.shape
    assert torch.equal(result[:, :, 0], image[:, :, 0] + 2)
    assert len(positions) == 1
    assert positions[0].shape == (1, 8, 3)
    assert [(index, length) for index, length, *_rest in calls] == [(0, 8), (1, 8)]
    assert [item[2] for item in calls] == cache
    assert all(item[4] is options for item in calls)


def test_cached_pose_forward_rejects_video_and_incomplete_cache(monkeypatch):
    fake_backend = types.ModuleType("backend")
    fake_backend_nn = types.ModuleType("backend.nn")
    fake_flux = types.ModuleType("backend.nn.flux")
    fake_flux.timestep_embedding = lambda timesteps, _dimension: timesteps[:, None]
    monkeypatch.setitem(sys.modules, "backend", fake_backend)
    monkeypatch.setitem(sys.modules, "backend.nn", fake_backend_nn)
    monkeypatch.setitem(sys.modules, "backend.nn.flux", fake_flux)

    dit = types.SimpleNamespace(
        patch=1,
        channels=1,
        first=lambda value: value,
        tdim=1,
        tmlp=lambda value: value,
        tproj=lambda value: value,
        txtfusion=lambda value, **_kwargs: value,
        txtmlp=lambda value: value,
        pe_embedder=lambda value: value,
        blocks=[object()],
        last=lambda value, _time: value,
    )
    context = torch.zeros(1, 1, 1)

    with pytest.raises(RuntimeError, match="image generation only"):
        pose._forward_with_cached_pose_refs(
            dit,
            torch.zeros(1, 1, 2, 2, 2),
            torch.tensor([0.5]),
            context,
            [("k", "v")],
            {},
        )
    with pytest.raises(RuntimeError, match="cache has 0 blocks; expected 1"):
        pose._forward_with_cached_pose_refs(
            dit,
            torch.zeros(1, 1, 1, 2, 2),
            torch.tensor([0.5]),
            context,
            [],
            {},
        )


@pytest.mark.parametrize("bad_output", ("temporal", "multi-batch"))
def test_pose_reference_vae_rejects_ambiguous_output_shapes(bad_output):
    class FakeVae:
        first_stage_model = types.SimpleNamespace(process_in=lambda value: value)

        def encode(self, pixels):
            if bad_output == "temporal":
                return torch.zeros(1, 1, 2, pixels.shape[1], pixels.shape[2])
            return torch.zeros(2, 1, pixels.shape[1], pixels.shape[2])

    engine = types.SimpleNamespace(forge_objects=types.SimpleNamespace(vae=FakeVae()))

    with pytest.raises(RuntimeError, match="Pose reference VAE"):
        pose.encode_pose_references(engine, torch.zeros(1, 8, 8, 3))
