"""Always-on Forge UI for the Krea 2 depth ControlNet-LoRA."""

from __future__ import annotations

import os

import gradio as gr
import torch

from forge_krea2_depth.adapter import (
    apply_depth_control,
    control_model_path,
    download_control_model,
    install_failure_guard,
    load_control_state_dict,
)
from forge_krea2_depth.images import create_depth_map, generation_dimensions
from modules import paths, scripts
from modules.ui_components import InputAccordion


def _preprocessor_choices() -> list[str]:
    from modules_forge.shared import supported_preprocessors

    preferred = ("depth_anything_v2", "depth_anything", "depth_midas")
    return ["None (already a depth map)"] + [
        name for name in preferred if name in supported_preprocessors
    ]


def _preview(image, preprocessor, resolution, invert):
    return create_depth_map(image, preprocessor, resolution, 512, 512, invert)


def _download_model():
    path = download_control_model(paths.models_path)
    return f"Model: `{path}` — ready"


class Krea2DepthControlScript(scripts.ScriptBuiltinUI):
    sorting_priority = 18110

    def title(self):
        return "Krea 2 Depth ControlNet-LoRA"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        del is_img2img
        choices = _preprocessor_choices()
        model_path = control_model_path(paths.models_path)
        with InputAccordion(False, label=self.title()) as enabled:
            gr.Markdown(
                "Structural depth control made specifically for Krea 2. "
                "Forge keeps ownership of Generate, Skip and Interrupt."
            )
            with gr.Row():
                control_image = gr.Image(
                    label="Source image or depth map",
                    type="numpy",
                    sources=["upload", "clipboard"],
                )
                preview = gr.Image(label="Depth preview", interactive=False)
            with gr.Row():
                preprocessor = gr.Dropdown(
                    choices=choices,
                    value="depth_anything_v2"
                    if "depth_anything_v2" in choices
                    else choices[0],
                    label="Depth preprocessor",
                )
                resolution = gr.Slider(
                    minimum=256,
                    maximum=2048,
                    value=768,
                    step=64,
                    label="Preprocessor resolution",
                )
                invert = gr.Checkbox(False, label="Invert depth map")
            with gr.Row():
                strength = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.05,
                    label="Control strength",
                )
                preview_button = gr.Button("Preview depth", variant="secondary")
                download_button = gr.Button(
                    "Download / verify model (862 MB)", variant="secondary"
                )
            model_status = gr.Markdown(
                f"Model: `{model_path}` — "
                + ("ready" if model_path.is_file() else "missing")
            )
        model_status.do_not_save_to_config = True
        preview_button.do_not_save_to_config = True
        download_button.do_not_save_to_config = True
        preview.do_not_save_to_config = True
        preview_button.click(
            fn=_preview,
            inputs=[control_image, preprocessor, resolution, invert],
            outputs=[preview],
        )
        download_button.click(fn=_download_model, inputs=[], outputs=[model_status])
        return (
            enabled,
            control_image,
            preprocessor,
            resolution,
            invert,
            strength,
        )

    @torch.inference_mode()
    def process_before_every_sampling(
        self,
        p,
        enabled,
        control_image,
        preprocessor,
        resolution,
        invert,
        strength,
        **kwargs,
    ):
        del kwargs
        if not enabled:
            return
        if float(strength) == 0.0:
            return
        try:
            if type(p.sd_model).__name__ != "Krea2":
                raise TypeError(
                    "Krea 2 Depth ControlNet-LoRA requires a Krea 2 checkpoint."
                )

            width, height = generation_dimensions(p)
            depth = create_depth_map(
                control_image,
                str(preprocessor),
                int(resolution),
                width,
                height,
                bool(invert),
            )
            image = torch.from_numpy(depth.copy()).float().div_(255.0).unsqueeze(0)
            _, latent = p.sd_model.encode_vision(image)
            if latent.ndim == 5 and latent.shape[2] == 1:
                latent = latent[:, :, 0]

            model_path = control_model_path(paths.models_path)
            state_dict = load_control_state_dict(model_path)
            p.sd_model.forge_objects.unet = apply_depth_control(
                p.sd_model.forge_objects.unet,
                latent,
                state_dict,
                float(strength),
            )
            p.extra_generation_params.update(
                {
                    "Krea 2 Depth Control": os.path.basename(model_path),
                    "Krea 2 Depth Preprocessor": str(preprocessor),
                    "Krea 2 Depth Strength": float(strength),
                }
            )
        except Exception as exc:
            install_failure_guard(p, exc)
            raise
