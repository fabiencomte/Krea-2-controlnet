"""Always-on Forge UI for Krea 2 Depth and Pose ControlNet-LoRAs."""

from __future__ import annotations

import copy
import logging
import math
import os
import uuid

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw

from forge_krea2_depth.adapter import (
    apply_depth_control,
    control_model_path,
    download_control_model,
    install_failure_guard,
    load_control_state_dict,
)
from forge_krea2_depth.controls import (
    require_checkbox,
    require_mode,
    require_preprocessor,
    require_resolution,
    require_strength,
)
from forge_krea2_depth.detection import (
    DEPTH_MAP,
    OPENPOSE_MAP,
    PHOTO,
    detect_control_kind,
)
from forge_krea2_depth.images import (
    alternating_image_indices,
    fit_control_map,
    generation_dimensions,
    normalize_images,
    preprocess_depth_map,
    preview_dimensions,
    ratio_warning,
)
from forge_krea2_depth.pose import (
    DWPOSE_FILES,
    apply_pose_control,
    build_pose_prompt_conditioning,
    download_dwpose_models,
    download_pose_model,
    dwpose_model_dir,
    encode_pose_references,
    load_pose_state_dict,
    pose_model_path,
    preprocess_pose_map,
)
from forge_krea2_depth.preprocessor_cache import (
    ControlMapCache,
    build_control_map_cache_key,
    source_fingerprint,
)
from modules import paths, scripts, shared
from modules.ui_components import InputAccordion


logger = logging.getLogger("Krea2Control")
DEPTH_MODE = "Depth"
POSE_MODE = "Pose / OpenPose"
DEPTH_DIRECT = "None (already a depth map)"
POSE_DIRECT = "None (already an OpenPose map)"
DWPOSE = "DWPose (photo to pose)"
KIND_LABELS = {
    PHOTO: "Photo / image",
    DEPTH_MAP: "Depth map",
    OPENPOSE_MAP: "OpenPose map",
}
ENTRY_TABLE_HEADERS = [
    "#",
    "File",
    "Detected",
    "Mode",
    "Settings",
]
_CONTROL_MAP_CACHE = ControlMapCache()


def _entry_input_summary(entry) -> str:
    preprocessor = entry["preprocessor"]
    if preprocessor == DEPTH_DIRECT:
        preprocessor = "direct Depth"
    elif preprocessor == POSE_DIRECT:
        preprocessor = "direct OpenPose"
    elif preprocessor == DWPOSE:
        preprocessor = "DWPose"
    return (
        f"{preprocessor} · {entry['resolution']}"
        + (" · inverted" if entry.get("invert") else "")
        + f" · ×{float(entry['strength']):g}"
    )


def _control_preview_grid(
    control_maps, indices, mode: str, max_tile_edge: int = 384
) -> Image.Image:
    """Build a small labelled grid of the controls used by the current batch."""

    if not control_maps or len(control_maps) != len(indices):
        raise ValueError("Active control previews require one source index per map.")
    images = [Image.fromarray(np.asarray(item, dtype=np.uint8), mode="RGB") for item in control_maps]
    source_width, source_height = images[0].size
    scale = min(1.0, max_tile_edge / max(source_width, source_height))
    tile_width = max(1, round(source_width * scale))
    tile_height = max(1, round(source_height * scale))
    header_height = 26
    columns = min(4, max(1, math.ceil(math.sqrt(len(images)))))
    rows = math.ceil(len(images) / columns)
    canvas = Image.new(
        "RGB", (columns * tile_width, rows * (tile_height + header_height)), "black"
    )
    drawing = ImageDraw.Draw(canvas)
    for slot, (item, source_index) in enumerate(zip(images, indices)):
        column, row = slot % columns, slot // columns
        left = column * tile_width
        top = row * (tile_height + header_height)
        if item.size != (tile_width, tile_height):
            item = item.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        drawing.text(
            (left + 7, top + 6),
            f"{mode} ref {int(source_index) + 1} / batch {slot + 1}",
            fill="white",
        )
        canvas.paste(item, (left, top + header_height))
    return canvas


def _publish_active_control_preview(control_maps, indices, mode: str) -> None:
    """Expose current controls through Forge's normal live-preview channel."""

    state = getattr(shared, "state", None)
    assign = getattr(state, "assign_current_image", None)
    if not getattr(shared.opts, "live_previews_enable", False) or not callable(assign):
        return
    try:
        assign(_control_preview_grid(control_maps, indices, mode))
        references = ", ".join(str(int(index) + 1) for index in indices)
        state.textinfo = f"Krea 2 {mode}: active control reference(s) {references}"
    except Exception:
        logger.debug("Could not publish the active Krea 2 control preview.", exc_info=True)


def _depth_preprocessor_choices() -> list[str]:
    from modules_forge.shared import supported_preprocessors

    preferred = ("depth_anything_v2", "depth_anything", "depth_midas")
    return [DEPTH_DIRECT] + [
        name for name in preferred if name in supported_preprocessors
    ]


def _preprocessor_choices(mode: str) -> list[str]:
    return [DWPOSE, POSE_DIRECT] if mode == POSE_MODE else _depth_preprocessor_choices()


def _default_preprocessor(mode: str, requested=None) -> str:
    choices = _preprocessor_choices(mode)
    if requested in choices:
        return requested
    if mode == POSE_MODE:
        return DWPOSE
    return "depth_anything_v2" if "depth_anything_v2" in choices else choices[0]


def _uploaded_paths(value) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    paths_list = []
    for item in items:
        if isinstance(item, dict):
            item = item.get("path", item.get("name"))
        elif hasattr(item, "name"):
            item = item.name
        if item:
            paths_list.append(str(item))
    return paths_list


def _new_entry(
    source,
    mode,
    preprocessor,
    resolution,
    invert,
    strength,
):
    mode = require_mode(mode)
    preprocessor = _default_preprocessor(mode, preprocessor)
    detection = detect_control_kind(source)
    detection_applied = False
    if detection.kind == OPENPOSE_MAP and detection.confidence >= 0.75:
        mode, preprocessor, invert = POSE_MODE, POSE_DIRECT, False
        detection_applied = True
    elif detection.kind == DEPTH_MAP and detection.confidence >= 0.84:
        mode, preprocessor = DEPTH_MODE, DEPTH_DIRECT
        detection_applied = True
    elif preprocessor in (DEPTH_DIRECT, POSE_DIRECT):
        preprocessor = _default_preprocessor(mode)
    return {
        "id": uuid.uuid4().hex,
        "source": source,
        "name": os.path.basename(str(source)) or "control image",
        "detected_kind": detection.kind,
        "detection_confidence": round(float(detection.confidence), 3),
        "detection_reason": detection.reason,
        "detection_applied": detection_applied,
        "mode": mode,
        "preprocessor": preprocessor,
        "resolution": require_resolution(resolution),
        "invert": require_checkbox(invert, "invert") if mode == DEPTH_MODE else False,
        "strength": require_strength(strength),
    }


def _entry_table(entries, selected_index=-1):
    entries = entries or []
    rows = []
    for index, entry in enumerate(entries):
        rows.append(
            [
                f"▶ {index + 1}" if index == int(selected_index) else str(index + 1),
                entry.get("name", f"control {index + 1}"),
                f"{KIND_LABELS.get(entry.get('detected_kind'), 'Unknown')} "
                f"{round(float(entry.get('detection_confidence', 0)) * 100)}%",
                entry["mode"],
                _entry_input_summary(entry),
            ]
        )
    return rows


def _entry_status(entries, selected_index=-1, message="") -> str:
    entries = entries or []
    if not entries:
        return message or "Add one or more images. New files inherit the current settings."
    selected_index = max(0, min(int(selected_index), len(entries) - 1))
    entry = entries[selected_index]
    detected = KIND_LABELS.get(entry.get("detected_kind"), "Unknown")
    confidence = round(float(entry.get("detection_confidence", 0)) * 100)
    prefix = f"{message}  \n" if message else ""
    decision = (
        "Detection applied automatically."
        if entry.get("detection_applied")
        else "Conservative suggestion only; current settings were kept."
    )
    return (
        f"{prefix}**Selected {selected_index + 1}/{len(entries)} — {entry['name']}**  \n"
        f"Detected: **{detected} ({confidence}%)** — {entry.get('detection_reason', '')}  \n"
        f"{decision} The mode and preprocessor below always win."
    )


def _preview_signature(entry):
    return (
        source_fingerprint(entry["source"]),
        entry["mode"],
        entry["preprocessor"],
        entry["resolution"],
        entry["invert"],
    )


def _cache_for_entry(cache, entry, dimensions=None):
    cached = (cache or {}).get(entry["id"])
    if not cached:
        return None
    if cached.get("signature") != _preview_signature(entry):
        return None
    if dimensions is not None and tuple(cached.get("dimensions", ())) != tuple(dimensions):
        return None
    return cached.get("image")


def _selected_outputs(entries, selected_index, cache=None, message=""):
    entries = entries or []
    if not entries:
        return (
            _entry_table([], -1),
            None,
            None,
            "No selected file or cached preview.",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            _entry_status([], -1, message),
        )
    selected_index = max(0, min(int(selected_index), len(entries) - 1))
    entry = entries[selected_index]
    preview = _cache_for_entry(cache, entry)
    preview_status = (
        f"✅ Loaded cached preview for **{entry['name']}**."
        if preview is not None
        else f"No cached preview for **{entry['name']}**. Click **Preview selected** "
        "or **Cache all previews**."
    )
    return (
        _entry_table(entries, selected_index),
        gr.update(value=entry["source"], label=f"Selected source — {entry['name']}"),
        gr.update(value=preview),
        preview_status,
        gr.update(value=entry["mode"]),
        gr.update(
            choices=_preprocessor_choices(entry["mode"]),
            value=entry["preprocessor"],
            label=f"{entry['mode']} preprocessor",
        ),
        gr.update(value=entry["resolution"]),
        gr.update(value=entry["invert"], visible=entry["mode"] == DEPTH_MODE),
        gr.update(value=entry["strength"]),
        _entry_status(entries, selected_index, message),
    )


def _add_entries(
    uploads,
    entries,
    mode,
    preprocessor,
    resolution,
    invert,
    strength,
    cache,
):
    entries = copy.deepcopy(entries or [])
    added = []
    for source in _uploaded_paths(uploads):
        entry = _new_entry(
            source, mode, preprocessor, resolution, invert, strength
        )
        entries.append(entry)
        added.append(entry)
    selected_index = len(entries) - 1 if added else (0 if entries else -1)
    message = f"✅ Added {len(added)} file(s)." if added else "No file was added."
    selected = _selected_outputs(entries, selected_index, cache, message)
    return (entries, selected_index, *selected)


def _event_row(evt) -> int:
    index = getattr(evt, "index", -1)
    if isinstance(index, (tuple, list)):
        index = index[0]
    return int(index)


def _select_entry(entries, cache, evt: gr.SelectData):
    entries = copy.deepcopy(entries or [])
    selected_index = _event_row(evt)
    if not 0 <= selected_index < len(entries):
        selected_index = 0 if entries else -1
    return (selected_index, *_selected_outputs(entries, selected_index, cache))


def _remove_entry(entries, selected_index, cache):
    entries = copy.deepcopy(entries or [])
    cache = dict(cache or {})
    if not entries:
        return (entries, -1, cache, *_selected_outputs(entries, -1, cache))
    selected_index = max(0, min(int(selected_index), len(entries) - 1))
    removed = entries.pop(selected_index)
    cache.pop(removed["id"], None)
    selected_index = min(selected_index, len(entries) - 1)
    message = f"Removed {removed['name']}."
    return (
        entries,
        selected_index,
        cache,
        *_selected_outputs(entries, selected_index, cache, message),
    )


def _clear_entries():
    return ([], -1, {}, *_selected_outputs([], -1, {}, "List cleared."))


def _move_entry(entries, selected_index, cache, direction):
    entries = copy.deepcopy(entries or [])
    if not entries:
        return (entries, -1, *_selected_outputs(entries, -1, cache))
    selected_index = max(0, min(int(selected_index), len(entries) - 1))
    target = max(0, min(selected_index + int(direction), len(entries) - 1))
    if target != selected_index:
        entries[selected_index], entries[target] = entries[target], entries[selected_index]
        selected_index = target
    return (
        entries,
        selected_index,
        *_selected_outputs(entries, selected_index, cache),
    )


def _invalidate_cache(cache, entry_id):
    cache = dict(cache or {})
    cache.pop(entry_id, None)
    return cache


def _invalidate_all_previews(entries, selected_index):
    """Drop UI previews when Forge's requested output dimensions change."""

    del selected_index
    count = len(entries or [])
    return (
        {},
        None,
        f"Target dimensions changed; cleared cached previews for {count} file(s).",
    )


def _change_entry_mode(entries, selected_index, mode, cache):
    mode = require_mode(mode)
    entries = copy.deepcopy(entries or [])
    if not entries or not 0 <= int(selected_index) < len(entries):
        return (
            entries,
            _entry_table(entries, selected_index),
            gr.update(
                choices=_preprocessor_choices(mode),
                value=_default_preprocessor(mode),
                label=f"{mode} preprocessor",
            ),
            gr.update(visible=mode == DEPTH_MODE, value=False),
            cache or {},
            None,
            "No cached preview for the selected settings.",
            _entry_status(entries, selected_index),
        )
    entry = entries[int(selected_index)]
    entry["mode"] = mode
    entry["preprocessor"] = _default_preprocessor(mode)
    entry["invert"] = False
    cache = _invalidate_cache(cache, entry["id"])
    return (
        entries,
        _entry_table(entries, selected_index),
        gr.update(
            choices=_preprocessor_choices(mode),
            value=entry["preprocessor"],
            label=f"{mode} preprocessor",
        ),
        gr.update(visible=mode == DEPTH_MODE, value=False),
        cache,
        None,
        "Settings changed; the selected preview cache was cleared.",
        _entry_status(entries, selected_index, "Settings updated; preview invalidated."),
    )


def _change_entry_settings(
    entries,
    selected_index,
    preprocessor,
    resolution,
    invert,
    strength,
    cache,
):
    entries = copy.deepcopy(entries or [])
    if not entries or not 0 <= int(selected_index) < len(entries):
        return (
            entries,
            _entry_table(entries, selected_index),
            cache or {},
            None,
            "No selected file or cached preview.",
        )
    entry = entries[int(selected_index)]
    old_signature = _preview_signature(entry)
    preprocessor = require_preprocessor(preprocessor)
    if preprocessor not in _preprocessor_choices(entry["mode"]):
        raise ValueError(f"{preprocessor} is not valid in {entry['mode']} mode.")
    entry.update(
        {
            "preprocessor": preprocessor,
            "resolution": require_resolution(resolution),
            "invert": require_checkbox(invert, "invert")
            if entry["mode"] == DEPTH_MODE
            else False,
            "strength": require_strength(strength),
        }
    )
    if _preview_signature(entry) != old_signature:
        cache = _invalidate_cache(cache, entry["id"])
        preview = None
        preview_status = "Settings changed; the selected preview cache was cleared."
    else:
        cache = cache or {}
        preview = _cache_for_entry(cache, entry)
        preview_status = gr.update()
    return entries, _entry_table(entries, selected_index), cache, preview, preview_status


def _redetect_entry(entries, selected_index, cache):
    entries = copy.deepcopy(entries or [])
    if not entries or not 0 <= int(selected_index) < len(entries):
        return (entries, int(selected_index), cache or {}, *_selected_outputs(entries, selected_index, cache))
    old = entries[int(selected_index)]
    detected = _new_entry(
        old["source"],
        old["mode"],
        old["preprocessor"],
        old["resolution"],
        old["invert"],
        old["strength"],
    )
    detected["id"] = old["id"]
    entries[int(selected_index)] = detected
    cache = _invalidate_cache(cache, old["id"])
    return (
        entries,
        int(selected_index),
        cache,
        *_selected_outputs(entries, selected_index, cache, "Detection refreshed."),
    )


def _mode_ui(mode):
    mode = require_mode(mode)
    choices = _preprocessor_choices(mode)
    return (
        gr.update(choices=choices, value=choices[0], label=f"{mode} preprocessor"),
        gr.update(visible=mode == DEPTH_MODE),
    )


def _create_control_map(
    mode, image, preprocessor, resolution, width, height, invert
):
    source = _preprocess_control_map(
        mode, image, preprocessor, resolution, invert
    )
    return np.ascontiguousarray(fit_control_map(source, width, height))


def _preprocess_control_map(mode, image, preprocessor, resolution, invert):
    """Reuse raw Depth/DWPose maps across previews and Generate clicks."""

    key = build_control_map_cache_key(
        image,
        mode,
        preprocessor,
        resolution,
        invert if mode == DEPTH_MODE else False,
    )

    def compute():
        if mode == POSE_MODE:
            return preprocess_pose_map(
                image,
                preprocessor,
                resolution,
                paths.models_path,
            )
        return preprocess_depth_map(image, preprocessor, resolution, invert)

    result, _hit = _CONTROL_MAP_CACHE.get_or_compute(key, compute)
    return result


def _clear_preprocessor_cache() -> None:
    _CONTROL_MAP_CACHE.clear(reset_stats=True)


def _preprocessor_cache_info() -> dict[str, int]:
    return _CONTROL_MAP_CACHE.info()


def _preview(
    mode,
    images,
    preprocessor,
    resolution,
    invert,
    width,
    height,
    enable_hr=False,
    hr_scale=1.0,
    hr_resize_x=0,
    hr_resize_y=0,
):
    mode = require_mode(mode)
    preprocessor = require_preprocessor(preprocessor)
    resolution = require_resolution(resolution)
    invert = require_checkbox(invert, "invert")
    if preprocessor not in _preprocessor_choices(mode):
        raise ValueError(f"{preprocessor} is not valid in {mode} mode.")
    width, height = preview_dimensions(
        width,
        height,
        enable_hr,
        hr_scale,
        hr_resize_x,
        hr_resize_y,
        getattr(shared.opts, "res_step", 8),
    )
    sources = normalize_images(images)
    warning = ratio_warning(sources, width, height)
    if warning:
        gr.Warning(warning)
    label = "Pose" if mode == POSE_MODE else "Depth"
    previews = [
        (
            _create_control_map(
                mode,
                source,
                preprocessor,
                resolution,
                width,
                height,
                invert,
            ),
            f"{label} {index + 1} — {width}×{height}",
        )
        for index, source in enumerate(sources)
    ]
    if warning:
        status = f"⚠️ {warning}"
    else:
        status = (
            f"✅ {len(previews)} {label.lower()} map"
            + ("" if len(previews) == 1 else "s")
            + f" prepared at **{width}×{height}** without cropping or distortion."
        )
    return previews, status


def _preview_entry(
    entry,
    width,
    height,
    enable_hr=False,
    hr_scale=1.0,
    hr_resize_x=0,
    hr_resize_y=0,
):
    width, height = preview_dimensions(
        width,
        height,
        enable_hr,
        hr_scale,
        hr_resize_x,
        hr_resize_y,
        getattr(shared.opts, "res_step", 8),
    )
    source = normalize_images(entry["source"])[0]
    warning = ratio_warning([source], width, height)
    result = _create_control_map(
        entry["mode"],
        source,
        entry["preprocessor"],
        entry["resolution"],
        width,
        height,
        entry["invert"],
    )
    return result, warning, width, height


def _preview_selected_entry(
    entries,
    selected_index,
    cache,
    width,
    height,
    enable_hr=False,
    hr_scale=1.0,
    hr_resize_x=0,
    hr_resize_y=0,
):
    entries = entries or []
    if not entries or not 0 <= int(selected_index) < len(entries):
        raise ValueError("Select a control file before preparing its preview.")
    entry = entries[int(selected_index)]
    result, warning, width, height = _preview_entry(
        entry,
        width,
        height,
        enable_hr,
        hr_scale,
        hr_resize_x,
        hr_resize_y,
    )
    cache = dict(cache or {})
    cache[entry["id"]] = {
        "signature": _preview_signature(entry),
        "image": result,
        "dimensions": (width, height),
        "warning": warning,
    }
    if warning:
        gr.Warning(warning)
        status = f"⚠️ {warning}"
    else:
        status = f"✅ Cached preview for **{entry['name']}** at **{width}×{height}**."
    return result, status, cache


def _cache_all_previews(
    entries,
    selected_index,
    cache,
    width,
    height,
    enable_hr=False,
    hr_scale=1.0,
    hr_resize_x=0,
    hr_resize_y=0,
):
    entries = entries or []
    if not entries:
        raise ValueError("Add at least one control file before caching previews.")
    cache = dict(cache or {})
    warnings = []
    expected_dimensions = preview_dimensions(
        width,
        height,
        enable_hr,
        hr_scale,
        hr_resize_x,
        hr_resize_y,
        getattr(shared.opts, "res_step", 8),
    )
    for entry in entries:
        result = _cache_for_entry(cache, entry, expected_dimensions)
        if result is None:
            result, warning, final_width, final_height = _preview_entry(
                entry,
                width,
                height,
                enable_hr,
                hr_scale,
                hr_resize_x,
                hr_resize_y,
            )
            cache[entry["id"]] = {
                "signature": _preview_signature(entry),
                "image": result,
                "dimensions": (final_width, final_height),
                "warning": warning,
            }
        else:
            warning = cache[entry["id"]].get("warning", "")
        if warning:
            warnings.append(f"{entry['name']}: {warning}")
    selected_index = max(0, min(int(selected_index), len(entries) - 1))
    selected = _cache_for_entry(cache, entries[selected_index])
    if warnings:
        gr.Warning(" Some controls use black letterboxing. ".join(warnings))
    status = (
        f"✅ Cached **{len(entries)}** control preview(s). Select a row to inspect "
        "its cached result."
    )
    if warnings:
        status += f"  \n⚠️ {len(warnings)} ratio warning(s); no image will be cropped."
    return selected, status, cache


def _download_models(entries, selected_mode):
    """Download exactly the adapter/preprocessor files needed by the list."""

    records = entries or []
    modes = {entry.get("mode") for entry in records if entry.get("strength", 1) > 0}
    if not modes:
        modes = {require_mode(selected_mode)}
    lines = []
    if DEPTH_MODE in modes:
        path = download_control_model(paths.models_path)
        lines.append(f"Depth adapter: `{path}` — ready")
    if POSE_MODE in modes:
        adapter = download_pose_model(paths.models_path)
        lines.append(f"Pose adapter: `{adapter}` — ready")
        needs_dwpose = any(
            entry.get("mode") == POSE_MODE
            and entry.get("preprocessor") == DWPOSE
            and entry.get("strength", 1) > 0
            for entry in records
        )
        if needs_dwpose or not records:
            dwpose = download_dwpose_models(paths.models_path)
            lines.append(f"DWPose: `{dwpose[0].parent}` — 2/2 verified")
    # A verified replacement at the same path can change preprocessing output.
    # Do not retain maps produced by the previous model files.
    _clear_preprocessor_cache()
    return "  \n".join(lines)


def _final_dimensions(process) -> tuple[int, int]:
    if bool(getattr(process, "enable_hr", False)):
        width = int(getattr(process, "hr_upscale_to_x", 0) or 0)
        height = int(getattr(process, "hr_upscale_to_y", 0) or 0)
        if width > 0 and height > 0:
            return width, height
    return int(process.width), int(process.height)


def _warn_ratio(process, sources, width: int, height: int) -> None:
    warning = ratio_warning(sources, width, height)
    warning_key = (width, height, warning)
    if warning and getattr(process, "_krea2_control_ratio_warning", None) != warning_key:
        logger.warning(warning)
        try:
            gr.Warning(warning)
        except Exception:
            logger.debug("Could not display the ratio warning in Gradio.", exc_info=True)
        process._krea2_control_ratio_warning = warning_key


def _generation_entries(
    control_images,
    mode,
    preprocessor,
    resolution,
    invert,
    strength,
):
    mode = require_mode(mode)
    preprocessor = require_preprocessor(preprocessor)
    resolution = require_resolution(resolution)
    invert = require_checkbox(invert, "invert")
    strength = require_strength(strength)
    values = control_images
    if isinstance(values, dict) and "source" in values:
        values = [values]
    records = (
        copy.deepcopy(values)
        if isinstance(values, list)
        and values
        and all(isinstance(item, dict) and "source" in item for item in values)
        else None
    )
    if records is None:
        sources = normalize_images(values)
        records = [
            {
                "id": f"legacy-{index}",
                "source": source,
                "name": f"control {index + 1}",
                "detected_kind": PHOTO,
                "detection_confidence": 0.0,
                "detection_reason": "legacy/API global settings",
                "mode": mode,
                "preprocessor": preprocessor,
                "resolution": resolution,
                "invert": invert,
                "strength": strength,
            }
            for index, source in enumerate(sources)
        ]
    if not records:
        raise ValueError("Add at least one Krea 2 control file before generating.")
    normalised = []
    used_ids = set()
    for index, record in enumerate(records):
        entry_mode = require_mode(record.get("mode", mode))
        entry_preprocessor = require_preprocessor(
            record.get("preprocessor", preprocessor)
        )
        if entry_preprocessor not in _preprocessor_choices(entry_mode):
            raise ValueError(
                f"{entry_preprocessor} is not valid for control file {index + 1} "
                f"in {entry_mode} mode."
            )
        entry_id = str(record.get("id") or f"control-{index}")
        if entry_id in used_ids:
            entry_id = f"{entry_id}-{index}"
        used_ids.add(entry_id)
        normalised.append(
            {
                **record,
                "id": entry_id,
                "name": str(record.get("name", f"control {index + 1}")),
                "mode": entry_mode,
                "preprocessor": entry_preprocessor,
                "resolution": require_resolution(record.get("resolution", resolution)),
                "invert": require_checkbox(record.get("invert", invert), "invert")
                if entry_mode == DEPTH_MODE
                else False,
                "strength": require_strength(record.get("strength", strength)),
            }
        )
    return normalised


def _scheduled_indices(entry_count, batch_size, n_iter):
    return [
        alternating_image_indices(entry_count, batch_size, iteration)
        for iteration in range(n_iter)
    ]


def _validate_schedule(entries, batch_size, n_iter):
    groups = _scheduled_indices(len(entries), int(batch_size), int(n_iter))
    for iteration, indices in enumerate(groups):
        active = [entries[index] for index in indices]
        modes = {entry["mode"] for entry in active if entry["strength"] > 0}
        strengths = {entry["strength"] for entry in active}
        if len(modes) > 1 or len(strengths) > 1:
            references = ", ".join(str(index + 1) for index in indices)
            raise ValueError(
                "Control references "
                f"{references} in batch {iteration + 1} use different modes or "
                "strengths. Forge applies one ControlNet-LoRA per simultaneous "
                "batch; set Batch size to 1 for per-file mode/strength settings."
            )
    return groups


def _requested_final_dimensions(process):
    return preview_dimensions(
        process.width,
        process.height,
        bool(getattr(process, "enable_hr", False)),
        float(getattr(process, "hr_scale", 1.0) or 1.0),
        int(getattr(process, "hr_resize_x", 0) or 0),
        int(getattr(process, "hr_resize_y", 0) or 0),
        getattr(shared.opts, "res_step", 8),
    )


def _prepare_generation_cache(process, entries):
    groups = _validate_schedule(
        entries, int(process.batch_size), int(getattr(process, "n_iter", 1))
    )
    scheduled = {index for group in groups for index in group}
    if len(scheduled) < len(entries):
        message = (
            f"Only {len(scheduled)} of {len(entries)} control files are scheduled. "
            "Increase Batch count to use every file."
        )
        logger.warning(message)
        try:
            gr.Warning(message)
        except Exception:
            logger.debug("Could not display the unused-control warning.", exc_info=True)

    final_width, final_height = _requested_final_dimensions(process)
    processed = {}
    previews = {}
    cancelled = False
    for index in sorted(scheduled):
        if bool(getattr(shared.state, "interrupted", False)) or bool(
            getattr(shared.state, "stopping_generation", False)
        ):
            cancelled = True
            break
        entry = entries[index]
        if entry["strength"] == 0:
            continue
        result = _preprocess_control_map(
            entry["mode"],
            entry["source"],
            entry["preprocessor"],
            entry["resolution"],
            entry["invert"],
        )
        processed[entry["id"]] = result
        previews[entry["id"]] = fit_control_map(result, final_width, final_height)
        _publish_active_control_preview(
            [previews[entry["id"]]], [index], f"{entry['mode']} · preparing"
        )

    if bool(getattr(shared.state, "interrupted", False)) or bool(
        getattr(shared.state, "stopping_generation", False)
    ):
        cancelled = True

    needed_modes = {
        entries[index]["mode"]
        for index in scheduled
        if not cancelled
        and entries[index]["strength"] > 0
        and entries[index]["id"] in processed
    }
    adapter_states = {}
    if DEPTH_MODE in needed_modes:
        adapter_states[DEPTH_MODE] = load_control_state_dict(
            control_model_path(paths.models_path)
        )
    if POSE_MODE in needed_modes:
        adapter_states[POSE_MODE] = load_pose_state_dict(
            pose_model_path(paths.models_path)
        )

    process._krea2_control_cache = {
        "entries": entries,
        "groups": groups,
        "processed": processed,
        "previews": previews,
        "adapter_states": adapter_states,
        "final_dimensions": (final_width, final_height),
        "cancelled": cancelled,
    }
    first = [entries[index] for index in groups[0]] if groups else []
    first_maps = [
        previews[entry["id"]]
        for entry in first
        if entry["strength"] > 0 and entry["id"] in previews
    ]
    first_indices = [
        groups[0][slot]
        for slot, entry in enumerate(first)
        if entry["strength"] > 0 and entry["id"] in previews
    ] if groups else []
    if first_maps:
        _publish_active_control_preview(first_maps, first_indices, first[0]["mode"])
    return process._krea2_control_cache


def _guard_failure_once(process, error):
    if getattr(process, "_krea2_control_failure_guarded", False):
        return
    install_failure_guard(process, error)
    process._krea2_control_failure_guarded = True


def _cleanup_control_cache(process):
    _restore_pose_conditioning(process)
    for attribute in (
        "_krea2_pose_context",
        "_krea2_control_active",
        "_krea2_control_cache",
        "_krea2_control_ratio_warning",
    ):
        if hasattr(process, attribute):
            delattr(process, attribute)


def _restore_pose_conditioning(process) -> None:
    original_setup = getattr(process, "_krea2_pose_original_setup_conds", None)
    if original_setup is not None:
        process.setup_conds = original_setup
        del process._krea2_pose_original_setup_conds
    original_hr = getattr(process, "_krea2_pose_original_calculate_hr_conds", None)
    if original_hr is not None:
        process.calculate_hr_conds = original_hr
        del process._krea2_pose_original_calculate_hr_conds


def _install_pose_conditioning(process, vision: torch.Tensor) -> None:
    """Replace conditioning for one Forge batch, preserving A/B/C visual refs."""

    from modules import sd_samplers

    _restore_pose_conditioning(process)
    process._krea2_pose_original_setup_conds = process.setup_conds
    original_hr = getattr(process, "calculate_hr_conds", None)
    if original_hr is not None:
        process._krea2_pose_original_calculate_hr_conds = original_hr

    def build(
        prompts,
        negative_prompts,
        steps,
        width,
        height,
        hires_steps=None,
        cfg_scale=None,
        distilled_cfg_scale=None,
    ):
        process.sd_model.set_clip_skip(int(shared.opts.CLIP_stop_at_last_layers))
        if (process.cfg_scale if cfg_scale is None else cfg_scale) == 1:
            unconditional = None
        else:
            unconditional = build_pose_prompt_conditioning(
                process.sd_model,
                negative_prompts,
                vision,
                steps,
                width=width,
                height=height,
                negative=True,
                hires_steps=hires_steps,
                distilled_cfg_scale=distilled_cfg_scale,
            )
        conditional = build_pose_prompt_conditioning(
            process.sd_model,
            prompts,
            vision,
            steps,
            width=width,
            height=height,
            multicond=True,
            hires_steps=hires_steps,
            distilled_cfg_scale=distilled_cfg_scale,
        )
        return conditional, unconditional

    def pose_setup_conds():
        sampler = sd_samplers.find_sampler_config(process.sampler_name)
        total_steps = sampler.total_steps(process.steps) if sampler else process.steps
        process.step_multiplier = total_steps // process.steps
        process.firstpass_steps = total_steps
        process.c, process.uc = build(
            process.prompts,
            process.negative_prompts,
            total_steps,
            process.width,
            process.height,
            cfg_scale=process.cfg_scale,
            distilled_cfg_scale=getattr(process, "distilled_cfg_scale", None),
        )
        if hasattr(process, "hr_c"):
            process.hr_c = None
            process.hr_uc = None

    def pose_calculate_hr_conds():
        if getattr(process, "hr_c", None) is not None:
            return
        sampler_name = process.hr_sampler_name or process.sampler_name
        sampler = sd_samplers.find_sampler_config(sampler_name)
        steps = process.hr_second_pass_steps or process.steps
        total_steps = sampler.total_steps(steps) if sampler else steps
        process.hr_c, process.hr_uc = build(
            process.hr_prompts,
            process.hr_negative_prompts,
            process.firstpass_steps,
            process.hr_upscale_to_x,
            process.hr_upscale_to_y,
            total_steps,
            cfg_scale=getattr(process, "hr_cfg", process.cfg_scale),
            distilled_cfg_scale=getattr(
                process,
                "hr_distilled_cfg",
                getattr(process, "distilled_cfg_scale", None),
            ),
        )

    process.setup_conds = pose_setup_conds
    if original_hr is not None:
        process.calculate_hr_conds = pose_calculate_hr_conds
    process.cached_c = [None, None, None]
    process.cached_uc = [None, None, None]


class Krea2DepthControlScript(scripts.ScriptBuiltinUI):
    sorting_priority = 18110

    def __init__(self):
        self._forge_components = {}

    def after_component(self, component, **_kwargs):
        elem_id = getattr(component, "elem_id", None)
        wanted = {f"{self.tabname}_width", f"{self.tabname}_height"}
        if self.is_txt2img:
            wanted.update(
                {
                    "txt2img_hr-checkbox",
                    "txt2img_hr_scale",
                    "txt2img_hr_resize_x",
                    "txt2img_hr_resize_y",
                }
            )
        if elem_id in wanted:
            self._forge_components[elem_id] = component

    def title(self):
        return "Krea 2 Depth / Pose ControlNet-LoRA"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        choices = _depth_preprocessor_choices()
        depth_path = control_model_path(paths.models_path)
        pose_path = pose_model_path(paths.models_path)
        pose_preprocessor_dir = dwpose_model_dir(paths.models_path)
        with InputAccordion(False, label=self.title()) as enabled:
            gr.Markdown(
                "Depth control and full body/hand/face Pose control for Krea 2. "
                "Add files to the ordered list, click a row to edit it, and generated "
                "images use the entries in A, B, C, A… order."
            )
            entries_state = gr.State([])
            selected_index = gr.State(-1)
            preview_cache = gr.State({})
            with gr.Row():
                upload_button = gr.UploadButton(
                    "Add image(s)",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                    variant="primary",
                )
                remove_button = gr.Button("Remove selected", variant="secondary")
                move_up_button = gr.Button("Move up", variant="secondary")
                move_down_button = gr.Button("Move down", variant="secondary")
                clear_button = gr.Button("Clear list", variant="secondary")
            control_list = gr.Dataframe(
                headers=ENTRY_TABLE_HEADERS,
                value=[],
                datatype=["str", "str", "str", "str", "str"],
                row_count=(0, "dynamic"),
                col_count=(len(ENTRY_TABLE_HEADERS), "fixed"),
                type="array",
                interactive=False,
                wrap=True,
                height=230,
                column_widths=[40, 125, 100, 90, 170],
                label="Ordered control list — click a row to select it",
                elem_id=self.elem_id("krea2_control_list"),
            )
            with gr.Row():
                with gr.Column():
                    selected_source = gr.Image(
                        label="Selected source",
                        type="filepath",
                        interactive=False,
                        height=320,
                        object_fit="contain",
                    )
                with gr.Column():
                    preview = gr.Image(
                        label="Selected control preview at final resolution",
                        type="numpy",
                        interactive=False,
                        height=320,
                        object_fit="contain",
                        elem_id=self.elem_id("krea2_control_preview"),
                    )
            selection_status = gr.Markdown(
                "Add one or more images. New files inherit the current settings."
            )
            mode = gr.Dropdown(
                choices=[DEPTH_MODE, POSE_MODE], value=DEPTH_MODE, label="Selected file mode"
            )
            with gr.Row():
                preprocessor = gr.Dropdown(
                    choices=choices,
                    value="depth_anything_v2" if "depth_anything_v2" in choices else choices[0],
                    label="Selected file preprocessor",
                )
                resolution = gr.Slider(
                    minimum=256,
                    maximum=2048,
                    value=768,
                    step=64,
                    label="Selected file preprocessor resolution",
                )
                invert = gr.Checkbox(False, label="Selected file: invert depth map")
                strength = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.05,
                    label="Selected file control strength",
                    info="0 disables this entry. Mixed strengths require Batch size 1.",
                )
            with gr.Row():
                redetect_button = gr.Button("Re-detect selected", variant="secondary")
                preview_button = gr.Button("Preview selected", variant="secondary")
                cache_all_button = gr.Button("Cache all previews", variant="secondary")
                download_button = gr.Button(
                    "Download / verify models needed by the list", variant="secondary"
                )
            ratio_status = gr.Markdown(
                "Different aspect ratios use centered black bars without crop or "
                "distortion. Preview results are cached per file and invalidated "
                "when its settings or the target dimensions change."
            )
            model_status = gr.Markdown(
                f"Depth: `{depth_path}` — {'present' if depth_path.is_file() else 'missing'}  \n"
                f"Pose: `{pose_path}` — {'present' if pose_path.is_file() else 'missing'}  \n"
                f"DWPose: `{pose_preprocessor_dir}` — "
                f"{sum((pose_preprocessor_dir / name).is_file() for name in DWPOSE_FILES)}/2 present. "
                "Every file is hash-verified before use."
            )
        for component in (
            entries_state,
            selected_index,
            preview_cache,
            upload_button,
            remove_button,
            move_up_button,
            move_down_button,
            clear_button,
            control_list,
            selected_source,
            selection_status,
            model_status,
            redetect_button,
            preview_button,
            cache_all_button,
            download_button,
            preview,
            ratio_status,
        ):
            component.do_not_save_to_config = True

        tabname = "img2img" if is_img2img else "txt2img"
        width = self._forge_components.get(f"{tabname}_width") or gr.Number(
            value=512, visible=False
        )
        height = self._forge_components.get(f"{tabname}_height") or gr.Number(
            value=512, visible=False
        )
        dimension_inputs = [width, height]
        if not is_img2img:
            dimension_inputs.extend(
                [
                    self._forge_components.get("txt2img_hr-checkbox")
                    or gr.Checkbox(False, visible=False),
                    self._forge_components.get("txt2img_hr_scale")
                    or gr.Number(value=1.0, visible=False),
                    self._forge_components.get("txt2img_hr_resize_x")
                    or gr.Number(value=0, visible=False),
                    self._forge_components.get("txt2img_hr_resize_y")
                    or gr.Number(value=0, visible=False),
                ]
            )

        selection_outputs = [
            control_list,
            selected_source,
            preview,
            ratio_status,
            mode,
            preprocessor,
            resolution,
            invert,
            strength,
            selection_status,
        ]
        upload_button.upload(
            fn=_add_entries,
            inputs=[
                upload_button,
                entries_state,
                mode,
                preprocessor,
                resolution,
                invert,
                strength,
                preview_cache,
            ],
            outputs=[entries_state, selected_index, *selection_outputs],
        )
        control_list.select(
            fn=_select_entry,
            inputs=[entries_state, preview_cache],
            outputs=[selected_index, *selection_outputs],
        )
        remove_button.click(
            fn=_remove_entry,
            inputs=[entries_state, selected_index, preview_cache],
            outputs=[
                entries_state,
                selected_index,
                preview_cache,
                *selection_outputs,
            ],
        )
        clear_button.click(
            fn=_clear_entries,
            outputs=[
                entries_state,
                selected_index,
                preview_cache,
                *selection_outputs,
            ],
        )
        move_up_button.click(
            fn=lambda entries, index, cache: _move_entry(entries, index, cache, -1),
            inputs=[entries_state, selected_index, preview_cache],
            outputs=[entries_state, selected_index, *selection_outputs],
        )
        move_down_button.click(
            fn=lambda entries, index, cache: _move_entry(entries, index, cache, 1),
            inputs=[entries_state, selected_index, preview_cache],
            outputs=[entries_state, selected_index, *selection_outputs],
        )
        redetect_button.click(
            fn=_redetect_entry,
            inputs=[entries_state, selected_index, preview_cache],
            outputs=[
                entries_state,
                selected_index,
                preview_cache,
                *selection_outputs,
            ],
        )
        mode.input(
            fn=_change_entry_mode,
            inputs=[entries_state, selected_index, mode, preview_cache],
            outputs=[
                entries_state,
                control_list,
                preprocessor,
                invert,
                preview_cache,
                preview,
                ratio_status,
                selection_status,
            ],
        )
        for setting in (preprocessor, resolution, invert, strength):
            setting.input(
                fn=_change_entry_settings,
                inputs=[
                    entries_state,
                    selected_index,
                    preprocessor,
                    resolution,
                    invert,
                    strength,
                    preview_cache,
                ],
                outputs=[
                    entries_state,
                    control_list,
                    preview_cache,
                    preview,
                    ratio_status,
                ],
            )
        preview_button.click(
            fn=_preview_selected_entry,
            inputs=[entries_state, selected_index, preview_cache, *dimension_inputs],
            outputs=[preview, ratio_status, preview_cache],
        )
        cache_all_button.click(
            fn=_cache_all_previews,
            inputs=[entries_state, selected_index, preview_cache, *dimension_inputs],
            outputs=[preview, ratio_status, preview_cache],
        )
        download_button.click(
            fn=_download_models, inputs=[entries_state, mode], outputs=[model_status]
        )
        for dimension in dimension_inputs:
            dimension.change(
                fn=_invalidate_all_previews,
                inputs=[entries_state, selected_index],
                outputs=[preview_cache, preview, ratio_status],
            )
        return (
            enabled,
            mode,
            entries_state,
            preprocessor,
            resolution,
            invert,
            strength,
        )

    @torch.inference_mode()
    def process(
        self,
        p,
        enabled,
        mode,
        control_images,
        preprocessor,
        resolution,
        invert,
        strength,
        **_kwargs,
    ):
        if hasattr(p, "_krea2_control_failure_guarded"):
            del p._krea2_control_failure_guarded
        _cleanup_control_cache(p)
        if not require_checkbox(enabled, "enabled"):
            return
        try:
            is_records = (
                isinstance(control_images, list)
                and control_images
                and all(
                    isinstance(item, dict) and "source" in item
                    for item in control_images
                )
            )
            if not is_records and require_strength(strength) == 0:
                return
            entries = _generation_entries(
                control_images,
                mode,
                preprocessor,
                resolution,
                invert,
                strength,
            )
            if any(entry["strength"] > 0 for entry in entries):
                if type(p.sd_model).__name__ != "Krea2":
                    raise TypeError("Krea 2 control requires a Krea 2 checkpoint.")
            final_width, final_height = _requested_final_dimensions(p)
            _warn_ratio(
                p,
                [normalize_images(entry["source"])[0] for entry in entries],
                final_width,
                final_height,
            )
            _prepare_generation_cache(p, entries)
            p.extra_generation_params.update(
                {
                    "Krea 2 Control Files": len(entries),
                    "Krea 2 Control Sequence": "A, B, C, A…",
                    "Krea 2 Per-file Settings": True,
                }
            )
        except Exception as exc:
            _cleanup_control_cache(p)
            _guard_failure_once(p, exc)
            raise

    @torch.inference_mode()
    def process_batch(
        self,
        p,
        enabled,
        mode,
        control_images,
        preprocessor,
        resolution,
        invert,
        strength,
        **_kwargs,
    ):
        _restore_pose_conditioning(p)
        if not require_checkbox(enabled, "enabled"):
            return
        try:
            cache = getattr(p, "_krea2_control_cache", None)
            if cache is None:
                self.process(
                    p,
                    enabled,
                    mode,
                    control_images,
                    preprocessor,
                    resolution,
                    invert,
                    strength,
                )
                cache = getattr(p, "_krea2_control_cache", None)
            if cache is None:
                return
            if cache.get("cancelled"):
                return
            iteration = int(getattr(p, "iteration", 0))
            indices = cache["groups"][iteration]
            active_entries = [cache["entries"][index] for index in indices]
            active_strength = active_entries[0]["strength"]
            if active_strength == 0:
                p._krea2_control_active = {
                    "mode": None,
                    "strength": 0.0,
                    "indices": indices,
                    "entries": active_entries,
                }
                return
            active_mode = active_entries[0]["mode"]
            processed_maps = [
                cache["processed"][entry["id"]] for entry in active_entries
            ]
            p._krea2_control_active = {
                "mode": active_mode,
                "strength": active_strength,
                "indices": indices,
                "entries": active_entries,
                "processed": processed_maps,
            }
            if active_mode != POSE_MODE:
                return

            width, height = cache["final_dimensions"]
            pose_maps = [fit_control_map(item, width, height) for item in processed_maps]
            _publish_active_control_preview(pose_maps, indices, "Pose")
            image = torch.from_numpy(np.stack(pose_maps).copy()).float().div_(255.0)
            vision, latent = encode_pose_references(p.sd_model, image)
            state_dict = cache["adapter_states"][POSE_MODE]
            p._krea2_pose_context = {
                "latent": latent,
                "state_dict": state_dict,
                "strength": active_strength,
                "maps": pose_maps,
                "indices": indices,
            }
            _install_pose_conditioning(p, vision)
            preprocessors = ", ".join(
                dict.fromkeys(entry["preprocessor"] for entry in active_entries)
            )
            p.extra_generation_params.update(
                {
                    "Krea 2 Control Mode": POSE_MODE,
                    "Krea 2 Pose Control": os.path.basename(
                        pose_model_path(paths.models_path)
                    ),
                    "Krea 2 Pose Preprocessor": preprocessors,
                    "Krea 2 Pose Strength": active_strength,
                    "Krea 2 Pose Active References": ", ".join(
                        str(index + 1) for index in indices
                    ),
                }
            )
        except Exception as exc:
            _cleanup_control_cache(p)
            _guard_failure_once(p, exc)
            raise

    @torch.inference_mode()
    def process_before_every_sampling(
        self,
        p,
        enabled,
        mode,
        control_images,
        preprocessor,
        resolution,
        invert,
        strength,
        **_kwargs,
    ):
        try:
            if not require_checkbox(enabled, "enabled"):
                return
            active = getattr(p, "_krea2_control_active", None)
            if not active:
                self.process_batch(
                    p,
                    enabled,
                    mode,
                    control_images,
                    preprocessor,
                    resolution,
                    invert,
                    strength,
                )
                active = getattr(p, "_krea2_control_active", None)
            if not active or active["strength"] == 0:
                return
            if active["mode"] == POSE_MODE:
                context = getattr(p, "_krea2_pose_context", None)
                if not context:
                    raise RuntimeError("Krea 2 Pose conditioning was not prepared.")
                _publish_active_control_preview(
                    context["maps"], context["indices"], "Pose"
                )
                p.sd_model.forge_objects.unet = apply_pose_control(
                    p.sd_model.forge_objects.unet,
                    context["latent"],
                    context["state_dict"],
                    context["strength"],
                )
                return

            width, height = generation_dimensions(p)
            depths = [fit_control_map(item, width, height) for item in active["processed"]]
            _publish_active_control_preview(depths, active["indices"], "Depth")
            image = torch.from_numpy(np.stack(depths).copy()).float().div_(255.0)
            _, latent = p.sd_model.encode_vision(image)
            if latent.ndim == 5 and latent.shape[2] == 1:
                latent = latent[:, :, 0]
            cache = p._krea2_control_cache
            state_dict = cache["adapter_states"][DEPTH_MODE]
            p.sd_model.forge_objects.unet = apply_depth_control(
                p.sd_model.forge_objects.unet,
                latent,
                state_dict,
                active["strength"],
            )
            preprocessors = ", ".join(
                dict.fromkeys(
                    entry["preprocessor"] for entry in active["entries"]
                )
            )
            p.extra_generation_params.update(
                {
                    "Krea 2 Control Mode": DEPTH_MODE,
                    "Krea 2 Depth Control": os.path.basename(
                        control_model_path(paths.models_path)
                    ),
                    "Krea 2 Depth Preprocessor": preprocessors,
                    "Krea 2 Depth Strength": active["strength"],
                    "Krea 2 Depth Active References": ", ".join(
                        str(index + 1) for index in active["indices"]
                    ),
                }
            )
        except Exception as exc:
            _cleanup_control_cache(p)
            _guard_failure_once(p, exc)
            raise

    def postprocess_batch(self, p, *args, **kwargs):
        del args, kwargs
        _restore_pose_conditioning(p)
        if hasattr(p, "_krea2_pose_context"):
            del p._krea2_pose_context
        if hasattr(p, "_krea2_control_active"):
            del p._krea2_control_active

    def postprocess(self, p, _processed, *args):
        del args
        _cleanup_control_cache(p)
        if hasattr(p, "_krea2_control_failure_guarded"):
            del p._krea2_control_failure_guarded
