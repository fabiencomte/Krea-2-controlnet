from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "examples" / "character-matrix"


def load_demo() -> dict:
    return json.loads((DEMO / "demo.json").read_text(encoding="utf-8"))


def test_demo_has_exactly_three_unique_openpose_sources_and_one_depth() -> None:
    demo = load_demo()
    pose = [
        source
        for source in demo["sources"].values()
        if source["mode"] == "Pose / OpenPose"
    ]
    depth = [source for source in demo["sources"].values() if source["mode"] == "Depth"]
    assert len(pose) == 3
    assert len(depth) == 1
    actual = []
    for source in pose:
        data = (DEMO / source["file"]).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        assert digest == source["sha256"]
        actual.append(digest)
    assert len(set(actual)) == 3


def test_demo_source_hashes_and_sizes_are_pinned() -> None:
    for source in load_demo()["sources"].values():
        path = DEMO / source["file"]
        assert path.stat().st_size == source["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]


def test_demo_matrix_has_four_characters_and_sixteen_valid_selections() -> None:
    demo = load_demo()
    runs = {run["id"]: run for run in demo["runs"]}
    assert len(runs) == 12
    assert demo["matrix"]["columns"] == [
        "Control input",
        "Deadpool",
        "Bart Simpson",
        "Black Widow",
        "Darth Vader",
    ]
    rows = demo["matrix"]["rows"]
    assert [row["source_id"] for row in rows] == ["standing", "dance", "jump", "depth"]
    selections = [selected for row in rows for selected in row["outputs"]]
    assert len(selections) == 16
    for selected in selections:
        assert selected["run"] in runs
        assert 0 <= selected["index"] < runs[selected["run"]]["batch_size"]


def test_demo_depth_runs_share_seed_background_and_framing() -> None:
    depth_runs = [run for run in load_demo()["runs"] if run["source_ids"] == ["depth"]]
    assert len(depth_runs) == 4
    assert {run["seed"] for run in depth_runs} == {8201}
    required = {
        "seamless uniform warm light-gray studio background",
        "flat even background color",
        "waist-up centered portrait",
    }
    for run in depth_runs:
        assert all(fragment in run["prompt"] for fragment in required)


def test_demo_models_are_fully_pinned() -> None:
    models = load_demo()["models"]
    assert {model["role"] for model in models} == {
        "checkpoint",
        "vae",
        "text_encoder",
        "pose_control",
        "depth_control",
    }
    for model in models:
        assert model["bytes"] > 0
        assert len(model["sha256"]) == 64
        int(model["sha256"], 16)


def test_removed_character_and_reference_images_are_absent() -> None:
    forbidden = "may" + "uri"
    for path in DEMO.rglob("*"):
        assert forbidden not in path.name.lower()
        if path.suffix.lower() in {".json", ".md", ".py"}:
            assert forbidden not in path.read_text(encoding="utf-8").lower()
    assert not (DEMO / "references").exists()


def load_image_ops_module():
    import importlib.util

    path = DEMO / "scripts" / "image_ops.py"
    spec = importlib.util.spec_from_file_location("demo_image_ops", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_depth_background_normalization_preserves_coloured_subject() -> None:
    import numpy as np
    from PIL import Image

    image_ops = load_image_ops_module()
    pixels = np.full((128, 128, 3), (120, 115, 105), dtype=np.uint8)
    pixels[20:, 42:86] = (180, 20, 20)
    normalized = np.asarray(
        image_ops.normalize_depth_background(Image.fromarray(pixels), "Deadpool")
    )
    assert tuple(normalized[0, 0]) == (163, 160, 146)
    assert np.abs(normalized[70, 64].astype(int) - (180, 20, 20)).max() <= 1


def test_depth_background_normalization_preserves_dark_subject() -> None:
    import numpy as np
    from PIL import Image

    image_ops = load_image_ops_module()
    pixels = np.full((128, 128, 3), (145, 135, 125), dtype=np.uint8)
    pixels[18:, 38:90] = (25, 25, 25)
    normalized = np.asarray(
        image_ops.normalize_depth_background(Image.fromarray(pixels), "Black Widow")
    )
    assert tuple(normalized[0, 0]) == (163, 160, 146)
    assert np.abs(normalized[70, 64].astype(int) - (25, 25, 25)).max() <= 1


def test_readme_visual_matrix_stays_near_the_top() -> None:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    image = "assets/forge-character-control-matrix.webp"
    assert readme.count(image) == 1
    section = readme.index("### What changed compared")
    table = readme.index("| Original standalone project", section)
    assert section < readme.index(image) < table
    image_line = readme[: readme.index(image)].count("\n") + 1
    assert image_line <= 35
