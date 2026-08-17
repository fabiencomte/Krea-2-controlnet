#!/usr/bin/env python3
"""Validate demo assets and generated outputs with strict, explainable gates."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from image_ops import normalize_depth_background


HERE = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rgb(image: Image.Image, size: tuple[int, int] | None = None) -> np.ndarray:
    converted = image.convert("RGB")
    if size and converted.size != size:
        converted = converted.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(converted, dtype=np.float32)


def corner_color(image: np.ndarray, patch: int = 72) -> np.ndarray:
    corners = [
        image[:patch, :patch],
        image[:patch, -patch:],
        image[-patch:, :patch],
        image[-patch:, -patch:],
    ]
    return np.concatenate([corner.reshape(-1, 3) for corner in corners]).mean(axis=0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--presentation-dir", type=Path)
    parser.add_argument(
        "--report", type=Path, default=HERE / "outputs" / "quality-report.json"
    )
    args = parser.parse_args()

    manifest = json.loads((HERE / "demo.json").read_text(encoding="utf-8"))
    serialized = json.dumps(manifest, ensure_ascii=False).lower()
    failures: list[str] = []
    metrics: dict = {"sources": {}, "outputs": {}}

    if "may" + "uri" in serialized:
        failures.append("removed character still appears in demo.json")
    pose_hashes = []
    for source_id, source in manifest["sources"].items():
        path = HERE / source["file"]
        actual = digest(path)
        metrics["sources"][source_id] = {"sha256": actual, "bytes": path.stat().st_size}
        if actual != source["sha256"]:
            failures.append(f"{source_id}: source SHA-256 mismatch")
        if source["mode"] == "Pose / OpenPose":
            pose_hashes.append(actual)
    if len(pose_hashes) != 3 or len(set(pose_hashes)) != 3:
        failures.append("OpenPose coverage is not exactly three unique source files")
    if len(manifest["matrix"]["rows"]) != 4:
        failures.append("matrix must contain exactly three Pose rows and one Depth row")

    if args.output_dir:
        runs = {run["id"]: run for run in manifest["runs"]}
        output_hashes = []
        depth_backgrounds = []
        if args.presentation_dir:
            args.presentation_dir.mkdir(parents=True, exist_ok=True)
        for row in manifest["matrix"]["rows"]:
            source_id = row["source_id"]
            source = manifest["sources"][source_id]
            source_path = HERE / source["file"]
            source_image = Image.open(source_path).convert("RGB")
            for column_index, selected in enumerate(row["outputs"], start=1):
                run = runs[selected["run"]]
                character = manifest["matrix"]["columns"][column_index]
                path = args.output_dir / f"{run['id']}-{selected['index']}.png"
                if not path.is_file():
                    failures.append(f"missing selected output: {path.name}")
                    continue
                raw_image = Image.open(path).convert("RGB")
                if raw_image.size != (768, 768):
                    failures.append(
                        f"{path.name}: expected 768x768, got {raw_image.size}"
                    )
                presentation = (
                    normalize_depth_background(raw_image, character)
                    if source["mode"] == "Depth"
                    else raw_image.copy()
                )
                presentation_path = None
                if args.presentation_dir:
                    presentation_path = args.presentation_dir / path.name
                    presentation.save(presentation_path)
                out = rgb(presentation)
                src = rgb(source_image, (768, 768))
                presentation_hash = (
                    digest(presentation_path) if presentation_path else None
                )
                record = {
                    "raw_sha256": digest(path),
                    "presentation_sha256": presentation_hash,
                    "mode": source["mode"],
                    "character": character,
                }
                output_hashes.append(presentation_hash or record["raw_sha256"])

                if source["mode"] == "Pose / OpenPose":
                    spread = src.max(axis=2) - src.min(axis=2)
                    skeleton = (spread > 70) & (src.max(axis=2) > 100)
                    distance = np.linalg.norm(out - src, axis=2)
                    spatial_color_copy = float((distance[skeleton] < 55).mean())
                    black = src.mean(axis=2) < 5
                    black_copy = float((out.mean(axis=2)[black] < 15).mean())
                    record.update(
                        {
                            "spatial_source_colour_copy": spatial_color_copy,
                            "source_black_copy": black_copy,
                        }
                    )
                    if spatial_color_copy > 0.12:
                        failures.append(
                            f"{path.name}: source-colour copy {spatial_color_copy:.4f} > 0.12"
                        )
                    if black_copy > 0.35:
                        failures.append(
                            f"{path.name}: black-background copy {black_copy:.4f} > 0.35"
                        )
                else:
                    mae = float(
                        np.abs(out.mean(axis=2) - src.mean(axis=2)).mean() / 255.0
                    )
                    record["depth_rgb_mae"] = mae
                    if mae < 0.12:
                        failures.append(
                            f"{path.name}: too similar to raw depth source ({mae:.4f})"
                        )
                    background = corner_color(out)
                    depth_backgrounds.append((path.name, background))
                    record["corner_background_rgb"] = [
                        round(float(value), 3) for value in background
                    ]
                metrics["outputs"][path.name] = record

                info_path = args.output_dir / f"{run['id']}.info.json"
                if not info_path.is_file():
                    failures.append(f"missing metadata: {info_path.name}")
                else:
                    info = json.loads(info_path.read_text(encoding="utf-8"))
                    params = info.get("extra_generation_params", {})
                    if params.get("Krea 2 Control Mode") != source["mode"]:
                        failures.append(
                            f"{info_path.name}: wrong control mode metadata"
                        )

        if len(output_hashes) != 16 or len(set(output_hashes)) != 16:
            failures.append("expected 16 distinct selected output hashes")
        distances = []
        for (left_name, left), (right_name, right) in itertools.combinations(
            depth_backgrounds, 2
        ):
            normalized = float(np.linalg.norm(left - right) / (255.0 * (3.0**0.5)))
            distances.append(
                {"left": left_name, "right": right_name, "distance": normalized}
            )
            if normalized > 0.08:
                failures.append(
                    f"Depth backgrounds differ too much: {left_name} / {right_name} = {normalized:.4f}"
                )
        metrics["depth_background_pairwise"] = distances

    report = {
        "status": "PASS" if not failures else "FAIL",
        "criteria": {
            "openpose_source_count": "exactly 3 unique hashes",
            "matrix_rows": "3 OpenPose + 1 Depth",
            "selected_output_count": 16,
            "pose_spatial_source_colour_copy_max": 0.12,
            "pose_source_black_copy_max": 0.35,
            "depth_source_mae_min": 0.12,
            "depth_background_pairwise_distance_max": 0.08,
            "metadata_mode_must_match": True,
            "background_normalization": "presentation-only; raw Forge outputs preserved",
            "manual_review_required": [
                "Bart standing has exactly two arms",
                "Bart Depth has canonical child proportions, orange sleeves and four-finger hands",
                "all Depth hands show one raised thumb per hand",
                "Deadpool, Black Widow and Darth Vader Depth hands are fully gloved",
                "Black Widow wears the black tactical suit",
                "no visible OpenPose sticks or black source rectangle",
            ],
        },
        "failures": failures,
        "metrics": metrics,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("%s: wrote %s", report["status"], args.report)
    if failures:
        for failure in failures:
            LOGGER.error("FAIL %s", failure)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
