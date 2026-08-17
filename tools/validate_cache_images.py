"""Exercise the persistent cache with the repository's real image fixtures.

This is an opt-in runtime check because Depth Anything V2 and DWPose load their
real local models. It never downloads inputs or modifies the tracked fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
FORGE_ROOT = REPO.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(FORGE_ROOT))
sys.path.insert(0, str(FORGE_ROOT / "modules_forge" / "packages"))
sys.path.insert(
    0, str(FORGE_ROOT / "extensions-builtin" / "forge_legacy_preprocessors")
)

from forge_krea2_depth.images import fit_control_map, normalize_image, preprocess_depth_map
from forge_krea2_depth.pose import preprocess_pose_map
from forge_krea2_depth.preprocessor_cache import (
    ControlMapCache,
    build_control_map_cache_key,
    prepare_control_source,
)


DEPTH_MODE = "Depth"
POSE_MODE = "Pose / OpenPose"
DEPTH_DIRECT = "None (already a depth map)"
POSE_DIRECT = "None (already an OpenPose map)"
DWPOSE = "DWPose (photo to pose)"


def digest(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def cached_call(
    cache: ControlMapCache,
    calls: Counter,
    label: str,
    source,
    mode: str,
    preprocessor: str,
    resolution: int,
    invert: bool,
    compute,
) -> tuple[np.ndarray, bool, float]:
    snapshot, source_digest = prepare_control_source(source)
    key = build_control_map_cache_key(
        snapshot,
        mode,
        preprocessor,
        resolution,
        invert if mode == DEPTH_MODE else False,
        source_digest=source_digest,
    )

    def invoke() -> np.ndarray:
        calls[label] += 1
        return np.ascontiguousarray(compute(snapshot))

    started = time.perf_counter()
    result, hit = cache.get_or_compute(key, invoke)
    return result, hit, time.perf_counter() - started


def save_image(directory: Path, name: str, image: np.ndarray) -> None:
    Image.fromarray(normalize_image(image), mode="RGB").save(directory / name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()
    artifacts = args.artifacts_dir
    temporary = None
    if artifacts is None:
        temporary = tempfile.TemporaryDirectory(prefix="krea-cache-images-")
        artifacts = Path(temporary.name)
    artifacts.mkdir(parents=True, exist_ok=True)

    sources = REPO / "examples" / "character-matrix" / "sources"
    dance = sources / "openpose-dance.png"
    jump = sources / "openpose-jump.png"
    standing = sources / "openpose-standing.png"
    depth_map = sources / "depth-thumbs.webp"
    matrix = Image.open(REPO / "assets" / "forge-character-control-matrix.webp")
    pose_photo_a = np.asarray(matrix.crop((462, 487, 756, 781)).convert("RGB"))
    pose_photo_b = np.asarray(matrix.crop((1094, 165, 1388, 459)).convert("RGB"))
    robot_strip = Image.open(REPO / "assets" / "image.webp")
    depth_photo = np.asarray(robot_strip.crop((0, 0, 1024, 1024)).convert("RGB"))

    cache = ControlMapCache()
    calls: Counter = Counter()
    checks = []
    timings = {}

    def check(condition: bool, name: str) -> None:
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def direct_pose(source):
        return cached_call(
            cache,
            calls,
            "pose-direct",
            source,
            POSE_MODE,
            POSE_DIRECT,
            args.resolution,
            False,
            lambda snapshot: preprocess_pose_map(
                snapshot, POSE_DIRECT, args.resolution, FORGE_ROOT / "models"
            ),
        )

    direct_results = [direct_pose(source) for source in (standing, dance, jump)]
    direct_repeat = [direct_pose(source) for source in (standing, dance, jump)]
    check(all(not item[1] for item in direct_results), "direct pose cold misses")
    check(all(item[1] for item in direct_repeat), "direct pose warm hits")
    check(calls["pose-direct"] == 3, "three unique direct pose computations")
    check(
        all(
            np.array_equal(cold[0], warm[0])
            for cold, warm in zip(direct_results, direct_repeat)
        ),
        "direct pose cache is pixel exact",
    )

    first_depth, first_hit, _ = cached_call(
        cache,
        calls,
        "depth-direct",
        depth_map,
        DEPTH_MODE,
        DEPTH_DIRECT,
        args.resolution,
        False,
        lambda snapshot: preprocess_depth_map(
            snapshot, DEPTH_DIRECT, args.resolution, False
        ),
    )
    inverted_depth, inverted_hit, _ = cached_call(
        cache,
        calls,
        "depth-direct",
        depth_map,
        DEPTH_MODE,
        DEPTH_DIRECT,
        args.resolution,
        True,
        lambda snapshot: preprocess_depth_map(
            snapshot, DEPTH_DIRECT, args.resolution, True
        ),
    )
    check(not first_hit and not inverted_hit, "depth inversion has a distinct key")
    check(
        np.array_equal(inverted_depth, 255 - first_depth),
        "depth inversion output is exact",
    )

    with tempfile.TemporaryDirectory(prefix="krea-cache-rewrite-") as directory:
        rewrite_cache = ControlMapCache()
        changing = Path(directory) / "same-path.png"
        changing.write_bytes(dance.read_bytes())

        def rewritten_pose():
            return cached_call(
                rewrite_cache,
                calls,
                "pose-rewritten-path",
                changing,
                POSE_MODE,
                POSE_DIRECT,
                args.resolution,
                False,
                lambda snapshot: preprocess_pose_map(
                    snapshot, POSE_DIRECT, args.resolution, FORGE_ROOT / "models"
                ),
            )

        old, old_hit, _ = rewritten_pose()
        changing.write_bytes(jump.read_bytes())
        changed, changed_hit, _ = rewritten_pose()
        changing.write_bytes(dance.read_bytes())
        revisited, revisited_hit, _ = rewritten_pose()
    check(not old_hit and not changed_hit, "same-path changed pixels miss")
    check(revisited_hit, "same-path restored pixels hit")
    check(not np.array_equal(old, changed), "changed file yields changed map")
    check(np.array_equal(old, revisited), "restored file reuses exact old map")

    from legacy_preprocessors import preprocessor as legacy

    def estimate_depth(snapshot, resolution):
        estimated, _display = legacy.depth_anything_v2(
            snapshot, res=resolution, colored=False
        )
        return normalize_image(estimated)

    depth_cold, depth_cold_hit, depth_cold_time = cached_call(
        cache,
        calls,
        f"depth-estimator-{args.resolution}",
        depth_photo,
        DEPTH_MODE,
        "depth_anything_v2",
        args.resolution,
        False,
        lambda snapshot: estimate_depth(snapshot, args.resolution),
    )
    depth_warm, depth_warm_hit, depth_warm_time = cached_call(
        cache,
        calls,
        f"depth-estimator-{args.resolution}",
        depth_photo.copy(),
        DEPTH_MODE,
        "depth_anything_v2",
        args.resolution,
        False,
        lambda snapshot: estimate_depth(snapshot, args.resolution),
    )
    check(not depth_cold_hit and depth_warm_hit, "real Depth cold then warm")
    check(
        calls[f"depth-estimator-{args.resolution}"] == 1,
        "real Depth executes once",
    )
    check(np.array_equal(depth_cold, depth_warm), "real Depth hit is pixel exact")
    check(np.unique(depth_cold).size > 32, "real Depth output has useful range")
    check(depth_warm_time < depth_cold_time, "real Depth cache hit is faster")
    timings["depth_cold_seconds"] = depth_cold_time
    timings["depth_warm_seconds"] = depth_warm_time

    def estimate_pose(snapshot):
        return preprocess_pose_map(
            snapshot, DWPOSE, args.resolution, FORGE_ROOT / "models"
        )

    pose_sequence = [pose_photo_a, pose_photo_b, pose_photo_a]
    pose_outputs = []
    pose_times = []
    pose_hits = []
    for cycle in range(2):
        for index, photo in enumerate(pose_sequence):
            output, hit, elapsed = cached_call(
                cache,
                calls,
                "dwpose",
                photo,
                POSE_MODE,
                DWPOSE,
                args.resolution,
                False,
                estimate_pose,
            )
            pose_outputs.append(output)
            pose_hits.append(hit)
            pose_times.append(elapsed)
            save_image(artifacts, f"dwpose-cycle{cycle + 1}-{index + 1}.png", output)
    check(pose_hits == [False, False, True, True, True, True], "DWPose A/B/A sequence hits")
    check(calls["dwpose"] == 2, "two unique DWPose photos execute twice")
    check(np.count_nonzero(pose_outputs[0]) > 500, "real DWPose output is non-empty")
    check(
        np.array_equal(pose_outputs[0], pose_outputs[2])
        and np.array_equal(pose_outputs[0], pose_outputs[3]),
        "DWPose repeated A is pixel exact",
    )
    timings["dwpose_cold_a_seconds"] = pose_times[0]
    timings["dwpose_cold_b_seconds"] = pose_times[1]
    timings["dwpose_warm_median_seconds"] = float(np.median(pose_times[2:]))
    check(
        timings["dwpose_warm_median_seconds"]
        < min(timings["dwpose_cold_a_seconds"], timings["dwpose_cold_b_seconds"]),
        "real DWPose cache hits are faster",
    )

    fitted_square = fit_control_map(depth_cold, 512, 512)
    fitted_wide = fit_control_map(depth_cold, 768, 512)
    check(fitted_square.shape == (512, 512, 3), "cached raw map fits square canvas")
    check(fitted_wide.shape == (512, 768, 3), "cached raw map refits wide canvas")

    save_image(artifacts, "source-depth-photo.png", depth_photo)
    save_image(artifacts, "source-pose-a.png", pose_photo_a)
    save_image(artifacts, "source-pose-b.png", pose_photo_b)
    save_image(artifacts, "depth-estimated.png", depth_cold)
    save_image(artifacts, "depth-fitted-square.png", fitted_square)
    save_image(artifacts, "depth-fitted-wide.png", fitted_wide)
    report = {
        "checks": checks,
        "calls": dict(calls),
        "timings": timings,
        "cache": cache.info(),
        "digests": {
            "depth": digest(depth_cold),
            "dwpose_a": digest(pose_outputs[0]),
            "dwpose_b": digest(pose_outputs[1]),
        },
        "artifacts": str(artifacts.resolve()),
    }
    (artifacts / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))  # noqa: T201 - CLI report

    legacy.unload_depth_anything_v2()
    if temporary is not None:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
