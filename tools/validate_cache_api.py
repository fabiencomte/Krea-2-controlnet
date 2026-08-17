"""Validate Depth and DWPose cache hits through real Forge API generations."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[1]


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Forge API returned HTTP {error.code}: {detail}") from error


def image_data_uri(image: Image.Image) -> str:
    encoded = io.BytesIO()
    image.convert("RGB").save(encoded, format="PNG")
    return "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode(
        "ascii"
    )


def stamp_source(image: Image.Image, nonce: str) -> Image.Image:
    """Make each validation campaign cache-cold without changing its subject."""

    pixels = np.asarray(image.convert("RGB")).copy()
    stamp = np.frombuffer(hashlib.sha256(nonce.encode("utf-8")).digest(), dtype=np.uint8)
    flat = pixels.reshape(-1, 3)
    flat[: len(stamp), 0] = stamp
    return Image.fromarray(pixels, mode="RGB")


def run_case(endpoint: str, directory: Path, case: dict) -> dict:
    payload = {
        "prompt": case["prompt"],
        "negative_prompt": "",
        "seed": case["seed"],
        "steps": 1,
        "sampler_name": "Euler",
        "scheduler": "Simple",
        "cfg_scale": 1.0,
        "width": 128,
        "height": 128,
        "batch_size": 1,
        "n_iter": 1,
        "save_images": False,
        "alwayson_scripts": {
            "Krea 2 Depth / Pose ControlNet-LoRA": {
                "args": [
                    True,
                    case["mode"],
                    [image_data_uri(case["source"])],
                    case["preprocessor"],
                    512,
                    False,
                    0.75,
                ]
            }
        },
    }
    runs = []
    for index in range(2):
        started = time.perf_counter()
        response = post_json(endpoint, payload)
        elapsed = time.perf_counter() - started
        images = response.get("images", [])
        if len(images) != 1:
            raise AssertionError(
                f"{case['name']} API run {index + 1} returned {len(images)} images"
            )
        raw = base64.b64decode(images[0].split(",", 1)[-1])
        (directory / f"{case['name']}-api-run-{index + 1}.png").write_bytes(raw)
        info = response.get("info", "")
        parsed = json.loads(info) if isinstance(info, str) else info
        infotexts = parsed.get("infotexts", [])
        infotext = infotexts[0] if infotexts else info
        runs.append(
            {
                "elapsed_seconds": elapsed,
                "infotext": infotext,
                "pixels": np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")),
            }
        )

    for run, marker in zip(runs, ("0 hit(s), 1 miss(es)", "1 hit(s), 0 miss(es)")):
        if marker not in run["infotext"]:
            raise AssertionError(
                f"{case['name']}: missing `{marker}` in infotext: {run['infotext']}"
            )
    if not np.array_equal(runs[0]["pixels"], runs[1]["pixels"]):
        raise AssertionError(
            f"{case['name']}: same seed/input changed pixels on cache hit"
        )
    if runs[1]["elapsed_seconds"] >= runs[0]["elapsed_seconds"]:
        raise AssertionError(f"{case['name']}: warm generation was not faster")
    return {
        "cold_seconds": runs[0]["elapsed_seconds"],
        "warm_seconds": runs[1]["elapsed_seconds"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://127.0.0.1:7860")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--nonce", default=str(time.time_ns()))
    args = parser.parse_args()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    strip = Image.open(REPO / "assets" / "image.webp")
    matrix = Image.open(REPO / "assets" / "forge-character-control-matrix.webp")
    cases = [
        {
            "name": "depth",
            "source": stamp_source(
                strip.crop((0, 0, 1024, 1024)), f"depth-{args.nonce}"
            ),
            "prompt": "a small friendly toy robot, studio product photo",
            "seed": 24681357,
            "mode": "Depth",
            "preprocessor": "depth_anything_v2",
        },
        {
            "name": "dwpose",
            "source": stamp_source(
                matrix.crop((462, 487, 756, 781)), f"dwpose-{args.nonce}"
            ),
            "prompt": "a masked dancer in a studio, full body",
            "seed": 97531864,
            "mode": "Pose / OpenPose",
            "preprocessor": "DWPose (photo to pose)",
        },
    ]
    endpoint = args.api_url.rstrip("/") + "/sdapi/v1/txt2img"
    results = {
        case["name"]: run_case(endpoint, args.artifacts_dir, case) for case in cases
    }

    report = {
        "checks": [
            "Depth and DWPose first generations report one cache miss",
            "Depth and DWPose second generations report one cache hit",
            "same seed/input remains pixel exact for both modes",
            "warm end-to-end generation is faster for both modes",
        ],
        "results": results,
        "nonce": args.nonce,
        "artifacts": str(args.artifacts_dir.resolve()),
    }
    (args.artifacts_dir / "api-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))  # noqa: T201 - CLI report
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
