#!/usr/bin/env python3
"""Reproduce the character matrix through Forge's txt2img API."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import mimetypes
import sys
import urllib.error
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
MANIFEST_PATH = HERE / "demo.json"
LOGGER = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def verify_sources(manifest: dict) -> None:
    pose_hashes = []
    for source_id, source in manifest["sources"].items():
        path = HERE / source["file"]
        actual = sha256(path)
        if actual != source["sha256"]:
            raise RuntimeError(f"{source_id}: SHA-256 mismatch for {path}")
        if source["mode"] == "Pose / OpenPose":
            pose_hashes.append(actual)
    if len(pose_hashes) != 3 or len(set(pose_hashes)) != 3:
        raise RuntimeError("Expected exactly three unique OpenPose source files.")


def verify_models(manifest: dict, forge_root: Path) -> None:
    for model in manifest["models"]:
        path = forge_root / model["file"]
        if not path.is_file():
            raise RuntimeError(f"Missing {model['role']} file: {path}")
        if path.stat().st_size != model["bytes"]:
            raise RuntimeError(f"Size mismatch for {path}")
        if sha256(path) != model["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {path}")
        LOGGER.info("model OK  %s: %s", model["role"], path.name)


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def build_payload(manifest: dict, run: dict) -> dict:
    defaults = dict(manifest["generation_defaults"])
    sources = [manifest["sources"][source_id] for source_id in run["source_ids"]]
    mode = sources[0]["mode"]
    if any(source["mode"] != mode for source in sources):
        raise RuntimeError(f"{run['id']}: mixed control modes are not supported")
    payload = {
        **defaults,
        "prompt": run["prompt"],
        "negative_prompt": run["negative_prompt"],
        "seed": run["seed"],
        "batch_size": run["batch_size"],
    }
    payload["alwayson_scripts"] = {
        manifest["alwayson_script"]: {
            "args": [
                True,
                mode,
                [data_uri(HERE / source["file"]) for source in sources],
                sources[0]["preprocessor"],
                sources[0]["resolution"],
                sources[0]["invert"],
                run["strength"],
            ]
        }
    }
    return payload


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


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://127.0.0.1:7860")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs" / "raw")
    parser.add_argument("--run", action="append", dest="run_ids")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-models", action="store_true")
    parser.add_argument("--forge-root", type=Path)
    args = parser.parse_args()

    manifest = load_manifest()
    verify_sources(manifest)
    if args.verify_models:
        if args.forge_root is None:
            parser.error("--verify-models requires --forge-root")
        verify_models(manifest, args.forge_root.resolve())

    known = {run["id"] for run in manifest["runs"]}
    requested = set(args.run_ids or known)
    unknown = requested - known
    if unknown:
        parser.error(f"unknown run id(s): {', '.join(sorted(unknown))}")

    runs = [run for run in manifest["runs"] if run["id"] in requested]
    LOGGER.info("validated sources; selected %d run(s)", len(runs))
    if args.dry_run:
        for run in runs:
            LOGGER.info(
                "DRY %s: seed=%s batch=%s sources=%s strength=%s",
                run["id"],
                run["seed"],
                run["batch_size"],
                ",".join(run["source_ids"]),
                run["strength"],
            )
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    endpoint = args.api_url.rstrip("/") + "/sdapi/v1/txt2img"
    for run in runs:
        LOGGER.info("generating %s ...", run["id"])
        response = post_json(endpoint, build_payload(manifest, run))
        images = response.get("images", [])
        if len(images) != run["batch_size"]:
            raise RuntimeError(
                f"{run['id']}: expected {run['batch_size']} images, got {len(images)}"
            )
        for index, encoded in enumerate(images):
            raw = base64.b64decode(encoded.split(",", 1)[-1])
            (args.output_dir / f"{run['id']}-{index}.png").write_bytes(raw)
        info = response.get("info", "")
        (args.output_dir / f"{run['id']}.info.json").write_text(
            info if isinstance(info, str) else json.dumps(info, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("wrote %d image(s) for %s", len(images), run["id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
