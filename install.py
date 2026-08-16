"""Install the small DWPose Python wrapper without downgrading Forge deps."""

from __future__ import annotations

from importlib import metadata

import launch


try:
    installed_version = metadata.version("easy-dwpose")
except metadata.PackageNotFoundError:
    installed_version = None

if installed_version != "1.0.2":
    # easy-dwpose pins old numpy/huggingface_hub versions even though its small
    # runtime wrapper is compatible with Forge's newer shared environment.
    # Installing without dependencies prevents it from downgrading Forge.
    launch.run_pip(
        "install --no-deps easy-dwpose==1.0.2",
        "easy-dwpose 1.0.2 (Krea 2 Pose preprocessor)",
    )
