from __future__ import annotations

from pathlib import Path

import pytest
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    REPO_ROOT.parent.parent
    / "models"
    / "ControlNet"
    / "Krea2"
    / "depth-control-lora.safetensors"
)


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="official runtime model not installed")
def test_official_checkpoint_has_complete_expected_layout():
    with safe_open(CHECKPOINT, framework="pt") as handle:
        keys = set(handle.keys())
        assert len(keys) == 450
        assert tuple(handle.get_slice("first.weight").get_shape()) == (6144, 128)
        assert tuple(handle.get_slice("first.bias").get_shape()) == (6144,)
        for block in range(28):
            for target in (
                "attn.wq",
                "attn.wk",
                "attn.wv",
                "attn.wo",
                "attn.gate",
                "mlp.gate",
                "mlp.up",
                "mlp.down",
            ):
                assert f"blocks.{block}.{target}.A" in keys
                assert f"blocks.{block}.{target}.B" in keys
