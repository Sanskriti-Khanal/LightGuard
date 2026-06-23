"""Shared pytest fixtures for LightGuard tests."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

# ── thrember stub ─────────────────────────────────────────────────────────────
# thrember is not installed in the local dev environment (it lives in Colab).
# Tests that exercise thrember code paths patch it explicitly via unittest.mock.
# This module-level stub lets `import thrember` succeed at collection time.
if "thrember" not in sys.modules:
    stub = types.ModuleType("thrember")
    stub.__version__ = "0.0.0-stub"          # type: ignore[attr-defined]
    stub.download_dataset = MagicMock()
    stub.download_models = MagicMock()
    stub.create_vectorized_features = MagicMock()
    stub.read_vectorized_features = MagicMock()
    stub.read_metadata = MagicMock()
    stub.train_model = MagicMock()
    sys.modules["thrember"] = stub


# ── Config & paths ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def config() -> dict:
    """Project config loaded from repo-root config.yaml."""
    root = Path(__file__).parent.parent
    with (root / "config.yaml").open() as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="session")
def sample_dir(config: dict) -> Path:
    """Absolute path to data/sample/; skips suite if fixtures are missing."""
    root = Path(__file__).parent.parent
    path = root / config["paths"]["data_sample"]
    npy_files = list(path.glob("*.npy"))
    if not npy_files:
        pytest.skip(
            "data/sample/ has no .npy fixtures — "
            "run `python scripts/make_sample.py` first."
        )
    return path


# ── Sample loader (no thrember required) ─────────────────────────────────────

def load_sample_split(
    sample_dir: Path,
    split: str,
) -> tuple:
    """Load (X, y) numpy arrays from data/sample/ for the given split."""
    import numpy as np

    X = np.load(sample_dir / f"X_{split}.npy")
    y = np.load(sample_dir / f"y_{split}.npy")
    return X, y
