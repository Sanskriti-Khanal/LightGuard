"""Load vectorized EMBER2024 feature arrays for a requested split.

PE-type filtering is enforced at download time: the Colab notebook downloads
only the file types listed in config.pe_file_types (default: Win32, Win64),
so the .dat arrays produced by create_vectorized_features already contain only
those types.  No runtime filtering is needed here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import yaml

# thrember is available in the Colab training environment and is stubbed out
# in tests/conftest.py for local test runs.  Top-level import makes it
# patchable via unittest.mock.patch("lightguard.data.loaders.thrember").
import thrember  # noqa: E402  (after stdlib/third-party block)

Split = Literal["train", "test", "challenge"]


def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Read and return the project config dict from *config_path*."""
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def load_split(
    data_dir: str | Path,
    split: Split,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) numpy arrays for *split* from vectorized .dat files.

    Requires thrember and the vectorized .dat files produced by
    create_vectorized_features.  Only available in the Colab training
    environment — not expected to exist on the local machine.

    Args:
        data_dir: directory containing X_{split}.dat and y_{split}.dat.
        split:    one of "train", "test", "challenge".

    Returns:
        X: float32 array of shape (n_samples, n_features).
        y: int32 label array of shape (n_samples,).
    """
    X, y = thrember.read_vectorized_features(str(data_dir), split)
    return X, y


def split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carve a validation set from the END of the training arrays.

    Preserves temporal order — rows arrive in chronological order from
    thrember, so taking the tail keeps the split time-consistent.
    Never shuffles.

    Returns:
        X_train, X_val, y_train, y_val
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    n_val = max(1, int(len(X) * val_fraction))
    split_idx = len(X) - n_val
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
