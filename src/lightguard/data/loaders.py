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

# EMBER2024 PE feature vector dimensionality (verified against PEFeatureExtractor.dim)
N_FEATURES: int = 2568


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


def load_split_memmap(
    data_dir: str | Path,
    split: Split,
    n_features: int = N_FEATURES,
) -> tuple[np.memmap, np.memmap]:
    """Open (X, y) for *split* as read-only memmaps — no data copied to RAM.

    Used in Colab so that only the pages LightGBM/sklearn actually touch are
    loaded by the OS, keeping peak RAM flat even for 700K-row splits.

    Args:
        data_dir:   directory containing X_{split}.dat and y_{split}.dat.
        split:      one of "train", "test", "challenge".
        n_features: feature vector width (default: 2568 for EMBER2024).

    Returns:
        X: float32 memmap of shape (n_rows, n_features).
        y: float32 memmap of shape (n_rows,).
    """
    data_dir = Path(data_dir)
    X_path = data_dir / f"X_{split}.dat"
    y_path = data_dir / f"y_{split}.dat"

    if not X_path.exists():
        raise FileNotFoundError(f"Feature array not found: {X_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Label array not found: {y_path}")

    n_rows = y_path.stat().st_size // 4  # float32 = 4 bytes
    X = np.memmap(X_path, dtype=np.float32, mode="r", shape=(n_rows, n_features))
    y = np.memmap(y_path, dtype=np.float32, mode="r", shape=(n_rows,))
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
