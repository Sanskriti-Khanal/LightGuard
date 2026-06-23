"""Smoke tests for src/lightguard/data/loaders.py.

All tests run locally against data/sample/ — no thrember, no raw data,
no network.  Tests that exercise the thrember code path mock it explicitly.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lightguard.data.loaders import load_config, load_split, split_train_val
from conftest import load_sample_split


# ── load_config ───────────────────────────────────────────────────────────────

class TestLoadConfig:
    def test_returns_dict(self, config: dict) -> None:
        assert isinstance(config, dict)

    def test_required_keys_present(self, config: dict) -> None:
        for key in ("paths", "random_seed", "lgbm", "split", "scoring", "monitor"):
            assert key in config, f"Missing top-level key: {key}"

    def test_pe_file_types_default(self, config: dict) -> None:
        assert config["pe_file_types"] == ["Win32", "Win64"]

    def test_val_fraction_in_range(self, config: dict) -> None:
        vf = config["split"]["val_fraction"]
        assert 0.0 < vf < 1.0

    def test_alert_threshold_in_range(self, config: dict) -> None:
        t = config["scoring"]["alert_threshold"]
        assert 0.0 < t < 1.0

    def test_nonexistent_config_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "no_such_file.yaml")


# ── load_split (mocked thrember) ──────────────────────────────────────────────

class TestLoadSplit:
    def _fake_arrays(self, n: int = 50, d: int = 10):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, d)).astype(np.float32)
        y = rng.integers(0, 2, n).astype(np.int32)
        return X, y

    def test_returns_numpy_arrays(self) -> None:
        X_fake, y_fake = self._fake_arrays()
        with patch("lightguard.data.loaders.thrember") as mt:
            mt.read_vectorized_features.return_value = (X_fake, y_fake)
            X, y = load_split("some/dir", "train")
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_passes_split_to_thrember(self) -> None:
        X_fake, y_fake = self._fake_arrays()
        with patch("lightguard.data.loaders.thrember") as mt:
            mt.read_vectorized_features.return_value = (X_fake, y_fake)
            load_split("/data", "challenge")
        mt.read_vectorized_features.assert_called_once_with("/data", "challenge")

    def test_shapes_preserved(self) -> None:
        X_fake, y_fake = self._fake_arrays(n=80, d=2381)
        with patch("lightguard.data.loaders.thrember") as mt:
            mt.read_vectorized_features.return_value = (X_fake, y_fake)
            X, y = load_split("d", "test")
        assert X.shape == (80, 2381)
        assert y.shape == (80,)


# ── split_train_val ───────────────────────────────────────────────────────────

class TestSplitTrainVal:
    def _arrays(self, n: int = 200):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 10)).astype(np.float32)
        y = rng.integers(0, 2, n).astype(np.float32)
        return X, y

    def test_sizes_sum_to_original(self) -> None:
        X, y = self._arrays(200)
        X_tr, X_val, y_tr, y_val = split_train_val(X, y, val_fraction=0.1)
        assert len(X_tr) + len(X_val) == 200
        assert len(y_tr) + len(y_val) == 200

    def test_val_size_is_tail(self) -> None:
        X, y = self._arrays(200)
        X_tr, X_val, y_tr, y_val = split_train_val(X, y, val_fraction=0.1)
        expected_val = max(1, int(200 * 0.1))
        assert len(X_val) == expected_val

    def test_temporal_order_preserved(self) -> None:
        """Train rows must be the first rows; val rows must be the last rows."""
        X, y = self._arrays(100)
        X_tr, X_val, y_tr, y_val = split_train_val(X, y, val_fraction=0.2)
        # Concatenating should reconstruct the original order exactly
        np.testing.assert_array_equal(np.concatenate([X_tr, X_val]), X)
        np.testing.assert_array_equal(np.concatenate([y_tr, y_val]), y)

    def test_no_shuffle(self) -> None:
        """Same call twice must produce identical splits."""
        X, y = self._arrays(100)
        r1 = split_train_val(X, y, val_fraction=0.15)
        r2 = split_train_val(X, y, val_fraction=0.15)
        for a, b in zip(r1, r2):
            np.testing.assert_array_equal(a, b)

    def test_invalid_fraction_raises(self) -> None:
        X, y = self._arrays(50)
        with pytest.raises(ValueError):
            split_train_val(X, y, val_fraction=0.0)
        with pytest.raises(ValueError):
            split_train_val(X, y, val_fraction=1.0)
        with pytest.raises(ValueError):
            split_train_val(X, y, val_fraction=1.5)


# ── sample fixture smoke tests ────────────────────────────────────────────────

class TestSampleFixtures:
    """Confirm the committed data/sample/ fixtures are well-formed."""

    def test_all_splits_present(self, sample_dir: Path) -> None:
        for split in ("train", "test", "challenge"):
            assert (sample_dir / f"X_{split}.npy").exists()
            assert (sample_dir / f"y_{split}.npy").exists()

    def test_feature_dim_consistent(self, sample_dir: Path) -> None:
        dims = set()
        for split in ("train", "test", "challenge"):
            X, _ = load_sample_split(sample_dir, split)
            dims.add(X.shape[1])
        assert len(dims) == 1, f"Inconsistent feature dims across splits: {dims}"

    def test_labels_binary(self, sample_dir: Path) -> None:
        for split in ("train", "test", "challenge"):
            _, y = load_sample_split(sample_dir, split)
            assert set(np.unique(y)).issubset({0.0, 1.0}), \
                f"Non-binary labels in {split}: {np.unique(y)}"

    def test_shapes_consistent_within_split(self, sample_dir: Path) -> None:
        for split in ("train", "test", "challenge"):
            X, y = load_sample_split(sample_dir, split)
            assert X.shape[0] == y.shape[0], \
                f"Row count mismatch in {split}: X={X.shape}, y={y.shape}"
            assert X.ndim == 2
            assert y.ndim == 1

    def test_split_train_val_on_sample(self, sample_dir: Path, config: dict) -> None:
        """End-to-end: load sample train split, apply val carve, check sizes."""
        X, y = load_sample_split(sample_dir, "train")
        vf = config["split"]["val_fraction"]
        X_tr, X_val, y_tr, y_val = split_train_val(X, y, val_fraction=vf)
        assert len(X_tr) > 0
        assert len(X_val) > 0
        assert len(X_tr) + len(X_val) == len(X)
