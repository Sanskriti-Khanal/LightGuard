#!/usr/bin/env python3
"""Generate committed data/sample/ fixtures used by the local test suite.

Run once from the repo root after initial setup.  Safe to re-run — overwrites
existing fixtures with the same seed, so output is deterministic.

Usage:
    python scripts/make_sample.py
    python scripts/make_sample.py --config path/to/config.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

# EMBER2024 PE feature dimensionality (PEFeatureExtractor.dim in thrember)
N_FEATURES = 2568
N_ROWS_PER_SPLIT = 200  # small enough to be fast; large enough to test splits


def _make_split(
    rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray]:
    X = rng.standard_normal((n, N_FEATURES)).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(np.float32)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"[error] Config not found: {config_path}")

    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)

    seed: int = cfg["random_seed"]
    sample_dir = Path(cfg["paths"]["data_sample"])
    sample_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    for split in ("train", "test", "challenge"):
        X, y = _make_split(rng, N_ROWS_PER_SPLIT)
        np.save(sample_dir / f"X_{split}.npy", X)
        np.save(sample_dir / f"y_{split}.npy", y)
        print(f"  {split:<12}: X{X.shape}  y{y.shape}  → {sample_dir}")

    print(f"\nSample fixtures written to {sample_dir.resolve()}")
    print("Run: PYTHONPATH=src pytest tests/ -v")


if __name__ == "__main__":
    main()
