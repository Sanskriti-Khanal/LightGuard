"""Extract the 2,568-dim EMBER v3 feature vector from a PE file on disk.

Uses thrember.PEFeatureExtractor — the same path as vectorisation during
training, so the resulting vector is directly compatible with the model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import thrember

# Lazy singleton — instantiated on first call so the thrember stub used in tests
# never has PEFeatureExtractor() called at module import time.
_extractor = None


def _get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = thrember.PEFeatureExtractor()
    return _extractor


def extract_features(pe_path: str | Path) -> np.ndarray:
    """Return the 2,568-dim float32 feature vector for the PE file at *pe_path*.

    Reads the raw bytes and feeds them through PEFeatureExtractor.feature_vector,
    which is the same call used by the EMBER2024 vectorisation pipeline.

    Args:
        pe_path: path to a PE file (.exe, .dll, .sys, …).

    Returns:
        float32 array of shape (2568,).

    Raises:
        FileNotFoundError: if *pe_path* does not exist.
        ValueError:        if the extracted vector has unexpected length.
    """
    pe_path = Path(pe_path)
    if not pe_path.exists():
        raise FileNotFoundError(f"PE file not found: {pe_path}")

    extractor = _get_extractor()
    bytez = pe_path.read_bytes()
    vec = extractor.feature_vector(bytez)
    vec = np.asarray(vec, dtype=np.float32)

    if vec.shape != (extractor.dim,):
        raise ValueError(
            f"Unexpected feature vector shape {vec.shape}; "
            f"expected ({_extractor.dim},)"
        )
    return vec
