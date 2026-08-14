"""SHAP-based local explainer for the LightGBM malware classifier.

Designed to run on the laptop using only:
  - the saved model file (models/lightguard.txt)
  - a small background sample exported during Colab training

No full EMBER2024 dataset is required at inference time.

Usage::

    from lightguard.malware.train import load_model
    from lightguard.explain.explainer import build_explainer, explain_prediction
    from lightguard.explain.translate import translate

    model = load_model("models/lightguard.txt")
    background = np.load("data/sample/test_holdout_X.npy")
    explainer = build_explainer(model, background)

    x = ...  # shape (2568,)
    top_features = explain_prediction(explainer, x, top_k=10)
    print(translate(top_features))
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import shap
import lightgbm as lgb

from lightguard.explain.translate import FEATURE_NAMES


def load_background(
    data_dir: str | Path = "data/sample",
    n: int = 200,
    seed: int = 42,
) -> np.ndarray | None:
    """Load a small background sample for the SHAP explainer from disk.

    Tries (in order):
      1. data_dir/test_holdout_X.npy  — exported by the Colab notebook
      2. data_dir/X_test.npy          — always present after make_sample.py

    Returns None if neither file exists so callers can skip explanation gracefully.
    """
    data_dir = Path(data_dir)
    for candidate in ("test_holdout_X.npy", "X_test.npy"):
        p = data_dir / candidate
        if p.exists():
            X = np.load(p)
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
            return X[idx].astype(np.float32)
    return None


def build_explainer(
    model: lgb.Booster,
    background: np.ndarray,
) -> shap.TreeExplainer:
    """Create a SHAP TreeExplainer for *model* using *background* as the reference.

    Background should be a representative sample (~100–2000 rows) drawn from
    the test set.  Interventional perturbation mode is used so that the
    background acts as the marginal distribution for feature imputation.
    """
    return shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
    )


def explain_prediction(
    explainer: shap.TreeExplainer,
    x: np.ndarray,
    top_k: int = 10,
) -> list[dict]:
    """Return the top-k features driving the model's prediction for sample *x*.

    Args:
        explainer: built by build_explainer().
        x:         1-D feature vector of shape (n_features,).
        top_k:     number of features to return (sorted by |SHAP| descending).

    Returns:
        List of dicts, each with keys:
          'feature_idx'  — int, index into FEATURE_NAMES
          'feature_name' — str
          'shap_value'   — float, signed contribution (positive = towards malicious)
          'raw_value'    — float, the original feature value
    """
    x2d = x.reshape(1, -1)
    shap_values = explainer.shap_values(x2d)

    # LightGBM binary classification returns shape (n_samples, n_features)
    if isinstance(shap_values, list):
        # Older SHAP versions return a list [neg_class, pos_class]
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    top_indices = np.argsort(np.abs(sv))[::-1][:top_k]

    return [
        {
            "feature_idx": int(i),
            "feature_name": FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}",
            "shap_value": float(sv[i]),
            "raw_value": float(x[i]),
        }
        for i in top_indices
    ]
