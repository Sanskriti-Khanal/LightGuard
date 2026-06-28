"""Network anomaly detector for LightGuard v2.

Fits a scikit-learn IsolationForest on the per-process features produced by
collector.py and scores live snapshots against it.

Typical workflow::

    from lightguard.network.collector import collect_baseline, load_baseline
    from lightguard.network.detector import train, score, save_detector, load_detector

    baseline_df  = load_baseline()
    detector     = train(baseline_df)
    save_detector(detector)

    snap         = collect_snapshot()
    results      = score(snap, detector)
    # results has: pid, proc_name, anomaly_score (0-100), label, + feature cols

Feature columns used (order is fixed and stored on the detector):
  conn_count, unique_remote_ips, unique_remote_ports, rare_port_count, conn_per_min
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# The five features passed to IsolationForest — order must not change.
FEATURE_COLS: list[str] = [
    "conn_count",
    "unique_remote_ips",
    "unique_remote_ports",
    "rare_port_count",
    "conn_per_min",
]

_DEFAULT_DETECTOR_PATH = Path("data") / "network" / "detector.joblib"

# Anomaly scores >= this threshold are labelled ANOMALOUS (out of 100).
DEFAULT_THRESHOLD: int = 50


@dataclass
class NetworkDetector:
    """Fitted detector bundled with the scaler and baseline statistics.

    Attributes:
        model:           fitted IsolationForest.
        scaler:          StandardScaler fitted on the same baseline data.
        feature_cols:    ordered list of feature column names.
        baseline_stats:  per-feature dict of {"mean": float, "std": float}
                         used by explain.py to describe deviations.
        threshold:       anomaly_score cutoff for ANOMALOUS label.
        contamination:   IsolationForest contamination parameter used at fit time.
    """
    model:          IsolationForest
    scaler:         StandardScaler
    feature_cols:   list[str]
    baseline_stats: dict[str, dict[str, float]]
    threshold:      int  = DEFAULT_THRESHOLD
    contamination:  float = 0.1


def train(
    baseline_df: pd.DataFrame,
    contamination: float = 0.1,
    random_seed: int = 42,
    threshold: int = DEFAULT_THRESHOLD,
) -> NetworkDetector:
    """Fit an IsolationForest on *baseline_df* and return a NetworkDetector.

    Args:
        baseline_df:   DataFrame from collect_baseline() or load_baseline().
                       Must contain all columns in FEATURE_COLS.
        contamination: expected fraction of anomalies in future snapshots.
                       Passed directly to IsolationForest.
        random_seed:   controls IsolationForest reproducibility — must come
                       from config.yaml in production (CLAUDE.md rule 4).
        threshold:     anomaly_score >= threshold → ANOMALOUS label (0-100).

    Returns:
        Fitted NetworkDetector ready for score().

    Raises:
        ValueError: if baseline_df is missing required feature columns or is empty.
    """
    missing = [c for c in FEATURE_COLS if c not in baseline_df.columns]
    if missing:
        raise ValueError(f"baseline_df missing columns: {missing}")
    if len(baseline_df) == 0:
        raise ValueError("baseline_df is empty — collect a baseline first.")

    X = baseline_df[FEATURE_COLS].fillna(0).astype(float).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=contamination,
        random_state=random_seed,
        n_estimators=100,
    )
    model.fit(X_scaled)

    # Store per-feature mean/std from the raw (unscaled) baseline for explain.py.
    stats: dict[str, dict[str, float]] = {}
    for i, col in enumerate(FEATURE_COLS):
        col_vals = X[:, i]
        stats[col] = {
            "mean": float(np.mean(col_vals)),
            "std":  float(np.std(col_vals)),
        }

    return NetworkDetector(
        model=model,
        scaler=scaler,
        feature_cols=FEATURE_COLS,
        baseline_stats=stats,
        threshold=threshold,
        contamination=contamination,
    )


def score(
    snapshot_df: pd.DataFrame,
    detector: NetworkDetector,
) -> pd.DataFrame:
    """Score each process in *snapshot_df* against the fitted detector.

    Processes with feature columns missing from *snapshot_df* are scored as
    0 (NORMAL).  Processes in the snapshot that were not seen at baseline time
    are still scored — IsolationForest generalises to unseen points.

    Args:
        snapshot_df: DataFrame from collect_snapshot() or collect_baseline().
        detector:    NetworkDetector from train() or load_detector().

    Returns:
        DataFrame with one row per process, columns:
          pid, proc_name, anomaly_score (int 0-100), label, conn_count,
          unique_remote_ips, unique_remote_ports, rare_port_count, conn_per_min
        Sorted by anomaly_score descending.
    """
    if snapshot_df.empty:
        return pd.DataFrame(columns=(
            ["pid", "proc_name", "anomaly_score", "label"] + FEATURE_COLS
        ))

    # Fill any missing feature columns with 0.
    for col in FEATURE_COLS:
        if col not in snapshot_df.columns:
            snapshot_df = snapshot_df.copy()
            snapshot_df[col] = 0.0

    X_raw = snapshot_df[FEATURE_COLS].fillna(0).astype(float).values
    X_scaled = detector.scaler.transform(X_raw)

    # decision_function: more negative = more anomalous.
    # Map to [0, 100] where 100 = most anomalous.
    # Empirical range of IsolationForest decision_function is roughly [-0.5, 0.5];
    # we clip after mapping so extreme values stay in range.
    raw_scores = detector.model.decision_function(X_scaled)
    anomaly_scores = np.clip((0.5 - raw_scores) * 100, 0, 100).astype(int)

    labels = [
        "ANOMALOUS" if s >= detector.threshold else "NORMAL"
        for s in anomaly_scores
    ]

    result = snapshot_df[["pid", "proc_name"] + FEATURE_COLS].copy()
    result.insert(2, "anomaly_score", anomaly_scores)
    result.insert(3, "label", labels)

    return result.sort_values("anomaly_score", ascending=False).reset_index(drop=True)


def save_detector(
    detector: NetworkDetector,
    path: str | Path | None = None,
) -> Path:
    """Persist *detector* to disk with joblib.

    Args:
        detector: NetworkDetector from train().
        path:     destination file.  Defaults to data/network/detector.joblib.

    Returns:
        Resolved path that was written.
    """
    dest = Path(path) if path else _DEFAULT_DETECTOR_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(detector, dest)
    return dest.resolve()


def load_detector(path: str | Path | None = None) -> NetworkDetector:
    """Load a persisted NetworkDetector from disk.

    Args:
        path: file to load.  Defaults to data/network/detector.joblib.

    Raises:
        FileNotFoundError: if no detector file exists at *path*.
    """
    src = Path(path) if path else _DEFAULT_DETECTOR_PATH
    if not src.exists():
        raise FileNotFoundError(
            f"No detector found at {src}. "
            "Run train() and save_detector() first."
        )
    return joblib.load(src)
