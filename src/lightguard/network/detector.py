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
# Set lower than 50 because IsolationForest alone is unreliable on small
# baselines; the rule-based component catches clear outliers first.
DEFAULT_THRESHOLD: int = 40

# Rule-based score: max z-score across features × this factor → score 0-100.
# z=4 → 40 (just above threshold), z=10 → 100.
_RULE_SCALE: float = 10.0


@dataclass
class NetworkDetector:
    """Fitted detector bundled with the scaler and baseline statistics.

    Attributes:
        model:           fitted IsolationForest.
        scaler:          StandardScaler fitted on the same baseline data.
        feature_cols:    ordered list of feature column names.
        baseline_stats:  per-feature dict of {"mean": float, "std": float}
                         used by explain.py to describe deviations.
        score_p95:       95th-percentile decision_function score on the baseline
                         (the "most normal" anchor for score normalisation).
        score_spread:    p95 minus the baseline minimum df score; sets the
                         denominator for normalising IsolationForest outputs.
        threshold:       anomaly_score cutoff for ANOMALOUS label.
        contamination:   IsolationForest contamination parameter used at fit time.
    """
    model:          IsolationForest
    scaler:         StandardScaler
    feature_cols:   list[str]
    baseline_stats: dict[str, dict[str, float]]
    score_p95:      float = 0.5
    score_spread:   float = 1.0
    threshold:      int   = DEFAULT_THRESHOLD
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

    # Store per-feature mean/std from the raw (unscaled) baseline for explain.py
    # and the rule-based scorer.
    stats: dict[str, dict[str, float]] = {}
    for i, col in enumerate(FEATURE_COLS):
        col_vals = X[:, i]
        stats[col] = {
            "mean": float(np.mean(col_vals)),
            "std":  float(np.std(col_vals)),
        }

    # Calibrate the score normalisation to this baseline's decision_function
    # distribution so scores spread across 0-100 relative to what the model
    # considers "normal" for this machine.
    baseline_df_scores = model.decision_function(X_scaled)
    score_p95   = float(np.percentile(baseline_df_scores, 95))
    score_min   = float(np.min(baseline_df_scores))
    score_spread = max(0.05, score_p95 - score_min)

    return NetworkDetector(
        model=model,
        scaler=scaler,
        feature_cols=FEATURE_COLS,
        baseline_stats=stats,
        score_p95=score_p95,
        score_spread=score_spread,
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

    # ── IsolationForest score ────────────────────────────────────────────────
    # Normalise against the baseline's own decision_function distribution so
    # the full 0-100 range is used regardless of how tight or wide that range is.
    # A process at the baseline p95 (most normal) → ~0; anything more anomalous
    # than the baseline minimum → ≥100.
    raw_scores = detector.model.decision_function(X_scaled)
    if_scores  = np.clip(
        (detector.score_p95 - raw_scores) / detector.score_spread * 100,
        0, 100,
    )

    # ── Rule-based score ────────────────────────────────────────────────────
    # IsolationForest on small / noisy baselines may miss obvious outliers.
    # This deterministic rule catches processes whose worst feature is many
    # standard deviations above the baseline mean regardless of the model's
    # opinion.  rule_score = min(100, max_z * RULE_SCALE).
    stats = detector.baseline_stats
    rule_scores = np.zeros(len(X_raw))
    for i, col in enumerate(detector.feature_cols):
        mean = stats[col]["mean"]
        std  = stats[col]["std"]
        z = (X_raw[:, i] - mean) / std if std > 1e-9 else np.zeros(len(X_raw))
        rule_scores = np.maximum(rule_scores, z * _RULE_SCALE)
    rule_scores = np.clip(rule_scores, 0, 100)

    # Combined: take whichever signal is stronger.
    anomaly_scores = np.maximum(if_scores, rule_scores).astype(int)

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
