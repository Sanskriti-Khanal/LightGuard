"""Plain-English explanations for network anomalies (LightGuard v2).

Mirrors the v1 translate.py approach: takes scored process rows and the
baseline statistics stored on the detector, then maps each feature's deviation
from normal into a human-readable sentence.

Usage::

    from lightguard.network.explain import explain_process, explain_snapshot

    results = score(snapshot_df, detector)
    for _, row in results[results.label == "ANOMALOUS"].iterrows():
        reasons = explain_process(row, detector)
        print(row.proc_name, "—", reasons)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from lightguard.network.detector import NetworkDetector

# ── Feature → sentence template ───────────────────────────────────────────────
# Each entry maps a feature name to a pair of (high-deviation, borderline) messages.
# The first string fires when z-score >= HIGH_Z; the second when >= BORDER_Z.

_HIGH_Z   = 1.5   # standard deviations above baseline mean → "far more than usual"
_BORDER_Z = 0.8   # z-score range that triggers the softer wording

_FEATURE_MESSAGES: dict[str, tuple[str, str]] = {
    "conn_per_min": (
        "Reaching out far more often than usual",
        "Making connections more frequently than normal",
    ),
    "rare_port_count": (
        "Using ports not normally used by this app",
        "Contacting an unusual port",
    ),
    "unique_remote_ips": (
        "Contacting far more external addresses than usual",
        "Contacting more external addresses than normal",
    ),
    "unique_remote_ports": (
        "Opening an unusually large number of different ports",
        "Opening more ports than this process normally uses",
    ),
    "conn_count": (
        "Holding far more connections than normal",
        "Holding more connections than usual",
    ),
}

# Absolute-value thresholds used when no baseline stats are available.
# Keys are feature names; values are (high, borderline) raw thresholds.
_ABS_THRESHOLDS: dict[str, tuple[float, float]] = {
    "conn_per_min":       (120.0, 60.0),
    "rare_port_count":    (5.0,   2.0),
    "unique_remote_ips":  (20.0,  10.0),
    "unique_remote_ports": (10.0,  5.0),
    "conn_count":         (50.0,  20.0),
}


def _z_score(value: float, mean: float, std: float) -> float:
    """Signed z-score; returns 0 if std is effectively zero."""
    return (value - mean) / std if std > 1e-9 else 0.0


def explain_process(
    row: pd.Series,
    detector: "NetworkDetector",
    top_k: int = 3,
) -> list[str]:
    """Return up to *top_k* plain-English reasons why *row* looks anomalous.

    Reasons are ranked by how far each feature deviates above its baseline
    mean (using the z-score stored in detector.baseline_stats).  Features that
    are at or below baseline are not mentioned.

    Args:
        row:      a single row from score() output (contains feature values).
        detector: NetworkDetector from train(); its baseline_stats are used
                  to compute z-scores.  If None, absolute thresholds are used.
        top_k:    maximum number of reasons to return.

    Returns:
        List of plain-English reason strings, most severe first.
        Empty list if no feature exceeds the borderline threshold.
    """
    stats = detector.baseline_stats if detector is not None else {}

    deviations: list[tuple[float, str]] = []

    for feat, (high_msg, border_msg) in _FEATURE_MESSAGES.items():
        value = float(row.get(feat, 0.0))

        if stats and feat in stats:
            z = _z_score(value, stats[feat]["mean"], stats[feat]["std"])
            if z >= _HIGH_Z:
                deviations.append((z, high_msg))
            elif z >= _BORDER_Z:
                deviations.append((z, border_msg))
        else:
            # No baseline stats — fall back to absolute thresholds.
            high_thresh, border_thresh = _ABS_THRESHOLDS.get(feat, (float("inf"), float("inf")))
            if value >= high_thresh:
                deviations.append((value, high_msg))
            elif value >= border_thresh:
                deviations.append((value, border_msg))

    # Sort by severity descending, return up to top_k.
    deviations.sort(key=lambda t: t[0], reverse=True)
    return [msg for _, msg in deviations[:top_k]]


def explain_snapshot(
    results_df: pd.DataFrame,
    detector: "NetworkDetector",
    top_k: int = 3,
) -> pd.DataFrame:
    """Add a 'reasons' column to the score() output DataFrame.

    Only anomalous processes get reasons; NORMAL rows get an empty list.

    Args:
        results_df: DataFrame from score(), must have 'label' and feature cols.
        detector:   NetworkDetector from train().
        top_k:      max reasons per process.

    Returns:
        Copy of *results_df* with a 'reasons' column (list[str] per row).
    """
    out = results_df.copy()

    def _reasons(row: pd.Series) -> list[str]:
        if row.get("label") != "ANOMALOUS":
            return []
        reasons = explain_process(row, detector, top_k=top_k)
        # If no z-score reason fired but process is still ANOMALOUS (model-only
        # flag), produce a generic fallback so the UI is never empty.
        if not reasons:
            reasons = ["Unusual combination of connection patterns detected"]
        return reasons

    out["reasons"] = out.apply(_reasons, axis=1)
    return out
