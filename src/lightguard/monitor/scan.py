"""Scan a single PE file and return a human-readable verdict.

Pipeline: read bytes → extract features → model.predict → verdict dict.
Runs fully offline; no network calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lightguard.monitor.extract import extract_features


@dataclass(frozen=True)
class Verdict:
    """Result of scanning one file."""
    filename: str       # basename of the scanned file
    risk_score: int     # 0-100 (raw model probability × 100, rounded)
    label: str          # "MALICIOUS" or "BENIGN"
    confidence: str     # "HIGH" / "MEDIUM" / "LOW"
    raw_prob: float     # model output in [0, 1] before scaling

    def __str__(self) -> str:
        return (
            f"{self.filename}  →  {self.label}  "
            f"(risk {self.risk_score}/100, confidence {self.confidence})"
        )


def _confidence(prob: float, threshold: float) -> str:
    """Map distance from the decision boundary to a confidence label."""
    distance = abs(prob - threshold)
    if distance >= 0.30:
        return "HIGH"
    if distance >= 0.15:
        return "MEDIUM"
    return "LOW"


def scan(
    pe_path: str | Path,
    model,
    threshold: float = 0.80,
) -> Verdict:
    """Extract features from *pe_path* and score it with *model*.

    Args:
        pe_path:   path to the PE file to scan.
        model:     loaded lgb.Booster (from lightguard.malware.train.load_model).
        threshold: risk probability above which the file is labelled MALICIOUS.
                   Defaults to config.scoring.alert_threshold (0.80).

    Returns:
        Verdict dataclass with filename, risk_score, label, and confidence.
    """
    pe_path = Path(pe_path)
    vec = extract_features(pe_path)

    prob = float(model.predict(vec.reshape(1, -1), num_iteration=model.best_iteration)[0])
    label = "MALICIOUS" if prob >= threshold else "BENIGN"

    return Verdict(
        filename=pe_path.name,
        risk_score=round(prob * 100),
        label=label,
        confidence=_confidence(prob, threshold),
        raw_prob=prob,
    )
