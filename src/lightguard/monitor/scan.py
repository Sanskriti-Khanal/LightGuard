"""Scan a single PE file and return a human-readable verdict.

Pipeline: read bytes → extract features → model.predict → (optional) SHAP explain
→ verdict.  Runs fully offline; no network calls.

Quick start with explanations::

    from lightguard.malware.train import load_model
    from lightguard.explain.explainer import build_explainer, load_background
    from lightguard.monitor.scan import scan

    model      = load_model("models/lightguard_lgbm.txt")
    background = load_background()                      # uses data/sample/
    explainer  = build_explainer(model, background)

    verdict = scan("suspicious.exe", model, explainer=explainer)
    for reason in verdict.reasons:
        print(reason)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lightguard.monitor.extract import extract_features


@dataclass(frozen=True)
class Verdict:
    """Result of scanning one file."""
    filename:   str              # basename of the scanned file
    risk_score: int              # 0-100 (raw model probability × 100, rounded)
    label:      str              # "MALICIOUS" or "BENIGN"
    confidence: str              # "HIGH" / "MEDIUM" / "LOW"
    raw_prob:   float            # model output in [0, 1] before scaling
    reasons:    tuple[str, ...]  # plain-English SHAP explanations (empty if no explainer)

    def __str__(self) -> str:
        base = (
            f"{self.filename}  →  {self.label}  "
            f"(risk {self.risk_score}/100, confidence {self.confidence})"
        )
        if self.reasons:
            reasons_block = "\n".join(f"  • {r}" for r in self.reasons)
            return f"{base}\n{reasons_block}"
        return base


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
    explainer=None,
    top_k: int = 5,
) -> Verdict:
    """Extract features from *pe_path*, score with *model*, and optionally explain.

    Args:
        pe_path:   path to the PE file to scan.
        model:     loaded lgb.Booster (from lightguard.malware.train.load_model).
        threshold: risk probability above which the file is labelled MALICIOUS.
        explainer: optional shap.TreeExplainer — if provided, the top *top_k*
                   SHAP features are translated to plain English and attached as
                   verdict.reasons.
        top_k:     number of SHAP reasons to include (default 5).

    Returns:
        Verdict with filename, risk_score, label, confidence, raw_prob, and
        reasons (empty tuple when no explainer is supplied).
    """
    pe_path = Path(pe_path)
    vec = extract_features(pe_path)

    prob = float(model.predict(vec.reshape(1, -1), num_iteration=model.best_iteration)[0])
    label = "MALICIOUS" if prob >= threshold else "BENIGN"

    reasons: tuple[str, ...] = ()
    if explainer is not None:
        from lightguard.explain.explainer import explain_prediction
        from lightguard.explain.translate import translate
        top_features = explain_prediction(explainer, vec, top_k=top_k)
        reasons = tuple(translate(top_features, verbose=True))

    return Verdict(
        filename=pe_path.name,
        risk_score=round(prob * 100),
        label=label,
        confidence=_confidence(prob, threshold),
        raw_prob=prob,
        reasons=reasons,
    )
