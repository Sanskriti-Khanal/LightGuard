#!/usr/bin/env python
"""Verify the SHAP explanation engine across the full risk spectrum.

Loads the trained model and the held-out sample vectors from data/sample/,
then prints a verdict + plain-English SHAP explanation for four hand-picked
cases:

  1. Clear BENIGN        — lowest risk score in the sample
  2. Mid-risk / borderline — score nearest 50 (the hardest call)
  3. Clear MALICIOUS     — highest risk score (~100/100)
  4. False positive      — a truly-benign row the model scored high
                           (if none exists: nearest benign below the threshold)

No PE files are required — everything runs on pre-extracted feature vectors.

Usage (from repo root):
    .venv/bin/python scripts/risk_spectrum_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import yaml

# ── config ────────────────────────────────────────────────────────────────────
cfg       = yaml.safe_load((REPO / "config.yaml").read_text())
THRESHOLD = cfg["scoring"]["alert_threshold"]
SEED      = cfg["random_seed"]
TOP_K     = 5

# ── load artifacts ────────────────────────────────────────────────────────────
from lightguard.malware.train import load_model
from lightguard.explain.explainer import build_explainer, explain_prediction, load_background
from lightguard.explain.translate import translate

MODEL_PATH = REPO / "models" / "lightguard_lgbm.txt"
SAMPLE_DIR = REPO / "data" / "sample"

# prefer holdout slice if the Colab notebook exported one
_xname = "test_holdout_X.npy" if (SAMPLE_DIR / "test_holdout_X.npy").exists() else "X_test.npy"
_yname = "test_holdout_y.npy" if (SAMPLE_DIR / "test_holdout_y.npy").exists() else "y_test.npy"

print(f"[risk-spectrum] Model  : {MODEL_PATH.name}")
print(f"[risk-spectrum] Vectors: {_xname}")

model = load_model(MODEL_PATH)

X = np.load(SAMPLE_DIR / _xname).astype(np.float32)
y = np.load(SAMPLE_DIR / _yname).astype(np.float32)

print(f"[risk-spectrum] Loaded {len(X)} rows  "
      f"({int((y==0).sum())} benign / {int((y==1).sum())} malicious)\n")

# ── score every row ───────────────────────────────────────────────────────────
probs = model.predict(X, num_iteration=model.best_iteration).astype(np.float64)

benign_idx    = np.where(y == 0)[0]
malicious_idx = np.where(y == 1)[0]

# case 1: clear benign
cb_row = int(benign_idx[np.argmin(probs[benign_idx])])

# case 2: mid-risk (score nearest 0.50 — may be truly benign or malicious)
mid_row = int(np.argmin(np.abs(probs - 0.50)))

# case 3: clear malicious
cm_row = int(malicious_idx[np.argmax(probs[malicious_idx])])

# case 4: false positive (benign row scored >= threshold)
fp_candidates = benign_idx[probs[benign_idx] >= THRESHOLD]
if len(fp_candidates) > 0:
    # pick the most dramatic one — highest score despite being benign
    fp_row    = int(fp_candidates[np.argmax(probs[fp_candidates])])
    fp_is_real = True
else:
    # no FP in this sample — use the nearest benign below the threshold
    below  = benign_idx[probs[benign_idx] < THRESHOLD]
    fp_row = int(below[np.argmax(probs[below])])
    fp_is_real = False

# ── build explainer (background = random 100-row subset, excluding targets) ──
target_rows = {cb_row, mid_row, cm_row, fp_row}
bg_pool     = [i for i in range(len(X)) if i not in target_rows]
rng         = np.random.default_rng(SEED)
bg_idx      = rng.choice(bg_pool, size=min(100, len(bg_pool)), replace=False)
background  = X[bg_idx]
explainer   = build_explainer(model, background)

# ── helpers ───────────────────────────────────────────────────────────────────
def _confidence(prob: float) -> str:
    dist = abs(prob - THRESHOLD)
    if dist >= 0.30: return "HIGH"
    if dist >= 0.15: return "MEDIUM"
    return "LOW"

_SEP = "─" * 72

def _print_case(title: str, row: int, note: str = "") -> None:
    prob      = probs[row]
    true_str  = "MALICIOUS" if y[row] == 1 else "BENIGN"
    pred_str  = "MALICIOUS" if prob >= THRESHOLD else "BENIGN"
    correct   = pred_str == true_str
    verdict   = "✓ correct" if correct else "✗ WRONG  ← model error"
    conf      = _confidence(prob)
    risk      = round(prob * 100)

    bar_filled = int(prob * 40)
    bar = ("█" * bar_filled).ljust(40, "░")

    top = explain_prediction(explainer, X[row], top_k=TOP_K)
    reasons = translate(top)

    print(_SEP)
    print(f"  {title}")
    if note:
        print(f"  {note}")
    print(_SEP)
    print(f"  Row          : {row}")
    print(f"  True label   : {true_str}")
    print(f"  Risk score   : {risk:3d}/100  [{bar}]")
    print(f"  Predicted    : {pred_str}  ({conf} confidence)  {verdict}")
    print()
    print(f"  Top-{TOP_K} SHAP reasons:")
    for r in reasons:
        print(f"    • {r}")
    print()

# ── print all four cases ──────────────────────────────────────────────────────
print(_SEP)
print("  LightGuard — explanation engine: risk-spectrum verification")
print(f"  threshold={THRESHOLD}  |  background={len(background)} rows  |  top_k={TOP_K}")
print(_SEP)
print()

_print_case(
    "CASE 1 — Clear BENIGN  (lowest risk score)",
    cb_row,
    "The model is most confident this file is safe.",
)

_print_case(
    "CASE 2 — Borderline / MID-RISK  (score nearest 50/100)",
    mid_row,
    f"True label is {'MALICIOUS' if y[mid_row]==1 else 'BENIGN'} — "
    "this is the hardest call the model has to make.",
)

_print_case(
    "CASE 3 — Clear MALICIOUS  (highest risk score)",
    cm_row,
    "The model is most confident this file is malicious.",
)

if fp_is_real:
    _print_case(
        "CASE 4 — FALSE POSITIVE  (benign file scored high)",
        fp_row,
        "The model is WRONG here. Check whether the SHAP reasons still make "
        "sense — they should describe real structural traits that look suspicious,\n"
        "  even though the file is not actually malware.",
    )
else:
    _print_case(
        "CASE 4 — Nearest benign below threshold  (no FP found in this sample)",
        fp_row,
        f"No false positives at threshold={THRESHOLD} in this {len(X)}-row sample. "
        "Showing the closest benign row instead.",
    )

print(_SEP)
print("  Done. All four cases explained using only local vectors — no network calls.")
print(_SEP)
