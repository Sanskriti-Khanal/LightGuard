"""Tests for src/lightguard/monitor/ (M4 — single-file scanning).

These tests require:
  1. A known-benign PE file at data/sample/benign.exe
  2. A trained model at models/lightguard_lgbm.txt

Both are skipped with a clear message when the file is absent so the test
suite never fails on a fresh clone.  NEVER add real malware here.

To add a benign test binary (run once on a Windows machine or via Wine):
  cp /path/to/notepad.exe data/sample/benign.exe
  git add data/sample/benign.exe
  git commit -m "data: add benign PE fixture for scan tests"
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT  = Path(__file__).parent.parent
BENIGN_EXE = REPO_ROOT / "data" / "sample" / "benign.exe"
MODEL_PATH = REPO_ROOT / "models" / "lightguard_lgbm.txt"

_SKIP_NO_EXE = pytest.mark.skipif(
    not BENIGN_EXE.exists(),
    reason=(
        f"Benign PE fixture missing: {BENIGN_EXE}\n"
        "Add one with:\n"
        "  cp /path/to/notepad.exe data/sample/benign.exe\n"
        "  git add data/sample/benign.exe && git commit"
    ),
)
_SKIP_NO_MODEL = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=(
        f"Trained model missing: {MODEL_PATH}\n"
        "Run the Colab training notebook and copy lightguard_lgbm.txt to models/."
    ),
)


# ── extract ───────────────────────────────────────────────────────────────────

class TestExtract:
    @_SKIP_NO_EXE
    def test_returns_float32_array(self) -> None:
        from lightguard.monitor.extract import extract_features
        vec = extract_features(BENIGN_EXE)
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32

    @_SKIP_NO_EXE
    def test_correct_dimension(self) -> None:
        from lightguard.monitor.extract import extract_features
        vec = extract_features(BENIGN_EXE)
        assert vec.shape == (2568,)

    @_SKIP_NO_EXE
    def test_non_zero_features(self) -> None:
        from lightguard.monitor.extract import extract_features
        vec = extract_features(BENIGN_EXE)
        assert vec.sum() != 0.0

    def test_missing_file_raises(self) -> None:
        from lightguard.monitor.extract import extract_features
        with pytest.raises(FileNotFoundError):
            extract_features("/no/such/file.exe")


# ── scan ──────────────────────────────────────────────────────────────────────

class TestScan:
    @_SKIP_NO_EXE
    @_SKIP_NO_MODEL
    def test_returns_verdict(self) -> None:
        from lightguard.malware.train import load_model
        from lightguard.monitor.scan import scan, Verdict
        model = load_model(MODEL_PATH)
        v = scan(BENIGN_EXE, model, threshold=0.80)
        assert isinstance(v, Verdict)

    @_SKIP_NO_EXE
    @_SKIP_NO_MODEL
    def test_verdict_fields(self) -> None:
        from lightguard.malware.train import load_model
        from lightguard.monitor.scan import scan
        model = load_model(MODEL_PATH)
        v = scan(BENIGN_EXE, model, threshold=0.80)
        assert v.filename == BENIGN_EXE.name
        assert 0 <= v.risk_score <= 100
        assert v.label in ("MALICIOUS", "BENIGN")
        assert v.confidence in ("HIGH", "MEDIUM", "LOW")
        assert 0.0 <= v.raw_prob <= 1.0

    @_SKIP_NO_EXE
    @_SKIP_NO_MODEL
    def test_benign_file_scores_low(self) -> None:
        from lightguard.malware.train import load_model
        from lightguard.monitor.scan import scan
        model = load_model(MODEL_PATH)
        v = scan(BENIGN_EXE, model, threshold=0.80)
        # A known-benign file must score below the alert threshold
        assert v.label == "BENIGN", (
            f"Known-benign file {BENIGN_EXE.name} scored {v.risk_score}/100 "
            f"(raw={v.raw_prob:.4f}) — model may need retraining or threshold adjustment"
        )

    @_SKIP_NO_EXE
    @_SKIP_NO_MODEL
    def test_verdict_str(self) -> None:
        from lightguard.malware.train import load_model
        from lightguard.monitor.scan import scan
        model = load_model(MODEL_PATH)
        v = scan(BENIGN_EXE, model, threshold=0.80)
        s = str(v)
        assert v.filename in s
        assert v.label in s

    @_SKIP_NO_EXE
    @_SKIP_NO_MODEL
    def test_threshold_zero_forces_malicious(self) -> None:
        from lightguard.malware.train import load_model
        from lightguard.monitor.scan import scan
        model = load_model(MODEL_PATH)
        v = scan(BENIGN_EXE, model, threshold=0.0)
        assert v.label == "MALICIOUS"

    @_SKIP_NO_EXE
    @_SKIP_NO_MODEL
    def test_threshold_one_forces_benign(self) -> None:
        from lightguard.malware.train import load_model
        from lightguard.monitor.scan import scan
        model = load_model(MODEL_PATH)
        v = scan(BENIGN_EXE, model, threshold=1.0)
        assert v.label == "BENIGN"


# ── watcher ───────────────────────────────────────────────────────────────────

class TestWatcher:
    @_SKIP_NO_EXE
    @_SKIP_NO_MODEL
    def test_scan_now(self) -> None:
        from lightguard.malware.train import load_model
        from lightguard.monitor.watch import Watcher
        model = load_model(MODEL_PATH)
        watcher = Watcher(BENIGN_EXE.parent, model, lambda v: None)
        v = watcher.scan_now(BENIGN_EXE)
        assert v.filename == BENIGN_EXE.name

    @_SKIP_NO_EXE
    @_SKIP_NO_MODEL
    def test_observer_detects_new_file(self, tmp_path: Path) -> None:
        """Drop a copy of the benign exe into a temp dir and confirm verdict arrives."""
        from lightguard.malware.train import load_model
        from lightguard.monitor.watch import Watcher

        model   = load_model(MODEL_PATH)
        results: queue.Queue = queue.Queue()
        watcher = Watcher(tmp_path, model, results.put)
        watcher.start()

        # Give the observer a moment to initialise before writing the file
        time.sleep(0.3)
        import shutil
        shutil.copy(BENIGN_EXE, tmp_path / "benign_copy.exe")

        try:
            verdict = results.get(timeout=10)
        except queue.Empty:
            pytest.fail("Watcher did not emit a verdict within 10 s")
        finally:
            watcher.stop()

        assert verdict.filename == "benign_copy.exe"
        assert verdict.label in ("MALICIOUS", "BENIGN")
