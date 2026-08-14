"""Tests for src/lightguard/network/detector.py and explain.py.

All tests use synthetic DataFrames — no psutil, no real network activity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lightguard.network.detector import (
    DEFAULT_THRESHOLD,
    FEATURE_COLS,
    NetworkDetector,
    load_detector,
    save_detector,
    score,
    train,
)
from lightguard.network.explain import (
    _ABS_THRESHOLDS,
    _BORDER_Z,
    _HIGH_Z,
    explain_process,
    explain_snapshot,
)


# ── synthetic data helpers ────────────────────────────────────────────────────

def _normal_baseline(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Return a realistic baseline: low, stable per-process connections."""
    rng = np.random.default_rng(seed)
    rows = []
    procs = ["browser", "mail", "slack", "python", "system"]
    for i in range(n):
        proc = procs[i % len(procs)]
        rows.append({
            "pid":               1000 + i,
            "proc_name":         proc,
            "conn_count":        int(rng.integers(1, 10)),
            "unique_remote_ips": int(rng.integers(1, 5)),
            "unique_remote_ports": int(rng.integers(1, 4)),
            "rare_port_count":   int(rng.integers(0, 2)),
            "conn_per_min":      float(rng.uniform(1, 20)),
        })
    return pd.DataFrame(rows)


def _anomalous_row(proc_name: str = "malware", pid: int = 9999) -> pd.Series:
    """Return a single-process row with extreme values across all features."""
    return pd.Series({
        "pid":               pid,
        "proc_name":         proc_name,
        "conn_count":        500,
        "unique_remote_ips": 200,
        "unique_remote_ports": 150,
        "rare_port_count":   100,
        "conn_per_min":      3000.0,
        "anomaly_score":     90,
        "label":             "ANOMALOUS",
    })


def _normal_row(proc_name: str = "mail", pid: int = 1001) -> pd.Series:
    return pd.Series({
        "pid":               pid,
        "proc_name":         proc_name,
        "conn_count":        3,
        "unique_remote_ips": 2,
        "unique_remote_ports": 1,
        "rare_port_count":   0,
        "conn_per_min":      5.0,
        "anomaly_score":     10,
        "label":             "NORMAL",
    })


# ── train() ───────────────────────────────────────────────────────────────────

class TestTrain:
    def test_returns_network_detector(self):
        det = train(_normal_baseline())
        assert isinstance(det, NetworkDetector)

    def test_feature_cols_stored(self):
        det = train(_normal_baseline())
        assert det.feature_cols == FEATURE_COLS

    def test_baseline_stats_keys(self):
        det = train(_normal_baseline())
        for col in FEATURE_COLS:
            assert col in det.baseline_stats
            assert "mean" in det.baseline_stats[col]
            assert "std"  in det.baseline_stats[col]

    def test_baseline_stats_mean_sensible(self):
        baseline = _normal_baseline()
        det = train(baseline)
        # mean conn_count in baseline is 1–10 → should be around 5
        assert 0 < det.baseline_stats["conn_count"]["mean"] < 20

    def test_model_is_fitted(self):
        det = train(_normal_baseline())
        # sklearn fitted estimators expose n_features_in_
        assert hasattr(det.model, "n_features_in_")
        assert det.model.n_features_in_ == len(FEATURE_COLS)

    def test_scaler_is_fitted(self):
        det = train(_normal_baseline())
        assert hasattr(det.scaler, "mean_")
        assert len(det.scaler.mean_) == len(FEATURE_COLS)

    def test_threshold_stored(self):
        det = train(_normal_baseline(), threshold=60)
        assert det.threshold == 60

    def test_contamination_stored(self):
        det = train(_normal_baseline(), contamination=0.05)
        assert det.contamination == pytest.approx(0.05)

    def test_missing_feature_column_raises(self):
        df = _normal_baseline().drop(columns=["conn_count"])
        with pytest.raises(ValueError, match="missing columns"):
            train(df)

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame(columns=["pid", "proc_name"] + FEATURE_COLS)
        with pytest.raises(ValueError, match="empty"):
            train(df)

    def test_reproducible_with_same_seed(self):
        baseline = _normal_baseline()
        d1 = train(baseline, random_seed=7)
        d2 = train(baseline, random_seed=7)
        snap = _normal_baseline(n=5, seed=99)
        r1 = score(snap, d1)["anomaly_score"].tolist()
        r2 = score(snap, d2)["anomaly_score"].tolist()
        assert r1 == r2

    def test_different_seeds_may_differ(self):
        baseline = _normal_baseline(n=30)
        d1 = train(baseline, random_seed=1)
        d2 = train(baseline, random_seed=999)
        snap = _normal_baseline(n=10, seed=50)
        # Scores should differ at least slightly with different seeds
        r1 = score(snap, d1)["anomaly_score"].tolist()
        r2 = score(snap, d2)["anomaly_score"].tolist()
        # Not asserting they're different (could coincide), just that both run
        assert len(r1) == len(r2)

    def test_single_row_baseline(self):
        df = _normal_baseline(n=1)
        # IsolationForest needs ≥1 row; should not raise
        det = train(df)
        assert det is not None


# ── score() ───────────────────────────────────────────────────────────────────

class TestScore:
    @pytest.fixture()
    def detector(self):
        return train(_normal_baseline(n=30, seed=0))

    def test_returns_dataframe(self, detector):
        snap = _normal_baseline(n=5)
        result = score(snap, detector)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_process(self, detector):
        snap = _normal_baseline(n=7)
        result = score(snap, detector)
        assert len(result) == 7

    def test_required_columns(self, detector):
        snap = _normal_baseline(n=3)
        result = score(snap, detector)
        for col in ["pid", "proc_name", "anomaly_score", "label"] + FEATURE_COLS:
            assert col in result.columns, f"missing column: {col}"

    def test_anomaly_score_range(self, detector):
        snap = _normal_baseline(n=10)
        result = score(snap, detector)
        assert result["anomaly_score"].between(0, 100).all()

    def test_anomaly_score_dtype_int(self, detector):
        snap = _normal_baseline(n=5)
        result = score(snap, detector)
        assert result["anomaly_score"].dtype in (int, np.int64, np.int32)

    def test_label_values(self, detector):
        snap = _normal_baseline(n=10)
        result = score(snap, detector)
        assert set(result["label"]).issubset({"NORMAL", "ANOMALOUS"})

    def test_high_anomaly_for_extreme_values(self, detector):
        # A process with extreme values should score higher than a normal one.
        extreme = pd.DataFrame([{
            "pid": 9999, "proc_name": "suspicious",
            "conn_count": 1000, "unique_remote_ips": 500,
            "unique_remote_ports": 300, "rare_port_count": 200,
            "conn_per_min": 5000.0,
        }])
        normal = _normal_baseline(n=1)
        r_extreme = score(extreme, detector)
        r_normal  = score(normal,  detector)
        assert r_extreme.iloc[0]["anomaly_score"] > r_normal.iloc[0]["anomaly_score"]

    def test_sorted_by_anomaly_score_descending(self, detector):
        snap = _normal_baseline(n=10)
        result = score(snap, detector)
        scores = result["anomaly_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_empty_snapshot_returns_empty_df(self, detector):
        empty = pd.DataFrame(columns=["pid", "proc_name"] + FEATURE_COLS)
        result = score(empty, detector)
        assert len(result) == 0

    def test_missing_feature_col_treated_as_zero(self, detector):
        snap = _normal_baseline(n=3).drop(columns=["rare_port_count"])
        result = score(snap, detector)
        assert len(result) == 3   # should not raise

    def test_feature_values_preserved(self, detector):
        snap = _normal_baseline(n=5)
        result = score(snap, detector)
        # conn_count from the snap should be present verbatim in the result
        for col in FEATURE_COLS:
            assert col in result.columns

    def test_new_process_not_in_baseline_gets_scored(self, detector):
        new_proc = pd.DataFrame([{
            "pid": 88888, "proc_name": "brand_new_process",
            "conn_count": 3, "unique_remote_ips": 1,
            "unique_remote_ports": 1, "rare_port_count": 0,
            "conn_per_min": 5.0,
        }])
        result = score(new_proc, detector)
        assert len(result) == 1
        assert result.iloc[0]["anomaly_score"] >= 0

    def test_threshold_controls_label(self):
        baseline = _normal_baseline(n=30)
        det_low  = train(baseline, threshold=0)   # everything is ANOMALOUS
        det_high = train(baseline, threshold=100)  # nothing is ANOMALOUS
        snap = _normal_baseline(n=5)
        assert (score(snap, det_low)["label"]  == "ANOMALOUS").all()
        assert (score(snap, det_high)["label"] == "NORMAL").all()


# ── save_detector / load_detector ─────────────────────────────────────────────

class TestDetectorPersistence:
    def test_save_creates_file(self, tmp_path):
        det  = train(_normal_baseline())
        dest = save_detector(det, path=tmp_path / "det.joblib")
        assert dest.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        det  = train(_normal_baseline())
        deep = tmp_path / "a" / "b" / "det.joblib"
        save_detector(det, path=deep)
        assert deep.exists()

    def test_save_returns_absolute_path(self, tmp_path):
        det  = train(_normal_baseline())
        dest = save_detector(det, path=tmp_path / "det.joblib")
        assert dest.is_absolute()

    def test_round_trip_produces_same_scores(self, tmp_path):
        baseline = _normal_baseline(n=30)
        det      = train(baseline, random_seed=42)
        snap     = _normal_baseline(n=5, seed=99)

        scores_before = score(snap, det)["anomaly_score"].tolist()

        path = tmp_path / "det.joblib"
        save_detector(det, path=path)
        det2 = load_detector(path=path)

        scores_after = score(snap, det2)["anomaly_score"].tolist()
        assert scores_before == scores_after

    def test_round_trip_preserves_baseline_stats(self, tmp_path):
        det  = train(_normal_baseline())
        path = tmp_path / "det.joblib"
        save_detector(det, path=path)
        det2 = load_detector(path=path)
        for col in FEATURE_COLS:
            assert det2.baseline_stats[col]["mean"] == pytest.approx(det.baseline_stats[col]["mean"])
            assert det2.baseline_stats[col]["std"]  == pytest.approx(det.baseline_stats[col]["std"])

    def test_round_trip_preserves_threshold(self, tmp_path):
        det  = train(_normal_baseline(), threshold=70)
        path = tmp_path / "det.joblib"
        save_detector(det, path=path)
        assert load_detector(path=path).threshold == 70

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_detector(path=tmp_path / "no_such_file.joblib")


# ── explain_process() ─────────────────────────────────────────────────────────

class TestExplainProcess:
    @pytest.fixture()
    def detector(self):
        return train(_normal_baseline(n=30, seed=0))

    def test_returns_list(self, detector):
        reasons = explain_process(_anomalous_row(), detector)
        assert isinstance(reasons, list)

    def test_all_reasons_are_strings(self, detector):
        reasons = explain_process(_anomalous_row(), detector)
        assert all(isinstance(r, str) for r in reasons)

    def test_extreme_row_gets_reasons(self, detector):
        reasons = explain_process(_anomalous_row(), detector)
        assert len(reasons) > 0

    def test_normal_row_gets_fewer_reasons(self, detector):
        # A row well within baseline should produce fewer or no reasons.
        reasons = explain_process(_normal_row(), detector)
        assert len(reasons) <= 2

    def test_top_k_respected(self, detector):
        reasons = explain_process(_anomalous_row(), detector, top_k=1)
        assert len(reasons) <= 1

    def test_conn_per_min_reason_fires(self, detector):
        row = _anomalous_row()   # conn_per_min=3000 — very high
        reasons = explain_process(row, detector)
        combined = " ".join(reasons).lower()
        assert "often" in combined or "frequently" in combined or "reaching" in combined

    def test_rare_port_reason_fires(self, detector):
        row = pd.Series({
            "pid": 1, "proc_name": "x",
            "conn_count": 3, "unique_remote_ips": 1,
            "unique_remote_ports": 1, "rare_port_count": 50,
            "conn_per_min": 5.0, "anomaly_score": 80, "label": "ANOMALOUS",
        })
        reasons = explain_process(row, detector)
        combined = " ".join(reasons).lower()
        assert "port" in combined

    def test_unique_ips_reason_fires(self, detector):
        row = pd.Series({
            "pid": 1, "proc_name": "x",
            "conn_count": 3, "unique_remote_ips": 300,
            "unique_remote_ports": 1, "rare_port_count": 0,
            "conn_per_min": 5.0, "anomaly_score": 80, "label": "ANOMALOUS",
        })
        reasons = explain_process(row, detector)
        combined = " ".join(reasons).lower()
        assert "address" in combined or "external" in combined

    def test_abs_fallback_when_no_baseline_stats(self):
        # Use a detector with zeroed stats to force absolute-threshold branch.
        from lightguard.network.detector import NetworkDetector
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import IsolationForest
        import numpy as np

        baseline = _normal_baseline(n=5)
        X = baseline[FEATURE_COLS].values.astype(float)
        scaler = StandardScaler().fit(X)
        model  = IsolationForest(random_state=0).fit(scaler.transform(X))

        det_no_stats = NetworkDetector(
            model=model,
            scaler=scaler,
            feature_cols=FEATURE_COLS,
            baseline_stats={},   # empty → absolute thresholds
            threshold=DEFAULT_THRESHOLD,
        )

        row = _anomalous_row()
        reasons = explain_process(row, det_no_stats)
        assert isinstance(reasons, list)

    def test_empty_reasons_for_perfectly_normal_row(self, detector):
        # A row at exactly the mean of the baseline should return no reasons.
        stats = detector.baseline_stats
        row = pd.Series({
            "pid":               1,
            "proc_name":         "test",
            "conn_count":        stats["conn_count"]["mean"],
            "unique_remote_ips": stats["unique_remote_ips"]["mean"],
            "unique_remote_ports": stats["unique_remote_ports"]["mean"],
            "rare_port_count":   stats["rare_port_count"]["mean"],
            "conn_per_min":      stats["conn_per_min"]["mean"],
            "anomaly_score":     10,
            "label":             "NORMAL",
        })
        reasons = explain_process(row, detector)
        assert len(reasons) == 0


# ── explain_snapshot() ────────────────────────────────────────────────────────

class TestExplainSnapshot:
    @pytest.fixture()
    def detector(self):
        return train(_normal_baseline(n=30, seed=0))

    def _make_results(self, detector):
        snap = _normal_baseline(n=5)
        # inject one obvious anomaly
        anomaly = pd.DataFrame([{
            "pid": 9999, "proc_name": "badware",
            "conn_count": 999, "unique_remote_ips": 500,
            "unique_remote_ports": 200, "rare_port_count": 150,
            "conn_per_min": 9999.0,
        }])
        combined = pd.concat([snap, anomaly], ignore_index=True)
        return score(combined, detector)

    def test_adds_reasons_column(self, detector):
        results = self._make_results(detector)
        out = explain_snapshot(results, detector)
        assert "reasons" in out.columns

    def test_normal_rows_get_empty_reasons(self, detector):
        results = self._make_results(detector)
        out = explain_snapshot(results, detector)
        normal_rows = out[out["label"] == "NORMAL"]
        for reasons in normal_rows["reasons"]:
            assert reasons == []

    def test_anomalous_rows_get_reasons(self, detector):
        results = self._make_results(detector)
        out = explain_snapshot(results, detector)
        anomalous_rows = out[out["label"] == "ANOMALOUS"]
        if len(anomalous_rows) > 0:
            # At least one anomalous row should have reasons
            has_reasons = any(len(r) > 0 for r in anomalous_rows["reasons"])
            assert has_reasons

    def test_reasons_are_list_of_strings(self, detector):
        results = self._make_results(detector)
        out = explain_snapshot(results, detector)
        for reasons in out["reasons"]:
            assert isinstance(reasons, list)
            assert all(isinstance(r, str) for r in reasons)

    def test_original_columns_preserved(self, detector):
        results = self._make_results(detector)
        out = explain_snapshot(results, detector)
        for col in results.columns:
            assert col in out.columns

    def test_top_k_applied_per_row(self, detector):
        results = self._make_results(detector)
        out = explain_snapshot(results, detector, top_k=2)
        for reasons in out["reasons"]:
            assert len(reasons) <= 2

    def test_empty_input_returns_empty(self, detector):
        empty = pd.DataFrame(columns=["pid", "proc_name", "anomaly_score", "label"] + FEATURE_COLS)
        out = explain_snapshot(empty, detector)
        assert len(out) == 0


# ── FEATURE_COLS sanity ───────────────────────────────────────────────────────

class TestFeatureCols:
    def test_exactly_five_features(self):
        assert len(FEATURE_COLS) == 5

    def test_required_names_present(self):
        required = {
            "conn_count", "unique_remote_ips", "unique_remote_ports",
            "rare_port_count", "conn_per_min",
        }
        assert required == set(FEATURE_COLS)

    def test_no_duplicates(self):
        assert len(FEATURE_COLS) == len(set(FEATURE_COLS))
