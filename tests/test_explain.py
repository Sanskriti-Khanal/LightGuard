"""Smoke tests for src/lightguard/explain/explainer.py and translate.py.

All tests run on data/sample/ fixtures — no EMBER2024 data, no network.
A tiny LightGBM model is trained on the sample train split for the SHAP tests.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from conftest import load_sample_split
from lightguard.explain.translate import (
    FEATURE_NAMES,
    feature_description,
    translate,
)


# ── FEATURE_NAMES ──────────────────────────────────────────────────────────────

class TestFeatureNames:
    def test_length(self) -> None:
        assert len(FEATURE_NAMES) == 2568

    def test_no_duplicates(self) -> None:
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

    def test_first_is_file_size(self) -> None:
        assert FEATURE_NAMES[0] == "general_file_size"

    def test_strings_group_start(self) -> None:
        assert FEATURE_NAMES[519] == "strings_count"

    def test_header_group_start(self) -> None:
        assert FEATURE_NAMES[696] == "header_coff_timestamp"

    def test_imports_group_start(self) -> None:
        assert FEATURE_NAMES[994] == "imports_num_dlls"

    def test_authenticode_group_start(self) -> None:
        assert FEATURE_NAMES[2472] == "authenticode_num_certs"

    def test_all_strings(self) -> None:
        assert all(isinstance(n, str) and len(n) > 0 for n in FEATURE_NAMES)


# ── feature_description ────────────────────────────────────────────────────────

class TestFeatureDescription:
    def test_file_size_readable(self) -> None:
        desc = feature_description("general_file_size", 12345.0)
        # 12345 bytes → shown as "12 KB" in plain-English mode
        assert "file size" in desc.lower() or "kb" in desc.lower() or "byte" in desc.lower()

    def test_string_count(self) -> None:
        desc = feature_description("strings_count", 300.0)
        assert "300" in desc

    def test_histogram_fallback(self) -> None:
        desc = feature_description("histogram_byte_42", 0.01)
        assert "byte" in desc.lower() or "unusual" in desc.lower()

    def test_imports_func_hash_fallback(self) -> None:
        desc = feature_description("imports_func_hash_7", 1.5)
        assert "import" in desc.lower() or "unusual" in desc.lower()

    def test_string_regex_feature(self) -> None:
        # powershell regex feature
        desc = feature_description("strings_regex_powershell", 3.0)
        assert "powershell" in desc.lower()
        assert "3" in desc

    def test_unknown_feature_returns_string(self) -> None:
        desc = feature_description("unknown_xyz_999", 7.0)
        assert isinstance(desc, str)
        assert len(desc) > 0


# ── translate ─────────────────────────────────────────────────────────────────

class TestTranslate:
    def _make_top_features(self, n: int = 5) -> list[dict]:
        rng = np.random.default_rng(0)
        return [
            {
                "feature_idx": i * 50,
                "feature_name": FEATURE_NAMES[i * 50],
                "shap_value": float(rng.uniform(-1, 1)),
                "raw_value": float(rng.uniform(0, 10)),
            }
            for i in range(n)
        ]

    def test_returns_one_sentence_per_feature(self) -> None:
        top = self._make_top_features(5)
        sentences = translate(top)
        assert len(sentences) == 5

    def test_sentences_are_strings(self) -> None:
        for s in translate(self._make_top_features(3)):
            assert isinstance(s, str) and len(s) > 0

    def test_positive_shap_labelled_high_risk(self) -> None:
        top = [{"feature_idx": 0, "feature_name": "general_file_size",
                "shap_value": 0.5, "raw_value": 1000.0}]
        assert "High-risk" in translate(top)[0]

    def test_negative_shap_labelled_low_risk(self) -> None:
        top = [{"feature_idx": 0, "feature_name": "general_file_size",
                "shap_value": -0.5, "raw_value": 1000.0}]
        assert "Low-risk" in translate(top)[0]

    def test_shap_value_appears_in_verbose_sentence(self) -> None:
        top = [{"feature_idx": 0, "feature_name": "general_file_size",
                "shap_value": 0.123, "raw_value": 100.0}]
        sentence = translate(top, verbose=True)[0]
        assert "0.123" in sentence

    def test_shap_value_absent_in_clean_sentence(self) -> None:
        top = [{"feature_idx": 0, "feature_name": "general_file_size",
                "shap_value": 0.123, "raw_value": 100.0}]
        sentence = translate(top, verbose=False)[0]
        assert "0.123" not in sentence
        assert "SHAP" not in sentence

    def test_empty_input(self) -> None:
        assert translate([]) == []


# ── explainer integration ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_model(sample_dir: Path, config: dict, tmp_path_factory):
    """Train a very small LightGBM on sample data for SHAP tests."""
    lgb = pytest.importorskip("lightgbm")
    import copy
    from lightguard.malware.train import train

    cfg = copy.deepcopy(config)
    cfg["lgbm"]["n_estimators"] = 20
    cfg["lgbm"]["early_stopping_rounds"] = 5

    X, y = load_sample_split(sample_dir, "train")
    y = y.astype(np.float32)
    model_path = tmp_path_factory.mktemp("models") / "tiny.txt"
    return train(X, y, cfg, model_path)


@pytest.fixture(scope="module")
def background(sample_dir: Path):
    X, _ = load_sample_split(sample_dir, "test")
    return X[:50]  # small background for speed


class TestBuildExplainer:
    def test_returns_tree_explainer(self, tiny_model, background) -> None:
        shap = pytest.importorskip("shap")
        from lightguard.explain.explainer import build_explainer
        exp = build_explainer(tiny_model, background)
        assert isinstance(exp, shap.TreeExplainer)


class TestExplainPrediction:
    def test_returns_list(self, tiny_model, background, sample_dir: Path) -> None:
        from lightguard.explain.explainer import build_explainer, explain_prediction
        exp = build_explainer(tiny_model, background)
        X, _ = load_sample_split(sample_dir, "test")
        result = explain_prediction(exp, X[0])
        assert isinstance(result, list)

    def test_default_top_k(self, tiny_model, background, sample_dir: Path) -> None:
        from lightguard.explain.explainer import build_explainer, explain_prediction
        exp = build_explainer(tiny_model, background)
        X, _ = load_sample_split(sample_dir, "test")
        result = explain_prediction(exp, X[0], top_k=10)
        assert len(result) == 10

    def test_custom_top_k(self, tiny_model, background, sample_dir: Path) -> None:
        from lightguard.explain.explainer import build_explainer, explain_prediction
        exp = build_explainer(tiny_model, background)
        X, _ = load_sample_split(sample_dir, "test")
        result = explain_prediction(exp, X[0], top_k=3)
        assert len(result) == 3

    def test_result_keys(self, tiny_model, background, sample_dir: Path) -> None:
        from lightguard.explain.explainer import build_explainer, explain_prediction
        exp = build_explainer(tiny_model, background)
        X, _ = load_sample_split(sample_dir, "test")
        entry = explain_prediction(exp, X[0], top_k=1)[0]
        assert set(entry.keys()) == {"feature_idx", "feature_name", "shap_value", "raw_value"}

    def test_sorted_by_abs_shap(self, tiny_model, background, sample_dir: Path) -> None:
        from lightguard.explain.explainer import build_explainer, explain_prediction
        exp = build_explainer(tiny_model, background)
        X, _ = load_sample_split(sample_dir, "test")
        result = explain_prediction(exp, X[0], top_k=5)
        shap_abs = [abs(e["shap_value"]) for e in result]
        assert shap_abs == sorted(shap_abs, reverse=True)

    def test_raw_value_matches_input(self, tiny_model, background, sample_dir: Path) -> None:
        from lightguard.explain.explainer import build_explainer, explain_prediction
        exp = build_explainer(tiny_model, background)
        X, _ = load_sample_split(sample_dir, "test")
        x = X[0]
        result = explain_prediction(exp, x, top_k=5)
        for entry in result:
            assert abs(entry["raw_value"] - float(x[entry["feature_idx"]])) < 1e-5

    def test_translate_compatible(self, tiny_model, background, sample_dir: Path) -> None:
        from lightguard.explain.explainer import build_explainer, explain_prediction
        exp = build_explainer(tiny_model, background)
        X, _ = load_sample_split(sample_dir, "test")
        result = explain_prediction(exp, X[0], top_k=5)
        sentences = translate(result)
        assert len(sentences) == 5
        assert all(isinstance(s, str) for s in sentences)
