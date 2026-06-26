"""Tests for src/lightguard/ui/ — Flask route handlers and helpers.

These tests use Flask's built-in test client and inject a pre-seeded
AppState so they never touch the model, disk, or network.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from lightguard.monitor.scan import Verdict
from lightguard.ui.app import _parse_reasons, create_app
from lightguard.ui.state import AppState, ScanEntry


# ── shared fixtures ────────────────────────────────────────────────────────────

def _make_verdict(label: str = "BENIGN", risk: int = 3) -> Verdict:
    return Verdict(
        filename="putty.exe",
        risk_score=risk,
        label=label,
        confidence="HIGH",
        raw_prob=risk / 100,
        reasons=(
            "Low-risk indicator (SHAP -2.500): File size is 1683456 bytes",
            "Low-risk indicator (SHAP -1.800): Target machine type code is 34404",
            "High-risk indicator (SHAP +0.300): byte frequency histogram bucket (value=0.0042)",
        ),
    )


def _make_entry(id: str = "test-abc-123", label: str = "BENIGN") -> ScanEntry:
    return ScanEntry(
        id=id,
        verdict=_make_verdict(label=label, risk=3 if label == "BENIGN" else 96),
        scanned_at=datetime(2026, 6, 26, 12, 0, 0),
        source="manual",
    )


@pytest.fixture()
def app():
    """Flask app with no model (read-only) and one pre-seeded scan entry."""
    flask_app = create_app()
    flask_app.config["TESTING"] = True
    flask_app.state.add_scan(_make_entry("test-abc-123", "BENIGN"))
    flask_app.state.add_scan(_make_entry("test-def-456", "MALICIOUS"))
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


# ── dashboard ──────────────────────────────────────────────────────────────────

class TestDashboard:
    def test_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_shows_scan_count(self, client):
        r = client.get("/")
        assert b"2" in r.data          # 2 scans seeded

    def test_shows_filename(self, client):
        r = client.get("/")
        assert b"putty.exe" in r.data

    def test_shows_malicious_badge(self, client):
        r = client.get("/")
        assert b"MALICIOUS" in r.data

    def test_shows_threat_count_when_nonzero(self, client):
        r = client.get("/")
        # 1 malicious entry → "1 threat detected" heading
        assert b"1 threat" in r.data


# ── scan page ──────────────────────────────────────────────────────────────────

class TestScanPage:
    def test_get_returns_200(self, client):
        assert client.get("/scan").status_code == 200

    def test_shows_no_model_notice(self, client):
        r = client.get("/scan")
        # model_loaded=False → notice about --model flag
        assert b"No model loaded" in r.data

    def test_post_no_model_returns_error(self, client):
        # When no model is loaded, any POST hits the model-missing branch first
        r = client.post("/scan", data={}, content_type="multipart/form-data")
        assert r.status_code == 200
        assert b"No model" in r.data

    def test_post_no_file_with_model_returns_error(self, app):
        # Inject a mock model so the file-missing branch is reachable
        from unittest.mock import MagicMock
        app._watcher = None
        # Patch via a fresh app that has a model set
        app2 = create_app(model=MagicMock())
        app2.config["TESTING"] = True
        with app2.test_client() as c:
            r = c.post("/scan", data={}, content_type="multipart/form-data")
            assert r.status_code == 200
            assert b"No file selected" in r.data


# ── result page ────────────────────────────────────────────────────────────────

class TestResultPage:
    def test_existing_id_returns_200(self, client):
        assert client.get("/result/test-abc-123").status_code == 200

    def test_missing_id_returns_404(self, client):
        assert client.get("/result/no-such-id").status_code == 404

    def test_shows_filename(self, client):
        r = client.get("/result/test-abc-123")
        assert b"putty.exe" in r.data

    def test_shows_risk_score(self, client):
        r = client.get("/result/test-abc-123")
        assert b"3/100" in r.data

    def test_shows_benign_label(self, client):
        r = client.get("/result/test-abc-123")
        assert b"BENIGN" in r.data

    def test_shows_explanation_rows(self, client):
        r = client.get("/result/test-abc-123")
        # reasons are parsed and rendered as .row divs
        assert b"Low-risk" in r.data or b"File size" in r.data

    def test_malicious_result_shows_threat_label(self, client):
        r = client.get("/result/test-def-456")
        assert b"MALICIOUS" in r.data


# ── feed page ──────────────────────────────────────────────────────────────────

class TestFeedPage:
    def test_returns_200(self, client):
        assert client.get("/feed").status_code == 200

    def test_shows_recent_entries(self, client):
        r = client.get("/feed")
        assert b"putty.exe" in r.data

    def test_no_watcher_shows_notice(self, client):
        r = client.get("/feed")
        assert b"No watcher" in r.data or b"No model" in r.data


# ── /api/recent ────────────────────────────────────────────────────────────────

class TestApiRecent:
    def test_returns_200(self, client):
        assert client.get("/api/recent").status_code == 200

    def test_content_type_is_json(self, client):
        r = client.get("/api/recent")
        assert "application/json" in r.content_type

    def test_returns_list(self, client):
        r = client.get("/api/recent")
        data = r.get_json()
        assert isinstance(data, list)

    def test_entries_have_required_keys(self, client):
        r = client.get("/api/recent")
        entry = r.get_json()[0]
        for key in ("id", "filename", "label", "risk_score", "confidence", "scanned_at"):
            assert key in entry, f"Missing key: {key}"

    def test_newest_first(self, client):
        r = client.get("/api/recent")
        data = r.get_json()
        # "test-def-456" was added after "test-abc-123" so it's first
        assert data[0]["id"] == "test-def-456"

    def test_filename_matches(self, client):
        r = client.get("/api/recent")
        assert r.get_json()[0]["filename"] == "putty.exe"


# ── /api/scan-vector (demo-only route) ────────────────────────────────────────

class TestScanVectorRoute:
    """Tests for the demo-only POST /api/scan-vector route."""

    # fixture: app with a real model so the route is enabled
    @pytest.fixture()
    def model_app(self):
        from unittest.mock import MagicMock
        import numpy as np

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.97])
        mock_model.best_iteration = -1

        flask_app = create_app(model=mock_model)
        flask_app.config["TESTING"] = True
        return flask_app

    @pytest.fixture()
    def model_client(self, model_app):
        return model_app.test_client()

    def test_no_model_returns_503(self, client):
        r = client.post("/api/scan-vector",
                        json={"row": 0},
                        content_type="application/json")
        assert r.status_code == 503

    def test_missing_row_returns_400(self, model_client):
        r = model_client.post("/api/scan-vector",
                              json={},
                              content_type="application/json")
        assert r.status_code == 400
        assert "row" in r.get_json()["error"]

    def test_out_of_range_row_returns_400(self, model_client):
        r = model_client.post("/api/scan-vector",
                              json={"row": 99999},
                              content_type="application/json")
        assert r.status_code == 400

    def test_valid_row_returns_201(self, model_client):
        r = model_client.post("/api/scan-vector",
                              json={"row": 0},
                              content_type="application/json")
        assert r.status_code == 201

    def test_response_has_required_fields(self, model_client):
        r = model_client.post("/api/scan-vector",
                              json={"row": 0},
                              content_type="application/json")
        data = r.get_json()
        for key in ("id", "row", "filename", "label", "risk_score",
                    "confidence", "raw_prob", "result_url", "_demo_only"):
            assert key in data, f"Missing key: {key}"

    def test_demo_only_flag_is_true(self, model_client):
        r = model_client.post("/api/scan-vector", json={"row": 0})
        assert r.get_json()["_demo_only"] is True

    def test_row_echoed_in_response(self, model_client):
        r = model_client.post("/api/scan-vector", json={"row": 5})
        assert r.get_json()["row"] == 5

    def test_custom_label_used_as_filename(self, model_client):
        r = model_client.post("/api/scan-vector",
                              json={"row": 0, "label": "demo_malware"})
        assert r.get_json()["filename"] == "demo_malware"

    def test_default_filename_contains_row(self, model_client):
        r = model_client.post("/api/scan-vector", json={"row": 7})
        assert "007" in r.get_json()["filename"]

    def test_entry_appears_in_recent(self, model_client):
        model_client.post("/api/scan-vector", json={"row": 0})
        recent = model_client.get("/api/recent").get_json()
        assert any(e["source"] == "demo-vector" for e in recent)

    def test_result_url_points_to_valid_entry(self, model_client):
        r = model_client.post("/api/scan-vector", json={"row": 0})
        result_url = r.get_json()["result_url"]
        r2 = model_client.get(result_url)
        assert r2.status_code == 200

    def test_entry_broadcast_to_sse(self, model_app):
        # Subscribe to SSE queue, inject a vector, confirm broadcast fires
        q = model_app.state.subscribe()
        with model_app.test_client() as c:
            c.post("/api/scan-vector", json={"row": 0})
        received = q.get_nowait()
        assert received.source == "demo-vector"
        model_app.state.unsubscribe(q)

    def test_negative_row_returns_400(self, model_client):
        r = model_client.post("/api/scan-vector", json={"row": -1})
        assert r.status_code == 400

    def test_malicious_label_for_high_prob(self, model_app):
        # mock returns 0.97 → MALICIOUS
        with model_app.test_client() as c:
            r = c.post("/api/scan-vector", json={"row": 0})
        assert r.get_json()["label"] == "MALICIOUS"

    def test_benign_label_for_low_prob(self):
        from unittest.mock import MagicMock
        import numpy as np
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.05])
        mock_model.best_iteration = -1
        flask_app = create_app(model=mock_model)
        flask_app.config["TESTING"] = True
        with flask_app.test_client() as c:
            r = c.post("/api/scan-vector", json={"row": 0})
        assert r.get_json()["label"] == "BENIGN"


# ── /api/events (SSE route registration) ──────────────────────────────────────

class TestApiEventsRoute:
    def test_route_is_registered(self, app):
        rules = [str(rule) for rule in app.url_map.iter_rules()]
        assert "/api/events" in rules

    def test_route_accepts_get(self, app):
        for rule in app.url_map.iter_rules():
            if str(rule) == "/api/events":
                assert "GET" in rule.methods
                return
        pytest.fail("/api/events rule not found")


# ── _parse_reasons helper ──────────────────────────────────────────────────────

class TestParseReasons:
    def test_empty_input(self):
        assert _parse_reasons(()) == []

    def test_high_risk_direction(self):
        reasons = ("High-risk indicator (SHAP +2.500): File size is 1234 bytes",)
        p = _parse_reasons(reasons)
        assert p[0]["direction"] == "High-risk"
        assert p[0]["severity"] == "bad"

    def test_low_risk_direction(self):
        reasons = ("Low-risk indicator (SHAP -1.200): Contains 100 printable strings",)
        p = _parse_reasons(reasons)
        assert p[0]["direction"] == "Low-risk"
        assert p[0]["severity"] == "ok"

    def test_shap_value_parsed(self):
        reasons = ("High-risk indicator (SHAP +2.500): Some desc",)
        p = _parse_reasons(reasons)
        assert p[0]["shap"] == pytest.approx(2.5)

    def test_negative_shap_parsed(self):
        reasons = ("Low-risk indicator (SHAP -1.200): Some desc",)
        p = _parse_reasons(reasons)
        assert p[0]["shap"] == pytest.approx(-1.2)

    def test_max_width_is_100(self):
        reasons = ("High-risk indicator (SHAP +4.000): Desc A",)
        p = _parse_reasons(reasons)
        assert p[0]["width"] == 100

    def test_widths_proportional(self):
        reasons = (
            "High-risk indicator (SHAP +4.000): Desc A",
            "Low-risk indicator (SHAP -2.000): Desc B",
        )
        p = _parse_reasons(reasons)
        assert p[0]["width"] == 100
        assert p[1]["width"] == 50

    def test_description_extracted(self):
        reasons = ("Low-risk indicator (SHAP -1.000): File size is 1234 bytes",)
        p = _parse_reasons(reasons)
        assert p[0]["desc"] == "File size is 1234 bytes"

    def test_unparseable_line_survives(self):
        # Malformed reasons should not crash; they get a fallback entry
        reasons = ("this is not a valid reason line",)
        p = _parse_reasons(reasons)
        assert len(p) == 1
        assert "this is not" in p[0]["desc"]


# ── AppState ───────────────────────────────────────────────────────────────────

class TestAppState:
    def test_add_and_get(self):
        state = AppState()
        entry = _make_entry("x-1")
        state.add_scan(entry)
        assert state.get("x-1") is entry

    def test_get_missing_returns_none(self):
        assert AppState().get("nope") is None

    def test_recent_newest_first(self):
        state = AppState()
        state.add_scan(_make_entry("first"))
        state.add_scan(_make_entry("second"))
        assert state.recent()[0].id == "second"

    def test_counts(self):
        state = AppState()
        state.add_scan(_make_entry("a", "BENIGN"))
        state.add_scan(_make_entry("b", "MALICIOUS"))
        c = state.counts()
        assert c["total"] == 2
        assert c["threats"] == 1
        assert c["clean"] == 1

    def test_max_history_enforced(self):
        state = AppState(max_history=3)
        for i in range(5):
            state.add_scan(_make_entry(str(i)))
        assert len(state.recent(10)) == 3

    def test_sse_broadcast(self):
        import queue
        state = AppState()
        q = state.subscribe()
        entry = _make_entry("broadcast-test")
        state.add_scan(entry)
        received = q.get_nowait()
        assert received is entry

    def test_sse_unsubscribe(self):
        import queue
        state = AppState()
        q = state.subscribe()
        state.unsubscribe(q)
        state.add_scan(_make_entry("after-unsub"))
        with pytest.raises(queue.Empty):
            q.get_nowait()
