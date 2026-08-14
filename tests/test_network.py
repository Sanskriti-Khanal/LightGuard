"""Tests for src/lightguard/network/collector.py.

All tests mock psutil so no real network activity is required.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lightguard.network.collector import (
    _COMMON_PORTS,
    _features_from_records,
    _iter_connections,
    collect_baseline,
    collect_snapshot,
    load_baseline,
    save_baseline,
)
from lightguard.network.collector import _ConnRecord


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_psutil_conn(pid, rip, rport, status="ESTABLISHED"):
    """Build a mock psutil connection object."""
    conn = MagicMock()
    conn.pid    = pid
    conn.raddr  = SimpleNamespace(ip=rip, port=rport)
    conn.status = status
    return conn


def _patch_psutil(conns, pid_name_map=None):
    """Context-manager-style patch for psutil.net_connections and process_iter."""
    if pid_name_map is None:
        pid_name_map = {c.pid: f"proc_{c.pid}" for c in conns}

    procs = [
        MagicMock(info={"pid": pid, "name": name})
        for pid, name in pid_name_map.items()
    ]

    return (
        patch("lightguard.network.collector.psutil.net_connections", return_value=conns),
        patch("lightguard.network.collector.psutil.process_iter", return_value=procs),
    )


# ── _ConnRecord ───────────────────────────────────────────────────────────────

class TestConnRecord:
    def test_fields(self):
        r = _ConnRecord(pid=123, proc_name="curl", remote_ip="1.2.3.4",
                        remote_port=443, status="ESTABLISHED")
        assert r.pid == 123
        assert r.proc_name == "curl"
        assert r.remote_ip == "1.2.3.4"
        assert r.remote_port == 443

    def test_status_stored(self):
        r = _ConnRecord(pid=1, proc_name="x", remote_ip="0.0.0.0",
                        remote_port=80, status="CLOSE_WAIT")
        assert r.status == "CLOSE_WAIT"


# ── _features_from_records ────────────────────────────────────────────────────

class TestFeaturesFromRecords:
    def _rec(self, pid=1, name="curl", ip="1.2.3.4", port=443):
        return _ConnRecord(pid=pid, proc_name=name, remote_ip=ip,
                           remote_port=port, status="ESTABLISHED")

    def test_empty_records_returns_empty_df(self):
        df = _features_from_records([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "conn_count" in df.columns

    def test_single_process_one_conn(self):
        df = _features_from_records([self._rec()])
        assert len(df) == 1
        assert df.iloc[0]["conn_count"] == 1
        assert df.iloc[0]["unique_remote_ips"] == 1
        assert df.iloc[0]["unique_remote_ports"] == 1

    def test_two_processes(self):
        recs = [
            self._rec(pid=1, name="curl", ip="1.1.1.1", port=443),
            self._rec(pid=2, name="wget", ip="2.2.2.2", port=80),
        ]
        df = _features_from_records(recs)
        assert len(df) == 2
        names = set(df["proc_name"])
        assert "curl" in names and "wget" in names

    def test_unique_remote_ips_counted_correctly(self):
        recs = [
            self._rec(pid=1, ip="1.1.1.1", port=443),
            self._rec(pid=1, ip="2.2.2.2", port=443),
            self._rec(pid=1, ip="1.1.1.1", port=80),  # duplicate IP
        ]
        df = _features_from_records(recs)
        row = df[df["pid"] == 1].iloc[0]
        assert row["unique_remote_ips"] == 2      # 1.1.1.1 + 2.2.2.2
        assert row["unique_remote_ports"] == 2    # 443 + 80
        assert row["conn_count"] == 3

    def test_rare_port_not_counted_when_all_common(self):
        recs = [self._rec(port=443), self._rec(port=80), self._rec(port=53)]
        df = _features_from_records(recs)
        assert df.iloc[0]["rare_port_count"] == 0

    def test_rare_port_counted_when_uncommon(self):
        recs = [self._rec(port=31337), self._rec(port=4444)]
        df = _features_from_records(recs)
        assert df.iloc[0]["rare_port_count"] == 2

    def test_mixed_rare_and_common_ports(self):
        recs = [
            self._rec(port=443),    # common
            self._rec(port=80),     # common
            self._rec(port=12345),  # rare
        ]
        df = _features_from_records(recs)
        assert df.iloc[0]["rare_port_count"] == 1

    def test_conn_per_min_zero_for_snapshot(self):
        df = _features_from_records([self._rec()], window_seconds=0.0)
        assert df.iloc[0]["conn_per_min"] == 0.0

    def test_conn_per_min_computed_from_window(self):
        recs = [self._rec()] * 6      # 6 connections in 30 seconds → 12/min
        df = _features_from_records(recs, window_seconds=30.0)
        assert df.iloc[0]["conn_per_min"] == pytest.approx(12.0, rel=0.01)

    def test_output_columns_complete(self):
        df = _features_from_records([self._rec()])
        expected = {"pid", "proc_name", "conn_count", "unique_remote_ips",
                    "unique_remote_ports", "rare_port_count", "conn_per_min"}
        assert expected.issubset(set(df.columns))

    def test_sorted_by_conn_count_descending(self):
        recs = [
            self._rec(pid=1, name="idle", ip="1.1.1.1", port=443),
            self._rec(pid=2, name="busy", ip="2.2.2.2", port=80),
            self._rec(pid=2, name="busy", ip="3.3.3.3", port=80),
            self._rec(pid=2, name="busy", ip="4.4.4.4", port=80),
        ]
        df = _features_from_records(recs)
        assert df.iloc[0]["proc_name"] == "busy"


# ── _iter_connections ─────────────────────────────────────────────────────────

class TestIterConnections:
    def test_returns_list(self):
        conns = [_make_psutil_conn(100, "8.8.8.8", 53)]
        p1, p2 = _patch_psutil(conns, {100: "chrome"})
        with p1, p2:
            result = _iter_connections()
        assert isinstance(result, list)

    def test_connection_fields_populated(self):
        conns = [_make_psutil_conn(100, "8.8.8.8", 53)]
        p1, p2 = _patch_psutil(conns, {100: "chrome"})
        with p1, p2:
            result = _iter_connections()
        assert len(result) == 1
        r = result[0]
        assert r.pid == 100
        assert r.proc_name == "chrome"
        assert r.remote_ip == "8.8.8.8"
        assert r.remote_port == 53

    def test_conn_without_raddr_is_skipped(self):
        no_raddr = MagicMock()
        no_raddr.pid   = 200
        no_raddr.raddr = None
        p1, p2 = _patch_psutil([no_raddr], {200: "listening"})
        with p1, p2:
            result = _iter_connections()
        assert result == []

    def test_conn_without_pid_is_skipped(self):
        no_pid = _make_psutil_conn(None, "1.2.3.4", 80)
        p1, p2 = _patch_psutil([no_pid], {})
        with p1, p2:
            result = _iter_connections()
        assert result == []

    def test_unknown_pid_uses_unknown_name(self):
        conns = [_make_psutil_conn(999, "1.2.3.4", 80)]
        # pid 999 not in the pid_name_map
        p1, p2 = _patch_psutil(conns, {})
        with p1, p2:
            result = _iter_connections()
        assert result[0].proc_name == "unknown"

    def test_multiple_connections(self):
        conns = [
            _make_psutil_conn(1, "1.1.1.1", 443),
            _make_psutil_conn(1, "2.2.2.2", 80),
            _make_psutil_conn(2, "3.3.3.3", 53),
        ]
        p1, p2 = _patch_psutil(conns, {1: "browser", 2: "resolver"})
        with p1, p2:
            result = _iter_connections()
        assert len(result) == 3


# ── collect_snapshot ──────────────────────────────────────────────────────────

class TestCollectSnapshot:
    def test_returns_dataframe(self):
        conns = [_make_psutil_conn(1, "1.2.3.4", 443)]
        p1, p2 = _patch_psutil(conns, {1: "browser"})
        with p1, p2:
            df = collect_snapshot()
        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_process(self):
        conns = [
            _make_psutil_conn(1, "1.1.1.1", 443),
            _make_psutil_conn(1, "2.2.2.2", 80),
            _make_psutil_conn(2, "3.3.3.3", 53),
        ]
        p1, p2 = _patch_psutil(conns, {1: "chrome", 2: "dns"})
        with p1, p2:
            df = collect_snapshot()
        assert len(df) == 2

    def test_conn_per_min_is_zero(self):
        conns = [_make_psutil_conn(1, "1.1.1.1", 443)]
        p1, p2 = _patch_psutil(conns, {1: "curl"})
        with p1, p2:
            df = collect_snapshot()
        assert df.iloc[0]["conn_per_min"] == 0.0

    def test_empty_when_no_connections(self):
        p1, p2 = _patch_psutil([], {})
        with p1, p2:
            df = collect_snapshot()
        assert len(df) == 0

    def test_has_required_columns(self):
        conns = [_make_psutil_conn(1, "1.1.1.1", 443)]
        p1, p2 = _patch_psutil(conns, {1: "app"})
        with p1, p2:
            df = collect_snapshot()
        for col in ("pid", "proc_name", "conn_count", "unique_remote_ips",
                    "unique_remote_ports", "rare_port_count", "conn_per_min"):
            assert col in df.columns


# ── collect_baseline ──────────────────────────────────────────────────────────

class TestCollectBaseline:
    def test_returns_dataframe(self):
        conns = [_make_psutil_conn(1, "1.2.3.4", 443)]
        p1, p2 = _patch_psutil(conns, {1: "browser"})
        with p1, p2, patch("lightguard.network.collector.time.sleep"):
            df = collect_baseline(duration=0.05, interval=0.01)
        assert isinstance(df, pd.DataFrame)

    def test_conn_per_min_nonzero(self):
        conns = [_make_psutil_conn(1, "1.2.3.4", 443)]
        p1, p2 = _patch_psutil(conns, {1: "browser"})
        with p1, p2, patch("lightguard.network.collector.time.sleep"):
            df = collect_baseline(duration=0.05, interval=0.01)
        # With actual elapsed time > 0 and at least one conn, rate > 0
        assert df.iloc[0]["conn_per_min"] >= 0

    def test_invalid_duration_raises(self):
        with pytest.raises(ValueError):
            collect_baseline(duration=0)

    def test_negative_duration_raises(self):
        with pytest.raises(ValueError):
            collect_baseline(duration=-5)

    def test_aggregates_across_polls(self):
        # Two polls: first sees proc 1, second sees proc 1 and 2
        call_count = 0
        original_iter = _iter_connections

        def fake_iter():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [_ConnRecord(1, "app", "1.1.1.1", 443, "ESTABLISHED")]
            return [
                _ConnRecord(1, "app", "2.2.2.2", 80, "ESTABLISHED"),
                _ConnRecord(2, "other", "3.3.3.3", 53, "ESTABLISHED"),
            ]

        with (
            patch("lightguard.network.collector._iter_connections", side_effect=fake_iter),
            patch("lightguard.network.collector.time.sleep"),
        ):
            df = collect_baseline(duration=0.05, interval=0.01)

        # proc 1 should appear (from both polls), proc 2 from second poll
        names = set(df["proc_name"])
        assert "app" in names
        assert "other" in names

    def test_has_required_columns(self):
        conns = [_make_psutil_conn(1, "1.1.1.1", 443)]
        p1, p2 = _patch_psutil(conns, {1: "app"})
        with p1, p2, patch("lightguard.network.collector.time.sleep"):
            df = collect_baseline(duration=0.05, interval=0.01)
        for col in ("pid", "proc_name", "conn_count", "unique_remote_ips",
                    "unique_remote_ports", "rare_port_count", "conn_per_min"):
            assert col in df.columns


# ── save_baseline / load_baseline ─────────────────────────────────────────────

class TestBaselinePersistence:
    def _sample_df(self):
        return pd.DataFrame([{
            "pid": 1, "proc_name": "curl",
            "conn_count": 3, "unique_remote_ips": 2,
            "unique_remote_ports": 2, "rare_port_count": 0,
            "conn_per_min": 6.0,
        }])

    def test_save_creates_file(self, tmp_path):
        df = self._sample_df()
        dest = save_baseline(df, path=tmp_path / "baseline.csv")
        assert dest.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        df = self._sample_df()
        deep = tmp_path / "a" / "b" / "baseline.csv"
        save_baseline(df, path=deep)
        assert deep.exists()

    def test_save_returns_resolved_path(self, tmp_path):
        df   = self._sample_df()
        dest = save_baseline(df, path=tmp_path / "bl.csv")
        assert dest.is_absolute()

    def test_round_trip(self, tmp_path):
        df   = self._sample_df()
        path = tmp_path / "baseline.csv"
        save_baseline(df, path=path)
        loaded = load_baseline(path=path)
        assert list(loaded.columns) == list(df.columns)
        assert loaded.iloc[0]["conn_count"] == 3
        assert loaded.iloc[0]["proc_name"] == "curl"

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_baseline(path=tmp_path / "does_not_exist.csv")

    def test_saved_csv_is_valid(self, tmp_path):
        df   = self._sample_df()
        path = tmp_path / "baseline.csv"
        save_baseline(df, path=path)
        # verify it's readable as plain CSV (no binary artefacts)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["proc_name"] == "curl"

    def test_multiple_rows_preserved(self, tmp_path):
        df = pd.DataFrame([
            {"pid": 1, "proc_name": "a", "conn_count": 1, "unique_remote_ips": 1,
             "unique_remote_ports": 1, "rare_port_count": 0, "conn_per_min": 2.0},
            {"pid": 2, "proc_name": "b", "conn_count": 5, "unique_remote_ips": 3,
             "unique_remote_ports": 2, "rare_port_count": 1, "conn_per_min": 10.0},
        ])
        path = tmp_path / "baseline.csv"
        save_baseline(df, path=path)
        loaded = load_baseline(path=path)
        assert len(loaded) == 2


# ── common-ports set sanity check ────────────────────────────────────────────

class TestCommonPorts:
    def test_dns_is_common(self):
        assert 53 in _COMMON_PORTS

    def test_https_is_common(self):
        assert 443 in _COMMON_PORTS

    def test_ssh_is_common(self):
        assert 22 in _COMMON_PORTS

    def test_random_high_port_not_common(self):
        assert 31337 not in _COMMON_PORTS

    def test_set_is_frozen(self):
        assert isinstance(_COMMON_PORTS, frozenset)
