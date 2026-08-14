"""Per-process network feature collector for LightGuard v2.

Uses psutil to observe live TCP/UDP connections and derive per-process
behavioural features suitable for anomaly detection:

  conn_count         — total ESTABLISHED connections
  unique_remote_ips  — distinct remote IP addresses contacted
  unique_remote_ports— distinct remote port numbers contacted
  rare_port_count    — connections to ports outside the common-service set
  conn_per_min       — connection rate (meaningful only from baseline windows)

Baseline workflow::

    baseline_df = collect_baseline(duration=120)   # 2-minute recording
    save_baseline(baseline_df)                      # → data/network/baseline.csv
    # ... later ...
    snap = collect_snapshot()
    ref  = load_baseline()

Single-shot workflow::

    snap = collect_snapshot()   # DataFrame, one row per observed process
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd
import psutil

# ── Ports considered "common" (low suspicion) ────────────────────────────────
# Covers DNS, HTTP(S), email, SSH, databases, and popular dev servers.
# Any destination port NOT in this set is counted as a rare-port connection.
_COMMON_PORTS: frozenset[int] = frozenset({
    20, 21,        # FTP
    22,            # SSH
    25, 465, 587,  # SMTP
    53,            # DNS
    67, 68,        # DHCP
    80, 8080,      # HTTP
    110, 995,      # POP3
    143, 993,      # IMAP
    443, 8443,     # HTTPS
    1194,          # OpenVPN
    3306,          # MySQL
    5432,          # PostgreSQL
    5900,          # VNC
    6379,          # Redis
    8888, 8889,    # Jupyter
    27017,         # MongoDB
})

# Default location for saved baselines (relative to repo root).
# Callers can override via the path= parameter on save/load functions.
_DEFAULT_BASELINE_DIR = Path("data") / "network"


@dataclass
class _ConnRecord:
    """Lightweight snapshot of one psutil connection entry."""
    pid:         int
    proc_name:   str
    remote_ip:   str
    remote_port: int
    status:      str


def _iter_connections() -> list[_ConnRecord]:
    """Yield one _ConnRecord per active inet connection that has a remote addr.

    Silently skips processes we have no permission to inspect (AccessDenied,
    NoSuchProcess) — common for system processes on macOS/Windows.
    """
    records: list[_ConnRecord] = []

    # Build pid→name map once to avoid repeated proc lookups.
    pid_name: dict[int, str] = {}
    for proc in psutil.process_iter(attrs=["pid", "name"]):
        try:
            pid_name[proc.info["pid"]] = proc.info["name"] or "unknown"
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    try:
        # Global call works on Linux and macOS with sudo.
        # pconn objects from net_connections() include a .pid field.
        conns = psutil.net_connections(kind="inet")
        for c in conns:
            if c.raddr and c.raddr.ip and c.pid is not None:
                records.append(_ConnRecord(
                    pid=c.pid,
                    proc_name=pid_name.get(c.pid, "unknown"),
                    remote_ip=c.raddr.ip,
                    remote_port=c.raddr.port,
                    status=c.status or "",
                ))
    except psutil.AccessDenied:
        # On macOS without sudo, net_connections raises globally.
        # Fall back to per-process iteration — pconn objects here do NOT have
        # a .pid field; we supply it from the process object instead.
        for proc in psutil.process_iter(attrs=["pid", "name"]):
            try:
                pid  = proc.info["pid"]
                name = proc.info.get("name") or "unknown"
                for c in proc.net_connections(kind="inet"):
                    if c.raddr and c.raddr.ip:
                        records.append(_ConnRecord(
                            pid=pid,
                            proc_name=pid_name.get(pid, name),
                            remote_ip=c.raddr.ip,
                            remote_port=c.raddr.port,
                            status=c.status or "",
                        ))
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                pass

    return records


def _features_from_records(
    records: list[_ConnRecord],
    window_seconds: float = 0.0,
) -> pd.DataFrame:
    """Aggregate _ConnRecords into one feature row per (pid, proc_name).

    Args:
        records:        list of connection records from one or more polls.
        window_seconds: elapsed time of the collection window; used to compute
                        conn_per_min.  Pass 0 for a single snapshot (rate = 0).

    Returns:
        DataFrame with columns:
          pid, proc_name, conn_count, unique_remote_ips,
          unique_remote_ports, rare_port_count, conn_per_min
    """
    grouped: dict[tuple[int, str], list[_ConnRecord]] = defaultdict(list)
    for r in records:
        grouped[(r.pid, r.proc_name)].append(r)

    rows = []
    for (pid, name), recs in grouped.items():
        remote_ips   = {r.remote_ip   for r in recs}
        remote_ports = {r.remote_port for r in recs}
        rare_ports   = {p for p in remote_ports if p not in _COMMON_PORTS}

        rate = (len(recs) / window_seconds * 60) if window_seconds > 0 else 0.0

        rows.append({
            "pid":               pid,
            "proc_name":         name,
            "conn_count":        len(recs),
            "unique_remote_ips": len(remote_ips),
            "unique_remote_ports": len(remote_ports),
            "rare_port_count":   len(rare_ports),
            "conn_per_min":      round(rate, 3),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "pid", "proc_name", "conn_count",
            "unique_remote_ips", "unique_remote_ports",
            "rare_port_count", "conn_per_min",
        ])

    return pd.DataFrame(rows).sort_values("conn_count", ascending=False).reset_index(drop=True)


# ── Public API ────────────────────────────────────────────────────────────────

def collect_snapshot() -> pd.DataFrame:
    """Return a single point-in-time network feature DataFrame.

    One row per process that has at least one active connection with a
    remote address.  conn_per_min is always 0 for a snapshot (no window).
    """
    records = _iter_connections()
    return _features_from_records(records, window_seconds=0.0)


def collect_baseline(
    duration: float,
    interval: float = 5.0,
) -> pd.DataFrame:
    """Poll connections for *duration* seconds and return aggregated features.

    Args:
        duration: how many seconds to record (e.g. 120 for a 2-minute baseline).
        interval: seconds between polls (default 5).  Lower values are more
                  accurate but increase CPU usage.

    Returns:
        DataFrame with the same columns as collect_snapshot(), but conn_count
        and unique_* reflect the entire window and conn_per_min is the observed
        rate over the full window.
    """
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")
    if interval <= 0 or interval > duration:
        interval = min(5.0, duration)

    all_records: list[_ConnRecord] = []
    start = time.monotonic()
    deadline = start + duration

    while time.monotonic() < deadline:
        all_records.extend(_iter_connections())
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(interval, remaining))

    elapsed = time.monotonic() - start
    return _features_from_records(all_records, window_seconds=max(elapsed, 1e-6))


def save_baseline(
    df: pd.DataFrame,
    path: str | Path | None = None,
) -> Path:
    """Write baseline DataFrame to disk as CSV.

    Args:
        df:   DataFrame from collect_baseline().
        path: destination file path.  Defaults to data/network/baseline.csv
              relative to the current working directory.

    Returns:
        The resolved path that was written.
    """
    dest = Path(path) if path else _DEFAULT_BASELINE_DIR / "baseline.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest.resolve()


def load_baseline(path: str | Path | None = None) -> pd.DataFrame:
    """Load a previously saved baseline from disk.

    Args:
        path: path to read.  Defaults to data/network/baseline.csv.

    Returns:
        DataFrame with the baseline features.

    Raises:
        FileNotFoundError: if the baseline file does not exist.
    """
    src = Path(path) if path else _DEFAULT_BASELINE_DIR / "baseline.csv"
    if not src.exists():
        raise FileNotFoundError(
            f"No baseline found at {src}. "
            "Run collect_baseline() and save_baseline() first."
        )
    return pd.read_csv(src)
