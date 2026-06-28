#!/usr/bin/env python
"""LightGuard v2 — network anomaly detection live demo.

Workflow
--------
1. Collect a baseline of normal per-process network activity.
2. Fit an IsolationForest detector on that baseline and save it.
3. Poll live snapshots every few seconds, score each process, and print a
   colour-coded table.  ANOMALOUS processes are highlighted with their top
   plain-English reasons.

Anomaly simulation (--simulate)
--------------------------------
Opens 40 simultaneous TCP connections from this process to a local server
on an unusual port (31337).  The current process will then show up in the
snapshot with a high conn_count, high rare_port_count, and high
conn_per_min — enough to push it well above the baseline and trigger an
ANOMALOUS label with reasons like "Using ports not normally used by this app".

macOS note
----------
For full process visibility, run with sudo:

    sudo .venv/bin/python scripts/network_demo.py

Without sudo, only processes owned by your login user are visible (typically
5–15 processes).  The anomaly simulation still works without sudo because the
generated connections belong to this process.

Usage
-----
    # Baseline 60 s, then live loop (Ctrl+C to stop):
    sudo .venv/bin/python scripts/network_demo.py

    # Shorter baseline, faster polling, with simulated anomaly:
    sudo .venv/bin/python scripts/network_demo.py \\
        --baseline-seconds 30 --interval 3 --simulate

    # Skip baseline collection (use a saved baseline):
    sudo .venv/bin/python scripts/network_demo.py --skip-baseline
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import yaml
import pandas as pd

from lightguard.network.collector import (
    collect_baseline,
    collect_snapshot,
    load_baseline,
    save_baseline,
)
from lightguard.network.detector import (
    load_detector,
    save_detector,
    score,
    train,
)
from lightguard.network.explain import explain_snapshot

# ── ANSI colours ──────────────────────────────────────────────────────────────

_RED    = "\033[91m"
_YELLOW = "\033[93m"
_GREEN  = "\033[92m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"

_USE_COLOUR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"{code}{text}{_RESET}" if _USE_COLOUR else text


def _bold(t: str) -> str: return _c(_BOLD, t)
def _red(t: str)  -> str: return _c(_RED  + _BOLD, t)
def _green(t: str)-> str: return _c(_GREEN, t)
def _cyan(t: str) -> str: return _c(_CYAN, t)
def _dim(t: str)  -> str: return _c(_DIM, t)


# ── Table printer ─────────────────────────────────────────────────────────────

_COL_WIDTHS = {
    "proc_name":    24,
    "pid":           7,
    "anomaly_score": 7,
    "label":         10,
    "conn_count":    6,
    "unique_ips":    7,
    "rare_ports":    6,
    "conn_per_min":  9,
}

_HEADER_LABELS = {
    "proc_name":    "PROCESS",
    "pid":          "PID",
    "anomaly_score":"SCORE",
    "label":        "VERDICT",
    "conn_count":   "CONNS",
    "unique_ips":   "UNIQ IP",
    "rare_ports":   "RARE",
    "conn_per_min": "C/MIN",
}


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


def _print_header() -> None:
    parts = []
    for col, w in _COL_WIDTHS.items():
        parts.append(_bold(_HEADER_LABELS[col].ljust(w)))
    print("  " + "  ".join(parts))
    sep = "  " + "  ".join("─" * w for w in _COL_WIDTHS.values())
    print(_dim(sep))


def _print_row(row: pd.Series, reasons: list[str]) -> None:
    is_anomalous = row.get("label") == "ANOMALOUS"
    score_val    = int(row.get("anomaly_score", 0))

    # Colour the score badge
    if score_val >= 75:
        score_str = _red(f"{score_val:3d}/100")
    elif score_val >= 50:
        score_str = _c(_YELLOW + _BOLD, f"{score_val:3d}/100")
    else:
        score_str = _green(f"{score_val:3d}/100")

    label_str = (
        _red("ANOMALOUS ")
        if is_anomalous
        else _green("NORMAL    ")
    )

    proc_raw  = _trunc(str(row.get("proc_name", "?")), _COL_WIDTHS["proc_name"])
    proc_str  = _red(proc_raw.ljust(_COL_WIDTHS["proc_name"])) if is_anomalous else proc_raw.ljust(_COL_WIDTHS["proc_name"])
    pid_str   = str(int(row.get("pid", 0))).rjust(_COL_WIDTHS["pid"])
    conns_str = str(int(row.get("conn_count", 0))).rjust(_COL_WIDTHS["conn_count"])
    ips_str   = str(int(row.get("unique_remote_ips", 0))).rjust(_COL_WIDTHS["unique_ips"])
    rare_str  = str(int(row.get("rare_port_count", 0))).rjust(_COL_WIDTHS["rare_ports"])
    cpm_str   = f"{float(row.get('conn_per_min', 0)):.1f}".rjust(_COL_WIDTHS["conn_per_min"])

    prefix = _red("▶ ") if is_anomalous else "  "
    print(f"{prefix}{proc_str}  {pid_str}  {score_str}  {label_str}  {conns_str}  {ips_str}  {rare_str}  {cpm_str}")

    for reason in reasons:
        print(_c(_YELLOW, f"      ↳  {reason}"))


def _print_snapshot_table(results: pd.DataFrame) -> None:
    _print_header()
    for _, row in results.iterrows():
        reasons = row.get("reasons", []) or []
        _print_row(row, reasons)


# ── Baseline collection ───────────────────────────────────────────────────────

def _collect_and_train(
    baseline_seconds: int,
    interval: float,
    baseline_path: Path,
    detector_path: Path,
    random_seed: int,
) -> object:
    """Collect baseline, train detector, save both.  Returns the detector."""
    print()
    print(_bold("─── Phase 1: baseline collection ───────────────────────────────────"))
    print(f"  Recording normal activity for {baseline_seconds} s "
          f"(polling every {interval:.0f} s).")
    print("  Browse normally — do not open unusual apps during this window.")
    print()

    poll_interval = min(interval, 5.0)
    n_polls       = max(1, int(baseline_seconds / poll_interval))
    deadline      = time.monotonic() + baseline_seconds

    # Progress bar
    bar_width = 40
    collected_records: list = []

    from lightguard.network.collector import _iter_connections, _features_from_records

    start = time.monotonic()
    poll  = 0
    while time.monotonic() < deadline:
        collected_records.extend(_iter_connections())
        poll += 1
        elapsed   = time.monotonic() - start
        remaining = max(0.0, baseline_seconds - elapsed)
        filled    = int(bar_width * elapsed / baseline_seconds)
        bar       = "█" * filled + "░" * (bar_width - filled)
        sys.stdout.write(
            f"\r  [{_cyan(bar)}] {elapsed:4.0f}/{baseline_seconds} s  "
            f"({poll} poll{'s' if poll != 1 else ''})  "
            f"{remaining:.0f} s remaining   "
        )
        sys.stdout.flush()
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(poll_interval, remaining))

    sys.stdout.write("\n")

    elapsed = time.monotonic() - start
    baseline_df = _features_from_records(collected_records, window_seconds=max(elapsed, 1e-6))

    n_proc = len(baseline_df)
    print(f"\n  Baseline: {n_proc} process{'es' if n_proc != 1 else ''} observed.")

    if n_proc == 0:
        print(_red("  No connections found.  Try running with sudo for full visibility."))
        sys.exit(1)

    dest = save_baseline(baseline_df, path=baseline_path)
    print(f"  Saved baseline → {dest}")

    print()
    print(_bold("─── Phase 2: training detector ─────────────────────────────────────"))
    detector = train(baseline_df, random_seed=random_seed)
    det_dest = save_detector(detector, path=detector_path)
    print(f"  IsolationForest fitted on {n_proc} process rows.")
    print(f"  Saved detector  → {det_dest}")

    return detector


# ── Anomaly simulator ─────────────────────────────────────────────────────────

_SIM_PORTS    = [31337, 4444]   # obviously unusual ports
_SIM_N_CONNS  = 120             # connections to hold open (60 per port)
_SIM_INTERVAL = 0.02            # seconds between connection bursts


class _AnomalySimulator:
    """Opens many simultaneous connections from this process to a local server
    on an unusual port, making this process appear anomalous to the detector.

    All sockets are local (127.0.0.1) so no external network access is needed.
    """

    def __init__(self) -> None:
        self._server:  socket.socket | None = None
        self._clients: list[socket.socket] = []
        self._thread:  threading.Thread | None = None
        self._stop     = threading.Event()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Give the server a moment to bind before the first snapshot poll
        time.sleep(0.5)

    def stop(self) -> None:
        self._stop.set()
        for s in self._clients:
            try: s.close()
            except Exception: pass
        if self._server:
            try: self._server.close()
            except Exception: pass
        self._clients.clear()

    def _run(self) -> None:
        # Start one accept server per unusual port.
        servers: list[socket.socket] = []
        accepted: list[socket.socket] = []

        for port in _SIM_PORTS:
            try:
                srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind(("127.0.0.1", port))
                srv.listen(256)
                srv.settimeout(1.0)
                servers.append(srv)
                self._server = srv   # keep last for cleanup reference
            except OSError as e:
                print(_red(f"\n  [simulator] Cannot bind port {port}: {e}"))

        if not servers:
            return

        def _acceptor(srv: socket.socket) -> None:
            while not self._stop.is_set():
                try:
                    conn, _ = srv.accept()
                    accepted.append(conn)
                except (socket.timeout, OSError):
                    pass

        for srv in servers:
            threading.Thread(target=_acceptor, args=(srv,), daemon=True).start()

        # Open _SIM_N_CONNS connections spread across all unusual ports.
        conns_per_port = max(1, _SIM_N_CONNS // len(servers))
        for port in _SIM_PORTS:
            for _ in range(conns_per_port):
                if self._stop.is_set():
                    break
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(2.0)
                    s.connect(("127.0.0.1", port))
                    self._clients.append(s)
                    time.sleep(_SIM_INTERVAL)
                except OSError:
                    pass

        # Hold all connections open until stopped.
        while not self._stop.is_set():
            time.sleep(0.5)

        for s in accepted:
            try: s.close()
            except Exception: pass
        for srv in servers:
            try: srv.close()
            except Exception: pass


# ── Live monitoring loop ──────────────────────────────────────────────────────

def _live_loop(
    detector,
    interval: float,
    simulate: bool,
) -> None:
    print()
    print(_bold("─── Phase 3: live monitoring ────────────────────────────────────────"))

    sim: _AnomalySimulator | None = None
    if simulate:
        print(_c(_YELLOW, f"  [simulate] Starting anomaly generator on port {_SIM_PORT}…"))
        sim = _AnomalySimulator()
        sim.start()
        print(_c(_YELLOW, f"  [simulate] Holding {_SIM_N_CONNS} open connections. "
                           "This process should appear ANOMALOUS shortly."))

    print(f"  Polling every {interval:.0f} s.  Press Ctrl+C to stop.\n")

    poll = 0
    try:
        while True:
            poll += 1
            ts = time.strftime("%H:%M:%S")
            print(_bold(f"  ── snapshot #{poll}  {ts} {'─' * 48}"))

            snap    = collect_snapshot()
            results = score(snap, detector)
            results = explain_snapshot(results, detector, top_k=3)

            n_anom = (results["label"] == "ANOMALOUS").sum()
            n_norm = (results["label"] == "NORMAL").sum()

            status = (
                _red(f"  ⚠  {n_anom} ANOMALOUS  /  {n_norm} NORMAL")
                if n_anom
                else _green(f"  ✓  All {n_norm} processes NORMAL")
            )
            print(status)
            print()
            _print_snapshot_table(results)
            print()

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        if sim:
            sim.stop()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="network_demo.py",
        description=(
            "LightGuard v2 — network anomaly detection demo.\n\n"
            "macOS note: run with sudo for full process visibility:\n"
            "  sudo .venv/bin/python scripts/network_demo.py\n\n"
            "Without sudo only processes owned by your user are visible."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--baseline-seconds", type=int, default=60, metavar="N",
        help="seconds to record normal baseline (default: 60)",
    )
    p.add_argument(
        "--interval", type=float, default=5.0, metavar="S",
        help="seconds between live snapshots (default: 5)",
    )
    p.add_argument(
        "--simulate", action="store_true",
        help=(
            f"inject a synthetic anomaly: open {_SIM_N_CONNS} connections to "
            f"localhost on ports {_SIM_PORTS} so this process appears ANOMALOUS"
        ),
    )
    p.add_argument(
        "--skip-baseline", action="store_true",
        help="skip collection and load a previously saved baseline/detector",
    )
    p.add_argument(
        "--baseline-path", type=Path,
        default=REPO / "data" / "network" / "baseline.csv",
        help="where to save/load the baseline CSV (default: data/network/baseline.csv)",
    )
    p.add_argument(
        "--detector-path", type=Path,
        default=REPO / "data" / "network" / "detector.joblib",
        help="where to save/load the detector (default: data/network/detector.joblib)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg         = yaml.safe_load((REPO / "config.yaml").read_text())
    random_seed = cfg.get("random_seed", 42)

    print()
    print(_bold("╔══════════════════════════════════════════════════════════════════╗"))
    print(_bold("║           LightGuard v2 — Network Anomaly Detection             ║"))
    print(_bold("╚══════════════════════════════════════════════════════════════════╝"))
    running_as_root = os.getuid() == 0 if hasattr(os, "getuid") else False
    if not running_as_root:
        print(_c(_YELLOW, "  ⚠  Running without sudo — only your processes are visible."))
        print(_c(_YELLOW, "     For full visibility: sudo .venv/bin/python scripts/network_demo.py"))
    else:
        print(_green("  ✓  Running as root — full process visibility enabled."))

    if args.skip_baseline:
        print()
        print(_bold("─── Loading saved baseline and detector ─────────────────────────────"))
        try:
            detector = load_detector(path=args.detector_path)
            print(f"  Loaded detector  ← {args.detector_path}")
        except FileNotFoundError:
            # Try rebuilding from saved baseline CSV
            try:
                baseline_df = load_baseline(path=args.baseline_path)
                print(f"  Loaded baseline  ← {args.baseline_path} ({len(baseline_df)} rows)")
                print("  Training detector from baseline…")
                detector = train(baseline_df, random_seed=random_seed)
                save_detector(detector, path=args.detector_path)
                print(f"  Saved detector   → {args.detector_path}")
            except FileNotFoundError:
                print(_red(f"  No baseline at {args.baseline_path}."))
                print("  Run without --skip-baseline first to collect a baseline.")
                sys.exit(1)
    else:
        detector = _collect_and_train(
            baseline_seconds=args.baseline_seconds,
            interval=args.interval,
            baseline_path=args.baseline_path,
            detector_path=args.detector_path,
            random_seed=random_seed,
        )

    _live_loop(detector, interval=args.interval, simulate=args.simulate)


if __name__ == "__main__":
    main()
