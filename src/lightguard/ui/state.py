"""Shared in-memory state for the Flask UI.

Holds the rolling scan history and the fan-out queue used by the SSE feed.
A single AppState instance is created by create_app() and attached to the
Flask app object so all request handlers share it.
"""

from __future__ import annotations

import queue
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ScanEntry:
    """One scan result stored in the history."""
    id: str
    verdict: object        # lightguard.monitor.scan.Verdict
    scanned_at: datetime
    source: str            # "manual" | "watch"


class AppState:
    """Thread-safe, bounded scan history with SSE fan-out."""

    def __init__(self, max_history: int = 50) -> None:
        self._scans: deque[ScanEntry] = deque(maxlen=max_history)
        self._sse_queues: list[queue.Queue] = []

    # ── history ───────────────────────────────────────────────────────────────

    def add_scan(self, entry: ScanEntry) -> None:
        """Prepend *entry* to history and broadcast to SSE listeners."""
        self._scans.appendleft(entry)
        for q in list(self._sse_queues):
            try:
                q.put_nowait(entry)
            except queue.Full:
                pass

    def recent(self, n: int = 20) -> list[ScanEntry]:
        """Return the *n* most recent entries, newest first."""
        return list(self._scans)[:n]

    def get(self, scan_id: str) -> ScanEntry | None:
        """Return entry by id, or None."""
        for e in self._scans:
            if e.id == scan_id:
                return e
        return None

    def counts(self) -> dict[str, int]:
        """Return total / threats / clean counts."""
        total = len(self._scans)
        threats = sum(1 for e in self._scans if e.verdict.label == "MALICIOUS")
        return {"total": total, "threats": threats, "clean": total - threats}

    # ── SSE fan-out ───────────────────────────────────────────────────────────

    def subscribe(self) -> queue.Queue:
        """Register a new SSE listener; returns its queue."""
        q: queue.Queue = queue.Queue(maxsize=20)
        self._sse_queues.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        """Remove an SSE listener queue."""
        try:
            self._sse_queues.remove(q)
        except ValueError:
            pass
