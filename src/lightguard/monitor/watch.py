"""Watchdog-based folder monitor — scans new PE files as they arrive.

Usage::

    from lightguard.malware.train import load_model
    from lightguard.monitor.watch import Watcher

    model = load_model("models/lightguard_lgbm.txt")

    def on_verdict(verdict):
        print(verdict)

    watcher = Watcher("~/Downloads", model, on_verdict, threshold=0.80)
    watcher.start()
    # … runs until watcher.stop() is called
"""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Callable

from watchdog.events import FileCreatedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from lightguard.monitor.scan import Verdict, scan

log = logging.getLogger(__name__)

# Extensions treated as PE files worth scanning
_PE_EXTENSIONS = {".exe", ".dll", ".sys", ".scr", ".com"}


class _PEHandler(FileSystemEventHandler):
    """Enqueues scan jobs for newly created PE files."""

    def __init__(self, job_queue: queue.Queue) -> None:
        super().__init__()
        self._q = job_queue

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() in _PE_EXTENSIONS:
            self._q.put(path)


class Watcher:
    """Watch *folder* and invoke *on_verdict* for every new PE file detected.

    Scanning runs on a dedicated worker thread so the watchdog callback
    returns immediately and never blocks the observer loop.

    Args:
        folder:     directory to watch (expanded with Path.expanduser).
        model:      loaded lgb.Booster.
        on_verdict: callable that receives one Verdict per scanned file.
        threshold:  risk probability cutoff for MALICIOUS label.
        explainer:  optional shap.TreeExplainer — when supplied, each Verdict
                    includes plain-English reasons from SHAP.
        top_k:      number of SHAP reasons to include per verdict.
    """

    def __init__(
        self,
        folder: str | Path,
        model,
        on_verdict: Callable[[Verdict], None],
        threshold: float = 0.80,
        explainer=None,
        top_k: int = 5,
    ) -> None:
        self._folder = Path(folder).expanduser()
        self._model = model
        self._on_verdict = on_verdict
        self._threshold = threshold
        self._explainer = explainer
        self._top_k = top_k

        self._q: queue.Queue[Path | None] = queue.Queue()
        self._observer = Observer()
        self._observer.schedule(_PEHandler(self._q), str(self._folder), recursive=False)

        self._worker = threading.Thread(target=self._scan_loop, daemon=True, name="lg-scanner")

    # ── public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the observer and the scanner worker thread."""
        self._folder.mkdir(parents=True, exist_ok=True)
        self._observer.start()
        self._worker.start()
        log.info("Watcher started on %s", self._folder)

    def stop(self) -> None:
        """Stop gracefully — drains the queue before returning."""
        self._observer.stop()
        self._observer.join()
        self._q.put(None)   # sentinel to unblock the worker
        self._worker.join()
        log.info("Watcher stopped")

    def scan_now(self, pe_path: str | Path) -> Verdict:
        """Scan *pe_path* immediately (synchronous, no queue).

        Useful for one-off manual scans without starting the full observer.
        """
        return scan(pe_path, self._model, self._threshold,
                    explainer=self._explainer, top_k=self._top_k)

    # ── internal ──────────────────────────────────────────────────────────────

    def _scan_loop(self) -> None:
        while True:
            path = self._q.get()
            if path is None:
                break
            try:
                verdict = scan(path, self._model, self._threshold,
                               explainer=self._explainer, top_k=self._top_k)
                self._on_verdict(verdict)
            except Exception:
                log.exception("Error scanning %s", path)
