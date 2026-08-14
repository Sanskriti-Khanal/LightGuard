"""LightGuard v2 — per-process network anomaly detection module.

Collector::

    from lightguard.network.collector import (
        collect_snapshot, collect_baseline, save_baseline, load_baseline,
    )

Detector::

    from lightguard.network.detector import (
        train, score, save_detector, load_detector, FEATURE_COLS,
    )

Explainer::

    from lightguard.network.explain import explain_process, explain_snapshot
"""

from lightguard.network.collector import (
    collect_baseline,
    collect_snapshot,
    load_baseline,
    save_baseline,
)
from lightguard.network.detector import (
    FEATURE_COLS,
    load_detector,
    save_detector,
    score,
    train,
)
from lightguard.network.explain import explain_process, explain_snapshot

__all__ = [
    "collect_baseline", "collect_snapshot", "load_baseline", "save_baseline",
    "train", "score", "save_detector", "load_detector", "FEATURE_COLS",
    "explain_process", "explain_snapshot",
]
