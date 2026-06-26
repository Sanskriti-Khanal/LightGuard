"""Flask application factory for the LightGuard local UI.

Creates a fully offline app that displays scan verdicts and SHAP explanations.
No requests leave the machine after startup.

Usage (programmatic)::

    from lightguard.ui.app import create_app
    app = create_app(model=lgb_booster, explainer=shap_explainer,
                     watch_folder="~/Downloads")
    app.run()

Usage (CLI): see scripts/run_app.py
"""

from __future__ import annotations

import json
import queue
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

from lightguard.ui.state import AppState, ScanEntry


def create_app(
    model=None,
    explainer=None,
    watch_folder: str | Path | None = None,
    top_k: int = 5,
) -> Flask:
    """Return a configured Flask app.

    Args:
        model:        loaded lgb.Booster, or None (UI read-only without scanning).
        explainer:    shap.TreeExplainer, or None (verdicts have no reasons).
        watch_folder: folder for the background Watcher, or None.
        top_k:        SHAP features shown per verdict.
    """
    app = Flask(__name__, instance_relative_config=False)
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload cap

    state = AppState()
    app.state = state         # type: ignore[attr-defined]
    app._watcher = None       # type: ignore[attr-defined]

    # ── optional background watcher ──────────────────────────────────────────
    if model is not None and watch_folder is not None:
        from lightguard.monitor.watch import Watcher

        def _on_verdict(verdict) -> None:
            entry = ScanEntry(
                id=str(uuid.uuid4()),
                verdict=verdict,
                scanned_at=datetime.now(),
                source="watch",
            )
            state.add_scan(entry)

        watcher = Watcher(
            watch_folder, model, _on_verdict,
            explainer=explainer, top_k=top_k,
        )
        watcher.start()
        app._watcher = watcher

    # ── routes ────────────────────────────────────────────────────────────────

    @app.route("/")
    def dashboard():
        counts = state.counts()
        recent = state.recent(10)
        return render_template(
            "dashboard.html",
            counts=counts,
            recent=recent,
            watch_folder=watch_folder,
            model_loaded=model is not None,
        )

    @app.route("/scan", methods=["GET", "POST"])
    def scan_file():
        if request.method == "GET":
            return render_template(
                "scan.html",
                model_loaded=model is not None,
                error=None,
            )

        if model is None:
            return render_template(
                "scan.html",
                model_loaded=False,
                error="No model loaded — start the app with --model <path>.",
            )

        f = request.files.get("file")
        if f is None or f.filename == "":
            return render_template(
                "scan.html",
                model_loaded=True,
                error="No file selected.",
            )

        # Save to a temp dir with the original name so Verdict.filename is readable
        tmpdir = Path(tempfile.mkdtemp())
        dest = tmpdir / Path(f.filename).name
        try:
            f.save(dest)
            from lightguard.monitor.scan import scan
            verdict = scan(dest, model, explainer=explainer, top_k=top_k)
        finally:
            dest.unlink(missing_ok=True)
            tmpdir.rmdir()

        entry = ScanEntry(
            id=str(uuid.uuid4()),
            verdict=verdict,
            scanned_at=datetime.now(),
            source="manual",
        )
        state.add_scan(entry)
        return redirect(url_for("result", scan_id=entry.id))

    @app.route("/result/<scan_id>")
    def result(scan_id: str):
        entry = state.get(scan_id)
        if entry is None:
            abort(404)
        reasons = _parse_reasons(entry.verdict.reasons)
        return render_template("result.html", entry=entry, reasons=reasons)

    @app.route("/feed")
    def feed():
        recent = state.recent(30)
        return render_template(
            "feed.html",
            recent=recent,
            watch_folder=watch_folder,
            model_loaded=model is not None,
        )

    # ── JSON / SSE API ────────────────────────────────────────────────────────

    @app.route("/api/recent")
    def api_recent():
        data = [
            {
                "id": e.id,
                "filename": e.verdict.filename,
                "label": e.verdict.label,
                "risk_score": e.verdict.risk_score,
                "confidence": e.verdict.confidence,
                "raw_prob": round(e.verdict.raw_prob, 4),
                "scanned_at": e.scanned_at.isoformat(),
                "source": e.source,
            }
            for e in state.recent(20)
        ]
        return jsonify(data)

    @app.route("/api/events")
    def api_events():
        """Server-Sent Events stream; one JSON object per new scan."""
        sub_queue = state.subscribe()

        def _stream():
            # Immediate greeting so the client confirms the connection opened
            yield 'data: {"type":"connected"}\n\n'
            try:
                while True:
                    try:
                        entry = sub_queue.get(timeout=25)
                        payload = json.dumps({
                            "type": "scan",
                            "id": entry.id,
                            "filename": entry.verdict.filename,
                            "label": entry.verdict.label,
                            "risk_score": entry.verdict.risk_score,
                            "confidence": entry.verdict.confidence,
                            "scanned_at": entry.scanned_at.isoformat(),
                        })
                        yield f"data: {payload}\n\n"
                    except queue.Empty:
                        yield "data: ping\n\n"
            except GeneratorExit:
                pass
            finally:
                state.unsubscribe(sub_queue)

        return Response(
            _stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


# ── helpers ───────────────────────────────────────────────────────────────────

_REASON_RE = re.compile(
    r"^(High-risk|Low-risk) indicator \(SHAP ([+-][\d.]+)\): (.+)$"
)


def _parse_reasons(reasons: tuple[str, ...]) -> list[dict]:
    """Convert translate() sentences into template-ready dicts with bar widths.

    Each dict has:
      direction  — "High-risk" | "Low-risk"
      shap       — float (signed)
      desc       — plain-English description string
      width      — int 0-100, proportional to |shap| within this verdict
      severity   — "bad" (high-risk) | "ok" (low-risk)
    """
    parsed: list[dict] = []
    for r in reasons:
        m = _REASON_RE.match(r)
        if m:
            direction, shap_str, desc = m.group(1), m.group(2), m.group(3)
            parsed.append({"direction": direction, "shap": float(shap_str), "desc": desc})
        else:
            parsed.append({"direction": "Low-risk", "shap": 0.0, "desc": r})

    if parsed:
        max_abs = max(abs(p["shap"]) for p in parsed) or 1.0
        for p in parsed:
            p["width"] = round(abs(p["shap"]) / max_abs * 100)
            p["severity"] = "bad" if p["direction"] == "High-risk" else "ok"

    return parsed
