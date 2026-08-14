#!/usr/bin/env python
"""Start the LightGuard local web UI.

The app runs fully offline on localhost. All ML inference stays on this
machine — no data leaves the device.

Quick start (model already in models/)::

    python scripts/run_app.py --model models/lightguard_lgbm.txt

With automatic folder watching::

    python scripts/run_app.py --model models/lightguard_lgbm.txt --watch ~/Downloads

Without a model (read-only history view)::

    python scripts/run_app.py

Flags::

    --model   PATH    Path to the trained LightGBM .txt model file.
    --watch   DIR     Folder to monitor for new PE files (requires --model).
    --host    HOST    Bind address (default: 127.0.0.1).
    --port    PORT    Port (default: 5000).
    --no-explain      Skip building the SHAP explainer (faster startup, no reasons).
    --top-k   N       Number of SHAP features shown per verdict (default: 5).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the src/ package tree is importable when running from the repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LightGuard local web UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model",      metavar="PATH", help="LightGBM .txt model file")
    p.add_argument("--watch",      metavar="DIR",  help="Folder to watch for new PE files")
    p.add_argument("--host",       default="127.0.0.1")
    p.add_argument("--port",       type=int, default=5000)
    p.add_argument("--no-explain", action="store_true", help="Disable SHAP explanations")
    p.add_argument("--top-k",      type=int, default=5, dest="top_k",
                   help="SHAP features shown per verdict (default 5)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    model     = None
    explainer = None

    # ── load model ────────────────────────────────────────────────────────────
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"[error] Model file not found: {model_path}", file=sys.stderr)
            sys.exit(1)
        print(f"[lightguard] Loading model from {model_path} …")
        from lightguard.malware.train import load_model
        model = load_model(model_path)
        print("[lightguard] Model ready.")
    else:
        print("[lightguard] No model specified — UI will show history only.")
        print("             Pass --model models/lightguard_lgbm.txt to enable scanning.")

    # ── build SHAP explainer ──────────────────────────────────────────────────
    if model is not None and not args.no_explain:
        print("[lightguard] Building SHAP explainer from data/sample/ …")
        from lightguard.explain.explainer import build_explainer, load_background
        sample_dir = _REPO_ROOT / "data" / "sample"
        background = load_background(sample_dir, n=200)
        if background is None:
            print("[lightguard] Warning: no background sample found — explanations disabled.")
            print("             Expected data/sample/X_test.npy or data/sample/test_holdout_X.npy")
        else:
            explainer = build_explainer(model, background)
            print(f"[lightguard] Explainer ready (background: {len(background)} samples).")

    # ── create app ────────────────────────────────────────────────────────────
    from lightguard.ui.app import create_app
    app = create_app(
        model=model,
        explainer=explainer,
        watch_folder=args.watch,
        top_k=args.top_k,
    )

    # ── start ─────────────────────────────────────────────────────────────────
    url = f"http://{args.host}:{args.port}"
    print(f"\n[lightguard] Starting on {url}")
    print(f"[lightguard] Open {url} in your browser.")
    print("[lightguard] Press Ctrl+C to stop.\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
