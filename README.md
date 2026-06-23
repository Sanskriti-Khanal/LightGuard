# LightGuard

Offline, explainable AI malware detector for home Windows users.

A LightGBM classifier trained on EMBER2024 PE features scores any Windows
executable with a risk score. SHAP values translate the model's reasoning into
plain English. A watchdog monitors your Downloads folder and a local Flask UI
shows verdicts — no cloud calls, no data leaves your machine at inference time.

---

## Storage-split architecture

| Task | Where it runs |
|---|---|
| Download EMBER2024, vectorize, train, generate SHAP plots | **Google Colab** |
| Serve UI, run watchdog, scan files, run tests | **Local machine** |

The local machine never needs the raw dataset (~100 GB). You download/train in
Colab, then copy the saved model and reports locally.

---

## Architecture

```
src/lightguard/
├── data/      — feature I/O helpers (thrember wrappers + local loader)
├── malware/   — LightGBM classifier: train (Colab), predict (local)
├── explain/   — SHAP explainer: compute + render plain-English summaries
├── monitor/   — Watchdog filesystem watcher (Downloads folder)
├── network/   — Reserved for v2 per-process network anomaly module
└── ui/        — Flask local web UI (verdicts + explanations)
```

All settings live in `config.yaml`. No hardcoded paths or magic numbers.

---

## Local setup

```bash
# 1. Clone and create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install thrember (not on PyPI)
git clone https://github.com/FutureComputing4AI/EMBER2024.git
pip install ./EMBER2024

# 3. Install remaining dependencies
pip install -r requirements.txt

# macOS ARM only — LightGBM needs OpenMP:
brew install libomp
```

---

## Colab workflow (one-time, run in Google Colab)

Open `notebooks/01_train.ipynb` in Colab and run all cells. It will:
1. Download the EMBER2024 PE dataset via `thrember.download_dataset`
2. Vectorize features via `thrember.create_vectorized_features`
3. Train the LightGBM classifier
4. Evaluate on the test and challenge sets
5. Generate SHAP summary plots
6. Save `lightguard_lgbm.pkl` — **download this to `models/` on your laptop**
7. Save reports — **download these to `reports/` on your laptop**

---

## Local usage (after Colab training)

### Scan a file

```bash
python scripts/scan.py path/to/file.exe
```

### Start the Downloads-folder watcher

```bash
python scripts/watch.py
```

### Launch the local web UI

```bash
flask --app src/lightguard/ui run
# Open http://127.0.0.1:5000
```

### Run tests

```bash
pytest tests/
```

---

## Data & model policy

- `data/raw/`, `data/processed/`, `data/external/`, and `models/` are git-ignored.
- `data/sample/` (tiny committed test fixtures) and `reports/` (committed evidence) are the exceptions.
- See `CLAUDE.md` for the full set of non-negotiable development rules.
