# LightGuard — Standing Project Brief for Claude Code

Read this before writing any code, script, or test. These rules are non-negotiable.

---

## What LightGuard is

An offline, explainable AI malware detector for home Windows users.

- **Classifier:** LightGBM trained on EMBER2024 PE features via the `thrember` package.
- **Explainability:** SHAP values rendered as plain-English summaries.
- **Real-file pipeline:** extracts features from PE files and scores them.
- **Monitor:** watchdog watches a configured folder (default: Downloads).
- **UI:** local Flask app shows verdicts with explanations.
- **v1 scope:** malware module only. `src/lightguard/network/` is an empty placeholder for v2.

All configuration lives in `config.yaml`. No hardcoded paths or magic numbers anywhere.

---

## Non-negotiable rules

### 1. DATA NEVER ENTERS GIT

`.gitignore` excludes:
- `data/raw/` — EMBER2024 raw download
- `data/processed/` — vectorized feature arrays
- `data/external/` — third-party reference data
- `models/` — saved model artefacts

**Committed:**
- `data/sample/` — tiny real-shaped fixture for tests
- `reports/` — metrics JSON, confusion matrices, SHAP plots (the evidence)

Never add exceptions to `.gitignore` for these directories without explicit user approval.

### 2. STORAGE-SPLIT ARCHITECTURE

**Heavy work runs in Google Colab, not on the local machine.**

| Step | Where |
|---|---|
| Download EMBER2024 (`thrember.download_dataset`) | Colab |
| Vectorize features (`thrember.create_vectorized_features`) | Colab |
| Train LightGBM (`scripts/train.py`) | Colab |
| Generate SHAP report plots | Colab |
| Final evaluation on test/challenge sets | Colab |
| Serve Flask UI, run watchdog, scan files | **Local machine** |
| Run unit tests against `data/sample/` | **Local machine** |

`data/raw/` and `data/processed/` are **never expected to exist locally**. Do not
write code that assumes they are present during scanning or UI operation. The
local machine holds only:
- `models/lightguard_lgbm.pkl` (exported from Colab)
- `data/sample/` (committed test fixtures)
- `reports/` (committed metrics + plots, exported from Colab)
- A benign `.exe` for smoke-testing the real-file scan pipeline

### 3. RESPECT THE TIME-BASED SPLIT

EMBER2024 canonical split:
- **Train:** weeks 1–52
- **Test:** weeks 53–64
- **Challenge set:** evasive malware (separate)

Rules:
- Train and cross-validate **only within the training set**.
- Test and challenge sets are touched **only for final evaluation**.
- **Never shuffle rows across the time boundary.**
- `val_fraction` in `config.yaml` takes the last `val_fraction` of training rows
  as validation — temporal order is preserved, no random shuffle.

### 4. REPRODUCIBILITY

All random seeds come from `config.yaml` (`random_seed`). Never call
`random.seed()`, `np.random.seed()`, or pass a literal integer to any stochastic
function. Always read the seed from the loaded config dict.

### 5. SAFETY

- This repo processes **pre-extracted feature vectors** and **known-benign executables only**.
- **Never** download, generate, request, store, or handle live malware samples.
- The real-file scan pipeline is demonstrated on benign files only.
- `thrember.download_dataset` / `download_models` calls belong in Colab notebooks, not in
  any code that runs locally.

### 6. OFFLINE AT INFERENCE

No network calls during scanning or UI operation. Everything needed for inference
(model, feature extractor) must already be on disk. Dataset and model downloads
happen once, in Colab.

### 7. MODULAR + TESTED

- Every module in `src/lightguard/` is independently testable against `data/sample/`.
- All public functions have type hints and a one-line docstring (the *why*, not
  the *what* — well-named identifiers handle the what).
- Keep functions small. Prefer composition over monolithic functions.
- Tests live in `tests/` and run with `pytest`. No test may touch `data/raw/`,
  `data/processed/`, or `models/`.

---

## Key external references

- **EMBER2024 repo:** `FutureComputing4AI/EMBER2024`
  - Canonical API examples: `examples/train_lgbm.py`, `examples/eval_lgbm.py`
  - Use these for correct `thrember` function signatures. Do not invent signatures.
- **thrember functions:** `download_dataset`, `download_models`,
  `create_vectorized_features`, `read_vectorized_features`, `read_metadata`
- **thrember notes:**
  - Not on PyPI — install from the cloned repo (`pip install ./EMBER2024`)
  - `signify==0.7.1` required (0.8+ breaks the import)
  - macOS ARM: `brew install libomp` required for LightGBM
  - `create_vectorized_features(data_dir)` reads raw files AND writes `.dat` arrays
    into the **same** directory — there is no separate output path
  - `read_metadata(data_dir)` returns a **tuple** of three polars DataFrames:
    `(train_df, test_df, challenge_df)` — use `.is_in()`, not `.isin()`

---

## Directory map

```
lightguard/
├── CLAUDE.md              ← this file
├── README.md
├── config.yaml            ← single source of truth for all settings
├── requirements.txt       ← pinned versions + install notes
├── .gitignore
├── data/
│   ├── raw/               ← NOT committed, NOT expected locally
│   ├── processed/         ← NOT committed, NOT expected locally
│   ├── external/          ← NOT committed
│   └── sample/            ← IS committed (test fixtures)
├── models/                ← NOT committed; populate by downloading from Colab output
├── reports/               ← IS committed (metrics + plots from Colab)
├── notebooks/             ← Colab notebooks (IS committed)
├── scripts/               ← training/eval scripts that run in Colab (IS committed)
├── tests/                 ← pytest suite, runs locally against data/sample/
└── src/lightguard/
    ├── data/              ← feature I/O wrappers (used by Colab scripts + local scan)
    ├── malware/           ← LightGBM classifier: train (Colab), predict (local)
    ├── explain/           ← SHAP explainer + plain-English renderer
    ├── monitor/           ← watchdog filesystem watcher
    ├── network/           ← PLACEHOLDER — reserved for v2 network anomaly module
    └── ui/                ← Flask local web UI
```
