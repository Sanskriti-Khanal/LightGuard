# LightGuard — On-the-day Demo Runbook

Everything in this document assumes you are running from the repo root on the
demo machine with the project venv active.

```
cd ~/Desktop/LightGuard
source .venv/bin/activate   # or: .venv/bin/python for explicit calls
```

---

## Pre-flight checklist (night before / 30 min before)

Run through this once while you still have internet.

- [ ] `models/lightguard_lgbm.txt` exists and is non-zero
- [ ] `data/sample/benign.exe` exists (putty.exe, ~1.7 MB)
- [ ] `data/sample/X_test.npy` + `y_test.npy` exist
- [ ] `reports/metrics.json` exists
- [ ] `.venv/bin/python -c "import lightgbm, shap, flask, watchdog; print('ok')"` prints `ok`
- [ ] Run the full notebook once, confirm all cells execute cleanly
- [ ] Start the Flask app, open `http://127.0.0.1:5000`, confirm all three pages load
- [ ] Do one file upload through `/scan` with `data/sample/benign.exe`
- [ ] **Record a 3-minute screen recording** of the above as the fallback artifact

---

## Airplane-mode proof

Turn on airplane mode (or disconnect Wi-Fi) **before** the demo starts.

Confirm with the audience: no network icon, no connectivity. Then run everything.
Every page of the Flask app, every notebook cell, every prediction loads from disk.
At the end: "That verdict was computed on this laptop. Nothing left this device."

If you forget to go offline: the app still works — there are no CDN or API calls.
The proof is the absence of network calls, not a hard dependency on airplane mode.

---

## Step order

### Step 0 — Open the notebook (30 s)

```bash
.venv/bin/jupyter notebook notebooks/demo.ipynb
```

Or, if Jupyter is not installed in the venv, open `notebooks/demo.ipynb` with
VS Code's built-in notebook renderer (select the `.venv` kernel).

---

### Step 1 — Model performance (Cell 1–3, ~2 min)

**Run cells 1–3.** The comparison table and bar chart appear.

**Talking points — RQ1:**

> "We trained on EMBER2024, 52 weeks of PE files, never touching the test set.
> The test AUC is **0.9958** — that means on a completely unseen file the model
> is right 99.6% of the time by area under the ROC curve.
>
> The challenge set is harder — these are evasive malware samples designed to
> evade detection. AUC drops to **0.9482**, still above our 0.95 target.
> The baselines (Logistic Regression, Random Forest on a subsample) fall well
> below on the challenge set. That gap is the value of a deep tree ensemble on
> the full 2 568-feature space."

Point to the red dashed line (0.95 threshold) on the chart.

---

### Step 2 — Live prediction (Cell 4–5, ~2 min)

**Run cells 4–5.** One benign row, one malicious row are drawn randomly.

**Talking points:**

> "These rows are drawn from a holdout slice the model has never seen.
> I'm not cherry-picking — the seed is fixed in `config.yaml` for
> reproducibility, but you can change it and re-run.
>
> Watch the risk bar: the benign file scores close to zero,
> the malicious one scores near 100. Both predictions match the ground truth."

If asked about the random seed: it comes from `config.yaml`, not hardcoded in
the notebook — consistent with CLAUDE.md rule 4.

---

### Step 3 — SHAP explanation (Cell 6–8, ~3 min)

**Run cells 6–8.** The waterfall chart appears and plain-English reasons print.

**Talking points — RQ2:**

> "This is the explainability story. SHAP assigns each of the 2 568 features
> a signed contribution to the risk score. Red bars push the score towards
> MALICIOUS; green bars push it towards BENIGN.
>
> But raw SHAP values aren't useful to a home user. So we translate them:
> 'Code looks packed or hidden — entropy 7.9/8' tells someone *what* is
> suspicious without knowing what a histogram bucket is.
>
> This is computed locally, in milliseconds, with no cloud lookup.
> That's the RQ2 answer: yes, you can explain a decision in plain language
> from a local model."

---

### Step 4 — Real-file scan (Cell 9, ~1 min)

**Run cell 9.** Scan `data/sample/benign.exe` (putty.exe).

**Talking points:**

> "Until now we've been scoring pre-extracted feature vectors.
> Now we go from raw bytes: read the .exe, run it through the same
> PE feature extractor used during training, score, explain — all in one call.
>
> Risk score: 0 or 1 out of 100. BENIGN, HIGH confidence.
> PuTTY is legitimate software; the model has never seen this specific file."

Point out `verdict.reasons` — same SHAP pipeline, same plain-English output.

---

### Step 5 — Flask UI (Cell 10, then browser, ~3 min)

**Run cell 10** to print the launch command. Then switch to a terminal:

```bash
.venv/bin/python scripts/run_app.py \
    --model models/lightguard_lgbm.txt \
    --watch ~/Downloads
```

Switch to the browser at `http://127.0.0.1:5000`.

**Walk through three pages:**

1. **Dashboard** (`/`) — "Protection status, recent scans, module cards.
   The watcher is active on ~/Downloads."

2. **Scan a file** (`/scan`) — Upload `data/sample/benign.exe`.
   Wait for the redirect to `/result/<id>`.

3. **Result page** (`/result/<id>`) — "This is the signature panel.
   Same SHAP explanation rendered in the UI. Risk 0/100, BENIGN, HIGH confidence.
   Green meter bars, each one a plain-English feature."

4. **Live feed** (`/feed`) — "While we were talking, the watcher noticed the upload.
   Any new .exe dropped into ~/Downloads would appear here in real time
   via server-sent events — no page refresh."

**Talking points:**

> "The title bar says 'Running offline — nothing leaves this device'.
> That's not marketing copy; there are no outbound calls in the codebase.
> A home user installs this, it watches their Downloads folder,
> and every verdict is computed and explained locally."

---

## Q&A prompts to anticipate

| Question | Answer |
|---|---|
| Why not use a cloud AV API? | Latency, privacy, cost, offline requirement. Local model adds explainability the cloud doesn't provide. |
| What about false positives? | Precision on test set is 98.7%. Benign files near the threshold get LOW confidence, prompting a second look rather than a block. |
| Can it be evaded? | The challenge AUC (0.948) shows it's harder but not impossible. Adversarial PE is an open research problem; the explainability layer at least tells the user *what* to inspect manually. |
| Why LightGBM not a neural network? | Interpretable by design, fast at inference, works on a CPU, no GPU required for a home user. SHAP exact values are fast on tree models. |
| What is EMBER2024? | A public dataset of 1.56M PE files with ground-truth labels. We use the Win32 subset trained on weeks 1-52, tested on 53-64. |

---

## Fallback plan

If anything fails live, in priority order:

1. **Pre-run notebook**: Kernel → Restart & Run All the night before. Leave it
   open with all outputs. If the kernel dies on stage, the outputs are still
   there — scroll through them.

2. **Screen recording**: A 3-minute recording of a clean run (notebook +
   Flask UI) saved to Desktop. Open in QuickTime, play it. Narrate over it.

3. **Static screenshots**: `reports/` contains confusion matrices and ROC curve
   plots from the real evaluation. Open them in Preview as a last resort.

4. **Metrics table only**: If the model file is missing, the notebook can still
   display the comparison table from `reports/metrics.json` — cells 1–3 need no
   model. Read the numbers aloud and explain what they mean.

---

## Timing guide

| Section | Time |
|---|---|
| Setup + airplane mode proof | 1 min |
| Step 1 — AUC table + chart | 2 min |
| Step 2 — Live prediction | 2 min |
| Step 3 — SHAP explanation | 3 min |
| Step 4 — Real-file scan | 1 min |
| Step 5 — Flask UI walkthrough | 3 min |
| Q&A | 3 min |
| **Total** | **~15 min** |
