# NIOR-AI: On-Device Semantic Annotation of Live Web Pages Without DOM Interference

Reference implementation for the paper **"NIOR-AI: On-Device Semantic Annotation of Live Web Pages Without DOM Interference"** — a Manifest V3 browser extension that classifies and overlays semantic regions using a sub-5 MB on-device XGBoost/ONNX classifier confined to a closed Shadow DOM renderer.

---

## Prerequisites

- **Node.js** ≥ 18
- **Python** ≥ 3.10 (for training / feature extraction only)
- **Chrome** ≥ 110 or **Edge** ≥ 110 (Manifest V3)

---

## Load the Extension

1. Open `chrome://extensions` (or `edge://extensions`)
2. Enable **Developer mode**
3. Click **Load unpacked** → select this directory (`nior-ai-extension/`)

The extension activates on all URLs. The ONNX classifier is loaded inside an offscreen document; no page content is sent to any external server.

> **Note:** place your compiled `nior_classifier.onnx` in `wasm/` before loading (see [Train the Classifier](#train-the-classifier) below). Without it the offscreen worker starts but skips inference.

---

## Run the Unit Tests

The two test suites reproduce the correctness properties proved in the paper.

### Test 1 — Feature Extractor (Section V-B, Table II)

Verifies that `extractFeatures()` produces a 72-dimensional `Float32Array` with correct one-hot encoding, geometric normalisation, and keyword indicators.

```bash
node --experimental-vm-modules test/test_features.js
```

### Test 2 — Masked Redraw Algorithm (Section VII, Propositions 2 & 3)

Verifies Algorithm 1: no ghost artefacts (old BCR included in dirty region), no overdraw outside the dirty region (clip precedes clearRect), dirty-flag lifecycle, and Phase-1 BCR-batch ordering.

```bash
node --experimental-vm-modules test/test_masked_redraw.js
```

### Run Both Tests

```bash
npm test
```

Expected output:

```
NIOR-AI Feature Extractor Tests

  ✓ output is Float32Array of length 72
  ✓ tag one-hot: nav → dim 0 = 1
  ...
12 passed, 0 failed

NIOR-AI Masked Redraw Algorithm Tests

  ✓ Proposition 2: old BCR included in dirty region (no ghost artefacts)
  ✓ Proposition 3: clip() called before clearRect (no overdraw outside dirty region)
  ...
6 passed, 0 failed
```

---

## Train the Classifier

The training pipeline reproduces the XGBoost model reported in Section VI-C (macro-F1 = 0.91, < 1 ms/element, < 5 MB ONNX).

### 1. Install Python dependencies

```bash
pip install -r train/requirements.txt
playwright install chromium
```

### 2. Extract features from labelled pages

```bash


```

`labels.jsonl` format — one JSON object per line:

```json
{"url": "https://example.com", "selector": "#hero", "label": "call_to_action"}
```

Supported labels: `main_content`, `navigation`, `call_to_action`, `form`, `heading`, `sidebar`, `media`, `advertisement`.

### 3. Train and export to ONNX

```bash
python train/train.py \
    --data   data/webui_augmented.csv \
    --output wasm/nior_classifier.onnx \
    --trials 50
```

`--trials` controls Optuna hyperparameter search iterations (paper used 50). The script prints a `classification_report` and writes the ONNX model to `wasm/`.

### 4. Reload the extension

After placing `nior_classifier.onnx` in `wasm/`, click **Reload** on the extension card in `chrome://extensions`.

---

## Repository Layout

```
nior-ai-extension/
├── manifest.json
├── src/
│   ├── background/        # Manifest V3 service worker
│   ├── classifier/        # features.js — 72-dim feature extractor
│   ├── content/           # content.js — DOM observer, BCR collection
│   ├── offscreen/         # ONNX Runtime Web inference worker
│   ├── shadow-renderer/   # Closed Shadow DOM overlay renderer
│   └── annotation-api/    # Public JS API for external callers
├── test/
│   ├── test_features.js       # Feature extractor unit tests
│   └── test_masked_redraw.js  # Masked redraw algorithm unit tests
├── train/
│   ├── extract_features.py    # Playwright-based feature extraction
│   ├── train.py               # XGBoost training + ONNX export
│   └── requirements.txt
└── wasm/                  # ONNX Runtime Web + compiled classifier
```
