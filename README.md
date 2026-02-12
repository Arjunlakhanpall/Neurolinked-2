# 🧠 Neurolinked: Mind-to-Script (EEG → Text)

**Neurolinked** (a.k.a. *Mind-to-Script*) is an end-to-end research and deployment toolkit that translates non-invasive EEG recordings into natural-language text. By bridging neural oscillations with Transformer-based decoders, it provides a reproducible pipeline for brain-to-text synthesis.

---

## 🚀 Key Features

* **Signal Processing:** MNE-based artifact removal and automated signal cleaning.
* **Neural Bridge:** A trainable **CNN + BiLSTM** projection layer that maps EEG features into the **BART** embedding space.
* **Data Ops:** High-performance dataset sharding (ZuCo-ready) for scalable supervised training.
* **Inference API:** GPU-ready FastAPI service with **SQI (Signal Quality Index)** guards and Prometheus metrics.

---

## 🛠️ Installation & Setup

### 1. Environment Initialization

```bash
# Create and activate a clean environment
python -m venv .venv
source .venv/bin/activate  # Unix/macOS
# .\.venv\Scripts\Activate.ps1 # Windows

# Install core dependencies
pip install -r requirements-ci.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.0+cpu
```

### 2. Validation

```bash
# Run the test suite to ensure environment integrity
pytest -q
```

---

## 🏗️ The Pipeline

The architecture is designed to handle the high noise-to-signal ratio of EEG data while utilizing the linguistic power of pretrained Large Language Models.

### Core Workflow

1. **Ingestion:** `scripts/load_zuco.py` converts raw brainwaves into canonical data formats.
2. **Sharding:** `scripts/build_manifest_and_shards.py` prepares data for high-throughput training.
3. **Training:** `scripts/train_from_shards.py` aligns the EEG encoder with the language decoder.

---

## 🌐 Deployment & API

Neurolinked is production-ready via a Dockerized FastAPI service.

### Launch Local Inference Server

```powershell
# Set model directory and launch
$env:MODEL_DIR = "./models"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Primary Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/infer` | `POST` | Translates raw EEG signals to text. |
| `/metrics` | `GET` | Prometheus metrics (Latency, SQI, GPU memory). |
| `/history` | `GET` | Recent inference logs for drift monitoring. |

---

## 🩺 Safety & Observability

* **SQI Hard Reject:** Signals with a Quality Index below **0.25** are automatically rejected to prevent hallucination.
* **DB Persistence:** All inference results are stored in a relational database for post-hoc research auditing.
* **Cloud Ready:** Native support for S3 artifact versioning and IAM instance roles.

---

## 📂 Repository Contents

* `app/`: FastAPI service, model loader, and DB integration.
* `scripts/`: Preprocessing, sharding, and training logic.
* `notebooks/`: End-to-end synthetic demos for rapid prototyping.
* `DESIGN.md`: Full architecture runbook and cloud deployment guide.

---
## Contributing, license & citations

Please add a `LICENSE` file and include proper citations for ZuCo and related papers when using this dataset. For contributions, follow the CI checks in `.github/workflows/ci.yml`.

---

## Questions or next steps

To run a full demo on your machine or cloud, follow `DESIGN.md`. I can also:
- generate a high-fidelity PNG architecture diagram, or  
- create Terraform templates for S3 + IAM + EC2 deployment.

Tell me which and I’ll add it.
 
---

## Folder reference & implementation guide

This section describes every top-level folder, what it contains, and step-by-step notes to implement and validate each component.

- `app/`
  - Contains the FastAPI service and model runtime.
  - Key files:
    - `main.py` — endpoints (/infer, /metrics, /history), startup model loader.
    - `model.py` — EEGEncoder (1D-CNN + BiLSTM), MindToScriptModel wrapper, SQI logic.
    - `db.py` — SQLAlchemy persistence helpers.
  - How to validate:
    1. Ensure `models/bridge.pt` exists or set `MODEL_S3_URI` to a versioned S3 prefix.
    2. Start server: `uvicorn app.main:app --reload` and open `/docs`.
    3. POST a small JSON test to `/infer` or run `scripts/infer_from_canonical.py`.

- `scripts/`
  - Data engineering, training, and helper scripts.
  - Notable scripts:
    - `load_zuco.py` — canonicalizes ZuCo .mat → .pkl
    - `align_zuco.py` — extracts word onsets/offsets
    - `build_manifest_and_shards.py` — creates train/val/test .pt shards
    - `train_from_shards.py` — training loop for bridge (encoder + projection)
    - `infer_from_canonical.py` — run inference on a canonical epoch (demo)
    - `summarize_canonical.py`, `export_epochs_csv.py` — dataset summaries/exports
  - How to validate:
    1. Place canonical `.pkl` in `data/canonical/`.
    2. Run `python scripts/summarize_canonical.py` and inspect `data/manifest.csv`.
    3. Build shards and run a short training (`--epochs 2`) to verify end-to-end.

- `data/`
  - Holds canonical pickles, shards, and generated manifests (do not commit raw large files).
  - Structure:
    - `canonical/` — compact per-file .pkl artifacts
    - `shards/<version>/{train,val,test}/` — .pt shard files
    - `manifest.csv`, `epochs.csv` — generated summaries
  - How to validate:
    - Open `data/manifest.csv` and sample a `data/canonical/*.pkl` with `scripts/inspect_zuco.py`.

- `models/`
  - Local model artifacts (gitignored in production).
  - Expected contents:
    - `bridge.pt` — checkpoint with encoder & projection state_dict + config
    - optional `decoder/` — HF-style decoder folder (if stored locally)
  - How to validate:
    - Start the server and confirm startup logs show "Found model artifact" and "Model loaded successfully."

- `notebooks/`
  - Demo notebooks for synthetic end-to-end flow.
  - How to validate:
    - Run `jupyter notebook` and execute `end_to_end_demo.ipynb`.

- `docker/` or root Dockerfile & `docker-compose.yml`
  - Docker config for GPU/CPU deployment and a one-shot `example` service.
  - How to validate:
    - `docker compose build` and `docker compose up web`, then query `/metrics`.

- `.github/`
  - Contains CI workflows (unit tests).
  - How to validate:
    - Push to GitHub and check Actions tab for CI status.

---

## Research & references (select)

- ZuCo 2.0 dataset (ZuCo project page / OSF): https://osf.io/2urht/
- Hollenstein et al., ZuCo 2.0 — dataset and benchmarks (see above)
- Suggested reading on EEG→Text and multimodal alignment:
  - Works on EEG-to-text decoding, contrastive pretraining, and multimodal adapters (cite in your papers as appropriate).

If you want, I can add clickable DOI/URLs for the specific papers you reference; paste them or I will add the ones I know.

---

## Flowcharts (ASCII + image)

Quick ASCII flowchart:

```
Client (EEG device) -> API (/infer) -> Preprocess (MNE: filter, ICA) -> Epoching -> Encoder (CNN+BiLSTM)
    -> Projection -> Decoder (BART) -> Response {text, confidence, meta}
    -> Metrics -> Prometheus / DB -> Monitoring/Alerts
```

Visual assets (already included) show the model internals. If you want a high-fidelity PNG flowchart, I can generate one and add it to the repo.
