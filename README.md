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

## Visuals

Model architecture and workflow diagrams (see repo root):

![EEG to Text Model Overview](./eeg-to-text-translation-a-model-for-deciphering-human-brain-activity-2.png)

![EEG2Text: Open-Vocabulary / Pretraining](./eeg2text-open-vocabulary-eeg-to-text-decoding-with-eeg-pre-training-and-multi-view-transformer-1.png)

---

## Contributing, license & citations

Please add a `LICENSE` file and include proper citations for ZuCo and related papers when using this dataset. For contributions, follow the CI checks in `.github/workflows/ci.yml`.

---

## Questions or next steps

To run a full demo on your machine or cloud, follow `DESIGN.md`. I can also:
- generate a high-fidelity PNG architecture diagram, or  
- create Terraform templates for S3 + IAM + EC2 deployment.

Tell me which and I’ll add it.
# Neurolinked-2
