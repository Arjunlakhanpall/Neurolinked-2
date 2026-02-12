# Mind-to-Script — Design Document (Revised)

Version: 2026-02-11 (revised)

Purpose
-------
Mind-to-Script is a non-invasive Brain-Computer Interface (BCI) system that decodes multi-channel EEG recordings into English text. The project is designed end-to-end: data ingestion, preprocessing, training a cross-modal "bridge" model (EEG → text), and a production-ready inference service optimized for AWS (S3, EC2 GPU, IAM, CloudWatch).

High-level goals
- Reproducible preprocessing for ZuCo 2.0 and similar EEG corpora.
- Train a bridge (1D-CNN + BiLSTM → projection) that feeds a frozen or lightly fine-tuned BART decoder.
- Production deployment with Docker, S3 model versioning, IAM-based credentials, monitoring, and a robust CI pipeline.

System Overview (ASCII diagram)
--------------------------------
Client (web/mobile)                API / FastAPI                    Inference (EC2 GPU)
-------------------              -----------------               ------------------------
 EEG capture device  --->  POST /infer JSON / npy  --->  Preprocess (MNE)   --->  Encoder (1D-CNN+BiLSTM)
       or file                 (API Gateway)             SQI check, epoching         Projection -> d_model
                                  |                             |                        |
                                  v                             v                        v
                              Auth/IAM                    SQI + metadata           Decoder (BART) -> text
                              Rate-limit                                                 |
                                  |                                                      v
                                  v                                              Response JSON {text, confidence, meta}
                               S3 <--- model weights & dataset artifacts (versioned: /v1.0.0/...)

Notes:
- All artifacts (datasets, shards, bridge checkpoints, decoder folders) are stored on S3 under versioned prefixes. Local disk holds transient data only.

Core Components
---------------
1) Data & Storage
- Raw: ZuCo .mat / .edf files (large). Store originals on S3: s3://bucket/mindtoscript/raw/v1.0.0/...
- Canonicalized pickles (.pkl): store per-file compact representation with keys: signals (n_ch, n_samples), sfreq, ch_names, sentences (words, onsets, offsets), meta.
- Sharded training data: PyTorch .pt shards (list of small dicts) stored under s3://bucket/mindtoscript/dataset/v1.0.0/{train,val,test}/
- Models: s3://bucket/mindtoscript/models/v1.0.0/{bridge.pt, decoder/}

2) Preprocessing
- Implemented in scripts/ (MNE-based):
  - Band-pass 0.5–50 Hz
  - ICA artifact removal (EOG/EMG)
  - Z-score normalization per channel
  - Per-word alignment (onset/offset -> sample indices)
  - Optional: Wavelet / STFT features or bandpower extraction

3) Model (Bridge)
- Encoder: 1D-CNN layers → BiLSTM (bidirectional) to produce sequence features
- Projection: linear layer maps encoder output per time-step to decoder hidden size (d_model)
- Decoder: BART-base (HF). Strategy: freeze lower layers, fine-tune top layers or train only projection/adapter.

4) Training
- Use shard-based data loading. Train on g4dn.xlarge or larger, preferring p3/p4 for substantial fine-tuning.
- Checkpointing: save bridge.pt with state_dicts and a small config object (in_channels, cnn_channels, lstm_hidden, d_model).

5) Serving & API
- FastAPI app (app/main.py)
- Endpoints:
  - POST /infer: accepts JSON signals or file upload, returns {text, confidence, meta}
  - GET /metrics: Prometheus metrics
  - GET /history: recent inferences from DB
- Robustness:
  - SQI (Signal Quality Index) computed for each request
  - Hard rejection if SQI < 0.25 or >20% channels exceed ±100 µV (HTTP 422)
  - Soft warning if 0.25 ≤ SQI < 0.5 (meta.is_unreliable)
  - Use torch.inference_mode() and torch.cuda.empty_cache() after generation

6) Database & Observability
- DB: SQLAlchemy-backed relational DB with InferenceResult table (default SQLite for dev, PostgreSQL for production).
- Metrics: Prometheus client exposing:
  - eeg_inference_latency_seconds (Histogram)
  - eeg_sqi_score (Gauge)
  - gpu_memory_used_bytes (Gauge)
  - eeg_rejected_requests_total (Counter)
- Logging: structured logs for SQI and request metadata (recommend JSON output).

Security & Cloud (S3, IAM, EC2)
--------------------------------
S3 layout (example)
- s3://mindtoscript/models/v1.0.0/bridge.pt
- s3://mindtoscript/models/v1.0.0/decoder/...
- s3://mindtoscript/dataset/v1.0.0/train/*.pt
- s3://mindtoscript/raw/v1.0.0/subjectX/

IAM Roles & Policies (recommended)
- EC2 instance role: read-only S3 access to s3://mindtoscript/* and CloudWatch put-metrics permission.
- Minimal policy example:
  - s3:GetObject, s3:ListBucket for the specific bucket/prefix
  - cloudwatch:PutMetricData
- Avoid storing static AWS keys; use instance profiles.

EC2 (Recommended)
- For development & inference: g4dn.xlarge (NVIDIA T4, 16GB VRAM)
- For heavier training: p3/ p4 instances (if needed)
- Use Deep Learning AMI (Ubuntu 22.04) or an AMI with CUDA 11.8 matching docker base image.

Deployment (Docker)
- Dockerfile based on pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
- entrypoint.sh downloads models (from MODEL_S3_URI) and either runs the one-shot example (RUN_EXAMPLE=1) or uvicorn.
- docker-compose.yml provides `web` and `example` services; use device_requests for GPU.

CI / CD
--------
- GitHub Actions CI runs unit tests and lints.
- For deploying to EC2:
  - Build image on CI or build on EC2.
  - Push model artifacts to S3 (versioned prefix).
  - Launch container on EC2 (systemd / docker-compose / ECS).

Monitoring & Alerts
-------------------
- Prometheus scrapes /metrics; Grafana dashboard shows latency, SQI, GPU memory, rejection rate.
- CloudWatch:
  - Billing alarm for threshold (e.g., $20/day or monthly cap)
  - EC2 CPU/GPU utilization and instance state alarms
- Alerts:
  - Alert when average SQI over 1 hour drops below 0.5
  - Alert on GPU memory leak (monotonic increase)
  - Alert on sudden rise in hard rejections

Database Schema (InferenceResult)
- id: int (PK)
- timestamp: datetime (index)
- text: string
- confidence: float
- sqi: float
- meta: JSON/text (request metadata)

End-to-End Data/Model Flow (detailed)
------------------------------------
1. Data ingestion: raw ZuCo files -> `scripts/load_zuco.py` -> canonical .pkl
2. Alignment: `scripts/align_zuco.py` converts onsets/offsets -> sample ranges and exports CSV if needed
3. Manifest & shards: `scripts/build_manifest_and_shards.py` -> versioned shards .pt
4. Training: `scripts/train_from_shards.py` trains bridge & saves models/bridge.pt
5. Serving: `app/main.py` loads bridge (bridge.pt) + HF decoder (or local folder) and serves /infer
6. Observability: metrics logged and saved; inference results persisted via `app/db.py`

30-day Implementation Timeline (approx.)
---------------------------------------
High-level cadence: 4 weeks (30 days) with milestones and deliverables.

Week 0 — Prep & Planning (Days 0–2)
- Tasks:
  - Finalize repo layout and DESIGN.md (this document)
  - Create AWS account / S3 bucket and enable versioning
  - Create IAM policies (dev role) and test S3 access via instance profile
- Deliverables: S3 bucket, versioning on, basic IAM role

Week 1 — Data & Preprocessing (Days 3–9)
- Days 3–4:
  - Download sample ZuCo subset or obtain preprocessed pickles
  - Run `scripts/inspect_zuco.py` and `scripts/load_zuco.py` on samples
- Days 5–7:
  - Implement and validate ICA + bandpass pipeline (MNE)
  - Implement per-word epoching (align_zuco.py) and SQI computation
- Days 8–9:
  - Generate canonical .pkl for several subjects
  - Run `scripts/build_manifest_and_shards.py` to create shards v0.0.1
- Deliverables: canonical pickles, shards v0.0.1, SQI reports

Week 2 — Model & Local Training (Days 10–16)
- Days 10–11:
  - Implement EEGEncoder + projection (app/model.py already present)
  - Create synthetic data notebook & run_demo_end_to_end
- Days 12–14:
  - Run `scripts/train_from_shards.py` on small shards locally (CPU or small GPU)
  - Validate bridge checkpoint saves (models/bridge.pt)
- Days 15–16:
  - Integrate bridge with HF decoder; run small inference tests
- Deliverables: trained bridge.pt (proof-of-concept), evaluation logs

Week 3 — Deployment & Monitoring (Days 17–23)
- Days 17–18:
  - Build Docker image and test locally (`docker compose build`)
  - Create entrypoint to fetch models from S3 (versioned)
- Days 19–20:
  - Launch EC2 g4dn.xlarge with Deep Learning AMI
  - Install Docker + NVIDIA Container Toolkit
- Days 21–23:
  - Run container on EC2; configure Prometheus scraping; set up CloudWatch
- Deliverables: running inference service on EC2, /metrics, logs

Week 4 — Hardening, Tests & CI/CD (Days 24–30)
- Days 24–26:
  - Add unit/integration tests and CI (GH Actions)
  - Add monitoring alerts (Prometheus/Grafana or CloudWatch)
- Days 27–28:
  - Create automated model download & reload workflow (S3 v prefix + MODEL_S3_URI)
  - Implement backup and rotation (old model versions) policy
- Days 29–30:
  - Run end-to-end evaluation (WER/BLEU), tune SQI thresholds and retrain as needed
  - Prepare handoff docs, runbook, and cost estimates
- Deliverables: production-ready service with monitoring & CI, evaluation report

Operational Notes & Runbook
---------------------------
- Short-term: use small subset of ZuCo and synthetic data for fast iteration.
- Model versioning: always store bridge.pt with a JSON config; require decoder d_model match or save projection compatible with decoder.dim.
- Cost-control: use CloudWatch alarms, stop EC2 outside working hours, prefer spot for experiments.
- Security: use IAM roles for EC2 to access S3, rotate any service credentials, use private S3 buckets and encrypt at rest.

Commands & Quickstart (summary)
--------------------------------
- Build image locally:
  docker compose build
- Run API locally:
  . .venv/Scripts/Activate.ps1
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
- Build shards:
  python scripts/build_manifest_and_shards.py --canonical ./data/canonical --out ./data/shards --version v1.0.0
- Train (small):
  python scripts/train_from_shards.py --version v1.0.0 --shard-root ./data/shards --model-dir ./models --epochs 3
- Run demo with HF decoder:
  python scripts/run_demo_with_bart.py

Appendix: Recommended IAM policy (minimal read-only S3)
------------------------------------------------------
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name",
        "arn:aws:s3:::your-bucket-name/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "cloudwatch:PutMetricData"
      ],
      "Resource": "*"
    }
  ]
}

Closing
-------
This document is a practical blueprint to take Mind-to-Script from prototype to a production-capable service within ~30 days of focused work. If you want, I can:
- Generate architecture diagrams in SVG/PNG,
- Produce terraform templates for S3 + IAM + EC2 (infrastructure as code),
- Or open a PR that wires this DESIGN.md into the README and links to runbooks.

Pick one and I’ll implement it next. 
