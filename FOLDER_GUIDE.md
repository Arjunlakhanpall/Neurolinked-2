# Repository Folder Guide — Neurolinked

This guide explains each top-level folder, what files live there, and step‑by‑step implementation / validation tasks you can run to make the component functional.

Purpose
-------
Make it easy for new contributors (or future you) to understand what lives where and which commands to run to verify functionality.

Top-level layout
----------------
- `app/` — production code
  - `main.py` — FastAPI app, endpoints (/infer, /metrics, /history)
  - `model.py` — MindToScriptModel, EEGEncoder, preprocessing helpers
  - `db.py` — SQLAlchemy persistence (InferenceResult)
  - Implementation steps:
    1. Inspect `app/model.py` to confirm `MindToScriptModel.load()` points to `MODEL_DIR`. Set `MODEL_DIR=./models`.
    2. Run locally: `uvicorn app.main:app --reload` and hit `/metrics` and `/docs`.
    3. Validate DB: call `/history` after running `scripts/infer_from_canonical.py`.

- `scripts/` — data engineering, training, helpers
  - `inspect_zuco.py` — file inspector for .mat/.edf/.pkl
  - `load_zuco.py` — canonicalize raw ZuCo -> .pkl
  - `align_zuco.py` — find word onsets/offsets and export CSV
  - `build_manifest_and_shards.py` — build per-word shards (train/val/test)
  - `train_from_shards.py` — training loop for bridge (encoder+proj) -> saves `models/bridge.pt`
  - `train_bridge.py` — alternate trainer
  - `download_weights.py` — S3 weight downloader (enforces versioning)
  - `run_demo*` scripts — demo/inference drivers
  - `infer_from_canonical.py` — convenience: infer a single epoch from canonical pickle
  - `summarize_canonical.py`, `export_epochs_csv.py` — data summarizers/exports
  - Implementation steps:
    1. Put canonical `.pkl` in `data/canonical/` (or run `load_zuco.py`).
    2. Run `python scripts/summarize_canonical.py` to verify pickles.
    3. Build shards: `python scripts/build_manifest_and_shards.py --canonical data/canonical --out data/shards --version v0.0.1`.
    4. Train a small bridge on synthetic data: `python scripts/train_from_shards.py --version v0.0.1 --epochs 2 --batch-size 4 --force-cpu`.

- `data/` — dataset artifacts (not checked into large objects)
  - `canonical/` — canonical .pkl files
  - `shards/` — versioned shards for training
  - `manifest.csv`, `epochs.csv` — generated summaries
  - Implementation steps:
    1. Place or generate canonical `.pkl` files.
    2. Run `scripts/export_epochs_csv.py` to produce `data/epochs.csv`.

- `models/` — local model artifacts (gitignored)
  - `bridge.pt` — saved encoder + projection
  - `decoder/` — optional HF decoder folder (if you exported locally)
  - Implementation steps:
    1. After training, confirm `models/bridge.pt` exists.
    2. Start server and ensure it detects the checkpoint on startup.

- `notebooks/` — demos & exploration
  - `end_to_end_demo.ipynb` — synthetic end-to-end demo
  - Implementation steps:
    1. Start Jupyter: `jupyter notebook notebooks/end_to_end_demo.ipynb`.
    2. Run sequentially to reproduce synthetic demo and saved bridge.

- `docker/` or root `Dockerfile` + `docker-compose.yml`
  - Docker image (PyTorch CUDA runtime), `entrypoint.sh` downloads weights and runs server or example.
  - Implementation steps:
    1. Build locally: `docker compose build`.
    2. Run web: `docker compose up web` (or `docker compose run --rm example` for one-shot).

- `.github/` — CI
  - `workflows/ci.yml` — GitHub Actions (runs tests on push/PR)
  - Implementation steps:
    1. Push branch and open PR to see CI run automatically.

- `tests/` — unit tests
  - `test_model.py`, `test_api.py`
  - Implementation steps:
    1. Run `pytest -q`. If failing, check missing deps (set PYTHONPATH="." if needed).

- repo root docs
  - `README.md`, `DESIGN.md`, `FOLDER_GUIDE.md` (this file)
  - `LICENSE` (add one)

Practical "make functional" checklist
------------------------------------
1. Environment
   - Create venv, install `requirements-ci.txt` and optionally full `requirements.txt`.
2. Data
   - Place canonical pickles into `data/canonical/` or run `scripts/load_zuco.py` on ZuCo raw files.
3. Inspect
   - `python scripts/summarize_canonical.py` and `python scripts/export_epochs_csv.py`.
4. Train (sanity)
   - `python scripts/train_from_shards.py --version v0.0.1 --epochs 2 --batch-size 4 --force-cpu`.
5. Serve
   - Put `models/bridge.pt` in `models/`.
   - Start server: `uvicorn app.main:app --reload`.
   - Test: `curl /infer` with small synthetic epoch or `python scripts/infer_from_canonical.py`.
6. Dockerize
   - `docker compose build` then `docker compose up web`.
7. CI & Repo
   - Ensure tests pass, commit changes, push to remote, open PR.

Maintenance & Best Practices
----------------------------
- Large binary artifacts (raw EEG, large models) should be stored on S3; commit only manifests.
- Always version data and models: `s3://bucket/path/v1.0.0/`.
- Use instance roles and avoid embedding credentials in `.env` or code.

If you want, I will:
- generate `FOLDER_GUIDE.md` (this file) into the repo (done),
- create a short script that walks a new developer through Steps 1–6 automatically,
- or open a PR with these docs and run tests in CI.

