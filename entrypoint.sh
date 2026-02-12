#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="./models"
mkdir -p "${MODEL_DIR}"

if [ -n "${MODEL_S3_URI:-}" ]; then
  echo "Downloading model from ${MODEL_S3_URI} to ${MODEL_DIR}"
  python3 scripts/download_weights.py --s3-uri "${MODEL_S3_URI}" --dest "${MODEL_DIR}"
else
  echo "No MODEL_S3_URI set; skipping download"
fi

# If RUN_EXAMPLE=1, run the example inference script and exit
if [ "${RUN_EXAMPLE:-0}" = "1" ]; then
  echo "RUN_EXAMPLE=1 -> running example inference script"
  python3 scripts/run_example_inference.py
  exit $?
fi

# Start the app (allow override by passing command)
if [ $# -eq 0 ]; then
  exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
else
  exec "$@"
fi

