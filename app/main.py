from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
from typing import Optional, List

from .model import MindToScriptModel
import time
import torch
from prometheus_client import Histogram, Gauge, Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from fastapi import BackgroundTasks
from . import db as _db

app = FastAPI(title="Mind-to-Script")

# Prometheus metrics
INFER_LATENCY = Histogram("eeg_inference_latency_seconds", "EEG inference latency seconds", buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0))
SQI_GAUGE = Gauge("eeg_sqi_score", "Latest signal quality index (0-1)")
GPU_MEM_GAUGE = Gauge("gpu_memory_used_bytes", "GPU memory allocated in bytes")
REJECT_COUNTER = Counter("eeg_rejected_requests_total", "Total number of hard-rejected EEG requests")


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

class EEGPayload(BaseModel):
    signals: List[List[float]]  # channels x samples
    sfreq: Optional[float] = None


@app.on_event("startup")
def load_model():
    model_dir = os.environ.get("MODEL_DIR", "./models")
    hf_name = os.environ.get("HF_MODEL_NAME", "facebook/bart-base")
    device = "cuda" if (os.environ.get("FORCE_CPU", "0") == "0" and __import__("torch").cuda.is_available()) else "cpu"
    print(f"Loading model on device={device} from model_dir={model_dir} or HF={hf_name}")
    m = MindToScriptModel(device=device)
    try:
        m.load(model_dir=model_dir, hf_model_name=hf_name)
        app.state.model = m
        print("Model loaded successfully.")
    except Exception as e:
        app.state.model = None
        print("Failed to load model:", e)
    # initialize DB
    try:
        _db.init_db()
        print("Database initialized:", _db.DATABASE_URL)
    except Exception as e:
        print("Failed to init DB:", e)


@app.post("/infer")
def infer(payload: EEGPayload):
    arr = np.array(payload.signals, dtype=float)
    if arr.ndim != 2:
        raise HTTPException(status_code=400, detail="signals must be 2D list (channels x samples)")
    model: MindToScriptModel = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    # Compute simple SQI and enforce soft/hard rejection policy
    try:
        sqi_score, sqi_details = model.compute_sqi(arr)
    except Exception:
        sqi_score, sqi_details = None, {}

    # Hard rejection: extremely low SQI or too many channels exceeding ±100µV
    bad_amp_fraction = sqi_details.get("bad_amp_fraction", 0.0)
    if (sqi_score is not None and sqi_score < 0.25) or (bad_amp_fraction > 0.2):
        REJECT_COUNTER.inc()
        raise HTTPException(status_code=422, detail="Signal quality too low for inference.")

    # Soft warning: proceed but mark as unreliable
    is_unreliable = False
    if sqi_score is not None and 0.25 <= sqi_score < 0.5:
        is_unreliable = True

    start = time.perf_counter()
    try:
        texts, confidences = model.predict(arr, sfreq=payload.sfreq)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    elapsed = time.perf_counter() - start
    INFER_LATENCY.observe(elapsed)

    # update SQI metric
    if sqi_score is not None:
        SQI_GAUGE.set(float(sqi_score))

    # update GPU memory metric if available
    try:
        if torch.cuda.is_available():
            GPU_MEM_GAUGE.set(int(torch.cuda.memory_allocated()))
    except Exception:
        pass

    meta = {"shape": arr.shape, "sqi": sqi_score, "sqi_details": sqi_details, "is_unreliable": is_unreliable}

    # save result in background
    try:
        BackgroundTasks  # type: ignore
    except Exception:
        pass
    def _save():
        try:
            _db.save_result(texts[0], float(confidences[0]), sqi_score, meta)
        except Exception as e:
            print("Failed to save inference result:", e)

    # schedule background save
    bg = BackgroundTasks()
    bg.add_task(_save)
    return {"text": texts[0], "confidence": float(confidences[0]), "meta": meta}


@app.get("/history")
def history(limit: int = 100):
    return _db.list_results(limit=limit)

