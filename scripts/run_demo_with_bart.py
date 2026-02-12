#!/usr/bin/env python3
"""
run_demo_with_bart.py

Load the saved bridge and a real HuggingFace BART-base decoder, run a single synthetic EEG through the pipeline.
"""
import torch
import numpy as np
from app.model import MindToScriptModel
import os

def main():
    model_dir = os.environ.get("MODEL_DIR", "./models")
    hf_name = os.environ.get("HF_MODEL_NAME", "facebook/bart-base")
    device = "cuda" if (os.environ.get("FORCE_CPU", "0") == "0" and torch.cuda.is_available()) else "cpu"
    print("Using device:", device)
    m = MindToScriptModel(device=device)
    # Will lazily import transformers inside load()
    m.load(model_dir=model_dir, hf_model_name=hf_name)
    # prepare synthetic signal (bridge config may specify in_channels)
    cfg = getattr(m, "_bridge_config", {}) or {}
    channels = int(cfg.get("in_channels", 8))
    length = 256
    sig = np.random.randn(channels, length).astype(float)
    texts, confidences = m.predict(sig, max_length=32, num_beams=2)
    print("Prediction:", texts[0])
    print("Confidence:", confidences[0])

if __name__ == "__main__":
    main()

