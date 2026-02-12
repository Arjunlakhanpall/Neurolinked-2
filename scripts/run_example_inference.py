#!/usr/bin/env python3
"""
run_example_inference.py

Load the bridge (if present) and BART decoder, run a single synthetic EEG through the pipeline,
and print the predicted text and confidence.
"""
import os
import numpy as np

from app.model import MindToScriptModel


def main():
    model_dir = os.environ.get("MODEL_DIR", "./models")
    hf_name = os.environ.get("HF_MODEL_NAME", "facebook/bart-base")
    device = "cuda" if (os.environ.get("FORCE_CPU", "0") == "0" and __import__("torch").cuda.is_available()) else "cpu"
    print("Using device:", device)
    m = MindToScriptModel(device=device)
    try:
        m.load(model_dir=model_dir, hf_model_name=hf_name)
    except Exception as e:
        print("Failed to load model:", e)
        return

    # Determine channels from bridge config or default to 8
    cfg = getattr(m, "_bridge_config", {}) or {}
    channels = int(cfg.get("in_channels", 8))
    length = 256
    sig = np.random.randn(channels, length).astype(float)
    texts, confidences = m.predict(sig)
    print("Prediction:", texts[0])
    print("Confidence:", confidences[0])


if __name__ == "__main__":
    main()

