#!/usr/bin/env python3
"""
infer_from_canonical.py

Load a canonical .pkl (ZuCo-style), extract the first word epoch, run MindToScriptModel.predict,
print text, confidence and SQI, and save result to DB.
"""
from pathlib import Path
import pickle
import numpy as np
import os

from app.model import MindToScriptModel
from app import db as _db

def find_canonical(dirpath: Path):
    files = sorted(dirpath.rglob("*.pkl"))
    return files[0] if files else None

def extract_first_word_epoch(p: Path):
    with open(p, "rb") as f:
        obj = pickle.load(f)
    signals = np.asarray(obj.get("signals"))
    sfreq = float(obj.get("sfreq", 500.0))
    sentences = obj.get("sentences", [])
    if not sentences:
        raise SystemExit("No sentences/annotations in canonical file")
    s = sentences[0]
    words = s.get("words", [])
    onsets = s.get("onsets", [])
    offsets = s.get("offsets", [])
    if not words or not onsets:
        raise SystemExit("No word onsets available")
    widx = 0
    t0 = float(onsets[widx])
    t1 = float(offsets[widx]) if offsets and widx < len(offsets) else t0 + 0.5
    s0 = max(0, int(round(t0 * sfreq)))
    s1 = min(signals.shape[1], int(round(t1 * sfreq)))
    epoch = signals[:, s0:s1]
    meta = {"source": str(p), "sentence_idx": 0, "word_idx": widx, "word": str(words[widx])}
    return epoch.astype(float), sfreq, meta, str(words[widx])

def main():
    data_dir = Path("data/canonical")
    p = find_canonical(data_dir)
    if not p:
        raise SystemExit("No canonical .pkl files found in data/canonical. Create them with scripts/load_zuco.py")
    print("Using canonical file:", p)
    epoch, sfreq, meta, target = extract_first_word_epoch(p)
    print("Target word:", target, "epoch shape:", epoch.shape, "sfreq:", sfreq)

    # load model
    device = "cpu"
    m = MindToScriptModel(device=device)
    m.load(model_dir=os.environ.get("MODEL_DIR", "./models"), hf_model_name=os.environ.get("HF_MODEL_NAME", "facebook/bart-base"))

    # compute SQI
    sqi_score, sqi_details = m.compute_sqi(epoch)
    print("SQI:", sqi_score, "details:", sqi_details)

    # predict
    texts, confidences = m.predict(epoch, sfreq=sfreq, max_length=64, num_beams=2)
    print("Prediction:", texts[0])
    print("Confidence:", confidences[0])

    # save to DB
    try:
        _db.init_db()
        rid = _db.save_result(texts[0], float(confidences[0]), float(sqi_score) if sqi_score is not None else None, {"meta": meta})
        print("Saved inference id:", rid)
    except Exception as e:
        print("Failed to save result to DB:", e)

if __name__ == "__main__":
    main()

