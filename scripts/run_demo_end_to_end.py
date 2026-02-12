#!/usr/bin/env python3
"""
Run a lightweight end-to-end demo: create synthetic canonical pickle,
build shards, create a bridge checkpoint, load model and run inference.
"""
import os
from pathlib import Path
import pickle
import numpy as np
import subprocess
import torch

ROOT = Path(__file__).resolve().parents[1]
os.makedirs(ROOT / "data" / "canonical", exist_ok=True)
os.makedirs(ROOT / "models", exist_ok=True)
os.makedirs(ROOT / "data" / "shards", exist_ok=True)

print("Creating synthetic canonical pickle...")
sfreq = 500.0
n_channels = 8
duration_s = 2.0
n_samples = int(sfreq * duration_s)
signals = np.random.randn(n_channels, n_samples).astype(np.float32)
sentences = [
    {
        "text": "hello world",
        "words": ["hello", "world"],
        "onsets": [0.1, 1.0],
        "offsets": [0.6, 1.5],
    }
]
canonical = {
    "signals": signals,
    "sfreq": sfreq,
    "ch_names": [f"EEG{i}" for i in range(n_channels)],
    "sentences": sentences,
    "meta": {"subject": "subj_synth"},
}
p = ROOT / "data" / "canonical" / "subj_synth_file1.pkl"
with open(p, "wb") as f:
    pickle.dump(canonical, f)
print("Wrote canonical pickle:", p)

print("Building shards (this calls build_manifest_and_shards.py)...")
subprocess.run(["python", "scripts/build_manifest_and_shards.py", "--canonical", "data/canonical", "--out", "data/shards", "--version", "v0.0.1", "--shard-size", "16"], check=True)
print("Shards built under data/shards/v0.0.1")

print("Creating and saving a bridge checkpoint (encoder + projection)...")
from app.model import EEGEncoder
cnn_channels = 32
lstm_hidden = 64
in_channels = n_channels
encoder = EEGEncoder(in_channels=in_channels, cnn_channels=cnn_channels, lstm_hidden=lstm_hidden)
projection = torch.nn.Linear(lstm_hidden * 2, 128)  # demo d_model=128
ckpt = {
    "encoder_state_dict": encoder.state_dict(),
    "projection_state_dict": projection.state_dict(),
    "config": {"in_channels": in_channels, "cnn_channels": cnn_channels, "lstm_hidden": lstm_hidden, "hf_model_name": "dummy"},
}
torch.save(ckpt, ROOT / "models" / "bridge.pt")
print("Saved bridge checkpoint to models/bridge.pt")

print("Loading bridge into runtime model and running a demo inference...")
from app.model import MindToScriptModel

class DummyTokenizer:
    def batch_decode(self, sequences, skip_special_tokens=True):
        return ["decoded text"] * (sequences.shape[0] if hasattr(sequences, "shape") else len(sequences))

class DummyDecoder:
    def __init__(self, d_model=128, vocab_size=100):
        self.config = type("C", (), {"d_model": d_model})
        self.vocab_size = vocab_size
    def generate(self, encoder_outputs=None, max_length=32, num_beams=1, return_dict_in_generate=True, output_scores=True):
        batch = encoder_outputs.last_hidden_state.shape[0]
        sequences = torch.randint(0, self.vocab_size, (batch, 5))
        scores = [torch.randn(batch, self.vocab_size) for _ in range(4)]
        return type("G", (), {"sequences": sequences, "scores": scores})()

m = MindToScriptModel(device="cpu")
m.tokenizer = DummyTokenizer()
m.decoder = DummyDecoder(d_model=128)
ck = torch.load(ROOT / "models" / "bridge.pt", map_location="cpu")
m._bridge_config = ck.get("config", {})
m._bridge_ckpt = str(ROOT / "models" / "bridge.pt")
# set d_model from decoder to allow projection init
m._d_model = getattr(m.decoder, "config", type("C", (), {"d_model": 128})).d_model
m._ensure_encoder(in_channels)
print("Model bridge loaded (encoder + projection).")

sig = np.random.randn(in_channels, 256).astype(float)
texts, confidences = m.predict(sig)
print("Predicted text:", texts[0])
print("Confidence:", confidences[0])

print("Demo complete.")

