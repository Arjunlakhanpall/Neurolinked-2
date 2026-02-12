#!/usr/bin/env python3
"""
train_bridge.py

Train the EEG -> text "bridge" (EEGEncoder + projection) to interface with a frozen BART decoder.

Usage:
  python scripts/train_bridge.py --data-dir ./data --model-dir ./models --epochs 5 --batch-size 4

Notes:
- Expects canonical .pkl files in data-dir with keys: 'signals' (n_channels, n_samples) and 'sentences' (list or single string).
- Saved bridge checkpoint: models/bridge.pt containing encoder & projection state_dicts.
- For quick demo use --use-synthetic to train on random signals with short target sentences.
"""
import argparse
import os
from pathlib import Path
import pickle
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from app.model import EEGEncoder


class ZucoDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.files = list(Path(data_dir).rglob("*.pkl"))
        if not self.files:
            raise RuntimeError(f"No .pkl files found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        with open(p, "rb") as f:
            obj = pickle.load(f)
        signals = obj.get("signals") or obj.get("eeg") or obj.get("X")
        # ensure numpy
        signals = np.asarray(signals, dtype=float)
        # canonicalize to (channels, samples)
        if signals.ndim == 2 and signals.shape[0] > signals.shape[1]:
            signals = signals.T
        sentence = None
        for k in ("sentences", "text", "labels"):
            if k in obj:
                sentence = obj[k]
                break
        if sentence is None:
            sentence = " ".join(["word"] * 5)
        if isinstance(sentence, list):
            sentence = sentence[0] if sentence else " ".join(["word"] * 5)
        return signals, sentence


class SyntheticDataset(Dataset):
    def __init__(self, n_samples=100, channels=8, length=256):
        self.n = n_samples
        self.channels = channels
        self.length = length

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        signals = np.random.randn(self.channels, self.length).astype(float)
        sentence = "hello world"
        return signals, sentence


def collate_fn(batch):
    signals_list, texts = zip(*batch)
    # pad signals to max length in batch
    max_len = max(s.shape[1] for s in signals_list)
    channels = signals_list[0].shape[0]
    arr = np.zeros((len(signals_list), channels, max_len), dtype=float)
    for i, s in enumerate(signals_list):
        arr[i, :, : s.shape[1]] = s
    return torch.from_numpy(arr).float(), list(texts)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print("Using device:", device)

    # tokenizer + decoder (frozen)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
    decoder = AutoModelForSeq2SeqLM.from_pretrained(args.hf_model_name).to(device)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    # dataset
    if args.use_synthetic:
        ds = SyntheticDataset(n_samples=200, channels=args.in_channels, length=args.sample_length)
    else:
        ds = ZucoDataset(Path(args.data_dir))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # encoder + projection
    encoder = EEGEncoder(in_channels=args.in_channels, cnn_channels=args.cnn_channels, lstm_hidden=args.lstm_hidden).to(device)
    proj = torch.nn.Linear(args.lstm_hidden * 2, decoder.config.d_model).to(device)

    # optimizer only for encoder + proj
    opt = AdamW(list(encoder.parameters()) + list(proj.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        encoder.train()
        for signals, texts in dl:
            signals = signals.to(device)  # (batch, channels, samples)
            # simple per-channel zscore
            mean = signals.mean(dim=2, keepdim=True)
            std = signals.std(dim=2, keepdim=True) + 1e-6
            signals = (signals - mean) / std

            enc_out = encoder(signals)  # (batch, seq_len, hidden*2)
            proj_out = proj(enc_out)  # (batch, seq_len, d_model)

            # tokenize targets
            tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
            labels = tokenized["input_ids"]

            # forward through decoder using encoder_outputs
            outputs = decoder(input_ids=None, attention_mask=None, encoder_outputs=(proj_out,), labels=labels)
            loss = outputs.loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * signals.size(0)

        avg = total_loss / len(ds)
        print(f"Epoch {epoch+1}/{args.epochs}  avg_loss={avg:.4f}")

    # save bridge
    os.makedirs(args.model_dir, exist_ok=True)
    ckpt = {
        "encoder_state_dict": encoder.state_dict(),
        "projection_state_dict": proj.state_dict(),
        "config": {
            "in_channels": args.in_channels,
            "cnn_channels": args.cnn_channels,
            "lstm_hidden": args.lstm_hidden,
            "hf_model_name": args.hf_model_name,
        },
    }
    outp = Path(args.model_dir) / "bridge.pt"
    torch.save(ckpt, str(outp))
    print("Saved bridge checkpoint to", outp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--model-dir", default="./models")
    p.add_argument("--hf-model-name", default="facebook/bart-base")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--in-channels", dest="in_channels", type=int, default=8)
    p.add_argument("--cnn-channels", dest="cnn_channels", type=int, default=64)
    p.add_argument("--lstm-hidden", dest="lstm_hidden", type=int, default=128)
    p.add_argument("--sample-length", type=int, default=256)
    p.add_argument("--use-synthetic", action="store_true")
    p.add_argument("--force-cpu", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

