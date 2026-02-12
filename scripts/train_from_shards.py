#!/usr/bin/env python3
"""
train_from_shards.py

Train EEG encoder + projection (bridge) from dataset shards created by build_manifest_and_shards.py.
"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.model import EEGEncoder
from torch.optim import AdamW


class ShardDataset(Dataset):
    def __init__(self, shard_paths):
        self.items = []
        for p in shard_paths:
            data = torch.load(str(p))
            # each data is list of dicts with 'signals','target_text','meta'
            for rec in data:
                self.items.append(rec)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        signals = np.asarray(rec["signals"], dtype=np.float32)  # (ch, samples)
        return signals, rec["target_text"]


def collate_batch(batch, tokenizer, label_max_len=64):
    signals_list, texts = zip(*batch)
    # pad signals
    max_s = max(s.shape[1] for s in signals_list)
    ch = signals_list[0].shape[0]
    arr = np.zeros((len(signals_list), ch, max_s), dtype=np.float32)
    for i, s in enumerate(signals_list):
        arr[i, :, : s.shape[1]] = s
    signals = torch.from_numpy(arr)
    tokenized = tokenizer(list(texts), padding=True, truncation=True, max_length=label_max_len, return_tensors="pt")
    labels = tokenized.input_ids
    labels[labels == tokenizer.pad_token_id] = -100
    return signals, labels


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
    decoder = AutoModelForSeq2SeqLM.from_pretrained(args.hf_model_name).to(device)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    # collect shard files for train/val
    version_dir = Path(args.shard_root) / args.version
    train_shards = sorted((version_dir / "train").glob("*.pt"))
    val_shards = sorted((version_dir / "val").glob("*.pt"))
    if not train_shards:
        raise SystemExit("No train shards found")

    train_ds = ShardDataset(train_shards)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=lambda b: collate_batch(b, tokenizer, args.label_max_len))

    # instantiate encoder + proj
    encoder = EEGEncoder(in_channels=args.in_channels, cnn_channels=args.cnn_channels, lstm_hidden=args.lstm_hidden).to(device)
    proj = nn.Linear(args.lstm_hidden * 2, decoder.config.d_model).to(device)

    opt = AdamW(list(encoder.parameters()) + list(proj.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        encoder.train()
        total = 0.0
        for signals, labels in train_dl:
            signals = signals.to(device)
            # normalize per-channel
            mean = signals.mean(dim=2, keepdim=True)
            std = signals.std(dim=2, keepdim=True) + 1e-6
            signals = (signals - mean) / std
            enc_out = encoder(signals)
            proj_out = proj(enc_out)
            # forward through decoder
            labels = labels.to(device)
            outputs = decoder(input_ids=None, attention_mask=None, encoder_outputs=(proj_out,), labels=labels)
            loss = outputs.loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * signals.size(0)
        avg = total / len(train_ds)
        print(f"Epoch {epoch+1}/{args.epochs} avg_loss={avg:.4f}")

    # save bridge checkpoint
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    ckpt = {"encoder_state_dict": encoder.state_dict(), "projection_state_dict": proj.state_dict(), "config": {"in_channels": args.in_channels, "cnn_channels": args.cnn_channels, "lstm_hidden": args.lstm_hidden, "hf_model_name": args.hf_model_name}}
    torch.save(ckpt, str(Path(args.model_dir) / "bridge.pt"))
    print("Saved bridge to", Path(args.model_dir) / "bridge.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard-root", default="./data/shards")
    p.add_argument("--version", required=True)
    p.add_argument("--model-dir", default="./models")
    p.add_argument("--hf-model-name", default="facebook/bart-base")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--in-channels", type=int, default=8)
    p.add_argument("--cnn-channels", type=int, default=64)
    p.add_argument("--lstm-hidden", type=int, default=128)
    p.add_argument("--label-max-len", dest="label_max_len", type=int, default=64)
    p.add_argument("--force-cpu", action="store_true")
    args = p.parse_args()
    train(args)

if __name__ == "__main__":
    main()

