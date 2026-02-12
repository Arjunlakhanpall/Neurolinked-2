#!/usr/bin/env python3
"""
build_manifest_and_shards.py

Scans canonical pickles in a directory, builds a manifest of per-word epochs,
splits by subject into train/val/test, and writes shard files containing
pre-extracted epochs (signals + target_text + meta) suitable for training.

Usage:
  python scripts/build_manifest_and_shards.py --canonical ./data/canonical --out ./data/shards --version v1.0.0 --shard-size 256
"""
from pathlib import Path
import argparse
import pickle
import json
import random
import os
from typing import List, Dict, Any
import numpy as np
import torch

def extract_epochs_from_file(path: Path, min_duration_s: float = 0.02):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    sfreq = float(obj.get("sfreq", 500.0))
    sentences = obj.get("sentences", [])
    subject = obj.get("meta", {}).get("subject", None) or obj.get("subject", None) or path.stem.split("_")[0]
    out = []
    for si, s in enumerate(sentences):
        words = s.get("words", [])
        onsets = s.get("onsets", [])
        offsets = s.get("offsets", [])
        for wi, w in enumerate(words):
            if onsets is None:
                continue
            if wi >= len(onsets):
                continue
            t0 = float(onsets[wi])
            t1 = float(offsets[wi]) if offsets and wi < len(offsets) else (t0 + 0.5)
            dur = t1 - t0
            if dur < min_duration_s:
                continue
            s0 = max(0, int(round(t0 * sfreq)))
            s1 = max(s0 + 1, int(round(t1 * sfreq)))
            signals = np.asarray(obj["signals"])
            # canonical expects (channels, samples)
            if signals.ndim != 2:
                continue
            if s1 > signals.shape[1]:
                s1 = signals.shape[1]
            epoch = signals[:, s0:s1]
            meta = {"source": str(path), "sentence_idx": si, "word_idx": wi, "subject": subject}
            out.append({"signals": epoch.astype(np.float32), "target_text": str(w), "meta": meta})
    return out

def shard_and_write(entries: List[Dict[str,Any]], out_dir: Path, shard_size: int, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    i = 0
    while i < len(entries):
        shard = entries[i:i+shard_size]
        fname = out_dir / f"{prefix}_shard_{idx:04d}.pt"
        torch.save(shard, str(fname))
        idx += 1
        i += shard_size
    return

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--canonical", required=True, help="Directory with canonical .pkl files")
    p.add_argument("--out", required=True, help="Output directory for shards and manifest")
    p.add_argument("--version", required=True, help="Dataset version tag (e.g. v1.0.0)")
    p.add_argument("--shard-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    base = Path(args.canonical)
    files = sorted(list(base.rglob("*.pkl")))
    print(f"Found {len(files)} canonical files")
    all_entries = []
    # group by subject for split
    subject_map = {}
    for f in files:
        try:
            epochs = extract_epochs_from_file(f)
        except Exception as e:
            print("skip", f, "err", e)
            continue
        if not epochs:
            continue
        subject = epochs[0]["meta"].get("subject") or f.stem.split("_")[0]
        subject_map.setdefault(subject, []).extend(epochs)

    subjects = list(subject_map.keys())
    random.shuffle(subjects)
    n = len(subjects)
    n_test = max(1, n // 10)
    n_val = max(1, n // 10)
    test_subs = subjects[:n_test]
    val_subs = subjects[n_test:n_test+n_val]
    train_subs = subjects[n_test+n_val:]

    splits = {"train": [], "val": [], "test": []}
    for s, entries in subject_map.items():
        if s in test_subs:
            splits["test"].extend(entries)
        elif s in val_subs:
            splits["val"].extend(entries)
        else:
            splits["train"].extend(entries)

    out_root = Path(args.out)
    manifest = {"version": args.version, "counts": {}, "shards": {}}
    for split, entries in splits.items():
        random.shuffle(entries)
        manifest["counts"][split] = len(entries)
        shard_dir = out_root / args.version / split
        shard_and_write(entries, shard_dir, args.shard_size, split)
        shard_files = sorted([str(p.name) for p in shard_dir.glob("*.pt")])
        manifest["shards"][split] = shard_files
        print(f"Wrote {len(shard_files)} shards for {split} ({len(entries)} examples)")

    manifest_path = out_root / args.version / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("Manifest written to", manifest_path)

if __name__ == "__main__":
    main()

