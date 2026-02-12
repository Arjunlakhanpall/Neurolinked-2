#!/usr/bin/env python3
"""
export_epochs_csv.py

Export per-word epochs from canonical pickles into a single CSV:
 columns: file, sentence_idx, word_idx, word, onset_s, offset_s, onset_sample, offset_sample, n_channels, n_samples, sfreq, duration_s, path

Usage:
  python scripts/export_epochs_csv.py --input data/canonical --out data/epochs.csv --sfreq 500
"""
import argparse
from pathlib import Path
import pickle
import csv
import numpy as np

def process_file(p: Path, sfreq_override=None):
    with open(p, "rb") as f:
        obj = pickle.load(f)
    sfreq = float(obj.get("sfreq", sfreq_override or 0.0))
    signals = np.asarray(obj.get("signals"))
    n_channels, n_samples = signals.shape if signals.ndim == 2 else (None, None)
    duration = n_samples / sfreq if sfreq else None
    sentences = obj.get("sentences", [])
    rows = []
    for si, s in enumerate(sentences):
        words = s.get("words") if isinstance(s, dict) else (s if isinstance(s, (list,tuple)) else None)
        onsets = s.get("onsets") if isinstance(s, dict) else None
        offsets = s.get("offsets") if isinstance(s, dict) else None
        if not words or onsets is None:
            continue
        onsets = list(map(float, onsets))
        offsets = list(map(float, offsets)) if offsets is not None else [None]*len(onsets)
        for wi, w in enumerate(words):
            t0 = onsets[wi] if wi < len(onsets) else None
            t1 = offsets[wi] if wi < len(offsets) else None
            s0 = int(round(t0 * sfreq)) if t0 is not None else None
            s1 = int(round(t1 * sfreq)) if t1 is not None else None
            rows.append({
                "file": p.name,
                "path": str(p),
                "sentence_idx": si,
                "word_idx": wi,
                "word": str(w),
                "onset_s": t0,
                "offset_s": t1,
                "onset_sample": s0,
                "offset_sample": s1,
                "n_channels": n_channels,
                "n_samples": n_samples,
                "sfreq": sfreq,
                "duration_s": duration
            })
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", default="data/canonical")
    p.add_argument("--out", "-o", default="data/epochs.csv")
    p.add_argument("--sfreq", type=float, default=None, help="optional sfreq override")
    args = p.parse_args()
    inp = Path(args.input)
    out = Path(args.out)
    files = sorted(inp.rglob("*.pkl"))
    if not files:
        print("No canonical files found in", inp)
        return
    allrows = []
    for fpath in files:
        try:
            allrows.extend(process_file(fpath, sfreq_override=args.sfreq))
        except Exception as e:
            print("skip", fpath, "err", e)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file","path","sentence_idx","word_idx","word","onset_s","offset_s","onset_sample","offset_sample","n_channels","n_samples","sfreq","duration_s"]
    with open(out, "w", newline="", encoding="utf-8") as csvf:
        w = csv.DictWriter(csvf, fieldnames=fieldnames)
        w.writeheader()
        for r in allrows:
            w.writerow(r)
    print("Wrote epochs CSV:", out, "rows:", len(allrows))

if __name__ == "__main__":
    main()

