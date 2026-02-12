#!/usr/bin/env python3
"""
summarize_canonical.py

Scan data/canonical/*.pkl and produce a manifest CSV with basic stats:
  file, n_channels, n_samples, sfreq, duration_s, n_sentences, n_words_total, path

Usage:
  python scripts/summarize_canonical.py --input data/canonical --out data/manifest.csv
"""
import argparse
from pathlib import Path
import pickle
import numpy as np
import csv

def inspect_file(p: Path):
    with open(p, "rb") as f:
        obj = pickle.load(f)
    signals = np.asarray(obj.get("signals"))
    sfreq = float(obj.get("sfreq", np.nan))
    n_channels, n_samples = signals.shape if signals.ndim == 2 else (np.nan, np.nan)
    duration = n_samples / sfreq if sfreq and not np.isnan(n_samples) else np.nan
    sentences = obj.get("sentences", [])
    n_sent = len(sentences) if hasattr(sentences, "__len__") else 0
    n_words = 0
    for s in sentences:
        w = s.get("words") if isinstance(s, dict) else None
        if w is None and isinstance(s, (list, tuple)):
            n_words += len(s)
        elif w is not None:
            n_words += len(w)
    return {
        "file": p.name,
        "path": str(p),
        "n_channels": int(n_channels) if not np.isnan(n_channels) else "",
        "n_samples": int(n_samples) if not np.isnan(n_samples) else "",
        "sfreq": float(sfreq) if not np.isnan(sfreq) else "",
        "duration_s": float(duration) if not np.isnan(duration) else "",
        "n_sentences": int(n_sent),
        "n_words_total": int(n_words),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", default="data/canonical")
    p.add_argument("--out", "-o", default="data/manifest.csv")
    args = p.parse_args()
    inp = Path(args.input)
    out = Path(args.out)
    files = sorted(inp.rglob("*.pkl"))
    if not files:
        print("No canonical .pkl files found in", inp)
        return
    rows = []
    for f in files:
        try:
            rows.append(inspect_file(f))
        except Exception as e:
            print("skip", f, "err", e)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as csvf:
        w = csv.DictWriter(csvf, fieldnames=["file","path","n_channels","n_samples","sfreq","duration_s","n_sentences","n_words_total"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("Wrote manifest:", out, "rows:", len(rows))

if __name__ == "__main__":
    main()

