#!/usr/bin/env python3
"""
align_zuco.py

Align EEG signals to per-word onsets/offsets and produce CSV of sample ranges.
(Heuristic-based for ZuCo-style .mat/.pkl files.)
"""
from pathlib import Path
import argparse
import pickle
import re
from scipy.io import loadmat
import numpy as np
import pandas as pd

def load_file(path: Path):
    suf = path.suffix.lower()
    if suf == ".mat":
        return loadmat(str(path), squeeze_me=True, struct_as_record=False)
    elif suf in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        raise SystemExit("Unsupported file")

def largest_2d(d):
    best_k = None
    best_arr = None
    best_size = 0
    for k,v in d.items():
        if k.startswith("__"):
            continue
        try:
            a = np.array(v)
            if a.ndim == 2 and a.size > best_size:
                best_size = a.size
                best_arr = a
                best_k = k
        except Exception:
            continue
    return best_k, best_arr

def find_sfreq(d):
    for key in ("sfreq","srate","fs","sampling_rate"):
        if key in d:
            try:
                return float(d[key])
            except Exception:
                pass
    # guess none
    return None

def find_annotations(d):
    # naive: look for words/onsets/offsets
    words = None
    onsets = None
    offsets = None
    for k,v in d.items():
        kl = k.lower()
        if "word" in kl and ("onset" in kl or "time" in kl):
            onsets = v
        if "word" in kl and ("offset" in kl or "end" in kl):
            offsets = v
        if kl in ("words","wordlist","tokens","text","sentences"):
            words = v
    return words, onsets, offsets

def align(path: Path, sfreq_override=None, out_csv=None):
    raw = load_file(path)
    k, arr = largest_2d(raw)
    if arr is None:
        raise SystemExit("No 2D EEG array found")
    signals = np.asarray(arr)
    if signals.shape[0] > signals.shape[1]:
        signals = signals.T
    n_channels, n_samples = signals.shape
    sfreq = find_sfreq(raw) or sfreq_override
    if sfreq is None:
        raise SystemExit("sfreq not found; pass --sfreq")
    words, onsets, offsets = find_annotations(raw)
    rows = []
    if words is None or onsets is None:
        print("No annotations found")
    else:
        onsets = np.array(onsets, dtype=float)
        if np.median(onsets) > 1000:
            onsets = onsets / 1000.0
        if offsets is not None:
            offsets = np.array(offsets, dtype=float)
            if np.median(offsets) > 1000:
                offsets = offsets / 1000.0
        for i, w in enumerate(words):
            t0 = onsets[i] if i < len(onsets) else None
            t1 = offsets[i] if (offsets is not None and i < len(offsets)) else None
            s0 = int(round(t0 * sfreq)) if t0 is not None else None
            s1 = int(round(t1 * sfreq)) if t1 is not None else None
            rows.append({"word_idx": i, "word": str(w), "onset_s": t0, "offset_s": t1, "onset_sample": s0, "offset_sample": s1})
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    print(df.head(200).to_string(index=False))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", "-p", required=True)
    p.add_argument("--sfreq", type=float)
    p.add_argument("--out")
    args = p.parse_args()
    align(Path(args.path), sfreq_override=args.sfreq, out_csv=Path(args.out) if args.out else None)

if __name__ == "__main__":
    main()

