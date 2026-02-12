#!/usr/bin/env python3
"""
load_zuco.py

Convert common ZuCo file types into a canonical compact pickle for training:
  dict with keys: signals (n_channels, n_samples), sfreq, ch_names, annotations, sentences
"""
from pathlib import Path
import argparse
import pickle
import numpy as np

def write_canonical(out_path, signals, sfreq=None, ch_names=None, annotations=None, sentences=None):
    obj = {
        "signals": np.asarray(signals),
        "sfreq": float(sfreq) if sfreq is not None else None,
        "ch_names": ch_names or [],
        "annotations": annotations or [],
        "sentences": sentences or []
    }
    with open(out_path, "wb") as f:
        pickle.dump(obj, f)
    print("Wrote canonical:", out_path, "shape:", obj["signals"].shape)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", required=True)
    args = p.parse_args()
    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        raise SystemExit("Input not found")
    suf = inp.suffix.lower()
    if suf == ".mat":
        from scipy.io import loadmat
        mat = loadmat(str(inp), squeeze_me=True, struct_as_record=False)
        # heuristics: largest 2D arr
        best = None
        best_k = None
        for k,v in mat.items():
            if k.startswith("__"):
                continue
            try:
                arr = np.array(v)
                if arr.ndim == 2 and (best is None or arr.size > best.size):
                    best = arr
                    best_k = k
            except Exception:
                continue
        if best is None:
            raise SystemExit("No 2D EEG array found in .mat")
        signals = best
        if signals.shape[0] > signals.shape[1]:
            signals = signals.T
        write_canonical(out, signals)
    elif suf in (".pkl", ".pickle"):
        with open(inp, "rb") as f:
            obj = pickle.load(f)
        # try known keys
        cand = None
        if isinstance(obj, dict):
            for key in ("signals","eeg","X","data"):
                if key in obj:
                    try:
                        cand = np.asarray(obj[key])
                        break
                    except Exception:
                        continue
        if cand is None:
            raise SystemExit("No signals found in pickle")
        signals = cand
        if signals.ndim == 2 and signals.shape[0] > signals.shape[1]:
            signals = signals.T
        write_canonical(out, signals, sfreq=obj.get("sfreq") if isinstance(obj, dict) else None)
    elif suf in (".npy", ".npz"):
        arr = np.load(str(inp), allow_pickle=True)
        signals = np.asarray(arr)
        if signals.ndim == 2 and signals.shape[0] > signals.shape[1]:
            signals = signals.T
        write_canonical(out, signals)
    else:
        raise SystemExit("Unsupported input type")

if __name__ == "__main__":
    main()

