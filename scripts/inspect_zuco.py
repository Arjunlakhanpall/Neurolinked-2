#!/usr/bin/env python3
"""
inspect_zuco.py

Quickly inspect ZuCo/EEG files (.edf, .mat, .pkl, .npy) and print shapes / metadata.
"""
from pathlib import Path
import argparse
import pickle
import numpy as np

def inspect_edf(path):
    import mne
    raw = mne.io.read_raw_edf(str(path), preload=False, verbose=False)
    print("EDF:", path)
    print("  n_channels:", raw.info["nchan"])
    print("  n_samples:", raw.n_times)
    print("  sfreq:", raw.info["sfreq"])
    print("  duration (s):", raw.n_times / raw.info["sfreq"])
    print("  channels:", raw.ch_names[:20])

def inspect_mat(path):
    from scipy.io import loadmat
    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    print("MAT:", path)
    # list candidate arrays
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        try:
            arr = np.array(v)
            if arr.ndim in (1,2) and arr.size > 0:
                print(f"  {k}: shape={arr.shape}")
        except Exception:
            continue

def inspect_pkl(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print("PKL:", path, "type:", type(obj).__name__)
    if isinstance(obj, dict):
        for k in ("signals","eeg","X","data","raw"):
            if k in obj:
                try:
                    arr = np.array(obj[k])
                    print(f"  key:{k} shape={arr.shape}")
                except Exception:
                    print(f"  key:{k} type={type(obj[k])}")

def inspect_npy(path):
    arr = np.load(str(path), allow_pickle=True)
    print("NPY/NPZ:", path, "shape:", getattr(arr, "shape", None), "dtype:", getattr(arr, "dtype", None))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", "-p", required=True)
    args = p.parse_args()
    path = Path(args.path)
    if not path.exists():
        raise SystemExit("Path not found")
    if path.is_dir():
        files = list(path.rglob("*"))
        print("Directory:", path, "files:", len(files))
        for f in files[:10]:
            try:
                inspect_file(f)
            except Exception as e:
                print("skip", f, e)
        return
    inspect_file(path)

def inspect_file(path):
    suf = path.suffix.lower()
    if suf == ".edf":
        inspect_edf(path)
    elif suf == ".mat":
        inspect_mat(path)
    elif suf in (".pkl", ".pickle"):
        inspect_pkl(path)
    elif suf in (".npy", ".npz"):
        inspect_npy(path)
    else:
        print("Unknown file type:", path)

if __name__ == "__main__":
    main()

