"""
wesad_preprocess_fixed.py
=========================

Fixed variant of ``wesad_preprocess.py``.

Fixes applied (relative to the faithful ``_forked`` port):

1. **ACC FIR filter now filters along time** (``axis=0``) instead of across the
   x/y/z channels (``axis=-1`` default in the original).
2. **FIR cutoff is normalized with the actual sampling rate** of each ACC signal
   (700 Hz for chest, 32 Hz for wrist) instead of always using 32 Hz.
3. **``labels_at_rate`` uses float64/int64 explicitly** to avoid the Windows
   int32 overflow (already present in the faithful port, kept here).

Everything else (EDA Butterworth + phasic decomposition, passthrough signals,
binary relabeling, output structure) is identical to ``wesad_preprocess.py``.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import scipy.signal as scisig

# --------------------------------------------------------------------------------------
# Paths & constants (same as wesad_preprocess.py)
# --------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
WESAD_DIR = REPO_ROOT / "WESAD"
OUTPUT_DIR = REPO_ROOT / "data" / "preprocessed" / "WESAD_fixed"

SUBJECT_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

FS = {
    "chest": {"ACC": 700, "ECG": 700, "EDA": 700, "EMG": 700, "Resp": 700, "Temp": 700},
    "wrist": {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4},
    "label": 700,
}
fs_dict = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4, "label": 700,
           "Resp": 700, "ECG": 700, "chest": 700}

WESAD_LABELS = {0: "transient", 1: "baseline", 2: "stress", 3: "amusement",
                4: "meditation", 5: "ignore", 6: "ignore", 7: "ignore"}
BINARY_MAP = {1: 0, 3: 0, 2: 1}
BINARY_NAMES = {0: "non-stress", 1: "stress"}
EXCLUDED = -1

# --------------------------------------------------------------------------------------
# Stage 1 - Loading (identical to wesad_preprocess.py)
# --------------------------------------------------------------------------------------
def load_subject(subject_id: int, wesad_dir: Path | str = WESAD_DIR) -> dict:
    name = f"S{subject_id}"
    path = Path(wesad_dir) / name / f"{name}.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data

def describe_subject(data: dict) -> list[dict]:
    rows = []
    for loc in ("chest", "wrist"):
        for key, arr in data["signal"][loc].items():
            arr = np.asarray(arr)
            fs = FS[loc][key]
            rows.append({"location": loc, "signal": key, "shape": arr.shape,
                         "fs_hz": fs, "duration_s": round(arr.shape[0] / fs, 1)})
    lab = np.asarray(data["label"])
    rows.append({"location": "-", "signal": "label", "shape": lab.shape,
                 "fs_hz": FS["label"], "duration_s": round(lab.shape[0] / FS["label"], 1)})
    return rows

# --------------------------------------------------------------------------------------
# Stage 2 - Filtering (FIXED)
# --------------------------------------------------------------------------------------
def butter_lowpass(cutoff: float, fs: float, order: int = 5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scisig.butter(order, normal_cutoff, btype="low", analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff: float, fs: float, order: int = 5) -> np.ndarray:
    b, a = butter_lowpass(cutoff, fs, order=order)
    return scisig.lfilter(b, a, np.asarray(data, dtype=float))

def filter_signal_fir(signal, cutoff: float = 0.4, numtaps: int = 64, fs: float = 32.0) -> np.ndarray:
    """FIR low-pass — FIXED version.

    Fixes vs ``wesad_preprocess.filter_signal_fir``:
    - ``fs`` is now a parameter (use the actual sampling rate of the signal).
    - ``lfilter`` is called with ``axis=0`` so a 2-D (N, 3) ACC array is
      filtered along time, not across the x/y/z channels.
    """
    f = cutoff / (fs / 2.0)
    fir_coeff = scisig.firwin(numtaps, f)
    return scisig.lfilter(fir_coeff, 1, np.asarray(signal, dtype=float), axis=0)

def eda_decompose(eda, fs: float) -> dict:
    import neurokit2 as nk

    eda = np.asarray(eda, dtype=float).ravel()
    eda_f = butter_lowpass_filter(eda, cutoff=1.0, fs=fs, order=6)
    phasic = nk.eda_phasic(nk.standardize(eda_f), sampling_rate=fs)
    return {"EDA": eda_f,
            "EDA_SCR": np.asarray(phasic["EDA_Phasic"]),
            "EDA_SCL": np.asarray(phasic["EDA_Tonic"])}

def filter_subject(data: dict) -> dict:
    """Apply the FIXED preprocessing to every sensor of one subject."""
    out = {"chest": {}, "wrist": {}}
    for loc in ("chest", "wrist"):
        for key, arr in data["signal"][loc].items():
            fs = FS[loc][key]
            arr = np.asarray(arr, dtype=float)
            if key == "EDA":
                out[loc].update(eda_decompose(arr, fs))
            elif key == "ACC":
                out[loc]["ACC"] = filter_signal_fir(arr, fs=fs)
            else:
                out[loc][key] = arr.ravel() if arr.ndim == 2 and arr.shape[1] == 1 else arr
    return out

# --------------------------------------------------------------------------------------
# Stage 3 - Labeling & saving (identical to wesad_preprocess.py)
# --------------------------------------------------------------------------------------
def relabel_binary(labels) -> np.ndarray:
    labels = np.asarray(labels).ravel()
    out = np.full(labels.shape, EXCLUDED, dtype=np.int8)
    for src, dst in BINARY_MAP.items():
        out[labels == src] = dst
    return out

def labels_at_rate(labels_700: np.ndarray, fs: float, n_samples: int) -> np.ndarray:
    idx = np.floor(np.arange(n_samples, dtype=np.float64) * FS["label"] / fs).astype(np.int64)
    idx = np.clip(idx, 0, len(labels_700) - 1)
    return labels_700[idx]

def build_clean_subject(data: dict, filtered: dict, drop_excluded: bool = True) -> dict:
    bin700 = relabel_binary(data["label"])
    clean = {"subject": data.get("subject", "unknown"),
             "fs": {"chest": {}, "wrist": {}},
             "signal": {"chest": {}, "wrist": {}},
             "label": {"chest": {}, "wrist": {}},
             "label_names": BINARY_NAMES,
             "drop_excluded": drop_excluded}
    for loc in ("chest", "wrist"):
        for name, arr in filtered[loc].items():
            base = "EDA" if name.startswith("EDA") else name
            fs = FS[loc][base]
            lab = labels_at_rate(bin700, fs, len(arr))
            if drop_excluded:
                keep = lab != EXCLUDED
                arr, lab = arr[keep], lab[keep]
            clean["fs"][loc][name] = fs
            clean["signal"][loc][name] = arr
            clean["label"][loc][name] = lab
    return clean

def save_subject(clean: dict, output_dir: Path | str = OUTPUT_DIR) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{clean['subject']}_clean.pkl"
    with open(path, "wb") as f:
        pickle.dump(clean, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path

def load_clean_subject(subject_id: int, output_dir: Path | str = OUTPUT_DIR) -> dict:
    with open(Path(output_dir) / f"S{subject_id}_clean.pkl", "rb") as f:
        return pickle.load(f)

def preprocess_subject(subject_id: int, drop_excluded: bool = True, save: bool = True) -> dict:
    data = load_subject(subject_id)
    filtered = filter_subject(data)
    clean = build_clean_subject(data, filtered, drop_excluded=drop_excluded)
    if save:
        save_subject(clean)
    return clean
