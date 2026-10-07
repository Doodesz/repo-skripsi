"""
wesad_preprocess.py
===================

Helper module for preprocessing the WESAD dataset (synchronized ``S*.pkl``).

Every function here is adapted from the ``_forked`` repository, mainly:

- ``_forked/fb_code/utils.py``               -> fs_dict, butter_lowpass(_filter), filterSignalFIR
- ``_forked/fb_code/feature_extraction.py``  -> SubjectData, compute_features (filter order / EDA decomposition)
- ``_forked/data_wrangling.py``              -> older copy of the same logic

Faithfulness to the original code:

This module is a FAITHFUL port of the ``_forked`` preprocessing: the same filters, the same
parameters, and the same (causal) ``lfilter`` calls are used, so the output matches what
``_forked`` produced. See the note at the bottom of ``00_overview.ipynb`` for the known
quirks that were intentionally kept.

Signals that ``_forked`` does NOT filter at the preprocessing stage (ECG, Resp, BVP, TEMP, EMG)
are passed through unchanged; they are filtered internally later by heartpy / biosppy during
feature extraction (see ``_forked/fb_code/ecg.py`` and ``_forked/fb_code/respiration.py``).
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import scipy.signal as scisig

# --------------------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
WESAD_DIR = REPO_ROOT / "WESAD"
OUTPUT_DIR = REPO_ROOT / "data" / "preprocessed" / "WESAD"

SUBJECT_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

# Sampling frequencies (Hz) - official WESAD readme / _forked/fb_code/utils.py
FS = {
    "chest": {"ACC": 700, "ECG": 700, "EDA": 700, "EMG": 700, "Resp": 700, "Temp": 700},
    "wrist": {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4},
    "label": 700,
}
# Flat dict kept for compatibility with _forked code
fs_dict = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4, "label": 700,
           "Resp": 700, "ECG": 700, "chest": 700}

# Original WESAD label ids
WESAD_LABELS = {0: "transient", 1: "baseline", 2: "stress", 3: "amusement",
                4: "meditation", 5: "ignore", 6: "ignore", 7: "ignore"}
# Binary target used in this thesis
BINARY_MAP = {1: 0, 3: 0, 2: 1}          # baseline/amusement -> 0 (non-stress), stress -> 1
BINARY_NAMES = {0: "non-stress", 1: "stress"}
EXCLUDED = -1


# --------------------------------------------------------------------------------------
# Stage 1 - Loading
# --------------------------------------------------------------------------------------
def load_subject(subject_id: int, wesad_dir: Path | str = WESAD_DIR) -> dict:
    """Load ``S{id}.pkl`` (same as ``SubjectData.__init__`` in _forked).

    Returns the raw dict with keys ``signal`` (``chest``/``wrist``), ``label``, ``subject``.
    """
    name = f"S{subject_id}"
    path = Path(wesad_dir) / name / f"{name}.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


def describe_subject(data: dict) -> list[dict]:
    """Return a list of rows (location, signal, shape, fs, duration_s) for quick inspection."""
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
# Stage 2 - Filtering (from _forked/fb_code/utils.py + compute_features)
# --------------------------------------------------------------------------------------
def butter_lowpass(cutoff: float, fs: float, order: int = 5):
    """Butterworth low-pass coefficients (identical to _forked)."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scisig.butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff: float, fs: float, order: int = 5) -> np.ndarray:
    """Apply Butterworth low-pass (identical to _forked: causal ``lfilter``)."""
    b, a = butter_lowpass(cutoff, fs, order=order)
    return scisig.lfilter(b, a, np.asarray(data, dtype=float))


def filter_signal_fir(signal, cutoff: float = 0.4, numtaps: int = 64) -> np.ndarray:
    """FIR low-pass, identical to ``utils.filterSignalFIR`` in _forked.

    NOTE: the cutoff is normalized with ``fs_dict['ACC']`` (32 Hz) regardless of the actual
    sampling rate, and ``lfilter`` is called without ``axis`` (so a 2-D input is filtered
    across the last axis). Both quirks are kept on purpose - see the note at the bottom of
    ``00_overview.ipynb``.
    Original reference: MIT eda-explorer AccelerometerFeatureExtractionScript.py
    """
    f = cutoff / (fs_dict["ACC"] / 2.0)
    fir_coeff = scisig.firwin(numtaps, f)
    return scisig.lfilter(fir_coeff, 1, np.asarray(signal, dtype=float))


def eda_decompose(eda, fs: float) -> dict:
    """EDA preprocessing as in ``compute_features``:

    1. Butterworth low-pass, cutoff 1 Hz, order 6
    2. ``nk.standardize``
    3. ``nk.eda_phasic`` -> SCR (phasic) and SCL (tonic)

    Returns dict with ``EDA`` (filtered, NOT standardized), ``EDA_SCR`` and ``EDA_SCL``.
    """
    import neurokit2 as nk  # imported lazily (slow import)

    eda = np.asarray(eda, dtype=float).ravel()
    eda_f = butter_lowpass_filter(eda, cutoff=1.0, fs=fs, order=6)
    phasic = nk.eda_phasic(nk.standardize(eda_f), sampling_rate=fs)
    return {"EDA": eda_f,
            "EDA_SCR": np.asarray(phasic["EDA_Phasic"]),
            "EDA_SCL": np.asarray(phasic["EDA_Tonic"])}


def filter_subject(data: dict) -> dict:
    """Apply the _forked preprocessing to every sensor of one subject.

    | Sensor           | Processing                                   |
    |------------------|----------------------------------------------|
    | chest/wrist EDA  | Butter LP 1 Hz (order 6) + SCR/SCL split     |
    | chest/wrist ACC  | FIR LP 0.4 Hz, 64 taps (as in _forked)       |
    | ECG, Resp, EMG, Temp (chest), BVP, TEMP (wrist) | passthrough   |

    Returns ``{'chest': {...}, 'wrist': {...}}`` with 1-D arrays (ACC stays (N, 3)).
    """
    out = {"chest": {}, "wrist": {}}
    for loc in ("chest", "wrist"):
        for key, arr in data["signal"][loc].items():
            fs = FS[loc][key]
            arr = np.asarray(arr, dtype=float)
            if key == "EDA":
                out[loc].update(eda_decompose(arr, fs))
            elif key == "ACC":
                out[loc]["ACC"] = filter_signal_fir(arr)
            else:
                out[loc][key] = arr.ravel() if arr.ndim == 2 and arr.shape[1] == 1 else arr
    return out


# --------------------------------------------------------------------------------------
# Stage 3 - Labeling & saving
# --------------------------------------------------------------------------------------
def relabel_binary(labels) -> np.ndarray:
    """Map WESAD labels to binary: stress(2)->1, baseline(1)/amusement(3)->0, else -1."""
    labels = np.asarray(labels).ravel()
    out = np.full(labels.shape, EXCLUDED, dtype=np.int8)
    for src, dst in BINARY_MAP.items():
        out[labels == src] = dst
    return out


def labels_at_rate(labels_700: np.ndarray, fs: float, n_samples: int) -> np.ndarray:
    """Resample a 700 Hz label vector to a signal of rate ``fs`` with ``n_samples`` samples
    (nearest-index lookup, signals are already synchronized in WESAD)."""
    # NOTE: use float64/int64 explicitly - on Windows NumPy's default int is int32 and
    # np.arange(n) * 700 overflows for n > ~3.07M samples (chest signals have ~4M+).
    idx = np.floor(np.arange(n_samples, dtype=np.float64) * FS["label"] / fs).astype(np.int64)
    idx = np.clip(idx, 0, len(labels_700) - 1)
    return labels_700[idx]


def build_clean_subject(data: dict, filtered: dict, drop_excluded: bool = True) -> dict:
    """Attach a binary label vector to every signal (at that signal's own rate) and
    optionally drop samples whose label is excluded (transient, meditation, ignore).

    Output structure::

        {
          'subject': 'S2',
          'fs':      {'chest': {...}, 'wrist': {...}},
          'signal':  {'chest': {name: array}, 'wrist': {name: array}},
          'label':   {'chest': {name: array}, 'wrist': {name: array}},   # binary 0/1 (or -1)
          'label_names': {0: 'non-stress', 1: 'stress'},
        }
    """
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
    """Save cleaned subject dict to ``data/preprocessed/WESAD/S{id}_clean.pkl``."""
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
    """Full pipeline for one subject: load -> filter -> binary label -> (save)."""
    data = load_subject(subject_id)
    filtered = filter_subject(data)
    clean = build_clean_subject(data, filtered, drop_excluded=drop_excluded)
    if save:
        save_subject(clean)
    return clean
