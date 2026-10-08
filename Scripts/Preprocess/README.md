# WESAD Preprocessing

Turns synchronized raw WESAD files (`WESAD/S*/S*.pkl`) into cleaned, binary-labelled signals using the methods from `_forked`.

Run in order:

| # | Notebook | Purpose | `_forked` source |
|---|---|---|---|
| 0 | `00_overview.ipynb` | Map of stages, sensors, output format, note on `_forked` quirks | — |
| 1 | `01_load_and_inspect.ipynb` | Load a subject, shapes / sampling rates / labels, raw plots | `fb_code/feature_extraction.py` → `SubjectData` |
| 2 | `02_filter_signals.ipynb` | Per-sensor filtering with before/after plots | `fb_code/feature_extraction.py` → `compute_features`; `fb_code/utils.py` |
| 3 | `03_label_and_save.ipynb` | Binary relabel, drop excluded samples, save all subjects | `fb_code/feature_extraction.py` → `label_dict` |
| 4 | `04_windowing.ipynb` | Fixed sliding window segmentation (60s general, 5s ACC, 0.25s shift) | `fb_code/feature_extraction.py` → `get_samples` |

Helper module: `wesad_preprocess.py`.

## Per-sensor processing

| Sensor | Processing |
|---|---|
| EDA (chest, wrist) | Butterworth LP 1 Hz, order 6 → standardize → `nk.eda_phasic` (SCR/SCL) |
| ACC (chest, wrist) | FIR LP 0.4 Hz, 64 taps (as in `_forked`) |
| ECG, EMG, Resp, Temp (chest), BVP, TEMP (wrist) | passthrough (as in `_forked`) |

## Faithfulness to `_forked`

The pipeline reproduces the original `_forked` preprocessing exactly (same filters, same parameters, same causal `lfilter` calls). Known quirks in the original code are kept as-is and documented in the note at the bottom of `00_overview.ipynb`.

## Labels

baseline (1) + amusement (3) → **0 non-stress**, stress (2) → **1 stress**, everything else dropped.

## Output

`data/preprocessed/WESAD/S{id}_clean.pkl` (git-ignored). See `00_overview.ipynb` for the structure.
