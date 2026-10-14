# WESAD Stress Detection — Skripsi Repository

Wearable stress detection using the [WESAD dataset](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection). The pipeline goes from raw physiological signals → preprocessing → feature extraction → ML model training (Random Forest, SVM, XGBoost, KNN).

---

## Repository Structure

```
repo-skripsi/
│
├── WESAD/                  # Raw WESAD dataset (S2–S17 .pkl files)
│
├── Data/
│   ├── preprocessed/
│   │   ├── WESAD/              # Cleaned signals  (S{id}_clean.pkl)
│   │   └── WESAD_windowed/     # Windowed segments (S{id}_windows.pkl)
│   ├── features_a/             # Features from pipeline A (sliding window)
│   └── features_b/             # Features from pipeline B (forked reference)
│
├── Scripts/
│   ├── requirements.txt        # Python dependencies
│   ├── environment.yml         # Conda environment spec
│   ├── Preprocess/             # Stage 0–4: raw → cleaned → windowed
│   └── Processing/             # Stage 5:  windowed → feature CSVs
│
├── model_build.ipynb           # Stage 6: train & evaluate ML models
│
├── _forked/                    # Original upstream code (reference only)
│   └── fb_code/                # Core library from the forked repo
│
├── skripsi-venv/               # Python virtual environment (git-ignored)
└── PDF/                        # Proposal / thesis documents
```

---

## Pipeline Overview

| Stage | Location | What it does |
|-------|----------|--------------|
| **0** | `Scripts/Preprocess/00_overview.ipynb` | Documents the full preprocessing plan, sensors, and output format |
| **1** | `Scripts/Preprocess/01_load_and_inspect.ipynb` | Load raw `.pkl`, inspect shapes, sampling rates, plot raw signals |
| **2** | `Scripts/Preprocess/02_filter_signals.ipynb` | Per-sensor filtering (Butterworth LP for EDA, FIR LP for ACC, etc.) |
| **3** | `Scripts/Preprocess/03_label_and_save.ipynb` | Binary relabel (stress vs. non-stress), drop unused labels, save `S{id}_clean.pkl` |
| **4** | `Scripts/Preprocess/04_windowing.ipynb` | Sliding-window segmentation (60 s general, 5 s ACC, 0.25 s shift) |
| **5A** | `Scripts/Processing/05A_feature_extraction_sliding.ipynb` | Extract features (ECG, EDA, ACC, EMG, Resp, Temp, BVP) → `Data/features_a/` |
| **5B** | `Scripts/Processing/05B_feature_extraction_forked.ipynb` | Fork-compatible feature extraction for comparison → `Data/features_b/` |
| **6** | `model_build.ipynb` | Combine features, train RF / SVM / XGBoost / KNN, evaluate |

---

## Where to Start

1. **Set up the environment**
   ```bash
   python -m venv skripsi-venv
   skripsi-venv\Scripts\activate      # Windows
   pip install -r Scripts/requirements.txt
   ```

2. **Place the WESAD dataset** in `WESAD/` (each subject folder `S2/`, `S3/`, … `S17/` containing its `.pkl` file).

3. **Run the preprocessing notebooks** in order:
   - Start with `Scripts/Preprocess/00_overview.ipynb` to understand the plan.
   - Then run `01` → `02` → `03` → `04` sequentially.

4. **Extract features** — pick one or both:
   - `Scripts/Processing/05A_feature_extraction_sliding.ipynb` (primary pipeline)
   - `Scripts/Processing/05B_feature_extraction_forked.ipynb` (fork-compatible reference)

5. **Train models** — open `model_build.ipynb` and run all cells.

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `neurokit2` | EDA decomposition (SCR / SCL) |
| `heartpy` | ECG-based heart rate features |
| `biosppy` | Respiration rate extraction |
| `scikit-learn` | ML models & evaluation |
| `xgboost` | Gradient-boosted trees |
| `scipy` | Signal filtering |

See `Scripts/requirements.txt` for the full list.

---

## Notes

- **Python version**: use the venv at `skripsi-venv/` (NumPy 1.23, Windows int32 overflow guard — use `dtype=np.int64` for large index arrays).
- **`_forked/`** is the original upstream repository kept for reference. The cleaned pipeline lives in `Scripts/`.
- Preprocessed data files (`*.pkl`, `*.csv`) are git-ignored; regenerate them by running the notebooks.
