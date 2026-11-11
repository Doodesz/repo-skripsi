---
tags:
  - skripsi/research-log
  - machine-learning
  - sensor-modalities
  - explainability
aliases:
  - Sensor Modality Contribution Analysis
---
# Research Note — Sensor Modality Contribution Analysis

## Context

The STAI 6-second reference-model workflow uses multimodal physiological features from 15 WESAD subjects. The target is a binary, participant-relative STAI label: `1` when a non-Base condition's STAI mean is at least `0.2` above that participant's Base mean; otherwise `0`. See [[Pipeline STAI User (v2_stai)]] and [[Data Artefacts Overview]].

The main training notebook is:

- `Scripts/4th try/stai_6s_reference_models.ipynb`
- Standardized duplicate: `Scripts/4th try/stai_6s_reference_models_stdized.ipynb`
- Analysis notebook: `Scripts/4th try/analyze_sensor_modality_weight.ipynb`

## Model artifacts

The reference workflow trains:

- Random Forest
- XGBoost
- SVM
- KNN

The standardized duplicate applies a leakage-safe pipeline:

```text
StandardScaler -> MinMaxScaler -> classifier
```

The scalers are fitted within each training split and are saved with the model pipeline. Standardized models are stored separately under `Models/stai_6s_standardized/`; original model artifacts are not overwritten. This is relevant because SVM and KNN are sensitive to feature scale, while tree models generally are not. See [[Machine Learning Models & Evaluation]].

## Initial modality-importance attempt

The first analysis aggregated native feature importance or permutation importance by modality. This showed ACC as the leading modality for all models, with KNN reaching approximately 99%.

That result was not sufficient as a scientific conclusion because the feature counts are highly imbalanced:

| Modality | Feature count |
| -------- | ------------: |
| ACC      |            82 |
| EDA      |            28 |
| TEMP     |            12 |
| ECG      |            11 |
| RESP     |            10 |
| EMG      |             6 |
| BVP      |             5 |

All modalities are available for the same windows; ACC does **not** have more windows. The bias comes from ACC contributing many more feature columns, possible feature-scale effects, and KNN's sensitivity to unscaled distances. Therefore, a raw block-level importance percentage must not be interpreted as intrinsic physiological responsibility.

## Decision-contribution method

The analysis notebook was changed to a per-window modality ablation method:

1. Predict each available window using all modality features.
2. Replace one modality's features with that modality's all-window median.
3. Predict the same windows again.
4. Measure absolute change in the model's class-1 decision score.
5. Measure the rate at which the final predicted label changes.
6. Report overall results and results grouped by predicted label (`0` / No Anxiety and `1` / Anxiety).

The primary metrics are:

- `mean_absolute_decision_change`
- `prediction_change_rate`
- `relative_contribution`

All models use the same complete window matrix. Results are saved under:

- `Data/Analysis/sensor_modality_decision_contribution/`

## ACC-excluded comparison

To test whether ACC was dominating because of its feature-block size or predictive signal, the notebook now includes a separate ACC-excluded section.

This section:

- Removes all ACC-derived features, including `ACC_sub_*`, `ACC_*`, and `net_acc_*`.
- Keeps the same complete set of windows and labels.
- Retrains separate standardized/normalized models for Random Forest, XGBoost, SVM, and KNN.
- Repeats the same per-window decision-ablation analysis on the remaining modalities.
- Saves the reduced-feature models under `Models/stai_6s_standardized_acc_excluded/`.
- Saves results under `Data/Analysis/sensor_modality_decision_contribution_acc_excluded/`.

The ACC-excluded comparison is necessary because the original models cannot accept a reduced feature matrix. It is not valid to pass ACC-removed data directly into models trained with the original feature count.

## Visualization

The analysis notebook generates horizontal bar charts for every original and ACC-excluded model. The bars show relative contribution to decision change, with percentage labels. PNG outputs are saved alongside the CSV outputs in the two analysis directories above.

## Interpretation and limitations

- ACC being ranked first by every model may reflect real signal, correlated ACC features, the larger ACC feature block, or interactions between these factors.
- The ACC-excluded experiment tests the effect of removing the entire modality, but it does not prove that the remaining modalities are independently causal.
- The decision-ablation baseline uses modality medians; this creates synthetic windows and should be described as a model-explanation perturbation, not a physiological intervention.
- Results computed on all windows describe model behavior on the available dataset, not generalization to unseen subjects.
- Subject-grouped validation or LOSO remains essential because overlapping windows from one subject are not independent. See [[Machine Learning Models & Evaluation]].
- The STAI label is subjective and baseline-relative, so modality contribution should be interpreted as contribution to the trained classifier's label decision, not clinical anxiety causation.

## Reproducibility

1. Run `stai_6s_reference_models_stdized.ipynb` to regenerate standardized model pipelines.
2. Open `analyze_sensor_modality_weight.ipynb`.
3. Run the shared loading/functions cells.
4. Run the four original standardized-model cells.
5. Run the ACC-excluded setup cell.
6. Run the four ACC-excluded model cells.
7. Inspect the CSV tables and bar plots in the two analysis output directories.

## Related notes

- [[Machine Learning Models & Evaluation]]
- [[Pipeline STAI User (v2_stai)]]
- [[Data Artefacts Overview]]
- [[Katalog Fitur Fisiologis]]
- [[Development Findings 1]]
- [[MOC - Skripsi Repository]]

## Reduced-Modality Notebook Lineage

This section records the notebook work completed in the conversation on 2026-11-05. The goal was to reproduce the STAI 60-second / 6-second-shift modeling workflow while progressively restricting the sensor modalities to those available from the Samsung smartwatch used in the study. The smartwatch compatibility context is documented in [[2026-10-29 Research Note]], especially the BVP/ECG permission findings.

### 1. Sliced smartwatch-compatible pipeline

A reduced notebook was created from `Scripts/4th try/stai_6s_reference_models.ipynb`:

- [[../Scripts/4th try/Sliced_sensor_modals/stai_6s_sliced_sensor_reference_models.ipynb|stai_6s_sliced_sensor_reference_models.ipynb]]

The retained WESAD channels are:

- Wrist ACC
- Wrist EDA
- Wrist TEMP

BVP and ECG were excluded because they were not available in the smartwatch data collection. Chest-only channels were also excluded: chest ACC, chest ECG, chest EDA, EMG, respiration, and chest temperature. The notebook uses separate sliced-data and model locations:

- `Data/Preprocessed n STAI-labeled 6s sliced sensor modalities`
- `Models/stai_6s_sliced_sensor_modalities`

Copied execution outputs were cleared and all code cells passed Python syntax validation.

### 2. Standardized and normalized sliced pipeline

A second notebook was created from the newer 5th-try pipeline `stai_6s_reference_models_stdized.ipynb`, which adds leakage-safe model preprocessing:

```text
StandardScaler -> MinMaxScaler -> classifier
```

The sliced version is:

- [[../Scripts/5th try/Sliced_sensor_modals/stai_6s_std_norm_sliced_sensor.ipynb|stai_6s_std_norm_sliced_sensor.ipynb]]

It preserves the wrist ACC, wrist EDA, and wrist TEMP slice, excludes BVP/ECG and chest-only channels, and uses:

- `Data/Preprocessed n STAI-labeled 6s std-norm-ed sliced sensor modalities`
- `Models/stai_6s_standardized_sliced_sensor_modalities`

Its generated feature files use the `_std_norm_sliced.pkl` suffix. Execution outputs were cleared and the notebook passed Python syntax validation.

### 3. Minimum ACC + EDA pipeline

The final requested reduction keeps only the two sensor modalities needed for the minimum-sensor experiment:

- Wrist ACC
- Wrist EDA

Wrist TEMP, BVP, ECG, and every chest-only modality are excluded. The notebook is:

- [[../Scripts/5th try/Minimum_sensor_modals/stai_6s_std_norm_minimum_sensor.ipynb|stai_6s_std_norm_minimum_sensor.ipynb]]

It retains the 5th-try standardized/normalized model behavior and writes to:

- `Data/Preprocessed n STAI-labeled 6s norm-minimum sensor modalities`
- `Models/stai_6s_norm_minimum_sensor`

The generated feature files use the `_norm_minimum.pkl` suffix. The notebook was initially generated with an orphaned line from the removed ECG feature loop:

```python
wdf[f'ECG_{key}'] = value
```

Because `key` and `value` were undefined in the ACC/EDA-only loop, this caused a runtime error during feature extraction. The line was removed, after which all notebook code cells passed Python syntax validation.

### Reproducibility and interpretation

These notebooks preserve the same STAI-baseline-relative target definition, 60-second windows, 6-second shifts, and ACC 5-second sub-window aggregation unless otherwise stated in the notebook. Reduced-modality accuracy must be compared using the same subject-independent/LOSO protocol and interpreted with class-sensitive metrics, not accuracy alone. See [[Machine Learning Models & Evaluation]], [[Pipeline STAI User (v2_stai)]], [[Katalog Fitur Fisiologis]], and [[Data Artefacts Overview]].

### Related research notes

- [[2026-10-29 Research Note]] — Samsung smartwatch sensor availability and BVP/ECG access findings.
- [[Pipeline STAI User (v2_stai)]] — STAI parsing, baseline-relative labels, and preprocessing conventions.
- [[Pra-Pemrosesan Sinyal & Filtering]] — signal filtering and feature-preprocessing context.
- [[Wrangling & Sinkronisasi Multimodal]] — multimodal synchronization context.
- [[Machine Learning Models & Evaluation]] — model evaluation and interpretation.
- [[Data Artefacts Overview]] — generated data and model artifact locations.
- [[MOC - Skripsi Repository]] — repository-wide note index.

## Difference Between Full-Modality and Reduced-Modality Pipelines

The two reduced notebooks are not different labeling or evaluation experiments; they are controlled feature-space reductions of the standardized full-modality pipeline. The comparison below describes what changes and what must remain constant when interpreting performance differences.

### Common pipeline behavior

All three 5th-try notebooks retain the same core protocol:

- WESAD subjects: `S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S13, S14, S15, S16, S17`.
- 60-second windows shifted every 6 seconds.
- STAI baseline-relative binary labels with the `0.2` increase threshold.
- Windows retained only when fully contained within one timed condition.
- Wrist ACC kept at 32 Hz; non-ACC retained signals resampled to 70 Hz and aligned to the 700 Hz label clock.
- ACC-specific 5-second sub-windows shifted by 0.25 seconds.
- The same four classifiers: Random Forest, SVM, XGBoost, and KNN.
- Leakage-safe model preprocessing: `StandardScaler -> MinMaxScaler -> classifier`, fitted within each training split.

Therefore, a difference in model results should primarily be attributed to the removed sensor modalities and their derived features, not to changed labels, window boundaries, or scaling policy.

### Full-modality reference

The full pipeline is [[../Scripts/5th try/stai_6s_reference_models_stdized.ipynb|stai_6s_reference_models_stdized.ipynb]]. It uses the complete WESAD signal set:

| Sensor group | Included channels |
|---|---|
| Wrist | ACC, BVP, EDA, TEMP |
| Chest | ECG, EDA, ACC, EMG, Resp, Temp |

Its feature matrix contains wrist and chest signal statistics plus custom modality features, including ECG heart-rate statistics, BVP peak frequency, respiration features, EMG features, temperature features, EDA slope/dynamic range, net acceleration, axis-level ACC features, and ACC sub-window aggregates. The full pipeline's ACC block includes both wrist and chest ACC axes.

It writes standardized data to `Data/Preprocessed n STAI-labeled 6s std-norm-ed` and model pipelines to `Models/stai_6s_standardized`.

### Reduced pipeline 1: smartwatch-compatible sliced modalities

Notebook: [[../Scripts/5th try/Sliced_sensor_modals/stai_6s_std_norm_sliced_sensor.ipynb|stai_6s_std_norm_sliced_sensor.ipynb]]

Retained modalities:

- Wrist ACC
- Wrist EDA
- Wrist TEMP

Removed relative to full modality:

- Wrist BVP
- Chest ECG
- Chest ACC
- Chest EDA
- Chest EMG
- Chest respiration
- Chest temperature

Consequently, this notebook keeps wrist ACC-derived features, wrist EDA raw/phasic/tonic and custom features, and wrist TEMP features. It removes the ECG, BVP, respiration, EMG, chest EDA, chest temperature, and chest-ACC feature families. Its ACC sub-window aggregates cover only wrist ACC axes, rather than both wrist and chest ACC.

This is the closest reduced experiment to the actual smartwatch sensor availability. It tests whether the available wrist sensor set can reproduce the full model's behavior without relying on unavailable BVP/ECG or chest sensors.

Artifacts:

- Data: `Data/Preprocessed n STAI-labeled 6s std-norm-ed sliced sensor modalities`
- Models: `Models/stai_6s_standardized_sliced_sensor_modalities`
- Feature files: `*_std_norm_sliced.pkl`

### Reduced pipeline 2: minimum ACC + EDA modalities

Notebook: [[../Scripts/5th try/Minimum_sensor_modals/stai_6s_std_norm_minimum_sensor.ipynb|stai_6s_std_norm_minimum_sensor.ipynb]]

Retained modalities:

- Wrist ACC
- Wrist EDA

Removed relative to the full pipeline:

- Wrist BVP
- Wrist TEMP
- Chest ECG
- Chest ACC
- Chest EDA
- Chest EMG
- Chest respiration
- Chest temperature

Relative to the sliced smartwatch-compatible notebook, the only additional removal is wrist TEMP. The minimum model therefore uses wrist ACC features, wrist net acceleration and ACC sub-window aggregates, and wrist EDA statistics/custom features. It has no temperature-derived features and no unavailable BVP/ECG/chest feature families.

This is a sensor-cost and deployment-minimum experiment rather than a complete reconstruction of the smartwatch capability list. It tests how much performance is retained using only the two selected modalities, and whether wrist EDA adds predictive value beyond the movement-related ACC signal.

Artifacts:

- Data: `Data/Preprocessed n STAI-labeled 6s norm-minimum sensor modalities`
- Models: `Models/stai_6s_norm_minimum_sensor`
- Feature files: `*_norm_minimum.pkl`

### Practical interpretation

The expected information hierarchy is:

```text
Full modality
  └─ removes unavailable BVP/ECG/chest channels
     └─ sliced smartwatch-compatible: wrist ACC + EDA + TEMP
        └─ minimum sensor: wrist ACC + EDA
```

The sliced-to-minimum comparison isolates the contribution of wrist TEMP. The full-to-sliced comparison combines two effects: removal of all chest modalities and BVP/ECG, while retaining the three wrist modalities available in the smartwatch design. A lower score in either reduced model indicates lost predictive information or altered feature interactions; it does not identify a single sensor as causally responsible.

The comparison is only valid when each model is evaluated with the feature matrix it was trained on. A full-modality model cannot be given reduced columns directly because its learned input dimensionality and feature ordering differ. Use the corresponding reduced notebook's own LOSO results and compare pooled accuracy, balanced accuracy, anxiety F1, subject-level results, and confusion matrices. Accuracy alone is insufficient because the anxiety classes are imbalanced.

### Related notes

- [[2026-10-29 Research Note]] — why BVP and ECG are excluded from the smartwatch-oriented experiment.
- [[Machine Learning Models & Evaluation]] — classifier behavior and evaluation metrics.
- [[Pipeline STAI User (v2_stai)]] — shared labels and preprocessing protocol.
- [[Katalog Fitur Fisiologis]] — physiological meaning of the retained and removed feature families.
- [[Data Artefacts Overview]] — generated dataset and model locations.
- [[Pra-Pemrosesan Sinyal & Filtering]] — filtering and signal preparation context.

## WESAD Wrist ACC Compatibility Investigation

### Research question

The upcoming testing will take place under controlled laboratory conditions. The question is whether wrist accelerometer (ACC) signals in WESAD are compatible with that setting, specifically:

- Do participants keep their hands/wrists mostly still?
- Do they show light, intermittent movement or fidgeting?
- Are there periods of larger arm/hand movement that could make WESAD less comparable to the planned test?

### Data and analysis notebook

The investigation uses the Empatica E4 wrist ACC signal from the WESAD subject files:

- Dataset: `WESAD/<subject>/<subject>.pkl`
- Example subject configured in the notebook: `S2`
- Wrist ACC sampling rate: 32 Hz
- Wrist ACC channels: three axes
- Label signal: WESAD labels at 700 Hz

The inspection notebook is [[../Scripts/Utils/inspect_acc_dataset.ipynb|inspect_acc_dataset.ipynb]]. It:

1. Loads one WESAD subject.
2. Extracts `data['signal']['wrist']['ACC']` and `data['label']`.
3. Computes the resultant wrist ACC magnitude:

   ```text
   sqrt(x² + y² + z²)
   ```

4. Plots the complete wrist ACC magnitude over time.
5. Provides a zoomed inspection window.
6. Marks label transitions to compare movement with experimental phases.
7. Reports standard deviation and peak-to-peak variation as simple movement indicators.

### Interpretation framework

- **Mostly still:** low short-window variation and a magnitude close to the gravity level.
- **Light fidgeting:** repeated small spikes or increased short-window variation without large sustained shifts.
- **Larger movement:** pronounced spikes, sustained changes, or a changing orientation/gravity component.

The ACC magnitude is useful for detecting movement intensity, but it does not identify the exact hand posture or distinguish all sources of movement. Orientation changes can also change the individual axes while leaving the resultant magnitude relatively stable.

### Current conclusion and limitation

WESAD participants were generally seated during protocol phases, so the dataset is relevant as a reference for controlled seated testing. However, the notebook is an inspection workflow rather than a completed multi-subject statistical result. It should be run across all available subjects and phases before concluding that WESAD wrist activity matches the planned laboratory protocol.

The most relevant comparison is expected to be:

- Baseline/rest: likely reference for controlled stillness.
- Meditation: likely reference for the lowest movement.
- Stress and amusement: useful for checking whether fidgeting or incidental movement increases.

The planned test should document posture, arm support, permitted hand movement, and whether fidgeting is allowed. These controls are necessary when comparing its ACC distribution with WESAD.

### Follow-up analysis

1. Run the notebook for every WESAD subject rather than only `S2`.
2. Align the 700 Hz labels to the 32 Hz ACC samples using nearest-neighbour or interval resampling.
3. Calculate per-subject and per-phase median, standard deviation, percentile range, and movement-event rate.
4. Add windowed thresholds for stillness, fidgeting, and larger movement based on the empirical WESAD distribution rather than arbitrary thresholds.
5. Compare those distributions with a pilot recording from the planned controlled laboratory test.

### Related notes

- [[Dataset WESAD]]
- [[Pipeline Forked (WESAD Protocol)]]
- [[Wrangling & Sinkronisasi Multimodal]]
- [[Pra-Pemrosesan Sinyal & Filtering]]
- [[Rencana Riset & Pembangunan Sistem Deteksi Anxiety]]

## Reduced-Modality Accuracy Shortfall Analysis

### Request and implementation

On 2026-11-05, an analysis was added to both reduced-modality notebooks:

- `Scripts/5th try/Minimum_sensor_modals/stai_6s_std_norm_minimum_sensor.ipynb`
- `Scripts/5th try/Sliced_sensor_modals/stai_6s_std_norm_sliced_sensor.ipynb`

The appended cells compare each reduced-modality model with the corresponding original full-modality model:

- Random Forest
- SVM
- XGBoost
- KNN

The analysis reports:

- Full-modality LOSO accuracy.
- Reduced-modality pooled LOSO accuracy.
- Absolute accuracy shortfall:
  `full_modality_accuracy - reduced_modality_accuracy`.
- Shortfall as a percentage of the full-modality accuracy.
- Mean shortfall across models.
- A bar chart comparing full- and reduced-modality accuracy.

The full-modality reference accuracies used in the comparison are:

| Model | Full-modality LOSO accuracy |
|---|---:|
| Random Forest | 0.84 |
| SVM | 0.87 |
| XGBoost | 0.84 |
| KNN | 0.86 |

### Error and correction

The first version referenced `model_metrics`, which was not defined in either notebook. The notebooks already produce LOSO results in `loso_summary_df`, so the comparison was corrected to use:

```python
reduced_modality_accuracy = loso_summary_df["pooled_accuracy"].rename(
    "reduced_modality_accuracy"
)
```

This ensures that the comparison uses the pooled LOSO accuracy calculated from the same reduced-modality evaluation rather than an undefined or unrelated result object.

### Interpretation

A positive `accuracy_shortfall` means that the reduced-modality model performs below the original full-modality model. A negative value means that the reduced-modality model performs better for that classifier. The result should be interpreted alongside the anxiety F1 score, balanced accuracy, subject-level results, and LOSO protocol; accuracy alone can conceal the effect of class imbalance.

### Related notes and artifacts

- [[Machine Learning Models & Evaluation]]
- [[Pipeline STAI User (v2_stai)]]
- [[Data Artefacts Overview]]
- [[Katalog Fitur Fisiologis]]
- [[Development Findings 1]]
- [[MOC - Skripsi Repository]]