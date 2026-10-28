"""Fourth-try STAI-labelled WESAD feature extraction.

Adapted from Scripts/3rd try/v3_downsampled.ipynb. The only intended
windowing change is a 6-second shift with 60-second windows.
"""

# %% Source notebook cell 1
import gc
import pickle
import sys
import time
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import heartpy as hp
import neurokit2 as nk

warnings.simplefilter('ignore')

# Prevent biosppy's optional interactive plotting modules from requiring tkinter.
sys.modules['biosppy.inter_plotting'] = MagicMock()
sys.modules['biosppy.inter_plotting.ecg'] = MagicMock()
sys.modules['biosppy.inter_plotting.acc'] = MagicMock()
import biosppy

REPO_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'WESAD').exists())
FB_CODE_CANDIDATES = [
    Path(__file__).resolve().parent / 'fb_code',
    REPO_ROOT / 'Scripts' / '3rd try' / 'fb_code',
    REPO_ROOT / '_forked' / 'fb_code',
    Path.cwd() / 'fb_code',
]
FB_CODE_DIR = next(
    path for path in FB_CODE_CANDIDATES
    if (path / 'respiration.py').exists() and (path / 'utils.py').exists()
)
if str(FB_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(FB_CODE_DIR))

from utils import (
    butter_lowpass_filter,
    filterSignalFIR as filter_signal_fir,
    get_net_accel,
    get_window_stats,
    get_peak_freq,
    get_absolute_integral,
    get_slope,
    get_dynamic_range,
)
from respiration import get_resp_features

# User-configurable pipeline settings.
SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
WINDOW_LENGTH_SECONDS = 60
WINDOW_SHIFT_SECONDS = 6  # fourth-try requirement
ACC_WINDOW_LENGTH_SECONDS = 5
ACC_WINDOW_SHIFT_SECONDS = 0.25
NON_ACC_FS = 70
STAI_INCREASE_THRESHOLD = 0.2  # anxiety if condition mean - personal Base mean >= this
SKIP_EXTRACTED_SUBJECTS = False  # existing files contain old protocol-derived labels
OVERWRITE_EXISTING_OUTPUTS = True  # set False to refuse replacing existing files

# RAW_FS describes the WESAD files; FS describes the feature-extraction rates.
RAW_FS = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700,
          'Resp': 700, 'ECG': 700, 'chest': 700}
FS = {'ACC': 32, 'BVP': NON_ACC_FS, 'EDA': NON_ACC_FS, 'TEMP': NON_ACC_FS, 'label': 700,
      'Resp': NON_ACC_FS, 'ECG': NON_ACC_FS, 'chest': 700}
WESAD_DIR = REPO_ROOT / 'WESAD'
OUTPUT_DIR = REPO_ROOT / 'Data' / 'Preprocessed n STAI-labeled 6s'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = REPO_ROOT / 'Models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

if WINDOW_LENGTH_SECONDS <= 0 or WINDOW_SHIFT_SECONDS <= 0:
    raise ValueError('WINDOW_LENGTH_SECONDS and WINDOW_SHIFT_SECONDS must be positive.')
if ACC_WINDOW_LENGTH_SECONDS <= 0 or ACC_WINDOW_SHIFT_SECONDS <= 0:
    raise ValueError('ACC_WINDOW_LENGTH_SECONDS and ACC_WINDOW_SHIFT_SECONDS must be positive.')
if ACC_WINDOW_LENGTH_SECONDS > WINDOW_LENGTH_SECONDS:
    raise ValueError('ACC_WINDOW_LENGTH_SECONDS cannot exceed WINDOW_LENGTH_SECONDS.')
print('Setup completed:', OUTPUT_DIR)
print(f'Window length={WINDOW_LENGTH_SECONDS}s, shift={WINDOW_SHIFT_SECONDS}s')
print(f'ACC sub-window length={ACC_WINDOW_LENGTH_SECONDS}s, shift={ACC_WINDOW_SHIFT_SECONDS}s')
print(f'STAI increase threshold={STAI_INCREASE_THRESHOLD}')

# %% Source notebook cell 3
def parse_clock_mmss(value):
    """Convert questionnaire MM.SS notation (e.g. 7.08 -> 428 seconds)."""
    text = str(value).strip()
    if not text:
        return np.nan
    if '.' in text:
        minute_text, second_text = text.split('.', 1)
        minutes = int(minute_text)
        seconds = int(second_text)
    else:
        minutes, seconds = int(text), 0
    if not 0 <= seconds < 60:
        raise ValueError(f'Invalid MM.SS time {value!r}: seconds must be 00-59')
    return minutes * 60 + seconds


def _parse_numeric_fields(parts):
    """Preserve CSV field alignment, converting empty fields to NaN."""
    values = []
    for part in parts:
        text = part.strip()
        if not text:
            values.append(np.nan)
        else:
            try:
                values.append(float(text))
            except ValueError:
                values.append(np.nan)
    return values


# WESAD uses a shortened six-item state-anxiety questionnaire. The positively
# worded items are reverse-scored before averaging: 1 <-> 4 and 2 <-> 3.
STAI_ITEM_NAMES = (
    'feel_at_ease', 'feel_nervous', 'jittery', 'relaxed', 'worried', 'pleasant'
)
STAI_REVERSE_ITEM_INDICES = {0, 3, 5}


def score_stai_responses(raw_responses):
    """Return raw and correctly reverse-scored six-item STAI responses."""
    raw = [float(value) if pd.notna(value) else np.nan for value in raw_responses]
    if len(raw) > len(STAI_ITEM_NAMES):
        raise ValueError(f'Expected at most {len(STAI_ITEM_NAMES)} shortened STAI items, got {len(raw)}')
    scored = [
        (5.0 - value) if index in STAI_REVERSE_ITEM_INDICES and pd.notna(value) else value
        for index, value in enumerate(raw)
    ]
    return raw, scored


def parse_quest_file(subject_id):
    quest_path = WESAD_DIR / f'S{subject_id}' / f'S{subject_id}_quest.csv'
    if not quest_path.exists():
        raise FileNotFoundError(f'Questionnaire file not found: {quest_path}')

    order = start_raw = end_raw = None
    stai_rows = []
    with open(quest_path, 'r', encoding='latin1') as f:
        for line in f:
            parts = [part.strip() for part in line.rstrip('\r\n').split(';')]
            tag = parts[0].strip() if parts else ''
            if tag.startswith('# ORDER'):
                order = [value.strip() for value in parts[1:]]
            elif tag.startswith('# START'):
                start_raw = parts[1:]
            elif tag.startswith('# END'):
                end_raw = parts[1:]
            elif tag.startswith('# STAI'):
                stai_rows.append(_parse_numeric_fields(parts[1:]))

    if order is None or start_raw is None or end_raw is None:
        raise ValueError(f'{quest_path.name}: missing ORDER, START, or END row')
    if len(start_raw) < len(order) or len(end_raw) < len(order):
        raise ValueError(f'{quest_path.name}: timing rows do not cover all conditions')
    if not stai_rows:
        raise ValueError(f'{quest_path.name}: no STAI response rows found')

    conditions = []
    for index, condition in enumerate(order):
        if not condition:
            continue
        start_text, end_text = start_raw[index].strip(), end_raw[index].strip()
        start_seconds = parse_clock_mmss(start_text) if start_text else np.nan
        end_seconds = parse_clock_mmss(end_text) if end_text else np.nan
        responses = [row[index] if index < len(row) else np.nan for row in stai_rows]
        raw_responses, scored_responses = score_stai_responses(responses)
        valid_raw = [float(value) for value in raw_responses if pd.notna(value)]
        valid_scored = [float(value) for value in scored_responses if pd.notna(value)]
        if any(value < 1 or value > 4 for value in valid_raw):
            raise ValueError(f'{quest_path.name}: STAI response outside 1-4 for condition {condition!r}')
        conditions.append({
            'condition': condition,
            'start_seconds': start_seconds,
            'end_seconds': end_seconds,
            'stai_responses_raw': valid_raw,
            'stai_responses_scored': valid_scored,
            'stai_item_count': len(valid_scored),
            'stai_mean_raw': float(np.mean(valid_raw)) if valid_raw else np.nan,
            'stai_mean': float(np.mean(valid_scored)) if valid_scored else np.nan,
        })

    if not any(item['condition'].strip().lower() == 'base' for item in conditions):
        raise ValueError(f'{quest_path.name}: no Base condition found')
    return conditions


# Inspect parsed condition data for every configured participant.
all_participant_conditions = []
for subject_id in SUBJECTS:
    participant_df = pd.DataFrame(parse_quest_file(subject_id))
    participant_df.insert(0, 'subject_id', subject_id)
    all_participant_conditions.append(participant_df)

all_conditions_df = pd.concat(all_participant_conditions, ignore_index=True)
print(all_conditions_df.sort_values(['subject_id', 'condition']).reset_index(drop=True).to_string(index=False))


# %% Source notebook cell 5
def load_raw_subject(subject_id):
    path = WESAD_DIR / f'S{subject_id}' / f'S{subject_id}.pkl'
    with open(path, 'rb') as file:
        return pickle.load(file, encoding='latin1')


def build_condition_labels(subject_id):
    conditions = parse_quest_file(subject_id)
    base = next(item for item in conditions if item['condition'].strip().lower() == 'base')
    if pd.isna(base['stai_mean']):
        raise ValueError(f'S{subject_id}: Base STAI mean cannot be calculated')
    base_mean = base['stai_mean']

    labeled = []
    for item in conditions:
        name = item['condition']
        if name.strip().lower() == 'base':
            continue
        if pd.isna(item['stai_mean']) or pd.isna(item['start_seconds']) or pd.isna(item['end_seconds']):
            continue
        if item['end_seconds'] <= item['start_seconds']:
            raise ValueError(f'S{subject_id}: invalid interval for {name}: {item}')
        item = dict(item)
        item['baseline_stai_mean'] = base_mean
        item['binary_anxiety'] = int(item['stai_mean'] - base_mean >= STAI_INCREASE_THRESHOLD)
        labeled.append(item)

    if not labeled:
        raise ValueError(f'S{subject_id}: no non-Base conditions have both valid STAI scores and timing')
    return labeled


def compute_features_stai(subject_id):
    """Filter and align raw physiological channels to the WESAD 700-Hz timeline."""
    raw_data = load_raw_subject(subject_id)
    wrist = raw_data['signal']['wrist']
    chest = raw_data['signal']['chest']
    labels = np.asarray(raw_data['label']).ravel()

    eda_df = pd.DataFrame(wrist['EDA'], columns=['EDA'])
    bvp_df = pd.DataFrame(wrist['BVP'], columns=['BVP'])
    acc_df = pd.DataFrame(wrist['ACC'], columns=['ACC_x', 'ACC_y', 'ACC_z'])
    temp_df = pd.DataFrame(wrist['TEMP'], columns=['TEMP'])
    acc_c_df = pd.DataFrame(chest['ACC'], columns=['ACC_x_C', 'ACC_y_C', 'ACC_z_C'])
    ecg_df = pd.DataFrame(chest['ECG'], columns=['ECG'])
    eda_c_df = pd.DataFrame(chest['EDA'], columns=['EDA_C'])
    emg_c_df = pd.DataFrame(chest['EMG'], columns=['EMG_C'])
    resp_c_df = pd.DataFrame(chest['Resp'], columns=['Resp_C'])
    temp_c_df = pd.DataFrame(chest['Temp'], columns=['Temp_C'])

    eda_df['EDA'] = butter_lowpass_filter(eda_df['EDA'], 1.0, RAW_FS['EDA'], 6)
    eda_c_df['EDA_C'] = butter_lowpass_filter(eda_c_df['EDA_C'], 1.0, FS['chest'], 6)
    eda_data = nk.eda_phasic(nk.standardize(eda_df['EDA']), sampling_rate=RAW_FS['EDA'])
    eda_df['EDA_SCR'] = eda_data['EDA_Phasic']
    eda_df['EDA_SCL'] = eda_data['EDA_Tonic']
    eda_data_c = nk.eda_phasic(nk.standardize(eda_c_df['EDA_C']), sampling_rate=FS['chest'])
    eda_c_df['EDA_SCR_C'] = eda_data_c['EDA_Phasic']
    eda_c_df['EDA_SCL_C'] = eda_data_c['EDA_Tonic']

    for col in acc_df.columns:
        acc_df[col] = filter_signal_fir(acc_df[col].values)
    for col in acc_c_df.columns:
        acc_c_df[col] = filter_signal_fir(acc_c_df[col].values)

    n_700 = len(labels)
    df = pd.DataFrame(index=np.arange(n_700, dtype=np.int64))
    df['protocol_label'] = labels

    def resample_to_fs(arr, source_fs, target_fs):
        if source_fs == target_fs:
            return np.asarray(arr)
        target_len = int(round(len(arr) * target_fs / source_fs))
        resampled = resample_poly(np.asarray(arr), target_fs, source_fs, axis=0)
        if len(resampled) < target_len:
            pad_width = [(0, target_len - len(resampled))]
            pad_width.extend([(0, 0)] * (resampled.ndim - 1))
            resampled = np.pad(resampled, pad_width, mode='edge')
        return resampled[:target_len]

    def add_signal(name, arr, fs, val_cols=None):
        arr = np.asarray(arr)
        idx = np.round(np.arange(len(arr), dtype=np.int64) * FS['label'] / fs).astype(np.int64)
        idx = np.clip(idx, 0, n_700 - 1)
        if arr.ndim == 1:
            df[name] = pd.Series(arr, index=idx).groupby(level=0).mean()
        else:
            for c_i, col_name in enumerate(val_cols):
                df[col_name] = pd.Series(arr[:, c_i], index=idx).groupby(level=0).mean()

    add_signal('EDA', resample_to_fs(eda_df['EDA'], RAW_FS['EDA'], FS['EDA']), FS['EDA'])
    add_signal('EDA_SCR', resample_to_fs(eda_df['EDA_SCR'], RAW_FS['EDA'], FS['EDA']), FS['EDA'])
    add_signal('EDA_SCL', resample_to_fs(eda_df['EDA_SCL'], RAW_FS['EDA'], FS['EDA']), FS['EDA'])
    add_signal('BVP', resample_to_fs(bvp_df['BVP'], RAW_FS['BVP'], FS['BVP']), FS['BVP'])
    add_signal('TEMP', resample_to_fs(temp_df['TEMP'], RAW_FS['TEMP'], FS['TEMP']), FS['TEMP'])
    add_signal('ACC', acc_df.values, RAW_FS['ACC'], ['ACC_x', 'ACC_y', 'ACC_z'])
    add_signal('ECG', resample_to_fs(ecg_df['ECG'], RAW_FS['ECG'], FS['ECG']), FS['ECG'])
    add_signal('EDA_C', resample_to_fs(eda_c_df['EDA_C'], RAW_FS['chest'], FS['EDA']), FS['EDA'])
    add_signal('EDA_SCR_C', resample_to_fs(eda_c_df['EDA_SCR_C'], RAW_FS['chest'], FS['EDA']), FS['EDA'])
    add_signal('EDA_SCL_C', resample_to_fs(eda_c_df['EDA_SCL_C'], RAW_FS['chest'], FS['EDA']), FS['EDA'])
    add_signal('ACC_C', acc_c_df.values, FS['chest'], ['ACC_x_C', 'ACC_y_C', 'ACC_z_C'])
    add_signal('EMG_C', resample_to_fs(emg_c_df['EMG_C'], RAW_FS['chest'], FS['ECG']), FS['ECG'])
    add_signal('Resp_C', resample_to_fs(resp_c_df['Resp_C'], RAW_FS['chest'], FS['Resp']), FS['Resp'])
    add_signal('Temp_C', resample_to_fs(temp_c_df['Temp_C'], RAW_FS['chest'], FS['TEMP']), FS['TEMP'])

    df.reset_index(names='sample_index', inplace=True)
    return df

print('Signal preprocessing and STAI condition-label functions defined.')

# %% Source notebook cell 7
def get_window_stats_ecg(data):
    if 'ECG' not in data or data['ECG'].dropna().empty:
        return {k: None for k in ['bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50']}
    try:
        _, measures = hp.process(data['ECG'].dropna().reset_index(drop=True), FS['ECG'])
        return {k: measures[k] for k in ['bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50']}
    except Exception:
        return {k: None for k in ['bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50']}


ACC_COLUMNS = ['ACC_x', 'ACC_y', 'ACC_z', 'ACC_x_C', 'ACC_y_C', 'ACC_z_C']


def _safe_acc_feature_values(series):
    series = pd.Series(series).dropna()
    if series.empty:
        return np.nan, np.nan, np.nan
    try:
        abs_integral = get_absolute_integral(series)
    except Exception:
        abs_integral = np.nan
    try:
        peak_freq = get_peak_freq(series)
    except Exception:
        peak_freq = np.nan
    return float(series.std()), abs_integral, peak_freq


def aggregate_acc_subwindows(window):
    """Aggregate ACC features from 5-second windows shifted by 0.25 seconds."""
    sub_len = int(round(FS['label'] * ACC_WINDOW_LENGTH_SECONDS))
    sub_step = int(round(FS['label'] * ACC_WINDOW_SHIFT_SECONDS))
    if sub_len <= 0 or sub_step <= 0:
        raise ValueError('ACC sub-window sample length and step must be positive.')

    subwindow_values = {f'{column}_{stat}': [] for column in ACC_COLUMNS for stat in ['std', 'abs_integral', 'peak_freq']}
    subwindow_count = 0
    for start in range(0, len(window) - sub_len + 1, sub_step):
        subwindow = window.iloc[start:start + sub_len]
        subwindow_count += 1
        for column in ACC_COLUMNS:
            std_value, integral_value, peak_value = _safe_acc_feature_values(subwindow[column])
            subwindow_values[f'{column}_std'].append(std_value)
            subwindow_values[f'{column}_abs_integral'].append(integral_value)
            subwindow_values[f'{column}_peak_freq'].append(peak_value)

    if subwindow_count == 0:
        raise ValueError('Main window is shorter than the configured ACC sub-window.')

    aggregated = {}
    for name, values in subwindow_values.items():
        numeric_values = np.asarray(values, dtype=float)
        aggregated[f'ACC_sub_{name}_mean'] = float(np.nanmean(numeric_values))
        aggregated[f'ACC_sub_{name}_std'] = float(np.nanstd(numeric_values))
    aggregated['ACC_subwindow_count'] = subwindow_count
    return aggregated


def extract_stai_features_subject(subject_id):
    df = compute_features_stai(subject_id)
    condition_labels = build_condition_labels(subject_id)
    samples = []
    window_len = int(round(FS['label'] * WINDOW_LENGTH_SECONDS))
    step = int(round(FS['label'] * WINDOW_SHIFT_SECONDS))
    feat_names = None
    candidate_windows = max(0, (len(df) - window_len) // step + 1)
    dropped_unlabeled_windows = 0
    dropped_ambiguous_windows = 0
    acc_subwindow_counts = []

    for start in range(0, len(df) - window_len + 1, step):
        end = start + window_len
        containing = [
            item for item in condition_labels
            if start >= item['start_seconds'] * FS['label']
            and end <= item['end_seconds'] * FS['label']
        ]
        if len(containing) != 1:
            if len(containing) > 1:
                dropped_ambiguous_windows += 1
            else:
                dropped_unlabeled_windows += 1
            continue
        condition = containing[0]
        window = df.iloc[start:end].drop(columns=['protocol_label', 'sample_index']).copy()

        net_acc = get_net_accel(window, 'wrist').rename('net_acc')
        window = pd.concat([net_acc, window], axis=1)
        net_acc_c = get_net_accel(window, 'chest').rename('net_acc_C')
        window = pd.concat([net_acc_c, window], axis=1)
        acc_features = aggregate_acc_subwindows(window)
        acc_subwindow_counts.append(acc_features['ACC_subwindow_count'])

        stats = get_window_stats(window, label=0)
        ecg_stats = get_window_stats_ecg(window)
        if 'label' in stats:
            del stats['label']
        x = pd.DataFrame(stats)
        if feat_names is None:
            feat_names = [f'{row}_{col}' for row in x.index for col in x.columns]
        wdf = pd.DataFrame(x.values.flatten()).T
        wdf.columns = feat_names

        wdf['binary_anxiety'] = condition['binary_anxiety']
        wdf['subject_id'] = subject_id
        wdf['condition'] = condition['condition']
        wdf['window_start_seconds'] = start / FS['label']
        wdf['window_end_seconds'] = end / FS['label']
        wdf['condition_stai_mean'] = condition['stai_mean']
        wdf['baseline_stai_mean'] = condition['baseline_stai_mean']
        for name, value in acc_features.items():
            wdf[name] = value

        wdf['BVP_peak_freq'] = get_peak_freq(window['BVP'].dropna())
        wdf['net_acc_abs_integral'] = get_absolute_integral(window['net_acc'].dropna())
        for col in ACC_COLUMNS:
            wdf[f'{col}_abs_integral'] = get_absolute_integral(window[col].dropna())
            series = window[col].dropna()
            wdf[f'{col}_peak_freq'] = None if series.empty else get_peak_freq(series)
        for key, value in ecg_stats.items():
            wdf[f'ECG_{key}'] = value
        wdf['EDA_slope'] = get_slope(window['EDA'].dropna())
        wdf['EDA_C_slope'] = get_slope(window['EDA_C'].dropna()) if window['EDA_C'].notna().any() else None
        wdf['EDA_drange'] = get_dynamic_range(window['EDA'].dropna())
        wdf['EDA_C_drange'] = get_dynamic_range(window['EDA_C'].dropna()) if window['EDA_C'].notna().any() else None
        wdf['EMG_drange'] = get_dynamic_range(window['EMG_C'].dropna())
        wdf['EMG_abs_integral'] = get_absolute_integral(window['EMG_C'].dropna())
        if window['Resp_C'].notna().any():
            try:
                resp_features = get_resp_features(window['Resp_C'].dropna())
            except (IndexError, ValueError):
                resp_features = (None,) * 6
            for r_name, r_val in zip([
                'Resp_C_rate', 'Resp_C_Inhal_mean', 'Resp_C_Inhal_std',
                'Resp_C_Exhal_mean', 'Resp_C_Exhal_std', 'Resp_C_I/E',
            ], resp_features):
                wdf[r_name] = r_val
        wdf['TEMP_drange'] = get_dynamic_range(window['TEMP'].dropna())
        wdf['TEMP_C_drange'] = get_dynamic_range(window['Temp_C'].dropna()) if window['Temp_C'].notna().any() else None
        wdf['TEMP_slope'] = get_slope(window['TEMP'].dropna())
        wdf['TEMP_C_slope'] = get_slope(window['Temp_C'].dropna()) if window['Temp_C'].notna().any() else None
        samples.append(wdf)

    result = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    result.attrs['candidate_windows'] = candidate_windows
    result.attrs['dropped_unlabeled_windows'] = dropped_unlabeled_windows
    result.attrs['dropped_ambiguous_windows'] = dropped_ambiguous_windows
    result.attrs['acc_subwindow_count'] = int(np.median(acc_subwindow_counts)) if acc_subwindow_counts else 0
    return result


# Process and save subjects to Data/Preprocessed n STAI-labeled/
summary = []
for subject_id in SUBJECTS:
    out_path = OUTPUT_DIR / f'S{subject_id}_stai_60s_shift6s.pkl'
    if out_path.exists() and SKIP_EXTRACTED_SUBJECTS:
        print(f'S{subject_id}: existing file skipped; ensure it was generated with this ACC sub-window version')
        continue
    if out_path.exists() and not OVERWRITE_EXISTING_OUTPUTS:
        print(f'WARNING: {out_path} exists and overwrite is disabled; skipping S{subject_id}.')
        continue

    start_t = time.time()
    features_df = extract_stai_features_subject(subject_id)
    if features_df.empty:
        print(f'S{subject_id}: no fully-contained labeled windows; no file saved')
        continue
    if features_df.filter(regex=r'^ACC_sub_').isna().any().any():
        raise ValueError(f'S{subject_id}: ACC sub-window aggregation produced NaN values')
    features_df.to_pickle(out_path)
    summary.append({
        'subject': subject_id,
        'candidate_windows': features_df.attrs['candidate_windows'],
        'retained_windows': len(features_df),
        'dropped_unlabeled_windows': features_df.attrs['dropped_unlabeled_windows'],
        'dropped_ambiguous_windows': features_df.attrs['dropped_ambiguous_windows'],
        'acc_subwindows_per_main_window': features_df.attrs['acc_subwindow_count'],
        'n_features': len(features_df.columns),
        'no_anxiety_windows': int((features_df['binary_anxiety'] == 0).sum()),
        'anxiety_windows': int((features_df['binary_anxiety'] == 1).sum()),
        'saved_to': out_path.name,
        'seconds': round(time.time() - start_t, 2),
    })
    print(summary[-1])
    del features_df
    gc.collect()

pd.DataFrame(summary)