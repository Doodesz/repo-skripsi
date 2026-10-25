"""
01_analyze_stai_drops.py
Parses WESAD questionnaire files across all subjects, computes participant baselines,
calculates delta STAI (condition mean - baseline mean), and analyzes the distribution
of conditions where STAI is lower than baseline (Delta < 0).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'WESAD').exists())
WESAD_DIR = REPO_ROOT / 'WESAD'
OUTPUT_DIR = REPO_ROOT / 'Data' / 'Preprocessed n STAI-labeled'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]


def parse_clock_mmss(value):
    text = str(value).strip()
    if not text:
        return np.nan
    if '.' in text:
        minute_text, second_text = text.split('.', 1)
        minutes = int(minute_text)
        seconds = int(second_text)
    else:
        minutes, seconds = int(text), 0
    return minutes * 60 + seconds


def _parse_numeric_fields(parts):
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

    conditions = []
    for index, condition in enumerate(order):
        if not condition:
            continue
        start_text = start_raw[index].strip() if index < len(start_raw) else ''
        end_text = end_raw[index].strip() if index < len(end_raw) else ''
        start_seconds = parse_clock_mmss(start_text) if start_text else np.nan
        end_seconds = parse_clock_mmss(end_text) if end_text else np.nan
        responses = [row[index] if index < len(row) else np.nan for row in stai_rows]
        valid_responses = [float(val) for val in responses if pd.notna(val)]
        conditions.append({
            'condition': condition,
            'start_seconds': start_seconds,
            'end_seconds': end_seconds,
            'stai_responses': valid_responses,
            'stai_mean': float(np.mean(valid_responses)) if valid_responses else np.nan,
        })
    return conditions


def main():
    all_records = []
    for subject_id in SUBJECTS:
        try:
            conds = parse_quest_file(subject_id)
        except Exception as e:
            print(f"Skipping S{subject_id}: {e}")
            continue

        base_items = [c for c in conds if c['condition'].strip().lower() == 'base']
        if not base_items or pd.isna(base_items[0]['stai_mean']):
            print(f"S{subject_id}: Base condition missing or invalid STAI mean.")
            continue
        base_mean = base_items[0]['stai_mean']

        for item in conds:
            c_name = item['condition']
            c_mean = item['stai_mean']
            if pd.isna(c_mean):
                continue
            delta = c_mean - base_mean
            all_records.append({
                'subject_id': subject_id,
                'condition': c_name,
                'stai_mean': c_mean,
                'baseline_stai_mean': base_mean,
                'stai_delta': delta,
                'is_lower_than_baseline': int(delta < 0),
            })

    df_deltas = pd.DataFrame(all_records)
    out_csv = OUTPUT_DIR / 'stai_condition_deltas_summary.csv'
    df_deltas.to_csv(out_csv, index=False)
    print(f"Saved STAI condition deltas summary to {out_csv}")
    print("\n--- STAI Delta Summary Statistics ---")
    print(df_deltas['stai_delta'].describe())
    print("\n--- Breakdown by Condition (Mean Delta & Proportion Lower than Baseline) ---")
    summary_cond = df_deltas.groupby('condition').agg(
        mean_delta=('stai_delta', 'mean'),
        median_delta=('stai_delta', 'median'),
        prop_lower=('is_lower_than_baseline', 'mean'),
        count=('stai_delta', 'count')
    ).reset_index()
    print(summary_cond.to_string(index=False))


if __name__ == '__main__':
    main()
