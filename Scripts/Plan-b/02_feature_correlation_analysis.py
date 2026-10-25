"""
02_feature_correlation_analysis.py
Loads preprocessed subject feature pickles, maps window features against
STAI delta and STAI decrease target (Delta < 0), and computes feature correlations
and group statistics to identify physiological markers of lowered state anxiety.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'WESAD').exists())
OUTPUT_DIR = REPO_ROOT / 'Data' / 'Preprocessed n STAI-labeled'


def main():
    pkl_files = sorted(OUTPUT_DIR.glob('S*_stai.pkl'))
    if not pkl_files:
        raise FileNotFoundError(f"No S*_stai.pkl files found in {OUTPUT_DIR}")

    dfs = []
    for path in pkl_files:
        df = pd.read_pickle(path)
        if 'condition_stai_mean' not in df.columns or 'baseline_stai_mean' not in df.columns:
            continue
        df['stai_delta'] = df['condition_stai_mean'] - df['baseline_stai_mean']
        df['stai_drop'] = (df['stai_delta'] < 0).astype(int)
        dfs.append(df)

    if not dfs:
        raise ValueError("No valid subject pickles with STAI provenance found.")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataset shape: {combined.shape}")
    print("STAI Drop Distribution (0 = Delta >= 0, 1 = Delta < 0):")
    print(combined['stai_drop'].value_counts())

    metadata_cols = {
        'binary_anxiety', 'subject_id', 'condition', 'window_start_seconds',
        'window_end_seconds', 'condition_stai_mean', 'baseline_stai_mean',
        'stai_delta', 'stai_drop'
    }
    feature_cols = [c for c in combined.columns if c not in metadata_cols]

    # Compute point-biserial correlation / difference between groups for features
    corrs = []
    X = combined[feature_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
    y = combined['stai_drop']

    for col in feature_cols:
        series = X[col]
        if series.nunique() <= 1:
            continue
        corr = series.corr(y, method='pearson')
        mean_drop = series[y == 1].mean()
        mean_nodrop = series[y == 0].mean()
        corrs.append({
            'feature': col,
            'correlation_with_drop': corr,
            'mean_stai_drop': mean_drop,
            'mean_stai_nodrop': mean_nodrop,
            'diff': mean_drop - mean_nodrop
        })

    df_corr = pd.DataFrame(corrs).sort_values(by='correlation_with_drop', key=abs, ascending=False)
    out_csv = OUTPUT_DIR / 'stai_drop_feature_correlations.csv'
    df_corr.to_csv(out_csv, index=False)
    print(f"\nSaved feature correlations to {out_csv}")
    print("\nTop 15 Features Correlated with STAI Decrease (Delta < 0):")
    print(df_corr.head(15).to_string(index=False))


if __name__ == '__main__':
    main()
