#!/usr/bin/env python3
"""
Phase 1a (sc_experiment_plan.md): Bootstrap confidence intervals for SC model
metrics.

For every model with saved per-record predictions on the test_binary_SC.csv
("Run A") test set -- the 9 deep-learning architectures
(results/evaluation/ann04062026_SC/predictions_*_SC.csv) and the 12
traditional-ML baselines (results/predictions/ann04062026_SC/predictions_ml_*_SC.csv)
-- this script performs stratified bootstrap resampling (positives and
negatives resampled with replacement separately, preserving the 21/479
SC-positive/Other ratio) and reports 95% percentile confidence intervals for
precision/recall/F1 (SC-positive class), accuracy, and macro F1.

No retraining; CPU only.

Usage:
    python scripts/evaluation/sc_bootstrap_ci.py
    python scripts/evaluation/sc_bootstrap_ci.py --n_bootstrap 5000 --seed 42
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
from sc_phase1_utils import (
    discover_predictions, load_predictions, binary_metrics, display_name,
    POSITIVE_LABEL, NEGATIVE_LABEL, PROJECT_ROOT,
)

_OUTPUT_DEFAULT = os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC/phase1_bootstrap_ci.csv')

METRICS = ['accuracy', 'precision_sc', 'recall_sc', 'f1_sc', 'macro_f1']


def bootstrap_metrics(df: pd.DataFrame, n_bootstrap: int, rng: np.random.Generator) -> pd.DataFrame:
    pos_idx = np.where(df['Label'].values == POSITIVE_LABEL)[0]
    neg_idx = np.where(df['Label'].values == NEGATIVE_LABEL)[0]

    y_true = df['Label'].values
    y_pred = df['predicted_label'].values

    rows = []
    for _ in range(n_bootstrap):
        sampled_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sampled_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([sampled_pos, sampled_neg])
        rows.append(binary_metrics(y_true[idx], y_pred[idx]))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Bootstrap CIs for SC model metrics')
    parser.add_argument('--n_bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_file', default=_OUTPUT_DEFAULT)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    predictions = discover_predictions()
    print(f"Found {len(predictions)} prediction files\n")

    rows = []
    for model_name, path in predictions.items():
        df = load_predictions(path)
        n_pos = int((df['Label'] == POSITIVE_LABEL).sum())
        point = binary_metrics(df['Label'].values, df['predicted_label'].values)
        boot = bootstrap_metrics(df, args.n_bootstrap, rng)

        row = {'model': model_name, 'display_name': display_name(model_name),
               'n': len(df), 'n_positive': n_pos}
        for m in METRICS:
            row[f'{m}_point'] = point[m]
            row[f'{m}_ci_lower'] = boot[m].quantile(0.025)
            row[f'{m}_ci_upper'] = boot[m].quantile(0.975)
            row[f'{m}_boot_mean'] = boot[m].mean()
            row[f'{m}_boot_std'] = boot[m].std()
        rows.append(row)
        print(f"  {row['display_name']:30s} F1_sc={point['f1_sc']:.3f}  "
              f"95% CI [{row['f1_sc_ci_lower']:.3f}, {row['f1_sc_ci_upper']:.3f}]")

    out_df = pd.DataFrame(rows).sort_values('f1_sc_point', ascending=False)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    out_df.to_csv(args.output_file, index=False)
    print(f"\nSaved -> {args.output_file}")


if __name__ == '__main__':
    main()
