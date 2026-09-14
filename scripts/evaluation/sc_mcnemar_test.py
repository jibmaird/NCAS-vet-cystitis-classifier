#!/usr/bin/env python3
"""
Phase 1b (sc_experiment_plan.md): Pairwise McNemar's test for SC models.

Compares the per-record correct/incorrect prediction vectors of pairs of
models on the test_binary_SC.csv ("Run A") test set, to determine whether
differences in the F1 ranking (tab:results_summary_sc) reflect a
statistically distinguishable difference or noise at n=21 positives.

Uses McNemar's exact test (binomial test on the discordant pairs), which is
appropriate for the small discordant counts expected here, plus the standard
chi-square statistic with continuity correction for reference.

By default compares the 4 top-performing DL architectures identified in
overleaf_methods_template4.tex (Multi-Knowledge Cross-Modal, Dual-Head
Cross-Attention, Dual-Head Weighted, Multi-Knowledge Gated), all pairwise.
Use --models to specify a different set, or --all_pairs to compare every
discovered model against every other.

No retraining; CPU only.

Usage:
    python scripts/evaluation/sc_mcnemar_test.py
    python scripts/evaluation/sc_mcnemar_test.py --models cross_modal gated ml_syndromes_only_svm
    python scripts/evaluation/sc_mcnemar_test.py --all_pairs
"""

import os
import sys
import argparse
import itertools
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
from sc_phase1_utils import discover_predictions, load_predictions, display_name, PROJECT_ROOT

_OUTPUT_DEFAULT = os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC/phase1_mcnemar.csv')

DEFAULT_MODELS = ['cross_modal', 'dualhead_cross_attention', 'dualhead_weighted', 'gated']


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """McNemar's test on two paired boolean correctness vectors."""
    a_right_b_wrong = int(np.sum(correct_a & ~correct_b))   # n10
    a_wrong_b_right = int(np.sum(~correct_a & correct_b))   # n01
    n_discordant = a_right_b_wrong + a_wrong_b_right

    if n_discordant == 0:
        p_exact = 1.0
    else:
        k = min(a_right_b_wrong, a_wrong_b_right)
        p_exact = binomtest(k, n_discordant, 0.5, alternative='two-sided').pvalue

    if n_discordant > 0:
        chi2_stat = (abs(a_right_b_wrong - a_wrong_b_right) - 1) ** 2 / n_discordant
        p_chi2 = chi2.sf(chi2_stat, df=1)
    else:
        chi2_stat, p_chi2 = 0.0, 1.0

    return {
        'n_a_right_b_wrong': a_right_b_wrong,
        'n_a_wrong_b_right': a_wrong_b_right,
        'n_discordant': n_discordant,
        'p_value_exact': p_exact,
        'chi2_corrected': chi2_stat,
        'p_value_chi2': p_chi2,
    }


def main():
    parser = argparse.ArgumentParser(description="Pairwise McNemar's test for SC models")
    parser.add_argument('--models', nargs='+', default=None,
                         help=f'Model names to compare (default: {DEFAULT_MODELS})')
    parser.add_argument('--all_pairs', action='store_true',
                         help='Compare every discovered model against every other')
    parser.add_argument('--output_file', default=_OUTPUT_DEFAULT)
    args = parser.parse_args()

    predictions = discover_predictions()
    print(f"Found {len(predictions)} prediction files")

    if args.all_pairs:
        model_names = list(predictions.keys())
    else:
        model_names = args.models if args.models else DEFAULT_MODELS

    missing = [m for m in model_names if m not in predictions]
    if missing:
        raise ValueError(f"Models not found: {missing}. Available: {sorted(predictions)}")

    dfs = {m: load_predictions(predictions[m]) for m in model_names}

    # Sanity check: all compared models must be evaluated on the same records
    # (same Label sequence), otherwise paired comparison is invalid.
    ref_labels = dfs[model_names[0]]['Label'].reset_index(drop=True)
    for m in model_names[1:]:
        labels = dfs[m]['Label'].reset_index(drop=True)
        if len(labels) != len(ref_labels) or not (labels == ref_labels).all():
            raise ValueError(
                f"Model '{m}' has a different Label sequence than '{model_names[0]}' -- "
                "McNemar's test requires paired predictions on the same records."
            )

    correct = {m: dfs[m]['correct'].values.astype(bool) for m in model_names}

    rows = []
    print(f"\nPairwise McNemar's test ({len(model_names)} models, "
          f"{len(list(itertools.combinations(model_names, 2)))} pairs):\n")
    for m_a, m_b in itertools.combinations(model_names, 2):
        result = mcnemar(correct[m_a], correct[m_b])
        row = {
            'model_a': m_a, 'model_b': m_b,
            'display_a': display_name(m_a), 'display_b': display_name(m_b),
            'acc_a': correct[m_a].mean(), 'acc_b': correct[m_b].mean(),
            **result,
            'significant_at_0.05': result['p_value_exact'] < 0.05,
        }
        rows.append(row)
        sig = '*' if row['significant_at_0.05'] else ' '
        print(f"  {row['display_a']:30s} vs {row['display_b']:30s} "
              f"n01={result['n_a_wrong_b_right']:>2} n10={result['n_a_right_b_wrong']:>2}  "
              f"p_exact={result['p_value_exact']:.4f} {sig}")

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    out_df.to_csv(args.output_file, index=False)
    print(f"\nSaved -> {args.output_file}")
    print("\n* = p < 0.05 (two-sided exact McNemar test)")


if __name__ == '__main__':
    main()
