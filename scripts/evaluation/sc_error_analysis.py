#!/usr/bin/env python3
"""
Phase 1d (sc_experiment_plan.md): Qualitative error analysis for SC models.

For the top-performing models (default: Multi-Knowledge Cross-Modal, Dual-Head
Cross-Attention, Dual-Head Weighted), pulls every false negative (FN: actual
Sporadic Cystitis, predicted Other) and false positive (FP: actual Other,
predicted Sporadic Cystitis) on the test_binary_SC.csv ("Run A") test set, and
characterises them by:
  - species (cat/dog)
  - text length (word count)
  - presence of cystitis-adjacent terminology (keyword match)
  - overlap with the original utd_other / utd_journey annotation columns

Per-record metadata (patient_species, utd_other, utd_journey,
consult_table_id) is recovered by joining the prediction CSV back to
data/ann04062026/test-all-syndromes.csv on the 'text' column (verified 1:1,
no duplicate texts in either file).

No retraining; CPU only.

Usage:
    python scripts/evaluation/sc_error_analysis.py
    python scripts/evaluation/sc_error_analysis.py --models cross_modal gated ml_syndromes_only_svm
"""

import os
import sys
import argparse
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
from sc_phase1_utils import discover_predictions, load_predictions, display_name, POSITIVE_LABEL, NEGATIVE_LABEL, PROJECT_ROOT

_OUTPUT_DIR_DEFAULT = os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC')
_TEST_SOURCE_DEFAULT = os.path.join(PROJECT_ROOT, 'data/ann04062026/test-all-syndromes.csv')

DEFAULT_MODELS = ['cross_modal', 'dualhead_cross_attention', 'dualhead_weighted']

# Cystitis-adjacent terminology -- substring match, case-insensitive.
CYSTITIS_KEYWORDS = [
    'cystitis', 'flutd', 'fic', 'urinat', 'haematuria', 'hematuria', 'stranguria',
    'pollakiuria', 'dysuria', 'bladder', 'urinary', 'uti', 'urolith', 'urethr',
    'crystalluria',
]


def has_cystitis_keyword(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in CYSTITIS_KEYWORDS)


def main():
    parser = argparse.ArgumentParser(description='Qualitative FN/FP error analysis for SC models')
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS)
    parser.add_argument('--test_source', default=_TEST_SOURCE_DEFAULT,
                         help='Original annotation CSV to recover species / utd_other / utd_journey from')
    parser.add_argument('--output_dir', default=_OUTPUT_DIR_DEFAULT)
    args = parser.parse_args()

    predictions = discover_predictions()
    missing = [m for m in args.models if m not in predictions]
    if missing:
        raise ValueError(f"Models not found: {missing}. Available: {sorted(predictions)}")

    meta = pd.read_csv(args.test_source, encoding='latin-1')
    meta_cols = meta[['text', 'consult_table_id', 'patient_species', 'utd_other', 'utd_journey']]
    if meta_cols['text'].duplicated().any():
        raise ValueError(f"{args.test_source} has duplicate 'text' values -- cannot join unambiguously")

    os.makedirs(args.output_dir, exist_ok=True)

    overview_rows = []
    for model_name in args.models:
        df = load_predictions(predictions[model_name])
        if df['text'].duplicated().any():
            raise ValueError(f"{predictions[model_name]} has duplicate 'text' values -- cannot join unambiguously")

        merged = df.merge(meta_cols, on='text', how='left')
        if merged['patient_species'].isna().any():
            n_unmatched = merged['patient_species'].isna().sum()
            raise ValueError(
                f"{n_unmatched} record(s) from {predictions[model_name]} did not match "
                f"{args.test_source} on 'text' -- check the prediction file is for test_binary_SC.csv"
            )

        merged['text_length_words'] = merged['text'].str.split().str.len()
        merged['has_cystitis_keyword'] = merged['text'].apply(has_cystitis_keyword)
        merged['has_utd_other'] = merged['utd_other'].notna()
        merged['has_utd_journey'] = merged['utd_journey'] == 1.0

        fn = merged[(merged['Label'] == POSITIVE_LABEL) & (merged['predicted_label'] == NEGATIVE_LABEL)].copy()
        fp = merged[(merged['Label'] == NEGATIVE_LABEL) & (merged['predicted_label'] == POSITIVE_LABEL)].copy()
        fn['error_type'] = 'FN'
        fp['error_type'] = 'FP'

        errors = pd.concat([fn, fp], ignore_index=True)
        cols = ['error_type', 'consult_table_id', 'patient_species', 'text_length_words',
                'prob_sporadic_cystitis', 'has_cystitis_keyword', 'has_utd_other', 'utd_other',
                'has_utd_journey', 'text']
        errors = errors[cols].sort_values(['error_type', 'prob_sporadic_cystitis'],
                                           ascending=[True, False])

        out_path = os.path.join(args.output_dir, f'phase1_errors_{model_name}_SC.csv')
        errors.to_csv(out_path, index=False)

        print(f"\n{'='*70}")
        print(f"  {display_name(model_name)}")
        print(f"{'='*70}")
        print(f"  FN: {len(fn)} / {(merged['Label'] == POSITIVE_LABEL).sum()} actual SC missed")
        print(f"  FP: {len(fp)} / {(merged['Label'] == NEGATIVE_LABEL).sum()} actual Other misclassified as SC")
        for err_type, sub in [('FN', fn), ('FP', fp)]:
            if len(sub) == 0:
                continue
            print(f"\n  {err_type} (n={len(sub)}):")
            print(f"    species          : {dict(sub['patient_species'].value_counts())}")
            print(f"    mean text length : {sub['text_length_words'].mean():.0f} words "
                  f"(overall test mean: {merged['text_length_words'].mean():.0f})")
            print(f"    has_cystitis_keyword : {int(sub['has_cystitis_keyword'].sum())}/{len(sub)}")
            print(f"    has_utd_other        : {int(sub['has_utd_other'].sum())}/{len(sub)}")
            if sub['has_utd_other'].any():
                print(f"      values: {dict(sub.loc[sub['has_utd_other'], 'utd_other'].value_counts())}")
            print(f"    has_utd_journey      : {int(sub['has_utd_journey'].sum())}/{len(sub)}")

        print(f"\n  Saved -> {out_path}")

        overview_rows.append({
            'model': model_name,
            'display_name': display_name(model_name),
            'n_fn': len(fn),
            'n_fp': len(fp),
            'fn_with_cystitis_keyword': int(fn['has_cystitis_keyword'].sum()) if len(fn) else 0,
            'fp_with_cystitis_keyword': int(fp['has_cystitis_keyword'].sum()) if len(fp) else 0,
            'fn_with_utd_other': int(fn['has_utd_other'].sum()) if len(fn) else 0,
            'fp_with_utd_other': int(fp['has_utd_other'].sum()) if len(fp) else 0,
            'fn_with_utd_journey': int(fn['has_utd_journey'].sum()) if len(fn) else 0,
            'fp_with_utd_journey': int(fp['has_utd_journey'].sum()) if len(fp) else 0,
        })

    overview_df = pd.DataFrame(overview_rows)
    overview_path = os.path.join(args.output_dir, 'phase1_error_overview.csv')
    overview_df.to_csv(overview_path, index=False)
    print(f"\n{'='*70}")
    print(f"Saved overview -> {overview_path}")


if __name__ == '__main__':
    main()
