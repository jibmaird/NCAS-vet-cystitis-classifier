#!/usr/bin/env python3
"""
Error review spreadsheet v2 (sc_project_overview_veterinarians.md,
project_sc_error_review_followup.md).

The multi-seed reruns + gate_drop1 analysis (sc_cascade_gate_all_seeds.py)
identified a new best system -- Multi-Knowledge Gated, seed43, gate_drop1,
F1=0.773 (up from Cross-Modal seed42's F1=0.756, the basis of
sc_error_review_cross_modal_SC.csv). This script builds the same TP/FN/FP
review spreadsheet for that new best system, and carries over any existing
domain-expert feedback (sc_error_review_cross_modal_SC_initial_feedback.csv,
8/11 rows filled in so far) for FN/FP cases that overlap between the two
error sets.

No retraining; CPU only.

Usage:
    python scripts/evaluation/sc_error_review_spreadsheet_v2.py
"""

import os
import sys
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
from sc_phase1_utils import load_predictions, display_name, POSITIVE_LABEL, NEGATIVE_LABEL, PROJECT_ROOT
from sc_error_review_spreadsheet import build_explanation

_PRED_PATH = os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC_seed43/predictions_gated_SC.csv')
_TEST_SOURCE = os.path.join(PROJECT_ROOT, 'data/ann04062026/test-all-syndromes.csv')
_CARRY_OVER_FROM = os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC/sc_error_review_cross_modal_SC_initial_feedback.csv')
_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC/sc_error_review_gated_seed43_SC.csv')

ARCH = 'gated'
MODEL_LABEL = f"{display_name(ARCH)} (seed43)"


def main():
    df = load_predictions(_PRED_PATH)
    if df['text'].duplicated().any():
        raise ValueError(f"{_PRED_PATH} has duplicate 'text' values -- cannot join unambiguously")

    meta = pd.read_csv(_TEST_SOURCE, encoding='latin-1')
    meta_cols = meta[['text', 'consult_table_id', 'patient_species', 'utd_other']]
    if meta_cols['text'].duplicated().any():
        raise ValueError(f"{_TEST_SOURCE} has duplicate 'text' values -- cannot join unambiguously")

    merged = df.merge(meta_cols, on='text', how='left')
    if merged['consult_table_id'].isna().any():
        n_unmatched = merged['consult_table_id'].isna().sum()
        raise ValueError(f"{n_unmatched} record(s) from {_PRED_PATH} did not match {_TEST_SOURCE} on 'text'")

    is_tp = (merged['Label'] == POSITIVE_LABEL) & (merged['predicted_label'] == POSITIVE_LABEL)
    is_fn = (merged['Label'] == POSITIVE_LABEL) & (merged['predicted_label'] == NEGATIVE_LABEL)
    is_fp = (merged['Label'] == NEGATIVE_LABEL) & (merged['predicted_label'] == POSITIVE_LABEL)

    merged['result'] = ''
    merged.loc[is_tp, 'result'] = 'TP'
    merged.loc[is_fn, 'result'] = 'FN'
    merged.loc[is_fp, 'result'] = 'FP'
    review = merged[is_tp | is_fn | is_fp].copy()

    review['ai_explanation'] = [
        build_explanation(row['result'], row['utd_other']) for _, row in review.iterrows()
    ]

    review['consult_table_id'] = review['consult_table_id'].astype(int)
    review['model_p_sporadic_cystitis'] = review['prob_sporadic_cystitis'].round(3)

    review = review.rename(columns={
        'Label': 'true_label',
        'predicted_label': 'model_prediction',
        'patient_species': 'species',
    })

    # Carry over existing domain-expert feedback for overlapping FN/FP cases.
    carry_over = pd.read_csv(_CARRY_OVER_FROM, sep='\t')
    feedback_map = dict(zip(carry_over['consult_table_id'], carry_over['domain_expert_feedback']))
    review['domain_expert_feedback'] = review['consult_table_id'].map(feedback_map).fillna('')

    cols = ['consult_table_id', 'species', 'result', 'true_label', 'model_prediction',
            'model_p_sporadic_cystitis', 'text', 'ai_explanation', 'domain_expert_feedback']
    review = review[cols]

    result_order = {'FN': 0, 'FP': 1, 'TP': 2}
    review['_result_order'] = review['result'].map(result_order)
    review['_prob_sort'] = review['model_p_sporadic_cystitis']
    review.loc[review['result'] == 'TP', '_prob_sort'] *= -1
    review = review.sort_values(['_result_order', '_prob_sort'], ascending=[True, False])
    review = review.drop(columns=['_result_order', '_prob_sort'])

    os.makedirs(os.path.dirname(_OUTPUT_PATH), exist_ok=True)
    review.to_csv(_OUTPUT_PATH, index=False)

    n_fn = int(is_fn.sum())
    n_fp = int(is_fp.sum())
    print(f"{MODEL_LABEL}: {len(review)} records (TP={int(is_tp.sum())}, FN={n_fn}, FP={n_fp})")

    errors = review[review['result'].isin(['FN', 'FP'])]
    n_carried = (errors['domain_expert_feedback'] != '').sum()
    n_needed = (errors['domain_expert_feedback'] == '').sum()
    print(f"Carried over feedback for {n_carried}/{n_fn + n_fp} FN/FP records")
    if n_needed:
        print(f"Still need domain-expert feedback for {n_needed} FN/FP record(s):")
        for _, row in errors[errors['domain_expert_feedback'] == ''].iterrows():
            in_old = row['consult_table_id'] in carry_over['consult_table_id'].values
            tag = '(was in old spreadsheet, feedback pending)' if in_old else '(new -- not in old spreadsheet)'
            print(f"  consult_table_id={row['consult_table_id']} {row['result']} {tag}")

    print(f"Saved -> {_OUTPUT_PATH}")


if __name__ == '__main__':
    main()
