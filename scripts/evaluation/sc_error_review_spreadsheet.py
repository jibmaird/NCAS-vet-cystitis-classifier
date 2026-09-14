#!/usr/bin/env python3
"""
Error review spreadsheet for domain experts (sc_project_overview_veterinarians.md).

Takes the best-performing SC architecture's test_binary_SC.csv predictions
(default: Multi-Knowledge Cross-Modal, F1=0.756) and produces a per-record
spreadsheet covering every TP, FN and FP (TNs excluded -- 472 "correctly
ignored Other" notes are not useful for review).

Adds two columns:
  - 'AI explanation': for FN/FP rows where the original annotation also
    flagged a clinically-related urinary diagnosis (utd_other -- FLUTD, FIC,
    recurrent cystitis, urolithiasis, "unclear"), a short note on that overlap
    and why it may explain the model's error. Left blank for TPs, and for
    FN/FP rows with no such overlap (no fabricated explanation).
  - 'domain_expert_feedback': empty, for the annotation team to fill in.

No retraining; CPU only.

Usage:
    python scripts/evaluation/sc_error_review_spreadsheet.py
    python scripts/evaluation/sc_error_review_spreadsheet.py --model cross_modal
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

# Maps a (lowercased) utd_other annotation to a short, factual note on its
# clinical relationship to Sporadic Cystitis. Only categories observed in the
# Cross-Modal FN/FP set are covered; anything else is left unexplained rather
# than guessed.
UTD_OTHER_NOTES = {
    'flutd': "FLUTD (Feline Lower Urinary Tract Disease) -- a broader urinary "
             "syndrome that clinically overlaps with Sporadic Cystitis",
    'urolithiasis': "urolithiasis (bladder/urinary stones) -- shares signs "
                    "such as haematuria, dysuria and straining with Sporadic Cystitis",
    'recurrent cystitis': "Recurrent Cystitis -- distinguished from Sporadic "
                          "Cystitis mainly by history of prior episodes, not "
                          "by this note's presentation alone",
    'fic': "FIC (Feline Idiopathic Cystitis) -- shares signs such as "
           "stranguria and haematuria with Sporadic Cystitis",
    'unclear': "the annotators themselves marked this case 'unclear'",
}


def build_explanation(error_type: str, utd_other) -> str:
    if error_type not in ('FN', 'FP'):
        return ''
    if not isinstance(utd_other, str) or not utd_other.strip():
        return ''
    note = UTD_OTHER_NOTES.get(utd_other.strip().lower())
    if note is None:
        return ''
    if utd_other.strip().lower() == 'unclear':
        if error_type == 'FN':
            return (f"Model predicted 'Other' for a true Sporadic Cystitis case, but "
                    f"{note} -- this may be a genuinely borderline note.")
        return (f"Model predicted 'Sporadic Cystitis', but the true label is 'Other'; "
                f"{note} -- this may be a genuinely borderline note.")
    if error_type == 'FN':
        return (f"Model predicted 'Other' for a true Sporadic Cystitis case. This note "
                f"was also annotated as {note}, which may explain the model's miss.")
    return (f"Model predicted 'Sporadic Cystitis', but the true label is 'Other'. "
            f"This note was instead annotated as {note}, which may explain the model's false alarm.")


def main():
    parser = argparse.ArgumentParser(description='Build TP/FN/FP error-review spreadsheet for domain experts')
    parser.add_argument('--model', default='cross_modal',
                         help="Model key as used by sc_phase1_utils (default: cross_modal, "
                              "i.e. Multi-Knowledge Cross-Modal, the best Run A architecture)")
    parser.add_argument('--test_source', default=_TEST_SOURCE_DEFAULT,
                         help='Original annotation CSV to recover consult_table_id / patient_species / utd_other from')
    parser.add_argument('--output_dir', default=_OUTPUT_DIR_DEFAULT)
    args = parser.parse_args()

    predictions = discover_predictions()
    if args.model not in predictions:
        raise ValueError(f"Model '{args.model}' not found. Available: {sorted(predictions)}")

    df = load_predictions(predictions[args.model])
    if df['text'].duplicated().any():
        raise ValueError(f"{predictions[args.model]} has duplicate 'text' values -- cannot join unambiguously")

    meta = pd.read_csv(args.test_source, encoding='latin-1')
    meta_cols = meta[['text', 'consult_table_id', 'patient_species', 'utd_other']]
    if meta_cols['text'].duplicated().any():
        raise ValueError(f"{args.test_source} has duplicate 'text' values -- cannot join unambiguously")

    merged = df.merge(meta_cols, on='text', how='left')
    if merged['consult_table_id'].isna().any():
        n_unmatched = merged['consult_table_id'].isna().sum()
        raise ValueError(
            f"{n_unmatched} record(s) from {predictions[args.model]} did not match "
            f"{args.test_source} on 'text' -- check the prediction file is for test_binary_SC.csv"
        )

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
    review['domain_expert_feedback'] = ''

    review['consult_table_id'] = review['consult_table_id'].astype(int)
    review['model_p_sporadic_cystitis'] = review['prob_sporadic_cystitis'].round(3)

    review = review.rename(columns={
        'Label': 'true_label',
        'predicted_label': 'model_prediction',
        'patient_species': 'species',
    })

    cols = ['consult_table_id', 'species', 'result', 'true_label', 'model_prediction',
            'model_p_sporadic_cystitis', 'text', 'ai_explanation', 'domain_expert_feedback']
    review = review[cols]

    result_order = {'FN': 0, 'FP': 1, 'TP': 2}
    review['_result_order'] = review['result'].map(result_order)
    # FN/FP: most-confident errors first. TP: least-confident (most borderline) first.
    review['_prob_sort'] = review['model_p_sporadic_cystitis']
    review.loc[review['result'] == 'TP', '_prob_sort'] *= -1
    review = review.sort_values(['_result_order', '_prob_sort'], ascending=[True, False])
    review = review.drop(columns=['_result_order', '_prob_sort'])

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'sc_error_review_{args.model}_SC.csv')
    review.to_csv(out_path, index=False)

    print(f"{display_name(args.model)}: {len(review)} records "
          f"(TP={int(is_tp.sum())}, FN={int(is_fn.sum())}, FP={int(is_fp.sum())})")
    n_explained = (review['ai_explanation'] != '').sum()
    print(f"AI explanation populated for {n_explained}/{int(is_fn.sum() + is_fp.sum())} FN/FP records")
    print(f"Saved -> {out_path}")


if __name__ == '__main__':
    main()
