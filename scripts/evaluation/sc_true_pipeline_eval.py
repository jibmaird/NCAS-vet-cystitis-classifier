#!/usr/bin/env python3
"""
True two-stage pipeline (sc-kv-revision, 2026-11-08): Stage-1 UTD gate -> Stage-2 SC classifier.

Stage 1 is the frozen VetBERTDx P(UTD) score with the dev-derived thresholds gate_thr0/gate_thr1
(lowest / second-lowest P(UTD) among dev SC notes, as in sc_shared_bootstrap_ci.py). Stage 2 is the
fine-tuned VetBERT trained only on the 804 UTD training notes (SC vs Other-UTD) in 5-fold CV
(sc_stage2_vetbert_cv.py --predict); its five fold models are averaged into one P(SC) per note.
A note is predicted SC only if it passes the gate and the Stage-2 ensemble gives P(SC) >= 0.5;
every note the gate rejects is Other. This differs from the "end-to-end + UTD filter" systems in
Table 6, where the gate only overrides a single-stage SC-vs-Other model.

For each training seed (42, 43, 44) and the 3-seed soft vote (mean P(SC)), reports dev and test
precision/recall/F1(SC) with TP/FP/FN/TN, the false positives split into Other-UTD vs non-UTD notes,
and F1(SC) within the gold-UTD subset. "no_gate" applies Stage 2 to every note, to show what the
gate contributes. Confidence intervals come from sc_shared_bootstrap_ci.py, which reads the
prediction files written here.

Inputs:
  results/evaluation/ann04062026_SC/utd_gate_probs_SC.csv
  results/evaluation/ann04062026_SC/true_pipeline/stage2_vetbert_seed{S}/{dev,test}_binary_SC_fold{k}_predictions.csv
  data/ann04062026/{dev,test}-all-syndromes.csv            (is_utd ground truth)
Outputs (results/evaluation/ann04062026_SC/true_pipeline/):
  true_pipeline_summary.csv
  predictions_true_pipeline_{run}_{method}_SC.csv           (test set, same row order as the gate file)

Usage:
    python scripts/evaluation/sc_true_pipeline_eval.py
"""
import os
import sys

import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
from sc_phase1_utils import POSITIVE_LABEL, NEGATIVE_LABEL, PROJECT_ROOT
from sc_two_stage_evaluation import build_is_utd

RES = os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC')
OUT_DIR = os.path.join(RES, 'true_pipeline')
UTD_PROBS = os.path.join(RES, 'utd_gate_probs_SC.csv')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data/ann04062026')
SEEDS = [42, 43, 44]
N_FOLDS = 5
SC_THRESHOLD = 0.5


def stage2_probs(seed, split):
    """Mean P(SC) of the five fold models for every note of `split`, in the gate file's row order."""
    d = os.path.join(OUT_DIR, f'stage2_vetbert_seed{seed}')
    folds = [pd.read_csv(os.path.join(d, f'{split}_binary_SC_fold{k}_predictions.csv')) for k in range(N_FOLDS)]
    for f in folds[1:]:
        assert (f['text'].values == folds[0]['text'].values).all(), f'{d}: fold files are not aligned'
    return folds[0]['text'].values, np.mean([f['prob_sc'].values for f in folds], axis=0)


def counts(y_true, y_pred):
    t, p = y_true == POSITIVE_LABEL, y_pred == POSITIVE_LABEL
    tp, fp, fn, tn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum()), int((~p & ~t).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1)


def main():
    gate_df = pd.read_csv(UTD_PROBS)
    dev = gate_df[gate_df['split'] == 'dev'].reset_index(drop=True)
    sc_dev = np.sort(dev.loc[dev['Label'] == POSITIVE_LABEL, 'p_urinary_tract_disorder'].values)
    gates = {'gate_thr0': sc_dev[0], 'gate_thr1': sc_dev[1], 'no_gate': -np.inf}
    print('gate thresholds:', {k: float(v) for k, v in gates.items() if k != 'no_gate'})

    rows = []
    for split in ('dev', 'test'):
        part = gate_df[gate_df['split'] == split].reset_index(drop=True)
        truth = build_is_utd(os.path.join(DATA_DIR, f'{split}-all-syndromes.csv'))
        # one dev note occurs three times (identical text and label); keep one truth row per text so the merge
        # cannot multiply rows
        assert (truth.groupby('text')['is_utd'].nunique() == 1).all(), f'{split}: duplicate texts with conflicting UTD labels'
        truth = truth.drop_duplicates('text')
        part = part.merge(truth[['text', 'is_utd']], on='text', how='left', validate='many_to_one')
        assert part['is_utd'].notna().all(), f'{split}: notes missing from the annotation file'
        is_utd = part['is_utd'].values.astype(bool)
        y_true = part['Label'].values
        p_utd = part['p_urinary_tract_disorder'].values

        probs = {}
        for s in SEEDS:
            text, p = stage2_probs(s, split)
            assert (text == part['text'].values).all(), f'seed {s} {split}: rows not aligned with the gate file'
            probs[f'seed{s}'] = p
        probs['soft_vote'] = np.mean([probs[f'seed{s}'] for s in SEEDS], axis=0)

        for run, p_sc in probs.items():
            for method, thr in gates.items():
                passed = p_utd >= thr
                y_pred = np.where(passed & (p_sc >= SC_THRESHOLD), POSITIVE_LABEL, NEGATIVE_LABEL)
                c = counts(y_true, y_pred)
                fp_mask = (y_pred == POSITIVE_LABEL) & (y_true != POSITIVE_LABEL)
                w = counts(y_true[is_utd], y_pred[is_utd])
                rows.append({'split': split, 'run': run, 'method': method, 'n': len(y_true),
                             'n_pass_gate': int(passed.sum()), **c,
                             'fp_other_utd': int((fp_mask & is_utd).sum()), 'fp_non_utd': int((fp_mask & ~is_utd).sum()),
                             'within_utd_n': int(is_utd.sum()), 'within_utd_tp': w['tp'], 'within_utd_fp': w['fp'],
                             'within_utd_fn': w['fn'], 'within_utd_tn': w['tn'], 'within_utd_f1': w['f1']})
                if split == 'test':
                    pd.DataFrame({'text': part['text'], 'Label': y_true, 'predicted_label': y_pred,
                                  'prob_sporadic_cystitis': p_sc, 'prob_other': 1 - p_sc,
                                  'p_urinary_tract_disorder': p_utd}).to_csv(
                        os.path.join(OUT_DIR, f'predictions_true_pipeline_{run}_{method}_SC.csv'), index=False)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, 'true_pipeline_summary.csv'), index=False)
    show = out[['split', 'run', 'method', 'n_pass_gate', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1',
                'fp_other_utd', 'fp_non_utd', 'within_utd_f1']]
    print(show.round(3).to_string(index=False))
    print(f"\nSaved -> {os.path.join(OUT_DIR, 'true_pipeline_summary.csv')}")


if __name__ == '__main__':
    main()
