#!/usr/bin/env python3
"""
Test-set F1(SC) 95% CIs for every system reported in the paper, computed on ONE shared set of
stratified bootstrap resamples (positives and negatives resampled separately, 21/479 kept).

Earlier scripts (sc_bootstrap_ci.py, sc_cascade_gate_all_seeds.py) drew fresh resamples for each
system from a running generator, so identical predictions could get different CIs. Here the
resample indices are drawn once and reused, so identical predictions always give identical CIs
and every interval refers to the same resamples.

Covers: the 18 seed-42 single-stage systems (Table 5 / Appendix A1) and, for the re-run
architectures (the four fusion models plus fine-tuned VetBERT), seeds 42/43/44 and the 3-seed
soft vote, ungated and at gate_thr0/gate_thr1 (Table 6). If sc_true_pipeline_eval.py has been
run, also the true two-stage pipeline (gate -> Stage-2 VetBERT) for each seed and the soft vote.

Usage: python scripts/evaluation/sc_shared_bootstrap_ci.py [--n_bootstrap 2000] [--seed 42]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sc_phase1_utils import (load_predictions, binary_metrics, display_name, POSITIVE_LABEL,
                             NEGATIVE_LABEL, PROJECT_ROOT)

RES = os.path.join(PROJECT_ROOT, 'results/evaluation')
PRED_DIRS = {42: os.path.join(RES, 'ann04062026_SC'),
             43: os.path.join(RES, 'ann04062026_SC_seed43'),
             44: os.path.join(RES, 'ann04062026_SC_seed44')}
ML_DIR = os.path.join(PROJECT_ROOT, 'results/predictions/ann04062026_SC')
UTD_PROBS = os.path.join(RES, 'ann04062026_SC/utd_gate_probs_SC.csv')
OUT = os.path.join(RES, 'ann04062026_SC/shared_bootstrap_ci.csv')

DL_SEED42 = ['cross_modal', 'dualhead_cross_attention', 'dualhead_weighted', 'gated', 'hierarchical',
             'vetbert_finetune', 'dualhead_multi_task', 'attention', 'dualhead_concatenation']
ML_SEED42 = ['syndromes_only_svm', 'hybrid_equal_svm', 'embeddings_only_svm', 'hybrid_equal_random_forest',
             'embeddings_only_random_forest', 'embeddings_only_logistic', 'hybrid_equal_logistic',
             'syndromes_only_logistic', 'syndromes_only_random_forest']
ML_GATED = ['embeddings_only_svm']   # selected on development macro-F1 (Section 2.6)
RERUN_ARCHS = ['gated', 'cross_modal', 'dualhead_cross_attention', 'dualhead_weighted', 'vetbert_finetune']
SEEDS = [42, 43, 44]
TRUE_PIPELINE_DIR = os.path.join(RES, 'ann04062026_SC/true_pipeline')
TRUE_PIPELINE_RUNS = ['seed42', 'seed43', 'seed44', 'soft_vote']
TRUE_PIPELINE_METHODS = ['gate_thr0', 'gate_thr1', 'no_gate']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_bootstrap', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output_file', default=OUT)
    args = ap.parse_args()

    test = pd.read_csv(UTD_PROBS)
    test = test[test['split'] == 'test'].reset_index(drop=True)
    assert test['text'].is_unique
    y_true = test['Label'].values
    p_utd = test['p_urinary_tract_disorder'].values

    dev = pd.read_csv(UTD_PROBS)
    dev = dev[dev['split'] == 'dev']
    sc_dev = np.sort(dev.loc[dev['Label'] == POSITIVE_LABEL, 'p_urinary_tract_disorder'].values)
    gates = {'gate_thr0': sc_dev[0], 'gate_thr1': sc_dev[1]}

    # one shared set of stratified resamples
    rng = np.random.default_rng(args.seed)
    pos = np.where(y_true == POSITIVE_LABEL)[0]
    neg = np.where(y_true == NEGATIVE_LABEL)[0]
    boots = [np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
             for _ in range(args.n_bootstrap)]

    def aligned(path):
        df = load_predictions(path).set_index('text')
        missing = set(test['text']) - set(df.index)
        assert not missing, f'{path}: {len(missing)} test records missing'
        return df.loc[test['text']].reset_index()

    boots = np.stack(boots)
    true_pos = (y_true == POSITIVE_LABEL)
    rows = []

    def add(system, run, method, y_pred):
        m = binary_metrics(y_true, y_pred)
        pred_pos = (y_pred == POSITIVE_LABEL)
        tp_b = (pred_pos & true_pos)[boots].sum(axis=1)
        fp_b = (pred_pos & ~true_pos)[boots].sum(axis=1)
        fn_b = (~pred_pos & true_pos)[boots].sum(axis=1)
        denom = 2 * tp_b + fp_b + fn_b
        f1s = np.where(denom > 0, 2 * tp_b / np.maximum(denom, 1), 0.0)  # same as sklearn, zero_division=0
        tp = int(((y_pred == POSITIVE_LABEL) & (y_true == POSITIVE_LABEL)).sum())
        fp = int(((y_pred == POSITIVE_LABEL) & (y_true == NEGATIVE_LABEL)).sum())
        fn = int(((y_pred == NEGATIVE_LABEL) & (y_true == POSITIVE_LABEL)).sum())
        rows.append({'system': system, 'display_name': display_name(system), 'run': run, 'method': method,
                     'f1_sc': m['f1_sc'], 'precision_sc': m['precision_sc'], 'recall_sc': m['recall_sc'],
                     'tp': tp, 'fp': fp, 'fn': fn,
                     'ci_lower': np.quantile(f1s, 0.025), 'ci_upper': np.quantile(f1s, 0.975)})

    for s in DL_SEED42:
        add(s, 'seed42', 'ungated', aligned(os.path.join(PRED_DIRS[42], f'predictions_{s}_SC.csv'))['predicted_label'].values)
    for s in ML_SEED42:
        y_pred = aligned(os.path.join(ML_DIR, f'predictions_ml_{s}_SC.csv'))['predicted_label'].values
        add('ml_' + s, 'seed42', 'ungated', y_pred)
        if s in ML_GATED:  # System B for the traditional-ML classifier kept in the main text
            for g, t in gates.items():
                add('ml_' + s, 'seed42', g, np.where(p_utd >= t, y_pred, NEGATIVE_LABEL))

    for arch in RERUN_ARCHS:
        dfs = {seed: aligned(os.path.join(PRED_DIRS[seed], f'predictions_{arch}_SC.csv')) for seed in SEEDS}
        avg = np.mean([dfs[s]['prob_sporadic_cystitis'].values for s in SEEDS], axis=0)
        runs = {f'seed{s}': dfs[s]['predicted_label'].values for s in SEEDS}
        runs['soft_vote'] = np.where(avg >= 0.5, POSITIVE_LABEL, NEGATIVE_LABEL)
        for run, y_pred in runs.items():
            if not (arch in DL_SEED42 and run == 'seed42'):  # seed-42 ungated already added above
                add(arch, run, 'ungated', y_pred)
            for g, t in gates.items():
                add(arch, run, g, np.where(p_utd >= t, y_pred, NEGATIVE_LABEL))

    # True two-stage pipeline (gate -> Stage-2 VetBERT); its gating is already in the predictions.
    tp_files = {(r, m): os.path.join(TRUE_PIPELINE_DIR, f'predictions_true_pipeline_{r}_{m}_SC.csv')
                for r in TRUE_PIPELINE_RUNS for m in TRUE_PIPELINE_METHODS}
    if all(os.path.exists(f) for f in tp_files.values()):
        for (run, method), f in tp_files.items():
            add('true_pipeline_vetbert', run, method, aligned(f)['predicted_label'].values)
    else:
        print('true-pipeline predictions not found; run sc_true_pipeline_eval.py to include them')

    out = pd.DataFrame(rows)
    out.to_csv(args.output_file, index=False)
    print(f'{len(out)} rows -> {args.output_file}')


if __name__ == '__main__':
    main()
