#!/usr/bin/env python3
"""Test-set results of Systems A, B and C for the revised paper (sc-kv-revision, Step 4).

System A = end-to-end classifiers (emb-SVM; fine-tuned VetBERTDx and MK Gated, seeds 42-44 + soft vote).
System B = A + UTD filter (SC predictions kept only if P(UTD) >= gate threshold).
System C = two-stage pipeline (gate, then the Stage-2 VetBERTDx fold ensemble); 'no_gate' = the Stage-2
           classifier applied to every note, for reference.

For every run this adds the false-positive breakdown (Other-UTD vs non-UTD) to the F1 and shared bootstrap CI
from sc_shared_bootstrap_ci.py, runs exact McNemar tests for the pairs discussed in the paper, and reports how
System C's errors overlap with the errors of MK Gated seed 43 at gate_thr1 that the domain expert reviewed.

Outputs (results/evaluation/ann04062026_SC/):
  systems_abc_summary.csv, systems_abc_mcnemar.csv, systems_abc_error_overlap.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sc_phase1_utils import POSITIVE_LABEL, NEGATIVE_LABEL, PROJECT_ROOT, load_predictions
from sc_two_stage_evaluation import build_is_utd
from sc_mcnemar_test import mcnemar

RES = os.path.join(PROJECT_ROOT, 'results/evaluation')
SC_RES = os.path.join(RES, 'ann04062026_SC')
PRED_DIRS = {42: SC_RES, 43: os.path.join(RES, 'ann04062026_SC_seed43'), 44: os.path.join(RES, 'ann04062026_SC_seed44')}
SVM = os.path.join(PROJECT_ROOT, 'results/predictions/ann04062026_SC/predictions_ml_embeddings_only_svm_SC.csv')
TP_DIR = os.path.join(SC_RES, 'true_pipeline')
UTD_PROBS = os.path.join(SC_RES, 'utd_gate_probs_SC.csv')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data/ann04062026')
SEEDS = [42, 43, 44]
NAMES = {'ml_embeddings_only_svm': 'SVM', 'vetbert_finetune': 'Fine-tuned VetBERTDx', 'gated': 'MK Gated',
         'true_pipeline_vetbert': 'Stage-2 classifier'}


def main():
    g = pd.read_csv(UTD_PROBS)
    test = g[g['split'] == 'test'].reset_index(drop=True)
    assert test['text'].is_unique
    truth = build_is_utd(os.path.join(DATA_DIR, 'test-all-syndromes.csv')).drop_duplicates('text')
    test = test.merge(truth[['text', 'is_utd']], on='text', how='left', validate='one_to_one')
    assert test['is_utd'].notna().all()
    y = test['Label'].values
    is_utd = test['is_utd'].values.astype(bool)
    p_utd = test['p_urinary_tract_disorder'].values
    dev = g[g['split'] == 'dev']
    sc_dev = np.sort(dev.loc[dev['Label'] == POSITIVE_LABEL, 'p_urinary_tract_disorder'].values)
    gates = {'gate_thr0': sc_dev[0], 'gate_thr1': sc_dev[1]}

    def aligned(path):
        df = load_predictions(path).set_index('text')
        return df.loc[test['text']].reset_index()

    # predictions per (system, classifier, run, method)
    preds = {}
    svm = aligned(SVM)['predicted_label'].values
    preds[('A', 'ml_embeddings_only_svm', 'single run', 'ungated')] = svm
    for m, t in gates.items():
        preds[('B', 'ml_embeddings_only_svm', 'single run', m)] = np.where(p_utd >= t, svm, NEGATIVE_LABEL)
    for arch in ('vetbert_finetune', 'gated'):
        dfs = {s: aligned(os.path.join(PRED_DIRS[s], f'predictions_{arch}_SC.csv')) for s in SEEDS}
        runs = {f'seed {s}': dfs[s]['predicted_label'].values for s in SEEDS}
        avg = np.mean([dfs[s]['prob_sporadic_cystitis'].values for s in SEEDS], axis=0)
        runs['soft vote'] = np.where(avg >= 0.5, POSITIVE_LABEL, NEGATIVE_LABEL)
        for run, yp in runs.items():
            preds[('A', arch, run, 'ungated')] = yp
            for m, t in gates.items():
                preds[('B', arch, run, m)] = np.where(p_utd >= t, yp, NEGATIVE_LABEL)
    for run in ('seed42', 'seed43', 'seed44', 'soft_vote'):
        for m in ('no_gate', 'gate_thr0', 'gate_thr1'):
            f = os.path.join(TP_DIR, f'predictions_true_pipeline_{run}_{m}_SC.csv')
            preds[('C', 'true_pipeline_vetbert', run.replace('seed', 'seed ').replace('_', ' '), m)] = \
                aligned(f)['predicted_label'].values

    boot = pd.read_csv(os.path.join(SC_RES, 'shared_bootstrap_ci.csv'))
    boot['run'] = boot['run'].str.replace('seed', 'seed ').str.replace('_', ' ')
    boot.loc[boot['system'] == 'ml_embeddings_only_svm', 'run'] = 'single run'
    rows = []
    for (system, clf, run, method), yp in preds.items():
        pos, true = yp == POSITIVE_LABEL, y == POSITIVE_LABEL
        b = boot[(boot['system'] == clf) & (boot['run'] == run) & (boot['method'] == method)]
        assert len(b) == 1, (clf, run, method)
        b = b.iloc[0]
        tp, fp, fn = int((pos & true).sum()), int((pos & ~true).sum()), int((~pos & true).sum())
        assert (tp, fp, fn) == (b['tp'], b['fp'], b['fn']), (clf, run, method)
        rows.append({'system': system, 'classifier': NAMES[clf], 'run': run, 'method': method,
                     'f1': round(b['f1_sc'], 3), 'ci_lower': round(b['ci_lower'], 3), 'ci_upper': round(b['ci_upper'], 3),
                     'precision': round(b['precision_sc'], 3), 'recall': round(b['recall_sc'], 3), 'tp': tp, 'fp': fp,
                     'fp_other_utd': int((pos & ~true & is_utd).sum()), 'fp_non_utd': int((pos & ~true & ~is_utd).sum()),
                     'fn': fn, 'n_pass_gate': int((p_utd >= gates[method]).sum()) if method in gates else 500})
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(SC_RES, 'systems_abc_summary.csv'), index=False)

    def correct(key):
        return preds[key] == y

    pairs = [
        (('C', 'true_pipeline_vetbert', 'soft vote', 'gate_thr1'), ('B', 'gated', 'soft vote', 'gate_thr1')),
        (('C', 'true_pipeline_vetbert', 'soft vote', 'gate_thr1'), ('B', 'vetbert_finetune', 'soft vote', 'gate_thr1')),
        (('C', 'true_pipeline_vetbert', 'soft vote', 'gate_thr1'), ('B', 'ml_embeddings_only_svm', 'single run', 'gate_thr1')),
        (('B', 'gated', 'soft vote', 'gate_thr1'), ('A', 'gated', 'soft vote', 'ungated')),
        (('B', 'vetbert_finetune', 'soft vote', 'gate_thr1'), ('A', 'vetbert_finetune', 'soft vote', 'ungated')),
    ] + [(('C', 'true_pipeline_vetbert', f'seed {s}', 'gate_thr1'), ('B', 'gated', f'seed {s}', 'gate_thr1')) for s in SEEDS] \
      + [(('C', 'true_pipeline_vetbert', f'seed {s}', 'gate_thr1'), ('B', 'vetbert_finetune', f'seed {s}', 'gate_thr1')) for s in SEEDS]
    mc = []
    for a, b in pairs:
        r = mcnemar(correct(a), correct(b))
        mc.append({'a': ' / '.join(a), 'b': ' / '.join(b), **{k: r[k] for k in r}})
    mc = pd.DataFrame(mc)
    mc.to_csv(os.path.join(SC_RES, 'systems_abc_mcnemar.csv'), index=False)

    # overlap with the errors the domain expert reviewed (MK Gated seed 43, gate_thr1)
    ref = preds[('B', 'gated', 'seed 43', 'gate_thr1')]
    ref_fp, ref_fn = (ref == POSITIVE_LABEL) & (y != POSITIVE_LABEL), (ref != POSITIVE_LABEL) & (y == POSITIVE_LABEL)
    ov = []
    for run in ('seed 42', 'seed 43', 'seed 44', 'soft vote'):
        yp = preds[('C', 'true_pipeline_vetbert', run, 'gate_thr1')]
        fp, fn = (yp == POSITIVE_LABEL) & (y != POSITIVE_LABEL), (yp != POSITIVE_LABEL) & (y == POSITIVE_LABEL)
        ov.append({'run': run, 'fp': int(fp.sum()), 'fp_also_reviewed': int((fp & ref_fp).sum()),
                   'fn': int(fn.sum()), 'fn_also_reviewed': int((fn & ref_fn).sum())})
    ov = pd.DataFrame(ov)
    ov.to_csv(os.path.join(SC_RES, 'systems_abc_error_overlap.csv'), index=False)

    pd.set_option('display.width', 250)
    print(summary.to_string(index=False))
    print(mc[['a', 'b', 'n_discordant' if 'n_discordant' in mc else mc.columns[2], 'p_value_exact']].to_string(index=False))
    print(ov.to_string(index=False))


if __name__ == '__main__':
    main()
