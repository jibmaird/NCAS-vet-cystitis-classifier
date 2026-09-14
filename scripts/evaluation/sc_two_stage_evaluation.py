#!/usr/bin/env python3
"""
Two-stage evaluation decomposition (sc_experiment_plan.md follow-up).

Stage 1 -- "is this note about a urinary tract disorder (UTD) at all?"
  Evaluated via the UTD-gate probability (p_urinary_tract_disorder, from
  utd_gate_probs_SC.csv) against an is_utd ground truth built from
  test-all-syndromes.csv (is_sc OR an utd_other label is present).
  Test set: 34 positive / 466 negative (n=500).

Stage 2 -- "given it's UTD, is it specifically Sporadic Cystitis vs. a close
  clinical relative (FLUTD/FIC/recurrent cystitis/urolithiasis/etc.)?"
  Evaluated by restricting each architecture/seed's existing
  predictions_*_SC.csv to the is_utd subset and recomputing precision/recall/
  F1_sc + bootstrap CI there.
  Test subset: 21 SC-positive / 13 other-UTD (n=34). Note 9 of the 21 SC-
  positive notes ALSO carry an utd_other annotation (e.g. "FLUTD") -- so the
  is_utd set (34) is smaller than is_sc + is_other_utd (21+22=43).

Purpose: diagnose whether the wide seed-to-seed / bootstrap-CI swings in the
full-test-set F1_sc are concentrated in stage 1 (larger n=34, likely easier
and more stable) or stage 2 (SC vs. its closest clinical relatives -- same 21
SC positives but only 13 negatives, n=34). This does NOT shrink the CIs --
both stages still draw on the same ~17-21 SC-positive test notes -- it only
locates where the instability lives.

Requires:
  - data/ann04062026/{dev,test}-all-syndromes.csv          (is_utd ground truth)
  - results/evaluation/ann04062026_SC/utd_gate_probs_SC.csv (stage-1 gate probs)
  - results/evaluation/ann04062026_SC{,_seed43,_seed44}/predictions_*_SC.csv

No GPU required.

Usage:
    python scripts/evaluation/sc_two_stage_evaluation.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
from sc_phase1_utils import (
    binary_metrics, display_name, load_predictions,
    POSITIVE_LABEL, NEGATIVE_LABEL, PROJECT_ROOT, DL_DISPLAY_NAMES,
)

_DATA_DIR = os.path.join(PROJECT_ROOT, 'data/ann04062026')
_UTD_PROBS_PATH = os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC/utd_gate_probs_SC.csv')

SEEDS = [42, 43, 44]
RERUN_ARCHS = ['cross_modal', 'gated', 'dualhead_cross_attention', 'dualhead_weighted']
_PRED_DIRS = {
    42: os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC'),
    43: os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC_seed43'),
    44: os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC_seed44'),
}
# Other architectures only run once (seed42 / "Run A").
SEED42_ONLY_ARCHS = ['vetbert_finetune', 'dualhead_concatenation', 'dualhead_multi_task',
                      'attention', 'hierarchical']

_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC')
_OUT_STAGE1_PATH = os.path.join(_OUTPUT_DIR, 'phase_two_stage_stage1_summary.csv')
_OUT_STAGE2_PATH = os.path.join(_OUTPUT_DIR, 'phase_two_stage_stage2_summary.csv')

N_BOOTSTRAP = 2000
BOOT_SEED = 42
DROP_K_FOR_GATES = [0, 1]


def build_is_utd(path):
    # Source CSVs are latin-1 encoded (see prepare_ann04062026_SC_data.py);
    # the binary-task CSVs (and everything derived from them, incl.
    # utd_gate_probs_SC.csv / predictions_*_SC.csv) were built from a
    # latin-1 read then re-saved as utf-8, so 'text' must be read the same
    # way here for the merge keys to match exactly.
    df = pd.read_csv(path, encoding='latin-1')
    is_sc = df['utd_spor_cyst'] == 1
    is_other_utd = df['utd_other'].notna()
    is_utd = is_sc | is_other_utd
    return df[['text']].assign(is_sc=is_sc.values, is_utd=is_utd.values)


def bootstrap_ci_metric(pos_idx, neg_idx, score_fn, rng):
    vals = []
    for _ in range(N_BOOTSTRAP):
        sampled_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sampled_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([sampled_pos, sampled_neg])
        vals.append(score_fn(idx))
    vals = np.array(vals)
    return np.quantile(vals, 0.025), np.quantile(vals, 0.975)


def evaluate_stage1():
    print('=' * 78)
    print('  Stage 1: UTD gate (p_urinary_tract_disorder) vs. is_utd ground truth')
    print('=' * 78)

    test_truth = build_is_utd(os.path.join(_DATA_DIR, 'test-all-syndromes.csv'))
    dev_truth = build_is_utd(os.path.join(_DATA_DIR, 'dev-all-syndromes.csv'))

    utd_df = pd.read_csv(_UTD_PROBS_PATH)
    dev_utd = utd_df[utd_df['split'] == 'dev'].reset_index(drop=True)
    test_utd = utd_df[utd_df['split'] == 'test'].reset_index(drop=True)

    test_merged = test_utd.merge(test_truth, on='text', how='left')
    if test_merged['is_utd'].isna().any():
        raise ValueError("Some test records in utd_gate_probs_SC.csv not found in test-all-syndromes.csv")

    dev_merged = dev_utd.merge(dev_truth, on='text', how='left')
    if dev_merged['is_utd'].isna().any():
        raise ValueError("Some dev records in utd_gate_probs_SC.csv not found in dev-all-syndromes.csv")

    y_true = test_merged['is_utd'].values.astype(int)
    scores = test_merged['p_urinary_tract_disorder'].values
    n_pos, n_neg = int(y_true.sum()), int((1 - y_true).sum())
    print(f"\ntest: n={len(y_true)} ({n_pos} is_utd-positive / {n_neg} negative)")

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    rng = np.random.default_rng(BOOT_SEED)

    auc_roc = roc_auc_score(y_true, scores)
    auc_pr = average_precision_score(y_true, scores)
    auc_roc_ci = bootstrap_ci_metric(pos_idx, neg_idx, lambda idx: roc_auc_score(y_true[idx], scores[idx]), rng)
    auc_pr_ci = bootstrap_ci_metric(pos_idx, neg_idx, lambda idx: average_precision_score(y_true[idx], scores[idx]), rng)

    print(f"\n  AUC-ROC = {auc_roc:.3f}  95% CI=[{auc_roc_ci[0]:.3f}, {auc_roc_ci[1]:.3f}]")
    print(f"  AUC-PR  = {auc_pr:.3f}  95% CI=[{auc_pr_ci[0]:.3f}, {auc_pr_ci[1]:.3f}]  (baseline @ prevalence={n_pos/len(y_true):.3f})")

    # --- Gate thresholds, derived from dev SC-positives (same convention as
    # sc_cascade_step1_dl_leakfree.py), evaluated here against is_utd truth ---
    dev_is_sc = dev_merged['is_sc'].values.astype(bool)
    sorted_sc_probs_dev = np.sort(dev_merged.loc[dev_is_sc, 'p_urinary_tract_disorder'].values)

    rows = []
    print("\n  Gate thresholds selected from dev SC-positives, applied to test is_utd ground truth:")
    for k in DROP_K_FOR_GATES:
        if k >= len(sorted_sc_probs_dev):
            continue
        t = sorted_sc_probs_dev[k]
        y_pred = (scores >= t).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(f"    gate_drop{k} (t={t:.6f}): TP={tp:2d} FN={fn:2d} FP={fp:3d} TN={tn:3d}  "
              f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}")
        rows.append({
            'gate': f'gate_drop{k}', 'threshold': t,
            'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
            'precision_is_utd': precision, 'recall_is_utd': recall, 'f1_is_utd': f1,
            'auc_roc': auc_roc, 'auc_roc_ci_lower': auc_roc_ci[0], 'auc_roc_ci_upper': auc_roc_ci[1],
            'auc_pr': auc_pr, 'auc_pr_ci_lower': auc_pr_ci[0], 'auc_pr_ci_upper': auc_pr_ci[1],
            'n_pos': n_pos, 'n_neg': n_neg,
        })

    out_df = pd.DataFrame(rows)
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    out_df.to_csv(_OUT_STAGE1_PATH, index=False)
    print(f"\nSaved -> {_OUT_STAGE1_PATH}")

    return test_truth


def f1_sc_at_idx(y_true, y_pred, idx):
    return binary_metrics(y_true[idx], y_pred[idx])['f1_sc']


def evaluate_one(merged, label_arch, label_method, rng):
    """Given a merged df with columns Label, predicted_label, is_utd, compute
    full-test-set and stage-2-subset (is_utd only) F1_sc + bootstrap CIs."""
    y_true_full = merged['Label'].values
    y_pred_full = merged['predicted_label'].values

    pos_idx_full = np.where(y_true_full == POSITIVE_LABEL)[0]
    neg_idx_full = np.where(y_true_full == NEGATIVE_LABEL)[0]
    m_full = binary_metrics(y_true_full, y_pred_full)
    ci_full = bootstrap_ci_metric(pos_idx_full, neg_idx_full,
                                   lambda idx: f1_sc_at_idx(y_true_full, y_pred_full, idx), rng)

    sub = merged[merged['is_utd']].reset_index(drop=True)
    y_true_sub = sub['Label'].values
    y_pred_sub = sub['predicted_label'].values
    pos_idx_sub = np.where(y_true_sub == POSITIVE_LABEL)[0]
    neg_idx_sub = np.where(y_true_sub == NEGATIVE_LABEL)[0]
    m_sub = binary_metrics(y_true_sub, y_pred_sub)
    ci_sub = bootstrap_ci_metric(pos_idx_sub, neg_idx_sub,
                                  lambda idx: f1_sc_at_idx(y_true_sub, y_pred_sub, idx), rng)

    tp_sub = int(((y_true_sub == POSITIVE_LABEL) & (y_pred_sub == POSITIVE_LABEL)).sum())
    fn_sub = int(((y_true_sub == POSITIVE_LABEL) & (y_pred_sub == NEGATIVE_LABEL)).sum())
    fp_sub = int(((y_true_sub == NEGATIVE_LABEL) & (y_pred_sub == POSITIVE_LABEL)).sum())
    tn_sub = int(((y_true_sub == NEGATIVE_LABEL) & (y_pred_sub == NEGATIVE_LABEL)).sum())

    return {
        'architecture': label_arch, 'display_name': display_name(label_arch),
        'method': label_method,
        'full_n_pos': len(pos_idx_full), 'full_n_neg': len(neg_idx_full),
        'full_f1_sc': m_full['f1_sc'], 'full_precision_sc': m_full['precision_sc'],
        'full_recall_sc': m_full['recall_sc'],
        'full_f1_ci_lower': ci_full[0], 'full_f1_ci_upper': ci_full[1],
        'stage2_n_pos': len(pos_idx_sub), 'stage2_n_neg': len(neg_idx_sub),
        'stage2_tp': tp_sub, 'stage2_fn': fn_sub, 'stage2_fp': fp_sub, 'stage2_tn': tn_sub,
        'stage2_f1_sc': m_sub['f1_sc'], 'stage2_precision_sc': m_sub['precision_sc'],
        'stage2_recall_sc': m_sub['recall_sc'],
        'stage2_f1_ci_lower': ci_sub[0], 'stage2_f1_ci_upper': ci_sub[1],
    }


def evaluate_stage2(test_truth):
    print('\n' + '=' * 78)
    print('  Stage 2: SC vs. other-UTD, restricted to the is_utd subset (n=34)')
    print('=' * 78)
    print(f"\n  is_utd subset: {test_truth['is_utd'].sum()} total "
          f"({test_truth['is_sc'].sum()} SC-positive / "
          f"{(test_truth['is_utd'] & ~test_truth['is_sc']).sum()} other-UTD)")

    rng = np.random.default_rng(BOOT_SEED)
    rows = []

    # --- 4 re-run architectures x 3 seeds, plus soft-vote ensemble ---
    for arch in RERUN_ARCHS:
        print(f"\n  -- {display_name(arch)} --")
        seed_dfs = {}
        for seed in SEEDS:
            path = os.path.join(_PRED_DIRS[seed], f'predictions_{arch}_SC.csv')
            df = load_predictions(path)
            merged = df.merge(test_truth[['text', 'is_utd']], on='text', how='left')
            if merged['is_utd'].isna().any():
                raise ValueError(f"{path}: some test records not found in test-all-syndromes.csv")
            seed_dfs[seed] = merged

            row = evaluate_one(merged, arch, f'seed{seed}', rng)
            rows.append(row)
            print(f"    seed{seed:<3} full F1={row['full_f1_sc']:.3f} "
                  f"95%CI=[{row['full_f1_ci_lower']:.3f},{row['full_f1_ci_upper']:.3f}]   "
                  f"stage2 F1={row['stage2_f1_sc']:.3f} "
                  f"95%CI=[{row['stage2_f1_ci_lower']:.3f},{row['stage2_f1_ci_upper']:.3f}] "
                  f"(TP={row['stage2_tp']},FN={row['stage2_fn']},FP={row['stage2_fp']},TN={row['stage2_tn']})")

        # Soft-vote ensemble across the 3 seeds.
        base = seed_dfs[42][['text', 'Label', 'is_utd']].reset_index(drop=True)
        avg_prob = np.mean([seed_dfs[s].set_index('text').loc[base['text'], 'prob_sporadic_cystitis'].values
                             for s in SEEDS], axis=0)
        ens = base.copy()
        ens['predicted_label'] = np.where(avg_prob >= 0.5, POSITIVE_LABEL, NEGATIVE_LABEL)
        row = evaluate_one(ens, arch, 'soft_vote_avg_prob', rng)
        rows.append(row)
        print(f"    {'soft-vote':<7} full F1={row['full_f1_sc']:.3f} "
              f"95%CI=[{row['full_f1_ci_lower']:.3f},{row['full_f1_ci_upper']:.3f}]   "
              f"stage2 F1={row['stage2_f1_sc']:.3f} "
              f"95%CI=[{row['stage2_f1_ci_lower']:.3f},{row['stage2_f1_ci_upper']:.3f}] "
              f"(TP={row['stage2_tp']},FN={row['stage2_fn']},FP={row['stage2_fp']},TN={row['stage2_tn']})")

    # --- Remaining seed42-only architectures, for context ---
    print(f"\n  -- Remaining seed42-only architectures --")
    for arch in SEED42_ONLY_ARCHS:
        path = os.path.join(_PRED_DIRS[42], f'predictions_{arch}_SC.csv')
        if not os.path.exists(path):
            print(f"    Skipping {arch}: {path} not found")
            continue
        df = load_predictions(path)
        merged = df.merge(test_truth[['text', 'is_utd']], on='text', how='left')
        if merged['is_utd'].isna().any():
            raise ValueError(f"{path}: some test records not found in test-all-syndromes.csv")
        row = evaluate_one(merged, arch, 'seed42', rng)
        rows.append(row)
        print(f"    {display_name(arch):30s} full F1={row['full_f1_sc']:.3f} "
              f"95%CI=[{row['full_f1_ci_lower']:.3f},{row['full_f1_ci_upper']:.3f}]   "
              f"stage2 F1={row['stage2_f1_sc']:.3f} "
              f"95%CI=[{row['stage2_f1_ci_lower']:.3f},{row['stage2_f1_ci_upper']:.3f}] "
              f"(TP={row['stage2_tp']},FN={row['stage2_fn']},FP={row['stage2_fp']},TN={row['stage2_tn']})")

    out_df = pd.DataFrame(rows)
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    out_df.to_csv(_OUT_STAGE2_PATH, index=False)
    print(f"\nSaved -> {_OUT_STAGE2_PATH}")


def main():
    test_truth = evaluate_stage1()
    evaluate_stage2(test_truth)


if __name__ == '__main__':
    main()
