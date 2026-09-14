#!/usr/bin/env python3
"""
Apply the existing UTD gate (sc_cascade_step1_dl_leakfree.py) to the seed43/44
re-run predictions, which it was never run on (sc_experiment_plan.md follow-up).

Motivation: sc_two_stage_evaluation.py showed that (a) the UTD gate is a
near-perfect, stable classifier for "is this note urinary at all"
(AUC-ROC=0.984, 95% CI=[0.969,0.995], 100% recall for is_utd at gate_drop0/1),
and (b) restricted to the is_utd subset, SC-vs-other-UTD discrimination is
decent and fairly consistent across every architecture/seed (F1 0.58-0.85) --
even for seeds whose full-test F1_sc collapsed (e.g. Cross-Modal seed43:
full F1=0.146, stage2 F1=0.684). That gap means the collapses are driven by
false positives on non-UTD notes, which the gate should suppress.
phase_cascade_dl_leakfree_summary.csv already shows this recovers roughly half
the gap for the worst seed42 (Run A) architectures, but was never applied to
seed43/44. This script does that, for the 4 re-run architectures x 3 seeds
(+ soft-vote ensembles), and reports ungated vs. gate_drop0 vs. gate_drop1
full-test F1_sc + bootstrap CI side by side.

Gate thresholds are the same dev-derived values used everywhere else
(gate_drop0=0.000144, gate_drop1=0.000282 from utd_gate_probs_SC.csv dev split).

No GPU required.

Usage:
    python scripts/evaluation/sc_cascade_gate_all_seeds.py
"""

import os
import sys
import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
from sc_phase1_utils import (
    binary_metrics, display_name, load_predictions,
    POSITIVE_LABEL, NEGATIVE_LABEL, PROJECT_ROOT,
)

SEEDS = [42, 43, 44]
RERUN_ARCHS = ['cross_modal', 'gated', 'dualhead_cross_attention', 'dualhead_weighted']
_PRED_DIRS = {
    42: os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC'),
    43: os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC_seed43'),
    44: os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC_seed44'),
}
_UTD_PROBS_PATH = os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC/utd_gate_probs_SC.csv')
_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC/phase_cascade_gate_all_seeds_summary.csv')

DROP_K_FOR_GATES = [0, 1]
N_BOOTSTRAP = 2000
BOOT_SEED = 42


def bootstrap_ci_f1(y_true, y_pred, rng):
    pos_idx = np.where(y_true == POSITIVE_LABEL)[0]
    neg_idx = np.where(y_true == NEGATIVE_LABEL)[0]
    f1s = []
    for _ in range(N_BOOTSTRAP):
        sampled_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sampled_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([sampled_pos, sampled_neg])
        f1s.append(binary_metrics(y_true[idx], y_pred[idx])['f1_sc'])
    f1s = np.array(f1s)
    return np.quantile(f1s, 0.025), np.quantile(f1s, 0.975)


def report_row(rng, y_true, y_pred, label):
    m = binary_metrics(y_true, y_pred)
    lo, hi = bootstrap_ci_f1(y_true, y_pred, rng)
    return {
        'f1_sc': m['f1_sc'], 'precision_sc': m['precision_sc'], 'recall_sc': m['recall_sc'],
        'ci_lower': lo, 'ci_upper': hi,
    }, f"{label:<14} F1={m['f1_sc']:.3f} P={m['precision_sc']:.3f} R={m['recall_sc']:.3f} 95%CI=[{lo:.3f},{hi:.3f}]"


def main():
    print('=' * 90)
    print('  Applying the UTD gate to seed43/44 (and re-checking seed42) full-test predictions')
    print('=' * 90)

    utd_df = pd.read_csv(_UTD_PROBS_PATH)
    dev_utd = utd_df[utd_df['split'] == 'test'].copy()  # placeholder overwritten below
    dev_utd = utd_df[utd_df['split'] == 'dev'].reset_index(drop=True)
    test_utd = utd_df[utd_df['split'] == 'test'].reset_index(drop=True)

    dev_is_sc = (dev_utd['Label'] == POSITIVE_LABEL).values
    sorted_sc_probs_dev = np.sort(dev_utd.loc[dev_is_sc, 'p_urinary_tract_disorder'].values)
    gate_thresholds = {}
    for k in DROP_K_FOR_GATES:
        gate_thresholds[f'gate_drop{k}'] = sorted_sc_probs_dev[k]
    print(f"\nGate thresholds (from dev SC-positives): "
          + ", ".join(f"{k}={v:.6f}" for k, v in gate_thresholds.items()))

    rng = np.random.default_rng(BOOT_SEED)
    rows = []

    for arch in RERUN_ARCHS:
        print(f"\n{'-'*90}\n  {display_name(arch)}\n{'-'*90}")
        seed_dfs = {}
        for seed in SEEDS:
            path = os.path.join(_PRED_DIRS[seed], f'predictions_{arch}_SC.csv')
            df = load_predictions(path)
            merged = df.merge(test_utd[['text', 'p_urinary_tract_disorder']], on='text', how='left')
            if merged['p_urinary_tract_disorder'].isna().any():
                raise ValueError(f"{path}: some test records not found in {_UTD_PROBS_PATH}")
            seed_dfs[seed] = merged

            y_true = merged['Label'].values
            y_pred_ungated = merged['predicted_label'].values
            row, line = report_row(rng, y_true, y_pred_ungated, f"seed{seed} ungated")
            row.update({'architecture': arch, 'display_name': display_name(arch), 'method': f'seed{seed}_ungated'})
            rows.append(row)
            print(f"  {line}")

            for gate_name, t in gate_thresholds.items():
                pass_mask = merged['p_urinary_tract_disorder'].values >= t
                y_pred_gated = np.where(pass_mask, y_pred_ungated, NEGATIVE_LABEL)
                row, line = report_row(rng, y_true, y_pred_gated, f"seed{seed} {gate_name}")
                row.update({'architecture': arch, 'display_name': display_name(arch), 'method': f'seed{seed}_{gate_name}'})
                rows.append(row)
                print(f"  {line}")
            print()

        # Soft-vote ensemble across the 3 seeds, then gate.
        base = seed_dfs[42][['text', 'Label', 'p_urinary_tract_disorder']].reset_index(drop=True)
        avg_prob = np.mean([seed_dfs[s].set_index('text').loc[base['text'], 'prob_sporadic_cystitis'].values
                             for s in SEEDS], axis=0)
        y_pred_ungated = np.where(avg_prob >= 0.5, POSITIVE_LABEL, NEGATIVE_LABEL)
        y_true = base['Label'].values
        row, line = report_row(rng, y_true, y_pred_ungated, "soft-vote ungated")
        row.update({'architecture': arch, 'display_name': display_name(arch), 'method': 'soft_vote_ungated'})
        rows.append(row)
        print(f"  {line}")

        for gate_name, t in gate_thresholds.items():
            pass_mask = base['p_urinary_tract_disorder'].values >= t
            y_pred_gated = np.where(pass_mask, y_pred_ungated, NEGATIVE_LABEL)
            row, line = report_row(rng, y_true, y_pred_gated, f"soft-vote {gate_name}")
            row.update({'architecture': arch, 'display_name': display_name(arch), 'method': f'soft_vote_{gate_name}'})
            rows.append(row)
            print(f"  {line}")

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(_OUTPUT_PATH), exist_ok=True)
    out_df.to_csv(_OUTPUT_PATH, index=False)
    print(f"\nSaved -> {_OUTPUT_PATH}")


if __name__ == '__main__':
    main()
