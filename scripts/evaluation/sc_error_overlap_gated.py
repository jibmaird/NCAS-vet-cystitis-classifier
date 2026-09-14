#!/usr/bin/env python3
"""
Error-overlap analysis on GATED predictions (sc_experiment_plan.md follow-up,
project_sc_error_review_followup.md).

sc_cascade_gate_all_seeds.py showed gate_drop1 closes most of the seed-to-seed
F1 gap for the 4 re-run architectures (Cross-Modal, Gated, Cross-Attention,
Weighted) across seeds 42/43/44, without retraining. This script re-runs the
"is the seed instability systematic or noise" question on those GATED
predictions (the un-gated version was confounded by non-UTD false positives,
which the gate removes).

For each architecture, for each of the 21 true-SC test records, records
whether gate_drop1 predicted TP or FN in each of seeds 42/43/44 -- and
classifies each record as 'stable_TP' (caught every time), 'stable_FN'
(missed every time), or 'flip' (seed-dependent).

Also builds a cross-architecture x cross-seed (12 combos) summary: for each
of the 21 true-SC records and for the gated FPs, how many of the 12
arch x seed combos get it right/wrong. This is the most decision-relevant
view for "which specific cases are genuinely hard vs. model-dependent."

No retraining; CPU only.

Usage:
    python scripts/evaluation/sc_error_overlap_gated.py
"""

import os
import sys
import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
from sc_phase1_utils import (
    display_name, load_predictions, POSITIVE_LABEL, NEGATIVE_LABEL, PROJECT_ROOT,
)

SEEDS = [42, 43, 44]
RERUN_ARCHS = ['cross_modal', 'gated', 'dualhead_cross_attention', 'dualhead_weighted']
_PRED_DIRS = {
    42: os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC'),
    43: os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC_seed43'),
    44: os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC_seed44'),
}
_UTD_PROBS_PATH = os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC/utd_gate_probs_SC.csv')
_TEST_SOURCE = os.path.join(PROJECT_ROOT, 'data/ann04062026/test-all-syndromes.csv')
_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC')


def outcome(label, pred):
    if label == POSITIVE_LABEL and pred == POSITIVE_LABEL:
        return 'TP'
    if label == POSITIVE_LABEL and pred == NEGATIVE_LABEL:
        return 'FN'
    if label == NEGATIVE_LABEL and pred == POSITIVE_LABEL:
        return 'FP'
    return 'TN'


def main():
    utd_df = pd.read_csv(_UTD_PROBS_PATH)
    dev_utd = utd_df[utd_df['split'] == 'dev'].reset_index(drop=True)
    test_utd = utd_df[utd_df['split'] == 'test'].reset_index(drop=True)

    dev_is_sc = (dev_utd['Label'] == POSITIVE_LABEL).values
    sorted_sc_probs_dev = np.sort(dev_utd.loc[dev_is_sc, 'p_urinary_tract_disorder'].values)
    gate_t = sorted_sc_probs_dev[1]  # gate_drop1
    print(f"gate_drop1 threshold = {gate_t:.6f}\n")

    meta = pd.read_csv(_TEST_SOURCE, encoding='latin-1')
    meta_cols = meta[['text', 'consult_table_id', 'utd_other']]

    # combo_outcomes[text] = {'Label': ..., 'consult_table_id': ..., 'utd_other': ...,
    #                          ('cross_modal', 42): 'TP', ...}
    combo_outcomes = {}
    base_info = {}

    print('=' * 90)
    print('  Within-architecture stability across seeds 42/43/44 (gate_drop1)')
    print('=' * 90)

    for arch in RERUN_ARCHS:
        seed_outcomes = {}  # text -> {seed: outcome}
        for seed in SEEDS:
            path = os.path.join(_PRED_DIRS[seed], f'predictions_{arch}_SC.csv')
            df = load_predictions(path)
            merged = df.merge(test_utd[['text', 'p_urinary_tract_disorder']], on='text', how='left')
            if merged['p_urinary_tract_disorder'].isna().any():
                raise ValueError(f"{path}: some test records not found in {_UTD_PROBS_PATH}")
            pass_mask = merged['p_urinary_tract_disorder'].values >= gate_t
            y_pred_gated = np.where(pass_mask, merged['predicted_label'].values, NEGATIVE_LABEL)

            for text, label, pred in zip(merged['text'], merged['Label'], y_pred_gated):
                oc = outcome(label, pred)
                seed_outcomes.setdefault(text, {})[seed] = oc
                base_info.setdefault(text, {'Label': label})
                combo_outcomes.setdefault(text, {})[(arch, seed)] = oc

        # Classify the 21 true-SC records.
        sc_rows = []
        for text, per_seed in seed_outcomes.items():
            if base_info[text]['Label'] != POSITIVE_LABEL:
                continue
            ocs = {seed: per_seed.get(seed, '?') for seed in SEEDS}
            tp_count = sum(1 for v in ocs.values() if v == 'TP')
            if tp_count == 3:
                cls = 'stable_TP'
            elif tp_count == 0:
                cls = 'stable_FN'
            else:
                cls = 'flip'
            sc_rows.append({'text': text, **{f'seed{s}': ocs[s] for s in SEEDS}, 'class': cls})

        n_stable_tp = sum(1 for r in sc_rows if r['class'] == 'stable_TP')
        n_stable_fn = sum(1 for r in sc_rows if r['class'] == 'stable_FN')
        n_flip = sum(1 for r in sc_rows if r['class'] == 'flip')

        # FPs: gated false positives, by how many seeds flag them.
        fp_counts = {}
        for text, per_seed in seed_outcomes.items():
            if base_info[text]['Label'] != NEGATIVE_LABEL:
                continue
            n_fp_seeds = sum(1 for v in per_seed.values() if v == 'FP')
            if n_fp_seeds > 0:
                fp_counts[text] = n_fp_seeds
        fp_in_1 = sum(1 for v in fp_counts.values() if v == 1)
        fp_in_2 = sum(1 for v in fp_counts.values() if v == 2)
        fp_in_3 = sum(1 for v in fp_counts.values() if v == 3)

        print(f"\n{display_name(arch)} (21 true-SC test records, gate_drop1):")
        print(f"  stable_TP (caught all 3 seeds) = {n_stable_tp}")
        print(f"  stable_FN (missed all 3 seeds) = {n_stable_fn}")
        print(f"  flip (seed-dependent)          = {n_flip}")
        print(f"  Gated FPs: {len(fp_counts)} unique 'Other' notes flagged in >=1 seed "
              f"(in 1 seed: {fp_in_1}, in 2: {fp_in_2}, in all 3: {fp_in_3})")
        if n_flip:
            print("  Flip cases:")
            for r in sc_rows:
                if r['class'] == 'flip':
                    print(f"    seed42={r['seed42']:<3} seed43={r['seed43']:<3} seed44={r['seed44']:<3}"
                          f"  | {r['text'][:80]}...")

    # ------------------------------------------------------------------
    # Cross-architecture x cross-seed (12 combos) view.
    # ------------------------------------------------------------------
    print('\n' + '=' * 90)
    print('  Cross-architecture x cross-seed (12 combos) view, gate_drop1')
    print('=' * 90)

    n_combos = len(RERUN_ARCHS) * len(SEEDS)

    sc_summary = []
    for text, info in base_info.items():
        if info['Label'] != POSITIVE_LABEL:
            continue
        combos = combo_outcomes.get(text, {})
        n_tp = sum(1 for v in combos.values() if v == 'TP')
        n_fn = sum(1 for v in combos.values() if v == 'FN')
        sc_summary.append({'text': text, 'n_tp_of_12': n_tp, 'n_fn_of_12': n_fn})

    sc_summary_df = pd.DataFrame(sc_summary).merge(meta_cols, on='text', how='left')
    sc_summary_df['consult_table_id'] = sc_summary_df['consult_table_id'].astype(int)
    sc_summary_df = sc_summary_df.sort_values('n_tp_of_12')

    n_never_missed = (sc_summary_df['n_tp_of_12'] == n_combos).sum()
    n_always_missed = (sc_summary_df['n_tp_of_12'] == 0).sum()
    n_mixed = len(sc_summary_df) - n_never_missed - n_always_missed
    print(f"\nTrue-SC records (n={len(sc_summary_df)}), out of {n_combos} arch x seed combos (gated):")
    print(f"  caught by ALL {n_combos} combos     = {n_never_missed}")
    print(f"  missed by ALL {n_combos} combos     = {n_always_missed}")
    print(f"  model-dependent (mixed)     = {n_mixed}")
    print("\n  Hardest true-SC records (lowest TP-rate across 12 combos):")
    for _, row in sc_summary_df.head(8).iterrows():
        print(f"    consult_table_id={row['consult_table_id']:<9} utd_other={str(row['utd_other']):<18} "
              f"caught {row['n_tp_of_12']:2d}/{n_combos}  | {row['text'][:80]}...")

    # FPs across all 12 combos.
    fp_summary = []
    for text, info in base_info.items():
        if info['Label'] != NEGATIVE_LABEL:
            continue
        combos = combo_outcomes.get(text, {})
        n_fp = sum(1 for v in combos.values() if v == 'FP')
        if n_fp > 0:
            fp_summary.append({'text': text, 'n_fp_of_12': n_fp})

    fp_summary_df = pd.DataFrame(fp_summary).merge(meta_cols, on='text', how='left')
    fp_summary_df['consult_table_id'] = fp_summary_df['consult_table_id'].astype(int)
    fp_summary_df = fp_summary_df.sort_values('n_fp_of_12', ascending=False)
    print(f"\n'Other' records ever gate_drop1-FP in >=1 of {n_combos} combos: {len(fp_summary_df)}")
    print("\n  Most-frequently-false-alarmed 'Other' records:")
    for _, row in fp_summary_df.head(8).iterrows():
        print(f"    consult_table_id={row['consult_table_id']:<9} utd_other={str(row['utd_other']):<18} "
              f"FP in {row['n_fp_of_12']:2d}/{n_combos}  | {row['text'][:80]}...")

    out_sc = os.path.join(_OUTPUT_DIR, 'phase_error_overlap_gated_sc_summary.csv')
    out_fp = os.path.join(_OUTPUT_DIR, 'phase_error_overlap_gated_fp_summary.csv')
    sc_summary_df.to_csv(out_sc, index=False)
    fp_summary_df.to_csv(out_fp, index=False)
    print(f"\nSaved -> {out_sc}")
    print(f"Saved -> {out_fp}")


if __name__ == '__main__':
    main()
