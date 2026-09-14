#!/usr/bin/env python3
"""
Cascade Step 1, leak-free variant (sc_experiment_plan.md follow-up).

The in-sample version (sc_cascade_step1_utd_gate.py) selected the gate
threshold on P(urinary tract disorder) using the same 1,000-record pool
(dev_binary_SC.csv + test_binary_SC.csv) it was then evaluated on -- an
optimistic, in-sample threshold choice (same caveat as the Phase 1 threshold
sweep).

This script selects the gate threshold using dev_binary_SC.csv ONLY (500
records, 17 SC-positive), then evaluates the resulting cascade (gate + each
of the 12 frozen ML Step-2 classifiers) on test_binary_SC.csv ONLY (500
records, 21 SC-positive) -- the Run A test set. The "step2_only" row therefore
reproduces the existing Run A test_f1_sporadic_cystitis numbers in
ml_comparison_ann04062026.csv (sanity check).

Two dev-derived gate thresholds:
  - gate_drop0: threshold = min P(urinary tract disorder) among dev's 17
    true-SC records (100% Step-1 recall on dev).
  - gate_drop1: threshold = 2nd-smallest such value (drops the single
    lowest-confidence true-SC record in dev, ~94% Step-1 recall on dev).

Both thresholds are applied, unchanged, to test_binary_SC.csv -- no threshold
information is taken from the test set.

No retraining; GPU used only for the one-off feature extraction (same cost as
Phase 2 / the in-sample cascade script).

Usage:
    python scripts/evaluation/sc_cascade_step1_utd_gate_leakfree.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
from sc_phase1_utils import binary_metrics, POSITIVE_LABEL, NEGATIVE_LABEL, PROJECT_ROOT

sys.path.insert(0, PROJECT_ROOT)
from src.models.vetbert_hybrid_extractor import VetBERTHybridFeatureExtractor
from scripts.training.train_evaluate_ml_ann04062026 import extract_all_features

_DEV_DEFAULT = os.path.join(PROJECT_ROOT, 'data/ann04062026/processed/dev_binary_SC.csv')
_TEST_DEFAULT = os.path.join(PROJECT_ROOT, 'data/ann04062026/processed/test_binary_SC.csv')
_MODELS_DEFAULT = os.path.join(PROJECT_ROOT, 'models/ml_ann04062026_SC')
_ML_COMPARISON_DEFAULT = os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC/ml_comparison_ann04062026.csv')
_OUTPUT_DIR_DEFAULT = os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC')

SYNDROME_OF_INTEREST = 'urinary tract disorder'
FEATURE_CONFIGS = ['embeddings_only', 'syndromes_only', 'hybrid_equal', 'hybrid_2x_syn']
CLASSIFIERS = ['random_forest', 'logistic', 'svm']
DROP_K_FOR_GATES = [0, 1]  # thresholds chosen on dev: retain (n_sc_dev - k) of dev's true-SC records


def main():
    print('=' * 70)
    print('  Cascade Step 1 (leak-free) -- gate threshold from dev, evaluated on test')
    print('=' * 70)

    dev_df = pd.read_csv(_DEV_DEFAULT)
    test_df = pd.read_csv(_TEST_DEFAULT)
    n_dev = len(dev_df)
    combined_df = pd.concat([dev_df, test_df], ignore_index=True)

    dev_labels = dev_df['Label'].values
    test_labels = test_df['Label'].values
    dev_is_sc = dev_labels == POSITIVE_LABEL
    test_is_sc = test_labels == POSITIVE_LABEL

    print(f"\ndev_binary_SC.csv : {len(dev_df)} records ({dev_is_sc.sum()} SC-positive, {dev_is_sc.mean() * 100:.1f}%)")
    print(f"test_binary_SC.csv: {len(test_df)} records ({test_is_sc.sum()} SC-positive, {test_is_sc.mean() * 100:.1f}%)")

    print('\nLoading VetBERT feature extractor (havocy28/VetBERTDx)...')
    extractor = VetBERTHybridFeatureExtractor()

    print('\nExtracting features (dev + test)...')
    texts = combined_df['text'].astype(str).tolist()
    feats = extract_all_features(extractor, texts)

    utd_idx = extractor.syndrome_labels.index(SYNDROME_OF_INTEREST)
    utd_probs = feats['syndromes_only'][:, utd_idx]
    utd_probs_dev = utd_probs[:n_dev]
    utd_probs_test = utd_probs[n_dev:]

    print(f"\nP({SYNDROME_OF_INTEREST}):")
    print(f"  dev  SC-pos : min={utd_probs_dev[dev_is_sc].min():.4f} mean={utd_probs_dev[dev_is_sc].mean():.4f}")
    print(f"  dev  Other  : mean={utd_probs_dev[~dev_is_sc].mean():.4f}")
    print(f"  test SC-pos : min={utd_probs_test[test_is_sc].min():.4f} mean={utd_probs_test[test_is_sc].mean():.4f}")
    print(f"  test Other  : mean={utd_probs_test[~test_is_sc].mean():.4f}")

    # ---- Select gate thresholds from dev only ----
    sorted_sc_probs_dev = np.sort(utd_probs_dev[dev_is_sc])  # ascending
    gate_thresholds = {}
    print('\nGate thresholds selected from dev_binary_SC.csv only:')
    for k in DROP_K_FOR_GATES:
        if k >= len(sorted_sc_probs_dev):
            continue
        t = sorted_sc_probs_dev[k]
        gate_name = f'gate_drop{k}'
        gate_thresholds[gate_name] = t

        dev_pass = utd_probs_dev >= t
        dev_recall = (dev_pass & dev_is_sc).sum() / dev_is_sc.sum()
        test_pass = utd_probs_test >= t
        test_recall = (test_pass & test_is_sc).sum() / test_is_sc.sum()
        test_prevalence_pass = (test_pass & test_is_sc).sum() / test_pass.sum()

        print(f"\n  {gate_name}: threshold t={t:.6f} (drop {k} lowest-confidence true-SC record(s) in dev)")
        print(f"    dev  recall = {(dev_pass & dev_is_sc).sum()}/{dev_is_sc.sum()} = {dev_recall * 100:.1f}%  "
              f"({dev_pass.sum()}/{len(dev_df)} pass)")
        print(f"    test recall = {(test_pass & test_is_sc).sum()}/{test_is_sc.sum()} = {test_recall * 100:.1f}%  "
              f"({test_pass.sum()}/{len(test_df)} pass, prevalence in pass-set = {test_prevalence_pass * 100:.1f}%)")

    # ---- Step2 (12 ML configs) on test only, gated by dev-derived thresholds ----
    print('\n' + '=' * 70)
    print('  Step2 (ML configs) on test_binary_SC.csv, gated by dev-derived thresholds')
    print('=' * 70)

    ml_comparison = None
    if os.path.exists(_ML_COMPARISON_DEFAULT):
        ml_comparison = pd.read_csv(_ML_COMPARISON_DEFAULT)

    rows = []
    for feat_name in FEATURE_CONFIGS:
        X_test = feats[feat_name][n_dev:]
        for clf_name in CLASSIFIERS:
            bundle_key = f"{feat_name}_{clf_name}"
            bundle_dir = os.path.join(_MODELS_DEFAULT, bundle_key)
            if not os.path.isdir(bundle_dir):
                continue

            clf = joblib.load(os.path.join(bundle_dir, 'classifier.joblib'))
            scaler = joblib.load(os.path.join(bundle_dir, 'scaler.joblib'))
            le = joblib.load(os.path.join(bundle_dir, 'label_encoder.joblib'))

            X_scaled = scaler.transform(X_test)
            y_pred_step2 = le.inverse_transform(clf.predict(X_scaled))
            step2_only = binary_metrics(test_labels, y_pred_step2)

            row = {
                'feature_config': feat_name,
                'classifier': clf_name,
                'step2_only_f1_sc': step2_only['f1_sc'],
                'step2_only_precision_sc': step2_only['precision_sc'],
                'step2_only_recall_sc': step2_only['recall_sc'],
            }

            for gate_name, t in gate_thresholds.items():
                pass_mask = utd_probs_test >= t
                y_pred_cascade = np.where(pass_mask, y_pred_step2, NEGATIVE_LABEL)
                m = binary_metrics(test_labels, y_pred_cascade)
                row[f'{gate_name}_f1_sc'] = m['f1_sc']
                row[f'{gate_name}_precision_sc'] = m['precision_sc']
                row[f'{gate_name}_recall_sc'] = m['recall_sc']

            if ml_comparison is not None:
                match = ml_comparison[(ml_comparison['feature_config'] == feat_name) &
                                       (ml_comparison['classifier'] == clf_name)]
                if len(match):
                    row['run_a_test_f1_sc'] = match.iloc[0]['test_f1_sporadic_cystitis']

            rows.append(row)
            print(f"\n  {bundle_key:30s} step2_only F1={step2_only['f1_sc']:.3f}  " +
                  "  ".join(f"{g}_F1={row[f'{g}_f1_sc']:.3f}" for g in gate_thresholds))

    summary_df = pd.DataFrame(rows).sort_values('step2_only_f1_sc', ascending=False)
    os.makedirs(_OUTPUT_DIR_DEFAULT, exist_ok=True)
    out_path = os.path.join(_OUTPUT_DIR_DEFAULT, 'phase_cascade_leakfree_summary.csv')
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == '__main__':
    main()
