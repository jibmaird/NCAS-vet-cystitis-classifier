#!/usr/bin/env python3
"""
Per-record predictions for the traditional-ML SC baselines.

The 9 deep-learning architectures each have per-record prediction CSVs saved
under results/evaluation/ann04062026_SC/predictions_*_SC.csv (columns: text,
Label, predicted_label, prob_sporadic_cystitis, prob_other, correct), but the
12 traditional-ML configs (4 feature sets x {LR, RF, SVM}, trained by
train_evaluate_ml_ann04062026.py) only have aggregate metrics saved in
ml_comparison_ann04062026.csv -- no per-record predictions.

This script loads the saved model bundles from models/ml_ann04062026_SC/ and
reproduces per-record predictions in the same format as the DL models, so
that Phase 1 statistical scripts (bootstrap CI, McNemar's test) can include
the ML baselines alongside the DL models.

Usage:
    python scripts/prediction/predict_ml_ann04062026_SC.py
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, '../..'))
sys.path.insert(0, _project_root)

from src.models.vetbert_hybrid_extractor import VetBERTHybridFeatureExtractor
from scripts.training.train_evaluate_ml_ann04062026 import extract_all_features

_TEST_DEFAULT = os.path.join(_project_root, 'data/ann04062026/processed/test_binary_SC.csv')
_MODELS_DEFAULT = os.path.join(_project_root, 'models/ml_ann04062026_SC')
_OUTPUT_DEFAULT = os.path.join(_project_root, 'results/predictions/ann04062026_SC')

POSITIVE_LABEL = 'Sporadic Cystitis'
NEGATIVE_LABEL = 'Other'


def main():
    parser = argparse.ArgumentParser(
        description='Generate per-record predictions for the traditional-ML SC baselines'
    )
    parser.add_argument('--test_file', default=_TEST_DEFAULT)
    parser.add_argument('--models_dir', default=_MODELS_DEFAULT)
    parser.add_argument('--output_dir', default=_OUTPUT_DEFAULT)
    parser.add_argument('--feature_configs', nargs='+',
                         default=['embeddings_only', 'syndromes_only', 'hybrid_equal', 'hybrid_2x_syn'],
                         choices=['embeddings_only', 'syndromes_only', 'hybrid_equal', 'hybrid_2x_syn'])
    parser.add_argument('--classifiers', nargs='+',
                         default=['random_forest', 'logistic', 'svm'],
                         choices=['random_forest', 'logistic', 'svm'])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 70)
    print('  Per-record predictions for traditional-ML SC baselines')
    print('=' * 70)

    test_df = pd.read_csv(args.test_file)
    texts = test_df['text'].astype(str).tolist()
    labels = test_df['Label'].values
    print(f"\nTest file : {args.test_file}")
    print(f"  {len(test_df)} samples")

    print('\nLoading VetBERT feature extractor (havocy28/VetBERTDx)...')
    extractor = VetBERTHybridFeatureExtractor()

    print('\nExtracting test features...')
    test_feats = extract_all_features(extractor, texts)

    for feat_name in args.feature_configs:
        X_test = test_feats[feat_name]

        for clf_name in args.classifiers:
            bundle_key = f"{feat_name}_{clf_name}"
            bundle_dir = os.path.join(args.models_dir, bundle_key)
            if not os.path.isdir(bundle_dir):
                print(f"\n  Skipping {bundle_key} -- no saved model at {bundle_dir}")
                continue

            print(f"\n  {bundle_key}")
            clf = joblib.load(os.path.join(bundle_dir, 'classifier.joblib'))
            scaler = joblib.load(os.path.join(bundle_dir, 'scaler.joblib'))
            le = joblib.load(os.path.join(bundle_dir, 'label_encoder.joblib'))

            X_scaled = scaler.transform(X_test)
            y_pred = clf.predict(X_scaled)
            probs = clf.predict_proba(X_scaled)

            classes = list(le.classes_)
            sc_idx = classes.index(POSITIVE_LABEL)
            other_idx = classes.index(NEGATIVE_LABEL)

            predicted_label = le.inverse_transform(y_pred)

            out = pd.DataFrame({
                'text': texts,
                'Label': labels,
                'predicted_label': predicted_label,
                'prob_sporadic_cystitis': probs[:, sc_idx],
                'prob_other': probs[:, other_idx],
            })
            out['correct'] = out['Label'] == out['predicted_label']

            out_path = os.path.join(args.output_dir, f'predictions_ml_{bundle_key}_SC.csv')
            out.to_csv(out_path, index=False)

            acc = out['correct'].mean()
            n_pos_pred = (out['predicted_label'] == POSITIVE_LABEL).sum()
            print(f"    accuracy={acc:.4f}  predicted-positive={n_pos_pred}  -> {out_path}")

    print(f"\nDone. Predictions saved -> {args.output_dir}")


if __name__ == '__main__':
    main()
