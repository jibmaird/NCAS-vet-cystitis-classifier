#!/usr/bin/env python3
"""
Traditional ML baselines on ann04062026 data using frozen VetBERT features.

Trains and evaluates three classifiers (Random Forest, Logistic Regression, SVM)
under four feature configurations:
  - embeddings_only   : 768-dim CLS token from VetBERT base
  - syndromes_only    : ~50-dim syndrome probability vector
  - hybrid_equal      : concat(embeddings, syndromes) equal weight
  - hybrid_2x_syn     : concat(embeddings, 2x syndromes)

Uses explicit train/dev/test splits from ann04062026/processed/.
Saves all trained models and writes a single comparison CSV.

Usage:
    python scripts/training/train_evaluate_ml_ann04062026.py
    python scripts/training/train_evaluate_ml_ann04062026.py \
        --train_file data/ann04062026/processed/train_binary.csv \
        --dev_file   data/ann04062026/processed/dev_binary.csv \
        --test_file  data/ann04062026/processed/test_binary.csv \
        --output_dir results/evaluation/ann04062026
"""

import os
import re
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Resolve project root so imports work regardless of cwd
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, '../..'))
sys.path.insert(0, _project_root)

from src.models.vetbert_hybrid_extractor import VetBERTHybridFeatureExtractor

_TRAIN_DEFAULT = os.path.join(_project_root, 'data/ann04062026/processed/train_binary.csv')
_DEV_DEFAULT   = os.path.join(_project_root, 'data/ann04062026/processed/dev_binary.csv')
_TEST_DEFAULT  = os.path.join(_project_root, 'data/ann04062026/processed/test_binary.csv')
_OUTPUT_DEFAULT = os.path.join(_project_root, 'results/evaluation/ann04062026')
_MODELS_DEFAULT = os.path.join(_project_root, 'models/ml_ann04062026')


def build_classifier(name: str):
    if name == 'random_forest':
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    if name == 'logistic':
        return LogisticRegression(max_iter=2000, random_state=42)
    if name == 'svm':
        return SVC(probability=True, random_state=42)
    raise ValueError(f"Unknown classifier: {name}")


def extract_all_features(extractor: VetBERTHybridFeatureExtractor, texts: list) -> dict:
    """Return dict of feature matrix variants for a list of texts."""
    emb = extractor.extract_embeddings(texts)
    syn = extractor.extract_syndrome_probabilities(texts)
    return {
        'embeddings_only': emb,
        'syndromes_only': syn,
        'hybrid_equal': np.concatenate([emb, syn], axis=1),
        'hybrid_2x_syn': np.concatenate([emb, syn * 2.0], axis=1),
    }


def _slugify_label(label: str) -> str:
    """Turn a class label into a CSV-column-safe slug, e.g. 'UTD Other' -> 'utd_other'."""
    return re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')


def evaluate_split(clf, scaler, X, y_true_str: np.ndarray, le: LabelEncoder) -> dict:
    X_scaled = scaler.transform(X)
    y_true = le.transform(y_true_str)
    y_pred = clf.predict(X_scaled)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    class_prec, class_rec, class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=np.arange(len(le.classes_))
    )
    result = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_precision': macro_prec,
        'macro_recall': macro_rec,
        'macro_f1': macro_f1,
        'report': classification_report(y_true_str, le.inverse_transform(y_pred)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }
    for idx, cls in enumerate(le.classes_):
        slug = _slugify_label(cls)
        result[f'precision_{slug}'] = class_prec[idx]
        result[f'recall_{slug}']    = class_rec[idx]
        result[f'f1_{slug}']        = class_f1[idx]
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Train and evaluate traditional ML baselines on ann04062026 data'
    )
    parser.add_argument('--train_file', default=_TRAIN_DEFAULT)
    parser.add_argument('--dev_file',   default=_DEV_DEFAULT)
    parser.add_argument('--test_file',  default=_TEST_DEFAULT)
    parser.add_argument('--output_dir', default=_OUTPUT_DEFAULT)
    parser.add_argument('--models_dir', default=_MODELS_DEFAULT)
    parser.add_argument('--classifiers', nargs='+',
                        default=['random_forest', 'logistic', 'svm'],
                        choices=['random_forest', 'logistic', 'svm'])
    parser.add_argument('--feature_configs', nargs='+',
                        default=['embeddings_only', 'syndromes_only', 'hybrid_equal', 'hybrid_2x_syn'],
                        choices=['embeddings_only', 'syndromes_only', 'hybrid_equal', 'hybrid_2x_syn'])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    print('='*70)
    print('  VetBERT Traditional ML Baselines — ann04062026')
    print('='*70)

    # --- Load data ---
    train_df = pd.read_csv(args.train_file)
    dev_df   = pd.read_csv(args.dev_file)
    test_df  = pd.read_csv(args.test_file)

    print(f"\nData loaded:")
    print(f"  Train : {len(train_df)} samples")
    print(f"  Dev   : {len(dev_df)} samples")
    print(f"  Test  : {len(test_df)} samples")

    train_texts = train_df['text'].astype(str).tolist()
    dev_texts   = dev_df['text'].astype(str).tolist()
    test_texts  = test_df['text'].astype(str).tolist()

    train_labels = train_df['Label'].values
    dev_labels   = dev_df['Label'].values
    test_labels  = test_df['Label'].values

    # Fit label encoder on training labels
    le = LabelEncoder()
    le.fit(train_labels)
    print(f"\nClasses: {le.classes_.tolist()}")

    # --- Extract features ---
    print('\nLoading VetBERT feature extractor (havocy28/VetBERTDx)...')
    extractor = VetBERTHybridFeatureExtractor()

    print('\nExtracting training features...')
    train_feats = extract_all_features(extractor, train_texts)
    print('Extracting dev features...')
    dev_feats   = extract_all_features(extractor, dev_texts)
    print('Extracting test features...')
    test_feats  = extract_all_features(extractor, test_texts)

    # --- Train and evaluate ---
    all_results = []

    for feat_name in args.feature_configs:
        X_train = train_feats[feat_name]
        X_dev   = dev_feats[feat_name]
        X_test  = test_feats[feat_name]

        print(f'\n{"="*60}')
        print(f'  Feature config: {feat_name}  (dim={X_train.shape[1]})')
        print(f'{"="*60}')

        for clf_name in args.classifiers:
            print(f'\n  Classifier: {clf_name}')

            # Scale features
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)

            clf = build_classifier(clf_name)
            y_train_enc = le.transform(train_labels)
            clf.fit(X_train_s, y_train_enc)

            # Evaluate on all splits
            dev_metrics  = evaluate_split(clf, scaler, X_dev,  dev_labels,  le)
            test_metrics = evaluate_split(clf, scaler, X_test, test_labels, le)

            print(f"    Dev  macro F1: {dev_metrics['macro_f1']:.4f}  "
                  f"acc: {dev_metrics['accuracy']:.4f}")
            print(f"    Test macro F1: {test_metrics['macro_f1']:.4f}  "
                  f"acc: {test_metrics['accuracy']:.4f}")
            print(f"\n    Test Classification Report:")
            print(test_metrics['report'])

            # Save model bundle
            bundle_key = f"{feat_name}_{clf_name}"
            save_dir = os.path.join(args.models_dir, bundle_key)
            os.makedirs(save_dir, exist_ok=True)
            joblib.dump(clf, os.path.join(save_dir, 'classifier.joblib'))
            joblib.dump(scaler, os.path.join(save_dir, 'scaler.joblib'))
            joblib.dump(le, os.path.join(save_dir, 'label_encoder.joblib'))
            print(f"    Model saved -> {save_dir}")

            result_row = {
                'feature_config': feat_name,
                'classifier': clf_name,
                'feature_dim': X_train.shape[1],
                'dev_accuracy':        round(dev_metrics['accuracy'], 4),
                'dev_macro_precision': round(dev_metrics['macro_precision'], 4),
                'dev_macro_recall':    round(dev_metrics['macro_recall'], 4),
                'dev_macro_f1':        round(dev_metrics['macro_f1'], 4),
                'test_accuracy':        round(test_metrics['accuracy'], 4),
                'test_macro_precision': round(test_metrics['macro_precision'], 4),
                'test_macro_recall':    round(test_metrics['macro_recall'], 4),
                'test_macro_f1':        round(test_metrics['macro_f1'], 4),
            }
            for cls in le.classes_:
                slug = _slugify_label(cls)
                result_row[f'test_precision_{slug}'] = round(test_metrics[f'precision_{slug}'], 4)
                result_row[f'test_recall_{slug}']    = round(test_metrics[f'recall_{slug}'], 4)
                result_row[f'test_f1_{slug}']        = round(test_metrics[f'f1_{slug}'], 4)
            all_results.append(result_row)

    # --- Save comparison table ---
    results_df = pd.DataFrame(all_results).sort_values('test_macro_f1', ascending=False)
    out_path = os.path.join(args.output_dir, 'ml_comparison_ann04062026.csv')
    results_df.to_csv(out_path, index=False)

    print('\n' + '='*70)
    print('  RESULTS SUMMARY (sorted by test macro F1)')
    print('='*70)
    print(results_df.to_string(index=False))
    print(f'\nComparison table saved -> {out_path}')
    print(f'Models saved          -> {args.models_dir}')


if __name__ == '__main__':
    main()
