#!/usr/bin/env python3
"""
Stage-2 ML cross-validation: 5-fold stratified CV on the 804 is_utd records
(575 Sporadic Cystitis / 229 Other-UTD), using pre-extracted VetBERT features
from sc_stage2_extract_features.py.

Runs 12 ML configs (4 feature sets x {logistic, random_forest, svm}).
Per-fold SC-class F1/precision/recall, macro-F1 and Other-UTD recall are
averaged across folds (F1 also gets its std), as in the paper's Table 4.
Bootstrap CIs (2000 resamples, stratified, seed=42) on pooled out-of-fold
predictions are computed for F1(SC), AUC-ROC and AUC-PR.

This is part of the paper's Result (3): SC vs other-UTD (Stage 2).

Input (from sc_stage2_extract_features.py):
  results/evaluation/ann04062026_SC/stage2_features/{embeddings.npy, syndromes.npy, meta.csv}

Outputs:
  results/evaluation/ann04062026_SC/phase_stage2_ml_cv_summary.csv

Run AFTER the GPU feature extraction job. No GPU required.
Usage:
    python scripts/evaluation/sc_stage2_ml_cv.py
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
from sc_phase1_utils import PROJECT_ROOT

_FEAT_DIR = os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC/stage2_features')
_OUT_PATH = os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC/phase_stage2_ml_cv_summary.csv')

N_FOLDS = 5
SEED = 42
N_BOOTSTRAP = 2000
POSITIVE_LABEL = 'Sporadic Cystitis'


def build_classifier(name: str):
    if name == 'random_forest':
        return RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    if name == 'logistic':
        return LogisticRegression(max_iter=2000, random_state=SEED)
    if name == 'svm':
        return SVC(probability=True, random_state=SEED)
    raise ValueError(f'Unknown classifier: {name}')


def bootstrap_ci(y_true_binary, y_score, score_fn, rng: np.random.Generator) -> tuple:
    """95% CI for score_fn via stratified bootstrap. y_true_binary: 0/1 array."""
    pos_idx = np.where(y_true_binary == 1)[0]
    neg_idx = np.where(y_true_binary == 0)[0]
    vals = []
    for _ in range(N_BOOTSTRAP):
        samp = np.concatenate([
            rng.choice(pos_idx, len(pos_idx), replace=True),
            rng.choice(neg_idx, len(neg_idx), replace=True),
        ])
        try:
            vals.append(score_fn(y_true_binary[samp], y_score[samp]))
        except ValueError:
            vals.append(float('nan'))
    vals = np.array(vals)
    valid = vals[~np.isnan(vals)]
    if len(valid) == 0:
        return float('nan'), float('nan')
    return float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))


def main():
    print('=' * 65)
    print('  Stage-2 ML 5-fold CV  (12 configs x 5 folds)')
    print('=' * 65)

    feat_dir = _FEAT_DIR
    if not os.path.isdir(feat_dir):
        sys.exit(f'ERROR: feature directory not found: {feat_dir}\n'
                 'Run sc_stage2_extract_features.py first.')

    print(f'\nLoading features from {feat_dir} ...')
    emb = np.load(os.path.join(feat_dir, 'embeddings.npy'))
    syn = np.load(os.path.join(feat_dir, 'syndromes.npy'))
    meta = pd.read_csv(os.path.join(feat_dir, 'meta.csv'))
    print(f'  embeddings : {emb.shape}')
    print(f'  syndromes  : {syn.shape}')

    le = LabelEncoder()
    y = le.fit_transform(meta['label'].values)
    # LabelEncoder sorts alphabetically: Other=0, Sporadic Cystitis=1
    sc_idx = list(le.classes_).index(POSITIVE_LABEL)
    y_binary = (y == sc_idx).astype(int)  # 1 = SC, 0 = Other
    n_pos = int(y_binary.sum())
    n_neg = int((1 - y_binary).sum())
    print(f'\nDataset: {len(y)} records  ({n_pos} SC / {n_neg} Other-UTD)\n')

    feat_configs = {
        'embeddings_only': emb,
        'syndromes_only': syn,
        'hybrid_equal': np.concatenate([emb, syn], axis=1),
        'hybrid_2x_syn': np.concatenate([emb, syn * 2.0], axis=1),
    }
    classifiers = ['random_forest', 'logistic', 'svm']

    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(kf.split(emb, y_binary))
    rng = np.random.default_rng(SEED)

    rows = []

    for feat_name, X in feat_configs.items():
        for clf_name in classifiers:
            key = f'{feat_name} / {clf_name}'
            print(f'  {key}')
            fold_f1, fold_prec, fold_rec = [], [], []
            fold_macro_f1, fold_rec_other = [], []
            all_y, all_proba, all_pred = [], [], []

            for fold_k, (train_idx, val_idx) in enumerate(folds):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y_binary[train_idx], y_binary[val_idx]

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_val_s = scaler.transform(X_val)

                clf = build_classifier(clf_name)
                clf.fit(X_tr_s, y_tr)

                proba = clf.predict_proba(X_val_s)[:, sc_idx]
                y_pred = clf.predict(X_val_s)

                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_val, y_pred, labels=[1], average=None, zero_division=0
                )
                fold_f1.append(float(f1[0]))
                fold_prec.append(float(prec[0]))
                fold_rec.append(float(rec[0]))
                _, rec_both, f1_both, _ = precision_recall_fscore_support(
                    y_val, y_pred, labels=[1, 0], average=None, zero_division=0
                )
                fold_macro_f1.append(float(f1_both.mean()))
                fold_rec_other.append(float(rec_both[1]))
                all_y.append(y_val)
                all_proba.append(proba)
                all_pred.append(y_pred)

                cm = confusion_matrix(y_val, y_pred, labels=[1, 0])
                tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
                print(f'    fold {fold_k}: F1={f1[0]:.3f}  P={prec[0]:.3f}  R={rec[0]:.3f}'
                      f'  TP={tp} FN={fn} FP={fp}')

            pool_y = np.concatenate(all_y)
            pool_proba = np.concatenate(all_proba)
            pool_pred = np.concatenate(all_pred)

            auc_roc = roc_auc_score(pool_y, pool_proba)
            auc_pr = average_precision_score(pool_y, pool_proba)
            roc_lo, roc_hi = bootstrap_ci(pool_y, pool_proba, roc_auc_score, rng)
            pr_lo, pr_hi = bootstrap_ci(pool_y, pool_proba, average_precision_score, rng)
            # Own generator, so the AUC CIs above keep their original random stream.
            f1_lo, f1_hi = bootstrap_ci(pool_y, pool_pred,
                                        lambda yt, yp: f1_score(yt, yp, zero_division=0),
                                        np.random.default_rng(SEED))

            row = {
                'feature_config': feat_name,
                'classifier': clf_name,
                'mean_f1_sc': float(np.mean(fold_f1)),
                'std_f1_sc': float(np.std(fold_f1)),
                'f1_sc_ci_lower': f1_lo,
                'f1_sc_ci_upper': f1_hi,
                'mean_precision_sc': float(np.mean(fold_prec)),
                'mean_recall_sc': float(np.mean(fold_rec)),
                'mean_macro_f1': float(np.mean(fold_macro_f1)),
                'mean_recall_other': float(np.mean(fold_rec_other)),
                'auc_roc': float(auc_roc),
                'auc_roc_ci_lower': roc_lo,
                'auc_roc_ci_upper': roc_hi,
                'auc_pr': float(auc_pr),
                'auc_pr_ci_lower': pr_lo,
                'auc_pr_ci_upper': pr_hi,
            }
            rows.append(row)
            print(f'    mean F1={np.mean(fold_f1):.3f} ± {np.std(fold_f1):.3f} [{f1_lo:.3f}, {f1_hi:.3f}]'
                  f'  macro-F1={np.mean(fold_macro_f1):.3f}'
                  f'  AUC-ROC={auc_roc:.3f} [{roc_lo:.3f}, {roc_hi:.3f}]\n')

    out_df = pd.DataFrame(rows).sort_values('mean_f1_sc', ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    out_df.to_csv(_OUT_PATH, index=False)

    print('=' * 65)
    print(f'Saved -> {_OUT_PATH}')
    print()
    print(out_df[['feature_config', 'classifier', 'mean_f1_sc', 'std_f1_sc', 'mean_macro_f1',
                  'auc_roc', 'auc_roc_ci_lower', 'auc_roc_ci_upper']].to_string(index=False))


if __name__ == '__main__':
    main()
