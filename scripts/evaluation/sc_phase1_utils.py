#!/usr/bin/env python3
"""
Shared utilities for Phase 1 SC statistical-analysis scripts
(docs/reports/sc_experiment_plan.md).

Discovers and loads the per-record prediction CSVs produced for the 9
deep-learning architectures (results/evaluation/ann04062026_SC/predictions_*_SC.csv)
and the 12 traditional-ML baselines
(results/predictions/ann04062026_SC/predictions_ml_*_SC.csv), all sharing the
columns: text, Label, predicted_label, prob_sporadic_cystitis, prob_other, correct.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score

_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_script_dir, '../..'))

POSITIVE_LABEL = 'Sporadic Cystitis'
NEGATIVE_LABEL = 'Other'
LABELS = [NEGATIVE_LABEL, POSITIVE_LABEL]

DEFAULT_PREDICTION_DIRS = [
    os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC'),
    os.path.join(PROJECT_ROOT, 'results/predictions/ann04062026_SC'),
]

# Friendly display names for the 9 DL architectures, used in tables/plots.
DL_DISPLAY_NAMES = {
    'vetbert_finetune': 'VetBERTBinary (fine-tuned)',
    'dualhead_weighted': 'Dual-Head: Weighted',
    'dualhead_concatenation': 'Dual-Head: Concatenation',
    'dualhead_cross_attention': 'Dual-Head: Cross-Attention',
    'dualhead_multi_task': 'Dual-Head: Multi-Task',
    'attention': 'Multi-Knowledge: Attention',
    'gated': 'Multi-Knowledge: Gated',
    'cross_modal': 'Multi-Knowledge: Cross-Modal',
    'hierarchical': 'Multi-Knowledge: Hierarchical',
}

_FNAME_RE = re.compile(r'^predictions_(.+)_SC\.csv$')

REQUIRED_COLUMNS = {'text', 'Label', 'predicted_label', 'prob_sporadic_cystitis', 'prob_other'}


def display_name(model_name: str) -> str:
    if model_name in DL_DISPLAY_NAMES:
        return DL_DISPLAY_NAMES[model_name]
    if model_name.startswith('ml_'):
        return 'ML: ' + model_name[len('ml_'):].replace('_', ' ')
    return model_name


def discover_predictions(dirs=None) -> dict:
    """Return {model_name: path} for all predictions_*_SC.csv files in dirs."""
    if dirs is None:
        dirs = DEFAULT_PREDICTION_DIRS
    found = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            m = _FNAME_RE.match(fname)
            if m:
                found[m.group(1)] = os.path.join(d, fname)
    return found


def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df


def binary_metrics(y_true, y_pred) -> dict:
    """Precision/recall/F1 for SC-positive and Other, plus accuracy and macro F1."""
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_sc': prec[1],
        'recall_sc': rec[1],
        'f1_sc': f1[1],
        'precision_other': prec[0],
        'recall_other': rec[0],
        'f1_other': f1[0],
        'macro_f1': macro_f1,
    }


def predictions_at_threshold(df: pd.DataFrame, threshold: float) -> np.ndarray:
    """Predicted labels obtained by thresholding prob_sporadic_cystitis."""
    return np.where(df['prob_sporadic_cystitis'].values >= threshold, POSITIVE_LABEL, NEGATIVE_LABEL)
