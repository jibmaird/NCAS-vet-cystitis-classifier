#!/usr/bin/env python3
"""
Cascade Step 1 (sc_experiment_plan.md follow-up): cache VetBERTDx's
P(urinary tract disorder) syndrome probability for dev_binary_SC.csv and
test_binary_SC.csv.

The leak-free ML cascade (sc_cascade_step1_utd_gate_leakfree.py) re-extracts
this signal via extract_all_features() (embeddings + syndrome probabilities)
every time it runs, since the 12 ML configs also need the embedding-based
feature sets. The DL cascade (sc_cascade_step1_dl_leakfree.py) only needs the
single "urinary tract disorder" syndrome probability, so this script extracts
just that (extract_syndrome_probabilities, no embeddings) and caches it to CSV
-- avoiding a GPU pass per analysis run.

No retraining; GPU used only for the one-off syndrome-probability extraction.

Usage:
    python scripts/evaluation/sc_cascade_step1_extract_utd_probs.py
"""

import os
import sys
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
from sc_phase1_utils import PROJECT_ROOT

sys.path.insert(0, PROJECT_ROOT)
from src.models.vetbert_hybrid_extractor import VetBERTHybridFeatureExtractor

_DEV_DEFAULT = os.path.join(PROJECT_ROOT, 'data/ann04062026/processed/dev_binary_SC.csv')
_TEST_DEFAULT = os.path.join(PROJECT_ROOT, 'data/ann04062026/processed/test_binary_SC.csv')
_OUTPUT_DEFAULT = os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC/utd_gate_probs_SC.csv')

SYNDROME_OF_INTEREST = 'urinary tract disorder'


def main():
    print('=' * 70)
    print('  Cascade Step 1 -- caching P(urinary tract disorder) for dev + test')
    print('=' * 70)

    dev_df = pd.read_csv(_DEV_DEFAULT)
    test_df = pd.read_csv(_TEST_DEFAULT)
    dev_df = dev_df[['text', 'Label']].copy()
    test_df = test_df[['text', 'Label']].copy()
    dev_df['split'] = 'dev'
    test_df['split'] = 'test'
    combined_df = pd.concat([dev_df, test_df], ignore_index=True)

    print(f"\ndev_binary_SC.csv : {len(dev_df)} records")
    print(f"test_binary_SC.csv: {len(test_df)} records")

    print('\nLoading VetBERT feature extractor (havocy28/VetBERTDx)...')
    extractor = VetBERTHybridFeatureExtractor()

    utd_idx = extractor.syndrome_labels.index(SYNDROME_OF_INTEREST)

    print('\nExtracting syndrome probabilities (dev + test)...')
    texts = combined_df['text'].astype(str).tolist()
    syndrome_probs = extractor.extract_syndrome_probabilities(texts)
    combined_df['p_urinary_tract_disorder'] = syndrome_probs[:, utd_idx]

    for split in ['dev', 'test']:
        sub = combined_df[combined_df['split'] == split]
        is_sc = sub['Label'] == 'Sporadic Cystitis'
        print(f"\n{split}: P(urinary tract disorder)")
        print(f"  SC-pos ({is_sc.sum()}): min={sub.loc[is_sc, 'p_urinary_tract_disorder'].min():.4f} "
              f"mean={sub.loc[is_sc, 'p_urinary_tract_disorder'].mean():.4f}")
        print(f"  Other  ({(~is_sc).sum()}): mean={sub.loc[~is_sc, 'p_urinary_tract_disorder'].mean():.4f}")

    os.makedirs(os.path.dirname(_OUTPUT_DEFAULT), exist_ok=True)
    combined_df.to_csv(_OUTPUT_DEFAULT, index=False)
    print(f"\nSaved -> {_OUTPUT_DEFAULT}")


if __name__ == '__main__':
    main()
