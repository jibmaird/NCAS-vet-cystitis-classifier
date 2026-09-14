#!/usr/bin/env python3
"""
Stage-2 feature extraction: extract VetBERT CLS embeddings and syndrome
probability vectors for the 804 is_utd records in dev-utd-predicted.csv.

is_utd = (utd_spor_cyst == 1) OR (utd_other is not null)
  -> 575 Sporadic Cystitis, 229 Other-UTD

Outputs (results/evaluation/ann04062026_SC/stage2_features/):
  embeddings.npy  -- (804, 768)          float32 CLS embeddings
  syndromes.npy   -- (804, n_syndromes)  float32 syndrome probabilities
  meta.csv        -- consult_table_id, text, label, utd_other

These files are the input to sc_stage2_ml_cv.py (5-fold ML cross-validation,
CPU). The VetBERTBinary CV (sc_stage2_vetbert_cv.py) tokenises from raw text
and does NOT use these cached features.

Requires GPU + vetbert_minimal conda env.
Usage:
    python scripts/evaluation/sc_stage2_extract_features.py
"""
import os
import sys
import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
from sc_phase1_utils import PROJECT_ROOT

sys.path.insert(0, PROJECT_ROOT)
from src.models.vetbert_hybrid_extractor import VetBERTHybridFeatureExtractor

_TRAIN_POOL = os.path.join(PROJECT_ROOT, 'data/ann04062026/dev-utd-predicted.csv')
_OUT_DIR = os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC/stage2_features')

POSITIVE_LABEL = 'Sporadic Cystitis'
NEGATIVE_LABEL = 'Other'


def load_is_utd_records(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='latin-1')
    is_sc = df['utd_spor_cyst'].astype(str).isin(['1', '1.0'])
    is_other_utd = df['utd_other'].notna()
    mask = is_sc | is_other_utd
    sub = df[mask].copy().reset_index(drop=True)
    sub['label'] = sub.apply(
        lambda r: POSITIVE_LABEL if str(r['utd_spor_cyst']) in ('1', '1.0') else NEGATIVE_LABEL,
        axis=1,
    )
    return sub


def main():
    print('=' * 65)
    print('  Stage-2 feature extraction — VetBERT CLS embeddings + syndromes')
    print('=' * 65)

    print(f'\nLoading is_utd records from:\n  {_TRAIN_POOL}')
    df = load_is_utd_records(_TRAIN_POOL)
    n_sc = int((df['label'] == POSITIVE_LABEL).sum())
    n_other = int((df['label'] == NEGATIVE_LABEL).sum())
    print(f'  Records: {len(df)}  (SC={n_sc}, Other-UTD={n_other})')

    texts = df['text'].astype(str).tolist()

    print('\nLoading VetBERT feature extractor ...')
    extractor = VetBERTHybridFeatureExtractor()

    print('\nExtracting CLS embeddings (768-dim) ...')
    embeddings = extractor.extract_embeddings(texts)
    print(f'  shape: {embeddings.shape}')

    print('\nExtracting syndrome probability vectors ...')
    syndromes = extractor.extract_syndrome_probabilities(texts)
    print(f'  shape: {syndromes.shape}')

    os.makedirs(_OUT_DIR, exist_ok=True)
    np.save(os.path.join(_OUT_DIR, 'embeddings.npy'), embeddings)
    np.save(os.path.join(_OUT_DIR, 'syndromes.npy'), syndromes)

    meta_cols = [c for c in ('consult_table_id', 'text', 'label', 'utd_other') if c in df.columns]
    df[meta_cols].to_csv(os.path.join(_OUT_DIR, 'meta.csv'), index=False)

    print(f'\nSaved -> {_OUT_DIR}/')
    print(f'  embeddings.npy  : {embeddings.shape}')
    print(f'  syndromes.npy   : {syndromes.shape}')
    print(f'  meta.csv        : {len(df)} records')
    print('\nNext step: python scripts/evaluation/sc_stage2_ml_cv.py')


if __name__ == '__main__':
    main()
