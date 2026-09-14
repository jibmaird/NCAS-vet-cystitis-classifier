#!/usr/bin/env python3
"""
Prepare ann04062026 annotation data for binary Sporadic Cystitis (SC) classification.

Converts multi-label annotation CSVs to the standard (text, Label) format
expected by the training scripts.

Unlike the UTD-only binary task (prepare_ann04062026_data.py), this task uses
*every* record in each file -- there is no exclusion step. Every consultation
is labelled as either a confirmed Sporadic Cystitis case or "Other" (which
covers UTD Other, non-UTD syndromes, and unannotated cases alike).

File mapping (train/dev/test):
  dev-utd-predicted.csv   -> processed/train_binary_SC.csv  (training set, predicted labels)
  dev-all-syndromes.csv   -> processed/dev_binary_SC.csv    (dev/validation, human annotated)
  test-all-syndromes.csv  -> processed/test_binary_SC.csv   (test set, human annotated)

Label derivation from the multi-label columns:
  - 'Sporadic Cystitis': utd_spor_cyst is numeric 1 or 1.0
  - 'Other':             everything else (UTD Other, non-UTD syndromes, unannotated)

Note: Files use latin-1 encoding. Output CSVs are written as UTF-8.
"""

import os
import sys
import argparse
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/ann04062026')
OUTPUT_DIR = os.path.join(DATA_DIR, 'processed')

POSITIVE_LABEL = 'Sporadic Cystitis'
NEGATIVE_LABEL = 'Other'


def _is_sporadic_cystitis(val) -> bool:
    """True if the utd_spor_cyst value represents a confirmed cystitis sporadic case."""
    if pd.isna(val):
        return False
    try:
        return float(val) == 1.0
    except (ValueError, TypeError):
        return False


def get_binary_label(row) -> str:
    """
    Derive the binary Sporadic-Cystitis-vs-Other label from a multi-label
    annotation row. Every row receives a label -- there is no exclusion.

    Returns 'Sporadic Cystitis' or 'Other'.
    """
    if _is_sporadic_cystitis(row.get('utd_spor_cyst')):
        return POSITIVE_LABEL
    return NEGATIVE_LABEL


def prepare_file(input_path: str, output_path: str, split_name: str) -> pd.DataFrame:
    """
    Read one annotation CSV, derive binary SC labels, and write the output CSV.

    Returns the filtered DataFrame (columns: text, Label).
    """
    print(f"\n{'='*60}")
    print(f"Processing {split_name}: {os.path.basename(input_path)}")

    df = pd.read_csv(input_path, encoding='latin-1')
    print(f"  Loaded {len(df):,} rows")

    df['Label'] = df.apply(get_binary_label, axis=1)

    sc_n = (df['Label'] == POSITIVE_LABEL).sum()
    other_n = (df['Label'] == NEGATIVE_LABEL).sum()

    print(f"  Label breakdown (before text filter):")
    print(f"    {POSITIVE_LABEL:<20}: {sc_n:>4}")
    print(f"    {NEGATIVE_LABEL:<20}: {other_n:>4}")

    out = df[['text', 'Label']].copy()
    out = out.dropna(subset=['text'])
    out = out[out['text'].astype(str).str.strip() != '']
    out = out.reset_index(drop=True)

    print(f"  Final rows after text cleanup: {len(out)}")
    if len(out) > 0:
        vc = out['Label'].value_counts()
        for lbl, cnt in vc.items():
            print(f"    {lbl:<20}: {cnt} ({cnt/len(out)*100:.1f}%)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False, encoding='utf-8')
    print(f"  Saved -> {output_path}")

    return out


def main():
    parser = argparse.ArgumentParser(
        description='Prepare ann04062026 annotation data for binary Sporadic Cystitis (SC) classification'
    )
    parser.add_argument(
        '--data_dir', type=str, default=DATA_DIR,
        help='Directory containing ann04062026 CSV files'
    )
    parser.add_argument(
        '--output_dir', type=str, default=OUTPUT_DIR,
        help='Directory for processed output files'
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    print("Binary Sporadic Cystitis (SC) data preparation — ann04062026")
    print(f"Source  : {data_dir}")
    print(f"Output  : {output_dir}")
    print(f"Labels  : '{POSITIVE_LABEL}' (positive) vs. '{NEGATIVE_LABEL}' (negative) -- no exclusions")

    train_df = prepare_file(
        input_path=os.path.join(data_dir, 'dev-utd-predicted.csv'),
        output_path=os.path.join(output_dir, 'train_binary_SC.csv'),
        split_name='TRAIN'
    )

    dev_df = prepare_file(
        input_path=os.path.join(data_dir, 'dev-all-syndromes.csv'),
        output_path=os.path.join(output_dir, 'dev_binary_SC.csv'),
        split_name='DEV'
    )

    test_df = prepare_file(
        input_path=os.path.join(data_dir, 'test-all-syndromes.csv'),
        output_path=os.path.join(output_dir, 'test_binary_SC.csv'),
        split_name='TEST'
    )

    print(f"\n{'='*60}")
    print("Summary")
    print(f"  Train : {len(train_df):>4} samples -> {os.path.join(output_dir, 'train_binary_SC.csv')}")
    print(f"  Dev   : {len(dev_df):>4} samples -> {os.path.join(output_dir, 'dev_binary_SC.csv')}")
    print(f"  Test  : {len(test_df):>4} samples -> {os.path.join(output_dir, 'test_binary_SC.csv')}")
    print(f"\nNext step — train with:")
    print(f"  python scripts/training/fine_tune_multiknowledge_endtoend.py \\")
    print(f"    --fusion_strategy cross_modal \\")
    print(f"    --train_file {os.path.join(output_dir, 'train_binary_SC.csv')} \\")
    print(f"    --val_file   {os.path.join(output_dir, 'dev_binary_SC.csv')}")


if __name__ == '__main__':
    main()
