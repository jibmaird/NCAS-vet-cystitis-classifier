#!/usr/bin/env python3
"""
Prepare ann04062026 annotation data for binary UTD classification.

Converts multi-label annotation CSVs to the standard (text, Label) format
expected by the training scripts.

File mapping (train/dev/test):
  dev-utd-predicted.csv   -> processed/train_binary.csv  (training set, predicted labels)
  dev-all-syndromes.csv   -> processed/dev_binary.csv    (dev/validation, human annotated)
  test-all-syndromes.csv  -> processed/test_binary.csv   (test set, human annotated)

Label derivation from the multi-label columns:
  - 'Cystitis Sporadic': utd_spor_cyst is numeric 1 or 1.0
  - 'UTD Other':         utd_spor_cyst is NOT cystitis AND
                         utd_other or utd_journey has a non-excluded value
  - Excluded:            no UTD label present, or label is 'unclear'/'0'

Note: Files use latin-1 encoding. Output CSVs are written as UTF-8.
"""

import os
import sys
import argparse
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/ann04062026')
OUTPUT_DIR = os.path.join(DATA_DIR, 'processed')

_EXCLUDE_STR = {'', '0', '0.0', 'nan', 'unclear', 'uncertain'}


def _is_sporadic_cystitis(val) -> bool:
    """True if the utd_spor_cyst value represents a confirmed cystitis sporadic case."""
    if pd.isna(val):
        return False
    try:
        return float(val) == 1.0
    except (ValueError, TypeError):
        return False


def _is_utd_other(val) -> bool:
    """True if an other-UTD column contains a meaningful (non-excluded) value."""
    if pd.isna(val):
        return False
    return str(val).strip().lower() not in _EXCLUDE_STR


def get_binary_label(row) -> str | None:
    """
    Derive the binary label from a multi-label annotation row.

    Returns 'Cystitis Sporadic', 'UTD Other', or None (exclude row).
    """
    if _is_sporadic_cystitis(row.get('utd_spor_cyst')):
        return 'Cystitis Sporadic'
    if _is_utd_other(row.get('utd_other')) or _is_utd_other(row.get('utd_journey')):
        return 'UTD Other'
    return None


def prepare_file(input_path: str, output_path: str, split_name: str) -> pd.DataFrame:
    """
    Read one annotation CSV, derive binary labels, and write the output CSV.

    Returns the filtered DataFrame (columns: text, Label).
    """
    print(f"\n{'='*60}")
    print(f"Processing {split_name}: {os.path.basename(input_path)}")

    df = pd.read_csv(input_path, encoding='latin-1')
    print(f"  Loaded {len(df):,} rows")

    df['Label'] = df.apply(get_binary_label, axis=1)

    total_labeled = df['Label'].notna().sum()
    cystitis_n = (df['Label'] == 'Cystitis Sporadic').sum()
    other_n = (df['Label'] == 'UTD Other').sum()
    excluded_n = df['Label'].isna().sum()

    print(f"  Label breakdown (before text filter):")
    print(f"    Cystitis Sporadic : {cystitis_n:>4}")
    print(f"    UTD Other         : {other_n:>4}")
    print(f"    Excluded (no UTD) : {excluded_n:>4}")

    out = df[df['Label'].notna()][['text', 'Label']].copy()
    out = out.dropna(subset=['text'])
    out = out[out['text'].astype(str).str.strip() != '']
    out = out.reset_index(drop=True)

    print(f"  Final rows after text cleanup: {len(out)}")
    if len(out) > 0:
        vc = out['Label'].value_counts()
        for lbl, cnt in vc.items():
            print(f"    {lbl:<22}: {cnt} ({cnt/len(out)*100:.1f}%)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False, encoding='utf-8')
    print(f"  Saved -> {output_path}")

    return out


def main():
    parser = argparse.ArgumentParser(
        description='Prepare ann04062026 annotation data for binary UTD classification'
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

    print("Binary UTD data preparation — ann04062026")
    print(f"Source  : {data_dir}")
    print(f"Output  : {output_dir}")

    train_df = prepare_file(
        input_path=os.path.join(data_dir, 'dev-utd-predicted.csv'),
        output_path=os.path.join(output_dir, 'train_binary.csv'),
        split_name='TRAIN'
    )

    dev_df = prepare_file(
        input_path=os.path.join(data_dir, 'dev-all-syndromes.csv'),
        output_path=os.path.join(output_dir, 'dev_binary.csv'),
        split_name='DEV'
    )

    test_df = prepare_file(
        input_path=os.path.join(data_dir, 'test-all-syndromes.csv'),
        output_path=os.path.join(output_dir, 'test_binary.csv'),
        split_name='TEST'
    )

    print(f"\n{'='*60}")
    print("Summary")
    print(f"  Train : {len(train_df):>4} samples -> {os.path.join(output_dir, 'train_binary.csv')}")
    print(f"  Dev   : {len(dev_df):>4} samples -> {os.path.join(output_dir, 'dev_binary.csv')}")
    print(f"  Test  : {len(test_df):>4} samples -> {os.path.join(output_dir, 'test_binary.csv')}")
    print(f"\nNext step — train with:")
    print(f"  python scripts/training/fine_tune_multiknowledge_endtoend.py \\")
    print(f"    --fusion_strategy cross_modal \\")
    print(f"    --train_file {os.path.join(output_dir, 'train_binary.csv')} \\")
    print(f"    --val_file   {os.path.join(output_dir, 'dev_binary.csv')}")


if __name__ == '__main__':
    main()
