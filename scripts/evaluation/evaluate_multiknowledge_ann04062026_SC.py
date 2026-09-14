#!/usr/bin/env python3
"""
Evaluate a trained VetBERT Multi-Knowledge model on the ann04062026 SC
(Sporadic Cystitis vs. Other) test data.

Loads the model, runs inference on the test set, and reports:
    - Accuracy
    - Precision, Recall, F1 (per class and macro)
    - Confusion matrix
    - Full classification report

Saves a summary CSV and a detailed predictions CSV under results/evaluation/.

Usage:
    python scripts/evaluation/evaluate_multiknowledge_ann04062026_SC.py \
        --model_path checkpoints/best_model_ann04062026_SC_cross_modal.pt \
        --fusion_strategy cross_modal \
        --test_file data/ann04062026/processed/test_binary_SC.csv

    # Evaluate all four saved checkpoints at once:
    for strat in attention gated cross_modal hierarchical; do
      python scripts/evaluation/evaluate_multiknowledge_ann04062026_SC.py \
        --model_path checkpoints/best_model_ann04062026_SC_${strat}.pt \
        --fusion_strategy $strat
    done
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'training'))
from fine_tune_multiknowledge_endtoend import VetBERTMultiKnowledgeClassifier

MODEL_NAME = 'havocy28/VetBERTDx'
CLASS_NAMES = ['Sporadic Cystitis', 'Other']
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def load_model(model_path, fusion_strategy, device):
    model = VetBERTMultiKnowledgeClassifier(
        base_model_path=MODEL_NAME,
        device=device,
        fusion_strategy=fusion_strategy,
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_inference(model, dataloader, device):
    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(np.argmax(probs, axis=-1))
            all_probs.extend(probs)
    return np.array(all_preds), np.array(all_probs)


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Multi-Knowledge VetBERT on ann04062026 SC test data'
    )
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--fusion_strategy', type=str, required=True,
                        choices=['attention', 'gated', 'cross_modal', 'hierarchical'])
    parser.add_argument('--test_file', type=str,
                        default='data/ann04062026/processed/test_binary_SC.csv',
                        help='Test CSV with columns: text, Label')
    parser.add_argument('--output_dir', type=str,
                        default='results/evaluation/ann04062026_SC',
                        help='Directory for evaluation output files')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=512)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print_section('VetBERT Multi-Knowledge Evaluation — ann04062026 SC')
    print(f"  Fusion strategy  : {args.fusion_strategy}")
    print(f"  Model checkpoint : {args.model_path}")
    print(f"  Test file        : {args.test_file}")
    print(f"  Device           : {device}")

    # --- Load test data ---
    df = pd.read_csv(args.test_file, encoding='utf-8')
    print(f"\nTest samples loaded: {len(df)}")

    if 'Label' not in df.columns:
        print("ERROR: test file must contain a 'Label' column for evaluation.")
        sys.exit(1)

    texts = df['text'].astype(str).tolist()
    true_labels_str = df['Label'].values
    true_labels = np.array([LABEL_TO_IDX[l] for l in true_labels_str])

    print("Label distribution:")
    for lbl, cnt in pd.Series(true_labels_str).value_counts().items():
        print(f"  {lbl:<20}: {cnt} ({cnt/len(df)*100:.1f}%)")

    # --- Tokenise ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    enc = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=args.max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    dataset = TensorDataset(enc['input_ids'], enc['attention_mask'])
    dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=args.batch_size,
    )

    # --- Load model ---
    print("\nLoading model...")
    if not os.path.exists(args.model_path):
        print(f"ERROR: checkpoint not found at {args.model_path}")
        print("Train first with the corresponding SBATCH script.")
        sys.exit(1)
    model = load_model(args.model_path, args.fusion_strategy, device)
    print("Model loaded.")

    # --- Inference ---
    print("Running inference...")
    pred_idx, probs = run_inference(model, dataloader, device)

    pred_labels = [CLASS_NAMES[p] for p in pred_idx]

    # --- Metrics ---
    acc = accuracy_score(true_labels, pred_idx)
    prec, rec, f1_mac, _ = precision_recall_fscore_support(
        true_labels, pred_idx, average='macro'
    )
    prec_per, rec_per, f1_per_class, _ = precision_recall_fscore_support(
        true_labels, pred_idx, average=None
    )
    cm = confusion_matrix(true_labels, pred_idx)

    print_section('Results')
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  Macro Precision  : {prec:.4f}")
    print(f"  Macro Recall     : {rec:.4f}")
    print(f"  Macro F1         : {f1_mac:.4f}")
    print()
    for i, cn in enumerate(CLASS_NAMES):
        print(f"  [{cn:<20}] Precision: {prec_per[i]:.4f}  Recall: {rec_per[i]:.4f}  F1: {f1_per_class[i]:.4f}")
    print()
    print("  Confusion matrix (rows=true, cols=pred):")
    print(f"                   {CLASS_NAMES[0]:<20} {CLASS_NAMES[1]}")
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:<20} {row[0]:>6}  {row[1]:>6}")
    print()
    print("  Classification Report:")
    print(classification_report(true_labels_str, pred_labels))

    # --- Save outputs ---
    os.makedirs(args.output_dir, exist_ok=True)

    # Summary CSV (one row per strategy for easy comparison)
    summary = {
        'fusion_strategy': args.fusion_strategy,
        'test_file': args.test_file,
        'n_samples': len(df),
        'accuracy': round(acc, 4),
        'macro_precision': round(prec, 4),
        'macro_recall': round(rec, 4),
        'macro_f1': round(f1_mac, 4),
    }
    for i, cn in enumerate(CLASS_NAMES):
        key = cn.lower().replace(' ', '_')
        summary[f'precision_{key}'] = round(prec_per[i], 4)
        summary[f'recall_{key}']    = round(rec_per[i], 4)
        summary[f'f1_{key}']        = round(f1_per_class[i], 4)

    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(
        args.output_dir, f'summary_{args.fusion_strategy}_SC.csv'
    )
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved -> {summary_path}")

    # Detailed predictions CSV
    detail_df = df.copy()
    detail_df['predicted_label'] = pred_labels
    detail_df['prob_sporadic_cystitis'] = probs[:, 0]
    detail_df['prob_other'] = probs[:, 1]
    detail_df['correct'] = detail_df['Label'] == detail_df['predicted_label']

    detail_path = os.path.join(
        args.output_dir, f'predictions_{args.fusion_strategy}_SC.csv'
    )
    detail_df.to_csv(detail_path, index=False)
    print(f"Detailed predictions saved -> {detail_path}")


if __name__ == '__main__':
    main()
