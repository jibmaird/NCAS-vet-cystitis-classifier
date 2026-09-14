#!/usr/bin/env python3
"""
Stage-2 VetBERTBinary cross-validation: 5-fold stratified CV on the 804
is_utd records from dev-utd-predicted.csv (575 SC / 229 other-UTD).

VetBERTBinary (havocy28/VetBERTDx + binary classifier head) is fine-tuned
from scratch on each fold's ~643 training records and evaluated on the ~161
held-out records. Label convention (matching fine_tune_on_cystitis_refined_binary.py):
  SC = 0  (positive class)
  Other = 1

Bootstrap CIs (2000 resamples, stratified, seed=42) are computed on the
pooled out-of-fold predictions. This is part of the paper's Result (3):
SC vs other-UTD (Stage 2).

Outputs:
  results/evaluation/ann04062026_SC/stage2_vetbert_cv/fold{k}_predictions.csv
  results/evaluation/ann04062026_SC/phase_stage2_vetbert_cv_summary.csv

Requires GPU + vetbert_minimal conda env. Estimated runtime: ~2-3 h for 5 folds.
Usage:
    python scripts/evaluation/sc_stage2_vetbert_cv.py
"""
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Subset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
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

# ── Hyper-parameters ──────────────────────────────────────────────────────────
MODEL_PATH = 'havocy28/VetBERTDx'
MAX_LENGTH = 512
N_FOLDS = 5
SEED = 42
MAX_EPOCHS = 60
PATIENCE = 10          # early-stop patience on val macro-F1
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WARMUP_STEPS = 10
GRADIENT_CLIP = 1.0
N_BOOTSTRAP = 2000

# ── Label convention: SC = 0, Other = 1 ───────────────────────────────────────
POSITIVE_LABEL = 'Sporadic Cystitis'
NEGATIVE_LABEL = 'Other'
SC_CLASS_IDX = 0       # class index for SC in VetBERTBinary output

_TRAIN_POOL = os.path.join(PROJECT_ROOT, 'data/ann04062026/dev-utd-predicted.csv')
_PRED_DIR = os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC/stage2_vetbert_cv')
_SUMMARY_PATH = os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC/phase_stage2_vetbert_cv_summary.csv')


# ── Model ──────────────────────────────────────────────────────────────────────
class VetBERTBinary(nn.Module):
    """VetBERT base + binary classifier head (SC vs Other)."""
    def __init__(self, base_model_path: str, device: torch.device):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(
            base_model_path,
            trust_remote_code=False,
            use_safetensors=True,
        )
        hidden_size = self.base_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 2),
        )
        self.to(device)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls)
        result = type('Out', (), {'logits': logits})
        if labels is not None:
            result.loss = nn.CrossEntropyLoss()(logits, labels)
        return result


# ── Utilities ─────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_is_utd_records(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='latin-1')
    is_sc = df['utd_spor_cyst'].astype(str).isin(['1', '1.0'])
    is_other_utd = df['utd_other'].notna()
    sub = df[is_sc | is_other_utd].copy().reset_index(drop=True)
    # SC = 0, Other = 1 (VetBERTBinary convention)
    sub['label_int'] = sub.apply(
        lambda r: 0 if str(r['utd_spor_cyst']) in ('1', '1.0') else 1, axis=1
    )
    sub['label_str'] = sub['label_int'].map({0: POSITIVE_LABEL, 1: NEGATIVE_LABEL})
    return sub


def tokenize(texts, tokenizer) -> tuple:
    enc = tokenizer.batch_encode_plus(
        [str(t) for t in texts],
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    return enc['input_ids'], enc['attention_mask']


# ── Training helpers ──────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids, attn_mask, labels = [t.to(device) for t in batch]
        model.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        scheduler.step()
        total_loss += out.loss.item()
    return total_loss / len(loader)


def evaluate_fold(model, loader, device) -> dict:
    """Return predictions, probabilities, and metric dict for a val loader."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids, attn_mask, labels = [t.to(device) for t in batch]
            out = model(input_ids=input_ids, attention_mask=attn_mask)
            all_logits.append(out.logits.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = logits.argmax(dim=-1).numpy()

    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[SC_CLASS_IDX], average=None, zero_division=0
    )
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)

    return {
        'labels': labels,
        'preds': preds,
        'prob_sc': probs[:, SC_CLASS_IDX],
        'prob_other': probs[:, 1 - SC_CLASS_IDX],
        'f1_sc': float(f1[0]),
        'precision_sc': float(prec[0]),
        'recall_sc': float(rec[0]),
        'macro_f1': float(macro_f1),
    }


# ── Bootstrap CI ──────────────────────────────────────────────────────────────
def bootstrap_ci(y_binary, y_score, score_fn, rng: np.random.Generator) -> tuple:
    """95% CI via stratified bootstrap. y_binary: 1=SC positive, 0=Other."""
    pos_idx = np.where(y_binary == 1)[0]
    neg_idx = np.where(y_binary == 0)[0]
    vals = []
    for _ in range(N_BOOTSTRAP):
        samp = np.concatenate([
            rng.choice(pos_idx, len(pos_idx), replace=True),
            rng.choice(neg_idx, len(neg_idx), replace=True),
        ])
        try:
            vals.append(score_fn(y_binary[samp], y_score[samp]))
        except ValueError:
            vals.append(float('nan'))
    vals = np.array(vals)
    valid = vals[~np.isnan(vals)]
    if len(valid) == 0:
        return float('nan'), float('nan')
    return float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('=' * 65)
    print('  Stage-2 VetBERTBinary 5-fold CV')
    print('=' * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')

    set_seed(SEED)

    print(f'\nLoading is_utd records from:\n  {_TRAIN_POOL}')
    df = load_is_utd_records(_TRAIN_POOL)
    n_sc = int((df['label_int'] == 0).sum())
    n_other = int((df['label_int'] == 1).sum())
    print(f'  Records: {len(df)}  (SC={n_sc}, Other-UTD={n_other})')

    print(f'\nTokenising {len(df)} records (max_length={MAX_LENGTH}) ...')
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=False, use_safetensors=True
    )
    input_ids, attention_masks = tokenize(df['text'].tolist(), tokenizer)
    labels_tensor = torch.tensor(df['label_int'].values, dtype=torch.long)
    dataset = TensorDataset(input_ids, attention_masks, labels_tensor)

    y = df['label_int'].values
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(kf.split(np.zeros(len(y)), y))

    os.makedirs(_PRED_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    fold_rows = []
    pool_labels, pool_prob_sc = [], []

    for fold_k, (train_idx, val_idx) in enumerate(folds):
        print(f'\n{"─" * 60}')
        print(f'  Fold {fold_k}  ({len(train_idx)} train / {len(val_idx)} val)')
        print(f'{"─" * 60}')

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(
            train_subset,
            sampler=RandomSampler(train_subset),
            batch_size=BATCH_SIZE,
        )
        val_loader = DataLoader(
            val_subset,
            sampler=SequentialSampler(val_subset),
            batch_size=BATCH_SIZE,
        )

        set_seed(SEED + fold_k)
        model = VetBERTBinary(base_model_path=MODEL_PATH, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=WARMUP_STEPS,
            num_training_steps=len(train_loader) * MAX_EPOCHS,
        )

        best_macro_f1 = -1.0
        best_state = None
        patience_count = 0

        for epoch in range(MAX_EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
            val_metrics = evaluate_fold(model, val_loader, device)

            if val_metrics['macro_f1'] > best_macro_f1:
                best_macro_f1 = val_metrics['macro_f1']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1

            if (epoch + 1) % 5 == 0 or patience_count == 0:
                print(f'  epoch {epoch:3d}  loss={train_loss:.4f}'
                      f'  F1_sc={val_metrics["f1_sc"]:.3f}'
                      f'  macro_F1={val_metrics["macro_f1"]:.3f}'
                      f'  {"* best" if patience_count == 0 else ""}')

            if patience_count >= PATIENCE:
                print(f'  Early stop at epoch {epoch} (patience={PATIENCE})')
                break

        # Reload best weights and do final eval
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        final = evaluate_fold(model, val_loader, device)

        val_labels = y[val_idx]
        sc_binary = (val_labels == 0).astype(int)  # 1=SC for AUC
        prob_sc = final['prob_sc']

        cm = confusion_matrix(val_labels, final['preds'], labels=[0, 1])
        tp, fn = cm[0, 0], cm[0, 1]
        fp, tn = cm[1, 0], cm[1, 1]

        print(f'\n  Fold {fold_k} best:  F1_sc={final["f1_sc"]:.3f}'
              f'  P={final["precision_sc"]:.3f}  R={final["recall_sc"]:.3f}'
              f'  TP={tp} FN={fn} FP={fp}')

        # Per-record predictions CSV
        fold_pred_df = df.iloc[val_idx].copy().reset_index(drop=True)
        fold_pred_df['fold'] = fold_k
        fold_pred_df['predicted_label'] = final['preds']
        fold_pred_df['predicted_label_str'] = pd.Series(final['preds']).map(
            {0: POSITIVE_LABEL, 1: NEGATIVE_LABEL}
        ).values
        fold_pred_df['prob_sc'] = prob_sc
        fold_pred_df['prob_other'] = final['prob_other']
        pred_cols = [c for c in ('consult_table_id', 'fold', 'label_int', 'label_str',
                                  'predicted_label', 'predicted_label_str',
                                  'prob_sc', 'prob_other') if c in fold_pred_df.columns]
        fold_pred_df[pred_cols].to_csv(
            os.path.join(_PRED_DIR, f'fold{fold_k}_predictions.csv'), index=False
        )

        fold_rows.append({
            'fold': fold_k,
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'f1_sc': final['f1_sc'],
            'precision_sc': final['precision_sc'],
            'recall_sc': final['recall_sc'],
            'macro_f1': final['macro_f1'],
            'tp': int(tp), 'fn': int(fn), 'fp': int(fp), 'tn': int(tn),
        })
        pool_labels.append(sc_binary)
        pool_prob_sc.append(prob_sc)

        del model
        torch.cuda.empty_cache()

    # Aggregate
    pool_labels = np.concatenate(pool_labels)
    pool_prob_sc = np.concatenate(pool_prob_sc)

    auc_roc = roc_auc_score(pool_labels, pool_prob_sc)
    auc_pr = average_precision_score(pool_labels, pool_prob_sc)
    roc_lo, roc_hi = bootstrap_ci(pool_labels, pool_prob_sc, roc_auc_score, rng)
    pr_lo, pr_hi = bootstrap_ci(pool_labels, pool_prob_sc, average_precision_score, rng)

    f1_vals = [r['f1_sc'] for r in fold_rows]
    prec_vals = [r['precision_sc'] for r in fold_rows]
    rec_vals = [r['recall_sc'] for r in fold_rows]

    summary_rows = fold_rows + [{
        'fold': 'mean',
        'f1_sc': float(np.mean(f1_vals)),
        'precision_sc': float(np.mean(prec_vals)),
        'recall_sc': float(np.mean(rec_vals)),
        'std_f1_sc': float(np.std(f1_vals)),
        'auc_roc': float(auc_roc),
        'auc_roc_ci_lower': roc_lo,
        'auc_roc_ci_upper': roc_hi,
        'auc_pr': float(auc_pr),
        'auc_pr_ci_lower': pr_lo,
        'auc_pr_ci_upper': pr_hi,
        'tp': sum(r['tp'] for r in fold_rows),
        'fn': sum(r['fn'] for r in fold_rows),
        'fp': sum(r['fp'] for r in fold_rows),
        'tn': sum(r['tn'] for r in fold_rows),
    }]
    out_df = pd.DataFrame(summary_rows)
    out_df.to_csv(_SUMMARY_PATH, index=False)

    print('\n' + '=' * 65)
    print(f'  5-fold summary')
    print('=' * 65)
    print(f'  mean F1_sc = {np.mean(f1_vals):.3f} ± {np.std(f1_vals):.3f}')
    print(f'  mean P_sc  = {np.mean(prec_vals):.3f}')
    print(f'  mean R_sc  = {np.mean(rec_vals):.3f}')
    print(f'  AUC-ROC    = {auc_roc:.3f}  95% CI [{roc_lo:.3f}, {roc_hi:.3f}]')
    print(f'  AUC-PR     = {auc_pr:.3f}  95% CI [{pr_lo:.3f}, {pr_hi:.3f}]')
    print(f'\n  Saved -> {_SUMMARY_PATH}')
    print(f'  Preds  -> {_PRED_DIR}/fold{{0-4}}_predictions.csv')


if __name__ == '__main__':
    main()
