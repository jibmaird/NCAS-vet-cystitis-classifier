#!/usr/bin/env python3
"""
System C (two-stage pipeline) inference: label free-text notes as Sporadic Cystitis (SC) or Other.

Stage 1 (UTD gate): frozen VetBERTDx (havocy28/VetBERTDx) gives P(urinary tract disorder); a note passes the gate
when P(UTD) >= the gate threshold (default: gate_thr1 from the paper, set on the development set).
Stage 2: the five cross-validation fold models of the Stage-2 classifier (VetBERTDx + SC head, trained only on UTD
notes; saved by sc_stage2_vetbert_cv.py --save_models) score the notes that pass the gate; their SC probabilities
are averaged and a note is labelled SC when the average is at least 0.5. Notes that do not pass the gate are Other.

Models come either from a local directory (stage2_fold0.pt ... stage2_fold4.pt) or from the Hugging Face Hub.

Usage:
    python scripts/prediction/predict_two_stage.py --input notes.csv --output predictions.csv --models_dir DIR
    python scripts/prediction/predict_two_stage.py --input notes.csv --output predictions.csv \
        --hf_repo jibmaird/sporadic-cystitis-two-stage --hf_subfolder system_c_seed42

The input CSV needs a `text` column; any other columns are copied to the output, which adds p_utd, passes_gate,
prob_sc (the fold-average SC probability; empty for notes that do not pass the gate) and predicted_label.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import AutoTokenizer

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, '../..'))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'scripts/evaluation'))
from src.models.vetbert_hybrid_extractor import VetBERTHybridFeatureExtractor  # noqa: E402
from sc_stage2_vetbert_cv import (VetBERTBinary, tokenize, MODEL_PATH, BATCH_SIZE, N_FOLDS,  # noqa: E402
                                  SC_CLASS_IDX, POSITIVE_LABEL, NEGATIVE_LABEL)

GATE_THR1 = 0.00028190302   # second-smallest P(UTD) among the development-set SC notes (paper: 0.000282)
SC_THRESHOLD = 0.5


def fold_paths(args):
    if args.models_dir:
        return [os.path.join(args.models_dir, f'stage2_fold{k}.pt') for k in range(N_FOLDS)]
    from huggingface_hub import hf_hub_download
    return [hf_hub_download(args.hf_repo, f'stage2_fold{k}.pt', subfolder=args.hf_subfolder) for k in range(N_FOLDS)]


def main():
    ap = argparse.ArgumentParser(description='System C: UTD gate, then the averaged Stage-2 fold models')
    ap.add_argument('--input', required=True, help='CSV with a `text` column')
    ap.add_argument('--output', required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--models_dir', help='directory with stage2_fold0.pt ... stage2_fold4.pt')
    src.add_argument('--hf_repo', help='Hugging Face model repo holding the fold models')
    ap.add_argument('--hf_subfolder', default='system_c_seed42')
    ap.add_argument('--gate_threshold', type=float, default=GATE_THR1)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    texts = df['text'].astype(str).tolist()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Stage 1: P(UTD) from frozen VetBERTDx
    extractor = VetBERTHybridFeatureExtractor(model_path=MODEL_PATH)
    utd_idx = extractor.syndrome_labels.index('urinary tract disorder')
    p_utd = extractor.extract_syndrome_probabilities(texts)[:, utd_idx]
    del extractor
    passes = p_utd >= args.gate_threshold
    print(f'{int(passes.sum())} of {len(df)} notes pass the gate (threshold {args.gate_threshold:g})')

    # Stage 2: average P(SC) of the five fold models over the notes that pass the gate
    prob_sc = np.full(len(df), np.nan)
    idx = np.where(passes)[0]
    if len(idx):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        ids, masks = tokenize([texts[i] for i in idx], tokenizer)
        loader = DataLoader(TensorDataset(ids, masks), sampler=SequentialSampler(ids), batch_size=BATCH_SIZE)
        fold_probs = []
        for path in fold_paths(args):
            model = VetBERTBinary(MODEL_PATH, device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.eval()
            out = []
            with torch.no_grad():
                for b_ids, b_mask in loader:
                    logits = model(b_ids.to(device), b_mask.to(device)).logits
                    out.append(torch.softmax(logits, dim=-1)[:, SC_CLASS_IDX].cpu().numpy())
            fold_probs.append(np.concatenate(out))
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        prob_sc[idx] = np.mean(fold_probs, axis=0)

    df['p_utd'] = p_utd
    df['passes_gate'] = passes
    df['prob_sc'] = prob_sc
    df['predicted_label'] = np.where(passes & (np.nan_to_num(prob_sc) >= SC_THRESHOLD), POSITIVE_LABEL, NEGATIVE_LABEL)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f'{int((df.predicted_label == POSITIVE_LABEL).sum())} notes labelled SC -> {args.output}')


if __name__ == '__main__':
    main()
