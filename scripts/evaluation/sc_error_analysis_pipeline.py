#!/usr/bin/env python3
"""
Error analysis for the two-stage pipeline (MK-Gated seed43, gate_drop1).

For each of the 154 gate-passing test records:
  - Assigns pipeline outcome (TP / FP / FN / TN)
  - Splits text into sections using clinical section headers
  - Runs frozen VetBERTDx on each section to get P(UTD) as a relevance signal
  - Outputs one row per section to a long-format CSV for domain-expert review

The segment P(UTD) score indicates which part of the note the UTI classifier
finds most relevant — useful for identifying the evidence behind each prediction.

Outputs:
  results/evaluation/ann04062026_SC/error_analysis_pipeline_record_level.csv
  results/evaluation/ann04062026_SC/error_analysis_pipeline_segments.csv

Usage:
    python scripts/evaluation/sc_error_analysis_pipeline.py
    (runs on CPU; expect ~10 minutes for VetBERTDx per-segment inference)
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
from sc_phase1_utils import POSITIVE_LABEL, NEGATIVE_LABEL, PROJECT_ROOT

# ── Paths ─────────────────────────────────────────────────────────────────────
_PRED_PATH    = os.path.join(PROJECT_ROOT,
    'results/evaluation/ann04062026_SC_seed43/predictions_gated_SC.csv')
_GATE_PATH    = os.path.join(PROJECT_ROOT,
    'results/evaluation/ann04062026_SC/utd_gate_probs_SC.csv')
_TEST_PATH    = os.path.join(PROJECT_ROOT,
    'data/ann04062026/test-all-syndromes.csv')
_OUT_DIR      = os.path.join(PROJECT_ROOT, 'results/evaluation/ann04062026_SC')
_OUT_RECORDS  = os.path.join(_OUT_DIR, 'error_analysis_pipeline_record_level.csv')
_OUT_SEGMENTS = os.path.join(_OUT_DIR, 'error_analysis_pipeline_segments.csv')

GATE_THRESHOLD = 0.000282   # gate_drop1
VETBERTDX_MODEL = 'havocy28/VetBERTDx'
UTD_SYNDROME    = 'urinary tract disorder'

# Section headers that reliably split clinical notes into meaningful units
_SECTION_RE = re.compile(
    r'(?<!\w)'                        # not mid-word
    r'(Reason|History|Examination|Assessment|Plan|Laboratory|Treatment'
    r'|Appointment Notes|Anaesthesia|Medications|Urinalysis|Objective'
    r'|TREATMENTS|History-Update|Sedation|Dental|Bloods|Findings'
    r'|Client Communication|Nurse|Client Present)'
    r':\s*',
    re.IGNORECASE
)


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Split clinical note text into (header, content) pairs using section headers.
    Unlabelled leading text gets header 'Preamble'.
    Returns list of (header, content) with empty content stripped.
    """
    parts = _SECTION_RE.split(text)
    # split returns: [pre, header1, content1, header2, content2, ...]
    sections = []
    if parts[0].strip():
        sections.append(('Preamble', parts[0].strip()))
    for i in range(1, len(parts) - 1, 2):
        header  = parts[i].strip().title()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ''
        if content:
            sections.append((header, content))
    # Fallback: if no headers found, return whole text as one segment
    if not sections:
        sections = [('Full text', text.strip())]
    return sections


def load_vetbertdx(device):
    print(f'Loading VetBERTDx from {VETBERTDX_MODEL} ...')
    tokenizer = AutoTokenizer.from_pretrained(VETBERTDX_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(VETBERTDX_MODEL)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    utd_idx = list(model.config.id2label.values()).index(UTD_SYNDROME)
    print(f'  UTD index: {utd_idx}  |  device: {device}')
    return tokenizer, model, utd_idx


def p_utd(texts: list[str], tokenizer, model, utd_idx, device, batch_size=16) -> np.ndarray:
    """Return P(UTD) for each text, processing in batches."""
    probs = []
    for i in tqdm(range(0, len(texts), batch_size), desc='VetBERTDx inference', leave=False):
        batch = texts[i:i + batch_size]
        enc   = tokenizer(batch, return_tensors='pt', padding=True,
                          truncation=True, max_length=512)
        enc   = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        p = torch.softmax(logits, dim=-1)[:, utd_idx].cpu().numpy()
        probs.extend(p.tolist())
    return np.array(probs)


def outcome_label(gold, pred):
    pos = (gold == POSITIVE_LABEL)
    hat = (pred == POSITIVE_LABEL)
    if pos and hat:  return 'TP'
    if not pos and hat:  return 'FP'
    if pos and not hat:  return 'FN'
    return 'TN'


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load predictions and gate probs ──────────────────────────────────────
    preds    = pd.read_csv(_PRED_PATH)
    gate_all = pd.read_csv(_GATE_PATH)
    test_syn = pd.read_csv(_TEST_PATH)    # for consult_table_id join
    gate_tst = gate_all[gate_all['split'] == 'test'].reset_index(drop=True)

    # Apply gate: keep only passing records
    passing_mask = gate_tst['p_urinary_tract_disorder'] >= GATE_THRESHOLD
    gate_passing = gate_tst[passing_mask].reset_index(drop=True)
    print(f'\nGate-passing records: {len(gate_passing)} / {len(gate_tst)}')

    # Merge predictions with gate probs (on text)
    merged = preds.merge(
        gate_passing[['text', 'p_urinary_tract_disorder']],
        on='text', how='inner'
    )
    assert len(merged) == len(gate_passing), \
        f'Merge mismatch: {len(merged)} vs {len(gate_passing)}'

    # Pipeline predicted label: gate-passing records keep Stage-2 prediction
    # (gate-failing records are excluded from this analysis)
    merged['predicted_pipeline'] = merged['predicted_label']
    merged['outcome'] = merged.apply(
        lambda r: outcome_label(r['Label'], r['predicted_pipeline']), axis=1)

    # Join consult_table_id for a stable record identifier
    id_map = test_syn[['text', 'consult_table_id']].drop_duplicates()
    merged = merged.merge(id_map, on='text', how='left')

    outcome_order = {'TP': 0, 'FP': 1, 'FN': 2, 'TN': 3}
    merged['_sort'] = merged['outcome'].map(outcome_order)
    merged = merged.sort_values(['_sort', 'prob_sporadic_cystitis'],
                                ascending=[True, False]).reset_index(drop=True)

    print('\nOutcome counts:')
    print(merged['outcome'].value_counts().to_string())

    # ── Record-level CSV ──────────────────────────────────────────────────────
    record_df = merged[[
        'consult_table_id', 'Label', 'predicted_pipeline', 'outcome',
        'prob_sporadic_cystitis', 'p_urinary_tract_disorder'
    ]].rename(columns={
        'Label': 'gold_label',
        'predicted_pipeline': 'predicted_label',
        'prob_sporadic_cystitis': 'p_sc',
        'p_urinary_tract_disorder': 'p_utd_record',
    })
    os.makedirs(_OUT_DIR, exist_ok=True)
    record_df.to_csv(_OUT_RECORDS, index=False)
    print(f'\nSaved record-level CSV -> {_OUT_RECORDS}')

    # ── Load VetBERTDx for per-segment inference ──────────────────────────────
    tokenizer, model, utd_idx = load_vetbertdx(device)

    # ── Segment-level CSV ─────────────────────────────────────────────────────
    seg_rows = []
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc='Segmenting records'):
        sections = split_into_sections(row['text'])
        seg_texts = [content for _, content in sections]
        seg_p_utd = p_utd(seg_texts, tokenizer, model, utd_idx, device)

        # Rank segments within this record by P(UTD) descending
        ranks = (-seg_p_utd).argsort().argsort() + 1   # 1 = most UTD-relevant

        for seg_idx, ((header, content), p_val, rank) in enumerate(
                zip(sections, seg_p_utd, ranks)):
            seg_rows.append({
                'consult_table_id':  row['consult_table_id'],
                'gold_label':        row['Label'],
                'predicted_label':   row['predicted_pipeline'],
                'outcome':           row['outcome'],
                'p_sc':              round(float(row['prob_sporadic_cystitis']), 4),
                'p_utd_record':      round(float(row['p_urinary_tract_disorder']), 6),
                'segment_idx':       seg_idx,
                'n_segments':        len(sections),
                'section_header':    header,
                'segment_text':      content,
                'p_utd_segment':     round(float(p_val), 4),
                'segment_utd_rank':  int(rank),
            })

    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(_OUT_SEGMENTS, index=False)
    print(f'Saved segment-level CSV -> {_OUT_SEGMENTS}')
    print(f'  Total segments: {len(seg_df)}')
    print(f'  Segments per record: median={seg_df.groupby("consult_table_id").size().median():.0f}')


if __name__ == '__main__':
    main()
