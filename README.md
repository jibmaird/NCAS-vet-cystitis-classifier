# NCAS Vet Cystitis Classifier

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow)](https://huggingface.co/jibmaird/sporadic-cystitis-two-stage)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

Reference implementation for two-stage detection of Sporadic Cystitis (SC) in
free-text veterinary clinical notes: a frozen-model topic gate that identifies
urinary-tract-disorder (UTD) notes, a cross-validated classifier that
discriminates SC from clinically similar UTD conditions within that gated
population, and the single-stage baselines (traditional ML and four
fine-tuned fusion architectures) the two-stage pipeline is compared against.

This is the companion code for:

> *Sporadic Cystitis Detection in Veterinary Clinical Text* [paper link -- to be completed once published]

Related work: [NCAS-GP-indication-classifier](https://github.com/jibmaird/NCAS-GP-indication-classifier)
(SNOMED CT coding of antimicrobial-prescribing indications, the analogous
GP-NAPS task).

---

## Overview

Detecting SC in a full, unfiltered clinical population conflates two
qualitatively different problems: at natural prevalence (~4%), topic
recognition (is this note even about a urinary case?) and within-topic
discrimination (which urinary condition is it?) have very different
difficulty, but a single end-to-end classifier has to solve both at once.
This repository decomposes them:

- **Stage 1 -- UTI identification (topic recognition):** is this note about a
  urinary tract disorder at all? Solved near-perfectly by a frozen,
  general-purpose model (VetBERTDx) with no SC-specific training
  (AUC-ROC = 0.984).
- **Stage 2 -- SC vs. Other-UTD (within-topic discrimination):** given a
  urinary case, is it Sporadic Cystitis specifically, or a clinically similar
  condition (FIC, FLUTD, urolithiasis, recurrent cystitis, urethral
  obstruction)? Tractable in isolation (CV F1 = 0.860), but unstable when
  folded into a single end-to-end classifier trained on the full, highly
  imbalanced population.

Explicitly gating Stage 2 behind Stage 1 at inference time removes most of
that instability without retraining anything (§3.4 of the paper).

**This repository does not include the real annotation data.** The paper's
dataset is private veterinary clinical data (a single annotation batch,
`ann04062026`) and cannot be redistributed. A synthetic data generator
(`tools/generate_synthetic_data.py`) produces a structurally analogous
stand-in so every code path can be run end-to-end -- see
[`data/ann04062026/README.md`](data/ann04062026/README.md) for what's
synthetic, why, and how to point the same scripts at real data if you have
access to it.

---

## Quick start

```bash
git clone https://github.com/jibmaird/NCAS-vet-cystitis-classifier.git
cd NCAS-vet-cystitis-classifier
conda env create -f environment.yml && conda activate sc-two-stage
# or: pip install -r requirements.txt
```

Requires Python 3.11+. Models are loaded from the Hugging Face Hub at runtime
(`havocy28/VetBERTDx`), so an internet connection is needed the first time
each script runs (weights are then cached locally).

Run the pipeline end to end on synthetic data (generate -> prepare labels ->
Stage-1 gate -> Stage-2 CV -> a single-stage baseline -> two-stage stats):

```bash
# 1. Generate fabricated raw data (~200/100/100 train/dev/test rows)
python tools/generate_synthetic_data.py --seed 0

# 2. Derive the two label schemes used throughout the paper:
#    - binary_SC:  Sporadic Cystitis vs. Other (single-stage baselines, §2.6)
#    - binary:     Cystitis Sporadic vs. UTD Other, is_utd records only (Stage 2, §2.5)
python scripts/data_preparation/prepare_ann04062026_SC_data.py
python scripts/data_preparation/prepare_ann04062026_data.py

# 3. Stage 1 -- cache frozen VetBERTDx P(urinary tract disorder) for dev+test
python scripts/evaluation/sc_cascade_step1_extract_utd_probs.py

# 4. Stage 2 -- extract frozen features, then run the 5-fold CV suite
python scripts/evaluation/sc_stage2_extract_features.py
python scripts/evaluation/sc_stage2_ml_cv.py
python scripts/evaluation/sc_stage2_vetbert_cv.py   # fine-tunes per fold; no checkpoint is saved (see Limitations)

# 5. A single-stage baseline, end to end (traditional ML, all 12 configs)
python scripts/training/train_evaluate_ml_ann04062026.py \
  --train_file data/ann04062026/processed/train_binary_SC.csv \
  --dev_file   data/ann04062026/processed/dev_binary_SC.csv \
  --test_file  data/ann04062026/processed/test_binary_SC.csv

# 6. A single-stage DL baseline (example: Multi-Knowledge, Gated fusion)
python scripts/training/fine_tune_multiknowledge_endtoend.py \
  --fusion_strategy gated \
  --train_file data/ann04062026/processed/train_binary_SC.csv \
  --val_file   data/ann04062026/processed/dev_binary_SC.csv \
  --checkpoint_name demo_gated.pt --epochs 5

python scripts/evaluation/evaluate_multiknowledge_ann04062026_SC.py \
  --model_path checkpoints/demo_gated.pt --fusion_strategy gated \
  --test_file data/ann04062026/processed/test_binary_SC.csv

# 7. Two-stage evaluation, stats
python scripts/evaluation/sc_bootstrap_ci.py
python scripts/evaluation/sc_mcnemar_test.py
# The paper's reported CIs come from sc_shared_bootstrap_ci.py, which needs the
# seed-42/43/44 and ML prediction files from the full runs.
```

`fine_tune_vetbert_dual_head.py` and `fine_tune_multiknowledge_endtoend.py`
both accept `--fusion_strategy` (`concatenation|weighted|cross_attention|
multi_task` for Dual-Head; `attention|gated|cross_modal|hierarchical` for
Multi-Knowledge), `--seed`, `--epochs`, and `--checkpoint_name` -- the paper's
multi-seed runs (§3.4) are produced by calling the same script three times
with different `--seed`/`--checkpoint_name` values, not by any code branch.

The three fine-tuning scripts log to [wandb](https://wandb.ai) during
training. Without (or not wanting) a wandb account, run with
`WANDB_MODE=disabled` prefixed -- checkpoints are still saved locally either
way.

**Scores from synthetic data are meaningless** -- the synthetic classes use
deliberately distinct vocabulary and will look trivially separable. They
confirm the code runs, nothing more.

---

## Repository structure

```
NCAS-vet-cystitis-classifier/
│
├── src/models/                      Frozen VetBERTDx wrapper + hybrid feature
│                                     extractor (Stage-1 gate, Stage-2 ML baselines)
│
├── scripts/
│   ├── data_preparation/            Raw annotation CSVs -> (text, Label) training files
│   ├── training/                    The 4 architecture families trained in the paper:
│   │                                  traditional ML, VetBERTBinary, Dual-Head fusion
│   │                                  (4 strategies), Multi-Knowledge fusion (4 strategies)
│   ├── prediction/                  Per-record prediction generation for the ML baselines
│   ├── evaluation/                  Stage-1 gate, Stage-2 CV, two-stage gating/evaluation,
│   │                                  bootstrap CIs, McNemar tests, error analysis
│   └── testing/                     Architecture smoke tests (no data required)
│
├── tools/generate_synthetic_data.py Fabricates the 3 raw CSVs the pipeline expects
│
├── data/ann04062026/                Synthetic data lives here (see its README)
│
├── huggingface/                     Model card + upload script for the Hugging Face
│                                     Hub release (see "Pretrained models" below)
│
├── environment.yml / requirements.txt
├── CITATION.cff
├── LICENSE                          Apache 2.0
└── .gitignore
```

Every script under `scripts/` is a thin wrapper that can be run directly with
`python scripts/.../some_script.py --help`; none require installing this
repo as a package.

### Which script maps to which paper section

| Paper section | Script(s) |
|---|---|
| §2.1–2.3 Data | `scripts/data_preparation/prepare_ann04062026{_SC,}_data.py` |
| §2.4 Stage-1 gate | `scripts/evaluation/sc_cascade_step1_extract_utd_probs.py`, `sc_cascade_step1_utd_gate_leakfree.py` |
| §2.5 Stage-2 CV | `scripts/evaluation/sc_stage2_extract_features.py`, `sc_stage2_ml_cv.py`, `sc_stage2_vetbert_cv.py` |
| §2.6 Single-stage baselines | `scripts/training/train_evaluate_ml_ann04062026.py`, `fine_tune_on_cystitis_refined_binary.py`, `fine_tune_vetbert_dual_head.py`, `fine_tune_multiknowledge_endtoend.py`, and the matching `scripts/evaluation/evaluate_*_ann04062026_SC.py` |
| §2.7 Two-stage pipeline | `scripts/evaluation/sc_cascade_gate_all_seeds.py`, `sc_two_stage_evaluation.py` |
| §2.8 Stats | `scripts/evaluation/sc_shared_bootstrap_ci.py` (the paper's reported CIs: one shared set of bootstrap resamples for all systems), `sc_bootstrap_ci.py`, `sc_mcnemar_test.py` |
| §4 Error analysis | `scripts/evaluation/sc_error_analysis.py`, `sc_error_analysis_pipeline.py`, `sc_error_overlap_gated.py`, `sc_error_review_spreadsheet{,_v2}.py` |

A handful of older/exploratory scripts from the original research codebase
(alternative architectures never reported in the paper, hyperparameter-sweep
tooling, dataset versions whose source files no longer exist) were left out
of this release as not relevant to reproducing the paper's results.

---

## Using your own data

The real `ann04062026` annotation batch is not publicly available. If you
have access to an equivalent dataset, every script accepts explicit
`--train_file`/`--val_file`/`--test_file` (or similar) CSV paths in the same
schema the synthetic generator produces -- see
[`data/ann04062026/README.md`](data/ann04062026/README.md) for the exact
column format and which columns are actually read.

---

## Pretrained models

Checkpoints for the paper's four best-performing single-stage architectures
(the ones fed into the two-stage pipeline evaluation, §3.4) are published at
[huggingface.co/jibmaird/sporadic-cystitis-two-stage](https://huggingface.co/jibmaird/sporadic-cystitis-two-stage):
Multi-Knowledge Gated (seed 43), Multi-Knowledge Cross-Modal (seed 42),
Dual-Head Cross-Attention (seed 43), and Dual-Head Weighted (seed 43). See
[`huggingface/README.md`](huggingface/README.md) for the model card, file
sizes, and a loading snippet.

Note: the paper's Stage-2-only VetBERT result (§2.5, §3.2) comes from 5-fold
cross-validation, where each fold's model is trained purely to compute
out-of-fold predictions and discarded -- there is currently no single
checkpoint corresponding to that result (see Limitations below).

---

## Limitations

- This repository is a **methods reference implementation**, not a validated
  clinical decision-support tool. All clinical decisions must remain with
  qualified veterinary professionals.
- Real performance numbers are reported in the companion paper, computed
  against the real, private annotation batch -- **numbers produced by this
  repository's synthetic data are illustrative only** and are not comparable
  to the paper's reported results.
- The paper's Stage-2-only VetBERT result (§2.5, §3.2) comes from 5-fold
  cross-validation; each fold's model is trained purely to compute
  out-of-fold predictions and discarded, so there is no purpose-built
  Stage-2-only checkpoint, on Hugging Face or otherwise (paper §4,
  limitations). The four published checkpoints are single-stage models
  (trained end-to-end on the full, oversampled population) that the
  two-stage pipeline gates post hoc at inference time.
- All results are from a single private annotation batch (`ann04062026`,
  2021 notes); no external validation on a different year, data
  source or practice management system.
- The paper's own error analysis (§4) finds the SC gold label itself is
  "soft": a meaningful share of positive labels rest on an antibiotic
  prescription plus a written differential diagnosis, without confirmatory
  cytology.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ncas_vet_cystitis_classifier,
  title   = {Sporadic Cystitis Detection in Veterinary Clinical Text},
  author  = {Martinez Iraola, David and others},
  journal = {[Journal -- to be completed]},
  year    = {2026},
  note    = {Under preparation}
}
```

See [`CITATION.cff`](CITATION.cff) for the machine-readable version. Please
also cite VetBERT/VetBERTDx (Hur et al., 2020), which this pipeline builds on
throughout.

---

## License

Apache 2.0 -- see [LICENSE](LICENSE). This does not cover the base VetBERTDx
model itself (loaded at runtime from the Hugging Face Hub, not redistributed
here) or the private annotation data used in the paper.
