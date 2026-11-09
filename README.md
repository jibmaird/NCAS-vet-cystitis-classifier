# NCAS Vet Cystitis Classifier

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow)](https://huggingface.co/jibmaird/sporadic-cystitis-two-stage)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

Reference implementation for two-stage detection of Sporadic Cystitis (SC) in
free-text veterinary clinical notes: a frozen-model topic gate that identifies
urinary-tract-disorder (UTD) notes, a classifier trained only on UTD notes that
discriminates SC from clinically similar UTD conditions, and the end-to-end
classifiers the two-stage pipeline is compared against (an SVM on frozen
VetBERTDx features, fine-tuned VetBERTDx and Multi-Knowledge Gated fusion, plus
the other fusion variants of the paper's exploratory sweep).

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

- **Stage 1 -- UTD identification (topic recognition):** is this note about a
  urinary tract disorder at all? Solved near-perfectly by a frozen,
  general-purpose model (VetBERTDx) with no SC-specific training
  (AUC-ROC = 0.984).
- **Stage 2 -- SC vs. Other-UTD (within-topic discrimination):** given a
  urinary case, is it Sporadic Cystitis specifically, or a clinically similar
  condition (FIC, FLUTD, urolithiasis, recurrent cystitis, urethral
  obstruction)? Tractable in isolation (CV F1 = 0.860), but unstable when
  folded into a single end-to-end classifier that must also reject the many
  non-UTD notes found at natural prevalence.

The paper compares three systems on the test set (§2.6–2.7, §3.3–3.4):

- **System A -- end-to-end classifiers:** one model labels every note as SC or
  Other, with no gate.
- **System B -- end-to-end + UTD filter:** System A's SC predictions are kept
  only for notes that pass Stage 1; nothing is retrained.
- **System C -- two-stage pipeline:** only notes that pass Stage 1 reach the
  Stage-2 classifier (fine-tuned VetBERTDx trained on UTD notes only; the
  models from its five cross-validation folds are averaged).

The end-to-end classifiers are unstable across random seeds; the filter
(System B) removes most of that instability, and the two-stage pipeline
(System C) is the most stable (§3.4).

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

Requires Python 3.11+. scikit-learn is pinned to 1.7.1, the version used for the
paper's Stage-2 ML results (newer versions shift a few of them in the third
decimal). Models are loaded from the Hugging Face Hub at runtime
(`havocy28/VetBERTDx`), so an internet connection is needed the first time
each script runs (weights are then cached locally).

Run the pipeline end to end on synthetic data (generate -> prepare labels ->
Stage-1 gate -> Stage-2 CV -> System C -> an end-to-end classifier -> stats):

```bash
# 1. Generate fabricated raw data (~200/100/100 train/dev/test rows)
python tools/generate_synthetic_data.py --seed 0

# 2. Derive the two label schemes:
#    - binary_SC:  Sporadic Cystitis vs. Other (end-to-end classifiers, Systems A and B, §2.6)
#    - binary:     Cystitis Sporadic vs. UTD Other, counting UTD_Journey as UTD Other (an
#                  earlier label scheme, not used for the paper's results; Stage 2, §2.5,
#                  reads its UTD notes directly from the raw training file, without UTD_Journey)
python scripts/data_preparation/prepare_ann04062026_SC_data.py
python scripts/data_preparation/prepare_ann04062026_data.py

# 3. Stage 1 -- cache frozen VetBERTDx P(urinary tract disorder) for dev+test
python scripts/evaluation/sc_cascade_step1_extract_utd_probs.py

# 4. Stage 2 -- extract frozen features, then run the 5-fold CV suite
python scripts/evaluation/sc_stage2_extract_features.py
python scripts/evaluation/sc_stage2_ml_cv.py
python scripts/evaluation/sc_stage2_vetbert_cv.py   # fine-tunes per fold; add --save_models DIR to keep the fold models
# --summary_only rebuilds its summary CSV from the saved fold predictions, without retraining

# 4b. System C (two-stage pipeline, §2.7): the same CV once per training seed (the fold
#     split stays fixed), with every fold model also scoring dev and test; then gate ->
#     average of the five fold models, per seed and as a soft vote across seeds
for SEED in 42 43 44; do
  python scripts/evaluation/sc_stage2_vetbert_cv.py --seed $SEED \
    --out_dir results/evaluation/ann04062026_SC/true_pipeline/stage2_vetbert_seed$SEED \
    --predict data/ann04062026/processed/dev_binary_SC.csv data/ann04062026/processed/test_binary_SC.csv
done
python scripts/evaluation/sc_true_pipeline_eval.py

# 5. End-to-end classifiers (System A): traditional ML, all 12 configs, then their
#    per-note test predictions (used by the statistics in step 7)
python scripts/training/train_evaluate_ml_ann04062026.py \
  --train_file data/ann04062026/processed/train_binary_SC.csv \
  --dev_file   data/ann04062026/processed/dev_binary_SC.csv \
  --test_file  data/ann04062026/processed/test_binary_SC.csv \
  --output_dir results/evaluation/ann04062026_SC --models_dir models/ml_ann04062026_SC
python scripts/prediction/predict_ml_ann04062026_SC.py

# 6. An end-to-end DL classifier (System A; example: Multi-Knowledge Gated fusion)
python scripts/training/fine_tune_multiknowledge_endtoend.py \
  --fusion_strategy gated \
  --train_file data/ann04062026/processed/train_binary_SC.csv \
  --val_file   data/ann04062026/processed/dev_binary_SC.csv \
  --checkpoint_name demo_gated.pt --epochs 5

python scripts/evaluation/evaluate_multiknowledge_ann04062026_SC.py \
  --model_path checkpoints/demo_gated.pt --fusion_strategy gated \
  --test_file data/ann04062026/processed/test_binary_SC.csv

# 7. Statistics on the end-to-end (System A) predictions: per-model bootstrap CIs
#    and pairwise McNemar tests
python scripts/evaluation/sc_bootstrap_ci.py
python scripts/evaluation/sc_mcnemar_test.py --all_pairs   # without --all_pairs it compares four
                                                           # fusion models, which this demo does not train
# The paper's reported CIs and Tables 5-6 (System A, System B = the UTD filter on
# System A's predictions, and System C) come from sc_shared_bootstrap_ci.py and
# sc_systems_abc_summary.py, which need the seed-42/43/44 and ML prediction files
# from the full runs.
```

`fine_tune_vetbert_dual_head.py` and `fine_tune_multiknowledge_endtoend.py`
both accept `--fusion_strategy` (`concatenation|weighted|cross_attention|
multi_task` for Dual-Head; `attention|gated|cross_modal|hierarchical` for
Multi-Knowledge), `--seed`, `--epochs`, and `--checkpoint_name`;
`fine_tune_on_cystitis_refined_binary.py` (fine-tuned VetBERTDx) accepts `--seed`
too. The paper's multi-seed runs (§3.3–3.4) are produced by calling the same
script three times with different `--seed`/`--checkpoint_name` values, not by
any code branch.

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
│   ├── training/                    The architecture families of the exploratory sweep:
│   │                                  traditional ML, fine-tuned VetBERTDx (VetBERTBinary),
│   │                                  Dual-Head and Multi-Knowledge fusion (4 strategies
│   │                                  each); System A uses the SVM, fine-tuned VetBERTDx
│   │                                  and MK Gated
│   ├── prediction/                  Per-note predictions for the ML baselines, and System C
│   │                                  inference on new notes (predict_two_stage.py)
│   ├── evaluation/                  Stage-1 gate, Stage-2 CV, Systems B and C,
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
| §2.1–2.3 Data | `scripts/data_preparation/prepare_ann04062026_SC_data.py` (`prepare_ann04062026_data.py` builds the earlier `binary` label scheme, not used for the paper's results) |
| §2.4 Stage-1 gate | `scripts/evaluation/sc_cascade_step1_extract_utd_probs.py`, `sc_cascade_step1_utd_gate_leakfree.py`; Table 3 (gate operating points, pass rates, F1 CIs) comes from `sc_two_stage_evaluation.py` |
| §2.5 Stage-2 CV | `scripts/evaluation/sc_stage2_extract_features.py`, `sc_stage2_ml_cv.py`, `sc_stage2_vetbert_cv.py` (Table 4 and Appendix A2: fold-mean F1, macro-F1 and recall, F1 and AUC-ROC CIs) |
| §2.6 End-to-end classifiers (System A) | `scripts/training/train_evaluate_ml_ann04062026.py` (SVM), `fine_tune_on_cystitis_refined_binary.py` (fine-tuned VetBERTDx), `fine_tune_multiknowledge_endtoend.py` (MK Gated; other strategies for the Appendix A1 sweep), `fine_tune_vetbert_dual_head.py` (Appendix A1 sweep), and the matching `scripts/evaluation/evaluate_*_ann04062026_SC.py` |
| §2.7 UTD filter (System B) and two-stage pipeline (System C) | System B: the filter is applied to System A's predictions in `scripts/evaluation/sc_shared_bootstrap_ci.py` and `sc_systems_abc_summary.py` (`sc_cascade_gate_all_seeds.py` does the same for an earlier four-architecture comparison); System C: `sc_stage2_vetbert_cv.py --seed --predict`, then `sc_true_pipeline_eval.py`; `scripts/prediction/predict_two_stage.py` runs System C on new notes with the published models |
| §2.8 Stats | `scripts/evaluation/sc_shared_bootstrap_ci.py` (the paper's reported CIs: one shared set of bootstrap resamples for all systems), `sc_bootstrap_ci.py`, `sc_mcnemar_test.py` |
| §3.3–3.4 Tables 5–6 | `scripts/evaluation/sc_systems_abc_summary.py` (F1 with shared CIs, false positives on non-UTD notes, McNemar tests between systems, overlap with the reviewed errors) |
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

The System C models are published in the `system_c_seed42/` folder of
[huggingface.co/jibmaird/sporadic-cystitis-two-stage](https://huggingface.co/jibmaird/sporadic-cystitis-two-stage):
the five Stage-2 fold models for training seed 42, with a `config.json` recording the gate
threshold and the decision rule. To label your own notes with System C:

```bash
python scripts/prediction/predict_two_stage.py --input notes.csv --output predictions.csv \
  --hf_repo jibmaird/sporadic-cystitis-two-stage
```

The input CSV needs a `text` column. The output adds P(UTD), the gate decision, the averaged
SC probability and the label. On the paper's test set these models reproduce the paper's
System C seed-42 results exactly (F1(SC) 0.756, Table 6).

The same repository also holds four end-to-end classifiers:
- Multi-Knowledge Gated (seed 43; a System A classifier in the paper, §2.6);
- Multi-Knowledge Cross-Modal (seed 42; exploratory sweep, Appendix A1);
- Dual-Head Cross-Attention and Dual-Head Weighted (seed 43; from an earlier multi-seed
  comparison that the revised paper no longer reports).

See [`huggingface/README.md`](huggingface/README.md) for the model card, file sizes, and a
loading snippet.

Note: the paper's Stage-2 cross-validation result (§2.5, §3.2) and System C's seeds 43 and 44
come from fold models that were not saved. `sc_stage2_vetbert_cv.py --save_models DIR` saves
the fold models of a new run.

---

## Limitations

- This repository is a **methods reference implementation**, not a validated
  clinical decision-support tool. All clinical decisions must remain with
  qualified veterinary professionals.
- Real performance numbers are reported in the companion paper, computed
  against the real, private annotation batch -- **numbers produced by this
  repository's synthetic data are illustrative only** and are not comparable
  to the paper's reported results.
- Only System C's seed-42 fold models are published; the fold models behind the paper's
  Stage-2 cross-validation result and System C's seeds 43 and 44 were not saved. GPU
  training is not exactly reproducible, so retraining the same seed can give slightly
  different models (the paper's limitations report a 0.035 difference in test F1(SC) for
  seed 42). The four end-to-end checkpoints were trained on all 1,200 UTD-enriched training
  notes; the paper's System B filters the predictions of such classifiers with the Stage-1
  gate.
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
