---
license: apache-2.0
language:
- en
tags:
- clinical-nlp
- veterinary
- bert
- binary-classification
- animal-health
pipeline_tag: text-classification
library_name: pytorch
base_model: havocy28/VetBERTDx
---

# Sporadic Cystitis Two-Stage Detector

Checkpoints accompanying the paper *"Sporadic Cystitis Detection in Veterinary Clinical Text."* All checkpoints fine-tune
[VetBERTDx](https://huggingface.co/havocy28/VetBERTDx) (Hur et al., 2020) to detect
Sporadic Cystitis (SC) in free-text veterinary clinical notes.

## Model description

The paper decomposes SC detection into two sub-problems of different difficulty:

- **Stage 1 — UTD identification:** is the note about a urinary tract disorder at all?
  Solved by a frozen, general-purpose VetBERTDx signal with no SC-specific training.
- **Stage 2 — SC vs. Other-UTD:** given a urinary case, is it Sporadic Cystitis
  specifically, or a clinically similar condition (FIC, FLUTD, urolithiasis, recurrent
  cystitis, urethral obstruction)?

### System C: two-stage pipeline (`system_c_seed42/`)

The folder `system_c_seed42/` holds the paper's System C models for training seed 42:
- the five Stage-2 fold models (`stage2_fold0.pt` to `stage2_fold4.pt`, about 434 MB each);
- `config.json`, which records the gate threshold, the decision rule and the training settings.

Each fold model is VetBERTDx with a new SC head, fine-tuned only on UTD notes (SC vs
Other-UTD) in one fold of a 5-fold cross-validation over the 804 UTD training notes.
System C was the most stable of the paper's three systems across training runs (§3.4).

To label a note, System C:
1. computes P(UTD) with the frozen, released VetBERTDx, and lets the note through if
   P(UTD) ≥ 0.00028190302 (the paper's gate_thr1, set on the development set);
2. averages the SC probabilities of the five fold models for the notes that pass;
3. labels a note SC when it passes and the average is at least 0.5, and Other otherwise.

On the paper's test set (500 notes at natural prevalence, 21 SC), this gives F1(SC) 0.756
(95% CI 0.622–0.884; 17 TP, 7 FP, 4 FN; Table 6 of the paper). The paper also reports seeds
43 and 44 and their soft vote; those fold models are not published.

`scripts/prediction/predict_two_stage.py` in the companion repository runs the whole pipeline:

```bash
python scripts/prediction/predict_two_stage.py --input notes.csv --output predictions.csv \
    --hf_repo jibmaird/sporadic-cystitis-two-stage --hf_subfolder system_c_seed42
```

### End-to-end classifiers

The four checkpoints below are end-to-end classifiers, of the kind used in the paper's
System A: trained on all 1,200 training notes to predict SC vs Other. In the paper's System B,
the SC predictions of such classifiers are filtered with the Stage-1 gate (§2.7, §3.4) — the
checkpoint itself is unchanged; the filter is a post-hoc veto using a separately computed
Stage-1 score.

| Architecture | Fusion strategy | Seed | File | Size | Role in paper |
|---|---|---|---|---|---|
| Multi-Knowledge | Gated | 43 | `mk_gated_seed43.pt` | 829 MB | System A classifier (§2.6); best System B run (§3.4) |
| Multi-Knowledge | Cross-Modal | 42 | `mk_cross_modal_seed42.pt` | 831 MB | Exploratory sweep (Appendix A1) |
| Dual-Head | Cross-Attention | 43 | `dh_cross_attention_seed43.pt` | 828 MB | Earlier multi-seed comparison; the paper reports only seed 42 (Appendix A1) |
| Dual-Head | Weighted | 43 | `dh_weighted_seed43.pt` | 829 MB | Earlier multi-seed comparison; the paper reports only seed 42 (Appendix A1) |

Both fusion families combine VetBERTDx's 768-dim CLS embedding with its 39-dim
syndrome-probability vector; Multi-Knowledge and Dual-Head differ in how the two
representations are combined (see paper §2.6 for MK Gated and Appendix A1 for the others).

Each file is a raw PyTorch `state_dict()` (not a wrapper dict) -- load it with the
matching model class from the companion repository:

```python
import torch
from huggingface_hub import hf_hub_download
# from scripts/training/fine_tune_multiknowledge_endtoend.py (companion repo)
from fine_tune_multiknowledge_endtoend import VetBERTMultiKnowledgeClassifier

path = hf_hub_download("jibmaird/sporadic-cystitis-two-stage", "mk_gated_seed43.pt")
model = VetBERTMultiKnowledgeClassifier(
    base_model_path="havocy28/VetBERTDx", device=torch.device("cpu"), fusion_strategy="gated"
)
model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
model.eval()
```

For Dual-Head checkpoints, use `VetBERTDualHead` from
`fine_tune_vetbert_dual_head.py` with the matching `fusion_strategy`
(`cross_attention` or `weighted`) instead.

## Status

**Published** at
[huggingface.co/jibmaird/sporadic-cystitis-two-stage](https://huggingface.co/jibmaird/sporadic-cystitis-two-stage).
The four end-to-end checkpoints were published following institutional review of publishing
weights fine-tuned on real clinical data; the System C models, trained on the same data, were
added on 9 November 2026. The training data itself remains private and is not released here
or in the companion GitHub repository.

Note: the paper's Stage-2 cross-validation result (§2.5, §3.2) and System C's seeds 43 and 44
come from fold models that were not saved, so no checkpoint for them is published.

## Intended use and limitations

- Trained and evaluated on a single private annotation batch (`ann04062026`, 2021
  notes); no external validation on a different year, data source or practice management system.
- Research use only. Not a validated clinical decision support tool — veterinary
  judgement should not be replaced by model output.
- The paper's own error analysis (§4) finds the SC gold label itself is "soft": a
  meaningful share of positive labels rest on an antibiotic prescription plus a written
  differential diagnosis, without confirmatory cytology. Treat model predictions with
  the same caveat as the labels they were trained on.

## Citation

See [`CITATION.cff`](https://github.com/jibmaird/NCAS-vet-cystitis-classifier/blob/main/CITATION.cff)
in the companion repository. Please also cite VetBERT/VetBERTDx (Hur et al., 2020),
which this pipeline builds on throughout.

## Repository

Training, evaluation, and reproduction code (including a synthetic-data quickstart) is
available at:
[https://github.com/jibmaird/NCAS-vet-cystitis-classifier](https://github.com/jibmaird/NCAS-vet-cystitis-classifier)

## License

Apache 2.0 — see [LICENSE](https://github.com/jibmaird/NCAS-vet-cystitis-classifier/blob/main/LICENSE).
This does not cover the base VetBERTDx model (loaded at runtime from the Hugging Face
Hub, not redistributed here) or the private annotation data used in the paper.
