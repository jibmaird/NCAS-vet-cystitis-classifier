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

Checkpoints accompanying the paper *"Two-Stage Decomposition of Sporadic Cystitis
Detection in Veterinary Clinical Text."* All checkpoints fine-tune
[VetBERTDx](https://huggingface.co/havocy28/VetBERTDx) (Hur et al., 2020) to detect
Sporadic Cystitis (SC) in free-text veterinary clinical notes.

## Model description

The paper decomposes SC detection into two sub-problems of different difficulty:

- **Stage 1 — UTI identification:** is the note about a urinary tract disorder at all?
  Solved by a frozen, general-purpose VetBERTDx signal with no SC-specific training.
- **Stage 2 — SC vs. Other-UTD:** given a urinary case, is it Sporadic Cystitis
  specifically, or a clinically similar condition (FIC, FLUTD, urolithiasis, recurrent
  cystitis, urethral obstruction)?

The checkpoints below are single-stage classifiers (trained end-to-end on the full,
imbalanced population) that the paper's two-stage pipeline gates behind the Stage-1
signal at inference time (§5.4) — the checkpoint itself is unchanged by gating; gating
is applied as a post-hoc veto using a separately-computed Stage-1 score.

| Architecture | Fusion strategy | Seed | File | Size | Role in paper |
|---|---|---|---|---|---|
| Multi-Knowledge | Gated | 43 | `mk_gated_seed43.pt` | 829 MB | Best-performing configuration overall (§5.4) |
| Multi-Knowledge | Cross-Modal | 42 | `mk_cross_modal_seed42.pt` | 831 MB | Second fusion family, §5.4 |
| Dual-Head | Cross-Attention | 43 | `dh_cross_attention_seed43.pt` | 828 MB | §5.4 |
| Dual-Head | Weighted | 43 | `dh_weighted_seed43.pt` | 829 MB | §5.4 |

Both fusion families combine VetBERTDx's 768-dim CLS embedding with its 39-dim
syndrome-probability vector (§3.2); Multi-Knowledge and Dual-Head differ in how the
two representations are combined (see paper §4.3 for architecture details).

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

**Published.** All four checkpoints and this model card are live at
[huggingface.co/jibmaird/sporadic-cystitis-two-stage](https://huggingface.co/jibmaird/sporadic-cystitis-two-stage),
following institutional review of publishing weights fine-tuned on real clinical
data. The training data itself remains private and is not released here or in the
companion GitHub repository.

Note: the paper's Stage-2-only VetBERT result (§4.2, §5.2) comes from 5-fold
cross-validation, where each fold's model is trained purely to compute out-of-fold
predictions and discarded — there is no single checkpoint corresponding to that result.

## Intended use and limitations

- Trained and evaluated on a single private annotation batch (`ann04062026`, 2021
  notes); no external validation on a different year, clinic, or annotation source.
- Research use only. Not a validated clinical decision support tool — veterinary
  judgement should not be replaced by model output.
- The paper's own error analysis (§6.4) finds the SC gold label itself is "soft": a
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
