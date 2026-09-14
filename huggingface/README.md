---
license: mit
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

| Architecture | Fusion strategy | Role in paper |
|---|---|---|
| Multi-Knowledge | Gated | Best-performing configuration overall (§5.4) |
| Multi-Knowledge | Cross-Modal | Second fusion family, §5.4 |
| Dual-Head | Cross-Attention | §5.4 |
| Dual-Head | Weighted | §5.4 |

Both fusion families combine VetBERTDx's 768-dim CLS embedding with its 39-dim
syndrome-probability vector (§3.2); Multi-Knowledge and Dual-Head differ in how the
two representations are combined (see paper §4.3 for architecture details).

## Status

**Checkpoints are not yet uploaded.** Release is pending institutional review of
publishing weights fine-tuned on real clinical data (the training data itself is
private and is not released here or in the companion GitHub repository). This model
card is a placeholder prepared ahead of that review; filenames, checkpoint format, and
per-architecture download instructions will be filled in once weights are approved for
release.

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

See [`CITATION.cff`](https://github.com/jibmaird/sporadic-cystitis-two-stage/blob/main/CITATION.cff)
in the companion repository. Please also cite VetBERT/VetBERTDx (Hur et al., 2020),
which this pipeline builds on throughout.

## Repository

Training, evaluation, and reproduction code (including a synthetic-data quickstart) is
available at:
[https://github.com/jibmaird/sporadic-cystitis-two-stage](https://github.com/jibmaird/sporadic-cystitis-two-stage)

## License

MIT — see [LICENSE](https://github.com/jibmaird/sporadic-cystitis-two-stage/blob/main/LICENSE).
This does not cover the base VetBERTDx model (loaded at runtime from the Hugging Face
Hub, not redistributed here) or the private annotation data used in the paper.
