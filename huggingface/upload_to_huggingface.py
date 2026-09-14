#!/usr/bin/env python3
"""
upload_to_huggingface.py -- Upload trained checkpoint(s) to the Hugging Face Hub.

Steps performed:
  1. Verifies each checkpoint file is valid and prints its metadata
  2. Creates the Hugging Face repository (if it does not already exist)
  3. Uploads each checkpoint under its own filename
  4. Uploads the model card (huggingface/README.md -> Hub README)

Prerequisites
-------------
    pip install huggingface_hub

Then authenticate with a WRITE-scoped token:
    hf auth login

Usage
-----
    # Single checkpoint
    python huggingface/upload_to_huggingface.py \
        --model checkpoints/mk_gated_seed43.pt --name mk_gated.pt \
        [--repo jibmaird/sporadic-cystitis-two-stage] [--private]

    # Multiple checkpoints in one call (repeat --model/--name pairs)
    python huggingface/upload_to_huggingface.py \
        --model checkpoints/mk_gated_seed43.pt          --name mk_gated.pt \
        --model checkpoints/mk_crossmodal_seed42.pt      --name mk_cross_modal.pt \
        --model checkpoints/dh_crossattention_seed43.pt  --name dh_cross_attention.pt \
        --model checkpoints/dh_weighted_seed43.pt        --name dh_weighted.pt

NOTE: this script is prepared ahead of release. Do not run it against real
fine-tuned checkpoints until institutional review of releasing weights trained
on real clinical data has cleared (see huggingface/README.md, "Status").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    sys.exit("Run: pip install huggingface_hub")

DEFAULT_REPO = "jibmaird/sporadic-cystitis-two-stage"
MODEL_CARD_PATH = Path(__file__).parent / "README.md"


def verify_checkpoint(path: str) -> dict:
    """Load and print checkpoint metadata without building the full model."""
    print(f"\nVerifying checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        sys.exit(f"ERROR: {path} does not contain 'model_state_dict'. "
                 "Expected a full checkpoint dict as produced by "
                 "scripts/training/fine_tune_multiknowledge_endtoend.py or "
                 "scripts/training/fine_tune_vetbert_dual_head.py.")

    n_params = sum(p.numel() for p in
                   [v for k, v in ckpt["model_state_dict"].items() if v.dim() > 0])

    info = {
        "fusion_strategy": ckpt.get("fusion_strategy", "?"),
        "seed":            ckpt.get("seed", "?"),
        "base_model":      ckpt.get("base_model", "havocy28/VetBERTDx"),
        "n_params":        n_params,
    }

    print(f"  Fusion strategy : {info['fusion_strategy']}")
    print(f"  Seed            : {info['seed']}")
    print(f"  Base model      : {info['base_model']}")
    print(f"  Parameters      : {n_params:,}")
    return info


def upload(checkpoints: list[tuple[str, str]], repo_id: str, private: bool) -> None:
    for path, _ in checkpoints:
        verify_checkpoint(path)

    api = HfApi()

    print(f"\nCreating / verifying repository: {repo_id}")
    create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    print("  Repository OK.")

    for path, name in checkpoints:
        size_mb = Path(path).stat().st_size / 1e6
        print(f"\nUploading {name} ({size_mb:.0f} MB) ...")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {name} checkpoint",
        )
        print(f"  ✓ {name}")

    if MODEL_CARD_PATH.exists():
        print("\nUploading model card (huggingface/README.md) ...")
        api.upload_file(
            path_or_fileobj=str(MODEL_CARD_PATH),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload model card",
        )
        print("  ✓ README.md (model card)")
    else:
        print(f"\nWARNING: model card not found at {MODEL_CARD_PATH}. Skipping.")

    print(f"\nDone! Model available at: https://huggingface.co/{repo_id}")


def main() -> None:
    p = argparse.ArgumentParser(description="Upload checkpoint(s) to the Hugging Face Hub")
    p.add_argument("--model", action="append", required=True, dest="models",
                   help="Path to a .pt checkpoint. Repeat --model/--name pairs to upload several.")
    p.add_argument("--name", action="append", required=True, dest="names",
                   help="Filename to use in the Hub repo for the matching --model.")
    p.add_argument("--repo", default=DEFAULT_REPO,
                   help=f"Hugging Face repo ID (default: {DEFAULT_REPO})")
    p.add_argument("--private", action="store_true",
                   help="Create a private repository (default: public)")
    args = p.parse_args()

    if len(args.models) != len(args.names):
        sys.exit("ERROR: --model and --name must be given the same number of times, in order.")

    print("=" * 60)
    print("Sporadic Cystitis Two-Stage Detector -- Hugging Face Upload")
    print("=" * 60)
    print(f"Repository : {args.repo}")
    print(f"Private    : {args.private}")
    print(f"Checkpoints: {len(args.models)}")

    upload(list(zip(args.models, args.names)), args.repo, args.private)


if __name__ == "__main__":
    main()
