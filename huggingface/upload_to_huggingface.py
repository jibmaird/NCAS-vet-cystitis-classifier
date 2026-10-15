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

The four paper checkpoints have already been published at
https://huggingface.co/jibmaird/sporadic-cystitis-two-stage following
institutional review of releasing weights trained on real clinical data (see
huggingface/README.md, "Status"). Re-running this script is safe -- it will
just overwrite the existing files with a new commit -- but is normally only
needed if you're publishing your own checkpoints trained with this repo's
scripts.
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
    """Load and print basic info about a checkpoint.

    scripts/training/fine_tune_vetbert_dual_head.py and
    fine_tune_multiknowledge_endtoend.py both save checkpoints via
    `torch.save(model.state_dict(), ...)` -- a raw state_dict (an OrderedDict
    of tensors), not a wrapper dict with training metadata. There is no
    'fusion_strategy'/'seed'/'base_model' field stored in the file itself;
    that information lives in the checkpoint's filename and the model card.
    """
    print(f"\nVerifying checkpoint: {path}")
    state_dict = torch.load(path, map_location="cpu", weights_only=True)

    if not isinstance(state_dict, dict) or not state_dict:
        sys.exit(f"ERROR: {path} is not a non-empty state_dict.")

    tensors = {k: v for k, v in state_dict.items() if torch.is_tensor(v)}
    if len(tensors) != len(state_dict):
        sys.exit(f"ERROR: {path} contains non-tensor values; expected a raw "
                 "model state_dict as produced by torch.save(model.state_dict(), ...).")

    n_params = sum(v.numel() for v in tensors.values())
    top_level_modules = sorted({k.split('.')[0] for k in tensors.keys()})

    info = {
        "n_keys": len(tensors),
        "n_params": n_params,
        "top_level_modules": top_level_modules,
    }

    print(f"  Keys             : {info['n_keys']}")
    print(f"  Parameters       : {n_params:,}")
    print(f"  Top-level modules: {top_level_modules}")
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
