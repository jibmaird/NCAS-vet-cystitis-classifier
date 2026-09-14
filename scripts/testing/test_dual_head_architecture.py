#!/usr/bin/env python3
"""
Test script for VetBERT Dual-Head architecture
Quick validation of all fusion strategies
"""

import sys
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Add the project root to path
vet_stream_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(vet_stream_root)

# Import our dual-head model
from scripts.training.fine_tune_vetbert_dual_head import VetBERTDualHead

def test_fusion_strategy(strategy: str, device: torch.device):
    """Test a specific fusion strategy."""
    print(f"\n🔧 Testing {strategy} fusion strategy...")
    
    try:
        # Initialize model
        model = VetBERTDualHead(
            base_model_path='havocy28/VetBERTDx',
            device=device,
            fusion_strategy=strategy
        )
        
        # Create dummy input
        batch_size = 2
        max_length = 512
        input_ids = torch.randint(0, 1000, (batch_size, max_length)).to(device)
        attention_mask = torch.ones(batch_size, max_length).to(device)
        labels = torch.randint(0, 2, (batch_size,)).to(device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        
        # Check outputs
        print(f"  ✅ Forward pass successful")
        print(f"  📊 Logits shape: {outputs.logits.shape}")
        print(f"  📊 Loss: {outputs.loss.item():.4f}")
        print(f"  📊 Syndrome logits shape: {outputs.syndrome_logits.shape}")
        print(f"  📊 Embeddings shape: {outputs.embeddings.shape}")
        
        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  🔢 Total parameters: {total_params:,}")
        print(f"  🔢 Trainable parameters: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    """Test all fusion strategies."""
    print("🚀 Testing VetBERT Dual-Head Architecture")
    print("=" * 60)
    
    # Set device
    device = torch.device('cpu')  # Use CPU for testing
    print(f"📱 Using device: {device}")
    
    # Test all fusion strategies
    strategies = ['concatenation', 'weighted', 'cross_attention', 'multi_task']
    results = {}
    
    for strategy in strategies:
        success = test_fusion_strategy(strategy, device)
        results[strategy] = success
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TESTING SUMMARY")
    print("=" * 60)
    
    successful = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]
    
    print(f"✅ Successful strategies: {successful}")
    if failed:
        print(f"❌ Failed strategies: {failed}")
    else:
        print("🎉 All fusion strategies working correctly!")
    
    print(f"\n🎯 Ready for training with {len(successful)}/{len(strategies)} strategies")

if __name__ == "__main__":
    main()