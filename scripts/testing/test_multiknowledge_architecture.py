#!/usr/bin/env python3
"""
Test script for VetBERT Multi-Modal End-to-End Architecture

This script validates that the multi-modal architecture works correctly
by testing all fusion strategies with sample data.

Author: AI Assistant
Date: October 2025
"""

import torch
import numpy as np
from transformers import AutoTokenizer
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.training.fine_tune_multiknowledge_endtoend import VetBERTMultiKnowledgeClassifier

def test_multiknowledge_architecture():
    """Test all fusion strategies of the multi-knowledge architecture."""
    
    print("🧪 Testing VetBERT Multi-Knowledge Architecture")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Model configuration
    model_path = 'havocy28/VetBERTDx'
    fusion_strategies = ['attention', 'gated', 'cross_modal', 'hierarchical']
    
    # Create sample data
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sample_texts = [
        "The cat shows signs of urinary tract infection with frequent urination",
        "Dog presents with lethargy and decreased appetite, possible systemic condition",
        "Feline patient has blood in urine, consistent with cystitis symptoms"
    ]
    
    # Tokenize sample data
    encoded = tokenizer.batch_encode_plus(
        sample_texts,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    labels = torch.tensor([0, 1, 0]).to(device)  # Binary labels
    
    print(f"📊 Sample batch size: {input_ids.shape[0]}")
    print(f"📊 Sequence length: {input_ids.shape[1]}")
    
    # Test each fusion strategy
    for strategy in fusion_strategies:
        print(f"\n🔬 Testing {strategy} fusion...")
        
        try:
            # Test trainable version
            model = VetBERTMultiKnowledgeClassifier(
                base_model_path=model_path,
                device=device,
                fusion_strategy=strategy,
                fusion_dim=256,
                dropout_rate=0.1,
                freeze_base=False
            )
            
            # Forward pass
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            print(f"   ✅ Forward pass successful")
            print(f"   📊 Output shape: {outputs.logits.shape}")
            print(f"   📊 Loss: {outputs.loss.item():.4f}")
            print(f"   📊 Embedding dim: {outputs.embeddings.shape}")
            print(f"   📊 Syndrome logits: {outputs.syndrome_logits.shape}")
            
            # Test frozen version
            model_frozen = VetBERTMultiKnowledgeClassifier(
                base_model_path=model_path,
                device=device,
                fusion_strategy=strategy,
                fusion_dim=256,
                dropout_rate=0.1,
                freeze_base=True
            )
            
            with torch.no_grad():
                outputs_frozen = model_frozen(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            print(f"   ✅ Frozen version successful")
            print(f"   🔒 Frozen loss: {outputs_frozen.loss.item():.4f}")
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_trainable = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
            
            print(f"   🔢 Total params: {total_params:,}")
            print(f"   🔢 Trainable (normal): {trainable_params:,}")
            print(f"   🔢 Trainable (frozen): {frozen_trainable:,}")
            
            del model, model_frozen
            
        except Exception as e:
            print(f"   ❌ Error testing {strategy}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎯 Architecture validation completed!")
    
    # Test gradient flow
    print(f"\n🔄 Testing gradient flow...")
    try:
        model = VetBERTMultiKnowledgeClassifier(
            base_model_path=model_path,
            device=device,
            fusion_strategy='attention',
            freeze_base=False
        )
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        
        # Check gradients
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        if has_gradients:
            print("   ✅ Gradient flow successful")
        else:
            print("   ❌ No gradients found")
            
    except Exception as e:
        print(f"   ❌ Gradient test failed: {str(e)}")
    
    print(f"\n🏆 Multi-knowledge architecture ready for training!")


if __name__ == '__main__':
    test_multiknowledge_architecture()