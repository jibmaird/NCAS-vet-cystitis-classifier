#!/usr/bin/env python3
"""
VetBERT Multi-Knowledge End-to-End Fine-tuning for Binary Cystitis Classification

This script implements a sophisticated approach that combines multiple knowledge sources:
1. VetBERT embeddings (768-dim semantic features)
2. VetBERT syndrome predictions (~50-dim domain knowledge)
3. End-to-end fine-tuning on binary UTD classification data

Architecture:
VetBERT Base → [Embeddings + Syndrome Logits] → Knowledge Fusion → Binary Classification

This approach achieves optimal performance by leveraging both semantic understanding
and domain-specific veterinary knowledge while allowing all components to adapt.

Author: AI Assistant
Date: October 2025
"""

import os
import time
import datetime
import random
from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Subset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    get_linear_schedule_with_warmup,
)
import wandb

# Repo root, independent of the caller's working directory (this file lives
# at <repo_root>/scripts/training/).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


class VetBERTMultiKnowledgeClassifier(nn.Module):
    """
    Multi-knowledge VetBERT classifier that combines different knowledge sources:
    embeddings and syndrome predictions with end-to-end fine-tuning.
    
    Architecture:
    1. VetBERT Base Model → CLS embeddings (768-dim semantic knowledge)
    2. VetBERT Syndrome Head → syndrome logits (~50-dim domain knowledge)  
    3. Knowledge Fusion Network → binary classification
    4. End-to-end training of all components
    """
    
    def __init__(self, 
                 base_model_path: str, 
                 device: torch.device,
                 fusion_strategy: str = "attention",
                 embedding_dim: int = 768,
                 fusion_dim: int = 256,
                 dropout_rate: float = 0.1,
                 freeze_base: bool = False):
        super().__init__()
        
        self.fusion_strategy = fusion_strategy
        self.freeze_base = freeze_base
        
        # Load VetBERT base model (for embeddings)
        self.base_model = AutoModel.from_pretrained(
            base_model_path,
            trust_remote_code=False,
            use_safetensors=True
        )
        
        # Load VetBERT classification model (for syndromes)
        self.syndrome_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            trust_remote_code=False,
            use_safetensors=True
        )
        
        # Get dimensions
        self.embedding_dim = self.base_model.config.hidden_size
        self.num_syndromes = self.syndrome_model.config.num_labels
        
        # Optionally freeze base models
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            for param in self.syndrome_model.parameters():
                param.requires_grad = False
            print(f"🔒 Base models frozen")
        else:
            print(f"🔓 Base models trainable")
        
        # Build fusion network based on strategy
        if fusion_strategy == "attention":
            self._build_attention_fusion(fusion_dim, dropout_rate)
        elif fusion_strategy == "gated":
            self._build_gated_fusion(fusion_dim, dropout_rate)
        elif fusion_strategy == "cross_modal":
            self._build_cross_modal_fusion(fusion_dim, dropout_rate)
        elif fusion_strategy == "hierarchical":
            self._build_hierarchical_fusion(fusion_dim, dropout_rate)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        # Move to device
        self.to(device)
        
        print(f"📊 Multi-knowledge architecture initialized:")
        print(f"   Embeddings: {self.embedding_dim}-dim")
        print(f"   Syndromes: {self.num_syndromes}-dim") 
        print(f"   Fusion: {fusion_strategy}")
        print(f"   Output: 2 classes (UTD binary)")
        
    def _build_attention_fusion(self, fusion_dim: int, dropout_rate: float):
        """Attention-based fusion of embeddings and syndromes."""
        # Project both modalities to common dimension
        self.embedding_projection = nn.Linear(self.embedding_dim, fusion_dim)
        self.syndrome_projection = nn.Linear(self.num_syndromes, fusion_dim)
        
        # Multi-head attention for cross-modal interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, 2)
        )
        
    def _build_gated_fusion(self, fusion_dim: int, dropout_rate: float):
        """Gated fusion with learnable weights."""
        # Project modalities
        self.embedding_projection = nn.Linear(self.embedding_dim, fusion_dim)
        self.syndrome_projection = nn.Linear(self.num_syndromes, fusion_dim)
        
        # Gating mechanism
        self.gate_network = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, 2),  # Gates for embeddings and syndromes
            nn.Softmax(dim=-1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, 2)
        )
        
    def _build_cross_modal_fusion(self, fusion_dim: int, dropout_rate: float):
        """Cross-modal interaction with feature enhancement."""
        # Initial projections
        self.embedding_projection = nn.Linear(self.embedding_dim, fusion_dim)
        self.syndrome_projection = nn.Linear(self.num_syndromes, fusion_dim)
        
        # Cross-modal enhancement layers
        self.emb_to_syn = nn.Linear(fusion_dim, fusion_dim)
        self.syn_to_emb = nn.Linear(fusion_dim, fusion_dim)
        
        # Feature combination
        self.feature_combiner = nn.Sequential(
            nn.Linear(fusion_dim * 4, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, 2)
        )
        
    def _build_hierarchical_fusion(self, fusion_dim: int, dropout_rate: float):
        """Hierarchical fusion with progressive integration."""
        # Level 1: Individual processing
        self.embedding_processor = nn.Sequential(
            nn.Linear(self.embedding_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.syndrome_processor = nn.Sequential(
            nn.Linear(self.num_syndromes, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Level 2: Cross-modal interaction
        self.interaction_layer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Level 3: Final classification
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, 2)
        )
    
    def forward(self, input_ids, attention_mask, labels=None):
        """Multi-modal forward pass."""
        
        # Extract embeddings from base model
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        embeddings = base_outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Extract syndrome predictions
        syndrome_outputs = self.syndrome_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        syndrome_logits = syndrome_outputs.logits
        syndrome_probs = F.softmax(syndrome_logits, dim=-1)
        
        # Apply fusion strategy
        if self.fusion_strategy == "attention":
            binary_logits = self._forward_attention(embeddings, syndrome_probs)
        elif self.fusion_strategy == "gated":
            binary_logits = self._forward_gated(embeddings, syndrome_probs)
        elif self.fusion_strategy == "cross_modal":
            binary_logits = self._forward_cross_modal(embeddings, syndrome_probs)
        elif self.fusion_strategy == "hierarchical":
            binary_logits = self._forward_hierarchical(embeddings, syndrome_probs)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(binary_logits, labels)
        
        return type('ModelOutput', (), {
            'loss': loss,
            'logits': binary_logits,
            'embeddings': embeddings,
            'syndrome_logits': syndrome_logits,
            'syndrome_probs': syndrome_probs
        })
    
    def _forward_attention(self, embeddings: torch.Tensor, syndrome_probs: torch.Tensor) -> torch.Tensor:
        """Attention-based fusion forward pass."""
        # Project to common dimension
        emb_proj = self.embedding_projection(embeddings).unsqueeze(1)  # [B, 1, D]
        syn_proj = self.syndrome_projection(syndrome_probs).unsqueeze(1)  # [B, 1, D]
        
        # Cross-attention
        emb_attended, _ = self.cross_attention(emb_proj, syn_proj, syn_proj)
        syn_attended, _ = self.cross_attention(syn_proj, emb_proj, emb_proj)
        
        # Combine and classify
        combined = torch.cat([emb_attended.squeeze(1), syn_attended.squeeze(1)], dim=-1)
        return self.classifier(combined)
    
    def _forward_gated(self, embeddings: torch.Tensor, syndrome_probs: torch.Tensor) -> torch.Tensor:
        """Gated fusion forward pass."""
        # Project features
        emb_proj = self.embedding_projection(embeddings)
        syn_proj = self.syndrome_projection(syndrome_probs)
        
        # Compute gates
        combined_for_gate = torch.cat([emb_proj, syn_proj], dim=-1)
        gates = self.gate_network(combined_for_gate)
        
        # Apply gating
        gated_features = gates[:, 0:1] * emb_proj + gates[:, 1:2] * syn_proj
        
        return self.classifier(gated_features)
    
    def _forward_cross_modal(self, embeddings: torch.Tensor, syndrome_probs: torch.Tensor) -> torch.Tensor:
        """Cross-modal interaction forward pass."""
        # Project features
        emb_proj = self.embedding_projection(embeddings)
        syn_proj = self.syndrome_projection(syndrome_probs)
        
        # Cross-modal enhancement
        emb_enhanced = emb_proj + self.syn_to_emb(syn_proj)
        syn_enhanced = syn_proj + self.emb_to_syn(emb_proj)
        
        # Combine all features
        all_features = torch.cat([emb_proj, syn_proj, emb_enhanced, syn_enhanced], dim=-1)
        combined_features = self.feature_combiner(all_features)
        
        return self.classifier(combined_features)
    
    def _forward_hierarchical(self, embeddings: torch.Tensor, syndrome_probs: torch.Tensor) -> torch.Tensor:
        """Hierarchical fusion forward pass."""
        # Level 1: Process modalities
        emb_processed = self.embedding_processor(embeddings)
        syn_processed = self.syndrome_processor(syndrome_probs)
        
        # Level 2: Interaction
        combined = torch.cat([emb_processed, syn_processed], dim=-1)
        interaction_features = self.interaction_layer(combined)
        
        # Level 3: Classification
        return self.classifier(interaction_features)
    
    def get_attention_weights(self):
        """Get attention weights for interpretability (if using attention fusion)."""
        if hasattr(self, 'gate_network'):
            # For gated fusion, return the last computed gates
            return getattr(self, '_last_gates', None)
        return None


def find_cystitis_class_index(target_classes: list) -> int:
    """
    Locate the Sporadic Cystitis class within the (casefolded) label-encoder
    classes, regardless of word order ('cystitis sporadic' vs 'sporadic cystitis').
    """
    for i, cls in enumerate(target_classes):
        if 'cystitis' in cls and 'sporadic' in cls:
            return i
    raise ValueError(
        f"No Sporadic Cystitis class found among label encoder classes: {target_classes}"
    )


def map_labels_to_binary(labels: torch.Tensor, cystitis_idx: int) -> torch.Tensor:
    """Maps labels to binary: 0=Sporadic Cystitis (cystitis_idx), 1=everything else"""
    return torch.where(labels == cystitis_idx, torch.tensor(0), torch.tensor(1))


class Config:
    # Project settings
    wandb_project = 'VETBERT_UTI_MultiKnowledge_EndToEnd'
    wandb_name = 'VetBERT_MultiKnowledge_Binary'
    model_path = 'havocy28/VetBERTDx'

    # Training settings
    max_length = 512
    warmup_steps = 10
    gradient_clip = 1.0
    seed = 42
    epochs = 40

    # Model architecture
    fusion_strategy = "attention"  # "attention", "gated", "cross_modal", "hierarchical"
    fusion_dim = 256
    dropout_rate = 0.1
    freeze_base = False  # Whether to freeze VetBERT base models

    # Hyperparameters
    learning_rate = 2e-5  # Lower for fine-tuning
    batch_size = 16       # Conservative for memory

    # Paths
    checkpoint_dir = os.path.join(_REPO_ROOT, 'checkpoints')
    resume_from_checkpoint = None

    # Optional separate validation file (overrides 80/20 internal split)
    val_file = None
    # Optional fixed checkpoint name saved under checkpoint_dir after each best epoch
    checkpoint_name = None

    # Diagnostic: save a checkpoint + dev metrics for every epoch up to
    # diagnostic_max_epoch, to compare checkpoint-selection criteria post-hoc.
    diagnostic_dir = None
    diagnostic_max_epoch = 15


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(file_path: str) -> Tuple[pd.DataFrame, preprocessing.LabelEncoder]:
    """Load and prepare the training data."""
    df = pd.read_csv(file_path)
    df['Label'] = df['Label'].str.split('-', n=2).str[:2].str.join('-').str.casefold()
    df.loc[df['Label'].str.contains('urolithiasis'), 'Label'] = 'urolithiasis'
    
    le = preprocessing.LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    return df, le


def tokenize_data(texts: np.ndarray, tokenizer: AutoTokenizer, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize text data."""
    texts = [str(text) for text in texts]
    encoded = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return encoded['input_ids'], encoded['attention_mask']


def create_dataloaders(dataset: TensorDataset, train_idx: np.ndarray, val_idx: np.ndarray, 
                      batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    return (
        DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size),
        DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
    )


def train_epoch(model: VetBERTMultiKnowledgeClassifier, dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer, scheduler, device: torch.device,
                cystitis_idx: int) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch
        
        # Map to binary labels
        binary_labels = map_labels_to_binary(labels, cystitis_idx).to(device)
        
        model.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=binary_labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.gradient_clip)
        optimizer.step()
        scheduler.step()
        
        wandb.log({
            'train_batch_loss': loss.item(),
            'learning_rate': scheduler.get_last_lr()[0],
            'fusion_strategy': Config.fusion_strategy
        })
    
    return total_loss / len(dataloader)


def evaluate(model: VetBERTMultiKnowledgeClassifier, dataloader: DataLoader,
            device: torch.device, cystitis_idx: int,
            class_names: list = ('sporadic cystitis', 'other')) -> Dict:
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch
            
            binary_labels = map_labels_to_binary(labels, cystitis_idx).to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=binary_labels
            )
            
            total_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(binary_labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_scores = f1_score(all_labels, all_preds, average=None)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    class_names = list(class_names)

    metrics = {
        'val_loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'fusion_strategy': Config.fusion_strategy
    }
    
    # Add per-class metrics
    for idx, class_name in enumerate(class_names):
        if idx < len(f1_scores):
            metrics[f'f1_{class_name}'] = f1_scores[idx]
        if idx < len(conf_matrix) and conf_matrix[idx].sum() > 0:
            metrics[f'accuracy_{class_name}'] = conf_matrix[idx, idx] / conf_matrix[idx].sum()

    return metrics


def main():
    """Main training function."""
    
    # Initialize wandb
    wandb.init(
        project=Config.wandb_project, 
        name=f"{Config.wandb_name}_{Config.fusion_strategy}",
        resume='allow'
    )
    
    # Update config
    wandb.config.update({
        'fusion_strategy': Config.fusion_strategy,
        'fusion_dim': Config.fusion_dim,
        'dropout_rate': Config.dropout_rate,
        'freeze_base': Config.freeze_base,
        'learning_rate': Config.learning_rate,
        'batch_size': Config.batch_size,
        'epochs': Config.epochs,
        'max_length': Config.max_length
    })
    
    print(f"🚀 Starting VetBERT Multi-Knowledge End-to-End Fine-tuning")
    print(f"🏗️  Fusion strategy: {Config.fusion_strategy}")
    print(f"🔒 Base models frozen: {Config.freeze_base}")
    
    # Set device and seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(Config.seed)
    print(f"📱 Using device: {device}")
    
    # Parse command-line arguments for train file
    import argparse
    parser = argparse.ArgumentParser(description="VetBERT Multi-Knowledge End-to-End Fine-tuning")
    parser.add_argument('--train_file', type=str, default=os.path.join(os.path.dirname(__file__), '../../data/clean/refined_model/cystitis_refined_train_binary.csv'),
                        help='Path to the training CSV file (default: cystitis_refined_train_binary.csv)')
    args, _ = parser.parse_known_args()

    # Load and prepare data
    data_path = args.train_file
    df, label_encoder = prepare_data(data_path)
    target_classes = label_encoder.classes_.tolist()
    cystitis_idx = find_cystitis_class_index(target_classes)
    other_classes = [c for i, c in enumerate(target_classes) if i != cystitis_idx]
    class_names = [
        target_classes[cystitis_idx],
        other_classes[0] if len(other_classes) == 1 else 'other',
    ]

    print(f"📊 Dataset: {len(df)} samples")
    print(f"📊 Classes: {target_classes}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_path)
    
    # Tokenize data
    input_ids, attention_masks = tokenize_data(df.text.values, tokenizer, Config.max_length)
    labels = torch.tensor(df.Label.values)
    
    # Create dataset
    dataset = TensorDataset(input_ids, attention_masks, labels)

    if Config.val_file:
        # Use all training data; load explicit validation set with the same encoder
        val_df_raw = pd.read_csv(Config.val_file)
        val_df_raw['Label'] = (
            val_df_raw['Label']
            .str.split('-', n=2).str[:2].str.join('-')
            .str.casefold()
        )
        val_df_raw['Label'] = label_encoder.transform(val_df_raw['Label'])
        val_input_ids, val_attention_masks = tokenize_data(
            val_df_raw.text.values, tokenizer, Config.max_length
        )
        val_labels = torch.tensor(val_df_raw.Label.values)
        val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)

        train_dataloader = DataLoader(
            dataset, sampler=RandomSampler(dataset), batch_size=Config.batch_size
        )
        val_dataloader = DataLoader(
            val_dataset, sampler=SequentialSampler(val_dataset), batch_size=Config.batch_size
        )
        print(f"📊 Train samples: {len(dataset)} (full training set)")
        print(f"📊 Validation samples: {len(val_dataset)} (from --val_file)")
    else:
        # Fall back to internal 80/20 stratified split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=Config.seed)
        train_idx, val_idx = next(sss.split(input_ids, labels))
        print(f"📊 Train samples: {len(train_idx)}")
        print(f"📊 Validation samples: {len(val_idx)}")
        train_dataloader, val_dataloader = create_dataloaders(
            dataset, train_idx, val_idx, Config.batch_size
        )
    
    # Initialize model
    model = VetBERTMultiKnowledgeClassifier(
        base_model_path=Config.model_path,
        device=device,
        fusion_strategy=Config.fusion_strategy,
        fusion_dim=Config.fusion_dim,
        dropout_rate=Config.dropout_rate,
        freeze_base=Config.freeze_base
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🔢 Total parameters: {total_params:,}")
    print(f"🔢 Trainable parameters: {trainable_params:,}")
    print(f"🔢 Frozen parameters: {total_params - trainable_params:,}")
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=Config.warmup_steps,
        num_training_steps=len(train_dataloader) * Config.epochs
    )
    
    # Training loop
    best_f1 = 0
    
    print(f"\n🏋️ Starting training for {Config.epochs} epochs...")
    
    for epoch in range(Config.epochs):
        print(f"\n📅 Epoch {epoch+1}/{Config.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device, cystitis_idx)
        
        # Evaluate
        metrics = evaluate(model, val_dataloader, device, cystitis_idx, class_names)
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            **metrics
        })
        
        print(f"📈 Train Loss: {train_loss:.4f}")
        print(f"📈 Val Loss: {metrics['val_loss']:.4f}")
        print(f"📈 Accuracy: {metrics['accuracy']:.4f}")
        print(f"📈 F1 Score: {metrics['macro_f1']:.4f}")
        
        # Save best model
        if metrics['macro_f1'] > best_f1:
            best_f1 = metrics['macro_f1']
            torch.save(model.state_dict(), f'{wandb.run.dir}/best_model.pt')
            print(f"💾 New best F1: {best_f1:.4f} - Model saved!")
            # Also save to a predictable fixed path when --checkpoint_name is given
            if Config.checkpoint_name:
                ckpt_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), Config.checkpoint_dir
                )
                os.makedirs(ckpt_dir, exist_ok=True)
                fixed_path = os.path.join(ckpt_dir, Config.checkpoint_name)
                torch.save(model.state_dict(), fixed_path)
                print(f"💾 Also saved to: {fixed_path}")

        # Diagnostic: save this epoch's checkpoint + dev metrics, regardless
        # of whether it's a new best, so selection criteria can be compared
        # post-hoc against test performance.
        if Config.diagnostic_dir and (epoch + 1) <= Config.diagnostic_max_epoch:
            os.makedirs(Config.diagnostic_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(Config.diagnostic_dir, f'epoch_{epoch+1:02d}.pt'))
            f1_sc = metrics.get(f'f1_{class_names[0]}')
            f1_other = metrics.get(f'f1_{class_names[1]}')
            metrics_path = os.path.join(Config.diagnostic_dir, 'epoch_metrics.csv')
            write_header = not os.path.exists(metrics_path)
            with open(metrics_path, 'a') as f:
                if write_header:
                    f.write('epoch,train_loss,val_loss,accuracy,macro_f1,f1_sc,f1_other\n')
                f.write(f'{epoch+1},{train_loss:.6f},{metrics["val_loss"]:.6f},'
                        f'{metrics["accuracy"]:.6f},{metrics["macro_f1"]:.6f},'
                        f'{f1_sc:.6f},{f1_other:.6f}\n')

    print(f"\n🎯 Training completed!")
    print(f"🏆 Best F1 Score: {best_f1:.4f}")
    
    return best_f1


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="VetBERT Multi-Knowledge End-to-End Fine-tuning")
    parser.add_argument(
        '--train_file', type=str,
        default=os.path.join(os.path.dirname(__file__), '../../data/clean/refined_model/cystitis_refined_train_binary.csv'),
        help='Path to training CSV (columns: text, Label)'
    )
    parser.add_argument(
        '--val_file', type=str, default=None,
        help='Path to separate validation CSV (columns: text, Label). '
             'If provided, overrides the internal 80/20 split.'
    )
    parser.add_argument(
        '--fusion_strategy', type=str,
        choices=['attention', 'gated', 'cross_modal', 'hierarchical'],
        help='Knowledge fusion strategy to use'
    )
    parser.add_argument(
        '--freeze_base', action='store_true',
        help='Freeze VetBERT base models'
    )
    parser.add_argument(
        '--checkpoint_name', type=str, default=None,
        help='Fixed filename (under checkpoints/) to overwrite with each new best model. '
             'Useful for downstream prediction scripts that need a predictable path.'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for model init / training (default: 42)'
    )
    parser.add_argument(
        '--diagnostic_dir', type=str, default=None,
        help='If set, save a checkpoint + dev metrics for every epoch up to '
             '--diagnostic_max_epoch under this directory (for post-hoc '
             'checkpoint-selection-criterion analysis).'
    )
    parser.add_argument(
        '--diagnostic_max_epoch', type=int, default=15,
        help='Max epoch (1-indexed) to save diagnostic checkpoints for (default: 15)'
    )
    args, unknown = parser.parse_known_args()

    # Apply args to Config
    if args.fusion_strategy:
        Config.fusion_strategy = args.fusion_strategy
    if args.freeze_base:
        Config.freeze_base = True
    if args.val_file:
        Config.val_file = args.val_file
    if args.checkpoint_name:
        Config.checkpoint_name = args.checkpoint_name
    Config.seed = args.seed
    Config.diagnostic_dir = args.diagnostic_dir
    Config.diagnostic_max_epoch = args.diagnostic_max_epoch

    main()