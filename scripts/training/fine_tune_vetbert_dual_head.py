#!/usr/bin/env python3
"""
Enhanced VetBERT Fine-tuning with Dual-Head Architecture

This script implements advanced fine-tuning approaches that leverage both:
1. VetBERT embeddings (semantic understanding)
2. VetBERT syndrome predictions (domain knowledge)

Four architectural approaches:
1. Simple Concatenation
2. Weighted Fusion
3. Cross-Attention Fusion
4. Multi-Task Learning
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
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Subset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    AutoModel,
)
import wandb
from torch.cuda.amp import autocast, GradScaler

# Repo root, independent of the caller's working directory (this file lives
# at <repo_root>/scripts/training/).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


class VetBERTDualHead(nn.Module):
    """Enhanced VetBERT with both embedding and syndrome classification heads."""
    
    def __init__(self, base_model_path: str, device: torch.device, fusion_strategy: str = "weighted"):
        super().__init__()
        self.fusion_strategy = fusion_strategy
        
        # Load the base VetBERT model
        self.base_model = AutoModel.from_pretrained(
            base_model_path,
            trust_remote_code=False,
            use_safetensors=True
        )
        
        # Load the syndrome classification model (same base but with classification head)
        self.syndrome_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            trust_remote_code=False,
            use_safetensors=True
        )
        
        # Get dimensions
        hidden_size = self.base_model.config.hidden_size
        num_syndromes = self.syndrome_model.config.num_labels  # Should be ~39 syndromes
        
        # Architecture components based on fusion strategy
        if fusion_strategy == "concatenation":
            self._build_concatenation_head(hidden_size, num_syndromes)
        elif fusion_strategy == "weighted":
            self._build_weighted_fusion_head(hidden_size, num_syndromes)
        elif fusion_strategy == "cross_attention":
            self._build_cross_attention_head(hidden_size, num_syndromes)
        elif fusion_strategy == "multi_task":
            self._build_multi_task_head(hidden_size, num_syndromes)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        self.to(device)
    
    def _build_concatenation_head(self, hidden_size: int, num_syndromes: int):
        """Simple concatenation approach."""
        concat_size = hidden_size + num_syndromes
        self.classifier = nn.Sequential(
            nn.Linear(concat_size, concat_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(concat_size // 2, concat_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(concat_size // 4, 2)
        )
    
    def _build_weighted_fusion_head(self, hidden_size: int, num_syndromes: int):
        """Learnable weighted fusion of embeddings and syndromes."""
        # Projection layers to common dimension
        fusion_dim = 256
        self.embedding_projection = nn.Linear(hidden_size, fusion_dim)
        self.syndrome_projection = nn.Linear(num_syndromes, fusion_dim)
        
        # Attention weights for fusion
        self.fusion_attention = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, 2),  # 2 weights: one for embeddings, one for syndromes
            nn.Softmax(dim=-1)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, 2)
        )
    
    def _build_cross_attention_head(self, hidden_size: int, num_syndromes: int):
        """Cross-attention between embeddings and syndromes."""
        self.attention_dim = 128
        
        # Transform embeddings and syndromes to same dimension
        self.embedding_transform = nn.Linear(hidden_size, self.attention_dim)
        self.syndrome_transform = nn.Linear(num_syndromes, self.attention_dim)
        
        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.attention_dim * 2, self.attention_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.attention_dim, 2)
        )
    
    def _build_multi_task_head(self, hidden_size: int, num_syndromes: int):
        """Multi-task learning with shared representations - ultra memory efficient."""
        # Ultra-reduced shared dimension for maximum memory efficiency
        shared_dim = 32  # Further reduced from 64
        
        # Minimal shared representation layer
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_size, shared_dim),
            nn.ReLU(),
        )
        
        # Task-specific heads (minimal size)
        self.syndrome_head = nn.Linear(shared_dim, num_syndromes)
        self.utd_head = nn.Linear(shared_dim, 2)
        
        # Minimal fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(shared_dim + num_syndromes, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, 2)
        )
    
    def forward(self, input_ids, attention_mask, labels=None, syndrome_labels=None):
        """Forward pass with different fusion strategies."""
        
        # Get embeddings from base model
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        embeddings = base_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Get syndrome predictions
        syndrome_outputs = self.syndrome_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        syndrome_logits = syndrome_outputs.logits
        syndrome_probs = F.softmax(syndrome_logits, dim=-1)
        
        # Apply fusion strategy
        if self.fusion_strategy == "concatenation":
            logits = self._forward_concatenation(embeddings, syndrome_probs)
            loss = self._compute_simple_loss(logits, labels)
            
        elif self.fusion_strategy == "weighted":
            logits = self._forward_weighted_fusion(embeddings, syndrome_probs)
            loss = self._compute_simple_loss(logits, labels)
            
        elif self.fusion_strategy == "cross_attention":
            logits = self._forward_cross_attention(embeddings, syndrome_probs)
            loss = self._compute_simple_loss(logits, labels)
            
        elif self.fusion_strategy == "multi_task":
            logits, syndrome_pred_logits = self._forward_multi_task(embeddings)
            loss = self._compute_multi_task_loss(logits, syndrome_pred_logits, labels, syndrome_labels)
        
        if labels is not None:
            return type('ModelOutput', (), {
                'loss': loss,
                'logits': logits,
                'syndrome_logits': syndrome_logits,
                'embeddings': embeddings
            })
        
        return type('ModelOutput', (), {
            'logits': logits,
            'syndrome_logits': syndrome_logits,
            'embeddings': embeddings
        })
    
    def _forward_concatenation(self, embeddings: torch.Tensor, syndrome_probs: torch.Tensor) -> torch.Tensor:
        """Simple concatenation forward pass."""
        combined = torch.cat([embeddings, syndrome_probs], dim=-1)
        return self.classifier(combined)
    
    def _forward_weighted_fusion(self, embeddings: torch.Tensor, syndrome_probs: torch.Tensor) -> torch.Tensor:
        """Weighted fusion forward pass."""
        # Project to common dimension
        emb_proj = self.embedding_projection(embeddings)
        syn_proj = self.syndrome_projection(syndrome_probs)
        
        # Compute attention weights
        combined_for_attention = torch.cat([emb_proj, syn_proj], dim=-1)
        attention_weights = self.fusion_attention(combined_for_attention)
        
        # Apply weighted fusion
        fused = attention_weights[:, 0:1] * emb_proj + attention_weights[:, 1:2] * syn_proj
        
        return self.classifier(fused)
    
    def _forward_cross_attention(self, embeddings: torch.Tensor, syndrome_probs: torch.Tensor) -> torch.Tensor:
        """Cross-attention forward pass."""
        # Transform to attention dimension
        emb_transformed = self.embedding_transform(embeddings).unsqueeze(1)  # [batch, 1, attention_dim]
        syn_transformed = self.syndrome_transform(syndrome_probs).unsqueeze(1)  # [batch, 1, attention_dim]
        
        # Cross-attention: embeddings attend to syndromes and vice versa
        emb_attended, _ = self.cross_attention(emb_transformed, syn_transformed, syn_transformed)
        syn_attended, _ = self.cross_attention(syn_transformed, emb_transformed, emb_transformed)
        
        # Combine attended representations
        combined = torch.cat([emb_attended.squeeze(1), syn_attended.squeeze(1)], dim=-1)
        
        return self.classifier(combined)
    
    def _forward_multi_task(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-task learning forward pass."""
        # Shared representation
        shared_repr = self.shared_layer(embeddings)
        
        # Generate syndrome predictions from shared representation
        syndrome_pred_logits = self.syndrome_head(shared_repr)
        syndrome_pred_probs = F.softmax(syndrome_pred_logits, dim=-1)
        
        # Fuse shared representation with syndrome predictions
        combined = torch.cat([shared_repr, syndrome_pred_probs], dim=-1)
        utd_logits = self.fusion_layer(combined)
        
        return utd_logits, syndrome_pred_logits
    
    def _compute_simple_loss(self, logits: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute loss for non-multi-task approaches."""
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            return loss_fct(logits, labels)
        return torch.tensor(0.0)
    
    def _compute_multi_task_loss(self, utd_logits: torch.Tensor, syndrome_logits: torch.Tensor, 
                                utd_labels: Optional[torch.Tensor], syndrome_labels: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute multi-task loss."""
        total_loss = torch.tensor(0.0, device=utd_logits.device)
        
        if utd_labels is not None:
            utd_loss = F.cross_entropy(utd_logits, utd_labels)
            total_loss += utd_loss
        
        if syndrome_labels is not None:
            syndrome_loss = F.cross_entropy(syndrome_logits, syndrome_labels)
            total_loss += 0.3 * syndrome_loss  # Weight syndrome loss lower
        
        return total_loss


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


def map_labels_to_two_classes(labels: torch.Tensor, cystitis_idx: int) -> torch.Tensor:
    """Maps the original labels to two classes: 0: Sporadic Cystitis (cystitis_idx), 1: everything else"""
    new_labels = torch.zeros_like(labels)
    new_labels = torch.where(labels == cystitis_idx, torch.tensor(0), torch.tensor(1))
    return new_labels


class Config:
    wandb_project = 'VETBERT_UTI_Analysis_DualHead'
    wandb_name = 'VetBERT_DualHead_Fusion'
    model_path = 'havocy28/VetBERTDx'
    max_length = 512
    warmup_steps = 10
    gradient_clip = 1.0
    seed = 42
    epochs = 60
    # Fusion strategy: "concatenation", "weighted", "cross_attention", "multi_task"
    fusion_strategy = "weighted"
    # Default values
    learning_rate = 2e-5
    batch_size = 16
    checkpoint_dir = os.path.join(_REPO_ROOT, 'checkpoints')
    resume_from_checkpoint = None
    # Memory optimization settings
    use_mixed_precision = False  # Set to True for multi_task to save memory
    use_gradient_checkpointing = False  # Set to True for multi_task
    memory_efficient = False  # Set to True for memory-intensive strategies
    gradient_accumulation_steps = 1  # Accumulate gradients for effective larger batch sizes
    # Ultra memory optimizations for multi-task
    ultra_memory_efficient = False  # Set to True for extreme memory constraints
    aggressive_gc = False  # Aggressive garbage collection
    cpu_offload = False  # Offload to CPU when possible


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(file_path: str) -> Tuple[pd.DataFrame, preprocessing.LabelEncoder]:
    df = pd.read_csv(file_path)
    df['Label'] = df['Label'].str.split('-', n=2).str[:2].str.join('-').str.casefold()
    df.loc[df['Label'].str.contains('urolithiasis'), 'Label'] = 'urolithiasis'
    le = preprocessing.LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    return df, le


def tokenize_data(texts: np.ndarray, tokenizer: AutoTokenizer, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    return (
        DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size),
        DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
    )


def train_epoch(model: VetBERTDualHead, dataloader: DataLoader, 
                optimizer: torch.optim.AdamW, scheduler, device: torch.device,
                cystitis_idx: int, scaler=None) -> float:
    model.train()
    total_loss = 0
    accumulation_steps = Config.gradient_accumulation_steps
    
    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch
        
        # Map labels to binary classes
        binary_labels = map_labels_to_two_classes(labels, cystitis_idx).to(device)
        
        # Use mixed precision if scaler is provided
        if scaler is not None:
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=binary_labels,
                    syndrome_labels=None
                )
                loss = outputs.loss / accumulation_steps  # Scale loss for accumulation
            
            # Scale loss and backpropagate
            scaler.scale(loss).backward()
        else:
            # Standard training
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=binary_labels,
                syndrome_labels=None
            )
            
            loss = outputs.loss / accumulation_steps  # Scale loss for accumulation
            loss.backward()
        
        # Only update weights every accumulation_steps
        if (step + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.gradient_clip)
                optimizer.step()
            
            optimizer.zero_grad()
            scheduler.step()
        
        total_loss += loss.item() * accumulation_steps  # Scale back up
        
        # Memory cleanup for memory-intensive strategies
        if Config.memory_efficient and (step + 1) % accumulation_steps == 0:
            torch.cuda.empty_cache()
        
        # Ultra memory optimizations
        if Config.ultra_memory_efficient and (step + 1) % accumulation_steps == 0:
            torch.cuda.empty_cache()
            if Config.aggressive_gc:
                import gc
                gc.collect()
        
        # Log less frequently for gradient accumulation
        if (step + 1) % accumulation_steps == 0:
            wandb.log({
                'train_batch_loss': loss.item() * accumulation_steps,
                'fusion_strategy': Config.fusion_strategy
            })
    
    # Final optimizer step if there are remaining accumulated gradients
    if (step + 1) % accumulation_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.gradient_clip)
            optimizer.step()
        
        optimizer.zero_grad()
        scheduler.step()
    
    return total_loss / len(dataloader)


def evaluate(model: VetBERTDualHead, dataloader: DataLoader,
            device: torch.device, cystitis_idx: int,
            class_names: list = ('sporadic cystitis', 'other')) -> Dict:
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch
            
            binary_labels = map_labels_to_two_classes(labels, cystitis_idx).to(device)
            
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
    f1_scores = f1_score(all_labels, all_preds, average=None)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_names = list(class_names)

    metrics = {
        'val_loss': total_loss / len(dataloader),
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
    # Initialize wandb
    wandb.init(
        project=Config.wandb_project, 
        name=f"{Config.wandb_name}_{Config.fusion_strategy}", 
        resume='allow'
    )
    
    # Get hyperparameters
    batch_size = getattr(wandb.config, 'batch_size', Config.batch_size)
    learning_rate = getattr(wandb.config, 'learning_rate', Config.learning_rate)
    fusion_strategy = getattr(wandb.config, 'fusion_strategy', Config.fusion_strategy)
    
    # Update config
    wandb.config.update({
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'fusion_strategy': fusion_strategy,
        'epochs': Config.epochs,
        'max_length': Config.max_length
    })
    
    print(f"Training with fusion strategy: {fusion_strategy}")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    
    # Set device and seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(Config.seed)
    
    # Load and prepare data
    data_path = Config.train_file
    df, label_encoder = prepare_data(data_path)
    target_classes = label_encoder.classes_.tolist()
    cystitis_idx = find_cystitis_class_index(target_classes)
    other_classes = [c for i, c in enumerate(target_classes) if i != cystitis_idx]
    class_names = [
        target_classes[cystitis_idx],
        other_classes[0] if len(other_classes) == 1 else 'other',
    ]

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_path)
    
    # Tokenize data
    input_ids, attention_masks = tokenize_data(df.text.values, tokenizer, Config.max_length)
    labels = torch.tensor(df.Label.values)
    
    # Create dataset
    dataset = TensorDataset(input_ids, attention_masks, labels)
    
    if getattr(Config, 'val_file', None):
        # Use explicit validation set; train on all training data
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
        val_labels_tensor = torch.tensor(val_df_raw.Label.values)
        val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels_tensor)

        train_dataloader = DataLoader(
            dataset, sampler=RandomSampler(dataset), batch_size=batch_size
        )
        val_dataloader = DataLoader(
            val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size
        )
        print(f"Train samples: {len(dataset)} (full set, --val_file used)")
        print(f"Val samples  : {len(val_dataset)} (from {Config.val_file})")
    else:
        # Internal 80/20 stratified split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=Config.seed)
        train_idx, val_idx = next(sss.split(input_ids, labels))
        train_dataloader, val_dataloader = create_dataloaders(dataset, train_idx, val_idx, batch_size)
    
    # Initialize enhanced model
    model = VetBERTDualHead(
        base_model_path=Config.model_path,
        device=device,
        fusion_strategy=fusion_strategy
    )
    
    # Apply memory optimizations for multi-task and cross-attention
    if fusion_strategy in ["multi_task", "cross_attention"]:
        Config.use_mixed_precision = True
        Config.use_gradient_checkpointing = True
        Config.memory_efficient = True
        Config.gradient_accumulation_steps = 4  # Accumulate over 4 steps
        print(f"Enabling memory optimizations for {fusion_strategy}: mixed precision + gradient checkpointing + accumulation")
        
        # Enable gradient checkpointing on the base model
        if hasattr(model.base_model, 'gradient_checkpointing_enable'):
            model.base_model.gradient_checkpointing_enable()
    
    # Initialize scaler for mixed precision
    scaler = GradScaler() if Config.use_mixed_precision else None
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=Config.warmup_steps,
        num_training_steps=len(train_dataloader) * Config.epochs
    )
    
    # Training variables
    best_f1 = 0
    
    # Training loop
    for epoch in range(Config.epochs):
        print(f"Starting epoch {epoch}/{Config.epochs}")
        
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device, cystitis_idx, scaler)
        metrics = evaluate(model, val_dataloader, device, cystitis_idx, class_names)
        
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            **metrics
        })
        
        if metrics['macro_f1'] > best_f1:
            best_f1 = metrics['macro_f1']
            # Save to WandB directory
            torch.save(model.state_dict(), f'{wandb.run.dir}/best_model.pt')
            # Also save to local checkpoints directory for test inference
            checkpoint_dir = os.path.join(_REPO_ROOT, 'checkpoints', f'dualhead_{fusion_strategy}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = f'{checkpoint_dir}/best_model_lr{learning_rate}_bs{batch_size}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best F1: {best_f1:.4f} with {fusion_strategy} fusion")
            print(f"Model saved to: {checkpoint_path}")
            # Optional fixed-name checkpoint for predictable downstream access
            if getattr(Config, 'checkpoint_name', None):
                fixed_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), Config.checkpoint_dir
                )
                os.makedirs(fixed_dir, exist_ok=True)
                fixed_path = os.path.join(fixed_dir, Config.checkpoint_name)
                torch.save(model.state_dict(), fixed_path)
                print(f"  -> also saved to checkpoints/{Config.checkpoint_name}")
    
    print(f"Training completed! Best F1: {best_f1:.4f}")
    return best_f1


# Sweep configurations for individual fusion strategies
def create_strategy_sweep_config(fusion_strategy: str):
    """Create sweep config for a specific fusion strategy."""
    return {
        'method': 'grid',
        'metric': {
            'name': 'macro_f1',
            'goal': 'maximize'
        },
        'parameters': {
            'fusion_strategy': {
                'values': [fusion_strategy]  # Fixed for this sweep
            },
            'learning_rate': {
                'values': [2e-5, 3e-5]
            },
            'batch_size': {
                'values': [16, 32]
            }
        }
    }

# Individual sweep configs for each strategy
SWEEP_CONFIGS = {
    'concatenation': create_strategy_sweep_config('concatenation'),
    'weighted': create_strategy_sweep_config('weighted'),
    'cross_attention': create_strategy_sweep_config('cross_attention'),
    'multi_task': create_strategy_sweep_config('multi_task')
}

if __name__ == '__main__':
    import sys
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fine-tune VetBERT with Dual-Head Architecture')
    parser.add_argument('--fusion_strategy', type=str, default='weighted',
                        choices=['concatenation', 'weighted', 'cross_attention', 'multi_task'],
                        help='Fusion strategy for dual-head architecture')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of training epochs')
    parser.add_argument('--use_mixed_precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--use_gradient_checkpointing', action='store_true',
                        help='Use gradient checkpointing to save memory')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--memory_efficient', action='store_true',
                        help='Use memory efficient settings')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='WandB run name')
    parser.add_argument('--train_file', type=str,
                        default='../../data/clean/refined_model/cystitis_refined_train_binary.csv',
                        help='Path to training CSV (text, Label)')
    parser.add_argument('--val_file', type=str, default=None,
                        help='Path to validation CSV; overrides internal 80/20 split')
    parser.add_argument('--checkpoint_name', type=str, default=None,
                        help='Filename for fixed-path best-model checkpoint under checkpoints/')
    parser.add_argument('--max_length', type=int, default=None,
                        help='Maximum tokenisation length (default: Config.max_length = 512)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for model init / training (default: 42)')

    args = parser.parse_args()

    # Update Config with parsed arguments
    Config.fusion_strategy = args.fusion_strategy
    Config.learning_rate = args.learning_rate
    Config.batch_size = args.batch_size
    Config.epochs = args.epochs
    Config.use_mixed_precision = args.use_mixed_precision
    Config.use_gradient_checkpointing = args.use_gradient_checkpointing
    Config.gradient_accumulation_steps = args.gradient_accumulation_steps
    Config.memory_efficient = args.memory_efficient
    Config.train_file = args.train_file
    Config.val_file = args.val_file
    Config.checkpoint_name = args.checkpoint_name
    Config.seed = args.seed
    if args.max_length is not None:
        Config.max_length = args.max_length

    if args.wandb_name:
        Config.wandb_name = args.wandb_name
    
    if len(sys.argv) > 2 and sys.argv[1] == 'sweep':
        strategy = sys.argv[2]
        if strategy not in SWEEP_CONFIGS:
            print(f"Error: Unknown fusion strategy '{strategy}'. Available: {list(SWEEP_CONFIGS.keys())}")
            sys.exit(1)
        
        sweep_id = wandb.sweep(SWEEP_CONFIGS[strategy], project="VETBERT_UTI_DualHead_Optimization")
        print(f"Created sweep for {strategy}: {sweep_id}")
        print(f"Run: wandb agent {sweep_id}")
        
    elif len(sys.argv) > 1 and sys.argv[1] == 'agent':
        if len(sys.argv) < 3:
            print("Usage: python script.py agent <sweep_id>")
            sys.exit(1)
        wandb.agent(sys.argv[2], function=main, project="VETBERT_UTI_DualHead_Optimization")
        
    elif len(sys.argv) > 1 and sys.argv[1] == 'run_strategy':
        # Run single strategy with fixed params (for testing)
        if len(sys.argv) < 3:
            print("Usage: python script.py run_strategy <strategy>")
            sys.exit(1)
        strategy = sys.argv[2]
        Config.fusion_strategy = strategy
        Config.wandb_name = f'VetBERT_DualHead_{strategy}_test'
        print(f"Running {strategy} fusion strategy...")
        best_f1 = main()
        print(f"Completed {strategy}: Best F1 = {best_f1:.4f}")
        
    else:
        main()