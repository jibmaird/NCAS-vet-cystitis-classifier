import os
import time
import datetime

import random
from typing import Tuple, Dict
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
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

# Repo root, independent of the caller's working directory (this file lives
# at <repo_root>/scripts/training/).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


class VetBERTBinary(nn.Module):
    def __init__(self, base_model_path: str, device: torch.device):
        super().__init__()
        # Load the base model without classification head
        self.base_model = AutoModel.from_pretrained(
            base_model_path,
            trust_remote_code=False,
            use_safetensors=True
        )
        # Define the classifier for 3 classes
        hidden_size = self.base_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 2)
        )
        self.to(device)
    def forward(self, input_ids, attention_mask, labels=None):
        # Get base model outputs (features)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Get the [CLS] token hidden state
        cls_hidden_state = outputs.last_hidden_state[:, 0, :]
        # Get logits for 3 classes
        logits = self.classifier(cls_hidden_state)
        if labels is not None:
            # Calculate loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return type('ModelOutput', (), {
                'loss': loss,
                'logits': logits
            })
        return type('ModelOutput', (), {
            'logits': logits
        })

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
    """
    Maps the original labels to two classes:
    0: Sporadic Cystitis (the class at cystitis_idx)
    1: everything else
    """
    new_labels = torch.zeros_like(labels)
    new_labels = torch.where(labels == cystitis_idx, torch.tensor(0), torch.tensor(1))
    return new_labels

class Config:
    wandb_project = 'VETBERT_UTI_Analysis_Binary'
    wandb_name = 'VetBERT_UTI_TESTS_1'
    model_path = 'havocy28/VetBERTDx'  # Changed to Hugging Face model ID
    max_length = 512
    warmup_steps = 10
    gradient_clip = 1.0
    seed = 42
    epochs = 60  # Restored to original for full training
    # Default values for non-sweep training
    learning_rate = 2e-5
    batch_size = 16
    checkpoint_dir = os.path.join(_REPO_ROOT, 'checkpoints')  # Directory to save checkpoints
    resume_from_checkpoint = None  # Path to checkpoint to resume from
    # Data paths — override via CLI args
    train_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '../../data/clean/refined_model/cystitis_refined_train_binary.csv')
    val_file = None        # If set, uses explicit validation set instead of 80/20 split
    checkpoint_name = None  # If set, saves best model to checkpoints/<checkpoint_name>

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, optimizer, scheduler, epoch, best_f1, checkpoint_dir, run_id):
    """Save training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_f1': best_f1,
        'run_id': run_id
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['best_f1'], checkpoint.get('run_id', None)

def prepare_data(file_path: str, min_samples: int = 7) -> Tuple[pd.DataFrame, preprocessing.LabelEncoder]:
    # This function remains largely the same as it handles the basic data preparation
    df = pd.read_csv(file_path)
    df['Label'] = df['Label'].str.split('-', n=2).str[:2].str.join('-').str.casefold()
    df.loc[df['Label'].str.contains('urolithiasis'), 'Label'] = 'urolithiasis'
    # Encode labels
    le = preprocessing.LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    return df, le

def tokenize_data(texts: np.ndarray, tokenizer: AutoTokenizer, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # This function remains exactly the same as it only handles tokenization
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
    # This function remains exactly the same as it only handles dataloader creation
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    return (
        DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size),
        DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
    )

def train_epoch(model: VetBERTBinary, dataloader: DataLoader, 
                optimizer: torch.optim.AdamW, scheduler, device: torch.device,
                cystitis_idx: int) -> float:
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch
        # Map labels to binary classes
        binary_labels = map_labels_to_two_classes(
            labels, cystitis_idx
        ).to(device)
        
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
        wandb.log({'train_batch_loss': loss.item()})
    return total_loss / len(dataloader)

def evaluate(model: VetBERTBinary, dataloader: DataLoader,
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
            # Map labels to binary classes
            binary_labels = map_labels_to_two_classes(
                labels, cystitis_idx
            ).to(device)
            
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
        'macro_f1': macro_f1
    }
    
    # Add per-class metrics
    for idx, class_name in enumerate(class_names):
        metrics[f'f1_{class_name}'] = f1_scores[idx]
        metrics[f'accuracy_{class_name}'] = conf_matrix[idx, idx] / conf_matrix[idx].sum()

    return metrics

def main():
    # Initialize wandb
    wandb.init(project=Config.wandb_project, name=Config.wandb_name, resume='allow')
    
    # Get hyperparameters (either from sweep or from Config defaults)
    batch_size = getattr(wandb.config, 'batch_size', Config.batch_size)
    learning_rate = getattr(wandb.config, 'learning_rate', Config.learning_rate)
    
    # Only update config if we're not in a sweep run
    # Check if this is a sweep by looking for the sweep_id in environment or wandb run
    is_sweep_run = (hasattr(wandb.run, 'sweep') and wandb.run.sweep is not None) or \
                   ('WANDB_SWEEP_ID' in os.environ) or \
                   (wandb.run.sweep_id is not None)
    
    if not is_sweep_run:
        # Not a sweep run, safe to update config
        wandb.config.update({
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'epochs': Config.epochs,
            'max_length': Config.max_length
        })
    
    print(f"Using batch_size: {batch_size}, learning_rate: {learning_rate}")
    print(f"Sweep run: {is_sweep_run}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(Config.seed)
    
    # Load and prepare data
    df, label_encoder = prepare_data(Config.train_file)
    target_classes = label_encoder.classes_.tolist()

    # Get index for the Sporadic Cystitis class (only need this for binary classification)
    cystitis_idx = find_cystitis_class_index(target_classes)
    other_classes = [c for i, c in enumerate(target_classes) if i != cystitis_idx]
    class_names = [
        target_classes[cystitis_idx],
        other_classes[0] if len(other_classes) == 1 else 'other',
    ]

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        Config.model_path,
        trust_remote_code=False,
        use_safetensors=True
    )

    # Tokenize data
    input_ids, attention_masks = tokenize_data(df.text.values, tokenizer, Config.max_length)
    labels = torch.tensor(df.Label.values)

    # Create dataset
    dataset = TensorDataset(input_ids, attention_masks, labels)

    if Config.val_file:
        # Use all training data; load an explicit validation set with the same encoder
        val_df = pd.read_csv(Config.val_file)
        val_df['Label'] = (
            val_df['Label']
            .str.split('-', n=2).str[:2].str.join('-')
            .str.casefold()
        )
        val_df['Label'] = label_encoder.transform(val_df['Label'])
        val_input_ids, val_attention_masks = tokenize_data(
            val_df.text.values, tokenizer, Config.max_length
        )
        val_labels = torch.tensor(val_df.Label.values)
        val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)

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
        train_dataloader, val_dataloader = create_dataloaders(
            dataset, train_idx, val_idx, batch_size
        )
    
    # Initialize model (now using binary version)
    model = VetBERTBinary(
        base_model_path=Config.model_path,
        device=device
    )
    
    # Initialize optimizer and scheduler - use local variables
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=Config.warmup_steps,
        num_training_steps=len(train_dataloader) * Config.epochs
    )
    
    # Initialize training variables
    start_epoch = 0
    best_f1 = 0
    
    # Check for existing checkpoint - only resume from an EXPLICITLY specified,
    # task-matched checkpoint path. Do NOT fall back to a generic
    # 'latest_checkpoint.pt': in a multi-task environment that file may belong
    # to a completely different run (different label scheme, different
    # Config.epochs), and silently loading it can skip training entirely while
    # reporting a stale best_f1 as if it were a real result.
    if not is_sweep_run and Config.resume_from_checkpoint and os.path.exists(Config.resume_from_checkpoint):
        print(f"Resuming from checkpoint: {Config.resume_from_checkpoint}")
        start_epoch, best_f1, saved_run_id = load_checkpoint(
            Config.resume_from_checkpoint, model, optimizer, scheduler
        )
        start_epoch += 1  # Start from next epoch
        print(f"Resumed from epoch {start_epoch-1}, best F1: {best_f1}")
    elif is_sweep_run:
        print("Sweep run detected - starting fresh training without checkpoint resume")
    else:
        print("Starting fresh training (no checkpoint resume requested)")
    
    # Training loop
    for epoch in range(start_epoch, Config.epochs):
        print(f"Starting epoch {epoch}/{Config.epochs}")
        
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, 
                               device, cystitis_idx)
        metrics = evaluate(model, val_dataloader, device, cystitis_idx, class_names)
        
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            **metrics
        })
        
        if metrics['macro_f1'] > best_f1:
            best_f1 = metrics['macro_f1']
            torch.save(model.state_dict(), f'{wandb.run.dir}/best_model.pt')
            print(f"New best F1: {best_f1:.4f}")
            if Config.checkpoint_name:
                ckpt_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), Config.checkpoint_dir
                )
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(ckpt_dir, Config.checkpoint_name))
                print(f"  -> also saved to checkpoints/{Config.checkpoint_name}")
        
        # Save checkpoint every 5 epochs or if it's the last epoch
        if (epoch + 1) % 5 == 0 or epoch == Config.epochs - 1:
            # Use run-specific checkpoint names for sweep runs
            checkpoint_dir = Config.checkpoint_dir
            if is_sweep_run:
                checkpoint_dir = os.path.join(Config.checkpoint_dir, f"sweep_{wandb.run.id}")
            save_checkpoint(model, optimizer, scheduler, epoch, best_f1, 
                          checkpoint_dir, wandb.run.id)
    
    print(f"Training completed! Best F1: {best_f1:.4f}")

# Sweep configuration for WandB - FULL SWEEP
sweep_config = {
    'method': 'bayes',  # Back to Bayesian for intelligent exploration
    'metric': {
        'name': 'macro_f1',
        'goal': 'maximize'
    },
    'program': 'fine_tune_on_cystitis_refined_binary.py',
    'parameters': {
        'learning_rate': {
            'values': [2e-5, 3e-5, 4e-5, 5e-5]  # All 4 learning rates
        },
        'batch_size': {
            'values': [8, 16, 32]  # All 3 batch sizes  
        }
    },
    # Full sweep limits
    'run_cap': 20,  # Allow more runs for Bayesian exploration
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 10,  # Allow at least 10 epochs before early termination
        'eta': 2,
    }
}

if __name__ == '__main__':
    import sys
    import argparse

    # Parse optional CLI overrides for data paths and checkpoint naming
    _parser = argparse.ArgumentParser(
        description='VetBERT Binary Fine-tuning',
        add_help=False  # let wandb/sweep args pass through
    )
    _parser.add_argument('--train_file', type=str, default=None,
                         help='Path to training CSV (text, Label columns)')
    _parser.add_argument('--val_file', type=str, default=None,
                         help='Path to validation CSV; overrides internal 80/20 split')
    _parser.add_argument('--checkpoint_name', type=str, default=None,
                         help='Filename for fixed-path best-model checkpoint under checkpoints/')
    _known, _remaining = _parser.parse_known_args()

    if _known.train_file:
        Config.train_file = _known.train_file
    if _known.val_file:
        Config.val_file = _known.val_file
    if _known.checkpoint_name:
        Config.checkpoint_name = _known.checkpoint_name

    # Reconstruct sys.argv without the custom flags so wandb/sweep parsing still works
    sys.argv = [sys.argv[0]] + _remaining

    if len(sys.argv) > 1 and sys.argv[1] == 'sweep':
        sweep_id = wandb.sweep(sweep_config, project="VETBERT_UTI_Analysis_Binary")
        print(f"Created sweep: {sweep_id}")
        print(f"Run: wandb agent {sweep_id}")
    elif len(sys.argv) > 1 and sys.argv[1] == 'agent':
        wandb.agent(sys.argv[2], function=main, project="VETBERT_UTI_Analysis_Binary")
    else:
        main()
