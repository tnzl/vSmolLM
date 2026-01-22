#!/usr/bin/env python3
"""Test Training Script - Runs a small number of iterations to verify training works"""

import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import GPT2
from src.data import WikiText103Dataset, GPT2Tokenizer
from src.data.collate import collate_fn
from src.config import load_config, Config
from src.training import Trainer
from src.training.utils import set_seed, get_device


def main():
    parser = argparse.ArgumentParser(description="Test GPT-2 training with small number of iterations")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/wikitext103.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum number of training steps"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for testing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config for testing
    config.training.batch_size = args.batch_size
    config.training.gradient_accumulation_steps = 1  # No accumulation for testing
    config.training.max_epochs = 1  # Just one epoch
    config.training.eval_interval = args.max_steps + 1  # Don't eval during test
    config.training.save_interval = args.max_steps + 1  # Don't save during test
    config.training.log_interval = 5  # Log every 5 steps
    config.wandb.enabled = False  # Disable W&B for testing
    config.data.num_workers = 0  # Avoid multiprocessing issues in test
    config.data.max_length = 128  # Shorter sequences for faster testing
    
    # Use smaller model for testing
    config.model.embed_dim = 256
    config.model.num_layers = 2
    config.model.num_heads = 4
    config.model.ffn_dim = 1024
    config.model.max_seq_len = 128
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    print(f"Testing with {args.max_steps} steps, batch size {args.batch_size}")
    print("="*60)
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = GPT2Tokenizer()
    
    # Load datasets (use a small subset)
    print("Loading datasets...")
    try:
        train_dataset = WikiText103Dataset(
            data_dir=config.data.data_dir,
            split="train",
            tokenizer=tokenizer,
            max_length=config.data.max_length,
            download=True
        )
        
        # Limit dataset size for testing (but ensure we have enough for the test)
        min_samples = args.max_steps * args.batch_size * 2  # Ensure enough samples
        if len(train_dataset) > min_samples:
            # Create a subset
            indices = torch.randperm(len(train_dataset))[:min_samples]
            train_dataset = torch.utils.data.Subset(train_dataset, indices)
            print(f"Using subset of {len(train_dataset)} samples for testing")
        else:
            print(f"Train dataset size: {len(train_dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating dummy dataset for testing...")
        # Create a simple dummy dataset
        class DummyDataset:
            def __init__(self, size=50):
                self.size = size
                self.tokenizer = tokenizer
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                # Create a simple dummy text
                text = "This is a test sentence. " * 10
                token_ids = self.tokenizer.encode(text, max_length=config.data.max_length, truncation=True)
                return {'input_ids': torch.tensor(token_ids, dtype=torch.long)}
        
        train_dataset = DummyDataset(max(50, args.max_steps * args.batch_size * 2))
        print(f"Using dummy dataset with {len(train_dataset)} samples")
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id=tokenizer.pad_token_id)
    )
    
    # Initialize model
    print("Initializing model...")
    model = GPT2(
        vocab_size=config.model.vocab_size,
        embed_dim=config.model.embed_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        ffn_dim=config.model.ffn_dim,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout,
        layer_norm_eps=config.model.layer_norm_eps,
        bias=config.model.bias
    )
    
    # Tie weights
    model.tie_weights()
    
    # Print model info
    num_params = model.get_num_params()
    print(f"Model parameters: {num_params:,}")
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,  # No validation for quick test
        config=config,
        device=device
    )
    
    # Track losses
    losses = []
    
    # Manual training loop for testing
    print("\nStarting test training...")
    print("="*60)
    model.train()
    
    step = 0
    for epoch in range(1):
        for batch_idx, batch in enumerate(train_loader):
            if step >= args.max_steps:
                break
                
            # Training step
            metrics = trainer.train_step(batch)
            
            # Optimizer step
            if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0:
                trainer.optimizer_step()
                step += 1
                
                loss = metrics['loss']
                losses.append(loss)
                
                if step % config.training.log_interval == 0:
                    print(f"Step {step:3d}/{args.max_steps} | Loss: {loss:.4f} | LR: {metrics['learning_rate']:.2e}")
            
            if step >= args.max_steps:
                break
    
    # Print summary
    print("="*60)
    print("Training Test Summary:")
    print(f"  Total steps: {len(losses)}")
    if len(losses) > 0:
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        if len(losses) > 1:
            loss_change = losses[-1] - losses[0]
            loss_change_pct = (loss_change / losses[0]) * 100 if losses[0] > 0 else 0
            print(f"  Loss change: {loss_change:+.4f} ({loss_change_pct:+.2f}%)")
            
            if loss_change < 0:
                print("  ✓ Loss is decreasing - training appears to be working!")
            elif loss_change > 0:
                print("  ⚠ Loss is increasing - may need to adjust learning rate or check implementation")
            else:
                print("  ⚠ Loss is not changing - check implementation")
    
    print("="*60)
    print("Test completed!")


if __name__ == "__main__":
    main()
