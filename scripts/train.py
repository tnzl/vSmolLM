#!/usr/bin/env python3
"""Main Training Script"""

import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import GPT2
from src.data import WikiText103Dataset, GPT2Tokenizer
from src.data.collate import collate_fn
from src.config import load_config
from src.training import Trainer
from src.training.utils import set_seed, get_device


def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 on WikiText-103")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/wikitext103.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--continue",
        action="store_true",
        dest="continue_training",
        help="Automatically resume from the latest checkpoint in outputs directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Handle resume options
    if args.continue_training:
        # Auto-find latest checkpoint
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all checkpoint files (step-based, epoch-based, and final)
        checkpoint_files = list(output_dir.glob("checkpoint_*.pt"))
        
        if checkpoint_files:
            # Sort by modification time, get latest
            latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            config.training.resume_from = str(latest_checkpoint)
            
            # Try to extract step/epoch info from filename for display
            step_match = re.search(r'step_(\d+)', latest_checkpoint.name)
            epoch_match = re.search(r'epoch_(\d+)', latest_checkpoint.name)
            
            info_parts = []
            if step_match:
                info_parts.append(f"step {step_match.group(1)}")
            if epoch_match:
                info_parts.append(f"epoch {epoch_match.group(1)}")
            
            info_str = f" ({', '.join(info_parts)})" if info_parts else ""
            
            print(f"🔄 Auto-resuming from latest checkpoint: {latest_checkpoint.name}{info_str}")
            print(f"   Path: {latest_checkpoint}")
            print(f"   Size: {latest_checkpoint.stat().st_size / 1e6:.1f} MB")
        else:
            print("⚠️  No checkpoint found in outputs directory. Starting fresh training.")
    elif args.resume:
        # Use explicitly provided checkpoint
        if not Path(args.resume).exists():
            print(f"❌ Error: Checkpoint file not found: {args.resume}")
            sys.exit(1)
        config.training.resume_from = args.resume
        print(f"🔄 Resuming from checkpoint: {args.resume}")
    
    # Override seed if provided
    if args.seed is not None:
        config.seed = args.seed
    
    # Set random seed
    set_seed(config.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = GPT2Tokenizer()
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = WikiText103Dataset(
        data_dir=config.data.data_dir,
        split="train",
        tokenizer=tokenizer,
        max_length=config.data.max_length,
        download=True
    )
    
    val_dataset = WikiText103Dataset(
        data_dir=config.data.data_dir,
        split="validation",  # HuggingFace uses 'validation' instead of 'valid'
        tokenizer=tokenizer,
        max_length=config.data.max_length,
        download=False
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id=tokenizer.pad_token_id)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
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
    
    # Tie weights (GPT-2 style)
    model.tie_weights()
    
    # Print model info
    num_params = model.get_num_params()
    num_params_non_embed = model.get_num_params(non_embedding=True)
    print(f"Model parameters: {num_params:,} ({num_params_non_embed:,} non-embedding)")
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
