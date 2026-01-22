#!/usr/bin/env python3
"""Evaluation Script"""

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
from src.training import Evaluator
from src.training.utils import get_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 on WikiText-103")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/wikitext103.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override batch size if provided
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Reconstruct config from checkpoint if available
    if checkpoint.get('config'):
        checkpoint_config_dict = checkpoint['config']
        # Merge with file config (checkpoint takes precedence for model config)
        if 'model' in checkpoint_config_dict:
            for key, value in checkpoint_config_dict['model'].items():
                setattr(config.model, key, value)
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = GPT2Tokenizer()
    
    # Load dataset
    print(f"Loading {args.split} dataset...")
    dataset = WikiText103Dataset(
        data_dir=config.data.data_dir,
        split=args.split,
        tokenizer=tokenizer,
        max_length=config.data.max_length,
        download=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
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
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Print model info
    num_params = model.get_num_params()
    print(f"Model parameters: {num_params:,}")
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = Evaluator(model, device)
    
    # Evaluate
    print(f"Evaluating on {args.split} split...")
    metrics = evaluator.evaluate(dataloader)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Evaluation Results ({args.split} split):")
    print(f"{'='*50}")
    for key, value in metrics.items():
        print(f"{key:20s}: {value:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
