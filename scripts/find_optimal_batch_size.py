#!/usr/bin/env python3
"""Find optimal batch size for maximum GPU utilization"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import GPT2
from src.data import GPT2Tokenizer
from src.config import load_config
from src.training.utils import get_device, set_seed


def find_max_batch_size(config_path: str, start_batch_size: int = 1, max_seq_len: int = 1024):
    """
    Find the maximum batch size that fits in GPU memory
    
    Args:
        config_path: Path to config file
        start_batch_size: Starting batch size to test
        max_seq_len: Maximum sequence length
    """
    config = load_config(config_path)
    device = get_device()
    set_seed(42)
    
    print(f"Finding optimal batch size on {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("="*60)
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer()
    
    # Initialize model
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
    model.tie_weights()
    model.to(device)
    model.train()
    
    # Use mixed precision
    use_amp = config.training.use_mixed_precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    batch_size = start_batch_size
    max_batch_size = None
    
    print(f"Testing batch sizes (sequence length: {max_seq_len})...")
    print("-"*60)
    
    while True:
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Create dummy batch
            input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, max_seq_len), device=device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            labels = input_ids.clone()
            
            # Forward pass
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs['loss']
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']
            
            # Backward pass
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Check memory usage
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            
            print(f"✓ Batch size {batch_size:3d} | "
                  f"Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
            
            max_batch_size = batch_size
            
            # Clear gradients
            model.zero_grad()
            
            # Try next batch size
            batch_size += 1  # Increase by 1 for more precise search
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"✗ Batch size {batch_size:3d} - OOM")
                torch.cuda.empty_cache()
                break
            else:
                raise e
    
    print("-"*60)
    if max_batch_size is None:
        print(f"\n⚠️  Even batch size {start_batch_size} causes OOM!")
        print(f"\nTry reducing sequence length:")
        print(f"  python scripts/find_optimal_batch_size.py --config {config_path} --max-seq-len 512")
        print(f"  python scripts/find_optimal_batch_size.py --config {config_path} --max-seq-len 256")
        return None
    
    print(f"\nMaximum batch size: {max_batch_size}")
    print(f"\nRecommended settings:")
    print(f"  batch_size: {max_batch_size}")
    print(f"  gradient_accumulation_steps: 1  # Or adjust to get desired effective batch size")
    print(f"  use_mixed_precision: {use_amp}")
    print(f"  max_seq_len: {max_seq_len}")
    
    if max_batch_size is not None:
        # Calculate effective batch size with gradient accumulation
        effective_batch_sizes = []
        for grad_acc in [1, 2, 4, 8]:
            effective = max_batch_size * grad_acc
            effective_batch_sizes.append((grad_acc, effective))
        
        print(f"\nEffective batch sizes with gradient accumulation:")
        for grad_acc, effective in effective_batch_sizes:
            print(f"  gradient_accumulation_steps={grad_acc}: effective batch size = {effective}")
    
    return max_batch_size


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Find optimal batch size for GPU")
    parser.add_argument("--config", type=str, default="configs/wikitext103.yaml")
    parser.add_argument("--start-batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    
    args = parser.parse_args()
    find_max_batch_size(args.config, args.start_batch_size, args.max_seq_len)
