"""Training Utilities"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Optional


def get_learning_rate_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
    scheduler_type: str = "cosine"
):
    """
    Get learning rate scheduler with warmup
    
    Args:
        optimizer: Optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate
        scheduler_type: Type of scheduler ('cosine' or 'linear')
        
    Returns:
        Learning rate scheduler
    """
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine or linear decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            if scheduler_type == "cosine":
                return max(min_lr, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))))
            else:  # linear
                return max(min_lr, 1.0 - progress)
    
    return LambdaLR(optimizer, lr_lambda)


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get available device (CUDA if available, else CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> int:
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
