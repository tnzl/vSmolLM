"""Evaluation and Metrics"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional

from ..models import GPT2


class Evaluator:
    """Evaluator for computing metrics on validation/test sets"""
    
    def __init__(self, model: GPT2, device: torch.device):
        """
        Initialize evaluator
        
        Args:
            model: GPT-2 model
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on dataset
        
        Args:
            dataloader: DataLoader for evaluation set
            
        Returns:
            Dictionary with evaluation metrics
        """
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Create labels (shifted input_ids)
                labels = input_ids.clone()
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                
                # Count non-padding tokens
                if attention_mask is not None:
                    num_tokens = attention_mask.sum().item()
                else:
                    num_tokens = input_ids.numel()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        # Compute average loss and perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity
        }
        
        return metrics
    
    def compute_perplexity(self, dataloader: DataLoader) -> float:
        """
        Compute perplexity on dataset
        
        Args:
            dataloader: DataLoader for evaluation set
            
        Returns:
            Perplexity value
        """
        metrics = self.evaluate(dataloader)
        return metrics['perplexity']
