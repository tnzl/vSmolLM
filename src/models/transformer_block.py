"""Transformer Decoder Block for GPT-2"""

import torch
import torch.nn as nn
from .attention import CausalSelfAttention


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation"""
    
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Single transformer decoder block (GPT-2 style)"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        
        # Pre-norm architecture (GPT-2 style)
        self.ln1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        
        self.ln2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-norm: LayerNorm -> Attention -> Residual
        residual = x
        x = self.ln1(x)
        x = self.attn(x, mask)
        x = self.dropout(x)
        x = residual + x
        
        # Pre-norm: LayerNorm -> FFN -> Residual
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x
