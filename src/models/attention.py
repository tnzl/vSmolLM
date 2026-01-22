"""Multi-head Self-Attention Mechanism for GPT-2"""

import torch
import torch.nn as nn
import math


class CausalSelfAttention(nn.Module):
    """Causal (masked) multi-head self-attention"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # [batch, seq_len, embed_dim]
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch, num_heads, seq_len, seq_len]
        
        # Apply causal mask
        if mask is None:
            # Create causal mask (lower triangular)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        scores = scores.masked_fill(mask, float('-inf'))
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [batch, num_heads, seq_len, head_dim]
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, seq_len, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, embed_dim)  # [batch, seq_len, embed_dim]
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output
