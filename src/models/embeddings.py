"""Token and Position Embeddings for GPT-2"""

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """Token embedding layer"""
    
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.embed_dim)


class PositionEmbedding(nn.Module):
    """Position embedding layer using learned embeddings"""
    
    def __init__(self, max_seq_len: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, embed_dim)
        self.max_seq_len = max_seq_len
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
        return self.embedding(positions)


class GPT2Embeddings(nn.Module):
    """Combined token and position embeddings for GPT-2"""
    
    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.position_embedding = PositionEmbedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(input_ids)
        embeddings = token_embeds + pos_embeds
        return self.dropout(embeddings)
