"""GPT-2 Model Architecture"""

import torch
import torch.nn as nn
from typing import Optional

from .embeddings import GPT2Embeddings
from .transformer_block import TransformerBlock


class GPT2(nn.Module):
    """GPT-2 Language Model"""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = True
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.embeddings = GPT2Embeddings(vocab_size, embed_dim, max_seq_len, dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim, dropout, layer_norm_eps)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Tie weights (optional, GPT-2 style)
        # In GPT-2, the embedding weights are tied with the output projection
        # We'll handle this in the forward pass or initialization
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def tie_weights(self):
        """Tie the embedding weights with the output projection"""
        if hasattr(self.embeddings.token_embedding, 'embedding'):
            self.lm_head.weight = self.embeddings.token_embedding.embedding.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Forward pass
        
        Args:
            input_ids: Token indices [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len] (optional)
            labels: Labels for language modeling [batch_size, seq_len] (optional)
            
        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        batch_size, seq_len = input_ids.size()
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device),
            diagonal=1
        ).bool().unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: [batch_size, seq_len] where 1 = attend, 0 = mask
            # Convert to [batch_size, 1, 1, seq_len] for broadcasting
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            # Combine with causal mask
            causal_mask = causal_mask | (~attn_mask.bool())
        
        # Embeddings
        x = self.embeddings(input_ids)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask=causal_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)  # [batch_size, seq_len, vocab_size]
        
        output = {'logits': logits}
        
        # Compute loss if labels are provided
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            output['loss'] = loss
        
        return output
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """Get number of parameters"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embeddings.token_embedding.embedding.weight.numel()
        return n_params
