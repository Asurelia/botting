"""
Transformer Core - Composants Transformer pour AlphaStar DOFUS
Architectures d'attention optimisées pour vision et séquences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any
import logging

from config import config, get_device

logger = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    """Attention multi-têtes optimisée pour AMD"""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        batch_size, seq_len, _ = query.shape

        # Projections
        q = self.q_proj(query)  # [B, L, H]
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape pour multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, NH, L, HD]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention avec optimisation AMD
        if hasattr(F, 'scaled_dot_product_attention') and config.amd.use_sdp_attention:
            try:
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    scale=self.scale
                )
                attn_weights = None
            except Exception:
                # Fallback
                attn_output, attn_weights = self._manual_attention(q, k, v, mask)
        else:
            attn_output, attn_weights = self._manual_attention(q, k, v, mask)

        # Reshape et projection finale
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        output = self.out_proj(attn_output)

        if return_attention:
            return output, attn_weights
        return output, None

    def _manual_attention(self,
                         q: torch.Tensor,
                         k: torch.Tensor,
                         v: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attention manuelle avec optimisations"""

        # Scores d'attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, NH, L, L]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax avec mixed precision
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(q.dtype)
        attn_weights = self.dropout(attn_weights)

        # Appliquer attention
        attn_output = torch.matmul(attn_weights, v)  # [B, NH, L, HD]

        return attn_output, attn_weights

class TransformerEncoderLayer(nn.Module):
    """Couche Transformer encodeur avec optimisations"""

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 ff_dim: int,
                 dropout: float = 0.1,
                 activation: str = "relu"):
        super().__init__()

        self.self_attn = MultiHeadAttention(hidden_size, num_heads, dropout)

        # Feed Forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size),
            nn.Dropout(dropout)
        )

        # Normalisations
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # Gradient checkpointing support
        self.gradient_checkpointing = config.amd.enable_gradient_checkpointing

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # Self-attention avec résiduelle
        if self.gradient_checkpointing and self.training:
            attn_output, attn_weights = torch.utils.checkpoint.checkpoint(
                self._attention_block, x, mask, return_attention
            )
        else:
            attn_output, attn_weights = self._attention_block(x, mask, return_attention)

        x = self.norm1(x + attn_output)

        # Feed forward avec résiduelle
        if self.gradient_checkpointing and self.training:
            ff_output = torch.utils.checkpoint.checkpoint(self.ff, x)
        else:
            ff_output = self.ff(x)

        x = self.norm2(x + ff_output)

        return x, attn_weights

    def _attention_block(self, x, mask, return_attention):
        return self.self_attn(x, x, x, mask, return_attention)

class TransformerEncoder(nn.Module):
    """Encodeur Transformer complet"""

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_layers: int,
                 ff_dim: Optional[int] = None,
                 dropout: float = 0.1):
        super().__init__()

        if ff_dim is None:
            ff_dim = hidden_size * 4

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        all_attention_weights = [] if return_attention else None

        for layer in self.layers:
            x, attn_weights = layer(x, mask, return_attention)
            if return_attention and attn_weights is not None:
                all_attention_weights.append(attn_weights)

        x = self.norm(x)

        if return_attention:
            # Moyenne des poids d'attention de toutes les couches
            if all_attention_weights:
                avg_attention = torch.stack(all_attention_weights).mean(0)
                return x, avg_attention

        return x, None

class SpatialTransformer(nn.Module):
    """Transformer pour données spatiales (grilles, cartes)"""

    def __init__(self,
                 input_channels: int,
                 hidden_size: int,
                 num_heads: int,
                 num_layers: int = 2,
                 patch_size: int = 8):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size

        # Conversion patches
        self.patch_embed = nn.Conv2d(
            input_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, 256, hidden_size))  # Max 256 patches

        # Transformer
        self.transformer = TransformerEncoder(
            hidden_size, num_heads, num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] données spatiales
        Returns:
            [B, N, hidden_size] features encodées
        """
        batch_size = x.size(0)

        # Conversion en patches
        patches = self.patch_embed(x)  # [B, hidden_size, H', W']
        _, _, h, w = patches.shape

        # Flatten patches
        patches = patches.flatten(2).transpose(1, 2)  # [B, N, hidden_size]
        num_patches = patches.size(1)

        # Position embeddings
        pos_embed = self.pos_embed[:, :num_patches, :]
        patches = patches + pos_embed

        # Transformer
        output, _ = self.transformer(patches)

        return output

class EntityTransformer(nn.Module):
    """Transformer pour entités (joueurs, monstres, objets)"""

    def __init__(self,
                 entity_dim: int,
                 hidden_size: int,
                 num_heads: int,
                 max_entities: int = 100,
                 num_layers: int = 2):
        super().__init__()

        self.entity_dim = entity_dim
        self.hidden_size = hidden_size
        self.max_entities = max_entities

        # Projection entités
        self.entity_proj = nn.Linear(entity_dim, hidden_size)

        # Type embeddings (player, monster, npc, item)
        self.type_embed = nn.Embedding(10, hidden_size)

        # Position embeddings
        self.pos_embed = nn.Embedding(max_entities, hidden_size)

        # Transformer
        self.transformer = TransformerEncoder(
            hidden_size, num_heads, num_layers
        )

    def forward(self, entities: torch.Tensor, entity_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            entities: [B, N, entity_dim] features des entités
            entity_types: [B, N] types d'entités (optionnel)
        Returns:
            [B, N, hidden_size] features encodées
        """
        batch_size, num_entities, _ = entities.shape

        # Projection des entités
        entity_features = self.entity_proj(entities)

        # Type embeddings si fournis
        if entity_types is not None:
            type_embeds = self.type_embed(entity_types)
            entity_features = entity_features + type_embeds

        # Position embeddings
        positions = torch.arange(num_entities, device=entities.device)
        pos_embeds = self.pos_embed(positions).unsqueeze(0).expand(batch_size, -1, -1)
        entity_features = entity_features + pos_embeds

        # Mask pour entités valides (si padding)
        mask = (entities.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]

        # Transformer
        output, _ = self.transformer(entity_features, mask)

        return output

# Utilitaires
def create_attention_mask(seq_len: int, causal: bool = False) -> torch.Tensor:
    """Crée un masque d'attention"""
    if causal:
        # Masque causal (pour autoregressive)
        mask = torch.tril(torch.ones(seq_len, seq_len))
    else:
        # Masque complet
        mask = torch.ones(seq_len, seq_len)

    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]

def create_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Crée un masque de padding"""
    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=lengths.device).expand(
        batch_size, max_len
    ) < lengths.unsqueeze(1)
    return mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]