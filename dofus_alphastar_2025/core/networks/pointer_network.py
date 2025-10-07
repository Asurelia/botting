"""
Pointer Networks - Réseaux de pointage pour cibles spatiales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class PointerNetwork(nn.Module):
    """Réseau de pointage générique"""

    def __init__(self, hidden_size: int, attention_size: int = 128):
        super().__init__()

        self.hidden_size = hidden_size
        self.attention_size = attention_size

        # Projections pour attention
        self.query_proj = nn.Linear(hidden_size, attention_size)
        self.key_proj = nn.Linear(hidden_size, attention_size)
        self.value_proj = nn.Linear(hidden_size, attention_size)

        self.scale = 1.0 / math.sqrt(attention_size)

    def forward(self,
                query: torch.Tensor,
                keys: torch.Tensor,
                values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [B, hidden_size] état de requête
            keys: [B, N, hidden_size] candidats à pointer
            values: [B, N, hidden_size] valeurs (optionnel)
        """
        if values is None:
            values = keys

        batch_size, num_candidates, _ = keys.shape

        # Projections
        q = self.query_proj(query).unsqueeze(1)  # [B, 1, A]
        k = self.key_proj(keys)  # [B, N, A]
        v = self.value_proj(values)  # [B, N, A]

        # Attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, 1, N]
        scores = scores.squeeze(1)  # [B, N]

        return scores

class AttentionPointer(nn.Module):
    """Pointeur avec mécanisme d'attention"""

    def __init__(self, hidden_size: int):
        super().__init__()

        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.pointer_head = nn.Linear(hidden_size, 1)

    def forward(self, query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        # Attention
        attended, _ = self.attention(
            query.unsqueeze(1),  # [B, 1, H]
            candidates,  # [B, N, H]
            candidates
        )

        # Pointer scores
        scores = self.pointer_head(attended).squeeze(-1)  # [B, 1]
        return scores

class SpatialPointer(nn.Module):
    """Pointeur spatial pour grilles/cartes"""

    def __init__(self, hidden_size: int, spatial_dim: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.spatial_dim = spatial_dim

        # Encoder spatial
        self.spatial_encoder = nn.Conv2d(spatial_dim, hidden_size, 3, padding=1)

        # Pointer network
        self.pointer = PointerNetwork(hidden_size)

    def forward(self, query: torch.Tensor, spatial_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, hidden_size] état de requête
            spatial_map: [B, spatial_dim, H, W] carte spatiale
        """
        batch_size = spatial_map.size(0)

        # Encoder la carte spatiale
        encoded_map = self.spatial_encoder(spatial_map)  # [B, hidden_size, H, W]

        # Flatten pour pointer network
        h, w = encoded_map.shape[-2:]
        flat_map = encoded_map.view(batch_size, self.hidden_size, -1).transpose(1, 2)  # [B, H*W, hidden_size]

        # Pointer scores
        scores = self.pointer(query, flat_map)  # [B, H*W]

        # Reshape vers grille
        scores = scores.view(batch_size, h, w)

        return scores.flatten(1)  # [B, H*W] pour compatibilité