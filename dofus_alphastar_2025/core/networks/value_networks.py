"""
Value Networks - Réseaux de valeur pour RL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNetwork(nn.Module):
    """Réseau de valeur d'état"""

    def __init__(self, input_size: int, hidden_size: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class AdvantageNetwork(nn.Module):
    """Réseau d'avantage pour Dueling DQN"""

    def __init__(self, input_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()

        self.action_size = action_size

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        advantages = self.network(x)
        # Centrer les avantages
        return advantages - advantages.mean(dim=-1, keepdim=True)

class QNetwork(nn.Module):
    """Q-Network avec architecture Dueling"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()

        # Feature extractor commun
        self.features = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Têtes séparées
        self.value_head = ValueNetwork(hidden_size, hidden_size // 2)
        self.advantage_head = AdvantageNetwork(hidden_size, action_size, hidden_size // 2)

    def forward(self, state: torch.Tensor, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.features(state)

        value = self.value_head(features)
        advantages = self.advantage_head(features)

        # Q-values = V(s) + A(s,a)
        q_values = value + advantages

        if actions is not None:
            # Sélectionner Q-values pour actions spécifiques
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        return q_values