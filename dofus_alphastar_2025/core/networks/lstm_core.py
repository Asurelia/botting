"""
LSTM Core - Composants LSTM pour séquences temporelles AlphaStar
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AdvancedLSTM(nn.Module):
    """LSTM avancé avec optimisations"""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Normalisation
        self.layer_norm = nn.LayerNorm(hidden_size * (2 if bidirectional else 1))

    def forward(self,
                x: torch.Tensor,
                hidden_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        output, new_states = self.lstm(x, hidden_states)
        output = self.layer_norm(output)

        return output, new_states

class BidirectionalLSTM(AdvancedLSTM):
    """LSTM bidirectionnel"""

    def __init__(self, *args, **kwargs):
        kwargs['bidirectional'] = True
        super().__init__(*args, **kwargs)

class StackedLSTM(nn.Module):
    """LSTM empilé avec connexions résiduelles"""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(
                AdvancedLSTM(layer_input_size, hidden_size, 1, dropout)
            )

        # Projections résiduelles si tailles différentes
        self.residual_projs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0 and input_size != hidden_size:
                self.residual_projs.append(nn.Linear(input_size, hidden_size))
            else:
                self.residual_projs.append(nn.Identity())

    def forward(self,
                x: torch.Tensor,
                hidden_states: Optional[list] = None) -> Tuple[torch.Tensor, list]:

        if hidden_states is None:
            hidden_states = [None] * len(self.layers)

        new_hidden_states = []
        current_input = x

        for i, (layer, proj) in enumerate(zip(self.layers, self.residual_projs)):
            # Résiduelle
            residual = proj(current_input)

            # LSTM
            output, new_hidden = layer(current_input, hidden_states[i])

            # Connexion résiduelle
            if residual.shape == output.shape:
                output = output + residual

            current_input = output
            new_hidden_states.append(new_hidden)

        return current_input, new_hidden_states