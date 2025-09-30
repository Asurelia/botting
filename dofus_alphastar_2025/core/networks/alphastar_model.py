"""
AlphaStar Model - Architecture complète pour DOFUS Unity
Transformer + LSTM + Pointer Network + HRM Integration optimisé AMD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging

from .transformer_core import TransformerEncoder, SpatialTransformer, EntityTransformer
from .lstm_core import AdvancedLSTM
from .pointer_network import PointerNetwork, SpatialPointer
from .value_networks import ValueNetwork, QNetwork
from ..hrm_reasoning import HRMAMDModel, HRMOutput, create_hrm_model
from config import config, get_device, get_dtype

logger = logging.getLogger(__name__)

@dataclass
class AlphaStarOutput:
    """Sortie complète du modèle AlphaStar"""
    # Actions
    action_logits: torch.Tensor
    action_type_logits: torch.Tensor
    target_logits: Optional[torch.Tensor] = None

    # Valeurs pour RL
    value: torch.Tensor
    advantage: Optional[torch.Tensor] = None

    # États cachés
    hidden_states: torch.Tensor
    lstm_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # Attention (pour analyse)
    attention_weights: Optional[torch.Tensor] = None

    # HRM Integration
    hrm_output: Optional[HRMOutput] = None
    reasoning_steps: int = 0

    # Métriques
    confidence: float = 0.0
    processing_time: float = 0.0

class AlphaStarModel(nn.Module):
    """
    Modèle AlphaStar pour DOFUS Unity
    Architecture inspirée de DeepMind AlphaStar avec adaptations DOFUS
    """

    def __init__(self,
                 observation_space_size: int = 512,
                 action_space_size: int = 200,
                 spatial_dim: int = 64,
                 entity_dim: int = 32,
                 use_hrm: bool = False):

        super().__init__()

        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        self.spatial_dim = spatial_dim
        self.entity_dim = entity_dim
        self.use_hrm = use_hrm

        # Configuration depuis config global
        self.hidden_size = config.alphastar.hidden_dim
        self.num_transformer_layers = config.alphastar.transformer_layers
        self.num_lstm_layers = config.alphastar.lstm_layers
        self.num_attention_heads = config.alphastar.attention_heads

        # Device pour optimisations AMD
        self.device = get_device()

        # === ENCODEURS D'OBSERVATION ===

        # Encodeur observations scalaires (HP, MP, niveau, etc.)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(observation_space_size, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(0.1)
        )

        # Encodeur spatial (grille de combat, minimap)
        self.spatial_encoder = SpatialTransformer(
            input_channels=spatial_dim,
            hidden_size=self.hidden_size,
            num_heads=self.num_attention_heads,
            num_layers=4
        )

        # Encodeur entités (joueurs, monstres, PNJs)
        self.entity_encoder = EntityTransformer(
            entity_dim=entity_dim,
            hidden_size=self.hidden_size,
            num_heads=self.num_attention_heads,
            max_entities=config.alphastar.max_entities
        )

        # === CORE TRANSFORMER ===

        self.core_transformer = TransformerEncoder(
            hidden_size=self.hidden_size,
            num_heads=self.num_attention_heads,
            num_layers=self.num_transformer_layers,
            dropout=0.1
        )

        # === LSTM CORE ===

        self.lstm_core = AdvancedLSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            dropout=0.1,
            bidirectional=False
        )

        # === HRM INTEGRATION ===

        if self.use_hrm:
            self.hrm_model = create_hrm_model()
            self.hrm_fusion = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # === TÊTES D'ACTION ===

        # Tête type d'action (move, attack, cast, etc.)
        self.action_type_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, 20)  # 20 types d'actions DOFUS
        )

        # Tête actions spécifiques
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.action_space_size)
        )

        # Pointer network pour cibles spatiales
        self.spatial_pointer = SpatialPointer(
            hidden_size=self.hidden_size,
            spatial_dim=spatial_dim
        )

        # === TÊTES DE VALEUR ===

        self.value_network = ValueNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size // 2
        )

        self.q_network = QNetwork(
            state_size=self.hidden_size,
            action_size=self.action_space_size,
            hidden_size=self.hidden_size
        )

        # === NORMALISATION ET OPTIMISATIONS AMD ===

        self.final_norm = nn.LayerNorm(self.hidden_size)

        # Initialisation des poids
        self.apply(self._init_weights)

        # Optimisations AMD
        if config.amd.use_mixed_precision:
            self = self.half()

        logger.info(f"AlphaStar Model initialisé avec {self.count_parameters():,} paramètres")

    def _init_weights(self, module):
        """Initialisation des poids optimisée pour AlphaStar"""
        if isinstance(module, nn.Linear):
            # Xavier/Glorot initialization
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param.data)

    def count_parameters(self) -> int:
        """Compte le nombre de paramètres entraînables"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self,
                observations: Dict[str, torch.Tensor],
                lstm_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                return_attention: bool = False,
                use_hrm_reasoning: bool = True) -> AlphaStarOutput:
        """
        Forward pass AlphaStar

        Args:
            observations: Dict avec clés 'scalars', 'spatial', 'entities'
            lstm_states: États LSTM précédents (h, c)
            return_attention: Retourner poids d'attention
            use_hrm_reasoning: Utiliser raisonnement HRM
        """

        import time
        start_time = time.time()

        batch_size = next(iter(observations.values())).size(0)

        # === ENCODAGE DES OBSERVATIONS ===

        encoded_features = []

        # Observations scalaires
        if 'scalars' in observations:
            scalar_features = self.scalar_encoder(observations['scalars'])
            encoded_features.append(scalar_features.unsqueeze(1))

        # Observations spatiales
        if 'spatial' in observations:
            spatial_features = self.spatial_encoder(observations['spatial'])
            encoded_features.append(spatial_features)

        # Entités
        if 'entities' in observations:
            entity_features = self.entity_encoder(observations['entities'])
            encoded_features.append(entity_features)

        # Concaténation features
        if encoded_features:
            # Pad to same sequence length if needed
            max_seq_len = max(f.size(1) for f in encoded_features)
            padded_features = []

            for features in encoded_features:
                seq_len = features.size(1)
                if seq_len < max_seq_len:
                    padding = torch.zeros(
                        batch_size, max_seq_len - seq_len, self.hidden_size,
                        device=features.device, dtype=features.dtype
                    )
                    features = torch.cat([features, padding], dim=1)
                padded_features.append(features)

            # Combinaison des features
            combined_features = torch.cat(padded_features, dim=2)
            combined_features = F.linear(
                combined_features,
                torch.randn(combined_features.size(-1), self.hidden_size, device=self.device)
            )
        else:
            # Fallback si pas d'observations
            combined_features = torch.zeros(
                batch_size, 1, self.hidden_size,
                device=self.device, dtype=get_dtype()
            )

        # === CORE TRANSFORMER ===

        transformer_output, attention_weights = self.core_transformer(
            combined_features,
            return_attention=return_attention
        )

        # === LSTM CORE ===

        lstm_output, new_lstm_states = self.lstm_core(transformer_output, lstm_states)

        # Prendre la dernière sortie de séquence
        final_hidden = lstm_output[:, -1, :]  # [batch_size, hidden_size]

        # === HRM REASONING ===

        hrm_output = None
        reasoning_steps = 0

        if self.use_hrm and use_hrm_reasoning:
            try:
                # Créer input tokens pour HRM (simplification)
                vocab_size = 32000
                seq_len = min(64, combined_features.size(1))
                input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)

                hrm_result = self.hrm_model(
                    input_ids=input_ids,
                    return_reasoning_details=True,
                    max_reasoning_steps=config.hrm.max_reasoning_steps
                )

                if isinstance(hrm_result, HRMOutput):
                    hrm_output = hrm_result
                    reasoning_steps = hrm_result.reasoning_steps

                    # Fusion HRM + AlphaStar
                    hrm_hidden = hrm_result.reasoning_output[:, -1, :]  # Dernière position
                    fused_hidden = self.hrm_fusion(torch.cat([final_hidden, hrm_hidden], dim=-1))
                    final_hidden = fused_hidden

            except Exception as e:
                logger.warning(f"Erreur HRM reasoning: {e}")

        # === NORMALISATION FINALE ===

        final_hidden = self.final_norm(final_hidden)

        # === TÊTES D'ACTION ===

        # Type d'action
        action_type_logits = self.action_type_head(final_hidden)

        # Actions spécifiques
        action_logits = self.action_head(final_hidden)

        # Pointer network pour cibles spatiales
        target_logits = None
        if 'spatial' in observations:
            target_logits = self.spatial_pointer(final_hidden, observations['spatial'])

        # === TÊTES DE VALEUR ===

        value = self.value_network(final_hidden)
        q_values = self.q_network(final_hidden, action_logits)

        # Calcul avantage (Dueling DQN style)
        advantage = q_values - q_values.mean(dim=-1, keepdim=True)

        # === MÉTRIQUES ===

        # Confiance basée sur entropie des actions
        action_probs = F.softmax(action_logits, dim=-1)
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1)
        max_entropy = math.log(self.action_space_size)
        confidence = 1.0 - (entropy.mean().item() / max_entropy)

        processing_time = (time.time() - start_time) * 1000  # ms

        return AlphaStarOutput(
            action_logits=action_logits,
            action_type_logits=action_type_logits,
            target_logits=target_logits,
            value=value,
            advantage=advantage,
            hidden_states=final_hidden,
            lstm_states=new_lstm_states,
            attention_weights=attention_weights,
            hrm_output=hrm_output,
            reasoning_steps=reasoning_steps,
            confidence=confidence,
            processing_time=processing_time
        )

    def get_action_distribution(self, output: AlphaStarOutput) -> Dict[str, torch.distributions.Distribution]:
        """Retourne les distributions d'actions pour sampling"""

        distributions = {}

        # Distribution type d'action
        action_type_probs = F.softmax(output.action_type_logits, dim=-1)
        distributions['action_type'] = torch.distributions.Categorical(action_type_probs)

        # Distribution actions
        action_probs = F.softmax(output.action_logits, dim=-1)
        distributions['action'] = torch.distributions.Categorical(action_probs)

        # Distribution cibles spatiales si disponible
        if output.target_logits is not None:
            target_probs = F.softmax(output.target_logits, dim=-1)
            distributions['target'] = torch.distributions.Categorical(target_probs)

        return distributions

    def select_action(self,
                     output: AlphaStarOutput,
                     deterministic: bool = False,
                     temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """Sélectionne des actions depuis la sortie du modèle"""

        actions = {}

        # Type d'action
        action_type_logits = output.action_type_logits / temperature
        if deterministic:
            actions['action_type'] = torch.argmax(action_type_logits, dim=-1)
        else:
            action_type_probs = F.softmax(action_type_logits, dim=-1)
            actions['action_type'] = torch.multinomial(action_type_probs, 1).squeeze(-1)

        # Action spécifique
        action_logits = output.action_logits / temperature
        if deterministic:
            actions['action'] = torch.argmax(action_logits, dim=-1)
        else:
            action_probs = F.softmax(action_logits, dim=-1)
            actions['action'] = torch.multinomial(action_probs, 1).squeeze(-1)

        # Cible spatiale si disponible
        if output.target_logits is not None:
            target_logits = output.target_logits / temperature
            if deterministic:
                actions['target'] = torch.argmax(target_logits, dim=-1)
            else:
                target_probs = F.softmax(target_logits, dim=-1)
                actions['target'] = torch.multinomial(target_probs, 1).squeeze(-1)

        return actions

    def compute_loss(self,
                     output: AlphaStarOutput,
                     actions: Dict[str, torch.Tensor],
                     rewards: torch.Tensor,
                     values_next: Optional[torch.Tensor] = None,
                     done: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Calcule les losses pour l'entraînement"""

        losses = {}

        # Policy loss (action type)
        action_type_loss = F.cross_entropy(
            output.action_type_logits,
            actions['action_type']
        )
        losses['action_type_loss'] = action_type_loss

        # Policy loss (action)
        action_loss = F.cross_entropy(
            output.action_logits,
            actions['action']
        )
        losses['action_loss'] = action_loss

        # Target loss si applicable
        if output.target_logits is not None and 'target' in actions:
            target_loss = F.cross_entropy(
                output.target_logits,
                actions['target']
            )
            losses['target_loss'] = target_loss

        # Value loss
        if values_next is not None and done is not None:
            gamma = config.rl.gamma
            target_values = rewards + gamma * values_next * (1 - done.float())
            value_loss = F.mse_loss(output.value.squeeze(), target_values)
            losses['value_loss'] = value_loss

        # Loss totale
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss

        return losses

class AlphaStarHRMModel(AlphaStarModel):
    """Modèle AlphaStar avec intégration HRM renforcée"""

    def __init__(self, *args, **kwargs):
        # Force l'utilisation de HRM
        kwargs['use_hrm'] = True
        super().__init__(*args, **kwargs)

        # HRM toujours activé
        self.hrm_always_active = True

        # Têtes spécialisées pour reasoning HRM
        self.reasoning_quality_head = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.reasoning_steps_head = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, config.hrm.max_reasoning_steps)
        )

        logger.info("AlphaStar HRM Model initialisé avec raisonnement avancé")

    def forward(self, *args, **kwargs):
        """Forward avec HRM toujours activé"""
        kwargs['use_hrm_reasoning'] = True
        output = super().forward(*args, **kwargs)

        # Prédictions de qualité de raisonnement
        reasoning_quality = self.reasoning_quality_head(output.hidden_states)
        reasoning_steps_pred = self.reasoning_steps_head(output.hidden_states)

        # Mise à jour de l'output
        output.confidence = max(output.confidence, reasoning_quality.mean().item())

        return output

# Factory functions
def create_alphastar_model(observation_space_size: int = 512,
                          action_space_size: int = 200,
                          use_hrm: bool = False) -> AlphaStarModel:
    """Crée un modèle AlphaStar configuré"""
    return AlphaStarModel(
        observation_space_size=observation_space_size,
        action_space_size=action_space_size,
        use_hrm=use_hrm
    )

def create_alphastar_hrm_model(observation_space_size: int = 512,
                              action_space_size: int = 200) -> AlphaStarHRMModel:
    """Crée un modèle AlphaStar avec HRM intégré"""
    return AlphaStarHRMModel(
        observation_space_size=observation_space_size,
        action_space_size=action_space_size
    )