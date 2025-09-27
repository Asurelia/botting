"""
HRM Core - Module d'intelligence pour TacticalBot
Intègre le Hierarchical Reasoning Model pour une IA adaptive

Auteur: Claude Code
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging

# Import DirectML si disponible
try:
    import torch_directml
    DIRECTML_AVAILABLE = True
except ImportError:
    DIRECTML_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class GameState:
    """État du jeu capturé en temps réel"""
    # Position et mouvement
    player_position: Tuple[int, int]
    player_health: float
    player_mana: float
    player_level: int

    # Environnement
    nearby_entities: List[Dict]
    available_actions: List[str]
    current_quest: Optional[str]
    inventory_state: Dict

    # Contexte temporel
    timestamp: float
    game_time: str

    # Métriques de performance
    fps: float
    latency: float

@dataclass
class HRMDecision:
    """Décision prise par HRM"""
    action: str
    confidence: float
    reasoning_path: List[str]
    expected_outcome: str
    priority: int
    execution_time: float

class HRMGameEncoder(nn.Module):
    """Encode l'état du jeu pour HRM"""

    def __init__(self, state_dim=512, action_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Encodeurs spécialisés
        self.position_encoder = nn.Linear(2, 64)
        self.stats_encoder = nn.Linear(4, 64)  # health, mana, level, fps
        self.inventory_encoder = nn.Linear(100, 128)  # inventaire simplifié
        self.context_encoder = nn.Linear(50, 64)  # contexte temporel

        # Fusion des features
        self.fusion_layer = nn.Sequential(
            nn.Linear(320, state_dim),  # 64+64+128+64 = 320
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(state_dim, state_dim)
        )

        # Encodeur d'actions
        self.action_encoder = nn.Embedding(50, action_dim)  # 50 actions max

    def encode_game_state(self, game_state: GameState) -> torch.Tensor:
        """Encode l'état du jeu en tenseur"""
        device = next(self.parameters()).device

        # Position
        pos = torch.tensor([game_state.player_position[0], game_state.player_position[1]],
                          dtype=torch.float32, device=device)
        pos_encoded = self.position_encoder(pos)

        # Statistiques joueur
        stats = torch.tensor([
            game_state.player_health,
            game_state.player_mana,
            game_state.player_level,
            game_state.fps
        ], dtype=torch.float32, device=device)
        stats_encoded = self.stats_encoder(stats)

        # Inventaire (simplifié)
        inventory_vector = self._encode_inventory(game_state.inventory_state, device)
        inventory_encoded = self.inventory_encoder(inventory_vector)

        # Contexte temporel
        context_vector = self._encode_context(game_state, device)
        context_encoded = self.context_encoder(context_vector)

        # Fusion
        combined = torch.cat([pos_encoded, stats_encoded, inventory_encoded, context_encoded])
        state_embedding = self.fusion_layer(combined)

        return state_embedding.unsqueeze(0)  # Batch dimension

    def _encode_inventory(self, inventory: Dict, device) -> torch.Tensor:
        """Encode l'inventaire en vecteur de taille fixe"""
        # Simplification : prendre les 100 premiers items ou padding
        vector = torch.zeros(100, device=device)

        for i, (item, count) in enumerate(inventory.items()):
            if i >= 100:
                break
            vector[i] = min(count, 999) / 999  # Normalisation

        return vector

    def _encode_context(self, game_state: GameState, device) -> torch.Tensor:
        """Encode le contexte temporel"""
        vector = torch.zeros(50, device=device)

        # Heure de jeu (cyclique)
        if game_state.game_time:
            hour = int(game_state.game_time.split(':')[0]) if ':' in game_state.game_time else 0
            vector[0] = np.sin(2 * np.pi * hour / 24)
            vector[1] = np.cos(2 * np.pi * hour / 24)

        # Nombre d'entités proches
        vector[2] = min(len(game_state.nearby_entities), 50) / 50

        # Quête active
        vector[3] = 1.0 if game_state.current_quest else 0.0

        # Latence normalisée
        vector[4] = min(game_state.latency, 1000) / 1000

        return vector

class HRMReasoningEngine(nn.Module):
    """Moteur de raisonnement hiérarchique pour gaming"""

    def __init__(self, state_dim=512, hidden_dim=512, num_heads=8, num_layers=6):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Couches de raisonnement hiérarchique
        self.high_level_reasoning = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ) for _ in range(num_layers // 2)
        ])

        self.low_level_reasoning = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                batch_first=True
            ) for _ in range(num_layers // 2)
        ])

        # Projection vers les actions
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 50)  # 50 actions possibles
        )

        # Tête de confiance
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, state_embedding: torch.Tensor, action_history: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass du raisonnement hiérarchique

        Returns:
            action_logits: Probabilités des actions
            confidence: Confiance dans la décision
        """
        x = state_embedding

        # Raisonnement haut niveau (stratégique)
        for layer in self.high_level_reasoning:
            x = layer(x)

        # Raisonnement bas niveau (tactique)
        for layer in self.low_level_reasoning:
            x = layer(x)

        # Prédiction d'action
        action_logits = self.action_head(x.squeeze(1))
        confidence = self.confidence_head(x.squeeze(1))

        return action_logits, confidence

class HRMBot:
    """Bot intelligent utilisant HRM pour la prise de décision"""

    def __init__(self, model_path: Optional[str] = None):
        self.device = self._setup_device()
        self.encoder = HRMGameEncoder().to(self.device)
        self.reasoning_engine = HRMReasoningEngine().to(self.device)

        # Actions possibles dans le jeu
        self.action_mapping = {
            0: "move_up", 1: "move_down", 2: "move_left", 3: "move_right",
            4: "attack", 5: "defend", 6: "use_skill_1", 7: "use_skill_2",
            8: "open_inventory", 9: "interact", 10: "cast_spell",
            11: "use_potion", 12: "teleport", 13: "rest", 14: "explore",
            15: "gather_resource", 16: "craft_item", 17: "trade",
            18: "accept_quest", 19: "complete_quest", 20: "wait",
            # Ajoutez plus d'actions selon votre jeu
        }

        # Historique pour l'apprentissage
        self.decision_history = []
        self.performance_metrics = {
            "decisions_made": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "average_confidence": 0.0,
            "learning_rate": 0.001
        }

        # Chargement du modèle pré-entraîné si disponible
        if model_path and Path(model_path).exists():
            self.load_model(model_path)

        logger.info(f"HRM Bot initialisé sur {self.device}")

    def _setup_device(self):
        """Configure le device optimal"""
        if DIRECTML_AVAILABLE:
            try:
                device = torch_directml.device()
                logger.info(f"DirectML activé: {device}")
                return device
            except:
                pass

        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"CUDA activé: {device}")
            return device

        device = torch.device("cpu")
        logger.info(f"Mode CPU: {device}")
        return device

    def decide_action(self, game_state: GameState) -> HRMDecision:
        """Prend une décision basée sur l'état du jeu"""
        start_time = time.time()

        with torch.no_grad():
            # Encoder l'état du jeu
            state_embedding = self.encoder.encode_game_state(game_state)

            # Raisonnement hiérarchique
            action_logits, confidence = self.reasoning_engine(state_embedding)

            # Sélection d'action
            action_probs = F.softmax(action_logits, dim=-1)
            action_idx = torch.argmax(action_probs, dim=-1).item()

            # Filtrer les actions valides
            valid_actions = self._filter_valid_actions(game_state.available_actions, action_idx)

            decision_time = time.time() - start_time

            decision = HRMDecision(
                action=self.action_mapping.get(valid_actions, "wait"),
                confidence=confidence.item(),
                reasoning_path=self._generate_reasoning_path(state_embedding, action_logits),
                expected_outcome=self._predict_outcome(game_state, valid_actions),
                priority=self._calculate_priority(game_state, valid_actions),
                execution_time=decision_time
            )

            # Mettre à jour les métriques
            self._update_metrics(decision)

            return decision

    def _filter_valid_actions(self, available_actions: List[str], preferred_action: int) -> int:
        """Filtre les actions valides dans le contexte actuel"""
        if not available_actions:
            return preferred_action

        # Vérifier si l'action préférée est disponible
        preferred_name = self.action_mapping.get(preferred_action, "wait")
        if preferred_name in available_actions:
            return preferred_action

        # Sinon, prendre la première action disponible qui correspond
        for action_idx, action_name in self.action_mapping.items():
            if action_name in available_actions:
                return action_idx

        return 20  # "wait" par défaut

    def _generate_reasoning_path(self, state_embedding: torch.Tensor, action_logits: torch.Tensor) -> List[str]:
        """Génère le chemin de raisonnement pour explicabilité"""
        top_actions = torch.topk(action_logits, k=3, dim=-1)
        reasoning = []

        for i, (score, action_idx) in enumerate(zip(top_actions.values[0], top_actions.indices[0])):
            action_name = self.action_mapping.get(action_idx.item(), "unknown")
            reasoning.append(f"Option {i+1}: {action_name} (score: {score:.3f})")

        return reasoning

    def _predict_outcome(self, game_state: GameState, action_idx: int) -> str:
        """Prédit le résultat attendu de l'action"""
        action_name = self.action_mapping.get(action_idx, "wait")

        # Logique de prédiction simple (à améliorer avec l'expérience)
        if "move" in action_name:
            return "Changement de position"
        elif "attack" in action_name:
            return "Dommages à l'ennemi"
        elif "use_potion" in action_name:
            return "Restauration de santé"
        elif "quest" in action_name:
            return "Progression de quête"
        else:
            return "Action contextuelle"

    def _calculate_priority(self, game_state: GameState, action_idx: int) -> int:
        """Calcule la priorité de l'action (1-10)"""
        action_name = self.action_mapping.get(action_idx, "wait")

        # Priorités dynamiques basées sur l'état
        if game_state.player_health < 0.3:
            if "potion" in action_name or "rest" in action_name:
                return 10  # Urgence maximale

        if game_state.current_quest and "quest" in action_name:
            return 8  # Priorité élevée pour les quêtes

        if "attack" in action_name and game_state.nearby_entities:
            return 7  # Combat prioritaire si ennemis présents

        return 5  # Priorité moyenne par défaut

    def _update_metrics(self, decision: HRMDecision):
        """Met à jour les métriques de performance"""
        self.performance_metrics["decisions_made"] += 1

        # Moyenne mobile de la confiance
        old_avg = self.performance_metrics["average_confidence"]
        new_avg = (old_avg * 0.9) + (decision.confidence * 0.1)
        self.performance_metrics["average_confidence"] = new_avg

        # Stocker la décision pour analyse
        self.decision_history.append({
            "timestamp": time.time(),
            "action": decision.action,
            "confidence": decision.confidence,
            "priority": decision.priority,
            "execution_time": decision.execution_time
        })

        # Garder seulement les 1000 dernières décisions
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]

    def learn_from_outcome(self, decision: HRMDecision, outcome_success: bool, reward: float = 0.0):
        """Apprend des résultats des actions"""
        if outcome_success:
            self.performance_metrics["successful_actions"] += 1
        else:
            self.performance_metrics["failed_actions"] += 1

        # TODO: Implémenter l'apprentissage par renforcement
        # Ici on pourrait mettre à jour les poids du modèle basé sur le succès
        logger.info(f"Action {decision.action}: {'SUCCÈS' if outcome_success else 'ÉCHEC'}, Reward: {reward}")

    def save_model(self, path: str):
        """Sauvegarde le modèle entraîné"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'reasoning_engine_state_dict': self.reasoning_engine.state_dict(),
            'performance_metrics': self.performance_metrics,
            'action_mapping': self.action_mapping
        }, path)
        logger.info(f"Modèle sauvegardé: {path}")

    def load_model(self, path: str):
        """Charge un modèle pré-entraîné"""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.reasoning_engine.load_state_dict(checkpoint['reasoning_engine_state_dict'])
        self.performance_metrics = checkpoint.get('performance_metrics', self.performance_metrics)
        self.action_mapping = checkpoint.get('action_mapping', self.action_mapping)
        logger.info(f"Modèle chargé: {path}")

    def get_performance_report(self) -> Dict:
        """Génère un rapport de performance"""
        total_actions = self.performance_metrics["successful_actions"] + self.performance_metrics["failed_actions"]
        success_rate = self.performance_metrics["successful_actions"] / max(total_actions, 1)

        return {
            "total_decisions": self.performance_metrics["decisions_made"],
            "success_rate": success_rate,
            "average_confidence": self.performance_metrics["average_confidence"],
            "recent_actions": len(self.decision_history),
            "device_used": str(self.device),
            "directml_available": DIRECTML_AVAILABLE
        }