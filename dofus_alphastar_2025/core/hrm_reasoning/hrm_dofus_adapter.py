"""
HRM DOFUS Adapter - Interface entre HRM et environnement DOFUS
Convertit l'état de jeu DOFUS en format compatible HRM et vice versa

Version: 1.0.0 - AlphaStar Integration
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

from .hrm_amd_core import HRMAMDModel, HRMOutput, create_hrm_model
from config import config

logger = logging.getLogger(__name__)

class DofusActionType(Enum):
    """Types d'actions possibles dans DOFUS"""
    # Mouvement
    MOVE = "move"
    TELEPORT = "teleport"

    # Combat
    ATTACK = "attack"
    CAST_SPELL = "cast_spell"
    END_TURN = "end_turn"

    # Interaction
    INTERACT = "interact"
    USE_ITEM = "use_item"
    OPEN_INVENTORY = "open_inventory"

    # Communication
    CHAT = "chat"

    # Interface
    OPEN_MAP = "open_map"
    CLOSE_INTERFACE = "close_interface"

    # Meta-actions
    WAIT = "wait"
    THINK = "think"  # Pour System 2 reasoning

@dataclass
class DofusPosition:
    """Position dans DOFUS"""
    x: int
    y: int
    map_id: Optional[int] = None
    cell_id: Optional[int] = None

@dataclass
class DofusEntity:
    """Entité dans le jeu (joueur, monstre, PNJ, objet)"""
    id: int
    name: str
    entity_type: str  # "player", "monster", "npc", "item", "resource"
    position: DofusPosition
    health: Optional[float] = None
    max_health: Optional[float] = None
    level: Optional[int] = None

    # Informations de combat
    action_points: Optional[int] = None
    movement_points: Optional[int] = None

    # État
    is_alive: bool = True
    is_enemy: bool = False
    is_ally: bool = False

@dataclass
class DofusGameState:
    """État complet du jeu DOFUS pour HRM"""
    # Joueur principal
    player_position: DofusPosition
    player_health: float
    player_max_health: float
    player_mana: Optional[float] = None
    player_max_mana: Optional[float] = None
    player_level: int = 1
    player_class: str = "iop"

    # Points de combat
    action_points: int = 0
    max_action_points: int = 6
    movement_points: int = 0
    max_movement_points: int = 3

    # Environnement
    entities: List[DofusEntity] = field(default_factory=list)
    map_id: Optional[int] = None

    # Combat
    in_combat: bool = False
    combat_turn: Optional[int] = None
    is_player_turn: bool = False

    # Interface UI
    interface_elements: Dict[str, Any] = field(default_factory=dict)
    chat_messages: List[str] = field(default_factory=list)

    # Objectifs et quêtes
    active_quest: Optional[str] = None
    quest_objectives: List[str] = field(default_factory=list)

    # Inventaire simplifié
    inventory_items: Dict[str, int] = field(default_factory=dict)

    # Métriques système
    timestamp: float = field(default_factory=time.time)
    fps: float = 30.0
    latency: float = 50.0

    # Contexte économique
    kamas: int = 0  # Monnaie DOFUS

    # Historique récent
    recent_actions: List[str] = field(default_factory=list)
    last_damage_taken: float = 0.0
    last_damage_dealt: float = 0.0

@dataclass
class DofusAction:
    """Action à exécuter dans DOFUS"""
    action_type: DofusActionType
    target_position: Optional[DofusPosition] = None
    target_entity: Optional[DofusEntity] = None
    spell_id: Optional[int] = None
    item_id: Optional[int] = None
    text: Optional[str] = None  # Pour chat
    confidence: float = 1.0

    # Méta-informations pour HRM
    reasoning_path: List[str] = field(default_factory=list)
    expected_outcome: str = ""
    priority: int = 5  # 1-10
    estimated_duration: float = 1.0  # secondes

    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'action en dictionnaire"""
        return {
            "type": self.action_type.value,
            "target_position": {
                "x": self.target_position.x,
                "y": self.target_position.y,
                "map_id": self.target_position.map_id,
                "cell_id": self.target_position.cell_id
            } if self.target_position else None,
            "target_entity_id": self.target_entity.id if self.target_entity else None,
            "spell_id": self.spell_id,
            "item_id": self.item_id,
            "text": self.text,
            "confidence": self.confidence,
            "priority": self.priority,
            "estimated_duration": self.estimated_duration
        }

class DofusStateEncoder(nn.Module):
    """Encodeur d'état DOFUS vers format HRM"""

    def __init__(self, hidden_size: int = 512):
        super().__init__()
        self.hidden_size = hidden_size

        # Encodeurs spécialisés pour différents aspects du jeu
        self.player_encoder = nn.Linear(16, 64)  # Stats joueur
        self.position_encoder = nn.Linear(4, 32)  # Position + map
        self.combat_encoder = nn.Linear(8, 32)   # État combat
        self.entities_encoder = nn.Linear(512, 128)  # Entités environnement
        self.inventory_encoder = nn.Linear(100, 64)  # Inventaire
        self.context_encoder = nn.Linear(32, 64)   # Contexte temporel

        # Fusion des features
        total_features = 64 + 32 + 32 + 128 + 64 + 64  # 384
        self.fusion = nn.Sequential(
            nn.Linear(384, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )

    def forward(self, game_state: DofusGameState) -> torch.Tensor:
        """Encode l'état DOFUS vers tenseur HRM"""
        device = next(self.parameters()).device

        # Encodage stats joueur
        player_stats = torch.tensor([
            game_state.player_health / max(game_state.player_max_health, 1),
            game_state.player_mana / max(game_state.player_max_mana or 100, 1),
            game_state.player_level / 200.0,  # Niveau max ~200
            game_state.action_points / max(game_state.max_action_points, 1),
            game_state.movement_points / max(game_state.max_movement_points, 1),
            1.0 if game_state.in_combat else 0.0,
            1.0 if game_state.is_player_turn else 0.0,
            game_state.kamas / 100000.0,  # Normalisation monnaie
            game_state.fps / 60.0,
            game_state.latency / 1000.0,
            len(game_state.entities) / 50.0,  # Normalisation nb entités
            len(game_state.inventory_items) / 100.0,
            game_state.last_damage_taken / 1000.0,
            game_state.last_damage_dealt / 1000.0,
            len(game_state.recent_actions) / 10.0,
            1.0 if game_state.active_quest else 0.0
        ], dtype=torch.float32, device=device)

        player_encoded = self.player_encoder(player_stats)

        # Encodage position
        position_data = torch.tensor([
            game_state.player_position.x / 1000.0,  # Normalisation coordonnées
            game_state.player_position.y / 1000.0,
            game_state.map_id / 10000.0 if game_state.map_id else 0.0,
            game_state.player_position.cell_id / 500.0 if game_state.player_position.cell_id else 0.0
        ], dtype=torch.float32, device=device)

        position_encoded = self.position_encoder(position_data)

        # Encodage combat
        combat_data = torch.tensor([
            game_state.action_points / 10.0,
            game_state.movement_points / 10.0,
            game_state.combat_turn / 20.0 if game_state.combat_turn else 0.0,
            1.0 if game_state.in_combat else 0.0,
            sum(1 for e in game_state.entities if e.is_enemy and e.is_alive) / 8.0,
            sum(1 for e in game_state.entities if e.is_ally and e.is_alive) / 8.0,
            sum(e.health or 0 for e in game_state.entities if e.is_enemy) / 5000.0,
            sum(e.health or 0 for e in game_state.entities if e.is_ally) / 5000.0
        ], dtype=torch.float32, device=device)

        combat_encoded = self.combat_encoder(combat_data)

        # Encodage entités (simplifié)
        entities_vector = torch.zeros(512, device=device)
        for i, entity in enumerate(game_state.entities[:32]):  # Max 32 entités
            base_idx = i * 16
            if base_idx + 15 < 512:
                entities_vector[base_idx] = entity.position.x / 1000.0
                entities_vector[base_idx + 1] = entity.position.y / 1000.0
                entities_vector[base_idx + 2] = (entity.health or 0) / 1000.0
                entities_vector[base_idx + 3] = (entity.max_health or 0) / 1000.0
                entities_vector[base_idx + 4] = (entity.level or 0) / 200.0
                entities_vector[base_idx + 5] = 1.0 if entity.is_enemy else 0.0
                entities_vector[base_idx + 6] = 1.0 if entity.is_ally else 0.0
                entities_vector[base_idx + 7] = 1.0 if entity.is_alive else 0.0
                # Types d'entités one-hot encodé
                entity_types = ["player", "monster", "npc", "item", "resource"]
                if entity.entity_type in entity_types:
                    entities_vector[base_idx + 8 + entity_types.index(entity.entity_type)] = 1.0

        entities_encoded = self.entities_encoder(entities_vector)

        # Encodage inventaire (top 100 items)
        inventory_vector = torch.zeros(100, device=device)
        for i, (item, count) in enumerate(list(game_state.inventory_items.items())[:100]):
            inventory_vector[i] = min(count, 999) / 999.0

        inventory_encoded = self.inventory_encoder(inventory_vector)

        # Encodage contexte temporel
        context_vector = torch.zeros(32, device=device)

        # Heure cyclique (si disponible)
        import datetime
        now = datetime.datetime.fromtimestamp(game_state.timestamp)
        context_vector[0] = np.sin(2 * np.pi * now.hour / 24)
        context_vector[1] = np.cos(2 * np.pi * now.hour / 24)
        context_vector[2] = np.sin(2 * np.pi * now.minute / 60)
        context_vector[3] = np.cos(2 * np.pi * now.minute / 60)

        # Actions récentes encodées
        for i, action in enumerate(game_state.recent_actions[-10:]):
            context_vector[4 + i] = hash(action) % 1000 / 1000.0  # Hash simpliste

        # Métriques de performance
        context_vector[14] = game_state.fps / 60.0
        context_vector[15] = game_state.latency / 1000.0

        # Objectifs de quête
        context_vector[16] = len(game_state.quest_objectives) / 10.0

        context_encoded = self.context_encoder(context_vector)

        # Fusion finale
        combined = torch.cat([
            player_encoded, position_encoded, combat_encoded,
            entities_encoded, inventory_encoded, context_encoded
        ])

        # Ajouter dimension batch
        state_embedding = self.fusion(combined).unsqueeze(0)

        return state_embedding

class DofusActionDecoder(nn.Module):
    """Décodeur HRM vers actions DOFUS"""

    def __init__(self, hidden_size: int = 512, num_actions: int = 200):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions

        # Têtes de prédiction spécialisées
        self.action_type_head = nn.Linear(hidden_size, len(DofusActionType))
        self.position_head = nn.Linear(hidden_size, 2)  # x, y relative
        self.spell_head = nn.Linear(hidden_size, 50)    # 50 sorts max
        self.item_head = nn.Linear(hidden_size, 100)    # 100 items différents
        self.confidence_head = nn.Linear(hidden_size, 1)

        # Tête de valeur pour RL
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Décode les états cachés HRM vers prédictions d'actions"""
        # Prendre la dernière position de séquence
        last_hidden = hidden_states[:, -1, :]

        # Prédictions
        action_type_logits = self.action_type_head(last_hidden)
        position_delta = self.position_head(last_hidden)
        spell_logits = self.spell_head(last_hidden)
        item_logits = self.item_head(last_hidden)
        confidence = torch.sigmoid(self.confidence_head(last_hidden))
        value = self.value_head(last_hidden)

        return {
            "action_type_logits": action_type_logits,
            "position_delta": position_delta,
            "spell_logits": spell_logits,
            "item_logits": item_logits,
            "confidence": confidence,
            "value": value
        }

class HRMDecisionMaker:
    """Système de prise de décision HRM pour DOFUS"""

    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Modèles
        self.hrm_model = create_hrm_model().to(self.device)
        self.state_encoder = DofusStateEncoder(config.hrm.hidden_size).to(self.device)
        self.action_decoder = DofusActionDecoder(config.hrm.hidden_size).to(self.device)

        # Tokenizer pour HRM (simplifié)
        self.vocab_size = 32000

        # Historique pour mémoire contextuelle
        self.state_history = []
        self.action_history = []
        self.max_history = config.hrm.memory_size

        # Métriques
        self.decision_count = 0
        self.total_reasoning_time = 0.0

        # Charger modèle pré-entraîné si disponible
        if model_path:
            self.load_model(model_path)

        logger.info("HRM Decision Maker initialisé avec succès")

    def decide_action(self,
                     game_state: DofusGameState,
                     use_system_two: bool = True,
                     return_details: bool = False) -> Union[DofusAction, Tuple[DofusAction, HRMOutput]]:
        """Prend une décision basée sur l'état de jeu DOFUS"""

        start_time = time.time()

        with torch.no_grad():
            # Encoder l'état du jeu
            state_embedding = self.state_encoder(game_state)

            # Créer input tokens (simplification pour HRM)
            # Dans une vraie implémentation, on tokenizerait l'état
            input_ids = torch.randint(0, self.vocab_size, (1, 64), device=self.device)

            # Forward pass HRM
            hrm_output = self.hrm_model(
                input_ids=input_ids,
                return_reasoning_details=return_details,
                max_reasoning_steps=config.hrm.max_reasoning_steps if use_system_two else 1
            )

            if return_details:
                hidden_states = hrm_output.reasoning_output
                reasoning_info = hrm_output
            else:
                hidden_states = hrm_output["hidden_states"]
                reasoning_info = None

            # Décoder vers actions DOFUS
            action_predictions = self.action_decoder(hidden_states)

            # Sélection d'action
            action = self._select_action(game_state, action_predictions)

            # Mise à jour historique
            self._update_history(game_state, action)

        decision_time = time.time() - start_time
        self.total_reasoning_time += decision_time
        self.decision_count += 1

        # Logging
        if self.decision_count % 100 == 0:
            avg_time = self.total_reasoning_time / self.decision_count * 1000
            logger.info(f"HRM Decision #{self.decision_count}: avg {avg_time:.2f}ms")

        if return_details and reasoning_info:
            return action, reasoning_info
        return action

    def _select_action(self,
                      game_state: DofusGameState,
                      predictions: Dict[str, torch.Tensor]) -> DofusAction:
        """Sélectionne l'action finale basée sur les prédictions"""

        # Type d'action
        action_type_probs = torch.softmax(predictions["action_type_logits"], dim=-1)
        action_type_idx = torch.argmax(action_type_probs, dim=-1).item()
        action_types = list(DofusActionType)
        action_type = action_types[action_type_idx]

        # Position cible (relative à position actuelle)
        position_delta = predictions["position_delta"][0]
        target_position = DofusPosition(
            x=game_state.player_position.x + int(position_delta[0].item() * 10),
            y=game_state.player_position.y + int(position_delta[1].item() * 10),
            map_id=game_state.map_id
        )

        # Sort/item selon action
        spell_id = None
        item_id = None
        target_entity = None

        if action_type == DofusActionType.CAST_SPELL:
            spell_probs = torch.softmax(predictions["spell_logits"], dim=-1)
            spell_id = torch.argmax(spell_probs, dim=-1).item()

            # Sélectionner entité cible si nécessaire
            enemies = [e for e in game_state.entities if e.is_enemy and e.is_alive]
            if enemies and len(enemies) > 0:
                target_entity = enemies[0]  # Simplification: première entité ennemie

        elif action_type == DofusActionType.USE_ITEM:
            item_probs = torch.softmax(predictions["item_logits"], dim=-1)
            item_id = torch.argmax(item_probs, dim=-1).item()

        # Confiance
        confidence = predictions["confidence"][0].item()

        # Validation et ajustements basés sur contexte
        action = self._validate_action(
            game_state, action_type, target_position,
            target_entity, spell_id, item_id, confidence
        )

        return action

    def _validate_action(self,
                        game_state: DofusGameState,
                        action_type: DofusActionType,
                        target_position: DofusPosition,
                        target_entity: Optional[DofusEntity],
                        spell_id: Optional[int],
                        item_id: Optional[int],
                        confidence: float) -> DofusAction:
        """Valide et ajuste l'action selon le contexte"""

        # Règles de validation contextuelles
        valid_action_type = action_type

        # Si pas en combat, pas d'actions de combat
        if not game_state.in_combat and action_type in [
            DofusActionType.ATTACK, DofusActionType.CAST_SPELL, DofusActionType.END_TURN
        ]:
            valid_action_type = DofusActionType.MOVE

        # Si pas assez de PA, pas de sorts
        if action_type == DofusActionType.CAST_SPELL and game_state.action_points < 2:
            valid_action_type = DofusActionType.END_TURN if game_state.in_combat else DofusActionType.WAIT

        # Si pas de PM, pas de mouvement
        if action_type == DofusActionType.MOVE and game_state.movement_points < 1:
            valid_action_type = DofusActionType.WAIT

        # Santé critique = priorité potion/fuite
        health_ratio = game_state.player_health / max(game_state.player_max_health, 1)
        if health_ratio < 0.2:
            valid_action_type = DofusActionType.USE_ITEM  # Potion
            item_id = 1  # ID potion santé
            confidence = min(confidence + 0.3, 1.0)  # Boost confiance

        return DofusAction(
            action_type=valid_action_type,
            target_position=target_position,
            target_entity=target_entity,
            spell_id=spell_id,
            item_id=item_id,
            confidence=confidence,
            reasoning_path=self._generate_reasoning_path(game_state, valid_action_type),
            expected_outcome=self._predict_outcome(valid_action_type),
            priority=self._calculate_priority(game_state, valid_action_type)
        )

    def _generate_reasoning_path(self, game_state: DofusGameState, action_type: DofusActionType) -> List[str]:
        """Génère le chemin de raisonnement pour explicabilité"""
        reasoning = []

        # Analyse de la situation
        if game_state.in_combat:
            reasoning.append("En combat - analyse tactique")
            if game_state.is_player_turn:
                reasoning.append("Mon tour - sélection d'action")
            else:
                reasoning.append("Tour ennemi - attente")
        else:
            reasoning.append("Exploration - recherche d'objectifs")

        # État du joueur
        health_ratio = game_state.player_health / max(game_state.player_max_health, 1)
        if health_ratio < 0.3:
            reasoning.append(f"Santé critique ({health_ratio:.1%}) - priorité survie")
        elif health_ratio < 0.6:
            reasoning.append(f"Santé modérée ({health_ratio:.1%}) - prudence")

        # Ressources
        if game_state.action_points > 0:
            reasoning.append(f"PA disponibles: {game_state.action_points}")
        if game_state.movement_points > 0:
            reasoning.append(f"PM disponibles: {game_state.movement_points}")

        # Action choisie
        reasoning.append(f"Action décidée: {action_type.value}")

        return reasoning

    def _predict_outcome(self, action_type: DofusActionType) -> str:
        """Prédit le résultat attendu de l'action"""
        outcomes = {
            DofusActionType.MOVE: "Changement de position",
            DofusActionType.ATTACK: "Dommages à l'ennemi",
            DofusActionType.CAST_SPELL: "Effet magique",
            DofusActionType.USE_ITEM: "Utilisation d'objet",
            DofusActionType.END_TURN: "Fin du tour",
            DofusActionType.WAIT: "Attendre évolution",
            DofusActionType.THINK: "Réflexion approfondie"
        }

        return outcomes.get(action_type, "Action contextuelle")

    def _calculate_priority(self, game_state: DofusGameState, action_type: DofusActionType) -> int:
        """Calcule la priorité de l'action (1-10)"""
        base_priority = 5

        # Urgences de survie
        health_ratio = game_state.player_health / max(game_state.player_max_health, 1)
        if health_ratio < 0.2 and action_type == DofusActionType.USE_ITEM:
            return 10  # Urgence maximale

        # Combat actif
        if game_state.in_combat and game_state.is_player_turn:
            if action_type in [DofusActionType.ATTACK, DofusActionType.CAST_SPELL]:
                return 8
            elif action_type == DofusActionType.END_TURN:
                return 6

        # Progression quête
        if game_state.active_quest and action_type in [DofusActionType.MOVE, DofusActionType.INTERACT]:
            return 7

        return base_priority

    def _update_history(self, game_state: DofusGameState, action: DofusAction):
        """Met à jour l'historique pour mémoire contextuelle"""
        # Ajouter état actuel
        self.state_history.append({
            "timestamp": game_state.timestamp,
            "health_ratio": game_state.player_health / max(game_state.player_max_health, 1),
            "in_combat": game_state.in_combat,
            "position": (game_state.player_position.x, game_state.player_position.y),
            "ap": game_state.action_points,
            "mp": game_state.movement_points
        })

        # Ajouter action
        self.action_history.append({
            "timestamp": time.time(),
            "action": action.action_type.value,
            "confidence": action.confidence,
            "priority": action.priority
        })

        # Limiter taille historique
        if len(self.state_history) > self.max_history:
            self.state_history = self.state_history[-self.max_history:]
        if len(self.action_history) > self.max_history:
            self.action_history = self.action_history[-self.max_history:]

    def get_performance_metrics(self) -> Dict[str, float]:
        """Retourne les métriques de performance"""
        if self.decision_count == 0:
            return {"avg_decision_time_ms": 0.0, "total_decisions": 0}

        avg_time_ms = (self.total_reasoning_time / self.decision_count) * 1000

        return {
            "avg_decision_time_ms": avg_time_ms,
            "total_decisions": self.decision_count,
            "total_reasoning_time_s": self.total_reasoning_time,
            "memory_usage_mb": self.hrm_model.get_memory_usage()["total_mb"]
        }

    def save_model(self, path: str):
        """Sauvegarde les modèles"""
        torch.save({
            "hrm_model": self.hrm_model.state_dict(),
            "state_encoder": self.state_encoder.state_dict(),
            "action_decoder": self.action_decoder.state_dict(),
            "decision_count": self.decision_count,
            "config": config.__dict__
        }, path)

        logger.info(f"Modèle HRM DOFUS sauvegardé: {path}")

    def load_model(self, path: str):
        """Charge les modèles pré-entraînés"""
        checkpoint = torch.load(path, map_location=self.device)

        self.hrm_model.load_state_dict(checkpoint["hrm_model"])
        self.state_encoder.load_state_dict(checkpoint["state_encoder"])
        self.action_decoder.load_state_dict(checkpoint["action_decoder"])
        self.decision_count = checkpoint.get("decision_count", 0)

        logger.info(f"Modèle HRM DOFUS chargé: {path}")

# Factory function
def create_hrm_decision_maker(model_path: Optional[str] = None) -> HRMDecisionMaker:
    """Crée un système de décision HRM pour DOFUS"""
    return HRMDecisionMaker(model_path)

# Classe d'agent compatible avec AlphaStar
class DofusHRMAgent:
    """Agent DOFUS utilisant HRM - compatible AlphaStar"""

    def __init__(self, agent_id: str = "hrm_agent", model_path: Optional[str] = None):
        self.agent_id = agent_id
        self.decision_maker = create_hrm_decision_maker(model_path)

        # Métriques agent
        self.games_played = 0
        self.total_score = 0.0
        self.win_rate = 0.0

        logger.info(f"DOFUS HRM Agent {agent_id} créé")

    def act(self, observation: DofusGameState) -> DofusAction:
        """Interface standard pour agent RL"""
        return self.decision_maker.decide_action(observation)

    def update_performance(self, reward: float, done: bool):
        """Met à jour les performances de l'agent"""
        if done:
            self.games_played += 1
            self.total_score += reward

            if self.games_played > 0:
                avg_score = self.total_score / self.games_played
                # Approximation win rate basée sur score moyen
                self.win_rate = max(0.0, min(1.0, (avg_score + 100) / 200))

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'agent"""
        return {
            "agent_id": self.agent_id,
            "games_played": self.games_played,
            "average_score": self.total_score / max(self.games_played, 1),
            "win_rate": self.win_rate,
            "decision_metrics": self.decision_maker.get_performance_metrics()
        }