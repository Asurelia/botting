"""
Intégration HRM AMD avec l'écosystème DOFUS
Adaptation du modèle HRM pour le gaming en temps réel avec optimisations AMD

Version: 2.0.0 - Spécialisé Gaming DOFUS
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json
import time
import logging
from pathlib import Path

from .hrm_amd_core import HRMAMDModel, AMDOptimizationConfig, AMDDeviceManager

logger = logging.getLogger(__name__)

@dataclass
class DofusGameState:
    """État du jeu DOFUS adapté pour HRM"""
    # Position et mouvement
    position: Tuple[int, int]
    map_id: int
    cell_id: int

    # État du personnage
    level: int
    health_percent: float
    energy_percent: float
    experience: int
    kamas: int

    # Combat
    in_combat: bool
    ap: int  # Points d'action
    mp: int  # Points de mouvement
    turn_number: int

    # Inventaire et équipement
    inventory_items: List[Dict]
    equipment: Dict[str, Optional[int]]

    # Contexte temporel
    timestamp: float
    server_time: str

    # Objectifs et quêtes
    active_quests: List[Dict]
    current_objective: Optional[str]

    # Entités visibles
    nearby_monsters: List[Dict]
    nearby_players: List[Dict]
    nearby_npcs: List[Dict]
    nearby_resources: List[Dict]

    # Contexte social
    guild_info: Optional[Dict]
    group_members: List[Dict]

@dataclass
class DofusAction:
    """Action DOFUS avec métadonnées HRM"""
    action_type: str  # movement, combat, interaction, quest, etc.
    parameters: Dict[str, Any]
    priority: int  # 1-10
    expected_duration: float  # en secondes
    prerequisites: List[str]
    confidence: float  # 0.0-1.0
    reasoning_path: List[str]

class DofusStateEncoder(nn.Module):
    """Encodeur d'état de jeu DOFUS pour HRM"""

    def __init__(self, config: AMDOptimizationConfig):
        super().__init__()
        self.config = config

        # Dimensions d'embedding
        self.position_dim = 64
        self.character_dim = 128
        self.inventory_dim = 256
        self.combat_dim = 128
        self.social_dim = 64
        self.temporal_dim = 32

        # Encodeurs spécialisés
        self.position_encoder = nn.Sequential(
            nn.Linear(3, 32),  # position x, y, map_id
            nn.ReLU(),
            nn.Linear(32, self.position_dim)
        )

        self.character_encoder = nn.Sequential(
            nn.Linear(6, 64),  # level, health, energy, experience, kamas, cell_id
            nn.ReLU(),
            nn.Linear(64, self.character_dim)
        )

        self.inventory_encoder = nn.Sequential(
            nn.Linear(200, 128),  # Inventaire simplifié
            nn.ReLU(),
            nn.Linear(128, self.inventory_dim)
        )

        self.combat_encoder = nn.Sequential(
            nn.Linear(4, 64),  # in_combat, ap, mp, turn_number
            nn.ReLU(),
            nn.Linear(64, self.combat_dim)
        )

        self.social_encoder = nn.Sequential(
            nn.Linear(20, 32),  # Contexte social simplifié
            nn.ReLU(),
            nn.Linear(32, self.social_dim)
        )

        self.temporal_encoder = nn.Sequential(
            nn.Linear(10, 16),  # Contexte temporel
            nn.ReLU(),
            nn.Linear(16, self.temporal_dim)
        )

        # Fusion des features
        total_dim = (self.position_dim + self.character_dim + self.inventory_dim +
                    self.combat_dim + self.social_dim + self.temporal_dim)

        self.fusion_layer = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512)  # Sortie compatible avec HRM
        )

    def encode_game_state(self, game_state: DofusGameState) -> torch.Tensor:
        """Encode l'état du jeu DOFUS en tenseur pour HRM"""
        device = next(self.parameters()).device

        # Encodage de la position
        position_data = torch.tensor([
            game_state.position[0] / 1000.0,  # Normalisation
            game_state.position[1] / 1000.0,
            game_state.map_id / 10000.0
        ], dtype=torch.float32, device=device)
        position_encoded = self.position_encoder(position_data)

        # Encodage du personnage
        character_data = torch.tensor([
            game_state.level / 200.0,  # Normalisation niveau max ~200
            game_state.health_percent,
            game_state.energy_percent,
            min(game_state.experience / 1000000, 1.0),  # Cap à 1M exp
            min(game_state.kamas / 10000000, 1.0),  # Cap à 10M kamas
            game_state.cell_id / 600.0  # 560 cellules max par map
        ], dtype=torch.float32, device=device)
        character_encoded = self.character_encoder(character_data)

        # Encodage de l'inventaire
        inventory_vector = self._encode_inventory(game_state.inventory_items, device)
        inventory_encoded = self.inventory_encoder(inventory_vector)

        # Encodage du combat
        combat_data = torch.tensor([
            1.0 if game_state.in_combat else 0.0,
            game_state.ap / 12.0,  # AP max généralement 12
            game_state.mp / 6.0,   # MP max généralement 6
            min(game_state.turn_number / 20.0, 1.0)  # Cap à 20 tours
        ], dtype=torch.float32, device=device)
        combat_encoded = self.combat_encoder(combat_data)

        # Encodage social
        social_vector = self._encode_social_context(game_state, device)
        social_encoded = self.social_encoder(social_vector)

        # Encodage temporel
        temporal_vector = self._encode_temporal_context(game_state, device)
        temporal_encoded = self.temporal_encoder(temporal_vector)

        # Fusion de toutes les features
        combined = torch.cat([
            position_encoded, character_encoded, inventory_encoded,
            combat_encoded, social_encoded, temporal_encoded
        ])

        state_embedding = self.fusion_layer(combined)
        return state_embedding.unsqueeze(0)  # Batch dimension

    def _encode_inventory(self, inventory: List[Dict], device) -> torch.Tensor:
        """Encode l'inventaire en vecteur de taille fixe"""
        vector = torch.zeros(200, device=device)

        for i, item in enumerate(inventory[:100]):  # Max 100 items
            if i * 2 >= 200:
                break

            # Item ID et quantité
            item_id = item.get('id', 0)
            quantity = item.get('quantity', 0)

            vector[i * 2] = min(item_id / 10000.0, 1.0)  # Normalisation ID
            vector[i * 2 + 1] = min(quantity / 1000.0, 1.0)  # Normalisation quantité

        return vector

    def _encode_social_context(self, game_state: DofusGameState, device) -> torch.Tensor:
        """Encode le contexte social"""
        vector = torch.zeros(20, device=device)

        # Groupe
        vector[0] = 1.0 if game_state.group_members else 0.0
        vector[1] = min(len(game_state.group_members) / 8.0, 1.0)  # Max 8 membres

        # Guilde
        vector[2] = 1.0 if game_state.guild_info else 0.0
        if game_state.guild_info:
            vector[3] = min(game_state.guild_info.get('level', 0) / 200.0, 1.0)

        # Entités proches
        vector[4] = min(len(game_state.nearby_players) / 50.0, 1.0)
        vector[5] = min(len(game_state.nearby_monsters) / 20.0, 1.0)
        vector[6] = min(len(game_state.nearby_npcs) / 10.0, 1.0)
        vector[7] = min(len(game_state.nearby_resources) / 30.0, 1.0)

        return vector

    def _encode_temporal_context(self, game_state: DofusGameState, device) -> torch.Tensor:
        """Encode le contexte temporel"""
        vector = torch.zeros(10, device=device)

        # Heure de la journée (cyclique)
        if game_state.server_time and ':' in game_state.server_time:
            hour = int(game_state.server_time.split(':')[0])
            vector[0] = np.sin(2 * np.pi * hour / 24)
            vector[1] = np.cos(2 * np.pi * hour / 24)

        # Jour de la semaine (fictif basé sur timestamp)
        day_of_week = int((game_state.timestamp / 86400) % 7)
        vector[2] = day_of_week / 7.0

        # Quêtes actives
        vector[3] = min(len(game_state.active_quests) / 30.0, 1.0)
        vector[4] = 1.0 if game_state.current_objective else 0.0

        return vector

class DofusActionDecoder(nn.Module):
    """Décodeur d'actions DOFUS à partir des sorties HRM"""

    def __init__(self, config: AMDOptimizationConfig):
        super().__init__()
        self.config = config

        # Paramètres d'action
        self.hidden_size = 512
        self.num_action_types = 50  # Types d'actions possibles

        # Têtes de prédiction spécialisées
        self.action_type_head = nn.Linear(self.hidden_size, self.num_action_types)
        self.movement_head = nn.Linear(self.hidden_size, 560)  # 560 cellules max
        self.combat_head = nn.Linear(self.hidden_size, 100)  # Actions de combat
        self.interaction_head = nn.Linear(self.hidden_size, 50)  # Interactions
        self.priority_head = nn.Linear(self.hidden_size, 10)  # Priorité 1-10
        self.confidence_head = nn.Linear(self.hidden_size, 1)  # Confiance

        # Mapping des actions
        self.action_mapping = {
            0: "move_to_cell", 1: "attack_monster", 2: "cast_spell",
            3: "use_item", 4: "talk_to_npc", 5: "gather_resource",
            6: "open_inventory", 7: "change_map", 8: "accept_quest",
            9: "complete_quest", 10: "join_fight", 11: "flee_fight",
            12: "wait", 13: "teleport", 14: "craft_item",
            15: "sell_item", 16: "buy_item", 17: "bank_deposit",
            18: "bank_withdraw", 19: "trade_with_player",
            # Ajouter plus d'actions selon les besoins
        }

    def decode_action(self, hrm_output: torch.Tensor, game_state: DofusGameState) -> DofusAction:
        """Décode la sortie HRM en action DOFUS"""

        # Moyenner sur la dimension de séquence si nécessaire
        if hrm_output.dim() == 3:
            hrm_output = hrm_output.mean(dim=1)  # [batch, seq, hidden] -> [batch, hidden]

        # Prédictions
        action_logits = self.action_type_head(hrm_output)
        movement_logits = self.movement_head(hrm_output)
        combat_logits = self.combat_head(hrm_output)
        interaction_logits = self.interaction_head(hrm_output)
        priority_logits = self.priority_head(hrm_output)
        confidence = torch.sigmoid(self.confidence_head(hrm_output))

        # Sélection d'action basée sur le contexte
        action_probs = torch.softmax(action_logits, dim=-1)
        action_idx = self._select_contextual_action(action_probs, game_state)

        # Génération des paramètres d'action
        parameters = self._generate_action_parameters(
            action_idx, movement_logits, combat_logits, interaction_logits, game_state
        )

        # Calcul de la priorité
        priority = int(torch.argmax(priority_logits, dim=-1).item()) + 1

        # Génération du chemin de raisonnement
        reasoning_path = self._generate_reasoning_path(action_idx, confidence.item(), game_state)

        return DofusAction(
            action_type=self.action_mapping.get(action_idx, "wait"),
            parameters=parameters,
            priority=priority,
            expected_duration=self._estimate_duration(action_idx, parameters),
            prerequisites=self._get_prerequisites(action_idx, game_state),
            confidence=confidence.item(),
            reasoning_path=reasoning_path
        )

    def _select_contextual_action(self, action_probs: torch.Tensor, game_state: DofusGameState) -> int:
        """Sélection d'action contextuelle"""

        # Masquage des actions impossibles
        mask = self._create_action_mask(game_state)
        masked_probs = action_probs * mask
        masked_probs = masked_probs / (masked_probs.sum() + 1e-8)

        # Sélection avec température pour exploration
        temperature = 0.7
        logits = torch.log(masked_probs + 1e-8) / temperature
        action_idx = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()

        return action_idx

    def _create_action_mask(self, game_state: DofusGameState) -> torch.Tensor:
        """Crée un masque pour les actions valides"""
        mask = torch.ones(self.num_action_types)

        # En combat, certaines actions sont limitées
        if game_state.in_combat:
            # Masquer les actions non-combat
            for action_idx in [6, 7, 13, 14, 15, 16, 17, 18]:  # Inventaire, changement de map, etc.
                if action_idx < self.num_action_types:
                    mask[action_idx] = 0.1  # Réduire plutôt qu'éliminer complètement

        # Si pas d'AP/MP, masquer les actions de combat/mouvement
        if game_state.ap == 0:
            mask[1] = 0.1  # attack_monster
            mask[2] = 0.1  # cast_spell

        if game_state.mp == 0:
            mask[0] = 0.1  # move_to_cell

        # Si pas de monstres proches, réduire l'attaque
        if not game_state.nearby_monsters:
            mask[1] = 0.3  # attack_monster

        return mask

    def _generate_action_parameters(self,
                                   action_idx: int,
                                   movement_logits: torch.Tensor,
                                   combat_logits: torch.Tensor,
                                   interaction_logits: torch.Tensor,
                                   game_state: DofusGameState) -> Dict[str, Any]:
        """Génère les paramètres spécifiques à l'action"""

        action_type = self.action_mapping.get(action_idx, "wait")
        parameters = {}

        if action_type == "move_to_cell":
            # Sélection de la cellule cible
            cell_probs = torch.softmax(movement_logits, dim=-1)
            target_cell = torch.multinomial(cell_probs, 1).item()
            parameters = {"target_cell": target_cell}

        elif action_type == "attack_monster":
            if game_state.nearby_monsters:
                # Sélection du monstre cible
                target_idx = min(len(game_state.nearby_monsters) - 1,
                               torch.argmax(combat_logits[:len(game_state.nearby_monsters)]).item())
                parameters = {"target_monster": game_state.nearby_monsters[target_idx]}

        elif action_type == "cast_spell":
            # Sélection du sort (simplifié)
            spell_idx = torch.argmax(combat_logits[:20]).item()  # 20 sorts max
            parameters = {"spell_id": spell_idx, "target_cell": game_state.cell_id}

        elif action_type == "talk_to_npc":
            if game_state.nearby_npcs:
                npc_idx = min(len(game_state.nearby_npcs) - 1,
                             torch.argmax(interaction_logits[:len(game_state.nearby_npcs)]).item())
                parameters = {"target_npc": game_state.nearby_npcs[npc_idx]}

        elif action_type == "gather_resource":
            if game_state.nearby_resources:
                resource_idx = min(len(game_state.nearby_resources) - 1,
                                 torch.argmax(interaction_logits[:len(game_state.nearby_resources)]).item())
                parameters = {"target_resource": game_state.nearby_resources[resource_idx]}

        return parameters

    def _estimate_duration(self, action_idx: int, parameters: Dict[str, Any]) -> float:
        """Estime la durée d'exécution de l'action"""
        action_type = self.action_mapping.get(action_idx, "wait")

        duration_map = {
            "move_to_cell": 1.0,
            "attack_monster": 2.0,
            "cast_spell": 1.5,
            "use_item": 0.5,
            "talk_to_npc": 3.0,
            "gather_resource": 4.0,
            "change_map": 5.0,
            "wait": 1.0
        }

        return duration_map.get(action_type, 2.0)

    def _get_prerequisites(self, action_idx: int, game_state: DofusGameState) -> List[str]:
        """Détermine les prérequis pour l'action"""
        action_type = self.action_mapping.get(action_idx, "wait")
        prerequisites = []

        if action_type == "attack_monster":
            if not game_state.nearby_monsters:
                prerequisites.append("target_monster_present")
            if game_state.ap == 0:
                prerequisites.append("action_points_available")

        elif action_type == "move_to_cell":
            if game_state.mp == 0:
                prerequisites.append("movement_points_available")

        elif action_type == "cast_spell":
            if game_state.ap == 0:
                prerequisites.append("action_points_available")

        return prerequisites

    def _generate_reasoning_path(self, action_idx: int, confidence: float, game_state: DofusGameState) -> List[str]:
        """Génère le chemin de raisonnement pour l'action"""
        action_type = self.action_mapping.get(action_idx, "wait")

        reasoning = [
            f"Action sélectionnée: {action_type}",
            f"Confiance: {confidence:.3f}",
        ]

        # Contexte spécifique
        if game_state.in_combat:
            reasoning.append(f"En combat (Tour {game_state.turn_number}, AP: {game_state.ap}, MP: {game_state.mp})")

        if game_state.nearby_monsters:
            reasoning.append(f"{len(game_state.nearby_monsters)} monstre(s) détecté(s)")

        if game_state.current_objective:
            reasoning.append(f"Objectif actuel: {game_state.current_objective}")

        return reasoning

class DofusHRMIntegration:
    """Intégration complète HRM-DOFUS pour gaming temps réel"""

    def __init__(self, model_path: Optional[str] = None, config: Optional[AMDOptimizationConfig] = None):
        self.config = config or AMDOptimizationConfig()
        self.device_manager = AMDDeviceManager(self.config)

        # Modèles
        self.hrm_model = HRMAMDModel(self.config).to_device()
        self.state_encoder = DofusStateEncoder(self.config).to(self.device_manager.device)
        self.action_decoder = DofusActionDecoder(self.config).to(self.device_manager.device)

        # Historique et métriques
        self.action_history = []
        self.performance_metrics = {
            "total_decisions": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "average_reasoning_steps": 0.0,
            "average_confidence": 0.0,
            "average_response_time": 0.0
        }

        # Chargement du modèle pré-entraîné
        if model_path and Path(model_path).exists():
            self.load_model(model_path)

        logger.info("Intégration DOFUS-HRM initialisée")

    def decide_action(self, game_state: DofusGameState) -> DofusAction:
        """Prend une décision d'action basée sur l'état du jeu"""
        start_time = time.time()

        try:
            # Encodage de l'état du jeu
            state_embedding = self.state_encoder.encode_game_state(game_state)

            # Conversion en tokens (simplifiée pour demo)
            # Dans un vrai système, il faudrait un tokenizer approprié
            dummy_tokens = torch.zeros(1, 1, dtype=torch.long, device=self.device_manager.device)

            # Raisonnement HRM
            with torch.no_grad():
                hrm_outputs = self.hrm_model(
                    input_ids=dummy_tokens,
                    max_reasoning_steps=8
                )

            # Utilisation de l'embedding d'état comme hidden state
            hrm_output = state_embedding  # Simplification

            # Décodage en action DOFUS
            action = self.action_decoder.decode_action(hrm_output, game_state)

            # Métriques
            response_time = time.time() - start_time
            self._update_metrics(action, hrm_outputs, response_time)

            # Historique
            self.action_history.append({
                "timestamp": time.time(),
                "game_state_summary": self._summarize_game_state(game_state),
                "action": action,
                "reasoning_steps": hrm_outputs.get("reasoning_steps", 0),
                "response_time": response_time
            })

            return action

        except Exception as e:
            logger.error(f"Erreur décision d'action: {e}")
            # Action de fallback
            return DofusAction(
                action_type="wait",
                parameters={},
                priority=1,
                expected_duration=1.0,
                prerequisites=[],
                confidence=0.1,
                reasoning_path=[f"Erreur: {str(e)}", "Action de fallback: wait"]
            )

    def _update_metrics(self, action: DofusAction, hrm_outputs: Dict, response_time: float):
        """Met à jour les métriques de performance"""
        self.performance_metrics["total_decisions"] += 1

        # Moyenne mobile
        old_avg_conf = self.performance_metrics["average_confidence"]
        new_avg_conf = (old_avg_conf * 0.9) + (action.confidence * 0.1)
        self.performance_metrics["average_confidence"] = new_avg_conf

        old_avg_steps = self.performance_metrics["average_reasoning_steps"]
        new_avg_steps = (old_avg_steps * 0.9) + (hrm_outputs.get("reasoning_steps", 0) * 0.1)
        self.performance_metrics["average_reasoning_steps"] = new_avg_steps

        old_avg_time = self.performance_metrics["average_response_time"]
        new_avg_time = (old_avg_time * 0.9) + (response_time * 0.1)
        self.performance_metrics["average_response_time"] = new_avg_time

    def _summarize_game_state(self, game_state: DofusGameState) -> Dict[str, Any]:
        """Résumé de l'état du jeu pour l'historique"""
        return {
            "position": game_state.position,
            "level": game_state.level,
            "health_percent": game_state.health_percent,
            "in_combat": game_state.in_combat,
            "nearby_entities": {
                "monsters": len(game_state.nearby_monsters),
                "players": len(game_state.nearby_players),
                "npcs": len(game_state.nearby_npcs),
                "resources": len(game_state.nearby_resources)
            }
        }

    def learn_from_outcome(self, action: DofusAction, success: bool, reward: float = 0.0):
        """Apprentissage à partir du résultat de l'action"""
        if success:
            self.performance_metrics["successful_actions"] += 1
        else:
            self.performance_metrics["failed_actions"] += 1

        # TODO: Implémenter l'apprentissage par renforcement
        # Mise à jour des poids du modèle basée sur le succès
        logger.debug(f"Action {action.action_type}: {'SUCCESS' if success else 'FAILURE'}, Reward: {reward}")

    def save_model(self, path: str):
        """Sauvegarde le modèle complet"""
        checkpoint = {
            'hrm_model_state_dict': self.hrm_model.state_dict(),
            'state_encoder_state_dict': self.state_encoder.state_dict(),
            'action_decoder_state_dict': self.action_decoder.state_dict(),
            'config': self.config,
            'performance_metrics': self.performance_metrics,
            'action_history': self.action_history[-1000:]  # Dernières 1000 actions
        }
        torch.save(checkpoint, path)
        logger.info(f"Modèle DOFUS-HRM sauvegardé: {path}")

    def load_model(self, path: str):
        """Charge un modèle pré-entraîné"""
        checkpoint = torch.load(path, map_location=self.device_manager.device)

        self.hrm_model.load_state_dict(checkpoint['hrm_model_state_dict'])
        self.state_encoder.load_state_dict(checkpoint['state_encoder_state_dict'])
        self.action_decoder.load_state_dict(checkpoint['action_decoder_state_dict'])

        self.performance_metrics = checkpoint.get('performance_metrics', self.performance_metrics)
        self.action_history = checkpoint.get('action_history', [])

        logger.info(f"Modèle DOFUS-HRM chargé: {path}")

    def get_performance_report(self) -> Dict[str, Any]:
        """Génère un rapport de performance détaillé"""
        success_rate = 0.0
        if self.performance_metrics["successful_actions"] + self.performance_metrics["failed_actions"] > 0:
            total = self.performance_metrics["successful_actions"] + self.performance_metrics["failed_actions"]
            success_rate = self.performance_metrics["successful_actions"] / total

        return {
            "performance": {
                "total_decisions": self.performance_metrics["total_decisions"],
                "success_rate": success_rate,
                "average_confidence": self.performance_metrics["average_confidence"],
                "average_reasoning_steps": self.performance_metrics["average_reasoning_steps"],
                "average_response_time_ms": self.performance_metrics["average_response_time"] * 1000
            },
            "hardware": {
                "device": str(self.device_manager.device),
                "device_properties": self.device_manager.device_props,
                "memory_usage": self.hrm_model.get_memory_usage()
            },
            "recent_actions": len(self.action_history),
            "last_action": self.action_history[-1] if self.action_history else None
        }