"""
HRM-DOFUS Integration - Intégration du système HRM existant avec Knowledge Base DOFUS
Combine l'intelligence HRM existante avec les données spécialisées DOFUS Unity
Approche 100% vision - Bridge intelligent entre les systèmes
"""

import sys
import os
from pathlib import Path
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json

# Ajout du chemin vers HRM existant
hrm_path = Path(__file__).parent.parent.parent / "core" / "hrm_intelligence"
sys.path.insert(0, str(hrm_path))

# Ajout du chemin vers Knowledge Base
kb_path = Path(__file__).parent.parent / "knowledge_base"
sys.path.insert(0, str(kb_path))

try:
    # Import de notre Knowledge Base DOFUS
    from knowledge_integration import get_knowledge_base, GameContext
    from spells_database import DofusClass
    KB_AVAILABLE = True

except ImportError as e:
    KB_AVAILABLE = False
    # Définition fallback pour DofusClass
    from enum import Enum
    class DofusClass(Enum):
        IOPS = "iops"
        CRA = "cra"
        SADI = "sadida"
        ENIRIPSA = "eniripsa"
        ECAFLIP = "ecaflip"
        ENUTROF = "enutrof"
        SRAM = "sram"
        XELOR = "xelor"
        PANDAWA = "pandawa"
        ROUBLARD = "roublard"
        ZOBAL = "zobal"
        STEAMER = "steamer"
        ELIOTROPE = "eliotrope"
        HUPPERMAGE = "huppermage"
        OUGINAK = "ouginak"
        FORGELANCE = "forgelance"

    logger = logging.getLogger(__name__)
    logger.error(f"Erreur import KB: {e}")

try:
    # Import du système HRM existant
    from hrm_core import HRMBot, GameState as HRMGameState, HRMDecision
    from amd_gpu_optimizer import AMDGPUOptimizer
    from adaptive_learner import AdaptiveLearner
    from intelligent_decision_maker import IntelligentDecisionMaker
    HRM_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Imports HRM réussis")

except ImportError as e:
    HRM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error(f"Erreur import HRM: {e}")

    # Définition des classes mock pour fonctionnement sans HRM
    class HRMGameState:
        """Mock HRM GameState pour fonctionnement autonome"""
        def __init__(self, player_position=None, player_health=1.0, player_mana=1.0,
                     player_level=150, nearby_entities=None, available_actions=None,
                     combat_context=None, **kwargs):
            self.player_position = player_position or (0, 0)
            self.player_health = player_health
            self.player_mana = player_mana
            self.player_level = player_level
            self.nearby_entities = nearby_entities or []
            self.available_actions = available_actions or []
            self.combat_context = combat_context or {}

    class HRMDecision:
        """Mock HRM Decision"""
        def __init__(self, action="wait", confidence=0.5, reasoning=None):
            self.action = action
            self.confidence = confidence
            self.reasoning = reasoning or []

    class HRMBot:
        """Mock HRM Bot pour fonctionnement autonome"""
        def __init__(self, **kwargs):
            self.initialized = True

        def decide_action(self, state):
            """Décision simple basée sur l'état"""
            if hasattr(state, 'combat_context') and state.combat_context:
                return HRMDecision("spell_cast", 0.7, ["Action de combat"])
            return HRMDecision("wait", 0.5, ["Action par défaut"])

@dataclass
class DofusGameState:
    """État de jeu DOFUS unifié pour HRM"""
    # État joueur DOFUS spécifique
    player_class: DofusClass
    player_level: int
    current_server: str
    current_map_id: Optional[int]

    # Combat DOFUS
    in_combat: bool
    available_ap: int
    available_mp: int
    current_health: int
    max_health: int

    # Position tactique
    player_position: Tuple[int, int]
    enemies_positions: List[Tuple[int, int]]
    allies_positions: List[Tuple[int, int]]

    # Interface DOFUS
    interface_elements_visible: List[str]
    spell_cooldowns: Dict[str, int]
    inventory_items: Dict[str, int]

    # Contexte économique
    current_kamas: int
    market_opportunities: List[str]

    # Métadonnées
    timestamp: float
    screenshot_path: Optional[str]

class ActionType(Enum):
    """Types d'actions DOFUS disponibles"""
    SPELL = "spell"
    MOVE = "move"
    INTERACT = "interact"
    ECONOMY = "economy"
    WAIT = "wait"
    FLEE = "flee"

@dataclass
class DofusAction:
    """Action DOFUS enrichie avec contexte HRM"""
    action_type: ActionType
    target_pos: Optional[Tuple[int, int]] = None
    spell_id: Optional[int] = None
    spell_name: Optional[str] = None
    confidence: float = 0.0
    reasoning: List[str] = None
    expected_outcome: str = ""
    tactical_priority: int = 1
    economic_value: float = 0.0

    def __post_init__(self):
        if self.reasoning is None:
            self.reasoning = []

class HRMDofusGameEncoder:
    """Encodeur spécialisé pour états de jeu DOFUS"""

    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.hrm_encoder = None
        if HRM_AVAILABLE:
            from hrm_core import HRMGameEncoder
            self.hrm_encoder = HRMGameEncoder(state_dim=768, action_dim=256)

    def encode_dofus_state(self, dofus_state: DofusGameState) -> HRMGameState:
        """Convertit un état DOFUS en état HRM compatible"""

        # Conversion vers format HRM
        hrm_state = HRMGameState(
            player_position=dofus_state.player_position,
            player_health=float(dofus_state.current_health / max(dofus_state.max_health, 1)),
            player_mana=float(dofus_state.available_ap / 12.0),  # Normalisation PA
            player_level=dofus_state.player_level,

            # Environnement tactique
            nearby_entities=self._encode_nearby_entities(dofus_state),
            available_actions=self._get_available_dofus_actions(dofus_state),
            current_quest=self._get_current_objective(dofus_state),
            inventory_state=dofus_state.inventory_items,

            # Contexte temporel
            timestamp=dofus_state.timestamp,
            game_time=time.strftime("%H:%M:%S"),

            # Métriques
            fps=60.0,  # Assumé stable pour DOFUS
            latency=50.0  # Latence moyenne
        )

        return hrm_state

    def _encode_nearby_entities(self, dofus_state: DofusGameState) -> List[Dict]:
        """Encode les entités proches en format HRM"""
        entities = []

        # Ennemis
        for i, pos in enumerate(dofus_state.enemies_positions):
            entities.append({
                "type": "enemy",
                "position": pos,
                "distance": self._calculate_distance(dofus_state.player_position, pos),
                "threat_level": self._calculate_threat_level(dofus_state, pos),
                "id": f"enemy_{i}"
            })

        # Alliés
        for i, pos in enumerate(dofus_state.allies_positions):
            entities.append({
                "type": "ally",
                "position": pos,
                "distance": self._calculate_distance(dofus_state.player_position, pos),
                "support_value": 0.8,
                "id": f"ally_{i}"
            })

        return entities

    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calcule la distance tactique DOFUS (hexagonale)"""
        x1, y1 = pos1
        x2, y2 = pos2

        # Distance hexagonale approximative
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        return dx + max(0, (dy - dx) // 2)

    def _calculate_threat_level(self, dofus_state: DofusGameState, enemy_pos: Tuple[int, int]) -> float:
        """Calcule le niveau de menace d'un ennemi"""
        distance = self._calculate_distance(dofus_state.player_position, enemy_pos)

        # Plus proche = plus menaçant
        base_threat = max(0.1, 1.0 - (distance / 10.0))

        # Modification selon les PA disponibles de l'ennemi (estimé)
        if distance <= 3:  # Portée d'attaque typique
            base_threat *= 1.5

        return min(1.0, base_threat)

    def _get_available_dofus_actions(self, dofus_state: DofusGameState) -> List[str]:
        """Retourne les actions DOFUS disponibles selon le contexte"""
        actions = []

        if dofus_state.in_combat:
            # Actions de combat
            if dofus_state.available_ap > 0:
                actions.extend(["attack", "cast_spell", "move_tactical"])
            if dofus_state.available_mp > 0:
                actions.extend(["move", "reposition"])
            actions.extend(["defend", "wait", "end_turn"])
        else:
            # Actions hors combat
            actions.extend([
                "move", "interact", "open_inventory", "use_zaap",
                "gather_resource", "accept_quest", "talk_npc",
                "open_map", "check_market"
            ])

        # Actions universelles
        actions.extend(["use_potion", "check_spells"])

        return actions

    def _get_current_objective(self, dofus_state: DofusGameState) -> Optional[str]:
        """Détermine l'objectif actuel basé sur le contexte"""
        if dofus_state.in_combat:
            if len(dofus_state.enemies_positions) > 0:
                return "eliminate_enemies"
            else:
                return "survive_combat"

        # Objectifs hors combat basés sur le niveau
        if dofus_state.player_level < 50:
            return "level_up_quests"
        elif dofus_state.player_level < 100:
            return "equipment_farming"
        else:
            return "endgame_content"

class DofusIntelligentDecisionMaker:
    """Gestionnaire de décisions intelligent pour DOFUS avec HRM"""

    def __init__(self):
        self.knowledge_base = get_knowledge_base() if HRM_AVAILABLE else None
        self.hrm_bot = None
        self.gpu_optimizer = None
        self.adaptive_learner = None
        self.game_encoder = HRMDofusGameEncoder(self.knowledge_base)

        # Configuration spécialisée DOFUS
        self.dofus_config = {
            "combat_decision_weight": 0.8,
            "economic_decision_weight": 0.6,
            "exploration_decision_weight": 0.4,
            "learning_rate": 0.001,
            "confidence_threshold": 0.7
        }

        if HRM_AVAILABLE:
            self._initialize_hrm_components()

    def _initialize_hrm_components(self):
        """Initialise les composants HRM existants"""
        try:
            # Optimiseur GPU AMD
            self.gpu_optimizer = AMDGPUOptimizer()
            gpu_initialized = self.gpu_optimizer.initialize()

            # Bot HRM principal
            model_path = Path("G:/Botting/models/test_hrm_model.pth")
            self.hrm_bot = HRMBot(str(model_path) if model_path.exists() else None)

            # Optimisation GPU si disponible
            if gpu_initialized:
                self.hrm_bot.encoder = self.gpu_optimizer.optimize_model(self.hrm_bot.encoder)
                self.hrm_bot.reasoning_engine = self.gpu_optimizer.optimize_model(self.hrm_bot.reasoning_engine)

            # Apprentissage adaptatif
            self.adaptive_learner = AdaptiveLearner()

            logger.info("Composants HRM initialisés avec succès")

        except Exception as e:
            logger.error(f"Erreur initialisation HRM: {e}")
            self.hrm_bot = None

    def decide_dofus_action(self, dofus_state: DofusGameState) -> DofusAction:
        """Prend une décision DOFUS intelligente avec HRM + Knowledge Base"""

        if not HRM_AVAILABLE or not self.hrm_bot:
            return self._fallback_decision(dofus_state)

        try:
            # 1. Mise à jour du contexte Knowledge Base
            self._update_game_context(dofus_state)

            # 2. Conversion vers format HRM
            hrm_state = self.game_encoder.encode_dofus_state(dofus_state)

            # 3. Décision HRM de base
            hrm_decision = self.hrm_bot.decide_action(hrm_state)

            # 4. Enrichissement avec Knowledge Base DOFUS
            enriched_action = self._enrich_with_dofus_knowledge(
                hrm_decision, dofus_state
            )

            # 5. Validation tactique
            validated_action = self._validate_tactical_action(
                enriched_action, dofus_state
            )

            # 6. Apprentissage adaptatif
            self._update_learning(dofus_state, validated_action)

            return validated_action

        except Exception as e:
            logger.error(f"Erreur décision HRM-DOFUS: {e}")
            return self._fallback_decision(dofus_state)

    def _update_game_context(self, dofus_state: DofusGameState):
        """Met à jour le contexte de la Knowledge Base"""
        if not self.knowledge_base:
            return

        context = GameContext(
            player_class=dofus_state.player_class,
            player_level=dofus_state.player_level,
            current_server=dofus_state.current_server,
            current_map_id=dofus_state.current_map_id,
            available_ap=dofus_state.available_ap,
            available_mp=dofus_state.available_mp,
            distance_to_target=self._calculate_nearest_enemy_distance(dofus_state),
            in_combat=dofus_state.in_combat
        )

        self.knowledge_base.update_game_context(context)

    def _calculate_nearest_enemy_distance(self, dofus_state: DofusGameState) -> int:
        """Calcule la distance au plus proche ennemi"""
        if not dofus_state.enemies_positions:
            return 10  # Distance par défaut

        min_distance = float('inf')
        for enemy_pos in dofus_state.enemies_positions:
            distance = self.game_encoder._calculate_distance(
                dofus_state.player_position, enemy_pos
            )
            min_distance = min(min_distance, distance)

        return int(min_distance)

    def _enrich_with_dofus_knowledge(self, hrm_decision: HRMDecision,
                                   dofus_state: DofusGameState) -> DofusAction:
        """Enrichit la décision HRM avec les connaissances DOFUS"""

        # Décision de base
        action = DofusAction(
            action_type=self._map_hrm_to_dofus_action(hrm_decision.action),
            target_pos=None,
            spell_id=None,
            spell_name=None,
            confidence=hrm_decision.confidence,
            reasoning=hrm_decision.reasoning_path,
            expected_outcome=hrm_decision.expected_outcome,
            tactical_priority=hrm_decision.priority
        )

        # Enrichissement selon le type d'action
        if dofus_state.in_combat and action.action_type in ["attack", "cast_spell"]:
            action = self._enrich_combat_action(action, dofus_state)
        elif action.action_type in ["gather_resource", "check_market"]:
            action = self._enrich_economic_action(action, dofus_state)

        return action

    def _enrich_combat_action(self, action: DofusAction,
                            dofus_state: DofusGameState) -> DofusAction:
        """Enrichit les actions de combat avec les sorts optimaux"""
        if not self.knowledge_base:
            return action

        try:
            # Requête sorts optimaux
            optimal_spells_result = self.knowledge_base.query_optimal_spells(
                ap_available=dofus_state.available_ap,
                distance=self._calculate_nearest_enemy_distance(dofus_state),
                class_type=dofus_state.player_class
            )

            if optimal_spells_result.success and optimal_spells_result.data:
                best_spell = optimal_spells_result.data[0]
                action.spell_id = best_spell.id
                action.spell_name = best_spell.name
                action.action_type = "cast_spell"
                action.reasoning.extend([
                    f"Sort optimal sélectionné: {best_spell.name}",
                    f"Coût PA: {best_spell.cost.ap_cost}",
                    f"Portée: {best_spell.range_info.min_range}-{best_spell.range_info.max_range}"
                ])

                # Cible prioritaire
                if dofus_state.enemies_positions:
                    action.target_pos = dofus_state.enemies_positions[0]  # Plus proche ennemi

                action.confidence *= 1.2  # Boost confiance avec Knowledge Base

        except Exception as e:
            logger.warning(f"Erreur enrichissement combat: {e}")

        return action

    def _enrich_economic_action(self, action: DofusAction,
                              dofus_state: DofusGameState) -> DofusAction:
        """Enrichit les actions économiques avec les opportunités marché"""
        if not self.knowledge_base:
            return action

        try:
            # Requête opportunités marché
            market_result = self.knowledge_base.query_market_opportunities()

            if market_result.success and market_result.data:
                best_opportunity = market_result.data[0]
                action.economic_value = best_opportunity.profit_potential
                action.reasoning.extend([
                    f"Opportunité marché: {best_opportunity.item_name}",
                    f"Profit potentiel: {best_opportunity.profit_potential} kamas",
                    f"Action recommandée: {best_opportunity.action}"
                ])

                # Modification de l'action selon l'opportunité
                if best_opportunity.action == "buy":
                    action.action_type = "check_market"
                elif best_opportunity.action == "sell":
                    action.action_type = "open_inventory"

        except Exception as e:
            logger.warning(f"Erreur enrichissement économique: {e}")

        return action

    def _validate_tactical_action(self, action: DofusAction,
                                dofus_state: DofusGameState) -> DofusAction:
        """Valide l'action tactiquement selon le contexte DOFUS"""

        # Validation des PA/PM
        if action.action_type == "cast_spell" and action.spell_name:
            # Vérifier si assez de PA (Knowledge Base pourrait avoir cette info)
            estimated_ap_cost = 4  # Coût moyen
            if dofus_state.available_ap < estimated_ap_cost:
                action.action_type = "wait"
                action.reasoning.append("PA insuffisants pour le sort")
                action.confidence *= 0.5

        # Validation de position
        if action.target_pos and dofus_state.in_combat:
            target_distance = self.game_encoder._calculate_distance(
                dofus_state.player_position, action.target_pos
            )
            if target_distance > 6:  # Portée maximum typique
                action.reasoning.append("Cible trop éloignée")
                action.confidence *= 0.7

        # Validation de sécurité
        if dofus_state.current_health / max(dofus_state.max_health, 1) < 0.3:
            if action.action_type not in ["use_potion", "defend", "flee"]:
                action.tactical_priority = max(action.tactical_priority, 9)
                action.reasoning.append("Santé critique - priorité défensive")

        return action

    def _update_learning(self, dofus_state: DofusGameState, action: DofusAction):
        """Met à jour l'apprentissage adaptatif"""
        if not self.adaptive_learner:
            return

        try:
            # Création d'une expérience d'apprentissage
            experience = {
                "state": asdict(dofus_state),
                "action": asdict(action),
                "context": {
                    "in_combat": dofus_state.in_combat,
                    "health_ratio": dofus_state.current_health / max(dofus_state.max_health, 1),
                    "enemy_count": len(dofus_state.enemies_positions),
                    "ap_available": dofus_state.available_ap
                },
                "timestamp": time.time()
            }

            # Ajout à l'apprentissage adaptatif
            self.adaptive_learner.add_experience(experience)

        except Exception as e:
            logger.warning(f"Erreur mise à jour apprentissage: {e}")

    def _map_hrm_to_dofus_action(self, hrm_action: str) -> str:
        """Mappe les actions HRM vers les actions DOFUS"""
        mapping = {
            "attack": "attack",
            "defend": "defend",
            "use_skill_1": "cast_spell",
            "use_skill_2": "cast_spell",
            "move_up": "move",
            "move_down": "move",
            "move_left": "move",
            "move_right": "move",
            "use_potion": "use_potion",
            "interact": "interact",
            "open_inventory": "open_inventory",
            "wait": "wait",
            "explore": "explore",
            "gather_resource": "gather_resource",
            "cast_spell": "cast_spell",
            "rest": "rest"
        }

        return mapping.get(hrm_action, "wait")

    def _fallback_decision(self, dofus_state: DofusGameState) -> DofusAction:
        """Décision de fallback si HRM indisponible"""
        if dofus_state.in_combat and dofus_state.enemies_positions:
            return DofusAction(
                action_type="attack",
                target_pos=dofus_state.enemies_positions[0],
                spell_id=None,
                spell_name=None,
                confidence=0.5,
                reasoning=["Fallback: Attaque de base"],
                expected_outcome="Dommages à l'ennemi",
                tactical_priority=5
            )
        else:
            return DofusAction(
                action_type="wait",
                target_pos=None,
                spell_id=None,
                spell_name=None,
                confidence=0.8,
                reasoning=["Fallback: Attente sécurisée"],
                expected_outcome="Pas d'action",
                tactical_priority=1
            )

    def get_performance_report(self) -> Dict[str, Any]:
        """Génère un rapport de performance HRM-DOFUS"""
        report = {
            "hrm_available": HRM_AVAILABLE,
            "knowledge_base_available": self.knowledge_base is not None,
            "gpu_optimization_active": False,
            "adaptive_learning_active": self.adaptive_learner is not None,
            "decisions_made": 0,
            "average_confidence": 0.0
        }

        if self.gpu_optimizer:
            gpu_report = self.gpu_optimizer.get_optimization_report()
            report["gpu_optimization_active"] = gpu_report["status"]["optimization_active"]
            report["gpu_model"] = gpu_report["gpu_capabilities"]["model"]

        if self.hrm_bot:
            hrm_report = self.hrm_bot.get_performance_report()
            report.update(hrm_report)

        return report

# Factory function pour faciliter l'utilisation
def create_dofus_hrm_integration() -> DofusIntelligentDecisionMaker:
    """Crée une instance du système intégré HRM-DOFUS"""
    return DofusIntelligentDecisionMaker()

# Test du module
if __name__ == "__main__":
    # Configuration logging
    logging.basicConfig(level=logging.INFO)

    # Test d'intégration
    print("Test HRM-DOFUS Integration...")

    decision_maker = create_dofus_hrm_integration()

    # État de test
    test_state = DofusGameState(
        player_class=DofusClass.IOPS,
        player_level=100,
        current_server="Julith",
        current_map_id=1,
        in_combat=True,
        available_ap=6,
        available_mp=3,
        current_health=150,
        max_health=200,
        player_position=(5, 8),
        enemies_positions=[(7, 9), (3, 6)],
        allies_positions=[(4, 7)],
        interface_elements_visible=["spells_bar", "health_bar"],
        spell_cooldowns={"Compulsion": 0, "Épée du Jugement": 2},
        inventory_items={"Pain": 5, "Potion de Santé": 3},
        current_kamas=50000,
        market_opportunities=["Fer en promotion"],
        timestamp=time.time(),
        screenshot_path=None
    )

    # Test de décision
    action = decision_maker.decide_dofus_action(test_state)

    print(f"Action décidée: {action.action_type}")
    print(f"Confiance: {action.confidence:.2f}")
    print(f"Raisonnement: {action.reasoning}")

    # Rapport de performance
    report = decision_maker.get_performance_report()
    print(f"\nRapport: {json.dumps(report, indent=2)}")