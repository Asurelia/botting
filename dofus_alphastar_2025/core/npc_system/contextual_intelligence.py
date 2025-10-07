#!/usr/bin/env python3
"""
ContextualIntelligence - Intelligence contextuelle pour décisions adaptatives DOFUS
Utilise HRM pour prendre des décisions intelligentes selon le contexte de jeu
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import json

import torch
import numpy as np

from config import config
from core.hrm_reasoning import create_hrm_model, HRMOutput

logger = logging.getLogger(__name__)

class GamePhase(Enum):
    """Phases de jeu"""
    EARLY_GAME = "early_game"      # Niveau 1-20
    MID_GAME = "mid_game"          # Niveau 20-60
    LATE_GAME = "late_game"        # Niveau 60-100
    END_GAME = "end_game"          # Niveau 100+
    TUTORIAL = "tutorial"          # Phase tutoriel

class PlayerGoal(Enum):
    """Objectifs du joueur"""
    LEVEL_UP = "level_up"
    EARN_KAMAS = "earn_kamas"
    COMPLETE_QUESTS = "complete_quests"
    EXPLORE = "explore"
    SOCIALIZE = "socialize"
    COLLECT = "collect"
    CRAFT = "craft"
    PVP = "pvp"
    ACHIEVEMENT = "achievement"

class ContextType(Enum):
    """Types de contexte"""
    COMBAT = "combat"
    SOCIAL = "social"
    ECONOMIC = "economic"
    EXPLORATION = "exploration"
    QUEST = "quest"
    EMERGENCY = "emergency"
    IDLE = "idle"

class DecisionConfidence(Enum):
    """Niveaux de confiance"""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.9

@dataclass
class GameContext:
    """Contexte de jeu complet"""
    # État joueur
    player_level: int = 1
    player_class: str = ""
    player_location: Tuple[int, int] = (0, 0)
    current_map: str = ""

    # Ressources
    current_hp: int = 100
    max_hp: int = 100
    current_mp: int = 100
    max_mp: int = 100
    kamas: int = 0
    experience: int = 0

    # État de jeu
    game_phase: GamePhase = GamePhase.EARLY_GAME
    current_goals: List[PlayerGoal] = field(default_factory=list)
    context_type: ContextType = ContextType.IDLE

    # Activité actuelle
    current_quest: Optional[str] = None
    combat_active: bool = False
    dialogue_active: bool = False

    # Environnement
    nearby_npcs: List[str] = field(default_factory=list)
    nearby_monsters: List[str] = field(default_factory=list)
    nearby_players: List[str] = field(default_factory=list)

    # Inventaire et équipement
    inventory_slots_used: int = 0
    inventory_slots_total: int = 100
    equipment_score: float = 1.0

    # Temporel
    real_time_hour: int = 12
    game_time: float = 0.0
    session_duration: float = 0.0

    # Métadonnées
    last_action: Optional[str] = None
    last_action_success: bool = True
    consecutive_failures: int = 0

    @property
    def hp_percentage(self) -> float:
        return (self.current_hp / max(self.max_hp, 1)) * 100

    @property
    def mp_percentage(self) -> float:
        return (self.current_mp / max(self.max_mp, 1)) * 100

    @property
    def inventory_full_percentage(self) -> float:
        return (self.inventory_slots_used / max(self.inventory_slots_total, 1)) * 100

    @property
    def is_in_danger(self) -> bool:
        return self.hp_percentage < 30 or self.combat_active

    @property
    def needs_rest(self) -> bool:
        return self.hp_percentage < 50 or self.mp_percentage < 30

@dataclass
class ContextualDecision:
    """Décision contextuelle"""
    action_type: str
    action_data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    reasoning: str = ""
    priority: int = 1  # 1=basse, 5=haute
    estimated_duration: float = 0.0
    expected_outcome: Dict[str, Any] = field(default_factory=dict)

    # Conditions d'exécution
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)

    # Risques et alternatives
    risk_level: float = 0.0
    alternative_actions: List[str] = field(default_factory=list)

class ContextAnalyzer:
    """Analyseur de contexte de jeu"""

    def __init__(self):
        # Seuils et paramètres
        self.danger_hp_threshold = 30.0
        self.rest_hp_threshold = 50.0
        self.inventory_full_threshold = 80.0

        # Patterns contextuels
        self.context_patterns = self._load_context_patterns()

    def _load_context_patterns(self) -> Dict[str, Any]:
        """Charge patterns contextuels"""
        return {
            "combat_indicators": [
                "nearby_monsters",
                "combat_active",
                "hp_percentage < 50"
            ],
            "social_indicators": [
                "nearby_players",
                "dialogue_active",
                "guild_context"
            ],
            "economic_indicators": [
                "inventory_full",
                "valuable_items",
                "market_opportunities"
            ],
            "exploration_indicators": [
                "new_map",
                "undiscovered_areas",
                "exploration_quests"
            ]
        }

    def analyze_context(self, game_context: GameContext) -> ContextType:
        """Analyse le contexte de jeu actuel"""

        # Contexte d'urgence
        if game_context.is_in_danger:
            return ContextType.EMERGENCY

        # Contexte de combat
        if game_context.combat_active or game_context.nearby_monsters:
            return ContextType.COMBAT

        # Contexte social
        if game_context.dialogue_active or len(game_context.nearby_players) > 0:
            return ContextType.SOCIAL

        # Contexte économique
        if game_context.inventory_full_percentage > self.inventory_full_threshold:
            return ContextType.ECONOMIC

        # Contexte de quête
        if game_context.current_quest:
            return ContextType.QUEST

        # Contexte d'exploration
        if game_context.nearby_npcs and not game_context.current_quest:
            return ContextType.EXPLORATION

        # Par défaut
        return ContextType.IDLE

    def detect_game_phase(self, game_context: GameContext) -> GamePhase:
        """Détecte la phase de jeu actuelle"""

        level = game_context.player_level

        if level <= 5 and game_context.session_duration < 3600:  # Première heure
            return GamePhase.TUTORIAL
        elif level < 20:
            return GamePhase.EARLY_GAME
        elif level < 60:
            return GamePhase.MID_GAME
        elif level < 100:
            return GamePhase.LATE_GAME
        else:
            return GamePhase.END_GAME

    def infer_player_goals(self, game_context: GameContext) -> List[PlayerGoal]:
        """Infère les objectifs probables du joueur"""

        goals = []

        # Objectifs par phase de jeu
        if game_context.game_phase in [GamePhase.TUTORIAL, GamePhase.EARLY_GAME]:
            goals.extend([PlayerGoal.LEVEL_UP, PlayerGoal.COMPLETE_QUESTS])

        elif game_context.game_phase == GamePhase.MID_GAME:
            goals.extend([PlayerGoal.LEVEL_UP, PlayerGoal.EARN_KAMAS, PlayerGoal.COMPLETE_QUESTS])

        else:
            goals.extend([PlayerGoal.EARN_KAMAS, PlayerGoal.ACHIEVEMENT, PlayerGoal.CRAFT])

        # Objectifs situationnels
        if game_context.inventory_full_percentage > 80:
            goals.insert(0, PlayerGoal.EARN_KAMAS)  # Priorité

        if game_context.current_quest:
            if PlayerGoal.COMPLETE_QUESTS not in goals:
                goals.insert(0, PlayerGoal.COMPLETE_QUESTS)

        if len(game_context.nearby_npcs) > 0 and not game_context.current_quest:
            goals.append(PlayerGoal.EXPLORE)

        return goals[:3]  # Limiter à 3 objectifs principaux

class DecisionEngine:
    """Moteur de décision contextuel"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hrm_model = create_hrm_model().to(self.device)

        # Stratégies de décision par contexte
        self.context_strategies = {
            ContextType.EMERGENCY: self._emergency_strategy,
            ContextType.COMBAT: self._combat_strategy,
            ContextType.SOCIAL: self._social_strategy,
            ContextType.ECONOMIC: self._economic_strategy,
            ContextType.QUEST: self._quest_strategy,
            ContextType.EXPLORATION: self._exploration_strategy,
            ContextType.IDLE: self._idle_strategy
        }

        # Cache de décisions
        self.decision_cache: Dict[str, ContextualDecision] = {}
        self.cache_duration = 30.0  # 30 secondes

    def make_contextual_decision(self, game_context: GameContext) -> ContextualDecision:
        """Prend une décision basée sur le contexte"""

        # Vérifier cache
        cache_key = self._create_cache_key(game_context)
        if cache_key in self.decision_cache:
            cached_decision = self.decision_cache[cache_key]
            # Vérifier si encore valide
            if time.time() - cached_decision.estimated_duration < self.cache_duration:
                return cached_decision

        try:
            # Utiliser HRM pour raisonnement complexe
            hrm_decision = self._make_hrm_decision(game_context)

            if hrm_decision:
                # Mettre en cache
                self.decision_cache[cache_key] = hrm_decision
                return hrm_decision

            # Fallback vers stratégies classiques
            strategy_func = self.context_strategies.get(
                game_context.context_type,
                self._idle_strategy
            )

            decision = strategy_func(game_context)
            self.decision_cache[cache_key] = decision
            return decision

        except Exception as e:
            logger.error(f"Erreur prise de décision: {e}")
            return self._safe_fallback_decision(game_context)

    def _make_hrm_decision(self, game_context: GameContext) -> Optional[ContextualDecision]:
        """Utilise HRM pour prise de décision avancée"""

        try:
            with torch.no_grad():
                # Préparer contexte pour HRM
                hrm_context = self._prepare_hrm_context(game_context)

                # Simuler input pour HRM
                input_ids = torch.randint(0, 32000, (1, 96), device=self.device)

                # Forward pass avec System 2 pour raisonnement
                hrm_output = self.hrm_model(
                    input_ids=input_ids,
                    return_reasoning_details=True,
                    max_reasoning_steps=min(5, config.hrm.max_reasoning_steps)
                )

                # Interpréter sortie HRM
                decision = self._interpret_hrm_output(hrm_output, game_context, hrm_context)
                return decision

        except Exception as e:
            logger.warning(f"Erreur HRM décision: {e}")
            return None

    def _prepare_hrm_context(self, game_context: GameContext) -> Dict[str, Any]:
        """Prépare contexte pour HRM"""

        return {
            "player_level": game_context.player_level,
            "hp_percentage": game_context.hp_percentage,
            "mp_percentage": game_context.mp_percentage,
            "context_type": game_context.context_type.value,
            "game_phase": game_context.game_phase.value,
            "in_danger": game_context.is_in_danger,
            "needs_rest": game_context.needs_rest,
            "inventory_full": game_context.inventory_full_percentage > 80,
            "has_quest": game_context.current_quest is not None,
            "nearby_entities": len(game_context.nearby_npcs) + len(game_context.nearby_monsters),
            "consecutive_failures": game_context.consecutive_failures
        }

    def _interpret_hrm_output(self,
                             hrm_output: Union[HRMOutput, Dict],
                             game_context: GameContext,
                             hrm_context: Dict[str, Any]) -> ContextualDecision:
        """Interprète sortie HRM en décision"""

        # Logique d'interprétation basée sur le contexte
        if game_context.is_in_danger:
            return ContextualDecision(
                action_type="seek_safety",
                reasoning="HRM détecte danger immédiat",
                confidence=0.9,
                priority=5
            )

        elif game_context.needs_rest:
            return ContextualDecision(
                action_type="rest_and_recover",
                reasoning="HRM recommande récupération",
                confidence=0.8,
                priority=4
            )

        elif game_context.current_quest:
            return ContextualDecision(
                action_type="continue_quest",
                action_data={"quest_id": game_context.current_quest},
                reasoning="HRM priorise progression de quête",
                confidence=0.7,
                priority=3
            )

        elif game_context.inventory_full_percentage > 80:
            return ContextualDecision(
                action_type="manage_inventory",
                reasoning="HRM détecte inventaire plein",
                confidence=0.8,
                priority=3
            )

        else:
            # Décision d'exploration par défaut
            return ContextualDecision(
                action_type="explore_and_interact",
                reasoning="HRM suggère exploration",
                confidence=0.6,
                priority=2
            )

    def _emergency_strategy(self, game_context: GameContext) -> ContextualDecision:
        """Stratégie d'urgence"""

        if game_context.hp_percentage < 10:
            return ContextualDecision(
                action_type="emergency_heal",
                reasoning="HP critique - heal immédiat",
                confidence=0.95,
                priority=5,
                risk_level=0.9
            )

        elif game_context.combat_active and game_context.hp_percentage < 30:
            return ContextualDecision(
                action_type="retreat_from_combat",
                reasoning="Combat dangereux - retraite",
                confidence=0.9,
                priority=5,
                risk_level=0.7
            )

        else:
            return ContextualDecision(
                action_type="seek_safe_location",
                reasoning="Chercher sécurité",
                confidence=0.8,
                priority=4
            )

    def _combat_strategy(self, game_context: GameContext) -> ContextualDecision:
        """Stratégie de combat"""

        if game_context.hp_percentage > 70 and game_context.mp_percentage > 50:
            return ContextualDecision(
                action_type="engage_combat",
                reasoning="Condition favorable au combat",
                confidence=0.8,
                priority=3,
                expected_outcome={"experience": "medium", "risk": "medium"}
            )

        elif game_context.hp_percentage < 40:
            return ContextualDecision(
                action_type="defensive_combat",
                reasoning="HP bas - combat défensif",
                confidence=0.7,
                priority=3,
                risk_level=0.6
            )

        else:
            return ContextualDecision(
                action_type="avoid_combat",
                reasoning="Conditions non optimales",
                confidence=0.6,
                priority=2
            )

    def _social_strategy(self, game_context: GameContext) -> ContextualDecision:
        """Stratégie sociale"""

        if game_context.dialogue_active:
            return ContextualDecision(
                action_type="continue_dialogue",
                reasoning="Dialogue en cours",
                confidence=0.8,
                priority=3
            )

        elif game_context.nearby_npcs:
            return ContextualDecision(
                action_type="interact_with_npc",
                reasoning="PNJ disponible pour interaction",
                confidence=0.7,
                priority=2,
                expected_outcome={"quests": "possible", "information": "likely"}
            )

        else:
            return ContextualDecision(
                action_type="observe_players",
                reasoning="Observer autres joueurs",
                confidence=0.5,
                priority=1
            )

    def _economic_strategy(self, game_context: GameContext) -> ContextualDecision:
        """Stratégie économique"""

        if game_context.inventory_full_percentage > 90:
            return ContextualDecision(
                action_type="sell_items_urgently",
                reasoning="Inventaire quasi plein",
                confidence=0.9,
                priority=4
            )

        elif game_context.kamas < 1000 and game_context.player_level > 10:
            return ContextualDecision(
                action_type="farming_for_kamas",
                reasoning="Manque de kamas",
                confidence=0.7,
                priority=3,
                estimated_duration=1800.0  # 30 minutes
            )

        else:
            return ContextualDecision(
                action_type="optimize_resources",
                reasoning="Gestion ressources optimale",
                confidence=0.6,
                priority=2
            )

    def _quest_strategy(self, game_context: GameContext) -> ContextualDecision:
        """Stratégie de quête"""

        return ContextualDecision(
            action_type="progress_quest",
            action_data={"quest_id": game_context.current_quest},
            reasoning="Continuer quête active",
            confidence=0.8,
            priority=3,
            expected_outcome={"experience": "high", "story_progress": "yes"}
        )

    def _exploration_strategy(self, game_context: GameContext) -> ContextualDecision:
        """Stratégie d'exploration"""

        if game_context.nearby_npcs:
            return ContextualDecision(
                action_type="talk_to_nearby_npc",
                reasoning="PNJ disponible - potentielle quête",
                confidence=0.7,
                priority=3
            )

        else:
            return ContextualDecision(
                action_type="explore_new_area",
                reasoning="Explorer zone inconnue",
                confidence=0.6,
                priority=2,
                estimated_duration=600.0  # 10 minutes
            )

    def _idle_strategy(self, game_context: GameContext) -> ContextualDecision:
        """Stratégie par défaut"""

        # Choisir action selon phase de jeu
        if game_context.game_phase == GamePhase.TUTORIAL:
            return ContextualDecision(
                action_type="follow_tutorial",
                reasoning="Phase tutoriel - suivre instructions",
                confidence=0.8,
                priority=3
            )

        elif game_context.player_level < 10:
            return ContextualDecision(
                action_type="find_beginner_quests",
                reasoning="Bas niveau - chercher quêtes débutant",
                confidence=0.7,
                priority=3
            )

        else:
            return ContextualDecision(
                action_type="general_exploration",
                reasoning="Exploration générale",
                confidence=0.5,
                priority=2
            )

    def _safe_fallback_decision(self, game_context: GameContext) -> ContextualDecision:
        """Décision de sécurité en cas d'erreur"""

        return ContextualDecision(
            action_type="wait_and_observe",
            reasoning="Décision de sécurité - attendre",
            confidence=0.3,
            priority=1,
            estimated_duration=10.0
        )

    def _create_cache_key(self, game_context: GameContext) -> str:
        """Crée clé de cache pour décision"""

        key_components = [
            str(game_context.context_type.value),
            str(game_context.player_level),
            str(int(game_context.hp_percentage / 10) * 10),  # Arrondi à 10%
            str(game_context.is_in_danger),
            str(game_context.current_quest is not None),
            str(len(game_context.nearby_npcs)),
            str(len(game_context.nearby_monsters))
        ]

        return "_".join(key_components)

class ContextualIntelligence:
    """Intelligence contextuelle principale"""

    def __init__(self):
        self.context_analyzer = ContextAnalyzer()
        self.decision_engine = DecisionEngine()

        # Historique et apprentissage
        self.decision_history: List[Dict[str, Any]] = []
        self.context_patterns: Dict[str, int] = {}

        # Métriques
        self.total_decisions = 0
        self.successful_decisions = 0

        logger.info("ContextualIntelligence initialisé avec succès")

    def process_game_state(self, game_context: GameContext) -> ContextualDecision:
        """Traite état de jeu et retourne décision"""

        self.total_decisions += 1

        try:
            # Analyser contexte
            game_context.context_type = self.context_analyzer.analyze_context(game_context)
            game_context.game_phase = self.context_analyzer.detect_game_phase(game_context)
            game_context.current_goals = self.context_analyzer.infer_player_goals(game_context)

            # Prendre décision
            decision = self.decision_engine.make_contextual_decision(game_context)

            # Enregistrer dans historique
            self._record_decision(game_context, decision)

            return decision

        except Exception as e:
            logger.error(f"Erreur traitement état: {e}")
            return ContextualDecision(
                action_type="error_recovery",
                reasoning=f"Erreur système: {e}",
                confidence=0.1,
                priority=1
            )

    def _record_decision(self, game_context: GameContext, decision: ContextualDecision):
        """Enregistre décision dans historique"""

        record = {
            "timestamp": time.time(),
            "context_type": game_context.context_type.value,
            "game_phase": game_context.game_phase.value,
            "player_level": game_context.player_level,
            "decision_action": decision.action_type,
            "confidence": decision.confidence,
            "priority": decision.priority
        }

        self.decision_history.append(record)

        # Limiter historique
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]

        # Mettre à jour patterns
        context_key = f"{game_context.context_type.value}_{decision.action_type}"
        self.context_patterns[context_key] = self.context_patterns.get(context_key, 0) + 1

    def update_decision_outcome(self, decision_id: str, success: bool, feedback: Dict[str, Any] = None):
        """Met à jour résultat d'une décision"""

        if success:
            self.successful_decisions += 1

        # Ici on pourrait implémenter apprentissage par renforcement
        # pour améliorer les futures décisions

    def get_intelligence_stats(self) -> Dict[str, Any]:
        """Statistiques d'intelligence contextuelle"""

        success_rate = (self.successful_decisions / max(self.total_decisions, 1)) * 100

        # Patterns les plus fréquents
        top_patterns = sorted(
            self.context_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return {
            "total_decisions": self.total_decisions,
            "successful_decisions": self.successful_decisions,
            "success_rate": success_rate,
            "decision_history_size": len(self.decision_history),
            "top_context_patterns": top_patterns,
            "unique_patterns": len(self.context_patterns)
        }

def create_contextual_intelligence() -> ContextualIntelligence:
    """Factory function pour créer ContextualIntelligence"""
    return ContextualIntelligence()