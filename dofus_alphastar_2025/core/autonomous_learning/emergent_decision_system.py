"""
Emergent Decision System - Système de décision émergente
Le bot prend des décisions autonomes basées sur sa conscience, ses besoins, et ses souvenirs

Approche "embodied cognition" (cognition incarnée):
- Les décisions émergent de l'interaction entre:
  * État interne (besoins, émotions, fatigue)
  * Perception du monde (environnement, opportunités, menaces)
  * Mémoire (expériences passées, apprentissages)
  * Personnalité (traits de caractère, préférences)

Pas de décision "script" - tout est émergent et contextuel, comme un humain
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

from .self_awareness import (
    SelfAwarenessEngine,
    EmotionalState,
    PhysicalNeed,
    CognitiveNeed,
    SocialNeed
)
from .continuous_learning import (
    ContinuousLearningEngine,
    ExperienceType,
    LearningMode
)
from .autobiographical_memory import (
    AutobiographicalMemory,
    MemoryCategory,
    MemoryImportance
)


logger = logging.getLogger(__name__)


class DecisionOrigin(Enum):
    """Origine de la décision"""
    SURVIVAL = "survival"           # Réponse de survie (réflexe)
    EMOTIONAL = "emotional"         # Réponse émotionnelle
    HABITUAL = "habitual"           # Habitude apprise
    DELIBERATE = "deliberate"       # Décision réfléchie
    CURIOUS = "curious"             # Exploration curieuse
    SOCIAL = "social"               # Interaction sociale
    GOAL_DIRECTED = "goal_directed" # Objectif à long terme


@dataclass
class Decision:
    """Une décision prise par le bot"""
    timestamp: float
    action_type: str
    action_details: Dict[str, Any]

    # Contexte de la décision
    origin: DecisionOrigin
    reasoning: str                  # Pourquoi cette décision?
    confidence: float               # Confiance dans la décision

    # Motivation
    driven_by_need: Optional[str] = None
    driven_by_emotion: Optional[str] = None
    driven_by_memory: Optional[str] = None

    # Métadonnées
    alternatives_considered: int = 0
    deliberation_time: float = 0.0


class MotivationSystem:
    """
    Système de motivation - Qu'est-ce qui pousse le bot à agir?
    Basé sur la hiérarchie des besoins de Maslow adaptée au gaming
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.MotivationSystem")

        # Poids des différentes motivations (personnalité)
        self.motivation_weights = {
            "survival": 1.0,        # Survie (toujours prioritaire)
            "growth": 0.7,          # Croissance (progression)
            "exploration": 0.6,     # Exploration (découverte)
            "mastery": 0.8,         # Maîtrise (compétence)
            "social": 0.4,          # Social (interactions)
            "achievement": 0.7      # Accomplissement (objectifs)
        }

    def compute_motivation(
        self,
        self_state,
        needs: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calcule les motivations actuelles basées sur les besoins
        Retourne un vecteur de motivation
        """
        motivations = {}

        # Motivation de survie (besoin de santé)
        if "PhysicalNeed.HEALTH" in needs:
            health_need = needs["PhysicalNeed.HEALTH"]
            motivations["survival"] = health_need * self.motivation_weights["survival"]

        # Motivation de croissance (progression)
        if "CognitiveNeed.LEARNING" in needs:
            learning_need = needs["CognitiveNeed.LEARNING"]
            motivations["growth"] = learning_need * self.motivation_weights["growth"]

        # Motivation d'exploration (curiosité)
        if "CognitiveNeed.EXPLORATION" in needs:
            exploration_need = needs["CognitiveNeed.EXPLORATION"]
            motivations["exploration"] = exploration_need * self.motivation_weights["exploration"]

        # Motivation de maîtrise
        if "CognitiveNeed.MASTERY" in needs:
            mastery_need = needs["CognitiveNeed.MASTERY"]
            motivations["mastery"] = mastery_need * self.motivation_weights["mastery"]

        # Motivation sociale
        if "SocialNeed.BELONGING" in needs:
            social_need = needs["SocialNeed.BELONGING"]
            motivations["social"] = social_need * self.motivation_weights["social"]

        # Motivation d'accomplissement
        if "CognitiveNeed.ACHIEVEMENT" in needs:
            achievement_need = needs["CognitiveNeed.ACHIEVEMENT"]
            motivations["achievement"] = achievement_need * self.motivation_weights["achievement"]

        return motivations

    def get_dominant_motivation(self, motivations: Dict[str, float]) -> Tuple[str, float]:
        """Retourne la motivation dominante"""
        if not motivations:
            return "exploration", 0.5  # Par défaut

        dominant = max(motivations.items(), key=lambda x: x[1])
        return dominant


class ActionGenerator:
    """
    Générateur d'actions possibles basé sur le contexte
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ActionGenerator")

    def generate_possible_actions(
        self,
        game_state: Any,
        world_perception: Any,
        motivations: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Génère une liste d'actions possibles dans le contexte actuel
        """
        possible_actions = []

        # Actions de survie (si HP bas)
        if game_state.character.hp_percent < 50:
            possible_actions.extend(self._generate_survival_actions(game_state))

        # Actions de combat (si en combat)
        if hasattr(game_state, 'combat') and game_state.combat.in_combat:
            possible_actions.extend(self._generate_combat_actions(game_state))

        # Actions d'exploration (si curiosité élevée)
        if motivations.get("exploration", 0.0) > 0.5:
            possible_actions.extend(self._generate_exploration_actions(game_state, world_perception))

        # Actions sociales (si joueurs visibles)
        if world_perception.social_context != "alone":
            possible_actions.extend(self._generate_social_actions(game_state, world_perception))

        # Actions de progression (farming, quêtes)
        if motivations.get("growth", 0.0) > 0.4:
            possible_actions.extend(self._generate_progression_actions(game_state))

        # Toujours au moins une action (attendre/observer)
        if not possible_actions:
            possible_actions.append({
                "action_type": "wait",
                "details": {},
                "expected_value": 0.1,
                "origin": DecisionOrigin.DELIBERATE
            })

        return possible_actions

    def _generate_survival_actions(self, game_state: Any) -> List[Dict[str, Any]]:
        """Actions de survie"""
        actions = []

        # Fuir si en danger
        if hasattr(game_state, 'combat') and game_state.combat.in_combat:
            actions.append({
                "action_type": "flee",
                "details": {},
                "expected_value": 0.9,  # Haute valeur de survie
                "origin": DecisionOrigin.SURVIVAL
            })

        # Se soigner
        actions.append({
            "action_type": "heal",
            "details": {},
            "expected_value": 0.8,
            "origin": DecisionOrigin.SURVIVAL
        })

        return actions

    def _generate_combat_actions(self, game_state: Any) -> List[Dict[str, Any]]:
        """Actions de combat"""
        actions = []

        if not game_state.combat.my_turn:
            return actions

        enemies = game_state.combat.get_alive_enemies()
        if not enemies:
            return actions

        # Attaquer l'ennemi le plus faible
        weakest = game_state.combat.get_weakest_enemy()
        if weakest and game_state.character.pa >= 3:
            actions.append({
                "action_type": "spell",
                "details": {
                    "spell_key": "1",
                    "target_x": weakest.position[0],
                    "target_y": weakest.position[1]
                },
                "expected_value": 0.7,
                "origin": DecisionOrigin.DELIBERATE
            })

        # Défendre
        actions.append({
            "action_type": "defend",
            "details": {},
            "expected_value": 0.5,
            "origin": DecisionOrigin.DELIBERATE
        })

        return actions

    def _generate_exploration_actions(self, game_state: Any, world_perception: Any) -> List[Dict[str, Any]]:
        """Actions d'exploration"""
        actions = []

        # Explorer une zone inconnue
        if world_perception.current_map:
            actions.append({
                "action_type": "move",
                "details": {
                    "x": np.random.randint(200, 800),
                    "y": np.random.randint(200, 500),
                    "reason": "exploration"
                },
                "expected_value": 0.6,
                "origin": DecisionOrigin.CURIOUS
            })

        return actions

    def _generate_social_actions(self, game_state: Any, world_perception: Any) -> List[Dict[str, Any]]:
        """Actions sociales"""
        actions = []

        # Observer les autres joueurs
        actions.append({
            "action_type": "observe_players",
            "details": {},
            "expected_value": 0.4,
            "origin": DecisionOrigin.SOCIAL
        })

        return actions

    def _generate_progression_actions(self, game_state: Any) -> List[Dict[str, Any]]:
        """Actions de progression"""
        actions = []

        # Farming
        actions.append({
            "action_type": "farm",
            "details": {},
            "expected_value": 0.6,
            "origin": DecisionOrigin.GOAL_DIRECTED
        })

        return actions


class EmergentDecisionSystem:
    """
    Système de décision émergente complet

    Les décisions émergent de l'interaction entre:
    1. Conscience de soi (état interne)
    2. Besoins et motivations
    3. Perception du monde
    4. Mémoire des expériences
    5. Apprentissage continu
    6. Personnalité

    Aucune décision n'est "scriptée" - tout est contextuel et émergent
    """

    def __init__(
        self,
        self_awareness: SelfAwarenessEngine,
        learning_engine: ContinuousLearningEngine,
        memory: AutobiographicalMemory
    ):
        self.logger = logging.getLogger(f"{__name__}.EmergentDecisionSystem")

        # Composants cognitifs
        self.self_awareness = self_awareness
        self.learning_engine = learning_engine
        self.memory = memory

        # Sous-systèmes
        self.motivation_system = MotivationSystem()
        self.action_generator = ActionGenerator()

        # Historique des décisions
        self.decision_history = deque(maxlen=1000)
        self.recent_events = deque(maxlen=50)

        # État mental actuel
        self.current_focus = None
        self.current_goal = None

        # Statistiques
        self.decision_stats = {
            "total_decisions": 0,
            "decisions_by_origin": defaultdict(int),
            "average_confidence": 0.0,
            "successful_decisions": 0,
            "failed_decisions": 0
        }

        self.logger.info("Emergent Decision System initialized - Prêt à décider de manière autonome")

    def decide(
        self,
        game_state: Any,
        vision_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Decision]:
        """
        Prend une décision autonome basée sur l'état complet du système

        Le processus de décision:
        1. Mise à jour de la conscience de soi
        2. Évaluation des besoins et motivations
        3. Rappel de souvenirs pertinents
        4. Génération d'actions possibles
        5. Évaluation des options
        6. Sélection de l'action
        7. Création du souvenir de la décision
        """
        decision_start_time = time.time()

        # 1. Mettre à jour la conscience de soi
        self_state = self.self_awareness.update_consciousness(
            game_state,
            list(self.recent_events)
        )

        world_perception = self.self_awareness.current_world_perception

        # 2. Calculer les motivations actuelles
        motivations = self.motivation_system.compute_motivation(
            self_state,
            self_state.needs
        )

        dominant_motivation, motivation_strength = self.motivation_system.get_dominant_motivation(motivations)

        # 3. Rappeler des souvenirs similaires
        context = {
            "current_map": world_perception.current_map,
            "category": self._context_to_category(game_state),
            "emotional_state": self_state.current_emotion.value
        }

        similar_memories = self.memory.recall_similar_memories(context, limit=3)

        # 4. Générer les actions possibles
        possible_actions = self.action_generator.generate_possible_actions(
            game_state,
            world_perception,
            motivations
        )

        if not possible_actions:
            return None

        # 5. Évaluer chaque action basée sur:
        #    - Valeur attendue
        #    - Alignement avec besoins/motivations
        #    - Expériences passées similaires
        #    - État émotionnel
        scored_actions = []

        for action in possible_actions:
            score = self._evaluate_action(
                action,
                motivations,
                self_state,
                similar_memories
            )
            scored_actions.append((score, action))

        # 6. Sélectionner la meilleure action (avec un peu de stochasticité pour la variété)
        scored_actions.sort(key=lambda x: x[0], reverse=True)

        # Choix stochastique parmi les top 3 (comportement humain)
        top_actions = scored_actions[:min(3, len(scored_actions))]
        weights = [score for score, _ in top_actions]

        if sum(weights) == 0:
            selected_action = top_actions[0][1]
        else:
            weights_normalized = np.array(weights) / sum(weights)
            selected_idx = np.random.choice(len(top_actions), p=weights_normalized)
            selected_action = top_actions[selected_idx][1]

        # 7. Créer la décision
        decision = Decision(
            timestamp=time.time(),
            action_type=selected_action["action_type"],
            action_details=selected_action["details"],
            origin=selected_action["origin"],
            reasoning=self._generate_reasoning(
                selected_action,
                dominant_motivation,
                self_state
            ),
            confidence=top_actions[0][0] if top_actions else 0.5,
            driven_by_need=dominant_motivation,
            driven_by_emotion=self_state.current_emotion.value,
            driven_by_memory=similar_memories[0].memory_id if similar_memories else None,
            alternatives_considered=len(possible_actions),
            deliberation_time=time.time() - decision_start_time
        )

        # Enregistrer la décision
        self.decision_history.append(decision)
        self.decision_stats["total_decisions"] += 1
        self.decision_stats["decisions_by_origin"][decision.origin.value] += 1

        # Log de la décision avec le raisonnement
        self.logger.debug(
            f"💡 Décision: {decision.action_type} | "
            f"Raison: {decision.reasoning} | "
            f"Confiance: {decision.confidence:.2f}"
        )

        return decision

    def _evaluate_action(
        self,
        action: Dict[str, Any],
        motivations: Dict[str, float],
        self_state,
        similar_memories: List
    ) -> float:
        """
        Évalue une action basée sur tous les facteurs contextuels
        """
        score = action.get("expected_value", 0.5)

        # Bonus si l'action est alignée avec la motivation dominante
        action_origin = action.get("origin")
        if action_origin:
            origin_value = action_origin.value

            if "survival" in origin_value and motivations.get("survival", 0) > 0.7:
                score += 0.3
            elif "exploration" in origin_value and motivations.get("exploration", 0) > 0.6:
                score += 0.2
            elif "growth" in origin_value and motivations.get("growth", 0) > 0.5:
                score += 0.2

        # Influence émotionnelle
        emotion = self_state.current_emotion

        if emotion == EmotionalState.ANXIOUS and action_origin == DecisionOrigin.SURVIVAL:
            score += 0.3  # Peur → fuir/se protéger

        elif emotion == EmotionalState.CURIOUS and action_origin == DecisionOrigin.CURIOUS:
            score += 0.2  # Curiosité → explorer

        elif emotion == EmotionalState.CONFIDENT and action_origin == DecisionOrigin.DELIBERATE:
            score += 0.1  # Confiance → actions réfléchies

        # Apprentissage des expériences passées
        if similar_memories:
            # Si des actions similaires ont fonctionné avant, bonus
            for memory in similar_memories:
                if memory.emotional_valence > 0.5:
                    score += 0.1

        # Fatigue cognitive réduit la qualité des décisions complexes
        if self_state.cognitive_load > 0.7:
            if action_origin == DecisionOrigin.DELIBERATE:
                score *= 0.8  # Décisions complexes moins bonnes sous fatigue

        return max(0.0, min(1.0, score))

    def _generate_reasoning(
        self,
        action: Dict[str, Any],
        dominant_motivation: str,
        self_state
    ) -> str:
        """Génère une explication de la décision (conscience verbale)"""
        action_type = action["action_type"]
        emotion = self_state.current_emotion.value

        # Raisons basées sur la motivation
        motivation_reasons = {
            "survival": f"Je dois assurer ma survie ({action_type})",
            "exploration": f"Je suis curieux d'explorer ({action_type})",
            "growth": f"Je veux progresser ({action_type})",
            "mastery": f"Je veux maîtriser cette compétence ({action_type})",
            "social": f"Je veux interagir avec les autres ({action_type})",
            "achievement": f"Je veux accomplir quelque chose ({action_type})"
        }

        base_reason = motivation_reasons.get(dominant_motivation, f"Je choisis de {action_type}")

        # Ajouter le contexte émotionnel
        if emotion in ["anxious", "focused"]:
            return f"{base_reason} - je me sens {emotion}"
        else:
            return base_reason

    def _context_to_category(self, game_state: Any) -> str:
        """Convertit le contexte actuel en catégorie de mémoire"""
        if hasattr(game_state, 'combat') and game_state.combat.in_combat:
            return MemoryCategory.COMBAT.value
        else:
            return MemoryCategory.EXPLORATION.value

    def record_decision_outcome(
        self,
        decision: Decision,
        outcome: str,  # "success", "failure", "neutral"
        reward: float = 0.0
    ):
        """
        Enregistre le résultat d'une décision (apprentissage)
        """
        # Créer une expérience pour l'apprentissage
        experience_type = ExperienceType.SUCCESS if outcome == "success" else ExperienceType.FAILURE

        self.learning_engine.record_experience(
            state_before={},  # Pourrait être enrichi
            action=decision.action_type,
            state_after={},
            reward=reward,
            experience_type=experience_type
        )

        # Créer un souvenir si important
        if abs(reward) > 0.5 or outcome == "failure":
            importance = MemoryImportance.SIGNIFICANT if abs(reward) > 0.7 else MemoryImportance.MODERATE

            self.memory.create_memory(
                what_happened=f"{decision.action_type}: {outcome}",
                where=self.self_awareness.current_world_perception.current_map,
                category=MemoryCategory.LEARNING,
                importance=importance,
                emotional_valence=reward,
                emotional_intensity=abs(reward),
                associated_emotion=self.self_awareness.current_self_state.current_emotion.value,
                lesson_learned=f"Action {decision.action_type} a mené à {outcome}"
            )

        # Statistiques
        if outcome == "success":
            self.decision_stats["successful_decisions"] += 1
        elif outcome == "failure":
            self.decision_stats["failed_decisions"] += 1

    def add_event(self, event: Dict[str, Any]):
        """Ajoute un événement récent au contexte"""
        self.recent_events.append(event)

    def get_state(self) -> Dict[str, Any]:
        """Retourne l'état du système de décision"""
        return {
            "total_decisions": self.decision_stats["total_decisions"],
            "decision_origins": dict(self.decision_stats["decisions_by_origin"]),
            "success_rate": (
                self.decision_stats["successful_decisions"] /
                max(1, self.decision_stats["total_decisions"])
            ),
            "current_focus": self.current_focus,
            "current_goal": self.current_goal
        }


def create_emergent_decision_system(
    self_awareness: SelfAwarenessEngine,
    learning_engine: ContinuousLearningEngine,
    memory: AutobiographicalMemory
) -> EmergentDecisionSystem:
    """Factory function"""
    return EmergentDecisionSystem(self_awareness, learning_engine, memory)
