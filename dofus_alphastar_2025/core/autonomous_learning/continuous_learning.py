"""
Continuous Learning Engine - Apprentissage autonome continu
Le bot apprend en permanence de ses expériences comme un humain

Inspiré de:
- Experience Replay (DeepMind)
- Curriculum Learning (Bengio)
- Meta-Learning (Learning to Learn)
- Intrinsic Motivation (Curiosity-driven learning)

Fonctionnalités:
- Apprentissage par expérience (succès et échecs)
- Apprentissage par curiosité (exploration intrinsèque)
- Apprentissage par imitation (observation d'autres joueurs)
- Meta-apprentissage (apprendre à mieux apprendre)
- Adaptation continue des stratégies
"""

import time
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import json
import pickle


logger = logging.getLogger(__name__)


class ExperienceType(Enum):
    """Types d'expériences à apprendre"""
    SUCCESS = "success"           # Succès (récompense positive)
    FAILURE = "failure"           # Échec (apprendre des erreurs)
    DISCOVERY = "discovery"       # Découverte (nouveau lieu, mécanique)
    OBSERVATION = "observation"   # Observation d'autres joueurs
    SURPRISE = "surprise"         # Événement inattendu
    ROUTINE = "routine"           # Expérience routinière


class LearningMode(Enum):
    """Modes d'apprentissage"""
    EXPLORATION = "exploration"   # Explorer activement pour apprendre
    EXPLOITATION = "exploitation" # Utiliser ce qui est appris
    IMITATION = "imitation"       # Apprendre par observation
    REFLECTION = "reflection"     # Réflexion sur les expériences passées


@dataclass
class Experience:
    """Une expérience vécue (équivalent d'une mémoire épisodique)"""
    timestamp: float
    experience_type: ExperienceType

    # Contexte (où, quand, quoi)
    state_before: Dict[str, Any]
    action_taken: str
    state_after: Dict[str, Any]

    # Résultat
    reward: float                 # Récompense immédiate
    long_term_value: float = 0.0  # Valeur à long terme (mise à jour plus tard)

    # Métadonnées
    surprise_level: float = 0.0   # Niveau de surprise (inattendu?)
    learning_potential: float = 0.0  # Potentiel d'apprentissage
    emotional_impact: float = 0.0    # Impact émotionnel
    replay_count: int = 0            # Nombre de fois rejouée

    # Annotations
    what_learned: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class LearningGoal:
    """Objectif d'apprentissage (ce que je veux apprendre)"""
    goal_name: str
    description: str
    priority: float
    progress: float = 0.0
    target_mastery: float = 0.8
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    sub_goals: List[str] = field(default_factory=list)


class ExperienceReplayBuffer:
    """
    Mémoire d'expériences avec replay intelligent
    Inspiré de Experience Replay de DeepMind, mais adapté pour apprentissage continu
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.experiences = deque(maxlen=max_size)

        # Index pour accès rapide
        self.experiences_by_type = defaultdict(list)
        self.high_value_experiences = []  # Expériences importantes

        # Statistiques
        self.total_experiences = 0
        self.replay_stats = defaultdict(int)

    def add_experience(self, experience: Experience):
        """Ajoute une expérience à la mémoire"""
        self.experiences.append(experience)
        self.experiences_by_type[experience.experience_type].append(experience)

        # Marquer les expériences de haute valeur
        if experience.learning_potential > 0.7 or experience.surprise_level > 0.8:
            self.high_value_experiences.append(experience)

        self.total_experiences += 1

    def sample_for_learning(
        self,
        batch_size: int = 32,
        prioritize_surprises: bool = True,
        prioritize_failures: bool = True
    ) -> List[Experience]:
        """
        Échantillonne des expériences pour l'apprentissage
        Priorité aux expériences surprenantes et aux échecs (plus informatifs)
        """
        if len(self.experiences) < batch_size:
            return list(self.experiences)

        # Calculer les probabilités d'échantillonnage
        priorities = []
        for exp in self.experiences:
            priority = 1.0  # Base

            # Bonus pour les surprises
            if prioritize_surprises:
                priority += exp.surprise_level * 2.0

            # Bonus pour les échecs (apprendre des erreurs)
            if prioritize_failures and exp.experience_type == ExperienceType.FAILURE:
                priority += 1.5

            # Bonus pour les découvertes
            if exp.experience_type == ExperienceType.DISCOVERY:
                priority += 1.0

            # Pénalité pour les expériences déjà beaucoup rejouées
            priority /= (1.0 + exp.replay_count * 0.1)

            priorities.append(priority)

        # Normaliser les probabilités
        priorities = np.array(priorities)
        probabilities = priorities / priorities.sum()

        # Échantillonner
        indices = np.random.choice(
            len(self.experiences),
            size=min(batch_size, len(self.experiences)),
            replace=False,
            p=probabilities
        )

        sampled = [self.experiences[i] for i in indices]

        # Mettre à jour les compteurs de replay
        for exp in sampled:
            exp.replay_count += 1
            self.replay_stats[exp.experience_type] += 1

        return sampled

    def get_experiences_by_type(self, exp_type: ExperienceType, limit: int = 100) -> List[Experience]:
        """Récupère les expériences d'un type donné"""
        return self.experiences_by_type[exp_type][-limit:]

    def get_high_value_experiences(self, limit: int = 50) -> List[Experience]:
        """Récupère les expériences de haute valeur"""
        return self.high_value_experiences[-limit:]


class CuriosityDrivenExploration:
    """
    Système d'exploration guidée par la curiosité
    Le bot explore activement pour découvrir de nouvelles choses
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.CuriosityDrivenExploration")

        # Modèle de prédiction (pour détecter la nouveauté)
        self.novelty_detector = self._init_novelty_detector()

        # Historique des états visités
        self.visited_states = set()
        self.state_visit_counts = defaultdict(int)

        # Carte de curiosité (où explorer?)
        self.curiosity_map = {}

        # Métriques
        self.exploration_metrics = {
            "novel_states_discovered": 0,
            "curiosity_rewards": 0.0,
            "exploration_efficiency": 0.0
        }

    def _init_novelty_detector(self) -> nn.Module:
        """Initialise un détecteur de nouveauté simple"""
        # Réseau simple pour prédire les états suivants
        # Si la prédiction est mauvaise = état nouveau/surprenant
        class NoveltyDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(128, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 128)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.fc3(x)

        return NoveltyDetector()

    def compute_intrinsic_reward(self, state: Dict[str, Any]) -> float:
        """
        Calcule une récompense intrinsèque basée sur la nouveauté
        Plus l'état est nouveau/inattendu, plus la récompense est élevée
        """
        # Représentation simplifiée de l'état
        state_key = self._state_to_key(state)

        # Récompense inversement proportionnelle au nombre de visites
        visit_count = self.state_visit_counts[state_key]
        intrinsic_reward = 1.0 / (1.0 + visit_count)

        # Bonus pour les états jamais vus
        if state_key not in self.visited_states:
            intrinsic_reward += 1.0
            self.visited_states.add(state_key)
            self.exploration_metrics["novel_states_discovered"] += 1

        # Enregistrer la visite
        self.state_visit_counts[state_key] += 1

        self.exploration_metrics["curiosity_rewards"] += intrinsic_reward

        return intrinsic_reward

    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """Convertit un état en clé hashable"""
        # Simplification: utiliser map + position
        map_name = state.get("current_map", "unknown")
        pos = state.get("character", {}).get("position", (0, 0))
        return f"{map_name}_{pos}"

    def should_explore(self, current_curiosity: float, exploitation_value: float) -> bool:
        """
        Décide s'il faut explorer ou exploiter
        Balance entre découvrir de nouvelles choses et utiliser ce qui est connu
        """
        # Stratégie epsilon-greedy adaptative
        epsilon = max(0.1, current_curiosity)  # Plus je suis curieux, plus j'explore

        if np.random.random() < epsilon:
            return True  # Explorer
        else:
            return exploitation_value < 0.5  # Exploiter si la valeur est élevée


class MetaLearner:
    """
    Meta-apprentissage: Apprendre à mieux apprendre
    Le bot adapte sa façon d'apprendre basée sur son expérience
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.MetaLearner")

        # Paramètres d'apprentissage adaptatifs
        self.learning_params = {
            "learning_rate": 0.001,
            "exploration_rate": 0.3,
            "memory_retention": 0.9,
            "replay_frequency": 10,
            "batch_size": 32
        }

        # Historique de performance
        self.performance_history = deque(maxlen=1000)

        # Stratégies d'apprentissage essayées
        self.learning_strategies = {}

    def adapt_learning_rate(self, recent_performance: float):
        """Adapte le taux d'apprentissage basé sur la performance"""
        if len(self.performance_history) < 10:
            return

        # Si la performance s'améliore, continuer
        # Si la performance stagne ou baisse, ajuster
        recent_avg = np.mean(list(self.performance_history)[-10:])
        older_avg = np.mean(list(self.performance_history)[-20:-10]) if len(self.performance_history) >= 20 else recent_avg

        if recent_avg > older_avg:
            # Performance en hausse: bon taux d'apprentissage
            self.logger.debug(f"Performance en hausse, learning_rate stable: {self.learning_params['learning_rate']}")
        elif recent_avg < older_avg * 0.95:
            # Performance en baisse: augmenter le learning rate (explorer plus)
            self.learning_params['learning_rate'] *= 1.1
            self.learning_params['exploration_rate'] = min(0.5, self.learning_params['exploration_rate'] * 1.2)
            self.logger.info(f"Performance en baisse, augmentation exploration: {self.learning_params['exploration_rate']:.3f}")
        else:
            # Performance stable: réduire légèrement le learning rate (stabiliser)
            self.learning_params['learning_rate'] *= 0.99

        # Contraintes
        self.learning_params['learning_rate'] = np.clip(self.learning_params['learning_rate'], 0.0001, 0.01)
        self.learning_params['exploration_rate'] = np.clip(self.learning_params['exploration_rate'], 0.05, 0.5)

    def record_performance(self, performance_metric: float):
        """Enregistre une métrique de performance"""
        self.performance_history.append({
            "timestamp": time.time(),
            "performance": performance_metric
        })

    def get_optimal_strategy(self, context: str) -> Dict[str, Any]:
        """Retourne la stratégie optimale pour un contexte donné"""
        if context not in self.learning_strategies:
            # Stratégie par défaut
            return {
                "approach": "balanced",
                "focus": "general_learning",
                "params": self.learning_params.copy()
            }

        return self.learning_strategies[context]


class ContinuousLearningEngine:
    """
    Moteur d'apprentissage continu autonome

    Le bot apprend continuellement de ses expériences comme un humain:
    1. Expériences → Mémoire
    2. Replay des expériences importantes
    3. Mise à jour des connaissances
    4. Adaptation des stratégies
    5. Exploration guidée par la curiosité
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ContinuousLearningEngine")

        # Composants principaux
        self.experience_buffer = ExperienceReplayBuffer(max_size=10000)
        self.curiosity_system = CuriosityDrivenExploration()
        self.meta_learner = MetaLearner()

        # Objectifs d'apprentissage
        self.learning_goals = []
        self.current_focus = None

        # Mode d'apprentissage actuel
        self.current_mode = LearningMode.EXPLORATION

        # Connaissances acquises
        self.learned_strategies = {}
        self.learned_patterns = {}
        self.skill_mastery_levels = defaultdict(float)

        # Métriques d'apprentissage
        self.learning_metrics = {
            "total_experiences": 0,
            "learning_sessions": 0,
            "skills_learned": 0,
            "average_learning_rate": 0.0,
            "knowledge_growth": 0.0
        }

        # Configuration
        self.config = {
            "replay_interval": 60.0,     # Replay toutes les 60 secondes
            "learning_batch_size": 32,
            "min_experiences_for_learning": 10,
            "curiosity_weight": 0.3,     # Poids de la curiosité dans les décisions
            "reflection_enabled": True
        }

        self.last_replay_time = 0.0

        self.logger.info("Continuous Learning Engine initialized - Prêt à apprendre!")

    def record_experience(
        self,
        state_before: Dict[str, Any],
        action: str,
        state_after: Dict[str, Any],
        reward: float,
        experience_type: ExperienceType = ExperienceType.ROUTINE
    ) -> Experience:
        """
        Enregistre une expérience vécue
        C'est l'équivalent de former une mémoire
        """
        # Calculer la surprise (différence entre attendu et réel)
        surprise_level = self._compute_surprise(state_before, action, state_after)

        # Calculer le potentiel d'apprentissage
        learning_potential = self._compute_learning_potential(
            experience_type,
            surprise_level,
            reward
        )

        # Récompense intrinsèque de curiosité
        curiosity_reward = self.curiosity_system.compute_intrinsic_reward(state_after)

        # Récompense totale = récompense extrinsèque + intrinsèque
        total_reward = reward + self.config["curiosity_weight"] * curiosity_reward

        # Impact émotionnel
        emotional_impact = abs(total_reward) + surprise_level

        # Créer l'expérience
        experience = Experience(
            timestamp=time.time(),
            experience_type=experience_type,
            state_before=state_before,
            action_taken=action,
            state_after=state_after,
            reward=total_reward,
            surprise_level=surprise_level,
            learning_potential=learning_potential,
            emotional_impact=emotional_impact
        )

        # Ajouter à la mémoire
        self.experience_buffer.add_experience(experience)
        self.learning_metrics["total_experiences"] += 1

        # Log des expériences importantes
        if learning_potential > 0.7:
            self.logger.info(f"🎯 Expérience importante enregistrée: {experience_type.value} (potentiel: {learning_potential:.2f})")

        return experience

    def learn_from_experiences(self, force: bool = False) -> Dict[str, Any]:
        """
        Apprend à partir des expériences stockées (experience replay)
        Similaire à la consolidation de mémoire pendant le sommeil humain
        """
        current_time = time.time()

        # Vérifier si c'est le moment d'apprendre
        if not force and current_time - self.last_replay_time < self.config["replay_interval"]:
            return {"status": "waiting", "next_replay_in": self.config["replay_interval"] - (current_time - self.last_replay_time)}

        # Vérifier si on a assez d'expériences
        if len(self.experience_buffer.experiences) < self.config["min_experiences_for_learning"]:
            return {"status": "insufficient_data", "experiences_needed": self.config["min_experiences_for_learning"] - len(self.experience_buffer.experiences)}

        self.logger.info(f"📚 Session d'apprentissage - {len(self.experience_buffer.experiences)} expériences disponibles")

        # Échantillonner des expériences pour l'apprentissage
        batch = self.experience_buffer.sample_for_learning(
            batch_size=self.config["learning_batch_size"],
            prioritize_surprises=True,
            prioritize_failures=True
        )

        # Apprendre de chaque expérience
        insights = []
        for exp in batch:
            insight = self._extract_insight_from_experience(exp)
            if insight:
                insights.append(insight)

        # Mettre à jour les connaissances
        knowledge_updates = self._update_knowledge_base(insights)

        # Adaptation méta-apprentissage
        avg_performance = np.mean([exp.reward for exp in batch])
        self.meta_learner.record_performance(avg_performance)
        self.meta_learner.adapt_learning_rate(avg_performance)

        # Mise à jour des métriques
        self.learning_metrics["learning_sessions"] += 1
        self.learning_metrics["knowledge_growth"] += len(insights)

        self.last_replay_time = current_time

        return {
            "status": "learned",
            "experiences_processed": len(batch),
            "insights_gained": len(insights),
            "knowledge_updates": knowledge_updates,
            "average_reward": avg_performance,
            "learning_params": self.meta_learner.learning_params
        }

    def _compute_surprise(self, state_before: Dict, action: str, state_after: Dict) -> float:
        """Calcule le niveau de surprise (inattendu?)"""
        # Simple heuristique: compare HP avant/après
        # Un changement inattendu = surprise
        hp_before = state_before.get("character", {}).get("hp_percent", 100)
        hp_after = state_after.get("character", {}).get("hp_percent", 100)

        hp_change = abs(hp_after - hp_before)

        # Perte de HP inattendue = surprise élevée
        if hp_change > 30:
            return min(1.0, hp_change / 50.0)

        # Changement de map = surprise modérée
        if state_before.get("current_map") != state_after.get("current_map"):
            return 0.5

        return 0.1  # Surprise faible par défaut

    def _compute_learning_potential(
        self,
        exp_type: ExperienceType,
        surprise: float,
        reward: float
    ) -> float:
        """Calcule le potentiel d'apprentissage d'une expérience"""
        potential = 0.0

        # Les échecs ont un fort potentiel d'apprentissage
        if exp_type == ExperienceType.FAILURE:
            potential += 0.8

        # Les découvertes aussi
        elif exp_type == ExperienceType.DISCOVERY:
            potential += 0.7

        # Les surprises sont informatives
        potential += surprise * 0.5

        # Récompenses extrêmes (très bonnes ou très mauvaises)
        potential += min(0.3, abs(reward) / 10.0)

        return min(1.0, potential)

    def _extract_insight_from_experience(self, exp: Experience) -> Optional[Dict[str, Any]]:
        """Extrait un insight (apprentissage) d'une expérience"""
        # Analyser l'expérience pour en tirer un enseignement
        insight = {
            "type": exp.experience_type.value,
            "lesson": "",
            "context": {},
            "confidence": 0.5
        }

        # Échecs: apprendre ce qu'il ne faut PAS faire
        if exp.experience_type == ExperienceType.FAILURE:
            insight["lesson"] = f"Action '{exp.action_taken}' a mené à un échec dans ce contexte"
            insight["context"] = {
                "state": exp.state_before,
                "action_to_avoid": exp.action_taken
            }
            insight["confidence"] = 0.7

        # Succès: renforcer ce qui marche
        elif exp.experience_type == ExperienceType.SUCCESS:
            if exp.reward > 0.5:
                insight["lesson"] = f"Action '{exp.action_taken}' efficace dans ce contexte"
                insight["context"] = {
                    "state": exp.state_before,
                    "good_action": exp.action_taken
                }
                insight["confidence"] = 0.8

        # Découvertes: enrichir la carte mentale du monde
        elif exp.experience_type == ExperienceType.DISCOVERY:
            insight["lesson"] = "Nouveau lieu ou mécanique découvert"
            insight["context"] = {
                "new_state": exp.state_after
            }
            insight["confidence"] = 0.9

        else:
            return None

        exp.what_learned = insight["lesson"]
        return insight

    def _update_knowledge_base(self, insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Met à jour la base de connaissances avec les nouveaux insights"""
        updates = {
            "new_strategies": 0,
            "updated_patterns": 0,
            "skills_improved": 0
        }

        for insight in insights:
            lesson_type = insight["type"]

            # Mise à jour des stratégies
            if lesson_type in ["success", "failure"]:
                action = insight["context"].get("good_action") or insight["context"].get("action_to_avoid")
                if action:
                    if action not in self.learned_strategies:
                        self.learned_strategies[action] = {
                            "success_count": 0,
                            "failure_count": 0,
                            "contexts": []
                        }

                    if lesson_type == "success":
                        self.learned_strategies[action]["success_count"] += 1
                    else:
                        self.learned_strategies[action]["failure_count"] += 1

                    self.learned_strategies[action]["contexts"].append(insight["context"])
                    updates["new_strategies"] += 1

            # Mise à jour des patterns
            pattern_key = f"{lesson_type}_{hash(str(insight['context']))}"
            if pattern_key not in self.learned_patterns:
                self.learned_patterns[pattern_key] = {
                    "pattern": insight,
                    "occurrences": 1,
                    "confidence": insight["confidence"]
                }
                updates["updated_patterns"] += 1
            else:
                self.learned_patterns[pattern_key]["occurrences"] += 1
                # Augmenter la confiance avec les répétitions
                self.learned_patterns[pattern_key]["confidence"] = min(
                    1.0,
                    self.learned_patterns[pattern_key]["confidence"] + 0.1
                )

        return updates

    def set_learning_goal(self, goal: LearningGoal):
        """Définit un objectif d'apprentissage"""
        self.learning_goals.append(goal)
        self.logger.info(f"🎯 Nouvel objectif d'apprentissage: {goal.goal_name}")

    def get_learning_progress(self) -> Dict[str, Any]:
        """Retourne la progression de l'apprentissage"""
        return {
            "metrics": self.learning_metrics,
            "active_goals": len(self.learning_goals),
            "knowledge_base_size": {
                "strategies": len(self.learned_strategies),
                "patterns": len(self.learned_patterns)
            },
            "learning_mode": self.current_mode.value,
            "meta_learning_params": self.meta_learner.learning_params
        }

    def should_explore_now(self, current_curiosity: float) -> bool:
        """Décide s'il faut explorer (apprendre) ou exploiter (utiliser)"""
        return self.curiosity_system.should_explore(
            current_curiosity,
            self.meta_learner.learning_params["exploration_rate"]
        )

    def get_state(self) -> Dict[str, Any]:
        """Retourne l'état complet du système d'apprentissage"""
        return {
            "metrics": self.learning_metrics,
            "buffer_size": len(self.experience_buffer.experiences),
            "learned_strategies": len(self.learned_strategies),
            "learned_patterns": len(self.learned_patterns),
            "current_mode": self.current_mode.value,
            "learning_goals": [g.goal_name for g in self.learning_goals]
        }


def create_continuous_learning_engine() -> ContinuousLearningEngine:
    """Factory function"""
    return ContinuousLearningEngine()
