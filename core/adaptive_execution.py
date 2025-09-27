"""
Module d'Exécution Adaptative pour l'IA DOFUS Évolutive
Phase 3 : Exécution Adaptative & Sociale

Ce module gère l'adaptation dynamique du comportement,
l'optimisation continue et l'apprentissage en temps réel.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
import json
import time
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AdaptationTrigger(Enum):
    PERFORMANCE_DROP = "performance_drop"
    CONTEXT_CHANGE = "context_change"
    FAILURE_PATTERN = "failure_pattern"
    OPPORTUNITY_DETECTED = "opportunity_detected"
    SCHEDULE_UPDATE = "schedule_update"
    USER_FEEDBACK = "user_feedback"

class ExecutionStyle(Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    STEALTH = "stealth"
    EFFICIENT = "efficient"
    SOCIAL = "social"

class OptimizationMetric(Enum):
    XP_PER_HOUR = "xp_per_hour"
    KAMAS_PER_HOUR = "kamas_per_hour"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    SAFETY_SCORE = "safety_score"
    SOCIAL_REPUTATION = "social_reputation"
    TASK_COMPLETION_RATE = "task_completion_rate"

@dataclass
class PerformanceMetrics:
    """Métriques de performance détaillées"""
    xp_gained: float = 0.0
    kamas_earned: float = 0.0
    resources_gathered: int = 0
    tasks_completed: int = 0
    failures_count: int = 0
    social_interactions: int = 0
    efficiency_score: float = 0.5
    safety_incidents: int = 0
    adaptation_count: int = 0
    learning_rate: float = 0.1

@dataclass
class AdaptationRule:
    """Règle d'adaptation comportementale"""
    trigger: AdaptationTrigger
    condition: Callable[[Dict[str, Any]], bool]
    adaptation: Callable[[Dict[str, Any]], Dict[str, Any]]
    priority: float = 0.5
    cooldown: timedelta = timedelta(minutes=5)
    last_triggered: Optional[datetime] = None

@dataclass
class ExecutionPlan:
    """Plan d'exécution adaptable"""
    primary_objective: str
    execution_style: ExecutionStyle
    time_horizon: timedelta
    priority_weights: Dict[OptimizationMetric, float]
    constraints: Dict[str, Any]
    fallback_strategies: List[str]
    adaptation_sensitivity: float = 0.7

class BehaviorLearningEngine:
    """Moteur d'apprentissage comportemental"""

    def __init__(self):
        self.behavior_history: List[Dict[str, Any]] = []
        self.success_patterns: Dict[str, float] = {}
        self.failure_patterns: Dict[str, float] = {}
        self.context_mappings: Dict[str, ExecutionStyle] = {}

        # Paramètres d'apprentissage
        self.learning_rate = 0.1
        self.memory_decay = 0.95
        self.exploration_rate = 0.1

        # Métriques d'apprentissage
        self.patterns_learned = 0
        self.adaptations_successful = 0
        self.total_adaptations = 0

    async def learn_from_outcome(self, context: Dict[str, Any],
                                actions: List[str], outcome: Dict[str, Any]):
        """Apprend d'un résultat d'exécution"""
        try:
            behavior_key = self._generate_behavior_key(context, actions)
            success_score = self._evaluate_outcome_success(outcome)

            # Mise à jour des patterns
            if success_score > 0.7:
                self.success_patterns[behavior_key] = (
                    self.success_patterns.get(behavior_key, 0.5) * (1 - self.learning_rate) +
                    success_score * self.learning_rate
                )
            elif success_score < 0.3:
                self.failure_patterns[behavior_key] = (
                    self.failure_patterns.get(behavior_key, 0.5) * (1 - self.learning_rate) +
                    (1 - success_score) * self.learning_rate
                )

            # Enregistrer dans l'historique
            self.behavior_history.append({
                "timestamp": datetime.now().isoformat(),
                "context": context,
                "actions": actions,
                "outcome": outcome,
                "success_score": success_score,
                "behavior_key": behavior_key
            })

            # Limiter la taille de l'historique
            if len(self.behavior_history) > 1000:
                self.behavior_history = self.behavior_history[-800:]

            self.patterns_learned += 1
            logger.info(f"Apprentissage comportemental: {behavior_key} -> {success_score:.2f}")

        except Exception as e:
            logger.error(f"Erreur apprentissage comportemental: {e}")

    def _generate_behavior_key(self, context: Dict[str, Any], actions: List[str]) -> str:
        """Génère une clé unique pour un comportement"""
        try:
            context_str = "_".join([f"{k}:{v}" for k, v in sorted(context.items())])
            actions_str = "_".join(sorted(actions))
            return f"{context_str}|{actions_str}"
        except:
            return "unknown_behavior"

    def _evaluate_outcome_success(self, outcome: Dict[str, Any]) -> float:
        """Évalue le succès d'un résultat"""
        try:
            success_factors = []

            # Facteurs de succès
            if outcome.get("task_completed", False):
                success_factors.append(0.4)
            if outcome.get("efficiency", 0) > 0.7:
                success_factors.append(0.3)
            if outcome.get("safety_maintained", True):
                success_factors.append(0.2)
            if outcome.get("no_errors", True):
                success_factors.append(0.1)

            return sum(success_factors) if success_factors else 0.5

        except Exception as e:
            logger.error(f"Erreur évaluation succès: {e}")
            return 0.5

    async def recommend_behavior(self, context: Dict[str, Any]) -> ExecutionStyle:
        """Recommande un style d'exécution basé sur l'apprentissage"""
        try:
            # Chercher des patterns similaires
            best_match = None
            best_score = 0.0

            for behavior_key, success_rate in self.success_patterns.items():
                similarity = self._calculate_context_similarity(context, behavior_key)
                score = similarity * success_rate

                if score > best_score:
                    best_score = score
                    best_match = behavior_key

            # Extraire le style recommandé
            if best_match and best_score > 0.6:
                return self._extract_style_from_behavior(best_match)
            else:
                # Exploration ou style par défaut
                if np.random.random() < self.exploration_rate:
                    return np.random.choice(list(ExecutionStyle))
                else:
                    return ExecutionStyle.BALANCED

        except Exception as e:
            logger.error(f"Erreur recommandation comportement: {e}")
            return ExecutionStyle.BALANCED

    def _calculate_context_similarity(self, context: Dict[str, Any], behavior_key: str) -> float:
        """Calcule la similarité entre contextes"""
        try:
            # Extraction du contexte du behavior_key
            context_part = behavior_key.split("|")[0]
            stored_context = dict(item.split(":") for item in context_part.split("_") if ":" in item)

            # Calcul de similarité simple
            common_keys = set(context.keys()) & set(stored_context.keys())
            if not common_keys:
                return 0.0

            matches = sum(1 for key in common_keys
                         if str(context[key]) == stored_context[key])

            return matches / len(common_keys)

        except Exception as e:
            logger.error(f"Erreur calcul similarité: {e}")
            return 0.0

    def _extract_style_from_behavior(self, behavior_key: str) -> ExecutionStyle:
        """Extrait le style d'exécution d'un comportement"""
        try:
            # Logique d'extraction basée sur les actions
            actions_part = behavior_key.split("|")[1]

            if "aggressive" in actions_part or "fast" in actions_part:
                return ExecutionStyle.AGGRESSIVE
            elif "safe" in actions_part or "careful" in actions_part:
                return ExecutionStyle.CONSERVATIVE
            elif "social" in actions_part or "group" in actions_part:
                return ExecutionStyle.SOCIAL
            elif "stealth" in actions_part or "avoid" in actions_part:
                return ExecutionStyle.STEALTH
            else:
                return ExecutionStyle.BALANCED

        except Exception as e:
            logger.error(f"Erreur extraction style: {e}")
            return ExecutionStyle.BALANCED

class AdaptiveExecutionEngine:
    """Moteur principal d'exécution adaptative"""

    def __init__(self):
        self.current_plan: Optional[ExecutionPlan] = None
        self.performance_metrics = PerformanceMetrics()
        self.adaptation_rules: List[AdaptationRule] = []
        self.behavior_engine = BehaviorLearningEngine()

        # Configuration adaptative
        self.adaptation_config = {
            "adaptation_enabled": True,
            "learning_enabled": True,
            "auto_optimization": True,
            "safety_first": True,
            "exploration_rate": 0.1
        }

        # Historique des adaptations
        self.adaptation_history: List[Dict[str, Any]] = []

        # Initialiser les règles par défaut
        self._initialize_default_rules()

    def _initialize_default_rules(self):
        """Initialise les règles d'adaptation par défaut"""
        try:
            # Règle de performance faible
            self.adaptation_rules.append(AdaptationRule(
                trigger=AdaptationTrigger.PERFORMANCE_DROP,
                condition=lambda metrics: metrics.get("efficiency_score", 0.5) < 0.3,
                adaptation=self._adapt_for_performance,
                priority=0.9,
                cooldown=timedelta(minutes=10)
            ))

            # Règle de détection d'échec répété
            self.adaptation_rules.append(AdaptationRule(
                trigger=AdaptationTrigger.FAILURE_PATTERN,
                condition=lambda metrics: metrics.get("failures_count", 0) > 3,
                adaptation=self._adapt_for_failures,
                priority=0.8,
                cooldown=timedelta(minutes=5)
            ))

            # Règle d'opportunité détectée
            self.adaptation_rules.append(AdaptationRule(
                trigger=AdaptationTrigger.OPPORTUNITY_DETECTED,
                condition=lambda metrics: metrics.get("opportunity_score", 0) > 0.8,
                adaptation=self._adapt_for_opportunity,
                priority=0.7,
                cooldown=timedelta(minutes=15)
            ))

            logger.info("Règles d'adaptation par défaut initialisées")

        except Exception as e:
            logger.error(f"Erreur initialisation règles: {e}")

    async def create_execution_plan(self, objective: str,
                                  context: Dict[str, Any]) -> ExecutionPlan:
        """Crée un plan d'exécution adaptatif"""
        try:
            # Recommandation de style basée sur l'apprentissage
            recommended_style = await self.behavior_engine.recommend_behavior(context)

            # Poids des métriques selon l'objectif
            if objective == "leveling":
                priority_weights = {
                    OptimizationMetric.XP_PER_HOUR: 0.4,
                    OptimizationMetric.SAFETY_SCORE: 0.3,
                    OptimizationMetric.TASK_COMPLETION_RATE: 0.2,
                    OptimizationMetric.RESOURCE_EFFICIENCY: 0.1
                }
            elif objective == "farming":
                priority_weights = {
                    OptimizationMetric.KAMAS_PER_HOUR: 0.4,
                    OptimizationMetric.RESOURCE_EFFICIENCY: 0.3,
                    OptimizationMetric.SAFETY_SCORE: 0.2,
                    OptimizationMetric.TASK_COMPLETION_RATE: 0.1
                }
            else:
                priority_weights = {metric: 0.25 for metric in OptimizationMetric}

            plan = ExecutionPlan(
                primary_objective=objective,
                execution_style=recommended_style,
                time_horizon=timedelta(hours=2),
                priority_weights=priority_weights,
                constraints=context.get("constraints", {}),
                fallback_strategies=["safe_mode", "basic_routine", "manual_override"]
            )

            self.current_plan = plan
            logger.info(f"Plan d'exécution créé: {objective} avec style {recommended_style.value}")

            return plan

        except Exception as e:
            logger.error(f"Erreur création plan d'exécution: {e}")
            return ExecutionPlan(
                primary_objective=objective,
                execution_style=ExecutionStyle.BALANCED,
                time_horizon=timedelta(hours=1),
                priority_weights={metric: 0.25 for metric in OptimizationMetric},
                constraints={},
                fallback_strategies=["safe_mode"]
            )

    async def execute_adaptive_cycle(self, game_state: Dict[str, Any]) -> List[str]:
        """Exécute un cycle d'adaptation basé sur l'état du jeu"""
        try:
            if not self.current_plan:
                return []

            # Mise à jour des métriques
            await self._update_performance_metrics(game_state)

            # Vérification des règles d'adaptation
            adaptations_triggered = await self._check_adaptation_triggers(game_state)

            # Application des adaptations
            actions = []
            for adaptation in adaptations_triggered:
                adapted_actions = await self._apply_adaptation(adaptation, game_state)
                actions.extend(adapted_actions)

            # Apprentissage continu
            if actions:
                await self._learn_from_execution(game_state, actions)

            return actions

        except Exception as e:
            logger.error(f"Erreur cycle adaptatif: {e}")
            return []

    async def _update_performance_metrics(self, game_state: Dict[str, Any]):
        """Met à jour les métriques de performance"""
        try:
            # Mise à jour basée sur l'état du jeu
            if "xp_gained" in game_state:
                self.performance_metrics.xp_gained += game_state["xp_gained"]

            if "kamas_earned" in game_state:
                self.performance_metrics.kamas_earned += game_state["kamas_earned"]

            if "task_completed" in game_state and game_state["task_completed"]:
                self.performance_metrics.tasks_completed += 1

            if "error_occurred" in game_state and game_state["error_occurred"]:
                self.performance_metrics.failures_count += 1

            # Calcul du score d'efficacité
            total_time = max(1, game_state.get("elapsed_time", 1))
            self.performance_metrics.efficiency_score = (
                self.performance_metrics.tasks_completed / total_time
            )

        except Exception as e:
            logger.error(f"Erreur mise à jour métriques: {e}")

    async def _check_adaptation_triggers(self, game_state: Dict[str, Any]) -> List[AdaptationRule]:
        """Vérifie les déclencheurs d'adaptation"""
        try:
            triggered_rules = []
            current_time = datetime.now()

            metrics_dict = {
                "efficiency_score": self.performance_metrics.efficiency_score,
                "failures_count": self.performance_metrics.failures_count,
                "tasks_completed": self.performance_metrics.tasks_completed,
                "opportunity_score": game_state.get("opportunity_score", 0.0)
            }

            for rule in self.adaptation_rules:
                # Vérifier le cooldown
                if (rule.last_triggered and
                    current_time - rule.last_triggered < rule.cooldown):
                    continue

                # Vérifier la condition
                if rule.condition(metrics_dict):
                    rule.last_triggered = current_time
                    triggered_rules.append(rule)

            # Trier par priorité
            triggered_rules.sort(key=lambda r: r.priority, reverse=True)

            return triggered_rules

        except Exception as e:
            logger.error(f"Erreur vérification déclencheurs: {e}")
            return []

    async def _apply_adaptation(self, rule: AdaptationRule,
                              game_state: Dict[str, Any]) -> List[str]:
        """Applique une adaptation"""
        try:
            adaptation_result = rule.adaptation(game_state)

            # Enregistrer l'adaptation
            self.adaptation_history.append({
                "timestamp": datetime.now().isoformat(),
                "trigger": rule.trigger.value,
                "adaptation": adaptation_result,
                "game_state_hash": hash(str(sorted(game_state.items())))
            })

            self.performance_metrics.adaptation_count += 1

            logger.info(f"Adaptation appliquée: {rule.trigger.value}")

            return adaptation_result.get("actions", [])

        except Exception as e:
            logger.error(f"Erreur application adaptation: {e}")
            return []

    def _adapt_for_performance(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptation pour améliorer les performances"""
        try:
            return {
                "actions": ["reduce_complexity", "focus_primary_task", "optimize_route"],
                "style_change": ExecutionStyle.EFFICIENT,
                "reason": "Performance trop faible détectée"
            }
        except Exception as e:
            logger.error(f"Erreur adaptation performance: {e}")
            return {"actions": ["safe_mode"]}

    def _adapt_for_failures(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptation pour réduire les échecs"""
        try:
            return {
                "actions": ["enable_safety_mode", "reduce_risk", "fallback_strategy"],
                "style_change": ExecutionStyle.CONSERVATIVE,
                "reason": "Pattern d'échecs détecté"
            }
        except Exception as e:
            logger.error(f"Erreur adaptation échecs: {e}")
            return {"actions": ["safe_mode"]}

    def _adapt_for_opportunity(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptation pour saisir une opportunité"""
        try:
            return {
                "actions": ["seize_opportunity", "optimize_for_gain", "temporary_focus"],
                "style_change": ExecutionStyle.AGGRESSIVE,
                "reason": "Opportunité détectée"
            }
        except Exception as e:
            logger.error(f"Erreur adaptation opportunité: {e}")
            return {"actions": ["continue_normal"]}

    async def _learn_from_execution(self, game_state: Dict[str, Any], actions: List[str]):
        """Apprend de l'exécution en cours"""
        try:
            # Préparer le contexte d'apprentissage
            context = {
                "game_state": game_state.get("scene_type", "unknown"),
                "current_style": self.current_plan.execution_style.value if self.current_plan else "balanced",
                "time_of_day": datetime.now().hour,
                "performance_level": "high" if self.performance_metrics.efficiency_score > 0.7 else "low"
            }

            # Préparer le résultat
            outcome = {
                "task_completed": game_state.get("task_completed", False),
                "efficiency": self.performance_metrics.efficiency_score,
                "safety_maintained": self.performance_metrics.safety_incidents == 0,
                "no_errors": self.performance_metrics.failures_count == 0
            }

            # Apprentissage
            await self.behavior_engine.learn_from_outcome(context, actions, outcome)

        except Exception as e:
            logger.error(f"Erreur apprentissage exécution: {e}")

class ContinuousOptimizer:
    """Optimiseur continu pour l'amélioration des performances"""

    def __init__(self):
        self.optimization_targets: Dict[OptimizationMetric, float] = {}
        self.optimization_history: List[Dict[str, Any]] = []
        self.current_baseline: Dict[OptimizationMetric, float] = {}

    async def set_optimization_targets(self, targets: Dict[OptimizationMetric, float]):
        """Définit les objectifs d'optimisation"""
        try:
            self.optimization_targets = targets
            logger.info(f"Objectifs d'optimisation définis: {targets}")

        except Exception as e:
            logger.error(f"Erreur définition objectifs: {e}")

    async def optimize_parameters(self, current_metrics: Dict[OptimizationMetric, float]) -> Dict[str, Any]:
        """Optimise les paramètres basés sur les métriques actuelles"""
        try:
            optimizations = {}

            for metric, target in self.optimization_targets.items():
                current_value = current_metrics.get(metric, 0.0)

                if current_value < target * 0.8:  # 20% en dessous de l'objectif
                    optimization = await self._generate_optimization(metric, current_value, target)
                    optimizations[metric.value] = optimization

            logger.info(f"Optimisations générées: {len(optimizations)}")
            return optimizations

        except Exception as e:
            logger.error(f"Erreur optimisation paramètres: {e}")
            return {}

    async def _generate_optimization(self, metric: OptimizationMetric,
                                   current: float, target: float) -> Dict[str, Any]:
        """Génère une optimisation spécifique"""
        try:
            gap = target - current

            if metric == OptimizationMetric.XP_PER_HOUR:
                return {
                    "action": "optimize_xp_farming",
                    "parameters": {"efficiency_boost": min(gap / target, 0.5)},
                    "expected_improvement": gap * 0.7
                }
            elif metric == OptimizationMetric.SAFETY_SCORE:
                return {
                    "action": "enhance_safety_measures",
                    "parameters": {"safety_margin": min(gap, 0.3)},
                    "expected_improvement": gap * 0.8
                }
            else:
                return {
                    "action": "general_optimization",
                    "parameters": {"improvement_factor": min(gap / target, 0.3)},
                    "expected_improvement": gap * 0.5
                }

        except Exception as e:
            logger.error(f"Erreur génération optimisation: {e}")
            return {"action": "no_optimization", "expected_improvement": 0.0}

# Intégration principale
class AdaptiveExecutionModule:
    """Module principal d'exécution adaptative"""

    def __init__(self):
        self.execution_engine = AdaptiveExecutionEngine()
        self.optimizer = ContinuousOptimizer()
        self.is_active = False

    async def initialize(self, config: Dict[str, Any]):
        """Initialise le module d'exécution adaptative"""
        try:
            # Configuration du module
            self.execution_engine.adaptation_config.update(config.get("adaptation", {}))

            # Définition des objectifs d'optimisation
            targets = {
                OptimizationMetric.XP_PER_HOUR: config.get("target_xp_per_hour", 100000.0),
                OptimizationMetric.SAFETY_SCORE: config.get("target_safety", 0.9),
                OptimizationMetric.TASK_COMPLETION_RATE: config.get("target_completion", 0.85)
            }
            await self.optimizer.set_optimization_targets(targets)

            self.is_active = True
            logger.info("Module d'exécution adaptative initialisé")

        except Exception as e:
            logger.error(f"Erreur initialisation module adaptatif: {e}")

    async def process_adaptive_frame(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Traite un frame d'adaptation"""
        try:
            if not self.is_active:
                return {"actions": [], "adaptations": []}

            # Cycle d'exécution adaptative
            adaptive_actions = await self.execution_engine.execute_adaptive_cycle(game_state)

            # Optimisation continue
            current_metrics = {
                OptimizationMetric.XP_PER_HOUR: game_state.get("xp_per_hour", 0.0),
                OptimizationMetric.SAFETY_SCORE: game_state.get("safety_score", 0.5),
                OptimizationMetric.TASK_COMPLETION_RATE: game_state.get("completion_rate", 0.5)
            }
            optimizations = await self.optimizer.optimize_parameters(current_metrics)

            return {
                "actions": adaptive_actions,
                "optimizations": optimizations,
                "current_plan": self.execution_engine.current_plan.__dict__ if self.execution_engine.current_plan else None,
                "performance_metrics": self.execution_engine.performance_metrics.__dict__
            }

        except Exception as e:
            logger.error(f"Erreur traitement frame adaptatif: {e}")
            return {"actions": [], "adaptations": []}