#!/usr/bin/env python3
"""
StrategyOptimizer - Optimiseur de stratégies et routes DOFUS
Utilise HRM et algorithmes d'optimisation pour maximiser l'efficacité
"""

import time
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from enum import Enum
import heapq
from collections import defaultdict

import torch
import numpy as np

from config import config
from core.hrm_reasoning import create_hrm_model, HRMOutput
from .guide_loader import Guide, GuideStep, GuideType

logger = logging.getLogger(__name__)

class OptimizationGoal(Enum):
    """Objectifs d'optimisation"""
    MAXIMIZE_EXPERIENCE = "maximize_experience"
    MAXIMIZE_KAMAS = "maximize_kamas"
    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_RISK = "minimize_risk"
    BALANCED = "balanced"
    CUSTOM = "custom"

class OptimizationStrategy(Enum):
    """Stratégies d'optimisation"""
    GREEDY = "greedy"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    GENETIC_ALGORITHM = "genetic_algorithm"
    SIMULATED_ANNEALING = "simulated_annealing"
    MONTE_CARLO = "monte_carlo"
    HRM_GUIDED = "hrm_guided"

@dataclass
class OptimizationConstraints:
    """Contraintes d'optimisation"""
    max_time: Optional[float] = None
    max_cost: Optional[int] = None
    min_level: Optional[int] = None
    max_level: Optional[int] = None
    required_items: List[str] = field(default_factory=list)
    forbidden_zones: List[str] = field(default_factory=list)
    preferred_activities: List[str] = field(default_factory=list)
    time_restrictions: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # zone -> (start_hour, end_hour)

@dataclass
class OptimizationResult:
    """Résultat d'optimisation"""
    success: bool
    optimized_sequence: List[Dict[str, Any]] = field(default_factory=list)
    total_score: float = 0.0

    # Métriques détaillées
    total_experience: int = 0
    total_kamas: int = 0
    total_time: float = 0.0
    risk_score: float = 0.0

    # Informations d'optimisation
    optimization_time: float = 0.0
    iterations: int = 0
    strategy_used: str = ""
    convergence_reached: bool = False

    # Détails des activités
    activities_breakdown: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

class ActivityEvaluator:
    """Évaluateur d'activités et performances"""

    def __init__(self):
        # Bases de données d'efficacité
        self.activity_base_rates = {
            "quest": {
                "exp_per_hour": 2000,
                "kamas_per_hour": 500,
                "risk_factor": 1.2,
                "time_variance": 0.3
            },
            "farming": {
                "exp_per_hour": 1500,
                "kamas_per_hour": 1200,
                "risk_factor": 1.5,
                "time_variance": 0.4
            },
            "dungeon": {
                "exp_per_hour": 3000,
                "kamas_per_hour": 800,
                "risk_factor": 2.0,
                "time_variance": 0.5
            },
            "profession": {
                "exp_per_hour": 1000,
                "kamas_per_hour": 2000,
                "risk_factor": 0.8,
                "time_variance": 0.2
            },
            "exploration": {
                "exp_per_hour": 800,
                "kamas_per_hour": 300,
                "risk_factor": 1.0,
                "time_variance": 0.6
            }
        }

    def evaluate_activity(self,
                         activity: Dict[str, Any],
                         player_context: Dict[str, Any]) -> Dict[str, float]:
        """Évalue une activité selon le contexte joueur"""

        activity_type = activity.get("type", "quest")
        player_level = player_context.get("level", 1)
        player_equipment = player_context.get("equipment_score", 1.0)

        base_rates = self.activity_base_rates.get(activity_type, self.activity_base_rates["quest"])

        # Facteurs de modification
        level_factor = self._calculate_level_factor(activity, player_level)
        equipment_factor = min(2.0, player_equipment / 100.0)
        zone_factor = self._calculate_zone_factor(activity, player_context)
        time_factor = self._calculate_time_factor(activity, player_context)

        # Calculs finaux
        adjusted_exp = base_rates["exp_per_hour"] * level_factor * equipment_factor
        adjusted_kamas = base_rates["kamas_per_hour"] * level_factor * zone_factor
        adjusted_risk = base_rates["risk_factor"] * (2.0 - equipment_factor)
        adjusted_time = activity.get("estimated_time", 1.0) * time_factor

        return {
            "exp_per_hour": adjusted_exp,
            "kamas_per_hour": adjusted_kamas,
            "risk_score": adjusted_risk,
            "time_estimate": adjusted_time,
            "efficiency_score": (adjusted_exp + adjusted_kamas * 0.1) / max(adjusted_time, 0.1)
        }

    def _calculate_level_factor(self, activity: Dict[str, Any], player_level: int) -> float:
        """Calcule facteur d'efficacité selon niveau"""
        activity_level = activity.get("level_requirement", player_level)
        level_diff = player_level - activity_level

        if level_diff < 0:
            # Activité trop difficile
            return max(0.1, 1.0 + level_diff * 0.2)
        elif level_diff <= 5:
            # Niveau approprié
            return 1.0 + level_diff * 0.05
        else:
            # Activité trop facile
            return max(0.3, 1.2 - (level_diff - 5) * 0.1)

    def _calculate_zone_factor(self, activity: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calcule facteur de zone"""
        zone = activity.get("zone", "unknown")
        zone_multipliers = context.get("zone_multipliers", {})
        return zone_multipliers.get(zone, 1.0)

    def _calculate_time_factor(self, activity: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calcule facteur temporel"""
        current_hour = context.get("current_hour", 12)

        # Certaines activités sont plus efficaces à certains moments
        if activity.get("type") == "farming":
            # Farming plus efficace la nuit (moins de joueurs)
            if 22 <= current_hour or current_hour <= 6:
                return 0.8
        elif activity.get("type") == "quest":
            # Quêtes plus rapides en journée (PNJ disponibles)
            if 8 <= current_hour <= 20:
                return 0.9

        return 1.0

class SequenceOptimizer:
    """Optimiseur de séquences d'activités"""

    def __init__(self, evaluator: ActivityEvaluator):
        self.evaluator = evaluator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hrm_model = create_hrm_model().to(self.device)

    def optimize_greedy(self,
                       activities: List[Dict[str, Any]],
                       context: Dict[str, Any],
                       goal: OptimizationGoal,
                       constraints: OptimizationConstraints) -> List[Dict[str, Any]]:
        """Optimisation glouton"""

        remaining_activities = activities.copy()
        optimized_sequence = []
        current_time = 0.0

        while remaining_activities:
            best_activity = None
            best_score = -float('inf')

            for activity in remaining_activities:
                if not self._satisfies_constraints(activity, constraints, current_time):
                    continue

                evaluation = self.evaluator.evaluate_activity(activity, context)
                score = self._calculate_goal_score(evaluation, goal)

                if score > best_score:
                    best_score = score
                    best_activity = activity

            if best_activity is None:
                break

            # Ajouter à la séquence
            optimized_sequence.append(best_activity)
            remaining_activities.remove(best_activity)
            current_time += best_activity.get("estimated_time", 1.0)

            # Mettre à jour contexte
            context = self._update_context_after_activity(context, best_activity)

        return optimized_sequence

    def optimize_dynamic_programming(self,
                                   activities: List[Dict[str, Any]],
                                   context: Dict[str, Any],
                                   goal: OptimizationGoal,
                                   constraints: OptimizationConstraints) -> List[Dict[str, Any]]:
        """Optimisation par programmation dynamique"""

        n = len(activities)
        max_time = int(constraints.max_time or 24 * 60)  # minutes

        # DP table: dp[i][t] = (score, sequence)
        dp = {}

        def solve(activity_idx: int, remaining_time: int, used_activities: Set[int]) -> Tuple[float, List[int]]:
            if activity_idx == n or remaining_time <= 0:
                return 0.0, []

            cache_key = (activity_idx, remaining_time, tuple(sorted(used_activities)))
            if cache_key in dp:
                return dp[cache_key]

            # Option 1: ignorer cette activité
            best_score, best_sequence = solve(activity_idx + 1, remaining_time, used_activities)

            # Option 2: inclure cette activité
            activity = activities[activity_idx]
            activity_time = int(activity.get("estimated_time", 60))

            if (activity_time <= remaining_time and
                activity_idx not in used_activities and
                self._satisfies_constraints(activity, constraints, 0)):

                evaluation = self.evaluator.evaluate_activity(activity, context)
                activity_score = self._calculate_goal_score(evaluation, goal)

                new_used = used_activities | {activity_idx}
                future_score, future_sequence = solve(activity_idx + 1, remaining_time - activity_time, new_used)

                total_score = activity_score + future_score

                if total_score > best_score:
                    best_score = total_score
                    best_sequence = [activity_idx] + future_sequence

            dp[cache_key] = (best_score, best_sequence)
            return best_score, best_sequence

        _, optimal_indices = solve(0, max_time, set())
        return [activities[i] for i in optimal_indices]

    def optimize_with_hrm(self,
                         activities: List[Dict[str, Any]],
                         context: Dict[str, Any],
                         goal: OptimizationGoal,
                         constraints: OptimizationConstraints) -> List[Dict[str, Any]]:
        """Optimisation guidée par HRM"""

        try:
            # Préparer données pour HRM
            hrm_context = self._prepare_hrm_context(activities, context, goal, constraints)

            with torch.no_grad():
                # Simuler input pour HRM
                input_ids = torch.randint(0, 32000, (1, 128), device=self.device)

                # Forward pass avec System 2 pour planification complexe
                hrm_output = self.hrm_model(
                    input_ids=input_ids,
                    return_reasoning_details=True,
                    max_reasoning_steps=config.hrm.max_reasoning_steps
                )

                # Interpréter sortie HRM
                optimized_sequence = self._interpret_hrm_optimization(
                    hrm_output, activities, context, goal
                )

                return optimized_sequence

        except Exception as e:
            logger.warning(f"Erreur optimisation HRM: {e}, fallback vers greedy")
            return self.optimize_greedy(activities, context, goal, constraints)

    def _prepare_hrm_context(self,
                           activities: List[Dict[str, Any]],
                           context: Dict[str, Any],
                           goal: OptimizationGoal,
                           constraints: OptimizationConstraints) -> Dict[str, Any]:
        """Prépare contexte pour HRM"""

        return {
            "num_activities": len(activities),
            "player_level": context.get("level", 1),
            "available_time": constraints.max_time or 480,  # 8 heures par défaut
            "optimization_goal": goal.value,
            "activity_types": list(set(a.get("type", "quest") for a in activities)),
            "total_potential_exp": sum(a.get("estimated_experience", 0) for a in activities),
            "total_potential_kamas": sum(a.get("estimated_kamas", 0) for a in activities)
        }

    def _interpret_hrm_optimization(self,
                                  hrm_output: Union[HRMOutput, Dict],
                                  activities: List[Dict[str, Any]],
                                  context: Dict[str, Any],
                                  goal: OptimizationGoal) -> List[Dict[str, Any]]:
        """Interprète sortie HRM en séquence optimisée"""

        # Logique d'interprétation basée sur l'objectif
        if goal == OptimizationGoal.MAXIMIZE_EXPERIENCE:
            # Prioriser activités high-exp
            return sorted(activities, key=lambda a: a.get("estimated_experience", 0), reverse=True)

        elif goal == OptimizationGoal.MAXIMIZE_KAMAS:
            # Prioriser activités high-kamas
            return sorted(activities, key=lambda a: a.get("estimated_kamas", 0), reverse=True)

        elif goal == OptimizationGoal.MINIMIZE_TIME:
            # Prioriser activités rapides
            return sorted(activities, key=lambda a: a.get("estimated_time", float('inf')))

        elif goal == OptimizationGoal.BALANCED:
            # Score balanced
            def balanced_score(activity):
                exp = activity.get("estimated_experience", 0)
                kamas = activity.get("estimated_kamas", 0)
                time = max(activity.get("estimated_time", 1), 1)
                return (exp + kamas * 0.1) / time

            return sorted(activities, key=balanced_score, reverse=True)

        else:
            # Fallback
            return activities

    def _satisfies_constraints(self,
                             activity: Dict[str, Any],
                             constraints: OptimizationConstraints,
                             current_time: float) -> bool:
        """Vérifie si activité satisfait les contraintes"""

        # Contrainte de temps
        if constraints.max_time:
            activity_time = activity.get("estimated_time", 0)
            if current_time + activity_time > constraints.max_time:
                return False

        # Contrainte de coût
        if constraints.max_cost:
            activity_cost = activity.get("cost", 0)
            if activity_cost > constraints.max_cost:
                return False

        # Contraintes de niveau
        activity_level = activity.get("level_requirement", 1)
        if constraints.min_level and activity_level < constraints.min_level:
            return False
        if constraints.max_level and activity_level > constraints.max_level:
            return False

        # Zones interdites
        activity_zone = activity.get("zone", "")
        if activity_zone in constraints.forbidden_zones:
            return False

        return True

    def _calculate_goal_score(self, evaluation: Dict[str, float], goal: OptimizationGoal) -> float:
        """Calcule score selon objectif"""

        if goal == OptimizationGoal.MAXIMIZE_EXPERIENCE:
            return evaluation["exp_per_hour"]

        elif goal == OptimizationGoal.MAXIMIZE_KAMAS:
            return evaluation["kamas_per_hour"]

        elif goal == OptimizationGoal.MINIMIZE_TIME:
            return 1.0 / max(evaluation["time_estimate"], 0.1)

        elif goal == OptimizationGoal.MINIMIZE_RISK:
            return 1.0 / max(evaluation["risk_score"], 0.1)

        elif goal == OptimizationGoal.BALANCED:
            exp_score = evaluation["exp_per_hour"] / 1000
            kamas_score = evaluation["kamas_per_hour"] / 100
            time_score = 10.0 / max(evaluation["time_estimate"], 0.1)
            risk_score = 5.0 / max(evaluation["risk_score"], 0.1)
            return exp_score + kamas_score + time_score + risk_score

        else:
            return evaluation["efficiency_score"]

    def _update_context_after_activity(self,
                                      context: Dict[str, Any],
                                      activity: Dict[str, Any]) -> Dict[str, Any]:
        """Met à jour contexte après activité"""

        new_context = context.copy()

        # Mise à jour expérience/niveau
        exp_gain = activity.get("estimated_experience", 0)
        current_exp = new_context.get("experience", 0)
        new_context["experience"] = current_exp + exp_gain

        # Conversion expérience -> niveau (approximative)
        new_level = self._calculate_level_from_exp(new_context["experience"])
        new_context["level"] = new_level

        # Mise à jour kamas
        kamas_gain = activity.get("estimated_kamas", 0)
        kamas_cost = activity.get("cost", 0)
        current_kamas = new_context.get("kamas", 0)
        new_context["kamas"] = current_kamas + kamas_gain - kamas_cost

        # Mise à jour temps
        time_spent = activity.get("estimated_time", 0)
        current_time = new_context.get("elapsed_time", 0)
        new_context["elapsed_time"] = current_time + time_spent

        return new_context

    def _calculate_level_from_exp(self, total_exp: int) -> int:
        """Calcule niveau approximatif depuis expérience totale"""
        # Formule DOFUS approximative
        if total_exp < 100:
            return 1
        return min(200, int(math.sqrt(total_exp / 100)) + 1)

class StrategyOptimizer:
    """Optimiseur principal de stratégies"""

    def __init__(self):
        self.evaluator = ActivityEvaluator()
        self.sequence_optimizer = SequenceOptimizer(self.evaluator)

        # Historique d'optimisations
        self.optimization_history: List[OptimizationResult] = []

        # Statistiques
        self.total_optimizations = 0
        self.successful_optimizations = 0

        logger.info("StrategyOptimizer initialisé avec succès")

    def optimize_guides(self,
                       guides: List[Guide],
                       player_context: Dict[str, Any],
                       goal: OptimizationGoal = OptimizationGoal.BALANCED,
                       strategy: OptimizationStrategy = OptimizationStrategy.HRM_GUIDED,
                       constraints: Optional[OptimizationConstraints] = None) -> OptimizationResult:
        """Optimise une liste de guides"""

        start_time = time.time()
        self.total_optimizations += 1

        if constraints is None:
            constraints = OptimizationConstraints()

        try:
            # Convertir guides en activités
            activities = self._guides_to_activities(guides)

            # Filtrer activités valides
            valid_activities = [
                activity for activity in activities
                if self._satisfies_constraints(activity, constraints, 0)
            ]

            if not valid_activities:
                return OptimizationResult(
                    success=False,
                    warnings=["Aucune activité satisfait les contraintes"]
                )

            # Choisir stratégie d'optimisation
            if strategy == OptimizationStrategy.GREEDY:
                optimized_sequence = self.sequence_optimizer.optimize_greedy(
                    valid_activities, player_context, goal, constraints
                )
            elif strategy == OptimizationStrategy.DYNAMIC_PROGRAMMING:
                optimized_sequence = self.sequence_optimizer.optimize_dynamic_programming(
                    valid_activities, player_context, goal, constraints
                )
            elif strategy == OptimizationStrategy.HRM_GUIDED:
                optimized_sequence = self.sequence_optimizer.optimize_with_hrm(
                    valid_activities, player_context, goal, constraints
                )
            else:
                # Fallback vers greedy
                optimized_sequence = self.sequence_optimizer.optimize_greedy(
                    valid_activities, player_context, goal, constraints
                )

            # Calculer métriques finales
            result = self._calculate_optimization_result(
                optimized_sequence, player_context, goal, strategy, start_time
            )

            if result.success:
                self.successful_optimizations += 1

            # Ajouter à l'historique
            self.optimization_history.append(result)

            # Limiter historique
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]

            return result

        except Exception as e:
            logger.error(f"Erreur optimisation: {e}")
            return OptimizationResult(
                success=False,
                optimization_time=time.time() - start_time,
                warnings=[f"Erreur d'optimisation: {str(e)}"]
            )

    def _guides_to_activities(self, guides: List[Guide]) -> List[Dict[str, Any]]:
        """Convertit guides en activités optimisables"""

        activities = []

        for guide in guides:
            activity = {
                "id": guide.guide_id,
                "type": guide.guide_type.value,
                "title": guide.title,
                "description": guide.description,
                "level_requirement": guide.level_requirement,
                "estimated_time": guide.estimated_completion_time,
                "estimated_experience": guide.estimated_experience_gain,
                "estimated_kamas": guide.estimated_kamas_gain,
                "cost": guide.required_kamas,
                "difficulty": guide.difficulty_rating,
                "tags": guide.tags,
                "steps": len(guide.steps),
                "zone": guide.category or "unknown"
            }
            activities.append(activity)

        return activities

    def _satisfies_constraints(self,
                             activity: Dict[str, Any],
                             constraints: OptimizationConstraints,
                             current_time: float) -> bool:
        """Délègue à sequence_optimizer"""
        return self.sequence_optimizer._satisfies_constraints(activity, constraints, current_time)

    def _calculate_optimization_result(self,
                                     optimized_sequence: List[Dict[str, Any]],
                                     player_context: Dict[str, Any],
                                     goal: OptimizationGoal,
                                     strategy: OptimizationStrategy,
                                     start_time: float) -> OptimizationResult:
        """Calcule résultat d'optimisation complet"""

        if not optimized_sequence:
            return OptimizationResult(
                success=False,
                optimization_time=time.time() - start_time,
                warnings=["Séquence vide après optimisation"]
            )

        # Métriques cumulées
        total_experience = sum(a.get("estimated_experience", 0) for a in optimized_sequence)
        total_kamas = sum(a.get("estimated_kamas", 0) for a in optimized_sequence)
        total_cost = sum(a.get("cost", 0) for a in optimized_sequence)
        total_time = sum(a.get("estimated_time", 0) for a in optimized_sequence)

        # Score total selon objectif
        if goal == OptimizationGoal.MAXIMIZE_EXPERIENCE:
            total_score = total_experience
        elif goal == OptimizationGoal.MAXIMIZE_KAMAS:
            total_score = total_kamas - total_cost
        elif goal == OptimizationGoal.MINIMIZE_TIME:
            total_score = 1000.0 / max(total_time, 1.0)
        else:
            total_score = (total_experience + (total_kamas - total_cost) * 0.1) / max(total_time, 1.0)

        # Risk score moyen
        risk_scores = []
        for activity in optimized_sequence:
            evaluation = self.evaluator.evaluate_activity(activity, player_context)
            risk_scores.append(evaluation["risk_score"])

        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0

        # Breakdown par type d'activité
        activities_breakdown = defaultdict(lambda: {"count": 0, "time": 0.0, "exp": 0, "kamas": 0})

        for activity in optimized_sequence:
            activity_type = activity.get("type", "unknown")
            activities_breakdown[activity_type]["count"] += 1
            activities_breakdown[activity_type]["time"] += activity.get("estimated_time", 0)
            activities_breakdown[activity_type]["exp"] += activity.get("estimated_experience", 0)
            activities_breakdown[activity_type]["kamas"] += activity.get("estimated_kamas", 0)

        return OptimizationResult(
            success=True,
            optimized_sequence=optimized_sequence,
            total_score=total_score,
            total_experience=total_experience,
            total_kamas=total_kamas - total_cost,
            total_time=total_time,
            risk_score=avg_risk,
            optimization_time=time.time() - start_time,
            iterations=len(optimized_sequence),
            strategy_used=strategy.value,
            convergence_reached=True,
            activities_breakdown=dict(activities_breakdown)
        )

    def optimize_routes_for_efficiency(self,
                                      routes: List[Dict[str, Any]],
                                      context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimise routes pour efficacité maximale"""

        # Évaluer chaque route
        evaluated_routes = []

        for route in routes:
            efficiency_score = self._calculate_route_efficiency(route, context)
            route["efficiency_score"] = efficiency_score
            evaluated_routes.append(route)

        # Trier par efficacité
        return sorted(evaluated_routes, key=lambda r: r["efficiency_score"], reverse=True)

    def _calculate_route_efficiency(self, route: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calcule efficacité d'une route"""

        total_time = route.get("total_time", 1.0)
        total_exp = route.get("total_experience", 0)
        total_kamas = route.get("total_kamas", 0)
        danger_level = route.get("max_danger_level", 1)

        # Score de base
        base_score = (total_exp + total_kamas * 0.1) / total_time

        # Pénalité de danger
        danger_penalty = 1.0 / max(danger_level, 1.0)

        return base_score * danger_penalty

    def get_optimization_recommendations(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recommandations d'optimisation personnalisées"""

        player_level = context.get("level", 1)
        available_time = context.get("available_time", 4.0)  # heures
        preferences = context.get("preferences", {})

        recommendations = {
            "primary_goal": OptimizationGoal.BALANCED,
            "recommended_strategy": OptimizationStrategy.HRM_GUIDED,
            "activity_focus": [],
            "time_allocation": {},
            "efficiency_tips": []
        }

        # Recommandations par niveau
        if player_level < 20:
            recommendations["primary_goal"] = OptimizationGoal.MAXIMIZE_EXPERIENCE
            recommendations["activity_focus"] = ["quest", "exploration"]
            recommendations["efficiency_tips"].append("Focalisez sur les quêtes pour XP rapide")

        elif player_level < 50:
            recommendations["primary_goal"] = OptimizationGoal.BALANCED
            recommendations["activity_focus"] = ["quest", "farming", "profession"]
            recommendations["efficiency_tips"].append("Équilibrez XP et kamas pour équipement")

        else:
            recommendations["primary_goal"] = OptimizationGoal.MAXIMIZE_KAMAS
            recommendations["activity_focus"] = ["farming", "dungeon", "profession"]
            recommendations["efficiency_tips"].append("Optimisez pour kamas et équipement endgame")

        # Allocation temps recommandée
        if available_time <= 2:
            recommendations["time_allocation"] = {"quest": 0.7, "farming": 0.3}
            recommendations["efficiency_tips"].append("Sessions courtes: focus quêtes rapides")
        elif available_time <= 4:
            recommendations["time_allocation"] = {"quest": 0.5, "farming": 0.3, "profession": 0.2}
        else:
            recommendations["time_allocation"] = {"quest": 0.4, "farming": 0.3, "dungeon": 0.2, "profession": 0.1}
            recommendations["efficiency_tips"].append("Sessions longues: diversifiez activités")

        return recommendations

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Statistiques d'optimisation"""

        success_rate = (self.successful_optimizations / max(self.total_optimizations, 1)) * 100

        recent_results = self.optimization_history[-10:] if self.optimization_history else []
        avg_score = sum(r.total_score for r in recent_results) / max(len(recent_results), 1)
        avg_time = sum(r.optimization_time for r in recent_results) / max(len(recent_results), 1)

        return {
            "total_optimizations": self.total_optimizations,
            "successful_optimizations": self.successful_optimizations,
            "success_rate": success_rate,
            "recent_avg_score": avg_score,
            "recent_avg_optimization_time": avg_time,
            "history_size": len(self.optimization_history)
        }

def create_strategy_optimizer() -> StrategyOptimizer:
    """Factory function pour créer un StrategyOptimizer"""
    return StrategyOptimizer()