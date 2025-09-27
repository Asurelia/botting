"""
Multi-Objective Decision Engine
Moteur de décision avancé pour optimisation multi-objectifs avec résolution de conflits
Implémentation de l'optimisation Pareto et planification temporelle stratégique
"""

import numpy as np
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import heapq
import itertools
from collections import defaultdict, deque
import json
import math

# Import modules internes
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.knowledge_graph import DofusKnowledgeGraph, EntityType, RelationType
from core.predictive_engine import PredictiveAnalyticsEngine, TimeWindow, PredictionType
from core.uncertainty import UncertaintyManager

logger = logging.getLogger(__name__)

class ObjectiveType(Enum):
    """Types d'objectifs possibles"""
    # Objectifs économiques
    MAXIMIZE_PROFIT = "maximize_profit"
    MINIMIZE_COST = "minimize_cost"
    OPTIMIZE_EFFICIENCY = "optimize_efficiency"

    # Objectifs de progression
    GAIN_EXPERIENCE = "gain_experience"
    IMPROVE_SKILLS = "improve_skills"
    COMPLETE_QUESTS = "complete_quests"

    # Objectifs sociaux
    BUILD_REPUTATION = "build_reputation"
    TEAM_COORDINATION = "team_coordination"
    GUILD_CONTRIBUTION = "guild_contribution"

    # Objectifs de sécurité
    MINIMIZE_RISK = "minimize_risk"
    AVOID_DETECTION = "avoid_detection"
    MAINTAIN_STABILITY = "maintain_stability"

    # Objectifs temporels
    COMPLETE_BEFORE_DEADLINE = "complete_before_deadline"
    OPTIMIZE_TIMING = "optimize_timing"
    PLAN_LONG_TERM = "plan_long_term"

class Priority(Enum):
    """Niveaux de priorité des objectifs"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    OPTIONAL = 5

class ConflictType(Enum):
    """Types de conflits entre objectifs"""
    RESOURCE_COMPETITION = "resource_competition"
    TIME_CONSTRAINT = "time_constraint"
    RISK_TOLERANCE = "risk_tolerance"
    STRATEGIC_ALIGNMENT = "strategic_alignment"
    CAPABILITY_LIMITATION = "capability_limitation"

@dataclass
class Objective:
    """Objectif individuel avec métriques et contraintes"""
    id: str
    name: str
    type: ObjectiveType
    priority: Priority

    # Métriques cibles
    target_value: float
    current_value: float = 0.0
    weight: float = 1.0

    # Contraintes temporelles
    deadline: Optional[datetime] = None
    estimated_duration: Optional[timedelta] = None

    # Dépendances
    dependencies: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)

    # Métadonnées
    created_at: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

    def progress_ratio(self) -> float:
        """Calcule le ratio de progression vers l'objectif"""
        if self.target_value == 0:
            return 1.0 if self.current_value >= 0 else 0.0
        return min(1.0, max(0.0, self.current_value / self.target_value))

    def is_completed(self) -> bool:
        """Vérifie si l'objectif est atteint"""
        return self.progress_ratio() >= 1.0

    def is_overdue(self) -> bool:
        """Vérifie si l'objectif est en retard"""
        if not self.deadline:
            return False
        return datetime.now() > self.deadline

@dataclass
class Action:
    """Action possible avec impacts et coûts"""
    id: str
    name: str
    type: str

    # Impacts sur les objectifs
    objective_impacts: Dict[str, float] = field(default_factory=dict)

    # Coûts et ressources
    resource_cost: Dict[str, float] = field(default_factory=dict)
    time_cost: timedelta = field(default_factory=lambda: timedelta(minutes=1))

    # Prérequis et contraintes
    prerequisites: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)

    # Métriques de qualité
    success_probability: float = 1.0
    risk_level: float = 0.0

    # Métadonnées
    context: Dict[str, Any] = field(default_factory=dict)

    def calculate_utility(self, objectives: Dict[str, Objective]) -> float:
        """Calcule l'utilité de l'action par rapport aux objectifs"""
        total_utility = 0.0

        for obj_id, impact in self.objective_impacts.items():
            if obj_id in objectives:
                obj = objectives[obj_id]
                # Utilité pondérée par priorité et poids
                priority_multiplier = (6 - obj.priority.value) / 5.0  # Inverse de la priorité
                weighted_impact = impact * obj.weight * priority_multiplier
                total_utility += weighted_impact

        # Ajustement par probabilité de succès et risque
        adjusted_utility = total_utility * self.success_probability * (1.0 - self.risk_level * 0.5)

        return adjusted_utility

@dataclass
class Conflict:
    """Conflit entre objectifs ou actions"""
    id: str
    type: ConflictType
    conflicting_objectives: List[str]
    conflicting_actions: List[str] = field(default_factory=list)

    # Description du conflit
    description: str = ""
    severity: float = 0.5  # 0.0 = mineur, 1.0 = critique

    # Solutions potentielles
    resolution_strategies: List[str] = field(default_factory=list)

    # Métadonnées
    detected_at: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Resolution:
    """Résolution d'un conflit"""
    conflict_id: str
    strategy: str

    # Modifications proposées
    objective_adjustments: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    action_modifications: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Métriques de la résolution
    resolution_score: float = 0.0
    trade_offs: Dict[str, float] = field(default_factory=dict)

    # Métadonnées
    resolved_at: datetime = field(default_factory=datetime.now)
    reasoning: List[str] = field(default_factory=list)

@dataclass
class ActionPlan:
    """Plan d'actions optimisé"""
    id: str
    name: str

    # Actions et séquencement
    actions: List[Action] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)

    # Métriques du plan
    total_utility: float = 0.0
    estimated_duration: timedelta = field(default_factory=timedelta)
    success_probability: float = 1.0

    # Objectifs adressés
    objective_coverage: Dict[str, float] = field(default_factory=dict)

    # Planification temporelle
    start_time: Optional[datetime] = None
    milestones: List[Tuple[datetime, str]] = field(default_factory=list)

    # Métadonnées
    created_at: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TemporalPlan:
    """Plan stratégique avec horizon temporel"""
    id: str
    name: str
    horizon: timedelta

    # Phases du plan
    short_term_goals: List[str] = field(default_factory=list)  # 0-1h
    medium_term_goals: List[str] = field(default_factory=list)  # 1h-1j
    long_term_goals: List[str] = field(default_factory=list)   # 1j+

    # Actions par période
    immediate_actions: List[Action] = field(default_factory=list)
    scheduled_actions: List[Tuple[datetime, Action]] = field(default_factory=list)

    # Adaptation et révision
    revision_intervals: List[timedelta] = field(default_factory=list)
    adaptation_triggers: List[str] = field(default_factory=list)

    # Métriques
    overall_progress: float = 0.0
    phase_progress: Dict[str, float] = field(default_factory=dict)

    # Métadonnées
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

class ParetoOptimizer:
    """Optimiseur Pareto pour solutions multi-objectifs"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ParetoOptimizer")

    def find_pareto_frontier(self, solutions: List[Dict[str, float]]) -> List[int]:
        """Trouve la frontière de Pareto parmi les solutions"""
        if not solutions:
            return []

        pareto_indices = []

        for i, solution_i in enumerate(solutions):
            is_dominated = False

            for j, solution_j in enumerate(solutions):
                if i == j:
                    continue

                # Vérifie si solution_j domine solution_i
                if self._dominates(solution_j, solution_i):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_indices.append(i)

        return pareto_indices

    def _dominates(self, solution_a: Dict[str, float], solution_b: Dict[str, float]) -> bool:
        """Vérifie si solution_a domine solution_b"""
        # solution_a domine solution_b si elle est au moins aussi bonne sur tous les objectifs
        # et strictement meilleure sur au moins un objectif

        at_least_as_good = True
        strictly_better_on_one = False

        common_objectives = set(solution_a.keys()) & set(solution_b.keys())

        for obj in common_objectives:
            if solution_a[obj] < solution_b[obj]:
                at_least_as_good = False
                break
            elif solution_a[obj] > solution_b[obj]:
                strictly_better_on_one = True

        return at_least_as_good and strictly_better_on_one

    def select_best_compromise(self, pareto_solutions: List[Dict[str, float]],
                             weights: Dict[str, float]) -> int:
        """Sélectionne la meilleure solution de compromis selon les poids"""
        if not pareto_solutions:
            return -1

        best_index = 0
        best_score = -float('inf')

        for i, solution in enumerate(pareto_solutions):
            # Score pondéré
            score = sum(solution.get(obj, 0) * weight
                       for obj, weight in weights.items())

            if score > best_score:
                best_score = score
                best_index = i

        return best_index

class ConflictResolver:
    """Résolveur de conflits entre objectifs"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ConflictResolver")

        # Stratégies de résolution par type de conflit
        self.resolution_strategies = {
            ConflictType.RESOURCE_COMPETITION: [
                "prioritize_high_value",
                "resource_sharing",
                "temporal_separation",
                "alternative_resources"
            ],
            ConflictType.TIME_CONSTRAINT: [
                "priority_scheduling",
                "parallel_execution",
                "deadline_negotiation",
                "scope_reduction"
            ],
            ConflictType.RISK_TOLERANCE: [
                "risk_mitigation",
                "conservative_approach",
                "risk_sharing",
                "contingency_planning"
            ],
            ConflictType.STRATEGIC_ALIGNMENT: [
                "objective_reformulation",
                "goal_hierarchy",
                "compromise_solution",
                "strategic_pivot"
            ],
            ConflictType.CAPABILITY_LIMITATION: [
                "capability_development",
                "external_assistance",
                "scope_adjustment",
                "technology_upgrade"
            ]
        }

    def detect_conflicts(self, objectives: Dict[str, Objective],
                        actions: List[Action]) -> List[Conflict]:
        """Détecte les conflits entre objectifs et actions"""
        conflicts = []

        # Conflits de ressources
        resource_conflicts = self._detect_resource_conflicts(objectives, actions)
        conflicts.extend(resource_conflicts)

        # Conflits temporels
        time_conflicts = self._detect_time_conflicts(objectives, actions)
        conflicts.extend(time_conflicts)

        # Conflits de priorités
        priority_conflicts = self._detect_priority_conflicts(objectives)
        conflicts.extend(priority_conflicts)

        return conflicts

    def resolve_conflict(self, conflict: Conflict, objectives: Dict[str, Objective],
                        actions: List[Action]) -> Resolution:
        """Résout un conflit spécifique"""
        try:
            # Sélection de la stratégie de résolution
            strategies = self.resolution_strategies.get(conflict.type, ["default_resolution"])
            best_strategy = self._select_best_strategy(conflict, strategies, objectives, actions)

            # Application de la stratégie
            resolution = self._apply_resolution_strategy(
                conflict, best_strategy, objectives, actions
            )

            return resolution

        except Exception as e:
            self.logger.error(f"Erreur résolution conflit {conflict.id}: {e}")
            return self._create_fallback_resolution(conflict)

    def _detect_resource_conflicts(self, objectives: Dict[str, Objective],
                                 actions: List[Action]) -> List[Conflict]:
        """Détecte les conflits de ressources"""
        conflicts = []

        # Groupement des actions par ressource utilisée
        resource_usage = defaultdict(list)

        for action in actions:
            for resource, amount in action.resource_cost.items():
                if amount > 0:
                    resource_usage[resource].append(action.id)

        # Détection des conflits quand plusieurs actions utilisent la même ressource
        for resource, action_ids in resource_usage.items():
            if len(action_ids) > 1:
                # Identifier les objectifs affectés
                affected_objectives = []
                for action in actions:
                    if action.id in action_ids:
                        affected_objectives.extend(action.objective_impacts.keys())

                affected_objectives = list(set(affected_objectives))

                if len(affected_objectives) > 1:
                    conflict = Conflict(
                        id=f"resource_conflict_{resource}_{int(datetime.now().timestamp())}",
                        type=ConflictType.RESOURCE_COMPETITION,
                        conflicting_objectives=affected_objectives,
                        conflicting_actions=action_ids,
                        description=f"Competition pour la ressource {resource}",
                        severity=min(1.0, len(action_ids) * 0.2)
                    )
                    conflicts.append(conflict)

        return conflicts

    def _detect_time_conflicts(self, objectives: Dict[str, Objective],
                             actions: List[Action]) -> List[Conflict]:
        """Détecte les conflits temporels"""
        conflicts = []

        # Vérification des deadlines impossibles
        now = datetime.now()

        for obj_id, objective in objectives.items():
            if objective.deadline and objective.estimated_duration:
                required_completion = now + objective.estimated_duration

                if required_completion > objective.deadline:
                    conflict = Conflict(
                        id=f"time_conflict_{obj_id}_{int(datetime.now().timestamp())}",
                        type=ConflictType.TIME_CONSTRAINT,
                        conflicting_objectives=[obj_id],
                        description=f"Deadline impossible pour {objective.name}",
                        severity=min(1.0, (required_completion - objective.deadline).total_seconds() / 3600 * 0.1)
                    )
                    conflicts.append(conflict)

        return conflicts

    def _detect_priority_conflicts(self, objectives: Dict[str, Objective]) -> List[Conflict]:
        """Détecte les conflits de priorités"""
        conflicts = []

        # Groupement par priorité
        by_priority = defaultdict(list)
        for obj_id, objective in objectives.items():
            by_priority[objective.priority].append(obj_id)

        # Détection de trop d'objectifs critiques simultanés
        critical_objectives = by_priority[Priority.CRITICAL]
        if len(critical_objectives) > 3:  # Seuil arbitraire
            conflict = Conflict(
                id=f"priority_conflict_{int(datetime.now().timestamp())}",
                type=ConflictType.STRATEGIC_ALIGNMENT,
                conflicting_objectives=critical_objectives,
                description=f"Trop d'objectifs critiques simultanés ({len(critical_objectives)})",
                severity=min(1.0, len(critical_objectives) * 0.15)
            )
            conflicts.append(conflict)

        return conflicts

    def _select_best_strategy(self, conflict: Conflict, strategies: List[str],
                            objectives: Dict[str, Objective], actions: List[Action]) -> str:
        """Sélectionne la meilleure stratégie de résolution"""
        # Pour l'instant, sélection simple basée sur le type de conflit
        if conflict.severity > 0.7:
            # Conflits sévères -> stratégies conservatrices
            conservative_strategies = ["prioritize_high_value", "conservative_approach", "scope_reduction"]
            for strategy in conservative_strategies:
                if strategy in strategies:
                    return strategy

        # Sélection par défaut : première stratégie disponible
        return strategies[0] if strategies else "default_resolution"

    def _apply_resolution_strategy(self, conflict: Conflict, strategy: str,
                                 objectives: Dict[str, Objective], actions: List[Action]) -> Resolution:
        """Applique une stratégie de résolution"""
        resolution = Resolution(
            conflict_id=conflict.id,
            strategy=strategy
        )

        if strategy == "prioritize_high_value":
            # Réorganise par priorité et valeur
            resolution.reasoning = ["Priorisation des objectifs à haute valeur"]
            resolution.resolution_score = 0.8

        elif strategy == "temporal_separation":
            # Sépare les actions dans le temps
            resolution.reasoning = ["Séparation temporelle des actions conflictuelles"]
            resolution.resolution_score = 0.7

        elif strategy == "resource_sharing":
            # Partage des ressources
            resolution.reasoning = ["Partage optimisé des ressources"]
            resolution.resolution_score = 0.6

        elif strategy == "scope_reduction":
            # Réduction de la portée
            resolution.reasoning = ["Réduction de la portée pour éviter les conflits"]
            resolution.resolution_score = 0.5

        else:
            # Stratégie par défaut
            resolution.reasoning = ["Résolution par défaut appliquée"]
            resolution.resolution_score = 0.4

        return resolution

    def _create_fallback_resolution(self, conflict: Conflict) -> Resolution:
        """Crée une résolution de fallback"""
        return Resolution(
            conflict_id=conflict.id,
            strategy="fallback",
            resolution_score=0.2,
            reasoning=["Résolution de fallback - échec de la résolution automatique"]
        )

class AdvancedDecisionEngine:
    """
    Moteur de décision multi-objectifs avancé
    Implémente l'optimisation Pareto et la résolution de conflits
    """

    def __init__(self, knowledge_graph: DofusKnowledgeGraph,
                 prediction_engine: PredictiveAnalyticsEngine,
                 uncertainty_manager: UncertaintyManager):
        self.knowledge_graph = knowledge_graph
        self.prediction_engine = prediction_engine
        self.uncertainty_manager = uncertainty_manager

        self.logger = logging.getLogger(__name__)

        # Composants spécialisés
        self.pareto_optimizer = ParetoOptimizer()
        self.conflict_resolver = ConflictResolver()

        # État du moteur
        self.active_objectives: Dict[str, Objective] = {}
        self.available_actions: List[Action] = []
        self.current_conflicts: List[Conflict] = []

        # Historique et apprentissage
        self.decision_history: deque = deque(maxlen=1000)
        self.performance_metrics = {
            'decisions_made': 0,
            'objectives_completed': 0,
            'conflicts_resolved': 0,
            'average_utility': 0.0
        }

    async def evaluate_action_portfolio(self, actions: List[Action]) -> ActionPlan:
        """Évalue un portefeuille d'actions avec optimisation Pareto"""
        try:
            self.logger.info(f"Évaluation portefeuille de {len(actions)} actions...")

            # Génération de solutions candidates
            candidate_solutions = await self._generate_candidate_solutions(actions)

            # Optimisation Pareto
            pareto_indices = self.pareto_optimizer.find_pareto_frontier(candidate_solutions)
            pareto_solutions = [candidate_solutions[i] for i in pareto_indices]

            # Sélection de la meilleure solution de compromis
            weights = self._calculate_objective_weights()
            best_index = self.pareto_optimizer.select_best_compromise(pareto_solutions, weights)

            if best_index >= 0 and best_index < len(pareto_solutions):
                # Construction du plan d'action optimal
                optimal_plan = await self._build_action_plan(
                    actions, pareto_solutions[best_index], pareto_indices[best_index]
                )

                self.logger.info(f"Plan optimal généré avec utilité {optimal_plan.total_utility:.2f}")
                return optimal_plan
            else:
                return self._create_fallback_plan(actions)

        except Exception as e:
            self.logger.error(f"Erreur évaluation portefeuille: {e}")
            return self._create_fallback_plan(actions)

    async def resolve_objective_conflicts(self, conflicts: List[Conflict]) -> List[Resolution]:
        """Résout intelligemment les conflits d'objectifs"""
        try:
            self.logger.info(f"Résolution de {len(conflicts)} conflits...")

            resolutions = []

            # Tri des conflits par sévérité (plus sévères en premier)
            sorted_conflicts = sorted(conflicts, key=lambda c: c.severity, reverse=True)

            for conflict in sorted_conflicts:
                resolution = self.conflict_resolver.resolve_conflict(
                    conflict, self.active_objectives, self.available_actions
                )
                resolutions.append(resolution)

                # Application de la résolution
                await self._apply_resolution(resolution)

            # Mise à jour des métriques
            self.performance_metrics['conflicts_resolved'] += len(resolutions)

            self.logger.info(f"{len(resolutions)} conflits résolus")
            return resolutions

        except Exception as e:
            self.logger.error(f"Erreur résolution conflits: {e}")
            return []

    async def plan_temporal_strategy(self, horizon: timedelta) -> TemporalPlan:
        """Planifie une stratégie sur un horizon temporel"""
        try:
            self.logger.info(f"Planification stratégique sur {horizon}...")

            # Catégorisation des objectifs par échelle temporelle
            short_term = []
            medium_term = []
            long_term = []

            now = datetime.now()

            for obj_id, objective in self.active_objectives.items():
                if objective.estimated_duration:
                    if objective.estimated_duration <= timedelta(hours=1):
                        short_term.append(obj_id)
                    elif objective.estimated_duration <= timedelta(days=1):
                        medium_term.append(obj_id)
                    else:
                        long_term.append(obj_id)
                else:
                    # Par défaut, objectifs à court terme
                    short_term.append(obj_id)

            # Planification par phases
            immediate_actions = await self._plan_immediate_actions(short_term)
            scheduled_actions = await self._plan_scheduled_actions(medium_term, long_term, horizon)

            # Création du plan temporel
            temporal_plan = TemporalPlan(
                id=f"temporal_plan_{int(datetime.now().timestamp())}",
                name=f"Plan Stratégique {horizon}",
                horizon=horizon,
                short_term_goals=short_term,
                medium_term_goals=medium_term,
                long_term_goals=long_term,
                immediate_actions=immediate_actions,
                scheduled_actions=scheduled_actions,
                revision_intervals=[
                    timedelta(minutes=15),  # Révision fréquente
                    timedelta(hours=1),     # Révision tactique
                    timedelta(hours=6)      # Révision stratégique
                ],
                adaptation_triggers=[
                    "objective_completion",
                    "unexpected_event",
                    "performance_deviation",
                    "resource_shortage"
                ]
            )

            # Calcul des métriques de progression
            temporal_plan.overall_progress = self._calculate_overall_progress()
            temporal_plan.phase_progress = {
                "short_term": self._calculate_phase_progress(short_term),
                "medium_term": self._calculate_phase_progress(medium_term),
                "long_term": self._calculate_phase_progress(long_term)
            }

            self.logger.info(f"Plan temporel créé avec {len(immediate_actions)} actions immédiates")
            return temporal_plan

        except Exception as e:
            self.logger.error(f"Erreur planification temporelle: {e}")
            return self._create_fallback_temporal_plan(horizon)

    async def add_objective(self, objective: Objective) -> bool:
        """Ajoute un nouvel objectif"""
        try:
            self.active_objectives[objective.id] = objective

            # Détection de conflits avec les objectifs existants
            conflicts = self.conflict_resolver.detect_conflicts(
                self.active_objectives, self.available_actions
            )

            if conflicts:
                self.current_conflicts.extend(conflicts)
                self.logger.warning(f"Ajout objectif {objective.name} génère {len(conflicts)} conflits")

            return True

        except Exception as e:
            self.logger.error(f"Erreur ajout objectif {objective.id}: {e}")
            return False

    async def update_objective_progress(self, objective_id: str, new_value: float) -> bool:
        """Met à jour la progression d'un objectif"""
        try:
            if objective_id in self.active_objectives:
                objective = self.active_objectives[objective_id]
                old_value = objective.current_value
                objective.current_value = new_value

                # Vérification de la completion
                if objective.is_completed() and old_value < objective.target_value:
                    self.performance_metrics['objectives_completed'] += 1
                    self.logger.info(f"Objectif {objective.name} complété !")

                return True
            else:
                self.logger.warning(f"Objectif {objective_id} non trouvé")
                return False

        except Exception as e:
            self.logger.error(f"Erreur mise à jour objectif {objective_id}: {e}")
            return False

    async def _generate_candidate_solutions(self, actions: List[Action]) -> List[Dict[str, float]]:
        """Génère des solutions candidates pour l'optimisation"""
        candidates = []

        # Génération de combinaisons d'actions
        max_combinations = min(100, 2 ** len(actions))  # Limitation pour performance

        for r in range(1, min(len(actions) + 1, 6)):  # Maximum 5 actions simultanées
            for action_combo in itertools.combinations(actions, r):
                if len(candidates) >= max_combinations:
                    break

                # Calcul des métriques pour cette combinaison
                solution_metrics = {}

                for obj_id in self.active_objectives.keys():
                    total_impact = sum(action.objective_impacts.get(obj_id, 0)
                                     for action in action_combo)
                    solution_metrics[obj_id] = total_impact

                candidates.append(solution_metrics)

        return candidates

    def _calculate_objective_weights(self) -> Dict[str, float]:
        """Calcule les poids des objectifs basés sur les priorités"""
        weights = {}

        for obj_id, objective in self.active_objectives.items():
            # Poids basé sur la priorité (inverse)
            priority_weight = (6 - objective.priority.value) / 5.0

            # Ajustement par urgence (deadline proche)
            urgency_weight = 1.0
            if objective.deadline:
                time_to_deadline = (objective.deadline - datetime.now()).total_seconds()
                if time_to_deadline > 0:
                    # Plus c'est urgent, plus le poids augmente
                    urgency_weight = 1.0 + (1.0 / max(1.0, time_to_deadline / 3600))

            weights[obj_id] = priority_weight * urgency_weight * objective.weight

        # Normalisation
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    async def _build_action_plan(self, actions: List[Action],
                               solution_metrics: Dict[str, float],
                               solution_index: int) -> ActionPlan:
        """Construit un plan d'action à partir d'une solution optimale"""
        # Pour simplifier, on prend les actions avec le meilleur ratio utilité/coût
        selected_actions = []

        for action in actions:
            utility = action.calculate_utility(self.active_objectives)
            if utility > 0.1:  # Seuil minimum d'utilité
                selected_actions.append(action)

        # Tri par utilité décroissante
        selected_actions.sort(key=lambda a: a.calculate_utility(self.active_objectives), reverse=True)

        # Limitation du nombre d'actions
        selected_actions = selected_actions[:5]

        # Calcul des métriques du plan
        total_utility = sum(action.calculate_utility(self.active_objectives)
                          for action in selected_actions)

        estimated_duration = sum((action.time_cost for action in selected_actions), timedelta())

        success_probability = 1.0
        for action in selected_actions:
            success_probability *= action.success_probability

        # Couverture des objectifs
        objective_coverage = {}
        for obj_id in self.active_objectives.keys():
            coverage = sum(action.objective_impacts.get(obj_id, 0)
                         for action in selected_actions)
            objective_coverage[obj_id] = min(1.0, max(0.0, coverage))

        plan = ActionPlan(
            id=f"action_plan_{int(datetime.now().timestamp())}",
            name=f"Plan Optimisé {len(selected_actions)} actions",
            actions=selected_actions,
            execution_order=[action.id for action in selected_actions],
            total_utility=total_utility,
            estimated_duration=estimated_duration,
            success_probability=success_probability,
            objective_coverage=objective_coverage
        )

        return plan

    async def _apply_resolution(self, resolution: Resolution) -> bool:
        """Applique une résolution de conflit"""
        try:
            # Application des ajustements d'objectifs
            for obj_id, adjustments in resolution.objective_adjustments.items():
                if obj_id in self.active_objectives:
                    objective = self.active_objectives[obj_id]

                    # Application des modifications
                    for attr, value in adjustments.items():
                        if hasattr(objective, attr):
                            setattr(objective, attr, value)

            # Application des modifications d'actions
            for action_id, modifications in resolution.action_modifications.items():
                for action in self.available_actions:
                    if action.id == action_id:
                        for attr, value in modifications.items():
                            if hasattr(action, attr):
                                setattr(action, attr, value)

            return True

        except Exception as e:
            self.logger.error(f"Erreur application résolution: {e}")
            return False

    async def _plan_immediate_actions(self, short_term_objectives: List[str]) -> List[Action]:
        """Planifie les actions immédiates"""
        immediate_actions = []

        for obj_id in short_term_objectives:
            # Recherche d'actions pour cet objectif
            relevant_actions = [
                action for action in self.available_actions
                if obj_id in action.objective_impacts and action.objective_impacts[obj_id] > 0
            ]

            # Sélection de la meilleure action
            if relevant_actions:
                best_action = max(relevant_actions,
                                key=lambda a: a.objective_impacts[obj_id] * a.success_probability)
                immediate_actions.append(best_action)

        return immediate_actions

    async def _plan_scheduled_actions(self, medium_term: List[str], long_term: List[str],
                                    horizon: timedelta) -> List[Tuple[datetime, Action]]:
        """Planifie les actions programmées"""
        scheduled_actions = []
        now = datetime.now()

        # Planification des objectifs à moyen terme
        for i, obj_id in enumerate(medium_term):
            schedule_time = now + timedelta(hours=1 + i)
            if schedule_time <= now + horizon:
                relevant_actions = [
                    action for action in self.available_actions
                    if obj_id in action.objective_impacts
                ]

                if relevant_actions:
                    best_action = max(relevant_actions,
                                    key=lambda a: a.objective_impacts[obj_id])
                    scheduled_actions.append((schedule_time, best_action))

        # Planification des objectifs à long terme
        for i, obj_id in enumerate(long_term):
            schedule_time = now + timedelta(days=1 + i)
            if schedule_time <= now + horizon:
                relevant_actions = [
                    action for action in self.available_actions
                    if obj_id in action.objective_impacts
                ]

                if relevant_actions:
                    best_action = max(relevant_actions,
                                    key=lambda a: a.objective_impacts[obj_id])
                    scheduled_actions.append((schedule_time, best_action))

        return scheduled_actions

    def _calculate_overall_progress(self) -> float:
        """Calcule la progression globale"""
        if not self.active_objectives:
            return 1.0

        total_progress = sum(obj.progress_ratio() for obj in self.active_objectives.values())
        return total_progress / len(self.active_objectives)

    def _calculate_phase_progress(self, objective_ids: List[str]) -> float:
        """Calcule la progression d'une phase"""
        if not objective_ids:
            return 1.0

        total_progress = 0.0
        for obj_id in objective_ids:
            if obj_id in self.active_objectives:
                total_progress += self.active_objectives[obj_id].progress_ratio()

        return total_progress / len(objective_ids)

    def _create_fallback_plan(self, actions: List[Action]) -> ActionPlan:
        """Crée un plan de fallback"""
        return ActionPlan(
            id=f"fallback_plan_{int(datetime.now().timestamp())}",
            name="Plan de Fallback",
            actions=actions[:3],  # Premières 3 actions
            execution_order=[action.id for action in actions[:3]],
            total_utility=0.5,
            estimated_duration=timedelta(minutes=30),
            success_probability=0.7
        )

    def _create_fallback_temporal_plan(self, horizon: timedelta) -> TemporalPlan:
        """Crée un plan temporel de fallback"""
        return TemporalPlan(
            id=f"fallback_temporal_{int(datetime.now().timestamp())}",
            name="Plan Temporel de Fallback",
            horizon=horizon,
            short_term_goals=list(self.active_objectives.keys())[:2],
            medium_term_goals=list(self.active_objectives.keys())[2:4],
            long_term_goals=list(self.active_objectives.keys())[4:],
            overall_progress=0.0
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de performance"""
        return {
            **self.performance_metrics,
            'active_objectives': len(self.active_objectives),
            'current_conflicts': len(self.current_conflicts),
            'overall_progress': self._calculate_overall_progress()
        }

# Interface utilitaire
async def create_decision_engine(knowledge_graph: DofusKnowledgeGraph,
                               prediction_engine: PredictiveAnalyticsEngine,
                               uncertainty_manager: UncertaintyManager) -> AdvancedDecisionEngine:
    """Crée et initialise le moteur de décision avancé"""
    engine = AdvancedDecisionEngine(knowledge_graph, prediction_engine, uncertainty_manager)

    # Ajout d'objectifs et d'actions d'exemple pour test
    sample_objective = Objective(
        id="farming_profit",
        name="Maximiser profits de farm",
        type=ObjectiveType.MAXIMIZE_PROFIT,
        priority=Priority.HIGH,
        target_value=10000.0,
        weight=1.5,
        estimated_duration=timedelta(hours=2)
    )

    await engine.add_objective(sample_objective)

    # Actions d'exemple
    engine.available_actions = [
        Action(
            id="farm_wheat",
            name="Farm du blé",
            type="farming",
            objective_impacts={"farming_profit": 50.0},
            resource_cost={"energy": 10},
            time_cost=timedelta(minutes=5),
            success_probability=0.9,
            risk_level=0.1
        ),
        Action(
            id="sell_resources",
            name="Vendre ressources",
            type="trading",
            objective_impacts={"farming_profit": 100.0},
            resource_cost={"time": 5},
            time_cost=timedelta(minutes=2),
            success_probability=0.95,
            risk_level=0.05
        )
    ]

    return engine

# Interface CLI pour tests
async def main():
    """Test du Decision Engine"""
    print("Test Advanced Decision Engine...")

    # Import des dépendances
    from core.knowledge_graph import create_dofus_knowledge_graph
    from core.predictive_engine import create_predictive_engine
    from core.uncertainty import UncertaintyManager

    # Création des composants
    knowledge_graph = await create_dofus_knowledge_graph()
    prediction_engine = await create_predictive_engine(knowledge_graph)
    uncertainty_manager = UncertaintyManager()

    # Création du decision engine
    decision_engine = await create_decision_engine(
        knowledge_graph, prediction_engine, uncertainty_manager
    )

    print(f"Decision Engine initialisé avec {len(decision_engine.active_objectives)} objectifs")

    # Test évaluation d'actions
    action_plan = await decision_engine.evaluate_action_portfolio(
        decision_engine.available_actions
    )

    print(f"Plan d'action généré:")
    print(f"  - {len(action_plan.actions)} actions")
    print(f"  - Utilité totale: {action_plan.total_utility:.2f}")
    print(f"  - Durée estimée: {action_plan.estimated_duration}")
    print(f"  - Probabilité succès: {action_plan.success_probability:.2f}")

    # Test planification temporelle
    temporal_plan = await decision_engine.plan_temporal_strategy(timedelta(days=1))

    print(f"Plan temporel créé:")
    print(f"  - Objectifs court terme: {len(temporal_plan.short_term_goals)}")
    print(f"  - Objectifs moyen terme: {len(temporal_plan.medium_term_goals)}")
    print(f"  - Actions immédiates: {len(temporal_plan.immediate_actions)}")
    print(f"  - Progression globale: {temporal_plan.overall_progress:.2f}")

    # Test résolution de conflits
    conflicts = decision_engine.conflict_resolver.detect_conflicts(
        decision_engine.active_objectives, decision_engine.available_actions
    )

    if conflicts:
        print(f"Conflits détectés: {len(conflicts)}")
        resolutions = await decision_engine.resolve_objective_conflicts(conflicts)
        print(f"Résolutions appliquées: {len(resolutions)}")

    # Métriques finales
    metrics = decision_engine.get_performance_metrics()
    print(f"Métriques de performance:")
    for key, value in metrics.items():
        print(f"  - {key}: {value}")

    print("Test Decision Engine terminé !")

if __name__ == "__main__":
    asyncio.run(main())