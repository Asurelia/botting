"""
Strategic Long-Term Planner pour TacticalBot
Planification stratégique sur plusieurs jours/semaines
Gestion progression personnage, économie, métiers, équipement

Fonctionnalités:
- Planification progression niveau (1-200)
- Gestion économique long-terme (millions de kamas)
- Développement métiers (niveau 200)
- Optimisation équipement progressif
- Adaptation dynamique aux résultats
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
from enum import Enum

from ...engine.module_interface import IModule, ModuleStatus
from ...state.realtime_state import GameState


class GoalType(Enum):
    """Types d'objectifs stratégiques"""
    LEVEL_PROGRESSION = "level_progression"
    ECONOMIC_GROWTH = "economic_growth"
    PROFESSION_MASTERY = "profession_mastery"
    EQUIPMENT_UPGRADE = "equipment_upgrade"
    QUEST_COMPLETION = "quest_completion"
    ACHIEVEMENT_HUNTING = "achievement_hunting"
    RESOURCE_STOCKPILING = "resource_stockpiling"


class GoalPriority(Enum):
    """Priorités des objectifs"""
    CRITICAL = 1  # Bloquant pour progression
    HIGH = 2      # Important mais pas bloquant
    MEDIUM = 3    # Bénéfique
    LOW = 4       # Optionnel


@dataclass
class StrategicGoal:
    """Objectif stratégique long-terme"""
    id: str
    name: str
    goal_type: GoalType
    priority: GoalPriority
    
    # Cibles
    target_value: float
    current_value: float = 0.0
    
    # Temporalité
    deadline: Optional[datetime] = None
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(days=1))
    
    # Dépendances
    prerequisites: List[str] = field(default_factory=list)
    sub_goals: List[str] = field(default_factory=list)
    
    # Métriques
    progress_rate: float = 0.0  # Progression par heure
    efficiency_score: float = 0.0
    
    # Métadonnées
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def progress_percentage(self) -> float:
        """Calcule le pourcentage de progression"""
        if self.target_value == 0:
            return 100.0
        return min(100.0, (self.current_value / self.target_value) * 100)
    
    def is_completed(self) -> bool:
        """Vérifie si l'objectif est atteint"""
        return self.current_value >= self.target_value
    
    def time_remaining(self) -> Optional[timedelta]:
        """Estime le temps restant"""
        if self.progress_rate <= 0:
            return None
        remaining = self.target_value - self.current_value
        hours_needed = remaining / self.progress_rate
        return timedelta(hours=hours_needed)


@dataclass
class ActivityPlan:
    """Plan d'activité pour atteindre un objectif"""
    goal_id: str
    activity_type: str  # "farming", "questing", "crafting", etc.
    location: str
    duration: timedelta
    expected_gain: Dict[str, float]  # {"xp": 50000, "kamas": 10000}
    requirements: Dict[str, Any]
    priority: int = 5


class LevelProgressionPlanner:
    """Planificateur de progression de niveau"""
    
    def __init__(self):
        self.xp_curves = self._load_xp_curves()
        self.optimal_zones = self._load_optimal_zones()
    
    def plan_level_progression(self, current_level: int, target_level: int,
                              current_xp: int) -> List[ActivityPlan]:
        """Planifie la progression de niveau"""
        plans = []
        
        # Calcul XP nécessaire
        total_xp_needed = self._calculate_xp_needed(current_level, target_level, current_xp)
        
        # Découpage en phases
        level_ranges = self._split_level_ranges(current_level, target_level)
        
        for level_range in level_ranges:
            # Sélection zone optimale
            zone = self._select_optimal_zone(level_range)
            
            # Estimation temps et gains
            xp_per_hour = self._estimate_xp_rate(level_range, zone)
            duration_hours = total_xp_needed / xp_per_hour
            
            plan = ActivityPlan(
                goal_id=f"level_{level_range[0]}_to_{level_range[1]}",
                activity_type="farming",
                location=zone,
                duration=timedelta(hours=duration_hours),
                expected_gain={
                    "xp": total_xp_needed,
                    "kamas": xp_per_hour * duration_hours * 0.5  # Estimation
                },
                requirements={"level": level_range[0]},
                priority=1
            )
            plans.append(plan)
        
        return plans
    
    def _calculate_xp_needed(self, current_level: int, target_level: int, current_xp: int) -> int:
        """Calcule l'XP totale nécessaire"""
        # Formule simplifiée (à adapter selon Dofus)
        xp_needed = 0
        for level in range(current_level, target_level):
            xp_needed += self._xp_for_level(level)
        return xp_needed - current_xp
    
    def _xp_for_level(self, level: int) -> int:
        """XP nécessaire pour un niveau"""
        # Formule exponentielle simplifiée
        return int(100 * (level ** 2.5))
    
    def _split_level_ranges(self, start: int, end: int) -> List[Tuple[int, int]]:
        """Découpe en ranges de 10 niveaux"""
        ranges = []
        current = start
        while current < end:
            next_level = min(current + 10, end)
            ranges.append((current, next_level))
            current = next_level
        return ranges
    
    def _select_optimal_zone(self, level_range: Tuple[int, int]) -> str:
        """Sélectionne la zone optimale pour un range de niveau"""
        avg_level = (level_range[0] + level_range[1]) / 2
        
        # Base de données simplifiée (à enrichir)
        if avg_level < 20:
            return "Astrub Plains"
        elif avg_level < 50:
            return "Cania Plains"
        elif avg_level < 100:
            return "Frigost"
        else:
            return "Eliocalypse"
    
    def _estimate_xp_rate(self, level_range: Tuple[int, int], zone: str) -> float:
        """Estime le taux XP/heure"""
        # Estimation basée sur niveau et zone
        avg_level = (level_range[0] + level_range[1]) / 2
        base_rate = 10000 * (avg_level / 10)
        
        # Bonus de zone
        zone_multipliers = {
            "Astrub Plains": 1.0,
            "Cania Plains": 1.2,
            "Frigost": 1.5,
            "Eliocalypse": 2.0
        }
        
        return base_rate * zone_multipliers.get(zone, 1.0)
    
    def _load_xp_curves(self) -> Dict:
        """Charge les courbes d'XP"""
        return {}
    
    def _load_optimal_zones(self) -> Dict:
        """Charge les zones optimales"""
        return {}


class EconomicPlanner:
    """Planificateur économique long-terme"""
    
    def __init__(self):
        self.market_data = {}
        self.profit_activities = []
    
    def plan_economic_growth(self, current_kamas: int, target_kamas: int,
                           available_time: timedelta) -> List[ActivityPlan]:
        """Planifie la croissance économique"""
        plans = []
        kamas_needed = target_kamas - current_kamas
        
        # Identification des activités rentables
        profitable_activities = self._identify_profitable_activities()
        
        # Diversification des sources de revenus
        for activity in profitable_activities[:3]:  # Top 3 activités
            portion = kamas_needed / 3
            time_needed = portion / activity["kamas_per_hour"]
            
            plan = ActivityPlan(
                goal_id=f"economic_{activity['name']}",
                activity_type=activity["type"],
                location=activity["location"],
                duration=timedelta(hours=time_needed),
                expected_gain={"kamas": portion},
                requirements=activity.get("requirements", {}),
                priority=2
            )
            plans.append(plan)
        
        return plans
    
    def _identify_profitable_activities(self) -> List[Dict]:
        """Identifie les activités les plus rentables"""
        activities = [
            {
                "name": "Resource Farming",
                "type": "farming",
                "location": "Astrub Forest",
                "kamas_per_hour": 50000,
                "requirements": {"level": 1}
            },
            {
                "name": "Dungeon Running",
                "type": "dungeon",
                "location": "Scaraleaf Dungeon",
                "kamas_per_hour": 100000,
                "requirements": {"level": 50}
            },
            {
                "name": "Crafting",
                "type": "crafting",
                "location": "Workshop",
                "kamas_per_hour": 75000,
                "requirements": {"profession_level": 50}
            }
        ]
        
        # Tri par rentabilité
        return sorted(activities, key=lambda x: x["kamas_per_hour"], reverse=True)
    
    def optimize_investments(self, available_kamas: int) -> Dict[str, int]:
        """Optimise les investissements"""
        investments = {}
        
        # Équipement (30%)
        investments["equipment"] = int(available_kamas * 0.3)
        
        # Ressources pour craft (40%)
        investments["resources"] = int(available_kamas * 0.4)
        
        # Réserve d'urgence (30%)
        investments["reserve"] = int(available_kamas * 0.3)
        
        return investments


class ProfessionPlanner:
    """Planificateur de développement des métiers"""
    
    def __init__(self):
        self.profession_data = self._load_profession_data()
    
    def plan_profession_mastery(self, profession: str, current_level: int,
                               target_level: int) -> List[ActivityPlan]:
        """Planifie la maîtrise d'un métier"""
        plans = []
        
        # Découpage en phases
        level_milestones = [20, 40, 60, 80, 100, 150, 200]
        relevant_milestones = [m for m in level_milestones if current_level < m <= target_level]
        
        for milestone in relevant_milestones:
            # Ressources nécessaires
            resources = self._calculate_resources_needed(profession, current_level, milestone)
            
            # Plan de collecte
            collection_plan = self._plan_resource_collection(resources)
            
            # Plan de craft
            craft_plan = ActivityPlan(
                goal_id=f"profession_{profession}_{milestone}",
                activity_type="crafting",
                location="Workshop",
                duration=timedelta(hours=10),  # Estimation
                expected_gain={"profession_xp": 1000000},
                requirements={"resources": resources},
                priority=3
            )
            
            plans.extend(collection_plan)
            plans.append(craft_plan)
            
            current_level = milestone
        
        return plans
    
    def _calculate_resources_needed(self, profession: str, start_level: int,
                                   end_level: int) -> Dict[str, int]:
        """Calcule les ressources nécessaires"""
        # Estimation simplifiée
        levels_to_gain = end_level - start_level
        return {
            "basic_resource": levels_to_gain * 100,
            "intermediate_resource": levels_to_gain * 50,
            "rare_resource": levels_to_gain * 10
        }
    
    def _plan_resource_collection(self, resources: Dict[str, int]) -> List[ActivityPlan]:
        """Planifie la collecte de ressources"""
        plans = []
        
        for resource, quantity in resources.items():
            plan = ActivityPlan(
                goal_id=f"collect_{resource}",
                activity_type="gathering",
                location=self._get_resource_location(resource),
                duration=timedelta(hours=quantity / 100),  # 100 par heure
                expected_gain={"resources": {resource: quantity}},
                requirements={},
                priority=4
            )
            plans.append(plan)
        
        return plans
    
    def _get_resource_location(self, resource: str) -> str:
        """Obtient la meilleure location pour une ressource"""
        locations = {
            "basic_resource": "Astrub Forest",
            "intermediate_resource": "Cania Plains",
            "rare_resource": "Frigost Mountains"
        }
        return locations.get(resource, "Unknown")
    
    def _load_profession_data(self) -> Dict:
        """Charge les données des métiers"""
        return {}


class StrategicPlanner(IModule):
    """
    Planificateur stratégique long-terme
    Gère les objectifs sur plusieurs jours/semaines
    """
    
    def __init__(self, name: str = "strategic_planner"):
        super().__init__(name)
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Planificateurs spécialisés
        self.level_planner = LevelProgressionPlanner()
        self.economic_planner = EconomicPlanner()
        self.profession_planner = ProfessionPlanner()
        
        # Objectifs actifs
        self.active_goals: Dict[str, StrategicGoal] = {}
        self.completed_goals: List[StrategicGoal] = []
        
        # Plans d'activités
        self.activity_queue: deque = deque()
        self.current_activity: Optional[ActivityPlan] = None
        
        # Configuration
        self.planning_horizon = timedelta(days=7)  # Planification sur 7 jours
        self.replan_interval = timedelta(hours=6)  # Replanification toutes les 6h
        self.last_planning = datetime.now()
        
        # Métriques
        self.metrics = {
            "goals_completed": 0,
            "total_xp_gained": 0,
            "total_kamas_earned": 0,
            "planning_accuracy": 0.0,
            "average_efficiency": 0.0
        }
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialise le planificateur"""
        try:
            self.status = ModuleStatus.INITIALIZING
            
            # Configuration
            self.planning_horizon = timedelta(days=config.get("planning_horizon_days", 7))
            self.replan_interval = timedelta(hours=config.get("replan_interval_hours", 6))
            
            # Chargement des objectifs sauvegardés
            self._load_saved_goals()
            
            self.status = ModuleStatus.ACTIVE
            self.logger.info("Planificateur stratégique initialisé")
            return True
        
        except Exception as e:
            self.logger.error(f"Erreur initialisation: {e}")
            self.status = ModuleStatus.ERROR
            return False
    
    def update(self, game_state: Any) -> Optional[Dict[str, Any]]:
        """Met à jour la planification"""
        if not self.is_active():
            return None
        
        try:
            current_time = datetime.now()
            
            # Mise à jour progression des objectifs
            self._update_goal_progress(game_state)
            
            # Replanification périodique
            if current_time - self.last_planning >= self.replan_interval:
                self._replan_strategy(game_state)
                self.last_planning = current_time
            
            # Sélection activité courante
            if not self.current_activity or self._is_activity_completed():
                self.current_activity = self._select_next_activity(game_state)
            
            return {
                "strategic_plan": {
                    "active_goals": len(self.active_goals),
                    "current_activity": self.current_activity.activity_type if self.current_activity else None,
                    "queue_size": len(self.activity_queue),
                    "next_replan": (self.last_planning + self.replan_interval).isoformat()
                }
            }
        
        except Exception as e:
            self.logger.error(f"Erreur mise à jour: {e}")
            return None
    
    def add_goal(self, goal: StrategicGoal) -> bool:
        """Ajoute un objectif stratégique"""
        try:
            self.active_goals[goal.id] = goal
            self.logger.info(f"Objectif ajouté: {goal.name} ({goal.goal_type.value})")
            
            # Replanification pour intégrer le nouvel objectif
            self._generate_activity_plans()
            
            return True
        
        except Exception as e:
            self.logger.error(f"Erreur ajout objectif: {e}")
            return False
    
    def _update_goal_progress(self, game_state: GameState):
        """Met à jour la progression des objectifs"""
        for goal_id, goal in list(self.active_goals.items()):
            # Mise à jour selon le type d'objectif
            if goal.goal_type == GoalType.LEVEL_PROGRESSION:
                goal.current_value = game_state.character.level
            
            elif goal.goal_type == GoalType.ECONOMIC_GROWTH:
                goal.current_value = game_state.character.kamas
            
            # Calcul taux de progression
            time_elapsed = (datetime.now() - goal.last_updated).total_seconds() / 3600
            if time_elapsed > 0:
                progress_delta = goal.current_value - goal.metadata.get("last_value", 0)
                goal.progress_rate = progress_delta / time_elapsed
                goal.metadata["last_value"] = goal.current_value
            
            goal.last_updated = datetime.now()
            
            # Vérification complétion
            if goal.is_completed():
                self._complete_goal(goal)
    
    def _complete_goal(self, goal: StrategicGoal):
        """Marque un objectif comme complété"""
        self.completed_goals.append(goal)
        del self.active_goals[goal.id]
        self.metrics["goals_completed"] += 1
        self.logger.info(f"✅ Objectif complété: {goal.name}")
    
    def _replan_strategy(self, game_state: GameState):
        """Replanifie la stratégie globale"""
        self.logger.info("Replanification stratégique...")
        
        # Évaluation des objectifs actuels
        self._evaluate_goal_feasibility()
        
        # Génération de nouveaux plans
        self._generate_activity_plans()
        
        # Optimisation de l'ordre d'exécution
        self._optimize_activity_order()
    
    def _generate_activity_plans(self):
        """Génère les plans d'activités pour tous les objectifs"""
        self.activity_queue.clear()
        
        for goal in self.active_goals.values():
            plans = []
            
            if goal.goal_type == GoalType.LEVEL_PROGRESSION:
                plans = self.level_planner.plan_level_progression(
                    int(goal.current_value),
                    int(goal.target_value),
                    0  # XP actuel (à récupérer du game_state)
                )
            
            elif goal.goal_type == GoalType.ECONOMIC_GROWTH:
                plans = self.economic_planner.plan_economic_growth(
                    int(goal.current_value),
                    int(goal.target_value),
                    self.planning_horizon
                )
            
            elif goal.goal_type == GoalType.PROFESSION_MASTERY:
                profession = goal.metadata.get("profession", "unknown")
                plans = self.profession_planner.plan_profession_mastery(
                    profession,
                    int(goal.current_value),
                    int(goal.target_value)
                )
            
            # Ajout à la queue
            for plan in plans:
                self.activity_queue.append(plan)
    
    def _optimize_activity_order(self):
        """Optimise l'ordre d'exécution des activités"""
        # Tri par priorité puis par efficacité
        sorted_activities = sorted(
            self.activity_queue,
            key=lambda x: (x.priority, -sum(x.expected_gain.values()))
        )
        self.activity_queue = deque(sorted_activities)
    
    def _select_next_activity(self, game_state: GameState) -> Optional[ActivityPlan]:
        """Sélectionne la prochaine activité à exécuter"""
        if not self.activity_queue:
            return None
        
        # Filtrage des activités faisables
        feasible_activities = [
            activity for activity in self.activity_queue
            if self._is_activity_feasible(activity, game_state)
        ]
        
        if not feasible_activities:
            return None
        
        # Sélection de la meilleure activité
        return feasible_activities[0]
    
    def _is_activity_feasible(self, activity: ActivityPlan, game_state: GameState) -> bool:
        """Vérifie si une activité est réalisable"""
        # Vérification niveau
        if "level" in activity.requirements:
            if game_state.character.level < activity.requirements["level"]:
                return False
        
        # Vérification ressources
        if "resources" in activity.requirements:
            # À implémenter: vérifier inventaire
            pass
        
        return True
    
    def _is_activity_completed(self) -> bool:
        """Vérifie si l'activité courante est terminée"""
        if not self.current_activity:
            return True
        
        # À implémenter: logique de vérification
        return False
    
    def _evaluate_goal_feasibility(self):
        """Évalue la faisabilité des objectifs"""
        for goal in self.active_goals.values():
            # Calcul temps restant estimé
            time_remaining = goal.time_remaining()
            
            if time_remaining and goal.deadline:
                if datetime.now() + time_remaining > goal.deadline:
                    self.logger.warning(f"⚠️ Objectif {goal.name} risque de ne pas être atteint à temps")
                    # Ajustement priorité ou objectif
                    goal.priority = GoalPriority.CRITICAL
    
    def _load_saved_goals(self):
        """Charge les objectifs sauvegardés"""
        try:
            # Placeholder pour chargement depuis fichier
            pass
        except Exception as e:
            self.logger.warning(f"Impossible de charger les objectifs: {e}")
    
    def get_state(self) -> Dict[str, Any]:
        """Retourne l'état du planificateur"""
        return {
            "status": self.status.value,
            "active_goals": len(self.active_goals),
            "completed_goals": len(self.completed_goals),
            "activity_queue_size": len(self.activity_queue),
            "current_activity": self.current_activity.activity_type if self.current_activity else None,
            "metrics": self.metrics
        }
    
    def cleanup(self) -> None:
        """Nettoie le planificateur"""
        try:
            # Sauvegarde des objectifs
            self._save_goals()
            self.logger.info("Planificateur stratégique nettoyé")
        except Exception as e:
            self.logger.error(f"Erreur nettoyage: {e}")
    
    def _save_goals(self):
        """Sauvegarde les objectifs"""
        try:
            # Placeholder pour sauvegarde
            pass
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde: {e}")
    
    def get_planning_report(self) -> Dict[str, Any]:
        """Génère un rapport de planification"""
        total_time_planned = sum(
            activity.duration.total_seconds() / 3600
            for activity in self.activity_queue
        )
        
        return {
            "active_goals": [
                {
                    "name": goal.name,
                    "type": goal.goal_type.value,
                    "progress": goal.progress_percentage(),
                    "time_remaining": str(goal.time_remaining()) if goal.time_remaining() else "Unknown"
                }
                for goal in self.active_goals.values()
            ],
            "total_activities_planned": len(self.activity_queue),
            "total_time_planned_hours": total_time_planned,
            "next_milestone": self._get_next_milestone(),
            "metrics": self.metrics
        }
    
    def _get_next_milestone(self) -> Optional[str]:
        """Obtient le prochain milestone important"""
        for goal in sorted(self.active_goals.values(), key=lambda g: g.priority.value):
            if not goal.is_completed():
                return f"{goal.name} - {goal.progress_percentage():.1f}%"
        return None
