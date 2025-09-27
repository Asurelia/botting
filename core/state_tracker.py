"""
Multi-Dimensional State Tracking System
Système de suivi d'état sur plusieurs dimensions temporelles pour l'IA DOFUS
Gestion des états immédiat, tactique, stratégique et méta
"""

import asyncio
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor

# Import modules internes
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

class StateLevel(Enum):
    """Niveaux de granularité temporelle"""
    IMMEDIATE = "immediate"      # Seconde à seconde
    TACTICAL = "tactical"        # Minute par minute
    STRATEGIC = "strategic"      # Heure par heure / Jour par jour
    META = "meta"               # Semaine / Mois

class StateCategory(Enum):
    """Catégories d'état du jeu"""
    # États du personnage
    CHARACTER = "character"
    POSITION = "position"
    INVENTORY = "inventory"
    SKILLS = "skills"

    # États de l'environnement
    ENVIRONMENT = "environment"
    WEATHER = "weather"
    SERVER = "server"

    # États sociaux
    TEAM = "team"
    GUILD = "guild"
    SOCIAL = "social"

    # États économiques
    MARKET = "market"
    RESOURCES = "resources"
    ECONOMY = "economy"

    # États de mission
    OBJECTIVES = "objectives"
    QUESTS = "quests"
    PROGRESSION = "progression"

@dataclass
class GameState:
    """État immédiat du jeu (seconde par seconde)"""
    timestamp: datetime = field(default_factory=datetime.now)

    # Position et mouvement
    position: Tuple[int, int] = (0, 0)
    map_id: str = "unknown"
    direction: str = "south"
    is_moving: bool = False

    # État du personnage
    health_percent: float = 100.0
    mana_percent: float = 100.0
    energy_percent: float = 100.0
    level: int = 1
    experience_percent: float = 0.0

    # Combat
    in_combat: bool = False
    target_id: Optional[str] = None
    combat_round: int = 0

    # Interface
    interface_state: Dict[str, bool] = field(default_factory=dict)
    active_windows: List[str] = field(default_factory=list)
    cursor_position: Tuple[int, int] = (0, 0)

    # Inventaire immédiat
    inventory_slots_used: int = 0
    inventory_slots_total: int = 60
    pods_used: int = 0
    pods_total: int = 1000

    # Métadonnées
    lag_ms: float = 0.0
    fps: float = 60.0

    def is_healthy(self) -> bool:
        """Vérifie si l'état est sain"""
        return (self.health_percent > 20.0 and
                self.mana_percent > 10.0 and
                self.energy_percent > 10.0)

    def get_load_factor(self) -> float:
        """Calcule le facteur de charge (inventaire/pods)"""
        return max(
            self.inventory_slots_used / max(1, self.inventory_slots_total),
            self.pods_used / max(1, self.pods_total)
        )

@dataclass
class TacticalState:
    """État tactique (minute par minute)"""
    timestamp: datetime = field(default_factory=datetime.now)

    # Activité actuelle
    current_activity: str = "idle"
    activity_start_time: datetime = field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None

    # Objectifs tactiques
    immediate_goals: List[str] = field(default_factory=list)
    goal_priorities: Dict[str, int] = field(default_factory=dict)

    # Performance récente
    actions_per_minute: float = 0.0
    efficiency_score: float = 1.0
    error_rate: float = 0.0

    # Ressources
    resource_consumption_rate: Dict[str, float] = field(default_factory=dict)
    resource_availability: Dict[str, float] = field(default_factory=dict)

    # Contexte environnemental
    zone_safety_level: float = 1.0
    player_density: int = 0
    resource_competition: float = 0.0

    # Adaptation
    last_decision_time: datetime = field(default_factory=datetime.now)
    decision_frequency: float = 1.0  # Décisions par minute
    adaptation_needed: bool = False

@dataclass
class StrategicPlanning:
    """Planification stratégique (heure/jour)"""
    timestamp: datetime = field(default_factory=datetime.now)

    # Objectifs stratégiques
    daily_goals: List[Dict[str, Any]] = field(default_factory=list)
    weekly_goals: List[Dict[str, Any]] = field(default_factory=list)
    goal_dependencies: Dict[str, List[str]] = field(default_factory=dict)

    # Planification temporelle
    planned_activities: List[Tuple[datetime, str, timedelta]] = field(default_factory=list)
    time_allocations: Dict[str, timedelta] = field(default_factory=dict)

    # Progression
    daily_progress: Dict[str, float] = field(default_factory=dict)
    weekly_progress: Dict[str, float] = field(default_factory=dict)
    milestone_achievements: List[Dict[str, Any]] = field(default_factory=list)

    # Optimisation
    efficiency_targets: Dict[str, float] = field(default_factory=dict)
    resource_budgets: Dict[str, float] = field(default_factory=dict)

    # Prédictions
    expected_outcomes: Dict[str, Any] = field(default_factory=dict)
    risk_assessments: Dict[str, float] = field(default_factory=dict)

    def calculate_overall_progress(self) -> float:
        """Calcule la progression globale"""
        if not self.daily_goals:
            return 1.0

        completed = sum(1 for goal in self.daily_goals if goal.get('completed', False))
        return completed / len(self.daily_goals)

@dataclass
class LongTermGoals:
    """Objectifs méta (semaine/mois)"""
    timestamp: datetime = field(default_factory=datetime.now)

    # Objectifs à long terme
    character_development: Dict[str, Any] = field(default_factory=dict)
    economic_goals: Dict[str, Any] = field(default_factory=dict)
    social_objectives: Dict[str, Any] = field(default_factory=dict)

    # Évolution
    skill_development_plan: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    equipment_upgrade_path: List[Dict[str, Any]] = field(default_factory=list)
    wealth_accumulation_strategy: Dict[str, Any] = field(default_factory=dict)

    # Métriques de succès
    key_performance_indicators: Dict[str, float] = field(default_factory=dict)
    benchmark_comparisons: Dict[str, float] = field(default_factory=dict)

    # Adaptation stratégique
    strategy_revisions: List[Dict[str, Any]] = field(default_factory=list)
    learning_insights: List[str] = field(default_factory=list)

    def get_priority_objectives(self, top_n: int = 3) -> List[Dict[str, Any]]:
        """Retourne les objectifs prioritaires"""
        all_objectives = []

        # Collecte de tous les objectifs avec leurs priorités
        for category in [self.character_development, self.economic_goals, self.social_objectives]:
            for obj_id, obj_data in category.items():
                if isinstance(obj_data, dict):
                    priority = obj_data.get('priority', 5)
                    all_objectives.append({
                        'id': obj_id,
                        'priority': priority,
                        'data': obj_data
                    })

        # Tri par priorité et retour des top_n
        all_objectives.sort(key=lambda x: x['priority'])
        return all_objectives[:top_n]

class StateTransition:
    """Gestionnaire de transitions d'état"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.StateTransition")
        self.transition_rules: Dict[str, List[Dict[str, Any]]] = {}
        self.state_validators: Dict[StateLevel, callable] = {}

    def add_transition_rule(self, from_state: str, to_state: str,
                          condition: callable, action: callable = None):
        """Ajoute une règle de transition d'état"""
        if from_state not in self.transition_rules:
            self.transition_rules[from_state] = []

        rule = {
            'to_state': to_state,
            'condition': condition,
            'action': action,
            'created_at': datetime.now()
        }

        self.transition_rules[from_state].append(rule)

    def check_transitions(self, current_state: str, context: Dict[str, Any]) -> Optional[str]:
        """Vérifie si une transition doit avoir lieu"""
        if current_state not in self.transition_rules:
            return None

        for rule in self.transition_rules[current_state]:
            try:
                if rule['condition'](context):
                    if rule['action']:
                        rule['action'](context)
                    return rule['to_state']
            except Exception as e:
                self.logger.error(f"Erreur évaluation transition: {e}")

        return None

class StatePredictor:
    """Prédicteur d'évolution d'état"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.StatePredictor")
        self.historical_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.prediction_models: Dict[str, Any] = {}

    def record_state_change(self, state_type: str, old_state: Any, new_state: Any, context: Dict[str, Any]):
        """Enregistre un changement d'état pour l'apprentissage"""
        change_record = {
            'timestamp': datetime.now(),
            'old_state': old_state,
            'new_state': new_state,
            'context': context,
            'change_magnitude': self._calculate_change_magnitude(old_state, new_state)
        }

        self.historical_patterns[state_type].append(change_record)

    def predict_state_evolution(self, current_state: Any, state_type: str,
                               horizon: timedelta) -> Dict[str, Any]:
        """Prédit l'évolution d'un état"""
        try:
            if state_type not in self.historical_patterns:
                return {'prediction': current_state, 'confidence': 0.1}

            patterns = list(self.historical_patterns[state_type])
            if len(patterns) < 10:
                return {'prediction': current_state, 'confidence': 0.3}

            # Analyse des patterns récents
            recent_changes = patterns[-20:]
            avg_change_rate = self._calculate_average_change_rate(recent_changes)

            # Prédiction simple basée sur la tendance
            predicted_state = self._apply_trend_prediction(current_state, avg_change_rate, horizon)

            # Calcul de la confiance
            confidence = min(0.9, len(patterns) / 100.0)

            return {
                'prediction': predicted_state,
                'confidence': confidence,
                'trend': avg_change_rate,
                'horizon': horizon
            }

        except Exception as e:
            self.logger.error(f"Erreur prédiction état: {e}")
            return {'prediction': current_state, 'confidence': 0.1}

    def _calculate_change_magnitude(self, old_state: Any, new_state: Any) -> float:
        """Calcule la magnitude du changement"""
        try:
            if hasattr(old_state, '__dict__') and hasattr(new_state, '__dict__'):
                old_dict = asdict(old_state) if hasattr(old_state, '__dataclass_fields__') else old_state.__dict__
                new_dict = asdict(new_state) if hasattr(new_state, '__dataclass_fields__') else new_state.__dict__

                changes = 0
                total_fields = 0

                for key in old_dict.keys():
                    if key in new_dict:
                        total_fields += 1
                        if old_dict[key] != new_dict[key]:
                            changes += 1

                return changes / max(1, total_fields)
            else:
                return 1.0 if old_state != new_state else 0.0

        except Exception:
            return 0.5  # Valeur par défaut

    def _calculate_average_change_rate(self, changes: List[Dict[str, Any]]) -> float:
        """Calcule le taux de changement moyen"""
        if not changes:
            return 0.0

        magnitudes = [change['change_magnitude'] for change in changes]
        return sum(magnitudes) / len(magnitudes)

    def _apply_trend_prediction(self, current_state: Any, change_rate: float, horizon: timedelta) -> Any:
        """Applique une prédiction de tendance"""
        # Implémentation simplifiée - retourne l'état actuel avec variations mineures
        return current_state

class MultiDimensionalStateTracker:
    """
    Tracker d'état principal gérant toutes les dimensions temporelles
    Coordonne les états immédiat, tactique, stratégique et méta
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # États actuels par dimension
        self.current_state = GameState()
        self.tactical_context = TacticalState()
        self.strategic_goals = StrategicPlanning()
        self.meta_objectives = LongTermGoals()

        # Historiques par niveau
        self.state_history: Dict[StateLevel, deque] = {
            StateLevel.IMMEDIATE: deque(maxlen=3600),    # 1 heure d'historique
            StateLevel.TACTICAL: deque(maxlen=1440),     # 24 heures d'historique
            StateLevel.STRATEGIC: deque(maxlen=168),     # 1 semaine d'historique
            StateLevel.META: deque(maxlen=365)           # 1 an d'historique
        }

        # Composants auxiliaires
        self.state_transitions = StateTransition()
        self.state_predictor = StatePredictor()

        # Configuration de mise à jour
        self.update_intervals = {
            StateLevel.IMMEDIATE: timedelta(seconds=1),
            StateLevel.TACTICAL: timedelta(minutes=1),
            StateLevel.STRATEGIC: timedelta(hours=1),
            StateLevel.META: timedelta(days=1)
        }

        self.last_updates = {level: datetime.now() for level in StateLevel}

        # Threading pour mises à jour asynchrones
        self.update_lock = threading.Lock()
        self.is_running = False
        self.update_tasks = {}

        # Métriques de performance
        self.update_metrics = {
            'total_updates': 0,
            'failed_updates': 0,
            'average_update_time': 0.0,
            'last_performance_check': datetime.now()
        }

        # Initialisation des règles de transition
        self._initialize_transition_rules()

    async def start_tracking(self) -> bool:
        """Démarre le suivi d'état en continu"""
        try:
            self.is_running = True
            self.logger.info("Démarrage du suivi d'état multi-dimensionnel...")

            # Lancement des tâches de mise à jour pour chaque niveau
            for level in StateLevel:
                self.update_tasks[level] = asyncio.create_task(
                    self._continuous_update_loop(level)
                )

            # Tâche de nettoyage périodique
            self.update_tasks['cleanup'] = asyncio.create_task(
                self._cleanup_loop()
            )

            self.logger.info("✅ Suivi d'état démarré avec succès")
            return True

        except Exception as e:
            self.logger.error(f"Erreur démarrage suivi d'état: {e}")
            return False

    async def stop_tracking(self):
        """Arrête le suivi d'état"""
        self.logger.info("Arrêt du suivi d'état...")
        self.is_running = False

        # Annulation de toutes les tâches
        for task in self.update_tasks.values():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self.logger.info("✅ Suivi d'état arrêté")

    async def update_immediate_state(self, new_state: GameState) -> bool:
        """Met à jour l'état immédiat"""
        try:
            with self.update_lock:
                # Enregistrement de l'ancien état
                old_state = self.current_state

                # Mise à jour
                self.current_state = new_state
                self.last_updates[StateLevel.IMMEDIATE] = datetime.now()

                # Historique
                self.state_history[StateLevel.IMMEDIATE].append({
                    'timestamp': new_state.timestamp,
                    'state': new_state,
                    'transition_from': old_state
                })

                # Enregistrement pour prédiction
                self.state_predictor.record_state_change(
                    'immediate', old_state, new_state, {}
                )

                # Vérification des transitions
                await self._check_state_transitions('immediate', new_state)

                self.update_metrics['total_updates'] += 1
                return True

        except Exception as e:
            self.logger.error(f"Erreur mise à jour état immédiat: {e}")
            self.update_metrics['failed_updates'] += 1
            return False

    async def update_tactical_context(self, updates: Dict[str, Any]) -> bool:
        """Met à jour le contexte tactique"""
        try:
            with self.update_lock:
                old_context = self.tactical_context

                # Application des mises à jour
                for key, value in updates.items():
                    if hasattr(self.tactical_context, key):
                        setattr(self.tactical_context, key, value)

                self.tactical_context.timestamp = datetime.now()
                self.last_updates[StateLevel.TACTICAL] = datetime.now()

                # Historique
                self.state_history[StateLevel.TACTICAL].append({
                    'timestamp': self.tactical_context.timestamp,
                    'state': self.tactical_context,
                    'updates': updates
                })

                # Calcul de métriques tactiques
                await self._calculate_tactical_metrics()

                return True

        except Exception as e:
            self.logger.error(f"Erreur mise à jour contexte tactique: {e}")
            return False

    async def update_strategic_planning(self, planning_updates: Dict[str, Any]) -> bool:
        """Met à jour la planification stratégique"""
        try:
            with self.update_lock:
                # Mise à jour des objectifs
                if 'daily_goals' in planning_updates:
                    self.strategic_goals.daily_goals = planning_updates['daily_goals']

                if 'weekly_goals' in planning_updates:
                    self.strategic_goals.weekly_goals = planning_updates['weekly_goals']

                # Mise à jour des allocations temporelles
                if 'time_allocations' in planning_updates:
                    self.strategic_goals.time_allocations.update(
                        planning_updates['time_allocations']
                    )

                # Mise à jour des progressions
                if 'progress_updates' in planning_updates:
                    for goal_id, progress in planning_updates['progress_updates'].items():
                        self.strategic_goals.daily_progress[goal_id] = progress

                self.strategic_goals.timestamp = datetime.now()
                self.last_updates[StateLevel.STRATEGIC] = datetime.now()

                # Historique
                self.state_history[StateLevel.STRATEGIC].append({
                    'timestamp': self.strategic_goals.timestamp,
                    'state': self.strategic_goals,
                    'updates': planning_updates
                })

                return True

        except Exception as e:
            self.logger.error(f"Erreur mise à jour planification stratégique: {e}")
            return False

    async def update_meta_objectives(self, meta_updates: Dict[str, Any]) -> bool:
        """Met à jour les objectifs méta"""
        try:
            with self.update_lock:
                # Mise à jour des KPIs
                if 'kpis' in meta_updates:
                    self.meta_objectives.key_performance_indicators.update(
                        meta_updates['kpis']
                    )

                # Mise à jour des objectifs de développement
                if 'character_development' in meta_updates:
                    self.meta_objectives.character_development.update(
                        meta_updates['character_development']
                    )

                # Ajout d'insights d'apprentissage
                if 'learning_insights' in meta_updates:
                    self.meta_objectives.learning_insights.extend(
                        meta_updates['learning_insights']
                    )

                self.meta_objectives.timestamp = datetime.now()
                self.last_updates[StateLevel.META] = datetime.now()

                # Historique
                self.state_history[StateLevel.META].append({
                    'timestamp': self.meta_objectives.timestamp,
                    'state': self.meta_objectives,
                    'updates': meta_updates
                })

                return True

        except Exception as e:
            self.logger.error(f"Erreur mise à jour objectifs méta: {e}")
            return False

    def get_state_summary(self, level: Optional[StateLevel] = None) -> Dict[str, Any]:
        """Retourne un résumé de l'état pour un niveau donné ou tous"""
        try:
            if level:
                return self._get_level_summary(level)
            else:
                return {
                    'immediate': self._get_level_summary(StateLevel.IMMEDIATE),
                    'tactical': self._get_level_summary(StateLevel.TACTICAL),
                    'strategic': self._get_level_summary(StateLevel.STRATEGIC),
                    'meta': self._get_level_summary(StateLevel.META),
                    'overall_health': self._calculate_overall_health(),
                    'tracking_metrics': self.update_metrics
                }

        except Exception as e:
            self.logger.error(f"Erreur génération résumé: {e}")
            return {}

    def predict_state_evolution(self, level: StateLevel, horizon: timedelta) -> Dict[str, Any]:
        """Prédit l'évolution d'un niveau d'état"""
        try:
            current_state = self._get_current_state_for_level(level)
            state_type = level.value

            prediction = self.state_predictor.predict_state_evolution(
                current_state, state_type, horizon
            )

            return prediction

        except Exception as e:
            self.logger.error(f"Erreur prédiction évolution: {e}")
            return {}

    async def _continuous_update_loop(self, level: StateLevel):
        """Boucle de mise à jour continue pour un niveau"""
        interval = self.update_intervals[level]

        while self.is_running:
            try:
                await asyncio.sleep(interval.total_seconds())

                if not self.is_running:
                    break

                # Mise à jour automatique selon le niveau
                if level == StateLevel.TACTICAL:
                    await self._auto_update_tactical()
                elif level == StateLevel.STRATEGIC:
                    await self._auto_update_strategic()
                elif level == StateLevel.META:
                    await self._auto_update_meta()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Erreur boucle mise à jour {level.value}: {e}")
                await asyncio.sleep(5)  # Pause en cas d'erreur

    async def _cleanup_loop(self):
        """Boucle de nettoyage périodique"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Nettoyage toutes les heures

                if not self.is_running:
                    break

                # Nettoyage des historiques
                self._cleanup_old_histories()

                # Mise à jour des métriques de performance
                self._update_performance_metrics()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Erreur nettoyage: {e}")

    def _initialize_transition_rules(self):
        """Initialise les règles de transition d'état"""
        # Transition vers combat
        self.state_transitions.add_transition_rule(
            'peaceful',
            'combat',
            lambda ctx: ctx.get('in_combat', False)
        )

        # Transition vers farming
        self.state_transitions.add_transition_rule(
            'idle',
            'farming',
            lambda ctx: ctx.get('current_activity') == 'farming'
        )

        # Transition vers social
        self.state_transitions.add_transition_rule(
            'solo',
            'social',
            lambda ctx: ctx.get('team_size', 0) > 1
        )

    async def _check_state_transitions(self, state_type: str, current_state: Any):
        """Vérifie et applique les transitions d'état"""
        try:
            context = {
                'current_state': current_state,
                'timestamp': datetime.now()
            }

            # Ajout de contexte spécifique selon le type d'état
            if hasattr(current_state, 'in_combat'):
                context['in_combat'] = current_state.in_combat

            if hasattr(current_state, 'current_activity'):
                context['current_activity'] = current_state.current_activity

            new_state = self.state_transitions.check_transitions(state_type, context)

            if new_state:
                self.logger.debug(f"Transition détectée: {state_type} -> {new_state}")

        except Exception as e:
            self.logger.error(f"Erreur vérification transitions: {e}")

    async def _auto_update_tactical(self):
        """Mise à jour automatique du niveau tactique"""
        # Calcul des métriques de performance
        await self._calculate_tactical_metrics()

        # Vérification de l'adaptation nécessaire
        self.tactical_context.adaptation_needed = self._check_adaptation_needed()

    async def _auto_update_strategic(self):
        """Mise à jour automatique du niveau stratégique"""
        # Mise à jour de la progression des objectifs
        self._update_goal_progress()

        # Vérification des échéances
        self._check_goal_deadlines()

    async def _auto_update_meta(self):
        """Mise à jour automatique du niveau méta"""
        # Calcul des KPIs
        self._calculate_kpis()

        # Évaluation des stratégies long-terme
        self._evaluate_long_term_strategies()

    async def _calculate_tactical_metrics(self):
        """Calcule les métriques tactiques"""
        try:
            # Calcul du taux d'actions par minute
            recent_states = list(self.state_history[StateLevel.IMMEDIATE])[-60:]  # Dernière minute

            if recent_states:
                self.tactical_context.actions_per_minute = len(recent_states)

                # Calcul du taux d'erreur
                errors = sum(1 for state in recent_states
                           if hasattr(state['state'], 'error_occurred') and state['state'].error_occurred)
                self.tactical_context.error_rate = errors / len(recent_states)

        except Exception as e:
            self.logger.error(f"Erreur calcul métriques tactiques: {e}")

    def _check_adaptation_needed(self) -> bool:
        """Vérifie si une adaptation est nécessaire"""
        try:
            # Vérification des seuils de performance
            if self.tactical_context.efficiency_score < 0.7:
                return True

            if self.tactical_context.error_rate > 0.1:
                return True

            # Vérification de la stagnation
            last_decision_age = datetime.now() - self.tactical_context.last_decision_time
            if last_decision_age > timedelta(minutes=5):
                return True

            return False

        except Exception:
            return False

    def _update_goal_progress(self):
        """Met à jour la progression des objectifs"""
        try:
            for goal in self.strategic_goals.daily_goals:
                goal_id = goal.get('id')
                if goal_id:
                    # Simulation de progression (à remplacer par vraie logique)
                    current_progress = self.strategic_goals.daily_progress.get(goal_id, 0.0)

                    # Increment basé sur l'activité
                    increment = 0.01 if self.tactical_context.current_activity != 'idle' else 0.0
                    new_progress = min(1.0, current_progress + increment)

                    self.strategic_goals.daily_progress[goal_id] = new_progress

                    if new_progress >= 1.0 and not goal.get('completed', False):
                        goal['completed'] = True
                        goal['completion_time'] = datetime.now().isoformat()

        except Exception as e:
            self.logger.error(f"Erreur mise à jour progression: {e}")

    def _check_goal_deadlines(self):
        """Vérifie les échéances des objectifs"""
        now = datetime.now()

        for goal in self.strategic_goals.daily_goals:
            deadline_str = goal.get('deadline')
            if deadline_str:
                try:
                    deadline = datetime.fromisoformat(deadline_str)
                    if now > deadline and not goal.get('completed', False):
                        goal['overdue'] = True
                        self.logger.warning(f"Objectif {goal.get('name')} en retard")
                except ValueError:
                    pass

    def _calculate_kpis(self):
        """Calcule les indicateurs de performance clés"""
        try:
            # KPI de progression générale
            overall_progress = self.strategic_goals.calculate_overall_progress()
            self.meta_objectives.key_performance_indicators['overall_progress'] = overall_progress

            # KPI d'efficacité tactique
            self.meta_objectives.key_performance_indicators['tactical_efficiency'] = (
                self.tactical_context.efficiency_score
            )

            # KPI de stabilité
            error_rate = self.tactical_context.error_rate
            stability_score = max(0.0, 1.0 - error_rate * 10)
            self.meta_objectives.key_performance_indicators['stability'] = stability_score

        except Exception as e:
            self.logger.error(f"Erreur calcul KPIs: {e}")

    def _evaluate_long_term_strategies(self):
        """Évalue les stratégies à long terme"""
        # Analyse des tendances de performance
        kpis = self.meta_objectives.key_performance_indicators

        # Détection de tendances négatives
        insights = []

        if kpis.get('overall_progress', 0) < 0.5:
            insights.append("Progression globale insuffisante - révision stratégique nécessaire")

        if kpis.get('tactical_efficiency', 1) < 0.7:
            insights.append("Efficacité tactique faible - optimisation des processus requise")

        if kpis.get('stability', 1) < 0.8:
            insights.append("Instabilité détectée - amélioration de la robustesse nécessaire")

        # Ajout des nouveaux insights
        for insight in insights:
            if insight not in self.meta_objectives.learning_insights:
                self.meta_objectives.learning_insights.append(insight)

    def _get_level_summary(self, level: StateLevel) -> Dict[str, Any]:
        """Retourne un résumé pour un niveau spécifique"""
        if level == StateLevel.IMMEDIATE:
            return {
                'state': asdict(self.current_state),
                'health_status': self.current_state.is_healthy(),
                'load_factor': self.current_state.get_load_factor(),
                'last_update': self.last_updates[level].isoformat()
            }
        elif level == StateLevel.TACTICAL:
            return {
                'context': asdict(self.tactical_context),
                'performance_score': self.tactical_context.efficiency_score,
                'adaptation_needed': self.tactical_context.adaptation_needed,
                'last_update': self.last_updates[level].isoformat()
            }
        elif level == StateLevel.STRATEGIC:
            return {
                'planning': {
                    'daily_goals_count': len(self.strategic_goals.daily_goals),
                    'overall_progress': self.strategic_goals.calculate_overall_progress(),
                    'active_activities': len(self.strategic_goals.planned_activities)
                },
                'last_update': self.last_updates[level].isoformat()
            }
        elif level == StateLevel.META:
            return {
                'objectives': {
                    'priority_count': len(self.meta_objectives.get_priority_objectives()),
                    'kpi_count': len(self.meta_objectives.key_performance_indicators),
                    'insights_count': len(self.meta_objectives.learning_insights)
                },
                'last_update': self.last_updates[level].isoformat()
            }
        else:
            return {}

    def _get_current_state_for_level(self, level: StateLevel) -> Any:
        """Retourne l'état actuel pour un niveau"""
        state_map = {
            StateLevel.IMMEDIATE: self.current_state,
            StateLevel.TACTICAL: self.tactical_context,
            StateLevel.STRATEGIC: self.strategic_goals,
            StateLevel.META: self.meta_objectives
        }
        return state_map.get(level)

    def _calculate_overall_health(self) -> Dict[str, float]:
        """Calcule la santé globale du système"""
        return {
            'immediate_health': 1.0 if self.current_state.is_healthy() else 0.5,
            'tactical_efficiency': self.tactical_context.efficiency_score,
            'strategic_progress': self.strategic_goals.calculate_overall_progress(),
            'meta_stability': 1.0 - self.tactical_context.error_rate
        }

    def _cleanup_old_histories(self):
        """Nettoie les anciens historiques"""
        for level, history in self.state_history.items():
            # Les deques ont une taille max automatique, pas de nettoyage nécessaire
            pass

    def _update_performance_metrics(self):
        """Met à jour les métriques de performance"""
        self.update_metrics['last_performance_check'] = datetime.now()

        # Calcul du taux de succès
        if self.update_metrics['total_updates'] > 0:
            success_rate = 1.0 - (self.update_metrics['failed_updates'] /
                                self.update_metrics['total_updates'])
            self.update_metrics['success_rate'] = success_rate

# Interface utilitaire
async def create_state_tracker() -> MultiDimensionalStateTracker:
    """Crée et initialise le tracker d'état multi-dimensionnel"""
    tracker = MultiDimensionalStateTracker()

    # Initialisation avec des états par défaut
    initial_state = GameState(
        position=(4, -13),
        map_id="incarnam",
        level=1,
        health_percent=100.0,
        mana_percent=100.0,
        energy_percent=100.0
    )

    await tracker.update_immediate_state(initial_state)

    # Configuration tactique initiale
    tactical_updates = {
        'current_activity': 'initialization',
        'immediate_goals': ['setup_complete'],
        'efficiency_score': 1.0
    }

    await tracker.update_tactical_context(tactical_updates)

    # Objectifs stratégiques initiaux
    strategic_updates = {
        'daily_goals': [
            {'id': 'learning', 'name': 'Apprentissage système', 'priority': 1, 'completed': False},
            {'id': 'efficiency', 'name': 'Optimisation efficacité', 'priority': 2, 'completed': False}
        ],
        'time_allocations': {
            'learning': timedelta(hours=2),
            'efficiency': timedelta(hours=1)
        }
    }

    await tracker.update_strategic_planning(strategic_updates)

    return tracker

# Interface CLI pour tests
async def main():
    """Test du State Tracker"""
    print("Test Multi-Dimensional State Tracker...")

    # Création du tracker
    tracker = await create_state_tracker()

    print(f"Tracker initialisé")

    # Démarrage du suivi
    if await tracker.start_tracking():
        print("✅ Suivi démarré")

        # Simulation de mises à jour d'état
        await asyncio.sleep(2)

        # Mise à jour d'état immédiat
        new_state = GameState(
            position=(5, -13),
            map_id="incarnam",
            level=1,
            health_percent=95.0,
            in_combat=True
        )

        await tracker.update_immediate_state(new_state)
        print("État immédiat mis à jour")

        # Mise à jour tactique
        tactical_updates = {
            'current_activity': 'combat',
            'efficiency_score': 0.9,
            'actions_per_minute': 45.0
        }

        await tracker.update_tactical_context(tactical_updates)
        print("Contexte tactique mis à jour")

        # Attente pour voir les mises à jour automatiques
        await asyncio.sleep(3)

        # Résumé de l'état
        summary = tracker.get_state_summary()
        print(f"Résumé de l'état:")
        print(f"  - Santé immédiate: {summary['overall_health']['immediate_health']}")
        print(f"  - Efficacité tactique: {summary['overall_health']['tactical_efficiency']:.2f}")
        print(f"  - Progression stratégique: {summary['overall_health']['strategic_progress']:.2f}")

        # Test de prédiction
        prediction = tracker.predict_state_evolution(StateLevel.IMMEDIATE, timedelta(minutes=5))
        print(f"Prédiction (confiance {prediction.get('confidence', 0):.2f})")

        # Arrêt du suivi
        await tracker.stop_tracking()
        print("✅ Suivi arrêté")

    else:
        print("❌ Échec démarrage suivi")

    print("Test State Tracker terminé !")

if __name__ == "__main__":
    asyncio.run(main())