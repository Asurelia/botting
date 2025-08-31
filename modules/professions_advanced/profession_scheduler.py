"""
Planificateur de sessions métiers avancé pour DOFUS.
Système de planification optimale avec gestion multi-personnages et synchronisation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import json
import math
import heapq
from pathlib import Path
from collections import defaultdict

from .advanced_farming import AdvancedFarmer, FarmingStrategy
from .craft_automation import CraftAutomation, CraftPriority
from .profession_optimizer import ProfessionOptimizer, OptimizationGoal
from .resource_predictor import ResourcePredictor, PredictionResult


class SchedulePriority(Enum):
    """Priorité des tâches planifiées"""
    CRITICAL = 1    # Doit être fait immédiatement
    HIGH = 2        # Haute priorité
    MEDIUM = 3      # Priorité normale
    LOW = 4         # Peut attendre
    BACKGROUND = 5  # Tâche de fond


class SessionType(Enum):
    """Types de sessions métiers"""
    FARMING = "farming"
    CRAFTING = "crafting"
    LEVELING = "leveling"
    MARKET_TRADE = "market_trade"
    RESOURCE_GATHERING = "resource_gathering"
    MULTI_PROFESSION = "multi_profession"
    MAINTENANCE = "maintenance"


class CharacterStatus(Enum):
    """Statut des personnages"""
    AVAILABLE = "disponible"
    BUSY = "occupé"
    RESTING = "repos"
    OFFLINE = "hors_ligne"
    MAINTENANCE = "maintenance"


@dataclass
class Character:
    """Données d'un personnage"""
    id: str
    name: str
    level: int
    professions: Dict[str, int]  # profession -> niveau
    current_map: Tuple[int, int]
    available_time_slots: List[Tuple[datetime, datetime]]
    status: CharacterStatus = CharacterStatus.AVAILABLE
    fatigue_level: float = 0.0  # 0-100
    inventory_space: int = 100
    current_kamas: int = 0
    specialization: List[str] = field(default_factory=list)  # Spécialisations
    efficiency_bonus: Dict[str, float] = field(default_factory=dict)
    last_activity: Optional[datetime] = None
    
    def is_available_at(self, time: datetime) -> bool:
        """Vérifie si le personnage est disponible à un moment donné"""
        for start_time, end_time in self.available_time_slots:
            if start_time <= time <= end_time:
                return True
        return False
    
    def get_profession_level(self, profession: str) -> int:
        """Retourne le niveau d'une profession"""
        return self.professions.get(profession, 0)


@dataclass
class ScheduledTask:
    """Tâche planifiée"""
    id: str
    character_id: str
    session_type: SessionType
    profession: str
    specific_action: str
    scheduled_start: datetime
    estimated_duration: timedelta
    priority: SchedulePriority
    prerequisites: List[str] = field(default_factory=list)  # IDs des tâches prérequises
    expected_rewards: Dict[str, float] = field(default_factory=dict)  # XP, Kamas, etc.
    resource_requirements: Dict[str, int] = field(default_factory=dict)
    success_probability: float = 1.0
    urgency_factor: float = 1.0  # Facteur d'urgence (marché, événement, etc.)
    
    @property
    def scheduled_end(self) -> datetime:
        return self.scheduled_start + self.estimated_duration
    
    @property
    def priority_score(self) -> float:
        """Score de priorité calculé"""
        base_score = 6 - self.priority.value  # Plus faible value = plus haute priorité
        
        # Bonus selon les récompenses attendues
        xp_bonus = self.expected_rewards.get('xp', 0) / 1000
        kamas_bonus = self.expected_rewards.get('kamas', 0) / 10000
        
        # Facteur d'urgence
        urgency_bonus = (self.urgency_factor - 1.0) * 2
        
        return base_score + xp_bonus + kamas_bonus + urgency_bonus


@dataclass
class Schedule:
    """Planning complet"""
    character_id: str
    tasks: List[ScheduledTask] = field(default_factory=list)
    start_date: datetime = field(default_factory=datetime.now)
    end_date: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=1))
    total_expected_xp: float = 0.0
    total_expected_kamas: float = 0.0
    efficiency_score: float = 0.0
    conflicts: List[str] = field(default_factory=list)  # Conflits détectés
    
    def add_task(self, task: ScheduledTask):
        """Ajoute une tâche au planning"""
        self.tasks.append(task)
        self._update_totals()
        self._sort_tasks()
    
    def _update_totals(self):
        """Met à jour les totaux attendus"""
        self.total_expected_xp = sum(task.expected_rewards.get('xp', 0) for task in self.tasks)
        self.total_expected_kamas = sum(task.expected_rewards.get('kamas', 0) for task in self.tasks)
    
    def _sort_tasks(self):
        """Trie les tâches par heure de début"""
        self.tasks.sort(key=lambda t: t.scheduled_start)
    
    def get_tasks_for_period(self, start: datetime, end: datetime) -> List[ScheduledTask]:
        """Retourne les tâches pour une période donnée"""
        return [
            task for task in self.tasks
            if not (task.scheduled_end < start or task.scheduled_start > end)
        ]


@dataclass
class SchedulingConstraint:
    """Contrainte de planification"""
    type: str  # "time", "resource", "profession", "character", "dependency"
    parameter: str
    value: Any
    operator: str = "="
    weight: float = 1.0  # Importance de la contrainte
    
    def is_violated_by(self, task: ScheduledTask, schedule: Schedule) -> bool:
        """Vérifie si une tâche viole cette contrainte"""
        if self.type == "time":
            if self.parameter == "max_duration":
                return task.estimated_duration.total_seconds() > self.value
            elif self.parameter == "working_hours":
                start_hour = task.scheduled_start.hour
                end_hour = task.scheduled_end.hour
                min_hour, max_hour = self.value
                return not (min_hour <= start_hour <= max_hour and min_hour <= end_hour <= max_hour)
        
        elif self.type == "resource":
            if self.parameter in task.resource_requirements:
                required = task.resource_requirements[self.parameter]
                return required > self.value
        
        return False


class ProfessionScheduler:
    """
    Planificateur avancé de sessions métiers pour DOFUS.
    Optimise la planification multi-personnages avec contraintes et dépendances.
    """
    
    def __init__(self,
                 optimizer: ProfessionOptimizer,
                 predictor: ResourcePredictor,
                 advanced_farmer: AdvancedFarmer = None,
                 craft_automation: CraftAutomation = None):
        
        self.optimizer = optimizer
        self.predictor = predictor
        self.advanced_farmer = advanced_farmer
        self.craft_automation = craft_automation
        
        # Gestion des personnages
        self.characters: Dict[str, Character] = {}
        self.schedules: Dict[str, Schedule] = {}
        
        # Contraintes globales
        self.global_constraints: List[SchedulingConstraint] = []
        
        # Queue de tâches prioritaires
        self.priority_queue: List[Tuple[float, ScheduledTask]] = []
        
        # Historique et métriques
        self.completed_tasks: List[ScheduledTask] = []
        self.scheduling_metrics = {
            'total_schedules_created': 0,
            'average_efficiency': 0.0,
            'task_completion_rate': 0.0,
            'resource_utilization': 0.0,
            'multi_character_synergy': 0.0
        }
        
        self._setup_default_constraints()
        print("📅 ProfessionScheduler initialisé")
    
    def _setup_default_constraints(self):
        """Configure les contraintes par défaut"""
        default_constraints = [
            SchedulingConstraint("time", "working_hours", (6, 23), weight=0.8),  # 6h-23h
            SchedulingConstraint("time", "max_session_hours", 6, "<=", weight=0.9),
            SchedulingConstraint("resource", "max_budget", 1000000, "<=", weight=0.7),
            SchedulingConstraint("character", "max_fatigue", 80, "<=", weight=0.8)
        ]
        self.global_constraints.extend(default_constraints)
    
    def add_character(self, character: Character):
        """Ajoute un personnage au système"""
        self.characters[character.id] = character
        self.schedules[character.id] = Schedule(character_id=character.id)
        print(f"👤 Personnage {character.name} ajouté (Niveau {character.level})")
    
    async def create_optimal_schedule(self,
                                    character_ids: List[str] = None,
                                    time_horizon_hours: int = 24,
                                    optimization_goal: OptimizationGoal = OptimizationGoal.BALANCED,
                                    constraints: List[SchedulingConstraint] = None) -> Dict[str, Schedule]:
        """
        Crée un planning optimal pour les personnages spécifiés.
        
        Args:
            character_ids: IDs des personnages (None = tous)
            time_horizon_hours: Horizon de planification en heures
            optimization_goal: Objectif d'optimisation
            constraints: Contraintes supplémentaires
            
        Returns:
            Dict[character_id, Schedule] des plannings optimisés
        """
        print(f"📋 Création planning optimal - Horizon: {time_horizon_hours}h, Objectif: {optimization_goal.value}")
        
        if not character_ids:
            character_ids = list(self.characters.keys())
        
        if not character_ids:
            print("⚠️ Aucun personnage disponible")
            return {}
        
        try:
            # Collecte des données actuelles
            current_state = await self._analyze_current_state(character_ids)
            
            # Génération des opportunités
            opportunities = await self._identify_opportunities(current_state, time_horizon_hours)
            
            # Optimisation multi-personnages
            optimized_assignments = await self._optimize_multi_character_assignment(
                character_ids, opportunities, optimization_goal, time_horizon_hours
            )
            
            # Création des plannings individuels
            new_schedules = {}
            for character_id in character_ids:
                character_schedule = await self._create_character_schedule(
                    character_id, optimized_assignments.get(character_id, []), 
                    time_horizon_hours, constraints
                )
                new_schedules[character_id] = character_schedule
                self.schedules[character_id] = character_schedule
            
            # Validation et résolution de conflits
            await self._resolve_schedule_conflicts(new_schedules)
            
            # Mise à jour des métriques
            self._update_scheduling_metrics(new_schedules)
            
            print(f"✅ Planning créé pour {len(new_schedules)} personnages")
            return new_schedules
            
        except Exception as e:
            print(f"❌ Erreur création planning: {e}")
            return {}
    
    async def _analyze_current_state(self, character_ids: List[str]) -> Dict[str, Any]:
        """Analyse l'état actuel des personnages et ressources"""
        current_state = {
            'characters': {},
            'global_resources': {},
            'market_conditions': {},
            'time_context': {
                'current_time': datetime.now(),
                'day_of_week': datetime.now().weekday(),
                'hour_of_day': datetime.now().hour
            }
        }
        
        # Analyse des personnages
        for char_id in character_ids:
            character = self.characters[char_id]
            current_state['characters'][char_id] = {
                'level': character.level,
                'professions': character.professions.copy(),
                'current_map': character.current_map,
                'fatigue_level': character.fatigue_level,
                'available_time_hours': self._calculate_available_time(character),
                'specialization': character.specialization.copy(),
                'efficiency_bonus': character.efficiency_bonus.copy(),
                'status': character.status
            }
        
        # État des ressources globales (simulation)
        current_state['global_resources'] = {
            'kamas_pool': sum(char.current_kamas for char in self.characters.values()),
            'shared_materials': {},  # Matériaux partagés entre personnages
            'guild_resources': {}   # Ressources de guilde
        }
        
        return current_state
    
    def _calculate_available_time(self, character: Character) -> float:
        """Calcule le temps disponible d'un personnage en heures"""
        now = datetime.now()
        end_of_day = now + timedelta(hours=24)
        
        total_available = 0.0
        for start_time, end_time in character.available_time_slots:
            # Intersection avec la période considérée
            actual_start = max(start_time, now)
            actual_end = min(end_time, end_of_day)
            
            if actual_start < actual_end:
                total_available += (actual_end - actual_start).total_seconds() / 3600
        
        return total_available
    
    async def _identify_opportunities(self, current_state: Dict[str, Any], time_horizon_hours: int) -> List[Dict[str, Any]]:
        """Identifie les opportunités de métiers selon l'état actuel"""
        opportunities = []
        
        current_time = current_state['time_context']['current_time']
        end_time = current_time + timedelta(hours=time_horizon_hours)
        
        # Opportunités de farming avec prédictions
        farming_opportunities = await self._identify_farming_opportunities(current_state, end_time)
        opportunities.extend(farming_opportunities)
        
        # Opportunités de craft avec analyse de marché
        crafting_opportunities = await self._identify_crafting_opportunities(current_state)
        opportunities.extend(crafting_opportunities)
        
        # Opportunités de leveling optimales
        leveling_opportunities = await self._identify_leveling_opportunities(current_state)
        opportunities.extend(leveling_opportunities)
        
        # Opportunités de synergies multi-personnages
        synergy_opportunities = await self._identify_synergy_opportunities(current_state)
        opportunities.extend(synergy_opportunities)
        
        print(f"🎯 {len(opportunities)} opportunités identifiées")
        return opportunities
    
    async def _identify_farming_opportunities(self, current_state: Dict[str, Any], end_time: datetime) -> List[Dict[str, Any]]:
        """Identifie les opportunités de farming avec prédictions"""
        opportunities = []
        
        # Ressources principales à considérer
        resources_to_check = ['wheat', 'barley', 'rice', 'ash_wood', 'oak_wood', 'iron_ore', 'copper_ore']
        
        for resource_id in resources_to_check:
            # Prédiction des respawns pour les prochaines heures
            current_time = datetime.now()
            for hour_offset in range(0, min(12, int((end_time - current_time).total_seconds() / 3600))):
                check_time = current_time + timedelta(hours=hour_offset)
                
                prediction = await self.predictor.predict_respawn_time(resource_id, check_time)
                
                if prediction.is_reliable():
                    opportunity = {
                        'type': 'farming',
                        'resource_id': resource_id,
                        'optimal_time': prediction.predicted_respawn_time,
                        'confidence': prediction.confidence_score,
                        'estimated_duration_hours': 1.0,
                        'expected_xp_per_hour': self._estimate_farming_xp(resource_id),
                        'expected_kamas_per_hour': self._estimate_farming_kamas(resource_id),
                        'profession_required': self._get_profession_for_resource(resource_id),
                        'min_level_required': self._get_min_level_for_resource(resource_id),
                        'competition_expected': prediction.factors_influence.get('competition', 0.5),
                        'priority_score': prediction.confidence_score + (100 - hour_offset * 5)  # Priorité temporelle
                    }
                    opportunities.append(opportunity)
        
        return opportunities
    
    async def _identify_crafting_opportunities(self, current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifie les opportunités de craft rentables"""
        opportunities = []
        
        # Simule l'analyse de marché pour différents crafts
        profitable_crafts = [
            {
                'craft_id': 'healing_potion',
                'profit_per_hour': 15000,
                'xp_per_hour': 400,
                'profession': 'alchemist',
                'min_level': 25,
                'investment_needed': 5000,
                'market_demand': 'high'
            },
            {
                'craft_id': 'strength_potion',
                'profit_per_hour': 22000,
                'xp_per_hour': 350,
                'profession': 'alchemist',
                'min_level': 50,
                'investment_needed': 8000,
                'market_demand': 'medium'
            },
            {
                'craft_id': 'ash_plank',
                'profit_per_hour': 18000,
                'xp_per_hour': 320,
                'profession': 'lumberjack',
                'min_level': 40,
                'investment_needed': 3000,
                'market_demand': 'high'
            }
        ]
        
        for craft_data in profitable_crafts:
            opportunity = {
                'type': 'crafting',
                'craft_id': craft_data['craft_id'],
                'optimal_time': datetime.now() + timedelta(minutes=30),  # Disponible bientôt
                'estimated_duration_hours': 2.0,
                'expected_xp_per_hour': craft_data['xp_per_hour'],
                'expected_kamas_per_hour': craft_data['profit_per_hour'],
                'profession_required': craft_data['profession'],
                'min_level_required': craft_data['min_level'],
                'investment_needed': craft_data['investment_needed'],
                'market_demand': craft_data['market_demand'],
                'priority_score': craft_data['profit_per_hour'] / 1000  # Score basé sur le profit
            }
            opportunities.append(opportunity)
        
        return opportunities
    
    async def _identify_leveling_opportunities(self, current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifie les opportunités de leveling optimales"""
        opportunities = []
        
        # Pour chaque personnage, identifier les métiers à améliorer
        for char_id, char_data in current_state['characters'].items():
            for profession, level in char_data['professions'].items():
                if level < 200:  # Niveau maximum
                    # Calcul de la priorité de leveling
                    xp_needed = self._calculate_xp_to_next_level(level)
                    leveling_efficiency = self._calculate_leveling_efficiency(profession, level)
                    
                    opportunity = {
                        'type': 'leveling',
                        'character_id': char_id,
                        'profession': profession,
                        'current_level': level,
                        'optimal_time': datetime.now(),
                        'estimated_duration_hours': max(1.0, xp_needed / leveling_efficiency / 3600),
                        'expected_xp_per_hour': leveling_efficiency,
                        'expected_kamas_per_hour': 5000,  # Leveling moins rentable
                        'profession_required': profession,
                        'min_level_required': level,
                        'xp_to_next_level': xp_needed,
                        'priority_score': (200 - level) * leveling_efficiency / 10000  # Priorité selon niveau
                    }
                    opportunities.append(opportunity)
        
        return opportunities
    
    async def _identify_synergy_opportunities(self, current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifie les opportunités de synergie multi-personnages"""
        opportunities = []
        
        # Synergies possibles entre métiers
        synergies = [
            {
                'type': 'resource_chain',
                'provider_profession': 'farmer',
                'consumer_profession': 'alchemist',
                'resource': 'wheat',
                'synergy_bonus': 1.5,  # 50% bonus d'efficacité
                'description': 'Farming -> Alchimie'
            },
            {
                'type': 'resource_chain',
                'provider_profession': 'lumberjack',
                'consumer_profession': 'craftsman',
                'resource': 'wood',
                'synergy_bonus': 1.3,
                'description': 'Bucheronnage -> Artisanat'
            }
        ]
        
        # Vérifier quelles synergies sont possibles avec les personnages disponibles
        character_professions = {}
        for char_id, char_data in current_state['characters'].items():
            for profession, level in char_data['professions'].items():
                if level >= 20:  # Niveau minimum pour synergie
                    if profession not in character_professions:
                        character_professions[profession] = []
                    character_professions[profession].append((char_id, level))
        
        for synergy in synergies:
            provider_prof = synergy['provider_profession']
            consumer_prof = synergy['consumer_profession']
            
            if provider_prof in character_professions and consumer_prof in character_professions:
                # Trouver les meilleures combinaisons
                providers = character_professions[provider_prof]
                consumers = character_professions[consumer_prof]
                
                for provider_char, provider_level in providers[:2]:  # Max 2 providers
                    for consumer_char, consumer_level in consumers[:2]:  # Max 2 consumers
                        if provider_char != consumer_char:  # Personnages différents
                            
                            opportunity = {
                                'type': 'synergy',
                                'synergy_type': synergy['type'],
                                'provider_character': provider_char,
                                'consumer_character': consumer_char,
                                'provider_profession': provider_prof,
                                'consumer_profession': consumer_prof,
                                'resource': synergy['resource'],
                                'synergy_bonus': synergy['synergy_bonus'],
                                'optimal_time': datetime.now() + timedelta(minutes=15),
                                'estimated_duration_hours': 3.0,
                                'expected_xp_per_hour': 800 * synergy['synergy_bonus'],
                                'expected_kamas_per_hour': 20000 * synergy['synergy_bonus'],
                                'priority_score': synergy['synergy_bonus'] * (provider_level + consumer_level) / 10,
                                'description': synergy['description']
                            }
                            opportunities.append(opportunity)
        
        return opportunities
    
    async def _optimize_multi_character_assignment(self,
                                                 character_ids: List[str],
                                                 opportunities: List[Dict[str, Any]],
                                                 goal: OptimizationGoal,
                                                 time_horizon_hours: int) -> Dict[str, List[Dict[str, Any]]]:
        """Optimise l'assignation des opportunités aux personnages"""
        
        assignments = {char_id: [] for char_id in character_ids}
        
        # Trier les opportunités par priorité
        sorted_opportunities = sorted(opportunities, key=lambda o: o.get('priority_score', 0), reverse=True)
        
        # Assignation gloutonne optimisée
        for opportunity in sorted_opportunities:
            best_character = await self._find_best_character_for_opportunity(
                opportunity, character_ids, assignments, goal
            )
            
            if best_character:
                # Vérifier les conflits temporels
                if not self._has_time_conflict(opportunity, assignments[best_character]):
                    assignments[best_character].append(opportunity)
        
        # Optimisation fine par algorithme génétique simplifié
        assignments = await self._fine_tune_assignments(assignments, goal)
        
        return assignments
    
    async def _find_best_character_for_opportunity(self,
                                                 opportunity: Dict[str, Any],
                                                 character_ids: List[str],
                                                 current_assignments: Dict[str, List[Dict[str, Any]]],
                                                 goal: OptimizationGoal) -> Optional[str]:
        """Trouve le meilleur personnage pour une opportunité"""
        
        best_character = None
        best_score = -1
        
        for char_id in character_ids:
            character = self.characters[char_id]
            
            # Vérifications de base
            if not self._character_can_do_opportunity(character, opportunity):
                continue
            
            # Calcul du score d'adéquation
            score = self._calculate_character_opportunity_score(
                character, opportunity, current_assignments[char_id], goal
            )
            
            if score > best_score:
                best_score = score
                best_character = char_id
        
        return best_character
    
    def _character_can_do_opportunity(self, character: Character, opportunity: Dict[str, Any]) -> bool:
        """Vérifie si un personnage peut faire une opportunité"""
        
        # Vérification profession et niveau
        required_prof = opportunity.get('profession_required')
        required_level = opportunity.get('min_level_required', 1)
        
        if required_prof and character.get_profession_level(required_prof) < required_level:
            return False
        
        # Vérification fatigue
        if character.fatigue_level > 80:
            return False
        
        # Vérification investissement si nécessaire
        investment_needed = opportunity.get('investment_needed', 0)
        if investment_needed > character.current_kamas:
            return False
        
        return True
    
    def _calculate_character_opportunity_score(self,
                                             character: Character,
                                             opportunity: Dict[str, Any],
                                             current_assignments: List[Dict[str, Any]],
                                             goal: OptimizationGoal) -> float:
        """Calcule le score d'adéquation personnage-opportunité"""
        
        score = 0.0
        
        # Score de base selon l'objectif
        if goal == OptimizationGoal.MAX_XP:
            score += opportunity.get('expected_xp_per_hour', 0) / 100
        elif goal == OptimizationGoal.MAX_KAMAS:
            score += opportunity.get('expected_kamas_per_hour', 0) / 1000
        else:  # BALANCED
            score += (opportunity.get('expected_xp_per_hour', 0) / 100 + 
                     opportunity.get('expected_kamas_per_hour', 0) / 1000) * 0.5
        
        # Bonus spécialisation
        required_prof = opportunity.get('profession_required')
        if required_prof in character.specialization:
            score *= 1.5
        
        # Bonus efficacité personnage
        if required_prof in character.efficiency_bonus:
            score *= character.efficiency_bonus[required_prof]
        
        # Pénalité fatigue
        fatigue_penalty = character.fatigue_level / 100
        score *= (1.0 - fatigue_penalty * 0.3)
        
        # Bonus faible charge de travail
        current_load = len(current_assignments)
        if current_load < 3:  # Moins de 3 tâches assignées
            score *= 1.2
        
        return score
    
    def _has_time_conflict(self, opportunity: Dict[str, Any], assignments: List[Dict[str, Any]]) -> bool:
        """Vérifie s'il y a conflit temporel avec les assignations existantes"""
        
        opp_start = opportunity.get('optimal_time', datetime.now())
        opp_duration = timedelta(hours=opportunity.get('estimated_duration_hours', 1))
        opp_end = opp_start + opp_duration
        
        for assignment in assignments:
            assign_start = assignment.get('optimal_time', datetime.now())
            assign_duration = timedelta(hours=assignment.get('estimated_duration_hours', 1))
            assign_end = assign_start + assign_duration
            
            # Vérification chevauchement
            if not (opp_end <= assign_start or opp_start >= assign_end):
                return True
        
        return False
    
    async def _fine_tune_assignments(self, assignments: Dict[str, List[Dict[str, Any]]], goal: OptimizationGoal) -> Dict[str, List[Dict[str, Any]]]:
        """Affine les assignations par optimisation locale"""
        
        # Optimisation simple par échange local
        improved = True
        iterations = 0
        max_iterations = 50
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Tenter des échanges entre personnages
            char_ids = list(assignments.keys())
            for i, char1 in enumerate(char_ids):
                for char2 in char_ids[i+1:]:
                    
                    if not assignments[char1] or not assignments[char2]:
                        continue
                    
                    # Échanger la dernière tâche de chaque personnage
                    task1 = assignments[char1][-1]
                    task2 = assignments[char2][-1]
                    
                    # Calculer le score avant échange
                    score_before = self._calculate_assignment_score(assignments, goal)
                    
                    # Échanger temporairement
                    assignments[char1][-1] = task2
                    assignments[char2][-1] = task1
                    
                    # Calculer le score après échange
                    score_after = self._calculate_assignment_score(assignments, goal)
                    
                    if score_after > score_before:
                        improved = True
                        break
                    else:
                        # Annuler l'échange
                        assignments[char1][-1] = task1
                        assignments[char2][-1] = task2
                
                if improved:
                    break
        
        return assignments
    
    def _calculate_assignment_score(self, assignments: Dict[str, List[Dict[str, Any]]], goal: OptimizationGoal) -> float:
        """Calcule le score global d'un ensemble d'assignations"""
        total_score = 0.0
        
        for char_id, opportunities in assignments.items():
            for opportunity in opportunities:
                if goal == OptimizationGoal.MAX_XP:
                    total_score += opportunity.get('expected_xp_per_hour', 0)
                elif goal == OptimizationGoal.MAX_KAMAS:
                    total_score += opportunity.get('expected_kamas_per_hour', 0)
                else:  # BALANCED
                    total_score += (opportunity.get('expected_xp_per_hour', 0) + 
                                  opportunity.get('expected_kamas_per_hour', 0) / 10) * 0.5
        
        return total_score
    
    async def _create_character_schedule(self,
                                       character_id: str,
                                       assigned_opportunities: List[Dict[str, Any]],
                                       time_horizon_hours: int,
                                       constraints: List[SchedulingConstraint] = None) -> Schedule:
        """Crée le planning détaillé pour un personnage"""
        
        character = self.characters[character_id]
        schedule = Schedule(
            character_id=character_id,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(hours=time_horizon_hours)
        )
        
        # Convertir les opportunités en tâches planifiées
        for i, opportunity in enumerate(assigned_opportunities):
            task = ScheduledTask(
                id=f"{character_id}_task_{i}",
                character_id=character_id,
                session_type=SessionType(opportunity.get('type', 'farming')),
                profession=opportunity.get('profession_required', 'unknown'),
                specific_action=opportunity.get('description', f"Action {opportunity.get('type', 'unknown')}"),
                scheduled_start=opportunity.get('optimal_time', datetime.now()),
                estimated_duration=timedelta(hours=opportunity.get('estimated_duration_hours', 1)),
                priority=self._determine_task_priority(opportunity),
                expected_rewards={
                    'xp': opportunity.get('expected_xp_per_hour', 0) * opportunity.get('estimated_duration_hours', 1),
                    'kamas': opportunity.get('expected_kamas_per_hour', 0) * opportunity.get('estimated_duration_hours', 1)
                },
                resource_requirements={
                    'investment': opportunity.get('investment_needed', 0)
                },
                success_probability=opportunity.get('confidence', 100) / 100,
                urgency_factor=opportunity.get('urgency_factor', 1.0)
            )
            
            schedule.add_task(task)
        
        # Validation des contraintes
        if constraints:
            await self._validate_schedule_constraints(schedule, constraints)
        
        return schedule
    
    def _determine_task_priority(self, opportunity: Dict[str, Any]) -> SchedulePriority:
        """Détermine la priorité d'une tâche selon l'opportunité"""
        priority_score = opportunity.get('priority_score', 50)
        
        if priority_score >= 80:
            return SchedulePriority.CRITICAL
        elif priority_score >= 60:
            return SchedulePriority.HIGH
        elif priority_score >= 40:
            return SchedulePriority.MEDIUM
        elif priority_score >= 20:
            return SchedulePriority.LOW
        else:
            return SchedulePriority.BACKGROUND
    
    async def _resolve_schedule_conflicts(self, schedules: Dict[str, Schedule]):
        """Résout les conflits dans les plannings"""
        
        for character_id, schedule in schedules.items():
            conflicts = []
            
            # Vérification des chevauchements temporels
            for i, task1 in enumerate(schedule.tasks):
                for j, task2 in enumerate(schedule.tasks[i+1:], i+1):
                    if self._tasks_overlap(task1, task2):
                        conflicts.append(f"Conflit temporel entre tâches {task1.id} et {task2.id}")
                        
                        # Résolution: décaler la tâche de priorité plus faible
                        if task1.priority.value > task2.priority.value:
                            task1.scheduled_start = task2.scheduled_end + timedelta(minutes=5)
                        else:
                            task2.scheduled_start = task1.scheduled_end + timedelta(minutes=5)
            
            # Vérification des ressources
            total_investment = sum(task.resource_requirements.get('investment', 0) for task in schedule.tasks)
            character = self.characters[character_id]
            
            if total_investment > character.current_kamas:
                conflicts.append(f"Budget insuffisant: {total_investment} > {character.current_kamas}")
                
                # Résolution: supprimer les tâches les moins prioritaires
                schedule.tasks.sort(key=lambda t: t.priority.value)
                running_cost = 0
                valid_tasks = []
                
                for task in schedule.tasks:
                    task_cost = task.resource_requirements.get('investment', 0)
                    if running_cost + task_cost <= character.current_kamas:
                        valid_tasks.append(task)
                        running_cost += task_cost
                
                schedule.tasks = valid_tasks
            
            schedule.conflicts = conflicts
            if conflicts:
                print(f"⚠️ {len(conflicts)} conflits résolus pour {character_id}")
    
    def _tasks_overlap(self, task1: ScheduledTask, task2: ScheduledTask) -> bool:
        """Vérifie si deux tâches se chevauchent temporellement"""
        return not (task1.scheduled_end <= task2.scheduled_start or 
                   task1.scheduled_start >= task2.scheduled_end)
    
    async def _validate_schedule_constraints(self, schedule: Schedule, constraints: List[SchedulingConstraint]):
        """Valide un planning contre des contraintes"""
        
        violations = []
        
        for constraint in constraints:
            for task in schedule.tasks:
                if constraint.is_violated_by(task, schedule):
                    violations.append(f"Contrainte {constraint.type}:{constraint.parameter} violée par {task.id}")
        
        if violations:
            print(f"⚠️ {len(violations)} violations de contraintes détectées")
            schedule.conflicts.extend(violations)
    
    def _update_scheduling_metrics(self, schedules: Dict[str, Schedule]):
        """Met à jour les métriques de planification"""
        
        self.scheduling_metrics['total_schedules_created'] += 1
        
        total_efficiency = 0.0
        total_tasks = 0
        total_expected_value = 0.0
        
        for schedule in schedules.values():
            # Calcul de l'efficacité du planning
            if schedule.estimated_duration_hours > 0:
                efficiency = (schedule.total_expected_xp + schedule.total_expected_kamas / 100) / schedule.estimated_duration_hours
                total_efficiency += efficiency
                
            total_tasks += len(schedule.tasks)
            total_expected_value += schedule.total_expected_xp + schedule.total_expected_kamas
        
        if schedules:
            avg_efficiency = total_efficiency / len(schedules)
            self.scheduling_metrics['average_efficiency'] = (
                self.scheduling_metrics['average_efficiency'] * 0.8 + avg_efficiency * 0.2
            )
        
        print(f"📊 Efficacité moyenne: {self.scheduling_metrics['average_efficiency']:.0f}")
    
    # Méthodes utilitaires pour les estimations
    
    def _estimate_farming_xp(self, resource_id: str) -> float:
        """Estime l'XP/heure pour une ressource de farming"""
        xp_rates = {
            'wheat': 420, 'barley': 580, 'rice': 750,
            'ash_wood': 380, 'oak_wood': 520,
            'iron_ore': 450, 'copper_ore': 600
        }
        return xp_rates.get(resource_id, 400)
    
    def _estimate_farming_kamas(self, resource_id: str) -> float:
        """Estime les Kamas/heure pour une ressource de farming"""
        kamas_rates = {
            'wheat': 12000, 'barley': 16000, 'rice': 22000,
            'ash_wood': 14000, 'oak_wood': 19000,
            'iron_ore': 15000, 'copper_ore': 20000
        }
        return kamas_rates.get(resource_id, 12000)
    
    def _get_profession_for_resource(self, resource_id: str) -> str:
        """Retourne la profession nécessaire pour une ressource"""
        profession_map = {
            'wheat': 'farmer', 'barley': 'farmer', 'rice': 'farmer',
            'ash_wood': 'lumberjack', 'oak_wood': 'lumberjack',
            'iron_ore': 'miner', 'copper_ore': 'miner'
        }
        return profession_map.get(resource_id, 'unknown')
    
    def _get_min_level_for_resource(self, resource_id: str) -> int:
        """Retourne le niveau minimum pour une ressource"""
        level_map = {
            'wheat': 1, 'barley': 15, 'rice': 30,
            'ash_wood': 1, 'oak_wood': 20,
            'iron_ore': 1, 'copper_ore': 25
        }
        return level_map.get(resource_id, 1)
    
    def _calculate_xp_to_next_level(self, current_level: int) -> int:
        """Calcule l'XP nécessaire pour le niveau suivant"""
        # Formule approximative DOFUS
        return int((current_level + 1) * 1000 * (1 + current_level * 0.01))
    
    def _calculate_leveling_efficiency(self, profession: str, level: int) -> float:
        """Calcule l'efficacité de leveling XP/seconde"""
        base_rates = {
            'farmer': 0.12, 'lumberjack': 0.10, 'miner': 0.11,
            'alchemist': 0.08, 'fisherman': 0.09, 'hunter': 0.07
        }
        
        base_rate = base_rates.get(profession, 0.08)  # XP/seconde de base
        
        # Malus pour niveaux élevés
        level_penalty = 1.0 - min(0.5, level * 0.002)
        
        return base_rate * level_penalty * 3600  # Conversion en XP/heure
    
    def get_schedule_summary(self, character_id: str) -> Dict[str, Any]:
        """Retourne un résumé du planning d'un personnage"""
        
        if character_id not in self.schedules:
            return {}
        
        schedule = self.schedules[character_id]
        character = self.characters[character_id]
        
        summary = {
            'character_name': character.name,
            'character_level': character.level,
            'schedule_start': schedule.start_date.isoformat(),
            'schedule_end': schedule.end_date.isoformat(),
            'total_tasks': len(schedule.tasks),
            'total_expected_xp': schedule.total_expected_xp,
            'total_expected_kamas': schedule.total_expected_kamas,
            'efficiency_score': schedule.efficiency_score,
            'conflicts_count': len(schedule.conflicts),
            'tasks_by_type': {},
            'tasks_by_priority': {},
            'hourly_breakdown': []
        }
        
        # Répartition par type
        for task in schedule.tasks:
            task_type = task.session_type.value
            if task_type not in summary['tasks_by_type']:
                summary['tasks_by_type'][task_type] = 0
            summary['tasks_by_type'][task_type] += 1
        
        # Répartition par priorité
        for task in schedule.tasks:
            priority = task.priority.name
            if priority not in summary['tasks_by_priority']:
                summary['tasks_by_priority'][priority] = 0
            summary['tasks_by_priority'][priority] += 1
        
        return summary
    
    def export_schedules(self, filepath: str):
        """Exporte tous les plannings vers un fichier"""
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'scheduling_metrics': self.scheduling_metrics,
            'characters': {
                char_id: {
                    'name': char.name,
                    'level': char.level,
                    'professions': char.professions,
                    'status': char.status.value,
                    'fatigue_level': char.fatigue_level
                }
                for char_id, char in self.characters.items()
            },
            'schedules': {}
        }
        
        for char_id, schedule in self.schedules.items():
            export_data['schedules'][char_id] = {
                'summary': self.get_schedule_summary(char_id),
                'detailed_tasks': [
                    {
                        'id': task.id,
                        'session_type': task.session_type.value,
                        'profession': task.profession,
                        'specific_action': task.specific_action,
                        'scheduled_start': task.scheduled_start.isoformat(),
                        'estimated_duration_hours': task.estimated_duration.total_seconds() / 3600,
                        'priority': task.priority.name,
                        'expected_rewards': task.expected_rewards,
                        'success_probability': task.success_probability
                    }
                    for task in schedule.tasks
                ],
                'conflicts': schedule.conflicts
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Plannings exportés vers: {filepath}")


# Exemple d'utilisation
if __name__ == "__main__":
    async def demo_profession_scheduler():
        """Démonstration du système de planification"""
        
        # Simulation des dépendances
        optimizer = None       # ProfessionOptimizer()
        predictor = None       # ResourcePredictor()
        
        scheduler = ProfessionScheduler(optimizer, predictor)
        
        print("🤖 Démonstration ProfessionScheduler")
        
        # Ajout de personnages de test
        char1 = Character(
            id='char1',
            name='MainFarmer',
            level=85,
            professions={'farmer': 85, 'alchemist': 65},
            current_map=(-5, -8),
            available_time_slots=[(datetime.now(), datetime.now() + timedelta(hours=8))],
            current_kamas=500000,
            specialization=['farmer']
        )
        
        char2 = Character(
            id='char2',
            name='CraftMaster',
            level=92,
            professions={'lumberjack': 78, 'craftsman': 82},
            current_map=(2, -15),
            available_time_slots=[(datetime.now(), datetime.now() + timedelta(hours=6))],
            current_kamas=750000,
            specialization=['craftsman']
        )
        
        scheduler.add_character(char1)
        scheduler.add_character(char2)
        
        # Création d'un planning optimisé
        schedules = await scheduler.create_optimal_schedule(
            character_ids=['char1', 'char2'],
            time_horizon_hours=12,
            optimization_goal=OptimizationGoal.BALANCED
        )
        
        # Affichage des résumés
        for char_id, schedule in schedules.items():
            summary = scheduler.get_schedule_summary(char_id)
            print(f"\n📋 Planning {summary['character_name']}:")
            print(f"   Tâches: {summary['total_tasks']}")
            print(f"   XP attendue: {summary['total_expected_xp']:.0f}")
            print(f"   Kamas attendus: {summary['total_expected_kamas']:.0f}")
            print(f"   Conflits: {summary['conflicts_count']}")
        
        # Export des plannings
        scheduler.export_schedules('profession_schedules.json')
    
    # asyncio.run(demo_profession_scheduler())
    print("Module ProfessionScheduler chargé avec succès ✅")