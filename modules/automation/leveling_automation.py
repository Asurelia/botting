"""
Module de leveling automation intelligent.

Bot de leveling avec multiples stratégies adaptatives :
- Stratégies de leveling optimisées par niveau et classe
- Calcul XP/heure multi-critères en temps réel  
- Adaptation automatique selon les conditions
- Gestion intelligente des ressources et pauses
"""

import logging
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LevelingStrategy(Enum):
    """Types de stratégies de leveling."""
    MONSTER_GRINDING = "monster_grinding"       # Farm de monstres
    DUNGEON_FARMING = "dungeon_farming"         # Farm de donjons
    QUEST_CHAIN = "quest_chain"                 # Chaîne de quêtes
    PROFESSION_LEVELING = "profession_leveling"  # Leveling par métiers
    MIXED_APPROACH = "mixed_approach"           # Approche mixte
    ACHIEVEMENT_HUNTING = "achievement_hunting"  # Chasse aux succès
    PVP_LEVELING = "pvp_leveling"              # Leveling PvP


class ResourceType(Enum):
    """Types de ressources à gérer."""
    HEALTH = "health"
    MANA = "mana" 
    ENERGY = "energy"
    PODS = "pods"
    KAMAS = "kamas"
    ITEMS = "items"


class OptimizationCriteria(Enum):
    """Critères d'optimisation du leveling."""
    XP_PER_HOUR = "xp_per_hour"                # XP/heure maximum
    SAFETY_FIRST = "safety_first"              # Sécurité maximale
    RESOURCE_EFFICIENCY = "resource_efficiency" # Efficacité des ressources
    TIME_EFFICIENCY = "time_efficiency"         # Efficacité temporelle
    BALANCED = "balanced"                       # Équilibré


@dataclass
class LevelingTarget:
    """Cible de leveling avec objectifs."""
    target_level: int
    target_xp: Optional[int] = None
    time_limit: Optional[datetime] = None
    preferred_strategy: Optional[LevelingStrategy] = None
    optimization_criteria: OptimizationCriteria = OptimizationCriteria.BALANCED
    min_safety_threshold: float = 0.8          # Seuil de sécurité minimum
    max_resource_usage: float = 0.9            # Usage max des ressources


@dataclass 
class LevelingSession:
    """Session de leveling avec métriques."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    start_level: int = 1
    current_level: int = 1
    start_xp: int = 0
    current_xp: int = 0
    target: Optional[LevelingTarget] = None
    
    # Métriques de performance
    total_xp_gained: int = 0
    monsters_killed: int = 0
    dungeons_completed: int = 0
    quests_completed: int = 0
    deaths: int = 0
    kamas_spent: int = 0
    kamas_earned: int = 0
    
    # Stratégie utilisée
    active_strategy: Optional[LevelingStrategy] = None
    strategy_switches: int = 0
    
    # État de la session
    is_active: bool = True
    pause_time: int = 0
    error_count: int = 0


class LevelingMetrics:
    """Calculateur de métriques de leveling."""
    
    def __init__(self):
        self.xp_history: List[Tuple[datetime, int]] = []
        self.level_history: List[Tuple[datetime, int]] = []
        self.death_times: List[datetime] = []
        self.strategy_performance: Dict[LevelingStrategy, Dict[str, float]] = {}
    
    def record_xp(self, xp: int):
        """Enregistre un point de données XP."""
        self.xp_history.append((datetime.now(), xp))
        
        # Garde seulement les 1000 derniers points
        if len(self.xp_history) > 1000:
            self.xp_history = self.xp_history[-1000:]
    
    def record_level(self, level: int):
        """Enregistre un changement de niveau."""
        self.level_history.append((datetime.now(), level))
    
    def record_death(self):
        """Enregistre une mort."""
        self.death_times.append(datetime.now())
    
    def calculate_xp_per_hour(self, time_window_minutes: int = 60) -> float:
        """Calcule l'XP/heure actuel."""
        if len(self.xp_history) < 2:
            return 0.0
        
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_data = [(t, xp) for t, xp in self.xp_history if t >= cutoff_time]
        
        if len(recent_data) < 2:
            return 0.0
        
        time_diff = (recent_data[-1][0] - recent_data[0][0]).total_seconds() / 3600
        xp_diff = recent_data[-1][1] - recent_data[0][1]
        
        return xp_diff / time_diff if time_diff > 0 else 0.0
    
    def calculate_death_rate(self, time_window_minutes: int = 60) -> float:
        """Calcule le taux de mortalité récent."""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_deaths = [t for t in self.death_times if t >= cutoff_time]
        
        return len(recent_deaths) / (time_window_minutes / 60)
    
    def get_strategy_performance(self, strategy: LevelingStrategy) -> Dict[str, float]:
        """Retourne les performances d'une stratégie."""
        return self.strategy_performance.get(strategy, {
            'xp_per_hour': 0.0,
            'death_rate': 0.0,
            'efficiency_score': 0.0
        })
    
    def update_strategy_performance(self, strategy: LevelingStrategy, 
                                  xp_per_hour: float, death_rate: float):
        """Met à jour les performances d'une stratégie."""
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {}
        
        perf = self.strategy_performance[strategy]
        perf['xp_per_hour'] = xp_per_hour
        perf['death_rate'] = death_rate
        
        # Score d'efficacité composite
        safety_factor = max(0.1, 1.0 - (death_rate * 0.2))
        perf['efficiency_score'] = xp_per_hour * safety_factor


class BaseLevelingStrategy(ABC):
    """Classe de base pour les stratégies de leveling."""
    
    def __init__(self, game_interface=None, navigation_system=None):
        self.game_interface = game_interface
        self.navigation_system = navigation_system
        self.is_active = False
        self.last_xp_check = 0
        self.last_health_check = 100
        self.strategy_metrics = {}
    
    @abstractmethod
    def can_execute(self, character_level: int, character_class: str) -> bool:
        """Vérifie si la stratégie peut être exécutée."""
        pass
    
    @abstractmethod
    def execute_cycle(self, session: LevelingSession) -> bool:
        """Exécute un cycle de la stratégie."""
        pass
    
    @abstractmethod
    def get_estimated_xp_per_hour(self, character_level: int) -> float:
        """Retourne l'estimation XP/heure pour cette stratégie."""
        pass
    
    @abstractmethod
    def get_safety_rating(self, character_level: int) -> float:
        """Retourne le rating de sécurité (0.0 à 1.0)."""
        pass
    
    def prepare_strategy(self, session: LevelingSession):
        """Prépare la stratégie avant exécution."""
        self.is_active = True
        logger.info(f"Préparation stratégie : {self.__class__.__name__}")
    
    def cleanup_strategy(self, session: LevelingSession):
        """Nettoie après l'exécution de la stratégie."""
        self.is_active = False
        logger.info(f"Nettoyage stratégie : {self.__class__.__name__}")


class MonsterGrindingStrategy(BaseLevelingStrategy):
    """Stratégie de farm de monstres."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimal_zones = self._load_optimal_zones()
        self.current_zone = None
        self.monsters_in_zone = []
        self.zone_efficiency = {}
    
    def _load_optimal_zones(self) -> Dict[int, List[Dict[str, Any]]]:
        """Charge les zones optimales par niveau."""
        return {
            1: [{"name": "Plaine de Cania", "monsters": ["Bouftou"], "xp_rate": 5000}],
            10: [{"name": "Forêt Maléfique", "monsters": ["Chafer"], "xp_rate": 15000}],
            20: [{"name": "Cimetière", "monsters": ["Chafer Archer"], "xp_rate": 25000}],
            30: [{"name": "Mine de Dolomite", "monsters": ["Kobolte"], "xp_rate": 40000}],
            50: [{"name": "Donjon Bouftou", "monsters": ["Bouftou Royal"], "xp_rate": 80000}],
            100: [{"name": "Frigost", "monsters": ["Glaçon"], "xp_rate": 200000}],
            150: [{"name": "Dimension Divine", "monsters": ["Divin"], "xp_rate": 500000}]
        }
    
    def can_execute(self, character_level: int, character_class: str) -> bool:
        """Vérifie si le grinding est possible."""
        return character_level >= 1  # Toujours possible
    
    def execute_cycle(self, session: LevelingSession) -> bool:
        """Exécute un cycle de grinding."""
        try:
            # 1. Sélectionner la zone optimale
            if not self.current_zone or self._should_change_zone(session.current_level):
                self._select_optimal_zone(session.current_level)
            
            # 2. Aller à la zone
            if self.navigation_system and self.current_zone:
                if not self.navigation_system.navigate_to(self.current_zone["name"]):
                    logger.error(f"Impossible d'aller à {self.current_zone['name']}")
                    return False
            
            # 3. Combattre les monstres
            monsters_fought = 0
            max_monsters_per_cycle = 10
            
            while monsters_fought < max_monsters_per_cycle:
                if self._find_and_fight_monster():
                    monsters_fought += 1
                    session.monsters_killed += 1
                    
                    # Vérification de la santé
                    if not self._check_and_restore_resources():
                        break
                else:
                    # Pas de monstre trouvé, changer de map
                    if not self._move_to_adjacent_map():
                        break
            
            return monsters_fought > 0
            
        except Exception as e:
            logger.error(f"Erreur dans le cycle de grinding : {e}")
            return False
    
    def get_estimated_xp_per_hour(self, character_level: int) -> float:
        """Retourne l'estimation XP/heure pour le grinding."""
        zone = self._get_optimal_zone_for_level(character_level)
        if zone:
            return zone["xp_rate"]
        return 10000  # Valeur par défaut
    
    def get_safety_rating(self, character_level: int) -> float:
        """Rating de sécurité pour le grinding."""
        # Sécurité moyenne, dépend de la zone
        return 0.7
    
    def _select_optimal_zone(self, level: int):
        """Sélectionne la zone optimale pour le niveau."""
        # Trouve la zone avec le plus haut niveau inférieur ou égal
        best_level = max([lvl for lvl in self.optimal_zones.keys() if lvl <= level], 
                        default=1)
        
        zones = self.optimal_zones[best_level]
        # Sélectionne la meilleure zone selon l'efficacité
        self.current_zone = max(zones, key=lambda z: z["xp_rate"])
        
        logger.info(f"Zone sélectionnée : {self.current_zone['name']}")
    
    def _get_optimal_zone_for_level(self, level: int) -> Optional[Dict[str, Any]]:
        """Retourne la zone optimale pour un niveau."""
        best_level = max([lvl for lvl in self.optimal_zones.keys() if lvl <= level], 
                        default=1)
        zones = self.optimal_zones[best_level]
        return max(zones, key=lambda z: z["xp_rate"]) if zones else None
    
    def _should_change_zone(self, current_level: int) -> bool:
        """Détermine s'il faut changer de zone."""
        if not self.current_zone:
            return True
        
        optimal_zone = self._get_optimal_zone_for_level(current_level)
        return optimal_zone != self.current_zone
    
    def _find_and_fight_monster(self) -> bool:
        """Trouve et combat un monstre."""
        # Simulation du combat
        time.sleep(15)  # Temps moyen d'un combat
        return True
    
    def _check_and_restore_resources(self) -> bool:
        """Vérifie et restaure les ressources."""
        # Simulation de la vérification des ressources
        return True
    
    def _move_to_adjacent_map(self) -> bool:
        """Se déplace vers une map adjacente."""
        time.sleep(5)
        return True


class DungeonFarmingStrategy(BaseLevelingStrategy):
    """Stratégie de farm de donjons."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dungeon_list = self._load_dungeon_data()
        self.current_dungeon = None
        self.completion_times = {}
    
    def _load_dungeon_data(self) -> Dict[str, Dict[str, Any]]:
        """Charge les données des donjons."""
        return {
            "Donjon Bouftou": {
                "min_level": 10,
                "max_level": 30,
                "xp_reward": 15000,
                "duration_minutes": 20,
                "difficulty": 2
            },
            "Donjon Bwork": {
                "min_level": 30,
                "max_level": 60,
                "xp_reward": 50000,
                "duration_minutes": 45,
                "difficulty": 3
            },
            "Donjon Gelée": {
                "min_level": 60,
                "max_level": 100,
                "xp_reward": 120000,
                "duration_minutes": 60,
                "difficulty": 4
            }
        }
    
    def can_execute(self, character_level: int, character_class: str) -> bool:
        """Vérifie si le farming de donjon est possible."""
        return any(
            dungeon["min_level"] <= character_level <= dungeon["max_level"]
            for dungeon in self.dungeon_list.values()
        )
    
    def execute_cycle(self, session: LevelingSession) -> bool:
        """Exécute un cycle de donjon."""
        try:
            # 1. Sélectionner le donjon optimal
            if not self.current_dungeon or not self._is_dungeon_suitable(session.current_level):
                self._select_optimal_dungeon(session.current_level)
            
            if not self.current_dungeon:
                return False
            
            # 2. Aller au donjon
            if self.navigation_system:
                if not self.navigation_system.navigate_to(f"Donjon {self.current_dungeon}"):
                    return False
            
            # 3. Entrer et compléter le donjon
            start_time = time.time()
            success = self._complete_dungeon()
            completion_time = time.time() - start_time
            
            if success:
                session.dungeons_completed += 1
                self.completion_times[self.current_dungeon] = completion_time
                logger.info(f"Donjon complété : {self.current_dungeon} ({completion_time:.1f}s)")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur dans le cycle de donjon : {e}")
            return False
    
    def get_estimated_xp_per_hour(self, character_level: int) -> float:
        """Estimation XP/heure pour les donjons."""
        dungeon = self._get_optimal_dungeon_for_level(character_level)
        if dungeon:
            dungeons_per_hour = 60 / dungeon["duration_minutes"]
            return dungeon["xp_reward"] * dungeons_per_hour
        return 0
    
    def get_safety_rating(self, character_level: int) -> float:
        """Rating de sécurité pour les donjons."""
        return 0.9  # Très sûr car environnement contrôlé
    
    def _select_optimal_dungeon(self, level: int):
        """Sélectionne le donjon optimal."""
        suitable_dungeons = [
            (name, data) for name, data in self.dungeon_list.items()
            if data["min_level"] <= level <= data["max_level"]
        ]
        
        if suitable_dungeons:
            # Sélectionne le donjon avec le meilleur ratio XP/temps
            best_dungeon = max(
                suitable_dungeons,
                key=lambda x: x[1]["xp_reward"] / x[1]["duration_minutes"]
            )
            self.current_dungeon = best_dungeon[0]
            logger.info(f"Donjon sélectionné : {self.current_dungeon}")
    
    def _get_optimal_dungeon_for_level(self, level: int) -> Optional[Dict[str, Any]]:
        """Retourne le donjon optimal pour un niveau."""
        suitable = [
            data for data in self.dungeon_list.values()
            if data["min_level"] <= level <= data["max_level"]
        ]
        
        if suitable:
            return max(suitable, key=lambda x: x["xp_reward"] / x["duration_minutes"])
        return None
    
    def _is_dungeon_suitable(self, level: int) -> bool:
        """Vérifie si le donjon actuel est encore adapté."""
        if not self.current_dungeon:
            return False
        
        dungeon_data = self.dungeon_list[self.current_dungeon]
        return dungeon_data["min_level"] <= level <= dungeon_data["max_level"]
    
    def _complete_dungeon(self) -> bool:
        """Complète un donjon."""
        # Simulation de la complétion de donjon
        dungeon_data = self.dungeon_list[self.current_dungeon]
        time.sleep(dungeon_data["duration_minutes"] * 60)  # Simulation temps réel
        return True


class QuestChainStrategy(BaseLevelingStrategy):
    """Stratégie de chaînes de quêtes."""
    
    def can_execute(self, character_level: int, character_class: str) -> bool:
        return True  # Toujours disponible
    
    def execute_cycle(self, session: LevelingSession) -> bool:
        # Implémentation simplifiée
        time.sleep(300)  # 5 minutes par quête moyenne
        session.quests_completed += 1
        return True
    
    def get_estimated_xp_per_hour(self, character_level: int) -> float:
        return 30000  # XP/heure moyen pour les quêtes
    
    def get_safety_rating(self, character_level: int) -> float:
        return 0.95  # Très sûr


class LevelingAutomation:
    """
    Système principal d'automatisation du leveling.
    
    Gère l'ensemble du processus de leveling automatisé :
    - Sélection de stratégies adaptatives
    - Optimisation XP/heure multi-critères
    - Gestion des ressources et pauses
    - Analyse de performance temps réel
    """
    
    def __init__(self, game_interface=None, navigation_system=None):
        self.game_interface = game_interface
        self.navigation_system = navigation_system
        
        # Stratégies disponibles
        self.strategies = {
            LevelingStrategy.MONSTER_GRINDING: MonsterGrindingStrategy(game_interface, navigation_system),
            LevelingStrategy.DUNGEON_FARMING: DungeonFarmingStrategy(game_interface, navigation_system),
            LevelingStrategy.QUEST_CHAIN: QuestChainStrategy(game_interface, navigation_system)
        }
        
        self.current_session: Optional[LevelingSession] = None
        self.active_strategy: Optional[BaseLevelingStrategy] = None
        self.metrics = LevelingMetrics()
        
        # Configuration
        self.auto_strategy_switching = True
        self.strategy_evaluation_interval = 300  # 5 minutes
        self.resource_check_interval = 60       # 1 minute
        self.max_deaths_per_hour = 3
        self.min_xp_per_hour_threshold = 10000
        
        # État du système
        self.is_running = False
        self.last_strategy_evaluation = 0
        self.last_resource_check = 0
        self.emergency_stop_triggered = False
        
        logger.info("Système de leveling automation initialisé")
    
    def start_leveling_session(self, target: LevelingTarget, 
                             character_level: int, character_xp: int,
                             character_class: str = "Unknown") -> str:
        """Démarre une nouvelle session de leveling."""
        if self.is_running:
            logger.warning("Une session de leveling est déjà en cours")
            return self.current_session.session_id if self.current_session else ""
        
        # Créer une nouvelle session
        session_id = f"session_{int(time.time())}"
        self.current_session = LevelingSession(
            session_id=session_id,
            start_time=datetime.now(),
            start_level=character_level,
            current_level=character_level,
            start_xp=character_xp,
            current_xp=character_xp,
            target=target
        )
        
        # Sélectionner la stratégie initiale
        initial_strategy = self._select_optimal_strategy(character_level, character_class, target)
        if initial_strategy:
            self._switch_strategy(initial_strategy)
            logger.info(f"Session de leveling démarrée : {session_id}")
            logger.info(f"Objectif : Niveau {target.target_level}")
            logger.info(f"Stratégie initiale : {initial_strategy.value}")
            
            self.is_running = True
            self._run_leveling_loop()
            return session_id
        else:
            logger.error("Impossible de sélectionner une stratégie initiale")
            return ""
    
    def stop_leveling_session(self):
        """Arrête la session de leveling en cours."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.current_session:
            self.current_session.end_time = datetime.now()
            self.current_session.is_active = False
            
            if self.active_strategy:
                self.active_strategy.cleanup_strategy(self.current_session)
            
            self._generate_session_report()
            logger.info("Session de leveling arrêtée")
    
    def _run_leveling_loop(self):
        """Boucle principale de leveling."""
        logger.info("Début de la boucle de leveling")
        
        try:
            while self.is_running and self.current_session:
                loop_start = time.time()
                
                # Vérifications d'arrêt d'urgence
                if self._check_emergency_stop():
                    break
                
                # Vérification de l'objectif atteint
                if self._is_target_reached():
                    logger.info("Objectif de leveling atteint !")
                    break
                
                # Vérification et mise à jour des ressources
                if time.time() - self.last_resource_check > self.resource_check_interval:
                    if not self._check_and_manage_resources():
                        continue  # Skip ce cycle si problème de ressources
                    self.last_resource_check = time.time()
                
                # Évaluation et changement de stratégie si nécessaire
                if (self.auto_strategy_switching and 
                    time.time() - self.last_strategy_evaluation > self.strategy_evaluation_interval):
                    self._evaluate_and_switch_strategy()
                    self.last_strategy_evaluation = time.time()
                
                # Exécution du cycle de la stratégie active
                if self.active_strategy:
                    cycle_success = self.active_strategy.execute_cycle(self.current_session)
                    
                    if not cycle_success:
                        logger.warning("Échec du cycle de stratégie")
                        self.current_session.error_count += 1
                        
                        # Trop d'erreurs consécutives = changement de stratégie
                        if self.current_session.error_count > 5:
                            self._handle_repeated_failures()
                
                # Mise à jour des métriques
                self._update_session_metrics()
                
                # Pause de sécurité entre les cycles
                cycle_duration = time.time() - loop_start
                if cycle_duration < 30:  # Cycle minimum de 30 secondes
                    time.sleep(30 - cycle_duration)
                
        except KeyboardInterrupt:
            logger.info("Arrêt demandé par l'utilisateur")
        except Exception as e:
            logger.error(f"Erreur dans la boucle de leveling : {e}")
            self.emergency_stop_triggered = True
        finally:
            self.stop_leveling_session()
    
    def _select_optimal_strategy(self, character_level: int, character_class: str, 
                               target: LevelingTarget) -> Optional[LevelingStrategy]:
        """Sélectionne la stratégie optimale selon les critères."""
        available_strategies = []
        
        for strategy_type, strategy_impl in self.strategies.items():
            if strategy_impl.can_execute(character_level, character_class):
                score = self._calculate_strategy_score(
                    strategy_type, strategy_impl, character_level, target
                )
                available_strategies.append((strategy_type, score))
        
        if available_strategies:
            # Trie par score décroissant
            available_strategies.sort(key=lambda x: x[1], reverse=True)
            return available_strategies[0][0]
        
        return None
    
    def _calculate_strategy_score(self, strategy_type: LevelingStrategy, 
                                strategy_impl: BaseLevelingStrategy,
                                character_level: int, target: LevelingTarget) -> float:
        """Calcule le score d'une stratégie selon les critères d'optimisation."""
        base_score = 0.0
        criteria = target.optimization_criteria
        
        # Métriques de la stratégie
        estimated_xp_per_hour = strategy_impl.get_estimated_xp_per_hour(character_level)
        safety_rating = strategy_impl.get_safety_rating(character_level)
        
        # Performances historiques
        historical_perf = self.metrics.get_strategy_performance(strategy_type)
        actual_xp_per_hour = historical_perf.get('xp_per_hour', estimated_xp_per_hour)
        death_rate = historical_perf.get('death_rate', 0.0)
        
        # Calcul selon les critères
        if criteria == OptimizationCriteria.XP_PER_HOUR:
            base_score = actual_xp_per_hour / 1000  # Normalisation
        
        elif criteria == OptimizationCriteria.SAFETY_FIRST:
            base_score = safety_rating * 100 - (death_rate * 50)
        
        elif criteria == OptimizationCriteria.RESOURCE_EFFICIENCY:
            # Favorise les stratégies peu coûteuses en ressources
            base_score = (actual_xp_per_hour / 1000) * (1.5 - death_rate)
        
        elif criteria == OptimizationCriteria.TIME_EFFICIENCY:
            # Favorise les stratégies rapides pour atteindre l'objectif
            levels_to_go = target.target_level - character_level
            time_to_target = self._estimate_time_to_target(actual_xp_per_hour, levels_to_go)
            base_score = 1000 / max(time_to_target, 1)  # Inverse du temps
        
        else:  # BALANCED
            # Combinaison équilibrée de tous les facteurs
            xp_score = actual_xp_per_hour / 1000
            safety_score = safety_rating * 50
            efficiency_score = historical_perf.get('efficiency_score', xp_score)
            
            base_score = (xp_score + safety_score + efficiency_score) / 3
        
        # Ajustements selon les seuils de sécurité
        if safety_rating < target.min_safety_threshold:
            base_score *= 0.5  # Pénalité pour stratégies non sûres
        
        return base_score
    
    def _estimate_time_to_target(self, xp_per_hour: float, levels_remaining: int) -> float:
        """Estime le temps pour atteindre l'objectif en heures."""
        if xp_per_hour <= 0:
            return float('inf')
        
        # Estimation simplifiée : 100k XP par niveau en moyenne
        estimated_xp_needed = levels_remaining * 100000
        return estimated_xp_needed / xp_per_hour
    
    def _switch_strategy(self, new_strategy: LevelingStrategy):
        """Change la stratégie active."""
        if self.active_strategy:
            self.active_strategy.cleanup_strategy(self.current_session)
        
        self.active_strategy = self.strategies[new_strategy]
        self.active_strategy.prepare_strategy(self.current_session)
        
        if self.current_session:
            self.current_session.active_strategy = new_strategy
            self.current_session.strategy_switches += 1
        
        logger.info(f"Changement de stratégie : {new_strategy.value}")
    
    def _evaluate_and_switch_strategy(self):
        """Évalue les performances et change de stratégie si nécessaire."""
        if not self.current_session or not self.current_session.target:
            return
        
        current_xp_per_hour = self.metrics.calculate_xp_per_hour(60)
        current_death_rate = self.metrics.calculate_death_rate(60)
        
        # Met à jour les performances de la stratégie actuelle
        if self.current_session.active_strategy:
            self.metrics.update_strategy_performance(
                self.current_session.active_strategy,
                current_xp_per_hour,
                current_death_rate
            )
        
        # Évalue si un changement est nécessaire
        if self._should_switch_strategy(current_xp_per_hour, current_death_rate):
            new_strategy = self._select_optimal_strategy(
                self.current_session.current_level,
                "Unknown",  # TODO: Récupérer la classe du personnage
                self.current_session.target
            )
            
            if new_strategy and new_strategy != self.current_session.active_strategy:
                self._switch_strategy(new_strategy)
    
    def _should_switch_strategy(self, current_xp_per_hour: float, 
                              current_death_rate: float) -> bool:
        """Détermine s'il faut changer de stratégie."""
        # Changement si performances insuffisantes
        if current_xp_per_hour < self.min_xp_per_hour_threshold:
            logger.warning(f"XP/heure trop faible : {current_xp_per_hour}")
            return True
        
        # Changement si trop de morts
        if current_death_rate > self.max_deaths_per_hour:
            logger.warning(f"Taux de mortalité trop élevé : {current_death_rate}")
            return True
        
        # Changement si seuil de sécurité non respecté
        if self.current_session and self.current_session.target:
            target_safety = self.current_session.target.min_safety_threshold
            if self.active_strategy:
                current_safety = self.active_strategy.get_safety_rating(
                    self.current_session.current_level
                )
                if current_safety < target_safety:
                    logger.warning("Seuil de sécurité non respecté")
                    return True
        
        return False
    
    def _check_emergency_stop(self) -> bool:
        """Vérifie les conditions d'arrêt d'urgence."""
        if self.emergency_stop_triggered:
            return True
        
        # Arrêt si trop de morts récentes
        recent_death_rate = self.metrics.calculate_death_rate(30)  # 30 minutes
        if recent_death_rate > 5:  # Plus de 5 morts en 30 minutes
            logger.error("Arrêt d'urgence : trop de morts récentes")
            self.emergency_stop_triggered = True
            return True
        
        # Arrêt si session trop longue
        if (self.current_session and 
            (datetime.now() - self.current_session.start_time).total_seconds() > 14400):  # 4 heures
            logger.warning("Arrêt : session trop longue (4h)")
            return True
        
        return False
    
    def _is_target_reached(self) -> bool:
        """Vérifie si l'objectif est atteint."""
        if not self.current_session or not self.current_session.target:
            return False
        
        target = self.current_session.target
        
        # Vérification niveau cible
        if self.current_session.current_level >= target.target_level:
            return True
        
        # Vérification XP cible
        if target.target_xp and self.current_session.current_xp >= target.target_xp:
            return True
        
        # Vérification limite de temps
        if target.time_limit and datetime.now() >= target.time_limit:
            logger.info("Limite de temps atteinte")
            return True
        
        return False
    
    def _check_and_manage_resources(self) -> bool:
        """Vérifie et gère les ressources du personnage."""
        # TODO: Implémenter la vérification réelle des ressources
        # - Santé/Mana
        # - Pods (inventaire plein)
        # - Kamas (pour achats nécessaires)
        # - Objets de consommation
        
        # Pour l'instant, simulation
        return True
    
    def _update_session_metrics(self):
        """Met à jour les métriques de la session."""
        if not self.current_session:
            return
        
        # TODO: Récupérer les vraies valeurs du jeu
        # Pour l'instant, simulation
        current_xp = self.current_session.current_xp + 1000  # Simulation gain XP
        current_level = self.current_session.current_level
        
        if current_xp != self.current_session.current_xp:
            self.current_session.current_xp = current_xp
            self.current_session.total_xp_gained = current_xp - self.current_session.start_xp
            self.metrics.record_xp(current_xp)
        
        if current_level != self.current_session.current_level:
            self.current_session.current_level = current_level
            self.metrics.record_level(current_level)
            logger.info(f"LEVEL UP ! Nouveau niveau : {current_level}")
    
    def _handle_repeated_failures(self):
        """Gère les échecs répétés."""
        logger.warning("Échecs répétés détectés, changement de stratégie forcé")
        
        if self.current_session:
            self.current_session.error_count = 0
        
        # Force le changement vers une stratégie plus sûre
        safer_strategies = [
            LevelingStrategy.QUEST_CHAIN,  # Plus sûr
            LevelingStrategy.DUNGEON_FARMING,  # Sécurisé
            LevelingStrategy.MONSTER_GRINDING  # En dernier recours
        ]
        
        for strategy in safer_strategies:
            if (strategy != self.current_session.active_strategy and 
                strategy in self.strategies):
                self._switch_strategy(strategy)
                break
    
    def _generate_session_report(self):
        """Génère un rapport de session."""
        if not self.current_session:
            return
        
        session = self.current_session
        duration = (session.end_time - session.start_time).total_seconds() / 3600
        
        logger.info("=== RAPPORT DE SESSION DE LEVELING ===")
        logger.info(f"ID de session : {session.session_id}")
        logger.info(f"Durée : {duration:.2f} heures")
        logger.info(f"Niveau initial : {session.start_level}")
        logger.info(f"Niveau final : {session.current_level}")
        logger.info(f"Niveaux gagnés : {session.current_level - session.start_level}")
        logger.info(f"XP totale gagnée : {session.total_xp_gained:,}")
        logger.info(f"XP/heure moyenne : {session.total_xp_gained/duration:.0f}")
        logger.info(f"Monstres tués : {session.monsters_killed}")
        logger.info(f"Donjons complétés : {session.dungeons_completed}")
        logger.info(f"Quêtes terminées : {session.quests_completed}")
        logger.info(f"Nombre de morts : {session.deaths}")
        logger.info(f"Changements de stratégie : {session.strategy_switches}")
        logger.info(f"Erreurs rencontrées : {session.error_count}")
        
        if session.kamas_earned > 0 or session.kamas_spent > 0:
            logger.info(f"Kamas gagnés : {session.kamas_earned:,}")
            logger.info(f"Kamas dépensés : {session.kamas_spent:,}")
            logger.info(f"Bénéfice net : {session.kamas_earned - session.kamas_spent:,}")
        
        logger.info("=====================================")
    
    def get_session_status(self) -> Optional[Dict[str, Any]]:
        """Retourne le statut actuel de la session."""
        if not self.current_session:
            return None
        
        session = self.current_session
        current_time = datetime.now()
        duration = (current_time - session.start_time).total_seconds()
        
        return {
            'session_id': session.session_id,
            'is_running': self.is_running,
            'start_time': session.start_time.isoformat(),
            'duration_seconds': int(duration),
            'current_level': session.current_level,
            'target_level': session.target.target_level if session.target else 0,
            'current_xp': session.current_xp,
            'total_xp_gained': session.total_xp_gained,
            'xp_per_hour': self.metrics.calculate_xp_per_hour(60),
            'death_rate': self.metrics.calculate_death_rate(60),
            'active_strategy': session.active_strategy.value if session.active_strategy else None,
            'progress': {
                'monsters_killed': session.monsters_killed,
                'dungeons_completed': session.dungeons_completed,
                'quests_completed': session.quests_completed,
                'deaths': session.deaths,
                'strategy_switches': session.strategy_switches
            },
            'efficiency_metrics': {
                'xp_per_hour_current': self.metrics.calculate_xp_per_hour(60),
                'xp_per_hour_session': session.total_xp_gained / (duration / 3600) if duration > 0 else 0,
                'death_rate_hourly': self.metrics.calculate_death_rate(60),
                'errors_per_hour': session.error_count / (duration / 3600) if duration > 0 else 0
            }
        }
    
    def pause_session(self, duration_minutes: int = 30):
        """Met en pause la session actuelle."""
        if self.is_running and self.current_session:
            self.is_running = False
            self.current_session.pause_time += duration_minutes * 60
            logger.info(f"Session mise en pause pour {duration_minutes} minutes")
            
            # Reprend automatiquement après la pause
            time.sleep(duration_minutes * 60)
            self.is_running = True
            logger.info("Session reprise automatiquement")
    
    def get_strategy_recommendations(self, character_level: int, 
                                   character_class: str = "Unknown") -> List[Dict[str, Any]]:
        """Retourne des recommandations de stratégies."""
        recommendations = []
        
        # Crée un target temporaire pour l'évaluation
        temp_target = LevelingTarget(
            target_level=character_level + 10,
            optimization_criteria=OptimizationCriteria.BALANCED
        )
        
        for strategy_type, strategy_impl in self.strategies.items():
            if strategy_impl.can_execute(character_level, character_class):
                score = self._calculate_strategy_score(
                    strategy_type, strategy_impl, character_level, temp_target
                )
                
                historical_perf = self.metrics.get_strategy_performance(strategy_type)
                
                recommendations.append({
                    'strategy': strategy_type.value,
                    'score': score,
                    'estimated_xp_per_hour': strategy_impl.get_estimated_xp_per_hour(character_level),
                    'safety_rating': strategy_impl.get_safety_rating(character_level),
                    'historical_performance': historical_perf,
                    'suitable_for_level': True
                })
        
        # Trie par score décroissant
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations