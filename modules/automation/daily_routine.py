"""
Module de routines quotidiennes intelligentes.

Gère l'automatisation des tâches récurrentes journalières :
- Récupération des récompenses quotidiennes
- Activités périodiques optimisées
- Planification automatique des tâches
- Gestion des événements temporaires
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Priorités des tâches quotidiennes."""
    CRITICAL = 1     # Tâches critiques (récompenses limitées)
    HIGH = 2         # Tâches importantes (XP, kamas)
    MEDIUM = 3       # Tâches utiles (optimisation)
    LOW = 4          # Tâches optionnelles (bonus)


class TaskStatus(Enum):
    """États des tâches."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DailyTask:
    """Représente une tâche quotidienne."""
    name: str
    description: str
    priority: TaskPriority
    execution_func: Callable[[], bool]
    prerequisites: List[str] = field(default_factory=list)
    reset_time: str = "06:00"  # Heure de reset quotidien
    max_attempts: int = 3
    estimated_duration: int = 300  # Secondes
    rewards: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    # État d'exécution
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    last_execution: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    error_message: Optional[str] = None


class EventType(Enum):
    """Types d'événements temporaires."""
    DOUBLE_XP = "double_xp"
    BONUS_KAMAS = "bonus_kamas"
    SPECIAL_DUNGEON = "special_dungeon"
    LIMITED_QUEST = "limited_quest"
    SEASONAL_EVENT = "seasonal_event"


@dataclass
class TemporaryEvent:
    """Événement temporaire du jeu."""
    name: str
    event_type: EventType
    start_time: datetime
    end_time: datetime
    bonus_multiplier: float = 1.0
    special_tasks: List[str] = field(default_factory=list)
    priority_boost: int = 0  # Boost de priorité


class DailyRoutineAutomation:
    """
    Système d'automatisation des routines quotidiennes.
    
    Gère l'exécution intelligente et optimisée des tâches récurrentes.
    """
    
    def __init__(self, game_interface=None):
        self.game_interface = game_interface
        self.tasks: Dict[str, DailyTask] = {}
        self.active_events: List[TemporaryEvent] = []
        self.execution_order: List[str] = []
        self.last_reset: Optional[datetime] = None
        self.is_running = False
        
        # Configuration
        self.max_daily_duration = 7200  # 2 heures max par jour
        self.break_interval = 1800  # Pause toutes les 30 minutes
        self.retry_delay = 300  # Délai entre les tentatives
        
        # Statistiques
        self.daily_stats = {
            'tasks_completed': 0,
            'total_xp_gained': 0,
            'total_kamas_gained': 0,
            'execution_time': 0,
            'errors_encountered': 0
        }
        
        self._initialize_default_tasks()
        logger.info("Système de routines quotidiennes initialisé")
    
    def _initialize_default_tasks(self):
        """Initialise les tâches quotidiennes par défaut."""
        default_tasks = [
            DailyTask(
                name="almanax_quest",
                description="Quête Almanax quotidienne",
                priority=TaskPriority.CRITICAL,
                execution_func=self._execute_almanax_quest,
                estimated_duration=600,
                rewards={'xp': 50000, 'kamas': 10000},
                conditions={'min_level': 30}
            ),
            DailyTask(
                name="daily_rewards",
                description="Récupération des récompenses de connexion",
                priority=TaskPriority.CRITICAL,
                execution_func=self._collect_daily_rewards,
                estimated_duration=120,
                rewards={'items': 'various'}
            ),
            DailyTask(
                name="guild_tasks",
                description="Tâches de guilde quotidiennes",
                priority=TaskPriority.HIGH,
                execution_func=self._execute_guild_tasks,
                estimated_duration=900,
                rewards={'guild_xp': 1000}
            ),
            DailyTask(
                name="profession_daily",
                description="Récoltes et crafts quotidiens optimisés",
                priority=TaskPriority.HIGH,
                execution_func=self._execute_profession_daily,
                estimated_duration=1800,
                rewards={'profession_xp': 20000}
            ),
            DailyTask(
                name="treasure_hunt",
                description="Chasses aux trésors quotidiennes",
                priority=TaskPriority.MEDIUM,
                execution_func=self._execute_treasure_hunts,
                estimated_duration=1200,
                rewards={'xp': 30000, 'kamas': 50000}
            ),
            DailyTask(
                name="kolossium_fights",
                description="Combats Kolossium pour récompenses",
                priority=TaskPriority.MEDIUM,
                execution_func=self._execute_kolossium_fights,
                estimated_duration=600,
                rewards={'honor': 500, 'tokens': 10}
            )
        ]
        
        for task in default_tasks:
            self.add_task(task)
    
    def add_task(self, task: DailyTask):
        """Ajoute une nouvelle tâche quotidienne."""
        self.tasks[task.name] = task
        self._update_execution_order()
        logger.info(f"Tâche ajoutée : {task.name}")
    
    def remove_task(self, task_name: str):
        """Supprime une tâche quotidienne."""
        if task_name in self.tasks:
            del self.tasks[task_name]
            self._update_execution_order()
            logger.info(f"Tâche supprimée : {task_name}")
    
    def add_temporary_event(self, event: TemporaryEvent):
        """Ajoute un événement temporaire."""
        self.active_events.append(event)
        self._adjust_priorities_for_events()
        logger.info(f"Événement temporaire ajouté : {event.name}")
    
    def _update_execution_order(self):
        """Met à jour l'ordre d'exécution optimal des tâches."""
        # Tri par priorité, puis par durée estimée
        sorted_tasks = sorted(
            self.tasks.values(),
            key=lambda t: (t.priority.value, t.estimated_duration)
        )
        self.execution_order = [task.name for task in sorted_tasks]
    
    def _adjust_priorities_for_events(self):
        """Ajuste les priorités selon les événements actifs."""
        current_time = datetime.now()
        
        for event in self.active_events:
            if event.start_time <= current_time <= event.end_time:
                # Boost les tâches liées à l'événement
                for task_name in event.special_tasks:
                    if task_name in self.tasks:
                        task = self.tasks[task_name]
                        # Diminue la valeur de priorité (plus prioritaire)
                        old_priority = task.priority.value
                        new_priority = max(1, old_priority - event.priority_boost)
                        task.priority = TaskPriority(new_priority)
        
        self._update_execution_order()
    
    def _check_daily_reset(self):
        """Vérifie si un reset quotidien est nécessaire."""
        now = datetime.now()
        reset_time = now.replace(hour=6, minute=0, second=0, microsecond=0)
        
        if now.hour < 6:
            reset_time -= timedelta(days=1)
        
        if self.last_reset is None or self.last_reset < reset_time:
            self._perform_daily_reset()
            self.last_reset = now
            return True
        return False
    
    def _perform_daily_reset(self):
        """Effectue le reset quotidien des tâches."""
        logger.info("Début du reset quotidien")
        
        for task in self.tasks.values():
            task.status = TaskStatus.PENDING
            task.attempts = 0
            task.completion_time = None
            task.error_message = None
        
        # Reset des statistiques
        self.daily_stats = {
            'tasks_completed': 0,
            'total_xp_gained': 0,
            'total_kamas_gained': 0,
            'execution_time': 0,
            'errors_encountered': 0
        }
        
        # Nettoyage des événements expirés
        current_time = datetime.now()
        self.active_events = [
            event for event in self.active_events 
            if event.end_time > current_time
        ]
        
        logger.info("Reset quotidien terminé")
    
    def start_daily_routine(self):
        """Lance l'exécution automatique des routines quotidiennes."""
        if self.is_running:
            logger.warning("Les routines quotidiennes sont déjà en cours")
            return
        
        self.is_running = True
        logger.info("Démarrage des routines quotidiennes")
        
        try:
            self._check_daily_reset()
            start_time = time.time()
            
            for task_name in self.execution_order:
                if not self.is_running:
                    break
                
                task = self.tasks[task_name]
                
                if self._should_execute_task(task):
                    success = self._execute_task(task)
                    if success:
                        self.daily_stats['tasks_completed'] += 1
                    
                    # Pause entre les tâches
                    if self.is_running:
                        time.sleep(60)  # 1 minute de pause
                
                # Vérification de la durée maximale
                elapsed_time = time.time() - start_time
                if elapsed_time > self.max_daily_duration:
                    logger.warning("Durée maximale quotidienne atteinte")
                    break
            
            self.daily_stats['execution_time'] = int(time.time() - start_time)
            self._generate_daily_report()
            
        except Exception as e:
            logger.error(f"Erreur lors des routines quotidiennes : {e}")
            self.daily_stats['errors_encountered'] += 1
        finally:
            self.is_running = False
            logger.info("Fin des routines quotidiennes")
    
    def _should_execute_task(self, task: DailyTask) -> bool:
        """Détermine si une tâche doit être exécutée."""
        if task.status in [TaskStatus.COMPLETED, TaskStatus.RUNNING]:
            return False
        
        if task.attempts >= task.max_attempts:
            task.status = TaskStatus.FAILED
            return False
        
        # Vérification des prérequis
        for prereq in task.prerequisites:
            if prereq in self.tasks:
                prereq_task = self.tasks[prereq]
                if prereq_task.status != TaskStatus.COMPLETED:
                    return False
        
        # Vérification des conditions
        if not self._check_task_conditions(task):
            task.status = TaskStatus.SKIPPED
            return False
        
        return True
    
    def _check_task_conditions(self, task: DailyTask) -> bool:
        """Vérifie les conditions d'exécution d'une tâche."""
        conditions = task.conditions
        
        if 'min_level' in conditions:
            # Ici on devrait vérifier le niveau du personnage
            # Pour l'exemple, on suppose que c'est vérifié
            pass
        
        if 'required_items' in conditions:
            # Vérification de la présence d'objets requis
            pass
        
        if 'time_window' in conditions:
            # Vérification de la plage horaire
            now = datetime.now().time()
            start_time = datetime.strptime(conditions['time_window']['start'], '%H:%M').time()
            end_time = datetime.strptime(conditions['time_window']['end'], '%H:%M').time()
            
            if not (start_time <= now <= end_time):
                return False
        
        return True
    
    def _execute_task(self, task: DailyTask) -> bool:
        """Exécute une tâche quotidienne."""
        logger.info(f"Exécution de la tâche : {task.name}")
        task.status = TaskStatus.RUNNING
        task.attempts += 1
        
        try:
            start_time = time.time()
            success = task.execution_func()
            
            if success:
                task.status = TaskStatus.COMPLETED
                task.completion_time = datetime.now()
                execution_time = int(time.time() - start_time)
                
                # Mise à jour des statistiques
                self._update_stats_from_task(task, execution_time)
                
                logger.info(f"Tâche terminée avec succès : {task.name} ({execution_time}s)")
                return True
            else:
                task.status = TaskStatus.FAILED
                task.error_message = "Échec de l'exécution"
                logger.warning(f"Échec de la tâche : {task.name}")
                return False
                
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            logger.error(f"Erreur lors de l'exécution de {task.name} : {e}")
            self.daily_stats['errors_encountered'] += 1
            return False
    
    def _update_stats_from_task(self, task: DailyTask, execution_time: int):
        """Met à jour les statistiques à partir d'une tâche terminée."""
        rewards = task.rewards
        
        if 'xp' in rewards:
            self.daily_stats['total_xp_gained'] += rewards['xp']
        
        if 'kamas' in rewards:
            self.daily_stats['total_kamas_gained'] += rewards['kamas']
        
        self.daily_stats['execution_time'] += execution_time
    
    def stop_daily_routine(self):
        """Arrête l'exécution des routines quotidiennes."""
        self.is_running = False
        logger.info("Arrêt des routines quotidiennes demandé")
    
    def get_task_status(self, task_name: str) -> Optional[TaskStatus]:
        """Retourne le statut d'une tâche."""
        if task_name in self.tasks:
            return self.tasks[task_name].status
        return None
    
    def get_daily_progress(self) -> Dict[str, Any]:
        """Retourne le progrès des tâches quotidiennes."""
        total_tasks = len(self.tasks)
        completed_tasks = sum(1 for task in self.tasks.values() 
                            if task.status == TaskStatus.COMPLETED)
        
        return {
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'progress_percentage': (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0,
            'stats': self.daily_stats,
            'task_details': {
                name: {
                    'status': task.status.value,
                    'attempts': task.attempts,
                    'completion_time': task.completion_time.isoformat() if task.completion_time else None,
                    'error': task.error_message
                }
                for name, task in self.tasks.items()
            }
        }
    
    def _generate_daily_report(self):
        """Génère un rapport quotidien détaillé."""
        progress = self.get_daily_progress()
        
        logger.info("=== RAPPORT QUOTIDIEN ===")
        logger.info(f"Tâches terminées : {progress['completed_tasks']}/{progress['total_tasks']}")
        logger.info(f"Progrès : {progress['progress_percentage']:.1f}%")
        logger.info(f"XP totale gagnée : {self.daily_stats['total_xp_gained']:,}")
        logger.info(f"Kamas totaux gagnés : {self.daily_stats['total_kamas_gained']:,}")
        logger.info(f"Temps d'exécution : {self.daily_stats['execution_time']} secondes")
        logger.info(f"Erreurs rencontrées : {self.daily_stats['errors_encountered']}")
        logger.info("=========================")
    
    # Méthodes d'exécution des tâches spécifiques
    
    def _execute_almanax_quest(self) -> bool:
        """Exécute la quête Almanax quotidienne."""
        logger.info("Exécution de la quête Almanax")
        try:
            # Logique d'interaction avec le jeu pour la quête Almanax
            # 1. Aller au temple Almanax
            # 2. Parler au PNJ
            # 3. Récupérer les objets nécessaires si pas en possession
            # 4. Valider la quête
            
            # Simulation pour l'exemple
            time.sleep(30)  # Simulation du temps d'exécution
            return True
        except Exception as e:
            logger.error(f"Erreur quête Almanax : {e}")
            return False
    
    def _collect_daily_rewards(self) -> bool:
        """Récupère les récompenses de connexion quotidiennes."""
        logger.info("Récupération des récompenses quotidiennes")
        try:
            # Logique pour cliquer sur les récompenses de connexion
            # 1. Ouvrir l'interface des récompenses
            # 2. Cliquer sur "Récupérer"
            # 3. Fermer l'interface
            
            time.sleep(10)
            return True
        except Exception as e:
            logger.error(f"Erreur récompenses quotidiennes : {e}")
            return False
    
    def _execute_guild_tasks(self) -> bool:
        """Exécute les tâches de guilde quotidiennes."""
        logger.info("Exécution des tâches de guilde")
        try:
            # Logique pour les tâches de guilde
            # 1. Ouvrir l'interface de guilde
            # 2. Consulter les tâches disponibles
            # 3. Exécuter les tâches optimales
            
            time.sleep(60)
            return True
        except Exception as e:
            logger.error(f"Erreur tâches de guilde : {e}")
            return False
    
    def _execute_profession_daily(self) -> bool:
        """Exécute les activités de profession quotidiennes."""
        logger.info("Exécution des activités de profession")
        try:
            # Logique pour les professions
            # 1. Identifier la profession la plus rentable du jour
            # 2. Récolter/crafter de manière optimisée
            # 3. Vendre les surplus
            
            time.sleep(120)
            return True
        except Exception as e:
            logger.error(f"Erreur activités profession : {e}")
            return False
    
    def _execute_treasure_hunts(self) -> bool:
        """Exécute les chasses aux trésors quotidiennes."""
        logger.info("Exécution des chasses aux trésors")
        try:
            # Logique pour les chasses aux trésors
            # 1. Acheter/obtenir des cartes au trésor
            # 2. Résoudre les indices automatiquement
            # 3. Récupérer les récompenses
            
            time.sleep(90)
            return True
        except Exception as e:
            logger.error(f"Erreur chasses aux trésors : {e}")
            return False
    
    def _execute_kolossium_fights(self) -> bool:
        """Exécute les combats Kolossium quotidiens."""
        logger.info("Exécution des combats Kolossium")
        try:
            # Logique pour le Kolossium
            # 1. Rejoindre la file d'attente
            # 2. Combattre automatiquement
            # 3. Récupérer les récompenses
            
            time.sleep(45)
            return True
        except Exception as e:
            logger.error(f"Erreur combats Kolossium : {e}")
            return False