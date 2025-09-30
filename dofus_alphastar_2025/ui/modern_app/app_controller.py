#!/usr/bin/env python3
"""
AppController - Contrôleur principal de l'application DOFUS AlphaStar
Gère la logique métier et la coordination entre les modules
"""

import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import queue

from config import config
from core.quest_system import create_quest_manager, QuestManager
from core.navigation_system import create_ganymede_navigator, GanymedeNavigator
from core.npc_system import create_npc_recognition, NPCRecognition
from core.npc_system import create_contextual_intelligence, ContextualIntelligence
from core.guide_system import create_guide_loader, create_strategy_optimizer
from core.alphastar_engine import create_league_system, LeagueManager
from core.rl_training import create_rllib_trainer, RLlibTrainer

logger = logging.getLogger(__name__)

class BotState(Enum):
    """États du bot"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"

class ActionType(Enum):
    """Types d'actions du bot"""
    QUEST = "quest"
    FARMING = "farming"
    NAVIGATION = "navigation"
    COMBAT = "combat"
    DIALOGUE = "dialogue"
    INVENTORY = "inventory"
    IDLE = "idle"

@dataclass
class BotStatus:
    """Status complet du bot"""
    state: BotState = BotState.STOPPED
    current_action: Optional[ActionType] = None
    current_map: str = ""
    player_level: int = 1
    player_hp: int = 100
    player_mp: int = 100
    kamas: int = 0
    experience: int = 0

    # Quêtes
    active_quests: List[str] = field(default_factory=list)
    completed_quests: int = 0

    # Performance
    actions_per_hour: float = 0.0
    exp_per_hour: float = 0.0
    kamas_per_hour: float = 0.0

    # Temps
    session_start_time: Optional[float] = None
    total_runtime: float = 0.0

    # Erreurs
    last_error: Optional[str] = None
    error_count: int = 0

@dataclass
class AppSettings:
    """Paramètres de l'application"""
    # Bot settings
    auto_start: bool = False
    auto_restart_on_error: bool = True
    max_session_duration: float = 14400.0  # 4 heures

    # UI settings
    theme: str = "dark"
    update_interval: float = 1.0  # secondes
    enable_notifications: bool = True
    enable_sound_alerts: bool = True

    # Performance
    enable_performance_mode: bool = False
    cpu_limit: float = 80.0
    memory_limit: float = 2048.0  # MB

    # Logging
    log_level: str = "INFO"
    max_log_entries: int = 1000
    save_detailed_logs: bool = True

class EventManager:
    """Gestionnaire d'événements pour l'UI"""

    def __init__(self):
        self.listeners: Dict[str, List[Callable]] = {}
        self.event_queue = queue.Queue()

    def subscribe(self, event_type: str, callback: Callable):
        """S'abonner à un type d'événement"""
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(callback)

    def emit(self, event_type: str, data: Any = None):
        """Émettre un événement"""
        self.event_queue.put((event_type, data, time.time()))

        # Notifier immédiatement les listeners
        if event_type in self.listeners:
            for callback in self.listeners[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Erreur callback événement {event_type}: {e}")

class PerformanceMonitor:
    """Moniteur de performance système"""

    def __init__(self):
        self.metrics_history: List[Dict[str, Any]] = []
        self.max_history = 1000

    def collect_metrics(self) -> Dict[str, Any]:
        """Collecte métriques système"""
        import psutil

        # CPU et mémoire
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        # GPU (si disponible)
        gpu_usage = 0.0
        gpu_memory = 0.0
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_usage = gpu_info.gpu

            gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory = (gpu_mem_info.used / gpu_mem_info.total) * 100
        except:
            pass  # GPU monitoring optional

        metrics = {
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_mb": memory.used / 1024 / 1024,
            "gpu_percent": gpu_usage,
            "gpu_memory_percent": gpu_memory
        }

        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)

        return metrics

class AppController:
    """Contrôleur principal de l'application"""

    def __init__(self):
        # Composants core
        self.quest_manager: Optional[QuestManager] = None
        self.navigator: Optional[GanymedeNavigator] = None
        self.npc_recognition: Optional[NPCRecognition] = None
        self.contextual_ai: Optional[ContextualIntelligence] = None
        self.guide_loader = None
        self.strategy_optimizer = None
        self.league_manager: Optional[LeagueManager] = None
        self.rl_trainer: Optional[RLlibTrainer] = None

        # État de l'application
        self.bot_status = BotStatus()
        self.app_settings = AppSettings()
        self.event_manager = EventManager()
        self.performance_monitor = PerformanceMonitor()

        # Threading
        self.main_thread: Optional[threading.Thread] = None
        self.monitoring_thread: Optional[threading.Thread] = None
        self.should_stop = threading.Event()

        # Logs et analytics
        self.log_buffer: List[Dict[str, Any]] = []
        self.analytics_data: Dict[str, Any] = {}

        logger.info("AppController initialisé")

    def initialize_systems(self) -> bool:
        """Initialise tous les systèmes du bot"""
        try:
            logger.info("Initialisation des systèmes...")

            # Core systems
            self.quest_manager = create_quest_manager()
            self.navigator = create_ganymede_navigator()
            self.npc_recognition = create_npc_recognition()
            self.contextual_ai = create_contextual_intelligence()

            # Guide et optimisation
            self.guide_loader = create_guide_loader()
            self.strategy_optimizer = create_strategy_optimizer()

            # RL et league (optionnel)
            if config.alphastar.enable_league:
                self.league_manager = create_league_system()

            if config.rl.enable_training:
                self.rl_trainer = create_rllib_trainer()

            self.event_manager.emit("systems_initialized", {
                "status": "success",
                "timestamp": time.time()
            })

            logger.info("Systèmes initialisés avec succès")
            return True

        except Exception as e:
            logger.error(f"Erreur initialisation systèmes: {e}")
            self.event_manager.emit("systems_initialization_failed", {"error": str(e)})
            return False

    def start_bot(self, mode: str = "auto") -> bool:
        """Démarre le bot"""
        if self.bot_status.state in [BotState.RUNNING, BotState.STARTING]:
            logger.warning("Bot déjà en cours d'exécution")
            return False

        try:
            self.bot_status.state = BotState.STARTING
            self.bot_status.session_start_time = time.time()
            self.bot_status.last_error = None

            self.event_manager.emit("bot_starting", {"mode": mode})

            # Initialiser si nécessaire
            if not self.quest_manager:
                if not self.initialize_systems():
                    self.bot_status.state = BotState.ERROR
                    return False

            # Démarrer threads
            self.should_stop.clear()

            self.main_thread = threading.Thread(
                target=self._main_bot_loop,
                args=(mode,),
                daemon=True
            )
            self.main_thread.start()

            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()

            self.bot_status.state = BotState.RUNNING
            self.event_manager.emit("bot_started", {"mode": mode})

            logger.info(f"Bot démarré en mode {mode}")
            return True

        except Exception as e:
            logger.error(f"Erreur démarrage bot: {e}")
            self.bot_status.state = BotState.ERROR
            self.bot_status.last_error = str(e)
            self.event_manager.emit("bot_error", {"error": str(e)})
            return False

    def stop_bot(self) -> bool:
        """Arrête le bot"""
        if self.bot_status.state == BotState.STOPPED:
            return True

        try:
            self.bot_status.state = BotState.STOPPING
            self.event_manager.emit("bot_stopping", {})

            # Signal d'arrêt
            self.should_stop.set()

            # Attendre threads
            if self.main_thread and self.main_thread.is_alive():
                self.main_thread.join(timeout=5.0)

            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=2.0)

            self.bot_status.state = BotState.STOPPED
            self.bot_status.current_action = None

            if self.bot_status.session_start_time:
                self.bot_status.total_runtime += time.time() - self.bot_status.session_start_time
                self.bot_status.session_start_time = None

            self.event_manager.emit("bot_stopped", {})

            logger.info("Bot arrêté")
            return True

        except Exception as e:
            logger.error(f"Erreur arrêt bot: {e}")
            self.bot_status.state = BotState.ERROR
            return False

    def pause_bot(self) -> bool:
        """Met en pause le bot"""
        if self.bot_status.state == BotState.RUNNING:
            self.bot_status.state = BotState.PAUSED
            self.event_manager.emit("bot_paused", {})
            logger.info("Bot mis en pause")
            return True
        return False

    def resume_bot(self) -> bool:
        """Reprend le bot"""
        if self.bot_status.state == BotState.PAUSED:
            self.bot_status.state = BotState.RUNNING
            self.event_manager.emit("bot_resumed", {})
            logger.info("Bot repris")
            return True
        return False

    def _main_bot_loop(self, mode: str):
        """Boucle principale du bot"""
        try:
            logger.info(f"Démarrage boucle principale (mode: {mode})")

            while not self.should_stop.is_set():
                if self.bot_status.state == BotState.PAUSED:
                    time.sleep(1.0)
                    continue

                if self.bot_status.state != BotState.RUNNING:
                    break

                try:
                    # Logique principale selon le mode
                    if mode == "auto":
                        self._execute_auto_mode()
                    elif mode == "quest_only":
                        self._execute_quest_mode()
                    elif mode == "farming":
                        self._execute_farming_mode()
                    else:
                        self._execute_auto_mode()

                    time.sleep(0.1)  # Petite pause

                except Exception as e:
                    logger.error(f"Erreur boucle principale: {e}")
                    self.bot_status.error_count += 1
                    self.bot_status.last_error = str(e)

                    if self.app_settings.auto_restart_on_error:
                        time.sleep(5.0)  # Pause avant retry
                    else:
                        break

        except Exception as e:
            logger.error(f"Erreur fatale boucle principale: {e}")
            self.bot_status.state = BotState.ERROR
            self.bot_status.last_error = str(e)
            self.event_manager.emit("bot_error", {"error": str(e)})

    def _execute_auto_mode(self):
        """Exécute mode automatique intelligent"""
        # Utiliser l'IA contextuelle pour décider
        if self.contextual_ai:
            # Simuler contexte de jeu
            from core.npc_system.contextual_intelligence import GameContext

            game_context = GameContext(
                player_level=self.bot_status.player_level,
                current_hp=self.bot_status.player_hp,
                current_mp=self.bot_status.player_mp,
                kamas=self.bot_status.kamas,
                experience=self.bot_status.experience
            )

            decision = self.contextual_ai.process_game_state(game_context)

            # Exécuter décision
            self._execute_decision(decision)

    def _execute_quest_mode(self):
        """Exécute mode quête uniquement"""
        if self.quest_manager and self.bot_status.active_quests:
            self.bot_status.current_action = ActionType.QUEST
            # Logique quête ici
            pass

    def _execute_farming_mode(self):
        """Exécute mode farming"""
        self.bot_status.current_action = ActionType.FARMING
        # Logique farming ici
        pass

    def _execute_decision(self, decision):
        """Exécute une décision de l'IA"""
        action_type = decision.action_type

        # Mapper action vers ActionType
        if "quest" in action_type:
            self.bot_status.current_action = ActionType.QUEST
        elif "combat" in action_type:
            self.bot_status.current_action = ActionType.COMBAT
        elif "dialogue" in action_type:
            self.bot_status.current_action = ActionType.DIALOGUE
        elif "navigate" in action_type:
            self.bot_status.current_action = ActionType.NAVIGATION
        else:
            self.bot_status.current_action = ActionType.IDLE

        # Émettre événement de progression
        self.event_manager.emit("bot_action_updated", {
            "action": self.bot_status.current_action.value,
            "decision": action_type,
            "confidence": decision.confidence
        })

    def _monitoring_loop(self):
        """Boucle de monitoring et métriques"""
        try:
            while not self.should_stop.is_set():
                # Collecter métriques
                metrics = self.performance_monitor.collect_metrics()

                # Mettre à jour analytics
                self._update_analytics(metrics)

                # Émettre événements de monitoring
                self.event_manager.emit("performance_metrics", metrics)
                self.event_manager.emit("bot_status_updated", self.get_status_dict())

                time.sleep(self.app_settings.update_interval)

        except Exception as e:
            logger.error(f"Erreur monitoring: {e}")

    def _update_analytics(self, metrics: Dict[str, Any]):
        """Met à jour les analytics"""
        current_time = time.time()

        # Calculer rates
        if self.bot_status.session_start_time:
            session_duration = current_time - self.bot_status.session_start_time
            if session_duration > 0:
                self.bot_status.exp_per_hour = (self.bot_status.experience / session_duration) * 3600
                self.bot_status.kamas_per_hour = (self.bot_status.kamas / session_duration) * 3600

        # Stocker analytics
        self.analytics_data.update({
            "last_update": current_time,
            "performance_metrics": metrics,
            "bot_stats": self.get_status_dict()
        })

    def get_status_dict(self) -> Dict[str, Any]:
        """Retourne statut sous forme de dictionnaire"""
        return {
            "state": self.bot_status.state.value,
            "current_action": self.bot_status.current_action.value if self.bot_status.current_action else None,
            "current_map": self.bot_status.current_map,
            "player_level": self.bot_status.player_level,
            "player_hp": self.bot_status.player_hp,
            "player_mp": self.bot_status.player_mp,
            "kamas": self.bot_status.kamas,
            "experience": self.bot_status.experience,
            "active_quests": self.bot_status.active_quests,
            "completed_quests": self.bot_status.completed_quests,
            "exp_per_hour": self.bot_status.exp_per_hour,
            "kamas_per_hour": self.bot_status.kamas_per_hour,
            "total_runtime": self.bot_status.total_runtime,
            "error_count": self.bot_status.error_count,
            "last_error": self.bot_status.last_error
        }

    def execute_manual_action(self, action: str, params: Dict[str, Any] = None):
        """Exécute action manuelle"""
        try:
            if action == "start_quest":
                quest_id = params.get("quest_id")
                if self.quest_manager and quest_id:
                    success = self.quest_manager.start_quest(quest_id)
                    if success:
                        self.bot_status.active_quests.append(quest_id)

            elif action == "navigate_to":
                location = params.get("location")
                if self.navigator and location:
                    route = self.navigator.navigate_to_location(location, self.bot_status.player_level)

            elif action == "use_strategy":
                strategy_name = params.get("strategy")
                # Appliquer stratégie
                pass

            self.event_manager.emit("manual_action_executed", {
                "action": action,
                "params": params,
                "timestamp": time.time()
            })

        except Exception as e:
            logger.error(f"Erreur action manuelle {action}: {e}")
            self.event_manager.emit("manual_action_failed", {
                "action": action,
                "error": str(e)
            })

    def get_analytics_data(self) -> Dict[str, Any]:
        """Retourne données analytics"""
        return self.analytics_data.copy()

    def get_performance_history(self, duration: int = 3600) -> List[Dict[str, Any]]:
        """Retourne historique performance (dernière heure par défaut)"""
        cutoff_time = time.time() - duration
        return [
            m for m in self.performance_monitor.metrics_history
            if m["timestamp"] >= cutoff_time
        ]

    def update_settings(self, new_settings: Dict[str, Any]):
        """Met à jour paramètres application"""
        for key, value in new_settings.items():
            if hasattr(self.app_settings, key):
                setattr(self.app_settings, key, value)

        self.event_manager.emit("settings_updated", new_settings)
        logger.info(f"Paramètres mis à jour: {new_settings}")

    def shutdown(self):
        """Arrêt propre de l'application"""
        logger.info("Arrêt de l'application...")

        # Arrêter bot
        self.stop_bot()

        # Sauvegarder données si nécessaire
        if self.quest_manager:
            self.quest_manager.save_progress("data/quest_progress.json")

        self.event_manager.emit("app_shutdown", {})

def create_app_controller() -> AppController:
    """Factory function pour créer AppController"""
    return AppController()