"""
Moteur central du bot - Orchestrateur principal de tous les modules
Ce fichier contient la classe BotEngine qui coordonne l'ensemble du système
"""

import time
import threading
import logging
from typing import Dict, Any, Optional, List, Type
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os
import sys
from pathlib import Path

# Imports locaux
from .module_interface import IModule, IGameModule, ModuleStatus
from .event_bus import EventBus, Event, EventType, EventPriority


@dataclass
class EngineConfig:
    """Configuration du moteur principal"""
    target_fps: int = 30                    # FPS cible pour la boucle principale
    decision_fps: int = 10                  # FPS pour les décisions (plus économique)
    max_modules: int = 50                   # Nombre maximum de modules
    enable_logging: bool = True             # Activation du logging
    log_level: str = "INFO"                 # Niveau de log
    performance_monitoring: bool = True      # Monitoring des performances
    auto_recovery: bool = True              # Récupération automatique des modules
    safety_checks: bool = True              # Vérifications de sécurité


class PerformanceMonitor:
    """
    Moniteur de performances pour le moteur
    Suit les métriques critiques du système
    """
    
    def __init__(self):
        self.metrics = {
            "loop_time": [],
            "fps_actual": 0,
            "memory_usage": 0,
            "cpu_usage": 0,
            "active_modules": 0,
            "events_per_second": 0,
            "errors_per_minute": 0
        }
        self.last_update = datetime.now()
        self.warning_thresholds = {
            "loop_time_avg": 0.040,  # 40ms = problème si > 33ms pour 30fps
            "memory_usage": 512,     # MB
            "cpu_usage": 80,         # %
            "errors_per_minute": 10
        }
    
    def update_loop_time(self, loop_time: float):
        """Met à jour le temps de boucle"""
        self.metrics["loop_time"].append(loop_time)
        
        # Garde seulement les 100 dernières mesures
        if len(self.metrics["loop_time"]) > 100:
            self.metrics["loop_time"].pop(0)
        
        # Calcul FPS réel
        if loop_time > 0:
            self.metrics["fps_actual"] = 1.0 / loop_time
    
    def get_average_loop_time(self) -> float:
        """Retourne le temps moyen de boucle"""
        if not self.metrics["loop_time"]:
            return 0.0
        return sum(self.metrics["loop_time"]) / len(self.metrics["loop_time"])
    
    def check_performance_warnings(self) -> List[str]:
        """Vérifie et retourne les avertissements de performance"""
        warnings = []
        
        avg_loop_time = self.get_average_loop_time()
        if avg_loop_time > self.warning_thresholds["loop_time_avg"]:
            warnings.append(f"Temps de boucle élevé: {avg_loop_time:.3f}s")
        
        if self.metrics["memory_usage"] > self.warning_thresholds["memory_usage"]:
            warnings.append(f"Usage mémoire élevé: {self.metrics['memory_usage']}MB")
        
        if self.metrics["errors_per_minute"] > self.warning_thresholds["errors_per_minute"]:
            warnings.append(f"Trop d'erreurs: {self.metrics['errors_per_minute']}/min")
        
        return warnings


class BotEngine:
    """
    Moteur central qui orchestre tous les modules du bot
    
    Responsabilités:
    - Gestion du cycle de vie des modules
    - Coordination entre modules via le bus d'événements
    - Monitoring des performances
    - Gestion des erreurs et récupération automatique
    - Interface principale pour contrôler le bot
    """
    
    def __init__(self, config: EngineConfig = None):
        """
        Initialise le moteur du bot
        
        Args:
            config: Configuration du moteur
        """
        self.config = config or EngineConfig()
        
        # Configuration du système de logging
        self._setup_logging()
        self.logger = logging.getLogger(f"{__name__}.BotEngine")
        
        # État du moteur
        self.is_running = False
        self.is_initialized = False
        self.start_time = None
        self.shutdown_requested = False
        
        # Gestion des modules
        self.modules: Dict[str, IModule] = {}
        self.module_load_order: List[str] = []
        self.module_dependencies: Dict[str, List[str]] = {}
        
        # Système d'événements
        self.event_bus = EventBus()
        
        # Monitoring et threading
        self.performance_monitor = PerformanceMonitor()
        self.main_thread = None
        self.thread_lock = threading.RLock()
        
        # État du jeu (sera mis à jour par les modules)
        self.game_state = None
        
        # Statistiques
        self.stats = {
            "total_cycles": 0,
            "total_actions_executed": 0,
            "total_errors": 0,
            "modules_restarted": 0,
            "uptime_seconds": 0
        }
        
        self.logger.info("BotEngine initialisé")
    
    def _setup_logging(self):
        """Configure le système de logging"""
        if not self.config.enable_logging:
            return
        
        # Configuration du format de log
        log_format = "[%(asctime)s] %(levelname)s [%(name)s] %(message)s"
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("logs/tacticalbot.log", encoding='utf-8')
            ]
        )
        
        # Création du dossier logs s'il n'existe pas
        Path("logs").mkdir(exist_ok=True)
    
    def initialize(self, modules_config: Dict[str, Dict[str, Any]] = None) -> bool:
        """
        Initialise le moteur et tous ses composants
        
        Args:
            modules_config: Configuration des modules à charger
            
        Returns:
            bool: True si l'initialisation réussit
        """
        try:
            self.logger.info("Démarrage de l'initialisation du moteur")
            
            # Démarrage du bus d'événements
            self.event_bus.start()
            
            # Abonnement aux événements système
            self.event_bus.subscribe(
                "engine_core",
                {EventType.MODULE_ERROR, EventType.SHUTDOWN_REQUESTED, 
                 EventType.PERFORMANCE_WARNING},
                self._handle_system_event
            )
            
            # Chargement des modules si configuration fournie
            if modules_config:
                for module_name, config in modules_config.items():
                    self._load_module_from_config(module_name, config)
            
            # Initialisation des modules dans l'ordre des dépendances
            if not self._initialize_modules():
                return False
            
            self.is_initialized = True
            self.logger.info("Moteur initialisé avec succès")
            
            # Événement de démarrage
            self.event_bus.publish_immediate(
                EventType.CONFIG_CHANGED,
                {"status": "engine_initialized", "modules_count": len(self.modules)},
                "engine_core",
                EventPriority.HIGH
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation: {e}")
            return False
    
    def register_module(self, module: IModule, dependencies: List[str] = None) -> bool:
        """
        Enregistre un nouveau module dans le moteur
        
        Args:
            module: Instance du module à enregistrer
            dependencies: Liste des modules dont celui-ci dépend
            
        Returns:
            bool: True si l'enregistrement réussit
        """
        try:
            if module.name in self.modules:
                self.logger.warning(f"Module {module.name} déjà enregistré")
                return False
            
            if len(self.modules) >= self.config.max_modules:
                self.logger.error("Nombre maximum de modules atteint")
                return False
            
            # Enregistrement du module
            self.modules[module.name] = module
            module.engine = self  # Référence vers le moteur
            
            # Gestion des dépendances
            if dependencies:
                self.module_dependencies[module.name] = dependencies
            
            self.logger.info(f"Module {module.name} enregistré")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'enregistrement de {module.name}: {e}")
            return False
    
    def start(self) -> bool:
        """
        Démarre la boucle principale du bot
        
        Returns:
            bool: True si le démarrage réussit
        """
        if not self.is_initialized:
            self.logger.error("Moteur non initialisé")
            return False
        
        if self.is_running:
            self.logger.warning("Moteur déjà en cours d'exécution")
            return False
        
        try:
            self.is_running = True
            self.start_time = datetime.now()
            self.shutdown_requested = False
            
            # Démarrage du thread principal
            self.main_thread = threading.Thread(
                target=self._main_loop,
                name="BotEngine-MainLoop",
                daemon=False
            )
            self.main_thread.start()
            
            self.logger.info("Moteur démarré")
            
            # Événement de démarrage
            self.event_bus.publish_immediate(
                EventType.CONFIG_CHANGED,
                {"status": "engine_started"},
                "engine_core",
                EventPriority.HIGH
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors du démarrage: {e}")
            self.is_running = False
            return False
    
    def stop(self, timeout: float = 5.0) -> None:
        """
        Arrête le moteur et tous ses modules
        
        Args:
            timeout: Timeout pour l'arrêt graceful en secondes
        """
        self.logger.info("Demande d'arrêt du moteur")
        
        # Signal d'arrêt
        self.shutdown_requested = True
        
        # Événement d'arrêt
        self.event_bus.publish_immediate(
            EventType.SHUTDOWN_REQUESTED,
            {"reason": "user_requested"},
            "engine_core",
            EventPriority.CRITICAL
        )
        
        # Attente de l'arrêt du thread principal
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=timeout)
        
        # Nettoyage des modules
        self._cleanup_modules()
        
        # Arrêt du bus d'événements
        self.event_bus.stop()
        
        self.is_running = False
        self.logger.info("Moteur arrêté")
    
    def _main_loop(self) -> None:
        """
        Boucle principale du moteur - cœur du système
        Fonctionne à 30 FPS avec gestion intelligente du timing
        """
        self.logger.info("Démarrage de la boucle principale")
        
        # Calcul du timing pour 30 FPS
        target_frame_time = 1.0 / self.config.target_fps
        decision_interval = 1.0 / self.config.decision_fps
        last_decision_time = 0
        
        cycle_count = 0
        
        while self.is_running and not self.shutdown_requested:
            cycle_start = time.perf_counter()
            
            try:
                # === PHASE 1: Mise à jour de l'état du jeu ===
                self._update_game_state()
                
                # === PHASE 2: Traitement des événements prioritaires ===
                # (les événements sont traités en arrière-plan par le bus)
                
                # === PHASE 3: Mise à jour des modules ===
                current_time = time.perf_counter()
                is_decision_frame = (current_time - last_decision_time) >= decision_interval
                
                self._update_modules(is_decision_frame)
                
                if is_decision_frame:
                    last_decision_time = current_time
                
                # === PHASE 4: Monitoring et statistiques ===
                if self.config.performance_monitoring:
                    self._update_performance_stats()
                
                # === PHASE 5: Vérifications de sécurité ===
                if self.config.safety_checks:
                    self._perform_safety_checks()
                
                cycle_count += 1
                self.stats["total_cycles"] += 1
                
            except Exception as e:
                self.logger.error(f"Erreur dans la boucle principale: {e}")
                self.stats["total_errors"] += 1
                
                if self.config.auto_recovery:
                    self._attempt_error_recovery(e)
            
            # === PHASE 6: Régulation du timing ===
            cycle_end = time.perf_counter()
            cycle_time = cycle_end - cycle_start
            
            # Mise à jour du monitoring
            self.performance_monitor.update_loop_time(cycle_time)
            
            # Attente pour maintenir le FPS cible
            sleep_time = max(0, target_frame_time - cycle_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.logger.info(f"Boucle principale terminée après {cycle_count} cycles")
    
    def _update_game_state(self) -> None:
        """
        Met à jour l'état global du jeu
        Collecte les informations de tous les modules d'analyse
        """
        # L'état du jeu sera mis à jour par le module state manager
        # Cette méthode sert d'orchestrateur
        
        # Calcul du temps de fonctionnement
        if self.start_time:
            self.stats["uptime_seconds"] = (datetime.now() - self.start_time).total_seconds()
    
    def _update_modules(self, is_decision_frame: bool) -> None:
        """
        Met à jour tous les modules actifs
        
        Args:
            is_decision_frame: True si c'est un cycle de décision
        """
        for module_name, module in list(self.modules.items()):
            try:
                if not module.is_active():
                    continue
                
                # Appel de la méthode update du module
                result = module.update(self.game_state)
                
                # Traitement du résultat si nécessaire
                if result and is_decision_frame:
                    self._process_module_result(module_name, result)
                    
            except Exception as e:
                self.logger.error(f"Erreur dans le module {module_name}: {e}")
                self._handle_module_error(module_name, e)
    
    def _process_module_result(self, module_name: str, result: Dict[str, Any]) -> None:
        """
        Traite le résultat retourné par un module
        
        Args:
            module_name: Nom du module
            result: Données retournées par le module
        """
        # Si le module suggère une action
        if "suggested_action" in result:
            action = result["suggested_action"]
            self.logger.debug(f"Action suggérée par {module_name}: {action}")
        
        # Si le module partage des données
        if "shared_data" in result:
            data = result["shared_data"]
            # Publier les données via le bus d'événements
            self.event_bus.publish_immediate(
                EventType.CONFIG_CHANGED,  # Type générique pour partage de données
                {"module": module_name, "data": data},
                module_name
            )
    
    def _handle_system_event(self, event: Event) -> bool:
        """
        Gestionnaire des événements système
        
        Args:
            event: Événement à traiter
            
        Returns:
            bool: True si l'événement a été traité
        """
        try:
            if event.type == EventType.SHUTDOWN_REQUESTED:
                self.shutdown_requested = True
                return True
            
            elif event.type == EventType.MODULE_ERROR:
                module_name = event.data.get("module_name")
                error = event.data.get("error")
                if module_name:
                    self._handle_module_error(module_name, error)
                return True
            
            elif event.type == EventType.PERFORMANCE_WARNING:
                warning = event.data.get("warning", "Performance warning")
                self.logger.warning(f"Alerte performance: {warning}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement d'événement système: {e}")
            return False
    
    def _handle_module_error(self, module_name: str, error: Exception) -> None:
        """
        Gère les erreurs de modules avec récupération automatique
        
        Args:
            module_name: Nom du module en erreur
            error: Exception survenue
        """
        if module_name not in self.modules:
            return
        
        module = self.modules[module_name]
        module.set_error(str(error))
        
        self.stats["total_errors"] += 1
        
        # Tentative de récupération automatique
        if self.config.auto_recovery and not module.is_critical():
            self.logger.warning(f"Tentative de récupération du module {module_name}")
            
            try:
                # Pause du module
                module.pause()
                time.sleep(0.1)
                
                # Remise à zéro des erreurs et redémarrage
                module.reset_errors()
                if hasattr(module, 'restart'):
                    module.restart()
                else:
                    module.resume()
                
                self.stats["modules_restarted"] += 1
                self.logger.info(f"Module {module_name} récupéré avec succès")
                
            except Exception as recovery_error:
                self.logger.error(f"Échec de la récupération de {module_name}: {recovery_error}")
    
    def _initialize_modules(self) -> bool:
        """
        Initialise tous les modules enregistrés en respectant les dépendances
        
        Returns:
            bool: True si tous les modules sont initialisés
        """
        # Tri topologique pour respecter les dépendances
        load_order = self._calculate_load_order()
        
        for module_name in load_order:
            if module_name not in self.modules:
                continue
            
            module = self.modules[module_name]
            
            try:
                self.logger.info(f"Initialisation du module {module_name}")
                module.status = ModuleStatus.INITIALIZING
                
                # Configuration par défaut si non fournie
                config = getattr(module, '_config', {})
                
                if module.initialize(config):
                    module.status = ModuleStatus.ACTIVE
                    self.logger.info(f"Module {module_name} initialisé")
                else:
                    module.status = ModuleStatus.ERROR
                    self.logger.error(f"Échec de l'initialisation de {module_name}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Erreur lors de l'initialisation de {module_name}: {e}")
                module.status = ModuleStatus.ERROR
                return False
        
        return True
    
    def _calculate_load_order(self) -> List[str]:
        """
        Calcule l'ordre de chargement des modules selon leurs dépendances
        
        Returns:
            List: Ordre de chargement des modules
        """
        # Implémentation simple - dans un vrai système, utiliser un tri topologique
        visited = set()
        load_order = []
        
        def visit_module(module_name: str):
            if module_name in visited or module_name not in self.modules:
                return
            
            visited.add(module_name)
            
            # Traiter d'abord les dépendances
            dependencies = self.module_dependencies.get(module_name, [])
            for dep in dependencies:
                visit_module(dep)
            
            load_order.append(module_name)
        
        # Visiter tous les modules
        for module_name in self.modules:
            visit_module(module_name)
        
        return load_order
    
    def _cleanup_modules(self) -> None:
        """Nettoie tous les modules lors de l'arrêt"""
        for module_name, module in self.modules.items():
            try:
                self.logger.info(f"Nettoyage du module {module_name}")
                module.cleanup()
                module.status = ModuleStatus.INACTIVE
            except Exception as e:
                self.logger.error(f"Erreur lors du nettoyage de {module_name}: {e}")
    
    def _update_performance_stats(self) -> None:
        """Met à jour les statistiques de performance"""
        self.performance_monitor.metrics["active_modules"] = len(
            [m for m in self.modules.values() if m.is_active()]
        )
        
        # Vérification des seuils d'alerte
        warnings = self.performance_monitor.check_performance_warnings()
        for warning in warnings:
            self.event_bus.publish_immediate(
                EventType.PERFORMANCE_WARNING,
                {"warning": warning},
                "engine_core",
                EventPriority.HIGH
            )
    
    def _perform_safety_checks(self) -> None:
        """Effectue les vérifications de sécurité"""
        # Vérification du temps de fonctionnement
        if self.stats["uptime_seconds"] > 14400:  # 4 heures
            self.logger.warning("Temps de fonctionnement élevé - recommandation de pause")
        
        # Vérification du taux d'erreurs
        if self.stats["total_errors"] > 100:
            self.logger.warning("Taux d'erreurs élevé - vérification recommandée")
    
    def _attempt_error_recovery(self, error: Exception) -> None:
        """
        Tente une récupération automatique après une erreur critique
        
        Args:
            error: Exception survenue
        """
        self.logger.info(f"Tentative de récupération après erreur: {error}")
        
        # Stratégies de récupération possibles
        # 1. Redémarrage des modules en erreur
        # 2. Réinitialisation de l'état du jeu
        # 3. Nettoyage de la mémoire
        
        # Implémentation basique
        time.sleep(0.5)  # Pause courte
    
    def get_module(self, name: str) -> Optional[IModule]:
        """
        Récupère un module par son nom
        
        Args:
            name: Nom du module
            
        Returns:
            IModule: Instance du module ou None
        """
        return self.modules.get(name)
    
    def get_active_modules(self) -> List[str]:
        """
        Retourne la liste des modules actifs
        
        Returns:
            List: Noms des modules actifs
        """
        return [name for name, module in self.modules.items() if module.is_active()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques complètes du moteur
        
        Returns:
            Dict: Statistiques détaillées
        """
        return {
            "engine": {
                "is_running": self.is_running,
                "uptime_seconds": self.stats["uptime_seconds"],
                "total_cycles": self.stats["total_cycles"],
                "total_errors": self.stats["total_errors"]
            },
            "modules": {
                "total": len(self.modules),
                "active": len(self.get_active_modules()),
                "restarted": self.stats["modules_restarted"]
            },
            "performance": self.performance_monitor.metrics,
            "events": self.event_bus.get_statistics()
        }
    
    def execute_action(self, module_name: str, action: Any) -> bool:
        """
        Demande à un module d'exécuter une action
        
        Args:
            module_name: Nom du module
            action: Action à exécuter
            
        Returns:
            bool: True si l'action a été exécutée
        """
        module = self.get_module(module_name)
        if not module or not isinstance(module, IGameModule):
            return False
        
        try:
            success = module.execute_action(action)
            if success:
                self.stats["total_actions_executed"] += 1
            return success
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution d'action: {e}")
            return False
    
    def _load_module_from_config(self, module_name: str, config: Dict[str, Any]) -> bool:
        """
        Charge un module à partir de sa configuration
        
        Args:
            module_name: Nom du module
            config: Configuration du module
            
        Returns:
            bool: True si le chargement réussit
        """
        # Cette méthode sera utilisée pour charger dynamiquement des modules
        # Pour l'instant, placeholder
        self.logger.info(f"Configuration reçue pour module {module_name}: {config}")
        return True