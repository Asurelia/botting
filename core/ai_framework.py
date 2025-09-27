"""
Core AI Framework - Cerveau Central de l'IA DOFUS
Méta-orchestrateur qui coordonne tous les sous-systèmes pour une autonomie complète
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import traceback

# Import des modules internes
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import des composants IA spécialisés
from core.knowledge_graph import DofusKnowledgeGraph, create_dofus_knowledge_graph
from core.predictive_engine import PredictiveAnalyticsEngine, create_predictive_engine, PredictionRequest, PredictionType, TimeWindow
from core.uncertainty import UncertaintyManager
from core.decision_engine import AdvancedDecisionEngine, create_decision_engine, Objective, Action, ObjectiveType, Priority
from core.emotional_state import EmotionalStateManager, create_emotional_manager, GameEvent, MoodState
from core.state_tracker import MultiDimensionalStateTracker, create_state_tracker, StateLevel, GameState
from core.social_intelligence import SocialModule, SocialAction, RelationshipType, SocialContext
from core.adaptive_execution import AdaptiveExecutionModule, ExecutionStyle, OptimizationMetric
from core.meta_evolution import MetaEvolutionModule, EvolutionTrigger, EvolutionStrategy
from core.genetic_learning import GeneticLearningModule, GeneType, SelectionMethod

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIModuleState(Enum):
    """États possibles des modules IA"""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    ACTIVE = auto()
    PAUSED = auto()
    ERROR = auto()
    SHUTDOWN = auto()

class Priority(Enum):
    """Niveaux de priorité pour les tâches"""
    CRITICAL = 1    # Sécurité, anti-détection
    HIGH = 2        # Combat, événements urgents
    MEDIUM = 3      # Objectifs principaux
    LOW = 4         # Optimisations, apprentissage
    BACKGROUND = 5  # Maintenance, monitoring

@dataclass
class AITask:
    """Tâche IA avec priorité et contexte"""
    name: str
    priority: Priority
    function: Callable
    args: tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 30.0
    retry_count: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModuleHealth:
    """État de santé d'un module IA"""
    state: AIModuleState = AIModuleState.UNINITIALIZED
    last_update: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    performance_score: float = 1.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    response_time: float = 0.0

class AIModule:
    """Classe de base pour tous les modules IA"""

    def __init__(self, name: str):
        self.name = name
        self.health = ModuleHealth()
        self.logger = logging.getLogger(f"ai_framework.{name}")
        self._shutdown_event = asyncio.Event()
        self._shared_data = {}

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialise le module"""
        self.health.state = AIModuleState.INITIALIZING
        try:
            result = await self._initialize_impl(config)
            self.health.state = AIModuleState.ACTIVE if result else AIModuleState.ERROR
            return result
        except Exception as e:
            self.logger.error(f"Erreur initialisation {self.name}: {e}")
            self.health.state = AIModuleState.ERROR
            return False

    async def _initialize_impl(self, config: Dict[str, Any]) -> bool:
        """Implémentation spécifique de l'initialisation"""
        return True

    async def process(self, data: Any) -> Any:
        """Traite des données (à surcharger)"""
        return data

    async def get_shared_data(self) -> Dict[str, Any]:
        """Retourne les données à partager avec les autres modules"""
        return self._shared_data.copy()

    async def receive_shared_data(self, shared_data: Dict[str, Any]):
        """Reçoit les données partagées des autres modules"""
        # Implémentation par défaut - peut être surchargée
        pass

    async def shutdown(self):
        """Arrêt propre du module"""
        self.health.state = AIModuleState.SHUTDOWN
        self._shutdown_event.set()

    def is_healthy(self) -> bool:
        """Vérifie si le module est en bonne santé"""
        return (
            self.health.state == AIModuleState.ACTIVE and
            self.health.error_count < 10 and
            self.health.performance_score > 0.3
        )

class MetaOrchestrator:
    """
    Méta-orchestrateur principal de l'IA DOFUS
    Coordonne perception, cognition et action pour une autonomie complète
    """

    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger("MetaOrchestrator")
        self.config_path = Path(config_path) if config_path else Path("config/ai_config.json")

        # État principal
        self.running = False
        self.modules: Dict[str, AIModule] = {}
        self.task_queue = asyncio.PriorityQueue()
        self.performance_metrics = {}

        # Configuration
        self.config = self._load_config()

        # Coordination
        self._coordination_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()

        # Monitoring
        self._monitor_task = None
        self._coordinator_task = None

        # Threading pour tâches bloquantes
        self.thread_executor = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 4)
        )

    def _load_config(self) -> Dict[str, Any]:
        """Charge la configuration IA"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                self.logger.info(f"Configuration chargée: {self.config_path}")
                return config
            else:
                self.logger.warning("Configuration par défaut utilisée")
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Erreur chargement config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Configuration par défaut"""
        return {
            "orchestrator": {
                "max_workers": 4,
                "monitor_interval": 1.0,
                "coordination_interval": 0.1,
                "error_threshold": 10
            },
            "modules": {
                "knowledge": {
                    "enabled": True,
                    "priority": 1,
                    "timeout": 10.0,
                    "data_path": "data/knowledge"
                },
                "prediction": {
                    "enabled": True,
                    "priority": 2,
                    "timeout": 15.0,
                    "confidence_threshold": 0.5
                },
                "uncertainty": {
                    "enabled": True,
                    "priority": 1,
                    "timeout": 5.0,
                    "risk_threshold": 0.7
                },
                "decision": {
                    "enabled": True,
                    "priority": 1,
                    "timeout": 20.0,
                    "max_objectives": 10
                },
                "emotional": {
                    "enabled": True,
                    "priority": 3,
                    "timeout": 10.0,
                    "personality": {
                        "openness": 0.7,
                        "conscientiousness": 0.8,
                        "extraversion": 0.5,
                        "agreeableness": 0.6,
                        "neuroticism": 0.3,
                        "risk_taking": 0.6,
                        "competitiveness": 0.7,
                        "patience": 0.7,
                        "curiosity": 0.8,
                        "social_tendency": 0.5
                    }
                },
                "state_tracking": {
                    "enabled": True,
                    "priority": 2,
                    "timeout": 5.0,
                    "auto_update": True
                }
            },
            "performance": {
                "target_fps": 30,
                "max_memory_mb": 2048,
                "optimization_enabled": True
            }
        }

    async def register_module(self, module: AIModule) -> bool:
        """Enregistre un module IA"""
        try:
            async with self._coordination_lock:
                if module.name in self.modules:
                    self.logger.warning(f"Module {module.name} déjà enregistré")
                    return False

                # Initialisation du module
                module_config = self.config.get('modules', {}).get(module.name, {})
                if await module.initialize(module_config):
                    self.modules[module.name] = module
                    self.logger.info(f"✅ Module {module.name} enregistré et initialisé")
                    return True
                else:
                    self.logger.error(f"❌ Échec initialisation module {module.name}")
                    return False

        except Exception as e:
            self.logger.error(f"Erreur enregistrement module {module.name}: {e}")
            return False

    async def submit_task(self, task: AITask) -> bool:
        """Soumet une tâche à exécuter"""
        try:
            # Priority Queue utilise la valeur de l'enum comme priorité
            await self.task_queue.put((task.priority.value, time.time(), task))
            self.logger.debug(f"Tâche soumise: {task.name} (priorité {task.priority.name})")
            return True
        except Exception as e:
            self.logger.error(f"Erreur soumission tâche {task.name}: {e}")
            return False

    async def _execute_task(self, task: AITask) -> Any:
        """Exécute une tâche avec gestion d'erreurs"""
        start_time = time.perf_counter()

        try:
            # Exécution avec timeout
            if asyncio.iscoroutinefunction(task.function):
                result = await asyncio.wait_for(
                    task.function(*task.args, **task.kwargs),
                    timeout=task.timeout
                )
            else:
                # Exécution dans le thread pool pour fonctions bloquantes
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        self.thread_executor,
                        lambda: task.function(*task.args, **task.kwargs)
                    ),
                    timeout=task.timeout
                )

            execution_time = time.perf_counter() - start_time
            self.logger.debug(f"Tâche {task.name} exécutée en {execution_time:.3f}s")

            return result

        except asyncio.TimeoutError:
            self.logger.error(f"Timeout tâche {task.name} après {task.timeout}s")
            raise
        except Exception as e:
            self.logger.error(f"Erreur exécution tâche {task.name}: {e}")
            raise

    async def _coordination_loop(self):
        """Boucle principale de coordination"""
        self.logger.info("🚀 Démarrage boucle de coordination")

        while self.running and not self._shutdown_event.is_set():
            try:
                # Traitement des tâches en attente
                await self._process_task_queue()

                # Coordination entre modules
                await self._coordinate_modules()

                # Pause pour éviter la surcharge CPU
                coordination_interval = self.config.get('orchestrator', {}).get(
                    'coordination_interval', 0.1
                )
                await asyncio.sleep(coordination_interval)

            except asyncio.CancelledError:
                self.logger.info("Coordination loop annulée")
                break
            except Exception as e:
                self.logger.error(f"Erreur boucle coordination: {e}")
                await asyncio.sleep(1.0)  # Pause en cas d'erreur

    async def _process_task_queue(self):
        """Traite les tâches dans la queue par priorité"""
        processed_tasks = 0
        max_tasks_per_cycle = 10  # Limite pour éviter le blocage

        while not self.task_queue.empty() and processed_tasks < max_tasks_per_cycle:
            try:
                # Récupération tâche avec priorité
                priority, timestamp, task = await asyncio.wait_for(
                    self.task_queue.get(), timeout=0.01
                )

                # Exécution de la tâche
                await self._execute_task(task)
                processed_tasks += 1

            except asyncio.TimeoutError:
                break  # Queue vide
            except Exception as e:
                self.logger.error(f"Erreur traitement tâche: {e}")

    async def _coordinate_modules(self):
        """Coordonne les modules entre eux"""
        try:
            # Vérification santé des modules
            for module_name, module in self.modules.items():
                if not module.is_healthy():
                    self.logger.warning(f"Module {module_name} en mauvaise santé")
                    # TODO: Tentative de récupération automatique

            # Synchronisation données entre modules
            await self._synchronize_module_data()

        except Exception as e:
            self.logger.error(f"Erreur coordination modules: {e}")

    async def _synchronize_module_data(self):
        """Synchronise les données entre modules"""
        # Collecte des données de tous les modules
        shared_data = {}

        for module_name, module in self.modules.items():
            try:
                if hasattr(module, 'get_shared_data'):
                    module_data = await module.get_shared_data()
                    if module_data:
                        shared_data[module_name] = module_data
            except Exception as e:
                self.logger.debug(f"Pas de données partagées pour {module_name}: {e}")

        # Distribution des données aux modules
        for module_name, module in self.modules.items():
            try:
                if hasattr(module, 'receive_shared_data'):
                    relevant_data = {k: v for k, v in shared_data.items() if k != module_name}
                    await module.receive_shared_data(relevant_data)
            except Exception as e:
                self.logger.debug(f"Erreur partage données vers {module_name}: {e}")

    async def _monitoring_loop(self):
        """Boucle de monitoring des performances"""
        self.logger.info("📊 Démarrage monitoring performances")

        while self.running and not self._shutdown_event.is_set():
            try:
                # Collecte métriques
                await self._collect_performance_metrics()

                # Optimisations automatiques
                if self.config.get('performance', {}).get('optimization_enabled', True):
                    await self._optimize_performance()

                # Intervalle monitoring
                monitor_interval = self.config.get('orchestrator', {}).get(
                    'monitor_interval', 1.0
                )
                await asyncio.sleep(monitor_interval)

            except asyncio.CancelledError:
                self.logger.info("Monitoring loop annulé")
                break
            except Exception as e:
                self.logger.error(f"Erreur monitoring: {e}")
                await asyncio.sleep(5.0)

    async def _collect_performance_metrics(self):
        """Collecte les métriques de performance"""
        try:
            import psutil

            # Métriques système
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()

            self.performance_metrics = {
                'timestamp': datetime.now().isoformat(),
                'system': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_mb': memory.available / 1024 / 1024
                },
                'modules': {},
                'task_queue_size': self.task_queue.qsize()
            }

            # Métriques par module
            for module_name, module in self.modules.items():
                self.performance_metrics['modules'][module_name] = {
                    'state': module.health.state.name,
                    'error_count': module.health.error_count,
                    'performance_score': module.health.performance_score,
                    'response_time': module.health.response_time
                }

        except Exception as e:
            self.logger.error(f"Erreur collecte métriques: {e}")

    async def _optimize_performance(self):
        """Optimisations automatiques des performances"""
        try:
            metrics = self.performance_metrics
            system_metrics = metrics.get('system', {})

            # Optimisation mémoire si usage élevé
            memory_percent = system_metrics.get('memory_percent', 0)
            if memory_percent > 80:
                self.logger.warning(f"Usage mémoire élevé: {memory_percent:.1f}%")
                await self._optimize_memory_usage()

            # Optimisation CPU si usage élevé
            cpu_percent = system_metrics.get('cpu_percent', 0)
            if cpu_percent > 90:
                self.logger.warning(f"Usage CPU élevé: {cpu_percent:.1f}%")
                await self._optimize_cpu_usage()

        except Exception as e:
            self.logger.error(f"Erreur optimisation performance: {e}")

    async def _optimize_memory_usage(self):
        """Optimise l'usage mémoire"""
        # Déclenchement garbage collection
        import gc
        collected = gc.collect()
        self.logger.info(f"Garbage collection: {collected} objets récupérés")

    async def _optimize_cpu_usage(self):
        """Optimise l'usage CPU"""
        # Réduction temporaire de la fréquence de traitement
        coordination_interval = self.config.get('orchestrator', {}).get(
            'coordination_interval', 0.1
        )
        await asyncio.sleep(coordination_interval * 2)

    async def start(self) -> bool:
        """Démarre l'orchestrateur IA"""
        try:
            self.logger.info("🚀 Démarrage MetaOrchestrator...")

            if self.running:
                self.logger.warning("Orchestrateur déjà en cours d'exécution")
                return False

            self.running = True

            # Démarrage des boucles principales
            self._coordinator_task = asyncio.create_task(self._coordination_loop())
            self._monitor_task = asyncio.create_task(self._monitoring_loop())

            self.logger.info("✅ MetaOrchestrator démarré avec succès")
            return True

        except Exception as e:
            self.logger.error(f"Erreur démarrage orchestrateur: {e}")
            return False

    async def stop(self):
        """Arrête l'orchestrateur proprement"""
        self.logger.info("🛑 Arrêt MetaOrchestrator...")

        self.running = False
        self._shutdown_event.set()

        # Arrêt des tâches principales
        if self._coordinator_task:
            self._coordinator_task.cancel()
            try:
                await self._coordinator_task
            except asyncio.CancelledError:
                pass

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # Arrêt des modules
        for module_name, module in self.modules.items():
            try:
                await module.shutdown()
                self.logger.info(f"Module {module_name} arrêté")
            except Exception as e:
                self.logger.error(f"Erreur arrêt module {module_name}: {e}")

        # Arrêt thread executor
        self.thread_executor.shutdown(wait=True)

        self.logger.info("✅ MetaOrchestrator arrêté")

    def get_status(self) -> Dict[str, Any]:
        """Retourne l'état actuel de l'orchestrateur"""
        return {
            'running': self.running,
            'modules_count': len(self.modules),
            'modules_healthy': sum(1 for m in self.modules.values() if m.is_healthy()),
            'task_queue_size': self.task_queue.qsize() if hasattr(self.task_queue, 'qsize') else 0,
            'performance_metrics': self.performance_metrics,
            'modules_status': {
                name: {
                    'state': module.health.state.name,
                    'healthy': module.is_healthy()
                }
                for name, module in self.modules.items()
            }
        }

class KnowledgeModule(AIModule):
    """Module de gestion des connaissances avec Knowledge Graph"""

    def __init__(self):
        super().__init__("knowledge")
        self.knowledge_graph: Optional[DofusKnowledgeGraph] = None

    async def _initialize_impl(self, config: Dict[str, Any]) -> bool:
        """Initialise le Knowledge Graph"""
        try:
            self.logger.info("🧠 Initialisation Knowledge Graph...")
            self.knowledge_graph = await create_dofus_knowledge_graph()

            # Mise à jour des données partagées
            self._shared_data = {
                'total_entities': len(self.knowledge_graph.entities),
                'knowledge_stats': self.knowledge_graph.get_statistics(),
                'last_update': datetime.now().isoformat()
            }

            self.logger.info(f"✅ Knowledge Graph initialisé: {len(self.knowledge_graph.entities)} entités")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation Knowledge Graph: {e}")
            return False

    async def process(self, data: Any) -> Any:
        """Traite les requêtes de connaissances"""
        if not self.knowledge_graph:
            return None

        try:
            query_type = data.get('type') if isinstance(data, dict) else None

            if query_type == 'find_entity':
                name = data.get('name')
                return self.knowledge_graph.find_entities_by_name(name)

            elif query_type == 'get_relations':
                entity_id = data.get('entity_id')
                return self.knowledge_graph.get_relations(entity_id)

            elif query_type == 'find_path':
                source = data.get('source')
                target = data.get('target')
                return self.knowledge_graph.infer_shortest_path(source, target)

            elif query_type == 'find_strategy':
                goal = data.get('goal')
                resources = data.get('resources', [])
                return self.knowledge_graph.find_optimal_strategy(goal, resources)

            return None

        except Exception as e:
            self.logger.error(f"Erreur traitement requête Knowledge: {e}")
            return None

    async def get_shared_data(self) -> Dict[str, Any]:
        """Partage les statistiques du Knowledge Graph"""
        if self.knowledge_graph:
            return {
                'knowledge_stats': self.knowledge_graph.get_statistics(),
                'entities_count': len(self.knowledge_graph.entities),
                'last_update': datetime.now().isoformat()
            }
        return {}

class PredictionModule(AIModule):
    """Module de prédiction avec Predictive Analytics Engine"""

    def __init__(self, knowledge_graph: DofusKnowledgeGraph):
        super().__init__("prediction")
        self.knowledge_graph = knowledge_graph
        self.prediction_engine: Optional[PredictiveAnalyticsEngine] = None

    async def _initialize_impl(self, config: Dict[str, Any]) -> bool:
        """Initialise le Predictive Engine"""
        try:
            self.logger.info("🔮 Initialisation Predictive Engine...")
            self.prediction_engine = await create_predictive_engine(self.knowledge_graph)

            # Mise à jour des données partagées
            self._shared_data = {
                'prediction_stats': self.prediction_engine.get_statistics(),
                'models_count': len(self.prediction_engine.predictors),
                'last_update': datetime.now().isoformat()
            }

            self.logger.info(f"✅ Predictive Engine initialisé: {len(self.prediction_engine.predictors)} modèles")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation Predictive Engine: {e}")
            return False

    async def process(self, data: Any) -> Any:
        """Traite les requêtes de prédiction"""
        if not self.prediction_engine:
            return None

        try:
            if isinstance(data, dict):
                prediction_type = data.get('prediction_type')
                target_entity_id = data.get('target_entity_id')
                time_window = data.get('time_window', TimeWindow.MEDIUM_TERM)
                context = data.get('context', {})

                if prediction_type and target_entity_id:
                    request = PredictionRequest(
                        prediction_type=PredictionType(prediction_type),
                        target_entity_id=target_entity_id,
                        time_window=time_window,
                        context=context
                    )

                    return await self.prediction_engine.predict(request)

            elif isinstance(data, PredictionRequest):
                return await self.prediction_engine.predict(data)

            return None

        except Exception as e:
            self.logger.error(f"Erreur traitement requête Prediction: {e}")
            return None

    async def get_shared_data(self) -> Dict[str, Any]:
        """Partage les statistiques du Predictive Engine"""
        if self.prediction_engine:
            return {
                'prediction_stats': self.prediction_engine.get_statistics(),
                'models_performance': {name: predictor.performance_metrics
                                     for name, predictor in self.prediction_engine.predictors.items()},
                'last_update': datetime.now().isoformat()
            }
        return {}

class UncertaintyModule(AIModule):
    """Module de gestion de l'incertitude"""

    def __init__(self):
        super().__init__("uncertainty")
        self.uncertainty_manager: Optional[UncertaintyManager] = None

    async def _initialize_impl(self, config: Dict[str, Any]) -> bool:
        """Initialise l'Uncertainty Manager"""
        try:
            self.logger.info("⚖️ Initialisation Uncertainty Manager...")
            self.uncertainty_manager = UncertaintyManager()

            self._shared_data = {
                'total_decisions': 0,
                'avg_confidence': 0.0,
                'checkpoints_count': 0,
                'last_update': datetime.now().isoformat()
            }

            self.logger.info("✅ Uncertainty Manager initialisé")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation Uncertainty Manager: {e}")
            return False

    async def process(self, data: Any) -> Any:
        """Traite les évaluations d'incertitude"""
        if not self.uncertainty_manager:
            return None

        try:
            if isinstance(data, dict):
                action_type = data.get('type')

                if action_type == 'evaluate_decision':
                    decision_data = data.get('decision_data', {})
                    return await self.uncertainty_manager.evaluate_decision(decision_data)

                elif action_type == 'create_checkpoint':
                    decision_id = data.get('decision_id')
                    state_snapshot = data.get('state_snapshot', {})
                    return await self.uncertainty_manager.create_checkpoint(decision_id, state_snapshot)

                elif action_type == 'rollback_decision':
                    decision_id = data.get('decision_id')
                    return await self.uncertainty_manager.rollback_decision(decision_id)

            return None

        except Exception as e:
            self.logger.error(f"Erreur traitement Uncertainty: {e}")
            return None

    async def get_shared_data(self) -> Dict[str, Any]:
        """Partage les statistiques d'incertitude"""
        if self.uncertainty_manager:
            return {
                'uncertainty_stats': {
                    'total_decisions': getattr(self.uncertainty_manager, 'decision_count', 0),
                    'active_checkpoints': len(getattr(self.uncertainty_manager, 'checkpoints', {}))
                },
                'last_update': datetime.now().isoformat()
            }
        return {}

class DecisionModule(AIModule):
    """Module de décision multi-objectifs avancé"""

    def __init__(self, knowledge_graph: DofusKnowledgeGraph,
                 prediction_engine: PredictiveAnalyticsEngine,
                 uncertainty_manager: UncertaintyManager):
        super().__init__("decision")
        self.knowledge_graph = knowledge_graph
        self.prediction_engine = prediction_engine
        self.uncertainty_manager = uncertainty_manager
        self.decision_engine: Optional[AdvancedDecisionEngine] = None

    async def _initialize_impl(self, config: Dict[str, Any]) -> bool:
        """Initialise le Decision Engine"""
        try:
            self.logger.info("🧠 Initialisation Decision Engine...")
            self.decision_engine = await create_decision_engine(
                self.knowledge_graph, self.prediction_engine, self.uncertainty_manager
            )

            # Mise à jour des données partagées
            self._shared_data = {
                'active_objectives': len(self.decision_engine.active_objectives),
                'decision_metrics': self.decision_engine.get_performance_metrics(),
                'last_update': datetime.now().isoformat()
            }

            self.logger.info(f"✅ Decision Engine initialisé: {len(self.decision_engine.active_objectives)} objectifs")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation Decision Engine: {e}")
            return False

    async def process(self, data: Any) -> Any:
        """Traite les requêtes de décision"""
        if not self.decision_engine:
            return None

        try:
            if isinstance(data, dict):
                action_type = data.get('type')

                if action_type == 'evaluate_actions':
                    actions = data.get('actions', [])
                    return await self.decision_engine.evaluate_action_portfolio(actions)

                elif action_type == 'resolve_conflicts':
                    conflicts = data.get('conflicts', [])
                    return await self.decision_engine.resolve_objective_conflicts(conflicts)

                elif action_type == 'plan_temporal':
                    horizon = data.get('horizon', timedelta(hours=1))
                    return await self.decision_engine.plan_temporal_strategy(horizon)

                elif action_type == 'add_objective':
                    objective = data.get('objective')
                    if objective:
                        return await self.decision_engine.add_objective(objective)

                elif action_type == 'update_progress':
                    obj_id = data.get('objective_id')
                    value = data.get('value')
                    if obj_id and value is not None:
                        return await self.decision_engine.update_objective_progress(obj_id, value)

            return None

        except Exception as e:
            self.logger.error(f"Erreur traitement Decision: {e}")
            return None

    async def get_shared_data(self) -> Dict[str, Any]:
        """Partage les métriques du Decision Engine"""
        if self.decision_engine:
            return {
                'decision_metrics': self.decision_engine.get_performance_metrics(),
                'active_objectives': len(self.decision_engine.active_objectives),
                'available_actions': len(self.decision_engine.available_actions),
                'current_conflicts': len(self.decision_engine.current_conflicts),
                'last_update': datetime.now().isoformat()
            }
        return {}

class EmotionalModule(AIModule):
    """Module de gestion des états émotionnels"""

    def __init__(self):
        super().__init__("emotional")
        self.emotional_manager: Optional[EmotionalStateManager] = None

    async def _initialize_impl(self, config: Dict[str, Any]) -> bool:
        """Initialise l'Emotional State Manager"""
        try:
            self.logger.info("😊 Initialisation Emotional State Manager...")

            # Configuration de personnalité depuis config
            personality_config = config.get('personality', None)
            self.emotional_manager = await create_emotional_manager(personality_config)

            # Mise à jour des données partagées
            self._shared_data = {
                'current_mood': self.emotional_manager.current_mood.value,
                'active_emotions': len(self.emotional_manager.active_emotions),
                'emotional_memory': len(self.emotional_manager.emotional_memory),
                'last_update': datetime.now().isoformat()
            }

            self.logger.info("✅ Emotional State Manager initialisé")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation Emotional Manager: {e}")
            return False

    async def process(self, data: Any) -> Any:
        """Traite les événements émotionnels"""
        if not self.emotional_manager:
            return None

        try:
            if isinstance(data, dict):
                action_type = data.get('type')

                if action_type == 'process_event':
                    event = data.get('event')
                    context = data.get('context', {})
                    if event and isinstance(event, GameEvent):
                        return await self.emotional_manager.process_game_event(event, context)

                elif action_type == 'simulate_mood':
                    return self.emotional_manager.simulate_player_mood()

                elif action_type == 'adjust_risk_tolerance':
                    mood = data.get('mood', self.emotional_manager.current_mood)
                    return self.emotional_manager.adjust_risk_tolerance(mood)

                elif action_type == 'generate_response':
                    situation = data.get('situation', {})
                    return self.emotional_manager.generate_personality_response(situation)

                elif action_type == 'get_motivation':
                    return self.emotional_manager.get_current_motivation()

                elif action_type == 'get_summary':
                    return self.emotional_manager.get_emotional_state_summary()

            return None

        except Exception as e:
            self.logger.error(f"Erreur traitement Emotional: {e}")
            return None

    async def get_shared_data(self) -> Dict[str, Any]:
        """Partage l'état émotionnel"""
        if self.emotional_manager:
            summary = self.emotional_manager.get_emotional_state_summary()
            return {
                'current_mood': summary['current_mood'],
                'active_emotions_count': len(summary['active_emotions']),
                'risk_tolerance': summary['risk_tolerance'],
                'motivation': summary['motivation'],
                'dominant_traits': summary['personality_summary']['dominant_traits'],
                'last_update': datetime.now().isoformat()
            }
        return {}

class SocialIntelligenceModule(AIModule):
    """Module d'intelligence sociale Phase 3"""

    def __init__(self):
        super().__init__("social_intelligence")
        self.social_module: Optional[SocialModule] = None

    async def _initialize_impl(self, config: Dict[str, Any]) -> bool:
        """Initialise le module d'intelligence sociale"""
        try:
            self.logger.info("👥 Initialisation Intelligence Sociale...")
            self.social_module = SocialModule()
            await self.social_module.initialize()

            # Mise à jour des données partagées
            self._shared_data = {
                'social_active': True,
                'social_contexts': [ctx.value for ctx in SocialContext],
                'relationship_types': [rel.value for rel in RelationshipType],
                'last_update': datetime.now().isoformat()
            }

            self.logger.info("✅ Intelligence Sociale initialisée")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation Intelligence Sociale: {e}")
            return False

    async def process(self, data: Any) -> Any:
        """Traite les requêtes sociales"""
        if not self.social_module:
            return None

        try:
            if isinstance(data, dict):
                action_type = data.get('type')

                if action_type == 'process_social_frame':
                    game_state = data.get('game_state', {})
                    return await self.social_module.process_social_frame(game_state)

                elif action_type == 'get_metrics':
                    return await self.social_module.get_social_metrics()

                elif action_type == 'update_player_profile':
                    player_name = data.get('player_name')
                    observations = data.get('observations', {})
                    if player_name:
                        await self.social_module.social_engine.update_player_profile(
                            player_name, observations
                        )
                        return {"status": "updated"}

                elif action_type == 'start_negotiation':
                    player_name = data.get('player_name')
                    item = data.get('item')
                    initial_offer = data.get('initial_offer', 0.0)
                    if player_name and item:
                        strategy = await self.social_module.negotiation_engine.start_negotiation(
                            player_name, item, initial_offer
                        )
                        return {"strategy": strategy.__dict__}

                elif action_type == 'register_account':
                    account_id = data.get('account_id')
                    capabilities = data.get('capabilities', [])
                    if account_id:
                        await self.social_module.multi_agent_coordinator.register_account(
                            account_id, capabilities
                        )
                        return {"status": "registered"}

            return None

        except Exception as e:
            self.logger.error(f"Erreur traitement Social: {e}")
            return None

    async def get_shared_data(self) -> Dict[str, Any]:
        """Partage les données sociales"""
        if self.social_module:
            metrics = await self.social_module.get_social_metrics()
            return {
                'social_metrics': metrics,
                'module_active': self.social_module.is_active,
                'last_update': datetime.now().isoformat()
            }
        return {}

class AdaptiveExecutionEngineModule(AIModule):
    """Module d'exécution adaptative Phase 3"""

    def __init__(self):
        super().__init__("adaptive_execution")
        self.adaptive_module: Optional[AdaptiveExecutionModule] = None

    async def _initialize_impl(self, config: Dict[str, Any]) -> bool:
        """Initialise le module d'exécution adaptative"""
        try:
            self.logger.info("🔄 Initialisation Exécution Adaptative...")
            self.adaptive_module = AdaptiveExecutionModule()

            # Configuration adaptée
            adaptive_config = config.get('adaptive_execution', {})
            await self.adaptive_module.initialize(adaptive_config)

            # Mise à jour des données partagées
            self._shared_data = {
                'adaptive_active': True,
                'execution_styles': [style.value for style in ExecutionStyle],
                'optimization_metrics': [metric.value for metric in OptimizationMetric],
                'last_update': datetime.now().isoformat()
            }

            self.logger.info("✅ Exécution Adaptative initialisée")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation Exécution Adaptative: {e}")
            return False

    async def process(self, data: Any) -> Any:
        """Traite les requêtes d'exécution adaptative"""
        if not self.adaptive_module:
            return None

        try:
            if isinstance(data, dict):
                action_type = data.get('type')

                if action_type == 'process_adaptive_frame':
                    game_state = data.get('game_state', {})
                    return await self.adaptive_module.process_adaptive_frame(game_state)

                elif action_type == 'create_execution_plan':
                    objective = data.get('objective', 'general')
                    context = data.get('context', {})
                    plan = await self.adaptive_module.execution_engine.create_execution_plan(
                        objective, context
                    )
                    return {"plan": plan.__dict__}

                elif action_type == 'get_performance_metrics':
                    return {
                        "metrics": self.adaptive_module.execution_engine.performance_metrics.__dict__
                    }

                elif action_type == 'set_optimization_targets':
                    targets = data.get('targets', {})
                    # Conversion des strings en OptimizationMetric
                    converted_targets = {}
                    for key, value in targets.items():
                        try:
                            metric = OptimizationMetric(key)
                            converted_targets[metric] = value
                        except ValueError:
                            continue

                    await self.adaptive_module.optimizer.set_optimization_targets(converted_targets)
                    return {"status": "targets_set"}

            return None

        except Exception as e:
            self.logger.error(f"Erreur traitement Adaptive: {e}")
            return None

    async def get_shared_data(self) -> Dict[str, Any]:
        """Partage les données d'exécution adaptative"""
        if self.adaptive_module:
            return {
                'module_active': self.adaptive_module.is_active,
                'current_plan': (self.adaptive_module.execution_engine.current_plan.__dict__
                               if self.adaptive_module.execution_engine.current_plan else None),
                'performance_metrics': self.adaptive_module.execution_engine.performance_metrics.__dict__,
                'adaptation_count': self.adaptive_module.execution_engine.performance_metrics.adaptation_count,
                'last_update': datetime.now().isoformat()
            }
        return {}

class MetaEvolutionEngineModule(AIModule):
    """Module de méta-évolution Phase 4"""

    def __init__(self):
        super().__init__("meta_evolution")
        self.meta_evolution_module: Optional[MetaEvolutionModule] = None

    async def _initialize_impl(self, config: Dict[str, Any]) -> bool:
        """Initialise le module de méta-évolution"""
        try:
            self.logger.info("🧬 Initialisation Méta-Évolution...")
            self.meta_evolution_module = MetaEvolutionModule()

            # Configuration du module
            meta_config = config.get("meta_evolution", {})
            await self.meta_evolution_module.initialize(meta_config)

            # Mise à jour des données partagées
            self._shared_data = {
                "meta_evolution_active": True,
                "evolution_triggers": [trigger.value for trigger in EvolutionTrigger],
                "evolution_strategies": [strategy.value for strategy in EvolutionStrategy],
                "last_update": datetime.now().isoformat()
            }

            self.logger.info("✅ Méta-Évolution initialisée")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation Méta-Évolution: {e}")
            return False

    async def process(self, data: Any) -> Any:
        """Traite les requêtes de méta-évolution"""
        if not self.meta_evolution_module:
            return None

        try:
            if isinstance(data, dict):
                action_type = data.get('type')

                if action_type == 'process_evolution_cycle':
                    framework_state = data.get('framework_state', {})
                    return await self.meta_evolution_module.process_evolution_cycle(framework_state)

                elif action_type == 'get_evolution_metrics':
                    return await self.meta_evolution_module.get_evolution_metrics()

                elif action_type == 'trigger_evolution':
                    trigger = data.get('trigger', EvolutionTrigger.SCHEDULED_EVOLUTION)
                    framework_state = data.get('framework_state', {})
                    # Simulation de déclenchement d'évolution
                    return {
                        "evolution_triggered": True,
                        "trigger": trigger.value if hasattr(trigger, 'value') else trigger,
                        "timestamp": datetime.now().isoformat()
                    }

            return None

        except Exception as e:
            self.logger.error(f"Erreur traitement Méta-Évolution: {e}")
            return None

    async def get_shared_data(self) -> Dict[str, Any]:
        """Partage les données de méta-évolution"""
        if self.meta_evolution_module:
            metrics = await self.meta_evolution_module.get_evolution_metrics()
            return {
                'meta_evolution_metrics': metrics,
                'module_active': self.meta_evolution_module.is_active,
                'last_update': datetime.now().isoformat()
            }
        return {}

class GeneticLearningEngineModule(AIModule):
    """Module d'apprentissage génétique Phase 4"""

    def __init__(self):
        super().__init__("genetic_learning")
        self.genetic_learning_module: Optional[GeneticLearningModule] = None

    async def _initialize_impl(self, config: Dict[str, Any]) -> bool:
        """Initialise le module d'apprentissage génétique"""
        try:
            self.logger.info("🧬 Initialisation Apprentissage Génétique...")
            self.genetic_learning_module = GeneticLearningModule()

            # Configuration du module
            genetic_config = config.get("genetic_learning", {})
            await self.genetic_learning_module.initialize(genetic_config)

            # Mise à jour des données partagées
            self._shared_data = {
                "genetic_learning_active": True,
                "gene_types": [gene_type.value for gene_type in GeneType],
                "selection_methods": [method.value for method in SelectionMethod],
                "last_update": datetime.now().isoformat()
            }

            self.logger.info("✅ Apprentissage Génétique initialisé")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation Apprentissage Génétique: {e}")
            return False

    async def process(self, data: Any) -> Any:
        """Traite les requêtes d'apprentissage génétique"""
        if not self.genetic_learning_module:
            return None

        try:
            if isinstance(data, dict):
                action_type = data.get('type')

                if action_type == 'run_learning_cycle':
                    environment_state = data.get('environment_state', {})
                    return await self.genetic_learning_module.run_learning_cycle(environment_state)

                elif action_type == 'get_learning_metrics':
                    return await self.genetic_learning_module.get_learning_metrics()

                elif action_type == 'get_best_configuration':
                    if self.genetic_learning_module.genetic_engine:
                        return await self.genetic_learning_module.genetic_engine.get_best_configuration()
                    return {"error": "No genetic engine available"}

                elif action_type == 'evolve_population':
                    environment_state = data.get('environment_state', {})
                    if self.genetic_learning_module.genetic_engine:
                        return await self.genetic_learning_module.genetic_engine.evolve_generation(environment_state)
                    return {"error": "No genetic engine available"}

            return None

        except Exception as e:
            self.logger.error(f"Erreur traitement Apprentissage Génétique: {e}")
            return None

    async def get_shared_data(self) -> Dict[str, Any]:
        """Partage les données d'apprentissage génétique"""
        if self.genetic_learning_module:
            metrics = await self.genetic_learning_module.get_learning_metrics()
            return {
                'genetic_learning_metrics': metrics,
                'module_active': self.genetic_learning_module.is_active,
                'last_update': datetime.now().isoformat()
            }
        return {}

class StateTrackingModule(AIModule):
    """Module de suivi d'état multi-dimensionnel"""

    def __init__(self):
        super().__init__("state_tracking")
        self.state_tracker: Optional[MultiDimensionalStateTracker] = None

    async def _initialize_impl(self, config: Dict[str, Any]) -> bool:
        """Initialise le State Tracker"""
        try:
            self.logger.info("📊 Initialisation State Tracker...")
            self.state_tracker = await create_state_tracker()

            # Démarrage du suivi continu
            if await self.state_tracker.start_tracking():
                self.logger.info("✅ State Tracker démarré")

                # Mise à jour des données partagées
                self._shared_data = {
                    'tracking_active': True,
                    'state_levels': [level.value for level in StateLevel],
                    'last_update': datetime.now().isoformat()
                }

                return True
            else:
                self.logger.error("Échec démarrage State Tracker")
                return False

        except Exception as e:
            self.logger.error(f"Erreur initialisation State Tracker: {e}")
            return False

    async def process(self, data: Any) -> Any:
        """Traite les mises à jour d'état"""
        if not self.state_tracker:
            return None

        try:
            if isinstance(data, dict):
                action_type = data.get('type')

                if action_type == 'update_immediate':
                    new_state = data.get('state')
                    if isinstance(new_state, GameState):
                        return await self.state_tracker.update_immediate_state(new_state)

                elif action_type == 'update_tactical':
                    updates = data.get('updates', {})
                    return await self.state_tracker.update_tactical_context(updates)

                elif action_type == 'update_strategic':
                    planning_updates = data.get('planning_updates', {})
                    return await self.state_tracker.update_strategic_planning(planning_updates)

                elif action_type == 'update_meta':
                    meta_updates = data.get('meta_updates', {})
                    return await self.state_tracker.update_meta_objectives(meta_updates)

                elif action_type == 'get_summary':
                    level = data.get('level')
                    return self.state_tracker.get_state_summary(level)

                elif action_type == 'predict_evolution':
                    level = data.get('level', StateLevel.IMMEDIATE)
                    horizon = data.get('horizon', timedelta(minutes=5))
                    return self.state_tracker.predict_state_evolution(level, horizon)

            return None

        except Exception as e:
            self.logger.error(f"Erreur traitement State Tracking: {e}")
            return None

    async def shutdown(self):
        """Arrêt propre du module"""
        if self.state_tracker:
            await self.state_tracker.stop_tracking()
        await super().shutdown()

    async def get_shared_data(self) -> Dict[str, Any]:
        """Partage l'état du suivi"""
        if self.state_tracker:
            summary = self.state_tracker.get_state_summary()
            return {
                'overall_health': summary.get('overall_health', {}),
                'tracking_metrics': summary.get('tracking_metrics', {}),
                'immediate_health': summary.get('immediate', {}).get('health_status', False),
                'tactical_efficiency': summary.get('tactical', {}).get('performance_score', 0.0),
                'strategic_progress': summary.get('strategic', {}).get('planning', {}).get('overall_progress', 0.0),
                'last_update': datetime.now().isoformat()
            }
        return {}

# Fonctions utilitaires pour l'initialisation

async def create_ai_framework(config_path: Optional[str] = None) -> MetaOrchestrator:
    """Crée et configure l'orchestrateur IA principal avec tous les modules"""
    orchestrator = MetaOrchestrator(config_path)

    try:
        # Initialisation du Knowledge Graph (requis pour les autres modules)
        knowledge_module = KnowledgeModule()
        await orchestrator.register_module(knowledge_module)

        # Récupération du Knowledge Graph pour les autres modules
        knowledge_graph = knowledge_module.knowledge_graph

        if knowledge_graph:
            # Module de prédiction
            prediction_module = PredictionModule(knowledge_graph)
            await orchestrator.register_module(prediction_module)

            # Module d'incertitude
            uncertainty_module = UncertaintyModule()
            await orchestrator.register_module(uncertainty_module)

            # Modules Phase 2 - Cerveau Multi-Dimensionnel

            # Module de décision avancé
            decision_module = DecisionModule(knowledge_graph, prediction_module.prediction_engine, uncertainty_module.uncertainty_manager)
            await orchestrator.register_module(decision_module)

            # Module émotionnel
            emotional_module = EmotionalModule()
            await orchestrator.register_module(emotional_module)

            # Module de suivi d'état
            state_tracking_module = StateTrackingModule()
            await orchestrator.register_module(state_tracking_module)

            # Modules Phase 3 - Exécution Adaptative & Sociale

            # Module d'intelligence sociale
            social_module = SocialIntelligenceModule()
            await orchestrator.register_module(social_module)

            # Module d'exécution adaptative
            adaptive_execution_module = AdaptiveExecutionEngineModule()
            await orchestrator.register_module(adaptive_execution_module)

            # Modules Phase 4 - Méta-Évolution & Auto-Amélioration

            # Module de méta-évolution
            meta_evolution_module = MetaEvolutionEngineModule()
            await orchestrator.register_module(meta_evolution_module)

            # Module d'apprentissage génétique
            genetic_learning_module = GeneticLearningEngineModule()
            await orchestrator.register_module(genetic_learning_module)

            orchestrator.logger.info("✅ Tous les modules IA (Phases 1-4) ont été enregistrés avec succès")
        else:
            orchestrator.logger.error("❌ Impossible d'initialiser les modules - Knowledge Graph manquant")

    except Exception as e:
        orchestrator.logger.error(f"Erreur lors de l'enregistrement des modules: {e}")

    return orchestrator

def save_default_config(config_path: str = "config/ai_config.json"):
    """Sauvegarde la configuration par défaut"""
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    orchestrator = MetaOrchestrator()
    config = orchestrator._get_default_config()

    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Configuration par défaut sauvegardée: {config_file}")

# Interface CLI
async def main():
    """Interface en ligne de commande"""
    import argparse

    parser = argparse.ArgumentParser(description="Core AI Framework - IA DOFUS")
    parser.add_argument("--init", action="store_true", help="Initialise la configuration")
    parser.add_argument("--config", default="config/ai_config.json", help="Chemin config")
    parser.add_argument("--test", action="store_true", help="Test du framework")

    args = parser.parse_args()

    if args.init:
        save_default_config(args.config)
        print("🚀 Configuration initialisée. Lancez ensuite:")
        print(f"python {__file__} --test --config {args.config}")
        return

    if args.test:
        print("🧪 Test du Core AI Framework...")

        orchestrator = await create_ai_framework(args.config)

        if await orchestrator.start():
            print("✅ Framework démarré avec succès")

            # Test de fonctionnement pendant 10 secondes
            await asyncio.sleep(10)

            status = orchestrator.get_status()
            print(f"📊 Status: {json.dumps(status, indent=2)}")

            await orchestrator.stop()
            print("✅ Test terminé avec succès")
        else:
            print("❌ Échec démarrage framework")

if __name__ == "__main__":
    asyncio.run(main())