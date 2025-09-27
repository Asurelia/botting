"""
Adaptateur de Détection Dynamique
Système intelligent qui adapte automatiquement la stratégie de détection
selon le contexte du jeu, les performances et les conditions en temps réel
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import threading
import json

# Import des modules internes
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from engine.module_interface import IAnalysisModule, ModuleStatus

# Import conditionnel des détecteurs
try:
    from .hybrid_detector import HybridDetector, HybridConfig, DetectionMethod, HybridDetection
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

try:
    from .vision_orchestrator import VisionOrchestrator, VisionConfig
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False

logger = logging.getLogger(__name__)

class GameContext(Enum):
    """Contextes de jeu possibles"""
    MENU = "menu"
    COMBAT = "combat"
    EXPLORATION = "exploration"
    HARVESTING = "harvesting"
    TRADING = "trading"
    DUNGEON = "dungeon"
    PVP = "pvp"
    UNKNOWN = "unknown"

class PerformanceProfile(Enum):
    """Profils de performance"""
    HIGH_PERFORMANCE = "high"      # GPU puissant, priorité qualité
    BALANCED = "balanced"          # Équilibre qualité/vitesse
    LOW_LATENCY = "low_latency"    # Priorité vitesse
    ENERGY_SAVER = "energy_saver"  # Économie d'énergie

@dataclass
class AdaptationRule:
    """Règle d'adaptation dynamique"""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    target_method: DetectionMethod
    priority: int = 1
    min_confidence: float = 0.6
    description: str = ""

@dataclass
class ContextAnalysis:
    """Analyse du contexte actuel"""
    game_context: GameContext
    confidence: float
    detected_elements: List[str]
    ui_density: float  # Densité d'éléments UI (0-1)
    movement_detected: bool
    animation_detected: bool
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AdapterConfig:
    """Configuration de l'adaptateur"""
    # Profil de performance
    performance_profile: PerformanceProfile = PerformanceProfile.BALANCED

    # Adaptation automatique
    enable_context_adaptation: bool = True
    enable_performance_adaptation: bool = True
    enable_predictive_switching: bool = True

    # Fenêtres d'analyse
    context_analysis_window: int = 30  # Secondes
    performance_window: int = 100      # Nombre de détections

    # Seuils d'adaptation
    performance_threshold: float = 0.15
    context_confidence_threshold: float = 0.7

    # Stratégies par contexte
    context_strategies: Dict[GameContext, DetectionMethod] = field(default_factory=lambda: {
        GameContext.MENU: DetectionMethod.TEMPLATE,
        GameContext.COMBAT: DetectionMethod.HYBRID,
        GameContext.EXPLORATION: DetectionMethod.YOLO,
        GameContext.HARVESTING: DetectionMethod.YOLO,
        GameContext.TRADING: DetectionMethod.TEMPLATE,
        GameContext.DUNGEON: DetectionMethod.HYBRID,
        GameContext.PVP: DetectionMethod.HYBRID,
        GameContext.UNKNOWN: DetectionMethod.AUTO
    })

    # Profils de performance
    performance_profiles: Dict[PerformanceProfile, Dict[str, Any]] = field(default_factory=lambda: {
        PerformanceProfile.HIGH_PERFORMANCE: {
            'primary_method': DetectionMethod.HYBRID,
            'yolo_confidence': 0.5,
            'template_confidence': 0.7,
            'enable_parallel': True,
            'max_workers': 6
        },
        PerformanceProfile.BALANCED: {
            'primary_method': DetectionMethod.AUTO,
            'yolo_confidence': 0.6,
            'template_confidence': 0.75,
            'enable_parallel': True,
            'max_workers': 4
        },
        PerformanceProfile.LOW_LATENCY: {
            'primary_method': DetectionMethod.YOLO,
            'yolo_confidence': 0.7,
            'template_confidence': 0.8,
            'enable_parallel': False,
            'max_workers': 2
        },
        PerformanceProfile.ENERGY_SAVER: {
            'primary_method': DetectionMethod.TEMPLATE,
            'yolo_confidence': 0.8,
            'template_confidence': 0.6,
            'enable_parallel': False,
            'max_workers': 1
        }
    })

class ContextAnalyzer:
    """Analyseur de contexte de jeu"""

    def __init__(self):
        self.ui_templates = {
            'combat_spell_bar': 'combat',
            'combat_timeline': 'combat',
            'inventory_grid': 'trading',
            'shop_window': 'trading',
            'resource_gauge': 'harvesting',
            'minimap': 'exploration',
            'dungeon_info': 'dungeon'
        }

        self.movement_detector = self._create_movement_detector()
        self.previous_frame = None

    def _create_movement_detector(self):
        """Crée un détecteur de mouvement"""
        return cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=50
        )

    def analyze_context(self, image: np.ndarray,
                       recent_detections: List[Dict[str, Any]]) -> ContextAnalysis:
        """Analyse le contexte du jeu depuis une image"""
        try:
            # Détection d'éléments UI spécifiques
            ui_elements = self._detect_ui_elements(image, recent_detections)
            ui_density = len(ui_elements) / max(1, len(self.ui_templates))

            # Détection de mouvement
            movement_detected = self._detect_movement(image)

            # Détection d'animations
            animation_detected = self._detect_animations(image)

            # Analyse du contexte basée sur les éléments détectés
            context, confidence = self._infer_context(ui_elements, movement_detected)

            return ContextAnalysis(
                game_context=context,
                confidence=confidence,
                detected_elements=ui_elements,
                ui_density=ui_density,
                movement_detected=movement_detected,
                animation_detected=animation_detected
            )

        except Exception as e:
            logger.error(f"Erreur analyse contexte: {e}")
            return ContextAnalysis(
                game_context=GameContext.UNKNOWN,
                confidence=0.0,
                detected_elements=[],
                ui_density=0.0,
                movement_detected=False,
                animation_detected=False
            )

    def _detect_ui_elements(self, image: np.ndarray,
                           recent_detections: List[Dict[str, Any]]) -> List[str]:
        """Détecte les éléments UI présents"""
        detected_elements = []

        # Basé sur les détections récentes
        for detection in recent_detections:
            object_type = detection.get('object_type', '')
            if 'ui_' in object_type:
                detected_elements.append(object_type)

        # Détection basique de formes géométriques UI
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Détection de rectangles (fenêtres)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            large_rectangles = 0
            for contour in contours:
                if cv2.contourArea(contour) > 5000:  # Seuil pour grandes fenêtres
                    large_rectangles += 1

            if large_rectangles > 2:
                detected_elements.append('multiple_windows')

        except Exception as e:
            logger.debug(f"Erreur détection UI: {e}")

        return list(set(detected_elements))

    def _detect_movement(self, image: np.ndarray) -> bool:
        """Détecte le mouvement dans l'image"""
        try:
            if self.previous_frame is None:
                self.previous_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                return False

            current_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Détection de mouvement par différence de frames
            diff = cv2.absdiff(self.previous_frame, current_frame)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

            movement_pixels = cv2.countNonZero(thresh)
            total_pixels = thresh.shape[0] * thresh.shape[1]

            movement_ratio = movement_pixels / total_pixels

            self.previous_frame = current_frame

            return movement_ratio > 0.05  # 5% de pixels en mouvement

        except Exception as e:
            logger.debug(f"Erreur détection mouvement: {e}")
            return False

    def _detect_animations(self, image: np.ndarray) -> bool:
        """Détecte les animations (effets visuels)"""
        try:
            # Détection basique basée sur la saturation des couleurs
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]

            # Les effets magiques/animations ont souvent des couleurs saturées
            high_saturation = np.sum(saturation > 200)
            total_pixels = saturation.shape[0] * saturation.shape[1]

            saturation_ratio = high_saturation / total_pixels

            return saturation_ratio > 0.02  # 2% de pixels très saturés

        except Exception as e:
            logger.debug(f"Erreur détection animations: {e}")
            return False

    def _infer_context(self, ui_elements: List[str], movement: bool) -> Tuple[GameContext, float]:
        """Infère le contexte du jeu"""
        scores = defaultdict(float)

        # Score basé sur les éléments UI
        for element in ui_elements:
            if 'combat' in element or 'spell' in element:
                scores[GameContext.COMBAT] += 0.3
            elif 'inventory' in element or 'shop' in element:
                scores[GameContext.TRADING] += 0.3
            elif 'resource' in element:
                scores[GameContext.HARVESTING] += 0.3
            elif 'minimap' in element:
                scores[GameContext.EXPLORATION] += 0.2

        # Score basé sur le mouvement
        if movement:
            scores[GameContext.EXPLORATION] += 0.2
            scores[GameContext.COMBAT] += 0.1
        else:
            scores[GameContext.MENU] += 0.2
            scores[GameContext.TRADING] += 0.1

        if not scores:
            return GameContext.UNKNOWN, 0.0

        best_context = max(scores, key=scores.get)
        confidence = scores[best_context]

        return best_context, min(1.0, confidence)

class DetectionAdapter(IAnalysisModule):
    """
    Adaptateur de détection dynamique
    Orchestre intelligemment les différentes stratégies de détection
    selon le contexte et les performances
    """

    def __init__(self, name: str = "detection_adapter", config: AdapterConfig = None):
        super().__init__(name)

        self.config = config or AdapterConfig()
        self.logger = logging.getLogger(f"{__name__}.DetectionAdapter")

        # Composants
        self.hybrid_detector: Optional[HybridDetector] = None
        self.vision_orchestrator: Optional[VisionOrchestrator] = None
        self.context_analyzer = ContextAnalyzer()

        # État
        self.current_context = GameContext.UNKNOWN
        self.current_strategy = DetectionMethod.AUTO
        self.last_adaptation = datetime.now()

        # Historique
        self.context_history = deque(maxlen=100)
        self.performance_history = deque(maxlen=self.config.performance_window)
        self.recent_detections = deque(maxlen=50)

        # Règles d'adaptation
        self.adaptation_rules = self._create_adaptation_rules()

        # Threading
        self._adaptation_lock = threading.Lock()

        # Statistiques
        self.stats = {
            'total_adaptations': 0,
            'context_switches': 0,
            'performance_switches': 0,
            'avg_adaptation_time': 0.0,
            'strategy_distribution': defaultdict(int),
            'context_distribution': defaultdict(int)
        }

    def _create_adaptation_rules(self) -> List[AdaptationRule]:
        """Crée les règles d'adaptation dynamique"""
        return [
            AdaptationRule(
                name="combat_emergency",
                condition=lambda data: (
                    data.get('context') == GameContext.COMBAT and
                    data.get('movement') and
                    data.get('recent_fps', 0) < 15
                ),
                target_method=DetectionMethod.YOLO,
                priority=3,
                description="Passage en YOLO pour combat intense"
            ),
            AdaptationRule(
                name="menu_optimization",
                condition=lambda data: (
                    data.get('context') == GameContext.MENU and
                    data.get('ui_density', 0) > 0.7
                ),
                target_method=DetectionMethod.TEMPLATE,
                priority=2,
                description="Template matching pour interfaces"
            ),
            AdaptationRule(
                name="exploration_boost",
                condition=lambda data: (
                    data.get('context') == GameContext.EXPLORATION and
                    data.get('movement') and
                    not data.get('ui_density', 0) > 0.5
                ),
                target_method=DetectionMethod.YOLO,
                priority=2,
                description="YOLO pour exploration dynamique"
            ),
            AdaptationRule(
                name="low_performance_fallback",
                condition=lambda data: data.get('avg_fps', 60) < 10,
                target_method=DetectionMethod.TEMPLATE,
                priority=1,
                description="Fallback template si performances faibles"
            )
        ]

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialise l'adaptateur de détection"""
        try:
            self.logger.info("Initialisation de l'adaptateur de détection")

            # Initialisation du détecteur hybride
            if HYBRID_AVAILABLE:
                hybrid_config = HybridConfig()
                self.hybrid_detector = HybridDetector(config=hybrid_config)

                if self.hybrid_detector.initialize(config.get('hybrid', {})):
                    self.logger.info("✅ Détecteur hybride initialisé")
                else:
                    self.logger.warning("❌ Échec initialisation détecteur hybride")

            # Initialisation de l'orchestrateur
            if ORCHESTRATOR_AVAILABLE:
                vision_config = VisionConfig()
                self.vision_orchestrator = VisionOrchestrator(config=vision_config)

                if self.vision_orchestrator.initialize(config.get('vision', {})):
                    self.logger.info("✅ Orchestrateur vision initialisé")
                else:
                    self.logger.warning("❌ Échec initialisation orchestrateur")

            # Vérification minimum requis
            if not (self.hybrid_detector or self.vision_orchestrator):
                self.logger.error("Aucun détecteur disponible")
                self.set_error("Aucun détecteur initialisé")
                return False

            # Application du profil de performance
            self._apply_performance_profile()

            self.status = ModuleStatus.ACTIVE
            self.logger.info("✅ Adaptateur de détection initialisé")

            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation adaptateur: {e}")
            self.set_error(str(e))
            return False

    def _apply_performance_profile(self):
        """Applique le profil de performance configuré"""
        try:
            profile_config = self.config.performance_profiles[self.config.performance_profile]

            # Configuration du détecteur hybride
            if self.hybrid_detector:
                if hasattr(self.hybrid_detector.config, 'enable_parallel_processing'):
                    self.hybrid_detector.config.enable_parallel_processing = profile_config['enable_parallel']
                if hasattr(self.hybrid_detector.config, 'max_workers'):
                    self.hybrid_detector.config.max_workers = profile_config['max_workers']

            # Configuration des seuils
            self.config.context_confidence_threshold = profile_config.get('confidence_threshold', 0.7)

            self.logger.info(f"Profil appliqué: {self.config.performance_profile.value}")

        except Exception as e:
            self.logger.error(f"Erreur application profil: {e}")

    def detect_objects_adaptive(self, image: np.ndarray, zone: str = "center_game",
                               target_classes: List[str] = None) -> List[HybridDetection]:
        """
        Détection d'objets avec adaptation dynamique

        Args:
            image: Image à analyser
            zone: Zone d'analyse
            target_classes: Classes spécifiques à chercher

        Returns:
            Liste des détections adaptées
        """
        start_time = time.perf_counter()

        try:
            # Analyse du contexte si activée
            if self.config.enable_context_adaptation:
                context_analysis = self.context_analyzer.analyze_context(
                    image, list(self.recent_detections)
                )
                self._update_context(context_analysis)

            # Adaptation de stratégie si nécessaire
            if self._should_adapt():
                self._adapt_strategy()

            # Détection avec la stratégie actuelle
            detections = self._execute_detection(image, zone, target_classes)

            # Mise à jour de l'historique
            processing_time = time.perf_counter() - start_time
            self._update_performance_history(processing_time, len(detections))

            # Sauvegarde des détections récentes
            for detection in detections:
                self.recent_detections.append({
                    'object_type': detection.object_type,
                    'confidence': detection.confidence,
                    'timestamp': detection.timestamp,
                    'method': detection.primary_method.value
                })

            self.logger.debug(f"Adaptateur: {len(detections)} détections en {processing_time:.3f}s")

            return detections

        except Exception as e:
            self.logger.error(f"Erreur détection adaptative: {e}")
            return []

    def _execute_detection(self, image: np.ndarray, zone: str,
                          target_classes: List[str]) -> List[HybridDetection]:
        """Exécute la détection avec la stratégie actuelle"""
        if self.hybrid_detector and self.hybrid_detector.is_active():
            return self.hybrid_detector.detect_objects(image, zone, target_classes)
        elif self.vision_orchestrator and self.vision_orchestrator.is_active():
            # Conversion des résultats de l'orchestrateur
            results = self.vision_orchestrator.find_objects(image, target_classes or [])
            return self._convert_orchestrator_results(results)
        else:
            return []

    def _convert_orchestrator_results(self, results: List[Dict[str, Any]]) -> List[HybridDetection]:
        """Convertit les résultats de l'orchestrateur vers le format HybridDetection"""
        hybrid_detections = []

        for result in results:
            try:
                detection = HybridDetection(
                    object_type=result.get('object_type', 'unknown'),
                    confidence=result.get('confidence', 0.0),
                    position=result.get('position', (0, 0)),
                    bounding_box=result.get('bounding_box', (0, 0, 0, 0)),
                    primary_method=DetectionMethod.AUTO,  # Déterminé par l'orchestrateur
                    additional_data=result.get('additional_data', {})
                )
                hybrid_detections.append(detection)

            except Exception as e:
                self.logger.debug(f"Erreur conversion résultat: {e}")

        return hybrid_detections

    def _update_context(self, context_analysis: ContextAnalysis):
        """Met à jour le contexte actuel"""
        with self._adaptation_lock:
            if context_analysis.confidence >= self.config.context_confidence_threshold:
                if self.current_context != context_analysis.game_context:
                    self.logger.info(f"Changement contexte: {self.current_context.value} -> {context_analysis.game_context.value}")
                    self.current_context = context_analysis.game_context
                    self.stats['context_switches'] += 1

            self.context_history.append(context_analysis)
            self.stats['context_distribution'][context_analysis.game_context.value] += 1

    def _should_adapt(self) -> bool:
        """Détermine si une adaptation est nécessaire"""
        now = datetime.now()

        # Limite de fréquence d'adaptation
        if (now - self.last_adaptation).total_seconds() < 1.0:
            return False

        # Vérification des règles d'adaptation
        current_data = self._get_current_data()

        for rule in self.adaptation_rules:
            if rule.condition(current_data):
                target_strategy = rule.target_method
                if target_strategy != self.current_strategy:
                    return True

        # Adaptation basée sur les performances
        if self.config.enable_performance_adaptation and len(self.performance_history) > 10:
            recent_performance = np.mean([p['processing_time'] for p in list(self.performance_history)[-10:]])
            if recent_performance > self.config.performance_threshold:
                return True

        return False

    def _get_current_data(self) -> Dict[str, Any]:
        """Récupère les données actuelles pour l'évaluation des règles"""
        data = {
            'context': self.current_context,
            'strategy': self.current_strategy,
            'movement': False,
            'ui_density': 0.0,
            'recent_fps': 0,
            'avg_fps': 0
        }

        if self.context_history:
            latest_context = self.context_history[-1]
            data['movement'] = latest_context.movement_detected
            data['ui_density'] = latest_context.ui_density

        if self.performance_history:
            recent_times = [p['processing_time'] for p in list(self.performance_history)[-10:]]
            if recent_times:
                avg_time = np.mean(recent_times)
                data['recent_fps'] = 1.0 / avg_time if avg_time > 0 else 0
                data['avg_fps'] = data['recent_fps']

        return data

    def _adapt_strategy(self):
        """Adapte la stratégie selon les conditions actuelles"""
        with self._adaptation_lock:
            start_time = time.perf_counter()

            current_data = self._get_current_data()
            new_strategy = self._determine_optimal_strategy(current_data)

            if new_strategy != self.current_strategy:
                old_strategy = self.current_strategy
                self.current_strategy = new_strategy

                # Application de la nouvelle stratégie
                self._apply_strategy(new_strategy)

                # Mise à jour des statistiques
                adaptation_time = time.perf_counter() - start_time
                self.stats['total_adaptations'] += 1
                self.stats['avg_adaptation_time'] = (
                    self.stats['avg_adaptation_time'] * 0.9 + adaptation_time * 0.1
                )
                self.stats['strategy_distribution'][new_strategy.value] += 1

                self.last_adaptation = datetime.now()

                self.logger.info(f"Adaptation: {old_strategy.value} -> {new_strategy.value} ({adaptation_time:.3f}s)")

    def _determine_optimal_strategy(self, current_data: Dict[str, Any]) -> DetectionMethod:
        """Détermine la stratégie optimale basée sur les données actuelles"""
        # Priorité aux règles d'adaptation
        applicable_rules = []
        for rule in self.adaptation_rules:
            if rule.condition(current_data):
                applicable_rules.append(rule)

        if applicable_rules:
            # Tri par priorité
            applicable_rules.sort(key=lambda r: r.priority, reverse=True)
            return applicable_rules[0].target_method

        # Stratégie basée sur le contexte
        context_strategy = self.config.context_strategies.get(
            current_data['context'], DetectionMethod.AUTO
        )

        return context_strategy

    def _apply_strategy(self, strategy: DetectionMethod):
        """Applique une stratégie spécifique"""
        try:
            if self.hybrid_detector:
                # Configuration du détecteur hybride selon la stratégie
                if strategy == DetectionMethod.YOLO:
                    self.hybrid_detector.current_strategy = DetectionMethod.YOLO
                elif strategy == DetectionMethod.TEMPLATE:
                    self.hybrid_detector.current_strategy = DetectionMethod.TEMPLATE
                elif strategy == DetectionMethod.HYBRID:
                    self.hybrid_detector.current_strategy = DetectionMethod.HYBRID
                else:  # AUTO
                    self.hybrid_detector.current_strategy = DetectionMethod.AUTO

            if self.vision_orchestrator:
                # Configuration de l'orchestrateur
                strategy_mapping = {
                    DetectionMethod.YOLO: 'yolo',
                    DetectionMethod.TEMPLATE: 'template',
                    DetectionMethod.HYBRID: 'hybrid',
                    DetectionMethod.AUTO: 'auto'
                }

                orchestrator_strategy = strategy_mapping.get(strategy, 'auto')
                if hasattr(self.vision_orchestrator, 'set_strategy'):
                    self.vision_orchestrator.set_strategy(orchestrator_strategy)

        except Exception as e:
            self.logger.error(f"Erreur application stratégie: {e}")

    def _update_performance_history(self, processing_time: float, detection_count: int):
        """Met à jour l'historique de performance"""
        performance_entry = {
            'processing_time': processing_time,
            'detection_count': detection_count,
            'fps': 1.0 / processing_time if processing_time > 0 else 0,
            'timestamp': datetime.now()
        }

        self.performance_history.append(performance_entry)

    def force_strategy(self, strategy: DetectionMethod) -> bool:
        """Force l'utilisation d'une stratégie spécifique"""
        try:
            with self._adaptation_lock:
                old_strategy = self.current_strategy
                self.current_strategy = strategy
                self._apply_strategy(strategy)

                self.logger.info(f"Stratégie forcée: {old_strategy.value} -> {strategy.value}")
                return True

        except Exception as e:
            self.logger.error(f"Erreur force stratégie: {e}")
            return False

    def set_performance_profile(self, profile: PerformanceProfile):
        """Change le profil de performance"""
        self.config.performance_profile = profile
        self._apply_performance_profile()
        self.logger.info(f"Profil de performance: {profile.value}")

    def get_adaptation_report(self) -> Dict[str, Any]:
        """Génère un rapport d'adaptation détaillé"""
        return {
            'current_state': {
                'context': self.current_context.value,
                'strategy': self.current_strategy.value,
                'performance_profile': self.config.performance_profile.value
            },
            'statistics': self.stats,
            'recent_performance': {
                'avg_fps': np.mean([p['fps'] for p in list(self.performance_history)[-10:]]) if self.performance_history else 0,
                'avg_detections': np.mean([p['detection_count'] for p in list(self.performance_history)[-10:]]) if self.performance_history else 0
            },
            'context_history': [
                {
                    'context': c.game_context.value,
                    'confidence': c.confidence,
                    'timestamp': c.timestamp.isoformat()
                }
                for c in list(self.context_history)[-10:]
            ]
        }

    def handle_event(self, event: Any) -> bool:
        """Gestion des événements"""
        return False

    def update(self, game_state: Any) -> Optional[Dict[str, Any]]:
        """Mise à jour de l'adaptateur"""
        try:
            if not self.is_active():
                return None

            return {
                "shared_data": {
                    "adapter_stats": self.stats,
                    "current_context": self.current_context.value,
                    "current_strategy": self.current_strategy.value,
                    "performance_profile": self.config.performance_profile.value
                },
                "module_status": "active"
            }

        except Exception as e:
            self.logger.error(f"Erreur update adaptateur: {e}")
            return None

    def cleanup(self) -> None:
        """Nettoyage des ressources"""
        self.logger.info("Arrêt de l'adaptateur de détection")

        if self.hybrid_detector:
            self.hybrid_detector.cleanup()

        if self.vision_orchestrator:
            self.vision_orchestrator.cleanup()

        self.status = ModuleStatus.INACTIVE
        self.logger.info("Adaptateur de détection arrêté")