"""
Module d'intégration hybride YOLO + Template Matching
Système intelligent qui combine le meilleur des deux approches selon le contexte
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import des modules internes
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from engine.module_interface import IAnalysisModule, ModuleStatus

# Import conditionnel des détecteurs
try:
    from .yolo_detector import DofusYOLODetector, YOLOConfig, YOLODetection
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from .template_matcher import TemplateMatcher, TemplateMatch
    TEMPLATE_AVAILABLE = True
except ImportError:
    TEMPLATE_AVAILABLE = False

logger = logging.getLogger(__name__)

class DetectionMethod(Enum):
    """Méthodes de détection disponibles"""
    YOLO = "yolo"
    TEMPLATE = "template"
    HYBRID = "hybrid"
    AUTO = "auto"

@dataclass
class HybridConfig:
    """Configuration du système hybride"""
    # Stratégies par zone
    zone_strategies: Dict[str, DetectionMethod] = field(default_factory=lambda: {
        "center_game": DetectionMethod.HYBRID,
        "ui_area": DetectionMethod.TEMPLATE,
        "minimap": DetectionMethod.TEMPLATE,
        "combat_interface": DetectionMethod.HYBRID
    })

    # Seuils de confiance
    yolo_confidence_threshold: float = 0.6
    template_confidence_threshold: float = 0.75
    consensus_threshold: float = 0.8

    # Performance
    enable_parallel_processing: bool = True
    max_workers: int = 4

    # Fusion
    enable_cross_validation: bool = True
    overlap_threshold: float = 0.5

    # Adaptation automatique
    enable_adaptive_strategy: bool = True
    performance_window: int = 50
    adaptation_threshold: float = 0.3

@dataclass
class HybridDetection:
    """Détection unifiée du système hybride"""
    object_type: str
    confidence: float
    position: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]

    # Métadonnées de détection
    primary_method: DetectionMethod
    yolo_confidence: Optional[float] = None
    template_confidence: Optional[float] = None
    consensus_score: Optional[float] = None

    # Validation
    cross_validated: bool = False
    validation_methods: List[str] = field(default_factory=list)

    timestamp: datetime = field(default_factory=datetime.now)
    additional_data: Dict[str, Any] = field(default_factory=dict)

class PerformanceMonitor:
    """Moniteur de performance pour adaptation automatique"""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.performance_data = {
            'yolo': {
                'times': [],
                'accuracies': [],
                'detection_counts': []
            },
            'template': {
                'times': [],
                'accuracies': [],
                'detection_counts': []
            },
            'hybrid': {
                'times': [],
                'consensus_rates': [],
                'total_detections': []
            }
        }

    def record_performance(self, method: str, execution_time: float,
                         accuracy: float = None, detection_count: int = 0):
        """Enregistre les performances d'une méthode"""
        if method not in self.performance_data:
            return

        data = self.performance_data[method]
        data['times'].append(execution_time)

        if accuracy is not None:
            data['accuracies'].append(accuracy)

        if 'detection_counts' in data:
            data['detection_counts'].append(detection_count)
        elif 'total_detections' in data:
            data['total_detections'].append(detection_count)

        # Limitation de la fenêtre
        for key in data:
            if len(data[key]) > self.window_size:
                data[key] = data[key][-self.window_size:]

    def get_best_method(self) -> DetectionMethod:
        """Détermine la meilleure méthode basée sur les performances"""
        scores = {}

        for method, data in self.performance_data.items():
            if not data['times'] or method == 'hybrid':
                continue

            # Score combiné: vitesse + précision
            avg_time = np.mean(data['times'])
            speed_score = min(1.0, 10.0 / avg_time)  # Normalisation à 10 FPS

            accuracy_score = np.mean(data['accuracies']) if data['accuracies'] else 0.5

            # Score final (pondération 50/50)
            scores[method] = 0.5 * speed_score + 0.5 * accuracy_score

        if not scores:
            return DetectionMethod.AUTO

        best_method = max(scores, key=scores.get)
        return DetectionMethod.YOLO if best_method == 'yolo' else DetectionMethod.TEMPLATE

class HybridDetector(IAnalysisModule):
    """
    Détecteur hybride intelligent combinant YOLO et Template Matching
    S'adapte automatiquement selon les performances et le contexte
    """

    def __init__(self, name: str = "hybrid_detector", config: HybridConfig = None):
        super().__init__(name)

        self.config = config or HybridConfig()
        self.logger = logging.getLogger(f"{__name__}.HybridDetector")

        # Détecteurs
        self.yolo_detector: Optional[DofusYOLODetector] = None
        self.template_matcher: Optional[TemplateMatcher] = None

        # État des détecteurs
        self.yolo_available = False
        self.template_available = False

        # Performance et adaptation
        self.performance_monitor = PerformanceMonitor()
        self.current_strategy = DetectionMethod.AUTO

        # Threading pour parallélisation
        self.thread_pool = None
        if self.config.enable_parallel_processing:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)

        # Cache des résultats récents
        self.detection_cache = {}
        self.cache_duration = 0.1  # 100ms

        # Statistiques
        self.stats = {
            'total_detections': 0,
            'yolo_detections': 0,
            'template_detections': 0,
            'hybrid_detections': 0,
            'consensus_achieved': 0,
            'strategy_adaptations': 0,
            'avg_processing_time': 0.0
        }

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialise le détecteur hybride"""
        try:
            self.logger.info("Initialisation du détecteur hybride")

            success_count = 0

            # Initialisation YOLO
            if YOLO_AVAILABLE:
                try:
                    yolo_config = config.get('yolo', {})
                    self.yolo_detector = DofusYOLODetector()

                    if self.yolo_detector.initialize(yolo_config):
                        self.yolo_available = True
                        success_count += 1
                        self.logger.info("✅ YOLO detector initialisé")
                    else:
                        self.logger.warning("❌ Échec initialisation YOLO")

                except Exception as e:
                    self.logger.error(f"Erreur YOLO: {e}")
            else:
                self.logger.warning("YOLO non disponible")

            # Initialisation Template Matcher
            if TEMPLATE_AVAILABLE:
                try:
                    template_config = config.get('template', {})
                    self.template_matcher = TemplateMatcher()

                    if self.template_matcher.initialize(template_config):
                        self.template_available = True
                        success_count += 1
                        self.logger.info("✅ Template matcher initialisé")
                    else:
                        self.logger.warning("❌ Échec initialisation Template Matcher")

                except Exception as e:
                    self.logger.error(f"Erreur Template Matcher: {e}")
            else:
                self.logger.warning("Template Matcher non disponible")

            # Vérification minimum requis
            if success_count == 0:
                self.logger.error("Aucun détecteur disponible")
                self.set_error("Aucun détecteur initialisé")
                return False

            # Adaptation de la stratégie selon les détecteurs disponibles
            self._adapt_strategies_to_available_detectors()

            self.status = ModuleStatus.ACTIVE
            self.logger.info(f"Détecteur hybride initialisé ({success_count}/2 détecteurs)")

            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation hybride: {e}")
            self.set_error(str(e))
            return False

    def _adapt_strategies_to_available_detectors(self):
        """Adapte les stratégies selon les détecteurs disponibles"""
        if not self.yolo_available and not self.template_available:
            self.current_strategy = DetectionMethod.AUTO
            return

        if self.yolo_available and self.template_available:
            self.current_strategy = DetectionMethod.HYBRID
        elif self.yolo_available:
            self.current_strategy = DetectionMethod.YOLO
            # Mise à jour des stratégies de zone
            for zone in self.config.zone_strategies:
                self.config.zone_strategies[zone] = DetectionMethod.YOLO
        elif self.template_available:
            self.current_strategy = DetectionMethod.TEMPLATE
            # Mise à jour des stratégies de zone
            for zone in self.config.zone_strategies:
                self.config.zone_strategies[zone] = DetectionMethod.TEMPLATE

        self.logger.info(f"Stratégie adaptée: {self.current_strategy.value}")

    def detect_objects(self, image: np.ndarray, zone: str = "center_game",
                      target_classes: List[str] = None) -> List[HybridDetection]:
        """
        Détection d'objets avec stratégie hybride

        Args:
            image: Image à analyser
            zone: Zone d'analyse
            target_classes: Classes spécifiques à chercher

        Returns:
            Liste des détections hybrides
        """
        start_time = time.perf_counter()

        try:
            # Stratégie pour cette zone
            strategy = self.config.zone_strategies.get(zone, self.current_strategy)

            # Adaptation automatique si activée
            if self.config.enable_adaptive_strategy:
                strategy = self._get_adaptive_strategy(zone)

            detections = []

            if strategy == DetectionMethod.HYBRID:
                detections = self._hybrid_detect(image, target_classes)
            elif strategy == DetectionMethod.YOLO:
                detections = self._yolo_detect(image, target_classes)
            elif strategy == DetectionMethod.TEMPLATE:
                detections = self._template_detect(image, target_classes)
            else:  # AUTO
                detections = self._auto_detect(image, target_classes)

            # Mise à jour des performances
            processing_time = time.perf_counter() - start_time
            self._update_performance_stats(strategy, processing_time, len(detections))

            self.logger.debug(f"Zone {zone}: {len(detections)} détections avec {strategy.value}")

            return detections

        except Exception as e:
            self.logger.error(f"Erreur détection hybride: {e}")
            return []

    def _hybrid_detect(self, image: np.ndarray, target_classes: List[str]) -> List[HybridDetection]:
        """Détection hybride combinant YOLO et Template Matching"""
        if not (self.yolo_available and self.template_available):
            return self._auto_detect(image, target_classes)

        try:
            results = {}

            # Exécution parallèle si possible
            if self.thread_pool:
                futures = {}

                # YOLO
                if self.yolo_available:
                    future = self.thread_pool.submit(self._yolo_detect_internal, image, target_classes)
                    futures['yolo'] = future

                # Template
                if self.template_available:
                    future = self.thread_pool.submit(self._template_detect_internal, image, target_classes)
                    futures['template'] = future

                # Collecte des résultats
                for method, future in futures.items():
                    try:
                        result = future.result(timeout=3.0)
                        results[method] = result
                    except Exception as e:
                        self.logger.error(f"Erreur {method} parallèle: {e}")
                        results[method] = []
            else:
                # Exécution séquentielle
                if self.yolo_available:
                    results['yolo'] = self._yolo_detect_internal(image, target_classes)
                if self.template_available:
                    results['template'] = self._template_detect_internal(image, target_classes)

            # Fusion intelligente des résultats
            hybrid_detections = self._merge_detections(results)

            self.stats['hybrid_detections'] += len(hybrid_detections)

            return hybrid_detections

        except Exception as e:
            self.logger.error(f"Erreur détection hybride: {e}")
            return []

    def _yolo_detect(self, image: np.ndarray, target_classes: List[str]) -> List[HybridDetection]:
        """Détection avec YOLO uniquement"""
        if not self.yolo_available:
            return self._template_detect(image, target_classes)

        detections = self._yolo_detect_internal(image, target_classes)
        self.stats['yolo_detections'] += len(detections)
        return detections

    def _template_detect(self, image: np.ndarray, target_classes: List[str]) -> List[HybridDetection]:
        """Détection avec Template Matching uniquement"""
        if not self.template_available:
            return self._yolo_detect(image, target_classes)

        detections = self._template_detect_internal(image, target_classes)
        self.stats['template_detections'] += len(detections)
        return detections

    def _auto_detect(self, image: np.ndarray, target_classes: List[str]) -> List[HybridDetection]:
        """Détection automatique avec le meilleur détecteur disponible"""
        if self.yolo_available and self.template_available:
            # Choix basé sur les performances
            best_method = self.performance_monitor.get_best_method()

            if best_method == DetectionMethod.YOLO:
                return self._yolo_detect(image, target_classes)
            else:
                return self._template_detect(image, target_classes)

        elif self.yolo_available:
            return self._yolo_detect(image, target_classes)
        elif self.template_available:
            return self._template_detect(image, target_classes)
        else:
            return []

    def _yolo_detect_internal(self, image: np.ndarray, target_classes: List[str]) -> List[HybridDetection]:
        """Détection YOLO interne"""
        try:
            yolo_detections = self.yolo_detector.detect(image, target_classes)

            hybrid_detections = []
            for detection in yolo_detections:
                if detection.confidence >= self.config.yolo_confidence_threshold:
                    hybrid_det = HybridDetection(
                        object_type=detection.class_name,
                        confidence=detection.confidence,
                        position=detection.center,
                        bounding_box=detection.bbox,
                        primary_method=DetectionMethod.YOLO,
                        yolo_confidence=detection.confidence,
                        additional_data={
                            'class_id': detection.class_id,
                            'area': detection.area
                        }
                    )
                    hybrid_detections.append(hybrid_det)

            return hybrid_detections

        except Exception as e:
            self.logger.error(f"Erreur YOLO interne: {e}")
            return []

    def _template_detect_internal(self, image: np.ndarray, target_classes: List[str]) -> List[HybridDetection]:
        """Détection Template Matching interne"""
        try:
            # Conversion des classes vers catégories template
            categories = self._map_classes_to_template_categories(target_classes)

            template_matches = []
            for category in categories:
                matches = self.template_matcher.find_templates(
                    image,
                    category=category,
                    min_confidence=self.config.template_confidence_threshold
                )
                template_matches.extend(matches)

            hybrid_detections = []
            for match in template_matches:
                if match.get('confidence', 0) >= self.config.template_confidence_threshold:
                    hybrid_det = HybridDetection(
                        object_type=match.get('template_name', 'unknown'),
                        confidence=match.get('confidence', 0),
                        position=match.get('position', (0, 0)),
                        bounding_box=match.get('bounding_box', (0, 0, 0, 0)),
                        primary_method=DetectionMethod.TEMPLATE,
                        template_confidence=match.get('confidence', 0),
                        additional_data=match.get('additional_data', {})
                    )
                    hybrid_detections.append(hybrid_det)

            return hybrid_detections

        except Exception as e:
            self.logger.error(f"Erreur Template interne: {e}")
            return []

    def _map_classes_to_template_categories(self, target_classes: List[str]) -> List[str]:
        """Mappe les classes YOLO vers les catégories template"""
        if not target_classes:
            return ['resources', 'monsters', 'npcs', 'ui']

        mapping = {
            'resource_tree': 'resources',
            'resource_ore': 'resources',
            'resource_plant': 'resources',
            'monster': 'monsters',
            'archmonster': 'monsters',
            'npc': 'npcs',
            'ui_button': 'ui',
            'ui_window': 'ui',
            'player': 'characters'
        }

        categories = set()
        for class_name in target_classes:
            category = mapping.get(class_name, 'unknown')
            if category != 'unknown':
                categories.add(category)

        return list(categories) if categories else ['resources', 'monsters']

    def _merge_detections(self, results: Dict[str, List[HybridDetection]]) -> List[HybridDetection]:
        """Fusionne intelligemment les détections de différentes méthodes"""
        if len(results) <= 1:
            return list(results.values())[0] if results else []

        yolo_detections = results.get('yolo', [])
        template_detections = results.get('template', [])

        # Étape 1: Cross-validation des détections
        if self.config.enable_cross_validation:
            yolo_detections = self._cross_validate_detections(yolo_detections, template_detections, 'yolo')
            template_detections = self._cross_validate_detections(template_detections, yolo_detections, 'template')

        merged_detections = []
        used_template_indices = set()

        # Pour chaque détection YOLO, chercher correspondance template
        for yolo_det in yolo_detections:
            best_match = None
            best_match_idx = None
            best_overlap = 0

            for idx, template_det in enumerate(template_detections):
                if idx in used_template_indices:
                    continue

                # Vérification de correspondance de type
                if self._objects_match(yolo_det.object_type, template_det.object_type):
                    overlap = self._calculate_overlap(yolo_det, template_det)

                    if overlap > best_overlap and overlap > self.config.overlap_threshold:
                        best_overlap = overlap
                        best_match = template_det
                        best_match_idx = idx

            if best_match:
                # Fusion des détections concordantes
                consensus_det = self._create_consensus_detection(yolo_det, best_match, best_overlap)
                merged_detections.append(consensus_det)
                used_template_indices.add(best_match_idx)
                self.stats['consensus_achieved'] += 1
            else:
                # Détection YOLO seule - appliquer filtrage de confiance
                if yolo_det.confidence >= self._get_adaptive_threshold('yolo', yolo_det.object_type):
                    merged_detections.append(yolo_det)

        # Ajout des détections template non matchées
        for idx, template_det in enumerate(template_detections):
            if idx not in used_template_indices:
                if template_det.confidence >= self._get_adaptive_threshold('template', template_det.object_type):
                    merged_detections.append(template_det)

        # Élimination des doublons résiduels
        merged_detections = self._remove_duplicate_detections(merged_detections)

        return merged_detections

    def _objects_match(self, yolo_type: str, template_type: str) -> bool:
        """Vérifie si deux types d'objets correspondent"""
        # Mapping simple - peut être étendu
        matches = {
            'resource_tree': ['tree', 'wood', 'ash'],
            'resource_ore': ['ore', 'iron', 'copper'],
            'resource_plant': ['plant', 'wheat', 'barley'],
            'monster': ['monster', 'mob'],
            'npc': ['npc', 'character'],
            'player': ['player', 'character']
        }

        for yolo_class, template_variants in matches.items():
            if yolo_type == yolo_class:
                return any(variant in template_type.lower() for variant in template_variants)

        return yolo_type.lower() == template_type.lower()

    def _calculate_overlap(self, det1: HybridDetection, det2: HybridDetection) -> float:
        """Calcule le chevauchement entre deux détections"""
        try:
            box1 = det1.bounding_box
            box2 = det2.bounding_box

            # Intersection
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            if x1 >= x2 or y1 >= y2:
                return 0.0

            intersection = (x2 - x1) * (y2 - y1)

            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

            union = area1 + area2 - intersection

            return intersection / union if union > 0 else 0.0

        except:
            return 0.0

    def _create_consensus_detection(self, yolo_det: HybridDetection,
                                  template_det: HybridDetection,
                                  overlap: float) -> HybridDetection:
        """Crée une détection consensus basée sur deux détections"""
        # Pondération par confiance
        yolo_weight = yolo_det.confidence
        template_weight = template_det.confidence
        total_weight = yolo_weight + template_weight

        if total_weight == 0:
            total_weight = 1.0
            yolo_weight = template_weight = 0.5
        else:
            yolo_weight /= total_weight
            template_weight /= total_weight

        # Position pondérée
        consensus_pos = (
            int(yolo_det.position[0] * yolo_weight + template_det.position[0] * template_weight),
            int(yolo_det.position[1] * yolo_weight + template_det.position[1] * template_weight)
        )

        # Confiance combinée
        consensus_confidence = (yolo_det.confidence + template_det.confidence) / 2

        # Score de consensus basé sur l'overlap et les confidences
        consensus_score = overlap * consensus_confidence

        return HybridDetection(
            object_type=yolo_det.object_type,  # Priorité YOLO pour le type
            confidence=min(1.0, consensus_confidence * 1.1),  # Bonus consensus
            position=consensus_pos,
            bounding_box=yolo_det.bounding_box,  # Priorité YOLO pour bbox
            primary_method=DetectionMethod.HYBRID,
            yolo_confidence=yolo_det.confidence,
            template_confidence=template_det.confidence,
            consensus_score=consensus_score,
            cross_validated=True,
            validation_methods=['yolo', 'template'],
            additional_data={
                'overlap': overlap,
                'yolo_data': yolo_det.additional_data,
                'template_data': template_det.additional_data
            }
        )

    def _get_adaptive_strategy(self, zone: str) -> DetectionMethod:
        """Détermine la stratégie adaptative pour une zone"""
        base_strategy = self.config.zone_strategies.get(zone, self.current_strategy)

        # Si performance monitoring indique un changement
        best_method = self.performance_monitor.get_best_method()

        # Adaptation seulement si différence significative
        if best_method != DetectionMethod.AUTO:
            return best_method

        return base_strategy

    def _update_performance_stats(self, strategy: DetectionMethod,
                                processing_time: float, detection_count: int):
        """Met à jour les statistiques de performance"""
        method_name = strategy.value

        # Enregistrement dans le moniteur
        self.performance_monitor.record_performance(
            method_name, processing_time, detection_count=detection_count
        )

        # Statistiques globales
        self.stats['total_detections'] += detection_count

        # Moyenne mobile du temps de traitement
        alpha = 0.1
        self.stats['avg_processing_time'] = (
            alpha * processing_time +
            (1 - alpha) * self.stats['avg_processing_time']
        )

    def get_performance_report(self) -> Dict[str, Any]:
        """Génère un rapport de performance détaillé"""
        return {
            'current_strategy': self.current_strategy.value,
            'detectors_available': {
                'yolo': self.yolo_available,
                'template': self.template_available
            },
            'statistics': self.stats,
            'zone_strategies': {zone: strategy.value for zone, strategy in self.config.zone_strategies.items()},
            'performance_data': self.performance_monitor.performance_data
        }

    def set_zone_strategy(self, zone: str, strategy: DetectionMethod):
        """Configure la stratégie pour une zone spécifique"""
        self.config.zone_strategies[zone] = strategy
        self.logger.info(f"Stratégie zone {zone}: {strategy.value}")

    def handle_event(self, event: Any) -> bool:
        """Gestion des événements"""
        return False

    def update(self, game_state: Any) -> Optional[Dict[str, Any]]:
        """Mise à jour du détecteur hybride"""
        try:
            if not self.is_active():
                return None

            return {
                "shared_data": {
                    "hybrid_stats": self.stats,
                    "current_strategy": self.current_strategy.value,
                    "detectors_status": {
                        "yolo_available": self.yolo_available,
                        "template_available": self.template_available
                    }
                },
                "module_status": "active"
            }

        except Exception as e:
            self.logger.error(f"Erreur update hybride: {e}")
            return None

    def get_state(self) -> Dict[str, Any]:
        """État du détecteur hybride"""
        return {
            "status": self.status.value,
            "current_strategy": self.current_strategy.value,
            "yolo_available": self.yolo_available,
            "template_available": self.template_available,
            "stats": self.stats
        }

    def cleanup(self) -> None:
        """Nettoyage des ressources"""
        self.logger.info("Arrêt du détecteur hybride")

        # Arrêt des détecteurs
        if self.yolo_detector:
            self.yolo_detector.cleanup()

        if self.template_matcher:
            self.template_matcher.cleanup()

        # Arrêt du thread pool
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)

        self.status = ModuleStatus.INACTIVE
        self.logger.info("Détecteur hybride arrêté")

    def _cross_validate_detections(self, primary_detections: List[HybridDetection],
                                 secondary_detections: List[HybridDetection],
                                 primary_method: str) -> List[HybridDetection]:
        """Cross-validation des détections entre méthodes"""
        if not self.config.enable_cross_validation:
            return primary_detections

        validated_detections = []

        for primary_det in primary_detections:
            validation_score = self._calculate_validation_score(primary_det, secondary_detections)

            # Ajustement de la confiance basé sur la validation
            if validation_score > 0.5:  # Validation positive
                primary_det.confidence = min(1.0, primary_det.confidence * (1 + validation_score * 0.2))
                primary_det.cross_validated = True
                primary_det.validation_methods.append(f"cross_validated_with_{secondary_detections[0].primary_method.value if secondary_detections else 'none'}")
            elif validation_score < 0.2:  # Validation négative
                primary_det.confidence *= 0.8  # Réduction de confiance

            validated_detections.append(primary_det)

        return validated_detections

    def _calculate_validation_score(self, detection: HybridDetection,
                                  reference_detections: List[HybridDetection]) -> float:
        """Calcule un score de validation croisée"""
        if not reference_detections:
            return 0.5  # Score neutre

        max_score = 0.0

        for ref_det in reference_detections:
            # Score basé sur proximité spatiale
            spatial_score = self._calculate_spatial_proximity(detection, ref_det)

            # Score basé sur similarité de type
            type_score = 1.0 if self._objects_match(detection.object_type, ref_det.object_type) else 0.0

            # Score combiné
            combined_score = 0.7 * spatial_score + 0.3 * type_score
            max_score = max(max_score, combined_score)

        return max_score

    def _calculate_spatial_proximity(self, det1: HybridDetection, det2: HybridDetection) -> float:
        """Calcule la proximité spatiale entre deux détections"""
        try:
            pos1 = det1.position
            pos2 = det2.position

            # Distance euclidienne
            distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

            # Normalisation basée sur la taille des objets
            bbox1 = det1.bounding_box
            bbox2 = det2.bounding_box

            avg_size = (
                (bbox1[2] - bbox1[0] + bbox1[3] - bbox1[1]) +
                (bbox2[2] - bbox2[0] + bbox2[3] - bbox2[1])
            ) / 4

            if avg_size == 0:
                avg_size = 50  # Taille par défaut

            # Score de proximité (1.0 = très proche, 0.0 = très éloigné)
            proximity_score = max(0.0, 1.0 - distance / (avg_size * 2))

            return proximity_score

        except:
            return 0.0

    def _get_adaptive_threshold(self, method: str, object_type: str) -> float:
        """Calcule un seuil adaptatif basé sur les performances historiques"""
        base_thresholds = {
            'yolo': self.config.yolo_confidence_threshold,
            'template': self.config.template_confidence_threshold
        }

        base_threshold = base_thresholds.get(method, 0.6)

        # Ajustement basé sur les performances récentes
        performance_data = self.performance_monitor.performance_data.get(method, {})
        recent_accuracies = performance_data.get('accuracies', [])

        if recent_accuracies:
            recent_accuracy = np.mean(recent_accuracies[-10:])  # 10 dernières mesures

            if recent_accuracy > 0.8:
                # Performance élevée -> seuil plus permissif
                adaptive_threshold = base_threshold * 0.9
            elif recent_accuracy < 0.6:
                # Performance faible -> seuil plus strict
                adaptive_threshold = base_threshold * 1.1
            else:
                adaptive_threshold = base_threshold
        else:
            adaptive_threshold = base_threshold

        # Ajustement par type d'objet
        type_adjustments = {
            'archmonster': 0.95,    # Plus strict pour les archimonstres
            'resource_tree': 0.85,  # Plus permissif pour les ressources
            'resource_ore': 0.85,
            'ui_button': 0.9,       # Strict pour l'UI
            'player': 0.88          # Équilibré pour les joueurs
        }

        adjustment = type_adjustments.get(object_type, 1.0)
        return adaptive_threshold * adjustment

    def _remove_duplicate_detections(self, detections: List[HybridDetection]) -> List[HybridDetection]:
        """Élimine les détections en double"""
        if len(detections) <= 1:
            return detections

        # Tri par confiance décroissante
        sorted_detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        unique_detections = []

        for detection in sorted_detections:
            is_duplicate = False

            for existing in unique_detections:
                # Vérification de duplication basée sur overlap
                overlap = self._calculate_overlap(detection, existing)

                # Si overlap élevé et même type -> duplication
                if overlap > 0.7 and self._objects_match(detection.object_type, existing.object_type):
                    is_duplicate = True

                    # Mise à jour de la détection existante si celle-ci est meilleure
                    if detection.cross_validated and not existing.cross_validated:
                        # Remplacement par la détection cross-validée
                        unique_detections.remove(existing)
                        unique_detections.append(detection)
                        is_duplicate = False
                    elif detection.consensus_score and existing.consensus_score:
                        if detection.consensus_score > existing.consensus_score:
                            unique_detections.remove(existing)
                            unique_detections.append(detection)
                            is_duplicate = False

                    break

            if not is_duplicate:
                unique_detections.append(detection)

        return unique_detections

    def force_strategy_adaptation(self) -> bool:
        """Force une adaptation de stratégie basée sur les performances"""
        try:
            best_method = self.performance_monitor.get_best_method()

            if best_method != self.current_strategy and best_method != DetectionMethod.AUTO:
                old_strategy = self.current_strategy
                self.current_strategy = best_method

                # Mise à jour des stratégies de zone
                for zone in self.config.zone_strategies:
                    if self.config.zone_strategies[zone] == old_strategy:
                        self.config.zone_strategies[zone] = best_method

                self.stats['strategy_adaptations'] += 1
                self.logger.info(f"Adaptation forcée: {old_strategy.value} -> {best_method.value}")

                return True

            return False

        except Exception as e:
            self.logger.error(f"Erreur adaptation forcée: {e}")
            return False

    def get_detection_quality_metrics(self) -> Dict[str, float]:
        """Calcule des métriques de qualité des détections"""
        try:
            total_detections = self.stats['total_detections']

            if total_detections == 0:
                return {'quality_score': 0.0, 'consensus_rate': 0.0, 'cross_validation_rate': 0.0}

            consensus_rate = self.stats['consensus_achieved'] / total_detections

            # Estimation du taux de cross-validation
            cross_validation_rate = 0.0
            if hasattr(self, '_recent_cross_validations'):
                cross_validation_rate = len(self._recent_cross_validations) / min(total_detections, 100)

            # Score de qualité combiné
            quality_score = (
                0.4 * consensus_rate +
                0.3 * cross_validation_rate +
                0.3 * min(1.0, total_detections / 1000)  # Score d'expérience
            )

            return {
                'quality_score': quality_score,
                'consensus_rate': consensus_rate,
                'cross_validation_rate': cross_validation_rate,
                'avg_processing_time': self.stats['avg_processing_time']
            }

        except Exception as e:
            self.logger.error(f"Erreur calcul métriques qualité: {e}")
            return {'quality_score': 0.0, 'consensus_rate': 0.0, 'cross_validation_rate': 0.0}