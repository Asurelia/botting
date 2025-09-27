"""
Orchestrateur de Vision DOFUS - Intégration YOLO + Template Matching
Système hybride intelligent qui combine le meilleur des deux approches
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Import des modules internes
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from engine.module_interface import IAnalysisModule, ModuleStatus
from engine.event_bus import EventType, EventPriority

# Import des modules de vision
from .yolo_detector import DofusYOLODetector, YOLOConfig, YOLODetection, YOLOTemplateAdapter
from .template_matcher import TemplateMatcher, TemplateMatch
from .screen_analyzer import ScreenAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class VisionConfig:
    """Configuration globale du système de vision"""
    # Stratégies de vision
    primary_method: str = "yolo"  # "yolo", "template", "hybrid"
    fallback_method: str = "template"
    confidence_threshold_yolo: float = 0.6
    confidence_threshold_template: float = 0.75

    # Performance
    parallel_processing: bool = True
    max_workers: int = 4
    enable_caching: bool = True
    cache_duration: float = 0.2  # secondes

    # Validation croisée
    enable_cross_validation: bool = True
    validation_threshold: float = 0.8
    require_consensus: bool = False

    # Adaptation automatique
    enable_adaptive_switching: bool = True
    performance_window: int = 100  # nombre de détections pour évaluation
    switch_threshold: float = 0.3  # différence de performance pour switch

@dataclass
class UnifiedDetection:
    """Détection unifiée combinant YOLO et Template Matching"""
    object_type: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: int

    # Métadonnées de détection
    detection_method: str  # "yolo", "template", "consensus"
    yolo_confidence: Optional[float] = None
    template_confidence: Optional[float] = None
    validation_score: Optional[float] = None

    # Tracking
    track_id: Optional[int] = None

    # Timestamps
    first_detected: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    # Données additionnelles
    additional_data: Dict[str, Any] = field(default_factory=dict)

    def to_legacy_format(self) -> Dict[str, Any]:
        """Conversion au format legacy pour compatibilité"""
        return {
            'template_name': self.object_type,
            'confidence': self.confidence,
            'position': self.center,
            'bounding_box': self.bbox,
            'method': self.detection_method,
            'additional_data': self.additional_data
        }

class PerformanceTracker:
    """Suivi des performances des différentes méthodes de détection"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

        # Métriques par méthode
        self.metrics = {
            'yolo': {
                'detection_times': [],
                'confidence_scores': [],
                'detection_counts': [],
                'accuracy_estimates': [],
                'false_positive_rate': 0.0,
                'total_detections': 0
            },
            'template': {
                'detection_times': [],
                'confidence_scores': [],
                'detection_counts': [],
                'accuracy_estimates': [],
                'false_positive_rate': 0.0,
                'total_detections': 0
            },
            'hybrid': {
                'detection_times': [],
                'consensus_rate': [],
                'validation_success_rate': [],
                'total_validations': 0
            }
        }

        # Performance globale
        self.overall_stats = {
            'preferred_method': 'yolo',
            'last_evaluation': time.time(),
            'adaptation_count': 0
        }

    def record_detection(self, method: str, detection_time: float,
                        detection_count: int, avg_confidence: float,
                        estimated_accuracy: float = None):
        """Enregistre une session de détection"""
        if method not in self.metrics:
            return

        metrics = self.metrics[method]

        # Ajout des nouvelles métriques
        metrics['detection_times'].append(detection_time)
        metrics['confidence_scores'].append(avg_confidence)
        metrics['detection_counts'].append(detection_count)

        if estimated_accuracy is not None:
            metrics['accuracy_estimates'].append(estimated_accuracy)

        metrics['total_detections'] += detection_count

        # Limite de la fenêtre
        for key in ['detection_times', 'confidence_scores', 'detection_counts', 'accuracy_estimates']:
            if len(metrics[key]) > self.window_size:
                metrics[key] = metrics[key][-self.window_size:]

    def get_performance_summary(self, method: str) -> Dict[str, float]:
        """Résumé des performances d'une méthode"""
        if method not in self.metrics:
            return {}

        metrics = self.metrics[method]

        if not metrics['detection_times']:
            return {'available': False}

        return {
            'available': True,
            'avg_detection_time': np.mean(metrics['detection_times']),
            'avg_confidence': np.mean(metrics['confidence_scores']),
            'avg_detection_count': np.mean(metrics['detection_counts']),
            'estimated_accuracy': np.mean(metrics['accuracy_estimates']) if metrics['accuracy_estimates'] else 0.0,
            'total_detections': metrics['total_detections'],
            'fps': 1.0 / max(np.mean(metrics['detection_times']), 0.001)
        }

    def should_adapt_strategy(self, config: VisionConfig) -> Tuple[bool, str]:
        """Détermine s'il faut adapter la stratégie de détection"""
        if not config.enable_adaptive_switching:
            return False, config.primary_method

        # Évaluation seulement si assez de données
        yolo_perf = self.get_performance_summary('yolo')
        template_perf = self.get_performance_summary('template')

        if not (yolo_perf.get('available') and template_perf.get('available')):
            return False, config.primary_method

        # Score combiné (vitesse + précision + confiance)
        def calculate_score(perf):
            if not perf.get('available'):
                return 0.0

            speed_score = min(perf['fps'] / 10, 1.0)  # Normalise à 10 FPS max
            accuracy_score = perf['estimated_accuracy']
            confidence_score = perf['avg_confidence']

            return (speed_score * 0.3 + accuracy_score * 0.5 + confidence_score * 0.2)

        yolo_score = calculate_score(yolo_perf)
        template_score = calculate_score(template_perf)

        # Décision d'adaptation
        current_method = self.overall_stats['preferred_method']

        if current_method == 'yolo' and template_score > yolo_score + config.switch_threshold:
            self.overall_stats['adaptation_count'] += 1
            return True, 'template'
        elif current_method == 'template' and yolo_score > template_score + config.switch_threshold:
            self.overall_stats['adaptation_count'] += 1
            return True, 'yolo'
        elif abs(yolo_score - template_score) < 0.1:  # Scores similaires -> hybride
            return True, 'hybrid'

        return False, current_method

class VisionOrchestrator(IAnalysisModule):
    """
    Orchestrateur principal du système de vision DOFUS
    Coordonne YOLO, Template Matching et Screen Analysis de manière intelligente
    """

    def __init__(self, name: str = "vision_orchestrator", config: VisionConfig = None):
        super().__init__(name)

        self.config = config or VisionConfig()
        self.logger = logging.getLogger(f"{__name__}.VisionOrchestrator")

        # Modules de vision
        self.yolo_detector: Optional[DofusYOLODetector] = None
        self.template_matcher: Optional[TemplateMatcher] = None
        self.screen_analyzer: Optional[ScreenAnalyzer] = None
        self.yolo_adapter: Optional[YOLOTemplateAdapter] = None

        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        self.current_strategy = self.config.primary_method

        # Cache et optimisations
        self.detection_cache = {}
        self.cache_timestamps = {}

        # Threading pour traitement parallèle
        self.thread_pool = None
        if self.config.parallel_processing:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)

        # Validation croisée
        self.validation_buffer = defaultdict(list)

        # Statistiques
        self.stats = {
            'total_analyses': 0,
            'yolo_analyses': 0,
            'template_analyses': 0,
            'hybrid_analyses': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'consensus_validations': 0,
            'strategy_adaptations': 0,
            'avg_analysis_time': 0.0
        }

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialise l'orchestrateur de vision"""
        try:
            self.logger.info("Initialisation de l'orchestrateur de vision")

            # Mise à jour de la configuration
            if 'vision_orchestrator' in config:
                vision_config = config['vision_orchestrator']
                for key, value in vision_config.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)

            # Initialisation des modules de vision
            success_count = 0

            # 1. YOLO Detector
            try:
                yolo_config = YOLOConfig()
                self.yolo_detector = DofusYOLODetector(config=yolo_config)

                if self.yolo_detector.initialize(config):
                    self.yolo_adapter = YOLOTemplateAdapter(self.yolo_detector)
                    success_count += 1
                    self.logger.info("YOLO Detector initialisé avec succès")
                else:
                    self.logger.warning("Échec initialisation YOLO Detector")
            except Exception as e:
                self.logger.error(f"Erreur initialisation YOLO: {e}")

            # 2. Template Matcher
            try:
                self.template_matcher = TemplateMatcher()
                if self.template_matcher.initialize(config):
                    success_count += 1
                    self.logger.info("Template Matcher initialisé avec succès")
                else:
                    self.logger.warning("Échec initialisation Template Matcher")
            except Exception as e:
                self.logger.error(f"Erreur initialisation Template Matcher: {e}")

            # 3. Screen Analyzer
            try:
                self.screen_analyzer = ScreenAnalyzer()
                if self.screen_analyzer.initialize(config):
                    success_count += 1
                    self.logger.info("Screen Analyzer initialisé avec succès")
                else:
                    self.logger.warning("Échec initialisation Screen Analyzer")
            except Exception as e:
                self.logger.error(f"Erreur initialisation Screen Analyzer: {e}")

            # Vérification du minimum requis
            if success_count == 0:
                self.logger.error("Aucun module de vision initialisé")
                self.set_error("Aucun module de vision disponible")
                return False

            # Adaptation de la stratégie selon les modules disponibles
            self._adapt_strategy_to_available_modules()

            self.status = ModuleStatus.ACTIVE
            self.logger.info(f"Orchestrateur de vision initialisé ({success_count}/3 modules)")
            self.logger.info(f"Stratégie active: {self.current_strategy}")

            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation orchestrateur: {e}")
            self.set_error(str(e))
            return False

    def _adapt_strategy_to_available_modules(self):
        """Adapte la stratégie selon les modules disponibles"""
        available_modules = []

        if self.yolo_detector and self.yolo_detector.model_loaded:
            available_modules.append('yolo')

        if self.template_matcher and self.template_matcher.is_active():
            available_modules.append('template')

        # Choix de stratégie
        if self.config.primary_method in available_modules:
            self.current_strategy = self.config.primary_method
        elif self.config.fallback_method in available_modules:
            self.current_strategy = self.config.fallback_method
        elif available_modules:
            self.current_strategy = available_modules[0]
        else:
            self.current_strategy = 'none'
            self.logger.warning("Aucun module de détection disponible")

    def analyze(self, image: np.ndarray, target_categories: List[str] = None,
                roi: str = "full_screen") -> Dict[str, Any]:
        """
        Analyse unifiée d'une image avec tous les modules disponibles

        Args:
            image: Image à analyser
            target_categories: Catégories spécifiques à chercher
            roi: Région d'intérêt

        Returns:
            Résultats de détection unifiés
        """
        start_time = time.perf_counter()

        try:
            # Vérification du cache
            cache_key = self._generate_cache_key(image, target_categories, roi)

            if self.config.enable_caching:
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    self.stats['cache_hits'] += 1
                    return cached_result
                else:
                    self.stats['cache_misses'] += 1

            # Analyse selon la stratégie courante
            if self.current_strategy == 'hybrid':
                result = self._hybrid_analysis(image, target_categories, roi)
            elif self.current_strategy == 'yolo' and self.yolo_detector:
                result = self._yolo_analysis(image, target_categories, roi)
            elif self.current_strategy == 'template' and self.template_matcher:
                result = self._template_analysis(image, target_categories, roi)
            else:
                # Fallback au premier module disponible
                result = self._fallback_analysis(image, target_categories, roi)

            # Post-traitement
            result = self._post_process_results(result)

            # Mise à jour des statistiques
            analysis_time = time.perf_counter() - start_time
            self._update_stats(result, analysis_time)

            # Mise en cache
            if self.config.enable_caching:
                self._cache_result(cache_key, result)

            # Évaluation et adaptation de stratégie
            self._evaluate_and_adapt_strategy(result, analysis_time)

            return result

        except Exception as e:
            self.logger.error(f"Erreur analyse orchestrateur: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(),
                "strategy_used": self.current_strategy
            }

    def _hybrid_analysis(self, image: np.ndarray, target_categories: List[str],
                        roi: str) -> Dict[str, Any]:
        """Analyse hybride combinant YOLO et Template Matching"""
        try:
            futures = {}
            results = {}

            # Lancement en parallèle si possible
            if self.thread_pool:
                # YOLO
                if self.yolo_detector:
                    future = self.thread_pool.submit(self._yolo_analysis, image, target_categories, roi)
                    futures['yolo'] = future

                # Template Matching
                if self.template_matcher:
                    future = self.thread_pool.submit(self._template_analysis, image, target_categories, roi)
                    futures['template'] = future

                # Collecte des résultats
                for method, future in futures.items():
                    try:
                        result = future.result(timeout=5.0)
                        results[method] = result
                    except Exception as e:
                        self.logger.error(f"Erreur {method} en parallèle: {e}")
            else:
                # Exécution séquentielle
                if self.yolo_detector:
                    results['yolo'] = self._yolo_analysis(image, target_categories, roi)

                if self.template_matcher:
                    results['template'] = self._template_analysis(image, target_categories, roi)

            # Fusion des résultats
            unified_result = self._merge_detection_results(results)
            unified_result['method'] = 'hybrid'

            self.stats['hybrid_analyses'] += 1

            return unified_result

        except Exception as e:
            self.logger.error(f"Erreur analyse hybride: {e}")
            return self._fallback_analysis(image, target_categories, roi)

    def _yolo_analysis(self, image: np.ndarray, target_categories: List[str],
                      roi: str) -> Dict[str, Any]:
        """Analyse avec YOLO uniquement"""
        if not self.yolo_detector or not self.yolo_detector.model_loaded:
            return self._create_empty_result('yolo')

        try:
            # Détection YOLO
            if target_categories:
                detections = self.yolo_detector.detect(image, target_categories)
            else:
                yolo_result = self.yolo_detector.analyze(image)
                detections = []

                # Conversion du format d'analyse
                if 'detections_by_class' in yolo_result:
                    for class_name, class_detections in yolo_result['detections_by_class'].items():
                        for det_dict in class_detections:
                            detection = YOLODetection(
                                class_name=det_dict['class_name'],
                                confidence=det_dict['confidence'],
                                bbox=tuple(det_dict['bbox']),
                                center=tuple(det_dict['center']),
                                area=det_dict['area'],
                                class_id=det_dict['class_id']
                            )
                            detections.append(detection)

            # Conversion en format unifié
            unified_detections = defaultdict(list)

            for detection in detections:
                if detection.confidence >= self.config.confidence_threshold_yolo:
                    unified_det = UnifiedDetection(
                        object_type=detection.class_name,
                        confidence=detection.confidence,
                        bbox=detection.bbox,
                        center=detection.center,
                        area=detection.area,
                        detection_method='yolo',
                        yolo_confidence=detection.confidence,
                        additional_data={'class_id': detection.class_id}
                    )

                    unified_detections[detection.class_name].append(unified_det.to_legacy_format())

            result = {
                "timestamp": datetime.now(),
                "detections_by_class": dict(unified_detections),
                "total_detections": len(detections),
                "method": "yolo",
                "roi_used": roi,
                "yolo_model_loaded": True
            }

            self.stats['yolo_analyses'] += 1
            return result

        except Exception as e:
            self.logger.error(f"Erreur analyse YOLO: {e}")
            return self._create_empty_result('yolo')

    def _template_analysis(self, image: np.ndarray, target_categories: List[str],
                          roi: str) -> Dict[str, Any]:
        """Analyse avec Template Matching uniquement"""
        if not self.template_matcher or not self.template_matcher.is_active():
            return self._create_empty_result('template')

        try:
            result = self.template_matcher.analyze(image, target_categories, roi)
            result['method'] = 'template'

            self.stats['template_analyses'] += 1
            return result

        except Exception as e:
            self.logger.error(f"Erreur analyse template: {e}")
            return self._create_empty_result('template')

    def _fallback_analysis(self, image: np.ndarray, target_categories: List[str],
                          roi: str) -> Dict[str, Any]:
        """Analyse de fallback avec le premier module disponible"""
        if self.yolo_detector and self.yolo_detector.model_loaded:
            return self._yolo_analysis(image, target_categories, roi)
        elif self.template_matcher and self.template_matcher.is_active():
            return self._template_analysis(image, target_categories, roi)
        else:
            return self._create_empty_result('none')

    def _merge_detection_results(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Fusionne les résultats de différentes méthodes de détection"""
        merged_detections = defaultdict(list)

        # Extraction de toutes les détections
        all_detections = []

        for method, result in results.items():
            if 'detections_by_class' in result:
                for class_name, detections in result['detections_by_class'].items():
                    for detection in detections:
                        detection['source_method'] = method
                        all_detections.append((class_name, detection))

        # Validation croisée et consensus
        if self.config.enable_cross_validation and len(results) > 1:
            validated_detections = self._cross_validate_detections(all_detections)
        else:
            validated_detections = all_detections

        # Regroupement par classe
        for class_name, detection in validated_detections:
            merged_detections[class_name].append(detection)

        # Création du résultat unifié
        total_detections = sum(len(dets) for dets in merged_detections.values())

        return {
            "timestamp": datetime.now(),
            "detections_by_class": dict(merged_detections),
            "total_detections": total_detections,
            "method": "hybrid",
            "source_methods": list(results.keys()),
            "cross_validated": self.config.enable_cross_validation
        }

    def _cross_validate_detections(self, all_detections: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """Validation croisée des détections pour éliminer les faux positifs"""
        validated = []

        # Groupement par classe et position approximative
        detection_groups = defaultdict(list)

        for class_name, detection in all_detections:
            bbox = detection['bounding_box']
            center = detection['position']

            # Clé basée sur position (grid 50x50)
            grid_x = center[0] // 50
            grid_y = center[1] // 50
            group_key = (class_name, grid_x, grid_y)

            detection_groups[group_key].append((class_name, detection))

        # Validation de chaque groupe
        for group_key, group_detections in detection_groups.items():
            if len(group_detections) == 1:
                # Détection unique - acceptée si confiance suffisante
                class_name, detection = group_detections[0]
                if detection['confidence'] >= self.config.validation_threshold:
                    validated.append((class_name, detection))
            else:
                # Détections multiples - consensus ou meilleure confiance
                if self.config.require_consensus:
                    # Consensus requis
                    methods = set(det[1]['source_method'] for det in group_detections)
                    if len(methods) >= 2:  # Au moins 2 méthodes d'accord
                        best_detection = max(group_detections, key=lambda x: x[1]['confidence'])
                        best_detection[1]['validation_score'] = len(methods) / len(group_detections)
                        validated.append(best_detection)
                        self.stats['consensus_validations'] += 1
                else:
                    # Meilleure confiance
                    best_detection = max(group_detections, key=lambda x: x[1]['confidence'])
                    validated.append(best_detection)

        return validated

    def _post_process_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Post-traitement des résultats"""
        # Ajout de métadonnées
        result['orchestrator_strategy'] = self.current_strategy
        result['performance_stats'] = {
            'total_analyses': self.stats['total_analyses'],
            'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses']),
            'avg_analysis_time': self.stats['avg_analysis_time']
        }

        return result

    def _generate_cache_key(self, image: np.ndarray, target_categories: List[str],
                           roi: str) -> str:
        """Génère une clé de cache pour les paramètres donnés"""
        # Hash rapide de l'image
        small_img = cv2.resize(image, (32, 32))
        img_hash = hash(small_img.tobytes())

        categories_str = "_".join(sorted(target_categories or []))

        return f"{img_hash}_{categories_str}_{roi}_{self.current_strategy}"

    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Récupère un résultat du cache s'il est valide"""
        if cache_key not in self.detection_cache:
            return None

        result, timestamp = self.detection_cache[cache_key]

        if time.time() - timestamp <= self.config.cache_duration:
            result = result.copy()
            result['cache_used'] = True
            return result
        else:
            # Cache expiré
            del self.detection_cache[cache_key]
            del self.cache_timestamps[cache_key]
            return None

    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Met en cache un résultat"""
        # Limitation de la taille du cache
        max_cache_size = 50
        if len(self.detection_cache) >= max_cache_size:
            # Suppression de la plus ancienne entrée
            oldest_key = min(self.cache_timestamps.keys(),
                           key=lambda k: self.cache_timestamps[k])
            del self.detection_cache[oldest_key]
            del self.cache_timestamps[oldest_key]

        self.detection_cache[cache_key] = (result.copy(), time.time())
        self.cache_timestamps[cache_key] = time.time()

    def _create_empty_result(self, method: str) -> Dict[str, Any]:
        """Crée un résultat vide"""
        return {
            "timestamp": datetime.now(),
            "detections_by_class": {},
            "total_detections": 0,
            "method": method,
            "error": f"Module {method} non disponible"
        }

    def _update_stats(self, result: Dict[str, Any], analysis_time: float):
        """Met à jour les statistiques"""
        self.stats['total_analyses'] += 1

        # Moyenne mobile du temps d'analyse
        alpha = 0.1
        self.stats['avg_analysis_time'] = (
            alpha * analysis_time +
            (1 - alpha) * self.stats['avg_analysis_time']
        )

        # Statistiques par méthode
        method = result.get('method', 'unknown')
        if method in ['yolo', 'template', 'hybrid']:
            self.stats[f'{method}_analyses'] = self.stats.get(f'{method}_analyses', 0) + 1

    def _evaluate_and_adapt_strategy(self, result: Dict[str, Any], analysis_time: float):
        """Évalue les performances et adapte la stratégie si nécessaire"""
        method = result.get('method', 'unknown')
        detection_count = result.get('total_detections', 0)
        avg_confidence = 0.0

        # Calcul confiance moyenne
        if result.get('detections_by_class'):
            confidences = []
            for detections in result['detections_by_class'].values():
                for detection in detections:
                    confidences.append(detection.get('confidence', 0.0))

            if confidences:
                avg_confidence = np.mean(confidences)

        # Enregistrement des performances
        self.performance_tracker.record_detection(
            method, analysis_time, detection_count, avg_confidence
        )

        # Vérification d'adaptation
        should_adapt, new_strategy = self.performance_tracker.should_adapt_strategy(self.config)

        if should_adapt and new_strategy != self.current_strategy:
            self.logger.info(f"Adaptation stratégie: {self.current_strategy} -> {new_strategy}")
            self.current_strategy = new_strategy
            self.stats['strategy_adaptations'] += 1

    def update(self, game_state: Any) -> Optional[Dict[str, Any]]:
        """Met à jour l'orchestrateur"""
        try:
            if not self.is_active():
                return None

            # Mise à jour des modules enfants
            shared_data = {}

            if self.yolo_detector:
                yolo_update = self.yolo_detector.update(game_state)
                if yolo_update:
                    shared_data.update(yolo_update.get('shared_data', {}))

            if self.template_matcher:
                template_update = self.template_matcher.update(game_state)
                if template_update:
                    shared_data.update(template_update.get('shared_data', {}))

            if self.screen_analyzer:
                screen_update = self.screen_analyzer.update(game_state)
                if screen_update:
                    shared_data.update(screen_update.get('shared_data', {}))

            # Données propres à l'orchestrateur
            shared_data['orchestrator_stats'] = self.stats.copy()
            shared_data['current_strategy'] = self.current_strategy
            shared_data['performance_summary'] = {
                method: self.performance_tracker.get_performance_summary(method)
                for method in ['yolo', 'template', 'hybrid']
            }

            return {
                "shared_data": shared_data,
                "module_status": "active"
            }

        except Exception as e:
            self.logger.error(f"Erreur update orchestrateur: {e}")
            return None

    def handle_event(self, event: Any) -> bool:
        """Gestion des événements"""
        # Propagation aux modules enfants
        handled = False

        if self.yolo_detector:
            handled |= self.yolo_detector.handle_event(event)

        if self.template_matcher:
            handled |= self.template_matcher.handle_event(event)

        if self.screen_analyzer:
            handled |= self.screen_analyzer.handle_event(event)

        return handled

    def get_state(self) -> Dict[str, Any]:
        """État de l'orchestrateur"""
        state = {
            "status": self.status.value,
            "current_strategy": self.current_strategy,
            "stats": self.stats,
            "modules": {}
        }

        # État des modules enfants
        if self.yolo_detector:
            state["modules"]["yolo"] = self.yolo_detector.get_state()

        if self.template_matcher:
            state["modules"]["template"] = self.template_matcher.get_state()

        if self.screen_analyzer:
            state["modules"]["screen"] = self.screen_analyzer.get_state()

        return state

    def cleanup(self) -> None:
        """Nettoyage des ressources"""
        self.logger.info("Arrêt de l'orchestrateur de vision")

        # Arrêt des modules enfants
        if self.yolo_detector:
            self.yolo_detector.cleanup()

        if self.template_matcher:
            self.template_matcher.cleanup()

        if self.screen_analyzer:
            self.screen_analyzer.cleanup()

        # Arrêt du thread pool
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)

        # Nettoyage des caches
        self.detection_cache.clear()
        self.cache_timestamps.clear()

        self.status = ModuleStatus.INACTIVE
        self.logger.info("Orchestrateur de vision arrêté")

    # Interface publique simplifiée pour compatibilité
    def find_objects(self, image: np.ndarray, object_types: List[str] = None,
                    min_confidence: float = None) -> List[Dict[str, Any]]:
        """Interface simplifiée pour trouver des objets"""
        # Application du seuil de confiance
        if min_confidence is None:
            min_confidence = self.config.confidence_threshold_yolo

        # Analyse
        result = self.analyze(image, object_types)

        if "error" in result:
            return []

        # Extraction et filtrage des détections
        all_detections = []

        for detections in result.get("detections_by_class", {}).values():
            for detection in detections:
                if detection.get('confidence', 0.0) >= min_confidence:
                    all_detections.append(detection)

        return all_detections

    def set_strategy(self, strategy: str):
        """Change manuellement la stratégie de détection"""
        if strategy in ['yolo', 'template', 'hybrid']:
            self.current_strategy = strategy
            self.logger.info(f"Stratégie changée manuellement: {strategy}")
        else:
            self.logger.warning(f"Stratégie inconnue: {strategy}")

    def get_performance_report(self) -> Dict[str, Any]:
        """Rapport de performance détaillé"""
        return {
            'current_strategy': self.current_strategy,
            'strategy_adaptations': self.stats['strategy_adaptations'],
            'performance_by_method': {
                method: self.performance_tracker.get_performance_summary(method)
                for method in ['yolo', 'template', 'hybrid']
            },
            'overall_stats': self.stats,
            'cache_performance': {
                'hit_rate': self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses']),
                'total_requests': self.stats['cache_hits'] + self.stats['cache_misses']
            }
        }