"""
Système de détection YOLO avancé pour DOFUS
Remplace le template matching par une détection d'objets robuste et moderne
"""

import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import json

# Import YOLO (ultralytics)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("Ultralytics YOLO non disponible. Installation requise: pip install ultralytics")

# Import des modules internes
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from engine.module_interface import IAnalysisModule, ModuleStatus
from engine.event_bus import EventType, EventPriority

logger = logging.getLogger(__name__)

@dataclass
class YOLODetection:
    """Résultat d'une détection YOLO"""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: int
    class_id: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire pour compatibilité"""
        return {
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'center': self.center,
            'area': self.area,
            'class_id': self.class_id,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class YOLOConfig:
    """Configuration du système YOLO"""
    model_path: str = "models/yolo/dofus_v8n.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    device: str = "auto"  # auto, cpu, cuda, mps
    input_size: Tuple[int, int] = (640, 640)
    augment_inference: bool = True
    half_precision: bool = True
    optimize_model: bool = True

    # Classes DOFUS spécifiques
    dofus_classes: Dict[int, str] = field(default_factory=lambda: {
        0: "player",
        1: "npc",
        2: "monster",
        3: "resource_tree",
        4: "resource_ore",
        5: "resource_plant",
        6: "loot_bag",
        7: "ui_button",
        8: "ui_window",
        9: "ui_inventory",
        10: "ui_spells",
        11: "ui_minimap",
        12: "portal",
        13: "door",
        14: "chest",
        15: "bank",
        16: "shop",
        17: "zaap",
        18: "archmonster",
        19: "quest_item",
        20: "pvp_player"
    })

class YOLODatasetBuilder:
    """Constructeur de dataset pour entraînement YOLO"""

    def __init__(self, dataset_path: str = "data/yolo_dataset"):
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / "images"
        self.labels_dir = self.dataset_path / "labels"
        self.setup_directories()

        # Compteurs
        self.images_captured = 0
        self.annotations_created = 0

        logger.info(f"Dataset builder initialisé: {self.dataset_path}")

    def setup_directories(self):
        """Crée la structure de répertoires YOLO"""
        for split in ["train", "val", "test"]:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)

    def capture_training_image(self, image: np.ndarray,
                             detections: List[Dict] = None,
                             split: str = "train") -> bool:
        """
        Capture une image pour l'entraînement avec annotations optionnelles

        Args:
            image: Image à sauvegarder
            detections: Détections existantes pour annotation automatique
            split: train/val/test
        """
        try:
            # Nom de fichier unique
            timestamp = int(time.time() * 1000)
            filename = f"dofus_{timestamp}_{self.images_captured:06d}"

            # Sauvegarde image
            img_path = self.images_dir / split / f"{filename}.jpg"
            cv2.imwrite(str(img_path), image)

            # Création du fichier d'annotation YOLO
            if detections:
                self._create_yolo_annotation(image, detections, filename, split)

            self.images_captured += 1

            if self.images_captured % 100 == 0:
                logger.info(f"Capturé {self.images_captured} images d'entraînement")

            return True

        except Exception as e:
            logger.error(f"Erreur capture image entraînement: {e}")
            return False

    def _create_yolo_annotation(self, image: np.ndarray, detections: List[Dict],
                               filename: str, split: str):
        """Crée un fichier d'annotation au format YOLO"""
        h, w = image.shape[:2]
        annotation_path = self.labels_dir / split / f"{filename}.txt"

        with open(annotation_path, 'w') as f:
            for detection in detections:
                # Conversion bbox absolue vers YOLO (normalisée)
                x1, y1, x2, y2 = detection['bbox']

                # Centre et dimensions normalisées
                center_x = ((x1 + x2) / 2) / w
                center_y = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h

                class_id = detection.get('class_id', 0)

                # Format YOLO: class_id center_x center_y width height
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

        self.annotations_created += 1

    def create_dataset_config(self) -> str:
        """Crée le fichier de configuration YAML pour YOLO"""
        config_content = f"""# Configuration dataset DOFUS YOLO
path: {self.dataset_path.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
nc: {len(YOLOConfig().dofus_classes)}
names: {list(YOLOConfig().dofus_classes.values())}
"""

        config_path = self.dataset_path / "dataset.yaml"
        with open(config_path, 'w') as f:
            f.write(config_content)

        logger.info(f"Configuration dataset créée: {config_path}")
        return str(config_path)

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du dataset"""
        stats = {
            'images_captured': self.images_captured,
            'annotations_created': self.annotations_created,
            'splits': {}
        }

        for split in ["train", "val", "test"]:
            img_count = len(list((self.images_dir / split).glob("*.jpg")))
            label_count = len(list((self.labels_dir / split).glob("*.txt")))
            stats['splits'][split] = {
                'images': img_count,
                'labels': label_count
            }

        return stats

class YOLOTrainer:
    """Entraîneur YOLO pour DOFUS"""

    def __init__(self, config: YOLOConfig):
        self.config = config
        self.model_dir = Path("models/yolo")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Vérification de YOLO
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics YOLO requis pour l'entraînement")

    def train_model(self, dataset_config: str, epochs: int = 100,
                   batch_size: int = 16, patience: int = 50) -> str:
        """
        Lance l'entraînement YOLO

        Args:
            dataset_config: Chemin vers le fichier de config dataset
            epochs: Nombre d'époques
            batch_size: Taille de batch
            patience: Early stopping patience

        Returns:
            Chemin vers le modèle entraîné
        """
        try:
            logger.info(f"Début entraînement YOLO - {epochs} époques")

            # Modèle de base (YOLOv8 nano pour démarrage rapide)
            base_model = YOLO('yolov8n.pt')

            # Configuration d'entraînement
            train_args = {
                'data': dataset_config,
                'epochs': epochs,
                'batch': batch_size,
                'patience': patience,
                'save_period': 10,
                'device': self.config.device,
                'project': str(self.model_dir),
                'name': 'dofus_training',
                'exist_ok': True,
                'verbose': True,
                'save': True,
                'plots': True,
                'val': True,
                'split': 0.8  # 80% train, 20% val
            }

            # Lancement de l'entraînement
            results = base_model.train(**train_args)

            # Chemin du meilleur modèle
            best_model_path = self.model_dir / "dofus_training" / "weights" / "best.pt"

            logger.info(f"Entraînement terminé. Modèle sauvegardé: {best_model_path}")

            # Copie vers le chemin de config
            final_model_path = Path(self.config.model_path)
            final_model_path.parent.mkdir(parents=True, exist_ok=True)

            if best_model_path.exists():
                import shutil
                shutil.copy2(best_model_path, final_model_path)
                logger.info(f"Modèle copié vers: {final_model_path}")

            return str(final_model_path)

        except Exception as e:
            logger.error(f"Erreur entraînement YOLO: {e}")
            raise

    def validate_model(self, model_path: str, dataset_config: str) -> Dict[str, float]:
        """Valide un modèle entraîné"""
        try:
            model = YOLO(model_path)
            results = model.val(data=dataset_config)

            metrics = {
                'map50': results.box.map50,
                'map': results.box.map,
                'precision': results.box.mp,
                'recall': results.box.mr
            }

            logger.info(f"Métriques validation: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Erreur validation modèle: {e}")
            return {}

class DofusYOLODetector(IAnalysisModule):
    """
    Détecteur YOLO principal pour DOFUS
    Module de vision de nouvelle génération basé sur la détection d'objets
    """

    def __init__(self, name: str = "yolo_detector", config: YOLOConfig = None):
        super().__init__(name)

        self.config = config or YOLOConfig()
        self.logger = logging.getLogger(f"{__name__}.DofusYOLODetector")

        # Modèle YOLO
        self.model = None
        self.model_loaded = False

        # Cache et optimisations
        self.detection_cache = {}
        self.cache_duration = 0.1  # 100ms de cache pour performances
        self.last_detection_time = 0

        # Statistiques
        self.stats = {
            'total_detections': 0,
            'avg_inference_time': 0,
            'fps': 0,
            'confidence_scores': defaultdict(list),
            'class_counts': defaultdict(int)
        }

        # Dataset builder pour collecte continue
        self.dataset_builder = YOLODatasetBuilder()
        self.auto_collect_data = False

        # Post-processing
        self.tracking_enabled = True
        self.object_tracker = {}  # Simple tracking par position
        self.next_track_id = 0

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialise le détecteur YOLO"""
        try:
            self.logger.info("Initialisation du détecteur YOLO DOFUS")

            # Vérification YOLO
            if not YOLO_AVAILABLE:
                self.logger.error("Ultralytics YOLO non disponible")
                self.set_error("YOLO non installé")
                return False

            # Chargement du modèle
            model_path = Path(self.config.model_path)

            if model_path.exists():
                self.logger.info(f"Chargement modèle YOLO: {model_path}")
                self.model = YOLO(str(model_path))

                # Optimisations
                if self.config.optimize_model:
                    self._optimize_model()

                self.model_loaded = True
                self.logger.info("Modèle YOLO chargé avec succès")
            else:
                self.logger.warning(f"Modèle non trouvé: {model_path}")
                self.logger.info("Utilisation du modèle YOLOv8 par défaut")
                self.model = YOLO('yolov8n.pt')
                self.model_loaded = True

            self.status = ModuleStatus.ACTIVE
            self.logger.info("Détecteur YOLO initialisé")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation YOLO: {e}")
            self.set_error(str(e))
            return False

    def _optimize_model(self):
        """Optimise le modèle pour l'inférence"""
        try:
            # Half precision si supporté
            if self.config.half_precision and torch.cuda.is_available():
                self.model.model.half()
                self.logger.info("Half precision activée")

            # Compilation du modèle (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                self.model.model = torch.compile(self.model.model)
                self.logger.info("Modèle compilé pour optimisation")

        except Exception as e:
            self.logger.warning(f"Optimisation modèle échouée: {e}")

    def detect(self, image: np.ndarray, classes: List[str] = None) -> List[YOLODetection]:
        """
        Détection d'objets YOLO sur une image

        Args:
            image: Image d'entrée
            classes: Classes spécifiques à détecter (None = toutes)

        Returns:
            Liste des détections
        """
        if not self.model_loaded:
            return []

        start_time = time.perf_counter()

        try:
            # Préparation de l'image
            input_image = self._preprocess_image(image)

            # Inférence YOLO
            results = self.model(
                input_image,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                max_det=self.config.max_detections,
                augment=self.config.augment_inference,
                verbose=False
            )

            # Conversion des résultats
            detections = self._process_results(results[0], image.shape[:2])

            # Filtrage par classes si spécifié
            if classes:
                detections = [d for d in detections if d.class_name in classes]

            # Tracking simple
            if self.tracking_enabled:
                detections = self._update_tracking(detections)

            # Mise à jour des statistiques
            inference_time = time.perf_counter() - start_time
            self._update_stats(detections, inference_time)

            # Collecte de données automatique si activée
            if self.auto_collect_data and len(detections) > 0:
                self._auto_collect_sample(image, detections)

            return detections

        except Exception as e:
            self.logger.error(f"Erreur détection YOLO: {e}")
            return []

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Préprocess l'image pour YOLO"""
        # YOLO attend RGB, pas BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def _process_results(self, result, original_shape: Tuple[int, int]) -> List[YOLODetection]:
        """Convertit les résultats YOLO en détections structurées"""
        detections = []

        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = box.astype(int)

            # Validation des coordonnées
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(original_shape[1], x2)
            y2 = min(original_shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Création de la détection
            detection = YOLODetection(
                class_name=self.config.dofus_classes.get(cls_id, f"unknown_{cls_id}"),
                confidence=float(conf),
                bbox=(x1, y1, x2, y2),
                center=((x1 + x2) // 2, (y1 + y2) // 2),
                area=(x2 - x1) * (y2 - y1),
                class_id=cls_id
            )

            detections.append(detection)

        return detections

    def _update_tracking(self, detections: List[YOLODetection]) -> List[YOLODetection]:
        """Mise à jour simple du tracking par position"""
        # Tracking basique par proximité (à améliorer avec DeepSORT)
        for detection in detections:
            detection.track_id = self._get_track_id(detection)

        return detections

    def _get_track_id(self, detection: YOLODetection) -> int:
        """Obtient un ID de tracking pour une détection"""
        center = detection.center

        # Recherche du tracker le plus proche
        min_distance = float('inf')
        best_track_id = None

        for track_id, track_data in self.object_tracker.items():
            if track_data['class_name'] == detection.class_name:
                distance = np.sqrt(
                    (center[0] - track_data['last_center'][0])**2 +
                    (center[1] - track_data['last_center'][1])**2
                )

                if distance < min_distance and distance < 100:  # Seuil de 100 pixels
                    min_distance = distance
                    best_track_id = track_id

        if best_track_id is not None:
            # Mise à jour tracker existant
            self.object_tracker[best_track_id]['last_center'] = center
            self.object_tracker[best_track_id]['last_seen'] = time.time()
            return best_track_id
        else:
            # Nouveau tracker
            track_id = self.next_track_id
            self.next_track_id += 1

            self.object_tracker[track_id] = {
                'class_name': detection.class_name,
                'last_center': center,
                'last_seen': time.time()
            }

            return track_id

    def _update_stats(self, detections: List[YOLODetection], inference_time: float):
        """Met à jour les statistiques de performance"""
        self.stats['total_detections'] += len(detections)

        # Moyenne mobile du temps d'inférence
        alpha = 0.1
        self.stats['avg_inference_time'] = (
            alpha * inference_time +
            (1 - alpha) * self.stats['avg_inference_time']
        )

        # FPS
        self.stats['fps'] = 1.0 / max(inference_time, 0.001)

        # Scores de confiance par classe
        for detection in detections:
            self.stats['confidence_scores'][detection.class_name].append(detection.confidence)
            self.stats['class_counts'][detection.class_name] += 1

    def _auto_collect_sample(self, image: np.ndarray, detections: List[YOLODetection]):
        """Collecte automatique d'échantillons pour amélioration du dataset"""
        # Collecte seulement si détections de qualité
        high_conf_detections = [d for d in detections if d.confidence > 0.8]

        if len(high_conf_detections) >= 2:  # Au moins 2 détections de qualité
            detection_dicts = [d.to_dict() for d in high_conf_detections]
            self.dataset_builder.capture_training_image(image, detection_dicts)

    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Interface d'analyse principale compatible avec l'architecture existante

        Args:
            image: Image à analyser

        Returns:
            Résultats de détection formatés
        """
        start_time = time.perf_counter()

        try:
            # Détection YOLO
            detections = self.detect(image)

            # Groupement par classe
            detections_by_class = defaultdict(list)
            for detection in detections:
                detections_by_class[detection.class_name].append(detection.to_dict())

            # Formatage compatible avec template_matcher
            result = {
                "timestamp": datetime.now(),
                "detections_by_class": dict(detections_by_class),
                "total_detections": len(detections),
                "processing_time": time.perf_counter() - start_time,
                "method": "yolo_v8",
                "model_loaded": self.model_loaded,
                "performance": {
                    "fps": self.stats['fps'],
                    "avg_inference_time": self.stats['avg_inference_time']
                }
            }

            return result

        except Exception as e:
            self.logger.error(f"Erreur analyse YOLO: {e}")
            return {"error": str(e), "timestamp": datetime.now()}

    def update(self, game_state: Any) -> Optional[Dict[str, Any]]:
        """Met à jour le détecteur YOLO"""
        try:
            if not self.is_active():
                return None

            return {
                "shared_data": {
                    "yolo_stats": self.stats,
                    "model_loaded": self.model_loaded,
                    "tracking_objects": len(self.object_tracker)
                },
                "module_status": "active"
            }

        except Exception as e:
            self.logger.error(f"Erreur update YOLO: {e}")
            return None

    def handle_event(self, event: Any) -> bool:
        """Gestion des événements"""
        return False

    def get_state(self) -> Dict[str, Any]:
        """État du module YOLO"""
        return {
            "status": self.status.value,
            "model_loaded": self.model_loaded,
            "model_path": self.config.model_path,
            "stats": self.stats,
            "auto_collect_enabled": self.auto_collect_data,
            "tracking_enabled": self.tracking_enabled
        }

    def cleanup(self) -> None:
        """Nettoyage des ressources"""
        self.logger.info("Arrêt du détecteur YOLO")

        if self.model:
            del self.model
            self.model = None

        self.model_loaded = False
        self.status = ModuleStatus.INACTIVE

        self.logger.info("Détecteur YOLO arrêté")

    def enable_data_collection(self, enabled: bool = True):
        """Active/désactive la collecte automatique de données"""
        self.auto_collect_data = enabled
        self.logger.info(f"Collecte automatique: {'activée' if enabled else 'désactivée'}")

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Statistiques du dataset collecté"""
        return self.dataset_builder.get_stats()

    def train_new_model(self, epochs: int = 100) -> str:
        """Lance l'entraînement d'un nouveau modèle avec les données collectées"""
        trainer = YOLOTrainer(self.config)
        dataset_config = self.dataset_builder.create_dataset_config()

        return trainer.train_model(dataset_config, epochs)

# Interface de compatibilité avec l'ancien système
class YOLOTemplateAdapter:
    """Adaptateur pour compatibilité avec l'ancien système de template matching"""

    def __init__(self, yolo_detector: DofusYOLODetector):
        self.yolo_detector = yolo_detector

    def find_templates(self, image: np.ndarray, category: str = None,
                      template_name: str = None, roi: str = "full_screen",
                      min_confidence: float = None) -> List[Dict]:
        """Interface compatible avec l'ancien TemplateMatcher.find_templates"""

        # Mapping des catégories vers classes YOLO
        category_mapping = {
            'resources': ['resource_tree', 'resource_ore', 'resource_plant'],
            'monsters': ['monster', 'archmonster'],
            'npcs': ['npc'],
            'ui': ['ui_button', 'ui_window', 'ui_inventory', 'ui_spells'],
            'interactive': ['portal', 'door', 'chest', 'bank', 'shop', 'zaap']
        }

        # Détermination des classes à chercher
        target_classes = None
        if category and category in category_mapping:
            target_classes = category_mapping[category]

        # Détection YOLO
        detections = self.yolo_detector.detect(image, target_classes)

        # Conversion au format legacy
        legacy_matches = []
        for detection in detections:
            if min_confidence and detection.confidence < min_confidence:
                continue

            if template_name and detection.class_name != template_name:
                continue

            # Format compatible
            match = {
                'template_name': detection.class_name,
                'template_type': category or 'auto_detected',
                'confidence': detection.confidence,
                'position': detection.center,
                'bounding_box': detection.bbox,
                'scale': 1.0,  # YOLO est scale-invariant
                'rotation': 0.0,
                'method': 'yolo_v8',
                'additional_data': {
                    'class_id': detection.class_id,
                    'area': detection.area,
                    'track_id': getattr(detection, 'track_id', None)
                }
            }

            legacy_matches.append(match)

        return legacy_matches