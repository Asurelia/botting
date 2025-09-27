"""
Dataset Bootstrap System pour YOLO
Système intelligent qui utilise les détections de template matching existantes
pour créer automatiquement un dataset d'entraînement YOLO initial
"""

import cv2
import numpy as np
import json
import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import shutil

# Import des modules internes
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from modules.vision.template_matcher import TemplateMatcher
    from modules.vision.screen_analyzer import ScreenAnalyzer
    TEMPLATE_AVAILABLE = True
except ImportError:
    TEMPLATE_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class BootstrapConfig:
    """Configuration du système de bootstrap"""
    # Dossiers de données
    output_directory: str = "data/bootstrap_dataset"
    template_data_dir: str = "data/templates"

    # Paramètres de collecte
    min_detections_per_image: int = 2
    max_detections_per_image: int = 20
    min_confidence_threshold: float = 0.7

    # Augmentation des données
    enable_data_augmentation: bool = True
    augmentation_factor: int = 3

    # Mapping des classes
    template_to_yolo_mapping: Dict[str, int] = field(default_factory=lambda: {
        'tree': 3,           # resource_tree
        'ash': 3,
        'oak': 3,
        'iron': 4,           # resource_ore
        'copper': 4,
        'coal': 4,
        'wheat': 5,          # resource_plant
        'barley': 5,
        'monster': 2,        # monster
        'mob': 2,
        'player': 0,         # player
        'character': 0,
        'npc': 1,           # npc
        'button': 7,        # ui_button
        'window': 8,        # ui_window
        'bag': 6,           # loot_bag
        'zaap': 17,         # zaap
        'door': 13,         # door
        'chest': 14,        # chest
        'bank': 15,         # bank
        'shop': 16          # shop
    })

    # Validation
    validation_split: float = 0.2
    enable_quality_check: bool = True

    # Performance
    max_workers: int = 4
    batch_size: int = 50

@dataclass
class BootstrapSample:
    """Échantillon d'entraînement bootstrap"""
    image_path: str
    image: np.ndarray
    detections: List[Dict[str, Any]]
    confidence_score: float
    template_method: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    yolo_annotations: List[Dict[str, Any]] = field(default_factory=list)

class DataAugmentor:
    """Augmentateur de données pour le bootstrap"""

    def __init__(self):
        self.augmentation_methods = [
            self._adjust_brightness,
            self._adjust_contrast,
            self._add_noise,
            self._slight_rotation,
            self._slight_scale
        ]

    def augment_sample(self, sample: BootstrapSample, num_variants: int = 3) -> List[BootstrapSample]:
        """Génère des variants augmentés d'un échantillon"""
        augmented_samples = [sample]  # Original

        for i in range(num_variants):
            try:
                # Application aléatoire d'augmentations
                augmented_image = sample.image.copy()
                applied_methods = []

                # Sélection aléatoire de 1-3 méthodes d'augmentation
                selected_methods = np.random.choice(
                    self.augmentation_methods,
                    size=np.random.randint(1, 4),
                    replace=False
                )

                for method in selected_methods:
                    augmented_image = method(augmented_image)
                    applied_methods.append(method.__name__)

                # Création de l'échantillon augmenté
                augmented_sample = BootstrapSample(
                    image_path=f"{sample.image_path}_aug_{i}",
                    image=augmented_image,
                    detections=sample.detections.copy(),
                    confidence_score=sample.confidence_score * 0.95,  # Légère réduction
                    template_method=sample.template_method,
                    metadata={
                        **sample.metadata,
                        'augmented': True,
                        'augmentation_methods': applied_methods,
                        'original_sample': sample.image_path
                    },
                    yolo_annotations=sample.yolo_annotations.copy()
                )

                augmented_samples.append(augmented_sample)

            except Exception as e:
                logger.warning(f"Erreur augmentation échantillon {i}: {e}")

        return augmented_samples

    def _adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        """Ajuste la luminosité"""
        factor = np.random.uniform(0.8, 1.2)
        return np.clip(image * factor, 0, 255).astype(np.uint8)

    def _adjust_contrast(self, image: np.ndarray) -> np.ndarray:
        """Ajuste le contraste"""
        factor = np.random.uniform(0.85, 1.15)
        return np.clip(((image - 128) * factor) + 128, 0, 255).astype(np.uint8)

    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """Ajoute du bruit gaussien"""
        noise = np.random.normal(0, np.random.uniform(1, 5), image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)

    def _slight_rotation(self, image: np.ndarray) -> np.ndarray:
        """Rotation légère"""
        angle = np.random.uniform(-2, 2)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))

    def _slight_scale(self, image: np.ndarray) -> np.ndarray:
        """Mise à l'échelle légère"""
        scale = np.random.uniform(0.95, 1.05)
        h, w = image.shape[:2]

        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))

        # Recadrage ou padding pour revenir à la taille originale
        if scale > 1.0:
            # Crop center
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            return resized[start_y:start_y + h, start_x:start_x + w]
        else:
            # Pad
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            padded = np.zeros((h, w, 3), dtype=np.uint8)
            padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
            return padded

class DatasetBootstrap:
    """
    Système de bootstrap de dataset YOLO utilisant template matching
    Convertit intelligemment les détections template en annotations YOLO
    """

    def __init__(self, config: BootstrapConfig = None):
        self.config = config or BootstrapConfig()
        self.logger = logging.getLogger(f"{__name__}.DatasetBootstrap")

        # Composants
        self.template_matcher: Optional[TemplateMatcher] = None
        self.screen_analyzer: Optional[ScreenAnalyzer] = None
        self.augmentor = DataAugmentor()

        # État
        self.samples: List[BootstrapSample] = []
        self.statistics = {
            'total_samples': 0,
            'total_detections': 0,
            'samples_by_class': {},
            'quality_distribution': {},
            'processing_time': 0.0
        }

        # Threading
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)

        # Chemins
        self.ensure_directories()

    def ensure_directories(self):
        """Assure que les répertoires nécessaires existent"""
        base_dir = Path(self.config.output_directory)

        directories = [
            base_dir,
            base_dir / "images" / "train",
            base_dir / "images" / "val",
            base_dir / "labels" / "train",
            base_dir / "labels" / "val",
            base_dir / "metadata"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> bool:
        """Initialise le système de bootstrap"""
        try:
            self.logger.info("Initialisation du système de bootstrap")

            if not TEMPLATE_AVAILABLE:
                self.logger.error("Template matcher non disponible")
                return False

            # Initialisation template matcher
            self.template_matcher = TemplateMatcher()
            if not self.template_matcher.initialize({}):
                self.logger.error("Échec initialisation template matcher")
                return False

            # Initialisation screen analyzer (optionnel)
            try:
                self.screen_analyzer = ScreenAnalyzer()
                self.screen_analyzer.initialize({})
                self.logger.info("✅ Screen analyzer initialisé")
            except Exception as e:
                self.logger.warning(f"Screen analyzer non disponible: {e}")

            self.logger.info("✅ Système de bootstrap initialisé")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation bootstrap: {e}")
            return False

    def bootstrap_from_screenshots(self, screenshot_dir: str,
                                  max_samples: int = 1000) -> bool:
        """Bootstrap à partir d'un dossier de screenshots"""
        try:
            self.logger.info(f"Bootstrap depuis {screenshot_dir}")

            screenshot_path = Path(screenshot_dir)
            if not screenshot_path.exists():
                self.logger.error(f"Dossier inexistant: {screenshot_dir}")
                return False

            # Collection des images
            image_files = list(screenshot_path.glob("*.png")) + \
                         list(screenshot_path.glob("*.jpg")) + \
                         list(screenshot_path.glob("*.jpeg"))

            if not image_files:
                self.logger.error("Aucune image trouvée")
                return False

            self.logger.info(f"Traitement de {len(image_files)} images")

            processed_count = 0
            start_time = time.time()

            # Traitement par batches
            for batch_start in range(0, len(image_files), self.config.batch_size):
                batch_end = min(batch_start + self.config.batch_size, len(image_files))
                batch_files = image_files[batch_start:batch_end]

                # Traitement parallèle du batch
                futures = []
                for image_file in batch_files:
                    if processed_count >= max_samples:
                        break

                    future = self.thread_pool.submit(self._process_image, str(image_file))
                    futures.append(future)

                # Collecte des résultats
                for future in futures:
                    try:
                        sample = future.result(timeout=30)
                        if sample:
                            self.samples.append(sample)
                            processed_count += 1

                            # Augmentation si activée
                            if self.config.enable_data_augmentation:
                                augmented = self.augmentor.augment_sample(
                                    sample, self.config.augmentation_factor
                                )
                                self.samples.extend(augmented[1:])  # Exclure l'original

                    except Exception as e:
                        self.logger.warning(f"Erreur traitement échantillon: {e}")

                self.logger.info(f"Batch traité: {processed_count}/{len(image_files)}")

            processing_time = time.time() - start_time
            self.statistics['processing_time'] = processing_time
            self.statistics['total_samples'] = len(self.samples)

            self.logger.info(f"Bootstrap terminé: {len(self.samples)} échantillons en {processing_time:.1f}s")

            return len(self.samples) > 0

        except Exception as e:
            self.logger.error(f"Erreur bootstrap screenshots: {e}")
            return False

    def _process_image(self, image_path: str) -> Optional[BootstrapSample]:
        """Traite une image pour extraction des détections"""
        try:
            # Chargement de l'image
            image = cv2.imread(image_path)
            if image is None:
                return None

            # Analyse avec template matcher
            detections = []

            # Test sur plusieurs catégories
            categories = ['resources', 'monsters', 'npcs', 'ui', 'characters']

            for category in categories:
                try:
                    category_detections = self.template_matcher.find_templates(
                        image,
                        category=category,
                        min_confidence=self.config.min_confidence_threshold
                    )

                    for detection in category_detections:
                        detection['category'] = category
                        detections.append(detection)

                except Exception as e:
                    self.logger.debug(f"Erreur catégorie {category}: {e}")

            # Filtrage par qualité
            if len(detections) < self.config.min_detections_per_image:
                return None

            if len(detections) > self.config.max_detections_per_image:
                # Tri par confiance et limitation
                detections.sort(key=lambda d: d.get('confidence', 0), reverse=True)
                detections = detections[:self.config.max_detections_per_image]

            # Calcul score de confiance global
            confidence_scores = [d.get('confidence', 0) for d in detections]
            avg_confidence = np.mean(confidence_scores)

            # Conversion en annotations YOLO
            yolo_annotations = self._convert_to_yolo_annotations(detections, image.shape)

            sample = BootstrapSample(
                image_path=image_path,
                image=image,
                detections=detections,
                confidence_score=avg_confidence,
                template_method="template_matching",
                metadata={
                    'image_shape': image.shape,
                    'detection_count': len(detections),
                    'categories_detected': list(set(d['category'] for d in detections)),
                    'timestamp': datetime.now().isoformat()
                },
                yolo_annotations=yolo_annotations
            )

            return sample

        except Exception as e:
            self.logger.debug(f"Erreur traitement {image_path}: {e}")
            return None

    def _convert_to_yolo_annotations(self, detections: List[Dict[str, Any]],
                                   image_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """Convertit les détections template en annotations YOLO"""
        h, w = image_shape[:2]
        annotations = []

        for detection in detections:
            try:
                # Extraction des informations
                template_name = detection.get('template_name', '')
                confidence = detection.get('confidence', 0)
                bbox = detection.get('bounding_box', None)

                if not bbox or len(bbox) != 4:
                    continue

                # Mapping vers classe YOLO
                yolo_class = self._map_template_to_yolo_class(template_name)
                if yolo_class is None:
                    continue

                # Conversion bbox vers format YOLO (x_center, y_center, width, height) normalisé
                x1, y1, x2, y2 = bbox

                # Vérification des bounds
                x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
                y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                # Conversion YOLO
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h

                # Validation des valeurs normalisées
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                       0 < width <= 1 and 0 < height <= 1):
                    continue

                annotation = {
                    'class_id': yolo_class,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'confidence': confidence,
                    'template_name': template_name
                }

                annotations.append(annotation)

            except Exception as e:
                self.logger.debug(f"Erreur conversion annotation: {e}")

        return annotations

    def _map_template_to_yolo_class(self, template_name: str) -> Optional[int]:
        """Mappe un nom de template vers une classe YOLO"""
        template_name_lower = template_name.lower()

        for template_key, yolo_class in self.config.template_to_yolo_mapping.items():
            if template_key in template_name_lower:
                return yolo_class

        return None

    def export_yolo_dataset(self) -> bool:
        """Exporte le dataset au format YOLO"""
        try:
            self.logger.info("Export du dataset YOLO")

            if not self.samples:
                self.logger.error("Aucun échantillon à exporter")
                return False

            # Division train/validation
            np.random.shuffle(self.samples)
            split_idx = int(len(self.samples) * (1 - self.config.validation_split))

            train_samples = self.samples[:split_idx]
            val_samples = self.samples[split_idx:]

            self.logger.info(f"Division: {len(train_samples)} train, {len(val_samples)} val")

            # Export des échantillons
            self._export_samples(train_samples, "train")
            self._export_samples(val_samples, "val")

            # Génération du fichier dataset.yaml
            self._generate_dataset_config()

            # Génération des statistiques
            self._generate_statistics()

            self.logger.info("✅ Dataset YOLO exporté avec succès")
            return True

        except Exception as e:
            self.logger.error(f"Erreur export dataset: {e}")
            return False

    def _export_samples(self, samples: List[BootstrapSample], split: str):
        """Exporte les échantillons pour un split donné"""
        base_dir = Path(self.config.output_directory)
        images_dir = base_dir / "images" / split
        labels_dir = base_dir / "labels" / split

        for idx, sample in enumerate(samples):
            try:
                # Nom de fichier unique
                filename = f"{split}_{idx:06d}"

                # Sauvegarde de l'image
                image_path = images_dir / f"{filename}.jpg"
                cv2.imwrite(str(image_path), sample.image)

                # Sauvegarde des annotations
                label_path = labels_dir / f"{filename}.txt"
                with open(label_path, 'w') as f:
                    for annotation in sample.yolo_annotations:
                        line = f"{annotation['class_id']} {annotation['x_center']:.6f} {annotation['y_center']:.6f} {annotation['width']:.6f} {annotation['height']:.6f}\n"
                        f.write(line)

                # Mise à jour des statistiques
                for annotation in sample.yolo_annotations:
                    class_id = annotation['class_id']
                    self.statistics['samples_by_class'][class_id] = \
                        self.statistics['samples_by_class'].get(class_id, 0) + 1

                self.statistics['total_detections'] += len(sample.yolo_annotations)

            except Exception as e:
                self.logger.warning(f"Erreur export échantillon {idx}: {e}")

    def _generate_dataset_config(self):
        """Génère le fichier dataset.yaml pour YOLO"""
        base_dir = Path(self.config.output_directory)

        # Classes DOFUS définies
        class_names = {
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
        }

        # Classes présentes dans le dataset
        present_classes = set(self.statistics['samples_by_class'].keys())
        filtered_names = [class_names.get(i, f"class_{i}") for i in sorted(present_classes)]

        dataset_config = {
            'path': str(base_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(present_classes),
            'names': filtered_names,
            'description': 'DOFUS Bootstrap Dataset from Template Matching',
            'created': datetime.now().isoformat(),
            'statistics': self.statistics
        }

        config_path = base_dir / "dataset.yaml"
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(dataset_config, f, default_flow_style=False)

        self.logger.info(f"Dataset config sauvegardé: {config_path}")

    def _generate_statistics(self):
        """Génère les statistiques détaillées du dataset"""
        base_dir = Path(self.config.output_directory)

        stats = {
            'bootstrap_config': {
                'min_detections_per_image': self.config.min_detections_per_image,
                'min_confidence_threshold': self.config.min_confidence_threshold,
                'data_augmentation': self.config.enable_data_augmentation,
                'augmentation_factor': self.config.augmentation_factor
            },
            'dataset_statistics': self.statistics,
            'class_distribution': self.statistics['samples_by_class'],
            'generation_date': datetime.now().isoformat()
        }

        stats_path = base_dir / "metadata" / "bootstrap_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f"Statistiques sauvegardées: {stats_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques actuelles"""
        return self.statistics

    def cleanup(self):
        """Nettoyage des ressources"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)

        if self.template_matcher:
            self.template_matcher.cleanup()

# Interface de commande
def main():
    """Interface en ligne de commande pour le bootstrap"""
    import argparse

    parser = argparse.ArgumentParser(description="Bootstrap de dataset YOLO depuis template matching")
    parser.add_argument("--screenshots", required=True, help="Dossier contenant les screenshots")
    parser.add_argument("--output", default="data/bootstrap_dataset", help="Dossier de sortie")
    parser.add_argument("--max-samples", type=int, default=1000, help="Nombre max d'échantillons")
    parser.add_argument("--confidence", type=float, default=0.7, help="Seuil de confiance minimum")
    parser.add_argument("--augment", action="store_true", help="Activer l'augmentation de données")

    args = parser.parse_args()

    # Configuration
    config = BootstrapConfig(
        output_directory=args.output,
        min_confidence_threshold=args.confidence,
        enable_data_augmentation=args.augment
    )

    # Bootstrap
    bootstrap = DatasetBootstrap(config)

    if not bootstrap.initialize():
        print("❌ Erreur d'initialisation")
        return False

    print(f"🚀 Démarrage du bootstrap depuis {args.screenshots}")

    if bootstrap.bootstrap_from_screenshots(args.screenshots, args.max_samples):
        print(f"✅ {len(bootstrap.samples)} échantillons collectés")

        if bootstrap.export_yolo_dataset():
            print(f"✅ Dataset exporté vers {args.output}")

            stats = bootstrap.get_statistics()
            print(f"📊 Statistiques:")
            print(f"  - Total échantillons: {stats['total_samples']}")
            print(f"  - Total détections: {stats['total_detections']}")
            print(f"  - Classes détectées: {len(stats['samples_by_class'])}")
            print(f"  - Temps de traitement: {stats['processing_time']:.1f}s")

            return True

    print("❌ Échec du bootstrap")
    return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()