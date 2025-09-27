"""
Collecteur de données automatisé pour entraînement YOLO DOFUS
Capture intelligente avec annotation semi-automatique et interface utilisateur
"""

import cv2
import numpy as np
import time
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
from collections import defaultdict

# Import des modules vision
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.vision.yolo_detector import YOLODatasetBuilder, YOLOConfig
from modules.vision.screen_analyzer import ScreenAnalyzer
from modules.vision.template_matcher import TemplateMatcher

logger = logging.getLogger(__name__)

@dataclass
class AnnotationSession:
    """Session d'annotation de données"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    images_annotated: int = 0
    annotations_created: int = 0
    user_name: str = "annotator"
    quality_score: float = 0.0
    notes: str = ""

@dataclass
class BoundingBoxAnnotation:
    """Annotation de boîte englobante"""
    x1: int
    y1: int
    x2: int
    y2: int
    class_name: str
    confidence: float = 1.0
    verified: bool = False
    annotator: str = "auto"

class SmartDataCollector:
    """
    Collecteur de données intelligent avec annotation assistée
    """

    def __init__(self, output_dir: str = "data/collected"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Modules de vision pour détection existante
        self.screen_analyzer = None
        self.template_matcher = None
        self.yolo_builder = YOLODatasetBuilder(str(self.output_dir / "yolo_dataset"))

        # Configuration
        self.config = YOLOConfig()

        # État de collecte
        self.is_collecting = False
        self.collection_thread = None
        self.frames_captured = 0
        self.auto_annotate = True

        # Filtres de collecte
        self.collection_filters = {
            'min_objects': 1,
            'max_objects': 20,
            'min_confidence': 0.3,
            'skip_empty_frames': True,
            'capture_interval': 2.0,  # secondes
            'max_similar_frames': 5
        }

        # Historique pour éviter les doublons
        self.frame_history = []
        self.similarity_threshold = 0.95

        # Statistiques
        self.stats = {
            'total_captures': 0,
            'auto_annotations': 0,
            'manual_annotations': 0,
            'rejected_frames': 0,
            'classes_found': defaultdict(int),
            'session_start': None
        }

        logger.info(f"Collecteur de données initialisé: {self.output_dir}")

    def initialize_vision_modules(self):
        """Initialise les modules de vision pour l'annotation automatique"""
        try:
            # Module d'analyse d'écran
            self.screen_analyzer = ScreenAnalyzer()
            self.screen_analyzer.initialize({})

            # Template matcher pour détection existante
            self.template_matcher = TemplateMatcher()
            self.template_matcher.initialize({})

            logger.info("Modules de vision initialisés pour l'annotation")
            return True

        except Exception as e:
            logger.error(f"Erreur initialisation modules vision: {e}")
            return False

    def start_automated_collection(self, duration_hours: float = 1.0):
        """
        Démarre la collecte automatisée de données

        Args:
            duration_hours: Durée de collecte en heures
        """
        if self.is_collecting:
            logger.warning("Collecte déjà en cours")
            return

        self.is_collecting = True
        self.stats['session_start'] = time.time()

        # Initialisation des modules si nécessaire
        if not self.screen_analyzer:
            self.initialize_vision_modules()

        # Thread de collecte
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            args=(duration_hours,),
            daemon=True
        )
        self.collection_thread.start()

        logger.info(f"Collecte automatisée démarrée pour {duration_hours}h")

    def _collection_loop(self, duration_hours: float):
        """Boucle principale de collecte"""
        end_time = time.time() + (duration_hours * 3600)
        last_capture = 0

        while self.is_collecting and time.time() < end_time:
            try:
                current_time = time.time()

                # Respect de l'intervalle de capture
                if current_time - last_capture < self.collection_filters['capture_interval']:
                    time.sleep(0.1)
                    continue

                # Capture d'écran
                if self.screen_analyzer:
                    screenshot = self.screen_analyzer.get_current_screenshot()
                    if screenshot is not None:
                        self._process_captured_frame(screenshot)
                        last_capture = current_time

                # Pause pour éviter la surcharge CPU
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Erreur boucle de collecte: {e}")
                time.sleep(1.0)

        self.is_collecting = False
        logger.info("Collecte automatisée terminée")
        self._generate_collection_report()

    def _process_captured_frame(self, frame: np.ndarray):
        """Traite une frame capturée"""
        try:
            # Vérification de similarité avec frames récentes
            if self._is_similar_to_recent(frame):
                self.stats['rejected_frames'] += 1
                return

            # Annotation automatique
            annotations = []
            if self.auto_annotate:
                annotations = self._auto_annotate_frame(frame)

            # Filtrage basé sur les critères
            if not self._meets_collection_criteria(annotations):
                self.stats['rejected_frames'] += 1
                return

            # Sauvegarde de la frame avec annotations
            success = self._save_annotated_frame(frame, annotations)

            if success:
                self.frames_captured += 1
                self.stats['total_captures'] += 1
                self.stats['auto_annotations'] += len(annotations)

                # Mise à jour des statistiques de classes
                for ann in annotations:
                    self.stats['classes_found'][ann.class_name] += 1

                # Ajout à l'historique
                self._add_to_history(frame)

                if self.frames_captured % 50 == 0:
                    logger.info(f"Capturé {self.frames_captured} frames avec annotations")

        except Exception as e:
            logger.error(f"Erreur traitement frame: {e}")

    def _auto_annotate_frame(self, frame: np.ndarray) -> List[BoundingBoxAnnotation]:
        """Annotation automatique d'une frame avec les détecteurs existants"""
        annotations = []

        try:
            # Analyse avec template matcher
            if self.template_matcher:
                template_result = self.template_matcher.analyze(frame)
                annotations.extend(self._convert_template_matches(template_result))

            # Analyse avec screen analyzer pour UI
            if self.screen_analyzer:
                screen_result = self.screen_analyzer.analyze(frame)
                annotations.extend(self._convert_screen_analysis(screen_result))

        except Exception as e:
            logger.error(f"Erreur annotation automatique: {e}")

        return annotations

    def _convert_template_matches(self, template_result: Dict) -> List[BoundingBoxAnnotation]:
        """Convertit les résultats de template matching en annotations"""
        annotations = []

        if "matches_by_category" not in template_result:
            return annotations

        for category, matches in template_result["matches_by_category"].items():
            for match in matches:
                # Mapping des catégories vers classes YOLO
                yolo_class = self._map_template_to_yolo_class(category, match.get('template_name', ''))

                if yolo_class:
                    bbox = match.get('bounding_box')
                    confidence = match.get('confidence', 0.0)

                    if bbox and confidence >= self.collection_filters['min_confidence']:
                        annotation = BoundingBoxAnnotation(
                            x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
                            class_name=yolo_class,
                            confidence=confidence,
                            annotator="template_matcher"
                        )
                        annotations.append(annotation)

        return annotations

    def _convert_screen_analysis(self, screen_result: Dict) -> List[BoundingBoxAnnotation]:
        """Convertit les résultats d'analyse d'écran en annotations"""
        annotations = []

        # UI elements
        if "ui_elements" in screen_result:
            for element in screen_result["ui_elements"]:
                if element.get('confidence', 0) >= self.collection_filters['min_confidence']:
                    bbox = element.get('bounding_box')
                    if bbox:
                        annotation = BoundingBoxAnnotation(
                            x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
                            class_name="ui_element",
                            confidence=element.get('confidence', 1.0),
                            annotator="screen_analyzer"
                        )
                        annotations.append(annotation)

        # Resources
        if "resources" in screen_result:
            for resource in screen_result["resources"]:
                resource_type = resource.get('type', 'resource')
                yolo_class = f"resource_{resource_type}"

                bbox = resource.get('bounding_box')
                if bbox:
                    annotation = BoundingBoxAnnotation(
                        x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
                        class_name=yolo_class,
                        confidence=resource.get('confidence', 1.0),
                        annotator="screen_analyzer"
                    )
                    annotations.append(annotation)

        return annotations

    def _map_template_to_yolo_class(self, category: str, template_name: str) -> Optional[str]:
        """Mappe les résultats de template vers les classes YOLO"""
        mapping = {
            'resources': {
                'tree': 'resource_tree',
                'ore': 'resource_ore',
                'plant': 'resource_plant',
                'wheat': 'resource_plant',
                'iron': 'resource_ore'
            },
            'monsters': {
                'default': 'monster'
            },
            'npcs': {
                'default': 'npc'
            },
            'ui': {
                'button': 'ui_button',
                'window': 'ui_window',
                'inventory': 'ui_inventory'
            }
        }

        if category in mapping:
            if template_name in mapping[category]:
                return mapping[category][template_name]
            elif 'default' in mapping[category]:
                return mapping[category]['default']

        return None

    def _meets_collection_criteria(self, annotations: List[BoundingBoxAnnotation]) -> bool:
        """Vérifie si une frame répond aux critères de collecte"""
        if self.collection_filters['skip_empty_frames'] and len(annotations) == 0:
            return False

        if len(annotations) < self.collection_filters['min_objects']:
            return False

        if len(annotations) > self.collection_filters['max_objects']:
            return False

        # Vérification de la qualité des annotations
        high_conf_count = sum(1 for ann in annotations if ann.confidence >= 0.7)
        if high_conf_count == 0:
            return False

        return True

    def _is_similar_to_recent(self, frame: np.ndarray) -> bool:
        """Vérifie la similarité avec les frames récentes"""
        if len(self.frame_history) == 0:
            return False

        # Downscale pour comparaison rapide
        small_frame = cv2.resize(frame, (64, 64))
        small_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        for hist_frame in self.frame_history[-self.collection_filters['max_similar_frames']:]:
            # Corrélation simple
            correlation = cv2.matchTemplate(small_gray, hist_frame, cv2.TM_CCOEFF_NORMED)[0, 0]
            if correlation > self.similarity_threshold:
                return True

        return False

    def _add_to_history(self, frame: np.ndarray):
        """Ajoute une frame à l'historique"""
        small_frame = cv2.resize(frame, (64, 64))
        small_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        self.frame_history.append(small_gray)

        # Limite de l'historique
        if len(self.frame_history) > 20:
            self.frame_history.pop(0)

    def _save_annotated_frame(self, frame: np.ndarray,
                            annotations: List[BoundingBoxAnnotation]) -> bool:
        """Sauvegarde une frame avec ses annotations"""
        try:
            # Conversion annotations en format YOLO
            yolo_annotations = []
            for ann in annotations:
                h, w = frame.shape[:2]

                # Vérification des coordonnées
                x1, y1, x2, y2 = ann.x1, ann.y1, ann.x2, ann.y2
                if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                    continue

                # Conversion en format YOLO (centre normalisé + dimensions)
                center_x = ((x1 + x2) / 2) / w
                center_y = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h

                # ID de classe
                class_id = self._get_class_id(ann.class_name)

                yolo_annotations.append({
                    'class_id': class_id,
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height,
                    'confidence': ann.confidence,
                    'bbox': (x1, y1, x2, y2)
                })

            # Sauvegarde via YOLODatasetBuilder
            return self.yolo_builder.capture_training_image(frame, yolo_annotations)

        except Exception as e:
            logger.error(f"Erreur sauvegarde frame annotée: {e}")
            return False

    def _get_class_id(self, class_name: str) -> int:
        """Récupère l'ID de classe YOLO"""
        for class_id, name in self.config.dofus_classes.items():
            if name == class_name:
                return class_id

        # Classe inconnue -> ajout automatique
        new_id = max(self.config.dofus_classes.keys()) + 1
        self.config.dofus_classes[new_id] = class_name
        logger.info(f"Nouvelle classe ajoutée: {class_name} (ID: {new_id})")
        return new_id

    def stop_collection(self):
        """Arrête la collecte de données"""
        self.is_collecting = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5.0)

        logger.info("Collecte de données arrêtée")

    def _generate_collection_report(self):
        """Génère un rapport de collecte"""
        if self.stats['session_start']:
            duration = time.time() - self.stats['session_start']

            report = {
                'session_duration_hours': duration / 3600,
                'total_captures': self.stats['total_captures'],
                'auto_annotations': self.stats['auto_annotations'],
                'rejected_frames': self.stats['rejected_frames'],
                'classes_found': dict(self.stats['classes_found']),
                'capture_rate': self.stats['total_captures'] / (duration / 60),  # per minute
                'dataset_stats': self.yolo_builder.get_stats()
            }

            # Sauvegarde du rapport
            report_path = self.output_dir / f"collection_report_{int(time.time())}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info(f"Rapport de collecte généré: {report_path}")
            logger.info(f"Résumé: {report['total_captures']} captures, "
                       f"{len(report['classes_found'])} classes différentes")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de collecte"""
        stats = self.stats.copy()
        stats['is_collecting'] = self.is_collecting
        stats['frames_captured'] = self.frames_captured
        stats['dataset_stats'] = self.yolo_builder.get_stats()
        return stats

class AnnotationGUI:
    """Interface graphique pour annotation manuelle"""

    def __init__(self, data_collector: SmartDataCollector):
        self.data_collector = data_collector
        self.root = tk.Tk()
        self.root.title("DOFUS YOLO Data Collector & Annotator")
        self.root.geometry("800x600")

        self.current_image = None
        self.annotations = []
        self.selected_class = None

        self.setup_gui()

    def setup_gui(self):
        """Configure l'interface graphique"""
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Section contrôles collecte
        collection_frame = ttk.LabelFrame(main_frame, text="Collecte Automatique")
        collection_frame.pack(fill=tk.X, pady=(0, 10))

        # Boutons de collecte
        btn_frame = ttk.Frame(collection_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        self.start_btn = ttk.Button(btn_frame, text="Démarrer Collecte",
                                   command=self.start_collection)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.stop_btn = ttk.Button(btn_frame, text="Arrêter Collecte",
                                  command=self.stop_collection, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Configuration collecte
        config_frame = ttk.Frame(collection_frame)
        config_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(config_frame, text="Durée (heures):").pack(side=tk.LEFT)
        self.duration_var = tk.DoubleVar(value=1.0)
        duration_spin = ttk.Spinbox(config_frame, from_=0.1, to=12.0,
                                   textvariable=self.duration_var, width=10)
        duration_spin.pack(side=tk.LEFT, padx=(5, 20))

        ttk.Label(config_frame, text="Intervalle (sec):").pack(side=tk.LEFT)
        self.interval_var = tk.DoubleVar(value=2.0)
        interval_spin = ttk.Spinbox(config_frame, from_=0.5, to=10.0,
                                   textvariable=self.interval_var, width=10)
        interval_spin.pack(side=tk.LEFT, padx=5)

        # Section statistiques
        stats_frame = ttk.LabelFrame(main_frame, text="Statistiques")
        stats_frame.pack(fill=tk.X, pady=(0, 10))

        self.stats_text = tk.Text(stats_frame, height=6, state=tk.DISABLED)
        self.stats_text.pack(fill=tk.X, padx=5, pady=5)

        # Section annotation manuelle
        annotation_frame = ttk.LabelFrame(main_frame, text="Annotation Manuelle")
        annotation_frame.pack(fill=tk.BOTH, expand=True)

        # Boutons d'annotation
        ann_btn_frame = ttk.Frame(annotation_frame)
        ann_btn_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(ann_btn_frame, text="Charger Image",
                  command=self.load_image).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(ann_btn_frame, text="Sauvegarder Annotations",
                  command=self.save_annotations).pack(side=tk.LEFT, padx=5)

        # Sélection de classe
        ttk.Label(ann_btn_frame, text="Classe:").pack(side=tk.LEFT, padx=(20, 5))
        self.class_var = tk.StringVar()
        class_combo = ttk.Combobox(ann_btn_frame, textvariable=self.class_var,
                                  values=list(self.data_collector.config.dofus_classes.values()))
        class_combo.pack(side=tk.LEFT, padx=5)

        # Zone d'affichage image (placeholder)
        self.image_label = ttk.Label(annotation_frame, text="Aucune image chargée")
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        # Mise à jour périodique des stats
        self.update_stats()

    def start_collection(self):
        """Démarre la collecte automatique"""
        duration = self.duration_var.get()
        interval = self.interval_var.get()

        # Mise à jour configuration
        self.data_collector.collection_filters['capture_interval'] = interval

        # Démarrage
        self.data_collector.start_automated_collection(duration)

        # Interface
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

    def stop_collection(self):
        """Arrête la collecte"""
        self.data_collector.stop_collection()

        # Interface
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def load_image(self):
        """Charge une image pour annotation manuelle"""
        file_path = filedialog.askopenfilename(
            title="Sélectionner une image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_path:
            try:
                self.current_image = cv2.imread(file_path)
                self.image_label.config(text=f"Image chargée: {Path(file_path).name}")
                self.annotations = []
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de charger l'image: {e}")

    def save_annotations(self):
        """Sauvegarde les annotations manuelles"""
        if self.current_image is None:
            messagebox.showwarning("Attention", "Aucune image chargée")
            return

        # TODO: Implémenter l'interface d'annotation interactive
        messagebox.showinfo("Info", "Interface d'annotation interactive à implémenter")

    def update_stats(self):
        """Met à jour les statistiques affichées"""
        try:
            stats = self.data_collector.get_collection_stats()

            # Formatage des statistiques
            stats_text = f"""
Collecte en cours: {'Oui' if stats['is_collecting'] else 'Non'}
Frames capturées: {stats['frames_captured']}
Total captures: {stats['total_captures']}
Annotations auto: {stats['auto_annotations']}
Frames rejetées: {stats['rejected_frames']}

Classes trouvées: {len(stats.get('classes_found', {}))}
"""

            # Ajout détail des classes
            if stats.get('classes_found'):
                stats_text += "\nRépartition des classes:\n"
                for class_name, count in stats['classes_found'].items():
                    stats_text += f"  {class_name}: {count}\n"

            # Mise à jour de l'affichage
            self.stats_text.config(state=tk.NORMAL)
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, stats_text)
            self.stats_text.config(state=tk.DISABLED)

        except Exception as e:
            logger.error(f"Erreur mise à jour stats GUI: {e}")

        # Planifier la prochaine mise à jour
        self.root.after(2000, self.update_stats)  # Toutes les 2 secondes

    def run(self):
        """Lance l'interface graphique"""
        self.root.mainloop()

def main():
    """Point d'entrée principal"""
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Collecteur de données
    collector = SmartDataCollector()

    # Mode interface graphique ou ligne de commande
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--gui':
        # Mode GUI
        app = AnnotationGUI(collector)
        app.run()
    else:
        # Mode ligne de commande
        print("DOFUS YOLO Data Collector")
        print("========================")
        print("1. Collecte automatique (1h)")
        print("2. Collecte rapide (10min)")
        print("3. Test de configuration")

        choice = input("Votre choix (1-3): ").strip()

        if choice == '1':
            print("Démarrage collecte 1h...")
            collector.start_automated_collection(1.0)

            try:
                while collector.is_collecting:
                    time.sleep(10)
                    stats = collector.get_collection_stats()
                    print(f"Progression: {stats['frames_captured']} frames capturées")
            except KeyboardInterrupt:
                print("\nArrêt demandé par l'utilisateur")
                collector.stop_collection()

        elif choice == '2':
            print("Démarrage collecte 10min...")
            collector.start_automated_collection(1/6)  # 10 minutes

            try:
                while collector.is_collecting:
                    time.sleep(5)
            except KeyboardInterrupt:
                collector.stop_collection()

        elif choice == '3':
            print("Test de configuration...")
            success = collector.initialize_vision_modules()
            print(f"Modules vision: {'OK' if success else 'ERREUR'}")

            stats = collector.get_collection_stats()
            print(f"Configuration: {stats}")

        print("Terminé !")

if __name__ == "__main__":
    main()