"""
Démonstration complète du système YOLO pour DOFUS
Script interactif pour tester et valider le nouveau système de vision
"""

import cv2
import numpy as np
import time
import logging
import argparse
from pathlib import Path
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration du logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import des modules développés
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from modules.vision.yolo_detector import DofusYOLODetector, YOLOConfig, YOLODatasetBuilder, YOLOTrainer
    from modules.vision.vision_orchestrator import VisionOrchestrator, VisionConfig
    from modules.vision.screen_analyzer import ScreenAnalyzer
    from tools.data_collector import SmartDataCollector
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Erreur import modules: {e}")
    MODULES_AVAILABLE = False

class YOLODemoGUI:
    """Interface graphique de démonstration YOLO"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DOFUS YOLO Vision System - Démonstration")
        self.root.geometry("1200x800")

        # État de la démo
        self.orchestrator = None
        self.current_image = None
        self.detection_results = None
        self.demo_running = False

        # Variables UI
        self.strategy_var = tk.StringVar(value="yolo")
        self.confidence_var = tk.DoubleVar(value=0.6)
        self.roi_var = tk.StringVar(value="full_screen")

        self.setup_gui()
        self.init_modules()

    def setup_gui(self):
        """Configure l'interface graphique"""
        # Menu principal
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Menu Fichier
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Fichier", menu=file_menu)
        file_menu.add_command(label="Charger Image", command=self.load_image)
        file_menu.add_command(label="Capture Écran", command=self.capture_screen)
        file_menu.add_command(label="Sauvegarder Résultats", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Quitter", command=self.root.quit)

        # Menu Outils
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Outils", menu=tools_menu)
        tools_menu.add_command(label="Collecteur de Données", command=self.open_data_collector)
        tools_menu.add_command(label="Entraîner Modèle", command=self.train_model)
        tools_menu.add_command(label="Rapport Performance", command=self.show_performance_report)

        # Frame principal
        main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Panel de contrôle (gauche)
        control_frame = ttk.Frame(main_frame)
        main_frame.add(control_frame, weight=1)

        # Panel d'affichage (droite)
        display_frame = ttk.Frame(main_frame)
        main_frame.add(display_frame, weight=3)

        self.setup_control_panel(control_frame)
        self.setup_display_panel(display_frame)

    def setup_control_panel(self, parent):
        """Configure le panel de contrôle"""
        # Configuration Vision
        config_frame = ttk.LabelFrame(parent, text="Configuration Vision")
        config_frame.pack(fill=tk.X, pady=(0, 10))

        # Stratégie
        ttk.Label(config_frame, text="Stratégie:").pack(anchor=tk.W, padx=5, pady=2)
        strategy_frame = ttk.Frame(config_frame)
        strategy_frame.pack(fill=tk.X, padx=5, pady=2)

        strategies = [("YOLO", "yolo"), ("Template", "template"), ("Hybride", "hybrid")]
        for text, value in strategies:
            ttk.Radiobutton(strategy_frame, text=text, variable=self.strategy_var,
                           value=value, command=self.update_strategy).pack(side=tk.LEFT)

        # Seuil de confiance
        ttk.Label(config_frame, text="Confiance min:").pack(anchor=tk.W, padx=5, pady=2)
        conf_scale = ttk.Scale(config_frame, from_=0.1, to=1.0,
                              variable=self.confidence_var, orient=tk.HORIZONTAL)
        conf_scale.pack(fill=tk.X, padx=5, pady=2)

        self.conf_label = ttk.Label(config_frame, text="0.6")
        self.conf_label.pack(anchor=tk.W, padx=5)

        conf_scale.bind("<Motion>", self.update_confidence_label)

        # ROI
        ttk.Label(config_frame, text="Région d'Intérêt:").pack(anchor=tk.W, padx=5, pady=2)
        roi_combo = ttk.Combobox(config_frame, textvariable=self.roi_var,
                                values=["full_screen", "center_game", "ui_area", "minimap"])
        roi_combo.pack(fill=tk.X, padx=5, pady=2)

        # Actions
        action_frame = ttk.LabelFrame(parent, text="Actions")
        action_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(action_frame, text="Analyser Image",
                  command=self.analyze_current_image).pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(action_frame, text="Démo Temps Réel",
                  command=self.toggle_realtime_demo).pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(action_frame, text="Test Benchmark",
                  command=self.run_benchmark).pack(fill=tk.X, padx=5, pady=2)

        # Statistiques
        stats_frame = ttk.LabelFrame(parent, text="Statistiques")
        stats_frame.pack(fill=tk.BOTH, expand=True)

        self.stats_text = tk.Text(stats_frame, height=15, width=30)
        stats_scroll = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL,
                                   command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scroll.set)

        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        stats_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_display_panel(self, parent):
        """Configure le panel d'affichage"""
        # Zone d'affichage image
        self.canvas = tk.Canvas(parent, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Status bar
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(0, 5))

        self.status_var = tk.StringVar(value="Prêt")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)

        self.fps_var = tk.StringVar(value="FPS: 0")
        ttk.Label(status_frame, textvariable=self.fps_var).pack(side=tk.RIGHT)

    def init_modules(self):
        """Initialise les modules de vision"""
        try:
            if not MODULES_AVAILABLE:
                self.status_var.set("Erreur: Modules non disponibles")
                return

            # Configuration
            vision_config = VisionConfig(
                primary_method="yolo",
                fallback_method="template",
                confidence_threshold_yolo=0.6,
                enable_adaptive_switching=True
            )

            # Orchestrateur
            self.orchestrator = VisionOrchestrator(config=vision_config)

            # Initialisation
            init_config = {
                'vision_orchestrator': {
                    'parallel_processing': True,
                    'enable_caching': True
                }
            }

            if self.orchestrator.initialize(init_config):
                self.status_var.set("Modules initialisés avec succès")
                self.update_stats_display()
            else:
                self.status_var.set("Erreur: Échec initialisation modules")

        except Exception as e:
            logger.error(f"Erreur initialisation: {e}")
            self.status_var.set(f"Erreur: {str(e)}")

    def load_image(self):
        """Charge une image depuis un fichier"""
        file_path = filedialog.askopenfilename(
            title="Sélectionner une image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_path:
            try:
                self.current_image = cv2.imread(file_path)
                if self.current_image is not None:
                    self.display_image(self.current_image)
                    self.status_var.set(f"Image chargée: {Path(file_path).name}")
                else:
                    messagebox.showerror("Erreur", "Impossible de charger l'image")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur chargement: {e}")

    def capture_screen(self):
        """Capture l'écran actuel"""
        try:
            if self.orchestrator and self.orchestrator.screen_analyzer:
                screenshot = self.orchestrator.screen_analyzer.get_current_screenshot()
                if screenshot is not None:
                    self.current_image = screenshot
                    self.display_image(self.current_image)
                    self.status_var.set("Capture d'écran effectuée")
                else:
                    messagebox.showwarning("Attention", "Impossible de capturer l'écran")
            else:
                messagebox.showwarning("Attention", "Module screen analyzer non disponible")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur capture: {e}")

    def display_image(self, image: np.ndarray, detections: List[Dict] = None):
        """Affiche une image avec les détections optionnelles"""
        try:
            # Copie pour annotation
            display_img = image.copy()

            # Dessin des détections
            if detections:
                self.draw_detections(display_img, detections)

            # Redimensionnement pour affichage
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                h, w = display_img.shape[:2]
                scale = min(canvas_width / w, canvas_height / h, 1.0)

                if scale < 1.0:
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    display_img = cv2.resize(display_img, (new_w, new_h))

            # Conversion pour Tkinter
            display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(display_img_rgb)
            self.photo = ImageTk.PhotoImage(pil_image)

            # Affichage
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=self.photo, anchor=tk.CENTER
            )

        except Exception as e:
            logger.error(f"Erreur affichage image: {e}")

    def draw_detections(self, image: np.ndarray, detections: List[Dict]):
        """Dessine les détections sur l'image"""
        colors = {
            'player': (0, 255, 0),
            'monster': (0, 0, 255),
            'npc': (255, 0, 0),
            'resource': (255, 255, 0),
            'ui': (255, 0, 255),
            'default': (128, 128, 128)
        }

        for detection in detections:
            bbox = detection.get('bounding_box', [0, 0, 0, 0])
            confidence = detection.get('confidence', 0.0)
            obj_type = detection.get('template_name', 'unknown')

            # Couleur selon le type
            color = colors.get(obj_type.split('_')[0], colors['default'])

            # Rectangle
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Label
            label = f"{obj_type}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

            # Fond du label
            cv2.rectangle(image,
                         (bbox[0], bbox[1] - label_size[1] - 5),
                         (bbox[0] + label_size[0], bbox[1]),
                         color, -1)

            # Texte du label
            cv2.putText(image, label,
                       (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def analyze_current_image(self):
        """Analyse l'image actuelle avec le système de vision"""
        if self.current_image is None:
            messagebox.showwarning("Attention", "Aucune image chargée")
            return

        if not self.orchestrator:
            messagebox.showerror("Erreur", "Orchestrateur non initialisé")
            return

        try:
            self.status_var.set("Analyse en cours...")

            # Mise à jour de la configuration
            self.orchestrator.set_strategy(self.strategy_var.get())

            # Analyse
            start_time = time.time()
            result = self.orchestrator.analyze(self.current_image, roi=self.roi_var.get())
            analysis_time = time.time() - start_time

            self.detection_results = result

            # Extraction des détections pour affichage
            all_detections = []
            confidence_threshold = self.confidence_var.get()

            if 'detections_by_class' in result:
                for class_detections in result['detections_by_class'].values():
                    for detection in class_detections:
                        if detection.get('confidence', 0.0) >= confidence_threshold:
                            all_detections.append(detection)

            # Affichage avec détections
            self.display_image(self.current_image, all_detections)

            # Mise à jour des statistiques
            self.update_stats_with_results(result, analysis_time)

            self.status_var.set(f"Analyse terminée - {len(all_detections)} détections")

        except Exception as e:
            logger.error(f"Erreur analyse: {e}")
            messagebox.showerror("Erreur", f"Erreur analyse: {e}")
            self.status_var.set("Erreur d'analyse")

    def toggle_realtime_demo(self):
        """Active/désactive la démo temps réel"""
        if not self.demo_running:
            if not self.orchestrator or not self.orchestrator.screen_analyzer:
                messagebox.showwarning("Attention", "Screen analyzer non disponible")
                return

            self.demo_running = True
            self.demo_thread = threading.Thread(target=self.realtime_demo_loop, daemon=True)
            self.demo_thread.start()
            self.status_var.set("Démo temps réel démarrée")
        else:
            self.demo_running = False
            self.status_var.set("Démo temps réel arrêtée")

    def realtime_demo_loop(self):
        """Boucle de démo temps réel"""
        fps_counter = 0
        last_fps_time = time.time()

        while self.demo_running:
            try:
                # Capture
                screenshot = self.orchestrator.screen_analyzer.get_current_screenshot()
                if screenshot is None:
                    time.sleep(0.1)
                    continue

                # Analyse
                result = self.orchestrator.analyze(screenshot)

                # Extraction détections
                all_detections = []
                confidence_threshold = self.confidence_var.get()

                if 'detections_by_class' in result:
                    for class_detections in result['detections_by_class'].values():
                        for detection in class_detections:
                            if detection.get('confidence', 0.0) >= confidence_threshold:
                                all_detections.append(detection)

                # Affichage (thread-safe)
                self.root.after(0, lambda: self.display_image(screenshot, all_detections))

                # FPS
                fps_counter += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    fps = fps_counter / (current_time - last_fps_time)
                    self.root.after(0, lambda: self.fps_var.set(f"FPS: {fps:.1f}"))
                    fps_counter = 0
                    last_fps_time = current_time

                # Pause pour réguler la fréquence
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Erreur démo temps réel: {e}")
                time.sleep(0.5)

    def run_benchmark(self):
        """Lance un benchmark de performance"""
        if not self.orchestrator:
            messagebox.showerror("Erreur", "Orchestrateur non initialisé")
            return

        messagebox.showinfo("Benchmark", "Le benchmark va commencer. Cela peut prendre quelques minutes.")

        try:
            # Test avec différentes stratégies
            strategies = ['yolo', 'template', 'hybrid']
            results = {}

            # Image de test
            if self.current_image is None:
                # Utiliser une capture d'écran
                if self.orchestrator.screen_analyzer:
                    test_image = self.orchestrator.screen_analyzer.get_current_screenshot()
                    if test_image is None:
                        # Image aléatoire pour test
                        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                else:
                    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            else:
                test_image = self.current_image

            self.status_var.set("Benchmark en cours...")

            for strategy in strategies:
                self.orchestrator.set_strategy(strategy)

                # Plusieurs runs pour moyenne
                times = []
                detection_counts = []

                for i in range(10):
                    start_time = time.time()
                    result = self.orchestrator.analyze(test_image)
                    end_time = time.time()

                    times.append(end_time - start_time)
                    detection_counts.append(result.get('total_detections', 0))

                    self.status_var.set(f"Benchmark {strategy}: {i+1}/10")

                results[strategy] = {
                    'avg_time': np.mean(times),
                    'std_time': np.std(times),
                    'avg_detections': np.mean(detection_counts),
                    'fps': 1.0 / np.mean(times)
                }

            # Affichage des résultats
            self.show_benchmark_results(results)

        except Exception as e:
            logger.error(f"Erreur benchmark: {e}")
            messagebox.showerror("Erreur", f"Erreur benchmark: {e}")

    def show_benchmark_results(self, results: Dict[str, Dict]):
        """Affiche les résultats de benchmark"""
        # Fenêtre des résultats
        result_window = tk.Toplevel(self.root)
        result_window.title("Résultats Benchmark")
        result_window.geometry("600x400")

        # Texte des résultats
        results_text = tk.Text(result_window, font=("Courier", 10))
        results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Formatage
        text_content = "RÉSULTATS BENCHMARK YOLO VISION SYSTEM\n"
        text_content += "=" * 50 + "\n\n"

        for strategy, metrics in results.items():
            text_content += f"Stratégie: {strategy.upper()}\n"
            text_content += f"  Temps moyen: {metrics['avg_time']:.3f}s (±{metrics['std_time']:.3f}s)\n"
            text_content += f"  FPS moyen: {metrics['fps']:.1f}\n"
            text_content += f"  Détections moyennes: {metrics['avg_detections']:.1f}\n"
            text_content += "\n"

        # Recommandation
        best_fps = max(results.values(), key=lambda x: x['fps'])
        best_strategy = next(k for k, v in results.items() if v['fps'] == best_fps['fps'])

        text_content += f"RECOMMANDATION: Utiliser '{best_strategy}' pour les meilleures performances\n"
        text_content += f"(FPS: {best_fps['fps']:.1f})\n"

        results_text.insert(1.0, text_content)
        results_text.config(state=tk.DISABLED)

    def update_strategy(self):
        """Met à jour la stratégie de l'orchestrateur"""
        if self.orchestrator:
            self.orchestrator.set_strategy(self.strategy_var.get())

    def update_confidence_label(self, event=None):
        """Met à jour le label de confiance"""
        self.conf_label.config(text=f"{self.confidence_var.get():.2f}")

    def update_stats_display(self):
        """Met à jour l'affichage des statistiques"""
        if self.orchestrator:
            try:
                state = self.orchestrator.get_state()
                report = self.orchestrator.get_performance_report()

                stats_text = "=== ÉTAT DU SYSTÈME ===\n"
                stats_text += f"Status: {state['status']}\n"
                stats_text += f"Stratégie: {state['current_strategy']}\n\n"

                stats_text += "=== MODULES ===\n"
                for module_name, module_state in state.get('modules', {}).items():
                    status = module_state.get('status', 'unknown')
                    stats_text += f"{module_name}: {status}\n"

                stats_text += "\n=== STATISTIQUES ===\n"
                stats = state.get('stats', {})
                for key, value in stats.items():
                    if isinstance(value, float):
                        stats_text += f"{key}: {value:.3f}\n"
                    else:
                        stats_text += f"{key}: {value}\n"

                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(1.0, stats_text)

            except Exception as e:
                logger.error(f"Erreur mise à jour stats: {e}")

        # Programmation de la prochaine mise à jour
        self.root.after(2000, self.update_stats_display)

    def update_stats_with_results(self, result: Dict, analysis_time: float):
        """Met à jour les stats avec les derniers résultats"""
        try:
            current_stats = self.stats_text.get(1.0, tk.END)

            # Ajout des nouveaux résultats
            new_stats = f"\n=== DERNIÈRE ANALYSE ===\n"
            new_stats += f"Temps: {analysis_time:.3f}s\n"
            new_stats += f"Méthode: {result.get('method', 'unknown')}\n"
            new_stats += f"Détections: {result.get('total_detections', 0)}\n"

            if 'detections_by_class' in result:
                new_stats += "\nDétections par classe:\n"
                for class_name, detections in result['detections_by_class'].items():
                    new_stats += f"  {class_name}: {len(detections)}\n"

            # Remplacement de la section précédente
            lines = current_stats.split('\n')
            try:
                start_idx = lines.index("=== DERNIÈRE ANALYSE ===")
                lines = lines[:start_idx]
            except ValueError:
                pass

            updated_stats = '\n'.join(lines) + new_stats

            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, updated_stats)

        except Exception as e:
            logger.error(f"Erreur mise à jour résultats: {e}")

    def save_results(self):
        """Sauvegarde les derniers résultats"""
        if self.detection_results is None:
            messagebox.showwarning("Attention", "Aucun résultat à sauvegarder")
            return

        file_path = filedialog.asksaveasfilename(
            title="Sauvegarder les résultats",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("Tous fichiers", "*.*")]
        )

        if file_path:
            try:
                # Préparation des données
                save_data = {
                    'timestamp': self.detection_results.get('timestamp', datetime.now()).isoformat(),
                    'results': self.detection_results,
                    'config': {
                        'strategy': self.strategy_var.get(),
                        'confidence_threshold': self.confidence_var.get(),
                        'roi': self.roi_var.get()
                    }
                }

                # Sauvegarde
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)

                messagebox.showinfo("Succès", f"Résultats sauvegardés: {file_path}")

            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur sauvegarde: {e}")

    def open_data_collector(self):
        """Ouvre l'interface du collecteur de données"""
        try:
            from tools.data_collector import AnnotationGUI, SmartDataCollector

            collector = SmartDataCollector()
            annotation_gui = AnnotationGUI(collector)
            annotation_gui.run()

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'ouvrir le collecteur: {e}")

    def train_model(self):
        """Lance l'entraînement d'un modèle YOLO"""
        response = messagebox.askyesno(
            "Entraînement",
            "L'entraînement d'un modèle YOLO peut prendre plusieurs heures.\n"
            "Voulez-vous continuer?"
        )

        if response:
            try:
                # Configuration d'entraînement
                dataset_path = "data/collected/yolo_dataset/dataset.yaml"

                if not Path(dataset_path).exists():
                    messagebox.showerror(
                        "Erreur",
                        f"Dataset non trouvé: {dataset_path}\n"
                        "Utilisez d'abord le collecteur de données."
                    )
                    return

                # Fenêtre de progression
                train_window = tk.Toplevel(self.root)
                train_window.title("Entraînement en cours...")
                train_window.geometry("400x200")

                progress_label = ttk.Label(train_window, text="Initialisation...")
                progress_label.pack(pady=20)

                progress_bar = ttk.Progressbar(train_window, mode='indeterminate')
                progress_bar.pack(fill=tk.X, padx=20, pady=10)
                progress_bar.start()

                # Thread d'entraînement
                def train_thread():
                    try:
                        trainer = YOLOTrainer(YOLOConfig())
                        model_path = trainer.train_model(dataset_path, epochs=50)

                        train_window.after(0, lambda: progress_bar.stop())
                        train_window.after(0, lambda: messagebox.showinfo(
                            "Succès", f"Modèle entraîné: {model_path}"
                        ))
                        train_window.after(0, train_window.destroy)

                    except Exception as e:
                        train_window.after(0, lambda: progress_bar.stop())
                        train_window.after(0, lambda: messagebox.showerror(
                            "Erreur", f"Erreur entraînement: {e}"
                        ))
                        train_window.after(0, train_window.destroy)

                threading.Thread(target=train_thread, daemon=True).start()

            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de lancer l'entraînement: {e}")

    def show_performance_report(self):
        """Affiche un rapport de performance détaillé"""
        if not self.orchestrator:
            messagebox.showwarning("Attention", "Orchestrateur non initialisé")
            return

        try:
            report = self.orchestrator.get_performance_report()

            # Fenêtre du rapport
            report_window = tk.Toplevel(self.root)
            report_window.title("Rapport de Performance")
            report_window.geometry("700x500")

            # Notebook pour organisation
            notebook = ttk.Notebook(report_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Onglet Performance
            perf_frame = ttk.Frame(notebook)
            notebook.add(perf_frame, text="Performance")

            perf_text = tk.Text(perf_frame, font=("Courier", 9))
            perf_text.pack(fill=tk.BOTH, expand=True)

            # Formatage du rapport
            report_content = "RAPPORT DE PERFORMANCE DÉTAILLÉ\n"
            report_content += "=" * 40 + "\n\n"

            report_content += f"Stratégie actuelle: {report['current_strategy']}\n"
            report_content += f"Adaptations: {report['strategy_adaptations']}\n\n"

            for method, metrics in report['performance_by_method'].items():
                if metrics.get('available'):
                    report_content += f"{method.upper()}:\n"
                    report_content += f"  Temps moyen: {metrics['avg_detection_time']:.3f}s\n"
                    report_content += f"  FPS: {metrics['fps']:.1f}\n"
                    report_content += f"  Confiance: {metrics['avg_confidence']:.3f}\n"
                    report_content += f"  Précision estimée: {metrics['estimated_accuracy']:.3f}\n"
                    report_content += f"  Total détections: {metrics['total_detections']}\n\n"

            # Cache
            cache_perf = report['cache_performance']
            report_content += f"CACHE:\n"
            report_content += f"  Taux de réussite: {cache_perf['hit_rate']:.2%}\n"
            report_content += f"  Requêtes totales: {cache_perf['total_requests']}\n"

            perf_text.insert(1.0, report_content)
            perf_text.config(state=tk.DISABLED)

            # Onglet Stats globales
            stats_frame = ttk.Frame(notebook)
            notebook.add(stats_frame, text="Statistiques")

            stats_text = tk.Text(stats_frame, font=("Courier", 9))
            stats_text.pack(fill=tk.BOTH, expand=True)

            stats_content = json.dumps(report['overall_stats'], indent=2)
            stats_text.insert(1.0, stats_content)
            stats_text.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur génération rapport: {e}")

    def run(self):
        """Lance l'interface graphique"""
        self.root.mainloop()

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Démonstration YOLO DOFUS Vision System")
    parser.add_argument("--mode", choices=["gui", "cli", "benchmark"],
                       default="gui", help="Mode de fonctionnement")
    parser.add_argument("--image", type=str, help="Chemin vers une image de test")
    parser.add_argument("--strategy", choices=["yolo", "template", "hybrid"],
                       default="yolo", help="Stratégie de vision à utiliser")

    args = parser.parse_args()

    if args.mode == "gui":
        # Mode interface graphique
        app = YOLODemoGUI()
        app.run()

    elif args.mode == "cli":
        # Mode ligne de commande
        print("DOFUS YOLO Vision Demo - Mode CLI")
        print("=" * 40)

        if not MODULES_AVAILABLE:
            print("Erreur: Modules de vision non disponibles")
            return

        # Initialisation
        orchestrator = VisionOrchestrator()
        if not orchestrator.initialize({}):
            print("Erreur: Échec initialisation orchestrateur")
            return

        orchestrator.set_strategy(args.strategy)

        # Test avec image
        if args.image and Path(args.image).exists():
            print(f"Analyse de l'image: {args.image}")

            image = cv2.imread(args.image)
            start_time = time.time()
            result = orchestrator.analyze(image)
            analysis_time = time.time() - start_time

            print(f"Temps d'analyse: {analysis_time:.3f}s")
            print(f"Méthode utilisée: {result.get('method', 'unknown')}")
            print(f"Détections totales: {result.get('total_detections', 0)}")

            if 'detections_by_class' in result:
                print("\nDétections par classe:")
                for class_name, detections in result['detections_by_class'].items():
                    print(f"  {class_name}: {len(detections)}")

        else:
            print("Aucune image spécifiée ou fichier non trouvé")

    elif args.mode == "benchmark":
        # Mode benchmark
        print("DOFUS YOLO Vision Benchmark")
        print("=" * 30)

        if not MODULES_AVAILABLE:
            print("Erreur: Modules de vision non disponibles")
            return

        # TODO: Implémenter benchmark CLI
        print("Benchmark CLI à implémenter")

if __name__ == "__main__":
    main()