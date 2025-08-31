#!/usr/bin/env python3
"""
Outil de calibration pour l'interface DOFUS.

Ce module permet de calibrer et configurer la détection des éléments
de l'interface du jeu DOFUS pour optimiser les performances du bot.

Fonctionnalités :
- Détection automatique de la fenêtre DOFUS
- Calibration des zones d'interface importantes
- Configuration des templates de reconnaissance
- Test de la précision de détection
- Sauvegarde des configurations de calibration

Usage:
    python calibrate.py --auto-detect
    python calibrate.py --manual-setup
    python calibrate.py --test-detection
    python calibrate.py --gui
"""

import sys
import os
import json
import argparse
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict

# Ajouter le répertoire racine au path Python
sys.path.insert(0, str(Path(__file__).parent))

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    from PIL import Image, ImageTk
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("Interface graphique non disponible - mode CLI uniquement")

try:
    import pyautogui
    import pygetwindow as gw
    SCREEN_AVAILABLE = True
    # Désactiver la protection fail-safe de pyautogui
    pyautogui.FAILSAFE = False
except ImportError:
    SCREEN_AVAILABLE = False
    print("Fonctionnalités d'écran non disponibles - certaines fonctions seront limitées")

try:
    from modules.vision.screen_analyzer import ScreenAnalyzer
    from modules.vision.template_matcher import TemplateMatcher
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    print("Modules de vision non disponibles")


@dataclass
class CalibrationZone:
    """Zone de calibration de l'interface."""
    name: str
    description: str
    x: int
    y: int
    width: int
    height: int
    confidence_threshold: float = 0.8
    template_path: Optional[str] = None
    color_range: Optional[Dict[str, Tuple[int, int, int]]] = None


@dataclass
class CalibrationConfig:
    """Configuration de calibration complète."""
    window_title: str = "Dofus"
    window_bounds: Tuple[int, int, int, int] = (0, 0, 1920, 1080)  # x, y, width, height
    resolution: Tuple[int, int] = (1920, 1080)
    created: str = ""
    last_updated: str = ""
    
    # Zones importantes de l'interface
    zones: Dict[str, CalibrationZone] = None
    
    # Configuration des couleurs importantes
    colors: Dict[str, Dict[str, Any]] = None
    
    # Paramètres de détection
    detection_settings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.zones is None:
            self.zones = self._get_default_zones()
        if self.colors is None:
            self.colors = self._get_default_colors()
        if self.detection_settings is None:
            self.detection_settings = self._get_default_detection_settings()
        if not self.created:
            self.created = datetime.now().isoformat()
            
    def _get_default_zones(self) -> Dict[str, CalibrationZone]:
        """Zones par défaut de l'interface DOFUS."""
        return {
            "chat_zone": CalibrationZone(
                name="chat_zone",
                description="Zone de chat",
                x=10, y=500, width=400, height=200,
                confidence_threshold=0.9
            ),
            "inventory": CalibrationZone(
                name="inventory",
                description="Inventaire",
                x=1500, y=100, width=400, height=600,
                confidence_threshold=0.85
            ),
            "character_stats": CalibrationZone(
                name="character_stats",
                description="Statistiques du personnage",
                x=20, y=20, width=300, height=150,
                confidence_threshold=0.9
            ),
            "minimap": CalibrationZone(
                name="minimap",
                description="Mini-carte",
                x=1650, y=20, width=250, height=200,
                confidence_threshold=0.8
            ),
            "action_bar": CalibrationZone(
                name="action_bar",
                description="Barre d'actions/sorts",
                x=600, y=800, width=720, height=100,
                confidence_threshold=0.9
            ),
            "game_area": CalibrationZone(
                name="game_area",
                description="Zone de jeu principale",
                x=400, y=100, width=1100, height=600,
                confidence_threshold=0.7
            ),
            "resource_nodes": CalibrationZone(
                name="resource_nodes",
                description="Noeuds de ressources",
                x=400, y=100, width=1100, height=600,
                confidence_threshold=0.75
            ),
            "monsters": CalibrationZone(
                name="monsters",
                description="Monstres/Ennemis",
                x=400, y=100, width=1100, height=600,
                confidence_threshold=0.8
            ),
            "npc": CalibrationZone(
                name="npc",
                description="PNJ",
                x=400, y=100, width=1100, height=600,
                confidence_threshold=0.8
            )
        }
    
    def _get_default_colors(self) -> Dict[str, Dict[str, Any]]:
        """Couleurs importantes par défaut."""
        return {
            "health_bar": {
                "color_rgb": (255, 0, 0),  # Rouge
                "hsv_range": {"lower": (0, 120, 120), "upper": (10, 255, 255)},
                "description": "Barre de vie"
            },
            "mana_bar": {
                "color_rgb": (0, 0, 255),  # Bleu
                "hsv_range": {"lower": (110, 120, 120), "upper": (130, 255, 255)},
                "description": "Barre de mana"
            },
            "experience_bar": {
                "color_rgb": (255, 255, 0),  # Jaune
                "hsv_range": {"lower": (20, 120, 120), "upper": (30, 255, 255)},
                "description": "Barre d'expérience"
            },
            "resource_highlight": {
                "color_rgb": (0, 255, 0),  # Vert
                "hsv_range": {"lower": (50, 120, 120), "upper": (70, 255, 255)},
                "description": "Surbrillance des ressources"
            },
            "enemy_highlight": {
                "color_rgb": (255, 100, 100),  # Rouge clair
                "hsv_range": {"lower": (0, 50, 150), "upper": (10, 200, 255)},
                "description": "Surbrillance des ennemis"
            }
        }
    
    def _get_default_detection_settings(self) -> Dict[str, Any]:
        """Paramètres de détection par défaut."""
        return {
            "screenshot_delay": 0.1,  # Délai entre les captures d'écran
            "detection_confidence": 0.8,  # Confidence générale
            "template_matching_method": cv2.TM_CCOEFF_NORMED,
            "color_detection_blur": 3,  # Flou pour la détection de couleur
            "edge_detection_threshold": (50, 150),  # Seuils pour la détection de contours
            "contour_area_min": 100,  # Aire minimale pour les contours
            "contour_area_max": 10000,  # Aire maximale pour les contours
            "adaptive_threshold": True,  # Seuil adaptatif
            "noise_reduction": True,  # Réduction du bruit
        }


class WindowDetector:
    """Détecteur de fenêtre DOFUS."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.WindowDetector")
        
    def find_dofus_window(self) -> Optional[Tuple[str, Tuple[int, int, int, int]]]:
        """Trouver la fenêtre DOFUS active."""
        if not SCREEN_AVAILABLE:
            self.logger.error("Fonctionnalités d'écran non disponibles")
            return None
            
        try:
            # Rechercher les fenêtres contenant "dofus" (insensible à la casse)
            windows = gw.getWindowsWithTitle("")
            dofus_windows = []
            
            for window in windows:
                if window.title and "dofus" in window.title.lower():
                    dofus_windows.append(window)
            
            if not dofus_windows:
                self.logger.warning("Aucune fenêtre DOFUS trouvée")
                return None
            
            # Prendre la première fenêtre DOFUS (ou la plus grande)
            target_window = max(dofus_windows, key=lambda w: w.width * w.height)
            
            bounds = (target_window.left, target_window.top, target_window.width, target_window.height)
            self.logger.info(f"Fenêtre DOFUS détectée : {target_window.title} - {bounds}")
            
            return target_window.title, bounds
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection de fenêtre : {e}")
            return None
    
    def capture_window(self, bounds: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Capturer une fenêtre spécifique."""
        if not SCREEN_AVAILABLE:
            return None
            
        try:
            x, y, width, height = bounds
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.logger.error(f"Erreur lors de la capture d'écran : {e}")
            return None


class CalibrationTool:
    """Outil principal de calibration."""
    
    def __init__(self, config_path: Path = None):
        self.config_path = config_path or Path("config/calibration.json")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.window_detector = WindowDetector()
        self.logger = logging.getLogger(f"{__name__}.CalibrationTool")
        
        self.config: Optional[CalibrationConfig] = None
        self.current_screenshot: Optional[np.ndarray] = None
        
        # Charger la configuration existante si disponible
        self.load_config()
        
    def load_config(self) -> bool:
        """Charger la configuration de calibration."""
        if not self.config_path.exists():
            self.logger.info("Aucune configuration trouvée, utilisation des valeurs par défaut")
            self.config = CalibrationConfig()
            return False
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruire les zones
            zones = {}
            if 'zones' in data and data['zones']:
                for zone_name, zone_data in data['zones'].items():
                    zones[zone_name] = CalibrationZone(**zone_data)
                data['zones'] = zones
                
            self.config = CalibrationConfig(**data)
            self.logger.info("Configuration de calibration chargée")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la configuration : {e}")
            self.config = CalibrationConfig()
            return False
    
    def save_config(self) -> bool:
        """Sauvegarder la configuration de calibration."""
        if not self.config:
            return False
            
        try:
            # Mettre à jour la date de modification
            self.config.last_updated = datetime.now().isoformat()
            
            # Convertir en dictionnaire
            data = asdict(self.config)
            
            # Convertir les zones en dictionnaires
            if 'zones' in data and data['zones']:
                zones_dict = {}
                for zone_name, zone in data['zones'].items():
                    if isinstance(zone, CalibrationZone):
                        zones_dict[zone_name] = asdict(zone)
                    else:
                        zones_dict[zone_name] = zone
                data['zones'] = zones_dict
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Configuration sauvegardée : {self.config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde : {e}")
            return False
    
    def auto_detect_window(self) -> bool:
        """Détecter automatiquement la fenêtre DOFUS."""
        detection_result = self.window_detector.find_dofus_window()
        
        if detection_result:
            title, bounds = detection_result
            if not self.config:
                self.config = CalibrationConfig()
                
            self.config.window_title = title
            self.config.window_bounds = bounds
            self.config.resolution = (bounds[2], bounds[3])
            
            self.logger.info(f"Fenêtre détectée automatiquement : {title}")
            return True
        else:
            self.logger.warning("Impossible de détecter la fenêtre DOFUS")
            return False
    
    def capture_current_screen(self) -> bool:
        """Capturer l'écran actuel."""
        if not self.config:
            self.logger.error("Configuration non initialisée")
            return False
            
        self.current_screenshot = self.window_detector.capture_window(self.config.window_bounds)
        
        if self.current_screenshot is not None:
            self.logger.info("Capture d'écran réussie")
            return True
        else:
            self.logger.error("Échec de la capture d'écran")
            return False
    
    def calibrate_zone(self, zone_name: str, x: int, y: int, width: int, height: int, confidence: float = 0.8) -> bool:
        """Calibrer une zone spécifique."""
        if not self.config:
            self.config = CalibrationConfig()
            
        # Créer ou mettre à jour la zone
        if zone_name in self.config.zones:
            zone = self.config.zones[zone_name]
            zone.x, zone.y, zone.width, zone.height = x, y, width, height
            zone.confidence_threshold = confidence
        else:
            zone = CalibrationZone(
                name=zone_name,
                description=f"Zone {zone_name}",
                x=x, y=y, width=width, height=height,
                confidence_threshold=confidence
            )
            self.config.zones[zone_name] = zone
        
        self.logger.info(f"Zone '{zone_name}' calibrée : ({x}, {y}, {width}, {height})")
        return True
    
    def test_zone_detection(self, zone_name: str) -> Dict[str, Any]:
        """Tester la détection d'une zone."""
        if not self.config or zone_name not in self.config.zones:
            return {"success": False, "error": "Zone non trouvée"}
            
        if not self.capture_current_screen():
            return {"success": False, "error": "Impossible de capturer l'écran"}
        
        zone = self.config.zones[zone_name]
        
        try:
            # Extraire la région de la zone
            region = self.current_screenshot[zone.y:zone.y+zone.height, zone.x:zone.x+zone.width]
            
            if region.size == 0:
                return {"success": False, "error": "Région vide"}
            
            # Analyse de base de la région
            mean_color = np.mean(region, axis=(0, 1))
            std_color = np.std(region, axis=(0, 1))
            
            # Détection de contours pour évaluer la complexité
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            return {
                "success": True,
                "zone_name": zone_name,
                "region_size": (zone.width, zone.height),
                "mean_color": mean_color.tolist(),
                "color_std": std_color.tolist(),
                "contours_count": len(contours),
                "complexity_score": len(contours) / (zone.width * zone.height) * 10000
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def auto_calibrate_interface(self) -> Dict[str, Any]:
        """Calibration automatique de l'interface."""
        results = {"zones_calibrated": [], "zones_failed": [], "screenshot_saved": False}
        
        # Détecter la fenêtre
        if not self.auto_detect_window():
            results["error"] = "Impossible de détecter la fenêtre DOFUS"
            return results
        
        # Capturer l'écran
        if not self.capture_current_screen():
            results["error"] = "Impossible de capturer l'écran"
            return results
        
        # Sauvegarder la capture pour référence
        try:
            screenshot_path = Path("config/calibration_screenshot.png")
            cv2.imwrite(str(screenshot_path), self.current_screenshot)
            results["screenshot_saved"] = True
            results["screenshot_path"] = str(screenshot_path)
        except:
            pass
        
        # Tenter de calibrer automatiquement certaines zones en fonction des couleurs
        self._auto_detect_ui_elements(results)
        
        return results
    
    def _auto_detect_ui_elements(self, results: Dict[str, Any]):
        """Détecter automatiquement les éléments de l'interface."""
        if self.current_screenshot is None:
            return
            
        try:
            # Convertir en HSV pour une meilleure détection de couleur
            hsv = cv2.cvtColor(self.current_screenshot, cv2.COLOR_BGR2HSV)
            h, w = hsv.shape[:2]
            
            # Zones probables basées sur la résolution et les conventions DOFUS
            probable_zones = {
                "chat_zone": (10, int(h * 0.6), int(w * 0.25), int(h * 0.35)),
                "inventory": (int(w * 0.75), int(h * 0.15), int(w * 0.25), int(h * 0.6)),
                "minimap": (int(w * 0.85), 10, int(w * 0.14), int(h * 0.2)),
                "character_stats": (10, 10, int(w * 0.2), int(h * 0.15)),
                "action_bar": (int(w * 0.3), int(h * 0.85), int(w * 0.4), int(h * 0.1))
            }
            
            for zone_name, (x, y, width, height) in probable_zones.items():
                # Vérifier que les coordonnées sont valides
                if x + width <= w and y + height <= h and x >= 0 and y >= 0:
                    if self.calibrate_zone(zone_name, x, y, width, height):
                        results["zones_calibrated"].append(zone_name)
                    else:
                        results["zones_failed"].append(zone_name)
                        
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection automatique : {e}")


class CalibrationGUI:
    """Interface graphique pour la calibration."""
    
    def __init__(self, calibration_tool: CalibrationTool):
        if not GUI_AVAILABLE:
            raise ImportError("Interface graphique non disponible")
            
        self.calibration_tool = calibration_tool
        self.root = tk.Tk()
        self.root.title("Calibration DOFUS - Interface")
        self.root.geometry("1200x800")
        
        self.current_image = None
        self.selection_start = None
        self.selection_end = None
        self.selection_rect = None
        
        self.create_interface()
        
    def create_interface(self):
        """Créer l'interface graphique."""
        # Frame principal avec onglets
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Onglet principal - Calibration
        main_frame = ttk.Frame(notebook)
        notebook.add(main_frame, text="Calibration Principale")
        
        # Onglet test
        test_frame = ttk.Frame(notebook)
        notebook.add(test_frame, text="Test de Détection")
        
        # Onglet configuration
        config_frame = ttk.Frame(notebook)
        notebook.add(config_frame, text="Configuration")
        
        self.create_main_tab(main_frame)
        self.create_test_tab(test_frame)
        self.create_config_tab(config_frame)
        
    def create_main_tab(self, parent):
        """Créer l'onglet principal de calibration."""
        # Frame de contrôles en haut
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="x", pady=(0, 10))
        
        # Boutons de contrôle
        ttk.Button(control_frame, text="Détecter Fenêtre", 
                  command=self.detect_window).pack(side="left", padx=(0, 5))
        ttk.Button(control_frame, text="Capturer Écran", 
                  command=self.capture_screen).pack(side="left", padx=(0, 5))
        ttk.Button(control_frame, text="Auto-Calibrer", 
                  command=self.auto_calibrate).pack(side="left", padx=(0, 5))
        ttk.Button(control_frame, text="Sauvegarder", 
                  command=self.save_config).pack(side="left", padx=(0, 5))
        ttk.Button(control_frame, text="Charger", 
                  command=self.load_config).pack(side="left", padx=(0, 5))
        
        # Frame principal avec image et zones
        main_content = ttk.Frame(parent)
        main_content.pack(fill="both", expand=True)
        
        # Frame gauche pour l'image
        image_frame = ttk.LabelFrame(main_content, text="Capture d'écran", padding="5")
        image_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        # Canvas pour l'affichage de l'image
        self.image_canvas = tk.Canvas(image_frame, bg="gray90")
        self.image_canvas.pack(fill="both", expand=True)
        
        # Bind des événements de souris pour la sélection
        self.image_canvas.bind("<Button-1>", self.start_selection)
        self.image_canvas.bind("<B1-Motion>", self.update_selection)
        self.image_canvas.bind("<ButtonRelease-1>", self.end_selection)
        
        # Frame droite pour les zones
        zones_frame = ttk.LabelFrame(main_content, text="Zones de Calibration", padding="5")
        zones_frame.pack(side="right", fill="y", padx=(5, 0))
        zones_frame.config(width=300)
        
        # Liste des zones
        self.zones_listbox = tk.Listbox(zones_frame, height=15)
        self.zones_listbox.pack(fill="both", expand=True, pady=(0, 5))
        self.zones_listbox.bind("<<ListboxSelect>>", self.on_zone_select)
        
        # Frame pour les informations de zone
        zone_info_frame = ttk.Frame(zones_frame)
        zone_info_frame.pack(fill="x", pady=(5, 0))
        
        # Champs d'information de zone
        ttk.Label(zone_info_frame, text="Nom:").grid(row=0, column=0, sticky="w")
        self.zone_name_var = tk.StringVar()
        ttk.Entry(zone_info_frame, textvariable=self.zone_name_var, width=20).grid(row=0, column=1, padx=(5, 0))
        
        ttk.Label(zone_info_frame, text="X:").grid(row=1, column=0, sticky="w")
        self.zone_x_var = tk.IntVar()
        ttk.Entry(zone_info_frame, textvariable=self.zone_x_var, width=10).grid(row=1, column=1, padx=(5, 0), sticky="w")
        
        ttk.Label(zone_info_frame, text="Y:").grid(row=2, column=0, sticky="w")
        self.zone_y_var = tk.IntVar()
        ttk.Entry(zone_info_frame, textvariable=self.zone_y_var, width=10).grid(row=2, column=1, padx=(5, 0), sticky="w")
        
        ttk.Label(zone_info_frame, text="Largeur:").grid(row=3, column=0, sticky="w")
        self.zone_width_var = tk.IntVar()
        ttk.Entry(zone_info_frame, textvariable=self.zone_width_var, width=10).grid(row=3, column=1, padx=(5, 0), sticky="w")
        
        ttk.Label(zone_info_frame, text="Hauteur:").grid(row=4, column=0, sticky="w")
        self.zone_height_var = tk.IntVar()
        ttk.Entry(zone_info_frame, textvariable=self.zone_height_var, width=10).grid(row=4, column=1, padx=(5, 0), sticky="w")
        
        # Boutons de zone
        zone_buttons_frame = ttk.Frame(zones_frame)
        zone_buttons_frame.pack(fill="x", pady=(5, 0))
        
        ttk.Button(zone_buttons_frame, text="Ajouter Zone", 
                  command=self.add_zone).pack(side="left", padx=(0, 2))
        ttk.Button(zone_buttons_frame, text="Modifier", 
                  command=self.modify_zone).pack(side="left", padx=(2, 2))
        ttk.Button(zone_buttons_frame, text="Supprimer", 
                  command=self.delete_zone).pack(side="left", padx=(2, 0))
        
        # Charger les zones existantes
        self.refresh_zones_list()
        
    def create_test_tab(self, parent):
        """Créer l'onglet de test."""
        # Frame de contrôles
        test_control_frame = ttk.Frame(parent)
        test_control_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(test_control_frame, text="Tester Toutes les Zones", 
                  command=self.test_all_zones).pack(side="left", padx=(0, 5))
        ttk.Button(test_control_frame, text="Tester Zone Sélectionnée", 
                  command=self.test_selected_zone).pack(side="left", padx=(0, 5))
        
        # Zone de résultats
        results_frame = ttk.LabelFrame(parent, text="Résultats des Tests", padding="5")
        results_frame.pack(fill="both", expand=True)
        
        self.results_text = tk.Text(results_frame, height=20, state="disabled")
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.pack(side="left", fill="both", expand=True)
        results_scrollbar.pack(side="right", fill="y")
        
    def create_config_tab(self, parent):
        """Créer l'onglet de configuration."""
        config_info_frame = ttk.LabelFrame(parent, text="Configuration Actuelle", padding="10")
        config_info_frame.pack(fill="both", expand=True)
        
        self.config_text = tk.Text(config_info_frame, height=25, state="disabled")
        config_scrollbar = ttk.Scrollbar(config_info_frame, orient="vertical", command=self.config_text.yview)
        self.config_text.configure(yscrollcommand=config_scrollbar.set)
        
        self.config_text.pack(side="left", fill="both", expand=True)
        config_scrollbar.pack(side="right", fill="y")
        
        # Boutons de configuration
        config_buttons_frame = ttk.Frame(parent)
        config_buttons_frame.pack(fill="x", pady=(10, 0))
        
        ttk.Button(config_buttons_frame, text="Actualiser", 
                  command=self.refresh_config_display).pack(side="left", padx=(0, 5))
        ttk.Button(config_buttons_frame, text="Exporter", 
                  command=self.export_config).pack(side="left", padx=(0, 5))
        ttk.Button(config_buttons_frame, text="Importer", 
                  command=self.import_config).pack(side="left", padx=(0, 5))
        
        # Afficher la configuration actuelle
        self.refresh_config_display()
    
    def detect_window(self):
        """Détecter la fenêtre DOFUS."""
        if self.calibration_tool.auto_detect_window():
            messagebox.showinfo("Succès", "Fenêtre DOFUS détectée automatiquement!")
            self.refresh_config_display()
        else:
            messagebox.showerror("Erreur", "Impossible de détecter la fenêtre DOFUS")
    
    def capture_screen(self):
        """Capturer l'écran."""
        if self.calibration_tool.capture_current_screen():
            self.display_screenshot()
            messagebox.showinfo("Succès", "Capture d'écran réussie!")
        else:
            messagebox.showerror("Erreur", "Échec de la capture d'écran")
    
    def display_screenshot(self):
        """Afficher la capture d'écran dans le canvas."""
        if self.calibration_tool.current_screenshot is None:
            return
            
        try:
            # Convertir l'image OpenCV en PIL puis en PhotoImage
            screenshot_rgb = cv2.cvtColor(self.calibration_tool.current_screenshot, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(screenshot_rgb)
            
            # Redimensionner pour s'adapter au canvas
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                pil_image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
                
            self.current_image = ImageTk.PhotoImage(pil_image)
            
            # Afficher l'image
            self.image_canvas.delete("all")
            self.image_canvas.create_image(0, 0, anchor="nw", image=self.current_image)
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'afficher l'image : {e}")
    
    def start_selection(self, event):
        """Commencer la sélection de zone."""
        self.selection_start = (event.x, event.y)
        if self.selection_rect:
            self.image_canvas.delete(self.selection_rect)
    
    def update_selection(self, event):
        """Mettre à jour la sélection."""
        if self.selection_start:
            if self.selection_rect:
                self.image_canvas.delete(self.selection_rect)
            self.selection_rect = self.image_canvas.create_rectangle(
                self.selection_start[0], self.selection_start[1],
                event.x, event.y, outline="red", width=2
            )
    
    def end_selection(self, event):
        """Terminer la sélection."""
        if self.selection_start:
            self.selection_end = (event.x, event.y)
            
            # Calculer les coordonnées réelles
            x1, y1 = self.selection_start
            x2, y2 = self.selection_end
            
            # Assurer que x1,y1 sont en haut à gauche
            x, y = min(x1, x2), min(y1, y2)
            width, height = abs(x2 - x1), abs(y2 - y1)
            
            # Mettre à jour les champs
            self.zone_x_var.set(x)
            self.zone_y_var.set(y)
            self.zone_width_var.set(width)
            self.zone_height_var.set(height)
    
    def add_zone(self):
        """Ajouter une nouvelle zone."""
        name = self.zone_name_var.get().strip()
        if not name:
            messagebox.showerror("Erreur", "Veuillez entrer un nom pour la zone")
            return
            
        x = self.zone_x_var.get()
        y = self.zone_y_var.get()
        width = self.zone_width_var.get()
        height = self.zone_height_var.get()
        
        if width <= 0 or height <= 0:
            messagebox.showerror("Erreur", "Largeur et hauteur doivent être positives")
            return
            
        if self.calibration_tool.calibrate_zone(name, x, y, width, height):
            self.refresh_zones_list()
            messagebox.showinfo("Succès", f"Zone '{name}' ajoutée!")
        else:
            messagebox.showerror("Erreur", "Impossible d'ajouter la zone")
    
    def modify_zone(self):
        """Modifier la zone sélectionnée."""
        selection = self.zones_listbox.curselection()
        if not selection:
            messagebox.showerror("Erreur", "Veuillez sélectionner une zone à modifier")
            return
            
        zone_name = self.zones_listbox.get(selection[0])
        
        x = self.zone_x_var.get()
        y = self.zone_y_var.get()
        width = self.zone_width_var.get()
        height = self.zone_height_var.get()
        
        if self.calibration_tool.calibrate_zone(zone_name, x, y, width, height):
            messagebox.showinfo("Succès", f"Zone '{zone_name}' modifiée!")
        else:
            messagebox.showerror("Erreur", "Impossible de modifier la zone")
    
    def delete_zone(self):
        """Supprimer la zone sélectionnée."""
        selection = self.zones_listbox.curselection()
        if not selection:
            messagebox.showerror("Erreur", "Veuillez sélectionner une zone à supprimer")
            return
            
        zone_name = self.zones_listbox.get(selection[0])
        
        if messagebox.askyesno("Confirmation", f"Supprimer la zone '{zone_name}' ?"):
            if self.calibration_tool.config and zone_name in self.calibration_tool.config.zones:
                del self.calibration_tool.config.zones[zone_name]
                self.refresh_zones_list()
                messagebox.showinfo("Succès", f"Zone '{zone_name}' supprimée!")
    
    def on_zone_select(self, event):
        """Gérer la sélection d'une zone."""
        selection = self.zones_listbox.curselection()
        if selection:
            zone_name = self.zones_listbox.get(selection[0])
            if (self.calibration_tool.config and 
                zone_name in self.calibration_tool.config.zones):
                zone = self.calibration_tool.config.zones[zone_name]
                
                self.zone_name_var.set(zone.name)
                self.zone_x_var.set(zone.x)
                self.zone_y_var.set(zone.y)
                self.zone_width_var.set(zone.width)
                self.zone_height_var.set(zone.height)
    
    def refresh_zones_list(self):
        """Actualiser la liste des zones."""
        self.zones_listbox.delete(0, tk.END)
        
        if self.calibration_tool.config and self.calibration_tool.config.zones:
            for zone_name in self.calibration_tool.config.zones:
                self.zones_listbox.insert(tk.END, zone_name)
    
    def auto_calibrate(self):
        """Lancer la calibration automatique."""
        result = self.calibration_tool.auto_calibrate_interface()
        
        success_count = len(result.get("zones_calibrated", []))
        failed_count = len(result.get("zones_failed", []))
        
        message = f"Calibration automatique terminée:\n"
        message += f"Zones calibrées: {success_count}\n"
        message += f"Zones échouées: {failed_count}"
        
        if "error" in result:
            message += f"\nErreur: {result['error']}"
            
        messagebox.showinfo("Calibration Automatique", message)
        
        self.refresh_zones_list()
        if result.get("screenshot_saved"):
            self.display_screenshot()
    
    def test_all_zones(self):
        """Tester toutes les zones."""
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        
        if not self.calibration_tool.config or not self.calibration_tool.config.zones:
            self.results_text.insert(tk.END, "Aucune zone à tester\n")
            self.results_text.config(state="disabled")
            return
        
        for zone_name in self.calibration_tool.config.zones:
            result = self.calibration_tool.test_zone_detection(zone_name)
            
            self.results_text.insert(tk.END, f"\n=== Test de la zone '{zone_name}' ===\n")
            
            if result["success"]:
                self.results_text.insert(tk.END, f"✓ Test réussi\n")
                self.results_text.insert(tk.END, f"Taille: {result['region_size']}\n")
                self.results_text.insert(tk.END, f"Couleur moyenne: {result['mean_color']}\n")
                self.results_text.insert(tk.END, f"Contours détectés: {result['contours_count']}\n")
                self.results_text.insert(tk.END, f"Score de complexité: {result['complexity_score']:.2f}\n")
            else:
                self.results_text.insert(tk.END, f"✗ Test échoué: {result['error']}\n")
        
        self.results_text.config(state="disabled")
    
    def test_selected_zone(self):
        """Tester la zone sélectionnée."""
        selection = self.zones_listbox.curselection()
        if not selection:
            messagebox.showerror("Erreur", "Veuillez sélectionner une zone à tester")
            return
            
        zone_name = self.zones_listbox.get(selection[0])
        result = self.calibration_tool.test_zone_detection(zone_name)
        
        if result["success"]:
            message = f"Test de la zone '{zone_name}' réussi!\n\n"
            message += f"Taille: {result['region_size']}\n"
            message += f"Contours: {result['contours_count']}\n"
            message += f"Complexité: {result['complexity_score']:.2f}"
            messagebox.showinfo("Test Zone", message)
        else:
            messagebox.showerror("Test Zone", f"Test échoué: {result['error']}")
    
    def save_config(self):
        """Sauvegarder la configuration."""
        if self.calibration_tool.save_config():
            messagebox.showinfo("Succès", "Configuration sauvegardée!")
        else:
            messagebox.showerror("Erreur", "Impossible de sauvegarder la configuration")
    
    def load_config(self):
        """Charger la configuration."""
        if self.calibration_tool.load_config():
            self.refresh_zones_list()
            self.refresh_config_display()
            messagebox.showinfo("Succès", "Configuration chargée!")
        else:
            messagebox.showerror("Erreur", "Impossible de charger la configuration")
    
    def refresh_config_display(self):
        """Actualiser l'affichage de configuration."""
        self.config_text.config(state="normal")
        self.config_text.delete(1.0, tk.END)
        
        if self.calibration_tool.config:
            config_str = json.dumps(asdict(self.calibration_tool.config), indent=2, ensure_ascii=False)
            self.config_text.insert(tk.END, config_str)
        else:
            self.config_text.insert(tk.END, "Aucune configuration chargée")
            
        self.config_text.config(state="disabled")
    
    def export_config(self):
        """Exporter la configuration."""
        if not self.calibration_tool.config:
            messagebox.showerror("Erreur", "Aucune configuration à exporter")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Exporter Configuration",
            defaultextension=".json",
            filetypes=[("Fichiers JSON", "*.json")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(asdict(self.calibration_tool.config), f, indent=2, ensure_ascii=False)
                messagebox.showinfo("Succès", f"Configuration exportée : {file_path}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible d'exporter : {e}")
    
    def import_config(self):
        """Importer une configuration."""
        file_path = filedialog.askopenfilename(
            title="Importer Configuration",
            filetypes=[("Fichiers JSON", "*.json")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Reconstruire les zones
                zones = {}
                if 'zones' in data and data['zones']:
                    for zone_name, zone_data in data['zones'].items():
                        zones[zone_name] = CalibrationZone(**zone_data)
                    data['zones'] = zones
                    
                self.calibration_tool.config = CalibrationConfig(**data)
                
                self.refresh_zones_list()
                self.refresh_config_display()
                messagebox.showinfo("Succès", "Configuration importée!")
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible d'importer : {e}")
    
    def run(self):
        """Lancer l'interface graphique."""
        self.root.mainloop()


def create_argument_parser():
    """Créer le parser d'arguments."""
    parser = argparse.ArgumentParser(
        description="Outil de calibration pour l'interface DOFUS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s --gui                    # Interface graphique
  %(prog)s --auto-detect           # Détection automatique
  %(prog)s --manual-setup          # Configuration manuelle
  %(prog)s --test-detection        # Test des détections
        """
    )
    
    # Modes d'opération
    parser.add_argument("--gui", action="store_true", help="Lancer l'interface graphique")
    parser.add_argument("--auto-detect", action="store_true", help="Détection automatique")
    parser.add_argument("--manual-setup", action="store_true", help="Configuration manuelle")
    parser.add_argument("--test-detection", action="store_true", help="Tester la détection")
    
    # Configuration
    parser.add_argument("--config-path", help="Chemin du fichier de configuration")
    parser.add_argument("--debug", action="store_true", help="Mode debug")
    
    # Actions spécifiques
    parser.add_argument("--save-screenshot", help="Sauvegarder une capture d'écran")
    parser.add_argument("--export-config", help="Exporter la configuration")
    
    return parser

def main():
    """Fonction principale."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configuration du logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Créer l'outil de calibration
    config_path = Path(args.config_path) if args.config_path else None
    calibration_tool = CalibrationTool(config_path)
    
    try:
        if args.gui:
            if not GUI_AVAILABLE:
                print("Interface graphique non disponible")
                return 1
            gui = CalibrationGUI(calibration_tool)
            gui.run()
            
        elif args.auto_detect:
            print("Détection automatique de l'interface DOFUS...")
            result = calibration_tool.auto_calibrate_interface()
            
            print(f"Zones calibrées: {len(result.get('zones_calibrated', []))}")
            print(f"Zones échouées: {len(result.get('zones_failed', []))}")
            
            if result.get("screenshot_saved"):
                print(f"Capture sauvegardée: {result['screenshot_path']}")
            
            if calibration_tool.save_config():
                print("Configuration sauvegardée avec succès")
            
        elif args.manual_setup:
            print("Configuration manuelle - Non implémenté en CLI")
            print("Utilisez --gui pour la configuration manuelle")
            
        elif args.test_detection:
            print("Test de détection des zones...")
            
            if not calibration_tool.config or not calibration_tool.config.zones:
                print("Aucune zone configurée pour les tests")
                return 1
            
            for zone_name in calibration_tool.config.zones:
                result = calibration_tool.test_zone_detection(zone_name)
                
                if result["success"]:
                    print(f"✓ Zone '{zone_name}': OK")
                    print(f"  Contours: {result['contours_count']}")
                    print(f"  Complexité: {result['complexity_score']:.2f}")
                else:
                    print(f"✗ Zone '{zone_name}': {result['error']}")
        else:
            # Mode par défaut
            print("=== OUTIL DE CALIBRATION DOFUS ===")
            print("Utilisez --help pour voir les options disponibles")
            print("Recommandé: --gui pour l'interface graphique complète")
            
    except KeyboardInterrupt:
        print("\nInterruption utilisateur")
        return 0
    except Exception as e:
        logging.error(f"Erreur critique : {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)