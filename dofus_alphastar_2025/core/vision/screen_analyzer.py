"""
Module d'analyse visuelle avancée pour DOFUS
Utilise les dernières techniques de computer vision optimisées pour le jeu
"""

import cv2
import numpy as np
import pytesseract
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import re
from collections import defaultdict
from pathlib import Path

# Imports pour capture d'écran optimisée
import mss
import platform

# Import conditionnel Windows
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

if IS_WINDOWS:
    try:
        import win32gui
        import win32con
        import win32api
        WIN32_AVAILABLE = True
    except ImportError:
        WIN32_AVAILABLE = False
else:
    WIN32_AVAILABLE = False

# Configuration OCR multiplateforme
if IS_WINDOWS:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    # Linux: tesseract généralement dans PATH
    # Si problème: sudo apt-get install tesseract-ocr
    pass

# Import des modules internes
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from engine.module_interface import IAnalysisModule, ModuleStatus
from engine.event_bus import EventType, EventPriority

# Import YOLO optionnel pour détection d'objets avancée
try:
    from .yolo_detector import DofusYOLODetector, YOLOConfig
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLO non disponible - fonctionnement en mode template uniquement")


@dataclass
class ScreenRegion:
    """Définit une région d'intérêt sur l'écran"""
    name: str
    x: int
    y: int
    width: int
    height: int
    update_frequency: float = 1.0  # Fréquence de mise à jour en secondes
    
    def get_bbox(self) -> Tuple[int, int, int, int]:
        """Retourne la bounding box (x1, y1, x2, y2)"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def contains_point(self, x: int, y: int) -> bool:
        """Vérifie si un point est dans cette région"""
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)


@dataclass
class DetectedElement:
    """Élément détecté sur l'écran"""
    element_type: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]
    additional_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def center_point(self) -> Tuple[int, int]:
        """Retourne le point central de l'élément"""
        x1, y1, x2, y2 = self.bounding_box
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class DofusUIDetector:
    """
    Détecteur spécialisé pour l'interface DOFUS
    Reconnaît tous les éléments UI critiques
    """
    
    def __init__(self):
        self.ui_templates = self._load_ui_templates()
        self.text_patterns = self._setup_text_patterns()
        self.color_ranges = self._setup_color_ranges()
        
    def _load_ui_templates(self) -> Dict[str, np.ndarray]:
        """Charge les templates des éléments UI"""
        templates = {}
        template_dir = Path("assets/templates/ui")
        
        if template_dir.exists():
            for template_file in template_dir.glob("*.png"):
                template_name = template_file.stem
                template_img = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
                templates[template_name] = template_img
        
        # Templates basiques si pas de fichiers
        return templates
    
    def _setup_text_patterns(self) -> Dict[str, re.Pattern]:
        """Configure les patterns de reconnaissance de texte"""
        return {
            "hp_bar": re.compile(r"(\d+)\s*/\s*(\d+)"),
            "pa_pm": re.compile(r"PA\s*:\s*(\d+)\s*/\s*(\d+).*PM\s*:\s*(\d+)\s*/\s*(\d+)"),
            "level": re.compile(r"Niveau\s*(\d+)"),
            "kamas": re.compile(r"(\d+(?:\s*\d+)*)\s*kamas?", re.IGNORECASE),
            "coordinates": re.compile(r"\[(-?\d+),(-?\d+)\]"),
            "combat_turn": re.compile(r"Tour\s*(\d+)", re.IGNORECASE),
            "time_remaining": re.compile(r"(\d+):(\d+)")
        }
    
    def _setup_color_ranges(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Configure les ranges de couleurs pour détection"""
        return {
            "hp_green": (np.array([40, 50, 50]), np.array([80, 255, 255])),
            "hp_red": (np.array([0, 50, 50]), np.array([20, 255, 255])),
            "pa_blue": (np.array([100, 50, 50]), np.array([130, 255, 255])),
            "pm_blue": (np.array([90, 50, 50]), np.array([110, 255, 255])),
            "gold_text": (np.array([15, 100, 100]), np.array([25, 255, 255])),
            "red_danger": (np.array([0, 100, 100]), np.array([10, 255, 255]))
        }
    
    def detect_hp_bar(self, image: np.ndarray, region: ScreenRegion) -> Optional[Dict[str, Any]]:
        """Détecte la barre de HP et extrait les valeurs"""
        x, y, w, h = region.x, region.y, region.width, region.height
        roi = image[y:y+h, x:x+w]
        
        # Détection par couleur (barre verte/rouge)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Masque pour HP vert
        green_mask = cv2.inRange(hsv, *self.color_ranges["hp_green"])
        red_mask = cv2.inRange(hsv, *self.color_ranges["hp_red"])
        
        # Calcul pourcentage par analyse des pixels
        total_pixels = green_mask.size
        green_pixels = cv2.countNonZero(green_mask)
        red_pixels = cv2.countNonZero(red_mask)
        
        if green_pixels + red_pixels > total_pixels * 0.1:  # Au moins 10% de pixels HP
            hp_percentage = (green_pixels / (green_pixels + red_pixels)) * 100
            
            # Tentative OCR pour valeurs exactes
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            ocr_text = pytesseract.image_to_string(gray_roi, config='--psm 8 -c tessedit_char_whitelist=0123456789/')
            
            current_hp = max_hp = None
            hp_match = self.text_patterns["hp_bar"].search(ocr_text)
            if hp_match:
                current_hp = int(hp_match.group(1))
                max_hp = int(hp_match.group(2))
            
            return {
                "hp_percentage": hp_percentage,
                "current_hp": current_hp,
                "max_hp": max_hp,
                "detection_method": "color_analysis"
            }
        
        return None
    
    def detect_action_points(self, image: np.ndarray, region: ScreenRegion) -> Optional[Dict[str, Any]]:
        """Détecte les points d'action (PA/PM)"""
        x, y, w, h = region.x, region.y, region.width, region.height
        roi = image[y:y+h, x:x+w]
        
        # OCR pour les valeurs PA/PM
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Préprocessing pour améliorer l'OCR
        gray_roi = cv2.resize(gray_roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray_roi = cv2.medianBlur(gray_roi, 3)
        
        ocr_text = pytesseract.image_to_string(gray_roi, config='--psm 8')
        
        # Recherche pattern PA/PM
        pa_pm_match = self.text_patterns["pa_pm"].search(ocr_text)
        if pa_pm_match:
            return {
                "current_pa": int(pa_pm_match.group(1)),
                "max_pa": int(pa_pm_match.group(2)),
                "current_pm": int(pa_pm_match.group(3)),
                "max_pm": int(pa_pm_match.group(4)),
                "detection_method": "ocr"
            }
        
        # Méthode alternative: comptage d'icônes PA/PM
        return self._count_action_points_icons(roi)
    
    def _count_action_points_icons(self, roi: np.ndarray) -> Dict[str, Any]:
        """Compte les icônes PA/PM visuellement"""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Détection des icônes bleues (PA/PM)
        blue_mask = cv2.inRange(hsv, *self.color_ranges["pa_blue"])
        
        # Morphologie pour nettoyer
        kernel = np.ones((3,3), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        
        # Détection de contours (icônes)
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrage par taille d'icône
        icons = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 <= area <= 500:  # Taille typique d'icône PA/PM
                icons.append(contour)
        
        # Estimation PA/PM basée sur le nombre d'icônes
        estimated_points = len(icons)
        
        return {
            "estimated_pa": min(estimated_points, 6),  # Max 6 PA habituellement
            "estimated_pm": 3,  # Valeur par défaut
            "detection_method": "icon_counting",
            "icons_detected": estimated_points
        }
    
    def detect_combat_interface(self, image: np.ndarray) -> Dict[str, Any]:
        """Détecte si l'interface de combat est active"""
        # Recherche d'éléments caractéristiques du combat
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Template matching pour boutons de combat
        combat_elements = {}
        
        # Détection par couleur des boutons de combat
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Boutons de sorts (généralement avec bordure dorée)
        gold_mask = cv2.inRange(hsv, *self.color_ranges["gold_text"])
        gold_contours, _ = cv2.findContours(gold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        spell_buttons = []
        for contour in gold_contours:
            area = cv2.contourArea(contour)
            if 800 <= area <= 3000:  # Taille typique bouton sort
                x, y, w, h = cv2.boundingRect(contour)
                spell_buttons.append((x, y, w, h))
        
        # Détection timer de tour
        ocr_result = pytesseract.image_to_string(gray, config='--psm 6')
        time_match = self.text_patterns["time_remaining"].search(ocr_result)
        turn_match = self.text_patterns["combat_turn"].search(ocr_result)
        
        return {
            "in_combat": len(spell_buttons) >= 4,  # Au moins 4 sorts visibles
            "spell_buttons_detected": len(spell_buttons),
            "spell_button_positions": spell_buttons,
            "turn_time_remaining": float(time_match.group(1)) * 60 + float(time_match.group(2)) if time_match else None,
            "turn_number": int(turn_match.group(1)) if turn_match else None
        }
    
    def detect_resources(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Détecte les ressources récoltables sur l'écran"""
        resources = []
        
        # Templates de ressources communes
        resource_templates = {
            "wheat": "assets/templates/resources/wheat.png",
            "iron": "assets/templates/resources/iron.png",
            "ash_tree": "assets/templates/resources/ash.png"
        }
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for resource_type, template_path in resource_templates.items():
            if Path(template_path).exists():
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                
                # Template matching multi-échelle
                scales = [0.8, 1.0, 1.2]
                for scale in scales:
                    resized_template = cv2.resize(template, None, fx=scale, fy=scale)
                    result = cv2.matchTemplate(gray, resized_template, cv2.TM_CCOEFF_NORMED)
                    
                    # Seuil de détection
                    threshold = 0.7
                    locations = np.where(result >= threshold)
                    
                    for pt in zip(*locations[::-1]):
                        h, w = resized_template.shape
                        resources.append({
                            "type": resource_type,
                            "position": (pt[0] + w//2, pt[1] + h//2),
                            "bounding_box": (pt[0], pt[1], pt[0] + w, pt[1] + h),
                            "confidence": float(result[pt[1], pt[0]]),
                            "scale": scale
                        })
        
        # Suppression des détections doublonnées
        resources = self._remove_duplicate_detections(resources, min_distance=50)
        
        return resources
    
    def _remove_duplicate_detections(self, detections: List[Dict], min_distance: int = 30) -> List[Dict]:
        """Supprime les détections doublonnées proches"""
        if not detections:
            return []
        
        # Tri par confiance décroissante
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        
        filtered = []
        for detection in detections:
            pos = detection["position"]
            
            # Vérifier distance avec détections déjà acceptées
            too_close = False
            for accepted in filtered:
                accepted_pos = accepted["position"]
                distance = np.sqrt((pos[0] - accepted_pos[0])**2 + (pos[1] - accepted_pos[1])**2)
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                filtered.append(detection)
        
        return filtered


class ScreenAnalyzer(IAnalysisModule):
    """
    Module d'analyse d'écran principal pour DOFUS
    Coordonne tous les sous-systèmes de vision
    """
    
    def __init__(self, name: str = "screen_analyzer"):
        super().__init__(name)

        # Configuration logging
        self.logger = logging.getLogger(f"{__name__}.ScreenAnalyzer")

        # Détecteur UI spécialisé
        self.ui_detector = DofusUIDetector()

        # NOUVEAU: Détecteur YOLO optionnel pour objets dynamiques
        self.yolo_detector = None
        self.enable_yolo = False
        self.yolo_zones = ["center_game"]  # Zones où YOLO est actif

        # Stratégie de détection hybride
        self.detection_strategy = {
            "center_game": "hybrid",      # YOLO + Template pour objets du jeu
            "ui_area": "template",        # Template pour UI précise
            "minimap": "template",        # Template pour minimap
            "combat_interface": "ui"      # UI detector existant
        }
        
        # Configuration de capture d'écran
        # MSS sera créé à la volée pour éviter les problèmes de thread
        self.dofus_window_title = "DOFUS"
        self.dofus_window_handle = None
        
        # Régions d'intérêt définies pour DOFUS
        self.screen_regions = self._setup_screen_regions()
        
        # Cache des dernières détections
        self.detection_cache = {}
        self.cache_timestamps = {}
        self.cache_duration = 0.5  # 500ms de cache
        
        # Statistiques de performance
        self.performance_stats = {
            "captures_per_second": 0,
            "analysis_time_avg": 0,
            "detections_per_frame": 0,
            "cache_hit_rate": 0
        }
        
        # Thread pour capture continue
        self.capture_thread = None
        self.is_capturing = False
        self.current_screenshot = None
        self.screenshot_lock = threading.RLock()
        
        # Configuration adaptative
        self.adaptive_config = {
            "capture_fps": 10,  # FPS de capture
            "analysis_fps": 5,  # FPS d'analyse complète
            "ui_update_fps": 2,  # FPS mise à jour UI critique
        }
    
    def _setup_screen_regions(self) -> Dict[str, ScreenRegion]:
        """Configure les régions d'écran importantes pour DOFUS"""
        # Coordonnées typiques pour résolution 1920x1080
        return {
            "character_stats": ScreenRegion("character_stats", 20, 20, 200, 100, 0.5),
            "hp_bar": ScreenRegion("hp_bar", 50, 50, 150, 30, 0.2),
            "pa_pm_bar": ScreenRegion("pa_pm_bar", 50, 85, 150, 25, 0.2),
            "chat_area": ScreenRegion("chat_area", 20, 600, 400, 200, 1.0),
            "spell_bar": ScreenRegion("spell_bar", 400, 950, 600, 80, 0.3),
            "minimap": ScreenRegion("minimap", 1700, 50, 200, 200, 2.0),
            "inventory": ScreenRegion("inventory", 1400, 300, 500, 600, 5.0),
            "combat_interface": ScreenRegion("combat_interface", 300, 800, 800, 200, 0.1),
            "center_game": ScreenRegion("center_game", 400, 200, 1120, 600, 1.0)
        }
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialise le module d'analyse d'écran"""
        try:
            self.logger.info("Initialisation du module d'analyse d'écran")

            # Recherche de la fenêtre DOFUS
            self.dofus_window_handle = self._find_dofus_window()
            if not self.dofus_window_handle:
                self.logger.warning("Fenêtre DOFUS non trouvée, utilisation écran complet")

            # NOUVEAU: Initialisation YOLO si activé
            yolo_config = config.get('yolo', {})
            if yolo_config.get('enable', False) and YOLO_AVAILABLE:
                self._initialize_yolo_detector(yolo_config)

            # Test de capture d'écran
            test_screenshot = self._capture_screen()
            if test_screenshot is None:
                self.logger.error("Impossible de capturer l'écran")
                # On continue quand même, la capture pourrait marcher plus tard
                # return False

            # Démarrage capture continue
            self._start_continuous_capture()

            self.status = ModuleStatus.ACTIVE
            self.logger.info(f"Module d'analyse d'écran initialisé (YOLO: {'✓' if self.enable_yolo else '✗'})")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur initialisation screen analyzer: {e}")
            self.set_error(str(e))
            return False
    
    def _find_dofus_window(self) -> Optional[int]:
        """Trouve la fenêtre DOFUS"""
        try:
            import win32gui
            
            def enum_windows_callback(hwnd, windows):
                try:
                    window_text = win32gui.GetWindowText(hwnd)
                    if "DOFUS" in window_text and win32gui.IsWindowVisible(hwnd):
                        windows.append(hwnd)
                except:
                    pass
                return True
            
            windows = []
            win32gui.EnumWindows(enum_windows_callback, windows)
            
            return windows[0] if windows else None
            
        except ImportError:
            self.logger.warning("win32gui non disponible, capture écran complet uniquement")
            return None
        except Exception as e:
            self.logger.error(f"Erreur recherche fenêtre DOFUS: {e}")
            return None
    
    def _start_continuous_capture(self) -> None:
        """Démarre la capture continue d'écran"""
        if not self.is_capturing:
            self.is_capturing = True
            self.capture_thread = threading.Thread(
                target=self._continuous_capture_loop,
                name="ScreenCapture",
                daemon=True
            )
            self.capture_thread.start()
            self.logger.info("Capture d'écran continue démarrée")
    
    def _continuous_capture_loop(self) -> None:
        """Boucle de capture continue"""
        capture_interval = 1.0 / self.adaptive_config["capture_fps"]
        
        while self.is_capturing:
            start_time = time.perf_counter()
            
            try:
                screenshot = self._capture_screen()
                if screenshot is not None:
                    with self.screenshot_lock:
                        self.current_screenshot = screenshot
                
            except Exception as e:
                self.logger.error(f"Erreur capture d'écran: {e}")
            
            # Régulation FPS
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, capture_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _capture_screen(self) -> Optional[np.ndarray]:
        """Capture l'écran ou la fenêtre DOFUS (multiplateforme)"""
        try:
            # Créer une nouvelle instance MSS pour chaque capture pour éviter les problèmes de thread
            with mss.mss() as sct:
                if self.dofus_window_handle:
                    try:
                        if IS_WINDOWS and WIN32_AVAILABLE:
                            import win32gui
                            # Capture de la fenêtre DOFUS spécifiquement (Windows)
                            rect = win32gui.GetWindowRect(self.dofus_window_handle)
                        elif IS_LINUX:
                            # Linux: utiliser informations fenêtre stockées
                            # Assume window_rect est stocké lors de la détection
                            if hasattr(self, 'window_rect') and self.window_rect:
                                rect = self.window_rect
                            else:
                                # Fallback: écran complet
                                rect = None
                        else:
                            rect = None

                        if rect:
                            monitor = {
                            "top": rect[1],
                            "left": rect[0], 
                            "width": rect[2] - rect[0],
                            "height": rect[3] - rect[1]
                        }
                    except:
                        # Si erreur, utiliser écran complet
                        monitor = sct.monitors[1] if len(sct.monitors) > 1 else {"top": 0, "left": 0, "width": 1920, "height": 1080}
                else:
                    # Capture écran complet - utiliser monitor 1 pour l'écran principal (0 est le combiné)
                    monitor = sct.monitors[1] if len(sct.monitors) > 1 else {"top": 0, "left": 0, "width": 1920, "height": 1080}
                
                # Capture avec MSS (plus rapide)
                screenshot = sct.grab(monitor)
                
                # Conversion en format OpenCV
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                return img
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la capture: {e}")
            return None
    
    def update(self, game_state: Any) -> Optional[Dict[str, Any]]:
        """Met à jour l'analyse d'écran"""
        try:
            if not self.is_active():
                return None
            
            # Récupération screenshot actuel
            with self.screenshot_lock:
                if self.current_screenshot is None:
                    return None
                screenshot = self.current_screenshot.copy()
            
            # Analyse complète
            analysis_result = self.analyze(screenshot)
            
            return {
                "shared_data": analysis_result,
                "module_status": "active"
            }
            
        except Exception as e:
            self.logger.error(f"Erreur update screen analyzer: {e}")
            return None
    
    def analyze(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """
        Analyse complète d'un screenshot
        
        Args:
            screenshot: Image à analyser
            
        Returns:
            Dict contenant tous les éléments détectés
        """
        start_time = time.perf_counter()
        
        try:
            analysis_result = {
                "timestamp": datetime.now(),
                "character": {},
                "combat": {},
                "map": {},
                "ui_elements": [],
                "resources": [],
                "performance": {}
            }
            
            # === ANALYSE DU PERSONNAGE ===
            character_data = self._analyze_character_info(screenshot)
            if character_data:
                analysis_result["character"] = character_data
            
            # === ANALYSE DU COMBAT ===
            combat_data = self._analyze_combat_interface(screenshot)
            if combat_data:
                analysis_result["combat"] = combat_data
            
            # === ANALYSE DE LA CARTE ===
            map_data = self._analyze_map_elements(screenshot)
            if map_data:
                analysis_result["map"] = map_data
            
            # === DÉTECTION DES RESSOURCES ===
            resources = self.ui_detector.detect_resources(screenshot)
            if resources:
                analysis_result["resources"] = resources

            # === NOUVEAU: DÉTECTION D'OBJETS DYNAMIQUES (YOLO) ===
            if self.enable_yolo:
                dynamic_objects = self._analyze_dynamic_objects(screenshot)
                if dynamic_objects:
                    analysis_result["dynamic_objects"] = dynamic_objects
                    # Fusion avec ressources existantes si pertinent
                    analysis_result["resources"].extend(dynamic_objects.get("resources", []))
            
            # === MÉTRIQUES DE PERFORMANCE ===
            analysis_time = time.perf_counter() - start_time
            analysis_result["performance"] = {
                "analysis_time": analysis_time,
                "elements_detected": len(analysis_result["ui_elements"]),
                "resources_detected": len(resources)
            }
            
            # Mise à jour des stats
            self._update_performance_stats(analysis_time, analysis_result)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse: {e}")
            return {"error": str(e), "timestamp": datetime.now()}
    
    def _analyze_character_info(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Analyse les informations du personnage"""
        character_info = {}
        
        # Analyse HP
        hp_region = self.screen_regions["hp_bar"]
        hp_data = self.ui_detector.detect_hp_bar(screenshot, hp_region)
        if hp_data:
            character_info.update(hp_data)
        
        # Analyse PA/PM
        pa_pm_region = self.screen_regions["pa_pm_bar"]
        pa_pm_data = self.ui_detector.detect_action_points(screenshot, pa_pm_region)
        if pa_pm_data:
            character_info.update(pa_pm_data)
        
        # Analyse zone stats générales
        stats_region = self.screen_regions["character_stats"]
        stats_data = self._extract_character_stats(screenshot, stats_region)
        if stats_data:
            character_info.update(stats_data)
        
        return character_info
    
    def _extract_character_stats(self, screenshot: np.ndarray, region: ScreenRegion) -> Dict[str, Any]:
        """Extrait les statistiques du personnage par OCR"""
        x, y, w, h = region.x, region.y, region.width, region.height
        roi = screenshot[y:y+h, x:x+w]
        
        # Préprocessing pour OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Amélioration contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # OCR
        # Utiliser anglais par défaut, français si disponible
        try:
            ocr_text = pytesseract.image_to_string(gray, config='--psm 6', lang='fra')
        except:
            ocr_text = pytesseract.image_to_string(gray, config='--psm 6')
        
        stats = {}
        
        # Recherche niveau
        level_match = self.ui_detector.text_patterns["level"].search(ocr_text)
        if level_match:
            stats["level"] = int(level_match.group(1))
        
        # Recherche kamas
        kamas_match = self.ui_detector.text_patterns["kamas"].search(ocr_text)
        if kamas_match:
            kamas_str = kamas_match.group(1).replace(" ", "")
            stats["kamas"] = int(kamas_str)
        
        # Recherche coordonnées
        coord_match = self.ui_detector.text_patterns["coordinates"].search(ocr_text)
        if coord_match:
            stats["map_coordinates"] = (int(coord_match.group(1)), int(coord_match.group(2)))
        
        return stats
    
    def _analyze_combat_interface(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Analyse l'interface de combat"""
        combat_data = self.ui_detector.detect_combat_interface(screenshot)
        
        if combat_data.get("in_combat", False):
            # Analyse approfondie des éléments de combat
            combat_region = self.screen_regions["combat_interface"]
            x, y, w, h = combat_region.x, combat_region.y, combat_region.width, combat_region.height
            combat_roi = screenshot[y:y+h, x:x+w]
            
            # Détection des entités en combat
            entities = self._detect_combat_entities(combat_roi)
            combat_data["entities"] = entities
            
            # Analyse de la grille de combat
            grid_info = self._analyze_combat_grid(screenshot)
            combat_data["grid"] = grid_info
        
        return combat_data
    
    def _detect_combat_entities(self, combat_roi: np.ndarray) -> List[Dict[str, Any]]:
        """Détecte les entités en combat"""
        entities = []
        
        # Détection par couleur des barres de HP
        hsv = cv2.cvtColor(combat_roi, cv2.COLOR_BGR2HSV)
        
        # Barres vertes (alliés) et rouges (ennemis)
        green_mask = cv2.inRange(hsv, *self.ui_detector.color_ranges["hp_green"])
        red_mask = cv2.inRange(hsv, *self.ui_detector.color_ranges["hp_red"])
        
        # Traitement des masques
        kernel = np.ones((3,3), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        # Détection contours
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Traitement alliés (barres vertes)
        for contour in green_contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filtrage par taille
                x, y, w, h = cv2.boundingRect(contour)
                entities.append({
                    "type": "ally",
                    "bounding_box": (x, y, x+w, y+h),
                    "hp_bar_area": area,
                    "estimated_hp": min(100, (area / 500) * 100)  # Estimation basée sur taille
                })
        
        # Traitement ennemis (barres rouges/oranges)
        for contour in red_contours:
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                entities.append({
                    "type": "enemy",
                    "bounding_box": (x, y, x+w, y+h),
                    "hp_bar_area": area,
                    "estimated_hp": min(100, (area / 500) * 100)
                })
        
        return entities
    
    def _analyze_combat_grid(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Analyse la grille de combat"""
        # Zone centrale où se trouve la grille
        center_region = self.screen_regions["center_game"]
        x, y, w, h = center_region.x, center_region.y, center_region.width, center_region.height
        grid_roi = screenshot[y:y+h, x:x+w]
        
        # Détection des cellules de la grille
        gray = cv2.cvtColor(grid_roi, cv2.COLOR_BGR2GRAY)
        
        # Détection de bordures pour cellules
        edges = cv2.Canny(gray, 30, 100)
        
        # Détection de lignes (grille isométrique)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        grid_info = {
            "grid_detected": lines is not None and len(lines) > 10,
            "lines_count": len(lines) if lines is not None else 0,
            "grid_center": (w//2, h//2),
            "estimated_cell_size": 43  # Taille typique cellule DOFUS
        }
        
        return grid_info
    
    def _analyze_map_elements(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Analyse les éléments de la carte"""
        map_data = {}
        
        # Analyse de la minimap
        minimap_region = self.screen_regions["minimap"]
        minimap_data = self._analyze_minimap(screenshot, minimap_region)
        if minimap_data:
            map_data["minimap"] = minimap_data
        
        # Détection des éléments interactifs sur la carte principale
        center_region = self.screen_regions["center_game"]
        interactive_elements = self._detect_interactive_elements(screenshot, center_region)
        if interactive_elements:
            map_data["interactive_elements"] = interactive_elements
        
        return map_data
    
    def _analyze_minimap(self, screenshot: np.ndarray, region: ScreenRegion) -> Dict[str, Any]:
        """Analyse la minimap"""
        x, y, w, h = region.x, region.y, region.width, region.height
        minimap_roi = screenshot[y:y+h, x:x+w]
        
        # Détection de la position du joueur (point central souvent)
        hsv = cv2.cvtColor(minimap_roi, cv2.COLOR_BGR2HSV)
        
        # Le joueur est souvent représenté par un point blanc/jaune
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        
        # Trouve le centre de masse
        moments = cv2.moments(white_mask)
        if moments["m00"] > 0:
            player_x = int(moments["m10"] / moments["m00"])
            player_y = int(moments["m01"] / moments["m00"])
            
            return {
                "player_position_minimap": (player_x, player_y),
                "minimap_size": (w, h)
            }
        
        return {}
    
    def _detect_interactive_elements(self, screenshot: np.ndarray, region: ScreenRegion) -> List[Dict[str, Any]]:
        """Détecte les éléments interactifs (PNJ, ressources, etc.)"""
        x, y, w, h = region.x, region.y, region.width, region.height
        roi = screenshot[y:y+h, x:x+w]
        
        elements = []
        
        # Détection par template matching si templates disponibles
        templates_dir = Path("assets/templates/interactive")
        if templates_dir.exists():
            for template_file in templates_dir.glob("*.png"):
                template = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                result = cv2.matchTemplate(gray_roi, template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= 0.6)
                
                for pt in zip(*locations[::-1]):
                    elements.append({
                        "type": template_file.stem,
                        "position": (pt[0] + region.x, pt[1] + region.y),
                        "confidence": float(result[pt[1], pt[0]])
                    })
        
        return elements
    
    def _update_performance_stats(self, analysis_time: float, analysis_result: Dict[str, Any]) -> None:
        """Met à jour les statistiques de performance"""
        # Moyenne mobile pour temps d'analyse
        alpha = 0.1
        self.performance_stats["analysis_time_avg"] = (
            alpha * analysis_time + 
            (1 - alpha) * self.performance_stats["analysis_time_avg"]
        )
        
        # Éléments détectés
        elements_count = len(analysis_result.get("ui_elements", [])) + len(analysis_result.get("resources", []))
        self.performance_stats["detections_per_frame"] = elements_count
    
    def handle_event(self, event: Any) -> bool:
        """Gestion des événements"""
        # Ce module d'analyse ne traite pas d'événements spécifiques
        return False
    
    def get_state(self) -> Dict[str, Any]:
        """Retourne l'état du module"""
        return {
            "status": self.status.value,
            "is_capturing": self.is_capturing,
            "window_found": self.dofus_window_handle is not None,
            "performance": self.performance_stats,
            "cache_size": len(self.detection_cache)
        }
    
    def _initialize_yolo_detector(self, yolo_config: Dict[str, Any]) -> None:
        """Initialise le détecteur YOLO de manière optionnelle"""
        try:
            self.logger.info("Initialisation du détecteur YOLO...")

            # Configuration YOLO
            config = YOLOConfig()
            if 'model_path' in yolo_config:
                config.model_path = yolo_config['model_path']
            if 'confidence_threshold' in yolo_config:
                config.confidence_threshold = yolo_config['confidence_threshold']
            if 'device' in yolo_config:
                config.device = yolo_config['device']

            # Initialisation
            self.yolo_detector = DofusYOLODetector(config=config)

            # Test d'initialisation
            if self.yolo_detector.initialize({}):
                self.enable_yolo = True
                self.logger.info("✅ Détecteur YOLO initialisé avec succès")

                # Configuration des zones YOLO
                if 'zones' in yolo_config:
                    self.yolo_zones = yolo_config['zones']
            else:
                self.logger.warning("❌ Échec initialisation YOLO - mode template uniquement")
                self.yolo_detector = None

        except Exception as e:
            self.logger.error(f"Erreur initialisation YOLO: {e}")
            self.yolo_detector = None
            self.enable_yolo = False

    def _analyze_dynamic_objects(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """
        Analyse les objets dynamiques avec YOLO dans les zones définies

        Args:
            screenshot: Image complète à analyser

        Returns:
            Dict contenant les objets détectés par zone
        """
        if not self.enable_yolo or not self.yolo_detector:
            return {}

        dynamic_results = {
            "zones_analyzed": [],
            "total_objects": 0,
            "objects_by_type": defaultdict(list),
            "resources": [],  # Format compatible avec l'existant
            "monsters": [],
            "npcs": [],
            "players": []
        }

        try:
            # Analyse des zones configurées pour YOLO
            for zone_name in self.yolo_zones:
                if zone_name in self.screen_regions:
                    zone_result = self._analyze_zone_with_yolo(screenshot, zone_name)
                    if zone_result:
                        dynamic_results["zones_analyzed"].append(zone_name)

                        # Fusion des résultats
                        for obj_type, objects in zone_result.get("objects_by_type", {}).items():
                            dynamic_results["objects_by_type"][obj_type].extend(objects)
                            dynamic_results["total_objects"] += len(objects)

                            # Classification pour compatibilité
                            self._classify_objects_for_compatibility(objects, obj_type, dynamic_results)

            self.logger.debug(f"YOLO détecté {dynamic_results['total_objects']} objets dans {len(dynamic_results['zones_analyzed'])} zones")

        except Exception as e:
            self.logger.error(f"Erreur analyse objets dynamiques: {e}")

        return dynamic_results

    def _analyze_zone_with_yolo(self, screenshot: np.ndarray, zone_name: str) -> Dict[str, Any]:
        """Analyse une zone spécifique avec YOLO"""
        try:
            # Extraction de la zone
            region = self.screen_regions[zone_name]
            x, y, w, h = region.x, region.y, region.width, region.height
            zone_image = screenshot[y:y+h, x:x+w]

            # Détection YOLO
            detections = self.yolo_detector.detect(zone_image)

            if not detections:
                return {}

            # Ajustement des coordonnées pour l'image complète
            adjusted_objects = defaultdict(list)

            for detection in detections:
                # Conversion des coordonnées relatives à absolues
                abs_bbox = (
                    detection.bbox[0] + x,
                    detection.bbox[1] + y,
                    detection.bbox[2] + x,
                    detection.bbox[3] + y
                )

                abs_center = (
                    detection.center[0] + x,
                    detection.center[1] + y
                )

                obj_data = {
                    "type": detection.class_name,
                    "confidence": detection.confidence,
                    "position": abs_center,
                    "bounding_box": abs_bbox,
                    "area": detection.area,
                    "method": "yolo",
                    "zone": zone_name,
                    "timestamp": detection.timestamp
                }

                adjusted_objects[detection.class_name].append(obj_data)

            return {
                "zone": zone_name,
                "objects_by_type": dict(adjusted_objects),
                "total_detections": len(detections)
            }

        except Exception as e:
            self.logger.error(f"Erreur analyse zone {zone_name}: {e}")
            return {}

    def _classify_objects_for_compatibility(self, objects: List[Dict], obj_type: str, results: Dict):
        """Classe les objets YOLO dans les catégories compatibles avec l'existant"""
        for obj in objects:
            if "resource" in obj_type:
                # Format compatible avec detect_resources
                resource_obj = {
                    "type": obj_type.replace("resource_", ""),
                    "position": obj["position"],
                    "bounding_box": obj["bounding_box"],
                    "confidence": obj["confidence"],
                    "scale": 1.0,  # YOLO est scale-invariant
                    "method": "yolo"
                }
                results["resources"].append(resource_obj)

            elif obj_type == "monster" or obj_type == "archmonster":
                results["monsters"].append(obj)

            elif obj_type == "npc":
                results["npcs"].append(obj)

            elif obj_type == "player" or obj_type == "pvp_player":
                results["players"].append(obj)

    def get_yolo_status(self) -> Dict[str, Any]:
        """Retourne le statut du système YOLO"""
        return {
            "available": YOLO_AVAILABLE,
            "enabled": self.enable_yolo,
            "detector_loaded": self.yolo_detector is not None,
            "zones_configured": self.yolo_zones,
            "model_info": self.yolo_detector.get_state() if self.yolo_detector else None
        }

    def toggle_yolo(self, enabled: bool) -> bool:
        """Active/désactive YOLO de manière dynamique"""
        if not YOLO_AVAILABLE:
            self.logger.warning("YOLO non disponible")
            return False

        if enabled and not self.yolo_detector:
            # Initialisation différée
            self._initialize_yolo_detector({"enable": True})

        self.enable_yolo = enabled and self.yolo_detector is not None
        self.logger.info(f"YOLO {'activé' if self.enable_yolo else 'désactivé'}")
        return self.enable_yolo

    def set_yolo_zones(self, zones: List[str]) -> None:
        """Configure les zones où YOLO est actif"""
        valid_zones = [z for z in zones if z in self.screen_regions]
        self.yolo_zones = valid_zones
        self.logger.info(f"Zones YOLO configurées: {valid_zones}")

    def cleanup(self) -> None:
        """Nettoie les ressources"""
        self.logger.info("Arrêt du module d'analyse d'écran")

        # Arrêt capture continue
        self.is_capturing = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)

        # NOUVEAU: Nettoyage YOLO
        if self.yolo_detector:
            self.yolo_detector.cleanup()
            self.yolo_detector = None

        # Pas besoin de fermer MSS car on utilise le context manager

        self.status = ModuleStatus.INACTIVE
        self.logger.info("Module d'analyse d'écran arrêté")
    
    def get_current_screenshot(self) -> Optional[np.ndarray]:
        """Retourne le screenshot actuel"""
        with self.screenshot_lock:
            return self.current_screenshot.copy() if self.current_screenshot is not None else None
    
    def save_screenshot(self, filepath: str, region: str = None) -> bool:
        """
        Sauvegarde un screenshot
        
        Args:
            filepath: Chemin de sauvegarde
            region: Région spécifique à sauvegarder (None = écran complet)
            
        Returns:
            bool: True si la sauvegarde réussit
        """
        try:
            screenshot = self.get_current_screenshot()
            if screenshot is None:
                return False
            
            if region and region in self.screen_regions:
                reg = self.screen_regions[region]
                screenshot = screenshot[reg.y:reg.y+reg.height, reg.x:reg.x+reg.width]
            
            cv2.imwrite(filepath, screenshot)
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde screenshot: {e}")
            return False