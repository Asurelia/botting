"""
DOFUS Unity Interface Reader - Vision Engine Spécialisé
Module de reconnaissance spécialisé pour l'interface DOFUS Unity
Approche 100% vision computer - Aucune injection mémoire/packet
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
import threading
from pathlib import Path

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameState(Enum):
    """États possibles du jeu DOFUS Unity"""
    UNKNOWN = "unknown"
    MAIN_MENU = "main_menu"
    CHARACTER_SELECTION = "character_selection"
    IN_GAME = "in_game"
    IN_COMBAT = "in_combat"
    IN_DIALOGUE = "in_dialogue"
    INVENTORY_OPEN = "inventory_open"
    MARKET_OPEN = "market_open"
    MAP_OPEN = "map_open"
    CRAFTING_OPEN = "crafting_open"
    LOADING = "loading"

@dataclass
class UIElement:
    """Élément d'interface détecté"""
    name: str
    region: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    state: str = "visible"
    text: Optional[str] = None
    clickable: bool = True

@dataclass
class CharacterStats:
    """Stats du personnage lues via OCR"""
    health_current: int = 0
    health_max: int = 0
    action_points: int = 0
    movement_points: int = 0
    level: int = 0
    experience: int = 0
    kamas: int = 0
    position_x: int = 0
    position_y: int = 0

class DofusUnityInterfaceReader:
    """
    Lecteur d'interface DOFUS Unity spécialisé
    Reconnaissance tous éléments UI via vision computer
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.current_state = GameState.UNKNOWN
        self.ui_elements: Dict[str, UIElement] = {}
        self.character_stats = CharacterStats()

        # Templates UI pour reconnaissance
        self.ui_templates = {}
        self.load_ui_templates()

        # OCR Engine pour lecture texte
        self.setup_ocr()

        # Thread de capture continue
        self.capture_thread = None
        self.running = False

        logger.info("DofusUnityInterfaceReader initialisé")

    def _get_default_config(self) -> Dict:
        """Configuration par défaut"""
        return {
            "capture_fps": 10,
            "template_threshold": 0.8,
            "ocr_engine": "tesseract",
            "ui_detection_region": (0, 0, 1920, 1080),
            "stats_region": (10, 10, 300, 150),
            "chat_region": (10, 600, 600, 200)
        }

    def setup_ocr(self):
        """Configuration OCR pour lecture texte interface"""
        try:
            import pytesseract
            import easyocr

            # Configuration Tesseract
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

            # EasyOCR pour backup
            self.easy_reader = easyocr.Reader(['fr', 'en'])

            logger.info("OCR configuré - Tesseract + EasyOCR")

        except ImportError as e:
            logger.warning(f"OCR partiellement disponible: {e}")

    def load_ui_templates(self):
        """Charge les templates UI pour reconnaissance"""
        templates_dir = Path("data/ui_templates")
        templates_dir.mkdir(parents=True, exist_ok=True)

        # Templates essentiels DOFUS Unity
        template_names = [
            "health_bar", "action_points", "movement_points",
            "inventory_icon", "spells_bar", "chat_window",
            "combat_timeline", "map_icon", "menu_button",
            "level_indicator", "kamas_display"
        ]

        for template_name in template_names:
            template_path = templates_dir / f"{template_name}.png"
            if template_path.exists():
                template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                self.ui_templates[template_name] = template
                logger.debug(f"Template chargé: {template_name}")

        logger.info(f"Templates UI chargés: {len(self.ui_templates)}")

    def detect_game_state(self, screenshot: np.ndarray) -> GameState:
        """Détecte l'état actuel du jeu via reconnaissance UI"""

        # Conversion grayscale pour template matching
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Détection états spécifiques
        if self._detect_ui_element(gray, "combat_timeline"):
            return GameState.IN_COMBAT
        elif self._detect_ui_element(gray, "inventory_icon"):
            return GameState.IN_GAME
        elif self._detect_ui_element(gray, "character_selection"):
            return GameState.CHARACTER_SELECTION
        elif self._detect_loading_screen(gray):
            return GameState.LOADING

        # État par défaut si éléments de jeu détectés
        if (self._detect_ui_element(gray, "health_bar") and
            self._detect_ui_element(gray, "spells_bar")):
            return GameState.IN_GAME

        return GameState.UNKNOWN

    def _detect_ui_element(self, gray_image: np.ndarray, element_name: str) -> bool:
        """Détecte un élément UI spécifique"""
        if element_name not in self.ui_templates:
            return False

        template = self.ui_templates[element_name]
        result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        threshold = self.config["template_threshold"]
        detected = max_val >= threshold

        if detected:
            h, w = template.shape
            region = (max_loc[0], max_loc[1], w, h)
            confidence = float(max_val)

            self.ui_elements[element_name] = UIElement(
                name=element_name,
                region=region,
                confidence=confidence
            )

        return detected

    def _detect_loading_screen(self, gray_image: np.ndarray) -> bool:
        """Détecte écran de chargement"""
        # Recherche indicateurs de chargement typiques
        height, width = gray_image.shape
        center_region = gray_image[height//3:2*height//3, width//3:2*width//3]

        # Détection barre de progression ou spinner
        edges = cv2.Canny(center_region, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Heuristiques pour loading screen
        if len(contours) < 5:  # Écran relativement vide
            return True

        return False

    def read_character_stats(self, screenshot: np.ndarray) -> CharacterStats:
        """Lit les stats du personnage via OCR"""
        stats_region = self.config["stats_region"]
        x, y, w, h = stats_region

        # Extraction région des stats
        stats_area = screenshot[y:y+h, x:x+w]

        # OCR pour lire les valeurs
        try:
            import pytesseract

            # Configuration OCR pour chiffres
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789/'
            text = pytesseract.image_to_string(stats_area, config=custom_config)

            # Parsing des stats (format typique DOFUS)
            stats = self._parse_stats_text(text)
            return stats

        except Exception as e:
            logger.warning(f"Erreur lecture stats: {e}")
            return self.character_stats

    def _parse_stats_text(self, text: str) -> CharacterStats:
        """Parse le texte OCR pour extraire les stats"""
        stats = CharacterStats()
        lines = text.strip().split('\n')

        for line in lines:
            line = line.strip()

            # Format PV: "142/186"
            if '/' in line and any(c.isdigit() for c in line):
                try:
                    current, max_val = line.split('/')
                    current = int(''.join(filter(str.isdigit, current)))
                    max_val = int(''.join(filter(str.isdigit, max_val)))

                    # Déterminer le type de stat par contexte
                    if max_val > 50:  # Probablement PV
                        stats.health_current = current
                        stats.health_max = max_val
                except:
                    continue

            # Format simple: nombre seul
            elif line.isdigit():
                value = int(line)
                # Logique d'attribution selon position/taille
                if value < 20:  # PA/PM
                    if not stats.action_points:
                        stats.action_points = value
                    elif not stats.movement_points:
                        stats.movement_points = value
                elif value > 1000:  # Kamas
                    stats.kamas = value

        return stats

    def detect_ui_elements(self, screenshot: np.ndarray) -> Dict[str, UIElement]:
        """Détecte tous les éléments UI visibles"""
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        detected_elements = {}

        # Test de tous les templates
        for element_name in self.ui_templates:
            if self._detect_ui_element(gray, element_name):
                detected_elements[element_name] = self.ui_elements[element_name]

        return detected_elements

    def start_continuous_capture(self):
        """Démarre la capture continue d'interface"""
        if self.running:
            return

        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        logger.info("Capture continue démarrée")

    def stop_continuous_capture(self):
        """Arrête la capture continue"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)

        logger.info("Capture continue arrêtée")

    def _capture_loop(self):
        """Boucle de capture continue"""
        from .screenshot_capture import capture_dofus_window

        fps_delay = 1.0 / self.config["capture_fps"]

        while self.running:
            try:
                screenshot = capture_dofus_window()
                if screenshot is not None:
                    # Analyse de l'interface
                    self.current_state = self.detect_game_state(screenshot)
                    self.character_stats = self.read_character_stats(screenshot)
                    self.ui_elements = self.detect_ui_elements(screenshot)

                time.sleep(fps_delay)

            except Exception as e:
                logger.error(f"Erreur capture loop: {e}")
                time.sleep(1.0)

    def get_current_game_state(self) -> GameState:
        """Retourne l'état actuel du jeu"""
        return self.current_state

    def get_character_stats(self) -> CharacterStats:
        """Retourne les stats actuelles du personnage"""
        return self.character_stats

    def get_ui_elements(self) -> Dict[str, UIElement]:
        """Retourne les éléments UI détectés"""
        return self.ui_elements.copy()

    def is_element_visible(self, element_name: str) -> bool:
        """Vérifie si un élément UI est visible"""
        return element_name in self.ui_elements

    def get_element_region(self, element_name: str) -> Optional[Tuple[int, int, int, int]]:
        """Retourne la région d'un élément UI"""
        if element_name in self.ui_elements:
            return self.ui_elements[element_name].region
        return None

def create_dofus_interface_reader(config: Optional[Dict] = None) -> DofusUnityInterfaceReader:
    """Factory pour créer le lecteur d'interface DOFUS"""
    return DofusUnityInterfaceReader(config)

# Test du module
if __name__ == "__main__":
    reader = create_dofus_interface_reader()
    reader.start_continuous_capture()

    try:
        while True:
            state = reader.get_current_game_state()
            stats = reader.get_character_stats()
            print(f"État: {state.value}, PV: {stats.health_current}/{stats.health_max}")
            time.sleep(2)
    except KeyboardInterrupt:
        reader.stop_continuous_capture()
        print("Arrêt du test")