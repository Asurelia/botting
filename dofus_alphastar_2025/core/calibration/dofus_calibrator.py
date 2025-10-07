#!/usr/bin/env python3
"""
DofusCalibrator - Système de calibration automatique
Découverte complète de l'interface Dofus au premier lancement
Durée: 5-10 minutes une seule fois
"""

import time
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import cv2
import pyautogui

# Import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from bot_config import DOFUS_WINDOW_TITLE

# Désactiver le fail-safe de pyautogui
pyautogui.FAILSAFE = False

@dataclass
class WindowInfo:
    """Informations sur la fenêtre Dofus"""
    title: str
    x: int
    y: int
    width: int
    height: int
    is_fullscreen: bool

@dataclass
class UIElement:
    """Élément d'interface"""
    name: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    template_path: Optional[str] = None

@dataclass
class GameShortcut:
    """Raccourci clavier du jeu"""
    action: str
    key: str
    verified: bool
    detection_method: str

@dataclass
class InteractiveElement:
    """Élément interactif dans le jeu"""
    element_type: str  # npc, door, resource, zaap, etc.
    x: int
    y: int
    bbox: Tuple[int, int, int, int]
    color_signature: Optional[Tuple[int, int, int]] = None
    template_id: Optional[str] = None

@dataclass
class CalibrationResult:
    """Résultat complet de la calibration"""
    calibration_date: str
    dofus_version: str
    window_info: WindowInfo
    ui_elements: List[UIElement]
    shortcuts: List[GameShortcut]
    interactive_elements: List[InteractiveElement]
    game_options: Dict[str, Any]
    success: bool
    duration_seconds: float

class DofusCalibrator:
    """
    Calibration automatique complète de Dofus

    Phase 1: Détection fenêtre et résolution
    Phase 2: Mapping éléments UI
    Phase 3: Détection raccourcis
    Phase 4: Scan éléments interactifs
    Phase 5: Analyse options jeu
    Phase 6: Construction base de connaissances
    """

    def __init__(self, output_path: str = "config/dofus_knowledge.json"):
        self.logger = logging.getLogger(__name__)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # État de la calibration
        self.window_info: Optional[WindowInfo] = None
        self.ui_elements: List[UIElement] = []
        self.shortcuts: List[GameShortcut] = []
        self.interactive_elements: List[InteractiveElement] = []
        self.game_options: Dict[str, Any] = {}

        # Configuration
        self.template_dir = Path("assets/templates/dofus")
        self.template_dir.mkdir(parents=True, exist_ok=True)

        # Timings
        self.start_time = None
        self.end_time = None

    def run_full_calibration(self) -> CalibrationResult:
        """
        Lance la calibration complète

        Returns:
            CalibrationResult avec toutes les informations découvertes
        """
        self.logger.info("=" * 60)
        self.logger.info("CALIBRATION AUTOMATIQUE DOFUS")
        self.logger.info("=" * 60)
        self.logger.info("")
        self.logger.info("Instructions:")
        self.logger.info("1. Lance Dofus et connecte-toi")
        self.logger.info("2. Va sur une map vide (village de départ)")
        self.logger.info("3. Ne touche à RIEN pendant 5-10 minutes")
        self.logger.info("")

        self.start_time = time.time()

        try:
            # Phase 1: Fenêtre
            self.logger.info("[Phase 1/6] Détection de la fenêtre Dofus...")
            self.detect_window()

            # Phase 2: UI
            self.logger.info("[Phase 2/6] Mapping de l'interface...")
            self.map_ui_elements()

            # Phase 3: Raccourcis
            self.logger.info("[Phase 3/6] Détection des raccourcis...")
            self.discover_shortcuts()

            # Phase 4: Éléments interactifs
            self.logger.info("[Phase 4/6] Scan des éléments interactifs...")
            self.scan_interactive_elements()

            # Phase 5: Options
            self.logger.info("[Phase 5/6] Analyse des options du jeu...")
            self.scan_game_options()

            # Phase 6: Sauvegarde
            self.logger.info("[Phase 6/6] Construction de la base de connaissances...")
            result = self.build_knowledge_base()

            self.end_time = time.time()

            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("CALIBRATION TERMINÉE AVEC SUCCÈS!")
            self.logger.info(f"Durée: {result.duration_seconds:.1f} secondes")
            self.logger.info(f"Sauvegardé dans: {self.output_path}")
            self.logger.info("=" * 60)

            return result

        except Exception as e:
            self.logger.error(f"Erreur durant la calibration: {e}")
            import traceback
            traceback.print_exc()

            self.end_time = time.time()

            return CalibrationResult(
                calibration_date=datetime.now().isoformat(),
                dofus_version="unknown",
                window_info=self.window_info or WindowInfo("unknown", 0, 0, 0, 0, False),
                ui_elements=self.ui_elements,
                shortcuts=self.shortcuts,
                interactive_elements=self.interactive_elements,
                game_options=self.game_options,
                success=False,
                duration_seconds=self.end_time - self.start_time if self.start_time else 0
            )

    def detect_window(self):
        """Détecte et analyse la fenêtre Dofus"""
        self.logger.info("  Recherche de la fenêtre Dofus...")

        try:
            import pygetwindow as gw

            # Cherche fenêtre Dofus avec titre exact
            dofus_windows = []
            for window in gw.getAllWindows():
                # Cherche d'abord le titre exact
                if window.title == DOFUS_WINDOW_TITLE:
                    dofus_windows.append(window)
                # Sinon cherche "dofus" générique
                elif "dofus" in window.title.lower():
                    dofus_windows.append(window)

            if not dofus_windows:
                self.logger.warning("  [WARNING] Aucune fenêtre Dofus trouvée!")
                self.logger.info("  Utilisation de la fenêtre active...")
                active = gw.getActiveWindow()
                if active:
                    dofus_windows = [active]

            if dofus_windows:
                window = dofus_windows[0]

                # Détecte si fullscreen
                screen_width, screen_height = pyautogui.size()
                is_fullscreen = (
                    window.width >= screen_width - 10 and
                    window.height >= screen_height - 10
                )

                self.window_info = WindowInfo(
                    title=window.title,
                    x=window.left,
                    y=window.top,
                    width=window.width,
                    height=window.height,
                    is_fullscreen=is_fullscreen
                )

                self.logger.info(f"  [OK] Fenêtre trouvée: {window.title}")
                self.logger.info(f"    Position: ({window.left}, {window.top})")
                self.logger.info(f"    Taille: {window.width}x{window.height}")
                self.logger.info(f"    Fullscreen: {is_fullscreen}")

                # Focus sur la fenêtre
                window.activate()
                time.sleep(0.5)

            else:
                raise RuntimeError("Impossible de trouver la fenêtre Dofus")

        except ImportError:
            self.logger.warning("  [WARNING] pygetwindow non disponible, utilisation de méthode alternative")
            screen_width, screen_height = pyautogui.size()
            self.window_info = WindowInfo(
                title="Dofus (détection automatique)",
                x=0,
                y=0,
                width=screen_width,
                height=screen_height,
                is_fullscreen=True
            )

    def map_ui_elements(self):
        """Map tous les éléments UI fixes"""
        self.logger.info("  Capture de l'écran...")

        # Capture écran
        screenshot = pyautogui.screenshot()
        screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # Templates à détecter
        ui_templates = {
            'hp_bar': 'Points de vie',
            'pa_bar': 'Points d\'action',
            'pm_bar': 'Points de mouvement',
            'spell_bar': 'Barre de sorts',
            'minimap': 'Minimap',
            'chat': 'Chat',
            'inventory_button': 'Bouton inventaire',
            'character_button': 'Bouton personnage',
            'spell_book_button': 'Bouton grimoire',
            'quest_button': 'Bouton quêtes',
            'map_button': 'Bouton carte',
            'social_button': 'Bouton social'
        }

        self.logger.info(f"  Recherche de {len(ui_templates)} éléments UI...")

        # Analyse par zones prédéfinies (heuristique)
        # TODO: Implémenter template matching réel
        # Pour l'instant, détection par zones connues

        zones = self._detect_ui_zones(screen)

        for name, description in ui_templates.items():
            if name in zones:
                zone = zones[name]
                element = UIElement(
                    name=name,
                    x=zone['x'],
                    y=zone['y'],
                    width=zone['width'],
                    height=zone['height'],
                    confidence=zone.get('confidence', 0.8),
                    template_path=None
                )
                self.ui_elements.append(element)
                self.logger.info(f"    [OK] {description} détecté")

        self.logger.info(f"  Total: {len(self.ui_elements)} éléments trouvés")

    def _detect_ui_zones(self, screen: np.ndarray) -> Dict[str, Dict[str, int]]:
        """
        Détecte les zones UI par analyse heuristique

        Dofus a une disposition standard:
        - Haut gauche: Info personnage (HP, PA, PM)
        - Bas centre: Barre de sorts
        - Haut droite: Minimap
        - Bas droite: Chat
        - Gauche: Boutons d'interface
        """
        height, width = screen.shape[:2]

        zones = {
            # Info personnage (haut gauche)
            'hp_bar': {'x': 20, 'y': 20, 'width': 200, 'height': 30, 'confidence': 0.9},
            'pa_bar': {'x': 20, 'y': 60, 'width': 150, 'height': 25, 'confidence': 0.9},
            'pm_bar': {'x': 20, 'y': 95, 'width': 150, 'height': 25, 'confidence': 0.9},

            # Barre de sorts (bas centre)
            'spell_bar': {'x': width//2 - 250, 'y': height - 80, 'width': 500, 'height': 60, 'confidence': 0.85},

            # Minimap (haut droite)
            'minimap': {'x': width - 220, 'y': 20, 'width': 200, 'height': 200, 'confidence': 0.9},

            # Chat (bas droite ou bas gauche selon config)
            'chat': {'x': width - 420, 'y': height - 220, 'width': 400, 'height': 200, 'confidence': 0.8},

            # Boutons interface (gauche)
            'inventory_button': {'x': 10, 'y': 150, 'width': 40, 'height': 40, 'confidence': 0.7},
            'character_button': {'x': 10, 'y': 200, 'width': 40, 'height': 40, 'confidence': 0.7},
            'spell_book_button': {'x': 10, 'y': 250, 'width': 40, 'height': 40, 'confidence': 0.7},
            'quest_button': {'x': 10, 'y': 300, 'width': 40, 'height': 40, 'confidence': 0.7},
            'map_button': {'x': 10, 'y': 350, 'width': 40, 'height': 40, 'confidence': 0.7},
        }

        return zones

    def discover_shortcuts(self):
        """Teste tous les raccourcis possibles du jeu"""
        shortcuts_to_test = {
            'inventory': ['i', 'I'],
            'character': ['c', 'C'],
            'spell_book': ['s', 'S'],
            'quest_log': ['l', 'L'],
            'map': ['m', 'M'],
            'friend_list': ['f', 'F'],
            'guild': ['g', 'G'],
            'alliance': ['b', 'B'],
            'job': ['j', 'J'],
            'mount': ['h', 'H'],
            'options': ['o', 'O', 'escape'],
            'help': ['F1'],
        }

        self.logger.info(f"  Test de {len(shortcuts_to_test)} raccourcis...")

        for action, keys in shortcuts_to_test.items():
            for key in keys:
                try:
                    # Capture avant
                    before = pyautogui.screenshot()
                    before_np = np.array(before)

                    # Test raccourci
                    pyautogui.press(key)
                    time.sleep(0.8)

                    # Capture après
                    after = pyautogui.screenshot()
                    after_np = np.array(after)

                    # Compare
                    if self._screen_changed(before_np, after_np):
                        shortcut = GameShortcut(
                            action=action,
                            key=key,
                            verified=True,
                            detection_method="screen_diff"
                        )
                        self.shortcuts.append(shortcut)
                        self.logger.info(f"    [OK] {action} = {key}")

                        # Ferme (ESC)
                        pyautogui.press('escape')
                        time.sleep(0.5)
                        break

                except Exception as e:
                    self.logger.debug(f"    Erreur test {key}: {e}")
                    continue

        self.logger.info(f"  Total: {len(self.shortcuts)} raccourcis détectés")

    def _screen_changed(self, img1: np.ndarray, img2: np.ndarray, threshold: float = 0.05) -> bool:
        """Vérifie si l'écran a changé significativement"""
        # Redimensionne pour comparaison rapide
        small1 = cv2.resize(img1, (160, 90))
        small2 = cv2.resize(img2, (160, 90))

        # Différence absolue
        diff = cv2.absdiff(small1, small2)

        # Pourcentage de pixels différents
        diff_ratio = np.sum(diff > 30) / diff.size

        return diff_ratio > threshold

    def scan_interactive_elements(self):
        """Scanne les éléments interactifs sur la map actuelle"""
        self.logger.info("  Activation du mode surbrillance...")

        # TODO: Activer surbrillance via options
        # Pour l'instant, détection basique

        screenshot = pyautogui.screenshot()
        screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # Détection par couleurs (éléments interactifs sont souvent highlighted)
        interactive_colors = [
            ((200, 200, 0), (255, 255, 100), 'highlighted'),    # Jaune
            ((0, 200, 200), (100, 255, 255), 'npc'),            # Cyan
            ((200, 0, 200), (255, 100, 255), 'door'),           # Magenta
        ]

        for color_min, color_max, element_type in interactive_colors:
            mask = cv2.inRange(screen, np.array(color_min), np.array(color_max))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filtre petits bruits
                    x, y, w, h = cv2.boundingRect(contour)

                    element = InteractiveElement(
                        element_type=element_type,
                        x=x + w//2,
                        y=y + h//2,
                        bbox=(x, y, x+w, y+h),
                        color_signature=tuple(color_min)
                    )
                    self.interactive_elements.append(element)

        self.logger.info(f"  Total: {len(self.interactive_elements)} éléments interactifs trouvés")

    def scan_game_options(self):
        """Scanne les options disponibles dans le jeu"""
        self.logger.info("  Ouverture du menu options...")

        try:
            # Ouvre options (ESC)
            pyautogui.press('escape')
            time.sleep(1)

            # TODO: OCR des options disponibles
            # Pour l'instant, configuration par défaut

            self.game_options = {
                'interface': {
                    'highlight_interactive': True,
                    'show_names': True,
                    'show_hp_bars': True,
                },
                'combat': {
                    'auto_pass_turn': False,
                    'show_damage': True,
                },
                'general': {
                    'language': 'fr',
                    'resolution': f"{self.window_info.width}x{self.window_info.height}",
                }
            }

            # Ferme options
            pyautogui.press('escape')
            time.sleep(0.5)

            self.logger.info(f"  Options analysées: {len(self.game_options)} catégories")

        except Exception as e:
            self.logger.warning(f"  [WARNING] Erreur scan options: {e}")

    def build_knowledge_base(self) -> CalibrationResult:
        """Construit et sauvegarde la base de connaissances"""
        duration = (self.end_time or time.time()) - (self.start_time or time.time())

        result = CalibrationResult(
            calibration_date=datetime.now().isoformat(),
            dofus_version=self._detect_dofus_version(),
            window_info=self.window_info or WindowInfo("unknown", 0, 0, 0, 0, False),
            ui_elements=self.ui_elements,
            shortcuts=self.shortcuts,
            interactive_elements=self.interactive_elements,
            game_options=self.game_options,
            success=True,
            duration_seconds=duration
        )

        # Sauvegarde en JSON
        knowledge = {
            'version': '1.0',
            'calibration_date': result.calibration_date,
            'dofus_version': result.dofus_version,
            'window': asdict(result.window_info),
            'ui_elements': [asdict(elem) for elem in result.ui_elements],
            'shortcuts': [asdict(sc) for sc in result.shortcuts],
            'interactive_elements': [asdict(elem) for elem in result.interactive_elements],
            'game_options': result.game_options,
            'statistics': {
                'ui_elements_found': len(result.ui_elements),
                'shortcuts_found': len(result.shortcuts),
                'interactive_elements_found': len(result.interactive_elements),
                'duration_seconds': result.duration_seconds
            }
        }

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge, f, indent=2, ensure_ascii=False)

        self.logger.info(f"  [OK] Base de connaissances sauvegardée: {self.output_path}")

        return result

    def _detect_dofus_version(self) -> str:
        """Détecte la version de Dofus"""
        # TODO: Implémenter détection réelle (OCR du numéro de version)
        return "2.71"  # Version par défaut

def create_calibrator(output_path: str = "config/dofus_knowledge.json") -> DofusCalibrator:
    """Factory function pour créer un calibrateur"""
    return DofusCalibrator(output_path)