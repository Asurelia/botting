"""
Realtime Vision - Capture et analyse l'écran en temps réel (MVP)
"""

import time
import logging
import numpy as np
import cv2
import pyautogui
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

try:
    from .ocr_detector import OCRDetector, create_ocr_detector
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logger = logging.getLogger(__name__)


class RealtimeVision:
    """
    Vision temps réel pour Dofus
    Version MVP: Détections basiques uniquement
    """
    
    def __init__(self, window_title: str = "Dofus"):
        self.window_title = window_title
        self.last_capture = None
        self.last_capture_time = 0
        
        # Régions d'intérêt (à calibrer)
        self.roi_hp = (50, 50, 150, 20)  # HP bar region
        self.roi_pa = (50, 80, 80, 20)   # PA region
        self.roi_pm = (140, 80, 80, 20)  # PM region
        
        # OCR Detector
        self.ocr_detector = create_ocr_detector() if OCR_AVAILABLE else None
        
        # Cache
        self.cache_duration = 0.5  # secondes
        
        # Stats
        self.stats = {
            'captures': 0,
            'detections': 0,
            'cache_hits': 0,
            'ocr_enabled': OCR_AVAILABLE
        }
        
        logger.info(f"RealtimeVision initialisé (OCR: {OCR_AVAILABLE})")
    
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Capture l'écran
        
        Args:
            region: (x, y, width, height) ou None pour tout l'écran
        
        Returns:
            Image en BGR (OpenCV format)
        """
        try:
            # Capture avec PyAutoGUI
            if region:
                screenshot = pyautogui.screenshot(region=region)
            else:
                screenshot = pyautogui.screenshot()
            
            # Convertir PIL → OpenCV (BGR)
            img_rgb = np.array(screenshot)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            self.stats['captures'] += 1
            return img_bgr
            
        except Exception as e:
            logger.error(f"Erreur capture écran: {e}")
            return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def detect_hp_bar(self, img: np.ndarray) -> Tuple[int, int, float]:
        """
        Détecte HP bar avec OCR si disponible
        
        Returns:
            (hp_current, hp_max, hp_percent)
        """
        try:
            if self.ocr_detector:
                # Utiliser OCR amélioré
                return self.ocr_detector.detect_hp_ocr(img)
            else:
                # Fallback: détection couleur simple
                return self.ocr_detector.detect_hp_simple(img) if self.ocr_detector else (100, 100, 100.0)
            
        except Exception as e:
            logger.error(f"Erreur détection HP: {e}")
            return 100, 100, 100.0
    
    def detect_pa_pm(self, img: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Détecte PA/PM (VERSION SIMPLIFIÉE)
        
        Returns:
            (pa_current, pa_max, pm_current, pm_max)
        """
        # MVP: Valeurs mock
        # TODO: OCR réel
        return 6, 6, 3, 3
    
    def detect_combat_state(self, img: np.ndarray) -> bool:
        """
        Détecte si en combat (DÉSACTIVÉ pour éviter faux positifs)
        
        Returns:
            False (toujours) - La détection de combat sera ajoutée plus tard
        """
        # TODO: Implémenter une vraie détection de la timeline de combat
        # Pour l'instant, désactivé car trop de faux positifs
        return False
    
    def detect_position(self, img: np.ndarray) -> Tuple[int, int]:
        """
        Détecte position sur la map (VERSION SIMPLIFIÉE)
        
        Returns:
            (x, y) coordonnées map
        """
        # MVP: Valeurs mock
        # TODO: OCR des coordonnées affichées en bas à droite
        return 0, 0
    
    def detect_entities(self, img: np.ndarray) -> list:
        """
        Détecte entités visibles (mobs, joueurs, ressources)
        
        Returns:
            Liste d'entités détectées
        """
        # MVP: Liste vide
        # TODO: YOLO ou template matching
        return []
    
    def extract_game_state(self) -> Dict[str, Any]:
        """
        Extrait l'état complet du jeu depuis la vision
        
        Returns:
            Dictionnaire avec toutes les infos détectées
        """
        # Check cache
        now = time.time()
        if self.last_capture is not None and (now - self.last_capture_time) < self.cache_duration:
            self.stats['cache_hits'] += 1
            return self.last_capture
        
        # Nouvelle capture
        img = self.capture_screen()
        
        # Détections
        hp_current, hp_max, hp_percent = self.detect_hp_bar(img)
        pa_current, pa_max, pm_current, pm_max = self.detect_pa_pm(img)
        in_combat = self.detect_combat_state(img)
        map_x, map_y = self.detect_position(img)
        entities = self.detect_entities(img)
        
        # Construire état
        game_state = {
            'timestamp': now,
            'character': {
                'hp': hp_current,
                'max_hp': hp_max,
                'hp_percent': hp_percent,
                'pa': pa_current,
                'max_pa': pa_max,
                'pm': pm_current,
                'max_pm': pm_max
            },
            'combat': {
                'in_combat': in_combat,
                'my_turn': False  # TODO: détecter
            },
            'position': {
                'map_x': map_x,
                'map_y': map_y
            },
            'entities': entities,
            'ui': {
                'window_active': True  # Assume active si on capture
            }
        }
        
        # Cache
        self.last_capture = game_state
        self.last_capture_time = now
        
        self.stats['detections'] += 1
        
        return game_state
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne statistiques"""
        return self.stats.copy()


def create_realtime_vision(window_title: str = "Dofus") -> RealtimeVision:
    """Factory function"""
    return RealtimeVision(window_title=window_title)
