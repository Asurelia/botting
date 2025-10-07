"""
OCR Detector - Détection texte simple pour HP/PA/PM
Utilise pytesseract pour OCR basique
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional
import re

# Try to import tesseract
try:
    import pytesseract
    # Configurer le chemin Tesseract pour Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("pytesseract non disponible - OCR désactivé")

logger = logging.getLogger(__name__)


class OCRDetector:
    """Détecteur OCR simple pour Dofus"""
    
    def __init__(self):
        self.tesseract_available = TESSERACT_AVAILABLE
        
        # Régions d'intérêt (coordonnées relatives à calibrer)
        # Ces valeurs sont à ajuster selon votre résolution
        self.roi_hp = None
        self.roi_pa = None
        self.roi_pm = None
        
        # Flag pour ne logger qu'une fois
        self._tesseract_warning_logged = False
        
        logger.info(f"OCRDetector initialisé (Tesseract: {self.tesseract_available})")
    
    def set_regions(self, hp_roi, pa_roi, pm_roi):
        """Configure les régions d'intérêt"""
        self.roi_hp = hp_roi
        self.roi_pa = pa_roi
        self.roi_pm = pm_roi
    
    def preprocess_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """
        Prétraite l'image pour améliorer l'OCR
        - Conversion grayscale
        - Seuillage
        - Agrandissement
        """
        # Grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Seuillage pour isoler le texte
        # Les valeurs de Dofus sont souvent en blanc sur fond sombre
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Agrandir x2 pour améliorer OCR
        scale = 2
        enlarged = cv2.resize(thresh, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        return enlarged
    
    def extract_numbers(self, text: str) -> Optional[Tuple[int, int]]:
        """
        Extrait les nombres du format "123/456"
        
        Returns:
            (current, max) ou None
        """
        # Chercher pattern "nombre/nombre"
        pattern = r'(\d+)\s*/\s*(\d+)'
        match = re.search(pattern, text)
        
        if match:
            current = int(match.group(1))
            max_val = int(match.group(2))
            return (current, max_val)
        
        # Chercher juste un nombre
        pattern_single = r'(\d+)'
        match_single = re.search(pattern_single, text)
        
        if match_single:
            val = int(match_single.group(1))
            return (val, val)
        
        return None
    
    def detect_hp_simple(self, img: np.ndarray) -> Tuple[int, int, float]:
        """
        Détection HP simple par analyse couleur
        (Fallback si pas de tesseract)
        
        Returns:
            (hp_current, hp_max, hp_percent)
        """
        # Zone HP (à gauche, en haut)
        h, w = img.shape[:2]
        hp_region = img[int(h*0.02):int(h*0.06), int(w*0.02):int(w*0.15)]
        
        # Convertir en HSV
        hsv = cv2.cvtColor(hp_region, cv2.COLOR_BGR2HSV)
        
        # Masque pour vert (HP)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Masque pour rouge (HP bas)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Calculer pourcentages
        total_pixels = hp_region.shape[0] * hp_region.shape[1]
        green_pixels = cv2.countNonZero(mask_green)
        red_pixels = cv2.countNonZero(mask_red)
        
        # HP percent basé sur vert + rouge
        hp_bar_pixels = green_pixels + red_pixels
        hp_percent = (green_pixels / hp_bar_pixels * 100) if hp_bar_pixels > 0 else 100.0
        
        # Estimer valeurs (à remplacer par OCR)
        hp_max = 100  # Valeur par défaut
        hp_current = int(hp_max * hp_percent / 100)
        
        return hp_current, hp_max, hp_percent
    
    def detect_hp_ocr(self, img: np.ndarray) -> Tuple[int, int, float]:
        """
        Détection HP avec OCR
        
        Returns:
            (hp_current, hp_max, hp_percent)
        """
        if not self.tesseract_available:
            if not self._tesseract_warning_logged:
                logger.warning("Tesseract non disponible - Utilisation détection couleur")
                self._tesseract_warning_logged = True
            return self.detect_hp_simple(img)
        
        try:
            # Zone HP
            h, w = img.shape[:2]
            hp_region = img[int(h*0.02):int(h*0.06), int(w*0.02):int(w*0.15)]
            
            # Prétraiter
            processed = self.preprocess_for_ocr(hp_region)
            
            # OCR
            text = pytesseract.image_to_string(
                processed,
                config='--psm 7 -c tessedit_char_whitelist=0123456789/'
            )
            
            # Extraire nombres
            numbers = self.extract_numbers(text)
            
            if numbers:
                hp_current, hp_max = numbers
                hp_percent = (hp_current / hp_max * 100) if hp_max > 0 else 100.0
                return hp_current, hp_max, hp_percent
            else:
                # Fallback
                return self.detect_hp_simple(img)
                
        except Exception as e:
            if not self._tesseract_warning_logged:
                logger.error(f"Erreur OCR HP: {e}")
                self._tesseract_warning_logged = True
            return self.detect_hp_simple(img)
    
    def detect_pa_pm_simple(self, img: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Détection PA/PM simple (valeurs fixes)
        
        Returns:
            (pa_current, pa_max, pm_current, pm_max)
        """
        # Valeurs par défaut pour un Cra
        return 6, 6, 3, 3
    
    def detect_combat_timeline(self, img: np.ndarray) -> bool:
        """
        Détecte si la timeline de combat est visible
        
        Returns:
            True si en combat
        """
        # Zone bas de l'écran
        h, w = img.shape[:2]
        timeline_region = img[int(h*0.85):h, int(w*0.1):int(w*0.9)]
        
        # Convertir HSV
        hsv = cv2.cvtColor(timeline_region, cv2.COLOR_BGR2HSV)
        
        # Détecter couleurs timeline (marron/orange/vert)
        # Marron/Orange
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([30, 255, 200])
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Vert (tour actif)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Combiner
        mask_combined = cv2.bitwise_or(mask_brown, mask_green)
        
        # Ratio de pixels timeline
        total_pixels = timeline_region.shape[0] * timeline_region.shape[1]
        timeline_pixels = cv2.countNonZero(mask_combined)
        ratio = timeline_pixels / total_pixels
        
        # Si > 15% de la zone = timeline présente
        return ratio > 0.15


def create_ocr_detector() -> OCRDetector:
    """Factory function"""
    return OCRDetector()
