"""
Détection de Combat SPÉCIFIQUE à DOFUS
Contrairement aux autres MMO, le combat n'est PAS automatique en voyant des monstres
"""

import cv2
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CombatIndicators:
    """Indicateurs de combat détectés"""
    has_combat_grid: bool = False
    has_timeline: bool = False
    has_large_papm: bool = False
    has_combat_background: bool = False
    confidence: float = 0.0

    def is_in_combat(self) -> bool:
        """Au moins 2 indicateurs = en combat"""
        indicators = sum([
            self.has_combat_grid,
            self.has_timeline,
            self.has_large_papm,
            self.has_combat_background
        ])
        return indicators >= 2


class DofusCombatDetector:
    """
    Détecteur de combat RÉALISTE pour DOFUS

    IMPORTANT:
    - Voir des monstres ≠ être en combat
    - Combat = grille visible + timeline + PA/PM gros
    - Il faut CLIQUER sur un groupe pour entrer en combat
    """

    def __init__(self):
        # Régions d'intérêt pour la détection
        self.roi_timeline = (0.2, 0.0, 0.8, 0.15)  # Haut centre (timeline)
        self.roi_papm = (0.85, 0.85, 1.0, 1.0)     # Bas droite (PA/PM)
        self.roi_grid = (0.2, 0.2, 0.8, 0.8)       # Centre (grille combat)

        # Couleurs caractéristiques DOFUS
        self.combat_grid_color = np.array([255, 255, 255])  # Blanc (cases)
        self.combat_bg_color = np.array([50, 80, 120])      # Bleu foncé (fond combat)
        self.timeline_color = np.array([200, 200, 200])     # Gris clair (timeline)

        # Seuils de détection
        self.color_tolerance = 30
        self.min_grid_cells = 10  # Minimum de cases pour confirmer grille
        self.min_timeline_width = 200  # Largeur minimum de la timeline

        logger.info("DofusCombatDetector initialisé")

    def is_in_combat(self, frame: np.ndarray) -> CombatIndicators:
        """
        Détecte si on est EN COMBAT (pas juste si des monstres sont visibles)

        Args:
            frame: Image capturée de l'écran

        Returns:
            CombatIndicators avec tous les indicateurs détectés
        """
        indicators = CombatIndicators()

        # 1. Détecter grille de combat (cases blanches)
        indicators.has_combat_grid = self._detect_combat_grid(frame)

        # 2. Détecter timeline en haut
        indicators.has_timeline = self._detect_timeline(frame)

        # 3. Détecter les gros PA/PM
        indicators.has_large_papm = self._detect_large_papm(frame)

        # 4. Détecter fond bleuté du combat
        indicators.has_combat_background = self._detect_combat_background(frame)

        # Calculer confiance
        total_indicators = sum([
            indicators.has_combat_grid,
            indicators.has_timeline,
            indicators.has_large_papm,
            indicators.has_combat_background
        ])
        indicators.confidence = total_indicators / 4.0

        if indicators.is_in_combat():
            logger.info(f"✓ EN COMBAT détecté (confiance: {indicators.confidence:.2f})")
        else:
            logger.debug(f"✗ PAS en combat (confiance: {indicators.confidence:.2f})")

        return indicators

    def _detect_combat_grid(self, frame: np.ndarray) -> bool:
        """
        Détecte la grille de combat (cases blanches/bleues)

        La grille apparaît UNIQUEMENT en combat
        """
        h, w = frame.shape[:2]
        x1 = int(w * self.roi_grid[0])
        y1 = int(h * self.roi_grid[1])
        x2 = int(w * self.roi_grid[2])
        y2 = int(h * self.roi_grid[3])

        roi = frame[y1:y2, x1:x2]

        # Convertir en HSV pour détecter blanc
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Masque pour couleurs claires (cases de la grille)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Détecter lignes (grille)
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=20,
            maxLineGap=10
        )

        if lines is not None:
            # Compter lignes horizontales et verticales
            h_lines = 0
            v_lines = 0

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1))

                if angle < 0.2:  # ~0 rad = horizontale
                    h_lines += 1
                elif angle > 1.4:  # ~π/2 rad = verticale
                    v_lines += 1

            # Grille = beaucoup de lignes croisées
            has_grid = (h_lines >= 3) and (v_lines >= 3)
            if has_grid:
                logger.debug(f"Grille détectée: {h_lines}H x {v_lines}V lignes")
            return has_grid

        return False

    def _detect_timeline(self, frame: np.ndarray) -> bool:
        """
        Détecte la timeline en haut (ordre des tours)

        Apparaît UNIQUEMENT en combat
        """
        h, w = frame.shape[:2]
        x1 = int(w * self.roi_timeline[0])
        y1 = int(h * self.roi_timeline[1])
        x2 = int(w * self.roi_timeline[2])
        y2 = int(h * self.roi_timeline[3])

        roi = frame[y1:y2, x1:x2]

        # Détecter barre horizontale grise/claire
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Chercher ligne horizontale longue
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=self.min_timeline_width,
            maxLineGap=20
        )

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = abs(x2 - x1)
                angle = abs(np.arctan2(y2 - y1, x2 - x1))

                # Ligne horizontale longue = timeline
                if length > self.min_timeline_width and angle < 0.2:
                    logger.debug(f"Timeline détectée (longueur: {length}px)")
                    return True

        return False

    def _detect_large_papm(self, frame: np.ndarray) -> bool:
        """
        Détecte les gros chiffres PA/PM en combat

        En combat, PA/PM sont BEAUCOUP plus gros qu'en exploration
        """
        h, w = frame.shape[:2]
        x1 = int(w * self.roi_papm[0])
        y1 = int(h * self.roi_papm[1])
        x2 = int(w * self.roi_papm[2])
        y2 = int(h * self.roi_papm[3])

        roi = frame[y1:y2, x1:x2]

        # Détecter gros texte (PA/PM en combat sont énormes)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Binariser
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Trouver contours (chiffres)
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Compter gros contours (chiffres de taille combat)
        large_digits = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Gros chiffre
                large_digits += 1

        # Au moins 2 gros chiffres = PA et PM affichés
        if large_digits >= 2:
            logger.debug(f"PA/PM combat détectés ({large_digits} gros chiffres)")
            return True

        return False

    def _detect_combat_background(self, frame: np.ndarray) -> bool:
        """
        Détecte le fond bleuté caractéristique du combat

        En combat, l'écran a une teinte bleue
        """
        # Échantillonner le centre de l'écran
        h, w = frame.shape[:2]
        center_y = h // 2
        center_x = w // 2
        sample_size = 50

        region = frame[
            center_y - sample_size:center_y + sample_size,
            center_x - sample_size:center_x + sample_size
        ]

        # Calculer couleur moyenne
        avg_color = np.mean(region, axis=(0, 1))

        # Vérifier si bleuté (B > R et B > G)
        b, g, r = avg_color
        is_blue_tinted = (b > r + 10) and (b > g + 10)

        if is_blue_tinted:
            logger.debug(f"Fond combat détecté (BGR: {b:.0f}, {g:.0f}, {r:.0f})")

        return is_blue_tinted

    def detect_monster_groups(
        self,
        frame: np.ndarray,
        show_names: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Détecte les groupes de monstres (cercles rouges)

        Args:
            frame: Image de l'écran
            show_names: Si True, simule TAB pour voir les noms

        Returns:
            Liste de groupes détectés avec positions
        """
        # Convertir en HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Masque pour rouge (cercles sous les groupes)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Morphologie pour nettoyer
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        # Détecter cercles
        circles = cv2.HoughCircles(
            red_mask,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=20,
            minRadius=10,
            maxRadius=40
        )

        monster_groups = []

        if circles is not None:
            circles = np.uint16(np.around(circles))

            for circle in circles[0, :]:
                x, y, r = circle

                group = {
                    'center': (int(x), int(y)),
                    'radius': int(r),
                    'type': 'monster_group',
                    'clickable': True
                }

                # Si show_names, on pourrait faire OCR ici
                # (mais il faudrait appuyer sur TAB avant)
                if show_names:
                    # TODO: Implémenter TAB + OCR
                    pass

                monster_groups.append(group)

            logger.info(f"✓ {len(monster_groups)} groupes de monstres détectés")

        return monster_groups

    def detect_npc_markers(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Détecte les marqueurs de NPCs (cercles jaunes/dorés)

        Returns:
            Liste de NPCs détectés
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Masque pour jaune/doré (NPCs)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Morphologie
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

        # Détecter cercles
        circles = cv2.HoughCircles(
            yellow_mask,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=20,
            minRadius=10,
            maxRadius=40
        )

        npcs = []

        if circles is not None:
            circles = np.uint16(np.around(circles))

            for circle in circles[0, :]:
                x, y, r = circle

                npc = {
                    'center': (int(x), int(y)),
                    'radius': int(r),
                    'type': 'npc',
                    'clickable': True
                }

                npcs.append(npc)

            logger.debug(f"{len(npcs)} NPCs détectés")

        return npcs


def create_dofus_combat_detector() -> DofusCombatDetector:
    """Factory function"""
    return DofusCombatDetector()
