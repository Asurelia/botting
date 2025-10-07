#!/usr/bin/env python3
"""
MapDiscovery - Auto-découverte des maps Dofus
Scanne automatiquement les maps visitées et construit le graph
"""

import cv2
import numpy as np
import pyautogui
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from .map_graph import MapGraph, MapNode, MapCoords, MapExit

@dataclass
class DiscoveredMap:
    """Résultat de la découverte d'une map"""
    coords: MapCoords
    name: Optional[str]
    exits: List[MapExit]
    npcs_count: int
    resources_count: int
    interactive_count: int
    screenshot: Optional[np.ndarray] = None
    walkable_cells: Optional[List[int]] = None

class MapDiscovery:
    """
    Auto-découverte des maps

    Scanne automatiquement chaque nouvelle map visitée:
    - OCR des coordonnées
    - Détection des sorties
    - Scan des éléments interactifs
    - Capture screenshot de référence
    """

    def __init__(self, map_graph: MapGraph):
        self.logger = logging.getLogger(__name__)
        self.map_graph = map_graph

        # Configuration OCR
        self.coord_region = None  # Zone où chercher les coordonnées

    def discover_current_map(self) -> Optional[DiscoveredMap]:
        """
        Découvre la map actuelle

        Returns:
            DiscoveredMap avec toutes les informations scannées
        """
        self.logger.info("Découverte de la map actuelle...")

        # Capture écran
        screenshot = pyautogui.screenshot()
        screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # 1. Détecte les coordonnées
        coords = self._detect_coordinates(screen)
        if not coords:
            self.logger.warning("Impossible de détecter les coordonnées de la map")
            return None

        self.logger.info(f"  Coordonnées: {coords}")

        # Vérifie si déjà découverte
        if coords in self.map_graph.discovered_maps:
            self.logger.info(f"  Map {coords} déjà découverte")
            return None

        # 2. Détecte les sorties
        exits = self._find_exits(screen)
        self.logger.info(f"  Sorties trouvées: {len(exits)}")

        # 3. Scanne éléments interactifs
        npcs = self._count_npcs(screen)
        resources = self._count_resources(screen)
        interactive = self._count_interactive(screen)

        self.logger.info(f"  NPCs: {npcs}, Ressources: {resources}, Interactifs: {interactive}")

        # 4. Détecte cellules marchables
        walkable = self._compute_walkable_cells(screen)

        # 5. OCR du nom de la map (si disponible)
        map_name = self._detect_map_name(screen)

        discovered = DiscoveredMap(
            coords=coords,
            name=map_name,
            exits=exits,
            npcs_count=npcs,
            resources_count=resources,
            interactive_count=interactive,
            screenshot=screen,
            walkable_cells=walkable
        )

        # Ajoute au graph
        self._add_to_graph(discovered)

        self.logger.info(f"[OK] Map {coords} découverte et ajoutée au graph")

        return discovered

    def _detect_coordinates(self, screen: np.ndarray) -> Optional[MapCoords]:
        """
        Détecte les coordonnées de la map par OCR

        Les coordonnées sont affichées en haut à gauche de la minimap
        Format: (X,Y)
        """
        # TODO: Implémenter OCR réel avec pytesseract ou easyocr
        # Pour l'instant, méthode simulée

        # Zone de recherche: près de la minimap (haut droite)
        height, width = screen.shape[:2]
        coord_region = screen[10:50, width-300:width-20]

        # OCR simulation (à remplacer par vrai OCR)
        # coords_text = pytesseract.image_to_string(coord_region)

        # Exemple: détection simulée
        # En production, parser le texte OCR pour extraire (X,Y)

        # Pour le moment, retourne None pour forcer calibration manuelle
        return None

    def _find_exits(self, screen: np.ndarray) -> List[MapExit]:
        """
        Détecte toutes les sorties de la map

        Sorties = bords de l'écran où le personnage peut sortir
        + portes, zaaps, passages spéciaux
        """
        exits = []
        height, width = screen.shape[:2]

        # Zones de bord
        edge_zones = {
            'north': {'y': 0, 'height': 80, 'x': 0, 'width': width},
            'south': {'y': height-80, 'height': 80, 'x': 0, 'width': width},
            'east': {'x': width-80, 'width': 80, 'y': 0, 'height': height},
            'west': {'x': 0, 'width': 80, 'y': 0, 'height': height}
        }

        # Détection par changement de curseur ou indicateur visuel
        for direction, zone in edge_zones.items():
            if self._has_exit_in_zone(screen, zone):
                exit_obj = MapExit(
                    direction=direction,
                    exit_type='edge',
                    cell_id=None
                )
                exits.append(exit_obj)
                self.logger.debug(f"    Sortie {direction} détectée")

        # Détection de portes/zaaps (template matching)
        doors = self._find_doors(screen)
        for door_pos in doors:
            exit_obj = MapExit(
                direction='door',
                exit_type='teleport',
                position=door_pos
            )
            exits.append(exit_obj)
            self.logger.debug(f"    Porte détectée à {door_pos}")

        return exits

    def _has_exit_in_zone(self, screen: np.ndarray, zone: Dict) -> bool:
        """Vérifie si une sortie existe dans la zone"""
        # Extraction de la zone
        region = screen[
            zone['y']:zone['y']+zone['height'],
            zone['x']:zone['x']+zone['width']
        ]

        # Détection basique: cherche des indicateurs visuels
        # TODO: Améliorer avec template matching d'icônes de sortie

        # Pour l'instant, heuristique simple
        # (En prod, analyser changement de curseur ou flèches directionnelles)

        return True  # Assume exits existent par défaut

    def _find_doors(self, screen: np.ndarray) -> List[Tuple[int, int]]:
        """Détecte les portes/zaaps par template matching"""
        doors = []

        # TODO: Template matching réel avec templates de portes
        # door_template = cv2.imread('assets/templates/door.png')
        # result = cv2.matchTemplate(screen, door_template, cv2.TM_CCOEFF_NORMED)

        return doors

    def _count_npcs(self, screen: np.ndarray) -> int:
        """Compte le nombre de NPCs sur la map"""
        # Détection par couleur ou template matching
        # NPCs ont souvent des noms en cyan au-dessus de leur tête

        # TODO: Implémenter détection réelle
        return 0

    def _count_resources(self, screen: np.ndarray) -> int:
        """Compte le nombre de ressources (arbres, minerais, etc.)"""
        # Détection par template matching ou analyse de couleurs

        # TODO: Implémenter détection réelle
        return 0

    def _count_interactive(self, screen: np.ndarray) -> int:
        """Compte le nombre d'éléments interactifs"""
        # Portes, coffres, PNJs, etc.

        # TODO: Implémenter détection réelle
        return 0

    def _compute_walkable_cells(self, screen: np.ndarray) -> List[int]:
        """Calcule les cellules marchables de la map"""
        # Dofus utilise une grille isométrique
        # Chaque map = 560 cellules (14x40)

        # TODO: Implémenter détection réelle des cellules
        # Nécessite analyse de la grille du jeu

        return []

    def _detect_map_name(self, screen: np.ndarray) -> Optional[str]:
        """Détecte le nom de la map (si affiché)"""
        # OCR de la zone où le nom apparaît (souvent au centre en haut)

        # TODO: Implémenter OCR réel
        return None

    def _add_to_graph(self, discovered: DiscoveredMap):
        """Ajoute la map découverte au graph"""
        # Crée le nœud
        map_node = MapNode(
            coords=discovered.coords,
            name=discovered.name,
            exits=discovered.exits,
            discovered=True
        )

        self.map_graph.add_map(map_node)
        self.map_graph.mark_discovered(discovered.coords)

        # Ajoute les connexions aux maps voisines (si connues)
        for exit in discovered.exits:
            if exit.destination:
                # TODO: Créer MapEdge et ajouter connexion
                pass

    def explore_unknown_neighbors(self, current_coords: MapCoords) -> Optional[MapCoords]:
        """
        Trouve la map voisine non découverte la plus proche

        Returns:
            Coordonnées de la map à explorer, ou None si toutes connues
        """
        neighbors = self.map_graph.get_neighbors(current_coords)

        for neighbor in neighbors:
            if neighbor not in self.map_graph.discovered_maps:
                self.logger.info(f"Map voisine non découverte trouvée: {neighbor}")
                return neighbor

        self.logger.debug("Toutes les maps voisines sont découvertes")
        return None

def create_map_discovery(map_graph: MapGraph) -> MapDiscovery:
    """Factory function pour créer MapDiscovery"""
    return MapDiscovery(map_graph)