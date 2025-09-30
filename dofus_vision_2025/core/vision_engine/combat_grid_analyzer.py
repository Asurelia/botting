"""
Combat Grid Analyzer - Analyseur Grille de Combat DOFUS Unity
Module spécialisé pour l'analyse de la grille de combat tactique DOFUS
Détection positions, portées sorts, ligne de vue, placement optimal
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)

class CellType(Enum):
    """Types de cellules de la grille"""
    EMPTY = "empty"
    OCCUPIED_ALLY = "occupied_ally"
    OCCUPIED_ENEMY = "occupied_enemy"
    BLOCKED = "blocked"
    HIGHLIGHTED = "highlighted"
    SPELL_RANGE = "spell_range"
    MOVEMENT_RANGE = "movement_range"

class Direction(Enum):
    """Directions pour ligne de vue"""
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)
    NORTHEAST = (1, -1)
    NORTHWEST = (-1, -1)
    SOUTHEAST = (1, 1)
    SOUTHWEST = (-1, 1)

@dataclass
class GridCell:
    """Cellule de la grille de combat"""
    x: int
    y: int
    cell_type: CellType
    screen_pos: Tuple[int, int]  # Position pixels à l'écran
    walkable: bool = True
    entity_id: Optional[str] = None
    spell_castable: bool = True

@dataclass
class Entity:
    """Entité sur la grille (joueur, monstre, invocation)"""
    entity_id: str
    name: str
    position: Tuple[int, int]
    entity_type: str  # "player", "ally", "enemy", "summon"
    health_percent: float = 100.0
    action_points: int = 0
    movement_points: int = 0
    is_current_turn: bool = False

@dataclass
class SpellRange:
    """Portée d'un sort"""
    spell_name: str
    min_range: int
    max_range: int
    line_of_sight: bool
    diagonal: bool
    castable_cells: Set[Tuple[int, int]]

class DofusCombatGridAnalyzer:
    """
    Analyseur de grille de combat DOFUS Unity
    Reconnaissance et analyse tactique de la grille hexagonale
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.grid: Dict[Tuple[int, int], GridCell] = {}
        self.entities: Dict[str, Entity] = {}
        self.grid_bounds = (0, 0, 0, 0)  # min_x, min_y, max_x, max_y
        self.cell_size = 40  # Taille approximative cellule en pixels
        self.grid_center = (0, 0)

        # Templates pour reconnaissance cellules
        self.cell_templates = {}
        self.load_cell_templates()

        logger.info("DofusCombatGridAnalyzer initialisé")

    def _get_default_config(self) -> Dict:
        """Configuration par défaut"""
        return {
            "grid_region": (100, 100, 800, 600),  # Région probable de la grille
            "cell_detection_threshold": 0.7,
            "entity_detection_threshold": 0.8,
            "grid_size_estimate": (15, 17),  # Taille grille typique DOFUS
            "hex_grid": True  # Grille hexagonale
        }

    def load_cell_templates(self):
        """Charge les templates pour détecter types de cellules"""
        templates = {
            "empty_cell": "templates/combat/empty_cell.png",
            "ally_occupied": "templates/combat/ally_cell.png",
            "enemy_occupied": "templates/combat/enemy_cell.png",
            "movement_range": "templates/combat/movement_cell.png",
            "spell_range": "templates/combat/spell_range_cell.png",
            "blocked_cell": "templates/combat/blocked_cell.png"
        }

        for name, path in templates.items():
            try:
                template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    self.cell_templates[name] = template
            except:
                # Créer templates synthétiques pour démo
                template = np.zeros((30, 30), dtype=np.uint8)
                if name == "ally_occupied":
                    cv2.circle(template, (15, 15), 12, 100, -1)
                elif name == "enemy_occupied":
                    cv2.circle(template, (15, 15), 12, 200, -1)
                elif name == "movement_range":
                    cv2.rectangle(template, (5, 5), (25, 25), 150, 2)

                self.cell_templates[name] = template

        logger.info(f"Templates cellules chargés: {len(self.cell_templates)}")

    def detect_grid_bounds(self, screenshot: np.ndarray) -> Tuple[int, int, int, int]:
        """Détecte les limites de la grille de combat"""
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Recherche contours hexagonaux/carrés pour identifier la grille
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrer contours par taille (cellules de grille)
        min_area = self.cell_size * self.cell_size * 0.5
        max_area = self.cell_size * self.cell_size * 2.0

        grid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Vérifier forme approximativement hexagonale/carrée
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if 4 <= len(approx) <= 8:  # Forme géométrique simple
                    grid_contours.append(contour)

        if grid_contours:
            # Calculer bounds global
            all_points = np.vstack(grid_contours)
            x_coords = all_points[:, 0, 0]
            y_coords = all_points[:, 0, 1]

            bounds = (
                int(np.min(x_coords)), int(np.min(y_coords)),
                int(np.max(x_coords)), int(np.max(y_coords))
            )

            self.grid_bounds = bounds
            return bounds

        # Fallback: utiliser région configurée
        return self.config["grid_region"]

    def extract_grid_cells(self, screenshot: np.ndarray) -> Dict[Tuple[int, int], GridCell]:
        """Extrait toutes les cellules de la grille"""
        self.grid = {}
        bounds = self.detect_grid_bounds(screenshot)
        min_x, min_y, max_x, max_y = bounds

        # Région de la grille
        grid_region = screenshot[min_y:max_y, min_x:max_x]
        gray_region = cv2.cvtColor(grid_region, cv2.COLOR_BGR2GRAY)

        # Estimation positions cellules
        grid_width = max_x - min_x
        grid_height = max_y - min_y
        estimated_cols, estimated_rows = self.config["grid_size_estimate"]

        cell_width = grid_width // estimated_cols
        cell_height = grid_height // estimated_rows

        # Analyse chaque position potentielle
        for row in range(estimated_rows):
            for col in range(estimated_cols):
                # Position dans la grille
                grid_x, grid_y = col, row

                # Position pixels (ajustement hexagonal si nécessaire)
                if self.config["hex_grid"] and row % 2 == 1:
                    pixel_x = min_x + col * cell_width + cell_width // 2
                else:
                    pixel_x = min_x + col * cell_width

                pixel_y = min_y + row * cell_height

                # Extraction cellule
                cell_region = self._extract_cell_region(
                    gray_region, pixel_x - min_x, pixel_y - min_y, cell_width, cell_height
                )

                if cell_region is not None:
                    cell_type = self._classify_cell(cell_region)
                    cell = GridCell(
                        x=grid_x,
                        y=grid_y,
                        cell_type=cell_type,
                        screen_pos=(pixel_x, pixel_y),
                        walkable=(cell_type not in [CellType.BLOCKED, CellType.OCCUPIED_ALLY, CellType.OCCUPIED_ENEMY])
                    )

                    self.grid[(grid_x, grid_y)] = cell

        logger.info(f"Grille extraite: {len(self.grid)} cellules")
        return self.grid

    def _extract_cell_region(self, gray_image: np.ndarray, x: int, y: int,
                           width: int, height: int) -> Optional[np.ndarray]:
        """Extrait la région d'une cellule"""
        h, w = gray_image.shape

        # Vérifier bounds
        if x < 0 or y < 0 or x + width > w or y + height > h:
            return None

        return gray_image[y:y+height, x:x+width]

    def _classify_cell(self, cell_image: np.ndarray) -> CellType:
        """Classifie le type d'une cellule via template matching"""
        best_match = CellType.EMPTY
        best_confidence = 0

        for template_name, template in self.cell_templates.items():
            if template.shape[0] > cell_image.shape[0] or template.shape[1] > cell_image.shape[1]:
                continue

            # Resize template si nécessaire
            if template.shape != cell_image.shape:
                template = cv2.resize(template, (cell_image.shape[1], cell_image.shape[0]))

            # Template matching
            result = cv2.matchTemplate(cell_image, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_confidence:
                best_confidence = max_val
                if template_name == "ally_occupied":
                    best_match = CellType.OCCUPIED_ALLY
                elif template_name == "enemy_occupied":
                    best_match = CellType.OCCUPIED_ENEMY
                elif template_name == "movement_range":
                    best_match = CellType.MOVEMENT_RANGE
                elif template_name == "spell_range":
                    best_match = CellType.SPELL_RANGE
                elif template_name == "blocked_cell":
                    best_match = CellType.BLOCKED

        return best_match

    def detect_entities(self, screenshot: np.ndarray) -> Dict[str, Entity]:
        """Détecte toutes les entités sur la grille"""
        self.entities = {}

        for pos, cell in self.grid.items():
            if cell.cell_type in [CellType.OCCUPIED_ALLY, CellType.OCCUPIED_ENEMY]:
                entity_type = "ally" if cell.cell_type == CellType.OCCUPIED_ALLY else "enemy"

                # Génération ID unique
                entity_id = f"{entity_type}_{pos[0]}_{pos[1]}"

                entity = Entity(
                    entity_id=entity_id,
                    name=f"{entity_type.title()} {pos}",
                    position=pos,
                    entity_type=entity_type
                )

                self.entities[entity_id] = entity

        return self.entities

    def calculate_movement_range(self, from_pos: Tuple[int, int],
                               movement_points: int) -> Set[Tuple[int, int]]:
        """Calcule les cellules accessibles avec les PM"""
        reachable = set()
        queue = [(from_pos, 0)]  # (position, cost)
        visited = set()

        while queue:
            pos, cost = queue.pop(0)
            if pos in visited or cost > movement_points:
                continue

            visited.add(pos)
            reachable.add(pos)

            # Cellules adjacentes (hexagonal)
            for neighbor in self._get_adjacent_cells(pos):
                if (neighbor not in visited and
                    neighbor in self.grid and
                    self.grid[neighbor].walkable):
                    queue.append((neighbor, cost + 1))

        return reachable

    def calculate_spell_range(self, from_pos: Tuple[int, int], spell_range: int,
                            line_of_sight: bool = False,
                            diagonal: bool = True) -> Set[Tuple[int, int]]:
        """Calcule les cellules dans portée d'un sort"""
        castable = set()

        for pos, cell in self.grid.items():
            distance = self._calculate_distance(from_pos, pos)

            if distance <= spell_range:
                if not line_of_sight or self._has_line_of_sight(from_pos, pos):
                    if diagonal or self._is_cardinal_direction(from_pos, pos):
                        castable.add(pos)

        return castable

    def _get_adjacent_cells(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Retourne les cellules adjacentes (hexagonal ou carré)"""
        x, y = pos

        if self.config["hex_grid"]:
            # Grille hexagonale
            if y % 2 == 0:  # Ligne paire
                neighbors = [
                    (x-1, y), (x+1, y),      # Gauche, droite
                    (x-1, y-1), (x, y-1),    # Haut gauche, haut droite
                    (x-1, y+1), (x, y+1)     # Bas gauche, bas droite
                ]
            else:  # Ligne impaire
                neighbors = [
                    (x-1, y), (x+1, y),      # Gauche, droite
                    (x, y-1), (x+1, y-1),    # Haut gauche, haut droite
                    (x, y+1), (x+1, y+1)     # Bas gauche, bas droite
                ]
        else:
            # Grille carrée
            neighbors = [
                (x-1, y), (x+1, y), (x, y-1), (x, y+1),  # Cardinal
                (x-1, y-1), (x+1, y-1), (x-1, y+1), (x+1, y+1)  # Diagonal
            ]

        return [(nx, ny) for nx, ny in neighbors if (nx, ny) in self.grid]

    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calcule la distance entre deux cellules"""
        if self.config["hex_grid"]:
            # Distance hexagonale
            x1, y1 = pos1
            x2, y2 = pos2

            # Conversion coordonnées axiales
            q1 = x1 - (y1 - (y1 & 1)) // 2
            r1 = y1
            q2 = x2 - (y2 - (y2 & 1)) // 2
            r2 = y2

            return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) // 2
        else:
            # Distance Manhattan
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _has_line_of_sight(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """Vérifie s'il y a ligne de vue entre deux positions"""
        # Algorithme de Bresenham adapté grille
        x1, y1 = from_pos
        x2, y2 = to_pos

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        x_step = 1 if x1 < x2 else -1
        y_step = 1 if y1 < y2 else -1

        error = dx - dy
        x, y = x1, y1

        while (x, y) != (x2, y2):
            # Vérifier si cellule bloque la vue
            if (x, y) in self.grid and not self.grid[(x, y)].walkable:
                return False

            e2 = 2 * error
            if e2 > -dy:
                error -= dy
                x += x_step
            if e2 < dx:
                error += dx
                y += y_step

        return True

    def _is_cardinal_direction(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """Vérifie si la direction est cardinale (pas diagonale)"""
        dx = abs(from_pos[0] - to_pos[0])
        dy = abs(from_pos[1] - to_pos[1])
        return dx == 0 or dy == 0

    def get_tactical_analysis(self) -> Dict:
        """Analyse tactique complète de la situation"""
        analysis = {
            "grid_cells": len(self.grid),
            "entities": len(self.entities),
            "allies": len([e for e in self.entities.values() if e.entity_type == "ally"]),
            "enemies": len([e for e in self.entities.values() if e.entity_type == "enemy"]),
            "strategic_positions": [],
            "threats": [],
            "opportunities": []
        }

        # Analyse positions stratégiques
        for pos, cell in self.grid.items():
            if cell.walkable:
                # Compter entités adjacentes
                adjacent_entities = 0
                for neighbor_pos in self._get_adjacent_cells(pos):
                    neighbor_cell = self.grid.get(neighbor_pos)
                    if neighbor_cell and neighbor_cell.cell_type in [CellType.OCCUPIED_ALLY, CellType.OCCUPIED_ENEMY]:
                        adjacent_entities += 1

                if adjacent_entities >= 2:
                    analysis["strategic_positions"].append(pos)

        return analysis

    def find_optimal_positions(self, entity_type: str = "ally") -> List[Tuple[int, int]]:
        """Trouve les positions optimales pour une entité"""
        optimal_positions = []

        for pos, cell in self.grid.items():
            if not cell.walkable:
                continue

            score = 0

            # Distance aux ennemis (plus c'est loin, mieux c'est pour supports)
            enemy_distances = []
            for entity in self.entities.values():
                if entity.entity_type == "enemy":
                    distance = self._calculate_distance(pos, entity.position)
                    enemy_distances.append(distance)

            if enemy_distances:
                avg_enemy_distance = sum(enemy_distances) / len(enemy_distances)
                score += avg_enemy_distance * 10

            # Proximité alliés (pour support)
            ally_count = 0
            for entity in self.entities.values():
                if entity.entity_type == "ally":
                    distance = self._calculate_distance(pos, entity.position)
                    if distance <= 3:
                        ally_count += 1

            score += ally_count * 5

            # Éviter coins/bords
            adjacent_count = len(self._get_adjacent_cells(pos))
            score += adjacent_count

            optimal_positions.append((pos, score))

        # Trier par score décroissant
        optimal_positions.sort(key=lambda x: x[1], reverse=True)
        return [pos for pos, score in optimal_positions[:5]]

def create_combat_grid_analyzer(config: Optional[Dict] = None) -> DofusCombatGridAnalyzer:
    """Factory pour créer l'analyseur de grille"""
    return DofusCombatGridAnalyzer(config)

# Test du module
if __name__ == "__main__":
    analyzer = create_combat_grid_analyzer()

    # Test avec image fictive
    test_image = np.zeros((600, 800, 3), dtype=np.uint8)

    grid = analyzer.extract_grid_cells(test_image)
    entities = analyzer.detect_entities(test_image)

    print(f"Grille détectée: {len(grid)} cellules")
    print(f"Entités détectées: {len(entities)}")

    if entities:
        analysis = analyzer.get_tactical_analysis()
        print(f"Analyse tactique: {analysis}")

        optimal_pos = analyzer.find_optimal_positions()
        print(f"Positions optimales: {optimal_pos[:3]}")