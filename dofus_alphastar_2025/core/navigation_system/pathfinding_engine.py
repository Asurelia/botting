#!/usr/bin/env python3
"""
PathfindingEngine - Moteur de pathfinding A* optimisé pour DOFUS
Gère la navigation précise sur grilles hexagonales et obstacles dynamiques
"""

import time
import heapq
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from enum import Enum
import math

import numpy as np

from config import config

logger = logging.getLogger(__name__)

class CellType(Enum):
    """Types de cellules de grille"""
    WALKABLE = "walkable"
    BLOCKED = "blocked"
    DANGEROUS = "dangerous"
    WATER = "water"
    TELEPORTER = "teleporter"
    NPC = "npc"
    MONSTER = "monster"
    UNKNOWN = "unknown"

class MovementType(Enum):
    """Types de mouvement"""
    WALK = "walk"
    RUN = "run"
    JUMP = "jump"
    TELEPORT = "teleport"

@dataclass
class PathNode:
    """Nœud de chemin pour A*"""
    position: Tuple[int, int]
    g_cost: float = float('inf')  # Coût depuis départ
    h_cost: float = 0.0  # Heuristique vers arrivée
    f_cost: float = float('inf')  # g_cost + h_cost
    parent: Optional['PathNode'] = None
    cell_type: CellType = CellType.WALKABLE
    movement_cost: float = 1.0

    def __lt__(self, other):
        return self.f_cost < other.f_cost

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):
        return hash(self.position)

@dataclass
class PathResult:
    """Résultat de pathfinding"""
    success: bool
    path: List[Tuple[int, int]] = field(default_factory=list)
    total_cost: float = 0.0
    total_distance: float = 0.0
    computation_time: float = 0.0
    nodes_explored: int = 0
    path_type: str = "direct"
    warnings: List[str] = field(default_factory=list)

class GridMap:
    """Carte de grille pour pathfinding"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # Grille des types de cellules
        self.grid = np.full((height, width), CellType.WALKABLE, dtype=object)

        # Coûts de mouvement personnalisés
        self.movement_costs = np.ones((height, width), dtype=float)

        # Obstacles temporaires (monstres, joueurs)
        self.dynamic_obstacles: Set[Tuple[int, int]] = set()

        # Cache de visibilité
        self.visibility_cache: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}

    def is_valid_position(self, x: int, y: int) -> bool:
        """Vérifie si position valide"""
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, x: int, y: int, ignore_dynamic: bool = False) -> bool:
        """Vérifie si cellule est marchable"""
        if not self.is_valid_position(x, y):
            return False

        # Obstacles dynamiques
        if not ignore_dynamic and (x, y) in self.dynamic_obstacles:
            return False

        # Type de cellule
        cell_type = self.grid[y, x]
        return cell_type in [CellType.WALKABLE, CellType.TELEPORTER]

    def get_movement_cost(self, x: int, y: int) -> float:
        """Récupère coût de mouvement"""
        if not self.is_valid_position(x, y):
            return float('inf')

        base_cost = self.movement_costs[y, x]

        # Coûts spéciaux par type
        cell_type = self.grid[y, x]
        if cell_type == CellType.DANGEROUS:
            base_cost *= 2.0
        elif cell_type == CellType.WATER:
            base_cost *= 1.5

        return base_cost

    def set_cell_type(self, x: int, y: int, cell_type: CellType, movement_cost: float = 1.0):
        """Définit type et coût d'une cellule"""
        if self.is_valid_position(x, y):
            self.grid[y, x] = cell_type
            self.movement_costs[y, x] = movement_cost

    def add_dynamic_obstacle(self, x: int, y: int):
        """Ajoute obstacle temporaire"""
        if self.is_valid_position(x, y):
            self.dynamic_obstacles.add((x, y))

    def remove_dynamic_obstacle(self, x: int, y: int):
        """Supprime obstacle temporaire"""
        self.dynamic_obstacles.discard((x, y))

    def clear_dynamic_obstacles(self):
        """Vide tous les obstacles temporaires"""
        self.dynamic_obstacles.clear()

    def get_neighbors_hexagonal(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Récupère voisins hexagonaux (DOFUS style)"""
        neighbors = []

        # Offsets hexagonaux selon parité de Y
        if y % 2 == 0:  # Ligne paire
            offsets = [(-1, -1), (0, -1), (-1, 0), (1, 0), (-1, 1), (0, 1)]
        else:  # Ligne impaire
            offsets = [(0, -1), (1, -1), (-1, 0), (1, 0), (0, 1), (1, 1)]

        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if self.is_valid_position(nx, ny):
                neighbors.append((nx, ny))

        return neighbors

    def get_neighbors_square(self, x: int, y: int, diagonal: bool = True) -> List[Tuple[int, int]]:
        """Récupère voisins carrés (8-directional ou 4-directional)"""
        neighbors = []

        if diagonal:
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if self.is_valid_position(nx, ny):
                neighbors.append((nx, ny))

        return neighbors

class Heuristics:
    """Fonctions heuristiques pour A*"""

    @staticmethod
    def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Distance de Manhattan"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @staticmethod
    def euclidean_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Distance euclidienne"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    @staticmethod
    def hexagonal_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Distance hexagonale (pour grilles DOFUS)"""
        x1, y1 = pos1
        x2, y2 = pos2

        # Conversion coordonnées hexagonales
        q1 = x1 - (y1 - (y1 & 1)) // 2
        r1 = y1
        s1 = -q1 - r1

        q2 = x2 - (y2 - (y2 & 1)) // 2
        r2 = y2
        s2 = -q2 - r2

        return max(abs(q1 - q2), abs(r1 - r2), abs(s1 - s2))

    @staticmethod
    def chebyshev_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Distance de Chebyshev"""
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))

class PathfindingEngine:
    """Moteur de pathfinding principal"""

    def __init__(self, grid_map: GridMap):
        self.grid_map = grid_map

        # Configuration
        self.use_hexagonal_grid = True  # DOFUS utilise grilles hexagonales
        self.heuristic_weight = 1.0
        self.max_iterations = 10000
        self.allow_diagonal = True

        # Heuristique par défaut
        if self.use_hexagonal_grid:
            self.heuristic_func = Heuristics.hexagonal_distance
            self.get_neighbors_func = grid_map.get_neighbors_hexagonal
        else:
            self.heuristic_func = Heuristics.euclidean_distance
            self.get_neighbors_func = lambda x, y: grid_map.get_neighbors_square(x, y, self.allow_diagonal)

        # Statistiques
        self.total_pathfinding_calls = 0
        self.total_computation_time = 0.0
        self.cache_hits = 0

        # Cache de chemins
        self.path_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], PathResult] = {}
        self.cache_max_size = 1000

        logger.info(f"PathfindingEngine initialisé (hexagonal: {self.use_hexagonal_grid})")

    def find_path(self,
                  start: Tuple[int, int],
                  goal: Tuple[int, int],
                  movement_type: MovementType = MovementType.WALK,
                  max_cost: Optional[float] = None,
                  avoid_dangerous: bool = True) -> PathResult:
        """Trouve chemin optimal avec A*"""

        start_time = time.time()
        self.total_pathfinding_calls += 1

        # Vérifier cache
        cache_key = (start, goal)
        if cache_key in self.path_cache:
            cached_result = self.path_cache[cache_key]
            # Vérifier si cache encore valide (pas d'obstacles dynamiques)
            if not self._path_has_dynamic_obstacles(cached_result.path):
                self.cache_hits += 1
                return cached_result

        # Validation initiale
        if not self.grid_map.is_valid_position(*start):
            return PathResult(success=False, warnings=["Position de départ invalide"])

        if not self.grid_map.is_valid_position(*goal):
            return PathResult(success=False, warnings=["Position d'arrivée invalide"])

        if not self.grid_map.is_walkable(*goal):
            return PathResult(success=False, warnings=["Position d'arrivée non marchable"])

        if start == goal:
            return PathResult(
                success=True,
                path=[start],
                total_cost=0.0,
                total_distance=0.0,
                computation_time=time.time() - start_time
            )

        # A* principal
        result = self._astar_pathfinding(start, goal, movement_type, max_cost, avoid_dangerous)

        # Computation time
        result.computation_time = time.time() - start_time
        self.total_computation_time += result.computation_time

        # Mettre en cache si succès
        if result.success and len(self.path_cache) < self.cache_max_size:
            self.path_cache[cache_key] = result

        return result

    def _astar_pathfinding(self,
                          start: Tuple[int, int],
                          goal: Tuple[int, int],
                          movement_type: MovementType,
                          max_cost: Optional[float],
                          avoid_dangerous: bool) -> PathResult:
        """Implémentation A* core"""

        # Structures de données A*
        open_set = []
        closed_set: Set[Tuple[int, int]] = set()
        nodes: Dict[Tuple[int, int], PathNode] = {}

        # Nœud de départ
        start_node = PathNode(
            position=start,
            g_cost=0.0,
            h_cost=self.heuristic_func(start, goal) * self.heuristic_weight
        )
        start_node.f_cost = start_node.g_cost + start_node.h_cost

        nodes[start] = start_node
        heapq.heappush(open_set, start_node)

        iterations = 0

        while open_set and iterations < self.max_iterations:
            iterations += 1

            # Nœud avec plus petit f_cost
            current_node = heapq.heappop(open_set)
            current_pos = current_node.position

            # Objectif atteint
            if current_pos == goal:
                path = self._reconstruct_path(current_node)
                total_cost = current_node.g_cost
                total_distance = len(path) - 1

                return PathResult(
                    success=True,
                    path=path,
                    total_cost=total_cost,
                    total_distance=total_distance,
                    nodes_explored=iterations,
                    path_type="astar"
                )

            closed_set.add(current_pos)

            # Explorer voisins
            neighbors = self.get_neighbors_func(*current_pos)

            for neighbor_pos in neighbors:
                nx, ny = neighbor_pos

                if neighbor_pos in closed_set:
                    continue

                # Vérifier si marchable
                if not self.grid_map.is_walkable(nx, ny):
                    continue

                # Éviter zones dangereuses si demandé
                if avoid_dangerous and self.grid_map.grid[ny, nx] == CellType.DANGEROUS:
                    continue

                # Calculer coût du mouvement
                move_cost = self._calculate_movement_cost(
                    current_pos, neighbor_pos, movement_type
                )

                tentative_g_cost = current_node.g_cost + move_cost

                # Vérifier limite de coût
                if max_cost and tentative_g_cost > max_cost:
                    continue

                # Créer ou récupérer nœud voisin
                if neighbor_pos not in nodes:
                    neighbor_node = PathNode(
                        position=neighbor_pos,
                        h_cost=self.heuristic_func(neighbor_pos, goal) * self.heuristic_weight
                    )
                    nodes[neighbor_pos] = neighbor_node
                else:
                    neighbor_node = nodes[neighbor_pos]

                # Meilleur chemin trouvé
                if tentative_g_cost < neighbor_node.g_cost:
                    neighbor_node.parent = current_node
                    neighbor_node.g_cost = tentative_g_cost
                    neighbor_node.f_cost = neighbor_node.g_cost + neighbor_node.h_cost

                    # Ajouter à open_set si pas déjà présent
                    if neighbor_node not in open_set:
                        heapq.heappush(open_set, neighbor_node)

        # Aucun chemin trouvé
        warnings = []
        if iterations >= self.max_iterations:
            warnings.append("Limite d'itérations atteinte")

        return PathResult(
            success=False,
            nodes_explored=iterations,
            warnings=warnings
        )

    def _calculate_movement_cost(self,
                               from_pos: Tuple[int, int],
                               to_pos: Tuple[int, int],
                               movement_type: MovementType) -> float:
        """Calcule coût de mouvement entre deux positions"""

        # Coût de base de la cellule destination
        base_cost = self.grid_map.get_movement_cost(*to_pos)

        # Distance
        if self.use_hexagonal_grid:
            distance = 1.0  # Hexagones sont uniformes
        else:
            distance = self.heuristic_func(from_pos, to_pos)

        # Modificateur par type de mouvement
        movement_modifier = {
            MovementType.WALK: 1.0,
            MovementType.RUN: 0.8,  # Plus rapide
            MovementType.JUMP: 1.2,  # Plus coûteux
            MovementType.TELEPORT: 0.1  # Très rapide
        }.get(movement_type, 1.0)

        return base_cost * distance * movement_modifier

    def _reconstruct_path(self, end_node: PathNode) -> List[Tuple[int, int]]:
        """Reconstitue le chemin depuis le nœud final"""
        path = []
        current = end_node

        while current:
            path.append(current.position)
            current = current.parent

        return list(reversed(path))

    def _path_has_dynamic_obstacles(self, path: List[Tuple[int, int]]) -> bool:
        """Vérifie si un chemin contient des obstacles dynamiques"""
        for pos in path:
            if pos in self.grid_map.dynamic_obstacles:
                return True
        return False

    def find_path_with_waypoints(self,
                                start: Tuple[int, int],
                                waypoints: List[Tuple[int, int]],
                                goal: Tuple[int, int],
                                movement_type: MovementType = MovementType.WALK) -> PathResult:
        """Trouve chemin passant par des waypoints"""

        full_path = []
        total_cost = 0.0
        total_distance = 0.0
        total_iterations = 0
        warnings = []

        # Séquence de destinations
        destinations = [start] + waypoints + [goal]

        for i in range(len(destinations) - 1):
            segment_start = destinations[i]
            segment_end = destinations[i + 1]

            segment_result = self.find_path(segment_start, segment_end, movement_type)

            if not segment_result.success:
                return PathResult(
                    success=False,
                    warnings=[f"Impossible d'atteindre waypoint {i}: {segment_end}"] + segment_result.warnings
                )

            # Fusionner segments (éviter duplication de positions)
            if i == 0:
                full_path.extend(segment_result.path)
            else:
                full_path.extend(segment_result.path[1:])  # Skip premier (déjà dans path)

            total_cost += segment_result.total_cost
            total_distance += segment_result.total_distance
            total_iterations += segment_result.nodes_explored
            warnings.extend(segment_result.warnings)

        return PathResult(
            success=True,
            path=full_path,
            total_cost=total_cost,
            total_distance=total_distance,
            nodes_explored=total_iterations,
            path_type="waypoints",
            warnings=warnings
        )

    def find_safe_path(self,
                      start: Tuple[int, int],
                      goal: Tuple[int, int],
                      safety_radius: int = 2) -> PathResult:
        """Trouve chemin évitant les zones dangereuses"""

        # Temporairement marquer cellules proches des dangers comme coûteuses
        original_costs = {}

        for y in range(self.grid_map.height):
            for x in range(self.grid_map.width):
                if self.grid_map.grid[y, x] in [CellType.DANGEROUS, CellType.MONSTER]:
                    # Augmenter coût dans le rayon
                    for dy in range(-safety_radius, safety_radius + 1):
                        for dx in range(-safety_radius, safety_radius + 1):
                            nx, ny = x + dx, y + dy
                            if self.grid_map.is_valid_position(nx, ny):
                                if (nx, ny) not in original_costs:
                                    original_costs[(nx, ny)] = self.grid_map.movement_costs[ny, nx]
                                    self.grid_map.movement_costs[ny, nx] *= 3.0

        # Pathfinding normal
        result = self.find_path(start, goal, avoid_dangerous=True)
        result.path_type = "safe"

        # Restaurer coûts originaux
        for (x, y), original_cost in original_costs.items():
            self.grid_map.movement_costs[y, x] = original_cost

        return result

    def optimize_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Optimise un chemin existant (smoothing)"""
        if len(path) <= 2:
            return path

        optimized = [path[0]]

        i = 0
        while i < len(path) - 1:
            # Chercher le point le plus loin directement accessible
            farthest = i + 1

            for j in range(i + 2, len(path)):
                if self._has_clear_line_of_sight(path[i], path[j]):
                    farthest = j
                else:
                    break

            optimized.append(path[farthest])
            i = farthest

        return optimized

    def _has_clear_line_of_sight(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Vérifie ligne de vue dégagée entre deux points"""

        # Algorithme de Bresenham pour tracer ligne
        x0, y0 = start
        x1, y1 = end

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        err = dx - dy

        while True:
            # Vérifier si position marchable
            if not self.grid_map.is_walkable(x0, y0):
                return False

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err

            if e2 > -dy:
                err -= dy
                x0 += sx

            if e2 < dx:
                err += dx
                y0 += sy

        return True

    def get_reachable_area(self, start: Tuple[int, int], max_cost: float) -> Set[Tuple[int, int]]:
        """Récupère toutes les positions atteignables dans un coût donné"""

        reachable = set()
        visited = set()
        queue = [(0.0, start)]  # (cost, position)

        while queue:
            current_cost, current_pos = heapq.heappop(queue)

            if current_pos in visited:
                continue

            visited.add(current_pos)

            if current_cost <= max_cost:
                reachable.add(current_pos)

                # Explorer voisins
                neighbors = self.get_neighbors_func(*current_pos)
                for neighbor_pos in neighbors:
                    if (neighbor_pos not in visited and
                        self.grid_map.is_walkable(*neighbor_pos)):

                        move_cost = self._calculate_movement_cost(
                            current_pos, neighbor_pos, MovementType.WALK
                        )
                        new_cost = current_cost + move_cost

                        if new_cost <= max_cost:
                            heapq.heappush(queue, (new_cost, neighbor_pos))

        return reachable

    def clear_cache(self):
        """Vide le cache de chemins"""
        self.path_cache.clear()

    def get_pathfinding_stats(self) -> Dict[str, Any]:
        """Statistiques du pathfinding"""
        avg_computation_time = (
            self.total_computation_time / max(self.total_pathfinding_calls, 1)
        )

        cache_hit_rate = (
            self.cache_hits / max(self.total_pathfinding_calls, 1) * 100
        )

        return {
            "total_pathfinding_calls": self.total_pathfinding_calls,
            "total_computation_time": self.total_computation_time,
            "average_computation_time": avg_computation_time,
            "cache_size": len(self.path_cache),
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "grid_size": (self.grid_map.width, self.grid_map.height),
            "dynamic_obstacles": len(self.grid_map.dynamic_obstacles),
            "use_hexagonal_grid": self.use_hexagonal_grid
        }

def create_pathfinding_engine(width: int = 100, height: int = 100) -> PathfindingEngine:
    """Factory function pour créer un PathfindingEngine"""
    grid_map = GridMap(width, height)
    return PathfindingEngine(grid_map)