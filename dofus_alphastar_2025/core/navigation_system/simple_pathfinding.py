"""
Simple Pathfinding - A* pour navigation dans Dofus
Version simplifiée et efficace
"""

import heapq
import logging
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


@dataclass
class PathNode:
    """Noeud pour pathfinding"""
    x: int
    y: int
    g_cost: float = 0  # Coût depuis le départ
    h_cost: float = 0  # Heuristique vers l'arrivée
    parent: Optional['PathNode'] = None
    
    @property
    def f_cost(self) -> float:
        """Coût total (g + h)"""
        return self.g_cost + self.h_cost
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))


class SimplePathfinder:
    """Pathfinding A* simplifié pour Dofus"""
    
    def __init__(self):
        # Obstacles (sera enrichi dynamiquement)
        self.obstacles: Set[Tuple[int, int]] = set()
        
        # Cache des chemins
        self.path_cache = {}
        
        # Stats
        self.stats = {
            'paths_computed': 0,
            'cache_hits': 0,
            'avg_path_length': 0
        }
        
        logger.info("SimplePathfinder initialisé")
    
    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacles: Optional[Set[Tuple[int, int]]] = None,
        max_iterations: int = 1000
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Trouve un chemin de start à goal avec A*
        
        Args:
            start: Position départ (x, y)
            goal: Position arrivée (x, y)
            obstacles: Set des positions bloquées
            max_iterations: Limite d'itérations
        
        Returns:
            Liste de positions ou None si pas de chemin
        """
        # Check cache
        cache_key = (start, goal)
        if cache_key in self.path_cache:
            self.stats['cache_hits'] += 1
            return self.path_cache[cache_key]
        
        # Mettre à jour obstacles
        if obstacles:
            self.obstacles = obstacles
        
        # Start = Goal
        if start == goal:
            return [start]
        
        # Init
        start_node = PathNode(start[0], start[1])
        goal_node = PathNode(goal[0], goal[1])
        
        open_set = [start_node]
        closed_set: Set[PathNode] = set()
        
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # Noeud avec le plus petit f_cost
            current = heapq.heappop(open_set)
            
            # Arrivée?
            if current == goal_node:
                path = self._reconstruct_path(current)
                self.path_cache[cache_key] = path
                self.stats['paths_computed'] += 1
                self.stats['avg_path_length'] = (
                    (self.stats['avg_path_length'] * (self.stats['paths_computed'] - 1) + len(path))
                    / self.stats['paths_computed']
                )
                return path
            
            closed_set.add(current)
            
            # Voisins
            for neighbor_pos in self._get_neighbors(current.x, current.y):
                # Obstacle?
                if neighbor_pos in self.obstacles:
                    continue
                
                neighbor = PathNode(neighbor_pos[0], neighbor_pos[1])
                
                # Déjà visité?
                if neighbor in closed_set:
                    continue
                
                # Coût pour aller au voisin
                tentative_g = current.g_cost + self._distance(
                    (current.x, current.y),
                    neighbor_pos
                )
                
                # Nouveau ou meilleur?
                existing = next((n for n in open_set if n == neighbor), None)
                
                if existing is None or tentative_g < existing.g_cost:
                    neighbor.g_cost = tentative_g
                    neighbor.h_cost = self._heuristic(neighbor_pos, goal)
                    neighbor.parent = current
                    
                    if existing:
                        open_set.remove(existing)
                    
                    heapq.heappush(open_set, neighbor)
        
        # Pas de chemin trouvé
        logger.warning(f"Pas de chemin trouvé de {start} à {goal} (iterations: {iterations})")
        return None
    
    def _get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Retourne les voisins (8 directions)"""
        neighbors = [
            (x + dx, y + dy)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            if (dx != 0 or dy != 0)
        ]
        return neighbors
    
    def _distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Distance euclidienne"""
        return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Heuristique (distance euclidienne)"""
        return self._distance(pos, goal)
    
    def _reconstruct_path(self, node: PathNode) -> List[Tuple[int, int]]:
        """Reconstruit le chemin depuis le noeud final"""
        path = []
        current = node
        
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        
        return list(reversed(path))
    
    def add_obstacle(self, x: int, y: int):
        """Ajoute un obstacle"""
        self.obstacles.add((x, y))
    
    def remove_obstacle(self, x: int, y: int):
        """Retire un obstacle"""
        self.obstacles.discard((x, y))
    
    def clear_obstacles(self):
        """Efface tous les obstacles"""
        self.obstacles.clear()
    
    def clear_cache(self):
        """Efface le cache des chemins"""
        self.path_cache.clear()
    
    def get_stats(self) -> dict:
        """Retourne les statistiques"""
        return self.stats.copy()


def create_pathfinder() -> SimplePathfinder:
    """Factory function"""
    return SimplePathfinder()
