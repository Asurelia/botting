"""
Algorithmes de pathfinding ultra-optimisés pour DOFUS.
Implémentations haute performance d'A*, Dijkstra, TSP avec cache intelligent.
Optimisations multi-threading et heuristiques spécialisées DOFUS.
"""

import heapq
import numpy as np
import threading
import time
from typing import Dict, List, Tuple, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from functools import lru_cache
import math
import itertools
from abc import ABC, abstractmethod

from .world_map import WorldMapManager, Position, CompactGraph


class PathfindingResult:
    """Résultat optimisé d'un calcul de chemin."""
    
    def __init__(self, path: List[int], total_cost: float, 
                 algorithm_used: str, computation_time: float,
                 nodes_explored: int = 0):
        self.path = path
        self.total_cost = total_cost
        self.algorithm_used = algorithm_used
        self.computation_time = computation_time
        self.nodes_explored = nodes_explored
        self.is_valid = len(path) > 0
    
    def get_positions(self, graph: CompactGraph) -> List[Position]:
        """Convertit le chemin en positions."""
        return [graph.get_node_position(node_idx) for node_idx in self.path]
    
    def get_movement_commands(self, graph: CompactGraph) -> List[str]:
        """Génère les commandes de mouvement optimisées pour DOFUS."""
        if len(self.path) < 2:
            return []
        
        commands = []
        positions = self.get_positions(graph)
        
        for i in range(len(positions) - 1):
            current = positions[i]
            next_pos = positions[i + 1]
            
            # Calcul de la direction optimisée
            dx = next_pos.x - current.x
            dy = next_pos.y - current.y
            
            # Conversion en commande de mouvement DOFUS
            if abs(dx) > abs(dy):
                direction = "RIGHT" if dx > 0 else "LEFT"
            else:
                direction = "DOWN" if dy > 0 else "UP"
            
            # Vérification des téléportations
            distance = abs(dx) + abs(dy)
            if distance > 10:  # Probablement une téléportation
                if graph.is_zaap(self.path[i]):
                    commands.append(f"ZAAP_TO:{next_pos.x},{next_pos.y}")
                else:
                    commands.append(f"TELEPORT_TO:{next_pos.x},{next_pos.y}")
            else:
                commands.append(f"MOVE_{direction}")
        
        return commands
    
    def __str__(self):
        return (f"Path[{len(self.path)} nodes, cost={self.total_cost:.2f}, "
                f"{self.algorithm_used}, {self.computation_time*1000:.2f}ms]")


class HeuristicFunction(ABC):
    """Interface pour les fonctions heuristiques."""
    
    @abstractmethod
    def estimate(self, from_node: int, to_node: int, graph: CompactGraph) -> float:
        pass


class ManhattanHeuristic(HeuristicFunction):
    """Heuristique de Manhattan optimisée pour DOFUS."""
    
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def estimate(self, from_node: int, to_node: int, graph: CompactGraph) -> float:
        pos1 = graph.get_node_position(from_node)
        pos2 = graph.get_node_position(to_node)
        return self.weight * pos1.distance_manhattan(pos2)


class EuclideanHeuristic(HeuristicFunction):
    """Heuristique euclidienne pour précision maximale."""
    
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def estimate(self, from_node: int, to_node: int, graph: CompactGraph) -> float:
        pos1 = graph.get_node_position(from_node)
        pos2 = graph.get_node_position(to_node)
        return self.weight * math.sqrt(pos1.distance_euclidean_squared(pos2))


class DOFUSHeuristic(HeuristicFunction):
    """Heuristique spécialisée DOFUS avec bonus zaap et malus terrain."""
    
    def __init__(self, zaap_bonus: float = 0.5, terrain_penalty: float = 1.2):
        self.zaap_bonus = zaap_bonus
        self.terrain_penalty = terrain_penalty
        self.manhattan = ManhattanHeuristic()
    
    def estimate(self, from_node: int, to_node: int, graph: CompactGraph) -> float:
        base_cost = self.manhattan.estimate(from_node, to_node, graph)
        
        # Bonus si destination est un zaap
        if graph.is_zaap(to_node):
            base_cost *= self.zaap_bonus
        
        # Pénalité terrain difficile
        if graph.node_costs[to_node] > 1:
            base_cost *= self.terrain_penalty
        
        return base_cost


@dataclass
class PathfindingConfig:
    """Configuration optimisée pour les algorithmes de pathfinding."""
    max_nodes_to_explore: int = 10000
    time_limit_seconds: float = 5.0
    use_bidirectional: bool = True
    heuristic_weight: float = 1.0
    cache_results: bool = True
    parallel_search: bool = False
    beam_width: int = 100  # Pour beam search


class IntelligentCache:
    """Cache intelligent avec LRU et invalidation sélective."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[Tuple[int, int], Tuple[PathfindingResult, float]] = {}
        self.access_count: Dict[Tuple[int, int], int] = defaultdict(int)
        self.lock = threading.RLock()
        
        # Statistiques
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, start: int, end: int) -> Optional[PathfindingResult]:
        """Récupère un chemin du cache avec vérification TTL."""
        with self.lock:
            key = (start, end)
            
            if key not in self.cache:
                self.misses += 1
                return None
            
            result, timestamp = self.cache[key]
            
            # Vérification TTL
            if time.time() - timestamp > self.ttl_seconds:
                del self.cache[key]
                self.misses += 1
                return None
            
            self.access_count[key] += 1
            self.hits += 1
            return result
    
    def put(self, start: int, end: int, result: PathfindingResult):
        """Stocke un chemin dans le cache avec éviction LRU."""
        with self.lock:
            key = (start, end)
            
            # Éviction si nécessaire
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = (result, time.time())
            self.access_count[key] = 1
    
    def _evict_lru(self):
        """Éviction LRU du cache."""
        if not self.cache:
            return
        
        # Trouve la clé la moins récemment utilisée
        lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
        
        del self.cache[lru_key]
        del self.access_count[lru_key]
        self.evictions += 1
    
    def invalidate_area(self, center_node: int, radius: int, graph: CompactGraph):
        """Invalide les chemins passant par une zone modifiée."""
        center_pos = graph.get_node_position(center_node)
        keys_to_remove = []
        
        with self.lock:
            for key in self.cache:
                result, _ = self.cache[key]
                
                # Vérifie si le chemin passe par la zone invalidée
                for node_idx in result.path:
                    node_pos = graph.get_node_position(node_idx)
                    if center_pos.distance_manhattan(node_pos) <= radius:
                        keys_to_remove.append(key)
                        break
            
            for key in keys_to_remove:
                del self.cache[key]
                if key in self.access_count:
                    del self.access_count[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiques du cache."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / max(1, total_requests)
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions
        }


class UltraPathfinder:
    """Pathfinder ultra-optimisé avec algorithmes multiples et cache intelligent."""
    
    def __init__(self, graph: CompactGraph, config: PathfindingConfig = None):
        self.graph = graph
        self.config = config or PathfindingConfig()
        
        # Heuristiques disponibles
        self.heuristics = {
            'manhattan': ManhattanHeuristic(self.config.heuristic_weight),
            'euclidean': EuclideanHeuristic(self.config.heuristic_weight),
            'dofus': DOFUSHeuristic()
        }
        
        # Cache intelligent
        self.cache = IntelligentCache()
        
        # Pool de threads pour recherches parallèles
        self.thread_pool = ThreadPoolExecutor(max_workers=4) if self.config.parallel_search else None
        
        # Métriques de performance
        self.metrics = {
            'total_calls': 0,
            'cache_hits': 0,
            'algorithm_usage': defaultdict(int),
            'avg_computation_time': 0.0,
            'total_nodes_explored': 0
        }
        
        # Lock pour thread-safety
        self.lock = threading.RLock()
    
    def find_path(self, start: int, end: int, algorithm: str = 'auto', 
                  heuristic: str = 'dofus') -> PathfindingResult:
        """Interface principale pour trouver un chemin optimal."""
        start_time = time.time()
        
        with self.lock:
            self.metrics['total_calls'] += 1
        
        # Vérification du cache
        if self.config.cache_results:
            cached_result = self.cache.get(start, end)
            if cached_result:
                with self.lock:
                    self.metrics['cache_hits'] += 1
                return cached_result
        
        # Validation des paramètres
        if not self._validate_nodes(start, end):
            return PathfindingResult([], float('inf'), 'invalid', 0)
        
        # Sélection automatique de l'algorithme
        if algorithm == 'auto':
            algorithm = self._select_optimal_algorithm(start, end)
        
        # Calcul du chemin
        result = self._compute_path(start, end, algorithm, heuristic)
        
        # Mise à jour des métriques
        computation_time = time.time() - start_time
        result.computation_time = computation_time
        
        with self.lock:
            self.metrics['algorithm_usage'][algorithm] += 1
            self.metrics['total_nodes_explored'] += result.nodes_explored
            
            # Moyenne mobile du temps de calcul
            total_calls = self.metrics['total_calls']
            prev_avg = self.metrics['avg_computation_time']
            self.metrics['avg_computation_time'] = (prev_avg * (total_calls - 1) + computation_time) / total_calls
        
        # Cache du résultat
        if self.config.cache_results and result.is_valid:
            self.cache.put(start, end, result)
        
        return result
    
    def _validate_nodes(self, start: int, end: int) -> bool:
        """Valide que les nodes de départ et arrivée sont valides."""
        return (0 <= start < self.graph.node_count and 
                0 <= end < self.graph.node_count and
                self.graph.is_walkable(start) and 
                self.graph.is_walkable(end))
    
    def _select_optimal_algorithm(self, start: int, end: int) -> str:
        """Sélectionne automatiquement l'algorithme optimal."""
        start_pos = self.graph.get_node_position(start)
        end_pos = self.graph.get_node_position(end)
        distance = start_pos.distance_manhattan(end_pos)
        
        # Critères de sélection optimisés
        if distance < 20:
            return 'astar'  # A* pour courtes distances
        elif distance < 100:
            return 'bidirectional_astar'  # Bidirectionnel pour distances moyennes
        elif self.graph.is_zaap(start) or self.graph.is_zaap(end):
            return 'dijkstra_zaap'  # Dijkstra optimisé zaap pour longues distances
        else:
            return 'beam_search'  # Beam search pour très longues distances
    
    def _compute_path(self, start: int, end: int, algorithm: str, heuristic: str) -> PathfindingResult:
        """Calcule le chemin avec l'algorithme spécifié."""
        heuristic_func = self.heuristics.get(heuristic, self.heuristics['dofus'])
        
        if algorithm == 'astar':
            return self._astar(start, end, heuristic_func)
        elif algorithm == 'dijkstra':
            return self._dijkstra(start, end)
        elif algorithm == 'dijkstra_zaap':
            return self._dijkstra_zaap_optimized(start, end)
        elif algorithm == 'bidirectional_astar':
            return self._bidirectional_astar(start, end, heuristic_func)
        elif algorithm == 'beam_search':
            return self._beam_search(start, end, heuristic_func)
        else:
            return self._astar(start, end, heuristic_func)
    
    def _astar(self, start: int, end: int, heuristic: HeuristicFunction) -> PathfindingResult:
        """A* ultra-optimisé avec optimisations DOFUS."""
        open_set = [(0, start, [start], 0)]  # (f_score, node, path, g_score)
        visited = set()
        nodes_explored = 0
        
        while open_set and nodes_explored < self.config.max_nodes_to_explore:
            f_score, current, path, g_score = heapq.heappop(open_set)
            
            if current in visited:
                continue
            
            visited.add(current)
            nodes_explored += 1
            
            if current == end:
                return PathfindingResult(path, g_score, 'astar', 0, nodes_explored)
            
            # Exploration des voisins
            for neighbor, edge_cost in self.graph.get_neighbors(current):
                if neighbor in visited or not self.graph.is_walkable(neighbor):
                    continue
                
                new_g_score = g_score + edge_cost
                h_score = heuristic.estimate(neighbor, end, self.graph)
                f_score = new_g_score + h_score
                new_path = path + [neighbor]
                
                heapq.heappush(open_set, (f_score, neighbor, new_path, new_g_score))
        
        return PathfindingResult([], float('inf'), 'astar', 0, nodes_explored)
    
    def _dijkstra(self, start: int, end: int) -> PathfindingResult:
        """Dijkstra optimisé avec early stopping."""
        distances = {start: 0}
        previous = {}
        pq = [(0, start)]
        visited = set()
        nodes_explored = 0
        
        while pq and nodes_explored < self.config.max_nodes_to_explore:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            nodes_explored += 1
            
            if current == end:
                # Reconstruction du chemin
                path = []
                while current in previous:
                    path.append(current)
                    current = previous[current]
                path.append(start)
                path.reverse()
                
                return PathfindingResult(path, distances[end], 'dijkstra', 0, nodes_explored)
            
            for neighbor, edge_cost in self.graph.get_neighbors(current):
                if neighbor in visited or not self.graph.is_walkable(neighbor):
                    continue
                
                new_dist = current_dist + edge_cost
                
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
        
        return PathfindingResult([], float('inf'), 'dijkstra', 0, nodes_explored)
    
    def _dijkstra_zaap_optimized(self, start: int, end: int) -> PathfindingResult:
        """Dijkstra optimisé pour les téléportations zaap."""
        # Phase 1: Chemin normal jusqu'au zaap le plus proche
        zaap_nodes = self.graph.get_zaap_nodes()
        
        if not zaap_nodes:
            return self._dijkstra(start, end)
        
        best_path = []
        best_cost = float('inf')
        nodes_explored = 0
        
        # Test de chaque zaap comme point de transition
        for zaap in zaap_nodes:
            if zaap == start or zaap == end:
                continue
            
            # Chemin start -> zaap
            path_to_zaap = self._dijkstra(start, zaap)
            if not path_to_zaap.is_valid:
                continue
            
            # Chemin zaap -> end (téléportation + marche)
            path_from_zaap = self._dijkstra(zaap, end)
            if not path_from_zaap.is_valid:
                continue
            
            # Coût total avec téléportation
            total_cost = path_to_zaap.total_cost + 1.0 + path_from_zaap.total_cost
            nodes_explored += path_to_zaap.nodes_explored + path_from_zaap.nodes_explored
            
            if total_cost < best_cost:
                best_cost = total_cost
                # Fusion des chemins
                best_path = path_to_zaap.path + path_from_zaap.path[1:]  # Éviter duplication du zaap
        
        if best_path:
            return PathfindingResult(best_path, best_cost, 'dijkstra_zaap', 0, nodes_explored)
        
        # Fallback sur dijkstra classique
        return self._dijkstra(start, end)
    
    def _bidirectional_astar(self, start: int, end: int, heuristic: HeuristicFunction) -> PathfindingResult:
        """A* bidirectionnel pour distances moyennes."""
        # Recherche depuis le début
        open_start = [(0, start, [start], 0)]
        visited_start = {}
        
        # Recherche depuis la fin
        open_end = [(0, end, [end], 0)]
        visited_end = {}
        
        nodes_explored = 0
        
        while (open_start or open_end) and nodes_explored < self.config.max_nodes_to_explore:
            # Alternance entre les deux recherches
            if open_start and (not open_end or len(open_start) <= len(open_end)):
                f_score, current, path, g_score = heapq.heappop(open_start)
                visited_dict = visited_start
                other_visited = visited_end
                direction = "forward"
            else:
                f_score, current, path, g_score = heapq.heappop(open_end)
                visited_dict = visited_end
                other_visited = visited_start
                direction = "backward"
            
            if current in visited_dict:
                continue
            
            visited_dict[current] = (path, g_score)
            nodes_explored += 1
            
            # Vérification de la rencontre
            if current in other_visited:
                other_path, other_g_score = other_visited[current]
                
                if direction == "forward":
                    final_path = path + other_path[1:][::-1]
                    total_cost = g_score + other_g_score
                else:
                    final_path = other_path + path[1:][::-1]
                    total_cost = other_g_score + g_score
                
                return PathfindingResult(final_path, total_cost, 'bidirectional_astar', 0, nodes_explored)
            
            # Expansion des voisins
            target = end if direction == "forward" else start
            open_set = open_start if direction == "forward" else open_end
            
            for neighbor, edge_cost in self.graph.get_neighbors(current):
                if neighbor in visited_dict or not self.graph.is_walkable(neighbor):
                    continue
                
                new_g_score = g_score + edge_cost
                h_score = heuristic.estimate(neighbor, target, self.graph)
                f_score = new_g_score + h_score
                new_path = path + [neighbor]
                
                heapq.heappush(open_set, (f_score, neighbor, new_path, new_g_score))
        
        return PathfindingResult([], float('inf'), 'bidirectional_astar', 0, nodes_explored)
    
    def _beam_search(self, start: int, end: int, heuristic: HeuristicFunction) -> PathfindingResult:
        """Beam search pour très longues distances avec limitation mémoire."""
        beam_width = self.config.beam_width
        current_level = [(0, start, [start], 0)]  # (f_score, node, path, g_score)
        visited = set()
        nodes_explored = 0
        
        while current_level and nodes_explored < self.config.max_nodes_to_explore:
            next_level = []
            
            # Traitement du niveau actuel
            for f_score, current, path, g_score in current_level:
                if current in visited:
                    continue
                
                visited.add(current)
                nodes_explored += 1
                
                if current == end:
                    return PathfindingResult(path, g_score, 'beam_search', 0, nodes_explored)
                
                # Expansion des voisins
                for neighbor, edge_cost in self.graph.get_neighbors(current):
                    if neighbor in visited or not self.graph.is_walkable(neighbor):
                        continue
                    
                    new_g_score = g_score + edge_cost
                    h_score = heuristic.estimate(neighbor, end, self.graph)
                    f_score = new_g_score + h_score
                    new_path = path + [neighbor]
                    
                    next_level.append((f_score, neighbor, new_path, new_g_score))
            
            # Limitation à beam_width meilleurs candidats
            next_level.sort(key=lambda x: x[0])
            current_level = next_level[:beam_width]
        
        return PathfindingResult([], float('inf'), 'beam_search', 0, nodes_explored)
    
    def find_multi_destination_path(self, start: int, destinations: List[int], 
                                   algorithm: str = 'tsp_greedy') -> PathfindingResult:
        """Trouve le chemin optimal visitant plusieurs destinations (TSP)."""
        if not destinations:
            return PathfindingResult([], 0, algorithm, 0)
        
        if algorithm == 'tsp_greedy':
            return self._tsp_greedy(start, destinations)
        elif algorithm == 'tsp_2opt':
            return self._tsp_2opt(start, destinations)
        else:
            return self._tsp_greedy(start, destinations)
    
    def _tsp_greedy(self, start: int, destinations: List[int]) -> PathfindingResult:
        """TSP glouton optimisé pour DOFUS."""
        if not destinations:
            return PathfindingResult([start], 0, 'tsp_greedy', 0)
        
        unvisited = set(destinations)
        current = start
        path = [start]
        total_cost = 0
        total_nodes_explored = 0
        
        while unvisited:
            # Trouve la destination la plus proche
            best_dest = None
            best_cost = float('inf')
            best_path = []
            
            for dest in unvisited:
                result = self.find_path(current, dest, algorithm='astar')
                if result.is_valid and result.total_cost < best_cost:
                    best_cost = result.total_cost
                    best_dest = dest
                    best_path = result.path[1:]  # Éviter duplication du node actuel
                    total_nodes_explored += result.nodes_explored
            
            if best_dest is None:
                break  # Aucun chemin trouvé
            
            # Ajout à la solution
            path.extend(best_path)
            total_cost += best_cost
            current = best_dest
            unvisited.remove(best_dest)
        
        return PathfindingResult(path, total_cost, 'tsp_greedy', 0, total_nodes_explored)
    
    def _tsp_2opt(self, start: int, destinations: List[int]) -> PathfindingResult:
        """TSP avec optimisation 2-opt pour améliorer la solution gloutonne."""
        # Commencer avec la solution gloutonne
        greedy_result = self._tsp_greedy(start, destinations)
        
        if not greedy_result.is_valid or len(destinations) < 3:
            return greedy_result
        
        # TODO: Implémentation 2-opt pour amélioration itérative
        # Pour l'instant, retourne la solution gloutonne
        return greedy_result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques complètes de performance."""
        cache_stats = self.cache.get_stats()
        
        return {
            'pathfinder': self.metrics.copy(),
            'cache': cache_stats,
            'config': {
                'max_nodes_to_explore': self.config.max_nodes_to_explore,
                'time_limit': self.config.time_limit_seconds,
                'bidirectional': self.config.use_bidirectional,
                'parallel': self.config.parallel_search
            }
        }
    
    def clear_cache(self):
        """Vide le cache pour libérer la mémoire."""
        self.cache = IntelligentCache()
    
    def __del__(self):
        """Nettoyage des ressources."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)


def benchmark_pathfinding():
    """Benchmark complet des algorithmes de pathfinding."""
    print("=== Benchmark UltraPathfinder ===")
    
    # Import et initialisation de la carte
    from .world_map import WorldMapManager
    
    world_map = WorldMapManager()
    world_map.initialize_dofus_world()
    
    pathfinder = UltraPathfinder(world_map.graph)
    
    # Tests de performance
    test_cases = [
        ("Courte distance", 0, 50),
        ("Distance moyenne", 0, 500),
        ("Longue distance", 0, 1000),
    ]
    
    for test_name, start, end in test_cases:
        if end >= world_map.graph.node_count:
            continue
        
        print(f"\n{test_name} (nodes {start} -> {end}):")
        
        for algorithm in ['astar', 'dijkstra', 'bidirectional_astar', 'beam_search']:
            start_time = time.time()
            result = pathfinder.find_path(start, end, algorithm=algorithm)
            duration = time.time() - start_time
            
            if result.is_valid:
                print(f"  {algorithm:20}: {duration*1000:6.2f}ms, "
                      f"cost={result.total_cost:6.1f}, "
                      f"nodes_explored={result.nodes_explored:4d}")
            else:
                print(f"  {algorithm:20}: FAILED")
    
    # Test TSP
    print(f"\nTest TSP (5 destinations):")
    destinations = list(range(1, min(6, world_map.graph.node_count)))
    tsp_result = pathfinder.find_multi_destination_path(0, destinations)
    print(f"  TSP: {len(tsp_result.path)} nodes, cost={tsp_result.total_cost:.1f}")
    
    # Statistiques finales
    stats = pathfinder.get_performance_stats()
    print(f"\nStatistiques: {stats}")
    
    return pathfinder


if __name__ == "__main__":
    pathfinder = benchmark_pathfinding()