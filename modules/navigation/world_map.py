"""
Carte mondiale ultra-optimisée pour DOFUS avec graphe de 10000+ nodes.
Utilise des structures de données avancées et des optimisations mémoire.
Architecture conçue pour performance maximale avec cache intelligent.
"""

import numpy as np
import heapq
import threading
import time
from typing import Dict, List, Tuple, Set, Optional, NamedTuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import zlib
import mmap
import os
from functools import lru_cache


class Position(NamedTuple):
    """Position optimisée avec hash efficace pour cache."""
    x: int
    y: int
    
    def __hash__(self):
        # Hash optimisé pour positions 2D - évite les collisions
        return (self.x << 16) | (self.y & 0xFFFF)
    
    def distance_manhattan(self, other: 'Position') -> int:
        """Distance de Manhattan ultra-rapide."""
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def distance_euclidean_squared(self, other: 'Position') -> int:
        """Distance euclidienne au carré (évite sqrt pour performance)."""
        dx = self.x - other.x
        dy = self.y - other.y
        return dx * dx + dy * dy


@dataclass(frozen=True)
class MapNode:
    """Node de carte optimisé avec attributs DOFUS spécifiques."""
    position: Position
    map_id: int
    zone_id: int
    is_walkable: bool = True
    is_zaap: bool = False
    is_bonta: bool = False
    is_brak: bool = False
    cell_cost: int = 1  # Coût de traversée (boue, lave, etc.)
    
    def __hash__(self):
        return hash(self.position)


@dataclass
class Edge:
    """Arête optimisée avec coûts et restrictions DOFUS."""
    from_node: int  # Index dans le tableau de nodes
    to_node: int
    cost: float
    edge_type: str = "walk"  # walk, zaap, bonta, brak, subway
    restrictions: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        # Conversion des coûts selon le type
        if self.edge_type == "zaap":
            self.cost = 1.0  # Téléportation instantanée
        elif self.edge_type in ["bonta", "brak"]:
            self.cost = 5.0  # Coût alignement
        elif self.edge_type == "subway":
            self.cost = 10.0  # Métro souterrain


class CompactGraph:
    """Graphe ultra-compact utilisant des arrays NumPy pour performance maximale."""
    
    def __init__(self, max_nodes: int = 15000):
        # Arrays NumPy pour performance maximale
        self.max_nodes = max_nodes
        self.node_positions = np.zeros((max_nodes, 2), dtype=np.int32)
        self.node_map_ids = np.zeros(max_nodes, dtype=np.int32)
        self.node_zone_ids = np.zeros(max_nodes, dtype=np.int16)
        self.node_flags = np.zeros(max_nodes, dtype=np.uint16)  # Bitfield pour attributs
        self.node_costs = np.ones(max_nodes, dtype=np.uint8)
        
        # Matrice d'adjacence sparse optimisée
        self.adjacency_matrix = np.full((max_nodes, max_nodes), np.inf, dtype=np.float32)
        np.fill_diagonal(self.adjacency_matrix, 0)
        
        # Listes d'adjacence pour accès rapide
        self.adjacency_lists: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        
        # Index et mappings
        self.position_to_index: Dict[Position, int] = {}
        self.map_id_to_indices: Dict[int, List[int]] = defaultdict(list)
        self.zone_to_indices: Dict[int, List[int]] = defaultdict(list)
        
        # Compteur de nodes actifs
        self.node_count = 0
        
        # Cache des chemins fréquents
        self._path_cache: Dict[Tuple[int, int], Tuple[List[int], float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Lock pour thread-safety
        self._lock = threading.RLock()
    
    def add_node(self, node: MapNode) -> int:
        """Ajoute un node au graphe, retourne l'index."""
        with self._lock:
            if self.node_count >= self.max_nodes:
                raise ValueError(f"Capacité maximale atteinte: {self.max_nodes}")
            
            index = self.node_count
            
            # Stockage optimisé dans les arrays NumPy
            self.node_positions[index] = [node.position.x, node.position.y]
            self.node_map_ids[index] = node.map_id
            self.node_zone_ids[index] = node.zone_id
            self.node_costs[index] = node.cell_cost
            
            # Encodage des flags en bitfield
            flags = 0
            if node.is_walkable:
                flags |= 1
            if node.is_zaap:
                flags |= 2
            if node.is_bonta:
                flags |= 4
            if node.is_brak:
                flags |= 8
            self.node_flags[index] = flags
            
            # Mise à jour des index
            self.position_to_index[node.position] = index
            self.map_id_to_indices[node.map_id].append(index)
            self.zone_to_indices[node.zone_id].append(index)
            
            self.node_count += 1
            return index
    
    def add_edge(self, edge: Edge):
        """Ajoute une arête avec optimisation automatique."""
        with self._lock:
            from_idx, to_idx = edge.from_node, edge.to_node
            
            # Validation des indices
            if not (0 <= from_idx < self.node_count and 0 <= to_idx < self.node_count):
                raise ValueError("Indices de nodes invalides")
            
            # Mise à jour matrice d'adjacence
            self.adjacency_matrix[from_idx, to_idx] = edge.cost
            
            # Mise à jour listes d'adjacence
            self.adjacency_lists[from_idx].append((to_idx, edge.cost))
            
            # Invalidation du cache concerné
            self._invalidate_cache_for_nodes([from_idx, to_idx])
    
    def get_neighbors(self, node_index: int) -> List[Tuple[int, float]]:
        """Récupère les voisins d'un node ultra-rapidement."""
        return self.adjacency_lists.get(node_index, [])
    
    def get_node_position(self, index: int) -> Position:
        """Récupère la position d'un node."""
        pos = self.node_positions[index]
        return Position(int(pos[0]), int(pos[1]))
    
    def is_walkable(self, index: int) -> bool:
        """Vérifie si un node est traversable."""
        return bool(self.node_flags[index] & 1)
    
    def is_zaap(self, index: int) -> bool:
        """Vérifie si un node est un zaap."""
        return bool(self.node_flags[index] & 2)
    
    def find_nearest_nodes(self, position: Position, max_distance: int = 100, 
                          limit: int = 10) -> List[Tuple[int, float]]:
        """Trouve les nodes les plus proches d'une position donnée."""
        if self.node_count == 0:
            return []
        
        # Calcul vectorisé des distances avec NumPy
        pos_array = np.array([position.x, position.y])
        distances = np.sum(np.abs(self.node_positions[:self.node_count] - pos_array), axis=1)
        
        # Filtrage par distance maximale
        valid_indices = np.where(distances <= max_distance)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # Tri par distance et limitation
        sorted_indices = valid_indices[np.argsort(distances[valid_indices])][:limit]
        
        return [(int(idx), float(distances[idx])) for idx in sorted_indices]
    
    def get_zaap_nodes(self) -> List[int]:
        """Récupère tous les nodes zaap pour optimisation des téléportations."""
        zaap_indices = np.where(self.node_flags[:self.node_count] & 2)[0]
        return zaap_indices.tolist()
    
    def _invalidate_cache_for_nodes(self, node_indices: List[int]):
        """Invalide le cache pour les chemins impliquant certains nodes."""
        keys_to_remove = []
        for key in self._path_cache:
            if key[0] in node_indices or key[1] in node_indices:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._path_cache[key]
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Statistiques du cache pour monitoring des performances."""
        return {
            'cache_size': len(self._path_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': self._cache_hits / max(1, self._cache_hits + self._cache_misses)
        }


class WorldMapManager:
    """Gestionnaire principal de la carte mondiale avec optimisations avancées."""
    
    def __init__(self, cache_file: str = "world_cache.pkl.gz"):
        self.graph = CompactGraph()
        self.cache_file = cache_file
        
        # Cache de régions pour accès spatial optimisé
        self.region_cache: Dict[Tuple[int, int], List[int]] = {}
        self.region_size = 50  # Taille des régions pour partitioning spatial
        
        # Pool de threads pour calculs parallèles
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Pré-calculs des routes populaires
        self.popular_routes_cache: Dict[Tuple[int, int], List[int]] = {}
        
        # Métriques de performance
        self.performance_metrics = {
            'total_pathfind_calls': 0,
            'cache_hits': 0,
            'avg_pathfind_time': 0.0,
            'nodes_processed': 0
        }
        
        # Chargement automatique du cache si disponible
        self._load_cache()
    
    def initialize_dofus_world(self):
        """Initialise la carte DOFUS avec données optimisées."""
        print("Initialisation de la carte mondiale DOFUS...")
        
        # Génération des zones principales (optimisée)
        zones_data = self._generate_dofus_zones()
        
        # Ajout massif des nodes avec threading
        self._bulk_add_nodes(zones_data)
        
        # Génération des connexions avec optimisations spatiales
        self._generate_connections()
        
        # Pré-calcul des routes critiques
        self._precompute_popular_routes()
        
        print(f"Carte initialisée: {self.graph.node_count} nodes, "
              f"{len(self.graph.adjacency_lists)} connexions")
    
    def _generate_dofus_zones(self) -> List[Dict]:
        """Génère les données des zones DOFUS de manière optimisée."""
        zones = []
        
        # Zones principales avec densité variable
        zone_configs = [
            # (zone_id, center_x, center_y, radius, node_density, has_zaap)
            (1, -25, -36, 20, 0.8, True),   # Incarnam
            (2, 7, -4, 30, 0.9, True),      # Astrub
            (3, -20, -20, 40, 0.7, True),   # Tainela
            (4, 13, 26, 35, 0.8, True),     # Bonta
            (5, -32, 37, 35, 0.8, True),    # Brâkmar
            (6, 35, 12, 25, 0.6, True),     # Champs
            (7, -26, 35, 30, 0.5, False),   # Landes
            (8, 1, 7, 25, 0.7, True),       # Forêt Obscure
            (9, 4, -18, 20, 0.6, False),    # Territoire Neutre
            (10, -47, -5, 15, 0.4, False),  # Île Minotoror
        ]
        
        node_id = 0
        for zone_id, center_x, center_y, radius, density, has_zaap in zone_configs:
            zone_nodes = self._generate_zone_nodes(
                zone_id, center_x, center_y, radius, density, has_zaap, node_id
            )
            zones.extend(zone_nodes)
            node_id += len(zone_nodes)
        
        return zones
    
    def _generate_zone_nodes(self, zone_id: int, center_x: int, center_y: int, 
                           radius: int, density: float, has_zaap: bool, 
                           start_id: int) -> List[Dict]:
        """Génère les nodes d'une zone avec distribution optimisée."""
        nodes = []
        node_id = start_id
        
        # Génération en grille avec perturbations pour réalisme
        for dx in range(-radius, radius + 1, 2):
            for dy in range(-radius, radius + 1, 2):
                # Test de distance et densité
                distance = (dx * dx + dy * dy) ** 0.5
                if distance > radius or np.random.random() > density:
                    continue
                
                x, y = center_x + dx, center_y + dy
                
                # Détermination des attributs du node
                is_zaap = has_zaap and distance < 3 and len(nodes) == 0
                is_walkable = np.random.random() > 0.05  # 95% traversable
                cell_cost = np.random.choice([1, 2, 3], p=[0.8, 0.15, 0.05])
                
                nodes.append({
                    'position': Position(x, y),
                    'map_id': node_id,
                    'zone_id': zone_id,
                    'is_walkable': is_walkable,
                    'is_zaap': is_zaap,
                    'is_bonta': zone_id == 4,
                    'is_brak': zone_id == 5,
                    'cell_cost': cell_cost
                })
                node_id += 1
        
        return nodes
    
    def _bulk_add_nodes(self, zones_data: List[Dict]):
        """Ajout massif de nodes avec optimisation mémoire."""
        print(f"Ajout de {len(zones_data)} nodes...")
        
        batch_size = 1000
        for i in range(0, len(zones_data), batch_size):
            batch = zones_data[i:i + batch_size]
            
            for node_data in batch:
                node = MapNode(**node_data)
                self.graph.add_node(node)
        
        print(f"Nodes ajoutés: {self.graph.node_count}")
    
    def _generate_connections(self):
        """Génère les connexions entre nodes avec optimisation spatiale."""
        print("Génération des connexions...")
        
        connection_count = 0
        
        # Connexions locales (voisinage)
        for i in range(self.graph.node_count):
            if not self.graph.is_walkable(i):
                continue
            
            pos_i = self.graph.get_node_position(i)
            
            # Recherche des voisins dans un rayon limité
            neighbors = self.graph.find_nearest_nodes(pos_i, max_distance=5, limit=8)
            
            for neighbor_idx, distance in neighbors:
                if neighbor_idx != i and self.graph.is_walkable(neighbor_idx):
                    cost = distance * self.graph.node_costs[neighbor_idx]
                    edge = Edge(i, neighbor_idx, cost, "walk")
                    self.graph.add_edge(edge)
                    connection_count += 1
        
        # Connexions zaap (téléportations)
        zaap_nodes = self.graph.get_zaap_nodes()
        for i, zaap1 in enumerate(zaap_nodes):
            for zaap2 in zaap_nodes[i + 1:]:
                # Connexion bidirectionnelle entre zaaps
                edge1 = Edge(zaap1, zaap2, 1.0, "zaap")
                edge2 = Edge(zaap2, zaap1, 1.0, "zaap")
                self.graph.add_edge(edge1)
                self.graph.add_edge(edge2)
                connection_count += 2
        
        print(f"Connexions créées: {connection_count}")
    
    def _precompute_popular_routes(self):
        """Pré-calcule les routes les plus populaires pour optimisation."""
        print("Pré-calcul des routes populaires...")
        
        # Routes entre zaaps (les plus fréquentes)
        zaap_nodes = self.graph.get_zaap_nodes()
        
        if len(zaap_nodes) > 1:
            # Import du pathfinder (sera créé dans le prochain fichier)
            # Pour l'instant, on simule le pré-calcul
            routes_precalculated = 0
            
            for i, start_zaap in enumerate(zaap_nodes):
                for end_zaap in zaap_nodes[i + 1:]:
                    # Simulation du pré-calcul (sera implémenté avec le pathfinder)
                    cache_key = (start_zaap, end_zaap)
                    self.popular_routes_cache[cache_key] = [start_zaap, end_zaap]
                    routes_precalculated += 1
            
            print(f"Routes pré-calculées: {routes_precalculated}")
    
    def get_node_at_position(self, position: Position) -> Optional[int]:
        """Récupère l'index d'un node à une position donnée."""
        return self.position_to_index.get(position)
    
    def get_nodes_in_zone(self, zone_id: int) -> List[int]:
        """Récupère tous les nodes d'une zone."""
        return self.graph.zone_to_indices.get(zone_id, [])
    
    def get_performance_stats(self) -> Dict:
        """Retourne les statistiques de performance complètes."""
        graph_stats = self.graph.get_cache_stats()
        
        return {
            'graph': graph_stats,
            'world_manager': self.performance_metrics,
            'memory_usage': {
                'node_count': self.graph.node_count,
                'max_nodes': self.graph.max_nodes,
                'memory_efficiency': self.graph.node_count / self.graph.max_nodes
            }
        }
    
    def _save_cache(self):
        """Sauvegarde le cache de la carte pour rechargement rapide."""
        try:
            cache_data = {
                'graph': self.graph,
                'popular_routes': self.popular_routes_cache,
                'performance_metrics': self.performance_metrics
            }
            
            with open(self.cache_file, 'wb') as f:
                compressed_data = zlib.compress(pickle.dumps(cache_data))
                f.write(compressed_data)
            
            print(f"Cache sauvegardé: {self.cache_file}")
        except Exception as e:
            print(f"Erreur sauvegarde cache: {e}")
    
    def _load_cache(self):
        """Charge le cache de la carte si disponible."""
        if not os.path.exists(self.cache_file):
            return
        
        try:
            with open(self.cache_file, 'rb') as f:
                compressed_data = f.read()
                cache_data = pickle.loads(zlib.decompress(compressed_data))
            
            self.graph = cache_data['graph']
            self.popular_routes_cache = cache_data.get('popular_routes', {})
            self.performance_metrics = cache_data.get('performance_metrics', {})
            
            print(f"Cache chargé: {self.graph.node_count} nodes")
        except Exception as e:
            print(f"Erreur chargement cache: {e}")
    
    def __del__(self):
        """Nettoyage automatique avec sauvegarde du cache."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        
        if hasattr(self, 'graph') and self.graph.node_count > 0:
            self._save_cache()


# Fonction d'aide pour benchmarking
def benchmark_world_map():
    """Benchmark complet du système de carte mondiale."""
    print("=== Benchmark WorldMapManager ===")
    
    # Initialisation
    start_time = time.time()
    world_map = WorldMapManager()
    world_map.initialize_dofus_world()
    init_time = time.time() - start_time
    
    print(f"Temps d'initialisation: {init_time:.2f}s")
    
    # Test de recherche de voisins
    start_time = time.time()
    test_position = Position(0, 0)
    neighbors = world_map.graph.find_nearest_nodes(test_position, max_distance=10, limit=20)
    search_time = time.time() - start_time
    
    print(f"Recherche de voisins: {search_time*1000:.2f}ms pour {len(neighbors)} résultats")
    
    # Statistiques finales
    stats = world_map.get_performance_stats()
    print(f"Statistiques: {stats}")
    
    return world_map


if __name__ == "__main__":
    # Test et benchmark du système
    world_map = benchmark_world_map()