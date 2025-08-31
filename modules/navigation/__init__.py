"""
Module de navigation ultra-optimisé pour DOFUS.
Système complet de pathfinding, carte mondiale et navigation de donjon.
Benchmarks intégrés et optimisations multi-threading avancées.
"""

import time
import threading
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import psutil
import numpy as np

from .world_map import WorldMapManager, Position, CompactGraph, benchmark_world_map
from .pathfinding import UltraPathfinder, PathfindingResult, PathfindingConfig, benchmark_pathfinding
from .dungeon_navigator import DungeonNavigator, FarmingConfig, FarmingStrategy, benchmark_dungeon_navigator


class NavigationManager:
    """Gestionnaire principal du système de navigation avec optimisations multi-threading."""
    
    def __init__(self, max_workers: Optional[int] = None):
        # Détection automatique du nombre optimal de workers
        if max_workers is None:
            cpu_count = multiprocessing.cpu_count()
            max_workers = min(cpu_count, 8)  # Limite à 8 pour éviter la sur-utilisation
        
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Composants principaux
        self.world_map: Optional[WorldMapManager] = None
        self.pathfinder: Optional[UltraPathfinder] = None
        self.dungeon_navigator: Optional[DungeonNavigator] = None
        
        # Cache global inter-composants
        self.global_cache: Dict[str, Any] = {}
        
        # Métriques de performance globales
        self.performance_metrics = {
            'initialization_time': 0.0,
            'total_navigation_calls': 0,
            'avg_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'memory_usage_mb': 0.0,
            'cpu_usage_percent': 0.0,
            'thread_pool_utilization': 0.0
        }
        
        # Lock pour thread-safety
        self._lock = threading.RLock()
        
        # Monitoring système
        self.process = psutil.Process()
        self.start_time = time.time()
        
        print(f"NavigationManager initialisé avec {max_workers} workers")
    
    def initialize_full_system(self, world_cache_file: str = "world_cache.pkl.gz") -> float:
        """Initialise tous les composants du système de navigation."""
        start_time = time.time()
        
        print("Initialisation complète du système de navigation...")
        
        # Initialisation en parallèle des composants
        futures = []
        
        # World Map (plus lourd, en priorité)
        future_world = self.thread_pool.submit(self._init_world_map, world_cache_file)
        futures.append(('world_map', future_world))
        
        # Dungeon Navigator (indépendant)
        future_dungeon = self.thread_pool.submit(self._init_dungeon_navigator)
        futures.append(('dungeon_navigator', future_dungeon))
        
        # Attente et récupération des résultats
        for component_name, future in futures:
            try:
                result = future.result(timeout=30)  # Timeout de 30s par composant
                print(f"  {component_name} initialisé: {result}")
            except Exception as e:
                print(f"  Erreur {component_name}: {e}")
        
        # Pathfinder (dépend de world_map)
        if self.world_map and self.world_map.graph.node_count > 0:
            future_pathfinder = self.thread_pool.submit(self._init_pathfinder)
            try:
                result = future_pathfinder.result(timeout=15)
                print(f"  pathfinder initialisé: {result}")
            except Exception as e:
                print(f"  Erreur pathfinder: {e}")
        
        # Calcul du temps d'initialisation
        init_time = time.time() - start_time
        self.performance_metrics['initialization_time'] = init_time
        
        print(f"Système complet initialisé en {init_time:.2f}s")
        return init_time
    
    def _init_world_map(self, cache_file: str) -> str:
        """Initialise la carte mondiale."""
        self.world_map = WorldMapManager(cache_file)
        self.world_map.initialize_dofus_world()
        return f"{self.world_map.graph.node_count} nodes chargés"
    
    def _init_pathfinder(self) -> str:
        """Initialise le pathfinder."""
        config = PathfindingConfig(
            max_nodes_to_explore=15000,
            use_bidirectional=True,
            parallel_search=True
        )
        self.pathfinder = UltraPathfinder(self.world_map.graph, config)
        return "pathfinder configuré"
    
    def _init_dungeon_navigator(self) -> str:
        """Initialise le navigateur de donjon."""
        self.dungeon_navigator = DungeonNavigator()
        return "dungeon navigator prêt"
    
    def find_optimal_path_threaded(self, requests: List[Tuple[Position, Position, str]]) -> List[PathfindingResult]:
        """Traite plusieurs requêtes de pathfinding en parallèle."""
        if not self.pathfinder:
            raise RuntimeError("Pathfinder non initialisé")
        
        start_time = time.time()
        
        # Conversion des positions en indices de nodes
        node_requests = []
        for start_pos, end_pos, algorithm in requests:
            start_nodes = self.world_map.graph.find_nearest_nodes(start_pos, max_distance=5, limit=1)
            end_nodes = self.world_map.graph.find_nearest_nodes(end_pos, max_distance=5, limit=1)
            
            if start_nodes and end_nodes:
                node_requests.append((start_nodes[0][0], end_nodes[0][0], algorithm))
        
        # Soumission des tâches en parallèle
        futures = []
        for start_node, end_node, algorithm in node_requests:
            future = self.thread_pool.submit(
                self.pathfinder.find_path, start_node, end_node, algorithm
            )
            futures.append(future)
        
        # Collecte des résultats
        results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=10)
                results.append(result)
            except Exception as e:
                print(f"Erreur pathfinding parallèle: {e}")
                results.append(PathfindingResult([], float('inf'), 'error', 0))
        
        # Mise à jour des métriques
        processing_time = time.time() - start_time
        with self._lock:
            self.performance_metrics['total_navigation_calls'] += len(requests)
            
            # Moyenne mobile du temps de réponse
            total_calls = self.performance_metrics['total_navigation_calls']
            prev_avg = self.performance_metrics['avg_response_time']
            self.performance_metrics['avg_response_time'] = (
                (prev_avg * (total_calls - len(requests)) + processing_time) / total_calls
            )
        
        return results
    
    def optimize_multiple_dungeons(self, dungeon_configs: List[Tuple[int, FarmingConfig]]) -> Dict[int, Any]:
        """Optimise plusieurs donjons en parallèle."""
        if not self.dungeon_navigator:
            raise RuntimeError("DungeonNavigator non initialisé")
        
        # Soumission des tâches d'optimisation
        futures = {}
        for dungeon_id, config in dungeon_configs:
            future = self.thread_pool.submit(
                self.dungeon_navigator.optimize_farming_session,
                dungeon_id, config
            )
            futures[dungeon_id] = future
        
        # Collecte des résultats
        results = {}
        for dungeon_id, future in futures.items():
            try:
                route = future.result(timeout=30)
                efficiency = self.dungeon_navigator.analyze_dungeon_efficiency(dungeon_id, config.player_level)
                
                results[dungeon_id] = {
                    'route': route,
                    'efficiency': efficiency,
                    'route_length': len(route)
                }
            except Exception as e:
                print(f"Erreur optimisation donjon {dungeon_id}: {e}")
                results[dungeon_id] = {'error': str(e)}
        
        return results
    
    def update_performance_metrics(self):
        """Met à jour les métriques de performance système."""
        with self._lock:
            # Utilisation mémoire
            memory_info = self.process.memory_info()
            self.performance_metrics['memory_usage_mb'] = memory_info.rss / 1024 / 1024
            
            # Utilisation CPU
            self.performance_metrics['cpu_usage_percent'] = self.process.cpu_percent()
            
            # Utilisation du thread pool
            active_threads = threading.active_count() - 1  # -1 pour le thread principal
            self.performance_metrics['thread_pool_utilization'] = active_threads / self.max_workers
            
            # Cache hit rate (moyenne des composants)
            cache_rates = []
            if self.world_map:
                world_stats = self.world_map.get_performance_stats()
                cache_rates.append(world_stats.get('graph', {}).get('hit_rate', 0))
            
            if self.pathfinder:
                pathfinder_stats = self.pathfinder.get_performance_stats()
                cache_rates.append(pathfinder_stats.get('cache', {}).get('hit_rate', 0))
            
            if cache_rates:
                self.performance_metrics['cache_hit_rate'] = sum(cache_rates) / len(cache_rates)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques complètes du système."""
        self.update_performance_metrics()
        
        stats = {
            'system': self.performance_metrics.copy(),
            'uptime_seconds': time.time() - self.start_time,
            'components': {}
        }
        
        # Statistiques des composants
        if self.world_map:
            stats['components']['world_map'] = self.world_map.get_performance_stats()
        
        if self.pathfinder:
            stats['components']['pathfinder'] = self.pathfinder.get_performance_stats()
        
        if self.dungeon_navigator:
            stats['components']['dungeon_navigator'] = self.dungeon_navigator.get_global_statistics()
        
        return stats
    
    def run_comprehensive_benchmark(self, iterations: int = 100) -> Dict[str, Any]:
        """Execute un benchmark complet du système."""
        print(f"=== Benchmark Complet du Système de Navigation ({iterations} itérations) ===")
        
        benchmark_results = {
            'iterations': iterations,
            'world_map': {},
            'pathfinding': {},
            'dungeon_navigation': {},
            'parallel_performance': {},
            'system_resources': {}
        }
        
        # Benchmark World Map
        if self.world_map:
            print("\n1. Benchmark World Map...")
            start_time = time.time()
            
            for _ in range(iterations // 10):  # Moins d'itérations pour les opérations lourdes
                test_pos = Position(np.random.randint(-50, 50), np.random.randint(-50, 50))
                neighbors = self.world_map.graph.find_nearest_nodes(test_pos, limit=10)
            
            world_time = time.time() - start_time
            benchmark_results['world_map'] = {
                'total_time': world_time,
                'avg_time_per_search': world_time / (iterations // 10) * 1000,  # ms
                'nodes_in_graph': self.world_map.graph.node_count
            }
            print(f"   Temps moyen recherche: {benchmark_results['world_map']['avg_time_per_search']:.2f}ms")
        
        # Benchmark Pathfinding
        if self.pathfinder:
            print("\n2. Benchmark Pathfinding...")
            start_time = time.time()
            
            successful_paths = 0
            total_cost = 0
            
            for _ in range(iterations // 5):  # Pathfinding est plus coûteux
                start_node = np.random.randint(0, min(1000, self.world_map.graph.node_count))
                end_node = np.random.randint(0, min(1000, self.world_map.graph.node_count))
                
                result = self.pathfinder.find_path(start_node, end_node)
                if result.is_valid:
                    successful_paths += 1
                    total_cost += result.total_cost
            
            pathfinding_time = time.time() - start_time
            benchmark_results['pathfinding'] = {
                'total_time': pathfinding_time,
                'avg_time_per_path': pathfinding_time / (iterations // 5) * 1000,  # ms
                'success_rate': successful_paths / (iterations // 5),
                'avg_path_cost': total_cost / max(1, successful_paths)
            }
            print(f"   Temps moyen pathfinding: {benchmark_results['pathfinding']['avg_time_per_path']:.2f}ms")
            print(f"   Taux de succès: {benchmark_results['pathfinding']['success_rate']:.2%}")
        
        # Benchmark Navigation Donjon
        if self.dungeon_navigator:
            print("\n3. Benchmark Dungeon Navigation...")
            start_time = time.time()
            
            for dungeon_id in range(1, min(4, iterations // 20)):
                config = FarmingConfig(player_level=np.random.randint(50, 100))
                self.dungeon_navigator.optimize_farming_session(dungeon_id, config)
            
            dungeon_time = time.time() - start_time
            benchmark_results['dungeon_navigation'] = {
                'total_time': dungeon_time,
                'avg_time_per_optimization': dungeon_time / 3 * 1000,  # ms
                'dungeons_tested': 3
            }
            print(f"   Temps moyen optimisation donjon: {benchmark_results['dungeon_navigation']['avg_time_per_optimization']:.2f}ms")
        
        # Benchmark Performance Parallèle
        print("\n4. Benchmark Performance Parallèle...")
        parallel_requests = []
        for _ in range(min(20, self.max_workers * 3)):  # 3 requêtes par worker
            start_pos = Position(np.random.randint(-30, 30), np.random.randint(-30, 30))
            end_pos = Position(np.random.randint(-30, 30), np.random.randint(-30, 30))
            parallel_requests.append((start_pos, end_pos, 'astar'))
        
        start_time = time.time()
        parallel_results = self.find_optimal_path_threaded(parallel_requests)
        parallel_time = time.time() - start_time
        
        successful_parallel = sum(1 for r in parallel_results if r.is_valid)
        
        benchmark_results['parallel_performance'] = {
            'total_requests': len(parallel_requests),
            'successful_requests': successful_parallel,
            'total_time': parallel_time,
            'avg_time_per_request': parallel_time / len(parallel_requests) * 1000,  # ms
            'requests_per_second': len(parallel_requests) / parallel_time,
            'success_rate': successful_parallel / len(parallel_requests)
        }
        print(f"   Requêtes par seconde (parallèle): {benchmark_results['parallel_performance']['requests_per_second']:.1f}")
        print(f"   Temps moyen par requête: {benchmark_results['parallel_performance']['avg_time_per_request']:.2f}ms")
        
        # Ressources Système
        self.update_performance_metrics()
        benchmark_results['system_resources'] = self.performance_metrics.copy()
        
        print(f"\n5. Utilisation des Ressources:")
        print(f"   Mémoire: {self.performance_metrics['memory_usage_mb']:.1f} MB")
        print(f"   CPU: {self.performance_metrics['cpu_usage_percent']:.1f}%")
        print(f"   Utilisation thread pool: {self.performance_metrics['thread_pool_utilization']:.1%}")
        print(f"   Taux de cache hits: {self.performance_metrics['cache_hit_rate']:.1%}")
        
        return benchmark_results
    
    def __del__(self):
        """Nettoyage des ressources."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)


# Fonctions utilitaires pour tests rapides
def quick_pathfinding_test(start_pos: Position, end_pos: Position) -> PathfindingResult:
    """Test rapide de pathfinding entre deux positions."""
    nav_manager = NavigationManager(max_workers=2)
    nav_manager.initialize_full_system()
    
    if nav_manager.pathfinder:
        # Conversion en indices de nodes
        start_nodes = nav_manager.world_map.graph.find_nearest_nodes(start_pos, limit=1)
        end_nodes = nav_manager.world_map.graph.find_nearest_nodes(end_pos, limit=1)
        
        if start_nodes and end_nodes:
            return nav_manager.pathfinder.find_path(start_nodes[0][0], end_nodes[0][0])
    
    return PathfindingResult([], float('inf'), 'error', 0)


def quick_dungeon_test(dungeon_id: int = 1) -> Dict[str, Any]:
    """Test rapide d'optimisation de donjon."""
    nav_manager = NavigationManager(max_workers=2)
    nav_manager._init_dungeon_navigator()
    
    config = FarmingConfig(strategy=FarmingStrategy.BALANCED, player_level=75)
    route = nav_manager.dungeon_navigator.optimize_farming_session(dungeon_id, config)
    efficiency = nav_manager.dungeon_navigator.analyze_dungeon_efficiency(dungeon_id, 75)
    
    return {
        'route_length': len(route),
        'efficiency': efficiency,
        'first_actions': route[:5]  # Premières actions
    }


def run_full_benchmark():
    """Execute un benchmark complet du système."""
    nav_manager = NavigationManager()
    init_time = nav_manager.initialize_full_system()
    
    if init_time > 0:
        results = nav_manager.run_comprehensive_benchmark(iterations=50)
        
        print("\n" + "="*60)
        print("RÉSUMÉ DU BENCHMARK")
        print("="*60)
        
        if 'pathfinding' in results:
            print(f"Pathfinding moyen: {results['pathfinding']['avg_time_per_path']:.2f}ms")
        
        if 'parallel_performance' in results:
            print(f"Performance parallèle: {results['parallel_performance']['requests_per_second']:.1f} req/s")
        
        print(f"Utilisation mémoire: {results['system_resources']['memory_usage_mb']:.1f} MB")
        print(f"Taux de cache hits: {results['system_resources']['cache_hit_rate']:.1%}")
        
        return results
    else:
        print("Échec de l'initialisation du système")
        return None


if __name__ == "__main__":
    # Test complet du système
    results = run_full_benchmark()