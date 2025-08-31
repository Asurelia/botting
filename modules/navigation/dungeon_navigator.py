"""
Navigateur spécialisé pour les donjons DOFUS avec optimisations avancées.
Gestion des salles, groupes de monstres, pièges et mécaniques spécifiques.
Intelligence artificielle pour optimisation des parcours de farming.
"""

import numpy as np
import heapq
import threading
import time
from typing import Dict, List, Tuple, Optional, Set, NamedTuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum, auto
import json
import math
from concurrent.futures import ThreadPoolExecutor

from .world_map import Position, CompactGraph
from .pathfinding import UltraPathfinder, PathfindingResult, PathfindingConfig


class CellType(Enum):
    """Types de cellules dans un donjon."""
    EMPTY = auto()
    WALL = auto()
    MONSTER_GROUP = auto()
    TRAP = auto()
    CHEST = auto()
    DOOR = auto()
    STAIRS_UP = auto()
    STAIRS_DOWN = auto()
    BOSS_ROOM = auto()
    SAFE_ZONE = auto()


class MonsterGroup(NamedTuple):
    """Groupe de monstres avec métadonnées."""
    position: Position
    monster_ids: List[int]
    level_range: Tuple[int, int]
    difficulty: int  # 1-5
    respawn_time: int  # en minutes
    last_killed: float  # timestamp
    experience_value: int
    drop_value: int  # estimation kamas


@dataclass
class DungeonRoom:
    """Salle de donjon optimisée avec pathfinding local."""
    room_id: int
    name: str
    level: int  # Étage du donjon
    dimensions: Tuple[int, int]  # largeur, hauteur
    cells: np.ndarray  # Grille des types de cellules
    monster_groups: List[MonsterGroup] = field(default_factory=list)
    traps: List[Position] = field(default_factory=list)
    doors: List[Position] = field(default_factory=list)
    safe_zones: List[Position] = field(default_factory=list)
    
    def __post_init__(self):
        if self.cells is None:
            self.cells = np.full(self.dimensions, CellType.EMPTY.value, dtype=np.int8)
    
    def is_walkable(self, position: Position) -> bool:
        """Vérifie si une position est traversable."""
        if not (0 <= position.x < self.dimensions[0] and 0 <= position.y < self.dimensions[1]):
            return False
        
        cell_type = CellType(self.cells[position.y, position.x])
        return cell_type not in [CellType.WALL, CellType.TRAP]
    
    def get_monster_at(self, position: Position) -> Optional[MonsterGroup]:
        """Récupère le groupe de monstres à une position."""
        for monster_group in self.monster_groups:
            if monster_group.position == position:
                return monster_group
        return None
    
    def update_monster_respawn(self, position: Position):
        """Met à jour le temps de respawn d'un groupe de monstres."""
        for i, monster_group in enumerate(self.monster_groups):
            if monster_group.position == position:
                # Création d'un nouveau MonsterGroup avec timestamp mis à jour
                updated_group = MonsterGroup(
                    position=monster_group.position,
                    monster_ids=monster_group.monster_ids,
                    level_range=monster_group.level_range,
                    difficulty=monster_group.difficulty,
                    respawn_time=monster_group.respawn_time,
                    last_killed=time.time(),
                    experience_value=monster_group.experience_value,
                    drop_value=monster_group.drop_value
                )
                self.monster_groups[i] = updated_group
                break
    
    def get_available_monsters(self, current_time: float) -> List[MonsterGroup]:
        """Récupère les groupes de monstres disponibles (respawnés)."""
        available = []
        for monster_group in self.monster_groups:
            time_since_kill = current_time - monster_group.last_killed
            if time_since_kill >= monster_group.respawn_time * 60:  # Conversion en secondes
                available.append(monster_group)
        return available
    
    def calculate_efficiency_map(self, player_level: int) -> np.ndarray:
        """Calcule une carte d'efficacité pour le farming."""
        efficiency_map = np.zeros(self.dimensions, dtype=np.float32)
        
        for monster_group in self.monster_groups:
            x, y = monster_group.position.x, monster_group.position.y
            
            # Facteur de niveau (éviter les groupes trop faibles/forts)
            level_factor = self._calculate_level_factor(player_level, monster_group.level_range)
            
            # Facteur de valeur (xp + drops)
            value_factor = (monster_group.experience_value + monster_group.drop_value) / 1000
            
            # Facteur de disponibilité
            availability_factor = 1.0 if monster_group.last_killed == 0 else 0.5
            
            # Score d'efficacité total
            efficiency = level_factor * value_factor * availability_factor / max(1, monster_group.difficulty)
            
            if 0 <= y < self.dimensions[1] and 0 <= x < self.dimensions[0]:
                efficiency_map[y, x] = efficiency
        
        return efficiency_map
    
    def _calculate_level_factor(self, player_level: int, monster_range: Tuple[int, int]) -> float:
        """Calcule le facteur d'efficacité basé sur la différence de niveau."""
        min_level, max_level = monster_range
        avg_level = (min_level + max_level) / 2
        level_diff = abs(player_level - avg_level)
        
        # Courbe d'efficacité optimisée
        if level_diff <= 5:
            return 1.0
        elif level_diff <= 15:
            return 0.8
        elif level_diff <= 30:
            return 0.5
        else:
            return 0.2


@dataclass
class DungeonLayout:
    """Layout complet d'un donjon avec toutes ses salles."""
    dungeon_id: int
    name: str
    recommended_level: Tuple[int, int]
    rooms: Dict[int, DungeonRoom] = field(default_factory=dict)
    connections: Dict[int, List[int]] = field(default_factory=dict)  # room_id -> [connected_room_ids]
    boss_room_id: Optional[int] = None
    
    def add_room(self, room: DungeonRoom):
        """Ajoute une salle au donjon."""
        self.rooms[room.room_id] = room
    
    def connect_rooms(self, room1_id: int, room2_id: int):
        """Connecte deux salles."""
        if room1_id not in self.connections:
            self.connections[room1_id] = []
        if room2_id not in self.connections:
            self.connections[room2_id] = []
        
        self.connections[room1_id].append(room2_id)
        self.connections[room2_id].append(room1_id)
    
    def get_path_between_rooms(self, start_room: int, end_room: int) -> List[int]:
        """Trouve le chemin entre deux salles (BFS simple)."""
        if start_room == end_room:
            return [start_room]
        
        visited = set()
        queue = deque([(start_room, [start_room])])
        
        while queue:
            current_room, path = queue.popleft()
            
            if current_room in visited:
                continue
            visited.add(current_room)
            
            for connected_room in self.connections.get(current_room, []):
                if connected_room == end_room:
                    return path + [connected_room]
                
                if connected_room not in visited:
                    queue.append((connected_room, path + [connected_room]))
        
        return []  # Aucun chemin trouvé


class FarmingStrategy(Enum):
    """Stratégies de farming optimisées."""
    EXPERIENCE_MAX = auto()  # Maximiser l'expérience
    KAMAS_MAX = auto()       # Maximiser les kamas
    BALANCED = auto()        # Équilibré xp/kamas
    SPEED_RUN = auto()       # Vitesse maximale
    BOSS_RUSH = auto()       # Direct au boss


@dataclass
class FarmingConfig:
    """Configuration pour l'optimisation de farming."""
    strategy: FarmingStrategy = FarmingStrategy.BALANCED
    player_level: int = 100
    time_limit_minutes: int = 60
    min_group_difficulty: int = 1
    max_group_difficulty: int = 5
    avoid_traps: bool = True
    prioritize_chests: bool = True
    rest_in_safe_zones: bool = True


class DungeonPathOptimizer:
    """Optimiseur de chemins spécialisé pour donjons."""
    
    def __init__(self, dungeon: DungeonLayout, config: FarmingConfig):
        self.dungeon = dungeon
        self.config = config
        
        # Cache des chemins calculés
        self._path_cache: Dict[Tuple[int, Position, Position], PathfindingResult] = {}
        
        # Métriques de farming
        self.farming_metrics = {
            'total_monsters_killed': 0,
            'total_experience_gained': 0,
            'total_kamas_gained': 0,
            'time_spent_seconds': 0,
            'rooms_cleared': set(),
            'efficiency_score': 0.0
        }
    
    def find_optimal_farming_route(self, start_room: int, start_position: Position,
                                 time_limit: Optional[int] = None) -> List[Tuple[int, Position, str]]:
        """
        Trouve la route de farming optimale.
        Returns: List de (room_id, position, action) où action = 'move', 'fight', 'chest', etc.
        """
        if time_limit is None:
            time_limit = self.config.time_limit_minutes * 60
        
        current_room = start_room
        current_position = start_position
        route = [(current_room, current_position, 'start')]
        
        start_time = time.time()
        total_value = 0
        
        # Algorithme glouton optimisé avec look-ahead
        while time.time() - start_time < time_limit:
            # Trouve la prochaine cible optimale
            next_target = self._find_next_optimal_target(
                current_room, current_position, route
            )
            
            if not next_target:
                break  # Plus de cibles intéressantes
            
            target_room, target_position, target_type, estimated_value = next_target
            
            # Calcul du chemin vers la cible
            path_to_target = self._calculate_path_to_target(
                current_room, current_position, target_room, target_position
            )
            
            if not path_to_target:
                continue  # Impossible d'atteindre la cible
            
            # Ajout du chemin à la route
            route.extend(path_to_target)
            
            # Action à la cible
            action = self._determine_action(target_type)
            route.append((target_room, target_position, action))
            
            # Mise à jour de la position actuelle
            current_room = target_room
            current_position = target_position
            total_value += estimated_value
            
            # Mise à jour des métriques
            self._update_farming_metrics(target_type, estimated_value)
        
        # Optimisation finale de la route
        optimized_route = self._optimize_route(route)
        
        return optimized_route
    
    def _find_next_optimal_target(self, current_room: int, current_position: Position,
                                current_route: List[Tuple]) -> Optional[Tuple[int, Position, str, float]]:
        """Trouve la prochaine cible optimale selon la stratégie."""
        candidates = []
        
        # Recherche dans toutes les salles accessibles
        for room_id, room in self.dungeon.rooms.items():
            # Groupes de monstres disponibles
            available_monsters = room.get_available_monsters(time.time())
            
            for monster_group in available_monsters:
                if (self.config.min_group_difficulty <= monster_group.difficulty <= 
                    self.config.max_group_difficulty):
                    
                    value = self._calculate_target_value(monster_group, room_id, current_room)
                    candidates.append((room_id, monster_group.position, 'monster', value))
            
            # Coffres (si priorité activée)
            if self.config.prioritize_chests:
                for pos in room.safe_zones:  # Approximation: safe zones contiennent souvent des coffres
                    value = self._calculate_chest_value(pos, room_id, current_room)
                    candidates.append((room_id, pos, 'chest', value))
        
        # Boss en dernier si stratégie appropriée
        if self.config.strategy == FarmingStrategy.BOSS_RUSH and self.dungeon.boss_room_id:
            boss_room = self.dungeon.rooms[self.dungeon.boss_room_id]
            if boss_room.monster_groups:
                boss_pos = boss_room.monster_groups[0].position
                value = self._calculate_boss_value(self.dungeon.boss_room_id, current_room)
                candidates.append((self.dungeon.boss_room_id, boss_pos, 'boss', value))
        
        # Sélection de la meilleure cible
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x[3], reverse=True)  # Tri par valeur décroissante
        return candidates[0]
    
    def _calculate_target_value(self, monster_group: MonsterGroup, target_room: int, current_room: int) -> float:
        """Calcule la valeur d'une cible monstre selon la stratégie."""
        base_value = 0
        
        if self.config.strategy == FarmingStrategy.EXPERIENCE_MAX:
            base_value = monster_group.experience_value
        elif self.config.strategy == FarmingStrategy.KAMAS_MAX:
            base_value = monster_group.drop_value
        else:  # BALANCED
            base_value = (monster_group.experience_value + monster_group.drop_value) / 2
        
        # Facteur de distance (pénalise les cibles éloignées)
        distance_penalty = len(self.dungeon.get_path_between_rooms(current_room, target_room))
        
        # Facteur de difficulté
        difficulty_factor = 1.0 / max(1, monster_group.difficulty - 2)
        
        # Facteur de niveau
        level_factor = self._calculate_level_efficiency(monster_group.level_range)
        
        return base_value * difficulty_factor * level_factor / max(1, distance_penalty)
    
    def _calculate_chest_value(self, position: Position, target_room: int, current_room: int) -> float:
        """Calcule la valeur d'un coffre."""
        base_value = 500  # Valeur estimée d'un coffre
        distance_penalty = len(self.dungeon.get_path_between_rooms(current_room, target_room))
        return base_value / max(1, distance_penalty)
    
    def _calculate_boss_value(self, boss_room: int, current_room: int) -> float:
        """Calcule la valeur du boss selon la stratégie."""
        if self.config.strategy == FarmingStrategy.BOSS_RUSH:
            return 10000  # Valeur très élevée pour prioriser le boss
        
        base_value = 2000  # Valeur estimée du boss
        distance_penalty = len(self.dungeon.get_path_between_rooms(current_room, boss_room))
        return base_value / max(1, distance_penalty)
    
    def _calculate_level_efficiency(self, monster_level_range: Tuple[int, int]) -> float:
        """Calcule l'efficacité basée sur la différence de niveau."""
        avg_monster_level = sum(monster_level_range) / 2
        level_diff = abs(self.config.player_level - avg_monster_level)
        
        if level_diff <= 10:
            return 1.0
        elif level_diff <= 25:
            return 0.7
        elif level_diff <= 50:
            return 0.4
        else:
            return 0.1
    
    def _calculate_path_to_target(self, start_room: int, start_pos: Position,
                                target_room: int, target_pos: Position) -> List[Tuple[int, Position, str]]:
        """Calcule le chemin détaillé vers une cible."""
        path = []
        
        # Chemin entre salles
        room_path = self.dungeon.get_path_between_rooms(start_room, target_room)
        
        if not room_path:
            return []
        
        current_pos = start_pos
        
        for i in range(len(room_path) - 1):
            from_room = room_path[i]
            to_room = room_path[i + 1]
            
            # Trouve la porte entre les salles
            door_pos = self._find_door_between_rooms(from_room, to_room)
            
            if door_pos:
                # Chemin dans la salle actuelle jusqu'à la porte
                intra_room_path = self._find_path_in_room(from_room, current_pos, door_pos)
                path.extend([(from_room, pos, 'move') for pos in intra_room_path[1:]])
                path.append((from_room, door_pos, 'use_door'))
                current_pos = door_pos  # Position après passage de porte
        
        # Chemin final dans la salle cible
        final_room = room_path[-1]
        if final_room == target_room:
            final_path = self._find_path_in_room(final_room, current_pos, target_pos)
            path.extend([(final_room, pos, 'move') for pos in final_path[1:]])
        
        return path
    
    def _find_door_between_rooms(self, room1: int, room2: int) -> Optional[Position]:
        """Trouve la position de la porte entre deux salles."""
        # Implémentation simplifiée - en réalité, les portes seraient pré-définies
        if room1 in self.dungeon.rooms and self.dungeon.rooms[room1].doors:
            return self.dungeon.rooms[room1].doors[0]
        return Position(0, 0)  # Position par défaut
    
    def _find_path_in_room(self, room_id: int, start: Position, end: Position) -> List[Position]:
        """Trouve le chemin optimal dans une salle (A* local)."""
        if room_id not in self.dungeon.rooms:
            return []
        
        room = self.dungeon.rooms[room_id]
        
        # A* simplifié pour pathfinding intra-salle
        open_set = [(0, start, [start], 0)]
        visited = set()
        
        while open_set:
            f_score, current, path, g_score = heapq.heappop(open_set)
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == end:
                return path
            
            # Exploration des voisins (8 directions)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    neighbor = Position(current.x + dx, current.y + dy)
                    
                    if (neighbor in visited or 
                        not room.is_walkable(neighbor) or
                        (self.config.avoid_traps and 
                         CellType(room.cells[neighbor.y, neighbor.x]) == CellType.TRAP)):
                        continue
                    
                    new_g_score = g_score + (1.4 if abs(dx) + abs(dy) == 2 else 1.0)  # Diagonale
                    h_score = abs(neighbor.x - end.x) + abs(neighbor.y - end.y)
                    f_score = new_g_score + h_score
                    
                    heapq.heappush(open_set, (f_score, neighbor, path + [neighbor], new_g_score))
        
        return []  # Aucun chemin trouvé
    
    def _determine_action(self, target_type: str) -> str:
        """Détermine l'action à effectuer selon le type de cible."""
        action_map = {
            'monster': 'fight',
            'chest': 'open_chest',
            'boss': 'fight_boss',
            'trap': 'disarm_trap'
        }
        return action_map.get(target_type, 'interact')
    
    def _update_farming_metrics(self, target_type: str, value: float):
        """Met à jour les métriques de farming."""
        if target_type == 'monster':
            self.farming_metrics['total_monsters_killed'] += 1
            self.farming_metrics['total_experience_gained'] += value
        elif target_type in ['chest', 'boss']:
            self.farming_metrics['total_kamas_gained'] += value
    
    def _optimize_route(self, route: List[Tuple[int, Position, str]]) -> List[Tuple[int, Position, str]]:
        """Optimise la route finale en supprimant les mouvements redondants."""
        if len(route) <= 2:
            return route
        
        optimized = [route[0]]
        
        for i in range(1, len(route)):
            current = route[i]
            previous = optimized[-1]
            
            # Supprime les mouvements consécutifs dans la même salle vers la même position
            if (current[0] == previous[0] and  # Même salle
                current[1] == previous[1] and  # Même position
                current[2] == 'move' and previous[2] == 'move'):
                continue
            
            optimized.append(current)
        
        return optimized
    
    def get_farming_statistics(self) -> Dict[str, Any]:
        """Récupère les statistiques de farming."""
        if self.farming_metrics['time_spent_seconds'] > 0:
            monsters_per_minute = (self.farming_metrics['total_monsters_killed'] * 60 / 
                                 self.farming_metrics['time_spent_seconds'])
            xp_per_minute = (self.farming_metrics['total_experience_gained'] * 60 / 
                           self.farming_metrics['time_spent_seconds'])
            kamas_per_minute = (self.farming_metrics['total_kamas_gained'] * 60 / 
                              self.farming_metrics['time_spent_seconds'])
        else:
            monsters_per_minute = xp_per_minute = kamas_per_minute = 0
        
        return {
            'farming_metrics': self.farming_metrics.copy(),
            'efficiency': {
                'monsters_per_minute': monsters_per_minute,
                'xp_per_minute': xp_per_minute,
                'kamas_per_minute': kamas_per_minute,
                'rooms_cleared': len(self.farming_metrics['rooms_cleared'])
            }
        }


class DungeonNavigator:
    """Navigateur principal pour donjons avec IA intégrée."""
    
    def __init__(self):
        self.known_dungeons: Dict[int, DungeonLayout] = {}
        self.current_dungeon: Optional[DungeonLayout] = None
        self.path_optimizers: Dict[int, DungeonPathOptimizer] = {}
        
        # Configuration par défaut
        self.default_config = FarmingConfig()
        
        # Métriques globales
        self.global_metrics = {
            'dungeons_explored': set(),
            'total_farming_time': 0,
            'most_efficient_dungeon': None,
            'best_efficiency_score': 0.0
        }
        
        # Cache des layouts de donjons
        self._layout_cache: Dict[str, DungeonLayout] = {}
    
    def load_dungeon_layout(self, dungeon_id: int, layout_file: Optional[str] = None) -> DungeonLayout:
        """Charge le layout d'un donjon depuis un fichier ou génère aléatoirement."""
        if layout_file and layout_file in self._layout_cache:
            return self._layout_cache[layout_file]
        
        # Génération de donjon de démonstration
        dungeon_layout = self._generate_sample_dungeon(dungeon_id)
        
        self.known_dungeons[dungeon_id] = dungeon_layout
        self.current_dungeon = dungeon_layout
        
        # Création de l'optimiseur associé
        self.path_optimizers[dungeon_id] = DungeonPathOptimizer(dungeon_layout, self.default_config)
        
        return dungeon_layout
    
    def _generate_sample_dungeon(self, dungeon_id: int) -> DungeonLayout:
        """Génère un donjon d'exemple pour démonstration."""
        dungeon = DungeonLayout(
            dungeon_id=dungeon_id,
            name=f"Donjon d'exemple {dungeon_id}",
            recommended_level=(50, 80)
        )
        
        # Génération de 5 salles connectées
        for room_id in range(5):
            room = DungeonRoom(
                room_id=room_id,
                name=f"Salle {room_id + 1}",
                level=room_id,
                dimensions=(20, 15)
            )
            
            # Génération des groupes de monstres
            for _ in range(np.random.randint(3, 8)):
                x, y = np.random.randint(1, 19), np.random.randint(1, 14)
                monster_group = MonsterGroup(
                    position=Position(x, y),
                    monster_ids=[np.random.randint(1, 100)],
                    level_range=(40 + room_id * 10, 60 + room_id * 10),
                    difficulty=np.random.randint(1, 4),
                    respawn_time=np.random.randint(5, 15),
                    last_killed=0,
                    experience_value=np.random.randint(100, 500),
                    drop_value=np.random.randint(50, 200)
                )
                room.monster_groups.append(monster_group)
            
            # Ajout de zones sûres
            room.safe_zones = [Position(0, 0), Position(19, 14)]
            
            # Ajout de portes
            if room_id < 4:  # Toutes les salles sauf la dernière ont une porte vers la suivante
                room.doors = [Position(19, 7)]  # Porte à droite
            
            dungeon.add_room(room)
        
        # Connexion des salles
        for i in range(4):
            dungeon.connect_rooms(i, i + 1)
        
        # La dernière salle est le boss
        dungeon.boss_room_id = 4
        
        return dungeon
    
    def optimize_farming_session(self, dungeon_id: int, config: FarmingConfig = None,
                                start_room: int = 0) -> List[Tuple[int, Position, str]]:
        """Optimise une session de farming complète."""
        if dungeon_id not in self.known_dungeons:
            self.load_dungeon_layout(dungeon_id)
        
        if config:
            self.path_optimizers[dungeon_id].config = config
        
        optimizer = self.path_optimizers[dungeon_id]
        dungeon = self.known_dungeons[dungeon_id]
        
        # Position de départ dans la première salle
        start_position = dungeon.rooms[start_room].safe_zones[0] if dungeon.rooms[start_room].safe_zones else Position(1, 1)
        
        # Optimisation de la route
        optimal_route = optimizer.find_optimal_farming_route(start_room, start_position)
        
        # Mise à jour des métriques globales
        self.global_metrics['dungeons_explored'].add(dungeon_id)
        
        stats = optimizer.get_farming_statistics()
        efficiency_score = stats['efficiency']['xp_per_minute'] + stats['efficiency']['kamas_per_minute']
        
        if efficiency_score > self.global_metrics['best_efficiency_score']:
            self.global_metrics['best_efficiency_score'] = efficiency_score
            self.global_metrics['most_efficient_dungeon'] = dungeon_id
        
        return optimal_route
    
    def analyze_dungeon_efficiency(self, dungeon_id: int, player_level: int) -> Dict[str, Any]:
        """Analyse l'efficacité d'un donjon pour un niveau de joueur donné."""
        if dungeon_id not in self.known_dungeons:
            self.load_dungeon_layout(dungeon_id)
        
        dungeon = self.known_dungeons[dungeon_id]
        analysis = {
            'dungeon_id': dungeon_id,
            'recommended_level': dungeon.recommended_level,
            'player_level': player_level,
            'total_rooms': len(dungeon.rooms),
            'total_monster_groups': 0,
            'avg_experience_per_group': 0,
            'avg_drop_value_per_group': 0,
            'difficulty_distribution': defaultdict(int),
            'level_compatibility': 0.0
        }
        
        total_xp = total_drops = 0
        
        for room in dungeon.rooms.values():
            analysis['total_monster_groups'] += len(room.monster_groups)
            
            for monster_group in room.monster_groups:
                total_xp += monster_group.experience_value
                total_drops += monster_group.drop_value
                analysis['difficulty_distribution'][monster_group.difficulty] += 1
        
        if analysis['total_monster_groups'] > 0:
            analysis['avg_experience_per_group'] = total_xp / analysis['total_monster_groups']
            analysis['avg_drop_value_per_group'] = total_drops / analysis['total_monster_groups']
        
        # Compatibilité de niveau
        recommended_avg = sum(dungeon.recommended_level) / 2
        level_diff = abs(player_level - recommended_avg)
        
        if level_diff <= 10:
            analysis['level_compatibility'] = 1.0
        elif level_diff <= 25:
            analysis['level_compatibility'] = 0.7
        else:
            analysis['level_compatibility'] = 0.3
        
        return analysis
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Récupère les statistiques globales de navigation."""
        optimizer_stats = {}
        for dungeon_id, optimizer in self.path_optimizers.items():
            optimizer_stats[dungeon_id] = optimizer.get_farming_statistics()
        
        return {
            'global_metrics': self.global_metrics.copy(),
            'per_dungeon_stats': optimizer_stats,
            'known_dungeons': list(self.known_dungeons.keys())
        }


def benchmark_dungeon_navigator():
    """Benchmark du système de navigation de donjon."""
    print("=== Benchmark DungeonNavigator ===")
    
    navigator = DungeonNavigator()
    
    # Test de génération et optimisation de donjons
    for dungeon_id in range(1, 4):
        print(f"\nTest du donjon {dungeon_id}:")
        
        # Chargement du donjon
        start_time = time.time()
        dungeon_layout = navigator.load_dungeon_layout(dungeon_id)
        load_time = time.time() - start_time
        
        print(f"  Chargement: {load_time*1000:.2f}ms")
        print(f"  Salles: {len(dungeon_layout.rooms)}")
        
        total_monsters = sum(len(room.monster_groups) for room in dungeon_layout.rooms.values())
        print(f"  Groupes de monstres: {total_monsters}")
        
        # Analyse d'efficacité
        efficiency = navigator.analyze_dungeon_efficiency(dungeon_id, player_level=75)
        print(f"  Compatibilité niveau 75: {efficiency['level_compatibility']:.2f}")
        print(f"  XP moyenne par groupe: {efficiency['avg_experience_per_group']:.0f}")
        
        # Optimisation de farming
        config = FarmingConfig(strategy=FarmingStrategy.BALANCED, player_level=75, time_limit_minutes=30)
        
        start_time = time.time()
        optimal_route = navigator.optimize_farming_session(dungeon_id, config)
        optimization_time = time.time() - start_time
        
        print(f"  Optimisation: {optimization_time*1000:.2f}ms")
        print(f"  Route optimale: {len(optimal_route)} étapes")
    
    # Statistiques globales
    global_stats = navigator.get_global_statistics()
    print(f"\nStatistiques globales:")
    print(f"  Donjons explorés: {len(global_stats['global_metrics']['dungeons_explored'])}")
    print(f"  Donjon le plus efficace: {global_stats['global_metrics']['most_efficient_dungeon']}")
    
    return navigator


if __name__ == "__main__":
    navigator = benchmark_dungeon_navigator()