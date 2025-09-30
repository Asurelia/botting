"""
DOFUS Maps Database - Knowledge Base Integration
Systeme complet de cartes avec ressources, spawns et navigation
Approche 100% vision - Reconnaissance automatique cartes via templates
"""

import json
import sqlite3
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging
import math

logger = logging.getLogger(__name__)

class MapType(Enum):
    """Types de cartes DOFUS"""
    OVERWORLD = "overworld"
    DUNGEON = "dungeon"
    HOUSE = "house"
    GUILD = "guild"
    TEMPLE = "temple"
    INSTANCE = "instance"

class ResourceType(Enum):
    """Types de ressources collectables"""
    TREE = "tree"
    MINERAL = "mineral"
    PLANT = "plant"
    FISH = "fish"
    CEREAL = "cereal"
    SPECIAL = "special"

class CellType(Enum):
    """Types de cellules sur une carte"""
    WALKABLE = "walkable"
    OBSTACLE = "obstacle"
    INTERACTIVE = "interactive"
    TELEPORTER = "teleporter"
    RESOURCE = "resource"
    SPAWN = "spawn"

@dataclass
class ResourceSpawn:
    """Point de spawn d'une ressource"""
    resource_name: str
    resource_type: ResourceType
    position: Tuple[int, int]  # Position sur la grille
    respawn_time: int  # Minutes
    level_required: int
    rarity: str  # "common", "rare", "epic"
    quantity_min: int = 1
    quantity_max: int = 1

@dataclass
class MonsterSpawn:
    """Point de spawn de monstre"""
    monster_name: str
    position: Tuple[int, int]
    level_min: int
    level_max: int
    spawn_rate: float  # 0.0 - 1.0
    group_size_min: int = 1
    group_size_max: int = 1
    respawn_time: int = 5  # Minutes

@dataclass
class MapTransition:
    """Transition vers une autre carte"""
    direction: str  # "north", "south", "east", "west", "up", "down"
    target_map_id: int
    trigger_position: Tuple[int, int]
    target_position: Tuple[int, int]
    conditions: Optional[str] = None  # Conditions speciales

@dataclass
class DofusMap:
    """Representation complete d'une carte DOFUS"""
    id: int
    name: str
    map_type: MapType
    coordinates: Tuple[int, int]  # Position carte monde (X, Y)

    # Structure de la carte
    width: int = 15
    height: int = 17
    cell_grid: Dict[Tuple[int, int], CellType] = None

    # Contenu de la carte
    resource_spawns: List[ResourceSpawn] = None
    monster_spawns: List[MonsterSpawn] = None
    transitions: List[MapTransition] = None

    # Informations zone
    zone_name: str = ""
    sub_area: str = ""
    recommended_level: int = 1

    # Navigation
    walkable_cells: Set[Tuple[int, int]] = None
    safe_cells: Set[Tuple[int, int]] = None  # Cellules sans monstres

    # Reconnaissance visuelle
    map_template: Optional[str] = None
    landmark_keywords: List[str] = None

    # Metadata
    is_pvp: bool = False
    is_sanctuary: bool = False
    has_zaap: bool = False
    has_houses: bool = False

    def __post_init__(self):
        if self.cell_grid is None:
            self.cell_grid = {}
        if self.resource_spawns is None:
            self.resource_spawns = []
        if self.monster_spawns is None:
            self.monster_spawns = []
        if self.transitions is None:
            self.transitions = []
        if self.walkable_cells is None:
            self.walkable_cells = set()
        if self.safe_cells is None:
            self.safe_cells = set()
        if self.landmark_keywords is None:
            self.landmark_keywords = []

class DofusMapsDatabase:
    """
    Base de donnees complete des cartes DOFUS Unity
    Navigation automatique + optimisation farming
    """

    def __init__(self, db_path: str = "data/dofus_maps.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.maps: Dict[int, DofusMap] = {}
        self.coordinate_maps: Dict[Tuple[int, int], DofusMap] = {}
        self.zone_maps: Dict[str, List[DofusMap]] = {}

        # Cache pour pathfinding
        self.pathfinding_cache: Dict[Tuple[int, int, int, int], List[Tuple[int, int]]] = {}

        self._init_database()
        self._load_maps_data()

        logger.info(f"MapsDatabase initialise: {len(self.maps)} cartes")

    def _init_database(self):
        """Initialise la base de donnees SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS maps (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                map_type TEXT,
                coordinates TEXT,
                zone_name TEXT,
                data JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS map_resources (
                map_id INTEGER,
                resource_name TEXT,
                position TEXT,
                level_required INTEGER,
                FOREIGN KEY (map_id) REFERENCES maps (id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS map_connections (
                from_map_id INTEGER,
                to_map_id INTEGER,
                direction TEXT,
                FOREIGN KEY (from_map_id) REFERENCES maps (id),
                FOREIGN KEY (to_map_id) REFERENCES maps (id)
            )
        ''')

        conn.commit()
        conn.close()

    def _load_maps_data(self):
        """Charge les donnees depuis DB et JSON"""
        json_path = self.db_path.parent / "maps_data.json"
        if json_path.exists():
            self._load_from_json(json_path)
        else:
            self._create_base_maps()

    def _load_from_json(self, json_path: Path):
        """Charge depuis fichier JSON"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for map_data in data.get('maps', []):
                map_obj = self._dict_to_map(map_data)
                self.add_map(map_obj)

        except Exception as e:
            logger.error(f"Erreur chargement maps JSON: {e}")

    def _dict_to_map(self, data: Dict) -> DofusMap:
        """Convertit dictionnaire en DofusMap"""
        # Reconstruction des objets complexes
        resource_spawns = []
        for res_data in data.get('resource_spawns', []):
            resource_spawns.append(ResourceSpawn(**res_data))

        monster_spawns = []
        for mon_data in data.get('monster_spawns', []):
            monster_spawns.append(MonsterSpawn(**mon_data))

        transitions = []
        for trans_data in data.get('transitions', []):
            transitions.append(MapTransition(**trans_data))

        # Conversion des sets depuis listes JSON
        walkable_cells = set(tuple(cell) for cell in data.get('walkable_cells', []))
        safe_cells = set(tuple(cell) for cell in data.get('safe_cells', []))

        # Conversion cell_grid avec tuples comme clés
        cell_grid = {}
        for key_str, value in data.get('cell_grid', {}).items():
            x, y = map(int, key_str.strip('()').split(', '))
            cell_grid[(x, y)] = CellType(value)

        return DofusMap(
            id=data['id'],
            name=data['name'],
            map_type=MapType(data['map_type']),
            coordinates=tuple(data['coordinates']),
            width=data.get('width', 15),
            height=data.get('height', 17),
            cell_grid=cell_grid,
            resource_spawns=resource_spawns,
            monster_spawns=monster_spawns,
            transitions=transitions,
            zone_name=data.get('zone_name', ''),
            sub_area=data.get('sub_area', ''),
            recommended_level=data.get('recommended_level', 1),
            walkable_cells=walkable_cells,
            safe_cells=safe_cells,
            map_template=data.get('map_template'),
            landmark_keywords=data.get('landmark_keywords', []),
            is_pvp=data.get('is_pvp', False),
            is_sanctuary=data.get('is_sanctuary', False),
            has_zaap=data.get('has_zaap', False),
            has_houses=data.get('has_houses', False)
        )

    def _create_base_maps(self):
        """Cree les cartes de base pour test"""
        base_maps = [
            # Incarnam - Zone debutant
            DofusMap(
                id=1, name="Village d'Incarnam", map_type=MapType.OVERWORLD,
                coordinates=(7, -4), zone_name="Incarnam", sub_area="Village",
                recommended_level=1,
                resource_spawns=[
                    ResourceSpawn("Frene", ResourceType.TREE, (5, 8), 10, 1, "common", 1, 3),
                    ResourceSpawn("Fer", ResourceType.MINERAL, (12, 5), 15, 1, "common", 1, 2)
                ],
                monster_spawns=[
                    MonsterSpawn("Bouftou", (3, 12), 1, 5, 0.8, 1, 2, 3),
                    MonsterSpawn("Moskito", (10, 3), 1, 3, 0.6, 1, 1, 2)
                ],
                transitions=[
                    MapTransition("south", 2, (7, 16), (7, 0)),
                    MapTransition("east", 3, (14, 8), (0, 8))
                ],
                walkable_cells={(x, y) for x in range(15) for y in range(17)
                              if not (5 <= x <= 9 and 6 <= y <= 10)},  # Zone batiment
                safe_cells={(x, y) for x in range(15) for y in range(17)
                           if x < 3 or x > 12 or y < 3 or y > 13},
                is_sanctuary=True, has_zaap=True,
                landmark_keywords=["village", "incarnam", "debutant", "maitre"]
            ),

            # Plaine de Cania
            DofusMap(
                id=2, name="Plaine de Cania", map_type=MapType.OVERWORLD,
                coordinates=(7, -5), zone_name="Cania", sub_area="Plaine",
                recommended_level=5,
                resource_spawns=[
                    ResourceSpawn("Ble", ResourceType.CEREAL, (4, 6), 8, 1, "common", 2, 4),
                    ResourceSpawn("Ortie", ResourceType.PLANT, (11, 9), 12, 5, "common", 1, 2)
                ],
                monster_spawns=[
                    MonsterSpawn("Bouftou", (6, 8), 3, 8, 0.9, 1, 3, 4),
                    MonsterSpawn("Pichon", (2, 14), 1, 6, 0.7, 1, 2, 3)
                ],
                transitions=[
                    MapTransition("north", 1, (7, 0), (7, 16)),
                    MapTransition("west", 4, (0, 8), (14, 8))
                ],
                walkable_cells={(x, y) for x in range(15) for y in range(17)},
                safe_cells={(0, 0), (14, 0), (0, 16), (14, 16)},  # Coins seulement
                landmark_keywords=["plaine", "cania", "ble", "champs"]
            )
        ]

        for map_obj in base_maps:
            self.add_map(map_obj)

    def add_map(self, map_obj: DofusMap):
        """Ajoute une carte a la base"""
        self.maps[map_obj.id] = map_obj
        self.coordinate_maps[map_obj.coordinates] = map_obj

        # Index par zone
        if map_obj.zone_name not in self.zone_maps:
            self.zone_maps[map_obj.zone_name] = []
        self.zone_maps[map_obj.zone_name].append(map_obj)

        self._save_map_to_db(map_obj)

    def _save_map_to_db(self, map_obj: DofusMap):
        """Sauvegarde en SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        map_dict = asdict(map_obj)
        # Conversion des enums et sets pour JSON
        map_dict['map_type'] = map_obj.map_type.value

        # Conversion cell_grid avec clés string
        cell_grid_str = {}
        for key, value in map_obj.cell_grid.items():
            cell_grid_str[str(key)] = value.value
        map_dict['cell_grid'] = cell_grid_str

        # Conversion des sets en listes
        map_dict['walkable_cells'] = list(map_obj.walkable_cells)
        map_dict['safe_cells'] = list(map_obj.safe_cells)

        # Conversion enums dans les spawns
        for res_spawn in map_dict['resource_spawns']:
            res_spawn['resource_type'] = res_spawn['resource_type'].value

        cursor.execute('''
            INSERT OR REPLACE INTO maps
            (id, name, map_type, coordinates, zone_name, data)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (map_obj.id, map_obj.name, map_obj.map_type.value,
              str(map_obj.coordinates), map_obj.zone_name,
              json.dumps(map_dict)))

        conn.commit()
        conn.close()

    def get_map(self, map_id: int) -> Optional[DofusMap]:
        """Recupere une carte par ID"""
        return self.maps.get(map_id)

    def get_map_by_coordinates(self, coordinates: Tuple[int, int]) -> Optional[DofusMap]:
        """Recupere une carte par coordonnees"""
        return self.coordinate_maps.get(coordinates)

    def get_maps_by_zone(self, zone_name: str) -> List[DofusMap]:
        """Recupere toutes les cartes d'une zone"""
        return self.zone_maps.get(zone_name, [])

    def get_maps_by_level_range(self, min_level: int, max_level: int) -> List[DofusMap]:
        """Recupere cartes par niveau recommande"""
        return [map_obj for map_obj in self.maps.values()
                if min_level <= map_obj.recommended_level <= max_level]

    def find_resource_maps(self, resource_name: str, min_level: int = 1) -> List[DofusMap]:
        """Trouve les cartes contenant une ressource specifique"""
        resource_maps = []

        for map_obj in self.maps.values():
            for spawn in map_obj.resource_spawns:
                if (spawn.resource_name.lower() == resource_name.lower() and
                    spawn.level_required <= min_level):
                    resource_maps.append(map_obj)
                    break

        return resource_maps

    def find_monster_maps(self, monster_name: str, player_level: int) -> List[DofusMap]:
        """Trouve les cartes contenant un monstre adapte au niveau"""
        monster_maps = []

        for map_obj in self.maps.values():
            for spawn in map_obj.monster_spawns:
                if (spawn.monster_name.lower() == monster_name.lower() and
                    spawn.level_min <= player_level <= spawn.level_max + 10):
                    monster_maps.append(map_obj)
                    break

        return monster_maps

    def calculate_distance(self, map1_id: int, map2_id: int) -> float:
        """Calcule la distance entre deux cartes"""
        map1 = self.get_map(map1_id)
        map2 = self.get_map(map2_id)

        if not map1 or not map2:
            return float('inf')

        x1, y1 = map1.coordinates
        x2, y2 = map2.coordinates

        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def find_path_between_maps(self, start_map_id: int, target_map_id: int) -> List[int]:
        """Trouve le chemin entre deux cartes (BFS)"""
        if start_map_id == target_map_id:
            return [start_map_id]

        # BFS pour trouver le chemin le plus court
        queue = [(start_map_id, [start_map_id])]
        visited = {start_map_id}

        while queue:
            current_map_id, path = queue.pop(0)
            current_map = self.get_map(current_map_id)

            if not current_map:
                continue

            # Verifie toutes les transitions de la carte actuelle
            for transition in current_map.transitions:
                next_map_id = transition.target_map_id

                if next_map_id == target_map_id:
                    return path + [next_map_id]

                if next_map_id not in visited:
                    visited.add(next_map_id)
                    queue.append((next_map_id, path + [next_map_id]))

        return []  # Pas de chemin trouve

    def get_optimal_farming_route(self, resource_name: str, player_level: int,
                                 max_maps: int = 5) -> List[DofusMap]:
        """Trouve une route optimale pour farmer une ressource"""
        resource_maps = self.find_resource_maps(resource_name, player_level)

        if not resource_maps:
            return []

        # Tri par densite de ressources et securite
        def map_score(map_obj):
            resource_count = sum(1 for spawn in map_obj.resource_spawns
                               if spawn.resource_name.lower() == resource_name.lower())
            safety_ratio = len(map_obj.safe_cells) / max(1, len(map_obj.walkable_cells))
            level_penalty = abs(map_obj.recommended_level - player_level) * 0.1

            return resource_count * safety_ratio - level_penalty

        resource_maps.sort(key=map_score, reverse=True)
        return resource_maps[:max_maps]

    def find_safe_path_on_map(self, map_id: int, start_pos: Tuple[int, int],
                             target_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Trouve un chemin sur une carte en evitant les monstres"""
        map_obj = self.get_map(map_id)
        if not map_obj:
            return []

        cache_key = (map_id, start_pos[0], start_pos[1], target_pos[0], target_pos[1])
        if cache_key in self.pathfinding_cache:
            return self.pathfinding_cache[cache_key]

        # A* pathfinding avec evitement des zones dangereuses
        from heapq import heappush, heappop

        def heuristic(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < map_obj.width and 0 <= ny < map_obj.height and
                    (nx, ny) in map_obj.walkable_cells):
                    neighbors.append((nx, ny))
            return neighbors

        def cell_cost(pos):
            if pos in map_obj.safe_cells:
                return 1
            # Penalite pour cellules proches des monstres
            monster_penalty = 0
            for spawn in map_obj.monster_spawns:
                distance = abs(pos[0] - spawn.position[0]) + abs(pos[1] - spawn.position[1])
                if distance <= 2:
                    monster_penalty += 5
            return 1 + monster_penalty

        # A* algorithm
        open_set = [(0, start_pos)]
        came_from = {}
        g_score = {start_pos: 0}
        f_score = {start_pos: heuristic(start_pos, target_pos)}

        while open_set:
            current = heappop(open_set)[1]

            if current == target_pos:
                # Reconstruction du chemin
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_pos)
                path.reverse()

                self.pathfinding_cache[cache_key] = path
                return path

            for neighbor in get_neighbors(current):
                tentative_g_score = g_score[current] + cell_cost(neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, target_pos)
                    heappush(open_set, (f_score[neighbor], neighbor))

        return []  # Pas de chemin trouve

    def export_to_json(self, output_path: str):
        """Exporte la base vers JSON"""
        export_data = {
            'version': '1.0',
            'total_maps': len(self.maps),
            'maps': []
        }

        for map_obj in self.maps.values():
            map_dict = asdict(map_obj)
            # Conversion enums et sets
            map_dict['map_type'] = map_obj.map_type.value

            # Conversion cell_grid
            cell_grid_str = {}
            for key, value in map_obj.cell_grid.items():
                cell_grid_str[str(key)] = value.value
            map_dict['cell_grid'] = cell_grid_str

            map_dict['walkable_cells'] = list(map_obj.walkable_cells)
            map_dict['safe_cells'] = list(map_obj.safe_cells)

            for res_spawn in map_dict['resource_spawns']:
                res_spawn['resource_type'] = res_spawn['resource_type'].value

            export_data['maps'].append(map_dict)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Base cartes exportee vers {output_path}")

# Instance globale
_maps_db_instance = None

def get_maps_database() -> DofusMapsDatabase:
    """Retourne l'instance singleton"""
    global _maps_db_instance
    if _maps_db_instance is None:
        _maps_db_instance = DofusMapsDatabase()
    return _maps_db_instance

# Test du module
if __name__ == "__main__":
    db = DofusMapsDatabase()

    # Test recherche
    incarnam = db.get_map_by_coordinates((7, -4))
    if incarnam:
        print(f"Carte trouvee: {incarnam.name} - Zone {incarnam.zone_name}")

        # Test ressources
        fer_maps = db.find_resource_maps("Fer", 1)
        print(f"Cartes avec Fer: {[m.name for m in fer_maps]}")

        # Test route farming
        route = db.get_optimal_farming_route("Frene", 10)
        print(f"Route farming Frene: {[m.name for m in route]}")

        # Test pathfinding
        path = db.find_safe_path_on_map(1, (0, 0), (14, 16))
        print(f"Chemin sur carte (longueur): {len(path)}")

    # Export
    db.export_to_json("test_maps_export.json")