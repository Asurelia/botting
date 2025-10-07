#!/usr/bin/env python3
"""
MapGraph - Graph de navigation des maps Dofus
Utilise NetworkX pour pathfinding A* global
"""

import networkx as nx
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
import logging

@dataclass
class MapCoords:
    """Coordonnées d'une map"""
    x: int
    y: int

    def __str__(self):
        return f"({self.x},{self.y})"

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return isinstance(other, MapCoords) and self.x == other.x and self.y == other.y

    @classmethod
    def from_string(cls, coords_str: str) -> 'MapCoords':
        """Parse '(5,-18)' -> MapCoords(5, -18)"""
        coords_str = coords_str.strip('()')
        x, y = map(int, coords_str.split(','))
        return cls(x, y)

@dataclass
class MapExit:
    """Sortie d'une map"""
    direction: str  # north, south, east, west, door, zaap
    exit_type: str  # edge, teleport, transition
    destination: Optional[MapCoords] = None
    position: Optional[Tuple[int, int]] = None  # Position cliquable si door/zaap
    cell_id: Optional[int] = None  # ID de la cellule de sortie

@dataclass
class MapNode:
    """Nœud représentant une map"""
    coords: MapCoords
    name: Optional[str] = None
    area: Optional[str] = None  # Incarnam, Astrub, etc.
    subarea: Optional[str] = None
    exits: List[MapExit] = None
    has_zaap: bool = False
    has_bank: bool = False
    has_phoenix: bool = False
    is_dungeon: bool = False
    is_pvp: bool = False
    is_safe: bool = True
    discovered: bool = False
    screenshot_path: Optional[str] = None

    def __post_init__(self):
        if self.exits is None:
            self.exits = []

@dataclass
class MapEdge:
    """Arête entre deux maps"""
    from_map: MapCoords
    to_map: MapCoords
    exit_direction: str
    travel_time: float = 2.0  # Secondes
    requires_zaap: bool = False
    requires_key: bool = False

class MapGraph:
    """
    Graph de navigation global de Dofus

    Utilise NetworkX pour pathfinding A* entre maps
    """

    def __init__(self, database_path: str = "config/maps_database.json"):
        self.logger = logging.getLogger(__name__)
        self.database_path = Path(database_path)

        # Graph NetworkX
        self.graph = nx.Graph()

        # Données des maps
        self.maps: Dict[MapCoords, MapNode] = {}

        # Maps découvertes
        self.discovered_maps: Set[MapCoords] = set()

        # Load database si existe
        if self.database_path.exists():
            self.load_database()

    def add_map(self, map_node: MapNode):
        """Ajoute une map au graph"""
        coords = map_node.coords

        # Ajoute le nœud
        self.maps[coords] = map_node
        self.graph.add_node(coords, **asdict(map_node))

        if map_node.discovered:
            self.discovered_maps.add(coords)

        self.logger.debug(f"Map ajoutée: {coords} - {map_node.name or 'Sans nom'}")

    def add_connection(self, edge: MapEdge):
        """Ajoute une connexion entre deux maps"""
        # Ajoute l'arête avec poids = temps de trajet
        self.graph.add_edge(
            edge.from_map,
            edge.to_map,
            weight=edge.travel_time,
            direction=edge.exit_direction,
            requires_zaap=edge.requires_zaap,
            requires_key=edge.requires_key
        )

        self.logger.debug(f"Connexion ajoutée: {edge.from_map} -> {edge.to_map} ({edge.exit_direction})")

    def find_path(self, from_coords: MapCoords, to_coords: MapCoords,
                  avoid_pvp: bool = True, use_zaaps: bool = True) -> Optional[List[MapCoords]]:
        """
        Trouve le chemin optimal entre deux maps

        Args:
            from_coords: Map de départ
            to_coords: Map d'arrivée
            avoid_pvp: Éviter les zones PvP
            use_zaaps: Autoriser les zaaps pour raccourcir

        Returns:
            Liste de coordonnées de maps, ou None si pas de chemin
        """
        try:
            # Filtre les nœuds selon critères
            filtered_graph = self._filter_graph(avoid_pvp, use_zaaps)

            # A* avec NetworkX
            path = nx.shortest_path(
                filtered_graph,
                source=from_coords,
                target=to_coords,
                weight='weight'
            )

            self.logger.info(f"Chemin trouvé: {len(path)} maps")
            self.logger.debug(f"  {' → '.join(str(c) for c in path)}")

            return path

        except nx.NetworkXNoPath:
            self.logger.warning(f"Aucun chemin trouvé de {from_coords} vers {to_coords}")
            return None
        except nx.NodeNotFound:
            self.logger.error(f"Map non trouvée dans le graph: {from_coords} ou {to_coords}")
            return None

    def _filter_graph(self, avoid_pvp: bool, use_zaaps: bool) -> nx.Graph:
        """Filtre le graph selon les critères"""
        filtered = self.graph.copy()

        # Retire nœuds PvP si demandé
        if avoid_pvp:
            pvp_nodes = [
                coords for coords, data in filtered.nodes(data=True)
                if data.get('is_pvp', False)
            ]
            filtered.remove_nodes_from(pvp_nodes)
            self.logger.debug(f"Filtrage PvP: {len(pvp_nodes)} maps retirées")

        # Retire connexions zaap si non autorisées
        if not use_zaaps:
            zaap_edges = [
                (u, v) for u, v, data in filtered.edges(data=True)
                if data.get('requires_zaap', False)
            ]
            filtered.remove_edges_from(zaap_edges)
            self.logger.debug(f"Filtrage zaaps: {len(zaap_edges)} connexions retirées")

        return filtered

    def find_nearest_zaap(self, from_coords: MapCoords) -> Optional[Tuple[MapCoords, List[MapCoords]]]:
        """
        Trouve le zaap le plus proche

        Returns:
            (coords_zaap, chemin) ou None
        """
        zaap_maps = [
            coords for coords, data in self.graph.nodes(data=True)
            if data.get('has_zaap', False)
        ]

        if not zaap_maps:
            self.logger.warning("Aucun zaap trouvé dans le graph")
            return None

        # Trouve le plus proche
        shortest_path = None
        shortest_length = float('inf')
        nearest_zaap = None

        for zaap_coords in zaap_maps:
            try:
                path = nx.shortest_path(
                    self.graph,
                    source=from_coords,
                    target=zaap_coords,
                    weight='weight'
                )

                if len(path) < shortest_length:
                    shortest_length = len(path)
                    shortest_path = path
                    nearest_zaap = zaap_coords

            except nx.NetworkXNoPath:
                continue

        if nearest_zaap:
            self.logger.info(f"Zaap le plus proche: {nearest_zaap} ({shortest_length} maps)")
            return nearest_zaap, shortest_path

        return None

    def find_maps_in_area(self, area: str, subarea: Optional[str] = None) -> List[MapCoords]:
        """Trouve toutes les maps d'une zone"""
        results = []

        for coords, data in self.graph.nodes(data=True):
            if data.get('area') == area:
                if subarea is None or data.get('subarea') == subarea:
                    results.append(coords)

        self.logger.info(f"Trouvé {len(results)} maps dans {area}/{subarea or 'all'}")
        return results

    def get_map_info(self, coords: MapCoords) -> Optional[MapNode]:
        """Récupère les infos d'une map"""
        return self.maps.get(coords)

    def get_neighbors(self, coords: MapCoords) -> List[MapCoords]:
        """Récupère les maps voisines"""
        if coords not in self.graph:
            return []

        return list(self.graph.neighbors(coords))

    def mark_discovered(self, coords: MapCoords):
        """Marque une map comme découverte"""
        if coords in self.maps:
            self.maps[coords].discovered = True
            self.discovered_maps.add(coords)
            self.graph.nodes[coords]['discovered'] = True

            self.logger.debug(f"Map {coords} marquée comme découverte")

    def get_discovery_progress(self) -> Tuple[int, int]:
        """
        Retourne (maps_découvertes, total_maps)
        """
        return len(self.discovered_maps), len(self.maps)

    def save_database(self):
        """Sauvegarde le graph en JSON"""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'version': '1.0',
            'maps': {
                str(coords): asdict(map_node)
                for coords, map_node in self.maps.items()
            },
            'edges': [
                {
                    'from': str(u),
                    'to': str(v),
                    'data': data
                }
                for u, v, data in self.graph.edges(data=True)
            ],
            'discovered': [str(c) for c in self.discovered_maps],
            'statistics': {
                'total_maps': len(self.maps),
                'discovered_maps': len(self.discovered_maps),
                'total_connections': self.graph.number_of_edges()
            }
        }

        with open(self.database_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Base de données sauvegardée: {self.database_path}")

    def load_database(self):
        """Charge le graph depuis JSON"""
        try:
            with open(self.database_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Charge les maps
            for coords_str, map_data in data.get('maps', {}).items():
                coords = MapCoords.from_string(coords_str)

                # Reconstruit MapNode
                map_data['coords'] = coords
                if 'exits' in map_data:
                    map_data['exits'] = [MapExit(**exit_data) for exit_data in map_data['exits']]

                map_node = MapNode(**map_data)
                self.add_map(map_node)

            # Charge les connexions
            for edge_data in data.get('edges', []):
                from_coords = MapCoords.from_string(edge_data['from'])
                to_coords = MapCoords.from_string(edge_data['to'])

                edge = MapEdge(
                    from_map=from_coords,
                    to_map=to_coords,
                    exit_direction=edge_data['data'].get('direction', 'unknown'),
                    travel_time=edge_data['data'].get('weight', 2.0),
                    requires_zaap=edge_data['data'].get('requires_zaap', False),
                    requires_key=edge_data['data'].get('requires_key', False)
                )
                self.add_connection(edge)

            # Charge maps découvertes
            for coords_str in data.get('discovered', []):
                coords = MapCoords.from_string(coords_str)
                self.discovered_maps.add(coords)

            self.logger.info(f"Base de données chargée: {len(self.maps)} maps, {self.graph.number_of_edges()} connexions")

        except Exception as e:
            self.logger.error(f"Erreur chargement base de données: {e}")

    def export_to_graphml(self, output_path: str):
        """Exporte le graph en GraphML (visualisation)"""
        nx.write_graphml(self.graph, output_path)
        self.logger.info(f"Graph exporté en GraphML: {output_path}")

def create_map_graph(database_path: str = "config/maps_database.json") -> MapGraph:
    """Factory function pour créer un MapGraph"""
    return MapGraph(database_path)