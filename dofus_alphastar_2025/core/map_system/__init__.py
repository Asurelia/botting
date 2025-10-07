"""
Map System - Navigation globale et graph des maps Dofus
Gère les 600+ maps interconnectées avec pathfinding intelligent
"""

from .map_graph import (
    MapGraph,
    MapCoords,
    MapNode,
    MapEdge,
    MapExit,
    create_map_graph
)

from .map_discovery import (
    MapDiscovery,
    DiscoveredMap,
    create_map_discovery
)

__all__ = [
    "MapGraph",
    "MapCoords",
    "MapNode",
    "MapEdge",
    "MapExit",
    "create_map_graph",
    "MapDiscovery",
    "DiscoveredMap",
    "create_map_discovery"
]