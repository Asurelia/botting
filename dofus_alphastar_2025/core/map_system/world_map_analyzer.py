"""
World Map Analyzer - Analyse avancée des cartes du monde DOFUS
Détection de régions, éléments, et topologie
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class MapElementType(Enum):
    """Types d'éléments sur la carte"""
    ZAAP = "zaap"
    NPC = "npc"
    MONSTER = "monster"
    RESOURCE = "resource"
    QUEST_MARKER = "quest_marker"
    CHEST = "chest"
    DOOR = "door"
    TELEPORTER = "teleporter"
    BANK = "bank"
    MARKET = "market"
    DUNGEON_ENTRANCE = "dungeon_entrance"
    SAFE_ZONE = "safe_zone"
    DANGER_ZONE = "danger_zone"


class MapRegionType(Enum):
    """Types de régions"""
    CITY = "city"
    VILLAGE = "village"
    FOREST = "forest"
    PLAINS = "plains"
    MOUNTAINS = "mountains"
    DESERT = "desert"
    SWAMP = "swamp"
    BEACH = "beach"
    DUNGEON = "dungeon"
    UNKNOWN = "unknown"


@dataclass
class MapElement:
    """Élément sur une carte"""
    element_id: str
    element_type: MapElementType
    position: Tuple[int, int]
    name: str = ""
    level_range: Tuple[int, int] = (1, 1)
    properties: Dict[str, Any] = field(default_factory=dict)

    def is_interactive(self) -> bool:
        """Vérifie si l'élément est interactif"""
        return self.element_type in [
            MapElementType.NPC,
            MapElementType.ZAAP,
            MapElementType.CHEST,
            MapElementType.DOOR,
            MapElementType.TELEPORTER,
            MapElementType.BANK,
            MapElementType.MARKET
        ]

    def is_dangerous(self) -> bool:
        """Vérifie si l'élément est dangereux"""
        return self.element_type in [
            MapElementType.MONSTER,
            MapElementType.DANGER_ZONE,
            MapElementType.DUNGEON_ENTRANCE
        ]


@dataclass
class MapRegion:
    """Région d'une carte du monde"""
    region_id: str
    region_type: MapRegionType
    bounds: Tuple[int, int, int, int]  # x_min, y_min, x_max, y_max
    center: Tuple[int, int]

    # Propriétés
    name: str = ""
    level_range: Tuple[int, int] = (1, 1)
    danger_level: int = 1  # 1-5

    # Éléments
    elements: List[MapElement] = field(default_factory=list)
    connections: List[str] = field(default_factory=list)  # IDs des régions connectées

    # Métadonnées
    properties: Dict[str, Any] = field(default_factory=dict)

    def get_elements_by_type(self, element_type: MapElementType) -> List[MapElement]:
        """Récupère les éléments d'un type spécifique"""
        return [elem for elem in self.elements if elem.element_type == element_type]

    def has_zaap(self) -> bool:
        """Vérifie si la région a un Zaap"""
        return any(elem.element_type == MapElementType.ZAAP for elem in self.elements)

    def has_bank(self) -> bool:
        """Vérifie si la région a une banque"""
        return any(elem.element_type == MapElementType.BANK for elem in self.elements)

    def is_safe(self) -> bool:
        """Vérifie si la région est sûre"""
        return self.danger_level <= 2

    def get_area(self) -> int:
        """Calcule l'aire de la région"""
        x_min, y_min, x_max, y_max = self.bounds
        return (x_max - x_min) * (y_max - y_min)


@dataclass
class WorldMap:
    """Carte du monde complète"""
    world_id: str
    name: str
    regions: Dict[str, MapRegion] = field(default_factory=dict)
    connections: Dict[str, List[str]] = field(default_factory=dict)

    def get_region(self, region_id: str) -> Optional[MapRegion]:
        """Récupère une région par son ID"""
        return self.regions.get(region_id)

    def get_regions_by_type(self, region_type: MapRegionType) -> List[MapRegion]:
        """Récupère toutes les régions d'un type"""
        return [region for region in self.regions.values() if region.region_type == region_type]

    def get_safe_regions(self) -> List[MapRegion]:
        """Récupère toutes les régions sûres"""
        return [region for region in self.regions.values() if region.is_safe()]

    def get_regions_with_zaap(self) -> List[MapRegion]:
        """Récupère toutes les régions avec un Zaap"""
        return [region for region in self.regions.values() if region.has_zaap()]

    def find_nearest_region(self, position: Tuple[int, int]) -> Optional[MapRegion]:
        """Trouve la région la plus proche d'une position"""
        if not self.regions:
            return None

        min_distance = float('inf')
        nearest_region = None

        for region in self.regions.values():
            distance = self._distance(position, region.center)
            if distance < min_distance:
                min_distance = distance
                nearest_region = region

        return nearest_region

    def _distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calcule la distance entre deux positions"""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


class WorldMapAnalyzer:
    """Analyseur de cartes du monde"""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = Path(data_path) if data_path else Path("data/maps")
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Cartes chargées
        self.world_maps: Dict[str, WorldMap] = {}

        # Cache d'analyse
        self.analysis_cache: Dict[str, Any] = {}

        # Statistiques
        self.stats = {
            "maps_loaded": 0,
            "regions_analyzed": 0,
            "elements_detected": 0
        }

        logger.info("WorldMapAnalyzer initialisé")

    def load_world_map(self, world_id: str) -> Optional[WorldMap]:
        """Charge une carte du monde depuis un fichier"""
        map_file = self.data_path / f"{world_id}.json"

        if not map_file.exists():
            logger.warning(f"Fichier de carte non trouvé: {map_file}")
            return None

        try:
            with open(map_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            world_map = self._parse_world_map(data)
            self.world_maps[world_id] = world_map
            self.stats["maps_loaded"] += 1

            logger.info(f"Carte chargée: {world_id} ({len(world_map.regions)} régions)")
            return world_map

        except Exception as e:
            logger.error(f"Erreur lors du chargement de la carte {world_id}: {e}")
            return None

    def _parse_world_map(self, data: Dict[str, Any]) -> WorldMap:
        """Parse les données JSON en WorldMap"""
        world_map = WorldMap(
            world_id=data.get("world_id", "unknown"),
            name=data.get("name", "Unknown World")
        )

        # Parser les régions
        for region_data in data.get("regions", []):
            region = self._parse_region(region_data)
            world_map.regions[region.region_id] = region
            self.stats["regions_analyzed"] += 1

        # Parser les connexions
        world_map.connections = data.get("connections", {})

        return world_map

    def _parse_region(self, data: Dict[str, Any]) -> MapRegion:
        """Parse une région depuis les données"""
        region = MapRegion(
            region_id=data.get("region_id", "unknown"),
            region_type=MapRegionType(data.get("region_type", "unknown")),
            bounds=tuple(data.get("bounds", [0, 0, 100, 100])),
            center=tuple(data.get("center", [50, 50])),
            name=data.get("name", ""),
            level_range=tuple(data.get("level_range", [1, 1])),
            danger_level=data.get("danger_level", 1),
            connections=data.get("connections", []),
            properties=data.get("properties", {})
        )

        # Parser les éléments
        for elem_data in data.get("elements", []):
            element = self._parse_element(elem_data)
            region.elements.append(element)
            self.stats["elements_detected"] += 1

        return region

    def _parse_element(self, data: Dict[str, Any]) -> MapElement:
        """Parse un élément depuis les données"""
        return MapElement(
            element_id=data.get("element_id", "unknown"),
            element_type=MapElementType(data.get("element_type", "unknown")),
            position=tuple(data.get("position", [0, 0])),
            name=data.get("name", ""),
            level_range=tuple(data.get("level_range", [1, 1])),
            properties=data.get("properties", {})
        )

    def analyze_region(self, region: MapRegion) -> Dict[str, Any]:
        """Analyse une région et retourne des métadonnées"""
        cache_key = f"region_{region.region_id}"

        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]

        analysis = {
            "region_id": region.region_id,
            "region_type": region.region_type.value,
            "area": region.get_area(),
            "has_zaap": region.has_zaap(),
            "has_bank": region.has_bank(),
            "is_safe": region.is_safe(),
            "danger_level": region.danger_level,
            "element_counts": self._count_elements_by_type(region),
            "connectivity": len(region.connections),
            "level_range": region.level_range,
            "recommended_for_level": self._get_recommended_level(region)
        }

        self.analysis_cache[cache_key] = analysis
        return analysis

    def _count_elements_by_type(self, region: MapRegion) -> Dict[str, int]:
        """Compte les éléments par type dans une région"""
        counts = {}
        for element in region.elements:
            elem_type = element.element_type.value
            counts[elem_type] = counts.get(elem_type, 0) + 1
        return counts

    def _get_recommended_level(self, region: MapRegion) -> int:
        """Calcule le niveau recommandé pour une région"""
        min_level, max_level = region.level_range
        # Recommander le niveau moyen
        return (min_level + max_level) // 2

    def find_farming_spots(
        self,
        world_map: WorldMap,
        player_level: int,
        max_danger: int = 3
    ) -> List[MapRegion]:
        """Trouve les spots de farming appropriés pour le niveau du joueur"""
        suitable_regions = []

        for region in world_map.regions.values():
            min_level, max_level = region.level_range

            # Vérifier si le niveau convient
            if min_level <= player_level <= max_level + 10:
                # Vérifier le danger
                if region.danger_level <= max_danger:
                    # Vérifier qu'il y a des monstres
                    monsters = region.get_elements_by_type(MapElementType.MONSTER)
                    if monsters:
                        suitable_regions.append(region)

        # Trier par pertinence (niveau proche, danger faible)
        suitable_regions.sort(
            key=lambda r: (
                abs(player_level - (r.level_range[0] + r.level_range[1]) / 2),
                r.danger_level
            )
        )

        return suitable_regions

    def find_nearest_zaap(
        self,
        world_map: WorldMap,
        position: Tuple[int, int]
    ) -> Optional[MapElement]:
        """Trouve le Zaap le plus proche d'une position"""
        zaap_regions = world_map.get_regions_with_zaap()

        if not zaap_regions:
            return None

        min_distance = float('inf')
        nearest_zaap = None

        for region in zaap_regions:
            zaaps = region.get_elements_by_type(MapElementType.ZAAP)
            for zaap in zaaps:
                distance = self._distance(position, zaap.position)
                if distance < min_distance:
                    min_distance = distance
                    nearest_zaap = zaap

        return nearest_zaap

    def _distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calcule la distance euclidienne"""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    def create_sample_world_map(self) -> WorldMap:
        """Crée une carte du monde exemple (Ganymède)"""
        world_map = WorldMap(
            world_id="ganymede",
            name="Ganymède"
        )

        # Région Centre
        center_region = MapRegion(
            region_id="ganymede_center",
            region_type=MapRegionType.VILLAGE,
            bounds=(0, 0, 800, 600),
            center=(400, 300),
            name="Centre de Ganymède",
            level_range=(1, 10),
            danger_level=1,
            connections=["ganymede_east", "ganymede_west", "ganymede_north", "ganymede_south"]
        )

        # Ajouter éléments
        center_region.elements.extend([
            MapElement(
                element_id="zaap_center",
                element_type=MapElementType.ZAAP,
                position=(400, 300),
                name="Zaap de Ganymède"
            ),
            MapElement(
                element_id="bank_center",
                element_type=MapElementType.BANK,
                position=(350, 280),
                name="Banque de Ganymède"
            ),
            MapElement(
                element_id="npc_tutorial",
                element_type=MapElementType.NPC,
                position=(420, 320),
                name="Guide Tutoriel",
                properties={"quest_giver": True}
            )
        ])

        world_map.regions["ganymede_center"] = center_region

        # Région Est (Forêt)
        east_region = MapRegion(
            region_id="ganymede_east",
            region_type=MapRegionType.FOREST,
            bounds=(800, 0, 1600, 600),
            center=(1200, 300),
            name="Forêt de Ganymède",
            level_range=(10, 25),
            danger_level=3,
            connections=["ganymede_center"]
        )

        # Ajouter monstres
        east_region.elements.extend([
            MapElement(
                element_id="monster_sanglier_1",
                element_type=MapElementType.MONSTER,
                position=(900, 200),
                name="Sanglier",
                level_range=(12, 15)
            ),
            MapElement(
                element_id="monster_moskito_1",
                element_type=MapElementType.MONSTER,
                position=(1100, 350),
                name="Moskito",
                level_range=(15, 18)
            ),
            MapElement(
                element_id="resource_tree_1",
                element_type=MapElementType.RESOURCE,
                position=(1000, 400),
                name="Frêne",
                properties={"resource_type": "wood", "level": 1}
            )
        ])

        world_map.regions["ganymede_east"] = east_region

        # Région Ouest (Plaines)
        west_region = MapRegion(
            region_id="ganymede_west",
            region_type=MapRegionType.PLAINS,
            bounds=(-800, 0, 0, 600),
            center=(-400, 300),
            name="Plaines de Ganymède",
            level_range=(5, 18),
            danger_level=2,
            connections=["ganymede_center"]
        )

        west_region.elements.extend([
            MapElement(
                element_id="monster_bouftou_1",
                element_type=MapElementType.MONSTER,
                position=(-600, 250),
                name="Bouftou",
                level_range=(8, 12)
            ),
            MapElement(
                element_id="monster_pissenlit_1",
                element_type=MapElementType.MONSTER,
                position=(-500, 350),
                name="Pissenlit Diabolique",
                level_range=(10, 14)
            ),
            MapElement(
                element_id="resource_wheat_1",
                element_type=MapElementType.RESOURCE,
                position=(-450, 300),
                name="Blé",
                properties={"resource_type": "cereal", "level": 1}
            )
        ])

        world_map.regions["ganymede_west"] = west_region

        # Connexions
        world_map.connections = {
            "ganymede_center": ["ganymede_east", "ganymede_west", "ganymede_north", "ganymede_south"],
            "ganymede_east": ["ganymede_center"],
            "ganymede_west": ["ganymede_center"]
        }

        return world_map

    def save_world_map(self, world_map: WorldMap):
        """Sauvegarde une carte du monde en JSON"""
        map_file = self.data_path / f"{world_map.world_id}.json"

        # Convertir en dict
        data = {
            "world_id": world_map.world_id,
            "name": world_map.name,
            "regions": [
                self._region_to_dict(region)
                for region in world_map.regions.values()
            ],
            "connections": world_map.connections
        }

        try:
            with open(map_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Carte sauvegardée: {map_file}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")

    def _region_to_dict(self, region: MapRegion) -> Dict[str, Any]:
        """Convertit une région en dictionnaire"""
        return {
            "region_id": region.region_id,
            "region_type": region.region_type.value,
            "bounds": list(region.bounds),
            "center": list(region.center),
            "name": region.name,
            "level_range": list(region.level_range),
            "danger_level": region.danger_level,
            "connections": region.connections,
            "properties": region.properties,
            "elements": [
                self._element_to_dict(elem)
                for elem in region.elements
            ]
        }

    def _element_to_dict(self, element: MapElement) -> Dict[str, Any]:
        """Convertit un élément en dictionnaire"""
        return {
            "element_id": element.element_id,
            "element_type": element.element_type.value,
            "position": list(element.position),
            "name": element.name,
            "level_range": list(element.level_range),
            "properties": element.properties
        }

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'analyseur"""
        return {
            **self.stats,
            "world_maps_loaded": len(self.world_maps),
            "cache_size": len(self.analysis_cache)
        }


def create_world_map_analyzer(data_path: Optional[str] = None) -> WorldMapAnalyzer:
    """Factory function pour créer un WorldMapAnalyzer"""
    analyzer = WorldMapAnalyzer(data_path)

    # Créer et sauvegarder une carte exemple
    sample_map = analyzer.create_sample_world_map()
    analyzer.world_maps["ganymede"] = sample_map
    analyzer.save_world_map(sample_map)

    return analyzer
