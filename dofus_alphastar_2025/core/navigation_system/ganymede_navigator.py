#!/usr/bin/env python3
"""
GanymedeNavigator - Navigateur spécialisé pour Ganymède
Connaît parfaitement la topologie et les chemins optimaux de Ganymède
"""

import time
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from pathlib import Path

import numpy as np
import torch

from config import config
from core.hrm_reasoning import create_hrm_model, HRMOutput
from .world_map_analyzer import WorldMapAnalyzer, MapRegion, MapElement, MapElementType

logger = logging.getLogger(__name__)

class NavigationType(Enum):
    """Types de navigation"""
    WALKING = "walking"
    RUNNING = "running"
    ZAAP = "zaap"
    BOAT = "boat"
    DRAG = "drag"
    TELEPORT = "teleport"

class NavigationPriority(Enum):
    """Priorités de navigation"""
    SAFETY = "safety"
    SPEED = "speed"
    EFFICIENCY = "efficiency"
    STEALTH = "stealth"

@dataclass
class NavigationStep:
    """Étape de navigation"""
    step_id: str
    step_type: NavigationType
    from_position: Tuple[int, int]
    to_position: Tuple[int, int]
    map_id: Optional[str] = None

    # Instructions
    description: str = ""
    actions: List[str] = field(default_factory=list)

    # Métadonnées
    estimated_time: float = 0.0
    danger_level: int = 1  # 1=sûr, 5=dangereux
    cost: int = 0  # En kamas si applicable
    requirements: List[str] = field(default_factory=list)

@dataclass
class NavigationRoute:
    """Route de navigation complète"""
    route_id: str
    from_location: str
    to_location: str
    steps: List[NavigationStep] = field(default_factory=list)

    # Caractéristiques
    total_distance: float = 0.0
    total_time: float = 0.0
    total_cost: int = 0
    max_danger_level: int = 1

    # Métadonnées
    route_type: NavigationPriority = NavigationPriority.EFFICIENCY
    level_requirement: int = 1
    notes: str = ""

class GanymedeMapData:
    """Données de carte spécifiques à Ganymède"""

    def __init__(self):
        # Topologie de Ganymède (positions relatives)
        self.ganymede_maps = {
            "ganymede_center": {
                "name": "Centre de Ganymède",
                "coordinates": (0, 0),
                "level_range": [1, 10],
                "monsters": ["Bouftou", "Pissenlit"],
                "npcs": ["Gardien Ganymède", "Marchand Novice"],
                "zaap": True,
                "connections": {
                    "right": "ganymede_east",
                    "left": "ganymede_west",
                    "up": "ganymede_north",
                    "down": "ganymede_south"
                }
            },
            "ganymede_east": {
                "name": "Ganymède Est",
                "coordinates": (1, 0),
                "level_range": [1, 15],
                "monsters": ["Bouftou", "Larve Bleue"],
                "npcs": ["Fermier"],
                "zaap": False,
                "connections": {
                    "left": "ganymede_center",
                    "right": "ganymede_forest"
                }
            },
            "ganymede_west": {
                "name": "Ganymède Ouest",
                "coordinates": (-1, 0),
                "level_range": [1, 15],
                "monsters": ["Pissenlit", "Abeille"],
                "npcs": ["Collecteur de Miel"],
                "zaap": False,
                "connections": {
                    "right": "ganymede_center",
                    "left": "ganymede_lake"
                }
            },
            "ganymede_north": {
                "name": "Ganymède Nord",
                "coordinates": (0, -1),
                "level_range": [5, 20],
                "monsters": ["Bouftou Mutant", "Sanglier"],
                "npcs": ["Chasseur Expérimenté"],
                "zaap": False,
                "connections": {
                    "down": "ganymede_center",
                    "up": "ganymede_mountains"
                }
            },
            "ganymede_south": {
                "name": "Ganymède Sud",
                "coordinates": (0, 1),
                "level_range": [1, 12],
                "monsters": ["Bouftou Bébé", "Pissenlit Diabolique"],
                "npcs": ["Fermière Débutante"],
                "zaap": False,
                "connections": {
                    "up": "ganymede_center",
                    "down": "ganymede_plains"
                }
            },
            "ganymede_forest": {
                "name": "Forêt de Ganymède",
                "coordinates": (2, 0),
                "level_range": [10, 25],
                "monsters": ["Sanglier", "Moskito", "Arakne"],
                "npcs": ["Bûcheron"],
                "zaap": False,
                "connections": {
                    "left": "ganymede_east"
                }
            },
            "ganymede_lake": {
                "name": "Lac de Ganymède",
                "coordinates": (-2, 0),
                "level_range": [8, 20],
                "monsters": ["Grenouille", "Tofu Maléfique"],
                "npcs": ["Pêcheur"],
                "zaap": False,
                "connections": {
                    "right": "ganymede_west"
                }
            },
            "ganymede_mountains": {
                "name": "Montagnes de Ganymède",
                "coordinates": (0, -2),
                "level_range": [15, 30],
                "monsters": ["Sanglier des Neiges", "Loup"],
                "npcs": ["Mineur"],
                "zaap": False,
                "connections": {
                    "down": "ganymede_north"
                }
            },
            "ganymede_plains": {
                "name": "Plaines de Ganymède",
                "coordinates": (0, 2),
                "level_range": [5, 18],
                "monsters": ["Bouftou Royal", "Chef Pissenlit"],
                "npcs": ["Éleveur"],
                "zaap": False,
                "connections": {
                    "up": "ganymede_south"
                }
            }
        }

        # Routes prédéfinies populaires
        self.preset_routes = {
            "zaap_to_forest": {
                "description": "Du Zaap à la Forêt",
                "steps": ["ganymede_center", "ganymede_east", "ganymede_forest"],
                "estimated_time": 45.0,
                "level_requirement": 1
            },
            "farming_route": {
                "description": "Route de farm optimale",
                "steps": ["ganymede_center", "ganymede_east", "ganymede_forest", "ganymede_east", "ganymede_center", "ganymede_south"],
                "estimated_time": 120.0,
                "level_requirement": 10
            },
            "quest_tutorial": {
                "description": "Tour complet pour tutoriel",
                "steps": ["ganymede_center", "ganymede_east", "ganymede_center", "ganymede_west", "ganymede_center", "ganymede_north", "ganymede_center", "ganymede_south"],
                "estimated_time": 180.0,
                "level_requirement": 1
            }
        }

        # Points d'intérêt spéciaux
        self.points_of_interest = {
            "ganymede_zaap": {
                "map": "ganymede_center",
                "position": (400, 300),
                "type": "zaap",
                "description": "Zaap de Ganymède"
            },
            "tutorial_npc": {
                "map": "ganymede_center",
                "position": (350, 250),
                "type": "npc",
                "description": "PNJ du tutoriel"
            },
            "first_quest_target": {
                "map": "ganymede_east",
                "position": (200, 200),
                "type": "quest_location",
                "description": "Zone de première quête"
            }
        }

class GanymedePathfinder:
    """Pathfinder spécialisé pour Ganymède"""

    def __init__(self, map_data: GanymedeMapData):
        self.map_data = map_data

    def find_shortest_path(self, from_map: str, to_map: str) -> List[str]:
        """Trouve le chemin le plus court entre deux cartes"""

        if from_map == to_map:
            return [from_map]

        # BFS pour trouver chemin le plus court
        visited = set()
        queue = [(from_map, [from_map])]

        while queue:
            current_map, path = queue.pop(0)

            if current_map in visited:
                continue

            visited.add(current_map)

            if current_map == to_map:
                return path

            # Explorer connexions
            map_info = self.map_data.ganymede_maps.get(current_map, {})
            connections = map_info.get("connections", {})

            for direction, next_map in connections.items():
                if next_map not in visited:
                    queue.append((next_map, path + [next_map]))

        return []  # Pas de chemin trouvé

    def calculate_route_danger(self, path: List[str], player_level: int) -> float:
        """Calcule le niveau de danger d'une route"""

        total_danger = 0.0

        for map_id in path:
            map_info = self.map_data.ganymede_maps.get(map_id, {})
            level_range = map_info.get("level_range", [1, 1])
            max_monster_level = level_range[1]

            # Danger basé sur différence de niveau
            level_diff = max_monster_level - player_level
            if level_diff > 0:
                total_danger += level_diff * 0.5

        return min(5.0, total_danger / len(path))

    def estimate_travel_time(self, path: List[str], navigation_type: NavigationType = NavigationType.WALKING) -> float:
        """Estime le temps de voyage"""

        if len(path) <= 1:
            return 0.0

        # Temps de base par carte
        base_time_per_map = {
            NavigationType.WALKING: 30.0,
            NavigationType.RUNNING: 20.0,
            NavigationType.ZAAP: 5.0
        }

        base_time = base_time_per_map.get(navigation_type, 30.0)

        # Temps de transition entre cartes
        transition_time = (len(path) - 1) * 5.0

        # Temps total
        total_time = (len(path) - 1) * base_time + transition_time

        return total_time

class GanymedeNavigator:
    """Navigateur principal pour Ganymède"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Données et analyseurs
        self.map_data = GanymedeMapData()
        self.pathfinder = GanymedePathfinder(self.map_data)
        self.world_analyzer = WorldMapAnalyzer()

        # HRM pour décisions intelligentes
        self.hrm_model = create_hrm_model().to(self.device)

        # État de navigation
        self.current_map = "ganymede_center"
        self.current_position = (400, 300)
        self.navigation_history: List[str] = []

        # Cache de routes
        self.route_cache: Dict[str, NavigationRoute] = {}

        # Préférences
        self.default_priority = NavigationPriority.EFFICIENCY
        self.avoid_dangerous_areas = True
        self.prefer_zaap_routes = True

        logger.info("GanymedeNavigator initialisé avec données complètes")

    def navigate_to_location(self,
                           target_location: str,
                           player_level: int = 1,
                           priority: NavigationPriority = None,
                           context: Dict[str, Any] = None) -> NavigationRoute:
        """Navigation principale vers une localisation"""

        priority = priority or self.default_priority

        # Résoudre location (nom vers map_id)
        target_map = self._resolve_location(target_location)
        if not target_map:
            raise ValueError(f"Location inconnue: {target_location}")

        # Vérifier cache
        cache_key = f"{self.current_map}_{target_map}_{priority.value}_{player_level}"
        if cache_key in self.route_cache:
            cached_route = self.route_cache[cache_key]
            # Vérifier si encore valide (moins de 10 minutes)
            if time.time() - cached_route.total_time < 600:
                return cached_route

        # Calculer nouvelle route
        route = self._calculate_optimal_route(target_map, player_level, priority, context)

        # Mettre en cache
        self.route_cache[cache_key] = route

        return route

    def _resolve_location(self, location: str) -> Optional[str]:
        """Résout un nom de location vers un map_id"""

        location_lower = location.lower()

        # Recherche directe par map_id
        if location_lower in self.map_data.ganymede_maps:
            return location_lower

        # Recherche par nom de carte
        for map_id, map_info in self.map_data.ganymede_maps.items():
            if location_lower in map_info.get("name", "").lower():
                return map_id

        # Recherche dans points d'intérêt
        for poi_id, poi_info in self.map_data.points_of_interest.items():
            if location_lower in poi_info.get("description", "").lower():
                return poi_info.get("map")

        # Recherche par NPCs
        for map_id, map_info in self.map_data.ganymede_maps.items():
            npcs = map_info.get("npcs", [])
            if any(location_lower in npc.lower() for npc in npcs):
                return map_id

        return None

    def _calculate_optimal_route(self,
                               target_map: str,
                               player_level: int,
                               priority: NavigationPriority,
                               context: Dict[str, Any] = None) -> NavigationRoute:
        """Calcule la route optimale selon les critères"""

        # Chemin de base
        path = self.pathfinder.find_shortest_path(self.current_map, target_map)

        if not path:
            raise ValueError(f"Pas de chemin trouvé vers {target_map}")

        # Optimiser selon priorité
        if priority == NavigationPriority.SAFETY:
            path = self._optimize_for_safety(path, player_level)
        elif priority == NavigationPriority.SPEED:
            path = self._optimize_for_speed(path, player_level)
        elif priority == NavigationPriority.STEALTH:
            path = self._optimize_for_stealth(path, player_level)

        # Créer étapes de navigation
        steps = self._create_navigation_steps(path, player_level)

        # Calculer métadonnées
        total_time = sum(step.estimated_time for step in steps)
        total_cost = sum(step.cost for step in steps)
        max_danger = max((step.danger_level for step in steps), default=1)

        route = NavigationRoute(
            route_id=f"route_{int(time.time())}",
            from_location=self.current_map,
            to_location=target_map,
            steps=steps,
            total_time=total_time,
            total_cost=total_cost,
            max_danger_level=max_danger,
            route_type=priority,
            level_requirement=player_level
        )

        return route

    def _optimize_for_safety(self, path: List[str], player_level: int) -> List[str]:
        """Optimise route pour sécurité"""

        # Éviter cartes dangereuses si possible
        safe_path = []

        for map_id in path:
            map_info = self.map_data.ganymede_maps.get(map_id, {})
            level_range = map_info.get("level_range", [1, 1])

            # Si trop dangereux, chercher alternative
            if level_range[1] > player_level + 10:
                # Pour Ganymède, essayer route par centre
                if map_id != "ganymede_center" and "ganymede_center" not in safe_path:
                    safe_path.append("ganymede_center")

            safe_path.append(map_id)

        return safe_path

    def _optimize_for_speed(self, path: List[str], player_level: int) -> List[str]:
        """Optimise route pour vitesse"""

        # Préférer zaaps si disponibles
        if self.prefer_zaap_routes and "ganymede_center" not in path:
            # Retourner au zaap puis direct
            return ["ganymede_center"] + path

        return path

    def _optimize_for_stealth(self, path: List[str], player_level: int) -> List[str]:
        """Optimise route pour discrétion"""

        # Éviter zones très peuplées en monstres
        stealthy_path = []

        for map_id in path:
            map_info = self.map_data.ganymede_maps.get(map_id, {})
            monsters = map_info.get("monsters", [])

            # Si beaucoup de monstres, ajouter délai ou route alternative
            if len(monsters) > 2:
                # Marquer pour mouvement prudent
                pass

            stealthy_path.append(map_id)

        return stealthy_path

    def _create_navigation_steps(self, path: List[str], player_level: int) -> List[NavigationStep]:
        """Crée les étapes détaillées de navigation"""

        steps = []

        for i in range(len(path) - 1):
            from_map = path[i]
            to_map = path[i + 1]

            # Informations des cartes
            from_info = self.map_data.ganymede_maps.get(from_map, {})
            to_info = self.map_data.ganymede_maps.get(to_map, {})

            # Trouver direction
            direction = self._find_direction(from_map, to_map)

            # Position approximative
            from_pos = self._get_map_center_position(from_map)
            to_pos = self._get_map_center_position(to_map)

            # Type de navigation
            nav_type = NavigationType.WALKING
            if from_info.get("zaap") and i == 0:
                nav_type = NavigationType.ZAAP

            # Calculer danger
            danger = self.pathfinder.calculate_route_danger([to_map], player_level)

            # Temps estimé
            estimated_time = self.pathfinder.estimate_travel_time([from_map, to_map], nav_type)

            step = NavigationStep(
                step_id=f"step_{i}",
                step_type=nav_type,
                from_position=from_pos,
                to_position=to_pos,
                map_id=to_map,
                description=f"Aller de {from_info.get('name', from_map)} vers {to_info.get('name', to_map)} ({direction})",
                actions=[f"move_{direction}", "check_monsters", "continue"],
                estimated_time=estimated_time,
                danger_level=max(1, int(danger)),
                requirements=[]
            )

            steps.append(step)

        return steps

    def _find_direction(self, from_map: str, to_map: str) -> str:
        """Trouve la direction entre deux cartes"""

        from_info = self.map_data.ganymede_maps.get(from_map, {})
        connections = from_info.get("connections", {})

        for direction, connected_map in connections.items():
            if connected_map == to_map:
                return direction

        return "unknown"

    def _get_map_center_position(self, map_id: str) -> Tuple[int, int]:
        """Récupère position centre d'une carte"""

        # Positions fixes pour Ganymède (simulées)
        positions = {
            "ganymede_center": (400, 300),
            "ganymede_east": (600, 300),
            "ganymede_west": (200, 300),
            "ganymede_north": (400, 100),
            "ganymede_south": (400, 500),
            "ganymede_forest": (800, 300),
            "ganymede_lake": (50, 300),
            "ganymede_mountains": (400, 50),
            "ganymede_plains": (400, 600)
        }

        return positions.get(map_id, (400, 300))

    def get_preset_route(self, route_name: str, player_level: int = 1) -> Optional[NavigationRoute]:
        """Récupère une route prédéfinie"""

        preset = self.map_data.preset_routes.get(route_name)
        if not preset:
            return None

        if player_level < preset.get("level_requirement", 1):
            return None

        # Convertir en vraie route
        path = preset["steps"]
        steps = self._create_navigation_steps(path, player_level)

        return NavigationRoute(
            route_id=f"preset_{route_name}",
            from_location=path[0],
            to_location=path[-1],
            steps=steps,
            total_time=preset.get("estimated_time", 0),
            route_type=NavigationPriority.EFFICIENCY,
            level_requirement=preset.get("level_requirement", 1),
            notes=preset.get("description", "")
        )

    def update_position(self, new_map: str, new_position: Tuple[int, int] = None):
        """Met à jour position actuelle"""

        if new_map in self.map_data.ganymede_maps:
            self.current_map = new_map
            self.current_position = new_position or self._get_map_center_position(new_map)
            self.navigation_history.append(new_map)

            # Limiter historique
            if len(self.navigation_history) > 50:
                self.navigation_history = self.navigation_history[-50:]

            logger.debug(f"Position mise à jour: {new_map} {self.current_position}")

    def get_nearby_points_of_interest(self, radius: int = 2) -> List[Dict[str, Any]]:
        """Récupère points d'intérêt proches"""

        current_coords = self.map_data.ganymede_maps.get(self.current_map, {}).get("coordinates", (0, 0))
        nearby_pois = []

        for poi_id, poi_info in self.map_data.points_of_interest.items():
            poi_map = poi_info.get("map")
            if poi_map in self.map_data.ganymede_maps:
                poi_coords = self.map_data.ganymede_maps[poi_map].get("coordinates", (0, 0))

                # Distance Manhattan
                distance = abs(current_coords[0] - poi_coords[0]) + abs(current_coords[1] - poi_coords[1])

                if distance <= radius:
                    nearby_pois.append({
                        "poi_id": poi_id,
                        "distance": distance,
                        "map": poi_map,
                        "description": poi_info.get("description", ""),
                        "type": poi_info.get("type", "unknown")
                    })

        return sorted(nearby_pois, key=lambda x: x["distance"])

    def get_navigation_stats(self) -> Dict[str, Any]:
        """Statistiques de navigation"""

        return {
            "current_map": self.current_map,
            "current_position": self.current_position,
            "maps_visited": len(set(self.navigation_history)),
            "total_movements": len(self.navigation_history),
            "cached_routes": len(self.route_cache),
            "available_maps": len(self.map_data.ganymede_maps),
            "preset_routes": len(self.map_data.preset_routes),
            "points_of_interest": len(self.map_data.points_of_interest)
        }

def create_ganymede_navigator() -> GanymedeNavigator:
    """Factory function pour créer un GanymedeNavigator"""
    return GanymedeNavigator()